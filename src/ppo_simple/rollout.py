"""双时间尺度 on-policy rollout、分离 buffer 与 GAE。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import time
from typing import Any, Mapping

import numpy as np

from decision import cache_configuration, slot_environment, uav_configuration

from . import interface
from .interface import _copy_uav_positions
from .policy import DualTimescaleActorCritic
from .primitive_control import (
    PrimitiveTensorSpec,
    _normalized_xy,
    _target_cache_proposals_with_features,
    _uav_target_position_feature,
)

EXECUTION_CLASSES = ("LOCAL", "SV", "RSU", "UAV", "CLOUD")


@dataclass(frozen=True)
class SlotTaskMetrics:
    """一个 slot 已完成系统执行的任务指标。"""

    task_count: int
    total_task_delay_s: float
    total_vehicle_energy_j: float
    execution_target_counts: dict[str, int]


def _require_mapping(value: Any, field_name: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _finite_nonnegative(value: Any, field_name: str, key: Any) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{field_name}[{key!r}] must be finite and non-negative")
    return numeric


def _execution_class(target: Any, cache_nodes: Mapping[str, Any]) -> str:
    target_id = str(target)
    if target_id == "LOCAL" or target_id.startswith("LOCAL:"):
        return "LOCAL"
    if target_id == "CLOUD":
        return "CLOUD"
    node_ids = [str(value) for value in cache_nodes["node_ids"]]
    if target_id not in node_ids:
        raise ValueError(f"unknown matching execution target: {target_id}")
    role = str(cache_nodes["node_roles"][node_ids.index(target_id)]).upper()
    if role not in {"SV", "RSU", "UAV"}:
        raise ValueError(f"unknown cache node execution role: {role}")
    return role


def task_metrics_from_result(result: Any, cache_nodes: Mapping[str, Any]) -> SlotTaskMetrics:
    """从匹配结果读取 slot 指标，不重复计算物理量。"""

    delay = _require_mapping(result.per_task_delay_s, "per_task_delay_s")
    energy = _require_mapping(result.per_task_vehicle_energy_j, "per_task_vehicle_energy_j")
    matching = _require_mapping(result.matching, "matching")
    keys = set(delay)
    if set(energy) != keys or set(matching) != keys or set(result.per_task_utility) != keys:
        raise ValueError("Matching 各任务结果的 key 集合必须一致。")
    counts = {name: 0 for name in EXECUTION_CLASSES}
    total_delay = 0.0
    total_energy = 0.0
    for task_id in keys:
        total_delay += _finite_nonnegative(delay[task_id], "delay", task_id)
        total_energy += _finite_nonnegative(energy[task_id], "energy", task_id)
        counts[_execution_class(matching[task_id], cache_nodes)] += 1
    return SlotTaskMetrics(
        task_count=len(keys),
        total_task_delay_s=float(total_delay),
        total_vehicle_energy_j=float(total_energy),
        execution_target_counts=counts,
    )


@dataclass(frozen=True)
class MicroTransition:
    """单个 slot 的 Micro transition，保存行为时的合法动作支持。"""

    slot: int
    origin_index: int
    observation: np.ndarray
    actions: np.ndarray
    old_joint_log_prob: float
    old_uav_log_probs: np.ndarray
    old_uav_entropies: np.ndarray
    old_joint_entropy: float
    old_value: float
    reward: float
    next_observation: np.ndarray
    done: bool
    uav_candidate_features: np.ndarray
    uav_candidate_masks: np.ndarray
    uav_candidate_positions: np.ndarray
    uav_candidate_probabilities: np.ndarray
    uav_candidate_sources: tuple[tuple[str, ...], tuple[str, ...]]
    uav0_position_condition: np.ndarray
    active_cache_configuration_id: str
    actual_task_utility: float
    actual_cache_update_count: int
    uav_movement_count: int
    cache_cost: float
    movement_cost: float
    task_count: int
    total_task_delay_s: float
    total_vehicle_energy_j: float
    execution_target_counts: dict[str, int]
    valid_candidate_count: np.ndarray
    policy_selection_seconds: float
    configuration_seconds: float
    matching_kkt_seconds: float
    slot_runtime_seconds: float
    matching_summary: dict[str, Any]

@dataclass(frozen=True)
class MacroTransition:
    """一个完整 Cache frame 的 Macro SMDP transition。"""

    observation: np.ndarray
    cache_candidate_features: np.ndarray
    cache_candidate_masks: np.ndarray
    duration_masks: np.ndarray
    cache_action: int
    duration_action: int
    requested_duration: int
    actual_duration: int
    old_macro_log_prob: float
    old_macro_entropy: float
    old_value: float
    frame_reward: float
    next_observation: np.ndarray
    done: bool
    cache_configuration_id: str
    actual_cache_update: bool

@dataclass
class MicroRolloutBuffer:
    """每 slot 一条的 Micro 数据；与 Macro buffer 完全分离。"""

    transitions: list[MicroTransition] = field(default_factory=list)

    def append(self, transition: MicroTransition) -> None:
        if self.transitions and bool(self.transitions[-1].done):
            raise ValueError("不能在 terminal Micro transition 后追加数据。")
        self.transitions.append(transition)

    def __len__(self) -> int:
        return len(self.transitions)


@dataclass
class MacroRolloutBuffer:
    """每 Cache frame 一条的 Macro 数据；与 Micro buffer 完全分离。"""

    transitions: list[MacroTransition] = field(default_factory=list)

    def append(self, transition: MacroTransition) -> None:
        if self.transitions and bool(self.transitions[-1].done):
            raise ValueError("不能在 terminal Macro transition 后追加数据。")
        if int(transition.actual_duration) <= 0:
            raise ValueError("Macro actual_duration 必须为正数。")
        self.transitions.append(transition)

    def __len__(self) -> int:
        return len(self.transitions)


@dataclass(frozen=True)
class EpisodeCollection:
    """完整 episode 的分离 rollout 与轻量运行计数。"""

    micro: MicroRolloutBuffer
    macro: MacroRolloutBuffer
    summary: dict[str, Any]


def actual_duration_for_frame(
    requested_duration: int,
    start_slot: int,
    episode_slots: int,
) -> int:
    """按 episode 尾部裁切请求时长，所有 Macro target 共用该结果。"""

    requested = int(requested_duration)
    start = int(start_slot)
    total = int(episode_slots)
    if requested <= 0 or total <= 0 or start < 0 or start >= total:
        raise ValueError("Macro frame bounds 或 requested duration 非法。")
    return min(requested, total - start)


def cache_update_mask(old_cache: Any, new_cache: Any) -> np.ndarray:
    """返回实际改变的 Cache 节点，未改变节点不会产生更新成本。"""

    old = np.asarray(old_cache, dtype=np.int8)
    new = np.asarray(new_cache, dtype=np.int8)
    if old.shape != new.shape or old.ndim != 2:
        raise ValueError("Cache configuration shape 必须一致且为二维。")
    return np.any(old != new, axis=1)


def _check_reward(result: slot_environment.SlotExecutionResult) -> float:
    """保留独立 Exact-Q diagnostic 所需的 slot utility 合同检查。"""

    if result.system_utility is None:
        raise AssertionError("feasible slot result 缺少 system_utility。")
    expected = (
        float(result.actual_task_utility)
        - float(result.cache_cost)
        - float(result.movement_cost)
    )
    if not np.isclose(float(result.system_utility), expected, rtol=1e-12, atol=1e-12):
        raise AssertionError("slot reward 与 system utility 不一致。")
    return float(result.system_utility)


def _check_request_contract(
    previous_cache: np.ndarray,
    previous_uav: dict[str, Any],
    result: slot_environment.SlotExecutionResult,
    action_spec: Any,
) -> None:
    """保留历史 Exact-Q diagnostic 对 trigger=0 的状态不变检查。"""

    if result.next_cache_state is None or result.next_uav_positions is None:
        raise AssertionError("可行 environment transition 缺少 next state。")
    next_cache = np.asarray(result.next_cache_state, dtype=np.int8)
    for node_index, request in enumerate(result.input_cache_triggers):
        if int(request) == 0 and not np.array_equal(
            previous_cache[node_index], next_cache[node_index]
        ):
            raise AssertionError("request=0 但 Cache actual state 发生变化。")
    for uav_id in action_spec.uav_ids:
        if int(result.input_uav_triggers[uav_id]) != 0:
            continue
        before = np.asarray(previous_uav[uav_id], dtype=np.float64)
        after = np.asarray(result.next_uav_positions[uav_id], dtype=np.float64)
        if not np.allclose(before, after, rtol=0.0, atol=1.0e-6):
            raise AssertionError("request=0 但 UAV actual state 发生变化。")


def _observation(
    cache: np.ndarray,
    uav: Mapping[str, Any],
    step: Any,
    context: Any,
) -> np.ndarray:
    to_go = getattr(context, "observation_time_to_go", None)
    override = None if to_go is None else float(to_go(int(step.origin_index)))
    result = interface.build_step_observation(
        cache,
        dict(uav),
        step,
        context.observation_spec,
        context.flight_boundary,
        context.train_origin_start,
        context.train_origin_end,
        time_to_go_override=override,
    )
    values = np.asarray(result, dtype=np.float32).reshape(-1)
    if values.shape != (int(context.observation_spec.observation_dim),):
        raise ValueError("当前正式 observation shape 与 schema 不一致。")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("observation 包含 NaN/Inf。")
    return values


def _cache_candidates(
    cache: np.ndarray,
    uav: Mapping[str, Any],
    step: Any,
    context: Any,
    tensor_spec: PrimitiveTensorSpec,
) -> tuple[list[np.ndarray], np.ndarray]:
    """candidate 0 是当前 Cache，其余候选复用现有节点 Cache proposal。"""

    current = np.asarray(cache, dtype=np.int8).copy()
    proposal_nodes = uav_configuration.cache_nodes_with_uav_positions(
        context.cache_nodes,
        {key: np.asarray(value, dtype=np.float64) for key, value in uav.items()},
    )
    candidates = [current.copy()]
    seen = {current.tobytes()}
    for node_index, _node_id in enumerate(context.action_spec.cache_node_ids):
        triggers = np.zeros(current.shape[0], dtype=np.int8)
        triggers[node_index] = 1
        node_candidates, _node_features = _target_cache_proposals_with_features(
            current,
            current,
            triggers,
            node_index,
            np.zeros(int(context.observation_spec.observation_dim), dtype=np.float32),
            step,
            context,
            proposal_nodes,
        )
        for item in node_candidates:
            target = current.copy()
            target[node_index] = np.asarray(item.target_node_cache, dtype=np.int8)
            key = target.tobytes()
            if key not in seen:
                seen.add(key)
                candidates.append(target)
    if not candidates or not np.array_equal(candidates[0], current):
        raise AssertionError("Macro cache candidate 0 必须等于当前实际 Cache。")
    flat_current = current.reshape(-1).astype(np.float32)
    features = np.stack(
        [
            np.concatenate(
                (
                    candidate.reshape(-1).astype(np.float32),
                    candidate.reshape(-1).astype(np.float32) - flat_current,
                    np.asarray(
                        [
                            float(np.count_nonzero(np.any(candidate != current, axis=1)))
                            / max(float(current.shape[0]), 1.0)
                        ],
                        dtype=np.float32,
                    ),
                )
            )
            for candidate in candidates
        ]
    ).astype(np.float32)
    return candidates, features


def _uav_candidates(
    uav_index: int,
    current_uav: dict[str, np.ndarray],
    previous_uav: dict[str, np.ndarray],
    cache: np.ndarray,
    step: Any,
    context: Any,
) -> tuple[list[uav_configuration.UavTargetCandidate], np.ndarray]:
    uav_ids = tuple(context.action_spec.uav_ids)
    uav_id = str(uav_ids[int(uav_index)])
    triggers = {key: 1 for key in uav_ids}
    moves = uav_configuration.enumerate_uav_target_candidates(
        current_uav,
        previous_uav,
        uav_id,
        triggers,
        context.cfg,
        context.flight_boundary,
        scenario_slice=step.scenario_slice,
        cache_state=cache,
        cache_nodes=context.cache_nodes,
        predicted_vehicle_positions=step.predicted_vehicle_positions,
    )
    stay_position = np.asarray(current_uav[uav_id], dtype=np.float64).copy()
    stay = uav_configuration.UavTargetCandidate(
        uav_id=uav_id,
        before_position=stay_position.copy(),
        target_position=stay_position.copy(),
        proposal_source="stay",
    )
    candidates = [stay, *moves]
    service_demand = cache_configuration.predicted_vehicle_service_demand(
        step.scenario_slice,
        int(context.service_catalog["num_services"]),
        beta_discount=context.cache_cfg.beta_discount,
    )
    features: list[np.ndarray] = []
    for index, candidate in enumerate(candidates):
        base = _uav_target_position_feature(
            candidate,
            current_uav,
            previous_uav,
            triggers,
            step.predicted_vehicle_positions,
            service_demand,
            cache,
            context,
        )
        group = np.asarray([1.0, 0.0] if index == 0 else [0.0, 1.0], dtype=np.float32)
        features.append(np.concatenate((base, group)))
    return candidates, np.stack(features).astype(np.float32)


def _positions_condition(
    position: Any,
    context: Any,
) -> np.ndarray:
    return _normalized_xy(position, context.flight_boundary).astype(np.float32)


def _pad_micro_candidates(
    features: list[np.ndarray],
    positions: list[np.ndarray],
    max_candidates: int,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    padded_features = np.zeros((2, max_candidates, feature_dim), dtype=np.float32)
    masks = np.zeros((2, max_candidates), dtype=bool)
    padded_positions = np.zeros((2, max_candidates, 2), dtype=np.float32)
    for head in range(2):
        count = int(len(features[head]))
        if count <= 0 or count > max_candidates:
            raise ValueError("UAV candidate 数量超出配置容量。")
        padded_features[head, :count] = features[head]
        masks[head, :count] = True
        padded_positions[head, :count] = positions[head]
    if not bool(np.all(masks[:, 0])):
        raise AssertionError("UAV candidate 0 (STAY) 必须合法。")
    return padded_features, masks, padded_positions


def collect_dual_timescale_episode(
    policy: DualTimescaleActorCritic,
    context: Any,
    aligned_steps: list[Any],
    duration_candidates: tuple[int, ...],
    *,
    deterministic: bool = False,
) -> EpisodeCollection:
    """冻结两套策略，采完整 episode，再分别交给 Micro/Macro 更新。"""

    if not aligned_steps:
        raise ValueError("完整 episode rollout 不能为空。")
    if len(context.action_spec.uav_ids) != 2:
        raise ValueError("第一版 autoregressive Micro PPO 固定要求两个 UAV。")
    if tuple(duration_candidates) != tuple(policy.duration_candidates):
        raise ValueError("Rollout 与 Macro policy duration candidates 不一致。")
    total_slots = len(aligned_steps)
    if total_slots != int(context.episode_slots):
        raise ValueError("双时间尺度正式 rollout 必须覆盖完整 episode。")
    tensor_spec = PrimitiveTensorSpec.create(context)
    max_uav_candidates = int(tensor_spec.uav_max_candidates) + 1
    micro_buffer = MicroRolloutBuffer()
    macro_buffer = MacroRolloutBuffer()
    cache = np.asarray(context.initial_cache, dtype=np.int8).copy()
    uav = {
        key: np.asarray(value, dtype=np.float64).copy()
        for key, value in _copy_uav_positions(context.initial_uav).items()
    }
    frame_start = 0
    frame_observation: np.ndarray | None = None
    frame_candidate_features: np.ndarray | None = None
    frame_candidate_masks: np.ndarray | None = None
    frame_duration_masks: np.ndarray | None = None
    frame_cache_index = 0
    frame_duration_index = 0
    frame_requested_duration = 0
    frame_actual_duration = 0
    frame_old_log_prob = 0.0
    frame_old_entropy = 0.0
    frame_old_value = 0.0
    frame_cache = cache.copy()
    frame_cache_changed = False
    frame_reward = 0.0
    elapsed_in_frame = 0
    keep_probabilities: list[float] = []
    update_group_probabilities: list[float] = []
    cache_entropies: list[float] = []
    duration_entropies: list[float] = []
    update_candidate_entropies: list[float] = []
    update_candidate_selection_counts: dict[bytes, int] = {}
    deterministic_cache_selections: list[dict[str, Any]] = []
    deterministic_uav_selections: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for slot_index, step in enumerate(aligned_steps):
        slot_started = time.perf_counter()
        previous_uav = {key: value.copy() for key, value in uav.items()}
        if slot_index == frame_start:
            frame_observation = _observation(cache, uav, step, context)
            _candidates, frame_candidate_features = _cache_candidates(
                cache, uav, step, context, tensor_spec
            )
            frame_candidate_masks = np.ones(len(_candidates), dtype=bool)
            if not bool(frame_candidate_masks[0]) or not np.array_equal(_candidates[0], cache):
                raise AssertionError("Cache candidate 0 必须为当前配置。")
            selected = policy.sample_macro(
                frame_observation,
                frame_candidate_features,
                frame_candidate_masks,
                deterministic=deterministic,
            )
            frame_cache_index = int(selected.cache_index)
            frame_duration_index = int(selected.duration_index)
            frame_requested_duration = int(selected.requested_duration)
            frame_actual_duration = actual_duration_for_frame(
                frame_requested_duration, slot_index, total_slots
            )
            frame_duration_masks = selected.duration_mask.copy()
            frame_old_log_prob = float(selected.log_prob.detach().cpu())
            frame_old_entropy = float(selected.entropy.detach().cpu())
            frame_old_value = float(selected.value.detach().cpu())
            cache_probabilities = np.asarray(selected.cache_probabilities, dtype=np.float64)
            if cache_probabilities.shape != frame_candidate_masks.shape or not np.all(
                np.isfinite(cache_probabilities)
            ) or not np.isclose(float(cache_probabilities.sum()), 1.0, rtol=1e-6, atol=1e-7):
                raise AssertionError("Macro Cache sampling probabilities are invalid or misaligned.")
            keep_probability = float(cache_probabilities[0])
            update_group_probability = float(cache_probabilities[1:].sum())
            keep_probabilities.append(keep_probability)
            update_group_probabilities.append(update_group_probability)
            cache_entropies.append(float(selected.cache_entropy.detach().cpu()))
            duration_entropies.append(float(selected.duration_entropy.detach().cpu()))
            update_candidate_probabilities = cache_probabilities[1:]
            if update_group_probability > 0.0:
                conditional_update_probabilities = (
                    update_candidate_probabilities / update_group_probability
                )
                positive_update_probabilities = conditional_update_probabilities[
                    conditional_update_probabilities > 0.0
                ]
                update_candidate_entropies.append(
                    float(-np.sum(positive_update_probabilities * np.log(positive_update_probabilities)))
                )
            else:
                update_candidate_entropies.append(0.0)
            frame_cache = _candidates[frame_cache_index].copy()
            frame_cache_changed = bool(np.any(frame_cache != cache))
            if deterministic:
                selected_cache_id = hashlib.sha256(
                    np.ascontiguousarray(frame_cache, dtype=np.int8).tobytes()
                ).hexdigest()[:16]
                deterministic_cache_selections.append(
                    {
                        "decision_slot": int(step.slot),
                        "selected_group": str(selected.cache_selected_group),
                        "selected_candidate_index": frame_cache_index,
                        "selected_candidate_id": selected_cache_id,
                        "selected_duration": frame_requested_duration,
                    }
                )
            if frame_cache_index > 0:
                candidate_signature = np.ascontiguousarray(
                    frame_cache, dtype=np.int8
                ).tobytes()
                update_candidate_selection_counts[candidate_signature] = (
                    update_candidate_selection_counts.get(candidate_signature, 0) + 1
                )
            frame_reward = 0.0
            elapsed_in_frame = 0

        if (
            frame_observation is None
            or frame_candidate_features is None
            or frame_candidate_masks is None
            or frame_duration_masks is None
        ):
            raise AssertionError("Macro frame 起点状态未初始化。")
        observation = _observation(frame_cache, uav, step, context)
        if micro_buffer.transitions and not micro_buffer.transitions[-1].done:
            micro_buffer.transitions[-1] = replace(
                micro_buffer.transitions[-1], next_observation=observation.copy()
            )
        policy_started = time.perf_counter()
        value = float(policy.micro_value(observation).detach().cpu())
        feature_lists: list[np.ndarray] = []
        position_lists: list[np.ndarray] = []
        actions: list[int] = []
        old_head_log_probs: list[Any] = []
        old_head_entropies: list[Any] = []
        candidate_probability_lists: list[np.ndarray] = []
        candidate_source_lists: list[tuple[str, ...]] = []
        valid_counts: list[int] = []
        selected_uav0_condition = np.zeros(2, dtype=np.float32)
        current_trial = {key: point.copy() for key, point in uav.items()}
        previous_positions = {key: point.copy() for key, point in previous_uav.items()}
        for uav_index, uav_id in enumerate(context.action_spec.uav_ids):
            candidates, candidate_features = _uav_candidates(
                uav_index,
                current_trial,
                previous_positions,
                frame_cache,
                step,
                context,
            )
            candidate_positions = [
                np.asarray(candidate.target_position, dtype=np.float64).copy()
                for candidate in candidates
            ]
            if uav_index == 0:
                condition = np.zeros(2, dtype=np.float32)
            else:
                condition = selected_uav0_condition.copy()
            sample = policy.sample_micro_head(
                observation,
                candidate_features,
                np.ones(len(candidate_features), dtype=bool),
                uav_index=uav_index,
                uav0_position_condition=condition,
                deterministic=deterministic,
            )
            selected_index = int(sample.selected_index)
            selected_candidate = candidates[selected_index]
            if deterministic:
                deterministic_uav_selections.append(
                    {
                        "slot": int(step.slot),
                        "uav_index": int(uav_index),
                        "selected_group": str(sample.selected_group),
                        "selected_candidate_index": selected_index,
                        "selected_candidate_source": str(
                            selected_candidate.proposal_source
                        ),
                        "selected_candidate_position": [
                            float(value) for value in selected_candidate.target_position
                        ],
                    }
                )
            uav_configuration.apply_uav_candidate(
                current_trial,
                previous_positions,
                str(uav_id),
                selected_candidate.target_position,
                {key: 1 for key in context.action_spec.uav_ids},
                context.cfg,
                context.flight_boundary,
            )
            if uav_index == 0:
                selected_uav0_condition = _positions_condition(
                    current_trial[str(uav_id)], context
                )
            actions.append(selected_index)
            old_head_log_probs.append(sample.log_prob)
            old_head_entropies.append(sample.entropy)
            candidate_probability_lists.append(sample.probabilities.copy())
            candidate_source_lists.append(tuple(item.proposal_source for item in candidates))
            valid_counts.append(len(candidates))
            feature_lists.append(candidate_features)
            position_lists.append(np.asarray(candidate_positions, dtype=np.float32))
        policy_seconds = time.perf_counter() - policy_started
        uav_configuration.validate_uav_configuration_transition(
            previous_positions,
            current_trial,
            {
                uav_id: int(
                    np.linalg.norm(current_trial[uav_id] - previous_positions[uav_id])
                    > float(
                        uav_configuration.uav_search_config_from_project(
                            context.cfg
                        ).position_tolerance_m
                    )
                )
                for uav_id in context.action_spec.uav_ids
            },
            context.cfg,
            context.flight_boundary,
        )
        features_padded, masks_padded, positions_padded = _pad_micro_candidates(
            feature_lists,
            position_lists,
            max_uav_candidates,
            int(tensor_spec.uav_feature_dim) + 2,
        )
        joint_log_prob = old_head_log_probs[0] + old_head_log_probs[1]
        joint_entropy = old_head_entropies[0] + old_head_entropies[1]
        probability_snapshot = np.zeros((2, max_uav_candidates), dtype=np.float32)
        for head_index in range(2):
            probability_snapshot[head_index, :valid_counts[head_index]] = candidate_probability_lists[head_index]
        cache_id = hashlib.sha256(
            np.ascontiguousarray(frame_cache, dtype=np.int8).tobytes()
        ).hexdigest()[:16]
        actual_cache_update_count = int(np.count_nonzero(cache_update_mask(cache, frame_cache)))
        performance_profile: dict[str, Any] = {}
        environment_started = time.perf_counter()
        result = slot_environment.execute_configured_slot(
            cache,
            previous_uav,
            frame_cache,
            current_trial,
            step.scenario_slice,
            step.actual_tasks,
            step.actual_vehicle_positions,
            context.local_cache,
            context.service_catalog,
            context.cfg,
            context.cache_nodes,
            context.cache_cfg,
            context.flight_boundary,
            slot_id=int(step.slot),
            time_s=float(step.time_s),
            performance_profile=performance_profile,
        )
        environment_seconds = time.perf_counter() - environment_started
        if not result.step_feasible or result.system_utility is None:
            raise RuntimeError(
                "正式选中配置导致 slot infeasible："
                f"origin={step.origin_index}, reason={result.infeasible_reason}"
            )
        _check_request_contract(cache, previous_uav, result, context.action_spec)
        reward = _check_reward(result)
        metrics = task_metrics_from_result(result, context.cache_nodes)
        done = slot_index == total_slots - 1
        next_cache = np.asarray(result.next_cache_state, dtype=np.int8).copy()
        next_uav = {
            key: np.asarray(value, dtype=np.float64).copy()
            for key, value in result.next_uav_positions.items()
        }
        next_observation = (
            np.zeros_like(observation)
            if done
            else _observation(
                next_cache,
                next_uav,
                aligned_steps[slot_index + 1],
                context,
            )
        )
        micro_buffer.append(
            MicroTransition(
                slot=int(step.slot),
                origin_index=int(step.origin_index),
                observation=observation.copy(),
                actions=np.asarray(actions, dtype=np.int64),
                old_joint_log_prob=float(joint_log_prob.detach().cpu()),
                old_uav_log_probs=np.asarray([item.detach().cpu() for item in old_head_log_probs], dtype=np.float32),
                old_uav_entropies=np.asarray([item.detach().cpu() for item in old_head_entropies], dtype=np.float32),
                old_joint_entropy=float(joint_entropy.detach().cpu()),
                old_value=value,
                reward=reward,
                next_observation=next_observation.copy(),
                done=done,
                uav_candidate_features=features_padded,
                uav_candidate_masks=masks_padded,
                uav_candidate_positions=positions_padded,
                uav_candidate_probabilities=probability_snapshot,
                uav_candidate_sources=(candidate_source_lists[0], candidate_source_lists[1]),
                uav0_position_condition=selected_uav0_condition.copy(),
                active_cache_configuration_id=cache_id,
                actual_task_utility=float(result.actual_task_utility),
                actual_cache_update_count=actual_cache_update_count,
                uav_movement_count=int(sum(action_index != 0 for action_index in actions)),
                cache_cost=float(result.cache_cost),
                movement_cost=float(result.movement_cost),
                task_count=metrics.task_count,
                total_task_delay_s=metrics.total_task_delay_s,
                total_vehicle_energy_j=metrics.total_vehicle_energy_j,
                execution_target_counts=dict(metrics.execution_target_counts),
                valid_candidate_count=np.asarray(valid_counts, dtype=np.int64),
                policy_selection_seconds=float(policy_seconds),
                configuration_seconds=float(performance_profile.get("config_seconds", 0.0)),
                matching_kkt_seconds=float(performance_profile.get("matching_kkt_seconds", 0.0)),
                slot_runtime_seconds=float(time.perf_counter() - slot_started),
                matching_summary=dict(result.matching_summary),
            )
        )
        frame_reward += (float(context.cfg["algorithm"]["ppo"]["gamma"]) ** elapsed_in_frame) * reward
        cache = next_cache
        uav = next_uav
        elapsed_in_frame += 1

        if elapsed_in_frame == frame_actual_duration or done:
            next_macro_state = (
                np.zeros_like(frame_observation)
                if done
                else _observation(cache, uav, aligned_steps[slot_index + 1], context)
            )
            macro_buffer.append(
                MacroTransition(
                    observation=frame_observation.copy(),
                    cache_candidate_features=frame_candidate_features.copy(),
                    cache_candidate_masks=frame_candidate_masks.copy(),
                    duration_masks=frame_duration_masks.copy(),
                    cache_action=frame_cache_index,
                    duration_action=frame_duration_index,
                    requested_duration=frame_requested_duration,
                    actual_duration=elapsed_in_frame,
                    old_macro_log_prob=frame_old_log_prob,
                    old_macro_entropy=frame_old_entropy,
                    old_value=frame_old_value,
                    frame_reward=float(frame_reward),
                    next_observation=next_macro_state.copy(),
                    done=done,
                    cache_configuration_id=cache_id,
                    actual_cache_update=frame_cache_changed,
                )
            )
            frame_start = slot_index + 1
    if not micro_buffer.transitions or not micro_buffer.transitions[-1].done:
        raise AssertionError("formal rollout must end at the true episode terminal")
    if not macro_buffer.transitions or not macro_buffer.transitions[-1].done:
        raise AssertionError("Macro buffer must end at the true episode terminal")

    durations = [int(item.requested_duration) for item in macro_buffer.transitions]
    actual_durations = [int(item.actual_duration) for item in macro_buffer.transitions]
    duration_distribution = {
        str(value): int(durations.count(value)) for value in policy.duration_candidates
    }
    actual_duration_distribution = {
        str(value): int(actual_durations.count(value))
        for value in sorted(set(actual_durations))
    }
    macro_count = len(macro_buffer)
    micro_count = len(micro_buffer)
    stay_counts = np.sum(
        np.stack([item.actions == 0 for item in micro_buffer.transitions]), axis=0
    )
    matching_summaries = [item.matching_summary for item in micro_buffer.transitions]
    episode_return = float(sum(item.reward for item in micro_buffer.transitions))
    keep_count = int(sum(item.cache_action == 0 for item in macro_buffer.transitions))
    update_count = int(macro_count - keep_count)
    requested_duration_counts = {
        str(value): int(durations.count(value)) for value in policy.duration_candidates
    }
    actual_duration_counts = {
        str(value): int(actual_durations.count(value))
        for value in policy.duration_candidates
    }
    update_candidate_unique_selected = len(update_candidate_selection_counts)
    update_candidate_top1_count = max(update_candidate_selection_counts.values(), default=0)
    update_candidate_top1_fraction = (
        float(update_candidate_top1_count / update_count) if update_count else 0.0
    )
    update_candidate_pattern = (
        "SINGLE_CANDIDATE_DOMINANCE"
        if update_candidate_top1_fraction >= 0.8
        else "NORMAL"
    )
    selected_update_duration_counts = {
        int(value) for value, count in requested_duration_counts.items() if count > 0
    }
    if keep_count == 0:
        cache_action_pattern = "ALL_UPDATE"
    elif update_count == 0:
        cache_action_pattern = "ALL_KEEP"
    else:
        cache_action_pattern = "NORMAL"
    cache_duration_pattern = (
        f"ALL_{next(iter(selected_update_duration_counts))}"
        if len(selected_update_duration_counts) == 1
        else "NORMAL"
    )
    uav_action_patterns = []
    for head_index in range(2):
        stay_count = int(stay_counts[head_index])
        move_count = int(micro_count - stay_count)
        uav_action_patterns.append(
            "NORMAL" if stay_count and move_count else "ALL_STAY" if stay_count else "ALL_MOVE"
        )
    update_candidate_valid_entropies = [
        entropy
        for entropy, update_probability in zip(
            update_candidate_entropies, update_group_probabilities, strict=True
        )
        if update_probability > 0.0
    ]
    update_candidate_selection_entropy = 0.0
    if update_count:
        for selected_count in update_candidate_selection_counts.values():
            selection_probability = selected_count / update_count
            update_candidate_selection_entropy -= selection_probability * float(
                np.log(selection_probability)
            )
    summary = {
        "episode_return": episode_return,
        "macro_return": float(sum(item.frame_reward for item in macro_buffer.transitions)),
        "macro_frame_count": macro_count,
        "macro_entropy": float(np.mean([item.old_macro_entropy for item in macro_buffer.transitions])),
        "macro_cache_entropy": float(np.mean(cache_entropies)) if cache_entropies else 0.0,
        "macro_duration_entropy": float(np.mean(duration_entropies)) if duration_entropies else 0.0,
        "macro_duration_distribution": duration_distribution,
        "requested_duration_mean": float(np.mean(durations)),
        "actual_duration_mean": float(
            np.mean([item.actual_duration for item in macro_buffer.transitions])
        ),
        "duration_distribution": duration_distribution,
        "actual_duration_distribution": actual_duration_distribution,
        "requested_duration_counts": requested_duration_counts,
        "actual_duration_counts": actual_duration_counts,
        "keep_count": keep_count,
        "update_count": update_count,
        "update_selected_candidate_count": update_count,
        "keep_rate": float(keep_count / macro_count) if macro_count else 0.0,
        "update_rate": float(update_count / macro_count) if macro_count else 0.0,
        "mean_keep_probability": float(np.mean(keep_probabilities)) if keep_probabilities else 0.0,
        "mean_update_group_probability": (
            float(np.mean(update_group_probabilities)) if update_group_probabilities else 0.0
        ),
        "update_candidate_unique_selected": update_candidate_unique_selected,
        "update_candidate_entropy": (
            float(np.mean(update_candidate_valid_entropies))
            if update_candidate_valid_entropies
            else 0.0
        ),
        "update_candidate_selection_entropy": float(update_candidate_selection_entropy),
        "update_candidate_top1_fraction": update_candidate_top1_fraction,
        "update_candidate_pattern": update_candidate_pattern,
        "cache_action_pattern": cache_action_pattern,
        "cache_duration_pattern": cache_duration_pattern,
        "actual_cache_update_rate": (
            float(np.mean([item.actual_cache_update for item in macro_buffer.transitions]))
            if macro_count
            else 0.0
        ),
        "uav0_stay_rate": float(stay_counts[0] / max(micro_count, 1)),
        "uav1_stay_rate": float(stay_counts[1] / max(micro_count, 1)),
        "uav0_move_rate": float(1.0 - stay_counts[0] / max(micro_count, 1)),
        "uav1_move_rate": float(1.0 - stay_counts[1] / max(micro_count, 1)),
        "uav0_action_pattern": uav_action_patterns[0],
        "uav1_action_pattern": uav_action_patterns[1],
        "uav0_entropy": float(np.mean([item.old_uav_entropies[0] for item in micro_buffer.transitions])),
        "uav1_entropy": float(np.mean([item.old_uav_entropies[1] for item in micro_buffer.transitions])),
        "micro_candidate_entropy": float(np.mean([item.old_joint_entropy for item in micro_buffer.transitions])),
        "uav_candidate_usage_by_index": [
            [int(sum(int(item.actions[head]) == candidate_index for item in micro_buffer.transitions)) for candidate_index in range(max_uav_candidates)]
            for head in range(2)
        ],
        "uav_proposal_source_usage": {
            f"uav{head}:{source}": int(sum(item.uav_candidate_sources[head][int(item.actions[head])] == source for item in micro_buffer.transitions))
            for head in range(2)
            for source in sorted({name for item in micro_buffer.transitions for name in item.uav_candidate_sources[head]})
        },
        "valid_candidate_count_mean": float(
            np.mean([np.mean(item.valid_candidate_count) for item in micro_buffer.transitions])
        ),
        "uav0_mean_valid_candidates": float(
            np.mean([item.valid_candidate_count[0] for item in micro_buffer.transitions])
        ),
        "uav1_mean_valid_candidates": float(
            np.mean([item.valid_candidate_count[1] for item in micro_buffer.transitions])
        ),
        "total_task_delay_s": float(
            sum(item.total_task_delay_s for item in micro_buffer.transitions)
        ),
        "total_vehicle_energy_j": float(
            sum(item.total_vehicle_energy_j for item in micro_buffer.transitions)
        ),
        "total_cache_cost": float(sum(item.cache_cost for item in micro_buffer.transitions)),
        "total_movement_cost": float(
            sum(item.movement_cost for item in micro_buffer.transitions)
        ),
        "total_task_utility": float(
            sum(item.actual_task_utility for item in micro_buffer.transitions)
        ),
        "mean_task_delay_s": float(
            sum(item.total_task_delay_s for item in micro_buffer.transitions)
            / max(sum(item.task_count for item in micro_buffer.transitions), 1)
        ),
        "mean_vehicle_energy_j": float(
            sum(item.total_vehicle_energy_j for item in micro_buffer.transitions)
            / max(sum(item.task_count for item in micro_buffer.transitions), 1)
        ),
        "matching_call_count": int(len(micro_buffer)),
        "kkt_call_count": int(
            sum(int(item.get("KKT_SOLVER_CALL_COUNT", 0)) for item in matching_summaries)
        ),
        "environment_execution_count": int(len(micro_buffer)),
        "mean_slot_runtime_seconds": float(
            np.mean([item.slot_runtime_seconds for item in micro_buffer.transitions])
        ),
        "episode_runtime_seconds": float(time.perf_counter() - start_time),
        "total_tasks": int(sum(item.task_count for item in micro_buffer.transitions)),
        "exact_q_branch_requests": 0,
        "exact_q_environment_executions": 0,
    }
    if deterministic:
        summary["deterministic_action_selection"] = {
            "cache": deterministic_cache_selections,
            "uav": deterministic_uav_selections,
        }
    return EpisodeCollection(micro_buffer, macro_buffer, summary)


def compute_gae(
    rewards: Any,
    values: Any,
    dones: Any,
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """普通 Micro GAE，terminal transition 不 bootstrap。"""

    rewards_a = np.asarray(rewards, dtype=np.float64).reshape(-1)
    values_a = np.asarray(values, dtype=np.float64).reshape(-1)
    dones_a = np.asarray(dones, dtype=bool).reshape(-1)
    if not len(rewards_a) or not (len(rewards_a) == len(values_a) == len(dones_a)):
        raise ValueError("Micro GAE input 长度必须一致且非空。")
    if not np.all(np.isfinite(rewards_a)) or not np.all(np.isfinite(values_a)):
        raise ValueError("Micro GAE 输入包含 NaN/Inf。")
    gamma = float(gamma)
    gae_lambda = float(gae_lambda)
    if not (0.0 <= gamma <= 1.0 and 0.0 <= gae_lambda <= 1.0):
        raise ValueError("gamma 和 gae_lambda 必须位于 [0,1]。")
    advantages = np.zeros(len(rewards_a), dtype=np.float64)
    next_advantage = 0.0
    for index in range(len(rewards_a) - 1, -1, -1):
        next_value = (
            float(bootstrap_value)
            if index == len(rewards_a) - 1
            else float(values_a[index + 1])
        )
        nonterminal = 0.0 if bool(dones_a[index]) else 1.0
        delta = rewards_a[index] + gamma * nonterminal * next_value - values_a[index]
        next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        advantages[index] = next_advantage
    returns = advantages + values_a
    if not np.all(np.isfinite(advantages)) or not np.all(np.isfinite(returns)):
        raise FloatingPointError("Micro GAE output 包含 NaN/Inf。")
    return advantages.astype(np.float32), returns.astype(np.float32)


def compute_macro_smdp_gae(
    frame_rewards: Any,
    values: Any,
    next_values: Any,
    actual_durations: Any,
    dones: Any,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """使用每个 Macro frame 实际 duration 的 SMDP TD 与 duration-aware GAE。"""

    rewards_a = np.asarray(frame_rewards, dtype=np.float64).reshape(-1)
    values_a = np.asarray(values, dtype=np.float64).reshape(-1)
    next_values_a = np.asarray(next_values, dtype=np.float64).reshape(-1)
    durations_a = np.asarray(actual_durations, dtype=np.int64).reshape(-1)
    dones_a = np.asarray(dones, dtype=bool).reshape(-1)
    n = len(rewards_a)
    if n <= 0 or any(len(value) != n for value in (values_a, next_values_a, durations_a, dones_a)):
        raise ValueError("Macro SMDP GAE input 长度必须一致且非空。")
    if np.any(durations_a <= 0):
        raise ValueError("每个 Macro actual_duration 必须为正数。")
    if not all(np.all(np.isfinite(value)) for value in (rewards_a, values_a, next_values_a)):
        raise ValueError("Macro SMDP GAE 输入包含 NaN/Inf。")
    gamma = float(gamma)
    gae_lambda = float(gae_lambda)
    if not (0.0 <= gamma <= 1.0 and 0.0 <= gae_lambda <= 1.0):
        raise ValueError("gamma 和 gae_lambda 必须位于 [0,1]。")
    advantages = np.zeros(n, dtype=np.float64)
    next_advantage = 0.0
    for index in range(n - 1, -1, -1):
        duration = int(durations_a[index])
        discount = gamma ** duration
        trace = (gamma * gae_lambda) ** duration
        nonterminal = 0.0 if bool(dones_a[index]) else 1.0
        delta = (
            rewards_a[index]
            + discount * next_values_a[index] * nonterminal
            - values_a[index]
        )
        next_advantage = delta + trace * nonterminal * next_advantage
        advantages[index] = next_advantage
    returns = advantages + values_a
    if not np.all(np.isfinite(advantages)) or not np.all(np.isfinite(returns)):
        raise FloatingPointError("Macro SMDP GAE output 包含 NaN/Inf。")
    return advantages.astype(np.float32), returns.astype(np.float32)
