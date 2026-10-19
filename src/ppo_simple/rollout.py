"""双时间尺度 on-policy rollout、分离 buffer 与 GAE。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
import random
import time
from typing import Any, Callable, Mapping

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - 仅静态导入环境
    torch = None  # type: ignore[assignment]

from decision import cache_configuration, slot_environment, uav_configuration

from . import interface
from .interface import _copy_uav_positions
from .policy import (
    DualTimescaleActorCritic,
    MultiNodeFactorizedActorCritic,
    SparseCoordinatedCacheActorCritic,
    _macro_log_prob_from_components,
    _sum_log_probabilities,
)
from .primitive_control import (
    CACHE_ECONOMIC_FEATURE_DIM,
    PrimitiveTensorSpec,
    _normalized_xy,
    cache_economic_candidate_features,
    _target_cache_proposals_with_features,
    _uav_target_position_feature,
)

EXECUTION_CLASSES = ("SV", "RSU", "UAV", "CLOUD")


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
    uav_candidate_sources: tuple[tuple[str, ...], ...]
    uav_ids: tuple[str, ...]
    uav_position_conditions: np.ndarray
    active_cache_configuration_id: str
    actual_task_utility: float
    actual_cache_update_count: int
    uav_movement_count: int
    cache_cost: float
    cache_cost_by_node: dict[str, float]
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


def _aggregate_node_task_counts(
    transitions: list[MicroTransition],
) -> dict[str, Any]:
    """Aggregate final Matching execution counts without rerunning allocation."""

    totals: dict[str, Any] = {
        "SV": 0,
        "SV_by_node": {},
        "RSU": 0,
        "RSU_by_node": {},
        "UAV": 0,
        "UAV0": 0,
        "UAV1": 0,
        "UAV_by_node": {},
        "Cloud": 0,
    }
    scalar_fields = ("SV", "RSU", "UAV", "UAV0", "UAV1", "Cloud")
    node_fields = ("SV_by_node", "RSU_by_node", "UAV_by_node")
    for transition in transitions:
        counts = transition.matching_summary.get("node_task_count")
        if not isinstance(counts, Mapping):
            raise ValueError("Matching summary lacks final per-node task counts.")
        for field in scalar_fields:
            totals[field] += int(counts.get(field, 0))
        for field in node_fields:
            by_node = counts.get(field, {})
            if not isinstance(by_node, Mapping):
                raise ValueError(f"Matching {field} must be a mapping.")
            for node_id, count in by_node.items():
                key = str(node_id)
                totals[field][key] = totals[field].get(key, 0) + int(count)
        slot_total = sum(
            int(counts.get(field, 0))
            for field in ("SV", "RSU", "UAV", "Cloud")
        )
        if slot_total != int(transition.task_count):
            raise ValueError(
                "Per-node Matching counts do not sum to the realized task count."
            )
    return totals


@dataclass(frozen=True)
class HoldingTransition:
    """单个 Holding termination 决策及其行为概率合同。"""

    observation: np.ndarray
    elapsed_slots: int
    action: int
    old_log_prob: float
    old_entropy: float


@dataclass(frozen=True)
class MacroTransition:
    """一个 Cache frame 的 SMDP transition 与其内部 Holding 决策序列。"""

    observation: np.ndarray
    cache_candidate_features: np.ndarray
    cache_candidate_masks: np.ndarray
    cache_action: Any
    actual_duration: int
    old_cache_log_prob: float
    old_node_cache_log_probs: np.ndarray | None
    old_node_cache_entropies: np.ndarray | None
    old_macro_log_prob: float
    old_macro_entropy: float
    holding_transitions: tuple[HoldingTransition, ...]
    old_value: float
    old_next_value: float
    frame_reward: float
    next_observation: np.ndarray
    done: bool
    cache_configuration_id: str
    actual_cache_update: bool
    actual_cache_update_by_node: np.ndarray | None

    @property
    def termination_observations(self) -> np.ndarray:
        if not self.holding_transitions:
            return np.empty((0, int(self.observation.shape[-1])), dtype=np.float32)
        return np.stack([item.observation for item in self.holding_transitions]).astype(
            np.float32, copy=False
        )

    @property
    def termination_elapsed_slots(self) -> np.ndarray:
        return np.asarray([item.elapsed_slots for item in self.holding_transitions], dtype=np.int64)

    @property
    def termination_actions(self) -> np.ndarray:
        return np.asarray([item.action for item in self.holding_transitions], dtype=np.int64)

    @property
    def old_termination_log_probs(self) -> np.ndarray:
        return np.asarray([item.old_log_prob for item in self.holding_transitions], dtype=np.float32)

    @property
    def old_termination_entropies(self) -> np.ndarray:
        return np.asarray([item.old_entropy for item in self.holding_transitions], dtype=np.float32)

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
    min_holding_slots: int = 1
    max_holding_slots: int = 5
    cache_action_mode: str = interface.project_config.CACHE_ACTION_MODE_SINGLE_NODE

    def append(self, transition: MacroTransition) -> None:
        if self.transitions and bool(self.transitions[-1].done):
            raise ValueError("不能在 terminal Macro transition 后追加数据。")
        duration_is_eligible = int(self.min_holding_slots) <= int(
            transition.actual_duration
        ) <= int(self.max_holding_slots)
        terminal_short_frame = bool(transition.done) and 0 < int(
            transition.actual_duration
        ) < int(self.min_holding_slots)
        if not (duration_is_eligible or terminal_short_frame):
            raise ValueError("Macro actual_duration 超出 adaptive holding bounds。")
        if not np.isfinite(float(transition.old_next_value)):
            raise ValueError("Macro rollout old_next_value must be finite.")
        holding_elapsed = [int(item.elapsed_slots) for item in transition.holding_transitions]
        if holding_elapsed != sorted(set(holding_elapsed)):
            raise ValueError("Holding decisions must have strictly increasing elapsed Slot indices.")
        for item in transition.holding_transitions:
            if (
                int(item.action) not in (0, 1)
                or not int(self.min_holding_slots) <= int(item.elapsed_slots) < int(self.max_holding_slots)
                or np.asarray(item.observation).shape != np.asarray(transition.observation).shape
                or not np.isfinite(np.asarray(item.observation, dtype=np.float32)).all()
                or not np.isfinite(float(item.old_log_prob))
                or not np.isfinite(float(item.old_entropy))
            ):
                raise ValueError("Holding transition violates its state/action/probability contract.")
        if not np.isfinite(float(transition.old_macro_log_prob)) or not np.isfinite(
            float(transition.old_cache_log_prob)
        ):
            raise ValueError("Macro rollout log-probabilities must be finite.")
        holding_log_prob_sum = np.float32(
            np.sum(
                np.asarray(
                    [item.old_log_prob for item in transition.holding_transitions],
                    dtype=np.float64,
                ),
                dtype=np.float64,
            )
        )
        expected_joint_log_prob = np.float32(
            np.float32(transition.old_cache_log_prob) + holding_log_prob_sum
        )
        if not np.isclose(
            float(transition.old_macro_log_prob),
            float(expected_joint_log_prob),
            rtol=0.0,
            atol=2.0e-6,
        ):
            raise ValueError("Macro behavior joint log-prob does not match Cache plus Holding factors.")
        action_values = np.asarray(transition.cache_action, dtype=np.int64)
        candidate_masks = np.asarray(transition.cache_candidate_masks, dtype=bool)
        if candidate_masks.ndim == 2:
            node_count = int(candidate_masks.shape[0])
            if action_values.shape != (node_count,) or not np.all(candidate_masks[:, 0]):
                raise ValueError("factorized Cache actions/masks must align by node")
            if np.any(action_values < 0) or np.any(
                action_values >= candidate_masks.shape[1]
            ):
                raise ValueError("factorized Cache action index out of range")
            if not np.all(
                candidate_masks[np.arange(node_count), action_values]
            ):
                raise ValueError("factorized Cache action is not legal under its saved mask")
            if self.cache_action_mode == interface.project_config.CACHE_ACTION_MODE_SPARSE_COORDINATED:
                if np.count_nonzero(action_values) > 1:
                    raise ValueError("sparse Cache rollout may update at most one node per frame")
                if (
                    transition.old_node_cache_log_probs is not None
                    or transition.old_node_cache_entropies is not None
                    or transition.actual_cache_update_by_node is None
                    or np.asarray(transition.actual_cache_update_by_node).shape != (node_count,)
                    or np.count_nonzero(transition.actual_cache_update_by_node) > 1
                ):
                    raise ValueError("sparse Cache transition must retain one joint probability and at most one update")
                if bool(np.any(transition.actual_cache_update_by_node)) and not bool(
                    np.any(action_values > 0)
                ):
                    raise ValueError("sparse Cache changed without a sampled UPDATE action")
            else:
                if (
                    transition.old_node_cache_log_probs is None
                    or transition.old_node_cache_entropies is None
                    or np.asarray(transition.old_node_cache_log_probs).shape != (node_count,)
                    or np.asarray(transition.old_node_cache_entropies).shape != (node_count,)
                    or not np.isfinite(transition.old_node_cache_log_probs).all()
                    or not np.isfinite(transition.old_node_cache_entropies).all()
                    or transition.actual_cache_update_by_node is None
                    or np.asarray(transition.actual_cache_update_by_node).shape != (node_count,)
                ):
                    raise ValueError("factorized Cache node probability/update trace is incomplete")
                expected_cache_log_prob = np.float32(
                    np.sum(
                        np.asarray(transition.old_node_cache_log_probs, dtype=np.float32),
                        dtype=np.float64,
                    )
                )
                if float(expected_cache_log_prob) != float(
                    np.float32(transition.old_cache_log_prob)
                ):
                    raise ValueError("node Cache log-probabilities do not sum to the joint Cache log-probability")
        elif candidate_masks.ndim != 1 or action_values.ndim != 0:
            raise ValueError("unified Cache actions and masks must retain scalar/indexed semantics")
        if bool(transition.done) and float(transition.old_next_value) != 0.0:
            raise ValueError("Terminal Macro transitions must have zero old_next_value.")
        trace_lengths = {
            len(transition.termination_observations),
            len(transition.termination_elapsed_slots),
            len(transition.termination_actions),
            len(transition.old_termination_log_probs),
            len(transition.old_termination_entropies),
        }
        if len(trace_lengths) != 1:
            raise ValueError("Termination trace arrays must have matching lengths.")
        trace_count = next(iter(trace_lengths))
        if trace_count > int(self.max_holding_slots) - 1:
            raise ValueError("Termination trace exceeds the number of eligible decisions.")
        if trace_count:
            if (
                transition.termination_observations.ndim != 2
                or np.any(transition.termination_elapsed_slots < 1)
                or np.any(
                    transition.termination_elapsed_slots
                    >= int(self.max_holding_slots)
                )
                or not np.array_equal(
                    transition.termination_elapsed_slots,
                    np.arange(1, trace_count + 1, dtype=np.int64),
                )
                or np.any(transition.termination_actions < 0)
                or np.any(transition.termination_actions > 1)
                or not np.isfinite(transition.termination_observations).all()
                or not np.isfinite(transition.old_termination_log_probs).all()
                or not np.isfinite(transition.old_termination_entropies).all()
            ):
                raise ValueError("Termination trace contains invalid sampled decisions.")
            if np.any(transition.termination_actions[:-1] == 1):
                raise ValueError("TERMINATE may appear only on the final sampled decision.")
        if bool(transition.done) and trace_count and np.any(
            transition.termination_actions == 1
        ):
            raise ValueError("Terminal Macro transitions cannot sample TERMINATE.")
        expected_duration = (
            int(transition.termination_elapsed_slots[-1])
            if trace_count and int(transition.termination_actions[-1]) == 1
            else trace_count + int(self.min_holding_slots)
        )
        if expected_duration != int(transition.actual_duration):
            raise ValueError("Termination trace does not explain the realized holding duration.")
        termination_log_prob_sum = np.float32(
            np.sum(
                np.asarray(transition.old_termination_log_probs, dtype=np.float32),
                dtype=np.float64,
            )
        )
        reconstructed_log_prob = np.float32(
            np.float32(transition.old_cache_log_prob) + termination_log_prob_sum
        )
        if float(reconstructed_log_prob) != float(
            np.float32(transition.old_macro_log_prob)
        ):
            raise ValueError("Stored Macro log-probability does not decompose into Cache and termination.")
        self.transitions.append(transition)

    def __len__(self) -> int:
        return len(self.transitions)


@dataclass(frozen=True)
class EpisodeCollection:
    """完整 episode 的分离 rollout 与轻量运行计数。"""

    micro: MicroRolloutBuffer
    macro: MacroRolloutBuffer
    summary: dict[str, Any]




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
    """检查未请求 Cache 或 UAV 更新时环境状态保持不变。"""

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
    active_service_locks: np.ndarray | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """candidate 0 是当前 Cache，其余候选复用现有节点 Cache proposal。"""

    current = np.asarray(cache, dtype=np.int8).copy()
    locks = (
        np.zeros(current.shape, dtype=bool)
        if active_service_locks is None
        else np.asarray(active_service_locks, dtype=bool)
    )
    if locks.shape != current.shape:
        raise ValueError("active Cache service locks do not align with node/service order")
    proposal_nodes = uav_configuration.cache_nodes_with_uav_positions(
        context.cache_nodes,
        {key: np.asarray(value, dtype=np.float64) for key, value in uav.items()},
    )
    candidates = [current.copy()]
    primitive_features = [
        np.zeros(int(tensor_spec.cache_feature_dim), dtype=np.float32)
    ]
    seen = {current.tobytes()}
    prediction_horizon_mask = getattr(step, "prediction_horizon_mask", None)
    for node_index, _node_id in enumerate(context.action_spec.cache_node_ids):
        triggers = np.zeros(current.shape[0], dtype=np.int8)
        triggers[node_index] = 1
        node_candidates, node_features = _target_cache_proposals_with_features(
            current,
            current,
            triggers,
            node_index,
            np.zeros(int(context.observation_spec.observation_dim), dtype=np.float32),
            step,
            context,
            proposal_nodes,
            prediction_horizon_mask=prediction_horizon_mask,
            locked_services=locks[node_index],
        )
        if node_features.shape != (
            len(node_candidates),
            int(tensor_spec.cache_feature_dim),
        ):
            raise AssertionError("selected Cache candidates and primitive features are misaligned")
        for item, node_feature in zip(node_candidates, node_features, strict=True):
            target = current.copy()
            target[node_index] = np.asarray(item.target_node_cache, dtype=np.int8)
            key = target.tobytes()
            if key not in seen:
                seen.add(key)
                candidates.append(target)
                primitive_features.append(np.asarray(node_feature, dtype=np.float32).copy())
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
    if context.policy_schema not in {
        interface.PROPOSED_POLICY_SCHEMA,
        interface.MULTI_NODE_FACTORIZED_POLICY_SCHEMA,
        interface.SPARSE_COORDINATED_CACHE_POLICY_SCHEMA,
    }:
        raise ValueError(f"unsupported formal Macro Cache schema: {context.policy_schema}")
    macro_base_dim = 2 * int(current.size) + 1
    if features.shape[1] != macro_base_dim:
        raise ValueError("Macro candidate base feature dimension does not match Cache state")
    economic_features = cache_economic_candidate_features(
        current,
        candidates,
        np.stack(primitive_features).astype(np.float32, copy=False),
        step,
        context,
        proposal_nodes,
        prediction_horizon_mask=prediction_horizon_mask,
    )
    if economic_features.shape != (len(candidates), CACHE_ECONOMIC_FEATURE_DIM):
        raise AssertionError("Cache economic features must align with candidate order")
    features = np.concatenate((features, economic_features), axis=1).astype(
        np.float32, copy=False
    )
    return candidates, features


def _cache_node_candidates(
    cache: np.ndarray,
    uav: Mapping[str, Any],
    step: Any,
    context: Any,
    tensor_spec: PrimitiveTensorSpec,
    active_service_locks: np.ndarray | None = None,
) -> tuple[list[list[np.ndarray]], np.ndarray, np.ndarray]:
    """Build one KEEP-plus-legal-UPDATE head per node from the same old cache."""

    current = np.asarray(cache, dtype=np.int8).copy()
    node_ids = tuple(str(value) for value in context.action_spec.cache_node_ids)
    if current.ndim != 2 or current.shape[0] != len(node_ids):
        raise ValueError("multi-node Cache candidate state does not match stable node order")
    locks = (
        np.zeros(current.shape, dtype=bool)
        if active_service_locks is None
        else np.asarray(active_service_locks, dtype=bool)
    )
    if locks.shape != current.shape:
        raise ValueError("active Cache service locks do not align with node/service order")
    proposal_nodes = uav_configuration.cache_nodes_with_uav_positions(
        context.cache_nodes,
        {key: np.asarray(value, dtype=np.float64) for key, value in uav.items()},
    )
    prediction_horizon_mask = getattr(step, "prediction_horizon_mask", None)
    node_candidates: list[list[np.ndarray]] = []
    primitive_rows: list[np.ndarray] = []
    flat_candidates: list[np.ndarray] = []
    candidate_owners: list[tuple[int, int]] = []
    for node_index, _node_id in enumerate(node_ids):
        triggers = np.zeros(current.shape[0], dtype=np.int8)
        triggers[node_index] = 1
        proposals, primitive = _target_cache_proposals_with_features(
            current,
            current,
            triggers,
            node_index,
            np.zeros(int(context.observation_spec.observation_dim), dtype=np.float32),
            step,
            context,
            proposal_nodes,
            prediction_horizon_mask=prediction_horizon_mask,
            locked_services=locks[node_index],
        )
        if primitive.shape != (
            len(proposals),
            int(tensor_spec.cache_feature_dim),
        ):
            raise AssertionError("per-node Cache proposals and primitive features are misaligned")
        candidates = [current.copy()]
        primitive_for_node = [
            np.zeros(int(tensor_spec.cache_feature_dim), dtype=np.float32)
        ]
        seen = {current.tobytes()}
        for proposal, primitive_feature in zip(proposals, primitive, strict=True):
            target = current.copy()
            target[node_index] = np.asarray(proposal.target_node_cache, dtype=np.int8)
            key = target.tobytes()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(target)
            primitive_for_node.append(np.asarray(primitive_feature, dtype=np.float32).copy())
        if len(candidates) > int(context.cache_cfg.target_candidate_capacity) + 1:
            raise AssertionError("per-node candidates exceeded the configured 64-update limit")
        node_candidates.append(candidates)
        for candidate_index, (target, primitive_feature) in enumerate(
            zip(candidates, primitive_for_node, strict=True)
        ):
            if candidate_index == 0 and node_index > 0:
                continue
            flat_candidates.append(target)
            primitive_rows.append(primitive_feature)
            candidate_owners.append((node_index, candidate_index))

    primitive_features = np.stack(primitive_rows).astype(np.float32, copy=False)
    economic = cache_economic_candidate_features(
        current,
        flat_candidates,
        primitive_features,
        step,
        context,
        proposal_nodes,
        prediction_horizon_mask=prediction_horizon_mask,
    )
    if economic.shape != (len(flat_candidates), CACHE_ECONOMIC_FEATURE_DIM):
        raise AssertionError("factorized Cache economic features do not align with candidates")
    max_candidates = max(len(items) for items in node_candidates)
    if max_candidates > int(context.cache_cfg.target_candidate_capacity) + 1:
        raise AssertionError("factorized Cache candidate padding exceeds the configured limit")
    features = np.zeros(
        (
            len(node_ids),
            max_candidates,
            2 * int(current.size) + 1 + CACHE_ECONOMIC_FEATURE_DIM,
        ),
        dtype=np.float32,
    )
    masks = np.zeros((len(node_ids), max_candidates), dtype=bool)
    flat_current = current.reshape(-1).astype(np.float32)
    for flat_index, (node_index, candidate_index) in enumerate(candidate_owners):
        candidate = node_candidates[node_index][candidate_index]
        base = np.concatenate(
            (
                candidate.reshape(-1).astype(np.float32),
                candidate.reshape(-1).astype(np.float32) - flat_current,
                np.asarray(
                    [float(candidate_index > 0) / max(float(len(node_ids)), 1.0)],
                    dtype=np.float32,
                ),
            )
        )
        row = np.concatenate((base, economic[flat_index])).astype(np.float32, copy=False)
        if row.shape != (2 * int(current.size) + 1 + CACHE_ECONOMIC_FEATURE_DIM,):
            raise AssertionError("factorized Cache candidate feature dimension changed")
        features[node_index, candidate_index] = row
        masks[node_index, candidate_index] = True
    for node_index in range(1, len(node_ids)):
        features[node_index, 0] = features[0, 0]
        masks[node_index, 0] = True
    if not np.all(masks[:, 0]) or not np.allclose(features[:, 0, -CACHE_ECONOMIC_FEATURE_DIM], 1.0):
        raise AssertionError("every factorized Cache head must retain legal KEEP candidate zero")
    if not np.isfinite(features[masks]).all():
        raise FloatingPointError("factorized Cache candidate features contain NaN/Inf")
    return node_candidates, features, masks


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
        prediction_horizon_mask=step.prediction_horizon_mask,
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
        prediction_horizon_mask=step.prediction_horizon_mask,
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
    head_count = len(features)
    if head_count <= 0 or len(positions) != head_count:
        raise ValueError("Micro candidate heads and positions must have matching positive counts.")
    padded_features = np.zeros((head_count, max_candidates, feature_dim), dtype=np.float32)
    masks = np.zeros((head_count, max_candidates), dtype=bool)
    padded_positions = np.zeros((head_count, max_candidates, 2), dtype=np.float32)
    for head in range(head_count):
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
    *,
    deterministic: bool = False,
    episode_seed: int | None = None,
    max_slots: int | None = None,
    progress_callback: Callable[[int, int, float, float], None] | None = None,
    performance_sample_slots: set[int] | frozenset[int] | None = None,
    performance_sample_callback: Callable[[dict[str, Any]], None] | None = None,
) -> EpisodeCollection:
    """冻结策略完成完整 episode rollout，并返回分离的 Micro/Macro 数据。

    内部性能计时仅在同时提供采样回调和 ``aligned_steps`` 零基索引时启用。
    """

    if not aligned_steps:
        raise ValueError("完整 episode rollout 不能为空。")
    uav_count = len(context.action_spec.uav_ids)
    if uav_count <= 0:
        raise ValueError("联合 MAPPO 至少要求一个 UAV action head。")
    adaptive_holding_enabled = bool(
        context.cfg["algorithm"]["ppo"].get("adaptive_holding_enabled", False)
    )
    min_holding_slots = int(policy.adaptive_holding_min_slots)
    max_holding_slots = int(policy.adaptive_holding_max_slots)
    if adaptive_holding_enabled:
        if min_holding_slots <= 0 or max_holding_slots < min_holding_slots:
            raise ValueError("Adaptive holding rollout contract is invalid.")
    elif min_holding_slots != 2 or max_holding_slots != 2:
        raise ValueError("Fixed2 compatibility mode requires a 2-slot Cache holding period.")
    if max_holding_slots > int(context.observation_spec.prediction_horizon):
        raise ValueError("Adaptive holding maximum exceeds prediction horizon.")
    total_slots = len(aligned_steps)
    if total_slots != int(context.episode_slots):
        raise ValueError("双时间尺度正式 rollout 必须覆盖完整 episode。")
    rollout_slots = total_slots if max_slots is None else int(max_slots)
    if rollout_slots <= 0 or rollout_slots > total_slots:
        raise ValueError("max_slots 必须位于完整 Episode 的 Slot 范围内。")
    window_truncated = rollout_slots < total_slots
    sampled_slot_indices = (
        frozenset(int(slot) for slot in performance_sample_slots)
        if performance_sample_slots is not None
        else None
    )
    if (sampled_slot_indices is None) != (performance_sample_callback is None):
        raise ValueError(
            "性能采样Slot与结果回调必须同时提供或同时省略。"
        )
    if sampled_slot_indices is not None and (
        not sampled_slot_indices
        or any(slot < 0 or slot >= total_slots for slot in sampled_slot_indices)
    ):
        raise ValueError("性能采样Slot索引必须非空且位于Episode范围内。")
    tensor_spec = PrimitiveTensorSpec.create(context)
    max_uav_candidates = int(tensor_spec.uav_max_candidates) + 1
    micro_buffer = MicroRolloutBuffer()
    macro_buffer = MacroRolloutBuffer(
        min_holding_slots=min_holding_slots,
        max_holding_slots=max_holding_slots,
        cache_action_mode=str(context.cache_action_mode),
    )
    cache = np.asarray(context.initial_cache, dtype=np.int8).copy()
    uav = {
        key: np.asarray(value, dtype=np.float64).copy()
        for key, value in _copy_uav_positions(context.initial_uav).items()
    }
    frame_start = 0
    frame_observation: np.ndarray | None = None
    frame_candidate_features: np.ndarray | None = None
    frame_candidate_masks: np.ndarray | None = None
    frame_cache_index = 0
    frame_cache_actions = np.asarray(0, dtype=np.int64)
    frame_old_cache_log_prob = 0.0
    frame_old_node_cache_log_probs: np.ndarray | None = None
    frame_old_node_cache_entropies: np.ndarray | None = None
    frame_old_macro_entropy = 0.0
    frame_old_value = 0.0
    frame_cache = cache.copy()
    frame_cache_changed = False
    frame_reward = 0.0
    elapsed_in_frame = 0
    frame_termination_observations: list[np.ndarray] = []
    frame_termination_elapsed_slots: list[int] = []
    frame_termination_actions: list[int] = []
    frame_old_termination_log_probs: list[float] = []
    frame_old_termination_entropies: list[float] = []
    termination_decision_diagnostics: list[dict[str, Any]] = []
    termination_entropies: list[float] = []
    termination_stop_probabilities: list[float] = []
    termination_actions: list[int] = []
    forced_termination_max_holding_count = 0
    terminal_frame_count = 0
    frame_macro_index = -1
    keep_probabilities: list[float] = []
    update_group_probabilities: list[float] = []
    raw_candidate0_probabilities: list[float] = []
    legal_update_candidate_counts: list[int] = []
    cache_entropies: list[float] = []
    cache_candidate_entropies: list[float] = []
    unified_candidate0_probabilities: list[float] = []
    unified_candidate_entropies: list[float] = []
    unified_update_margins: list[float] = []
    unified_candidate_state_diagnostics: list[dict[str, Any]] = []
    update_candidate_selection_counts: dict[bytes, int] = {}
    deterministic_cache_selections: list[dict[str, Any]] = []
    deterministic_uav_selections: list[dict[str, Any]] = []
    cache_group_rule_violation_count = 0
    cache_update_inner_argmax_violation_count = 0
    cache_probability_sum_error_max = 0.0
    cache_probability_nonfinite_count = 0
    cache_probability_nan_count = 0
    cache_probability_inf_count = 0
    node_keep_probability_records: list[np.ndarray] = []
    node_update_probability_records: list[np.ndarray] = []
    node_raw_keep_probability_records: list[np.ndarray] = []
    node_entropy_records: list[np.ndarray] = []
    node_log_prob_records: list[np.ndarray] = []
    node_selected_update_records: list[np.ndarray] = []
    node_actual_update_records: list[np.ndarray] = []
    update_nodes_per_macro: list[int] = []
    start_time = time.perf_counter()
    matching_kkt_elapsed_seconds = 0.0

    for slot_index, step in enumerate(aligned_steps[:rollout_slots]):
        slot_started = time.perf_counter()
        previous_uav = {key: value.copy() for key, value in uav.items()}
        if slot_index == frame_start:
            frame_observation = _observation(cache, uav, step, context)
            multi_node = (
                context.cache_action_mode
                != interface.project_config.CACHE_ACTION_MODE_SINGLE_NODE
            )
            sparse_cache = (
                context.cache_action_mode
                == interface.project_config.CACHE_ACTION_MODE_SPARSE_COORDINATED
            )
            if multi_node:
                node_candidates, frame_candidate_features, frame_candidate_masks = (
                    _cache_node_candidates(
                        cache,
                        uav,
                        step,
                        context,
                        tensor_spec,
                    )
                )
                if sparse_cache:
                    if not isinstance(policy, SparseCoordinatedCacheActorCritic):
                        raise TypeError("sparse Cache configuration requires its versioned policy")
                    selected_nodes = policy.sample_sparse_cache(
                        frame_observation,
                        frame_candidate_features,
                        frame_candidate_masks,
                        deterministic=deterministic,
                    )
                    if np.count_nonzero(selected_nodes.cache_actions) > 1:
                        raise AssertionError("sparse Cache sampler selected multiple UPDATE nodes")
                    frame_old_node_cache_log_probs = None
                    frame_old_node_cache_entropies = None
                    frame_old_cache_log_prob = float(
                        selected_nodes.cache_log_prob.detach().cpu()
                    )
                    frame_old_macro_entropy = float(
                        selected_nodes.cache_entropy.detach().cpu()
                    )
                    node_p_keep = np.asarray(
                        selected_nodes.node_p_keep.detach().cpu(), dtype=np.float64
                    )
                    node_p_update = np.asarray(
                        selected_nodes.node_p_update.detach().cpu(), dtype=np.float64
                    )
                    node_entropies = np.asarray(
                        selected_nodes.node_marginal_entropies.detach().cpu(),
                        dtype=np.float64,
                    )
                    node_logs = None
                    node_raw_keep = None
                    group_probabilities = selected_nodes.group_probabilities
                    keep_probability = float(group_probabilities[0])
                    update_group_probability = float(np.sum(group_probabilities[1:]))
                    candidate_entropy = frame_old_macro_entropy
                    raw_probabilities = None
                else:
                    selected_nodes = policy.sample_macro_nodes(
                        frame_observation,
                        frame_candidate_features,
                        frame_candidate_masks,
                        deterministic=deterministic,
                    )
                    frame_old_node_cache_log_probs = np.asarray(
                        selected_nodes.node_log_probs.detach().cpu(), dtype=np.float32
                    ).copy()
                    frame_old_node_cache_entropies = np.asarray(
                        selected_nodes.node_entropies.detach().cpu(), dtype=np.float32
                    ).copy()
                    cache_log_prob = _sum_log_probabilities(
                        selected_nodes.node_log_probs.unsqueeze(0), dim=1
                    )[0]
                    frame_old_cache_log_prob = float(cache_log_prob.detach().cpu())
                    frame_old_macro_entropy = float(
                        np.sum(frame_old_node_cache_entropies, dtype=np.float64)
                    )
                    node_entropies = frame_old_node_cache_entropies.astype(np.float64)
                    node_p_keep = np.asarray(
                        selected_nodes.node_p_keep.detach().cpu(), dtype=np.float64
                    )
                    node_p_update = np.asarray(
                        selected_nodes.node_p_update.detach().cpu(), dtype=np.float64
                    )
                    node_logs = frame_old_node_cache_log_probs.astype(np.float64)
                    raw_probabilities = np.asarray(
                        selected_nodes.raw_node_probabilities, dtype=np.float64
                    )
                    node_raw_keep = raw_probabilities[:, 0]
                    keep_probability = float(
                        np.mean(selected_nodes.node_p_keep.detach().cpu().numpy())
                    )
                    update_group_probability = float(
                        np.mean(selected_nodes.node_p_update.detach().cpu().numpy())
                    )
                    candidate_entropy = float(np.sum(node_entropies))
                frame_cache_actions = np.asarray(
                    selected_nodes.cache_actions, dtype=np.int64
                ).copy()
                frame_cache_index = frame_cache_actions.copy()
                frame_old_value = float(selected_nodes.value.detach().cpu())
                probabilities = np.asarray(
                    selected_nodes.node_probabilities, dtype=np.float64
                )
                if (
                    probabilities.shape != frame_candidate_masks.shape
                    or not np.isfinite(probabilities).all()
                    or not np.allclose(probabilities.sum(axis=-1), 1.0, atol=1.0e-6, rtol=0.0)
                    or np.any(probabilities[~frame_candidate_masks] != 0.0)
                ):
                    raise AssertionError("multi-node Cache probabilities/masks are invalid")
                if not sparse_cache and (
                    raw_probabilities is None
                    or raw_probabilities.shape != frame_candidate_masks.shape
                    or not np.isfinite(raw_probabilities).all()
                    or not np.allclose(
                        raw_probabilities.sum(axis=-1), 1.0, atol=1.0e-6, rtol=0.0
                    )
                ):
                    raise AssertionError("factorized raw Cache probabilities are invalid")
                node_selected_updates = frame_cache_actions > 0
                target_cache = cache.copy()
                for node_index, candidate_index in enumerate(frame_cache_actions):
                    chosen = node_candidates[node_index][int(candidate_index)]
                    if candidate_index == 0 and not np.array_equal(chosen, cache):
                        raise AssertionError("node KEEP candidate changed the global Cache")
                    target_cache[node_index] = chosen[node_index]
                frame_cache = target_cache
                frame_actual_update_by_node = cache_update_mask(cache, frame_cache)
                if np.any(frame_actual_update_by_node & ~node_selected_updates):
                    raise AssertionError("Cache changed on a node whose sampled action was KEEP")
                if sparse_cache:
                    if int(np.count_nonzero(frame_actual_update_by_node)) > 1:
                        raise AssertionError("sparse Cache execution changed more than one node")
                    if selected_nodes.group_action == 0 and not np.array_equal(frame_cache, cache):
                        raise AssertionError("ALL_KEEP changed the Cache configuration")
                    cache_configuration.validate_cache_shape_and_capacity(
                        frame_cache, context.service_catalog, context.cache_nodes
                    )
                node_keep_probability_records.append(node_p_keep.copy())
                node_update_probability_records.append(node_p_update.copy())
                node_entropy_records.append(node_entropies.copy())
                if node_raw_keep is not None:
                    node_raw_keep_probability_records.append(node_raw_keep.copy())
                if node_logs is not None:
                    node_log_prob_records.append(node_logs.copy())
                node_selected_update_records.append(node_selected_updates.copy())
                node_actual_update_records.append(frame_actual_update_by_node.copy())
                keep_probabilities.append(keep_probability)
                update_group_probabilities.append(update_group_probability)
                if node_raw_keep is not None:
                    raw_candidate0_probabilities.append(float(np.mean(node_raw_keep)))
                legal_update_candidate_counts.append(
                    int(np.count_nonzero(frame_candidate_masks[:, 1:]))
                )
                cache_entropies.append(frame_old_macro_entropy)
                cache_candidate_entropies.append(candidate_entropy)
                unified_candidate_entropies.append(candidate_entropy)
                unified_candidate0_probabilities.append(keep_probability)
                update_margin = update_group_probability - keep_probability
                unified_update_margins.append(update_margin)
                state_diagnostic = {
                    "macro_index": int(len(macro_buffer.transitions)),
                    "selected_candidate_ids_by_node": frame_cache_actions.tolist(),
                    "mean_keep_probability_by_node": float(np.mean(node_p_keep)),
                    "mean_update_probability_by_node": float(np.mean(node_p_update)),
                    "joint_cache_entropy_sum_by_node": candidate_entropy,
                    "mean_per_node_cache_entropy": float(np.mean(node_entropies)),
                    "update_margin_mean_by_node": update_margin,
                }
                if sparse_cache:
                    state_diagnostic.update(
                        {
                            "joint_cache_entropy_sum_by_node": None,
                            "group_keep_probability": keep_probability,
                            "group_update_probability": update_group_probability,
                            "cache_entropy": candidate_entropy,
                        }
                    )
                unified_candidate_state_diagnostics.append(state_diagnostic)
                if deterministic:
                    if sparse_cache:
                        deterministic_cache_selections.append(
                            {
                                "frame_index": int(len(deterministic_cache_selections)),
                                "decision_slot": int(step.slot),
                                "legal_candidate_count_by_node": frame_candidate_masks.sum(axis=-1).astype(int).tolist(),
                                "selected_candidate_index_by_node": frame_cache_actions.tolist(),
                                "selection": selected_nodes.deterministic_diagnostics,
                                "probability_finite": bool(np.isfinite(probabilities).all()),
                                "probability_sum_error_max": float(np.max(np.abs(probabilities.sum(axis=-1) - 1.0))),
                            }
                        )
                    else:
                        deterministic_node_records = (
                            []
                            if selected_nodes.deterministic_diagnostics is None
                            else list(selected_nodes.deterministic_diagnostics)
                        )
                        if len(deterministic_node_records) != len(context.action_spec.cache_node_ids):
                            raise AssertionError("deterministic factorized Cache selection lacks node diagnostics")
                        for record, node_id in zip(
                            deterministic_node_records,
                            context.action_spec.cache_node_ids,
                            strict=True,
                        ):
                            record["node_id"] = str(node_id)
                        deterministic_cache_selections.append(
                            {
                                "frame_index": int(len(deterministic_cache_selections)),
                                "decision_slot": int(step.slot),
                                "legal_candidate_count_by_node": frame_candidate_masks.sum(axis=-1).astype(int).tolist(),
                                "selected_candidate_index_by_node": frame_cache_actions.tolist(),
                                "selected_group_by_node": [item["selected_group"] for item in deterministic_node_records],
                                "nodes": deterministic_node_records,
                                "probability_finite": bool(np.isfinite(probabilities).all()),
                                "probability_sum_error_max": float(np.max(np.abs(probabilities.sum(axis=-1) - 1.0))),
                            }
                        )
                frame_cache_changed = bool(np.any(frame_actual_update_by_node))
                frame_macro_index = len(macro_buffer.transitions)
                update_nodes_per_macro.append(int(np.count_nonzero(frame_actual_update_by_node)))
                cache_probability_sum_error_max = max(
                    cache_probability_sum_error_max,
                    float(np.max(np.abs(probabilities.sum(axis=-1) - 1.0))),
                )
                cache_probability_nan_count += int(np.isnan(probabilities).sum())
                cache_probability_inf_count += int(np.isinf(probabilities).sum())
                if frame_cache_changed:
                    candidate_signature = np.ascontiguousarray(
                        frame_cache, dtype=np.int8
                    ).tobytes()
                    update_candidate_selection_counts[candidate_signature] = (
                        update_candidate_selection_counts.get(candidate_signature, 0) + 1
                    )
            else:
                _candidates, frame_candidate_features = _cache_candidates(
                    cache,
                    uav,
                    step,
                    context,
                    tensor_spec,
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
                frame_cache_actions = np.asarray(frame_cache_index, dtype=np.int64)
                frame_old_node_cache_log_probs = None
                frame_old_node_cache_entropies = None
                frame_old_cache_log_prob = float(selected.log_prob.detach().cpu())
                frame_old_macro_entropy = float(selected.cache_entropy.detach().cpu())
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
                raw_cache_probabilities = np.asarray(
                    selected.raw_cache_probabilities, dtype=np.float64
                )
                if (
                    raw_cache_probabilities.shape != frame_candidate_masks.shape
                    or not np.isfinite(raw_cache_probabilities).all()
                    or not np.isclose(
                        float(raw_cache_probabilities.sum()),
                        1.0,
                        rtol=1.0e-6,
                        atol=1.0e-7,
                    )
                ):
                    raise AssertionError("Raw flat Cache Candidate probabilities are invalid.")
                raw_candidate0_probabilities.append(float(raw_cache_probabilities[0]))
                legal_update_candidate_counts.append(
                    int(np.count_nonzero(frame_candidate_masks[1:]))
                )
                cache_entropies.append(float(selected.cache_entropy.detach().cpu()))
                positive_candidate_probabilities = cache_probabilities[cache_probabilities > 0.0]
                candidate_entropy = float(
                    -np.sum(positive_candidate_probabilities * np.log(positive_candidate_probabilities))
                )
                best_nonzero_candidate_probability = (
                    float(np.max(cache_probabilities[1:]))
                    if bool(np.any(frame_candidate_masks[1:]))
                    else 0.0
                )
                update_margin = update_group_probability - keep_probability
                unified_candidate0_probabilities.append(keep_probability)
                unified_candidate_entropies.append(candidate_entropy)
                cache_candidate_entropies.append(candidate_entropy)
                unified_update_margins.append(update_margin)
                unified_candidate_state_diagnostics.append(
                    {
                        "macro_index": int(len(macro_buffer.transitions)),
                        "selected_candidate_id": int(frame_cache_index),
                        "candidate0_probability": keep_probability,
                        "best_nonzero_candidate_probability": best_nonzero_candidate_probability,
                        "candidate_entropy": candidate_entropy,
                        "update_margin": float(update_margin),
                    }
                )
                frame_cache = _candidates[frame_cache_index].copy()
                frame_actual_update_by_node = cache_update_mask(cache, frame_cache)
                frame_cache_changed = bool(np.any(frame_actual_update_by_node))
                frame_macro_index = len(macro_buffer.transitions)
                if deterministic:
                    selected_cache_id = hashlib.sha256(
                        np.ascontiguousarray(frame_cache, dtype=np.int8).tobytes()
                    ).hexdigest()[:16]
                    selection_record = {
                        "frame_index": int(len(deterministic_cache_selections)),
                        "decision_slot": int(step.slot),
                        "legal_candidate_count": int(np.count_nonzero(frame_candidate_masks)),
                        "selected_group": str(selected.cache_selected_group),
                        "selected_candidate_index": frame_cache_index,
                        "selected_candidate_id": selected_cache_id,
                        "probability_finite": bool(np.isfinite(cache_probabilities).all()),
                        "probability_nan_count": int(np.isnan(cache_probabilities).sum()),
                        "probability_inf_count": int(np.isinf(cache_probabilities).sum()),
                    }
                    protocol_diagnostic = selected.deterministic_cache_diagnostics
                    if protocol_diagnostic is None:
                        raise AssertionError(
                            "Deterministic corrected Cache frame lacks protocol diagnostics."
                        )
                    p_keep = float(protocol_diagnostic["p_keep"])
                    p_update_group = float(protocol_diagnostic["p_update_group"])
                    expected_group = "KEEP" if p_keep >= p_update_group else "UPDATE"
                    if str(protocol_diagnostic["new_selected_group"]) != expected_group:
                        cache_group_rule_violation_count += 1
                    if str(protocol_diagnostic["new_selected_group"]) == "UPDATE" and bool(
                        protocol_diagnostic["update_inner_argmax_violation"]
                    ):
                        cache_update_inner_argmax_violation_count += 1
                    probability_sum_error = float(protocol_diagnostic["probability_sum_error"])
                    cache_probability_sum_error_max = max(
                        cache_probability_sum_error_max, probability_sum_error
                    )
                    if not np.isfinite([p_keep, p_update_group, probability_sum_error]).all():
                        cache_probability_nonfinite_count += 1
                    cache_probability_nan_count += int(np.isnan(cache_probabilities).sum())
                    cache_probability_inf_count += int(np.isinf(cache_probabilities).sum())
                    selection_record["selected_update_raw_score_best"] = (
                        None
                        if str(protocol_diagnostic["new_selected_group"]) != "UPDATE"
                        else bool(
                            protocol_diagnostic["selected_candidate_is_legal"]
                            and protocol_diagnostic["selected_raw_score"]
                            == protocol_diagnostic["legal_update_best_raw_score"]
                        )
                    )
                    selection_record.update(protocol_diagnostic)
                    deterministic_cache_selections.append(selection_record)
                if frame_cache_changed:
                    candidate_signature = np.ascontiguousarray(
                        frame_cache, dtype=np.int8
                    ).tobytes()
                    update_candidate_selection_counts[candidate_signature] = (
                        update_candidate_selection_counts.get(candidate_signature, 0) + 1
                    )
                update_nodes_per_macro.append(int(np.count_nonzero(frame_actual_update_by_node)))
            frame_termination_observations = []
            frame_termination_elapsed_slots = []
            frame_termination_actions = []
            frame_old_termination_log_probs = []
            frame_old_termination_entropies = []
            elapsed_in_frame = 0
            frame_reward = 0.0

        if (
            frame_observation is None
            or frame_candidate_features is None
            or frame_candidate_masks is None
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
        condition_dim = int(policy.micro_condition_dim)
        selected_uav_conditions = np.zeros((uav_count, condition_dim), dtype=np.float32)
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
            condition = (
                np.concatenate(
                    [
                        _positions_condition(current_trial[str(previous_id)], context)
                        for previous_id in context.action_spec.uav_ids[:uav_index]
                    ]
                ).astype(np.float32, copy=False)
                if uav_index > 0
                else np.zeros(condition_dim, dtype=np.float32)
            )
            if condition.shape != (condition_dim,):
                raise ValueError("Autoregressive UAV condition does not match policy schema.")
            sample = policy.sample_micro_head(
                observation,
                candidate_features,
                np.ones(len(candidate_features), dtype=bool),
                uav_index=uav_index,
                prior_uav_position_condition=condition,
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
            selected_uav_conditions[uav_index] = condition
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
        joint_log_prob = _sum_log_probabilities(
            torch.stack(old_head_log_probs).reshape(1, -1), dim=1
        )[0]
        joint_entropy = torch.stack(old_head_entropies).sum()
        probability_snapshot = np.zeros((uav_count, max_uav_candidates), dtype=np.float32)
        for head_index in range(uav_count):
            probability_snapshot[head_index, :valid_counts[head_index]] = candidate_probability_lists[head_index]
        cache_id = hashlib.sha256(
            np.ascontiguousarray(frame_cache, dtype=np.int8).tobytes()
        ).hexdigest()[:16]
        actual_cache_update_count = int(np.count_nonzero(cache_update_mask(cache, frame_cache)))
        performance_profile: dict[str, Any] = {}
        exact_q_timing_profile = (
            {}
            if sampled_slot_indices is not None and slot_index in sampled_slot_indices
            else None
        )
        environment_started = time.perf_counter()
        result = slot_environment.execute_configured_slot(
            cache,
            previous_uav,
            frame_cache,
            current_trial,
            step.scenario_slice,
            step.actual_tasks,
            step.actual_vehicle_positions,
            context.service_catalog,
            context.cfg,
            context.cache_nodes,
            context.cache_cfg,
            context.flight_boundary,
            slot_id=int(step.slot),
            time_s=float(step.time_s),
            performance_profile=performance_profile,
            exact_q_timing_profile=exact_q_timing_profile,
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
        if exact_q_timing_profile is not None and performance_sample_callback is not None:
            matching_summary = result.matching_summary
            performance_sample_callback(
                {
                    "slot_index": int(slot_index),
                    "slot_id": int(step.slot),
                    "task_count": int(metrics.task_count),
                    "task_utility": float(result.actual_task_utility),
                    "system_utility": float(result.system_utility),
                    "cache_cost": float(result.cache_cost),
                    "movement_cost": float(result.movement_cost),
                    "matching": dict(result.matching),
                    "matching_seconds": float(
                        performance_profile.get("matching_kkt_seconds", 0.0)
                    ),
                    "matching_diagnostics": {
                        key: matching_summary.get(key)
                        for key in (
                            "resource_feasible",
                            "resource",
                            "node_task_count",
                            "updates",
                            "transfer_updates",
                            "exchange_updates",
                            "candidate_definitions_total",
                            "candidate_evaluations_total",
                            "upper_bound_screen_calls",
                            "upper_bound_pruned_candidates",
                            "full_evaluator_calls",
                            "KKT_SOLVER_CALL_COUNT",
                            "KKT_CACHE_REQUEST_COUNT",
                            "KKT_CACHE_HIT_COUNT",
                            "NODE_EVAL_CACHE_REQUEST_COUNT",
                            "NODE_EVAL_CACHE_HIT_COUNT",
                            "KKT_SOLVE_COUNT",
                            "KKT_CONVERGED_COUNT",
                            "KKT_MAX_ITER_HIT_COUNT",
                            "KKT_FINITE_COUNT",
                            "KKT_ITERATION_MEAN",
                            "KKT_ITERATION_P95",
                            "KKT_ITERATION_MAX",
                            "stop_reason",
                            "transfer_exchange_stable",
                        )
                    },
                    "timing_profile_s": {
                        key: float(exact_q_timing_profile.get(key, 0.0))
                        for key in (
                            "matching_total_runtime",
                            "matching_other_runtime",
                            "matching_candidate_generation_runtime",
                            "matching_upper_bound_runtime",
                            "matching_ideal_utility_runtime",
                            "matching_exact_eval_orchestration_runtime",
                            "kkt_runtime",
                            "kkt_solver_core_runtime",
                            "kkt_edge_cpu_solver_core_runtime",
                            "kkt_edge_bandwidth_solver_core_runtime",
                            "kkt_cloud_cpu_solver_core_runtime",
                            "node_eval_miss_runtime",
                        )
                    },
                }
            )
        done = slot_index == total_slots - 1
        truncated = window_truncated and slot_index == rollout_slots - 1
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
                uav_candidate_sources=tuple(candidate_source_lists),
                uav_ids=tuple(str(value) for value in context.action_spec.uav_ids),
                uav_position_conditions=selected_uav_conditions.copy(),
                active_cache_configuration_id=cache_id,
                actual_task_utility=float(result.actual_task_utility),
                actual_cache_update_count=actual_cache_update_count,
                uav_movement_count=int(sum(action_index != 0 for action_index in actions)),
                cache_cost=float(result.cache_cost),
                cache_cost_by_node={
                    str(node_id): float(
                        result.configuration_summary.get("cache_cost_by_node", {}).get(
                            str(node_id), 0.0
                        )
                    )
                    for node_id in context.action_spec.cache_node_ids
                },
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

        end_frame = bool(done)
        if done:
            terminal_frame_count += 1
        elif truncated:
            # A short frozen-evaluation window is a time-limit truncation, not
            # an Episode terminal. Close its final Macro frame while retaining
            # the next observed state for value bootstrapping.
            end_frame = True
        elif elapsed_in_frame >= max_holding_slots:
            forced_termination_max_holding_count += 1
            end_frame = True
        elif elapsed_in_frame >= min_holding_slots:
            termination_sample = policy.sample_termination(
                next_observation,
                elapsed_in_frame,
                deterministic=deterministic,
            )
            termination_action = int(termination_sample.selected_index)
            termination_log_prob = float(termination_sample.log_prob.detach().cpu())
            termination_entropy = float(termination_sample.entropy.detach().cpu())
            sampled_termination_probabilities = np.asarray(
                termination_sample.probabilities, dtype=np.float64
            ).reshape(-1)
            if sampled_termination_probabilities.shape != (2,) or not np.isfinite(
                sampled_termination_probabilities
            ).all():
                raise FloatingPointError("Termination probabilities are invalid.")
            if not np.isclose(
                float(sampled_termination_probabilities.sum()), 1.0, rtol=1.0e-6, atol=1.0e-7
            ):
                raise AssertionError("Termination probabilities must sum to one.")
            terminate_probability = float(sampled_termination_probabilities[1])
            frame_termination_observations.append(next_observation.copy())
            frame_termination_elapsed_slots.append(int(elapsed_in_frame))
            frame_termination_actions.append(termination_action)
            frame_old_termination_log_probs.append(termination_log_prob)
            frame_old_termination_entropies.append(termination_entropy)
            frame_old_macro_entropy += termination_entropy
            termination_stop_probabilities.append(terminate_probability)
            termination_entropies.append(termination_entropy)
            termination_actions.append(termination_action)
            termination_decision_diagnostics.append(
                {
                    "elapsed_holding_slots": int(elapsed_in_frame),
                    "elapsed_holding_fraction": float(
                        elapsed_in_frame / max_holding_slots
                    ),
                    "termination_probability": terminate_probability,
                    "selected_termination_action": (
                        "TERMINATE" if termination_action == 1 else "CONTINUE"
                    ),
                    "selected_termination_action_index": termination_action,
                }
            )
            end_frame = termination_action == 1

        if end_frame:
            policy_device = next(policy.parameters()).device
            cache_log_prob_tensor = torch.as_tensor(
                [frame_old_cache_log_prob], dtype=torch.float32, device=policy_device
            )
            termination_log_prob_tensor = torch.as_tensor(
                frame_old_termination_log_probs,
                dtype=cache_log_prob_tensor.dtype,
                device=policy_device,
            ).unsqueeze(0)
            _, joint_macro_log_prob = _macro_log_prob_from_components(
                cache_log_prob_tensor,
                termination_log_prob_tensor,
                dim=1,
            )
            frame_old_macro_log_prob = float(joint_macro_log_prob[0].detach().cpu())
            next_macro_state = np.zeros_like(frame_observation) if done else next_observation
            if done:
                frame_old_next_value = 0.0
            else:
                if torch is None:
                    raise ImportError("Formal Macro rollout requires PyTorch.")
                with torch.no_grad():
                    frame_old_next_value = float(
                        policy.macro_value(next_macro_state).detach().cpu()
                    )
            macro_buffer.append(
                MacroTransition(
                    observation=frame_observation.copy(),
                    cache_candidate_features=frame_candidate_features.copy(),
                    cache_candidate_masks=frame_candidate_masks.copy(),
                    cache_action=np.asarray(frame_cache_actions, dtype=np.int64).copy(),
                    actual_duration=elapsed_in_frame,
                    old_cache_log_prob=frame_old_cache_log_prob,
                    old_node_cache_log_probs=(
                        None
                        if frame_old_node_cache_log_probs is None
                        else frame_old_node_cache_log_probs.copy()
                    ),
                    old_node_cache_entropies=(
                        None
                        if frame_old_node_cache_entropies is None
                        else frame_old_node_cache_entropies.copy()
                    ),
                    old_macro_log_prob=frame_old_macro_log_prob,
                    old_macro_entropy=frame_old_macro_entropy,
                    holding_transitions=tuple(
                        HoldingTransition(
                            observation=np.asarray(observation_value, dtype=np.float32).copy(),
                            elapsed_slots=int(elapsed_value),
                            action=int(action_value),
                            old_log_prob=float(log_prob_value),
                            old_entropy=float(entropy_value),
                        )
                        for observation_value, elapsed_value, action_value, log_prob_value, entropy_value in zip(
                            frame_termination_observations,
                            frame_termination_elapsed_slots,
                            frame_termination_actions,
                            frame_old_termination_log_probs,
                            frame_old_termination_entropies,
                            strict=True,
                        )
                    ),
                    old_value=frame_old_value,
                    old_next_value=frame_old_next_value,
                    frame_reward=float(frame_reward),
                    next_observation=next_macro_state.copy(),
                    done=done,
                    cache_configuration_id=cache_id,
                    actual_cache_update=frame_cache_changed,
                    actual_cache_update_by_node=frame_actual_update_by_node.astype(bool, copy=True),
                )
            )
            frame_start = slot_index + 1
        if progress_callback is not None:
            matching_kkt_elapsed_seconds += float(
                performance_profile.get("matching_kkt_seconds", 0.0)
            )
            slots_completed = slot_index + 1
            if slots_completed % 30 == 0 or slots_completed == rollout_slots:
                progress_callback(
                    slots_completed,
                    rollout_slots,
                    time.perf_counter() - start_time,
                    matching_kkt_elapsed_seconds,
                )
    if not micro_buffer.transitions or (
        not micro_buffer.transitions[-1].done and not window_truncated
    ):
        raise AssertionError("formal rollout must end at an Episode terminal or evaluation truncation")
    if not macro_buffer.transitions or (
        not macro_buffer.transitions[-1].done and not window_truncated
    ):
        raise AssertionError("Macro buffer must end at an Episode terminal or evaluation truncation")

    actual_durations = [int(item.actual_duration) for item in macro_buffer.transitions]
    holding_duration_distribution = {
        str(value): int(actual_durations.count(value))
        for value in sorted(set(actual_durations))
    }
    macro_count = len(macro_buffer)
    micro_count = len(micro_buffer)
    stay_counts = np.sum(
        np.stack([item.actions == 0 for item in micro_buffer.transitions]), axis=0
    )
    matching_summaries = [item.matching_summary for item in micro_buffer.transitions]
    episode_node_task_counts = _aggregate_node_task_counts(micro_buffer.transitions)
    episode_return = float(sum(item.reward for item in micro_buffer.transitions))
    factorized_cache = np.asarray(
        macro_buffer.transitions[0].cache_candidate_masks
    ).ndim == 2
    sparse_cache_mode = (
        context.cache_action_mode
        == interface.project_config.CACHE_ACTION_MODE_SPARSE_COORDINATED
    )
    if factorized_cache:
        node_count = int(macro_buffer.transitions[0].cache_candidate_masks.shape[0])
        action_matrix = np.stack(
            [np.asarray(item.cache_action, dtype=np.int64) for item in macro_buffer.transitions]
        )
        actual_update_matrix = np.stack(
            [np.asarray(item.actual_cache_update_by_node, dtype=bool) for item in macro_buffer.transitions]
        )
        if action_matrix.shape != (macro_count, node_count) or actual_update_matrix.shape != action_matrix.shape:
            raise ValueError("factorized Cache action and actual update summaries have invalid shape")
        frames_with_updates = np.any(actual_update_matrix, axis=1)
        keep_count = int(np.count_nonzero(~frames_with_updates))
        update_count = int(np.count_nonzero(frames_with_updates))
        node_selected_update_counts = np.count_nonzero(action_matrix > 0, axis=0)
        node_actual_update_counts = np.count_nonzero(actual_update_matrix, axis=0)
        node_cost_totals = {
            str(node_id): float(
                sum(item.cache_cost_by_node.get(str(node_id), 0.0) for item in micro_buffer.transitions)
            )
            for node_id in context.action_spec.cache_node_ids
        }
        per_node_cache_diagnostics = []
        if node_keep_probability_records:
            keep_probabilities_array = np.stack(node_keep_probability_records)
            update_probabilities_array = np.stack(node_update_probability_records)
            entropy_array = np.stack(node_entropy_records)
            raw_keep_probabilities_array = (
                np.stack(node_raw_keep_probability_records)
                if len(node_raw_keep_probability_records) == macro_count
                else None
            )
            log_prob_array = (
                np.stack(node_log_prob_records)
                if len(node_log_prob_records) == macro_count
                else None
            )
            for node_index, node_id in enumerate(context.action_spec.cache_node_ids):
                node_diagnostic = {
                    "node_index": int(node_index),
                    "node_id": str(node_id),
                    "keep_probability_mean": float(np.mean(keep_probabilities_array[:, node_index])),
                    "update_probability_mean": float(np.mean(update_probabilities_array[:, node_index])),
                    "raw_keep_probability_mean": (
                        None
                        if raw_keep_probabilities_array is None
                        else float(np.mean(raw_keep_probabilities_array[:, node_index]))
                    ),
                    "sampled_update_frequency": float(node_selected_update_counts[node_index] / max(macro_count, 1)),
                    "actual_update_frequency": float(node_actual_update_counts[node_index] / max(macro_count, 1)),
                    "candidate_entropy_mean": float(np.mean(entropy_array[:, node_index])),
                    "old_log_prob_mean": (
                        None
                        if log_prob_array is None
                        else float(np.mean(log_prob_array[:, node_index]))
                    ),
                    "cache_cost_total": node_cost_totals[str(node_id)],
                }
                if sparse_cache_mode:
                    node_diagnostic["probability_semantics"] = (
                        "sparse node marginal; not an independent policy head"
                    )
                per_node_cache_diagnostics.append(node_diagnostic)
        else:
            per_node_cache_diagnostics = []
        update_nodes_per_macro = np.count_nonzero(actual_update_matrix, axis=1).astype(int).tolist()
    else:
        action_matrix = np.asarray([int(item.cache_action) for item in macro_buffer.transitions])
        keep_count = int(np.count_nonzero(action_matrix == 0))
        update_count = int(macro_count - keep_count)
        node_selected_update_counts = np.asarray([], dtype=np.int64)
        node_actual_update_counts = np.asarray([], dtype=np.int64)
        node_cost_totals = {
            str(node_id): float(
                sum(item.cache_cost_by_node.get(str(node_id), 0.0) for item in micro_buffer.transitions)
            )
            for node_id in context.action_spec.cache_node_ids
        }
        per_node_cache_diagnostics = []
    nonzero_candidate_transitions = [
        item
        for item in macro_buffer.transitions
        if (
            bool(np.any(np.asarray(item.cache_action) > 0))
            if factorized_cache
            else int(item.cache_action) != 0
        )
    ]
    nonzero_candidate_no_effect_rate = (
        float(sum(not item.actual_cache_update for item in nonzero_candidate_transitions)
              / len(nonzero_candidate_transitions))
        if nonzero_candidate_transitions
        else 0.0
    )
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
    if keep_count == 0:
        cache_action_pattern = "ALL_UPDATE"
    elif update_count == 0:
        cache_action_pattern = "ALL_KEEP"
    else:
        cache_action_pattern = "NORMAL"
    uav_action_patterns = []
    for head_index in range(len(context.action_spec.uav_ids)):
        stay_count = int(stay_counts[head_index])
        move_count = int(micro_count - stay_count)
        uav_action_patterns.append(
            "NORMAL" if stay_count and move_count else "ALL_STAY" if stay_count else "ALL_MOVE"
        )
    termination_sample_count = len(termination_actions)
    termination_continue_count = int(sum(action == 0 for action in termination_actions))
    termination_stop_count = int(sum(action == 1 for action in termination_actions))
    termination_probability_by_elapsed: dict[str, dict[str, Any]] = {}
    for elapsed in range(min_holding_slots, max_holding_slots):
        elapsed_records = [
            record
            for record in termination_decision_diagnostics
            if int(record["elapsed_holding_slots"]) == elapsed
        ]
        termination_probability_by_elapsed[str(elapsed)] = {
            "sample_count": len(elapsed_records),
            "mean_termination_probability": (
                float(np.mean([record["termination_probability"] for record in elapsed_records]))
                if elapsed_records
                else None
            ),
            "selected_termination_rate": (
                float(
                    np.mean(
                        [
                            int(record["selected_termination_action_index"]) == 1
                            for record in elapsed_records
                        ]
                    )
                )
                if elapsed_records
                else None
            ),
        }
    update_candidate_selection_entropy = 0.0
    if update_count:
        for selected_count in update_candidate_selection_counts.values():
            selection_probability = selected_count / update_count
            update_candidate_selection_entropy -= selection_probability * float(
                np.log(selection_probability)
            )
    uav_count = len(context.action_spec.uav_ids)
    micro_group_probabilities = np.zeros((micro_count, uav_count, 2), dtype=np.float64)
    micro_group_entropies = np.zeros((micro_count, uav_count), dtype=np.float64)
    for transition_index, transition in enumerate(micro_buffer.transitions):
        for head_index in range(uav_count):
            candidate_count = int(transition.valid_candidate_count[head_index])
            candidate_probabilities = np.asarray(
                transition.uav_candidate_probabilities[head_index, :candidate_count],
                dtype=np.float64,
            )
            if candidate_count <= 0 or not np.isfinite(candidate_probabilities).all():
                raise ValueError("Micro 候选概率快照必须为有限且非空的分布。")
            p_stay = float(candidate_probabilities[0])
            p_move = float(np.sum(candidate_probabilities[1:], dtype=np.float64))
            if not math.isclose(p_stay + p_move, 1.0, rel_tol=0.0, abs_tol=5.0e-6):
                raise ValueError("Micro STAY/MOVE 概率质量必须和为 1。")
            micro_group_probabilities[transition_index, head_index] = (
                p_stay,
                p_move,
            )
            terms = [
                -probability * math.log(probability)
                for probability in (p_stay, p_move)
                if probability > 0.0
            ]
            micro_group_entropies[transition_index, head_index] = sum(terms)
    micro_group_probability_soft_cap = float(
        context.cfg["algorithm"]["ppo"]["micro_group_probability_soft_cap"]
    )
    micro_group_entropy_floor = (
        -micro_group_probability_soft_cap * math.log(micro_group_probability_soft_cap)
        - (1.0 - micro_group_probability_soft_cap)
        * math.log(1.0 - micro_group_probability_soft_cap)
    )
    group_probability_max = np.max(micro_group_probabilities, axis=-1)
    summary = {
        "episode_return": episode_return,
        "window_slot_count": int(micro_count),
        "window_truncated": bool(window_truncated),
        "episode_terminal_observed": bool(not window_truncated),
        "task_execution_model": interface.PER_SLOT_ANALYTIC_TASK_MODEL,
        "cache_action_mode": str(context.cache_action_mode),
        "episode_seed": None if episode_seed is None else int(episode_seed),
        "macro_return": float(sum(item.frame_reward for item in macro_buffer.transitions)),
        "macro_frame_count": macro_count,
        "macro_entropy": float(np.mean([item.old_macro_entropy for item in macro_buffer.transitions])),
        "macro_cache_entropy": float(np.mean(cache_entropies)) if cache_entropies else 0.0,
        "macro_cache_entropy_aggregation": (
            "H(group) + sum_i P(group=i) H(candidate|group=i)"
            if sparse_cache_mode
            else "sum of independent per-node categorical entropies; its scale is not directly comparable to single-node entropy"
            if factorized_cache
            else "single unified Cache categorical entropy"
        ),
        "macro_cache_entropy_joint_sum_mean": (
            float(np.mean(cache_entropies))
            if sparse_cache_mode
            else float(np.mean([np.sum(item.old_node_cache_entropies) for item in macro_buffer.transitions]))
            if factorized_cache
            else (float(np.mean(cache_entropies)) if cache_entropies else 0.0)
        ),
        "macro_cache_entropy_per_node_mean": (
            float(np.mean(np.stack(node_entropy_records)))
            if node_entropy_records
            else None
        ),
        "adaptive_holding_duration_mean": float(np.mean(actual_durations)),
        "adaptive_holding_duration_std": float(np.std(actual_durations)),
        "holding_duration_distribution": holding_duration_distribution,
        "termination_sample_count": termination_sample_count,
        "termination_eligible_count": termination_sample_count,
        "termination_continue_count": termination_continue_count,
        "termination_stop_count": termination_stop_count,
        "termination_continue_rate": (
            float(termination_continue_count / termination_sample_count)
            if termination_sample_count
            else 0.0
        ),
        "termination_stop_rate": (
            float(termination_stop_count / termination_sample_count)
            if termination_sample_count
            else 0.0
        ),
        "termination_entropy_mean": (
            float(np.mean(termination_entropies)) if termination_entropies else 0.0
        ),
        "termination_probability_mean": (
            float(np.mean(termination_stop_probabilities))
            if termination_stop_probabilities
            else 0.0
        ),
        "termination_probability_by_elapsed": termination_probability_by_elapsed,
        "termination_decision_diagnostics": termination_decision_diagnostics,
        "forced_termination_max_holding_count": int(
            forced_termination_max_holding_count
        ),
        "terminal_frame_count": int(terminal_frame_count),
        "keep_count": keep_count,
        "update_count": update_count,
        "update_selected_candidate_count": update_count,
        "keep_rate": float(keep_count / macro_count) if macro_count else 0.0,
        "update_rate": float(update_count / macro_count) if macro_count else 0.0,
        "cache_node_ids": [str(value) for value in context.action_spec.cache_node_ids],
        "cache_node_sampled_update_counts": (
            node_selected_update_counts.astype(int).tolist()
            if factorized_cache
            else None
        ),
        "cache_node_actual_update_counts": (
            node_actual_update_counts.astype(int).tolist()
            if factorized_cache
            else None
        ),
        "cache_node_sampled_update_frequency": (
            (node_selected_update_counts / max(macro_count, 1)).astype(float).tolist()
            if factorized_cache
            else None
        ),
        "cache_node_actual_update_frequency": (
            (node_actual_update_counts / max(macro_count, 1)).astype(float).tolist()
            if factorized_cache
            else None
        ),
        "cache_node_cost_total": node_cost_totals,
        "cache_node_diagnostics": per_node_cache_diagnostics,
        "updated_nodes_per_macro_distribution": {
            str(count): int(update_nodes_per_macro.count(count))
            for count in range(len(context.action_spec.cache_node_ids) + 1)
        },
        "mean_keep_probability": float(np.mean(keep_probabilities)) if keep_probabilities else 0.0,
        "mean_update_group_probability": (
            float(np.mean(update_group_probabilities)) if update_group_probabilities else 0.0
        ),
        "candidate0_probability_mean": (
            float(np.mean(unified_candidate0_probabilities))
            if unified_candidate0_probabilities
            else 0.0
        ),
        "corrected_candidate0_probability_mean": (
            float(np.mean(unified_candidate0_probabilities))
            if unified_candidate0_probabilities
            else None
        ),
        "raw_candidate0_probability_mean": (
            float(np.mean(raw_candidate0_probabilities))
            if raw_candidate0_probabilities
            else None
        ),
        "corrected_update_group_probability_mean": (
            float(np.mean(update_group_probabilities))
            if update_group_probabilities
            else None
        ),
        "legal_update_candidate_count_mean": (
            float(np.mean(legal_update_candidate_counts))
            if legal_update_candidate_counts
            else None
        ),
        "legal_update_candidate_count_min": (
            int(min(legal_update_candidate_counts))
            if legal_update_candidate_counts
            else None
        ),
        "legal_update_candidate_count_max": (
            int(max(legal_update_candidate_counts))
            if legal_update_candidate_counts
            else None
        ),
        "update_candidate_probability_sum_mean": (
            float(np.mean(update_group_probabilities)) if update_group_probabilities else 0.0
        ),
        "candidate_entropy_mean": (
            float(np.mean(unified_candidate_entropies))
            if unified_candidate_entropies
            else 0.0
        ),
        "candidate0_probability_state_std": (
            float(np.std(unified_candidate0_probabilities))
            if unified_candidate0_probabilities
            else 0.0
        ),
        "update_margin_mean": (
            float(np.mean(unified_update_margins)) if unified_update_margins else 0.0
        ),
        "update_margin_std": (
            float(np.std(unified_update_margins)) if unified_update_margins else 0.0
        ),
        "update_margin_positive_fraction": (
            float(np.mean(np.asarray(unified_update_margins) > 0.0))
            if unified_update_margins
            else 0.0
        ),
        "macro_candidate_state_diagnostics": unified_candidate_state_diagnostics,
        "update_candidate_unique_selected": update_candidate_unique_selected,
        "cache_candidate_entropy": (
            float(np.mean(cache_candidate_entropies))
            if cache_candidate_entropies
            else 0.0
        ),
        "update_candidate_selection_entropy": float(update_candidate_selection_entropy),
        "update_candidate_top1_fraction": update_candidate_top1_fraction,
        "update_candidate_pattern": update_candidate_pattern,
        "cache_action_pattern": cache_action_pattern,
        "actual_cache_update_rate": (
            float(np.mean([item.actual_cache_update for item in macro_buffer.transitions]))
            if macro_count
            else 0.0
        ),
        "actual_cache_configuration_change_rate": (
            float(np.mean([item.actual_cache_update for item in macro_buffer.transitions]))
            if macro_count
            else 0.0
        ),
        "nonzero_candidate_no_effect_rate": nonzero_candidate_no_effect_rate,
        "uav_count": int(uav_count),
        "uav_metrics": [
            {
                "uav_id": str(uav_id),
                "stay_rate": float(stay_counts[index] / max(micro_count, 1)),
                "move_rate": float(1.0 - stay_counts[index] / max(micro_count, 1)),
                "stay_count": int(stay_counts[index]),
                "move_count": int(micro_count - stay_counts[index]),
                "action_pattern": uav_action_patterns[index],
                "entropy": float(np.mean([item.old_uav_entropies[index] for item in micro_buffer.transitions])),
                "group_stay_probability_mean": float(np.mean(micro_group_probabilities[:, index, 0])),
                "group_move_probability_mean": float(np.mean(micro_group_probabilities[:, index, 1])),
                "group_entropy_mean": float(np.mean(micro_group_entropies[:, index])),
                "group_low_entropy_fraction": float(np.mean(micro_group_entropies[:, index] < micro_group_entropy_floor)),
                "group_probability_over_90_fraction": float(np.mean(group_probability_max[:, index] > micro_group_probability_soft_cap)),
                "mean_valid_candidates": float(np.mean([item.valid_candidate_count[index] for item in micro_buffer.transitions])),
            }
            for index, uav_id in enumerate(context.action_spec.uav_ids)
        ],
        "uav0_stay_rate": float(stay_counts[0] / max(micro_count, 1)),
        "uav0_stay_count": int(stay_counts[0]),
        "uav0_move_count": int(micro_count - stay_counts[0]),
        "uav1_stay_rate": float(stay_counts[1] / max(micro_count, 1)) if uav_count > 1 else None,
        "uav1_stay_count": int(stay_counts[1]) if uav_count > 1 else None,
        "uav1_move_count": int(micro_count - stay_counts[1]) if uav_count > 1 else None,
        "uav0_move_rate": float(1.0 - stay_counts[0] / max(micro_count, 1)),
        "uav1_move_rate": float(1.0 - stay_counts[1] / max(micro_count, 1)) if uav_count > 1 else None,
        "uav0_action_pattern": uav_action_patterns[0],
        "uav1_action_pattern": uav_action_patterns[1] if uav_count > 1 else None,
        "uav0_entropy": float(np.mean([item.old_uav_entropies[0] for item in micro_buffer.transitions])),
        "uav1_entropy": float(np.mean([item.old_uav_entropies[1] for item in micro_buffer.transitions])) if uav_count > 1 else None,
        "micro_candidate_entropy": float(np.mean([item.old_joint_entropy for item in micro_buffer.transitions])),
        "uav0_group_stay_probability_mean": float(
            np.mean(micro_group_probabilities[:, 0, 0])
        ),
        "uav0_group_move_probability_mean": float(
            np.mean(micro_group_probabilities[:, 0, 1])
        ),
        "uav0_group_entropy_mean": float(np.mean(micro_group_entropies[:, 0])),
        "uav1_group_stay_probability_mean": float(np.mean(micro_group_probabilities[:, 1, 0])) if uav_count > 1 else None,
        "uav1_group_move_probability_mean": float(np.mean(micro_group_probabilities[:, 1, 1])) if uav_count > 1 else None,
        "uav1_group_entropy_mean": float(np.mean(micro_group_entropies[:, 1])) if uav_count > 1 else None,
        "micro_group_entropy_mean": float(np.mean(micro_group_entropies)),
        "uav0_group_low_entropy_fraction": float(
            np.mean(micro_group_entropies[:, 0] < micro_group_entropy_floor)
        ),
        "uav1_group_low_entropy_fraction": float(np.mean(micro_group_entropies[:, 1] < micro_group_entropy_floor)) if uav_count > 1 else None,
        "uav0_group_probability_over_90_fraction": float(
            np.mean(group_probability_max[:, 0] > micro_group_probability_soft_cap)
        ),
        "uav1_group_probability_over_90_fraction": float(np.mean(group_probability_max[:, 1] > micro_group_probability_soft_cap)) if uav_count > 1 else None,
        "uav_candidate_usage_by_index": [
            [int(sum(int(item.actions[head]) == candidate_index for item in micro_buffer.transitions)) for candidate_index in range(max_uav_candidates)]
            for head in range(uav_count)
        ],
        "uav_proposal_source_usage": {
            f"uav{head}:{source}": int(sum(item.uav_candidate_sources[head][int(item.actions[head])] == source for item in micro_buffer.transitions))
            for head in range(uav_count)
            for source in sorted({name for item in micro_buffer.transitions for name in item.uav_candidate_sources[head]})
        },
        "valid_candidate_count_mean": float(
            np.mean([np.mean(item.valid_candidate_count) for item in micro_buffer.transitions])
        ),
        "uav0_mean_valid_candidates": float(
            np.mean([item.valid_candidate_count[0] for item in micro_buffer.transitions])
        ),
        "uav1_mean_valid_candidates": float(np.mean([item.valid_candidate_count[1] for item in micro_buffer.transitions])) if uav_count > 1 else None,
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
        "node_task_count": episode_node_task_counts,
        "exact_q_branch_requests": 0,
        "exact_q_environment_executions": 0,
    }
    if deterministic:
        summary["deterministic_action_selection"] = {
            "cache": deterministic_cache_selections,
            "uav": deterministic_uav_selections,
        }
    if deterministic:
        summary["deterministic_cache_protocol"] = {
            "cache_deterministic_protocol": "group_mass_then_raw_score_argmax",
            "cache_group_decision": "argmax_keep_vs_update_group_mass",
            "cache_update_candidate_decision": (
                "argmax_raw_score_within_legal_update_group"
            ),
            "cache_group_tie_break": "KEEP",
            "cardinality_corrected_cache_action": True,
        }
        summary["deterministic_cache_protocol_checks"] = {
            "group_rule_violation_count": cache_group_rule_violation_count,
            "update_inner_argmax_violation_count": (
                cache_update_inner_argmax_violation_count
            ),
            "probability_sum_error_max": cache_probability_sum_error_max,
            "probability_nonfinite_count": cache_probability_nonfinite_count,
            "probability_nan_count": cache_probability_nan_count,
            "probability_inf_count": cache_probability_inf_count,
            "probability_finite": (
                cache_probability_nonfinite_count == 0
                and cache_probability_nan_count == 0
                and cache_probability_inf_count == 0
            ),
            "group_tie_break": "KEEP",
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
