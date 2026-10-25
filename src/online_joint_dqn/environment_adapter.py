"""把候选条件化 DQN 接到正式的一 Slot 系统执行接口。"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import pickle
import sys
import time
from collections import Counter, OrderedDict
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from common import project_config, system_model
from decision import (
    cache_configuration,
    slot_environment,
    task_lifecycle,
    task_allocation,
    uav_configuration,
)
from .candidate_evaluator import (
    SearchBudget,
    SlotInput,
    SystemModelContext,
)
from .cache_optimizer import CacheOptimizer
from .uav_scheduler import UavScheduler

from .candidate_features import (
    ActionCandidate,
    CANDIDATE_FEATURE_CONTRACT_VERSION,
    CandidateFeatureEncoder,
    CandidateOption,
    DecisionForecast,
    _state_fingerprint,
)


CANDIDATE_GENERATION_CONTRACT_VERSION = "state_ranked_bounded_pair_cache_replace_v3_lifecycle"
DQN_CROSS_NODE_PAIR_PROPOSAL_LIMIT = 8
DQN_CROSS_NODE_PAIR_SCAN_LIMIT = 32
CANDIDATE_REBUILD_CONTEXT_SCHEMA_VERSION = "dqn_candidate_rebuild_context_v10_active_task_state"
EPISODE_REBUILD_SPEC_SCHEMA_VERSION = "dqn_episode_candidate_rebuild_spec_v3_active_task_state"


def candidate_generation_contract(
    *,
    max_cache_candidates_including_keep: int,
    max_uav_candidates_per_uav_including_stay: int,
) -> dict[str, Any]:
    """返回影响候选集合及其顺序的可序列化合同。"""

    return {
        "version": CANDIDATE_GENERATION_CONTRACT_VERSION,
        "single_node_cache": {
            "enumeration": "formal_exhaustive_legal_replace_per_node_v1",
            "dqn_single_node_candidate_cap": None,
            "checks": ["formal_replace", "per_node_capacity"],
            "unchanged_nodes": "preserve_previous_configuration",
        },
        "bounded_cache_optimizer_proposals": {
            "generator": "CacheOptimizer.generate_candidates",
            "per_node_proposal_generator": (
                "formal_generate_bounded_cache_target_configurations"
            ),
            "candidate_cap_including_keep": int(
                max_cache_candidates_including_keep
            ),
            "included_in_dqn_pool": True,
            "composite_proposal": "one_first_proposal_per_active_node",
            "single_node_selection": "slot_rotated_round_robin_by_proposal_rank",
            "deduplication": "target_cache_matrix_bytes_v1",
        },
        "active_task_cache_protection": "retain_assigned_edge_services_until_termination",
        "cache_updates_after_arrival_horizon": "forbidden",
        "cross_node_pair_proposals": {
            "representatives_per_node": 1,
            "representative_priority": (
                "proposal_source_priority_then_state_scoped_sha256"
            ),
            "pair_priority": "state_scoped_sha256_over_complete_pair_space_v1",
            "priority_before_scan_truncation": True,
            "scan_limit": int(DQN_CROSS_NODE_PAIR_SCAN_LIMIT),
            "retained_limit": int(DQN_CROSS_NODE_PAIR_PROPOSAL_LIMIT),
            "checks": ["formal_replace_per_changed_node", "per_node_capacity"],
        },
        "uav_move_proposals": {
            "generator": "formal_UavScheduler_enumerate_move_candidates",
            "candidate_cap_per_uav_including_stay": int(
                max_uav_candidates_per_uav_including_stay
            ),
            "proposal_order": "formal_scheduler_order",
        },
        "candidate_filtering": {
            "keep_stay_count": 1,
            "drop_unchanged_non_keep": True,
            "deduplicate_cache_targets": "full_node_service_matrix_bytes",
            "deduplicate_final_actions": "source_state_and_target_action_id",
            "reject_invalid_shapes_or_contract_mismatch": True,
        },
        "candidate_feature_contract_version": CANDIDATE_FEATURE_CONTRACT_VERSION,
        "legality_contract": {
            "cache": "formal_REPLACE_and_per_node_capacity",
            "uav": "formal_scheduler_boundary_distance_and_separation_validation",
        },
        "cross_operation_constraint": "one_cache_or_uav_operation_per_decision_v1",
        "candidate_identity": "source_state_and_full_target_configuration_v1",
    }


@dataclass(frozen=True)
class EnvironmentTransition:
    """正式执行器返回的真实单 Slot 奖励及推进后的状态。"""

    reward: float
    next_state_features: np.ndarray | None
    next_state_fingerprint: str | None
    next_legal_candidates: tuple[CandidateOption, ...]
    next_candidate_rebuild_context: Mapping[str, Any] | None
    terminated: bool
    duration_slots: int
    operation: str
    candidate_id: str
    final_cache_state: np.ndarray
    final_uav_positions: Mapping[str, np.ndarray]
    executor_calls: int
    matching_calls: int
    kkt_solver_calls: int
    state_isolation_pass: bool
    mutual_exclusion_pass: bool
    diagnostics: Mapping[str, Any]


def _input_fingerprint(slot_input: SlotInput) -> str:
    payload = (
        slot_input.slot_id,
        slot_input.time_s,
        slot_input.scenario_slice,
        slot_input.tasks,
        slot_input.vehicle_positions,
        slot_input.predicted_vehicle_positions,
        slot_input.prediction_horizon_mask,
        slot_input.prediction_metadata,
    )
    return hashlib.sha256(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


def _cache_nodes_at_uav_positions(
    cache_nodes: Mapping[str, Any],
    uav_positions: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy cache-node geometry and place each UAV at the supplied XY position."""

    result = deepcopy(dict(cache_nodes))
    node_ids = tuple(str(value) for value in result["node_ids"])
    node_roles = tuple(str(value).upper() for value in result["node_roles"])
    fixed_positions = result["fixed_positions"]
    if not isinstance(fixed_positions, dict):
        raise TypeError("cache node fixed_positions must be a mapping")
    for node_id, role in zip(node_ids, node_roles, strict=True):
        if role != "UAV":
            continue
        if node_id not in uav_positions:
            raise KeyError(f"UAV cache node {node_id} has no state position")
        position = np.asarray(uav_positions[node_id], dtype=np.float64).reshape(-1)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError(f"UAV {node_id} position must be a finite XY pair")
        fixed_positions[node_id] = position.copy()
    return result


_CANDIDATE_REBUILD_SCENARIO_FIELDS = (
    "vehicle_ids",
    "task_vehicle_ids",
    "positions_xy",
    "arrival",
    "service_id",
    "rho_hat",
)
_CANDIDATE_REBUILD_IDENTITY_FIELDS = (
    "schema_version",
    "source_state_fingerprint",
    "state_features",
    "cache_state",
    "uav_positions",
    "slot_id",
    "time_s",
    "prediction_scenario_sha256",
    "prediction_scenario",
    "decision_forecast",
    "prediction_horizon_mask",
    "protected_services_by_node",
    "active_task_states",
    "cache_updates_allowed",
    "state_feature_contract_version",
    "candidate_feature_contract_version",
    "candidate_generation_contract_version",
    "candidate_generation_contract",
    "candidate_capacity_limits",
    "network_architecture_version",
    "episode_seed",
    "episode_rebuild_id",
    "episode_rebuild_spec_sha256",
    "episode_vehicle_ids",
    "episode_task_vehicle_ids",
    "episode_cache_node_ids",
    "canonical_vehicle_ids",
    "canonical_cache_node_ids",
    "candidate_count",
    "candidate_ids_sha256",
)
_CANDIDATE_REBUILD_CACHE_MAX_ENTRIES = 16
_CANDIDATE_REBUILD_CACHE_MAX_CANDIDATES = 4096


def _prediction_scenario_sha256(scenario: Mapping[str, Any]) -> str:
    """对完整预测场景求摘要，避免在回放中重复保存无关预测字段。"""

    return hashlib.sha256(
        pickle.dumps(dict(scenario), protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def _candidate_rebuild_payload_sha256(context: Mapping[str, Any]) -> str:
    payload = {
        key: context[key]
        for key in _CANDIDATE_REBUILD_IDENTITY_FIELDS
        if key in context
    }
    if len(payload) != len(_CANDIDATE_REBUILD_IDENTITY_FIELDS):
        raise ValueError("候选重建上下文缺少身份字段")
    # 逐字段哈希，避免 pickle 对象图中的共享引用布局改变摘要但数据本身相同。
    digest = hashlib.sha256()
    for key in _CANDIDATE_REBUILD_IDENTITY_FIELDS:
        name = key.encode("utf-8")
        value = payload[key]
        if key == "state_features":
            # Replay stores shared state vectors as read-only float32 arrays,
            # then restores that same value into the compact candidate context.
            # NumPy pickles array writeability/layout flags, so hash a writable,
            # contiguous copy to keep the context digest stable across that
            # storage boundary while still hashing the full feature values.
            value = np.array(value, dtype=np.float32, copy=True, order="C").reshape(-1)
        encoded = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _candidate_ids_sha256(candidates: Sequence[CandidateOption]) -> str:
    digest = hashlib.sha256()
    for candidate in candidates:
        digest.update(candidate.action.candidate_id.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _episode_rebuild_spec_sha256(payload: Mapping[str, Any]) -> str:
    """逐字段计算重建规格摘要，避免跨字段对象共享影响 Pickle 图布局。"""

    digest = hashlib.sha256()
    for key in sorted(payload):
        key_bytes = str(key).encode("utf-8")
        value_bytes = pickle.dumps(payload[key], protocol=pickle.HIGHEST_PROTOCOL)
        digest.update(len(key_bytes).to_bytes(4, "big"))
        digest.update(key_bytes)
        digest.update(len(value_bytes).to_bytes(8, "big"))
        digest.update(value_bytes)
    return digest.hexdigest()


def _canonical_episode_vehicle_slots(
    slot_input: SlotInput,
    forecast: DecisionForecast,
    service_count: int,
    initial_cache_state: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """用 Episode 首个决策的有效预测和初始缓存建立稳定角色内空间槽位。"""

    scenario = dict(slot_input.scenario_slice)
    vehicle_ids = tuple(str(value) for value in np.asarray(scenario["vehicle_ids"]).tolist())
    task_vehicle_ids = tuple(
        str(value) for value in np.asarray(scenario["task_vehicle_ids"]).tolist()
    )
    service_vehicle_ids = tuple(
        str(value) for value in np.asarray(scenario["service_vehicle_ids"]).tolist()
    )
    mask = np.asarray(forecast.prediction_horizon_mask, dtype=np.int8).reshape(-1)
    valid_horizons = tuple(int(index) for index in np.flatnonzero(mask))
    if not valid_horizons:
        raise ValueError("规范车辆槽位需要至少一个有效预测时域")
    if set(task_vehicle_ids) & set(service_vehicle_ids) or set(task_vehicle_ids) | set(
        service_vehicle_ids
    ) != set(vehicle_ids):
        raise ValueError("规范车辆槽位的 SV/TV 身份集合不完整")
    demand_by_horizon = cache_configuration.predicted_vehicle_service_demand_by_horizon(
        scenario,
        int(service_count),
        beta_discount=float(
            project_config.get_config_value(
                slot_input.model.project_cfg, "cache.discount_beta", 1.0
            )
        ),
        prediction_horizon_mask=mask,
    )
    if demand_by_horizon.shape != (
        len(mask), len(vehicle_ids), int(service_count)
    ):
        raise ValueError("规范车辆槽位的预测需求维度与正式预测合同不一致")
    vehicle_index = {vehicle_id: index for index, vehicle_id in enumerate(vehicle_ids)}
    cache_node_ids = tuple(
        str(value) for value in slot_input.model.cache_nodes["node_ids"]
    )
    cache_state = np.asarray(initial_cache_state, dtype=np.int8)
    if cache_state.shape != (len(cache_node_ids), int(service_count)):
        raise ValueError("规范车辆槽位初始缓存与正式缓存节点顺序不一致")
    initial_cache_by_node = {
        node_id: tuple(int(value) for value in cache_state[index])
        for index, node_id in enumerate(cache_node_ids)
    }

    def role_order(role_ids: Sequence[str]) -> tuple[str, ...]:
        def spatial_key(vehicle_id: str) -> tuple[Any, ...]:
            positions: list[float] = []
            for horizon_index in valid_horizons:
                horizon_positions = forecast.predicted_vehicle_positions_horizon[
                    horizon_index
                ]
                if vehicle_id not in horizon_positions:
                    raise ValueError(
                        f"规范车辆槽位预测缺少车辆 {vehicle_id} 的时域 {horizon_index} 位置"
                    )
                xy = np.asarray(horizon_positions[vehicle_id], dtype=np.float64).reshape(2)
                if not np.isfinite(xy).all():
                    raise ValueError("规范车辆槽位预测坐标必须为有限数值")
                positions.extend(float(value) for value in xy)
            first_x, first_y = positions[:2]
            predicted_demand_signature = tuple(
                float(value)
                for value in demand_by_horizon[
                    list(valid_horizons), vehicle_index[vehicle_id], :
                ].reshape(-1)
            )
            # 预测和已有缓存均相同时，车辆在当前可见信息下不可区分，再用本 Episode 源顺序打破并列。
            episode_order_tiebreak = tuple(role_ids).index(vehicle_id)
            return (
                first_x,
                first_y,
                *positions,
                *predicted_demand_signature,
                initial_cache_by_node.get(vehicle_id, ()),
                episode_order_tiebreak,
            )

        return tuple(sorted((str(value) for value in role_ids), key=spatial_key))

    service_order = role_order(service_vehicle_ids)
    task_order = role_order(task_vehicle_ids)
    return service_order + task_order, service_order


def _bounded_cross_node_pair_cache_targets(
    current_cache_state: Any,
    exact_targets_by_node: Mapping[int, Sequence[Any]],
    node_ids: Sequence[str],
    service_ids: Sequence[int],
    service_sizes_mb: Any,
    capacities_mb: Any,
    *,
    source_state_fingerprint: str,
    existing_target_keys: set[bytes],
) -> tuple[list[tuple[np.ndarray, str]], dict[str, Any]]:
    """按状态绑定的稳定优先级扫描有界双节点 REPLACE 提议。"""

    cache_state = np.asarray(current_cache_state, dtype=np.int8)
    sizes = np.asarray(service_sizes_mb, dtype=np.float64).reshape(-1)
    capacities = np.asarray(capacities_mb, dtype=np.float64).reshape(-1)
    if cache_state.ndim != 2 or cache_state.shape != (len(node_ids), len(sizes)):
        raise ValueError("双节点缓存提议的状态、节点和服务维度不一致")
    service_id_values = tuple(int(value) for value in service_ids)
    if len(service_id_values) != len(sizes) or len(set(service_id_values)) != len(sizes):
        raise ValueError("双节点缓存提议的服务 ID 与目录维度不一致")
    if capacities.shape != (len(node_ids),) or not np.isfinite(capacities).all():
        raise ValueError("双节点缓存提议的节点容量无效")
    if not source_state_fingerprint:
        raise ValueError("双节点提议优先级必须绑定来源状态指纹")

    def canonical_node_target(node_index: int, target: np.ndarray) -> dict[str, Any]:
        return {
            "node_id": str(node_ids[node_index]),
            "service_ids": sorted(
                service_id_values[index]
                for index, present in enumerate(target.tolist())
                if int(present) == 1
            ),
        }

    def stable_priority(scope: str, value: Any) -> str:
        encoded = json.dumps(
            {
                "contract": CANDIDATE_GENERATION_CONTRACT_VERSION,
                "source_state_fingerprint": str(source_state_fingerprint),
                "scope": scope,
                "candidate": value,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    source_priority = {
        "prediction": 0,
        "coverage_anchor": 1,
        "intermediate": 2,
        "low_reconfiguration": 3,
        "exhaustive_feasible": 4,
    }
    representatives: dict[int, tuple[np.ndarray, str]] = {}
    for node_index, node_id in enumerate(node_ids):
        by_target: dict[bytes, tuple[np.ndarray, str]] = {}
        for candidate in exact_targets_by_node.get(node_index, ()):
            if (
                int(candidate.node_index) != node_index
                or str(candidate.node_id) != str(node_id)
            ):
                raise ValueError("单节点缓存候选的节点索引与 ID 不一致")
            target = np.asarray(candidate.target_node_cache, dtype=np.int8).reshape(-1)
            if (
                target.shape != (cache_state.shape[1],)
                or not np.all(np.isin(target, [0, 1]))
                or not cache_configuration.is_legal_cache_replacement(
                    cache_state[node_index], target
                )
                or not cache_configuration.is_node_feasible(
                    target, sizes, float(capacities[node_index])
                )
            ):
                continue
            key = np.ascontiguousarray(target, dtype=np.int8).tobytes()
            source = str(candidate.proposal_source)
            previous = by_target.get(key)
            if previous is None or (
                source_priority.get(source, len(source_priority)), source
            ) < (
                source_priority.get(previous[1], len(source_priority)), previous[1]
            ):
                by_target[key] = (target.copy(), source)
        if by_target:
            representatives[node_index] = min(
                by_target.values(),
                key=lambda item: (
                    source_priority.get(item[1], len(source_priority)),
                    stable_priority(
                        "node_representative",
                        {
                            **canonical_node_target(node_index, item[0]),
                            "proposal_source": item[1],
                        },
                    ),
                    item[1],
                ),
            )

    active_nodes = sorted(representatives)
    potential_pair_count = len(active_nodes) * (len(active_nodes) - 1) // 2
    ranked_pairs: list[tuple[str, str, int, int]] = []
    for left_node, right_node in itertools.combinations(active_nodes, 2):
        canonical_pair = sorted(
            (
                canonical_node_target(left_node, representatives[left_node][0]),
                canonical_node_target(right_node, representatives[right_node][0]),
            ),
            key=lambda item: item["node_id"],
        )
        canonical_payload = json.dumps(
            canonical_pair,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        ranked_pairs.append(
            (
                stable_priority("cross_node_pair", canonical_pair),
                canonical_payload,
                left_node,
                right_node,
            )
        )
    ranked_pairs.sort(key=lambda item: (item[0], item[1]))
    scan_pairs = ranked_pairs[:DQN_CROSS_NODE_PAIR_SCAN_LIMIT]
    retained: list[tuple[np.ndarray, str]] = []
    seen_keys: set[bytes] = set()
    scanned_pair_count = 0
    legal_pair_count = 0
    rejected_legality_count = 0
    duplicate_existing_count = 0
    duplicate_pair_count = 0
    retention_omitted_count = 0
    for _, _, left_node, right_node in scan_pairs:
        scanned_pair_count += 1
        target_state = cache_state.copy()
        target_state[left_node] = representatives[left_node][0]
        target_state[right_node] = representatives[right_node][0]
        changed_indices = tuple(
            int(index)
            for index in np.flatnonzero(np.any(target_state != cache_state, axis=1))
        )
        valid_pair = changed_indices == (left_node, right_node)
        if valid_pair:
            for node_index in changed_indices:
                valid_pair = cache_configuration.is_legal_cache_replacement(
                    cache_state[node_index], target_state[node_index]
                ) and cache_configuration.is_node_feasible(
                    target_state[node_index], sizes, float(capacities[node_index])
                )
                if not valid_pair:
                    break
        if not valid_pair:
            rejected_legality_count += 1
            continue
        legal_pair_count += 1
        key = np.ascontiguousarray(target_state, dtype=np.int8).tobytes()
        if key in existing_target_keys:
            duplicate_existing_count += 1
            continue
        if key in seen_keys:
            duplicate_pair_count += 1
            continue
        seen_keys.add(key)
        if len(retained) >= DQN_CROSS_NODE_PAIR_PROPOSAL_LIMIT:
            retention_omitted_count += 1
            continue
        retained.append(
            (
                target_state,
                "state_ranked_pair:"
                f"{node_ids[left_node]}:{representatives[left_node][1]}+"
                f"{node_ids[right_node]}:{representatives[right_node][1]}",
            )
        )
    unscanned_pair_count = max(0, potential_pair_count - scanned_pair_count)
    truncated = bool(unscanned_pair_count or retention_omitted_count)
    return retained, {
        "cache_cross_node_pair_candidate_source_node_count": len(active_nodes),
        "cache_cross_node_pair_potential_representative_count": potential_pair_count,
        "cache_cross_node_pair_priority_ranked_count": len(ranked_pairs),
        "cache_cross_node_pair_proposals_scanned_count": scanned_pair_count,
        "cache_cross_node_pair_legality_pass_count": legal_pair_count,
        "cache_cross_node_pair_legality_rejected_count": rejected_legality_count,
        "cache_cross_node_pair_deduplicated_existing_count": duplicate_existing_count,
        "cache_cross_node_pair_deduplicated_pair_count": duplicate_pair_count,
        "cache_cross_node_pair_new_unique_candidate_count_scanned": max(
            0, legal_pair_count - duplicate_existing_count - duplicate_pair_count
        ),
        "cache_cross_node_pair_candidates_retained_count": len(retained),
        "cache_cross_node_pair_retention_limit_omitted_count": retention_omitted_count,
        "cache_cross_node_pair_candidate_limit": DQN_CROSS_NODE_PAIR_PROPOSAL_LIMIT,
        "cache_cross_node_pair_scan_limit": DQN_CROSS_NODE_PAIR_SCAN_LIMIT,
        "cache_cross_node_pair_unscanned_count": unscanned_pair_count,
        "cache_cross_node_pair_candidates_truncated": bool(truncated),
        "cache_cross_node_pair_generation_complete_within_policy": not truncated,
        "cache_cross_node_pair_combinations_exhaustive": False,
        "cache_cross_node_pair_priority_contract": (
            "state_scoped_sha256_over_complete_pair_space_v1"
        ),
    }


def _validate_active_task_vehicle_positions(
    *,
    slot_id: int,
    arrival_horizon_slots: int,
    active_vehicle_ids: Sequence[str],
    vehicle_positions: Mapping[str, Any],
) -> None:
    """Require current physical positions for active task links without inventing a departure policy."""

    missing = sorted(
        {str(vehicle_id) for vehicle_id in active_vehicle_ids}
        - {str(vehicle_id) for vehicle_id in vehicle_positions}
    )
    if not missing:
        return
    if int(slot_id) >= int(arrival_horizon_slots):
        raise RuntimeError(
            "UNRESOLVED_VEHICLE_DEPARTURE_SEMANTICS: "
            f"execution slot={slot_id}, active task vehicles lack current physical "
            f"positions={missing}; no continuation, migration, failure, or "
            "trajectory-fill policy is configured."
        )
    raise ValueError(
        "MISSING_CURRENT_ACTIVE_TASK_VEHICLE_POSITION: "
        f"slot={slot_id}, vehicles={missing}"
    )


class ExclusiveDQNEnvironment:
    """按真实 Episode 顺序推进，每次只执行一个合法互斥配置。"""

    def __init__(
        self,
        *,
        context: Any,
        slot_inputs: Sequence[SlotInput],
        decision_steps: Sequence[Any],
        search_budget: SearchBudget | None = None,
        decision_count: int | None = None,
        terminal_at_end: bool = True,
    ) -> None:
        if not slot_inputs:
            raise ValueError("环境至少需要一个已对齐的正式 Slot 输入")
        if any(item.source_kind != "current_actual" for item in slot_inputs):
            raise ValueError("训练环境只接受当前实际 Slot，不接受未来真实输入冒充预测")
        self.context = context
        self.slot_inputs = tuple(slot_inputs)
        self.decision_forecasts = tuple(
            DecisionForecast.from_aligned_step(step) for step in decision_steps
        )
        if len(self.decision_forecasts) != len(self.slot_inputs):
            raise ValueError("正式对齐预测步骤与 Slot 输入数量不一致")
        if any(
            forecast.slot != slot_input.slot_id
            for forecast, slot_input in zip(
                self.decision_forecasts, self.slot_inputs, strict=True
            )
        ):
            raise ValueError("正式预测步骤与 Slot 输入未按同一时隙对齐")
        self.model = self.slot_inputs[0].model
        lifecycle_parameters = project_config.get_cross_slot_task_parameters(
            dict(self.model.project_cfg)
        )
        self.arrival_horizon_slots = int(
            lifecycle_parameters["arrival_horizon_slots"]
        )
        self.k_max_slots = int(lifecycle_parameters["k_max_slots"])
        self.execution_horizon_slots = int(
            lifecycle_parameters["execution_horizon_slots"]
        )
        self.terminal_execution_slot = self.execution_horizon_slots - 1
        self.task_lifecycle = task_lifecycle.TaskLifecycleManager(
            arrival_horizon_slots=self.arrival_horizon_slots,
            k_max_slots=self.k_max_slots,
            dt_s=float(lifecycle_parameters["dt_s"]),
        )
        self.terminal_utility_parameters = (
            project_config.get_terminal_task_utility_parameters(
                dict(self.model.project_cfg)
            )
        )
        if len(self.slot_inputs) < self.execution_horizon_slots:
            raise ValueError(
                "MISSING_CAUSAL_EXECUTION_TAIL_INPUTS: "
                f"need {self.execution_horizon_slots} aligned physical/decision slots "
                f"(T={self.arrival_horizon_slots}, K_max={self.k_max_slots}), "
                f"available={len(self.slot_inputs)}, "
                f"missing={self.execution_horizon_slots - len(self.slot_inputs)} tail slots; "
                "future ground-truth positions cannot substitute for decision forecasts."
            )
        if int(self.slot_inputs[0].slot_id) != 0 or any(
            int(item.slot_id) != index
            for index, item in enumerate(self.slot_inputs[: self.execution_horizon_slots])
        ):
            raise ValueError("跨时隙执行输入必须按零基连续时隙从 slot 0 开始")
        tail_arrivals = [
            (int(item.slot_id), len(item.tasks))
            for item in self.slot_inputs[: self.execution_horizon_slots]
            if int(item.slot_id) >= self.arrival_horizon_slots and item.tasks
        ]
        if tail_arrivals:
            raise ValueError(
                "task arrivals are forbidden in execution tail slots "
                f"{self.arrival_horizon_slots}..{self.execution_horizon_slots - 1}: "
                f"found={tail_arrivals[:10]}"
            )
        trajectory_times = [
            item.time_s for item in self.slot_inputs[: self.execution_horizon_slots]
        ]
        if any(value is None for value in trajectory_times) or any(
            not np.isclose(
                float(trajectory_times[index + 1]) - float(trajectory_times[index]),
                1.0,
                rtol=0.0,
                atol=1e-9,
            )
            for index in range(len(trajectory_times) - 1)
        ):
            raise ValueError("扩展执行轨迹时间戳必须按 1 秒连续对齐")
        self.search_budget = search_budget or self._candidate_capacity_budget()
        first_slot_input = self.slot_inputs[0]
        first_forecast = self.decision_forecasts[0]
        (
            self.canonical_vehicle_ids,
            canonical_service_vehicle_ids,
        ) = _canonical_episode_vehicle_slots(
            first_slot_input,
            first_forecast,
            len(tuple(self.model.service_catalog["service_ids"])),
            context.initial_cache,
        )
        source_cache_node_ids = tuple(
            str(value) for value in self.model.cache_nodes["node_ids"]
        )
        canonical_cache_node_ids = canonical_service_vehicle_ids + tuple(
            value
            for value in source_cache_node_ids
            if value not in set(canonical_service_vehicle_ids)
        )
        if len(canonical_cache_node_ids) != len(source_cache_node_ids) or set(
            canonical_cache_node_ids
        ) != set(source_cache_node_ids):
            raise ValueError("规范缓存节点槽位无法与正式缓存节点一一映射")
        self.canonical_cache_node_ids = canonical_cache_node_ids
        self.episode_seed = first_slot_input.prediction_metadata.get("episode_seed")
        self.episode_seed = (
            None if self.episode_seed is None else int(self.episode_seed)
        )
        self.decision_count = len(self.slot_inputs) if decision_count is None else int(decision_count)
        self.terminal_at_end = bool(terminal_at_end)
        if self.decision_count <= 0 or self.decision_count > len(self.slot_inputs):
            raise ValueError("决策 Slot 数量与对齐输入长度不匹配")
        if not self.terminal_at_end and self.decision_count >= len(self.slot_inputs):
            raise ValueError("截断窗口需要额外的下一 Slot 观测用于 Double DQN bootstrap")
        if self.decision_count != self.execution_horizon_slots or not self.terminal_at_end:
            raise ValueError(
                "跨时隙物理 Episode 必须执行完整扩展时域并在末尾终止："
                f"required={self.execution_horizon_slots}, "
                f"decision_count={self.decision_count}, terminal_at_end={self.terminal_at_end}"
            )
        self.evaluator = None
        self.uav_scheduler = UavScheduler()
        self.cache_optimizer = CacheOptimizer(self.uav_scheduler)
        self.encoder = CandidateFeatureEncoder(
            self.slot_inputs[0],
            observation_spec=context.observation_spec,
            train_origin_start=int(context.train_origin_start),
            train_origin_end=int(context.train_origin_end),
            k_max_slots=self.k_max_slots,
            canonical_vehicle_ids=self.canonical_vehicle_ids,
            canonical_cache_node_ids=self.canonical_cache_node_ids,
        )
        self._initial_cache = np.asarray(context.initial_cache, dtype=np.int8).copy()
        self._initial_uav = {
            key: np.asarray(value, dtype=np.float64).copy()
            for key, value in context.initial_uav.items()
        }
        self._candidate_rebuild_cache: OrderedDict[
            tuple[str, ...], tuple[CandidateOption, ...]
        ] = OrderedDict()
        self._candidate_rebuild_cache_candidate_count = 0
        self._candidate_rebuild_cache_stats = {
            "hit_count": 0,
            "miss_count": 0,
            "generation_count": 0,
            "eviction_count": 0,
            "bypass_count": 0,
        }
        self.reset()
        (
            self._episode_rebuild_id,
            self._episode_rebuild_spec_sha256,
        ) = EpisodeCandidateRebuildAdapter.identity_for_environment(self)

    def _candidate_capacity_budget(self) -> SearchBudget:
        """将 DQN 适配层上限设为正式候选生成器容量的自然上界。"""

        cache_node_count = len(tuple(self.model.cache_nodes["node_ids"]))
        cache_capacity = int(self.model.cache_config.target_candidate_capacity)
        uav_capacity = int(
            uav_configuration.uav_search_config_from_project(
                dict(self.model.project_cfg)
            ).target_candidate_capacity
        )
        # 缓存目标包含 KEEP、首提议组合及每节点提议；UAV 上限还包含 STAY。
        return SearchBudget(
            max_cache_candidates=2 + cache_node_count * cache_capacity,
            max_uav_candidates_per_uav=uav_capacity + 1,
        )

    def _candidate_capacity_limits(self) -> dict[str, int]:
        return {
            "max_cache_candidates_including_keep": int(
                self.search_budget.max_cache_candidates
            ),
            "max_uav_candidates_per_uav_including_stay": int(
                self.search_budget.max_uav_candidates_per_uav
            ),
        }

    @classmethod
    def from_formal_training_episode(
        cls,
        project_root: str | Path,
        *,
        episode_seed: int,
        origin_start: int = 0,
        max_steps: int | None = None,
        search_budget: SearchBudget | None = None,
    ) -> "ExclusiveDQNEnvironment":
        """读取当前正式 Training Pool 的一个完整生命周期 Episode。"""

        return cls.from_formal_split_episode(
            project_root,
            split="train",
            episode_seed=episode_seed,
            origin_start=origin_start,
            max_steps=max_steps,
            search_budget=search_budget,
        )

    @classmethod
    def from_formal_split_episode(
        cls,
        project_root: str | Path,
        *,
        split: str,
        episode_seed: int,
        origin_start: int = 0,
        max_steps: int | None = None,
        search_budget: SearchBudget | None = None,
        validated_pool: tuple[Path, Mapping[str, Any]] | None = None,
    ) -> "ExclusiveDQNEnvironment":
        """读取正式 Train/Validation/Test 池中的一个完整 Episode。"""

        if max_steps is not None and int(max_steps) <= 0:
            raise ValueError("max_steps 必须大于 0")
        root = Path(project_root).resolve()
        training_scripts = root / "scripts" / "training"
        if str(training_scripts) not in sys.path:
            sys.path.insert(0, str(training_scripts))
        import training_pool_contract as pool_contract
        from common.environment_interface import aligned_execution_steps

        split_contracts = {
            "train": (
                pool_contract.TRAIN_POOL_RELATIVE,
                pool_contract.DEFAULT_TRAINING_SEED_START,
                pool_contract.FORMAL_TRAINING_EPISODE_COUNT,
            ),
            "validation": (
                pool_contract.VALIDATION_POOL_RELATIVE,
                30001,
                30,
            ),
            "test": (
                pool_contract.TEST_POOL_RELATIVE,
                pool_contract.DEFAULT_TEST_SEED_START,
                pool_contract.FORMAL_TEST_EPISODE_COUNT,
            ),
        }
        split = str(split).strip().lower()
        if split not in split_contracts:
            raise ValueError(f"Unsupported formal split: {split!r}")
        relative_pool, expected_seed_start, expected_episode_count = split_contracts[split]
        if validated_pool is None:
            runtime_config = project_config.parse_toml_file(
                root / pool_contract.CONFIG_RELATIVE
            )
            pool, pool_info = pool_contract._validate_pool(
                root,
                relative_pool,
                expected_split=split,
                expected_seed_start=expected_seed_start,
                expected_episode_count=expected_episode_count,
                runtime_config=runtime_config,
            )
        else:
            pool, pool_info = validated_pool
            if str(pool_info.get("split")) != split:
                raise ValueError(
                    "Validated pool split does not match requested Episode split: "
                    f"requested={split!r}, validated={pool_info.get('split')!r}"
                )
            if Path(pool).resolve() != (root / relative_pool).resolve():
                raise ValueError("Validated pool path does not match formal split path")
            if list(pool_info.get("active_episode_seeds", ())) != list(
                range(expected_seed_start, expected_seed_start + expected_episode_count)
            ):
                raise ValueError(f"Validated {split} active seed set is not the formal set")
        context = pool_contract._episode_context(root, pool, int(episode_seed), pool_info)
        lifecycle_parameters = project_config.get_cross_slot_task_parameters(
            dict(context.cfg)
        )
        arrival_horizon = int(lifecycle_parameters["arrival_horizon_slots"])
        execution_horizon = int(lifecycle_parameters["execution_horizon_slots"])
        if int(origin_start) != 0:
            raise ValueError(
                "跨时隙生命周期必须从完整 Episode 的 slot 0 开始；"
                f"received origin_start={origin_start}"
            )
        if max_steps is not None and int(max_steps) != execution_horizon:
            raise ValueError(
                "跨时隙生命周期必须使用完整执行时域，不能用到达时域长度代替："
                f"requested max_steps={max_steps}, "
                f"arrival_horizon={arrival_horizon}, "
                f"required execution_horizon={execution_horizon}"
            )
        steps = aligned_execution_steps(context)
        decision_count = len(steps)
        terminal_at_end = decision_count == execution_horizon
        model = SystemModelContext(
            service_catalog=context.service_catalog,
            project_cfg=context.cfg,
            cache_nodes=context.cache_nodes,
            cache_config=context.cache_cfg,
            flight_boundary=context.flight_boundary,
        )
        slot_inputs = tuple(
            SlotInput(
                slot_id=int(step.slot),
                time_s=float(step.time_s),
                scenario_slice=step.scenario_slice,
                tasks=tuple(step.actual_tasks),
                vehicle_positions=step.actual_vehicle_positions,
                predicted_vehicle_positions=step.predicted_vehicle_positions,
                model=model,
                prediction_horizon_mask=step.prediction_horizon_mask,
                scenario_id=f"formal_train:{int(episode_seed)}:{int(step.slot)}",
                source_kind="current_actual",
                prediction_metadata={
                    "source": "formal_aligned_execution_steps",
                    "episode_seed": int(episode_seed),
                    "origin_slot": int(step.slot),
                    "future_ground_truth_used_for_candidate_features": False,
                },
            )
            for step in steps
        )
        return cls(
            context=context,
            slot_inputs=slot_inputs,
            decision_steps=steps,
            search_budget=search_budget,
            decision_count=decision_count,
            terminal_at_end=terminal_at_end,
        )

    def reset(self) -> None:
        """回到该 realization 的初始配置和窗口起点。"""

        self.index = 0
        self.cache_state = self._initial_cache.copy()
        self.task_lifecycle.reset()
        self._settled_task_ids: set[str] = set()
        self.uav_positions = {
            key: value.copy() for key, value in self._initial_uav.items()
        }
        self._cached_state_fingerprint: str | None = None
        self._cached_state_features: np.ndarray | None = None
        self._cached_candidates: tuple[CandidateOption, ...] | None = None
        self._candidate_generation_diagnostics: dict[str, Any] = {}
        self.executor_calls = 0
        self.matching_calls = 0
        self.kkt_solver_calls = 0
        self._candidate_rebuild_cache.clear()
        self._candidate_rebuild_cache_candidate_count = 0
        for key in self._candidate_rebuild_cache_stats:
            self._candidate_rebuild_cache_stats[key] = 0

    @property
    def candidate_rebuild_cache_metrics(self) -> dict[str, int]:
        """返回有界回放候选缓存的命中、淘汰及容量统计。"""

        return {
            **self._candidate_rebuild_cache_stats,
            "retained_context_count": len(self._candidate_rebuild_cache),
            "retained_candidate_count": self._candidate_rebuild_cache_candidate_count,
            "max_context_count": _CANDIDATE_REBUILD_CACHE_MAX_ENTRIES,
            "max_candidate_count": _CANDIDATE_REBUILD_CACHE_MAX_CANDIDATES,
        }

    @property
    def terminated(self) -> bool:
        return self.index >= self.decision_count and self.terminal_at_end

    @property
    def can_act(self) -> bool:
        """窗口内是否还有需要执行动作的 Slot。"""

        return self.index < self.decision_count

    def observe(self) -> tuple[np.ndarray, tuple[CandidateOption, ...]]:
        """返回当前观测及与该状态匹配的全部有界合法候选。"""

        if not self.can_act:
            raise StopIteration("当前窗口没有更多待执行决策 Slot")
        return self._prepare_observation()

    def _prepare_observation(self) -> tuple[np.ndarray, tuple[CandidateOption, ...]]:
        if self.index >= len(self.slot_inputs):
            raise StopIteration("没有可用于当前状态或 bootstrap 的下一 Slot 输入")
        slot_input = self.slot_inputs[self.index]
        forecast = self.decision_forecasts[self.index]
        formal_observation = self._formal_observation(forecast)
        state_features = self.encoder.encode_state(
            formal_observation,
            forecast,
            slot_input.scenario_slice,
            self._active_task_state_snapshots(),
        )
        protected_services = self.task_lifecycle.protected_services_by_node
        cache_updates_allowed = (
            int(slot_input.slot_id) + 1 <= self.arrival_horizon_slots
        )
        fingerprint = self._decision_state_fingerprint(
            slot_input,
            forecast,
            state_features,
            protected_services_by_node=protected_services,
            cache_updates_allowed=cache_updates_allowed,
        )
        if fingerprint != self._cached_state_fingerprint:
            input_before = _input_fingerprint(slot_input)
            cache_before = self.cache_state.copy()
            uav_before = {key: value.copy() for key, value in self.uav_positions.items()}
            self._cached_state_features = state_features
            (
                self._cached_candidates,
                self._candidate_generation_diagnostics,
            ) = self._generate_legal_candidates(
                slot_input,
                forecast,
                fingerprint,
                current_cache_state=self.cache_state,
                current_uav_positions=self.uav_positions,
                protected_services_by_node=protected_services,
                cache_updates_allowed=cache_updates_allowed,
            )
            generation_isolation = (
                _input_fingerprint(slot_input) == input_before
                and np.array_equal(cache_before, self.cache_state)
                and all(
                    np.array_equal(uav_before[key], self.uav_positions[key])
                    for key in uav_before
                )
            )
            if not generation_isolation:
                raise RuntimeError("候选生成意外修改了环境输入或当前配置状态")
            self._candidate_generation_diagnostics[
                "candidate_generation_state_isolation_pass"
            ] = True
            self._cached_state_fingerprint = fingerprint
        assert self._cached_state_features is not None
        assert self._cached_candidates is not None
        return self._cached_state_features.copy(), self._cached_candidates

    def _active_task_state_snapshots(self) -> tuple[dict[str, Any], ...]:
        """Return existing work plus current requests visible before the action."""

        snapshots = [
            state.snapshot() for state in self.task_lifecycle.active_tasks.values()
        ]
        if self.index < len(self.slot_inputs):
            slot_input = self.slot_inputs[self.index]
            arrival_slot = int(slot_input.slot_id) + 1
            if arrival_slot > self.arrival_horizon_slots and slot_input.tasks:
                raise ValueError("new task requests appear beyond the arrival horizon")
            seen_current_vehicles: set[str] = set()
            active_ids = {str(state["task_id"]) for state in snapshots}
            for task in slot_input.tasks:
                if str(task.execution_stage).upper() != "NEW":
                    raise ValueError("current task requests must enter as NEW")
                vehicle_id = str(task.vehicle_id)
                if vehicle_id in seen_current_vehicles:
                    raise ValueError(
                        f"multiple current requests for TV {vehicle_id} in slot {arrival_slot}"
                    )
                seen_current_vehicles.add(vehicle_id)
                task_id = f"{vehicle_id}@{arrival_slot}"
                input_bits = float(task.input_bits)
                cpu_cycles = float(task.cpu_cycles)
                if (
                    not math.isfinite(input_bits)
                    or input_bits <= 0.0
                    or not math.isfinite(cpu_cycles)
                    or cpu_cycles <= 0.0
                ):
                    raise ValueError(f"current request {task_id} has invalid demand")
                if task_id in active_ids:
                    raise ValueError(f"current request {task_id} already exists in lifecycle")
                snapshots.append(
                    {
                        "task_id": task_id,
                        "vehicle_id": vehicle_id,
                        "arrival_slot": arrival_slot,
                        "service_id": int(task.service_id),
                        "input_bits": input_bits,
                        "compute_density_cycles_per_bit": cpu_cycles / input_bits,
                        "cpu_cycles": cpu_cycles,
                        "remaining_upload_bits": input_bits,
                        "remaining_compute_cycles": cpu_cycles,
                        "remaining_slots": self.k_max_slots,
                        "execution_node": None,
                        "cumulative_upload_energy_j": 0.0,
                        "execution_stage": "UPLOADING",
                        "status": "ACTIVE",
                        "terminal_slot": None,
                    }
                )
                active_ids.add(task_id)
        return tuple(
            sorted(
                snapshots,
                key=lambda item: (
                    str(item["vehicle_id"]),
                    int(item["arrival_slot"]),
                    str(item["task_id"]),
                ),
            )
        )

    def _generate_legal_candidates(
        self,
        slot_input: SlotInput,
        forecast: DecisionForecast,
        source_state_fingerprint: str,
        *,
        current_cache_state: Any,
        current_uav_positions: Mapping[str, Any],
        protected_services_by_node: Mapping[str, Sequence[int]] | None = None,
        cache_updates_allowed: bool = True,
    ) -> tuple[tuple[CandidateOption, ...], dict[str, Any]]:
        # 候选器只接收预测场景及配置，不接触当前真实任务和实际车辆轨迹。
        candidate_input = replace(slot_input, tasks=(), vehicle_positions={})
        if candidate_input.tasks or candidate_input.vehicle_positions:
            raise RuntimeError("候选生成输入未能隔离当前真实任务或车辆位置")
        cache_state = np.asarray(current_cache_state, dtype=np.int8).copy()
        uav_positions = {
            key: np.asarray(value, dtype=np.float64).copy()
            for key, value in current_uav_positions.items()
        }
        cache_complementarity_context = self.encoder.prepare_cache_complementarity_context(
            previous_cache_state=cache_state,
            previous_uav_positions=uav_positions,
            slot_input=candidate_input,
            decision_forecast=forecast,
        )
        candidates: list[CandidateOption] = []
        keep = ActionCandidate.from_states(
            previous_cache_state=cache_state,
            previous_uav_positions=uav_positions,
            target_cache_state=cache_state,
            target_uav_positions=uav_positions,
            slot_input=candidate_input,
            source_state_fingerprint=source_state_fingerprint,
            proposal_source="KEEP_STAY",
        )
        candidates.append(
            CandidateOption.create(
                keep,
                self.encoder.encode_candidate(
                    keep,
                    previous_cache_state=cache_state,
                    previous_uav_positions=uav_positions,
                    slot_input=candidate_input,
                    decision_forecast=forecast,
                    cache_complementarity_context=cache_complementarity_context,
                ),
            )
        )
        cache_generation_diagnostics: dict[str, Any] = {}
        bounded_cache_targets, bounded_cache_complete = self.cache_optimizer.generate_candidates(
            current_cache_state=cache_state,
            slot_input=candidate_input,
            search_budget=self.search_budget,
            generation_diagnostics=cache_generation_diagnostics,
        )
        service_catalog = dict(candidate_input.model.service_catalog)
        cache_nodes = deepcopy(dict(candidate_input.model.cache_nodes))
        cache_targets_by_key: dict[bytes, tuple[np.ndarray, str]] = {}
        cache_proposal_count = 0
        single_node_diagnostics: dict[str, dict[str, Any]] = {}
        single_node_counts: dict[str, int] = {}
        exact_targets_by_node: dict[int, tuple[Any, ...]] = {}
        raw_combination_count = 0
        capacity_feasible_count = 0
        legal_replace_count = 0
        for node_index, node_id in enumerate(cache_nodes["node_ids"]):
            node_diagnostics: dict[str, Any] = {}
            exact_targets = cache_configuration.enumerate_all_cache_target_configurations(
                cache_state,
                node_index,
                service_catalog,
                deepcopy(cache_nodes),
                deepcopy(dict(candidate_input.scenario_slice)),
                candidate_input.model.cache_config,
                prediction_horizon_mask=deepcopy(
                    candidate_input.prediction_horizon_mask
                ),
                generation_diagnostics=node_diagnostics,
            )
            single_node_diagnostics[str(node_id)] = node_diagnostics
            single_node_counts[str(node_id)] = len(exact_targets)
            exact_targets_by_node[node_index] = tuple(exact_targets)
            raw_combination_count += int(node_diagnostics["raw_combination_count"])
            capacity_feasible_count += int(
                node_diagnostics["capacity_feasible_count"]
            )
            legal_replace_count += int(node_diagnostics["legal_replace_count"])
            for target in exact_targets:
                target_state = cache_state.copy()
                target_state[node_index] = np.asarray(
                    target.target_node_cache, dtype=np.int8
                )
                key = np.ascontiguousarray(target_state, dtype=np.int8).tobytes()
                cache_proposal_count += 1
                cache_targets_by_key.setdefault(
                    key,
                    (
                        target_state,
                        f"single_node_exhaustive:{node_id}:{target.proposal_source}",
                    ),
                )
        bounded_cache_candidate_count = 0
        for target in bounded_cache_targets:
            target_state = np.asarray(target.target_state, dtype=np.int8)
            if np.array_equal(target_state, cache_state):
                continue
            bounded_cache_candidate_count += 1
            cache_proposal_count += 1
            key = np.ascontiguousarray(target_state, dtype=np.int8).tobytes()
            cache_targets_by_key.setdefault(
                key,
                (target_state.copy(), f"bounded_proposal:{target.proposal_source}"),
            )
        pair_targets, pair_diagnostics = _bounded_cross_node_pair_cache_targets(
            cache_state,
            exact_targets_by_node,
            tuple(str(item) for item in cache_nodes["node_ids"]),
            tuple(
                int(item)
                for item in candidate_input.model.service_catalog["service_ids"]
            ),
            candidate_input.model.service_catalog["service_sizes_mb"],
            cache_nodes["capacities_mb"],
            source_state_fingerprint=source_state_fingerprint,
            existing_target_keys=set(cache_targets_by_key),
        )
        for target_state, proposal_source in pair_targets:
            cache_proposal_count += 1
            key = np.ascontiguousarray(target_state, dtype=np.int8).tobytes()
            cache_targets_by_key.setdefault(
                key, (target_state.copy(), proposal_source)
            )
        cache_candidates_deduplicated_count = (
            cache_proposal_count
            - len(cache_targets_by_key)
            + int(
                pair_diagnostics["cache_cross_node_pair_deduplicated_existing_count"]
            )
            + int(
                pair_diagnostics["cache_cross_node_pair_deduplicated_pair_count"]
            )
        )
        protected = {
            str(node_id): {int(service_id) for service_id in service_ids}
            for node_id, service_ids in (protected_services_by_node or {}).items()
        }
        node_indices = {
            str(node_id): index
            for index, node_id in enumerate(cache_nodes["node_ids"])
        }
        service_indices = {
            int(service_id): index
            for index, service_id in enumerate(service_catalog["service_ids"])
        }
        protection_rejected_count = 0
        for key, (target_state, _) in tuple(cache_targets_by_key.items()):
            violates_protection = any(
                node_id not in node_indices
                or service_id not in service_indices
                or int(
                    target_state[
                        node_indices[node_id], service_indices[service_id]
                    ]
                ) != 1
                for node_id, service_ids in protected.items()
                for service_id in service_ids
            )
            if violates_protection:
                cache_targets_by_key.pop(key)
                protection_rejected_count += 1
        forbidden_tail_candidate_count = 0
        if not cache_updates_allowed:
            forbidden_tail_candidate_count = len(cache_targets_by_key)
            cache_targets_by_key.clear()
        for target_state, proposal_source in cache_targets_by_key.values():
            action = ActionCandidate.from_states(
                previous_cache_state=cache_state,
                previous_uav_positions=uav_positions,
                target_cache_state=target_state,
                target_uav_positions=uav_positions,
                slot_input=candidate_input,
                source_state_fingerprint=source_state_fingerprint,
                proposal_source=proposal_source,
            )
            if action.operation != "CACHE_UPDATE":
                raise RuntimeError("缓存候选枚举产生了非 REPLACE 或无变化动作")
            candidates.append(
                CandidateOption.create(
                    action,
                    self.encoder.encode_candidate(
                        action,
                        previous_cache_state=cache_state,
                        previous_uav_positions=uav_positions,
                        slot_input=candidate_input,
                        decision_forecast=forecast,
                        cache_complementarity_context=cache_complementarity_context,
                    ),
                )
            )
        uav_generation_diagnostics: dict[str, Any] = {}
        moves, uav_generation_complete = self.uav_scheduler.enumerate_move_candidates(
            previous_cache_state=cache_state,
            current_uav_positions=uav_positions,
            previous_uav_positions=uav_positions,
            slot_input=candidate_input,
            search_budget=self.search_budget,
            generation_diagnostics=uav_generation_diagnostics,
        )
        for move in moves:
            action = ActionCandidate.from_states(
                previous_cache_state=cache_state,
                previous_uav_positions=uav_positions,
                target_cache_state=cache_state,
                target_uav_positions=move.target_uav_positions,
                slot_input=candidate_input,
                source_state_fingerprint=source_state_fingerprint,
                proposal_source=move.proposal_source,
            )
            candidates.append(
                CandidateOption.create(
                    action,
                    self.encoder.encode_candidate(
                        action,
                        previous_cache_state=cache_state,
                        previous_uav_positions=uav_positions,
                        slot_input=candidate_input,
                        decision_forecast=forecast,
                        cache_complementarity_context=cache_complementarity_context,
                    ),
                )
            )
        unique: dict[str, CandidateOption] = {}
        duplicate_action_id_count = 0
        for option in candidates:
            if option.action.source_state_fingerprint != source_state_fingerprint:
                raise RuntimeError("候选动作来源状态与当前环境状态不一致")
            if option.action.candidate_id in unique:
                duplicate_action_id_count += 1
                continue
            unique[option.action.candidate_id] = option
        if not unique or next(iter(unique.values())).action.operation != "KEEP_STAY":
            raise RuntimeError("合法候选集合必须始终包含 KEEP_STAY")
        unique_cache_targets = len(cache_targets_by_key)
        cache_generation_diagnostics = {
            **cache_generation_diagnostics,
            "raw_combination_count": int(raw_combination_count),
            "capacity_feasible_count": int(capacity_feasible_count),
            "legal_replace_count": int(legal_replace_count),
            "cache_raw_combination_count": int(raw_combination_count),
            "cache_capacity_feasible_count": int(capacity_feasible_count),
            "cache_legal_replace_count": int(legal_replace_count),
            "cache_candidates_retained_count": int(unique_cache_targets),
            "cache_candidate_target_proposals_materialized_count": int(
                cache_proposal_count
            ),
            "cache_candidate_target_unique_count": int(unique_cache_targets),
            "cache_candidates_deduplicated_count": int(
                cache_candidates_deduplicated_count
            ),
            "cache_candidates_deduplicated_by_target_matrix_count": int(
                cache_candidates_deduplicated_count
            ),
            # 此字段专指已经枚举的全部合法单节点 REPLACE，要求其截断数为零。
            "cache_candidates_truncated_count": 0,
            "cache_single_node_candidates_truncated_count": 0,
            "cache_single_node_candidate_counts": single_node_counts,
            "cache_single_node_generation_diagnostics": single_node_diagnostics,
            "cache_single_node_enumeration_complete": True,
            "cache_bounded_proposal_candidate_count": int(
                bounded_cache_candidate_count
            ),
            "cache_bounded_proposal_truncated_count": int(
                cache_generation_diagnostics.get(
                    "omitted_generated_proposal_count", 0
                )
            ),
            "cache_cross_node_candidates_retained_count": sum(
                int(np.count_nonzero(np.any(target_state != cache_state, axis=1))) > 1
                for target_state, _ in cache_targets_by_key.values()
            ),
            "cache_cross_node_combinations_exhaustive": False,
            **pair_diagnostics,
            "cache_bounded_generator_complete": bool(bounded_cache_complete),
            "cache_generation_contract_version": CANDIDATE_GENERATION_CONTRACT_VERSION,
            "cache_updates_allowed": bool(cache_updates_allowed),
            "cache_candidates_rejected_active_task_protection": int(
                protection_rejected_count
            ),
            "cache_candidates_removed_after_arrival_horizon": int(
                forbidden_tail_candidate_count
            ),
        }
        diagnostics = {
            "candidate_count": len(unique),
            "candidate_action_id_deduplicated_count": int(
                duplicate_action_id_count
            ),
            "candidate_count_by_operation": {
                operation: sum(item.action.operation == operation for item in unique.values())
                for operation in ("KEEP_STAY", "CACHE_UPDATE", "UAV_MOVE")
            },
            "cache_candidate_generation_complete": bool(
                bounded_cache_complete
                and pair_diagnostics[
                    "cache_cross_node_pair_generation_complete_within_policy"
                ]
            ),
            "uav_candidate_generation_complete": bool(uav_generation_complete),
            "candidate_generation_complete_within_bounded_proposal_policy": bool(
                bounded_cache_complete
                and pair_diagnostics[
                    "cache_cross_node_pair_generation_complete_within_policy"
                ]
                and uav_generation_complete
            ),
            "candidate_legal_action_space_exhaustive": False,
            "candidate_generation_contract_version": CANDIDATE_GENERATION_CONTRACT_VERSION,
            "cache_candidate_generation_diagnostics": cache_generation_diagnostics,
            "uav_candidate_generation_diagnostics": uav_generation_diagnostics,
            "dqn_adapter_additional_candidate_truncation": bool(
                pair_diagnostics["cache_cross_node_pair_candidates_truncated"]
            ),
            "dqn_adapter_cache_candidate_limit": int(
                self.search_budget.max_cache_candidates
            ),
            "dqn_adapter_uav_candidates_per_uav_limit_including_stay": int(
                self.search_budget.max_uav_candidates_per_uav
            ),
            "candidate_generation_used_matching_or_kkt": False,
            "candidate_generation_received_actual_tasks": False,
            "candidate_generation_received_actual_vehicle_positions": False,
            "state_features_used_current_actual_task_data": bool(
                self._active_task_state_snapshots()
            ),
            "state_features_used_current_actual_vehicle_positions": False,
            "candidate_features_used_current_actual_vehicle_positions": False,
            "decision_state_feature_contract_version": self.encoder.state_feature_contract_version,
            "decision_candidate_feature_contract_version": self.encoder.candidate_feature_contract_version,
            "prediction_horizon_count": self.encoder.horizon_count,
            "prediction_horizon_mask": self.encoder._forecast_mask(forecast).tolist(),
        }
        return tuple(unique.values()), diagnostics

    def _formal_observation(self, forecast: DecisionForecast) -> np.ndarray:
        """复用共享系统观测构造器，输入对象仅含当前决策可见的预测。"""

        from common.environment_interface import build_step_observation

        time_to_go_override = None
        if self.context.observation_spec.time_to_go_slice is not None:
            final_origin = self.execution_horizon_slots - 1
            current_origin = int(forecast.origin_index)
            if current_origin < 0 or current_origin > final_origin:
                raise ValueError(
                    "decision forecast origin is outside the extended execution "
                    f"horizon: origin={current_origin}, final={final_origin}"
                )
            time_to_go_override = (
                0.0
                if final_origin == 0
                else (final_origin - current_origin) / final_origin
            )
        return build_step_observation(
            self.cache_state,
            self.uav_positions,
            forecast,
            self.context.observation_spec,
            tuple(self.model.flight_boundary),
            int(self.context.train_origin_start),
            int(self.context.train_origin_end),
            time_to_go_override=time_to_go_override,
        )

    def candidate_rebuild_context(self) -> dict[str, Any]:
        """导出由合法状态和预测输入组成的紧凑候选重建上下文。"""

        if not self.can_act:
            raise StopIteration("当前状态没有可重建的合法动作集合")
        self._prepare_observation()
        if (
            self._cached_state_features is None
            or self._cached_state_fingerprint is None
            or self._cached_candidates is None
        ):
            raise RuntimeError("当前状态候选尚未完成生成")
        slot_input = self.slot_inputs[self.index]
        forecast = self.decision_forecasts[self.index]
        scenario_slice = dict(slot_input.scenario_slice)
        missing_scenario_fields = [
            key for key in _CANDIDATE_REBUILD_SCENARIO_FIELDS if key not in scenario_slice
        ]
        if missing_scenario_fields:
            raise ValueError(
                "候选重建预测输入缺少字段：" + ", ".join(missing_scenario_fields)
            )
        context = {
            "schema_version": CANDIDATE_REBUILD_CONTEXT_SCHEMA_VERSION,
            "source_state_fingerprint": self._cached_state_fingerprint,
            "episode_seed": self.episode_seed,
            "episode_rebuild_id": self._episode_rebuild_id,
            "episode_rebuild_spec_sha256": self._episode_rebuild_spec_sha256,
            "episode_vehicle_ids": self.encoder.source_vehicle_ids,
            "episode_task_vehicle_ids": self.encoder.source_task_vehicle_ids,
            "episode_cache_node_ids": self.encoder.source_node_ids,
            "canonical_vehicle_ids": self.canonical_vehicle_ids,
            "canonical_cache_node_ids": self.canonical_cache_node_ids,
            "state_features": self._cached_state_features.copy(),
            "cache_state": self.cache_state.copy(),
            "uav_positions": {
                key: value.copy() for key, value in self.uav_positions.items()
            },
            "slot_id": int(slot_input.slot_id),
            "time_s": None if slot_input.time_s is None else float(slot_input.time_s),
            # 候选生成只读取这些预测字段；完整预测场景仍通过摘要进入状态身份。
            "prediction_scenario_sha256": _prediction_scenario_sha256(scenario_slice),
            "prediction_scenario": {
                key: deepcopy(scenario_slice[key])
                for key in _CANDIDATE_REBUILD_SCENARIO_FIELDS
            },
            "decision_forecast": deepcopy(forecast),
            "prediction_horizon_mask": (
                None
                if slot_input.prediction_horizon_mask is None
                else np.asarray(slot_input.prediction_horizon_mask, dtype=np.int8).copy()
            ),
            "protected_services_by_node": {
                str(node_id): tuple(int(value) for value in service_ids)
                for node_id, service_ids in self.task_lifecycle.protected_services_by_node.items()
            },
            "active_task_states": self._active_task_state_snapshots(),
            "cache_updates_allowed": bool(
                int(slot_input.slot_id) + 1 <= self.arrival_horizon_slots
            ),
            "state_feature_contract_version": self.encoder.state_feature_contract_version,
            "candidate_feature_contract_version": self.encoder.candidate_feature_contract_version,
            "candidate_generation_contract_version": CANDIDATE_GENERATION_CONTRACT_VERSION,
            "candidate_capacity_limits": self._candidate_capacity_limits(),
            "candidate_generation_contract": candidate_generation_contract(
                **self._candidate_capacity_limits()
            ),
            "network_architecture_version": self.encoder.network_architecture_version,
            "candidate_count": len(self._cached_candidates),
            "candidate_ids_sha256": _candidate_ids_sha256(self._cached_candidates),
        }
        context["candidate_rebuild_payload_sha256"] = (
            _candidate_rebuild_payload_sha256(context)
        )
        return context

    def candidate_rebuild_adapter(self) -> "EpisodeCandidateRebuildAdapter":
        """创建不保留完整 Episode 数据的回放候选重建器。"""

        return EpisodeCandidateRebuildAdapter.from_environment(self)

    def rebuild_candidates(
        self,
        context: Mapping[str, Any],
    ) -> tuple[CandidateOption, ...]:
        """仅由回放中的合法状态上下文和预测输入确定性重建候选。"""

        self.validate_candidate_rebuild_context_metadata(context)
        forecast = context.get("decision_forecast")
        cache_state = np.asarray(context.get("cache_state"), dtype=np.int8).copy()
        uav_positions = {
            str(key): np.asarray(value, dtype=np.float64).copy()
            for key, value in dict(context.get("uav_positions", {})).items()
        }
        state_features = np.asarray(
            context.get("state_features"), dtype=np.float32
        ).reshape(-1)
        scenario_slice = deepcopy(dict(context.get("prediction_scenario", {})))
        slot_input = SlotInput(
            slot_id=int(context["slot_id"]),
            time_s=(
                None if context.get("time_s") is None else float(context["time_s"])
            ),
            scenario_slice=scenario_slice,
            tasks=(),
            vehicle_positions={},
            predicted_vehicle_positions=deepcopy(
                dict(forecast.predicted_vehicle_positions)
            ),
            model=self.model,
            prediction_horizon_mask=(
                None
                if context.get("prediction_horizon_mask") is None
                else np.asarray(context["prediction_horizon_mask"], dtype=np.int8).copy()
            ),
            scenario_id=f"replay:{int(context['slot_id'])}",
            source_kind="forecast",
            prediction_metadata={"source": "stored_decision_prediction_context"},
        )
        fingerprint = str(context["source_state_fingerprint"])
        protected_services = {
            str(node_id): tuple(int(value) for value in service_ids)
            for node_id, service_ids in dict(
                context["protected_services_by_node"]
            ).items()
        }
        cache_updates_allowed = bool(context["cache_updates_allowed"])
        rebuilt_fingerprint = self._decision_state_fingerprint(
            slot_input,
            forecast,
            state_features,
            cache_state=cache_state,
            uav_positions=uav_positions,
            prediction_scenario_sha256=str(
                context.get("prediction_scenario_sha256", "")
            ),
            protected_services_by_node=protected_services,
            cache_updates_allowed=cache_updates_allowed,
        )
        if rebuilt_fingerprint != fingerprint:
            raise ValueError("回放候选上下文重建的状态指纹不匹配")
        canonical_observation = state_features[
            : int(self.context.observation_spec.observation_dim)
        ]
        formal_observation = self.encoder.restore_source_formal_observation(
            canonical_observation
        )
        reencoded_state = self.encoder.encode_state(
            formal_observation,
            forecast,
            scenario_slice,
            tuple(dict(item) for item in context["active_task_states"]),
        )
        if not np.array_equal(reencoded_state, state_features):
            raise ValueError("回放状态向量与保存的预测候选生成上下文不一致")
        cache_key = (
            str(context["candidate_rebuild_payload_sha256"]),
            fingerprint,
            str(context["state_feature_contract_version"]),
            str(context["candidate_feature_contract_version"]),
            str(context["network_architecture_version"]),
        )
        candidates = self._candidate_rebuild_cache.get(cache_key)
        if candidates is None:
            self._candidate_rebuild_cache_stats["miss_count"] += 1
            candidates, _ = self._generate_legal_candidates(
                slot_input,
                forecast,
                fingerprint,
                current_cache_state=cache_state,
                current_uav_positions=uav_positions,
                protected_services_by_node=protected_services,
                cache_updates_allowed=cache_updates_allowed,
            )
            self._candidate_rebuild_cache_stats["generation_count"] += 1
        else:
            self._candidate_rebuild_cache.move_to_end(cache_key)
            self._candidate_rebuild_cache_stats["hit_count"] += 1
        if (
            len(candidates) != int(context["candidate_count"])
            or _candidate_ids_sha256(candidates)
            != str(context["candidate_ids_sha256"])
            or len({item.action.candidate_id for item in candidates}) != len(candidates)
            or any(
                item.action.source_state_fingerprint != fingerprint
                or len(item.features) != self.encoder.candidate_dim
                or not np.isfinite(np.asarray(item.features, dtype=np.float32)).all()
                for item in candidates
            )
        ):
            raise ValueError("回放候选集合与经过校验的上下文身份不匹配")
        if candidates is not None and cache_key not in self._candidate_rebuild_cache:
            if len(candidates) > _CANDIDATE_REBUILD_CACHE_MAX_CANDIDATES:
                self._candidate_rebuild_cache_stats["bypass_count"] += 1
            else:
                self._candidate_rebuild_cache[cache_key] = candidates
                self._candidate_rebuild_cache_candidate_count += len(candidates)
                while (
                    len(self._candidate_rebuild_cache)
                    > _CANDIDATE_REBUILD_CACHE_MAX_ENTRIES
                    or self._candidate_rebuild_cache_candidate_count
                    > _CANDIDATE_REBUILD_CACHE_MAX_CANDIDATES
                ):
                    _, evicted = self._candidate_rebuild_cache.popitem(last=False)
                    self._candidate_rebuild_cache_candidate_count -= len(evicted)
                    self._candidate_rebuild_cache_stats["eviction_count"] += 1
        return candidates

    def validate_candidate_rebuild_context_metadata(
        self, context: Mapping[str, Any]
    ) -> None:
        """在恢复 Replay 前校验候选上下文版本、摘要和生成合同。"""

        if context.get("schema_version") != CANDIDATE_REBUILD_CONTEXT_SCHEMA_VERSION:
            raise ValueError(
                "回放候选重建上下文版本不兼容："
                f"stored={context.get('schema_version')!r}, "
                f"required={CANDIDATE_REBUILD_CONTEXT_SCHEMA_VERSION!r}"
            )
        missing_fields = [
            key for key in _CANDIDATE_REBUILD_IDENTITY_FIELDS if key not in context
        ]
        if missing_fields:
            raise ValueError(
                "回放候选重建上下文缺少身份字段：" + ", ".join(missing_fields)
            )
        if context.get("candidate_rebuild_payload_sha256") != (
            _candidate_rebuild_payload_sha256(context)
        ):
            raise ValueError("回放候选重建上下文摘要不匹配")
        if context.get("candidate_feature_contract_version") != self.encoder.candidate_feature_contract_version:
            raise ValueError("回放候选特征合同与当前编码器不一致")
        if context.get("candidate_generation_contract_version") != CANDIDATE_GENERATION_CONTRACT_VERSION:
            raise ValueError("回放候选生成合同与当前生成器不一致")
        expected_capacity_limits = self._candidate_capacity_limits()
        if context.get("candidate_capacity_limits") != expected_capacity_limits:
            raise ValueError(
                "回放候选容量限制与当前训练合同不一致："
                f"stored={context.get('candidate_capacity_limits')!r}, "
                f"expected={expected_capacity_limits!r}"
            )
        expected_generation_contract = candidate_generation_contract(
            **expected_capacity_limits
        )
        if context.get("candidate_generation_contract") != expected_generation_contract:
            raise ValueError("回放候选生成详细合同与当前实现不一致")
        if context.get("state_feature_contract_version") != self.encoder.state_feature_contract_version:
            raise ValueError("回放状态特征合同与当前编码器不一致")
        if context.get("network_architecture_version") != self.encoder.network_architecture_version:
            raise ValueError("回放网络架构合同与当前编码器不一致")
        if context.get("episode_rebuild_id") != self._episode_rebuild_id:
            raise ValueError("回放候选属于其他 Episode 重建身份")
        if context.get("episode_rebuild_spec_sha256") != self._episode_rebuild_spec_sha256:
            raise ValueError("回放 Episode 重建规格摘要不匹配")
        if context.get("episode_seed") != self.episode_seed:
            raise ValueError("回放 Episode 种子与重建规格不匹配")
        if (
            tuple(context.get("episode_vehicle_ids", ()))
            != self.encoder.source_vehicle_ids
            or tuple(context.get("episode_task_vehicle_ids", ()))
            != self.encoder.source_task_vehicle_ids
            or tuple(context.get("episode_cache_node_ids", ()))
            != self.encoder.source_node_ids
            or tuple(context.get("canonical_vehicle_ids", ()))
            != self.canonical_vehicle_ids
            or tuple(context.get("canonical_cache_node_ids", ()))
            != self.canonical_cache_node_ids
        ):
            raise ValueError("回放候选的车辆/缓存节点映射与 Episode 重建规格不匹配")
        if not isinstance(context.get("decision_forecast"), DecisionForecast):
            raise ValueError("回放候选重建缺少正式 DecisionForecast")
        candidate_count = int(context.get("candidate_count", 0))
        candidate_ids_sha256 = str(context.get("candidate_ids_sha256", ""))
        prediction_scenario_sha256 = str(
            context.get("prediction_scenario_sha256", "")
        )
        try:
            bytes.fromhex(candidate_ids_sha256)
            bytes.fromhex(prediction_scenario_sha256)
        except ValueError:
            raise ValueError("回放候选上下文哈希字段不是有效十六进制摘要") from None
        if (
            not str(context.get("source_state_fingerprint", ""))
            or candidate_count <= 0
            or len(candidate_ids_sha256) != 64
            or len(prediction_scenario_sha256) != 64
        ):
            raise ValueError("回放候选上下文缺少候选数量或顺序指纹")

    def _decision_state_fingerprint(
        self,
        slot_input: SlotInput,
        forecast: DecisionForecast,
        state_features: np.ndarray,
        *,
        cache_state: Any | None = None,
        uav_positions: Mapping[str, Any] | None = None,
        prediction_scenario_sha256: str | None = None,
        protected_services_by_node: Mapping[str, Sequence[int]] | None = None,
        cache_updates_allowed: bool | None = None,
    ) -> str:
        if protected_services_by_node is None and hasattr(self, "task_lifecycle"):
            protected_services_by_node = self.task_lifecycle.protected_services_by_node
        protected = {
            str(node_id): tuple(sorted(int(value) for value in service_ids))
            for node_id, service_ids in (protected_services_by_node or {}).items()
        }
        if cache_updates_allowed is None:
            cache_updates_allowed = (
                int(slot_input.slot_id) + 1 <= int(self.arrival_horizon_slots)
                if hasattr(self, "arrival_horizon_slots")
                else True
            )
        prediction_context = {
            "origin_index": int(forecast.origin_index),
            "slot": int(forecast.slot),
            "prediction_horizon_mask": (
                None
                if forecast.prediction_horizon_mask is None
                else np.asarray(forecast.prediction_horizon_mask, dtype=np.int8)
            ),
            # 完整预测场景摘要保留指纹语义，回放只需保存候选生成必需字段。
            "prediction_scenario_sha256": (
                _prediction_scenario_sha256(slot_input.scenario_slice)
                if prediction_scenario_sha256 is None
                else str(prediction_scenario_sha256)
            ),
            "predicted_vehicle_positions_horizon": forecast.predicted_vehicle_positions_horizon,
            "predicted_service_popularity_horizon": forecast.predicted_service_popularity_horizon,
            "protected_services_by_node": protected,
            "cache_updates_allowed": bool(cache_updates_allowed),
        }
        return _state_fingerprint(
            self.cache_state if cache_state is None else cache_state,
            self.uav_positions if uav_positions is None else uav_positions,
            state_features,
            prediction_context,
        )

    def step(
        self,
        option: CandidateOption,
        *,
        on_phase: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> EnvironmentTransition:
        """只把所选候选送入正式执行器，并推进一个真实 Slot。"""

        slot_id = (
            int(self.slot_inputs[self.index].slot_id)
            if self.can_act and self.index < len(self.slot_inputs)
            else None
        )

        def phase(name: str, event: str, **details: Any) -> None:
            if on_phase is not None:
                on_phase(
                    {
                        "phase": name,
                        "event": event,
                        "slot_id": slot_id,
                        **details,
                    }
                )

        validation_started = time.perf_counter()
        phase("environment.step.action_validation", "started")
        state, legal_candidates = self.observe()
        if not isinstance(option, CandidateOption):
            raise TypeError("step 只接受当前状态的 CandidateOption")
        match = next(
            (
                item
                for item in legal_candidates
                if item.action.candidate_id == option.action.candidate_id
            ),
            None,
        )
        if match is None or match.action.to_dict() != option.action.to_dict():
            raise ValueError("所选动作不是当前状态下生成的合法候选")
        candidate_generation_diagnostics = dict(self._candidate_generation_diagnostics)
        action = match.action
        if action.source_state_fingerprint != self._current_state_hash():
            raise ValueError("候选动作属于其他状态，拒绝执行")
        phase(
            "environment.step.action_validation",
            "completed",
            duration_s=time.perf_counter() - validation_started,
            legal_candidate_count=len(legal_candidates),
        )
        preparation_started = time.perf_counter()
        phase("environment.step.prepare_execution", "started")
        previous_cache = self.cache_state.copy()
        previous_uav = {key: value.copy() for key, value in self.uav_positions.items()}
        target_cache = action.cache_array()
        target_uav = action.uav_dict()
        changed_cache = not np.array_equal(previous_cache, target_cache)
        moved_uav = any(
            not np.array_equal(previous_uav[uav_id], target_uav[uav_id])
            for uav_id in previous_uav
        )
        if changed_cache != bool(action.changed_node_ids):
            raise RuntimeError("候选 changed_node_ids 与缓存目标差异不一致")
        if moved_uav != bool(action.moved_uav_ids):
            raise RuntimeError("候选 moved_uav_ids 与 UAV 目标差异不一致")
        if changed_cache and moved_uav:
            raise ValueError("候选动作违反缓存与 UAV 互斥约束")
        slot_input = self.slot_inputs[self.index]
        decision_task_state_count = len(self._active_task_state_snapshots())
        active_vehicle_ids = {
            str(state["vehicle_id"])
            for state in self._active_task_state_snapshots()
        } | {str(task.vehicle_id) for task in slot_input.tasks}
        _validate_active_task_vehicle_positions(
            slot_id=int(slot_input.slot_id),
            arrival_horizon_slots=self.arrival_horizon_slots,
            active_vehicle_ids=tuple(active_vehicle_ids),
            vehicle_positions=slot_input.vehicle_positions,
        )
        prepared_task_slot = self.task_lifecycle.prepare_slot(
            int(slot_input.slot_id), tuple(slot_input.tasks)
        )
        input_hash = _input_fingerprint(slot_input)
        context_cache_before = np.asarray(self.context.initial_cache, dtype=np.int8).copy()
        context_uav_before = {
            key: np.asarray(value, dtype=np.float64).copy()
            for key, value in self.context.initial_uav.items()
        }
        cache_triggers = np.any(target_cache != previous_cache, axis=1).astype(np.int8)
        tolerance = float(
            self.model.project_cfg.get("algorithm", {})
            .get("uav", {})
            .get("position_tolerance_m", 1.0e-6)
        )
        uav_triggers = {
            uav_id: int(float(np.linalg.norm(target_uav[uav_id] - previous_uav[uav_id])) > tolerance)
            for uav_id in previous_uav
        }
        cache_configuration.validate_cache_shape_and_capacity(
            target_cache,
            dict(self.model.service_catalog),
            dict(self.model.cache_nodes),
        )
        cache_replace_pass = all(
            cache_configuration.is_legal_cache_replacement(
                previous_cache[node_index], target_cache[node_index]
            )
            for node_index, node_id in enumerate(self.model.cache_nodes["node_ids"])
            if str(node_id) in action.changed_node_ids
        )
        if not cache_replace_pass:
            raise ValueError("候选缓存差异违反正式 REPLACE 规则")
        flight_boundary = tuple(float(value) for value in self.model.flight_boundary)
        uav_boundary_pass = all(
            uav_configuration._inside_boundary(
                np.asarray(position, dtype=np.float64), flight_boundary
            )
            for position in target_uav.values()
        )
        uav_search_config = uav_configuration.uav_search_config_from_project(
            dict(self.model.project_cfg)
        )
        uav_safety_distance_pass = all(
            uav_configuration._satisfies_min_safety_distance(
                uav_id,
                target_uav[uav_id],
                target_uav,
                uav_search_config,
            )
            for uav_id in target_uav
        )
        if not uav_boundary_pass:
            raise ValueError("候选 UAV 目标越过正式飞行边界")
        if not uav_safety_distance_pass:
            raise ValueError("候选 UAV 目标违反正式安全距离")
        uav_configuration.validate_uav_configuration_transition(
            previous_uav,
            target_uav,
            uav_triggers,
            dict(self.model.project_cfg),
            flight_boundary,
        )
        phase(
            "environment.step.prepare_execution",
            "completed",
            duration_s=time.perf_counter() - preparation_started,
            prepared_task_count=len(prepared_task_slot.tasks),
            mutable_task_count=len(prepared_task_slot.mutable_task_ids),
        )
        kkt_cache = task_allocation.KKTResourceAllocationCache()
        performance_profile: dict[str, float] = {}
        execution_started = time.perf_counter()
        phase(
            "environment.step.matching_kkt_execution",
            "started",
            prepared_task_count=len(prepared_task_slot.tasks),
            mutable_task_count=len(prepared_task_slot.mutable_task_ids),
            legal_candidate_count=len(legal_candidates),
        )
        result = slot_environment.execute_configured_slot(
            previous_cache,
            previous_uav,
            target_cache,
            target_uav,
            dict(slot_input.scenario_slice),
            list(prepared_task_slot.tasks),
            dict(slot_input.vehicle_positions),
            dict(self.model.service_catalog),
            dict(self.model.project_cfg),
            dict(self.model.cache_nodes),
            self.model.cache_config,
            flight_boundary,
            slot_id=int(slot_input.slot_id),
            time_s=slot_input.time_s,
            performance_profile=performance_profile,
            matching_kkt_cache=kkt_cache,
            node_evaluation_cache=task_allocation.NodeEvaluationCache(),
            lifecycle_mode=True,
            fixed_matching=dict(prepared_task_slot.fixed_matching),
            mutable_task_ids=set(prepared_task_slot.mutable_task_ids),
            protected_services_by_node=dict(
                prepared_task_slot.protected_services_by_node
            ),
        )
        phase(
            "environment.step.matching_kkt_execution",
            "completed",
            duration_s=time.perf_counter() - execution_started,
            kkt_solver_calls=int(kkt_cache.solver_calls),
            matching_produced=bool(result.matching_summary),
        )
        if not result.step_feasible:
            raise RuntimeError(
                "正式 Slot 执行未产生可用转移："
                f"slot={slot_input.slot_id}, reason={result.infeasible_reason}"
            )
        if result.next_cache_state is None or result.next_uav_positions is None:
            raise RuntimeError("正式 Slot 执行器未返回下一缓存/UAV 状态")
        postprocess_started = time.perf_counter()
        phase("environment.step.lifecycle_and_postprocess", "started")
        lifecycle_result = self.task_lifecycle.advance_slot(
            int(slot_input.slot_id),
            result.matching,
            result.per_task_upload_bits,
            result.per_task_compute_cycles,
            result.per_task_slot_upload_energy_j,
        )
        terminal_task_utility_by_id: dict[str, float] = {}
        terminal_task_events: list[dict[str, Any]] = []
        for raw_event in lifecycle_result.termination_events:
            event = dict(raw_event)
            task_id = str(event["task_id"])
            if task_id in self._settled_task_ids:
                raise RuntimeError(f"terminal utility was already settled for task {task_id}")
            utility = system_model.terminal_task_utility(
                event, **self.terminal_utility_parameters
            )
            self._settled_task_ids.add(task_id)
            terminal_task_utility_by_id[task_id] = float(utility)
            event["terminal_utility"] = float(utility)
            terminal_task_events.append(event)
        actual_task_utility = float(
            math.fsum(terminal_task_utility_by_id.values())
        )
        if result.cache_cost is None or result.movement_cost is None:
            raise RuntimeError("lifecycle system reward requires both normalized system costs")
        slot_reward = system_model.lifecycle_system_reward(
            actual_task_utility,
            float(result.cache_cost),
            float(result.movement_cost),
        )
        actual_matching = getattr(result, "matching", None)
        actual_execution_node_data_available = isinstance(actual_matching, Mapping)
        actual_execution_nodes = (
            [
                {
                    "task_id": str(task_id),
                    "execution_node": str(execution_node),
                }
                for task_id, execution_node in sorted(
                    actual_matching.items(), key=lambda item: str(item[0])
                )
            ]
            if actual_execution_node_data_available
            else None
        )
        actual_execution_node_counts = (
            dict(
                sorted(
                    Counter(
                        str(execution_node)
                        for execution_node in actual_matching.values()
                    ).items()
                )
            )
            if actual_execution_node_data_available
            else None
        )
        final_cache = np.asarray(result.next_cache_state, dtype=np.int8).copy()
        final_uav = {
            key: np.asarray(value, dtype=np.float64).copy()
            for key, value in result.next_uav_positions.items()
        }
        if not np.array_equal(final_cache, target_cache) or any(
            not np.array_equal(final_uav[key], target_uav[key]) for key in target_uav
        ):
            raise RuntimeError("执行器推进后的配置与已选候选不一致")
        actual_cache_changed = not np.array_equal(previous_cache, final_cache)
        actual_uav_moved = any(
            float(np.linalg.norm(final_uav[key] - previous_uav[key])) > tolerance
            for key in previous_uav
        )
        mutual_exclusion_pass = not (actual_cache_changed and actual_uav_moved)
        if not mutual_exclusion_pass:
            raise RuntimeError("正式执行结果同时改变缓存与 UAV 位置")
        isolation_pass = (
            _input_fingerprint(slot_input) == input_hash
            and np.array_equal(context_cache_before, self.context.initial_cache)
            and all(
                np.array_equal(context_uav_before[key], self.context.initial_uav[key])
                for key in context_uav_before
            )
            and np.array_equal(previous_cache, self.cache_state)
            and all(np.array_equal(previous_uav[key], self.uav_positions[key]) for key in previous_uav)
        )
        if not isolation_pass:
            raise RuntimeError("候选执行意外修改了输入 realization 或执行前状态")
        predicted_cache_coverage_before: float | None = None
        predicted_cache_coverage_after: float | None = None
        predicted_cache_coverage_error: str | None = None
        predicted_service_demand_by_service: dict[int, float] | None = None
        cache_service_demand_details: dict[str, dict[str, list[dict[str, Any]]]] = {}
        cache_services_added: dict[str, list[int]] = {}
        cache_services_removed: dict[str, list[int]] = {}
        service_ids = tuple(
            int(value) for value in self.model.service_catalog["service_ids"]
        )
        for node_index, node_id in enumerate(self.model.cache_nodes["node_ids"]):
            before_services = {
                service_ids[index]
                for index in np.flatnonzero(previous_cache[node_index])
            }
            after_services = {
                service_ids[index]
                for index in np.flatnonzero(final_cache[node_index])
            }
            removed_services = sorted(before_services - after_services)
            added_services = sorted(after_services - before_services)
            if removed_services:
                cache_services_removed[str(node_id)] = removed_services
            if added_services:
                cache_services_added[str(node_id)] = added_services
        try:
            cache_nodes_before = _cache_nodes_at_uav_positions(
                self.model.cache_nodes, previous_uav
            )
            cache_nodes_after = _cache_nodes_at_uav_positions(
                self.model.cache_nodes, final_uav
            )
            predicted_cache_coverage_before = (
                cache_configuration.predicted_cache_coverage_ratio(
                    previous_cache,
                    dict(slot_input.scenario_slice),
                    cache_nodes_before,
                    self.model.cache_config,
                    prediction_horizon_mask=slot_input.prediction_horizon_mask,
                )
            )
            predicted_cache_coverage_after = (
                cache_configuration.predicted_cache_coverage_ratio(
                    final_cache,
                    dict(slot_input.scenario_slice),
                    cache_nodes_after,
                    self.model.cache_config,
                    prediction_horizon_mask=slot_input.prediction_horizon_mask,
                )
            )
            predicted_demand = cache_configuration.predicted_vehicle_service_demand_by_horizon(
                dict(slot_input.scenario_slice),
                len(tuple(self.model.service_catalog["service_ids"])),
                beta_discount=float(self.model.cache_config.beta_discount),
                prediction_horizon_mask=slot_input.prediction_horizon_mask,
            )
            demand_by_service = np.sum(predicted_demand, axis=(0, 1))
            predicted_service_demand_by_service = {
                int(service_id): float(demand_by_service[index])
                for index, service_id in enumerate(
                    self.model.service_catalog["service_ids"]
                )
            }
            replicas_before = np.sum(previous_cache, axis=0).astype(int)
            replicas_after = np.sum(final_cache, axis=0).astype(int)
            for node_index, node_id in enumerate(self.model.cache_nodes["node_ids"]):
                node_id = str(node_id)
                if node_id not in cache_services_added and node_id not in cache_services_removed:
                    continue
                node_demand = cache_configuration.node_specific_predicted_demand(
                    dict(slot_input.scenario_slice),
                    node_index,
                    cache_nodes_before,
                    self.model.cache_config,
                    prediction_horizon_mask=slot_input.prediction_horizon_mask,
                )

                def changed_service_details(service_values: Sequence[int]) -> list[dict[str, Any]]:
                    details: list[dict[str, Any]] = []
                    for service_id in service_values:
                        service_index = tuple(
                            int(value)
                            for value in self.model.service_catalog["service_ids"]
                        ).index(int(service_id))
                        details.append(
                            {
                                "service_id": int(service_id),
                                "predicted_demand_for_node": float(node_demand[service_index]),
                                "predicted_demand_system_total": predicted_service_demand_by_service[
                                    int(service_id)
                                ],
                                "replicas_before": int(replicas_before[service_index]),
                                "replicas_after": int(replicas_after[service_index]),
                            }
                        )
                    return details

                cache_service_demand_details[node_id] = {
                    "removed_services": changed_service_details(
                        cache_services_removed.get(node_id, [])
                    ),
                    "added_services": changed_service_details(
                        cache_services_added.get(node_id, [])
                    ),
                }
        except (KeyError, TypeError, ValueError, IndexError, FloatingPointError) as exc:
            predicted_cache_coverage_error = f"{type(exc).__name__}: {exc}"
            predicted_cache_coverage_before = None
            predicted_cache_coverage_after = None
            predicted_service_demand_by_service = None
            cache_service_demand_details = {}
        phase(
            "environment.step.lifecycle_and_postprocess",
            "completed",
            duration_s=time.perf_counter() - postprocess_started,
            termination_event_count=len(lifecycle_result.termination_events),
            active_task_count=len(self.task_lifecycle.active_tasks),
        )
        executor_calls = 1
        matching_calls = int(bool(result.matching_summary))
        kkt_calls = int(kkt_cache.solver_calls)
        self.executor_calls += executor_calls
        self.matching_calls += matching_calls
        self.kkt_solver_calls += kkt_calls
        self.cache_state = final_cache
        self.uav_positions = final_uav
        self.index += 1
        self._cached_state_fingerprint = None
        self._cached_state_features = None
        self._cached_candidates = None
        done = self.terminated
        if done and self.task_lifecycle.active_tasks:
            raise RuntimeError(
                "扩展执行时域结束时仍有活动任务："
                f"count={len(self.task_lifecycle.active_tasks)}"
            )
        next_state: np.ndarray | None = None
        next_state_fingerprint: str | None = None
        next_candidates: tuple[CandidateOption, ...] = ()
        next_candidate_rebuild_context: Mapping[str, Any] | None = None
        next_observation_started = time.perf_counter()
        phase("environment.step.next_observation_and_candidates", "started")
        if not done:
            next_state, next_candidates = self._prepare_observation()
            next_state_fingerprint = self._current_state_hash()
            next_candidate_rebuild_context = self.candidate_rebuild_context()
        next_candidate_counts = {
            operation: sum(
                candidate.action.operation == operation for candidate in next_candidates
            )
            for operation in ("KEEP_STAY", "CACHE_UPDATE", "UAV_MOVE")
        }
        phase(
            "environment.step.next_observation_and_candidates",
            "completed",
            duration_s=time.perf_counter() - next_observation_started,
            next_slot_id=(
                None
                if done
                else int(self.slot_inputs[self.index].slot_id)
            ),
            next_candidate_count=len(next_candidates),
            next_candidate_counts_by_operation=next_candidate_counts,
            terminated=bool(done),
        )
        return EnvironmentTransition(
            reward=slot_reward,
            next_state_features=next_state,
            next_state_fingerprint=next_state_fingerprint,
            next_legal_candidates=next_candidates,
            next_candidate_rebuild_context=next_candidate_rebuild_context,
            terminated=done,
            duration_slots=1,
            operation=action.operation,
            candidate_id=action.candidate_id,
            final_cache_state=final_cache.copy(),
            final_uav_positions={key: value.copy() for key, value in final_uav.items()},
            executor_calls=executor_calls,
            matching_calls=matching_calls,
            kkt_solver_calls=kkt_calls,
            state_isolation_pass=isolation_pass,
            mutual_exclusion_pass=mutual_exclusion_pass,
            diagnostics={
                "slot_id": int(slot_input.slot_id),
                "candidate_source_state_fingerprint": action.source_state_fingerprint,
                "candidate_current_legal_set_membership_pass": True,
                "candidate_source_state_fingerprint_matches_current_pass": True,
                "candidate_mutual_exclusion_pass": not (changed_cache and moved_uav),
                "candidate_cache_replace_rule_pass": cache_replace_pass,
                "candidate_cache_capacity_pass": True,
                "candidate_uav_boundary_pass": uav_boundary_pass,
                "candidate_uav_safety_distance_pass": uav_safety_distance_pass,
                "candidate_uav_transition_constraints_pass": True,
                "candidate_target_matches_execution_pass": True,
                "pre_cache_state": previous_cache.tolist(),
                "post_cache_state": final_cache.tolist(),
                "pre_uav_positions_m": {
                    key: value.tolist() for key, value in previous_uav.items()
                },
                "post_uav_positions_m": {
                    key: value.tolist() for key, value in final_uav.items()
                },
                "actual_system_utility": slot_reward,
                "actual_task_utility": actual_task_utility,
                "system_utility_formula": "G(t)=Q(t)-C_ca(t)-C_q(t)",
                "system_utility_is_lifecycle_terminal_utility": True,
                "terminal_task_utility_by_task": terminal_task_utility_by_id,
                "terminal_task_cumulative_upload_energy_j": {
                    str(event["task_id"]): float(event["cumulative_upload_energy_j"])
                    for event in terminal_task_events
                },
                "actual_matching_node_task_counts": result.matching_summary.get(
                    "node_task_count"
                ),
                "actual_task_count": (
                    len(actual_matching)
                    if actual_execution_node_data_available
                    else None
                ),
                "arrived_task_count": len(lifecycle_result.arrived_task_ids),
                "arrived_task_ids": list(lifecycle_result.arrived_task_ids),
                "completed_task_count": lifecycle_result.completed_task_count,
                "timed_out_task_count": lifecycle_result.timed_out_task_count,
                "active_task_count": lifecycle_result.active_task_count,
                "active_task_states": [
                    dict(state) for state in lifecycle_result.active_tasks
                ],
                "task_termination_events": terminal_task_events,
                "per_task_slot_execution": {
                    task_id: dict(metrics)
                    for task_id, metrics in lifecycle_result.per_task_slot_metrics.items()
                },
                "per_task_slot_upload_energy_j": dict(
                    result.per_task_slot_upload_energy_j
                ),
                "episode_vehicle_upload_energy_j": (
                    lifecycle_result.episode_upload_energy_j
                ),
                "lifecycle_task_arrivals_closed": (
                    int(slot_input.slot_id) + 1 > self.arrival_horizon_slots
                ),
                "cache_updates_allowed": (
                    int(slot_input.slot_id) + 1 <= self.arrival_horizon_slots
                ),
                "per_task_delay_s": dict(result.per_task_delay_s),
                "per_task_vehicle_energy_j": dict(
                    result.per_task_slot_upload_energy_j
                ),
                "matching_utility_per_task_vehicle_energy_j": dict(
                    result.per_task_vehicle_energy_j
                ),
                "per_task_utility": terminal_task_utility_by_id,
                "matching_proxy_per_task_utility": dict(result.per_task_utility),
                "actual_total_task_delay_s": None,
                "actual_total_vehicle_energy_j": float(
                    lifecycle_result.slot_upload_energy_j
                ),
                "cache_download_delay_s": (
                    None
                    if result.cache_download_delay_s is None
                    else float(result.cache_download_delay_s)
                ),
                "uav_movement_energy_j": (
                    None
                    if result.movement_energy is None
                    else float(result.movement_energy)
                ),
                "predicted_cache_coverage_before": predicted_cache_coverage_before,
                "predicted_cache_coverage_after": predicted_cache_coverage_after,
                "predicted_cache_coverage_delta": (
                    None
                    if predicted_cache_coverage_before is None
                    or predicted_cache_coverage_after is None
                    else predicted_cache_coverage_after
                    - predicted_cache_coverage_before
                ),
                "predicted_cache_coverage_metric": True,
                "predicted_cache_coverage_metric_scope": (
                    "predicted demand weighted cache serviceability; UAV nodes use state positions"
                ),
                "predicted_cache_coverage_error": predicted_cache_coverage_error,
                "predicted_service_demand_by_service": predicted_service_demand_by_service,
                "cache_service_demand_details": cache_service_demand_details,
                "cache_cost": (
                    None if result.cache_cost is None else float(result.cache_cost)
                ),
                "cache_update_cost_available": result.cache_cost is not None,
                "movement_cost": (
                    None
                    if result.movement_cost is None
                    else float(result.movement_cost)
                ),
                "uav_movement_cost_available": result.movement_cost is not None,
                "beta_ca": float(result.beta_ca),
                "beta_q": float(result.beta_q),
                "matching_summary": result.matching_summary,
                "actual_execution_node_counts": actual_execution_node_counts,
                "actual_execution_nodes": actual_execution_nodes,
                "actual_execution_node_data_available": (
                    actual_execution_node_data_available
                ),
                "same_slot_matching_alternative_comparison_available": False,
                "executor_performance_profile": performance_profile,
                "duration_semantics": "每个动作恰好执行一个正式 Slot",
                **candidate_generation_diagnostics,
                "state_features_used_future_actual_data": False,
                "state_features_used_current_actual_task_data": (
                    decision_task_state_count > 0
                ),
                "candidate_generation_received_actual_tasks": False,
                "candidate_generation_received_actual_vehicle_positions": False,
                "candidate_features_used_current_actual_vehicle_positions": False,
                "decision_state_feature_contract_version": self.encoder.state_feature_contract_version,
                "decision_candidate_feature_contract_version": self.encoder.candidate_feature_contract_version,
                "prediction_horizon_count": self.encoder.horizon_count,
                "prediction_horizon_mask": self.encoder._forecast_mask(
                    self.decision_forecasts[self.index - 1]
                ).tolist(),
            },
        )

    def _current_state_hash(self) -> str:
        if self.index >= len(self.slot_inputs):
            return _state_fingerprint(self.cache_state, self.uav_positions)
        slot_input = self.slot_inputs[self.index]
        forecast = self.decision_forecasts[self.index]
        formal_observation = self._formal_observation(forecast)
        state_features = self.encoder.encode_state(
            formal_observation,
            forecast,
            slot_input.scenario_slice,
            self._active_task_state_snapshots(),
        )
        return self._decision_state_fingerprint(
            slot_input, forecast, state_features
        )


class EpisodeCandidateRebuildAdapter:
    """仅保存一个 Episode 的模型、映射和特征合同，不保留任务或轨迹序列。"""

    def __init__(
        self,
        *,
        model: SystemModelContext,
        encoder: CandidateFeatureEncoder,
        observation_spec: Any,
        train_origin_start: int,
        train_origin_end: int,
        search_budget: SearchBudget,
        episode_seed: int | None,
        episode_rebuild_id: str,
        episode_rebuild_spec_sha256: str,
    ) -> None:
        self.model = deepcopy(model)
        self.encoder = deepcopy(encoder)
        self.context = SimpleNamespace(observation_spec=deepcopy(observation_spec))
        self.canonical_vehicle_ids = self.encoder.vehicle_ids
        self.canonical_cache_node_ids = self.encoder.node_ids
        self.train_origin_start = int(train_origin_start)
        self.train_origin_end = int(train_origin_end)
        self.search_budget = deepcopy(search_budget)
        self.episode_seed = None if episode_seed is None else int(episode_seed)
        self._episode_rebuild_id = str(episode_rebuild_id)
        self._episode_rebuild_spec_sha256 = str(episode_rebuild_spec_sha256)
        self.uav_scheduler = UavScheduler()
        self.cache_optimizer = CacheOptimizer(self.uav_scheduler)
        self._candidate_rebuild_cache: OrderedDict[
            tuple[str, ...], tuple[CandidateOption, ...]
        ] = OrderedDict()
        self._candidate_rebuild_cache_candidate_count = 0
        self._candidate_rebuild_cache_stats = {
            "hit_count": 0,
            "miss_count": 0,
            "generation_count": 0,
            "eviction_count": 0,
            "bypass_count": 0,
        }

    @staticmethod
    def _active_task_state_snapshots() -> tuple[dict[str, Any], ...]:
        """Candidate rebuilding has no live task state to expose to generation.

        The shared candidate generator calls this only to populate a diagnostic
        flag. Replay reconstruction uses the task snapshots stored in its
        validated context when re-encoding the observation, while candidate
        generation receives an input with tasks stripped.
        """

        return ()

    @staticmethod
    def _identity_payload(environment: ExclusiveDQNEnvironment) -> dict[str, Any]:
        return {
            "schema_version": EPISODE_REBUILD_SPEC_SCHEMA_VERSION,
            "episode_seed": environment.episode_seed,
            "model": environment.model,
            "encoder": environment.encoder,
            "observation_spec": environment.context.observation_spec,
            "train_origin_start": int(environment.context.train_origin_start),
            "train_origin_end": int(environment.context.train_origin_end),
            "search_budget": environment.search_budget,
            "episode_vehicle_ids": environment.encoder.source_vehicle_ids,
            "episode_task_vehicle_ids": environment.encoder.source_task_vehicle_ids,
            "episode_cache_node_ids": environment.encoder.source_node_ids,
            "canonical_vehicle_ids": environment.canonical_vehicle_ids,
            "canonical_cache_node_ids": environment.canonical_cache_node_ids,
            "state_feature_contract_version": environment.encoder.state_feature_contract_version,
            "candidate_feature_contract_version": environment.encoder.candidate_feature_contract_version,
            "candidate_generation_contract_version": CANDIDATE_GENERATION_CONTRACT_VERSION,
            "network_architecture_version": environment.encoder.network_architecture_version,
        }

    @classmethod
    def identity_for_environment(
        cls, environment: ExclusiveDQNEnvironment
    ) -> tuple[str, str]:
        payload = cls._identity_payload(environment)
        spec_sha256 = _episode_rebuild_spec_sha256(payload)
        episode_identity = hashlib.sha256(
            pickle.dumps(
                (environment.episode_seed, spec_sha256),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        ).hexdigest()
        return episode_identity, spec_sha256

    @classmethod
    def from_environment(
        cls, environment: ExclusiveDQNEnvironment
    ) -> "EpisodeCandidateRebuildAdapter":
        return cls(
            model=environment.model,
            encoder=environment.encoder,
            observation_spec=environment.context.observation_spec,
            train_origin_start=environment.context.train_origin_start,
            train_origin_end=environment.context.train_origin_end,
            search_budget=environment.search_budget,
            episode_seed=environment.episode_seed,
            episode_rebuild_id=environment._episode_rebuild_id,
            episode_rebuild_spec_sha256=environment._episode_rebuild_spec_sha256,
        )

    def state_dict(self) -> dict[str, Any]:
        """序列化必要的 Episode 重建规格，不保存完整环境或真实任务。"""

        return {
            "schema_version": EPISODE_REBUILD_SPEC_SCHEMA_VERSION,
            "episode_seed": self.episode_seed,
            "model": self.model,
            "encoder": self.encoder,
            "observation_spec": self.context.observation_spec,
            "train_origin_start": self.train_origin_start,
            "train_origin_end": self.train_origin_end,
            "search_budget": self.search_budget,
            "episode_rebuild_id": self._episode_rebuild_id,
            "episode_rebuild_spec_sha256": self._episode_rebuild_spec_sha256,
        }

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any]
    ) -> "EpisodeCandidateRebuildAdapter":
        if state.get("schema_version") != EPISODE_REBUILD_SPEC_SCHEMA_VERSION:
            raise ValueError("Replay Episode 重建规格版本不兼容")
        adapter = cls(
            model=state["model"],
            encoder=state["encoder"],
            observation_spec=state["observation_spec"],
            train_origin_start=int(state["train_origin_start"]),
            train_origin_end=int(state["train_origin_end"]),
            search_budget=state["search_budget"],
            episode_seed=state.get("episode_seed"),
            episode_rebuild_id=str(state["episode_rebuild_id"]),
            episode_rebuild_spec_sha256=str(
                state["episode_rebuild_spec_sha256"]
            ),
        )
        if adapter._episode_rebuild_spec_sha256 != adapter._calculate_spec_sha256():
            raise ValueError("Replay Episode 重建规格摘要不匹配")
        expected_id = hashlib.sha256(
            pickle.dumps(
                (adapter.episode_seed, adapter._episode_rebuild_spec_sha256),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        ).hexdigest()
        if adapter._episode_rebuild_id != expected_id:
            raise ValueError("Replay Episode 身份与重建规格不匹配")
        return adapter

    def _calculate_spec_sha256(self) -> str:
        payload = {
            "schema_version": EPISODE_REBUILD_SPEC_SCHEMA_VERSION,
            "episode_seed": self.episode_seed,
            "model": self.model,
            "encoder": self.encoder,
            "observation_spec": self.context.observation_spec,
            "train_origin_start": self.train_origin_start,
            "train_origin_end": self.train_origin_end,
            "search_budget": self.search_budget,
            "episode_vehicle_ids": self.encoder.source_vehicle_ids,
            "episode_task_vehicle_ids": self.encoder.source_task_vehicle_ids,
            "episode_cache_node_ids": self.encoder.source_node_ids,
            "canonical_vehicle_ids": self.encoder.vehicle_ids,
            "canonical_cache_node_ids": self.encoder.node_ids,
            "state_feature_contract_version": self.encoder.state_feature_contract_version,
            "candidate_feature_contract_version": self.encoder.candidate_feature_contract_version,
            "candidate_generation_contract_version": CANDIDATE_GENERATION_CONTRACT_VERSION,
            "network_architecture_version": self.encoder.network_architecture_version,
        }
        return _episode_rebuild_spec_sha256(payload)

    _candidate_capacity_limits = ExclusiveDQNEnvironment._candidate_capacity_limits
    _generate_legal_candidates = ExclusiveDQNEnvironment._generate_legal_candidates
    _decision_state_fingerprint = ExclusiveDQNEnvironment._decision_state_fingerprint
    validate_candidate_rebuild_context_metadata = (
        ExclusiveDQNEnvironment.validate_candidate_rebuild_context_metadata
    )
    rebuild_candidates = ExclusiveDQNEnvironment.rebuild_candidates


__all__ = [
    "EnvironmentTransition",
    "EpisodeCandidateRebuildAdapter",
    "ExclusiveDQNEnvironment",
]
