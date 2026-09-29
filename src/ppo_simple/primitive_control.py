"""构造 Cache/UAV 候选并按顺序应用 PPO 联合决策。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from common import project_config
from decision import cache_configuration, uav_configuration

from . import interface
TARGET_UAV_PROPOSAL_SOURCES = (
    "demand_centroid",
    "projected_demand",
    "geometry_exploration",
)


@dataclass(frozen=True)
class PrimitiveTensorSpec:
    num_cache_heads: int
    num_uav_heads: int
    observation_dim: int
    cache_feature_dim: int
    uav_feature_dim: int
    cache_max_candidates: int
    uav_max_candidates: int

    @classmethod
    def create(cls, context: Any) -> "PrimitiveTensorSpec":
        service_count = int(context.service_catalog["num_services"])
        uav_count = len(context.action_spec.uav_ids)
        search = uav_configuration.uav_search_config_from_project(context.cfg)
        return cls(
            num_cache_heads=len(context.action_spec.cache_node_ids),
            num_uav_heads=uav_count,
            observation_dim=int(context.observation_spec.observation_dim),
            cache_feature_dim=21 + 6 * service_count,
            uav_feature_dim=17 + uav_count + 2 * service_count,
            cache_max_candidates=int(context.cache_cfg.target_candidate_capacity),
            uav_max_candidates=int(search.target_candidate_capacity),
        )



@dataclass(frozen=True)
class CacheOperationPrimitiveSelectionResult:
    """保存一次顺序 Cache 操作与候选决策的结果。"""

    final_cache_state: np.ndarray
    final_uav_positions: dict[str, tuple[float, float]]
    derived_action: np.ndarray
    value: Any
    total_log_prob: Any
    total_entropy: Any
    operation_actions: np.ndarray
    operation_masks: np.ndarray
    operation_old_log_probs: np.ndarray
    operation_entropies: np.ndarray
    cache_contexts: np.ndarray
    cache_candidate_features: np.ndarray
    cache_candidate_masks: np.ndarray
    cache_selected_indices: np.ndarray
    cache_old_log_probs: np.ndarray
    cache_conditional_actions: tuple[dict[str, Any], ...]
    cache_candidate_counts: np.ndarray
    cache_trigger_node_local_workload: np.ndarray
    cache_trigger_aux: np.ndarray
    cache_trigger_has_legal_target: np.ndarray
    cache_trigger_node_local_workload_norms: np.ndarray
    cache_trigger_aux_norms: np.ndarray


@dataclass(frozen=True)
class JointCacheUavOperationPrimitiveSelectionResult:
    """保存一次 Cache 决策和显式 UAV 操作决策的联合结果。"""

    final_cache_state: np.ndarray
    final_uav_positions: dict[str, tuple[float, float]]
    derived_action: np.ndarray
    value: Any
    total_log_prob: Any
    total_entropy: Any
    cache_operation_actions: np.ndarray
    cache_operation_masks: np.ndarray
    cache_operation_old_log_probs: np.ndarray
    cache_operation_entropies: np.ndarray
    cache_contexts: np.ndarray
    cache_candidate_features: np.ndarray
    cache_candidate_masks: np.ndarray
    cache_selected_indices: np.ndarray
    cache_old_log_probs: np.ndarray
    cache_conditional_actions: tuple[dict[str, Any], ...]
    cache_candidate_counts: np.ndarray
    cache_trigger_node_local_workload: np.ndarray
    cache_trigger_aux: np.ndarray
    cache_trigger_has_legal_target: np.ndarray
    cache_trigger_node_local_workload_norms: np.ndarray
    cache_trigger_aux_norms: np.ndarray
    uav_operation_actions: np.ndarray
    uav_operation_masks: np.ndarray
    uav_operation_old_log_probs: np.ndarray
    uav_operation_entropies: np.ndarray
    uav_operation_probabilities: np.ndarray
    uav_operation_logit_margins: np.ndarray
    uav_contexts: np.ndarray
    uav_candidate_features: np.ndarray
    uav_candidate_masks: np.ndarray
    uav_selected_indices: np.ndarray
    uav_old_log_probs: np.ndarray
    uav_conditional_actions: tuple[dict[str, Any], ...]
    uav_candidate_counts: np.ndarray


def _intermediate_observation(
    cache_state: np.ndarray,
    uav_positions: dict[str, Any],
    step: Any,
    context: Any,
) -> np.ndarray:
    observation_time_to_go = getattr(context, "observation_time_to_go", None)
    time_to_go_override = (
        None
        if observation_time_to_go is None
        else float(observation_time_to_go(int(step.origin_index)))
    )
    observation = interface.build_step_observation(
        cache_state,
        uav_positions,
        step,
        context.observation_spec,
        context.flight_boundary,
        context.train_origin_start,
        context.train_origin_end,
        time_to_go_override=time_to_go_override,
    )
    return observation



def _cache_target_configuration_feature(
    candidate: cache_configuration.CacheTargetCandidate,
    current_cache: np.ndarray,
    previous_cache: np.ndarray,
    cache_triggers: np.ndarray,
    node_specific_demand: np.ndarray,
    observed_service_usage: np.ndarray,
    context: Any,
) -> np.ndarray:
    """编码完整 Cache 目标配置及其状态特征。"""

    service_count = int(context.service_catalog["num_services"])
    node_count = len(context.cache_nodes["node_ids"])
    before = np.asarray(candidate.before_node_cache, dtype=np.float64).reshape(-1)
    target = np.asarray(candidate.target_node_cache, dtype=np.float64).reshape(-1)
    demand = np.asarray(node_specific_demand, dtype=np.float64).reshape(-1)
    observed = np.asarray(observed_service_usage, dtype=np.float64).reshape(-1)
    if before.shape != (service_count,) or target.shape != before.shape:
        raise ValueError("target cache feature state shape mismatch")
    if demand.shape != before.shape or observed.shape != before.shape:
        raise ValueError("target cache feature demand shape mismatch")
    if not np.all(np.isin(before, [0.0, 1.0])) or not np.all(
        np.isin(target, [0.0, 1.0])
    ):
        raise ValueError("target cache feature states must be binary")
    delta = target - before
    sizes = np.asarray(context.service_catalog["service_sizes_mb"], dtype=np.float64)
    capacity = float(context.cache_nodes["capacities_mb"][candidate.node_index])
    trial = np.asarray(current_cache, dtype=np.int8).copy()
    trial[candidate.node_index] = target.astype(np.int8)
    total_cost, loading_energy = cache_configuration.cache_update_cost(
        previous_cache,
        trial,
        cache_triggers,
        sizes,
        context.cache_nodes,
        context.cache_cfg,
    )
    before_used = cache_configuration.node_used_capacity_mb(before.astype(np.int8), sizes)
    target_used = cache_configuration.node_used_capacity_mb(target.astype(np.int8), sizes)
    loaded_mb = float(np.dot(np.maximum(delta, 0.0), sizes))
    released_mb = float(np.dot(np.maximum(-delta, 0.0), sizes))
    demand_before = demand * before
    demand_after = demand * target
    demand_gain = demand_after - demand_before
    observed_before = observed * before
    observed_after = observed * target
    observed_gain = observed_after - observed_before
    current_replica_counts = np.sum(np.asarray(current_cache, dtype=np.float64), axis=0)
    target_replica_counts = np.sum(np.asarray(trial, dtype=np.float64), axis=0)
    current_diversity = float(np.count_nonzero(current_replica_counts > 0.0))
    target_diversity = float(np.count_nonzero(target_replica_counts > 0.0))
    zero_after = float(np.count_nonzero(target_replica_counts == 0.0))
    single_after = float(np.count_nonzero(target_replica_counts == 1.0))
    role = str(context.cache_nodes["node_roles"][candidate.node_index]).upper()
    role_feature = np.asarray(
        [float(role == "SV"), float(role == "RSU"), float(role == "UAV")],
        dtype=np.float64,
    )
    scalar_features = np.asarray(
        [
            before_used / max(capacity, 1.0e-12),
            target_used / max(capacity, 1.0e-12),
            (capacity - target_used) / max(capacity, 1.0e-12),
            loaded_mb / max(capacity, 1.0e-12),
            released_mb / max(capacity, 1.0e-12),
            float(np.sum(demand_before)),
            float(np.sum(demand_after)),
            float(np.sum(demand_gain)),
            float(np.sum(observed_before)),
            float(np.sum(observed_after)),
            float(np.sum(observed_gain)),
            math.log1p(max(float(total_cost), 0.0)),
            math.log1p(max(float(loading_energy[candidate.node_index]), 0.0)),
            float(np.mean(np.abs(delta))),
            (target_diversity - current_diversity) / max(float(service_count), 1.0),
            zero_after / max(float(service_count), 1.0),
            single_after / max(float(service_count), 1.0),
        ],
        dtype=np.float64,
    )
    values = np.concatenate(
        [
            np.asarray(
                [float(candidate.node_index) / max(node_count - 1, 1)],
                dtype=np.float64,
            ),
            role_feature,
            before,
            target,
            delta,
            demand_before,
            demand_after,
            demand_gain,
            scalar_features,
        ]
    )
    expected_dim = 21 + 6 * service_count
    if values.shape != (expected_dim,) or not np.all(np.isfinite(values)):
        raise AssertionError("target cache configuration feature shape/finite check failed")
    return values.astype(np.float32)


def _all_cache_target_configurations_with_features(
    current_cache: np.ndarray,
    previous_cache: np.ndarray,
    sequential_triggers: np.ndarray,
    node_index: int,
    step: Any,
    context: Any,
    proposal_cache_nodes: dict[str, Any],
) -> tuple[list[cache_configuration.CacheTargetCandidate], np.ndarray]:
    """枚举全部可行配置，并只计算低成本特征。"""

    all_candidates = cache_configuration.enumerate_all_cache_target_configurations(
        current_cache,
        int(node_index),
        context.service_catalog,
        proposal_cache_nodes,
        step.scenario_slice,
        context.local_cache,
        context.cache_cfg,
    )
    if not all_candidates:
        return [], np.zeros(
            (0, 21 + 6 * int(context.service_catalog["num_services"])),
            dtype=np.float32,
        )
    node_demand = cache_configuration.node_specific_predicted_demand(
        step.scenario_slice,
        int(node_index),
        proposal_cache_nodes,
        context.cache_cfg,
    )
    observed_usage = cache_configuration.observed_local_service_usage(
        context.local_cache,
        int(context.service_catalog["num_services"]),
    )
    all_features = np.stack(
        [
            _cache_target_configuration_feature(
                candidate,
                current_cache,
                previous_cache,
                sequential_triggers,
                node_demand,
                observed_usage,
                context,
            )
            for candidate in all_candidates
        ]
    ).astype(np.float32, copy=False)
    return all_candidates, all_features


def _target_cache_proposals_with_features(
    current_cache: np.ndarray,
    previous_cache: np.ndarray,
    sequential_triggers: np.ndarray,
    node_index: int,
    conditional_context: np.ndarray,
    step: Any,
    context: Any,
    proposal_cache_nodes: dict[str, Any],
    *,
    surrogate_predictor: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    surrogate_ready: bool = False,
) -> tuple[list[cache_configuration.CacheTargetCandidate], np.ndarray]:
    """根据全部可行配置构造 rollout 候选集合。"""

    all_candidates, all_features = _all_cache_target_configurations_with_features(
        current_cache,
        previous_cache,
        sequential_triggers,
        node_index,
        step,
        context,
        proposal_cache_nodes,
    )
    if not all_candidates:
        return all_candidates, all_features
    predicted_delta_q = None
    predicted_sigma = None
    if surrogate_ready and surrogate_predictor is not None:
        predicted_delta_q, predicted_sigma = surrogate_predictor(
            np.asarray(conditional_context, dtype=np.float32), all_features
        )
    selected = cache_configuration.select_cache_target_configurations(
        all_candidates,
        np.asarray(current_cache, dtype=np.int8)[int(node_index)],
        int(context.cache_cfg.target_candidate_capacity),
        predicted_delta_q=predicted_delta_q,
        predicted_sigma=predicted_sigma,
        diversity_reserve=min(
            12,
            max(0, int(context.cache_cfg.target_candidate_capacity) - 1),
        ),
    )
    selected_index_by_configuration = {
        np.ascontiguousarray(candidate.target_node_cache).tobytes(): index
        for index, candidate in enumerate(all_candidates)
    }
    selected_indices = [
        selected_index_by_configuration[
            np.ascontiguousarray(candidate.target_node_cache).tobytes()
        ]
        for candidate in selected
    ]
    return selected, all_features[np.asarray(selected_indices, dtype=np.int64)]


def _normalized_xy(
    xy: Any,
    boundary: tuple[float, float, float, float],
) -> np.ndarray:
    x_min, y_min, x_max, y_max = [float(value) for value in boundary]
    value = np.asarray(xy, dtype=np.float64)
    return np.asarray(
        [
            (float(value[0]) - x_min) / (x_max - x_min),
            (float(value[1]) - y_min) / (y_max - y_min),
        ],
        dtype=np.float64,
    )



def _uav_target_position_feature(
    candidate: uav_configuration.UavTargetCandidate,
    current_uav: dict[str, np.ndarray],
    previous_uav: dict[str, np.ndarray],
    uav_triggers: dict[str, int],
    predicted_vehicle_positions: dict[str, Any],
    predicted_vehicle_service_demand: np.ndarray,
    cache_state: np.ndarray,
    context: Any,
) -> np.ndarray:
    """编码考虑服务需求的 UAV 目标位置和实际移动成本。"""

    uav_ids = tuple(context.action_spec.uav_ids)
    uav_index = uav_ids.index(candidate.uav_id)
    service_count = int(context.service_catalog["num_services"])
    search = uav_configuration.uav_search_config_from_project(context.cfg)
    trial = {
        key: np.asarray(value, dtype=np.float64).copy()
        for key, value in current_uav.items()
    }
    trial[candidate.uav_id] = np.asarray(candidate.target_position, dtype=np.float64).copy()
    movement_energy, movement_cost, _, _, _ = uav_configuration.movement_summary(
        previous_uav,
        trial,
        uav_triggers,
        search,
    )
    before_normalized = _normalized_xy(candidate.before_position, context.flight_boundary)
    target_normalized = _normalized_xy(candidate.target_position, context.flight_boundary)
    x_min, y_min, x_max, y_max = [float(value) for value in context.flight_boundary]
    scale = np.asarray([x_max - x_min, y_max - y_min], dtype=np.float64)
    delta = (np.asarray(candidate.target_position) - np.asarray(candidate.before_position)) / scale
    distance = float(
        np.linalg.norm(
            np.asarray(candidate.target_position) - np.asarray(candidate.before_position)
        )
    )
    vehicle_ids = tuple(context.observation_spec.vehicle_ids)
    vehicle_xy = np.asarray(
        [
            _normalized_xy(predicted_vehicle_positions[vehicle_id], context.flight_boundary)
            for vehicle_id in vehicle_ids
        ],
        dtype=np.float64,
    )
    demand = np.asarray(predicted_vehicle_service_demand, dtype=np.float64)
    if demand.shape != (len(vehicle_ids), service_count) or not np.all(np.isfinite(demand)):
        raise ValueError("UAV target feature demand shape/finite check failed")
    node_ids = [str(value) for value in context.cache_nodes["node_ids"]]
    node_index = node_ids.index(str(candidate.uav_id))
    cache_vector = np.asarray(cache_state[node_index], dtype=np.float64).reshape(-1)
    if cache_vector.shape != (service_count,):
        raise ValueError("UAV target feature cache vector shape mismatch")
    service_weights = np.maximum(demand @ cache_vector, 0.0)
    if float(np.sum(service_weights)) > 1.0e-12:
        demand_centroid = np.sum(
            vehicle_xy * service_weights[:, None],
            axis=0,
        ) / float(np.sum(service_weights))
    else:
        demand_centroid = _normalized_xy(candidate.before_position, context.flight_boundary)
    range_m = max(float(context.cache_cfg.uav_range_m), 1.0e-12)
    target_xy = np.asarray(candidate.target_position, dtype=np.float64)
    target_distance = np.linalg.norm(
        np.asarray([target_xy[0], target_xy[1]], dtype=np.float64)[None, :]
        - np.asarray(
            [predicted_vehicle_positions[vehicle_id] for vehicle_id in vehicle_ids],
            dtype=np.float64,
        ),
        axis=1,
    )
    serviceability = np.maximum(0.0, 1.0 - target_distance / range_m)
    target_service_demand = np.sum(demand * serviceability[:, None], axis=0)
    uav_one_hot = np.zeros(len(uav_ids), dtype=np.float64)
    uav_one_hot[uav_index] = 1.0
    source_feature = np.asarray(
        [float(candidate.proposal_source == value) for value in TARGET_UAV_PROPOSAL_SOURCES],
        dtype=np.float64,
    )
    values = np.concatenate(
        [
            np.asarray([float(uav_index) / max(len(uav_ids) - 1, 1)], dtype=np.float64),
            uav_one_hot,
            source_feature,
            before_normalized,
            target_normalized,
            delta,
            np.asarray(
                [
                    distance / max(search.max_move_distance_m, 1.0e-12),
                    math.log1p(max(float(movement_energy), 0.0)),
                    math.log1p(max(float(movement_cost), 0.0)),
                ],
                dtype=np.float64,
            ),
            demand_centroid,
            np.asarray(
                [
                    float(np.linalg.norm(target_normalized - demand_centroid)),
                    float(np.sum(target_service_demand)),
                ],
                dtype=np.float64,
            ),
            cache_vector,
            target_service_demand,
        ]
    )
    expected_dim = 17 + len(uav_ids) + 2 * service_count
    if values.shape != (expected_dim,) or not np.all(np.isfinite(values)):
        raise AssertionError("UAV target position feature shape/finite check failed")
    return values.astype(np.float32)






def sample_and_apply_cache_reconfiguration_configuration(
    policy: Any,
    observation: np.ndarray,
    previous_cache_state: np.ndarray,
    previous_uav_positions: dict[str, Any],
    step: Any,
    context: Any,
    tensor_spec: PrimitiveTensorSpec,
    *,
    deterministic: bool = False,
    cache_surrogate_predictor: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    cache_surrogate_ready: bool = False,
) -> CacheOperationPrimitiveSelectionResult:
    """采样 KEEP/UPDATE，再采样一个条件目标配置。"""

    trigger_order = ("KEEP", "UPDATE")
    if tuple(getattr(policy, "trigger_order", ())) != trigger_order:
        raise ValueError("Cache trigger order must be KEEP/UPDATE")
    cache = np.asarray(previous_cache_state, dtype=np.int8).copy()
    previous_cache = cache.copy()
    current_uav = {
        uav_id: np.asarray(previous_uav_positions[uav_id], dtype=np.float64).copy()
        for uav_id in context.action_spec.uav_ids
    }
    if tensor_spec.num_cache_heads != len(context.action_spec.cache_node_ids):
        raise ValueError("Cache tensor spec head count does not match action spec")
    proposal_cache_nodes = uav_configuration.cache_nodes_with_uav_positions(
        context.cache_nodes,
        current_uav,
    )
    input_jitter = float(
        project_config.get_config_value(context.cfg, "task.input_relative_jitter", 0.0)
    )
    load_jitter = float(
        project_config.get_config_value(context.cfg, "task.load_relative_jitter", 0.0)
    )
    cache_trigger_node_local_workload = (
        cache_configuration.node_local_predicted_service_workload(
            step.scenario_slice,
            proposal_cache_nodes,
            context.cache_cfg,
            context.service_catalog,
            input_relative_jitter=input_jitter,
            load_relative_jitter=load_jitter,
        )
    )
    if cache_trigger_node_local_workload.shape != (
        tensor_spec.num_cache_heads,
        int(context.service_catalog["num_services"]),
        3,
    ):
        raise AssertionError("node-local Trigger workload shape mismatch")
    if cache_trigger_node_local_workload.shape[1] * 3 != int(
        getattr(policy, "trigger_workload_dim", -1)
    ):
        raise AssertionError("policy and node-local Trigger workload dimensions differ")
    trigger_aux_dim = int(getattr(policy, "trigger_aux_dim", -1))
    if trigger_aux_dim != cache_trigger_node_local_workload.shape[1] * 3:
        raise AssertionError("policy and Trigger auxiliary dimensions differ")
    cache_trigger_aux = cache_trigger_node_local_workload.reshape(
        tensor_spec.num_cache_heads, trigger_aux_dim
    ).astype(np.float32, copy=True)
    cache_trigger_has_legal_target = np.zeros(
        tensor_spec.num_cache_heads, dtype=bool
    )
    cache_trigger_node_local_workload_norms = np.linalg.norm(
        cache_trigger_node_local_workload.reshape(tensor_spec.num_cache_heads, -1),
        axis=-1,
    ).astype(np.float32)
    cache_trigger_aux_norms = np.linalg.norm(
        cache_trigger_aux, axis=-1
    ).astype(np.float32)

    operation_actions = np.zeros(tensor_spec.num_cache_heads, dtype=np.int64)
    operation_masks = np.zeros(
        (tensor_spec.num_cache_heads, len(trigger_order)),
        dtype=bool,
    )
    operation_old_log_probs = np.zeros(tensor_spec.num_cache_heads, dtype=np.float32)
    operation_entropies = np.zeros(tensor_spec.num_cache_heads, dtype=np.float32)
    cache_contexts = np.zeros(
        (tensor_spec.num_cache_heads, tensor_spec.observation_dim), dtype=np.float32
    )
    cache_features = np.zeros(
        (
            tensor_spec.num_cache_heads,
            tensor_spec.cache_max_candidates,
            tensor_spec.cache_feature_dim,
        ),
        dtype=np.float32,
    )
    cache_masks = np.zeros(
        (tensor_spec.num_cache_heads, tensor_spec.cache_max_candidates),
        dtype=bool,
    )
    cache_masks[:, 0] = True
    cache_selected = np.zeros(tensor_spec.num_cache_heads, dtype=np.int64)
    cache_old_log_probs = np.zeros(tensor_spec.num_cache_heads, dtype=np.float32)
    cache_counts = np.zeros(tensor_spec.num_cache_heads, dtype=np.int64)
    cache_triggers = np.zeros(tensor_spec.num_cache_heads, dtype=np.int8)
    cache_actions: list[dict[str, Any]] = []
    total_log_prob: Any | None = None
    total_entropy: Any | None = None

    for node_index, node_id in enumerate(context.action_spec.cache_node_ids):
        conditional_context = _intermediate_observation(
            cache,
            current_uav,
            step,
            context,
        )
        has_legal_target = cache_configuration.has_legal_cache_target_configuration(
            cache,
            node_index,
            context.service_catalog,
            context.cache_nodes,
        )
        cache_trigger_has_legal_target[node_index] = has_legal_target
        trigger_aux = cache_trigger_aux[node_index]
        operation_mask = np.asarray(
            [True, has_legal_target],
            dtype=bool,
        )
        sampled_trigger = policy.sample_cache_trigger(
            conditional_context,
            node_index,
            operation_mask,
            trigger_aux=trigger_aux,
            deterministic=bool(deterministic),
        )
        selected_trigger = int(sampled_trigger.selected_index)
        operation_actions[node_index] = selected_trigger
        operation_masks[node_index] = operation_mask
        operation_old_log_probs[node_index] = float(
            sampled_trigger.log_prob.detach().cpu()
        )
        operation_entropies[node_index] = float(
            sampled_trigger.entropy.detach().cpu()
        )
        cache_contexts[node_index] = conditional_context
        total_log_prob = (
            sampled_trigger.log_prob
            if total_log_prob is None
            else total_log_prob + sampled_trigger.log_prob
        )
        total_entropy = (
            sampled_trigger.entropy
            if total_entropy is None
            else total_entropy + sampled_trigger.entropy
        )

        if selected_trigger == 0:
            cache_actions.append(
                {
                    "head_type": "cache",
                    "head_id": str(node_id),
                    "head_index": node_index,
                    "candidate_count": 0,
                    "has_legal_update_target": bool(has_legal_target),
                    "selected_index": None,
                    "conditional_action": "KEEP",
                    "actual_changed": False,
                    "TRIGGER_NODE_LOCAL_WORKLOAD_NORM": float(
                        cache_trigger_node_local_workload_norms[node_index]
                    ),
                    "TRIGGER_AUX_NORM": float(cache_trigger_aux_norms[node_index]),
                }
            )
            continue

        if not has_legal_target:
            raise AssertionError("selected UPDATE without a cheap legal target")
        cache_triggers[node_index] = 1
        hypothetical_triggers = cache_triggers.copy()
        update_candidates, candidate_features = _target_cache_proposals_with_features(
            cache,
            previous_cache,
            hypothetical_triggers,
            node_index,
            conditional_context,
            step,
            context,
            proposal_cache_nodes,
            surrogate_predictor=cache_surrogate_predictor,
            surrogate_ready=bool(cache_surrogate_ready),
        )
        count = len(update_candidates)
        if count == 0:
            raise AssertionError(
                "cheap legality reported UPDATE feasible but proposal set is empty"
            )
        if count > tensor_spec.cache_max_candidates:
            raise AssertionError("Cache candidate count exceeds padded capacity")
        if candidate_features is None:
            raise AssertionError("UPDATE requires candidate features")
        sampled_candidate = policy.sample_cache_update_candidate(
            conditional_context,
            candidate_features,
            deterministic=bool(deterministic),
        )
        selected_index = int(sampled_candidate.selected_index)
        selected = update_candidates[selected_index]
        applied = cache_configuration.apply_cache_target_configuration(
            cache,
            node_index,
            selected.target_node_cache,
            context.service_catalog,
            context.cache_nodes,
        )
        cache_features[node_index, :count] = candidate_features
        cache_masks[node_index, :count] = True
        cache_selected[node_index] = selected_index
        cache_old_log_probs[node_index] = float(
            sampled_candidate.log_prob.detach().cpu()
        )
        cache_counts[node_index] = count
        total_log_prob = total_log_prob + sampled_candidate.log_prob
        total_entropy = total_entropy + sampled_candidate.entropy
        cache_actions.append(
            {
                "head_type": "cache",
                "head_id": str(node_id),
                "head_index": node_index,
                "candidate_count": count,
                "has_legal_update_target": True,
                "selected_index": selected_index,
                "conditional_action": "TARGET_CONFIGURATION",
                "target_cache": selected.target_node_cache.astype(int).tolist(),
                "proposal_source": selected.proposal_source,
                "actual_changed": bool(applied["actual_changed"]),
                "TRIGGER_NODE_LOCAL_WORKLOAD_NORM": float(
                    cache_trigger_node_local_workload_norms[node_index]
                ),
                "TRIGGER_AUX_NORM": float(cache_trigger_aux_norms[node_index]),
            }
        )

    if total_log_prob is None or total_entropy is None:
        raise RuntimeError("Cache trigger selection produced no heads")
    cache_configuration.validate_cache_shape_and_capacity(
        cache,
        context.service_catalog,
        context.cache_nodes,
    )
    derived_action = np.concatenate(
        [
            cache_triggers,
            np.zeros(len(context.action_spec.uav_ids), dtype=np.int8),
        ]
    ).astype(np.int8)
    value = policy.critic_value(observation)
    return CacheOperationPrimitiveSelectionResult(
        final_cache_state=cache.copy(),
        final_uav_positions={
            key: (float(value[0]), float(value[1]))
            for key, value in current_uav.items()
        },
        derived_action=derived_action,
        value=value,
        total_log_prob=total_log_prob,
        total_entropy=total_entropy,
        operation_actions=operation_actions,
        operation_masks=operation_masks,
        operation_old_log_probs=operation_old_log_probs,
        operation_entropies=operation_entropies,
        cache_contexts=cache_contexts,
        cache_candidate_features=cache_features,
        cache_candidate_masks=cache_masks,
        cache_selected_indices=cache_selected,
        cache_old_log_probs=cache_old_log_probs,
        cache_conditional_actions=tuple(cache_actions),
        cache_candidate_counts=cache_counts,
        cache_trigger_node_local_workload=cache_trigger_node_local_workload.copy(),
        cache_trigger_aux=cache_trigger_aux.copy(),
        cache_trigger_has_legal_target=cache_trigger_has_legal_target.copy(),
        cache_trigger_node_local_workload_norms=cache_trigger_node_local_workload_norms.copy(),
        cache_trigger_aux_norms=cache_trigger_aux_norms.copy(),
    )


def sample_and_apply_joint_cache_uav_operation_configuration(
    policy: Any,
    observation: np.ndarray,
    previous_cache_state: np.ndarray,
    previous_uav_positions: dict[str, Any],
    step: Any,
    context: Any,
    tensor_spec: PrimitiveTensorSpec,
    *,
    deterministic: bool = False,
    cache_surrogate_predictor: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    cache_surrogate_ready: bool = False,
) -> JointCacheUavOperationPrimitiveSelectionResult:
    """先完成 Cache 目标重配置，再按顺序采样 UAV 的 STAY/MOVE 决策。"""

    operation_order = ("STAY", "MOVE")
    if tuple(getattr(policy, "uav_operation_order", ())) != operation_order:
        raise ValueError("UAV operation order must be STAY/MOVE")
    cache_result = sample_and_apply_cache_reconfiguration_configuration(
        policy,
        observation,
        previous_cache_state,
        previous_uav_positions,
        step,
        context,
        tensor_spec,
        deterministic=bool(deterministic),
        cache_surrogate_predictor=cache_surrogate_predictor,
        cache_surrogate_ready=bool(cache_surrogate_ready),
    )
    current_uav = {
        uav_id: np.asarray(previous_uav_positions[uav_id], dtype=np.float64).copy()
        for uav_id in context.action_spec.uav_ids
    }
    previous_uav = {
        uav_id: np.asarray(previous_uav_positions[uav_id], dtype=np.float64).copy()
        for uav_id in context.action_spec.uav_ids
    }
    uav_operation_actions = np.zeros(tensor_spec.num_uav_heads, dtype=np.int64)
    uav_operation_masks = np.zeros(
        (tensor_spec.num_uav_heads, len(operation_order)),
        dtype=bool,
    )
    uav_operation_old_log_probs = np.zeros(
        tensor_spec.num_uav_heads,
        dtype=np.float32,
    )
    uav_operation_entropies = np.zeros(
        tensor_spec.num_uav_heads,
        dtype=np.float32,
    )
    uav_operation_probabilities = np.zeros(
        (tensor_spec.num_uav_heads, len(operation_order)),
        dtype=np.float32,
    )
    uav_operation_logit_margins = np.zeros(
        tensor_spec.num_uav_heads,
        dtype=np.float32,
    )
    uav_contexts = np.zeros(
        (tensor_spec.num_uav_heads, tensor_spec.observation_dim),
        dtype=np.float32,
    )
    uav_features = np.zeros(
        (
            tensor_spec.num_uav_heads,
            tensor_spec.uav_max_candidates,
            tensor_spec.uav_feature_dim,
        ),
        dtype=np.float32,
    )
    uav_masks = np.zeros(
        (tensor_spec.num_uav_heads, tensor_spec.uav_max_candidates),
        dtype=bool,
    )
    uav_masks[:, 0] = True
    uav_selected = np.zeros(tensor_spec.num_uav_heads, dtype=np.int64)
    uav_old_log_probs = np.zeros(tensor_spec.num_uav_heads, dtype=np.float32)
    uav_counts = np.zeros(tensor_spec.num_uav_heads, dtype=np.int64)
    uav_actions: list[dict[str, Any]] = []
    uav_triggers = {
        uav_id: 0 for uav_id in context.action_spec.uav_ids
    }
    total_log_prob = cache_result.total_log_prob
    total_entropy = cache_result.total_entropy

    for uav_index, uav_id in enumerate(context.action_spec.uav_ids):
        # 候选枚举要求当前目标 UAV 的 trigger 为 1。
        # 保留先前已处理的目标，使后续 UAV 能看到更新后的位置和移动状态。
        uav_triggers[uav_id] = 1
        conditional_context = _intermediate_observation(
            cache_result.final_cache_state,
            current_uav,
            step,
            context,
        )
        candidates = uav_configuration.enumerate_uav_target_candidates(
            current_uav,
            previous_uav,
            uav_id,
            uav_triggers,
            context.cfg,
            context.flight_boundary,
            scenario_slice=step.scenario_slice,
            cache_state=cache_result.final_cache_state,
            cache_nodes=context.cache_nodes,
            predicted_vehicle_positions=step.predicted_vehicle_positions,
        )
        move_candidates = candidates
        if len(move_candidates) > tensor_spec.uav_max_candidates:
            raise AssertionError("UAV candidate count exceeds padded capacity")
        operation_mask = np.asarray(
            [True, bool(move_candidates)],
            dtype=bool,
        )
        sampled_operation = policy.sample_uav_operation(
            conditional_context,
            uav_index,
            operation_mask,
            deterministic=bool(deterministic),
        )
        selected_operation_index = int(sampled_operation.selected_index)
        selected_operation = operation_order[selected_operation_index]
        uav_operation_actions[uav_index] = selected_operation_index
        uav_operation_masks[uav_index] = operation_mask
        uav_operation_old_log_probs[uav_index] = float(
            sampled_operation.log_prob.detach().cpu()
        )
        uav_operation_entropies[uav_index] = float(
            sampled_operation.entropy.detach().cpu()
        )
        if (
            sampled_operation.raw_logits is None
            or sampled_operation.probabilities is None
        ):
            raise RuntimeError(
                "UAV operation sampling did not return raw diagnostics"
            )
        raw_logits = np.asarray(
            sampled_operation.raw_logits.detach().cpu().numpy(),
            dtype=np.float64,
        ).reshape(-1)
        raw_probabilities = np.asarray(
            sampled_operation.probabilities.detach().cpu().numpy(),
            dtype=np.float64,
        ).reshape(-1)
        if raw_logits.shape != (len(operation_order),):
            raise AssertionError("UAV raw operation logits shape mismatch")
        if raw_probabilities.shape != (len(operation_order),):
            raise AssertionError("UAV raw operation probabilities shape mismatch")
        if not np.all(np.isfinite(raw_logits)) or not np.all(
            np.isfinite(raw_probabilities)
        ):
            raise FloatingPointError("UAV raw operation diagnostics are non-finite")
        if not np.isclose(float(raw_probabilities.sum()), 1.0):
            raise AssertionError("UAV raw operation probabilities must sum to one")
        uav_operation_probabilities[uav_index] = raw_probabilities.astype(np.float32)
        uav_operation_logit_margins[uav_index] = float(
            raw_logits[0] - raw_logits[1]
        )
        uav_contexts[uav_index] = conditional_context
        total_log_prob = total_log_prob + sampled_operation.log_prob
        total_entropy = total_entropy + sampled_operation.entropy

        count = len(move_candidates)
        if count:
            candidate_features = np.stack(
                [
                    _uav_target_position_feature(
                        candidate,
                        current_uav,
                        previous_uav,
                        uav_triggers,
                        step.predicted_vehicle_positions,
                        cache_configuration.predicted_vehicle_service_demand(
                            step.scenario_slice,
                            int(context.service_catalog["num_services"]),
                            beta_discount=context.cache_cfg.beta_discount,
                        ),
                        cache_result.final_cache_state,
                        context,
                    )
                    for candidate in move_candidates
                ]
            )
            uav_features[uav_index, :count] = candidate_features
            uav_masks[uav_index, :count] = True
        else:
            candidate_features = np.zeros(
                (0, tensor_spec.uav_feature_dim),
                dtype=np.float32,
            )
        uav_counts[uav_index] = count

        if selected_operation == "STAY":
            uav_triggers[uav_id] = 0
            before = current_uav[uav_id].copy()
            uav_actions.append(
                {
                    "head_type": "uav",
                    "head_id": str(uav_id),
                    "head_index": uav_index,
                    "candidate_count": count,
                    "selected_index": None,
                    "conditional_action": "STAY",
                    "before_position": [float(value) for value in before],
                    "after_position": [float(value) for value in before],
                    "actual_moved": False,
                }
            )
            continue

        if not move_candidates:
            raise AssertionError("selected MOVE without a feasible MOVE candidate")
        if selected_operation != "MOVE":
            raise AssertionError("UAV operation selected an unsupported operation")
        sampled_candidate = policy.sample_uav_candidate(
            conditional_context,
            candidate_features,
            deterministic=bool(deterministic),
        )
        selected_index = int(sampled_candidate.selected_index)
        selected = move_candidates[selected_index]
        applied = uav_configuration.apply_uav_target_position(
            current_uav,
            previous_uav,
            uav_id,
            selected.target_position,
            uav_triggers,
            context.cfg,
            context.flight_boundary,
        )
        if not bool(applied["actual_moved"]):
            raise AssertionError("MOVE operation did not produce actual movement")
        uav_selected[uav_index] = selected_index
        uav_old_log_probs[uav_index] = float(
            sampled_candidate.log_prob.detach().cpu()
        )
        total_log_prob = total_log_prob + sampled_candidate.log_prob
        total_entropy = total_entropy + sampled_candidate.entropy
        uav_actions.append(
            {
                "head_type": "uav",
                "head_id": str(uav_id),
                "head_index": uav_index,
                "candidate_count": count,
                "selected_index": selected_index,
                "conditional_action": "TARGET_POSITION",
                "proposal_source": selected.proposal_source,
                "before_position": [float(value) for value in selected.before_position],
                "target_position": [float(value) for value in selected.target_position],
                "after_position": [float(value) for value in selected.target_position],
                "actual_moved": True,
            }
        )

    uav_configuration.validate_uav_configuration_transition(
        previous_uav,
        current_uav,
        uav_triggers,
        context.cfg,
        context.flight_boundary,
    )
    derived_action = np.concatenate(
        [
            np.asarray(cache_result.derived_action[: tensor_spec.num_cache_heads], dtype=np.int8),
            np.asarray(
                [uav_triggers[uav_id] for uav_id in context.action_spec.uav_ids],
                dtype=np.int8,
            ),
        ]
    ).astype(np.int8)
    return JointCacheUavOperationPrimitiveSelectionResult(
        final_cache_state=cache_result.final_cache_state.copy(),
        final_uav_positions={
            key: (float(value[0]), float(value[1]))
            for key, value in current_uav.items()
        },
        derived_action=derived_action,
        value=cache_result.value,
        total_log_prob=total_log_prob,
        total_entropy=total_entropy,
        cache_operation_actions=cache_result.operation_actions,
        cache_operation_masks=cache_result.operation_masks,
        cache_operation_old_log_probs=cache_result.operation_old_log_probs,
        cache_operation_entropies=cache_result.operation_entropies,
        cache_contexts=cache_result.cache_contexts,
        cache_candidate_features=cache_result.cache_candidate_features,
        cache_candidate_masks=cache_result.cache_candidate_masks,
        cache_selected_indices=cache_result.cache_selected_indices,
        cache_old_log_probs=cache_result.cache_old_log_probs,
        cache_conditional_actions=cache_result.cache_conditional_actions,
        cache_candidate_counts=cache_result.cache_candidate_counts,
        cache_trigger_node_local_workload=cache_result.cache_trigger_node_local_workload,
        cache_trigger_aux=cache_result.cache_trigger_aux,
        cache_trigger_has_legal_target=cache_result.cache_trigger_has_legal_target,
        cache_trigger_node_local_workload_norms=cache_result.cache_trigger_node_local_workload_norms,
        cache_trigger_aux_norms=cache_result.cache_trigger_aux_norms,
        uav_operation_actions=uav_operation_actions,
        uav_operation_masks=uav_operation_masks,
        uav_operation_old_log_probs=uav_operation_old_log_probs,
        uav_operation_entropies=uav_operation_entropies,
        uav_operation_probabilities=uav_operation_probabilities,
        uav_operation_logit_margins=uav_operation_logit_margins,
        uav_contexts=uav_contexts,
        uav_candidate_features=uav_features,
        uav_candidate_masks=uav_masks,
        uav_selected_indices=uav_selected,
        uav_old_log_probs=uav_old_log_probs,
        uav_conditional_actions=tuple(uav_actions),
        uav_candidate_counts=uav_counts,
    )
