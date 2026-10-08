"""复用正式 Cache/UAV 候选生成与候选特征构造。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from decision import cache_configuration, uav_configuration

TARGET_UAV_PROPOSAL_SOURCES = (
    "demand_centroid",
    "projected_demand",
    "geometry_exploration",
)

CACHE_ECONOMIC_FEATURE_DIM = 19
CACHE_ECONOMIC_BASE_MACRO_FEATURE_DIM = 181
CACHE_ECONOMIC_DEMAND_HORIZONS = 5
CACHE_ECONOMIC_FEATURE_NAMES = (
    "is_keep",
    "node_index",
    "node_role_sv",
    "node_role_rsu",
    "node_role_uav",
    "log_cache_cost",
    "loaded_fraction",
    "released_fraction",
    "replica_diversity_delta",
    "demand_added_h0",
    "demand_added_h1",
    "demand_added_h2",
    "demand_added_h3",
    "demand_added_h4",
    "demand_removed_h0",
    "demand_removed_h1",
    "demand_removed_h2",
    "demand_removed_h3",
    "demand_removed_h4",
)


def _intermediate_observation(
    cache_state: np.ndarray,
    uav_positions: dict[str, Any],
    step: Any,
    context: Any,
) -> np.ndarray:
    """为保留的 Exact-Q diagnostic 构造中间状态 observation。"""

    from . import interface

    time_to_go = getattr(context, "observation_time_to_go", None)
    override = None if time_to_go is None else float(time_to_go(int(step.origin_index)))
    return interface.build_step_observation(
        cache_state,
        uav_positions,
        step,
        context.observation_spec,
        context.flight_boundary,
        context.train_origin_start,
        context.train_origin_end,
        time_to_go_override=override,
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


def cache_economic_candidate_features(
    current_cache: np.ndarray,
    candidates: list[np.ndarray],
    primitive_features: np.ndarray,
    step: Any,
    context: Any,
    proposal_cache_nodes: dict[str, Any],
) -> np.ndarray:
    """Build the 19 economic features appended to the unchanged Macro features.

    Demand values are prediction-only proxies averaged over prediction
    scenarios and normalized by the fixed number of predicted task vehicles.
    No realized tasks or future realized positions are consulted.
    """

    current = np.asarray(current_cache, dtype=np.int8)
    primitive = np.asarray(primitive_features, dtype=np.float32)
    service_count = int(context.service_catalog["num_services"])
    node_count = len(context.cache_nodes["node_ids"])
    expected_primitive_dim = 21 + 6 * service_count
    if current.ndim != 2 or current.shape[0] != node_count:
        raise ValueError("Cache economic feature current state shape mismatch")
    if not candidates or not np.array_equal(np.asarray(candidates[0]), current):
        raise ValueError("Cache economic candidate 0 must be KEEP/current Cache")
    if primitive.shape != (len(candidates), expected_primitive_dim):
        raise ValueError(
            "Cache economic primitive features must align with candidates and the existing 81-D schema"
        )

    result = np.zeros((len(candidates), CACHE_ECONOMIC_FEATURE_DIM), dtype=np.float32)
    result[0, 0] = 1.0
    if len(candidates) == 1:
        return result

    scenario = step.scenario_slice
    arrivals = np.asarray(scenario.get("arrival"), dtype=np.float64)
    service_ids = np.asarray(scenario.get("service_id"), dtype=np.int64)
    if arrivals.ndim != 3 or service_ids.shape != arrivals.shape:
        raise ValueError(
            "prediction arrival/service_id must share [scenario,horizon,task_vehicle]"
        )
    task_vehicle_ids = scenario.get("task_vehicle_ids")
    task_vehicle_count = 0 if task_vehicle_ids is None else len(task_vehicle_ids)
    if task_vehicle_count <= 0 or arrivals.shape[2] != task_vehicle_count:
        raise ValueError("predicted task vehicle count must align with scenario requests")
    horizon_count = int(arrivals.shape[1])
    horizon_mask_raw = getattr(step, "prediction_horizon_mask", None)
    if horizon_mask_raw is None:
        raise ValueError("Cache economic features require prediction_horizon_mask")
    horizon_mask = np.asarray(horizon_mask_raw, dtype=bool).reshape(-1)
    if horizon_mask.shape != (horizon_count,):
        raise ValueError("prediction_horizon_mask must align with scenario horizons")

    vehicle_demand = cache_configuration.predicted_vehicle_service_demand_by_horizon(
        scenario,
        service_count,
        beta_discount=1.0,
    )
    if vehicle_demand.shape[:2] != (horizon_count, len(scenario["vehicle_ids"])):
        raise ValueError("predicted per-vehicle demand does not align with scenario positions")
    available_horizons = min(
        CACHE_ECONOMIC_DEMAND_HORIZONS,
        horizon_count,
        int(vehicle_demand.shape[0]),
    )
    demand_by_node = np.zeros(
        (node_count, CACHE_ECONOMIC_DEMAND_HORIZONS, service_count),
        dtype=np.float64,
    )
    for node_index in range(node_count):
        serviceability = cache_configuration._node_predicted_serviceability_weights(
            scenario,
            node_index,
            proposal_cache_nodes,
            context.cache_cfg,
        )
        if serviceability.shape != vehicle_demand.shape[:2]:
            raise ValueError("predicted serviceability and demand shapes do not align")
        per_horizon = np.einsum("hv,hvf->hf", serviceability, vehicle_demand)
        for horizon_index in range(available_horizons):
            if horizon_mask[horizon_index]:
                demand_by_node[node_index, horizon_index] = (
                    per_horizon[horizon_index] / float(task_vehicle_count)
                )

    scalar_start = 4 + 6 * service_count
    cost_index = scalar_start + 11
    loaded_index = scalar_start + 3
    released_index = scalar_start + 4
    diversity_index = scalar_start + 14
    for candidate_index, target_raw in enumerate(candidates[1:], start=1):
        target = np.asarray(target_raw, dtype=np.int8)
        if target.shape != current.shape:
            raise ValueError("Cache economic target shape mismatch")
        changed_nodes = np.flatnonzero(np.any(target != current, axis=1))
        if changed_nodes.shape != (1,):
            raise ValueError("Cache economic UPDATE candidates must change exactly one node")
        node_index = int(changed_nodes[0])
        row = primitive[candidate_index]
        if not np.isfinite(row).all():
            raise FloatingPointError("existing 81-D Cache node features are non-finite")

        result[candidate_index, 1] = row[0]
        result[candidate_index, 2:5] = row[1:4]
        result[candidate_index, 5] = row[cost_index]
        result[candidate_index, 6] = row[loaded_index]
        result[candidate_index, 7] = row[released_index]
        result[candidate_index, 8] = row[diversity_index]

        added_services = np.maximum(target[node_index] - current[node_index], 0.0)
        removed_services = np.maximum(current[node_index] - target[node_index], 0.0)
        demand = demand_by_node[node_index]
        result[candidate_index, 9:14] = demand @ added_services
        result[candidate_index, 14:19] = demand @ removed_services

    if not np.isfinite(result).all():
        raise FloatingPointError("Cache economic candidate features contain NaN/Inf")
    return result


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
