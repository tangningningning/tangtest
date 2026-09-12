"""Score-free primitive Cache/UAV candidates and sequential PPO application."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from decision import cache_configuration, uav_configuration

from . import interface


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
            cache_feature_dim=17 + 3 * service_count,
            uav_feature_dim=16 + uav_count,
            cache_max_candidates=1 + service_count + (service_count * service_count) // 4,
            uav_max_candidates=(
                1
                + len(search.coarse_radius_fractions) * search.coarse_num_angles
                + search.refine_rounds * search.refine_num_angles
            ),
        )


@dataclass(frozen=True)
class PrimitiveSelectionResult:
    final_cache_state: np.ndarray
    final_uav_positions: dict[str, tuple[float, float]]
    total_log_prob: Any
    total_entropy: Any
    cache_contexts: np.ndarray
    cache_candidate_features: np.ndarray
    cache_candidate_masks: np.ndarray
    cache_selected_indices: np.ndarray
    cache_old_log_probs: np.ndarray
    uav_contexts: np.ndarray
    uav_candidate_features: np.ndarray
    uav_candidate_masks: np.ndarray
    uav_selected_indices: np.ndarray
    uav_old_log_probs: np.ndarray
    cache_conditional_actions: tuple[dict[str, Any], ...]
    uav_conditional_actions: tuple[dict[str, Any], ...]
    cache_candidate_counts: np.ndarray
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
    return interface.build_step_observation(
        cache_state,
        uav_positions,
        step,
        context.observation_spec,
        context.flight_boundary,
        context.train_origin_start,
        context.train_origin_end,
        time_to_go_override=time_to_go_override,
    )


def _cache_candidate_feature(
    candidate: cache_configuration.CachePrimitiveCandidate,
    current_cache: np.ndarray,
    previous_cache: np.ndarray,
    cache_triggers: np.ndarray,
    predicted_popularity: dict[int, float],
    context: Any,
) -> np.ndarray:
    service_count = int(context.service_catalog["num_services"])
    node_count = len(context.cache_nodes["node_ids"])
    role = str(context.cache_nodes["node_roles"][candidate.node_index]).upper()
    role_feature = np.asarray(
        [float(role == "SV"), float(role == "RSU"), float(role == "UAV")],
        dtype=np.float64,
    )
    operation_types = ("NO_CHANGE", "ADD", "DELETE", "SWAP")
    operation_feature = np.asarray(
        [float(candidate.operation_type == value) for value in operation_types],
        dtype=np.float64,
    )
    if candidate.operation_type not in operation_types:
        raise ValueError(f"unsupported cache primitive operation: {candidate.operation_type}")
    before = np.asarray(candidate.before_node_cache, dtype=np.float64)
    after = np.asarray(candidate.after_node_cache, dtype=np.float64)
    delta = after - before
    sizes = np.asarray(context.service_catalog["service_sizes_mb"], dtype=np.float64)
    capacity = float(context.cache_nodes["capacities_mb"][candidate.node_index])
    trial = np.asarray(current_cache, dtype=np.int8).copy()
    trial[candidate.node_index] = candidate.after_node_cache
    total_cost, loading_energy = cache_configuration.cache_update_cost(
        previous_cache,
        trial,
        cache_triggers,
        sizes,
        context.cache_nodes,
        context.cache_cfg,
    )
    popularity = np.asarray(
        [float(predicted_popularity[index]) for index in range(service_count)],
        dtype=np.float64,
    )
    added_size = float(sum(sizes[index] for index in candidate.added_service_ids))
    removed_size = float(sum(sizes[index] for index in candidate.removed_service_ids))
    added_popularity = float(sum(popularity[index] for index in candidate.added_service_ids))
    removed_popularity = float(sum(popularity[index] for index in candidate.removed_service_ids))
    before_used = cache_configuration.node_used_capacity_mb(before.astype(np.int8), sizes)
    after_used = cache_configuration.node_used_capacity_mb(after.astype(np.int8), sizes)
    values = np.concatenate(
        [
            np.asarray(
                [float(candidate.node_index) / max(node_count - 1, 1)],
                dtype=np.float64,
            ),
            role_feature,
            operation_feature,
            before,
            after,
            delta,
            np.asarray(
                [
                    before_used / max(capacity, 1.0e-12),
                    after_used / max(capacity, 1.0e-12),
                    (capacity - after_used) / max(capacity, 1.0e-12),
                    added_size / max(capacity, 1.0e-12),
                    removed_size / max(capacity, 1.0e-12),
                    added_popularity,
                    removed_popularity,
                    math.log1p(max(float(total_cost), 0.0)),
                    math.log1p(max(float(loading_energy[candidate.node_index]), 0.0)),
                ],
                dtype=np.float64,
            ),
        ]
    )
    if values.shape != (17 + 3 * service_count,) or not np.all(np.isfinite(values)):
        raise AssertionError("cache primitive candidate feature shape/finite check failed。")
    return values.astype(np.float32)


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


def _uav_candidate_feature(
    candidate: uav_configuration.UavPrimitiveCandidate,
    current_uav: dict[str, np.ndarray],
    previous_uav: dict[str, np.ndarray],
    uav_triggers: dict[str, int],
    predicted_vehicle_positions: dict[str, Any],
    context: Any,
) -> np.ndarray:
    uav_ids = tuple(context.action_spec.uav_ids)
    uav_index = uav_ids.index(candidate.uav_id)
    search = uav_configuration.uav_search_config_from_project(context.cfg)
    trial = {key: np.asarray(value, dtype=np.float64).copy() for key, value in current_uav.items()}
    trial[candidate.uav_id] = np.asarray(candidate.after_position, dtype=np.float64).copy()
    movement_energy, movement_cost, _, _, _ = uav_configuration.movement_summary(
        previous_uav,
        trial,
        uav_triggers,
        search,
    )
    before_normalized = _normalized_xy(candidate.before_position, context.flight_boundary)
    after_normalized = _normalized_xy(candidate.after_position, context.flight_boundary)
    x_min, y_min, x_max, y_max = [float(value) for value in context.flight_boundary]
    scale = np.asarray([x_max - x_min, y_max - y_min], dtype=np.float64)
    delta = (np.asarray(candidate.after_position) - np.asarray(candidate.before_position)) / scale
    distance = float(np.linalg.norm(np.asarray(candidate.after_position) - np.asarray(candidate.before_position)))
    vehicle_xy = np.stack(
        [
            _normalized_xy(predicted_vehicle_positions[vehicle_id], context.flight_boundary)
            for vehicle_id in context.observation_spec.vehicle_ids
        ]
    )
    uav_one_hot = np.zeros(len(uav_ids), dtype=np.float64)
    uav_one_hot[uav_index] = 1.0
    values = np.concatenate(
        [
            np.asarray([float(uav_index) / max(len(uav_ids) - 1, 1)], dtype=np.float64),
            uav_one_hot,
            np.asarray(
                [
                    float(candidate.operation_type == "STAY"),
                    float(candidate.operation_type == "MOVE"),
                ],
                dtype=np.float64,
            ),
            before_normalized,
            after_normalized,
            delta,
            np.asarray(
                [
                    distance / max(search.max_move_distance_m, 1.0e-12),
                    math.log1p(max(float(movement_energy), 0.0)),
                    math.log1p(max(float(movement_cost), 0.0)),
                ],
                dtype=np.float64,
            ),
            vehicle_xy.mean(axis=0),
            vehicle_xy.std(axis=0),
        ]
    )
    if values.shape != (16 + len(uav_ids),) or not np.all(np.isfinite(values)):
        raise AssertionError("UAV primitive candidate feature shape/finite check failed。")
    return values.astype(np.float32)


def sample_and_apply_primitive_configuration(
    policy: Any,
    trigger_sample: Any,
    previous_cache_state: np.ndarray,
    previous_uav_positions: dict[str, Any],
    cache_triggers: np.ndarray,
    uav_triggers: dict[str, int],
    step: Any,
    context: Any,
    tensor_spec: PrimitiveTensorSpec,
    *,
    deterministic: bool = False,
    conditional_action_mode: str | None = None,
) -> PrimitiveSelectionResult:
    """Sequentially sample and apply at most one primitive per triggered head."""

    if conditional_action_mode not in (
        None,
        "candidate_deterministic",
        "grouped_deterministic",
        "stochastic",
    ):
        raise ValueError(
            "conditional_action_mode must be one of: candidate_deterministic, "
            "grouped_deterministic, stochastic"
        )

    cache = np.asarray(previous_cache_state, dtype=np.int8).copy()
    previous_cache = cache.copy()
    previous_uav = {
        uav_id: np.asarray(previous_uav_positions[uav_id], dtype=np.float64).copy()
        for uav_id in context.action_spec.uav_ids
    }
    current_uav = {key: value.copy() for key, value in previous_uav.items()}
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
        (tensor_spec.num_cache_heads, tensor_spec.cache_max_candidates), dtype=bool
    )
    cache_masks[:, 0] = True
    cache_selected = np.zeros(tensor_spec.num_cache_heads, dtype=np.int64)
    cache_old_log_probs = np.zeros(tensor_spec.num_cache_heads, dtype=np.float32)
    cache_counts = np.zeros(tensor_spec.num_cache_heads, dtype=np.int64)
    uav_contexts = np.zeros(
        (tensor_spec.num_uav_heads, tensor_spec.observation_dim), dtype=np.float32
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
        (tensor_spec.num_uav_heads, tensor_spec.uav_max_candidates), dtype=bool
    )
    uav_masks[:, 0] = True
    uav_selected = np.zeros(tensor_spec.num_uav_heads, dtype=np.int64)
    uav_old_log_probs = np.zeros(tensor_spec.num_uav_heads, dtype=np.float32)
    uav_counts = np.zeros(tensor_spec.num_uav_heads, dtype=np.int64)
    cache_actions: list[dict[str, Any]] = []
    uav_actions: list[dict[str, Any]] = []
    conditional_log_prob = trigger_sample.log_prob.new_zeros(())
    conditional_entropy = trigger_sample.entropy.new_zeros(())

    for node_index, node_id in enumerate(context.action_spec.cache_node_ids):
        if int(cache_triggers[node_index]) == 0:
            continue
        conditional_context = _intermediate_observation(cache, current_uav, step, context)
        candidates = cache_configuration.enumerate_cache_primitive_actions(
            cache,
            node_index,
            context.service_catalog,
            context.cache_nodes,
        )
        candidate_features = np.stack(
            [
                _cache_candidate_feature(
                    candidate,
                    cache,
                    previous_cache,
                    cache_triggers,
                    step.predicted_service_popularity,
                    context,
                )
                for candidate in candidates
            ]
        )
        count = len(candidates)
        if count > tensor_spec.cache_max_candidates:
            raise AssertionError("cache candidate count exceeds padded tensor capacity。")
        if conditional_action_mode == "grouped_deterministic":
            sampled = policy.sample_cache_candidate_grouped_deterministic(
                conditional_context,
                candidate_features,
            )
        elif conditional_action_mode == "candidate_deterministic" or (
            conditional_action_mode is None and deterministic
        ):
            sampled = policy.sample_cache_candidate(
                conditional_context,
                candidate_features,
                deterministic=True,
            )
        else:
            sampled = policy.sample_cache_candidate(conditional_context, candidate_features)
        selected_index = int(sampled.selected_index)
        selected = candidates[selected_index]
        applied = cache_configuration.apply_cache_candidate(
            cache,
            node_index,
            selected.after_node_cache,
            context.service_catalog,
            context.cache_nodes,
        )
        cache_contexts[node_index] = conditional_context
        cache_features[node_index, :count] = candidate_features
        cache_masks[node_index, :count] = True
        cache_selected[node_index] = selected_index
        cache_old_log_probs[node_index] = float(sampled.log_prob.detach().cpu())
        cache_counts[node_index] = count
        conditional_log_prob = conditional_log_prob + sampled.log_prob
        conditional_entropy = conditional_entropy + sampled.entropy
        cache_actions.append(
            {
                "head_type": "cache",
                "head_id": str(node_id),
                "head_index": node_index,
                "candidate_count": count,
                "selected_index": selected_index,
                "operation_type": selected.operation_type,
                "added_service_ids": list(selected.added_service_ids),
                "removed_service_ids": list(selected.removed_service_ids),
                "actual_changed": bool(applied["actual_changed"]),
            }
        )

    for uav_index, uav_id in enumerate(context.action_spec.uav_ids):
        if int(uav_triggers[uav_id]) == 0:
            continue
        conditional_context = _intermediate_observation(cache, current_uav, step, context)
        candidates = uav_configuration.enumerate_uav_primitive_actions(
            current_uav,
            previous_uav,
            uav_id,
            uav_triggers,
            context.cfg,
            context.flight_boundary,
        )
        candidate_features = np.stack(
            [
                _uav_candidate_feature(
                    candidate,
                    current_uav,
                    previous_uav,
                    uav_triggers,
                    step.predicted_vehicle_positions,
                    context,
                )
                for candidate in candidates
            ]
        )
        count = len(candidates)
        if count > tensor_spec.uav_max_candidates:
            raise AssertionError("UAV candidate count exceeds padded tensor capacity。")
        if conditional_action_mode == "grouped_deterministic":
            sampled = policy.sample_uav_candidate_grouped_deterministic(
                conditional_context,
                candidate_features,
            )
        elif conditional_action_mode == "candidate_deterministic" or (
            conditional_action_mode is None and deterministic
        ):
            sampled = policy.sample_uav_candidate(
                conditional_context,
                candidate_features,
                deterministic=True,
            )
        else:
            sampled = policy.sample_uav_candidate(conditional_context, candidate_features)
        selected_index = int(sampled.selected_index)
        selected = candidates[selected_index]
        applied = uav_configuration.apply_uav_candidate(
            current_uav,
            previous_uav,
            uav_id,
            selected.after_position,
            uav_triggers,
            context.cfg,
            context.flight_boundary,
        )
        uav_contexts[uav_index] = conditional_context
        uav_features[uav_index, :count] = candidate_features
        uav_masks[uav_index, :count] = True
        uav_selected[uav_index] = selected_index
        uav_old_log_probs[uav_index] = float(sampled.log_prob.detach().cpu())
        uav_counts[uav_index] = count
        conditional_log_prob = conditional_log_prob + sampled.log_prob
        conditional_entropy = conditional_entropy + sampled.entropy
        uav_actions.append(
            {
                "head_type": "uav",
                "head_id": str(uav_id),
                "head_index": uav_index,
                "candidate_count": count,
                "selected_index": selected_index,
                "operation_type": selected.operation_type,
                "before_position": [float(value) for value in selected.before_position],
                "after_position": [float(value) for value in selected.after_position],
                "actual_moved": bool(applied["actual_moved"]),
            }
        )

    cache_configuration.validate_cache_shape_and_capacity(
        cache,
        context.service_catalog,
        context.cache_nodes,
    )
    return PrimitiveSelectionResult(
        final_cache_state=cache.copy(),
        final_uav_positions={
            key: (float(value[0]), float(value[1])) for key, value in current_uav.items()
        },
        total_log_prob=trigger_sample.log_prob + conditional_log_prob,
        total_entropy=trigger_sample.entropy + conditional_entropy,
        cache_contexts=cache_contexts,
        cache_candidate_features=cache_features,
        cache_candidate_masks=cache_masks,
        cache_selected_indices=cache_selected,
        cache_old_log_probs=cache_old_log_probs,
        uav_contexts=uav_contexts,
        uav_candidate_features=uav_features,
        uav_candidate_masks=uav_masks,
        uav_selected_indices=uav_selected,
        uav_old_log_probs=uav_old_log_probs,
        cache_conditional_actions=tuple(cache_actions),
        uav_conditional_actions=tuple(uav_actions),
        cache_candidate_counts=cache_counts,
        uav_candidate_counts=uav_counts,
    )
