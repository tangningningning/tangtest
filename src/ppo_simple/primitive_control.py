"""Score-free primitive Cache/UAV candidates and sequential PPO application."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from decision import cache_configuration, uav_configuration

from . import interface


REPLICA_AWARE_POLICY_SCHEMA = (
    "adaptive_cache_operation_group_replica_aware_v8_popart"
)
REPLICA_AWARE_CACHE_FEATURE_NAMES = (
    "removed_replica_before_ratio",
    "removed_replica_after_ratio",
    "removed_after_single_replica",
    "removed_after_zero_replica",
    "added_replica_before_ratio",
    "added_replica_after_ratio",
)
REPLICA_AWARE_CACHE_FEATURE_DIM = len(REPLICA_AWARE_CACHE_FEATURE_NAMES)


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
        replica_aware = (
            str(getattr(context, "policy_schema", ""))
            == REPLICA_AWARE_POLICY_SCHEMA
        )
        return cls(
            num_cache_heads=len(context.action_spec.cache_node_ids),
            num_uav_heads=uav_count,
            observation_dim=int(context.observation_spec.observation_dim),
            cache_feature_dim=(
                17
                + 3 * service_count
                + (REPLICA_AWARE_CACHE_FEATURE_DIM if replica_aware else 0)
            ),
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


@dataclass(frozen=True)
class CacheOperationPrimitiveSelectionResult:
    """One sequential V7 Cache operation/candidate decision result."""

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
    *,
    include_replica_awareness: bool = False,
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
    replica_features = np.zeros(REPLICA_AWARE_CACHE_FEATURE_DIM, dtype=np.float64)
    if include_replica_awareness:
        current_cache_array = np.asarray(current_cache, dtype=np.int8)
        trial_cache_array = np.asarray(trial, dtype=np.int8)
        if current_cache_array.ndim != 2 or trial_cache_array.shape != current_cache_array.shape:
            raise AssertionError("replica-aware cache state shape check failed.")
        if current_cache_array.shape[1] != service_count or not current_cache_array.shape[0]:
            raise AssertionError("replica-aware cache state dimension check failed.")
        num_cache_nodes = int(current_cache_array.shape[0])

        removed_service = (
            None
            if not candidate.removed_service_ids
            else int(candidate.removed_service_ids[0])
        )
        added_service = (
            None
            if not candidate.added_service_ids
            else int(candidate.added_service_ids[0])
        )

        def _replica_ratio_pair(service_id: int | None) -> tuple[float, float]:
            if service_id is None:
                return 0.0, 0.0
            if service_id < 0 or service_id >= service_count:
                raise AssertionError("cache candidate service ID is out of range.")
            before_count = int(current_cache_array[:, service_id].sum())
            after_count = int(trial_cache_array[:, service_id].sum())
            return (
                before_count / float(num_cache_nodes),
                after_count / float(num_cache_nodes),
            )

        removed_before_ratio, removed_after_ratio = _replica_ratio_pair(removed_service)
        added_before_ratio, added_after_ratio = _replica_ratio_pair(added_service)
        removed_after_count = (
            0
            if removed_service is None
            else int(trial_cache_array[:, removed_service].sum())
        )
        replica_features = np.asarray(
            [
                removed_before_ratio,
                removed_after_ratio,
                float(removed_after_count == 1) if removed_service is not None else 0.0,
                float(removed_after_count == 0) if removed_service is not None else 0.0,
                added_before_ratio,
                added_after_ratio,
            ],
            dtype=np.float64,
        )
    feature_parts = [
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
    if include_replica_awareness:
        feature_parts.append(replica_features)
    values = np.concatenate(feature_parts)
    expected_dim = 17 + 3 * service_count + (
        REPLICA_AWARE_CACHE_FEATURE_DIM if include_replica_awareness else 0
    )
    if values.shape != (expected_dim,) or not np.all(np.isfinite(values)):
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


def sample_and_apply_v7_cache_operation_configuration(
    policy: Any,
    observation: np.ndarray,
    previous_cache_state: np.ndarray,
    previous_uav_positions: dict[str, Any],
    step: Any,
    context: Any,
    tensor_spec: PrimitiveTensorSpec,
    *,
    deterministic: bool = False,
    include_replica_awareness: bool = False,
) -> CacheOperationPrimitiveSelectionResult:
    """Sequentially sample V7 Cache operations and within-group candidates.

    The operation mask is derived from the production candidate enumeration at
    the current Cache state.  A NO_CHANGE decision ends that node's decision;
    it never invokes a second candidate categorical distribution.
    """

    operation_order = ("NO_CHANGE", "ADD", "DELETE", "SWAP")
    if tuple(getattr(policy, "operation_order", ())) != operation_order:
        raise ValueError("V7 policy operation order must be NO_CHANGE/ADD/DELETE/SWAP")
    cache = np.asarray(previous_cache_state, dtype=np.int8).copy()
    previous_cache = cache.copy()
    previous_uav = {
        uav_id: np.asarray(previous_uav_positions[uav_id], dtype=np.float64).copy()
        for uav_id in context.action_spec.uav_ids
    }
    current_uav = {key: value.copy() for key, value in previous_uav.items()}
    if tensor_spec.num_cache_heads != len(context.action_spec.cache_node_ids):
        raise ValueError("V7 tensor spec cache head count does not match action spec")

    operation_actions = np.zeros(tensor_spec.num_cache_heads, dtype=np.int64)
    operation_masks = np.zeros(
        (tensor_spec.num_cache_heads, len(operation_order)),
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
        candidates = cache_configuration.enumerate_cache_primitive_actions(
            cache,
            node_index,
            context.service_catalog,
            context.cache_nodes,
        )
        operation_mask = np.zeros(len(operation_order), dtype=bool)
        operation_mask[0] = True
        for candidate in candidates:
            try:
                operation_index = operation_order.index(str(candidate.operation_type))
            except ValueError as exc:
                raise ValueError(
                    f"unsupported production cache operation: {candidate.operation_type}"
                ) from exc
            operation_mask[operation_index] = True
        sampled_operation = policy.sample_cache_operation(
            conditional_context,
            node_index,
            operation_mask,
            deterministic=bool(deterministic),
        )
        selected_operation_index = int(sampled_operation.selected_index)
        selected_operation = operation_order[selected_operation_index]
        operation_actions[node_index] = selected_operation_index
        operation_masks[node_index] = operation_mask
        operation_old_log_probs[node_index] = float(
            sampled_operation.log_prob.detach().cpu()
        )
        operation_entropies[node_index] = float(
            sampled_operation.entropy.detach().cpu()
        )
        cache_contexts[node_index] = conditional_context
        total_log_prob = (
            sampled_operation.log_prob
            if total_log_prob is None
            else total_log_prob + sampled_operation.log_prob
        )
        total_entropy = (
            sampled_operation.entropy
            if total_entropy is None
            else total_entropy + sampled_operation.entropy
        )

        if selected_operation_index == 0:
            cache_actions.append(
                {
                    "head_type": "cache",
                    "head_id": str(node_id),
                    "head_index": node_index,
                    "candidate_count": 0,
                    "selected_index": None,
                    "operation_type": "NO_CHANGE",
                    "added_service_ids": [],
                    "removed_service_ids": [],
                    "actual_changed": False,
                }
            )
            continue

        cache_triggers[node_index] = 1
        group_candidates = [
            candidate
            for candidate in candidates
            if str(candidate.operation_type) == selected_operation
        ]
        if not group_candidates:
            raise AssertionError(
                "V7 selected an operation whose production candidate group is empty"
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
                    include_replica_awareness=include_replica_awareness,
                )
                for candidate in group_candidates
            ]
        )
        count = len(group_candidates)
        if count > tensor_spec.cache_max_candidates:
            raise AssertionError("V7 cache candidate count exceeds padded capacity")
        sampled_candidate = policy.sample_cache_candidate_v7(
            conditional_context,
            candidate_features,
            deterministic=bool(deterministic),
        )
        selected_index = int(sampled_candidate.selected_index)
        selected = group_candidates[selected_index]
        applied = cache_configuration.apply_cache_candidate(
            cache,
            node_index,
            selected.after_node_cache,
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
                "selected_index": selected_index,
                "operation_type": selected.operation_type,
                "added_service_ids": list(selected.added_service_ids),
                "removed_service_ids": list(selected.removed_service_ids),
                "actual_changed": bool(applied["actual_changed"]),
            }
        )

    if total_log_prob is None or total_entropy is None:
        raise RuntimeError("V7 Cache operation selection produced no heads")
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
    )


def sample_and_apply_v8_cache_operation_configuration(
    policy: Any,
    observation: np.ndarray,
    previous_cache_state: np.ndarray,
    previous_uav_positions: dict[str, Any],
    step: Any,
    context: Any,
    tensor_spec: PrimitiveTensorSpec,
    *,
    deterministic: bool = False,
) -> CacheOperationPrimitiveSelectionResult:
    """Run the unchanged V7 hierarchy with V8 replica-aware candidate features."""

    return sample_and_apply_v7_cache_operation_configuration(
        policy,
        observation,
        previous_cache_state,
        previous_uav_positions,
        step,
        context,
        tensor_spec,
        deterministic=bool(deterministic),
        include_replica_awareness=True,
    )
