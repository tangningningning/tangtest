"""Environment-facing rollout and GAE for the minimal PPO v2 trainer."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Callable

import numpy as np

from decision import cache_configuration, slot_environment

from . import interface
from .evaluation_metrics import task_metrics_from_result
from .interface import _copy_uav_positions
from ablations.no_prediction import (
    NO_PREDICTION_POLICY_SCHEMA,
    is_no_prediction_context,
    mask_no_prediction_observation,
)
from .policy import (
    AdaptiveCacheOnlyActorCritic,
    AdaptiveReplicaAwareCacheOnlyActorCritic,
    AdaptiveNoPredictionCacheOnlyActorCritic,
    AdaptiveJointCacheUAVPPOActorCritic,
    AdaptivePrimitiveActorCritic,
    BernoulliActorCritic,
    PolicySample,
)
from .primitive_control import (
    PrimitiveTensorSpec,
    sample_and_apply_primitive_configuration,
    sample_and_apply_replica_aware_cache_operation_configuration,
    sample_and_apply_no_prediction_cache_operation_configuration,
    sample_and_apply_cache_operation_configuration,
    sample_and_apply_joint_cache_uav_operation_configuration,
)


def _snapshot_trigger_feature(value: Any) -> np.ndarray:
    """Copy a Trigger feature to finite, detached rollout-owned float32 memory."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float32).copy()
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Cache Trigger rollout snapshot is non-finite")
    return result


@dataclass(frozen=True)
class SimpleTransition:
    observation: np.ndarray
    action: np.ndarray
    action_mask: np.ndarray
    old_log_prob: float
    old_value: float
    reward: float
    done: bool
    slot: int
    origin_index: int
    actual_task_utility: float
    cache_cost: float
    movement_cost: float
    task_count: int
    total_task_delay_s: float
    total_vehicle_energy_j: float
    execution_target_counts: dict[str, int]
    actual_cache_update_count: int
    actual_uav_movement_count: int
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
    policy_selection_seconds: float
    configuration_seconds: float
    actual_matching_seconds: float
    online_k100_configuration_calls: int
    matching_summary: dict[str, Any]
    origin_runtime_seconds: float
    # State at the beginning of this transition.  proposed uses it to replay exact
    # operation counterfactuals without reconstructing the rollout prefix.
    previous_cache_state: np.ndarray
    previous_uav_positions: dict[str, tuple[float, float]]
    next_cache_state: np.ndarray
    next_uav_positions: dict[str, tuple[float, float]]
    cache_operation_actions: np.ndarray | None = None
    cache_operation_masks: np.ndarray | None = None
    cache_operation_log_probs: np.ndarray | None = None
    cache_operation_entropies: np.ndarray | None = None
    uav_operation_actions: np.ndarray | None = None
    uav_operation_masks: np.ndarray | None = None
    uav_operation_log_probs: np.ndarray | None = None
    uav_operation_entropies: np.ndarray | None = None
    uav_operation_probabilities: np.ndarray | None = None
    uav_operation_logit_margins: np.ndarray | None = None
    # V3 Cache Trigger-only pre-decision features are frozen behavior-policy
    # snapshots. PPO consumes these arrays verbatim across all update epochs.
    cache_trigger_node_local_workload: np.ndarray | None = None
    cache_trigger_candidate_summary: np.ndarray | None = None
    cache_trigger_aux: np.ndarray | None = None
    cache_trigger_legal_candidate_counts: np.ndarray | None = None
    cache_trigger_node_local_workload_norms: np.ndarray | None = None
    cache_trigger_candidate_summary_norms: np.ndarray | None = None
    cache_trigger_aux_norms: np.ndarray | None = None
    cache_trigger_weighted_loaded_mb: np.ndarray | None = None
    cache_trigger_weighted_cache_cost: np.ndarray | None = None
    cache_trigger_weighted_predicted_demand_gain: np.ndarray | None = None
    pre_trigger_and_execution_candidate_logit_max_diff: float = 0.0


@dataclass(frozen=True)
class SimpleRollout:
    transitions: tuple[SimpleTransition, ...]
    observations: np.ndarray
    actions: np.ndarray
    action_masks: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    cache_contexts: np.ndarray
    cache_candidate_features: np.ndarray
    cache_candidate_masks: np.ndarray
    cache_selected_indices: np.ndarray
    uav_contexts: np.ndarray
    uav_candidate_features: np.ndarray
    uav_candidate_masks: np.ndarray
    uav_selected_indices: np.ndarray
    next_cache_state: np.ndarray
    next_uav_positions: dict[str, tuple[float, float]]
    bootstrap_observation: np.ndarray | None
    cache_operation_actions: np.ndarray | None = None
    cache_operation_masks: np.ndarray | None = None
    cache_operation_log_probs: np.ndarray | None = None
    cache_operation_entropies: np.ndarray | None = None
    uav_operation_actions: np.ndarray | None = None
    uav_operation_masks: np.ndarray | None = None
    uav_operation_log_probs: np.ndarray | None = None
    uav_operation_entropies: np.ndarray | None = None
    # Candidate old log-probabilities are already stored per transition.  The
    # rollout-level stacks make branch diagnostics indexable without logging
    # any new arrays or changing the sampled action path.
    cache_old_log_probs: np.ndarray | None = None
    uav_old_log_probs: np.ndarray | None = None
    cache_trigger_node_local_workload: np.ndarray | None = None
    cache_trigger_candidate_summary: np.ndarray | None = None
    cache_trigger_aux: np.ndarray | None = None
    cache_trigger_legal_candidate_counts: np.ndarray | None = None
    cache_trigger_node_local_workload_norms: np.ndarray | None = None
    cache_trigger_candidate_summary_norms: np.ndarray | None = None
    cache_trigger_aux_norms: np.ndarray | None = None
    cache_trigger_weighted_loaded_mb: np.ndarray | None = None
    cache_trigger_weighted_cache_cost: np.ndarray | None = None
    cache_trigger_weighted_predicted_demand_gain: np.ndarray | None = None
    pre_trigger_and_execution_candidate_logit_max_diff: float = 0.0


def _observation(
    previous_cache: np.ndarray,
    previous_uav: dict[str, Any],
    step: Any,
    observation_spec: interface.ConfigurationObservationSpec,
    flight_boundary: tuple[float, float, float, float],
    train_origin_start: int,
    train_origin_end: int,
    *,
    time_to_go_override: float | None = None,
    no_prediction: bool = False,
) -> np.ndarray:
    observation = interface.build_step_observation(
        previous_cache,
        previous_uav,
        step,
        observation_spec,
        flight_boundary,
        train_origin_start,
        train_origin_end,
        time_to_go_override=time_to_go_override,
    )
    if no_prediction:
        return mask_no_prediction_observation(observation, observation_spec)
    return observation


def _check_reward(result: slot_environment.SlotExecutionResult) -> float:
    if result.system_utility is None:
        raise AssertionError("feasible step 缺少 system_utility。")
    expected = float(result.actual_task_utility) - float(result.cache_cost) - float(result.movement_cost)
    if not np.isclose(float(result.system_utility), expected, rtol=1e-12, atol=1e-12):
        raise AssertionError("PPO v2 reward 不是 execute_slot().system_utility。")
    return float(result.system_utility)


def _check_request_contract(
    previous_cache: np.ndarray,
    previous_uav: dict[str, Any],
    result: slot_environment.SlotExecutionResult,
    action_spec: interface.TriggerActionSpec,
) -> None:
    if result.next_cache_state is None or result.next_uav_positions is None:
        raise AssertionError("可行 environment transition 缺少 next state。")
    next_cache = np.asarray(result.next_cache_state, dtype=np.int8)
    for node_index, request in enumerate(result.input_cache_triggers):
        if int(request) == 0 and not np.array_equal(previous_cache[node_index], next_cache[node_index]):
            raise AssertionError("request=0 但 cache actual state 发生变化。")
    for uav_id in action_spec.uav_ids:
        if int(result.input_uav_triggers[uav_id]) != 0:
            continue
        before = np.asarray(previous_uav[uav_id], dtype=np.float64)
        after = np.asarray(result.next_uav_positions[uav_id], dtype=np.float64)
        if not np.allclose(before, after, rtol=0.0, atol=1.0e-6):
            raise AssertionError("request=0 但 UAV actual state 发生变化。")


def collect_rollout(
    policy: BernoulliActorCritic | AdaptivePrimitiveActorCritic,
    context: Any,
    aligned_steps: list[Any],
    *,
    initial_cache: np.ndarray | None = None,
    initial_uav: dict[str, Any] | None = None,
    bootstrap_step: Any | None = None,
    terminal_origin_index: int | None = None,
    observation_time_to_go: Callable[[int], float] | None = None,
    deterministic: bool = False,
    conditional_action_mode: str | None = None,
    policy_diagnostic_callback: Callable[[int, Any], None] | None = None,
    action_transform: Callable[[np.ndarray], Any] | None = None,
) -> SimpleRollout:
    """Collect one contiguous request rollout from a supplied environment state.

    A rollout chunk is a truncation boundary, not an environment reset.  The
    caller supplies the state produced by the previous chunk and, when the
    chunk is not terminal, the next aligned step used to build the critic
    bootstrap observation.
    """

    if not aligned_steps:
        raise ValueError("aligned_steps 不能为空。")
    execution_steps = aligned_steps
    if terminal_origin_index is None:
        terminal_origin_index = int(execution_steps[-1].origin_index)
    terminal_origin_index = int(terminal_origin_index)
    if int(execution_steps[-1].origin_index) > terminal_origin_index:
        raise ValueError("aligned_steps 不能超过 true episode terminal。")

    cache_source = context.initial_cache if initial_cache is None else initial_cache
    previous_cache = np.asarray(cache_source, dtype=np.int8).copy()
    if previous_cache.shape != np.asarray(context.initial_cache).shape:
        raise ValueError("initial_cache shape 与 environment context 不一致。")
    uav_source = context.initial_uav if initial_uav is None else initial_uav
    previous_uav = _copy_uav_positions(uav_source)
    tensor_spec = PrimitiveTensorSpec.create(context)
    transitions: list[SimpleTransition] = []
    for step_index, step in enumerate(execution_steps):
        origin_started_at = time.perf_counter()
        time_to_go_override = (
            None
            if observation_time_to_go is None
            else float(observation_time_to_go(int(step.origin_index)))
        )
        observation = _observation(
            previous_cache,
            previous_uav,
            step,
            context.observation_spec,
            context.flight_boundary,
            context.train_origin_start,
            context.train_origin_end,
            time_to_go_override=time_to_go_override,
            no_prediction=is_no_prediction_context(context),
        )
        if observation.shape != (int(context.observation_spec.observation_dim),):
            raise AssertionError("PPO v2 observation 不符合 observation_spec。")
        selection_started_at = time.perf_counter()
        cache_primitive: Any | None = None
        joint_primitive: Any | None = None
        if context.configuration_control_mode == "adaptive_joint_cache_uav_reconfiguration":
            if not isinstance(policy, AdaptiveJointCacheUAVPPOActorCritic):
                raise TypeError(
                    "adaptive_joint_cache_uav_reconfiguration requires the proposed joint policy"
                )
            if action_transform is not None:
                raise ValueError("proposed joint rollout does not support action_transform")
            if policy_diagnostic_callback is not None:
                raise ValueError(
                    "proposed joint rollout does not support binary-trigger diagnostics"
                )
            joint_primitive = sample_and_apply_joint_cache_uav_operation_configuration(
                policy,
                observation,
                previous_cache,
                previous_uav,
                step,
                context,
                tensor_spec,
                deterministic=bool(deterministic),
            )
            sample = PolicySample(
                action=np.asarray(joint_primitive.derived_action, dtype=np.int8),
                log_prob=joint_primitive.total_log_prob,
                entropy=joint_primitive.total_entropy,
                value=joint_primitive.value,
                action_mask=np.ones(
                    (int(context.action_spec.action_dim), 2),
                    dtype=bool,
                ),
            )
        elif context.configuration_control_mode == "policy_primitive_cache_operation":
            if not isinstance(policy, AdaptiveCacheOnlyActorCritic):
                raise TypeError(
                    "policy_primitive_cache_operation requires a Cache-only policy"
                )
            if action_transform is not None:
                raise ValueError("Cache-only rollout does not support action_transform")
            if policy_diagnostic_callback is not None:
                raise ValueError(
                    "Cache-only rollout does not support binary-trigger diagnostics"
                )
            if context.policy_schema == NO_PREDICTION_POLICY_SCHEMA:
                if (
                    type(policy)
                    is not AdaptiveNoPredictionCacheOnlyActorCritic
                ):
                    raise TypeError(
                        "no-prediction schema requires the exact no-prediction Cache-only policy"
                    )
                cache_primitive = (
                    sample_and_apply_no_prediction_cache_operation_configuration(
                        policy,
                        observation,
                        previous_cache,
                        previous_uav,
                        step,
                        context,
                        tensor_spec,
                        deterministic=bool(deterministic),
                    )
                )
            elif isinstance(
                policy, AdaptiveReplicaAwareCacheOnlyActorCritic
            ):
                cache_primitive = sample_and_apply_replica_aware_cache_operation_configuration(
                    policy,
                    observation,
                    previous_cache,
                    previous_uav,
                    step,
                    context,
                    tensor_spec,
                    deterministic=bool(deterministic),
                )
            else:
                cache_primitive = sample_and_apply_cache_operation_configuration(
                    policy,
                    observation,
                    previous_cache,
                    previous_uav,
                    step,
                    context,
                    tensor_spec,
                    deterministic=bool(deterministic),
                )
            sample = PolicySample(
                action=np.asarray(cache_primitive.derived_action, dtype=np.int8),
                log_prob=cache_primitive.total_log_prob,
                entropy=cache_primitive.total_entropy,
                value=cache_primitive.value,
                action_mask=np.ones(
                    (int(context.action_spec.action_dim), 2),
                    dtype=bool,
                ),
            )
        else:
            action_mask_provider = interface.build_trigger_action_mask_provider(
                previous_cache,
                previous_uav,
                context.service_catalog,
                context.cfg,
                context.cache_nodes,
                context.flight_boundary,
                context.action_spec,
            )
            sample = policy.sample(
                observation,
                action_mask=action_mask_provider,
                deterministic=bool(deterministic),
                record_diagnostics=policy_diagnostic_callback is not None,
            )
        if sample.action.shape != (int(context.action_spec.action_dim),):
            raise AssertionError("PPO v2 action 不符合 action_spec。")
        if policy_diagnostic_callback is not None:
            if (
                sample.action_probabilities is None
                or sample.deterministic_action is None
            ):
                raise RuntimeError(
                    "policy diagnostics requested but policy sample did not provide "
                    "trigger distribution details"
                )
            policy_diagnostic_callback(int(step.origin_index), sample)
        if action_transform is not None:
            transformed_action = np.asarray(
                action_transform(np.asarray(sample.action, dtype=np.int8).copy()),
                dtype=np.int8,
            ).reshape(-1)
            expected_action_shape = (int(context.action_spec.action_dim),)
            if transformed_action.shape != expected_action_shape:
                raise ValueError(
                    "action_transform returned an action with an invalid shape: "
                    f"{transformed_action.shape}, expected {expected_action_shape}"
                )
            if not bool(np.all(np.isin(transformed_action, [0, 1]))):
                raise ValueError("action_transform must return a binary request vector")
            transformed_indices = np.arange(int(context.action_spec.action_dim))
            if not bool(
                np.all(
                    np.asarray(sample.action_mask, dtype=bool)[
                        transformed_indices, transformed_action
                    ]
                )
            ):
                raise ValueError("action_transform returned a request blocked by the action mask")
            sample = replace(sample, action=transformed_action)
        cache_triggers, uav_triggers = context.action_spec.action_vector_to_triggers(sample.action)
        cache_contexts = np.zeros(
            (tensor_spec.num_cache_heads, tensor_spec.observation_dim), dtype=np.float32
        )
        cache_candidate_features = np.zeros(
            (
                tensor_spec.num_cache_heads,
                tensor_spec.cache_max_candidates,
                tensor_spec.cache_feature_dim,
            ),
            dtype=np.float32,
        )
        cache_candidate_masks = np.zeros(
            (tensor_spec.num_cache_heads, tensor_spec.cache_max_candidates), dtype=bool
        )
        cache_candidate_masks[:, 0] = True
        cache_selected_indices = np.zeros(tensor_spec.num_cache_heads, dtype=np.int64)
        cache_old_log_probs = np.zeros(tensor_spec.num_cache_heads, dtype=np.float32)
        uav_contexts = np.zeros(
            (tensor_spec.num_uav_heads, tensor_spec.observation_dim), dtype=np.float32
        )
        uav_candidate_features = np.zeros(
            (
                tensor_spec.num_uav_heads,
                tensor_spec.uav_max_candidates,
                tensor_spec.uav_feature_dim,
            ),
            dtype=np.float32,
        )
        uav_candidate_masks = np.zeros(
            (tensor_spec.num_uav_heads, tensor_spec.uav_max_candidates), dtype=bool
        )
        uav_candidate_masks[:, 0] = True
        uav_selected_indices = np.zeros(tensor_spec.num_uav_heads, dtype=np.int64)
        uav_old_log_probs = np.zeros(tensor_spec.num_uav_heads, dtype=np.float32)
        cache_conditional_actions: tuple[dict[str, Any], ...] = ()
        uav_conditional_actions: tuple[dict[str, Any], ...] = ()
        cache_candidate_counts = np.zeros(tensor_spec.num_cache_heads, dtype=np.int64)
        uav_candidate_counts = np.zeros(tensor_spec.num_uav_heads, dtype=np.int64)
        cache_trigger_node_local_workload: np.ndarray | None = None
        cache_trigger_candidate_summary: np.ndarray | None = None
        cache_trigger_aux: np.ndarray | None = None
        cache_trigger_legal_candidate_counts: np.ndarray | None = None
        cache_trigger_node_local_workload_norms: np.ndarray | None = None
        cache_trigger_candidate_summary_norms: np.ndarray | None = None
        cache_trigger_aux_norms: np.ndarray | None = None
        cache_trigger_weighted_loaded_mb: np.ndarray | None = None
        cache_trigger_weighted_cache_cost: np.ndarray | None = None
        cache_trigger_weighted_predicted_demand_gain: np.ndarray | None = None
        pre_trigger_candidate_logit_max_diff = 0.0
        cache_operation_actions: np.ndarray | None = None
        cache_operation_masks: np.ndarray | None = None
        cache_operation_log_probs: np.ndarray | None = None
        cache_operation_entropies: np.ndarray | None = None
        uav_operation_actions: np.ndarray | None = None
        uav_operation_masks: np.ndarray | None = None
        uav_operation_log_probs: np.ndarray | None = None
        uav_operation_entropies: np.ndarray | None = None
        uav_operation_probabilities: np.ndarray | None = None
        uav_operation_logit_margins: np.ndarray | None = None
        total_log_prob = sample.log_prob
        configuration_override = None
        if context.configuration_control_mode == "adaptive_joint_cache_uav_reconfiguration":
            if joint_primitive is None:
                raise RuntimeError("proposed joint primitive selection was not produced")
            cache_triggers, uav_triggers = context.action_spec.action_vector_to_triggers(
                sample.action
            )
            total_log_prob = joint_primitive.total_log_prob
            cache_contexts = joint_primitive.cache_contexts
            cache_candidate_features = joint_primitive.cache_candidate_features
            cache_candidate_masks = joint_primitive.cache_candidate_masks
            cache_selected_indices = joint_primitive.cache_selected_indices
            cache_old_log_probs = joint_primitive.cache_old_log_probs
            cache_conditional_actions = joint_primitive.cache_conditional_actions
            cache_candidate_counts = joint_primitive.cache_candidate_counts
            cache_trigger_node_local_workload = (
                joint_primitive.cache_trigger_node_local_workload
            )
            cache_trigger_candidate_summary = (
                joint_primitive.cache_trigger_candidate_summary
            )
            cache_trigger_aux = joint_primitive.cache_trigger_aux
            cache_trigger_legal_candidate_counts = (
                joint_primitive.cache_trigger_legal_candidate_counts
            )
            cache_trigger_node_local_workload_norms = (
                joint_primitive.cache_trigger_node_local_workload_norms
            )
            cache_trigger_candidate_summary_norms = (
                joint_primitive.cache_trigger_candidate_summary_norms
            )
            cache_trigger_aux_norms = joint_primitive.cache_trigger_aux_norms
            cache_trigger_weighted_loaded_mb = (
                joint_primitive.cache_trigger_weighted_loaded_mb
            )
            cache_trigger_weighted_cache_cost = (
                joint_primitive.cache_trigger_weighted_cache_cost
            )
            cache_trigger_weighted_predicted_demand_gain = (
                joint_primitive.cache_trigger_weighted_predicted_demand_gain
            )
            pre_trigger_candidate_logit_max_diff = (
                joint_primitive.pre_trigger_and_execution_candidate_logit_max_diff
            )
            uav_contexts = joint_primitive.uav_contexts
            uav_candidate_features = joint_primitive.uav_candidate_features
            uav_candidate_masks = joint_primitive.uav_candidate_masks
            uav_selected_indices = joint_primitive.uav_selected_indices
            uav_old_log_probs = joint_primitive.uav_old_log_probs
            uav_conditional_actions = joint_primitive.uav_conditional_actions
            uav_candidate_counts = joint_primitive.uav_candidate_counts
            cache_operation_actions = joint_primitive.cache_operation_actions
            cache_operation_masks = joint_primitive.cache_operation_masks
            cache_operation_log_probs = joint_primitive.cache_operation_old_log_probs
            cache_operation_entropies = joint_primitive.cache_operation_entropies
            uav_operation_actions = joint_primitive.uav_operation_actions
            uav_operation_masks = joint_primitive.uav_operation_masks
            uav_operation_log_probs = joint_primitive.uav_operation_old_log_probs
            uav_operation_entropies = joint_primitive.uav_operation_entropies
            uav_operation_probabilities = joint_primitive.uav_operation_probabilities
            uav_operation_logit_margins = joint_primitive.uav_operation_logit_margins
            configuration_override = slot_environment.build_policy_target_configuration(
                previous_cache,
                previous_uav,
                joint_primitive.final_cache_state,
                joint_primitive.final_uav_positions,
                cache_triggers,
                uav_triggers,
                context.service_catalog,
                context.cfg,
                context.cache_nodes,
                context.cache_cfg,
                context.flight_boundary,
                cache_actions=cache_conditional_actions,
                uav_actions=uav_conditional_actions,
            )
        elif context.configuration_control_mode == "policy_primitive_cache_operation":
            if cache_primitive is None:
                raise RuntimeError("Cache-only primitive selection was not produced")
            cache_triggers, uav_triggers = context.action_spec.action_vector_to_triggers(
                sample.action
            )
            total_log_prob = cache_primitive.total_log_prob
            cache_contexts = cache_primitive.cache_contexts
            cache_candidate_features = cache_primitive.cache_candidate_features
            cache_candidate_masks = cache_primitive.cache_candidate_masks
            cache_selected_indices = cache_primitive.cache_selected_indices
            cache_old_log_probs = cache_primitive.cache_old_log_probs
            cache_conditional_actions = cache_primitive.cache_conditional_actions
            cache_candidate_counts = cache_primitive.cache_candidate_counts
            cache_operation_actions = cache_primitive.operation_actions
            cache_operation_masks = cache_primitive.operation_masks
            cache_operation_log_probs = cache_primitive.operation_old_log_probs
            cache_operation_entropies = cache_primitive.operation_entropies
            configuration_override = slot_environment.build_policy_target_configuration(
                previous_cache,
                previous_uav,
                cache_primitive.final_cache_state,
                cache_primitive.final_uav_positions,
                cache_triggers,
                uav_triggers,
                context.service_catalog,
                context.cfg,
                context.cache_nodes,
                context.cache_cfg,
                context.flight_boundary,
                cache_actions=cache_conditional_actions,
                uav_actions=(),
            )
        elif context.configuration_control_mode == "policy_primitive":
            if not isinstance(policy, AdaptivePrimitiveActorCritic):
                raise TypeError("policy_primitive mode requires AdaptivePrimitiveActorCritic。")
            primitive = sample_and_apply_primitive_configuration(
                policy,
                sample,
                previous_cache,
                previous_uav,
                cache_triggers,
                uav_triggers,
                step,
                context,
                tensor_spec,
                deterministic=bool(deterministic),
                conditional_action_mode=conditional_action_mode,
            )
            total_log_prob = primitive.total_log_prob
            cache_contexts = primitive.cache_contexts
            cache_candidate_features = primitive.cache_candidate_features
            cache_candidate_masks = primitive.cache_candidate_masks
            cache_selected_indices = primitive.cache_selected_indices
            cache_old_log_probs = primitive.cache_old_log_probs
            uav_contexts = primitive.uav_contexts
            uav_candidate_features = primitive.uav_candidate_features
            uav_candidate_masks = primitive.uav_candidate_masks
            uav_selected_indices = primitive.uav_selected_indices
            uav_old_log_probs = primitive.uav_old_log_probs
            cache_conditional_actions = primitive.cache_conditional_actions
            uav_conditional_actions = primitive.uav_conditional_actions
            cache_candidate_counts = primitive.cache_candidate_counts
            uav_candidate_counts = primitive.uav_candidate_counts
            configuration_override = slot_environment.build_policy_target_configuration(
                previous_cache,
                previous_uav,
                primitive.final_cache_state,
                primitive.final_uav_positions,
                cache_triggers,
                uav_triggers,
                context.service_catalog,
                context.cfg,
                context.cache_nodes,
                context.cache_cfg,
                context.flight_boundary,
                cache_actions=cache_conditional_actions,
                uav_actions=uav_conditional_actions,
            )
        elif context.configuration_control_mode != "legacy_exact":
            raise ValueError(
                f"unknown configuration_control_mode={context.configuration_control_mode}"
            )
        policy_selection_seconds = time.perf_counter() - selection_started_at
        performance_profile: dict[str, Any] = {}
        configuration_profile = cache_configuration.ConfigurationEvaluationProfile()
        result = slot_environment.execute_slot(
            previous_cache,
            previous_uav,
            cache_triggers,
            uav_triggers,
            step.scenario_slice,
            step.actual_tasks,
            step.actual_vehicle_positions,
            context.local_cache,
            context.service_catalog,
            context.cfg,
            context.cache_nodes,
            context.cache_cfg,
            context.flight_boundary,
            slot_id=step.slot,
            time_s=step.time_s,
            performance_profile=performance_profile,
            configuration_profile=configuration_profile,
            configuration_override=configuration_override,
        )
        if not result.step_feasible:
            raise RuntimeError(
                "PPO v2 request produced an infeasible environment transition: "
                f"origin={step.origin_index}, slot={step.slot}, reason={result.infeasible_reason}"
            )
        _check_request_contract(previous_cache, previous_uav, result, context.action_spec)
        reward = _check_reward(result)
        task_metrics = task_metrics_from_result(result, context.cache_nodes)
        done = int(step.origin_index) == terminal_origin_index
        if done and step_index != len(execution_steps) - 1:
            raise ValueError("true episode terminal 必须是 rollout 的最后一步。")
        next_cache = np.asarray(result.next_cache_state, dtype=np.int8).copy()
        next_uav = _copy_uav_positions(result.next_uav_positions)
        actual_cache_update_count = int(
            np.any(next_cache != previous_cache, axis=1).sum()
        )
        actual_uav_movement_count = sum(
            int(
                np.linalg.norm(
                    np.asarray(next_uav[uav_id], dtype=np.float64)
                    - np.asarray(previous_uav[uav_id], dtype=np.float64)
                )
                > 1.0e-6
            )
            for uav_id in context.action_spec.uav_ids
        )
        transitions.append(
            SimpleTransition(
                observation=np.asarray(observation, dtype=np.float32).copy(),
                action=np.asarray(sample.action, dtype=np.int8).copy(),
                action_mask=np.asarray(sample.action_mask, dtype=bool).copy(),
                old_log_prob=float(total_log_prob.detach().cpu()),
                old_value=float(sample.value.detach().cpu()),
                reward=reward,
                done=done,
                slot=int(step.slot),
                origin_index=int(step.origin_index),
                actual_task_utility=float(result.actual_task_utility),
                cache_cost=float(result.cache_cost),
                movement_cost=float(result.movement_cost),
                task_count=int(task_metrics.task_count),
                total_task_delay_s=float(task_metrics.total_task_delay_s),
                total_vehicle_energy_j=float(task_metrics.total_vehicle_energy_j),
                execution_target_counts=dict(task_metrics.execution_target_counts),
                actual_cache_update_count=actual_cache_update_count,
                actual_uav_movement_count=actual_uav_movement_count,
                cache_contexts=cache_contexts.copy(),
                cache_candidate_features=cache_candidate_features.copy(),
                cache_candidate_masks=cache_candidate_masks.copy(),
                cache_selected_indices=cache_selected_indices.copy(),
                cache_old_log_probs=cache_old_log_probs.copy(),
                uav_contexts=uav_contexts.copy(),
                uav_candidate_features=uav_candidate_features.copy(),
                uav_candidate_masks=uav_candidate_masks.copy(),
                uav_selected_indices=uav_selected_indices.copy(),
                uav_old_log_probs=uav_old_log_probs.copy(),
                cache_conditional_actions=cache_conditional_actions,
                uav_conditional_actions=uav_conditional_actions,
                cache_candidate_counts=cache_candidate_counts.copy(),
                uav_candidate_counts=uav_candidate_counts.copy(),
                policy_selection_seconds=float(policy_selection_seconds),
                configuration_seconds=float(performance_profile.get("config_seconds", 0.0)),
                actual_matching_seconds=float(
                    performance_profile.get("matching_kkt_seconds", 0.0)
                ),
                online_k100_configuration_calls=int(
                    performance_profile.get("online_k100_configuration_calls", 0)
                ),
                matching_summary=dict(result.matching_summary),
                origin_runtime_seconds=float(
                    time.perf_counter() - origin_started_at
                ),
                previous_cache_state=previous_cache.copy(),
                previous_uav_positions=dict(previous_uav),
                next_cache_state=next_cache.copy(),
                next_uav_positions=dict(next_uav),
                cache_operation_actions=(
                    None
                    if cache_operation_actions is None
                    else cache_operation_actions.copy()
                ),
                cache_operation_masks=(
                    None
                    if cache_operation_masks is None
                    else cache_operation_masks.copy()
                ),
                cache_operation_log_probs=(
                    None
                    if cache_operation_log_probs is None
                    else cache_operation_log_probs.copy()
                ),
                cache_operation_entropies=(
                    None
                    if cache_operation_entropies is None
                    else cache_operation_entropies.copy()
                ),
                uav_operation_actions=(
                    None
                    if uav_operation_actions is None
                    else uav_operation_actions.copy()
                ),
                uav_operation_masks=(
                    None
                    if uav_operation_masks is None
                    else uav_operation_masks.copy()
                ),
                uav_operation_log_probs=(
                    None
                    if uav_operation_log_probs is None
                    else uav_operation_log_probs.copy()
                ),
                uav_operation_entropies=(
                    None
                    if uav_operation_entropies is None
                    else uav_operation_entropies.copy()
                ),
                uav_operation_probabilities=(
                    None
                    if uav_operation_probabilities is None
                    else uav_operation_probabilities.copy()
                ),
                uav_operation_logit_margins=(
                    None
                    if uav_operation_logit_margins is None
                    else uav_operation_logit_margins.copy()
                ),
                cache_trigger_node_local_workload=(
                    None
                    if cache_trigger_node_local_workload is None
                    else _snapshot_trigger_feature(cache_trigger_node_local_workload)
                ),
                cache_trigger_candidate_summary=(
                    None
                    if cache_trigger_candidate_summary is None
                    else _snapshot_trigger_feature(cache_trigger_candidate_summary)
                ),
                cache_trigger_aux=(
                    None
                    if cache_trigger_aux is None
                    else _snapshot_trigger_feature(cache_trigger_aux)
                ),
                cache_trigger_legal_candidate_counts=(
                    None
                    if cache_trigger_legal_candidate_counts is None
                    else cache_trigger_legal_candidate_counts.copy()
                ),
                cache_trigger_node_local_workload_norms=(
                    None
                    if cache_trigger_node_local_workload_norms is None
                    else cache_trigger_node_local_workload_norms.copy()
                ),
                cache_trigger_candidate_summary_norms=(
                    None
                    if cache_trigger_candidate_summary_norms is None
                    else cache_trigger_candidate_summary_norms.copy()
                ),
                cache_trigger_aux_norms=(
                    None
                    if cache_trigger_aux_norms is None
                    else cache_trigger_aux_norms.copy()
                ),
                cache_trigger_weighted_loaded_mb=(
                    None
                    if cache_trigger_weighted_loaded_mb is None
                    else cache_trigger_weighted_loaded_mb.copy()
                ),
                cache_trigger_weighted_cache_cost=(
                    None
                    if cache_trigger_weighted_cache_cost is None
                    else cache_trigger_weighted_cache_cost.copy()
                ),
                cache_trigger_weighted_predicted_demand_gain=(
                    None
                    if cache_trigger_weighted_predicted_demand_gain is None
                    else cache_trigger_weighted_predicted_demand_gain.copy()
                ),
                pre_trigger_and_execution_candidate_logit_max_diff=(
                    float(pre_trigger_candidate_logit_max_diff)
                ),
            )
        )
        previous_cache = next_cache
        previous_uav = next_uav

    bootstrap_observation: np.ndarray | None = None
    if not transitions[-1].done:
        if bootstrap_step is None:
            raise ValueError("非 terminal rollout chunk 缺少 bootstrap step。")
        if int(bootstrap_step.origin_index) != int(execution_steps[-1].origin_index) + 1:
            raise ValueError("bootstrap step 必须紧跟 rollout chunk 的最后 origin。")
        time_to_go_override = (
            None
            if observation_time_to_go is None
            else float(observation_time_to_go(int(bootstrap_step.origin_index)))
        )
        bootstrap_observation = _observation(
            previous_cache,
            previous_uav,
            bootstrap_step,
            context.observation_spec,
            context.flight_boundary,
            context.train_origin_start,
            context.train_origin_end,
            time_to_go_override=time_to_go_override,
            no_prediction=is_no_prediction_context(context),
        )

    def stack_optional_transition_field(name: str, dtype: Any) -> np.ndarray | None:
        values = [getattr(item, name) for item in transitions]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise RuntimeError(f"rollout field {name} is only partially populated")
        return np.stack(values).astype(dtype)

    stored_trigger_workload = stack_optional_transition_field(
        "cache_trigger_node_local_workload", np.float32
    )
    stored_candidate_summary = stack_optional_transition_field(
        "cache_trigger_candidate_summary", np.float32
    )
    stored_trigger_aux = stack_optional_transition_field(
        "cache_trigger_aux", np.float32
    )
    if stored_trigger_aux is not None:
        if stored_trigger_workload is None or stored_candidate_summary is None:
            raise RuntimeError("V3 rollout Trigger auxiliary components are incomplete")
        expected_trigger_aux = np.concatenate(
            [
                stored_trigger_workload.reshape(
                    len(transitions), stored_trigger_workload.shape[1], -1
                ),
                stored_candidate_summary,
            ],
            axis=-1,
        ).astype(np.float32, copy=False)
        if not np.array_equal(stored_trigger_aux, expected_trigger_aux):
            raise RuntimeError("stored V3 Trigger auxiliary differs from its frozen parts")

    return SimpleRollout(
        transitions=tuple(transitions),
        observations=np.stack([item.observation for item in transitions]).astype(np.float32),
        actions=np.stack([item.action for item in transitions]).astype(np.int8),
        action_masks=np.stack([item.action_mask for item in transitions]).astype(bool),
        old_log_probs=np.asarray([item.old_log_prob for item in transitions], dtype=np.float32),
        old_values=np.asarray([item.old_value for item in transitions], dtype=np.float32),
        rewards=np.asarray([item.reward for item in transitions], dtype=np.float32),
        dones=np.asarray([item.done for item in transitions], dtype=bool),
        cache_contexts=np.stack([item.cache_contexts for item in transitions]).astype(np.float32),
        cache_candidate_features=np.stack(
            [item.cache_candidate_features for item in transitions]
        ).astype(np.float32),
        cache_candidate_masks=np.stack(
            [item.cache_candidate_masks for item in transitions]
        ).astype(bool),
        cache_selected_indices=np.stack(
            [item.cache_selected_indices for item in transitions]
        ).astype(np.int64),
        uav_contexts=np.stack([item.uav_contexts for item in transitions]).astype(np.float32),
        uav_candidate_features=np.stack(
            [item.uav_candidate_features for item in transitions]
        ).astype(np.float32),
        uav_candidate_masks=np.stack(
            [item.uav_candidate_masks for item in transitions]
        ).astype(bool),
        uav_selected_indices=np.stack(
            [item.uav_selected_indices for item in transitions]
        ).astype(np.int64),
        next_cache_state=previous_cache.copy(),
        next_uav_positions=dict(previous_uav),
        bootstrap_observation=bootstrap_observation,
        cache_operation_actions=(
            None
            if any(item.cache_operation_actions is None for item in transitions)
            else np.stack(
                [item.cache_operation_actions for item in transitions]
            ).astype(np.int64)
        ),
        cache_operation_masks=(
            None
            if any(item.cache_operation_masks is None for item in transitions)
            else np.stack(
                [item.cache_operation_masks for item in transitions]
            ).astype(bool)
        ),
        cache_operation_log_probs=(
            None
            if any(item.cache_operation_log_probs is None for item in transitions)
            else np.stack(
                [item.cache_operation_log_probs for item in transitions]
            ).astype(np.float32)
        ),
        cache_operation_entropies=(
            None
            if any(item.cache_operation_entropies is None for item in transitions)
            else np.stack(
                [item.cache_operation_entropies for item in transitions]
            ).astype(np.float32)
        ),
        uav_operation_actions=(
            None
            if any(item.uav_operation_actions is None for item in transitions)
            else np.stack(
                [item.uav_operation_actions for item in transitions]
            ).astype(np.int64)
        ),
        uav_operation_masks=(
            None
            if any(item.uav_operation_masks is None for item in transitions)
            else np.stack(
                [item.uav_operation_masks for item in transitions]
            ).astype(bool)
        ),
        uav_operation_log_probs=(
            None
            if any(item.uav_operation_log_probs is None for item in transitions)
            else np.stack(
                [item.uav_operation_log_probs for item in transitions]
            ).astype(np.float32)
        ),
        uav_operation_entropies=(
            None
            if any(item.uav_operation_entropies is None for item in transitions)
            else np.stack(
                [item.uav_operation_entropies for item in transitions]
            ).astype(np.float32)
        ),
        cache_old_log_probs=np.stack(
            [item.cache_old_log_probs for item in transitions]
        ).astype(np.float32),
        uav_old_log_probs=np.stack(
            [item.uav_old_log_probs for item in transitions]
        ).astype(np.float32),
        cache_trigger_node_local_workload=stored_trigger_workload,
        cache_trigger_candidate_summary=stored_candidate_summary,
        cache_trigger_aux=stored_trigger_aux,
        cache_trigger_legal_candidate_counts=stack_optional_transition_field(
            "cache_trigger_legal_candidate_counts", np.int64
        ),
        cache_trigger_node_local_workload_norms=stack_optional_transition_field(
            "cache_trigger_node_local_workload_norms", np.float32
        ),
        cache_trigger_candidate_summary_norms=stack_optional_transition_field(
            "cache_trigger_candidate_summary_norms", np.float32
        ),
        cache_trigger_aux_norms=stack_optional_transition_field(
            "cache_trigger_aux_norms", np.float32
        ),
        cache_trigger_weighted_loaded_mb=stack_optional_transition_field(
            "cache_trigger_weighted_loaded_mb", np.float32
        ),
        cache_trigger_weighted_cache_cost=stack_optional_transition_field(
            "cache_trigger_weighted_cache_cost", np.float32
        ),
        cache_trigger_weighted_predicted_demand_gain=stack_optional_transition_field(
            "cache_trigger_weighted_predicted_demand_gain", np.float32
        ),
        pre_trigger_and_execution_candidate_logit_max_diff=max(
            float(item.pre_trigger_and_execution_candidate_logit_max_diff)
            for item in transitions
        ),
    )


def compute_gae(
    rewards: Any,
    values: Any,
    dones: Any,
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute standard GAE and returns without any environment-specific terms."""

    rewards_a = np.asarray(rewards, dtype=np.float32).reshape(-1)
    values_a = np.asarray(values, dtype=np.float32).reshape(-1)
    dones_a = np.asarray(dones, dtype=bool).reshape(-1)
    if not (len(rewards_a) == len(values_a) == len(dones_a)) or not len(rewards_a):
        raise ValueError("GAE input 长度必须一致且非空。")
    if not np.all(np.isfinite(rewards_a)) or not np.all(np.isfinite(values_a)):
        raise ValueError("GAE input 包含 NaN/Inf。")
    advantages = np.zeros_like(rewards_a, dtype=np.float32)
    next_advantage = 0.0
    for index in range(len(rewards_a) - 1, -1, -1):
        next_value = float(bootstrap_value) if index == len(rewards_a) - 1 else float(values_a[index + 1])
        nonterminal = 0.0 if bool(dones_a[index]) else 1.0
        delta = float(rewards_a[index]) + float(gamma) * nonterminal * next_value - float(values_a[index])
        next_advantage = delta + float(gamma) * float(gae_lambda) * nonterminal * next_advantage
        advantages[index] = next_advantage
    returns = advantages + values_a
    if not np.all(np.isfinite(advantages)) or not np.all(np.isfinite(returns)):
        raise ValueError("GAE result 包含 NaN/Inf。")
    return advantages.astype(np.float32), returns.astype(np.float32)
