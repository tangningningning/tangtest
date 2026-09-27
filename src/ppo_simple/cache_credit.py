"""Cache node-specific counterfactual credit with safe hybrid screening.

The helper deliberately owns no optimizer, reward, Matching, KKT, or system
model logic.  It replays the production target-configuration transition with
one Cache node replaced and uses the existing global critic only for the
next-state bootstrap.  The proposed binary trigger view is KEEP versus UPDATE;
UPDATE branches replace the node with one complete, feasible target cache
configuration rather than an ADD/DELETE/SWAP primitive.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from decision import cache_configuration, slot_environment, uav_configuration

from .primitive_control import (
    PrimitiveTensorSpec,
    _cache_candidate_feature,
    _cache_target_configuration_feature,
    _intermediate_observation,
    sample_and_apply_joint_cache_uav_operation_configuration,
)
from .policy import CandidatePolicySample
from .rollout import _check_request_contract, _check_reward


OPERATION_ORDER = ("NO_CHANGE", "ADD", "DELETE", "SWAP")


def _position_dict(values: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        str(key): np.asarray(value, dtype=np.float64).copy()
        for key, value in values.items()
    }


def _production_action_map(transition: Any) -> dict[int, dict[str, Any]]:
    actions = tuple(dict(value) for value in transition.cache_conditional_actions)
    mapping = {int(item["head_index"]): item for item in actions}
    expected = set(range(len(transition.previous_cache_state)))
    if set(mapping) != expected:
        raise RuntimeError(
            "proposed production Cache action record is not one entry per node: "
            f"got={sorted(mapping)}, expected={sorted(expected)}"
        )
    return mapping


def _replay_cache_prefix(transition: Any, target_index: int) -> tuple[np.ndarray, np.ndarray]:
    previous_cache = np.asarray(transition.previous_cache_state, dtype=np.int8).copy()
    production_final = np.asarray(transition.next_cache_state, dtype=np.int8)
    working = previous_cache.copy()
    action_map = _production_action_map(transition)
    for index in range(int(target_index)):
        working[index] = production_final[index]
        is_keep = str(action_map[index].get("conditional_action", "")).upper() == "KEEP" or (
            str(action_map[index].get("operation_type", "")).upper() == "NO_CHANGE"
        )
        if is_keep and not np.array_equal(
            working[index], previous_cache[index]
        ):
            raise RuntimeError("production KEEP changed a Cache row")
    return working, previous_cache


def _stored_node_cache_trigger_aux(
    policy: Any,
    transition: Any,
    node_index: int,
    node_count: int,
) -> np.ndarray | None:
    """Return the V3 auxiliary row frozen on this rollout transition."""

    if not bool(getattr(policy, "true_binary_trigger", False)):
        return None
    auxiliary_dim = getattr(policy, "trigger_aux_dim", None)
    if auxiliary_dim is None:
        raise RuntimeError("V3 Cache Trigger policy does not expose trigger_aux_dim")
    stored_auxiliary = getattr(transition, "cache_trigger_aux", None)
    if stored_auxiliary is None:
        raise RuntimeError(
            "V3 Cache Trigger credit requires stored rollout cache_trigger_aux"
        )
    auxiliary = np.asarray(stored_auxiliary, dtype=np.float32)
    expected_shape = (int(node_count), int(auxiliary_dim))
    if tuple(auxiliary.shape) != expected_shape:
        raise RuntimeError(
            "stored V3 Cache Trigger auxiliary has shape "
            f"{tuple(auxiliary.shape)}, expected {expected_shape}"
        )
    if not np.all(np.isfinite(auxiliary)):
        raise FloatingPointError("stored V3 Cache Trigger auxiliary is non-finite")
    return auxiliary[int(node_index)].copy()


def _masked_operation_probabilities(
    policy: Any,
    conditional_context: np.ndarray,
    node_index: int,
    operation_mask: np.ndarray,
    trigger_aux: np.ndarray | None = None,
) -> np.ndarray:
    if bool(getattr(policy, "true_binary_trigger", False)):
        if trigger_aux is None:
            raise RuntimeError(
                "V3 Cache Trigger operation logits require stored trigger_aux"
            )
        logits, single = policy._operation_logits(conditional_context, trigger_aux)
    else:
        # Legacy non-binary policies retain their one-argument operation API.
        logits, single = policy._operation_logits(conditional_context)
    if not single:
        raise RuntimeError("Cache operation distribution requires one context")
    raw_logits = logits[int(node_index)]
    mask = torch.as_tensor(operation_mask, dtype=torch.bool, device=raw_logits.device)
    distribution = torch.distributions.Categorical(
        logits=raw_logits.masked_fill(~mask, -float("inf"))
    )
    probabilities = distribution.probs.detach().cpu().numpy().astype(np.float64)
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("Cache operation probabilities are invalid")
    probabilities /= total
    return probabilities


def _candidate_probabilities(
    policy: Any,
    conditional_context: np.ndarray,
    candidate_features: np.ndarray,
) -> np.ndarray:
    scores, single = policy._candidate_scores(
        conditional_context,
        candidate_features,
        policy.cache_feature_dim,
        policy.cache_candidate_scorer,
    )
    if not single:
        raise RuntimeError("Cache candidate distribution requires one context")
    probabilities = torch.softmax(scores.squeeze(0), dim=-1)
    return probabilities.detach().cpu().numpy().astype(np.float64)


def _node_policy_state(
    trainer: Any,
    transition: Any,
    step: Any,
    node_index: int,
    feature_dim: int,
) -> dict[str, Any]:
    context = trainer.context
    trigger_aux = _stored_node_cache_trigger_aux(
        trainer.policy,
        transition,
        node_index,
        len(context.action_spec.cache_node_ids),
    )
    working_cache, previous_cache = _replay_cache_prefix(transition, node_index)
    current_uav = _position_dict(transition.previous_uav_positions)
    proposal_cache_nodes = uav_configuration.cache_nodes_with_uav_positions(
        context.cache_nodes,
        current_uav,
    )
    conditional_context = _intermediate_observation(
        working_cache,
        current_uav,
        step,
        context,
    )
    if bool(getattr(trainer.policy, "true_binary_trigger", False)):
        # The proposed mainline enumerates complete target configurations.  No
        # primitive operation label reaches the policy, replay, or candidate
        # tensor; proposal_source is diagnostics-only provenance.
        update_candidates = cache_configuration.enumerate_cache_target_configurations(
            working_cache,
            node_index,
            context.service_catalog,
            proposal_cache_nodes,
            step.scenario_slice,
            context.local_cache,
            context.cache_cfg,
        )
        action_map = _production_action_map(transition)
        sequential_triggers = np.zeros(
            len(context.action_spec.cache_node_ids), dtype=np.int8
        )
        for index in range(node_index):
            if str(action_map[index].get("conditional_action", "")).upper() != "KEEP":
                sequential_triggers[index] = 1
        sequential_triggers[node_index] = 1
        if update_candidates:
            node_demand = cache_configuration.node_specific_predicted_demand(
                step.scenario_slice,
                node_index,
                proposal_cache_nodes,
                context.cache_cfg,
            )
            observed_usage = cache_configuration.observed_local_service_usage(
                context.local_cache,
                int(context.service_catalog["num_services"]),
            )
            unified_features = np.stack(
                [
                    _cache_target_configuration_feature(
                        candidate,
                        working_cache,
                        previous_cache,
                        sequential_triggers,
                        node_demand,
                        observed_usage,
                        context,
                    )
                    for candidate in update_candidates
                ]
            ).astype(np.float32)
            candidate_probabilities = _candidate_probabilities(
                trainer.policy,
                np.asarray(conditional_context, dtype=np.float32),
                unified_features,
            )
            candidate_features = [
                np.asarray(feature, dtype=np.float32)
                for feature in unified_features
            ]
            candidate_operations = np.asarray(
                np.zeros(len(update_candidates), dtype=np.int64),
                dtype=np.int64,
            )
        else:
            candidate_probabilities = np.zeros(0, dtype=np.float64)
            candidate_features = []
            candidate_operations = np.zeros(0, dtype=np.int64)
        operation_mask = np.asarray([True, bool(update_candidates)], dtype=bool)
        operation_probabilities = _masked_operation_probabilities(
            trainer.policy,
            np.asarray(conditional_context, dtype=np.float32),
            node_index,
            operation_mask,
            trigger_aux=trigger_aux,
        )
        p_keep = float(operation_probabilities[0])
        p_update = float(operation_probabilities[1])
        if candidate_probabilities.size:
            candidate_probabilities = candidate_probabilities / float(
                candidate_probabilities.sum()
            )
        return {
            "working_cache": working_cache,
            "previous_cache": previous_cache,
            "conditional_context": np.asarray(conditional_context, dtype=np.float32),
            "operation_mask": operation_mask,
            "operation_probabilities": operation_probabilities,
            "p_keep": p_keep,
            "p_update": p_update,
            "candidates": update_candidates,
            "candidate_features": candidate_features,
            "candidate_operations": candidate_operations,
            "candidate_probabilities": np.asarray(candidate_probabilities, dtype=np.float64),
            "grouped_candidates": {
                name: [] for name in OPERATION_ORDER[1:]
            },
            "grouped_features": {
                name: np.zeros((0, int(feature_dim)), dtype=np.float32)
                for name in OPERATION_ORDER[1:]
            },
        }
    all_candidates = cache_configuration.enumerate_cache_primitive_actions(
        working_cache,
        node_index,
        context.service_catalog,
        context.cache_nodes,
    )
    operation_mask = np.zeros(len(OPERATION_ORDER), dtype=bool)
    operation_mask[0] = True
    grouped_candidates: dict[str, list[Any]] = {
        name: [] for name in OPERATION_ORDER[1:]
    }
    for candidate in all_candidates:
        operation_index = OPERATION_ORDER.index(str(candidate.operation_type))
        operation_mask[operation_index] = True
        if operation_index != 0:
            grouped_candidates[str(candidate.operation_type)].append(candidate)

    action_map = _production_action_map(transition)
    sequential_triggers = np.zeros(len(context.action_spec.cache_node_ids), dtype=np.int8)
    for index in range(node_index):
        if str(action_map[index].get("operation_type")) != "NO_CHANGE":
            sequential_triggers[index] = 1
    # Candidate features are evaluated in the UPDATE conditional state.
    sequential_triggers[node_index] = 1

    grouped_features: dict[str, np.ndarray] = {}
    for operation_name in OPERATION_ORDER[1:]:
        candidates = grouped_candidates[operation_name]
        if not candidates:
            grouped_features[operation_name] = np.zeros(
                (0, int(feature_dim)), dtype=np.float32
            )
            continue
        grouped_features[operation_name] = np.stack(
            [
                _cache_candidate_feature(
                    candidate,
                    working_cache,
                    previous_cache,
                    sequential_triggers,
                    step.predicted_service_popularity,
                    context,
                    include_replica_awareness=True,
                )
                for candidate in candidates
            ]
        ).astype(np.float32)

    operation_probabilities = _masked_operation_probabilities(
        trainer.policy,
        np.asarray(conditional_context, dtype=np.float32),
        node_index,
        operation_mask,
        trigger_aux=trigger_aux,
    )
    p_keep = float(operation_probabilities[0])
    p_update = float(operation_probabilities[1:].sum())
    candidates: list[Any] = []
    candidate_features: list[np.ndarray] = []
    candidate_operations: list[int] = []
    candidate_probabilities: list[float] = []
    for operation_name in OPERATION_ORDER[1:]:
        group_candidates = grouped_candidates[operation_name]
        if not group_candidates:
            continue
        group_index = OPERATION_ORDER.index(operation_name)
        conditional_probabilities = _candidate_probabilities(
            trainer.policy,
            np.asarray(conditional_context, dtype=np.float32),
            grouped_features[operation_name],
        )
        group_probability = float(operation_probabilities[group_index])
        for candidate, feature, probability in zip(
            group_candidates,
            grouped_features[operation_name],
            conditional_probabilities,
            strict=True,
        ):
            candidates.append(candidate)
            candidate_features.append(np.asarray(feature, dtype=np.float32))
            candidate_operations.append(int(group_index))
            candidate_probabilities.append(
                (group_probability / p_update) * float(probability)
                if p_update > 0.0
                else 0.0
            )
    candidate_probabilities_array = np.asarray(candidate_probabilities, dtype=np.float64)
    if candidate_probabilities_array.size:
        if p_update <= 0.0:
            raise RuntimeError("UPDATE probability is zero despite legal candidates")
        candidate_probabilities_array /= float(candidate_probabilities_array.sum())
    return {
        "working_cache": working_cache,
        "previous_cache": previous_cache,
        "conditional_context": np.asarray(conditional_context, dtype=np.float32),
        "operation_mask": operation_mask,
        "operation_probabilities": operation_probabilities,
        "p_keep": p_keep,
        "p_update": p_update,
        "candidates": candidates,
        "candidate_features": candidate_features,
        "candidate_operations": np.asarray(candidate_operations, dtype=np.int64),
        "candidate_probabilities": candidate_probabilities_array,
        "grouped_candidates": grouped_candidates,
        "grouped_features": grouped_features,
    }


def _execute_cache_branch(
    trainer: Any,
    transition: Any,
    step: Any,
    next_step: Any | None,
    node_index: int,
    state: dict[str, Any],
    candidate: Any | None,
) -> dict[str, Any]:
    context = trainer.context
    previous_cache = np.asarray(transition.previous_cache_state, dtype=np.int8).copy()
    previous_uav = _position_dict(transition.previous_uav_positions)
    final_cache = np.asarray(transition.next_cache_state, dtype=np.int8).copy()
    final_uav = _position_dict(transition.next_uav_positions)
    production_triggers, production_uav_triggers = context.action_spec.action_vector_to_triggers(
        transition.action
    )
    branch_triggers = np.asarray(production_triggers, dtype=np.int8).copy()
    production_actions = _production_action_map(transition)
    branch_actions: list[dict[str, Any]] = []
    for index, node_id in enumerate(context.action_spec.cache_node_ids):
        action = dict(production_actions[index])
        if index == node_index:
            if candidate is None:
                branch_triggers[index] = 0
                final_cache[index] = state["working_cache"][index]
                action = {
                    "head_type": "cache",
                    "head_id": str(node_id),
                    "head_index": int(index),
                    "candidate_count": 0,
                    "selected_index": None,
                    "conditional_action": "KEEP",
                    "actual_changed": False,
                }
            else:
                branch_triggers[index] = 1
                final_cache[index] = np.asarray(
                    candidate.target_node_cache, dtype=np.int8
                ).copy()
                action = {
                    "head_type": "cache",
                    "head_id": str(node_id),
                    "head_index": int(index),
                    "candidate_count": -1,
                    "selected_index": None,
                    "conditional_action": "TARGET_CONFIGURATION",
                    "target_cache": np.asarray(
                        candidate.target_node_cache, dtype=np.int8
                    ).astype(int).tolist(),
                    "proposal_source": str(candidate.proposal_source),
                    "actual_changed": bool(
                        not np.array_equal(
                            candidate.before_node_cache,
                            candidate.target_node_cache,
                        )
                    ),
                }
        branch_actions.append(action)
    branch_actions.sort(key=lambda item: int(item["head_index"]))
    production_uav_triggers = {
        str(key): int(value) for key, value in production_uav_triggers.items()
    }
    configuration_override = slot_environment.build_policy_target_configuration(
        previous_cache,
        previous_uav,
        final_cache,
        final_uav,
        branch_triggers,
        production_uav_triggers,
        context.service_catalog,
        context.cfg,
        context.cache_nodes,
        context.cache_cfg,
        context.flight_boundary,
        cache_actions=tuple(branch_actions),
        uav_actions=tuple(dict(value) for value in transition.uav_conditional_actions),
    )
    result = slot_environment.execute_slot(
        previous_cache,
        previous_uav,
        branch_triggers,
        production_uav_triggers,
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
        performance_profile=None,
        configuration_profile=cache_configuration.ConfigurationEvaluationProfile(),
        configuration_override=configuration_override,
    )
    if not result.step_feasible:
        raise RuntimeError(
            "Exact Cache counterfactual is infeasible: "
            f"origin={step.origin_index}, node={node_index}"
        )
    _check_request_contract(previous_cache, previous_uav, result, context.action_spec)
    reward = _check_reward(result)
    next_observation = None
    if next_step is not None:
        next_observation = _intermediate_observation(
            np.asarray(result.next_cache_state, dtype=np.int8),
            result.next_uav_positions,
            next_step,
            context,
        )
    return {
        "reward": float(reward),
        "next_observation": next_observation,
        "result": result,
    }


class _DeterministicReplayOverrides:
    """Frozen policy proxy for deterministic downstream replay diagnostics."""

    def __init__(
        self,
        policy: Any,
        forced_triggers: dict[int, int],
        forced_candidates: dict[int, int],
    ) -> None:
        self._policy = policy
        self._forced_triggers = dict(forced_triggers)
        self._forced_candidates = dict(forced_candidates)
        self._active_cache_head: int | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    def sample_cache_trigger(
        self,
        observation: Any,
        node_index: int,
        operation_mask: Any,
        *,
        trigger_aux: Any,
        deterministic: bool = False,
    ) -> Any:
        node_index = int(node_index)
        self._active_cache_head = node_index
        sample = self._policy.sample_cache_trigger(
            observation,
            node_index,
            operation_mask,
            trigger_aux=trigger_aux,
            deterministic=True,
        )
        forced = self._forced_triggers.get(node_index)
        if forced is None:
            return sample
        return _override_operation_sample(sample, forced)

    def sample_cache_update_candidate_from_logits(
        self,
        candidate_logits: Any,
        deterministic: bool = False,
    ) -> CandidatePolicySample:
        sample = self._policy.sample_cache_update_candidate_from_logits(
            candidate_logits,
            deterministic=True,
        )
        if self._active_cache_head is None:
            raise RuntimeError("Cache candidate replay has no active Cache head")
        forced = self._forced_candidates.get(self._active_cache_head)
        if forced is None:
            return sample
        return _override_operation_sample(sample, forced)


def _override_operation_sample(sample: Any, selected_index: int) -> Any:
    selected_index = int(selected_index)
    logits = torch.as_tensor(sample.logits)
    distribution = torch.distributions.Categorical(logits=logits)
    if selected_index < 0 or selected_index >= int(logits.numel()):
        raise IndexError("forced deterministic replay action is out of range")
    if not bool(torch.isfinite(distribution.logits[selected_index]).item()):
        raise ValueError("forced deterministic replay action is masked")
    selected = torch.as_tensor(
        selected_index,
        dtype=torch.long,
        device=distribution.logits.device,
    )
    return replace(
        sample,
        selected_index=selected_index,
        log_prob=distribution.log_prob(selected),
    )


def _execute_policy_consistent_cache_branch(
    trainer: Any,
    transition: Any,
    step: Any,
    next_step: Any | None,
    node_index: int,
    state: dict[str, Any],
    candidate: Any | None,
    *,
    force_update_candidate_index: int | None,
) -> dict[str, Any]:
    """Replay later Cache and UAV choices after a forced target-head action."""

    actions = _production_action_map(transition)
    forced_triggers: dict[int, int] = {}
    forced_candidates: dict[int, int] = {}
    for index in range(int(node_index)):
        record = actions[index]
        action = str(record.get("conditional_action", "")).upper()
        if action == "KEEP":
            forced_triggers[index] = 0
            continue
        if action != "TARGET_CONFIGURATION":
            raise RuntimeError(
                "policy-consistent replay cannot pin an unknown Cache prefix action: "
                f"head={index}, action={action!r}"
            )
        selected_index = record.get("selected_index")
        if selected_index is None:
            raise RuntimeError("production Cache prefix UPDATE lacks selected_index")
        forced_triggers[index] = 1
        forced_candidates[index] = int(selected_index)

    forced_triggers[int(node_index)] = 0 if candidate is None else 1
    if candidate is not None:
        if force_update_candidate_index is None:
            raise ValueError("forced UPDATE replay requires an exact candidate index")
        forced_candidates[int(node_index)] = int(force_update_candidate_index)

    replay_policy = _DeterministicReplayOverrides(
        trainer.policy,
        forced_triggers,
        forced_candidates,
    )
    with torch.no_grad():
        replay = sample_and_apply_joint_cache_uav_operation_configuration(
            replay_policy,
            np.asarray(transition.observation, dtype=np.float32),
            np.asarray(transition.previous_cache_state, dtype=np.int8),
            transition.previous_uav_positions,
            step,
            trainer.context,
            PrimitiveTensorSpec.create(trainer.context),
            deterministic=True,
        )

    expected_prefix = np.asarray(state["working_cache"], dtype=np.int8)
    if not np.array_equal(
        np.asarray(replay.final_cache_state, dtype=np.int8)[: int(node_index)],
        expected_prefix[: int(node_index)],
    ):
        raise RuntimeError("deterministic replay changed the fixed Cache prefix")
    replay_transition = replace(
        transition,
        action=np.asarray(replay.derived_action, dtype=np.int8),
        next_cache_state=np.asarray(replay.final_cache_state, dtype=np.int8).copy(),
        next_uav_positions={
            key: tuple(float(value) for value in position)
            for key, position in replay.final_uav_positions.items()
        },
        cache_conditional_actions=tuple(replay.cache_conditional_actions),
        uav_conditional_actions=tuple(replay.uav_conditional_actions),
    )
    return _execute_cache_branch(
        trainer,
        replay_transition,
        step,
        next_step,
        node_index,
        state,
        candidate,
    )


def _diagnostic_sign(delta: float | None, *, epsilon: float = 1.0e-10) -> str | None:
    if delta is None:
        return None
    if float(delta) > float(epsilon):
        return "UPDATE_BETTER"
    if float(delta) < -float(epsilon):
        return "KEEP_BETTER"
    return "TIE"


def evaluate_cache_trigger_causal_gate(
    trainer: Any,
    rollout: Any,
    *,
    target_transition_indices: tuple[int, ...],
    target_node_indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Evaluate one small full-exact Cache Trigger causal-gate batch.

    Every legal UPDATE target configuration is evaluated by the current exact
    environment branch.  The policy-consistent values use deterministic frozen
    policy replay for later Cache heads and UAV decisions; they are not exact
    stochastic-policy Q values.  This diagnostic is intentionally not called
    by the PPO training path.
    """

    all_transitions = tuple(rollout.transitions)
    selected_indices = tuple(int(index) for index in target_transition_indices)
    if not selected_indices or len(set(selected_indices)) != len(selected_indices):
        raise ValueError("causal-gate transition indices must be non-empty and unique")
    if len(selected_indices) > 8:
        raise ValueError("causal-gate diagnostic is limited to eight transitions")
    if any(index < 0 or index >= len(all_transitions) for index in selected_indices):
        raise IndexError("causal-gate transition index is out of range")

    node_count = len(trainer.context.action_spec.cache_node_ids)
    if target_node_indices is None:
        node_indices = tuple(range(node_count))
    else:
        node_indices = tuple(int(index) for index in target_node_indices)
        if not node_indices or len(set(node_indices)) != len(node_indices):
            raise ValueError("causal-gate node indices must be non-empty and unique")
        if any(index < 0 or index >= node_count for index in node_indices):
            raise IndexError("causal-gate node index is out of range")
    if node_count != 9:
        raise ValueError(f"causal-gate expects the configured nine Cache heads, got {node_count}")
    if not bool(getattr(trainer.policy, "true_binary_trigger", False)):
        raise RuntimeError("causal-gate requires the binary KEEP/UPDATE policy")
    if rollout.cache_candidate_features is None:
        raise ValueError("causal-gate rollout lacks Cache candidate tensors")
    feature_dim = int(rollout.cache_candidate_features.shape[3])

    transitions = tuple(all_transitions[index] for index in selected_indices)
    origin_start = min(int(item.origin_index) for item in transitions)
    origin_end = max(int(item.origin_index) for item in transitions)
    support_count = origin_end - origin_start + 1
    if not any(
        bool(item.done) and int(item.origin_index) == origin_end
        for item in transitions
    ):
        support_count += 1
    from .trainer import aligned_train_steps

    full_information = evaluate_cache_full_information_values(
        trainer,
        rollout,
        target_transition_indices=selected_indices,
        target_node_indices=node_indices,
        trigger_temperature=float(
            getattr(trainer, "cache_trigger_teacher_temperature", 1.0)
        ),
        candidate_temperature=float(
            getattr(trainer, "cache_candidate_teacher_temperature", 1.0)
        ),
    )
    evaluation_position = {
        int(index): position for position, index in enumerate(selected_indices)
    }

    aligned_steps = aligned_train_steps(
        trainer.context,
        origin_start,
        support_count,
        include_bootstrap_step=False,
    )
    steps_by_origin = {int(item.origin_index): item for item in aligned_steps}
    rows: list[dict[str, Any]] = []
    full_exact_update_logical_count = 0
    full_exact_update_environment_count = 0
    fixed_environment_count = 0
    policy_consistent_environment_count = 0
    started_at = time.perf_counter()

    for transition_index in selected_indices:
        transition = all_transitions[transition_index]
        step = steps_by_origin.get(int(transition.origin_index))
        if step is None:
            raise RuntimeError("causal-gate replay is missing its current aligned step")
        next_step = steps_by_origin.get(int(transition.origin_index) + 1)
        if not bool(transition.done) and next_step is None:
            raise RuntimeError("causal-gate replay is missing its next aligned step")

        for node_index in node_indices:
            state = _node_policy_state(
                trainer,
                transition,
                step,
                node_index,
                feature_dim,
            )
            actual_action_value = (
                int(transition.cache_operation_actions[node_index])
                if transition.cache_operation_actions is not None
                else int(np.asarray(transition.action, dtype=np.int8)[node_index])
            )
            if actual_action_value not in (0, 1):
                raise RuntimeError("causal-gate sampled trigger is not KEEP/UPDATE")
            actual_action = "UPDATE" if actual_action_value else "KEEP"
            candidates = tuple(state["candidates"])
            candidate_probabilities = np.asarray(
                state["candidate_probabilities"], dtype=np.float64
            )
            if len(candidates) != len(candidate_probabilities):
                raise RuntimeError("causal-gate candidate and policy distributions differ")
            if len(candidates) and (
                not np.all(np.isfinite(candidate_probabilities))
                or np.any(candidate_probabilities < 0.0)
                or not np.isclose(float(candidate_probabilities.sum()), 1.0)
            ):
                raise FloatingPointError("causal-gate candidate probabilities are invalid")

            sample_position = evaluation_position[int(transition_index)]
            q_keep_fixed = float(
                full_information["q_keep"][sample_position, node_index]
            )
            exact_candidate_q = np.asarray(
                full_information["candidate_q_values"][
                    sample_position, node_index, : len(candidates)
                ],
                dtype=np.float64,
            )
            if len(candidates) and not np.all(np.isfinite(exact_candidate_q)):
                raise FloatingPointError(
                    "causal-gate full-information candidate Q is non-finite"
                )
            full_exact_update_logical_count += len(candidates)

            selected_candidate_index: int | None = None
            if actual_action_value == 1:
                if transition.cache_selected_indices is not None:
                    selected_candidate_index = int(
                        transition.cache_selected_indices[node_index]
                    )
                else:
                    stored = _production_action_map(transition)[node_index].get(
                        "selected_index"
                    )
                    selected_candidate_index = None if stored is None else int(stored)
                if selected_candidate_index is None or not (
                    0 <= selected_candidate_index < len(candidates)
                ):
                    raise RuntimeError("sampled UPDATE lacks a matching legal candidate")
                stored_target = _production_action_map(transition)[node_index].get(
                    "target_cache"
                )
                if stored_target is not None and not np.array_equal(
                    np.asarray(stored_target, dtype=np.int8),
                    np.asarray(
                        candidates[selected_candidate_index].target_node_cache,
                        dtype=np.int8,
                    ),
                ):
                    raise RuntimeError(
                        "sampled UPDATE index does not identify the stored target configuration"
                    )

            policy_argmax_candidate_index: int | None = None
            best_exact_candidate_index: int | None = None
            q_sampled_update_fixed: float | None = None
            q_policy_argmax_update_fixed: float | None = None
            q_best_exact_update_fixed: float | None = None
            delta_sampled_fixed: float | None = None
            delta_policy_argmax_fixed: float | None = None
            delta_best_exact_fixed: float | None = None
            selected_rank: int | None = None
            selected_regret: float | None = None
            policy_argmax_rank: int | None = None
            policy_argmax_regret: float | None = None
            best_candidate_probability: float | None = None
            q_keep_policy_consistent: float | None = None
            q_best_policy_consistent: float | None = None
            delta_best_policy_consistent: float | None = None

            if len(candidates):
                policy_argmax_candidate_index = int(np.argmax(candidate_probabilities))
                best_exact_candidate_index = int(np.argmax(exact_candidate_q))
                q_policy_argmax_update_fixed = float(
                    exact_candidate_q[policy_argmax_candidate_index]
                )
                q_best_exact_update_fixed = float(
                    exact_candidate_q[best_exact_candidate_index]
                )
                delta_policy_argmax_fixed = (
                    q_policy_argmax_update_fixed - q_keep_fixed
                )
                delta_best_exact_fixed = q_best_exact_update_fixed - q_keep_fixed
                best_candidate_probability = float(
                    candidate_probabilities[best_exact_candidate_index]
                )
                max_q = float(np.max(exact_candidate_q))
                rank_tolerance = 1.0e-10 * max(1.0, abs(max_q))
                ranks = 1 + np.sum(
                    exact_candidate_q[None, :]
                    > exact_candidate_q[:, None] + rank_tolerance,
                    axis=1,
                )
                policy_argmax_rank = int(ranks[policy_argmax_candidate_index])
                policy_argmax_regret = max(
                    0.0,
                    max_q - float(exact_candidate_q[policy_argmax_candidate_index]),
                )
                if selected_candidate_index is not None:
                    q_sampled_update_fixed = float(
                        exact_candidate_q[selected_candidate_index]
                    )
                    delta_sampled_fixed = q_sampled_update_fixed - q_keep_fixed
                    selected_rank = int(ranks[selected_candidate_index])
                    selected_regret = max(
                        0.0,
                        max_q - float(exact_candidate_q[selected_candidate_index]),
                    )

                keep_policy_branch = _execute_policy_consistent_cache_branch(
                    trainer,
                    transition,
                    step,
                    next_step,
                    node_index,
                    state,
                    None,
                    force_update_candidate_index=None,
                )
                q_keep_policy_consistent = float(
                    _branch_q_values(trainer, [keep_policy_branch])[0]
                )
                policy_consistent_environment_count += 1
                best_policy_branch = _execute_policy_consistent_cache_branch(
                    trainer,
                    transition,
                    step,
                    next_step,
                    node_index,
                    state,
                    candidates[best_exact_candidate_index],
                    force_update_candidate_index=best_exact_candidate_index,
                )
                q_best_policy_consistent = float(
                    _branch_q_values(trainer, [best_policy_branch])[0]
                )
                policy_consistent_environment_count += 1
                delta_best_policy_consistent = (
                    q_best_policy_consistent - q_keep_policy_consistent
                )
            else:
                keep_policy_branch = _execute_policy_consistent_cache_branch(
                    trainer,
                    transition,
                    step,
                    next_step,
                    node_index,
                    state,
                    None,
                    force_update_candidate_index=None,
                )
                q_keep_policy_consistent = float(
                    _branch_q_values(trainer, [keep_policy_branch])[0]
                )
                policy_consistent_environment_count += 1

            trigger_teacher_row = np.asarray(
                full_information["trigger_teacher_probabilities"][
                    sample_position, node_index
                ],
                dtype=np.float64,
            )
            candidate_teacher_row = np.asarray(
                full_information["candidate_teacher_probabilities"][
                    sample_position, node_index, : len(candidates)
                ],
                dtype=np.float64,
            )
            candidate_teacher_top1 = (
                int(np.argmax(candidate_teacher_row)) if len(candidates) else None
            )
            candidate_teacher_best_probability = (
                float(candidate_teacher_row[best_exact_candidate_index])
                if best_exact_candidate_index is not None
                else None
            )
            trigger_policy_row = np.asarray(
                full_information["trigger_policy_probabilities"][
                    sample_position, node_index
                ],
                dtype=np.float64,
            )
            node_id = str(trainer.context.action_spec.cache_node_ids[node_index])
            rows.append(
                {
                    "transition_index": int(transition_index),
                    "origin_index": int(transition.origin_index),
                    "slot": int(transition.slot),
                    "node_index": int(node_index),
                    "node_id": node_id,
                    "actual_action": actual_action,
                    "p_keep": float(state["p_keep"]),
                    "p_update": float(state["p_update"]),
                    "legal_candidate_count": int(len(candidates)),
                    "q_keep_fixed": q_keep_fixed,
                    "q_best_update_fixed": q_best_exact_update_fixed,
                    "delta_best": delta_best_exact_fixed,
                    "best_candidate_index": best_exact_candidate_index,
                    "trigger_teacher_keep_probability": float(trigger_teacher_row[0]),
                    "trigger_teacher_update_probability": float(trigger_teacher_row[1]),
                    "candidate_teacher_top1_index": candidate_teacher_top1,
                    "candidate_teacher_best_probability": (
                        None
                        if candidate_teacher_top1 is None
                        else float(candidate_teacher_row[candidate_teacher_top1])
                    ),
                    "trigger_policy_keep_probability": float(trigger_policy_row[0]),
                    "trigger_policy_update_probability": float(trigger_policy_row[1]),
                    "candidate_policy_top1_index": policy_argmax_candidate_index,
                    "candidate_policy_best_exact_probability": best_candidate_probability,
                    "trigger_teacher_valid": bool(
                        full_information["trigger_teacher_valid_mask"][
                            sample_position, node_index
                        ]
                    ),
                    "q_sampled_update_fixed": q_sampled_update_fixed,
                    "q_policy_argmax_update_fixed": q_policy_argmax_update_fixed,
                    "q_best_exact_update_fixed": q_best_exact_update_fixed,
                    "best_exact_candidate_index": best_exact_candidate_index,
                    "delta_sampled_fixed": delta_sampled_fixed,
                    "delta_policy_argmax_fixed": delta_policy_argmax_fixed,
                    "delta_best_exact_fixed": delta_best_exact_fixed,
                    "selected_candidate_index": selected_candidate_index,
                    "policy_argmax_candidate_index": policy_argmax_candidate_index,
                    "selected_candidate_exact_rank": selected_rank,
                    "policy_argmax_candidate_exact_rank": policy_argmax_rank,
                    "selected_candidate_exact_regret": selected_regret,
                    "policy_argmax_candidate_exact_regret": policy_argmax_regret,
                    "best_exact_candidate_policy_probability": best_candidate_probability,
                    "q_keep_policy_consistent": q_keep_policy_consistent,
                    "q_best_update_policy_consistent": q_best_policy_consistent,
                    "delta_best_policy_consistent": delta_best_policy_consistent,
                    "fixed_downstream_sign": _diagnostic_sign(delta_best_exact_fixed),
                    "policy_consistent_sign": _diagnostic_sign(
                        delta_best_policy_consistent
                    ),
                    "downstream_sign_flip": (
                        _diagnostic_sign(delta_best_exact_fixed)
                        != _diagnostic_sign(delta_best_policy_consistent)
                        if delta_best_policy_consistent is not None
                        else None
                    ),
                    "delta_sampled_sign": _diagnostic_sign(delta_sampled_fixed),
                    "delta_policy_argmax_sign": _diagnostic_sign(
                        delta_policy_argmax_fixed
                    ),
                    "delta_best_exact_sign": _diagnostic_sign(delta_best_exact_fixed),
                    "cache_changed": bool(
                        not np.array_equal(
                            transition.previous_cache_state[node_index],
                            transition.next_cache_state[node_index],
                        )
                    ),
                }
            )

    full_exact_update_logical_count = int(full_information["logical_candidate_count"])
    fixed_environment_count = int(full_information["keep_branch_count"])
    full_exact_update_environment_count = int(
        full_information["counterfactual_environment_executions"]
        - full_information["keep_branch_count"]
    )
    runtime_seconds = float(time.perf_counter() - started_at)
    return {
        "policy_consistent_mode": "DETERMINISTIC_FROZEN_POLICY_REPLAY",
        "q_definition": "Q_UPDATE_MINUS_Q_KEEP",
        "target_transition_indices": list(selected_indices),
        "target_node_indices": list(node_indices),
        "rows": rows,
        "value_evaluator": str(full_information["value_evaluator"]),
        "downstream_mode": str(full_information["downstream_mode"]),
        "full_exact_update_logical_configurations": int(
            full_exact_update_logical_count
        ),
        "full_exact_update_environment_branches": int(
            full_exact_update_environment_count
        ),
        "fixed_downstream_environment_branches": int(fixed_environment_count),
        "policy_consistent_environment_branches": int(
            policy_consistent_environment_count
        ),
        "runtime_seconds": runtime_seconds,
    }


def _cache_branch_identity(
    transition: Any,
    node_index: int,
    state: dict[str, Any],
    candidate: Any | None,
) -> tuple[int, bytes, tuple[tuple[str, float, float], ...]]:
    """Return the complete physical identity of one Cache branch.

    The transition origin identifies the fixed task, vehicle, and prediction
    inputs.  The remaining identity is the complete post-configuration Cache
    state plus all UAV positions.  Operation labels and candidate indices are
    intentionally absent: two logical candidates are reusable only when this
    complete physical state is identical.
    """

    final_cache = np.asarray(transition.next_cache_state, dtype=np.int8).copy()
    if candidate is None:
        final_cache[int(node_index)] = np.asarray(
            state["working_cache"][int(node_index)], dtype=np.int8
        )
    else:
        final_cache[int(node_index)] = np.asarray(
            candidate.target_node_cache, dtype=np.int8
        )
    final_uav = tuple(
        sorted(
            (
                str(node_id),
                float(np.asarray(xy, dtype=np.float64)[0]),
                float(np.asarray(xy, dtype=np.float64)[1]),
            )
            for node_id, xy in transition.next_uav_positions.items()
        )
    )
    return int(transition.origin_index), final_cache.tobytes(), final_uav


def _branch_q_values(trainer: Any, branches: list[dict[str, Any]]) -> np.ndarray:
    observations = [branch["next_observation"] for branch in branches]
    if any(value is None for value in observations):
        if any(value is not None for value in observations):
            raise RuntimeError("Cache branch next observations are partially missing")
        next_values = np.zeros(len(branches), dtype=np.float64)
    else:
        observation_batch = np.stack(
            [np.asarray(value, dtype=np.float32) for value in observations], axis=0
        )
        with torch.no_grad():
            value_tensor = trainer.policy.critic_value(observation_batch)
        next_values = value_tensor.detach().cpu().numpy().reshape(-1).astype(np.float64)
    rewards = np.asarray([branch["reward"] for branch in branches], dtype=np.float64)
    q_values = rewards + float(trainer.hyperparameters.gamma) * next_values
    if not np.all(np.isfinite(q_values)):
        raise FloatingPointError("Exact Cache Q values are non-finite")
    return q_values


def _rank_correlation(predicted: np.ndarray, exact: np.ndarray) -> float:
    """Return a dependency-free Spearman-style rank correlation."""

    predicted_a = np.asarray(predicted, dtype=np.float64).reshape(-1)
    exact_a = np.asarray(exact, dtype=np.float64).reshape(-1)
    if len(predicted_a) < 2 or len(exact_a) < 2:
        return float("nan")
    if not np.all(np.isfinite(predicted_a)) or not np.all(np.isfinite(exact_a)):
        raise FloatingPointError("Cache surrogate rank inputs are non-finite")

    def _average_ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        sorted_values = values[order]
        start = 0
        while start < len(values):
            stop = start + 1
            while stop < len(values) and sorted_values[stop] == sorted_values[start]:
                stop += 1
            ranks[order[start:stop]] = 0.5 * float(start + stop - 1)
            start = stop
        return ranks

    predicted_ranks = _average_ranks(predicted_a)
    exact_ranks = _average_ranks(exact_a)
    predicted_centered = predicted_ranks - float(np.mean(predicted_ranks))
    exact_centered = exact_ranks - float(np.mean(exact_ranks))
    denominator = float(
        np.sqrt(np.sum(predicted_centered**2) * np.sum(exact_centered**2))
    )
    if denominator <= 0.0:
        return float("nan")
    return float(np.sum(predicted_centered * exact_centered) / denominator)


def _screening_gate_metrics(rows: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    """Aggregate full-Exact rows for K selection and safety gating."""

    max_k = 12
    recall_values: dict[int, list[float]] = {k: [] for k in range(1, max_k + 1)}
    regret_values: dict[int, list[float]] = {k: [] for k in range(1, max_k + 1)}
    normalized_regret_values: dict[int, list[float]] = {
        k: [] for k in range(1, max_k + 1)
    }
    rank_values: list[float] = []
    for row in rows:
        predicted = np.asarray(row["predicted_delta_q"], dtype=np.float64).reshape(-1)
        exact_q = np.asarray(row["exact_q"], dtype=np.float64).reshape(-1)
        if len(predicted) != len(exact_q) or not len(predicted):
            continue
        if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(exact_q)):
            raise FloatingPointError("Cache screening gate inputs are non-finite")
        rank = _rank_correlation(predicted, exact_q)
        if np.isfinite(rank):
            rank_values.append(rank)
        best_q = float(np.max(exact_q))
        best_mask = np.isclose(exact_q, best_q, rtol=1.0e-10, atol=1.0e-10)
        ranking = np.argsort(-predicted, kind="mergesort")
        q_range = float(np.max(exact_q) - np.min(exact_q))
        for k in range(1, min(max_k, len(predicted)) + 1):
            top = ranking[:k]
            recall_values[k].append(float(np.any(best_mask[top])))
            selected_best_q = float(np.max(exact_q[top]))
            regret = max(0.0, best_q - selected_best_q)
            normalized = regret / (q_range + 1.0e-12)
            regret_values[k].append(regret)
            normalized_regret_values[k].append(normalized)

    def _stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {
                "mean": None,
                "median": None,
                "p90": None,
                "p95": None,
                "max": None,
            }
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90.0)),
            "p95": float(np.percentile(array, 95.0)),
            "max": float(np.max(array)),
        }

    recall_at_k = {
        str(k): float(np.mean(values))
        for k, values in recall_values.items()
        if values
    }
    regret_by_k = {
        str(k): _stats(values)
        for k, values in regret_values.items()
        if values
    }
    normalized_regret_by_k = {
        str(k): _stats(values)
        for k, values in normalized_regret_values.items()
        if values
    }
    normalized_p95 = {
        str(k): float(stats["p95"])
        for k, stats in normalized_regret_by_k.items()
        if stats["p95"] is not None
    }
    return {
        "row_count": int(len(rows)),
        "recall_at_k": recall_at_k,
        "best_miss_rate_at_k": {
            key: float(1.0 - value) for key, value in recall_at_k.items()
        },
        "regret_at_k": regret_by_k,
        "normalized_regret_at_k": normalized_regret_by_k,
        "normalized_regret_p95_at_k": normalized_p95,
        "rank_correlation": (
            None if not rank_values else float(np.mean(np.asarray(rank_values)))
        ),
    }


def _selected_cache_candidate_index(
    transition: Any,
    node_index: int,
    candidates: list[Any],
) -> int | None:
    """Resolve the production target candidate without trusting padding."""

    if not candidates:
        return None
    operation_actions = getattr(transition, "cache_operation_actions", None)
    if operation_actions is not None and int(operation_actions[node_index]) == 0:
        return None
    selected_indices = getattr(transition, "cache_selected_indices", None)
    if selected_indices is not None:
        selected_index = int(selected_indices[node_index])
        if 0 <= selected_index < len(candidates):
            return selected_index
    actions = _production_action_map(transition)
    action = actions.get(int(node_index), {})
    target = action.get("target_cache")
    if target is None:
        return None
    target_array = np.asarray(target, dtype=np.int8)
    for candidate_index, candidate in enumerate(candidates):
        if np.array_equal(
            target_array,
            np.asarray(candidate.target_node_cache, dtype=np.int8),
        ):
            return int(candidate_index)
    return None


def compute_exact_node_specific_cache_credit(
    trainer: Any,
    rollout: Any,
    *,
    target_transition_indices: tuple[int, ...] | None = None,
    target_node_indices: tuple[int, ...] | None = None,
    include_details: bool = False,
    reuse_identical_branches: bool = True,
    force_full_exact: bool = False,
) -> dict[str, Any]:
    """Replay Cache branches with the legacy hybrid path or full exact audit.

    ``force_full_exact`` is used by the sparse Action-Q audit path.  It keeps
    the existing Matching/KKT/reward implementation as the single exact
    implementation while disabling legacy surrogate screening and training.
    """

    all_transitions = tuple(rollout.transitions)
    if not all_transitions:
        raise ValueError("Exact Cache credit requires a non-empty rollout")
    if target_transition_indices is None:
        selected_indices = tuple(range(len(all_transitions)))
    else:
        selected_indices = tuple(int(index) for index in target_transition_indices)
        if not selected_indices or len(set(selected_indices)) != len(selected_indices):
            raise ValueError("Cache target transition indices must be non-empty and unique")
        if any(index < 0 or index >= len(all_transitions) for index in selected_indices):
            raise IndexError("Cache target transition index is out of range")
    transitions = tuple(all_transitions[index] for index in selected_indices)
    origin_start = min(int(item.origin_index) for item in transitions)
    origin_end = max(int(item.origin_index) for item in transitions)
    from .trainer import aligned_train_steps
    from .primitive_control import PrimitiveTensorSpec

    support_step_count = origin_end - origin_start + 1
    if not any(
        bool(item.done) and int(item.origin_index) == origin_end
        for item in transitions
    ):
        support_step_count += 1
    aligned_steps = aligned_train_steps(
        trainer.context,
        origin_start,
        support_step_count,
    )
    steps_by_origin = {int(step.origin_index): step for step in aligned_steps}
    node_count = len(trainer.context.action_spec.cache_node_ids)
    if target_node_indices is None:
        node_indices = tuple(range(node_count))
    else:
        node_indices = tuple(int(index) for index in target_node_indices)
        if not node_indices or len(set(node_indices)) != len(node_indices):
            raise ValueError("Cache target node indices must be non-empty and unique")
        if any(index < 0 or index >= node_count for index in node_indices):
            raise IndexError("Cache target node index is out of range")
    if rollout.cache_candidate_features is None:
        raise ValueError("Rollout does not expose Cache candidate tensor capacity")
    candidate_capacity = int(rollout.cache_candidate_features.shape[2])
    feature_dim = int(rollout.cache_candidate_features.shape[3])
    primitive_spec = PrimitiveTensorSpec.create(trainer.context)
    if int(primitive_spec.cache_feature_dim) != feature_dim:
        raise RuntimeError(
            "Rollout Cache feature width does not match the production tensor spec"
        )
    sample_count = len(transitions)
    trigger_advantages = np.zeros((sample_count, node_count), dtype=np.float64)
    trigger_old_probabilities = np.zeros((sample_count, node_count, 2), dtype=np.float64)
    q_keep = np.zeros((sample_count, node_count), dtype=np.float64)
    q_update = np.zeros((sample_count, node_count), dtype=np.float64)
    baseline = np.zeros((sample_count, node_count), dtype=np.float64)
    candidate_q_values = np.zeros(
        (sample_count, node_count, candidate_capacity), dtype=np.float64
    )
    candidate_exact_q_values = np.full(
        (sample_count, node_count, candidate_capacity), np.nan, dtype=np.float64
    )
    candidate_surrogate_q_values = np.full(
        (sample_count, node_count, candidate_capacity), np.nan, dtype=np.float64
    )
    candidate_predicted_delta_q = np.zeros_like(candidate_q_values)
    candidate_predicted_sigma = np.zeros_like(candidate_q_values)
    candidate_probabilities = np.zeros_like(candidate_q_values)
    candidate_valid_mask = np.zeros(
        (sample_count, node_count, candidate_capacity), dtype=bool
    )
    candidate_exact_evaluated_mask = np.zeros(
        (sample_count, node_count, candidate_capacity), dtype=bool
    )
    candidate_operation_indices = np.zeros(
        (sample_count, node_count, candidate_capacity), dtype=np.int64
    )
    candidate_proposal_sources = np.full(
        (sample_count, node_count, candidate_capacity), "", dtype=object
    )
    candidate_q_sources = np.full(
        (sample_count, node_count, candidate_capacity), "", dtype=object
    )
    candidate_hamming_distances = np.full(
        (sample_count, node_count, candidate_capacity), np.nan, dtype=np.float64
    )
    candidate_features = np.zeros(
        (sample_count, node_count, candidate_capacity, feature_dim), dtype=np.float32
    )
    candidate_q_update = q_update.copy()
    candidate_weighted_mean_residual = np.zeros((sample_count, node_count), dtype=np.float64)
    trigger_weighted_mean_residual = np.zeros((sample_count, node_count), dtype=np.float64)
    candidate_counts = np.zeros((sample_count, node_count), dtype=np.int64)
    details: list[dict[str, Any]] = []
    environment_executions = 0
    logical_branch_count = 0
    dedup_reused_branch_count = 0
    logical_candidate_count = 0
    exact_candidate_count = 0
    surrogate_candidate_count = 0
    keep_exact_count = 0
    full_exact_audit_count = 0
    full_exact_environment_baseline = 0
    screening_rows: list[dict[str, np.ndarray]] = []
    exact_context_labels: list[np.ndarray] = []
    exact_feature_labels: list[np.ndarray] = []
    exact_delta_labels: list[float] = []
    exact_prediction_labels: list[float] = []
    uncertainty_values: list[float] = []
    surrogate_inference_runtime_seconds = 0.0
    shadow_epoch = bool(force_full_exact) or not bool(
        getattr(trainer, "cache_surrogate_shadow_complete", False)
    )
    screening_active = bool(
        not force_full_exact
        and getattr(trainer, "cache_surrogate_screening_active", False)
        and getattr(trainer, "cache_surrogate_screening_ready", False)
    )
    surrogate_candidate_source = (
        "FULL_EXACT_AUDIT"
        if force_full_exact
        else ("HYBRID_EXACT_SURROGATE" if screening_active else "FULL_EXACT_Q")
    )
    branch_cache: dict[
        tuple[int, bytes, tuple[tuple[str, float, float], ...]], dict[str, Any]
    ] = {}
    started_at = time.perf_counter()

    for sample_index, transition in enumerate(transitions):
        # Reuse is valid only within one fixed transition.  Clearing here
        # avoids retaining large Matching/KKT result dictionaries across the
        # 300-slot rollout while preserving the complete-state identity rule.
        branch_cache.clear()
        full_exact_branch_keys: set[
            tuple[int, bytes, tuple[tuple[str, float, float], ...]]
        ] = set()
        step = steps_by_origin.get(int(transition.origin_index))
        if step is None:
            raise RuntimeError("Cache replay is missing the aligned current step")
        next_step = steps_by_origin.get(int(transition.origin_index) + 1)
        for node_index in node_indices:
            state = _node_policy_state(
                trainer,
                transition,
                step,
                node_index,
                feature_dim,
            )
            candidates = state["candidates"]
            if len(candidates) > candidate_capacity:
                raise RuntimeError(
                    "Legal Cache candidate count exceeds rollout tensor capacity: "
                    f"count={len(candidates)}, capacity={candidate_capacity}"
                )
            candidate_count = len(candidates)
            logical_candidate_count += candidate_count
            keep_exact_count += 1
            for full_candidate in (None, *candidates):
                full_exact_branch_keys.add(
                    _cache_branch_identity(
                        transition,
                        node_index,
                        state,
                        full_candidate,
                    )
                )
            predicted_delta_q = np.zeros(candidate_count, dtype=np.float64)
            predicted_sigma = np.zeros(candidate_count, dtype=np.float64)
            candidate_feature_matrix = np.zeros(
                (candidate_count, feature_dim), dtype=np.float32
            )
            exact_indices: list[int] = []
            full_exact_for_row = bool(force_full_exact or shadow_epoch or not screening_active)
            periodic_audit = False
            if candidate_count:
                candidate_feature_matrix = np.stack(state["candidate_features"]).astype(
                    np.float32
                )
            if candidate_count and not force_full_exact:
                inference_started_at = time.perf_counter()
                predicted_delta_q, predicted_sigma = trainer._predict_cache_surrogate(
                    np.asarray(state["conditional_context"], dtype=np.float32),
                    candidate_feature_matrix,
                )
                surrogate_inference_runtime_seconds += float(
                    time.perf_counter() - inference_started_at
                )
                uncertainty_values.extend(predicted_sigma.tolist())
                periodic_audit = bool(
                    screening_active
                    and (
                        (
                            int(transition.origin_index)
                            - int(trainer.context.train_origin_start)
                        )
                        % int(getattr(trainer, "cache_surrogate_audit_interval", 20))
                        == 0
                    )
                )
                full_exact_for_row = bool(full_exact_for_row or periodic_audit)
                if full_exact_for_row:
                    exact_indices = list(range(candidate_count))
                    if periodic_audit:
                        full_exact_audit_count += 1
                else:
                    selected_indices = set()
                    selected_index = _selected_cache_candidate_index(
                        transition,
                        node_index,
                        candidates,
                    )
                    if selected_index is not None:
                        selected_indices.add(int(selected_index))
                    selected_k = int(
                        getattr(trainer, "cache_surrogate_selected_k", None) or 1
                    )
                    selected_k = max(1, min(selected_k, candidate_count))
                    ucb_score = predicted_delta_q + 2.0 * predicted_sigma
                    selected_indices.update(
                        int(index)
                        for index in np.argsort(-ucb_score, kind="mergesort")[:selected_k]
                    )
                    unselected = sorted(set(range(candidate_count)) - selected_indices)
                    if unselected:
                        selected_indices.add(
                            int(
                                trainer.cache_surrogate_rng.choice(
                                    np.asarray(unselected, dtype=np.int64)
                                )
                            )
                        )
                    max_uncertainty_index = int(np.argmax(predicted_sigma))
                    selected_indices.add(max_uncertainty_index)
                    exact_indices = sorted(selected_indices)
            if force_full_exact and candidate_count:
                exact_indices = list(range(candidate_count))
            exact_index_set = set(exact_indices)
            exact_candidate_count += len(exact_indices)
            surrogate_candidate_count += candidate_count - len(exact_indices)
            logical_branch_count += 1 + candidate_count
            branches: list[dict[str, Any]] = []
            branch_candidates: list[Any | None] = [None]
            branch_candidates.extend(candidates[index] for index in exact_indices)
            for candidate in branch_candidates:
                branch_key = _cache_branch_identity(
                    transition,
                    node_index,
                    state,
                    candidate,
                )
                branch = branch_cache.get(branch_key) if reuse_identical_branches else None
                if branch is None:
                    branch = _execute_cache_branch(
                        trainer,
                        transition,
                        step,
                        next_step,
                        node_index,
                        state,
                        candidate,
                    )
                    if reuse_identical_branches:
                        branch_cache[branch_key] = branch
                    environment_executions += 1
                else:
                    dedup_reused_branch_count += 1
                branches.append(branch)
            q_values = _branch_q_values(trainer, branches)
            keep_q = float(q_values[0])
            exact_q_by_index = np.full(candidate_count, np.nan, dtype=np.float64)
            if exact_indices:
                exact_q_by_index[np.asarray(exact_indices, dtype=np.int64)] = q_values[1:]
            candidate_q = keep_q + np.asarray(predicted_delta_q, dtype=np.float64)
            if exact_indices:
                candidate_q[np.asarray(exact_indices, dtype=np.int64)] = exact_q_by_index[
                    np.asarray(exact_indices, dtype=np.int64)
                ]
            update_q = (
                float(np.dot(state["candidate_probabilities"], candidate_q))
                if len(candidate_q)
                else keep_q
            )
            state_baseline = float(
                state["p_keep"] * keep_q + state["p_update"] * update_q
            )
            trigger_weighted_mean_residual[sample_index, node_index] = float(
                state["p_keep"] * (keep_q - state_baseline)
                + state["p_update"] * (update_q - state_baseline)
            )
            selected_update = int(transition.cache_operation_actions[node_index]) != 0
            trigger_advantages[sample_index, node_index] = (
                update_q - state_baseline if selected_update else keep_q - state_baseline
            )
            trigger_old_probabilities[sample_index, node_index] = (
                state["p_keep"],
                state["p_update"],
            )
            q_keep[sample_index, node_index] = keep_q
            q_update[sample_index, node_index] = update_q
            baseline[sample_index, node_index] = state_baseline
            candidate_counts[sample_index, node_index] = candidate_count
            if candidate_count:
                candidate_q_values[sample_index, node_index, :candidate_count] = candidate_q
                candidate_exact_q_values[
                    sample_index, node_index, :candidate_count
                ] = exact_q_by_index
                surrogate_q_row = keep_q + np.asarray(
                    predicted_delta_q, dtype=np.float64
                )
                if exact_indices:
                    surrogate_q_row[np.asarray(exact_indices, dtype=np.int64)] = np.nan
                candidate_surrogate_q_values[
                    sample_index, node_index, :candidate_count
                ] = surrogate_q_row
                candidate_predicted_delta_q[
                    sample_index, node_index, :candidate_count
                ] = predicted_delta_q
                candidate_predicted_sigma[
                    sample_index, node_index, :candidate_count
                ] = predicted_sigma
                candidate_probabilities[
                    sample_index, node_index, :candidate_count
                ] = state["candidate_probabilities"]
                candidate_valid_mask[
                    sample_index, node_index, :candidate_count
                ] = True
                candidate_exact_evaluated_mask[
                    sample_index, node_index, :candidate_count
                ] = False
                if exact_indices:
                    candidate_exact_evaluated_mask[
                        sample_index,
                        node_index,
                        np.asarray(exact_indices, dtype=np.int64),
                    ] = True
                candidate_operation_indices[
                    sample_index, node_index, :candidate_count
                ] = state["candidate_operations"]
                candidate_proposal_sources[
                    sample_index, node_index, :candidate_count
                ] = [str(candidate.proposal_source) for candidate in candidates]
                candidate_hamming_distances[
                    sample_index, node_index, :candidate_count
                ] = [
                    float(
                        np.count_nonzero(
                            np.asarray(candidate.target_node_cache, dtype=np.int8)
                            != np.asarray(candidate.before_node_cache, dtype=np.int8)
                        )
                    )
                    for candidate in candidates
                ]
                candidate_features[
                    sample_index, node_index, :candidate_count
                ] = candidate_feature_matrix
                candidate_q_sources[
                    sample_index, node_index, :candidate_count
                ] = np.where(
                    candidate_exact_evaluated_mask[
                        sample_index, node_index, :candidate_count
                    ],
                    "EXACT_Q",
                    "SURROGATE_Q",
                )
                candidate_weighted_mean_residual[sample_index, node_index] = float(
                    np.dot(
                        state["candidate_probabilities"],
                        candidate_q - update_q,
                    )
                )
                for candidate_index in exact_indices:
                    exact_context_labels.append(
                        np.asarray(state["conditional_context"], dtype=np.float32)
                    )
                    exact_feature_labels.append(
                        np.asarray(candidate_feature_matrix[candidate_index], dtype=np.float32)
                    )
                    exact_delta_labels.append(
                        float(exact_q_by_index[candidate_index] - keep_q)
                    )
                    exact_prediction_labels.append(
                        float(predicted_delta_q[candidate_index])
                    )
                if full_exact_for_row and not force_full_exact:
                    screening_rows.append(
                        {
                            "predicted_delta_q": np.asarray(predicted_delta_q, dtype=np.float64),
                            "exact_q": np.asarray(exact_q_by_index, dtype=np.float64),
                        }
                    )
            candidate_q_update[sample_index, node_index] = update_q
            if include_details:
                details.append(
                    {
                        "transition_index": int(selected_indices[sample_index]),
                        "origin_index": int(transition.origin_index),
                        "slot": int(transition.slot),
                        "node_index": int(node_index),
                        "node_id": str(trainer.context.action_spec.cache_node_ids[node_index]),
                        "candidate_count": int(candidate_count),
                        "q_keep": keep_q,
                        "q_update": update_q,
                        "baseline": state_baseline,
                        "trigger_advantage": float(
                            trigger_advantages[sample_index, node_index]
                        ),
                    }
                )
        full_exact_environment_baseline += len(full_exact_branch_keys)
    if not np.all(np.isfinite(trigger_advantages)) or not np.all(
        np.isfinite(candidate_q_values)
    ):
        raise FloatingPointError("Cache hybrid credit contains non-finite values")
    valid_candidate_values = candidate_q_values[candidate_valid_mask]
    if not np.all(np.isfinite(valid_candidate_values)):
        raise FloatingPointError("Cache hybrid candidate Q values are non-finite")
    if np.any(candidate_exact_evaluated_mask & ~candidate_valid_mask):
        raise AssertionError("Cache exact candidate mask contains invalid padding")
    surrogate_train_started_at = time.perf_counter()
    if exact_context_labels and not force_full_exact:
        context_dim = int(getattr(trainer, "cache_surrogate_context_dim"))
        feature_width = int(getattr(trainer, "cache_surrogate_feature_dim"))
        surrogate_train_metrics = trainer._train_cache_surrogate(
            np.asarray(exact_context_labels, dtype=np.float32).reshape(
                -1, context_dim
            ),
            np.asarray(exact_feature_labels, dtype=np.float32).reshape(
                -1, feature_width
            ),
            np.asarray(exact_delta_labels, dtype=np.float32),
        )
    else:
        surrogate_train_metrics = {
            "count": 0,
            "loss": math.nan,
            "mae": math.nan,
            "rmse": math.nan,
        }
    surrogate_train_runtime_seconds = float(
        time.perf_counter() - surrogate_train_started_at
    )
    gate_metrics = _screening_gate_metrics(screening_rows)
    label_predictions = np.asarray(exact_prediction_labels, dtype=np.float64)
    label_targets = np.asarray(exact_delta_labels, dtype=np.float64)
    if len(label_predictions):
        if not np.all(np.isfinite(label_predictions)) or not np.all(
            np.isfinite(label_targets)
        ):
            raise FloatingPointError("Cache surrogate label diagnostics are non-finite")
        label_residual = label_predictions - label_targets
        delta_q_mae = float(np.mean(np.abs(label_residual)))
        delta_q_rmse = float(np.sqrt(np.mean(label_residual**2)))
    else:
        delta_q_mae = float("nan")
        delta_q_rmse = float("nan")
    rank_correlation = gate_metrics.get("rank_correlation")
    uncertainty_array = np.asarray(uncertainty_values, dtype=np.float64)
    uncertainty_mean = (
        float(np.mean(uncertainty_array)) if len(uncertainty_array) else float("nan")
    )
    uncertainty_p90 = (
        float(np.percentile(uncertainty_array, 90.0))
        if len(uncertainty_array)
        else float("nan")
    )
    candidate_q_source = (
        "FULL_EXACT_AUDIT"
        if force_full_exact
        else (
            "HYBRID_EXACT_SURROGATE"
            if surrogate_candidate_count > 0
            else "FULL_EXACT_Q"
        )
    )
    candidate_advantages = np.zeros_like(candidate_q_values)
    candidate_advantages[candidate_valid_mask] = (
        candidate_q_values[candidate_valid_mask]
        - np.repeat(candidate_q_update[..., None], candidate_capacity, axis=-1)[
            candidate_valid_mask
        ]
    )
    node_roles = np.asarray(
        [str(value).upper() for value in trainer.context.cache_nodes["node_roles"]],
        dtype=object,
    )
    trigger_mean_abs_by_role = {}
    for role in ("SV", "RSU", "UAV"):
        role_mask = node_roles == role
        trigger_mean_abs_by_role[role] = (
            float(np.mean(np.abs(trigger_advantages[:, role_mask])))
            if np.any(role_mask)
            else float("nan")
        )
    return {
        "trigger_advantages": trigger_advantages.astype(np.float32),
        "trigger_old_probabilities": trigger_old_probabilities.astype(np.float32),
        "q_keep": q_keep.astype(np.float32),
        "q_update": q_update.astype(np.float32),
        "baseline": baseline.astype(np.float32),
        "candidate_q_values": candidate_q_values.astype(np.float32),
        "candidate_exact_q_values": candidate_exact_q_values.astype(np.float32),
        "candidate_surrogate_q_values": candidate_surrogate_q_values.astype(np.float32),
        "candidate_predicted_delta_q": candidate_predicted_delta_q.astype(np.float32),
        "candidate_predicted_sigma": candidate_predicted_sigma.astype(np.float32),
        "candidate_probabilities": candidate_probabilities.astype(np.float32),
        "candidate_valid_mask": candidate_valid_mask,
        "candidate_exact_evaluated_mask": candidate_exact_evaluated_mask,
        "candidate_operation_indices": candidate_operation_indices,
        "candidate_proposal_sources": candidate_proposal_sources,
        "candidate_q_sources": candidate_q_sources,
        "candidate_hamming_distances": candidate_hamming_distances,
        "candidate_features": candidate_features,
        "candidate_q_update": candidate_q_update.astype(np.float32),
        "candidate_weighted_mean_residual": candidate_weighted_mean_residual.astype(
            np.float32
        ),
        "trigger_weighted_mean_residual": trigger_weighted_mean_residual.astype(
            np.float32
        ),
        "trigger_advantage_mean_abs": float(np.mean(np.abs(trigger_advantages))),
        "candidate_advantage_mean_abs": (
            float(np.mean(np.abs(candidate_advantages[candidate_valid_mask])))
            if np.any(candidate_valid_mask)
            else 0.0
        ),
        "trigger_advantage_mean_abs_by_role": trigger_mean_abs_by_role,
        "candidate_counts": candidate_counts,
        "counterfactual_environment_executions": int(environment_executions),
        "logical_candidate_count": int(logical_candidate_count),
        "exact_candidate_count": int(exact_candidate_count),
        "surrogate_candidate_count": int(surrogate_candidate_count),
        "keep_exact_count": int(keep_exact_count),
        "full_exact_audit_count": int(full_exact_audit_count),
        "full_exact_environment_baseline": int(full_exact_environment_baseline),
        "counterfactual_executions_saved": int(
            max(full_exact_environment_baseline - environment_executions, 0)
        ),
        "counterfactual_execution_reduction_fraction": float(
            max(full_exact_environment_baseline - environment_executions, 0)
            / float(max(full_exact_environment_baseline, 1))
        ),
        "surrogate_candidate_source": candidate_q_source,
        "surrogate_screening_active_before": bool(screening_active),
        "shadow_epoch": bool(shadow_epoch),
        "full_exact_evaluation": bool(force_full_exact or screening_rows),
        "full_exact_audit": bool(force_full_exact),
        "surrogate_inference_runtime_seconds": float(
            surrogate_inference_runtime_seconds
        ),
        "surrogate_train_runtime_seconds": float(surrogate_train_runtime_seconds),
        "surrogate_training": dict(surrogate_train_metrics),
        "surrogate_delta_q_mae": delta_q_mae,
        "surrogate_delta_q_rmse": delta_q_rmse,
        "surrogate_rank_correlation": (
            float(rank_correlation) if rank_correlation is not None else float("nan")
        ),
        "surrogate_uncertainty_mean": uncertainty_mean,
        "surrogate_uncertainty_p90": uncertainty_p90,
        "gate_metrics": gate_metrics if screening_rows else None,
        "logical_branch_count": int(logical_branch_count),
        "dedup_reused_branch_count": int(dedup_reused_branch_count),
        "unique_full_branch_states": int(environment_executions),
        "mean_candidate_count": float(np.mean(candidate_counts)),
        "credit_runtime_seconds": float(time.perf_counter() - started_at),
        "advantage_normalization": "NONE",
        "advantage_centering": False,
        "ratio_granularity": "NODE_ONLY_AND_NODE_CONDITIONAL",
        "details": details,
    }


def _stable_masked_softmax(
    values: np.ndarray,
    mask: np.ndarray,
    temperature: float,
) -> np.ndarray:
    q_values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    scale = float(temperature)
    if q_values.shape != valid.shape or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Cache teacher Q/mask/temperature is invalid")
    probabilities = np.zeros_like(q_values, dtype=np.float64)
    if np.any(valid):
        legal_logits = q_values[valid] / scale
        if not np.all(np.isfinite(legal_logits)):
            raise FloatingPointError("Cache teacher received non-finite legal Q values")
        legal_logits -= float(np.max(legal_logits))
        weights = np.exp(legal_logits)
        probabilities[valid] = weights / float(np.sum(weights))
    return probabilities


def evaluate_cache_full_information_values(
    trainer: Any,
    rollout: Any,
    *,
    target_transition_indices: tuple[int, ...] | None = None,
    target_node_indices: tuple[int, ...] | None = None,
    trigger_temperature: float = 1.0,
    candidate_temperature: float = 1.0,
) -> dict[str, Any]:
    """Evaluate fixed-downstream full-exact Cache values and detached teachers.

    Each state/head shares this result between the binary Trigger and the
    conditional Candidate policy. Legal UPDATE branches use the existing
    production environment/Matching/KKT/reward executor and frozen critic
    bootstrap; identical complete physical branches are deduplicated.
    """

    trigger_scale = float(trigger_temperature)
    candidate_scale = float(candidate_temperature)
    if (
        not math.isfinite(trigger_scale)
        or trigger_scale <= 0.0
        or not math.isfinite(candidate_scale)
        or candidate_scale <= 0.0
    ):
        raise ValueError("Cache teacher temperatures must be positive and finite")
    transitions = tuple(rollout.transitions)
    if not transitions:
        raise ValueError("full-information Cache evaluation requires transitions")
    if target_transition_indices is None:
        selected_indices = tuple(range(len(transitions)))
    else:
        selected_indices = tuple(int(index) for index in target_transition_indices)
        if not selected_indices or len(set(selected_indices)) != len(selected_indices):
            raise ValueError("Cache evaluator transition indices must be non-empty and unique")
        if any(index < 0 or index >= len(transitions) for index in selected_indices):
            raise IndexError("Cache evaluator transition index is out of range")
    node_count = len(trainer.context.action_spec.cache_node_ids)
    if target_node_indices is None:
        node_indices = tuple(range(node_count))
    else:
        node_indices = tuple(int(index) for index in target_node_indices)
        if not node_indices or len(set(node_indices)) != len(node_indices):
            raise ValueError("Cache evaluator node indices must be non-empty and unique")
        if any(index < 0 or index >= node_count for index in node_indices):
            raise IndexError("Cache evaluator node index is out of range")
    if not bool(getattr(trainer.policy, "true_binary_trigger", False)):
        raise RuntimeError("full-information Cache teacher requires binary KEEP/UPDATE")
    if rollout.cache_candidate_features is None:
        raise ValueError("rollout does not expose Cache candidate tensor capacity")
    candidate_capacity = int(rollout.cache_candidate_features.shape[2])
    feature_dim = int(rollout.cache_candidate_features.shape[3])

    selected_transitions = tuple(transitions[index] for index in selected_indices)
    origin_start = min(int(item.origin_index) for item in selected_transitions)
    origin_end = max(int(item.origin_index) for item in selected_transitions)
    support_count = origin_end - origin_start + 1
    if not any(
        bool(item.done) and int(item.origin_index) == origin_end
        for item in selected_transitions
    ):
        support_count += 1
    from .trainer import aligned_train_steps

    aligned_steps = aligned_train_steps(
        trainer.context,
        origin_start,
        support_count,
        include_bootstrap_step=False,
    )
    steps_by_origin = {int(item.origin_index): item for item in aligned_steps}
    sample_count = len(selected_transitions)
    q_keep = np.full((sample_count, node_count), np.nan, dtype=np.float64)
    candidate_q = np.full(
        (sample_count, node_count, candidate_capacity), np.nan, dtype=np.float64
    )
    candidate_valid = np.zeros(
        (sample_count, node_count, candidate_capacity), dtype=bool
    )
    candidate_features = np.zeros(
        (sample_count, node_count, candidate_capacity, feature_dim), dtype=np.float32
    )
    candidate_operation_indices = np.zeros(
        (sample_count, node_count, candidate_capacity), dtype=np.int64
    )
    candidate_policy_probabilities = np.zeros_like(candidate_q)
    trigger_policy_probabilities = np.full(
        (sample_count, node_count, 2), np.nan, dtype=np.float64
    )
    candidate_counts = np.zeros((sample_count, node_count), dtype=np.int64)
    environment_executions = 0
    deduplicated_branches = 0
    logical_candidate_count = 0
    keep_branch_count = 0
    started_at = time.perf_counter()

    for sample_index, transition in enumerate(selected_transitions):
        step = steps_by_origin.get(int(transition.origin_index))
        if step is None:
            raise RuntimeError("full-information Cache evaluation lacks current aligned step")
        next_step = steps_by_origin.get(int(transition.origin_index) + 1)
        if not bool(transition.done) and next_step is None:
            raise RuntimeError("full-information Cache evaluation lacks next aligned step")
        branch_cache: dict[
            tuple[int, bytes, tuple[tuple[str, float, float], ...]], float
        ] = {}
        for node_index in node_indices:
            state = _node_policy_state(
                trainer,
                transition,
                step,
                node_index,
                feature_dim,
            )
            candidates = tuple(state["candidates"])
            count = len(candidates)
            if count > candidate_capacity:
                raise RuntimeError(
                    "legal Cache candidate count exceeds rollout capacity: "
                    f"count={count}, capacity={candidate_capacity}"
                )
            candidate_counts[sample_index, node_index] = count
            logical_candidate_count += count
            keep_branch_count += 1
            trigger_policy_probabilities[sample_index, node_index] = (
                float(state["p_keep"]),
                float(state["p_update"]),
            )
            if count:
                features = np.stack(state["candidate_features"]).astype(
                    np.float32, copy=False
                )
                candidate_features[sample_index, node_index, :count] = features
                candidate_operation_indices[sample_index, node_index, :count] = (
                    np.asarray(state["candidate_operations"], dtype=np.int64)
                )
                candidate_policy_probabilities[sample_index, node_index, :count] = (
                    np.asarray(state["candidate_probabilities"], dtype=np.float64)
                )
                candidate_valid[sample_index, node_index, :count] = True

            branch_values: list[float] = []
            for candidate in (None, *candidates):
                identity = _cache_branch_identity(
                    transition,
                    node_index,
                    state,
                    candidate,
                )
                q_value = branch_cache.get(identity)
                if q_value is None:
                    branch = _execute_cache_branch(
                        trainer,
                        transition,
                        step,
                        next_step,
                        node_index,
                        state,
                        candidate,
                    )
                    q_value = float(_branch_q_values(trainer, [branch])[0])
                    branch_cache[identity] = q_value
                    environment_executions += 1
                else:
                    deduplicated_branches += 1
                branch_values.append(q_value)
            q_keep[sample_index, node_index] = branch_values[0]
            if count:
                candidate_q[sample_index, node_index, :count] = np.asarray(
                    branch_values[1:], dtype=np.float64
                )

    q_best = np.full((sample_count, node_count), np.nan, dtype=np.float64)
    delta_best = np.full_like(q_best, np.nan)
    best_index = np.full((sample_count, node_count), -1, dtype=np.int64)
    trigger_teacher = np.zeros((sample_count, node_count, 2), dtype=np.float64)
    trigger_teacher[..., 0] = 1.0
    candidate_teacher = np.zeros_like(candidate_q)
    trigger_valid = candidate_valid.any(axis=-1)
    for sample_index in range(sample_count):
        for head_index in node_indices:
            legal = candidate_valid[sample_index, head_index]
            if not np.any(legal):
                continue
            legal_q = candidate_q[sample_index, head_index, legal]
            legal_indices = np.flatnonzero(legal)
            selected = int(legal_indices[int(np.argmax(legal_q))])
            best_index[sample_index, head_index] = selected
            q_best[sample_index, head_index] = float(
                candidate_q[sample_index, head_index, selected]
            )
            delta_best[sample_index, head_index] = (
                q_best[sample_index, head_index] - q_keep[sample_index, head_index]
            )
            trigger_teacher[sample_index, head_index] = _stable_masked_softmax(
                np.asarray(
                    [q_keep[sample_index, head_index], q_best[sample_index, head_index]],
                    dtype=np.float64,
                ),
                np.asarray([True, True], dtype=bool),
                trigger_scale,
            )
            candidate_teacher[sample_index, head_index] = _stable_masked_softmax(
                candidate_q[sample_index, head_index],
                legal,
                candidate_scale,
            )
    if not np.all(np.isfinite(q_keep[:, node_indices])):
        raise FloatingPointError("full-exact Cache KEEP values are non-finite")
    if not np.all(np.isfinite(candidate_q[candidate_valid])):
        raise FloatingPointError("full-exact Cache candidate values are non-finite")
    return {
        "q_keep": q_keep.astype(np.float32),
        "candidate_q_values": candidate_q.astype(np.float32),
        "candidate_valid_mask": candidate_valid,
        "best_candidate_index": best_index,
        "q_best_update": q_best.astype(np.float32),
        "delta_best": delta_best.astype(np.float32),
        "trigger_teacher_probabilities": trigger_teacher.astype(np.float32),
        "trigger_teacher_valid_mask": trigger_valid,
        "candidate_teacher_probabilities": candidate_teacher.astype(np.float32),
        "candidate_policy_probabilities": candidate_policy_probabilities.astype(np.float32),
        "trigger_policy_probabilities": trigger_policy_probabilities.astype(np.float32),
        "candidate_features": candidate_features,
        "candidate_operation_indices": candidate_operation_indices,
        "candidate_counts": candidate_counts,
        "candidate_exact_evaluated_mask": candidate_valid.copy(),
        "counterfactual_environment_executions": int(environment_executions),
        "logical_candidate_count": int(logical_candidate_count),
        "keep_branch_count": int(keep_branch_count),
        "deduplicated_branch_count": int(deduplicated_branches),
        "runtime_seconds": float(time.perf_counter() - started_at),
        "value_evaluator": "FULL_EXACT",
        "downstream_mode": "FIXED_DOWNSTREAM",
        "trigger_temperature": trigger_scale,
        "candidate_temperature": candidate_scale,
    }


def _audit_value_stats(values: list[float]) -> dict[str, float | None]:
    """Return the required compact regret/error distribution statistics."""

    if not values:
        return {
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise FloatingPointError("Cache Action-Q audit statistics are non-finite")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _action_q_audit_row(
    predicted: np.ndarray,
    exact: np.ndarray,
) -> dict[str, Any]:
    """Compare one raw predicted action-Q vector with one exact audit vector."""

    predicted_a = np.asarray(predicted, dtype=np.float64).reshape(-1)
    exact_a = np.asarray(exact, dtype=np.float64).reshape(-1)
    if len(predicted_a) != len(exact_a) or not len(predicted_a):
        raise ValueError("Cache Action-Q audit vectors must be non-empty and aligned")
    if not np.all(np.isfinite(predicted_a)) or not np.all(np.isfinite(exact_a)):
        raise FloatingPointError("Cache Action-Q audit vectors are non-finite")
    error = predicted_a - exact_a
    best_q = float(np.max(exact_a))
    best_mask = np.isclose(exact_a, best_q, rtol=1.0e-10, atol=1.0e-10)
    ranking = np.argsort(-predicted_a, kind="mergesort")
    predicted_best = int(ranking[0])
    regret = max(0.0, best_q - float(exact_a[predicted_best]))
    q_range = float(np.max(exact_a) - np.min(exact_a))
    normalized_regret = regret / (q_range + 1.0e-12)
    recall_at = {}
    for k in (3, 5):
        top = ranking[: min(k, len(ranking))]
        recall_at[str(k)] = float(np.any(best_mask[top]))
    rank = _rank_correlation(predicted_a, exact_a)
    return {
        "q_mae": float(np.mean(np.abs(error))),
        "q_rmse": float(np.sqrt(np.mean(error**2))),
        "rank_correlation": None if not np.isfinite(rank) else float(rank),
        "argmax_accuracy": float(best_mask[predicted_best]),
        "recall_at_3": recall_at["3"],
        "recall_at_5": recall_at["5"],
        "selected_exact_regret": float(regret),
        "normalized_selected_exact_regret": float(normalized_regret),
        "predicted_argmax": predicted_best,
        "exact_argmax": int(np.flatnonzero(best_mask)[0]),
        "action_count": int(len(predicted_a)),
    }


def _aggregate_action_q_audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-node Action-Q audit rows without hiding per-slot rows."""

    metric_names = (
        "q_mae",
        "q_rmse",
        "rank_correlation",
        "argmax_accuracy",
        "recall_at_3",
        "recall_at_5",
        "selected_exact_regret",
        "normalized_selected_exact_regret",
    )
    aggregate: dict[str, Any] = {
        "row_count": int(len(rows)),
        "action_count": int(sum(int(row["action_count"]) for row in rows)),
    }
    for name in metric_names:
        values = [
            float(row[name])
            for row in rows
            if row.get(name) is not None and math.isfinite(float(row[name]))
        ]
        if name in {"q_mae", "q_rmse", "selected_exact_regret", "normalized_selected_exact_regret"}:
            aggregate[name + "_stats"] = _audit_value_stats(values)
        else:
            aggregate[name] = None if not values else float(np.mean(np.asarray(values)))
    aggregate["normalized_regret_p95"] = aggregate[
        "normalized_selected_exact_regret_stats"
    ]["p95"]
    return aggregate


def _action_q_usable_status(audit_metrics: dict[str, Any] | None) -> str:
    """Classify evidence without imposing the old surrogate's hard gate."""

    if not audit_metrics or int(audit_metrics.get("row_count", 0)) < 3:
        return "INCONCLUSIVE"
    rank = audit_metrics.get("rank_correlation")
    recall3 = audit_metrics.get("recall_at_3")
    recall5 = audit_metrics.get("recall_at_5")
    regret = audit_metrics.get("normalized_regret_p95")
    if any(value is None or not math.isfinite(float(value)) for value in (rank, recall3, recall5, regret)):
        return "INCONCLUSIVE"
    # This is an evidence label, not a pre-declared policy-performance gate:
    # positive ordering, non-zero top-k coverage, and a non-worst-case p95
    # regret jointly support usability; mixed evidence remains inconclusive.
    positive = float(rank) > 0.0 and float(recall3) > 0.0 and float(recall5) >= float(recall3) and float(regret) < 1.0
    negative = float(rank) < 0.0 and float(recall5) == 0.0 and float(regret) >= 1.0
    if positive:
        return "YES"
    if negative:
        return "NO"
    return "INCONCLUSIVE"


def compute_cache_action_q_credit(
    trainer: Any,
    rollout: Any,
    *,
    audit_transition_indices: tuple[int, ...] | None = None,
    heldout_audit_transition_indices: tuple[int, ...] | None = None,
    gate_mode: bool = False,
) -> dict[str, Any]:
    """Compute sparse Action-Q Cache credit with exact audit supervision.

    Normal slots enumerate legal actions and use the independent Action-Q
    critic.  The production action is anchored with its real rollout TD-Q;
    no normal candidate enters ``_execute_cache_branch``.  Audit slots reuse
    ``compute_exact_node_specific_cache_credit(..., force_full_exact=True)``
    and are labelled only after held-out predictions have been measured.
    """

    all_transitions = tuple(rollout.transitions)
    if not all_transitions:
        raise ValueError("Cache Action-Q credit requires a non-empty rollout")
    from .primitive_control import PrimitiveTensorSpec
    from .trainer import aligned_train_steps

    transition_count = len(all_transitions)
    node_count = len(trainer.context.action_spec.cache_node_ids)
    if rollout.cache_candidate_features is None:
        raise ValueError("Rollout does not expose Cache candidate tensor capacity")
    candidate_capacity = int(rollout.cache_candidate_features.shape[2])
    feature_dim = int(rollout.cache_candidate_features.shape[3])
    primitive_spec = PrimitiveTensorSpec.create(trainer.context)
    if feature_dim != int(primitive_spec.cache_feature_dim):
        raise RuntimeError("Rollout Cache feature width does not match tensor spec")

    if audit_transition_indices is None:
        audit_indices = tuple(
            index
            for index, transition in enumerate(all_transitions)
            if (
                int(transition.origin_index) - int(trainer.context.train_origin_start)
            )
            % int(getattr(trainer, "cache_action_q_audit_interval", 20))
            == 0
        )
    else:
        audit_indices = tuple(int(index) for index in audit_transition_indices)
    if len(set(audit_indices)) != len(audit_indices) or any(
        index < 0 or index >= transition_count for index in audit_indices
    ):
        raise ValueError("Cache Action-Q audit transition indices are invalid")
    audit_set = set(audit_indices)
    if heldout_audit_transition_indices is None:
        heldout_set = set(audit_indices if gate_mode else ())
    else:
        heldout_set = set(int(index) for index in heldout_audit_transition_indices)
    if not heldout_set.issubset(audit_set):
        raise ValueError("heldout Action-Q audits must be a subset of audit slots")

    origin_start = min(int(item.origin_index) for item in all_transitions)
    origin_end = max(int(item.origin_index) for item in all_transitions)
    support_step_count = origin_end - origin_start + 1
    if not any(
        bool(item.done) and int(item.origin_index) == origin_end
        for item in all_transitions
    ):
        support_step_count += 1
    aligned_steps = aligned_train_steps(
        trainer.context,
        origin_start,
        support_step_count,
    )
    steps_by_origin = {int(step.origin_index): step for step in aligned_steps}

    started_at = time.perf_counter()
    q_prod = np.zeros(transition_count, dtype=np.float64)
    next_observations: list[np.ndarray | None] = []
    for transition in all_transitions:
        if bool(transition.done):
            next_observations.append(None)
            continue
        next_step = steps_by_origin.get(int(transition.origin_index) + 1)
        if next_step is None:
            raise RuntimeError("Cache Action-Q production Q is missing next aligned step")
        next_observations.append(
            np.asarray(
                _intermediate_observation(
                    np.asarray(transition.next_cache_state, dtype=np.int8),
                    transition.next_uav_positions,
                    next_step,
                    trainer.context,
                ),
                dtype=np.float32,
            )
        )
    bootstrap_observations = [value for value in next_observations if value is not None]
    next_values_by_transition: dict[int, float] = {}
    if bootstrap_observations:
        with torch.no_grad():
            tensor = torch.as_tensor(
                np.stack(bootstrap_observations),
                dtype=torch.float32,
                device=trainer.device,
            )
            values = trainer.policy.critic_value(tensor).detach().cpu().numpy().reshape(-1)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("Cache production TD-Q bootstrap is non-finite")
        value_index = 0
        for index, observation in enumerate(next_observations):
            if observation is not None:
                next_values_by_transition[index] = float(values[value_index])
                value_index += 1
    for index, transition in enumerate(all_transitions):
        q_prod[index] = float(transition.reward) + (
            0.0
            if bool(transition.done)
            else float(trainer.hyperparameters.gamma) * next_values_by_transition[index]
        )
    if not np.all(np.isfinite(q_prod)):
        raise FloatingPointError("Cache production TD-Q targets are non-finite")

    trigger_advantages = np.zeros((transition_count, node_count), dtype=np.float64)
    trigger_old_probabilities = np.zeros((transition_count, node_count, 2), dtype=np.float64)
    q_keep = np.zeros((transition_count, node_count), dtype=np.float64)
    q_update = np.zeros((transition_count, node_count), dtype=np.float64)
    baseline = np.zeros((transition_count, node_count), dtype=np.float64)
    candidate_q_values = np.zeros(
        (transition_count, node_count, candidate_capacity), dtype=np.float64
    )
    candidate_exact_q_values = np.full_like(candidate_q_values, np.nan)
    candidate_surrogate_q_values = np.full_like(candidate_q_values, np.nan)
    candidate_predicted_delta_q = np.zeros_like(candidate_q_values)
    candidate_predicted_sigma = np.zeros_like(candidate_q_values)
    candidate_probabilities = np.zeros_like(candidate_q_values)
    candidate_valid_mask = np.zeros(
        (transition_count, node_count, candidate_capacity), dtype=bool
    )
    candidate_exact_evaluated_mask = np.zeros_like(candidate_valid_mask)
    candidate_operation_indices = np.zeros(
        (transition_count, node_count, candidate_capacity), dtype=np.int64
    )
    candidate_proposal_sources = np.full(
        (transition_count, node_count, candidate_capacity), "", dtype=object
    )
    candidate_q_sources = np.full(
        (transition_count, node_count, candidate_capacity), "", dtype=object
    )
    candidate_hamming_distances = np.full_like(candidate_q_values, np.nan)
    candidate_features = np.zeros(
        (transition_count, node_count, candidate_capacity, feature_dim), dtype=np.float32
    )
    candidate_q_update = np.zeros((transition_count, node_count), dtype=np.float64)
    candidate_weighted_mean_residual = np.zeros((transition_count, node_count), dtype=np.float64)
    trigger_weighted_mean_residual = np.zeros((transition_count, node_count), dtype=np.float64)
    candidate_counts = np.zeros((transition_count, node_count), dtype=np.int64)
    details: list[dict[str, Any]] = []
    row_states: dict[tuple[int, int], dict[str, Any]] = {}
    q_inference_started_at = time.perf_counter()
    for transition_index, transition in enumerate(all_transitions):
        step = steps_by_origin.get(int(transition.origin_index))
        if step is None:
            raise RuntimeError("Cache Action-Q credit is missing the aligned current step")
        for node_index in range(node_count):
            state = _node_policy_state(
                trainer,
                transition,
                step,
                node_index,
                feature_dim,
            )
            candidates = list(state["candidates"])
            candidate_count = len(candidates)
            if candidate_count > candidate_capacity:
                raise RuntimeError(
                    "Legal Cache candidate count exceeds rollout tensor capacity: "
                    f"count={candidate_count}, capacity={candidate_capacity}"
                )
            feature_matrix = (
                np.stack(state["candidate_features"]).astype(np.float32)
                if candidate_count
                else np.zeros((0, feature_dim), dtype=np.float32)
            )
            context_array = np.asarray(state["conditional_context"], dtype=np.float32)
            with torch.no_grad():
                context_tensor = torch.as_tensor(
                    context_array[None, :], dtype=torch.float32, device=trainer.device
                )
                v_context = float(
                    trainer.policy.critic_value(context_tensor).detach().cpu().numpy().reshape(-1)[0]
                )
            actions = 1 + candidate_count
            action_features = np.zeros((actions, feature_dim), dtype=np.float32)
            if candidate_count:
                action_features[1:] = feature_matrix
            action_contexts = np.repeat(context_array[None, :], actions, axis=0)
            node_identity = np.zeros((actions, node_count), dtype=np.float32)
            node_identity[:, node_index] = 1.0
            update_flags = np.zeros((actions, 1), dtype=np.float32)
            if candidate_count:
                update_flags[1:, 0] = 1.0
            residuals = trainer._predict_cache_action_q(
                action_contexts,
                node_identity,
                update_flags,
                action_features,
            )
            predicted_all_q = v_context + np.asarray(residuals, dtype=np.float64)
            if not np.all(np.isfinite(predicted_all_q)):
                raise FloatingPointError("Cache Action-Q prediction is non-finite")
            row_states[(transition_index, node_index)] = {
                "state": state,
                "candidates": candidates,
                "feature_matrix": feature_matrix,
                "v_context": v_context,
                "predicted_all_q": predicted_all_q,
            }
    q_inference_seconds = float(time.perf_counter() - q_inference_started_at)

    audit_credit: dict[str, Any] | None = None
    audit_local_index: dict[int, int] = {}
    audit_exact_seconds = 0.0
    if audit_indices:
        audit_started_at = time.perf_counter()
        previous_sparse_mode = getattr(trainer, "cache_action_q_sparse_mode", False)
        trainer.cache_action_q_sparse_mode = True
        try:
            audit_credit = compute_exact_node_specific_cache_credit(
                trainer,
                rollout,
                target_transition_indices=tuple(sorted(audit_indices)),
                force_full_exact=True,
            )
        finally:
            trainer.cache_action_q_sparse_mode = previous_sparse_mode
        audit_exact_seconds = float(time.perf_counter() - audit_started_at)
        audit_local_index = {
            int(transition_index): local_index
            for local_index, transition_index in enumerate(sorted(audit_indices))
        }

    audit_rows: list[dict[str, Any]] = []
    per_slot_audit_rows: dict[int, list[dict[str, Any]]] = {}
    if audit_credit is not None:
        for transition_index in sorted(audit_indices):
            local_index = audit_local_index[transition_index]
            transition = all_transitions[transition_index]
            for node_index in range(node_count):
                row_state = row_states[(transition_index, node_index)]
                candidate_count = len(row_state["candidates"])
                exact_candidates = np.asarray(
                    audit_credit["candidate_q_values"][local_index, node_index, :candidate_count],
                    dtype=np.float64,
                )
                exact_all_q = np.concatenate(
                    [
                        np.asarray(
                            [audit_credit["q_keep"][local_index, node_index]],
                            dtype=np.float64,
                        ),
                        exact_candidates,
                    ]
                )
                metric = _action_q_audit_row(
                    row_state["predicted_all_q"],
                    exact_all_q,
                )
                metric.update(
                    {
                        "transition_index": int(transition_index),
                        "origin_index": int(transition.origin_index),
                        "slot": int(transition.slot),
                        "node_index": int(node_index),
                        "node_id": str(trainer.context.action_spec.cache_node_ids[node_index]),
                        "candidate_count": int(candidate_count),
                    }
                )
                audit_rows.append(metric)
                per_slot_audit_rows.setdefault(int(transition_index), []).append(metric)
    heldout_rows = [row for row in audit_rows if int(row["transition_index"]) in heldout_set]
    warmup_rows = [row for row in audit_rows if int(row["transition_index"]) not in heldout_set]
    heldout_metrics = _aggregate_action_q_audit_rows(heldout_rows) if heldout_rows else None
    audit_metrics = _aggregate_action_q_audit_rows(audit_rows) if audit_rows else None
    slot_metrics = {
        str(index): _aggregate_action_q_audit_rows(rows)
        for index, rows in sorted(per_slot_audit_rows.items())
    }

    train_contexts: list[np.ndarray] = []
    train_nodes: list[np.ndarray] = []
    train_updates: list[float] = []
    train_features: list[np.ndarray] = []
    train_targets: list[float] = []

    def append_label(
        context_array: np.ndarray,
        node_index: int,
        is_update: float,
        feature: np.ndarray,
        target: float,
    ) -> None:
        train_contexts.append(np.asarray(context_array, dtype=np.float32))
        identity = np.zeros(node_count, dtype=np.float32)
        identity[int(node_index)] = 1.0
        train_nodes.append(identity)
        train_updates.append(float(is_update))
        train_features.append(np.asarray(feature, dtype=np.float32))
        train_targets.append(float(target))

    logical_branch_count = 0
    full_exact_audit_branch_count = 0
    logical_candidate_count = 0
    exact_candidate_count = 0
    keep_exact_count = 0
    normal_logical_branch_count = 0
    selected_anchor_count = 0
    for transition_index, transition in enumerate(all_transitions):
        is_audit = transition_index in audit_set
        for node_index in range(node_count):
            row_state = row_states[(transition_index, node_index)]
            state = row_state["state"]
            candidates = row_state["candidates"]
            feature_matrix = row_state["feature_matrix"]
            candidate_count = len(candidates)
            logical_branches = 1 + candidate_count
            logical_branch_count += logical_branches
            logical_candidate_count += candidate_count
            candidate_counts[transition_index, node_index] = candidate_count
            if is_audit:
                full_exact_audit_branch_count += logical_branches
                exact_candidate_count += candidate_count
                keep_exact_count += 1
            else:
                normal_logical_branch_count += logical_branches
            if candidate_count:
                candidate_features[transition_index, node_index, :candidate_count] = feature_matrix
                candidate_probabilities[transition_index, node_index, :candidate_count] = state[
                    "candidate_probabilities"
                ]
                candidate_valid_mask[transition_index, node_index, :candidate_count] = True
                candidate_operation_indices[transition_index, node_index, :candidate_count] = state[
                    "candidate_operations"
                ]
                candidate_proposal_sources[transition_index, node_index, :candidate_count] = [
                    str(candidate.proposal_source) for candidate in candidates
                ]
                candidate_hamming_distances[transition_index, node_index, :candidate_count] = [
                    float(
                        np.count_nonzero(
                            np.asarray(candidate.target_node_cache, dtype=np.int8)
                            != np.asarray(candidate.before_node_cache, dtype=np.int8)
                        )
                    )
                    for candidate in candidates
                ]
                candidate_predicted_delta_q[transition_index, node_index, :candidate_count] = (
                    row_state["predicted_all_q"][1:] - row_state["v_context"]
                )
                candidate_surrogate_q_values[transition_index, node_index, :candidate_count] = (
                    row_state["predicted_all_q"][1:]
                )

            if is_audit:
                if audit_credit is None:
                    raise RuntimeError("Audit row is missing exact credit")
                local_index = audit_local_index[transition_index]
                exact_keep = float(audit_credit["q_keep"][local_index, node_index])
                exact_candidates = np.asarray(
                    audit_credit["candidate_q_values"][local_index, node_index, :candidate_count],
                    dtype=np.float64,
                )
                action_q = np.concatenate(([exact_keep], exact_candidates))
                candidate_exact_q_values[transition_index, node_index, :candidate_count] = exact_candidates
                candidate_exact_evaluated_mask[transition_index, node_index, :candidate_count] = True
                source = "FULL_EXACT_AUDIT"
            else:
                action_q = np.asarray(row_state["predicted_all_q"], dtype=np.float64).copy()
                source = "ACTION_CONDITIONED_Q_CRITIC_WITH_ACTUAL_ACTION_ANCHOR"

            selected_update = (
                transition.cache_operation_actions is not None
                and int(transition.cache_operation_actions[node_index]) != 0
            )
            selected_index = _selected_cache_candidate_index(
                transition,
                node_index,
                candidates,
            )
            if selected_update and selected_index is None:
                raise RuntimeError(
                    "Production Cache UPDATE action cannot be resolved to a legal candidate: "
                    f"transition={transition_index}, node={node_index}"
                )
            actual_action_index = (1 + int(selected_index)) if selected_update else 0
            if not is_audit:
                action_q[actual_action_index] = q_prod[transition_index]
                selected_anchor_count += 1
            selected_action_feature = (
                np.zeros(feature_dim, dtype=np.float32)
                if not selected_update
                else np.asarray(feature_matrix[int(selected_index)], dtype=np.float32)
            )
            append_label(
                row_state["state"]["conditional_context"],
                node_index,
                1.0 if selected_update else 0.0,
                selected_action_feature,
                q_prod[transition_index] - row_state["v_context"],
            )
            if is_audit:
                append_label(
                    row_state["state"]["conditional_context"],
                    node_index,
                    0.0,
                    np.zeros(feature_dim, dtype=np.float32),
                    action_q[0] - row_state["v_context"],
                )
                for candidate_index in range(candidate_count):
                    append_label(
                        row_state["state"]["conditional_context"],
                        node_index,
                        1.0,
                        feature_matrix[candidate_index],
                        action_q[candidate_index + 1] - row_state["v_context"],
                    )

            keep_q_value = float(action_q[0])
            candidate_q = np.asarray(action_q[1:], dtype=np.float64)
            update_q_value = (
                float(np.dot(state["candidate_probabilities"], candidate_q))
                if candidate_count
                else keep_q_value
            )
            state_baseline = float(
                state["p_keep"] * keep_q_value + state["p_update"] * update_q_value
            )
            trigger_advantages[transition_index, node_index] = (
                update_q_value - state_baseline if selected_update else keep_q_value - state_baseline
            )
            trigger_old_probabilities[transition_index, node_index] = (
                state["p_keep"],
                state["p_update"],
            )
            q_keep[transition_index, node_index] = keep_q_value
            q_update[transition_index, node_index] = update_q_value
            baseline[transition_index, node_index] = state_baseline
            candidate_q_update[transition_index, node_index] = update_q_value
            trigger_weighted_mean_residual[transition_index, node_index] = float(
                state["p_keep"] * (keep_q_value - state_baseline)
                + state["p_update"] * (update_q_value - state_baseline)
            )
            if candidate_count:
                candidate_q_values[transition_index, node_index, :candidate_count] = candidate_q
                candidate_q_sources[transition_index, node_index, :candidate_count] = source
                candidate_weighted_mean_residual[transition_index, node_index] = float(
                    np.dot(state["candidate_probabilities"], candidate_q - update_q_value)
                )
            if is_audit:
                candidate_q_source = "FULL_EXACT_AUDIT"
            else:
                candidate_q_source = "ACTION_CONDITIONED_Q_CRITIC_WITH_ACTUAL_ACTION_ANCHOR"
            if gate_mode or details:
                details.append(
                    {
                        "transition_index": int(transition_index),
                        "origin_index": int(transition.origin_index),
                        "slot": int(transition.slot),
                        "node_index": int(node_index),
                        "node_id": str(trainer.context.action_spec.cache_node_ids[node_index]),
                        "candidate_count": int(candidate_count),
                        "q_keep": keep_q_value,
                        "q_update": update_q_value,
                        "baseline": state_baseline,
                        "source": candidate_q_source,
                    }
                )

    train_started_at = time.perf_counter()
    if train_targets:
        train_metrics = trainer._train_cache_action_q(
            np.stack(train_contexts),
            np.stack(train_nodes),
            np.asarray(train_updates, dtype=np.float32)[:, None],
            np.stack(train_features),
            np.asarray(train_targets, dtype=np.float32),
        )
    else:
        train_metrics = {"count": 0, "loss": math.nan, "mae": math.nan, "rmse": math.nan}
    train_seconds = float(time.perf_counter() - train_started_at)

    if not np.all(np.isfinite(trigger_advantages)) or not np.all(np.isfinite(candidate_q_values[candidate_valid_mask])):
        raise FloatingPointError("Cache Action-Q credit contains non-finite values")
    candidate_advantages = np.zeros_like(candidate_q_values)
    candidate_advantages[candidate_valid_mask] = (
        candidate_q_values[candidate_valid_mask]
        - np.repeat(candidate_q_update[..., None], candidate_capacity, axis=-1)[candidate_valid_mask]
    )
    node_roles = np.asarray(
        [str(value).upper() for value in trainer.context.cache_nodes["node_roles"]],
        dtype=object,
    )
    trigger_mean_abs_by_role = {}
    for role in ("SV", "RSU", "UAV"):
        role_mask = node_roles == role
        trigger_mean_abs_by_role[role] = (
            float(np.mean(np.abs(trigger_advantages[:, role_mask])))
            if np.any(role_mask)
            else float("nan")
        )
    total_environment_executions = int(
        0 if audit_credit is None else audit_credit.get("counterfactual_environment_executions", 0)
    )
    source = (
        "FULL_EXACT_AUDIT"
        if audit_indices and not (set(range(transition_count)) - audit_set)
        else (
            "ACTION_CONDITIONED_Q_CRITIC_WITH_ACTUAL_ACTION_ANCHOR_AND_SPARSE_FULL_EXACT_AUDIT"
            if audit_indices
            else "ACTION_CONDITIONED_Q_CRITIC_WITH_ACTUAL_ACTION_ANCHOR"
        )
    )
    avoided = max(logical_branch_count - total_environment_executions, 0)
    audit_before_training = bool(audit_rows) and not bool(train_metrics.get("count", 0) == 0)
    return {
        "trigger_advantages": trigger_advantages.astype(np.float32),
        "trigger_old_probabilities": trigger_old_probabilities.astype(np.float32),
        "q_keep": q_keep.astype(np.float32),
        "q_update": q_update.astype(np.float32),
        "baseline": baseline.astype(np.float32),
        "candidate_q_values": candidate_q_values.astype(np.float32),
        "candidate_exact_q_values": candidate_exact_q_values.astype(np.float32),
        "candidate_surrogate_q_values": candidate_surrogate_q_values.astype(np.float32),
        "candidate_predicted_delta_q": candidate_predicted_delta_q.astype(np.float32),
        "candidate_predicted_sigma": candidate_predicted_sigma.astype(np.float32),
        "candidate_probabilities": candidate_probabilities.astype(np.float32),
        "candidate_valid_mask": candidate_valid_mask,
        "candidate_exact_evaluated_mask": candidate_exact_evaluated_mask,
        "candidate_operation_indices": candidate_operation_indices,
        "candidate_proposal_sources": candidate_proposal_sources,
        "candidate_q_sources": candidate_q_sources,
        "candidate_hamming_distances": candidate_hamming_distances,
        "candidate_features": candidate_features,
        "candidate_q_update": candidate_q_update.astype(np.float32),
        "candidate_weighted_mean_residual": candidate_weighted_mean_residual.astype(np.float32),
        "trigger_weighted_mean_residual": trigger_weighted_mean_residual.astype(np.float32),
        "trigger_advantage_mean_abs": float(np.mean(np.abs(trigger_advantages))),
        "candidate_advantage_mean_abs": (
            float(np.mean(np.abs(candidate_advantages[candidate_valid_mask])))
            if np.any(candidate_valid_mask)
            else 0.0
        ),
        "trigger_advantage_mean_abs_by_role": trigger_mean_abs_by_role,
        "candidate_counts": candidate_counts,
        "counterfactual_environment_executions": total_environment_executions,
        "logical_candidate_count": int(logical_candidate_count),
        "exact_candidate_count": int(exact_candidate_count),
        "surrogate_candidate_count": 0,
        "action_q_candidate_count": int(logical_candidate_count - exact_candidate_count),
        "keep_exact_count": int(keep_exact_count),
        "full_exact_audit_count": int(len(audit_rows)),
        "full_exact_audit_branch_count": int(full_exact_audit_branch_count),
        "normal_slot_counterfactual_branch_count": 0,
        "total_logical_branch_count": int(logical_branch_count),
        "normal_logical_branch_count": int(normal_logical_branch_count),
        "counterfactual_executions_saved": int(avoided),
        "counterfactual_executions_avoided": int(avoided),
        "counterfactual_execution_reduction_fraction": float(
            avoided / float(max(logical_branch_count, 1))
        ),
        "surrogate_candidate_source": source,
        "candidate_value_source": source,
        "cache_candidate_value_source": source,
        "actual_action_q_anchor": "YES",
        "actual_action_anchor_count": int(selected_anchor_count),
        "cache_q_critic_usable_for_policy_credit": _action_q_usable_status(heldout_metrics),
        "cache_q_critic_inference_seconds": float(q_inference_seconds),
        "cache_q_critic_train_seconds": float(train_seconds),
        "cache_audit_exact_seconds": float(audit_exact_seconds),
        "cache_q_critic_training": dict(train_metrics),
        "audit_metrics": audit_metrics,
        "heldout_audit_metrics": heldout_metrics,
        "heldout_audit_slot_metrics": {
            str(index): slot_metrics[str(index)]
            for index in sorted(heldout_set)
            if str(index) in slot_metrics
        },
        "audit_slot_metrics": slot_metrics,
        "audit_transition_indices": [int(index) for index in sorted(audit_set)],
        "heldout_audit_transition_indices": [int(index) for index in sorted(heldout_set)],
        "heldout_audit_before_training": bool(audit_before_training),
        "full_exact_evaluation": bool(audit_indices),
        "gate_metrics": heldout_metrics,
        "logical_branch_count": int(logical_branch_count),
        "dedup_reused_branch_count": int(
            0 if audit_credit is None else audit_credit.get("dedup_reused_branch_count", 0)
        ),
        "unique_full_branch_states": int(
            0 if audit_credit is None else audit_credit.get("unique_full_branch_states", 0)
        ),
        "mean_candidate_count": float(np.mean(candidate_counts)),
        "credit_runtime_seconds": float(time.perf_counter() - started_at),
        "advantage_normalization": "NONE",
        "advantage_centering": False,
        "ratio_granularity": "NODE_ONLY_AND_NODE_CONDITIONAL",
        "details": details,
        "production_q_targets": q_prod.astype(np.float32),
    }
