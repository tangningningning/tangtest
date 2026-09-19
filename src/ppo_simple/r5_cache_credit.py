"""Exact node-specific Cache counterfactual credit for Recovery R5.

The helper deliberately owns no optimizer, reward, Matching, KKT, or system
model logic.  It replays the production primitive transition with one Cache
node replaced and uses the existing global critic only for the next-state
bootstrap.  The operation head remains the production four-way
``NO_CHANGE/ADD/DELETE/SWAP`` head; R5's binary trigger view is the exact
aggregation ``KEEP=NO_CHANGE`` and ``UPDATE={ADD,DELETE,SWAP}``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from decision import cache_configuration, slot_environment

from .primitive_control import (
    _cache_candidate_feature,
    _intermediate_observation,
)
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
            "R5 production Cache action record is not one entry per node: "
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
        if str(action_map[index].get("operation_type")) == "NO_CHANGE" and not np.array_equal(
            working[index], previous_cache[index]
        ):
            raise RuntimeError("production NO_CHANGE changed a Cache row")
    return working, previous_cache


def _masked_operation_probabilities(
    policy: Any,
    conditional_context: np.ndarray,
    node_index: int,
    operation_mask: np.ndarray,
) -> np.ndarray:
    logits, single = policy._operation_logits(conditional_context)
    if not single:
        raise RuntimeError("R5 operation distribution requires one context")
    raw_logits = logits[int(node_index)]
    mask = torch.as_tensor(operation_mask, dtype=torch.bool, device=raw_logits.device)
    distribution = torch.distributions.Categorical(
        logits=raw_logits.masked_fill(~mask, -float("inf"))
    )
    probabilities = distribution.probs.detach().cpu().numpy().astype(np.float64)
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("R5 Cache operation probabilities are invalid")
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
        raise RuntimeError("R5 candidate distribution requires one context")
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
    working_cache, previous_cache = _replay_cache_prefix(transition, node_index)
    current_uav = _position_dict(transition.previous_uav_positions)
    conditional_context = _intermediate_observation(
        working_cache,
        current_uav,
        step,
        context,
    )
    all_candidates = cache_configuration.enumerate_cache_primitive_actions(
        working_cache,
        node_index,
        context.service_catalog,
        context.cache_nodes,
    )
    if bool(getattr(trainer.policy, "true_binary_trigger", False)):
        # R6 keeps the exact production enumerator but removes operation-type
        # multiplicity from the high-level trigger.  ADD/DELETE/SWAP remain
        # separate candidate metadata/features in one UPDATE conditional set.
        update_candidates = [
            candidate
            for candidate in all_candidates
            if str(candidate.operation_type) != "NO_CHANGE"
        ]
        action_map = _production_action_map(transition)
        sequential_triggers = np.zeros(
            len(context.action_spec.cache_node_ids), dtype=np.int8
        )
        for index in range(node_index):
            if str(action_map[index].get("operation_type")) != "NO_CHANGE":
                sequential_triggers[index] = 1
        sequential_triggers[node_index] = 1
        if update_candidates:
            unified_features = np.stack(
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
                [
                    OPERATION_ORDER.index(str(candidate.operation_type))
                    for candidate in update_candidates
                ],
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
            raise RuntimeError("R5 UPDATE probability is zero despite legal candidates")
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
                    "operation_type": "NO_CHANGE",
                    "added_service_ids": [],
                    "removed_service_ids": [],
                    "actual_changed": False,
                }
            else:
                branch_triggers[index] = 1
                final_cache[index] = np.asarray(
                    candidate.after_node_cache, dtype=np.int8
                ).copy()
                action = {
                    "head_type": "cache",
                    "head_id": str(node_id),
                    "head_index": int(index),
                    "candidate_count": -1,
                    "selected_index": None,
                    "operation_type": str(candidate.operation_type),
                    "added_service_ids": list(candidate.added_service_ids),
                    "removed_service_ids": list(candidate.removed_service_ids),
                    "actual_changed": bool(
                        not np.array_equal(
                            candidate.before_node_cache,
                            candidate.after_node_cache,
                        )
                    ),
                }
        branch_actions.append(action)
    branch_actions.sort(key=lambda item: int(item["head_index"]))
    production_uav_triggers = {
        str(key): int(value) for key, value in production_uav_triggers.items()
    }
    configuration_override = slot_environment.build_policy_primitive_configuration(
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
            "R5 exact Cache counterfactual is infeasible: "
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
            candidate.after_node_cache, dtype=np.int8
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
            raise RuntimeError("R5 branch next observations are partially missing")
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
        raise FloatingPointError("R5 exact Cache Q values are non-finite")
    return q_values


def compute_exact_node_specific_cache_credit(
    trainer: Any,
    rollout: Any,
    *,
    target_transition_indices: tuple[int, ...] | None = None,
    target_node_indices: tuple[int, ...] | None = None,
    include_details: bool = False,
    reuse_identical_branches: bool = True,
) -> dict[str, Any]:
    """Replay exact KEEP/update branches for every Cache node-time entry."""

    all_transitions = tuple(rollout.transitions)
    if not all_transitions:
        raise ValueError("R5 Cache credit requires a non-empty rollout")
    if target_transition_indices is None:
        selected_indices = tuple(range(len(all_transitions)))
    else:
        selected_indices = tuple(int(index) for index in target_transition_indices)
        if not selected_indices or len(set(selected_indices)) != len(selected_indices):
            raise ValueError("R5 target transition indices must be non-empty and unique")
        if any(index < 0 or index >= len(all_transitions) for index in selected_indices):
            raise IndexError("R5 target transition index is out of range")
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
            raise ValueError("R5 target node indices must be non-empty and unique")
        if any(index < 0 or index >= node_count for index in node_indices):
            raise IndexError("R5 target node index is out of range")
    if rollout.cache_candidate_features is None:
        raise ValueError("R5 rollout does not expose Cache candidate tensor capacity")
    candidate_capacity = int(rollout.cache_candidate_features.shape[2])
    feature_dim = int(rollout.cache_candidate_features.shape[3])
    primitive_spec = PrimitiveTensorSpec.create(trainer.context)
    if int(primitive_spec.cache_feature_dim) != feature_dim:
        raise RuntimeError(
            "R5 rollout Cache feature width does not match the production tensor spec"
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
    candidate_probabilities = np.zeros_like(candidate_q_values)
    candidate_valid_mask = np.zeros(
        (sample_count, node_count, candidate_capacity), dtype=bool
    )
    candidate_operation_indices = np.zeros(
        (sample_count, node_count, candidate_capacity), dtype=np.int64
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
    branch_cache: dict[
        tuple[int, bytes, tuple[tuple[str, float, float], ...]], dict[str, Any]
    ] = {}
    started_at = time.perf_counter()

    for sample_index, transition in enumerate(transitions):
        # Reuse is valid only within one fixed transition.  Clearing here
        # avoids retaining large Matching/KKT result dictionaries across the
        # 300-slot rollout while preserving the complete-state identity rule.
        branch_cache.clear()
        step = steps_by_origin.get(int(transition.origin_index))
        if step is None:
            raise RuntimeError("R5 replay is missing the aligned current step")
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
                    "R5 legal Cache candidate count exceeds rollout tensor capacity: "
                    f"count={len(candidates)}, capacity={candidate_capacity}"
                )
            branches: list[dict[str, Any]] = []
            for candidate in (None, *candidates):
                logical_branch_count += 1
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
            candidate_q = q_values[1:]
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
            selected_update = int(transition.v7_cache_operation_actions[node_index]) != 0
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
            candidate_count = len(candidates)
            candidate_counts[sample_index, node_index] = candidate_count
            if candidate_count:
                candidate_q_values[sample_index, node_index, :candidate_count] = candidate_q
                candidate_probabilities[
                    sample_index, node_index, :candidate_count
                ] = state["candidate_probabilities"]
                candidate_valid_mask[
                    sample_index, node_index, :candidate_count
                ] = True
                candidate_operation_indices[
                    sample_index, node_index, :candidate_count
                ] = state["candidate_operations"]
                candidate_features[
                    sample_index, node_index, :candidate_count
                ] = np.stack(state["candidate_features"])
                candidate_weighted_mean_residual[sample_index, node_index] = float(
                    np.dot(
                        state["candidate_probabilities"],
                        candidate_q - update_q,
                    )
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
    if not np.all(np.isfinite(trigger_advantages)) or not np.all(
        np.isfinite(candidate_q_values)
    ):
        raise FloatingPointError("R5 Cache credit contains non-finite values")
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
        "candidate_probabilities": candidate_probabilities.astype(np.float32),
        "candidate_valid_mask": candidate_valid_mask,
        "candidate_operation_indices": candidate_operation_indices,
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
