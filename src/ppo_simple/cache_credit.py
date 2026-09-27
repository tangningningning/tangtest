"""Full-information Cache value evaluation with exact-oracle hybrid screening.

KEEP is always evaluated exactly. A policy-independent delta-Q ensemble may
screen legal UPDATE candidates only after passing the existing rolling exact
quality gate; selected candidates are evaluated by the production exact branch
executor and override predictions.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch

from decision import cache_configuration, slot_environment, uav_configuration

from .primitive_control import (
    _cache_candidate_feature,
    _intermediate_observation,
    _target_cache_proposals_with_features,
)
from .rollout import _check_request_contract, _check_reward


OPERATION_ORDER = ("NO_CHANGE", "ADD", "DELETE", "SWAP")
CACHE_SURROGATE_K_CANDIDATES = (4, 6, 8, 10, 12)


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
        action_map = _production_action_map(transition)
        sequential_triggers = np.zeros(
            len(context.action_spec.cache_node_ids), dtype=np.int8
        )
        for index in range(node_index):
            if str(action_map[index].get("conditional_action", "")).upper() != "KEEP":
                sequential_triggers[index] = 1
        sequential_triggers[node_index] = 1
        update_candidates, unified_features = _target_cache_proposals_with_features(
            working_cache,
            previous_cache,
            sequential_triggers,
            node_index,
            np.asarray(conditional_context, dtype=np.float32),
            step,
            context,
            proposal_cache_nodes,
            surrogate_predictor=(
                trainer._predict_cache_surrogate
                if getattr(trainer, "cache_surrogate_ensemble", None) is not None
                else None
            ),
            surrogate_ready=bool(
                getattr(trainer, "cache_surrogate_screening_ready", False)
                and getattr(trainer, "cache_surrogate_screening_active", False)
            ),
        )
        if update_candidates:
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

    recall_values: dict[int, list[float]] = {
        k: [] for k in CACHE_SURROGATE_K_CANDIDATES
    }
    regret_values: dict[int, list[float]] = {
        k: [] for k in CACHE_SURROGATE_K_CANDIDATES
    }
    normalized_regret_values: dict[int, list[float]] = {
        k: [] for k in CACHE_SURROGATE_K_CANDIDATES
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
        for k in CACHE_SURROGATE_K_CANDIDATES:
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


def _robust_mad_scale(values: np.ndarray, epsilon: float, maximum: float) -> float:
    """Estimate a bounded scale from training values without outlier leverage."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    lower = float(epsilon)
    upper = float(maximum)
    if not len(array) or not math.isfinite(lower) or lower <= 0.0:
        return lower
    if not math.isfinite(upper) or upper < lower:
        raise ValueError("Cache teacher scale bounds are invalid")
    center = float(np.median(array))
    mad = float(np.median(np.abs(array - center)))
    robust_std = 1.4826 * mad
    typical_magnitude = float(np.median(np.abs(array)))
    estimate = max(robust_std, typical_magnitude, lower)
    return float(np.clip(estimate, lower, upper))


def _cache_teacher_probability_stats(
    probabilities: np.ndarray,
    valid_mask: np.ndarray,
    *,
    trigger: bool,
) -> dict[str, Any]:
    probs = np.asarray(probabilities, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if trigger:
        if probs.shape[:-1] != valid.shape or probs.shape[-1] != 2:
            raise ValueError("Cache Trigger teacher statistics have invalid shapes")
        update = probs[..., 1][valid]
        entropy = -np.sum(
            np.where(probs > 0.0, probs * np.log(np.maximum(probs, 1.0e-300)), 0.0),
            axis=-1,
        )[valid]
        return {
            "entropy_mean": float(np.mean(entropy)) if len(entropy) else 0.0,
            "entropy_median": float(np.median(entropy)) if len(entropy) else 0.0,
            "p_update_lt_0_01": float(np.mean(update < 0.01)) if len(update) else 0.0,
            "p_update_gt_0_99": float(np.mean(update > 0.99)) if len(update) else 0.0,
            "p_update_lt_0_05": float(np.mean(update < 0.05)) if len(update) else 0.0,
            "p_update_gt_0_95": float(np.mean(update > 0.95)) if len(update) else 0.0,
        }
    if probs.shape != valid.shape or probs.ndim < 2:
        raise ValueError("Cache Candidate teacher statistics have invalid shapes")
    entropy_values: list[float] = []
    top1_values: list[float] = []
    top3_values: list[float] = []
    near_one_hot: list[float] = []
    for row, mask in zip(probs.reshape(-1, probs.shape[-1]), valid.reshape(-1, valid.shape[-1])):
        legal = row[mask]
        if not len(legal):
            continue
        ordered = np.sort(legal)[::-1]
        entropy_values.append(float(-np.sum(legal * np.log(np.maximum(legal, 1.0e-300)))) )
        top1_values.append(float(ordered[0]))
        top3_values.append(float(np.sum(ordered[: min(3, len(ordered))])))
        near_one_hot.append(float(ordered[0] >= 0.99))
    return {
        "entropy_mean": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "top1_probability_mean": float(np.mean(top1_values)) if top1_values else 0.0,
        "top3_probability_mass_mean": float(np.mean(top3_values)) if top3_values else 0.0,
        "near_one_hot_fraction": float(np.mean(near_one_hot)) if near_one_hot else 0.0,
    }


def evaluate_cache_full_information_values(
    trainer: Any,
    rollout: Any,
    *,
    target_transition_indices: tuple[int, ...] | None = None,
    target_node_indices: tuple[int, ...] | None = None,
    trigger_temperature: float | None = None,
    candidate_temperature: float | None = None,
    force_full_exact: bool = False,
    train_surrogate: bool = True,
) -> dict[str, Any]:
    """Evaluate fixed-downstream Cache values and detached full-information teachers.

    FULL_EXACT is the oracle. After the surrogate passes its rolling exact-audit
    gate, the same evaluator uses HYBRID_EXACT_SURROGATE candidate completion.
    KEEP stays exact and every selected exact branch overrides its prediction.
    """

    trigger_scale = None if trigger_temperature is None else float(trigger_temperature)
    candidate_scale = None if candidate_temperature is None else float(candidate_temperature)
    if any(
        value is not None and (not math.isfinite(value) or value <= 0.0)
        for value in (trigger_scale, candidate_scale)
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
    candidate_predicted_delta_q = np.zeros_like(candidate_q)
    candidate_predicted_sigma = np.zeros_like(candidate_q)
    candidate_exact_evaluated = np.zeros_like(candidate_valid)
    candidate_q_sources = np.full(candidate_q.shape, "", dtype=object)
    trigger_policy_probabilities = np.full(
        (sample_count, node_count, 2), np.nan, dtype=np.float64
    )
    candidate_counts = np.zeros((sample_count, node_count), dtype=np.int64)
    environment_executions = 0
    deduplicated_branches = 0
    logical_candidate_count = 0
    keep_branch_count = 0
    exact_candidate_count = 0
    surrogate_candidate_count = 0
    full_exact_audit_count = 0
    full_exact_environment_baseline = 0
    surrogate_inference_runtime_seconds = 0.0
    surrogate_training_metrics: dict[str, Any] = {
        "count": 0,
        "loss": math.nan,
        "mae": math.nan,
        "rmse": math.nan,
    }
    exact_context_labels: list[np.ndarray] = []
    exact_feature_labels: list[np.ndarray] = []
    exact_delta_labels: list[float] = []
    exact_group_labels: list[int] = []
    screening_rows: list[dict[str, np.ndarray]] = []
    surrogate_ready = bool(
        not force_full_exact
        and getattr(trainer, "cache_surrogate_screening_ready", False)
        and getattr(trainer, "cache_surrogate_screening_active", False)
    )
    surrogate_rng = getattr(trainer, "cache_surrogate_rng", np.random.default_rng(0))
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
        full_exact_branch_keys: set[
            tuple[int, bytes, tuple[tuple[str, float, float], ...]]
        ] = set()
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
            for candidate in (None, *candidates):
                full_exact_branch_keys.add(
                    _cache_branch_identity(
                        transition,
                        node_index,
                        state,
                        candidate,
                    )
                )

            predicted_delta = np.zeros(count, dtype=np.float64)
            predicted_sigma = np.zeros(count, dtype=np.float64)
            if (
                count
                and not force_full_exact
                and getattr(trainer, "cache_surrogate_ensemble", None) is not None
            ):
                inference_started = time.perf_counter()
                predicted_delta, predicted_sigma = trainer._predict_cache_surrogate(
                    np.asarray(state["conditional_context"], dtype=np.float32),
                    np.stack(state["candidate_features"]).astype(np.float32),
                )
                surrogate_inference_runtime_seconds += time.perf_counter() - inference_started
                predicted_delta = np.asarray(predicted_delta, dtype=np.float64)
                predicted_sigma = np.asarray(predicted_sigma, dtype=np.float64)
            if count:
                candidate_predicted_delta_q[sample_index, node_index, :count] = predicted_delta
                candidate_predicted_sigma[sample_index, node_index, :count] = predicted_sigma

            periodic_audit = bool(
                surrogate_ready
                and count
                and (int(transition.origin_index) - int(trainer.context.train_origin_start))
                % max(int(getattr(trainer, "cache_surrogate_audit_interval", 20)), 1)
                == 0
            )
            full_exact_row = bool(force_full_exact or not surrogate_ready or periodic_audit)
            if full_exact_row:
                exact_indices = list(range(count))
                if periodic_audit:
                    full_exact_audit_count += 1
            else:
                selected = set()
                selected_rollout = _selected_cache_candidate_index(
                    transition,
                    node_index,
                    list(candidates),
                )
                if selected_rollout is not None:
                    selected.add(int(selected_rollout))
                selected_k = max(
                    1,
                    min(
                        int(getattr(trainer, "cache_surrogate_selected_k", None) or 1),
                        count,
                    ),
                )
                selected.update(
                    int(index)
                    for index in np.argsort(-predicted_delta, kind="mergesort")[:selected_k]
                )
                ucb = predicted_delta + 2.0 * predicted_sigma
                selected.update(
                    int(index)
                    for index in np.argsort(-ucb, kind="mergesort")[:selected_k]
                )
                selected.add(int(np.argmax(predicted_sigma)))
                unselected = sorted(set(range(count)) - selected)
                if unselected:
                    selected.add(int(surrogate_rng.choice(np.asarray(unselected, dtype=np.int64))))
                exact_indices = sorted(selected)
            exact_candidate_count += len(exact_indices)
            surrogate_candidate_count += count - len(exact_indices)
            branch_candidates: list[Any | None] = [None]
            branch_candidates.extend(candidates[index] for index in exact_indices)
            exact_values: list[float] = []
            for candidate in branch_candidates:
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
                exact_values.append(q_value)
            keep_value = float(exact_values[0])
            q_keep[sample_index, node_index] = keep_value
            if count:
                candidate_q[sample_index, node_index, :count] = keep_value + predicted_delta
                exact_by_index = np.full(count, np.nan, dtype=np.float64)
                if exact_indices:
                    exact_by_index[np.asarray(exact_indices, dtype=np.int64)] = np.asarray(
                        exact_values[1:], dtype=np.float64
                    )
                    candidate_q[sample_index, node_index, np.asarray(exact_indices)] = (
                        exact_by_index[np.asarray(exact_indices)]
                    )
                    candidate_exact_evaluated[sample_index, node_index, np.asarray(exact_indices)] = True
                candidate_q_sources[sample_index, node_index, :count] = np.where(
                    candidate_exact_evaluated[sample_index, node_index, :count],
                    "EXACT_Q",
                    "SURROGATE_Q",
                )
                exact_target_indices = np.asarray(exact_indices, dtype=np.int64)
                for candidate_index in exact_target_indices:
                    exact_context_labels.append(
                        np.asarray(state["conditional_context"], dtype=np.float32)
                    )
                    exact_feature_labels.append(
                        np.asarray(state["candidate_features"][int(candidate_index)], dtype=np.float32)
                    )
                    exact_delta_labels.append(
                        float(exact_by_index[int(candidate_index)] - keep_value)
                    )
                    exact_group_labels.append(
                        int(sample_index * node_count + node_index)
                    )
                if full_exact_row:
                    exact_delta = exact_by_index - keep_value
                    screening_rows.append(
                        {
                            "predicted_delta_q": predicted_delta.copy(),
                            "exact_q": exact_delta.copy(),
                        }
                    )
        full_exact_environment_baseline += len(full_exact_branch_keys)

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
    trigger_deltas = delta_best[np.isfinite(delta_best)]
    candidate_centered_values: list[float] = []
    candidate_ranges: list[float] = []
    candidate_stds: list[float] = []
    for sample_index in range(sample_count):
        for head_index in node_indices:
            legal = candidate_valid[sample_index, head_index]
            if not np.any(legal):
                continue
            row = candidate_q[sample_index, head_index, legal]
            center = float(np.median(row))
            candidate_centered_values.extend((row - center).tolist())
            candidate_ranges.append(float(np.max(row) - np.min(row)))
            candidate_stds.append(float(np.std(row)))

    scale_epsilon = float(getattr(trainer, "cache_teacher_scale_epsilon", 1.0e-3))
    scale_maximum = float(getattr(trainer, "cache_teacher_scale_max", 1.0e3))
    calibration = dict(getattr(trainer, "cache_teacher_scale_calibration", {}) or {})

    def resolve_scale(
        field: str,
        supplied: float | None,
        values: np.ndarray,
    ) -> float:
        if supplied is not None:
            return float(supplied)
        batch_scale = _robust_mad_scale(values, scale_epsilon, scale_maximum)
        previous = calibration.get(field)
        if previous is None or not math.isfinite(float(previous)) or float(previous) <= 0.0:
            resolved = batch_scale
        else:
            previous_scale = float(previous)
            clipped_batch = float(
                np.clip(batch_scale, previous_scale * 0.5, previous_scale * 2.0)
            )
            resolved = float(np.clip(
                0.5 * previous_scale + 0.5 * clipped_batch,
                scale_epsilon,
                scale_maximum,
            ))
        calibration[field] = resolved
        calibration[field + "_latest_batch_mad_scale"] = batch_scale
        calibration[field + "_source"] = "TRAIN_ROLLOUT_MAD"
        return resolved

    trigger_scale = resolve_scale("trigger_temperature", trigger_scale, trigger_deltas)
    candidate_scale = resolve_scale(
        "candidate_temperature",
        candidate_scale,
        np.asarray(candidate_centered_values, dtype=np.float64),
    )
    if trigger_temperature is None or candidate_temperature is None:
        calibration.update(
            {
                "method": "CLIPPED_RUNNING_MAD",
                "source": "TRAIN_ROLLOUT_ONLY",
                "epsilon": scale_epsilon,
                "maximum": scale_maximum,
                "training_value_count": int(len(trigger_deltas)),
            }
        )
        trainer.cache_teacher_scale_calibration = calibration
        trainer.cache_trigger_teacher_temperature = float(trigger_scale)
        trainer.cache_candidate_teacher_temperature = float(candidate_scale)

    for sample_index in range(sample_count):
        for head_index in node_indices:
            legal = candidate_valid[sample_index, head_index]
            if not np.any(legal):
                continue
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
        raise FloatingPointError("Cache candidate values are non-finite")
    surrogate_train_started = time.perf_counter()
    if exact_delta_labels and train_surrogate and not force_full_exact:
        surrogate_training_metrics = trainer._train_cache_surrogate(
            np.asarray(exact_context_labels, dtype=np.float32).reshape(
                -1, int(trainer.cache_surrogate_context_dim)
            ),
                np.asarray(exact_feature_labels, dtype=np.float32).reshape(
                    -1, int(trainer.cache_surrogate_feature_dim)
                ),
                np.asarray(exact_delta_labels, dtype=np.float32),
                ranking_group_ids=np.asarray(exact_group_labels, dtype=np.int64),
            )
    surrogate_training_runtime_seconds = float(
        time.perf_counter() - surrogate_train_started
    )
    gate_metrics = _screening_gate_metrics(screening_rows) if screening_rows else None
    delta_residuals = [
        float(predicted - exact)
        for row in screening_rows
        for predicted, exact in zip(row["predicted_delta_q"], row["exact_q"])
    ]
    delta_q_mae = (
        float(np.mean(np.abs(delta_residuals))) if delta_residuals else math.nan
    )
    delta_q_rmse = (
        float(np.sqrt(np.mean(np.square(delta_residuals))))
        if delta_residuals
        else math.nan
    )
    trigger_scale_stats = _cache_teacher_probability_stats(
        trigger_teacher,
        trigger_valid,
        trigger=True,
    )
    candidate_scale_stats = _cache_teacher_probability_stats(
        candidate_teacher,
        candidate_valid,
        trigger=False,
    )
    trigger_delta_stats = _audit_value_stats(trigger_deltas.tolist())
    trigger_delta_median = float(np.median(trigger_deltas)) if len(trigger_deltas) else 0.0
    trigger_delta_mad = (
        float(np.median(np.abs(trigger_deltas - trigger_delta_median)))
        if len(trigger_deltas)
        else 0.0
    )
    trigger_delta_stats.update(
        {
            "std": float(np.std(trigger_deltas)) if len(trigger_deltas) else 0.0,
            "mad": trigger_delta_mad,
            "p10": float(np.percentile(trigger_deltas, 10.0)) if len(trigger_deltas) else 0.0,
            "p25": float(np.percentile(trigger_deltas, 25.0)) if len(trigger_deltas) else 0.0,
            "p50": trigger_delta_median,
            "p75": float(np.percentile(trigger_deltas, 75.0)) if len(trigger_deltas) else 0.0,
            "p90": float(np.percentile(trigger_deltas, 90.0)) if len(trigger_deltas) else 0.0,
            "abs_p50": float(np.percentile(np.abs(trigger_deltas), 50.0)) if len(trigger_deltas) else 0.0,
            "abs_p90": float(np.percentile(np.abs(trigger_deltas), 90.0)) if len(trigger_deltas) else 0.0,
            "abs_p95": float(np.percentile(np.abs(trigger_deltas), 95.0)) if len(trigger_deltas) else 0.0,
        }
    )
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
        "candidate_predicted_delta_q": candidate_predicted_delta_q.astype(np.float32),
        "candidate_predicted_sigma": candidate_predicted_sigma.astype(np.float32),
        "candidate_exact_evaluated_mask": candidate_exact_evaluated,
        "candidate_q_sources": candidate_q_sources,
        "trigger_policy_probabilities": trigger_policy_probabilities.astype(np.float32),
        "candidate_features": candidate_features,
        "candidate_operation_indices": candidate_operation_indices,
        "candidate_counts": candidate_counts,
        "counterfactual_environment_executions": int(environment_executions),
        "logical_candidate_count": int(logical_candidate_count),
        "keep_branch_count": int(keep_branch_count),
        "deduplicated_branch_count": int(deduplicated_branches),
        "runtime_seconds": float(time.perf_counter() - started_at),
        "full_exact_environment_baseline": int(full_exact_environment_baseline),
        "exact_candidate_count": int(exact_candidate_count),
        "surrogate_candidate_count": int(surrogate_candidate_count),
        "full_exact_audit_count": int(full_exact_audit_count),
        "surrogate_inference_runtime_seconds": float(surrogate_inference_runtime_seconds),
        "surrogate_training_runtime_seconds": surrogate_training_runtime_seconds,
        "surrogate_training": surrogate_training_metrics,
        "surrogate_selected_k": getattr(trainer, "cache_surrogate_selected_k", None),
        "surrogate_gate_metrics": gate_metrics,
        "surrogate_delta_q_mae": delta_q_mae,
        "surrogate_delta_q_rmse": delta_q_rmse,
        "surrogate_screening_active": bool(surrogate_ready),
        "full_exact_evaluation": bool(gate_metrics is not None),
        "cache_teacher_scale_calibration": calibration,
        "trigger_delta_best_statistics": trigger_delta_stats,
        "candidate_q_range_mean": float(np.mean(candidate_ranges)) if candidate_ranges else 0.0,
        "candidate_q_std_mean": float(np.mean(candidate_stds)) if candidate_stds else 0.0,
        "trigger_teacher_statistics": trigger_scale_stats,
        "candidate_teacher_statistics": candidate_scale_stats,
        "value_evaluator": (
            "HYBRID_EXACT_SURROGATE"
            if surrogate_candidate_count > 0
            else "FULL_EXACT"
        ),
        "downstream_mode": "FIXED_DOWNSTREAM",
        "trigger_temperature": trigger_scale,
        "candidate_temperature": candidate_scale,
    }


def _audit_value_stats(values: list[float]) -> dict[str, float | None]:
    """Return compact finite distribution statistics for teacher values."""

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
        raise FloatingPointError("Cache teacher audit statistics are non-finite")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }
