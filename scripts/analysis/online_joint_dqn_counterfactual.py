#!/usr/bin/env python3
"""Evaluate selected logged candidates on isolated copies of a frozen DQN Slot.

This entry point is opt-in. Calling ``review_frozen_slot`` executes the
formal Slot executor once per planned action, including Matching/KKT. The
ordinary log audit never imports or calls this module.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import pickle
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from online_joint_dqn.candidate_features import CandidateOption  # noqa: E402
from online_joint_dqn.environment_adapter import (  # noqa: E402
    ExclusiveDQNEnvironment,
    _input_fingerprint,
)
from common import system_model  # noqa: E402


class CounterfactualInputError(ValueError):
    """The frozen state record and current legal candidate set do not align."""


class CounterfactualStateNotReconstructable(CounterfactualInputError):
    """The logged data is insufficient to restore the requested Slot state."""


def _candidate_id(record: Mapping[str, Any]) -> str | None:
    candidate_id = record.get("selected_candidate_id", record.get("candidate_id"))
    if candidate_id is not None:
        return str(candidate_id)
    selected_action = record.get("selected_candidate_action")
    if isinstance(selected_action, Mapping) and selected_action.get("candidate_id"):
        return str(selected_action["candidate_id"])
    return None


def _best_candidate_record(
    record: Mapping[str, Any], operation: str
) -> Mapping[str, Any] | None:
    explicit = record.get("best_candidate_by_operation")
    if isinstance(explicit, Mapping):
        candidate = explicit.get(operation)
        if isinstance(candidate, Mapping):
            return candidate
    q_statistics = record.get("candidate_q_statistics_by_operation")
    if isinstance(q_statistics, Mapping):
        candidate = q_statistics.get(operation)
        if isinstance(candidate, Mapping):
            candidate_id = candidate.get("max_candidate_id")
            if candidate_id is None:
                return None
            return {"candidate_id": candidate_id, "q_value": candidate.get("max")}
    return None


def build_counterfactual_plan(
    step_record: Mapping[str, Any],
    legal_candidates: Sequence[CandidateOption],
    current_state_fingerprint: str,
) -> list[dict[str, Any]]:
    """Select only logged action IDs present in the supplied legal set.

    This function only validates identities. It does not generate candidates,
    construct replacement targets, run features, or call the executor.
    """

    state_fingerprint = str(current_state_fingerprint)
    logged_fingerprint = step_record.get("state_fingerprint")
    if logged_fingerprint is None or str(logged_fingerprint) != state_fingerprint:
        raise CounterfactualInputError(
            "logged state_fingerprint does not match the frozen environment"
        )
    logged_source_fingerprint = step_record.get("candidate_source_state_fingerprint")
    if (
        logged_source_fingerprint is not None
        and str(logged_source_fingerprint) != state_fingerprint
    ):
        raise CounterfactualInputError(
            "logged candidate source fingerprint does not match the frozen environment"
        )
    legal_by_id: dict[str, CandidateOption] = {}
    for candidate in legal_candidates:
        action = candidate.action
        if action.source_state_fingerprint != state_fingerprint:
            raise CounterfactualInputError(
                f"legal candidate {action.candidate_id} belongs to another state"
            )
        if action.candidate_id in legal_by_id:
            raise CounterfactualInputError(
                f"duplicate legal candidate ID: {action.candidate_id}"
            )
        legal_by_id[action.candidate_id] = candidate

    keep_candidates = [
        candidate
        for candidate in legal_candidates
        if candidate.action.operation == "KEEP_STAY"
    ]
    if len(keep_candidates) != 1:
        raise CounterfactualInputError(
            f"frozen legal set must contain exactly one KEEP_STAY; found {len(keep_candidates)}"
        )
    logged_candidate_counts = step_record.get("candidate_counts_by_operation")
    if not isinstance(logged_candidate_counts, Mapping):
        raise CounterfactualInputError("logged candidate counts are unavailable")
    expected_operations = ("KEEP_STAY", "CACHE_UPDATE", "UAV_MOVE")
    if any(operation not in logged_candidate_counts for operation in expected_operations):
        raise CounterfactualInputError("logged candidate counts omit an operation category")
    actual_counts = {
        operation: sum(
            candidate.action.operation == operation
            for candidate in legal_candidates
        )
        for operation in expected_operations
    }
    for operation in expected_operations:
        try:
            parsed_count = int(logged_candidate_counts[operation])
        except (TypeError, ValueError) as exc:
            raise CounterfactualInputError(
                f"logged candidate count for {operation} is invalid"
            ) from exc
        if parsed_count != actual_counts[operation]:
            raise CounterfactualInputError(
                f"candidate count mismatch for {operation}: "
                f"logged={parsed_count}, frozen={actual_counts[operation]}"
            )

    selected_q = step_record.get(
        "selected_q_value", step_record.get("selected_candidate_q")
    )
    try:
        parsed_selected_q = float(selected_q)
        selected_q = parsed_selected_q if math.isfinite(parsed_selected_q) else None
    except (TypeError, ValueError):
        selected_q = None
    keep_best = _best_candidate_record(step_record, "KEEP_STAY")
    keep_q_candidate_id = None
    if keep_best is not None:
        logged_keep_id = _candidate_id(keep_best) or keep_best.get("max_candidate_id")
        if (
            logged_keep_id is not None
            and str(logged_keep_id) != keep_candidates[0].action.candidate_id
        ):
            raise CounterfactualInputError(
                "logged KEEP_STAY Q record refers to a different candidate ID"
            )
        if logged_keep_id is not None:
            keep_q_candidate_id = str(logged_keep_id)
    keep_q_value = (
        keep_best.get("q_value")
        if keep_best is not None
        and keep_q_candidate_id == keep_candidates[0].action.candidate_id
        else None
    )
    requested: list[tuple[str, str | None, float | None]] = [
        ("KEEP_STAY", keep_candidates[0].action.candidate_id, keep_q_value),
        ("DQN_SELECTED", _candidate_id(step_record), selected_q),
    ]
    for operation in ("CACHE_UPDATE", "UAV_MOVE"):
        best = _best_candidate_record(step_record, operation)
        candidate_id = None if best is None else best.get("candidate_id")
        q_value = None
        if best is not None:
            try:
                parsed_q = float(best.get("q_value"))
                q_value = parsed_q if math.isfinite(parsed_q) else None
            except (TypeError, ValueError):
                q_value = None
        requested.append(
            (
                f"BEST_{operation}_Q",
                None if candidate_id is None else str(candidate_id),
                q_value,
            )
        )

    plan: list[dict[str, Any]] = []
    index_by_candidate_id: dict[str, int] = {}
    for role, candidate_id, q_value in requested:
        if candidate_id is None:
            plan.append(
                {
                    "role": role,
                    "status": "SKIPPED",
                    "reason": "logged candidate identity is unavailable",
                }
            )
            continue
        candidate = legal_by_id.get(candidate_id)
        if candidate is None:
            if role == "DQN_SELECTED":
                raise CounterfactualInputError(
                    "logged DQN-selected candidate is not in the frozen state's legal set"
                )
            plan.append(
                {
                    "role": role,
                    "candidate_id": candidate_id,
                    "status": "SKIPPED",
                    "reason": "candidate ID is absent from the current legal set",
                }
            )
            continue
        selected_action = step_record.get("selected_candidate_action")
        if role == "DQN_SELECTED" and isinstance(selected_action, Mapping):
            if candidate.action.to_dict() != dict(selected_action):
                raise CounterfactualInputError(
                    "logged selected action payload differs from its current legal candidate"
                )
        if role == "DQN_SELECTED" and step_record.get("operation") is not None:
            if candidate.action.operation != str(step_record["operation"]):
                raise CounterfactualInputError(
                    "logged selected operation differs from its current legal candidate"
                )
        if role.startswith("BEST_"):
            expected_operation = role.removeprefix("BEST_").removesuffix("_Q")
            if candidate.action.operation != expected_operation:
                plan.append(
                    {
                        "role": role,
                        "candidate_id": candidate_id,
                        "status": "SKIPPED",
                        "reason": "logged Q-best candidate has a different operation",
                    }
                )
                continue
        if candidate_id in index_by_candidate_id:
            existing = plan[index_by_candidate_id[candidate_id]]
            existing["roles"].append(role)
            existing.setdefault("logged_q_value_by_role", {})[role] = q_value
            continue
        index_by_candidate_id[candidate_id] = len(plan)
        plan.append(
            {
                "role": role,
                "roles": [role],
                "candidate_id": candidate_id,
                "operation": candidate.action.operation,
                "logged_q_value_by_role": {role: q_value},
                "candidate_action": candidate.action.to_dict(),
                "logged_q_value_semantics": (
                    "logged Dueling DQN estimate for the original frozen state; "
                    "not a realized reward or long-horizon return"
                ),
                "status": "PLANNED",
            }
        )
    if not any(
        item.get("operation") == "KEEP_STAY" and item.get("status") == "PLANNED"
        for item in plan
    ):
        raise CounterfactualInputError("KEEP_STAY could not be resolved from the legal set")
    return plan


def _environment_signature(environment: ExclusiveDQNEnvironment) -> dict[str, Any]:
    index = int(environment.index)
    if not 0 <= index < len(environment.slot_inputs):
        raise CounterfactualInputError("frozen environment does not point at an active Slot")
    slot_input = environment.slot_inputs[index]
    config_payload = pickle.dumps(
        (
            slot_input.model.project_cfg,
            slot_input.model.cache_nodes,
            slot_input.model.service_catalog,
            slot_input.model.cache_config,
            slot_input.model.flight_boundary,
            environment.context.initial_cache,
            environment.context.initial_uav,
        ),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    return {
        "index": index,
        "state_fingerprint": environment._current_state_hash(),
        "slot_input_fingerprint": _input_fingerprint(slot_input),
        "system_configuration_sha256": hashlib.sha256(config_payload).hexdigest(),
        "task_lifecycle": {
            "active_task_states": [
                state.snapshot()
                for _, state in sorted(
                    environment.task_lifecycle.active_tasks.items()
                )
            ],
            "terminal_events": [
                dict(event) for event in environment.task_lifecycle.terminal_events
            ],
            "episode_upload_energy_j": float(
                environment.task_lifecycle.episode_upload_energy_j
            ),
            "next_slot_index": int(environment.task_lifecycle.next_slot_index),
            "prepared_slot_index": (
                None
                if environment.task_lifecycle._prepared is None
                else int(environment.task_lifecycle._prepared.slot_index)
            ),
            "prepared_slot_sha256": (
                None
                if environment.task_lifecycle._prepared is None
                else hashlib.sha256(
                    pickle.dumps(
                        environment.task_lifecycle._prepared,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                ).hexdigest()
            ),
            "settled_task_ids": sorted(environment._settled_task_ids),
        },
        "cache_state": np.asarray(environment.cache_state, dtype=np.int8).tolist(),
        "uav_positions": {
            key: np.asarray(value, dtype=np.float64).tolist()
            for key, value in environment.uav_positions.items()
        },
    }


def restore_pre_action_state_from_step_records(
    environment: ExclusiveDQNEnvironment,
    episode_step_records: Sequence[Mapping[str, Any]],
    target_step_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore only an exact initial Slot state; reject cross-Slot reconstruction.

    Existing step logs do not serialize the full lifecycle manager state needed
    to restore later Slots, including residual task work and terminal accounting.
    """

    if int(environment.index) != 0:
        raise CounterfactualInputError(
            "state restoration requires an Episode environment at index 0"
        )
    if not environment.slot_inputs:
        raise CounterfactualInputError("Episode environment has no aligned Slot inputs")
    target_slot_id = target_step_record.get("slot_id")
    if target_slot_id is None:
        raise CounterfactualInputError("target step record has no slot_id")
    try:
        target_slot_id = int(target_slot_id)
    except (TypeError, ValueError) as exc:
        raise CounterfactualInputError("target slot_id is invalid") from exc
    target_indices = [
        index
        for index, slot_input in enumerate(environment.slot_inputs)
        if int(slot_input.slot_id) == target_slot_id
    ]
    if len(target_indices) != 1:
        raise CounterfactualInputError(
            f"target Slot {target_slot_id} does not map to exactly one Episode input"
        )
    target_index = target_indices[0]
    if target_index >= int(environment.decision_count):
        raise CounterfactualInputError(
            "target Slot is outside the environment decision window"
        )
    if target_index != 0:
        raise CounterfactualStateNotReconstructable(
            "cross-Slot state is not reconstructable from the available logs: "
            "they do not contain a complete active-task lifecycle snapshot "
            "(remaining upload/compute work, execution node, cumulative task energy, "
            "terminal ledger and next lifecycle Slot)"
        )
    episode_index = target_step_record.get("episode_index")
    if episode_index is None:
        raise CounterfactualInputError("target step record has no episode_index")
    episode_seed = target_step_record.get("episode_seed")
    if episode_seed is None or environment.episode_seed is None:
        raise CounterfactualInputError(
            "target step and formal Episode must both provide an episode seed"
        )
    try:
        target_seed = int(episode_seed)
        environment_seed = int(environment.episode_seed)
    except (TypeError, ValueError) as exc:
        raise CounterfactualInputError(
            "target or formal Episode seed is invalid"
        ) from exc
    if target_seed != environment_seed:
        raise CounterfactualInputError("target step and formal Episode seeds differ")

    rows_by_slot: dict[int, Mapping[str, Any]] = {}
    for row in episode_step_records:
        if episode_index is not None and row.get("episode_index") != episode_index:
            continue
        raw_slot_id = row.get("slot_id")
        if raw_slot_id is None:
            continue
        slot_id = int(raw_slot_id)
        if slot_id in rows_by_slot:
            raise CounterfactualInputError(f"duplicate logged row for Slot {slot_id}")
        rows_by_slot[slot_id] = row
    if target_slot_id not in rows_by_slot:
        raise CounterfactualInputError(
            "target Slot record is absent from supplied Episode logs"
        )
    target_log_row = rows_by_slot[target_slot_id]
    target_log_seed = target_log_row.get("episode_seed")
    if target_log_seed is None:
        raise CounterfactualInputError("matching target log row has no episode_seed")
    try:
        target_log_seed = int(target_log_seed)
    except (TypeError, ValueError) as exc:
        raise CounterfactualInputError(
            "matching target log row has an invalid episode_seed"
        ) from exc
    if target_log_seed != target_seed:
        raise CounterfactualInputError(
            "target step and matching log row episode seeds differ"
        )
    if _candidate_id(target_step_record) is None or _candidate_id(target_log_row) is None:
        raise CounterfactualInputError(
            "target step or matching log row has no selected candidate identity"
        )
    if (
        target_log_row.get("episode_index") != episode_index
        or target_log_row.get("state_fingerprint")
        != target_step_record.get("state_fingerprint")
        or _candidate_id(target_log_row) != _candidate_id(target_step_record)
    ):
        raise CounterfactualInputError(
            "target step record differs from the matching row in the supplied Episode log"
        )

    if target_step_record.get("state_fingerprint") is None:
        raise CounterfactualInputError("target step record has no state_fingerprint")
    environment.reset()
    restored_fingerprint = environment._current_state_hash()
    if str(target_step_record.get("state_fingerprint", "")) != restored_fingerprint:
        raise CounterfactualInputError(
            "restored pre-action state fingerprint does not match the target log row"
        )
    return {
        "episode_index": episode_index,
        "episode_seed": environment.episode_seed,
        "target_slot_id": target_slot_id,
        "environment_index": target_index,
        "restoration_status": "RECONSTRUCTED_INITIAL_SLOT",
        "prior_logged_slots_replayed": 0,
        "lifecycle_manager_reset_to_formal_episode_initial_state": True,
        "state_fingerprint_verified": True,
        "matching_or_kkt_called_during_restoration": False,
        "state_fingerprint": restored_fingerprint,
    }


def _transition_result(
    planned: Mapping[str, Any],
    transition: Any,
) -> dict[str, Any]:
    diagnostics = transition.diagnostics
    cache_cost = diagnostics.get("cache_cost")
    movement_cost = diagnostics.get("movement_cost")
    beta_ca = diagnostics.get("beta_ca")
    beta_q = diagnostics.get("beta_q")
    task_utility = diagnostics.get("actual_task_utility")
    reward_components = {
        "actual_task_utility_Q_task": task_utility,
        "cache_cost_C_ca": cache_cost,
        "movement_cost_C_q": movement_cost,
    }
    missing_reward_components = [
        field for field, value in reward_components.items() if value is None
    ]
    if missing_reward_components:
        recomposition = {
            "status": "UNAVAILABLE_MISSING_COMPONENTS",
            "formula": "G(t)=Q_task(t)-C_ca(t)-C_q(t)",
            "value": None,
            "residual": None,
            "missing_components": missing_reward_components,
        }
    else:
        recomposed_reward = system_model.lifecycle_system_reward(
            float(task_utility), float(cache_cost), float(movement_cost)
        )
        recomposition = {
            "status": "AVAILABLE",
            "formula": "G(t)=Q_task(t)-C_ca(t)-C_q(t)",
            "value": recomposed_reward,
            "residual": float(transition.reward) - recomposed_reward,
            "missing_components": [],
        }
    return {
        "status": "SUCCEEDED",
        "roles": list(planned["roles"]),
        "logged_q_value_by_role": dict(
            planned.get("logged_q_value_by_role", {})
        ),
        "logged_q_value_semantics": planned.get("logged_q_value_semantics"),
        "candidate_id": transition.candidate_id,
        "operation": transition.operation,
        "candidate_target": dict(planned["candidate_action"]),
        "actual_cache_state": np.asarray(transition.final_cache_state, dtype=np.int8).tolist(),
        "actual_uav_positions_m": {
            key: np.asarray(value, dtype=np.float64).tolist()
            for key, value in transition.final_uav_positions.items()
        },
        "candidate_target_matches_execution": bool(
            diagnostics.get("candidate_target_matches_execution_pass") is True
        ),
        "counterfactual_slot_reward_G": float(transition.reward),
        "counterfactual_task_utility_Q_task": task_utility,
        "counterfactual_task_termination_events": diagnostics.get(
            "task_termination_events"
        ),
        "counterfactual_terminal_task_utility_by_task": diagnostics.get(
            "terminal_task_utility_by_task"
        ),
        "actual_task_utility_semantics": (
            "sum of one-time terminal utilities for tasks ending in this Slot; "
            "not utility for tasks still active"
        ),
        "counterfactual_cache_cost_C_ca": cache_cost,
        "legacy_beta_ca_config_not_applied": beta_ca,
        "cache_download_delay_s": diagnostics.get("cache_download_delay_s"),
        "counterfactual_movement_cost_C_q": movement_cost,
        "legacy_beta_q_config_not_applied": beta_q,
        "uav_movement_energy_j": diagnostics.get("uav_movement_energy_j"),
        "slot_reward_formula": "G(t)=Q_task(t)-C_ca(t)-C_q(t)",
        "reward_recomposition": recomposition,
        "actual_task_delay_sum_s": diagnostics.get("actual_total_task_delay_s"),
        "actual_vehicle_energy_sum_j": diagnostics.get("actual_total_vehicle_energy_j"),
        "actual_execution_nodes": diagnostics.get("actual_execution_nodes"),
        "actual_execution_node_data_available": diagnostics.get(
            "actual_execution_node_data_available"
        ),
        "matching_calls": transition.matching_calls,
        "kkt_solver_calls": transition.kkt_solver_calls,
        "predicted_features": {
            "cache_coverage_before": diagnostics.get(
                "predicted_cache_coverage_before"
            ),
            "cache_coverage_after": diagnostics.get(
                "predicted_cache_coverage_after"
            ),
            "cache_coverage_delta": diagnostics.get(
                "predicted_cache_coverage_delta"
            ),
            "semantics": (
                "predicted demand-weighted cache serviceability feature; "
                "not realized task utility or reward"
            ),
        },
        "executor_failure_reason": None,
    }


def review_frozen_slot(
    environment: ExclusiveDQNEnvironment,
    step_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the selected comparison set on one deep-copied environment per action."""

    before_signature = _environment_signature(environment)
    feature_contract = step_record.get("candidate_feature_contract_version")
    if feature_contract is None or str(feature_contract) != str(
        environment.encoder.candidate_feature_contract_version
    ):
        raise CounterfactualInputError(
            "candidate feature contract is missing or differs from the frozen environment"
        )
    frozen_copy = copy.deepcopy(environment)
    if _environment_signature(frozen_copy) != before_signature:
        raise CounterfactualInputError(
            "deep-copied environment does not match the frozen state/configuration"
        )
    _, legal_candidates = frozen_copy.observe()
    plan = build_counterfactual_plan(
        step_record,
        legal_candidates,
        before_signature["state_fingerprint"],
    )
    results: list[dict[str, Any]] = []
    for planned in plan:
        if planned.get("status") != "PLANNED":
            results.append(dict(planned))
            continue
        clone = copy.deepcopy(frozen_copy)
        if _environment_signature(clone) != before_signature:
            raise CounterfactualInputError(
                "isolated environment copy does not match the frozen initial state"
            )
        _, clone_legal_candidates = clone.observe()
        candidate = next(
            (
                item
                for item in clone_legal_candidates
                if item.action.candidate_id == planned["candidate_id"]
            ),
            None,
        )
        if (
            candidate is None
            or candidate.action.to_dict() != planned["candidate_action"]
        ):
            raise CounterfactualInputError(
                f"candidate {planned['candidate_id']} did not reconstruct in the isolated legal set"
            )
        try:
            transition = clone.step(candidate)
            result = _transition_result(planned, transition)
        except Exception as exc:  # Preserve executor failures as per-candidate evidence.
            result = {
                "logged_q_value_by_role": dict(
                    planned.get("logged_q_value_by_role", {})
                ),
                "logged_q_value_semantics": planned.get(
                    "logged_q_value_semantics"
                ),
                "roles": list(planned["roles"]),
                "candidate_id": planned["candidate_id"],
                "operation": planned["operation"],
                "candidate_target": dict(planned["candidate_action"]),
                "status": "FAILED",
                "executor_failure_reason": f"{type(exc).__name__}: {exc}",
            }
        if _environment_signature(environment) != before_signature:
            raise RuntimeError("counterfactual evaluation modified the original environment")
        results.append(result)

    keep_result = next(
        (
            result
            for result in results
            if result.get("operation") == "KEEP_STAY"
            and result.get("status") == "SUCCEEDED"
            and result.get("candidate_target_matches_execution") is True
            and "counterfactual_slot_reward_G" in result
        ),
        None,
    )
    keep_utility = (
        None
        if keep_result is None
        else float(keep_result["counterfactual_slot_reward_G"])
    )
    for result in results:
        comparable = (
            result.get("status") == "SUCCEEDED"
            and result.get("candidate_target_matches_execution") is True
            and keep_utility is not None
        )
        if comparable:
            result["immediate_delta_G_vs_KEEP_STAY"] = float(
                result["counterfactual_slot_reward_G"]
            ) - keep_utility
            result["immediate_comparison_status"] = "COMPUTED"
            result["immediate_comparison_reason"] = None
        else:
            result["immediate_delta_G_vs_KEEP_STAY"] = None
            result["immediate_comparison_status"] = "UNAVAILABLE"
            if keep_utility is None:
                result["immediate_comparison_reason"] = (
                    "KEEP_STAY did not complete with its requested target executed exactly"
                )
            elif result.get("status") != "SUCCEEDED":
                result["immediate_comparison_reason"] = (
                    "candidate execution did not complete successfully"
                )
            else:
                result["immediate_comparison_reason"] = (
                    "formal execution did not match the requested candidate target"
                )
        result["long_horizon_strategy_return"] = {
            "status": "UNSUPPORTED",
            "value": None,
            "reason": (
                "independent one-Slot counterfactuals do not roll out future "
                "policy decisions"
            ),
        }
        result["cache_uav_same_slot_interaction"] = {
            "status": "UNSUPPORTED",
            "value": None,
            "reason": (
                "the comparison evaluates single legal action candidates "
                "independently and does not estimate a combined cache-plus-UAV effect"
            ),
        }

    original_unchanged = _environment_signature(environment) == before_signature
    partial = any(
        result.get("status") != "SUCCEEDED"
        or result.get("candidate_target_matches_execution") is not True
        or result.get("reward_recomposition", {}).get("status")
        != "AVAILABLE"
        or result.get("counterfactual_task_termination_events") is None
        for result in results
    ) or any(result.get("status") == "SKIPPED" for result in results)
    return {
        "schema": "online_joint_dqn_counterfactual_review_v2",
        "analysis_status": "PARTIAL" if partial else "COMPLETED",
        "episode_index": step_record.get("episode_index"),
        "slot_id": step_record.get("slot_id"),
        "state_fingerprint": before_signature["state_fingerprint"],
        "initial_state_signature": before_signature,
        "state_reconstruction_mode": "formal Episode initial state reset plus exact logged fingerprint match; cross-Slot replay is refused",
        "supported_state_scope": "formal Episode initial Slot only (environment index 0)",
        "candidate_comparisons_use_only_regenerated_legal_candidates": True,
        "logged_candidate_ids_revalidated": True,
        "evaluation_used_as_dqn_input": False,
        "original_environment_unchanged": original_unchanged,
        "training_pool_config_and_existing_outputs_written": False,
        "comparison_scope": {
            "included": [
                "unique legal KEEP_STAY candidate",
                "logged DQN-selected candidate when its identity matches the regenerated legal set",
                "logged Q-best CACHE_UPDATE candidate when its identity and category match the regenerated legal set",
                "logged Q-best UAV_MOVE candidate when its identity and category match the regenerated legal set",
            ],
            "exhaustive_operation_category_search": False,
            "logged_q_values_are_realized_rewards": False,
            "predicted_features_are_realized_rewards": False,
        },
        "unsupported_metrics": {
            "long_horizon_strategy_return": "not computed: future policy decisions are not rolled out",
            "cache_uav_same_slot_interaction_gain": (
                "not computed: combined cache-plus-UAV counterfactual is not evaluated"
            ),
        },
        "results": results,
        "interpretation": (
            "counterfactual_slot_reward_G is the immediate formal Slot reward; Q_task contains only "
            "one-time terminal utilities of tasks ending in this Slot. Logged Q values and "
            "predicted cache features are separate quantities. Same-Slot reward differences "
            "do not establish long-horizon policy advantage or cache-UAV interaction gains."
        ),
    }


def _load_step_records(path: Path) -> list[dict[str, Any]]:
    content = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in content.splitlines() if line.strip()]
    if isinstance(parsed, dict):
        records = [parsed]
    elif isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        records = parsed
    else:
        raise CounterfactualInputError("step record input must contain JSON object rows")
    if not records:
        raise CounterfactualInputError("step record input contains no rows")
    return records


def _select_step_record(
    records: Sequence[Mapping[str, Any]],
    episode_index: int | None,
    slot_id: int | None,
) -> dict[str, Any]:
    if episode_index is None or slot_id is None:
        if len(records) != 1:
            raise CounterfactualInputError(
                "multiple step rows require both --episode-index and --slot-id"
            )
        return dict(records[0])
    matches = [
        record
        for record in records
        if int(record.get("episode_index", -1)) == episode_index
        and int(record.get("slot_id", -1)) == slot_id
    ]
    if len(matches) != 1:
        raise CounterfactualInputError(
            f"expected one step row for episode={episode_index}, slot={slot_id}; found {len(matches)}"
        )
    return dict(matches[0])


def _load_provider(specification: str) -> Any:
    module_name, separator, function_name = specification.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("--environment-provider must use module:function syntax")
    module = importlib.import_module(module_name)
    provider = getattr(module, function_name)
    if not callable(provider):
        raise TypeError("environment provider is not callable")
    return provider


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-record", type=Path, required=True)
    parser.add_argument(
        "--environment-provider",
        required=True,
        help="Importable module:function returning the formal Episode environment at its initial state.",
    )
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--slot-id", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        help="New JSON output path. Existing files are never overwritten.",
    )
    return parser.parse_args(argv)


def _write_result(result: Mapping[str, Any], output: Path | None) -> None:
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if output is None:
        print(payload)
    else:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    step_records = _load_step_records(args.step_record)
    step_record = _select_step_record(
        step_records, args.episode_index, args.slot_id
    )
    environment = _load_provider(args.environment_provider)(step_record)
    episode_records = [
        row
        for row in step_records
        if row.get("episode_index") == step_record.get("episode_index")
    ]
    try:
        restoration = restore_pre_action_state_from_step_records(
            environment, episode_records, step_record
        )
    except CounterfactualStateNotReconstructable as exc:
        _write_result(
            {
                "schema": "online_joint_dqn_counterfactual_review_v2",
                "analysis_status": "STATE_NOT_RECONSTRUCTABLE",
                "episode_index": step_record.get("episode_index"),
                "slot_id": step_record.get("slot_id"),
                "state_restoration": {
                    "status": "NOT_RECONSTRUCTABLE",
                    "reason": str(exc),
                    "state_fingerprint_verified": False,
                },
                "supported_state_scope": "formal Episode initial Slot only (environment index 0)",
                "unsupported_metrics": {
                    "long_horizon_strategy_return": "not computed: future policy decisions are not rolled out",
                    "cache_uav_same_slot_interaction_gain": "not computed: combined cache-plus-UAV counterfactual is not evaluated",
                },
            },
            args.output,
        )
        return 2
    except CounterfactualInputError as exc:
        _write_result(
            {
                "schema": "online_joint_dqn_counterfactual_review_v2",
                "analysis_status": "REJECTED_INPUT_OR_STATE_MISMATCH",
                "episode_index": step_record.get("episode_index"),
                "slot_id": step_record.get("slot_id"),
                "reason": str(exc),
            },
            args.output,
        )
        return 2
    try:
        result = review_frozen_slot(environment, step_record)
    except CounterfactualInputError as exc:
        _write_result(
            {
                "schema": "online_joint_dqn_counterfactual_review_v2",
                "analysis_status": "REJECTED_INPUT_OR_STATE_MISMATCH",
                "episode_index": step_record.get("episode_index"),
                "slot_id": step_record.get("slot_id"),
                "reason": str(exc),
            },
            args.output,
        )
        return 2
    result["state_restoration"] = restoration
    _write_result(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
