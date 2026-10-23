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
from online_joint_dqn.candidate_features import ActionCandidate  # noqa: E402
from online_joint_dqn.environment_adapter import (  # noqa: E402
    ExclusiveDQNEnvironment,
    _input_fingerprint,
)
from decision import cache_configuration  # noqa: E402
from common import project_config  # noqa: E402


class CounterfactualInputError(ValueError):
    """The frozen state record and current legal candidate set do not align."""


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

    selected_q = step_record.get("selected_q_value", step_record.get("selected_candidate_q"))
    try:
        parsed_selected_q = float(selected_q)
        selected_q = parsed_selected_q if math.isfinite(parsed_selected_q) else None
    except (TypeError, ValueError):
        selected_q = None
    keep_best = _best_candidate_record(step_record, "KEEP_STAY")
    keep_q_value = None if keep_best is None else keep_best.get("q_value")
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
        requested.append((f"BEST_{operation}_Q", None if candidate_id is None else str(candidate_id), q_value))

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
            existing.setdefault("q_value_by_role", {})[role] = q_value
            continue
        index_by_candidate_id[candidate_id] = len(plan)
        plan.append(
            {
                "role": role,
                "roles": [role],
                "candidate_id": candidate_id,
                "operation": candidate.action.operation,
                "q_value": q_value,
                "q_value_by_role": {role: q_value},
                "candidate_action": candidate.action.to_dict(),
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
        "cache_state": np.asarray(environment.cache_state, dtype=np.int8).tolist(),
        "uav_positions": {
            key: np.asarray(value, dtype=np.float64).tolist()
            for key, value in environment.uav_positions.items()
        },
    }


def _clear_observation_cache(environment: ExclusiveDQNEnvironment) -> None:
    environment._cached_state_fingerprint = None
    environment._cached_state_features = None
    environment._cached_candidates = None


def restore_pre_action_state_from_step_records(
    environment: ExclusiveDQNEnvironment,
    episode_step_records: Sequence[Mapping[str, Any]],
    target_step_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore a frozen pre-action state from logged targets without executing Slots.

    ``environment`` must be a freshly constructed formal Episode at its initial
    state. Prior logged target configurations are replayed as state assignments;
    the recorded source fingerprints and formal configuration constraints are
    checked at every step. Matching and KKT are never called here.
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
        raise CounterfactualInputError("target Slot is outside the environment decision window")
    episode_index = target_step_record.get("episode_index")
    episode_seed = target_step_record.get("episode_seed")
    if (
        episode_seed is not None
        and environment.episode_seed is not None
        and int(episode_seed) != int(environment.episode_seed)
    ):
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
        raise CounterfactualInputError("target Slot record is absent from supplied Episode logs")
    target_log_row = rows_by_slot[target_slot_id]
    if (
        target_log_row.get("state_fingerprint") != target_step_record.get("state_fingerprint")
        or _candidate_id(target_log_row) != _candidate_id(target_step_record)
    ):
        raise CounterfactualInputError(
            "target step record differs from the matching row in the supplied Episode log"
        )

    environment.reset()
    for index in range(target_index):
        slot_input = environment.slot_inputs[index]
        slot_id = int(slot_input.slot_id)
        row = rows_by_slot.get(slot_id)
        if row is None:
            raise CounterfactualInputError(
                f"missing prior action record for Episode Slot {slot_id}"
            )
        current_fingerprint = environment._current_state_hash()
        action_payload = row.get("selected_candidate_action")
        if not isinstance(action_payload, Mapping):
            raise CounterfactualInputError(
                f"prior Slot {slot_id} lacks selected_candidate_action"
            )
        if str(row.get("state_fingerprint", "")) != current_fingerprint:
            raise CounterfactualInputError(
                f"prior Slot {slot_id} state fingerprint does not match reconstructed state"
            )
        if str(action_payload.get("source_state_fingerprint", "")) != current_fingerprint:
            raise CounterfactualInputError(
                f"prior Slot {slot_id} action belongs to a different source state"
            )
        try:
            target_cache = np.asarray(
                action_payload["target_cache_state"], dtype=np.int8
            )
            target_uav = {
                str(item[0]): np.asarray((float(item[1]), float(item[2])), dtype=np.float64)
                for item in action_payload["target_uav_positions"]
            }
            action = ActionCandidate.from_states(
                previous_cache_state=environment.cache_state,
                previous_uav_positions=environment.uav_positions,
                target_cache_state=target_cache,
                target_uav_positions=target_uav,
                slot_input=slot_input,
                source_state_fingerprint=current_fingerprint,
                proposal_source=str(action_payload.get("proposal_source", "replay")),
            )
        except (KeyError, TypeError, ValueError, IndexError, FloatingPointError) as exc:
            raise CounterfactualInputError(
                f"prior Slot {slot_id} target failed formal action validation: {exc}"
            ) from exc
        expected_candidate_id = str(
            row.get("selected_candidate_id", row.get("candidate_id", ""))
        )
        if action.candidate_id != expected_candidate_id:
            raise CounterfactualInputError(
                f"prior Slot {slot_id} candidate ID does not match its target configuration"
            )
        if action.operation != str(row.get("operation", action.operation)):
            raise CounterfactualInputError(
                f"prior Slot {slot_id} operation does not match its target configuration"
            )
        cache_configuration.validate_cache_shape_and_capacity(
            target_cache,
            dict(environment.model.service_catalog),
            dict(environment.model.cache_nodes),
        )
        environment.cache_state = target_cache.copy()
        environment.uav_positions = {
            key: value.copy() for key, value in target_uav.items()
        }
        environment.index = index + 1
        _clear_observation_cache(environment)

    if environment.index != target_index:
        raise CounterfactualInputError("state restoration stopped at the wrong Slot index")
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
        "prior_logged_slots_replayed_as_state_assignments": target_index,
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
    weighted_cache_cost = (
        None if cache_cost is None or beta_ca is None else float(cache_cost) * float(beta_ca)
    )
    weighted_movement_cost = (
        None
        if movement_cost is None or beta_q is None
        else float(movement_cost) * float(beta_q)
    )
    reconstructed_utility = (
        None
        if task_utility is None
        or weighted_cache_cost is None
        or weighted_movement_cost is None
        else float(task_utility) - weighted_cache_cost - weighted_movement_cost
    )
    return {
        "status": "SUCCEEDED",
        "roles": list(planned["roles"]),
        "q_value_by_role": dict(planned.get("q_value_by_role", {})),
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
        "actual_system_utility_G": float(transition.reward),
        "actual_task_utility_Q_task": task_utility,
        "cache_cost_C_ca": cache_cost,
        "beta_ca": beta_ca,
        "weighted_cache_cost": weighted_cache_cost,
        "cache_download_delay_s": diagnostics.get("cache_download_delay_s"),
        "movement_cost_C_q": movement_cost,
        "beta_q": beta_q,
        "weighted_movement_cost": weighted_movement_cost,
        "uav_movement_energy_j": diagnostics.get("uav_movement_energy_j"),
        "reward_formula": "G=Q_task-beta_ca*C_ca-beta_q*C_q",
        "reconstructed_system_utility_G": reconstructed_utility,
        "reward_recomposition_residual": (
            None
            if reconstructed_utility is None
            else float(transition.reward) - reconstructed_utility
        ),
        "actual_task_delay_sum_s": diagnostics.get("actual_total_task_delay_s"),
        "actual_vehicle_energy_sum_j": diagnostics.get("actual_total_vehicle_energy_j"),
        "actual_execution_nodes": diagnostics.get("actual_execution_nodes"),
        "actual_execution_node_data_available": diagnostics.get(
            "actual_execution_node_data_available"
        ),
        "matching_calls": transition.matching_calls,
        "kkt_solver_calls": transition.kkt_solver_calls,
        "predicted_cache_coverage_before": diagnostics.get(
            "predicted_cache_coverage_before"
        ),
        "predicted_cache_coverage_after": diagnostics.get(
            "predicted_cache_coverage_after"
        ),
        "predicted_cache_coverage_delta": diagnostics.get(
            "predicted_cache_coverage_delta"
        ),
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
    for field, config_path in (
        ("beta_ca", "objective.beta_ca"),
        ("beta_q", "objective.beta_q"),
    ):
        logged_weight = step_record.get(field)
        if logged_weight is None:
            raise CounterfactualInputError(f"logged {field} is unavailable")
        try:
            logged_weight = float(logged_weight)
            configured_weight = float(
                project_config.get_config_value(
                    environment.model.project_cfg, config_path
                )
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise CounterfactualInputError(
                f"could not validate reward weight {field}: {exc}"
            ) from exc
        if not math.isclose(
            logged_weight, configured_weight, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise CounterfactualInputError(
                f"logged {field}={logged_weight} differs from frozen config value {configured_weight}"
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
        if candidate is None or candidate.action.to_dict() != planned["candidate_action"]:
            raise CounterfactualInputError(
                f"candidate {planned['candidate_id']} did not reconstruct in the isolated legal set"
            )
        try:
            transition = clone.step(candidate)
            result = _transition_result(planned, transition)
        except Exception as exc:  # Preserve executor failures as per-candidate evidence.
            result = {
                "q_value_by_role": dict(planned.get("q_value_by_role", {})),
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
            and result.get("executor_failure_reason") is None
            and "actual_system_utility_G" in result
        ),
        None,
    )
    keep_utility = (
        None if keep_result is None else float(keep_result["actual_system_utility_G"])
    )
    for result in results:
        if "actual_system_utility_G" in result and keep_utility is not None:
            result["immediate_delta_G_vs_KEEP_STAY"] = (
                float(result["actual_system_utility_G"]) - keep_utility
            )
        else:
            result["immediate_delta_G_vs_KEEP_STAY"] = None
        result["long_horizon_return_difference"] = None

    original_unchanged = _environment_signature(environment) == before_signature
    return {
        "schema": "online_joint_dqn_counterfactual_review_v1",
        "episode_index": step_record.get("episode_index"),
        "slot_id": step_record.get("slot_id"),
        "state_fingerprint": before_signature["state_fingerprint"],
        "initial_state_signature": before_signature,
        "state_reconstruction_mode": "logged target states with fingerprint and formal constraint validation",
        "candidate_selection_used_only_logged_legal_ids": True,
        "evaluation_used_as_dqn_input": False,
        "original_environment_unchanged": original_unchanged,
        "training_pool_config_and_existing_outputs_written": False,
        "results": results,
        "interpretation": (
            "immediate_delta_G_vs_KEEP_STAY is a same-Slot realized utility difference only; "
            "it does not establish a long-horizon policy advantage"
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
    restoration = restore_pre_action_state_from_step_records(
        environment, episode_records, step_record
    )
    result = review_frozen_slot(environment, step_record)
    result["state_restoration"] = restoration
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is None:
        print(payload)
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
