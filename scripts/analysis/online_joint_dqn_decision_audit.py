#!/usr/bin/env python3
"""Review recorded Online Joint DQN decisions without executing the environment."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPERATIONS = ("CACHE_UPDATE", "UAV_MOVE", "KEEP_STAY")
_Q_FIELDS = (
    "count",
    "mean",
    "std",
    "min",
    "p50",
    "p90",
    "max",
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    items = sorted(values)
    if not items:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(items),
        "mean": statistics.fmean(items),
        "median": statistics.median(items),
        "min": items[0],
        "max": items[-1],
    }


def _read_step_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    malformed_count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed_count += 1
                continue
            if not isinstance(row, dict):
                malformed_count += 1
                continue
            rows.append(row)
    return rows, malformed_count


def _action_id(row: Mapping[str, Any]) -> str | None:
    direct = row.get("selected_candidate_id", row.get("candidate_id"))
    if direct is not None:
        return str(direct)
    action = row.get("selected_candidate_action")
    if isinstance(action, Mapping) and action.get("candidate_id") is not None:
        return str(action["candidate_id"])
    return None


def _action_operation(row: Mapping[str, Any]) -> str | None:
    operation = row.get("operation", row.get("selected_action_type"))
    if operation is None:
        action = row.get("selected_candidate_action")
        operation = action.get("operation") if isinstance(action, Mapping) else None
    value = None if operation is None else str(operation)
    return value if value in OPERATIONS else None


def _selection_mode(row: Mapping[str, Any]) -> str | None:
    mode = row.get("selection_mode")
    if mode in {"exploration", "greedy"}:
        return str(mode)
    explored = row.get("explored")
    if isinstance(explored, bool):
        return "exploration" if explored else "greedy"
    return None


def _q_statistics(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("candidate_q_statistics_by_operation")
    return value if isinstance(value, Mapping) else {}


def _candidate_best_q(row: Mapping[str, Any], operation: str) -> float | None:
    explicit = row.get("best_candidate_by_operation")
    if isinstance(explicit, Mapping):
        candidate = explicit.get(operation)
        if isinstance(candidate, Mapping):
            value = _number(candidate.get("q_value"))
            if value is not None:
                return value
    statistics_by_operation = _q_statistics(row)
    item = statistics_by_operation.get(operation)
    if isinstance(item, Mapping):
        # This is the maximum complete candidate Q, never category_value.
        return _number(item.get("max"))
    return None


def _greedy_argmax_result(row: Mapping[str, Any]) -> bool | None:
    """Reconstruct the logged global-Q choice from per-operation maxima.

    Per-operation statistics preserve each full-Q maximum and its stable
    candidate ID, so historical rows can be checked even when they predate
    the explicit ``greedy_global_argmax_pass`` field.
    """

    if _selection_mode(row) != "greedy":
        return None
    selected_q = _selected_q(row)
    selected_id = _action_id(row)
    if selected_q is None or selected_id is None:
        return None

    statistics = _q_statistics(row)
    best_by_operation = row.get("best_candidate_by_operation")
    maxima: list[tuple[float, str]] = []
    for operation in OPERATIONS:
        best_q = _candidate_best_q(row, operation)
        if best_q is None:
            continue
        candidate_id: Any = None
        if isinstance(best_by_operation, Mapping):
            best_record = best_by_operation.get(operation)
            if isinstance(best_record, Mapping):
                candidate_id = best_record.get("candidate_id")
        if candidate_id is None:
            stat = statistics.get(operation)
            if isinstance(stat, Mapping):
                candidate_id = stat.get("max_candidate_id")
        if candidate_id is None:
            return None
        maxima.append((best_q, str(candidate_id)))
    if not maxima:
        return None

    maximum_q = max(q_value for q_value, _ in maxima)
    expected_id = min(
        candidate_id for q_value, candidate_id in maxima if q_value == maximum_q
    )
    return selected_q == maximum_q and selected_id == expected_id


def _selected_q(row: Mapping[str, Any]) -> float | None:
    return _number(row.get("selected_q_value", row.get("selected_candidate_q")))


def _keep_q(row: Mapping[str, Any]) -> float | None:
    value = _number(row.get("keep_stay_q_value"))
    return value if value is not None else _candidate_best_q(row, "KEEP_STAY")


def _reward(row: Mapping[str, Any]) -> float | None:
    value = _number(row.get("actual_system_utility"))
    return value if value is not None else _number(row.get("reward"))


def _changed(row: Mapping[str, Any], operation: str) -> bool | None:
    if isinstance(row.get("actual_state_changed"), bool):
        return bool(row["actual_state_changed"])
    if operation == "CACHE_UPDATE" and isinstance(row.get("cache_state_changed"), bool):
        return bool(row["cache_state_changed"])
    if operation == "UAV_MOVE" and isinstance(row.get("uav_state_changed"), bool):
        return bool(row["uav_state_changed"])
    return None


def _candidate_counts(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("candidate_counts_by_operation")
    return value if isinstance(value, Mapping) else {}


def _state_level_q_summary(rows: list[dict[str, Any]], operation: str) -> dict[str, Any]:
    per_slot: dict[str, list[float]] = {key: [] for key in _Q_FIELDS if key != "count"}
    candidate_counts: list[float] = []
    for row in rows:
        count = _number(_candidate_counts(row).get(operation))
        if count is not None:
            candidate_counts.append(count)
        values = _q_statistics(row).get(operation)
        if not isinstance(values, Mapping):
            continue
        for key in per_slot:
            value = _number(values.get(key))
            if value is not None:
                per_slot[key].append(value)
    return {
        "candidate_count_per_slot": _summarize(candidate_counts),
        "state_weighted_q_statistics": {
            key: _summarize(values) for key, values in per_slot.items()
        },
    }


def _pearson_correlation(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    left, right = zip(*pairs, strict=True)
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss <= 0.0 or right_ss <= 0.0:
        return None
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in pairs
    )
    return covariance / math.sqrt(left_ss * right_ss)


def analyze_step_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate recorded decisions while preserving unavailable values."""

    by_operation: dict[str, list[dict[str, Any]]] = {
        operation: [] for operation in OPERATIONS
    }
    selection_mode_counts = {operation: {"exploration": 0, "greedy": 0} for operation in OPERATIONS}
    changed_counts = {operation: {"n": 0, "changed": 0, "unchanged": 0} for operation in OPERATIONS}
    reward_signs = {
        operation: {"positive": 0, "zero": 0, "negative": 0, "n": 0}
        for operation in OPERATIONS
    }
    q_gaps_by_operation: dict[str, list[float]] = {operation: [] for operation in OPERATIONS}
    keep_q_gaps: list[float] = []
    system_utility: list[float] = []
    task_utility: list[float] = []
    cache_cost_raw: list[float] = []
    cache_cost_weighted: list[float] = []
    movement_cost_raw: list[float] = []
    movement_cost_weighted: list[float] = []
    task_delay: list[float] = []
    vehicle_energy: list[float] = []
    recomposition_residuals: list[float] = []
    scoped_prediction_pairs: list[tuple[float, float]] = []
    raw_prediction_delta_count = 0
    prediction_scope_missing_count = 0
    missing_fields: dict[str, int] = {}
    cost_unavailable = {"cache": 0, "movement": 0}
    truncation = {"n": 0, "truncated": 0, "bounded_complete": 0, "exhaustive": 0}
    search_policy = {
        "cache_diagnostics_n": 0,
        "cache_candidate_budget_reached_n": 0,
        "cache_generated_candidate_counts": [],
        "cache_candidate_limits": [],
        "cache_omitted_generated_proposal_total": 0,
        "cross_node_pair_truncated_n": 0,
        "cross_node_pair_scan_limit_touched_n": 0,
        "cross_node_pair_retention_limit_omitted_n": 0,
        "cross_node_pair_retention_limit_omitted_total": 0,
        "cross_node_pair_unscanned_total": 0,
        "cross_node_pair_candidates_retained_total": 0,
        "uav_diagnostics_n": 0,
        "uav_candidate_limit_reached_n": 0,
        "incomplete_reason_counts": Counter(),
    }
    anomalies = {
        "cache_update_without_cache_change": 0,
        "uav_move_without_uav_change": 0,
        "logged_target_execution_mismatch": 0,
        "candidate_state_fingerprint_mismatch": 0,
        "selected_action_not_marked_legal": 0,
        "selected_action_not_marked_constraint_valid": 0,
        "greedy_global_argmax_failure": 0,
        "greedy_global_argmax_flag_reconstruction_mismatch": 0,
        "reward_recomposition_residual_over_1e-8": 0,
    }
    greedy_argmax_check_count = 0
    greedy_argmax_unverifiable_count = 0
    required_fields = (
        "candidate_counts_by_operation",
        "candidate_q_statistics_by_operation",
        "best_candidate_by_operation",
        "selected_q_minus_best_candidate_q_by_operation",
        "keep_stay_q_value",
        "selected_q_minus_keep_stay_q",
        "selected_candidate_action",
        "candidate_source_state_fingerprint",
        "candidate_source_state_fingerprint_matches_current_pass",
        "exploration_selected_operation",
        "exploration_selected_candidate_id",
        "state_fingerprint",
        "actual_system_utility",
        "actual_task_utility",
        "actual_execution_nodes",
        "actual_execution_node_data_available",
        "predicted_cache_coverage_delta",
        "predicted_cache_coverage_metric_scope",
        "predicted_service_demand_by_service",
        "cache_service_demand_details",
        "cache_update_cost_available",
        "cache_update_cost_definition",
        "cache_update_cost_unit",
        "uav_movement_cost_available",
        "uav_movement_cost_definition",
        "uav_movement_cost_unit",
        "reward_formula",
        "actual_task_utility_unit",
        "reward_unit",
        "reward_recomposition_residual",
        "candidate_current_legal_set_membership_pass",
        "candidate_mutual_exclusion_pass",
        "candidate_cache_replace_rule_pass",
        "candidate_cache_capacity_pass",
        "candidate_uav_boundary_pass",
        "candidate_uav_safety_distance_pass",
        "candidate_uav_transition_constraints_pass",
        "candidate_target_matches_execution_pass",
        "pre_cache_state",
        "post_cache_state",
        "pre_uav_positions_m",
        "post_uav_positions_m",
        "same_slot_matching_alternative_comparison_available",
    )

    for row in rows:
        for field in required_fields:
            if field not in row:
                missing_fields[field] = missing_fields.get(field, 0) + 1
        operation = _action_operation(row)
        if operation is None:
            continue
        by_operation[operation].append(row)
        mode = _selection_mode(row)
        if mode is not None:
            selection_mode_counts[operation][mode] += 1
        changed = _changed(row, operation)
        if changed is not None:
            changed_counts[operation]["n"] += 1
            changed_counts[operation]["changed" if changed else "unchanged"] += 1
        if operation == "CACHE_UPDATE" and changed is False:
            anomalies["cache_update_without_cache_change"] += 1
        if operation == "UAV_MOVE" and changed is False:
            anomalies["uav_move_without_uav_change"] += 1

        selection_q = _selected_q(row)
        keep_q = _keep_q(row)
        if selection_q is not None and keep_q is not None:
            keep_q_gaps.append(selection_q - keep_q)
        for candidate_operation in OPERATIONS:
            best_q = _candidate_best_q(row, candidate_operation)
            if selection_q is not None and best_q is not None:
                q_gaps_by_operation[candidate_operation].append(selection_q - best_q)

        utility = _reward(row)
        if utility is not None:
            system_utility.append(utility)
            sign_counts = reward_signs[operation]
            sign_counts["n"] += 1
            if utility > 1.0e-12:
                sign_counts["positive"] += 1
            elif utility < -1.0e-12:
                sign_counts["negative"] += 1
            else:
                sign_counts["zero"] += 1
        value = _number(row.get("actual_task_utility"))
        if value is not None:
            task_utility.append(value)
        value = _number(row.get("cache_update_cost", row.get("cache_download_cost")))
        cache_available = row.get("cache_update_cost_available")
        if value is not None and cache_available is not False:
            cache_cost_raw.append(value)
            weighted = _number(row.get("cache_cost_weighted"))
            if weighted is None:
                beta_ca = _number(row.get("beta_ca"))
                weighted = None if beta_ca is None else beta_ca * value
            if weighted is not None:
                cache_cost_weighted.append(weighted)
        else:
            cost_unavailable["cache"] += 1
        value = _number(row.get("uav_movement_cost"))
        movement_available = row.get("uav_movement_cost_available")
        if value is not None and movement_available is not False:
            movement_cost_raw.append(value)
            weighted = _number(row.get("uav_movement_cost_weighted"))
            if weighted is None:
                beta_q = _number(row.get("beta_q"))
                weighted = None if beta_q is None else beta_q * value
            if weighted is not None:
                movement_cost_weighted.append(weighted)
        else:
            cost_unavailable["movement"] += 1
        value = _number(row.get("actual_total_task_delay_s"))
        if value is not None:
            task_delay.append(value)
        value = _number(row.get("actual_total_vehicle_energy_j"))
        if value is not None:
            vehicle_energy.append(value)
        value = _number(row.get("reward_recomposition_residual"))
        if value is None:
            task_value = _number(row.get("actual_task_utility"))
            cache_value = _number(row.get("cache_cost_weighted"))
            movement_value = _number(row.get("uav_movement_cost_weighted"))
            if cache_value is None:
                raw_cache_value = _number(
                    row.get("cache_update_cost", row.get("cache_download_cost"))
                )
                beta_ca = _number(row.get("beta_ca"))
                if raw_cache_value is not None and beta_ca is not None:
                    cache_value = raw_cache_value * beta_ca
            if movement_value is None:
                raw_movement_value = _number(row.get("uav_movement_cost"))
                beta_q = _number(row.get("beta_q"))
                if raw_movement_value is not None and beta_q is not None:
                    movement_value = raw_movement_value * beta_q
            if (
                utility is not None
                and task_value is not None
                and cache_value is not None
                and movement_value is not None
            ):
                value = utility - (task_value - cache_value - movement_value)
        if value is not None:
            recomposition_residuals.append(value)
            if abs(value) > 1.0e-8:
                anomalies["reward_recomposition_residual_over_1e-8"] += 1

        predicted_delta = _number(row.get("predicted_cache_coverage_delta"))
        if predicted_delta is not None:
            raw_prediction_delta_count += 1
            actual_task = _number(row.get("actual_task_utility"))
            scope = row.get("predicted_cache_coverage_metric_scope")
            coverage_available = row.get("predicted_cache_coverage_available")
            if (
                actual_task is not None
                and coverage_available is True
                and isinstance(scope, str)
                and "state positions" in scope
            ):
                scoped_prediction_pairs.append((predicted_delta, actual_task))
            elif scope is None:
                prediction_scope_missing_count += 1

        truncation["n"] += 1
        is_truncated = (
            row.get("dqn_adapter_additional_candidate_truncation") is True
            or row.get("candidate_generation_complete_within_bounded_proposal_policy") is False
        )
        truncation["truncated"] += int(is_truncated)
        truncation["bounded_complete"] += int(
            row.get("candidate_generation_complete_within_bounded_proposal_policy") is True
        )
        truncation["exhaustive"] += int(row.get("candidate_legal_action_space_exhaustive") is True)

        cache_search = row.get("cache_candidate_generation_diagnostics")
        if isinstance(cache_search, Mapping):
            search_policy["cache_diagnostics_n"] += 1
            search_policy["incomplete_reason_counts"].update(
                str(reason)
                for reason in cache_search.get("incomplete_reasons", ())
            )
            generated_count = _number(
                cache_search.get("generated_candidate_count_including_keep")
            )
            candidate_limit = _number(
                cache_search.get("candidate_limit_including_keep")
            )
            if generated_count is not None:
                search_policy["cache_generated_candidate_counts"].append(generated_count)
            if candidate_limit is not None:
                search_policy["cache_candidate_limits"].append(candidate_limit)
            if bool(cache_search.get("per_node_limit_reached")) or (
                generated_count is not None
                and candidate_limit is not None
                and generated_count >= candidate_limit
            ):
                search_policy["cache_candidate_budget_reached_n"] += 1
            search_policy["cache_omitted_generated_proposal_total"] += int(
                _number(cache_search.get("omitted_generated_proposal_count")) or 0
            )

            pair_truncated = cache_search.get(
                "cache_cross_node_pair_candidates_truncated"
            ) is True
            search_policy["cross_node_pair_truncated_n"] += int(pair_truncated)
            scan_limit = _number(cache_search.get("cache_cross_node_pair_scan_limit"))
            scanned = _number(
                cache_search.get("cache_cross_node_pair_proposals_scanned_count")
            )
            if scan_limit is not None and scanned is not None and scanned >= scan_limit:
                search_policy["cross_node_pair_scan_limit_touched_n"] += 1
            retention_omitted = int(
                _number(
                    cache_search.get(
                        "cache_cross_node_pair_retention_limit_omitted_count"
                    )
                )
                or 0
            )
            search_policy["cross_node_pair_retention_limit_omitted_n"] += int(
                retention_omitted > 0
            )
            search_policy["cross_node_pair_retention_limit_omitted_total"] += (
                retention_omitted
            )
            search_policy["cross_node_pair_unscanned_total"] += int(
                _number(cache_search.get("cache_cross_node_pair_unscanned_count"))
                or 0
            )
            search_policy["cross_node_pair_candidates_retained_total"] += int(
                _number(
                    cache_search.get("cache_cross_node_pair_candidates_retained_count")
                )
                or 0
            )

        uav_search = row.get("uav_candidate_generation_diagnostics")
        if isinstance(uav_search, Mapping):
            search_policy["uav_diagnostics_n"] += 1
            search_policy["incomplete_reason_counts"].update(
                str(reason) for reason in uav_search.get("incomplete_reasons", ())
            )
            if bool(uav_search.get("per_uav_limit_reached")):
                search_policy["uav_candidate_limit_reached_n"] += 1

        action = row.get("selected_candidate_action")
        target_execution_mismatch = False
        if isinstance(action, Mapping):
            state_hash = row.get("state_fingerprint")
            source_hash = row.get("candidate_source_state_fingerprint")
            if source_hash is None:
                source_hash = action.get("source_state_fingerprint")
            if state_hash is not None and source_hash is not None and state_hash != source_hash:
                anomalies["candidate_state_fingerprint_mismatch"] += 1
            if _action_id(row) != str(action.get("candidate_id")):
                target_execution_mismatch = True
            post_cache = row.get("post_cache_state")
            target_cache = action.get("target_cache_state")
            post_uav = row.get("post_uav_positions_m")
            target_uav = action.get("target_uav_positions")
            if post_cache is not None and target_cache is not None and post_cache != target_cache:
                target_execution_mismatch = True
            if post_uav is not None and target_uav is not None:
                expected_uav = {
                    str(item[0]): [float(item[1]), float(item[2])]
                    for item in target_uav
                }
                if post_uav != expected_uav:
                    target_execution_mismatch = True
        if row.get("candidate_current_legal_set_membership_pass") is False:
            anomalies["selected_action_not_marked_legal"] += 1
        if row.get("candidate_target_matches_execution_pass") is False:
            target_execution_mismatch = True
        anomalies["logged_target_execution_mismatch"] += int(target_execution_mismatch)
        if any(
            row.get(field) is False
            for field in (
                "candidate_mutual_exclusion_pass",
                "candidate_cache_replace_rule_pass",
                "candidate_cache_capacity_pass",
                "candidate_uav_boundary_pass",
                "candidate_uav_safety_distance_pass",
                "candidate_uav_transition_constraints_pass",
            )
        ):
            anomalies["selected_action_not_marked_constraint_valid"] += 1
        if _selection_mode(row) == "greedy":
            reconstructed_argmax_pass = _greedy_argmax_result(row)
            recorded_argmax_pass = row.get("greedy_global_argmax_pass")
            if reconstructed_argmax_pass is not None:
                greedy_argmax_check_count += 1
                anomalies["greedy_global_argmax_failure"] += int(
                    not reconstructed_argmax_pass
                )
                if (
                    isinstance(recorded_argmax_pass, bool)
                    and recorded_argmax_pass != reconstructed_argmax_pass
                ):
                    anomalies[
                        "greedy_global_argmax_flag_reconstruction_mismatch"
                    ] += 1
            elif isinstance(recorded_argmax_pass, bool):
                greedy_argmax_check_count += 1
                anomalies["greedy_global_argmax_failure"] += int(
                    not recorded_argmax_pass
                )
            else:
                greedy_argmax_unverifiable_count += 1

    action_summary: dict[str, Any] = {}
    outcome_summary: dict[str, Any] = {}
    for operation, operation_rows in by_operation.items():
        selection_n = sum(selection_mode_counts[operation].values())
        changes = changed_counts[operation]
        action_summary[operation] = {
            "n": len(operation_rows),
            "exploration_n": selection_mode_counts[operation]["exploration"],
            "greedy_n": selection_mode_counts[operation]["greedy"],
            "exploration_share": (
                selection_mode_counts[operation]["exploration"] / selection_n
                if selection_n
                else None
            ),
            "greedy_share": (
                selection_mode_counts[operation]["greedy"] / selection_n
                if selection_n
                else None
            ),
            "actual_state_change_n": changes["changed"],
            "actual_state_unchanged_n": changes["unchanged"],
            "actual_state_change_rate": (
                changes["changed"] / changes["n"] if changes["n"] else None
            ),
            "candidate_counts_and_state_q_distributions": _state_level_q_summary(
                rows, operation
            ),
            "realized_system_utility_sign": reward_signs[operation],
        }
        outcome_summary[operation] = {
            "n": len(operation_rows),
            "actual_system_utility": _summarize(
                value for value in (_reward(row) for row in operation_rows) if value is not None
            ),
            "actual_task_utility": _summarize(
                value
                for value in (_number(row.get("actual_task_utility")) for row in operation_rows)
                if value is not None
            ),
            "cache_cost_raw": _summarize(
                value
                for value in (
                    _number(row.get("cache_update_cost", row.get("cache_download_cost")))
                    for row in operation_rows
                    if row.get("cache_update_cost_available") is not False
                )
                if value is not None
            ),
            "cache_cost_weighted": _summarize(
                value
                for value in (_number(row.get("cache_cost_weighted")) for row in operation_rows)
                if value is not None
            ),
            "movement_cost_raw": _summarize(
                value
                for value in (
                    _number(row.get("uav_movement_cost"))
                    for row in operation_rows
                    if row.get("uav_movement_cost_available") is not False
                )
                if value is not None
            ),
            "movement_cost_weighted": _summarize(
                value
                for value in (_number(row.get("uav_movement_cost_weighted")) for row in operation_rows)
                if value is not None
            ),
            "actual_task_delay_sum_s": _summarize(
                value
                for value in (_number(row.get("actual_total_task_delay_s")) for row in operation_rows)
                if value is not None
            ),
            "actual_vehicle_energy_sum_j": _summarize(
                value
                for value in (_number(row.get("actual_total_vehicle_energy_j")) for row in operation_rows)
                if value is not None
            ),
            "same_slot_matching_comparison_n": sum(
                row.get("same_slot_matching_alternative_comparison_available") is True
                for row in operation_rows
            ),
            "actual_execution_node_map_n": sum(
                row.get("actual_execution_node_data_available") is True
                and isinstance(row.get("actual_execution_nodes"), list)
                for row in operation_rows
            ),
            "actual_execution_node_task_counts": dict(
                sorted(
                    Counter(
                        str(item["execution_node"])
                        for row in operation_rows
                        if row.get("actual_execution_node_data_available") is True
                        and isinstance(row.get("actual_execution_nodes"), list)
                        for item in row["actual_execution_nodes"]
                        if isinstance(item, Mapping) and item.get("execution_node") is not None
                    ).items()
                )
            ),
        }

    return {
        "slot_n": len(rows),
        "action_summary_by_operation": action_summary,
        "q_choice_review": {
            "selected_minus_keep_stay_q": _summarize(keep_q_gaps),
            "selected_minus_best_candidate_q_by_operation": {
                operation: _summarize(values)
                for operation, values in q_gaps_by_operation.items()
            },
            "greedy_argmax_check_n": greedy_argmax_check_count,
            "greedy_argmax_failure_n": anomalies["greedy_global_argmax_failure"],
            "greedy_argmax_unverifiable_n": greedy_argmax_unverifiable_count,
        },
        "outcome_summary_by_operation": outcome_summary,
        "reward_decomposition": {
            "actual_system_utility": _summarize(system_utility),
            "actual_task_utility": _summarize(task_utility),
            "cache_cost_weighted": _summarize(cache_cost_weighted),
            "movement_cost_weighted": _summarize(movement_cost_weighted),
            "task_delay_sum_s": _summarize(task_delay),
            "vehicle_energy_sum_j": _summarize(vehicle_energy),
            "reward_recomposition_residual": _summarize(recomposition_residuals),
            "cost_unavailable_n": cost_unavailable,
            "beta_ca_values": sorted(
                {
                    value
                    for row in rows
                    if (value := _number(row.get("beta_ca"))) is not None
                }
            ),
            "beta_q_values": sorted(
                {
                    value
                    for row in rows
                    if (value := _number(row.get("beta_q"))) is not None
                }
            ),
        },
        "predicted_vs_actual": {
            "raw_predicted_coverage_delta_n": raw_prediction_delta_count,
            "explicitly_scoped_prediction_task_utility_pair_n": len(scoped_prediction_pairs),
            "pearson_r_between_predicted_coverage_delta_and_actual_task_utility_level": (
                _pearson_correlation(scoped_prediction_pairs)
            ),
            "rows_missing_position_aware_coverage_scope_n": prediction_scope_missing_count,
            "actual_task_utility_delta_vs_keep_stay": "unavailable without same-slot counterfactual",
            "interpretation": (
                "A cross-slot correlation of prediction deltas with realized utility levels is descriptive only; "
                "it is not an action effect or a paired utility change."
            ),
        },
        "candidate_search": {
            **truncation,
            "truncation_rate": truncation["truncated"] / truncation["n"] if truncation["n"] else None,
            "candidate_q_distributions_are_state_weighted_summaries": True,
            "q_values_reconstructed_from_category_value": False,
            "search_policy_diagnostics": {
                **{
                    key: value
                    for key, value in search_policy.items()
                    if key
                    not in {
                        "cache_generated_candidate_counts",
                        "cache_candidate_limits",
                        "incomplete_reason_counts",
                    }
                },
                "cache_generated_candidate_count_including_keep": _summarize(
                    search_policy["cache_generated_candidate_counts"]
                ),
                "cache_candidate_limit_including_keep": _summarize(
                    search_policy["cache_candidate_limits"]
                ),
                "incomplete_reason_counts": dict(
                    sorted(search_policy["incomplete_reason_counts"].items())
                ),
            },
        },
        "anomalies": anomalies,
        "missing_field_row_counts": missing_fields,
        "posthoc_limitations": {
            "same_slot_matching_alternative_comparison_available": False,
            "actual_system_utility_sign_is_not_incremental_gain": True,
            "counterfactual_long_horizon_return_available": False,
        },
    }


def _format_number(value: Any) -> str:
    number = _number(value)
    return "unavailable" if number is None else f"{number:.6g}"


def _render_markdown(analyses: list[dict[str, Any]]) -> str:
    total_rows = sum(item["analysis"]["slot_n"] for item in analyses)
    lines = [
        "# Online Joint DQN decision audit",
        "",
        f"Analyzed Slot rows: **{total_rows}** across **{len(analyses)}** run(s).",
        "",
        "This is a retrospective log review. A positive realized system utility is not an incremental action gain. "
        "No same-Slot KEEP_STAY baseline exists in these logs, so no causal gain or long-horizon claim is made.",
        "",
        "## Runs",
        "",
        "| Run | Slot rows | Malformed lines |",
        "|---|---:|---:|",
    ]
    for item in analyses:
        lines.append(
            f"| `{item['run']}` | {item['analysis']['slot_n']} | {item['malformed_lines']} |"
        )
    lines.extend(
        [
            "",
            "## Actions, selection mode, and state changes",
            "",
            "| Operation | n | Explore n (%) | Greedy n (%) | State changed n / observed n | Change rate | No-op count |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    combined_actions = {
        operation: {"n": 0, "exploration_n": 0, "greedy_n": 0, "changed": 0, "change_n": 0, "no_op": 0}
        for operation in OPERATIONS
    }
    for item in analyses:
        for operation, values in item["analysis"]["action_summary_by_operation"].items():
            aggregate = combined_actions[operation]
            aggregate["n"] += values["n"]
            aggregate["exploration_n"] += values["exploration_n"]
            aggregate["greedy_n"] += values["greedy_n"]
            aggregate["changed"] += values["actual_state_change_n"]
            aggregate["change_n"] += values["actual_state_change_n"] + values["actual_state_unchanged_n"]
            aggregate["no_op"] += values["actual_state_unchanged_n"]
    for operation, values in combined_actions.items():
        mode_n = values["exploration_n"] + values["greedy_n"]
        explore_share = values["exploration_n"] / mode_n if mode_n else None
        greedy_share = values["greedy_n"] / mode_n if mode_n else None
        change_rate = values["changed"] / values["change_n"] if values["change_n"] else None
        lines.append(
            f"| {operation} | {values['n']} | {values['exploration_n']} ({_format_number(None if explore_share is None else 100.0 * explore_share)}%) "
            f"| {values['greedy_n']} ({_format_number(None if greedy_share is None else 100.0 * greedy_share)}%) | {values['changed']} / {values['change_n']} "
            f"| {_format_number(change_rate)} | {values['no_op']} |"
        )

    lines.extend(
        [
            "",
            "## Q-value choice evidence",
            "",
            "Differences use complete candidate Q values from each operation's `max`; category values are not used as action maxima.",
            "",
            "| Run | n | selected - KEEP Q (n / mean) | selected - best CACHE Q (n / mean) | selected - best UAV Q (n / mean) | selected - best KEEP Q (n / mean) | Greedy argmax failures |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in analyses:
        analysis = item["analysis"]
        q = analysis["q_choice_review"]
        gaps = q["selected_minus_best_candidate_q_by_operation"]
        lines.append(
            f"| `{item['run']}` | {analysis['slot_n']} | {q['selected_minus_keep_stay_q']['n']} / {_format_number(q['selected_minus_keep_stay_q']['mean'])} "
            f"| {gaps['CACHE_UPDATE']['n']} / {_format_number(gaps['CACHE_UPDATE']['mean'])} "
            f"| {gaps['UAV_MOVE']['n']} / {_format_number(gaps['UAV_MOVE']['mean'])} "
            f"| {gaps['KEEP_STAY']['n']} / {_format_number(gaps['KEEP_STAY']['mean'])} "
            f"| {q['greedy_argmax_failure_n']} |"
        )

    lines.extend(
        [
            "",
            "### Candidate counts and state-level Q distributions",
            "",
            "Candidate counts summarize each Slot's legal set; Q statistics summarize each Slot's logged candidate distribution.",
            "",
            "| Run | Operation | Slots with count | Candidate count mean [min, max] | Slots with Q stats | Mean of per-Slot Q means | Mean of per-Slot best Qs |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in analyses:
        for operation, values in item["analysis"]["action_summary_by_operation"].items():
            candidate_values = values["candidate_counts_and_state_q_distributions"]["candidate_count_per_slot"]
            q_values = values["candidate_counts_and_state_q_distributions"]["state_weighted_q_statistics"]
            lines.append(
                f"| `{item['run']}` | {operation} | {candidate_values['n']} | "
                f"{_format_number(candidate_values['mean'])} [{_format_number(candidate_values['min'])}, {_format_number(candidate_values['max'])}] | "
                f"{q_values['mean']['n']} | {_format_number(q_values['mean']['mean'])} | {_format_number(q_values['max']['mean'])} |"
            )
    lines.extend(
        [
            "",
            "## Realized system outcome and costs",
            "",
            "Values are summaries of the selected action's formal executor output. `n` is the number with that field available.",
            "",
            "| Run | Operation | n | Realized G mean | Task utility mean | Weighted cache cost mean | Weighted movement cost mean | Delay sum mean (s) | Vehicle energy sum mean (J) | Matching maps n |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in analyses:
        for operation, values in item["analysis"]["outcome_summary_by_operation"].items():
            lines.append(
                f"| `{item['run']}` | {operation} | {values['n']} | {_format_number(values['actual_system_utility']['mean'])} "
                f"| {_format_number(values['actual_task_utility']['mean'])} | {_format_number(values['cache_cost_weighted']['mean'])} "
                f"| {_format_number(values['movement_cost_weighted']['mean'])} | {_format_number(values['actual_task_delay_sum_s']['mean'])} "
                f"| {_format_number(values['actual_vehicle_energy_sum_j']['mean'])} "
                f"| {values['actual_execution_node_map_n']} |"
            )
    lines.extend(
        [
            "",
            "Actual Matching execution-node assignments by operation (task counts; not a same-Slot comparison):",
        ]
    )
    for item in analyses:
        for operation, values in item["analysis"]["outcome_summary_by_operation"].items():
            lines.append(
                f"- `{item['run']}` {operation}: maps n={values['actual_execution_node_map_n']}; "
                f"task counts={json.dumps(values['actual_execution_node_task_counts'], sort_keys=True)}"
            )
    lines.extend(
        [
            "",
            "### Reward weights and recomposition",
            "",
            "The logged reward is checked as `G = Q_task - beta_ca*C_ca - beta_q*C_q`; normalized cache and movement costs are dimensionless ratios.",
            "",
            "| Run | beta_ca values | beta_q values in executed Slots | Recomposition residual n / mean / max abs |",
            "|---|---|---|---:|",
        ]
    )
    for item in analyses:
        reward = item["analysis"]["reward_decomposition"]
        residual = reward["reward_recomposition_residual"]
        max_abs_residual = (
            None
            if residual["n"] == 0
            else max(abs(float(residual["min"])), abs(float(residual["max"])))
        )
        lines.append(
            f"| `{item['run']}` | {reward['beta_ca_values']} | {reward['beta_q_values']} "
            f"| {residual['n']} / {_format_number(residual['mean'])} / {_format_number(max_abs_residual)} |"
        )
    lines.extend(
        [
            "",
            "`Positive / zero / negative` below classifies the realized system utility sign, not utility improvement versus KEEP_STAY.",
            "",
            "| Run | Operation | n | Positive n (%) | Zero n (%) | Negative n (%) | Same-Slot matching comparison n |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in analyses:
        for operation, values in item["analysis"]["outcome_summary_by_operation"].items():
            signs = item["analysis"]["action_summary_by_operation"][operation]["realized_system_utility_sign"]
            sign_n = int(signs["n"])
            lines.append(
                f"| `{item['run']}` | {operation} | {sign_n} | {signs['positive']} ({_format_number(100.0 * signs['positive'] / sign_n if sign_n else None)}%) "
                f"| {signs['zero']} ({_format_number(100.0 * signs['zero'] / sign_n if sign_n else None)}%) "
                f"| {signs['negative']} ({_format_number(100.0 * signs['negative'] / sign_n if sign_n else None)}%) "
                f"| {values['same_slot_matching_comparison_n']} |"
            )

    lines.extend(
        [
            "",
            "## Prediction and candidate search",
            "",
            "| Run | Raw predicted coverage deltas | Position-aware scoped pairs | Pearson r with actual task-utility level | Search truncated n / slots | Bounded complete n | Exhaustive legal set n |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in analyses:
        analysis = item["analysis"]
        prediction = analysis["predicted_vs_actual"]
        search = analysis["candidate_search"]
        lines.append(
            f"| `{item['run']}` | {prediction['raw_predicted_coverage_delta_n']} | "
            f"{prediction['explicitly_scoped_prediction_task_utility_pair_n']} | "
            f"{_format_number(prediction['pearson_r_between_predicted_coverage_delta_and_actual_task_utility_level'])} | "
            f"{search['truncated']} / {search['n']} | {search['bounded_complete']} | {search['exhaustive']} |"
        )
    lines.extend(
        [
            "",
            "### Search budget diagnostics",
            "",
            "| Run | Cache candidates generated / cap (mean) | Cache cap reached | Cross-node pairs truncated | Scan cap touched | Unscanned pairs total | Pair retention omitted total | UAV cap reached |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in analyses:
        policy = item["analysis"]["candidate_search"]["search_policy_diagnostics"]
        generated = policy["cache_generated_candidate_count_including_keep"]
        limit = policy["cache_candidate_limit_including_keep"]
        lines.append(
            f"| `{item['run']}` | {_format_number(generated['mean'])} / {_format_number(limit['mean'])} "
            f"(n={generated['n']}) | {policy['cache_candidate_budget_reached_n']} / {policy['cache_diagnostics_n']} "
            f"| {policy['cross_node_pair_truncated_n']} / {policy['cache_diagnostics_n']} "
            f"| {policy['cross_node_pair_scan_limit_touched_n']} / {policy['cache_diagnostics_n']} "
            f"| {policy['cross_node_pair_unscanned_total']} "
            f"| {policy['cross_node_pair_retention_limit_omitted_total']} "
            f"| {policy['uav_candidate_limit_reached_n']} / {policy['uav_diagnostics_n']} |"
        )
    for item in analyses:
        policy = item["analysis"]["candidate_search"]["search_policy_diagnostics"]
        reasons = policy["incomplete_reason_counts"]
        lines.append(
            f"- `{item['run']}` bounded-search reasons: "
            + (
                ", ".join(f"`{reason}` {count}" for reason, count in reasons.items())
                or "unavailable"
            )
        )
    lines.extend(
        [
            "",
            "The correlation uses cross-Slot realized task-utility levels and is descriptive only. It is omitted unless the log marks the coverage calculation as using current state UAV positions. Historical runs without that scope marker are not treated as position-valid, and their actual task-utility change versus KEEP_STAY is unavailable.",
            "",
            "Candidate counts and Q summaries are state-weighted summaries of the per-Slot aggregate statistics; they do not reconstruct an unlogged full candidate tensor.",
            "",
            "## Missing fields and anomalies",
            "",
        ]
    )
    for item in analyses:
        analysis = item["analysis"]
        lines.append(f"### `{item['run']}`")
        lines.append("")
        lines.append(
            "- Missing field row counts: "
            + (
                ", ".join(
                    f"`{name}` {count}/{analysis['slot_n']}"
                    for name, count in sorted(analysis["missing_field_row_counts"].items())
                )
                or "none"
            )
        )
        lines.append(
            "- Cost records unavailable: "
            f"cache {analysis['reward_decomposition']['cost_unavailable_n']['cache']}, "
            f"movement {analysis['reward_decomposition']['cost_unavailable_n']['movement']}"
        )
        lines.append(
            "- Anomaly counters: "
            + ", ".join(
                f"{name}={count}"
                for name, count in sorted(analysis["anomalies"].items())
            )
        )
        lines.append("")
    lines.append("Sample counts accompany all summaries. Small samples are observations only and do not establish causality or convergence.")
    return "\n".join(lines)


def analyze_run_directories(run_directories: Iterable[Path]) -> list[dict[str, Any]]:
    analyses: list[dict[str, Any]] = []
    for run_directory in sorted({path.resolve() for path in run_directories}):
        step_path = run_directory / "step_metrics.jsonl"
        if not step_path.is_file():
            continue
        rows, malformed_count = _read_step_rows(step_path)
        analyses.append(
            {
                "run": run_directory.name,
                "malformed_lines": malformed_count,
                "analysis": analyze_step_rows(rows),
            }
        )
    if not analyses:
        raise FileNotFoundError("No run directory with step_metrics.jsonl was found")
    return analyses


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        help="Run directory to review; may be supplied more than once.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "runs",
        help="Root searched when --run-dir is omitted.",
    )
    parser.add_argument(
        "--pattern",
        default="online_joint_dqn*",
        help="Directory glob under --runs-root when --run-dir is omitted.",
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_directories = args.run_dir
    if run_directories is None:
        run_directories = [
            path
            for path in args.runs_root.glob(args.pattern)
            if path.is_dir()
        ]
    analyses = analyze_run_directories(run_directories)
    if args.format == "json":
        print(json.dumps(analyses, ensure_ascii=False, indent=2))
    else:
        print(_render_markdown(analyses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
