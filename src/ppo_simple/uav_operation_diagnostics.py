"""Pure diagnostics for the proposed UAV STAY/MOVE exact-Q credit.

The functions in this module only consume arrays that are already produced by
``PPOV2Trainer._compute_exact_uav_operation_credit`` and the frozen rollout.
They must not call the environment, Matching, KKT, or any policy update path.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


DEFAULT_Q_EQUALITY_TOLERANCE = 1.0e-12


def _as_finite_array(value: Any, *, name: str, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(f"{name} contains NaN/Inf")
    return array


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    if values.size == 0:
        return None
    result = float(np.percentile(values, percentile))
    if not np.isfinite(result):
        raise FloatingPointError("UAV operation diagnostic percentile is non-finite")
    return result


def _mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    result = float(np.mean(values))
    if not np.isfinite(result):
        raise FloatingPointError("UAV operation diagnostic mean is non-finite")
    return result


def _median(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    result = float(np.median(values))
    if not np.isfinite(result):
        raise FloatingPointError("UAV operation diagnostic median is non-finite")
    return result


def _fraction(mask: np.ndarray) -> float | None:
    if mask.size == 0:
        return None
    result = float(np.mean(mask.astype(np.float64)))
    if not np.isfinite(result):
        raise FloatingPointError("UAV operation diagnostic fraction is non-finite")
    return result


def _aligned_rollout_arrays(rollout: Any) -> tuple[np.ndarray, np.ndarray]:
    transitions = tuple(rollout.transitions)
    if not transitions:
        raise ValueError("UAV operation diagnostics require a non-empty rollout")
    probability_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    for transition in transitions:
        if (
            transition.uav_operation_probabilities is None
            or transition.uav_operation_actions is None
        ):
            raise ValueError("rollout is missing UAV operation probabilities/actions")
        probabilities = np.asarray(
            transition.uav_operation_probabilities, dtype=np.float64
        )
        actions = np.asarray(transition.uav_operation_actions, dtype=np.int64)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError("UAV operation probabilities must have shape [heads, 2]")
        if actions.ndim != 1 or actions.shape[0] != probabilities.shape[0]:
            raise ValueError("UAV operation actions are not aligned with probabilities")
        if np.any(actions < 0) or np.any(actions > 1):
            raise ValueError("UAV operation actions must be STAY=0 or MOVE=1")
        if not np.all(np.isfinite(probabilities)):
            raise FloatingPointError("UAV operation probabilities contain NaN/Inf")
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise ValueError("UAV operation probabilities are outside [0, 1]")
        if not np.allclose(
            probabilities.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError("UAV operation probabilities do not sum to one")
        probability_rows.append(probabilities)
        action_rows.append(actions)
    probabilities_array = np.stack(probability_rows, axis=0)
    actions_array = np.stack(action_rows, axis=0)
    return probabilities_array, actions_array


def diagnostic_arrays_from_rollout(
    rollout: Any,
    credit: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Extract aligned, non-logged arrays from an existing exact-Q result."""

    probabilities, operation_actions = _aligned_rollout_arrays(rollout)
    q_stay = _as_finite_array(credit.get("q_stay"), name="q_stay", dtype=np.float64)
    q_move = _as_finite_array(credit.get("q_move"), name="q_move", dtype=np.float64)
    raw_advantages = _as_finite_array(
        credit.get("raw_advantages"),
        name="raw_advantages",
        dtype=np.float64,
    )
    candidate_q_values = _as_finite_array(
        credit.get("candidate_q_values"),
        name="candidate_q_values",
        dtype=np.float64,
    )
    candidate_probabilities = _as_finite_array(
        credit.get("candidate_probabilities"),
        name="candidate_probabilities",
        dtype=np.float64,
    )
    candidate_valid_mask = np.asarray(
        credit.get("candidate_valid_mask"), dtype=bool
    )
    operation_valid_mask_value = credit.get("counterfactual_operation_valid_mask")
    expected_shape = q_stay.shape
    if q_stay.ndim != 2 or q_move.shape != expected_shape:
        raise ValueError("exact operation Q arrays must have shape [samples, heads]")
    sample_count, head_count = [int(value) for value in expected_shape]
    expected_candidate_shape = (sample_count, head_count)
    if candidate_q_values.ndim != 3 or candidate_q_values.shape[:2] != expected_candidate_shape:
        raise ValueError("candidate Q array shape does not align with operation Q")
    if candidate_probabilities.shape != candidate_q_values.shape:
        raise ValueError("candidate probabilities shape does not align with candidate Q")
    if candidate_valid_mask.shape != candidate_q_values.shape:
        raise ValueError("candidate valid mask shape does not align with candidate Q")
    if operation_valid_mask_value is None:
        operation_valid_mask = np.ones(
            (sample_count, head_count, 2),
            dtype=bool,
        )
    else:
        operation_valid_mask = np.asarray(operation_valid_mask_value, dtype=bool)
    if operation_valid_mask.shape != (sample_count, head_count, 2):
        raise ValueError(
            "counterfactual operation validity mask does not align with exact operation Q"
        )
    if probabilities.shape != (sample_count, head_count, 2):
        raise ValueError(
            "rollout operation probabilities do not align with exact operation Q"
        )
    if operation_actions.shape != (sample_count, head_count):
        raise ValueError("rollout operation actions do not align with exact operation Q")
    if raw_advantages.shape != expected_shape:
        raise ValueError("exact operation raw advantages do not align with exact operation Q")
    legal_probability_mass = np.sum(
        np.where(candidate_valid_mask, candidate_probabilities, 0.0), axis=-1
    )
    active = candidate_valid_mask.any(axis=-1)
    if np.any(~np.isclose(legal_probability_mass[active], 1.0, atol=1.0e-5)):
        raise ValueError("legal UAV candidate probabilities do not sum to one")
    if np.any(~active & (legal_probability_mass != 0.0)):
        raise ValueError("inactive UAV candidate heads have non-zero probability mass")
    if np.any(candidate_probabilities < 0.0):
        raise ValueError("candidate probabilities must be non-negative")
    selected_operation_valid = operation_valid_mask[
        np.arange(sample_count)[:, None],
        np.arange(head_count)[None, :],
        operation_actions,
    ]
    if not bool(np.all(selected_operation_valid)):
        raise ValueError("production-selected UAV operation is not teacher-valid")
    return {
        "q_stay": q_stay,
        "q_move": q_move,
        "raw_advantages": raw_advantages,
        "candidate_q_values": candidate_q_values,
        "candidate_probabilities": candidate_probabilities,
        "candidate_valid_mask": candidate_valid_mask,
        "p_stay": probabilities[..., 0],
        "p_move": probabilities[..., 1],
        "operation_actions": operation_actions,
        "operation_valid_mask": operation_valid_mask,
    }


def _summary_from_arrays(
    arrays: dict[str, np.ndarray],
    *,
    q_equality_tolerance: float = DEFAULT_Q_EQUALITY_TOLERANCE,
) -> dict[str, Any]:
    tolerance = float(q_equality_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("q_equality_tolerance must be finite and non-negative")

    q_stay = arrays["q_stay"]
    q_move = arrays["q_move"]
    candidate_q_values = arrays["candidate_q_values"]
    candidate_valid_mask = arrays["candidate_valid_mask"]
    p_stay = arrays["p_stay"]
    p_move = arrays["p_move"]
    operation_actions = arrays["operation_actions"]
    operation_valid_mask = arrays.get(
        "operation_valid_mask",
        np.ones((*q_stay.shape, 2), dtype=bool),
    )
    sample_count, head_count = q_stay.shape
    total_samples = int(q_stay.size)
    if total_samples <= 0:
        raise ValueError("UAV operation diagnostics require at least one sample")
    if operation_valid_mask.shape != (sample_count, head_count, 2):
        raise ValueError("UAV operation validity mask shape mismatch")

    q_gap = q_move - q_stay
    stay_valid = operation_valid_mask[..., 0]
    move_valid = operation_valid_mask[..., 1]
    comparable = stay_valid & move_valid
    move_better = comparable & (q_gap > tolerance)
    stay_better = comparable & (q_gap < -tolerance)
    tie = comparable & ~(move_better | stay_better)
    comparable_count = int(comparable.sum())
    if int(move_better.sum() + stay_better.sum() + tie.sum()) != comparable_count:
        raise AssertionError("MOVE/STAY Q classification does not cover comparable samples")
    candidate_move_feasible = candidate_valid_mask.any(axis=-1)
    if not np.array_equal(candidate_move_feasible, move_valid):
        raise ValueError("candidate validity and operation validity are not aligned")

    move_feasible = move_valid
    move_feasible_count = int(move_feasible.sum())
    candidate_q_masked = np.where(
        candidate_valid_mask,
        candidate_q_values,
        -np.inf,
    )
    move_best = np.max(candidate_q_masked, axis=-1)
    move_best_feasible = move_best[move_feasible]
    move_policy_feasible = q_move[move_feasible]
    move_best_minus_policy = move_best_feasible - move_policy_feasible
    if move_best_feasible.size and not np.all(np.isfinite(move_best_feasible)):
        raise FloatingPointError("feasible UAV MOVE-best Q contains NaN/Inf")
    if move_best_minus_policy.size and not np.all(np.isfinite(move_best_minus_policy)):
        raise FloatingPointError("UAV target regret contains NaN/Inf")
    target_regret = np.maximum(move_best_minus_policy, 0.0)

    policy_probabilities = np.stack((p_stay, p_move), axis=-1)
    policy_probabilities = np.where(
        operation_valid_mask,
        policy_probabilities,
        0.0,
    )
    policy_probability_normalizer = policy_probabilities.sum(axis=-1, keepdims=True)
    if np.any(policy_probability_normalizer <= 0.0) or not np.all(
        np.isfinite(policy_probability_normalizer)
    ):
        raise ValueError("UAV operation diagnostic validity mask has empty probability mass")
    policy_probabilities = policy_probabilities / policy_probability_normalizer
    safe_q_stay = np.where(stay_valid, q_stay, 0.0)
    safe_q_move = np.where(move_valid, q_move, 0.0)
    expected_q = (
        policy_probabilities[..., 0] * safe_q_stay
        + policy_probabilities[..., 1] * safe_q_move
    )
    oracle_q = np.where(
        comparable,
        np.maximum(safe_q_stay, safe_q_move),
        np.where(move_valid, safe_q_move, safe_q_stay),
    )
    operation_regret = np.maximum(oracle_q - expected_q, 0.0)
    if not np.all(np.isfinite(expected_q)) or not np.all(np.isfinite(operation_regret)):
        raise FloatingPointError("UAV operation expected Q/regret contains NaN/Inf")

    # np.argmax provides a stable STAY tie-break, matching the operation index
    # order in the proposed policy (STAY=0, MOVE=1).  Q ties are still exposed
    # separately through the exact-Q tie count/fraction.
    policy_argmax_move = policy_probabilities[..., 1] > policy_probabilities[..., 0]
    q_argmax_move = q_move > q_stay + tolerance
    policy_q_agreement = comparable & (policy_argmax_move == q_argmax_move)
    policy_stay_when_move_better = move_better & (policy_probabilities[..., 1] < 0.5)
    move_better_pmove_lt_010 = move_better & (policy_probabilities[..., 1] < 0.10)
    move_better_pmove_lt_001 = move_better & (policy_probabilities[..., 1] < 0.01)

    conditional_policy_bottleneck = move_feasible & (
        comparable
        & (move_best > q_stay + tolerance)
        & (q_move <= q_stay + tolerance)
    )
    move_policy_better = comparable & (q_move > q_stay + tolerance)
    move_best_better = comparable & move_feasible & (
        move_best > q_stay + tolerance
    )

    raw_advantages = arrays["raw_advantages"]
    move_advantages = raw_advantages[operation_actions == 1]
    stay_advantages = raw_advantages[operation_actions == 0]
    comparable_q_gap = q_gap[comparable]
    comparable_count = int(comparable_q_gap.size)
    comparable_move_better_count = int(move_better.sum())
    comparable_stay_better_count = int(stay_better.sum())
    comparable_tie_count = int(tie.sum())
    comparable_policy_agreement_count = int(policy_q_agreement.sum())
    comparable_policy_disagreement_count = int(
        (comparable & ~policy_q_agreement).sum()
    )
    move_better_denominator = max(comparable_count, 1)
    move_best_comparison_mask = move_feasible & comparable
    move_best_comparison_count = int(move_best_comparison_mask.sum())

    summary: dict[str, Any] = {
        "UAV_OPERATION_DIAGNOSTIC_SAMPLE_COUNT": total_samples,
        "UAV_OPERATION_DIAGNOSTIC_TRANSITION_COUNT": sample_count,
        "UAV_OPERATION_DIAGNOSTIC_UAV_HEAD_COUNT": head_count,
        "UAV_OPERATION_Q_EQUALITY_TOLERANCE": tolerance,
        "UAV_OPERATION_MOVE_FEASIBLE_COUNT": move_feasible_count,
        "UAV_OPERATION_MOVE_FEASIBLE_RATE": float(move_feasible.mean()),
        "UAV_MOVE_FEASIBLE_RATE": float(move_feasible.mean()),
        "UAV_P_MOVE_MEAN": float(np.mean(p_move)),
        "UAV_P_MOVE_MEDIAN": float(np.median(p_move)),
        "UAV_P_MOVE_P10": float(np.percentile(p_move, 10.0)),
        "UAV_P_MOVE_P90": float(np.percentile(p_move, 90.0)),
        "UAV_OPERATION_Q_COMPARABLE_COUNT": comparable_count,
        "UAV_OPERATION_Q_GAP_MEAN": _mean(comparable_q_gap),
        "UAV_OPERATION_Q_GAP_MEDIAN": _median(comparable_q_gap),
        "UAV_OPERATION_Q_GAP_P10": _percentile(comparable_q_gap, 10.0),
        "UAV_OPERATION_Q_GAP_P25": _percentile(comparable_q_gap, 25.0),
        "UAV_OPERATION_Q_GAP_P75": _percentile(comparable_q_gap, 75.0),
        "UAV_OPERATION_Q_GAP_P90": _percentile(comparable_q_gap, 90.0),
        "UAV_OPERATION_Q_GAP_MIN": (
            None if comparable_count == 0 else float(np.min(comparable_q_gap))
        ),
        "UAV_OPERATION_Q_GAP_MAX": (
            None if comparable_count == 0 else float(np.max(comparable_q_gap))
        ),
        "UAV_OPERATION_MOVE_BETTER_COUNT": comparable_move_better_count,
        "UAV_OPERATION_MOVE_BETTER_FRACTION": (
            None
            if comparable_count == 0
            else float(comparable_move_better_count / move_better_denominator)
        ),
        "UAV_OPERATION_STAY_BETTER_COUNT": comparable_stay_better_count,
        "UAV_OPERATION_STAY_BETTER_FRACTION": (
            None
            if comparable_count == 0
            else float(comparable_stay_better_count / move_better_denominator)
        ),
        "UAV_OPERATION_TIE_COUNT": comparable_tie_count,
        "UAV_OPERATION_TIE_FRACTION": (
            None
            if comparable_count == 0
            else float(comparable_tie_count / move_better_denominator)
        ),
        "UAV_OPERATION_POLICY_Q_ARGMAX_AGREEMENT_COUNT": comparable_policy_agreement_count,
        "UAV_OPERATION_POLICY_Q_ARGMAX_AGREEMENT_RATE": (
            None
            if comparable_count == 0
            else float(comparable_policy_agreement_count / move_better_denominator)
        ),
        "UAV_OPERATION_POLICY_Q_ARGMAX_DISAGREEMENT_COUNT": comparable_policy_disagreement_count,
        "UAV_OPERATION_POLICY_Q_ARGMAX_DISAGREEMENT_RATE": (
            None
            if comparable_count == 0
            else float(comparable_policy_disagreement_count / move_better_denominator)
        ),
        "UAV_OPERATION_Q_ARGMAX_TIE_COUNT": comparable_tie_count,
        "UAV_MOVE_BETTER_BUT_POLICY_STAY_COUNT": int(
            policy_stay_when_move_better.sum()
        ),
        "UAV_MOVE_BETTER_BUT_POLICY_STAY_FRACTION": (
            None
            if comparable_count == 0
            else float(policy_stay_when_move_better.sum() / move_better_denominator)
        ),
        "UAV_MOVE_BETTER_BUT_POLICY_STAY_AMONG_MOVE_BETTER_FRACTION": (
            None
            if not bool(move_better.any())
            else float(
                policy_stay_when_move_better.sum() / float(move_better.sum())
            )
        ),
        "UAV_MOVE_BETTER_BUT_PMOVE_LT_0_10_COUNT": int(
            move_better_pmove_lt_010.sum()
        ),
        "UAV_MOVE_BETTER_BUT_PMOVE_LT_0_10_FRACTION": (
            None
            if comparable_count == 0
            else float(move_better_pmove_lt_010.sum() / move_better_denominator)
        ),
        "UAV_MOVE_BETTER_BUT_PMOVE_LT_0_10_AMONG_MOVE_BETTER_FRACTION": (
            None
            if not bool(move_better.any())
            else float(move_better_pmove_lt_010.sum() / float(move_better.sum()))
        ),
        "UAV_MOVE_BETTER_BUT_PMOVE_LT_0_01_COUNT": int(
            move_better_pmove_lt_001.sum()
        ),
        "UAV_MOVE_BETTER_BUT_PMOVE_LT_0_01_FRACTION": (
            None
            if comparable_count == 0
            else float(move_better_pmove_lt_001.sum() / move_better_denominator)
        ),
        "UAV_MOVE_BETTER_BUT_PMOVE_LT_0_01_AMONG_MOVE_BETTER_FRACTION": (
            None
            if not bool(move_better.any())
            else float(move_better_pmove_lt_001.sum() / float(move_better.sum()))
        ),
        "UAV_OPERATION_EXPECTED_Q_CURRENT": _mean(expected_q.reshape(-1)),
        "UAV_OPERATION_EXPECTED_EXACT_Q_MEAN": _mean(expected_q.reshape(-1)),
        "UAV_OPERATION_ORACLE_EXACT_Q_MEAN": _mean(oracle_q.reshape(-1)),
        "UAV_OPERATION_EXACT_Q_REGRET_MEAN": _mean(operation_regret.reshape(-1)),
        "UAV_OPERATION_EXACT_Q_REGRET_MEDIAN": _median(operation_regret.reshape(-1)),
        "UAV_OPERATION_EXACT_Q_REGRET_P90": _percentile(
            operation_regret.reshape(-1), 90.0
        ),
        "UAV_OPERATION_EXACT_Q_REGRET_MAX": float(np.max(operation_regret)),
        "UAV_OPERATION_EXPECTED_Q_POLICY_MOVE_WEIGHTED_MEAN": _mean(
            (policy_probabilities[..., 1] * safe_q_move).reshape(-1)
        ),
        "UAV_MOVE_POLICY_Q_MEAN": _mean(q_move[move_valid]),
        "UAV_MOVE_BEST_Q_MEAN": _mean(move_best_feasible),
        "UAV_MOVE_BEST_MINUS_POLICY_Q_MEAN": _mean(move_best_minus_policy),
        "UAV_MOVE_BEST_MINUS_POLICY_Q_MEDIAN": _median(move_best_minus_policy),
        "UAV_MOVE_BEST_MINUS_POLICY_Q_P90": _percentile(
            move_best_minus_policy, 90.0
        ),
        "UAV_MOVE_POLICY_BETTER_THAN_STAY_COUNT": int(move_policy_better.sum()),
        "UAV_MOVE_POLICY_BETTER_THAN_STAY_FRACTION": (
            None
            if comparable_count == 0
            else float(move_policy_better.sum() / move_better_denominator)
        ),
        "UAV_MOVE_BEST_BETTER_THAN_STAY_COUNT": int(move_best_better.sum()),
        "UAV_MOVE_BEST_BETTER_THAN_STAY_FRACTION": (
            None
            if move_best_comparison_count == 0
            else float(move_best_better.sum() / float(move_best_comparison_count))
        ),
        "UAV_MOVE_CONDITIONAL_POLICY_BOTTLENECK_COUNT": int(
            conditional_policy_bottleneck.sum()
        ),
        "UAV_MOVE_CONDITIONAL_POLICY_BOTTLENECK_FRACTION": (
            None
            if move_best_comparison_count == 0
            else float(
                conditional_policy_bottleneck.sum() / float(move_best_comparison_count)
            )
        ),
        "UAV_MOVE_TARGET_REGRET_MEAN": _mean(target_regret),
        "UAV_MOVE_TARGET_REGRET_MEDIAN": _median(target_regret),
        "UAV_MOVE_TARGET_REGRET_P90": _percentile(target_regret, 90.0),
        "UAV_MOVE_TARGET_REGRET_MAX": (
            None if target_regret.size == 0 else float(np.max(target_regret))
        ),
        "uav_exact_operation_move_advantage_mean": _mean(move_advantages),
        "uav_exact_operation_stay_advantage_mean": _mean(stay_advantages),
        "uav_exact_operation_move_advantage_count": int(move_advantages.size),
        "uav_exact_operation_stay_advantage_count": int(stay_advantages.size),
        "UAV_OPERATION_DIAGNOSTIC_CONDITIONAL_DENOMINATOR": (
            "MOVE_FEASIBLE_SAMPLES"
        ),
        "UAV_OPERATION_DIAGNOSTIC_ALL_SAMPLE_DENOMINATOR": total_samples,
    }

    for head_index in range(head_count):
        head_prefix = f"UAV_{head_index}"
        head_q_gap = q_gap[:, head_index]
        head_move_better = move_better[:, head_index]
        head_move_feasible = move_feasible[:, head_index]
        head_move_best_better = move_best_better[:, head_index]
        head_bottleneck = conditional_policy_bottleneck[:, head_index]
        head_regret = operation_regret[:, head_index]
        head_feasible_count = int(head_move_feasible.sum())
        head_comparable = comparable[:, head_index]
        head_comparable_count = int(head_comparable.sum())
        head_move_best_comparison = move_best_comparison_mask[:, head_index]
        head_move_best_comparison_count = int(head_move_best_comparison.sum())
        summary[f"{head_prefix}_MOVE_BETTER_FRACTION"] = (
            None
            if head_comparable_count == 0
            else float(head_move_better[head_comparable].mean())
        )
        summary[f"{head_prefix}_P_MOVE_MEAN"] = float(np.mean(p_move[:, head_index]))
        summary[f"{head_prefix}_P_MOVE_MEDIAN"] = float(
            np.median(p_move[:, head_index])
        )
        summary[f"{head_prefix}_Q_GAP_MEAN"] = _mean(head_q_gap[head_comparable])
        summary[f"{head_prefix}_OPERATION_REGRET_MEAN"] = _mean(head_regret)
        summary[f"{head_prefix}_CONDITIONAL_BOTTLENECK_FRACTION"] = (
            None
            if head_move_best_comparison_count == 0
            else float(
                head_bottleneck.sum() / float(head_move_best_comparison_count)
            )
        )
        summary[f"{head_prefix}_MOVE_FEASIBLE_RATE"] = float(
            head_move_feasible.mean()
        )
        summary[f"{head_prefix}_MOVE_BEST_BETTER_THAN_STAY_FRACTION"] = (
            None
            if head_move_best_comparison_count == 0
            else float(
                head_move_best_better.sum() / float(head_move_best_comparison_count)
            )
        )
        summary[f"{head_prefix}_CONDITIONAL_DENOMINATOR_COUNT"] = (
            head_move_best_comparison_count
        )
        summary[f"{head_prefix}_Q_COMPARABLE_COUNT"] = head_comparable_count

    bucket_masks = {
        "P_MOVE_GE_0_50": p_move >= 0.5,
        "P_MOVE_0_10_TO_0_50": (p_move >= 0.1) & (p_move < 0.5),
        "P_MOVE_0_01_TO_0_10": (p_move >= 0.01) & (p_move < 0.1),
        "P_MOVE_LT_0_01": p_move < 0.01,
    }
    bucket_records: dict[str, dict[str, Any]] = {}
    for bucket_name, bucket_mask in bucket_masks.items():
        flat_mask = bucket_mask.reshape(-1)
        flat_comparison_mask = (bucket_mask & comparable).reshape(-1)
        bucket_q_gap = q_gap.reshape(-1)[flat_comparison_mask]
        bucket_move_better = move_better.reshape(-1)[flat_comparison_mask]
        bucket_regret = operation_regret.reshape(-1)[flat_mask]
        bucket_records[bucket_name] = {
            "sample_count": int(flat_mask.sum()),
            "q_comparable_count": int(flat_comparison_mask.sum()),
            "Q_GAP_MEAN": _mean(bucket_q_gap),
            "MOVE_BETTER_FRACTION": _fraction(bucket_move_better),
            "OPERATION_REGRET_MEAN": _mean(bucket_regret),
        }
    summary["UAV_OPERATION_P_MOVE_BUCKETS"] = bucket_records
    return summary


def compute_uav_operation_diagnostics(
    rollout: Any,
    credit: dict[str, Any],
    *,
    q_equality_tolerance: float = DEFAULT_Q_EQUALITY_TOLERANCE,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute compact diagnostics and return private arrays for episode merging."""

    arrays = diagnostic_arrays_from_rollout(rollout, credit)
    summary = _summary_from_arrays(
        arrays,
        q_equality_tolerance=q_equality_tolerance,
    )
    return summary, arrays


def merge_uav_operation_diagnostic_arrays(
    records: Iterable[dict[str, np.ndarray]],
    *,
    q_equality_tolerance: float = DEFAULT_Q_EQUALITY_TOLERANCE,
) -> dict[str, Any] | None:
    """Merge chunk-local private arrays into one episode-level summary."""

    materialized = list(records)
    if not materialized:
        return None
    keys = (
        "q_stay",
        "q_move",
        "raw_advantages",
        "candidate_q_values",
        "candidate_probabilities",
        "candidate_valid_mask",
        "p_stay",
        "p_move",
        "operation_actions",
        "operation_valid_mask",
    )
    for key in keys:
        if any(key not in record for record in materialized):
            if key == "operation_valid_mask":
                for record in materialized:
                    sample_count, head_count = np.asarray(record["q_stay"]).shape
                    record[key] = np.ones(
                        (sample_count, head_count, 2),
                        dtype=bool,
                    )
                continue
            raise ValueError(f"UAV operation diagnostic merge is missing {key}")
    candidate_shapes = {
        tuple(np.asarray(record["candidate_q_values"]).shape[2:])
        for record in materialized
    }
    if len(candidate_shapes) != 1:
        raise ValueError("UAV candidate capacity changed within one episode")
    arrays = {
        key: np.concatenate(
            [np.asarray(record[key]) for record in materialized],
            axis=0,
        )
        for key in keys
    }
    return _summary_from_arrays(
        arrays,
        q_equality_tolerance=q_equality_tolerance,
    )
