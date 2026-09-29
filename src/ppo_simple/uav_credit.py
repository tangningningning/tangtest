"""汇总 proposed_joint 的 UAV STAY/MOVE Exact-Q 信用信息。

本模块只处理 Exact-Q 和 rollout 已生成的数据，不调用环境、Matching、KKT
或策略更新逻辑。
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - 仅在调用 Exact-Q tensor helper 时需要
    torch = None  # type: ignore[assignment]


DEFAULT_Q_EQUALITY_TOLERANCE = 1.0e-12


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("UAV Exact-Q helper 需要 PyTorch。")


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
    """从已有 Exact-Q 结果中提取对齐且不写入日志的数组。"""

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

    # np.argmax 按 proposed_joint 的操作顺序稳定地优先选择 STAY；Exact-Q 平局
    # 数量和比例仍会单独记录。
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
    """计算汇总指标并返回供 episode 合并使用的内部数组。"""

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
    """将 rollout 分块数组合并为 episode 级汇总。"""

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

def _exact_q_policy_quality_metrics(
    probabilities_before: Any,
    probabilities_after: Any,
    q_values: Any,
    valid_mask: Any,
) -> dict[str, float]:
    """在同一批冻结的 Exact-Q 候选数据上比较两个策略。"""

    before = np.asarray(probabilities_before, dtype=np.float64)
    after = np.asarray(probabilities_after, dtype=np.float64)
    q = np.asarray(q_values, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if before.shape != after.shape or before.shape != q.shape or before.shape != valid.shape:
        raise ValueError("exact-Q policy quality tensor shapes do not match")
    if before.ndim < 1 or not np.all(np.isfinite(q[valid])):
        raise ValueError("exact-Q policy quality tensors are invalid")
    top1_before: list[float] = []
    top1_after: list[float] = []
    regret_before: list[float] = []
    regret_after: list[float] = []
    expected_before: list[float] = []
    expected_after: list[float] = []
    policy_kls: list[float] = []
    for prefix in np.ndindex(q.shape[:-1]):
        legal = np.flatnonzero(valid[prefix])
        if legal.size == 0:
            continue
        p_before = before[prefix][legal]
        p_after = after[prefix][legal]
        q_row = q[prefix][legal]
        if (
            not np.all(np.isfinite(p_before))
            or not np.all(np.isfinite(p_after))
            or np.any(p_before < 0.0)
            or np.any(p_after < 0.0)
            or not np.all(np.isfinite(q_row))
        ):
            raise FloatingPointError("exact-Q policy quality contains NaN/Inf")
        before_sum = float(p_before.sum())
        after_sum = float(p_after.sum())
        if before_sum <= 0.0 or after_sum <= 0.0:
            raise FloatingPointError("exact-Q candidate policy probability mass is empty")
        p_before = p_before / before_sum
        p_after = p_after / after_sum
        best_q = float(np.max(q_row))
        best_mask = np.isclose(q_row, best_q, rtol=1.0e-10, atol=1.0e-10)
        selected_before = int(np.argmax(p_before))
        selected_after = int(np.argmax(p_after))
        selected_q_before = float(q_row[selected_before])
        selected_q_after = float(q_row[selected_after])
        top1_before.append(float(bool(best_mask[selected_before])))
        top1_after.append(float(bool(best_mask[selected_after])))
        regret_before.append(max(0.0, best_q - selected_q_before))
        regret_after.append(max(0.0, best_q - selected_q_after))
        expected_before.append(float(np.dot(p_before, q_row)))
        expected_after.append(float(np.dot(p_after, q_row)))
        positive = p_before > 0.0
        kl = float(
            np.sum(
                p_before[positive]
                * (
                    np.log(p_before[positive])
                    - np.log(np.maximum(p_after[positive], 1.0e-300))
                )
            )
        )
        policy_kls.append(max(0.0, kl))
    if not top1_before:
        return {
            "count": 0.0,
            "top1_before": math.nan,
            "top1_after": math.nan,
            "mean_regret_before": math.nan,
            "mean_regret_after": math.nan,
            "expected_q_before": math.nan,
            "expected_q_after": math.nan,
            "expected_q_delta": math.nan,
            "policy_kl_old_new": math.nan,
        }
    expected_q_before = float(np.mean(expected_before))
    expected_q_after = float(np.mean(expected_after))
    result = {
        "count": float(len(top1_before)),
        "top1_before": float(np.mean(top1_before)),
        "top1_after": float(np.mean(top1_after)),
        "mean_regret_before": float(np.mean(regret_before)),
        "mean_regret_after": float(np.mean(regret_after)),
        "expected_q_before": expected_q_before,
        "expected_q_after": expected_q_after,
        "expected_q_delta": expected_q_after - expected_q_before,
        "policy_kl_old_new": float(np.mean(policy_kls)),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("exact-Q policy quality is non-finite")
    return result

def _uav_candidate_policy_metrics(
    probabilities: Any,
    teacher_probabilities: Any,
    q_values: Any,
    valid_mask: Any,
    logits: Any,
) -> dict[str, float]:
    """汇总冻结更新批次中的合法 UAV 候选分布。"""

    policy = np.asarray(probabilities, dtype=np.float64)
    teacher = np.asarray(teacher_probabilities, dtype=np.float64)
    q = np.asarray(q_values, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    raw_logits = np.asarray(logits, dtype=np.float64)
    if not (
        policy.shape == teacher.shape == q.shape == valid.shape == raw_logits.shape
    ) or policy.ndim < 1:
        raise ValueError("UAV candidate diagnostic tensor shapes do not match")

    policy_entropies: list[float] = []
    teacher_entropies: list[float] = []
    best_policy_probabilities: list[float] = []
    best_policy_ranks: list[float] = []
    best_top1: list[float] = []
    best_top3: list[float] = []
    logit_stds: list[float] = []
    logit_ranges: list[float] = []
    for prefix in np.ndindex(q.shape[:-1]):
        legal = np.flatnonzero(valid[prefix])
        if legal.size == 0:
            continue
        p = policy[prefix][legal]
        t = teacher[prefix][legal]
        q_row = q[prefix][legal]
        logits_row = raw_logits[prefix][legal]
        if (
            not np.all(np.isfinite(p))
            or not np.all(np.isfinite(t))
            or not np.all(np.isfinite(q_row))
            or not np.all(np.isfinite(logits_row))
            or np.any(p < 0.0)
            or np.any(t < 0.0)
        ):
            raise FloatingPointError("UAV candidate diagnostic contains NaN/Inf")
        p_sum = float(np.sum(p))
        t_sum = float(np.sum(t))
        if p_sum <= 0.0 or t_sum <= 0.0:
            raise FloatingPointError("UAV candidate diagnostic has empty legal mass")
        p = p / p_sum
        t = t / t_sum
        policy_entropies.append(
            float(-np.sum(np.where(p > 0.0, p * np.log(np.maximum(p, 1.0e-12)), 0.0)))
        )
        teacher_entropies.append(
            float(-np.sum(np.where(t > 0.0, t * np.log(np.maximum(t, 1.0e-12)), 0.0)))
        )
        best_q = float(np.max(q_row))
        best_mask = np.isclose(q_row, best_q, rtol=1.0e-10, atol=1.0e-10)
        descending = np.argsort(-p, kind="stable")
        best_positions = np.flatnonzero(best_mask)
        best_policy_probabilities.append(float(np.sum(p[best_mask])))
        best_policy_ranks.append(
            float(1 + min(int(np.flatnonzero(descending == position)[0]) for position in best_positions))
        )
        best_top1.append(float(bool(np.any(best_mask[descending[:1]]))))
        best_top3.append(float(bool(np.any(best_mask[descending[: min(3, len(descending))]]))))
        logit_stds.append(float(np.std(logits_row)))
        logit_ranges.append(float(np.max(logits_row) - np.min(logits_row)))

    if not policy_entropies:
        return {
            "policy_entropy_mean": math.nan,
            "teacher_entropy_mean": math.nan,
            "best_policy_prob_mean": math.nan,
            "best_policy_prob_median": math.nan,
            "best_policy_rank_mean": math.nan,
            "best_policy_top1_rate": math.nan,
            "best_policy_top3_rate": math.nan,
            "logit_std_mean": math.nan,
            "logit_range_mean": math.nan,
        }
    result = {
        "policy_entropy_mean": float(np.mean(policy_entropies)),
        "teacher_entropy_mean": float(np.mean(teacher_entropies)),
        "best_policy_prob_mean": float(np.mean(best_policy_probabilities)),
        "best_policy_prob_median": float(np.median(best_policy_probabilities)),
        "best_policy_rank_mean": float(np.mean(best_policy_ranks)),
        "best_policy_top1_rate": float(np.mean(best_top1)),
        "best_policy_top3_rate": float(np.mean(best_top3)),
        "logit_std_mean": float(np.mean(logit_stds)),
        "logit_range_mean": float(np.mean(logit_ranges)),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("UAV candidate diagnostic is non-finite")
    return result

def _uav_candidate_old_new_drift_summary(
    kl_values: Any,
    argmax_flip_values: Any,
    best_policy_prob_change_values: Any,
) -> dict[str, Any]:
    """汇总更新前后的合法候选分布。每个条目对应一个合法候选 head；保留数组供调用方计算更新级分位数，使其统计合法 head，而不是已经聚合的 epoch 均值。"""

    kl = np.asarray(kl_values, dtype=np.float64).reshape(-1)
    flips = np.asarray(argmax_flip_values, dtype=np.float64).reshape(-1)
    best_changes = np.asarray(
        best_policy_prob_change_values, dtype=np.float64
    ).reshape(-1)
    if not (kl.shape == flips.shape == best_changes.shape):
        raise ValueError("UAV candidate drift value shapes do not match")
    if kl.size == 0:
        return {
            "kl_values": kl,
            "argmax_flip_values": flips,
            "best_policy_prob_change_values": best_changes,
            "kl_mean": math.nan,
            "kl_p50": math.nan,
            "kl_p90": math.nan,
            "kl_max": math.nan,
            "argmax_flip_rate": math.nan,
            "best_policy_prob_change_mean": math.nan,
            "best_policy_prob_change_p10": math.nan,
        }
    if not (
        np.all(np.isfinite(kl))
        and np.all(np.isfinite(flips))
        and np.all(np.isfinite(best_changes))
    ):
        raise FloatingPointError("UAV candidate old-vs-new drift is non-finite")
    return {
        "kl_values": kl,
        "argmax_flip_values": flips,
        "best_policy_prob_change_values": best_changes,
        "kl_mean": float(np.mean(kl)),
        "kl_p50": float(np.percentile(kl, 50.0)),
        "kl_p90": float(np.percentile(kl, 90.0)),
        "kl_max": float(np.max(kl)),
        "argmax_flip_rate": float(np.mean(flips)),
        "best_policy_prob_change_mean": float(np.mean(best_changes)),
        "best_policy_prob_change_p10": float(
            np.percentile(best_changes, 10.0)
        ),
    }

def _uav_candidate_old_new_drift_metrics(
    old_probabilities: Any,
    new_probabilities: Any,
    teacher_probabilities: Any,
    q_values: Any,
    valid_mask: Any,
) -> dict[str, Any]:
    """测量固定合法 Exact-Q 候选 head 上的策略变化。旧、新分布只在冻结的合法 mask 内重新归一化。最优候选概率变化按冻结的最高 Q 候选集合计算，因此并列候选作为集合处理，不依赖任意选出的单个索引。"""

    old = np.asarray(old_probabilities, dtype=np.float64)
    new = np.asarray(new_probabilities, dtype=np.float64)
    teacher = np.asarray(teacher_probabilities, dtype=np.float64)
    q = np.asarray(q_values, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if not (old.shape == new.shape == teacher.shape == q.shape == valid.shape):
        raise ValueError("UAV candidate old-vs-new tensor shapes do not match")
    if old.ndim < 1:
        raise ValueError("UAV candidate old-vs-new tensors must have a candidate axis")

    kl_values: list[float] = []
    argmax_flip_values: list[float] = []
    best_policy_prob_change_values: list[float] = []
    for prefix in np.ndindex(q.shape[:-1]):
        legal = np.flatnonzero(valid[prefix])
        if legal.size == 0:
            continue
        old_row = old[prefix][legal]
        new_row = new[prefix][legal]
        teacher_row = teacher[prefix][legal]
        q_row = q[prefix][legal]
        if not (
            np.all(np.isfinite(old_row))
            and np.all(np.isfinite(new_row))
            and np.all(np.isfinite(teacher_row))
            and np.all(np.isfinite(q_row))
            and np.all(old_row >= 0.0)
            and np.all(new_row >= 0.0)
            and np.all(teacher_row >= 0.0)
        ):
            raise FloatingPointError(
                "UAV candidate old-vs-new drift contains NaN/Inf or negative mass"
            )
        old_sum = float(np.sum(old_row))
        new_sum = float(np.sum(new_row))
        teacher_sum = float(np.sum(teacher_row))
        if old_sum <= 0.0 or new_sum <= 0.0 or teacher_sum <= 0.0:
            raise FloatingPointError(
                "UAV candidate old-vs-new drift has empty legal probability mass"
            )
        old_row = old_row / old_sum
        new_row = new_row / new_sum
        teacher_row = teacher_row / teacher_sum
        positive = old_row > 0.0
        kl = float(
            np.sum(
                old_row[positive]
                * (
                    np.log(np.maximum(old_row[positive], 1.0e-300))
                    - np.log(np.maximum(new_row[positive], 1.0e-300))
                )
            )
        )
        kl_values.append(max(0.0, kl))
        argmax_flip_values.append(
            float(int(np.argmax(old_row)) != int(np.argmax(new_row)))
        )
        best_q = float(np.max(q_row))
        best_mask = np.isclose(q_row, best_q, rtol=1.0e-10, atol=1.0e-10)
        # teacher 是损失函数使用的 detached Exact-Q target；
        # 将它保留在校验路径中，防止诊断无提示地改用
        # 其他 target 张量。
        if not np.isclose(float(np.sum(teacher_row)), 1.0, atol=1.0e-8):
            raise FloatingPointError("UAV candidate teacher mass is not normalized")
        best_policy_prob_change_values.append(
            float(np.sum(new_row[best_mask]) - np.sum(old_row[best_mask]))
        )
    return _uav_candidate_old_new_drift_summary(
        kl_values,
        argmax_flip_values,
        best_policy_prob_change_values,
    )

def _proposal_source_diagnostics(
    q_values: Any,
    valid_mask: Any,
    sources: Any,
    distances: Any,
) -> dict[str, Any]:
    """汇总固定 proposal 集合，不额外评估新分支。"""

    q = np.asarray(q_values, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    source_array = np.asarray(sources, dtype=object)
    distance_array = np.asarray(distances, dtype=np.float64)
    if q.shape != valid.shape or q.shape != source_array.shape or q.shape != distance_array.shape:
        raise ValueError("proposal diagnostic tensor shapes do not match")
    counts = valid.sum(axis=-1).reshape(-1)
    active_counts = counts[counts > 0]
    source_counts: dict[str, int] = {}
    best_distances: list[float] = []
    for prefix in np.ndindex(q.shape[:-1]):
        legal = np.flatnonzero(valid[prefix])
        if legal.size == 0:
            continue
        best_index = int(legal[int(np.argmax(q[prefix][legal]))])
        source = str(source_array[prefix + (best_index,)])
        if not source:
            raise ValueError("proposal diagnostic is missing a candidate source")
        source_counts[source] = source_counts.get(source, 0) + 1
        distance = float(distance_array[prefix + (best_index,)])
        if not math.isfinite(distance):
            raise FloatingPointError("proposal diagnostic distance is non-finite")
        best_distances.append(distance)
    total = int(sum(source_counts.values()))
    fractions = {
        source: float(count) / float(total)
        for source, count in sorted(source_counts.items())
    }
    return {
        "candidate_count_mean": (
            0.0 if active_counts.size == 0 else float(np.mean(active_counts))
        ),
        "candidate_count_max": (
            0 if active_counts.size == 0 else int(np.max(active_counts))
        ),
        "best_q_source_counts": {
            source: int(count) for source, count in sorted(source_counts.items())
        },
        "best_q_source_fractions": fractions,
        "best_q_distance_mean": (
            math.nan if not best_distances else float(np.mean(best_distances))
        ),
        "best_q_distance_median": (
            math.nan if not best_distances else float(np.median(best_distances))
        ),
        "best_q_distance_max": (
            math.nan if not best_distances else float(np.max(best_distances))
        ),
    }

def _exact_q_listwise_teacher(
    q_values: Any,
    candidate_valid_mask: Any,
    temperature: float,
) -> Any:
    """根据各状态的 Exact-Q 构造已分离梯度的合法候选 teacher。温度 softmax 前会按 state/head 分别标准化 Q 值；相同 Q 值的合法集合得到均匀 teacher，padding 候选的质量始终为零。"""

    _require_torch()
    q_tensor = torch.as_tensor(q_values, dtype=torch.float32)
    valid = torch.as_tensor(
        candidate_valid_mask, dtype=torch.bool, device=q_tensor.device
    )
    if q_tensor.shape != valid.shape or q_tensor.ndim < 1:
        raise ValueError("exact-Q teacher tensor shape mismatch")
    scale = float(temperature)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("exact-Q teacher temperature must be positive and finite")
    if not bool(torch.all(torch.isfinite(q_tensor[valid])).item()):
        raise FloatingPointError("exact-Q teacher received non-finite legal Q values")
    legal_count = valid.sum(dim=-1, keepdim=True)
    active_heads = legal_count > 0
    safe_valid = valid | ~active_heads
    safe_q = torch.where(valid, q_tensor, torch.zeros_like(q_tensor))
    count = legal_count.clamp_min(1).to(torch.float32)
    mean = safe_q.sum(dim=-1, keepdim=True) / count
    centered = torch.where(valid, safe_q - mean, torch.zeros_like(safe_q))
    variance = (centered.square().sum(dim=-1, keepdim=True) / count).clamp_min(0.0)
    std = torch.sqrt(variance)
    standardized = torch.where(
        valid & (std > 1.0e-8),
        centered / std.clamp_min(1.0e-8),
        torch.zeros_like(centered),
    )
    logits = standardized / scale
    logits = logits.masked_fill(~safe_valid, -float("inf"))
    teacher = torch.softmax(logits, dim=-1)
    return torch.where(valid & active_heads, teacher, torch.zeros_like(teacher)).detach()

def _exact_q_listwise_confidence(
    q_values: Any,
    candidate_valid_mask: Any,
    reference_scale: float,
) -> Any:
    """根据合法候选 Q 值的离散程度返回各 head 的 detached confidence。总体标准差使用与 _exact_q_listwise_teacher 相同的合法 mask 和逐 head 居中方式；此辅助函数不会改动标准化 teacher logits。"""

    _require_torch()
    q_tensor = torch.as_tensor(q_values, dtype=torch.float32)
    valid = torch.as_tensor(
        candidate_valid_mask, dtype=torch.bool, device=q_tensor.device
    )
    if q_tensor.shape != valid.shape or q_tensor.ndim < 1:
        raise ValueError("exact-Q confidence tensor shape mismatch")
    scale = float(reference_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("exact-Q confidence reference scale must be positive and finite")
    if not bool(torch.all(torch.isfinite(q_tensor[valid])).item()):
        raise FloatingPointError("exact-Q confidence received non-finite legal Q values")
    legal_count = valid.sum(dim=-1, keepdim=True)
    active_heads = legal_count > 0
    safe_q = torch.where(valid, q_tensor, torch.zeros_like(q_tensor))
    count = legal_count.clamp_min(1).to(torch.float32)
    mean = safe_q.sum(dim=-1, keepdim=True) / count
    centered = torch.where(valid, safe_q - mean, torch.zeros_like(safe_q))
    variance = (centered.square().sum(dim=-1, keepdim=True) / count).clamp_min(0.0)
    q_std = torch.sqrt(variance).squeeze(-1)
    confidence = torch.clamp(q_std / scale, min=0.0, max=1.0)
    return torch.where(
        active_heads.squeeze(-1), confidence, torch.zeros_like(confidence)
    ).detach()

def _masked_operation_probabilities(probabilities: Any, masks: Any) -> Any:
    """在合法动作范围内归一化二元 Operation 分布。"""

    _require_torch()
    values = torch.as_tensor(probabilities, dtype=torch.float32)
    legal = torch.as_tensor(masks, dtype=torch.bool, device=values.device)
    if values.shape != legal.shape or values.ndim < 1 or int(values.shape[-1]) != 2:
        raise ValueError("operation probability/mask shape mismatch")
    if not bool(torch.all(torch.isfinite(values)).item()):
        raise FloatingPointError("operation probabilities are non-finite")
    masked = torch.where(legal, values, torch.zeros_like(values))
    normalizer = masked.sum(dim=-1, keepdim=True)
    if bool(torch.any(normalizer <= 0.0).item()):
        raise ValueError("operation mask blocks every operation")
    normalized = masked / normalizer
    if not bool(torch.all(torch.isfinite(normalized)).item()):
        raise FloatingPointError("masked operation probabilities are non-finite")
    if not torch.allclose(
        normalized.sum(dim=-1),
        torch.ones_like(normalized.sum(dim=-1)),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise FloatingPointError("masked operation probabilities do not sum to one")
    return normalized

def _exact_q_operation_teacher(
    q_stay: Any,
    candidate_q_values: Any,
    rollout_candidate_probabilities: Any,
    candidate_valid_mask: Any,
    operation_valid_mask: Any,
    rollout_operation_masks: Any,
    temperature: float,
    reference_scale: float,
) -> tuple[Any, Any, Any, Any, Any]:
    """根据冻结 rollout 价值构造二元 Operation teacher。Candidate teacher 和 Operation teacher 使用的 MOVE 价值来自同一个合法候选 Exact-Q listwise 分布。rollout 候选概率仅用于诊断对比。候选集合未激活时没有 MOVE 价值，只保留经过 mask 的 STAY teacher，不赋予硬最大值或构造虚假 Operation target。"""

    _require_torch()
    q_values = torch.as_tensor(candidate_q_values, dtype=torch.float32)
    valid = torch.as_tensor(
        candidate_valid_mask, dtype=torch.bool, device=q_values.device
    )
    rollout_probabilities = torch.as_tensor(
        rollout_candidate_probabilities,
        dtype=torch.float32,
        device=q_values.device,
    )
    stay_q = torch.as_tensor(q_stay, dtype=torch.float32, device=q_values.device)
    teacher_masks = torch.as_tensor(
        operation_valid_mask,
        dtype=torch.bool,
        device=q_values.device,
    )
    rollout_masks = torch.as_tensor(
        rollout_operation_masks,
        dtype=torch.bool,
        device=q_values.device,
    )
    if (
        q_values.shape != valid.shape
        or rollout_probabilities.shape != q_values.shape
        or q_values.ndim < 1
    ):
        raise ValueError("operation teacher candidate Q shape mismatch")
    if stay_q.shape != q_values.shape[:-1]:
        raise ValueError("operation teacher STAY Q shape mismatch")
    if teacher_masks.shape != stay_q.shape + (2,):
        raise ValueError("operation teacher validity mask shape mismatch")
    if rollout_masks.shape != teacher_masks.shape:
        raise ValueError("operation teacher rollout mask shape mismatch")
    scale = float(reference_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("operation teacher reference scale must be positive and finite")
    if not bool(torch.all(torch.isfinite(stay_q)).item()):
        raise FloatingPointError("operation teacher STAY Q is non-finite")
    if not bool(torch.all(torch.isfinite(rollout_probabilities)).item()):
        raise FloatingPointError(
            "operation teacher rollout candidate probabilities are non-finite"
        )
    if bool(torch.any(rollout_probabilities < 0.0).item()):
        raise ValueError(
            "operation teacher rollout candidate probabilities are negative"
        )
    candidate_teacher = _exact_q_listwise_teacher(
        q_values,
        valid,
        temperature,
    ).to(q_values.device)
    q_move_listwise_teacher = torch.sum(candidate_teacher * q_values, dim=-1)
    rollout_probabilities = torch.where(
        valid,
        rollout_probabilities,
        torch.zeros_like(rollout_probabilities),
    )
    rollout_probability_mass = rollout_probabilities.sum(dim=-1, keepdim=True)
    active_candidates = valid.any(dim=-1, keepdim=True)
    if bool(
        torch.any(
            active_candidates
            & ~torch.isclose(
                rollout_probability_mass,
                torch.ones_like(rollout_probability_mass),
                rtol=0.0,
                atol=1.0e-5,
            )
        ).item()
    ):
        raise ValueError(
            "operation teacher rollout candidate probabilities are not normalized"
        )
    if bool(
        torch.any(
            (~active_candidates)
            & (rollout_probability_mass != 0.0)
        ).item()
    ):
        raise ValueError(
            "inactive operation teacher candidate heads have non-zero probability mass"
        )
    q_move_operation = torch.sum(rollout_probabilities * q_values, dim=-1)
    move_legal = teacher_masks[..., 1]
    if bool(torch.any(teacher_masks & ~rollout_masks).item()):
        raise ValueError(
            "Exact-Q teacher-valid operation is not rollout-legal"
        )
    if bool(torch.any(~teacher_masks.any(dim=-1)).item()):
        raise ValueError("Exact-Q operation teacher has no valid operation")
    stay_legal = teacher_masks[..., 0]
    both_legal = stay_legal & move_legal
    delta = q_move_listwise_teacher - stay_q
    move_probability = torch.where(
        both_legal,
        torch.sigmoid(delta / scale),
        torch.where(move_legal, torch.ones_like(delta), torch.zeros_like(delta)),
    )
    raw_teacher = torch.stack(
        (1.0 - move_probability, move_probability),
        dim=-1,
    )
    teacher = _masked_operation_probabilities(raw_teacher, teacher_masks)
    if not bool(torch.all(torch.isfinite(teacher)).item()):
        raise FloatingPointError("operation teacher is non-finite")
    return (
        teacher.detach(),
        q_move_operation.detach(),
        q_move_listwise_teacher.detach(),
        move_probability.detach(),
        move_legal,
    )

def _operation_teacher_kl(
    operation_log_probs_all: Any,
    teacher_probabilities: Any,
    operation_masks: Any,
) -> Any:
    """返回完整 state/head 上 Exact-Q Operation teacher 到策略的 KL。"""

    _require_torch()
    log_probs = torch.as_tensor(operation_log_probs_all, dtype=torch.float32)
    teacher = torch.as_tensor(
        teacher_probabilities, dtype=torch.float32, device=log_probs.device
    )
    masks = torch.as_tensor(
        operation_masks, dtype=torch.bool, device=log_probs.device
    )
    if log_probs.shape != teacher.shape or teacher.shape != masks.shape:
        raise ValueError("operation teacher KL shape mismatch")
    safe_teacher = torch.where(masks, teacher, torch.zeros_like(teacher))
    safe_log_probs = torch.where(
        masks, log_probs, torch.zeros_like(log_probs)
    )
    kl = torch.sum(
        safe_teacher
        * (torch.log(safe_teacher.clamp_min(1.0e-12)) - safe_log_probs),
        dim=-1,
    )
    value = kl.mean()
    if not bool(torch.isfinite(value).item()):
        raise FloatingPointError("operation teacher KL is non-finite")
    return value

def _exact_q_listwise_candidate_loss(
    candidate_log_probs: Any,
    q_values: Any,
    candidate_valid_mask: Any,
    temperature: float,
    confidence_reference_scale: float | None = None,
) -> tuple[Any, Any]:
    """返回可选 confidence 加权 KL 和 detached teacher。"""

    _require_torch()
    log_probs = torch.as_tensor(candidate_log_probs, dtype=torch.float32)
    valid = torch.as_tensor(
        candidate_valid_mask, dtype=torch.bool, device=log_probs.device
    )
    q_tensor = torch.as_tensor(q_values, dtype=torch.float32, device=log_probs.device)
    if log_probs.shape != valid.shape or q_tensor.shape != valid.shape:
        raise ValueError("exact-Q listwise candidate tensor shape mismatch")
    teacher = _exact_q_listwise_teacher(q_tensor, valid, temperature).to(
        log_probs.device
    )
    active_heads = valid.any(dim=-1)
    if not bool(torch.any(active_heads).item()):
        return log_probs.sum() * 0.0, teacher
    safe_log_probs = torch.where(
        valid, log_probs, torch.zeros_like(log_probs)
    )
    safe_teacher = torch.where(
        valid, teacher, torch.zeros_like(teacher)
    )
    kl = torch.sum(
        safe_teacher
        * (
            torch.log(safe_teacher.clamp_min(1.0e-12))
            - safe_log_probs
        ),
        dim=-1,
    )
    if confidence_reference_scale is not None:
        confidence = _exact_q_listwise_confidence(
            q_tensor,
            valid,
            confidence_reference_scale,
        ).to(log_probs.device)
        if tuple(confidence.shape) != tuple(active_heads.shape):
            raise ValueError("exact-Q confidence head shape mismatch")
        if not bool(torch.all(torch.isfinite(confidence)).item()):
            raise FloatingPointError("exact-Q confidence is non-finite")
        if bool(torch.any(confidence < 0.0).item()):
            raise ValueError("exact-Q confidence is negative")
        kl = kl * confidence.detach()
    return kl[active_heads].mean(), teacher
