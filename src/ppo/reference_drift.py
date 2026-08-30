#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functional reference-state probability drift gate for V3 Direct-CF.

The gate is deliberately outside the Direct-CF loss.  It consumes the one
normal Adam candidate step, evaluates the masked conditional probabilities on
the current frozen-policy batch's unselected reference decisions, and scales
only that parameter displacement.  Adam's post-step state is never recomputed
or advanced a second time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


DEFAULT_REFERENCE_PROBABILITY_DRIFT_CAP = 0.005
# The single initial cap is below the four confirmed harmful FIRST shifts
# (0.006254, 0.006794, 0.008207, 0.009735); it is not a cap sweep or an
# assertion of theoretical optimality.
REFERENCE_DRIFT_CONSTRAINT_METHOD = "masked_conditional_probability_backtracking_bisection"
REFERENCE_DRIFT_CONSTRAINT_VERSION = "v1"
REFERENCE_DRIFT_BISECTION_STEPS = 10


@dataclass(frozen=True)
class ReferenceDriftGateResult:
    """All scalar diagnostics produced by one functional gate."""

    alpha: float
    constraint_active: bool
    cap: float
    tolerance: float
    probability_dtype: str
    reference_count: int
    selected_count: int
    intersection_count: int
    raw_reference_drift: dict[str, float]
    constrained_reference_drift: dict[str, float]
    raw_selected_drift: dict[str, float]
    constrained_selected_drift: dict[str, float]
    per_trigger: dict[str, dict[str, float | int]]
    raw_reference_p1: np.ndarray
    constrained_reference_p1: np.ndarray
    raw_selected_p1: np.ndarray
    constrained_selected_p1: np.ndarray


def _require_torch() -> None:
    if torch is None:
        raise ImportError("reference_drift.py 需要 PyTorch。")


def _finite_cap(cap: float) -> float:
    result = float(cap)
    if not math.isfinite(result) or result <= 0.0 or result > 1.0:
        raise ValueError("reference_probability_drift_cap 必须位于 (0,1]。")
    return result


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("reference drift statistics包含NaN/Inf。")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _policy_device(policy: Any) -> Any:
    _require_torch()
    try:
        return next(policy.parameters()).device
    except StopIteration as exc:
        raise ValueError("policy必须包含参数。") from exc


def _conditional_probability_one_tensor(policy: Any, records: Iterable[Any]) -> tuple[Any, Any]:
    """Return the actual masked conditional probability tensor and its dtype."""

    _require_torch()
    items = list(records)
    if not items:
        raise ValueError("conditional probability requires at least one reference record.")
    observations = np.stack(
        [np.asarray(item.observation, dtype=np.float32) for item in items],
        axis=0,
    )
    actions = np.stack(
        [np.asarray(item.action_vector, dtype=np.float32) for item in items],
        axis=0,
    )
    masks = np.stack(
        [np.asarray(item.action_mask, dtype=bool) for item in items],
        axis=0,
    )
    target_indices = np.asarray(
        [int(item.trigger_index) for item in items],
        dtype=np.int64,
    )
    device = _policy_device(policy)
    with torch.no_grad():
        evaluation = policy.evaluate_actions(
            torch.as_tensor(observations, dtype=torch.float32, device=device),
            torch.as_tensor(actions, dtype=torch.float32, device=device),
            action_masks=torch.as_tensor(masks, dtype=torch.bool, device=device),
        )
        rows = torch.arange(len(items), device=device)
        values = evaluation.per_action_probabilities[
            rows,
            torch.as_tensor(target_indices, dtype=torch.long, device=device),
        ]
    detached_values = values.detach()
    if not bool(torch.isfinite(detached_values).all().item()):
        raise FloatingPointError("reference conditional p(action=1)包含NaN/Inf。")
    return detached_values, detached_values.dtype


def conditional_probability_one_with_dtype(
    policy: Any,
    records: Iterable[Any],
) -> tuple[np.ndarray, Any]:
    """Recompute p(action=1), retaining the actual model-output dtype."""

    values, probability_dtype = _conditional_probability_one_tensor(policy, records)
    result = values.cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("reference conditional p(action=1)包含NaN/Inf。")
    return result, probability_dtype


def conditional_probability_one(policy: Any, records: Iterable[Any]) -> np.ndarray:
    """Recompute masked conditional p(action=1) for reference records."""

    result, _ = conditional_probability_one_with_dtype(policy, records)
    return result


def dtype_aware_reference_tolerance(probability_dtype: Any) -> float:
    """Return the dtype-derived absolute Gate-H tolerance (2 ULP scale)."""

    _require_torch()
    try:
        epsilon = torch.finfo(probability_dtype).eps
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"probability_dtype必须是支持finfo的浮点dtype：{probability_dtype!r}"
        ) from exc
    return float(2.0 * epsilon)


def old_p_reconstruction_diagnostics(
    policy: Any,
    records: Iterable[Any],
) -> tuple[float, float, str]:
    """Return max error, dtype-aware tolerance and actual probability dtype."""

    items = list(records)
    if not items:
        raise ValueError("old-p reconstruction requires at least one reference record.")
    stored_probability = np.asarray(
        [float(item.p_old) for item in items],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(stored_probability)):
        raise FloatingPointError("stored reference p_old包含NaN/Inf。")
    recomputed_probability, probability_dtype = conditional_probability_one_with_dtype(
        policy,
        items,
    )
    if recomputed_probability.shape != stored_probability.shape:
        raise AssertionError("recomputed/stored reference p_old shape不一致。")
    difference = np.abs(recomputed_probability - stored_probability)
    if not np.all(np.isfinite(difference)):
        raise FloatingPointError("old-p reconstruction error包含NaN/Inf。")
    return (
        float(np.max(difference)),
        dtype_aware_reference_tolerance(probability_dtype),
        str(probability_dtype),
    )


def old_p_reconstruction_max_error(policy: Any, records: Iterable[Any]) -> float:
    """Compare theta-current conditional p(action=1) with stored ``p_old``.

    The caller invokes this before the RefDrift Adam step, when ``policy`` is
    still the collection/old policy.  This is a read-only correctness guard;
    it does not construct a gradient and does not alter any optimizer state.
    """

    maximum, _, _ = old_p_reconstruction_diagnostics(policy, records)
    return maximum


def validate_reference_partition(
    records: Iterable[Any],
    selected_ids: Iterable[str],
) -> tuple[list[Any], list[Any], dict[str, int]]:
    """Split R_all into selected S and unselected R_ref by stable identifier."""

    items = list(records)
    identifiers = [str(item.decision_id) for item in items]
    if any(not value for value in identifiers):
        raise ValueError("reference decision_id不能为空。")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("reference decision_id必须唯一。")
    selected = [str(value) for value in selected_ids]
    if len(set(selected)) != len(selected):
        raise ValueError("selected Direct-CF decision_id必须唯一。")
    all_ids = set(identifiers)
    selected_set = set(selected)
    missing = selected_set - all_ids
    if missing:
        raise ValueError(
            "selected Direct-CF decisions必须来自当前R_all；缺少："
            f"{sorted(missing)[:5]}"
        )
    selected_items = [item for item in items if str(item.decision_id) in selected_set]
    reference_items = [item for item in items if str(item.decision_id) not in selected_set]
    intersection_count = len(
        {str(item.decision_id) for item in selected_items}
        & {str(item.decision_id) for item in reference_items}
    )
    return selected_items, reference_items, {
        "all_count": len(items),
        "selected_count": len(selected_items),
        "reference_count": len(reference_items),
        "intersection_count": intersection_count,
    }


def _per_trigger_stats(
    records: list[Any],
    old_probability: np.ndarray,
    new_probability: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[int]] = {}
    for index, item in enumerate(records):
        grouped.setdefault(str(item.trigger_label), []).append(index)
    result: dict[str, dict[str, float | int]] = {}
    for label in sorted(grouped):
        indices = np.asarray(grouped[label], dtype=np.int64)
        signed = new_probability[indices] - old_probability[indices]
        absolute = np.abs(signed)
        result[label] = {
            "count": int(indices.size),
            "mean_abs_drift": float(np.mean(absolute)) if indices.size else 0.0,
            "p95_abs_drift": float(np.percentile(absolute, 95)) if indices.size else 0.0,
            "max_abs_drift": float(np.max(absolute)) if indices.size else 0.0,
            "mean_signed_drift": float(np.mean(signed)) if indices.size else 0.0,
        }
    return result


def constrain_after_adam_step(
    policy: Any,
    old_parameters: dict[str, Any],
    raw_parameters: dict[str, Any],
    all_reference_records: Iterable[Any],
    selected_ids: Iterable[str],
    cap: float,
    *,
    tolerance: float | None = None,
    bisection_steps: int = REFERENCE_DRIFT_BISECTION_STEPS,
) -> ReferenceDriftGateResult:
    """Scale old->raw parameters after exactly one already-completed Adam step."""

    _require_torch()
    cap_value = _finite_cap(cap)
    tolerance_value = None if tolerance is None else float(tolerance)
    if tolerance_value is not None and (
        not math.isfinite(tolerance_value) or tolerance_value < 0.0
    ):
        raise ValueError("reference drift tolerance必须是有限非负数。")
    if int(bisection_steps) < 0:
        raise ValueError("bisection_steps不能为负。")
    selected_records, reference_records, partition = validate_reference_partition(
        all_reference_records,
        selected_ids,
    )
    if not reference_records:
        raise ValueError("R_ref不能为空；selected state不能全部占用R_all。")
    parameter_names = list(old_parameters)
    if parameter_names != list(raw_parameters):
        raise ValueError("old/raw parameter keys不一致。")
    for name in parameter_names:
        if name not in dict(policy.named_parameters()):
            raise ValueError(f"policy缺少参数：{name}")

    old_probability = np.asarray(
        [float(item.p_old) for item in reference_records],
        dtype=np.float64,
    )
    selected_old_probability = np.asarray(
        [float(item.p_old) for item in selected_records],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(old_probability)) or not np.all(np.isfinite(selected_old_probability)):
        raise FloatingPointError("reference p_old包含NaN/Inf。")

    named_parameters = dict(policy.named_parameters())

    def set_alpha(alpha: float) -> None:
        with torch.no_grad():
            for name in parameter_names:
                parameter = named_parameters[name]
                old = old_parameters[name].to(device=parameter.device)
                raw = raw_parameters[name].to(device=parameter.device)
                parameter.copy_(old + float(alpha) * (raw - old))

    set_alpha(1.0)
    raw_probability, probability_dtype = conditional_probability_one_with_dtype(
        policy,
        reference_records,
    )
    if tolerance_value is None:
        tolerance_value = dtype_aware_reference_tolerance(probability_dtype)
    raw_selected_probability = conditional_probability_one(policy, selected_records)
    raw_stats = _stats(np.abs(raw_probability - old_probability))
    raw_selected_stats = _stats(np.abs(raw_selected_probability - selected_old_probability))

    def passes(alpha: float) -> tuple[bool, np.ndarray]:
        set_alpha(alpha)
        probability = conditional_probability_one(policy, reference_records)
        maximum = float(np.max(np.abs(probability - old_probability)))
        return maximum <= cap_value + tolerance_value, probability

    if raw_stats["max"] <= cap_value + tolerance_value:
        alpha = 1.0
        constrained_probability = raw_probability
        constraint_active = False
    else:
        constraint_active = True
        failing_alpha = 1.0
        passing_alpha = 0.5
        passed, constrained_probability = passes(passing_alpha)
        while not passed and passing_alpha > np.finfo(np.float64).tiny:
            failing_alpha = passing_alpha
            passing_alpha *= 0.5
            passed, constrained_probability = passes(passing_alpha)
        if not passed:
            raise FloatingPointError("backtracking无法找到满足reference drift cap的正alpha。")
        for _ in range(int(bisection_steps)):
            midpoint = 0.5 * (passing_alpha + failing_alpha)
            midpoint_passed, midpoint_probability = passes(midpoint)
            if midpoint_passed:
                passing_alpha = midpoint
                constrained_probability = midpoint_probability
            else:
                failing_alpha = midpoint
        alpha = float(passing_alpha)

    set_alpha(alpha)
    constrained_probability = conditional_probability_one(policy, reference_records)
    constrained_selected_probability = conditional_probability_one(policy, selected_records)
    constrained_stats = _stats(np.abs(constrained_probability - old_probability))
    constrained_selected_stats = _stats(
        np.abs(constrained_selected_probability - selected_old_probability)
    )
    if constrained_stats["max"] > cap_value + tolerance_value:
        raise AssertionError(
            "constrained reference drift超过cap："
            f"{constrained_stats['max']} > {cap_value}+{tolerance_value}"
        )
    return ReferenceDriftGateResult(
        alpha=float(alpha),
        constraint_active=bool(constraint_active),
        cap=cap_value,
        tolerance=float(tolerance_value),
        probability_dtype=str(probability_dtype),
        reference_count=int(partition["reference_count"]),
        selected_count=int(partition["selected_count"]),
        intersection_count=int(partition["intersection_count"]),
        raw_reference_drift=raw_stats,
        constrained_reference_drift=constrained_stats,
        raw_selected_drift=raw_selected_stats,
        constrained_selected_drift=constrained_selected_stats,
        per_trigger=_per_trigger_stats(
            reference_records,
            old_probability,
            constrained_probability,
        ),
        raw_reference_p1=raw_probability,
        constrained_reference_p1=constrained_probability,
        raw_selected_p1=raw_selected_probability,
        constrained_selected_p1=constrained_selected_probability,
    )
