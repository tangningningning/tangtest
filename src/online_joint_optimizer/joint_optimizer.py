"""Budgeted cache/UAV coordination entry point."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .cache_optimizer import (
    CacheCandidateScore,
    CacheOptimizationResult,
    CacheOptimizer,
    ForecastTailResult,
)
from .candidate_evaluator import (
    CandidateEvaluation,
    CandidateEvaluator,
    EvaluationBudget,
    PredictedSlotInput,
    SearchBudget,
    SlotInput,
    copy_uav_positions,
)
from .uav_scheduler import UavScheduleResult, UavScheduler


@dataclass(frozen=True)
class JointOptimizationResult:
    is_feasible: bool
    search_complete: bool
    target_cache_state: np.ndarray
    target_uav_positions: Mapping[str, np.ndarray]
    baseline_current_utility: float | None
    final_current_utility: float | None
    current_utility_delta: float | None
    baseline_horizon_utility: float | None
    final_horizon_utility: float | None
    horizon_utility_delta: float | None
    evaluation_count: int
    status: str
    failure_reason: str = ""
    cache_search: CacheOptimizationResult | None = None
    uav_searches: tuple[UavScheduleResult, ...] = ()
    final_evaluation: CandidateEvaluation | None = None
    effective_prediction_scenarios: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class JointOptimizer:
    """Coordinate bounded cache search, per-UAV search, and joint re-evaluation."""

    def __init__(
        self,
        evaluator: CandidateEvaluator | None = None,
        cache_optimizer: CacheOptimizer | None = None,
        uav_scheduler: UavScheduler | None = None,
    ) -> None:
        self.evaluator = evaluator or CandidateEvaluator()
        self.uav_scheduler = uav_scheduler or UavScheduler(self.evaluator)
        self.cache_optimizer = cache_optimizer or CacheOptimizer(self.uav_scheduler)

    def optimize(
        self,
        *,
        current_cache_state: Any,
        current_uav_positions: Mapping[str, Any],
        slot_input: SlotInput,
        forecast_slots: Sequence[PredictedSlotInput] | None = None,
        search_budget: SearchBudget | None = None,
    ) -> JointOptimizationResult:
        """Return one bounded joint target pair and its system-utility diagnostics."""

        settings = search_budget or SearchBudget()
        ledger = settings.new_ledger()
        cache = np.asarray(current_cache_state)
        if cache.ndim != 2 or not np.all(np.isin(cache, [0, 1])):
            raise ValueError("current_cache_state must be a binary node-by-service matrix")
        cache = cache.astype(np.int8, copy=True)
        uav = copy_uav_positions(current_uav_positions)
        forecasts = tuple(forecast_slots or ())
        if any(not isinstance(item, PredictedSlotInput) for item in forecasts):
            raise TypeError("future inputs must be explicit PredictedSlotInput values")

        baseline = self.evaluator.evaluate(
            slot_input=slot_input,
            previous_cache_state=cache,
            previous_uav_positions=uav,
            candidate_cache_state=cache,
            candidate_uav_positions=uav,
            budget=ledger,
            budget_category="joint",
        )
        if not baseline.is_feasible or baseline.system_utility is None:
            return JointOptimizationResult(
                is_feasible=False,
                search_complete=False,
                target_cache_state=cache,
                target_uav_positions=uav,
                baseline_current_utility=None,
                final_current_utility=None,
                current_utility_delta=None,
                baseline_horizon_utility=None,
                final_horizon_utility=None,
                horizon_utility_delta=None,
                evaluation_count=ledger.evaluations_used,
                status="baseline_evaluation_failed",
                failure_reason=baseline.failure_reason,
                final_evaluation=baseline,
                diagnostics={"budget_counts": dict(ledger.category_counts)},
            )

        baseline_tail = self.cache_optimizer.evaluate_forecast_tail(
            target_cache_state=cache,
            current_uav_positions=uav,
            forecast_slots=forecasts[: settings.max_forecast_slots],
            model=slot_input.model,
            search_budget=settings,
            evaluation_budget=ledger,
            budget_category="joint",
            max_candidates_per_uav=2,
            max_coordinate_rounds=1,
        )
        baseline_objective = self._total_utility(baseline.system_utility, baseline_tail)
        if forecasts and (
            not baseline_tail.is_available
            or not baseline_tail.is_feasible
            or baseline_objective is None
        ):
            return JointOptimizationResult(
                is_feasible=True,
                search_complete=False,
                target_cache_state=cache.copy(),
                target_uav_positions={key: value.copy() for key, value in uav.items()},
                baseline_current_utility=baseline.system_utility,
                final_current_utility=baseline.system_utility,
                current_utility_delta=0.0,
                baseline_horizon_utility=None,
                final_horizon_utility=None,
                horizon_utility_delta=None,
                evaluation_count=ledger.evaluations_used,
                status="baseline_forecast_evaluation_failed",
                failure_reason=(
                    baseline_tail.failure_reason or "baseline_forecast_utility_unavailable"
                ),
                final_evaluation=baseline,
                effective_prediction_scenarios=tuple(
                    baseline_tail.effective_prediction_scenarios
                ),
                diagnostics={
                    "budget_counts": dict(ledger.category_counts),
                    "budget_limits": dict(ledger.category_limits),
                    "budget_remaining": ledger.remaining,
                    "cache_search_skipped": True,
                    "reason": "full predicted-horizon baseline is required for comparable cache scoring",
                },
            )

        cache_search = self.cache_optimizer.optimize(
            current_cache_state=cache,
            current_uav_positions=uav,
            slot_input=slot_input,
            forecast_slots=forecasts,
            search_budget=settings,
            evaluation_budget=ledger,
        )
        feasible_cache_scores = [
            item
            for item in cache_search.candidates
            if item.is_feasible and item.horizon_system_utility is not None
        ]
        keep_score = next(
            (item for item in feasible_cache_scores if not item.changed_node_ids),
            None,
        )
        selected_score = cache_search.selected_candidate
        proposed_scores: list[CacheCandidateScore] = []
        if selected_score is not None:
            proposed_scores.append(selected_score)
        if keep_score is not None and all(
            item.candidate_index != keep_score.candidate_index
            for item in proposed_scores
        ):
            proposed_scores.append(keep_score)
        # A single additional bounded alternative permits one cache/UAV refinement
        # round; the KEEP plan is retained as the tie-breaking reference.
        if len(proposed_scores) < int(settings.max_joint_rounds):
            ranked = sorted(
                feasible_cache_scores,
                key=lambda item: (
                    -float(
                        item.horizon_system_utility
                        if item.horizon_system_utility is not None
                        else -math.inf
                    ),
                    0 if not item.changed_node_ids else 1,
                    int(item.candidate_index),
                ),
            )
            for item in ranked:
                if all(
                    existing.candidate_index != item.candidate_index
                    for existing in proposed_scores
                ):
                    proposed_scores.append(item)
                if len(proposed_scores) >= int(settings.max_joint_rounds):
                    break

        plan_targets: list[tuple[np.ndarray, str]] = []
        if proposed_scores:
            for item in proposed_scores[: int(settings.max_joint_rounds)]:
                plan_targets.append((item.target_cache_state.copy(), item.proposal_source))
        else:
            # Forecast task inputs are unavailable: cache changes remain unscored,
            # while the current-slot UAV search is still well-defined.
            plan_targets.append((cache.copy(), "KEEP_forecast_unavailable"))

        best_cache = cache.copy()
        best_positions = {key: value.copy() for key, value in uav.items()}
        best_evaluation = baseline
        best_tail = baseline_tail
        best_objective = baseline_objective
        schedule_rows: list[UavScheduleResult] = []
        search_complete = cache_search.search_complete
        evaluated_plans = 0

        for target_cache, source in plan_targets:
            schedule = self.uav_scheduler.optimize(
                previous_cache_state=cache,
                target_cache_state=target_cache,
                # Carry the best already accepted UAV target into the next
                # cache/UAV comparison. Movement legality and cost still use
                # the real pre-decision positions in ``uav``.
                current_uav_positions=best_positions,
                previous_uav_positions=uav,
                slot_input=slot_input,
                search_budget=settings,
                evaluation_budget=ledger,
                budget_category="uav",
            )
            schedule_rows.append(schedule)
            search_complete = search_complete and schedule.search_complete
            if not schedule.is_feasible or schedule.selected_evaluation is None:
                if ledger.remaining == 0:
                    break
                continue
            proposed_positions = copy_uav_positions(schedule.target_uav_positions)
            joint_evaluation = self.evaluator.evaluate(
                slot_input=slot_input,
                previous_cache_state=cache,
                previous_uav_positions=uav,
                candidate_cache_state=target_cache,
                candidate_uav_positions=proposed_positions,
                budget=ledger,
                budget_category="joint",
            )
            if not joint_evaluation.is_feasible or joint_evaluation.system_utility is None:
                search_complete = False
                if ledger.remaining == 0:
                    break
                continue
            tail = self.cache_optimizer.evaluate_forecast_tail(
                target_cache_state=target_cache,
                current_uav_positions=(
                    joint_evaluation.slot_result.final_uav_positions
                    if joint_evaluation.slot_result is not None
                    and joint_evaluation.slot_result.final_uav_positions is not None
                    else proposed_positions
                ),
                forecast_slots=forecasts[: settings.max_forecast_slots],
                model=slot_input.model,
                search_budget=settings,
                evaluation_budget=ledger,
                budget_category="joint",
                max_candidates_per_uav=2,
                max_coordinate_rounds=1,
            )
            if forecasts and (not tail.is_available or not tail.is_feasible):
                search_complete = False
                if ledger.remaining == 0:
                    break
                continue
            objective = self._total_utility(joint_evaluation.system_utility, tail)
            evaluated_plans += 1
            if objective > best_objective + float(settings.utility_tolerance):
                best_objective = objective
                best_cache = target_cache.copy()
                best_positions = proposed_positions
                best_evaluation = joint_evaluation
                best_tail = tail
            elif (
                abs(objective - best_objective) <= float(settings.utility_tolerance)
                and not np.array_equal(target_cache, cache)
            ):
                # Exact/tolerance ties preserve the original Cache and its KEEP action.
                pass
            if ledger.remaining == 0:
                search_complete = False
                break

        if best_evaluation is baseline:
            status = (
                "forecast_unavailable_uav_only"
                if not forecasts
                else "no_strict_joint_improvement"
            )
        elif np.array_equal(best_cache, cache):
            status = "uav_improved"
        else:
            status = "joint_cache_uav_improved"

        current_delta = (
            None
            if best_evaluation.system_utility is None
            else float(best_evaluation.system_utility) - float(baseline.system_utility)
        )
        has_comparable_horizon = bool(
            forecasts
            and baseline_tail.is_available
            and best_tail.is_available
            and best_tail.is_feasible
            and best_objective is not None
            and baseline_objective is not None
        )
        horizon_delta = (
            float(best_objective) - float(baseline_objective)
            if has_comparable_horizon
            else None
        )
        scenario_ids: tuple[str, ...] = ()
        if best_tail.is_available:
            scenario_ids = tuple(best_tail.effective_prediction_scenarios)
        return JointOptimizationResult(
            is_feasible=best_evaluation.is_feasible,
            search_complete=search_complete,
            target_cache_state=best_cache.copy(),
            target_uav_positions={key: value.copy() for key, value in best_positions.items()},
            baseline_current_utility=baseline.system_utility,
            final_current_utility=best_evaluation.system_utility,
            current_utility_delta=current_delta,
            baseline_horizon_utility=(
                float(baseline_objective) if baseline_tail.is_available else None
            ),
            final_horizon_utility=(
                float(best_objective) if has_comparable_horizon else None
            ),
            horizon_utility_delta=horizon_delta,
            evaluation_count=ledger.evaluations_used,
            status=status,
            failure_reason="" if best_evaluation.is_feasible else best_evaluation.failure_reason,
            cache_search=cache_search,
            uav_searches=tuple(schedule_rows),
            final_evaluation=best_evaluation,
            effective_prediction_scenarios=scenario_ids,
            diagnostics={
                "budget_counts": dict(ledger.category_counts),
                "budget_limits": dict(ledger.category_limits),
                "budget_remaining": ledger.remaining,
                "candidate_plans_evaluated": evaluated_plans,
                "cache_proposal_source": (
                    selected_score.proposal_source if selected_score is not None else "KEEP"
                ),
                "cache_forecast_failure": cache_search.failure_reason,
                "future_prediction_slots_requested": len(forecasts),
                "future_prediction_slots_used": min(
                    len(forecasts), int(settings.max_forecast_slots)
                ),
                "cache_and_uav_evaluated_jointly": True,
                "global_optimality_guaranteed": False,
            },
        )

    @staticmethod
    def _total_utility(
        current_utility: float | None,
        tail: ForecastTailResult,
    ) -> float | None:
        if current_utility is None or not math.isfinite(float(current_utility)):
            return None
        if not tail.is_available:
            return float(current_utility)
        if not tail.is_feasible or tail.system_utility is None:
            return None
        return math.fsum((float(current_utility), float(tail.system_utility)))


__all__ = ["JointOptimizationResult", "JointOptimizer"]
