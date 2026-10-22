"""在有限预算内互斥优化缓存更新与 UAV 移动。"""

from __future__ import annotations

import hashlib
import math
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .cache_optimizer import (
    CacheCandidateScore,
    CacheOptimizationResult,
    CacheOptimizer,
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


@dataclass(frozen=True)
class _ActionCandidate:
    operation: str
    candidate_index: int
    cache_state: np.ndarray
    uav_positions: Mapping[str, np.ndarray]
    changed_node_ids: tuple[str, ...]
    moved_uav_ids: tuple[str, ...]
    proposal_source: str


@dataclass(frozen=True)
class _ActionScore:
    candidate: _ActionCandidate
    current_evaluation: CandidateEvaluation
    horizon_utility: float | None
    horizon_task_utility: float | None
    horizon_cache_cost: float | None
    horizon_movement_cost: float | None
    evaluation_horizon_slots: tuple[int, ...]
    effective_prediction_scenarios: tuple[str, ...]
    evaluation_count: int
    is_complete: bool
    search_complete: bool
    failure_reason: str = ""


class JointOptimizer:
    """在同一预算账本下比较 KEEP+STAY、缓存+STAY 和 KEEP+UAV-MOVE。"""

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
        """在统一预测时域内选择一种互斥操作。

        当前 Slot 对缓存操作保持 UAV 不动，对 UAV 操作保持缓存不变。选中的缓存状态
        延续到预测 Slot；预测期间缓存固定，并可评价合法 UAV 调度。所有 Slot 执行器调用
        共用同一硬预算账本。
        """

        started_at = time.perf_counter()
        settings = search_budget or SearchBudget()
        ledger = settings.new_ledger()
        forecasts = tuple(() if forecast_slots is None else forecast_slots)
        state_fingerprint = self._state_fingerprint(
            np.asarray(current_cache_state),
            current_uav_positions,
            slot_input,
            forecasts,
        )
        cache = np.asarray(current_cache_state)
        if cache.ndim != 2 or not np.all(np.isin(cache, [0, 1])):
            raise ValueError("current_cache_state must be a binary node-by-service matrix")
        cache = cache.astype(np.int8, copy=True)
        uav = copy_uav_positions(current_uav_positions)
        if any(not isinstance(item, PredictedSlotInput) for item in forecasts):
            raise TypeError("future inputs must be explicit PredictedSlotInput values")
        effective_forecasts = forecasts[: int(settings.max_forecast_slots)]

        baseline_candidate = _ActionCandidate(
            operation="KEEP_STAY",
            candidate_index=0,
            cache_state=cache.copy(),
            uav_positions=copy_uav_positions(uav),
            changed_node_ids=(),
            moved_uav_ids=(),
            proposal_source="KEEP+STAY",
        )
        baseline = self._evaluate_action(
            candidate=baseline_candidate,
            previous_cache_state=cache,
            previous_uav_positions=uav,
            slot_input=slot_input,
            forecast_slots=effective_forecasts,
            ledger=ledger,
            search_budget=settings,
            budget_category="joint",
        )
        if not baseline.current_evaluation.is_feasible or baseline.current_evaluation.system_utility is None:
            return self._failed_baseline_result(
                cache=cache,
                uav=uav,
                baseline=baseline.current_evaluation,
                ledger=ledger,
                started_at=started_at,
                state_isolation_pass=(
                    state_fingerprint is not None
                    and state_fingerprint
                    == self._state_fingerprint(
                        np.asarray(current_cache_state),
                        current_uav_positions,
                        slot_input,
                        forecasts,
                    )
                ),
            )
        if effective_forecasts and baseline.horizon_utility is None:
            state_unchanged = (
                state_fingerprint is not None
                and state_fingerprint
                == self._state_fingerprint(
                    np.asarray(current_cache_state),
                    current_uav_positions,
                    slot_input,
                    forecasts,
                )
            )
            return JointOptimizationResult(
                is_feasible=True,
                search_complete=False,
                target_cache_state=cache.copy(),
                target_uav_positions=copy_uav_positions(uav),
                baseline_current_utility=baseline.current_evaluation.system_utility,
                final_current_utility=baseline.current_evaluation.system_utility,
                current_utility_delta=0.0,
                baseline_horizon_utility=None,
                final_horizon_utility=None,
                horizon_utility_delta=None,
                evaluation_count=ledger.evaluations_used,
                status="keep_stay_baseline_horizon_unavailable",
                failure_reason=baseline.failure_reason or "baseline_forecast_evaluation_failed",
                final_evaluation=baseline.current_evaluation,
                diagnostics={
                    "J_keep_stay": None,
                    "J_cache_update": None,
                    "J_uav_move": None,
                    "candidate_count": 1,
                    "evaluated_candidate_count": 0,
                    "fully_evaluated_candidate_count": 0,
                    "submitted_candidate_count": 1,
                    "budget_evaluations_used": ledger.evaluations_used,
                    "budget_limit": ledger.max_total_evaluations,
                    "budget_remaining": ledger.remaining,
                    "budget_category_limits": dict(ledger.category_limits),
                    "budget_borrowed_counts_by_action": dict(
                        ledger.category_borrowed_counts
                    ),
                    "actual_executor_calls": ledger.executor_calls,
                    "actual_matching_calls": ledger.matching_calls,
                    "actual_kkt_solver_calls": ledger.kkt_solver_calls,
                    "executor_elapsed_seconds": ledger.elapsed_seconds,
                    "elapsed_seconds": time.perf_counter() - started_at,
                    "final_cache_changed_node_ids": [],
                    "final_cache_changed_node_count": 0,
                    "final_uav_moved_ids": [],
                    "final_uav_move_count": 0,
                    "KEEP_STAY_VALID": True,
                    "CACHE_UPDATE_MOVE_MUTUAL_EXCLUSION_PASS": True,
                    "CACHE_CANDIDATE_COVERAGE_PASS": False,
                    "UAV_CANDIDATE_COVERAGE_PASS": False,
                    "CACHE_CANDIDATE_EVALUATION_PASS": False,
                    "UAV_CANDIDATE_EVALUATION_PASS": False,
                    "STATE_ISOLATION_PASS": state_unchanged,
                    "BUDGET_WITHIN_LIMIT": ledger.evaluations_used
                    <= ledger.max_total_evaluations,
                    "SEARCH_COMPLETION_RECORDED": True,
                    "FINAL_ACTION_MATCHES_EVALUATED_CANDIDATE": self._evaluation_matches_candidate(
                        baseline.current_evaluation, baseline_candidate
                    ),
                    "search_complete": False,
                    "search_incomplete_reasons": ["keep_stay_horizon_incomplete"],
                    "reason": baseline.failure_reason or "baseline_forecast_evaluation_failed",
                },
            )

        cache_generation_error = ""
        try:
            cache_targets, cache_generation_complete = self.cache_optimizer.generate_candidates(
                current_cache_state=cache,
                slot_input=slot_input,
                search_budget=settings,
            )
        except (KeyError, TypeError, ValueError, IndexError, FloatingPointError) as exc:
            cache_targets = ()
            cache_generation_complete = False
            cache_generation_error = f"{type(exc).__name__}: {exc}"
        cache_candidates = [
            _ActionCandidate(
                operation="CACHE_UPDATE",
                candidate_index=index,
                cache_state=target.target_state.copy(),
                uav_positions=copy_uav_positions(uav),
                changed_node_ids=target.changed_node_ids,
                moved_uav_ids=(),
                proposal_source=target.proposal_source,
            )
            for index, target in enumerate(cache_targets)
            if target.changed_node_ids
        ]
        uav_generation_error = ""
        try:
            uav_moves, uav_generation_complete = self.uav_scheduler.enumerate_move_candidates(
                previous_cache_state=cache,
                current_uav_positions=uav,
                previous_uav_positions=uav,
                slot_input=slot_input,
                search_budget=settings,
            )
        except (KeyError, TypeError, ValueError, IndexError, FloatingPointError) as exc:
            uav_moves = ()
            uav_generation_complete = False
            uav_generation_error = f"{type(exc).__name__}: {exc}"
        uav_candidates = [
            _ActionCandidate(
                operation="UAV_MOVE",
                candidate_index=index,
                cache_state=cache.copy(),
                uav_positions=copy_uav_positions(move.target_uav_positions),
                changed_node_ids=(),
                moved_uav_ids=move.moved_uav_ids,
                proposal_source=move.proposal_source,
            )
            for index, move in enumerate(uav_moves)
        ]

        mutually_exclusive_candidates = all(
            not (
                not np.array_equal(candidate.cache_state, cache)
                and any(
                    not np.array_equal(candidate.uav_positions[uav_id], uav[uav_id])
                    for uav_id in uav
                )
            )
            for candidate in (*cache_candidates, *uav_candidates)
        )
        cache_candidate_coverage_pass = not cache_generation_error and all(
            any(
                np.array_equal(candidate.cache_state, target.target_state)
                for candidate in cache_candidates
            )
            for target in cache_targets
            if target.changed_node_ids
        )
        uav_candidate_coverage_pass = not uav_generation_error and all(
            any(
                all(
                    np.array_equal(
                        candidate.uav_positions[uav_id], move.target_uav_positions[uav_id]
                    )
                    for uav_id in candidate.uav_positions
                )
                for candidate in uav_candidates
            )
            for move in uav_moves
        )
        candidate_order = self._interleave(cache_candidates, uav_candidates)
        action_scores: list[_ActionScore] = []
        exhausted_categories: set[str] = set()
        for candidate_index, candidate in enumerate(candidate_order):
            budget_category = "cache" if candidate.operation == "CACHE_UPDATE" else "uav"
            if budget_category in exhausted_categories:
                continue
            score = self._evaluate_action(
                candidate=candidate,
                previous_cache_state=cache,
                previous_uav_positions=uav,
                slot_input=slot_input,
                forecast_slots=effective_forecasts,
                ledger=ledger,
                search_budget=settings,
                budget_category=budget_category,
            )
            action_scores.append(score)
            if score.failure_reason == "evaluation_budget_exhausted":
                if ledger.remaining == 0:
                    break
                exhausted_categories.add(budget_category)
                continue
            if ledger.remaining == 0 and candidate_index + 1 < len(candidate_order):
                break

        final_score = baseline
        for score in action_scores:
            if (
                score.is_complete
                and score.horizon_utility is not None
                and final_score.horizon_utility is not None
                and score.horizon_utility
                > final_score.horizon_utility + float(settings.utility_tolerance)
            ):
                final_score = score

        selected_candidate = final_score.candidate
        final_cache = selected_candidate.cache_state.copy()
        final_uav = copy_uav_positions(selected_candidate.uav_positions)
        final_evaluation = final_score.current_evaluation
        cache_scores = [
            self._cache_candidate_score(score)
            for score in action_scores
            if score.candidate.operation == "CACHE_UPDATE"
        ]
        keep_cache_score = self._cache_candidate_score(baseline)
        cache_score_rows = [keep_cache_score, *cache_scores]
        selected_cache_score = max(
            (item for item in cache_score_rows if item.is_feasible and item.horizon_system_utility is not None),
            key=lambda item: float(item.horizon_system_utility),
            default=None,
        )
        cache_candidate_evaluation_complete = all(
            score.is_complete
            for score in action_scores
            if score.candidate.operation == "CACHE_UPDATE"
        ) and len(cache_scores) == len(cache_candidates)
        cache_search_complete = bool(
            cache_generation_complete
            and cache_candidate_evaluation_complete
            and all(
                score.search_complete
                for score in action_scores
                if score.candidate.operation == "CACHE_UPDATE"
            )
        )
        cache_candidate_evaluation_pass = not cache_generation_error and (
            not cache_candidates or any(
                score.candidate.operation == "CACHE_UPDATE"
                and (
                    score.current_evaluation.evaluation_count > 0
                    or bool(score.current_evaluation.diagnostics.get("evaluation_cache_hit"))
                )
                for score in action_scores
            )
        )
        uav_candidate_evaluation_pass = not uav_generation_error and (
            not uav_candidates or any(
                score.candidate.operation == "UAV_MOVE"
                and (
                    score.current_evaluation.evaluation_count > 0
                    or bool(score.current_evaluation.diagnostics.get("evaluation_cache_hit"))
                )
                for score in action_scores
            )
        )
        cache_search = CacheOptimizationResult(
            is_feasible=selected_cache_score is not None,
            search_complete=cache_search_complete,
            target_cache_state=(
                cache.copy()
                if selected_cache_score is None
                else selected_cache_score.target_cache_state.copy()
            ),
            selected_candidate=selected_cache_score,
            candidates=tuple(cache_score_rows),
            evaluation_count=baseline.evaluation_count
            + sum(
                score.evaluation_count
                for score in action_scores
                if score.candidate.operation == "CACHE_UPDATE"
            ),
            failure_reason=(
                ""
                if cache_search_complete
                else "bounded_cache_candidate_or_evaluation_search_incomplete"
            ),
            diagnostics={
                "generated_candidate_count": len(cache_targets),
                "scored_candidate_count": len(cache_score_rows),
                "candidate_generation_complete": cache_generation_complete,
                "candidate_evaluation_complete": cache_candidate_evaluation_complete,
                "search_complete": cache_search_complete,
                "budget_remaining": ledger.remaining,
                "cache_current_slot_uses_uav_stay": True,
                "cache_target_fixed_over_forecast_slots": True,
            },
        )

        fully_scored_actions = [score for score in action_scores if score.is_complete]
        evaluated_candidates = 1 + len(fully_scored_actions)
        all_candidates_evaluated = len(action_scores) == len(candidate_order) and all(
            score.is_complete for score in action_scores
        )
        future_search_complete = baseline.search_complete and all(
            score.search_complete for score in action_scores
        )
        search_complete = bool(
            baseline.is_complete
            and cache_generation_complete
            and uav_generation_complete
            and all_candidates_evaluated
            and future_search_complete
            and cache_candidate_evaluation_complete
        )
        if any(score.failure_reason == "evaluation_budget_exhausted" for score in action_scores):
            search_complete = False
        search_incomplete_reasons: list[str] = []
        if not baseline.is_complete:
            search_incomplete_reasons.append("keep_stay_horizon_incomplete")
        if not cache_generation_complete:
            search_incomplete_reasons.append("cache_candidate_generation_incomplete")
        if not uav_generation_complete:
            search_incomplete_reasons.append("uav_candidate_generation_incomplete")
        if not all_candidates_evaluated:
            search_incomplete_reasons.append("not_all_generated_candidates_received_horizon_scores")
        if not future_search_complete:
            search_incomplete_reasons.append("future_uav_candidate_search_incomplete")
        if not cache_candidate_evaluation_complete:
            search_incomplete_reasons.append("cache_candidate_evaluation_incomplete")
        if any(score.failure_reason == "evaluation_budget_exhausted" for score in action_scores):
            search_incomplete_reasons.append("evaluation_budget_exhausted")

        final_action_matches_evaluated = any(
            score is final_score for score in (baseline, *action_scores)
        ) and np.array_equal(final_cache, selected_candidate.cache_state) and all(
            np.array_equal(final_uav[key], selected_candidate.uav_positions[key])
            for key in final_uav
        ) and self._evaluation_matches_candidate(final_evaluation, selected_candidate)
        final_state_fingerprint = self._state_fingerprint(
            np.asarray(current_cache_state),
            current_uav_positions,
            slot_input,
            forecasts,
        )
        initial_state_unchanged = (
            state_fingerprint is not None
            and final_state_fingerprint is not None
            and state_fingerprint == final_state_fingerprint
        )
        final_cache_changed_node_ids = tuple(
            str(node_id)
            for node_index, node_id in enumerate(slot_input.model.cache_nodes["node_ids"])
            if not np.array_equal(final_cache[node_index], cache[node_index])
        )
        final_uav_moved_ids = tuple(
            uav_id
            for uav_id in uav
            if not np.array_equal(final_uav[uav_id], uav[uav_id])
        )
        final_action_exclusive = not (final_cache_changed_node_ids and final_uav_moved_ids)
        operation_utilities: dict[str, float | None] = {
            "J_keep_stay": baseline.horizon_utility,
            "J_cache_update": self._best_action_utility(action_scores, "CACHE_UPDATE"),
            "J_uav_move": self._best_action_utility(action_scores, "UAV_MOVE"),
        }
        current_delta = float(final_evaluation.system_utility) - float(
            baseline.current_evaluation.system_utility
        )
        baseline_horizon = baseline.horizon_utility
        final_horizon = final_score.horizon_utility
        scenario_ids = final_score.effective_prediction_scenarios
        elapsed_seconds = time.perf_counter() - started_at
        diagnostics = {
            **operation_utilities,
            "candidate_count": 1 + len(cache_candidates) + len(uav_candidates),
            "evaluated_candidate_count": evaluated_candidates,
            "fully_evaluated_candidate_count": evaluated_candidates,
            "submitted_candidate_count": 1 + len(action_scores),
            "candidate_count_by_action": {
                "KEEP_STAY": 1,
                "CACHE_UPDATE": len(cache_candidates),
                "UAV_MOVE": len(uav_candidates),
            },
            "evaluated_candidate_count_by_action": {
                "KEEP_STAY": 1,
                "CACHE_UPDATE": sum(
                    score.candidate.operation == "CACHE_UPDATE" and score.is_complete
                    for score in action_scores
                ),
                "UAV_MOVE": sum(
                    score.candidate.operation == "UAV_MOVE" and score.is_complete
                    for score in action_scores
                ),
            },
            "submitted_candidate_count_by_action": {
                "KEEP_STAY": 1,
                "CACHE_UPDATE": sum(
                    score.candidate.operation == "CACHE_UPDATE" for score in action_scores
                ),
                "UAV_MOVE": sum(
                    score.candidate.operation == "UAV_MOVE" for score in action_scores
                ),
            },
            "unsubmitted_candidate_count_by_action": {
                "CACHE_UPDATE": len(cache_candidates)
                - sum(score.candidate.operation == "CACHE_UPDATE" for score in action_scores),
                "UAV_MOVE": len(uav_candidates)
                - sum(score.candidate.operation == "UAV_MOVE" for score in action_scores),
            },
            "final_cache_changed_node_ids": list(final_cache_changed_node_ids),
            "final_cache_changed_node_count": len(final_cache_changed_node_ids),
            "final_uav_moved_ids": list(final_uav_moved_ids),
            "final_uav_move_count": len(final_uav_moved_ids),
            "cache_candidate_coverage_pass": cache_candidate_coverage_pass,
            "uav_candidate_coverage_pass": uav_candidate_coverage_pass,
            "cache_candidate_evaluation_pass": cache_candidate_evaluation_pass,
            "uav_candidate_evaluation_pass": uav_candidate_evaluation_pass,
            "candidate_generation_complete": bool(
                cache_generation_complete and uav_generation_complete
            ),
            "cache_candidate_generation_complete": cache_generation_complete,
            "uav_candidate_generation_complete": uav_generation_complete,
            "cache_candidate_generation_error": cache_generation_error,
            "uav_candidate_generation_error": uav_generation_error,
            "candidate_evaluation_complete": all_candidates_evaluated,
            "cache_candidate_evaluation_complete": cache_candidate_evaluation_complete,
            "future_candidate_search_complete": future_search_complete,
            "search_complete": search_complete,
            "search_incomplete_reasons": search_incomplete_reasons,
            "search_complete_semantics": (
                "all generated legal current-action candidates received a completed horizon evaluation, "
                "future UAV candidate searches completed, no generator truncation, and the shared hard total budget sufficed"
            ),
            "budget_evaluations_used": ledger.evaluations_used,
            "budget_limit": ledger.max_total_evaluations,
            "budget_remaining": ledger.remaining,
            "budget_category_limits": dict(ledger.category_limits),
            "budget_counts_by_action": dict(ledger.category_counts),
            "budget_borrowed_counts_by_action": dict(
                ledger.category_borrowed_counts
            ),
            "budget_categories_at_limit": [
                category
                for category, limit in ledger.category_limits.items()
                if int(ledger.category_counts.get(category, 0)) >= int(limit)
            ],
            "evaluator_cache_hits": ledger.cache_hits,
            "actual_executor_calls": ledger.executor_calls,
            "actual_matching_calls": ledger.matching_calls,
            "actual_kkt_solver_calls": ledger.kkt_solver_calls,
            "executor_elapsed_seconds": ledger.elapsed_seconds,
            "elapsed_seconds": elapsed_seconds,
            "future_prediction_slots_requested": len(forecasts),
            "future_prediction_slots_used": len(effective_forecasts),
            "future_prediction_slots_truncated": max(
                0, len(forecasts) - len(effective_forecasts)
            ),
            "evaluated_horizon_slots": list(final_score.evaluation_horizon_slots),
            "evaluated_prediction_scenarios": list(final_score.effective_prediction_scenarios),
            "CACHE_UPDATE_MOVE_MUTUAL_EXCLUSION_PASS": bool(
                mutually_exclusive_candidates and final_action_exclusive
            ),
            "KEEP_STAY_VALID": bool(
                baseline.current_evaluation.is_feasible
                and baseline.current_evaluation.system_utility is not None
            ),
            "CACHE_CANDIDATE_COVERAGE_PASS": cache_candidate_coverage_pass,
            "UAV_CANDIDATE_COVERAGE_PASS": uav_candidate_coverage_pass,
            "CACHE_CANDIDATE_EVALUATION_PASS": cache_candidate_evaluation_pass,
            "UAV_CANDIDATE_EVALUATION_PASS": uav_candidate_evaluation_pass,
            "STATE_ISOLATION_PASS": initial_state_unchanged,
            "BUDGET_WITHIN_LIMIT": ledger.evaluations_used <= ledger.max_total_evaluations,
            "SEARCH_COMPLETION_RECORDED": True,
            "FINAL_ACTION_MATCHES_EVALUATED_CANDIDATE": bool(
                final_action_matches_evaluated and final_score.is_complete
            ),
            "candidate_results": [self._score_summary(score) for score in action_scores],
            "baseline_result": self._score_summary(baseline),
            "final_action": selected_candidate.operation,
        }
        return JointOptimizationResult(
            is_feasible=bool(final_evaluation.is_feasible),
            search_complete=search_complete,
            target_cache_state=final_cache,
            target_uav_positions=final_uav,
            baseline_current_utility=baseline.current_evaluation.system_utility,
            final_current_utility=final_evaluation.system_utility,
            current_utility_delta=current_delta,
            baseline_horizon_utility=baseline_horizon,
            final_horizon_utility=final_horizon,
            horizon_utility_delta=(
                None
                if baseline_horizon is None or final_horizon is None
                else float(final_horizon) - float(baseline_horizon)
            ),
            evaluation_count=ledger.evaluations_used,
            status=selected_candidate.operation.lower(),
            failure_reason=(
                ""
                if final_score.is_complete
                else final_score.failure_reason or "selected_action_horizon_evaluation_incomplete"
            ),
            cache_search=cache_search,
            uav_searches=(),
            final_evaluation=final_evaluation,
            effective_prediction_scenarios=tuple(scenario_ids),
            diagnostics=diagnostics,
        )

    def _evaluate_action(
        self,
        *,
        candidate: _ActionCandidate,
        previous_cache_state: np.ndarray,
        previous_uav_positions: Mapping[str, np.ndarray],
        slot_input: SlotInput,
        forecast_slots: Sequence[PredictedSlotInput],
        ledger: EvaluationBudget,
        search_budget: SearchBudget,
        budget_category: str,
    ) -> _ActionScore:
        """评价一种互斥的当前操作及其预测未来时域。"""

        cache_changed = not np.array_equal(previous_cache_state, candidate.cache_state)
        uav_moved = any(
            not np.array_equal(previous_uav_positions[uav_id], candidate.uav_positions[uav_id])
            for uav_id in previous_uav_positions
        )
        if cache_changed and uav_moved:
            raise ValueError("a current action cannot update Cache and move a UAV together")
        current = self.evaluator.evaluate(
            slot_input=slot_input,
            previous_cache_state=previous_cache_state,
            previous_uav_positions=previous_uav_positions,
            candidate_cache_state=candidate.cache_state,
            candidate_uav_positions=candidate.uav_positions,
            budget=ledger,
            budget_category=budget_category,
        )
        if not current.is_feasible or current.system_utility is None:
            is_complete = current.failure_reason.startswith("formal_slot_infeasible:")
            return _ActionScore(
                candidate=candidate,
                current_evaluation=current,
                horizon_utility=None,
                horizon_task_utility=None,
                horizon_cache_cost=None,
                horizon_movement_cost=None,
                evaluation_horizon_slots=(int(slot_input.slot_id),),
                effective_prediction_scenarios=(str(slot_input.scenario_id),),
                evaluation_count=current.evaluation_count,
                is_complete=is_complete,
                search_complete=is_complete,
                failure_reason=current.failure_reason,
            )

        utilities = [float(current.system_utility)]
        task_utilities = [float(current.task_utility or 0.0)]
        cache_costs = [float(current.cache_cost or 0.0)]
        movement_costs = [float(current.movement_cost or 0.0)]
        slot_ids = [int(slot_input.slot_id)]
        scenario_ids = [str(slot_input.scenario_id)]
        evaluation_count = int(current.evaluation_count)
        future_search_complete = True
        if forecast_slots:
            final_uav = (
                current.slot_result.final_uav_positions
                if current.slot_result is not None
                and current.slot_result.final_uav_positions is not None
                else candidate.uav_positions
            )
            tail = self.cache_optimizer.evaluate_forecast_tail(
                target_cache_state=candidate.cache_state,
                current_uav_positions=final_uav,
                forecast_slots=forecast_slots,
                model=slot_input.model,
                search_budget=search_budget,
                evaluation_budget=ledger,
                budget_category=budget_category,
                max_candidates_per_uav=2,
                max_coordinate_rounds=1,
            )
            evaluation_count += int(tail.evaluation_count)
            future_search_complete = tail.search_complete
            if not tail.is_available or not tail.is_feasible or tail.system_utility is None:
                return _ActionScore(
                    candidate=candidate,
                    current_evaluation=current,
                    horizon_utility=None,
                    horizon_task_utility=None,
                    horizon_cache_cost=None,
                    horizon_movement_cost=None,
                    evaluation_horizon_slots=tuple(slot_ids) + tail.evaluation_horizon_slots,
                    effective_prediction_scenarios=tuple(scenario_ids)
                    + tail.effective_prediction_scenarios,
                    evaluation_count=evaluation_count,
                    is_complete=False,
                    search_complete=False,
                    failure_reason=tail.failure_reason or "forecast_tail_evaluation_failed",
                )
            utilities.append(float(tail.system_utility))
            task_utilities.append(float(tail.task_utility or 0.0))
            cache_costs.append(float(tail.cache_cost or 0.0))
            movement_costs.append(float(tail.movement_cost or 0.0))
            slot_ids.extend(tail.evaluation_horizon_slots)
            scenario_ids.extend(tail.effective_prediction_scenarios)

        return _ActionScore(
            candidate=candidate,
            current_evaluation=current,
            horizon_utility=math.fsum(utilities),
            horizon_task_utility=math.fsum(task_utilities),
            horizon_cache_cost=math.fsum(cache_costs),
            horizon_movement_cost=math.fsum(movement_costs),
            evaluation_horizon_slots=tuple(slot_ids),
            effective_prediction_scenarios=tuple(scenario_ids),
            evaluation_count=evaluation_count,
            is_complete=True,
            search_complete=future_search_complete,
            failure_reason=(
                ""
                if future_search_complete
                else "forecast_UAV_candidate_search_incomplete"
            ),
        )

    @staticmethod
    def _interleave(
        cache_candidates: Sequence[_ActionCandidate],
        uav_candidates: Sequence[_ActionCandidate],
    ) -> list[_ActionCandidate]:
        """交替安排两类操作候选，使两类候选共用同一评价预算。"""

        ordered: list[_ActionCandidate] = []
        for index in range(max(len(cache_candidates), len(uav_candidates))):
            if index < len(cache_candidates):
                ordered.append(cache_candidates[index])
            if index < len(uav_candidates):
                ordered.append(uav_candidates[index])
        return ordered

    @staticmethod
    def _cache_candidate_score(score: _ActionScore) -> CacheCandidateScore:
        candidate = score.candidate
        return CacheCandidateScore(
            candidate_index=int(candidate.candidate_index),
            target_cache_state=candidate.cache_state.copy(),
            changed_node_ids=candidate.changed_node_ids,
            proposal_source=candidate.proposal_source,
            is_feasible=(
                score.is_complete
                and score.horizon_utility is not None
                and score.current_evaluation.is_feasible
            ),
            horizon_system_utility=score.horizon_utility,
            horizon_task_utility=score.horizon_task_utility,
            horizon_cache_cost=score.horizon_cache_cost,
            horizon_movement_cost=score.horizon_movement_cost,
            evaluation_horizon_slots=score.evaluation_horizon_slots,
            effective_prediction_scenarios=score.effective_prediction_scenarios,
            evaluation_count=score.evaluation_count,
            failure_reason=score.failure_reason,
        )

    @staticmethod
    def _best_action_utility(
        scores: Sequence[_ActionScore], operation: str
    ) -> float | None:
        values = [
            float(score.horizon_utility)
            for score in scores
            if score.candidate.operation == operation
            and score.is_complete
            and score.horizon_utility is not None
        ]
        return max(values) if values else None

    @staticmethod
    def _score_summary(score: _ActionScore) -> dict[str, Any]:
        candidate = score.candidate
        return {
            "operation": candidate.operation,
            "candidate_index": candidate.candidate_index,
            "proposal_source": candidate.proposal_source,
            "changed_node_ids": list(candidate.changed_node_ids),
            "moved_uav_ids": list(candidate.moved_uav_ids),
            "current_utility": score.current_evaluation.system_utility,
            "horizon_utility": score.horizon_utility,
            "is_complete": score.is_complete,
            "evaluation_count": score.evaluation_count,
            "evaluation_horizon_slots": list(score.evaluation_horizon_slots),
            "failure_reason": score.failure_reason,
        }

    @staticmethod
    def _evaluation_matches_candidate(
        evaluation: CandidateEvaluation,
        candidate: _ActionCandidate,
    ) -> bool:
        observed_cache = evaluation.diagnostics.get("final_cache_state")
        observed_uav = evaluation.diagnostics.get("final_uav_positions")
        if observed_cache is None or not isinstance(observed_uav, Mapping):
            return False
        if not np.array_equal(np.asarray(observed_cache), candidate.cache_state):
            return False
        return tuple(observed_uav) == tuple(candidate.uav_positions) and all(
            np.array_equal(
                np.asarray(observed_uav[uav_id], dtype=np.float64),
                np.asarray(candidate.uav_positions[uav_id], dtype=np.float64),
            )
            for uav_id in candidate.uav_positions
        )

    @staticmethod
    def _state_fingerprint(
        cache: np.ndarray,
        uav: Mapping[str, np.ndarray],
        slot_input: SlotInput,
        forecasts: Sequence[PredictedSlotInput],
    ) -> bytes | None:
        payload = (
            np.asarray(cache, dtype=np.int8),
            copy_uav_positions(uav),
            slot_input,
            tuple(forecasts),
        )
        try:
            return hashlib.sha256(pickle.dumps(payload, protocol=5)).digest()
        except (pickle.PickleError, TypeError, ValueError, AttributeError):
            return None

    def _failed_baseline_result(
        self,
        *,
        cache: np.ndarray,
        uav: Mapping[str, np.ndarray],
        baseline: CandidateEvaluation,
        ledger: EvaluationBudget,
        started_at: float,
        state_isolation_pass: bool,
    ) -> JointOptimizationResult:
        return JointOptimizationResult(
            is_feasible=False,
            search_complete=False,
            target_cache_state=cache.copy(),
            target_uav_positions=copy_uav_positions(uav),
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
            diagnostics={
                "budget_evaluations_used": ledger.evaluations_used,
                "budget_limit": ledger.max_total_evaluations,
                "budget_remaining": ledger.remaining,
                "budget_category_limits": dict(ledger.category_limits),
                "budget_borrowed_counts_by_action": dict(
                    ledger.category_borrowed_counts
                ),
                "actual_executor_calls": ledger.executor_calls,
                "actual_matching_calls": ledger.matching_calls,
                "actual_kkt_solver_calls": ledger.kkt_solver_calls,
                "executor_elapsed_seconds": ledger.elapsed_seconds,
                "elapsed_seconds": time.perf_counter() - started_at,
                "candidate_count": 1,
                "evaluated_candidate_count": 0,
                "fully_evaluated_candidate_count": 0,
                "submitted_candidate_count": 1,
                "search_complete": False,
                "search_incomplete_reasons": ["keep_stay_current_evaluation_failed"],
                "KEEP_STAY_VALID": False,
                "CACHE_UPDATE_MOVE_MUTUAL_EXCLUSION_PASS": True,
                "CACHE_CANDIDATE_COVERAGE_PASS": False,
                "UAV_CANDIDATE_COVERAGE_PASS": False,
                "CACHE_CANDIDATE_EVALUATION_PASS": False,
                "UAV_CANDIDATE_EVALUATION_PASS": False,
                "BUDGET_WITHIN_LIMIT": ledger.evaluations_used <= ledger.max_total_evaluations,
                "STATE_ISOLATION_PASS": state_isolation_pass,
                "SEARCH_COMPLETION_RECORDED": True,
                "FINAL_ACTION_MATCHES_EVALUATED_CANDIDATE": False,
                "final_cache_changed_node_ids": [],
                "final_cache_changed_node_count": 0,
                "final_uav_moved_ids": [],
                "final_uav_move_count": 0,
            },
        )


__all__ = ["JointOptimizationResult", "JointOptimizer"]
