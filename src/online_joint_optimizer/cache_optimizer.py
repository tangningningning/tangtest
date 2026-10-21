"""Finite candidate cache optimization over explicit predicted Slot inputs."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from decision import cache_configuration, uav_configuration

from .candidate_evaluator import (
    EvaluationBudget,
    PredictedSlotInput,
    SearchBudget,
    SlotInput,
    copy_uav_positions,
)
from .uav_scheduler import UavScheduleResult, UavScheduler


@dataclass(frozen=True)
class CacheCandidateScore:
    candidate_index: int
    target_cache_state: np.ndarray
    changed_node_ids: tuple[str, ...]
    proposal_source: str
    is_feasible: bool
    horizon_system_utility: float | None
    horizon_task_utility: float | None
    horizon_cache_cost: float | None
    horizon_movement_cost: float | None
    evaluation_horizon_slots: tuple[int, ...]
    effective_prediction_scenarios: tuple[str, ...]
    evaluation_count: int
    failure_reason: str = ""
    first_slot_schedule: UavScheduleResult | None = None
    future_slot_schedules: tuple[UavScheduleResult, ...] = ()


@dataclass(frozen=True)
class ForecastTailResult:
    is_available: bool
    is_feasible: bool
    system_utility: float | None
    task_utility: float | None
    cache_cost: float | None
    movement_cost: float | None
    final_uav_positions: Mapping[str, np.ndarray]
    evaluation_horizon_slots: tuple[int, ...]
    effective_prediction_scenarios: tuple[str, ...]
    evaluation_count: int
    slot_schedules: tuple[UavScheduleResult, ...] = ()
    failure_reason: str = ""


@dataclass(frozen=True)
class CacheOptimizationResult:
    is_feasible: bool
    search_complete: bool
    target_cache_state: np.ndarray
    selected_candidate: CacheCandidateScore | None
    candidates: tuple[CacheCandidateScore, ...]
    evaluation_count: int
    failure_reason: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _CacheTarget:
    target_state: np.ndarray
    changed_node_ids: tuple[str, ...]
    proposal_source: str


class CacheOptimizer:
    """Propose legal REPLACE configurations and score them over forecast Slots."""

    def __init__(self, uav_scheduler: UavScheduler | None = None) -> None:
        self.uav_scheduler = uav_scheduler or UavScheduler()

    def optimize(
        self,
        *,
        current_cache_state: Any,
        current_uav_positions: Mapping[str, Any],
        slot_input: SlotInput,
        forecast_slots: Sequence[PredictedSlotInput] | None,
        search_budget: SearchBudget | None = None,
        evaluation_budget: EvaluationBudget | None = None,
    ) -> CacheOptimizationResult:
        """Optimize a fixed Cache target over current and explicitly predicted Slots.

        The current Slot uses the supplied current task input. Later Slots must be
        caller-provided ``PredictedSlotInput`` values. If they are unavailable,
        candidates are returned as unscored and KEEP is retained.
        """

        settings = search_budget or SearchBudget()
        ledger = evaluation_budget or settings.new_ledger()
        current = np.asarray(current_cache_state)
        if current.ndim != 2 or not np.all(np.isin(current, [0, 1])):
            raise ValueError("current_cache_state must be a binary node-by-service matrix")
        current = current.astype(np.int8, copy=True)
        start_uav = copy_uav_positions(current_uav_positions)
        if slot_input.source_kind != "current_actual":
            raise ValueError("cache search requires a current-actual first Slot input")
        future = tuple(forecast_slots or ())
        if any(not isinstance(item, PredictedSlotInput) for item in future):
            raise TypeError("future cache-score inputs must be PredictedSlotInput, not future actual data")
        effective_future = future[: int(settings.max_forecast_slots)]

        try:
            targets = self._generate_targets(current, slot_input, settings)
        except (KeyError, TypeError, ValueError, IndexError, FloatingPointError) as exc:
            return CacheOptimizationResult(
                is_feasible=False,
                search_complete=False,
                target_cache_state=current.copy(),
                selected_candidate=None,
                candidates=(),
                evaluation_count=0,
                failure_reason=f"cache_candidate_generation_failed: {type(exc).__name__}: {exc}",
            )

        if not effective_future:
            unscored = tuple(
                CacheCandidateScore(
                    candidate_index=index,
                    target_cache_state=target.target_state.copy(),
                    changed_node_ids=target.changed_node_ids,
                    proposal_source=target.proposal_source,
                    is_feasible=False,
                    horizon_system_utility=None,
                    horizon_task_utility=None,
                    horizon_cache_cost=None,
                    horizon_movement_cost=None,
                    evaluation_horizon_slots=(),
                    effective_prediction_scenarios=(),
                    evaluation_count=0,
                    failure_reason=(
                        "future_predicted_slot_inputs_unavailable; single-slot utility is not a long-horizon score"
                    ),
                )
                for index, target in enumerate(targets)
            )
            return CacheOptimizationResult(
                is_feasible=False,
                search_complete=True,
                target_cache_state=current.copy(),
                selected_candidate=None,
                candidates=unscored,
                evaluation_count=0,
                failure_reason="future_predicted_slot_inputs_unavailable",
                diagnostics={
                    "generated_candidate_count": len(targets),
                    "effective_prediction_scenarios": [],
                    "cache_update_applied": False,
                },
            )

        score_rows: list[CacheCandidateScore] = []
        total_evaluations = 0
        search_complete = True
        for index, target in enumerate(targets):
            first_schedule = self.uav_scheduler.optimize(
                previous_cache_state=current,
                target_cache_state=target.target_state,
                current_uav_positions=start_uav,
                previous_uav_positions=start_uav,
                slot_input=slot_input,
                search_budget=settings,
                evaluation_budget=ledger,
                budget_category="cache",
                max_candidates_per_uav=2,
                max_coordinate_rounds=1,
            )
            count = int(first_schedule.evaluation_count)
            if not first_schedule.is_feasible or not first_schedule.search_complete:
                failure = first_schedule.failure_reason or "first_slot_uav_search_incomplete"
                score_rows.append(
                    self._failed_score(index, target, failure, count, (slot_input.slot_id,), (slot_input.scenario_id,))
                )
                total_evaluations += count
                search_complete = False
                if ledger.remaining == 0:
                    break
                continue
            first_eval = first_schedule.selected_evaluation
            if first_eval is None or first_eval.system_utility is None:
                score_rows.append(
                    self._failed_score(index, target, "first_slot_utility_missing", count, (slot_input.slot_id,), (slot_input.scenario_id,))
                )
                total_evaluations += count
                continue

            tail = self.evaluate_forecast_tail(
                target_cache_state=target.target_state,
                current_uav_positions=first_schedule.target_uav_positions,
                forecast_slots=effective_future,
                model=slot_input.model,
                search_budget=settings,
                evaluation_budget=ledger,
                budget_category="cache",
                max_candidates_per_uav=2,
                max_coordinate_rounds=1,
            )
            count += int(tail.evaluation_count)
            total_evaluations += count
            if not tail.is_available or not tail.is_feasible or tail.system_utility is None:
                score_rows.append(
                    self._failed_score(
                        index,
                        target,
                        tail.failure_reason or "forecast_tail_incomplete",
                        count,
                        (slot_input.slot_id, *tail.evaluation_horizon_slots),
                        (slot_input.scenario_id, *tail.effective_prediction_scenarios),
                        first_schedule,
                    )
                )
                search_complete = False
                if ledger.remaining == 0:
                    break
                continue

            score_rows.append(
                CacheCandidateScore(
                    candidate_index=index,
                    target_cache_state=target.target_state.copy(),
                    changed_node_ids=target.changed_node_ids,
                    proposal_source=target.proposal_source,
                    is_feasible=True,
                    horizon_system_utility=math.fsum(
                        [float(first_eval.system_utility), float(tail.system_utility)]
                    ),
                    horizon_task_utility=math.fsum(
                        [float(first_eval.task_utility or 0.0), float(tail.task_utility or 0.0)]
                    ),
                    horizon_cache_cost=math.fsum(
                        [float(first_eval.cache_cost or 0.0), float(tail.cache_cost or 0.0)]
                    ),
                    horizon_movement_cost=math.fsum(
                        [float(first_eval.movement_cost or 0.0), float(tail.movement_cost or 0.0)]
                    ),
                    evaluation_horizon_slots=(
                        int(slot_input.slot_id),
                        *tail.evaluation_horizon_slots,
                    ),
                    effective_prediction_scenarios=(
                        str(slot_input.scenario_id),
                        *tail.effective_prediction_scenarios,
                    ),
                    evaluation_count=count,
                    first_slot_schedule=first_schedule,
                    future_slot_schedules=tail.slot_schedules,
                )
            )
            if ledger.remaining == 0 and index + 1 < len(targets):
                search_complete = False
                break

        keep_score = next(
            (item for item in score_rows if not item.changed_node_ids),
            None,
        )
        if keep_score is None or not keep_score.is_feasible:
            return CacheOptimizationResult(
                is_feasible=False,
                search_complete=False,
                target_cache_state=current.copy(),
                selected_candidate=None,
                candidates=tuple(score_rows),
                evaluation_count=total_evaluations,
                failure_reason=(
                    "keep_baseline_unavailable"
                    if keep_score is None
                    else f"keep_baseline_failed: {keep_score.failure_reason}"
                ),
                diagnostics={"budget_remaining": ledger.remaining},
            )

        selected = keep_score
        for score in score_rows:
            if (
                score.is_feasible
                and score.horizon_system_utility is not None
                and selected.horizon_system_utility is not None
                and score.horizon_system_utility
                > selected.horizon_system_utility + float(settings.utility_tolerance)
            ):
                selected = score
        return CacheOptimizationResult(
            is_feasible=True,
            search_complete=search_complete,
            target_cache_state=selected.target_cache_state.copy(),
            selected_candidate=selected,
            candidates=tuple(score_rows),
            evaluation_count=total_evaluations,
            failure_reason="" if search_complete else "candidate_or_evaluation_budget_incomplete",
            diagnostics={
                "budget_category": "cache",
                "budget_remaining": ledger.remaining,
                "generated_candidate_count": len(targets),
                "scored_candidate_count": len(score_rows),
                "future_prediction_slots_requested": len(future),
                "future_prediction_slots_used": len(effective_future),
                "truncated_prediction_slots": max(0, len(future) - len(effective_future)),
                "cache_candidate_budget_includes_keep": True,
                "cache_target_fixed_over_horizon": True,
                "uav_rescheduled_each_slot": True,
                "selected_proposal_source": selected.proposal_source,
            },
        )

    def evaluate_forecast_tail(
        self,
        *,
        target_cache_state: Any,
        current_uav_positions: Mapping[str, Any],
        forecast_slots: Sequence[PredictedSlotInput],
        model: Any,
        search_budget: SearchBudget,
        evaluation_budget: EvaluationBudget,
        budget_category: str,
        max_candidates_per_uav: int = 2,
        max_coordinate_rounds: int = 1,
    ) -> ForecastTailResult:
        """Evaluate predicted future Slots, reoptimizing UAV positions at each Slot."""

        positions = copy_uav_positions(current_uav_positions)
        if not forecast_slots:
            return ForecastTailResult(
                is_available=False,
                is_feasible=False,
                system_utility=None,
                task_utility=None,
                cache_cost=None,
                movement_cost=None,
                final_uav_positions=positions,
                evaluation_horizon_slots=(),
                effective_prediction_scenarios=(),
                evaluation_count=0,
                failure_reason="future_predicted_slot_inputs_unavailable",
            )
        cache_target = np.asarray(target_cache_state, dtype=np.int8).copy()
        schedule_rows: list[UavScheduleResult] = []
        utilities: list[float] = []
        task_utilities: list[float] = []
        cache_costs: list[float] = []
        movement_costs: list[float] = []
        slot_ids: list[int] = []
        scenario_ids: list[str] = []
        evaluation_count = 0
        for forecast in forecast_slots:
            predicted_input = forecast.as_slot_input(model)
            schedule = self.uav_scheduler.optimize(
                previous_cache_state=cache_target,
                target_cache_state=cache_target,
                current_uav_positions=positions,
                previous_uav_positions=positions,
                slot_input=predicted_input,
                search_budget=search_budget,
                evaluation_budget=evaluation_budget,
                budget_category=budget_category,
                max_candidates_per_uav=max_candidates_per_uav,
                max_coordinate_rounds=max_coordinate_rounds,
            )
            evaluation_count += int(schedule.evaluation_count)
            if not schedule.is_feasible or not schedule.search_complete:
                return ForecastTailResult(
                    is_available=True,
                    is_feasible=False,
                    system_utility=None,
                    task_utility=None,
                    cache_cost=None,
                    movement_cost=None,
                    final_uav_positions=positions,
                    evaluation_horizon_slots=tuple(slot_ids),
                    effective_prediction_scenarios=tuple(scenario_ids),
                    evaluation_count=evaluation_count,
                    slot_schedules=tuple(schedule_rows),
                    failure_reason=schedule.failure_reason or "forecast_uav_search_incomplete",
                )
            selected = schedule.selected_evaluation
            if selected is None or selected.system_utility is None or selected.slot_result is None:
                return ForecastTailResult(
                    is_available=True,
                    is_feasible=False,
                    system_utility=None,
                    task_utility=None,
                    cache_cost=None,
                    movement_cost=None,
                    final_uav_positions=positions,
                    evaluation_horizon_slots=tuple(slot_ids),
                    effective_prediction_scenarios=tuple(scenario_ids),
                    evaluation_count=evaluation_count,
                    slot_schedules=tuple(schedule_rows),
                    failure_reason="forecast_slot_result_missing",
                )
            utilities.append(float(selected.system_utility))
            task_utilities.append(float(selected.task_utility or 0.0))
            cache_costs.append(float(selected.cache_cost or 0.0))
            movement_costs.append(float(selected.movement_cost or 0.0))
            schedule_rows.append(schedule)
            slot_ids.append(int(forecast.slot_id))
            scenario_ids.append(str(forecast.scenario_id))
            final_positions = selected.slot_result.final_uav_positions
            if final_positions is None:
                return ForecastTailResult(
                    is_available=True,
                    is_feasible=False,
                    system_utility=None,
                    task_utility=None,
                    cache_cost=None,
                    movement_cost=None,
                    final_uav_positions=positions,
                    evaluation_horizon_slots=tuple(slot_ids),
                    effective_prediction_scenarios=tuple(scenario_ids),
                    evaluation_count=evaluation_count,
                    slot_schedules=tuple(schedule_rows),
                    failure_reason="forecast_slot_final_uav_positions_missing",
                )
            positions = copy_uav_positions(final_positions)

        return ForecastTailResult(
            is_available=True,
            is_feasible=True,
            system_utility=math.fsum(utilities),
            task_utility=math.fsum(task_utilities),
            cache_cost=math.fsum(cache_costs),
            movement_cost=math.fsum(movement_costs),
            final_uav_positions=positions,
            evaluation_horizon_slots=tuple(slot_ids),
            effective_prediction_scenarios=tuple(scenario_ids),
            evaluation_count=evaluation_count,
            slot_schedules=tuple(schedule_rows),
        )

    def _generate_targets(
        self,
        current: np.ndarray,
        slot_input: SlotInput,
        settings: SearchBudget,
    ) -> list[_CacheTarget]:
        node_ids = tuple(str(item) for item in slot_input.model.cache_nodes["node_ids"])
        if current.shape[0] != len(node_ids):
            raise ValueError("current Cache rows do not align with configured cache nodes")
        targets = [_CacheTarget(current.copy(), (), "KEEP")]
        per_node: list[list[Any]] = []
        for node_index, node_id in enumerate(node_ids):
            proposals = cache_configuration.generate_bounded_cache_target_configurations(
                current,
                node_index,
                dict(slot_input.model.service_catalog),
                copy.deepcopy(dict(slot_input.model.cache_nodes)),
                copy.deepcopy(dict(slot_input.scenario_slice)),
                slot_input.model.cache_config,
                prediction_horizon_mask=copy.deepcopy(slot_input.prediction_horizon_mask),
                max_candidates=max(1, int(settings.max_cache_candidates) - 1),
            )
            for proposal in proposals:
                if str(proposal.node_id) != node_id:
                    raise ValueError("cache candidate generator returned a misaligned node ID")
            per_node.append(proposals)

        composite = current.copy()
        composite_nodes: list[str] = []
        composite_sources: list[str] = []
        for node_index, proposals in enumerate(per_node):
            if proposals:
                composite[node_index] = np.asarray(
                    proposals[0].target_node_cache, dtype=np.int8
                ).copy()
                composite_nodes.append(node_ids[node_index])
                composite_sources.append(str(proposals[0].proposal_source))
        if composite_nodes:
            self._append_target(
                targets,
                current,
                composite,
                node_ids,
                "all_nodes_first_proposal:" + "+".join(composite_sources),
                int(settings.max_cache_candidates),
            )

        rank = 0
        while len(targets) < int(settings.max_cache_candidates):
            added = False
            for node_index, proposals in enumerate(per_node):
                if rank >= len(proposals):
                    continue
                proposal = proposals[rank]
                candidate = current.copy()
                candidate[node_index] = np.asarray(
                    proposal.target_node_cache, dtype=np.int8
                ).copy()
                self._append_target(
                    targets,
                    current,
                    candidate,
                    node_ids,
                    f"{node_ids[node_index]}:{proposal.proposal_source}",
                    int(settings.max_cache_candidates),
                )
                added = True
                if len(targets) >= int(settings.max_cache_candidates):
                    break
            if not added:
                break
            rank += 1
        return targets

    @staticmethod
    def _append_target(
        targets: list[_CacheTarget],
        current: np.ndarray,
        candidate: np.ndarray,
        node_ids: Sequence[str],
        proposal_source: str,
        limit: int,
    ) -> None:
        if len(targets) >= limit:
            return
        target = np.asarray(candidate, dtype=np.int8)
        if target.shape != current.shape or not np.all(np.isin(target, [0, 1])):
            raise ValueError("composed cache candidate has an invalid shape or values")
        changed = tuple(
            node_ids[index]
            for index in range(len(node_ids))
            if not np.array_equal(current[index], target[index])
        )
        if not changed:
            return
        for index, node_id in enumerate(node_ids):
            if not np.array_equal(current[index], target[index]) and not cache_configuration.is_legal_cache_replacement(
                current[index], target[index]
            ):
                raise ValueError(f"cache candidate for {node_id} is not a legal REPLACE")
        key = np.ascontiguousarray(target).tobytes()
        if any(np.ascontiguousarray(item.target_state).tobytes() == key for item in targets):
            return
        targets.append(_CacheTarget(target.copy(), changed, proposal_source))

    @staticmethod
    def _failed_score(
        index: int,
        target: _CacheTarget,
        reason: str,
        evaluation_count: int,
        slots: Sequence[int],
        scenarios: Sequence[str],
        first_schedule: UavScheduleResult | None = None,
    ) -> CacheCandidateScore:
        return CacheCandidateScore(
            candidate_index=int(index),
            target_cache_state=target.target_state.copy(),
            changed_node_ids=target.changed_node_ids,
            proposal_source=target.proposal_source,
            is_feasible=False,
            horizon_system_utility=None,
            horizon_task_utility=None,
            horizon_cache_cost=None,
            horizon_movement_cost=None,
            evaluation_horizon_slots=tuple(int(value) for value in slots),
            effective_prediction_scenarios=tuple(str(value) for value in scenarios),
            evaluation_count=int(evaluation_count),
            failure_reason=str(reason),
            first_slot_schedule=first_schedule,
        )


__all__ = [
    "CacheCandidateScore",
    "CacheOptimizationResult",
    "CacheOptimizer",
    "ForecastTailResult",
]
