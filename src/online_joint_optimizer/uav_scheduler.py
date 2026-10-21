"""Finite-budget deterministic scheduling over formal UAV target candidates."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from decision import uav_configuration

from .candidate_evaluator import (
    CandidateEvaluation,
    CandidateEvaluator,
    EvaluationBudget,
    SearchBudget,
    SlotInput,
    copy_uav_positions,
)


@dataclass(frozen=True)
class UavCandidateEvaluation:
    uav_id: str
    target_position: tuple[float, float]
    proposal_source: str
    is_feasible: bool
    system_utility: float | None
    evaluation_count: int
    failure_reason: str = ""


@dataclass(frozen=True)
class UavScheduleResult:
    is_feasible: bool
    search_complete: bool
    target_uav_positions: Mapping[str, np.ndarray]
    selected_evaluation: CandidateEvaluation | None
    candidate_evaluations: tuple[UavCandidateEvaluation, ...]
    evaluation_count: int
    coordinate_rounds_completed: int
    failure_reason: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class UavScheduler:
    """Improve one UAV at a time while scoring each move with full system utility."""

    def __init__(self, evaluator: CandidateEvaluator | None = None) -> None:
        self.evaluator = evaluator or CandidateEvaluator()

    def optimize(
        self,
        *,
        previous_cache_state: Any,
        target_cache_state: Any,
        current_uav_positions: Mapping[str, Any],
        previous_uav_positions: Mapping[str, Any],
        slot_input: SlotInput,
        search_budget: SearchBudget | None = None,
        evaluation_budget: EvaluationBudget | None = None,
        budget_category: str = "uav",
        max_candidates_per_uav: int | None = None,
        max_coordinate_rounds: int | None = None,
    ) -> UavScheduleResult:
        """Coordinate-search legal UAV targets from a common pre-action state."""

        settings = search_budget or SearchBudget()
        ledger = evaluation_budget or settings.new_ledger()
        candidate_limit = int(
            settings.max_uav_candidates_per_uav
            if max_candidates_per_uav is None
            else max_candidates_per_uav
        )
        round_limit = int(
            settings.max_uav_coordinate_rounds
            if max_coordinate_rounds is None
            else max_coordinate_rounds
        )
        if candidate_limit < 1 or round_limit < 1:
            raise ValueError("UAV candidate and coordinate-round limits must be positive")
        original_positions = copy_uav_positions(current_uav_positions)
        previous_positions = copy_uav_positions(previous_uav_positions)
        if tuple(original_positions) != tuple(previous_positions):
            raise ValueError("current and previous UAV IDs/order must match")
        uav_ids = tuple(uav_configuration.uav_ids_from_project(dict(slot_input.model.project_cfg)))
        if tuple(original_positions) != uav_ids:
            raise ValueError("UAV positions must follow configured UAV ID order")

        selected_positions = copy_uav_positions(original_positions)
        selected = self.evaluator.evaluate(
            slot_input=slot_input,
            previous_cache_state=previous_cache_state,
            previous_uav_positions=previous_positions,
            candidate_cache_state=target_cache_state,
            candidate_uav_positions=selected_positions,
            budget=ledger,
            budget_category=budget_category,
        )
        candidate_rows: list[UavCandidateEvaluation] = []
        evaluation_count = selected.evaluation_count
        if not selected.is_feasible:
            return UavScheduleResult(
                is_feasible=False,
                search_complete=selected.failure_reason != "evaluation_budget_exhausted",
                target_uav_positions=selected_positions,
                selected_evaluation=None,
                candidate_evaluations=(),
                evaluation_count=evaluation_count,
                coordinate_rounds_completed=0,
                failure_reason=selected.failure_reason,
            )

        search_complete = True
        completed_rounds = 0
        tolerance = float(settings.utility_tolerance)
        all_triggers = {uav_id: 1 for uav_id in uav_ids}
        for round_index in range(round_limit):
            improved_in_round = False
            for uav_id in uav_ids:
                try:
                    moves = uav_configuration.enumerate_uav_target_candidates(
                        selected_positions,
                        previous_positions,
                        uav_id,
                        all_triggers,
                        dict(slot_input.model.project_cfg),
                        tuple(float(value) for value in slot_input.model.flight_boundary),
                        scenario_slice=dict(slot_input.scenario_slice),
                        cache_state=np.asarray(target_cache_state, dtype=np.int8).copy(),
                        cache_nodes=copy.deepcopy(dict(slot_input.model.cache_nodes)),
                        predicted_vehicle_positions=copy.deepcopy(
                            dict(slot_input.predicted_vehicle_positions)
                        ),
                        prediction_horizon_mask=copy.deepcopy(
                            slot_input.prediction_horizon_mask
                        ),
                    )
                except (KeyError, TypeError, ValueError, IndexError, FloatingPointError) as exc:
                    search_complete = False
                    candidate_rows.append(
                        UavCandidateEvaluation(
                            uav_id=uav_id,
                            target_position=tuple(float(x) for x in selected_positions[uav_id]),
                            proposal_source="candidate_generation",
                            is_feasible=False,
                            system_utility=None,
                            evaluation_count=0,
                            failure_reason=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    continue

                stay = uav_configuration.UavTargetCandidate(
                    uav_id=uav_id,
                    before_position=selected_positions[uav_id].copy(),
                    target_position=selected_positions[uav_id].copy(),
                    proposal_source="stay",
                )
                candidates = [stay, *moves][:candidate_limit]
                for candidate in candidates:
                    target = np.asarray(candidate.target_position, dtype=np.float64).reshape(-1)
                    if target.shape != (2,) or not np.all(np.isfinite(target)):
                        candidate_rows.append(
                            UavCandidateEvaluation(
                                uav_id=uav_id,
                                target_position=(float("nan"), float("nan")),
                                proposal_source=str(candidate.proposal_source),
                                is_feasible=False,
                                system_utility=None,
                                evaluation_count=0,
                                failure_reason="candidate_target_is_not_a_finite_2_vector",
                            )
                        )
                        continue
                    trial_positions = copy_uav_positions(selected_positions)
                    if np.array_equal(target, trial_positions[uav_id]):
                        candidate_rows.append(
                            UavCandidateEvaluation(
                                uav_id=uav_id,
                                target_position=(float(target[0]), float(target[1])),
                                proposal_source=str(candidate.proposal_source),
                                is_feasible=True,
                                system_utility=float(selected.system_utility),
                                evaluation_count=0,
                                failure_reason="already_selected",
                            )
                        )
                        continue
                    try:
                        applied = uav_configuration.apply_uav_candidate(
                            trial_positions,
                            previous_positions,
                            uav_id,
                            target,
                            all_triggers,
                            dict(slot_input.model.project_cfg),
                            tuple(float(value) for value in slot_input.model.flight_boundary),
                        )
                        if not bool(applied["actual_moved"]):
                            raise ValueError("MOVE candidate did not change the UAV position")
                    except (KeyError, TypeError, ValueError, IndexError, FloatingPointError) as exc:
                        candidate_rows.append(
                            UavCandidateEvaluation(
                                uav_id=uav_id,
                                target_position=(float(target[0]), float(target[1])),
                                proposal_source=str(candidate.proposal_source),
                                is_feasible=False,
                                system_utility=None,
                                evaluation_count=0,
                                failure_reason=f"formal_candidate_rejected: {type(exc).__name__}: {exc}",
                            )
                        )
                        continue

                    evaluation = self.evaluator.evaluate(
                        slot_input=slot_input,
                        previous_cache_state=previous_cache_state,
                        previous_uav_positions=previous_positions,
                        candidate_cache_state=target_cache_state,
                        candidate_uav_positions=trial_positions,
                        budget=ledger,
                        budget_category=budget_category,
                    )
                    evaluation_count += evaluation.evaluation_count
                    candidate_rows.append(
                        UavCandidateEvaluation(
                            uav_id=uav_id,
                            target_position=(float(target[0]), float(target[1])),
                            proposal_source=str(candidate.proposal_source),
                            is_feasible=evaluation.is_feasible,
                            system_utility=evaluation.system_utility,
                            evaluation_count=evaluation.evaluation_count,
                            failure_reason=evaluation.failure_reason,
                        )
                    )
                    if evaluation.failure_reason == "evaluation_budget_exhausted":
                        search_complete = False
                        break
                    if (
                        evaluation.is_feasible
                        and evaluation.system_utility is not None
                        and selected.system_utility is not None
                        and evaluation.system_utility > selected.system_utility + tolerance
                    ):
                        selected = evaluation
                        selected_positions = trial_positions
                        improved_in_round = True
                if not search_complete:
                    break
            completed_rounds = round_index + 1
            if not search_complete or not improved_in_round:
                break

        return UavScheduleResult(
            is_feasible=selected.is_feasible,
            search_complete=search_complete,
            target_uav_positions={
                key: value.copy() for key, value in selected_positions.items()
            },
            selected_evaluation=selected,
            candidate_evaluations=tuple(candidate_rows),
            evaluation_count=evaluation_count,
            coordinate_rounds_completed=completed_rounds,
            failure_reason="" if selected.is_feasible else selected.failure_reason,
            diagnostics={
                "budget_category": budget_category,
                "budget_remaining": ledger.remaining,
                "uav_order": list(uav_ids),
                "max_candidates_per_uav": candidate_limit,
                "max_coordinate_rounds": round_limit,
            },
        )


__all__ = ["UavCandidateEvaluation", "UavScheduleResult", "UavScheduler"]
