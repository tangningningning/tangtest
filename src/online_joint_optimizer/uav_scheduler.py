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
class UavMoveCandidate:
    target_uav_positions: Mapping[str, np.ndarray]
    moved_uav_ids: tuple[str, ...]
    proposal_source: str


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

    def enumerate_move_candidates(
        self,
        *,
        previous_cache_state: Any,
        current_uav_positions: Mapping[str, Any],
        previous_uav_positions: Mapping[str, Any],
        slot_input: SlotInput,
        search_budget: SearchBudget | None = None,
        max_candidates_per_uav: int | None = None,
        generation_diagnostics: dict[str, Any] | None = None,
    ) -> tuple[tuple[UavMoveCandidate, ...], bool]:
        """生成有界合法 UAV 移动。

        返回布尔值只表示有限提议集是否触发上限或组合截断；需求锚点与几何
        提议不会穷举连续可行区域，诊断字典会明确记录这一限制。
        """

        settings = search_budget or SearchBudget()
        candidate_limit = int(
            settings.max_uav_candidates_per_uav
            if max_candidates_per_uav is None
            else max_candidates_per_uav
        )
        if candidate_limit < 1:
            raise ValueError("UAV candidate limit must be positive")
        previous = copy_uav_positions(previous_uav_positions)
        current = copy_uav_positions(current_uav_positions)
        if tuple(previous) != tuple(current):
            raise ValueError("current and previous UAV IDs/order must match")
        uav_ids = tuple(uav_configuration.uav_ids_from_project(dict(slot_input.model.project_cfg)))
        if tuple(current) != uav_ids:
            raise ValueError("UAV positions must follow configured UAV ID order")
        triggers = {uav_id: 1 for uav_id in uav_ids}
        cache = np.asarray(previous_cache_state, dtype=np.int8).copy()
        per_uav_moves: dict[str, list[Any]] = {}
        raw_move_counts: dict[str, int] = {}
        complete = True
        move_limit = max(0, candidate_limit - 1)
        for uav_id in uav_ids:
            moves = uav_configuration.enumerate_uav_target_candidates(
                current,
                previous,
                uav_id,
                triggers,
                dict(slot_input.model.project_cfg),
                tuple(float(value) for value in slot_input.model.flight_boundary),
                scenario_slice=dict(slot_input.scenario_slice),
                cache_state=cache.copy(),
                cache_nodes=copy.deepcopy(dict(slot_input.model.cache_nodes)),
                predicted_vehicle_positions=copy.deepcopy(
                    dict(slot_input.predicted_vehicle_positions)
                ),
                prediction_horizon_mask=copy.deepcopy(slot_input.prediction_horizon_mask),
            )
            raw_move_counts[uav_id] = len(moves)
            if len(moves) > move_limit:
                complete = False
            per_uav_moves[uav_id] = moves[:move_limit]
        active_uavs = [uav_id for uav_id, moves in per_uav_moves.items() if moves]
        if len(active_uavs) > 2 or (
            len(active_uavs) == 2
            and any(len(per_uav_moves[uav_id]) > 1 for uav_id in active_uavs)
        ):
            complete = False

        candidates: list[UavMoveCandidate] = []
        seen: set[bytes] = set()
        candidate_append_attempts = 0
        candidate_application_failures: list[str] = []

        def append_candidate(
            positions: Mapping[str, Any], moved_ids: tuple[str, ...], source: str
        ) -> None:
            nonlocal candidate_append_attempts
            if not moved_ids:
                return
            candidate_append_attempts += 1
            copied = copy_uav_positions(positions)
            key = b"".join(
                str(uav_id).encode("utf-8")
                + np.ascontiguousarray(copied[uav_id], dtype=np.float64).tobytes()
                for uav_id in uav_ids
            )
            if key in seen:
                return
            seen.add(key)
            candidates.append(
                UavMoveCandidate(copied, tuple(moved_ids), str(source))
            )

        combined_positions = copy_uav_positions(current)
        combined_ids: list[str] = []
        combined_sources: list[str] = []
        for uav_id in uav_ids:
            moves = uav_configuration.enumerate_uav_target_candidates(
                combined_positions,
                previous,
                uav_id,
                triggers,
                dict(slot_input.model.project_cfg),
                tuple(float(value) for value in slot_input.model.flight_boundary),
                scenario_slice=dict(slot_input.scenario_slice),
                cache_state=cache.copy(),
                cache_nodes=copy.deepcopy(dict(slot_input.model.cache_nodes)),
                predicted_vehicle_positions=copy.deepcopy(
                    dict(slot_input.predicted_vehicle_positions)
                ),
                prediction_horizon_mask=copy.deepcopy(slot_input.prediction_horizon_mask),
            )
            if not moves or candidate_limit <= 1:
                continue
            move = moves[0]
            try:
                applied = uav_configuration.apply_uav_candidate(
                    combined_positions,
                    previous,
                    uav_id,
                    np.asarray(move.target_position, dtype=np.float64),
                    triggers,
                    dict(slot_input.model.project_cfg),
                    tuple(float(value) for value in slot_input.model.flight_boundary),
                )
            except (KeyError, TypeError, ValueError, IndexError, FloatingPointError):
                complete = False
                candidate_application_failures.append(f"coordinated:{uav_id}")
                continue
            if not bool(applied["actual_moved"]):
                continue
            combined_positions = copy_uav_positions(combined_positions)
            combined_ids.append(uav_id)
            combined_sources.append(str(move.proposal_source))
        append_candidate(
            combined_positions,
            tuple(combined_ids),
            "coordinated_first_proposal:" + "+".join(combined_sources),
        )

        for uav_id, moves in per_uav_moves.items():
            for move in moves:
                target = np.asarray(move.target_position, dtype=np.float64).reshape(-1)
                trial = copy_uav_positions(current)
                try:
                    applied = uav_configuration.apply_uav_candidate(
                        trial,
                        previous,
                        uav_id,
                        target,
                        triggers,
                        dict(slot_input.model.project_cfg),
                        tuple(float(value) for value in slot_input.model.flight_boundary),
                    )
                except (KeyError, TypeError, ValueError, IndexError, FloatingPointError):
                    complete = False
                    candidate_application_failures.append(f"single:{uav_id}")
                    continue
                if not bool(applied["actual_moved"]):
                    continue
                append_candidate(trial, (uav_id,), str(move.proposal_source))

        if generation_diagnostics is not None:
            reasons: list[str] = []
            capped_uavs = [
                uav_id
                for uav_id in uav_ids
                if raw_move_counts.get(uav_id, 0) > move_limit
            ]
            if capped_uavs:
                reasons.append("per_uav_move_candidate_cap_omitted_proposals")
            multi_uav_combinations_limited = len(active_uavs) > 2 or (
                len(active_uavs) == 2
                and any(len(per_uav_moves[uav_id]) > 1 for uav_id in active_uavs)
            )
            if multi_uav_combinations_limited:
                reasons.append("multi_uav_target_combinations_not_enumerated")
            if candidate_application_failures:
                reasons.append("formal_candidate_application_rejected_proposals")
            # 底层只产生需求锚点和有限几何点，不穷举连续可行区域中的所有位置。
            reasons.append("finite_heuristic_target_generator_does_not_enumerate_all_legal_positions")
            if candidate_append_attempts > len(candidates):
                reasons.append("duplicate_uav_configurations_removed")
            generation_diagnostics.update(
                {
                    "generation_complete_within_bounded_proposal_policy": bool(complete),
                    "legal_action_space_exhaustive": False,
                    "candidate_limit_per_uav_including_stay": int(candidate_limit),
                    "move_limit_per_uav": int(move_limit),
                    "formal_target_candidate_capacity_per_uav": int(
                        uav_configuration.uav_search_config_from_project(
                            dict(slot_input.model.project_cfg)
                        ).target_candidate_capacity
                    ),
                    "raw_target_proposal_count_by_uav": dict(raw_move_counts),
                    "retained_target_proposal_count_by_uav": {
                        uav_id: len(per_uav_moves[uav_id]) for uav_id in uav_ids
                    },
                    "per_uav_limit_reached": capped_uavs,
                    "active_uav_ids": list(active_uavs),
                    "multi_uav_combinations_limited": bool(
                        multi_uav_combinations_limited
                    ),
                    "generated_candidate_count_unique": len(candidates),
                    "candidate_deduplications": candidate_append_attempts - len(candidates),
                    "candidate_application_failures": candidate_application_failures,
                    "incomplete_reasons": reasons,
                }
            )

        return tuple(candidates), complete

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

        evaluation_complete = True
        candidate_generation_complete = True
        coordinate_round_limit_reached = False
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
                    candidate_generation_complete = False
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
                if len(moves) > max(0, candidate_limit - 1):
                    candidate_generation_complete = False
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
                        evaluation_complete = False
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
                if not evaluation_complete:
                    break
            completed_rounds = round_index + 1
            if not evaluation_complete or not improved_in_round:
                break
            if round_index + 1 == round_limit:
                coordinate_round_limit_reached = True

        search_complete = (
            evaluation_complete
            and candidate_generation_complete
            and not coordinate_round_limit_reached
        )

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
                "candidate_generation_complete": candidate_generation_complete,
                "candidate_evaluation_complete": evaluation_complete,
                "coordinate_round_limit_reached": coordinate_round_limit_reached,
            },
        )


__all__ = [
    "UavCandidateEvaluation",
    "UavMoveCandidate",
    "UavScheduleResult",
    "UavScheduler",
]
