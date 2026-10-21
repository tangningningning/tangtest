"""Isolated one-slot evaluation through the project's formal execution path."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from decision import slot_environment, task_allocation


@dataclass(frozen=True)
class SystemModelContext:
    """Static inputs required by the formal Slot executor."""

    service_catalog: Mapping[str, Any]
    project_cfg: Mapping[str, Any]
    cache_nodes: Mapping[str, Any]
    cache_config: Any
    flight_boundary: tuple[float, float, float, float]


@dataclass(frozen=True)
class SlotInput:
    """Complete current Slot input and prediction fields used for proposals."""

    slot_id: int
    time_s: float | None
    scenario_slice: Mapping[str, Any]
    tasks: Sequence[Any]
    vehicle_positions: Mapping[str, Any]
    predicted_vehicle_positions: Mapping[str, Any]
    model: SystemModelContext
    prediction_horizon_mask: Any | None = None
    scenario_id: str = ""
    source_kind: str = "current_actual"


@dataclass(frozen=True)
class PredictedSlotInput:
    """Explicit per-future-Slot forecast inputs; never inferred from future truth."""

    slot_id: int
    time_s: float | None
    scenario_slice: Mapping[str, Any]
    predicted_tasks: Sequence[Any]
    predicted_vehicle_positions: Mapping[str, Any]
    prediction_horizon_mask: Any | None = None
    scenario_id: str = ""

    def as_slot_input(self, model: SystemModelContext) -> SlotInput:
        """Adapt caller-supplied forecasts to the formal deterministic Slot API."""

        return SlotInput(
            slot_id=int(self.slot_id),
            time_s=None if self.time_s is None else float(self.time_s),
            scenario_slice=self.scenario_slice,
            tasks=self.predicted_tasks,
            vehicle_positions=self.predicted_vehicle_positions,
            predicted_vehicle_positions=self.predicted_vehicle_positions,
            model=model,
            prediction_horizon_mask=self.prediction_horizon_mask,
            scenario_id=str(self.scenario_id),
            source_kind="forecast",
        )


@dataclass(frozen=True)
class SearchBudget:
    """Hard limits shared by cache, UAV, and joint candidate evaluations."""

    max_total_evaluations: int = 256
    max_cache_evaluations: int = 128
    max_uav_evaluations: int = 64
    max_joint_evaluations: int = 64
    max_cache_candidates: int = 8
    max_forecast_slots: int = 3
    max_uav_candidates_per_uav: int = 4
    max_uav_coordinate_rounds: int = 2
    max_joint_rounds: int = 2
    utility_tolerance: float = 1.0e-9

    def __post_init__(self) -> None:
        integer_fields = (
            self.max_total_evaluations,
            self.max_cache_evaluations,
            self.max_uav_evaluations,
            self.max_joint_evaluations,
            self.max_cache_candidates,
            self.max_forecast_slots,
            self.max_uav_candidates_per_uav,
            self.max_uav_coordinate_rounds,
            self.max_joint_rounds,
        )
        if any(isinstance(value, bool) or int(value) <= 0 for value in integer_fields):
            raise ValueError("all search budgets and candidate limits must be positive integers")
        category_total = (
            int(self.max_cache_evaluations)
            + int(self.max_uav_evaluations)
            + int(self.max_joint_evaluations)
        )
        if category_total > int(self.max_total_evaluations):
            raise ValueError("per-stage evaluation limits exceed max_total_evaluations")
        if not math.isfinite(float(self.utility_tolerance)) or self.utility_tolerance < 0.0:
            raise ValueError("utility_tolerance must be finite and non-negative")

    def new_ledger(self) -> EvaluationBudget:
        return EvaluationBudget(
            max_total_evaluations=int(self.max_total_evaluations),
            category_limits={
                "cache": int(self.max_cache_evaluations),
                "uav": int(self.max_uav_evaluations),
                "joint": int(self.max_joint_evaluations),
            },
        )


@dataclass
class EvaluationBudget:
    """Mutable ledger counting attempted formal Slot evaluations."""

    max_total_evaluations: int
    category_limits: Mapping[str, int]
    evaluations_used: int = 0
    category_counts: dict[str, int] = field(default_factory=dict)

    def try_acquire(self, category: str) -> bool:
        if category not in self.category_limits:
            raise KeyError(f"unknown evaluation budget category: {category}")
        count = int(self.category_counts.get(category, 0))
        if (
            self.evaluations_used >= int(self.max_total_evaluations)
            or count >= int(self.category_limits[category])
        ):
            return False
        self.evaluations_used += 1
        self.category_counts[category] = count + 1
        return True

    @property
    def remaining(self) -> int:
        return max(0, int(self.max_total_evaluations) - int(self.evaluations_used))


@dataclass(frozen=True)
class CandidateEvaluation:
    """One official Slot result or an explicit evaluation failure."""

    is_feasible: bool
    system_utility: float | None
    task_utility: float | None
    cache_cost: float | None
    movement_cost: float | None
    slot_id: int
    time_s: float | None
    source_kind: str
    scenario_id: str
    evaluation_horizon_slots: tuple[int, ...]
    effective_prediction_scenarios: tuple[str, ...]
    evaluation_count: int
    failure_reason: str = ""
    error_type: str = ""
    task_assignments: Mapping[str, str] = field(default_factory=dict)
    cpu_allocations: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    bandwidth_allocations: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    per_task_delay_s: Mapping[str, float] = field(default_factory=dict)
    per_task_utility: Mapping[str, float] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    slot_result: Any | None = None


def copy_uav_positions(positions: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return a finite, independent 2-D position map preserving key order."""

    result: dict[str, np.ndarray] = {}
    for raw_id, raw_position in positions.items():
        position = np.asarray(raw_position, dtype=np.float64).reshape(-1)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError(f"UAV position for {raw_id!r} must be a finite 2-vector")
        result[str(raw_id)] = position.copy()
    if not result:
        raise ValueError("at least one UAV position is required")
    return result


def _failure(
    slot_input: SlotInput,
    reason: str,
    *,
    evaluation_count: int = 0,
    error_type: str = "",
) -> CandidateEvaluation:
    return CandidateEvaluation(
        is_feasible=False,
        system_utility=None,
        task_utility=None,
        cache_cost=None,
        movement_cost=None,
        slot_id=int(slot_input.slot_id),
        time_s=None if slot_input.time_s is None else float(slot_input.time_s),
        source_kind=str(slot_input.source_kind),
        scenario_id=str(slot_input.scenario_id),
        evaluation_horizon_slots=(int(slot_input.slot_id),),
        effective_prediction_scenarios=(str(slot_input.scenario_id),),
        evaluation_count=int(evaluation_count),
        failure_reason=str(reason),
        error_type=str(error_type),
    )


class CandidateEvaluator:
    """Run one candidate through execute_configured_slot with fresh mutable state."""

    def evaluate(
        self,
        *,
        slot_input: SlotInput,
        previous_cache_state: Any,
        previous_uav_positions: Mapping[str, Any],
        candidate_cache_state: Any,
        candidate_uav_positions: Mapping[str, Any],
        budget: EvaluationBudget | None = None,
        budget_category: str = "joint",
    ) -> CandidateEvaluation:
        """Evaluate a configured cache/UAV pair without mutating caller-owned inputs."""

        ledger = budget or EvaluationBudget(1, {budget_category: 1})
        if not ledger.try_acquire(budget_category):
            return _failure(slot_input, "evaluation_budget_exhausted")

        try:
            old_cache_raw = np.asarray(previous_cache_state)
            target_cache_raw = np.asarray(candidate_cache_state)
            if (
                old_cache_raw.ndim != 2
                or target_cache_raw.shape != old_cache_raw.shape
                or not np.all(np.isin(old_cache_raw, [0, 1]))
                or not np.all(np.isin(target_cache_raw, [0, 1]))
            ):
                raise ValueError("previous and candidate Cache states must be aligned binary matrices")
            old_cache = old_cache_raw.astype(np.int8, copy=True)
            target_cache = target_cache_raw.astype(np.int8, copy=True)
            old_uav = copy_uav_positions(previous_uav_positions)
            target_uav = copy_uav_positions(candidate_uav_positions)
            if tuple(old_uav) != tuple(target_uav):
                raise ValueError("previous and candidate UAV maps must have identical configured order")

            model = slot_input.model
            # Each candidate receives independent copies of data and every mutable
            # diagnostic/cache object used by Matching and KKT.
            scenario_slice = copy.deepcopy(dict(slot_input.scenario_slice))
            tasks = copy.deepcopy(list(slot_input.tasks))
            vehicle_positions = copy.deepcopy(dict(slot_input.vehicle_positions))
            service_catalog = copy.deepcopy(dict(model.service_catalog))
            project_cfg = copy.deepcopy(dict(model.project_cfg))
            cache_nodes = copy.deepcopy(dict(model.cache_nodes))
            cache_config = copy.deepcopy(model.cache_config)
            flight_boundary = tuple(float(value) for value in model.flight_boundary)
            performance_profile: dict[str, Any] = {}
            exact_q_timing_profile: dict[str, float] = {}
            result = slot_environment.execute_configured_slot(
                old_cache,
                old_uav,
                target_cache,
                target_uav,
                scenario_slice,
                tasks,
                vehicle_positions,
                service_catalog,
                project_cfg,
                cache_nodes,
                cache_config,
                flight_boundary,
                slot_id=int(slot_input.slot_id),
                time_s=None if slot_input.time_s is None else float(slot_input.time_s),
                performance_profile=performance_profile,
                matching_kkt_cache=task_allocation.KKTResourceAllocationCache(),
                node_evaluation_cache=task_allocation.NodeEvaluationCache(),
                exact_q_timing_profile=exact_q_timing_profile,
            )
            if not bool(result.step_feasible):
                return _failure(
                    slot_input,
                    f"formal_slot_infeasible: {result.infeasible_reason}",
                    evaluation_count=1,
                )
            values = {
                "system_utility": result.system_utility,
                "task_utility": result.actual_task_utility,
                "cache_cost": result.cache_cost,
                "movement_cost": result.movement_cost,
            }
            for name, value in values.items():
                if value is None or not math.isfinite(float(value)):
                    return _failure(
                        slot_input,
                        f"non_finite_{name}",
                        evaluation_count=1,
                    )
            return CandidateEvaluation(
                is_feasible=True,
                system_utility=float(result.system_utility),
                task_utility=float(result.actual_task_utility),
                cache_cost=float(result.cache_cost),
                movement_cost=float(result.movement_cost),
                slot_id=int(slot_input.slot_id),
                time_s=None if slot_input.time_s is None else float(slot_input.time_s),
                source_kind=str(slot_input.source_kind),
                scenario_id=str(slot_input.scenario_id),
                evaluation_horizon_slots=(int(slot_input.slot_id),),
                effective_prediction_scenarios=(str(slot_input.scenario_id),),
                evaluation_count=1,
                task_assignments=copy.deepcopy(result.matching),
                cpu_allocations=copy.deepcopy(result.cpu_allocations),
                bandwidth_allocations=copy.deepcopy(result.bandwidth_allocations),
                per_task_delay_s=copy.deepcopy(result.per_task_delay_s),
                per_task_utility=copy.deepcopy(result.per_task_utility),
                diagnostics={
                    "matching_summary": copy.deepcopy(result.matching_summary),
                    "configuration_summary": copy.deepcopy(result.configuration_summary),
                    "predictive_configuration_utility": result.predictive_configuration_utility,
                    "predictive_configuration_objective": result.predictive_configuration_objective,
                    "movement_energy": result.movement_energy,
                    "cache_download_delay_s": result.cache_download_delay_s,
                    "performance_profile": copy.deepcopy(performance_profile),
                    "exact_q_timing_profile": copy.deepcopy(exact_q_timing_profile),
                    "final_cache_state": None
                    if result.final_cache_state is None
                    else np.asarray(result.final_cache_state).copy(),
                    "final_uav_positions": copy.deepcopy(result.final_uav_positions),
                },
                slot_result=result,
            )
        except Exception as exc:
            return _failure(
                slot_input,
                f"candidate_evaluation_error: {type(exc).__name__}: {exc}",
                evaluation_count=1,
                error_type=type(exc).__name__,
            )


__all__ = [
    "CandidateEvaluation",
    "CandidateEvaluator",
    "EvaluationBudget",
    "PredictedSlotInput",
    "SearchBudget",
    "SlotInput",
    "SystemModelContext",
    "copy_uav_positions",
]
