"""Expected task utility for one *held* Cache/UAV configuration.

Prediction scenarios come exclusively from the forecast artifact: no realized
future positions, arrivals, service IDs or task sizes enter this evaluator.
This is a building block for non-anticipative joint Cache/UAV MPC, not an
end-to-end MPC policy or a replacement for the formal rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from decision import (
    assignment_optimizer,
    cache_configuration,
    task_allocation,
)


@dataclass(frozen=True)
class PredictedAssignmentValue:
    mean_discounted_task_utility: float
    scenario_utilities: tuple[float, ...]
    scenario_count: int
    slot_evaluations: int
    assignment_oracle_evaluations: int


def evaluate_held_configuration(
    origin_scenario: dict[str, Any],
    edge_cache_state: np.ndarray,
    uav_positions: dict[str, np.ndarray],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    *,
    scenario_indices: Sequence[int] | None = None,
    horizon_mask: Sequence[bool] | None = None,
    discount_beta: float = 1.0,
    assignment_budget_per_slot: int = 24,
    assignment_exact_task_limit: int = 4,
) -> PredictedAssignmentValue:
    """Score predicted task utility with the same configuration held for H.

    Cache/UAV reconfiguration costs are intentionally EXCLUDED; an outer joint
    planner must subtract their actual state-transition costs *once*.
    Further future Cache/UAV decisions are not modeled in this evaluator.
    All scenarios share the provided configuration, preserving the first-stage
    non-anticipativity contract.
    """
    arrival = np.asarray(origin_scenario["arrival"])
    service = np.asarray(origin_scenario["service_id"])
    input_bits = np.asarray(origin_scenario["input_size"], dtype=float)
    cpu_cycles = np.asarray(origin_scenario["cpu_cycles"], dtype=float)
    positions = np.asarray(origin_scenario["positions_xy"], dtype=float)
    if arrival.ndim != 3:
        raise ValueError("arrival must have dimensions [scenarios, horizon, TV]")
    s, h, n_tv = arrival.shape
    if any(x.shape != arrival.shape for x in (service, input_bits, cpu_cycles)):
        raise ValueError("predicted task arrays must align with arrivals")
    if positions.ndim != 4 or positions.shape[:2] != (s, h) or positions.shape[-1] != 2:
        raise ValueError("positions must have dimensions [scenarios,horizon,vehicles,2]")
    vehicle_ids = [str(x) for x in origin_scenario["vehicle_ids"]]
    tv_ids = [str(x) for x in origin_scenario["task_vehicle_ids"]]
    if len(vehicle_ids) != positions.shape[2] or len(tv_ids) != n_tv:
        raise ValueError("scenario vehicle and TV IDs are not aligned")
    if len(set(vehicle_ids)) != len(vehicle_ids) or len(set(tv_ids)) != len(tv_ids):
        raise ValueError("scenario vehicle IDs must be unique")
    if not set(tv_ids).issubset(vehicle_ids):
        raise ValueError("predicted TV IDs must occur in vehicle_ids")
    if not np.isfinite(discount_beta) or not 0 < discount_beta <= 1:
        raise ValueError("discount_beta must be in (0,1]")
    selected = list(range(s)) if scenario_indices is None else list(scenario_indices)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("scenario_indices must identify nonempty unique scenarios")
    if any(i < 0 or i >= s for i in selected):
        raise ValueError("scenario index out of range")
    mask = (
        np.ones(h, dtype=bool)
        if horizon_mask is None
        else np.asarray(horizon_mask)
    )
    if mask.shape != (h,) or not np.all(np.isin(mask, [0, 1, False, True])):
        raise ValueError("horizon_mask must contain one binary entry per horizon")
    mask = mask.astype(bool)
    if not np.any(mask):
        raise ValueError("at least one horizon must be active")

    totals: list[float] = []
    evaluations = 0
    assignment_calls = 0
    for scenario_index in selected:
        discounted_utility = 0.0
        for step in range(h):
            if not mask[step]:
                continue
            xy = positions[scenario_index, step]
            if not np.all(np.isfinite(xy)):
                raise ValueError("forecast contains invalid vehicle positions")
            vehicle_positions = {
                vid: xy[index].copy()
                for index, vid in enumerate(vehicle_ids)
            }
            node_positions = task_allocation.build_service_node_positions(
                cache_nodes, vehicle_positions, uav_positions
            )
            forecast_tasks: list[task_allocation.Task] = []
            for index, vid in enumerate(tv_ids):
                if int(arrival[scenario_index, step, index]) != 1:
                    continue
                size = float(input_bits[scenario_index, step, index])
                cycles = float(cpu_cycles[scenario_index, step, index])
                if not np.isfinite(size) or not np.isfinite(cycles) or size < 0 or cycles < 0:
                    raise ValueError("forecast task size/cycles must be finite nonnegative")
                forecast_tasks.append(task_allocation.Task(
                    task_id=vid,
                    vehicle_id=vid,
                    service_id=int(service[scenario_index, step, index]),
                    input_bits=size,
                    cpu_cycles=cycles,
                ))
            solution = assignment_optimizer.optimize_assignment_with_resource_oracle(
                forecast_tasks,
                vehicle_positions,
                node_positions,
                edge_cache_state,
                project_cfg,
                cache_nodes,
                cache_config,
                max_oracle_evaluations=assignment_budget_per_slot,
                exact_task_limit=assignment_exact_task_limit,
            )
            discounted_utility += (
                discount_beta ** step * solution.task_utility_sum
            )
            assignment_calls += solution.oracle_evaluations
            evaluations += 1
        totals.append(float(discounted_utility))
    return PredictedAssignmentValue(
        mean_discounted_task_utility=float(np.mean(totals)),
        scenario_utilities=tuple(totals),
        scenario_count=len(selected),
        slot_evaluations=evaluations,
        assignment_oracle_evaluations=assignment_calls,
    )
