"""Budgeted joint Cache/UAV action search using forecast task-resource values.

This is a held-configuration receding-horizon prototype: one joint action is
selected now and held through the forecast horizon when it is scored. It does
NOT implement a multistage stochastic scenario-tree MPC or anticipate future
observed arrivals. The caller supplies physically meaningful candidate states.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np

from decision import (
    cache_configuration,
    predictive_assignment_value,
    uav_configuration,
)


@dataclass(frozen=True)
class JointHeldActionResult:
    cache_state: np.ndarray
    uav_positions: dict[str, np.ndarray]
    expected_discounted_utility: float
    cache_cost: float
    movement_cost: float
    predicted_net_value: float
    baseline_predicted_net_value: float
    screened_candidates: int
    full_candidates: int
    total_slot_evaluations: int
    total_assignment_oracle_evaluations: int


def optimize_joint_held_action(
    previous_cache: np.ndarray,
    previous_uav_positions: dict[str, np.ndarray],
    cache_candidates: Sequence[np.ndarray],
    uav_candidates: Sequence[Mapping[str, np.ndarray]],
    origin_scenario: dict[str, Any],
    service_catalog: dict[str, Any],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    flight_boundary: tuple[float, float, float, float],
    *,
    screen_scenarios: int = 2,
    full_scenarios: int = 10,
    final_candidate_count: int = 3,
    max_joint_candidates: int = 80,
    screen_assignment_budget: int = 8,
    full_assignment_budget: int = 24,
    objective_tolerance: float = 1.0e-9,
) -> JointHeldActionResult:
    """Screen joint actions with common forecasts; full-score finalists.

    The KEEP+STAY baseline is always fully evaluated, and the returned action
    is accepted only if its estimated full-budget net value improves it.
    Actual realized long-run improvement is NOT guaranteed.
    """
    if (
        screen_scenarios < 1 or full_scenarios < 1
        or final_candidate_count < 1 or max_joint_candidates < 1
        or screen_assignment_budget < 1 or full_assignment_budget < 1
        or objective_tolerance < 0 or not np.isfinite(objective_tolerance)
    ):
        raise ValueError("invalid joint-action search budget/tolerance")

    old_cache = np.asarray(previous_cache, dtype=np.int8).copy()
    cache_configuration.validate_cache_shape_and_capacity(
        old_cache, service_catalog, cache_nodes
    )
    previous_uav = {
        str(k): np.asarray(v, dtype=float).reshape(2).copy()
        for k, v in previous_uav_positions.items()
    }
    search_config = uav_configuration.uav_search_config_from_project(project_cfg)
    base_cache = [old_cache] + [np.asarray(c, dtype=np.int8) for c in cache_candidates]
    base_uav = [previous_uav] + [
        {str(k): np.asarray(v, dtype=float).reshape(2) for k, v in c.items()}
        for c in uav_candidates
    ]
    if len(base_cache) * len(base_uav) > max_joint_candidates:
        raise ValueError("joint candidate Cartesian product exceeds explicit budget")

    s = int(np.asarray(origin_scenario["arrival"]).shape[0])
    coarse_indices = list(range(min(screen_scenarios, s)))
    detailed_indices = list(range(min(full_scenarios, s)))
    beta = float(cache_config.beta_discount)
    if not 0 < beta <= 1:
        raise ValueError("cache discount must be in (0,1]")

    # Unique physically feasible joint configurations, always starting with
    # KEEP+STAY. A candidate may change both Cache and multiple UAVs together.
    configurations: list[tuple[np.ndarray, dict[str, np.ndarray], float, float]] = []
    seen: set[tuple[bytes, tuple[tuple[str, bytes], ...]]] = set()
    for cache_state, positions in product(base_cache, base_uav):
        cache_state = np.asarray(cache_state, dtype=np.int8).copy()
        cache_configuration.validate_cache_shape_and_capacity(
            cache_state, service_catalog, cache_nodes
        )
        new_uav = {k: np.asarray(v, dtype=float).copy() for k, v in positions.items()}
        if set(new_uav) != set(previous_uav):
            raise ValueError("each UAV candidate must contain all configured UAV IDs")
        cache_triggers = np.any(cache_state != old_cache, axis=1).astype(np.int8)
        uav_triggers = {
            k: int(
                np.linalg.norm(new_uav[k] - previous_uav[k])
                > search_config.position_tolerance_m
            )
            for k in previous_uav
        }
        uav_configuration.validate_uav_configuration_transition(
            previous_uav, new_uav, uav_triggers, project_cfg, flight_boundary
        )
        cache_cost, _ = cache_configuration.cache_update_cost(
            old_cache, cache_state, cache_triggers,
            service_catalog["service_sizes_mb"], cache_nodes, cache_config
        )
        _, movement_cost, _, _, _ = uav_configuration.movement_summary(
            previous_uav, new_uav, uav_triggers, search_config
        )
        key = (
            np.ascontiguousarray(cache_state).tobytes(),
            tuple(
                (k, np.ascontiguousarray(new_uav[k]).tobytes())
                for k in sorted(new_uav)
            ),
        )
        if key not in seen:
            seen.add(key)
            configurations.append((cache_state, new_uav, cache_cost, movement_cost))
    if not configurations:
        raise AssertionError("KEEP+STAY baseline is missing")

    total_slots = 0
    total_oracles = 0

    def score(index: int, scenario_ids: list[int], budget: int) -> tuple[float, float]:
        nonlocal total_slots, total_oracles
        cache_state, positions, cache_cost, move_cost = configurations[index]
        result = predictive_assignment_value.evaluate_held_configuration(
            origin_scenario, cache_state, positions, project_cfg,
            cache_nodes, cache_config,
            scenario_indices=scenario_ids,
            discount_beta=beta,
            assignment_budget_per_slot=budget,
            assignment_exact_task_limit=4,
        )
        total_slots += result.slot_evaluations
        total_oracles += result.assignment_oracle_evaluations
        return (
            result.mean_discounted_task_utility - cache_cost - move_cost,
            result.mean_discounted_task_utility,
        )

    coarse = [(i, score(i, coarse_indices, screen_assignment_budget)[0])
              for i in range(len(configurations))]
    coarse.sort(key=lambda row: (-row[1], row[0]))
    finalists = [0]
    for index, _ in coarse:
        if index != 0 and len(finalists) < final_candidate_count:
            finalists.append(index)
    baseline_net, baseline_q = score(0, detailed_indices, full_assignment_budget)
    best_index, best_net, best_q = 0, baseline_net, baseline_q
    for index in finalists:
        if index == 0:
            continue
        net, task_q = score(index, detailed_indices, full_assignment_budget)
        if net > best_net + objective_tolerance:
            best_index, best_net, best_q = index, net, task_q
    cache_state, new_uav, cache_cost, move_cost = configurations[best_index]
    return JointHeldActionResult(
        cache_state=cache_state.copy(),
        uav_positions={k: value.copy() for k, value in new_uav.items()},
        expected_discounted_utility=float(best_q),
        cache_cost=float(cache_cost),
        movement_cost=float(move_cost),
        predicted_net_value=float(best_net),
        baseline_predicted_net_value=float(baseline_net),
        screened_candidates=len(configurations),
        full_candidates=len(finalists),
        total_slot_evaluations=total_slots,
        total_assignment_oracle_evaluations=total_oracles,
    )
