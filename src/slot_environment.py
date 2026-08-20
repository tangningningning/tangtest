#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-slot system execution.

本模块把一个时隙内已经完成的模块按 11.pdf 时间语义串起来：

previous configuration state + external trigger Z(t) + prediction scenario slice
-> trigger-conditioned configuration
-> actual-state Transfer-Exchange Matching + KKT
-> actual task utility Q(t)
-> system utility G(t)
-> next configuration state

本模块不实现 PPO、不定义 observation/action distribution、不跑 episode。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import cache_configuration
import configuration_optimizer
import kkt_resource_allocation as kkt
import project_config
import task_matching
import task_scenario
import uav_configuration


@dataclass(frozen=True)
class SlotExecutionResult:
    slot_id: int | None
    time_s: float | None
    step_feasible: bool
    infeasible_reason: str
    input_cache_triggers: list[int]
    input_uav_triggers: dict[str, int]
    previous_cache_state: np.ndarray
    previous_uav_positions: dict[str, tuple[float, float]]
    final_cache_state: np.ndarray | None
    final_uav_positions: dict[str, tuple[float, float]] | None
    predictive_configuration_utility: float | None
    predictive_configuration_objective: float | None
    cache_cost: float | None
    movement_energy: float | None
    movement_cost: float | None
    actual_task_utility: float | None
    system_utility: float | None
    matching: dict[str, str]
    cpu_allocations: dict[str, dict[str, float]]
    bandwidth_allocations: dict[str, dict[str, float]]
    per_task_delay_s: dict[str, float]
    per_task_vehicle_energy_j: dict[str, float]
    per_task_utility: dict[str, float]
    configuration_summary: dict[str, Any]
    matching_summary: dict[str, Any]
    next_cache_state: np.ndarray | None
    next_uav_positions: dict[str, tuple[float, float]] | None


def _position_tuples(positions: dict[str, Any]) -> dict[str, tuple[float, float]]:
    return {
        str(node_id): (float(np.asarray(xy)[0]), float(np.asarray(xy)[1]))
        for node_id, xy in positions.items()
    }


def _matching_summary(result: task_matching.TransferExchangeMatchingResult) -> dict[str, Any]:
    return {
        "updates": result.updates,
        "transfer_updates": result.transfer_updates,
        "exchange_updates": result.exchange_updates,
        "candidate_evaluations_total": result.candidate_evaluations_total,
        "stop_reason": result.stop_reason,
        "transfer_exchange_stable": result.transfer_exchange_stable,
        "stable_check_passed": result.stable_check_passed,
        "max_remaining_delta": result.max_remaining_delta,
        "history": result.history,
    }


def _configuration_summary(result: configuration_optimizer.JointConfigurationResult) -> dict[str, Any]:
    return {
        "trigger_feasible": result.trigger_feasible,
        "infeasible_reason": result.infeasible_reason,
        "predictive_utility": result.predictive_utility,
        "predictive_objective": result.objective_value,
        "cache_cost": result.cache_cost,
        "movement_energy": result.movement_energy,
        "movement_cost": result.movement_cost,
        "outer_rounds": result.outer_rounds,
        "accepted_cache_updates": result.accepted_cache_updates,
        "accepted_uav_updates": result.accepted_uav_updates,
        "stop_reason": result.stop_reason,
    }


def execute_slot(
    previous_cache_state: np.ndarray,
    previous_uav_positions: dict[str, Any],
    cache_triggers: np.ndarray,
    uav_triggers: dict[str, int],
    scenario_slice: dict[str, Any],
    actual_tasks: list[kkt.Task],
    actual_vehicle_positions: dict[str, np.ndarray],
    local_cache_state: dict[str, set[int]],
    service_catalog: dict[str, Any],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    flight_boundary: tuple[float, float, float, float],
    slot_id: int | None = None,
    time_s: float | None = None,
) -> SlotExecutionResult:
    """Execute one slot from external trigger to actual system utility.

    Core API does not read disk. The caller supplies prediction scenario slice for
    configuration and actual current state for scheduling. Reward/system utility
    uses actual Matching/KKT output, not predictive configuration utility.
    """

    previous_cache = np.asarray(previous_cache_state, dtype=np.int8).copy()
    previous_uav = _position_tuples(previous_uav_positions)
    cache_trigger = np.asarray(cache_triggers, dtype=np.int8).copy()

    configuration = configuration_optimizer.optimize_joint_configuration(
        previous_cache,
        previous_uav,
        cache_trigger,
        dict(uav_triggers),
        scenario_slice,
        service_catalog,
        project_cfg,
        cache_nodes,
        cache_config,
        flight_boundary,
    )
    if not configuration.trigger_feasible:
        return SlotExecutionResult(
            slot_id=slot_id,
            time_s=time_s,
            step_feasible=False,
            infeasible_reason=configuration.infeasible_reason,
            input_cache_triggers=cache_trigger.astype(int).tolist(),
            input_uav_triggers=dict(uav_triggers),
            previous_cache_state=previous_cache.copy(),
            previous_uav_positions=previous_uav,
            final_cache_state=None,
            final_uav_positions=None,
            predictive_configuration_utility=None,
            predictive_configuration_objective=None,
            cache_cost=None,
            movement_energy=None,
            movement_cost=None,
            actual_task_utility=None,
            system_utility=None,
            matching={},
            cpu_allocations={},
            bandwidth_allocations={},
            per_task_delay_s={},
            per_task_vehicle_energy_j={},
            per_task_utility={},
            configuration_summary=_configuration_summary(configuration),
            matching_summary={},
            next_cache_state=None,
            next_uav_positions=None,
        )

    final_cache = np.asarray(configuration.final_cache_state, dtype=np.int8).copy()
    final_uav = dict(configuration.final_uav_positions)
    service_positions = kkt.build_service_node_positions(
        cache_nodes,
        actual_vehicle_positions,
        {uav_id: np.asarray(xy, dtype=np.float64) for uav_id, xy in final_uav.items()},
    )
    matching = task_matching.optimize_task_matching(
        actual_tasks,
        actual_vehicle_positions,
        service_positions,
        local_cache_state,
        final_cache,
        project_cfg,
        cache_nodes,
        cache_config,
    )
    if not matching.resource_feasible:
        return SlotExecutionResult(
            slot_id=slot_id,
            time_s=time_s,
            step_feasible=False,
            infeasible_reason="matching resource infeasible",
            input_cache_triggers=cache_trigger.astype(int).tolist(),
            input_uav_triggers=dict(uav_triggers),
            previous_cache_state=previous_cache.copy(),
            previous_uav_positions=previous_uav,
            final_cache_state=None,
            final_uav_positions=None,
            predictive_configuration_utility=None,
            predictive_configuration_objective=None,
            cache_cost=None,
            movement_energy=None,
            movement_cost=None,
            actual_task_utility=None,
            system_utility=None,
            matching={},
            cpu_allocations={},
            bandwidth_allocations={},
            per_task_delay_s={},
            per_task_vehicle_energy_j={},
            per_task_utility={},
            configuration_summary=_configuration_summary(configuration),
            matching_summary=_matching_summary(matching),
            next_cache_state=None,
            next_uav_positions=None,
        )

    actual_q = float(matching.task_utility_sum)
    cache_cost = float(configuration.cache_cost)
    movement_cost = float(configuration.movement_cost)
    system_utility = actual_q - cache_cost - movement_cost
    if not math.isclose(
        system_utility,
        actual_q - cache_cost - movement_cost,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError("G(t)=Q(t)-C_ca(t)-C_q(t) 恒等式失败。")

    return SlotExecutionResult(
        slot_id=slot_id,
        time_s=time_s,
        step_feasible=True,
        infeasible_reason="",
        input_cache_triggers=cache_trigger.astype(int).tolist(),
        input_uav_triggers=dict(uav_triggers),
        previous_cache_state=previous_cache.copy(),
        previous_uav_positions=previous_uav,
        final_cache_state=final_cache.copy(),
        final_uav_positions=final_uav,
        predictive_configuration_utility=float(configuration.predictive_utility),
        predictive_configuration_objective=float(configuration.objective_value),
        cache_cost=cache_cost,
        movement_energy=float(configuration.movement_energy),
        movement_cost=movement_cost,
        actual_task_utility=actual_q,
        system_utility=float(system_utility),
        matching=matching.matching,
        cpu_allocations=matching.cpu_allocations,
        bandwidth_allocations=matching.bandwidth_allocations,
        per_task_delay_s=matching.per_task_delay_s,
        per_task_vehicle_energy_j=matching.per_task_vehicle_energy_j,
        per_task_utility=matching.per_task_utility,
        configuration_summary=_configuration_summary(configuration),
        matching_summary=_matching_summary(matching),
        next_cache_state=final_cache.copy(),
        next_uav_positions=final_uav,
    )


def _changed_cache_nodes(previous: np.ndarray, final: np.ndarray | None) -> list[int]:
    if final is None:
        return []
    return [
        int(index)
        for index in range(previous.shape[0])
        if not np.array_equal(previous[index], final[index])
    ]


def _changed_uavs(
    previous: dict[str, tuple[float, float]],
    final: dict[str, tuple[float, float]] | None,
) -> list[str]:
    if final is None:
        return []
    return [
        uav_id
        for uav_id, old_xy in previous.items()
        if float(np.linalg.norm(np.asarray(final[uav_id]) - np.asarray(old_xy))) > 1e-6
    ]


def run_slot20_validation(config_path: str = "system_config.toml") -> dict[str, SlotExecutionResult]:
    cfg = project_config.load_config(config_path)
    project_config.validate_config(cfg)
    project_config.validate_for_formal_experiment(cfg)
    root = Path(cfg["_meta"]["project_root"])
    cache_cfg = cache_configuration.cache_config_from_project(cfg)
    scenarios = cache_configuration.load_prediction_scenarios(
        cache_configuration.resolve_path(root, project_config.get_config_value(cfg, "paths.prediction_scenarios"))
    )
    origin = cache_configuration.scenario_at_origin(scenarios, 0)
    service_catalog = cache_configuration.load_service_catalog(
        cache_configuration.resolve_path(root, project_config.get_config_value(cfg, "paths.service_catalog"))
    )
    cache_nodes = cache_configuration.build_cache_nodes(origin, cache_cfg)
    previous_cache = cache_configuration.deterministic_initial_cache(
        origin,
        service_catalog,
        cache_nodes,
        cache_cfg,
    )
    previous_uav = uav_configuration.initial_uav_positions_from_project(cfg)
    slot = 20
    time_s = 120.0
    actual_tasks = kkt.load_actual_tasks(root / "data/generated/tasks.csv", slot)
    roles = cache_configuration.load_json(root / "data/generated/vehicle_roles.json")
    sv_ids = {str(value) for value in roles["service_vehicles"]}
    tv_ids = {task.vehicle_id for task in actual_tasks}
    actual_positions = kkt.load_vehicle_positions(
        root / "data/generated/trajectory.csv",
        time_s,
        tv_ids | sv_ids,
    )
    local_cache = task_scenario.load_tv_local_cache(
        cache_configuration.resolve_path(root, project_config.get_config_value(cfg, "paths.tv_local_cache"))
    )
    boundary = uav_configuration.boundary_from_project(cfg)
    node_count = previous_cache.shape[0]
    first_sv = cache_nodes["node_roles"].index("SV")
    cases = {
        "A": (np.zeros(node_count, dtype=np.int8), {"UAV_0": 0, "UAV_1": 0}),
        "B": (np.eye(1, node_count, first_sv, dtype=np.int8).reshape(-1), {"UAV_0": 0, "UAV_1": 0}),
        "C": (np.zeros(node_count, dtype=np.int8), {"UAV_0": 1, "UAV_1": 0}),
        "D": (np.eye(1, node_count, first_sv, dtype=np.int8).reshape(-1), {"UAV_0": 1, "UAV_1": 0}),
    }
    results: dict[str, SlotExecutionResult] = {}
    for label, (cache_trigger, uav_trigger) in cases.items():
        result = execute_slot(
            previous_cache,
            previous_uav,
            cache_trigger,
            uav_trigger,
            origin,
            actual_tasks,
            actual_positions,
            local_cache,
            service_catalog,
            cfg,
            cache_nodes,
            cache_cfg,
            boundary,
            slot_id=slot,
            time_s=time_s,
        )
        if not result.step_feasible:
            raise AssertionError(f"Case {label} unexpectedly infeasible: {result.infeasible_reason}")
        if not math.isclose(
            float(result.system_utility),
            float(result.actual_task_utility) - float(result.cache_cost) - float(result.movement_cost),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"Case {label} G identity failed.")
        repeat = execute_slot(
            previous_cache,
            previous_uav,
            cache_trigger,
            uav_trigger,
            origin,
            actual_tasks,
            actual_positions,
            local_cache,
            service_catalog,
            cfg,
            cache_nodes,
            cache_cfg,
            boundary,
            slot_id=slot,
            time_s=time_s,
        )
        if result.matching != repeat.matching or not math.isclose(
            float(result.system_utility),
            float(repeat.system_utility),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"Case {label} repeated-run not deterministic.")
        results[label] = result
    return results


def _summary(result: SlotExecutionResult) -> dict[str, Any]:
    return {
        "step_feasible": result.step_feasible,
        "cache_triggers": result.input_cache_triggers,
        "uav_triggers": result.input_uav_triggers,
        "predictive_configuration_utility": result.predictive_configuration_utility,
        "predictive_configuration_objective": result.predictive_configuration_objective,
        "actual_task_utility": result.actual_task_utility,
        "cache_cost": result.cache_cost,
        "movement_energy": result.movement_energy,
        "movement_cost": result.movement_cost,
        "system_utility": result.system_utility,
        "changed_cache_nodes": _changed_cache_nodes(result.previous_cache_state, result.final_cache_state),
        "changed_uavs": _changed_uavs(result.previous_uav_positions, result.final_uav_positions),
        "matching": result.matching,
        "configuration_summary": result.configuration_summary,
        "matching_summary": result.matching_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute one actual slot from trigger to reward.")
    parser.add_argument("--config", default="system_config.toml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = run_slot20_validation(args.config)
    print(json.dumps({case: _summary(result) for case, result in results.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
