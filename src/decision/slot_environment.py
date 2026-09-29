#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-slot system execution.

本模块把一个时隙内已经完成的模块按 12.pdf 时间语义串起来：

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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common import project_config
from decision import cache_configuration, task_allocation, uav_configuration
from scenario import task_scenario


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


def _matching_summary(result: task_allocation.TransferExchangeMatchingResult) -> dict[str, Any]:
    return {
        "updates": result.updates,
        "transfer_updates": result.transfer_updates,
        "exchange_updates": result.exchange_updates,
        "candidate_evaluations_total": result.candidate_evaluations_total,
        "candidate_definitions_total": result.candidate_definitions_total,
        "upper_bound_screen_calls": result.upper_bound_screen_calls,
        "upper_bound_pruned_candidates": result.upper_bound_pruned_candidates,
        "transfer_upper_bound_pruned": result.transfer_upper_bound_pruned,
        "exchange_upper_bound_pruned": result.exchange_upper_bound_pruned,
        "full_evaluator_calls": result.full_evaluator_calls,
        "kkt_cache_requests": result.kkt_cache_requests,
        "kkt_cache_hits": result.kkt_cache_hits,
        "kkt_cache_misses": result.kkt_cache_misses,
        "KKT_CACHE_REQUEST_COUNT": result.kkt_cache_requests,
        "KKT_CACHE_HIT_COUNT": result.kkt_cache_hits,
        "KKT_CACHE_MISS_COUNT": result.kkt_cache_misses,
        "KKT_SOLVER_CALL_COUNT": result.kkt_solver_calls,
        "KKT_SOLVE_COUNT": result.kkt_solve_count,
        "KKT_CONVERGED_COUNT": result.kkt_converged_count,
        "KKT_MAX_ITER_HIT_COUNT": result.kkt_max_iterations_hit_count,
        "KKT_FINITE_COUNT": result.kkt_finite_count,
        "KKT_ITERATION_MEAN": result.kkt_iteration_mean,
        "KKT_ITERATION_P95": result.kkt_iteration_p95,
        "KKT_ITERATION_MAX": result.kkt_iteration_max,
        "KKT_RESOURCE_RESIDUAL_MEAN": result.kkt_resource_residual_mean,
        "KKT_RESOURCE_RESIDUAL_MAX": result.kkt_resource_residual_max,
        "KKT_STATIONARITY_RESIDUAL_MEAN": result.kkt_stationarity_residual_mean,
        "KKT_STATIONARITY_RESIDUAL_MAX": result.kkt_stationarity_residual_max,
        "KKT_COMPLEMENTARY_SLACKNESS_RESIDUAL_MEAN": (
            result.kkt_complementary_slackness_residual_mean
        ),
        "KKT_COMPLEMENTARY_SLACKNESS_RESIDUAL_MAX": (
            result.kkt_complementary_slackness_residual_max
        ),
        "KKT_SOLVE_DIAGNOSTICS": [
            dict(record) for record in result.kkt_solve_diagnostics
        ],
        "NODE_EVAL_CACHE_REQUEST_COUNT": result.node_eval_cache_requests,
        "NODE_EVAL_CACHE_HIT_COUNT": result.node_eval_cache_hits,
        "NODE_EVAL_CACHE_MISS_COUNT": result.node_eval_cache_misses,
        "NODE_EVAL_FULL_EXECUTION_COUNT": result.node_eval_full_executions,
        "IDEAL_UTILITY_REQUEST_COUNT": result.ideal_utility_cache_requests,
        "IDEAL_UTILITY_CACHE_HIT_COUNT": result.ideal_utility_cache_hits,
        "IDEAL_UTILITY_CACHE_MISS_COUNT": result.ideal_utility_cache_misses,
        "IMPACT_SCOPE_REQUEST_COUNT": result.impact_scope_requests,
        "IMPACT_SCOPE_COMPUTE_COUNT": result.impact_scope_computes,
        "IMPACT_SCOPE_REUSE_COUNT": result.impact_scope_reuses,
        "stop_reason": result.stop_reason,
        # Diagnostic aliases only: the Matching decision logic continues to
        # use the original stop_reason values above.
        "MAX_UPDATES_REACHED": result.stop_reason == "stopped_by_max_updates",
        "NO_IMPROVING_TRANSFER_EXCHANGE": result.stop_reason
        == "converged_by_no_improvement",
        "transfer_exchange_stable": result.transfer_exchange_stable,
        "stable_check_passed": result.stable_check_passed,
        "max_remaining_delta": result.max_remaining_delta,
        "history": result.history,
    }


def _configuration_summary(
    result: JointConfigurationResult,
    uav_tolerance_m: float = 1.0e-6,
) -> dict[str, Any]:
    actual_cache_update = np.any(
        result.final_cache_state != result.previous_cache_state,
        axis=1,
    ).astype(int).tolist()
    actual_uav_movement = {
        uav_id: int(
            float(
                np.linalg.norm(
                    np.asarray(result.final_uav_positions[uav_id], dtype=np.float64)
                    - np.asarray(result.previous_uav_positions[uav_id], dtype=np.float64)
                )
            )
            > float(uav_tolerance_m)
        )
        for uav_id in result.previous_uav_positions
    }
    prediction_skipped = result.stop_reason in {
        "no_trigger_prediction_skipped",
        "policy_target_prediction_skipped",
    }
    return {
        "trigger_feasible": result.trigger_feasible,
        "infeasible_reason": result.infeasible_reason,
        "requested_cache_trigger": list(result.cache_triggers),
        "actual_cache_update": actual_cache_update,
        "requested_uav_trigger": dict(result.uav_triggers),
        "actual_uav_movement": actual_uav_movement,
        "predictive_utility": None if prediction_skipped else result.predictive_utility,
        "predictive_objective": None if prediction_skipped else result.objective_value,
        "cache_cost": result.cache_cost,
        "movement_energy": result.movement_energy,
        "movement_cost": result.movement_cost,
        "outer_rounds": result.outer_rounds,
        "accepted_cache_updates": result.accepted_cache_updates,
        "accepted_uav_updates": result.accepted_uav_updates,
        "cache_candidate_diagnostics": result.cache_result_summary.get(
            "candidate_diagnostics", {}
        ),
        "uav_candidate_diagnostics": result.uav_result_summary.get(
            "candidate_diagnostics", {}
        ),
        "stop_reason": result.stop_reason,
        "predictive_evaluation_skipped": prediction_skipped,
    }


def _record_slot_performance_profile(
    profile: dict[str, Any] | None,
    configuration_seconds: float,
    matching_kkt_seconds: float,
) -> None:
    """Accumulate optional diagnostic timing without changing slot semantics."""

    if profile is None:
        return
    profile["config_seconds"] = float(profile.get("config_seconds", 0.0)) + float(
        configuration_seconds
    )
    profile["matching_kkt_seconds"] = float(profile.get("matching_kkt_seconds", 0.0)) + float(
        matching_kkt_seconds
    )


def _assert_requested_actual_trigger_contract(
    configuration: JointConfigurationResult,
    requested_cache_trigger: np.ndarray,
    requested_uav_trigger: dict[str, int],
    uav_tolerance_m: float,
) -> None:
    actual_cache_update = np.any(
        configuration.final_cache_state != configuration.previous_cache_state,
        axis=1,
    ).astype(np.int8)
    if np.any(actual_cache_update > requested_cache_trigger):
        raise AssertionError(
            "cache actual update exceeds requested trigger in slot configuration"
        )
    for uav_id, requested in requested_uav_trigger.items():
        actual = int(
            float(
                np.linalg.norm(
                    np.asarray(configuration.final_uav_positions[uav_id], dtype=np.float64)
                    - np.asarray(configuration.previous_uav_positions[uav_id], dtype=np.float64)
                )
            )
            > float(uav_tolerance_m)
        )
        if actual > int(requested):
            raise AssertionError(
                f"UAV {uav_id} actual movement exceeds requested trigger in slot configuration"
            )


def execute_slot(
    previous_cache_state: np.ndarray,
    previous_uav_positions: dict[str, Any],
    cache_triggers: np.ndarray,
    uav_triggers: dict[str, int],
    scenario_slice: dict[str, Any],
    actual_tasks: list[task_allocation.Task],
    actual_vehicle_positions: dict[str, np.ndarray],
    local_cache_state: dict[str, set[int]],
    service_catalog: dict[str, Any],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    flight_boundary: tuple[float, float, float, float],
    slot_id: int | None = None,
    time_s: float | None = None,
    performance_profile: dict[str, Any] | None = None,
    configuration_profile: cache_configuration.ConfigurationEvaluationProfile | None = None,
    configuration_override: JointConfigurationResult | None = None,
    matching_kkt_cache: task_allocation.KKTResourceAllocationCache | None = None,
    node_evaluation_cache: task_allocation.NodeEvaluationCache | None = None,
    exact_q_timing_profile: dict[str, float] | None = None,
) -> SlotExecutionResult:
    """Execute one slot from external trigger to actual system utility.

    Core API does not read disk. The caller supplies prediction scenario slice for
    configuration and actual current state for scheduling. Reward/system utility
    uses actual Matching/KKT output, not predictive configuration utility.
    """

    previous_cache = np.asarray(previous_cache_state, dtype=np.int8).copy()
    previous_uav = _position_tuples(previous_uav_positions)
    cache_trigger = np.asarray(cache_triggers, dtype=np.int8).copy()
    cache_configuration.validate_cache_shape_and_capacity(
        previous_cache,
        service_catalog,
        cache_nodes,
    )
    if cache_trigger.shape != (previous_cache.shape[0],) or not np.all(
        np.isin(cache_trigger, [0, 1])
    ):
        raise ValueError("cache_triggers 必须是与 cache node 数量一致的0/1向量。")
    if set(uav_triggers) != set(previous_uav) or any(
        int(value) not in (0, 1) for value in uav_triggers.values()
    ):
        raise ValueError("uav_triggers 必须覆盖全部 UAV 且只能取0/1。")

    no_configuration_trigger = (
        int(cache_trigger.sum()) == 0
        and not any(int(value) == 1 for value in uav_triggers.values())
    )
    configuration_requests_before = (
        int(configuration_profile.utility_requests)
        if configuration_profile is not None
        else 0
    )
    prediction_evaluation_skipped = False
    configuration_started_at = time.perf_counter() if performance_profile is not None else 0.0
    if configuration_override is not None:
        configuration = _validated_configuration_override(
            configuration_override,
            previous_cache,
            previous_uav,
            cache_trigger,
            {key: int(value) for key, value in uav_triggers.items()},
            service_catalog,
            cache_nodes,
            cache_config,
            project_cfg,
            flight_boundary,
        )
        prediction_evaluation_skipped = True
    elif no_configuration_trigger:
        # Gate=NULL 时配置状态严格保持不变；预测目标既不参与 Matching，也不参与
        # 当前 reward，因此不执行昂贵且无决策作用的场景配置评价。
        configuration = JointConfigurationResult(
            previous_cache_state=previous_cache.copy(),
            final_cache_state=previous_cache.copy(),
            previous_uav_positions=dict(previous_uav),
            final_uav_positions=dict(previous_uav),
            cache_triggers=cache_trigger.astype(int).tolist(),
            uav_triggers={key: int(value) for key, value in uav_triggers.items()},
            trigger_feasible=True,
            infeasible_reason="",
            predictive_utility=0.0,
            cache_cost=0.0,
            movement_energy=0.0,
            movement_cost=0.0,
            objective_value=0.0,
            initial_objective=0.0,
            objective_history=[],
            outer_rounds=0,
            accepted_cache_updates=0,
            accepted_uav_updates=0,
            stop_reason="no_trigger_prediction_skipped",
            cache_result_summary={},
            uav_result_summary={},
        )
        prediction_evaluation_skipped = True
    else:
        evaluation_context = cache_configuration.build_surrogate_evaluation_context(
            scenario_slice,
            local_cache_state,
            cache_nodes,
            cache_config,
            profile=configuration_profile,
            prediction_batch_parallel_workers=(
                cache_configuration.prediction_batch_parallel_workers(project_cfg)
            ),
        )
        try:
            configuration = optimize_joint_configuration(
                previous_cache,
                previous_uav,
                cache_trigger,
                dict(uav_triggers),
                scenario_slice,
                local_cache_state,
                service_catalog,
                project_cfg,
                cache_nodes,
                cache_config,
                flight_boundary,
                evaluation_context=evaluation_context,
            )
        finally:
            cache_configuration.close_prediction_batch_pool(evaluation_context)
    if performance_profile is not None:
        configuration_requests_after = (
            int(configuration_profile.utility_requests)
            if configuration_profile is not None
            else configuration_requests_before
        )
        performance_profile["online_k100_configuration_calls"] = int(
            performance_profile.get("online_k100_configuration_calls", 0)
        ) + max(configuration_requests_after - configuration_requests_before, 0)
    configuration_seconds = (
        time.perf_counter() - configuration_started_at
        if performance_profile is not None
        else 0.0
    )
    _assert_requested_actual_trigger_contract(
        configuration,
        cache_trigger,
        {key: int(value) for key, value in uav_triggers.items()},
        float(project_config.get_config_value(
            project_cfg,
            "algorithm.uav.position_tolerance_m",
            1.0e-6,
        )),
    )
    if not configuration.trigger_feasible:
        _record_slot_performance_profile(
            performance_profile,
            configuration_seconds,
            0.0,
        )
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
            configuration_summary=_configuration_summary(
                configuration,
                float(project_config.get_config_value(
                    project_cfg,
                    "algorithm.uav.position_tolerance_m",
                    1.0e-6,
                )),
            ),
            matching_summary={},
            next_cache_state=None,
            next_uav_positions=None,
        )

    matching_started_at = time.perf_counter() if performance_profile is not None else 0.0
    final_cache = np.asarray(configuration.final_cache_state, dtype=np.int8).copy()
    final_uav = dict(configuration.final_uav_positions)
    service_positions = task_allocation.build_service_node_positions(
        cache_nodes,
        actual_vehicle_positions,
        {uav_id: np.asarray(xy, dtype=np.float64) for uav_id, xy in final_uav.items()},
    )
    exact_q_matching_started_at = (
        time.perf_counter() if exact_q_timing_profile is not None else 0.0
    )
    exact_q_node_runtime_before = (
        0.0
        if exact_q_timing_profile is None
        else sum(
            float(exact_q_timing_profile.get(key, 0.0))
            for key in (
                "node_cache_lookup_runtime",
                "node_eval_miss_runtime",
                "kkt_runtime",
            )
        )
    )
    matching = task_allocation.optimize_task_matching(
        actual_tasks,
        actual_vehicle_positions,
        service_positions,
        local_cache_state,
        final_cache,
        project_cfg,
        cache_nodes,
        cache_config,
        kkt_cache=matching_kkt_cache,
        node_evaluation_cache=node_evaluation_cache,
        timing_profile=exact_q_timing_profile,
    )
    if exact_q_timing_profile is not None:
        exact_q_matching_elapsed = time.perf_counter() - exact_q_matching_started_at
        exact_q_node_elapsed = sum(
            float(exact_q_timing_profile.get(key, 0.0))
            for key in (
                "node_cache_lookup_runtime",
                "node_eval_miss_runtime",
                "kkt_runtime",
            )
        ) - exact_q_node_runtime_before
        exact_q_timing_profile["matching_runtime"] += max(
            exact_q_matching_elapsed - exact_q_node_elapsed,
            0.0,
        )
    matching_kkt_seconds = (
        time.perf_counter() - matching_started_at
        if performance_profile is not None
        else 0.0
    )
    if not matching.resource_feasible:
        _record_slot_performance_profile(
            performance_profile,
            configuration_seconds,
            matching_kkt_seconds,
        )
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
            configuration_summary=_configuration_summary(
                configuration,
                float(project_config.get_config_value(
                    project_cfg,
                    "algorithm.uav.position_tolerance_m",
                    1.0e-6,
                )),
            ),
            matching_summary=_matching_summary(matching),
            next_cache_state=None,
            next_uav_positions=None,
        )

    actual_q = float(matching.task_utility_sum)
    reward_cost_started_at = (
        time.perf_counter() if exact_q_timing_profile is not None else 0.0
    )
    cache_cost = float(configuration.cache_cost)
    movement_cost = float(configuration.movement_cost)
    if exact_q_timing_profile is not None:
        exact_q_timing_profile["reward_cost_aggregation_runtime"] += (
            time.perf_counter() - reward_cost_started_at
        )
    reward_final_started_at = (
        time.perf_counter() if exact_q_timing_profile is not None else 0.0
    )
    system_utility = actual_q - cache_cost - movement_cost
    if exact_q_timing_profile is not None:
        exact_q_timing_profile["reward_final_combination_runtime"] += (
            time.perf_counter() - reward_final_started_at
        )
    if not math.isclose(
        system_utility,
        actual_q - cache_cost - movement_cost,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError("G(t)=Q(t)-C_ca(t)-C_q(t) 恒等式失败。")

    _record_slot_performance_profile(
        performance_profile,
        configuration_seconds,
        matching_kkt_seconds,
    )
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
        predictive_configuration_utility=(
            None
            if prediction_evaluation_skipped
            else float(configuration.predictive_utility)
        ),
        predictive_configuration_objective=(
            None
            if prediction_evaluation_skipped
            else float(configuration.objective_value)
        ),
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
        configuration_summary=_configuration_summary(
            configuration,
            float(project_config.get_config_value(
                project_cfg,
                "algorithm.uav.position_tolerance_m",
                1.0e-6,
            )),
        ),
        matching_summary=_matching_summary(matching),
        next_cache_state=final_cache.copy(),
        next_uav_positions=final_uav,
    )


def run_slot20_validation(config_path: str = "configs/proposed/system_config.toml") -> dict[str, SlotExecutionResult]:
    cfg = project_config.load_config(config_path)
    project_config.validate_config(cfg)
    project_config.validate_for_formal_experiment(cfg)
    root = Path(cfg["_meta"]["project_root"])
    cache_cfg = cache_configuration.cache_config_from_project(cfg)
    slot = 20
    time_s = 120.0
    scenarios = cache_configuration.load_prediction_scenarios(
        cache_configuration.resolve_path(root, project_config.get_config_value(cfg, "paths.prediction_scenarios"))
    )
    origin = cache_configuration.scenario_at_origin(scenarios, slot)
    if int(origin["origin_slot"]) != slot:
        raise AssertionError(
            f"prediction origin_slot={origin['origin_slot']} 与 slot={slot} 不一致。"
        )
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
    actual_tasks = task_allocation.load_actual_tasks(root / "data/generated/tasks.csv", slot)
    roles = cache_configuration.load_json(root / "data/generated/vehicle_roles.json")
    sv_ids = {str(value) for value in roles["service_vehicles"]}
    tv_ids = {task.vehicle_id for task in actual_tasks}
    actual_positions = task_allocation.load_vehicle_positions(
        root / "data/generated/trajectory.csv",
        time_s,
        tv_ids | sv_ids,
    )
    local_cache = task_scenario.load_tv_local_cache(
        cache_configuration.resolve_path(root, project_config.get_config_value(cfg, "paths.tv_local_cache"))
    )
    boundary = uav_configuration.boundary_from_project(cfg)
    node_count = previous_cache.shape[0]
    first_cache_index = 0
    uav_ids = tuple(str(node_id) for node_id in previous_uav)
    zero_uav_triggers = {uav_id: 0 for uav_id in uav_ids}
    first_uav_triggers = dict(zero_uav_triggers)
    first_uav_triggers[uav_ids[0]] = 1
    first_cache_trigger = np.zeros(node_count, dtype=np.int8)
    first_cache_trigger[first_cache_index] = 1
    cases = {
        "A": (np.zeros(node_count, dtype=np.int8), zero_uav_triggers),
        "B": (first_cache_trigger, zero_uav_triggers),
        "C": (np.zeros(node_count, dtype=np.int8), first_uav_triggers),
        "D": (first_cache_trigger, first_uav_triggers),
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
        "requested_cache_trigger": result.input_cache_triggers,
        "actual_cache_update": result.configuration_summary.get(
            "actual_cache_update", []
        ),
        "requested_uav_trigger": result.input_uav_triggers,
        "actual_uav_movement": result.configuration_summary.get(
            "actual_uav_movement", {}
        ),
        # Legacy aliases retained for existing terminal consumers.
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
    parser.add_argument("--config", default="configs/proposed/system_config.toml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = run_slot20_validation(args.config)
    print(json.dumps({case: _summary(result) for case, result in results.items()}, ensure_ascii=False, indent=2))
    return 0

# ---- Joint cache/UAV configuration orchestration ----

@dataclass(frozen=True)
class JointConfigurationResult:
    previous_cache_state: np.ndarray
    final_cache_state: np.ndarray
    previous_uav_positions: dict[str, tuple[float, float]]
    final_uav_positions: dict[str, tuple[float, float]]
    cache_triggers: list[int]
    uav_triggers: dict[str, int]
    trigger_feasible: bool
    infeasible_reason: str
    predictive_utility: float
    cache_cost: float
    movement_energy: float
    movement_cost: float
    objective_value: float
    initial_objective: float
    objective_history: list[float]
    outer_rounds: int
    accepted_cache_updates: int
    accepted_uav_updates: int
    stop_reason: str
    cache_result_summary: dict[str, Any]
    uav_result_summary: dict[str, Any]


def _configuration_algorithm_config(project_cfg: dict[str, Any]) -> tuple[int, float]:
    get = project_config.get_config_value
    return (
        int(get(project_cfg, "algorithm.configuration.max_outer_rounds")),
        float(get(project_cfg, "algorithm.configuration.objective_tolerance")),
    )


def _position_arrays(positions: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        str(uav_id): np.asarray(xy, dtype=np.float64).copy()
        for uav_id, xy in positions.items()
    }


def _cache_cost(
    previous_cache_state: np.ndarray,
    cache_state: np.ndarray,
    cache_triggers: np.ndarray,
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
) -> tuple[float, np.ndarray]:
    """Cache cost 始终由 C_prev -> C_candidate 一次性计算。"""

    return cache_configuration.cache_update_cost(
        previous_cache_state,
        cache_state,
        cache_triggers,
        service_catalog["service_sizes_mb"],
        cache_nodes,
        cache_config,
    )


def _movement_cost(
    previous_uav_positions: dict[str, np.ndarray],
    uav_positions: dict[str, np.ndarray],
    uav_triggers: dict[str, int],
    scenario_slice: dict[str, Any],
    cache_state: np.ndarray,
    local_cache_state: dict[str, set[int]],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    evaluation_context: cache_configuration.SurrogateEvaluationContext | None = None,
) -> dict[str, Any]:
    """Movement cost 始终由 Q_prev -> Q_candidate 一次性计算。"""

    search_cfg = uav_configuration.uav_search_config_from_project(project_cfg)
    return uav_configuration.evaluate_uav_configuration(
        uav_positions,
        previous_uav_positions,
        uav_triggers,
        scenario_slice,
        cache_state,
        local_cache_state,
        cache_nodes,
        cache_config,
        search_cfg,
        cache_cost_constant=0.0,
        evaluation_context=evaluation_context,
    )


def build_policy_target_configuration(
    previous_cache_state: np.ndarray,
    previous_uav_positions: dict[str, Any],
    final_cache_state: np.ndarray,
    final_uav_positions: dict[str, Any],
    cache_triggers: np.ndarray,
    uav_triggers: dict[str, int],
    service_catalog: dict[str, Any],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    flight_boundary: tuple[float, float, float, float],
    cache_actions: tuple[dict[str, Any], ...] = (),
    uav_actions: tuple[dict[str, Any], ...] = (),
) -> JointConfigurationResult:
    """Build a direct-policy target configuration without a prediction solve."""

    previous_cache = np.asarray(previous_cache_state, dtype=np.int8).copy()
    final_cache = np.asarray(final_cache_state, dtype=np.int8).copy()
    previous_uav = _position_arrays(previous_uav_positions)
    final_uav = _position_arrays(final_uav_positions)
    cache_trigger = np.asarray(cache_triggers, dtype=np.int8).copy()
    cache_configuration.validate_cache_shape_and_capacity(
        previous_cache, service_catalog, cache_nodes
    )
    cache_configuration.validate_cache_shape_and_capacity(
        final_cache, service_catalog, cache_nodes
    )
    if cache_trigger.shape != (previous_cache.shape[0],) or not np.all(
        np.isin(cache_trigger, [0, 1])
    ):
        raise ValueError("policy target cache triggers 非法。")
    actual_cache = np.any(final_cache != previous_cache, axis=1).astype(np.int8)
    if np.any(actual_cache > cache_trigger):
        raise ValueError("policy target cache state 违反 trigger=0 保持不变语义。")
    uav_configuration.validate_uav_configuration_transition(
        previous_uav,
        final_uav,
        uav_triggers,
        project_cfg,
        flight_boundary,
    )
    cache_cost, _ = _cache_cost(
        previous_cache,
        final_cache,
        cache_trigger,
        service_catalog,
        cache_nodes,
        cache_config,
    )
    search = uav_configuration.uav_search_config_from_project(project_cfg)
    movement_energy, movement_cost, _, _, _ = uav_configuration.movement_summary(
        previous_uav,
        final_uav,
        {key: int(value) for key, value in uav_triggers.items()},
        search,
    )
    actual_uav = sum(
        int(
            np.linalg.norm(final_uav[uav_id] - previous_uav[uav_id])
            > search.position_tolerance_m
        )
        for uav_id in previous_uav
    )
    return JointConfigurationResult(
        previous_cache_state=previous_cache,
        final_cache_state=final_cache,
        previous_uav_positions=_position_tuples(previous_uav),
        final_uav_positions=_position_tuples(final_uav),
        cache_triggers=cache_trigger.astype(int).tolist(),
        uav_triggers={key: int(value) for key, value in uav_triggers.items()},
        trigger_feasible=True,
        infeasible_reason="",
        predictive_utility=0.0,
        cache_cost=float(cache_cost),
        movement_energy=float(movement_energy),
        movement_cost=float(movement_cost),
        objective_value=0.0,
        initial_objective=0.0,
        objective_history=[],
        outer_rounds=0,
        accepted_cache_updates=int(actual_cache.sum()),
        accepted_uav_updates=int(actual_uav),
        stop_reason="policy_target_prediction_skipped",
        cache_result_summary={"target_actions": [dict(value) for value in cache_actions]},
        uav_result_summary={"target_actions": [dict(value) for value in uav_actions]},
    )


def _validated_configuration_override(
    configuration: JointConfigurationResult,
    previous_cache: np.ndarray,
    previous_uav: dict[str, tuple[float, float]],
    cache_trigger: np.ndarray,
    uav_triggers: dict[str, int],
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    project_cfg: dict[str, Any],
    flight_boundary: tuple[float, float, float, float],
) -> JointConfigurationResult:
    """Reject any override that is not an authoritative direct-policy transition."""

    if not isinstance(configuration, JointConfigurationResult):
        raise TypeError("configuration_override 必须是 JointConfigurationResult。")
    if configuration.stop_reason != "policy_target_prediction_skipped":
        raise ValueError("configuration_override 仅接受 policy target result。")
    if not np.array_equal(configuration.previous_cache_state, previous_cache):
        raise ValueError("configuration_override previous cache 不匹配。")
    if configuration.cache_triggers != cache_trigger.astype(int).tolist():
        raise ValueError("configuration_override cache triggers 不匹配。")
    if configuration.uav_triggers != uav_triggers:
        raise ValueError("configuration_override UAV triggers 不匹配。")
    for uav_id, xy in previous_uav.items():
        if not np.array_equal(
            np.asarray(configuration.previous_uav_positions[uav_id], dtype=np.float64),
            np.asarray(xy, dtype=np.float64),
        ):
            raise ValueError("configuration_override previous UAV state 不匹配。")
    rebuilt = build_policy_target_configuration(
        previous_cache,
        previous_uav,
        configuration.final_cache_state,
        configuration.final_uav_positions,
        cache_trigger,
        uav_triggers,
        service_catalog,
        project_cfg,
        cache_nodes,
        cache_config,
        flight_boundary,
        cache_actions=tuple(configuration.cache_result_summary.get("target_actions", [])),
        uav_actions=tuple(configuration.uav_result_summary.get("target_actions", [])),
    )
    if not math.isclose(rebuilt.cache_cost, configuration.cache_cost, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("configuration_override cache cost 不匹配。")
    if not math.isclose(
        rebuilt.movement_cost, configuration.movement_cost, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("configuration_override movement cost 不匹配。")
    return rebuilt


def evaluate_joint_configuration(
    cache_state: np.ndarray,
    uav_positions: dict[str, Any],
    previous_cache_state: np.ndarray,
    previous_uav_positions: dict[str, Any],
    cache_triggers: np.ndarray,
    uav_triggers: dict[str, int],
    scenario_slice: dict[str, Any],
    local_cache_state: dict[str, set[int]],
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    project_cfg: dict[str, Any],
    evaluation_context: cache_configuration.SurrogateEvaluationContext | None = None,
) -> dict[str, Any]:
    """唯一 joint evaluator：J = Q_hat - C_ca - C_q。

    Q_hat 通过 cache_configuration.evaluate_fixed_configuration() 计算，以保证
    Local/SV/RSU/UAV/Cloud surrogate 口径一致。C_ca 和 C_q 都只扣一次，并且
    始终相对上一时隙真实 reference state。
    """

    cache_state = np.asarray(cache_state, dtype=np.int8)
    previous_cache_state = np.asarray(previous_cache_state, dtype=np.int8)
    uav_pos = _position_arrays(uav_positions)
    uav_prev = _position_arrays(previous_uav_positions)
    nodes = uav_configuration.cache_nodes_with_uav_positions(cache_nodes, uav_pos)
    if evaluation_context is None:
        evaluation_context = cache_configuration.build_surrogate_evaluation_context(
            scenario_slice,
            local_cache_state,
            cache_nodes,
            cache_config,
        )
    predictive_utility = cache_configuration.evaluate_fixed_configuration(
        cache_state,
        scenario_slice,
        local_cache_state,
        nodes,
        cache_config,
        evaluation_context=evaluation_context,
        profile_source="joint",
    )["predicted_utility"]
    cache_cost, cache_energy = _cache_cost(
        previous_cache_state,
        cache_state,
        cache_triggers,
        service_catalog,
        cache_nodes,
        cache_config,
    )
    movement_eval = _movement_cost(
        uav_prev,
        uav_pos,
        uav_triggers,
        scenario_slice,
        cache_state,
        local_cache_state,
        project_cfg,
        cache_nodes,
        cache_config,
        evaluation_context=evaluation_context,
    )
    movement_cost = float(movement_eval["movement_cost"])
    objective = float(predictive_utility - cache_cost - movement_cost)
    if not math.isfinite(objective):
        raise AssertionError("joint objective 非 finite。")
    return {
        "predictive_utility": float(predictive_utility),
        "cache_cost": float(cache_cost),
        "cache_update_energy": cache_energy,
        "movement_energy": float(movement_eval["movement_energy"]),
        "movement_cost": movement_cost,
        "objective_value": objective,
        "per_uav_displacement_m": movement_eval["per_uav_displacement_m"],
        "per_uav_movement_cost": movement_eval["per_uav_movement_cost"],
    }


def _cache_summary(result: dict[str, Any]) -> dict[str, Any]:
    stats = result.get("statistics", {})
    return {
        "trigger_feasible": bool(result.get("trigger_feasible", False)),
        "requested_cache_trigger": list(
            result.get("requested_cache_trigger", [])
        ),
        "actual_cache_update": np.asarray(
            result.get("actual_cache_trigger", []),
            dtype=np.int8,
        ).astype(int).tolist(),
        "cache_update_cost": float(result.get("cache_update_cost", 0.0)),
        "predicted_utility": float(result.get("predicted_utility", 0.0)),
        "objective": float(result.get("objective", 0.0)),
        "coordinate_rounds": int(stats.get("num_coordinate_rounds", 0)),
        "accepted_moves": int(stats.get("num_accepted_moves", 0)),
        "converged_by_no_improvement": bool(stats.get("converged_by_no_improvement", False)),
        "hit_max_rounds": bool(stats.get("hit_max_rounds", False)),
        "candidate_diagnostics": stats.get("candidate_diagnostics", {}),
    }


def _uav_summary(result: uav_configuration.UavOptimizationResult) -> dict[str, Any]:
    return {
        "trigger_feasible": result.trigger_feasible,
        "requested_uav_trigger": {
            str(item["uav_id"]): int(item["requested_trigger"])
            for item in result.candidate_diagnostics.values()
        },
        "actual_uav_triggers": result.actual_uav_triggers,
        "per_uav_displacement_m": result.per_uav_displacement_m,
        "movement_energy": result.movement_energy,
        "movement_cost": result.movement_cost,
        "objective": result.objective_value,
        "search_rounds": result.search_rounds,
        "accepted_improvements": result.accepted_improvements,
        "stop_reason": result.stop_reason,
        "candidate_diagnostics": result.candidate_diagnostics,
    }


def _infeasible_result(
    previous_cache_state: np.ndarray,
    current_cache: np.ndarray,
    previous_uav_positions: dict[str, np.ndarray],
    current_uav_positions: dict[str, np.ndarray],
    cache_triggers: np.ndarray,
    uav_triggers: dict[str, int],
    reason: str,
    cache_summary: dict[str, Any] | None = None,
    uav_summary: dict[str, Any] | None = None,
) -> JointConfigurationResult:
    return JointConfigurationResult(
        previous_cache_state=previous_cache_state.copy(),
        final_cache_state=current_cache.copy(),
        previous_uav_positions=_position_tuples(previous_uav_positions),
        final_uav_positions=_position_tuples(current_uav_positions),
        cache_triggers=cache_triggers.astype(int).tolist(),
        uav_triggers=dict(uav_triggers),
        trigger_feasible=False,
        infeasible_reason=reason,
        predictive_utility=float("-inf"),
        cache_cost=float("inf"),
        movement_energy=float("inf"),
        movement_cost=float("inf"),
        objective_value=float("-inf"),
        initial_objective=float("-inf"),
        objective_history=[],
        outer_rounds=0,
        accepted_cache_updates=0,
        accepted_uav_updates=0,
        stop_reason="infeasible_trigger",
        cache_result_summary=cache_summary or {},
        uav_result_summary=uav_summary or {},
    )


def optimize_joint_configuration(
    previous_cache_state: np.ndarray,
    previous_uav_positions: dict[str, Any],
    cache_triggers: np.ndarray,
    uav_triggers: dict[str, int],
    scenario_slice: dict[str, Any],
    local_cache_state: dict[str, set[int]],
    service_catalog: dict[str, Any],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    flight_boundary: tuple[float, float, float, float],
    evaluation_context: cache_configuration.SurrogateEvaluationContext | None = None,
) -> JointConfigurationResult:
    """外层 Cache ↔ UAV block-coordinate configuration optimizer."""

    max_outer_rounds, outer_tol = _configuration_algorithm_config(project_cfg)
    previous_cache = np.asarray(previous_cache_state, dtype=np.int8).copy()
    cache_trigger = np.asarray(cache_triggers, dtype=np.int8).copy()
    q_prev = _position_arrays(previous_uav_positions)
    q_current = {uav_id: xy.copy() for uav_id, xy in q_prev.items()}
    c_current = previous_cache.copy()
    if evaluation_context is None:
        evaluation_context = cache_configuration.build_surrogate_evaluation_context(
            scenario_slice,
            local_cache_state,
            cache_nodes,
            cache_config,
        )

    # Case A 快路径：没有任何 trigger 时，不做无意义搜索。
    if int(cache_trigger.sum()) == 0 and not any(int(value) == 1 for value in uav_triggers.values()):
        evaluation = evaluate_joint_configuration(
            c_current,
            q_current,
            previous_cache,
            q_prev,
            cache_trigger,
            uav_triggers,
            scenario_slice,
            local_cache_state,
            service_catalog,
            cache_nodes,
            cache_config,
            project_cfg,
            evaluation_context=evaluation_context,
        )
        return JointConfigurationResult(
            previous_cache_state=previous_cache.copy(),
            final_cache_state=c_current.copy(),
            previous_uav_positions=_position_tuples(q_prev),
            final_uav_positions=_position_tuples(q_current),
            cache_triggers=cache_trigger.astype(int).tolist(),
            uav_triggers=dict(uav_triggers),
            trigger_feasible=True,
            infeasible_reason="",
            predictive_utility=evaluation["predictive_utility"],
            cache_cost=evaluation["cache_cost"],
            movement_energy=evaluation["movement_energy"],
            movement_cost=evaluation["movement_cost"],
            objective_value=evaluation["objective_value"],
            initial_objective=evaluation["objective_value"],
            objective_history=[evaluation["objective_value"]],
            outer_rounds=0,
            accepted_cache_updates=0,
            accepted_uav_updates=0,
            stop_reason="no_trigger_no_search",
            cache_result_summary={},
            uav_result_summary={},
        )

    # Trigger-conditioned initialization: Cache block first, fixed Q=Q_prev.
    cache_nodes_at_q_prev = uav_configuration.cache_nodes_with_uav_positions(cache_nodes, q_prev)
    cache_init = cache_configuration.optimize_cache(
        previous_cache,
        cache_trigger,
        scenario_slice,
        service_catalog,
        local_cache_state,
        cache_nodes_at_q_prev,
        cache_config,
        evaluation_context=evaluation_context,
    )
    if not bool(cache_init["trigger_feasible"]):
        return _infeasible_result(
            previous_cache,
            c_current,
            q_prev,
            q_current,
            cache_trigger,
            uav_triggers,
            "cache trigger infeasible",
            cache_summary=_cache_summary(cache_init),
        )
    c_current = np.asarray(cache_init["new_cache"], dtype=np.int8).copy()

    # Trigger-conditioned initialization: UAV block second, fixed C=C_init.
    current_cache_cost, _ = _cache_cost(
        previous_cache,
        c_current,
        cache_trigger,
        service_catalog,
        cache_nodes,
        cache_config,
    )
    uav_init = uav_configuration.optimize_uav_configuration(
        q_prev,
        uav_triggers,
        scenario_slice,
        c_current,
        local_cache_state,
        project_cfg,
        cache_nodes,
        cache_config,
        flight_boundary,
        cache_cost_constant=current_cache_cost,
        evaluation_context=evaluation_context,
    )
    if not uav_init.trigger_feasible:
        return _infeasible_result(
            previous_cache,
            c_current,
            q_prev,
            q_current,
            cache_trigger,
            uav_triggers,
            f"uav trigger infeasible: {uav_init.reason}",
            cache_summary=_cache_summary(cache_init),
            uav_summary=_uav_summary(uav_init),
        )
    q_current = _position_arrays(uav_init.final_positions)

    evaluation = evaluate_joint_configuration(
        c_current,
        q_current,
        previous_cache,
        q_prev,
        cache_trigger,
        uav_triggers,
        scenario_slice,
        local_cache_state,
        service_catalog,
        cache_nodes,
        cache_config,
        project_cfg,
        evaluation_context=evaluation_context,
    )
    current_objective = float(evaluation["objective_value"])
    initial_objective = current_objective
    history = [current_objective]
    accepted_cache = 0
    accepted_uav = 0
    completed_rounds = 0
    stop_reason = "stopped_by_max_rounds"
    last_cache_summary = _cache_summary(cache_init)
    last_uav_summary = _uav_summary(uav_init)

    for round_index in range(1, max_outer_rounds + 1):
        completed_rounds = round_index
        round_improved = False

        # Cache refinement: search starts at C_current, but costs and trigger semantics reference C_prev.
        cache_nodes_at_current_q = uav_configuration.cache_nodes_with_uav_positions(
            cache_nodes,
            q_current,
        )
        cache_candidate = cache_configuration.optimize_cache(
            previous_cache,
            cache_trigger,
            scenario_slice,
            service_catalog,
            local_cache_state,
            cache_nodes_at_current_q,
            cache_config,
            initial_cache_state=c_current,
            evaluation_context=evaluation_context,
        )
        last_cache_summary = _cache_summary(cache_candidate)
        if not bool(cache_candidate["trigger_feasible"]):
            return _infeasible_result(
                previous_cache,
                c_current,
                q_prev,
                q_current,
                cache_trigger,
                uav_triggers,
                "cache refinement returned infeasible",
                cache_summary=last_cache_summary,
                uav_summary=last_uav_summary,
            )
        c_candidate = np.asarray(cache_candidate["new_cache"], dtype=np.int8)
        candidate_eval = evaluate_joint_configuration(
            c_candidate,
            q_current,
            previous_cache,
            q_prev,
            cache_trigger,
            uav_triggers,
            scenario_slice,
            local_cache_state,
            service_catalog,
            cache_nodes,
            cache_config,
            project_cfg,
            evaluation_context=evaluation_context,
        )
        if candidate_eval["objective_value"] > current_objective + outer_tol:
            c_current = c_candidate.copy()
            current_objective = float(candidate_eval["objective_value"])
            evaluation = candidate_eval
            history.append(current_objective)
            accepted_cache += 1
            round_improved = True

        # UAV refinement: search starts at Q_current, but movement/cost/constraint reference Q_prev.
        current_cache_cost, _ = _cache_cost(
            previous_cache,
            c_current,
            cache_trigger,
            service_catalog,
            cache_nodes,
            cache_config,
        )
        uav_candidate = uav_configuration.optimize_uav_configuration(
            q_prev,
            uav_triggers,
            scenario_slice,
            c_current,
            local_cache_state,
            project_cfg,
            cache_nodes,
            cache_config,
            flight_boundary,
            cache_cost_constant=current_cache_cost,
            initial_candidate_positions=q_current,
            evaluation_context=evaluation_context,
        )
        last_uav_summary = _uav_summary(uav_candidate)
        if not uav_candidate.trigger_feasible:
            return _infeasible_result(
                previous_cache,
                c_current,
                q_prev,
                q_current,
                cache_trigger,
                uav_triggers,
                f"uav refinement returned infeasible: {uav_candidate.reason}",
                cache_summary=last_cache_summary,
                uav_summary=last_uav_summary,
            )
        q_candidate = _position_arrays(uav_candidate.final_positions)
        candidate_eval = evaluate_joint_configuration(
            c_current,
            q_candidate,
            previous_cache,
            q_prev,
            cache_trigger,
            uav_triggers,
            scenario_slice,
            local_cache_state,
            service_catalog,
            cache_nodes,
            cache_config,
            project_cfg,
            evaluation_context=evaluation_context,
        )
        if candidate_eval["objective_value"] > current_objective + outer_tol:
            q_current = q_candidate
            current_objective = float(candidate_eval["objective_value"])
            evaluation = candidate_eval
            history.append(current_objective)
            accepted_uav += 1
            round_improved = True

        if not round_improved:
            stop_reason = "converged_by_no_improvement"
            break

    return JointConfigurationResult(
        previous_cache_state=previous_cache.copy(),
        final_cache_state=c_current.copy(),
        previous_uav_positions=_position_tuples(q_prev),
        final_uav_positions=_position_tuples(q_current),
        cache_triggers=cache_trigger.astype(int).tolist(),
        uav_triggers=dict(uav_triggers),
        trigger_feasible=True,
        infeasible_reason="",
        predictive_utility=float(evaluation["predictive_utility"]),
        cache_cost=float(evaluation["cache_cost"]),
        movement_energy=float(evaluation["movement_energy"]),
        movement_cost=float(evaluation["movement_cost"]),
        objective_value=float(evaluation["objective_value"]),
        initial_objective=initial_objective,
        objective_history=history,
        outer_rounds=completed_rounds,
        accepted_cache_updates=accepted_cache,
        accepted_uav_updates=accepted_uav,
        stop_reason=stop_reason,
        cache_result_summary=last_cache_summary,
        uav_result_summary=last_uav_summary,
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
    tol: float = 1.0e-6,
) -> list[str]:
    if final is None:
        return []
    changed = []
    for uav_id, xy in previous.items():
        if float(np.linalg.norm(np.asarray(final[uav_id]) - np.asarray(xy))) > tol:
            changed.append(uav_id)
    return changed


def validate_joint_optimizer_once(
    origin_scenario: dict[str, Any],
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    local_cache_state: dict[str, set[int]],
    project_cfg: dict[str, Any],
    flight_boundary: tuple[float, float, float, float],
) -> dict[str, JointConfigurationResult]:
    previous_cache = cache_configuration.deterministic_initial_cache(
        origin_scenario,
        service_catalog,
        cache_nodes,
        cache_config,
    )
    previous_uav_positions = uav_configuration.initial_uav_positions_from_project(project_cfg)
    uav_ids = uav_configuration.uav_ids_from_project(project_cfg)
    num_nodes = previous_cache.shape[0]
    node_roles = cache_nodes["node_roles"]
    first_sv = node_roles.index("SV")
    first_rsu_or_uav = next(
        index for index, role in enumerate(node_roles) if role in {"RSU", "UAV"}
    )
    cases: dict[str, tuple[np.ndarray, dict[str, int]]] = {}
    cases["A"] = (np.zeros(num_nodes, dtype=np.int8), {uav_id: 0 for uav_id in uav_ids})
    z_b = np.zeros(num_nodes, dtype=np.int8)
    z_b[first_sv] = 1
    cases["B"] = (z_b, {uav_id: 0 for uav_id in uav_ids})
    cases["C"] = (np.zeros(num_nodes, dtype=np.int8), {uav_id: int(i == 0) for i, uav_id in enumerate(uav_ids)})
    z_d = np.zeros(num_nodes, dtype=np.int8)
    z_d[first_sv] = 1
    cases["D"] = (z_d, {uav_id: int(i == 0) for i, uav_id in enumerate(uav_ids)})
    z_e = np.zeros(num_nodes, dtype=np.int8)
    z_e[first_rsu_or_uav] = 1
    cases["E"] = (z_e, {uav_id: 1 for uav_id in uav_ids})

    results: dict[str, JointConfigurationResult] = {}
    for label, (cache_trigger, uav_trigger) in cases.items():
        result = optimize_joint_configuration(
            previous_cache,
            previous_uav_positions,
            cache_trigger,
            uav_trigger,
            origin_scenario,
            local_cache_state,
            service_catalog,
            project_cfg,
            cache_nodes,
            cache_config,
            flight_boundary,
        )
        results[label] = result
        if not result.trigger_feasible:
            raise AssertionError(f"Case {label} unexpectedly infeasible: {result.infeasible_reason}")
        if any(
            result.objective_history[i + 1] <= result.objective_history[i]
            + project_config.get_config_value(project_cfg, "algorithm.configuration.objective_tolerance")
            for i in range(len(result.objective_history) - 1)
        ):
            raise AssertionError(f"Case {label} accepted objective history 非严格提升。")
        direct = evaluate_joint_configuration(
            result.final_cache_state,
            result.final_uav_positions,
            previous_cache,
            previous_uav_positions,
            cache_trigger,
            uav_trigger,
            origin_scenario,
            local_cache_state,
            service_catalog,
            cache_nodes,
            cache_config,
            project_cfg,
        )
        if not math.isclose(
            direct["objective_value"],
            result.objective_value,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise AssertionError(f"Case {label} final objective 直接复算不一致。")
        changed_nodes = _changed_cache_nodes(previous_cache, result.final_cache_state)
        changed_uav = _changed_uavs(result.previous_uav_positions, result.final_uav_positions)
        for node_index, trigger in enumerate(cache_trigger.tolist()):
            if trigger == 0 and node_index in changed_nodes:
                raise AssertionError(f"Case {label} z_ca=0 cache node changed。")
        for uav_id, trigger in uav_trigger.items():
            if trigger == 0 and uav_id in changed_uav:
                raise AssertionError(f"Case {label} z_q=0 UAV moved。")
    repeat = optimize_joint_configuration(
        previous_cache,
        previous_uav_positions,
        cases["D"][0],
        cases["D"][1],
        origin_scenario,
        local_cache_state,
        service_catalog,
        project_cfg,
        cache_nodes,
        cache_config,
        flight_boundary,
    )
    if not np.array_equal(results["D"].final_cache_state, repeat.final_cache_state):
        raise AssertionError("Case D repeated-run cache 不确定。")
    if results["D"].final_uav_positions != repeat.final_uav_positions:
        raise AssertionError("Case D repeated-run UAV 不确定。")
    return results


def _result_summary(result: JointConfigurationResult) -> dict[str, Any]:
    return {
        "trigger_feasible": result.trigger_feasible,
        "requested_cache_trigger": result.cache_triggers,
        "actual_cache_update": np.any(
            result.final_cache_state != result.previous_cache_state,
            axis=1,
        ).astype(int).tolist(),
        "requested_uav_trigger": result.uav_triggers,
        "actual_uav_movement": {
            uav_id: int(
                float(
                    np.linalg.norm(
                        np.asarray(result.final_uav_positions[uav_id], dtype=np.float64)
                        - np.asarray(result.previous_uav_positions[uav_id], dtype=np.float64)
                    )
                )
                > 1.0e-6
            )
            for uav_id in result.previous_uav_positions
        },
        # Legacy aliases retained for existing validation consumers.
        "cache_trigger": result.cache_triggers,
        "uav_trigger": result.uav_triggers,
        "cache_candidate_diagnostics": result.cache_result_summary.get(
            "candidate_diagnostics", {}
        ),
        "uav_candidate_diagnostics": result.uav_result_summary.get(
            "candidate_diagnostics", {}
        ),
        "changed_cache_nodes": _changed_cache_nodes(
            result.previous_cache_state,
            result.final_cache_state,
        ),
        "changed_uavs": _changed_uavs(result.previous_uav_positions, result.final_uav_positions),
        "predictive_utility": result.predictive_utility,
        "cache_cost": result.cache_cost,
        "movement_energy": result.movement_energy,
        "movement_cost": result.movement_cost,
        "objective": result.objective_value,
        "initial_objective": result.initial_objective,
        "objective_history": result.objective_history,
        "outer_rounds": result.outer_rounds,
        "accepted_cache_updates": result.accepted_cache_updates,
        "accepted_uav_updates": result.accepted_uav_updates,
        "stop_reason": result.stop_reason,
        "final_uav_positions": result.final_uav_positions,
    }


def build_configuration_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache + UAV trigger-conditioned configuration optimizer.")
    parser.add_argument("--config", default="configs/proposed/system_config.toml")
    parser.add_argument("--origin-index", type=int, default=0)
    return parser


def configuration_main(argv: list[str] | None = None) -> int:
    args = build_configuration_parser().parse_args(argv)
    cfg = project_config.load_config(args.config)
    project_config.validate_config(cfg)
    project_config.validate_for_formal_experiment(cfg)
    project_root = Path(cfg["_meta"]["project_root"])
    cache_cfg = cache_configuration.cache_config_from_project(cfg)
    scenarios = cache_configuration.load_prediction_scenarios(
        cache_configuration.resolve_path(
            project_root,
            project_config.get_config_value(cfg, "paths.prediction_scenarios"),
        )
    )
    origin = cache_configuration.scenario_at_origin(scenarios, args.origin_index)
    service_catalog = cache_configuration.load_service_catalog(
        cache_configuration.resolve_path(
            project_root,
            project_config.get_config_value(cfg, "paths.service_catalog"),
        )
    )
    cache_nodes = cache_configuration.build_cache_nodes(origin, cache_cfg)
    local_cache_state = task_scenario.load_tv_local_cache(
        cache_configuration.resolve_path(
            project_root,
            project_config.get_config_value(cfg, "paths.tv_local_cache"),
        )
    )
    boundary = uav_configuration.boundary_from_project(cfg)
    results = validate_joint_optimizer_once(
        origin,
        service_catalog,
        cache_nodes,
        cache_cfg,
        local_cache_state,
        cfg,
        boundary,
    )
    print("Joint configuration minimal validation PASS")
    print(f"origin_index={args.origin_index}, origin_slot={origin['origin_slot']}, origin_time_s={origin['origin_time_s']}")
    for label, result in results.items():
        print(f"Case {label}: {json.dumps(_result_summary(result), ensure_ascii=False)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
