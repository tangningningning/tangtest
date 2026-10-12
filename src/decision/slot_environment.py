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

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from common import project_config
from decision import cache_configuration, task_allocation, uav_configuration


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


def _matching_summary(
    result: task_allocation.TransferExchangeMatchingResult,
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
) -> dict[str, Any]:
    node_ids = [str(value) for value in cache_nodes["node_ids"]]
    node_roles = {
        node_id: str(role)
        for node_id, role in zip(node_ids, cache_nodes["node_roles"], strict=True)
    }
    counts_by_node = {node_id: 0 for node_id in node_ids}
    local_count = 0
    cloud_count = 0
    for target in result.matching.values():
        if target == task_allocation.CLOUD:
            cloud_count += 1
        elif target == task_allocation.LOCAL or target.startswith(
            task_allocation.LOCAL_PREFIX
        ):
            local_count += 1
        elif target in counts_by_node:
            counts_by_node[target] += 1
        else:
            raise ValueError(f"matching summary 遇到未知执行节点：{target}")

    counts_by_role = {
        role: {
            node_id: int(counts_by_node[node_id])
            for node_id in node_ids
            if node_roles[node_id] == role
        }
        for role in ("SV", "RSU", "UAV")
    }
    uav_ids = list(counts_by_role["UAV"])
    edge_cpu_by_node: dict[str, float] = {}
    for node_id, fraction_by_task in result.cpu_allocations.items():
        role = node_roles.get(node_id)
        if role is None:
            continue
        edge_cpu_by_node[node_id] = float(
            math.fsum(fraction_by_task.values())
            * task_allocation._node_cpu(role, cache_config)
        )
    edge_cpu_total = float(math.fsum(edge_cpu_by_node.values()))
    task_count = len(result.matching)
    mean_delay = (
        float(math.fsum(result.per_task_delay_s.values()) / task_count)
        if task_count
        else None
    )
    mean_energy = (
        float(math.fsum(result.per_task_vehicle_energy_j.values()) / task_count)
        if task_count
        else None
    )
    mean_utility = (
        float(math.fsum(result.per_task_utility.values()) / task_count)
        if task_count
        else None
    )
    summary = {
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
        "tie_break_count": result.tie_break_count,
        "cloud_resource_mode": cache_config.cloud_resource_mode,
        "cloud_fallback_mode": cache_config.cloud_fallback_mode,
        "node_task_count": {
            "Local": int(local_count),
            "SV": int(sum(counts_by_role["SV"].values())),
            "SV_by_node": counts_by_role["SV"],
            "RSU": int(sum(counts_by_role["RSU"].values())),
            "RSU_by_node": counts_by_role["RSU"],
            "UAV": int(sum(counts_by_role["UAV"].values())),
            "UAV0": int(counts_by_node[uav_ids[0]]) if len(uav_ids) > 0 else 0,
            "UAV1": int(counts_by_node[uav_ids[1]]) if len(uav_ids) > 1 else 0,
            "UAV_by_node": counts_by_role["UAV"],
            "Cloud": int(cloud_count),
        },
        "resource": {
            "cloud_cpu_usage_cycles_per_s": float(
                result.cloud_cpu_used_cycles_per_s
            ),
            "cloud_cpu_capacity_cycles_per_s": (
                None
                if result.cloud_cpu_capacity_cycles_per_s is None
                else float(result.cloud_cpu_capacity_cycles_per_s)
            ),
            "cloud_cpu_remaining_cycles_per_s": (
                None
                if result.cloud_cpu_capacity_cycles_per_s is None
                else max(
                    float(result.cloud_cpu_capacity_cycles_per_s)
                    - float(result.cloud_cpu_used_cycles_per_s),
                    0.0,
                )
            ),
            "cloud_cpu_allocations_cycles_per_s": dict(
                result.cloud_cpu_allocations_cycles_per_s
            ),
            "edge_cpu_usage_cycles_per_s": edge_cpu_total,
            "edge_cpu_usage_by_node_cycles_per_s": edge_cpu_by_node,
        },
        "performance": {
            "mean_delay_s": mean_delay,
            "mean_energy_j": mean_energy,
            "task_utility_sum": float(result.task_utility_sum),
            "mean_task_utility": mean_utility,
        },
    }
    return summary


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
        raise ValueError(
            "configuration_override is required when any configuration trigger is active."
        )
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
        cloud_fallback_mode=cache_config.cloud_fallback_mode,
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
            matching_summary=_matching_summary(
                matching,
                cache_nodes,
                cache_config,
            ),
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
        matching_summary=_matching_summary(
            matching,
            cache_nodes,
            cache_config,
        ),
        next_cache_state=final_cache.copy(),
        next_uav_positions=final_uav,
    )









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


def execute_configured_slot(
    previous_cache_state: np.ndarray,
    previous_uav_positions: dict[str, Any],
    final_cache_state: np.ndarray,
    final_uav_positions: dict[str, Any],
    scenario_slice: dict[str, Any],
    actual_tasks: list[task_allocation.Task],
    actual_vehicle_positions: dict[str, np.ndarray],
    local_cache_state: dict[str, set[int]],
    service_catalog: dict[str, Any],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    flight_boundary: tuple[float, float, float, float],
    *,
    slot_id: int | None = None,
    time_s: float | None = None,
    performance_profile: dict[str, Any] | None = None,
    matching_kkt_cache: task_allocation.KKTResourceAllocationCache | None = None,
    node_evaluation_cache: task_allocation.NodeEvaluationCache | None = None,
) -> SlotExecutionResult:
    """执行控制器已确定的最终 Cache/UAV 配置及真实 Matching/KKT。"""

    previous_cache = np.asarray(previous_cache_state, dtype=np.int8).copy()
    final_cache = np.asarray(final_cache_state, dtype=np.int8).copy()
    previous_uav = _position_tuples(previous_uav_positions)
    final_uav = _position_tuples(final_uav_positions)
    if previous_cache.shape != final_cache.shape:
        raise ValueError("最终 Cache 配置 shape 必须与当前 Cache 一致。")
    cache_updates = np.any(final_cache != previous_cache, axis=1).astype(np.int8)
    tolerance = float(
        project_config.get_config_value(
            project_cfg, "algorithm.uav.position_tolerance_m", 1.0e-6
        )
    )
    uav_moves = {
        uav_id: int(
            np.linalg.norm(
                np.asarray(final_uav[uav_id], dtype=np.float64)
                - np.asarray(previous_uav[uav_id], dtype=np.float64)
            )
            > tolerance
        )
        for uav_id in previous_uav
    }
    configuration = build_policy_target_configuration(
        previous_cache,
        previous_uav,
        final_cache,
        final_uav,
        cache_updates,
        uav_moves,
        service_catalog,
        project_cfg,
        cache_nodes,
        cache_config,
        flight_boundary,
    )
    if not bool(np.any(cache_updates)) and configuration.cache_cost != 0.0:
        raise AssertionError("Cache 配置未变化时不得产生 reconfiguration cost。")
    return execute_slot(
        previous_cache,
        previous_uav,
        cache_updates,
        uav_moves,
        scenario_slice,
        actual_tasks,
        actual_vehicle_positions,
        local_cache_state,
        service_catalog,
        project_cfg,
        cache_nodes,
        cache_config,
        flight_boundary,
        slot_id=slot_id,
        time_s=time_s,
        performance_profile=performance_profile,
        configuration_override=configuration,
        matching_kkt_cache=matching_kkt_cache,
        node_evaluation_cache=node_evaluation_cache,
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
