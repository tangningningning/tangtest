#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-slot configuration plus cross-slot task execution.

本模块把一个时隙内的配置决策、Matching/KKT 和生命周期物理执行串起来：

previous configuration state + external trigger Z(t) + prediction scenario slice
-> trigger-conditioned configuration
-> actual-state Transfer-Exchange Matching + KKT
-> per-slot realized task work
-> next configuration state and task lifecycle events
-> next configuration state

本模块不实现策略、不定义 observation/action distribution、不管理 Episode 生命周期。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from common import project_config
from common import system_model
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
    cache_download_delay_s: float | None
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
    beta_ca: float | None = None
    beta_q: float | None = None
    per_task_upload_bits: dict[str, float] = field(default_factory=dict)
    per_task_compute_cycles: dict[str, float] = field(default_factory=dict)
    per_task_slot_upload_energy_j: dict[str, float] = field(default_factory=dict)


def _position_tuples(positions: dict[str, Any]) -> dict[str, tuple[float, float]]:
    return {
        str(node_id): (float(np.asarray(xy)[0]), float(np.asarray(xy)[1]))
        for node_id, xy in positions.items()
    }


def _assert_active_task_cache_protection(
    cache_state: np.ndarray,
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    protected_services_by_node: dict[str, tuple[int, ...]] | None,
) -> None:
    """Fail closed if an edge active task's required service is not cached."""

    if not protected_services_by_node:
        return
    node_indices = {
        str(node_id): index
        for index, node_id in enumerate(cache_nodes["node_ids"])
    }
    service_indices = {
        int(service_id): index
        for index, service_id in enumerate(service_catalog["service_ids"])
    }
    cache = np.asarray(cache_state, dtype=np.int8)
    for node_id, service_ids in protected_services_by_node.items():
        if node_id not in node_indices:
            raise ValueError(f"active task references missing edge node {node_id}")
        for service_id in service_ids:
            if service_id not in service_indices:
                raise ValueError(f"active task references missing service {service_id}")
            if int(cache[node_indices[node_id], service_indices[service_id]]) != 1:
                raise ValueError(
                    "cache update removed an active task's required service: "
                    f"node={node_id}, service={service_id}"
                )


def _realized_lifecycle_work(
    tasks: list[task_allocation.Task],
    matching: task_allocation.TransferExchangeMatchingResult,
    vehicle_positions: dict[str, np.ndarray],
    service_node_positions: dict[str, np.ndarray],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    dt_s: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Calculate actual one-slot work from KKT shares and the PDF link model."""

    uploaded_bits: dict[str, float] = {}
    computed_cycles: dict[str, float] = {}
    upload_energy_j: dict[str, float] = {}
    for task in tasks:
        node_id = matching.matching[task.task_id]
        uploaded_bits[task.task_id] = 0.0
        computed_cycles[task.task_id] = 0.0
        upload_energy_j[task.task_id] = 0.0
        if task.execution_stage == "UPLOADING":
            if task.input_bits <= 0.0:
                raise ValueError(f"uploading task {task.task_id} has no remaining data")
            if node_id == task_allocation.CLOUD:
                rate_bps = float(cache_config.cloud_uplink_rate_bps)
            else:
                role = task_allocation._edge_node_role(node_id, cache_nodes)
                if role is None:
                    raise ValueError(f"task {task.task_id} references unknown node {node_id}")
                bandwidth_fraction = float(
                    matching.bandwidth_allocations.get(node_id, {}).get(task.task_id, 0.0)
                )
                gain = task_allocation._channel_gain(
                    role,
                    vehicle_positions[task.vehicle_id],
                    service_node_positions[node_id],
                    cache_config,
                )
                rate_bps = float(
                    system_model.uplink_rate(
                        task_allocation._node_bandwidth(role, cache_config),
                        bandwidth_fraction,
                        cache_config.tx_power_w,
                        gain,
                        cache_config.noise_power_w,
                        noise_model=cache_config.noise_model,
                        noise_psd_w_per_hz=cache_config.noise_psd_w_per_hz,
                        receiver_noise_figure_db=cache_config.receiver_noise_figure_db,
                    )
                )
            if not math.isfinite(rate_bps) or rate_bps < 0.0:
                raise ValueError(f"task {task.task_id} has invalid uplink rate")
            actual_upload = min(float(task.input_bits), rate_bps * dt_s)
            uploaded_bits[task.task_id] = actual_upload
            if rate_bps > 0.0:
                upload_energy_j[task.task_id] = (
                    cache_config.tx_power_w * actual_upload / rate_bps
                )
        elif task.execution_stage == "COMPUTING":
            if task.cpu_cycles <= 0.0:
                raise ValueError(f"computing task {task.task_id} has no remaining work")
            if node_id == task_allocation.CLOUD:
                compute_rate = float(cache_config.cloud_cpu_cycles_per_s)
            else:
                role = task_allocation._edge_node_role(node_id, cache_nodes)
                if role is None:
                    raise ValueError(f"task {task.task_id} references unknown node {node_id}")
                cpu_fraction = float(
                    matching.cpu_allocations.get(node_id, {}).get(task.task_id, 0.0)
                )
                compute_rate = (
                    cpu_fraction * task_allocation._node_cpu(role, cache_config)
                )
            if not math.isfinite(compute_rate) or compute_rate < 0.0:
                raise ValueError(f"task {task.task_id} has invalid compute rate")
            computed_cycles[task.task_id] = min(
                float(task.cpu_cycles), compute_rate * dt_s
            )
        else:
            raise ValueError(
                f"cross-slot lifecycle does not accept stage {task.execution_stage!r}"
            )
    return uploaded_bits, computed_cycles, upload_energy_j


def _finite_allocation_total(values: Any) -> tuple[float | None, bool]:
    allocation_values = [float(value) for value in values]
    if not all(math.isfinite(value) for value in allocation_values):
        return None, False
    try:
        total = float(math.fsum(allocation_values))
    except OverflowError:
        return None, False
    if not math.isfinite(total):
        return None, False
    return total, True


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
    cloud_count = 0
    for target in result.matching.values():
        if target == task_allocation.CLOUD:
            cloud_count += 1
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
    edge_cpu_fraction_sum_by_node = {
        node_id: float(math.fsum(result.cpu_allocations.get(node_id, {}).values()))
        for node_id in node_ids
    }
    edge_bandwidth_fraction_sum_by_node = {
        node_id: float(math.fsum(result.bandwidth_allocations.get(node_id, {}).values()))
        for node_id in node_ids
    }
    edge_bandwidth_usage_hz_by_node = {
        node_id: edge_bandwidth_fraction_sum_by_node[node_id]
        * task_allocation._node_bandwidth(node_roles[node_id], cache_config)
        for node_id in node_ids
    }
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
    cloud_cpu_allocated, cloud_cpu_allocations_finite = _finite_allocation_total(
        result.cloud_cpu_allocations_cycles_per_s.values()
    )
    cloud_uplink_allocated, cloud_uplink_allocations_finite = (
        _finite_allocation_total(result.cloud_uplink_rate_allocations_bps.values())
    )
    cloud_uplink_capacity = result.cloud_uplink_rate_capacity_bps
    cloud_cpu_capacity = result.cloud_cpu_capacity_cycles_per_s
    cloud_uplink_capacity = (
        float(cloud_uplink_capacity)
        if cloud_uplink_capacity is not None
        and math.isfinite(float(cloud_uplink_capacity))
        else None
    )
    cloud_cpu_capacity = (
        float(cloud_cpu_capacity)
        if cloud_cpu_capacity is not None
        and math.isfinite(float(cloud_cpu_capacity))
        else None
    )
    selected_cloud_allocations_finite = (
        cloud_cpu_allocations_finite and cloud_uplink_allocations_finite
    )
    cloud_uplink_violation = (
        None
        if cloud_uplink_allocated is None or cloud_uplink_capacity is None
        else max(0.0, cloud_uplink_allocated - cloud_uplink_capacity)
    )
    cloud_cpu_violation = (
        None
        if cloud_cpu_allocated is None or cloud_cpu_capacity is None
        else max(0.0, cloud_cpu_allocated - cloud_cpu_capacity)
    )
    summary = {
        "resource_feasible": bool(result.resource_feasible),
        "matching_assignment": dict(result.matching),
        "matching_candidate_task_utility_sum": float(result.task_utility_sum),
        "matching_assignment_complete_pass": bool(result.resource_feasible),
        "matching_solution_scope": "transfer_exchange_local_stability",
        "matching_global_optimality_claimed": False,
        "matching_stability_claim": (
            "no improving enumerated transfer/exchange candidates under predicted-terminal-utility objective"
            if result.transfer_exchange_stable
            else "transfer/exchange search stopped before local stability"
        ),
        "candidate_evaluation_objective": (
            "predicted_terminal_utility_under_current_resource_estimates"
        ),
        "candidate_evaluation_is_prediction_only": True,
        "candidate_prediction_assumption": (
            "hold current candidate upload rates; current computing Edge tasks use their candidate KKT CPU shares; "
            "uploading Edge tasks on affected nodes use a one-shot KKT forecast that promotes the current "
            "candidate task set to the compute stage; round upload and compute slots separately; "
            "do not model staggered upload completion, future arrivals, or future mobility"
        ),
        "candidate_score_is_actual_reward": False,
        "kkt_objective_scope": "per-slot reciprocal delay-energy resource surrogate",
        "kkt_optimal_for_terminal_finite_horizon_objective": False,
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
        "cloud_resource_model": "independent_per_task_fixed_Rc_Fc",
        "cloud_configured_legacy_resource_mode": cache_config.cloud_resource_mode,
        "cloud_fallback_mode": cache_config.cloud_fallback_mode,
        "cloud_task_count": int(cloud_count),
        "cloud_uplink_allocated_bps": cloud_uplink_allocated,
        "cloud_uplink_capacity_bps": cloud_uplink_capacity,
        "cloud_uplink_capacity_violation_bps": cloud_uplink_violation,
        "cloud_cpu_allocated_cycles_per_s": cloud_cpu_allocated,
        "cloud_cpu_capacity_cycles_per_s": cloud_cpu_capacity,
        "cloud_cpu_capacity_violation_cycles_per_s": cloud_cpu_violation,
        "cloud_selected_resource_values_finite": bool(
            selected_cloud_allocations_finite
        ),
        "cloud_kkt_solve_count": int(result.cloud_kkt_solve_count),
        "cloud_kkt_max_uplink_allocated_bps": (
            result.cloud_kkt_max_uplink_allocated_bps
        ),
        "cloud_kkt_max_cpu_allocated_cycles_per_s": (
            result.cloud_kkt_max_cpu_allocated_cycles_per_s
        ),
        "cloud_kkt_max_uplink_capacity_violation_bps": (
            result.cloud_kkt_max_uplink_capacity_violation_bps
        ),
        "cloud_kkt_max_cpu_capacity_violation_cycles_per_s": (
            result.cloud_kkt_max_cpu_capacity_violation_cycles_per_s
        ),
        "cloud_kkt_nonfinite_result_count": int(
            result.cloud_kkt_nonfinite_result_count
        ),
        "node_task_count": {
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
            "cloud_uplink_rate_usage_bps": float(
                result.cloud_uplink_rate_used_bps
            ),
            "cloud_uplink_rate_capacity_bps": (
                None
                if result.cloud_uplink_rate_capacity_bps is None
                else float(result.cloud_uplink_rate_capacity_bps)
            ),
            "cloud_uplink_rate_remaining_bps": (
                None
                if result.cloud_uplink_rate_capacity_bps is None
                else max(
                    float(result.cloud_uplink_rate_capacity_bps)
                    - float(result.cloud_uplink_rate_used_bps),
                    0.0,
                )
            ),
            "cloud_uplink_rate_allocations_bps": dict(
                result.cloud_uplink_rate_allocations_bps
            ),
            "rsu_radio_bandwidth_fraction_sum_by_node": {
                node_id: float(math.fsum(result.bandwidth_allocations.get(node_id, {}).values()))
                for node_id in node_ids
                if node_roles[node_id] == "RSU"
            },
            "edge_cpu_fraction_sum_by_node": edge_cpu_fraction_sum_by_node,
            "edge_bandwidth_fraction_sum_by_node": edge_bandwidth_fraction_sum_by_node,
            "edge_bandwidth_usage_hz_by_node": edge_bandwidth_usage_hz_by_node,
            "edge_cpu_usage_cycles_per_s": edge_cpu_total,
            "edge_cpu_usage_by_node_cycles_per_s": edge_cpu_by_node,
        },
        "performance": {
            "mean_delay_s": mean_delay,
            "mean_energy_j": mean_energy,
            "matching_candidate_terminal_utility_sum": float(
                result.task_utility_sum
            ),
            "mean_matching_delay_energy_proxy_utility": mean_utility,
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
        "cache_download_delay_s": result.cache_download_delay_s,
        "cache_download_delay_normalized": result.cache_result_summary.get(
            "cache_download_delay_normalized"
        ),
        "cache_download_delay_reference_s": result.cache_result_summary.get(
            "cache_download_delay_reference_s"
        ),
        "cache_download_delay_reference_s_total": result.cache_result_summary.get(
            "cache_download_delay_reference_s_total",
            result.cache_result_summary.get("cache_download_delay_reference_s"),
        ),
        "cache_download_delay_reference_s_by_node": dict(
            result.cache_result_summary.get(
                "cache_download_delay_reference_s_by_node", {}
            )
        ),
        "cache_cost_by_node": dict(
            result.cache_result_summary.get("cache_cost_by_node", {})
        ),
        "cache_download_delay_s_by_node": dict(
            result.cache_result_summary.get("cache_download_delay_s_by_node", {})
        ),
        "movement_energy": result.movement_energy,
        "movement_energy_normalized": result.uav_result_summary.get(
            "movement_energy_normalized"
        ),
        "movement_energy_reference_j": result.uav_result_summary.get(
            "movement_energy_reference_j"
        ),
        "movement_energy_reference_j_total": result.uav_result_summary.get(
            "movement_energy_reference_j_total",
            result.uav_result_summary.get("movement_energy_reference_j"),
        ),
        "movement_energy_reference_j_by_uav": dict(
            result.uav_result_summary.get("movement_energy_reference_j_by_uav", {})
        ),
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
    lifecycle_mode: bool = False,
    fixed_matching: dict[str, str] | None = None,
    mutable_task_ids: set[str] | None = None,
    protected_services_by_node: dict[str, tuple[int, ...]] | None = None,
) -> SlotExecutionResult:
    """Execute one configured slot and realize only the current task phase.

    Core API does not read disk. The caller supplies prediction scenario slice for
    configuration and actual current state for scheduling. Cross-slot work requires
    an explicit lifecycle manager and never falls back to independent-slot service.
    """

    lifecycle_parameters = project_config.get_cross_slot_task_parameters(project_cfg)
    if not lifecycle_mode:
        raise ValueError(
            "independent_per_slot execution is disabled; use cross_slot_lifecycle"
        )
    execution_horizon_slots = int(lifecycle_parameters["execution_horizon_slots"])
    if slot_id is None or int(slot_id) < 0 or int(slot_id) >= execution_horizon_slots:
        raise ValueError(
            "cross-slot execution requires a zero-based slot_id inside the extended "
            f"horizon 0..{execution_horizon_slots - 1}"
        )

    if exact_q_timing_profile is not None:
        for key in (
            "matching_runtime",
            "matching_total_runtime",
            "node_cache_lookup_runtime",
            "node_eval_miss_runtime",
            "kkt_runtime",
            "kkt_coefficient_build_runtime",
            "kkt_solver_core_runtime",
            "kkt_result_assembly_runtime",
            "reward_cost_aggregation_runtime",
            "reward_final_combination_runtime",
        ):
            exact_q_timing_profile.setdefault(key, 0.0)

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
            cache_download_delay_s=0.0,
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
            cache_download_delay_s=None,
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
    if int(slot_id) + 1 > int(lifecycle_parameters["arrival_horizon_slots"]):
        if np.any(final_cache != previous_cache):
            raise ValueError("cache updates are forbidden after the task-arrival horizon")
    _assert_active_task_cache_protection(
        final_cache,
        service_catalog,
        cache_nodes,
        protected_services_by_node,
    )
    if lifecycle_mode and any(
        task.execution_stage not in {"UPLOADING", "COMPUTING"}
        for task in actual_tasks
    ):
        raise ValueError(
            "cross-slot lifecycle accepts only UPLOADING or COMPUTING tasks"
        )
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
        final_cache,
        project_cfg,
        cache_nodes,
        cache_config,
        kkt_cache=matching_kkt_cache,
        node_evaluation_cache=node_evaluation_cache,
        timing_profile=exact_q_timing_profile,
        cloud_fallback_mode=cache_config.cloud_fallback_mode,
        fixed_matching=fixed_matching,
        mutable_task_ids=mutable_task_ids,
    )
    if lifecycle_mode:
        expected_task_ids = {str(task.task_id) for task in actual_tasks}
        fixed = {str(key): str(value) for key, value in (fixed_matching or {}).items()}
        mutable = {str(value) for value in (mutable_task_ids or set())}
        if set(matching.matching) != expected_task_ids:
            raise RuntimeError("Matching result does not contain exactly the active task set")
        if set(fixed) & mutable or set(fixed) | mutable != expected_task_ids:
            raise RuntimeError("lifecycle Matching fixed/mutable task partition is invalid")
        if any(matching.matching.get(task_id) != node for task_id, node in fixed.items()):
            raise RuntimeError("Matching changed a previously fixed active-task node")
    if exact_q_timing_profile is not None:
        exact_q_matching_elapsed = time.perf_counter() - exact_q_matching_started_at
        exact_q_timing_profile["matching_total_runtime"] += exact_q_matching_elapsed
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
            cache_download_delay_s=None,
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

    if lifecycle_mode:
        dt_s = float(project_config.get_config_value(project_cfg, "simulation.dt_s"))
        if not math.isclose(dt_s, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("cross-slot lifecycle requires one-second slots")
        (
            per_task_upload_bits,
            per_task_compute_cycles,
            per_task_upload_energy_j,
        ) = _realized_lifecycle_work(
            actual_tasks,
            matching,
            actual_vehicle_positions,
            service_positions,
            cache_nodes,
            cache_config,
            dt_s,
        )
    else:
        per_task_upload_bits = {}
        per_task_compute_cycles = {}
        per_task_upload_energy_j = {}

    # Matching's current-slot terminal score is only a candidate diagnostic.
    # The lifecycle manager below the adapter is the single source of actual Q(t).
    actual_q = None
    reward_cost_started_at = (
        time.perf_counter() if exact_q_timing_profile is not None else 0.0
    )
    cache_cost = float(configuration.cache_cost)
    cache_download_delay_s = float(configuration.cache_download_delay_s)
    movement_cost = float(configuration.movement_cost)
    beta_ca = float(project_config.get_config_value(project_cfg, "objective.beta_ca"))
    beta_q = float(project_config.get_config_value(project_cfg, "objective.beta_q"))
    if exact_q_timing_profile is not None:
        exact_q_timing_profile["reward_cost_aggregation_runtime"] += (
            time.perf_counter() - reward_cost_started_at
        )
    reward_final_started_at = (
        time.perf_counter() if exact_q_timing_profile is not None else 0.0
    )
    system_utility = None
    if exact_q_timing_profile is not None:
        exact_q_timing_profile["reward_final_combination_runtime"] += (
            time.perf_counter() - reward_final_started_at
        )
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
        cache_download_delay_s=cache_download_delay_s,
        movement_energy=float(configuration.movement_energy),
        movement_cost=movement_cost,
        actual_task_utility=None,
        system_utility=None,
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
        matching_summary={
            **_matching_summary(matching, cache_nodes, cache_config),
            "fixed_matching_preservation_pass": True,
            "matching_task_set_complete_pass": True,
            "matching_candidate_objective": (
                "predicted_terminal_utility_under_current_resource_estimates"
            ),
            "candidate_evaluation_is_prediction_only": True,
            "candidate_score_is_actual_reward": False,
            "matching_global_optimality_claimed": False,
        },
        next_cache_state=final_cache.copy(),
        next_uav_positions=final_uav,
        beta_ca=beta_ca,
        beta_q=beta_q,
        per_task_upload_bits=per_task_upload_bits,
        per_task_compute_cycles=per_task_compute_cycles,
        per_task_slot_upload_energy_j=per_task_upload_energy_j,
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
    cache_download_delay_s: float
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
    for node_index, trigger in enumerate(cache_trigger):
        changed = bool(actual_cache[node_index])
        if int(trigger) == 1 and not changed:
            raise ValueError(
                f"policy Cache UPDATE for node {node_index} must produce an actual REPLACE"
            )
        if changed and not cache_configuration.is_legal_cache_replacement(
            previous_cache[node_index], final_cache[node_index]
        ):
            raise ValueError(
                f"policy Cache target for node {node_index} must delete at least one "
                "cached service and add at least one previously uncached service"
            )
    uav_configuration.validate_uav_configuration_transition(
        previous_uav,
        final_uav,
        uav_triggers,
        project_cfg,
        flight_boundary,
    )
    cache_result = cache_configuration.cache_update_cost_details(
        previous_cache,
        final_cache,
        cache_trigger,
        service_catalog["service_sizes_mb"],
        cache_nodes,
        cache_config,
    )
    search = uav_configuration.uav_search_config_from_project(project_cfg)
    (
        movement_energy,
        movement_cost,
        movement_distance_by_uav,
        movement_energy_by_uav,
        movement_cost_by_uav,
        movement_energy_normalized,
    ) = uav_configuration.movement_summary(
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
        cache_cost=float(cache_result["cache_cost"]),
        cache_download_delay_s=float(cache_result["cache_download_delay_s"]),
        movement_energy=float(movement_energy),
        movement_cost=float(movement_cost),
        objective_value=0.0,
        initial_objective=0.0,
        objective_history=[],
        outer_rounds=0,
        accepted_cache_updates=int(actual_cache.sum()),
        accepted_uav_updates=int(actual_uav),
        stop_reason="policy_target_prediction_skipped",
        cache_result_summary={
            "target_actions": [dict(value) for value in cache_actions],
            "cache_download_delay_s": float(cache_result["cache_download_delay_s"]),
            "cache_download_delay_normalized": float(
                cache_result["cache_download_delay_normalized"]
            ),
            "cache_download_delay_reference_s": float(
                cache_result["cache_download_delay_reference_s"]
            ),
            "cache_download_delay_reference_s_total": float(
                cache_result["cache_download_delay_reference_s_total"]
            ),
            "cache_download_delay_reference_s_by_node": {
                str(node_id): float(
                    cache_result["cache_download_delay_reference_s_by_node"][index]
                )
                for index, node_id in enumerate(cache_nodes["node_ids"])
            },
            "cache_cost_reference_kind": cache_nodes[
                "cache_cost_normalization"
            ]["reference_kind"],
            "cache_download_delay_s_by_node": {
                str(node_id): float(
                    cache_result["cache_download_delay_s_by_node"][index]
                )
                for index, node_id in enumerate(cache_nodes["node_ids"])
            },
            "cache_cost_by_node": {
                str(node_id): float(cache_result["cache_cost_by_node"][index])
                for index, node_id in enumerate(cache_nodes["node_ids"])
            },
        },
        uav_result_summary={
            "target_actions": [dict(value) for value in uav_actions],
            "movement_energy_normalized": movement_energy_normalized,
            "movement_energy_reference_j": search.movement_energy_reference_j,
            "movement_energy_reference_j_total": search.movement_energy_reference_j,
            "movement_energy_reference_j_by_uav": dict(
                search.movement_energy_reference_j_by_uav
            ),
            "movement_distance_m_by_uav": movement_distance_by_uav,
            "movement_energy_j_by_uav": movement_energy_by_uav,
            "movement_cost_by_uav": movement_cost_by_uav,
            "movement_cost_reference_kind": (
                "per_uav_conservative_speed_limit_upper_bound"
                if search.movement_energy_reference_j > 0.0
                else "zero_no_possible_positive_movement_energy"
            ),
        },
    )


def execute_configured_slot(
    previous_cache_state: np.ndarray,
    previous_uav_positions: dict[str, Any],
    final_cache_state: np.ndarray,
    final_uav_positions: dict[str, Any],
    scenario_slice: dict[str, Any],
    actual_tasks: list[task_allocation.Task],
    actual_vehicle_positions: dict[str, np.ndarray],
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
    exact_q_timing_profile: dict[str, float] | None = None,
    lifecycle_mode: bool = False,
    fixed_matching: dict[str, str] | None = None,
    mutable_task_ids: set[str] | None = None,
    protected_services_by_node: dict[str, tuple[int, ...]] | None = None,
) -> SlotExecutionResult:
    """执行确定的 Cache/UAV 配置及 Matching/KKT。"""

    lifecycle_parameters = project_config.get_cross_slot_task_parameters(project_cfg)
    if not lifecycle_mode:
        raise ValueError(
            "independent_per_slot execution is disabled; use cross_slot_lifecycle"
        )
    previous_cache = np.asarray(previous_cache_state, dtype=np.int8).copy()
    final_cache = np.asarray(final_cache_state, dtype=np.int8).copy()
    if slot_id is None or int(slot_id) < 0 or int(slot_id) >= int(
        lifecycle_parameters["execution_horizon_slots"]
    ):
        raise ValueError("execute_configured_slot requires a slot inside the extended horizon")
    if int(slot_id) + 1 > int(lifecycle_parameters["arrival_horizon_slots"]):
        if np.any(final_cache != previous_cache):
            raise ValueError("cache updates are forbidden after the task-arrival horizon")
    previous_uav = _position_tuples(previous_uav_positions)
    final_uav = _position_tuples(final_uav_positions)
    if previous_cache.shape != final_cache.shape:
        raise ValueError("最终 Cache 配置 shape 必须与当前 Cache 一致。")
    cache_configuration.validate_cache_shape_and_capacity(
        previous_cache, service_catalog, cache_nodes
    )
    cache_configuration.validate_cache_shape_and_capacity(
        final_cache, service_catalog, cache_nodes
    )
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
    if not bool(np.any(cache_updates)) and configuration.cache_download_delay_s != 0.0:
        raise AssertionError("Cache 配置未变化时下载时延必须为零。")
    return execute_slot(
        previous_cache,
        previous_uav,
        cache_updates,
        uav_moves,
        scenario_slice,
        actual_tasks,
        actual_vehicle_positions,
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
        exact_q_timing_profile=exact_q_timing_profile,
        lifecycle_mode=lifecycle_mode,
        fixed_matching=fixed_matching,
        mutable_task_ids=mutable_task_ids,
        protected_services_by_node=protected_services_by_node,
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
        rebuilt.cache_download_delay_s,
        configuration.cache_download_delay_s,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("configuration_override cache download delay 不匹配。")
    if not math.isclose(
        rebuilt.movement_cost, configuration.movement_cost, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("configuration_override movement cost 不匹配。")
    return rebuilt
