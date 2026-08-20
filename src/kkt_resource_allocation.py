#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Given matching -> KKT resource allocation -> actual-state task utility.

本模块只负责：给定已经确定的任务匹配 ``mu`` 后，使用 11.pdf F 节的 KKT
思想分配连续计算资源比例 ``w`` 和带宽比例 ``b``，并在 actual current state
下评价 ``J(mu)=sum_v Q_v``。

它不搜索 matching，不生成 transfer/exchange，不读取 prediction_scenarios，
也不扣 cache/UAV configuration cost。Actual-State Matching/KKT 阶段只使用
当前 slot 已观测任务和车辆位置。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import cache_configuration
import project_config
import system_model

LOCAL = "LOCAL"
CLOUD = "CLOUD"


@dataclass(frozen=True)
class KKTConfig:
    min_cpu_fraction: float
    min_bandwidth_fraction: float
    bisection_tolerance: float
    max_bisection_iterations: int


@dataclass(frozen=True)
class Task:
    task_id: str
    vehicle_id: str
    service_id: int
    input_bits: float
    cpu_cycles: float


@dataclass(frozen=True)
class KKTResourceAllocationResult:
    feasible: bool
    infeasible_reason: str
    cpu_allocations: dict[str, dict[str, float]]
    bandwidth_allocations: dict[str, dict[str, float]]
    per_node_cpu_sum: dict[str, float]
    per_node_bandwidth_sum: dict[str, float]
    cpu_multipliers: dict[str, float]
    bandwidth_multipliers: dict[str, float]


@dataclass(frozen=True)
class KKTMatchingEvaluation:
    resource_feasible: bool
    infeasible_reason: str
    task_utility_sum: float
    cpu_allocations: dict[str, dict[str, float]]
    bandwidth_allocations: dict[str, dict[str, float]]
    per_task_delay_s: dict[str, float]
    per_task_vehicle_energy_j: dict[str, float]
    per_task_utility: dict[str, float]
    per_task_execution_target: dict[str, str]
    per_node_task_counts: dict[str, int]
    cpu_resource_sums: dict[str, float]
    bandwidth_resource_sums: dict[str, float]


def kkt_config_from_project(config: dict[str, Any]) -> KKTConfig:
    get = project_config.get_config_value
    return KKTConfig(
        min_cpu_fraction=float(get(config, "algorithm.kkt.min_cpu_fraction")),
        min_bandwidth_fraction=float(get(config, "algorithm.kkt.min_bandwidth_fraction")),
        bisection_tolerance=float(get(config, "algorithm.kkt.bisection_tolerance")),
        max_bisection_iterations=int(get(config, "algorithm.kkt.max_bisection_iterations")),
    )


def _edge_node_role(node_id: str, cache_nodes: dict[str, Any]) -> str | None:
    if node_id in (LOCAL, CLOUD):
        return None
    node_ids = [str(value) for value in cache_nodes["node_ids"]]
    if node_id not in node_ids:
        raise ValueError(f"未知执行节点：{node_id}")
    role = str(cache_nodes["node_roles"][node_ids.index(node_id)])
    if role not in {"SV", "RSU", "UAV"}:
        raise ValueError(f"未知 edge node role：{role}")
    return role


def _node_cpu(role: str, cache_config: cache_configuration.CacheOptimizationConfig) -> float:
    if role == "SV":
        return cache_config.sv_cpu_cycles_per_s
    if role == "RSU":
        return cache_config.rsu_cpu_cycles_per_s
    if role == "UAV":
        return cache_config.uav_cpu_cycles_per_s
    raise AssertionError("未知 edge role。")


def _node_bandwidth(role: str, cache_config: cache_configuration.CacheOptimizationConfig) -> float:
    if role == "SV":
        return cache_config.sv_bandwidth_hz
    if role == "RSU":
        return cache_config.rsu_bandwidth_hz
    if role == "UAV":
        return cache_config.uav_bandwidth_hz
    raise AssertionError("未知 edge role。")


def _channel_gain(
    role: str,
    tv_xy: np.ndarray,
    node_xy: np.ndarray,
    cache_config: cache_configuration.CacheOptimizationConfig,
) -> float:
    horizontal = float(np.linalg.norm(tv_xy - node_xy))
    if role == "UAV":
        return float(
            system_model.air_channel_gain(
                horizontal,
                cache_config.uav_altitude_m,
                cache_config.air_eta1,
                cache_config.air_eta2,
                cache_config.air_beta_los,
                cache_config.air_beta_nlos,
                cache_config.air_alpha_los,
                cache_config.air_alpha_nlos,
            )["channel_gain"]
        )
    # Actual-State KKT 不使用 surrogate range；距离为0时只做数值保护。
    return float(
        system_model.ground_channel_gain(
            max(horizontal, 1e-9),
            cache_config.channel_gain_at_1m,
            cache_config.pathloss_exponent,
        )
    )


def _base_rate(
    role: str,
    tv_xy: np.ndarray,
    node_xy: np.ndarray,
    cache_config: cache_configuration.CacheOptimizationConfig,
) -> float:
    gain = _channel_gain(role, tv_xy, node_xy, cache_config)
    return float(
        system_model.uplink_rate(
            _node_bandwidth(role, cache_config),
            1.0,
            cache_config.tx_power_w,
            gain,
            cache_config.noise_power_w,
        )
    )


def solve_kkt_fraction_allocation(
    coefficients: list[float],
    epsilon: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[list[float], float, bool, str]:
    """解 min sum c_i / r_i, epsilon <= r_i <= 1, sum r_i <= 1。

    对正系数任务使用 ``r_i=clip(sqrt(c_i/lambda), epsilon, 1)``，并用确定性
    bisection 找到 ``sum_i r_i≈1``。零系数任务只保留 epsilon；若所有系数为0，
    返回确定性均分。
    """

    coeff = np.asarray(coefficients, dtype=np.float64)
    n = int(coeff.size)
    if n == 0:
        return [], 0.0, True, ""
    if epsilon * n > 1.0 + tolerance:
        return [], math.nan, False, f"epsilon * N = {epsilon * n} > 1"
    if np.any(~np.isfinite(coeff)) or np.any(coeff < 0.0):
        return [], math.nan, False, "KKT coefficient 必须 finite 且非负。"
    if n == 1:
        return [1.0], 0.0, True, ""
    positive = coeff > 0.0
    if not bool(np.any(positive)):
        return [1.0 / n] * n, 0.0, True, ""

    def alloc_for(lam: float) -> np.ndarray:
        result = np.full(n, epsilon, dtype=np.float64)
        result[positive] = np.sqrt(coeff[positive] / lam)
        return np.clip(result, epsilon, 1.0)

    low = 0.0
    high = 1.0
    while float(np.sum(alloc_for(high))) > 1.0 and high < 1.0e300:
        high *= 2.0
    if not math.isfinite(high):
        return [], math.nan, False, "KKT bisection high bound overflow。"
    mid = high
    allocation = alloc_for(high)
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        allocation = alloc_for(mid)
        total = float(np.sum(allocation))
        if abs(total - 1.0) <= tolerance:
            break
        if total > 1.0:
            low = mid
        else:
            high = mid
    # 正系数目标随资源增加下降，因此正常可行时尽量用满资源。
    total = float(np.sum(allocation))
    if total > 1.0 + max(tolerance * 10.0, 1e-10):
        return [], mid, False, f"KKT allocation sum {total} > 1。"
    return [float(value) for value in allocation], float(mid), True, ""


def _group_edge_tasks(
    tasks: list[Task],
    matching: dict[str, str],
    cache_nodes: dict[str, Any],
) -> dict[str, list[Task]]:
    grouped: dict[str, list[Task]] = {}
    for task in tasks:
        target = matching.get(task.task_id)
        if target is None:
            raise ValueError(f"matching 缺少任务 {task.task_id}")
        role = _edge_node_role(target, cache_nodes)
        if role is not None:
            grouped.setdefault(target, []).append(task)
    return grouped


def allocate_resources_kkt(
    tasks: list[Task],
    matching: dict[str, str],
    vehicle_positions: dict[str, np.ndarray],
    service_node_positions: dict[str, np.ndarray],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
) -> KKTResourceAllocationResult:
    """给定 matching 后，为 SV/RSU/UAV 节点分配 w 和 b。"""

    kkt_cfg = kkt_config_from_project(project_cfg)
    grouped = _group_edge_tasks(tasks, matching, cache_nodes)
    cpu_alloc: dict[str, dict[str, float]] = {}
    bw_alloc: dict[str, dict[str, float]] = {}
    cpu_sums: dict[str, float] = {}
    bw_sums: dict[str, float] = {}
    cpu_mu: dict[str, float] = {}
    bw_nu: dict[str, float] = {}

    for node_id, node_tasks in grouped.items():
        role = _edge_node_role(node_id, cache_nodes)
        assert role is not None
        n = len(node_tasks)
        if kkt_cfg.min_cpu_fraction * n > 1.0:
            return KKTResourceAllocationResult(False, f"{node_id} CPU epsilon infeasible", {}, {}, {}, {}, {}, {})
        if kkt_cfg.min_bandwidth_fraction * n > 1.0:
            return KKTResourceAllocationResult(False, f"{node_id} bandwidth epsilon infeasible", {}, {}, {}, {}, {}, {})
        node_cpu = _node_cpu(role, cache_config)
        cpu_coeff = [
            cache_config.omega_t * task.cpu_cycles / (cache_config.time_ref_s * node_cpu)
            for task in node_tasks
        ]
        cpu_values, mu, ok, reason = solve_kkt_fraction_allocation(
            cpu_coeff,
            kkt_cfg.min_cpu_fraction,
            kkt_cfg.bisection_tolerance,
            kkt_cfg.max_bisection_iterations,
        )
        if not ok:
            return KKTResourceAllocationResult(False, f"{node_id} CPU infeasible: {reason}", {}, {}, {}, {}, {}, {})

        bw_coeff = []
        for task in node_tasks:
            tv_xy = vehicle_positions[task.vehicle_id]
            node_xy = service_node_positions[node_id]
            base_rate = _base_rate(role, tv_xy, node_xy, cache_config)
            if base_rate <= 0.0:
                return KKTResourceAllocationResult(False, f"{node_id} base rate <= 0", {}, {}, {}, {}, {}, {})
            # bandwidth_loss_coefficient = D/Rbar * (omega_T/T_ref + omega_E*p/E_ref)
            bw_coeff.append(
                task.input_bits
                / base_rate
                * (
                    cache_config.omega_t / cache_config.time_ref_s
                    + cache_config.omega_e * cache_config.tx_power_w / cache_config.energy_ref_j
                )
            )
        bw_values, nu, ok, reason = solve_kkt_fraction_allocation(
            bw_coeff,
            kkt_cfg.min_bandwidth_fraction,
            kkt_cfg.bisection_tolerance,
            kkt_cfg.max_bisection_iterations,
        )
        if not ok:
            return KKTResourceAllocationResult(False, f"{node_id} bandwidth infeasible: {reason}", {}, {}, {}, {}, {}, {})

        cpu_alloc[node_id] = {
            task.task_id: float(value) for task, value in zip(node_tasks, cpu_values)
        }
        bw_alloc[node_id] = {
            task.task_id: float(value) for task, value in zip(node_tasks, bw_values)
        }
        cpu_sums[node_id] = float(sum(cpu_values))
        bw_sums[node_id] = float(sum(bw_values))
        cpu_mu[node_id] = float(mu)
        bw_nu[node_id] = float(nu)

    return KKTResourceAllocationResult(
        True,
        "",
        cpu_alloc,
        bw_alloc,
        cpu_sums,
        bw_sums,
        cpu_mu,
        bw_nu,
    )


def evaluate_matching_with_kkt(
    tasks: list[Task],
    matching: dict[str, str],
    vehicle_positions: dict[str, np.ndarray],
    service_node_positions: dict[str, np.ndarray],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
) -> KKTMatchingEvaluation:
    """评价给定 matching 的 J(mu)=sum_v Q_v。"""

    if not tasks:
        return KKTMatchingEvaluation(True, "", 0.0, {}, {}, {}, {}, {}, {}, {}, {}, {})
    allocation = allocate_resources_kkt(
        tasks,
        matching,
        vehicle_positions,
        service_node_positions,
        project_cfg,
        cache_nodes,
        cache_config,
    )
    if not allocation.feasible:
        return KKTMatchingEvaluation(
            False,
            allocation.infeasible_reason,
            float("-inf"),
            allocation.cpu_allocations,
            allocation.bandwidth_allocations,
            {},
            {},
            {},
            {},
            {},
            allocation.per_node_cpu_sum,
            allocation.per_node_bandwidth_sum,
        )

    per_delay: dict[str, float] = {}
    per_energy: dict[str, float] = {}
    per_utility: dict[str, float] = {}
    per_target: dict[str, str] = {}
    counts: dict[str, int] = {}
    total_utility = 0.0

    for task in tasks:
        target = matching[task.task_id]
        per_target[task.task_id] = target
        counts[target] = counts.get(target, 0) + 1
        if target == LOCAL:
            result = system_model.local_execution(
                task.cpu_cycles,
                cache_config.local_cpu_cycles_per_s,
                cache_config.local_energy_per_cycle_j,
            )
            delay = float(result["delay_s"])
            energy = float(result["vehicle_energy_j"])
        elif target == CLOUD:
            result = system_model.cloud_execution(
                task.input_bits,
                task.cpu_cycles,
                cache_config.cloud_uplink_rate_bps,
                cache_config.cloud_cpu_cycles_per_s,
                cache_config.cloud_delay_factor,
                cache_config.tx_power_w,
            )
            delay = float(result["total_delay_s"])
            energy = float(result["vehicle_energy_j"])
        else:
            role = _edge_node_role(target, cache_nodes)
            assert role is not None
            tv_xy = vehicle_positions[task.vehicle_id]
            node_xy = service_node_positions[target]
            gain = _channel_gain(role, tv_xy, node_xy, cache_config)
            edge = system_model.edge_execution(
                task.input_bits,
                task.cpu_cycles,
                _node_bandwidth(role, cache_config),
                allocation.bandwidth_allocations[target][task.task_id],
                cache_config.tx_power_w,
                gain,
                cache_config.noise_power_w,
                _node_cpu(role, cache_config),
                allocation.cpu_allocations[target][task.task_id],
            )
            delay = float(edge["total_delay_s"])
            energy = float(edge["vehicle_energy_j"])
        utility = float(
            system_model.task_utility(
                delay,
                energy,
                cache_config.time_ref_s,
                cache_config.energy_ref_j,
                cache_config.omega_t,
                cache_config.omega_e,
            )
        )
        per_delay[task.task_id] = delay
        per_energy[task.task_id] = energy
        per_utility[task.task_id] = utility
        total_utility += utility

    return KKTMatchingEvaluation(
        True,
        "",
        float(total_utility),
        allocation.cpu_allocations,
        allocation.bandwidth_allocations,
        per_delay,
        per_energy,
        per_utility,
        per_target,
        counts,
        allocation.per_node_cpu_sum,
        allocation.per_node_bandwidth_sum,
    )


def _evaluate_matching_equal_allocation(
    tasks: list[Task],
    matching: dict[str, str],
    vehicle_positions: dict[str, np.ndarray],
    service_node_positions: dict[str, np.ndarray],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
) -> float:
    """仅用于最小验证的 equal allocation 对照，不作为正式 solver 暴露。"""

    grouped = _group_edge_tasks(tasks, matching, cache_nodes)
    total = 0.0
    for task in tasks:
        target = matching[task.task_id]
        if target == LOCAL:
            result = system_model.local_execution(
                task.cpu_cycles,
                cache_config.local_cpu_cycles_per_s,
                cache_config.local_energy_per_cycle_j,
            )
            delay = result["delay_s"]
            energy = result["vehicle_energy_j"]
        elif target == CLOUD:
            result = system_model.cloud_execution(
                task.input_bits,
                task.cpu_cycles,
                cache_config.cloud_uplink_rate_bps,
                cache_config.cloud_cpu_cycles_per_s,
                cache_config.cloud_delay_factor,
                cache_config.tx_power_w,
            )
            delay = result["total_delay_s"]
            energy = result["vehicle_energy_j"]
        else:
            role = _edge_node_role(target, cache_nodes)
            assert role is not None
            share = 1.0 / len(grouped[target])
            gain = _channel_gain(
                role,
                vehicle_positions[task.vehicle_id],
                service_node_positions[target],
                cache_config,
            )
            result = system_model.edge_execution(
                task.input_bits,
                task.cpu_cycles,
                _node_bandwidth(role, cache_config),
                share,
                cache_config.tx_power_w,
                gain,
                cache_config.noise_power_w,
                _node_cpu(role, cache_config),
                share,
            )
            delay = result["total_delay_s"]
            energy = result["vehicle_energy_j"]
        total += system_model.task_utility(
            delay,
            energy,
            cache_config.time_ref_s,
            cache_config.energy_ref_j,
            cache_config.omega_t,
            cache_config.omega_e,
        )
    return float(total)


def load_actual_tasks(tasks_csv: Path, slot: int) -> list[Task]:
    tasks: list[Task] = []
    with tasks_csv.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["slot"]) != slot or int(row["arrival"]) != 1:
                continue
            vehicle_id = str(row["vehicle_id"])
            tasks.append(
                Task(
                    task_id=vehicle_id,
                    vehicle_id=vehicle_id,
                    service_id=int(row["service_id"]),
                    input_bits=float(row["input_size"]),
                    cpu_cycles=float(row["cpu_cycles"]),
                )
            )
    return tasks


def load_vehicle_positions(trajectory_csv: Path, time_s: float, vehicle_ids: set[str]) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    with trajectory_csv.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            vehicle_id = str(row["vehicle_id"])
            if vehicle_id not in vehicle_ids:
                continue
            if not math.isclose(float(row["time_s"]), time_s, rel_tol=0.0, abs_tol=1e-9):
                continue
            positions[vehicle_id] = np.asarray([float(row["x_m"]), float(row["y_m"])], dtype=np.float64)
    missing = sorted(vehicle_ids - set(positions))
    if missing:
        raise ValueError(f"trajectory.csv 缺少当前 time_s={time_s} 的车辆位置：{missing}")
    return positions


def build_service_node_positions(
    cache_nodes: dict[str, Any],
    actual_vehicle_positions: dict[str, np.ndarray],
    uav_positions: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    for node_id, role in zip(cache_nodes["node_ids"], cache_nodes["node_roles"]):
        node_id = str(node_id)
        role = str(role)
        if role == "SV":
            positions[node_id] = actual_vehicle_positions[node_id]
        elif role == "RSU":
            positions[node_id] = np.asarray(cache_nodes["fixed_positions"][node_id], dtype=np.float64)
        elif role == "UAV":
            positions[node_id] = np.asarray(uav_positions[node_id], dtype=np.float64)
    return positions


def _find_edge_case(
    tasks: list[Task],
    cache_state: np.ndarray,
    cache_nodes: dict[str, Any],
    prefer_two: bool,
) -> tuple[str, list[Task]] | None:
    node_ids = [str(value) for value in cache_nodes["node_ids"]]
    for node_index, node_id in enumerate(node_ids):
        matched = [task for task in tasks if int(cache_state[node_index, task.service_id]) == 1]
        if prefer_two and len(matched) >= 2:
            return node_id, matched[:2]
        if not prefer_two and matched:
            return node_id, matched[:1]
    return None


def run_minimal_validation(config_path: str = "system_config.toml") -> dict[str, Any]:
    cfg = project_config.load_config(config_path)
    project_config.validate_config(cfg)
    project_config.validate_for_formal_experiment(cfg)
    project_root = Path(cfg["_meta"]["project_root"])
    cache_cfg = cache_configuration.cache_config_from_project(cfg)
    # 只用 configuration 模块的数据读取 helper 生成一个确定性 C(t) 供 KKT actual-state 验证。
    scenarios = cache_configuration.load_prediction_scenarios(
        cache_configuration.resolve_path(
            project_root,
            project_config.get_config_value(cfg, "paths.prediction_scenarios"),
        )
    )
    origin = cache_configuration.scenario_at_origin(scenarios, 0)
    catalog = cache_configuration.load_service_catalog(
        cache_configuration.resolve_path(
            project_root,
            project_config.get_config_value(cfg, "paths.service_catalog"),
        )
    )
    cache_nodes = cache_configuration.build_cache_nodes(origin, cache_cfg)
    cache_state = cache_configuration.deterministic_initial_cache(
        origin,
        catalog,
        cache_nodes,
        cache_cfg,
    )
    slot = 20
    time_s = 120.0
    tasks = load_actual_tasks(project_root / "data/generated/tasks.csv", slot)
    roles = cache_configuration.load_json(project_root / "data/generated/vehicle_roles.json")
    sv_ids = {str(value) for value in roles["service_vehicles"]}
    tv_ids = {task.vehicle_id for task in tasks}
    actual_positions = load_vehicle_positions(
        project_root / "data/generated/trajectory.csv",
        time_s,
        tv_ids | sv_ids,
    )
    uav_positions = {
        str(item["id"]): np.asarray([float(item["initial_x_m"]), float(item["initial_y_m"])], dtype=np.float64)
        for item in project_config.get_config_value(cfg, "nodes.uav.instances", [])
    }
    service_positions = build_service_node_positions(cache_nodes, actual_positions, uav_positions)

    math_coeff = [1.0, 4.0, 9.0]
    expected = (np.sqrt(math_coeff) / np.sum(np.sqrt(math_coeff))).tolist()
    got, _, ok, reason = solve_kkt_fraction_allocation(
        math_coeff,
        1.0e-6,
        1.0e-12,
        100,
    )
    if not ok:
        raise AssertionError(reason)
    closed_form_error = float(np.max(np.abs(np.asarray(got) - np.asarray(expected))))
    single, _, ok, reason = solve_kkt_fraction_allocation([3.0], 1.0e-6, 1.0e-12, 100)
    if not ok or not math.isclose(single[0], 1.0, abs_tol=1e-12):
        raise AssertionError(f"单任务 KKT 验证失败：{single}, {reason}")

    matching_cloud = {task.task_id: CLOUD for task in tasks}
    case_a = evaluate_matching_with_kkt(
        tasks,
        matching_cloud,
        actual_positions,
        service_positions,
        cfg,
        cache_nodes,
        cache_cfg,
    )

    case_b = None
    case_c = None
    equal_c = None
    edge_one = _find_edge_case(tasks, cache_state, cache_nodes, prefer_two=False)
    if edge_one is not None:
        node_id, selected = edge_one
        matching_b = dict(matching_cloud)
        matching_b[selected[0].task_id] = node_id
        case_b = evaluate_matching_with_kkt(
            tasks,
            matching_b,
            actual_positions,
            service_positions,
            cfg,
            cache_nodes,
            cache_cfg,
        )
    edge_two = _find_edge_case(tasks, cache_state, cache_nodes, prefer_two=True)
    if edge_two is not None:
        node_id, selected = edge_two
        matching_c = dict(matching_cloud)
        for task in selected:
            matching_c[task.task_id] = node_id
        case_c = evaluate_matching_with_kkt(
            tasks,
            matching_c,
            actual_positions,
            service_positions,
            cfg,
            cache_nodes,
            cache_cfg,
        )
        equal_c = _evaluate_matching_equal_allocation(
            tasks,
            matching_c,
            actual_positions,
            service_positions,
            cache_nodes,
            cache_cfg,
        )
        if case_c.task_utility_sum + 1e-9 < equal_c:
            raise AssertionError("KKT utility 小于 equal allocation。")

    repeat = evaluate_matching_with_kkt(
        tasks,
        matching_cloud,
        actual_positions,
        service_positions,
        cfg,
        cache_nodes,
        cache_cfg,
    )
    if case_a != repeat:
        raise AssertionError("KKT repeated-run 不确定。")

    # 代表任务逐项复算：Cloud case 中取第一个任务。
    per_task_check = True
    if tasks:
        task = tasks[0]
        cloud = system_model.cloud_execution(
            task.input_bits,
            task.cpu_cycles,
            cache_cfg.cloud_uplink_rate_bps,
            cache_cfg.cloud_cpu_cycles_per_s,
            cache_cfg.cloud_delay_factor,
            cache_cfg.tx_power_w,
        )
        q = system_model.task_utility(
            cloud["total_delay_s"],
            cloud["vehicle_energy_j"],
            cache_cfg.time_ref_s,
            cache_cfg.energy_ref_j,
            cache_cfg.omega_t,
            cache_cfg.omega_e,
        )
        per_task_check = (
            math.isclose(case_a.per_task_delay_s[task.task_id], cloud["total_delay_s"], rel_tol=1e-12)
            and math.isclose(case_a.per_task_vehicle_energy_j[task.task_id], cloud["vehicle_energy_j"], rel_tol=1e-12)
            and math.isclose(case_a.per_task_utility[task.task_id], q, rel_tol=1e-12)
        )
    if not per_task_check:
        raise AssertionError("per-task system_model 复算不一致。")

    return {
        "slot": slot,
        "time_s": time_s,
        "arrival_task_count": len(tasks),
        "closed_form_error": closed_form_error,
        "single_task_allocation": single,
        "case_a": case_a,
        "case_b": case_b,
        "case_c": case_c,
        "equal_case_c_utility": equal_c,
        "per_task_check": per_task_check,
    }


def _summary_dict(result: KKTMatchingEvaluation | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "resource_feasible": result.resource_feasible,
        "infeasible_reason": result.infeasible_reason,
        "J_mu": result.task_utility_sum,
        "per_node_task_counts": result.per_node_task_counts,
        "cpu_allocations": result.cpu_allocations,
        "bandwidth_allocations": result.bandwidth_allocations,
        "cpu_sums": result.cpu_resource_sums,
        "bandwidth_sums": result.bandwidth_resource_sums,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Given matching -> KKT resource allocation validation.")
    parser.add_argument("--config", default="system_config.toml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_minimal_validation(args.config)
    printable = {
        "slot": result["slot"],
        "time_s": result["time_s"],
        "arrival_task_count": result["arrival_task_count"],
        "closed_form_error": result["closed_form_error"],
        "single_task_allocation": result["single_task_allocation"],
        "case_a": _summary_dict(result["case_a"]),
        "case_b": _summary_dict(result["case_b"]),
        "case_c": _summary_dict(result["case_c"]),
        "equal_case_c_utility": result["equal_case_c_utility"],
        "per_task_check": result["per_task_check"],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
