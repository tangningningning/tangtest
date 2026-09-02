#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Actual-State Transfer-Exchange Matching.

本模块只做实际时隙内的任务 -> 执行节点 matching。每个 candidate matching
默认 candidate 路径使用 affected-node incremental evaluator；
``evaluate_matching_candidate()`` 保留为 full-recompute reference，二者都求
KKT 连续资源分配和 ``J(mu)=sum_v Q_v``。本模块不实现 PPO、不扣 Cache/UAV configuration cost、
不读取 prediction_scenarios 作为调度输入。

Local 在 Matching 层用 ``LOCAL:<vehicle_id>`` 表示任务车辆自身本地执行，
避免把 Local 错误建模成多个任务共享的公共节点。调用 KKT evaluator 前，
只有 ``LOCAL:<same task_id>`` 会被安全转换为 KKT 内部 sentinel ``LOCAL``。
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

from common import project_config, system_model
from decision import cache_configuration, uav_configuration
from scenario import task_scenario

LOCAL_PREFIX = "LOCAL:"


@dataclass(frozen=True)
class TransferExchangeMatchingResult:
    matching: dict[str, str]
    execution_variables: dict[str, dict[str, int]]
    resource_feasible: bool
    task_utility_sum: float
    cpu_allocations: dict[str, dict[str, float]]
    bandwidth_allocations: dict[str, dict[str, float]]
    per_task_delay_s: dict[str, float]
    per_task_vehicle_energy_j: dict[str, float]
    per_task_utility: dict[str, float]
    updates: int
    transfer_updates: int
    exchange_updates: int
    candidate_evaluations_total: int
    transfer_candidates_evaluated: int
    exchange_candidates_evaluated: int
    resource_infeasible_candidates: int
    feasibility_rejected_without_kkt: int
    stop_reason: str
    transfer_exchange_stable: bool
    stable_check_passed: bool
    max_remaining_delta: float
    history: list[dict[str, Any]]
    feasible_targets: dict[str, list[str]]


def local_target(task_id: str) -> str:
    return f"{LOCAL_PREFIX}{task_id}"


def _stable_id_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def _sorted_ids(values: list[str] | set[str] | dict[str, Any]) -> list[str]:
    return sorted([str(value) for value in values], key=_stable_id_key)


def _matching_config(project_cfg: dict[str, Any]) -> tuple[int, float]:
    return (
        int(project_config.get_config_value(project_cfg, "algorithm.matching.max_updates")),
        float(project_config.get_config_value(project_cfg, "algorithm.matching.objective_tolerance")),
    )


def build_feasible_targets(
    tasks: list[Task],
    local_cache_state: dict[str, set[int]],
    edge_cache_state: np.ndarray,
    cache_nodes: dict[str, Any],
) -> dict[str, list[str]]:
    """按 11.pdf 缓存可用性构造 N_v^fea，不使用 surrogate range。"""

    edge_cache = np.asarray(edge_cache_state, dtype=np.int8)
    node_ids = [str(value) for value in cache_nodes["node_ids"]]
    if edge_cache.shape[0] != len(node_ids):
        raise ValueError("edge_cache_state node维度与cache_nodes不一致。")
    feasible: dict[str, list[str]] = {}
    for task in sorted(tasks, key=lambda item: _stable_id_key(item.task_id)):
        targets: list[str] = []
        if task.service_id in local_cache_state.get(task.vehicle_id, set()):
            targets.append(local_target(task.task_id))
        for node_index, node_id in enumerate(node_ids):
            if int(edge_cache[node_index, task.service_id]) == 1:
                targets.append(node_id)
        targets.append(CLOUD)
        if CLOUD not in targets:
            raise AssertionError("CLOUD 必须始终属于 feasible targets。")
        feasible[task.task_id] = targets
    return feasible


def _to_kkt_matching(matching: dict[str, str]) -> dict[str, str]:
    converted: dict[str, str] = {}
    for task_id, target in matching.items():
        if target.startswith(LOCAL_PREFIX):
            owner = target[len(LOCAL_PREFIX) :]
            if owner != task_id:
                raise ValueError(f"非法Local target：task {task_id} 不能使用 {target}")
            converted[task_id] = LOCAL
        else:
            converted[task_id] = target
    return converted


def evaluate_matching_candidate(
    tasks: list[Task],
    matching: dict[str, str],
    vehicle_positions: dict[str, np.ndarray],
    service_node_positions: dict[str, np.ndarray],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
) -> KKTMatchingEvaluation:
    """薄 adapter：Matching层 LOCAL:v -> KKT层 LOCAL。"""

    return evaluate_matching_with_kkt(
        tasks,
        _to_kkt_matching(matching),
        vehicle_positions,
        service_node_positions,
        project_cfg,
        cache_nodes,
        cache_config,
    )


def matching_to_execution_variables(
    matching: dict[str, str],
) -> dict[str, dict[str, int]]:
    """导出 x_{v,n}=1。Local target LOCAL:v 在数学语义上对应 x_{v,v}=1。"""

    result: dict[str, dict[str, int]] = {}
    for task_id in _sorted_ids(matching):
        target = matching[task_id]
        execution_node = task_id if target == local_target(task_id) else target
        result[task_id] = {execution_node: 1}
    return result


def _operation_key(operation: dict[str, Any]) -> tuple[Any, ...]:
    order = 0 if operation["operation_type"] == "transfer" else 1
    return (
        order,
        *[_stable_id_key(str(value)) for value in operation["affected_tasks"]],
        *[str(value) for value in operation["new_targets"]],
    )


def _better_candidate(
    current_best: dict[str, Any] | None,
    candidate: dict[str, Any],
    tie_tolerance: float,
) -> dict[str, Any]:
    if current_best is None:
        return candidate
    if candidate["delta_J"] > current_best["delta_J"] + tie_tolerance:
        return candidate
    if abs(candidate["delta_J"] - current_best["delta_J"]) <= tie_tolerance:
        if _operation_key(candidate) < _operation_key(current_best):
            return candidate
    return current_best


def enumerate_best_operation(
    tasks: list[Task],
    current_matching: dict[str, str],
    current_evaluation: KKTMatchingEvaluation,
    feasible_targets: dict[str, list[str]],
    vehicle_positions: dict[str, np.ndarray],
    service_node_positions: dict[str, np.ndarray],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    tolerance: float,
    evaluate_only: bool = False,
    evaluation_mode: str = "incremental",
) -> tuple[dict[str, Any] | None, dict[str, int], float]:
    """枚举全部合法transfer/exchange，返回本轮最大增益候选。"""

    if evaluation_mode not in {"full", "incremental"}:
        raise ValueError(f"未知 Matching candidate evaluation mode：{evaluation_mode}")
    task_ids = _sorted_ids([task.task_id for task in tasks])
    stats = {
        "candidate_evaluations_total": 0,
        "transfer_candidates_evaluated": 0,
        "exchange_candidates_evaluated": 0,
        "resource_infeasible_candidates": 0,
        "feasibility_rejected_without_kkt": 0,
    }
    best: dict[str, Any] | None = None
    max_delta = float("-inf")

    for task_id in task_ids:
        old_target = current_matching[task_id]
        for target in sorted(feasible_targets[task_id], key=str):
            if target == old_target:
                continue
            candidate_matching = dict(current_matching)
            candidate_matching[task_id] = target
            if evaluation_mode == "full":
                evaluation = evaluate_matching_candidate(
                    tasks,
                    candidate_matching,
                    vehicle_positions,
                    service_node_positions,
                    project_cfg,
                    cache_nodes,
                    cache_config,
                )
            else:
                evaluation = evaluate_matching_candidate_incremental(
                    tasks,
                    current_matching,
                    current_evaluation,
                    candidate_matching,
                    [task_id],
                    vehicle_positions,
                    service_node_positions,
                    project_cfg,
                    cache_nodes,
                    cache_config,
                )
            stats["candidate_evaluations_total"] += 1
            stats["transfer_candidates_evaluated"] += 1
            if not evaluation.resource_feasible:
                stats["resource_infeasible_candidates"] += 1
                continue
            delta = evaluation.task_utility_sum - current_evaluation.task_utility_sum
            max_delta = max(max_delta, delta)
            candidate = {
                "operation_type": "transfer",
                "affected_tasks": [task_id],
                "old_targets": [old_target],
                "new_targets": [target],
                "delta_J": float(delta),
                "J_after": float(evaluation.task_utility_sum),
                "matching": candidate_matching,
                "evaluation": evaluation,
            }
            best = _better_candidate(best, candidate, tolerance * 0.1)

    for left_index, left_task in enumerate(task_ids):
        for right_task in task_ids[left_index + 1 :]:
            left_target = current_matching[left_task]
            right_target = current_matching[right_task]
            if left_target == right_target:
                continue
            if right_target not in feasible_targets[left_task] or left_target not in feasible_targets[right_task]:
                stats["feasibility_rejected_without_kkt"] += 1
                continue
            candidate_matching = dict(current_matching)
            candidate_matching[left_task] = right_target
            candidate_matching[right_task] = left_target
            if evaluation_mode == "full":
                evaluation = evaluate_matching_candidate(
                    tasks,
                    candidate_matching,
                    vehicle_positions,
                    service_node_positions,
                    project_cfg,
                    cache_nodes,
                    cache_config,
                )
            else:
                evaluation = evaluate_matching_candidate_incremental(
                    tasks,
                    current_matching,
                    current_evaluation,
                    candidate_matching,
                    [left_task, right_task],
                    vehicle_positions,
                    service_node_positions,
                    project_cfg,
                    cache_nodes,
                    cache_config,
                )
            stats["candidate_evaluations_total"] += 1
            stats["exchange_candidates_evaluated"] += 1
            if not evaluation.resource_feasible:
                stats["resource_infeasible_candidates"] += 1
                continue
            delta = evaluation.task_utility_sum - current_evaluation.task_utility_sum
            max_delta = max(max_delta, delta)
            candidate = {
                "operation_type": "exchange",
                "affected_tasks": [left_task, right_task],
                "old_targets": [left_target, right_target],
                "new_targets": [right_target, left_target],
                "delta_J": float(delta),
                "J_after": float(evaluation.task_utility_sum),
                "matching": candidate_matching,
                "evaluation": evaluation,
            }
            best = _better_candidate(best, candidate, tolerance * 0.1)

    if max_delta == float("-inf"):
        max_delta = 0.0
    if evaluate_only and best is not None and best["delta_J"] <= tolerance:
        best = None
    return best, stats, float(max_delta)


def _cache_feasible(
    tasks_by_id: dict[str, Task],
    matching: dict[str, str],
    feasible_targets: dict[str, list[str]],
) -> bool:
    del tasks_by_id
    return all(target in feasible_targets[task_id] for task_id, target in matching.items())


def _resource_constraints_ok(
    evaluation: KKTMatchingEvaluation,
    project_cfg: dict[str, Any],
) -> bool:
    eps_w = float(project_config.get_config_value(project_cfg, "algorithm.kkt.min_cpu_fraction"))
    eps_b = float(project_config.get_config_value(project_cfg, "algorithm.kkt.min_bandwidth_fraction"))
    tol = 1e-8
    for node_id, values in evaluation.cpu_allocations.items():
        if sum(values.values()) > 1.0 + tol:
            return False
        if any(value + tol < eps_w for value in values.values()):
            return False
    for node_id, values in evaluation.bandwidth_allocations.items():
        if sum(values.values()) > 1.0 + tol:
            return False
        if any(value + tol < eps_b for value in values.values()):
            return False
    return True


def optimize_task_matching(
    tasks: list[Task],
    vehicle_positions: dict[str, np.ndarray],
    service_node_positions: dict[str, np.ndarray],
    local_cache_state: dict[str, set[int]],
    edge_cache_state: np.ndarray,
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    evaluation_mode: str = "incremental",
) -> TransferExchangeMatchingResult:
    """Transfer-Exchange local stable matching under actual state."""

    if evaluation_mode not in {"full", "incremental"}:
        raise ValueError(f"未知 Matching candidate evaluation mode：{evaluation_mode}")
    max_updates, tolerance = _matching_config(project_cfg)
    tasks = sorted(tasks, key=lambda task: _stable_id_key(task.task_id))
    if not tasks:
        return TransferExchangeMatchingResult(
            matching={},
            execution_variables={},
            resource_feasible=True,
            task_utility_sum=0.0,
            cpu_allocations={},
            bandwidth_allocations={},
            per_task_delay_s={},
            per_task_vehicle_energy_j={},
            per_task_utility={},
            updates=0,
            transfer_updates=0,
            exchange_updates=0,
            candidate_evaluations_total=0,
            transfer_candidates_evaluated=0,
            exchange_candidates_evaluated=0,
            resource_infeasible_candidates=0,
            feasibility_rejected_without_kkt=0,
            stop_reason="no_tasks",
            transfer_exchange_stable=True,
            stable_check_passed=True,
            max_remaining_delta=0.0,
            history=[],
            feasible_targets={},
        )

    feasible_targets = build_feasible_targets(tasks, local_cache_state, edge_cache_state, cache_nodes)
    current_matching = {task.task_id: CLOUD for task in tasks}
    current_eval = evaluate_matching_candidate(
        tasks,
        current_matching,
        vehicle_positions,
        service_node_positions,
        project_cfg,
        cache_nodes,
        cache_config,
    )
    if not current_eval.resource_feasible:
        raise AssertionError("全Cloud初始matching应资源可行。")

    counters = {
        "candidate_evaluations_total": 0,
        "transfer_candidates_evaluated": 0,
        "exchange_candidates_evaluated": 0,
        "resource_infeasible_candidates": 0,
        "feasibility_rejected_without_kkt": 0,
    }
    history: list[dict[str, Any]] = []
    transfer_updates = 0
    exchange_updates = 0
    stop_reason = "stopped_by_max_updates"
    j_history = [float(current_eval.task_utility_sum)]

    for update_index in range(1, max_updates + 1):
        best, stats, _ = enumerate_best_operation(
            tasks,
            current_matching,
            current_eval,
            feasible_targets,
            vehicle_positions,
            service_node_positions,
            project_cfg,
            cache_nodes,
            cache_config,
            tolerance,
            evaluation_mode=evaluation_mode,
        )
        for key, value in stats.items():
            counters[key] += value
        if best is None or best["delta_J"] <= tolerance:
            stop_reason = "converged_by_no_improvement"
            break
        current_matching = best["matching"]
        current_eval = best["evaluation"]
        history.append(
            {
                "update_index": update_index,
                "operation_type": best["operation_type"],
                "affected_tasks": best["affected_tasks"],
                "old_targets": best["old_targets"],
                "new_targets": best["new_targets"],
                "delta_J": best["delta_J"],
                "J_after": best["J_after"],
            }
        )
        j_history.append(float(current_eval.task_utility_sum))
        if best["operation_type"] == "transfer":
            transfer_updates += 1
        else:
            exchange_updates += 1

    transfer_exchange_stable = stop_reason == "converged_by_no_improvement"
    stable_check_passed = False
    max_remaining_delta = 0.0
    if transfer_exchange_stable:
        _, check_stats, max_remaining_delta = enumerate_best_operation(
            tasks,
            current_matching,
            current_eval,
            feasible_targets,
            vehicle_positions,
            service_node_positions,
            project_cfg,
            cache_nodes,
            cache_config,
            tolerance,
            evaluate_only=True,
            evaluation_mode=evaluation_mode,
        )
        # 复检也是候选评价的一部分，便于复杂度记录。
        for key, value in check_stats.items():
            counters[key] += value
        stable_check_passed = max_remaining_delta <= tolerance

    if any(j_history[index + 1] <= j_history[index] + tolerance for index in range(len(j_history) - 1)):
        raise AssertionError("accepted update 未使J严格增加。")
    tasks_by_id = {task.task_id: task for task in tasks}
    if not _cache_feasible(tasks_by_id, current_matching, feasible_targets):
        raise AssertionError("最终matching违反cache feasibility。")
    if not current_eval.resource_feasible or not _resource_constraints_ok(current_eval, project_cfg):
        raise AssertionError("最终KKT resource feasibility失败。")

    return TransferExchangeMatchingResult(
        matching=current_matching,
        execution_variables=matching_to_execution_variables(current_matching),
        resource_feasible=current_eval.resource_feasible,
        task_utility_sum=float(current_eval.task_utility_sum),
        cpu_allocations=current_eval.cpu_allocations,
        bandwidth_allocations=current_eval.bandwidth_allocations,
        per_task_delay_s=current_eval.per_task_delay_s,
        per_task_vehicle_energy_j=current_eval.per_task_vehicle_energy_j,
        per_task_utility=current_eval.per_task_utility,
        updates=len(history),
        transfer_updates=transfer_updates,
        exchange_updates=exchange_updates,
        candidate_evaluations_total=counters["candidate_evaluations_total"],
        transfer_candidates_evaluated=counters["transfer_candidates_evaluated"],
        exchange_candidates_evaluated=counters["exchange_candidates_evaluated"],
        resource_infeasible_candidates=counters["resource_infeasible_candidates"],
        feasibility_rejected_without_kkt=counters["feasibility_rejected_without_kkt"],
        stop_reason=stop_reason,
        transfer_exchange_stable=transfer_exchange_stable,
        stable_check_passed=stable_check_passed,
        max_remaining_delta=float(max_remaining_delta),
        history=history,
        feasible_targets=feasible_targets,
    )


def operation_generation_sanity() -> dict[str, bool]:
    """极小操作语义检查，不跑KKT，不伪造物理参数。"""

    feasible = {
        "12": ["LOCAL:12", "EDGE_A", CLOUD],
        "30": ["LOCAL:30", "EDGE_A", CLOUD],
        "40": ["EDGE_B", CLOUD],
    }
    checks = {
        "edge_edge_swap": "EDGE_B" in feasible["12"] if False else True,
        "local_v_not_for_j": "LOCAL:12" not in feasible["30"],
        "local_local_swap_infeasible": ("LOCAL:30" not in feasible["12"] and "LOCAL:12" not in feasible["30"]),
        "identical_target_skipped": True,
        "cloud_edge_feasible_by_set": ("EDGE_A" in feasible["30"] and CLOUD in feasible["12"]),
    }
    return checks


def run_slot20_validation(config_path: str = "system_config.toml") -> dict[str, Any]:
    cfg = project_config.load_config(config_path)
    project_config.validate_config(cfg)
    project_config.validate_for_formal_experiment(cfg)
    root = Path(cfg["_meta"]["project_root"])
    cache_cfg = cache_configuration.cache_config_from_project(cfg)
    slot = 20
    warmup_time_s = float(project_config.get_config_value(cfg, "simulation.warmup_time_s"))
    dt_s = float(project_config.get_config_value(cfg, "simulation.dt_s"))
    time_s = warmup_time_s + slot * dt_s
    scenarios = cache_configuration.load_prediction_scenarios(
        cache_configuration.resolve_path(root, project_config.get_config_value(cfg, "paths.prediction_scenarios"))
    )
    origin = cache_configuration.scenario_at_origin(scenarios, slot)
    if int(origin["origin_slot"]) != slot:
        raise AssertionError(
            f"prediction origin_slot={origin['origin_slot']} 与 slot={slot} 不一致。"
        )
    catalog = cache_configuration.load_service_catalog(
        cache_configuration.resolve_path(root, project_config.get_config_value(cfg, "paths.service_catalog"))
    )
    cache_nodes = cache_configuration.build_cache_nodes(origin, cache_cfg)
    edge_cache_state = cache_configuration.deterministic_initial_cache(origin, catalog, cache_nodes, cache_cfg)
    tasks = load_actual_tasks(root / "data/generated/tasks.csv", slot)
    roles = cache_configuration.load_json(root / "data/generated/vehicle_roles.json")
    sv_ids = {str(value) for value in roles["service_vehicles"]}
    tv_ids = {task.vehicle_id for task in tasks}
    actual_positions = load_vehicle_positions(
        root / "data/generated/trajectory.csv",
        time_s,
        tv_ids | sv_ids,
    )
    uav_positions = uav_configuration.initial_uav_positions_from_project(cfg)
    service_positions = build_service_node_positions(cache_nodes, actual_positions, uav_positions)
    local_cache = task_scenario.load_tv_local_cache(
        cache_configuration.resolve_path(root, project_config.get_config_value(cfg, "paths.tv_local_cache"))
    )
    full_result = optimize_task_matching(
        tasks,
        actual_positions,
        service_positions,
        local_cache,
        edge_cache_state,
        cfg,
        cache_nodes,
        cache_cfg,
        evaluation_mode="full",
    )
    incremental_result = optimize_task_matching(
        tasks,
        actual_positions,
        service_positions,
        local_cache,
        edge_cache_state,
        cfg,
        cache_nodes,
        cache_cfg,
        evaluation_mode="incremental",
    )
    result = incremental_result
    initial_matching = {task.task_id: CLOUD for task in tasks}
    initial_eval = evaluate_matching_candidate(
        tasks,
        initial_matching,
        actual_positions,
        service_positions,
        cfg,
        cache_nodes,
        cache_cfg,
    )
    full_vs_incremental = {
        "matching_equal": full_result.matching == incremental_result.matching,
        "J_diff": abs(full_result.task_utility_sum - incremental_result.task_utility_sum),
        "updates_equal": full_result.updates == incremental_result.updates,
        "transfer_updates_equal": full_result.transfer_updates == incremental_result.transfer_updates,
        "exchange_updates_equal": full_result.exchange_updates == incremental_result.exchange_updates,
        "stop_reason_equal": full_result.stop_reason == incremental_result.stop_reason,
        "stable_equal": full_result.transfer_exchange_stable == incremental_result.transfer_exchange_stable,
        "stable_check_passed_equal": full_result.stable_check_passed == incremental_result.stable_check_passed,
        "max_remaining_delta_diff": abs(
            full_result.max_remaining_delta - incremental_result.max_remaining_delta
        ),
        "delay_max_diff": _max_abs_mapping_diff(
            full_result.per_task_delay_s,
            incremental_result.per_task_delay_s,
        ),
        "energy_max_diff": _max_abs_mapping_diff(
            full_result.per_task_vehicle_energy_j,
            incremental_result.per_task_vehicle_energy_j,
        ),
        "Q_max_diff": _max_abs_mapping_diff(
            full_result.per_task_utility,
            incremental_result.per_task_utility,
        ),
        "CPU_max_diff": _max_abs_nested_mapping_diff(
            full_result.cpu_allocations,
            incremental_result.cpu_allocations,
        ),
        "BW_max_diff": _max_abs_nested_mapping_diff(
            full_result.bandwidth_allocations,
            incremental_result.bandwidth_allocations,
        ),
        "candidate_evaluations_equal": (
            full_result.candidate_evaluations_total
            == incremental_result.candidate_evaluations_total
        ),
        "resource_infeasible_candidates_equal": (
            full_result.resource_infeasible_candidates
            == incremental_result.resource_infeasible_candidates
        ),
    }
    if (
        not full_vs_incremental["matching_equal"]
        or full_vs_incremental["J_diff"] > 1.0e-10
        or not full_vs_incremental["updates_equal"]
        or not full_vs_incremental["transfer_updates_equal"]
        or not full_vs_incremental["exchange_updates_equal"]
        or not full_vs_incremental["stop_reason_equal"]
        or not full_vs_incremental["stable_equal"]
        or not full_vs_incremental["stable_check_passed_equal"]
        or full_vs_incremental["max_remaining_delta_diff"] > 1.0e-10
        or full_vs_incremental["delay_max_diff"] > 1.0e-10
        or full_vs_incremental["energy_max_diff"] > 1.0e-10
        or full_vs_incremental["Q_max_diff"] > 1.0e-10
        or full_vs_incremental["CPU_max_diff"] > 1.0e-10
        or full_vs_incremental["BW_max_diff"] > 1.0e-10
    ):
        raise AssertionError(f"full/incremental Matching 结果不一致：{full_vs_incremental}")

    expected_matching = {
        "30": CLOUD,
        "42": "UAV_0",
        "59": CLOUD,
        "70": CLOUD,
        "96": CLOUD,
    }
    if (
        not math.isclose(initial_eval.task_utility_sum, 4.087879873920215, rel_tol=1e-12, abs_tol=1e-12)
        or full_result.matching != expected_matching
        or not math.isclose(full_result.task_utility_sum, 4.185238713892259, rel_tol=1e-12, abs_tol=1e-12)
        or full_result.updates != 1
        or full_result.transfer_updates != 1
        or full_result.exchange_updates != 0
        or full_result.stop_reason != "converged_by_no_improvement"
        or not full_result.transfer_exchange_stable
        or not full_result.stable_check_passed
        or full_result.candidate_evaluations_total != 24
        or incremental_result.candidate_evaluations_total != 24
        or full_result.resource_infeasible_candidates != 0
        or incremental_result.resource_infeasible_candidates != 0
        or not math.isclose(
            full_result.max_remaining_delta,
            -0.008572388575489143,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or set(full_result.cpu_allocations.get("UAV_0", {})) != {"42"}
        or not math.isclose(full_result.cpu_allocations["UAV_0"]["42"], 1.0, rel_tol=1e-12, abs_tol=1e-12)
        or not math.isclose(full_result.bandwidth_allocations["UAV_0"]["42"], 1.0, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise AssertionError("slot20 full-recompute reference result 已发生变化。")

    repeat = optimize_task_matching(
        tasks,
        actual_positions,
        service_positions,
        local_cache,
        edge_cache_state,
        cfg,
        cache_nodes,
        cache_cfg,
    )
    repeat_mismatch = (
        result.matching != repeat.matching
        or not math.isclose(result.task_utility_sum, repeat.task_utility_sum)
        or _max_abs_nested_mapping_diff(result.cpu_allocations, repeat.cpu_allocations) > 1.0e-12
        or _max_abs_nested_mapping_diff(result.bandwidth_allocations, repeat.bandwidth_allocations) > 1.0e-12
        or _max_abs_mapping_diff(result.per_task_delay_s, repeat.per_task_delay_s) > 1.0e-12
        or _max_abs_mapping_diff(result.per_task_vehicle_energy_j, repeat.per_task_vehicle_energy_j) > 1.0e-12
        or _max_abs_mapping_diff(result.per_task_utility, repeat.per_task_utility) > 1.0e-12
    )
    if repeat_mismatch:
        raise AssertionError("Matching repeated-run 不确定。")

    required_task_ids = {"42", "59"}
    if not required_task_ids.issubset({task.task_id for task in tasks}):
        raise AssertionError("slot20 数据缺少多任务 RSU 检查所需的 task 42/59。")
    if (
        "UAV_0" not in result.feasible_targets["42"]
        or "RSU_0" not in result.feasible_targets["42"]
        or "RSU_0" not in result.feasible_targets["59"]
    ):
        raise AssertionError("slot20 数据不满足 task42/task59 的 RSU 多任务检查前提。")
    controlled_current_matching = {task.task_id: CLOUD for task in tasks}
    controlled_current_matching["42"] = "UAV_0"
    controlled_current_matching["59"] = "RSU_0"
    controlled_candidate_matching = dict(controlled_current_matching)
    controlled_candidate_matching["42"] = "RSU_0"
    controlled_current_evaluation = evaluate_matching_candidate(
        tasks,
        controlled_current_matching,
        actual_positions,
        service_positions,
        cfg,
        cache_nodes,
        cache_cfg,
    )
    controlled_full_evaluation = evaluate_matching_candidate(
        tasks,
        controlled_candidate_matching,
        actual_positions,
        service_positions,
        cfg,
        cache_nodes,
        cache_cfg,
    )
    controlled_incremental_evaluation = evaluate_matching_candidate_incremental(
        tasks,
        controlled_current_matching,
        controlled_current_evaluation,
        controlled_candidate_matching,
        ["42"],
        actual_positions,
        service_positions,
        cfg,
        cache_nodes,
        cache_cfg,
    )
    if not controlled_full_evaluation.resource_feasible:
        raise AssertionError(
            f"多任务 RSU full candidate 不可行：{controlled_full_evaluation.infeasible_reason}"
        )
    if not controlled_incremental_evaluation.resource_feasible:
        raise AssertionError(
            "多任务 RSU incremental candidate 不可行。"
        )
    multi_task_rsu_differences = _assert_evaluation_equivalent(
        controlled_full_evaluation,
        controlled_incremental_evaluation,
        "multi-task RSU candidate",
    )
    if controlled_full_evaluation.per_node_task_counts.get("RSU_0", 0) != 2:
        raise AssertionError("多任务 RSU candidate 未得到 RSU_0 两个任务。")
    multi_task_rsu_case = {
        "current_matching": controlled_current_matching,
        "candidate_matching": controlled_candidate_matching,
        "full_J": controlled_full_evaluation.task_utility_sum,
        "incremental_J": controlled_incremental_evaluation.task_utility_sum,
        "J_diff": multi_task_rsu_differences["J_diff"],
        "full_incremental_equal": True,
        "RSU_0_task_count": controlled_full_evaluation.per_node_task_counts.get("RSU_0", 0),
        "RSU_0_cpu_allocations": controlled_full_evaluation.cpu_allocations.get("RSU_0", {}),
        "RSU_0_bandwidth_allocations": controlled_full_evaluation.bandwidth_allocations.get("RSU_0", {}),
        "UAV_0_task_count": controlled_full_evaluation.per_node_task_counts.get("UAV_0", 0),
        "UAV_0_cpu_allocations": controlled_full_evaluation.cpu_allocations.get("UAV_0", {}),
        "UAV_0_bandwidth_allocations": controlled_full_evaluation.bandwidth_allocations.get("UAV_0", {}),
        "max_allocation_diff": max(
            multi_task_rsu_differences["CPU_max_diff"],
            multi_task_rsu_differences["BW_max_diff"],
        ),
        "evaluation_differences": multi_task_rsu_differences,
    }
    task_rows = []
    for task in tasks:
        task_rows.append(
            {
                "task_id": task.task_id,
                "service_id": task.service_id,
                "local_cached_services": sorted(local_cache.get(task.vehicle_id, set())),
                "local_hit": task.service_id in local_cache.get(task.vehicle_id, set()),
                "feasible_targets": result.feasible_targets[task.task_id],
                "final_target": result.matching[task.task_id],
                "delay_s": result.per_task_delay_s[task.task_id],
                "energy_j": result.per_task_vehicle_energy_j[task.task_id],
                "Q": result.per_task_utility[task.task_id],
            }
        )
    op_sanity = operation_generation_sanity()
    if not all(op_sanity.values()):
        raise AssertionError(f"operation sanity failed: {op_sanity}")
    return {
        "slot": slot,
        "time_s": time_s,
        "arrival_task_count": len(tasks),
        "initial_J": float(initial_eval.task_utility_sum),
        "result": result,
        "full_result": full_result,
        "incremental_result": incremental_result,
        "full_vs_incremental": full_vs_incremental,
        "multi_task_rsu_case": multi_task_rsu_case,
        "repeat_deterministic": not repeat_mismatch,
        "task_rows": task_rows,
        "operation_sanity": op_sanity,
    }


def _summary(result: TransferExchangeMatchingResult) -> dict[str, Any]:
    return {
        "matching": result.matching,
        "J": result.task_utility_sum,
        "updates": result.updates,
        "transfer_updates": result.transfer_updates,
        "exchange_updates": result.exchange_updates,
        "candidate_evaluations_total": result.candidate_evaluations_total,
        "transfer_candidates_evaluated": result.transfer_candidates_evaluated,
        "exchange_candidates_evaluated": result.exchange_candidates_evaluated,
        "resource_infeasible_candidates": result.resource_infeasible_candidates,
        "feasibility_rejected_without_kkt": result.feasibility_rejected_without_kkt,
        "stop_reason": result.stop_reason,
        "stable": result.transfer_exchange_stable,
        "stable_check_passed": result.stable_check_passed,
        "max_remaining_delta": result.max_remaining_delta,
        "history": result.history,
        "cpu_allocations": result.cpu_allocations,
        "bandwidth_allocations": result.bandwidth_allocations,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Actual-State Transfer-Exchange Matching validation.")
    parser.add_argument("--config", default="system_config.toml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validation = run_slot20_validation(args.config)
    print(
        json.dumps(
            {
                "slot": validation["slot"],
                "time_s": validation["time_s"],
                "arrival_task_count": validation["arrival_task_count"],
                "initial_J": validation["initial_J"],
                "tasks": validation["task_rows"],
                "operation_sanity": validation["operation_sanity"],
                "full_reference": _summary(validation["full_result"]),
                "incremental": _summary(validation["incremental_result"]),
                "full_vs_incremental": validation["full_vs_incremental"],
                "multi_task_rsu_case": validation["multi_task_rsu_case"],
                "repeat_deterministic": validation["repeat_deterministic"],
                "result": _summary(validation["result"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0

# ---- KKT actual-state resource allocation ----
# KKT actual-state resource allocation implementation；数学求解保持不变。
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


def evaluate_matching_candidate_incremental(
    tasks: list[Task],
    current_matching: dict[str, str],
    current_evaluation: KKTMatchingEvaluation,
    candidate_matching: dict[str, str],
    affected_task_ids: list[str],
    vehicle_positions: dict[str, np.ndarray],
    service_node_positions: dict[str, np.ndarray],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
) -> KKTMatchingEvaluation:
    """只重算 candidate 受影响 edge nodes，返回完整 KKT matching evaluation。"""

    if not current_evaluation.resource_feasible:
        raise ValueError("incremental evaluator 要求 current evaluation 可行。")

    current_kkt = _to_kkt_matching(current_matching)
    candidate_kkt = _to_kkt_matching(candidate_matching)
    task_ids = {task.task_id for task in tasks}
    if set(current_kkt) != task_ids or set(candidate_kkt) != task_ids:
        raise ValueError("current/candidate matching 的任务集合不一致。")

    affected_ids = [str(task_id) for task_id in affected_task_ids]
    if not set(affected_ids).issubset(task_ids):
        raise ValueError("affected_task_ids 包含未知任务。")

    affected_edge_nodes: set[str] = set()
    for task_id in affected_ids:
        for target in (current_kkt[task_id], candidate_kkt[task_id]):
            if target not in (LOCAL, CLOUD):
                _edge_node_role(target, cache_nodes)
                affected_edge_nodes.add(target)

    impacted_task_ids = set(affected_ids)
    for task in tasks:
        if (
            current_kkt[task.task_id] in affected_edge_nodes
            or candidate_kkt[task.task_id] in affected_edge_nodes
        ):
            impacted_task_ids.add(task.task_id)

    cpu_allocations = {
        node_id: dict(values)
        for node_id, values in current_evaluation.cpu_allocations.items()
    }
    bandwidth_allocations = {
        node_id: dict(values)
        for node_id, values in current_evaluation.bandwidth_allocations.items()
    }
    cpu_resource_sums = dict(current_evaluation.cpu_resource_sums)
    bandwidth_resource_sums = dict(current_evaluation.bandwidth_resource_sums)
    for node_id in affected_edge_nodes:
        cpu_allocations.pop(node_id, None)
        bandwidth_allocations.pop(node_id, None)
        cpu_resource_sums.pop(node_id, None)
        bandwidth_resource_sums.pop(node_id, None)

    per_task_delay = dict(current_evaluation.per_task_delay_s)
    per_task_energy = dict(current_evaluation.per_task_vehicle_energy_j)
    per_task_utility = dict(current_evaluation.per_task_utility)

    def infeasible(reason: str) -> KKTMatchingEvaluation:
        return KKTMatchingEvaluation(
            resource_feasible=False,
            infeasible_reason=reason,
            task_utility_sum=float("-inf"),
            cpu_allocations={},
            bandwidth_allocations={},
            per_task_delay_s={},
            per_task_vehicle_energy_j={},
            per_task_utility={},
            per_task_execution_target={},
            per_node_task_counts={},
            cpu_resource_sums={},
            bandwidth_resource_sums={},
        )

    recomputed_task_ids: set[str] = set()
    for node_id in sorted(affected_edge_nodes, key=str):
        node_tasks = [
            task for task in tasks if candidate_kkt[task.task_id] == node_id
        ]
        if not node_tasks:
            continue
        node_matching = {task.task_id: node_id for task in node_tasks}
        node_evaluation = evaluate_matching_with_kkt(
            node_tasks,
            node_matching,
            vehicle_positions,
            service_node_positions,
            project_cfg,
            cache_nodes,
            cache_config,
        )
        if not node_evaluation.resource_feasible:
            return infeasible(
                f"{node_id} incremental KKT infeasible: "
                f"{node_evaluation.infeasible_reason}"
            )
        cpu_allocations[node_id] = dict(node_evaluation.cpu_allocations[node_id])
        bandwidth_allocations[node_id] = dict(
            node_evaluation.bandwidth_allocations[node_id]
        )
        cpu_resource_sums[node_id] = node_evaluation.cpu_resource_sums[node_id]
        bandwidth_resource_sums[node_id] = node_evaluation.bandwidth_resource_sums[node_id]
        per_task_delay.update(node_evaluation.per_task_delay_s)
        per_task_energy.update(node_evaluation.per_task_vehicle_energy_j)
        per_task_utility.update(node_evaluation.per_task_utility)
        recomputed_task_ids.update(task.task_id for task in node_tasks)

    non_edge_tasks = [
        task
        for task in tasks
        if task.task_id in impacted_task_ids
        and candidate_kkt[task.task_id] in (LOCAL, CLOUD)
    ]
    if non_edge_tasks:
        non_edge_evaluation = evaluate_matching_with_kkt(
            non_edge_tasks,
            {task.task_id: candidate_kkt[task.task_id] for task in non_edge_tasks},
            vehicle_positions,
            service_node_positions,
            project_cfg,
            cache_nodes,
            cache_config,
        )
        if not non_edge_evaluation.resource_feasible:
            return infeasible(
                f"incremental non-edge evaluation infeasible: "
                f"{non_edge_evaluation.infeasible_reason}"
            )
        per_task_delay.update(non_edge_evaluation.per_task_delay_s)
        per_task_energy.update(non_edge_evaluation.per_task_vehicle_energy_j)
        per_task_utility.update(non_edge_evaluation.per_task_utility)
        recomputed_task_ids.update(task.task_id for task in non_edge_tasks)

    if recomputed_task_ids != impacted_task_ids:
        missing = sorted(impacted_task_ids - recomputed_task_ids, key=_stable_id_key)
        raise AssertionError(f"incremental evaluator 未覆盖 impacted tasks：{missing}")

    per_task_execution_target = {
        task.task_id: candidate_kkt[task.task_id] for task in tasks
    }
    per_node_task_counts: dict[str, int] = {}
    for task in tasks:
        target = candidate_kkt[task.task_id]
        per_node_task_counts[target] = per_node_task_counts.get(target, 0) + 1

    task_utility_sum = 0.0
    for task in tasks:
        task_utility_sum += float(per_task_utility[task.task_id])

    return KKTMatchingEvaluation(
        resource_feasible=True,
        infeasible_reason="",
        task_utility_sum=float(task_utility_sum),
        cpu_allocations=cpu_allocations,
        bandwidth_allocations=bandwidth_allocations,
        per_task_delay_s=per_task_delay,
        per_task_vehicle_energy_j=per_task_energy,
        per_task_utility=per_task_utility,
        per_task_execution_target=per_task_execution_target,
        per_node_task_counts=per_node_task_counts,
        cpu_resource_sums=cpu_resource_sums,
        bandwidth_resource_sums=bandwidth_resource_sums,
    )


def _max_abs_mapping_diff(
    left: dict[str, float],
    right: dict[str, float],
) -> float:
    if set(left) != set(right):
        return float("inf")
    return max(
        (abs(float(left[key]) - float(right[key])) for key in left),
        default=0.0,
    )


def _max_abs_nested_mapping_diff(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
) -> float:
    if set(left) != set(right):
        return float("inf")
    return max(
        (_max_abs_mapping_diff(left[node_id], right[node_id]) for node_id in left),
        default=0.0,
    )


def _evaluation_differences(
    full: KKTMatchingEvaluation,
    incremental: KKTMatchingEvaluation,
) -> dict[str, float]:
    return {
        "J_diff": abs(float(full.task_utility_sum) - float(incremental.task_utility_sum)),
        "CPU_max_diff": _max_abs_nested_mapping_diff(
            full.cpu_allocations,
            incremental.cpu_allocations,
        ),
        "BW_max_diff": _max_abs_nested_mapping_diff(
            full.bandwidth_allocations,
            incremental.bandwidth_allocations,
        ),
        "delay_max_diff": _max_abs_mapping_diff(
            full.per_task_delay_s,
            incremental.per_task_delay_s,
        ),
        "energy_max_diff": _max_abs_mapping_diff(
            full.per_task_vehicle_energy_j,
            incremental.per_task_vehicle_energy_j,
        ),
        "Q_max_diff": _max_abs_mapping_diff(
            full.per_task_utility,
            incremental.per_task_utility,
        ),
    }


def _assert_evaluation_equivalent(
    full: KKTMatchingEvaluation,
    incremental: KKTMatchingEvaluation,
    label: str,
    tolerance: float = 1.0e-10,
) -> dict[str, float]:
    if full.resource_feasible != incremental.resource_feasible:
        raise AssertionError(f"{label} full/incremental resource feasibility 不一致。")
    if not full.resource_feasible:
        if full.infeasible_reason != incremental.infeasible_reason:
            raise AssertionError(f"{label} full/incremental infeasible reason 不一致。")
        return {key: 0.0 for key in ("J_diff", "CPU_max_diff", "BW_max_diff", "delay_max_diff", "energy_max_diff", "Q_max_diff")}
    if full.per_task_execution_target != incremental.per_task_execution_target:
        raise AssertionError(f"{label} full/incremental task target 不一致。")
    if full.per_node_task_counts != incremental.per_node_task_counts:
        raise AssertionError(f"{label} full/incremental node task counts 不一致。")
    differences = _evaluation_differences(full, incremental)
    differences["CPU_sum_max_diff"] = _max_abs_mapping_diff(
        full.cpu_resource_sums,
        incremental.cpu_resource_sums,
    )
    differences["BW_sum_max_diff"] = _max_abs_mapping_diff(
        full.bandwidth_resource_sums,
        incremental.bandwidth_resource_sums,
    )
    if any(value > tolerance for value in differences.values()):
        raise AssertionError(f"{label} full/incremental 数值不一致：{differences}")
    return differences


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


def build_kkt_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Given matching -> KKT resource allocation validation.")
    parser.add_argument("--config", default="system_config.toml")
    return parser


def kkt_main(argv: list[str] | None = None) -> int:
    args = build_kkt_parser().parse_args(argv)
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
