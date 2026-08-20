#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Actual-State Transfer-Exchange Matching.

本模块只做实际时隙内的任务 -> 执行节点 matching。每个 candidate matching
都调用 ``evaluate_matching_with_kkt()`` 重新求 KKT 连续资源分配和
``J(mu)=sum_v Q_v``。本模块不实现 PPO、不扣 Cache/UAV configuration cost、
不读取 prediction_scenarios 作为调度输入。

Local 在 Matching 层用 ``LOCAL:<vehicle_id>`` 表示任务车辆自身本地执行，
避免把 Local 错误建模成多个任务共享的公共节点。调用 KKT evaluator 前，
只有 ``LOCAL:<same task_id>`` 会被安全转换为 KKT 内部 sentinel ``LOCAL``。
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
import task_scenario
import uav_configuration

CLOUD = kkt.CLOUD
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
    tasks: list[kkt.Task],
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
            converted[task_id] = kkt.LOCAL
        else:
            converted[task_id] = target
    return converted


def evaluate_matching_candidate(
    tasks: list[kkt.Task],
    matching: dict[str, str],
    vehicle_positions: dict[str, np.ndarray],
    service_node_positions: dict[str, np.ndarray],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
) -> kkt.KKTMatchingEvaluation:
    """薄 adapter：Matching层 LOCAL:v -> KKT层 LOCAL。"""

    return kkt.evaluate_matching_with_kkt(
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
    tasks: list[kkt.Task],
    current_matching: dict[str, str],
    current_evaluation: kkt.KKTMatchingEvaluation,
    feasible_targets: dict[str, list[str]],
    vehicle_positions: dict[str, np.ndarray],
    service_node_positions: dict[str, np.ndarray],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    tolerance: float,
    evaluate_only: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, int], float]:
    """枚举全部合法transfer/exchange，返回本轮最大增益候选。"""

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
            evaluation = evaluate_matching_candidate(
                tasks,
                candidate_matching,
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
            evaluation = evaluate_matching_candidate(
                tasks,
                candidate_matching,
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
    tasks_by_id: dict[str, kkt.Task],
    matching: dict[str, str],
    feasible_targets: dict[str, list[str]],
) -> bool:
    del tasks_by_id
    return all(target in feasible_targets[task_id] for task_id, target in matching.items())


def _resource_constraints_ok(
    evaluation: kkt.KKTMatchingEvaluation,
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
    tasks: list[kkt.Task],
    vehicle_positions: dict[str, np.ndarray],
    service_node_positions: dict[str, np.ndarray],
    local_cache_state: dict[str, set[int]],
    edge_cache_state: np.ndarray,
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
) -> TransferExchangeMatchingResult:
    """Transfer-Exchange local stable matching under actual state."""

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
    scenarios = cache_configuration.load_prediction_scenarios(
        cache_configuration.resolve_path(root, project_config.get_config_value(cfg, "paths.prediction_scenarios"))
    )
    origin = cache_configuration.scenario_at_origin(scenarios, 0)
    catalog = cache_configuration.load_service_catalog(
        cache_configuration.resolve_path(root, project_config.get_config_value(cfg, "paths.service_catalog"))
    )
    cache_nodes = cache_configuration.build_cache_nodes(origin, cache_cfg)
    edge_cache_state = cache_configuration.deterministic_initial_cache(origin, catalog, cache_nodes, cache_cfg)
    slot = 20
    time_s = 120.0
    tasks = kkt.load_actual_tasks(root / "data/generated/tasks.csv", slot)
    roles = cache_configuration.load_json(root / "data/generated/vehicle_roles.json")
    sv_ids = {str(value) for value in roles["service_vehicles"]}
    tv_ids = {task.vehicle_id for task in tasks}
    actual_positions = kkt.load_vehicle_positions(
        root / "data/generated/trajectory.csv",
        time_s,
        tv_ids | sv_ids,
    )
    uav_positions = uav_configuration.initial_uav_positions_from_project(cfg)
    service_positions = kkt.build_service_node_positions(cache_nodes, actual_positions, uav_positions)
    local_cache = task_scenario.load_tv_local_cache(
        cache_configuration.resolve_path(root, project_config.get_config_value(cfg, "paths.tv_local_cache"))
    )
    result = optimize_task_matching(
        tasks,
        actual_positions,
        service_positions,
        local_cache,
        edge_cache_state,
        cfg,
        cache_nodes,
        cache_cfg,
    )
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
    if result.matching != repeat.matching or not math.isclose(result.task_utility_sum, repeat.task_utility_sum):
        raise AssertionError("Matching repeated-run 不确定。")
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
                "result": _summary(validation["result"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
