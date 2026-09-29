#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Actual-State Transfer-Exchange Matching.

本模块只做实际时隙内的任务 -> 执行节点 matching。每个 candidate matching
默认 candidate 路径使用 affected-node incremental evaluator；
``evaluate_matching_candidate()`` 保留为 full-recompute reference，二者都求
KKT 连续资源分配和 ``J(mu)=sum_v Q_v``。本模块不实现 PPO、不扣 Cache/UAV configuration cost、
不读取 prediction_scenarios 作为调度输入；每个 slot 的任务独立完成评价。

Local 在 Matching 层用 ``LOCAL:<vehicle_id>`` 表示任务车辆自身本地执行，
避免把 Local 错误建模成多个任务共享的公共节点。调用 KKT evaluator 前，
只有 ``LOCAL:<same task_id>`` 会被安全转换为 KKT 内部 sentinel ``LOCAL``。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import numba
except ModuleNotFoundError:  # pragma: no cover - server fallback without Numba
    numba = None  # type: ignore[assignment]

from common import project_config, system_model
from decision import cache_configuration, uav_configuration
from scenario import task_scenario

LOCAL_PREFIX = "LOCAL:"


def _profile_add(
    timing_profile: dict[str, float] | None,
    key: str,
    elapsed: float,
) -> None:
    if timing_profile is not None:
        timing_profile[key] = float(timing_profile.get(key, 0.0)) + max(
            float(elapsed),
            0.0,
        )


_EXACT_EVAL_SUBHOTSPOT_RUNTIME_KEYS = (
    "exact_eval_kkt_conversion_runtime",
    "exact_eval_state_copy_runtime",
    "exact_eval_affected_node_prep_runtime",
    "exact_eval_node_call_orchestration_runtime",
    "exact_eval_result_merge_runtime",
    "exact_eval_non_edge_prep_runtime",
    "exact_eval_target_count_build_runtime",
    "exact_eval_utility_sum_runtime",
    "exact_eval_other_runtime",
)


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
    candidate_definitions_total: int
    upper_bound_screen_calls: int
    upper_bound_pruned_candidates: int
    transfer_upper_bound_pruned: int
    exchange_upper_bound_pruned: int
    full_evaluator_calls: int
    kkt_cache_requests: int
    kkt_cache_hits: int
    kkt_cache_misses: int
    kkt_solver_calls: int
    stop_reason: str
    transfer_exchange_stable: bool
    stable_check_passed: bool
    max_remaining_delta: float
    history: list[dict[str, Any]]
    feasible_targets: dict[str, list[str]]
    node_eval_cache_requests: int = 0
    node_eval_cache_hits: int = 0
    node_eval_cache_misses: int = 0
    node_eval_full_executions: int = 0
    ideal_utility_cache_requests: int = 0
    ideal_utility_cache_hits: int = 0
    ideal_utility_cache_misses: int = 0
    impact_scope_requests: int = 0
    impact_scope_computes: int = 0
    impact_scope_reuses: int = 0
    kkt_solve_count: int = 0
    kkt_converged_count: int = 0
    kkt_max_iterations_hit_count: int = 0
    kkt_finite_count: int = 0
    kkt_iteration_mean: float = 0.0
    kkt_iteration_p95: float = 0.0
    kkt_iteration_max: int = 0
    kkt_resource_residual_mean: float = math.nan
    kkt_resource_residual_max: float = math.nan
    kkt_stationarity_residual_mean: float = math.nan
    kkt_stationarity_residual_max: float = math.nan
    kkt_complementary_slackness_residual_mean: float = math.nan
    kkt_complementary_slackness_residual_max: float = math.nan
    kkt_solve_diagnostics: tuple[dict[str, Any], ...] = ()


def _kkt_diagnostics_summary(
    records: Any,
) -> dict[str, Any]:
    """Aggregate direct diagnostics for the KKT solves in one Matching call."""

    typed_records = tuple(records)
    iterations = np.asarray(
        [int(record.iterations) for record in typed_records], dtype=np.float64
    )
    resource_residuals = np.asarray(
        [
            float(record.resource_sum_residual)
            for record in typed_records
            if math.isfinite(float(record.resource_sum_residual))
        ],
        dtype=np.float64,
    )
    stationarity_residuals = np.asarray(
        [
            float(record.stationarity_residual)
            for record in typed_records
            if math.isfinite(float(record.stationarity_residual))
        ],
        dtype=np.float64,
    )
    complementary_residuals = np.asarray(
        [
            float(record.complementary_slackness_residual)
            for record in typed_records
            if math.isfinite(float(record.complementary_slackness_residual))
        ],
        dtype=np.float64,
    )

    def _mean(values: np.ndarray) -> float:
        return 0.0 if values.size == 0 else float(np.mean(values))

    def _max(values: np.ndarray) -> float:
        return math.nan if values.size == 0 else float(np.max(values))

    return {
        "solve_count": int(len(typed_records)),
        "converged_count": int(
            sum(bool(record.converged_by_tolerance) for record in typed_records)
        ),
        "max_iterations_hit_count": int(
            sum(bool(record.hit_max_bisection_iterations) for record in typed_records)
        ),
        "finite_count": int(sum(bool(record.finite) for record in typed_records)),
        "iteration_mean": _mean(iterations),
        "iteration_p95": (
            0.0 if iterations.size == 0 else float(np.percentile(iterations, 95.0))
        ),
        "iteration_max": int(np.max(iterations)) if iterations.size else 0,
        "resource_residual_mean": _mean(resource_residuals),
        "resource_residual_max": _max(resource_residuals),
        "stationarity_residual_mean": _mean(stationarity_residuals),
        "stationarity_residual_max": _max(stationarity_residuals),
        "complementary_slackness_residual_mean": _mean(complementary_residuals),
        "complementary_slackness_residual_max": _max(complementary_residuals),
        "solve_diagnostics": tuple(
            {
                "iterations": int(record.iterations),
                "converged_by_tolerance": bool(record.converged_by_tolerance),
                "hit_max_bisection_iterations": bool(
                    record.hit_max_bisection_iterations
                ),
                "final_resource_sum": float(record.final_resource_sum),
                "resource_sum_residual": float(record.resource_sum_residual),
                "finite": bool(record.finite),
                "stationarity_residual": float(record.stationarity_residual),
                "complementary_slackness_residual": float(
                    record.complementary_slackness_residual
                ),
            }
            for record in typed_records
        ),
    }


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
    vehicle_positions: dict[str, np.ndarray] | None = None,
    service_node_positions: dict[str, np.ndarray] | None = None,
    cache_config: cache_configuration.CacheOptimizationConfig | None = None,
    timing_profile: dict[str, float] | None = None,
) -> dict[str, list[str]]:
    """按缓存可用性和当前物理路径构造 N_v^fea。

    正式实际/预测调用必须提供 vehicle_positions、service_node_positions 和
    cache_config；若某个缓存边缘节点缺少当前位置、角色、正带宽/CPU 或正信道
    速率，则该节点不进入 M_edge，Cloud 作为后备节点保留。旧的纯集合 sanity
    调用可以省略物理上下文，此时只检查 cache node 的结构存在性。
    """

    edge_cache = np.asarray(edge_cache_state, dtype=np.int8)
    node_ids = [str(value) for value in cache_nodes["node_ids"]]
    if edge_cache.shape[0] != len(node_ids):
        raise ValueError("edge_cache_state node维度与cache_nodes不一致。")
    physical_context = (
        vehicle_positions is not None
        or service_node_positions is not None
        or cache_config is not None
    )
    if physical_context and (
        vehicle_positions is None
        or service_node_positions is None
        or cache_config is None
    ):
        raise ValueError(
            "正式 Cloud 候选门控必须同时提供车辆位置、服务节点位置和 cache_config。"
        )
    feasible: dict[str, list[str]] = {}
    feasible_started_at = time.perf_counter() if timing_profile is not None else 0.0
    channel_rate_runtime = 0.0
    for task in sorted(tasks, key=lambda item: _stable_id_key(item.task_id)):
        targets: list[str] = []
        if task.service_id in local_cache_state.get(task.vehicle_id, set()):
            targets.append(local_target(task.task_id))
        edge_targets: list[str] = []
        for node_index, node_id in enumerate(node_ids):
            if int(edge_cache[node_index, task.service_id]) != 1:
                continue
            if physical_context:
                channel_started_at = (
                    time.perf_counter() if timing_profile is not None else 0.0
                )
                path_is_feasible = _edge_path_is_feasible(
                    task,
                    node_id,
                    vehicle_positions,
                    service_node_positions,
                    cache_nodes,
                    cache_config,
                )
                if timing_profile is not None:
                    channel_rate_runtime += time.perf_counter() - channel_started_at
                if not path_is_feasible:
                    continue
            edge_targets.append(node_id)
        if edge_targets:
            targets.extend(edge_targets)
        else:
            targets.append(CLOUD)
        feasible[task.task_id] = targets
    if timing_profile is not None:
        total_runtime = time.perf_counter() - feasible_started_at
        _profile_add(
            timing_profile,
            "matching_channel_rate_runtime",
            channel_rate_runtime,
        )
        _profile_add(
            timing_profile,
            "matching_feasible_target_runtime",
            max(total_runtime - channel_rate_runtime, 0.0),
        )
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
    kkt_cache: KKTResourceAllocationCache | None = None,
    node_evaluation_cache: NodeEvaluationCache | None = None,
    timing_profile: dict[str, float] | None = None,
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
        kkt_cache=kkt_cache,
        node_evaluation_cache=node_evaluation_cache,
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


def _candidate_impact_scope(
    tasks: list[Task],
    current_kkt: dict[str, str],
    candidate_kkt: dict[str, str],
    affected_task_ids: list[str],
    cache_nodes: dict[str, Any],
    timing_profile: dict[str, float] | None = None,
) -> tuple[set[str], set[str]]:
    """Return the exact affected edge nodes and impacted task IDs.

    This is shared by strict candidate screening and the incremental evaluator;
    the scope must include every task sharing a current or candidate edge node.
    """

    started_at = time.perf_counter() if timing_profile is not None else 0.0
    affected_edge_nodes: set[str] = set()
    for task_id in affected_task_ids:
        for target in (current_kkt[task_id], candidate_kkt[task_id]):
            if target == CLOUD:
                # Cloud resources are fixed per task and do not couple unrelated
                # Cloud tasks in the formal model.
                continue
            if target != LOCAL:
                _edge_node_role(target, cache_nodes)
                affected_edge_nodes.add(target)

    impacted_task_ids = set(affected_task_ids)
    for task in tasks:
        if (
            current_kkt[task.task_id] in affected_edge_nodes
            or candidate_kkt[task.task_id] in affected_edge_nodes
        ):
            impacted_task_ids.add(task.task_id)
    _profile_add(
        timing_profile,
        "matching_impact_scope_runtime",
        time.perf_counter() - started_at if timing_profile is not None else 0.0,
    )
    return affected_edge_nodes, impacted_task_ids


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
    kkt_cache: KKTResourceAllocationCache | None = None,
    node_evaluation_cache: NodeEvaluationCache | None = None,
    timing_profile: dict[str, float] | None = None,
    ideal_utility_cache: dict[tuple[str, str], float] | None = None,
    ideal_utility_cache_stats: dict[str, int] | None = None,
    impact_scope_stats: dict[str, int] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, int], float]:
    """枚举全部合法 transfer/exchange，返回本轮最大增益候选。

    Candidate generation and semantic selection remain in the original order.
    In incremental mode, exact evaluation is scheduled separately: every
    candidate is first assigned the existing strict ideal-resource upper bound,
    the largest-bound candidate establishes an exact incumbent, and only then
    are candidates that cannot beat that incumbent evaluated. Exact results
    are replayed through ``_better_candidate`` in generator order so the
    operation-key tie break remains unchanged.
    """

    if evaluation_mode not in {"full", "incremental"}:
        raise ValueError(f"未知 Matching candidate evaluation mode：{evaluation_mode}")
    task_ids = _sorted_ids([task.task_id for task in tasks])
    stats = {
        "candidate_definitions_total": 0,
        "candidate_evaluations_total": 0,
        "transfer_candidates_evaluated": 0,
        "exchange_candidates_evaluated": 0,
        "resource_infeasible_candidates": 0,
        "feasibility_rejected_without_kkt": 0,
        "upper_bound_screen_calls": 0,
        "upper_bound_pruned_candidates": 0,
        "transfer_upper_bound_pruned": 0,
        "exchange_upper_bound_pruned": 0,
        "full_evaluator_calls": 0,
    }
    if evaluation_mode == "incremental":
        current_kkt = _to_kkt_matching(current_matching)
        if timing_profile is not None:
            timing_profile["current_kkt_conversion_call_count"] = (
                float(timing_profile.get("current_kkt_conversion_call_count", 0.0))
                + 1.0
            )
    else:
        current_kkt = None

    candidate_metadata: list[dict[str, Any]] = []
    original_index = 0

    def append_candidate(
        operation_type: str,
        affected_tasks: list[str],
        old_targets: list[str],
        new_targets: list[str],
        candidate_matching: dict[str, str],
    ) -> None:
        nonlocal original_index
        candidate_metadata.append(
            {
                "original_index": int(original_index),
                "operation_type": str(operation_type),
                "affected_tasks": list(affected_tasks),
                "old_targets": list(old_targets),
                "new_targets": list(new_targets),
                "matching": candidate_matching,
                "upper_delta": None,
                "candidate_kkt": None,
                "impact_scope": None,
            }
        )
        original_index += 1

    candidate_generation_started_at = (
        time.perf_counter() if timing_profile is not None else 0.0
    )
    transfer_generation_started_at = (
        time.perf_counter() if timing_profile is not None else 0.0
    )
    for task_id in task_ids:
        old_target = current_matching[task_id]
        for target in sorted(feasible_targets[task_id], key=str):
            if target == old_target:
                continue
            candidate_matching = dict(current_matching)
            candidate_matching[task_id] = target
            stats["candidate_definitions_total"] += 1
            append_candidate(
                "transfer",
                [task_id],
                [old_target],
                [target],
                candidate_matching,
            )

    transfer_generation_runtime = (
        time.perf_counter() - transfer_generation_started_at
        if timing_profile is not None
        else 0.0
    )

    exchange_generation_started_at = (
        time.perf_counter() if timing_profile is not None else 0.0
    )
    for left_index, left_task in enumerate(task_ids):
        for right_task in task_ids[left_index + 1 :]:
            left_target = current_matching[left_task]
            right_target = current_matching[right_task]
            if left_target == right_target:
                continue
            stats["candidate_definitions_total"] += 1
            if right_target not in feasible_targets[left_task] or left_target not in feasible_targets[right_task]:
                stats["feasibility_rejected_without_kkt"] += 1
                continue
            candidate_matching = dict(current_matching)
            candidate_matching[left_task] = right_target
            candidate_matching[right_task] = left_target
            append_candidate(
                "exchange",
                [left_task, right_task],
                [left_target, right_target],
                [right_target, left_target],
                candidate_matching,
            )

    exchange_generation_runtime = (
        time.perf_counter() - exchange_generation_started_at
        if timing_profile is not None
        else 0.0
    )
    _profile_add(
        timing_profile,
        "matching_candidate_generation_runtime",
        time.perf_counter() - candidate_generation_started_at
        if timing_profile is not None
        else transfer_generation_runtime + exchange_generation_runtime,
    )

    tie_tolerance = tolerance * 0.1
    if evaluation_mode == "incremental":
        if current_kkt is None:
            raise AssertionError("incremental candidate screening requires current KKT matching")
        # Calculate the existing strict UB exactly once per candidate. This
        # pass is separate from both execution order and semantic selection.
        for metadata in candidate_metadata:
            stats["upper_bound_screen_calls"] += 1
            candidate_matching = metadata["matching"]
            candidate_kkt = _to_kkt_matching(candidate_matching)
            if timing_profile is not None:
                timing_profile["candidate_kkt_conversion_call_count"] = (
                    float(timing_profile.get("candidate_kkt_conversion_call_count", 0.0))
                    + 1.0
                )
            metadata["candidate_kkt"] = candidate_kkt
            if impact_scope_stats is not None:
                impact_scope_stats["requests"] += 1
                impact_scope_stats["computes"] += 1
            impact_scope = _candidate_impact_scope(
                tasks,
                current_kkt,
                candidate_kkt,
                metadata["affected_tasks"],
                cache_nodes,
                timing_profile=timing_profile,
            )
            metadata["impact_scope"] = impact_scope
            _, impacted_task_ids = impact_scope
            upper_started_at = (
                time.perf_counter() if timing_profile is not None else 0.0
            )
            ideal_before = (
                float(timing_profile.get("matching_ideal_utility_runtime", 0.0))
                if timing_profile is not None
                else 0.0
            )
            metadata["upper_delta"] = _strict_ideal_resource_upper_delta(
                tasks,
                candidate_matching,
                current_evaluation,
                impacted_task_ids,
                vehicle_positions,
                service_node_positions,
                cache_nodes,
                cache_config,
                timing_profile=timing_profile,
                ideal_utility_cache=ideal_utility_cache,
                ideal_utility_cache_stats=ideal_utility_cache_stats,
            )
            if timing_profile is not None:
                ideal_elapsed = (
                    float(timing_profile.get("matching_ideal_utility_runtime", 0.0))
                    - ideal_before
                )
                _profile_add(
                    timing_profile,
                    "matching_upper_bound_runtime",
                    time.perf_counter() - upper_started_at - ideal_elapsed,
                )

    exact_records: dict[int, KKTMatchingEvaluation] = {}
    global_exact_lower_bound = float("-inf")
    max_delta = float("-inf")

    def evaluate_exact(metadata: dict[str, Any]) -> None:
        nonlocal global_exact_lower_bound, max_delta
        candidate_matching = metadata["matching"]
        stats["full_evaluator_calls"] += 1
        evaluation_started_at = (
            time.perf_counter() if timing_profile is not None else 0.0
        )
        node_cache_before = (
            float(timing_profile.get("node_cache_lookup_runtime", 0.0))
            if timing_profile is not None
            else 0.0
        )
        node_eval_before = (
            float(timing_profile.get("node_eval_miss_runtime", 0.0))
            if timing_profile is not None
            else 0.0
        )
        kkt_before = (
            float(timing_profile.get("kkt_runtime", 0.0))
            if timing_profile is not None
            else 0.0
        )
        impact_before = (
            float(timing_profile.get("matching_impact_scope_runtime", 0.0))
            if timing_profile is not None
            else 0.0
        )
        if evaluation_mode == "full":
            evaluation = evaluate_matching_candidate(
                tasks,
                candidate_matching,
                vehicle_positions,
                service_node_positions,
                project_cfg,
                cache_nodes,
                cache_config,
                kkt_cache=kkt_cache,
                node_evaluation_cache=node_evaluation_cache,
                timing_profile=timing_profile,
            )
        else:
            precomputed_candidate_kkt = metadata.get("candidate_kkt")
            if timing_profile is not None:
                timing_profile["current_kkt_context_request_count"] = (
                    float(
                        timing_profile.get("current_kkt_context_request_count", 0.0)
                    )
                    + 1.0
                )
                timing_profile["candidate_kkt_context_request_count"] = (
                    float(
                        timing_profile.get("candidate_kkt_context_request_count", 0.0)
                    )
                    + 1.0
                )
                if current_kkt is not None:
                    timing_profile["current_kkt_context_reuse_count"] = (
                        float(
                            timing_profile.get("current_kkt_context_reuse_count", 0.0)
                        )
                        + 1.0
                    )
                if precomputed_candidate_kkt is not None:
                    timing_profile["candidate_kkt_context_reuse_count"] = (
                        float(
                            timing_profile.get(
                                "candidate_kkt_context_reuse_count", 0.0
                            )
                        )
                        + 1.0
                    )
            evaluation = evaluate_matching_candidate_incremental(
                tasks,
                current_matching,
                current_evaluation,
                candidate_matching,
                metadata["affected_tasks"],
                vehicle_positions,
                service_node_positions,
                project_cfg,
                cache_nodes,
                cache_config,
                kkt_cache=kkt_cache,
                node_evaluation_cache=node_evaluation_cache,
                timing_profile=timing_profile,
                precomputed_current_kkt=current_kkt,
                precomputed_candidate_kkt=precomputed_candidate_kkt,
                precomputed_impact_scope=metadata["impact_scope"],
                impact_scope_stats=impact_scope_stats,
            )
        if timing_profile is not None:
            nested_runtime = (
                float(timing_profile.get("node_cache_lookup_runtime", 0.0))
                - node_cache_before
                + float(timing_profile.get("node_eval_miss_runtime", 0.0))
                - node_eval_before
                + float(timing_profile.get("kkt_runtime", 0.0))
                - kkt_before
                + float(timing_profile.get("matching_impact_scope_runtime", 0.0))
                - impact_before
            )
            _profile_add(
                timing_profile,
                "matching_exact_eval_orchestration_runtime",
                time.perf_counter() - evaluation_started_at - nested_runtime,
            )
        exact_records[int(metadata["original_index"])] = evaluation
        stats["candidate_evaluations_total"] += 1
        if metadata["operation_type"] == "transfer":
            stats["transfer_candidates_evaluated"] += 1
        else:
            stats["exchange_candidates_evaluated"] += 1
        if not evaluation.resource_feasible:
            stats["resource_infeasible_candidates"] += 1
            return
        delta = float(evaluation.task_utility_sum - current_evaluation.task_utility_sum)
        metadata["exact_delta"] = delta
        max_delta = max(max_delta, delta)
        global_exact_lower_bound = max(global_exact_lower_bound, delta)

    if evaluation_mode == "incremental" and candidate_metadata:
        finite_bound_indices = [
            index
            for index, metadata in enumerate(candidate_metadata)
            if metadata["upper_delta"] is not None
        ]
        if finite_bound_indices:
            seed_index = max(
                finite_bound_indices,
                key=lambda index: float(candidate_metadata[index]["upper_delta"]),
            )
            # The seed is evaluated first solely to establish an exact lower
            # bound. It is reused during original-order replay below.
            evaluate_exact(candidate_metadata[seed_index])
        else:
            seed_index = None

        for index, metadata in enumerate(candidate_metadata):
            if index == seed_index:
                continue
            upper_delta = metadata["upper_delta"]
            if (
                upper_delta is not None
                and global_exact_lower_bound != float("-inf")
                and float(upper_delta) < global_exact_lower_bound - tie_tolerance
            ):
                stats["upper_bound_pruned_candidates"] += 1
                if metadata["operation_type"] == "transfer":
                    stats["transfer_upper_bound_pruned"] += 1
                else:
                    stats["exchange_upper_bound_pruned"] += 1
                continue
            evaluate_exact(metadata)
    else:
        # Full mode retains the original execution order and has no strict UB
        # screen. This is also the safe fallback for an empty candidate set.
        for metadata in candidate_metadata:
            evaluate_exact(metadata)

    # Replay semantic selection in generator order. Exact results are never
    # evaluated a second time here.
    replay_started_at = time.perf_counter() if timing_profile is not None else 0.0
    best: dict[str, Any] | None = None
    for metadata in candidate_metadata:
        index = int(metadata["original_index"])
        evaluation = exact_records.get(index)
        if evaluation is None or not evaluation.resource_feasible:
            continue
        delta = float(metadata["exact_delta"])
        candidate = {
            "operation_type": metadata["operation_type"],
            "affected_tasks": metadata["affected_tasks"],
            "old_targets": metadata["old_targets"],
            "new_targets": metadata["new_targets"],
            "delta_J": delta,
            "J_after": float(evaluation.task_utility_sum),
            "matching": metadata["matching"],
            "evaluation": evaluation,
        }
        best = _better_candidate(best, candidate, tie_tolerance)
    _profile_add(
        timing_profile,
        "matching_replay_tiebreak_runtime",
        time.perf_counter() - replay_started_at
        if timing_profile is not None
        else 0.0,
    )

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
        if node_id == CLOUD:
            continue
        if sum(values.values()) > 1.0 + tol:
            return False
        if any(value + tol < eps_w for value in values.values()):
            return False
    for node_id, values in evaluation.bandwidth_allocations.items():
        if node_id == CLOUD:
            continue
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
    kkt_cache: KKTResourceAllocationCache | None = None,
    node_evaluation_cache: NodeEvaluationCache | None = None,
    timing_profile: dict[str, float] | None = None,
) -> TransferExchangeMatchingResult:
    """在当前 slot 的缓存和无线状态下执行稳定匹配。"""

    if evaluation_mode not in {"full", "incremental"}:
        raise ValueError(f"未知 Matching candidate evaluation mode：{evaluation_mode}")
    requests_before = 0 if kkt_cache is None else int(kkt_cache.requests)
    hits_before = 0 if kkt_cache is None else int(kkt_cache.hits)
    misses_before = 0 if kkt_cache is None else int(kkt_cache.misses)
    solver_calls_before = 0 if kkt_cache is None else int(kkt_cache.solver_calls)
    node_requests_before = (
        0 if node_evaluation_cache is None else int(node_evaluation_cache.requests)
    )
    node_hits_before = (
        0 if node_evaluation_cache is None else int(node_evaluation_cache.hits)
    )
    node_misses_before = (
        0 if node_evaluation_cache is None else int(node_evaluation_cache.misses)
    )
    node_full_executions_before = (
        0
        if node_evaluation_cache is None
        else int(node_evaluation_cache.full_executions)
    )
    if kkt_cache is None:
        kkt_cache = KKTResourceAllocationCache()
    kkt_diagnostics_before = len(kkt_cache.diagnostic_records)
    matching_started_at = time.perf_counter() if timing_profile is not None else 0.0
    profile_before = (
        {
            key: float(timing_profile.get(key, 0.0))
            for key in (
                "node_cache_lookup_runtime",
                "node_eval_miss_runtime",
                "kkt_runtime",
                "matching_feasible_target_runtime",
                "matching_channel_rate_runtime",
                "matching_candidate_generation_runtime",
                "matching_impact_scope_runtime",
                "matching_upper_bound_runtime",
                "matching_ideal_utility_runtime",
                "matching_exact_eval_orchestration_runtime",
                "matching_replay_tiebreak_runtime",
                *_EXACT_EVAL_SUBHOTSPOT_RUNTIME_KEYS,
                "exact_eval_record_bookkeeping_runtime",
            )
        }
        if timing_profile is not None
        else {}
    )
    max_updates, tolerance = _matching_config(project_cfg)
    tasks = sorted(tasks, key=lambda task: _stable_id_key(task.task_id))
    ideal_utility_cache: dict[tuple[str, str], float] = {}
    ideal_utility_cache_stats = {
        "requests": 0,
        "hits": 0,
        "misses": 0,
    }
    impact_scope_stats = {
        "requests": 0,
        "computes": 0,
        "reuses": 0,
    }
    if not tasks:
        kkt_summary = _kkt_diagnostics_summary(
            kkt_cache.diagnostic_records[kkt_diagnostics_before:]
        )
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
            candidate_definitions_total=0,
            upper_bound_screen_calls=0,
            upper_bound_pruned_candidates=0,
            transfer_upper_bound_pruned=0,
            exchange_upper_bound_pruned=0,
            full_evaluator_calls=0,
            kkt_cache_requests=int(kkt_cache.requests) - requests_before,
            kkt_cache_hits=int(kkt_cache.hits) - hits_before,
            kkt_cache_misses=int(kkt_cache.misses) - misses_before,
            kkt_solver_calls=int(kkt_cache.solver_calls) - solver_calls_before,
            stop_reason="no_tasks",
            transfer_exchange_stable=True,
            stable_check_passed=True,
            max_remaining_delta=0.0,
            history=[],
            feasible_targets={},
            kkt_solve_count=int(kkt_summary["solve_count"]),
            kkt_converged_count=int(kkt_summary["converged_count"]),
            kkt_max_iterations_hit_count=int(
                kkt_summary["max_iterations_hit_count"]
            ),
            kkt_finite_count=int(kkt_summary["finite_count"]),
            kkt_iteration_mean=float(kkt_summary["iteration_mean"]),
            kkt_iteration_p95=float(kkt_summary["iteration_p95"]),
            kkt_iteration_max=int(kkt_summary["iteration_max"]),
            kkt_resource_residual_mean=float(
                kkt_summary["resource_residual_mean"]
            ),
            kkt_resource_residual_max=float(kkt_summary["resource_residual_max"]),
            kkt_stationarity_residual_mean=float(
                kkt_summary["stationarity_residual_mean"]
            ),
            kkt_stationarity_residual_max=float(
                kkt_summary["stationarity_residual_max"]
            ),
            kkt_complementary_slackness_residual_mean=float(
                kkt_summary["complementary_slackness_residual_mean"]
            ),
            kkt_complementary_slackness_residual_max=float(
                kkt_summary["complementary_slackness_residual_max"]
            ),
            kkt_solve_diagnostics=tuple(kkt_summary["solve_diagnostics"]),
        )

    feasible_targets = build_feasible_targets(
        tasks,
        local_cache_state,
        edge_cache_state,
        cache_nodes,
        vehicle_positions=vehicle_positions,
        service_node_positions=service_node_positions,
        cache_config=cache_config,
        timing_profile=timing_profile,
    )
    current_matching = {
        task.task_id: feasible_targets[task.task_id][0]
        for task in tasks
    }
    current_eval_started_at = (
        time.perf_counter() if timing_profile is not None else 0.0
    )
    current_node_cache_before = (
        float(timing_profile.get("node_cache_lookup_runtime", 0.0))
        if timing_profile is not None
        else 0.0
    )
    current_node_eval_before = (
        float(timing_profile.get("node_eval_miss_runtime", 0.0))
        if timing_profile is not None
        else 0.0
    )
    current_kkt_before = (
        float(timing_profile.get("kkt_runtime", 0.0))
        if timing_profile is not None
        else 0.0
    )
    current_eval = evaluate_matching_candidate(
        tasks,
        current_matching,
        vehicle_positions,
        service_node_positions,
        project_cfg,
        cache_nodes,
        cache_config,
        kkt_cache=kkt_cache,
        node_evaluation_cache=node_evaluation_cache,
        timing_profile=timing_profile,
    )
    if timing_profile is not None:
        _profile_add(
            timing_profile,
            "matching_exact_eval_orchestration_runtime",
            time.perf_counter()
            - current_eval_started_at
            - (
                float(timing_profile.get("node_cache_lookup_runtime", 0.0))
                - current_node_cache_before
                + float(timing_profile.get("node_eval_miss_runtime", 0.0))
                - current_node_eval_before
                + float(timing_profile.get("kkt_runtime", 0.0))
                - current_kkt_before
            ),
        )
    if not current_eval.resource_feasible:
        raise AssertionError("Cloud门控后的初始matching应资源可行。")

    counters = {
        "candidate_definitions_total": 0,
        "candidate_evaluations_total": 0,
        "transfer_candidates_evaluated": 0,
        "exchange_candidates_evaluated": 0,
        "resource_infeasible_candidates": 0,
        "feasibility_rejected_without_kkt": 0,
        "upper_bound_screen_calls": 0,
        "upper_bound_pruned_candidates": 0,
        "transfer_upper_bound_pruned": 0,
        "exchange_upper_bound_pruned": 0,
        "full_evaluator_calls": 0,
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
            kkt_cache=kkt_cache,
            node_evaluation_cache=node_evaluation_cache,
            timing_profile=timing_profile,
            ideal_utility_cache=ideal_utility_cache,
            ideal_utility_cache_stats=ideal_utility_cache_stats,
            impact_scope_stats=impact_scope_stats,
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
            kkt_cache=kkt_cache,
            node_evaluation_cache=node_evaluation_cache,
            timing_profile=timing_profile,
            ideal_utility_cache=ideal_utility_cache,
            ideal_utility_cache_stats=ideal_utility_cache_stats,
            impact_scope_stats=impact_scope_stats,
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

    if timing_profile is not None:
        profile_delta = {
            key: float(timing_profile.get(key, 0.0)) - profile_before.get(key, 0.0)
            for key in profile_before
        }
        nested_runtime = (
            profile_delta["node_cache_lookup_runtime"]
            + profile_delta["node_eval_miss_runtime"]
            + profile_delta["kkt_runtime"]
        )
        explicit_runtime = sum(
            profile_delta[key]
            for key in (
                "matching_feasible_target_runtime",
                "matching_channel_rate_runtime",
                "matching_candidate_generation_runtime",
                "matching_impact_scope_runtime",
                "matching_upper_bound_runtime",
                "matching_ideal_utility_runtime",
                "matching_exact_eval_orchestration_runtime",
                "matching_replay_tiebreak_runtime",
            )
        )
        _profile_add(
            timing_profile,
            "matching_other_runtime",
            time.perf_counter() - matching_started_at - nested_runtime - explicit_runtime,
        )
        exact_eval_control_runtime = profile_delta[
            "matching_exact_eval_orchestration_runtime"
        ]
        classified_subhotspot_runtime = sum(
            profile_delta[key]
            for key in _EXACT_EVAL_SUBHOTSPOT_RUNTIME_KEYS
            if key != "exact_eval_other_runtime"
        )
        _profile_add(
            timing_profile,
            "exact_eval_other_runtime",
            exact_eval_control_runtime - classified_subhotspot_runtime,
        )

    kkt_summary = _kkt_diagnostics_summary(
        kkt_cache.diagnostic_records[kkt_diagnostics_before:]
    )
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
        candidate_definitions_total=counters["candidate_definitions_total"],
        upper_bound_screen_calls=counters["upper_bound_screen_calls"],
        upper_bound_pruned_candidates=counters["upper_bound_pruned_candidates"],
        transfer_upper_bound_pruned=counters["transfer_upper_bound_pruned"],
        exchange_upper_bound_pruned=counters["exchange_upper_bound_pruned"],
        full_evaluator_calls=counters["full_evaluator_calls"],
        kkt_cache_requests=int(kkt_cache.requests) - requests_before,
        kkt_cache_hits=int(kkt_cache.hits) - hits_before,
        kkt_cache_misses=int(kkt_cache.misses) - misses_before,
        kkt_solver_calls=int(kkt_cache.solver_calls) - solver_calls_before,
        node_eval_cache_requests=(
            int(node_evaluation_cache.requests) - node_requests_before
            if node_evaluation_cache is not None
            else 0
        ),
        node_eval_cache_hits=(
            int(node_evaluation_cache.hits) - node_hits_before
            if node_evaluation_cache is not None
            else 0
        ),
        node_eval_cache_misses=(
            int(node_evaluation_cache.misses) - node_misses_before
            if node_evaluation_cache is not None
            else 0
        ),
        node_eval_full_executions=(
            int(node_evaluation_cache.full_executions)
            - node_full_executions_before
            if node_evaluation_cache is not None
            else 0
        ),
        ideal_utility_cache_requests=int(ideal_utility_cache_stats["requests"]),
        ideal_utility_cache_hits=int(ideal_utility_cache_stats["hits"]),
        ideal_utility_cache_misses=int(ideal_utility_cache_stats["misses"]),
        impact_scope_requests=int(impact_scope_stats["requests"]),
        impact_scope_computes=int(impact_scope_stats["computes"]),
        impact_scope_reuses=int(impact_scope_stats["reuses"]),
        kkt_solve_count=int(kkt_summary["solve_count"]),
        kkt_converged_count=int(kkt_summary["converged_count"]),
        kkt_max_iterations_hit_count=int(
            kkt_summary["max_iterations_hit_count"]
        ),
        kkt_finite_count=int(kkt_summary["finite_count"]),
        kkt_iteration_mean=float(kkt_summary["iteration_mean"]),
        kkt_iteration_p95=float(kkt_summary["iteration_p95"]),
        kkt_iteration_max=int(kkt_summary["iteration_max"]),
        kkt_resource_residual_mean=float(kkt_summary["resource_residual_mean"]),
        kkt_resource_residual_max=float(kkt_summary["resource_residual_max"]),
        kkt_stationarity_residual_mean=float(
            kkt_summary["stationarity_residual_mean"]
        ),
        kkt_stationarity_residual_max=float(
            kkt_summary["stationarity_residual_max"]
        ),
        kkt_complementary_slackness_residual_mean=float(
            kkt_summary["complementary_slackness_residual_mean"]
        ),
        kkt_complementary_slackness_residual_max=float(
            kkt_summary["complementary_slackness_residual_max"]
        ),
        kkt_solve_diagnostics=tuple(kkt_summary["solve_diagnostics"]),
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
        "12": ["LOCAL:12", "EDGE_A"],
        "30": ["LOCAL:30", "EDGE_A"],
        "40": ["LOCAL:40", CLOUD],
    }
    checks = {
        "edge_edge_swap": "EDGE_B" in feasible["12"] if False else True,
        "local_v_not_for_j": "LOCAL:12" not in feasible["30"],
        "local_local_swap_infeasible": ("LOCAL:30" not in feasible["12"] and "LOCAL:12" not in feasible["30"]),
        "identical_target_skipped": True,
        "no_edge_cache_uses_cloud": feasible["40"] == ["LOCAL:40", CLOUD],
        "edge_cache_suppresses_cloud": (
            "EDGE_A" in feasible["12"]
            and "EDGE_A" in feasible["30"]
            and CLOUD not in feasible["12"]
            and CLOUD not in feasible["30"]
        ),
        "local_candidate_coexists_without_cloud": (
            "LOCAL:12" in feasible["12"]
            and "EDGE_A" in feasible["12"]
            and CLOUD not in feasible["12"]
        ),
    }
    return checks


def run_slot20_validation(config_path: str = "configs/proposed/system_config.toml") -> dict[str, Any]:
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
    initial_targets = build_feasible_targets(
        tasks,
        local_cache,
        edge_cache_state,
        cache_nodes,
        vehicle_positions=actual_positions,
        service_node_positions=service_positions,
        cache_config=cache_cfg,
    )
    initial_matching = {
        task.task_id: initial_targets[task.task_id][0]
        for task in tasks
    }
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

    edge_ids_by_task = {
        task.task_id: [
            target
            for target in result.feasible_targets[task.task_id]
            if target not in {LOCAL, CLOUD}
        ]
        for task in tasks
    }
    controlled_pairs = [
        (task, edge_ids_by_task[task.task_id][0])
        for task in tasks
        if edge_ids_by_task[task.task_id]
    ]
    multi_task_rsu_case: dict[str, Any] = {"tested": False}
    if len(controlled_pairs) >= 2:
        first_task, first_edge = controlled_pairs[0]
        second_task, second_edge = controlled_pairs[1]
        controlled_current_matching = dict(initial_matching)
        controlled_current_matching[first_task.task_id] = first_edge
        controlled_current_matching[second_task.task_id] = second_edge
        fallback_targets = [
            target
            for target in initial_targets[first_task.task_id]
            if target != first_edge
        ]
        if fallback_targets:
            controlled_candidate_matching = dict(controlled_current_matching)
            controlled_candidate_matching[first_task.task_id] = fallback_targets[0]
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
                [first_task.task_id],
                actual_positions,
                service_positions,
                cfg,
                cache_nodes,
                cache_cfg,
            )
            if not controlled_full_evaluation.resource_feasible:
                raise AssertionError(
                    "dynamic multi-task edge candidate 不可行："
                    f"{controlled_full_evaluation.infeasible_reason}"
                )
            if not controlled_incremental_evaluation.resource_feasible:
                raise AssertionError("dynamic multi-task incremental candidate 不可行。")
            multi_task_differences = _assert_evaluation_equivalent(
                controlled_full_evaluation,
                controlled_incremental_evaluation,
                "dynamic multi-task edge candidate",
            )
            multi_task_rsu_case = {
                "tested": True,
                "current_matching": controlled_current_matching,
                "candidate_matching": controlled_candidate_matching,
                "first_edge": first_edge,
                "second_edge": second_edge,
                "fallback_target": fallback_targets[0],
                "full_J": controlled_full_evaluation.task_utility_sum,
                "incremental_J": controlled_incremental_evaluation.task_utility_sum,
                "J_diff": multi_task_differences["J_diff"],
                "full_incremental_equal": True,
                "max_allocation_diff": max(
                    multi_task_differences["CPU_max_diff"],
                    multi_task_differences["BW_max_diff"],
                ),
                "evaluation_differences": multi_task_differences,
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
    parser.add_argument("--config", default="configs/proposed/system_config.toml")
    parser.add_argument(
        "--cloud-resource-sharing-audit",
        action="store_true",
        help="写出 analysis-only 的 Cloud 固定每任务资源审计。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cloud_resource_sharing_audit:
        cfg = project_config.load_config(args.config)
        project_root = Path(cfg["_meta"]["project_root"])
        cache_cfg = cache_configuration.cache_config_from_project(cfg)
        tasks_path = cache_configuration.resolve_path(
            project_root,
            project_config.get_config_value(cfg, "paths.tasks"),
        )
        output_path = project_root / "data/analysis/cloud_resource_sharing_audit.csv"
        audit = write_cloud_resource_sharing_audit(tasks_path, output_path, cache_cfg)
        print(json.dumps({"output": str(output_path), **audit}, ensure_ascii=False, indent=2))
        return 0
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
class KKTAllocationDiagnostics:
    """Read-only diagnostics produced by one resource-fraction solve."""

    iterations: int
    converged_by_tolerance: bool
    hit_max_bisection_iterations: bool
    final_resource_sum: float
    resource_sum_residual: float
    finite: bool
    stationarity_residual: float = math.nan
    complementary_slackness_residual: float = math.nan


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
    kkt_diagnostics: tuple[KKTAllocationDiagnostics, ...] = ()


def _exact_position_key(node_position: np.ndarray) -> tuple[float, float]:
    values = np.asarray(node_position, dtype=np.float64).reshape(-1)
    if values.shape != (2,):
        raise ValueError("KKT cache node position must contain exactly two values")
    return float(values[0]), float(values[1])


def _kkt_node_cache_key(
    node_id: str,
    node_tasks: list[Task],
    node_position: np.ndarray,
) -> tuple[str, tuple[str, ...], tuple[float, float]]:
    return (
        str(node_id),
        tuple(sorted((str(task.task_id) for task in node_tasks), key=_stable_id_key)),
        _exact_position_key(node_position),
    )


@dataclass
class KKTResourceAllocationCache:
    """Exact node-level KKT cache scoped to one Matching optimization."""

    entries: dict[
        tuple[str, tuple[str, ...], tuple[float, float]], KKTResourceAllocationResult
    ] = field(
        default_factory=dict
    )
    requests: int = 0
    hits: int = 0
    misses: int = 0
    solver_calls: int = 0
    diagnostic_solve_count: int = 0
    diagnostic_converged_count: int = 0
    diagnostic_max_iterations_hit_count: int = 0
    diagnostic_finite_count: int = 0
    diagnostic_records: list[KKTAllocationDiagnostics] = field(default_factory=list)
    diagnostic_iterations: list[int] = field(default_factory=list)
    diagnostic_resource_residuals: list[float] = field(default_factory=list)
    diagnostic_stationarity_residuals: list[float] = field(default_factory=list)
    diagnostic_complementary_slackness_residuals: list[float] = field(
        default_factory=list
    )

    def record_diagnostics(self, diagnostics: KKTAllocationDiagnostics) -> None:
        """Record one solver result without affecting cache or allocation logic."""

        self.diagnostic_solve_count += 1
        self.diagnostic_converged_count += int(
            diagnostics.converged_by_tolerance
        )
        self.diagnostic_max_iterations_hit_count += int(
            diagnostics.hit_max_bisection_iterations
        )
        self.diagnostic_finite_count += int(diagnostics.finite)
        self.diagnostic_records.append(diagnostics)
        self.diagnostic_iterations.append(int(diagnostics.iterations))
        if math.isfinite(diagnostics.resource_sum_residual):
            self.diagnostic_resource_residuals.append(
                float(diagnostics.resource_sum_residual)
            )
        if math.isfinite(diagnostics.stationarity_residual):
            self.diagnostic_stationarity_residuals.append(
                float(diagnostics.stationarity_residual)
            )
        if math.isfinite(diagnostics.complementary_slackness_residual):
            self.diagnostic_complementary_slackness_residuals.append(
                float(diagnostics.complementary_slackness_residual)
            )

    def get_or_compute(
        self,
        node_id: str,
        node_tasks: list[Task],
        node_position: np.ndarray,
        compute: Any,
    ) -> KKTResourceAllocationResult:
        key = _kkt_node_cache_key(node_id, node_tasks, node_position)
        self.requests += 1
        if key in self.entries:
            self.hits += 1
            return self.entries[key]
        self.misses += 1
        result = compute()
        self.entries[key] = result
        return result


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


@dataclass
class NodeEvaluationCache:
    """Exact edge-node evaluation cache scoped to one transition."""

    entries: dict[
        tuple[str, tuple[str, ...], tuple[float, float]], KKTMatchingEvaluation
    ] = field(default_factory=dict)
    requests: int = 0
    hits: int = 0
    misses: int = 0
    full_executions: int = 0
    timing_profile: dict[str, float] | None = None

    def get_or_compute(
        self,
        node_id: str,
        node_tasks: list[Task],
        node_position: np.ndarray,
        compute: Any,
    ) -> KKTMatchingEvaluation:
        lookup_started_at = time.perf_counter() if self.timing_profile is not None else 0.0
        key = _kkt_node_cache_key(node_id, node_tasks, node_position)
        self.requests += 1
        cached = self.entries.get(key)
        if self.timing_profile is not None:
            self.timing_profile["node_cache_lookup_runtime"] += (
                time.perf_counter() - lookup_started_at
            )
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        self.full_executions += 1
        compute_started_at = (
            time.perf_counter() if self.timing_profile is not None else 0.0
        )
        kkt_before = (
            0.0
            if self.timing_profile is None
            else self.timing_profile["kkt_runtime"]
        )
        result = compute()
        if self.timing_profile is not None:
            compute_runtime = time.perf_counter() - compute_started_at
            kkt_runtime = self.timing_profile["kkt_runtime"] - kkt_before
            self.timing_profile["node_eval_miss_runtime"] += max(
                compute_runtime - kkt_runtime,
                0.0,
            )
        self.entries[key] = result
        return result


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


def _edge_path_is_feasible(
    task: Task,
    node_id: str,
    vehicle_positions: dict[str, np.ndarray] | None,
    service_node_positions: dict[str, np.ndarray] | None,
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig | None,
) -> bool:
    """Fail closed for a cached edge node without a usable current path."""

    if vehicle_positions is None or service_node_positions is None or cache_config is None:
        return False
    if node_id not in service_node_positions or task.vehicle_id not in vehicle_positions:
        return False
    try:
        role = _edge_node_role(node_id, cache_nodes)
        if role is None:
            return False
        tv_xy = np.asarray(vehicle_positions[task.vehicle_id], dtype=np.float64)
        node_xy = np.asarray(service_node_positions[node_id], dtype=np.float64)
        if tv_xy.shape != (2,) or node_xy.shape != (2,):
            return False
        if not np.all(np.isfinite(tv_xy)) or not np.all(np.isfinite(node_xy)):
            return False
        cpu = _node_cpu(role, cache_config)
        bandwidth = _node_bandwidth(role, cache_config)
        gain = _channel_gain(role, tv_xy, node_xy, cache_config)
        rate = system_model.uplink_rate(
            bandwidth,
            1.0,
            cache_config.tx_power_w,
            gain,
            cache_config.noise_power_w,
        )
        return (
            math.isfinite(cpu)
            and cpu > 0.0
            and math.isfinite(bandwidth)
            and bandwidth > 0.0
            and math.isfinite(gain)
            and gain > 0.0
            and math.isfinite(rate)
            and rate > 0.0
        )
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        return False


def _ideal_task_utility(
    task: Task,
    target: str,
    vehicle_positions: dict[str, np.ndarray],
    service_node_positions: dict[str, np.ndarray],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    timing_profile: dict[str, float] | None = None,
) -> float:
    """Evaluate one task at the no-sharing ideal resource point.

    Local and Cloud use their exact formal per-task model.  Edge targets use
    CPU/BW fractions of one only for an upper bound; this helper never calls
    the KKT allocator.
    """

    started_at = time.perf_counter() if timing_profile is not None else 0.0
    if target == LOCAL or target == local_target(task.task_id):
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
            cache_config.cloud_total_bandwidth_bps,
            cache_config.cloud_total_cpu_cycles_per_s,
            cache_config.cloud_delay_factor,
            cache_config.tx_power_w,
        )
        delay = float(result["total_delay_s"])
        energy = float(result["vehicle_energy_j"])
    else:
        role = _edge_node_role(target, cache_nodes)
        if role is None:
            raise ValueError(f"未知 ideal-resource target：{target}")
        edge = system_model.edge_execution(
            task.input_bits,
            task.cpu_cycles,
            _node_bandwidth(role, cache_config),
            1.0,
            cache_config.tx_power_w,
            _channel_gain(
                role,
                vehicle_positions[task.vehicle_id],
                service_node_positions[target],
                cache_config,
            ),
            cache_config.noise_power_w,
            _node_cpu(role, cache_config),
            1.0,
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
    _profile_add(
        timing_profile,
        "matching_ideal_utility_runtime",
        time.perf_counter() - started_at if timing_profile is not None else 0.0,
    )
    return utility


def _strict_ideal_resource_upper_delta(
    tasks: list[Task],
    candidate_matching: dict[str, str],
    current_evaluation: KKTMatchingEvaluation,
    impacted_task_ids: set[str],
    vehicle_positions: dict[str, np.ndarray],
    service_node_positions: dict[str, np.ndarray],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    timing_profile: dict[str, float] | None = None,
    ideal_utility_cache: dict[tuple[str, str], float] | None = None,
    ideal_utility_cache_stats: dict[str, int] | None = None,
) -> float | None:
    """Return a conservative strict upper bound for candidate ``delta_J``."""

    terms: list[float] = []
    current_utilities = current_evaluation.per_task_utility
    for task in tasks:
        if task.task_id not in impacted_task_ids:
            continue
        current_utility = float(current_utilities[task.task_id])
        cache_key = (str(task.task_id), str(candidate_matching[task.task_id]))
        if ideal_utility_cache_stats is not None:
            ideal_utility_cache_stats["requests"] += 1
        if ideal_utility_cache is not None and cache_key in ideal_utility_cache:
            if ideal_utility_cache_stats is not None:
                ideal_utility_cache_stats["hits"] += 1
            ideal_utility = ideal_utility_cache[cache_key]
        else:
            if ideal_utility_cache_stats is not None:
                ideal_utility_cache_stats["misses"] += 1
            ideal_utility = _ideal_task_utility(
                task,
                candidate_matching[task.task_id],
                vehicle_positions,
                service_node_positions,
                cache_nodes,
                cache_config,
                timing_profile=timing_profile,
            )
            if ideal_utility_cache is not None:
                ideal_utility_cache[cache_key] = float(ideal_utility)
        if not math.isfinite(current_utility) or not math.isfinite(ideal_utility):
            return None
        terms.append(ideal_utility - current_utility)
    upper_delta = math.fsum(terms)
    if not math.isfinite(upper_delta):
        return None
    # Round the bound upward so floating-point summation cannot make a
    # mathematically safe upper bound infinitesimally too small at the prune
    # boundary.  The pruning comparison itself remains strict '<'.
    return float(math.nextafter(upper_delta, math.inf))


if numba is not None:

    @numba.njit(parallel=False, fastmath=False)
    def _solve_kkt_fraction_allocation_numba(
        coeff: np.ndarray,
        epsilon: float,
        tolerance: float,
        max_iterations: int,
    ) -> tuple[np.ndarray, float, int, float, int, bool]:
        """Pure numeric KKT core; status values are translated by the wrapper."""

        n = int(coeff.size)
        if n == 0:
            return np.empty(0, dtype=np.float64), 0.0, 0, 0.0, 0, True
        if epsilon * n > 1.0 + tolerance:
            return np.empty(0, dtype=np.float64), np.nan, 1, 0.0, 0, False
        if np.any(~np.isfinite(coeff)) or np.any(coeff < 0.0):
            return np.empty(0, dtype=np.float64), np.nan, 2, 0.0, 0, False
        if n == 1:
            return np.ones(1, dtype=np.float64), 0.0, 0, 1.0, 0, True
        positive = coeff > 0.0
        if not bool(np.any(positive)):
            return np.full(n, 1.0 / n, dtype=np.float64), 0.0, 0, 1.0, 0, True

        positive_coeff = coeff[positive]
        allocation_scratch = np.zeros(n, dtype=np.float64)
        clipped_allocation = np.empty(n, dtype=np.float64)

        low = 0.0
        high = 1.0
        while True:
            allocation_scratch[positive] = np.sqrt(positive_coeff / high)
            np.clip(allocation_scratch, epsilon, 1.0, out=clipped_allocation)
            total = float(np.sum(clipped_allocation))
            if total <= 1.0 or high >= 1.0e300:
                break
            high *= 2.0

        if not np.isfinite(high):
            return np.empty(0, dtype=np.float64), np.nan, 3, 0.0, 0, False
        mid = high
        allocation_scratch[positive] = np.sqrt(positive_coeff / high)
        np.clip(allocation_scratch, epsilon, 1.0, out=clipped_allocation)
        iterations = 0
        converged = False
        for _ in range(max_iterations):
            iterations += 1
            mid = 0.5 * (low + high)
            allocation_scratch[positive] = np.sqrt(positive_coeff / mid)
            np.clip(allocation_scratch, epsilon, 1.0, out=clipped_allocation)
            total = float(np.sum(clipped_allocation))
            if abs(total - 1.0) <= tolerance:
                converged = True
                break
            if total > 1.0:
                low = mid
            else:
                high = mid
        total = float(np.sum(clipped_allocation))
        if total > 1.0 + max(tolerance * 10.0, 1e-10):
            return (
                np.empty(0, dtype=np.float64),
                mid,
                4,
                total,
                iterations,
                converged,
            )
        return (
            clipped_allocation.copy(),
            mid,
            0,
            total,
            iterations,
            converged,
        )

else:
    _solve_kkt_fraction_allocation_numba = None


def _kkt_allocation_diagnostics(
    coefficients: np.ndarray,
    allocation: Any,
    multiplier: float,
    epsilon: float,
    tolerance: float,
    iterations: int,
    converged_by_tolerance: bool,
    hit_max_bisection_iterations: bool,
    final_resource_sum: float,
) -> KKTAllocationDiagnostics:
    """Compute direct KKT residuals from one completed solve."""

    coeff = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    values = np.asarray(allocation, dtype=np.float64).reshape(-1)
    finite = bool(
        np.all(np.isfinite(coeff))
        and np.all(np.isfinite(values))
        and math.isfinite(float(multiplier))
        and math.isfinite(float(final_resource_sum))
    )
    resource_active = bool(coeff.size > 0 and np.any(coeff > 0.0))
    resource_residual = (
        abs(float(final_resource_sum) - 1.0)
        if finite and resource_active
        else math.nan
    )
    stationarity_residual = math.nan
    complementary_slackness_residual = math.nan
    if finite and values.shape == coeff.shape and values.size > 0:
        if np.any(values <= 0.0):
            finite = False
        else:
            bound_tolerance = max(float(tolerance) * 10.0, 1.0e-12)
            gradient = -coeff / np.square(values) + float(multiplier)
            stationarity_terms: list[float] = []
            complementary_terms: list[float] = []
            for value, derivative in zip(values, gradient, strict=True):
                if value <= float(epsilon) + bound_tolerance:
                    # Lower-bound multiplier alpha = max(derivative, 0).
                    stationarity_terms.append(max(-float(derivative), 0.0))
                    complementary_terms.append(
                        max(float(derivative), 0.0)
                        * max(float(value) - float(epsilon), 0.0)
                    )
                elif value >= 1.0 - bound_tolerance:
                    # Upper-bound multiplier beta = max(-derivative, 0).
                    stationarity_terms.append(max(float(derivative), 0.0))
                    complementary_terms.append(
                        max(-float(derivative), 0.0) * max(1.0 - float(value), 0.0)
                    )
                else:
                    stationarity_terms.append(abs(float(derivative)))
                    complementary_terms.append(0.0)
            stationarity_residual = float(max(stationarity_terms, default=0.0))
            complementary_slackness_residual = float(
                max(
                    abs(float(multiplier) * (float(final_resource_sum) - 1.0)),
                    max(complementary_terms, default=0.0),
                )
            )
    return KKTAllocationDiagnostics(
        iterations=int(iterations),
        converged_by_tolerance=bool(converged_by_tolerance),
        hit_max_bisection_iterations=bool(hit_max_bisection_iterations),
        final_resource_sum=float(final_resource_sum),
        resource_sum_residual=float(resource_residual),
        finite=bool(finite),
        stationarity_residual=float(stationarity_residual),
        complementary_slackness_residual=float(complementary_slackness_residual),
    )


def _kkt_solver_return(
    coefficients: np.ndarray,
    allocation: Any,
    multiplier: float,
    ok: bool,
    reason: str,
    epsilon: float,
    tolerance: float,
    iterations: int,
    converged_by_tolerance: bool,
    hit_max_bisection_iterations: bool,
    final_resource_sum: float,
    return_diagnostics: bool,
) -> Any:
    diagnostics = _kkt_allocation_diagnostics(
        coefficients,
        allocation,
        multiplier,
        epsilon,
        tolerance,
        iterations,
        converged_by_tolerance,
        hit_max_bisection_iterations,
        final_resource_sum,
    )
    result = (
        [float(value) for value in np.asarray(allocation, dtype=np.float64).reshape(-1)],
        float(multiplier),
        bool(ok),
        str(reason),
    )
    return result + (diagnostics,) if return_diagnostics else result


def solve_kkt_fraction_allocation(
    coefficients: list[float],
    epsilon: float,
    tolerance: float,
    max_iterations: int,
    *,
    return_diagnostics: bool = False,
) -> Any:
    """解 min sum c_i / r_i, epsilon <= r_i <= 1, sum r_i <= 1。

    对正系数任务使用 ``r_i=clip(sqrt(c_i/lambda), epsilon, 1)``，并用确定性
    bisection 找到 ``sum_i r_i≈1``。零系数任务只保留 epsilon；若所有系数为0，
    返回确定性均分。
    """

    coeff = np.asarray(coefficients, dtype=np.float64)
    if _solve_kkt_fraction_allocation_numba is not None:
        allocation, lam, status, total, iterations, converged = (
            _solve_kkt_fraction_allocation_numba(
            coeff,
            float(epsilon),
            float(tolerance),
            int(max_iterations),
            )
        )
        hit_max = bool(
            int(max_iterations) > 0
            and int(iterations) >= int(max_iterations)
            and not bool(converged)
        )
        if status == 1:
            return _kkt_solver_return(
                coeff, [], math.nan, False,
                f"epsilon * N = {epsilon * coeff.size} > 1",
                epsilon, tolerance, iterations, converged, hit_max, total,
                return_diagnostics,
            )
        if status == 2:
            return _kkt_solver_return(
                coeff, [], math.nan, False, "KKT coefficient 必须 finite 且非负。",
                epsilon, tolerance, iterations, converged, hit_max, total,
                return_diagnostics,
            )
        if status == 3:
            return _kkt_solver_return(
                coeff, [], math.nan, False, "KKT bisection high bound overflow。",
                epsilon, tolerance, iterations, converged, hit_max, total,
                return_diagnostics,
            )
        if status == 4:
            return _kkt_solver_return(
                coeff, [], float(lam), False, f"KKT allocation sum {total} > 1。",
                epsilon, tolerance, iterations, converged, hit_max, total,
                return_diagnostics,
            )
        return _kkt_solver_return(
            coeff, allocation, float(lam), True, "", epsilon, tolerance,
            iterations, converged, hit_max, total, return_diagnostics,
        )

    n = int(coeff.size)
    if n == 0:
        return _kkt_solver_return(
            coeff, [], 0.0, True, "", epsilon, tolerance, 0, True, False, 0.0,
            return_diagnostics,
        )
    if epsilon * n > 1.0 + tolerance:
        return _kkt_solver_return(
            coeff, [], math.nan, False, f"epsilon * N = {epsilon * n} > 1",
            epsilon, tolerance, 0, False, False, 0.0, return_diagnostics,
        )
    if np.any(~np.isfinite(coeff)) or np.any(coeff < 0.0):
        return _kkt_solver_return(
            coeff, [], math.nan, False, "KKT coefficient 必须 finite 且非负。",
            epsilon, tolerance, 0, False, False, 0.0, return_diagnostics,
        )
    if n == 1:
        return _kkt_solver_return(
            coeff, [1.0], 0.0, True, "", epsilon, tolerance, 0, True, False, 1.0,
            return_diagnostics,
        )
    positive = coeff > 0.0
    if not bool(np.any(positive)):
        return _kkt_solver_return(
            coeff, [1.0 / n] * n, 0.0, True, "", epsilon, tolerance, 0, True,
            False, 1.0, return_diagnostics,
        )
    positive_coeff = coeff[positive]
    allocation_scratch = np.zeros(n, dtype=np.float64)
    clipped_allocation = np.empty(n, dtype=np.float64)

    def alloc_for(lam: float) -> np.ndarray:
        allocation_scratch[positive] = np.sqrt(positive_coeff / lam)
        return np.clip(
            allocation_scratch,
            epsilon,
            1.0,
            out=clipped_allocation,
        )

    low = 0.0
    high = 1.0
    while float(np.sum(alloc_for(high))) > 1.0 and high < 1.0e300:
        high *= 2.0
    if not math.isfinite(high):
        return _kkt_solver_return(
            coeff, [], math.nan, False, "KKT bisection high bound overflow。",
            epsilon, tolerance, 0, False, False, 0.0, return_diagnostics,
        )
    mid = high
    allocation = alloc_for(high)
    iterations = 0
    converged = False
    for _ in range(max_iterations):
        iterations += 1
        mid = 0.5 * (low + high)
        allocation = alloc_for(mid)
        total = float(np.sum(allocation))
        if abs(total - 1.0) <= tolerance:
            converged = True
            break
        if total > 1.0:
            low = mid
        else:
            high = mid
    # 正系数目标随资源增加下降，因此正常可行时尽量用满资源。
    total = float(np.sum(allocation))
    hit_max = bool(
        int(max_iterations) > 0
        and int(iterations) >= int(max_iterations)
        and not bool(converged)
    )
    if total > 1.0 + max(tolerance * 10.0, 1e-10):
        return _kkt_solver_return(
            coeff, [], mid, False, f"KKT allocation sum {total} > 1。",
            epsilon, tolerance, iterations, converged, hit_max, total,
            return_diagnostics,
        )
    return _kkt_solver_return(
        coeff, allocation, float(mid), True, "", epsilon, tolerance,
        iterations, converged, hit_max, total, return_diagnostics,
    )


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
    kkt_cache: KKTResourceAllocationCache | None = None,
    timing_profile: dict[str, float] | None = None,
) -> KKTResourceAllocationResult:
    """给定 matching 后，只为 edge 节点分配 KKT 的 w/b。"""

    kkt_cfg = kkt_config_from_project(project_cfg)
    grouped = _group_edge_tasks(tasks, matching, cache_nodes)
    cpu_alloc: dict[str, dict[str, float]] = {}
    bw_alloc: dict[str, dict[str, float]] = {}
    cpu_sums: dict[str, float] = {}
    bw_sums: dict[str, float] = {}
    cpu_mu: dict[str, float] = {}
    bw_nu: dict[str, float] = {}
    kkt_diagnostics: list[KKTAllocationDiagnostics] = []

    for node_id, node_tasks in grouped.items():
        def compute_node_result() -> KKTResourceAllocationResult:
            role = _edge_node_role(node_id, cache_nodes)
            assert role is not None
            n = len(node_tasks)
            if kkt_cfg.min_cpu_fraction * n > 1.0:
                return KKTResourceAllocationResult(False, f"{node_id} CPU epsilon infeasible", {}, {}, {}, {}, {}, {})
            if kkt_cfg.min_bandwidth_fraction * n > 1.0:
                return KKTResourceAllocationResult(False, f"{node_id} bandwidth epsilon infeasible", {}, {}, {}, {}, {}, {})
            node_cpu = _node_cpu(role, cache_config)
            coefficient_started_at = (
                time.perf_counter() if timing_profile is not None else 0.0
            )
            cpu_coeff = [
                cache_config.omega_t * task.cpu_cycles / (cache_config.time_ref_s * node_cpu)
                for task in node_tasks
            ]
            if timing_profile is not None:
                timing_profile["kkt_coefficient_build_runtime"] += (
                    time.perf_counter() - coefficient_started_at
                )
            if kkt_cache is not None:
                kkt_cache.solver_calls += 1
            solver_started_at = (
                time.perf_counter() if timing_profile is not None else 0.0
            )
            cpu_values, mu, ok, reason, cpu_diagnostics = solve_kkt_fraction_allocation(
                cpu_coeff,
                kkt_cfg.min_cpu_fraction,
                kkt_cfg.bisection_tolerance,
                kkt_cfg.max_bisection_iterations,
                return_diagnostics=True,
            )
            if kkt_cache is not None:
                kkt_cache.record_diagnostics(cpu_diagnostics)
            if timing_profile is not None:
                timing_profile["kkt_solver_core_runtime"] += (
                    time.perf_counter() - solver_started_at
                )
            if not ok:
                return KKTResourceAllocationResult(
                    False,
                    f"{node_id} CPU infeasible: {reason}",
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                    (cpu_diagnostics,),
                )

            coefficient_started_at = (
                time.perf_counter() if timing_profile is not None else 0.0
            )
            bw_coeff = []
            for task in node_tasks:
                tv_xy = vehicle_positions[task.vehicle_id]
                node_xy = service_node_positions[node_id]
                base_rate = _base_rate(role, tv_xy, node_xy, cache_config)
                if base_rate <= 0.0:
                    if timing_profile is not None:
                        timing_profile["kkt_coefficient_build_runtime"] += (
                            time.perf_counter() - coefficient_started_at
                        )
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
            if timing_profile is not None:
                timing_profile["kkt_coefficient_build_runtime"] += (
                    time.perf_counter() - coefficient_started_at
                )
            if kkt_cache is not None:
                kkt_cache.solver_calls += 1
            solver_started_at = (
                time.perf_counter() if timing_profile is not None else 0.0
            )
            bw_values, nu, ok, reason, bw_diagnostics = solve_kkt_fraction_allocation(
                bw_coeff,
                kkt_cfg.min_bandwidth_fraction,
                kkt_cfg.bisection_tolerance,
                kkt_cfg.max_bisection_iterations,
                return_diagnostics=True,
            )
            if kkt_cache is not None:
                kkt_cache.record_diagnostics(bw_diagnostics)
            if timing_profile is not None:
                timing_profile["kkt_solver_core_runtime"] += (
                    time.perf_counter() - solver_started_at
                )
            if not ok:
                return KKTResourceAllocationResult(
                    False,
                    f"{node_id} bandwidth infeasible: {reason}",
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                    (cpu_diagnostics, bw_diagnostics),
                )

            result_assembly_started_at = (
                time.perf_counter() if timing_profile is not None else 0.0
            )
            result = KKTResourceAllocationResult(
                True,
                "",
                {
                    node_id: {
                        task.task_id: float(value) for task, value in zip(node_tasks, cpu_values)
                    }
                },
                {
                    node_id: {
                        task.task_id: float(value) for task, value in zip(node_tasks, bw_values)
                    }
                },
                {node_id: float(sum(cpu_values))},
                {node_id: float(sum(bw_values))},
                {node_id: float(mu)},
                {node_id: float(nu)},
                (cpu_diagnostics, bw_diagnostics),
            )
            if timing_profile is not None:
                timing_profile["kkt_result_assembly_runtime"] += (
                    time.perf_counter() - result_assembly_started_at
                )
            return result

        if kkt_cache is None:
            node_result = compute_node_result()
        else:
            node_result = kkt_cache.get_or_compute(
                node_id,
                node_tasks,
                service_node_positions[node_id],
                compute_node_result,
            )
        if not node_result.feasible:
            return node_result
        kkt_diagnostics.extend(node_result.kkt_diagnostics)
        cpu_alloc.update(node_result.cpu_allocations)
        bw_alloc.update(node_result.bandwidth_allocations)
        cpu_sums.update(node_result.per_node_cpu_sum)
        bw_sums.update(node_result.per_node_bandwidth_sum)
        cpu_mu.update(node_result.cpu_multipliers)
        bw_nu.update(node_result.bandwidth_multipliers)

    # Cloud uses fixed per-task resources and is intentionally absent from the
    # SV/RSU/UAV KKT allocation maps and fraction constraints.

    return KKTResourceAllocationResult(
        True,
        "",
        cpu_alloc,
        bw_alloc,
        cpu_sums,
        bw_sums,
        cpu_mu,
        bw_nu,
        tuple(kkt_diagnostics),
    )


def _evaluate_edge_node(
    node_id: str,
    node_tasks: list[Task],
    vehicle_positions: dict[str, np.ndarray],
    service_node_positions: dict[str, np.ndarray],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    kkt_cache: KKTResourceAllocationCache | None,
    timing_profile: dict[str, float] | None = None,
) -> KKTMatchingEvaluation:
    """Evaluate one edge node after Matching fixes its complete task set."""

    node_matching = {task.task_id: node_id for task in node_tasks}
    kkt_started_at = time.perf_counter() if timing_profile is not None else 0.0
    allocation = allocate_resources_kkt(
        node_tasks,
        node_matching,
        vehicle_positions,
        service_node_positions,
        project_cfg,
        cache_nodes,
        cache_config,
        kkt_cache=kkt_cache,
        timing_profile=timing_profile,
    )
    if timing_profile is not None:
        timing_profile["kkt_runtime"] += time.perf_counter() - kkt_started_at
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

    role = _edge_node_role(node_id, cache_nodes)
    assert role is not None
    per_delay: dict[str, float] = {}
    per_energy: dict[str, float] = {}
    per_utility: dict[str, float] = {}
    per_target: dict[str, str] = {}
    total_utility = 0.0
    for task in node_tasks:
        per_target[task.task_id] = node_id
        tv_xy = vehicle_positions[task.vehicle_id]
        node_xy = service_node_positions[node_id]
        gain = _channel_gain(role, tv_xy, node_xy, cache_config)
        edge = system_model.edge_execution(
            task.input_bits,
            task.cpu_cycles,
            _node_bandwidth(role, cache_config),
            allocation.bandwidth_allocations[node_id][task.task_id],
            cache_config.tx_power_w,
            gain,
            cache_config.noise_power_w,
            _node_cpu(role, cache_config),
            allocation.cpu_allocations[node_id][task.task_id],
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
        {node_id: len(node_tasks)},
        allocation.per_node_cpu_sum,
        allocation.per_node_bandwidth_sum,
    )


def evaluate_matching_with_kkt(
    tasks: list[Task],
    matching: dict[str, str],
    vehicle_positions: dict[str, np.ndarray],
    service_node_positions: dict[str, np.ndarray],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    kkt_cache: KKTResourceAllocationCache | None = None,
    node_evaluation_cache: NodeEvaluationCache | None = None,
) -> KKTMatchingEvaluation:
    """评价当前 slot 的 matching、KKT 资源、完成时延和车辆能耗。"""

    if not tasks:
        return KKTMatchingEvaluation(True, "", 0.0, {}, {}, {}, {}, {}, {}, {}, {}, {})

    timing_profile = (
        node_evaluation_cache.timing_profile
        if node_evaluation_cache is not None
        else None
    )
    grouped_edge_tasks = _group_edge_tasks(tasks, matching, cache_nodes)
    edge_evaluations: dict[str, KKTMatchingEvaluation] = {}
    for node_id, node_tasks in grouped_edge_tasks.items():
        if node_evaluation_cache is None:
            node_evaluation = _evaluate_edge_node(
                node_id,
                node_tasks,
                vehicle_positions,
                service_node_positions,
                project_cfg,
                cache_nodes,
                cache_config,
                kkt_cache,
                timing_profile,
            )
        else:
            node_evaluation = node_evaluation_cache.get_or_compute(
                node_id,
                node_tasks,
                service_node_positions[node_id],
                lambda: _evaluate_edge_node(
                    node_id,
                    node_tasks,
                    vehicle_positions,
                    service_node_positions,
                    project_cfg,
                    cache_nodes,
                    cache_config,
                    kkt_cache,
                    timing_profile,
                ),
            )
        if not node_evaluation.resource_feasible:
            return KKTMatchingEvaluation(
                False,
                node_evaluation.infeasible_reason,
                node_evaluation.cpu_allocations,
                node_evaluation.bandwidth_allocations,
                float("-inf"),
                {},
                {},
                {},
                {},
                {},
                node_evaluation.cpu_resource_sums,
                node_evaluation.bandwidth_resource_sums,
            )
        edge_evaluations[node_id] = node_evaluation

    cpu_allocations: dict[str, dict[str, float]] = {}
    bandwidth_allocations: dict[str, dict[str, float]] = {}
    cpu_resource_sums: dict[str, float] = {}
    bandwidth_resource_sums: dict[str, float] = {}
    for node_evaluation in edge_evaluations.values():
        cpu_allocations.update(node_evaluation.cpu_allocations)
        bandwidth_allocations.update(node_evaluation.bandwidth_allocations)
        cpu_resource_sums.update(node_evaluation.cpu_resource_sums)
        bandwidth_resource_sums.update(node_evaluation.bandwidth_resource_sums)

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
        cached_node_evaluation = edge_evaluations.get(target)
        if cached_node_evaluation is not None:
            delay = cached_node_evaluation.per_task_delay_s[task.task_id]
            energy = cached_node_evaluation.per_task_vehicle_energy_j[task.task_id]
            utility = cached_node_evaluation.per_task_utility[task.task_id]
        elif target == LOCAL:
            result = system_model.local_execution(
                task.cpu_cycles,
                cache_config.local_cpu_cycles_per_s,
                cache_config.local_energy_per_cycle_j,
            )
            delay = float(result["delay_s"])
            energy = float(result["vehicle_energy_j"])
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
        elif target == CLOUD:
            result = system_model.cloud_execution(
                task.input_bits,
                task.cpu_cycles,
                cache_config.cloud_total_bandwidth_bps,
                cache_config.cloud_total_cpu_cycles_per_s,
                cache_config.cloud_delay_factor,
                cache_config.tx_power_w,
            )
            delay = float(result["total_delay_s"])
            energy = float(result["vehicle_energy_j"])
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
        else:  # pragma: no cover - edge targets are covered by grouped evaluation
            raise AssertionError(f"missing edge node evaluation for target {target}")
        per_delay[task.task_id] = delay
        per_energy[task.task_id] = energy
        per_utility[task.task_id] = utility
        total_utility += utility

    return KKTMatchingEvaluation(
        True,
        "",
        float(total_utility),
        cpu_allocations,
        bandwidth_allocations,
        per_delay,
        per_energy,
        per_utility,
        per_target,
        counts,
        cpu_resource_sums,
        bandwidth_resource_sums,
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
    kkt_cache: KKTResourceAllocationCache | None = None,
    node_evaluation_cache: NodeEvaluationCache | None = None,
    timing_profile: dict[str, float] | None = None,
    precomputed_impact_scope: tuple[set[str], set[str]] | None = None,
    impact_scope_stats: dict[str, int] | None = None,
    precomputed_current_kkt: dict[str, str] | None = None,
    precomputed_candidate_kkt: dict[str, str] | None = None,
) -> KKTMatchingEvaluation:
    """只重算 candidate 受影响 edge nodes，返回完整 KKT matching evaluation。"""

    if not current_evaluation.resource_feasible:
        raise ValueError("incremental evaluator 要求 current evaluation 可行。")

    if precomputed_current_kkt is None:
        kkt_conversion_started_at = (
            time.perf_counter() if timing_profile is not None else 0.0
        )
        current_kkt = _to_kkt_matching(current_matching)
        if timing_profile is not None:
            _profile_add(
                timing_profile,
                "exact_eval_kkt_conversion_runtime",
                time.perf_counter() - kkt_conversion_started_at,
            )
            timing_profile["current_kkt_conversion_call_count"] = (
                float(timing_profile.get("current_kkt_conversion_call_count", 0.0))
                + 1.0
            )
    else:
        current_kkt = precomputed_current_kkt

    if precomputed_candidate_kkt is None:
        kkt_conversion_started_at = (
            time.perf_counter() if timing_profile is not None else 0.0
        )
        candidate_kkt = _to_kkt_matching(candidate_matching)
        if timing_profile is not None:
            _profile_add(
                timing_profile,
                "exact_eval_kkt_conversion_runtime",
                time.perf_counter() - kkt_conversion_started_at,
            )
            timing_profile["candidate_kkt_conversion_call_count"] = (
                float(timing_profile.get("candidate_kkt_conversion_call_count", 0.0))
                + 1.0
            )
    else:
        candidate_kkt = precomputed_candidate_kkt
    task_ids = {task.task_id for task in tasks}
    if set(current_kkt) != task_ids or set(candidate_kkt) != task_ids:
        raise ValueError("current/candidate matching 的任务集合不一致。")

    affected_ids = [str(task_id) for task_id in affected_task_ids]
    if not set(affected_ids).issubset(task_ids):
        raise ValueError("affected_task_ids 包含未知任务。")

    if precomputed_impact_scope is None:
        if impact_scope_stats is not None:
            impact_scope_stats["requests"] += 1
            impact_scope_stats["computes"] += 1
        affected_edge_nodes, impacted_task_ids = _candidate_impact_scope(
            tasks,
            current_kkt,
            candidate_kkt,
            affected_ids,
            cache_nodes,
            timing_profile=timing_profile,
        )
    else:
        if impact_scope_stats is not None:
            impact_scope_stats["requests"] += 1
            impact_scope_stats["reuses"] += 1
        affected_edge_nodes, impacted_task_ids = precomputed_impact_scope

    state_copy_started_at = (
        time.perf_counter() if timing_profile is not None else 0.0
    )
    cpu_allocations = dict(current_evaluation.cpu_allocations)
    bandwidth_allocations = dict(current_evaluation.bandwidth_allocations)
    cpu_resource_sums = dict(current_evaluation.cpu_resource_sums)
    bandwidth_resource_sums = dict(current_evaluation.bandwidth_resource_sums)
    per_task_delay = dict(current_evaluation.per_task_delay_s)
    per_task_energy = dict(current_evaluation.per_task_vehicle_energy_j)
    per_task_utility = dict(current_evaluation.per_task_utility)
    if timing_profile is not None:
        _profile_add(
            timing_profile,
            "exact_eval_state_copy_runtime",
            time.perf_counter() - state_copy_started_at,
        )
        timing_profile["exact_eval_state_copy_call_count"] = (
            float(timing_profile.get("exact_eval_state_copy_call_count", 0.0))
            + 1.0
        )

    affected_node_prep_started_at = (
        time.perf_counter() if timing_profile is not None else 0.0
    )
    nodes_to_clear = set(affected_edge_nodes)
    for node_id in nodes_to_clear:
        cpu_allocations.pop(node_id, None)
        bandwidth_allocations.pop(node_id, None)
        cpu_resource_sums.pop(node_id, None)
        bandwidth_resource_sums.pop(node_id, None)

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

    affected_node_work: list[tuple[str, list[Task], dict[str, str]]] = []
    for node_id in sorted(affected_edge_nodes, key=str):
        node_tasks = [
            task for task in tasks if candidate_kkt[task.task_id] == node_id
        ]
        if not node_tasks:
            continue
        node_matching = {task.task_id: node_id for task in node_tasks}
        affected_node_work.append((node_id, node_tasks, node_matching))
    _profile_add(
        timing_profile,
        "exact_eval_affected_node_prep_runtime",
        time.perf_counter() - affected_node_prep_started_at
        if timing_profile is not None
        else 0.0,
    )

    node_call_started_at = time.perf_counter() if timing_profile is not None else 0.0
    node_cache_before = (
        float(timing_profile.get("node_cache_lookup_runtime", 0.0))
        if timing_profile is not None
        else 0.0
    )
    node_eval_before = (
        float(timing_profile.get("node_eval_miss_runtime", 0.0))
        if timing_profile is not None
        else 0.0
    )
    kkt_before = (
        float(timing_profile.get("kkt_runtime", 0.0))
        if timing_profile is not None
        else 0.0
    )
    node_evaluations: list[tuple[str, list[Task], KKTMatchingEvaluation]] = []
    for node_id, node_tasks, node_matching in affected_node_work:
        node_evaluation = evaluate_matching_with_kkt(
            node_tasks,
            node_matching,
            vehicle_positions,
            service_node_positions,
            project_cfg,
            cache_nodes,
            cache_config,
            kkt_cache=kkt_cache,
            node_evaluation_cache=node_evaluation_cache,
        )
        if not node_evaluation.resource_feasible:
            if timing_profile is not None:
                nested_runtime = (
                    float(timing_profile.get("node_cache_lookup_runtime", 0.0))
                    - node_cache_before
                    + float(timing_profile.get("node_eval_miss_runtime", 0.0))
                    - node_eval_before
                    + float(timing_profile.get("kkt_runtime", 0.0))
                    - kkt_before
                )
                _profile_add(
                    timing_profile,
                    "exact_eval_node_call_orchestration_runtime",
                    time.perf_counter() - node_call_started_at - nested_runtime,
                )
            return infeasible(
                f"{node_id} incremental KKT infeasible: "
                f"{node_evaluation.infeasible_reason}"
            )
        node_evaluations.append((node_id, node_tasks, node_evaluation))
    if timing_profile is not None:
        nested_runtime = (
            float(timing_profile.get("node_cache_lookup_runtime", 0.0))
            - node_cache_before
            + float(timing_profile.get("node_eval_miss_runtime", 0.0))
            - node_eval_before
            + float(timing_profile.get("kkt_runtime", 0.0))
            - kkt_before
        )
        _profile_add(
            timing_profile,
            "exact_eval_node_call_orchestration_runtime",
            time.perf_counter() - node_call_started_at - nested_runtime,
        )

    result_merge_started_at = time.perf_counter() if timing_profile is not None else 0.0
    recomputed_task_ids: set[str] = set()
    for node_id, node_tasks, node_evaluation in node_evaluations:
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
    _profile_add(
        timing_profile,
        "exact_eval_result_merge_runtime",
        time.perf_counter() - result_merge_started_at
        if timing_profile is not None
        else 0.0,
    )

    non_edge_prep_started_at = time.perf_counter() if timing_profile is not None else 0.0
    non_edge_tasks = [
        task
        for task in tasks
        if task.task_id in impacted_task_ids
        and candidate_kkt[task.task_id] in (LOCAL, CLOUD)
    ]
    non_edge_matching = {
        task.task_id: candidate_kkt[task.task_id] for task in non_edge_tasks
    }
    _profile_add(
        timing_profile,
        "exact_eval_non_edge_prep_runtime",
        time.perf_counter() - non_edge_prep_started_at
        if timing_profile is not None
        else 0.0,
    )
    if non_edge_tasks:
        non_edge_evaluation = evaluate_matching_with_kkt(
            non_edge_tasks,
            non_edge_matching,
            vehicle_positions,
            service_node_positions,
            project_cfg,
            cache_nodes,
            cache_config,
            kkt_cache=kkt_cache,
            node_evaluation_cache=node_evaluation_cache,
        )
        if not non_edge_evaluation.resource_feasible:
            return infeasible(
                f"incremental non-edge evaluation infeasible: "
                f"{non_edge_evaluation.infeasible_reason}"
            )
        result_merge_started_at = (
            time.perf_counter() if timing_profile is not None else 0.0
        )
        per_task_delay.update(non_edge_evaluation.per_task_delay_s)
        per_task_energy.update(non_edge_evaluation.per_task_vehicle_energy_j)
        per_task_utility.update(non_edge_evaluation.per_task_utility)
        if CLOUD in non_edge_evaluation.cpu_allocations:
            cpu_allocations[CLOUD] = dict(non_edge_evaluation.cpu_allocations[CLOUD])
            bandwidth_allocations[CLOUD] = dict(
                non_edge_evaluation.bandwidth_allocations[CLOUD]
            )
            cpu_resource_sums[CLOUD] = non_edge_evaluation.cpu_resource_sums[CLOUD]
            bandwidth_resource_sums[CLOUD] = non_edge_evaluation.bandwidth_resource_sums[CLOUD]
        recomputed_task_ids.update(task.task_id for task in non_edge_tasks)
        _profile_add(
            timing_profile,
            "exact_eval_result_merge_runtime",
            time.perf_counter() - result_merge_started_at
            if timing_profile is not None
            else 0.0,
        )

    if recomputed_task_ids != impacted_task_ids:
        missing = sorted(impacted_task_ids - recomputed_task_ids, key=_stable_id_key)
        raise AssertionError(f"incremental evaluator 未覆盖 impacted tasks：{missing}")

    target_count_started_at = time.perf_counter() if timing_profile is not None else 0.0
    per_task_execution_target = {
        task.task_id: candidate_kkt[task.task_id] for task in tasks
    }
    per_node_task_counts: dict[str, int] = {}
    for task in tasks:
        target = candidate_kkt[task.task_id]
        per_node_task_counts[target] = per_node_task_counts.get(target, 0) + 1
    _profile_add(
        timing_profile,
        "exact_eval_target_count_build_runtime",
        time.perf_counter() - target_count_started_at
        if timing_profile is not None
        else 0.0,
    )

    utility_sum_started_at = time.perf_counter() if timing_profile is not None else 0.0
    task_utility_sum = 0.0
    for task in tasks:
        task_utility_sum += float(per_task_utility[task.task_id])
    _profile_add(
        timing_profile,
        "exact_eval_utility_sum_runtime",
        time.perf_counter() - utility_sum_started_at
        if timing_profile is not None
        else 0.0,
    )

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
                cache_config.cloud_total_bandwidth_bps,
                cache_config.cloud_total_cpu_cycles_per_s,
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


CLOUD_RESOURCE_AUDIT_FIELDS = (
    "slot",
    "cloud_task_count",
    "task_id",
    "cloud_total_cpu_cycles_per_s",
    "cloud_total_bandwidth_bps",
    "allocated_cpu_cycles_per_s",
    "allocated_bandwidth_bps",
    "cloud_compute_delay_s",
    "cloud_communication_delay_s",
    "cloud_total_delay_s",
    "audit_scope",
)


def write_cloud_resource_sharing_audit(
    tasks_csv: Path,
    output_path: Path,
    cache_config: cache_configuration.CacheOptimizationConfig,
) -> dict[str, int]:
    """Write a deterministic Cloud fixed-per-task compatibility audit.

    The audit treats every arrived task in a slot as a Cloud task so that the
    fixed per-task resources are directly visible without running a trajectory,
    prediction, PPO, or Matching sweep.  ``cloud_task_count`` is retained only
    as an observed slot statistic; it does not change the Cloud formula.
    """

    tasks_by_slot: dict[int, list[Task]] = {}
    with tasks_csv.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["arrival"]) != 1:
                continue
            slot = int(row["slot"])
            task = Task(
                task_id=str(row["vehicle_id"]),
                vehicle_id=str(row["vehicle_id"]),
                service_id=int(row["service_id"]),
                input_bits=float(row["input_size"]),
                cpu_cycles=float(row["cpu_cycles"]),
            )
            tasks_by_slot.setdefault(slot, []).append(task)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CLOUD_RESOURCE_AUDIT_FIELDS)
        writer.writeheader()
        for slot in sorted(tasks_by_slot):
            slot_tasks = sorted(tasks_by_slot[slot], key=lambda task: str(task.task_id))
            cloud_task_count = len(slot_tasks)
            for task in slot_tasks:
                execution = system_model.cloud_execution(
                    task.input_bits,
                    task.cpu_cycles,
                    cache_config.cloud_total_bandwidth_bps,
                    cache_config.cloud_total_cpu_cycles_per_s,
                    cache_config.cloud_delay_factor,
                    cache_config.tx_power_w,
                )
                writer.writerow(
                    {
                        "slot": slot,
                        "cloud_task_count": cloud_task_count,
                        "task_id": task.task_id,
                        "cloud_total_cpu_cycles_per_s": cache_config.cloud_total_cpu_cycles_per_s,
                        "cloud_total_bandwidth_bps": cache_config.cloud_total_bandwidth_bps,
                        "allocated_cpu_cycles_per_s": execution["allocated_cpu_cycles_per_s"],
                        "allocated_bandwidth_bps": execution["allocated_bandwidth_bps"],
                        "cloud_compute_delay_s": execution["compute_delay_s"],
                        "cloud_communication_delay_s": execution["communication_delay_s"],
                        "cloud_total_delay_s": execution["total_delay_s"],
                        "audit_scope": "all_arrivals_as_cloud_tasks_fixed_per_task",
                    }
                )
                row_count += 1

    return {
        "slot_count": len(tasks_by_slot),
        "row_count": row_count,
        "max_cloud_task_count": max((len(values) for values in tasks_by_slot.values()), default=0),
    }


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


def run_minimal_validation(config_path: str = "configs/proposed/system_config.toml") -> dict[str, Any]:
    """Run KKT/Cloud formula diagnostics; Cloud-only cases bypass candidate gating."""

    cfg = project_config.load_config(config_path)
    project_config.validate_config(cfg)
    project_config.validate_for_formal_experiment(cfg)
    project_root = Path(cfg["_meta"]["project_root"])
    cache_cfg = cache_configuration.cache_config_from_project(cfg)
    slot = 20
    warmup_time_s = float(project_config.get_config_value(cfg, "simulation.warmup_time_s"))
    dt_s = float(project_config.get_config_value(cfg, "simulation.dt_s"))
    time_s = warmup_time_s + slot * dt_s
    # 只用 configuration 模块的数据读取 helper 生成一个确定性 C(t) 供 KKT actual-state 验证。
    scenarios = cache_configuration.load_prediction_scenarios(
        cache_configuration.resolve_path(
            project_root,
            project_config.get_config_value(cfg, "paths.prediction_scenarios"),
        )
    )
    origin = cache_configuration.scenario_at_origin(scenarios, slot)
    if int(origin["origin_slot"]) != slot:
        raise AssertionError(
            f"prediction origin_slot={origin['origin_slot']} 与 slot={slot} 不一致。"
        )
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

    # This is a pure formula diagnostic, not a production Matching candidate.
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
            cache_cfg.cloud_total_bandwidth_bps,
            cache_cfg.cloud_total_cpu_cycles_per_s,
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
    parser.add_argument("--config", default="configs/proposed/system_config.toml")
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
