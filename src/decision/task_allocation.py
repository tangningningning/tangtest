#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Actual-State Transfer-Exchange Matching.

本模块只做实际时隙内的任务 -> 执行节点 matching。每个 candidate matching
默认 candidate 路径使用 affected-node incremental evaluator；
``evaluate_matching_candidate()`` 保留为 full-recompute reference，二者都求
KKT 连续资源分配和 ``J(mu)=sum_v Q_v``。本模块不实现 PPO、不扣 Cache/UAV configuration cost、
不读取 prediction_scenarios 作为调度输入；每个 slot 的任务独立完成评价。

TV 不执行任务。正式 Matching 目标仅包含满足缓存与链路约束的 Edge 节点以及
Cloud 后备路径。
"""

from __future__ import annotations

import csv
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
from decision import cache_configuration

LEGACY_EDGE_GATE = "LEGACY_EDGE_GATE"
RESOURCE_AWARE_FALLBACK = "RESOURCE_AWARE_FALLBACK"
LEGACY_CLOUD_RESOURCE_MODE = "legacy_unlimited"
FINITE_CLOUD_RESOURCE_MODE = "finite_pool"
_CLOUD_FALLBACK_MODES = frozenset(
    {LEGACY_EDGE_GATE, RESOURCE_AWARE_FALLBACK}
)



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
    cloud_cpu_allocations_cycles_per_s: dict[str, float] = field(default_factory=dict)
    cloud_cpu_capacity_cycles_per_s: float | None = None
    cloud_cpu_used_cycles_per_s: float = 0.0
    tie_break_count: int = 0


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
    edge_cache_state: np.ndarray,
    cache_nodes: dict[str, Any],
    vehicle_positions: dict[str, np.ndarray] | None = None,
    service_node_positions: dict[str, np.ndarray] | None = None,
    cache_config: cache_configuration.CacheOptimizationConfig | None = None,
    timing_profile: dict[str, float] | None = None,
    cloud_fallback_mode: str = LEGACY_EDGE_GATE,
) -> dict[str, list[str]]:
    """按缓存可用性和当前物理路径构造 N_v^fea。

    正式实际/预测调用必须提供 vehicle_positions、service_node_positions 和
    cache_config；若某个缓存边缘节点缺少当前位置、角色、正带宽/CPU 或正信道
    速率，则该节点不进入 M_edge。LEGACY_EDGE_GATE 仅在没有合法 Edge 时保留
    Cloud；RESOURCE_AWARE_FALLBACK 始终保留 Cloud 与全部合法 Edge 候选。省略
    物理上下文时，此函数只检查 cache node 的结构存在性。
    """

    if cloud_fallback_mode not in _CLOUD_FALLBACK_MODES:
        raise ValueError(f"未知 Cloud 候选模式：{cloud_fallback_mode}")
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
        if cloud_fallback_mode == RESOURCE_AWARE_FALLBACK:
            targets.extend(edge_targets)
            targets.append(CLOUD)
        elif edge_targets:
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
    """Full-recompute reference for a legal Edge/Cloud matching."""

    return evaluate_matching_with_kkt(
        tasks,
        matching,
        vehicle_positions,
        service_node_positions,
        project_cfg,
        cache_nodes,
        cache_config,
        kkt_cache=kkt_cache,
        node_evaluation_cache=node_evaluation_cache,
        timing_profile=timing_profile,
    )


def matching_to_execution_variables(
    matching: dict[str, str],
) -> dict[str, dict[str, int]]:
    """导出每个任务对应的 x_{v,n}=1。"""

    result: dict[str, dict[str, int]] = {}
    for task_id in _sorted_ids(matching):
        target = matching[task_id]
        result[task_id] = {target: 1}
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
    prefer_edge_on_tie: bool = False,
) -> dict[str, Any]:
    if current_best is None:
        return candidate
    if candidate["delta_J"] > current_best["delta_J"] + tie_tolerance:
        return candidate
    if abs(candidate["delta_J"] - current_best["delta_J"]) <= tie_tolerance:
        if prefer_edge_on_tie:
            candidate_cloud_targets = sum(
                target == CLOUD for target in candidate["matching"].values()
            )
            current_cloud_targets = sum(
                target == CLOUD for target in current_best["matching"].values()
            )
            if candidate_cloud_targets != current_cloud_targets:
                return (
                    candidate
                    if candidate_cloud_targets < current_cloud_targets
                    else current_best
                )
        if _operation_key(candidate) < _operation_key(current_best):
            return candidate
    return current_best


def _candidate_impact_scope(
    tasks: list[Task],
    current_kkt: dict[str, str],
    candidate_kkt: dict[str, str],
    affected_task_ids: list[str],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig | None = None,
    timing_profile: dict[str, float] | None = None,
) -> tuple[set[str], set[str]]:
    """Return the exact affected edge nodes and impacted task IDs.

    This is shared by strict candidate screening and the incremental evaluator;
    the scope must include every task sharing a current/candidate Edge node and,
    for a finite Cloud pool, every task in the old or new Cloud allocation set.
    """

    started_at = time.perf_counter() if timing_profile is not None else 0.0
    affected_edge_nodes: set[str] = set()
    for task_id in affected_task_ids:
        for target in (current_kkt[task_id], candidate_kkt[task_id]):
            if target == CLOUD:
                # Cloud task coupling, when enabled, is added as a complete pool
                # set below; Cloud is not an Edge node in this node scope.
                continue
            _edge_node_role(target, cache_nodes)
            affected_edge_nodes.add(target)

    impacted_task_ids = set(affected_task_ids)
    if (
        cache_config is not None
        and cache_config.cloud_resource_mode == FINITE_CLOUD_RESOURCE_MODE
    ):
        current_cloud_tasks = {
            task.task_id
            for task in tasks
            if current_kkt[task.task_id] == CLOUD
        }
        candidate_cloud_tasks = {
            task.task_id
            for task in tasks
            if candidate_kkt[task.task_id] == CLOUD
        }
        if current_cloud_tasks != candidate_cloud_tasks:
            impacted_task_ids.update(current_cloud_tasks | candidate_cloud_tasks)
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
    prefer_edge_on_tie: bool = False,
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
        # Matching and KKT now use the same legal Edge/Cloud target IDs.
        # Keep a separate mapping snapshot for incremental evaluation state.
        current_kkt = dict(current_matching)
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

    tie_tolerance = tolerance if prefer_edge_on_tie else tolerance * 0.1
    if evaluation_mode == "incremental":
        if current_kkt is None:
            raise AssertionError("incremental candidate screening requires current KKT matching")
        # Calculate the existing strict UB exactly once per candidate. This
        # pass is separate from both execution order and semantic selection.
        for metadata in candidate_metadata:
            stats["upper_bound_screen_calls"] += 1
            candidate_matching = metadata["matching"]
            candidate_kkt = dict(candidate_matching)
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
                cache_config=cache_config,
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
    current_cloud_count = sum(
        target == CLOUD for target in current_matching.values()
    )
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
            "final_cloud_count": sum(
                target == CLOUD for target in metadata["matching"].values()
            ),
        }
        if prefer_edge_on_tie and delta <= tolerance:
            edge_tie_move = (
                abs(delta) <= tolerance
                and candidate["final_cloud_count"] < current_cloud_count
            )
            if not edge_tie_move:
                continue
        best = _better_candidate(
            best,
            candidate,
            tie_tolerance,
            prefer_edge_on_tie=prefer_edge_on_tie,
        )
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
        edge_tie_move = (
            prefer_edge_on_tie
            and abs(best["delta_J"]) <= tolerance
            and best["final_cloud_count"] < current_cloud_count
        )
        if not edge_tie_move:
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
    if evaluation.cloud_cpu_capacity_cycles_per_s is not None:
        capacity = float(evaluation.cloud_cpu_capacity_cycles_per_s)
        allocated = float(
            math.fsum(evaluation.cloud_cpu_allocations_cycles_per_s.values())
        )
        absolute_tol = max(capacity * tol, 1.0e-6)
        if (
            not math.isfinite(capacity)
            or capacity <= 0.0
            or not math.isfinite(allocated)
            or allocated > capacity
            or not math.isclose(
                allocated,
                float(evaluation.cloud_cpu_used_cycles_per_s),
                rel_tol=1.0e-12,
                abs_tol=absolute_tol,
            )
        ):
            return False
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in evaluation.cloud_cpu_allocations_cycles_per_s.values()
        ):
            return False
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
    edge_cache_state: np.ndarray,
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    evaluation_mode: str = "incremental",
    kkt_cache: KKTResourceAllocationCache | None = None,
    node_evaluation_cache: NodeEvaluationCache | None = None,
    timing_profile: dict[str, float] | None = None,
    cloud_fallback_mode: str = LEGACY_EDGE_GATE,
) -> TransferExchangeMatchingResult:
    """在当前 slot 的缓存和无线状态下执行稳定匹配。"""

    if evaluation_mode not in {"full", "incremental"}:
        raise ValueError(f"未知 Matching candidate evaluation mode：{evaluation_mode}")
    if cloud_fallback_mode not in _CLOUD_FALLBACK_MODES:
        raise ValueError(f"未知 Cloud 候选模式：{cloud_fallback_mode}")
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
        edge_cache_state,
        cache_nodes,
        vehicle_positions=vehicle_positions,
        service_node_positions=service_node_positions,
        cache_config=cache_config,
        timing_profile=timing_profile,
        cloud_fallback_mode=cloud_fallback_mode,
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
    tie_break_count = 0
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
            prefer_edge_on_tie=(
                cloud_fallback_mode == RESOURCE_AWARE_FALLBACK
            ),
        )
        for key, value in stats.items():
            counters[key] += value
        current_cloud_count = sum(
            target == CLOUD for target in current_matching.values()
        )
        edge_tie_update = (
            best is not None
            and cloud_fallback_mode == RESOURCE_AWARE_FALLBACK
            and abs(float(best["delta_J"])) <= tolerance
            and sum(
                target == CLOUD for target in best["matching"].values()
            ) < current_cloud_count
        )
        if best is None or (
            best["delta_J"] <= tolerance and not edge_tie_update
        ):
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
                "tie_break": bool(edge_tie_update),
                "cloud_count_before": int(current_cloud_count),
                "cloud_count_after": int(
                    sum(target == CLOUD for target in best["matching"].values())
                ),
            }
        )
        tie_break_count += int(edge_tie_update)
        j_history.append(float(current_eval.task_utility_sum))
        if best["operation_type"] == "transfer":
            transfer_updates += 1
        else:
            exchange_updates += 1

    transfer_exchange_stable = stop_reason == "converged_by_no_improvement"
    stable_check_passed = False
    max_remaining_delta = 0.0
    if transfer_exchange_stable:
        check_best, check_stats, max_remaining_delta = enumerate_best_operation(
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
            prefer_edge_on_tie=(
                cloud_fallback_mode == RESOURCE_AWARE_FALLBACK
            ),
        )
        # 复检也是候选评价的一部分，便于复杂度记录。
        for key, value in check_stats.items():
            counters[key] += value
        stable_check_passed = check_best is None

    if cloud_fallback_mode == RESOURCE_AWARE_FALLBACK:
        for item in history:
            delta = float(item["delta_J"])
            if item["tie_break"]:
                if (
                    abs(delta) > tolerance
                    or int(item["cloud_count_after"])
                    >= int(item["cloud_count_before"])
                ):
                    raise AssertionError("非法Edge平局优先更新。")
            elif delta <= tolerance:
                raise AssertionError("非平局Matching更新未超过目标容差。")
    elif any(
        j_history[index + 1] <= j_history[index] + tolerance
        for index in range(len(j_history) - 1)
    ):
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
        cloud_cpu_allocations_cycles_per_s=(
            current_eval.cloud_cpu_allocations_cycles_per_s
        ),
        cloud_cpu_capacity_cycles_per_s=current_eval.cloud_cpu_capacity_cycles_per_s,
        cloud_cpu_used_cycles_per_s=current_eval.cloud_cpu_used_cycles_per_s,
        tie_break_count=tie_break_count,
    )











# ---- KKT actual-state resource allocation ----
# KKT actual-state resource allocation implementation；数学求解保持不变。

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


def load_actual_tasks(tasks_csv: Path, slot: int) -> list[Task]:
    """Load arrived tasks for one formal episode slot."""
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


def load_vehicle_positions(
    trajectory_csv: Path,
    time_s: float,
    vehicle_ids: set[str],
) -> dict[str, np.ndarray]:
    """Load exact-time vehicle positions needed by formal rollout execution."""
    positions: dict[str, np.ndarray] = {}
    with trajectory_csv.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            vehicle_id = str(row["vehicle_id"])
            if vehicle_id not in vehicle_ids:
                continue
            if not math.isclose(
                float(row["time_s"]), time_s, rel_tol=0.0, abs_tol=1e-9
            ):
                continue
            positions[vehicle_id] = np.asarray(
                [float(row["x_m"]), float(row["y_m"])], dtype=np.float64
            )
    missing = sorted(vehicle_ids - set(positions))
    if missing:
        raise ValueError(
            f"trajectory.csv 缺少当前 time_s={time_s} 的车辆位置：{missing}"
        )
    return positions


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
    cloud_cpu_allocations_cycles_per_s: dict[str, float] = field(default_factory=dict)
    cloud_cpu_capacity_cycles_per_s: float | None = None
    cloud_cpu_used_cycles_per_s: float = 0.0


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
    if node_id == CLOUD:
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
            noise_model=cache_config.noise_model,
            noise_psd_w_per_hz=cache_config.noise_psd_w_per_hz,
            receiver_noise_figure_db=cache_config.receiver_noise_figure_db,
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
            noise_model=cache_config.noise_model,
            noise_psd_w_per_hz=cache_config.noise_psd_w_per_hz,
            receiver_noise_figure_db=cache_config.receiver_noise_figure_db,
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

    Cloud uses its exact formal per-task model. Edge targets use CPU/BW
    fractions of one only for an upper bound; this helper never calls the KKT
    allocator.
    """

    started_at = time.perf_counter() if timing_profile is not None else 0.0
    if target == CLOUD:
        cloud_cpu_upper_bound = (
            cache_config.cloud_cpu_capacity_cycles_per_s
            if cache_config.cloud_resource_mode == FINITE_CLOUD_RESOURCE_MODE
            else cache_config.cloud_total_cpu_cycles_per_s
        )
        result = system_model.cloud_execution(
            task.input_bits,
            task.cpu_cycles,
            cache_config.cloud_total_bandwidth_bps,
            cloud_cpu_upper_bound,
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
            noise_model=cache_config.noise_model,
            noise_psd_w_per_hz=cache_config.noise_psd_w_per_hz,
            receiver_noise_figure_db=cache_config.receiver_noise_figure_db,
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


def _solve_bandwidth_v1_kkt_python_core(
    coefficients: np.ndarray,
    channel_gains: np.ndarray,
    bandwidth_hz: float,
    transmit_power_w: float,
    noise_psd_w_per_hz: float,
    receiver_noise_figure_db: float,
    lower_marginal: np.ndarray,
    upper_marginal: np.ndarray,
    epsilon: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, float, float, int, bool]:
    """Original NumPy bisection path used as reference and Numba fallback."""

    def allocation_for(multiplier: float) -> np.ndarray:
        result = np.empty(coefficients.size, dtype=np.float64)
        at_lower = multiplier >= lower_marginal
        at_upper = multiplier <= upper_marginal
        result[at_lower] = epsilon
        result[at_upper & ~at_lower] = 1.0
        interior = ~(at_lower | at_upper)
        if not np.any(interior):
            return result
        lo = np.full(int(np.count_nonzero(interior)), epsilon, dtype=np.float64)
        hi = np.ones_like(lo)
        target = np.full_like(lo, multiplier)
        interior_coeff = coefficients[interior]
        interior_gains = channel_gains[interior]
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            rates, derivatives, _ = system_model.uplink_rate_and_derivative(
                bandwidth_hz,
                mid,
                transmit_power_w,
                interior_gains,
                1.0,
                noise_model=system_model.BANDWIDTH_DEPENDENT_NOISE_V1,
                noise_psd_w_per_hz=noise_psd_w_per_hz,
                receiver_noise_figure_db=receiver_noise_figure_db,
            )
            rates_array = np.asarray(rates, dtype=np.float64)
            derivatives_array = np.asarray(derivatives, dtype=np.float64)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                current = interior_coeff * derivatives_array / np.square(rates_array)
            needs_more = current > target
            lo = np.where(needs_more, mid, lo)
            hi = np.where(needs_more, hi, mid)
        result[interior] = 0.5 * (lo + hi)
        return result

    low_lambda = 0.0
    high_lambda = float(np.max(lower_marginal))
    allocation = np.full(coefficients.size, epsilon, dtype=np.float64)
    multiplier = high_lambda
    total = float(np.sum(allocation))
    iterations = 0
    converged = False
    for iterations in range(1, max_iterations + 1):
        multiplier = 0.5 * (low_lambda + high_lambda)
        allocation = allocation_for(multiplier)
        total = float(np.sum(allocation))
        if abs(total - 1.0) <= tolerance:
            converged = True
            break
        if total > 1.0:
            low_lambda = multiplier
        else:
            high_lambda = multiplier
    return allocation, multiplier, total, iterations, converged


def _bandwidth_v1_kkt_numba_core_impl(
    coefficients: np.ndarray,
    a_values: np.ndarray,
    lower_marginal: np.ndarray,
    upper_marginal: np.ndarray,
    bandwidth_hz: float,
    epsilon: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, float, float, int, bool]:
    """Pure scalar implementation of the existing V1 bandwidth bisections."""

    task_count = coefficients.size
    allocation = np.empty(task_count, dtype=np.float64)
    low_lambda = 0.0
    high_lambda = float(np.max(lower_marginal))
    multiplier = high_lambda
    total = float(task_count) * epsilon
    iterations = 0
    converged = False
    log_two = math.log(2.0)

    for iterations in range(1, max_iterations + 1):
        multiplier = 0.5 * (low_lambda + high_lambda)
        for task_index in range(task_count):
            if multiplier >= lower_marginal[task_index]:
                task_allocation = epsilon
            elif multiplier <= upper_marginal[task_index]:
                task_allocation = 1.0
            else:
                low_allocation = epsilon
                high_allocation = 1.0
                a_value = a_values[task_index]
                coefficient = coefficients[task_index]
                for _ in range(64):
                    mid = 0.5 * (low_allocation + high_allocation)
                    allocated_bandwidth = bandwidth_hz * mid
                    ratio = a_value / allocated_bandwidth
                    log_term = np.log1p(ratio)
                    rate = allocated_bandwidth * log_term / log_two
                    derivative = bandwidth_hz / log_two * (
                        log_term - a_value / (allocated_bandwidth + a_value)
                    )
                    current = coefficient * derivative / (rate * rate)
                    if current > multiplier:
                        low_allocation = mid
                    else:
                        high_allocation = mid
                task_allocation = 0.5 * (low_allocation + high_allocation)
            allocation[task_index] = task_allocation
        total = float(np.sum(allocation))

        if abs(total - 1.0) <= tolerance:
            converged = True
            break
        if total > 1.0:
            low_lambda = multiplier
        else:
            high_lambda = multiplier
    return allocation, multiplier, total, iterations, converged


if numba is not None:
    _solve_bandwidth_v1_kkt_numba_core = numba.njit(
        cache=False,
        fastmath=False,
    )(_bandwidth_v1_kkt_numba_core_impl)
else:  # pragma: no cover - optional dependency fallback
    _solve_bandwidth_v1_kkt_numba_core = None


def _solve_kkt_bandwidth_dependent_noise_allocation_impl(
    coefficients: list[float],
    bandwidth_hz: float,
    transmit_power_w: float,
    channel_gains: list[float],
    noise_psd_w_per_hz: float,
    receiver_noise_figure_db: float,
    epsilon: float,
    tolerance: float,
    max_iterations: int,
    *,
    return_diagnostics: bool = False,
    _use_numba: bool = True,
) -> Any:
    """Solve the V1 bandwidth KKT using the exact rate derivative.

    For task ``i`` the bandwidth-dependent part of the normalized system loss
    is ``c_i / R_i(b_i)``.  Its KKT stationarity is
    ``c_i R_i'(b_i) / R_i(b_i)^2 = lambda`` for interior allocations, where
    ``R_i`` and ``R_i'`` come from the shared system-model implementation.
    """

    coeff = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    gains = np.asarray(channel_gains, dtype=np.float64).reshape(-1)
    bandwidth = float(bandwidth_hz)
    power = float(transmit_power_w)
    psd = float(noise_psd_w_per_hz)
    nf_db = float(receiver_noise_figure_db)
    epsilon = float(epsilon)
    tolerance = float(tolerance)
    max_iterations = int(max_iterations)
    n = int(coeff.size)

    def finish(
        allocation: np.ndarray | list[float],
        multiplier: float,
        ok: bool,
        reason: str,
        iterations: int,
        converged: bool,
        hit_max: bool,
        total: float,
        stationarity: float = math.nan,
        complementary: float = math.nan,
    ) -> Any:
        values = np.asarray(allocation, dtype=np.float64).reshape(-1)
        finite = bool(
            math.isfinite(float(multiplier))
            and math.isfinite(float(total))
            and np.all(np.isfinite(values))
            and math.isfinite(float(stationarity))
            and math.isfinite(float(complementary))
        )
        result = (
            [float(value) for value in values],
            float(multiplier),
            bool(ok),
            str(reason),
        )
        if not return_diagnostics:
            return result
        diagnostics = KKTAllocationDiagnostics(
            iterations=int(iterations),
            converged_by_tolerance=bool(converged),
            hit_max_bisection_iterations=bool(hit_max),
            final_resource_sum=float(total),
            resource_sum_residual=float(total - 1.0) if n else 0.0,
            finite=finite,
            stationarity_residual=float(stationarity),
            complementary_slackness_residual=float(complementary),
        )
        return result + (diagnostics,)

    if n == 0:
        return finish([], 0.0, True, "", 0, True, False, 0.0, 0.0, 0.0)
    if (
        not math.isfinite(bandwidth)
        or bandwidth <= 0.0
        or not math.isfinite(power)
        or power < 0.0
        or not math.isfinite(psd)
        or psd <= 0.0
        or not math.isfinite(nf_db)
        or nf_db < 0.0
        or not math.isfinite(epsilon)
        or not 0.0 < epsilon <= 1.0
        or not math.isfinite(tolerance)
        or tolerance <= 0.0
        or max_iterations < 1
        or gains.shape != coeff.shape
        or np.any(~np.isfinite(coeff))
        or np.any(coeff < 0.0)
        or np.any(~np.isfinite(gains))
        or np.any(gains < 0.0)
    ):
        return finish([], math.nan, False, "invalid V1 bandwidth-KKT input", 0, False, False, math.nan)
    if epsilon * n > 1.0 + tolerance:
        return finish(
            [], math.nan, False, f"epsilon * N = {epsilon * n} > 1",
            0, False, False, 0.0,
        )
    if n == 1:
        return finish([1.0], 0.0, True, "", 0, True, False, 1.0, 0.0, 0.0)
    if not bool(np.any(coeff > 0.0)):
        allocation = np.full(n, 1.0 / n, dtype=np.float64)
        return finish(allocation, 0.0, True, "", 0, True, False, 1.0, 0.0, 0.0)

    def marginal(allocation: np.ndarray) -> np.ndarray:
        rates, derivatives, _ = system_model.uplink_rate_and_derivative(
            bandwidth,
            allocation,
            power,
            gains,
            1.0,
            noise_model=system_model.BANDWIDTH_DEPENDENT_NOISE_V1,
            noise_psd_w_per_hz=psd,
            receiver_noise_figure_db=nf_db,
        )
        rates_array = np.asarray(rates, dtype=np.float64)
        derivative_array = np.asarray(derivatives, dtype=np.float64)
        if np.any(rates_array <= 0.0):
            raise ValueError("V1 bandwidth KKT encountered a zero allocated rate")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            values = coeff * derivative_array / np.square(rates_array)
        if np.any(~np.isfinite(values)):
            raise ValueError("V1 bandwidth KKT marginal utility is non-finite")
        return values

    try:
        lower_allocation = np.full(n, epsilon, dtype=np.float64)
        upper_allocation = np.ones(n, dtype=np.float64)
        lower_marginal = marginal(lower_allocation)
        upper_marginal = marginal(upper_allocation)
    except (ValueError, OverflowError, FloatingPointError) as exc:
        return finish([], math.nan, False, str(exc), 0, False, False, 0.0)
    if not np.all(np.isfinite(lower_marginal)):
        return finish([], math.nan, False, "V1 KKT upper multiplier is non-finite", 0, False, False, 0.0)

    if epsilon * n >= 1.0 - tolerance:
        allocation = lower_allocation
        total = float(np.sum(allocation))
        multiplier = float(np.max(lower_marginal))
        stationarity = float(np.max(np.maximum(lower_marginal - multiplier, 0.0)))
        return finish(
            allocation,
            multiplier,
            total <= 1.0 + tolerance,
            "" if total <= 1.0 + tolerance else f"allocation sum {total} > 1",
            0,
            abs(total - 1.0) <= tolerance,
            False,
            total,
            stationarity,
            abs(multiplier * (1.0 - total)),
        )

    high_lambda = float(np.max(lower_marginal))
    if not math.isfinite(high_lambda):
        return finish([], math.nan, False, "V1 KKT multiplier overflow", 0, False, False, 0.0)
    if _use_numba and _solve_bandwidth_v1_kkt_numba_core is not None:
        nf_linear = 10.0 ** (nf_db / 10.0)
        effective_psd = psd * nf_linear
        a_values = power * gains / effective_psd
        try:
            allocation, multiplier, total, iterations, converged = (
                _solve_bandwidth_v1_kkt_numba_core(
                    coeff,
                    a_values,
                    lower_marginal,
                    upper_marginal,
                    bandwidth,
                    epsilon,
                    tolerance,
                    max_iterations,
                )
            )
        except numba.core.errors.NumbaError:
            allocation, multiplier, total, iterations, converged = (
                _solve_bandwidth_v1_kkt_python_core(
                    coeff,
                    gains,
                    bandwidth,
                    power,
                    psd,
                    nf_db,
                    lower_marginal,
                    upper_marginal,
                    epsilon,
                    tolerance,
                    max_iterations,
                )
            )
    else:
        allocation, multiplier, total, iterations, converged = (
            _solve_bandwidth_v1_kkt_python_core(
                coeff,
                gains,
                bandwidth,
                power,
                psd,
                nf_db,
                lower_marginal,
                upper_marginal,
                epsilon,
                tolerance,
                max_iterations,
            )
        )
    hit_max = iterations >= max_iterations and not converged
    if total > 1.0 + max(tolerance * 10.0, 1.0e-10):
        return finish(
            allocation, multiplier, False, f"V1 KKT allocation sum {total} > 1",
            iterations, converged, hit_max, total,
        )
    try:
        final_marginal = marginal(allocation)
    except (ValueError, OverflowError, FloatingPointError) as exc:
        return finish(allocation, multiplier, False, str(exc), iterations, converged, hit_max, total)
    bound_tolerance = max(tolerance * 10.0, 1.0e-12)
    at_lower = allocation <= epsilon + bound_tolerance
    at_upper = allocation >= 1.0 - bound_tolerance
    interior = ~(at_lower | at_upper)
    stationarity_terms = np.zeros(n, dtype=np.float64)
    stationarity_terms[at_lower] = np.maximum(
        final_marginal[at_lower] - multiplier, 0.0
    )
    stationarity_terms[at_upper] = np.maximum(
        multiplier - final_marginal[at_upper], 0.0
    )
    stationarity_terms[interior] = np.abs(
        final_marginal[interior] - multiplier
    )
    lower_dual = np.where(at_lower, np.maximum(multiplier - final_marginal, 0.0), 0.0)
    upper_dual = np.where(at_upper, np.maximum(final_marginal - multiplier, 0.0), 0.0)
    complementary = max(
        abs(multiplier * (1.0 - total)),
        float(np.max(np.abs(lower_dual * (allocation - epsilon)))),
        float(np.max(np.abs(upper_dual * (1.0 - allocation)))),
    )
    return finish(
        allocation,
        multiplier,
        True,
        "",
        iterations,
        converged,
        hit_max,
        total,
        float(np.max(stationarity_terms)),
        complementary,
    )


def solve_kkt_bandwidth_dependent_noise_allocation(
    coefficients: list[float],
    bandwidth_hz: float,
    transmit_power_w: float,
    channel_gains: list[float],
    noise_psd_w_per_hz: float,
    receiver_noise_figure_db: float,
    epsilon: float,
    tolerance: float,
    max_iterations: int,
    *,
    return_diagnostics: bool = False,
) -> Any:
    """Solve the V1 bandwidth KKT with an optional Numba numerical core."""

    return _solve_kkt_bandwidth_dependent_noise_allocation_impl(
        coefficients,
        bandwidth_hz,
        transmit_power_w,
        channel_gains,
        noise_psd_w_per_hz,
        receiver_noise_figure_db,
        epsilon,
        tolerance,
        max_iterations,
        return_diagnostics=return_diagnostics,
        _use_numba=True,
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
                solver_runtime = time.perf_counter() - solver_started_at
                _profile_add(
                    timing_profile,
                    "kkt_edge_cpu_solver_core_runtime",
                    solver_runtime,
                )
                _profile_add(
                    timing_profile,
                    "kkt_solver_core_runtime",
                    solver_runtime,
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
            if cache_config.noise_model == system_model.BANDWIDTH_DEPENDENT_NOISE_V1:
                upload_weight = (
                    cache_config.omega_t / cache_config.time_ref_s
                    + cache_config.omega_e
                    * cache_config.tx_power_w
                    / cache_config.energy_ref_j
                )
                bw_coeff = [task.input_bits * upload_weight for task in node_tasks]
                channel_gains = [
                    _channel_gain(
                        role,
                        vehicle_positions[task.vehicle_id],
                        service_node_positions[node_id],
                        cache_config,
                    )
                    for task in node_tasks
                ]
                if any(not math.isfinite(value) or value <= 0.0 for value in channel_gains):
                    if timing_profile is not None:
                        timing_profile["kkt_coefficient_build_runtime"] += (
                            time.perf_counter() - coefficient_started_at
                        )
                    return KKTResourceAllocationResult(
                        False, f"{node_id} channel gain <= 0", {}, {}, {}, {}, {}, {}
                    )
            else:
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
                    # In V0, R(b)=b*R(1), so the upload loss is coefficient/b.
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
            if cache_config.noise_model == system_model.BANDWIDTH_DEPENDENT_NOISE_V1:
                if cache_config.noise_psd_w_per_hz is None or cache_config.receiver_noise_figure_db is None:
                    return KKTResourceAllocationResult(
                        False,
                        "V1 noise configuration disappeared before KKT allocation",
                        {}, {}, {}, {}, {}, {},
                    )
                bw_values, nu, ok, reason, bw_diagnostics = (
                    solve_kkt_bandwidth_dependent_noise_allocation(
                        bw_coeff,
                        _node_bandwidth(role, cache_config),
                        cache_config.tx_power_w,
                        channel_gains,
                        cache_config.noise_psd_w_per_hz,
                        cache_config.receiver_noise_figure_db,
                        kkt_cfg.min_bandwidth_fraction,
                        kkt_cfg.bisection_tolerance,
                        kkt_cfg.max_bisection_iterations,
                        return_diagnostics=True,
                    )
                )
            else:
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
                solver_runtime = time.perf_counter() - solver_started_at
                _profile_add(
                    timing_profile,
                    "kkt_edge_bandwidth_solver_core_runtime",
                    solver_runtime,
                )
                _profile_add(
                    timing_profile,
                    "kkt_solver_core_runtime",
                    solver_runtime,
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


def _allocate_cloud_cpu_kkt(
    cloud_tasks: list[Task],
    cache_config: cache_configuration.CacheOptimizationConfig,
    project_cfg: dict[str, Any],
    kkt_cache: KKTResourceAllocationCache | None,
    timing_profile: dict[str, float] | None = None,
) -> KKTResourceAllocationResult:
    """Allocate the complete Cloud task set under the configured CPU contract.

    Legacy mode returns the old independent 5 GHz/task contract.  Finite-pool
    mode solves the same normalized convex CPU allocation used by Edge KKT,
    while keeping Cloud uplink rates independent and task-specific.
    """

    if not cloud_tasks:
        return KKTResourceAllocationResult(True, "", {}, {}, {}, {}, {}, {})

    if cache_config.cloud_resource_mode == LEGACY_CLOUD_RESOURCE_MODE:
        allocations = {
            task.task_id: float(cache_config.cloud_total_cpu_cycles_per_s)
            for task in cloud_tasks
        }
        return KKTResourceAllocationResult(
            True,
            "",
            {CLOUD: allocations},
            {},
            {CLOUD: float(len(cloud_tasks))},
            {},
            {},
            {},
        )

    capacity = float(cache_config.cloud_cpu_capacity_cycles_per_s)
    kkt_cfg = kkt_config_from_project(project_cfg)
    if kkt_cfg.min_cpu_fraction * len(cloud_tasks) > 1.0:
        return KKTResourceAllocationResult(
            False,
            "Cloud CPU epsilon infeasible",
            {},
            {},
            {},
            {},
            {},
            {},
        )

    def compute_pool() -> KKTResourceAllocationResult:
        coefficients = [
            cache_config.omega_t
            * task.cpu_cycles
            / (cache_config.time_ref_s * capacity)
            for task in cloud_tasks
        ]
        if kkt_cache is not None:
            kkt_cache.solver_calls += 1
        solver_started = (
            time.perf_counter() if timing_profile is not None else 0.0
        )
        fractions, multiplier, ok, reason, diagnostics = (
            solve_kkt_fraction_allocation(
                coefficients,
                kkt_cfg.min_cpu_fraction,
                kkt_cfg.bisection_tolerance,
                kkt_cfg.max_bisection_iterations,
                return_diagnostics=True,
            )
        )
        if timing_profile is not None:
            solver_runtime = time.perf_counter() - solver_started
            _profile_add(
                timing_profile,
                "kkt_cloud_cpu_solver_core_runtime",
                solver_runtime,
            )
            _profile_add(
                timing_profile,
                "kkt_solver_core_runtime",
                solver_runtime,
            )
        if kkt_cache is not None:
            kkt_cache.record_diagnostics(diagnostics)
        if not ok:
            return KKTResourceAllocationResult(
                False,
                f"Cloud CPU KKT infeasible: {reason}",
                {},
                {},
                {},
                {},
                {},
                {},
                (diagnostics,),
            )
        fraction_values = np.asarray(fractions, dtype=np.float64)
        fraction_total = float(math.fsum(float(value) for value in fraction_values))
        if fraction_total > 1.0:
            fraction_values = fraction_values / fraction_total
        allocation_map = {
            task.task_id: float(fraction) * capacity
            for task, fraction in zip(cloud_tasks, fraction_values, strict=True)
        }
        allocated_total = float(math.fsum(allocation_map.values()))
        if allocated_total > capacity:
            largest_task_id = max(allocation_map, key=allocation_map.__getitem__)
            allocation_map[largest_task_id] -= allocated_total - capacity
            allocated_total = float(math.fsum(allocation_map.values()))
            if allocated_total > capacity:
                allocation_map[largest_task_id] = math.nextafter(
                    allocation_map[largest_task_id], 0.0
                )
                allocated_total = float(math.fsum(allocation_map.values()))
            if allocated_total > capacity:
                return KKTResourceAllocationResult(
                    False,
                    "Cloud CPU allocation rounding exceeded capacity",
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                    (diagnostics,),
                )
        return KKTResourceAllocationResult(
            True,
            "",
            {CLOUD: allocation_map},
            {},
            {CLOUD: float(math.fsum(float(value) for value in fraction_values))},
            {},
            {CLOUD: float(multiplier)},
            {},
            (diagnostics,),
        )

    kkt_started = time.perf_counter() if timing_profile is not None else 0.0
    if kkt_cache is None:
        result = compute_pool()
    else:
        # The Cloud pool cache is keyed by the full sorted Cloud task set.  Its
        # capacity is fixed for the lifetime of this Matching call.
        result = kkt_cache.get_or_compute(
            CLOUD,
            cloud_tasks,
            np.zeros(2, dtype=np.float64),
            compute_pool,
        )
    if timing_profile is not None:
        timing_profile["kkt_runtime"] += time.perf_counter() - kkt_started
    return result


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
            noise_model=cache_config.noise_model,
            noise_psd_w_per_hz=cache_config.noise_psd_w_per_hz,
            receiver_noise_figure_db=cache_config.receiver_noise_figure_db,
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
    timing_profile: dict[str, float] | None = None,
) -> KKTMatchingEvaluation:
    """评价当前 slot 的 matching、KKT 资源、完成时延和车辆能耗。"""

    if not tasks:
        return KKTMatchingEvaluation(
            True,
            "",
            0.0,
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            cloud_cpu_capacity_cycles_per_s=(
                cache_config.cloud_cpu_capacity_cycles_per_s
                if cache_config.cloud_resource_mode == FINITE_CLOUD_RESOURCE_MODE
                else None
            ),
        )

    if node_evaluation_cache is not None:
        if timing_profile is None:
            timing_profile = node_evaluation_cache.timing_profile
        else:
            node_evaluation_cache.timing_profile = timing_profile
    grouped_edge_tasks = _group_edge_tasks(tasks, matching, cache_nodes)
    edge_evaluations: dict[str, KKTMatchingEvaluation] = {}
    for node_id, node_tasks in grouped_edge_tasks.items():
        if node_evaluation_cache is None:
            node_evaluation_started_at = (
                time.perf_counter() if timing_profile is not None else 0.0
            )
            kkt_runtime_before = (
                float(timing_profile.get("kkt_runtime", 0.0))
                if timing_profile is not None
                else 0.0
            )
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
            if timing_profile is not None:
                _profile_add(
                    timing_profile,
                    "node_eval_miss_runtime",
                    time.perf_counter()
                    - node_evaluation_started_at
                    - (
                        float(timing_profile.get("kkt_runtime", 0.0))
                        - kkt_runtime_before
                    ),
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
                resource_feasible=False,
                infeasible_reason=node_evaluation.infeasible_reason,
                task_utility_sum=float("-inf"),
                cpu_allocations=node_evaluation.cpu_allocations,
                bandwidth_allocations=node_evaluation.bandwidth_allocations,
                per_task_delay_s={},
                per_task_vehicle_energy_j={},
                per_task_utility={},
                per_task_execution_target={},
                per_node_task_counts={},
                cpu_resource_sums=node_evaluation.cpu_resource_sums,
                bandwidth_resource_sums=node_evaluation.bandwidth_resource_sums,
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

    cloud_tasks = [task for task in tasks if matching[task.task_id] == CLOUD]
    cloud_kkt_started_at = (
        time.perf_counter() if timing_profile is not None else 0.0
    )
    cloud_allocation = _allocate_cloud_cpu_kkt(
        cloud_tasks,
        cache_config,
        project_cfg,
        kkt_cache,
        timing_profile,
    )
    if timing_profile is not None:
        _profile_add(
            timing_profile,
            "kkt_runtime",
            time.perf_counter() - cloud_kkt_started_at,
        )
    if not cloud_allocation.feasible:
        return KKTMatchingEvaluation(
            resource_feasible=False,
            infeasible_reason=cloud_allocation.infeasible_reason,
            task_utility_sum=float("-inf"),
            cpu_allocations=cpu_allocations,
            bandwidth_allocations=bandwidth_allocations,
            per_task_delay_s={},
            per_task_vehicle_energy_j={},
            per_task_utility={},
            per_task_execution_target={},
            per_node_task_counts={},
            cpu_resource_sums=cpu_resource_sums,
            bandwidth_resource_sums=bandwidth_resource_sums,
        )
    cloud_cpu_allocations = dict(
        cloud_allocation.cpu_allocations.get(CLOUD, {})
    )
    cloud_cpu_capacity = (
        float(cache_config.cloud_cpu_capacity_cycles_per_s)
        if cache_config.cloud_resource_mode == FINITE_CLOUD_RESOURCE_MODE
        else None
    )
    cloud_cpu_used = float(math.fsum(cloud_cpu_allocations.values()))

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
        elif target == CLOUD:
            result = system_model.cloud_execution(
                task.input_bits,
                task.cpu_cycles,
                cache_config.cloud_total_bandwidth_bps,
                cloud_cpu_allocations[task.task_id],
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
        cloud_cpu_allocations,
        cloud_cpu_capacity,
        cloud_cpu_used,
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
        current_kkt = dict(current_matching)
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
        candidate_kkt = dict(candidate_matching)
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
    current_cloud_task_ids = {
        task_id for task_id, target in current_kkt.items() if target == CLOUD
    }
    candidate_cloud_task_ids = {
        task_id for task_id, target in candidate_kkt.items() if target == CLOUD
    }
    finite_cloud_pool_changed = (
        cache_config.cloud_resource_mode == FINITE_CLOUD_RESOURCE_MODE
        and current_cloud_task_ids != candidate_cloud_task_ids
    )

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
            cache_config=cache_config,
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
    cloud_cpu_allocations = dict(
        current_evaluation.cloud_cpu_allocations_cycles_per_s
    )
    cloud_cpu_capacity = current_evaluation.cloud_cpu_capacity_cycles_per_s
    cloud_cpu_used = float(current_evaluation.cloud_cpu_used_cycles_per_s)
    if finite_cloud_pool_changed:
        # The entire pool allocation is coupled: discard the old set so the
        # candidate evaluator replaces it with a fresh allocation over every
        # Cloud task in candidate_matching.
        cloud_cpu_allocations.clear()
        cloud_cpu_capacity = float(cache_config.cloud_cpu_capacity_cycles_per_s)
        cloud_cpu_used = 0.0
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
            timing_profile=timing_profile,
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
        and candidate_kkt[task.task_id] == CLOUD
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
            timing_profile=timing_profile,
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
        if finite_cloud_pool_changed:
            cloud_cpu_allocations = dict(
                non_edge_evaluation.cloud_cpu_allocations_cycles_per_s
            )
            cloud_cpu_capacity = (
                non_edge_evaluation.cloud_cpu_capacity_cycles_per_s
            )
        elif cache_config.cloud_resource_mode == LEGACY_CLOUD_RESOURCE_MODE:
            for task_id in impacted_task_ids:
                cloud_cpu_allocations.pop(task_id, None)
            cloud_cpu_allocations.update(
                non_edge_evaluation.cloud_cpu_allocations_cycles_per_s
            )
        cloud_cpu_used = float(math.fsum(cloud_cpu_allocations.values()))
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
        cloud_cpu_allocations_cycles_per_s=cloud_cpu_allocations,
        cloud_cpu_capacity_cycles_per_s=cloud_cpu_capacity,
        cloud_cpu_used_cycles_per_s=cloud_cpu_used,
    )



















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
