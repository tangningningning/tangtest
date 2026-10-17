"""Alternative task-assignment search with an exact convex-resource response oracle.

This research-only module does not replace the formal Matching/PPO pipeline.
It uses the existing physical task evaluator, but searches discrete executor
assignments independently of transfer-exchange matching.  Cloud uses the
configured finite shared CPU pool; Edge shares CPU and bandwidth.

Search guarantees apply only to independent, NEW, per-slot tasks with fixed
cache/UAV configuration and the presently interference-free channel model.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import Any, Mapping

import numpy as np

from decision import cache_configuration, task_allocation


@dataclass(frozen=True)
class AssignmentSearchResult:
    """Best feasible incumbent and a conservative global utility bound."""

    matching: dict[str, str]
    evaluation: task_allocation.KKTMatchingEvaluation
    feasible_targets: dict[str, list[str]]
    oracle_evaluations: int
    upper_bound_prunes: int
    global_upper_bound: float
    certified_absolute_gap: float
    globally_optimal_within_tolerance: bool
    stop_reason: str

    @property
    def task_utility_sum(self) -> float:
        return float(self.evaluation.task_utility_sum)


class _EvaluationBudgetExceeded(Exception):
    pass


def optimize_assignment_with_resource_oracle(
    tasks: list[task_allocation.Task],
    vehicle_positions: dict[str, np.ndarray],
    service_node_positions: dict[str, np.ndarray],
    edge_cache_state: np.ndarray,
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    *,
    max_oracle_evaluations: int = 512,
    exact_task_limit: int = 6,
    neighborhood_size: int = 3,
    neighborhood_focus: int = 8,
    target_shortlist_size: int = 3,
    max_neighborhood_rounds: int = 2,
    objective_tolerance: float = 1.0e-9,
    initial_matching: Mapping[str, str] | None = None,
) -> AssignmentSearchResult:
    """Optimize executor assignment under the *current* physical model.

    Small sets: complete branch-and-bound until the oracle budget is exhausted.
    Larger sets: deterministic bounded joint-reassignment neighborhoods.
    In either case, the returned matching is feasible and the global utility
    upper bound is conservative, not an empirical estimate.

    One oracle evaluation means a complete conditional CPU/BW optimization,
    possibly spanning multiple Edge nodes plus the finite Cloud CPU pool.
    It is not a limit on individual KKT solver calls or wall-clock time.
    """

    if (
        max_oracle_evaluations < 1
        or exact_task_limit < 0
        or neighborhood_size < 1
        or neighborhood_focus < 1
        or target_shortlist_size < 1
        or max_neighborhood_rounds < 1
        or not isfinite(objective_tolerance)
        or objective_tolerance < 0.0
    ):
        raise ValueError("invalid assignment-search budgets or tolerance")
    if cache_config.cloud_resource_mode != task_allocation.FINITE_CLOUD_RESOURCE_MODE:
        raise ValueError("this research solver requires finite shared Cloud CPU")
    if cache_config.cloud_fallback_mode != task_allocation.RESOURCE_AWARE_FALLBACK:
        raise ValueError("this research solver requires resource-aware Cloud eligibility")

    ordered = sorted(tasks, key=lambda task: str(task.task_id))
    ids = [str(task.task_id) for task in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("task IDs must be unique")
    if any(
        task.execution_stage != "NEW" or task.transmit_time_share != 1.0
        for task in ordered
    ):
        raise ValueError("only independent NEW per-slot tasks are supported")

    feasible = task_allocation.build_feasible_targets(
        ordered,
        edge_cache_state,
        cache_nodes,
        vehicle_positions=vehicle_positions,
        service_node_positions=service_node_positions,
        cache_config=cache_config,
        cloud_fallback_mode=cache_config.cloud_fallback_mode,
    )
    if any(not feasible[tid] for tid in ids):
        raise ValueError("a task has no legal execution targets")

    # A single task receiving all of its node's CPU/bandwidth has optimistic
    # utility. Sharing cannot improve that task's utility when other tasks
    # join the same node (provided channel gains do not depend on assignment).
    optimistic = {
        tid: {
            target: float(task_allocation._ideal_task_utility(
                task, target, vehicle_positions, service_node_positions,
                cache_nodes, cache_config,
            ))
            for target in feasible[tid]
        }
        for tid, task in zip(ids, ordered, strict=True)
    }
    ideal_best = {tid: max(optimistic[tid].values()) for tid in ids}
    root_upper_bound = float(sum(ideal_best.values()))
    task_by_id = {str(task.task_id): task for task in ordered}
    kkt_cache = task_allocation.KKTResourceAllocationCache()
    node_cache = task_allocation.NodeEvaluationCache()
    memo: dict[tuple[tuple[str, str], ...], task_allocation.KKTMatchingEvaluation] = {}
    oracle_evaluations = 0
    upper_bound_prunes = 0

    def evaluate(mapping: Mapping[str, str]) -> task_allocation.KKTMatchingEvaluation:
        nonlocal oracle_evaluations
        key = tuple(sorted((str(tid), str(target)) for tid, target in mapping.items()))
        if key in memo:
            return memo[key]
        if oracle_evaluations >= max_oracle_evaluations:
            raise _EvaluationBudgetExceeded
        selected_tasks = [task_by_id[tid] for tid, _ in key]
        result = task_allocation.evaluate_matching_candidate(
            selected_tasks,
            dict(key),
            vehicle_positions,
            service_node_positions,
            project_cfg,
            cache_nodes,
            cache_config,
            kkt_cache=kkt_cache,
            node_evaluation_cache=node_cache,
        )
        oracle_evaluations += 1
        memo[key] = result
        return result

    if not ids:
        empty = evaluate({})
        return AssignmentSearchResult(
            matching={},
            evaluation=empty,
            feasible_targets={},
            oracle_evaluations=oracle_evaluations,
            upper_bound_prunes=0,
            global_upper_bound=0.0,
            certified_absolute_gap=0.0,
            globally_optimal_within_tolerance=True,
            stop_reason="no_tasks",
        )

    def option_rank(tid: str, target: str) -> tuple[float, bool, str]:
        return (-optimistic[tid][target], target == task_allocation.CLOUD, target)

    # Build an independent feasible incumbent; no Matching call is needed.
    incumbent = {
        tid: min(feasible[tid], key=lambda target: option_rank(tid, target))
        for tid in ids
    }
    incumbent_eval = evaluate(incumbent)
    if not incumbent_eval.resource_feasible:
        # The formal resource-aware mode permits Cloud for every task.
        incumbent = {tid: task_allocation.CLOUD for tid in ids}
        incumbent_eval = evaluate(incumbent)
        if not incumbent_eval.resource_feasible:
            raise ValueError("even all-Cloud assignment is resource-infeasible")

    if initial_matching is not None:
        candidate = {str(tid): str(target) for tid, target in initial_matching.items()}
        if set(candidate) != set(ids) or any(
            target not in feasible[tid] for tid, target in candidate.items()
        ):
            raise ValueError("initial_matching must cover exactly the legal tasks")
        trial = evaluate(candidate)
        if trial.resource_feasible and (
            trial.task_utility_sum > incumbent_eval.task_utility_sum + objective_tolerance
        ):
            incumbent, incumbent_eval = candidate, trial

    def search_block(
        change_ids: list[str],
        *,
        restrict_targets: bool,
    ) -> bool:
        """Return True if every unpruned branch was fully processed."""
        nonlocal incumbent, incumbent_eval, upper_bound_prunes
        changing = set(change_ids)
        fixed = {tid: incumbent[tid] for tid in ids if tid not in changing}
        branching = sorted(change_ids, key=lambda tid: (len(feasible[tid]), tid))
        choices: dict[str, list[str]] = {}
        for tid in branching:
            options = sorted(feasible[tid], key=lambda target: option_rank(tid, target))
            if restrict_targets:
                options = options[:target_shortlist_size]
                if incumbent[tid] not in options:
                    options.append(incumbent[tid])
            choices[tid] = options

        def dfs(partial: dict[str, str], position: int) -> None:
            nonlocal incumbent, incumbent_eval, upper_bound_prunes
            partial_eval = evaluate(partial)
            if not partial_eval.resource_feasible:
                return
            unassigned = branching[position:]
            optimistic_upper = (
                float(partial_eval.task_utility_sum)
                + sum(ideal_best[tid] for tid in unassigned)
            )
            if optimistic_upper <= incumbent_eval.task_utility_sum + objective_tolerance:
                upper_bound_prunes += 1
                return
            if position == len(branching):
                if partial_eval.task_utility_sum > (
                    incumbent_eval.task_utility_sum + objective_tolerance
                ):
                    incumbent = dict(partial)
                    incumbent_eval = partial_eval
                return
            tid = branching[position]
            for target in choices[tid]:
                partial[tid] = target
                dfs(partial, position + 1)
            del partial[tid]

        try:
            dfs(dict(fixed), 0)
            return True
        except _EvaluationBudgetExceeded:
            return False

    complete_exact_search = False
    if len(ids) <= exact_task_limit:
        complete_exact_search = search_block(ids, restrict_targets=False)
        reason = "exact_search_complete" if complete_exact_search else "oracle_budget"
    else:
        reason = "neighborhood_search_complete"
        for _ in range(max_neighborhood_rounds):
            start_utility = float(incumbent_eval.task_utility_sum)
            # The gap between the ideal solo utility and realized utility
            # prioritizes tasks affected by bandwidth/CPU contention.
            focus = sorted(
                ids,
                key=lambda tid: (
                    -(ideal_best[tid] - incumbent_eval.per_task_utility[tid]),
                    tid,
                ),
            )[:neighborhood_focus]
            block_size = min(neighborhood_size, len(focus))
            budget_reached = False
            for block in combinations(focus, block_size):
                if not search_block(list(block), restrict_targets=True):
                    budget_reached = True
                    reason = "oracle_budget"
                    break
            if budget_reached:
                break
            if incumbent_eval.task_utility_sum <= start_utility + objective_tolerance:
                break

    if not incumbent_eval.resource_feasible:
        raise AssertionError("internal search returned an infeasible matching")
    upper = (
        float(incumbent_eval.task_utility_sum)
        if complete_exact_search
        else max(root_upper_bound, float(incumbent_eval.task_utility_sum))
    )
    return AssignmentSearchResult(
        matching=dict(incumbent),
        evaluation=incumbent_eval,
        feasible_targets=feasible,
        oracle_evaluations=oracle_evaluations,
        upper_bound_prunes=upper_bound_prunes,
        global_upper_bound=upper,
        certified_absolute_gap=max(upper - float(incumbent_eval.task_utility_sum), 0.0),
        globally_optimal_within_tolerance=complete_exact_search,
        stop_reason=reason,
    )
