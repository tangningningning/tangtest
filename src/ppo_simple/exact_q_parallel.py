"""Deterministic transition-local workers for UAV Exact-Q evaluation.

The policy plan is intentionally built in the trainer process.  This module
contains only the environment/Matching/KKT branch evaluator so that worker
processes do not touch Torch, NumPy, or Python random state for sampling.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from decision import cache_configuration, slot_environment, task_allocation, uav_configuration

from .primitive_control import _intermediate_observation
from .rollout import _check_request_contract, _check_reward


_WORKER_CONTEXT: Any | None = None


@dataclass(frozen=True)
class ExactQHeadPlan:
    head_index: int
    uav_id: str
    current_position: np.ndarray
    move_candidates: tuple[Any, ...]
    raw_operation_probabilities: np.ndarray
    operation_mask: np.ndarray
    candidate_probabilities: np.ndarray


@dataclass(frozen=True)
class ExactQTransitionPlan:
    transition_index: int
    origin_index: int
    done: bool
    step: Any
    next_step: Any | None
    previous_cache: np.ndarray
    previous_uav: dict[str, np.ndarray]
    production_final_uav: dict[str, np.ndarray]
    next_cache_state: np.ndarray
    production_cache_triggers: np.ndarray
    production_uav_triggers: dict[str, int]
    production_cache_actions: tuple[dict[str, Any], ...]
    production_uav_actions: tuple[dict[str, Any], ...]
    uav_ids: tuple[str, ...]
    previous_uav_key: tuple[tuple[str, float, float], ...]
    heads: tuple[ExactQHeadPlan, ...]
    use_node_evaluation_cache: bool


def new_timing_profile() -> dict[str, float]:
    """Return the same Exact-Q timing keys used by the serial evaluator."""

    return {
        "matching_runtime": 0.0,
        "node_cache_lookup_runtime": 0.0,
        "node_eval_miss_runtime": 0.0,
        "kkt_runtime": 0.0,
        "kkt_coefficient_build_runtime": 0.0,
        "kkt_solver_core_runtime": 0.0,
        "kkt_result_assembly_runtime": 0.0,
        "reward_delay_aggregation_runtime": 0.0,
        "reward_energy_aggregation_runtime": 0.0,
        "reward_cost_aggregation_runtime": 0.0,
        "reward_final_combination_runtime": 0.0,
        "branch_state_build_runtime": 0.0,
        "reward_assembly_runtime": 0.0,
        "critic_runtime": 0.0,
        "exact_eval_kkt_conversion_runtime": 0.0,
        "exact_eval_state_copy_runtime": 0.0,
        "exact_eval_affected_node_prep_runtime": 0.0,
        "exact_eval_node_call_orchestration_runtime": 0.0,
        "exact_eval_result_merge_runtime": 0.0,
        "exact_eval_non_edge_prep_runtime": 0.0,
        "exact_eval_target_count_build_runtime": 0.0,
        "exact_eval_utility_sum_runtime": 0.0,
        "exact_eval_record_bookkeeping_runtime": 0.0,
        "exact_eval_other_runtime": 0.0,
        "current_kkt_conversion_call_count": 0.0,
        "candidate_kkt_conversion_call_count": 0.0,
        "exact_eval_state_copy_call_count": 0.0,
        "current_kkt_context_request_count": 0.0,
        "current_kkt_context_reuse_count": 0.0,
        "candidate_kkt_context_request_count": 0.0,
        "candidate_kkt_context_reuse_count": 0.0,
    }


def initialize_worker(context: Any) -> None:
    """Install immutable context and force one computational thread per worker."""

    global _WORKER_CONTEXT
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:  # pragma: no cover - depends on worker import order
        raise RuntimeError("Exact-Q worker could not enforce Torch single-thread mode") from exc
    _WORKER_CONTEXT = context


def _position_dict(source: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        str(key): np.asarray(source[key], dtype=np.float64).copy()
        for key in source
    }


def _target_action(
    uav_id: str,
    head_index: int,
    candidate_count: int,
    operation: str,
    before: np.ndarray,
    after: np.ndarray,
    selected_index: int | None,
    proposal_source: str | None = None,
) -> dict[str, Any]:
    action = {
        "head_type": "uav",
        "head_id": uav_id,
        "head_index": int(head_index),
        "candidate_count": int(candidate_count),
        "selected_index": selected_index,
        "conditional_action": "STAY" if operation == "STAY" else "TARGET_POSITION",
        "before_position": [float(value) for value in before],
        "after_position": [float(value) for value in after],
        "actual_moved": bool(operation == "MOVE"),
    }
    if proposal_source is not None:
        action["proposal_source"] = str(proposal_source)
    if operation == "MOVE":
        action["target_position"] = [float(value) for value in after]
    return action


def _full_branch_is_feasible(
    previous_positions: dict[str, np.ndarray],
    final_positions: dict[str, np.ndarray],
    triggers: dict[str, int],
    context: Any,
) -> bool:
    try:
        uav_configuration.validate_uav_configuration_transition(
            previous_positions,
            final_positions,
            triggers,
            context.cfg,
            context.flight_boundary,
        )
    except ValueError:
        return False
    return True


def evaluate_transition(plan: ExactQTransitionPlan, context: Any | None = None) -> dict[str, Any]:
    """Evaluate all valid branches for one transition without policy inference."""

    active_context = _WORKER_CONTEXT if context is None else context
    if active_context is None:
        raise RuntimeError("Exact-Q worker context is not initialized")

    timing_profile = new_timing_profile()
    counters: dict[str, int] = {
        "counterfactual_environment_executions": 0,
        "branch_request_count": 0,
        "unique_branch_count": 0,
        "branch_cache_hit_count": 0,
        "execute_slot_call_count": 0,
        "matching_call_count": 0,
        "kkt_call_count": 0,
        "kkt_cache_request_count": 0,
        "kkt_cache_hit_count": 0,
        "kkt_cache_miss_count": 0,
        "kkt_solver_call_count": 0,
        "node_eval_cache_request_count": 0,
        "node_eval_cache_hit_count": 0,
        "node_eval_cache_miss_count": 0,
        "node_eval_full_execution_count": 0,
        "ideal_utility_request_count": 0,
        "ideal_utility_cache_hit_count": 0,
        "ideal_utility_cache_miss_count": 0,
        "impact_scope_request_count": 0,
        "impact_scope_compute_count": 0,
        "impact_scope_reuse_count": 0,
        "stay_request_count": 0,
        "stay_infeasible_count": 0,
        "move_candidate_count": 0,
        "move_full_branch_feasible_count": 0,
        "move_full_branch_infeasible_count": 0,
        "heads_with_no_feasible_move": 0,
        "heads_with_infeasible_stay": 0,
        "selected_action_infeasible_count": 0,
        "invalid_full_branch_execute_slot_count": 0,
    }
    slot_branch_cache: dict[tuple[Any, ...], tuple[float, np.ndarray | None]] = {}
    transition_kkt_cache = task_allocation.KKTResourceAllocationCache()
    transition_node_evaluation_cache = (
        task_allocation.NodeEvaluationCache(timing_profile=timing_profile)
        if plan.use_node_evaluation_cache
        else None
    )
    head_results: list[dict[str, Any]] = []

    for head in plan.heads:
        uav_id = str(head.uav_id)
        current_position = np.asarray(head.current_position, dtype=np.float64).copy()
        move_candidates = head.move_candidates
        move_count = len(move_candidates)
        counters["stay_request_count"] += 1
        stay_branch_final_uav = {
            key: value.copy() for key, value in plan.production_final_uav.items()
        }
        stay_branch_final_uav[uav_id] = current_position.copy()
        stay_branch_triggers = dict(plan.production_uav_triggers)
        stay_branch_triggers[uav_id] = 0
        stay_full_branch_feasible = _full_branch_is_feasible(
            plan.previous_uav,
            stay_branch_final_uav,
            stay_branch_triggers,
            active_context,
        )
        if not stay_full_branch_feasible:
            counters["stay_infeasible_count"] += 1
            counters["heads_with_infeasible_stay"] += 1

        move_full_branch_feasible_mask = np.zeros(move_count, dtype=bool)
        for candidate_index, candidate in enumerate(move_candidates):
            branch_final_uav = {
                key: value.copy() for key, value in plan.production_final_uav.items()
            }
            branch_final_uav[uav_id] = np.asarray(
                candidate.target_position,
                dtype=np.float64,
            ).copy()
            branch_triggers = dict(plan.production_uav_triggers)
            branch_triggers[uav_id] = 1
            move_full_branch_feasible_mask[candidate_index] = _full_branch_is_feasible(
                plan.previous_uav,
                branch_final_uav,
                branch_triggers,
                active_context,
            )
        counters["move_candidate_count"] += move_count
        counters["move_full_branch_feasible_count"] += int(
            move_full_branch_feasible_mask.sum()
        )
        counters["move_full_branch_infeasible_count"] += int(
            (~move_full_branch_feasible_mask).sum()
        )
        if not bool(np.any(move_full_branch_feasible_mask)):
            counters["heads_with_no_feasible_move"] += 1

        teacher_operation_mask = np.asarray(
            [
                bool(stay_full_branch_feasible),
                bool(np.any(move_full_branch_feasible_mask)),
            ],
            dtype=bool,
        )
        if not bool(np.any(teacher_operation_mask)):
            raise RuntimeError("Exact-Q counterfactual operation has no feasible operation")

        production_action = plan.production_uav_actions[head.head_index]
        production_operation = str(
            production_action.get("conditional_action", "")
        ).upper()
        if production_operation == "STAY":
            selected_full_branch_feasible = stay_full_branch_feasible
        elif production_operation == "TARGET_POSITION":
            selected_index_value = production_action.get("selected_index")
            if selected_index_value is None:
                raise RuntimeError("production MOVE action is missing selected_index")
            selected_index_value = int(selected_index_value)
            if selected_index_value < 0 or selected_index_value >= move_count:
                raise RuntimeError("production MOVE action selected_index is out of range")
            selected_full_branch_feasible = bool(
                move_full_branch_feasible_mask[selected_index_value]
            )
        else:
            raise RuntimeError(
                "production UAV action has an unsupported conditional_action"
            )
        if not selected_full_branch_feasible:
            counters["selected_action_infeasible_count"] += 1
            raise RuntimeError("PRODUCTION_SELECTED_BRANCH_FULL_FEASIBLE = NO")

        next_step = plan.next_step

        def evaluate_branch(
            operation: str,
            selected_index: int | None,
            before: np.ndarray,
            after: np.ndarray,
        ) -> tuple[float, np.ndarray | None]:
            branch_state_started_at = time.perf_counter()
            branch_final_uav = {
                key: value.copy() for key, value in plan.production_final_uav.items()
            }
            branch_final_uav[uav_id] = np.asarray(after, dtype=np.float64).copy()
            branch_uav_triggers = dict(plan.production_uav_triggers)
            branch_uav_triggers[uav_id] = int(operation == "MOVE")
            branch_key = (
                int(plan.origin_index),
                np.asarray(plan.previous_cache, dtype=np.int8).tobytes(),
                np.asarray(plan.next_cache_state, dtype=np.int8).tobytes(),
                plan.previous_uav_key,
                tuple(
                    (
                        str(key),
                        float(np.asarray(branch_final_uav[key], dtype=np.float64)[0]),
                        float(np.asarray(branch_final_uav[key], dtype=np.float64)[1]),
                    )
                    for key in plan.uav_ids
                ),
                tuple(
                    (str(key), int(branch_uav_triggers[key]))
                    for key in plan.uav_ids
                ),
            )
            counters["branch_request_count"] += 1
            cached = slot_branch_cache.get(branch_key)
            if cached is not None:
                counters["branch_cache_hit_count"] += 1
                cached_reward, cached_observation = cached
                timing_profile["branch_state_build_runtime"] += (
                    time.perf_counter() - branch_state_started_at
                )
                return float(cached_reward), (
                    None
                    if cached_observation is None
                    else np.asarray(cached_observation, dtype=np.float32).copy()
                )
            counters["unique_branch_count"] += 1
            branch_uav_actions = list(plan.production_uav_actions)
            branch_uav_actions[head.head_index] = _target_action(
                uav_id,
                head.head_index,
                move_count,
                operation,
                np.asarray(before, dtype=np.float64),
                np.asarray(after, dtype=np.float64),
                selected_index,
                (
                    move_candidates[selected_index].proposal_source
                    if operation == "MOVE" and selected_index is not None
                    else None
                ),
            )
            configuration_override = slot_environment.build_policy_target_configuration(
                plan.previous_cache,
                plan.previous_uav,
                plan.next_cache_state,
                branch_final_uav,
                plan.production_cache_triggers,
                branch_uav_triggers,
                active_context.service_catalog,
                active_context.cfg,
                active_context.cache_nodes,
                active_context.cache_cfg,
                active_context.flight_boundary,
                cache_actions=plan.production_cache_actions,
                uav_actions=tuple(branch_uav_actions),
            )
            timing_profile["branch_state_build_runtime"] += (
                time.perf_counter() - branch_state_started_at
            )
            counters["execute_slot_call_count"] += 1
            counters["matching_call_count"] += 1
            result = slot_environment.execute_slot(
                plan.previous_cache,
                plan.previous_uav,
                plan.production_cache_triggers,
                branch_uav_triggers,
                plan.step.scenario_slice,
                plan.step.actual_tasks,
                plan.step.actual_vehicle_positions,
                active_context.local_cache,
                active_context.service_catalog,
                active_context.cfg,
                active_context.cache_nodes,
                active_context.cache_cfg,
                active_context.flight_boundary,
                slot_id=plan.step.slot,
                time_s=plan.step.time_s,
                performance_profile=None,
                configuration_profile=cache_configuration.ConfigurationEvaluationProfile(),
                configuration_override=configuration_override,
                matching_kkt_cache=transition_kkt_cache,
                node_evaluation_cache=transition_node_evaluation_cache,
                exact_q_timing_profile=timing_profile,
            )
            reward_assembly_started_at = time.perf_counter()
            counters["counterfactual_environment_executions"] += 1
            summary = result.matching_summary
            counters["kkt_call_count"] += int(summary.get("kkt_cache_requests", 0))
            counters["kkt_cache_request_count"] += int(
                summary.get("KKT_CACHE_REQUEST_COUNT", 0)
            )
            counters["kkt_cache_hit_count"] += int(summary.get("KKT_CACHE_HIT_COUNT", 0))
            counters["kkt_cache_miss_count"] += int(summary.get("KKT_CACHE_MISS_COUNT", 0))
            counters["kkt_solver_call_count"] += int(summary.get("KKT_SOLVER_CALL_COUNT", 0))
            counters["node_eval_cache_request_count"] += int(
                summary.get("NODE_EVAL_CACHE_REQUEST_COUNT", 0)
            )
            counters["node_eval_cache_hit_count"] += int(
                summary.get("NODE_EVAL_CACHE_HIT_COUNT", 0)
            )
            counters["node_eval_cache_miss_count"] += int(
                summary.get("NODE_EVAL_CACHE_MISS_COUNT", 0)
            )
            counters["node_eval_full_execution_count"] += int(
                summary.get("NODE_EVAL_FULL_EXECUTION_COUNT", 0)
            )
            counters["ideal_utility_request_count"] += int(
                summary.get("IDEAL_UTILITY_REQUEST_COUNT", 0)
            )
            counters["ideal_utility_cache_hit_count"] += int(
                summary.get("IDEAL_UTILITY_CACHE_HIT_COUNT", 0)
            )
            counters["ideal_utility_cache_miss_count"] += int(
                summary.get("IDEAL_UTILITY_CACHE_MISS_COUNT", 0)
            )
            counters["impact_scope_request_count"] += int(
                summary.get("IMPACT_SCOPE_REQUEST_COUNT", 0)
            )
            counters["impact_scope_compute_count"] += int(
                summary.get("IMPACT_SCOPE_COMPUTE_COUNT", 0)
            )
            counters["impact_scope_reuse_count"] += int(
                summary.get("IMPACT_SCOPE_REUSE_COUNT", 0)
            )
            if not result.step_feasible:
                raise RuntimeError(
                    "proposed counterfactual transition is infeasible: "
                    f"origin={plan.step.origin_index}, UAV={uav_id}, operation={operation}"
                )
            _check_request_contract(
                plan.previous_cache,
                plan.previous_uav,
                result,
                active_context.action_spec,
            )
            reward = _check_reward(result)
            if next_step is None:
                next_observation = None
            else:
                next_observation = _intermediate_observation(
                    np.asarray(result.next_cache_state, dtype=np.int8),
                    result.next_uav_positions,
                    next_step,
                    active_context,
                )
            safe_observation = (
                None
                if next_observation is None
                else np.asarray(next_observation, dtype=np.float32).copy()
            )
            slot_branch_cache[branch_key] = (float(reward), safe_observation)
            timing_profile["reward_assembly_runtime"] += (
                time.perf_counter() - reward_assembly_started_at
            )
            return float(reward), (
                None if safe_observation is None else safe_observation.copy()
            )

        branch_rewards: list[float] = []
        branch_observations: list[np.ndarray | None] = []
        branch_labels: list[tuple[str, int | None]] = []
        if stay_full_branch_feasible:
            stay_reward, stay_observation = evaluate_branch(
                "STAY",
                None,
                current_position,
                current_position,
            )
            branch_rewards.append(stay_reward)
            branch_observations.append(stay_observation)
            branch_labels.append(("STAY", None))
        for candidate_index, candidate in enumerate(move_candidates):
            if not bool(move_full_branch_feasible_mask[candidate_index]):
                continue
            move_reward, move_observation = evaluate_branch(
                "MOVE",
                candidate_index,
                np.asarray(candidate.before_position, dtype=np.float64),
                np.asarray(candidate.target_position, dtype=np.float64),
            )
            branch_rewards.append(move_reward)
            branch_observations.append(move_observation)
            branch_labels.append(("MOVE", candidate_index))
        if not branch_rewards:
            raise RuntimeError("Exact-Q counterfactual branch set is unexpectedly empty")
        head_results.append(
            {
                "stay_full_branch_feasible": bool(stay_full_branch_feasible),
                "move_full_branch_feasible_mask": move_full_branch_feasible_mask,
                "teacher_operation_mask": teacher_operation_mask,
                "branch_rewards": np.asarray(branch_rewards, dtype=np.float64),
                "branch_observations": tuple(branch_observations),
                "branch_labels": tuple(branch_labels),
            }
        )

    return {
        "transition_index": int(plan.transition_index),
        "head_results": tuple(head_results),
        "counters": counters,
        "timing": timing_profile,
    }
