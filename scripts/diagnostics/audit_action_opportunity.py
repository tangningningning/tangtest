"""Audit realized value of one-step adaptive Cache/UAV actions.

This utility is deliberately posthoc.  It starts from the canonical static-zero
reference trajectory, forks one feasible production primitive at a selected
origin, and evaluates the resulting five real slots through ``execute_slot``.
Actual future tasks and vehicle positions are used only by that oracle
measurement path; they are never passed to the policy observation or ranking.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decision import cache_configuration, slot_environment, uav_configuration
from ppo_simple import evaluator
from ppo_simple import primitive_control
from ppo_simple.rollout import _observation

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - CLI import diagnostics
    torch = None  # type: ignore[assignment]


DEFAULT_CONFIG = "system_config_ppo_v6_full_episode_randomized.toml"
DEFAULT_CHECKPOINT = (
    "runs/ppo_v6_popart_actor1e4_lazy_stage1_20260911/checkpoint_ep50.pt"
)
DEFAULT_REALIZATION = "runs/ppo_v6_dev_eval_pool_30001_30020/episode_30001"
DEFAULT_OUTPUT = "runs/diagnostics/p50_action_opportunity_seed30001.json"
DEFAULT_ORIGINS = (60, 150, 240)
DEFAULT_HORIZON = 5
DEFAULT_TOLERANCE = 1.0e-9
CACHE_OPERATION_NAMES = ("NO_CHANGE", "ADD", "DELETE", "SWAP")
UAV_OPERATION_NAMES = ("STAY", "MOVE")


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def _copy_uav_positions(positions: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        str(uav_id): np.asarray(xy, dtype=np.float64).copy()
        for uav_id, xy in positions.items()
    }


def _zero_triggers(context: Any) -> tuple[np.ndarray, dict[str, int]]:
    return (
        np.zeros(len(context.action_spec.cache_node_ids), dtype=np.int8),
        {str(uav_id): 0 for uav_id in context.action_spec.uav_ids},
    )


def _execute_step(
    context: Any,
    step: Any,
    previous_cache: np.ndarray,
    previous_uav: dict[str, Any],
    cache_triggers: np.ndarray,
    uav_triggers: dict[str, int],
    configuration_override: Any | None = None,
) -> Any:
    result = slot_environment.execute_slot(
        previous_cache,
        previous_uav,
        cache_triggers,
        uav_triggers,
        step.scenario_slice,
        step.actual_tasks,
        step.actual_vehicle_positions,
        context.local_cache,
        context.service_catalog,
        context.cfg,
        context.cache_nodes,
        context.cache_cfg,
        context.flight_boundary,
        slot_id=int(step.slot),
        time_s=float(step.time_s),
        configuration_override=configuration_override,
    )
    if not result.step_feasible:
        raise RuntimeError(
            "counterfactual execute_slot returned infeasible: "
            f"origin={step.origin_index}, reason={result.infeasible_reason}"
        )
    if result.system_utility is None:
        raise RuntimeError("counterfactual execute_slot returned no reward")
    expected_reward = (
        float(result.actual_task_utility)
        - float(result.cache_cost)
        - float(result.movement_cost)
    )
    if not np.isclose(
        float(result.system_utility),
        expected_reward,
        rtol=1.0e-12,
        atol=1.0e-9,
    ):
        raise AssertionError(
            f"G=Q-cache_cost-movement_cost identity failed at origin={step.origin_index}"
        )
    return result


def _summarize_results(results: list[Any]) -> dict[str, Any]:
    if not results:
        raise ValueError("counterfactual branch must contain at least one slot")
    rewards = np.asarray([float(item.system_utility) for item in results], dtype=np.float64)
    q_values = np.asarray(
        [float(item.actual_task_utility) for item in results], dtype=np.float64
    )
    cache_costs = np.asarray(
        [float(item.cache_cost) for item in results], dtype=np.float64
    )
    movement_costs = np.asarray(
        [float(item.movement_cost) for item in results], dtype=np.float64
    )
    values = np.concatenate((rewards, q_values, cache_costs, movement_costs))
    if not bool(np.all(np.isfinite(values))):
        raise FloatingPointError("counterfactual branch contains non-finite metrics")
    return {
        "G1": float(rewards[0]),
        "G5": float(rewards.sum()),
        "Q1": float(q_values[0]),
        "Q5": float(q_values.sum()),
        "cache_cost": float(cache_costs.sum()),
        "movement_cost": float(movement_costs.sum()),
        "slot_rewards": [float(value) for value in rewards],
        "slot_q": [float(value) for value in q_values],
    }


def _run_branch(
    context: Any,
    steps_by_origin: dict[int, Any],
    origin: int,
    horizon: int,
    initial_cache: np.ndarray,
    initial_uav: dict[str, Any],
    first_cache_triggers: np.ndarray,
    first_uav_triggers: dict[str, int],
    first_configuration_override: Any | None,
) -> tuple[dict[str, Any], float]:
    current_cache = np.asarray(initial_cache, dtype=np.int8).copy()
    current_uav = _copy_uav_positions(initial_uav)
    zero_cache, zero_uav = _zero_triggers(context)
    results: list[Any] = []
    started_at = time.perf_counter()
    for offset in range(int(horizon)):
        step = steps_by_origin[int(origin) + offset]
        if offset == 0:
            cache_triggers = np.asarray(first_cache_triggers, dtype=np.int8).copy()
            uav_triggers = {key: int(value) for key, value in first_uav_triggers.items()}
            configuration_override = first_configuration_override
        else:
            cache_triggers = zero_cache.copy()
            uav_triggers = dict(zero_uav)
            configuration_override = None
        result = _execute_step(
            context,
            step,
            current_cache,
            current_uav,
            cache_triggers,
            uav_triggers,
            configuration_override,
        )
        results.append(result)
        current_cache = np.asarray(result.next_cache_state, dtype=np.int8).copy()
        current_uav = _copy_uav_positions(result.next_uav_positions)
    return _summarize_results(results), time.perf_counter() - started_at


def _build_static_zero_reference(
    context: Any,
    steps: list[Any],
) -> tuple[dict[int, dict[str, Any]], dict[int, Any], float]:
    current_cache = np.asarray(context.initial_cache, dtype=np.int8).copy()
    current_uav = _copy_uav_positions(context.initial_uav)
    zero_cache, zero_uav = _zero_triggers(context)
    states: dict[int, dict[str, Any]] = {}
    results: dict[int, Any] = {}
    started_at = time.perf_counter()
    for step in steps:
        origin = int(step.origin_index)
        states[origin] = {
            "cache": current_cache.copy(),
            "uav": _copy_uav_positions(current_uav),
        }
        result = _execute_step(
            context,
            step,
            current_cache,
            current_uav,
            zero_cache,
            zero_uav,
        )
        results[origin] = result
        current_cache = np.asarray(result.next_cache_state, dtype=np.int8).copy()
        current_uav = _copy_uav_positions(result.next_uav_positions)
    return states, results, time.perf_counter() - started_at


def _configuration_override(
    context: Any,
    previous_cache: np.ndarray,
    previous_uav: dict[str, Any],
    final_cache: np.ndarray,
    final_uav: dict[str, Any],
    cache_triggers: np.ndarray,
    uav_triggers: dict[str, int],
) -> Any:
    return slot_environment.build_policy_primitive_configuration(
        previous_cache,
        previous_uav,
        final_cache,
        final_uav,
        cache_triggers,
        uav_triggers,
        context.service_catalog,
        context.cfg,
        context.cache_nodes,
        context.cache_cfg,
        context.flight_boundary,
    )


def _policy_candidate_distribution(
    policy: Any,
    conditional_context: np.ndarray,
    candidate_features: np.ndarray,
    candidate_kind: str,
) -> tuple[int, np.ndarray]:
    with torch.inference_mode():
        if candidate_kind == "cache":
            sample = policy.sample_cache_candidate(
                conditional_context,
                candidate_features,
                deterministic=True,
            )
        elif candidate_kind == "uav":
            sample = policy.sample_uav_candidate(
                conditional_context,
                candidate_features,
                deterministic=True,
            )
        else:
            raise ValueError(f"unknown candidate kind: {candidate_kind}")
        probabilities = torch.softmax(sample.logits, dim=-1).detach().cpu().numpy()
    if probabilities.ndim != 1 or not bool(np.all(np.isfinite(probabilities))):
        raise ValueError("policy conditional candidate probabilities are invalid")
    if not np.isclose(float(probabilities.sum()), 1.0, rtol=0.0, atol=1.0e-6):
        raise ValueError("policy conditional candidate probabilities do not sum to one")
    return int(sample.selected_index), np.asarray(probabilities, dtype=np.float64)


def _cache_candidate_inputs(
    context: Any,
    state: dict[str, Any],
    step: Any,
    node_index: int,
    cache_triggers: np.ndarray,
) -> tuple[list[Any], np.ndarray, np.ndarray]:
    cache = np.asarray(state["cache"], dtype=np.int8)
    previous_cache = cache.copy()
    candidates = cache_configuration.enumerate_cache_primitive_actions(
        cache,
        node_index,
        context.service_catalog,
        context.cache_nodes,
    )
    conditional_context = primitive_control._intermediate_observation(
        cache,
        state["uav"],
        step,
        context,
    )
    candidate_features = np.stack(
        [
            primitive_control._cache_candidate_feature(
                candidate,
                cache,
                previous_cache,
                cache_triggers,
                step.predicted_service_popularity,
                context,
            )
            for candidate in candidates
        ]
    ).astype(np.float32)
    return candidates, candidate_features, conditional_context


def _uav_candidate_inputs(
    context: Any,
    state: dict[str, Any],
    step: Any,
    uav_index: int,
    uav_triggers: dict[str, int],
) -> tuple[list[Any], np.ndarray, np.ndarray, str]:
    current_uav = _copy_uav_positions(state["uav"])
    previous_uav = _copy_uav_positions(state["uav"])
    uav_id = str(context.action_spec.uav_ids[uav_index])
    candidates = uav_configuration.enumerate_uav_primitive_actions(
        current_uav,
        previous_uav,
        uav_id,
        uav_triggers,
        context.cfg,
        context.flight_boundary,
    )
    conditional_context = primitive_control._intermediate_observation(
        state["cache"],
        current_uav,
        step,
        context,
    )
    candidate_features = np.stack(
        [
            primitive_control._uav_candidate_feature(
                candidate,
                current_uav,
                previous_uav,
                uav_triggers,
                step.predicted_vehicle_positions,
                context,
            )
            for candidate in candidates
        ]
    ).astype(np.float32)
    return candidates, candidate_features, conditional_context, uav_id


def _ranked_indices(delta_by_index: dict[int, float]) -> list[int]:
    return sorted(delta_by_index, key=lambda index: (-float(delta_by_index[index]), index))


def _cache_head_audit(
    context: Any,
    policy: Any,
    steps_by_origin: dict[int, Any],
    origin: int,
    node_index: int,
    state: dict[str, Any],
    baseline: dict[str, Any],
    horizon: int,
    tolerance: float,
) -> tuple[dict[str, Any], float]:
    cache_triggers = np.zeros(len(context.action_spec.cache_node_ids), dtype=np.int8)
    cache_triggers[node_index] = 1
    zero_uav = {str(uav_id): 0 for uav_id in context.action_spec.uav_ids}
    step = steps_by_origin[int(origin)]
    candidates, candidate_features, conditional_context = _cache_candidate_inputs(
        context,
        state,
        step,
        node_index,
        cache_triggers,
    )
    policy_top_index, policy_probabilities = _policy_candidate_distribution(
        policy,
        conditional_context,
        candidate_features,
        "cache",
    )
    started_at = time.perf_counter()
    active_results: list[dict[str, Any]] = []
    delta_by_index: dict[int, float] = {0: 0.0}
    result_by_index: dict[int, dict[str, Any]] = {}
    for candidate_index, candidate in enumerate(candidates):
        if str(candidate.operation_type) == "NO_CHANGE":
            if candidate_index != 0:
                raise AssertionError("cache candidate list must start with NO_CHANGE")
            continue
        branch_cache = np.asarray(state["cache"], dtype=np.int8).copy()
        applied = cache_configuration.apply_cache_candidate(
            branch_cache,
            node_index,
            candidate.after_node_cache,
            context.service_catalog,
            context.cache_nodes,
        )
        override = _configuration_override(
            context,
            np.asarray(state["cache"], dtype=np.int8),
            state["uav"],
            branch_cache,
            state["uav"],
            cache_triggers,
            zero_uav,
        )
        branch_summary, branch_runtime = _run_branch(
            context,
            steps_by_origin,
            origin,
            horizon,
            np.asarray(state["cache"], dtype=np.int8),
            state["uav"],
            cache_triggers,
            zero_uav,
            override,
        )
        delta_g1 = float(branch_summary["G1"] - baseline["G1"])
        delta_g5 = float(branch_summary["G5"] - baseline["G5"])
        delta_q1 = float(branch_summary["Q1"] - baseline["Q1"])
        delta_q5 = float(branch_summary["Q5"] - baseline["Q5"])
        record = {
            "candidate_index": int(candidate_index),
            "node_id": str(candidate.node_id),
            "operation": str(candidate.operation_type),
            "added_service_ids": [int(value) for value in candidate.added_service_ids],
            "removed_service_ids": [int(value) for value in candidate.removed_service_ids],
            "cache_cost": float(branch_summary["cache_cost"]),
            "movement_cost": float(branch_summary["movement_cost"]),
            "G_candidate_1": float(branch_summary["G1"]),
            "G_candidate_5": float(branch_summary["G5"]),
            "Q_candidate_1": float(branch_summary["Q1"]),
            "Q_candidate_5": float(branch_summary["Q5"]),
            "delta_G1": delta_g1,
            "delta_G5": delta_g5,
            "delta_Q1": delta_q1,
            "delta_Q5": delta_q5,
            "beneficial_H5": bool(delta_g5 > tolerance),
            "actual_changed": bool(applied["actual_changed"]),
            "runtime_seconds": float(branch_runtime),
        }
        active_results.append(record)
        result_by_index[int(candidate_index)] = record
        delta_by_index[int(candidate_index)] = delta_g5
    if not active_results:
        raise ValueError("cache head has no active candidates")
    ranking = _ranked_indices(delta_by_index)
    best_available_index = ranking[0]
    best_available_delta = float(delta_by_index[best_available_index])
    policy_top_record = result_by_index.get(policy_top_index)
    policy_top_delta = float(delta_by_index[policy_top_index])
    return (
        {
            "head_index": int(node_index),
            "node_id": str(context.action_spec.cache_node_ids[node_index]),
            "active_candidate_count": len(active_results),
            "positive_delta_G1_count": int(
                sum(float(record["delta_G1"]) > tolerance for record in active_results)
            ),
            "positive_delta_G5_count": int(
                sum(bool(record["beneficial_H5"]) for record in active_results)
            ),
            "delayed_benefit_count": int(
                sum(
                    float(record["delta_G1"]) <= tolerance
                    and float(record["delta_G5"]) > tolerance
                    for record in active_results
                )
            ),
            "best_active_delta_G1": float(
                max(float(record["delta_G1"]) for record in active_results)
            ),
            "best_active_delta_G5": float(
                max(float(record["delta_G5"]) for record in active_results)
            ),
            "best_realized_candidate_delta_G5": best_available_delta,
            "best_realized_candidate_index": int(best_available_index),
            "best_realized_candidate_operation": (
                "NO_CHANGE"
                if best_available_index == 0
                else str(result_by_index[best_available_index]["operation"])
            ),
            "policy_top_candidate": int(policy_top_index),
            "policy_top_candidate_operation": (
                "NO_CHANGE"
                if policy_top_index == 0
                else str(policy_top_record["operation"])
            ),
            "policy_top_candidate_probability": float(policy_probabilities[policy_top_index]),
            "policy_top_candidate_delta_G5": policy_top_delta,
            "policy_top_candidate_realized_rank": int(ranking.index(policy_top_index) + 1),
            "PPO_REGRET": float(best_available_delta - policy_top_delta),
            "candidate_results": active_results,
            "policy_candidate_probabilities": [
                float(value) for value in policy_probabilities
            ],
            "ranking_includes_no_change_baseline": True,
        },
        time.perf_counter() - started_at,
    )


def _uav_head_audit(
    context: Any,
    policy: Any,
    steps_by_origin: dict[int, Any],
    origin: int,
    uav_index: int,
    state: dict[str, Any],
    baseline: dict[str, Any],
    horizon: int,
    tolerance: float,
) -> tuple[dict[str, Any], float]:
    zero_cache = np.zeros(len(context.action_spec.cache_node_ids), dtype=np.int8)
    uav_id = str(context.action_spec.uav_ids[uav_index])
    uav_triggers = {str(item): int(str(item) == uav_id) for item in context.action_spec.uav_ids}
    step = steps_by_origin[int(origin)]
    candidates, candidate_features, conditional_context, _ = _uav_candidate_inputs(
        context,
        state,
        step,
        uav_index,
        uav_triggers,
    )
    policy_top_index, policy_probabilities = _policy_candidate_distribution(
        policy,
        conditional_context,
        candidate_features,
        "uav",
    )
    started_at = time.perf_counter()
    active_results: list[dict[str, Any]] = []
    delta_by_index: dict[int, float] = {0: 0.0}
    result_by_index: dict[int, dict[str, Any]] = {}
    for candidate_index, candidate in enumerate(candidates):
        if str(candidate.operation_type) == "STAY":
            if candidate_index != 0:
                raise AssertionError("UAV candidate list must start with STAY")
            continue
        branch_uav = _copy_uav_positions(state["uav"])
        applied = uav_configuration.apply_uav_candidate(
            branch_uav,
            state["uav"],
            uav_id,
            candidate.after_position,
            uav_triggers,
            context.cfg,
            context.flight_boundary,
        )
        override = _configuration_override(
            context,
            np.asarray(state["cache"], dtype=np.int8),
            state["uav"],
            np.asarray(state["cache"], dtype=np.int8),
            branch_uav,
            zero_cache,
            uav_triggers,
        )
        branch_summary, branch_runtime = _run_branch(
            context,
            steps_by_origin,
            origin,
            horizon,
            np.asarray(state["cache"], dtype=np.int8),
            state["uav"],
            zero_cache,
            uav_triggers,
            override,
        )
        delta_g1 = float(branch_summary["G1"] - baseline["G1"])
        delta_g5 = float(branch_summary["G5"] - baseline["G5"])
        delta_q1 = float(branch_summary["Q1"] - baseline["Q1"])
        delta_q5 = float(branch_summary["Q5"] - baseline["Q5"])
        record = {
            "candidate_index": int(candidate_index),
            "uav_id": str(candidate.uav_id),
            "operation": str(candidate.operation_type),
            "before_xy": [float(value) for value in candidate.before_position],
            "after_xy": [float(value) for value in candidate.after_position],
            "distance": float(np.linalg.norm(candidate.after_position - candidate.before_position)),
            "movement_cost": float(branch_summary["movement_cost"]),
            "cache_cost": float(branch_summary["cache_cost"]),
            "G_candidate_1": float(branch_summary["G1"]),
            "G_candidate_5": float(branch_summary["G5"]),
            "Q_candidate_1": float(branch_summary["Q1"]),
            "Q_candidate_5": float(branch_summary["Q5"]),
            "delta_G1": delta_g1,
            "delta_G5": delta_g5,
            "delta_Q1": delta_q1,
            "delta_Q5": delta_q5,
            "beneficial_H5": bool(delta_g5 > tolerance),
            "actual_moved": bool(applied["actual_moved"]),
            "runtime_seconds": float(branch_runtime),
        }
        active_results.append(record)
        result_by_index[int(candidate_index)] = record
        delta_by_index[int(candidate_index)] = delta_g5
    if not active_results:
        raise ValueError("UAV head has no active MOVE candidates")
    ranking = _ranked_indices(delta_by_index)
    best_available_index = ranking[0]
    best_available_delta = float(delta_by_index[best_available_index])
    policy_top_record = result_by_index.get(policy_top_index)
    policy_top_delta = float(delta_by_index[policy_top_index])
    return (
        {
            "head_index": int(uav_index),
            "uav_id": uav_id,
            "active_candidate_count": len(active_results),
            "positive_delta_G1_count": int(
                sum(float(record["delta_G1"]) > tolerance for record in active_results)
            ),
            "positive_delta_G5_count": int(
                sum(bool(record["beneficial_H5"]) for record in active_results)
            ),
            "delayed_benefit_count": int(
                sum(
                    float(record["delta_G1"]) <= tolerance
                    and float(record["delta_G5"]) > tolerance
                    for record in active_results
                )
            ),
            "best_active_delta_G1": float(
                max(float(record["delta_G1"]) for record in active_results)
            ),
            "best_active_delta_G5": float(
                max(float(record["delta_G5"]) for record in active_results)
            ),
            "best_realized_candidate_delta_G5": best_available_delta,
            "best_realized_candidate_index": int(best_available_index),
            "best_realized_candidate_operation": (
                "STAY"
                if best_available_index == 0
                else str(result_by_index[best_available_index]["operation"])
            ),
            "policy_top_candidate": int(policy_top_index),
            "policy_top_candidate_operation": (
                "STAY"
                if policy_top_index == 0
                else str(policy_top_record["operation"])
            ),
            "policy_top_candidate_probability": float(policy_probabilities[policy_top_index]),
            "policy_top_candidate_delta_G5": policy_top_delta,
            "policy_top_candidate_realized_rank": int(ranking.index(policy_top_index) + 1),
            "PPO_REGRET": float(best_available_delta - policy_top_delta),
            "candidate_results": active_results,
            "policy_candidate_probabilities": [
                float(value) for value in policy_probabilities
            ],
            "ranking_includes_stay_baseline": True,
        },
        time.perf_counter() - started_at,
    )


def _initial_state_audit(context: Any) -> dict[str, Any]:
    initial_cache = np.asarray(context.initial_cache, dtype=np.int8)
    service_sizes = np.asarray(
        context.service_catalog["service_sizes_mb"], dtype=np.float64
    )
    cache_configuration.validate_cache_shape_and_capacity(
        initial_cache,
        context.service_catalog,
        context.cache_nodes,
    )
    catalog_count = int(context.service_catalog["num_services"])
    total_catalog_size = float(service_sizes.sum())
    capacity_rows: list[dict[str, Any]] = []
    uniform_maximal = True
    all_services_fit = True
    for node_index, node_id in enumerate(context.cache_nodes["node_ids"]):
        capacity = float(context.cache_nodes["capacities_mb"][node_index])
        cached = initial_cache[node_index].astype(bool)
        used = float(cache_configuration.node_used_capacity_mb(cached, service_sizes))
        remaining = capacity - used
        maximal = all(
            bool(cached[service_id])
            or float(service_sizes[service_id]) > remaining + 1.0e-9
            for service_id in range(catalog_count)
        )
        fits = bool(capacity + 1.0e-9 >= total_catalog_size)
        uniform_maximal = uniform_maximal and maximal
        all_services_fit = all_services_fit and fits
        capacity_rows.append(
            {
                "node_id": str(node_id),
                "role": str(context.cache_nodes["node_roles"][node_index]),
                "capacity_mb": capacity,
                "used_mb": used,
                "utilization_ratio": 0.0 if capacity <= 0.0 else used / capacity,
                "cached_service_count": int(cached.sum()),
                "catalog_service_count": catalog_count,
                "catalog_coverage_ratio": (
                    0.0 if catalog_count == 0 else float(cached.sum()) / catalog_count
                ),
                "all_services_fit_on_node": fits,
                "maximal_feasible_under_uniform_initializer": maximal,
            }
        )
    configured_uav = uav_configuration.initial_uav_positions_from_project(context.cfg)
    fixed_uav_match = all(
        np.array_equal(
            np.asarray(context.initial_uav[uav_id], dtype=np.float64),
            np.asarray(configured_uav[uav_id], dtype=np.float64),
        )
        for uav_id in context.initial_uav
    )
    uses_actual_future_popularity = False
    uses_actual_future_tasks = False
    uses_future_trajectory = False
    uses_predictions = False
    uses_uav_future_trajectory = False
    return {
        "initial_cache_uses_actual_future_popularity": uses_actual_future_popularity,
        "initial_cache_uses_actual_future_tasks": uses_actual_future_tasks,
        "initial_cache_uses_future_trajectory": uses_future_trajectory,
        "initial_cache_uses_predictions": uses_predictions,
        "initial_cache_seeded_uniform_feasible": bool(uniform_maximal),
        "initial_cache_initializer": "cache_configuration.popularity_aware_initial_cache",
        "initial_cache_initializer_inputs": [
            "service_catalog",
            "cache_nodes",
            "cache_config.initial_cache_seed",
        ],
        "initial_uav_uses_future_trajectory": uses_uav_future_trajectory,
        "initial_uav_from_fixed_config": bool(fixed_uav_match),
        "initial_uav_initializer": "uav_configuration.initial_uav_positions_from_project",
        "future_information_leak_in_canonical_initial_state": bool(
            uses_actual_future_popularity
            or uses_actual_future_tasks
            or uses_future_trajectory
            or uses_predictions
            or uses_uav_future_trajectory
        ),
        "cache_capacity_summary": capacity_rows,
        "all_services_fit_on_node": bool(all_services_fit),
        "catalog_total_size_mb": total_catalog_size,
        "initial_cache_seed": int(context.cache_cfg.initial_cache_seed),
        "initial_cache": initial_cache.tolist(),
        "initial_uav": {
            str(uav_id): [float(value) for value in xy]
            for uav_id, xy in context.initial_uav.items()
        },
    }


def _origin_summary(
    origin: int,
    baseline: dict[str, Any],
    cache_heads: list[dict[str, Any]],
    uav_heads: list[dict[str, Any]],
    reference_state: dict[str, Any],
    tolerance: float,
    runtime_seconds: float,
) -> dict[str, Any]:
    cache_candidates = [
        record for head in cache_heads for record in head["candidate_results"]
    ]
    uav_candidates = [
        record for head in uav_heads for record in head["candidate_results"]
    ]
    cache_best = max(cache_candidates, key=lambda record: float(record["delta_G5"]))
    uav_best = max(uav_candidates, key=lambda record: float(record["delta_G5"]))
    return {
        "origin": int(origin),
        "reference_state": {
            "cache": np.asarray(reference_state["cache"], dtype=np.int8).tolist(),
            "uav": {
                str(uav_id): [float(value) for value in xy]
                for uav_id, xy in reference_state["uav"].items()
            },
        },
        "reference_no_op": baseline,
        "CACHE_ACTIVE_CANDIDATE_COUNT": int(
            sum(head["active_candidate_count"] for head in cache_heads)
        ),
        "CACHE_POSITIVE_DELTA_G1_COUNT": int(
            sum(head["positive_delta_G1_count"] for head in cache_heads)
        ),
        "CACHE_POSITIVE_DELTA_G5_COUNT": int(
            sum(head["positive_delta_G5_count"] for head in cache_heads)
        ),
        "CACHE_POSITIVE_DELTA_G5_RATE": float(
            sum(head["positive_delta_G5_count"] for head in cache_heads)
            / max(sum(head["active_candidate_count"] for head in cache_heads), 1)
        ),
        "CACHE_BEST_DELTA_G1": float(
            max(head["best_active_delta_G1"] for head in cache_heads)
        ),
        "CACHE_BEST_DELTA_G5": float(
            max(head["best_active_delta_G5"] for head in cache_heads)
        ),
        "CACHE_BEST_NODE": str(cache_best["node_id"]),
        "CACHE_BEST_OPERATION": str(cache_best["operation"]),
        "CACHE_POLICY_TOP_DELTA_G5": [
            float(head["policy_top_candidate_delta_G5"]) for head in cache_heads
        ],
        "CACHE_POLICY_TOP_REALIZED_RANK": [
            int(head["policy_top_candidate_realized_rank"]) for head in cache_heads
        ],
        "CACHE_POLICY_REGRET": [float(head["PPO_REGRET"]) for head in cache_heads],
        "CACHE_DELAYED_BENEFIT_COUNT": int(
            sum(head["delayed_benefit_count"] for head in cache_heads)
        ),
        "UAV_MOVE_CANDIDATE_COUNT": int(
            sum(head["active_candidate_count"] for head in uav_heads)
        ),
        "UAV_POSITIVE_DELTA_G1_COUNT": int(
            sum(head["positive_delta_G1_count"] for head in uav_heads)
        ),
        "UAV_POSITIVE_DELTA_G5_COUNT": int(
            sum(head["positive_delta_G5_count"] for head in uav_heads)
        ),
        "UAV_POSITIVE_DELTA_G5_RATE": float(
            sum(head["positive_delta_G5_count"] for head in uav_heads)
            / max(sum(head["active_candidate_count"] for head in uav_heads), 1)
        ),
        "UAV_BEST_DELTA_G1": float(
            max(head["best_active_delta_G1"] for head in uav_heads)
        ),
        "UAV_BEST_DELTA_G5": float(
            max(head["best_active_delta_G5"] for head in uav_heads)
        ),
        "UAV_BEST_ID": str(uav_best["uav_id"]),
        "UAV_POLICY_TOP_DELTA_G5": [
            float(head["policy_top_candidate_delta_G5"]) for head in uav_heads
        ],
        "UAV_POLICY_TOP_REALIZED_RANK": [
            int(head["policy_top_candidate_realized_rank"]) for head in uav_heads
        ],
        "UAV_POLICY_REGRET": [float(head["PPO_REGRET"]) for head in uav_heads],
        "UAV_DELAYED_BENEFIT_COUNT": int(
            sum(head["delayed_benefit_count"] for head in uav_heads)
        ),
        "numerical_tolerance": float(tolerance),
        "cache_heads": cache_heads,
        "uav_heads": uav_heads,
        "runtime_seconds": float(runtime_seconds),
    }


def _aggregate_origin_summaries(
    origin_summaries: list[dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    cache_active = sum(int(item["CACHE_ACTIVE_CANDIDATE_COUNT"]) for item in origin_summaries)
    cache_positive_g1 = sum(
        int(item["CACHE_POSITIVE_DELTA_G1_COUNT"]) for item in origin_summaries
    )
    cache_positive_g5 = sum(
        int(item["CACHE_POSITIVE_DELTA_G5_COUNT"]) for item in origin_summaries
    )
    cache_delayed = sum(int(item["CACHE_DELAYED_BENEFIT_COUNT"]) for item in origin_summaries)
    uav_active = sum(int(item["UAV_MOVE_CANDIDATE_COUNT"]) for item in origin_summaries)
    uav_positive_g1 = sum(
        int(item["UAV_POSITIVE_DELTA_G1_COUNT"]) for item in origin_summaries
    )
    uav_positive_g5 = sum(
        int(item["UAV_POSITIVE_DELTA_G5_COUNT"]) for item in origin_summaries
    )
    uav_delayed = sum(int(item["UAV_DELAYED_BENEFIT_COUNT"]) for item in origin_summaries)
    cache_heads = [head for item in origin_summaries for head in item["cache_heads"]]
    uav_heads = [head for item in origin_summaries for head in item["uav_heads"]]
    cache_regrets = np.asarray([float(head["PPO_REGRET"]) for head in cache_heads])
    uav_regrets = np.asarray([float(head["PPO_REGRET"]) for head in uav_heads])
    cache_best_active = np.asarray(
        [float(item["CACHE_BEST_DELTA_G5"]) for item in origin_summaries],
        dtype=np.float64,
    )
    uav_best_active = np.asarray(
        [float(item["UAV_BEST_DELTA_G5"]) for item in origin_summaries],
        dtype=np.float64,
    )
    cache_miss_rate = float(np.mean(cache_regrets > tolerance)) if cache_regrets.size else 0.0
    uav_miss_rate = float(np.mean(uav_regrets > tolerance)) if uav_regrets.size else 0.0
    cache_opportunity = cache_positive_g5 > 0
    uav_opportunity = uav_positive_g5 > 0
    cache_missed = cache_miss_rate >= 0.5
    uav_missed = uav_miss_rate >= 0.5
    if cache_opportunity and uav_opportunity and cache_missed and uav_missed:
        final_case = "CASE_5_BOTH_HAVE_OPPORTUNITY"
        conclusion = (
            "Sampled local evidence shows positive realized H=5 actions for both "
            "Cache and UAV, while P50 policy-top choices miss them frequently."
        )
    elif cache_opportunity and not uav_opportunity:
        final_case = "CASE_3_CACHE_OPPORTUNITY_ONLY"
        conclusion = (
            "Sampled local evidence shows Cache opportunity but no positive UAV MOVE "
            "in the audited neighborhood; this is not a global proof."
        )
    elif uav_opportunity and not cache_opportunity:
        final_case = "CASE_4_UAV_OPPORTUNITY_ONLY"
        conclusion = (
            "Sampled local evidence shows UAV opportunity but no positive Cache action "
            "in the audited neighborhood; this is not a global proof."
        )
    elif (cache_opportunity or uav_opportunity) and (cache_missed or uav_missed):
        final_case = "CASE_1_ENVIRONMENT_HAS_OPPORTUNITY_POLICY_MISSES"
        conclusion = (
            "Sampled local evidence shows a positive realized action and a frequent "
            "P50 ranking miss; this points to action learning rather than a global "
            "training-length conclusion."
        )
    elif not cache_opportunity and not uav_opportunity:
        final_case = "CASE_2_STATIC_STATE_LOCALLY_STRONG"
        conclusion = (
            "Across the three sampled origins, no active primitive had positive realized "
            "H=5 value; this is local evidence that the canonical static state is strong, "
            "not a global proof."
        )
    else:
        final_case = "NO_DOMINANT_DIAGNOSTIC_CASE"
        conclusion = (
            "The sampled opportunity and P50 ranking evidence do not satisfy one of the "
            "dominant classifications."
        )
    return {
        "CACHE_ANY_POSITIVE_H5_ACTION": bool(cache_opportunity),
        "UAV_ANY_POSITIVE_H5_ACTION": bool(uav_opportunity),
        "CACHE_POSITIVE_H5_ACTION_FRACTION": float(cache_positive_g5 / max(cache_active, 1)),
        "UAV_POSITIVE_H5_ACTION_FRACTION": float(uav_positive_g5 / max(uav_active, 1)),
        "CACHE_ACTIVE_CANDIDATE_COUNT": int(cache_active),
        "CACHE_POSITIVE_DELTA_G1_COUNT": int(cache_positive_g1),
        "CACHE_POSITIVE_DELTA_G5_ACTION_COUNT": int(cache_positive_g5),
        "CACHE_MEAN_BEST_DELTA_G5": float(np.mean(cache_best_active)),
        "CACHE_MEAN_BEST_ACTIVE_DELTA_G5": float(np.mean(cache_best_active)),
        "CACHE_POLICY_MEAN_REGRET": float(np.mean(cache_regrets)) if cache_regrets.size else 0.0,
        "CACHE_POLICY_MISS_RATE": cache_miss_rate,
        "CACHE_DELAYED_BENEFIT_COUNT": int(cache_delayed),
        "UAV_MOVE_CANDIDATE_COUNT": int(uav_active),
        "UAV_POSITIVE_DELTA_G1_COUNT": int(uav_positive_g1),
        "UAV_POSITIVE_DELTA_G5_ACTION_COUNT": int(uav_positive_g5),
        "UAV_MEAN_BEST_DELTA_G5": float(np.mean(uav_best_active)),
        "UAV_MEAN_BEST_ACTIVE_DELTA_G5": float(np.mean(uav_best_active)),
        "UAV_POLICY_MEAN_REGRET": float(np.mean(uav_regrets)) if uav_regrets.size else 0.0,
        "UAV_POLICY_MISS_RATE": uav_miss_rate,
        "UAV_DELAYED_BENEFIT_COUNT": int(uav_delayed),
        "policy_miss_definition": "PPO_REGRET > numerical_tolerance for at least half of forced active heads",
        "FINAL_DIAGNOSTIC_CASE": final_case,
        "FINAL_CONCLUSION": conclusion,
    }


def run_audit(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    realization_dir: str | Path,
    output_path: str | Path,
    origins: tuple[int, ...] = DEFAULT_ORIGINS,
    counterfactual_horizon: int = DEFAULT_HORIZON,
    numerical_tolerance: float = DEFAULT_TOLERANCE,
    device: str = "cpu",
) -> dict[str, Any]:
    if torch is None:
        raise ImportError("action opportunity audit requires PyTorch")
    if tuple(origins) != DEFAULT_ORIGINS:
        raise ValueError("this audit is pinned to origins 60, 150, 240")
    if counterfactual_horizon != DEFAULT_HORIZON:
        raise ValueError("this audit is pinned to H_cf=5")
    if not math.isfinite(float(numerical_tolerance)) or numerical_tolerance <= 0.0:
        raise ValueError("numerical_tolerance must be a positive finite number")
    root = PROJECT_ROOT.resolve()
    resolved_config = _resolve_path(config_path, root)
    resolved_checkpoint = _resolve_path(checkpoint_path, root)
    resolved_realization = _resolve_path(realization_dir, root)
    resolved_output = _resolve_path(output_path, root)
    if resolved_output.exists():
        raise FileExistsError(
            "action opportunity audit output already exists; refusing to overwrite: "
            f"{resolved_output}"
        )
    if not resolved_config.is_file():
        raise FileNotFoundError(f"config is not readable: {resolved_config}")
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"P50 checkpoint is not readable: {resolved_checkpoint}")
    if not resolved_realization.is_dir():
        raise FileNotFoundError(
            f"development realization is not readable: {resolved_realization}"
        )

    audit_started_at = time.perf_counter()
    context = evaluator._build_full_episode_context(
        resolved_config,
        realization_dir=resolved_realization,
    )
    steps = evaluator._prepare_full_episode_steps(context)
    steps_by_origin = {int(step.origin_index): step for step in steps}
    expected_origins = tuple(range(300))
    if tuple(sorted(steps_by_origin)) != expected_origins:
        raise ValueError("audit realization must contain exactly origins 0..299")
    for origin in origins:
        if int(origin) + int(counterfactual_horizon) - 1 > 299:
            raise ValueError(f"audit horizon exceeds origin 299 at origin={origin}")

    config_sha256 = evaluator._sha256(resolved_config)
    training_git_sha = evaluator._load_full_episode_training_provenance(
        resolved_checkpoint,
        config_sha256,
    )
    evaluation_git_sha, evaluation_git_dirty = evaluator._git_identity(root)
    policy, checkpoint_payload = evaluator._load_frozen_v6_policy(
        context,
        resolved_checkpoint,
        config_sha256,
        torch.device(device),
        expected_completed_epochs=50,
        expected_protocol=evaluator.FULL_EPISODE_PROTOCOL,
    )
    policy.eval()
    initial_audit = _initial_state_audit(context)
    reference_states, reference_results, reference_runtime = _build_static_zero_reference(
        context,
        steps,
    )

    print(
        "ACTION_OPPORTUNITY_AUDIT reference=static_zero origins=60,150,240 H=5 "
        f"reference_runtime_seconds={reference_runtime:.3f}",
        flush=True,
    )
    origin_summaries: list[dict[str, Any]] = []
    policy_parameters_snapshot = [
        parameter.detach().clone() for parameter in policy.parameters()
    ]
    for origin in origins:
        origin_started_at = time.perf_counter()
        state = reference_states[int(origin)]
        baseline_results = [
            reference_results[int(origin) + offset]
            for offset in range(int(counterfactual_horizon))
        ]
        baseline = _summarize_results(baseline_results)
        baseline.update(
            {
                "G0_1": float(baseline["G1"]),
                "G0_5": float(baseline["G5"]),
                "Q0_1": float(baseline["Q1"]),
                "Q0_5": float(baseline["Q5"]),
            }
        )
        cache_heads: list[dict[str, Any]] = []
        uav_heads: list[dict[str, Any]] = []
        cache_runtime = 0.0
        uav_runtime = 0.0
        for node_index in range(len(context.action_spec.cache_node_ids)):
            head, elapsed = _cache_head_audit(
                context,
                policy,
                steps_by_origin,
                int(origin),
                node_index,
                state,
                baseline,
                counterfactual_horizon,
                numerical_tolerance,
            )
            cache_heads.append(head)
            cache_runtime += elapsed
            print(
                f"ACTION_OPPORTUNITY_AUDIT origin={origin} cache_node={head['node_id']} "
                f"active_candidates={head['active_candidate_count']} elapsed_seconds={elapsed:.3f}",
                flush=True,
            )
        for uav_index in range(len(context.action_spec.uav_ids)):
            head, elapsed = _uav_head_audit(
                context,
                policy,
                steps_by_origin,
                int(origin),
                uav_index,
                state,
                baseline,
                counterfactual_horizon,
                numerical_tolerance,
            )
            uav_heads.append(head)
            uav_runtime += elapsed
            print(
                f"ACTION_OPPORTUNITY_AUDIT origin={origin} uav={head['uav_id']} "
                f"active_candidates={head['active_candidate_count']} elapsed_seconds={elapsed:.3f}",
                flush=True,
            )
        origin_summary = _origin_summary(
            int(origin),
            baseline,
            cache_heads,
            uav_heads,
            state,
            numerical_tolerance,
            time.perf_counter() - origin_started_at,
        )
        origin_summary["runtime_breakdown_seconds"] = {
            "cache_counterfactuals": float(cache_runtime),
            "uav_counterfactuals": float(uav_runtime),
        }
        origin_summaries.append(origin_summary)
        print(
            f"ACTION_OPPORTUNITY_AUDIT origin={origin} "
            f"cache_positive_H5={origin_summary['CACHE_POSITIVE_DELTA_G5_COUNT']} "
            f"uav_positive_H5={origin_summary['UAV_POSITIVE_DELTA_G5_COUNT']} "
            f"elapsed_seconds={origin_summary['runtime_seconds']:.3f}",
            flush=True,
        )

    policy_parameters_updated = any(
        not torch.equal(before, after.detach())
        for before, after in zip(policy_parameters_snapshot, policy.parameters(), strict=True)
    )
    if policy_parameters_updated:
        raise RuntimeError("action opportunity audit modified policy parameters")
    aggregate = _aggregate_origin_summaries(origin_summaries, numerical_tolerance)
    output = {
        "provenance": {
            "diagnostic_role": "oracle_posthoc_action_value_diagnostic",
            "config_path": str(resolved_config),
            "config_sha256": config_sha256,
            "checkpoint_path": str(resolved_checkpoint),
            "checkpoint_sha256": evaluator._sha256(resolved_checkpoint),
            "checkpoint_completed_epochs": int(
                checkpoint_payload.get(
                    "completed_epochs",
                    checkpoint_payload.get("completed_episodes", -1),
                )
            ),
            "training_git_sha": training_git_sha,
            "evaluation_git_sha": evaluation_git_sha,
            "evaluation_git_dirty": evaluation_git_dirty,
            "realization_directory": str(resolved_realization),
            "realization_seed": 30001,
            "episode_slots": 300,
            "origin_range": [0, 299],
            "audit_origins": [int(value) for value in origins],
            "counterfactual_horizon": int(counterfactual_horizon),
            "numerical_tolerance": float(numerical_tolerance),
            "policy_action_selection": "P50 candidate conditional distribution; deterministic, no stochastic sample",
            "policy_observation_excludes_counterfactual_future_ground_truth": True,
            "counterfactual_future_ground_truth_used_only_for_posthoc_value": True,
            "training_started": False,
            "optimizer_steps": 0,
            "policy_parameters_updated": bool(policy_parameters_updated),
            "new_realization_generated": False,
        },
        "initial_state_audit": initial_audit,
        "origins": origin_summaries,
        "aggregate": aggregate,
        "runtime_seconds": {
            "reference_static_zero_trajectory": float(reference_runtime),
            "total_audit": float(time.perf_counter() - audit_started_at),
        },
    }
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    evaluator._write_json(resolved_output, output)
    print(f"ACTION_OPPORTUNITY_AUDIT output={resolved_output}", flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit realized H=5 value of feasible adaptive Cache/UAV primitives."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--realization-dir", default=DEFAULT_REALIZATION)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--origins", nargs="+", type=int, default=list(DEFAULT_ORIGINS))
    parser.add_argument("--counterfactual-horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--numerical-tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_audit(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            realization_dir=args.realization_dir,
            output_path=args.output,
            origins=tuple(int(value) for value in args.origins),
            counterfactual_horizon=int(args.counterfactual_horizon),
            numerical_tolerance=float(args.numerical_tolerance),
            device=args.device,
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
