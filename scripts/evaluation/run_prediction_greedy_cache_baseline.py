"""Prediction-driven, single-node greedy Cache baseline (Training-only runner).

The predictor supplies Monte Carlo arrival/service/workload scenarios and
vehicle positions. Candidate decisions use only those forecasts; realized
Training tasks and positions are used only after the decision to measure the
paired outcome. UAV positions are held fixed (STAY).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _import_root in (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "scripts" / "training",
):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from decision import cache_configuration, slot_environment, task_allocation
from ppo_simple.interface import prepare_aligned_rollout_steps
from ppo_simple.mappo_baseline import MACRO_INTERVAL_SLOTS


DEFAULT_EPISODE_SEED = 20031
DEFAULT_WINDOW_SLOTS = 10
DEFAULT_FORECAST_SAMPLE_BUDGET = 4
FORECAST_VALUE_HORIZON_SLOTS = 5
MAX_CACHE_CANDIDATE_EVALUATIONS_PER_MACRO = 9


@dataclass(frozen=True)
class CandidateEstimate:
    node_index: int
    target_cache: np.ndarray
    proposal_source: str
    estimated_net_utility: float
    estimated_net_gain: float
    predicted_task_utility: float
    predicted_cache_cost: float
    predicted_uav_cost: float


def _forecast_sample_indices(sample_count: int, budget: int) -> tuple[int, ...]:
    count = int(sample_count)
    limit = min(count, int(budget))
    if count <= 0 or limit <= 0:
        raise ValueError("Prediction scenarios must contain at least one sample")
    indices = np.rint(np.linspace(0, count - 1, limit)).astype(np.int64)
    return tuple(dict.fromkeys(int(value) for value in indices.tolist()))


def _forecast_tasks_and_positions(
    step: Any,
    *,
    sample_index: int,
    horizon_index: int,
) -> tuple[list[task_allocation.Task], dict[str, np.ndarray]]:
    """Materialize one forecast sample without touching realized task fields."""

    scenario = step.scenario_slice
    arrivals = np.asarray(scenario.get("arrival"))
    services = np.asarray(scenario.get("service_id"))
    input_sizes = np.asarray(scenario.get("input_size"), dtype=np.float64)
    cpu_cycles = np.asarray(scenario.get("cpu_cycles"), dtype=np.float64)
    positions = np.asarray(scenario.get("positions_xy"), dtype=np.float64)
    if (
        arrivals.ndim != 3
        or services.shape != arrivals.shape
        or input_sizes.shape != arrivals.shape
        or cpu_cycles.shape != arrivals.shape
    ):
        raise ValueError("Forecast task arrays must align as [scenario,horizon,task_vehicle]")
    if (
        positions.ndim != 4
        or positions.shape[:2] != arrivals.shape[:2]
        or positions.shape[-1] != 2
    ):
        raise ValueError("Forecast vehicle positions must align as [scenario,horizon,vehicle,2]")
    if not (0 <= sample_index < arrivals.shape[0]):
        raise IndexError("Forecast sample index is outside the prediction artifact")
    if not (0 <= horizon_index < arrivals.shape[1]):
        raise IndexError("Forecast horizon index is outside the prediction artifact")

    vehicle_ids = tuple(str(value) for value in scenario["vehicle_ids"].tolist())
    task_vehicle_ids = tuple(
        str(value) for value in scenario["task_vehicle_ids"].tolist()
    )
    if positions.shape[2] != len(vehicle_ids) or arrivals.shape[2] != len(task_vehicle_ids):
        raise ValueError("Forecast vehicle IDs do not align with prediction tensors")
    vehicle_positions = {
        vehicle_id: positions[sample_index, horizon_index, vehicle_index].copy()
        for vehicle_index, vehicle_id in enumerate(vehicle_ids)
    }

    tasks: list[task_allocation.Task] = []
    for task_index, vehicle_id in enumerate(task_vehicle_ids):
        if int(arrivals[sample_index, horizon_index, task_index]) != 1:
            continue
        service_id = int(services[sample_index, horizon_index, task_index])
        input_bits = float(input_sizes[sample_index, horizon_index, task_index])
        cycles = float(cpu_cycles[sample_index, horizon_index, task_index])
        if service_id < 0 or not all(
            math.isfinite(value) and value > 0.0 for value in (input_bits, cycles)
        ):
            raise ValueError("Forecast task has invalid service or workload values")
        tasks.append(
            task_allocation.Task(
                task_id=f"forecast_{step.origin_index}_{sample_index}_{horizon_index}_{vehicle_id}",
                vehicle_id=vehicle_id,
                service_id=service_id,
                input_bits=input_bits,
                cpu_cycles=cycles,
            )
        )
    if not all(np.isfinite(value).all() for value in vehicle_positions.values()):
        raise FloatingPointError("Forecast vehicle position is non-finite")
    return tasks, vehicle_positions


def _forecast_horizons(step: Any) -> tuple[int, ...]:
    scenario = step.scenario_slice
    horizon_count = int(np.asarray(scenario.get("arrival")).shape[1])
    if horizon_count <= 0:
        raise ValueError("Prediction artifact has no forecast horizons")
    mask_raw = getattr(step, "prediction_horizon_mask", None)
    if mask_raw is None:
        mask = np.ones(horizon_count, dtype=bool)
    else:
        mask = cache_configuration.normalize_prediction_horizon_mask(
            mask_raw, horizon_count
        )
    macro_horizons = tuple(
        int(index)
        for index in range(min(FORECAST_VALUE_HORIZON_SLOTS, horizon_count))
        if bool(mask[index])
    )
    if not macro_horizons:
        raise ValueError("No valid forecast horizon overlaps the Macro action duration")
    return macro_horizons


def _execute_configuration(
    context: Any,
    step: Any,
    previous_cache: np.ndarray,
    final_cache: np.ndarray,
    previous_uav: dict[str, np.ndarray],
    tasks: list[task_allocation.Task],
    vehicle_positions: dict[str, np.ndarray],
    *,
    slot_id: int,
    time_s: float,
) -> Any:
    result = slot_environment.execute_configured_slot(
        previous_cache,
        previous_uav,
        final_cache,
        previous_uav,
        step.scenario_slice,
        tasks,
        vehicle_positions,
        context.service_catalog,
        context.cfg,
        context.cache_nodes,
        context.cache_cfg,
        context.flight_boundary,
        slot_id=int(slot_id),
        time_s=float(time_s),
    )
    if not result.step_feasible or result.system_utility is None:
        raise RuntimeError(
            f"Matching+KKT rejected Cache baseline branch at slot {slot_id}: "
            f"{result.infeasible_reason}"
        )
    values = (result.system_utility, result.actual_task_utility, result.cache_cost, result.movement_cost)
    if any(value is None or not math.isfinite(float(value)) for value in values):
        raise FloatingPointError("Formal environment returned a non-finite baseline metric")
    if not np.array_equal(np.asarray(result.next_cache_state), final_cache):
        raise ValueError("Formal environment changed the requested Cache target")
    return result


def _forecast_configuration_metrics(
    context: Any,
    step: Any,
    current_cache: np.ndarray,
    target_cache: np.ndarray,
    fixed_uav: dict[str, np.ndarray],
    sample_indices: tuple[int, ...],
    horizons: tuple[int, ...],
) -> dict[str, float]:
    """Estimate fixed-Cache utility over up to five valid forecast slots."""

    scenario = step.scenario_slice
    dt_s = float(context.cfg["simulation"]["dt_s"])
    total = {key: 0.0 for key in ("net", "task", "cache", "uav")}
    for horizon_index in horizons:
        for sample_index in sample_indices:
            tasks, predicted_positions = _forecast_tasks_and_positions(
                step,
                sample_index=sample_index,
                horizon_index=horizon_index,
            )
            previous = current_cache if horizon_index == horizons[0] else target_cache
            result = _execute_configuration(
                context,
                step,
                previous,
                target_cache,
                fixed_uav,
                tasks,
                predicted_positions,
                slot_id=int(step.slot) + int(horizon_index),
                time_s=float(step.time_s) + float(horizon_index) * dt_s,
            )
            total["net"] += float(result.system_utility) / len(sample_indices)
            total["task"] += float(result.actual_task_utility) / len(sample_indices)
            total["cache"] += float(result.cache_cost) / len(sample_indices)
            total["uav"] += float(result.movement_cost) / len(sample_indices)
    if any(not math.isfinite(value) for value in total.values()):
        raise FloatingPointError("Forecast utility estimate is non-finite")
    return total


def _bounded_node_candidate(
    context: Any,
    step: Any,
    current_cache: np.ndarray,
    node_index: int,
) -> tuple[np.ndarray, str] | None:
    legal = cache_configuration.enumerate_all_cache_target_configurations(
        current_cache,
        int(node_index),
        context.service_catalog,
        context.cache_nodes,
        step.scenario_slice,
        context.cache_cfg,
        prediction_horizon_mask=step.prediction_horizon_mask,
    )
    if not legal:
        return None
    prediction_anchors = [item for item in legal if item.proposal_source == "prediction"]
    if prediction_anchors:
        selected = prediction_anchors[0]
        return np.asarray(selected.target_node_cache, dtype=np.int8).copy(), str(selected.proposal_source)

    demand = cache_configuration.node_specific_predicted_demand(
        step.scenario_slice,
        int(node_index),
        context.cache_nodes,
        context.cache_cfg,
        prediction_horizon_mask=step.prediction_horizon_mask,
    )
    selected = max(
        legal,
        key=lambda item: (
            float(np.dot(demand, np.asarray(item.target_node_cache, dtype=np.float64))),
            -int(np.count_nonzero(item.target_node_cache != current_cache[node_index])),
            tuple(-int(value) for value in item.target_node_cache.tolist()),
        ),
    )
    return np.asarray(selected.target_node_cache, dtype=np.int8).copy(), "predicted_demand_fallback"


def choose_cache_target(
    context: Any,
    step: Any,
    current_cache: np.ndarray,
    fixed_uav: dict[str, np.ndarray],
    *,
    forecast_sample_budget: int = DEFAULT_FORECAST_SAMPLE_BUDGET,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose the single best positive prediction-estimated node update."""

    current = np.asarray(current_cache, dtype=np.int8).copy()
    node_count = len(context.cache_nodes["node_ids"])
    if current.shape != (node_count, int(context.service_catalog["num_services"])):
        raise ValueError("Current Cache shape does not match the environment schema")
    scenario = step.scenario_slice
    arrivals = np.asarray(scenario.get("arrival"))
    if arrivals.ndim != 3:
        raise ValueError("Prediction arrival tensor must be [scenario,horizon,task_vehicle]")
    samples = _forecast_sample_indices(arrivals.shape[0], forecast_sample_budget)
    horizons = _forecast_horizons(step)

    # Evaluate KEEP once, then reuse this paired reference for every candidate.
    keep_metrics = _forecast_configuration_metrics(
        context, step, current, current, fixed_uav, samples, horizons
    )
    estimates: list[CandidateEstimate] = []
    for node_index in range(node_count):
        candidate = _bounded_node_candidate(context, step, current, node_index)
        if candidate is None:
            continue
        target_row, source = candidate
        target = current.copy()
        target[node_index] = target_row
        metrics = _forecast_configuration_metrics(
            context, step, current, target, fixed_uav, samples, horizons
        )
        estimates.append(
            CandidateEstimate(
                node_index=int(node_index),
                target_cache=target,
                proposal_source=source,
                estimated_net_utility=float(metrics["net"]),
                estimated_net_gain=float(metrics["net"] - keep_metrics["net"]),
                predicted_task_utility=float(metrics["task"]),
                predicted_cache_cost=float(metrics["cache"]),
                predicted_uav_cost=float(metrics["uav"]),
            )
        )
    if len(estimates) > MAX_CACHE_CANDIDATE_EVALUATIONS_PER_MACRO:
        raise AssertionError("Candidate evaluation budget exceeded the Cache node count")
    estimates.sort(
        key=lambda item: (
            -item.estimated_net_gain,
            item.node_index,
            tuple(int(value) for value in item.target_cache[item.node_index].tolist()),
        )
    )
    chosen = estimates[0] if estimates and estimates[0].estimated_net_gain > 0.0 else None
    target = current.copy() if chosen is None else chosen.target_cache.copy()
    diagnostics = {
        "slot": int(step.slot),
        "forecast_sample_indices": list(samples),
        "forecast_horizons": list(horizons),
        "candidate_evaluations": len(estimates),
        "candidate_evaluation_budget": MAX_CACHE_CANDIDATE_EVALUATIONS_PER_MACRO,
        "keep_predicted_metrics": keep_metrics,
        "ranked_candidates": [
            {
                "node_index": int(item.node_index),
                "proposal_source": item.proposal_source,
                "estimated_net_gain": float(item.estimated_net_gain),
                "predicted_task_utility": float(item.predicted_task_utility),
                "predicted_cache_cost": float(item.predicted_cache_cost),
                "predicted_uav_cost": float(item.predicted_uav_cost),
            }
            for item in estimates
        ],
        "selected_node_index": None if chosen is None else int(chosen.node_index),
        "selected_predicted_net_gain": 0.0 if chosen is None else float(chosen.estimated_net_gain),
    }
    return target, diagnostics


def _actual_metrics(
    context: Any,
    steps: list[Any],
    *,
    predictive_greedy: bool,
    forecast_sample_budget: int,
) -> dict[str, Any]:
    cache = np.asarray(context.initial_cache, dtype=np.int8).copy()
    uav = {
        str(key): np.asarray(value, dtype=np.float64).copy()
        for key, value in context.initial_uav.items()
    }
    initial_cache = cache.copy()
    initial_uav = {key: value.copy() for key, value in uav.items()}
    slot_net = 0.0
    slot_task = 0.0
    slot_cache_cost = 0.0
    slot_uav_cost = 0.0
    cache_node_changes = 0
    macro_update_count = 0
    decisions: list[dict[str, Any]] = []
    predicted_matching_calls = 0
    prediction_seconds = 0.0
    actual_execution_seconds = 0.0
    total_started = time.perf_counter()
    for index, step in enumerate(steps):
        if predictive_greedy and index % int(MACRO_INTERVAL_SLOTS) == 0:
            prediction_started = time.perf_counter()
            target, decision = choose_cache_target(
                context,
                step,
                cache,
                uav,
                forecast_sample_budget=forecast_sample_budget,
            )
            prediction_seconds += time.perf_counter() - prediction_started
            target = np.asarray(target, dtype=np.int8)
            if target.shape != cache.shape:
                raise ValueError("Greedy Cache target shape changed")
            changed_nodes = np.flatnonzero(np.any(target != cache, axis=1))
            if len(changed_nodes) > 1:
                raise ValueError("Greedy Cache baseline may update at most one node per Macro")
            if len(changed_nodes):
                macro_update_count += 1
            decision["selected_node_id"] = (
                None
                if not len(changed_nodes)
                else str(context.cache_nodes["node_ids"][int(changed_nodes[0])])
            )
            decisions.append(decision)
            predicted_matching_calls += int(decision["candidate_evaluations"] + 1) * len(
                decision["forecast_sample_indices"]
            ) * len(decision["forecast_horizons"])
        else:
            target = cache.copy()
        previous = cache.copy()
        actual_started = time.perf_counter()
        result = _execute_configuration(
            context,
            step,
            previous,
            target,
            uav,
            step.actual_tasks,
            step.actual_vehicle_positions,
            slot_id=int(step.slot),
            time_s=float(step.time_s),
        )
        actual_execution_seconds += time.perf_counter() - actual_started
        cache_node_changes += int(np.count_nonzero(np.any(target != previous, axis=1)))
        cache = np.asarray(result.next_cache_state, dtype=np.int8).copy()
        uav = {
            str(key): np.asarray(value, dtype=np.float64).copy()
            for key, value in result.next_uav_positions.items()
        }
        slot_net += float(result.system_utility)
        slot_task += float(result.actual_task_utility)
        slot_cache_cost += float(result.cache_cost)
        slot_uav_cost += float(result.movement_cost)
    total_seconds = time.perf_counter() - total_started
    if any(not np.array_equal(uav[key], initial_uav[key]) for key in uav):
        raise ValueError("UAV STAY baseline unexpectedly moved a UAV")
    decomposition_error = slot_net - (slot_task - slot_cache_cost - slot_uav_cost)
    if abs(decomposition_error) > 1.0e-8:
        raise ValueError(f"Net utility decomposition mismatch: {decomposition_error}")
    return {
        "slot_count": len(steps),
        "slot_start": int(steps[0].slot),
        "slot_end_inclusive": int(steps[-1].slot),
        "task_utility": float(slot_task),
        "cache_cost": float(slot_cache_cost),
        "uav_movement_cost": float(slot_uav_cost),
        "net_utility": float(slot_net),
        "actual_cache_node_changes": int(cache_node_changes),
        "macro_update_count": int(macro_update_count),
        "cache_change_rate_per_node_slot": float(cache_node_changes / max(len(steps) * cache.shape[0], 1)),
        "uav_movement_count": 0,
        "uav_stay_verified": True,
        "prediction_candidate_matching_calls": int(predicted_matching_calls),
        "prediction_estimation_runtime_seconds": float(prediction_seconds),
        "actual_matching_runtime_seconds": float(actual_execution_seconds),
        "total_runtime_seconds": float(total_seconds),
        "decomposition_abs_error": float(abs(decomposition_error)),
        "decisions": decisions,
        "final_cache_sha256": hashlib.sha256(np.ascontiguousarray(cache).tobytes()).hexdigest(),
        "initial_cache_sha256": hashlib.sha256(np.ascontiguousarray(initial_cache).tobytes()).hexdigest(),
    }


def _training_context_and_steps(
    project_root: Path,
    episode_seed: int,
    slot_count: int,
) -> tuple[Any, list[Any], dict[str, Any]]:
    import mappo_stage3_pool_contract as pool_contract
    from common import project_config

    pool, pool_info = pool_contract._validate_pool(
        project_root,
        pool_contract.TRAIN_POOL_RELATIVE,
        expected_split="train",
        expected_seed_start=pool_contract.DEFAULT_TRAINING_SEED_START,
        expected_episode_count=300,
    )
    pool_contract._validate_episode_snapshot(pool, episode_seed, pool_info)
    context = pool_contract._episode_context(project_root, pool, episode_seed, pool_info)
    if context.realization is None:
        raise ValueError("Prediction greedy baseline requires the reviewed Training realization")
    start = int(context.train_origin_start)
    slots = int(slot_count)
    if slots <= 0 or start + slots - 1 > int(context.train_origin_end):
        raise ValueError("Requested Training window is outside the reviewed Training origins")
    horizon = int(context.observation_spec.prediction_horizon)
    dt_s = float(project_config.get_config_value(context.cfg, "simulation.dt_s"))
    steps = prepare_aligned_rollout_steps(
        context.root,
        context.scenarios,
        start,
        slots,
        context.realization.vehicle_predictions,
        context.realization.popularity_predictions,
        context.realization.tasks,
        context.realization.trajectory,
        include_bootstrap_step=False,
        prediction_horizon=horizon,
        prediction_dt_s=dt_s,
        prediction_terminal_origin=int(context.train_origin_end),
        expected_vehicle_ids=context.observation_spec.vehicle_ids,
        expected_task_vehicle_ids=context.observation_spec.task_vehicle_ids,
        expected_service_ids=context.observation_spec.service_ids,
        include_prediction_horizon_mask=(context.observation_spec.horizon_mask_slice is not None),
    )
    if len(steps) != slots:
        raise ValueError("Aligned Training window has a different Slot count")
    slot_ids = [int(step.slot) for step in steps]
    if slot_ids != list(range(slot_ids[0], slot_ids[0] + slots)):
        raise ValueError("Training window Slots are not consecutive")
    return context, steps, pool_info


def run_paired_training_window(
    project_root: Path = PROJECT_ROOT,
    *,
    episode_seed: int = DEFAULT_EPISODE_SEED,
    slot_count: int = DEFAULT_WINDOW_SLOTS,
    forecast_sample_budget: int = DEFAULT_FORECAST_SAMPLE_BUDGET,
) -> dict[str, Any]:
    if int(forecast_sample_budget) <= 0:
        raise ValueError("forecast_sample_budget must be positive")
    context, steps, pool_info = _training_context_and_steps(
        project_root.resolve(), int(episode_seed), int(slot_count)
    )
    greedy = _actual_metrics(
        context,
        steps,
        predictive_greedy=True,
        forecast_sample_budget=int(forecast_sample_budget),
    )
    keep_stay = _actual_metrics(
        context,
        steps,
        predictive_greedy=False,
        forecast_sample_budget=int(forecast_sample_budget),
    )
    return {
        "episode_seed": int(episode_seed),
        "split": "train",
        "training_pool_manifest_sha256": pool_info["manifest_sha256"],
        "slot_count": int(slot_count),
        "macro_interval_slots": int(MACRO_INTERVAL_SLOTS),
        "forecast_horizon_requested": int(FORECAST_VALUE_HORIZON_SLOTS),
        "forecast_horizon_evaluated": max(
            (len(item["forecast_horizons"]) for item in greedy["decisions"]),
            default=0,
        ),
        "forecast_sample_budget": int(forecast_sample_budget),
        "max_candidate_evaluations_per_macro": int(MAX_CACHE_CANDIDATE_EVALUATIONS_PER_MACRO),
        "candidate_strategy": "one predicted-demand target per Cache node; max 9 single-node candidates per Macro",
        "uav_policy": "STAY",
        "greedy": greedy,
        "keep_stay": keep_stay,
        "paired_net_utility_difference": float(greedy["net_utility"] - keep_stay["net_utility"]),
        "paired_task_utility_difference": float(greedy["task_utility"] - keep_stay["task_utility"]),
        "paired_cache_cost_difference": float(greedy["cache_cost"] - keep_stay["cache_cost"]),
        "future_realized_tasks_used_for_decision": False,
        "validation_or_test_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-seed", type=int, default=DEFAULT_EPISODE_SEED)
    parser.add_argument("--slots", type=int, default=DEFAULT_WINDOW_SLOTS)
    parser.add_argument(
        "--forecast-sample-budget",
        type=int,
        default=DEFAULT_FORECAST_SAMPLE_BUDGET,
    )
    args = parser.parse_args(argv)
    result = run_paired_training_window(
        PROJECT_ROOT,
        episode_seed=args.episode_seed,
        slot_count=args.slots,
        forecast_sample_budget=args.forecast_sample_budget,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
