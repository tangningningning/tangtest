"""Prediction-Popularity Greedy Cache baseline evaluator."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from common import project_config
from decision import cache_configuration, slot_environment

from .evaluation_metrics import (
    aggregate_episode_task_metrics,
    infer_episode_seed,
    slot_task_metrics_dict,
    task_metrics_from_result,
)
from .interface import _copy_uav_positions
from .trainer import aligned_train_steps, load_environment_context


GREEDY_DATA_PROTOCOL = project_config.PPO_DATA_PROTOCOL_FULL_EPISODE_RANDOMIZED


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def _ensure_fresh_output(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(
            f"Popularity-greedy evaluation output is not a directory: {path}"
        )
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            "fresh popularity-greedy evaluation run-dir already contains artifacts; "
            "use a new --run-dir"
        )


def _validate_context(context: Any) -> None:
    protocol = project_config.get_ppo_data_protocol(context.cfg)
    if protocol["protocol"] != GREEDY_DATA_PROTOCOL:
        raise ValueError(
            "Popularity-greedy evaluation requires full_episode_randomized protocol"
        )
    if (
        int(context.train_origin_start),
        int(context.train_origin_end),
        int(context.episode_slots),
        int(context.terminal_origin),
    ) != (0, 299, 300, 299):
        raise ValueError(
            "Popularity-greedy evaluation requires the 0..299 episode protocol"
        )


def _prediction_horizon_average(
    predicted_service_popularity_horizon: Any,
    prediction_horizon_mask: Any,
    service_ids: tuple[int, ...],
) -> dict[int, float]:
    try:
        horizon_values = list(predicted_service_popularity_horizon)
    except TypeError as exc:
        raise ValueError("predicted popularity horizon must be iterable") from exc
    if not horizon_values:
        raise ValueError("predicted popularity horizon must not be empty")

    if prediction_horizon_mask is None:
        valid_mask = np.ones(len(horizon_values), dtype=bool)
    else:
        raw_mask = np.asarray(prediction_horizon_mask).reshape(-1)
        if raw_mask.shape != (len(horizon_values),) or not np.all(
            np.isin(raw_mask, [0, 1])
        ):
            raise ValueError("prediction_horizon_mask shape/value is invalid")
        valid_mask = raw_mask.astype(bool)
    valid_count = int(valid_mask.sum())
    if valid_count <= 0:
        raise ValueError("prediction horizon must contain at least one valid step")

    expected_keys = set(service_ids)
    total = np.zeros(len(service_ids), dtype=np.float64)
    for horizon_index, popularity in enumerate(horizon_values):
        if not valid_mask[horizon_index]:
            continue
        if not isinstance(popularity, Mapping) or set(popularity) != expected_keys:
            raise ValueError(
                "valid predicted popularity horizon must contain exactly all services"
            )
        values = np.asarray(
            [float(popularity[service_id]) for service_id in service_ids],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                f"valid popularity horizon {horizon_index} contains invalid values"
            )
        if not math.isclose(
            float(values.sum()),
            1.0,
            rel_tol=1.0e-5,
            abs_tol=1.0e-5,
        ):
            raise ValueError(
                f"valid popularity horizon {horizon_index} is not normalized"
            )
        total += values

    average = total / float(valid_count)
    if not np.all(np.isfinite(average)) or np.any(average < 0.0):
        raise AssertionError("average predicted popularity is invalid")
    return {
        service_id: float(average[index])
        for index, service_id in enumerate(service_ids)
    }


def build_popularity_greedy_cache_target(
    predicted_service_popularity_horizon: Any,
    prediction_horizon_mask: Any,
    service_catalog: Mapping[str, Any],
    cache_nodes: Mapping[str, Any],
) -> np.ndarray:
    """Build one independent, popularity-ranked, capacity-feasible cache target."""

    service_ids = tuple(int(value) for value in service_catalog["service_ids"])
    service_sizes = np.asarray(service_catalog["service_sizes_mb"], dtype=np.float64)
    node_ids = tuple(str(value) for value in cache_nodes["node_ids"])
    capacities = np.asarray(cache_nodes["capacities_mb"], dtype=np.float64)
    if len(service_ids) != len(service_sizes):
        raise ValueError("service IDs and service sizes have inconsistent lengths")
    if len(node_ids) != len(capacities):
        raise ValueError("cache node IDs and capacities have inconsistent lengths")
    if not np.all(np.isfinite(service_sizes)) or np.any(service_sizes < 0.0):
        raise ValueError("service program sizes must be finite and non-negative")
    if not np.all(np.isfinite(capacities)) or np.any(capacities < 0.0):
        raise ValueError("cache capacities must be finite and non-negative")

    average_popularity = _prediction_horizon_average(
        predicted_service_popularity_horizon,
        prediction_horizon_mask,
        service_ids,
    )
    ranking = sorted(
        range(len(service_ids)),
        key=lambda index: (
            -float(average_popularity[service_ids[index]]),
            int(service_ids[index]),
        ),
    )
    target = np.zeros((len(node_ids), len(service_ids)), dtype=np.int8)
    for node_index, capacity in enumerate(capacities):
        remaining = float(capacity)
        for service_index in ranking:
            size = float(service_sizes[service_index])
            if size <= remaining + 1.0e-9:
                target[node_index, service_index] = 1
                remaining -= size
    cache_configuration.validate_cache_shape_and_capacity(
        target,
        dict(service_catalog),
        dict(cache_nodes),
    )
    return target


def _assert_fixed_uav_state(
    previous_uav: dict[str, tuple[float, float]],
    result: slot_environment.SlotExecutionResult,
) -> dict[str, tuple[float, float]]:
    if result.next_uav_positions is None:
        raise AssertionError("popularity-greedy result is missing next UAV state")
    next_uav = _copy_uav_positions(result.next_uav_positions)
    if set(next_uav) != set(previous_uav):
        raise AssertionError("popularity-greedy baseline changed UAV IDs")
    for uav_id in previous_uav:
        if not np.array_equal(
            np.asarray(next_uav[uav_id], dtype=np.float64),
            np.asarray(previous_uav[uav_id], dtype=np.float64),
        ):
            raise AssertionError(f"popularity-greedy baseline moved UAV {uav_id}")
    return next_uav


def _slot_reward_metrics(
    result: slot_environment.SlotExecutionResult,
) -> tuple[float, float, float, float]:
    if (
        not result.step_feasible
        or result.system_utility is None
        or result.actual_task_utility is None
        or result.cache_cost is None
        or result.movement_cost is None
    ):
        raise RuntimeError(
            "popularity-greedy baseline produced an incomplete/infeasible slot: "
            f"slot={result.slot_id}, reason={result.infeasible_reason}"
        )
    reward = float(result.system_utility)
    q = float(result.actual_task_utility)
    cache_cost = float(result.cache_cost)
    movement_cost = float(result.movement_cost)
    if not all(np.isfinite(value) for value in (reward, q, cache_cost, movement_cost)):
        raise FloatingPointError("popularity-greedy reward metrics are non-finite")
    if not np.isclose(
        reward,
        q - cache_cost - movement_cost,
        rtol=1.0e-12,
        atol=1.0e-9,
    ):
        raise AssertionError("popularity-greedy reward accounting failed")
    if not np.isclose(movement_cost, 0.0, rtol=0.0, atol=1.0e-12):
        raise AssertionError("popularity-greedy movement cost is non-zero")
    return reward, q, cache_cost, movement_cost


def evaluate_popularity_greedy_episode(
    run_dir: str | Path,
    config_path: str | Path = "configs/baselines/system_config_ppo_v8_cache_operation_group_replica_aware.toml",
    realization_dir: str | Path | None = None,
    episode_seed: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate Prediction-Popularity Greedy Cache on one complete episode."""

    project_root = project_config.find_project_root()
    resolved_config = _resolve_path(project_root, config_path)
    output_dir = _resolve_path(project_root, run_dir)
    _ensure_fresh_output(output_dir)
    context = load_environment_context(
        str(resolved_config),
        realization_dir=realization_dir,
    )
    _validate_context(context)
    resolved_episode_seed = None if episode_seed is None else int(episode_seed)
    if resolved_episode_seed is None:
        resolved_episode_seed = infer_episode_seed(realization_dir)
    if resolved_episode_seed is None:
        raise ValueError(
            "popularity-greedy evaluation requires an explicit episode_seed or "
            "episode_<seed> realization"
        )
    experiment_seed = int(
        project_config.get_config_value(context.cfg, "experiment.seed")
    )
    steps = aligned_train_steps(context)
    if len(steps) != 300 or tuple(int(step.origin_index) for step in steps) != tuple(range(300)):
        raise ValueError(
            "popularity-greedy evaluation steps must cover origins 0..299 exactly"
        )

    previous_cache = np.asarray(context.initial_cache, dtype=np.int8).copy()
    previous_uav = _copy_uav_positions(context.initial_uav)
    rewards: list[float] = []
    q_values: list[float] = []
    cache_costs: list[float] = []
    movement_costs: list[float] = []
    task_metrics: list[Any] = []
    cache_change_counts: list[int] = []
    services_added: list[int] = []
    services_removed: list[int] = []
    slot_rows: list[dict[str, Any]] = []

    for step in steps:
        target_cache = build_popularity_greedy_cache_target(
            step.predicted_service_popularity_horizon,
            step.prediction_horizon_mask,
            context.service_catalog,
            context.cache_nodes,
        )
        cache_triggers = np.any(target_cache != previous_cache, axis=1).astype(np.int8)
        uav_triggers = {uav_id: 0 for uav_id in context.action_spec.uav_ids}
        configuration_override = slot_environment.build_policy_primitive_configuration(
            previous_cache,
            previous_uav,
            target_cache,
            previous_uav,
            cache_triggers,
            uav_triggers,
            context.service_catalog,
            context.cfg,
            context.cache_nodes,
            context.cache_cfg,
            context.flight_boundary,
        )
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
            slot_id=step.slot,
            time_s=step.time_s,
            configuration_override=configuration_override,
        )
        reward, q, cache_cost, movement_cost = _slot_reward_metrics(result)
        if result.next_cache_state is None:
            raise AssertionError("popularity-greedy result is missing next Cache state")
        next_cache = np.asarray(result.next_cache_state, dtype=np.int8).copy()
        if not np.array_equal(next_cache, target_cache):
            raise AssertionError("execute_slot did not apply the greedy cache target")
        next_uav = _assert_fixed_uav_state(previous_uav, result)
        cache_diff = next_cache != previous_cache
        added = int(np.sum((previous_cache == 0) & (next_cache == 1)))
        removed = int(np.sum((previous_cache == 1) & (next_cache == 0)))
        cache_change_count = int(np.any(cache_diff, axis=1).sum())
        slot_task_metrics = task_metrics_from_result(result, context.cache_nodes)
        slot_task_summary = slot_task_metrics_dict(slot_task_metrics)
        rewards.append(reward)
        q_values.append(q)
        cache_costs.append(cache_cost)
        movement_costs.append(movement_cost)
        task_metrics.append(slot_task_metrics)
        cache_change_counts.append(cache_change_count)
        services_added.append(added)
        services_removed.append(removed)
        slot_rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "episode_seed": int(resolved_episode_seed),
                "origin": int(step.origin_index),
                "reward": reward,
                "q": q,
                "cache_cost": cache_cost,
                "movement_cost": movement_cost,
                "cache_change_count": cache_change_count,
                "services_added": added,
                "services_removed": removed,
                "uav_movement_count": 0,
                **slot_task_summary,
            }
        )
        previous_cache = next_cache
        previous_uav = next_uav

    if len(rewards) != 300:
        raise AssertionError("popularity-greedy evaluation must contain 300 transitions")
    task_summary = aggregate_episode_task_metrics(task_metrics)
    total_return = float(np.sum(np.asarray(rewards, dtype=np.float64)))
    total_q = float(np.sum(np.asarray(q_values, dtype=np.float64)))
    total_cache_cost = float(np.sum(np.asarray(cache_costs, dtype=np.float64)))
    total_movement_cost = float(np.sum(np.asarray(movement_costs, dtype=np.float64)))
    cache_node_slots = float(300 * len(context.action_spec.cache_node_ids))
    summary = {
        "experiment_seed": int(experiment_seed),
        "episode_seed": int(resolved_episode_seed),
        "episode_slots": 300,
        "origin_start": 0,
        "origin_end": 299,
        "num_transitions": len(rewards),
        "total_return": total_return,
        "total_q": total_q,
        "total_cache_cost": total_cache_cost,
        "total_movement_cost": total_movement_cost,
        "cache_change_count": int(sum(cache_change_counts)),
        "cache_change_rate": float(sum(cache_change_counts)) / cache_node_slots,
        "mean_services_added_per_slot": float(np.mean(np.asarray(services_added, dtype=np.float64))),
        "mean_services_removed_per_slot": float(np.mean(np.asarray(services_removed, dtype=np.float64))),
        "uav_movement_count": 0,
        "uav_movement_rate": 0.0,
        "policy_dependency": "NONE",
        "prediction_source": "step.predicted_service_popularity_horizon",
        "horizon_aggregation": "arithmetic_mean_over_valid_prediction_horizon",
        **task_summary,
    }
    if not np.isclose(total_return, total_q - total_cache_cost, rtol=1.0e-12, atol=1.0e-9):
        raise AssertionError("popularity-greedy total reward accounting failed")
    if not np.isclose(total_movement_cost, 0.0, rtol=0.0, atol=1.0e-12):
        raise AssertionError("popularity-greedy total movement cost is non-zero")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_type": "prediction_popularity_greedy_full_episode_evaluation",
        "config_path": str(resolved_config),
        "config_sha256": _sha256(resolved_config),
        "experiment_seed": int(experiment_seed),
        "episode_seed": int(resolved_episode_seed),
        "data_protocol": GREEDY_DATA_PROTOCOL,
        "episode_slots": 300,
        "origin_start": 0,
        "origin_end": 299,
        "terminal_origin": 299,
        "policy_dependency": "NONE",
        "prediction_source": "step.predicted_service_popularity_horizon",
        "invalid_horizon_excluded": True,
        "cache_ranking": "descending_average_predicted_popularity_then_service_id_ascending",
        "cache_target": "independent_per_node_greedy_capacity_fill",
        "uav_fixed": True,
        "uav_triggers": "all_zero",
        "optimizer_steps": 0,
    }
    _write_json(output_dir / "popularity_greedy_episode_manifest.json", manifest)
    _write_json(output_dir / "popularity_greedy_episode_summary.json", summary)
    with (output_dir / "popularity_greedy_slot_metrics.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for row in slot_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return manifest, summary
