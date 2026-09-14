"""Explicit frozen Static Cache baseline evaluator.

This evaluator has no policy or checkpoint dependency.  It keeps the canonical
initial Cache/UAV state fixed and delegates every slot's real task execution to
``slot_environment.execute_slot``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from common import project_config
from decision import slot_environment

from .evaluation_metrics import (
    aggregate_episode_task_metrics,
    infer_episode_seed,
    slot_task_metrics_dict,
    task_metrics_from_result,
)
from .interface import _copy_uav_positions
from .trainer import aligned_train_steps, load_environment_context


STATIC_DATA_PROTOCOL = project_config.PPO_DATA_PROTOCOL_FULL_EPISODE_RANDOMIZED


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
        raise ValueError(f"Static evaluation output is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            "fresh Static evaluation run-dir already contains artifacts; "
            "use a new --run-dir"
        )


def _validate_static_context(context: Any) -> None:
    protocol = project_config.get_ppo_data_protocol(context.cfg)
    if protocol["protocol"] != STATIC_DATA_PROTOCOL:
        raise ValueError("Static evaluation requires full_episode_randomized protocol")
    if (
        int(context.train_origin_start),
        int(context.train_origin_end),
        int(context.episode_slots),
        int(context.terminal_origin),
    ) != (0, 299, 300, 299):
        raise ValueError("Static evaluation requires the 0..299 episode protocol")


def _assert_fixed_state(
    previous_cache: np.ndarray,
    previous_uav: dict[str, tuple[float, float]],
    result: slot_environment.SlotExecutionResult,
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    if not result.step_feasible:
        raise RuntimeError(
            "Static baseline produced an infeasible slot: "
            f"slot={result.slot_id}, reason={result.infeasible_reason}"
        )
    if result.next_cache_state is None or result.next_uav_positions is None:
        raise AssertionError("Static baseline result is missing next state")
    next_cache = np.asarray(result.next_cache_state, dtype=np.int8).copy()
    next_uav = _copy_uav_positions(result.next_uav_positions)
    if not np.array_equal(next_cache, previous_cache):
        raise AssertionError("Static baseline changed Cache state")
    if set(next_uav) != set(previous_uav):
        raise AssertionError("Static baseline changed the UAV ID set")
    for uav_id in previous_uav:
        if not np.array_equal(
            np.asarray(next_uav[uav_id], dtype=np.float64),
            np.asarray(previous_uav[uav_id], dtype=np.float64),
        ):
            raise AssertionError(f"Static baseline moved UAV {uav_id}")
    return next_cache, next_uav


def _assert_zero_costs_and_reward(
    result: slot_environment.SlotExecutionResult,
) -> tuple[float, float, float, float]:
    if (
        result.system_utility is None
        or result.actual_task_utility is None
        or result.cache_cost is None
        or result.movement_cost is None
    ):
        raise AssertionError("Static baseline result is missing reward metrics")
    reward = float(result.system_utility)
    q = float(result.actual_task_utility)
    cache_cost = float(result.cache_cost)
    movement_cost = float(result.movement_cost)
    if not all(np.isfinite(value) for value in (reward, q, cache_cost, movement_cost)):
        raise FloatingPointError("Static baseline reward metrics are non-finite")
    if not np.isclose(cache_cost, 0.0, rtol=0.0, atol=1.0e-12):
        raise AssertionError(f"Static baseline cache cost is non-zero: {cache_cost}")
    if not np.isclose(movement_cost, 0.0, rtol=0.0, atol=1.0e-12):
        raise AssertionError(
            f"Static baseline movement cost is non-zero: {movement_cost}"
        )
    if not np.isclose(
        reward,
        q - cache_cost - movement_cost,
        rtol=1.0e-12,
        atol=1.0e-9,
    ):
        raise AssertionError("Static baseline reward/Q/cost identity failed")
    return reward, q, cache_cost, movement_cost


def _summary(
    context: Any,
    episode_seed: int,
    experiment_seed: int,
    rewards: list[float],
    q_values: list[float],
    cache_costs: list[float],
    movement_costs: list[float],
    task_metrics: list[Any],
) -> dict[str, Any]:
    task_summary = aggregate_episode_task_metrics(task_metrics)
    total_return = float(np.sum(np.asarray(rewards, dtype=np.float64)))
    total_q = float(np.sum(np.asarray(q_values, dtype=np.float64)))
    total_cache_cost = float(np.sum(np.asarray(cache_costs, dtype=np.float64)))
    total_movement_cost = float(np.sum(np.asarray(movement_costs, dtype=np.float64)))
    if not np.isclose(total_return, total_q, rtol=1.0e-12, atol=1.0e-9):
        raise AssertionError("Static baseline total_return must equal total_q")
    if not np.isclose(total_cache_cost, 0.0, rtol=0.0, atol=1.0e-12):
        raise AssertionError("Static baseline total cache cost is non-zero")
    if not np.isclose(total_movement_cost, 0.0, rtol=0.0, atol=1.0e-12):
        raise AssertionError("Static baseline total movement cost is non-zero")
    return {
        "experiment_seed": int(experiment_seed),
        "episode_seed": int(episode_seed),
        "episode_slots": 300,
        "origin_start": 0,
        "origin_end": 299,
        "num_transitions": len(rewards),
        "total_return": total_return,
        "total_q": total_q,
        "total_cache_cost": total_cache_cost,
        "total_movement_cost": total_movement_cost,
        "cache_change_count": 0,
        "cache_change_rate": 0.0,
        "uav_movement_count": 0,
        "uav_movement_rate": 0.0,
        **task_summary,
        "policy_dependency": "NONE",
        "data_protocol": STATIC_DATA_PROTOCOL,
        "policy_schema": None,
        "configuration_control_mode": str(context.configuration_control_mode),
    }


def evaluate_static_episode(
    run_dir: str | Path,
    config_path: str | Path = "configs/baselines/system_config_ppo_v8_cache_operation_group_replica_aware.toml",
    realization_dir: str | Path | None = None,
    episode_seed: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate the explicit fixed-state baseline on one complete episode."""

    project_root = project_config.find_project_root()
    resolved_config = _resolve_path(project_root, config_path)
    output_dir = _resolve_path(project_root, run_dir)
    _ensure_fresh_output(output_dir)
    context = load_environment_context(
        str(resolved_config),
        realization_dir=realization_dir,
    )
    _validate_static_context(context)
    resolved_episode_seed = None if episode_seed is None else int(episode_seed)
    if resolved_episode_seed is None:
        resolved_episode_seed = infer_episode_seed(realization_dir)
    if resolved_episode_seed is None:
        raise ValueError(
            "Static evaluation requires an explicit episode_seed or "
            "episode_<seed> realization"
        )
    experiment_seed = int(
        project_config.get_config_value(context.cfg, "experiment.seed")
    )
    steps = aligned_train_steps(context)
    if len(steps) != 300 or tuple(int(step.origin_index) for step in steps) != tuple(range(300)):
        raise ValueError("Static evaluation steps must cover origins 0..299 exactly")

    previous_cache = np.asarray(context.initial_cache, dtype=np.int8).copy()
    previous_uav = _copy_uav_positions(context.initial_uav)
    rewards: list[float] = []
    q_values: list[float] = []
    cache_costs: list[float] = []
    movement_costs: list[float] = []
    task_metrics: list[Any] = []
    slot_rows: list[dict[str, Any]] = []
    for step in steps:
        cache_triggers = np.zeros(
            len(context.action_spec.cache_node_ids),
            dtype=np.int8,
        )
        uav_triggers = {uav_id: 0 for uav_id in context.action_spec.uav_ids}
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
        )
        reward, q, cache_cost, movement_cost = _assert_zero_costs_and_reward(result)
        previous_cache, previous_uav = _assert_fixed_state(
            previous_cache,
            previous_uav,
            result,
        )
        slot_task_metrics = task_metrics_from_result(result, context.cache_nodes)
        slot_task_summary = slot_task_metrics_dict(slot_task_metrics)
        rewards.append(reward)
        q_values.append(q)
        cache_costs.append(cache_cost)
        movement_costs.append(movement_cost)
        task_metrics.append(slot_task_metrics)
        slot_rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "episode_seed": int(resolved_episode_seed),
                "origin": int(step.origin_index),
                "reward": reward,
                "q": q,
                "cache_cost": cache_cost,
                "movement_cost": movement_cost,
                **slot_task_summary,
            }
        )

    if len(rewards) != 300:
        raise AssertionError("Static evaluation must contain 300 transitions")
    summary = _summary(
        context,
        resolved_episode_seed,
        experiment_seed,
        rewards,
        q_values,
        cache_costs,
        movement_costs,
        task_metrics,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_type": "static_full_episode_evaluation",
        "config_path": str(resolved_config),
        "config_sha256": _sha256(resolved_config),
        "experiment_seed": int(experiment_seed),
        "episode_seed": int(resolved_episode_seed),
        "data_protocol": STATIC_DATA_PROTOCOL,
        "episode_slots": 300,
        "origin_start": 0,
        "origin_end": 299,
        "terminal_origin": 299,
        "policy_dependency": "NONE",
        "cache_fixed": True,
        "uav_fixed": True,
        "cache_triggers": "all_zero",
        "uav_triggers": "all_zero",
        "optimizer_steps": 0,
    }
    _write_json(output_dir / "static_episode_manifest.json", manifest)
    _write_json(output_dir / "static_episode_summary.json", summary)
    with (output_dir / "static_slot_metrics.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for row in slot_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return manifest, summary
