"""Frozen full-episode evaluator for the V7 Cache-only policy."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - import-only environments
    torch = None  # type: ignore[assignment]

from common import project_config

from . import interface
from .evaluation_metrics import (
    aggregate_episode_task_metrics,
    infer_episode_seed,
    slot_task_metrics_dict,
)
from .interface import _copy_uav_positions
from .policy import AdaptiveCacheOperationGroupV7PopArtActorCritic
from .rollout import collect_rollout
from .trainer import (
    PPOV2Trainer,
    V7_POLICY_SCHEMAS,
    aligned_train_steps,
    load_environment_context,
)


V7_OPERATION_ORDER = ("NO_CHANGE", "ADD", "DELETE", "SWAP")
V7_POLICY_ACTION_MODES = ("deterministic", "stochastic")


def _require_torch() -> None:
    if torch is None:
        raise ImportError("V7 evaluation requires PyTorch")


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
        raise ValueError(f"V7 evaluation output is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            "fresh V7 evaluation run-dir already contains evaluation artifacts; "
            "use a new --run-dir"
        )


def _seed_sampling(seed: int) -> None:
    value = int(seed)
    if value < 0:
        raise ValueError("policy_sampling_seed must be non-negative")
    random.seed(value)
    np.random.seed(value)
    _require_torch()
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _validate_v7_context(context: Any) -> None:
    if context.configuration_control_mode != "policy_primitive_v7_cache_operation":
        raise ValueError(
            "V7 evaluation requires configuration_control_mode="
            "policy_primitive_v7_cache_operation"
        )
    if context.policy_schema not in V7_POLICY_SCHEMAS:
        raise ValueError(
            "V7 evaluation requires policy_schema=adaptive_cache_operation_group_v7_popart"
        )
    protocol = project_config.get_ppo_data_protocol(context.cfg)
    if protocol["protocol"] != project_config.PPO_DATA_PROTOCOL_FULL_EPISODE_RANDOMIZED:
        raise ValueError("V7 evaluation requires full_episode_randomized protocol")
    if (
        context.train_origin_start,
        context.train_origin_end,
        context.episode_slots,
        context.terminal_origin,
    ) != (0, 299, 300, 299):
        raise ValueError("V7 evaluation requires the 0..299 episode protocol")
    if int(context.observation_spec.prediction_horizon) != 5:
        raise ValueError("V7 evaluation requires prediction horizon H=5")
    if context.observation_spec.horizon_mask_slice is None:
        raise ValueError("V7 evaluation requires a prediction horizon mask")


def _summary(
    context: Any,
    rollout: Any,
    policy_action_mode: str,
    *,
    experiment_seed: int,
    episode_seed: int,
) -> dict[str, Any]:
    if rollout.v7_cache_operation_actions is None:
        raise ValueError("V7 rollout is missing operation actions")
    if rollout.v7_cache_operation_masks is None:
        raise ValueError("V7 rollout is missing operation masks")
    if len(rollout.transitions) != 300:
        raise ValueError("V7 evaluation must contain 300 transitions")
    operation_actions = np.asarray(rollout.v7_cache_operation_actions, dtype=np.int64)
    operation_masks = np.asarray(rollout.v7_cache_operation_masks, dtype=bool)
    cache_count = len(context.action_spec.cache_node_ids)
    uav_count = len(context.action_spec.uav_ids)
    if operation_actions.shape != (300, cache_count):
        raise ValueError("V7 operation action shape mismatch")
    if operation_masks.shape != (300, cache_count, 4):
        raise ValueError("V7 operation mask shape mismatch")
    if not bool(np.all(operation_masks[np.arange(300)[:, None], np.arange(cache_count)[None, :], operation_actions])):
        raise ValueError("V7 evaluation found an operation outside its mask")
    counts = np.bincount(operation_actions.reshape(-1), minlength=4)
    total_cache_heads = 300 * cache_count
    operation_rates = {
        name: float(counts[index]) / float(total_cache_heads)
        for index, name in enumerate(V7_OPERATION_ORDER)
    }
    total_return = float(np.asarray(rollout.rewards, dtype=np.float64).sum())
    total_q = float(
        sum(float(item.actual_task_utility) for item in rollout.transitions)
    )
    total_cache_cost = float(
        sum(float(item.cache_cost) for item in rollout.transitions)
    )
    total_movement_cost = float(
        sum(float(item.movement_cost) for item in rollout.transitions)
    )
    actual_cache_updates = int(
        sum(int(item.actual_cache_update_count) for item in rollout.transitions)
    )
    actual_uav_movements = int(
        sum(int(item.actual_uav_movement_count) for item in rollout.transitions)
    )
    if actual_uav_movements != 0 or total_movement_cost != 0.0:
        raise AssertionError("V7 Cache-only evaluation moved a UAV")
    task_metrics = aggregate_episode_task_metrics(rollout.transitions)
    return {
        "experiment_seed": int(experiment_seed),
        "episode_seed": int(episode_seed),
        "policy_schema": str(context.policy_schema),
        "policy_action_mode": policy_action_mode,
        "episode_slots": 300,
        "origin_start": 0,
        "origin_end": 299,
        "num_transitions": 300,
        "total_return": total_return,
        "total_q": total_q,
        "total_cache_cost": total_cache_cost,
        "total_movement_cost": total_movement_cost,
        "cache_change_rate": float(actual_cache_updates) / float(total_cache_heads),
        "operation_rates": operation_rates,
        "uav_movement_rate": float(actual_uav_movements)
        / float(300 * uav_count),
        "uav_movement_count": actual_uav_movements,
        **task_metrics,
        "selected_action_sha256": hashlib.sha256(
            np.ascontiguousarray(rollout.actions, dtype=np.int8).tobytes(order="C")
        ).hexdigest(),
    }


def evaluate_v7_episode(
    checkpoint_path: str | Path,
    run_dir: str | Path,
    config_path: str | Path = "configs/baselines/system_config_ppo_v7_cache_operation_group.toml",
    realization_dir: str | Path | None = None,
    device: str = "cpu",
    policy_action_mode: str = "deterministic",
    policy_sampling_seed: int | None = None,
    episode_seed: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate one frozen V7 policy on one complete 0..299 realization."""

    _require_torch()
    mode = str(policy_action_mode)
    if mode not in V7_POLICY_ACTION_MODES:
        raise ValueError("V7 policy_action_mode must be deterministic or stochastic")
    project_root = project_config.find_project_root()
    resolved_config = _resolve_path(project_root, config_path)
    resolved_checkpoint = _resolve_path(project_root, checkpoint_path)
    output_dir = _resolve_path(project_root, run_dir)
    _ensure_fresh_output(output_dir)
    context = load_environment_context(
        str(resolved_config),
        realization_dir=realization_dir,
    )
    _validate_v7_context(context)
    resolved_episode_seed = (
        None if episode_seed is None else int(episode_seed)
    )
    if resolved_episode_seed is None:
        resolved_episode_seed = infer_episode_seed(realization_dir)
    if resolved_episode_seed is None:
        raise ValueError(
            "V7 evaluation requires an explicit episode_seed or episode_<seed> realization"
        )
    experiment_seed = int(
        project_config.get_config_value(context.cfg, "experiment.seed")
    )
    trainer = PPOV2Trainer(context, device=device)
    config_sha256 = _sha256(resolved_config)
    cursor = trainer.load_checkpoint(resolved_checkpoint, config_sha256)
    if not isinstance(trainer.policy, AdaptiveCacheOperationGroupV7PopArtActorCritic):
        raise TypeError("V7 checkpoint did not restore the V7 Cache policy")
    trainer.policy.eval()
    steps = aligned_train_steps(context)
    if policy_sampling_seed is not None:
        _seed_sampling(int(policy_sampling_seed))
    parameter_snapshot = [parameter.detach().clone() for parameter in trainer.policy.parameters()]
    popart_snapshot = trainer.policy.popart_statistics()
    with torch.inference_mode():
        rollout = collect_rollout(
            trainer.policy,
            context,
            steps,
            initial_cache=np.asarray(context.initial_cache, dtype=np.int8).copy(),
            initial_uav=_copy_uav_positions(context.initial_uav),
            terminal_origin_index=299,
            deterministic=mode == "deterministic",
        )
    if any(
        not torch.equal(before, after.detach())
        for before, after in zip(parameter_snapshot, trainer.policy.parameters(), strict=True)
    ):
        raise RuntimeError("V7 evaluation modified policy parameters")
    if trainer.policy.popart_statistics() != popart_snapshot:
        raise RuntimeError("V7 evaluation modified PopArt statistics")
    summary = _summary(
        context,
        rollout,
        mode,
        experiment_seed=experiment_seed,
        episode_seed=resolved_episode_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_type": "v7_full_episode_evaluation",
        "config_path": str(resolved_config),
        "config_sha256": config_sha256,
        "checkpoint_path": str(resolved_checkpoint),
        "checkpoint_completed_epochs": int(cursor.completed_episodes),
        "experiment_seed": int(experiment_seed),
        "episode_seed": int(resolved_episode_seed),
        "policy_schema": str(context.policy_schema),
        "configuration_control_mode": str(context.configuration_control_mode),
        "data_protocol": project_config.PPO_DATA_PROTOCOL_FULL_EPISODE_RANDOMIZED,
        "episode_slots": 300,
        "origin_start": 0,
        "origin_end": 299,
        "terminal_origin": 299,
        "policy_action_mode": mode,
        "policy_sampling_seed": policy_sampling_seed,
        "optimizer_steps": 0,
        "uav_policy_active": False,
        "uav_fixed": True,
    }
    _write_json(output_dir / "v7_episode_manifest.json", manifest)
    _write_json(output_dir / "v7_episode_summary.json", summary)
    with (output_dir / "v7_slot_metrics.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for transition, operation_row in zip(
            rollout.transitions,
            np.asarray(rollout.v7_cache_operation_actions, dtype=np.int64),
            strict=True,
        ):
            task_metrics = slot_task_metrics_dict(transition)
            stream.write(
                json.dumps(
                    {
                        "experiment_seed": int(experiment_seed),
                        "episode_seed": int(resolved_episode_seed),
                        "origin": int(transition.origin_index),
                        "reward": float(transition.reward),
                        "q": float(transition.actual_task_utility),
                        "cache_cost": float(transition.cache_cost),
                        "movement_cost": float(transition.movement_cost),
                        "cache_change_count": int(transition.actual_cache_update_count),
                        "uav_movement_count": int(transition.actual_uav_movement_count),
                        **task_metrics,
                        "operation_counts": {
                            name: int(np.sum(operation_row == index))
                            for index, name in enumerate(V7_OPERATION_ORDER)
                        },
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return manifest, summary
