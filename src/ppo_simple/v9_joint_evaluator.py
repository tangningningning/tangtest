"""Frozen full-episode evaluator for the Joint V9 Cache/UAV policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - import-only environments
    torch = None  # type: ignore[assignment]

from common import project_config

from .evaluation_metrics import (
    aggregate_episode_task_metrics,
    infer_episode_seed,
    slot_task_metrics_dict,
)
from .interface import _copy_uav_positions
from .policy import AdaptiveJointCacheReplicaAwareUAVOperationV9PopArtActorCritic
from .rollout import collect_rollout
from .state_trace import STATE_TRACE_FILENAME, TRACE_FIELDS, write_state_trace
from .trainer import PPOV2Trainer, aligned_train_steps, load_environment_context
from .v7_evaluator import (
    _ensure_fresh_output,
    _resolve_path,
    _seed_sampling,
    _sha256,
    _write_json,
)


V9_POLICY_SCHEMA = "adaptive_joint_cache_replica_aware_uav_operation_v9_popart"
V9_CONTROL_MODE = "policy_primitive_v9_joint_cache_uav_operation"
V9_CACHE_OPERATION_ORDER = ("NO_CHANGE", "ADD", "DELETE", "SWAP")
V9_UAV_OPERATION_ORDER = ("STAY", "MOVE")
V9_POLICY_ACTION_MODES = ("deterministic", "stochastic")


def _require_torch() -> None:
    if torch is None:
        raise ImportError("V9 evaluation requires PyTorch")


def _validate_v9_context(context: Any) -> None:
    configured_control_mode = str(
        project_config.get_config_value(
            context.cfg,
            "algorithm.ppo.configuration_control_mode",
        )
    )
    if context.configuration_control_mode != configured_control_mode:
        raise ValueError("V9 context configuration_control_mode disagrees with config")
    if configured_control_mode != V9_CONTROL_MODE:
        raise ValueError(
            "V9 evaluation requires configuration_control_mode="
            f"{V9_CONTROL_MODE}"
        )
    configured_schema = str(
        project_config.get_config_value(context.cfg, "algorithm.ppo.policy_schema")
    )
    if str(context.policy_schema) != configured_schema:
        raise ValueError("V9 context policy_schema disagrees with config")
    if str(context.policy_schema) != V9_POLICY_SCHEMA:
        raise ValueError(
            "V9 evaluation requires policy_schema=" f"{V9_POLICY_SCHEMA}"
        )
    if not bool(project_config.get_config_value(context.cfg, "algorithm.uav.enabled")):
        raise ValueError("V9 evaluation requires the configured UAV policy to be enabled")
    protocol = project_config.get_ppo_data_protocol(context.cfg)
    if protocol["protocol"] != project_config.PPO_DATA_PROTOCOL_FULL_EPISODE_RANDOMIZED:
        raise ValueError("V9 evaluation requires full_episode_randomized protocol")
    if (
        context.train_origin_start,
        context.train_origin_end,
        context.episode_slots,
        context.terminal_origin,
    ) != (0, 299, 300, 299):
        raise ValueError("V9 evaluation requires the 0..299 episode protocol")
    if int(context.observation_spec.prediction_horizon) != 5:
        raise ValueError("V9 evaluation requires prediction horizon H=5")
    if context.observation_spec.horizon_mask_slice is None:
        raise ValueError("V9 evaluation requires a prediction horizon mask")
    if int(context.observation_spec.observation_dim) != 400:
        raise ValueError("V9 evaluation requires observation_dim=400")


def _summary(
    context: Any,
    rollout: Any,
    policy_action_mode: str,
    *,
    experiment_seed: int,
    episode_seed: int,
) -> dict[str, Any]:
    required = (
        rollout.v7_cache_operation_actions,
        rollout.v7_cache_operation_masks,
        rollout.uav_operation_actions,
        rollout.uav_operation_masks,
    )
    if any(value is None for value in required):
        raise ValueError("V9 rollout is missing Cache/UAV operation arrays")
    if len(rollout.transitions) != 300:
        raise ValueError("V9 evaluation must contain 300 transitions")
    cache_actions = np.asarray(rollout.v7_cache_operation_actions, dtype=np.int64)
    cache_masks = np.asarray(rollout.v7_cache_operation_masks, dtype=bool)
    uav_actions = np.asarray(rollout.uav_operation_actions, dtype=np.int64)
    uav_masks = np.asarray(rollout.uav_operation_masks, dtype=bool)
    cache_count = len(context.action_spec.cache_node_ids)
    uav_count = len(context.action_spec.uav_ids)
    if cache_actions.shape != (300, cache_count):
        raise ValueError("V9 Cache operation action shape mismatch")
    if cache_masks.shape != (300, cache_count, len(V9_CACHE_OPERATION_ORDER)):
        raise ValueError("V9 Cache operation mask shape mismatch")
    if bool(np.any(cache_actions < 0)) or bool(
        np.any(cache_actions >= len(V9_CACHE_OPERATION_ORDER))
    ):
        raise ValueError("V9 Cache operation action index out of range")
    if not bool(
        np.all(
            cache_masks[
                np.arange(300)[:, None],
                np.arange(cache_count)[None, :],
                cache_actions,
            ]
        )
    ):
        raise ValueError("V9 evaluation found a Cache operation outside its mask")
    if uav_actions.shape != (300, uav_count):
        raise ValueError("V9 UAV operation action shape mismatch")
    if uav_masks.shape != (300, uav_count, len(V9_UAV_OPERATION_ORDER)):
        raise ValueError("V9 UAV operation mask shape mismatch")
    if bool(np.any(uav_actions < 0)) or bool(
        np.any(uav_actions >= len(V9_UAV_OPERATION_ORDER))
    ):
        raise ValueError("V9 UAV operation action index out of range")
    if not bool(
        np.all(
            uav_masks[
                np.arange(300)[:, None],
                np.arange(uav_count)[None, :],
                uav_actions,
            ]
        )
    ):
        raise ValueError("V9 evaluation found a UAV operation outside its mask")
    cache_operation_counts = np.bincount(
        cache_actions.reshape(-1), minlength=len(V9_CACHE_OPERATION_ORDER)
    )
    uav_operation_counts = np.bincount(
        uav_actions.reshape(-1), minlength=len(V9_UAV_OPERATION_ORDER)
    )
    total_cache_heads = 300 * cache_count
    total_uav_heads = 300 * uav_count
    operation_rates = {
        name: float(cache_operation_counts[index]) / float(total_cache_heads)
        for index, name in enumerate(V9_CACHE_OPERATION_ORDER)
    }
    uav_operation_rates = {
        name: float(uav_operation_counts[index]) / float(total_uav_heads)
        for index, name in enumerate(V9_UAV_OPERATION_ORDER)
    }
    total_return = float(np.asarray(rollout.rewards, dtype=np.float64).sum())
    total_q = float(sum(float(item.actual_task_utility) for item in rollout.transitions))
    total_cache_cost = float(sum(float(item.cache_cost) for item in rollout.transitions))
    total_movement_cost = float(
        sum(float(item.movement_cost) for item in rollout.transitions)
    )
    actual_cache_updates = int(
        sum(int(item.actual_cache_update_count) for item in rollout.transitions)
    )
    actual_uav_movements = int(
        sum(int(item.actual_uav_movement_count) for item in rollout.transitions)
    )
    selected_moves = int(np.sum(uav_actions == 1))
    if selected_moves != actual_uav_movements:
        raise ValueError("V9 MOVE selections and actual UAV movements disagree")
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
        "uav_operation_rates": uav_operation_rates,
        "uav_stay_rate": uav_operation_rates["STAY"],
        "uav_move_rate": uav_operation_rates["MOVE"],
        "uav_movement_rate": float(actual_uav_movements) / float(total_uav_heads),
        "uav_movement_count": actual_uav_movements,
        **task_metrics,
        "selected_action_sha256": hashlib.sha256(
            np.ascontiguousarray(rollout.actions, dtype=np.int8).tobytes(order="C")
        ).hexdigest(),
    }


def evaluate_v9_joint_episode(
    checkpoint_path: str | Path,
    run_dir: str | Path,
    config_path: str | Path = "configs/proposed/system_config_ppo_v9_joint_cache_uav.toml",
    realization_dir: str | Path | None = None,
    device: str = "cpu",
    policy_action_mode: str = "deterministic",
    policy_sampling_seed: int | None = None,
    episode_seed: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate one frozen V9 policy on one complete 0..299 realization."""

    _require_torch()
    mode = str(policy_action_mode)
    if mode not in V9_POLICY_ACTION_MODES:
        raise ValueError("V9 policy_action_mode must be deterministic or stochastic")
    project_root = project_config.find_project_root()
    resolved_config = _resolve_path(project_root, config_path)
    resolved_checkpoint = _resolve_path(project_root, checkpoint_path)
    output_dir = _resolve_path(project_root, run_dir)
    _ensure_fresh_output(output_dir)
    context = load_environment_context(
        str(resolved_config),
        realization_dir=realization_dir,
    )
    _validate_v9_context(context)
    resolved_episode_seed = None if episode_seed is None else int(episode_seed)
    if resolved_episode_seed is None:
        resolved_episode_seed = infer_episode_seed(realization_dir)
    if resolved_episode_seed is None:
        raise ValueError(
            "V9 evaluation requires an explicit episode_seed or episode_<seed> realization"
        )
    experiment_seed = int(
        project_config.get_config_value(context.cfg, "experiment.seed")
    )
    trainer = PPOV2Trainer(context, device=device)
    config_sha256 = _sha256(resolved_config)
    cursor = trainer.load_checkpoint(resolved_checkpoint, config_sha256)
    if type(trainer.policy) is not AdaptiveJointCacheReplicaAwareUAVOperationV9PopArtActorCritic:
        raise TypeError("V9 checkpoint did not restore the exact Joint V9 policy")
    if trainer.policy.architecture_id != V9_POLICY_SCHEMA:
        raise TypeError("V9 checkpoint restored an unexpected policy schema")
    trainer.policy.eval()
    steps = aligned_train_steps(context)
    if policy_sampling_seed is not None:
        _seed_sampling(int(policy_sampling_seed))
    parameter_snapshot = [
        parameter.detach().clone() for parameter in trainer.policy.parameters()
    ]
    popart_snapshot = trainer.policy.popart_statistics()
    optimizer_steps_before_eval = (
        int(trainer.optimizer_steps),
        int(trainer.actor_optimizer_steps),
        int(trainer.critic_optimizer_steps),
    )
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
        for before, after in zip(
            parameter_snapshot, trainer.policy.parameters(), strict=True
        )
    ):
        raise RuntimeError("V9 evaluation modified policy parameters")
    if trainer.policy.popart_statistics() != popart_snapshot:
        raise RuntimeError("V9 evaluation modified PopArt statistics")
    if (
        int(trainer.optimizer_steps),
        int(trainer.actor_optimizer_steps),
        int(trainer.critic_optimizer_steps),
    ) != optimizer_steps_before_eval:
        raise RuntimeError("V9 evaluation performed an optimizer step")
    summary = _summary(
        context,
        rollout,
        mode,
        experiment_seed=experiment_seed,
        episode_seed=resolved_episode_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_type": "v9_joint_full_episode_evaluation",
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
        "prediction_horizon": 5,
        "observation_dim": 400,
        "policy_action_mode": mode,
        "policy_sampling_seed": policy_sampling_seed,
        "optimizer_steps": 0,
        "uav_policy_active": True,
        "uav_fixed": False,
        "uav_operation_order": list(V9_UAV_OPERATION_ORDER),
        "state_trace_filename": STATE_TRACE_FILENAME,
        "state_trace_fields": list(TRACE_FIELDS),
    }
    _write_json(output_dir / "v9_joint_episode_manifest.json", manifest)
    _write_json(output_dir / "v9_joint_episode_summary.json", summary)
    write_state_trace(
        output_dir / STATE_TRACE_FILENAME,
        context,
        rollout.transitions,
        initial_cache=np.asarray(context.initial_cache, dtype=np.int8).copy(),
        initial_uav=_copy_uav_positions(context.initial_uav),
        origin_start=0,
        global_training_slot_start=0,
    )
    with (output_dir / "v9_joint_slot_metrics.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for transition, cache_row, uav_row in zip(
            rollout.transitions,
            np.asarray(rollout.v7_cache_operation_actions, dtype=np.int64),
            np.asarray(rollout.uav_operation_actions, dtype=np.int64),
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
                            name: int(np.sum(cache_row == index))
                            for index, name in enumerate(V9_CACHE_OPERATION_ORDER)
                        },
                        "uav_operation_counts": {
                            name: int(np.sum(uav_row == index))
                            for index, name in enumerate(V9_UAV_OPERATION_ORDER)
                        },
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return manifest, summary
