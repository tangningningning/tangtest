"""Frozen evaluator for the no-prediction cache-only PPO ablation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - import-only environments
    torch = None  # type: ignore[assignment]

from common import project_config

from ppo_simple.evaluation_metrics import infer_episode_seed, slot_task_metrics_dict
from ppo_simple.interface import _copy_uav_positions
from .no_prediction import NO_PREDICTION_POLICY_SCHEMA
from ppo_simple.policy import (
    AdaptiveNoPredictionCacheOnlyActorCritic,
)
from ppo_simple.rollout import collect_rollout
from ppo_simple.trainer import PPOV2Trainer, aligned_train_steps, load_environment_context
from baselines.cache_only_evaluator import (
    _ensure_fresh_output,
    _resolve_path,
    _seed_sampling,
    _sha256,
    _write_json,
)
from baselines.replica_aware_cache_only_evaluator import (
    REPLICA_AWARE_CACHE_OPERATION_ORDER,
    _summary as _replica_aware_summary,
)


NO_PREDICTION_CACHE_POLICY_ACTION_MODES = ("deterministic", "stochastic")


def _require_torch() -> None:
    if torch is None:
        raise ImportError("Replica-aware Cache-only no-prediction evaluation requires PyTorch")


def _validate_no_prediction_context(context: Any) -> None:
    configured_control_mode = str(
        project_config.get_config_value(
            context.cfg,
            "algorithm.ppo.configuration_control_mode",
        )
    )
    if context.configuration_control_mode != configured_control_mode:
        raise ValueError(
            "Replica-aware Cache-only no-prediction context configuration_control_mode disagrees with config"
        )
    if configured_control_mode != "policy_primitive_cache_operation":
        raise ValueError(
            "Replica-aware Cache-only no-prediction evaluation requires configuration_control_mode="
            "policy_primitive_cache_operation"
        )
    configured_schema = str(
        project_config.get_config_value(context.cfg, "algorithm.ppo.policy_schema")
    )
    if str(context.policy_schema) != configured_schema:
        raise ValueError("Replica-aware Cache-only no-prediction context policy_schema disagrees with config")
    if str(context.policy_schema) != NO_PREDICTION_POLICY_SCHEMA:
        raise ValueError(
            "Replica-aware Cache-only no-prediction evaluation requires policy_schema="
            f"{NO_PREDICTION_POLICY_SCHEMA}"
        )
    protocol = project_config.get_ppo_data_protocol(context.cfg)
    if protocol["protocol"] != project_config.PPO_DATA_PROTOCOL_FULL_EPISODE_RANDOMIZED:
        raise ValueError(
            "Replica-aware Cache-only no-prediction evaluation requires full_episode_randomized protocol"
        )
    if (
        context.train_origin_start,
        context.train_origin_end,
        context.episode_slots,
        context.terminal_origin,
    ) != (0, 299, 300, 299):
        raise ValueError(
            "Replica-aware Cache-only no-prediction evaluation requires the 0..299 episode protocol"
        )
    if int(context.observation_spec.prediction_horizon) != 5:
        raise ValueError("Replica-aware Cache-only no-prediction evaluation requires prediction horizon H=5")
    if context.observation_spec.horizon_mask_slice is None:
        raise ValueError("Replica-aware Cache-only no-prediction evaluation requires a prediction horizon mask")
    if context.observation_spec.time_to_go_slice is None:
        raise ValueError("Replica-aware Cache-only no-prediction evaluation requires time-to-go")
    if int(context.observation_spec.observation_dim) != 400:
        raise ValueError("Replica-aware Cache-only no-prediction evaluation requires observation_dim=400")


def evaluate_no_prediction_episode(
    checkpoint_path: str | Path,
    run_dir: str | Path,
    config_path: str | Path = "configs/ablations/no_prediction.toml",
    realization_dir: str | Path | None = None,
    device: str = "cpu",
    policy_action_mode: str = "deterministic",
    policy_sampling_seed: int | None = None,
    episode_seed: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate one frozen no-prediction policy on one complete realization."""

    _require_torch()
    mode = str(policy_action_mode)
    if mode not in NO_PREDICTION_CACHE_POLICY_ACTION_MODES:
        raise ValueError(
            "Replica-aware Cache-only no-prediction policy_action_mode must be deterministic or stochastic"
        )
    project_root = project_config.find_project_root()
    resolved_config = _resolve_path(project_root, config_path)
    resolved_checkpoint = _resolve_path(project_root, checkpoint_path)
    output_dir = _resolve_path(project_root, run_dir)
    _ensure_fresh_output(output_dir)
    context = load_environment_context(
        str(resolved_config),
        realization_dir=realization_dir,
    )
    _validate_no_prediction_context(context)
    resolved_episode_seed = None if episode_seed is None else int(episode_seed)
    if resolved_episode_seed is None:
        resolved_episode_seed = infer_episode_seed(realization_dir)
    if resolved_episode_seed is None:
        raise ValueError(
            "Replica-aware Cache-only no-prediction evaluation requires episode_seed or episode_<seed> realization"
        )
    experiment_seed = int(
        project_config.get_config_value(context.cfg, "experiment.seed")
    )
    trainer = PPOV2Trainer(context, device=device)
    config_sha256 = _sha256(resolved_config)
    cursor = trainer.load_checkpoint(resolved_checkpoint, config_sha256)
    if (
        type(trainer.policy)
        is not AdaptiveNoPredictionCacheOnlyActorCritic
    ):
        raise TypeError(
            "checkpoint did not restore the exact Replica-aware Cache-only no-prediction policy type"
        )
    if trainer.policy.architecture_id != NO_PREDICTION_POLICY_SCHEMA:
        raise TypeError("checkpoint restored an unexpected no-prediction policy schema")
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
        raise RuntimeError("Replica-aware Cache-only no-prediction evaluation modified policy parameters")
    if trainer.policy.popart_statistics() != popart_snapshot:
        raise RuntimeError("Replica-aware Cache-only no-prediction evaluation modified PopArt statistics")
    if (
        int(trainer.optimizer_steps),
        int(trainer.actor_optimizer_steps),
        int(trainer.critic_optimizer_steps),
    ) != optimizer_steps_before_eval:
        raise RuntimeError("Replica-aware Cache-only no-prediction evaluation performed an optimizer step")

    summary = _replica_aware_summary(
        context,
        rollout,
        mode,
        experiment_seed=experiment_seed,
        episode_seed=resolved_episode_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_type": "no_prediction_full_episode_evaluation",
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
        "zeroed_observation_slices": [
            "predicted_vehicle_slice",
            "popularity_slice",
        ],
    }
    _write_json(output_dir / "no_prediction_episode_manifest.json", manifest)
    _write_json(output_dir / "no_prediction_episode_summary.json", summary)
    with (output_dir / "no_prediction_slot_metrics.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for transition, operation_row in zip(
            rollout.transitions,
            np.asarray(rollout.cache_operation_actions, dtype=np.int64),
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
                            for index, name in enumerate(REPLICA_AWARE_CACHE_OPERATION_ORDER)
                        },
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return manifest, summary
