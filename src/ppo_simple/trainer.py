"""训练 Macro Cache SMDP PPO 与 Micro UAV autoregressive PPO。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - 仅静态导入环境
    torch = None  # type: ignore[assignment]

from common import project_config
from scenario.realization import (
    REALIZATION_ARTIFACT_FILENAMES,
    _realization_artifacts_from_directory,
)
from .interface import (
    PROPOSED_POLICY_SCHEMA,
    _assert_realization_context_compatible,
    aligned_train_steps,
    load_environment_context,
    macro_duration_candidates_from_project,
)
from .macro_targets import (
    build_macro_n2_target,
    build_macro_smdp_lambda_return_target,
)
from .policy import DualTimescaleActorCritic
from .primitive_control import PrimitiveTensorSpec
from .reporting import (
    RUN_MANIFEST_FILENAME,
    RUN_SUMMARY_FILENAME,
    SLOT_METRICS_FILENAME,
    append_dual_timescale_record,
    append_dual_timescale_slot_metrics,
    build_macro_transition_trace_arrays,
    checkpoint_path_for_resume,
    checkpoint_path_for_write,
    ensure_fresh_run_dir_is_clean,
    ensure_run_artifact_directories,
    format_training_epoch_log,
    update_training_epoch_return_plot,
    write_dual_timescale_run_manifest,
    write_macro_transition_trace_artifact,
)
from .rollout import (
    EpisodeCollection,
    MacroRolloutBuffer,
    MicroRolloutBuffer,
    collect_dual_timescale_episode,
    compute_gae,
    compute_macro_smdp_gae,
)

CHECKPOINT_METHOD = "ppo_dual_timescale_v1"
CHECKPOINT_FILENAME = "latest_checkpoint.pt"
BEST_TRAIN_RETURN_CHECKPOINT_FILENAME = "best_train_return_checkpoint.pt"
CHECKPOINT_VERSION = 33
PROPOSED_CONFIG_PATH = "configs/proposed/system_config.toml"
MACRO_ACTOR_ADVANTAGE_MODE = "duration_aware_smdp_gae"
MACRO_CRITIC_TARGET_MODES = {
    "formal_gae",
    "one_step_smdp_td",
    "two_step_smdp_td",
    "smdp_lambda_return",
}


def _require_torch() -> None:
    if torch is None:
        raise ImportError("正式双时间尺度 PPO 需要 PyTorch。")


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve(strict=False)


def _pool_children(pool_path: Path) -> list[Path]:
    child_directories = [
        child
        for child in pool_path.iterdir()
        if child.is_dir()
        and any((child / filename).is_file() for filename in REALIZATION_ARTIFACT_FILENAMES.values())
    ]
    child_directories.sort(
        key=lambda child: (
            int(match.group(1)) if (match := re.search(r"(\d+)$", child.name)) else math.inf,
            child.name,
        )
    )
    if child_directories:
        return child_directories
    if any((pool_path / filename).is_file() for filename in REALIZATION_ARTIFACT_FILENAMES.values()):
        return [pool_path]
    return []


def _load_episode_generator(project_root: Path) -> Any:
    generator_path = project_root / "scripts" / "generation" / "generate_episode_realization.py"
    if not generator_path.is_file():
        raise FileNotFoundError(f"episode generator missing: {generator_path}")
    module_name = "_adaptive_cache_mec_dual_episode_generator"
    spec = importlib.util.spec_from_file_location(module_name, generator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load episode generator: {generator_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_training_episodes(
    *,
    project_root: Path,
    config_path: str | Path,
    episode_pool: str | Path | None,
    episode_seed_start: int | None,
    count: int,
    generate_missing: bool,
    generation_device: str,
) -> tuple[Path, ...]:
    if episode_pool is None:
        raise ValueError("双时间尺度训练要求显式指定预生成 --episode-pool。")
    if generate_missing and episode_seed_start is None:
        raise ValueError(
            "--generate-missing-episodes 必须同时指定 --episode-seed-start。"
        )
    pool_path = _resolve_path(project_root, episode_pool)
    if pool_path.exists() and not pool_path.is_dir():
        raise FileNotFoundError(f"episode pool 不是目录：{pool_path}")
    if not pool_path.exists():
        if not generate_missing:
            raise FileNotFoundError(f"episode pool 不存在：{pool_path}")
        pool_path.mkdir(parents=True, exist_ok=False)
    if episode_seed_start is not None:
        seeds = [int(episode_seed_start) + index for index in range(int(count))]
        paths = [pool_path / f"episode_{seed}" for seed in seeds]
        missing = [path for path in paths if not path.is_dir()]
        if missing and not generate_missing:
            raise FileNotFoundError(f"episode realization 缺失：{missing[0]}")
        generator = _load_episode_generator(project_root) if missing else None
        for seed, path in zip(seeds, paths, strict=True):
            if path.exists():
                _realization_artifacts_from_directory(path)
                continue
            path.mkdir(parents=False, exist_ok=False)
            assert generator is not None
            generator.run_episode(
                _resolve_path(project_root, config_path),
                seed,
                path,
                None,
                generation_device,
            )
            _realization_artifacts_from_directory(path)
        return tuple(paths)
    paths = _pool_children(pool_path)
    if not paths:
        raise ValueError(f"episode pool 中没有完整 realization：{pool_path}")
    if int(count) > len(paths):
        raise RuntimeError(
            "episode pool exhausted: "
            f"requested={count}, available={len(paths)}"
        )
    return tuple(paths[: int(count)])


def _normalize_advantages(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    mean = float(array.mean())
    std = float(array.std())
    if std <= 1.0e-8:
        return np.zeros_like(array), mean, std
    return (array - mean) / (std + 1.0e-8), mean, std


def _one_step_smdp_td_targets(
    rewards: Any,
    old_next_values: Any,
    actual_durations: Any,
    dones: Any,
    gamma: float,
) -> np.ndarray:
    """Build detached rollout-frozen N1 targets using actual SMDP durations."""

    reward_values = np.asarray(rewards, dtype=np.float64).reshape(-1)
    next_values = np.asarray(old_next_values, dtype=np.float64).reshape(-1)
    durations = np.asarray(actual_durations, dtype=np.int64).reshape(-1)
    terminal = np.asarray(dones, dtype=bool).reshape(-1)
    if not (
        reward_values.shape == next_values.shape == durations.shape == terminal.shape
    ):
        raise ValueError("N1 Macro target arrays must have matching shapes.")
    if np.any(durations <= 0):
        raise ValueError("N1 Macro target actual durations must be positive.")
    gamma_value = float(gamma)
    if not np.isfinite(gamma_value) or gamma_value < 0.0:
        raise ValueError("N1 Macro target gamma must be finite and non-negative.")
    if not all(np.isfinite(values).all() for values in (reward_values, next_values)):
        raise ValueError("N1 Macro target inputs contain NaN/Inf.")
    discounts = np.power(gamma_value, durations.astype(np.float64))
    targets = reward_values + discounts * next_values * (~terminal)
    if not np.isfinite(targets).all():
        raise ValueError("N1 Macro targets contain NaN/Inf.")
    return targets.astype(np.float32)


def _realized_smdp_return_to_go(
    rewards: Any, actual_durations: Any, dones: Any, gamma: float
) -> np.ndarray:
    """Compute an episode's realized SMDP return-to-go for diagnostics only."""

    reward_values = np.asarray(rewards, dtype=np.float64).reshape(-1)
    durations = np.asarray(actual_durations, dtype=np.int64).reshape(-1)
    terminal = np.asarray(dones, dtype=bool).reshape(-1)
    if not (reward_values.shape == durations.shape == terminal.shape):
        raise ValueError("Realized Macro return arrays must have matching shapes.")
    if np.any(durations <= 0) or not np.isfinite(reward_values).all():
        raise ValueError("Realized Macro return inputs are invalid.")
    gamma_value = float(gamma)
    if not np.isfinite(gamma_value) or gamma_value < 0.0:
        raise ValueError("Realized Macro return gamma must be finite and non-negative.")
    result = np.zeros_like(reward_values)
    running_return = 0.0
    for index in range(len(reward_values) - 1, -1, -1):
        if terminal[index]:
            running_return = 0.0
        running_return = float(
            reward_values[index]
            + gamma_value ** int(durations[index]) * running_return
        )
        result[index] = running_return
    if not np.isfinite(result).all():
        raise ValueError("Realized Macro return-to-go contains NaN/Inf.")
    return result.astype(np.float32)


def _gradient_norm(parameters: Any) -> float:
    norms = [
        torch.linalg.vector_norm(parameter.grad.detach(), ord=2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms), ord=2).detach().cpu())


def _explained_variance(targets: np.ndarray, predictions: np.ndarray) -> float | None:
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    prediction = np.asarray(predictions, dtype=np.float64).reshape(-1)
    variance = float(np.var(target))
    if variance <= 1.0e-12:
        return None
    return float(1.0 - np.var(target - prediction) / variance)


def _pearson_correlation(targets: np.ndarray, predictions: np.ndarray) -> float | None:
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    prediction = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape or len(target) < 2:
        return None
    if float(np.std(target)) <= 1.0e-12 or float(np.std(prediction)) <= 1.0e-12:
        return None
    return float(np.corrcoef(target, prediction)[0, 1])


def _macro_batch(buffer: MacroRolloutBuffer) -> dict[str, np.ndarray]:
    items = buffer.transitions
    if not items:
        raise ValueError("Macro rollout buffer 不能为空。")
    batch_size = len(items)
    max_candidates = max(len(item.cache_candidate_masks) for item in items)
    feature_dim = int(items[0].cache_candidate_features.shape[-1])
    features = np.zeros((batch_size, max_candidates, feature_dim), dtype=np.float32)
    masks = np.zeros((batch_size, max_candidates), dtype=bool)
    for index, item in enumerate(items):
        count = len(item.cache_candidate_masks)
        if item.cache_candidate_features.shape != (count, feature_dim):
            raise ValueError("Macro candidate descriptor shape changed within rollout。")
        features[index, :count] = item.cache_candidate_features
        masks[index, :count] = item.cache_candidate_masks
    return {
        "observations": np.stack([item.observation for item in items]).astype(np.float32),
        "next_observations": np.stack([item.next_observation for item in items]).astype(np.float32),
        "candidate_features": features,
        "candidate_masks": masks,
        "duration_masks": np.stack([item.duration_masks for item in items]).astype(bool),
        "cache_actions": np.asarray([item.cache_action for item in items], dtype=np.int64),
        "duration_actions": np.asarray([item.duration_action for item in items], dtype=np.int64),
        "requested_durations": np.asarray(
            [item.requested_duration for item in items], dtype=np.int64
        ),
        "old_log_probs": np.asarray([item.old_macro_log_prob for item in items], dtype=np.float32),
        "old_values": np.asarray([item.old_value for item in items], dtype=np.float32),
        "old_next_values": np.asarray(
            [item.old_next_value for item in items], dtype=np.float32
        ),
        "rewards": np.asarray([item.frame_reward for item in items], dtype=np.float32),
        "durations": np.asarray([item.actual_duration for item in items], dtype=np.int64),
        "dones": np.asarray([item.done for item in items], dtype=bool),
    }


def _micro_batch(buffer: MicroRolloutBuffer) -> dict[str, np.ndarray]:
    items = buffer.transitions
    if not items:
        raise ValueError("Micro rollout buffer 不能为空。")
    return {
        "observations": np.stack([item.observation for item in items]).astype(np.float32),
        "candidate_features": np.stack([item.uav_candidate_features for item in items]).astype(np.float32),
        "candidate_masks": np.stack([item.uav_candidate_masks for item in items]).astype(bool),
        "actions": np.stack([item.actions for item in items]).astype(np.int64),
        "conditions": np.stack([item.uav0_position_condition for item in items]).astype(np.float32),
        "old_log_probs": np.asarray([item.old_joint_log_prob for item in items], dtype=np.float32),
        "old_values": np.asarray([item.old_value for item in items], dtype=np.float32),
        "rewards": np.asarray([item.reward for item in items], dtype=np.float32),
        "dones": np.asarray([item.done for item in items], dtype=bool),
    }


class PPOTrainer:
    """持有分离 Macro/Micro Actor-Critic、优化器与 episode update。"""

    checkpoint_version = CHECKPOINT_VERSION

    def __init__(
        self,
        context: Any,
        *,
        device: str = "cpu",
        actor_learning_rate_override: float | None = None,
        macro_actor_learning_rate_override: float | None = None,
        critic_learning_rate_override: float | None = None,
        capture_macro_transition_trace: bool = False,
    ) -> None:
        _require_torch()
        self.context = context
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable。")
        self.config = context.cfg["algorithm"]["ppo"]
        self.capture_macro_transition_trace = bool(capture_macro_transition_trace)
        self.last_macro_transition_trace: dict[str, np.ndarray] | None = None
        self.macro_critic_target_mode = str(
            self.config.get("macro_critic_target_mode", "formal_gae")
        )
        if self.macro_critic_target_mode not in MACRO_CRITIC_TARGET_MODES:
            raise ValueError(
                "macro_critic_target_mode must be one of "
                f"{sorted(MACRO_CRITIC_TARGET_MODES)}."
            )
        raw_critic_lambda = self.config.get("macro_critic_lambda_return_lambda")
        self.macro_critic_lambda_return_lambda = (
            None if raw_critic_lambda is None else float(raw_critic_lambda)
        )
        if self.macro_critic_lambda_return_lambda is not None and (
            not math.isfinite(self.macro_critic_lambda_return_lambda)
            or not 0.0 <= self.macro_critic_lambda_return_lambda <= 1.0
        ):
            raise ValueError(
                "macro_critic_lambda_return_lambda must be finite and in [0, 1]."
            )
        if (
            self.macro_critic_target_mode == "smdp_lambda_return"
            and self.macro_critic_lambda_return_lambda is None
        ):
            raise ValueError(
                "smdp_lambda_return requires macro_critic_lambda_return_lambda."
            )
        self.long_value_enabled = bool(self.config.get("long_value_enabled", False))
        self.long_value_target = str(
            self.config.get("long_value_target", "realized_return_to_go")
        )
        if not self.long_value_enabled:
            raise ValueError("architecture v10 requires long_value_enabled = true.")
        if self.long_value_target != "realized_return_to_go":
            raise ValueError(
                "Long-Value Critic target must be realized_return_to_go."
            )
        self.long_value_steps_per_update = int(
            self.config.get("long_value_steps_per_update", 50)
        )
        if self.long_value_steps_per_update != 50:
            raise ValueError("Long-Value Critic update budget is fixed at 50 steps.")
        self.long_value_max_grad_norm = float(
            self.config.get("long_value_max_grad_norm", 0.5)
        )
        if (
            not math.isfinite(self.long_value_max_grad_norm)
            or self.long_value_max_grad_norm <= 0.0
        ):
            raise ValueError("long_value_max_grad_norm must be finite and positive.")
        self.duration_candidates = macro_duration_candidates_from_project(context.cfg)
        tensor_spec = PrimitiveTensorSpec.create(context)
        macro_feature_dim = 2 * int(np.asarray(context.initial_cache).size) + 1
        micro_feature_dim = int(tensor_spec.uav_feature_dim) + 2
        self.hidden_sizes = tuple(int(value) for value in self.config["hidden_sizes"])
        self.policy = DualTimescaleActorCritic(
            int(context.observation_spec.observation_dim),
            macro_feature_dim,
            micro_feature_dim,
            self.duration_candidates,
            int(context.observation_spec.prediction_horizon),
            self.hidden_sizes,
        ).to(self.device)
        actor_lr = float(
            actor_learning_rate_override
            if actor_learning_rate_override is not None
            else self.config.get("actor_learning_rate", self.config["learning_rate"])
        )
        macro_actor_lr = float(
            macro_actor_learning_rate_override
            if macro_actor_learning_rate_override is not None
            else actor_lr
        )
        if not math.isfinite(macro_actor_lr) or macro_actor_lr <= 0.0:
            raise ValueError("macro actor learning rate must be finite and positive.")
        critic_lr = float(
            critic_learning_rate_override
            if critic_learning_rate_override is not None
            else self.config.get("critic_learning_rate", self.config["learning_rate"])
        )
        macro_critic_lr = float(
            critic_learning_rate_override
            if critic_learning_rate_override is not None
            else self.config.get("macro_critic_learning_rate", critic_lr)
        )
        micro_critic_lr = float(
            critic_learning_rate_override
            if critic_learning_rate_override is not None
            else self.config.get("micro_critic_learning_rate", critic_lr)
        )
        long_value_lr = float(self.config.get("long_value_learning_rate", 1.0e-3))
        if not math.isfinite(long_value_lr) or long_value_lr <= 0.0:
            raise ValueError("long_value_learning_rate must be finite and positive.")
        self.macro_optimizer = torch.optim.Adam(
            [
                {"params": self.policy.macro_actor_parameters(), "lr": macro_actor_lr},
                {"params": self.policy.macro_critic_parameters(), "lr": macro_critic_lr},
            ],
            eps=1.0e-5,
        )
        self.micro_optimizer = torch.optim.Adam(
            [
                {"params": self.policy.micro_actor_parameters(), "lr": actor_lr},
                {"params": self.policy.micro_critic_parameters(), "lr": micro_critic_lr},
            ],
            eps=1.0e-5,
        )
        self.long_value_optimizer = torch.optim.Adam(
            self.policy.long_value_critic_parameters(),
            lr=long_value_lr,
            eps=1.0e-5,
        )
        self.learning_rates = {
            "actor": actor_lr,
            "macro_actor": macro_actor_lr,
            "micro_actor": actor_lr,
            "critic": critic_lr,
            "macro_critic": macro_critic_lr,
            "micro_critic": micro_critic_lr,
            "long_value": long_value_lr,
        }
        self.training_counters = {
            "completed_episodes": 0,
            "micro_optimizer_steps": 0,
            "macro_optimizer_steps": 0,
            "micro_actor_optimizer_steps": 0,
            "micro_critic_optimizer_steps": 0,
            "macro_actor_optimizer_steps": 0,
            "macro_critic_optimizer_steps": 0,
            "long_value_optimizer_steps": 0,
            "long_value_update_count": 0,
            "micro_update_count": 0,
            "macro_update_count": 0,
            "environment_executions": 0,
            "matching_calls": 0,
            "kkt_calls": 0,
        }
        self.macro_critic_replay_batches: list[dict[str, Any]] = []

    @property
    def config_contract(self) -> dict[str, Any]:
        return {
            "observation_dim": int(self.policy.observation_dim),
            "macro_cache_feature_dim": int(self.policy.macro_cache_feature_dim),
            "micro_uav_feature_dim": int(self.policy.micro_uav_feature_dim),
            "duration_candidates": list(self.policy.duration_candidates),
            "prediction_horizon": int(self.policy.prediction_horizon),
            "hidden_sizes": list(self.hidden_sizes),
            "policy_schema": str(self.policy.architecture_id),
            "architecture_version": int(self.policy.architecture_version),
            "macro_critic_target_mode": self.macro_critic_target_mode,
            "macro_critic_lambda_return_lambda": self.macro_critic_lambda_return_lambda,
            "long_value_enabled": self.long_value_enabled,
            "long_value_target": self.long_value_target,
            "long_value_learning_rate": self.learning_rates["long_value"],
            "long_value_steps_per_update": self.long_value_steps_per_update,
            "long_value_max_grad_norm": self.long_value_max_grad_norm,
        }

    def update_from_episode(self, collection: EpisodeCollection) -> dict[str, Any]:
        """在完整 rollout 后依次更新 Micro 与 Macro 两套策略。"""

        micro_metrics = self._update_micro(collection.micro)
        macro_metrics = self._update_macro(collection.macro)
        self.training_counters["completed_episodes"] += 1
        self.training_counters["environment_executions"] += int(
            collection.summary["environment_execution_count"]
        )
        self.training_counters["matching_calls"] += int(
            collection.summary["matching_call_count"]
        )
        self.training_counters["kkt_calls"] += int(collection.summary["kkt_call_count"])
        return {**micro_metrics, **macro_metrics}

    def _update_micro(self, buffer: MicroRolloutBuffer) -> dict[str, Any]:
        batch = _micro_batch(buffer)
        count = len(batch["rewards"])
        values = batch["old_values"].astype(np.float64)
        advantages, returns = compute_gae(
            batch["rewards"],
            values,
            batch["dones"],
            0.0,
            float(self.config["gamma"]),
            float(self.config["gae_lambda"]),
        )
        normalized, adv_mean, adv_std = _normalize_advantages(advantages)
        batch["advantages"] = normalized
        batch["returns"] = returns
        minibatches = int(self.config["num_minibatches"])
        if count < minibatches:
            minibatches = count
        if minibatches <= 0:
            raise ValueError("Micro rollout 不能为空。")
        epochs = int(self.config["update_epochs"])
        clip = float(self.config["clip_epsilon"])
        value_coefficient = float(self.config["value_loss_coefficient"])
        entropy_coefficient = float(self.config["entropy_coefficient"])
        target_kl = float(self.config["target_kl"])
        max_grad_norm = float(self.config["max_grad_norm"])
        records: list[dict[str, float]] = []
        stopped = False
        for _epoch in range(epochs):
            indices = np.random.permutation(count)
            for index_group in np.array_split(indices, minibatches):
                if len(index_group) == 0:
                    continue
                tensors = self._tensor_batch(batch, index_group)
                evaluated = self.policy.evaluate_micro_joint(
                    tensors["observations"],
                    tensors["candidate_features"],
                    tensors["candidate_masks"],
                    tensors["actions"],
                    tensors["conditions"],
                )
                log_ratio = evaluated.log_prob - tensors["old_log_probs"]
                ratio = torch.exp(log_ratio)
                advantage = tensors["advantages"]
                actor_loss = -torch.minimum(
                    ratio * advantage,
                    torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage,
                ).mean()
                value_loss = torch.mean((evaluated.value - tensors["returns"]) ** 2)
                entropy = evaluated.entropy.mean()
                loss = actor_loss + value_coefficient * value_loss - entropy_coefficient * entropy
                self.micro_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                micro_actor_parameters = self.policy.micro_actor_parameters()
                micro_critic_parameters = self.policy.micro_critic_parameters()
                micro_actor_before_clip = float(
                    torch.nn.utils.clip_grad_norm_(micro_actor_parameters, max_grad_norm)
                    .detach()
                    .cpu()
                )
                micro_actor_after_clip = _gradient_norm(micro_actor_parameters)
                micro_critic_before_clip = float(
                    torch.nn.utils.clip_grad_norm_(micro_critic_parameters, max_grad_norm)
                    .detach()
                    .cpu()
                )
                micro_critic_after_clip = _gradient_norm(micro_critic_parameters)
                self.micro_optimizer.step()
                self.training_counters["micro_optimizer_steps"] += 1
                self.training_counters["micro_actor_optimizer_steps"] += 1
                self.training_counters["micro_critic_optimizer_steps"] += 1
                with torch.no_grad():
                    approx_kl = torch.mean((ratio - 1.0) - log_ratio)
                    clip_fraction = torch.mean((torch.abs(ratio - 1.0) > clip).float())
                records.append(
                    {
                        "actor_loss": float(actor_loss.detach().cpu()),
                        "value_loss": float(value_loss.detach().cpu()),
                        "entropy": float(entropy.detach().cpu()),
                        "approx_kl": float(approx_kl.detach().cpu()),
                        "clip_fraction": float(clip_fraction.detach().cpu()),
                        "actor_grad_norm_before_clip": micro_actor_before_clip,
                        "actor_grad_norm_after_clip": micro_actor_after_clip,
                        "critic_grad_norm_before_clip": micro_critic_before_clip,
                        "critic_grad_norm_after_clip": micro_critic_after_clip,
                    }
                )
                if target_kl > 0.0 and records[-1]["approx_kl"] > 1.5 * target_kl:
                    stopped = True
                    break
            if stopped:
                break
        self.training_counters["micro_update_count"] += 1
        result = self._aggregate_update_metrics("micro", records)
        with torch.no_grad():
            predictions = (
                self.policy.micro_value(
                    torch.as_tensor(
                        batch["observations"], dtype=torch.float32, device=self.device
                    )
                )
                .detach()
                .cpu()
                .numpy()
            )
        result.update(
            {
                "micro_actor_optimizer_steps_this_update": len(records),
                "micro_critic_optimizer_steps_this_update": len(records),
                "micro_adv_mean": float(adv_mean),
                "micro_adv_std": float(adv_std),
                "micro_normalized_adv_mean": float(np.mean(normalized)),
                "micro_normalized_adv_std": float(np.std(normalized)),
                "micro_value_prediction_mean": float(np.mean(predictions)),
                "micro_value_prediction_std": float(np.std(predictions)),
                "micro_value_target_mean": float(np.mean(returns)),
                "micro_value_target_std": float(np.std(returns)),
                "micro_value_target_prediction_corr": _pearson_correlation(
                    returns, predictions
                ),
                "micro_value_rmse": float(
                    np.sqrt(np.mean((returns - predictions) ** 2))
                ),
                "micro_explained_variance": _explained_variance(
                    returns, predictions
                ),
                "micro_early_stop": bool(stopped),
            }
        )
        return result

    def _update_macro(self, buffer: MacroRolloutBuffer) -> dict[str, Any]:
        batch = _macro_batch(buffer)
        count = len(batch["rewards"])
        next_values = batch["old_next_values"].astype(np.float32, copy=True)
        values = batch["old_values"].astype(np.float64)
        macro_actor_advantage, gae_value_targets = compute_macro_smdp_gae(
            batch["rewards"],
            values,
            next_values,
            batch["durations"],
            batch["dones"],
            float(self.config["gamma"]),
            float(self.config["gae_lambda"]),
        )
        normalized_actor_advantage, adv_mean, adv_std = _normalize_advantages(
            macro_actor_advantage
        )
        if self.macro_critic_target_mode == "one_step_smdp_td":
            macro_critic_target = _one_step_smdp_td_targets(
                batch["rewards"],
                next_values,
                batch["durations"],
                batch["dones"],
                float(self.config["gamma"]),
            )
        elif self.macro_critic_target_mode == "two_step_smdp_td":
            macro_critic_target = build_macro_n2_target(
                batch["rewards"],
                values,
                batch["durations"],
                batch["dones"],
                float(self.config["gamma"]),
            ).astype(np.float32)
        elif self.macro_critic_target_mode == "smdp_lambda_return":
            assert self.macro_critic_lambda_return_lambda is not None
            macro_critic_target = build_macro_smdp_lambda_return_target(
                batch["rewards"],
                next_values,
                batch["durations"],
                batch["dones"],
                float(self.config["gamma"]),
                self.macro_critic_lambda_return_lambda,
            ).astype(np.float32, copy=True)
        else:
            macro_critic_target = np.asarray(gae_value_targets, dtype=np.float32)
        realized_return_to_go = _realized_smdp_return_to_go(
            batch["rewards"],
            batch["durations"],
            batch["dones"],
            float(self.config["gamma"]),
        )
        self.last_macro_transition_trace = None
        if self.capture_macro_transition_trace:
            self.last_macro_transition_trace = {
                "observation": batch["observations"].astype(np.float32, copy=True),
                "next_observation": batch["next_observations"].astype(np.float32, copy=True),
                "macro_reward": batch["rewards"].astype(np.float32, copy=True),
                "requested_duration": batch["requested_durations"].astype(np.int64, copy=True),
                "actual_duration": batch["durations"].astype(np.int64, copy=True),
                "done": batch["dones"].astype(bool, copy=True),
                "old_value": batch["old_values"].astype(np.float32, copy=True),
                "old_next_value": next_values.astype(np.float32, copy=True),
                "critic_training_target": macro_critic_target.astype(np.float32, copy=True),
                "realized_return_to_go": realized_return_to_go.astype(np.float32, copy=True),
            }
            if self.macro_critic_target_mode == "two_step_smdp_td":
                n2_bootstrap_old_value = np.zeros(count, dtype=np.float32)
                n2_bootstrap_frame_index = np.full(count, -1, dtype=np.int64)
                for index in range(count):
                    if batch["dones"][index] or batch["dones"][index + 1]:
                        continue
                    n2_bootstrap_old_value[index] = batch["old_values"][index + 2]
                    n2_bootstrap_frame_index[index] = index + 2
                self.last_macro_transition_trace.update(
                    {
                        "n2_target": macro_critic_target.astype(np.float32, copy=True),
                        "n2_bootstrap_old_value": n2_bootstrap_old_value,
                        "n2_bootstrap_frame_index": n2_bootstrap_frame_index,
                    }
                )
            elif self.macro_critic_target_mode == "smdp_lambda_return":
                assert self.macro_critic_lambda_return_lambda is not None
                self.last_macro_transition_trace.update(
                    {
                        "lambda_return_target": macro_critic_target.astype(
                            np.float32, copy=True
                        ),
                        "macro_critic_lambda_return_lambda": np.full(
                            count,
                            self.macro_critic_lambda_return_lambda,
                            dtype=np.float64,
                        ),
                    }
                )
        batch["macro_actor_advantage"] = normalized_actor_advantage
        batch["macro_critic_targets"] = macro_critic_target.copy()
        critic_training_batch = self._append_macro_critic_replay_batch(
            batch["observations"],
            macro_critic_target,
            episode_index=int(self.training_counters["completed_episodes"]) + 1,
        )
        clip = float(self.config["clip_epsilon"])
        value_coefficient = float(self.config["value_loss_coefficient"])
        entropy_coefficient = float(self.config["entropy_coefficient"])
        target_kl = float(self.config["target_kl"])
        max_grad_norm = float(self.config["max_grad_norm"])
        records: list[dict[str, float]] = []
        stopped = False
        all_indices = np.arange(count, dtype=np.int64)
        for _epoch in range(int(self.config["update_epochs"])):
            tensors = self._tensor_batch(batch, all_indices)
            log_prob, entropy, value, _cache_log_prob, _duration_log_prob = (
                self.policy.evaluate_macro(
                    tensors["observations"],
                    tensors["candidate_features"],
                    tensors["candidate_masks"],
                    tensors["cache_actions"],
                    tensors["duration_masks"],
                    tensors["duration_actions"],
                )
            )
            log_ratio = log_prob - tensors["old_log_probs"]
            ratio = torch.exp(log_ratio)
            advantage = tensors["macro_actor_advantage"]
            actor_loss = -torch.minimum(
                ratio * advantage,
                torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage,
            ).mean()
            value_loss = torch.mean((value - tensors["macro_critic_targets"]) ** 2)
            entropy_mean = entropy.mean()
            loss = actor_loss - entropy_coefficient * entropy_mean
            self.macro_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            macro_actor_parameters = self.policy.macro_actor_parameters()
            macro_critic_parameters = self.policy.macro_critic_parameters()
            if any(parameter.grad is not None for parameter in macro_critic_parameters):
                raise RuntimeError("Macro actor PPO update produced Macro Critic gradients.")
            macro_actor_before_clip = float(
                torch.nn.utils.clip_grad_norm_(macro_actor_parameters, max_grad_norm)
                .detach()
                .cpu()
            )
            macro_actor_after_clip = _gradient_norm(macro_actor_parameters)
            self.macro_optimizer.step()
            self.training_counters["macro_optimizer_steps"] += 1
            self.training_counters["macro_actor_optimizer_steps"] += 1
            with torch.no_grad():
                approx_kl = torch.mean((ratio - 1.0) - log_ratio)
                clip_fraction = torch.mean((torch.abs(ratio - 1.0) > clip).float())
            records.append(
                {
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy_mean.detach().cpu()),
                    "approx_kl": float(approx_kl.detach().cpu()),
                    "clip_fraction": float(clip_fraction.detach().cpu()),
                    "actor_grad_norm_before_clip": macro_actor_before_clip,
                    "actor_grad_norm_after_clip": macro_actor_after_clip,
                    "critic_grad_norm_before_clip": 0.0,
                    "critic_grad_norm_after_clip": 0.0,
                }
            )
            if target_kl > 0.0 and records[-1]["approx_kl"] > 1.5 * target_kl:
                stopped = True
                break
        self.training_counters["macro_update_count"] += 1
        result = self._aggregate_update_metrics("macro", records)
        critic_steps = int(self.config.get("macro_critic_steps_per_update", 50))
        critic_fit = self._update_macro_critic_only(
            critic_training_batch["observations"],
            critic_training_batch["returns"],
            steps=critic_steps,
            value_loss_coefficient=value_coefficient,
            max_grad_norm=max_grad_norm,
            evaluation_observations=batch["observations"],
            evaluation_targets=macro_critic_target,
            evaluation_references=realized_return_to_go,
        )
        long_value_fit = self._update_long_value_critic_only(
            batch["observations"],
            realized_return_to_go,
            steps=self.long_value_steps_per_update,
            max_grad_norm=self.long_value_max_grad_norm,
        )
        with torch.no_grad():
            predictions = (
                self.policy.macro_value(
                    torch.as_tensor(
                        batch["observations"], dtype=torch.float32, device=self.device
                    )
                )
                .detach()
                .cpu()
                .numpy()
            )
        result.update(
            {
                "macro_actor_optimizer_steps_this_update": len(records),
                "macro_critic_optimizer_steps_this_update": critic_fit["optimizer_steps"],
                "long_value_optimizer_steps_this_update": long_value_fit[
                    "optimizer_steps"
                ],
                "macro_actor_advantage_mode": MACRO_ACTOR_ADVANTAGE_MODE,
                "macro_critic_target_mode": self.macro_critic_target_mode,
                "macro_critic_lambda_return_lambda": self.macro_critic_lambda_return_lambda,
                "macro_actor_advantage_mean": float(adv_mean),
                "macro_actor_advantage_std": float(adv_std),
                "macro_cache_advantage_mean": float(adv_mean),
                "macro_cache_advantage_std": float(adv_std),
                "macro_duration_advantage_mean": float(adv_mean),
                "macro_duration_advantage_std": float(adv_std),
                "macro_critic_target_mean": float(np.mean(macro_critic_target)),
                "macro_critic_target_std": float(np.std(macro_critic_target)),
                "macro_critic_target_min": float(np.min(macro_critic_target)),
                "macro_critic_target_max": float(np.max(macro_critic_target)),
                "old_value_mean": float(np.mean(batch["old_values"])),
                "old_value_std": float(np.std(batch["old_values"])),
                "old_next_value_mean": float(np.mean(next_values)),
                "old_next_value_std": float(np.std(next_values)),
                "td_error_mean": float(np.mean(macro_critic_target - batch["old_values"])),
                "td_error_std": float(np.std(macro_critic_target - batch["old_values"])),
                "realized_return_to_go_mean": float(np.mean(realized_return_to_go)),
                "realized_return_to_go_std": float(np.std(realized_return_to_go)),
                "critic_target_detached": critic_fit["target_detached"],
                "critic_target_frozen_during_update": critic_fit["target_frozen"],
                "critic_target_sha256_before": critic_fit["target_sha256_before"],
                "critic_target_sha256_after": critic_fit["target_sha256_after"],
                "macro_adv_mean": float(adv_mean),
                "macro_adv_std": float(adv_std),
                "macro_normalized_adv_mean": float(np.mean(normalized_actor_advantage)),
                "macro_normalized_adv_std": float(np.std(normalized_actor_advantage)),
                "macro_value_prediction_mean": float(np.mean(predictions)),
                "macro_value_prediction_std": float(np.std(predictions)),
                "macro_value_target_mean": float(np.mean(macro_critic_target)),
                "macro_value_target_std": float(np.std(macro_critic_target)),
                "macro_value_target_prediction_corr": _pearson_correlation(
                    macro_critic_target, predictions
                ),
                "macro_value_rmse": float(
                    np.sqrt(np.mean((macro_critic_target - predictions) ** 2))
                ),
                "macro_explained_variance": _explained_variance(
                    macro_critic_target, predictions
                ),
                "pre_update_target_ev": critic_fit["snapshots"]["0"]["explained_variance"],
                "post_update_target_ev": critic_fit["snapshots"][str(critic_steps)]["explained_variance"],
                "pre_update_target_correlation": critic_fit["snapshots"]["0"]["correlation"],
                "post_update_target_correlation": critic_fit["snapshots"][str(critic_steps)]["correlation"],
                "pre_update_target_std_ratio": critic_fit["snapshots"]["0"]["std_ratio"],
                "post_update_target_std_ratio": critic_fit["snapshots"][str(critic_steps)]["std_ratio"],
                "pre_update_return_ev": critic_fit["snapshots"]["0"]["realized_return"]["explained_variance"],
                "post_update_return_ev": critic_fit["snapshots"][str(critic_steps)]["realized_return"]["explained_variance"],
                "pre_update_return_correlation": critic_fit["snapshots"]["0"]["realized_return"]["correlation"],
                "post_update_return_correlation": critic_fit["snapshots"][str(critic_steps)]["realized_return"]["correlation"],
                "pre_update_return_std_ratio": critic_fit["snapshots"]["0"]["realized_return"]["std_ratio"],
                "post_update_return_std_ratio": critic_fit["snapshots"][str(critic_steps)]["realized_return"]["std_ratio"],
                "macro_critic_fit_snapshots": critic_fit["snapshots"],
                "n2_pre_ev": critic_fit["snapshots"]["0"]["explained_variance"],
                "n2_post_ev": critic_fit["snapshots"][str(critic_steps)][
                    "explained_variance"
                ],
                "n2_pre_correlation": critic_fit["snapshots"]["0"][
                    "correlation"
                ],
                "n2_post_correlation": critic_fit["snapshots"][str(critic_steps)][
                    "correlation"
                ],
                "n2_pre_std_ratio": critic_fit["snapshots"]["0"]["std_ratio"],
                "n2_post_std_ratio": critic_fit["snapshots"][str(critic_steps)][
                    "std_ratio"
                ],
                "n2_std_ratio": critic_fit["snapshots"][str(critic_steps)][
                    "std_ratio"
                ],
                "long_value_target": self.long_value_target,
                "long_value_fit_snapshots": long_value_fit["snapshots"],
                "long_value_target_detached": long_value_fit["target_detached"],
                "long_value_target_frozen_during_update": long_value_fit[
                    "target_frozen"
                ],
                "long_value_target_sha256_before": long_value_fit[
                    "target_sha256_before"
                ],
                "long_value_target_sha256_after": long_value_fit[
                    "target_sha256_after"
                ],
                "long_value_pre_ev": long_value_fit["snapshots"]["0"][
                    "explained_variance"
                ],
                "long_value_post_ev": long_value_fit["snapshots"][
                    str(self.long_value_steps_per_update)
                ]["explained_variance"],
                "long_value_pre_correlation": long_value_fit["snapshots"]["0"][
                    "correlation"
                ],
                "long_value_post_correlation": long_value_fit["snapshots"][
                    str(self.long_value_steps_per_update)
                ]["correlation"],
                "long_value_pre_std_ratio": long_value_fit["snapshots"]["0"][
                    "std_ratio"
                ],
                "long_value_post_std_ratio": long_value_fit["snapshots"][
                    str(self.long_value_steps_per_update)
                ]["std_ratio"],
                "long_value_rmse": long_value_fit["snapshots"][
                    str(self.long_value_steps_per_update)
                ]["rmse"],
                "macro_critic_replay_episode_indices": critic_training_batch[
                    "episode_indices"
                ],
                "macro_critic_replay_episode_count": len(
                    critic_training_batch["episode_indices"]
                ),
                "macro_critic_replay_sample_count": len(
                    critic_training_batch["returns"]
                ),
                "macro_critic_replay_fit_snapshots": critic_fit[
                    "training_snapshots"
                ],
                "macro_critic_replay_fit_final": critic_fit["training_final"],
                "macro_critic_grad_norm_before_clip": critic_fit[
                    "grad_norm_before_clip_mean"
                ],
                "macro_critic_grad_norm_after_clip": critic_fit[
                    "grad_norm_after_clip_mean"
                ],
                "macro_value_loss": critic_fit["final"]["rmse"] ** 2,
                "macro_early_stop": bool(stopped),
            }
        )
        return result

    def _append_macro_critic_replay_batch(
        self,
        observations: np.ndarray,
        value_targets: np.ndarray,
        *,
        episode_index: int,
    ) -> dict[str, Any]:
        """Append one frozen Macro batch and return the bounded current+history batch."""

        history_episodes = int(
            self.config.get("macro_critic_replay_history_episodes", 4)
        )
        if history_episodes < 0:
            raise ValueError("macro_critic_replay_history_episodes 不能为负数。")
        observation_values = np.asarray(observations, dtype=np.float32)
        target_values = np.asarray(value_targets, dtype=np.float32).reshape(-1)
        if observation_values.ndim != 2 or len(observation_values) != len(target_values):
            raise ValueError("Macro Critic replay batch observations/targets shape mismatch。")
        if len(target_values) == 0:
            raise ValueError("Macro Critic replay batch 不能为空。")
        self.macro_critic_replay_batches.append(
            {
                "episode_index": int(episode_index),
                "observations": observation_values.copy(),
                "returns": target_values.copy(),
            }
        )
        del self.macro_critic_replay_batches[: -max(1, history_episodes + 1)]
        return {
            "episode_indices": [
                int(item["episode_index"]) for item in self.macro_critic_replay_batches
            ],
            "observations": np.concatenate(
                [item["observations"] for item in self.macro_critic_replay_batches], axis=0
            ),
            "returns": np.concatenate(
                [item["returns"] for item in self.macro_critic_replay_batches], axis=0
            ),
        }

    def _update_macro_critic_only(
        self,
        observations: np.ndarray,
        value_targets: np.ndarray,
        *,
        steps: int,
        value_loss_coefficient: float,
        max_grad_norm: float,
        evaluation_observations: np.ndarray | None = None,
        evaluation_targets: np.ndarray | None = None,
        evaluation_references: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Update only the Macro Critic on one frozen full-episode target batch."""

        if int(steps) <= 0:
            raise ValueError("macro_critic_steps_per_update 必须为正整数。")
        observation_tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        ).detach()
        target_tensor = torch.as_tensor(
            value_targets, dtype=torch.float32, device=self.device
        ).reshape(-1).detach().clone()
        frozen_targets = target_tensor.clone()
        if (evaluation_observations is None) != (evaluation_targets is None):
            raise ValueError("Macro Critic evaluation observations/targets 必须同时提供。")
        if evaluation_observations is None:
            evaluation_observation_tensor = observation_tensor
            evaluation_target_tensor = target_tensor
        else:
            evaluation_observation_tensor = torch.as_tensor(
                evaluation_observations, dtype=torch.float32, device=self.device
            ).detach()
            evaluation_target_tensor = torch.as_tensor(
                evaluation_targets, dtype=torch.float32, device=self.device
            ).reshape(-1).detach().clone()
        frozen_evaluation_targets = evaluation_target_tensor.clone()
        actor_parameters = self.policy.macro_actor_parameters()
        critic_parameters = self.policy.macro_critic_parameters()
        evaluation_reference_tensor = None
        if evaluation_references is not None:
            evaluation_reference_tensor = torch.as_tensor(
                evaluation_references, dtype=torch.float32, device=self.device
            ).reshape(-1).detach().clone()
            if len(evaluation_reference_tensor) != len(evaluation_observation_tensor):
                raise ValueError("Macro Critic realized-return reference shape mismatch.")
        frozen_evaluation_references = (
            None if evaluation_reference_tensor is None else evaluation_reference_tensor.clone()
        )

        def evaluation_snapshot() -> dict[str, Any]:
            metrics: dict[str, Any] = self._macro_critic_fit_metrics(
                evaluation_observation_tensor, evaluation_target_tensor
            )
            if evaluation_reference_tensor is not None:
                metrics["realized_return"] = self._macro_critic_fit_metrics(
                    evaluation_observation_tensor, evaluation_reference_tensor
                )
            return metrics

        snapshots: dict[str, dict[str, Any]] = {"0": evaluation_snapshot()}
        training_snapshots: dict[str, dict[str, float | None]] = {
            "0": self._macro_critic_fit_metrics(observation_tensor, target_tensor)
        }
        snapshot_steps = {10, 25, 50}
        gradient_before: list[float] = []
        gradient_after: list[float] = []
        optimizer_steps = 0
        target_sha256_before = hashlib.sha256(
            target_tensor.detach().cpu().numpy().tobytes()
        ).hexdigest()
        for step in range(1, int(steps) + 1):
            self.macro_optimizer.zero_grad(set_to_none=True)
            predictions = self.policy.macro_value(observation_tensor)
            value_loss = torch.mean((predictions - target_tensor) ** 2)
            (float(value_loss_coefficient) * value_loss).backward()
            if any(parameter.grad is not None for parameter in actor_parameters):
                raise RuntimeError("Macro Critic-only update produced Macro Actor gradients.")
            grad_before = float(
                torch.nn.utils.clip_grad_norm_(critic_parameters, max_grad_norm)
                .detach()
                .cpu()
            )
            grad_after = _gradient_norm(critic_parameters)
            gradient_before.append(grad_before)
            gradient_after.append(grad_after)
            self.macro_optimizer.step()
            optimizer_steps += 1
            self.training_counters["macro_optimizer_steps"] += 1
            self.training_counters["macro_critic_optimizer_steps"] += 1
            if not torch.equal(target_tensor, frozen_targets):
                raise RuntimeError("Macro value targets changed during critic-only updates.")
            if not torch.equal(evaluation_target_tensor, frozen_evaluation_targets):
                raise RuntimeError("Macro Critic evaluation targets changed during updates.")
            if frozen_evaluation_references is not None and not torch.equal(
                evaluation_reference_tensor, frozen_evaluation_references
            ):
                raise RuntimeError("Macro Critic realized-return references changed during updates.")
            if step in snapshot_steps:
                snapshots[str(step)] = evaluation_snapshot()
                training_snapshots[str(step)] = self._macro_critic_fit_metrics(
                    observation_tensor, target_tensor
                )
        if optimizer_steps != int(steps):
            raise RuntimeError("Macro Critic optimizer step count did not match its budget.")
        final = snapshots.get(str(int(steps)))
        if final is None:
            final = evaluation_snapshot()
            snapshots[str(int(steps))] = final
        training_final = training_snapshots.get(str(int(steps)))
        if training_final is None:
            training_final = self._macro_critic_fit_metrics(observation_tensor, target_tensor)
            training_snapshots[str(int(steps))] = training_final
        target_sha256_after = hashlib.sha256(
            target_tensor.detach().cpu().numpy().tobytes()
        ).hexdigest()
        target_frozen = bool(
            torch.equal(target_tensor, frozen_targets)
            and target_sha256_before == target_sha256_after
        )
        target_detached = not bool(target_tensor.requires_grad)
        if not target_detached:
            raise RuntimeError("Macro Critic target must be detached before the update.")
        if not target_frozen:
            raise RuntimeError("Macro Critic frozen target changed during critic-only updates.")
        return {
            "optimizer_steps": optimizer_steps,
            "target_detached": target_detached,
            "target_frozen": target_frozen,
            "target_sha256_before": target_sha256_before,
            "target_sha256_after": target_sha256_after,
            "snapshots": snapshots,
            "final": final,
            "training_snapshots": training_snapshots,
            "training_final": training_final,
            "grad_norm_before_clip_mean": float(np.mean(gradient_before)),
            "grad_norm_after_clip_mean": float(np.mean(gradient_after)),
        }

    def _update_long_value_critic_only(
        self,
        observations: np.ndarray,
        realized_return_to_go: np.ndarray,
        *,
        steps: int,
        max_grad_norm: float,
    ) -> dict[str, Any]:
        """Fit the independent Long Critic to one episode's frozen realized returns."""

        if int(steps) != 50:
            raise ValueError("Long-Value Critic update budget is fixed at 50 steps.")
        observation_tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        ).detach()
        target_tensor = torch.as_tensor(
            realized_return_to_go, dtype=torch.float32, device=self.device
        ).reshape(-1).detach().clone()
        if (
            observation_tensor.ndim != 2
            or int(observation_tensor.shape[-1]) != int(self.policy.observation_dim)
            or int(observation_tensor.shape[0]) != int(target_tensor.shape[0])
            or int(target_tensor.numel()) == 0
        ):
            raise ValueError("Long-Value Critic episode observation/target shape mismatch.")
        if not bool(torch.all(torch.isfinite(observation_tensor)).item()) or not bool(
            torch.all(torch.isfinite(target_tensor)).item()
        ):
            raise FloatingPointError("Long-Value Critic batch contains NaN/Inf.")

        frozen_targets = target_tensor.clone()
        target_sha256_before = hashlib.sha256(
            target_tensor.detach().cpu().numpy().tobytes()
        ).hexdigest()
        long_parameters = self.policy.long_value_critic_parameters()
        long_ids = {id(parameter) for parameter in long_parameters}
        macro_optimizer_ids = {
            id(parameter)
            for group in self.macro_optimizer.param_groups
            for parameter in group["params"]
        }
        micro_optimizer_ids = {
            id(parameter)
            for group in self.micro_optimizer.param_groups
            for parameter in group["params"]
        }
        long_optimizer_ids = {
            id(parameter)
            for group in self.long_value_optimizer.param_groups
            for parameter in group["params"]
        }
        if (
            long_optimizer_ids != long_ids
            or long_ids.intersection(macro_optimizer_ids)
            or long_ids.intersection(micro_optimizer_ids)
        ):
            raise RuntimeError(
                "Long-Value Critic optimizer parameters overlap with Actor/N2/Micro."
            )
        protected_parameters = tuple(
            self.policy.macro_actor_parameters()
            + self.policy.macro_critic_parameters()
            + self.policy.micro_actor_parameters()
            + self.policy.micro_critic_parameters()
        )
        protected_before = tuple(parameter.detach().clone() for parameter in protected_parameters)
        self.macro_optimizer.zero_grad(set_to_none=True)
        self.micro_optimizer.zero_grad(set_to_none=True)
        self.long_value_optimizer.zero_grad(set_to_none=True)

        def snapshot() -> dict[str, float | None]:
            return self._long_value_fit_metrics(observation_tensor, target_tensor)

        snapshots: dict[str, dict[str, float | None]] = {"0": snapshot()}
        gradient_before: list[float] = []
        gradient_after: list[float] = []
        for step in range(1, int(steps) + 1):
            self.long_value_optimizer.zero_grad(set_to_none=True)
            prediction = self.policy.long_value(observation_tensor)
            loss = torch.mean((prediction - target_tensor.detach()) ** 2)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Long-Value Critic loss is non-finite.")
            loss.backward()
            if any(parameter.grad is not None for parameter in protected_parameters):
                raise RuntimeError(
                    "Long-Value Critic update produced Actor/N2/Micro gradients."
                )
            grad_before = float(
                torch.nn.utils.clip_grad_norm_(long_parameters, max_grad_norm)
                .detach()
                .cpu()
            )
            grad_after = _gradient_norm(long_parameters)
            if not math.isfinite(grad_before) or not math.isfinite(grad_after):
                raise FloatingPointError("Long-Value Critic gradient is non-finite.")
            gradient_before.append(grad_before)
            gradient_after.append(grad_after)
            self.long_value_optimizer.step()
            self.training_counters["long_value_optimizer_steps"] += 1
            if not torch.equal(target_tensor, frozen_targets):
                raise RuntimeError(
                    "Long-Value Critic realized-return target changed during updates."
                )
            if step in {10, 25, 50}:
                snapshots[str(step)] = snapshot()

        if any(
            not torch.equal(before, parameter.detach())
            for before, parameter in zip(
                protected_before, protected_parameters, strict=True
            )
        ):
            raise RuntimeError("Long-Value Critic update changed Actor/N2/Micro parameters.")
        target_sha256_after = hashlib.sha256(
            target_tensor.detach().cpu().numpy().tobytes()
        ).hexdigest()
        target_detached = not bool(target_tensor.requires_grad)
        target_frozen = bool(
            torch.equal(target_tensor, frozen_targets)
            and target_sha256_before == target_sha256_after
        )
        if not target_detached or not target_frozen:
            raise RuntimeError("Long-Value Critic target must remain detached and frozen.")
        self.training_counters["long_value_update_count"] += 1
        return {
            "optimizer_steps": int(steps),
            "target_detached": target_detached,
            "target_frozen": target_frozen,
            "target_sha256_before": target_sha256_before,
            "target_sha256_after": target_sha256_after,
            "snapshots": snapshots,
            "grad_norm_before_clip_mean": float(np.mean(gradient_before)),
            "grad_norm_after_clip_mean": float(np.mean(gradient_after)),
        }

    def _macro_critic_fit_metrics(
        self, observations: Any, value_targets: Any
    ) -> dict[str, float | None]:
        with torch.no_grad():
            predictions = (
                self.policy.macro_value(observations)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        targets = value_targets.detach().cpu().numpy().astype(np.float64)
        target_std = float(np.std(targets))
        prediction_std = float(np.std(predictions))
        rmse = float(np.sqrt(np.mean((targets - predictions) ** 2)))
        return {
            "explained_variance": _explained_variance(targets, predictions),
            "rmse": rmse,
            "mse": float(rmse**2),
            "prediction_std": prediction_std,
            "target_std": target_std,
            "std_ratio": prediction_std / target_std if target_std > 1.0e-12 else None,
            "correlation": _pearson_correlation(targets, predictions),
        }

    def _long_value_fit_metrics(
        self, observations: Any, value_targets: Any
    ) -> dict[str, float | None]:
        with torch.no_grad():
            predictions = (
                self.policy.long_value(observations)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        targets = value_targets.detach().cpu().numpy().astype(np.float64)
        target_std = float(np.std(targets))
        prediction_std = float(np.std(predictions))
        rmse = float(np.sqrt(np.mean((targets - predictions) ** 2)))
        return {
            "explained_variance": _explained_variance(targets, predictions),
            "rmse": rmse,
            "mse": float(rmse**2),
            "prediction_std": prediction_std,
            "target_std": target_std,
            "std_ratio": prediction_std / target_std if target_std > 1.0e-12 else None,
            "correlation": _pearson_correlation(targets, predictions),
        }

    @staticmethod
    def _aggregate_update_metrics(
        prefix: str, records: list[dict[str, float]]
    ) -> dict[str, float]:
        if not records:
            raise ValueError(f"{prefix} PPO 没有完成任何 optimizer step。")
        return {
            f"{prefix}_{name}": float(np.mean([record[name] for record in records]))
            for name in (
                "actor_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "actor_grad_norm_before_clip",
                "actor_grad_norm_after_clip",
                "critic_grad_norm_before_clip",
                "critic_grad_norm_after_clip",
            )
        }

    def _tensor_batch(
        self, batch: dict[str, np.ndarray], indices: np.ndarray
    ) -> dict[str, Any]:
        return {
            key: torch.as_tensor(value[indices], device=self.device)
            for key, value in batch.items()
            if key
            not in {
                "durations",
                "rewards",
                "old_values",
                "old_next_values",
                "next_observations",
                "requested_durations",
                "dones",
            }
        }

    def checkpoint_payload(
        self,
        *,
        config_raw_sha256: str,
        config_semantic_sha256: str,
        episode_pool: str,
        seed: int,
        best_episode_return: float | None = None,
    ) -> dict[str, Any]:
        """构造带独立策略、优化器和结构合同的 checkpoint。"""

        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "macro_critic_target_mode": self.macro_critic_target_mode,
            "macro_critic_lambda_return_lambda": self.macro_critic_lambda_return_lambda,
            "long_value_enabled": self.long_value_enabled,
            "long_value_target": self.long_value_target,
            "method": CHECKPOINT_METHOD,
            "policy_schema": self.policy.architecture_id,
            "architecture_version": int(self.policy.architecture_version),
            "config_contract": self.config_contract,
            "config_raw_sha256": str(config_raw_sha256),
            "config_semantic_sha256": str(config_semantic_sha256),
            "episode_pool": str(episode_pool),
            "data_protocol": self.context.data_protocol,
            "episode_slots": int(self.context.episode_slots),
            "origin_start": int(self.context.train_origin_start),
            "origin_end": int(self.context.train_origin_end),
            "terminal_origin": int(self.context.terminal_origin),
            "micro_state_dict": self.policy.micro_state_dict(),
            "macro_state_dict": self.policy.macro_state_dict(),
            "n2_critic_state_dict": self.policy.n2_critic_state_dict(),
            "long_value_critic_state_dict": self.policy.long_value_critic_state_dict(),
            "micro_optimizer_state_dict": self.micro_optimizer.state_dict(),
            "macro_optimizer_state_dict": self.macro_optimizer.state_dict(),
            "long_value_optimizer_state_dict": self.long_value_optimizer.state_dict(),
            "value_normalization_states": {"micro": None, "macro": None},
            "training_counters": dict(self.training_counters),
            "macro_critic_replay_batches": [
                {
                    "episode_index": int(item["episode_index"]),
                    "observations": np.asarray(item["observations"], dtype=np.float32).copy(),
                    "returns": np.asarray(item["returns"], dtype=np.float32).copy(),
                }
                for item in self.macro_critic_replay_batches
            ],
            "seed": int(seed),
            "best_episode_return": best_episode_return,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_states": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        }

    def save_checkpoint(
        self,
        path: Path,
        *,
        config_raw_sha256: str,
        config_semantic_sha256: str,
        episode_pool: str,
        seed: int,
        best_episode_return: float | None = None,
    ) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.checkpoint_payload(
                config_raw_sha256=config_raw_sha256,
                config_semantic_sha256=config_semantic_sha256,
                episode_pool=episode_pool,
                seed=seed,
                best_episode_return=best_episode_return,
            ),
            destination,
        )
        return destination

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        config_raw_sha256: str | None = None,
        config_semantic_sha256: str | None = None,
        episode_pool: str | None = None,
        restore_rng: bool = False,
    ) -> dict[str, Any]:
        """严格恢复当前结构；旧架构 checkpoint 不做迁移。"""

        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload 必须是 mapping。")
        if "macro_critic_target_mode" not in payload:
            raise ValueError(
                "checkpoint is missing macro_critic_target_mode; legacy Macro Critic "
                "target semantics cannot be interpreted as the current mode."
            )
        if payload["macro_critic_target_mode"] != self.macro_critic_target_mode:
            raise ValueError(
                "checkpoint macro_critic_target_mode mismatch: "
                f"{payload['macro_critic_target_mode']!r} != "
                f"{self.macro_critic_target_mode!r}"
            )
        if payload.get("macro_critic_lambda_return_lambda") != (
            self.macro_critic_lambda_return_lambda
        ):
            raise ValueError(
                "checkpoint macro_critic_lambda_return_lambda mismatch: "
                f"{payload.get('macro_critic_lambda_return_lambda')!r} != "
                f"{self.macro_critic_lambda_return_lambda!r}"
            )
        if payload.get("long_value_enabled") is not self.long_value_enabled:
            raise ValueError("checkpoint long_value_enabled mismatch.")
        if payload.get("long_value_target") != self.long_value_target:
            raise ValueError(
                "checkpoint long_value_target mismatch: "
                f"{payload.get('long_value_target')!r} != {self.long_value_target!r}"
            )
        required = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "method": CHECKPOINT_METHOD,
            "policy_schema": self.policy.architecture_id,
            "architecture_version": int(self.policy.architecture_version),
        }
        for field, expected in required.items():
            if payload.get(field) != expected:
                raise ValueError(
                    f"checkpoint {field} mismatch: {payload.get(field)!r} != {expected!r}"
                )
        checks = (
            ("config_raw_sha256", config_raw_sha256),
            ("config_semantic_sha256", config_semantic_sha256),
            ("episode_pool", episode_pool),
        )
        for field, expected in checks:
            if expected is not None and payload.get(field) != expected:
                raise ValueError(f"checkpoint {field} mismatch。")
        if payload.get("config_contract") != self.config_contract:
            raise ValueError("checkpoint policy tensor/configuration contract mismatch。")
        if payload.get("data_protocol") != self.context.data_protocol:
            raise ValueError("checkpoint data protocol mismatch。")
        for field, expected in (
            ("episode_slots", self.context.episode_slots),
            ("origin_start", self.context.train_origin_start),
            ("origin_end", self.context.train_origin_end),
            ("terminal_origin", self.context.terminal_origin),
        ):
            if int(payload.get(field, -1)) != int(expected):
                raise ValueError(f"checkpoint {field} mismatch。")
        macro_state = payload["macro_state_dict"]
        explicit_n2_state = payload.get("n2_critic_state_dict")
        if not isinstance(explicit_n2_state, dict):
            raise ValueError("checkpoint is missing the explicit N2 Critic state.")
        embedded_n2_state = {
            key.removeprefix("macro_critic."): value
            for key, value in macro_state.items()
            if key.startswith("macro_critic.")
        }
        if set(embedded_n2_state) != set(explicit_n2_state) or any(
            not torch.equal(embedded_n2_state[key], explicit_n2_state[key])
            for key in embedded_n2_state
        ):
            raise ValueError("checkpoint N2 Critic state copies disagree.")
        self.policy.load_separated_state_dicts(
            macro_state,
            payload["micro_state_dict"],
            payload["long_value_critic_state_dict"],
        )
        self.macro_optimizer.load_state_dict(payload["macro_optimizer_state_dict"])
        self.micro_optimizer.load_state_dict(payload["micro_optimizer_state_dict"])
        self.long_value_optimizer.load_state_dict(
            payload["long_value_optimizer_state_dict"]
        )
        counters = payload.get("training_counters")
        legacy_counter_keys = set(self.training_counters) - {
            "micro_actor_optimizer_steps",
            "micro_critic_optimizer_steps",
            "macro_actor_optimizer_steps",
            "macro_critic_optimizer_steps",
        }
        counter_keys = set(counters) if isinstance(counters, dict) else set()
        if counter_keys not in (legacy_counter_keys, set(self.training_counters)):
            raise ValueError("checkpoint training counter schema mismatch。")
        restored_counters = dict(self.training_counters)
        restored_counters.update({key: int(value) for key, value in counters.items()})
        self.training_counters = restored_counters
        replay_batches = payload.get("macro_critic_replay_batches", [])
        if not isinstance(replay_batches, list):
            raise ValueError("checkpoint Macro Critic replay batches must be a list.")
        history_episodes = int(
            self.config.get("macro_critic_replay_history_episodes", 4)
        )
        max_replay_batches = history_episodes + 1
        if len(replay_batches) > max_replay_batches:
            raise ValueError("checkpoint Macro Critic replay buffer exceeds configured window.")
        restored_replay_batches: list[dict[str, Any]] = []
        for item in replay_batches:
            if not isinstance(item, dict) or set(item) != {
                "episode_index", "observations", "returns"
            }:
                raise ValueError("checkpoint Macro Critic replay batch schema mismatch.")
            observations = np.asarray(item["observations"], dtype=np.float32)
            returns = np.asarray(item["returns"], dtype=np.float32).reshape(-1)
            if observations.ndim != 2 or observations.shape[1] != self.policy.observation_dim:
                raise ValueError("checkpoint Macro Critic replay observation shape mismatch.")
            if len(observations) != len(returns) or len(returns) == 0:
                raise ValueError("checkpoint Macro Critic replay target shape mismatch.")
            if not np.isfinite(observations).all() or not np.isfinite(returns).all():
                raise ValueError("checkpoint Macro Critic replay data contains NaN/Inf.")
            restored_replay_batches.append(
                {
                    "episode_index": int(item["episode_index"]),
                    "observations": observations.copy(),
                    "returns": returns.copy(),
                }
            )
        self.macro_critic_replay_batches = restored_replay_batches
        if restore_rng:
            if "python_rng_state" not in payload or "numpy_rng_state" not in payload:
                raise ValueError("checkpoint 缺少可恢复的 RNG 状态。")
            random.setstate(payload["python_rng_state"])
            np.random.set_state(payload["numpy_rng_state"])
            torch.set_rng_state(payload["torch_rng_state"].cpu())
            cuda_states = payload.get("torch_cuda_rng_states")
            if cuda_states is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_states)
        return payload


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def run_training(
    config_path: str | Path,
    target_episodes: int,
    run_dir: str | Path,
    *,
    episode_pool: str | Path,
    episode_seed_start: int | None = None,
    generate_missing_episodes: bool = False,
    episode_generation_device: str = "cpu",
    actor_learning_rate_override: float | None = None,
    macro_actor_learning_rate_override: float | None = None,
    critic_learning_rate_override: float | None = None,
    seed: int = 0,
    resume: bool = False,
    verbose: bool = False,
    save_macro_transition_trace: bool = False,
) -> dict[str, Any]:
    """按完整 episode 收集 on-policy 数据，然后分别更新两套策略。"""

    _require_torch()
    if int(target_episodes) <= 0:
        raise ValueError("--epochs 必须为正整数。")
    if save_macro_transition_trace and (resume or episode_seed_start is None):
        raise ValueError("transition-trace reporting requires a fresh run with --episode-seed-start.")
    config_file = Path(config_path).expanduser().resolve(strict=True)
    cfg = project_config.load_config(str(config_file))
    project_config.validate_for_formal_experiment(cfg)
    project_root = Path(cfg["_meta"]["project_root"])
    episode_paths = _resolve_training_episodes(
        project_root=project_root,
        config_path=config_file,
        episode_pool=episode_pool,
        episode_seed_start=episode_seed_start,
        count=int(target_episodes),
        generate_missing=bool(generate_missing_episodes),
        generation_device=episode_generation_device,
    )
    pool_path = _resolve_path(project_root, episode_pool)
    context = load_environment_context(str(config_file), realization_dir=episode_paths[0])
    raw_hash = project_config.raw_config_sha256(config_file)
    semantic_hash = project_config.semantic_config_hash(context.cfg)
    run_path = _resolve_path(project_root, run_dir)
    manifest_path = run_path / RUN_MANIFEST_FILENAME
    checkpoint_path = checkpoint_path_for_resume(run_path, CHECKPOINT_FILENAME)
    _seed_everything(seed)
    trainer = PPOTrainer(
        context,
        actor_learning_rate_override=actor_learning_rate_override,
        macro_actor_learning_rate_override=macro_actor_learning_rate_override,
        critic_learning_rate_override=critic_learning_rate_override,
        capture_macro_transition_trace=save_macro_transition_trace,
    )
    completed = 0
    best_return: float | None = None
    if resume:
        if not checkpoint_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("resume 需要现有 run manifest 与 latest checkpoint。")
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        for field, expected in (
            ("config_raw_sha256", raw_hash),
            ("config_semantic_sha256", semantic_hash),
            ("episode_pool", str(pool_path)),
            ("policy_schema", trainer.policy.architecture_id),
        ):
            if manifest.get(field) != expected:
                raise ValueError(f"run manifest {field} mismatch。")
        payload = trainer.load_checkpoint(
            checkpoint_path,
            config_raw_sha256=raw_hash,
            config_semantic_sha256=semantic_hash,
            episode_pool=str(pool_path),
            restore_rng=True,
        )
        completed = int(trainer.training_counters["completed_episodes"])
        best_return = payload.get("best_episode_return")
        if completed > int(target_episodes):
            raise ValueError("checkpoint completed episodes exceed --epochs。")
    else:
        ensure_fresh_run_dir_is_clean(run_path)
        run_path.mkdir(parents=True, exist_ok=True)
    ensure_run_artifact_directories(run_path)
    write_dual_timescale_run_manifest(
        manifest_path,
        config_path=config_file,
        config_raw_sha256=raw_hash,
        config_semantic_sha256=semantic_hash,
        seed=seed,
        trainer=trainer,
        context=context,
        target_episodes=target_episodes,
        episode_pool=str(pool_path),
        episode_seed_start=episode_seed_start,
        resume=resume,
    )

    episode_metrics_path = run_path / "episode_metrics.jsonl"
    update_metrics_path = run_path / "update_metrics.jsonl"
    slot_metrics_path = run_path / SLOT_METRICS_FILENAME
    trace_episode_batches: list[dict[str, Any]] = []
    trace_summary = None
    for episode_index in range(completed, int(target_episodes)):
        epoch_started_at = time.perf_counter()
        realization_path = episode_paths[episode_index]
        next_context = load_environment_context(
            str(config_file), realization_dir=realization_path
        )
        _assert_realization_context_compatible(context, next_context)
        trainer.context = context = next_context
        collection = collect_dual_timescale_episode(
            trainer.policy,
            context,
            aligned_train_steps(context),
            trainer.duration_candidates,
            deterministic=False,
        )
        update_metrics = trainer.update_from_episode(collection)
        training_epoch_runtime = time.perf_counter() - epoch_started_at
        completed = episode_index + 1
        if save_macro_transition_trace:
            trace = trainer.last_macro_transition_trace
            if trace is None:
                raise RuntimeError("Macro transition trace capture was enabled but no trace was produced.")
            trace_count = int(len(trace["macro_reward"]))
            trace_episode = {
                "episode_id": completed,
                "episode_seed": int(episode_seed_start) + episode_index,
                "macro_frame_index": np.arange(trace_count, dtype=np.int64),
                **trace,
            }
            expected_count = int(collection.summary["macro_frame_count"])
            if trace_count != expected_count:
                raise RuntimeError(
                    f"Macro trace E{completed} has {trace_count} transitions; "
                    f"episode summary reports {expected_count}."
                )
            _, episode_trace_check = build_macro_transition_trace_arrays(
                [trace_episode],
                gamma=float(cfg["algorithm"]["ppo"]["gamma"]),
                target_mode=trainer.macro_critic_target_mode,
                lambda_return_lambda=trainer.macro_critic_lambda_return_lambda,
                expected_macro_frame_counts={completed: expected_count},
            )
            if episode_trace_check["transition_count"] <= 0:
                raise RuntimeError(f"Macro trace E{completed} contains no transitions.")
            trace_episode_batches.append(trace_episode)
        episode_return = float(collection.summary["episode_return"])
        if best_return is None or episode_return > float(best_return):
            best_return = episode_return
            trainer.save_checkpoint(
                checkpoint_path_for_write(run_path, BEST_TRAIN_RETURN_CHECKPOINT_FILENAME),
                config_raw_sha256=raw_hash,
                config_semantic_sha256=semantic_hash,
                episode_pool=str(pool_path),
                seed=seed,
                best_episode_return=best_return,
            )
        episode_record = {
            "record_type": "episode",
            "episode": completed,
            "episode_seed": (
                int(episode_seed_start) + episode_index
                if episode_seed_start is not None
                else realization_path.name
            ),
            "training_epoch_runtime_seconds": float(training_epoch_runtime),
            **collection.summary,
        }
        episode_record.update(
            {
                key: value
                for key, value in update_metrics.items()
                if key.startswith(
                    (
                        "macro_actor_advantage",
                        "macro_cache_advantage",
                        "macro_duration_advantage",
                        "macro_critic_target",
                        "n2_",
                        "long_value_",
                        "old_value_",
                        "old_next_value_",
                        "td_error_",
                        "pre_update_target_",
                        "post_update_target_",
                        "pre_update_return_",
                        "post_update_return_",
                        "critic_target_",
                    )
                )
                or key in {
                    "macro_actor_advantage_mode",
                    "macro_critic_target_mode",
                    "macro_critic_lambda_return_lambda",
                    "macro_critic_fit_snapshots",
                    "long_value_fit_snapshots",
                    "long_value_target",
                    "long_value_target_detached",
                    "long_value_target_frozen_during_update",
                    "long_value_target_sha256_before",
                    "long_value_target_sha256_after",
                    "macro_approx_kl",
                    "macro_clip_fraction",
                    "macro_entropy",
                    "micro_explained_variance",
                    "micro_approx_kl",
                    "micro_clip_fraction",
                    "micro_entropy",
                    "micro_value_rmse",
                    "micro_early_stop",
                }
            }
        )
        optimizer_steps_this_episode = {
            key: int(update_metrics[key])
            for key in (
                "macro_actor_optimizer_steps_this_update",
                "macro_critic_optimizer_steps_this_update",
                "micro_actor_optimizer_steps_this_update",
                "micro_critic_optimizer_steps_this_update",
            )
        }
        episode_record.update(optimizer_steps_this_episode)
        update_record = {
            "record_type": "update",
            "episode": completed,
            **update_metrics,
            **optimizer_steps_this_episode,
            **trainer.training_counters,
        }
        append_dual_timescale_record(episode_metrics_path, episode_record)
        append_dual_timescale_record(update_metrics_path, update_record)
        print(
            format_training_epoch_log(
                episode_record,
                target_epochs=int(target_episodes),
                macro_actor_learning_rate=trainer.learning_rates["macro_actor"],
            ),
            flush=True,
        )
        append_dual_timescale_slot_metrics(
            slot_metrics_path, episode=completed, transitions=collection.micro.transitions
        )
        trainer.save_checkpoint(
            checkpoint_path,
            config_raw_sha256=raw_hash,
            config_semantic_sha256=semantic_hash,
            episode_pool=str(pool_path),
            seed=seed,
            best_episode_return=best_return,
        )
        write_dual_timescale_run_manifest(
            manifest_path,
            config_path=config_file,
            config_raw_sha256=raw_hash,
            config_semantic_sha256=semantic_hash,
            seed=seed,
            trainer=trainer,
            context=context,
            target_episodes=target_episodes,
            episode_pool=str(pool_path),
            episode_seed_start=episode_seed_start,
            resume=resume,
        )
        summary = {
            "experiment_type": "macro_cache_smdp_micro_uav_ppo",
            "completed_episodes": completed,
            "target_episodes": int(target_episodes),
            "best_episode_return": best_return,
            "training_counters": dict(trainer.training_counters),
            "last_episode": episode_record,
            "last_update": update_record,
        }
        summary_path = run_path / RUN_SUMMARY_FILENAME
        with summary_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(summary, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
        update_training_epoch_return_plot(run_path)
        if save_macro_transition_trace:
            # Persist each validated cumulative trace before starting the next
            # episode so a later report/finalization failure cannot lose raw rows.
            trace_summary = write_macro_transition_trace_artifact(
                run_path,
                trace_episode_batches,
                gamma=float(cfg["algorithm"]["ppo"]["gamma"]),
                target_mode=trainer.macro_critic_target_mode,
                lambda_return_lambda=trainer.macro_critic_lambda_return_lambda,
            )
        if verbose:
            print(json.dumps(update_record, ensure_ascii=False, sort_keys=True))
    return {
        "run_dir": str(run_path),
        "completed_episodes": completed,
        "target_episodes": int(target_episodes),
        "checkpoint": str(checkpoint_path),
        "best_episode_return": best_return,
        "macro_transition_trace": trace_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练双时间尺度 Cache/UAV PPO。")
    parser.add_argument("--config", default=PROPOSED_CONFIG_PATH)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--episode-pool", required=True)
    parser.add_argument("--episode-seed-start", type=int, default=None)
    parser.add_argument("--generate-missing-episodes", action="store_true")
    parser.add_argument("--episode-generation-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--actor-learning-rate", type=float, default=None)
    parser.add_argument("--macro-actor-learning-rate", type=float, default=None)
    parser.add_argument("--critic-learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-macro-transition-trace", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_training(
        args.config,
        args.epochs,
        args.run_dir,
        episode_pool=args.episode_pool,
        episode_seed_start=args.episode_seed_start,
        generate_missing_episodes=bool(args.generate_missing_episodes),
        episode_generation_device=args.episode_generation_device,
        actor_learning_rate_override=args.actor_learning_rate,
        macro_actor_learning_rate_override=args.macro_actor_learning_rate,
        critic_learning_rate_override=args.critic_learning_rate,
        seed=args.seed,
        resume=bool(args.resume),
        verbose=bool(args.verbose),
        save_macro_transition_trace=bool(args.save_macro_transition_trace),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
