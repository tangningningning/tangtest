"""训练 Macro Cache SMDP PPO 与 Micro UAV autoregressive PPO。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import sys
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
from .policy import DualTimescaleActorCritic
from .primitive_control import PrimitiveTensorSpec
from .reporting import (
    RUN_MANIFEST_FILENAME,
    RUN_SUMMARY_FILENAME,
    SLOT_METRICS_FILENAME,
    append_dual_timescale_record,
    append_dual_timescale_slot_metrics,
    checkpoint_path_for_resume,
    checkpoint_path_for_write,
    ensure_fresh_run_dir_is_clean,
    ensure_run_artifact_directories,
    write_dual_timescale_run_manifest,
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
CHECKPOINT_VERSION = 27
PROPOSED_CONFIG_PATH = "configs/proposed/system_config.toml"


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


def _explained_variance(targets: np.ndarray, predictions: np.ndarray) -> float | None:
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    prediction = np.asarray(predictions, dtype=np.float64).reshape(-1)
    variance = float(np.var(target))
    if variance <= 1.0e-12:
        return None
    return float(1.0 - np.var(target - prediction) / variance)


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
        "candidate_features": features,
        "candidate_masks": masks,
        "duration_masks": np.stack([item.duration_masks for item in items]).astype(bool),
        "cache_actions": np.asarray([item.cache_action for item in items], dtype=np.int64),
        "duration_actions": np.asarray([item.duration_action for item in items], dtype=np.int64),
        "old_log_probs": np.asarray([item.old_macro_log_prob for item in items], dtype=np.float32),
        "old_values": np.asarray([item.old_value for item in items], dtype=np.float32),
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
        critic_learning_rate_override: float | None = None,
    ) -> None:
        _require_torch()
        self.context = context
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable。")
        self.config = context.cfg["algorithm"]["ppo"]
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
        critic_lr = float(
            critic_learning_rate_override
            if critic_learning_rate_override is not None
            else self.config.get("critic_learning_rate", self.config["learning_rate"])
        )
        self.macro_optimizer = torch.optim.Adam(
            [
                {"params": self.policy.macro_actor_parameters(), "lr": actor_lr},
                {"params": self.policy.macro_critic_parameters(), "lr": critic_lr},
            ],
            eps=1.0e-5,
        )
        self.micro_optimizer = torch.optim.Adam(
            [
                {"params": self.policy.micro_actor_parameters(), "lr": actor_lr},
                {"params": self.policy.micro_critic_parameters(), "lr": critic_lr},
            ],
            eps=1.0e-5,
        )
        self.learning_rates = {"actor": actor_lr, "critic": critic_lr}
        self.training_counters = {
            "completed_episodes": 0,
            "micro_optimizer_steps": 0,
            "macro_optimizer_steps": 0,
            "micro_update_count": 0,
            "macro_update_count": 0,
            "environment_executions": 0,
            "matching_calls": 0,
            "kkt_calls": 0,
        }

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
                torch.nn.utils.clip_grad_norm_(
                    self.policy.micro_parameters(), max_grad_norm
                )
                self.micro_optimizer.step()
                self.training_counters["micro_optimizer_steps"] += 1
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
                    }
                )
                if target_kl > 0.0 and records[-1]["approx_kl"] > 1.5 * target_kl:
                    stopped = True
                    break
            if stopped:
                break
        self.training_counters["micro_update_count"] += 1
        result = self._aggregate_update_metrics("micro", records)
        result.update(
            {
                "micro_adv_mean": float(adv_mean),
                "micro_adv_std": float(adv_std),
                "micro_value_rmse": float(np.sqrt(np.mean((returns - values) ** 2))),
                "micro_explained_variance": _explained_variance(returns, values),
                "micro_early_stop": bool(stopped),
            }
        )
        return result

    def _update_macro(self, buffer: MacroRolloutBuffer) -> dict[str, Any]:
        batch = _macro_batch(buffer)
        count = len(batch["rewards"])
        next_values = np.zeros(count, dtype=np.float32)
        nonterminal_indices = np.flatnonzero(~batch["dones"])
        if len(nonterminal_indices):
            observations = np.stack(
                [buffer.transitions[int(i)].next_observation for i in nonterminal_indices]
            ).astype(np.float32)
            with torch.no_grad():
                next_values[nonterminal_indices] = (
                    self.policy.macro_value(
                        torch.as_tensor(observations, dtype=torch.float32, device=self.device)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
        values = batch["old_values"].astype(np.float64)
        advantages, returns = compute_macro_smdp_gae(
            batch["rewards"],
            values,
            next_values,
            batch["durations"],
            batch["dones"],
            float(self.config["gamma"]),
            float(self.config["gae_lambda"]),
        )
        normalized, adv_mean, adv_std = _normalize_advantages(advantages)
        batch["advantages"] = normalized
        batch["returns"] = returns
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
            advantage = tensors["advantages"]
            actor_loss = -torch.minimum(
                ratio * advantage,
                torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage,
            ).mean()
            value_loss = torch.mean((value - tensors["returns"]) ** 2)
            entropy_mean = entropy.mean()
            loss = actor_loss + value_coefficient * value_loss - entropy_coefficient * entropy_mean
            self.macro_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.macro_parameters(), max_grad_norm)
            self.macro_optimizer.step()
            self.training_counters["macro_optimizer_steps"] += 1
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
                }
            )
            if target_kl > 0.0 and records[-1]["approx_kl"] > 1.5 * target_kl:
                stopped = True
                break
        self.training_counters["macro_update_count"] += 1
        result = self._aggregate_update_metrics("macro", records)
        result.update(
            {
                "macro_adv_mean": float(adv_mean),
                "macro_adv_std": float(adv_std),
                "macro_value_rmse": float(np.sqrt(np.mean((returns - values) ** 2))),
                "macro_explained_variance": _explained_variance(returns, values),
                "macro_early_stop": bool(stopped),
            }
        )
        return result

    @staticmethod
    def _aggregate_update_metrics(
        prefix: str, records: list[dict[str, float]]
    ) -> dict[str, float]:
        if not records:
            raise ValueError(f"{prefix} PPO 没有完成任何 optimizer step。")
        return {
            f"{prefix}_{name}": float(np.mean([record[name] for record in records]))
            for name in ("actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")
        }

    def _tensor_batch(
        self, batch: dict[str, np.ndarray], indices: np.ndarray
    ) -> dict[str, Any]:
        return {
            key: torch.as_tensor(value[indices], device=self.device)
            for key, value in batch.items()
            if key not in {"durations", "rewards", "old_values", "dones"}
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
            "micro_optimizer_state_dict": self.micro_optimizer.state_dict(),
            "macro_optimizer_state_dict": self.macro_optimizer.state_dict(),
            "value_normalization_states": {"micro": None, "macro": None},
            "training_counters": dict(self.training_counters),
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
        self.policy.load_separated_state_dicts(
            payload["macro_state_dict"], payload["micro_state_dict"]
        )
        self.macro_optimizer.load_state_dict(payload["macro_optimizer_state_dict"])
        self.micro_optimizer.load_state_dict(payload["micro_optimizer_state_dict"])
        counters = payload.get("training_counters")
        if not isinstance(counters, dict) or set(counters) != set(self.training_counters):
            raise ValueError("checkpoint training counter schema mismatch。")
        self.training_counters = {key: int(value) for key, value in counters.items()}
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
    critic_learning_rate_override: float | None = None,
    seed: int = 0,
    resume: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """按完整 episode 收集 on-policy 数据，然后分别更新两套策略。"""

    _require_torch()
    if int(target_episodes) <= 0:
        raise ValueError("--epochs 必须为正整数。")
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
        critic_learning_rate_override=critic_learning_rate_override,
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
        resume=resume,
    )

    episode_metrics_path = run_path / "episode_metrics.jsonl"
    update_metrics_path = run_path / "update_metrics.jsonl"
    slot_metrics_path = run_path / SLOT_METRICS_FILENAME
    for episode_index in range(completed, int(target_episodes)):
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
        completed = episode_index + 1
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
            **collection.summary,
        }
        update_record = {
            "record_type": "update",
            "episode": completed,
            **update_metrics,
            **trainer.training_counters,
        }
        append_dual_timescale_record(episode_metrics_path, episode_record)
        append_dual_timescale_record(update_metrics_path, update_record)
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
        if verbose:
            print(json.dumps(update_record, ensure_ascii=False, sort_keys=True))
    return {
        "run_dir": str(run_path),
        "completed_episodes": completed,
        "target_episodes": int(target_episodes),
        "checkpoint": str(checkpoint_path),
        "best_episode_return": best_return,
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
    parser.add_argument("--critic-learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
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
        critic_learning_rate_override=args.critic_learning_rate,
        seed=args.seed,
        resume=bool(args.resume),
        verbose=bool(args.verbose),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
