"""Minimal PPO v2 trainer for adaptive Cache/UAV request actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - CLI help/import-only environments
    torch = None  # type: ignore[assignment]

from common import project_config
from decision import cache_configuration, uav_configuration
from scenario import task_scenario

from . import interface
from .interface import _copy_uav_positions
from .policy import AdaptivePrimitiveActorCritic, BernoulliActorCritic
from .primitive_control import PrimitiveTensorSpec
from .rollout import SimpleRollout, collect_rollout, compute_gae


PPO_V2_METHOD = "ppo_simple_v2"
CHECKPOINT_FILENAME = "latest_checkpoint.pt"
CHECKPOINT_VERSION = 2
PRIMITIVE_CHECKPOINT_VERSION = 3
DEFAULT_ROLLOUT_STEPS = 16


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PPO v2 training requires PyTorch.")


@dataclass(frozen=True)
class EnvironmentContext:
    cfg: dict[str, Any]
    root: Path
    cache_cfg: cache_configuration.CacheOptimizationConfig
    scenarios: dict[str, Any]
    service_catalog: dict[str, Any]
    cache_nodes: dict[str, Any]
    local_cache: dict[str, set[int]]
    flight_boundary: tuple[float, float, float, float]
    action_spec: interface.TriggerActionSpec
    observation_spec: interface.ConfigurationObservationSpec
    initial_cache: np.ndarray
    initial_uav: dict[str, np.ndarray]
    train_origin_start: int
    train_origin_end: int
    configuration_control_mode: str


@dataclass(frozen=True)
class PPOV2Hyperparameters:
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    learning_rate: float
    value_loss_coefficient: float
    entropy_coefficient: float
    max_grad_norm: float
    update_epochs: int
    num_minibatches: int
    hidden_sizes: tuple[int, ...]


@dataclass(frozen=True)
class UpdateResult:
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    total_loss: float
    gradient_finite: bool
    returns_finite: bool
    advantages_finite: bool
    gradient_norm: float
    policy_loss_min: float
    policy_loss_max: float
    value_loss_min: float
    value_loss_max: float
    approximate_kl_min: float
    approximate_kl_max: float
    gradient_norm_min: float
    gradient_norm_max: float
    parameter_delta_norm: float
    optimizer_steps: int


@dataclass(frozen=True)
class TrainingCursor:
    completed_episodes: int
    current_train_origin: int
    current_cache_state: np.ndarray
    current_uav_positions: dict[str, tuple[float, float]]


def _project_path(root: Path, value: Any) -> Path:
    return cache_configuration.resolve_path(root, value)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one finite, structured training metric record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


def load_environment_context(config_path: str = "system_config.toml") -> EnvironmentContext:
    cfg = project_config.load_config(config_path)
    project_config.validate_for_formal_experiment(cfg)
    root = Path(cfg["_meta"]["project_root"])
    cache_cfg = cache_configuration.cache_config_from_project(cfg)
    scenarios = cache_configuration.load_prediction_scenarios(
        _project_path(root, project_config.get_config_value(cfg, "paths.prediction_scenarios"))
    )
    service_catalog = cache_configuration.load_service_catalog(
        _project_path(root, project_config.get_config_value(cfg, "paths.service_catalog"))
    )
    train_start = int(project_config.get_config_value(cfg, "algorithm.ppo.data_protocol.train_origin_start"))
    train_end = int(project_config.get_config_value(cfg, "algorithm.ppo.data_protocol.train_origin_end"))
    origin_count = int(scenarios["origin_slots"].shape[0])
    if train_start < 0 or train_end < train_start or train_end >= origin_count:
        raise ValueError("PPO v2 train origin range 超出 prediction_scenarios。")
    first_origin = cache_configuration.scenario_at_origin(scenarios, train_start)
    cache_nodes = cache_configuration.build_cache_nodes(first_origin, cache_cfg)
    local_cache = task_scenario.load_tv_local_cache(
        _project_path(root, project_config.get_config_value(cfg, "paths.tv_local_cache"))
    )
    action_spec = interface.TriggerActionSpec.from_cache_nodes_and_project(cache_nodes, cfg)
    observation_spec = interface.ConfigurationObservationSpec.from_cache_nodes(
        cache_nodes,
        service_catalog["service_ids"].tolist(),
        action_spec.uav_ids,
        [str(value) for value in first_origin["vehicle_ids"].tolist()],
    )
    initial_cache = cache_configuration.popularity_aware_initial_cache(
        service_catalog,
        cache_nodes,
        cache_cfg,
    )
    initial_uav = uav_configuration.initial_uav_positions_from_project(cfg)
    configuration_control_mode = str(
        project_config.get_config_value(
            cfg,
            "algorithm.ppo.configuration_control_mode",
            "legacy_exact",
        )
    )
    return EnvironmentContext(
        cfg=cfg,
        root=root,
        cache_cfg=cache_cfg,
        scenarios=scenarios,
        service_catalog=service_catalog,
        cache_nodes=cache_nodes,
        local_cache=local_cache,
        flight_boundary=uav_configuration.boundary_from_project(cfg),
        action_spec=action_spec,
        observation_spec=observation_spec,
        initial_cache=initial_cache,
        initial_uav=initial_uav,
        train_origin_start=train_start,
        train_origin_end=train_end,
        configuration_control_mode=configuration_control_mode,
    )


def hyperparameters_from_project(cfg: dict[str, Any]) -> PPOV2Hyperparameters:
    get = project_config.get_config_value
    hidden_sizes = tuple(int(value) for value in get(cfg, "algorithm.ppo.hidden_sizes"))
    values = PPOV2Hyperparameters(
        gamma=float(get(cfg, "algorithm.ppo.gamma")),
        gae_lambda=float(get(cfg, "algorithm.ppo.gae_lambda")),
        clip_epsilon=float(get(cfg, "algorithm.ppo.clip_epsilon")),
        learning_rate=float(get(cfg, "algorithm.ppo.learning_rate")),
        value_loss_coefficient=float(get(cfg, "algorithm.ppo.value_loss_coefficient")),
        entropy_coefficient=float(get(cfg, "algorithm.ppo.entropy_coefficient")),
        max_grad_norm=float(get(cfg, "algorithm.ppo.max_grad_norm")),
        update_epochs=int(get(cfg, "algorithm.ppo.update_epochs")),
        num_minibatches=int(get(cfg, "algorithm.ppo.num_minibatches")),
        hidden_sizes=hidden_sizes,
    )
    if values.gamma < 0.0 or values.gamma > 1.0:
        raise ValueError("gamma 必须位于 [0,1]。")
    if values.gae_lambda < 0.0 or values.gae_lambda > 1.0:
        raise ValueError("gae_lambda 必须位于 [0,1]。")
    if values.clip_epsilon < 0.0 or values.clip_epsilon >= 1.0:
        raise ValueError("clip_epsilon 必须位于 [0,1)。")
    if values.learning_rate <= 0.0 or values.value_loss_coefficient < 0.0:
        raise ValueError("PPO v2 learning/value coefficient 非法。")
    if values.entropy_coefficient < 0.0 or values.max_grad_norm <= 0.0:
        raise ValueError("PPO v2 entropy/max_grad_norm 非法。")
    if values.update_epochs <= 0 or values.num_minibatches <= 0:
        raise ValueError("PPO v2 update_epochs/num_minibatches 必须为正整数。")
    return values


def aligned_train_steps(
    context: EnvironmentContext,
    origin_start: int | None = None,
    max_steps: int | None = None,
    *,
    include_bootstrap_step: bool = False,
) -> list[Any]:
    start = context.train_origin_start if origin_start is None else int(origin_start)
    if start < context.train_origin_start or start > context.train_origin_end:
        raise ValueError("origin_start 必须位于 formal train origins 内。")
    available_steps = context.train_origin_end - start + 1
    step_count = available_steps
    if max_steps is not None:
        if int(max_steps) <= 0:
            raise ValueError("max_steps 必须大于0。")
        step_count = min(available_steps, int(max_steps))
    if include_bootstrap_step and start + step_count > context.train_origin_end:
        raise ValueError("train terminal 后不允许读取 test origin 作为 bootstrap step。")
    return interface.prepare_aligned_rollout_steps(
        context.root,
        context.scenarios,
        start,
        step_count,
        _project_path(context.root, project_config.get_config_value(context.cfg, "paths.vehicle_predictions")),
        context.root / "data/generated/popularity_predictions.csv",
        _project_path(context.root, project_config.get_config_value(context.cfg, "paths.tasks")),
        _project_path(context.root, project_config.get_config_value(context.cfg, "paths.trajectory")),
        include_bootstrap_step=bool(include_bootstrap_step),
    )


class PPOV2Trainer:
    """PPO core for legacy trigger-only and adaptive primitive control modes."""

    def __init__(self, context: EnvironmentContext, device: str = "cpu") -> None:
        _require_torch()
        self.context = context
        self.hyperparameters = hyperparameters_from_project(context.cfg)
        self.device = torch.device(device)
        if context.configuration_control_mode == "policy_primitive":
            primitive_spec = PrimitiveTensorSpec.create(context)
            self.policy = AdaptivePrimitiveActorCritic(
                int(context.observation_spec.observation_dim),
                int(context.action_spec.action_dim),
                primitive_spec.cache_feature_dim,
                primitive_spec.uav_feature_dim,
                self.hyperparameters.hidden_sizes,
            ).to(self.device)
        elif context.configuration_control_mode == "legacy_exact":
            self.policy = BernoulliActorCritic(
                int(context.observation_spec.observation_dim),
                int(context.action_spec.action_dim),
                self.hyperparameters.hidden_sizes,
            ).to(self.device)
        else:
            raise ValueError(
                f"unknown configuration_control_mode={context.configuration_control_mode}"
            )
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.hyperparameters.learning_rate,
        )
        self.optimizer_steps = 0

    def collect_chunk(
        self,
        origin_start: int,
        cache_state: np.ndarray,
        uav_positions: dict[str, Any],
        rollout_steps: int,
    ) -> SimpleRollout:
        """Collect one contiguous chunk without resetting environment state."""

        start = int(origin_start)
        if start < self.context.train_origin_start or start > self.context.train_origin_end:
            raise ValueError("chunk origin_start 超出 formal train origins。")
        if int(rollout_steps) <= 0:
            raise ValueError("rollout_steps 必须大于0。")
        remaining = self.context.train_origin_end - start + 1
        chunk_size = min(int(rollout_steps), remaining)
        prepared_steps = aligned_train_steps(
            self.context,
            origin_start=start,
            max_steps=chunk_size,
            include_bootstrap_step=chunk_size < remaining,
        )
        if len(prepared_steps) < chunk_size:
            raise RuntimeError("aligned train steps 少于请求的 rollout chunk。")
        execution_steps = prepared_steps[:chunk_size]
        bootstrap_step = prepared_steps[chunk_size] if len(prepared_steps) > chunk_size else None
        return collect_rollout(
            self.policy,
            self.context,
            execution_steps,
            initial_cache=cache_state,
            initial_uav=uav_positions,
            bootstrap_step=bootstrap_step,
            terminal_origin_index=self.context.train_origin_end,
        )

    def collect_episode(self, max_steps: int | None = None) -> SimpleRollout:
        """Compatibility helper that collects from the initial train state."""

        requested_steps = self.context.train_origin_end - self.context.train_origin_start + 1
        if max_steps is not None:
            requested_steps = min(requested_steps, int(max_steps))
        return self.collect_chunk(
            self.context.train_origin_start,
            self.context.initial_cache,
            self.context.initial_uav,
            requested_steps,
        )

    def update(self, rollout: SimpleRollout) -> UpdateResult:
        hyper = self.hyperparameters
        bootstrap_value = 0.0
        if rollout.bootstrap_observation is not None:
            with torch.no_grad():
                _, bootstrap_tensor = self.policy(rollout.bootstrap_observation)
            bootstrap_value = float(bootstrap_tensor.detach().cpu())
        advantages, returns = compute_gae(
            rollout.rewards,
            rollout.old_values,
            rollout.dones,
            bootstrap_value,
            hyper.gamma,
            hyper.gae_lambda,
        )
        returns_finite = bool(np.all(np.isfinite(returns)))
        advantages_finite = bool(np.all(np.isfinite(advantages)))
        if not returns_finite or not advantages_finite:
            raise FloatingPointError("PPO v2 return/advantage 非 finite。")
        sample_count = int(rollout.observations.shape[0])
        if sample_count < hyper.num_minibatches or sample_count % hyper.num_minibatches != 0:
            raise ValueError("PPO v2 rollout length 必须可被 num_minibatches 整除。")
        observations = torch.as_tensor(rollout.observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(rollout.actions, dtype=torch.long, device=self.device)
        masks = torch.as_tensor(rollout.action_masks, dtype=torch.bool, device=self.device)
        cache_contexts = torch.as_tensor(
            rollout.cache_contexts, dtype=torch.float32, device=self.device
        )
        cache_candidate_features = torch.as_tensor(
            rollout.cache_candidate_features, dtype=torch.float32, device=self.device
        )
        cache_candidate_masks = torch.as_tensor(
            rollout.cache_candidate_masks, dtype=torch.bool, device=self.device
        )
        cache_selected_indices = torch.as_tensor(
            rollout.cache_selected_indices, dtype=torch.long, device=self.device
        )
        uav_contexts = torch.as_tensor(
            rollout.uav_contexts, dtype=torch.float32, device=self.device
        )
        uav_candidate_features = torch.as_tensor(
            rollout.uav_candidate_features, dtype=torch.float32, device=self.device
        )
        uav_candidate_masks = torch.as_tensor(
            rollout.uav_candidate_masks, dtype=torch.bool, device=self.device
        )
        uav_selected_indices = torch.as_tensor(
            rollout.uav_selected_indices, dtype=torch.long, device=self.device
        )
        old_log_probs = torch.as_tensor(rollout.old_log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        if sample_count > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1.0e-8)

        before = [parameter.detach().clone() for parameter in self.policy.parameters()]
        policy_loss_value = 0.0
        value_loss_value = 0.0
        entropy_value = 0.0
        approximate_kl_value = 0.0
        clip_fraction_value = 0.0
        total_loss_value = 0.0
        policy_loss_values: list[float] = []
        value_loss_values: list[float] = []
        approximate_kl_values: list[float] = []
        gradient_norm_values: list[float] = []
        gradient_finite = True
        minibatch_size = sample_count // hyper.num_minibatches
        for _ in range(hyper.update_epochs):
            permutation = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                if self.context.configuration_control_mode == "policy_primitive":
                    if not isinstance(self.policy, AdaptivePrimitiveActorCritic):
                        raise TypeError("policy_primitive update requires adaptive policy。")
                    evaluation = self.policy.evaluate_primitive(
                        observations[indices],
                        actions[indices],
                        masks[indices],
                        cache_contexts[indices],
                        cache_candidate_features[indices],
                        cache_candidate_masks[indices],
                        cache_selected_indices[indices],
                        uav_contexts[indices],
                        uav_candidate_features[indices],
                        uav_candidate_masks[indices],
                        uav_selected_indices[indices],
                        len(self.context.action_spec.cache_node_ids),
                    )
                else:
                    evaluation = self.policy.evaluate(
                        observations[indices],
                        actions[indices],
                        masks[indices],
                    )
                ratio = torch.exp(evaluation.log_prob - old_log_probs[indices])
                unclipped = ratio * advantages_t[indices]
                clipped = torch.clamp(
                    ratio,
                    1.0 - hyper.clip_epsilon,
                    1.0 + hyper.clip_epsilon,
                ) * advantages_t[indices]
                approximate_kl = torch.mean(old_log_probs[indices] - evaluation.log_prob)
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1.0) > hyper.clip_epsilon).to(torch.float32)
                )
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * torch.mean((evaluation.value - returns_t[indices]) ** 2)
                entropy = evaluation.entropy.mean()
                total_loss = (
                    policy_loss
                    + hyper.value_loss_coefficient * value_loss
                    - hyper.entropy_coefficient * entropy
                )
                if not bool(torch.isfinite(total_loss).item()):
                    raise FloatingPointError("PPO v2 total loss 非 finite。")
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                for parameter in self.policy.parameters():
                    if parameter.grad is not None and not bool(torch.all(torch.isfinite(parameter.grad)).item()):
                        gradient_finite = False
                if not gradient_finite:
                    raise FloatingPointError("PPO v2 gradient 非 finite。")
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), hyper.max_grad_norm
                )
                gradient_norm_value = float(gradient_norm.detach().cpu())
                if not np.isfinite(gradient_norm_value):
                    raise FloatingPointError("PPO v2 gradient norm 非 finite。")
                self.optimizer.step()
                self.optimizer_steps += 1
                policy_loss_value = float(policy_loss.detach().cpu())
                value_loss_value = float(value_loss.detach().cpu())
                entropy_value = float(entropy.detach().cpu())
                approximate_kl_value = float(approximate_kl.detach().cpu())
                clip_fraction_value = float(clip_fraction.detach().cpu())
                total_loss_value = float(total_loss.detach().cpu())
                policy_loss_values.append(policy_loss_value)
                value_loss_values.append(value_loss_value)
                approximate_kl_values.append(approximate_kl_value)
                gradient_norm_values.append(gradient_norm_value)
        delta_squared = 0.0
        for old_parameter, new_parameter in zip(before, self.policy.parameters()):
            delta_squared += float(torch.sum((new_parameter.detach() - old_parameter) ** 2).cpu())
        return UpdateResult(
            policy_loss=policy_loss_value,
            value_loss=value_loss_value,
            entropy=entropy_value,
            approximate_kl=approximate_kl_value,
            clip_fraction=clip_fraction_value,
            total_loss=total_loss_value,
            gradient_finite=gradient_finite,
            returns_finite=returns_finite,
            advantages_finite=advantages_finite,
            gradient_norm=gradient_norm_values[-1],
            policy_loss_min=float(min(policy_loss_values)),
            policy_loss_max=float(max(policy_loss_values)),
            value_loss_min=float(min(value_loss_values)),
            value_loss_max=float(max(value_loss_values)),
            approximate_kl_min=float(min(approximate_kl_values)),
            approximate_kl_max=float(max(approximate_kl_values)),
            gradient_norm_min=float(min(gradient_norm_values)),
            gradient_norm_max=float(max(gradient_norm_values)),
            parameter_delta_norm=float(np.sqrt(delta_squared)),
            optimizer_steps=self.optimizer_steps,
        )

    def checkpoint_payload(
        self,
        completed_episodes: int,
        current_train_origin: int,
        current_cache_state: np.ndarray,
        current_uav_positions: dict[str, Any],
        config_hash: str,
        rollout_steps: int,
    ) -> dict[str, Any]:
        return {
            "checkpoint_version": (
                PRIMITIVE_CHECKPOINT_VERSION
                if self.context.configuration_control_mode == "policy_primitive"
                else CHECKPOINT_VERSION
            ),
            "method": PPO_V2_METHOD,
            "policy_schema": self.policy.architecture_id,
            "configuration_control_mode": self.context.configuration_control_mode,
            "episode": int(completed_episodes),
            "completed_episodes": int(completed_episodes),
            "current_train_origin": int(current_train_origin),
            "current_cache_state": np.asarray(current_cache_state, dtype=np.int8).copy(),
            "current_uav_positions": _copy_uav_positions(current_uav_positions),
            "rollout_steps": int(rollout_steps),
            "config_sha256": str(config_hash),
            "observation_dim": int(self.context.observation_spec.observation_dim),
            "action_dim": int(self.context.action_spec.action_dim),
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "optimizer_steps": int(self.optimizer_steps),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
        }

    def save_checkpoint(
        self,
        run_dir: Path,
        completed_episodes: int,
        current_train_origin: int,
        current_cache_state: np.ndarray,
        current_uav_positions: dict[str, Any],
        config_hash: str,
        rollout_steps: int,
    ) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = run_dir / CHECKPOINT_FILENAME
        torch.save(
            self.checkpoint_payload(
                completed_episodes,
                current_train_origin,
                current_cache_state,
                current_uav_positions,
                config_hash,
                rollout_steps,
            ),
            checkpoint,
        )
        return checkpoint

    def load_checkpoint(self, checkpoint: Path, config_hash: str) -> TrainingCursor:
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        expected_version = (
            PRIMITIVE_CHECKPOINT_VERSION
            if self.context.configuration_control_mode == "policy_primitive"
            else CHECKPOINT_VERSION
        )
        if int(payload.get("checkpoint_version", -1)) != expected_version:
            raise ValueError("PPO v2 checkpoint version 不匹配。")
        expected_schema = self.policy.architecture_id
        checkpoint_schema = payload.get("policy_schema")
        if checkpoint_schema is None and expected_schema == BernoulliActorCritic.architecture_id:
            checkpoint_schema = BernoulliActorCritic.architecture_id
        if checkpoint_schema != expected_schema:
            raise ValueError(
                "PPO checkpoint policy schema incompatible: "
                f"checkpoint={checkpoint_schema}, expected={expected_schema}。"
            )
        if payload.get("configuration_control_mode", "legacy_exact") != (
            self.context.configuration_control_mode
        ):
            raise ValueError("PPO checkpoint configuration_control_mode 不匹配。")
        if payload.get("method") != PPO_V2_METHOD or payload.get("config_sha256") != config_hash:
            raise ValueError("PPO v2 checkpoint method/config hash 不匹配。")
        if int(payload.get("observation_dim")) != int(self.context.observation_spec.observation_dim):
            raise ValueError("PPO v2 checkpoint observation_dim 不匹配。")
        if int(payload.get("action_dim")) != int(self.context.action_spec.action_dim):
            raise ValueError("PPO v2 checkpoint action_dim 不匹配。")
        self.policy.load_state_dict(payload["policy_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.optimizer_steps = int(payload.get("optimizer_steps", 0))
        rng_state = payload.get("rng_state", {})
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        current_cache_state = np.asarray(payload["current_cache_state"], dtype=np.int8).copy()
        if current_cache_state.shape != np.asarray(self.context.initial_cache).shape:
            raise ValueError("checkpoint current_cache_state shape 不匹配。")
        current_uav_positions = _copy_uav_positions(payload["current_uav_positions"])
        if set(current_uav_positions) != set(self.context.action_spec.uav_ids):
            raise ValueError("checkpoint current_uav_positions UAV 集合不匹配。")
        current_train_origin = int(payload["current_train_origin"])
        if not self.context.train_origin_start <= current_train_origin <= self.context.train_origin_end + 1:
            raise ValueError("checkpoint current_train_origin 超出 formal train range。")
        return TrainingCursor(
            completed_episodes=int(payload["completed_episodes"]),
            current_train_origin=current_train_origin,
            current_cache_state=current_cache_state,
            current_uav_positions=current_uav_positions,
        )


def _config_hash(config_path: str) -> str:
    path = Path(config_path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_training(
    config_path: str,
    episodes: int,
    run_dir: str,
    *,
    rollout_steps: int = DEFAULT_ROLLOUT_STEPS,
    resume: bool = False,
) -> None:
    _require_torch()
    if int(episodes) <= 0:
        raise ValueError("episodes 必须大于0。")
    if int(rollout_steps) <= 0:
        raise ValueError("rollout_steps 必须大于0。")
    context = load_environment_context(config_path)
    seed = int(project_config.get_config_value(context.cfg, "experiment.seed"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainer = PPOV2Trainer(context)
    config_hash = _config_hash(config_path)
    completed_episodes = 0
    current_train_origin = context.train_origin_start
    current_cache_state = np.asarray(context.initial_cache, dtype=np.int8).copy()
    current_uav_positions = _copy_uav_positions(context.initial_uav)
    run_path = Path(run_dir)
    checkpoint = run_path / CHECKPOINT_FILENAME
    episode_metrics_path = run_path / "episode_metrics.jsonl"
    update_metrics_path = run_path / "update_metrics.jsonl"
    if resume:
        if not checkpoint.exists():
            raise FileNotFoundError(f"resume checkpoint 不存在：{checkpoint}")
        cursor = trainer.load_checkpoint(checkpoint, config_hash)
        completed_episodes = cursor.completed_episodes
        current_train_origin = cursor.current_train_origin
        current_cache_state = cursor.current_cache_state
        current_uav_positions = cursor.current_uav_positions
        if completed_episodes > int(episodes):
            raise ValueError("checkpoint completed_episodes 超过本次 episodes。")

    episode_started_at = time.perf_counter()
    episode_optimizer_start = trainer.optimizer_steps
    episode_rewards: list[float] = []
    episode_g_values: list[float] = []
    episode_cache_requests = 0
    episode_actual_cache_updates = 0
    episode_uav_requests = 0
    episode_actual_uav_movements = 0
    episode_cache_cost = 0.0
    episode_movement_cost = 0.0
    episode_update_count = 0
    episode_policy_loss_mins: list[float] = []
    episode_policy_loss_maxs: list[float] = []
    episode_value_loss_mins: list[float] = []
    episode_value_loss_maxs: list[float] = []
    episode_kl_mins: list[float] = []
    episode_kl_maxs: list[float] = []
    episode_gradient_norm_mins: list[float] = []
    episode_gradient_norm_maxs: list[float] = []

    while completed_episodes < int(episodes):
        if current_train_origin > context.train_origin_end:
            current_train_origin = context.train_origin_start
            current_cache_state = np.asarray(context.initial_cache, dtype=np.int8).copy()
            current_uav_positions = _copy_uav_positions(context.initial_uav)
            episode_started_at = time.perf_counter()
            episode_optimizer_start = trainer.optimizer_steps
            episode_rewards = []
            episode_g_values = []
            episode_cache_requests = 0
            episode_actual_cache_updates = 0
            episode_uav_requests = 0
            episode_actual_uav_movements = 0
            episode_cache_cost = 0.0
            episode_movement_cost = 0.0
            episode_update_count = 0
            episode_policy_loss_mins = []
            episode_policy_loss_maxs = []
            episode_value_loss_mins = []
            episode_value_loss_maxs = []
            episode_kl_mins = []
            episode_kl_maxs = []
            episode_gradient_norm_mins = []
            episode_gradient_norm_maxs = []
        episode_number = completed_episodes + 1
        chunk_origin_start = current_train_origin
        rollout = trainer.collect_chunk(
            current_train_origin,
            current_cache_state,
            current_uav_positions,
            int(rollout_steps),
        )
        update = trainer.update(rollout)
        next_train_origin = current_train_origin + len(rollout.transitions)
        terminal = next_train_origin > context.train_origin_end
        if terminal:
            completed_episodes += 1
        checkpoint_path = trainer.save_checkpoint(
            Path(run_dir),
            completed_episodes,
            next_train_origin,
            rollout.next_cache_state,
            rollout.next_uav_positions,
            config_hash,
            int(rollout_steps),
        )
        cache_count = len(context.action_spec.cache_node_ids)
        transition_count = len(rollout.transitions)
        cache_requests = int(rollout.actions[:, :cache_count].sum())
        uav_requests = int(rollout.actions[:, cache_count:].sum())
        actual_cache_updates = int(
            sum(item.actual_cache_update_count for item in rollout.transitions)
        )
        actual_uav_movements = int(
            sum(item.actual_uav_movement_count for item in rollout.transitions)
        )
        transition_rewards = np.asarray(
            [item.reward for item in rollout.transitions], dtype=np.float64
        )
        transition_g_values = np.asarray(
            [
                float(item.actual_task_utility)
                - float(item.cache_cost)
                - float(item.movement_cost)
                for item in rollout.transitions
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(transition_rewards)):
            raise FloatingPointError("PPO v2 reward 非 finite。")
        if not np.all(np.isfinite(transition_g_values)):
            raise FloatingPointError("PPO v2 G(t) 非 finite。")
        if not np.allclose(transition_rewards, transition_g_values, rtol=1e-12, atol=1e-12):
            raise AssertionError("PPO v2 reward 与 actual G(t) 不一致。")

        episode_rewards.extend(transition_rewards.tolist())
        episode_g_values.extend(transition_g_values.tolist())
        episode_cache_requests += cache_requests
        episode_actual_cache_updates += actual_cache_updates
        episode_uav_requests += uav_requests
        episode_actual_uav_movements += actual_uav_movements
        episode_cache_cost += float(sum(item.cache_cost for item in rollout.transitions))
        episode_movement_cost += float(sum(item.movement_cost for item in rollout.transitions))
        episode_update_count += 1
        episode_policy_loss_mins.append(update.policy_loss_min)
        episode_policy_loss_maxs.append(update.policy_loss_max)
        episode_value_loss_mins.append(update.value_loss_min)
        episode_value_loss_maxs.append(update.value_loss_max)
        episode_kl_mins.append(update.approximate_kl_min)
        episode_kl_maxs.append(update.approximate_kl_max)
        episode_gradient_norm_mins.append(update.gradient_norm_min)
        episode_gradient_norm_maxs.append(update.gradient_norm_max)
        _append_jsonl(
            update_metrics_path,
            {
                "record_type": "update",
                "seed": seed,
                "episode": episode_number,
                "origin_start": int(chunk_origin_start),
                "origin_end": int(next_train_origin - 1),
                "num_transitions": transition_count,
                "terminal": bool(terminal),
                "optimizer_steps": int(update.optimizer_steps),
                "policy_loss": float(update.policy_loss),
                "policy_loss_min": float(update.policy_loss_min),
                "policy_loss_max": float(update.policy_loss_max),
                "value_loss": float(update.value_loss),
                "value_loss_min": float(update.value_loss_min),
                "value_loss_max": float(update.value_loss_max),
                "entropy": float(update.entropy),
                "approximate_kl": float(update.approximate_kl),
                "approximate_kl_min": float(update.approximate_kl_min),
                "approximate_kl_max": float(update.approximate_kl_max),
                "clip_fraction": float(update.clip_fraction),
                "gradient_norm": float(update.gradient_norm),
                "gradient_norm_min": float(update.gradient_norm_min),
                "gradient_norm_max": float(update.gradient_norm_max),
                "gradient_finite": bool(update.gradient_finite),
                "returns_finite": bool(update.returns_finite),
                "advantages_finite": bool(update.advantages_finite),
                "total_loss": float(update.total_loss),
                "total_loss_finite": bool(np.isfinite(update.total_loss)),
                "checkpoint": str(checkpoint_path),
            },
        )
        print(
            f"[PPO-v2] update={update.optimizer_steps} episode={episode_number} "
            f"completed_episodes={completed_episodes} "
            f"origins={chunk_origin_start}:{next_train_origin - 1} "
            f"steps={len(rollout.transitions)} terminal={terminal} "
            f"reward_mean={float(rollout.rewards.mean()):.6f} "
            f"reward_min={float(rollout.rewards.min()):.6f} "
            f"reward_max={float(rollout.rewards.max()):.6f} "
            f"cache_requests={cache_requests}/{transition_count * cache_count} "
            f"uav_requests={uav_requests}/{transition_count * len(context.action_spec.uav_ids)} "
            f"actual_cache_updates={actual_cache_updates} "
            f"actual_uav_movements={actual_uav_movements} "
            f"policy_loss={update.policy_loss:.6f} "
            f"value_loss={update.value_loss:.6f} "
            f"entropy={update.entropy:.6f} "
            f"approx_kl={update.approximate_kl:.6f} "
            f"clip_fraction={update.clip_fraction:.6f} "
            f"parameter_delta={update.parameter_delta_norm:.6e} "
            f"checkpoint={checkpoint_path} "
            f"next_origin={next_train_origin}",
            flush=True,
        )
        if terminal:
            episode_reward_array = np.asarray(episode_rewards, dtype=np.float64)
            episode_g_array = np.asarray(episode_g_values, dtype=np.float64)
            if not np.all(np.isfinite(episode_reward_array)):
                raise FloatingPointError("PPO v2 episode reward 非 finite。")
            if not np.all(np.isfinite(episode_g_array)):
                raise FloatingPointError("PPO v2 episode G(t) 非 finite。")
            episode_finite = {
                "reward": bool(np.all(np.isfinite(episode_reward_array))),
                "return": bool(np.isfinite(float(episode_reward_array.sum()))),
                "G": bool(np.all(np.isfinite(episode_g_array))),
                "loss": bool(
                    np.isfinite(episode_policy_loss_mins).all()
                    and np.isfinite(episode_policy_loss_maxs).all()
                    and np.isfinite(episode_value_loss_mins).all()
                    and np.isfinite(episode_value_loss_maxs).all()
                    and np.isfinite(episode_kl_mins).all()
                    and np.isfinite(episode_kl_maxs).all()
                ),
                "gradient": bool(
                    np.isfinite(episode_gradient_norm_mins).all()
                    and np.isfinite(episode_gradient_norm_maxs).all()
                    and update.gradient_finite
                ),
                "returns": bool(update.returns_finite),
                "advantages": bool(update.advantages_finite),
            }
            if not all(episode_finite.values()):
                raise FloatingPointError("PPO v2 episode metrics 非 finite。")
            _append_jsonl(
                episode_metrics_path,
                {
                    "record_type": "episode",
                    "seed": seed,
                    "episode": int(episode_number),
                    "completed_episodes": int(completed_episodes),
                    "num_transitions": int(len(episode_reward_array)),
                    "episode_return": float(episode_reward_array.sum()),
                    "mean_reward": float(episode_reward_array.mean()),
                    "mean_G": float(episode_g_array.mean()),
                    "reward_std": float(episode_reward_array.std()),
                    "cache_request_count": int(episode_cache_requests),
                    "actual_cache_update_count": int(episode_actual_cache_updates),
                    "uav_request_count": int(episode_uav_requests),
                    "actual_uav_movement_count": int(episode_actual_uav_movements),
                    "cache_cost": float(episode_cache_cost),
                    "movement_cost": float(episode_movement_cost),
                    "ppo_updates": int(episode_update_count),
                    "optimizer_steps": int(update.optimizer_steps),
                    "episode_optimizer_steps": int(update.optimizer_steps - episode_optimizer_start),
                    "policy_loss_range": {
                        "min": float(min(episode_policy_loss_mins)),
                        "max": float(max(episode_policy_loss_maxs)),
                    },
                    "value_loss_range": {
                        "min": float(min(episode_value_loss_mins)),
                        "max": float(max(episode_value_loss_maxs)),
                    },
                    "approximate_kl_range": {
                        "min": float(min(episode_kl_mins)),
                        "max": float(max(episode_kl_maxs)),
                    },
                    "gradient_norm_range": {
                        "min": float(min(episode_gradient_norm_mins)),
                        "max": float(max(episode_gradient_norm_maxs)),
                    },
                    "finite_checks": episode_finite,
                    "checkpoint": str(checkpoint_path),
                    "wall_clock_runtime_seconds": float(time.perf_counter() - episode_started_at),
                },
            )
        current_train_origin = next_train_origin
        current_cache_state = np.asarray(rollout.next_cache_state, dtype=np.int8).copy()
        current_uav_positions = _copy_uav_positions(rollout.next_uav_positions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal PPO v2 Cache/UAV request trainer")
    parser.add_argument("--config", default="system_config.toml")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=DEFAULT_ROLLOUT_STEPS,
        help=f"连续 train slots per PPO update (default: {DEFAULT_ROLLOUT_STEPS})",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_training(
        args.config,
        args.episodes,
        args.run_dir,
        rollout_steps=args.rollout_steps,
        resume=bool(args.resume),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
