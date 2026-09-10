"""Minimal PPO v2 trainer for adaptive Cache/UAV request actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from .plotting import (
    SLOT_METRICS_FILENAME,
    append_training_slot_metrics,
    update_training_slot_reward_plot,
)
from .policy import (
    AdaptivePrimitiveActorCritic,
    AdaptivePrimitiveGroupedV2ActorCritic,
    AdaptivePrimitiveHorizonGroupedV3ActorCritic,
    AdaptivePrimitiveHorizonGroupedSeparateCriticV4ActorCritic,
    AdaptivePrimitiveHorizonGroupedSeparateCriticOptimizationV5ActorCritic,
    AdaptivePrimitiveV2ActorCritic,
    BernoulliActorCritic,
)
from .primitive_control import PrimitiveTensorSpec
from .rollout import SimpleRollout, collect_rollout, compute_gae


PPO_V2_METHOD = "ppo_simple_v2"
CHECKPOINT_FILENAME = "latest_checkpoint.pt"
BEST_TRAIN_RETURN_CHECKPOINT_FILENAME = "best_train_return_checkpoint.pt"
CHECKPOINT_VERSION = 2
PRIMITIVE_CHECKPOINT_VERSION = 3
VALUE_NORMALIZED_CHECKPOINT_VERSION = 4
SEPARATE_OPTIMIZATION_CHECKPOINT_VERSION = 5
DEFAULT_ROLLOUT_STEPS = 16
ADVANTAGE_NORMALIZATION_CENTERED_STD = "centered_std"
ADVANTAGE_NORMALIZATION_STD_ONLY = "std_only"
DEFAULT_ADVANTAGE_NORMALIZATION_MODE = ADVANTAGE_NORMALIZATION_CENTERED_STD
CRITIC_VALUE_SCALE_NONE = "none"
CRITIC_VALUE_SCALE_RUNNING_RETURN_STD = "running_return_std"
DEFAULT_CRITIC_VALUE_SCALE_EPSILON = 1.0e-8
DEFAULT_CRITIC_VALUE_SCALE_WARMUP_SAMPLES = 64
V2_POLICY_SCHEMAS = frozenset(
    {
        AdaptivePrimitiveV2ActorCritic.architecture_id,
        AdaptivePrimitiveGroupedV2ActorCritic.architecture_id,
    }
)
V3_POLICY_SCHEMAS = frozenset(
    {AdaptivePrimitiveHorizonGroupedV3ActorCritic.architecture_id}
)
V4_POLICY_SCHEMAS = frozenset(
    {AdaptivePrimitiveHorizonGroupedSeparateCriticV4ActorCritic.architecture_id}
)
V5_POLICY_SCHEMAS = frozenset(
    {
        AdaptivePrimitiveHorizonGroupedSeparateCriticOptimizationV5ActorCritic.architecture_id
    }
)
HORIZON_POLICY_SCHEMAS = V3_POLICY_SCHEMAS | V4_POLICY_SCHEMAS | V5_POLICY_SCHEMAS
MODERN_POLICY_SCHEMAS = V2_POLICY_SCHEMAS | HORIZON_POLICY_SCHEMAS


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
    policy_schema: str


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
    advantage_normalization_mode: str
    critic_value_scale_mode: str
    critic_value_scale_epsilon: float
    critic_value_scale_warmup_samples: int


@dataclass
class RunningReturnStatistics:
    """Population Welford statistics for completed training raw returns."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def scale(self, epsilon: float, warmup_samples: int) -> float:
        minimum_count = max(2, int(warmup_samples))
        if self.count < minimum_count:
            return 1.0
        variance = self.m2 / float(self.count)
        if not math.isfinite(variance) or variance <= float(epsilon):
            return 1.0
        scale = math.sqrt(variance + float(epsilon))
        if not math.isfinite(scale) or scale <= 0.0:
            return 1.0
        return float(scale)

    def update(self, values: Any) -> None:
        values_a = np.asarray(values, dtype=np.float64).reshape(-1)
        if not len(values_a) or not np.all(np.isfinite(values_a)):
            raise ValueError("running return statistics requires finite non-empty values")
        for value in values_a:
            self.count += 1
            delta = float(value) - self.mean
            self.mean += delta / float(self.count)
            delta_after = float(value) - self.mean
            self.m2 += delta * delta_after
        if (
            self.count < 0
            or not math.isfinite(self.mean)
            or not math.isfinite(self.m2)
            or self.m2 < 0.0
        ):
            raise FloatingPointError("running return statistics became non-finite")

    @classmethod
    def from_checkpoint(cls, count: Any, mean: Any, m2: Any) -> "RunningReturnStatistics":
        count_i = int(count)
        mean_f = float(mean)
        m2_f = float(m2)
        if (
            count_i < 0
            or not math.isfinite(mean_f)
            or not math.isfinite(m2_f)
            or m2_f < 0.0
        ):
            raise ValueError("checkpoint running return statistics are invalid")
        if count_i == 0 and (mean_f != 0.0 or m2_f != 0.0):
            raise ValueError("empty checkpoint running return statistics must be zero")
        return cls(count=count_i, mean=mean_f, m2=m2_f)


@dataclass(frozen=True)
class UpdateResult:
    policy_loss: float
    value_loss: float
    raw_value_loss: float
    critic_return_scale: float
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
    actor_gradient_norm: float
    actor_gradient_norm_min: float
    actor_gradient_norm_max: float
    critic_gradient_norm: float
    critic_gradient_norm_min: float
    critic_gradient_norm_max: float
    parameter_delta_norm: float
    optimizer_steps: int
    actor_optimizer_steps: int
    critic_optimizer_steps: int
    raw_advantages: np.ndarray
    raw_returns: np.ndarray
    bootstrap_value: float


@dataclass(frozen=True)
class TrainingCursor:
    completed_episodes: int
    current_train_origin: int
    current_cache_state: np.ndarray
    current_uav_positions: dict[str, tuple[float, float]]
    best_epoch: int
    best_epoch_return: float | None


def _project_path(root: Path, value: Any) -> Path:
    return cache_configuration.resolve_path(root, value)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one finite, structured training metric record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


def _critic_episode_metrics(
    old_values: list[float],
    raw_targets: list[float],
) -> tuple[float | None, float, float]:
    """Summarize collection-time critic values against production GAE targets."""

    values = np.asarray(old_values, dtype=np.float64)
    targets = np.asarray(raw_targets, dtype=np.float64)
    if values.shape != targets.shape or values.ndim != 1 or not len(values):
        raise ValueError("critic episode metric arrays must be aligned and non-empty")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(targets)):
        raise FloatingPointError("critic episode metrics contain NaN/Inf")
    target_variance = float(np.var(targets))
    value_std = float(np.std(values))
    target_std = float(np.std(targets))
    if not all(np.isfinite(value) for value in (target_variance, value_std, target_std)):
        raise FloatingPointError("critic episode metric summary is non-finite")
    explained_variance: float | None
    if target_variance <= 1.0e-12:
        explained_variance = None
    else:
        explained_variance = float(
            1.0 - float(np.var(targets - values)) / target_variance
        )
        if not np.isfinite(explained_variance):
            raise FloatingPointError("critic explained variance is non-finite")
    return explained_variance, value_std, target_std


def _append_critic_diagnostics(
    path: Path,
    *,
    seed: int,
    epoch: int,
    chunk_origin_start: int,
    optimizer_steps: int,
    rollout: SimpleRollout,
    update: UpdateResult,
) -> None:
    """Append v2 raw critic targets already computed by the PPO update."""

    transitions = rollout.transitions
    if not (
        len(transitions)
        == len(update.raw_advantages)
        == len(update.raw_returns)
    ):
        raise AssertionError("critic diagnostics length mismatch")
    for transition, raw_advantage, raw_return in zip(
        transitions,
        update.raw_advantages,
        update.raw_returns,
        strict=True,
    ):
        old_value = float(transition.old_value)
        target = float(raw_return)
        residual = old_value - target
        normalized_residual = residual / float(update.critic_return_scale)
        values = (old_value, float(raw_advantage), target, update.bootstrap_value, residual)
        if not all(np.isfinite(value) for value in values):
            raise FloatingPointError("critic diagnostics contain non-finite values")
        _append_jsonl(
            path,
            {
                "record_type": "critic_transition",
                "seed": int(seed),
                "episode": int(epoch),
                "epoch": int(epoch),
                "chunk_origin_start": int(chunk_origin_start),
                "optimizer_steps": int(optimizer_steps),
                "origin": int(transition.origin_index),
                "done": bool(transition.done),
                "old_value": old_value,
                "raw_advantage": float(raw_advantage),
                "raw_return": target,
                "value_target": target,
                "bootstrap_value": float(update.bootstrap_value),
                "value_residual": residual,
                "abs_value_residual": abs(residual),
                "critic_return_scale": float(update.critic_return_scale),
                "normalized_value_residual": normalized_residual,
            },
        )


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
    policy_schema = str(
        project_config.get_config_value(
            cfg,
            "algorithm.ppo.policy_schema",
            "adaptive_primitive_v1",
        )
    )
    horizon_policy = policy_schema in HORIZON_POLICY_SCHEMAS
    prediction_horizon = (
        int(project_config.get_config_value(cfg, "prediction.vehicle.horizon"))
        if horizon_policy
        else 1
    )
    observation_spec = interface.ConfigurationObservationSpec.from_cache_nodes(
        cache_nodes,
        service_catalog["service_ids"].tolist(),
        action_spec.uav_ids,
        [str(value) for value in first_origin["vehicle_ids"].tolist()],
        include_time_to_go=policy_schema in MODERN_POLICY_SCHEMAS,
        prediction_horizon=prediction_horizon,
        include_prediction_horizon_mask=horizon_policy,
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
        policy_schema=policy_schema,
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
        advantage_normalization_mode=str(
            get(
                cfg,
                "algorithm.ppo.advantage_normalization_mode",
                DEFAULT_ADVANTAGE_NORMALIZATION_MODE,
            )
        ),
        critic_value_scale_mode=str(
            get(
                cfg,
                "algorithm.ppo.critic_value_scale_mode",
                CRITIC_VALUE_SCALE_NONE,
            )
        ),
        critic_value_scale_epsilon=float(
            get(
                cfg,
                "algorithm.ppo.critic_value_scale_epsilon",
                DEFAULT_CRITIC_VALUE_SCALE_EPSILON,
            )
        ),
        critic_value_scale_warmup_samples=int(
            get(
                cfg,
                "algorithm.ppo.critic_value_scale_warmup_samples",
                DEFAULT_CRITIC_VALUE_SCALE_WARMUP_SAMPLES,
            )
        ),
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
    if values.advantage_normalization_mode not in {
        ADVANTAGE_NORMALIZATION_CENTERED_STD,
        ADVANTAGE_NORMALIZATION_STD_ONLY,
    }:
        raise ValueError(
            "advantage_normalization_mode 只支持 centered_std/std_only。"
        )
    if values.critic_value_scale_mode not in {
        CRITIC_VALUE_SCALE_NONE,
        CRITIC_VALUE_SCALE_RUNNING_RETURN_STD,
    }:
        raise ValueError(
            "critic_value_scale_mode 只支持 none/running_return_std。"
        )
    if (
        not math.isfinite(values.critic_value_scale_epsilon)
        or values.critic_value_scale_epsilon <= 0.0
    ):
        raise ValueError("critic_value_scale_epsilon 必须是正的有限数。")
    if values.critic_value_scale_warmup_samples < 0:
        raise ValueError("critic_value_scale_warmup_samples 必须为非负整数。")
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
        prediction_horizon=int(context.observation_spec.prediction_horizon),
        prediction_dt_s=float(
            project_config.get_config_value(context.cfg, "simulation.dt_s")
        ),
        prediction_terminal_origin=int(context.train_origin_end),
        expected_vehicle_ids=context.observation_spec.vehicle_ids,
        expected_service_ids=context.observation_spec.service_ids,
        include_prediction_horizon_mask=(
            context.observation_spec.horizon_mask_slice is not None
        ),
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
            policy_class: type[AdaptivePrimitiveActorCritic]
            if context.policy_schema == AdaptivePrimitiveActorCritic.architecture_id:
                policy_class = AdaptivePrimitiveActorCritic
            elif context.policy_schema == AdaptivePrimitiveV2ActorCritic.architecture_id:
                policy_class = AdaptivePrimitiveV2ActorCritic
            elif context.policy_schema == AdaptivePrimitiveGroupedV2ActorCritic.architecture_id:
                policy_class = AdaptivePrimitiveGroupedV2ActorCritic
            elif context.policy_schema == AdaptivePrimitiveHorizonGroupedV3ActorCritic.architecture_id:
                policy_class = AdaptivePrimitiveHorizonGroupedV3ActorCritic
            elif (
                context.policy_schema
                == AdaptivePrimitiveHorizonGroupedSeparateCriticV4ActorCritic.architecture_id
            ):
                policy_class = AdaptivePrimitiveHorizonGroupedSeparateCriticV4ActorCritic
            elif (
                context.policy_schema
                == AdaptivePrimitiveHorizonGroupedSeparateCriticOptimizationV5ActorCritic.architecture_id
            ):
                policy_class = AdaptivePrimitiveHorizonGroupedSeparateCriticOptimizationV5ActorCritic
            else:
                raise ValueError(
                    "unknown policy_primitive policy_schema="
                    f"{context.policy_schema}"
                )
            self.policy = policy_class(
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
        self.optimizer_steps = 0
        self.actor_optimizer_steps = 0
        self.critic_optimizer_steps = 0
        if self._uses_separate_optimization():
            if not isinstance(
                self.policy,
                AdaptivePrimitiveHorizonGroupedSeparateCriticOptimizationV5ActorCritic,
            ):
                raise TypeError("V5 separate optimization requires the V5 actor-critic policy.")
            actor_parameters = list(self.policy.actor_parameters())
            critic_parameters = list(self.policy.critic_parameters())
            actor_ids = {id(parameter) for parameter in actor_parameters}
            critic_ids = {id(parameter) for parameter in critic_parameters}
            if len(actor_ids) != len(actor_parameters) or len(critic_ids) != len(critic_parameters):
                raise RuntimeError("V5 actor/critic optimizer parameter groups contain duplicates.")
            if actor_ids & critic_ids:
                raise RuntimeError("V5 actor/critic optimizer parameter groups overlap.")
            self.actor_optimizer = torch.optim.Adam(
                actor_parameters,
                lr=self.hyperparameters.learning_rate,
            )
            self.critic_optimizer = torch.optim.Adam(
                critic_parameters,
                lr=self.hyperparameters.learning_rate,
            )
            self.optimizer = None
        else:
            self.optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=self.hyperparameters.learning_rate,
            )
            self.actor_optimizer = None
            self.critic_optimizer = None
        self.critic_return_statistics = RunningReturnStatistics()

    def _uses_separate_optimization(self) -> bool:
        return self.policy.architecture_id in V5_POLICY_SCHEMAS

    def _critic_return_scale(self) -> float:
        if self.hyperparameters.critic_value_scale_mode == CRITIC_VALUE_SCALE_NONE:
            return 1.0
        return self.critic_return_statistics.scale(
            self.hyperparameters.critic_value_scale_epsilon,
            self.hyperparameters.critic_value_scale_warmup_samples,
        )

    def _expected_checkpoint_version(self) -> int:
        if self.policy.architecture_id in V5_POLICY_SCHEMAS:
            return SEPARATE_OPTIMIZATION_CHECKPOINT_VERSION
        if (
            self.hyperparameters.critic_value_scale_mode
            == CRITIC_VALUE_SCALE_RUNNING_RETURN_STD
        ):
            return VALUE_NORMALIZED_CHECKPOINT_VERSION
        return (
            PRIMITIVE_CHECKPOINT_VERSION
            if self.context.configuration_control_mode == "policy_primitive"
            else CHECKPOINT_VERSION
        )

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
        critic_return_scale = self._critic_return_scale()
        if not np.isfinite(critic_return_scale) or critic_return_scale <= 0.0:
            raise FloatingPointError("PPO v2 critic return scale 非 finite。")
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
            advantage_std = advantages_t.std(unbiased=False) + 1.0e-8
            if (
                hyper.advantage_normalization_mode
                == ADVANTAGE_NORMALIZATION_CENTERED_STD
            ):
                advantages_t = (advantages_t - advantages_t.mean()) / advantage_std
            elif (
                hyper.advantage_normalization_mode
                == ADVANTAGE_NORMALIZATION_STD_ONLY
            ):
                # std_only divides by a positive scale and therefore preserves
                # every raw advantage sign, including exact zeros.
                advantages_t = advantages_t / advantage_std
            else:  # pragma: no cover - validated by hyperparameters_from_project
                raise ValueError(
                    "unsupported advantage_normalization_mode: "
                    f"{hyper.advantage_normalization_mode}"
                )

        before = [parameter.detach().clone() for parameter in self.policy.parameters()]
        policy_loss_value = 0.0
        value_loss_value = 0.0
        raw_value_loss_value = 0.0
        entropy_value = 0.0
        approximate_kl_value = 0.0
        clip_fraction_value = 0.0
        total_loss_value = 0.0
        policy_loss_values: list[float] = []
        value_loss_values: list[float] = []
        approximate_kl_values: list[float] = []
        gradient_norm_values: list[float] = []
        actor_gradient_norm_values: list[float] = []
        critic_gradient_norm_values: list[float] = []
        gradient_finite = True
        actor_parameters: list[Any] | None = None
        critic_parameters: list[Any] | None = None
        if self._uses_separate_optimization():
            if not isinstance(
                self.policy,
                AdaptivePrimitiveHorizonGroupedSeparateCriticOptimizationV5ActorCritic,
            ):
                raise TypeError("V5 separate optimization requires the V5 actor-critic policy.")
            actor_parameters = list(self.policy.actor_parameters())
            critic_parameters = list(self.policy.critic_parameters())
            if self.actor_optimizer is None or self.critic_optimizer is None:
                raise RuntimeError("V5 actor/critic optimizers are not initialized.")
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
                value_residual = (
                    evaluation.value.detach() - returns_t[indices]
                    if self._uses_separate_optimization()
                    else evaluation.value - returns_t[indices]
                )
                raw_value_loss = 0.5 * torch.mean(value_residual**2)
                normalized_value_residual = value_residual / critic_return_scale
                value_loss = 0.5 * torch.mean(normalized_value_residual**2)
                entropy = evaluation.entropy.mean()
                if self._uses_separate_optimization():
                    assert actor_parameters is not None
                    assert critic_parameters is not None
                    assert self.actor_optimizer is not None
                    assert self.critic_optimizer is not None
                    actor_loss = (
                        policy_loss - hyper.entropy_coefficient * entropy
                    )
                    if not all(
                        bool(torch.all(torch.isfinite(value)).item())
                        for value in (
                            ratio,
                            approximate_kl,
                            clip_fraction,
                            policy_loss,
                            entropy,
                            actor_loss,
                        )
                    ):
                        raise FloatingPointError("PPO v5 actor metrics/loss 非 finite。")

                    # Clear both disjoint groups before the actor backward so a
                    # stale critic gradient cannot be mistaken for actor output.
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    if any(parameter.grad is not None for parameter in critic_parameters):
                        raise RuntimeError("PPO v5 actor backward touched critic parameters.")
                    for parameter in actor_parameters:
                        if parameter.grad is not None and not bool(
                            torch.all(torch.isfinite(parameter.grad)).item()
                        ):
                            gradient_finite = False
                    if not gradient_finite:
                        raise FloatingPointError("PPO v5 actor gradient 非 finite。")
                    actor_gradient_norm = torch.nn.utils.clip_grad_norm_(
                        actor_parameters, hyper.max_grad_norm
                    )
                    actor_gradient_norm_value = float(actor_gradient_norm.detach().cpu())
                    if not np.isfinite(actor_gradient_norm_value):
                        raise FloatingPointError("PPO v5 actor gradient norm 非 finite。")
                    self.actor_optimizer.step()
                    self.actor_optimizer_steps += 1
                    self.optimizer_steps = self.actor_optimizer_steps

                    # Rebuild the critic graph after the actor step.  No graph
                    # is retained and the actor group is cleared before critic
                    # backward to enforce the reverse disjointness invariant.
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    critic_values = self.policy.critic_value(observations[indices])
                    critic_value_residual = critic_values - returns_t[indices]
                    raw_value_loss = 0.5 * torch.mean(critic_value_residual**2)
                    normalized_value_residual = critic_value_residual / critic_return_scale
                    value_loss = 0.5 * torch.mean(normalized_value_residual**2)
                    # V5 backpropagates the configured scaled loss.  The raw
                    # value loss remains a diagnostic-only metric.
                    critic_loss = value_loss
                    if not bool(torch.isfinite(raw_value_loss).item()) or not bool(
                        torch.isfinite(value_loss).item()
                    ) or not bool(torch.isfinite(critic_loss).item()):
                        raise FloatingPointError("PPO v5 critic loss 非 finite。")
                    critic_loss.backward()
                    if any(parameter.grad is not None for parameter in actor_parameters):
                        raise RuntimeError("PPO v5 critic backward touched actor parameters.")
                    for parameter in critic_parameters:
                        if parameter.grad is not None and not bool(
                            torch.all(torch.isfinite(parameter.grad)).item()
                        ):
                            gradient_finite = False
                    if not gradient_finite:
                        raise FloatingPointError("PPO v5 critic gradient 非 finite。")
                    critic_gradient_norm = torch.nn.utils.clip_grad_norm_(
                        critic_parameters, hyper.max_grad_norm
                    )
                    critic_gradient_norm_value = float(critic_gradient_norm.detach().cpu())
                    if not np.isfinite(critic_gradient_norm_value):
                        raise FloatingPointError("PPO v5 critic gradient norm 非 finite。")
                    self.critic_optimizer.step()
                    self.critic_optimizer_steps += 1
                    total_loss = actor_loss.detach() + critic_loss.detach()
                    if not bool(torch.isfinite(total_loss).item()):
                        raise FloatingPointError("PPO v5 monitor total loss 非 finite。")
                    gradient_norm_value = max(
                        actor_gradient_norm_value,
                        critic_gradient_norm_value,
                    )
                else:
                    total_loss = (
                        policy_loss
                        + hyper.value_loss_coefficient * value_loss
                        - hyper.entropy_coefficient * entropy
                    )
                    if not bool(torch.isfinite(raw_value_loss).item()) or not bool(
                        torch.isfinite(value_loss).item()
                    ) or not bool(torch.isfinite(total_loss).item()):
                        raise FloatingPointError("PPO v2 total loss 非 finite。")
                    if self.optimizer is None:
                        raise RuntimeError("PPO v2 optimizer is not initialized.")
                    self.optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    for parameter in self.policy.parameters():
                        if parameter.grad is not None and not bool(
                            torch.all(torch.isfinite(parameter.grad)).item()
                        ):
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
                    self.actor_optimizer_steps = self.optimizer_steps
                    self.critic_optimizer_steps = self.optimizer_steps
                    actor_gradient_norm_value = gradient_norm_value
                    critic_gradient_norm_value = gradient_norm_value
                policy_loss_value = float(policy_loss.detach().cpu())
                value_loss_value = float(value_loss.detach().cpu())
                raw_value_loss_value = float(raw_value_loss.detach().cpu())
                entropy_value = float(entropy.detach().cpu())
                approximate_kl_value = float(approximate_kl.detach().cpu())
                clip_fraction_value = float(clip_fraction.detach().cpu())
                total_loss_value = float(total_loss.detach().cpu())
                policy_loss_values.append(policy_loss_value)
                value_loss_values.append(value_loss_value)
                approximate_kl_values.append(approximate_kl_value)
                gradient_norm_values.append(gradient_norm_value)
                actor_gradient_norm_values.append(actor_gradient_norm_value)
                critic_gradient_norm_values.append(critic_gradient_norm_value)
        if self.hyperparameters.critic_value_scale_mode == CRITIC_VALUE_SCALE_RUNNING_RETURN_STD:
            self.critic_return_statistics.update(returns)
        delta_squared = 0.0
        for old_parameter, new_parameter in zip(before, self.policy.parameters()):
            delta_squared += float(torch.sum((new_parameter.detach() - old_parameter) ** 2).cpu())
        return UpdateResult(
            policy_loss=policy_loss_value,
            value_loss=value_loss_value,
            raw_value_loss=raw_value_loss_value,
            critic_return_scale=critic_return_scale,
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
            actor_gradient_norm=actor_gradient_norm_values[-1],
            actor_gradient_norm_min=float(min(actor_gradient_norm_values)),
            actor_gradient_norm_max=float(max(actor_gradient_norm_values)),
            critic_gradient_norm=critic_gradient_norm_values[-1],
            critic_gradient_norm_min=float(min(critic_gradient_norm_values)),
            critic_gradient_norm_max=float(max(critic_gradient_norm_values)),
            parameter_delta_norm=float(np.sqrt(delta_squared)),
            optimizer_steps=self.optimizer_steps,
            actor_optimizer_steps=self.actor_optimizer_steps,
            critic_optimizer_steps=self.critic_optimizer_steps,
            raw_advantages=advantages.copy(),
            raw_returns=returns.copy(),
            bootstrap_value=bootstrap_value,
        )

    def checkpoint_payload(
        self,
        completed_episodes: int,
        current_train_origin: int,
        current_cache_state: np.ndarray,
        current_uav_positions: dict[str, Any],
        config_hash: str,
        rollout_steps: int,
        best_epoch: int = 0,
        best_epoch_return: float | None = None,
    ) -> dict[str, Any]:
        payload = {
            "checkpoint_version": self._expected_checkpoint_version(),
            "method": PPO_V2_METHOD,
            "policy_schema": self.policy.architecture_id,
            "configuration_control_mode": self.context.configuration_control_mode,
            "advantage_normalization_mode": (
                self.hyperparameters.advantage_normalization_mode
            ),
            "critic_value_scale_mode": self.hyperparameters.critic_value_scale_mode,
            "critic_value_scale_epsilon": self.hyperparameters.critic_value_scale_epsilon,
            "critic_value_scale_warmup_samples": (
                self.hyperparameters.critic_value_scale_warmup_samples
            ),
            "critic_return_scale_count": int(self.critic_return_statistics.count),
            "critic_return_scale_mean": float(self.critic_return_statistics.mean),
            "critic_return_scale_m2": float(self.critic_return_statistics.m2),
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
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
        }
        if self._uses_separate_optimization():
            if self.actor_optimizer is None or self.critic_optimizer is None:
                raise RuntimeError("V5 actor/critic optimizers are not initialized.")
            payload.update(
                {
                    "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                    "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                    "actor_optimizer_steps": int(self.actor_optimizer_steps),
                    "critic_optimizer_steps": int(self.critic_optimizer_steps),
                }
            )
        else:
            if self.optimizer is None:
                raise RuntimeError("PPO optimizer is not initialized.")
            payload.update(
                {
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "optimizer_steps": int(self.optimizer_steps),
                }
            )
        if self.policy.architecture_id in MODERN_POLICY_SCHEMAS:
            payload["best_epoch"] = int(best_epoch)
            payload["best_epoch_return"] = (
                None if best_epoch_return is None else float(best_epoch_return)
            )
        return payload

    def save_checkpoint(
        self,
        run_dir: Path,
        completed_episodes: int,
        current_train_origin: int,
        current_cache_state: np.ndarray,
        current_uav_positions: dict[str, Any],
        config_hash: str,
        rollout_steps: int,
        *,
        filename: str = CHECKPOINT_FILENAME,
        best_epoch: int = 0,
        best_epoch_return: float | None = None,
    ) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = run_dir / filename
        torch.save(
            self.checkpoint_payload(
                completed_episodes,
                current_train_origin,
                current_cache_state,
                current_uav_positions,
                config_hash,
                rollout_steps,
                best_epoch,
                best_epoch_return,
            ),
            checkpoint,
        )
        return checkpoint

    def load_checkpoint(self, checkpoint: Path, config_hash: str) -> TrainingCursor:
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        expected_scale_mode = self.hyperparameters.critic_value_scale_mode
        expected_advantage_mode = self.hyperparameters.advantage_normalization_mode
        checkpoint_advantage_mode = str(
            payload.get(
                "advantage_normalization_mode",
                DEFAULT_ADVANTAGE_NORMALIZATION_MODE,
            )
        )
        if checkpoint_advantage_mode != expected_advantage_mode:
            raise ValueError(
                "PPO checkpoint advantage_normalization_mode mismatch; refusing "
                "silent cross-mode resume: "
                f"checkpoint={checkpoint_advantage_mode}, "
                f"expected={expected_advantage_mode}."
            )
        checkpoint_scale_mode = str(
            payload.get("critic_value_scale_mode", CRITIC_VALUE_SCALE_NONE)
        )
        if checkpoint_scale_mode != expected_scale_mode:
            raise ValueError(
                "PPO checkpoint critic_value_scale_mode mismatch; refusing silent "
                f"cross-mode resume: checkpoint={checkpoint_scale_mode}, "
                f"expected={expected_scale_mode}."
            )
        expected_version = self._expected_checkpoint_version()
        if int(payload.get("checkpoint_version", -1)) != expected_version:
            raise ValueError("PPO v2 checkpoint version 不匹配。")
        restored_statistics = RunningReturnStatistics()
        if expected_scale_mode == CRITIC_VALUE_SCALE_RUNNING_RETURN_STD:
            checkpoint_epsilon = float(
                payload.get("critic_value_scale_epsilon", float("nan"))
            )
            checkpoint_warmup = int(
                payload.get("critic_value_scale_warmup_samples", -1)
            )
            if not math.isclose(
                checkpoint_epsilon,
                self.hyperparameters.critic_value_scale_epsilon,
                rel_tol=0.0,
                abs_tol=0.0,
            ) or checkpoint_warmup != self.hyperparameters.critic_value_scale_warmup_samples:
                raise ValueError("PPO checkpoint critic scale configuration mismatch.")
            restored_statistics = RunningReturnStatistics.from_checkpoint(
                payload.get("critic_return_scale_count", -1),
                payload.get("critic_return_scale_mean", float("nan")),
                payload.get("critic_return_scale_m2", float("nan")),
            )
        checkpoint_observation_dim = int(payload.get("observation_dim", -1))
        expected_observation_dim = int(self.context.observation_spec.observation_dim)
        if checkpoint_observation_dim != expected_observation_dim:
            raise ValueError(
                "PPO checkpoint observation_dim incompatible; refusing silent schema resume: "
                f"checkpoint={checkpoint_observation_dim}, expected={expected_observation_dim}."
            )
        expected_schema = self.policy.architecture_id
        checkpoint_schema = payload.get("policy_schema")
        if checkpoint_schema is None and expected_schema == BernoulliActorCritic.architecture_id:
            checkpoint_schema = BernoulliActorCritic.architecture_id
        if checkpoint_schema != expected_schema:
            raise ValueError(
                "PPO checkpoint policy schema incompatible: "
                f"checkpoint={checkpoint_schema}, expected={expected_schema}."
            )
        if payload.get("configuration_control_mode", "legacy_exact") != (
            self.context.configuration_control_mode
        ):
            raise ValueError("PPO checkpoint configuration_control_mode mismatch.")
        if payload.get("method") != PPO_V2_METHOD or payload.get("config_sha256") != config_hash:
            raise ValueError("PPO v2 checkpoint method/config hash mismatch.")
        if int(payload.get("action_dim")) != int(self.context.action_spec.action_dim):
            raise ValueError("PPO v2 checkpoint action_dim 不匹配。")
        self.policy.load_state_dict(payload["policy_state_dict"], strict=True)
        if self._uses_separate_optimization():
            if self.actor_optimizer is None or self.critic_optimizer is None:
                raise RuntimeError("V5 actor/critic optimizers are not initialized.")
            if (
                "actor_optimizer_state_dict" not in payload
                or "critic_optimizer_state_dict" not in payload
            ):
                raise ValueError(
                    "V5 checkpoint must contain separate actor/critic optimizer state."
                )
            self.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
            self.actor_optimizer_steps = int(payload.get("actor_optimizer_steps", 0))
            self.critic_optimizer_steps = int(payload.get("critic_optimizer_steps", 0))
            if self.actor_optimizer_steps < 0 or self.critic_optimizer_steps < 0:
                raise ValueError("V5 checkpoint optimizer steps must be non-negative.")
            self.optimizer_steps = self.actor_optimizer_steps
        else:
            if self.optimizer is None:
                raise RuntimeError("PPO optimizer is not initialized.")
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            self.optimizer_steps = int(payload.get("optimizer_steps", 0))
            self.actor_optimizer_steps = self.optimizer_steps
            self.critic_optimizer_steps = self.optimizer_steps
        self.critic_return_statistics = restored_statistics
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
        best_epoch = int(payload.get("best_epoch", 0))
        raw_best_epoch_return = payload.get("best_epoch_return")
        best_epoch_return = (
            None if raw_best_epoch_return is None else float(raw_best_epoch_return)
        )
        if best_epoch_return is not None and not np.isfinite(best_epoch_return):
            raise ValueError("checkpoint best_epoch_return must be finite when present")
        return TrainingCursor(
            completed_episodes=int(payload["completed_episodes"]),
            current_train_origin=current_train_origin,
            current_cache_state=current_cache_state,
            current_uav_positions=current_uav_positions,
            best_epoch=best_epoch,
            best_epoch_return=best_epoch_return,
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
    verbose: bool = False,
) -> None:
    _require_torch()
    target_epochs = int(episodes)
    if target_epochs <= 0:
        raise ValueError("epochs must be greater than zero")
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
    critic_diagnostics_path = run_path / "critic_diagnostics.jsonl"
    slot_metrics_path = run_path / SLOT_METRICS_FILENAME
    best_epoch = 0
    best_epoch_return: float | None = None
    if resume:
        if not checkpoint.exists():
            raise FileNotFoundError(f"resume checkpoint 不存在：{checkpoint}")
        cursor = trainer.load_checkpoint(checkpoint, config_hash)
        completed_episodes = cursor.completed_episodes
        current_train_origin = cursor.current_train_origin
        current_cache_state = cursor.current_cache_state
        current_uav_positions = cursor.current_uav_positions
        best_epoch = cursor.best_epoch
        best_epoch_return = cursor.best_epoch_return
        if completed_episodes > target_epochs:
            raise ValueError("checkpoint completed epochs exceed target epochs")

    origins_per_epoch = context.train_origin_end - context.train_origin_start + 1
    if resume:
        displayed_origin = (
            0
            if current_train_origin > context.train_origin_end
            else current_train_origin - context.train_origin_start
        )
        print(
            "[PPO] Resume\n"
            f"Completed Epochs: {completed_episodes}\n"
            f"Next Epoch: {completed_episodes + 1}\n"
            f"Current Origin: {displayed_origin}/{origins_per_epoch}\n"
            f"Target Epochs: {target_epochs}",
            flush=True,
        )
    else:
        print(
            "[PPO] Training start\n"
            f"Target Epochs: {target_epochs}\n"
            f"Origins per Epoch: {origins_per_epoch}\n"
            f"Rollout Steps: {int(rollout_steps)}",
            flush=True,
        )

    episode_started_at = time.perf_counter()
    episode_optimizer_start = trainer.optimizer_steps
    episode_actor_optimizer_start = trainer.actor_optimizer_steps
    episode_critic_optimizer_start = trainer.critic_optimizer_steps
    episode_rewards: list[float] = []
    episode_g_values: list[float] = []
    episode_cache_requests = 0
    episode_actual_cache_updates = 0
    episode_uav_requests = 0
    episode_actual_uav_movements = 0
    episode_cache_cost = 0.0
    episode_movement_cost = 0.0
    episode_critic_values: list[float] = []
    episode_critic_targets: list[float] = []
    episode_update_count = 0
    episode_policy_loss_mins: list[float] = []
    episode_policy_loss_maxs: list[float] = []
    episode_value_loss_mins: list[float] = []
    episode_value_loss_maxs: list[float] = []
    episode_kl_mins: list[float] = []
    episode_kl_maxs: list[float] = []
    episode_gradient_norm_mins: list[float] = []
    episode_gradient_norm_maxs: list[float] = []
    episode_actor_gradient_norm_mins: list[float] = []
    episode_actor_gradient_norm_maxs: list[float] = []
    episode_critic_gradient_norm_mins: list[float] = []
    episode_critic_gradient_norm_maxs: list[float] = []
    episode_entropies: list[float] = []
    episode_approximate_kls: list[float] = []

    while completed_episodes < target_epochs:
        if current_train_origin > context.train_origin_end:
            current_train_origin = context.train_origin_start
            current_cache_state = np.asarray(context.initial_cache, dtype=np.int8).copy()
            current_uav_positions = _copy_uav_positions(context.initial_uav)
            episode_started_at = time.perf_counter()
            episode_optimizer_start = trainer.optimizer_steps
            episode_actor_optimizer_start = trainer.actor_optimizer_steps
            episode_critic_optimizer_start = trainer.critic_optimizer_steps
            episode_rewards = []
            episode_g_values = []
            episode_cache_requests = 0
            episode_actual_cache_updates = 0
            episode_uav_requests = 0
            episode_actual_uav_movements = 0
            episode_cache_cost = 0.0
            episode_movement_cost = 0.0
            episode_critic_values = []
            episode_critic_targets = []
            episode_update_count = 0
            episode_policy_loss_mins = []
            episode_policy_loss_maxs = []
            episode_value_loss_mins = []
            episode_value_loss_maxs = []
            episode_kl_mins = []
            episode_kl_maxs = []
            episode_gradient_norm_mins = []
            episode_gradient_norm_maxs = []
            episode_actor_gradient_norm_mins = []
            episode_actor_gradient_norm_maxs = []
            episode_critic_gradient_norm_mins = []
            episode_critic_gradient_norm_maxs = []
            episode_entropies = []
            episode_approximate_kls = []
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
            best_epoch=best_epoch,
            best_epoch_return=best_epoch_return,
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
        episode_critic_values.extend(
            np.asarray(rollout.old_values, dtype=np.float64).tolist()
        )
        episode_critic_targets.extend(
            np.asarray(update.raw_returns, dtype=np.float64).tolist()
        )
        episode_update_count += 1
        episode_policy_loss_mins.append(update.policy_loss_min)
        episode_policy_loss_maxs.append(update.policy_loss_max)
        episode_value_loss_mins.append(update.value_loss_min)
        episode_value_loss_maxs.append(update.value_loss_max)
        episode_kl_mins.append(update.approximate_kl_min)
        episode_kl_maxs.append(update.approximate_kl_max)
        episode_gradient_norm_mins.append(update.gradient_norm_min)
        episode_gradient_norm_maxs.append(update.gradient_norm_max)
        episode_actor_gradient_norm_mins.append(update.actor_gradient_norm_min)
        episode_actor_gradient_norm_maxs.append(update.actor_gradient_norm_max)
        episode_critic_gradient_norm_mins.append(update.critic_gradient_norm_min)
        episode_critic_gradient_norm_maxs.append(update.critic_gradient_norm_max)
        episode_entropies.append(update.entropy)
        episode_approximate_kls.append(update.approximate_kl)
        if context.policy_schema in MODERN_POLICY_SCHEMAS:
            _append_critic_diagnostics(
                critic_diagnostics_path,
                seed=seed,
                epoch=episode_number,
                chunk_origin_start=chunk_origin_start,
                optimizer_steps=update.optimizer_steps,
                rollout=rollout,
                update=update,
            )
        _append_jsonl(
            update_metrics_path,
            {
                "record_type": "update",
                "seed": seed,
                "episode": episode_number,
                "epoch": episode_number,
                "completed_epochs": int(completed_episodes),
                "origin_start": int(chunk_origin_start),
                "origin_end": int(next_train_origin - 1),
                "num_transitions": transition_count,
                "terminal": bool(terminal),
                "advantage_normalization_mode": (
                    trainer.hyperparameters.advantage_normalization_mode
                ),
                "optimizer_steps": int(update.optimizer_steps),
                "actor_optimizer_steps": int(update.actor_optimizer_steps),
                "critic_optimizer_steps": int(update.critic_optimizer_steps),
                "policy_loss": float(update.policy_loss),
                "policy_loss_min": float(update.policy_loss_min),
                "policy_loss_max": float(update.policy_loss_max),
                "value_loss": float(update.value_loss),
                "normalized_value_loss": float(update.value_loss),
                "raw_value_loss": float(update.raw_value_loss),
                "critic_return_scale": float(update.critic_return_scale),
                "critic_return_scale_count": int(trainer.critic_return_statistics.count),
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
                "actor_gradient_norm": float(update.actor_gradient_norm),
                "actor_gradient_norm_min": float(update.actor_gradient_norm_min),
                "actor_gradient_norm_max": float(update.actor_gradient_norm_max),
                "critic_gradient_norm": float(update.critic_gradient_norm),
                "critic_gradient_norm_min": float(update.critic_gradient_norm_min),
                "critic_gradient_norm_max": float(update.critic_gradient_norm_max),
                "gradient_finite": bool(update.gradient_finite),
                "returns_finite": bool(update.returns_finite),
                "advantages_finite": bool(update.advantages_finite),
                "total_loss": float(update.total_loss),
                "total_loss_finite": bool(np.isfinite(update.total_loss)),
                "checkpoint": str(checkpoint_path),
            },
        )
        if verbose:
            print(
                f"[PPO] optimizer_step={update.optimizer_steps} "
                f"epoch={episode_number} completed_epochs={completed_episodes} "
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
                f"gradient_norm={update.gradient_norm:.6f} "
                f"parameter_delta={update.parameter_delta_norm:.6e} "
                f"checkpoint={checkpoint_path} next_origin={next_train_origin}",
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
            epoch_runtime_seconds = float(time.perf_counter() - episode_started_at)
            epoch_return = float(episode_reward_array.sum())
            (
                critic_explained_variance,
                critic_value_std,
                critic_target_std,
            ) = _critic_episode_metrics(
                episode_critic_values,
                episode_critic_targets,
            )
            cache_change_rate = float(episode_actual_cache_updates) / float(
                len(episode_reward_array) * cache_count
            )
            uav_move_rate = float(episode_actual_uav_movements) / float(
                len(episode_reward_array) * len(context.action_spec.uav_ids)
            )
            epoch_entropy = float(np.mean(episode_entropies))
            epoch_approximate_kl = float(np.mean(episode_approximate_kls))
            if (
                context.policy_schema in MODERN_POLICY_SCHEMAS
                and (best_epoch_return is None or epoch_return > best_epoch_return)
            ):
                best_epoch = int(episode_number)
                best_epoch_return = epoch_return
                trainer.save_checkpoint(
                    run_path,
                    completed_episodes,
                    next_train_origin,
                    rollout.next_cache_state,
                    rollout.next_uav_positions,
                    config_hash,
                    int(rollout_steps),
                    filename=BEST_TRAIN_RETURN_CHECKPOINT_FILENAME,
                    best_epoch=best_epoch,
                    best_epoch_return=best_epoch_return,
                )
                checkpoint_path = trainer.save_checkpoint(
                    run_path,
                    completed_episodes,
                    next_train_origin,
                    rollout.next_cache_state,
                    rollout.next_uav_positions,
                    config_hash,
                    int(rollout_steps),
                    best_epoch=best_epoch,
                    best_epoch_return=best_epoch_return,
                )
            _append_jsonl(
                episode_metrics_path,
                {
                    "record_type": "episode",
                    "seed": seed,
                    "episode": int(episode_number),
                    "epoch": int(episode_number),
                    "completed_episodes": int(completed_episodes),
                    "completed_epochs": int(completed_episodes),
                    "num_transitions": int(len(episode_reward_array)),
                    "episode_return": epoch_return,
                    "mean_reward": float(episode_reward_array.mean()),
                    "mean_G": float(episode_g_array.mean()),
                    "reward_std": float(episode_reward_array.std()),
                    "critic_explained_variance": critic_explained_variance,
                    "critic_value_std": critic_value_std,
                    "critic_target_std": critic_target_std,
                    "cache_request_count": int(episode_cache_requests),
                    "actual_cache_update_count": int(episode_actual_cache_updates),
                    "uav_request_count": int(episode_uav_requests),
                    "actual_uav_movement_count": int(episode_actual_uav_movements),
                    "cache_cost": float(episode_cache_cost),
                    "movement_cost": float(episode_movement_cost),
                    "ppo_updates": int(episode_update_count),
                    "optimizer_steps": int(update.optimizer_steps),
                    "episode_optimizer_steps": int(update.optimizer_steps - episode_optimizer_start),
                    "actor_optimizer_steps": int(update.actor_optimizer_steps),
                    "critic_optimizer_steps": int(update.critic_optimizer_steps),
                    "episode_actor_optimizer_steps": int(
                        update.actor_optimizer_steps - episode_actor_optimizer_start
                    ),
                    "episode_critic_optimizer_steps": int(
                        update.critic_optimizer_steps - episode_critic_optimizer_start
                    ),
                    "best_epoch": int(best_epoch),
                    "best_epoch_return": best_epoch_return,
                    "entropy_mean": epoch_entropy,
                    "approximate_kl_mean": epoch_approximate_kl,
                    "cache_actual_change_rate": cache_change_rate,
                    "uav_actual_movement_rate": uav_move_rate,
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
                    "actor_gradient_norm_range": {
                        "min": float(min(episode_actor_gradient_norm_mins)),
                        "max": float(max(episode_actor_gradient_norm_maxs)),
                    },
                    "critic_gradient_norm_range": {
                        "min": float(min(episode_critic_gradient_norm_mins)),
                        "max": float(max(episode_critic_gradient_norm_maxs)),
                    },
                    "finite_checks": episode_finite,
                    "checkpoint": str(checkpoint_path),
                    "wall_clock_runtime_seconds": epoch_runtime_seconds,
                },
            )
            append_training_slot_metrics(
                slot_metrics_path,
                epoch=episode_number,
                rewards=episode_reward_array.tolist(),
                episode_return=epoch_return,
            )
            try:
                plot_path = update_training_slot_reward_plot(run_path)
            except Exception as exc:  # plotting must not interrupt PPO training
                print(f"[PLOT] Warning: {exc}", flush=True)
            else:
                print(f"[PLOT] Updated: {plot_path}", flush=True)
            print(
                f"[PPO] Epoch {completed_episodes}/{target_epochs} | "
                f"Return={epoch_return:.2f} | "
                f"CacheChange={cache_change_rate:.1%} | "
                f"UAVMove={uav_move_rate:.1%} | "
                f"Entropy={epoch_entropy:.2f} | "
                f"KL={epoch_approximate_kl:.4f} | "
                f"Time={epoch_runtime_seconds:.1f}s",
                flush=True,
            )
        current_train_origin = next_train_origin
        current_cache_state = np.asarray(rollout.next_cache_state, dtype=np.int8).copy()
        current_uav_positions = _copy_uav_positions(rollout.next_uav_positions)

    final_plot_path = run_path / "plots" / "training_slot_reward.png"
    if completed_episodes >= target_epochs:
        if final_plot_path.exists():
            print(f"[PLOT] Final: {final_plot_path}", flush=True)
        else:
            print(
                f"[PLOT] Warning: final plot was not created: {final_plot_path}",
                flush=True,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PPO v2 Cache/UAV trainer (1 Epoch = one complete train-origin pass)"
    )
    parser.add_argument("--config", default="system_config.toml")
    parser.add_argument(
        "--epochs",
        type=int,
        help="total target Epochs; 1 Epoch = train origins 0..239",
    )
    parser.add_argument("--episodes", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=DEFAULT_ROLLOUT_STEPS,
        help=f"consecutive train origins per PPO optimizer step (default: {DEFAULT_ROLLOUT_STEPS})",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show detailed metrics for each rollout chunk",
    )
    return parser


def _target_epochs_from_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> int:
    epochs = args.epochs
    legacy_episodes = args.episodes
    if epochs is None and legacy_episodes is None:
        parser.error("--epochs EPOCHS is required")
    if (
        epochs is not None
        and legacy_episodes is not None
        and int(epochs) != int(legacy_episodes)
    ):
        parser.error("--epochs and legacy --episodes have conflicting values")
    return int(epochs if epochs is not None else legacy_episodes)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    target_epochs = _target_epochs_from_args(parser, args)
    run_training(
        args.config,
        target_epochs,
        args.run_dir,
        rollout_steps=args.rollout_steps,
        resume=bool(args.resume),
        verbose=bool(args.verbose),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
