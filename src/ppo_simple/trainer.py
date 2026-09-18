"""Minimal PPO v2 trainer for adaptive Cache/UAV request actions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - CLI help/import-only environments
    torch = None  # type: ignore[assignment]

from common import project_config
from decision import cache_configuration, slot_environment, uav_configuration
from scenario import task_scenario

from . import interface
from .interface import _copy_uav_positions
from .no_prediction import NO_PREDICTION_POLICY_SCHEMA
from .plotting import (
    SLOT_METRICS_FILENAME,
    TRAINING_EPOCH_RETURN_PLOT_FILENAME,
    append_training_slot_metrics,
    update_training_epoch_return_plot,
)
from .reporting import (
    RUN_MANIFEST_FILENAME,
    RUN_SUMMARY_FILENAME,
    ensure_fresh_run_dir_is_clean,
    update_run_summary,
    write_run_manifest,
)
from .policy import (
    AdaptiveCacheOperationGroupV7PopArtActorCritic,
    AdaptiveCacheOperationGroupReplicaAwareV8PopArtActorCritic,
    AdaptiveCacheOperationGroupReplicaAwareV8NoPredictionPopArtActorCritic,
    AdaptiveJointCacheReplicaAwareUAVOperationV9PopArtActorCritic,
    AdaptiveJointCacheReplicaAwareUAVOperationV10BranchObservableActorCritic,
    AdaptiveJointCacheReplicaAwareUAVOperationV10BranchwiseActorCritic,
    AdaptiveJointCacheReplicaAwareUAVOperationV10CDecoupledEncoderActorCritic,
    AdaptiveJointCacheReplicaAwareUAVOperationV10DBranchSpecificKLActorCritic,
    AdaptiveJointCacheReplicaAwareUAVOperationRecoveryR1ActorCritic,
    AdaptiveJointCacheReplicaAwareUAVOperationRecoveryR2ActorCritic,
    AdaptiveJointCacheReplicaAwareUAVOperationV11ACacheNodeCFActorCritic,
    AdaptivePrimitiveActorCritic,
    AdaptivePrimitiveHorizonGroupedSeparateCriticOptimizationV5ActorCritic,
    AdaptivePrimitiveHorizonGroupedSeparateCriticPopArtV6ActorCritic,
    BernoulliActorCritic,
)
from .primitive_control import PrimitiveTensorSpec, _intermediate_observation
from .rollout import (
    SimpleRollout,
    _check_request_contract,
    _check_reward,
    collect_rollout,
    compute_gae,
)


PPO_V2_METHOD = "ppo_simple_v2"
CHECKPOINT_FILENAME = "latest_checkpoint.pt"
BEST_TRAIN_RETURN_CHECKPOINT_FILENAME = "best_train_return_checkpoint.pt"
CHECKPOINT_VERSION = 2
PRIMITIVE_CHECKPOINT_VERSION = 3
VALUE_NORMALIZED_CHECKPOINT_VERSION = 4
POPART_CHECKPOINT_VERSION = 6
V7_CHECKPOINT_VERSION = 7
V8_CHECKPOINT_VERSION = 8
V8_NO_PREDICTION_CHECKPOINT_VERSION = 9
V9_JOINT_CHECKPOINT_VERSION = 10
V10_BRANCHWISE_CHECKPOINT_VERSION = 11
V10C_DECOUPLED_CHECKPOINT_VERSION = 12
V10D_BRANCH_SPECIFIC_KL_CHECKPOINT_VERSION = 13
V11A_CACHE_NODE_CF_CHECKPOINT_VERSION = 16
KL_GATE_MODE_JOINT = "joint"
KL_GATE_MODE_BRANCH_SPECIFIC = "branch_specific"
CACHE_KL_CONTROL_EPOCH = "epoch"
CACHE_KL_CONTROL_MINIBATCH_EARLY_STOP = "minibatch_early_stop"
DEFAULT_CACHE_KL_CONTROL = CACHE_KL_CONTROL_EPOCH
DEFAULT_ROLLOUT_STEPS = 16
PLOT_UPDATE_INTERVAL_EPOCHS = 10
ADVANTAGE_NORMALIZATION_CENTERED_STD = "centered_std"
ADVANTAGE_NORMALIZATION_STD_ONLY = "std_only"
DEFAULT_ADVANTAGE_NORMALIZATION_MODE = ADVANTAGE_NORMALIZATION_CENTERED_STD
CRITIC_VALUE_SCALE_NONE = "none"
CRITIC_VALUE_SCALE_RUNNING_RETURN_STD = "running_return_std"
CRITIC_VALUE_SCALE_POPART_NORMALIZED = "popart_normalized"
DEFAULT_CRITIC_VALUE_SCALE_EPSILON = 1.0e-8
DEFAULT_CRITIC_VALUE_SCALE_WARMUP_SAMPLES = 64
# V6 is the sole production schema.  Its V3/V4/V5 base classes remain in
# policy.py because their module layout is part of the formal checkpoint.
V6_POLICY_SCHEMAS = frozenset(
    {AdaptivePrimitiveHorizonGroupedSeparateCriticPopArtV6ActorCritic.architecture_id}
)
V7_POLICY_SCHEMAS = frozenset(
    {AdaptiveCacheOperationGroupV7PopArtActorCritic.architecture_id}
)
V8_POLICY_SCHEMAS = frozenset(
    {AdaptiveCacheOperationGroupReplicaAwareV8PopArtActorCritic.architecture_id}
)
NO_PREDICTION_POLICY_SCHEMAS = frozenset({NO_PREDICTION_POLICY_SCHEMA})
JOINT_V9_POLICY_SCHEMAS = frozenset(
    {AdaptiveJointCacheReplicaAwareUAVOperationV9PopArtActorCritic.architecture_id}
)
JOINT_V10_POLICY_SCHEMAS = frozenset(
    {AdaptiveJointCacheReplicaAwareUAVOperationV10BranchObservableActorCritic.architecture_id}
)
JOINT_V10C_DECOUPLED_POLICY_SCHEMAS = frozenset(
    {AdaptiveJointCacheReplicaAwareUAVOperationV10CDecoupledEncoderActorCritic.architecture_id}
)
JOINT_V10D_BRANCH_SPECIFIC_KL_POLICY_SCHEMAS = frozenset(
    {AdaptiveJointCacheReplicaAwareUAVOperationV10DBranchSpecificKLActorCritic.architecture_id}
)
JOINT_RECOVERY_R1_POLICY_SCHEMAS = frozenset(
    {AdaptiveJointCacheReplicaAwareUAVOperationRecoveryR1ActorCritic.architecture_id}
)
JOINT_RECOVERY_R2_POLICY_SCHEMAS = frozenset(
    {AdaptiveJointCacheReplicaAwareUAVOperationRecoveryR2ActorCritic.architecture_id}
)
JOINT_V11A_CACHE_NODE_CF_POLICY_SCHEMAS = frozenset(
    {AdaptiveJointCacheReplicaAwareUAVOperationV11ACacheNodeCFActorCritic.architecture_id}
)
JOINT_V10D_BRANCH_SPECIFIC_KL_POLICY_SCHEMAS = (
    JOINT_V10D_BRANCH_SPECIFIC_KL_POLICY_SCHEMAS
    | JOINT_RECOVERY_R1_POLICY_SCHEMAS
    | JOINT_RECOVERY_R2_POLICY_SCHEMAS
    | JOINT_V11A_CACHE_NODE_CF_POLICY_SCHEMAS
)
JOINT_V10_BRANCHWISE_POLICY_SCHEMAS = frozenset(
    {AdaptiveJointCacheReplicaAwareUAVOperationV10BranchwiseActorCritic.architecture_id}
) | JOINT_V10C_DECOUPLED_POLICY_SCHEMAS | JOINT_V10D_BRANCH_SPECIFIC_KL_POLICY_SCHEMAS
JOINT_CACHE_UAV_POLICY_SCHEMAS = (
    JOINT_V9_POLICY_SCHEMAS
    | JOINT_V10_POLICY_SCHEMAS
    | JOINT_V10_BRANCHWISE_POLICY_SCHEMAS
)
CACHE_OPERATION_POLICY_SCHEMAS = (
    V7_POLICY_SCHEMAS | V8_POLICY_SCHEMAS | NO_PREDICTION_POLICY_SCHEMAS
)
POPART_POLICY_SCHEMAS = (
    V6_POLICY_SCHEMAS | CACHE_OPERATION_POLICY_SCHEMAS | JOINT_CACHE_UAV_POLICY_SCHEMAS
)
SEPARATE_OPTIMIZATION_POLICY_SCHEMAS = POPART_POLICY_SCHEMAS
HORIZON_POLICY_SCHEMAS = POPART_POLICY_SCHEMAS
MODERN_POLICY_SCHEMAS = POPART_POLICY_SCHEMAS

REALIZATION_ARTIFACT_FILENAMES = {
    "trajectory": "trajectory.csv",
    "vehicle_roles": "vehicle_roles.json",
    "tasks": "tasks.csv",
    "popularity": "popularity.csv",
    "popularity_history": "popularity_history.csv",
    "vehicle_predictions": "vehicle_predictions.csv",
    "popularity_predictions": "popularity_predictions.csv",
    "prediction_scenarios": "prediction_scenarios.npz",
    "tv_local_cache": "tv_local_cache.json",
}
# These files are emitted by the existing single-episode generator alongside
# the nine artifacts consumed by the trainer.  Keeping them in the completion
# contract prevents a lazy run from treating a partial generator output as
# reusable.
GENERATOR_EXTRA_ARTIFACT_FILENAMES = (
    "sumo_demand.rou.xml",
    "sumo_demand.meta.json",
)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PPO v2 training requires PyTorch.")


class CacheNodeCounterfactualCritic(
    torch.nn.Module if torch is not None else object  # type: ignore[misc]
):
    """Independent centralized action-conditioned critic for one Cache node.

    Q_cache conditions on the raw transition state, a deterministic one-hot
    target-node identity, and the complete deterministic joint action
    encoding.  The encoding contains every Cache node's operation/primitive
    feature and every UAV's operation/primitive feature, so counterfactuals
    replace only the target node slice while holding all other actions fixed.
    The network owns no Actor or global-critic parameter tensor.
    """

    def __init__(
        self,
        observation_dim: int,
        num_cache_nodes: int,
        joint_action_dim: int,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        if torch is None:  # pragma: no cover - guarded by trainer construction
            raise ImportError("V11-A Cache critic requires PyTorch.")
        super().__init__()
        if int(observation_dim) <= 0 or int(num_cache_nodes) <= 0 or int(joint_action_dim) <= 0:
            raise ValueError("V11-A Cache critic dimensions must be positive")
        layers: list[Any] = []
        last_dim = int(observation_dim) + int(num_cache_nodes) + int(joint_action_dim)
        for hidden_size in hidden_sizes:
            width = int(hidden_size)
            if width <= 0:
                raise ValueError("V11-A Cache critic hidden sizes must be positive")
            layers.extend([torch.nn.Linear(last_dim, width), torch.nn.Tanh()])
            last_dim = width
        layers.append(torch.nn.Linear(last_dim, 1))
        self.network = torch.nn.Sequential(*layers)
        self.observation_dim = int(observation_dim)
        self.num_cache_nodes = int(num_cache_nodes)
        self.joint_action_dim = int(joint_action_dim)

    def forward(self, observations: Any, node_indices: Any, joint_action_encoding: Any) -> Any:
        device = next(self.parameters()).device
        observations_t = torch.as_tensor(
            observations, dtype=torch.float32, device=device
        )
        node_indices_t = torch.as_tensor(node_indices, dtype=torch.long, device=device)
        joint_action_encoding_t = torch.as_tensor(
            joint_action_encoding, dtype=torch.float32, device=device
        )
        if observations_t.ndim == 1:
            observations_t = observations_t.unsqueeze(0)
        if node_indices_t.ndim == 0:
            node_indices_t = node_indices_t.reshape(1)
        else:
            node_indices_t = node_indices_t.reshape(-1)
        if joint_action_encoding_t.ndim == 1:
            joint_action_encoding_t = joint_action_encoding_t.unsqueeze(0)
        if (
            observations_t.ndim != 2
            or int(observations_t.shape[1]) != self.observation_dim
            or joint_action_encoding_t.ndim != 2
            or tuple(joint_action_encoding_t.shape[1:]) != (self.joint_action_dim,)
            or int(observations_t.shape[0]) != int(node_indices_t.shape[0])
            or int(joint_action_encoding_t.shape[0]) != int(node_indices_t.shape[0])
        ):
            raise ValueError("V11-A Cache critic input shape mismatch")
        if not bool(torch.all(torch.isfinite(observations_t)).item()) or not bool(
            torch.all(torch.isfinite(joint_action_encoding_t)).item()
        ):
            raise ValueError("V11-A Cache critic input contains NaN/Inf")
        if bool(torch.any(node_indices_t < 0).item()) or bool(
            torch.any(node_indices_t >= self.num_cache_nodes).item()
        ):
            raise ValueError("V11-A Cache critic node index out of range")
        node_one_hot = torch.nn.functional.one_hot(
            node_indices_t, num_classes=self.num_cache_nodes
        ).to(dtype=torch.float32)
        values = self.network(
            torch.cat([observations_t, node_one_hot, joint_action_encoding_t], dim=-1)
        ).squeeze(-1)
        if not bool(torch.all(torch.isfinite(values)).item()):
            raise FloatingPointError("V11-A Cache critic output contains NaN/Inf")
        return values


def _gradient_group_norm(parameters: Any) -> float:
    """Return a pre-clipping L2 norm for one diagnostic parameter group."""

    squared_norm = None
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        component = torch.sum(gradient.detach() ** 2)
        squared_norm = component if squared_norm is None else squared_norm + component
    if squared_norm is None:
        return 0.0
    return float(torch.sqrt(squared_norm).detach().cpu())


def _branch_specific_kl_gate_transition(
    *,
    cache_active: bool,
    uav_active: bool,
    cache_epoch_kl: float,
    uav_epoch_kl: float,
    target_kl: float | None,
    update_epoch: int,
    update_epochs: int,
    cache_gate_enabled: bool = True,
    uav_gate_enabled: bool = True,
) -> dict[str, Any]:
    """Apply one V10-D branch-KL gate transition after a full PPO epoch.

    The transition is deliberately pure so the four required control-flow
    cases can be tested without collecting a rollout or changing a policy.
    A branch can only stop when a subsequent update epoch exists; the final
    epoch remains a diagnostic-only observation, matching the legacy joint
    gate boundary.
    """

    epoch_index = int(update_epoch)
    total_epochs = int(update_epochs)
    if epoch_index < 0 or total_epochs <= 0 or epoch_index >= total_epochs:
        raise ValueError("invalid PPO update epoch for KL gate transition")
    if not bool(cache_active) and not bool(uav_active):
        raise ValueError("branch-specific KL gate cannot transition from both inactive")
    if target_kl is None:
        cache_over_target = None
        uav_over_target = None
    else:
        target = float(target_kl)
        if not math.isfinite(target) or target <= 0.0:
            raise ValueError("target_kl must be positive and finite")
        cache_over_target = (
            bool(float(cache_epoch_kl) > target)
            if bool(cache_gate_enabled)
            else False
        )
        uav_over_target = (
            bool(float(uav_epoch_kl) > target)
            if bool(uav_gate_enabled)
            else False
        )
    remaining_epochs = max(total_epochs - epoch_index - 1, 0)
    can_suppress = remaining_epochs > 0
    cache_newly_stopped = bool(
        can_suppress
        and bool(cache_gate_enabled)
        and bool(cache_active)
        and cache_over_target is True
    )
    uav_newly_stopped = bool(
        can_suppress
        and bool(uav_gate_enabled)
        and bool(uav_active)
        and uav_over_target is True
    )
    next_cache_active = bool(cache_active) and not cache_newly_stopped
    next_uav_active = bool(uav_active) and not uav_newly_stopped
    both_inactive_stop = bool(
        can_suppress and not next_cache_active and not next_uav_active
    )
    return {
        "cache_over_target": cache_over_target,
        "uav_over_target": uav_over_target,
        "cache_newly_stopped": cache_newly_stopped,
        "uav_newly_stopped": uav_newly_stopped,
        "cache_independent_stop": bool(
            cache_newly_stopped and not uav_newly_stopped
        ),
        "uav_independent_stop": bool(
            uav_newly_stopped and not cache_newly_stopped
        ),
        "both_inactive_stop": both_inactive_stop,
        "cache_active_next": next_cache_active,
        "uav_active_next": next_uav_active,
        "remaining_epochs_suppressed": int(remaining_epochs),
    }


def _branch_actor_policy_loss(
    cache_branch_policy_loss: Any,
    uav_branch_policy_loss: Any,
    *,
    cache_active: bool,
    uav_active: bool,
) -> Any:
    """Keep the V10-D actor branch scale at 0.5 for every active branch.

    When both branches have been stopped by the branch-specific KL gate, the
    actor contribution is a graph-connected zero.  This lets the separate
    global critic schedule continue without manufacturing gradients for the
    stopped actor parameters.
    """

    if not bool(cache_active) and not bool(uav_active):
        return 0.0 * (cache_branch_policy_loss + uav_branch_policy_loss)
    loss = cache_branch_policy_loss.new_zeros(())
    if bool(cache_active):
        loss = loss + 0.5 * cache_branch_policy_loss
    if bool(uav_active):
        loss = loss + 0.5 * uav_branch_policy_loss
    return loss


def _clear_inactive_branch_gradients(
    branch_gradient_parameter_groups: dict[str, tuple[Any, ...]],
    *,
    cache_active: bool,
    uav_active: bool,
) -> None:
    """Set stopped actor-branch gradients to ``None`` for stale-Adam safety."""

    if not bool(cache_active):
        for parameter in (
            branch_gradient_parameter_groups.get("cache", ())
            + branch_gradient_parameter_groups.get("cache_encoder", ())
        ):
            parameter.grad = None
    if not bool(uav_active):
        for parameter in (
            branch_gradient_parameter_groups.get("uav", ())
            + branch_gradient_parameter_groups.get("uav_encoder", ())
        ):
            parameter.grad = None
    if not bool(cache_active) and not bool(uav_active):
        for parameter in branch_gradient_parameter_groups.get("shared", ()):
            parameter.grad = None


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
    data_protocol: str
    episode_slots: int
    terminal_origin: int
    configured_rollout_steps: int | None
    minibatch_size: int
    configuration_control_mode: str
    policy_schema: str
    realization: "RealizationArtifacts | None" = None
    observation_time_to_go: Callable[[int], float] | None = None


@dataclass(frozen=True)
class RealizationArtifacts:
    """Paths for one complete randomized episode realization."""

    directory: Path
    trajectory: Path
    vehicle_roles: Path
    tasks: Path
    popularity: Path
    popularity_history: Path
    vehicle_predictions: Path
    popularity_predictions: Path
    prediction_scenarios: Path
    tv_local_cache: Path


@dataclass(frozen=True)
class PPOV2Hyperparameters:
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    learning_rate: float
    actor_learning_rate: float
    critic_learning_rate: float
    value_loss_coefficient: float
    entropy_coefficient: float
    max_grad_norm: float
    update_epochs: int
    target_kl: float | None
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

    def active_statistics(self, epsilon: float, warmup_samples: int) -> tuple[float, float]:
        """Return the active PopArt raw-return mean and standard deviation."""

        if self.count < int(warmup_samples) or self.count <= 0:
            return 0.0, 1.0
        variance = self.m2 / float(self.count)
        if not math.isfinite(variance) or variance < 0.0:
            raise FloatingPointError("running return statistics variance is non-finite")
        scale = math.sqrt(variance + float(epsilon))
        if not math.isfinite(self.mean) or not math.isfinite(scale) or scale <= 0.0:
            raise FloatingPointError("active PopArt statistics are non-finite")
        return float(self.mean), float(scale)

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
    kl_early_stop_triggered: bool
    ppo_update_epochs_executed: int
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
    # The exact normalized advantage tensor consumed by the PPO actor loss.
    # This is exposed for compact action-conditional diagnostics only; it is
    # not recomputed and does not participate in any additional update.
    ppo_advantages: np.ndarray
    raw_returns: np.ndarray
    bootstrap_value: float
    cache_approx_kl: float = math.nan
    uav_approx_kl: float = math.nan
    cache_clip_fraction: float = math.nan
    uav_clip_fraction: float = math.nan
    cache_entropy: float = math.nan
    uav_entropy: float = math.nan
    cache_head_gradient_norm: float = math.nan
    uav_head_gradient_norm: float = math.nan
    shared_actor_gradient_norm: float = math.nan
    cache_policy_loss: float = math.nan
    uav_policy_loss: float = math.nan
    actor_policy_loss: float = math.nan
    uav_branch_old_log_probs: np.ndarray | None = None
    uav_branch_new_log_probs: np.ndarray | None = None
    uav_branch_ratios: np.ndarray | None = None
    uav_branch_clipped: np.ndarray | None = None
    cache_encoder_gradient_norm: float = math.nan
    uav_encoder_gradient_norm: float = math.nan
    kl_minibatch_diagnostics: tuple[dict[str, Any], ...] = ()
    kl_epoch_diagnostics: tuple[dict[str, Any], ...] = ()
    kl_gate_mode: str = KL_GATE_MODE_JOINT
    cache_active_per_update_epoch: tuple[bool, ...] = ()
    uav_active_per_update_epoch: tuple[bool, ...] = ()
    cache_stop_epoch: int | None = None
    uav_stop_epoch: int | None = None
    cache_suppressed_update_epochs: int = 0
    uav_suppressed_update_epochs: int = 0
    joint_kl_diagnostic: bool = False
    cache_independent_stop: bool = False
    uav_independent_stop: bool = False
    both_inactive_stop: bool = False
    q_cache_loss: float = math.nan
    q_cache_explained_variance: float = math.nan
    q_cache_grad_norm_raw: float = math.nan
    q_cache_grad_norm_clipped: float = math.nan
    q_cache_optimizer_steps: int = 0
    q_cache_optimizer_steps_per_rollout: int = 0
    q_cache_loss_before: float = math.nan
    q_cache_loss_after: float = math.nan
    q_cache_ev_before: float = math.nan
    q_cache_ev_after: float = math.nan
    q_cache_epoch_diagnostics: tuple[dict[str, Any], ...] = ()
    cf_adv_raw: np.ndarray | None = None
    cf_adv_normalized: np.ndarray | None = None
    cf_adv_valid_mask: np.ndarray | None = None
    q_cache_selected_values: np.ndarray | None = None
    q_cache_counterfactual_baseline: np.ndarray | None = None
    cf_adv_raw_mean: float = math.nan
    cf_adv_raw_std: float = math.nan
    cf_adv_normalized_mean: float = math.nan
    cf_adv_normalized_std: float = math.nan
    cf_adv_min: float = math.nan
    cf_adv_max: float = math.nan
    cf_adv_positive_fraction: float = math.nan
    cf_adv_negative_fraction: float = math.nan
    cache_node_logprob_sum_max_abs_error: float = math.nan
    cache_surrogate_reduction: str = ""
    valid_cache_node_time_count: int = 0
    cache_loss_scale_relative_to_v10d: float = math.nan
    cache_kl_control: str = DEFAULT_CACHE_KL_CONTROL
    cache_stop_update_epoch: int | None = None
    cache_stop_minibatch: int | None = None
    cache_kl_before_stop: float | None = None
    cache_kl_after_stop: float | None = None
    cache_minibatches_executed: int = 0
    cache_minibatches_suppressed: int = 0
    uav_minibatches_executed: int = 0
    global_critic_steps: int = 0
    # Recovery R2 operation-credit diagnostics.  These are populated only for
    # the R2 schema and are deliberately separate from the legacy global-GAE
    # and joint UAV branch fields above.
    uav_operation_advantages_raw: np.ndarray | None = None
    uav_operation_advantages_scaled: np.ndarray | None = None
    uav_operation_q_stay: np.ndarray | None = None
    uav_operation_q_move: np.ndarray | None = None
    uav_operation_baseline: np.ndarray | None = None
    uav_operation_counterfactual_env_executions: int = 0
    uav_operation_mean_move_candidate_count: float = math.nan
    uav_operation_credit_runtime_seconds: float = math.nan
    uav_operation_advantage_normalization: str = ""
    uav_operation_advantage_centering: bool | None = None
    uav_operation_advantage_sign_flip_count: int = 0
    uav_operation_ratio_granularity: str = ""


@dataclass(frozen=True)
class TrainingCursor:
    completed_episodes: int
    current_train_origin: int
    current_cache_state: np.ndarray
    current_uav_positions: dict[str, tuple[float, float]]
    best_epoch: int
    best_epoch_return: float | None
    next_pool_index: int | None = None
    episode_seed_start: int | None = None


def _project_path(root: Path, value: Any) -> Path:
    return cache_configuration.resolve_path(root, value)


def _configured_realization_artifacts(
    cfg: dict[str, Any], root: Path
) -> RealizationArtifacts:
    get = project_config.get_config_value
    paths = {
        field: _project_path(root, get(cfg, f"paths.{field}"))
        for field in REALIZATION_ARTIFACT_FILENAMES
    }
    return RealizationArtifacts(
        directory=paths["trajectory"].parent,
        trajectory=paths["trajectory"],
        vehicle_roles=paths["vehicle_roles"],
        tasks=paths["tasks"],
        popularity=paths["popularity"],
        popularity_history=paths["popularity_history"],
        vehicle_predictions=paths["vehicle_predictions"],
        popularity_predictions=paths["popularity_predictions"],
        prediction_scenarios=paths["prediction_scenarios"],
        tv_local_cache=paths["tv_local_cache"],
    )


def _realization_artifacts_from_directory(directory: Path) -> RealizationArtifacts:
    realization_dir = Path(directory).expanduser().resolve(strict=False)
    if not realization_dir.is_dir():
        raise FileNotFoundError(
            f"PPO episode realization directory does not exist: {realization_dir}"
        )
    paths = {
        field: realization_dir / filename
        for field, filename in REALIZATION_ARTIFACT_FILENAMES.items()
    }
    missing = [
        filename
        for field, filename in REALIZATION_ARTIFACT_FILENAMES.items()
        if not paths[field].is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "PPO episode realization is missing required artifacts: "
            f"directory={realization_dir}, missing={missing}"
        )
    return RealizationArtifacts(
        directory=realization_dir,
        trajectory=paths["trajectory"],
        vehicle_roles=paths["vehicle_roles"],
        tasks=paths["tasks"],
        popularity=paths["popularity"],
        popularity_history=paths["popularity_history"],
        vehicle_predictions=paths["vehicle_predictions"],
        popularity_predictions=paths["popularity_predictions"],
        prediction_scenarios=paths["prediction_scenarios"],
        tv_local_cache=paths["tv_local_cache"],
    )


def _load_episode_pool(
    project_root: Path, episode_pool: str | Path
) -> tuple[RealizationArtifacts, ...]:
    pool_path = Path(episode_pool).expanduser()
    if not pool_path.is_absolute():
        pool_path = project_root / pool_path
    pool_path = pool_path.resolve(strict=False)
    if not pool_path.is_dir():
        raise FileNotFoundError(
            f"PPO episode pool directory does not exist: {pool_path}"
        )

    child_directories = sorted(
        (child for child in pool_path.iterdir() if child.is_dir()),
        key=lambda child: child.name,
    )
    if child_directories:
        realization_directories = child_directories
    elif any(
        (pool_path / filename).exists()
        for filename in REALIZATION_ARTIFACT_FILENAMES.values()
    ):
        # A directory containing one realization is a useful minimal pool and
        # still follows the same no-reuse/exhaustion contract.
        realization_directories = [pool_path]
    else:
        raise ValueError(
            f"PPO episode pool contains no realization directories: {pool_path}"
        )

    realizations = tuple(
        _realization_artifacts_from_directory(directory)
        for directory in realization_directories
    )
    if not realizations:
        raise ValueError(f"PPO episode pool is empty: {pool_path}")
    return realizations


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_episode_pool_directory(
    project_root: Path,
    episode_pool: str | Path,
    *,
    allow_missing: bool,
) -> Path:
    """Resolve a pool root and protect immutable project/formal artifacts."""

    project_root = project_root.resolve(strict=False)
    pool_path = Path(episode_pool).expanduser()
    if not pool_path.is_absolute():
        pool_path = project_root / pool_path
    pool_path = pool_path.resolve(strict=False)
    forbidden_subtrees = (
        project_root / "data" / "generated",
        project_root / "runs" / "ppo_v6_popart_actor1e4_fresh_20260910",
    )
    if pool_path == project_root or any(
        pool_path == forbidden or _path_is_within(pool_path, forbidden)
        for forbidden in forbidden_subtrees
    ):
        raise ValueError(
            "PPO episode pool must not be the repository root, data/generated, "
            "or the formal V6 run directory"
        )
    if pool_path.exists() and not pool_path.is_dir():
        raise FileNotFoundError(
            f"PPO episode pool path is not a directory: {pool_path}"
        )
    if not pool_path.exists():
        if not allow_missing:
            raise FileNotFoundError(
                f"PPO episode pool directory does not exist: {pool_path}"
            )
        pool_path.mkdir(parents=True, exist_ok=True)
    return pool_path


def _resolve_config_file(project_root: Path, config_path: str | Path) -> Path:
    config_file = Path(config_path).expanduser()
    if not config_file.is_absolute():
        config_file = project_root / config_file
    return config_file.resolve(strict=False)


def _load_single_episode_generator(project_root: Path) -> Any:
    """Load the existing generator only when lazy generation is requested."""

    generator_path = project_root / "scripts" / "generation" / "generate_episode_realization.py"
    if not generator_path.is_file():
        raise FileNotFoundError(
            f"single-episode generator is not available: {generator_path}"
        )
    module_name = "_adaptive_cache_mec_episode_generator"
    spec = importlib.util.spec_from_file_location(module_name, generator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load single-episode generator: {generator_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _validate_episode_realization_complete(
    directory: Path,
    context: EnvironmentContext,
    generator_module: Any,
) -> RealizationArtifacts:
    required_names = (
        *REALIZATION_ARTIFACT_FILENAMES.values(),
        *GENERATOR_EXTRA_ARTIFACT_FILENAMES,
    )
    missing = [name for name in required_names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "PPO episode realization is incomplete: "
            f"directory={directory}, missing={missing}"
        )
    artifacts = _realization_artifacts_from_directory(directory)
    get = project_config.get_config_value
    generator_module.validate_written_outputs(
        directory,
        task_vehicle_count=int(get(context.cfg, "simulation.num_task_vehicles")),
        research_vehicle_count=int(
            get(context.cfg, "simulation.num_research_vehicles")
        ),
        horizon_steps=int(get(context.cfg, "prediction.vehicle.horizon")),
    )
    return artifacts


def _ensure_episode_realization(
    context: EnvironmentContext,
    config_path: str | Path,
    directory: Path,
    episode_seed: int,
    *,
    generate_missing: bool,
    generation_device: str,
    generator_module: Any,
) -> tuple[RealizationArtifacts, str]:
    """Load one complete realization or generate exactly one missing episode."""

    realization_dir = Path(directory).expanduser().resolve(strict=False)
    if realization_dir.exists():
        if not realization_dir.is_dir():
            raise RuntimeError(
                "PPO episode realization path is not a directory; refusing to "
                f"overwrite it: {realization_dir}"
            )
        try:
            artifacts = _validate_episode_realization_complete(
                realization_dir, context, generator_module
            )
        except Exception as exc:
            raise RuntimeError(
                "PPO episode realization exists but is incomplete or invalid; "
                "refusing to regenerate or overwrite it: "
                f"{realization_dir}"
            ) from exc
        return artifacts, "LOAD"

    if not generate_missing:
        raise FileNotFoundError(
            "PPO episode realization is missing; use --generate-missing-episodes: "
            f"{realization_dir}"
        )

    realization_dir.mkdir(parents=True, exist_ok=False)
    try:
        generator_module.run_episode(
            _resolve_config_file(context.root, config_path),
            int(episode_seed),
            realization_dir,
            None,
            generation_device,
        )
        artifacts = _validate_episode_realization_complete(
            realization_dir, context, generator_module
        )
    except Exception as exc:
        raise RuntimeError(
            "lazy PPO episode realization generation failed; the partial "
            "directory was left in place and will not be overwritten: "
            f"{realization_dir}"
        ) from exc
    return artifacts, "GENERATE"


def _assert_realization_context_compatible(
    reference: EnvironmentContext,
    candidate: EnvironmentContext,
) -> None:
    """Ensure a new realization has the same PPO tensor/protocol contract."""

    scalar_fields = (
        "data_protocol",
        "episode_slots",
        "train_origin_start",
        "train_origin_end",
        "terminal_origin",
        "configured_rollout_steps",
        "minibatch_size",
        "configuration_control_mode",
        "policy_schema",
    )
    for field in scalar_fields:
        if getattr(reference, field) != getattr(candidate, field):
            raise ValueError(
                "PPO episode realization changes the fixed training contract: "
                f"field={field}, reference={getattr(reference, field)!r}, "
                f"candidate={getattr(candidate, field)!r}"
            )
    reference_observation = reference.observation_spec
    candidate_observation = candidate.observation_spec
    if (
        reference_observation.observation_dim
        != candidate_observation.observation_dim
        or len(reference_observation.vehicle_ids)
        != len(candidate_observation.vehicle_ids)
        or reference_observation.service_ids != candidate_observation.service_ids
        or reference_observation.uav_ids != candidate_observation.uav_ids
        or reference_observation.prediction_horizon
        != candidate_observation.prediction_horizon
        or (reference_observation.horizon_mask_slice is None)
        != (candidate_observation.horizon_mask_slice is None)
    ):
        raise ValueError(
            "PPO episode realization changes the observation tensor contract: "
            f"reference_dim={reference_observation.observation_dim}, "
            f"candidate_dim={candidate_observation.observation_dim}"
        )
    if (
        reference.action_spec.action_dim != candidate.action_spec.action_dim
        or len(reference.action_spec.cache_node_ids)
        != len(candidate.action_spec.cache_node_ids)
        or reference.action_spec.uav_ids != candidate.action_spec.uav_ids
    ):
        raise ValueError(
            "PPO episode realization changes the action tensor contract: "
            f"reference_dim={reference.action_spec.action_dim}, "
            f"candidate_dim={candidate.action_spec.action_dim}"
        )
    if np.asarray(reference.initial_cache).shape != np.asarray(candidate.initial_cache).shape:
        raise ValueError("PPO episode realization changes the initial cache tensor shape")
    if set(reference.initial_uav) != set(candidate.initial_uav):
        raise ValueError("PPO episode realization changes the initial UAV ID set")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one finite, structured training metric record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


def _binary_action_bitmask(action: Any) -> int:
    """Encode a binary action vector as one compact, deterministic integer."""

    values = np.asarray(action, dtype=np.int8).reshape(-1)
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("binary action bitmask requires only 0/1 values")
    bitmask = 0
    for index, value in enumerate(values.tolist()):
        if int(value):
            bitmask |= 1 << int(index)
    return int(bitmask)


def _cache_operation_epoch_metrics(
    transitions: Any,
    initial_cache: np.ndarray,
    cache_nodes: dict[str, Any],
    service_sizes_mb: np.ndarray,
) -> dict[str, Any]:
    """Summarize cache primitive behavior without affecting the rollout/update."""

    operation_names = ("KEEP", "ADD", "DELETE", "SWAP")
    roles = ("SV", "RSU", "UAV")
    node_roles = [str(value).upper() for value in cache_nodes["node_roles"]]
    node_count = len(node_roles)
    capacities = np.asarray(cache_nodes["capacities_mb"], dtype=np.float64).reshape(-1)
    sizes = np.asarray(service_sizes_mb, dtype=np.float64).reshape(-1)
    operation_counts = {
        role: {operation: 0 for operation in operation_names} for role in roles
    }
    role_samples = {role: 0 for role in roles}
    previous_cache = np.asarray(initial_cache, dtype=np.int8).copy()
    rsu_service_count: list[float] = []
    rsu_capacity_utilization: list[float] = []
    rsu_nonempty: list[bool] = []
    rsu_requested_updates = 0
    rsu_actual_changes = 0
    rsu_samples = 0

    for transition in transitions:
        next_cache = np.asarray(transition.next_cache_state, dtype=np.int8)
        if next_cache.shape != previous_cache.shape:
            raise ValueError("cache behavior diagnostics encountered a state shape change")
        actions = {
            int(item["head_index"]): item
            for item in transition.cache_conditional_actions
            if "head_index" in item
        }
        for node_index, role in enumerate(node_roles):
            role_samples[role] = role_samples.get(role, 0) + 1
            action = actions.get(node_index)
            if action is not None and role in operation_counts:
                raw_operation = str(action.get("operation_type", "")).upper()
                operation = "KEEP" if raw_operation == "NO_CHANGE" else raw_operation
                if operation in operation_counts[role]:
                    operation_counts[role][operation] += 1
            if role != "RSU":
                continue
            row = next_cache[node_index]
            used_mb = float(np.dot(row.astype(np.float64), sizes))
            capacity = float(capacities[node_index])
            if not math.isfinite(used_mb) or not math.isfinite(capacity) or capacity <= 0.0:
                raise FloatingPointError("RSU cache behavior diagnostic is non-finite")
            rsu_service_count.append(float(row.sum()))
            rsu_capacity_utilization.append(used_mb / capacity)
            rsu_nonempty.append(bool(row.sum() > 0))
            rsu_requested_updates += int(
                np.asarray(transition.action, dtype=np.int8)[node_index] == 1
            )
            rsu_actual_changes += int(
                np.any(row != previous_cache[node_index])
            )
            rsu_samples += 1
        previous_cache = next_cache.copy()

    def _rate(count: int, denominator: int) -> float:
        return 0.0 if denominator <= 0 else float(count) / float(denominator)

    operation_rates = {
        role: {
            operation: _rate(count, role_samples[role])
            for operation, count in counts.items()
        }
        for role, counts in operation_counts.items()
    }
    return {
        "cache_operation_counts": operation_counts,
        "cache_operation_rates": operation_rates,
        "RSU_mean_cached_service_count": (
            0.0 if not rsu_service_count else float(np.mean(rsu_service_count))
        ),
        "RSU_nonempty_fraction": (
            0.0 if not rsu_nonempty else float(np.mean(rsu_nonempty))
        ),
        "RSU_cache_capacity_utilization": (
            0.0
            if not rsu_capacity_utilization
            else float(np.mean(rsu_capacity_utilization))
        ),
        "RSU_requested_update_rate": _rate(rsu_requested_updates, rsu_samples),
        "RSU_actual_change_rate": _rate(rsu_actual_changes, rsu_samples),
        "RSU_operation_sample_count": int(rsu_samples),
    }


def _branch_update_metric_fields(update: UpdateResult) -> dict[str, float]:
    """Return finite branch diagnostics without changing legacy log schemas."""

    values = {
        "cache_approx_kl": float(update.cache_approx_kl),
        "uav_approx_kl": float(update.uav_approx_kl),
        "cache_clip_fraction": float(update.cache_clip_fraction),
        "uav_clip_fraction": float(update.uav_clip_fraction),
        "cache_entropy": float(update.cache_entropy),
        "uav_entropy": float(update.uav_entropy),
        "cache_head_gradient_norm": float(update.cache_head_gradient_norm),
        "uav_head_gradient_norm": float(update.uav_head_gradient_norm),
        "shared_actor_gradient_norm": float(update.shared_actor_gradient_norm),
    }
    fields = values if all(math.isfinite(value) for value in values.values()) else {}
    policy_losses = {
        "cache_policy_loss": float(update.cache_policy_loss),
        "uav_policy_loss": float(update.uav_policy_loss),
        "actor_policy_loss": float(update.actor_policy_loss),
    }
    if all(math.isfinite(value) for value in policy_losses.values()):
        fields.update(policy_losses)
    encoder_gradient_norms = {
        "cache_encoder_gradient_norm": float(update.cache_encoder_gradient_norm),
        "uav_encoder_gradient_norm": float(update.uav_encoder_gradient_norm),
    }
    if all(math.isfinite(value) for value in encoder_gradient_norms.values()):
        fields.update(encoder_gradient_norms)
    return fields


def _v11a_update_metric_fields(
    update: UpdateResult,
    rollout: SimpleRollout,
    cache_nodes: dict[str, Any],
    *,
    fixed_transition_seed: int = 4401,
) -> dict[str, Any]:
    """Return the bounded V11-A diagnostics persisted for a gate run."""

    if (
        update.cf_adv_raw is None
        or update.cf_adv_normalized is None
        or update.cf_adv_valid_mask is None
    ):
        return {}
    raw = np.asarray(update.cf_adv_raw, dtype=np.float64)
    normalized = np.asarray(update.cf_adv_normalized, dtype=np.float64)
    valid = np.asarray(update.cf_adv_valid_mask, dtype=bool)
    if raw.shape != normalized.shape or raw.shape != valid.shape:
        raise AssertionError("V11-A diagnostics have inconsistent CF advantage shapes")
    roles = np.asarray([str(value).upper() for value in cache_nodes["node_roles"]])
    if raw.ndim != 2 or raw.shape[1] != len(roles):
        raise AssertionError("V11-A diagnostics do not match Cache node roles")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(normalized)):
        raise FloatingPointError("V11-A diagnostics contain non-finite CF advantages")
    fields: dict[str, Any] = {
        "q_cache_loss": float(update.q_cache_loss),
        "q_cache_ev": float(update.q_cache_explained_variance),
        "q_cache_grad_norm": float(update.q_cache_grad_norm_raw),
        "q_cache_grad_norm_raw": float(update.q_cache_grad_norm_raw),
        "q_cache_grad_norm_clipped": float(update.q_cache_grad_norm_clipped),
        "q_cache_optimizer_steps": int(update.q_cache_optimizer_steps),
        "q_cache_optimizer_steps_per_rollout": int(
            update.q_cache_optimizer_steps_per_rollout
        ),
        "q_cache_loss_before": float(update.q_cache_loss_before),
        "q_cache_loss_after": float(update.q_cache_loss_after),
        "q_cache_ev_before": float(update.q_cache_ev_before),
        "q_cache_ev_after": float(update.q_cache_ev_after),
        "q_cache_epoch_diagnostics": [
            dict(record) for record in update.q_cache_epoch_diagnostics
        ],
        "cf_adv_raw_mean": float(update.cf_adv_raw_mean),
        "cf_adv_raw_std": float(update.cf_adv_raw_std),
        "cf_adv_normalized_mean": float(update.cf_adv_normalized_mean),
        "cf_adv_normalized_std": float(update.cf_adv_normalized_std),
        "cf_adv_min": float(update.cf_adv_min),
        "cf_adv_max": float(update.cf_adv_max),
        "cf_adv_positive_rate": float(update.cf_adv_positive_fraction),
        "cf_adv_negative_rate": float(update.cf_adv_negative_fraction),
        "cache_node_logprob_sum_max_abs_error": float(
            update.cache_node_logprob_sum_max_abs_error
        ),
        "cache_surrogate_reduction": str(update.cache_surrogate_reduction),
        "valid_cache_node_time_count": int(update.valid_cache_node_time_count),
        "cache_loss_scale_relative_to_v10d": float(
            update.cache_loss_scale_relative_to_v10d
        ),
        "cache_kl_control": str(update.cache_kl_control),
        "cache_stop_update_epoch": update.cache_stop_update_epoch,
        "cache_stop_minibatch": update.cache_stop_minibatch,
        "cache_kl_before_stop": update.cache_kl_before_stop,
        "cache_kl_after_stop": update.cache_kl_after_stop,
        "cache_minibatches_executed": int(update.cache_minibatches_executed),
        "cache_minibatches_suppressed": int(update.cache_minibatches_suppressed),
        "uav_minibatches_executed": int(update.uav_minibatches_executed),
        "global_critic_steps": int(update.global_critic_steps),
        "q_cache_steps": int(update.q_cache_optimizer_steps_per_rollout),
    }
    node_type_stats: dict[str, Any] = {}
    for role in ("SV", "RSU", "UAV"):
        role_values = normalized[:, roles == role]
        role_valid = valid[:, roles == role]
        selected = role_values[role_valid]
        output_role = "UAV-cache" if role == "UAV" else role
        if selected.size:
            node_type_stats[output_role] = {
                "count": int(selected.size),
                "cf_adv_mean": float(np.mean(selected)),
                "cf_adv_std": float(np.std(selected)),
                "positive_rate": float(np.mean(selected > 0.0)),
                "negative_rate": float(np.mean(selected < 0.0)),
            }
        else:
            node_type_stats[output_role] = {
                "count": 0,
                "cf_adv_mean": None,
                "cf_adv_std": None,
                "positive_rate": None,
                "negative_rate": None,
            }
    fields["cf_adv_by_cache_node_type"] = node_type_stats
    if not rollout.transitions:
        raise AssertionError("V11-A fixed diagnostic transition requires a rollout")
    q_selected = (
        None
        if update.q_cache_selected_values is None
        else np.asarray(update.q_cache_selected_values, dtype=np.float64)
    )
    q_baseline = (
        None
        if update.q_cache_counterfactual_baseline is None
        else np.asarray(update.q_cache_counterfactual_baseline, dtype=np.float64)
    )
    if q_selected is None or q_baseline is None:
        raise AssertionError("V11-A fixed diagnostic transition requires Q values")
    if q_selected.shape != raw.shape or q_baseline.shape != raw.shape:
        raise AssertionError("V11-A Q diagnostics do not match Cache node roles")
    if not np.all(np.isfinite(q_selected)) or not np.all(np.isfinite(q_baseline)):
        raise FloatingPointError("V11-A fixed Q diagnostics contain NaN/Inf")
    fixed_index = int(np.random.default_rng(int(fixed_transition_seed)).integers(0, len(rollout.transitions)))
    transition = rollout.transitions[fixed_index]
    action_records = []
    for node_index, action in enumerate(transition.cache_conditional_actions):
        action_records.append(
            {
                "node_index": int(node_index),
                "node_id": str(action.get("head_id")),
                "operation": str(action.get("operation_type")),
                "selected_index": action.get("selected_index"),
            }
        )
    fields["cf_adv_fixed_transition"] = {
        "transition_index": fixed_index,
        "origin": int(transition.origin_index),
        "selected_action": action_records,
        "A_global": float(update.ppo_advantages[fixed_index]),
        "A_global_raw": float(update.raw_advantages[fixed_index]),
        "A_cf_by_cache_node": [
            float(value) for value in raw[fixed_index].tolist()
        ],
        "A_cf_normalized_by_cache_node": [
            float(value) for value in normalized[fixed_index].tolist()
        ],
        "valid_node_time": [bool(value) for value in valid[fixed_index].tolist()],
        "Q_selected_by_cache_node": [
            float(value) for value in q_selected[fixed_index].tolist()
        ],
        "Q_counterfactual_baseline_by_cache_node": [
            float(value) for value in q_baseline[fixed_index].tolist()
        ],
    }
    return fields


def _v10c_kl_update_metric_fields(update: UpdateResult) -> dict[str, Any]:
    """Return V10-C-only minibatch/epoch KL diagnostics for update JSONL."""

    if not update.kl_epoch_diagnostics:
        return {}
    epoch_fields = (
        "cache_only_kl_trigger_count",
        "uav_only_kl_trigger_count",
        "both_kl_trigger_count",
        "neither_kl_trigger_count",
    )
    fields: dict[str, Any] = {
        "kl_minibatch_diagnostics": [
            dict(record) for record in update.kl_minibatch_diagnostics
        ],
        "kl_epoch_diagnostics": [
            dict(record) for record in update.kl_epoch_diagnostics
        ],
        **{
            field: int(
                sum(int(record[field]) for record in update.kl_epoch_diagnostics)
            )
            for field in epoch_fields
        },
        "cache_trigger_suppressed_uav_update": bool(
            any(
                record["cache_trigger_suppressed_uav_update"]
                for record in update.kl_epoch_diagnostics
            )
        ),
        "uav_trigger_suppressed_cache_update": bool(
            any(
                record["uav_trigger_suppressed_cache_update"]
                for record in update.kl_epoch_diagnostics
            )
        ),
    }
    if update.cache_active_per_update_epoch or update.uav_active_per_update_epoch:
        fields.update(
            {
                "kl_gate_mode": str(update.kl_gate_mode),
                "cache_active_per_update_epoch": [
                    bool(value) for value in update.cache_active_per_update_epoch
                ],
                "uav_active_per_update_epoch": [
                    bool(value) for value in update.uav_active_per_update_epoch
                ],
                "cache_stop_epoch": update.cache_stop_epoch,
                "uav_stop_epoch": update.uav_stop_epoch,
                "cache_suppressed_update_epochs": int(
                    update.cache_suppressed_update_epochs
                ),
                "uav_suppressed_update_epochs": int(
                    update.uav_suppressed_update_epochs
                ),
                "joint_kl_diagnostic": bool(update.joint_kl_diagnostic),
                "cache_independent_stop": bool(update.cache_independent_stop),
                "uav_independent_stop": bool(update.uav_independent_stop),
                "both_inactive_stop": bool(update.both_inactive_stop),
                "joint_kl_control_enabled": False,
            }
        )
    return fields


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


def _uav_episode_diagnostics(
    *,
    available: bool,
    move_probabilities: list[float],
    stay_probabilities: list[float],
    logit_margins: list[float],
    move_feasible: list[bool],
    move_selected: list[bool],
    actual_movement: list[bool],
    move_advantages: list[float],
    stay_advantages: list[float],
    global_advantages: list[float],
    global_return_targets: list[float],
    rewards: list[float],
    q_values: list[float],
    movement_costs: list[float],
    delays: list[float],
    energies: list[float],
    old_log_probs: list[float],
    new_log_probs: list[float],
    ratios: list[float],
    clipped: list[bool],
    feasible_action_counts: list[int],
    move_candidate_counts: list[int],
) -> dict[str, Any]:
    """Summarize sampled UAV diagnostics without another policy call.

    The required MOVE/STAY action-conditional fields are grouped by the
    verified physical ``actual_movement`` flag.  Selected operation labels are
    retained separately so a future run can audit action-vs-physical mismatches
    without silently treating an action index as a movement outcome.
    """

    empty = {
        "uav_move_probability_mean": None,
        "uav_move_probability_median": None,
        "uav_move_probability_p10": None,
        "uav_move_probability_p90": None,
        "uav_move_probability_gt_0_5_rate": None,
        "uav_stay_probability_mean": None,
        "uav_head_entropy": None,
        "uav_binary_entropy_mean": None,
        "uav_stay_move_logit_margin_mean": None,
        "uav_stay_move_logit_margin_median": None,
        "uav_move_feasible_count": None,
        "uav_move_feasible_rate": None,
        "uav_move_selected_count": None,
        "uav_move_selected_rate": None,
        "uav_move_given_selected": None,
        "uav_action_advantage_logging_available": False,
        "mean_advantage_for_uav_move": None,
        "median_advantage_for_uav_move": None,
        "mean_advantage_for_uav_stay": None,
        "median_advantage_for_uav_stay": None,
        "uav_action_conditional_logging_available": False,
        "uav_move_count": None,
        "uav_stay_count": None,
        "uav_move_adv_mean": None,
        "uav_stay_adv_mean": None,
        "uav_move_adv_median": None,
        "uav_stay_adv_median": None,
        "uav_move_adv_positive_fraction": None,
        "uav_stay_adv_positive_fraction": None,
        "uav_move_reward_mean": None,
        "uav_stay_reward_mean": None,
        "uav_move_q_total_mean": None,
        "uav_stay_q_total_mean": None,
        "uav_move_movement_cost_mean": None,
        "uav_stay_movement_cost_mean": None,
        "uav_move_delay_mean": None,
        "uav_stay_delay_mean": None,
        "uav_move_energy_mean": None,
        "uav_stay_energy_mean": None,
        "uav_move_return_target_mean": None,
        "uav_stay_return_target_mean": None,
        "uav_move_old_log_prob_mean": None,
        "uav_stay_old_log_prob_mean": None,
        "uav_move_new_log_prob_mean": None,
        "uav_stay_new_log_prob_mean": None,
        "uav_move_ratio_mean": None,
        "uav_stay_ratio_mean": None,
        "uav_move_clip_fraction": None,
        "uav_stay_clip_fraction": None,
        "uav_selected_move_count": None,
        "uav_selected_stay_count": None,
        "uav_selected_move_actual_move_count": None,
        "uav_selected_move_actual_stay_count": None,
        "uav_selected_stay_actual_move_count": None,
        "uav_selected_stay_actual_stay_count": None,
        "uav_feasible_action_count_mean": None,
        "uav_feasible_action_count_min": None,
        "uav_feasible_action_count_max": None,
        "uav_move_candidate_count_mean": None,
        "uav_move_candidate_count_min": None,
        "uav_move_candidate_count_max": None,
        "uav_action_mask_summary": None,
    }
    if not available:
        return empty

    move = np.asarray(move_probabilities, dtype=np.float64)
    stay = np.asarray(stay_probabilities, dtype=np.float64)
    margins = np.asarray(logit_margins, dtype=np.float64)
    feasible = np.asarray(move_feasible, dtype=bool)
    selected = np.asarray(move_selected, dtype=bool)
    actual = np.asarray(actual_movement, dtype=bool)
    expected_length = len(move)
    if expected_length <= 0 or not all(
        array.ndim == 1 and len(array) == expected_length
        for array in (stay, margins, feasible, selected, actual)
    ):
        raise AssertionError("UAV diagnostic arrays must be aligned and non-empty")
    if not (
        np.all(np.isfinite(move))
        and np.all(np.isfinite(stay))
        and np.all(np.isfinite(margins))
    ):
        raise FloatingPointError("UAV diagnostic arrays contain NaN/Inf")
    if not (
        np.all((move >= 0.0) & (move <= 1.0))
        and np.all((stay >= 0.0) & (stay <= 1.0))
        and np.allclose(move + stay, 1.0, rtol=0.0, atol=1.0e-6)
    ):
        raise AssertionError("UAV diagnostic probabilities are invalid")

    def _aligned_float_array(values: list[float], field: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or len(array) != expected_length:
            raise AssertionError(f"{field} must align with UAV diagnostic samples")
        if not np.all(np.isfinite(array)):
            raise FloatingPointError(f"{field} contains NaN/Inf")
        return array

    advantages = _aligned_float_array(global_advantages, "global_advantages")
    return_targets = _aligned_float_array(
        global_return_targets, "global_return_targets"
    )
    rewards_a = _aligned_float_array(rewards, "rewards")
    q_values_a = _aligned_float_array(q_values, "q_values")
    movement_costs_a = _aligned_float_array(movement_costs, "movement_costs")
    delays_a = _aligned_float_array(delays, "delays")
    energies_a = _aligned_float_array(energies, "energies")
    old_log_probs_a = _aligned_float_array(old_log_probs, "old_log_probs")
    new_log_probs_a = _aligned_float_array(new_log_probs, "new_log_probs")
    ratios_a = _aligned_float_array(ratios, "ratios")
    if np.any(ratios_a < 0.0):
        raise AssertionError("UAV branch ratios must be non-negative")
    clipped_a = np.asarray(clipped, dtype=bool)
    if clipped_a.ndim != 1 or len(clipped_a) != expected_length:
        raise AssertionError("clipped must align with UAV diagnostic samples")
    feasible_counts_a = np.asarray(feasible_action_counts, dtype=np.float64)
    candidate_counts_a = np.asarray(move_candidate_counts, dtype=np.float64)
    if not (
        feasible_counts_a.ndim == candidate_counts_a.ndim == 1
        and len(feasible_counts_a) == len(candidate_counts_a) == expected_length
        and np.all(np.isfinite(feasible_counts_a))
        and np.all(np.isfinite(candidate_counts_a))
        and np.all(feasible_counts_a >= 0.0)
        and np.all(candidate_counts_a >= 0.0)
        and np.all(feasible_counts_a == np.floor(feasible_counts_a))
        and np.all(candidate_counts_a == np.floor(candidate_counts_a))
    ):
        raise AssertionError("UAV action mask counts are invalid")

    def _percentile(values: np.ndarray, percentile: float) -> float:
        return float(np.percentile(values, percentile))

    def _advantage_summary(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        values_a = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(values_a)):
            raise FloatingPointError("UAV action advantages contain NaN/Inf")
        return float(np.mean(values_a)), float(np.median(values_a))

    def _group_stats(mask: np.ndarray) -> dict[str, Any]:
        if mask.ndim != 1 or len(mask) != expected_length:
            raise AssertionError("UAV action-conditional mask is not aligned")
        if not bool(np.any(mask)):
            return {
                "count": 0,
                "adv_mean": None,
                "adv_median": None,
                "adv_positive_fraction": None,
                "reward_mean": None,
                "q_total_mean": None,
                "movement_cost_mean": None,
                "delay_mean": None,
                "energy_mean": None,
                "return_target_mean": None,
                "old_log_prob_mean": None,
                "new_log_prob_mean": None,
                "ratio_mean": None,
                "clip_fraction": None,
            }

        def _mean(values: np.ndarray) -> float:
            result = float(np.mean(values[mask]))
            if not np.isfinite(result):
                raise FloatingPointError("UAV action-conditional mean is non-finite")
            return result

        def _median(values: np.ndarray) -> float:
            result = float(np.median(values[mask]))
            if not np.isfinite(result):
                raise FloatingPointError("UAV action-conditional median is non-finite")
            return result

        return {
            "count": int(np.sum(mask)),
            "adv_mean": _mean(advantages),
            "adv_median": _median(advantages),
            "adv_positive_fraction": float(np.mean(advantages[mask] > 0.0)),
            "reward_mean": _mean(rewards_a),
            "q_total_mean": _mean(q_values_a),
            "movement_cost_mean": _mean(movement_costs_a),
            "delay_mean": _mean(delays_a),
            "energy_mean": _mean(energies_a),
            "return_target_mean": _mean(return_targets),
            "old_log_prob_mean": _mean(old_log_probs_a),
            "new_log_prob_mean": _mean(new_log_probs_a),
            "ratio_mean": _mean(ratios_a),
            "clip_fraction": float(np.mean(clipped_a[mask])),
        }

    move_advantage_mean, move_advantage_median = _advantage_summary(move_advantages)
    stay_advantage_mean, stay_advantage_median = _advantage_summary(stay_advantages)
    actual_move_stats = _group_stats(actual)
    actual_stay_stats = _group_stats(~actual)
    probability_entropy = -(
        np.where(move > 0.0, move * np.log(np.maximum(move, 1.0e-12)), 0.0)
        + np.where(stay > 0.0, stay * np.log(np.maximum(stay, 1.0e-12)), 0.0)
    )
    selected_count = int(np.sum(selected))
    return {
        "uav_move_probability_mean": float(np.mean(move)),
        "uav_move_probability_median": float(np.median(move)),
        "uav_move_probability_p10": _percentile(move, 10.0),
        "uav_move_probability_p90": _percentile(move, 90.0),
        "uav_move_probability_gt_0_5_rate": float(np.mean(move > 0.5)),
        "uav_stay_probability_mean": float(np.mean(stay)),
        "uav_head_entropy": float(np.mean(probability_entropy)),
        "uav_binary_entropy_mean": float(np.mean(probability_entropy)),
        "uav_stay_move_logit_margin_mean": float(np.mean(margins)),
        "uav_stay_move_logit_margin_median": float(np.median(margins)),
        "uav_move_feasible_count": int(np.sum(feasible)),
        "uav_move_feasible_rate": float(np.mean(feasible)),
        "uav_move_selected_count": selected_count,
        "uav_move_selected_rate": float(np.mean(selected)),
        "uav_move_given_selected": (
            None
            if selected_count == 0
            else float(np.sum(actual & selected)) / float(selected_count)
        ),
        "uav_action_advantage_logging_available": True,
        "mean_advantage_for_uav_move": move_advantage_mean,
        "median_advantage_for_uav_move": move_advantage_median,
        "mean_advantage_for_uav_stay": stay_advantage_mean,
        "median_advantage_for_uav_stay": stay_advantage_median,
        "uav_action_conditional_logging_available": True,
        "uav_move_count": actual_move_stats["count"],
        "uav_stay_count": actual_stay_stats["count"],
        "uav_move_adv_mean": actual_move_stats["adv_mean"],
        "uav_stay_adv_mean": actual_stay_stats["adv_mean"],
        "uav_move_adv_median": actual_move_stats["adv_median"],
        "uav_stay_adv_median": actual_stay_stats["adv_median"],
        "uav_move_adv_positive_fraction": actual_move_stats[
            "adv_positive_fraction"
        ],
        "uav_stay_adv_positive_fraction": actual_stay_stats[
            "adv_positive_fraction"
        ],
        "uav_move_reward_mean": actual_move_stats["reward_mean"],
        "uav_stay_reward_mean": actual_stay_stats["reward_mean"],
        "uav_move_q_total_mean": actual_move_stats["q_total_mean"],
        "uav_stay_q_total_mean": actual_stay_stats["q_total_mean"],
        "uav_move_movement_cost_mean": actual_move_stats["movement_cost_mean"],
        "uav_stay_movement_cost_mean": actual_stay_stats["movement_cost_mean"],
        "uav_move_delay_mean": actual_move_stats["delay_mean"],
        "uav_stay_delay_mean": actual_stay_stats["delay_mean"],
        "uav_move_energy_mean": actual_move_stats["energy_mean"],
        "uav_stay_energy_mean": actual_stay_stats["energy_mean"],
        "uav_move_return_target_mean": actual_move_stats["return_target_mean"],
        "uav_stay_return_target_mean": actual_stay_stats["return_target_mean"],
        "uav_move_old_log_prob_mean": actual_move_stats["old_log_prob_mean"],
        "uav_stay_old_log_prob_mean": actual_stay_stats["old_log_prob_mean"],
        "uav_move_new_log_prob_mean": actual_move_stats["new_log_prob_mean"],
        "uav_stay_new_log_prob_mean": actual_stay_stats["new_log_prob_mean"],
        "uav_move_ratio_mean": actual_move_stats["ratio_mean"],
        "uav_stay_ratio_mean": actual_stay_stats["ratio_mean"],
        "uav_move_clip_fraction": actual_move_stats["clip_fraction"],
        "uav_stay_clip_fraction": actual_stay_stats["clip_fraction"],
        "uav_selected_move_count": selected_count,
        "uav_selected_stay_count": int(expected_length - selected_count),
        "uav_selected_move_actual_move_count": int(np.sum(selected & actual)),
        "uav_selected_move_actual_stay_count": int(np.sum(selected & ~actual)),
        "uav_selected_stay_actual_move_count": int(np.sum(~selected & actual)),
        "uav_selected_stay_actual_stay_count": int(np.sum(~selected & ~actual)),
        "uav_feasible_action_count_mean": float(np.mean(feasible_counts_a)),
        "uav_feasible_action_count_min": int(np.min(feasible_counts_a)),
        "uav_feasible_action_count_max": int(np.max(feasible_counts_a)),
        "uav_move_candidate_count_mean": float(np.mean(candidate_counts_a)),
        "uav_move_candidate_count_min": int(np.min(candidate_counts_a)),
        "uav_move_candidate_count_max": int(np.max(candidate_counts_a)),
        "uav_action_mask_summary": {
            "operation_width": 2,
            "feasible_operation_count_mean": float(np.mean(feasible_counts_a)),
            "feasible_operation_count_min": int(np.min(feasible_counts_a)),
            "feasible_operation_count_max": int(np.max(feasible_counts_a)),
            "move_candidate_count_mean": float(np.mean(candidate_counts_a)),
            "move_candidate_count_min": int(np.min(candidate_counts_a)),
            "move_candidate_count_max": int(np.max(candidate_counts_a)),
        },
    }


def _append_critic_diagnostics(
    path: Path,
    *,
    seed: int,
    epoch: int,
    chunk_origin_start: int,
    optimizer_steps: int,
    rollout: SimpleRollout,
    update: UpdateResult,
    cache_count: int | None = None,
) -> None:
    """Append v2 raw critic targets already computed by the PPO update."""

    transitions = rollout.transitions
    if not (
        len(transitions)
        == len(update.raw_advantages)
        == len(update.ppo_advantages)
        == len(update.raw_returns)
    ):
        raise AssertionError("critic diagnostics length mismatch")
    for transition_index, (transition, raw_advantage, raw_return) in enumerate(zip(
        transitions,
        update.raw_advantages,
        update.raw_returns,
        strict=True,
    )):
        old_value = float(transition.old_value)
        target = float(raw_return)
        residual = old_value - target
        normalized_residual = residual / float(update.critic_return_scale)
        global_advantage = float(update.ppo_advantages[transition_index])
        values = (
            old_value,
            float(raw_advantage),
            global_advantage,
            target,
            update.bootstrap_value,
            residual,
        )
        if not all(np.isfinite(value) for value in values):
            raise FloatingPointError("critic diagnostics contain non-finite values")
        record: dict[str, Any] = {
            "record_type": "critic_transition",
            "seed": int(seed),
            "episode": int(epoch),
            "epoch": int(epoch),
            "chunk_origin_start": int(chunk_origin_start),
            "optimizer_steps": int(optimizer_steps),
            "origin": int(transition.origin_index),
            "slot": int(transition.slot),
            "done": bool(transition.done),
            "old_value": old_value,
            "raw_advantage": float(raw_advantage),
            "global_advantage": global_advantage,
            "advantage_source": "actual_PPO_advantage",
            "raw_return": target,
            "global_return_target": target,
            "value_target": target,
            "bootstrap_value": float(update.bootstrap_value),
            "value_residual": residual,
            "abs_value_residual": abs(residual),
            "critic_return_scale": float(update.critic_return_scale),
            "normalized_value_residual": normalized_residual,
        }
        if transition.uav_operation_actions is not None:
            if cache_count is None:
                raise AssertionError(
                    "cache_count is required for joint UAV action diagnostics"
                )
            if (
                transition.uav_operation_masks is None
                or transition.uav_candidate_counts is None
                or update.uav_branch_old_log_probs is None
                or update.uav_branch_new_log_probs is None
                or update.uav_branch_ratios is None
                or update.uav_branch_clipped is None
            ):
                raise AssertionError(
                    "joint UAV action diagnostics are incomplete"
                )
            operation_actions = np.asarray(
                transition.uav_operation_actions, dtype=np.int64
            ).reshape(-1)
            operation_masks = np.asarray(
                transition.uav_operation_masks, dtype=bool
            )
            candidate_counts = np.asarray(
                transition.uav_candidate_counts, dtype=np.int64
            ).reshape(-1)
            actual_flags = np.asarray(
                [
                    bool(action.get("actual_moved", False))
                    for action in transition.uav_conditional_actions
                ],
                dtype=bool,
            ).reshape(-1)
            uav_count = len(operation_actions)
            if not (
                operation_masks.shape == (uav_count, 2)
                and len(candidate_counts) == uav_count
                and len(actual_flags) == uav_count
                and np.all(operation_actions >= 0)
                and np.all(operation_actions < 2)
                and np.all(candidate_counts >= 0)
                and np.all(np.isfinite(operation_masks))
            ):
                raise AssertionError("joint UAV action diagnostic shapes are invalid")
            if int(np.sum(actual_flags)) != int(transition.actual_uav_movement_count):
                raise AssertionError(
                    "UAV actual movement flags disagree with rollout movement count"
                )
            branch_old = float(update.uav_branch_old_log_probs[transition_index])
            branch_new = float(update.uav_branch_new_log_probs[transition_index])
            branch_ratio = float(update.uav_branch_ratios[transition_index])
            branch_clipped = bool(update.uav_branch_clipped[transition_index])
            branch_values = (branch_old, branch_new, branch_ratio)
            if not all(np.isfinite(value) for value in branch_values):
                raise FloatingPointError(
                    "joint UAV branch action diagnostics are non-finite"
                )
            cache_action = np.asarray(
                transition.action[: int(cache_count)], dtype=np.int8
            ).reshape(-1)
            if len(cache_action) != int(cache_count) or not np.all(
                (cache_action == 0) | (cache_action == 1)
            ):
                raise AssertionError("selected Cache action is not binary")
            record.update(
                {
                    "selected_cache_action": [int(value) for value in cache_action],
                    "selected_uav_action": [
                        "MOVE" if int(value) == 1 else "STAY"
                        for value in operation_actions
                    ],
                    "actual_uav_move": [int(value) for value in actual_flags],
                    "old_uav_log_prob": branch_old,
                    "new_uav_log_prob": branch_new,
                    "uav_ratio": branch_ratio,
                    "uav_clipped": branch_clipped,
                    "uav_log_prob_scope": "joint_uav_branch_per_transition",
                    "feasible_uav_action_count": [
                        int(value) for value in operation_masks.sum(axis=1)
                    ],
                    "uav_action_mask": [
                        [int(value) for value in row]
                        for row in operation_masks
                    ],
                    "uav_move_candidate_count": [
                        int(value) for value in candidate_counts
                    ],
                    "reward": float(transition.reward),
                    "q_total": float(transition.actual_task_utility),
                    "movement_cost": float(transition.movement_cost),
                    "delay": float(transition.total_task_delay_s),
                    "energy": float(transition.total_vehicle_energy_j),
                }
            )
        _append_jsonl(
            path,
            record,
        )


def load_environment_context(
    config_path: str = "configs/system_config.toml",
    realization_dir: str | Path | None = None,
) -> EnvironmentContext:
    cfg = project_config.load_config(config_path)
    project_config.validate_for_formal_experiment(cfg)
    data_protocol = project_config.get_ppo_data_protocol(cfg)
    root = Path(cfg["_meta"]["project_root"])
    realization = (
        _configured_realization_artifacts(cfg, root)
        if realization_dir is None
        else _realization_artifacts_from_directory(
            Path(realization_dir)
            if Path(realization_dir).is_absolute()
            else root / Path(realization_dir)
        )
    )
    cache_cfg = cache_configuration.cache_config_from_project(cfg)
    scenarios = cache_configuration.load_prediction_scenarios(
        realization.prediction_scenarios
    )
    service_catalog = cache_configuration.load_service_catalog(
        _project_path(root, project_config.get_config_value(cfg, "paths.service_catalog"))
    )
    train_start = int(data_protocol["train_origin_start"])
    train_end = int(data_protocol["train_origin_end"])
    origin_count = int(scenarios["origin_slots"].shape[0])
    if train_start < 0 or train_end < train_start or train_end >= origin_count:
        raise ValueError("PPO v2 train origin range 超出 prediction_scenarios。")
    first_origin = cache_configuration.scenario_at_origin(scenarios, train_start)
    cache_nodes = cache_configuration.build_cache_nodes(first_origin, cache_cfg)
    local_cache = task_scenario.load_tv_local_cache(
        realization.tv_local_cache
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
        data_protocol=str(data_protocol["protocol"]),
        episode_slots=int(data_protocol["episode_slots"]),
        terminal_origin=int(data_protocol["terminal_origin"]),
        configured_rollout_steps=(
            None
            if data_protocol["rollout_steps"] is None
            else int(data_protocol["rollout_steps"])
        ),
        minibatch_size=int(data_protocol["minibatch_size"]),
        configuration_control_mode=configuration_control_mode,
        policy_schema=policy_schema,
        realization=realization,
    )


def hyperparameters_from_project(cfg: dict[str, Any]) -> PPOV2Hyperparameters:
    get = project_config.get_config_value
    hidden_sizes = tuple(int(value) for value in get(cfg, "algorithm.ppo.hidden_sizes"))
    learning_rate = float(get(cfg, "algorithm.ppo.learning_rate"))
    raw_target_kl = get(cfg, "algorithm.ppo.target_kl")
    values = PPOV2Hyperparameters(
        gamma=float(get(cfg, "algorithm.ppo.gamma")),
        gae_lambda=float(get(cfg, "algorithm.ppo.gae_lambda")),
        clip_epsilon=float(get(cfg, "algorithm.ppo.clip_epsilon")),
        learning_rate=learning_rate,
        actor_learning_rate=float(
            get(cfg, "algorithm.ppo.actor_learning_rate", learning_rate)
        ),
        critic_learning_rate=float(
            get(cfg, "algorithm.ppo.critic_learning_rate", learning_rate)
        ),
        value_loss_coefficient=float(get(cfg, "algorithm.ppo.value_loss_coefficient")),
        entropy_coefficient=float(get(cfg, "algorithm.ppo.entropy_coefficient")),
        max_grad_norm=float(get(cfg, "algorithm.ppo.max_grad_norm")),
        update_epochs=int(get(cfg, "algorithm.ppo.update_epochs")),
        target_kl=(None if raw_target_kl is None else float(raw_target_kl)),
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
    if (
        not math.isfinite(values.actor_learning_rate)
        or values.actor_learning_rate <= 0.0
        or not math.isfinite(values.critic_learning_rate)
        or values.critic_learning_rate <= 0.0
    ):
        raise ValueError("actor/critic learning rate 必须是正的有限数。")
    if values.learning_rate <= 0.0 or values.value_loss_coefficient < 0.0:
        raise ValueError("PPO v2 learning/value coefficient 非法。")
    if values.entropy_coefficient < 0.0 or values.max_grad_norm <= 0.0:
        raise ValueError("PPO v2 entropy/max_grad_norm 非法。")
    if values.update_epochs <= 0 or values.num_minibatches <= 0:
        raise ValueError("PPO v2 update_epochs/num_minibatches 必须为正整数。")
    if values.target_kl is not None and (
        not math.isfinite(values.target_kl) or values.target_kl <= 0.0
    ):
        raise ValueError("target_kl 必须是正的有限数，或省略以禁用 KL protection。")
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
        CRITIC_VALUE_SCALE_POPART_NORMALIZED,
    }:
        raise ValueError(
            "critic_value_scale_mode 只支持 none/running_return_std/popart_normalized。"
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
        (
            context.realization.vehicle_predictions
            if context.realization is not None
            else _project_path(
                context.root,
                project_config.get_config_value(
                    context.cfg,
                    "paths.vehicle_predictions",
                ),
            )
        ),
        (
            context.realization.popularity_predictions
            if context.realization is not None
            else _project_path(
                context.root,
                project_config.get_config_value(
                    context.cfg,
                    "paths.popularity_predictions",
                ),
            )
        ),
        (
            context.realization.tasks
            if context.realization is not None
            else _project_path(
                context.root,
                project_config.get_config_value(context.cfg, "paths.tasks"),
            )
        ),
        (
            context.realization.trajectory
            if context.realization is not None
            else _project_path(
                context.root,
                project_config.get_config_value(context.cfg, "paths.trajectory"),
            )
        ),
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
    """PPO core for the formal V6 path and retained base-config compatibility."""

    def __init__(self, context: EnvironmentContext, device: str = "cpu") -> None:
        _require_torch()
        self.context = context
        self.hyperparameters = hyperparameters_from_project(context.cfg)
        self.cache_kl_control = str(
            project_config.get_config_value(
                context.cfg,
                "algorithm.ppo.cache_kl_control",
                DEFAULT_CACHE_KL_CONTROL,
            )
        )
        if self.cache_kl_control not in {
            CACHE_KL_CONTROL_EPOCH,
            CACHE_KL_CONTROL_MINIBATCH_EARLY_STOP,
        }:
            raise ValueError(
                "algorithm.ppo.cache_kl_control 只支持 epoch/minibatch_early_stop。"
            )
        if (
            self.cache_kl_control == CACHE_KL_CONTROL_MINIBATCH_EARLY_STOP
            and context.policy_schema not in JOINT_V11A_CACHE_NODE_CF_POLICY_SCHEMAS
        ):
            raise ValueError(
                "minibatch_early_stop cache KL control requires the V11-A policy schema."
            )
        if (
            self.cache_kl_control == CACHE_KL_CONTROL_MINIBATCH_EARLY_STOP
            and self.hyperparameters.target_kl is None
        ):
            raise ValueError(
                "minibatch_early_stop cache KL control requires target_kl."
            )
        self.kl_gate_mode = str(
            project_config.get_config_value(
                context.cfg,
                "algorithm.ppo.kl_gate_mode",
                KL_GATE_MODE_JOINT,
            )
        )
        if self.kl_gate_mode not in {
            KL_GATE_MODE_JOINT,
            KL_GATE_MODE_BRANCH_SPECIFIC,
        }:
            raise ValueError(
                "algorithm.ppo.kl_gate_mode 只支持 joint/branch_specific。"
            )
        if (
            self.kl_gate_mode == KL_GATE_MODE_BRANCH_SPECIFIC
            and context.policy_schema not in JOINT_V10D_BRANCH_SPECIFIC_KL_POLICY_SCHEMAS
        ):
            raise ValueError(
                "branch_specific KL gate requires the V10-D policy schema"
            )
        if (
            context.policy_schema in JOINT_V10D_BRANCH_SPECIFIC_KL_POLICY_SCHEMAS
            and self.kl_gate_mode != KL_GATE_MODE_BRANCH_SPECIFIC
        ):
            raise ValueError("V10-D policy schema requires kl_gate_mode=branch_specific")
        self.device = torch.device(device)
        if context.configuration_control_mode in {
            "policy_primitive",
            "policy_primitive_v7_cache_operation",
            "policy_primitive_v9_joint_cache_uav_operation",
        }:
            primitive_spec = PrimitiveTensorSpec.create(context)
            policy_class: type[AdaptivePrimitiveActorCritic]
            if context.configuration_control_mode == "policy_primitive_v9_joint_cache_uav_operation":
                if context.policy_schema not in JOINT_CACHE_UAV_POLICY_SCHEMAS:
                    raise ValueError(
                        "policy_primitive_v9_joint_cache_uav_operation requires a supported Joint Cache/UAV schema"
                    )
                if context.policy_schema in JOINT_V11A_CACHE_NODE_CF_POLICY_SCHEMAS:
                    policy_class = (
                        AdaptiveJointCacheReplicaAwareUAVOperationV11ACacheNodeCFActorCritic
                    )
                elif context.policy_schema in JOINT_RECOVERY_R2_POLICY_SCHEMAS:
                    policy_class = AdaptiveJointCacheReplicaAwareUAVOperationRecoveryR2ActorCritic
                elif context.policy_schema in JOINT_RECOVERY_R1_POLICY_SCHEMAS:
                    policy_class = AdaptiveJointCacheReplicaAwareUAVOperationRecoveryR1ActorCritic
                elif context.policy_schema in JOINT_V10D_BRANCH_SPECIFIC_KL_POLICY_SCHEMAS:
                    policy_class = (
                        AdaptiveJointCacheReplicaAwareUAVOperationV10DBranchSpecificKLActorCritic
                    )
                elif context.policy_schema in JOINT_V10C_DECOUPLED_POLICY_SCHEMAS:
                    policy_class = (
                        AdaptiveJointCacheReplicaAwareUAVOperationV10CDecoupledEncoderActorCritic
                    )
                elif (
                    context.policy_schema
                    == AdaptiveJointCacheReplicaAwareUAVOperationV10BranchwiseActorCritic.architecture_id
                ):
                    policy_class = (
                        AdaptiveJointCacheReplicaAwareUAVOperationV10BranchwiseActorCritic
                    )
                elif (
                    context.policy_schema
                    == AdaptiveJointCacheReplicaAwareUAVOperationV10BranchObservableActorCritic.architecture_id
                ):
                    policy_class = (
                        AdaptiveJointCacheReplicaAwareUAVOperationV10BranchObservableActorCritic
                    )
                else:
                    policy_class = AdaptiveJointCacheReplicaAwareUAVOperationV9PopArtActorCritic
                self.policy = policy_class(
                    int(context.observation_spec.observation_dim),
                    int(context.action_spec.action_dim),
                    primitive_spec.cache_feature_dim,
                    primitive_spec.uav_feature_dim,
                    primitive_spec.num_cache_heads,
                    self.hyperparameters.hidden_sizes,
                ).to(self.device)
            elif context.configuration_control_mode == "policy_primitive_v7_cache_operation":
                if context.policy_schema not in CACHE_OPERATION_POLICY_SCHEMAS:
                    raise ValueError(
                        "policy_primitive_v7_cache_operation requires a supported "
                        "Cache operation policy schema"
                    )
                policy_class = (
                    AdaptiveCacheOperationGroupReplicaAwareV8NoPredictionPopArtActorCritic
                    if context.policy_schema == NO_PREDICTION_POLICY_SCHEMA
                    else (
                        AdaptiveCacheOperationGroupReplicaAwareV8PopArtActorCritic
                        if context.policy_schema
                        == AdaptiveCacheOperationGroupReplicaAwareV8PopArtActorCritic.architecture_id
                        else AdaptiveCacheOperationGroupV7PopArtActorCritic
                    )
                )
                self.policy = policy_class(
                    int(context.observation_spec.observation_dim),
                    int(context.action_spec.action_dim),
                    primitive_spec.cache_feature_dim,
                    primitive_spec.uav_feature_dim,
                    primitive_spec.num_cache_heads,
                    self.hyperparameters.hidden_sizes,
                ).to(self.device)
            else:
                if (
                    context.policy_schema
                    == AdaptivePrimitiveHorizonGroupedSeparateCriticPopArtV6ActorCritic.architecture_id
                ):
                    policy_class = AdaptivePrimitiveHorizonGroupedSeparateCriticPopArtV6ActorCritic
                elif context.policy_schema == AdaptivePrimitiveActorCritic.architecture_id:
                    # configs/system_config.toml intentionally retains the original primitive
                    # schema as a compatibility/default configuration.
                    policy_class = AdaptivePrimitiveActorCritic
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
        if (
            self.policy.architecture_id in POPART_POLICY_SCHEMAS
        ) != (
            self.hyperparameters.critic_value_scale_mode
            == CRITIC_VALUE_SCALE_POPART_NORMALIZED
        ):
            raise ValueError(
                "V6 PopArt policy requires critic_value_scale_mode=popart_normalized, "
                "and popart_normalized is reserved for the V6 policy."
            )
        self.optimizer_steps = 0
        self.actor_optimizer_steps = 0
        self.critic_optimizer_steps = 0
        if self._uses_separate_optimization():
            if not isinstance(
                self.policy,
                AdaptivePrimitiveHorizonGroupedSeparateCriticOptimizationV5ActorCritic,
            ):
                raise TypeError("separate optimization requires a V5/V6 actor-critic policy.")
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
                lr=self.hyperparameters.actor_learning_rate,
            )
            self.critic_optimizer = torch.optim.Adam(
                critic_parameters,
                lr=self.hyperparameters.critic_learning_rate,
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
        self.q_cache: CacheNodeCounterfactualCritic | None = None
        self.q_cache_optimizer: Any | None = None
        self.q_cache_optimizer_steps = 0
        if self._uses_v11a_cache_node_cf():
            primitive_spec = PrimitiveTensorSpec.create(context)
            cache_operation_count = len(getattr(self.policy, "operation_order", ()))
            uav_operation_count = len(getattr(self.policy, "uav_operation_order", ()))
            if cache_operation_count <= 0 or uav_operation_count <= 0:
                raise ValueError("V11-A action orders must be non-empty")
            cache_action_encoding_dim = (
                int(cache_operation_count)
                + int(primitive_spec.cache_max_candidates)
                + int(primitive_spec.cache_feature_dim)
            )
            uav_action_encoding_dim = (
                int(uav_operation_count)
                + int(primitive_spec.uav_max_candidates)
                + int(primitive_spec.uav_feature_dim)
            )
            joint_action_dim = (
                int(primitive_spec.num_cache_heads) * cache_action_encoding_dim
                + int(primitive_spec.num_uav_heads) * uav_action_encoding_dim
            )
            self.q_cache = CacheNodeCounterfactualCritic(
                int(context.observation_spec.observation_dim),
                int(primitive_spec.num_cache_heads),
                joint_action_dim,
                self.hyperparameters.hidden_sizes,
            ).to(self.device)
            self.v11a_cache_action_encoding_dim = int(cache_action_encoding_dim)
            self.v11a_uav_action_encoding_dim = int(uav_action_encoding_dim)
            self.v11a_joint_action_dim = int(joint_action_dim)
            policy_parameter_ids = {id(parameter) for parameter in self.policy.parameters()}
            q_parameter_ids = {id(parameter) for parameter in self.q_cache.parameters()}
            if policy_parameter_ids & q_parameter_ids:
                raise RuntimeError(
                    "V11-A Q_cache parameters unexpectedly share tensors with Actor/global critic."
                )
            self.q_cache_optimizer = torch.optim.Adam(
                self.q_cache.parameters(),
                lr=self.hyperparameters.critic_learning_rate,
            )

    def _uses_separate_optimization(self) -> bool:
        return self.policy.architecture_id in SEPARATE_OPTIMIZATION_POLICY_SCHEMAS

    def _uses_v11a_cache_node_cf(self) -> bool:
        return self.policy.architecture_id in JOINT_V11A_CACHE_NODE_CF_POLICY_SCHEMAS

    def _uses_r2_operation_credit(self) -> bool:
        return self.policy.architecture_id in JOINT_RECOVERY_R2_POLICY_SCHEMAS

    def _uses_popart_critic(self) -> bool:
        return self.policy.architecture_id in POPART_POLICY_SCHEMAS

    def _critic_return_scale(self) -> float:
        if self.hyperparameters.critic_value_scale_mode == CRITIC_VALUE_SCALE_NONE:
            return 1.0
        if self._uses_popart_critic():
            if not isinstance(self.policy, AdaptivePrimitiveHorizonGroupedSeparateCriticPopArtV6ActorCritic):
                raise TypeError("PopArt scaling requires the V6 actor-critic policy.")
            return self.policy.popart_statistics()[1]
        return self.critic_return_statistics.scale(
            self.hyperparameters.critic_value_scale_epsilon,
            self.hyperparameters.critic_value_scale_warmup_samples,
        )

    def _expected_checkpoint_version(self) -> int:
        if self._uses_v11a_cache_node_cf():
            return V11A_CACHE_NODE_CF_CHECKPOINT_VERSION
        if self.policy.architecture_id in JOINT_V10D_BRANCH_SPECIFIC_KL_POLICY_SCHEMAS:
            return V10D_BRANCH_SPECIFIC_KL_CHECKPOINT_VERSION
        if self.policy.architecture_id in JOINT_V10C_DECOUPLED_POLICY_SCHEMAS:
            return V10C_DECOUPLED_CHECKPOINT_VERSION
        if self.policy.architecture_id in JOINT_V10_BRANCHWISE_POLICY_SCHEMAS:
            return V10_BRANCHWISE_CHECKPOINT_VERSION
        if self.policy.architecture_id in JOINT_CACHE_UAV_POLICY_SCHEMAS:
            return V9_JOINT_CHECKPOINT_VERSION
        if self.policy.architecture_id in NO_PREDICTION_POLICY_SCHEMAS:
            return V8_NO_PREDICTION_CHECKPOINT_VERSION
        if self.policy.architecture_id in V8_POLICY_SCHEMAS:
            return V8_CHECKPOINT_VERSION
        if self.policy.architecture_id in V7_POLICY_SCHEMAS:
            return V7_CHECKPOINT_VERSION
        if self.policy.architecture_id in V6_POLICY_SCHEMAS:
            return POPART_CHECKPOINT_VERSION
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

    def _v11a_build_joint_action_encoding(
        self,
        *,
        cache_operation_actions: Any,
        all_candidate_features: Any,
        all_candidate_masks: Any,
        cache_selected_indices: Any,
        uav_operation_actions: Any,
        uav_candidate_features: Any,
        uav_candidate_masks: Any,
        uav_selected_indices: Any,
    ) -> Any:
        """Encode the sampled production action for every Cache node and UAV.

        Each Cache slice is ``[operation one-hot, candidate one-hot,
        primitive feature]``.  Each UAV slice uses the same representation
        with the existing ``STAY/MOVE`` operation and candidate tensors.  The
        result is deterministic for a rollout transition and is the only
        action input consumed by the centralized Q_cache.
        """

        if not self._uses_v11a_cache_node_cf():
            raise RuntimeError("V11-A joint action encoding requested for a non-V11-A policy")
        if self.q_cache is None:
            raise RuntimeError("V11-A Cache critic is not initialized")
        cache_actions_t = torch.as_tensor(
            cache_operation_actions, dtype=torch.long, device=self.device
        )
        all_features_t = torch.as_tensor(
            all_candidate_features, dtype=torch.float32, device=self.device
        )
        all_masks_t = torch.as_tensor(
            all_candidate_masks, dtype=torch.bool, device=self.device
        )
        cache_indices_t = torch.as_tensor(
            cache_selected_indices, dtype=torch.long, device=self.device
        )
        uav_actions_t = torch.as_tensor(
            uav_operation_actions, dtype=torch.long, device=self.device
        )
        uav_features_t = torch.as_tensor(
            uav_candidate_features, dtype=torch.float32, device=self.device
        )
        uav_masks_t = torch.as_tensor(
            uav_candidate_masks, dtype=torch.bool, device=self.device
        )
        uav_indices_t = torch.as_tensor(
            uav_selected_indices, dtype=torch.long, device=self.device
        )
        if cache_actions_t.ndim != 2:
            raise ValueError("V11-A Cache operation actions must be [T,H]")
        sample_count, node_count = [int(value) for value in cache_actions_t.shape]
        if all_features_t.ndim != 5:
            raise ValueError("V11-A all Cache candidate features must be [T,H,O,C,F]")
        if tuple(all_features_t.shape[:2]) != (sample_count, node_count):
            raise ValueError("V11-A Cache action/state sample shape mismatch")
        operation_count = len(getattr(self.policy, "operation_order", ()))
        uav_operation_count = len(getattr(self.policy, "uav_operation_order", ()))
        if operation_count <= 0 or uav_operation_count <= 0:
            raise ValueError("V11-A operation orders must be non-empty")
        candidate_count = int(all_features_t.shape[3])
        cache_feature_dim = int(all_features_t.shape[4])
        if tuple(all_features_t.shape[2:]) != (
            operation_count,
            candidate_count,
            cache_feature_dim,
        ):
            raise ValueError("V11-A Cache feature tensor shape mismatch")
        if tuple(all_masks_t.shape) != (
            sample_count,
            node_count,
            operation_count,
            candidate_count,
        ):
            raise ValueError("V11-A Cache mask tensor shape mismatch")
        if tuple(cache_indices_t.shape) != (sample_count, node_count):
            raise ValueError("V11-A Cache selected-index tensor shape mismatch")
        if node_count != self.q_cache.num_cache_nodes:
            raise ValueError("V11-A Cache critic node count mismatch")
        if int(all_features_t.shape[0]) != sample_count:
            raise ValueError("V11-A Cache feature sample count mismatch")
        if uav_actions_t.ndim != 2 or int(uav_actions_t.shape[0]) != sample_count:
            raise ValueError("V11-A UAV operation actions must be [T,U]")
        uav_count = int(uav_actions_t.shape[1])
        if uav_features_t.ndim != 4:
            raise ValueError("V11-A UAV candidate features must be [T,U,C,F]")
        uav_candidate_count = int(uav_features_t.shape[2])
        uav_feature_dim = int(uav_features_t.shape[3])
        if tuple(uav_masks_t.shape) != (sample_count, uav_count, uav_candidate_count):
            raise ValueError("V11-A UAV candidate mask tensor shape mismatch")
        if tuple(uav_indices_t.shape) != (sample_count, uav_count):
            raise ValueError("V11-A UAV selected-index tensor shape mismatch")
        if int(uav_features_t.shape[1]) != uav_count:
            raise ValueError("V11-A UAV feature head count mismatch")
        if uav_count <= 0:
            raise ValueError("V11-A rollout must contain at least one UAV action")
        if not (
            bool(torch.all(torch.isfinite(all_features_t)).item())
            and bool(torch.all(torch.isfinite(uav_features_t)).item())
        ):
            raise FloatingPointError("V11-A joint action features contain NaN/Inf")
        if bool(torch.any(cache_actions_t < 0).item()) or bool(
            torch.any(cache_actions_t >= operation_count).item()
        ):
            raise ValueError("V11-A Cache operation action is out of range")
        if bool(torch.any(cache_indices_t < 0).item()) or bool(
            torch.any(cache_indices_t >= candidate_count).item()
        ):
            raise ValueError("V11-A Cache candidate action is out of range")
        if bool(torch.any(uav_actions_t < 0).item()) or bool(
            torch.any(uav_actions_t >= uav_operation_count).item()
        ):
            raise ValueError("V11-A UAV operation action is out of range")
        if bool(torch.any(uav_indices_t < 0).item()) or bool(
            torch.any(uav_indices_t >= uav_candidate_count).item()
        ):
            raise ValueError("V11-A UAV candidate action is out of range")
        time_grid = torch.arange(sample_count, device=self.device)[:, None]
        cache_node_grid = torch.arange(node_count, device=self.device)[None, :]
        selected_cache_valid = all_masks_t[
            time_grid,
            cache_node_grid,
            cache_actions_t,
            cache_indices_t,
        ]
        if not bool(torch.all(selected_cache_valid).item()):
            raise ValueError("V11-A selected Cache primitive is outside the legal action set")
        uav_time_grid = torch.arange(sample_count, device=self.device)[:, None]
        uav_head_grid = torch.arange(uav_count, device=self.device)[None, :]
        selected_uav_valid = uav_masks_t[
            uav_time_grid,
            uav_head_grid,
            uav_indices_t,
        ]
        if not bool(torch.all(selected_uav_valid).item()):
            raise ValueError("V11-A selected UAV primitive is outside the legal action set")
        selected_cache_features = all_features_t[
            time_grid,
            cache_node_grid,
            cache_actions_t,
            cache_indices_t,
        ]
        selected_uav_features = uav_features_t[
            uav_time_grid,
            uav_head_grid,
            uav_indices_t,
        ]
        cache_encoding = torch.cat(
            [
                torch.nn.functional.one_hot(
                    cache_actions_t, num_classes=operation_count
                ).to(dtype=torch.float32),
                torch.nn.functional.one_hot(
                    cache_indices_t, num_classes=candidate_count
                ).to(dtype=torch.float32),
                selected_cache_features,
            ],
            dim=-1,
        )
        uav_encoding = torch.cat(
            [
                torch.nn.functional.one_hot(
                    uav_actions_t, num_classes=uav_operation_count
                ).to(dtype=torch.float32),
                torch.nn.functional.one_hot(
                    uav_indices_t, num_classes=uav_candidate_count
                ).to(dtype=torch.float32),
                selected_uav_features,
            ],
            dim=-1,
        )
        joint_action_encoding = torch.cat(
            [cache_encoding.reshape(sample_count, -1), uav_encoding.reshape(sample_count, -1)],
            dim=-1,
        )
        if int(joint_action_encoding.shape[1]) != self.q_cache.joint_action_dim:
            raise ValueError(
                "V11-A centralized action encoding dimension mismatch: "
                f"expected={self.q_cache.joint_action_dim}, "
                f"got={int(joint_action_encoding.shape[1])}"
            )
        return joint_action_encoding

    def _v11a_selected_q_values(
        self,
        observations_t: Any,
        joint_action_encoding_t: Any,
    ) -> Any:
        if self.q_cache is None:
            raise RuntimeError("V11-A Cache critic is not initialized")
        if observations_t.ndim != 2 or joint_action_encoding_t.ndim != 2:
            raise ValueError("V11-A selected Q inputs must be [T,D] and [T,J]")
        sample_count = int(observations_t.shape[0])
        node_count = int(self.q_cache.num_cache_nodes)
        if int(joint_action_encoding_t.shape[0]) != sample_count:
            raise ValueError("V11-A selected Q state/action sample mismatch")
        node_grid = torch.arange(node_count, dtype=torch.long, device=self.device)[None, :]
        node_grid = node_grid.expand(sample_count, node_count)
        return self.q_cache(
            observations_t[:, None, :].expand(sample_count, node_count, -1).reshape(
                sample_count * node_count, -1
            ),
            node_grid.reshape(-1),
            joint_action_encoding_t[:, None, :].expand(
                sample_count, node_count, -1
            ).reshape(sample_count * node_count, -1),
        ).reshape(sample_count, node_count)

    def _v11a_q_selected_metrics(
        self,
        observations_t: Any,
        joint_action_encoding_t: Any,
        targets_t: Any,
    ) -> tuple[float, float]:
        with torch.no_grad():
            predictions = self._v11a_selected_q_values(
                observations_t, joint_action_encoding_t
            )
            target_matrix = targets_t[:, None].expand_as(predictions)
            residual = predictions - target_matrix
            loss = 0.5 * torch.mean(residual**2)
            target_variance = torch.var(target_matrix.reshape(-1), unbiased=False)
            if float(target_variance.detach().cpu()) <= 1.0e-12:
                explained_variance = 0.0
            else:
                explained_variance = float(
                    (
                        1.0
                        - torch.var(residual.reshape(-1), unbiased=False)
                        / target_variance
                    ).detach().cpu()
                )
            loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value) or not math.isfinite(explained_variance):
            raise FloatingPointError("V11-A Q_cache selected metrics are non-finite")
        return loss_value, explained_variance

    def _train_v11a_cache_node_critic(
        self,
        *,
        observations: Any,
        cache_contexts: Any,
        operation_masks: Any,
        operation_actions: Any,
        all_candidate_features: Any,
        all_candidate_masks: Any,
        selected_indices: Any,
        uav_operation_actions: Any,
        uav_candidate_features: Any,
        uav_candidate_masks: Any,
        uav_selected_indices: Any,
        critic_targets: Any,
    ) -> dict[str, Any]:
        """Train centralized Q_cache first, then freeze it for CF PPO signals."""

        if not self._uses_v11a_cache_node_cf():
            raise RuntimeError("V11-A Cache critic requested for a non-V11-A policy")
        if self.q_cache is None or self.q_cache_optimizer is None:
            raise RuntimeError("V11-A Cache critic optimizer is not initialized")
        observations_t = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        contexts_t = torch.as_tensor(cache_contexts, dtype=torch.float32, device=self.device)
        operation_masks_t = torch.as_tensor(
            operation_masks, dtype=torch.bool, device=self.device
        )
        operation_actions_t = torch.as_tensor(
            operation_actions, dtype=torch.long, device=self.device
        )
        all_features_t = torch.as_tensor(
            all_candidate_features, dtype=torch.float32, device=self.device
        )
        all_masks_t = torch.as_tensor(
            all_candidate_masks, dtype=torch.bool, device=self.device
        )
        selected_indices_t = torch.as_tensor(
            selected_indices, dtype=torch.long, device=self.device
        )
        uav_operation_actions_t = torch.as_tensor(
            uav_operation_actions, dtype=torch.long, device=self.device
        )
        uav_candidate_features_t = torch.as_tensor(
            uav_candidate_features, dtype=torch.float32, device=self.device
        )
        uav_candidate_masks_t = torch.as_tensor(
            uav_candidate_masks, dtype=torch.bool, device=self.device
        )
        uav_selected_indices_t = torch.as_tensor(
            uav_selected_indices, dtype=torch.long, device=self.device
        )
        targets_t = torch.as_tensor(
            critic_targets, dtype=torch.float32, device=self.device
        )
        if observations_t.ndim != 2:
            raise ValueError("V11-A Q_cache observations must be [T,D]")
        sample_count, observation_dim = [int(value) for value in observations_t.shape]
        if contexts_t.ndim != 3 or int(contexts_t.shape[0]) != sample_count:
            raise ValueError("V11-A Cache context tensor shape mismatch")
        node_count = int(operation_actions_t.shape[1])
        operation_count = len(getattr(self.policy, "operation_order", ()))
        if tuple(operation_masks_t.shape) != (sample_count, node_count, operation_count):
            raise ValueError("V11-A Cache operation mask tensor shape mismatch")
        if tuple(operation_actions_t.shape) != tuple(selected_indices_t.shape):
            raise ValueError("V11-A Cache operation/index shape mismatch")
        if tuple(targets_t.shape) != (sample_count,):
            raise ValueError("V11-A Cache critic target shape mismatch")
        if observation_dim != self.q_cache.observation_dim:
            raise ValueError("V11-A Cache critic observation dimension mismatch")
        if not (
            bool(torch.all(torch.isfinite(observations_t)).item())
            and bool(torch.all(torch.isfinite(contexts_t)).item())
            and bool(torch.all(torch.isfinite(targets_t)).item())
        ):
            raise FloatingPointError("V11-A Q_cache state/target inputs contain NaN/Inf")
        joint_action_encoding_t = self._v11a_build_joint_action_encoding(
            cache_operation_actions=operation_actions_t,
            all_candidate_features=all_features_t,
            all_candidate_masks=all_masks_t,
            cache_selected_indices=selected_indices_t,
            uav_operation_actions=uav_operation_actions_t,
            uav_candidate_features=uav_candidate_features_t,
            uav_candidate_masks=uav_candidate_masks_t,
            uav_selected_indices=uav_selected_indices_t,
        )
        hyper = self.hyperparameters
        if sample_count < hyper.num_minibatches or sample_count % hyper.num_minibatches != 0:
            raise ValueError("V11-A Q rollout length must be divisible by num_minibatches")
        minibatch_size = sample_count // hyper.num_minibatches
        q_steps_before = int(self.q_cache_optimizer_steps)
        q_loss_before, q_ev_before = self._v11a_q_selected_metrics(
            observations_t, joint_action_encoding_t, targets_t
        )
        q_epoch_diagnostics: list[dict[str, Any]] = []
        q_raw_gradient_norms: list[float] = []
        q_clipped_gradient_norms: list[float] = []
        for q_epoch in range(hyper.update_epochs):
            permutation = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                q_predictions = self._v11a_selected_q_values(
                    observations_t[indices], joint_action_encoding_t[indices]
                )
                q_target_batch = targets_t[indices, None].expand_as(q_predictions)
                q_loss_tensor = 0.5 * torch.mean((q_predictions - q_target_batch) ** 2)
                if not bool(torch.isfinite(q_loss_tensor).item()):
                    raise FloatingPointError("V11-A Q_cache loss is non-finite")
                self.q_cache_optimizer.zero_grad(set_to_none=True)
                q_loss_tensor.backward()
                for parameter in self.q_cache.parameters():
                    if parameter.grad is not None and not bool(
                        torch.all(torch.isfinite(parameter.grad)).item()
                    ):
                        raise FloatingPointError("V11-A Q_cache gradient is non-finite")
                q_grad_norm_raw = _gradient_group_norm(self.q_cache.parameters())
                if not math.isfinite(q_grad_norm_raw):
                    raise FloatingPointError("V11-A Q_cache raw gradient norm is non-finite")
                clipped_norm = torch.nn.utils.clip_grad_norm_(
                    self.q_cache.parameters(), hyper.max_grad_norm
                )
                if not bool(torch.isfinite(clipped_norm).item()):
                    raise FloatingPointError(
                        "V11-A Q_cache gradient clipping norm is non-finite"
                    )
                q_grad_norm_clipped = _gradient_group_norm(self.q_cache.parameters())
                if not math.isfinite(q_grad_norm_clipped):
                    raise FloatingPointError(
                        "V11-A Q_cache clipped gradient norm is non-finite"
                    )
                self.q_cache_optimizer.step()
                self.q_cache_optimizer_steps += 1
                q_raw_gradient_norms.append(float(q_grad_norm_raw))
                q_clipped_gradient_norms.append(float(q_grad_norm_clipped))
            q_loss_epoch, q_ev_epoch = self._v11a_q_selected_metrics(
                observations_t, joint_action_encoding_t, targets_t
            )
            q_epoch_diagnostics.append(
                {
                    "q_epoch": int(q_epoch + 1),
                    "q_loss": float(q_loss_epoch),
                    "q_ev": float(q_ev_epoch),
                    "q_grad_norm_raw": float(np.mean(q_raw_gradient_norms[-hyper.num_minibatches:])),
                    "q_grad_norm_clipped": float(
                        np.mean(q_clipped_gradient_norms[-hyper.num_minibatches:])
                    ),
                    "q_optimizer_steps": int(self.q_cache_optimizer_steps),
                }
            )
        q_steps_per_rollout = int(self.q_cache_optimizer_steps) - q_steps_before
        expected_q_steps = int(hyper.update_epochs * hyper.num_minibatches)
        if q_steps_per_rollout != expected_q_steps:
            raise RuntimeError(
                "V11-A Q schedule step count mismatch: "
                f"expected={expected_q_steps}, got={q_steps_per_rollout}"
            )
        q_loss_after, q_ev_after = self._v11a_q_selected_metrics(
            observations_t, joint_action_encoding_t, targets_t
        )
        with torch.no_grad():
            all_action_log_probs = self.policy.evaluate_cache_all_action_log_probs(
                contexts_t,
                operation_masks_t,
                all_features_t,
                all_masks_t,
            )
            if tuple(all_action_log_probs.shape) != tuple(all_masks_t.shape):
                raise ValueError("V11-A Cache legal action log-probability shape mismatch")
            legal_log_prob_mask = torch.isfinite(all_action_log_probs) | torch.isneginf(
                all_action_log_probs
            )
            if not bool(torch.all(legal_log_prob_mask).item()):
                raise FloatingPointError("V11-A Cache legal action log-probabilities contain NaN/Inf")
            action_probabilities = torch.where(
                torch.isfinite(all_action_log_probs),
                torch.exp(all_action_log_probs),
                torch.zeros_like(all_action_log_probs),
            )
            probability_sums = action_probabilities.sum(dim=(2, 3))
            probability_sum_error = float(
                torch.max(torch.abs(probability_sums - 1.0)).detach().cpu()
            )
            if not math.isfinite(probability_sum_error) or probability_sum_error > 1.0e-5:
                raise FloatingPointError(
                    "V11-A legal Cache action probabilities do not sum to one: "
                    f"max_abs_error={probability_sum_error}"
                )
            legal_positions = torch.nonzero(all_masks_t, as_tuple=False)
            if int(legal_positions.shape[0]) <= 0:
                raise ValueError("V11-A Cache action support is empty")
            all_q_values = torch.zeros(
                all_masks_t.shape, dtype=torch.float32, device=self.device
            )
            cache_action_dim = int(self.v11a_cache_action_encoding_dim)
            q_chunk_size = 4096
            for start in range(0, int(legal_positions.shape[0]), q_chunk_size):
                positions = legal_positions[start : start + q_chunk_size]
                time_indices = positions[:, 0]
                target_nodes = positions[:, 1]
                alternative_operations = positions[:, 2]
                alternative_indices = positions[:, 3]
                alternative_cache_encoding = torch.cat(
                    [
                        torch.nn.functional.one_hot(
                            alternative_operations,
                            num_classes=operation_count,
                        ).to(dtype=torch.float32),
                        torch.nn.functional.one_hot(
                            alternative_indices,
                            num_classes=int(all_features_t.shape[3]),
                        ).to(dtype=torch.float32),
                        all_features_t[
                            time_indices,
                            target_nodes,
                            alternative_operations,
                            alternative_indices,
                        ],
                    ],
                    dim=-1,
                )
                alternative_joint_encoding = joint_action_encoding_t[time_indices].clone()
                offsets = target_nodes * cache_action_dim
                for local_index in range(int(positions.shape[0])):
                    offset = int(offsets[local_index].item())
                    alternative_joint_encoding[
                        local_index, offset : offset + cache_action_dim
                    ] = alternative_cache_encoding[local_index]
                alternative_q_values = self.q_cache(
                    observations_t[time_indices],
                    target_nodes,
                    alternative_joint_encoding,
                )
                all_q_values[
                    time_indices,
                    target_nodes,
                    alternative_operations,
                    alternative_indices,
                ] = alternative_q_values
            time_grid = torch.arange(sample_count, device=self.device)[:, None]
            node_grid = torch.arange(node_count, device=self.device)[None, :]
            selected_q_values = self._v11a_selected_q_values(
                observations_t, joint_action_encoding_t
            )
            baseline = (action_probabilities * all_q_values).sum(dim=(2, 3))
            cf_raw = selected_q_values - baseline
            valid_node_mask = all_masks_t.sum(dim=(2, 3)) > 1
            valid_values = cf_raw[valid_node_mask]
            if int(valid_values.numel()) <= 0:
                raise ValueError("V11-A rollout has no Cache node with policy freedom")
            cf_raw_mean_tensor = valid_values.mean()
            cf_raw_std_tensor = valid_values.std(unbiased=False)
            cf_scale = cf_raw_std_tensor + 1.0e-8
            cf_normalized = torch.zeros_like(cf_raw)
            cf_normalized[valid_node_mask] = (
                valid_values - cf_raw_mean_tensor
            ) / cf_scale
        if not bool(torch.all(torch.isfinite(cf_raw)).item()) or not bool(
            torch.all(torch.isfinite(cf_normalized)).item()
        ):
            raise FloatingPointError("V11-A CF advantages contain NaN/Inf")
        cf_values = valid_values.detach().cpu().numpy().astype(np.float64)
        cf_normalized_values = cf_normalized[valid_node_mask].detach().cpu().numpy().astype(
            np.float64
        )
        return {
            "q_cache_loss": float(q_loss_after),
            "q_cache_explained_variance": float(q_ev_after),
            "q_cache_grad_norm_raw": float(q_raw_gradient_norms[-1]),
            "q_cache_grad_norm_clipped": float(q_clipped_gradient_norms[-1]),
            "q_cache_optimizer_steps": int(self.q_cache_optimizer_steps),
            "q_cache_optimizer_steps_per_rollout": int(q_steps_per_rollout),
            "q_cache_loss_before": float(q_loss_before),
            "q_cache_loss_after": float(q_loss_after),
            "q_cache_ev_before": float(q_ev_before),
            "q_cache_ev_after": float(q_ev_after),
            "q_cache_epoch_diagnostics": tuple(q_epoch_diagnostics),
            "cf_adv_raw": cf_raw.detach().cpu().numpy().astype(np.float32),
            "cf_adv_normalized": cf_normalized.detach().cpu().numpy().astype(np.float32),
            "cf_adv_valid_mask": valid_node_mask.detach().cpu().numpy().astype(bool),
            "cf_adv_raw_mean": float(cf_values.mean()),
            "cf_adv_raw_std": float(cf_values.std()),
            "cf_adv_normalized_mean": float(cf_normalized_values.mean()),
            "cf_adv_normalized_std": float(cf_normalized_values.std()),
            "cf_adv_min": float(cf_values.min()),
            "cf_adv_max": float(cf_values.max()),
            "cf_adv_positive_fraction": float(np.mean(cf_normalized_values > 0.0)),
            "cf_adv_negative_fraction": float(np.mean(cf_normalized_values < 0.0)),
            "cache_action_probability_sum_max_abs_error": probability_sum_error,
            "q_cache_selected_values": selected_q_values.detach().cpu().numpy().astype(
                np.float32
            ),
            "q_cache_counterfactual_baseline": baseline.detach().cpu().numpy().astype(
                np.float32
            ),
            "cache_node_old_q_values": selected_q_values.detach().cpu().numpy().astype(
                np.float32
            ),
        }

    def run_v11a_q_readiness(self, rollout: SimpleRollout) -> dict[str, Any]:
        """Run the bounded frozen-policy Q readiness gate on one rollout."""

        if not self._uses_v11a_cache_node_cf():
            raise RuntimeError("V11-A Q readiness requested for a non-V11-A policy")
        if any(parameter.requires_grad for parameter in self.policy.parameters()):
            raise RuntimeError("V11-A Q readiness requires Actor/global critic parameters frozen")
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
            self.hyperparameters.gamma,
            self.hyperparameters.gae_lambda,
        )
        if not np.all(np.isfinite(advantages)) or not np.all(np.isfinite(returns)):
            raise FloatingPointError("V11-A Q readiness GAE targets contain NaN/Inf")
        if self._uses_popart_critic():
            readiness_statistics = RunningReturnStatistics.from_checkpoint(
                self.critic_return_statistics.count,
                self.critic_return_statistics.mean,
                self.critic_return_statistics.m2,
            )
            readiness_statistics.update(returns)
            target_mean, target_scale = readiness_statistics.active_statistics(
                self.hyperparameters.critic_value_scale_epsilon,
                self.hyperparameters.critic_value_scale_warmup_samples,
            )
            critic_targets = (returns - target_mean) / target_scale
        else:
            critic_targets = returns
        if not np.all(np.isfinite(critic_targets)):
            raise FloatingPointError("V11-A Q readiness targets contain NaN/Inf")
        if (
            rollout.v7_cache_operation_masks is None
            or rollout.v7_cache_operation_actions is None
            or rollout.cache_all_candidate_features is None
            or rollout.cache_all_candidate_masks is None
            or rollout.uav_operation_actions is None
        ):
            raise ValueError("V11-A Q readiness rollout is missing joint action tensors")
        return self._train_v11a_cache_node_critic(
            observations=rollout.observations,
            cache_contexts=rollout.cache_contexts,
            operation_masks=rollout.v7_cache_operation_masks,
            operation_actions=rollout.v7_cache_operation_actions,
            all_candidate_features=rollout.cache_all_candidate_features,
            all_candidate_masks=rollout.cache_all_candidate_masks,
            selected_indices=rollout.cache_selected_indices,
            uav_operation_actions=rollout.uav_operation_actions,
            uav_candidate_features=rollout.uav_candidate_features,
            uav_candidate_masks=rollout.uav_candidate_masks,
            uav_selected_indices=rollout.uav_selected_indices,
            critic_targets=critic_targets,
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
        if self.context.data_protocol == project_config.PPO_DATA_PROTOCOL_FULL_EPISODE_RANDOMIZED:
            if start != self.context.train_origin_start:
                raise ValueError(
                    "full_episode_randomized 必须从 train_origin_start 开始收集完整 episode。"
                )
            if int(rollout_steps) != self.context.episode_slots:
                raise ValueError(
                    "full_episode_randomized rollout_steps 必须等于 episode_slots。"
                )
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

    def _evaluate_cache_branch_approx_kl_full_rollout(
        self,
        *,
        observations: Any,
        cache_contexts: Any,
        operation_masks: Any,
        operation_actions: Any,
        candidate_features: Any,
        candidate_masks: Any,
        selected_indices: Any,
        old_operation_log_probs: Any,
        old_candidate_log_probs: Any,
    ) -> float:
        """Evaluate the production sampled Cache branch KL on one full rollout.

        This is the same estimator used by the existing per-minibatch branch
        diagnostics, but it is evaluated on the complete frozen rollout after
        an actor optimizer step so the minibatch control observes cumulative
        policy drift rather than a pre-step minibatch-local value.
        """

        with torch.no_grad():
            evaluation = self.policy.evaluate_cache_operation_primitive(
                observations,
                cache_contexts,
                operation_masks,
                operation_actions,
                candidate_features,
                candidate_masks,
                selected_indices,
            )
            if evaluation.cache_log_prob is None:
                raise RuntimeError(
                    "Cache branch KL evaluation did not expose cache_log_prob"
                )
            old_cache_log_prob = (
                old_operation_log_probs + old_candidate_log_probs
            ).sum(dim=-1)
            delta = evaluation.cache_log_prob - old_cache_log_prob
            if not bool(torch.all(torch.isfinite(delta)).item()):
                raise FloatingPointError(
                    "Cache full-rollout KL estimator produced NaN/Inf"
                )
            value = float(torch.mean(-delta).detach().cpu())
        if not math.isfinite(value):
            raise FloatingPointError("Cache full-rollout KL estimator is non-finite")
        return value

    def _compute_r2_operation_credit(self, rollout: SimpleRollout) -> dict[str, Any]:
        """Compute detached per-UAV counterfactual one-step TD credit.

        This is intentionally a trainer-side replay of the already collected
        primitive transition.  Cache actions and every non-target UAV action
        are copied from production; only the target operation is replaced.
        Each branch goes through the real ``execute_slot`` path with a direct
        primitive configuration override, so Matching/KKT and all frozen
        reward/cost semantics remain authoritative.
        """

        if not self._uses_r2_operation_credit():
            return {}
        transitions = tuple(rollout.transitions)
        if not transitions:
            raise ValueError("R2 operation credit requires a non-empty rollout")
        uav_ids = tuple(str(value) for value in self.context.action_spec.uav_ids)
        head_count = len(uav_ids)
        sample_count = len(transitions)
        for transition in transitions:
            if (
                transition.previous_cache_state is None
                or transition.previous_uav_positions is None
                or transition.uav_operation_actions is None
                or transition.uav_operation_masks is None
                or transition.uav_operation_probabilities is None
                or transition.uav_contexts is None
                or transition.uav_candidate_features is None
            ):
                raise ValueError("R2 rollout is missing operation counterfactual tensors")

        include_bootstrap_step = not bool(transitions[-1].done)
        aligned_steps = aligned_train_steps(
            self.context,
            origin_start=int(transitions[0].origin_index),
            max_steps=sample_count + int(include_bootstrap_step),
            include_bootstrap_step=include_bootstrap_step,
        )
        if len(aligned_steps) < sample_count + int(include_bootstrap_step):
            raise RuntimeError("R2 counterfactual replay is missing aligned next steps")
        for transition, step in zip(transitions, aligned_steps[:sample_count]):
            if int(transition.origin_index) != int(step.origin_index):
                raise RuntimeError("R2 counterfactual replay origin alignment failed")

        q_stay = np.zeros((sample_count, head_count), dtype=np.float64)
        q_move = np.zeros((sample_count, head_count), dtype=np.float64)
        baseline = np.zeros((sample_count, head_count), dtype=np.float64)
        move_candidate_counts = np.zeros((sample_count, head_count), dtype=np.int64)
        counterfactual_environment_executions = 0
        started_at = time.perf_counter()

        def _position_dict(source: dict[str, Any]) -> dict[str, np.ndarray]:
            return {
                uav_id: np.asarray(source[uav_id], dtype=np.float64).copy()
                for uav_id in uav_ids
            }

        def _primitive_action(
            uav_id: str,
            head_index: int,
            candidate_count: int,
            operation: str,
            before: np.ndarray,
            after: np.ndarray,
            selected_index: int | None,
        ) -> dict[str, Any]:
            return {
                "head_type": "uav",
                "head_id": uav_id,
                "head_index": int(head_index),
                "candidate_count": int(candidate_count),
                "selected_index": selected_index,
                "operation_type": operation,
                "before_position": [float(value) for value in before],
                "after_position": [float(value) for value in after],
                "actual_moved": bool(operation == "MOVE"),
            }

        for transition_index, (transition, step) in enumerate(
            zip(transitions, aligned_steps[:sample_count])
        ):
            previous_cache = np.asarray(
                transition.previous_cache_state, dtype=np.int8
            ).copy()
            previous_uav = _position_dict(transition.previous_uav_positions)
            production_final_uav = _position_dict(transition.next_uav_positions)
            production_cache_triggers, production_uav_triggers = (
                self.context.action_spec.action_vector_to_triggers(transition.action)
            )
            production_cache_triggers = np.asarray(
                production_cache_triggers, dtype=np.int8
            )
            production_uav_triggers = {
                str(key): int(value)
                for key, value in production_uav_triggers.items()
            }
            production_cache_actions = tuple(
                dict(value) for value in transition.cache_conditional_actions
            )
            production_uav_actions = [
                dict(value) for value in transition.uav_conditional_actions
            ]
            if len(production_uav_actions) != head_count:
                raise RuntimeError("R2 production UAV action count mismatch")

            for head_index, uav_id in enumerate(uav_ids):
                # Reconstruct exactly the sequential conditional state used
                # by the production sampler for this head.
                current_uav = _position_dict(transition.previous_uav_positions)
                enumeration_triggers = {candidate_id: 0 for candidate_id in uav_ids}
                for prior_index, prior_id in enumerate(uav_ids[:head_index]):
                    prior_action = production_uav_actions[prior_index]
                    if str(prior_action.get("operation_type")) == "MOVE":
                        current_uav[prior_id] = np.asarray(
                            prior_action["after_position"], dtype=np.float64
                        )
                        enumeration_triggers[prior_id] = 1
                enumeration_triggers[uav_id] = 1
                candidates = uav_configuration.enumerate_uav_primitive_actions(
                    current_uav,
                    previous_uav,
                    uav_id,
                    enumeration_triggers,
                    self.context.cfg,
                    self.context.flight_boundary,
                )
                move_candidates = [
                    candidate
                    for candidate in candidates
                    if str(candidate.operation_type) == "MOVE"
                ]
                move_count = len(move_candidates)
                move_candidate_counts[transition_index, head_index] = move_count
                if move_count > int(transition.uav_candidate_features.shape[1]):
                    raise RuntimeError("R2 legal UAV candidates exceed rollout capacity")

                raw_probabilities = np.asarray(
                    transition.uav_operation_probabilities[head_index],
                    dtype=np.float64,
                ).reshape(-1)
                operation_mask = np.asarray(
                    transition.uav_operation_masks[head_index], dtype=bool
                ).reshape(-1)
                if raw_probabilities.shape != (2,) or operation_mask.shape != (2,):
                    raise RuntimeError("R2 UAV operation probability/mask shape mismatch")
                if move_count > 0 and bool(operation_mask[1]):
                    probability_sum = float(raw_probabilities.sum())
                    if (
                        not np.all(np.isfinite(raw_probabilities))
                        or probability_sum <= 0.0
                    ):
                        raise FloatingPointError(
                            "R2 UAV operation probabilities are non-finite"
                        )
                    p_stay, p_move = (
                        raw_probabilities / probability_sum
                    ).tolist()
                else:
                    # The masked production operation distribution has only
                    # STAY when no legal MOVE exists.
                    p_stay, p_move = 1.0, 0.0

                if move_count:
                    with torch.no_grad():
                        candidate_sample = self.policy.sample_uav_candidate(
                            transition.uav_contexts[head_index],
                            np.asarray(
                                transition.uav_candidate_features[
                                    head_index, :move_count
                                ],
                                dtype=np.float32,
                            ),
                            deterministic=False,
                        )
                        candidate_probabilities = torch.softmax(
                            candidate_sample.logits, dim=-1
                        ).detach().cpu().numpy().astype(np.float64)
                    if (
                        candidate_probabilities.shape != (move_count,)
                        or not np.all(np.isfinite(candidate_probabilities))
                        or not np.isclose(
                            float(candidate_probabilities.sum()), 1.0, atol=1.0e-5
                        )
                    ):
                        raise FloatingPointError(
                            "R2 candidate probabilities are non-finite or do not sum to one"
                        )
                else:
                    candidate_probabilities = np.zeros(0, dtype=np.float64)

                next_step = (
                    None
                    if bool(transition.done)
                    else aligned_steps[transition_index + 1]
                )

                def _evaluate_branch(
                    operation: str,
                    selected_index: int | None,
                    before: np.ndarray,
                    after: np.ndarray,
                ) -> float:
                    nonlocal counterfactual_environment_executions
                    branch_final_uav = {
                        key: value.copy() for key, value in production_final_uav.items()
                    }
                    branch_final_uav[uav_id] = np.asarray(after, dtype=np.float64).copy()
                    branch_uav_triggers = dict(production_uav_triggers)
                    branch_uav_triggers[uav_id] = int(operation == "MOVE")
                    branch_uav_actions = list(production_uav_actions)
                    branch_uav_actions[head_index] = _primitive_action(
                        uav_id,
                        head_index,
                        move_count,
                        operation,
                        np.asarray(before, dtype=np.float64),
                        np.asarray(after, dtype=np.float64),
                        selected_index,
                    )
                    configuration_override = (
                        slot_environment.build_policy_primitive_configuration(
                            previous_cache,
                            previous_uav,
                            transition.next_cache_state,
                            branch_final_uav,
                            production_cache_triggers,
                            branch_uav_triggers,
                            self.context.service_catalog,
                            self.context.cfg,
                            self.context.cache_nodes,
                            self.context.cache_cfg,
                            self.context.flight_boundary,
                            cache_actions=production_cache_actions,
                            uav_actions=tuple(branch_uav_actions),
                        )
                    )
                    result = slot_environment.execute_slot(
                        previous_cache,
                        previous_uav,
                        production_cache_triggers,
                        branch_uav_triggers,
                        step.scenario_slice,
                        step.actual_tasks,
                        step.actual_vehicle_positions,
                        self.context.local_cache,
                        self.context.service_catalog,
                        self.context.cfg,
                        self.context.cache_nodes,
                        self.context.cache_cfg,
                        self.context.flight_boundary,
                        slot_id=step.slot,
                        time_s=step.time_s,
                        performance_profile=None,
                        configuration_profile=cache_configuration.ConfigurationEvaluationProfile(),
                        configuration_override=configuration_override,
                    )
                    counterfactual_environment_executions += 1
                    if not result.step_feasible:
                        raise RuntimeError(
                            "R2 counterfactual transition is infeasible: "
                            f"origin={step.origin_index}, UAV={uav_id}, operation={operation}"
                        )
                    _check_request_contract(
                        previous_cache,
                        previous_uav,
                        result,
                        self.context.action_spec,
                    )
                    reward = _check_reward(result)
                    if next_step is None:
                        next_value = 0.0
                    else:
                        next_observation = _intermediate_observation(
                            np.asarray(result.next_cache_state, dtype=np.int8),
                            result.next_uav_positions,
                            next_step,
                            self.context,
                        )
                        with torch.no_grad():
                            _, value_tensor = self.policy(next_observation)
                        next_value = float(value_tensor.detach().cpu().reshape(-1)[0])
                    q_value = reward + float(self.hyperparameters.gamma) * next_value
                    if not math.isfinite(q_value):
                        raise FloatingPointError("R2 counterfactual Q is non-finite")
                    return float(q_value)

                target_before = current_uav[uav_id].copy()
                stay_q = _evaluate_branch(
                    "STAY",
                    None,
                    target_before,
                    target_before,
                )
                q_stay[transition_index, head_index] = stay_q
                if move_count:
                    move_values = np.asarray(
                        [
                            _evaluate_branch(
                                "MOVE",
                                candidate_index,
                                np.asarray(candidate.before_position, dtype=np.float64),
                                np.asarray(candidate.after_position, dtype=np.float64),
                            )
                            for candidate_index, candidate in enumerate(move_candidates)
                        ],
                        dtype=np.float64,
                    )
                    q_move[transition_index, head_index] = float(
                        np.dot(candidate_probabilities, move_values)
                    )
                else:
                    # No MOVE branch exists, so its counterfactual value is
                    # made equal to STAY and the masked MOVE probability is 0.
                    q_move[transition_index, head_index] = stay_q
                baseline[transition_index, head_index] = (
                    float(p_stay) * q_stay[transition_index, head_index]
                    + float(p_move) * q_move[transition_index, head_index]
                )

        sampled_operations = np.asarray(
            [
                np.asarray(transition.uav_operation_actions, dtype=np.int64)
                for transition in transitions
            ],
            dtype=np.int64,
        )
        raw_advantages = np.where(
            sampled_operations == 0,
            q_stay - baseline,
            q_move - baseline,
        )
        if not np.all(np.isfinite(raw_advantages)):
            raise FloatingPointError("R2 operation advantages are non-finite")
        population_std = float(np.std(raw_advantages, ddof=0))
        if not math.isfinite(population_std):
            raise FloatingPointError("R2 operation advantage std is non-finite")
        scaled_advantages = raw_advantages / (population_std + 1.0e-8)
        nonzero = raw_advantages != 0.0
        sign_flip_count = int(
            np.count_nonzero(
                nonzero
                & (np.signbit(raw_advantages) != np.signbit(scaled_advantages))
            )
        )
        if sign_flip_count != 0:
            raise FloatingPointError("R2 std-only operation scaling flipped a sign")
        if not np.all(np.isfinite(scaled_advantages)):
            raise FloatingPointError("R2 scaled operation advantages are non-finite")
        return {
            "raw_advantages": raw_advantages.astype(np.float32),
            "scaled_advantages": scaled_advantages.astype(np.float32),
            "q_stay": q_stay.astype(np.float32),
            "q_move": q_move.astype(np.float32),
            "baseline": baseline.astype(np.float32),
            "counterfactual_environment_executions": int(
                counterfactual_environment_executions
            ),
            "mean_move_candidate_count": float(np.mean(move_candidate_counts)),
            "move_candidate_counts": move_candidate_counts,
            "credit_runtime_seconds": float(time.perf_counter() - started_at),
            "advantage_normalization": "STD_ONLY",
            "advantage_centering": False,
            "advantage_sign_flip_count": sign_flip_count,
            "ratio_granularity": "OPERATION_ONLY",
        }

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
        popart_critic = self._uses_popart_critic()
        critic_target_mean = 0.0
        if not popart_critic:
            critic_return_scale = self._critic_return_scale()
            if not np.isfinite(critic_return_scale) or critic_return_scale <= 0.0:
                raise FloatingPointError("PPO v2 critic return scale 非 finite。")
        sample_count = int(rollout.observations.shape[0])
        if self.context.data_protocol == project_config.PPO_DATA_PROTOCOL_FULL_EPISODE_RANDOMIZED:
            if sample_count != self.context.episode_slots:
                raise ValueError(
                    "full_episode_randomized PPO update 必须使用完整 episode："
                    f"expected={self.context.episode_slots}, got={sample_count}。"
                )
            if sample_count // hyper.num_minibatches != self.context.minibatch_size:
                raise ValueError(
                    "full_episode_randomized minibatch_size 与 protocol 不一致。"
                )
        if sample_count < hyper.num_minibatches or sample_count % hyper.num_minibatches != 0:
            raise ValueError("PPO v2 rollout length 必须可被 num_minibatches 整除。")
        if popart_critic:
            if not isinstance(
                self.policy,
                AdaptivePrimitiveHorizonGroupedSeparateCriticPopArtV6ActorCritic,
            ):
                raise TypeError("PopArt update requires the V6 actor-critic policy.")
            expected_mean, expected_std = self.critic_return_statistics.active_statistics(
                hyper.critic_value_scale_epsilon,
                hyper.critic_value_scale_warmup_samples,
            )
            active_mean, active_std = self.policy.popart_statistics()
            if not math.isclose(active_mean, expected_mean, rel_tol=1.0e-6, abs_tol=1.0e-6) or not math.isclose(
                active_std, expected_std, rel_tol=1.0e-6, abs_tol=1.0e-6
            ):
                raise RuntimeError("V6 PopArt policy statistics are out of sync with Welford state.")
            # The trainer's Welford state is the sole statistics source.  This
            # is the one and only update for this completed raw-return batch.
            self.critic_return_statistics.update(returns)
            critic_target_mean, critic_return_scale = self.critic_return_statistics.active_statistics(
                hyper.critic_value_scale_epsilon,
                hyper.critic_value_scale_warmup_samples,
            )
            self.policy.update_popart_statistics(critic_target_mean, critic_return_scale)
            if not np.isfinite(critic_return_scale) or critic_return_scale <= 0.0:
                raise FloatingPointError("PPO v2 critic return scale 非 finite。")
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
        v11a_enabled = self._uses_v11a_cache_node_cf()
        cache_all_candidate_features = None
        cache_all_candidate_masks = None
        if v11a_enabled:
            if (
                rollout.cache_all_candidate_features is None
                or rollout.cache_all_candidate_masks is None
            ):
                raise ValueError(
                    "V11-A rollout is missing complete legal Cache candidate groups"
                )
            cache_all_candidate_features = torch.as_tensor(
                rollout.cache_all_candidate_features,
                dtype=torch.float32,
                device=self.device,
            )
            cache_all_candidate_masks = torch.as_tensor(
                rollout.cache_all_candidate_masks,
                dtype=torch.bool,
                device=self.device,
            )
        v7_operation_actions = None
        v7_operation_masks = None
        v7_operation_old_log_probs = None
        if self.context.configuration_control_mode in {
            "policy_primitive_v7_cache_operation",
            "policy_primitive_v9_joint_cache_uav_operation",
        }:
            if (
                rollout.v7_cache_operation_actions is None
                or rollout.v7_cache_operation_masks is None
            ):
                raise ValueError("Cache-operation rollout is missing operation actions/masks")
            v7_operation_actions = torch.as_tensor(
                rollout.v7_cache_operation_actions,
                dtype=torch.long,
                device=self.device,
            )
            v7_operation_masks = torch.as_tensor(
                rollout.v7_cache_operation_masks,
                dtype=torch.bool,
                device=self.device,
            )
            if rollout.v7_cache_operation_log_probs is not None:
                v7_operation_old_log_probs = torch.as_tensor(
                    rollout.v7_cache_operation_log_probs,
                    dtype=torch.float32,
                    device=self.device,
                )
        uav_operation_actions = None
        uav_operation_masks = None
        uav_operation_old_log_probs = None
        if self.context.configuration_control_mode == "policy_primitive_v9_joint_cache_uav_operation":
            if rollout.uav_operation_actions is None or rollout.uav_operation_masks is None:
                raise ValueError("V9 rollout is missing UAV operation actions/masks")
            uav_operation_actions = torch.as_tensor(
                rollout.uav_operation_actions,
                dtype=torch.long,
                device=self.device,
            )
            uav_operation_masks = torch.as_tensor(
                rollout.uav_operation_masks,
                dtype=torch.bool,
                device=self.device,
            )
            if rollout.uav_operation_log_probs is not None:
                uav_operation_old_log_probs = torch.as_tensor(
                    rollout.uav_operation_log_probs,
                    dtype=torch.float32,
                    device=self.device,
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
        r2_enabled = self._uses_r2_operation_credit()
        branch_diagnostics_enabled = (
            self.policy.architecture_id in JOINT_CACHE_UAV_POLICY_SCHEMAS
        )
        branchwise_policy_enabled = (
            self.policy.architecture_id in JOINT_V10_BRANCHWISE_POLICY_SCHEMAS
        )
        v10c_kl_instrumentation_enabled = (
            self.policy.architecture_id in JOINT_V10C_DECOUPLED_POLICY_SCHEMAS
            or self.policy.architecture_id in JOINT_V10D_BRANCH_SPECIFIC_KL_POLICY_SCHEMAS
        )
        branch_specific_kl_gate_enabled = (
            self.policy.architecture_id in JOINT_V10D_BRANCH_SPECIFIC_KL_POLICY_SCHEMAS
            and self.kl_gate_mode == KL_GATE_MODE_BRANCH_SPECIFIC
        )
        cache_minibatch_kl_control_enabled = (
            self.cache_kl_control == CACHE_KL_CONTROL_MINIBATCH_EARLY_STOP
            and v11a_enabled
            and branch_diagnostics_enabled
        )
        cache_candidate_old_log_probs = None
        uav_candidate_old_log_probs = None
        if branch_diagnostics_enabled:
            if (
                v7_operation_old_log_probs is None
                or uav_operation_old_log_probs is None
                or rollout.cache_old_log_probs is None
                or rollout.uav_old_log_probs is None
            ):
                raise ValueError(
                    "Joint Cache/UAV rollout is missing branch old log-probabilities"
                )
            cache_candidate_old_log_probs = torch.as_tensor(
                rollout.cache_old_log_probs,
                dtype=torch.float32,
                device=self.device,
            )
            uav_candidate_old_log_probs = torch.as_tensor(
                rollout.uav_old_log_probs,
                dtype=torch.float32,
                device=self.device,
            )
            if tuple(v7_operation_old_log_probs.shape) != tuple(
                v7_operation_actions.shape
            ):
                raise ValueError(
                    "Joint Cache rollout operation old log-probabilities shape mismatch"
                )
            if tuple(cache_candidate_old_log_probs.shape) != tuple(
                cache_selected_indices.shape
            ):
                raise ValueError(
                    "Joint Cache candidate old log-probabilities shape mismatch"
                )
            if tuple(uav_operation_old_log_probs.shape) != tuple(
                uav_operation_actions.shape
            ):
                raise ValueError(
                    "Joint UAV operation old log-probabilities shape mismatch"
                )
            if tuple(uav_candidate_old_log_probs.shape) != tuple(
                uav_selected_indices.shape
            ):
                raise ValueError(
                    "Joint UAV candidate old log-probabilities shape mismatch"
                )
        r2_operation_credit: dict[str, Any] = {}
        if r2_enabled:
            if not branch_diagnostics_enabled:
                raise RuntimeError("R2 operation credit requires joint branch diagnostics")
            r2_operation_credit = self._compute_r2_operation_credit(rollout)
        cache_kl_control_previous: float | None = None
        if cache_minibatch_kl_control_enabled:
            if (
                v7_operation_old_log_probs is None
                or cache_candidate_old_log_probs is None
                or v7_operation_masks is None
                or v7_operation_actions is None
            ):
                raise RuntimeError(
                    "minibatch Cache KL control is missing Cache old action tensors"
                )
            cache_kl_control_previous = self._evaluate_cache_branch_approx_kl_full_rollout(
                observations=observations,
                cache_contexts=cache_contexts,
                operation_masks=v7_operation_masks,
                operation_actions=v7_operation_actions,
                candidate_features=cache_candidate_features,
                candidate_masks=cache_candidate_masks,
                selected_indices=cache_selected_indices,
                old_operation_log_probs=v7_operation_old_log_probs,
                old_candidate_log_probs=cache_candidate_old_log_probs,
            )
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        r2_operation_advantages_scaled_t = None
        if r2_enabled:
            r2_operation_advantages_scaled_t = torch.as_tensor(
                r2_operation_credit["scaled_advantages"],
                dtype=torch.float32,
                device=self.device,
            )
            if tuple(r2_operation_advantages_scaled_t.shape) != tuple(
                uav_operation_actions.shape
            ):
                raise RuntimeError("R2 operation advantage shape mismatch")
        critic_targets_t = (
            (returns_t - critic_target_mean) / critic_return_scale
            if popart_critic
            else returns_t
        )
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
        # Keep exactly the tensor used by the actor loss.  The copy is made
        # after the configured global normalization and is only returned for
        # diagnostics; no second or heuristic advantage is constructed.
        ppo_advantages = advantages_t.detach().cpu().numpy().astype(
            np.float32, copy=True
        )

        v11a_credit: dict[str, Any] = {}
        if v11a_enabled:
            if (
                v7_operation_masks is None
                or v7_operation_actions is None
                or cache_all_candidate_features is None
                or cache_all_candidate_masks is None
                or uav_operation_actions is None
            ):
                raise RuntimeError("V11-A Cache operation tensors are incomplete")
            v11a_credit = self._train_v11a_cache_node_critic(
                observations=observations,
                cache_contexts=cache_contexts,
                operation_masks=v7_operation_masks,
                operation_actions=v7_operation_actions,
                all_candidate_features=cache_all_candidate_features,
                all_candidate_masks=cache_all_candidate_masks,
                selected_indices=cache_selected_indices,
                uav_operation_actions=uav_operation_actions,
                uav_candidate_features=uav_candidate_features,
                uav_candidate_masks=uav_candidate_masks,
                uav_selected_indices=uav_selected_indices,
                critic_targets=critic_targets_t,
            )
        cf_adv_normalized_t = None
        cf_adv_valid_mask_t = None
        cache_node_logprob_sum_max_abs_error = math.nan
        if v11a_enabled:
            cf_adv_normalized_t = torch.as_tensor(
                v11a_credit["cf_adv_normalized"],
                dtype=torch.float32,
                device=self.device,
            )
            cf_adv_valid_mask_t = torch.as_tensor(
                v11a_credit["cf_adv_valid_mask"],
                dtype=torch.bool,
                device=self.device,
            )
            if tuple(cf_adv_normalized_t.shape) != tuple(cache_selected_indices.shape):
                raise ValueError("V11-A CF advantage shape mismatch")
            if tuple(cf_adv_valid_mask_t.shape) != tuple(cache_selected_indices.shape):
                raise ValueError("V11-A CF valid-mask shape mismatch")

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
        cache_approx_kl_values: list[float] = []
        uav_approx_kl_values: list[float] = []
        cache_clip_fraction_values: list[float] = []
        uav_clip_fraction_values: list[float] = []
        cache_entropy_values: list[float] = []
        uav_entropy_values: list[float] = []
        cache_head_gradient_norm_values: list[float] = []
        uav_head_gradient_norm_values: list[float] = []
        cache_encoder_gradient_norm_values: list[float] = []
        uav_encoder_gradient_norm_values: list[float] = []
        shared_actor_gradient_norm_values: list[float] = []
        kl_minibatch_diagnostics: list[dict[str, Any]] = []
        kl_epoch_diagnostics: list[dict[str, Any]] = []
        cache_policy_loss_values: list[float] = []
        uav_policy_loss_values: list[float] = []
        actor_policy_loss_values: list[float] = []
        cache_node_logprob_sum_errors: list[float] = []
        gradient_finite = True
        actor_parameters: list[Any] | None = None
        critic_parameters: list[Any] | None = None
        branch_gradient_parameter_groups: dict[str, tuple[Any, ...]] | None = None
        kl_early_stop_triggered = False
        ppo_update_epochs_executed = 0
        cache_active = True
        uav_active = True
        cache_active_per_update_epoch: list[bool] = []
        uav_active_per_update_epoch: list[bool] = []
        cache_stop_epoch: int | None = None
        uav_stop_epoch: int | None = None
        cache_suppressed_update_epochs = 0
        uav_suppressed_update_epochs = 0
        cache_independent_stop = False
        uav_independent_stop = False
        both_inactive_stop = False
        cache_stop_minibatch: int | None = None
        cache_kl_before_stop: float | None = None
        cache_kl_after_stop: float | None = None
        cache_minibatches_executed = 0
        uav_minibatches_executed = 0
        global_critic_steps = 0
        cache_minibatches_planned = int(hyper.update_epochs * hyper.num_minibatches)
        uav_branch_old_log_probs_values: np.ndarray | None = None
        uav_branch_new_log_probs_values: np.ndarray | None = None
        uav_branch_ratios_values: np.ndarray | None = None
        uav_branch_clipped_values: np.ndarray | None = None
        if branch_diagnostics_enabled:
            uav_branch_old_log_probs_values = np.full(
                sample_count, np.nan, dtype=np.float64
            )
            uav_branch_new_log_probs_values = np.full(
                sample_count, np.nan, dtype=np.float64
            )
            uav_branch_ratios_values = np.full(
                sample_count, np.nan, dtype=np.float64
            )
            uav_branch_clipped_values = np.zeros(sample_count, dtype=bool)
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
            if branch_diagnostics_enabled:
                group_builder = getattr(
                    self.policy,
                    "actor_gradient_parameter_groups",
                    None,
                )
                if group_builder is None:
                    raise RuntimeError(
                        "Joint Cache/UAV policy does not expose branch gradient groups"
                    )
                branch_gradient_parameter_groups = group_builder()
        minibatch_size = sample_count // hyper.num_minibatches
        for update_epoch in range(hyper.update_epochs):
            cache_active_this_epoch = bool(cache_active)
            uav_active_this_epoch = bool(uav_active)
            if branch_specific_kl_gate_enabled:
                cache_active_per_update_epoch.append(cache_active_this_epoch)
                uav_active_per_update_epoch.append(uav_active_this_epoch)
            permutation = torch.randperm(sample_count, device=self.device)
            update_epoch_approximate_kls: list[float] = []
            update_epoch_cache_approx_kls: list[float] = []
            update_epoch_uav_approx_kls: list[float] = []
            epoch_diagnostic_start = len(kl_minibatch_diagnostics)
            for start in range(0, sample_count, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                cache_active_this_minibatch = bool(cache_active)
                if self.context.configuration_control_mode == "policy_primitive_v9_joint_cache_uav_operation":
                    if not isinstance(
                        self.policy,
                        AdaptiveJointCacheReplicaAwareUAVOperationV9PopArtActorCritic,
                    ):
                        raise TypeError("V9/V10 update requires a Joint policy")
                    assert v7_operation_actions is not None
                    assert v7_operation_masks is not None
                    assert uav_operation_actions is not None
                    assert uav_operation_masks is not None
                    evaluation = self.policy.evaluate_joint_cache_uav_operation(
                        observations[indices],
                        cache_contexts[indices],
                        v7_operation_masks[indices],
                        v7_operation_actions[indices],
                        cache_candidate_features[indices],
                        cache_candidate_masks[indices],
                        cache_selected_indices[indices],
                        uav_contexts[indices],
                        uav_operation_masks[indices],
                        uav_operation_actions[indices],
                        uav_candidate_features[indices],
                        uav_candidate_masks[indices],
                        uav_selected_indices[indices],
                    )
                elif self.context.configuration_control_mode == "policy_primitive_v7_cache_operation":
                    if not isinstance(
                        self.policy,
                        AdaptiveCacheOperationGroupV7PopArtActorCritic,
                    ):
                        raise TypeError("V7 update requires the V7 Cache policy")
                    assert v7_operation_actions is not None
                    assert v7_operation_masks is not None
                    evaluation = self.policy.evaluate_cache_operation_primitive(
                        observations[indices],
                        cache_contexts[indices],
                        v7_operation_masks[indices],
                        v7_operation_actions[indices],
                        cache_candidate_features[indices],
                        cache_candidate_masks[indices],
                        cache_selected_indices[indices],
                    )
                elif self.context.configuration_control_mode == "policy_primitive":
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
                branch_policy_loss = None
                cache_branch_policy_loss = None
                uav_branch_policy_loss = None
                if branch_diagnostics_enabled:
                    if (
                        evaluation.cache_log_prob is None
                        or evaluation.uav_log_prob is None
                        or evaluation.cache_entropy is None
                        or evaluation.uav_entropy is None
                        or cache_candidate_old_log_probs is None
                        or uav_candidate_old_log_probs is None
                        or v7_operation_old_log_probs is None
                        or uav_operation_old_log_probs is None
                        or (v11a_enabled and evaluation.cache_node_log_probs is None)
                        or (
                            r2_enabled
                            and (
                                evaluation.uav_operation_log_probs is None
                                or evaluation.uav_candidate_log_probs is None
                            )
                        )
                    ):
                        raise RuntimeError(
                            "Joint Cache/UAV evaluation did not expose branch diagnostics"
                        )
                    cache_old_log_prob = (
                        v7_operation_old_log_probs[indices].sum(dim=-1)
                        + cache_candidate_old_log_probs[indices].sum(dim=-1)
                    )
                    cache_node_old_log_prob = (
                        v7_operation_old_log_probs[indices]
                        + cache_candidate_old_log_probs[indices]
                    )
                    if v11a_enabled:
                        assert evaluation.cache_node_log_probs is not None
                        cache_node_sum_error = float(
                            torch.max(
                                torch.abs(
                                    evaluation.cache_node_log_probs.sum(dim=-1)
                                    - evaluation.cache_log_prob
                                )
                            ).detach().cpu()
                        )
                        if not math.isfinite(cache_node_sum_error):
                            raise FloatingPointError(
                                "V11-A Cache node log-probability sum is non-finite"
                            )
                        cache_node_logprob_sum_errors.append(cache_node_sum_error)
                        if not bool(
                            torch.all(torch.isfinite(cache_node_old_log_prob)).item()
                        ):
                            raise FloatingPointError(
                                "V11-A old Cache node log-probabilities are non-finite"
                            )
                    if r2_enabled:
                        assert evaluation.uav_operation_log_probs is not None
                        uav_old_log_prob = uav_operation_old_log_probs[indices].sum(
                            dim=-1
                        )
                        uav_new_log_prob = evaluation.uav_operation_log_probs.sum(
                            dim=-1
                        )
                    else:
                        uav_old_log_prob = (
                            uav_operation_old_log_probs[indices].sum(dim=-1)
                            + uav_candidate_old_log_probs[indices].sum(dim=-1)
                        )
                        uav_new_log_prob = evaluation.uav_log_prob
                    cache_branch_delta = evaluation.cache_log_prob - cache_old_log_prob
                    uav_branch_delta = uav_new_log_prob - uav_old_log_prob
                    cache_branch_ratio = torch.exp(cache_branch_delta)
                    uav_branch_ratio = torch.exp(uav_branch_delta)
                    if (
                        uav_branch_old_log_probs_values is None
                        or uav_branch_new_log_probs_values is None
                        or uav_branch_ratios_values is None
                        or uav_branch_clipped_values is None
                    ):
                        raise RuntimeError(
                            "Joint UAV branch diagnostic arrays are unavailable"
                        )
                    sample_indices = indices.detach().cpu().numpy()
                    uav_branch_old_log_probs_values[sample_indices] = (
                        uav_old_log_prob.detach().cpu().numpy()
                    )
                    uav_branch_new_log_probs_values[sample_indices] = (
                        uav_new_log_prob.detach().cpu().numpy()
                    )
                    uav_branch_ratios_values[sample_indices] = (
                        uav_branch_ratio.detach().cpu().numpy()
                    )
                    uav_branch_clipped_values[sample_indices] = (
                        torch.abs(uav_branch_ratio.detach() - 1.0)
                        > hyper.clip_epsilon
                    ).cpu().numpy()
                    branch_metric_tensors = (
                        cache_branch_delta.detach(),
                        uav_branch_delta.detach(),
                        cache_branch_ratio.detach(),
                        uav_branch_ratio.detach(),
                        evaluation.cache_entropy.detach(),
                        evaluation.uav_entropy.detach(),
                    )
                    cache_node_delta = None
                    cache_node_ratio = None
                    if branchwise_policy_enabled:
                        if v11a_enabled:
                            assert evaluation.cache_node_log_probs is not None
                            assert cf_adv_normalized_t is not None
                            assert cf_adv_valid_mask_t is not None
                            cache_node_delta = (
                                evaluation.cache_node_log_probs
                                - cache_node_old_log_prob
                            )
                            cache_node_ratio = torch.exp(cache_node_delta)
                            cache_cf_advantage = cf_adv_normalized_t[indices]
                            cache_valid = cf_adv_valid_mask_t[indices]
                            cache_node_unclipped = (
                                cache_node_ratio * cache_cf_advantage
                            )
                            cache_node_clipped = torch.clamp(
                                cache_node_ratio,
                                1.0 - hyper.clip_epsilon,
                                1.0 + hyper.clip_epsilon,
                            ) * cache_cf_advantage
                            cache_node_surrogate = torch.minimum(
                                cache_node_unclipped,
                                cache_node_clipped,
                            )
                            if bool(torch.any(cache_valid).item()):
                                cache_branch_policy_loss = -cache_node_surrogate[cache_valid].mean()
                            else:
                                # Keep a graph-connected zero so an empty
                                # minibatch contributes no Cache actor update.
                                cache_branch_policy_loss = cache_node_ratio.sum() * 0.0
                            branch_metric_tensors += (
                                cache_node_delta.detach(),
                                cache_node_ratio.detach(),
                                cache_cf_advantage.detach(),
                            )
                        else:
                            cache_branch_unclipped = (
                                cache_branch_ratio * advantages_t[indices]
                            )
                            cache_branch_clipped = torch.clamp(
                                cache_branch_ratio,
                                1.0 - hyper.clip_epsilon,
                                1.0 + hyper.clip_epsilon,
                            ) * advantages_t[indices]
                            cache_branch_policy_loss = -torch.minimum(
                                cache_branch_unclipped,
                                cache_branch_clipped,
                            ).mean()
                        if r2_enabled:
                            assert (
                                r2_operation_advantages_scaled_t is not None
                                and evaluation.uav_operation_log_probs is not None
                                and evaluation.uav_candidate_log_probs is not None
                                and uav_operation_actions is not None
                                and uav_operation_old_log_probs is not None
                                and uav_candidate_old_log_probs is not None
                            )
                            operation_ratio = torch.exp(
                                evaluation.uav_operation_log_probs
                                - uav_operation_old_log_probs[indices]
                            )
                            operation_advantage = (
                                r2_operation_advantages_scaled_t[indices]
                            )
                            operation_unclipped = operation_ratio * operation_advantage
                            operation_clipped = torch.clamp(
                                operation_ratio,
                                1.0 - hyper.clip_epsilon,
                                1.0 + hyper.clip_epsilon,
                            ) * operation_advantage
                            operation_policy_loss = -torch.minimum(
                                operation_unclipped,
                                operation_clipped,
                            ).mean()
                            candidate_ratio = torch.exp(
                                evaluation.uav_candidate_log_probs
                                - uav_candidate_old_log_probs[indices]
                            )
                            candidate_advantage = advantages_t[indices].unsqueeze(-1)
                            candidate_unclipped = candidate_ratio * candidate_advantage
                            candidate_clipped = torch.clamp(
                                candidate_ratio,
                                1.0 - hyper.clip_epsilon,
                                1.0 + hyper.clip_epsilon,
                            ) * candidate_advantage
                            candidate_surrogate = torch.minimum(
                                candidate_unclipped,
                                candidate_clipped,
                            )
                            candidate_active = (
                                uav_operation_actions[indices] == 1
                            )
                            if bool(torch.any(candidate_active).item()):
                                candidate_policy_loss = -candidate_surrogate[
                                    candidate_active
                                ].mean()
                            else:
                                candidate_policy_loss = candidate_ratio.sum() * 0.0
                            # R2's UAV branch is explicitly the mean of the
                            # operation and candidate PPO losses.  It is kept
                            # outside _branch_actor_policy_loss so the latter's
                            # legacy 0.5 branch weight cannot halve R2 twice.
                            uav_branch_policy_loss = 0.5 * (
                                operation_policy_loss + candidate_policy_loss
                            )
                            branch_metric_tensors += (
                                operation_ratio.detach(),
                                candidate_ratio.detach(),
                                operation_advantage.detach(),
                                candidate_policy_loss.detach(),
                            )
                            if branch_specific_kl_gate_enabled:
                                branch_policy_loss = 0.0 * (
                                    cache_branch_policy_loss + uav_branch_policy_loss
                                )
                                if cache_active_this_minibatch:
                                    branch_policy_loss = (
                                        branch_policy_loss
                                        + 0.5 * cache_branch_policy_loss
                                    )
                                if uav_active_this_epoch:
                                    branch_policy_loss = (
                                        branch_policy_loss + uav_branch_policy_loss
                                    )
                            else:
                                branch_policy_loss = 0.5 * (
                                    cache_branch_policy_loss + uav_branch_policy_loss
                                )
                        else:
                            uav_branch_unclipped = (
                                uav_branch_ratio * advantages_t[indices]
                            )
                            uav_branch_clipped = torch.clamp(
                                uav_branch_ratio,
                                1.0 - hyper.clip_epsilon,
                                1.0 + hyper.clip_epsilon,
                            ) * advantages_t[indices]
                            uav_branch_policy_loss = -torch.minimum(
                                uav_branch_unclipped,
                                uav_branch_clipped,
                            ).mean()
                            if branch_specific_kl_gate_enabled:
                                branch_policy_loss = _branch_actor_policy_loss(
                                    cache_branch_policy_loss,
                                    uav_branch_policy_loss,
                                    cache_active=cache_active_this_minibatch,
                                    uav_active=uav_active_this_epoch,
                                )
                            else:
                                branch_policy_loss = 0.5 * (
                                    cache_branch_policy_loss + uav_branch_policy_loss
                                )
                        branch_metric_tensors += (
                            cache_branch_policy_loss,
                            uav_branch_policy_loss,
                            branch_policy_loss,
                        )
                    if not all(
                        bool(torch.all(torch.isfinite(value)).item())
                        for value in branch_metric_tensors
                    ):
                        raise FloatingPointError(
                            "Joint Cache/UAV branch metrics are non-finite"
                        )
                    cache_minibatch_approx_kl = float(
                        torch.mean(-cache_branch_delta.detach()).cpu()
                    )
                    uav_minibatch_approx_kl = float(
                        torch.mean(-uav_branch_delta.detach()).cpu()
                    )
                    cache_approx_kl_values.append(cache_minibatch_approx_kl)
                    uav_approx_kl_values.append(uav_minibatch_approx_kl)
                    if v10c_kl_instrumentation_enabled:
                        update_epoch_cache_approx_kls.append(
                            cache_minibatch_approx_kl
                        )
                        update_epoch_uav_approx_kls.append(
                            uav_minibatch_approx_kl
                        )
                    if (
                        v11a_enabled
                        and cache_node_ratio is not None
                        and cache_valid is not None
                        and bool(torch.any(cache_valid).item())
                    ):
                        cache_clip_metric = torch.mean(
                            (
                                torch.abs(cache_node_ratio.detach() - 1.0)
                                > hyper.clip_epsilon
                            )[cache_valid].to(torch.float32)
                        )
                    else:
                        cache_clip_metric = torch.mean(
                            (
                                torch.abs(
                                    (
                                        cache_node_ratio
                                        if cache_node_ratio is not None
                                        else cache_branch_ratio
                                    ).detach()
                                    - 1.0
                                )
                                > hyper.clip_epsilon
                            ).to(torch.float32)
                        )
                    cache_clip_fraction_values.append(
                        float(cache_clip_metric.detach().cpu())
                    )
                    uav_clip_fraction_values.append(
                        float(
                            torch.mean(
                                (
                                    torch.abs(uav_branch_ratio.detach() - 1.0)
                                    > hyper.clip_epsilon
                                )
                                .to(torch.float32)
                            ).cpu()
                        )
                    )
                    cache_entropy_values.append(
                        float(torch.mean(evaluation.cache_entropy.detach()).cpu())
                    )
                    uav_entropy_values.append(
                        float(torch.mean(evaluation.uav_entropy.detach()).cpu())
                    )
                    if branchwise_policy_enabled:
                        assert (
                            cache_branch_policy_loss is not None
                            and uav_branch_policy_loss is not None
                            and branch_policy_loss is not None
                        )
                        cache_policy_loss_values.append(
                            float(cache_branch_policy_loss.detach().cpu())
                        )
                        uav_policy_loss_values.append(
                            float(uav_branch_policy_loss.detach().cpu())
                        )
                        actor_policy_loss_values.append(
                            float(branch_policy_loss.detach().cpu())
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
                if branch_policy_loss is not None:
                    policy_loss = branch_policy_loss
                value_residual = (
                    evaluation.value.detach() - returns_t[indices]
                    if self._uses_separate_optimization()
                    else evaluation.value - returns_t[indices]
                )
                raw_value_loss = 0.5 * torch.mean(value_residual**2)
                if popart_critic:
                    normalized_value_residual = (
                        self.policy.critic_normalized_value(observations[indices]).detach()
                        - critic_targets_t[indices]
                    )
                else:
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
                    if branch_specific_kl_gate_enabled:
                        if branch_gradient_parameter_groups is None:
                            raise RuntimeError(
                                "Joint Cache/UAV branch gradient groups are unavailable"
                            )
                        # A zero-valued contribution is not sufficient for a
                        # unified Adam optimizer: Adam can continue moving a
                        # parameter from its historical momentum even when its
                        # current gradient is exactly zero.  Removing the
                        # stopped branch gradients makes optimizer.step() skip
                        # those parameters entirely, which is the required
                        # hard freeze after a Cache minibatch KL crossing.
                        _clear_inactive_branch_gradients(
                            branch_gradient_parameter_groups,
                            cache_active=cache_active_this_minibatch,
                            uav_active=uav_active_this_epoch,
                        )
                    if any(parameter.grad is not None for parameter in critic_parameters):
                        raise RuntimeError("PPO v5 actor backward touched critic parameters.")
                    for parameter in actor_parameters:
                        if parameter.grad is not None and not bool(
                            torch.all(torch.isfinite(parameter.grad)).item()
                        ):
                            gradient_finite = False
                    if not gradient_finite:
                        raise FloatingPointError("PPO v5 actor gradient 非 finite。")
                    if branch_diagnostics_enabled:
                        if branch_gradient_parameter_groups is None:
                            raise RuntimeError(
                                "Joint Cache/UAV branch gradient groups are unavailable"
                            )
                        cache_head_gradient_norm_value = _gradient_group_norm(
                            branch_gradient_parameter_groups["cache"]
                        )
                        uav_head_gradient_norm_value = _gradient_group_norm(
                            branch_gradient_parameter_groups["uav"]
                        )
                        shared_actor_gradient_norm_value = _gradient_group_norm(
                            branch_gradient_parameter_groups["shared"]
                        )
                        cache_encoder_gradient_norm_value = (
                            _gradient_group_norm(
                                branch_gradient_parameter_groups["cache_encoder"]
                            )
                            if "cache_encoder" in branch_gradient_parameter_groups
                            else math.nan
                        )
                        uav_encoder_gradient_norm_value = (
                            _gradient_group_norm(
                                branch_gradient_parameter_groups["uav_encoder"]
                            )
                            if "uav_encoder" in branch_gradient_parameter_groups
                            else math.nan
                        )
                        if not all(
                            np.isfinite(value)
                            for value in (
                                cache_head_gradient_norm_value,
                                uav_head_gradient_norm_value,
                                shared_actor_gradient_norm_value,
                            )
                        ) or (
                            v10c_kl_instrumentation_enabled
                            and not all(
                                np.isfinite(value)
                                for value in (
                                    cache_encoder_gradient_norm_value,
                                    uav_encoder_gradient_norm_value,
                                )
                            )
                        ):
                            raise FloatingPointError(
                                "Joint Cache/UAV branch gradient norm 非 finite。"
                            )
                        cache_head_gradient_norm_values.append(
                            cache_head_gradient_norm_value
                        )
                        uav_head_gradient_norm_values.append(
                            uav_head_gradient_norm_value
                        )
                        shared_actor_gradient_norm_values.append(
                            shared_actor_gradient_norm_value
                        )
                        if v10c_kl_instrumentation_enabled:
                            cache_encoder_gradient_norm_values.append(
                                cache_encoder_gradient_norm_value
                            )
                            uav_encoder_gradient_norm_values.append(
                                uav_encoder_gradient_norm_value
                            )
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
                    if popart_critic:
                        critic_normalized_values = self.policy.critic_normalized_value(
                            observations[indices]
                        )
                        critic_values = (
                            self.policy.popart_mean
                            + self.policy.popart_std * critic_normalized_values
                        )
                        critic_value_residual = critic_values.detach() - returns_t[indices]
                        raw_value_loss = 0.5 * torch.mean(critic_value_residual**2)
                        normalized_value_residual = (
                            critic_normalized_values - critic_targets_t[indices]
                        )
                    else:
                        critic_values = self.policy.critic_value(observations[indices])
                        critic_value_residual = critic_values - returns_t[indices]
                        raw_value_loss = 0.5 * torch.mean(critic_value_residual**2)
                        normalized_value_residual = critic_value_residual / critic_return_scale
                    value_loss = 0.5 * torch.mean(normalized_value_residual**2)
                    # V5 backpropagates the configured scaled loss.  V6
                    # backpropagates normalized-space loss.  In both cases
                    # raw value loss remains a diagnostic-only metric.
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
                global_critic_steps += 1
                if branchwise_policy_enabled and bool(uav_active_this_epoch):
                    uav_minibatches_executed += 1
                if branchwise_policy_enabled and bool(cache_active_this_minibatch):
                    cache_minibatches_executed += 1
                cache_approx_kl_after_step: float | None = None
                cache_stop_triggered_after_step = False
                if cache_minibatch_kl_control_enabled:
                    if cache_active_this_minibatch:
                        if (
                            v7_operation_masks is None
                            or v7_operation_actions is None
                            or v7_operation_old_log_probs is None
                            or cache_candidate_old_log_probs is None
                        ):
                            raise RuntimeError(
                                "minibatch Cache KL control tensors are incomplete"
                            )
                        cache_approx_kl_after_step = (
                            self._evaluate_cache_branch_approx_kl_full_rollout(
                                observations=observations,
                                cache_contexts=cache_contexts,
                                operation_masks=v7_operation_masks,
                                operation_actions=v7_operation_actions,
                                candidate_features=cache_candidate_features,
                                candidate_masks=cache_candidate_masks,
                                selected_indices=cache_selected_indices,
                                old_operation_log_probs=v7_operation_old_log_probs,
                                old_candidate_log_probs=cache_candidate_old_log_probs,
                            )
                        )
                        if cache_kl_control_previous is None:
                            raise RuntimeError(
                                "minibatch Cache KL control has no pre-step KL"
                            )
                        if (
                            hyper.target_kl is not None
                            and cache_approx_kl_after_step > hyper.target_kl
                        ):
                            cache_stop_triggered_after_step = True
                            cache_active = False
                            if cache_stop_epoch is None:
                                cache_stop_epoch = int(update_epoch + 1)
                                cache_stop_minibatch = int(
                                    start // minibatch_size + 1
                                )
                                cache_kl_before_stop = float(
                                    cache_kl_control_previous
                                )
                                cache_kl_after_stop = float(
                                    cache_approx_kl_after_step
                                )
                        cache_kl_control_previous = float(cache_approx_kl_after_step)
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
                update_epoch_approximate_kls.append(approximate_kl_value)
                if v10c_kl_instrumentation_enabled:
                    if not (
                        update_epoch_cache_approx_kls
                        and update_epoch_uav_approx_kls
                    ):
                        raise RuntimeError(
                            "V10-C branch KL diagnostics are not aligned with minibatches"
                        )
                    target_kl_value = hyper.target_kl
                    kl_minibatch_diagnostics.append(
                        {
                            "update_epoch": int(update_epoch + 1),
                            "minibatch": int(
                                len(kl_minibatch_diagnostics)
                                - epoch_diagnostic_start
                                + 1
                            ),
                            "cache_approx_kl": float(
                                update_epoch_cache_approx_kls[-1]
                            ),
                            "uav_approx_kl": float(
                                update_epoch_uav_approx_kls[-1]
                            ),
                            "joint_approx_kl": float(approximate_kl_value),
                            "target_kl": (
                                None
                                if target_kl_value is None
                                else float(target_kl_value)
                            ),
                            "cache_over_target": (
                                None
                                if target_kl_value is None
                                else bool(
                                    update_epoch_cache_approx_kls[-1]
                                    > target_kl_value
                                )
                            ),
                            "uav_over_target": (
                                None
                                if target_kl_value is None
                                else bool(
                                    update_epoch_uav_approx_kls[-1]
                                    > target_kl_value
                                )
                            ),
                            "joint_over_target": (
                                None
                                if target_kl_value is None
                                else bool(approximate_kl_value > target_kl_value)
                            ),
                            "cache_active": bool(cache_active_this_minibatch),
                            "cache_kl_control": str(self.cache_kl_control),
                            "cache_update_executed": bool(
                                cache_active_this_minibatch
                            ),
                            "cache_approx_kl_after_step": (
                                None
                                if cache_approx_kl_after_step is None
                                else float(cache_approx_kl_after_step)
                            ),
                            "cache_stop_triggered_after_step": bool(
                                cache_stop_triggered_after_step
                            ),
                            "cache_active_next_minibatch": bool(cache_active),
                            "uav_active": bool(uav_active_this_epoch),
                            "joint_gate_triggered_after_epoch": False,
                            "remaining_update_epochs_suppressed": 0,
                        }
                    )
                gradient_norm_values.append(gradient_norm_value)
                actor_gradient_norm_values.append(actor_gradient_norm_value)
                critic_gradient_norm_values.append(critic_gradient_norm_value)
            ppo_update_epochs_executed += 1
            joint_gate_triggered = False
            if (
                not branch_specific_kl_gate_enabled
                and
                hyper.target_kl is not None
                and update_epoch + 1 < hyper.update_epochs
                and float(np.mean(update_epoch_approximate_kls)) > hyper.target_kl
            ):
                # Check only after every minibatch in this update epoch has
                # completed.  The shared V9 loop therefore stops both actor
                # and critic updates for the remaining update epochs.
                kl_early_stop_triggered = True
                joint_gate_triggered = True
            if v10c_kl_instrumentation_enabled:
                if not (
                    update_epoch_cache_approx_kls
                    and update_epoch_uav_approx_kls
                    and len(update_epoch_cache_approx_kls)
                    == len(update_epoch_uav_approx_kls)
                    == len(update_epoch_approximate_kls)
                ):
                    raise RuntimeError("V10-C epoch KL diagnostics are incomplete")
                cache_epoch_approx_kl = float(
                    np.mean(update_epoch_cache_approx_kls)
                )
                uav_epoch_approx_kl = float(
                    np.mean(update_epoch_uav_approx_kls)
                )
                joint_epoch_approx_kl = float(
                    np.mean(update_epoch_approximate_kls)
                )
                target_kl_value = hyper.target_kl
                cache_epoch_over_target = (
                    None
                    if target_kl_value is None
                    else bool(cache_epoch_approx_kl > target_kl_value)
                )
                uav_epoch_over_target = (
                    None
                    if target_kl_value is None
                    else bool(uav_epoch_approx_kl > target_kl_value)
                )
                cache_only_trigger = (
                    cache_epoch_over_target is True
                    and uav_epoch_over_target is False
                )
                uav_only_trigger = (
                    uav_epoch_over_target is True
                    and cache_epoch_over_target is False
                )
                both_trigger = (
                    cache_epoch_over_target is True
                    and uav_epoch_over_target is True
                )
                neither_trigger = (
                    cache_epoch_over_target is False
                    and uav_epoch_over_target is False
                )
                cache_newly_stopped = False
                uav_newly_stopped = False
                cache_independent_stop_event = False
                uav_independent_stop_event = False
                both_inactive_stop_event = False
                if branch_specific_kl_gate_enabled:
                    transition = _branch_specific_kl_gate_transition(
                        cache_active=(
                            cache_active
                            if cache_minibatch_kl_control_enabled
                            else cache_active_this_epoch
                        ),
                        uav_active=uav_active_this_epoch,
                        cache_epoch_kl=cache_epoch_approx_kl,
                        uav_epoch_kl=uav_epoch_approx_kl,
                        target_kl=target_kl_value,
                        update_epoch=update_epoch,
                        update_epochs=hyper.update_epochs,
                        cache_gate_enabled=not cache_minibatch_kl_control_enabled,
                        uav_gate_enabled=True,
                    )
                    cache_newly_stopped = bool(
                        transition["cache_newly_stopped"]
                    )
                    uav_newly_stopped = bool(transition["uav_newly_stopped"])
                    cache_independent_stop_event = bool(
                        transition["cache_independent_stop"]
                    )
                    uav_independent_stop_event = bool(
                        transition["uav_independent_stop"]
                    )
                    both_inactive_stop_event = bool(
                        transition["both_inactive_stop"]
                    )
                    cache_active = bool(transition["cache_active_next"])
                    uav_active = bool(transition["uav_active_next"])
                    if cache_newly_stopped and cache_stop_epoch is None:
                        cache_stop_epoch = int(update_epoch + 1)
                        cache_suppressed_update_epochs = int(
                            transition["remaining_epochs_suppressed"]
                        )
                    if uav_newly_stopped and uav_stop_epoch is None:
                        uav_stop_epoch = int(update_epoch + 1)
                        uav_suppressed_update_epochs = int(
                            transition["remaining_epochs_suppressed"]
                        )
                    cache_independent_stop = (
                        cache_independent_stop or cache_independent_stop_event
                    )
                    uav_independent_stop = (
                        uav_independent_stop or uav_independent_stop_event
                    )
                    both_inactive_stop = (
                        both_inactive_stop or both_inactive_stop_event
                    )
                    if both_inactive_stop_event:
                        # The triggering epoch's critic update has already
                        # completed.  Only later epochs are suppressed.
                        kl_early_stop_triggered = True
                    remaining_epochs_suppressed = 0
                else:
                    remaining_epochs_suppressed = (
                        int(hyper.update_epochs - update_epoch - 1)
                        if joint_gate_triggered
                        else 0
                    )
                for minibatch_record in kl_minibatch_diagnostics[
                    epoch_diagnostic_start:
                ]:
                    minibatch_record[
                        "joint_gate_triggered_after_epoch"
                    ] = bool(joint_gate_triggered)
                    minibatch_record[
                        "remaining_update_epochs_suppressed"
                    ] = remaining_epochs_suppressed
                    if branch_specific_kl_gate_enabled:
                        minibatch_record["cache_active_next_epoch"] = bool(
                            cache_active
                        )
                        minibatch_record["uav_active_next_epoch"] = bool(
                            uav_active
                        )
                        minibatch_record["cache_independent_stop"] = bool(
                            cache_independent_stop_event
                        )
                        minibatch_record["uav_independent_stop"] = bool(
                            uav_independent_stop_event
                        )
                        minibatch_record["both_inactive_stop"] = bool(
                            both_inactive_stop_event
                        )
                epoch_record = {
                        "update_epoch": int(update_epoch + 1),
                        "minibatch_count": int(
                            len(update_epoch_approximate_kls)
                        ),
                        "cache_approx_kl": cache_epoch_approx_kl,
                        "uav_approx_kl": uav_epoch_approx_kl,
                        "joint_approx_kl": joint_epoch_approx_kl,
                        "target_kl": (
                            None
                            if target_kl_value is None
                            else float(target_kl_value)
                        ),
                        "cache_over_target": cache_epoch_over_target,
                        "uav_over_target": uav_epoch_over_target,
                        "joint_over_target": (
                            None
                            if target_kl_value is None
                            else bool(joint_epoch_approx_kl > target_kl_value)
                        ),
                        "joint_gate_triggered": bool(joint_gate_triggered),
                        "remaining_update_epochs_suppressed": (
                            remaining_epochs_suppressed
                        ),
                        "cache_only_kl_trigger_count": int(cache_only_trigger),
                        "uav_only_kl_trigger_count": int(uav_only_trigger),
                        "both_kl_trigger_count": int(both_trigger),
                        "neither_kl_trigger_count": int(neither_trigger),
                        "cache_trigger_suppressed_uav_update": bool(
                            joint_gate_triggered and cache_only_trigger
                        ),
                        "uav_trigger_suppressed_cache_update": bool(
                            joint_gate_triggered and uav_only_trigger
                        ),
                    }
                if branch_specific_kl_gate_enabled:
                    epoch_record.update(
                        {
                            "kl_gate_mode": KL_GATE_MODE_BRANCH_SPECIFIC,
                            "cache_active": bool(cache_active_this_epoch),
                            "uav_active": bool(uav_active_this_epoch),
                            "cache_active_next_epoch": bool(cache_active),
                            "uav_active_next_epoch": bool(uav_active),
                            "cache_stop_epoch": cache_stop_epoch,
                            "uav_stop_epoch": uav_stop_epoch,
                            "cache_suppressed_update_epochs": int(
                                cache_suppressed_update_epochs
                            ),
                            "uav_suppressed_update_epochs": int(
                                uav_suppressed_update_epochs
                            ),
                            "joint_kl_diagnostic": True,
                            "joint_kl_control_enabled": False,
                            "cache_independent_stop": bool(
                                cache_independent_stop_event
                            ),
                            "uav_independent_stop": bool(
                                uav_independent_stop_event
                            ),
                            "both_inactive_stop": bool(
                                both_inactive_stop_event
                            ),
                        }
                    )
                kl_epoch_diagnostics.append(epoch_record)
            # Branch-specific actor gates never terminate the outer update:
            # the global critic must execute every configured epoch/minibatch
            # step even after both actor branches become inactive.
            if joint_gate_triggered:
                break
        if self.hyperparameters.critic_value_scale_mode == CRITIC_VALUE_SCALE_RUNNING_RETURN_STD:
            self.critic_return_statistics.update(returns)
        if branch_diagnostics_enabled:
            assert uav_branch_old_log_probs_values is not None
            assert uav_branch_new_log_probs_values is not None
            assert uav_branch_ratios_values is not None
            assert uav_branch_clipped_values is not None
            if not (
                np.all(np.isfinite(uav_branch_old_log_probs_values))
                and np.all(np.isfinite(uav_branch_new_log_probs_values))
                and np.all(np.isfinite(uav_branch_ratios_values))
            ):
                raise FloatingPointError(
                    "Joint UAV branch action diagnostics contain NaN/Inf"
                )
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
            kl_early_stop_triggered=kl_early_stop_triggered,
            ppo_update_epochs_executed=ppo_update_epochs_executed,
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
            ppo_advantages=ppo_advantages.copy(),
            raw_returns=returns.copy(),
            bootstrap_value=bootstrap_value,
            cache_approx_kl=(
                float(np.mean(cache_approx_kl_values))
                if cache_approx_kl_values
                else math.nan
            ),
            uav_approx_kl=(
                float(np.mean(uav_approx_kl_values))
                if uav_approx_kl_values
                else math.nan
            ),
            cache_clip_fraction=(
                float(np.mean(cache_clip_fraction_values))
                if cache_clip_fraction_values
                else math.nan
            ),
            uav_clip_fraction=(
                float(np.mean(uav_clip_fraction_values))
                if uav_clip_fraction_values
                else math.nan
            ),
            cache_entropy=(
                float(np.mean(cache_entropy_values))
                if cache_entropy_values
                else math.nan
            ),
            uav_entropy=(
                float(np.mean(uav_entropy_values))
                if uav_entropy_values
                else math.nan
            ),
            cache_head_gradient_norm=(
                float(np.mean(cache_head_gradient_norm_values))
                if cache_head_gradient_norm_values
                else math.nan
            ),
            uav_head_gradient_norm=(
                float(np.mean(uav_head_gradient_norm_values))
                if uav_head_gradient_norm_values
                else math.nan
            ),
            shared_actor_gradient_norm=(
                float(np.mean(shared_actor_gradient_norm_values))
                if shared_actor_gradient_norm_values
                else math.nan
            ),
            cache_encoder_gradient_norm=(
                float(np.mean(cache_encoder_gradient_norm_values))
                if cache_encoder_gradient_norm_values
                else math.nan
            ),
            uav_encoder_gradient_norm=(
                float(np.mean(uav_encoder_gradient_norm_values))
                if uav_encoder_gradient_norm_values
                else math.nan
            ),
            cache_policy_loss=(
                float(np.mean(cache_policy_loss_values))
                if cache_policy_loss_values
                else math.nan
            ),
            uav_policy_loss=(
                float(np.mean(uav_policy_loss_values))
                if uav_policy_loss_values
                else math.nan
            ),
            actor_policy_loss=(
                float(np.mean(actor_policy_loss_values))
                if actor_policy_loss_values
                else math.nan
            ),
            uav_branch_old_log_probs=uav_branch_old_log_probs_values,
            uav_branch_new_log_probs=uav_branch_new_log_probs_values,
            uav_branch_ratios=uav_branch_ratios_values,
            uav_branch_clipped=uav_branch_clipped_values,
            kl_minibatch_diagnostics=tuple(
                dict(record) for record in kl_minibatch_diagnostics
            ),
            kl_epoch_diagnostics=tuple(
                dict(record) for record in kl_epoch_diagnostics
            ),
            kl_gate_mode=self.kl_gate_mode,
            cache_active_per_update_epoch=tuple(cache_active_per_update_epoch),
            uav_active_per_update_epoch=tuple(uav_active_per_update_epoch),
            cache_stop_epoch=cache_stop_epoch,
            uav_stop_epoch=uav_stop_epoch,
            cache_suppressed_update_epochs=int(cache_suppressed_update_epochs),
            uav_suppressed_update_epochs=int(uav_suppressed_update_epochs),
            joint_kl_diagnostic=bool(v10c_kl_instrumentation_enabled),
            cache_independent_stop=bool(cache_independent_stop),
            uav_independent_stop=bool(uav_independent_stop),
            both_inactive_stop=bool(both_inactive_stop),
            q_cache_loss=float(v11a_credit.get("q_cache_loss", math.nan)),
            q_cache_explained_variance=float(
                v11a_credit.get("q_cache_explained_variance", math.nan)
            ),
            q_cache_grad_norm_raw=float(
                v11a_credit.get("q_cache_grad_norm_raw", math.nan)
            ),
            q_cache_grad_norm_clipped=float(
                v11a_credit.get("q_cache_grad_norm_clipped", math.nan)
            ),
            q_cache_optimizer_steps=int(
                v11a_credit.get("q_cache_optimizer_steps", 0)
            ),
            q_cache_optimizer_steps_per_rollout=int(
                v11a_credit.get("q_cache_optimizer_steps_per_rollout", 0)
            ),
            q_cache_loss_before=float(
                v11a_credit.get("q_cache_loss_before", math.nan)
            ),
            q_cache_loss_after=float(
                v11a_credit.get("q_cache_loss_after", math.nan)
            ),
            q_cache_ev_before=float(
                v11a_credit.get("q_cache_ev_before", math.nan)
            ),
            q_cache_ev_after=float(
                v11a_credit.get("q_cache_ev_after", math.nan)
            ),
            q_cache_epoch_diagnostics=tuple(
                dict(record)
                for record in v11a_credit.get("q_cache_epoch_diagnostics", ())
            ),
            cf_adv_raw=v11a_credit.get("cf_adv_raw"),
            cf_adv_normalized=v11a_credit.get("cf_adv_normalized"),
            cf_adv_valid_mask=v11a_credit.get("cf_adv_valid_mask"),
            q_cache_selected_values=v11a_credit.get("q_cache_selected_values"),
            q_cache_counterfactual_baseline=v11a_credit.get(
                "q_cache_counterfactual_baseline"
            ),
            cf_adv_raw_mean=float(v11a_credit.get("cf_adv_raw_mean", math.nan)),
            cf_adv_raw_std=float(v11a_credit.get("cf_adv_raw_std", math.nan)),
            cf_adv_normalized_mean=float(
                v11a_credit.get("cf_adv_normalized_mean", math.nan)
            ),
            cf_adv_normalized_std=float(
                v11a_credit.get("cf_adv_normalized_std", math.nan)
            ),
            cf_adv_min=float(v11a_credit.get("cf_adv_min", math.nan)),
            cf_adv_max=float(v11a_credit.get("cf_adv_max", math.nan)),
            cf_adv_positive_fraction=float(
                v11a_credit.get("cf_adv_positive_fraction", math.nan)
            ),
            cf_adv_negative_fraction=float(
                v11a_credit.get("cf_adv_negative_fraction", math.nan)
            ),
            cache_node_logprob_sum_max_abs_error=(
                float(max(cache_node_logprob_sum_errors))
                if cache_node_logprob_sum_errors
                else math.nan
            ),
            cache_surrogate_reduction=(
                "mean_over_valid_cache_node_time"
                if v11a_enabled
                else "transition_mean"
            ),
            valid_cache_node_time_count=(
                int(torch.sum(cf_adv_valid_mask_t).detach().cpu())
                if v11a_enabled and cf_adv_valid_mask_t is not None
                else 0
            ),
            cache_loss_scale_relative_to_v10d=(
                float(sample_count)
                / float(torch.sum(cf_adv_valid_mask_t).detach().cpu())
                if v11a_enabled
                and cf_adv_valid_mask_t is not None
                and int(torch.sum(cf_adv_valid_mask_t).detach().cpu()) > 0
                else math.nan
            ),
            cache_kl_control=self.cache_kl_control,
            cache_stop_update_epoch=cache_stop_epoch,
            cache_stop_minibatch=cache_stop_minibatch,
            cache_kl_before_stop=cache_kl_before_stop,
            cache_kl_after_stop=cache_kl_after_stop,
            cache_minibatches_executed=int(cache_minibatches_executed),
            cache_minibatches_suppressed=int(
                max(cache_minibatches_planned - cache_minibatches_executed, 0)
            ),
            uav_minibatches_executed=int(uav_minibatches_executed),
            global_critic_steps=int(global_critic_steps),
            uav_operation_advantages_raw=(
                None
                if not r2_enabled
                else np.asarray(
                    r2_operation_credit["raw_advantages"], dtype=np.float32
                ).copy()
            ),
            uav_operation_advantages_scaled=(
                None
                if not r2_enabled
                else np.asarray(
                    r2_operation_credit["scaled_advantages"], dtype=np.float32
                ).copy()
            ),
            uav_operation_q_stay=(
                None
                if not r2_enabled
                else np.asarray(r2_operation_credit["q_stay"], dtype=np.float32).copy()
            ),
            uav_operation_q_move=(
                None
                if not r2_enabled
                else np.asarray(r2_operation_credit["q_move"], dtype=np.float32).copy()
            ),
            uav_operation_baseline=(
                None
                if not r2_enabled
                else np.asarray(r2_operation_credit["baseline"], dtype=np.float32).copy()
            ),
            uav_operation_counterfactual_env_executions=int(
                r2_operation_credit.get("counterfactual_environment_executions", 0)
            ),
            uav_operation_mean_move_candidate_count=float(
                r2_operation_credit.get("mean_move_candidate_count", math.nan)
            ),
            uav_operation_credit_runtime_seconds=float(
                r2_operation_credit.get("credit_runtime_seconds", math.nan)
            ),
            uav_operation_advantage_normalization=str(
                r2_operation_credit.get("advantage_normalization", "")
            ),
            uav_operation_advantage_centering=(
                None
                if not r2_enabled
                else bool(r2_operation_credit.get("advantage_centering", False))
            ),
            uav_operation_advantage_sign_flip_count=int(
                r2_operation_credit.get("advantage_sign_flip_count", 0)
            ),
            uav_operation_ratio_granularity=str(
                r2_operation_credit.get("ratio_granularity", "")
            ),
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
        next_pool_index: int | None = None,
        episode_seed_start: int | None = None,
    ) -> dict[str, Any]:
        completed_training_episodes = int(completed_episodes)
        if completed_training_episodes < 0:
            raise ValueError("completed_training_episodes must be non-negative")
        resolved_pool_index = (
            completed_training_episodes
            if next_pool_index is None
            else int(next_pool_index)
        )
        if resolved_pool_index < 0:
            raise ValueError("next_pool_index must be non-negative")
        payload = {
            "checkpoint_version": self._expected_checkpoint_version(),
            "method": PPO_V2_METHOD,
            "policy_schema": self.policy.architecture_id,
            "cache_credit_granularity": (
                "FULL_NODE_ACTION"
                if self._uses_v11a_cache_node_cf()
                else None
            ),
            "cache_kl_control": str(self.cache_kl_control),
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
            "completed_epochs": int(completed_episodes),
            "completed_training_episodes": completed_training_episodes,
            "next_pool_index": resolved_pool_index,
            "data_protocol": str(self.context.data_protocol),
            "episode_slots": int(self.context.episode_slots),
            "origin_start": int(self.context.train_origin_start),
            "origin_end": int(self.context.train_origin_end),
            "terminal_origin": int(self.context.terminal_origin),
            "current_train_origin": int(current_train_origin),
            "current_cache_state": np.asarray(current_cache_state, dtype=np.int8).copy(),
            "current_uav_positions": _copy_uav_positions(current_uav_positions),
            "rollout_steps": int(rollout_steps),
            "num_minibatches": int(self.hyperparameters.num_minibatches),
            "minibatch_size": int(self.context.minibatch_size),
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
        if episode_seed_start is not None:
            payload["episode_seed_start"] = int(episode_seed_start)
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
        if self._uses_v11a_cache_node_cf():
            if self.q_cache is None or self.q_cache_optimizer is None:
                raise RuntimeError("V11-A Q_cache optimizer is not initialized.")
            payload.update(
                {
                    "q_cache_state_dict": self.q_cache.state_dict(),
                    "q_cache_optimizer_state_dict": self.q_cache_optimizer.state_dict(),
                    "q_cache_optimizer_steps": int(self.q_cache_optimizer_steps),
                    "q_cache_learning_rate": float(self.hyperparameters.critic_learning_rate),
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
        next_pool_index: int | None = None,
        episode_seed_start: int | None = None,
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
                next_pool_index,
                episode_seed_start,
            ),
            checkpoint,
        )
        return checkpoint

    def load_checkpoint(self, checkpoint: Path, config_hash: str) -> TrainingCursor:
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        checkpoint_protocol = str(
            payload.get(
                "data_protocol",
                project_config.PPO_DATA_PROTOCOL_LEGACY_SPLIT,
            )
        )
        if checkpoint_protocol != self.context.data_protocol:
            raise ValueError(
                "PPO checkpoint data_protocol mismatch; refusing cross-protocol resume: "
                f"checkpoint={checkpoint_protocol}, expected={self.context.data_protocol}."
            )
        if self.context.data_protocol == project_config.PPO_DATA_PROTOCOL_FULL_EPISODE_RANDOMIZED:
            expected_protocol_fields = {
                "episode_slots": self.context.episode_slots,
                "origin_start": self.context.train_origin_start,
                "origin_end": self.context.train_origin_end,
                "terminal_origin": self.context.terminal_origin,
                "rollout_steps": self.context.episode_slots,
                "num_minibatches": self.hyperparameters.num_minibatches,
                "minibatch_size": self.context.minibatch_size,
            }
            for field, expected in expected_protocol_fields.items():
                if field not in payload or int(payload[field]) != int(expected):
                    raise ValueError(
                        "PPO full_episode_randomized checkpoint protocol mismatch: "
                        f"field={field}, checkpoint={payload.get(field)!r}, expected={expected}."
                    )
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
        checkpoint_cache_kl_control = str(
            payload.get("cache_kl_control", DEFAULT_CACHE_KL_CONTROL)
        )
        if checkpoint_cache_kl_control != self.cache_kl_control:
            raise ValueError(
                "PPO checkpoint cache_kl_control mismatch; refusing silent "
                f"cross-mode resume: checkpoint={checkpoint_cache_kl_control}, "
                f"expected={self.cache_kl_control}."
            )
        restored_statistics = RunningReturnStatistics()
        if expected_scale_mode in {
            CRITIC_VALUE_SCALE_RUNNING_RETURN_STD,
            CRITIC_VALUE_SCALE_POPART_NORMALIZED,
        }:
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
        if self._uses_v11a_cache_node_cf() and payload.get(
            "cache_credit_granularity"
        ) != "FULL_NODE_ACTION":
            raise ValueError("V11-A checkpoint cache credit granularity mismatch.")
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
        if self._uses_v11a_cache_node_cf():
            if self.q_cache is None or self.q_cache_optimizer is None:
                raise RuntimeError("V11-A Q_cache optimizer is not initialized.")
            if (
                "q_cache_state_dict" not in payload
                or "q_cache_optimizer_state_dict" not in payload
            ):
                raise ValueError(
                    "V11-A checkpoint must contain Q_cache and optimizer state."
                )
            self.q_cache.load_state_dict(payload["q_cache_state_dict"], strict=True)
            self.q_cache_optimizer.load_state_dict(payload["q_cache_optimizer_state_dict"])
            self.q_cache_optimizer_steps = int(payload.get("q_cache_optimizer_steps", 0))
            if self.q_cache_optimizer_steps < 0:
                raise ValueError("V11-A Q_cache optimizer steps must be non-negative.")
        self.critic_return_statistics = restored_statistics
        if self._uses_popart_critic():
            if not isinstance(
                self.policy,
                AdaptivePrimitiveHorizonGroupedSeparateCriticPopArtV6ActorCritic,
            ):
                raise TypeError("PopArt checkpoint restore requires the V6 actor-critic policy.")
            expected_mean, expected_std = self.critic_return_statistics.active_statistics(
                self.hyperparameters.critic_value_scale_epsilon,
                self.hyperparameters.critic_value_scale_warmup_samples,
            )
            active_mean, active_std = self.policy.popart_statistics()
            if not math.isclose(active_mean, expected_mean, rel_tol=1.0e-6, abs_tol=1.0e-6) or not math.isclose(
                active_std, expected_std, rel_tol=1.0e-6, abs_tol=1.0e-6
            ):
                raise ValueError(
                    "PPO V6 checkpoint PopArt statistics do not match Welford state."
                )
        rng_state = payload.get("rng_state", {})
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        completed_episodes = int(payload.get("completed_episodes", -1))
        completed_training_episodes = int(
            payload.get("completed_training_episodes", completed_episodes)
        )
        if (
            completed_episodes < 0
            or completed_training_episodes < 0
            or completed_training_episodes != completed_episodes
        ):
            raise ValueError(
                "PPO checkpoint completed_training_episodes is invalid or inconsistent."
            )
        raw_next_pool_index = payload.get("next_pool_index", completed_training_episodes)
        next_pool_index = int(raw_next_pool_index)
        if next_pool_index < 0:
            raise ValueError("PPO checkpoint next_pool_index must be non-negative.")
        raw_episode_seed_start = payload.get("episode_seed_start")
        episode_seed_start = (
            None
            if raw_episode_seed_start is None
            else int(raw_episode_seed_start)
        )
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
            completed_episodes=completed_episodes,
            current_train_origin=current_train_origin,
            current_cache_state=current_cache_state,
            current_uav_positions=current_uav_positions,
            best_epoch=best_epoch,
            best_epoch_return=best_epoch_return,
            next_pool_index=next_pool_index,
            episode_seed_start=episode_seed_start,
        )


def _config_hash(config_path: str) -> str:
    path = Path(config_path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_protocol_metadata(
    context: EnvironmentContext,
    rollout_steps: int,
) -> dict[str, Any]:
    return {
        "data_protocol": str(context.data_protocol),
        "episode_slots": int(context.episode_slots),
        "origin_start": int(context.train_origin_start),
        "origin_end": int(context.train_origin_end),
        "terminal_origin": int(context.terminal_origin),
        "rollout_steps": int(rollout_steps),
        "num_minibatches": int(context.cfg["algorithm"]["ppo"]["num_minibatches"]),
        "minibatch_size": int(context.minibatch_size),
    }


def run_training(
    config_path: str,
    episodes: int,
    run_dir: str,
    *,
    rollout_steps: int | None = None,
    episode_pool: str | Path | None = None,
    episode_seed_start: int | None = None,
    generate_missing_episodes: bool = False,
    episode_generation_device: str = "cpu",
    resume: bool = False,
    verbose: bool = False,
) -> None:
    _require_torch()
    target_epochs = int(episodes)
    if target_epochs <= 0:
        raise ValueError("epochs must be greater than zero")
    if episode_generation_device not in {"cpu", "cuda"}:
        raise ValueError(
            "episode_generation_device must be either 'cpu' or 'cuda'"
        )
    base_context = load_environment_context(config_path)
    run_path = Path(run_dir)
    if not resume:
        ensure_fresh_run_dir_is_clean(run_path)
    full_episode_protocol = (
        base_context.data_protocol
        == project_config.PPO_DATA_PROTOCOL_FULL_EPISODE_RANDOMIZED
    )
    if episode_seed_start is not None:
        episode_seed_start = int(episode_seed_start)
        if not full_episode_protocol:
            raise ValueError(
                "--episode-seed-start is supported only for "
                "full_episode_randomized PPO training"
            )
    if generate_missing_episodes and not full_episode_protocol:
        raise ValueError(
            "--generate-missing-episodes is supported only for "
            "full_episode_randomized PPO training"
        )
    if episode_pool is None and episode_seed_start is not None:
        raise ValueError("--episode-seed-start requires --episode-pool")
    if generate_missing_episodes and episode_pool is None:
        raise ValueError(
            "--generate-missing-episodes requires --episode-pool and "
            "--episode-seed-start"
        )
    if generate_missing_episodes and episode_seed_start is None:
        raise ValueError(
            "--generate-missing-episodes requires --episode-seed-start"
        )

    seed_based_pool = episode_pool is not None and episode_seed_start is not None
    episode_pool_directory: Path | None = None
    episode_generator: Any | None = None
    if seed_based_pool:
        episode_pool_directory = _resolve_episode_pool_directory(
            base_context.root,
            episode_pool,
            allow_missing=generate_missing_episodes,
        )
        episode_generator = _load_single_episode_generator(base_context.root)
        episode_pool_paths = None
    elif episode_pool is None:
        episode_pool_paths = None
    else:
        episode_pool_paths = _load_episode_pool(base_context.root, episode_pool)

    if full_episode_protocol and episode_pool is None:
        raise ValueError(
            "full_episode_randomized PPO training requires --episode-pool"
        )
    if (
        episode_pool_paths is not None
        and not resume
        and target_epochs > len(episode_pool_paths)
    ):
        raise RuntimeError(
            "PPO episode pool exhausted before training could start: "
            f"requested={target_epochs}, available={len(episode_pool_paths)}"
        )
    active_episode_seed: int | None = None
    active_realization_action: str | None = None
    if seed_based_pool and not resume:
        assert episode_pool_directory is not None
        assert episode_seed_start is not None
        first_artifacts, active_realization_action = _ensure_episode_realization(
            base_context,
            config_path,
            episode_pool_directory / f"episode_{episode_seed_start}",
            episode_seed_start,
            generate_missing=generate_missing_episodes,
            generation_device=episode_generation_device,
            generator_module=episode_generator,
        )
        active_episode_seed = int(episode_seed_start)
        context = load_environment_context(
            config_path,
            realization_dir=first_artifacts.directory,
        )
    elif episode_pool_paths is not None:
        context = load_environment_context(
            config_path,
            realization_dir=episode_pool_paths[0].directory,
        )
        active_realization_action = "LOAD"
    else:
        # On a seed-mapped resume, use the static context only to construct the
        # policy and restore the checkpoint.  The checkpoint cursor selects the
        # next realization before any rollout begins.
        context = base_context
    if episode_pool_paths is not None or seed_based_pool:
        _assert_realization_context_compatible(base_context, context)
    if context.data_protocol == project_config.PPO_DATA_PROTOCOL_FULL_EPISODE_RANDOMIZED:
        effective_rollout_steps = (
            context.configured_rollout_steps
            if rollout_steps is None
            else int(rollout_steps)
        )
        if effective_rollout_steps != context.episode_slots:
            raise ValueError(
                "full_episode_randomized requires rollout_steps == episode_slots; "
                f"rollout_steps={effective_rollout_steps}, "
                f"episode_slots={context.episode_slots}."
            )
    else:
        effective_rollout_steps = (
            DEFAULT_ROLLOUT_STEPS if rollout_steps is None else int(rollout_steps)
        )
    if effective_rollout_steps is None or int(effective_rollout_steps) <= 0:
        raise ValueError("rollout_steps 必须大于0。")
    rollout_steps = int(effective_rollout_steps)
    seed = int(project_config.get_config_value(context.cfg, "experiment.seed"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainer = PPOV2Trainer(context)
    config_hash = _config_hash(config_path)
    completed_episodes = 0
    next_pool_index = 0
    current_train_origin = context.train_origin_start
    current_cache_state = np.asarray(context.initial_cache, dtype=np.int8).copy()
    current_uav_positions = _copy_uav_positions(context.initial_uav)
    checkpoint = run_path / CHECKPOINT_FILENAME
    episode_metrics_path = run_path / "episode_metrics.jsonl"
    update_metrics_path = run_path / "update_metrics.jsonl"
    critic_diagnostics_path = run_path / "critic_diagnostics.jsonl"
    slot_metrics_path = run_path / SLOT_METRICS_FILENAME
    protocol_metadata = _run_protocol_metadata(context, int(rollout_steps))
    manifest = write_run_manifest(
        run_path / RUN_MANIFEST_FILENAME,
        config_path=config_path,
        config_sha256=config_hash,
        seed=seed,
        trainer=trainer,
        context=context,
        target_epochs=target_epochs,
        rollout_steps=int(rollout_steps),
        resume=resume,
    )
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
        next_pool_index = (
            completed_episodes
            if cursor.next_pool_index is None
            else int(cursor.next_pool_index)
        )
        if completed_episodes > target_epochs:
            raise ValueError("checkpoint completed epochs exceed target epochs")
        if seed_based_pool:
            if (
                cursor.episode_seed_start is not None
                and cursor.episode_seed_start != episode_seed_start
            ):
                raise ValueError(
                    "PPO checkpoint episode_seed_start mismatch; refusing to "
                    "resume a different episode-seed sequence: "
                    f"checkpoint={cursor.episode_seed_start}, "
                    f"expected={episode_seed_start}"
                )
            if (
                cursor.episode_seed_start is None
                and completed_episodes > 0
            ):
                raise ValueError(
                    "PPO checkpoint lacks episode_seed_start provenance; "
                    "refusing to infer a resumed episode-seed sequence"
                )
        if episode_pool_paths is not None or seed_based_pool:
            if next_pool_index != completed_episodes:
                raise ValueError(
                    "PPO checkpoint next_pool_index must equal completed_training_episodes: "
                    f"next_pool_index={next_pool_index}, "
                    f"completed_training_episodes={completed_episodes}"
                )
            remaining_episodes = target_epochs - completed_episodes
            if (
                episode_pool_paths is not None
                and remaining_episodes > len(episode_pool_paths) - next_pool_index
            ):
                raise RuntimeError(
                    "PPO episode pool exhausted before resume could complete: "
                    f"requested_additional={remaining_episodes}, "
                    f"available={len(episode_pool_paths) - next_pool_index}"
                )
            if remaining_episodes > 0:
                if seed_based_pool:
                    assert episode_pool_directory is not None
                    assert episode_seed_start is not None
                    next_episode_seed = int(
                        episode_seed_start + next_pool_index
                    )
                    next_artifacts, active_realization_action = (
                        _ensure_episode_realization(
                            base_context,
                            config_path,
                            episode_pool_directory
                            / f"episode_{next_episode_seed}",
                            next_episode_seed,
                            generate_missing=generate_missing_episodes,
                            generation_device=episode_generation_device,
                            generator_module=episode_generator,
                        )
                    )
                    active_episode_seed = next_episode_seed
                    next_context = load_environment_context(
                        config_path,
                        realization_dir=next_artifacts.directory,
                    )
                else:
                    assert episode_pool_paths is not None
                    next_context = load_environment_context(
                        config_path,
                        realization_dir=episode_pool_paths[next_pool_index].directory,
                    )
                    active_realization_action = "LOAD"
                _assert_realization_context_compatible(context, next_context)
                context = next_context
                trainer.context = context
                if current_train_origin > context.train_origin_end:
                    current_train_origin = context.train_origin_start
                    current_cache_state = np.asarray(
                        context.initial_cache, dtype=np.int8
                    ).copy()
                    current_uav_positions = _copy_uav_positions(context.initial_uav)

    origins_per_epoch = context.train_origin_end - context.train_origin_start + 1
    update_run_summary(
        run_path / RUN_SUMMARY_FILENAME,
        episode_metrics_path=episode_metrics_path,
        slot_metrics_path=slot_metrics_path,
        completed_epochs=completed_episodes,
        target_epochs=target_epochs,
        status="completed" if completed_episodes >= target_epochs else "running",
        protocol_metadata=protocol_metadata,
    )
    git_sha_display = manifest.get("git_commit_sha") or "null"
    if len(git_sha_display) > 12:
        git_sha_display = git_sha_display[:12]
    identity_output = (
        f"Policy Schema: {manifest['policy_schema']}\n"
        f"Advantage Mode: {manifest['advantage_normalization_mode']}\n"
        f"Critic Scale Mode: {manifest['critic_value_scale_mode']}\n"
        f"Config SHA256: {str(config_hash)[:12]}\n"
        f"Git Commit: {git_sha_display}"
    )
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
            f"Target Epochs: {target_epochs}\n"
            f"{identity_output}",
            flush=True,
        )
    else:
        print(
            "[PPO] Training start\n"
            f"Target Epochs: {target_epochs}\n"
            f"Origins per Epoch: {origins_per_epoch}\n"
            f"Rollout Steps: {int(rollout_steps)}\n"
            f"{identity_output}",
            flush=True,
        )

    episode_started_at = time.perf_counter()
    episode_optimizer_start = trainer.optimizer_steps
    episode_actor_optimizer_start = trainer.actor_optimizer_steps
    episode_critic_optimizer_start = trainer.critic_optimizer_steps
    episode_rewards: list[float] = []
    episode_g_values: list[float] = []
    episode_q_values: list[float] = []
    episode_cache_costs: list[float] = []
    episode_movement_costs: list[float] = []
    episode_cache_trigger_counts: list[int] = []
    episode_cache_change_counts: list[int] = []
    episode_uav_trigger_counts: list[int] = []
    episode_uav_movement_counts: list[int] = []
    episode_uav_move_probabilities: list[float] = []
    episode_uav_stay_probabilities: list[float] = []
    episode_uav_logit_margins: list[float] = []
    episode_uav_move_feasible: list[bool] = []
    episode_uav_move_selected: list[bool] = []
    episode_uav_actual_movement: list[bool] = []
    episode_uav_move_advantages: list[float] = []
    episode_uav_stay_advantages: list[float] = []
    episode_uav_global_advantages: list[float] = []
    episode_uav_return_targets: list[float] = []
    episode_uav_rewards: list[float] = []
    episode_uav_q_values: list[float] = []
    episode_uav_movement_costs: list[float] = []
    episode_uav_delays: list[float] = []
    episode_uav_energies: list[float] = []
    episode_uav_old_log_probs: list[float] = []
    episode_uav_new_log_probs: list[float] = []
    episode_uav_ratios: list[float] = []
    episode_uav_clipped: list[bool] = []
    episode_uav_feasible_action_counts: list[int] = []
    episode_uav_move_candidate_counts: list[int] = []
    episode_cache_action_bitmasks: list[int] = []
    episode_uav_diagnostics_available = False
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
            if seed_based_pool:
                assert episode_pool_directory is not None
                assert episode_seed_start is not None
                next_episode_seed = int(episode_seed_start + next_pool_index)
                next_artifacts, active_realization_action = (
                    _ensure_episode_realization(
                        base_context,
                        config_path,
                        episode_pool_directory
                        / f"episode_{next_episode_seed}",
                        next_episode_seed,
                        generate_missing=generate_missing_episodes,
                        generation_device=episode_generation_device,
                        generator_module=episode_generator,
                    )
                )
                active_episode_seed = next_episode_seed
                next_context = load_environment_context(
                    config_path,
                    realization_dir=next_artifacts.directory,
                )
                _assert_realization_context_compatible(context, next_context)
                context = next_context
                trainer.context = context
            elif episode_pool_paths is not None:
                if next_pool_index >= len(episode_pool_paths):
                    raise RuntimeError(
                        "PPO episode pool exhausted; no realization is available "
                        f"for training episode {completed_episodes + 1}."
                    )
                next_context = load_environment_context(
                    config_path,
                    realization_dir=episode_pool_paths[next_pool_index].directory,
                )
                _assert_realization_context_compatible(context, next_context)
                context = next_context
                trainer.context = context
                active_episode_seed = None
                active_realization_action = "LOAD"
            current_train_origin = context.train_origin_start
            current_cache_state = np.asarray(context.initial_cache, dtype=np.int8).copy()
            current_uav_positions = _copy_uav_positions(context.initial_uav)
            episode_started_at = time.perf_counter()
            episode_optimizer_start = trainer.optimizer_steps
            episode_actor_optimizer_start = trainer.actor_optimizer_steps
            episode_critic_optimizer_start = trainer.critic_optimizer_steps
            episode_rewards = []
            episode_g_values = []
            episode_q_values = []
            episode_cache_costs = []
            episode_movement_costs = []
            episode_cache_trigger_counts = []
            episode_cache_change_counts = []
            episode_uav_trigger_counts = []
            episode_uav_movement_counts = []
            episode_uav_move_probabilities = []
            episode_uav_stay_probabilities = []
            episode_uav_logit_margins = []
            episode_uav_move_feasible = []
            episode_uav_move_selected = []
            episode_uav_actual_movement = []
            episode_uav_move_advantages = []
            episode_uav_stay_advantages = []
            episode_uav_global_advantages = []
            episode_uav_return_targets = []
            episode_uav_rewards = []
            episode_uav_q_values = []
            episode_uav_movement_costs = []
            episode_uav_delays = []
            episode_uav_energies = []
            episode_uav_old_log_probs = []
            episode_uav_new_log_probs = []
            episode_uav_ratios = []
            episode_uav_clipped = []
            episode_uav_feasible_action_counts = []
            episode_uav_move_candidate_counts = []
            episode_cache_action_bitmasks = []
            episode_uav_diagnostics_available = False
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
        active_pool_index = (
            None
            if episode_pool_paths is None and not seed_based_pool
            else int(next_pool_index)
        )
        if seed_based_pool:
            if active_episode_seed is None or active_realization_action is None:
                raise RuntimeError(
                    "PPO episode realization was not selected before rollout"
                )
            print(
                f"[PPO] Episode {episode_number}/{target_epochs} | "
                f"seed={active_episode_seed} | "
                f"realization={active_realization_action}",
                flush=True,
            )
        elif episode_pool_paths is not None:
            print(
                f"[PPO] Episode {episode_number}/{target_epochs} | "
                "realization=LOAD",
                flush=True,
            )
        chunk_origin_start = current_train_origin
        rollout_started_at = time.perf_counter()
        rollout = trainer.collect_chunk(
            current_train_origin,
            current_cache_state,
            current_uav_positions,
            int(rollout_steps),
        )
        rollout_runtime_seconds = float(time.perf_counter() - rollout_started_at)
        if context.data_protocol == project_config.PPO_DATA_PROTOCOL_FULL_EPISODE_RANDOMIZED:
            if len(rollout.transitions) != context.episode_slots:
                raise AssertionError(
                    "full_episode_randomized rollout must contain one complete episode"
                )
            if not bool(rollout.dones[-1]) or bool(np.any(rollout.dones[:-1])):
                raise AssertionError(
                    "full_episode_randomized must terminate only at origin 299"
                )
        update_started_at = time.perf_counter()
        update = trainer.update(rollout)
        update_runtime_seconds = float(time.perf_counter() - update_started_at)
        next_train_origin = current_train_origin + len(rollout.transitions)
        terminal = next_train_origin > context.train_origin_end
        if terminal:
            completed_episodes += 1
            if episode_pool_paths is not None or seed_based_pool:
                next_pool_index += 1
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
            next_pool_index=next_pool_index,
            episode_seed_start=episode_seed_start,
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
        cache_operation_metrics = _cache_operation_epoch_metrics(
            rollout.transitions,
            current_cache_state,
            context.cache_nodes,
            context.service_catalog["service_sizes_mb"],
        )
        v11a_update_metrics = _v11a_update_metric_fields(
            update,
            rollout,
            context.cache_nodes,
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
        episode_q_values.extend(
            float(item.actual_task_utility) for item in rollout.transitions
        )
        episode_cache_costs.extend(float(item.cache_cost) for item in rollout.transitions)
        episode_movement_costs.extend(
            float(item.movement_cost) for item in rollout.transitions
        )
        episode_cache_trigger_counts.extend(
            int(np.asarray(item.action[:cache_count], dtype=np.int64).sum())
            for item in rollout.transitions
        )
        episode_cache_change_counts.extend(
            int(item.actual_cache_update_count) for item in rollout.transitions
        )
        episode_uav_trigger_counts.extend(
            int(np.asarray(item.action[cache_count:], dtype=np.int64).sum())
            for item in rollout.transitions
        )
        episode_cache_action_bitmasks.extend(
            _binary_action_bitmask(item.action[:cache_count])
            for item in rollout.transitions
        )
        episode_uav_movement_counts.extend(
            int(item.actual_uav_movement_count) for item in rollout.transitions
        )
        for transition_index, (transition, raw_advantage) in enumerate(zip(
            rollout.transitions,
            update.raw_advantages,
            strict=True,
        )):
            if transition.uav_operation_probabilities is None:
                continue
            if (
                update.uav_branch_old_log_probs is None
                or update.uav_branch_new_log_probs is None
                or update.uav_branch_ratios is None
                or update.uav_branch_clipped is None
            ):
                raise AssertionError(
                    "joint UAV rollout is missing PPO branch log-probability diagnostics"
                )
            probabilities = np.asarray(
                transition.uav_operation_probabilities,
                dtype=np.float64,
            )
            margins = np.asarray(
                transition.uav_operation_logit_margins,
                dtype=np.float64,
            )
            operation_actions = np.asarray(
                transition.uav_operation_actions,
                dtype=np.int64,
            )
            operation_masks = np.asarray(
                transition.uav_operation_masks,
                dtype=bool,
            )
            actual_flags = np.asarray(
                [
                    bool(action.get("actual_moved", False))
                    for action in transition.uav_conditional_actions
                ],
                dtype=bool,
            )
            if (
                probabilities.ndim != 2
                or probabilities.shape[1] != 2
                or margins.ndim != 1
                or operation_actions.ndim != 1
                or operation_masks.ndim != 2
                or operation_masks.shape[1] != 2
                or actual_flags.ndim != 1
                or probabilities.shape[0]
                != margins.shape[0]
                != operation_actions.shape[0]
                != operation_masks.shape[0]
                != actual_flags.shape[0]
            ):
                raise AssertionError("UAV diagnostic transition shapes are inconsistent")
            if not np.all(np.isfinite(probabilities)) or not np.all(
                np.isfinite(margins)
            ):
                raise FloatingPointError(
                    "UAV diagnostic transition values are non-finite"
                )
            candidate_counts = np.asarray(
                transition.uav_candidate_counts, dtype=np.int64
            ).reshape(-1)
            if len(candidate_counts) != len(actual_flags) or np.any(candidate_counts < 0):
                raise AssertionError(
                    "UAV candidate-count diagnostics are inconsistent"
                )
            if int(np.sum(actual_flags)) != int(transition.actual_uav_movement_count):
                raise AssertionError(
                    "UAV actual movement flags disagree with physical movement count"
                )
            global_advantage_value = float(update.ppo_advantages[transition_index])
            return_target_value = float(update.raw_returns[transition_index])
            branch_old_value = float(
                update.uav_branch_old_log_probs[transition_index]
            )
            branch_new_value = float(
                update.uav_branch_new_log_probs[transition_index]
            )
            branch_ratio_value = float(update.uav_branch_ratios[transition_index])
            branch_clipped_value = bool(update.uav_branch_clipped[transition_index])
            action_condition_values = (
                global_advantage_value,
                return_target_value,
                branch_old_value,
                branch_new_value,
                branch_ratio_value,
                float(transition.reward),
                float(transition.actual_task_utility),
                float(transition.movement_cost),
                float(transition.total_task_delay_s),
                float(transition.total_vehicle_energy_j),
            )
            if not all(np.isfinite(value) for value in action_condition_values):
                raise FloatingPointError(
                    "UAV action-conditional logging values are non-finite"
                )
            episode_uav_diagnostics_available = True
            move_probabilities = probabilities[:, 1]
            stay_probabilities = probabilities[:, 0]
            move_selected = operation_actions == 1
            move_feasible = operation_masks[:, 1]
            feasible_action_counts = operation_masks.sum(axis=1).astype(np.int64)
            episode_uav_move_probabilities.extend(move_probabilities.tolist())
            episode_uav_stay_probabilities.extend(stay_probabilities.tolist())
            episode_uav_logit_margins.extend(margins.tolist())
            episode_uav_move_feasible.extend(move_feasible.tolist())
            episode_uav_move_selected.extend(move_selected.tolist())
            episode_uav_actual_movement.extend(actual_flags.tolist())
            episode_uav_global_advantages.extend(
                [global_advantage_value] * len(actual_flags)
            )
            episode_uav_return_targets.extend(
                [return_target_value] * len(actual_flags)
            )
            episode_uav_rewards.extend(
                [float(transition.reward)] * len(actual_flags)
            )
            episode_uav_q_values.extend(
                [float(transition.actual_task_utility)] * len(actual_flags)
            )
            episode_uav_movement_costs.extend(
                [float(transition.movement_cost)] * len(actual_flags)
            )
            episode_uav_delays.extend(
                [float(transition.total_task_delay_s)] * len(actual_flags)
            )
            episode_uav_energies.extend(
                [float(transition.total_vehicle_energy_j)] * len(actual_flags)
            )
            episode_uav_old_log_probs.extend(
                [branch_old_value] * len(actual_flags)
            )
            episode_uav_new_log_probs.extend(
                [branch_new_value] * len(actual_flags)
            )
            episode_uav_ratios.extend([branch_ratio_value] * len(actual_flags))
            episode_uav_clipped.extend([branch_clipped_value] * len(actual_flags))
            episode_uav_feasible_action_counts.extend(
                feasible_action_counts.tolist()
            )
            episode_uav_move_candidate_counts.extend(candidate_counts.tolist())
            raw_advantage_value = float(raw_advantage)
            if not np.isfinite(raw_advantage_value):
                raise FloatingPointError("UAV action advantage is non-finite")
            if np.any(move_selected):
                episode_uav_move_advantages.extend(
                    [raw_advantage_value] * int(np.sum(move_selected))
                )
            if np.any(~move_selected):
                episode_uav_stay_advantages.extend(
                    [raw_advantage_value] * int(np.sum(~move_selected))
                )
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
                cache_count=cache_count,
            )
        _append_jsonl(
            update_metrics_path,
            {
                "record_type": "update",
                "seed": seed,
                "episode": episode_number,
                "epoch": episode_number,
                "training_episode_index": int(episode_number),
                "realization_pool_index": active_pool_index,
                "realization_directory": (
                    None
                    if context.realization is None
                    else str(context.realization.directory)
                ),
                "completed_epochs": int(completed_episodes),
                "origin_start": int(chunk_origin_start),
                "origin_end": int(next_train_origin - 1),
                "global_training_slot_start": int(
                    (episode_number - 1) * context.episode_slots
                    + (chunk_origin_start - context.train_origin_start)
                ),
                "global_training_slot_end": int(
                    (episode_number - 1) * context.episode_slots
                    + (next_train_origin - context.train_origin_start - 1)
                ),
                "num_transitions": transition_count,
                "terminal": bool(terminal),
                "rollout_runtime_seconds": rollout_runtime_seconds,
                "update_runtime_seconds": update_runtime_seconds,
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
                "joint_approx_kl": float(update.approximate_kl),
                "approximate_kl_min": float(update.approximate_kl_min),
                "approximate_kl_max": float(update.approximate_kl_max),
                "kl_early_stop_triggered": bool(update.kl_early_stop_triggered),
                "ppo_update_epochs_executed": int(update.ppo_update_epochs_executed),
                "clip_fraction": float(update.clip_fraction),
                "joint_clip_fraction": float(update.clip_fraction),
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
                "global_critic_steps": int(update.global_critic_steps),
                "total_loss": float(update.total_loss),
                "total_loss_finite": bool(np.isfinite(update.total_loss)),
                "checkpoint": str(checkpoint_path),
                **cache_operation_metrics,
                **_branch_update_metric_fields(update),
                **_v10c_kl_update_metric_fields(update),
                **v11a_update_metrics,
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
            episode_q_array = np.asarray(episode_q_values, dtype=np.float64)
            episode_cache_cost_array = np.asarray(
                episode_cache_costs, dtype=np.float64
            )
            episode_movement_cost_array = np.asarray(
                episode_movement_costs, dtype=np.float64
            )
            if not (
                episode_q_array.shape
                == episode_cache_cost_array.shape
                == episode_movement_cost_array.shape
                == episode_reward_array.shape
            ):
                raise AssertionError("PPO v2 episode slot metric arrays are misaligned")
            if not np.all(
                np.isfinite(episode_q_array)
                & np.isfinite(episode_cache_cost_array)
                & np.isfinite(episode_movement_cost_array)
            ):
                raise FloatingPointError("PPO v2 episode slot metrics 非 finite。")
            if not np.allclose(
                episode_reward_array,
                episode_q_array - episode_cache_cost_array - episode_movement_cost_array,
                rtol=1.0e-12,
                atol=1.0e-9,
            ):
                raise AssertionError("PPO v2 episode slot reward contract mismatch")
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
            cache_trigger_rate = float(episode_cache_requests) / float(
                len(episode_reward_array) * cache_count
            )
            cache_change_given_trigger = (
                None
                if episode_cache_requests == 0
                else float(episode_actual_cache_updates)
                / float(episode_cache_requests)
            )
            uav_trigger_rate = float(episode_uav_requests) / float(
                len(episode_reward_array) * len(context.action_spec.uav_ids)
            )
            uav_move_given_trigger = (
                None
                if episode_uav_requests == 0
                else float(episode_actual_uav_movements)
                / float(episode_uav_requests)
            )
            epoch_entropy = float(np.mean(episode_entropies))
            epoch_approximate_kl = float(np.mean(episode_approximate_kls))
            uav_diagnostic_metrics = _uav_episode_diagnostics(
                available=episode_uav_diagnostics_available,
                move_probabilities=episode_uav_move_probabilities,
                stay_probabilities=episode_uav_stay_probabilities,
                logit_margins=episode_uav_logit_margins,
                move_feasible=episode_uav_move_feasible,
                move_selected=episode_uav_move_selected,
                actual_movement=episode_uav_actual_movement,
                move_advantages=episode_uav_move_advantages,
                stay_advantages=episode_uav_stay_advantages,
                global_advantages=episode_uav_global_advantages,
                global_return_targets=episode_uav_return_targets,
                rewards=episode_uav_rewards,
                q_values=episode_uav_q_values,
                movement_costs=episode_uav_movement_costs,
                delays=episode_uav_delays,
                energies=episode_uav_energies,
                old_log_probs=episode_uav_old_log_probs,
                new_log_probs=episode_uav_new_log_probs,
                ratios=episode_uav_ratios,
                clipped=episode_uav_clipped,
                feasible_action_counts=episode_uav_feasible_action_counts,
                move_candidate_counts=episode_uav_move_candidate_counts,
            )
            if len(episode_cache_action_bitmasks) != len(episode_reward_array):
                raise AssertionError(
                    "selected Cache action bitmasks are not aligned with episode slots"
                )
            cache_action_bitmask_histogram: dict[str, int] = {}
            for bitmask in episode_cache_action_bitmasks:
                key = str(int(bitmask))
                cache_action_bitmask_histogram[key] = (
                    cache_action_bitmask_histogram.get(key, 0) + 1
                )
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
                    next_pool_index=next_pool_index,
                    episode_seed_start=episode_seed_start,
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
                    next_pool_index=next_pool_index,
                    episode_seed_start=episode_seed_start,
                )
            _append_jsonl(
                episode_metrics_path,
                {
                    "record_type": "episode",
                    "seed": seed,
                    "episode": int(episode_number),
                    "epoch": int(episode_number),
                    "training_episode_index": int(episode_number),
                    "realization_pool_index": active_pool_index,
                    "realization_directory": (
                        None
                        if context.realization is None
                        else str(context.realization.directory)
                    ),
                    "completed_episodes": int(completed_episodes),
                    "completed_epochs": int(completed_episodes),
                    "global_training_slot_start": int(
                        (episode_number - 1) * context.episode_slots
                    ),
                    "global_training_slot_end": int(
                        (episode_number - 1) * context.episode_slots
                        + (context.train_origin_end - context.train_origin_start)
                    ),
                    "num_transitions": int(len(episode_reward_array)),
                    "episode_return": epoch_return,
                    "mean_reward": float(episode_reward_array.mean()),
                    "mean_G": float(episode_g_array.mean()),
                    "reward_std": float(episode_reward_array.std()),
                    "q_total": float(episode_q_array.sum()),
                    "critic_value_mean": float(np.mean(episode_critic_values)),
                    "critic_target_mean": float(np.mean(episode_critic_targets)),
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
                    "kl_early_stop_triggered": bool(update.kl_early_stop_triggered),
                    "ppo_update_epochs_executed": int(update.ppo_update_epochs_executed),
                    "cache_trigger_rate": cache_trigger_rate,
                    "cache_change_given_trigger": cache_change_given_trigger,
                    "uav_trigger_rate": uav_trigger_rate,
                    "uav_move_given_trigger": uav_move_given_trigger,
                    "cache_actual_change_rate": cache_change_rate,
                    "uav_actual_movement_rate": uav_move_rate,
                    **cache_operation_metrics,
                    **v11a_update_metrics,
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
                    "rollout_runtime_seconds": rollout_runtime_seconds,
                    "update_runtime_seconds": update_runtime_seconds,
                    "advantage_source": "actual_PPO_advantage",
                    "selected_cache_action_bitmask_encoding": (
                        "bit_i=selected cache trigger at cache head i; little-endian"
                    ),
                    "selected_cache_action_bitmask_per_slot": [
                        int(value) for value in episode_cache_action_bitmasks
                    ],
                    "selected_cache_action_bitmask_histogram": (
                        cache_action_bitmask_histogram
                    ),
                    "uav_action_conditional_grouping": (
                        "actual_uav_move" if episode_uav_diagnostics_available else None
                    ),
                    "uav_action_log_prob_scope": (
                        "joint_uav_branch_per_transition"
                        if episode_uav_diagnostics_available
                        else None
                    ),
                    "uav_action_conditioning_observational": (
                        True if episode_uav_diagnostics_available else None
                    ),
                    **uav_diagnostic_metrics,
                },
            )
            append_training_slot_metrics(
                slot_metrics_path,
                epoch=episode_number,
                rewards=episode_reward_array.tolist(),
                episode_return=epoch_return,
                episode_slots=context.episode_slots,
                origin_start=context.train_origin_start,
                q_values=episode_q_array.tolist(),
                cache_costs=episode_cache_cost_array.tolist(),
                movement_costs=episode_movement_cost_array.tolist(),
                cache_trigger_counts=episode_cache_trigger_counts,
                cache_change_counts=episode_cache_change_counts,
                uav_trigger_counts=episode_uav_trigger_counts,
                uav_movement_counts=episode_uav_movement_counts,
            )
            update_run_summary(
                run_path / RUN_SUMMARY_FILENAME,
                episode_metrics_path=episode_metrics_path,
                slot_metrics_path=slot_metrics_path,
                completed_epochs=completed_episodes,
                target_epochs=target_epochs,
                status=(
                    "completed"
                    if completed_episodes >= target_epochs
                    else "running"
                ),
                protocol_metadata=protocol_metadata,
            )
            should_update_plot = (
                completed_episodes % PLOT_UPDATE_INTERVAL_EPOCHS == 0
                or completed_episodes == target_epochs
            )
            if should_update_plot:
                try:
                    plot_path = update_training_epoch_return_plot(run_path)
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

    final_plot_path = run_path / "plots" / TRAINING_EPOCH_RETURN_PLOT_FILENAME
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
    parser.add_argument("--config", default="configs/system_config.toml")
    parser.add_argument(
        "--epochs",
        type=int,
        help="total target Epochs; one Epoch is one complete active train episode",
    )
    parser.add_argument("--episodes", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help=(
            "consecutive train origins per PPO optimizer step; legacy default: "
            f"{DEFAULT_ROLLOUT_STEPS}, full_episode_randomized is fixed by config"
        ),
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--episode-pool",
        default=None,
        help=(
            "directory containing pre-generated episode realization directories; "
            "one realization is consumed per full_episode_randomized episode"
        ),
    )
    parser.add_argument(
        "--episode-seed-start",
        type=int,
        default=None,
        help=(
            "episode seed for training episode index 0; maps episode e to "
            "episode_seed_start + e"
        ),
    )
    parser.add_argument(
        "--generate-missing-episodes",
        action="store_true",
        help=(
            "lazily generate a missing full-episode realization in --episode-pool "
            "before its PPO episode starts"
        ),
    )
    parser.add_argument(
        "--episode-generation-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="device used by the existing generator for missing episodes",
    )
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
        episode_pool=args.episode_pool,
        episode_seed_start=args.episode_seed_start,
        generate_missing_episodes=bool(args.generate_missing_episodes),
        episode_generation_device=args.episode_generation_device,
        resume=bool(args.resume),
        verbose=bool(args.verbose),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
