"""Minimal PPO v2 trainer for adaptive Cache/UAV request actions."""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.util
import json
import math
import random
import sys
import stat
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
from ablations.no_prediction import NO_PREDICTION_POLICY_SCHEMA
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
    AdaptiveCacheOnlyActorCritic,
    AdaptiveReplicaAwareCacheOnlyActorCritic,
    AdaptiveNoPredictionCacheOnlyActorCritic,
    AdaptiveJointCacheUAVPPOActorCritic,
    AdaptivePrimitiveActorCritic,
    AdaptivePrimitiveHorizonGroupedSeparateCriticOptimizationV5ActorCritic,
    AdaptivePrimitiveHorizonGroupedSeparateCriticPopArtV6ActorCritic,
    BernoulliActorCritic,
)
from .cache_action_q import CacheActionQCritic
from .cache_surrogate import CacheCounterfactualDeltaQSurrogate
from .primitive_control import PrimitiveTensorSpec, _intermediate_observation
from .rollout import (
    SimpleRollout,
    _check_request_contract,
    _check_reward,
    collect_rollout,
    compute_gae,
)
from .uav_operation_diagnostics import (
    compute_uav_operation_diagnostics,
    merge_uav_operation_diagnostic_arrays,
)


PPO_V2_METHOD = "ppo_simple_v2"
CHECKPOINT_FILENAME = "latest_checkpoint.pt"
BEST_TRAIN_RETURN_CHECKPOINT_FILENAME = "best_train_return_checkpoint.pt"
CHECKPOINT_VERSION = 2
PRIMITIVE_CHECKPOINT_VERSION = 3
VALUE_NORMALIZED_CHECKPOINT_VERSION = 4
POPART_CHECKPOINT_VERSION = 6
CACHE_ONLY_CHECKPOINT_VERSION = 7
REPLICA_AWARE_CACHE_ONLY_CHECKPOINT_VERSION = 8
NO_PREDICTION_CACHE_ONLY_CHECKPOINT_VERSION = 9
# The proposed Cache objective is now actual-rollout PPO with global GAE.  It
# is not resume-compatible with the earlier Exact-Q/Action-Q proposed
# checkpoints, so the version bump is deliberately explicit.
JOINT_PROPOSED_CHECKPOINT_VERSION = 17
CACHE_POLICY_CREDIT_SOURCE = "ACTUAL_ROLLOUT_GAE"
CACHE_ACTION_Q_USED_FOR_POLICY_CREDIT = "NO"
CACHE_COUNTERFACTUAL_MODE = "NONE_FOR_PRODUCTION"
CACHE_VALUE_MODEL = "NONE"
CACHE_NORMAL_COUNTERFACTUAL_MODE = "NONE_FOR_PRODUCTION"
CACHE_AUDIT_MODE = "DIAGNOSTIC_ONLY"
CACHE_ACTUAL_ACTION_Q_SOURCE = "NONE"
CACHE_UNEXECUTED_ACTION_Q_SOURCE = "NONE"
CACHE_ACTION_Q_HIDDEN_DIM = 64
CACHE_ACTION_Q_LEARNING_RATE = 1.0e-3
CACHE_ACTION_Q_AUDIT_INTERVAL = 20
CACHE_SURROGATE_TYPE = "DELTA_Q_ENSEMBLE"
CACHE_SURROGATE_ENSEMBLE_SIZE = 3
CACHE_SURROGATE_HIDDEN_DIM = 64
CACHE_SURROGATE_LEARNING_RATE = 1.0e-3
CACHE_SURROGATE_AUDIT_INTERVAL = 20
CACHE_SURROGATE_AUDIT_WINDOW = 3
CACHE_SURROGATE_MAX_K = 12
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
OPERATION_ADVANTAGE_NORMALIZATION_STD_ONLY = "std_only"
OPERATION_ADVANTAGE_NORMALIZATION_FIXED_REFERENCE_SCALE = "fixed_reference_scale"
DEFAULT_OPERATION_ADVANTAGE_NORMALIZATION_MODE = (
    OPERATION_ADVANTAGE_NORMALIZATION_STD_ONLY
)
DEFAULT_OPERATION_ADVANTAGE_REFERENCE_SCALE = 0.02
CANDIDATE_POLICY_IMPROVEMENT_LEGACY_PPO = "LEGACY_PPO"
CANDIDATE_POLICY_IMPROVEMENT_EXACT_Q_LISTWISE = "EXACT_Q_LISTWISE"
DEFAULT_EXACT_Q_LISTWISE_TEMPERATURE = 1.0
CRITIC_VALUE_SCALE_NONE = "none"
CRITIC_VALUE_SCALE_RUNNING_RETURN_STD = "running_return_std"
CRITIC_VALUE_SCALE_POPART_NORMALIZED = "popart_normalized"
DEFAULT_CRITIC_VALUE_SCALE_EPSILON = 1.0e-8
DEFAULT_CRITIC_VALUE_SCALE_WARMUP_SAMPLES = 64
# Primitive V6 and semantic Cache-only schemas remain available to their
# explicit baseline/ablation evaluators.  The proposed joint schema is the
# sole current joint PPO implementation.
V6_POLICY_SCHEMAS = frozenset(
    {AdaptivePrimitiveHorizonGroupedSeparateCriticPopArtV6ActorCritic.architecture_id}
)
CACHE_ONLY_POLICY_SCHEMAS = frozenset(
    {AdaptiveCacheOnlyActorCritic.architecture_id}
)
REPLICA_AWARE_CACHE_ONLY_POLICY_SCHEMAS = frozenset(
    {AdaptiveReplicaAwareCacheOnlyActorCritic.architecture_id}
)
NO_PREDICTION_POLICY_SCHEMAS = frozenset({NO_PREDICTION_POLICY_SCHEMA})
PROPOSED_POLICY_SCHEMAS = frozenset(
    {AdaptiveJointCacheUAVPPOActorCritic.architecture_id}
)
CANDIDATE_CREDIT_POLICY_SCHEMAS = PROPOSED_POLICY_SCHEMAS
# Exact Cache-Q/Action-Q remains available only to explicit diagnostic
# runners; no production policy schema may enter that path.
EXACT_CACHE_CREDIT_POLICY_SCHEMAS = frozenset()
JOINT_CACHE_UAV_POLICY_SCHEMAS = PROPOSED_POLICY_SCHEMAS
BRANCH_SPECIFIC_KL_POLICY_SCHEMAS = PROPOSED_POLICY_SCHEMAS
CACHE_OPERATION_POLICY_SCHEMAS = (
    CACHE_ONLY_POLICY_SCHEMAS | REPLICA_AWARE_CACHE_ONLY_POLICY_SCHEMAS | NO_PREDICTION_POLICY_SCHEMAS
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


class EpisodeFilesystemPermissionError(RuntimeError):
    """The current training process cannot read a canonical episode artifact."""


def _is_filesystem_permission_error(error: OSError) -> bool:
    return isinstance(error, PermissionError) or getattr(error, "errno", None) in {
        errno.EACCES,
        errno.EPERM,
        errno.EROFS,
    }


def _read_episode_artifact_for_training(path: Path) -> None:
    """Require a manifest-declared artifact to be readable by this process."""

    try:
        file_stat = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"PPO episode artifact is missing: {path}") from exc
    except OSError as exc:
        if _is_filesystem_permission_error(exc):
            raise EpisodeFilesystemPermissionError(
                "EPISODE_FILESYSTEM_PERMISSION_FAILURE: "
                f"cannot stat canonical episode artifact: {path}"
            ) from exc
        raise
    if not stat.S_ISREG(file_stat.st_mode):
        raise RuntimeError(f"PPO episode artifact is not a regular file: {path}")

    try:
        with path.open("rb") as stream:
            if stream.read(1) == b"":
                raise RuntimeError(f"PPO episode artifact is empty: {path}")
    except OSError as exc:
        if _is_filesystem_permission_error(exc):
            raise EpisodeFilesystemPermissionError(
                "EPISODE_FILESYSTEM_PERMISSION_FAILURE: "
                f"cannot open canonical episode artifact for rb: {path}"
            ) from exc
        raise


def _read_episode_manifest_for_training(path: Path) -> bytes:
    try:
        manifest_stat = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "PPO episode realization is missing episode_manifest.json: "
            f"{path}"
        ) from exc
    except OSError as exc:
        if _is_filesystem_permission_error(exc):
            raise EpisodeFilesystemPermissionError(
                "EPISODE_FILESYSTEM_PERMISSION_FAILURE: "
                f"cannot stat episode manifest: {path}"
            ) from exc
        raise
    if not stat.S_ISREG(manifest_stat.st_mode):
        raise RuntimeError(f"PPO episode manifest is not a regular file: {path}")

    try:
        with path.open("rb") as stream:
            return stream.read()
    except OSError as exc:
        if _is_filesystem_permission_error(exc):
            raise EpisodeFilesystemPermissionError(
                "EPISODE_FILESYSTEM_PERMISSION_FAILURE: "
                f"cannot open episode manifest for rb: {path}"
            ) from exc
        raise


def _check_episode_filesystem_readability(directory: Path) -> None:
    """Read the manifest and every declared artifact before provenance checks."""

    manifest_path = directory / "episode_manifest.json"
    manifest_bytes = _read_episode_manifest_for_training(manifest_path)

    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"PPO episode manifest is not valid UTF-8 JSON: {manifest_path}"
        ) from exc

    generated_artifacts = manifest.get("generated_artifacts")
    if not isinstance(generated_artifacts, list):
        raise RuntimeError(
            "PPO episode manifest has no valid generated_artifacts list: "
            f"{manifest_path}"
        )
    for artifact_name in generated_artifacts:
        if not isinstance(artifact_name, str) or not artifact_name:
            raise RuntimeError(
                "PPO episode manifest contains an invalid generated artifact name: "
                f"{manifest_path}"
            )
        _read_episode_artifact_for_training(directory / artifact_name)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PPO v2 training requires PyTorch.")


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
    """Apply one branch-specific KL gate transition after a full PPO epoch.

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
    # Both branches inactive is a legal absorbing state.  The triggering
    # epoch still runs the critic, and subsequent epochs keep the actor frozen
    # while the global critic schedule continues normally.
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
        can_suppress
        and (cache_newly_stopped or uav_newly_stopped)
        and not next_cache_active
        and not next_uav_active
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
    """Keep the actor branch scale at 0.5 for every active branch.

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
    operation_advantage_normalization_mode: str
    operation_advantage_reference_scale: float
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
    cache_trigger_policy_loss: float = math.nan
    cache_candidate_policy_loss: float = math.nan
    uav_policy_loss: float = math.nan
    uav_operation_policy_loss: float = math.nan
    # Conditional candidate-credit diagnostics.  These are
    # intentionally separate from the legacy selected-candidate/global-GAE
    # fields above.
    uav_candidate_policy_loss: float = math.nan
    uav_candidate_advantage_mean_abs: float = math.nan
    uav_candidate_advantage_weighted_mean_residual: float = math.nan
    uav_candidate_advantage_normalization: str = ""
    uav_candidate_advantage_source: str = ""
    # Exact node-specific Cache credit diagnostics.
    cache_trigger_advantage_mean_abs: float = math.nan
    cache_candidate_advantage_mean_abs: float = math.nan
    cache_trigger_advantage_weighted_mean_residual: float = math.nan
    cache_candidate_advantage_weighted_mean_residual: float = math.nan
    cache_trigger_advantage_mean_abs_by_role: dict[str, float] | None = None
    cache_credit_counterfactual_env_executions: int = 0
    cache_credit_runtime_seconds: float = math.nan
    cache_candidate_q_source: str = ""
    cache_logical_candidate_count: int = 0
    cache_exact_candidate_count: int = 0
    cache_surrogate_candidate_count: int = 0
    cache_keep_exact_count: int = 0
    cache_full_exact_audit_count: int = 0
    cache_counterfactual_executions_saved: int = 0
    cache_counterfactual_execution_reduction_fraction: float = math.nan
    cache_surrogate_inference_runtime_seconds: float = math.nan
    cache_surrogate_train_runtime_seconds: float = math.nan
    cache_surrogate_delta_q_mae: float = math.nan
    cache_surrogate_delta_q_rmse: float = math.nan
    cache_surrogate_rank_correlation: float = math.nan
    cache_surrogate_uncertainty_mean: float = math.nan
    cache_surrogate_uncertainty_p90: float = math.nan
    cache_surrogate_screening_active: bool = False
    cache_surrogate_screening_ready: bool = False
    cache_surrogate_selected_k: int | None = None
    cache_surrogate_audit_statistics: dict[str, Any] | None = None
    cache_total_logical_branches: int = 0
    cache_full_exact_audit_branches: int = 0
    cache_normal_slot_counterfactual_branches: int = 0
    cache_counterfactual_executions_avoided: int = 0
    cache_candidate_value_source: str = ""
    cache_q_critic_mae: float = math.nan
    cache_q_critic_rmse: float = math.nan
    cache_q_critic_rank_correlation: float = math.nan
    cache_q_critic_argmax_accuracy: float = math.nan
    cache_q_critic_recall_at_3: float = math.nan
    cache_q_critic_recall_at_5: float = math.nan
    cache_q_critic_selected_exact_regret: dict[str, Any] | None = None
    cache_q_critic_normalized_regret_p95: float = math.nan
    cache_q_critic_usable_for_policy_credit: str = "INCONCLUSIVE"
    cache_q_critic_train_seconds: float = math.nan
    cache_q_critic_inference_seconds: float = math.nan
    cache_audit_exact_seconds: float = math.nan
    cache_heldout_audit_before_training: bool = False
    cache_actual_action_q_anchor: str = "NO"
    cache_q_critic_audit_metrics: dict[str, Any] | None = None
    cache_target_scorer_parameter_delta_norm: float = math.nan
    uav_target_scorer_parameter_delta_norm: float = math.nan
    cache_top1_exact_q_accuracy_before: float = math.nan
    cache_top1_exact_q_accuracy_after: float = math.nan
    cache_mean_exact_q_regret_before: float = math.nan
    cache_mean_exact_q_regret_after: float = math.nan
    cache_expected_exact_q_before: float = math.nan
    cache_expected_exact_q_after: float = math.nan
    cache_expected_exact_q_delta: float = math.nan
    cache_target_policy_kl_old_new: float = math.nan
    uav_top1_exact_q_accuracy_before: float = math.nan
    uav_top1_exact_q_accuracy_after: float = math.nan
    uav_mean_exact_q_regret_before: float = math.nan
    uav_mean_exact_q_regret_after: float = math.nan
    uav_expected_exact_q_before: float = math.nan
    uav_expected_exact_q_after: float = math.nan
    uav_expected_exact_q_delta: float = math.nan
    uav_target_policy_kl_old_new: float = math.nan
    cache_proposal_diagnostics: dict[str, Any] | None = None
    uav_proposal_diagnostics: dict[str, Any] | None = None
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
    cache_kl_control: str = DEFAULT_CACHE_KL_CONTROL
    cache_stop_update_epoch: int | None = None
    cache_stop_minibatch: int | None = None
    cache_kl_before_stop: float | None = None
    cache_kl_after_stop: float | None = None
    cache_minibatches_executed: int = 0
    cache_minibatches_suppressed: int = 0
    uav_minibatches_executed: int = 0
    # Proposed UAV operation/candidate routing diagnostics.  Operation steps
    # count full-information binary-teacher updates; candidate updates remain
    # active after the operation KL gate freezes the operation head/encoder.
    uav_operation_steps_executed: int = 0
    uav_operation_steps_suppressed: int = 0
    uav_candidate_full_steps: int = 0
    uav_candidate_only_steps_after_operation_stop: int = 0
    uav_operation_full_rollout_kl: float = math.nan
    uav_operation_teacher_diagnostics: dict[str, Any] | None = None
    uav_operation_head_parameter_delta_norm: float = math.nan
    uav_encoder_parameter_delta_norm: float = math.nan
    global_critic_steps: int = 0
    # Proposed operation-credit diagnostics.  These are populated only for
    # the proposed schema and are deliberately separate from the legacy global-GAE
# and joint UAV branch fields above.
    uav_operation_advantages_raw: np.ndarray | None = None
    uav_operation_advantages_scaled: np.ndarray | None = None
    uav_operation_q_stay: np.ndarray | None = None
    uav_operation_q_move: np.ndarray | None = None
    uav_operation_baseline: np.ndarray | None = None
    uav_operation_counterfactual_env_executions: int = 0
    uav_exact_q_branch_request_count: int = 0
    uav_exact_q_unique_branch_count: int = 0
    uav_exact_q_duplicate_branch_count: int = 0
    uav_exact_q_cache_hit_count: int = 0
    uav_exact_q_execute_slot_call_count: int = 0
    uav_exact_q_matching_call_count: int = 0
    uav_exact_q_kkt_call_count: int = 0
    uav_operation_mean_move_candidate_count: float = math.nan
    uav_operation_credit_runtime_seconds: float = math.nan
    uav_operation_advantage_normalization: str = ""
    uav_operation_advantage_centering: bool | None = None
    uav_operation_advantage_sign_flip_count: int = 0
    uav_operation_ratio_granularity: str = ""
    # Compact exact-Q operation diagnostics.  The private arrays are retained
    # only in memory so a multi-chunk episode can be summarized without
    # writing per-transition raw arrays to JSONL.
    uav_operation_diagnostics: dict[str, Any] | None = None
    uav_operation_diagnostic_arrays: dict[str, np.ndarray] | None = None


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
    episode_seed: int,
) -> RealizationArtifacts:
    _check_episode_filesystem_readability(directory)
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
    generator_module.validate_episode_provenance(
        directory,
        context.cfg,
        int(episode_seed),
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
                realization_dir,
                context,
                generator_module,
                int(episode_seed),
            )
        except EpisodeFilesystemPermissionError:
            raise
        except PermissionError as exc:
            raise EpisodeFilesystemPermissionError(
                "EPISODE_FILESYSTEM_PERMISSION_FAILURE: "
                f"existing canonical episode is not readable: {realization_dir}"
            ) from exc
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
            realization_dir,
            context,
            generator_module,
            int(episode_seed),
        )
    except EpisodeFilesystemPermissionError:
        raise
    except PermissionError as exc:
        raise EpisodeFilesystemPermissionError(
            "EPISODE_FILESYSTEM_PERMISSION_FAILURE: "
            f"generated canonical episode is not readable: {realization_dir}"
        ) from exc
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
    """Summarize KEEP/UPDATE behavior without affecting rollout or PPO update."""

    operation_names = ("KEEP", "UPDATE")
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
                conditional_action = str(action.get("conditional_action", "")).upper()
                raw_operation = str(action.get("operation_type", "")).upper()
                if conditional_action == "KEEP" or raw_operation == "NO_CHANGE":
                    operation = "KEEP"
                elif conditional_action == "TARGET_CONFIGURATION" or raw_operation in {
                    "ADD",
                    "DELETE",
                    "SWAP",
                }:
                    operation = "UPDATE"
                else:
                    operation = ""
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


def _exact_q_policy_quality_metrics(
    probabilities_before: Any,
    probabilities_after: Any,
    q_values: Any,
    valid_mask: Any,
) -> dict[str, float]:
    """Compare two policies on one frozen exact-Q candidate batch."""

    before = np.asarray(probabilities_before, dtype=np.float64)
    after = np.asarray(probabilities_after, dtype=np.float64)
    q = np.asarray(q_values, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if before.shape != after.shape or before.shape != q.shape or before.shape != valid.shape:
        raise ValueError("exact-Q policy quality tensor shapes do not match")
    if before.ndim < 1 or not np.all(np.isfinite(q[valid])):
        raise ValueError("exact-Q policy quality tensors are invalid")
    top1_before: list[float] = []
    top1_after: list[float] = []
    regret_before: list[float] = []
    regret_after: list[float] = []
    expected_before: list[float] = []
    expected_after: list[float] = []
    policy_kls: list[float] = []
    for prefix in np.ndindex(q.shape[:-1]):
        legal = np.flatnonzero(valid[prefix])
        if legal.size == 0:
            continue
        p_before = before[prefix][legal]
        p_after = after[prefix][legal]
        q_row = q[prefix][legal]
        if (
            not np.all(np.isfinite(p_before))
            or not np.all(np.isfinite(p_after))
            or np.any(p_before < 0.0)
            or np.any(p_after < 0.0)
            or not np.all(np.isfinite(q_row))
        ):
            raise FloatingPointError("exact-Q policy quality contains NaN/Inf")
        before_sum = float(p_before.sum())
        after_sum = float(p_after.sum())
        if before_sum <= 0.0 or after_sum <= 0.0:
            raise FloatingPointError("exact-Q candidate policy probability mass is empty")
        p_before = p_before / before_sum
        p_after = p_after / after_sum
        best_q = float(np.max(q_row))
        best_mask = np.isclose(q_row, best_q, rtol=1.0e-10, atol=1.0e-10)
        selected_before = int(np.argmax(p_before))
        selected_after = int(np.argmax(p_after))
        selected_q_before = float(q_row[selected_before])
        selected_q_after = float(q_row[selected_after])
        top1_before.append(float(bool(best_mask[selected_before])))
        top1_after.append(float(bool(best_mask[selected_after])))
        regret_before.append(max(0.0, best_q - selected_q_before))
        regret_after.append(max(0.0, best_q - selected_q_after))
        expected_before.append(float(np.dot(p_before, q_row)))
        expected_after.append(float(np.dot(p_after, q_row)))
        positive = p_before > 0.0
        kl = float(
            np.sum(
                p_before[positive]
                * (
                    np.log(p_before[positive])
                    - np.log(np.maximum(p_after[positive], 1.0e-300))
                )
            )
        )
        policy_kls.append(max(0.0, kl))
    if not top1_before:
        return {
            "count": 0.0,
            "top1_before": math.nan,
            "top1_after": math.nan,
            "mean_regret_before": math.nan,
            "mean_regret_after": math.nan,
            "expected_q_before": math.nan,
            "expected_q_after": math.nan,
            "expected_q_delta": math.nan,
            "policy_kl_old_new": math.nan,
        }
    expected_q_before = float(np.mean(expected_before))
    expected_q_after = float(np.mean(expected_after))
    result = {
        "count": float(len(top1_before)),
        "top1_before": float(np.mean(top1_before)),
        "top1_after": float(np.mean(top1_after)),
        "mean_regret_before": float(np.mean(regret_before)),
        "mean_regret_after": float(np.mean(regret_after)),
        "expected_q_before": expected_q_before,
        "expected_q_after": expected_q_after,
        "expected_q_delta": expected_q_after - expected_q_before,
        "policy_kl_old_new": float(np.mean(policy_kls)),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("exact-Q policy quality is non-finite")
    return result


def _proposal_source_diagnostics(
    q_values: Any,
    valid_mask: Any,
    sources: Any,
    distances: Any,
) -> dict[str, Any]:
    """Summarize the fixed proposal set without evaluating new branches."""

    q = np.asarray(q_values, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    source_array = np.asarray(sources, dtype=object)
    distance_array = np.asarray(distances, dtype=np.float64)
    if q.shape != valid.shape or q.shape != source_array.shape or q.shape != distance_array.shape:
        raise ValueError("proposal diagnostic tensor shapes do not match")
    counts = valid.sum(axis=-1).reshape(-1)
    active_counts = counts[counts > 0]
    source_counts: dict[str, int] = {}
    best_distances: list[float] = []
    for prefix in np.ndindex(q.shape[:-1]):
        legal = np.flatnonzero(valid[prefix])
        if legal.size == 0:
            continue
        best_index = int(legal[int(np.argmax(q[prefix][legal]))])
        source = str(source_array[prefix + (best_index,)])
        if not source:
            raise ValueError("proposal diagnostic is missing a candidate source")
        source_counts[source] = source_counts.get(source, 0) + 1
        distance = float(distance_array[prefix + (best_index,)])
        if not math.isfinite(distance):
            raise FloatingPointError("proposal diagnostic distance is non-finite")
        best_distances.append(distance)
    total = int(sum(source_counts.values()))
    fractions = {
        source: float(count) / float(total)
        for source, count in sorted(source_counts.items())
    }
    return {
        "candidate_count_mean": (
            0.0 if active_counts.size == 0 else float(np.mean(active_counts))
        ),
        "candidate_count_max": (
            0 if active_counts.size == 0 else int(np.max(active_counts))
        ),
        "best_q_source_counts": {
            source: int(count) for source, count in sorted(source_counts.items())
        },
        "best_q_source_fractions": fractions,
        "best_q_distance_mean": (
            math.nan if not best_distances else float(np.mean(best_distances))
        ),
        "best_q_distance_median": (
            math.nan if not best_distances else float(np.median(best_distances))
        ),
        "best_q_distance_max": (
            math.nan if not best_distances else float(np.max(best_distances))
        ),
    }


def _branch_update_metric_fields(update: UpdateResult) -> dict[str, Any]:
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
        "cache_trigger_policy_loss": float(update.cache_trigger_policy_loss),
        "cache_candidate_listwise_loss": float(
            update.cache_candidate_policy_loss
        ),
        "uav_policy_loss": float(update.uav_policy_loss),
        "uav_operation_policy_loss": float(update.uav_operation_policy_loss),
        "uav_candidate_listwise_loss": float(update.uav_candidate_policy_loss),
        "actor_policy_loss": float(update.actor_policy_loss),
    }
    if all(math.isfinite(value) for value in policy_losses.values()):
        fields.update(policy_losses)
    candidate_metrics = {
        "candidate_loss": float(update.uav_candidate_policy_loss),
        "mean_abs_candidate_advantage": float(
            update.uav_candidate_advantage_mean_abs
        ),
        "candidate_adv_weighted_mean_residual": float(
            update.uav_candidate_advantage_weighted_mean_residual
        ),
    }
    if all(math.isfinite(value) for value in candidate_metrics.values()):
        fields.update(candidate_metrics)
        fields["candidate_advantage_normalization"] = str(
            update.uav_candidate_advantage_normalization
        )
        fields["candidate_advantage_source"] = str(
            update.uav_candidate_advantage_source
        )
    scorer_deltas = {
        "cache_target_scorer_parameter_delta_norm": float(
            update.cache_target_scorer_parameter_delta_norm
        ),
        "uav_target_scorer_parameter_delta_norm": float(
            update.uav_target_scorer_parameter_delta_norm
        ),
    }
    if all(math.isfinite(value) for value in scorer_deltas.values()):
        fields.update(scorer_deltas)
    exact_q_quality = {
        "cache_top1_exact_q_accuracy_before": float(
            update.cache_top1_exact_q_accuracy_before
        ),
        "cache_top1_exact_q_accuracy_after": float(
            update.cache_top1_exact_q_accuracy_after
        ),
        "cache_mean_exact_q_regret_before": float(
            update.cache_mean_exact_q_regret_before
        ),
        "cache_mean_exact_q_regret_after": float(
            update.cache_mean_exact_q_regret_after
        ),
        "cache_expected_exact_q_before": float(update.cache_expected_exact_q_before),
        "cache_expected_exact_q_after": float(update.cache_expected_exact_q_after),
        "cache_expected_exact_q_delta": float(update.cache_expected_exact_q_delta),
        "cache_target_policy_kl_old_new": float(update.cache_target_policy_kl_old_new),
        "uav_top1_exact_q_accuracy_before": float(
            update.uav_top1_exact_q_accuracy_before
        ),
        "uav_top1_exact_q_accuracy_after": float(
            update.uav_top1_exact_q_accuracy_after
        ),
        "uav_mean_exact_q_regret_before": float(
            update.uav_mean_exact_q_regret_before
        ),
        "uav_mean_exact_q_regret_after": float(
            update.uav_mean_exact_q_regret_after
        ),
        "uav_expected_exact_q_before": float(update.uav_expected_exact_q_before),
        "uav_expected_exact_q_after": float(update.uav_expected_exact_q_after),
        "uav_expected_exact_q_delta": float(update.uav_expected_exact_q_delta),
        "uav_target_policy_kl_old_new": float(update.uav_target_policy_kl_old_new),
    }
    if all(math.isfinite(value) for value in exact_q_quality.values()):
        fields.update(exact_q_quality)
    if update.cache_proposal_diagnostics is not None:
        fields["cache_proposal_diagnostics"] = dict(update.cache_proposal_diagnostics)
    if update.uav_proposal_diagnostics is not None:
        fields["uav_proposal_diagnostics"] = dict(update.uav_proposal_diagnostics)
    cache_credit_metrics = {
        "cache_trigger_advantage_mean_abs": float(
            update.cache_trigger_advantage_mean_abs
        ),
        "cache_candidate_advantage_mean_abs": float(
            update.cache_candidate_advantage_mean_abs
        ),
        "cache_trigger_advantage_weighted_mean_residual": float(
            update.cache_trigger_advantage_weighted_mean_residual
        ),
        "cache_candidate_advantage_weighted_mean_residual": float(
            update.cache_candidate_advantage_weighted_mean_residual
        ),
        "cache_credit_runtime_seconds": float(update.cache_credit_runtime_seconds),
    }
    if all(math.isfinite(value) for value in cache_credit_metrics.values()):
        fields.update(cache_credit_metrics)
        fields["cache_credit_counterfactual_env_executions"] = int(
            update.cache_credit_counterfactual_env_executions
        )
        if update.cache_trigger_advantage_mean_abs_by_role is not None:
            fields["cache_trigger_advantage_mean_abs_by_role"] = {
                str(role): float(value)
                for role, value in update.cache_trigger_advantage_mean_abs_by_role.items()
                if math.isfinite(float(value))
            }
    hybrid_cache_metrics = {
        "CACHE_LOGICAL_CANDIDATE_COUNT": int(update.cache_logical_candidate_count),
        "CACHE_EXACT_CANDIDATE_COUNT": int(update.cache_exact_candidate_count),
        "CACHE_SURROGATE_CANDIDATE_COUNT": int(
            update.cache_surrogate_candidate_count
        ),
        "CACHE_KEEP_EXACT_COUNT": int(update.cache_keep_exact_count),
        "CACHE_FULL_EXACT_AUDIT_COUNT": int(update.cache_full_exact_audit_count),
        "CACHE_COUNTERFACTUAL_ENV_EXECUTIONS": int(
            update.cache_credit_counterfactual_env_executions
        ),
        "CACHE_COUNTERFACTUAL_EXECUTIONS_SAVED": int(
            update.cache_counterfactual_executions_saved
        ),
        "CACHE_COUNTERFACTUAL_EXECUTIONS_AVOIDED": int(
            update.cache_counterfactual_executions_avoided
        ),
        "CACHE_SURROGATE_SCREENING_ACTIVE": bool(
            update.cache_surrogate_screening_active
        ),
        "CACHE_SURROGATE_SCREENING_READY": bool(
            update.cache_surrogate_screening_ready
        ),
        "CACHE_SURROGATE_SELECTED_K": update.cache_surrogate_selected_k,
        "CACHE_TRIGGER_Q_SOURCE": str(update.cache_candidate_q_source),
        "CACHE_TOTAL_LOGICAL_BRANCHES": int(update.cache_total_logical_branches),
        "CACHE_FULL_EXACT_AUDIT_BRANCHES": int(update.cache_full_exact_audit_branches),
        "CACHE_NORMAL_SLOT_COUNTERFACTUAL_BRANCHES": int(
            update.cache_normal_slot_counterfactual_branches
        ),
        "CACHE_CANDIDATE_VALUE_SOURCE": str(update.cache_candidate_value_source),
        "CACHE_ACTUAL_ACTION_Q_ANCHOR": str(update.cache_actual_action_q_anchor),
        "CACHE_HELDOUT_AUDIT_BEFORE_TRAINING": bool(
            update.cache_heldout_audit_before_training
        ),
        "CACHE_Q_CRITIC_USABLE_FOR_POLICY_CREDIT": str(
            update.cache_q_critic_usable_for_policy_credit
        ),
    }
    fields.update(hybrid_cache_metrics)
    runtime_metrics = {
        "CACHE_COUNTERFACTUAL_EXECUTION_REDUCTION_FRACTION": float(
            update.cache_counterfactual_execution_reduction_fraction
        ),
        "SURROGATE_INFERENCE_RUNTIME_SECONDS": float(
            update.cache_surrogate_inference_runtime_seconds
        ),
        "SURROGATE_TRAIN_RUNTIME_SECONDS": float(
            update.cache_surrogate_train_runtime_seconds
        ),
        "CACHE_SURROGATE_DELTA_Q_MAE": float(update.cache_surrogate_delta_q_mae),
        "CACHE_SURROGATE_DELTA_Q_RMSE": float(update.cache_surrogate_delta_q_rmse),
        "CACHE_SURROGATE_RANK_CORRELATION": float(
            update.cache_surrogate_rank_correlation
        ),
        "CACHE_SURROGATE_UNCERTAINTY_MEAN": float(
            update.cache_surrogate_uncertainty_mean
        ),
        "CACHE_SURROGATE_UNCERTAINTY_P90": float(
            update.cache_surrogate_uncertainty_p90
        ),
        "CACHE_Q_CRITIC_MAE": float(update.cache_q_critic_mae),
        "CACHE_Q_CRITIC_RMSE": float(update.cache_q_critic_rmse),
        "CACHE_Q_CRITIC_RANK_CORRELATION": float(
            update.cache_q_critic_rank_correlation
        ),
        "CACHE_Q_CRITIC_ARGMAX_ACCURACY": float(
            update.cache_q_critic_argmax_accuracy
        ),
        "CACHE_Q_CRITIC_RECALL_AT_3": float(update.cache_q_critic_recall_at_3),
        "CACHE_Q_CRITIC_RECALL_AT_5": float(update.cache_q_critic_recall_at_5),
        "CACHE_Q_CRITIC_NORMALIZED_REGRET_P95": float(
            update.cache_q_critic_normalized_regret_p95
        ),
        "CACHE_Q_CRITIC_TRAIN_SECONDS": float(update.cache_q_critic_train_seconds),
        "CACHE_Q_INFERENCE_SECONDS": float(update.cache_q_critic_inference_seconds),
        "CACHE_AUDIT_EXACT_SECONDS": float(update.cache_audit_exact_seconds),
    }
    fields.update(
        {
            key: value
            for key, value in runtime_metrics.items()
            if math.isfinite(float(value))
        }
    )
    if update.cache_surrogate_audit_statistics is not None:
        fields["CACHE_SURROGATE_AUDIT_STATISTICS"] = dict(
            update.cache_surrogate_audit_statistics
        )
    if update.cache_q_critic_selected_exact_regret is not None:
        fields["CACHE_Q_CRITIC_SELECTED_EXACT_REGRET"] = dict(
            update.cache_q_critic_selected_exact_regret
        )
    if update.cache_q_critic_audit_metrics is not None:
        fields["CACHE_Q_CRITIC_AUDIT_METRICS"] = dict(
            update.cache_q_critic_audit_metrics
        )
    if math.isfinite(float(update.uav_operation_credit_runtime_seconds)):
        fields["uav_operation_credit_runtime_seconds"] = float(
            update.uav_operation_credit_runtime_seconds
        )
        fields["uav_operation_counterfactual_env_executions"] = int(
            update.uav_operation_counterfactual_env_executions
        )
        fields.update(
            {
                "UAV_EXACT_Q_BRANCH_REQUEST_COUNT": int(
                    update.uav_exact_q_branch_request_count
                ),
                "UAV_EXACT_Q_UNIQUE_BRANCH_COUNT": int(
                    update.uav_exact_q_unique_branch_count
                ),
                "UAV_EXACT_Q_DUPLICATE_BRANCH_COUNT": int(
                    update.uav_exact_q_duplicate_branch_count
                ),
                "UAV_EXACT_Q_CACHE_HIT_COUNT": int(
                    update.uav_exact_q_cache_hit_count
                ),
                "EXECUTE_SLOT_CALL_COUNT": int(
                    update.uav_exact_q_execute_slot_call_count
                ),
                "MATCHING_CALL_COUNT": int(update.uav_exact_q_matching_call_count),
                # This is the exact KKT resource-cache request count reported
                # by the unchanged Matching/KKT path. The targeted probe also
                # records bottom-level solver calls separately.
                "KKT_CALL_COUNT": int(update.uav_exact_q_kkt_call_count),
            }
        )
    routing_metrics = {
        "UAV_OPERATION_FULL_ROLLOUT_KL": float(
            update.uav_operation_full_rollout_kl
        ),
        "uav_operation_head_parameter_delta_norm": float(
            update.uav_operation_head_parameter_delta_norm
        ),
        "uav_encoder_parameter_delta_norm": float(
            update.uav_encoder_parameter_delta_norm
        ),
    }
    if all(math.isfinite(value) for value in routing_metrics.values()):
        fields.update(routing_metrics)
    fields.update(
        {
            "UAV_OPERATION_STEPS_EXECUTED": int(update.uav_operation_steps_executed),
            "UAV_OPERATION_STEPS_SUPPRESSED": int(update.uav_operation_steps_suppressed),
            "UAV_CANDIDATE_FULL_STEPS": int(update.uav_candidate_full_steps),
            "UAV_CANDIDATE_ONLY_STEPS_AFTER_OPERATION_STOP": int(
                update.uav_candidate_only_steps_after_operation_stop
            ),
        }
    )
    if update.uav_operation_teacher_diagnostics is not None:
        fields.update(dict(update.uav_operation_teacher_diagnostics))
    if update.uav_operation_diagnostics is not None:
        fields.update(dict(update.uav_operation_diagnostics))
    encoder_gradient_norms = {
        "cache_encoder_gradient_norm": float(update.cache_encoder_gradient_norm),
        "uav_encoder_gradient_norm": float(update.uav_encoder_gradient_norm),
    }
    if all(math.isfinite(value) for value in encoder_gradient_norms.values()):
        fields.update(encoder_gradient_norms)
    return fields


def _branch_kl_update_metric_fields(update: UpdateResult) -> dict[str, Any]:
    """Return branch-specific minibatch/epoch KL diagnostics for update JSONL."""

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
    config_path: str = "configs/proposed/system_config.toml",
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
        operation_advantage_normalization_mode=str(
            get(
                cfg,
                "algorithm.ppo.operation_advantage_normalization_mode",
                DEFAULT_OPERATION_ADVANTAGE_NORMALIZATION_MODE,
            )
        ),
        operation_advantage_reference_scale=float(
            get(
                cfg,
                "algorithm.ppo.operation_advantage_reference_scale",
                DEFAULT_OPERATION_ADVANTAGE_REFERENCE_SCALE,
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
    if values.operation_advantage_normalization_mode not in {
        OPERATION_ADVANTAGE_NORMALIZATION_STD_ONLY,
        OPERATION_ADVANTAGE_NORMALIZATION_FIXED_REFERENCE_SCALE,
    }:
        raise ValueError(
            "operation_advantage_normalization_mode 只支持 "
            "std_only/fixed_reference_scale。"
        )
    if (
        not math.isfinite(values.operation_advantage_reference_scale)
        or values.operation_advantage_reference_scale <= 0.0
    ):
        raise ValueError(
            "operation_advantage_reference_scale 必须是正的有限数。"
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


_LEARNING_RATE_METADATA_FIELDS = (
    "experiment_id",
    "policy_schema",
    "control_mode",
    "configured_actor_learning_rate",
    "configured_critic_learning_rate",
    "effective_actor_learning_rate",
    "effective_critic_learning_rate",
    "actor_learning_rate_overridden",
    "critic_learning_rate_overridden",
    "lr_experiment_axis",
    "lr_experiment_value",
)
_LEARNING_RATE_METADATA_FLOAT_FIELDS = {
    "configured_actor_learning_rate",
    "configured_critic_learning_rate",
    "effective_actor_learning_rate",
    "effective_critic_learning_rate",
    "lr_experiment_value",
}


def _validate_learning_rate_value(value: float | None, field: str) -> float:
    if value is None:
        raise ValueError(f"{field} must be a positive finite number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{field} must be a positive finite number")
    return resolved


def _validate_learning_rate_metadata(
    checkpoint_metadata: Any,
    expected_metadata: dict[str, Any],
) -> None:
    """Reject resuming a run with a different effective learning-rate axis."""

    if not isinstance(checkpoint_metadata, dict):
        raise ValueError(
            "Resume rejected: checkpoint learning-rate metadata is missing"
        )
    for field in _LEARNING_RATE_METADATA_FIELDS:
        checkpoint_value = checkpoint_metadata.get(field)
        expected_value = expected_metadata.get(field)
        if field in _LEARNING_RATE_METADATA_FLOAT_FIELDS:
            if checkpoint_value is None or expected_value is None:
                matches = checkpoint_value is expected_value
            else:
                matches = math.isclose(
                    float(checkpoint_value),
                    float(expected_value),
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
        else:
            matches = checkpoint_value == expected_value
        if not matches:
            raise ValueError(
                "Resume rejected: checkpoint "
                f"{field}={checkpoint_value!r} expected={expected_value!r}"
            )


class PPOV2Trainer:
    """PPO core for the proposed mainline and explicit baseline paths."""

    def __init__(
        self,
        context: EnvironmentContext,
        device: str = "cpu",
        *,
        actor_learning_rate_override: float | None = None,
        critic_learning_rate_override: float | None = None,
        lr_experiment_axis: str | None = None,
        lr_experiment_value: float | None = None,
    ) -> None:
        _require_torch()
        self.context = context
        self.hyperparameters = hyperparameters_from_project(context.cfg)
        self.configured_actor_learning_rate = float(
            self.hyperparameters.actor_learning_rate
        )
        self.configured_critic_learning_rate = float(
            self.hyperparameters.critic_learning_rate
        )
        self.actor_learning_rate_overridden = actor_learning_rate_override is not None
        self.critic_learning_rate_overridden = critic_learning_rate_override is not None
        self.effective_actor_learning_rate = (
            self.configured_actor_learning_rate
            if actor_learning_rate_override is None
            else _validate_learning_rate_value(
                actor_learning_rate_override,
                "actor_learning_rate_override",
            )
        )
        self.effective_critic_learning_rate = (
            self.configured_critic_learning_rate
            if critic_learning_rate_override is None
            else _validate_learning_rate_value(
                critic_learning_rate_override,
                "critic_learning_rate_override",
            )
        )
        if lr_experiment_axis is None and actor_learning_rate_override is not None:
            lr_experiment_axis = "actor_learning_rate"
        self.lr_experiment_axis = (
            None if lr_experiment_axis is None else str(lr_experiment_axis)
        )
        if self.lr_experiment_axis == "actor_learning_rate":
            resolved_lr_experiment_value = self.effective_actor_learning_rate
        elif self.lr_experiment_axis == "critic_learning_rate":
            resolved_lr_experiment_value = self.effective_critic_learning_rate
        elif self.lr_experiment_axis is None:
            resolved_lr_experiment_value = None
        else:
            raise ValueError(
                "lr_experiment_axis must be actor_learning_rate, "
                "critic_learning_rate, or omitted"
            )
        if lr_experiment_value is not None and resolved_lr_experiment_value is None:
            raise ValueError(
                "lr_experiment_value requires lr_experiment_axis"
            )
        if lr_experiment_value is not None:
            supplied_lr_experiment_value = _validate_learning_rate_value(
                lr_experiment_value,
                "lr_experiment_value",
            )
            if not math.isclose(
                supplied_lr_experiment_value,
                float(resolved_lr_experiment_value),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise ValueError(
                    "lr_experiment_value must match the effective learning rate "
                    f"for axis {self.lr_experiment_axis!r}"
                )
        self.lr_experiment_value = resolved_lr_experiment_value
        self.candidate_policy_improvement = str(
            project_config.get_config_value(
                context.cfg,
                "algorithm.ppo.candidate_policy_improvement",
                CANDIDATE_POLICY_IMPROVEMENT_LEGACY_PPO,
            )
        )
        if self.candidate_policy_improvement not in {
            CANDIDATE_POLICY_IMPROVEMENT_LEGACY_PPO,
            CANDIDATE_POLICY_IMPROVEMENT_EXACT_Q_LISTWISE,
        }:
            raise ValueError(
                "algorithm.ppo.candidate_policy_improvement 只支持 "
                "LEGACY_PPO/EXACT_Q_LISTWISE。"
            )
        self.exact_q_listwise_temperature = float(
            project_config.get_config_value(
                context.cfg,
                "algorithm.ppo.exact_q_listwise_temperature",
                DEFAULT_EXACT_Q_LISTWISE_TEMPERATURE,
            )
        )
        if (
            not math.isfinite(self.exact_q_listwise_temperature)
            or self.exact_q_listwise_temperature <= 0.0
        ):
            raise ValueError("exact_q_listwise_temperature 必须是正的有限数。")
        if (
            self.candidate_policy_improvement
            == CANDIDATE_POLICY_IMPROVEMENT_EXACT_Q_LISTWISE
            and context.policy_schema not in PROPOSED_POLICY_SCHEMAS
        ):
            raise ValueError(
                "EXACT_Q_LISTWISE candidate improvement is reserved for the proposed joint schema."
            )
        if (
            context.policy_schema in PROPOSED_POLICY_SCHEMAS
            and self.candidate_policy_improvement
            != CANDIDATE_POLICY_IMPROVEMENT_EXACT_Q_LISTWISE
        ):
            raise ValueError(
                "the proposed joint schema requires candidate_policy_improvement=EXACT_Q_LISTWISE"
            )
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
        if self.cache_kl_control == CACHE_KL_CONTROL_MINIBATCH_EARLY_STOP:
            raise ValueError(
                "minibatch_early_stop cache KL control is not part of the proposed mainline."
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
            and context.policy_schema not in BRANCH_SPECIFIC_KL_POLICY_SCHEMAS
        ):
            raise ValueError(
                "branch_specific KL gate requires the proposed policy schema"
            )
        if (
            context.policy_schema in BRANCH_SPECIFIC_KL_POLICY_SCHEMAS
            and self.kl_gate_mode != KL_GATE_MODE_BRANCH_SPECIFIC
        ):
            raise ValueError("The proposed policy schema requires kl_gate_mode=branch_specific")
        self.device = torch.device(device)
        if context.configuration_control_mode in {
            "policy_primitive",
            "policy_primitive_cache_operation",
            "policy_primitive_joint_cache_uav_operation",
            "adaptive_joint_cache_uav_reconfiguration",
        }:
            primitive_spec = PrimitiveTensorSpec.create(context)
            policy_class: type[AdaptivePrimitiveActorCritic]
            if context.configuration_control_mode in {
                "policy_primitive_joint_cache_uav_operation",
                "adaptive_joint_cache_uav_reconfiguration",
            }:
                if context.policy_schema not in PROPOSED_POLICY_SCHEMAS:
                    raise ValueError(
                        "the proposed joint reconfiguration control mode requires the proposed joint policy schema"
                    )
                policy_class = AdaptiveJointCacheUAVPPOActorCritic
                self.policy = policy_class(
                    int(context.observation_spec.observation_dim),
                    int(context.action_spec.action_dim),
                    primitive_spec.cache_feature_dim,
                    primitive_spec.uav_feature_dim,
                    primitive_spec.num_cache_heads,
                    self.hyperparameters.hidden_sizes,
                    num_uav_heads=primitive_spec.num_uav_heads,
                ).to(self.device)
            elif context.configuration_control_mode == "policy_primitive_cache_operation":
                if context.policy_schema not in CACHE_OPERATION_POLICY_SCHEMAS:
                    raise ValueError(
                        "policy_primitive_cache_operation requires a supported "
                        "Cache operation policy schema"
                    )
                policy_class = (
                    AdaptiveNoPredictionCacheOnlyActorCritic
                    if context.policy_schema == NO_PREDICTION_POLICY_SCHEMA
                    else (
                        AdaptiveReplicaAwareCacheOnlyActorCritic
                        if context.policy_schema
                        == AdaptiveReplicaAwareCacheOnlyActorCritic.architecture_id
                        else AdaptiveCacheOnlyActorCritic
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
        if (
            not self._uses_separate_optimization()
            and (
                actor_learning_rate_override is not None
                or critic_learning_rate_override is not None
            )
        ):
            raise ValueError(
                "actor/critic learning-rate overrides require separate actor/critic optimization"
            )

        # Cache counterfactual screening is deliberately a separate learning
        # system.  The ensemble is not attached to ``self.policy`` and its
        # optimizer is never used by PPO, so surrogate regression cannot send
        # gradients into the Cache encoder/scorer or the actor/critic.
        self.cache_surrogate_ensemble: Any | None = None
        self.cache_surrogate_optimizer: Any | None = None
        self.cache_surrogate_context_dim: int | None = None
        self.cache_surrogate_feature_dim: int | None = None
        self.cache_surrogate_training_count = 0
        self.cache_surrogate_shadow_complete = False
        self.cache_surrogate_screening_ready = False
        self.cache_surrogate_screening_active = False
        self.cache_surrogate_selected_k: int | None = None
        self.cache_surrogate_audit_history: list[dict[str, Any]] = []
        self.cache_surrogate_audit_statistics: dict[str, Any] | None = None
        self.cache_surrogate_audit_interval = CACHE_SURROGATE_AUDIT_INTERVAL
        self.cache_surrogate_rng = np.random.default_rng(0xCACE5AFE)
        if self._uses_exact_cache_credit():
            primitive_spec = PrimitiveTensorSpec.create(context)
            self.cache_surrogate_context_dim = int(
                context.observation_spec.observation_dim
            )
            self.cache_surrogate_feature_dim = int(primitive_spec.cache_feature_dim)
            self.cache_surrogate_ensemble = torch.nn.ModuleList(
                [
                    CacheCounterfactualDeltaQSurrogate(
                        self.cache_surrogate_context_dim,
                        self.cache_surrogate_feature_dim,
                        CACHE_SURROGATE_HIDDEN_DIM,
                    )
                    for _ in range(CACHE_SURROGATE_ENSEMBLE_SIZE)
                ]
            ).to(self.device)
            self.cache_surrogate_optimizer = torch.optim.Adam(
                self.cache_surrogate_ensemble.parameters(),
                lr=CACHE_SURROGATE_LEARNING_RATE,
            )
        # The proposed Cache mainline uses one fresh, policy-independent
        # action-conditioned residual Q critic.  It is intentionally separate
        # from the PPO global critic: the latter supplies detached V(s), while
        # this module learns only A_phi(s, a).
        self.cache_action_q_context_dim: int | None = None
        self.cache_action_q_node_count: int | None = None
        self.cache_action_q_feature_dim: int | None = None
        self.cache_action_q_critic: Any | None = None
        self.cache_action_q_optimizer: Any | None = None
        self.cache_action_q_training_count = 0
        self.cache_action_q_optimizer_steps = 0
        self.cache_action_q_audit_interval = CACHE_ACTION_Q_AUDIT_INTERVAL
        self.cache_action_q_audit_statistics: dict[str, Any] | None = None
        self.cache_action_q_sparse_mode = False
        if self._uses_exact_cache_credit():
            primitive_spec = PrimitiveTensorSpec.create(context)
            self.cache_action_q_context_dim = int(context.observation_spec.observation_dim)
            self.cache_action_q_node_count = int(primitive_spec.num_cache_heads)
            self.cache_action_q_feature_dim = int(primitive_spec.cache_feature_dim)
            self.cache_action_q_critic = CacheActionQCritic(
                self.cache_action_q_context_dim,
                self.cache_action_q_node_count,
                self.cache_action_q_feature_dim,
                CACHE_ACTION_Q_HIDDEN_DIM,
            ).to(self.device)
            policy_parameter_ids = {id(parameter) for parameter in self.policy.parameters()}
            action_q_parameter_ids = {
                id(parameter) for parameter in self.cache_action_q_critic.parameters()
            }
            if policy_parameter_ids & action_q_parameter_ids:
                raise RuntimeError("Cache Action-Q critic shares parameters with PPO policy")
            self.cache_action_q_optimizer = torch.optim.Adam(
                self.cache_action_q_critic.parameters(),
                lr=CACHE_ACTION_Q_LEARNING_RATE,
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
                lr=self.effective_actor_learning_rate,
            )
            self.critic_optimizer = torch.optim.Adam(
                critic_parameters,
                lr=self.effective_critic_learning_rate,
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

    def learning_rate_metadata(self) -> dict[str, Any]:
        """Return the immutable LR identity persisted in manifests/checkpoints."""

        return {
            "experiment_id": str(
                project_config.get_config_value(
                    self.context.cfg,
                    "experiment.name",
                    "unknown",
                )
            ),
            "policy_schema": str(self.policy.architecture_id),
            "control_mode": str(self.context.configuration_control_mode),
            "configured_actor_learning_rate": float(
                self.configured_actor_learning_rate
            ),
            "configured_critic_learning_rate": float(
                self.configured_critic_learning_rate
            ),
            "effective_actor_learning_rate": float(
                self.effective_actor_learning_rate
            ),
            "effective_critic_learning_rate": float(
                self.effective_critic_learning_rate
            ),
            "actor_learning_rate_overridden": bool(
                self.actor_learning_rate_overridden
            ),
            "critic_learning_rate_overridden": bool(
                self.critic_learning_rate_overridden
            ),
            "lr_experiment_axis": self.lr_experiment_axis,
            "lr_experiment_value": (
                None
                if self.lr_experiment_value is None
                else float(self.lr_experiment_value)
            ),
        }

    def _uses_separate_optimization(self) -> bool:
        return self.policy.architecture_id in SEPARATE_OPTIMIZATION_POLICY_SCHEMAS

    def _uses_exact_uav_operation_credit(self) -> bool:
        return self.policy.architecture_id in JOINT_CACHE_UAV_POLICY_SCHEMAS

    def _uses_exact_uav_candidate_credit(self) -> bool:
        return self.policy.architecture_id in CANDIDATE_CREDIT_POLICY_SCHEMAS

    def _uses_exact_q_listwise_candidate(self) -> bool:
        return (
            self.candidate_policy_improvement
            == CANDIDATE_POLICY_IMPROVEMENT_EXACT_Q_LISTWISE
        )

    def _uses_exact_cache_credit(self) -> bool:
        return self.policy.architecture_id in EXACT_CACHE_CREDIT_POLICY_SCHEMAS

    def cache_action_q_metadata(self) -> dict[str, Any]:
        """Return formal Cache-credit metadata without advertising Action-Q."""

        return {
            "CACHE_POLICY_CREDIT_SOURCE": CACHE_POLICY_CREDIT_SOURCE,
            "CACHE_ACTION_Q_CRITIC_INITIALIZED": bool(
                self.cache_action_q_critic is not None
            ),
            "CACHE_ACTION_Q_USED_FOR_POLICY_CREDIT": CACHE_ACTION_Q_USED_FOR_POLICY_CREDIT,
            "CACHE_COUNTERFACTUAL_MODE": CACHE_COUNTERFACTUAL_MODE,
            "CACHE_VALUE_MODEL": CACHE_VALUE_MODEL,
            "cache_policy_credit_source": CACHE_POLICY_CREDIT_SOURCE,
            "cache_action_q_critic_initialized": bool(
                self.cache_action_q_critic is not None
            ),
            "cache_action_q_used_for_policy_credit": CACHE_ACTION_Q_USED_FOR_POLICY_CREDIT,
            "cache_counterfactual_mode": CACHE_COUNTERFACTUAL_MODE,
        }

    def cache_surrogate_metadata(self) -> dict[str, Any]:
        """Compatibility alias for old manifest callers."""

        return self.cache_action_q_metadata()

    def _predict_cache_action_q(
        self,
        conditional_contexts: np.ndarray,
        node_identities: np.ndarray,
        is_update: np.ndarray,
        candidate_features: np.ndarray,
    ) -> np.ndarray:
        """Return detached residual Action-Q predictions."""

        if self.cache_action_q_critic is None:
            raise RuntimeError("Cache Action-Q critic is not initialized")
        with torch.no_grad():
            residual = self.cache_action_q_critic(
                conditional_contexts,
                node_identities,
                is_update,
                candidate_features,
            ).detach().cpu().numpy().reshape(-1)
        if not np.all(np.isfinite(residual)):
            raise FloatingPointError("Cache Action-Q residual prediction is non-finite")
        return residual.astype(np.float64)

    def _train_cache_action_q(
        self,
        conditional_contexts: np.ndarray,
        node_identities: np.ndarray,
        is_update: np.ndarray,
        candidate_features: np.ndarray,
        target_residual: np.ndarray,
    ) -> dict[str, float | int]:
        """Fit only the independent Action-Q residual critic."""

        if self.cache_action_q_critic is None or self.cache_action_q_optimizer is None:
            raise RuntimeError("Cache Action-Q optimizer is not initialized")
        contexts = np.asarray(conditional_contexts, dtype=np.float32)
        nodes = np.asarray(node_identities, dtype=np.float32)
        updates = np.asarray(is_update, dtype=np.float32)
        features = np.asarray(candidate_features, dtype=np.float32)
        targets = np.asarray(target_residual, dtype=np.float32).reshape(-1)
        count = len(targets)
        if contexts.shape != (count, int(self.cache_action_q_context_dim)):
            raise ValueError("Cache Action-Q context label shape mismatch")
        if nodes.shape != (count, int(self.cache_action_q_node_count)):
            raise ValueError("Cache Action-Q node label shape mismatch")
        if updates.shape != (count, 1):
            raise ValueError("Cache Action-Q is_update label shape mismatch")
        if features.shape != (count, int(self.cache_action_q_feature_dim)):
            raise ValueError("Cache Action-Q feature label shape mismatch")
        if not all(
            np.all(np.isfinite(value))
            for value in (contexts, nodes, updates, features, targets)
        ):
            raise FloatingPointError("Cache Action-Q labels contain NaN/Inf")
        if count == 0:
            return {"count": 0, "loss": math.nan, "mae": math.nan, "rmse": math.nan}
        context_tensor = torch.as_tensor(contexts, dtype=torch.float32, device=self.device)
        node_tensor = torch.as_tensor(nodes, dtype=torch.float32, device=self.device)
        update_tensor = torch.as_tensor(updates, dtype=torch.float32, device=self.device)
        feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
        self.cache_action_q_optimizer.zero_grad(set_to_none=True)
        prediction = self.cache_action_q_critic(
            context_tensor,
            node_tensor,
            update_tensor,
            feature_tensor,
        )
        loss = torch.mean((prediction - target_tensor) ** 2)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("Cache Action-Q regression loss is non-finite")
        loss.backward()
        if any(parameter.grad is not None for parameter in self.policy.parameters()):
            raise RuntimeError("Cache Action-Q loss produced a PPO policy gradient")
        torch.nn.utils.clip_grad_norm_(
            self.cache_action_q_critic.parameters(), max_norm=10.0
        )
        self.cache_action_q_optimizer.step()
        self.cache_action_q_training_count += count
        self.cache_action_q_optimizer_steps += 1
        with torch.no_grad():
            residual = prediction - target_tensor
            mae = float(torch.mean(torch.abs(residual)).detach().cpu())
            rmse = float(torch.sqrt(torch.mean(residual**2)).detach().cpu())
        return {
            "count": int(count),
            "loss": float(loss.detach().cpu()),
            "mae": mae,
            "rmse": rmse,
        }

    def _predict_cache_surrogate(
        self,
        conditional_context: np.ndarray,
        candidate_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ensemble mean/std for one Cache conditional state."""

        if self.cache_surrogate_ensemble is None:
            raise RuntimeError("Cache surrogate ensemble is not initialized")
        context_tensor = torch.as_tensor(
            np.asarray(conditional_context, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        feature_tensor = torch.as_tensor(
            np.asarray(candidate_features, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            predictions = torch.stack(
                [member(context_tensor, feature_tensor) for member in self.cache_surrogate_ensemble],
                dim=0,
            )
        prediction_array = predictions.detach().cpu().numpy().reshape(
            CACHE_SURROGATE_ENSEMBLE_SIZE, -1
        )
        mu = np.mean(prediction_array, axis=0, dtype=np.float64)
        sigma = np.std(prediction_array, axis=0, dtype=np.float64)
        if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
            raise FloatingPointError("Cache surrogate prediction contains NaN/Inf")
        return mu, sigma

    def _train_cache_surrogate(
        self,
        conditional_contexts: np.ndarray,
        candidate_features: np.ndarray,
        target_delta_q: np.ndarray,
    ) -> dict[str, float | int]:
        """Fit only the independent ensemble on labels from exact branches."""

        if self.cache_surrogate_ensemble is None or self.cache_surrogate_optimizer is None:
            raise RuntimeError("Cache surrogate optimizer is not initialized")
        contexts = np.asarray(conditional_contexts, dtype=np.float32)
        features = np.asarray(candidate_features, dtype=np.float32)
        targets = np.asarray(target_delta_q, dtype=np.float32).reshape(-1)
        if not len(targets):
            return {"count": 0, "loss": math.nan, "mae": math.nan, "rmse": math.nan}
        if contexts.shape != (
            len(targets),
            int(self.cache_surrogate_context_dim),
        ):
            raise ValueError("Cache surrogate context label shape mismatch")
        if features.shape != (
            len(targets),
            int(self.cache_surrogate_feature_dim),
        ):
            raise ValueError("Cache surrogate feature label shape mismatch")
        if not (
            np.all(np.isfinite(contexts))
            and np.all(np.isfinite(features))
            and np.all(np.isfinite(targets))
        ):
            raise FloatingPointError("Cache surrogate labels contain NaN/Inf")
        context_tensor = torch.as_tensor(contexts, dtype=torch.float32, device=self.device)
        feature_tensor = torch.as_tensor(
            features[:, None, :], dtype=torch.float32, device=self.device
        )
        target_tensor = torch.as_tensor(
            targets, dtype=torch.float32, device=self.device
        )
        self.cache_surrogate_optimizer.zero_grad(set_to_none=True)
        predictions = torch.stack(
            [member(context_tensor, feature_tensor).squeeze(-1) for member in self.cache_surrogate_ensemble],
            dim=0,
        )
        loss = torch.mean((predictions - target_tensor.unsqueeze(0)) ** 2)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("Cache surrogate regression loss is non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.cache_surrogate_ensemble.parameters(),
            max_norm=10.0,
        )
        self.cache_surrogate_optimizer.step()
        with torch.no_grad():
            mean_prediction = predictions.mean(dim=0)
            residual = mean_prediction - target_tensor
            mae = float(torch.mean(torch.abs(residual)).detach().cpu())
            rmse = float(torch.sqrt(torch.mean(residual**2)).detach().cpu())
        self.cache_surrogate_training_count += int(len(targets))
        return {
            "count": int(len(targets)),
            "loss": float(loss.detach().cpu()),
            "mae": mae,
            "rmse": rmse,
        }

    def _update_cache_surrogate_gate(self, cache_credit: dict[str, Any]) -> None:
        """Update readiness from shadow/audit evidence and enforce fallback."""

        if not cache_credit:
            return
        gate_metrics = cache_credit.get("gate_metrics")
        if not bool(cache_credit.get("full_exact_evaluation", False)) or not gate_metrics:
            return
        if int(gate_metrics.get("row_count", 0)) <= 0:
            return
        self.cache_surrogate_audit_history.append(dict(gate_metrics))
        self.cache_surrogate_audit_history = self.cache_surrogate_audit_history[
            -CACHE_SURROGATE_AUDIT_WINDOW:
        ]
        self.cache_surrogate_audit_statistics = dict(gate_metrics)
        self.cache_surrogate_audit_statistics["window_size"] = int(
            len(self.cache_surrogate_audit_history)
        )
        recalls: dict[int, list[float]] = {}
        p95_values: dict[int, list[float]] = {}
        for record in self.cache_surrogate_audit_history:
            for key, value in record.get("recall_at_k", {}).items():
                recalls.setdefault(int(key), []).append(float(value))
            for key, value in record.get("normalized_regret_p95_at_k", {}).items():
                p95_values.setdefault(int(key), []).append(float(value))
        eligible: list[int] = []
        window_recall: dict[str, float] = {}
        window_p95: dict[str, float] = {}
        for k in range(1, CACHE_SURROGATE_MAX_K + 1):
            if k not in recalls or k not in p95_values:
                continue
            recall = float(np.mean(recalls[k]))
            p95 = float(np.max(p95_values[k]))
            window_recall[str(k)] = recall
            window_p95[str(k)] = p95
            if recall >= 0.99 and p95 <= 0.01:
                eligible.append(k)
        self.cache_surrogate_audit_statistics.update(
            {
                "window_recall_at_k": window_recall,
                "window_normalized_regret_p95_at_k": window_p95,
            }
        )
        selected_k = min(eligible) if eligible else None
        self.cache_surrogate_shadow_complete = True
        self.cache_surrogate_selected_k = selected_k
        self.cache_surrogate_screening_ready = selected_k is not None
        self.cache_surrogate_screening_active = selected_k is not None

    def _uses_binary_cache_trigger(self) -> bool:
        return self.policy.architecture_id in JOINT_CACHE_UAV_POLICY_SCHEMAS

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
        if self.policy.architecture_id in JOINT_CACHE_UAV_POLICY_SCHEMAS:
            return JOINT_PROPOSED_CHECKPOINT_VERSION
        if self.policy.architecture_id in NO_PREDICTION_POLICY_SCHEMAS:
            return NO_PREDICTION_CACHE_ONLY_CHECKPOINT_VERSION
        if self.policy.architecture_id in REPLICA_AWARE_CACHE_ONLY_POLICY_SCHEMAS:
            return REPLICA_AWARE_CACHE_ONLY_CHECKPOINT_VERSION
        if self.policy.architecture_id in CACHE_ONLY_POLICY_SCHEMAS:
            return CACHE_ONLY_CHECKPOINT_VERSION
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

    def _evaluate_uav_branch_approx_kl_full_rollout(
        self,
        *,
        observations: Any,
        cache_contexts: Any,
        cache_operation_masks: Any,
        cache_operation_actions: Any,
        cache_candidate_features: Any,
        cache_candidate_masks: Any,
        cache_selected_indices: Any,
        uav_contexts: Any,
        uav_operation_masks: Any,
        uav_operation_actions: Any,
        uav_candidate_features: Any,
        uav_candidate_masks: Any,
        uav_selected_indices: Any,
        old_uav_operation_probabilities: Any,
    ) -> float:
        """Evaluate categorical STAY/MOVE KL on the complete frozen rollout.

        This is deliberately a full-distribution operation KL.  It does not
        include candidate log-probabilities and it does not reduce to the
        sampled operation's log-probability delta, so the gate observes the
        complete binary operation policy for every state and UAV head.
        """

        with torch.no_grad():
            evaluation = self.policy.evaluate_joint_cache_uav_operation(
                observations,
                cache_contexts,
                cache_operation_masks,
                cache_operation_actions,
                cache_candidate_features,
                cache_candidate_masks,
                cache_selected_indices,
                uav_contexts,
                uav_operation_masks,
                uav_operation_actions,
                uav_candidate_features,
                uav_candidate_masks,
                uav_selected_indices,
            )
            if evaluation.uav_operation_log_probs_all is None:
                raise RuntimeError(
                    "UAV branch KL evaluation did not expose full operation log-probabilities"
                )
            old_probabilities = torch.as_tensor(
                old_uav_operation_probabilities,
                dtype=torch.float32,
                device=evaluation.uav_operation_log_probs_all.device,
            )
            masks = torch.as_tensor(
                uav_operation_masks,
                dtype=torch.bool,
                device=old_probabilities.device,
            )
            new_log_probabilities = evaluation.uav_operation_log_probs_all
            if tuple(old_probabilities.shape) != tuple(new_log_probabilities.shape):
                raise RuntimeError("UAV full-rollout operation KL shape mismatch")
            if tuple(masks.shape) != tuple(old_probabilities.shape):
                raise RuntimeError("UAV full-rollout operation mask shape mismatch")
            if not bool(torch.all(torch.isfinite(old_probabilities)).item()):
                raise FloatingPointError(
                    "UAV full-rollout old operation probabilities contain NaN/Inf"
                )
            if not bool(torch.all(torch.isfinite(new_log_probabilities[masks])).item()):
                raise FloatingPointError(
                    "UAV full-rollout KL estimator produced NaN/Inf"
                )
            old_log_probabilities = torch.log(old_probabilities.clamp_min(1.0e-12))
            safe_old_log_probabilities = torch.where(
                masks,
                old_log_probabilities,
                torch.zeros_like(old_log_probabilities),
            )
            safe_new_log_probabilities = torch.where(
                masks,
                new_log_probabilities,
                torch.zeros_like(new_log_probabilities),
            )
            kl = torch.sum(
                torch.where(
                    masks,
                    old_probabilities
                    * (safe_old_log_probabilities - safe_new_log_probabilities),
                    torch.zeros_like(old_probabilities),
                ),
                dim=-1,
            )
            if not bool(torch.all(torch.isfinite(kl)).item()):
                raise FloatingPointError(
                    "UAV full-rollout operation KL estimator produced NaN/Inf"
                )
            value = max(float(torch.mean(kl).detach().cpu()), 0.0)
        if not math.isfinite(value):
            raise FloatingPointError("UAV full-rollout KL estimator is non-finite")
        return value

    def _compute_exact_uav_operation_credit(
        self,
        rollout: SimpleRollout,
        *,
        target_transition_indices: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        """Compute detached per-UAV counterfactual one-step TD credit.

        This is intentionally a trainer-side replay of the already collected
        target transition.  Cache actions and every non-target UAV action are
        copied from production; only the target STAY/target-position branch is
        replaced.
        Each branch goes through the real ``execute_slot`` path with a direct
        primitive configuration override, so Matching/KKT and all frozen
        reward/cost semantics remain authoritative.
        """

        if not self._uses_exact_uav_operation_credit():
            return {}
        all_transitions = tuple(rollout.transitions)
        if target_transition_indices is None:
            transitions = all_transitions
        else:
            selected_indices = tuple(int(index) for index in target_transition_indices)
            if not selected_indices:
                raise ValueError("proposed counterfactual target transition selection is empty")
            if len(set(selected_indices)) != len(selected_indices):
                raise ValueError("proposed counterfactual target transition selection repeats an index")
            if any(index < 0 or index >= len(all_transitions) for index in selected_indices):
                raise IndexError("proposed counterfactual target transition index is out of range")
            transitions = tuple(all_transitions[index] for index in selected_indices)
        if not transitions:
            raise ValueError("proposed operation credit requires a non-empty rollout")
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
                raise ValueError("proposed rollout is missing operation counterfactual tensors")

        first_origin = int(transitions[0].origin_index)
        last_origin = max(int(transition.origin_index) for transition in transitions)
        include_bootstrap_step = not bool(
            all(
                bool(transition.done)
                for transition in transitions
                if int(transition.origin_index) == last_origin
            )
        )
        aligned_span = last_origin - first_origin + 1 + int(include_bootstrap_step)
        aligned_steps = aligned_train_steps(
            self.context,
            origin_start=first_origin,
            max_steps=aligned_span,
            include_bootstrap_step=include_bootstrap_step,
        )
        if len(aligned_steps) < aligned_span:
            raise RuntimeError("proposed counterfactual replay is missing aligned next steps")
        aligned_steps_by_origin = {
            int(step.origin_index): step for step in aligned_steps
        }
        selected_aligned_steps = []
        for transition in transitions:
            try:
                selected_aligned_steps.append(
                    aligned_steps_by_origin[int(transition.origin_index)]
                )
            except KeyError as exc:
                raise RuntimeError(
                    "proposed counterfactual replay is missing a selected aligned step"
                ) from exc
        for transition, step in zip(transitions, selected_aligned_steps, strict=True):
            if int(transition.origin_index) != int(step.origin_index):
                raise RuntimeError("proposed counterfactual replay origin alignment failed")

        q_stay = np.zeros((sample_count, head_count), dtype=np.float64)
        q_move = np.zeros((sample_count, head_count), dtype=np.float64)
        baseline = np.zeros((sample_count, head_count), dtype=np.float64)
        move_candidate_counts = np.zeros((sample_count, head_count), dtype=np.int64)
        candidate_capacity = int(transitions[0].uav_candidate_features.shape[1])
        if candidate_capacity <= 0:
            raise RuntimeError("proposed rollout candidate capacity must be positive")
        if any(
            int(transition.uav_candidate_features.shape[1]) != candidate_capacity
            for transition in transitions
        ):
            raise RuntimeError("proposed rollout candidate capacity is not time-aligned")
        # The exact branch evaluator below already produces Q_c for every legal
        # MOVE candidate.  Keep those values and the same frozen-policy
        # probabilities so the candidate update can consume them without another
        # execute_slot/Matching/KKT enumeration.
        candidate_q_values = np.zeros(
            (sample_count, head_count, candidate_capacity), dtype=np.float64
        )
        candidate_probabilities_values = np.zeros_like(candidate_q_values)
        candidate_valid_mask = np.zeros(
            (sample_count, head_count, candidate_capacity), dtype=bool
        )
        candidate_proposal_sources = np.full(
            (sample_count, head_count, candidate_capacity), "", dtype=object
        )
        candidate_move_distances = np.full(
            (sample_count, head_count, candidate_capacity), np.nan, dtype=np.float64
        )
        candidate_weighted_mean_residual = np.zeros(
            (sample_count, head_count), dtype=np.float64
        )
        counterfactual_environment_executions = 0
        branch_request_count = 0
        unique_branch_count = 0
        branch_cache_hit_count = 0
        execute_slot_call_count = 0
        matching_call_count = 0
        kkt_call_count = 0
        started_at = time.perf_counter()

        def _position_dict(source: dict[str, Any]) -> dict[str, np.ndarray]:
            return {
                uav_id: np.asarray(source[uav_id], dtype=np.float64).copy()
                for uav_id in uav_ids
            }

        def _target_action(
            uav_id: str,
            head_index: int,
            candidate_count: int,
            operation: str,
            before: np.ndarray,
            after: np.ndarray,
            selected_index: int | None,
            proposal_source: str | None = None,
        ) -> dict[str, Any]:
            action = {
                "head_type": "uav",
                "head_id": uav_id,
                "head_index": int(head_index),
                "candidate_count": int(candidate_count),
                "selected_index": selected_index,
                "conditional_action": (
                    "STAY" if operation == "STAY" else "TARGET_POSITION"
                ),
                "before_position": [float(value) for value in before],
                "after_position": [float(value) for value in after],
                "actual_moved": bool(operation == "MOVE"),
            }
            if proposal_source is not None:
                action["proposal_source"] = str(proposal_source)
            if operation == "MOVE":
                action["target_position"] = [float(value) for value in after]
            return action

        for transition_index, (transition, step) in enumerate(
            zip(transitions, selected_aligned_steps, strict=True)
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
                raise RuntimeError("proposed production UAV action count mismatch")

            # This memo is deliberately created inside one transition.  The
            # actual tasks, vehicle positions, scenario slice, previous
            # UAV positions, and next exogenous state are therefore fixed by
            # construction; the key still includes every mutable
            # configuration input and never uses candidate index as identity.
            slot_branch_cache: dict[
                tuple[Any, ...], tuple[float, np.ndarray | None]
            ] = {}
            previous_uav_key = tuple(
                (
                    str(key),
                    float(np.asarray(previous_uav[key], dtype=np.float64)[0]),
                    float(np.asarray(previous_uav[key], dtype=np.float64)[1]),
                )
                for key in uav_ids
            )

            for head_index, uav_id in enumerate(uav_ids):
                # Reconstruct exactly the sequential conditional state used
                # by the production sampler for this head.
                current_uav = _position_dict(transition.previous_uav_positions)
                enumeration_triggers = {candidate_id: 0 for candidate_id in uav_ids}
                for prior_index, prior_id in enumerate(uav_ids[:head_index]):
                    prior_action = production_uav_actions[prior_index]
                    if str(prior_action.get("conditional_action", "")).upper() == "TARGET_POSITION":
                        current_uav[prior_id] = np.asarray(
                            prior_action.get(
                                "target_position", prior_action["after_position"]
                            ),
                            dtype=np.float64,
                        )
                        enumeration_triggers[prior_id] = 1
                enumeration_triggers[uav_id] = 1
                candidates = uav_configuration.enumerate_uav_target_candidates(
                    current_uav,
                    previous_uav,
                    uav_id,
                    enumeration_triggers,
                    self.context.cfg,
                    self.context.flight_boundary,
                    scenario_slice=step.scenario_slice,
                    cache_state=transition.next_cache_state,
                    cache_nodes=self.context.cache_nodes,
                    predicted_vehicle_positions=step.predicted_vehicle_positions,
                )
                move_candidates = candidates
                move_count = len(move_candidates)
                move_candidate_counts[transition_index, head_index] = move_count
                if move_count > int(transition.uav_candidate_features.shape[1]):
                    raise RuntimeError("proposed legal UAV candidates exceed rollout capacity")

                raw_probabilities = np.asarray(
                    transition.uav_operation_probabilities[head_index],
                    dtype=np.float64,
                ).reshape(-1)
                operation_mask = np.asarray(
                    transition.uav_operation_masks[head_index], dtype=bool
                ).reshape(-1)
                if raw_probabilities.shape != (2,) or operation_mask.shape != (2,):
                    raise RuntimeError("proposed UAV operation probability/mask shape mismatch")
                if move_count > 0 and bool(operation_mask[1]):
                    probability_sum = float(raw_probabilities.sum())
                    if (
                        not np.all(np.isfinite(raw_probabilities))
                        or probability_sum <= 0.0
                    ):
                        raise FloatingPointError(
                            "proposed UAV operation probabilities are non-finite"
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
                            "proposed candidate probabilities are non-finite or do not sum to one"
                        )
                    if self._uses_exact_uav_candidate_credit():
                        candidate_probability_sum = float(candidate_probabilities.sum())
                        candidate_probabilities = (
                            candidate_probabilities / candidate_probability_sum
                        )
                else:
                    candidate_probabilities = np.zeros(0, dtype=np.float64)

                next_step = (
                    None
                    if bool(transition.done)
                    else aligned_steps_by_origin.get(int(transition.origin_index) + 1)
                )
                if not bool(transition.done) and next_step is None:
                    raise RuntimeError(
                        "proposed counterfactual replay is missing the next aligned step"
                    )

                def _evaluate_branch(
                    operation: str,
                    selected_index: int | None,
                    before: np.ndarray,
                    after: np.ndarray,
                ) -> tuple[float, np.ndarray | None]:
                    nonlocal counterfactual_environment_executions
                    nonlocal branch_request_count, unique_branch_count
                    nonlocal branch_cache_hit_count
                    nonlocal execute_slot_call_count, matching_call_count, kkt_call_count
                    branch_final_uav = {
                        key: value.copy() for key, value in production_final_uav.items()
                    }
                    branch_final_uav[uav_id] = np.asarray(after, dtype=np.float64).copy()
                    branch_uav_triggers = dict(production_uav_triggers)
                    branch_uav_triggers[uav_id] = int(operation == "MOVE")
                    branch_key = (
                        int(transition.origin_index),
                        np.asarray(previous_cache, dtype=np.int8).tobytes(),
                        np.asarray(transition.next_cache_state, dtype=np.int8).tobytes(),
                        previous_uav_key,
                        tuple(
                            (
                                str(key),
                                float(np.asarray(branch_final_uav[key], dtype=np.float64)[0]),
                                float(np.asarray(branch_final_uav[key], dtype=np.float64)[1]),
                            )
                            for key in uav_ids
                        ),
                        tuple(
                            (str(key), int(branch_uav_triggers[key]))
                            for key in uav_ids
                        ),
                    )
                    branch_request_count += 1
                    cached = slot_branch_cache.get(branch_key)
                    if cached is not None:
                        branch_cache_hit_count += 1
                        cached_reward, cached_observation = cached
                        return float(cached_reward), (
                            None
                            if cached_observation is None
                            else np.asarray(cached_observation, dtype=np.float32).copy()
                        )
                    unique_branch_count += 1
                    branch_uav_actions = list(production_uav_actions)
                    branch_uav_actions[head_index] = _target_action(
                        uav_id,
                        head_index,
                        move_count,
                        operation,
                        np.asarray(before, dtype=np.float64),
                        np.asarray(after, dtype=np.float64),
                        selected_index,
                        (
                            move_candidates[selected_index].proposal_source
                            if operation == "MOVE" and selected_index is not None
                            else None
                        ),
                    )
                    configuration_override = (
                        slot_environment.build_policy_target_configuration(
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
                    execute_slot_call_count += 1
                    matching_call_count += 1
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
                    kkt_call_count += int(
                        result.matching_summary.get("kkt_cache_requests", 0)
                    )
                    if not result.step_feasible:
                        raise RuntimeError(
                            "proposed counterfactual transition is infeasible: "
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
                        next_observation = None
                    else:
                        next_observation = _intermediate_observation(
                            np.asarray(result.next_cache_state, dtype=np.int8),
                            result.next_uav_positions,
                            next_step,
                            self.context,
                        )
                    safe_observation = (
                        None
                        if next_observation is None
                        else np.asarray(next_observation, dtype=np.float32).copy()
                    )
                    slot_branch_cache[branch_key] = (float(reward), safe_observation)
                    return float(reward), (
                        None
                        if safe_observation is None
                        else safe_observation.copy()
                    )

                target_before = current_uav[uav_id].copy()
                branch_rewards: list[float] = []
                branch_observations: list[np.ndarray | None] = []
                stay_reward, stay_observation = _evaluate_branch(
                    "STAY",
                    None,
                    target_before,
                    target_before,
                )
                branch_rewards.append(stay_reward)
                branch_observations.append(stay_observation)
                if move_count:
                    for candidate_index, candidate in enumerate(move_candidates):
                        move_reward, move_observation = _evaluate_branch(
                            "MOVE",
                            candidate_index,
                            np.asarray(candidate.before_position, dtype=np.float64),
                            np.asarray(candidate.target_position, dtype=np.float64),
                        )
                        branch_rewards.append(move_reward)
                        branch_observations.append(move_observation)

                if next_step is None:
                    next_values = np.zeros(len(branch_rewards), dtype=np.float64)
                else:
                    if len(branch_observations) != len(branch_rewards) or any(
                        observation is None for observation in branch_observations
                    ):
                        raise RuntimeError(
                            "proposed counterfactual branch observation batch is incomplete"
                        )
                    # Every branch above has already gone through the exact
                    # environment/Matching/KKT path.  Only the critic inference
                    # is batched here; branch order is STAY followed by the
                    # stable legal MOVE candidate order.
                    next_observation_batch = np.stack(
                        [
                            np.asarray(observation, dtype=np.float32)
                            for observation in branch_observations
                        ],
                        axis=0,
                    )
                    with torch.no_grad():
                        if not hasattr(self.policy, "critic_value"):
                            raise RuntimeError(
                                "proposed policy does not expose the exact critic_value path"
                            )
                        value_tensor = self.policy.critic_value(next_observation_batch)
                    next_values = (
                        value_tensor.detach().cpu().numpy().reshape(-1).astype(np.float64)
                    )
                    if next_values.shape != (len(branch_rewards),):
                        raise RuntimeError(
                            "proposed counterfactual critic batch shape mismatch"
                        )
                    # Preserve the historical scalar STAY critic result exactly.
                    # The MOVE candidates still use the batched path; this one
                    # extra scalar call is negligible relative to the exact
                    # Matching/KKT branches and keeps the operation-side PPO
                    # loss within its stricter numerical equivalence gate.
                    with torch.no_grad():
                        _, stay_value_tensor = self.policy(branch_observations[0])
                    next_values[0] = float(
                        stay_value_tensor.detach().cpu().reshape(-1)[0]
                    )
                q_values = np.asarray(branch_rewards, dtype=np.float64)
                q_values += float(self.hyperparameters.gamma) * next_values
                if not np.all(np.isfinite(q_values)):
                    raise FloatingPointError("proposed counterfactual Q is non-finite")
                stay_q = float(q_values[0])
                q_stay[transition_index, head_index] = stay_q
                if move_count:
                    move_values = q_values[1:]
                    candidate_q_values[
                        transition_index, head_index, :move_count
                    ] = move_values
                    candidate_proposal_sources[
                        transition_index, head_index, :move_count
                    ] = [str(candidate.proposal_source) for candidate in move_candidates]
                    candidate_move_distances[
                        transition_index, head_index, :move_count
                    ] = [
                        float(
                            np.linalg.norm(
                                np.asarray(candidate.target_position, dtype=np.float64)
                                - np.asarray(candidate.before_position, dtype=np.float64)
                            )
                        )
                        for candidate in move_candidates
                    ]
                    candidate_probabilities_values[
                        transition_index, head_index, :move_count
                    ] = candidate_probabilities
                    candidate_valid_mask[
                        transition_index, head_index, :move_count
                    ] = True
                    q_move[transition_index, head_index] = float(
                        np.dot(candidate_probabilities, move_values)
                    )
                    candidate_weighted_mean_residual[
                        transition_index, head_index
                    ] = float(
                        np.dot(
                            candidate_probabilities,
                            move_values - q_move[transition_index, head_index],
                        )
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
            raise FloatingPointError("proposed operation advantages are non-finite")
        operation_normalization_mode = (
            self.hyperparameters.operation_advantage_normalization_mode
        )
        if (
            operation_normalization_mode
            == OPERATION_ADVANTAGE_NORMALIZATION_STD_ONLY
        ):
            population_std = float(np.std(raw_advantages, ddof=0))
            if not math.isfinite(population_std):
                raise FloatingPointError("proposed operation advantage std is non-finite")
            scaled_advantages = raw_advantages / (population_std + 1.0e-8)
            advantage_normalization = "STD_ONLY"
            advantage_reference_scale = float(population_std)
        elif (
            operation_normalization_mode
            == OPERATION_ADVANTAGE_NORMALIZATION_FIXED_REFERENCE_SCALE
        ):
            advantage_reference_scale = float(
                self.hyperparameters.operation_advantage_reference_scale
            )
            if (
                not math.isfinite(advantage_reference_scale)
                or advantage_reference_scale <= 0.0
            ):
                raise FloatingPointError(
                    "fixed-reference fixed operation advantage reference scale is invalid"
                )
            scaled_advantages = raw_advantages / advantage_reference_scale
            advantage_normalization = "FIXED_REFERENCE_SCALE"
        else:  # pragma: no cover - validated by hyperparameters_from_project
            raise ValueError(
                "unsupported operation advantage normalization mode: "
                f"{operation_normalization_mode}"
            )
        nonzero = raw_advantages != 0.0
        sign_flip_count = int(
            np.count_nonzero(
                nonzero
                & (np.signbit(raw_advantages) != np.signbit(scaled_advantages))
            )
        )
        if sign_flip_count != 0:
            raise FloatingPointError("proposed std-only operation scaling flipped a sign")
        if not np.all(np.isfinite(scaled_advantages)):
            raise FloatingPointError("proposed scaled operation advantages are non-finite")
        return {
            "raw_advantages": raw_advantages.astype(np.float32),
            "scaled_advantages": scaled_advantages.astype(np.float32),
            "q_stay": q_stay.astype(np.float32),
            "q_move": q_move.astype(np.float32),
            "baseline": baseline.astype(np.float32),
            "counterfactual_environment_executions": int(
                counterfactual_environment_executions
            ),
            "uav_exact_q_branch_request_count": int(branch_request_count),
            "uav_exact_q_unique_branch_count": int(unique_branch_count),
            "uav_exact_q_duplicate_branch_count": int(
                branch_request_count - unique_branch_count
            ),
            "uav_exact_q_cache_hit_count": int(branch_cache_hit_count),
            "execute_slot_call_count": int(execute_slot_call_count),
            "matching_call_count": int(matching_call_count),
            # This is the exact number of KKT resource-cache requests made by
            # the Matching path for the unique UAV exact-Q branches. The
            # Matching/KKT implementation itself remains unchanged.
            "kkt_call_count": int(kkt_call_count),
            "mean_move_candidate_count": float(np.mean(move_candidate_counts)),
            "move_candidate_counts": move_candidate_counts,
            "candidate_q_values": candidate_q_values,
            "candidate_probabilities": candidate_probabilities_values,
            "candidate_valid_mask": candidate_valid_mask,
            "candidate_proposal_sources": candidate_proposal_sources,
            "candidate_move_distances": candidate_move_distances,
            "candidate_q_move": q_move.copy(),
            "candidate_weighted_mean_residual": candidate_weighted_mean_residual,
            "credit_runtime_seconds": float(time.perf_counter() - started_at),
            "advantage_normalization": advantage_normalization,
            "advantage_reference_scale": float(advantage_reference_scale),
            "advantage_centering": False,
            "advantage_sign_flip_count": sign_flip_count,
            "ratio_granularity": "OPERATION_ONLY",
        }

    @staticmethod
    def _exact_q_listwise_teacher(
        q_values: Any,
        candidate_valid_mask: Any,
        temperature: float,
    ) -> Any:
        """Build a detached legal-candidate teacher from state-wise exact Q.

        Q values are standardized independently for each state/head before the
        temperature softmax.  Equal-Q legal sets therefore produce a uniform
        teacher, while padded candidates always receive zero mass.
        """

        q_tensor = torch.as_tensor(q_values, dtype=torch.float32)
        valid = torch.as_tensor(
            candidate_valid_mask, dtype=torch.bool, device=q_tensor.device
        )
        if q_tensor.shape != valid.shape or q_tensor.ndim < 1:
            raise ValueError("exact-Q teacher tensor shape mismatch")
        scale = float(temperature)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("exact-Q teacher temperature must be positive and finite")
        if not bool(torch.all(torch.isfinite(q_tensor[valid])).item()):
            raise FloatingPointError("exact-Q teacher received non-finite legal Q values")
        legal_count = valid.sum(dim=-1, keepdim=True)
        active_heads = legal_count > 0
        safe_valid = valid | ~active_heads
        safe_q = torch.where(valid, q_tensor, torch.zeros_like(q_tensor))
        count = legal_count.clamp_min(1).to(torch.float32)
        mean = safe_q.sum(dim=-1, keepdim=True) / count
        centered = torch.where(valid, safe_q - mean, torch.zeros_like(safe_q))
        variance = (centered.square().sum(dim=-1, keepdim=True) / count).clamp_min(0.0)
        std = torch.sqrt(variance)
        standardized = torch.where(
            valid & (std > 1.0e-8),
            centered / std.clamp_min(1.0e-8),
            torch.zeros_like(centered),
        )
        logits = standardized / scale
        logits = logits.masked_fill(~safe_valid, -float("inf"))
        teacher = torch.softmax(logits, dim=-1)
        return torch.where(valid & active_heads, teacher, torch.zeros_like(teacher)).detach()

    @staticmethod
    def _masked_operation_probabilities(probabilities: Any, masks: Any) -> Any:
        """Normalize a binary operation distribution over its legal actions."""

        values = torch.as_tensor(probabilities, dtype=torch.float32)
        legal = torch.as_tensor(masks, dtype=torch.bool, device=values.device)
        if values.shape != legal.shape or values.ndim < 1 or int(values.shape[-1]) != 2:
            raise ValueError("operation probability/mask shape mismatch")
        if not bool(torch.all(torch.isfinite(values)).item()):
            raise FloatingPointError("operation probabilities are non-finite")
        masked = torch.where(legal, values, torch.zeros_like(values))
        normalizer = masked.sum(dim=-1, keepdim=True)
        if bool(torch.any(normalizer <= 0.0).item()):
            raise ValueError("operation mask blocks every operation")
        normalized = masked / normalizer
        if not bool(torch.all(torch.isfinite(normalized)).item()):
            raise FloatingPointError("masked operation probabilities are non-finite")
        if not torch.allclose(
            normalized.sum(dim=-1),
            torch.ones_like(normalized.sum(dim=-1)),
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise FloatingPointError("masked operation probabilities do not sum to one")
        return normalized

    @classmethod
    def _exact_q_operation_teacher(
        cls,
        q_stay: Any,
        candidate_q_values: Any,
        candidate_valid_mask: Any,
        operation_masks: Any,
        temperature: float,
        reference_scale: float,
    ) -> tuple[Any, Any, Any, Any]:
        """Construct the binary operation teacher from the unchanged Q teacher.

        The MOVE value is the legal-candidate Exact-Q teacher expectation.  An
        inactive candidate set has no MOVE value and is represented only by a
        masked STAY teacher; it is never assigned a hard-max or fabricated
        operation target.
        """

        q_values = torch.as_tensor(candidate_q_values, dtype=torch.float32)
        valid = torch.as_tensor(
            candidate_valid_mask, dtype=torch.bool, device=q_values.device
        )
        stay_q = torch.as_tensor(q_stay, dtype=torch.float32, device=q_values.device)
        masks = torch.as_tensor(operation_masks, dtype=torch.bool, device=q_values.device)
        if q_values.shape != valid.shape or q_values.ndim < 1:
            raise ValueError("operation teacher candidate Q shape mismatch")
        if stay_q.shape != q_values.shape[:-1]:
            raise ValueError("operation teacher STAY Q shape mismatch")
        if masks.shape != stay_q.shape + (2,):
            raise ValueError("operation teacher operation mask shape mismatch")
        scale = float(reference_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("operation teacher reference scale must be positive and finite")
        if not bool(torch.all(torch.isfinite(stay_q)).item()):
            raise FloatingPointError("operation teacher STAY Q is non-finite")
        candidate_teacher = cls._exact_q_listwise_teacher(
            q_values,
            valid,
            temperature,
        ).to(q_values.device)
        q_move_teacher = torch.sum(candidate_teacher * q_values, dim=-1)
        move_legal = valid.any(dim=-1)
        if bool(torch.any(move_legal & ~masks[..., 1]).item()) or bool(
            torch.any(~move_legal & masks[..., 1]).item()
        ):
            raise ValueError(
                "Exact-Q candidate legality disagrees with the rollout MOVE mask"
            )
        delta = q_move_teacher - stay_q
        move_probability = torch.sigmoid(delta / scale)
        raw_teacher = torch.stack(
            (1.0 - move_probability, move_probability),
            dim=-1,
        )
        stay_only = torch.zeros_like(raw_teacher)
        stay_only[..., 0] = 1.0
        raw_teacher = torch.where(move_legal.unsqueeze(-1), raw_teacher, stay_only)
        teacher = cls._masked_operation_probabilities(raw_teacher, masks)
        if not bool(torch.all(torch.isfinite(teacher)).item()):
            raise FloatingPointError("operation teacher is non-finite")
        return teacher.detach(), q_move_teacher.detach(), move_probability.detach(), move_legal

    @staticmethod
    def _operation_teacher_kl(
        operation_log_probs_all: Any,
        teacher_probabilities: Any,
        operation_masks: Any,
    ) -> Any:
        """Return full-state/head KL(Exact-Q operation teacher || policy)."""

        log_probs = torch.as_tensor(operation_log_probs_all, dtype=torch.float32)
        teacher = torch.as_tensor(
            teacher_probabilities, dtype=torch.float32, device=log_probs.device
        )
        masks = torch.as_tensor(
            operation_masks, dtype=torch.bool, device=log_probs.device
        )
        if log_probs.shape != teacher.shape or teacher.shape != masks.shape:
            raise ValueError("operation teacher KL shape mismatch")
        safe_teacher = torch.where(masks, teacher, torch.zeros_like(teacher))
        safe_log_probs = torch.where(
            masks, log_probs, torch.zeros_like(log_probs)
        )
        kl = torch.sum(
            safe_teacher
            * (torch.log(safe_teacher.clamp_min(1.0e-12)) - safe_log_probs),
            dim=-1,
        )
        value = kl.mean()
        if not bool(torch.isfinite(value).item()):
            raise FloatingPointError("operation teacher KL is non-finite")
        return value

    @classmethod
    def _exact_q_listwise_candidate_loss(
        cls,
        candidate_log_probs: Any,
        q_values: Any,
        candidate_valid_mask: Any,
        temperature: float,
    ) -> tuple[Any, Any]:
        """Return KL(teacher(Q) || current candidate policy) and the teacher."""

        log_probs = torch.as_tensor(candidate_log_probs, dtype=torch.float32)
        valid = torch.as_tensor(
            candidate_valid_mask, dtype=torch.bool, device=log_probs.device
        )
        q_tensor = torch.as_tensor(q_values, dtype=torch.float32, device=log_probs.device)
        if log_probs.shape != valid.shape or q_tensor.shape != valid.shape:
            raise ValueError("exact-Q listwise candidate tensor shape mismatch")
        teacher = cls._exact_q_listwise_teacher(q_tensor, valid, temperature).to(
            log_probs.device
        )
        active_heads = valid.any(dim=-1)
        if not bool(torch.any(active_heads).item()):
            return log_probs.sum() * 0.0, teacher
        safe_log_probs = torch.where(
            valid, log_probs, torch.zeros_like(log_probs)
        )
        safe_teacher = torch.where(
            valid, teacher, torch.zeros_like(teacher)
        )
        kl = torch.sum(
            safe_teacher
            * (
                torch.log(safe_teacher.clamp_min(1.0e-12))
                - safe_log_probs
            ),
            dim=-1,
        )
        return kl[active_heads].mean(), teacher

    def _cache_policy_terms(
        self,
        cache_contexts: Any,
        operation_masks: Any,
        operation_actions: Any,
        candidate_features: Any,
        candidate_operation_indices: Any,
        candidate_valid_mask: Any,
    ) -> tuple[Any, Any]:
        """Return node-only trigger log-probs and node-conditional candidate probs.

        The returned candidate probabilities are normalized over all legal
        target configurations for each node after conditioning on the
        aggregate UPDATE event.  No whole-Cache joint probability enters this
        helper.
        """

        if not self._uses_exact_cache_credit():
            raise RuntimeError("Exact Cache cache policy terms requested for a non-Exact Cache policy")
        contexts = torch.as_tensor(
            cache_contexts, dtype=torch.float32, device=self.device
        )
        masks = torch.as_tensor(
            operation_masks, dtype=torch.bool, device=self.device
        )
        actions = torch.as_tensor(
            operation_actions, dtype=torch.long, device=self.device
        )
        features = torch.as_tensor(
            candidate_features, dtype=torch.float32, device=self.device
        )
        operation_indices = torch.as_tensor(
            candidate_operation_indices, dtype=torch.long, device=self.device
        )
        valid = torch.as_tensor(
            candidate_valid_mask, dtype=torch.bool, device=self.device
        )
        if contexts.ndim != 3 or masks.ndim != 3 or actions.ndim != 2:
            raise ValueError("Exact Cache node policy tensor rank mismatch")
        batch_size, node_count, observation_dim = [int(value) for value in contexts.shape]
        trigger_width = (
            2
            if self._uses_binary_cache_trigger()
            else len(AdaptiveJointCacheUAVPPOActorCritic.operation_order)
        )
        if tuple(masks.shape) != (batch_size, node_count, trigger_width):
            raise ValueError("Exact Cache operation mask shape mismatch")
        if tuple(actions.shape) != (batch_size, node_count):
            raise ValueError("Exact Cache operation action shape mismatch")
        if features.ndim != 4 or tuple(features.shape[:2]) != (batch_size, node_count):
            raise ValueError("Exact Cache candidate feature shape mismatch")
        if tuple(operation_indices.shape) != tuple(valid.shape) or tuple(valid.shape) != tuple(features.shape[:3]):
            raise ValueError("Exact Cache candidate tensor shape mismatch")
        flat_contexts = contexts.reshape(batch_size * node_count, observation_dim)
        if self._uses_binary_cache_trigger():
            all_logits, _ = self.policy._trigger_logits(flat_contexts)
        else:
            all_logits, _ = self.policy._operation_logits(flat_contexts)
        all_logits = all_logits.reshape(
            batch_size,
            node_count,
            self.policy.num_cache_heads,
            trigger_width,
        )
        head_indices = torch.arange(node_count, dtype=torch.long, device=self.device)
        logits = all_logits[:, head_indices, head_indices, :]
        logits = logits.masked_fill(~masks, -float("inf"))
        operation_probabilities = torch.softmax(logits, dim=-1)
        p_update = (
            operation_probabilities[..., 1]
            if self._uses_binary_cache_trigger()
            else operation_probabilities[..., 1:].sum(dim=-1)
        )
        selected_update = actions != 0
        selected_probability = torch.where(
            selected_update,
            p_update,
            operation_probabilities[..., 0],
        )
        selected_log_prob = torch.log(selected_probability.clamp_min(1.0e-12))

        candidate_count = int(features.shape[2])
        flat_features = features.reshape(
            batch_size * node_count,
            candidate_count,
            int(features.shape[3]),
        )
        raw_scores, _ = self.policy._candidate_scores(
            flat_contexts,
            flat_features,
            self.policy.cache_feature_dim,
            self.policy.cache_candidate_scorer,
        )
        raw_scores = raw_scores.reshape(batch_size, node_count, candidate_count)
        if self._uses_binary_cache_trigger():
            unified_logits = raw_scores.masked_fill(~valid, -float("inf"))
            unified_probabilities = torch.softmax(unified_logits, dim=-1)
            candidate_probabilities = torch.where(
                valid,
                unified_probabilities,
                torch.zeros_like(unified_probabilities),
            )
        else:
            candidate_probabilities = torch.zeros_like(raw_scores)
            for operation_index in range(1, len(self.policy.operation_order)):
                group_mask = valid & (operation_indices == int(operation_index))
                group_exists = group_mask.any(dim=-1)
                safe_group_mask = group_mask | (~group_exists).unsqueeze(-1)
                group_logits = raw_scores.masked_fill(~safe_group_mask, -float("inf"))
                group_probabilities = torch.softmax(group_logits, dim=-1)
                group_probabilities = torch.where(
                    group_mask,
                    group_probabilities,
                    torch.zeros_like(group_probabilities),
                )
                candidate_probabilities = candidate_probabilities + (
                    operation_probabilities[..., operation_index]
                    / p_update.clamp_min(1.0e-12)
                ).unsqueeze(-1) * group_probabilities
        return selected_log_prob, candidate_probabilities

    def _conditional_candidate_probabilities(
        self,
        contexts: Any,
        candidate_features: Any,
        candidate_valid_mask: Any,
        feature_dim: int,
        scorer: Any,
    ) -> Any:
        """Forward a frozen candidate batch through one target scorer."""

        contexts_t = torch.as_tensor(contexts, dtype=torch.float32, device=self.device)
        features_t = torch.as_tensor(
            candidate_features, dtype=torch.float32, device=self.device
        )
        valid_t = torch.as_tensor(
            candidate_valid_mask, dtype=torch.bool, device=self.device
        )
        if contexts_t.ndim != 3 or features_t.ndim != 4 or valid_t.ndim != 3:
            raise ValueError("conditional candidate probability tensor rank mismatch")
        batch_size, head_count, observation_dim = [int(value) for value in contexts_t.shape]
        candidate_count = int(features_t.shape[2])
        if tuple(features_t.shape[:3]) != (batch_size, head_count, candidate_count):
            raise ValueError("conditional candidate feature batch shape mismatch")
        if tuple(valid_t.shape) != (batch_size, head_count, candidate_count):
            raise ValueError("conditional candidate mask batch shape mismatch")
        flat_scores, _ = self.policy._candidate_scores(
            contexts_t.reshape(batch_size * head_count, observation_dim),
            features_t.reshape(batch_size * head_count, candidate_count, int(feature_dim)),
            int(feature_dim),
            scorer,
        )
        scores = flat_scores.reshape(batch_size, head_count, candidate_count)
        active = valid_t.any(dim=-1)
        safe_valid = valid_t | ~active.unsqueeze(-1)
        logits = scores.masked_fill(~safe_valid, -float("inf"))
        probabilities = torch.softmax(logits, dim=-1)
        return torch.where(
            valid_t & active.unsqueeze(-1),
            probabilities,
            torch.zeros_like(probabilities),
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
        cache_operation_actions = None
        cache_operation_masks = None
        cache_operation_old_log_probs = None
        if self.context.configuration_control_mode in {
            "policy_primitive_cache_operation",
            "policy_primitive_joint_cache_uav_operation",
            "adaptive_joint_cache_uav_reconfiguration",
        }:
            if (
                rollout.cache_operation_actions is None
                or rollout.cache_operation_masks is None
            ):
                raise ValueError("Cache-operation rollout is missing operation actions/masks")
            cache_operation_actions = torch.as_tensor(
                rollout.cache_operation_actions,
                dtype=torch.long,
                device=self.device,
            )
            cache_operation_masks = torch.as_tensor(
                rollout.cache_operation_masks,
                dtype=torch.bool,
                device=self.device,
            )
            if rollout.cache_operation_log_probs is not None:
                cache_operation_old_log_probs = torch.as_tensor(
                    rollout.cache_operation_log_probs,
                    dtype=torch.float32,
                    device=self.device,
                )
        uav_operation_actions = None
        uav_operation_masks = None
        uav_operation_old_log_probs = None
        if self.context.configuration_control_mode in {
            "policy_primitive_joint_cache_uav_operation",
            "adaptive_joint_cache_uav_reconfiguration",
        }:
            if rollout.uav_operation_actions is None or rollout.uav_operation_masks is None:
                raise ValueError("proposed rollout is missing UAV operation actions/masks")
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
        exact_uav_credit_enabled = self._uses_exact_uav_operation_credit()
        branch_diagnostics_enabled = (
            self.policy.architecture_id in JOINT_CACHE_UAV_POLICY_SCHEMAS
        )
        branchwise_policy_enabled = exact_uav_credit_enabled
        branch_kl_instrumentation_enabled = exact_uav_credit_enabled
        branch_specific_kl_gate_enabled = (
            self.policy.architecture_id in BRANCH_SPECIFIC_KL_POLICY_SCHEMAS
            and self.kl_gate_mode == KL_GATE_MODE_BRANCH_SPECIFIC
        )
        cache_minibatch_kl_control_enabled = False
        cache_candidate_old_log_probs = None
        uav_candidate_old_log_probs = None
        if branch_diagnostics_enabled:
            if (
                cache_operation_old_log_probs is None
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
            if tuple(cache_operation_old_log_probs.shape) != tuple(
                cache_operation_actions.shape
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
        uav_operation_credit: dict[str, Any] = {}
        uav_operation_diagnostics: dict[str, Any] | None = None
        uav_operation_diagnostic_arrays: dict[str, np.ndarray] | None = None
        if exact_uav_credit_enabled:
            if not branch_diagnostics_enabled:
                raise RuntimeError("proposed operation credit requires joint branch diagnostics")
            uav_operation_credit = self._compute_exact_uav_operation_credit(rollout)
            (
                uav_operation_diagnostics,
                uav_operation_diagnostic_arrays,
            ) = compute_uav_operation_diagnostics(
                rollout,
                uav_operation_credit,
            )
        cache_credit: dict[str, Any] = {}
        cache_proposal_diagnostics: dict[str, Any] | None = None
        if cache_credit:
            cache_proposal_diagnostics = _proposal_source_diagnostics(
                cache_credit["candidate_q_values"],
                cache_credit["candidate_valid_mask"],
                cache_credit["candidate_proposal_sources"],
                cache_credit["candidate_hamming_distances"],
            )
            cache_proposal_diagnostics.update(
                {
                    "cache_candidate_q_source": str(
                        cache_credit.get("candidate_value_source", "")
                    ),
                    "cache_exact_candidate_count": int(
                        cache_credit.get("exact_candidate_count", 0)
                    ),
                    "cache_surrogate_candidate_count": int(
                        cache_credit.get("action_q_candidate_count", 0)
                    ),
                }
            )
        uav_proposal_diagnostics: dict[str, Any] | None = None
        if uav_operation_credit:
            uav_proposal_diagnostics = _proposal_source_diagnostics(
                uav_operation_credit["candidate_q_values"],
                uav_operation_credit["candidate_valid_mask"],
                uav_operation_credit["candidate_proposal_sources"],
                uav_operation_credit["candidate_move_distances"],
            )
        cache_kl_control_previous: float | None = None
        if cache_minibatch_kl_control_enabled:
            if (
                cache_operation_old_log_probs is None
                or cache_candidate_old_log_probs is None
                or cache_operation_masks is None
                or cache_operation_actions is None
            ):
                raise RuntimeError(
                    "minibatch Cache KL control is missing Cache old action tensors"
                )
            cache_kl_control_previous = self._evaluate_cache_branch_approx_kl_full_rollout(
                observations=observations,
                cache_contexts=cache_contexts,
                operation_masks=cache_operation_masks,
                operation_actions=cache_operation_actions,
                candidate_features=cache_candidate_features,
                candidate_masks=cache_candidate_masks,
                selected_indices=cache_selected_indices,
                old_operation_log_probs=cache_operation_old_log_probs,
                old_candidate_log_probs=cache_candidate_old_log_probs,
            )
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        uav_operation_advantages_scaled_t = None
        if exact_uav_credit_enabled:
            uav_operation_advantages_scaled_t = torch.as_tensor(
                uav_operation_credit["scaled_advantages"],
                dtype=torch.float32,
                device=self.device,
            )
            if tuple(uav_operation_advantages_scaled_t.shape) != tuple(
                uav_operation_actions.shape
            ):
                raise RuntimeError("proposed operation advantage shape mismatch")
        conditional_candidate_q_values_t = None
        conditional_candidate_old_probabilities_t = None
        conditional_candidate_advantages_t = None
        conditional_candidate_valid_mask_t = None
        conditional_candidate_weighted_mean_residual_t = None
        uav_operation_old_probabilities_t = None
        uav_operation_teacher_probabilities_t = None
        uav_operation_teacher_move_probabilities_t = None
        uav_operation_q_move_teacher_t = None
        uav_operation_move_legal_t = None
        if self._uses_exact_uav_candidate_credit():
            for key in (
                "candidate_q_values",
                "candidate_probabilities",
                "candidate_valid_mask",
                "candidate_q_move",
            ):
                if key not in uav_operation_credit:
                    raise RuntimeError(f"conditional candidate operation credit is missing {key}")
            conditional_candidate_q_values_t = torch.as_tensor(
                uav_operation_credit["candidate_q_values"],
                dtype=torch.float32,
                device=self.device,
            )
            conditional_candidate_old_probabilities_t = torch.as_tensor(
                uav_operation_credit["candidate_probabilities"],
                dtype=torch.float32,
                device=self.device,
            )
            conditional_candidate_valid_mask_t = torch.as_tensor(
                uav_operation_credit["candidate_valid_mask"],
                dtype=torch.bool,
                device=self.device,
            )
            conditional_candidate_advantages_t = torch.where(
                conditional_candidate_valid_mask_t,
                conditional_candidate_q_values_t
                - torch.as_tensor(
                    uav_operation_credit["candidate_q_move"],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(-1),
                torch.zeros_like(conditional_candidate_q_values_t),
            )
            conditional_candidate_weighted_mean_residual_t = torch.as_tensor(
                uav_operation_credit["candidate_weighted_mean_residual"],
                dtype=torch.float32,
                device=self.device,
            )
            expected_candidate_shape = (
                sample_count,
                int(uav_operation_actions.shape[1]),
                int(uav_candidate_features.shape[2]),
            )
            for name, tensor in (
                ("candidate_q_values", conditional_candidate_q_values_t),
                ("candidate_probabilities", conditional_candidate_old_probabilities_t),
                ("candidate_valid_mask", conditional_candidate_valid_mask_t),
                ("candidate_advantages", conditional_candidate_advantages_t),
            ):
                if tuple(tensor.shape) != expected_candidate_shape:
                    raise RuntimeError(
                        f"conditional candidate {name} shape mismatch: "
                        f"expected={expected_candidate_shape}, got={tuple(tensor.shape)}"
                    )
            if tuple(conditional_candidate_weighted_mean_residual_t.shape) != tuple(
                uav_operation_actions.shape
            ):
                raise RuntimeError("conditional candidate candidate residual shape mismatch")
            if tuple(uav_operation_credit["candidate_q_move"].shape) != tuple(
                uav_operation_actions.shape
            ):
                raise RuntimeError("conditional candidate candidate q_move shape mismatch")
            if bool(
                torch.any(
                    conditional_candidate_old_probabilities_t[~conditional_candidate_valid_mask_t] != 0.0
                ).item()
            ) or bool(
                torch.any(
                    conditional_candidate_old_probabilities_t[conditional_candidate_valid_mask_t] <= 0.0
                ).item()
            ):
                raise FloatingPointError("conditional candidate candidate probabilities are invalid")
            if not bool(
                torch.all(torch.isfinite(conditional_candidate_advantages_t)).item()
            ):
                raise FloatingPointError("conditional candidate candidate advantages are non-finite")
        if exact_uav_credit_enabled:
            if not self._uses_exact_uav_candidate_credit():
                raise RuntimeError(
                    "proposed operation teacher requires exact candidate Q tensors"
                )
            if uav_operation_masks is None:
                raise RuntimeError("operation teacher is missing UAV operation masks")
            raw_operation_probabilities = np.stack(
                [
                    np.asarray(
                        transition.uav_operation_probabilities,
                        dtype=np.float32,
                    )
                    for transition in rollout.transitions
                ],
                axis=0,
            )
            expected_operation_probability_shape = tuple(uav_operation_masks.shape)
            if tuple(raw_operation_probabilities.shape) != expected_operation_probability_shape:
                raise RuntimeError(
                    "rollout UAV operation probability shape mismatch: "
                    f"expected={expected_operation_probability_shape}, "
                    f"got={tuple(raw_operation_probabilities.shape)}"
                )
            uav_operation_old_probabilities_t = self._masked_operation_probabilities(
                torch.as_tensor(
                    raw_operation_probabilities,
                    dtype=torch.float32,
                    device=self.device,
                ),
                uav_operation_masks,
            )
            assert conditional_candidate_q_values_t is not None
            assert conditional_candidate_valid_mask_t is not None
            q_stay_t = torch.as_tensor(
                uav_operation_credit["q_stay"],
                dtype=torch.float32,
                device=self.device,
            )
            (
                uav_operation_teacher_probabilities_t,
                uav_operation_q_move_teacher_t,
                uav_operation_teacher_move_probabilities_t,
                uav_operation_move_legal_t,
            ) = self._exact_q_operation_teacher(
                q_stay_t,
                conditional_candidate_q_values_t,
                conditional_candidate_valid_mask_t,
                uav_operation_masks,
                self.exact_q_listwise_temperature,
                hyper.operation_advantage_reference_scale,
            )
        cache_trigger_advantages_t = None
        cache_trigger_old_probabilities_t = None
        cache_candidate_features_t = None
        cache_candidate_operation_indices_t = None
        cache_candidate_q_values_t = None
        cache_candidate_old_probabilities_t = None
        cache_candidate_valid_mask_t = None
        cache_candidate_advantages_t = None
        if self._uses_exact_cache_credit():
            required_cache_keys = (
                "trigger_advantages",
                "trigger_old_probabilities",
                "candidate_features",
                "candidate_operation_indices",
                "candidate_q_values",
                "candidate_probabilities",
                "candidate_valid_mask",
                "candidate_q_update",
            )
            for key in required_cache_keys:
                if key not in cache_credit:
                    raise RuntimeError(f"Exact Cache credit is missing {key}")
            cache_trigger_advantages_t = torch.as_tensor(
                cache_credit["trigger_advantages"],
                dtype=torch.float32,
                device=self.device,
            )
            cache_trigger_old_probabilities_t = torch.as_tensor(
                cache_credit["trigger_old_probabilities"],
                dtype=torch.float32,
                device=self.device,
            )
            cache_candidate_features_t = torch.as_tensor(
                cache_credit["candidate_features"],
                dtype=torch.float32,
                device=self.device,
            )
            cache_candidate_operation_indices_t = torch.as_tensor(
                cache_credit["candidate_operation_indices"],
                dtype=torch.long,
                device=self.device,
            )
            cache_candidate_q_values_t = torch.as_tensor(
                cache_credit["candidate_q_values"],
                dtype=torch.float32,
                device=self.device,
            )
            cache_candidate_old_probabilities_t = torch.as_tensor(
                cache_credit["candidate_probabilities"],
                dtype=torch.float32,
                device=self.device,
            )
            cache_candidate_valid_mask_t = torch.as_tensor(
                cache_credit["candidate_valid_mask"],
                dtype=torch.bool,
                device=self.device,
            )
            cache_candidate_advantages_t = torch.where(
                cache_candidate_valid_mask_t,
                cache_candidate_q_values_t
                - torch.as_tensor(
                    cache_credit["candidate_q_update"],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(-1),
                torch.zeros_like(cache_candidate_q_values_t),
            )
            expected_cache_trigger_shape = tuple(cache_operation_actions.shape)
            if tuple(cache_trigger_advantages_t.shape) != expected_cache_trigger_shape:
                raise RuntimeError("Exact Cache trigger advantage shape mismatch")
            if tuple(cache_trigger_old_probabilities_t.shape) != (
                sample_count,
                int(cache_operation_actions.shape[1]),
                2,
            ):
                raise RuntimeError("Exact Cache trigger probability shape mismatch")
            expected_cache_candidate_shape = (
                sample_count,
                int(cache_operation_actions.shape[1]),
                int(cache_candidate_features_t.shape[2]),
            )
            for name, tensor in (
                ("candidate features", cache_candidate_features_t),
                ("candidate operation indices", cache_candidate_operation_indices_t),
                ("candidate q values", cache_candidate_q_values_t),
                ("candidate old probabilities", cache_candidate_old_probabilities_t),
                ("candidate valid mask", cache_candidate_valid_mask_t),
                ("candidate advantages", cache_candidate_advantages_t),
            ):
                if tuple(tensor.shape[:3]) != expected_cache_candidate_shape:
                    raise RuntimeError(f"Exact Cache {name} shape mismatch")
            if not bool(torch.all(torch.isfinite(cache_trigger_advantages_t)).item()):
                raise FloatingPointError("Exact Cache trigger advantages are non-finite")
            if not bool(torch.all(torch.isfinite(cache_candidate_advantages_t)).item()):
                raise FloatingPointError("Exact Cache candidate advantages are non-finite")
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

        cache_target_policy_probabilities_before = None
        uav_target_policy_probabilities_before = None
        cache_exact_q_quality: dict[str, float] | None = None
        uav_exact_q_quality: dict[str, float] | None = None
        with torch.no_grad():
            if self._uses_exact_cache_credit():
                if (
                    cache_candidate_features_t is None
                    or cache_candidate_valid_mask_t is None
                    or cache_candidate_q_values_t is None
                ):
                    raise RuntimeError("Cache exact-Q tensors are unavailable for gate diagnostics")
                cache_target_policy_probabilities_before = (
                    self._conditional_candidate_probabilities(
                        cache_contexts,
                        cache_candidate_features_t,
                        cache_candidate_valid_mask_t,
                        int(self.policy.cache_feature_dim),
                        self.policy.cache_candidate_scorer,
                    )
                    .detach()
                    .clone()
                )
                cache_exact_q_quality = _exact_q_policy_quality_metrics(
                    cache_target_policy_probabilities_before.detach().cpu().numpy(),
                    cache_target_policy_probabilities_before.detach().cpu().numpy(),
                    cache_candidate_q_values_t.detach().cpu().numpy(),
                    cache_candidate_valid_mask_t.detach().cpu().numpy(),
                )
            if self._uses_exact_uav_candidate_credit():
                if (
                    conditional_candidate_valid_mask_t is None
                    or conditional_candidate_q_values_t is None
                ):
                    raise RuntimeError("UAV exact-Q tensors are unavailable for gate diagnostics")
                uav_target_policy_probabilities_before = (
                    self._conditional_candidate_probabilities(
                        uav_contexts,
                        uav_candidate_features,
                        conditional_candidate_valid_mask_t,
                        int(self.policy.uav_feature_dim),
                        self.policy.uav_candidate_scorer,
                    )
                    .detach()
                    .clone()
                )
                uav_exact_q_quality = _exact_q_policy_quality_metrics(
                    uav_target_policy_probabilities_before.detach().cpu().numpy(),
                    uav_target_policy_probabilities_before.detach().cpu().numpy(),
                    conditional_candidate_q_values_t.detach().cpu().numpy(),
                    conditional_candidate_valid_mask_t.detach().cpu().numpy(),
                )

        before = [parameter.detach().clone() for parameter in self.policy.parameters()]
        cache_target_scorer_before = {
            str(name): parameter.detach().clone()
            for name, parameter in self.policy.cache_candidate_scorer.named_parameters()
        }
        uav_target_scorer_before = {
            str(name): parameter.detach().clone()
            for name, parameter in self.policy.uav_candidate_scorer.named_parameters()
        }
        uav_operation_head_before: dict[str, Any] | None = None
        uav_encoder_before: dict[str, Any] | None = None
        if exact_uav_credit_enabled:
            uav_operation_head_before = {
                str(name): parameter.detach().clone()
                for name, parameter in self.policy.uav_operation_head.named_parameters()
            }
            uav_encoder_before = {
                str(name): parameter.detach().clone()
                for name, parameter in self.policy.uav_encoder.named_parameters()
            }
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
        uav_operation_full_rollout_kl_values: list[float] = []
        shared_actor_gradient_norm_values: list[float] = []
        kl_minibatch_diagnostics: list[dict[str, Any]] = []
        kl_epoch_diagnostics: list[dict[str, Any]] = []
        cache_policy_loss_values: list[float] = []
        cache_trigger_policy_loss_values: list[float] = []
        cache_candidate_policy_loss_values: list[float] = []
        uav_policy_loss_values: list[float] = []
        uav_operation_policy_loss_values: list[float] = []
        actor_policy_loss_values: list[float] = []
        candidate_policy_loss_values: list[float] = []
        candidate_advantage_mean_abs_values: list[float] = []
        candidate_advantage_weighted_mean_residual_values: list[float] = []
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
        uav_operation_steps_executed = 0
        uav_operation_steps_suppressed = 0
        uav_candidate_full_steps = 0
        uav_candidate_only_steps_after_operation_stop = 0
        uav_operation_full_rollout_kl_current: float | None = None
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
            update_epoch_uav_operation_full_rollout_kls: list[float] = []
            epoch_diagnostic_start = len(kl_minibatch_diagnostics)
            for start in range(0, sample_count, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                cache_active_this_minibatch = bool(cache_active)
                uav_active_this_minibatch = bool(uav_active)
                uav_candidate_active_this_minibatch = bool(
                    (
                        exact_uav_credit_enabled
                        and self._uses_exact_uav_candidate_credit()
                    )
                    or uav_active_this_minibatch
                )
                if exact_uav_credit_enabled:
                    if uav_active_this_minibatch:
                        uav_operation_steps_executed += 1
                        uav_candidate_full_steps += 1
                    else:
                        uav_operation_steps_suppressed += 1
                        uav_candidate_only_steps_after_operation_stop += 1
                    # The candidate scorer remains trainable for every
                    # proposed minibatch.  Once the operation KL gate closes,
                    # only the binary operation head and its UAV encoder are
                    # frozen; the candidate scorer still receives Exact-Q
                    # listwise gradients.
                    for parameter in self.policy.uav_operation_head.parameters():
                        parameter.requires_grad_(uav_active_this_minibatch)
                    for parameter in self.policy.uav_encoder.parameters():
                        parameter.requires_grad_(uav_active_this_minibatch)
                uav_approx_kl_after_step: float | None = None
                uav_operation_full_rollout_kl_after_step: float | None = None
                uav_stop_triggered_after_step = False
                if self.context.configuration_control_mode in {
                    "policy_primitive_joint_cache_uav_operation",
                    "adaptive_joint_cache_uav_reconfiguration",
                }:
                    if not isinstance(
                        self.policy,
                        AdaptiveJointCacheUAVPPOActorCritic,
                    ):
                        raise TypeError("proposed update requires the proposed joint policy")
                    assert cache_operation_actions is not None
                    assert cache_operation_masks is not None
                    assert uav_operation_actions is not None
                    assert uav_operation_masks is not None
                    evaluation = self.policy.evaluate_joint_cache_uav_operation(
                        observations[indices],
                        cache_contexts[indices],
                        cache_operation_masks[indices],
                        cache_operation_actions[indices],
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
                elif self.context.configuration_control_mode == "policy_primitive_cache_operation":
                    if not isinstance(
                        self.policy,
                        AdaptiveCacheOnlyActorCritic,
                    ):
                        raise TypeError("cache-only update requires the cache-only policy")
                    assert cache_operation_actions is not None
                    assert cache_operation_masks is not None
                    evaluation = self.policy.evaluate_cache_operation_primitive(
                        observations[indices],
                        cache_contexts[indices],
                        cache_operation_masks[indices],
                        cache_operation_actions[indices],
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
                        or cache_operation_old_log_probs is None
                        or uav_operation_old_log_probs is None
                        or (
                            exact_uav_credit_enabled
                            and (
                                evaluation.uav_operation_log_probs is None
                                or evaluation.uav_operation_log_probs_all is None
                                or evaluation.uav_candidate_log_probs is None
                                or (
                                    self._uses_exact_uav_candidate_credit()
                                    and evaluation.uav_candidate_log_probs_all is None
                                )
                            )
                        )
                    ):
                        raise RuntimeError(
                            "Joint Cache/UAV evaluation did not expose branch diagnostics"
                        )
                    cache_old_log_prob = (
                        cache_operation_old_log_probs[indices].sum(dim=-1)
                        + cache_candidate_old_log_probs[indices].sum(dim=-1)
                    )
                    if exact_uav_credit_enabled:
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
                        if self._uses_exact_cache_credit():
                            assert (
                                cache_trigger_advantages_t is not None
                                and cache_trigger_old_probabilities_t is not None
                                and cache_candidate_features_t is not None
                                and cache_candidate_operation_indices_t is not None
                                and cache_candidate_old_probabilities_t is not None
                                and cache_candidate_valid_mask_t is not None
                                and cache_candidate_advantages_t is not None
                                and cache_operation_actions is not None
                                and cache_operation_masks is not None
                            )
                            cache_trigger_new_log_probs, cache_candidate_new_probabilities = (
                                self._cache_policy_terms(
                                    cache_contexts[indices],
                                    cache_operation_masks[indices],
                                    cache_operation_actions[indices],
                                    cache_candidate_features_t[indices],
                                    cache_candidate_operation_indices_t[indices],
                                    cache_candidate_valid_mask_t[indices],
                                )
                            )
                            selected_update = cache_operation_actions[indices] != 0
                            old_selected_probability = torch.where(
                                selected_update,
                                cache_trigger_old_probabilities_t[indices, ..., 1],
                                cache_trigger_old_probabilities_t[indices, ..., 0],
                            )
                            trigger_ratio = torch.exp(
                                cache_trigger_new_log_probs
                                - torch.log(old_selected_probability.clamp_min(1.0e-12))
                            )
                            trigger_advantage = cache_trigger_advantages_t[indices]
                            trigger_unclipped = trigger_ratio * trigger_advantage
                            trigger_clipped = torch.clamp(
                                trigger_ratio,
                                1.0 - hyper.clip_epsilon,
                                1.0 + hyper.clip_epsilon,
                            ) * trigger_advantage
                            trigger_surrogate = torch.minimum(
                                trigger_unclipped,
                                trigger_clipped,
                            )
                            trigger_policy_loss = -trigger_surrogate.mean()
                            cache_trigger_policy_loss_values.append(
                                float(trigger_policy_loss.detach().cpu())
                            )

                            candidate_valid = cache_candidate_valid_mask_t[indices]
                            candidate_ratio = torch.where(
                                candidate_valid,
                                cache_candidate_new_probabilities
                                / cache_candidate_old_probabilities_t[indices].clamp_min(
                                    1.0e-12
                                ),
                                torch.ones_like(cache_candidate_new_probabilities),
                            )
                            candidate_policy_loss, _ = self._exact_q_listwise_candidate_loss(
                                torch.log(
                                    cache_candidate_new_probabilities.clamp_min(
                                        1.0e-12
                                    )
                                ),
                                cache_candidate_q_values_t[indices],
                                candidate_valid,
                                self.exact_q_listwise_temperature,
                            )
                            cache_candidate_policy_loss_values.append(
                                float(candidate_policy_loss.detach().cpu())
                            )
                            cache_branch_policy_loss = 0.5 * (
                                trigger_policy_loss + candidate_policy_loss
                            )
                            branch_metric_tensors += (
                                trigger_ratio.detach(),
                                candidate_ratio.detach(),
                                trigger_advantage.detach(),
                                candidate_policy_loss.detach(),
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
                        if exact_uav_credit_enabled:
                            assert (
                                evaluation.uav_operation_log_probs is not None
                                and evaluation.uav_operation_log_probs_all is not None
                                and evaluation.uav_candidate_log_probs is not None
                                and uav_operation_actions is not None
                                and uav_operation_old_log_probs is not None
                                and uav_candidate_old_log_probs is not None
                                and uav_operation_teacher_probabilities_t is not None
                                and uav_operation_q_move_teacher_t is not None
                                and conditional_candidate_q_values_t is not None
                                and conditional_candidate_valid_mask_t is not None
                            )
                            operation_ratio = torch.exp(
                                evaluation.uav_operation_log_probs
                                - uav_operation_old_log_probs[indices]
                            )
                            operation_advantage = (
                                uav_operation_q_move_teacher_t[indices]
                                - torch.as_tensor(
                                    uav_operation_credit["q_stay"],
                                    dtype=torch.float32,
                                    device=self.device,
                                )[indices]
                            )
                            operation_teacher_policy_loss = self._operation_teacher_kl(
                                evaluation.uav_operation_log_probs_all,
                                uav_operation_teacher_probabilities_t[indices],
                                uav_operation_masks[indices],
                            )
                            operation_policy_loss = operation_teacher_policy_loss
                            if uav_active_this_minibatch:
                                uav_operation_policy_loss_values.append(
                                    float(operation_policy_loss.detach().cpu())
                                )
                            candidate_ratio = torch.exp(
                                evaluation.uav_candidate_log_probs
                                - uav_candidate_old_log_probs[indices]
                            )
                            if self._uses_exact_uav_candidate_credit():
                                assert (
                                    evaluation.uav_candidate_log_probs_all is not None
                                    and conditional_candidate_old_probabilities_t is not None
                                    and conditional_candidate_advantages_t is not None
                                    and conditional_candidate_valid_mask_t is not None
                                )
                                candidate_advantage = conditional_candidate_advantages_t[indices]
                                candidate_valid = conditional_candidate_valid_mask_t[indices]
                                candidate_policy_loss, _ = self._exact_q_listwise_candidate_loss(
                                    evaluation.uav_candidate_log_probs_all,
                                    conditional_candidate_q_values_t[indices],
                                    candidate_valid,
                                    self.exact_q_listwise_temperature,
                                )
                                candidate_active = candidate_valid
                                candidate_residual = (
                                    conditional_candidate_old_probabilities_t[indices]
                                    * candidate_advantage
                                ).sum(dim=-1)
                                active_heads = candidate_valid.any(dim=-1)
                                if bool(torch.any(active_heads).item()):
                                    candidate_policy_loss_values.append(
                                        float(candidate_policy_loss.detach().cpu())
                                    )
                                    candidate_advantage_mean_abs_values.append(
                                        float(
                                            torch.abs(candidate_advantage[active_heads])
                                            .masked_select(candidate_valid[active_heads])
                                            .mean()
                                            .detach()
                                            .cpu()
                                        )
                                    )
                                    candidate_advantage_weighted_mean_residual_values.append(
                                        float(
                                            torch.abs(candidate_residual[active_heads])
                                            .mean()
                                            .detach()
                                            .cpu()
                                        )
                                    )
                            else:
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
                            # The proposed UAV branch is the mean of the
                            # full-information operation teacher KL and the
                            # unchanged candidate Exact-Q listwise KL.  After
                            # the operation KL stop, candidate-only updates
                            # retain the candidate half while the operation
                            # head/encoder are frozen.
                            if uav_active_this_minibatch:
                                uav_branch_policy_loss = 0.5 * (
                                    operation_policy_loss + candidate_policy_loss
                                )
                            else:
                                uav_branch_policy_loss = 0.5 * candidate_policy_loss
                            branch_metric_tensors += (
                                operation_ratio.detach(),
                                candidate_ratio.detach(),
                                operation_advantage.detach(),
                                operation_teacher_policy_loss.detach(),
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
                                if uav_candidate_active_this_minibatch:
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
                                    uav_active=uav_active_this_minibatch,
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
                    if branch_kl_instrumentation_enabled:
                        update_epoch_cache_approx_kls.append(
                            cache_minibatch_approx_kl
                        )
                        update_epoch_uav_approx_kls.append(
                            uav_minibatch_approx_kl
                        )
                    cache_clip_metric = torch.mean(
                        (
                            torch.abs(cache_branch_ratio.detach() - 1.0)
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
                        if exact_uav_credit_enabled:
                            # The proposed candidate scorer is intentionally
                            # still active after the operation stop.  Clear
                            # only operation-head/encoder gradients, while
                            # preserving candidate-scorer gradients.
                            _clear_inactive_branch_gradients(
                                branch_gradient_parameter_groups,
                                cache_active=cache_active_this_minibatch,
                                uav_active=True,
                            )
                            if not uav_active_this_minibatch:
                                for parameter in self.policy.uav_operation_head.parameters():
                                    parameter.grad = None
                                for parameter in self.policy.uav_encoder.parameters():
                                    parameter.grad = None
                        else:
                            _clear_inactive_branch_gradients(
                                branch_gradient_parameter_groups,
                                cache_active=cache_active_this_minibatch,
                                uav_active=uav_active_this_minibatch,
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
                            branch_kl_instrumentation_enabled
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
                        if branch_kl_instrumentation_enabled:
                            cache_encoder_gradient_norm_values.append(
                                cache_encoder_gradient_norm_value
                            )
                            uav_encoder_gradient_norm_values.append(
                                uav_encoder_gradient_norm_value
                            )
                    actor_update_active = bool(
                        cache_active_this_minibatch
                        or uav_candidate_active_this_minibatch
                    )
                    if not actor_update_active:
                        if any(parameter.grad is not None for parameter in actor_parameters):
                            raise RuntimeError(
                                "PPO v5 both-inactive actor backward left gradients."
                            )
                        actor_gradient_norm_value = 0.0
                    else:
                        actor_gradient_norm = torch.nn.utils.clip_grad_norm_(
                            actor_parameters, hyper.max_grad_norm
                        )
                        actor_gradient_norm_value = float(
                            actor_gradient_norm.detach().cpu()
                        )
                        if not np.isfinite(actor_gradient_norm_value):
                            raise FloatingPointError(
                                "PPO v5 actor gradient norm 非 finite。"
                            )
                        self.actor_optimizer.step()
                        self.actor_optimizer_steps += 1
                        self.optimizer_steps = self.actor_optimizer_steps
                        if (
                            branch_specific_kl_gate_enabled
                            and exact_uav_credit_enabled
                            and uav_active_this_minibatch
                        ):
                            if (
                                cache_operation_masks is None
                                or cache_operation_actions is None
                                or uav_operation_masks is None
                                or uav_operation_actions is None
                                or uav_operation_old_probabilities_t is None
                            ):
                                raise RuntimeError(
                                    "minibatch UAV KL control tensors are incomplete"
                                )
                            uav_operation_full_rollout_kl_after_step = (
                                self._evaluate_uav_branch_approx_kl_full_rollout(
                                    observations=observations,
                                    cache_contexts=cache_contexts,
                                    cache_operation_masks=cache_operation_masks,
                                    cache_operation_actions=cache_operation_actions,
                                    cache_candidate_features=cache_candidate_features,
                                    cache_candidate_masks=cache_candidate_masks,
                                    cache_selected_indices=cache_selected_indices,
                                    uav_contexts=uav_contexts,
                                    uav_operation_masks=uav_operation_masks,
                                    uav_operation_actions=uav_operation_actions,
                                    uav_candidate_features=uav_candidate_features,
                                    uav_candidate_masks=uav_candidate_masks,
                                    uav_selected_indices=uav_selected_indices,
                                    old_uav_operation_probabilities=(
                                        uav_operation_old_probabilities_t
                                    ),
                                )
                            )
                            # Keep the legacy minibatch field populated while
                            # exposing the operation-only metric explicitly.
                            uav_approx_kl_after_step = (
                                uav_operation_full_rollout_kl_after_step
                            )
                            uav_operation_full_rollout_kl_current = (
                                uav_operation_full_rollout_kl_after_step
                            )
                            if (
                                hyper.target_kl is not None
                                and uav_operation_full_rollout_kl_after_step
                                > hyper.target_kl
                            ):
                                uav_stop_triggered_after_step = True
                                uav_active = False
                                if uav_stop_epoch is None:
                                    uav_stop_epoch = int(update_epoch + 1)
                                    uav_suppressed_update_epochs = int(
                                        max(hyper.update_epochs - update_epoch - 1, 0)
                                    )

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
                if branchwise_policy_enabled and bool(uav_active_this_minibatch):
                    uav_minibatches_executed += 1
                if branchwise_policy_enabled and bool(cache_active_this_minibatch):
                    cache_minibatches_executed += 1
                cache_approx_kl_after_step: float | None = None
                cache_stop_triggered_after_step = False
                if cache_minibatch_kl_control_enabled:
                    if cache_active_this_minibatch:
                        if (
                            cache_operation_masks is None
                            or cache_operation_actions is None
                            or cache_operation_old_log_probs is None
                            or cache_candidate_old_log_probs is None
                        ):
                            raise RuntimeError(
                                "minibatch Cache KL control tensors are incomplete"
                            )
                        cache_approx_kl_after_step = (
                            self._evaluate_cache_branch_approx_kl_full_rollout(
                                observations=observations,
                                cache_contexts=cache_contexts,
                                operation_masks=cache_operation_masks,
                                operation_actions=cache_operation_actions,
                                candidate_features=cache_candidate_features,
                                candidate_masks=cache_candidate_masks,
                                selected_indices=cache_selected_indices,
                                old_operation_log_probs=cache_operation_old_log_probs,
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
                if exact_uav_credit_enabled:
                    if uav_operation_full_rollout_kl_current is None:
                        # A branch-specific proposed update normally computes
                        # this after its first operation step.  Keep the
                        # per-minibatch/epoch schema finite even when a caller
                        # disables the branch gate entirely.
                        uav_operation_full_rollout_kl_current = 0.0
                    update_epoch_uav_operation_full_rollout_kls.append(
                        float(uav_operation_full_rollout_kl_current)
                    )
                    uav_operation_full_rollout_kl_values.append(
                        float(uav_operation_full_rollout_kl_current)
                    )
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
                if branch_kl_instrumentation_enabled:
                    if not (
                        update_epoch_cache_approx_kls
                        and update_epoch_uav_approx_kls
                    ):
                        raise RuntimeError(
                            "branch KL diagnostics are not aligned with minibatches"
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
                                    uav_operation_full_rollout_kl_current
                                    > target_kl_value
                                )
                            ),
                            "uav_branch_over_target": (
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
                            "uav_active": bool(uav_active_this_minibatch),
                            "uav_update_executed": bool(
                                uav_active_this_minibatch
                            ),
                            "uav_candidate_update_executed": bool(
                                uav_candidate_active_this_minibatch
                            ),
                            "uav_operation_full_rollout_kl": float(
                                uav_operation_full_rollout_kl_current
                            ),
                            "uav_operation_over_target": (
                                None
                                if target_kl_value is None
                                else bool(
                                    uav_operation_full_rollout_kl_current
                                    > target_kl_value
                                )
                            ),
                            "uav_approx_kl_after_step": (
                                None
                                if uav_approx_kl_after_step is None
                                else float(uav_approx_kl_after_step)
                            ),
                            "uav_stop_triggered_after_step": bool(
                                uav_stop_triggered_after_step
                            ),
                            "uav_active_next_minibatch": bool(uav_active),
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
                # completed.  The shared proposed loop therefore stops both actor
                # and critic updates for the remaining update epochs.
                kl_early_stop_triggered = True
                joint_gate_triggered = True
            if branch_kl_instrumentation_enabled:
                if not (
                    update_epoch_cache_approx_kls
                    and update_epoch_uav_approx_kls
                    and len(update_epoch_cache_approx_kls)
                    == len(update_epoch_uav_approx_kls)
                    == len(update_epoch_approximate_kls)
                    == len(update_epoch_uav_operation_full_rollout_kls)
                ):
                    raise RuntimeError("branch epoch KL diagnostics are incomplete")
                cache_epoch_approx_kl = float(
                    np.mean(update_epoch_cache_approx_kls)
                )
                uav_epoch_approx_kl = float(
                    np.mean(update_epoch_uav_approx_kls)
                )
                uav_operation_epoch_full_rollout_kl = float(
                    np.mean(update_epoch_uav_operation_full_rollout_kls)
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
                    else bool(
                        uav_operation_epoch_full_rollout_kl > target_kl_value
                    )
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
                        uav_active=bool(uav_active),
                        cache_epoch_kl=cache_epoch_approx_kl,
                        uav_epoch_kl=uav_operation_epoch_full_rollout_kl,
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
                        "uav_operation_full_rollout_kl": (
                            uav_operation_epoch_full_rollout_kl
                        ),
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
        if exact_uav_credit_enabled:
            # The freeze is scoped to this outer PPO update.  Restore module
            # trainability before the next rollout/update while retaining the
            # parameter-delta diagnostics captured below.
            for parameter in self.policy.uav_operation_head.parameters():
                parameter.requires_grad_(True)
            for parameter in self.policy.uav_encoder.parameters():
                parameter.requires_grad_(True)
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

        def _named_parameter_delta_norm(before_parameters: dict[str, Any], module: Any) -> float:
            delta_squared_local = 0.0
            current_names = set()
            for name, parameter in module.named_parameters():
                key = str(name)
                current_names.add(key)
                if key not in before_parameters:
                    raise RuntimeError(f"candidate scorer parameter disappeared: {key}")
                delta_squared_local += float(
                    torch.sum((parameter.detach() - before_parameters[key]) ** 2).cpu()
                )
            if current_names != set(before_parameters):
                raise RuntimeError("candidate scorer parameter names changed during PPO update")
            return float(np.sqrt(delta_squared_local))

        cache_target_scorer_parameter_delta_norm = _named_parameter_delta_norm(
            cache_target_scorer_before,
            self.policy.cache_candidate_scorer,
        )
        uav_target_scorer_parameter_delta_norm = _named_parameter_delta_norm(
            uav_target_scorer_before,
            self.policy.uav_candidate_scorer,
        )
        if exact_uav_credit_enabled:
            assert uav_operation_head_before is not None
            assert uav_encoder_before is not None
            uav_operation_head_parameter_delta_norm = _named_parameter_delta_norm(
                uav_operation_head_before,
                self.policy.uav_operation_head,
            )
            uav_encoder_parameter_delta_norm = _named_parameter_delta_norm(
                uav_encoder_before,
                self.policy.uav_encoder,
            )
        else:
            uav_operation_head_parameter_delta_norm = math.nan
            uav_encoder_parameter_delta_norm = math.nan
        uav_operation_teacher_diagnostics: dict[str, Any] | None = None
        if exact_uav_credit_enabled:
            assert uav_operation_masks is not None
            assert uav_operation_old_probabilities_t is not None
            assert uav_operation_teacher_probabilities_t is not None
            assert uav_operation_q_move_teacher_t is not None
            assert conditional_candidate_q_values_t is not None
            assert conditional_candidate_valid_mask_t is not None
            with torch.no_grad():
                final_operation_evaluation = self.policy.evaluate_joint_cache_uav_operation(
                    observations,
                    cache_contexts,
                    cache_operation_masks,
                    cache_operation_actions,
                    cache_candidate_features,
                    cache_candidate_masks,
                    cache_selected_indices,
                    uav_contexts,
                    uav_operation_masks,
                    uav_operation_actions,
                    uav_candidate_features,
                    uav_candidate_masks,
                    uav_selected_indices,
                )
            if final_operation_evaluation.uav_operation_log_probs_all is None:
                raise RuntimeError(
                    "final UAV operation evaluation did not expose full log-probabilities"
                )
            final_operation_probabilities = torch.exp(
                final_operation_evaluation.uav_operation_log_probs_all
            )
            if not bool(torch.all(torch.isfinite(final_operation_probabilities)).item()):
                raise FloatingPointError("final UAV operation probabilities are non-finite")
            teacher_probabilities = uav_operation_teacher_probabilities_t
            q_stay_tensor = torch.as_tensor(
                uav_operation_credit["q_stay"],
                dtype=torch.float32,
                device=self.device,
            )
            q_move_tensor = torch.as_tensor(
                uav_operation_credit["q_move"],
                dtype=torch.float32,
                device=self.device,
            )
            q_move_teacher_tensor = uav_operation_q_move_teacher_t
            candidate_valid = conditional_candidate_valid_mask_t
            move_legal = candidate_valid.any(dim=-1)
            candidate_q_masked = torch.where(
                candidate_valid,
                conditional_candidate_q_values_t,
                torch.full_like(conditional_candidate_q_values_t, -float("inf")),
            )
            q_move_best_tensor = candidate_q_masked.amax(dim=-1)
            policy_expected_q = (
                final_operation_probabilities[..., 0] * q_stay_tensor
                + final_operation_probabilities[..., 1] * q_move_tensor
            )
            teacher_expected_q = (
                teacher_probabilities[..., 0] * q_stay_tensor
                + teacher_probabilities[..., 1] * q_move_teacher_tensor
            )
            operation_exact_q_regret = torch.clamp(
                teacher_expected_q - policy_expected_q,
                min=0.0,
            )
            legal_count = int(move_legal.sum().detach().cpu())

            def _active_mean(values: Any, mask: Any) -> float | None:
                if not bool(torch.any(mask).item()):
                    return None
                return float(values[mask].mean().detach().cpu())

            operation_full_rollout_kl = self._evaluate_uav_branch_approx_kl_full_rollout(
                observations=observations,
                cache_contexts=cache_contexts,
                cache_operation_masks=cache_operation_masks,
                cache_operation_actions=cache_operation_actions,
                cache_candidate_features=cache_candidate_features,
                cache_candidate_masks=cache_candidate_masks,
                cache_selected_indices=cache_selected_indices,
                uav_contexts=uav_contexts,
                uav_operation_masks=uav_operation_masks,
                uav_operation_actions=uav_operation_actions,
                uav_candidate_features=uav_candidate_features,
                uav_candidate_masks=uav_candidate_masks,
                uav_selected_indices=uav_selected_indices,
                old_uav_operation_probabilities=uav_operation_old_probabilities_t,
            )
            q_move_teacher_better = move_legal & (
                q_move_teacher_tensor > q_stay_tensor
            )
            q_move_best_better = move_legal & (q_move_best_tensor > q_stay_tensor)
            uav_operation_teacher_diagnostics = {
                "Q_MOVE_POLICY_MEAN": float(q_move_tensor.mean().detach().cpu()),
                "Q_MOVE_TEACHER_MEAN": _active_mean(
                    q_move_teacher_tensor, move_legal
                ),
                "Q_MOVE_BEST_MEAN": _active_mean(q_move_best_tensor, move_legal),
                "Q_STAY_MEAN": float(q_stay_tensor.mean().detach().cpu()),
                "MOVE_POLICY_BETTER_THAN_STAY_FRACTION": float(
                    (q_move_tensor > q_stay_tensor)
                    .to(torch.float32)
                    .mean()
                    .detach()
                    .cpu()
                ),
                "MOVE_TEACHER_BETTER_THAN_STAY_FRACTION": (
                    None
                    if legal_count == 0
                    else float(q_move_teacher_better.sum().detach().cpu())
                    / float(legal_count)
                ),
                "MOVE_BEST_BETTER_THAN_STAY_FRACTION": (
                    None
                    if legal_count == 0
                    else float(q_move_best_better.sum().detach().cpu())
                    / float(legal_count)
                ),
                "OPERATION_TEACHER_MOVE_PROB_MEAN": float(
                    teacher_probabilities[..., 1].mean().detach().cpu()
                ),
                "OPERATION_POLICY_MOVE_PROB_MEAN": float(
                    final_operation_probabilities[..., 1].mean().detach().cpu()
                ),
                "OPERATION_TEACHER_POLICY_KL": float(
                    self._operation_teacher_kl(
                        final_operation_evaluation.uav_operation_log_probs_all,
                        teacher_probabilities,
                        uav_operation_masks,
                    )
                    .detach()
                    .cpu()
                ),
                "OPERATION_EXPECTED_EXACT_Q": float(
                    policy_expected_q.mean().detach().cpu()
                ),
                "OPERATION_TEACHER_EXPECTED_EXACT_Q": float(
                    teacher_expected_q.mean().detach().cpu()
                ),
                "OPERATION_EXACT_Q_REGRET": float(
                    operation_exact_q_regret.mean().detach().cpu()
                ),
                "UAV_OPERATION_FULL_ROLLOUT_KL": float(operation_full_rollout_kl),
            }
            uav_operation_full_rollout_kl_current = float(operation_full_rollout_kl)
        with torch.no_grad():
            if self._uses_exact_cache_credit():
                assert cache_target_policy_probabilities_before is not None
                assert cache_candidate_features_t is not None
                assert cache_candidate_valid_mask_t is not None
                assert cache_candidate_q_values_t is not None
                cache_target_policy_probabilities_after = (
                    self._conditional_candidate_probabilities(
                        cache_contexts,
                        cache_candidate_features_t,
                        cache_candidate_valid_mask_t,
                        int(self.policy.cache_feature_dim),
                        self.policy.cache_candidate_scorer,
                    )
                )
                if str(cache_credit.get("candidate_value_source", "")) in {
                    "FULL_EXACT_Q",
                    "FULL_EXACT_AUDIT",
                }:
                    cache_exact_q_quality = _exact_q_policy_quality_metrics(
                        cache_target_policy_probabilities_before.detach().cpu().numpy(),
                        cache_target_policy_probabilities_after.detach().cpu().numpy(),
                        cache_candidate_q_values_t.detach().cpu().numpy(),
                        cache_candidate_valid_mask_t.detach().cpu().numpy(),
                    )
            if self._uses_exact_uav_candidate_credit():
                assert uav_target_policy_probabilities_before is not None
                assert conditional_candidate_valid_mask_t is not None
                assert conditional_candidate_q_values_t is not None
                uav_target_policy_probabilities_after = (
                    self._conditional_candidate_probabilities(
                        uav_contexts,
                        uav_candidate_features,
                        conditional_candidate_valid_mask_t,
                        int(self.policy.uav_feature_dim),
                        self.policy.uav_candidate_scorer,
                    )
                )
                uav_exact_q_quality = _exact_q_policy_quality_metrics(
                    uav_target_policy_probabilities_before.detach().cpu().numpy(),
                    uav_target_policy_probabilities_after.detach().cpu().numpy(),
                    conditional_candidate_q_values_t.detach().cpu().numpy(),
                    conditional_candidate_valid_mask_t.detach().cpu().numpy(),
                )
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
            cache_trigger_policy_loss=(
                float(np.mean(cache_trigger_policy_loss_values))
                if cache_trigger_policy_loss_values
                else math.nan
            ),
            cache_candidate_policy_loss=(
                float(np.mean(cache_candidate_policy_loss_values))
                if cache_candidate_policy_loss_values
                else math.nan
            ),
            uav_policy_loss=(
                float(np.mean(uav_policy_loss_values))
                if uav_policy_loss_values
                else math.nan
            ),
            uav_operation_policy_loss=(
                float(np.mean(uav_operation_policy_loss_values))
                if uav_operation_policy_loss_values
                else math.nan
            ),
            actor_policy_loss=(
                float(np.mean(actor_policy_loss_values))
                if actor_policy_loss_values
                else math.nan
            ),
            uav_candidate_policy_loss=(
                float(np.mean(candidate_policy_loss_values))
                if candidate_policy_loss_values
                else math.nan
            ),
            uav_candidate_advantage_mean_abs=(
                float(np.mean(candidate_advantage_mean_abs_values))
                if candidate_advantage_mean_abs_values
                else math.nan
            ),
            uav_candidate_advantage_weighted_mean_residual=(
                float(np.mean(candidate_advantage_weighted_mean_residual_values))
                if candidate_advantage_weighted_mean_residual_values
                else math.nan
            ),
            uav_candidate_advantage_normalization=(
                "EXACT_Q_LISTWISE" if self._uses_exact_uav_candidate_credit() else ""
            ),
            uav_candidate_advantage_source=(
                "EXACT_Q_LISTWISE"
                if self._uses_exact_uav_candidate_credit()
                else ""
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
            joint_kl_diagnostic=bool(branch_kl_instrumentation_enabled),
            cache_independent_stop=bool(cache_independent_stop),
            uav_independent_stop=bool(uav_independent_stop),
            both_inactive_stop=bool(both_inactive_stop),
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
            uav_operation_steps_executed=int(uav_operation_steps_executed),
            uav_operation_steps_suppressed=int(uav_operation_steps_suppressed),
            uav_candidate_full_steps=int(uav_candidate_full_steps),
            uav_candidate_only_steps_after_operation_stop=int(
                uav_candidate_only_steps_after_operation_stop
            ),
            uav_operation_full_rollout_kl=(
                float(uav_operation_full_rollout_kl_current)
                if uav_operation_full_rollout_kl_current is not None
                else math.nan
            ),
            uav_operation_teacher_diagnostics=uav_operation_teacher_diagnostics,
            uav_operation_head_parameter_delta_norm=float(
                uav_operation_head_parameter_delta_norm
            ),
            uav_encoder_parameter_delta_norm=float(uav_encoder_parameter_delta_norm),
            global_critic_steps=int(global_critic_steps),
            uav_operation_advantages_raw=(
                None
                if not exact_uav_credit_enabled
                else np.asarray(
                    uav_operation_credit["raw_advantages"], dtype=np.float32
                ).copy()
            ),
            uav_operation_advantages_scaled=(
                None
                if not exact_uav_credit_enabled
                else np.asarray(
                    uav_operation_credit["scaled_advantages"], dtype=np.float32
                ).copy()
            ),
            uav_operation_q_stay=(
                None
                if not exact_uav_credit_enabled
                else np.asarray(uav_operation_credit["q_stay"], dtype=np.float32).copy()
            ),
            uav_operation_q_move=(
                None
                if not exact_uav_credit_enabled
                else np.asarray(uav_operation_credit["q_move"], dtype=np.float32).copy()
            ),
            uav_operation_baseline=(
                None
                if not exact_uav_credit_enabled
                else np.asarray(uav_operation_credit["baseline"], dtype=np.float32).copy()
            ),
            uav_operation_counterfactual_env_executions=int(
                uav_operation_credit.get("counterfactual_environment_executions", 0)
            ),
            uav_exact_q_branch_request_count=int(
                uav_operation_credit.get("uav_exact_q_branch_request_count", 0)
            ),
            uav_exact_q_unique_branch_count=int(
                uav_operation_credit.get("uav_exact_q_unique_branch_count", 0)
            ),
            uav_exact_q_duplicate_branch_count=int(
                uav_operation_credit.get("uav_exact_q_duplicate_branch_count", 0)
            ),
            uav_exact_q_cache_hit_count=int(
                uav_operation_credit.get("uav_exact_q_cache_hit_count", 0)
            ),
            uav_exact_q_execute_slot_call_count=int(
                uav_operation_credit.get("execute_slot_call_count", 0)
            ),
            uav_exact_q_matching_call_count=int(
                uav_operation_credit.get("matching_call_count", 0)
            ),
            uav_exact_q_kkt_call_count=int(
                uav_operation_credit.get("kkt_call_count", 0)
            ),
            uav_operation_mean_move_candidate_count=float(
                uav_operation_credit.get("mean_move_candidate_count", math.nan)
            ),
            uav_operation_credit_runtime_seconds=float(
                uav_operation_credit.get("credit_runtime_seconds", math.nan)
            ),
            uav_operation_advantage_normalization=str(
                uav_operation_credit.get("advantage_normalization", "")
            ),
            uav_operation_advantage_centering=(
                None
                if not exact_uav_credit_enabled
                else bool(uav_operation_credit.get("advantage_centering", False))
            ),
            uav_operation_advantage_sign_flip_count=int(
                uav_operation_credit.get("advantage_sign_flip_count", 0)
            ),
            uav_operation_ratio_granularity=str(
                uav_operation_credit.get("ratio_granularity", "")
            ),
            uav_operation_diagnostics=uav_operation_diagnostics,
            uav_operation_diagnostic_arrays=uav_operation_diagnostic_arrays,
            cache_trigger_advantage_mean_abs=(
                float(cache_credit.get("trigger_advantage_mean_abs", math.nan))
                if cache_credit
                else math.nan
            ),
            cache_candidate_advantage_mean_abs=(
                float(cache_credit.get("candidate_advantage_mean_abs", math.nan))
                if cache_credit
                else math.nan
            ),
            cache_trigger_advantage_weighted_mean_residual=(
                float(
                    np.mean(
                        np.abs(
                            np.asarray(
                                cache_credit.get(
                                    "trigger_weighted_mean_residual", []
                                ),
                                dtype=np.float64,
                            )
                        )
                    )
                )
                if cache_credit
                else math.nan
            ),
            cache_candidate_advantage_weighted_mean_residual=(
                float(
                    np.mean(
                        np.abs(
                            np.asarray(
                                cache_credit.get(
                                    "candidate_weighted_mean_residual", []
                                ),
                                dtype=np.float64,
                            )
                        )
                    )
                )
                if cache_credit
                else math.nan
            ),
            cache_trigger_advantage_mean_abs_by_role=(
                dict(cache_credit.get("trigger_advantage_mean_abs_by_role", {}))
                if cache_credit
                else None
            ),
            cache_credit_counterfactual_env_executions=int(
                cache_credit.get("counterfactual_environment_executions", 0)
            ),
            cache_credit_runtime_seconds=float(
                cache_credit.get("credit_runtime_seconds", math.nan)
            ),
            cache_candidate_q_source=str(
                cache_credit.get("candidate_value_source", "")
            ),
            cache_candidate_value_source=str(
                cache_credit.get("candidate_value_source", "")
            ),
            cache_logical_candidate_count=int(
                cache_credit.get("logical_candidate_count", 0)
            ),
            cache_exact_candidate_count=int(
                cache_credit.get("exact_candidate_count", 0)
            ),
            cache_surrogate_candidate_count=int(
                cache_credit.get("surrogate_candidate_count", 0)
            ),
            cache_keep_exact_count=int(cache_credit.get("keep_exact_count", 0)),
            cache_full_exact_audit_count=int(
                cache_credit.get("full_exact_audit_count", 0)
            ),
            cache_counterfactual_executions_saved=int(
                cache_credit.get("counterfactual_executions_saved", 0)
            ),
            cache_counterfactual_execution_reduction_fraction=float(
                cache_credit.get(
                    "counterfactual_execution_reduction_fraction", math.nan
                )
            ),
            cache_surrogate_inference_runtime_seconds=float(
                cache_credit.get("surrogate_inference_runtime_seconds", math.nan)
            ),
            cache_surrogate_train_runtime_seconds=float(
                cache_credit.get("surrogate_train_runtime_seconds", math.nan)
            ),
            cache_surrogate_delta_q_mae=float(
                cache_credit.get("surrogate_delta_q_mae", math.nan)
            ),
            cache_surrogate_delta_q_rmse=float(
                cache_credit.get("surrogate_delta_q_rmse", math.nan)
            ),
            cache_surrogate_rank_correlation=float(
                cache_credit.get("surrogate_rank_correlation", math.nan)
            ),
            cache_surrogate_uncertainty_mean=float(
                cache_credit.get("surrogate_uncertainty_mean", math.nan)
            ),
            cache_surrogate_uncertainty_p90=float(
                cache_credit.get("surrogate_uncertainty_p90", math.nan)
            ),
            cache_surrogate_screening_active=bool(
                self.cache_surrogate_screening_active
            ),
            cache_surrogate_screening_ready=bool(
                self.cache_surrogate_screening_ready
            ),
            cache_surrogate_selected_k=(
                None
                if self.cache_surrogate_selected_k is None
                else int(self.cache_surrogate_selected_k)
            ),
            cache_surrogate_audit_statistics=(
                None
                if self.cache_surrogate_audit_statistics is None
                else dict(self.cache_surrogate_audit_statistics)
            ),
            cache_total_logical_branches=int(
                cache_credit.get("total_logical_branch_count", 0)
            ),
            cache_full_exact_audit_branches=int(
                cache_credit.get("full_exact_audit_branch_count", 0)
            ),
            cache_normal_slot_counterfactual_branches=int(
                cache_credit.get("normal_slot_counterfactual_branch_count", 0)
            ),
            cache_counterfactual_executions_avoided=int(
                cache_credit.get("counterfactual_executions_avoided", 0)
            ),
            cache_q_critic_mae=float(
                (cache_credit.get("heldout_audit_metrics") or {}).get("q_mae_stats", {}).get("mean", math.nan)
            ),
            cache_q_critic_rmse=float(
                (cache_credit.get("heldout_audit_metrics") or {}).get("q_rmse_stats", {}).get("mean", math.nan)
            ),
            cache_q_critic_rank_correlation=float(
                (cache_credit.get("heldout_audit_metrics") or {}).get("rank_correlation", math.nan)
            ),
            cache_q_critic_argmax_accuracy=float(
                (cache_credit.get("heldout_audit_metrics") or {}).get("argmax_accuracy", math.nan)
            ),
            cache_q_critic_recall_at_3=float(
                (cache_credit.get("heldout_audit_metrics") or {}).get("recall_at_3", math.nan)
            ),
            cache_q_critic_recall_at_5=float(
                (cache_credit.get("heldout_audit_metrics") or {}).get("recall_at_5", math.nan)
            ),
            cache_q_critic_selected_exact_regret=(
                None
                if cache_credit.get("heldout_audit_metrics") is None
                else dict(
                    cache_credit["heldout_audit_metrics"].get(
                        "selected_exact_regret_stats", {}
                    )
                )
            ),
            cache_q_critic_normalized_regret_p95=float(
                (cache_credit.get("heldout_audit_metrics") or {}).get(
                    "normalized_regret_p95", math.nan
                )
            ),
            cache_q_critic_usable_for_policy_credit=str(
                cache_credit.get("cache_q_critic_usable_for_policy_credit", "INCONCLUSIVE")
            ),
            cache_q_critic_train_seconds=float(
                cache_credit.get("cache_q_critic_train_seconds", math.nan)
            ),
            cache_q_critic_inference_seconds=float(
                cache_credit.get("cache_q_critic_inference_seconds", math.nan)
            ),
            cache_audit_exact_seconds=float(
                cache_credit.get("cache_audit_exact_seconds", math.nan)
            ),
            cache_heldout_audit_before_training=bool(
                cache_credit.get("heldout_audit_before_training", False)
            ),
            cache_actual_action_q_anchor=str(
                cache_credit.get("actual_action_q_anchor", "NO")
            ),
            cache_q_critic_audit_metrics=(
                None
                if cache_credit.get("heldout_audit_metrics") is None
                else dict(cache_credit["heldout_audit_metrics"])
            ),
            cache_target_scorer_parameter_delta_norm=float(
                cache_target_scorer_parameter_delta_norm
            ),
            uav_target_scorer_parameter_delta_norm=float(
                uav_target_scorer_parameter_delta_norm
            ),
            cache_top1_exact_q_accuracy_before=(
                math.nan
                if cache_exact_q_quality is None
                else float(cache_exact_q_quality["top1_before"])
            ),
            cache_top1_exact_q_accuracy_after=(
                math.nan
                if cache_exact_q_quality is None
                else float(cache_exact_q_quality["top1_after"])
            ),
            cache_mean_exact_q_regret_before=(
                math.nan
                if cache_exact_q_quality is None
                else float(cache_exact_q_quality["mean_regret_before"])
            ),
            cache_mean_exact_q_regret_after=(
                math.nan
                if cache_exact_q_quality is None
                else float(cache_exact_q_quality["mean_regret_after"])
            ),
            cache_expected_exact_q_before=(
                math.nan
                if cache_exact_q_quality is None
                else float(cache_exact_q_quality["expected_q_before"])
            ),
            cache_expected_exact_q_after=(
                math.nan
                if cache_exact_q_quality is None
                else float(cache_exact_q_quality["expected_q_after"])
            ),
            cache_expected_exact_q_delta=(
                math.nan
                if cache_exact_q_quality is None
                else float(cache_exact_q_quality["expected_q_delta"])
            ),
            cache_target_policy_kl_old_new=(
                math.nan
                if cache_exact_q_quality is None
                else float(cache_exact_q_quality["policy_kl_old_new"])
            ),
            uav_top1_exact_q_accuracy_before=(
                math.nan
                if uav_exact_q_quality is None
                else float(uav_exact_q_quality["top1_before"])
            ),
            uav_top1_exact_q_accuracy_after=(
                math.nan
                if uav_exact_q_quality is None
                else float(uav_exact_q_quality["top1_after"])
            ),
            uav_mean_exact_q_regret_before=(
                math.nan
                if uav_exact_q_quality is None
                else float(uav_exact_q_quality["mean_regret_before"])
            ),
            uav_mean_exact_q_regret_after=(
                math.nan
                if uav_exact_q_quality is None
                else float(uav_exact_q_quality["mean_regret_after"])
            ),
            uav_expected_exact_q_before=(
                math.nan
                if uav_exact_q_quality is None
                else float(uav_exact_q_quality["expected_q_before"])
            ),
            uav_expected_exact_q_after=(
                math.nan
                if uav_exact_q_quality is None
                else float(uav_exact_q_quality["expected_q_after"])
            ),
            uav_expected_exact_q_delta=(
                math.nan
                if uav_exact_q_quality is None
                else float(uav_exact_q_quality["expected_q_delta"])
            ),
            uav_target_policy_kl_old_new=(
                math.nan
                if uav_exact_q_quality is None
                else float(uav_exact_q_quality["policy_kl_old_new"])
            ),
            cache_proposal_diagnostics=cache_proposal_diagnostics,
            uav_proposal_diagnostics=uav_proposal_diagnostics,
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
        learning_rate_metadata = self.learning_rate_metadata()
        payload = {
            "checkpoint_version": self._expected_checkpoint_version(),
            "method": PPO_V2_METHOD,
            "policy_schema": self.policy.architecture_id,
            "experiment_id": learning_rate_metadata["experiment_id"],
            "control_mode": learning_rate_metadata["control_mode"],
            "learning_rate_metadata": learning_rate_metadata,
            **learning_rate_metadata,
            "cache_credit_granularity": None,
            "cache_kl_control": str(self.cache_kl_control),
            **self.cache_surrogate_metadata(),
            "configuration_control_mode": self.context.configuration_control_mode,
            "advantage_normalization_mode": (
                self.hyperparameters.advantage_normalization_mode
            ),
            "operation_advantage_normalization_mode": (
                self.hyperparameters.operation_advantage_normalization_mode
            ),
            "operation_advantage_reference_scale": (
                self.hyperparameters.operation_advantage_reference_scale
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
        if self.cache_surrogate_ensemble is not None:
            payload.update(
                {
                    "cache_surrogate_training_count": int(
                        self.cache_surrogate_training_count
                    ),
                    "cache_surrogate_screening_ready": bool(
                        self.cache_surrogate_screening_ready
                    ),
                    "cache_surrogate_screening_active": bool(
                        self.cache_surrogate_screening_active
                    ),
                    "cache_surrogate_selected_k": (
                        None
                        if self.cache_surrogate_selected_k is None
                        else int(self.cache_surrogate_selected_k)
                    ),
                    "cache_surrogate_shadow_complete": bool(
                        self.cache_surrogate_shadow_complete
                    ),
                    "cache_surrogate_audit_statistics": (
                        None
                        if self.cache_surrogate_audit_statistics is None
                        else dict(self.cache_surrogate_audit_statistics)
                    ),
                    "cache_surrogate_audit_history": [
                        dict(item) for item in self.cache_surrogate_audit_history
                    ],
                    "rng_state": {
                        **payload["rng_state"],
                        "cache_surrogate": self.cache_surrogate_rng.bit_generator.state,
                    },
                }
            )
        if self.cache_action_q_critic is not None:
            payload.update(
                {
                    "cache_action_q_training_count": int(
                        self.cache_action_q_training_count
                    ),
                    "cache_action_q_optimizer_steps": int(
                        self.cache_action_q_optimizer_steps
                    ),
                    "cache_action_q_audit_statistics": (
                        None
                        if self.cache_action_q_audit_statistics is None
                        else dict(self.cache_action_q_audit_statistics)
                    ),
                }
            )
        if self.cache_surrogate_ensemble is not None:
            if self.cache_surrogate_optimizer is None:
                raise RuntimeError("Cache surrogate optimizer is not initialized.")
            payload.update(
                {
                    "cache_surrogate_state_dict": self.cache_surrogate_ensemble.state_dict(),
                    "cache_surrogate_optimizer_state_dict": self.cache_surrogate_optimizer.state_dict(),
                }
            )
        if self.cache_action_q_critic is not None:
            if self.cache_action_q_optimizer is None:
                raise RuntimeError("Cache Action-Q optimizer is not initialized.")
            payload.update(
                {
                    "cache_action_q_state_dict": self.cache_action_q_critic.state_dict(),
                    "cache_action_q_optimizer_state_dict": self.cache_action_q_optimizer.state_dict(),
                }
            )
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
        _validate_learning_rate_metadata(
            payload.get("learning_rate_metadata"),
            self.learning_rate_metadata(),
        )
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
        if self._uses_exact_cache_credit():
            expected_action_q_metadata = self.cache_action_q_metadata()
            for field, expected in expected_action_q_metadata.items():
                if payload.get(field) != expected:
                    raise ValueError(
                        "PPO Cache Action-Q checkpoint metadata mismatch: "
                        f"field={field}, checkpoint={payload.get(field)!r}, "
                        f"expected={expected!r}."
                    )
            if self.cache_action_q_critic is None or self.cache_action_q_optimizer is None:
                raise RuntimeError("Cache Action-Q critic is not initialized for resume")
            if (
                "cache_action_q_state_dict" not in payload
                or "cache_action_q_optimizer_state_dict" not in payload
            ):
                raise ValueError(
                    "Cache Action-Q checkpoint must contain critic and optimizer state."
                )
            self.cache_action_q_critic.load_state_dict(
                payload["cache_action_q_state_dict"], strict=True
            )
            self.cache_action_q_optimizer.load_state_dict(
                payload["cache_action_q_optimizer_state_dict"]
            )
            self.cache_action_q_training_count = int(
                payload.get("cache_action_q_training_count", -1)
            )
            self.cache_action_q_optimizer_steps = int(
                payload.get("cache_action_q_optimizer_steps", -1)
            )
            if self.cache_action_q_training_count < 0 or self.cache_action_q_optimizer_steps < 0:
                raise ValueError("Cache Action-Q training counters must be non-negative")
            raw_action_q_statistics = payload.get("cache_action_q_audit_statistics")
            self.cache_action_q_audit_statistics = (
                None if raw_action_q_statistics is None else dict(raw_action_q_statistics)
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
        if "cache_surrogate" in rng_state:
            self.cache_surrogate_rng.bit_generator.state = rng_state[
                "cache_surrogate"
            ]
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
    actor_learning_rate_override: float | None = None,
    critic_learning_rate_override: float | None = None,
    lr_experiment_axis: str | None = None,
    lr_experiment_value: float | None = None,
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
    trainer = PPOV2Trainer(
        context,
        actor_learning_rate_override=actor_learning_rate_override,
        critic_learning_rate_override=critic_learning_rate_override,
        lr_experiment_axis=lr_experiment_axis,
        lr_experiment_value=lr_experiment_value,
    )
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
    episode_uav_operation_diagnostic_arrays: list[dict[str, np.ndarray]] = []

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
            episode_uav_operation_diagnostic_arrays = []
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
        if update.uav_operation_diagnostic_arrays is not None:
            episode_uav_operation_diagnostic_arrays.append(
                update.uav_operation_diagnostic_arrays
            )
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
                "parameter_delta_norm": float(update.parameter_delta_norm),
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
                **_branch_kl_update_metric_fields(update),
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
            exact_uav_operation_diagnostic_metrics = (
                merge_uav_operation_diagnostic_arrays(
                    episode_uav_operation_diagnostic_arrays
                )
                if episode_uav_operation_diagnostic_arrays
                else {}
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
                    **_branch_update_metric_fields(update),
                    **exact_uav_operation_diagnostic_metrics,
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
    parser.add_argument(
        "--config",
        default="configs/proposed/system_config.toml",
    )
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
    parser.add_argument(
        "--actor-learning-rate",
        type=float,
        default=None,
        help="runtime actor learning-rate override; default: use config",
    )
    parser.add_argument(
        "--critic-learning-rate",
        type=float,
        default=None,
        help="runtime critic learning-rate override; default: use config",
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
        actor_learning_rate_override=args.actor_learning_rate,
        critic_learning_rate_override=args.critic_learning_rate,
        resume=bool(args.resume),
        verbose=bool(args.verbose),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
