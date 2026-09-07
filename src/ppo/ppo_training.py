#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PPO formal training protocol, dry-run, and bounded integration sanity.

默认CLI只执行 FormalTrainingProtocol dry-run/static validation：
不执行完整episode、不调用backward、不调用optimizer.step。

本模块同时提供正式训练/评估执行器函数；这些函数只有在未来caller显式传入
``execute=True`` 或直接调用执行器时才会真正进入environment/update loop。

保留最多2个rollout、2次update的开发期集成验证：

    rollout_1 -> update_1 -> rollout_2 -> update_2 -> STOP

硬限制写在代码中，不写入 system_config.toml，不代表未来正式训练预算。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import inspect
import math
import os
import random
import tempfile
import time
import warnings
from dataclasses import MISSING, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from common import project_config
from decision import cache_configuration, slot_environment, task_allocation, uav_configuration
from ppo import (
    causal_artifacts,
    hierarchical_reconfiguration_gate,
    ppo_rollout,
    ppo_trigger_policy,
    ppo_update,
    reference_drift,
)
from scenario import task_scenario

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SANITY_MAX_ROLLOUTS = 2
SANITY_MAX_STEPS_PER_ROLLOUT = 8
SANITY_MAX_TOTAL_ENV_STEPS = 16
SANITY_MAX_OPTIMIZER_STEPS = 2

# 正式训练持久化版本。checkpoint/history只在caller显式传入run_dir且
# execute=True时写入；不会写入data/generated或固定results目录。
CHECKPOINT_VERSION = 1
TRAINING_HISTORY_SCHEMA_VERSION = 1
COUNTERFACTUAL_CHECKPOINT_VERSION = 2
COUNTERFACTUAL_TRAINING_HISTORY_SCHEMA_VERSION = 2
# 固定 MC return scale 改变 centralized critic 的 target/output 语义；旧 V2
# checkpoint/history 不允许静默按新尺度 resume，因此显式递增 centralized schema。
CENTRALIZED_COUNTERFACTUAL_CHECKPOINT_VERSION = 4
CENTRALIZED_COUNTERFACTUAL_TRAINING_HISTORY_SCHEMA_VERSION = 4
# Bumped because the formal sparse Direct-CF Actor now has a different
# parameter layout (encoder + one head per dynamic trigger; autoregressive block frozen).
SPARSE_DIRECT_CF_CHECKPOINT_VERSION = 6
SPARSE_DIRECT_CF_TRAINING_HISTORY_SCHEMA_VERSION = 5
# Frozen-policy multi-episode batching is a distinct formal method/checkpoint
# schema.  It deliberately does not reuse the immediate-update V3 schema.
FROZEN_POLICY_BATCH_CHECKPOINT_VERSION = 7
FROZEN_POLICY_BATCH_TRAINING_HISTORY_SCHEMA_VERSION = 1
# Causal pre/post update artifacts and episode-start snapshots are now part of
# the RefDrift persistence contract.  The algorithm/method/architecture IDs do
# not change; only persistence schemas advance.
REFERENCE_DRIFT_CHECKPOINT_VERSION = 9
REFERENCE_DRIFT_TRAINING_HISTORY_SCHEMA_VERSION = 3
# Hierarchical WHEN (Gate) + existing WHAT formal path.  This is a distinct
# persistence schema because it carries an independent policy/Adam/RNG state
# and a second pending counterfactual stream.
HIERARCHICAL_GATE_CHECKPOINT_VERSION = 10
HIERARCHICAL_GATE_TRAINING_HISTORY_SCHEMA_VERSION = 4
HIERARCHICAL_GATE_CF_SAMPLES_PER_EPISODE = 8
HIERARCHICAL_GATE_CF_BATCH_EPISODES = 4
LATEST_CHECKPOINT_FILENAME = "latest_checkpoint.pt"
TRAINING_HISTORY_FILENAME = "training_history.json"
EXECUTION_LOCATION_ORDER = ("Local", "SV", "RSU", "UAV", "Cloud")
K8_EPISODE1_BASELINE = {
    "steps": 276,
    "G": 780.4936401668566,
    "Q": 1089.5331440803836,
    "cache": 245.07275647419945,
    "UAV": 63.966747439327555,
}


def summary_to_builtin(value: Any) -> Any:
    """递归将正式summary转换为可安全JSON序列化的Python builtin。

    该函数只用于reporting层，不修改输入对象，也不改变训练核心返回的
    dataclass、numpy数组或Tensor。非finite数值明确报错，避免掩盖数值异常。
    """

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: summary_to_builtin(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {
            summary_to_builtin(key): summary_to_builtin(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [summary_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return summary_to_builtin(value.tolist())
    if isinstance(value, np.generic):
        return summary_to_builtin(value.item())
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim == 0:
            return summary_to_builtin(tensor.item())
        return summary_to_builtin(tensor.tolist())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("summary中包含非finite Python float，拒绝静默序列化。")
        return value
    raise TypeError(f"summary包含不支持的类型：{type(value).__name__}")


def format_training_summary(value: Any) -> str:
    """将正式summary转换并格式化为console JSON字符串，不写入文件。"""

    return json.dumps(
        summary_to_builtin(value),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )


@dataclass(frozen=True)
class TrainingSanityContext:
    cfg: dict[str, Any]
    cache_cfg: cache_configuration.CacheOptimizationConfig
    scenarios: dict[str, Any]
    service_catalog: dict[str, Any]
    cache_nodes: dict[str, Any]
    local_cache: dict[str, set[int]]
    flight_boundary: tuple[float, float, float, float]
    action_spec: ppo_trigger_policy.TriggerActionSpec
    observation_spec: ppo_trigger_policy.ConfigurationObservationSpec
    gamma: float
    gae_lambda: float
    terminal_slot: int
    initial_cache: np.ndarray
    initial_uav: dict[str, np.ndarray]


@dataclass(frozen=True)
class UpdateSanitySummary:
    policy_version_before: int
    policy_version_after: int
    pre_ratio_mean: float
    pre_ratio_min: float
    pre_ratio_max: float
    pre_clip_fraction: float
    pre_approx_kl: float
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    gradient_finite: bool
    gradient_norm: float
    parameter_delta_norm: float
    post_ratio_mean: float
    post_ratio_min: float
    post_ratio_max: float
    post_clip_fraction: float
    post_approx_kl: float
    optimizer_step_count_after: int


@dataclass(frozen=True)
class RolloutSanitySummary:
    rollout_index: int
    collected_with_policy_version: int
    origin_indices: list[int]
    slots: list[int]
    times_s: list[float]
    actions: list[list[int]]
    rewards: list[float]
    reward_sum: float
    reward_mean: float
    actual_q_sum: float
    cache_cost_sum: float
    movement_cost_sum: float
    step_count: int
    final_cache_state: np.ndarray
    final_uav_positions: dict[str, tuple[float, float]]
    truncated: bool
    true_terminal: bool


@dataclass(frozen=True)
class BoundedTrainingSanityResult:
    rollout_1: RolloutSanitySummary
    update_1: UpdateSanitySummary
    rollout_2: RolloutSanitySummary | None
    update_2: UpdateSanitySummary | None
    final_policy_version: int
    rollout_count: int
    environment_step_count: int
    optimizer_step_count: int
    optimizer_created_once: bool
    optimizer_type: str
    optimizer_only_policy_parameters: bool
    rollout_origins_continuous_non_overlapping: bool
    rollout_2_first_cache_matches_rollout_1_final: bool | None
    rollout_2_first_uav_matches_rollout_1_final: bool | None
    reward_is_system_utility: bool
    repeated_bounded_sanity_deterministic: bool | None
    formal_validation: bool
    blocker: str


@dataclass(frozen=True)
class PPOOriginSplit:
    """Single source of truth for the PPO train/test origin protocol."""

    train_origin_start: int
    train_origin_end: int
    test_origin_start: int
    test_origin_end: int

    @property
    def train_count(self) -> int:
        return int(self.train_origin_end - self.train_origin_start + 1)

    @property
    def test_count(self) -> int:
        return int(self.test_origin_end - self.test_origin_start + 1)


@dataclass(frozen=True)
class FormalTrainingProtocol:
    """正式PPO训练/held-out test协议定义；只描述协议，不执行训练。

    每个训练episode只使用配置中的 train origin 区间；held-out test 使用同一
    artifact 的独立 test origin 区间。两者均在自己的区间内顺序传播 C/Q，
    test 从与正式训练相同的共同初始 C/Q reset，且不执行 optimizer。
    """

    origin_split: PPOOriginSplit
    episode_origin_indices: np.ndarray
    episode_slots: np.ndarray
    episode_times_s: np.ndarray
    test_origin_indices: np.ndarray
    test_slots: np.ndarray
    test_times_s: np.ndarray
    steps_per_episode: int
    test_steps_per_evaluation: int
    update_epochs: int
    num_minibatches: int
    minibatch_size: int
    expected_optimizer_steps_per_rollout: int
    true_terminal_origin: int
    true_terminal_slot: int
    true_terminal_time_s: float
    test_true_terminal_origin: int
    test_true_terminal_slot: int
    test_true_terminal_time_s: float
    single_trace_training: bool
    initial_cache_source: str
    initial_uav_source: str
    tv_local_cache_source: str
    evaluation_semantics: str

    @property
    def critic_return_scale(self) -> float:
        """centralized critic 固定 return scale，来源是正式episode horizon。"""

        scale = float(self.steps_per_episode)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("FormalTrainingProtocol.steps_per_episode 必须产生有限正 scale。")
        return scale


def _ppo_origin_split_from_project_config(
    project_cfg: dict[str, Any],
    total_origin_count: int | None = None,
) -> PPOOriginSplit:
    """读取并严格校验 PPO train/test origin 边界。

    该函数不复制实验边界常量；具体 start/end 全部来自
    ``algorithm.ppo.data_protocol``。``total_origin_count`` 只用于确认区间
    覆盖当前 prediction scenario artifact 的完整正式 origin 集合。
    """

    raw = project_config.get_config_value(project_cfg, "algorithm.ppo.data_protocol")
    if not isinstance(raw, dict):
        raise ValueError("缺少 algorithm.ppo.data_protocol 配置表。")
    keys = (
        "train_origin_start",
        "train_origin_end",
        "test_origin_start",
        "test_origin_end",
    )
    values: dict[str, int] = {}
    for key in keys:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"algorithm.ppo.data_protocol.{key} 必须是整数。"
            )
        values[key] = int(value)
    split = PPOOriginSplit(**values)
    if split.train_origin_start != 0:
        raise ValueError("PPO train_origin_start 必须为正式 origin 的起点0。")
    if split.train_origin_end < split.train_origin_start:
        raise ValueError("PPO train origin 区间必须非空且按升序排列。")
    if split.test_origin_start != split.train_origin_end + 1:
        raise ValueError("PPO train/test origin 区间必须连续、无重叠且无空洞。")
    if split.test_origin_end < split.test_origin_start:
        raise ValueError("PPO test origin 区间必须非空且按升序排列。")
    if total_origin_count is not None:
        total = int(total_origin_count)
        if total <= 0:
            raise ValueError("prediction scenario origin 数量必须大于0。")
        if split.test_origin_end != total - 1:
            raise ValueError(
                "PPO train/test origin 区间必须完整覆盖 prediction scenario origins："
                f"test_end={split.test_origin_end}, expected={total - 1}。"
            )
        if split.train_origin_start < 0 or split.test_origin_end >= total:
            raise ValueError("PPO train/test origin 边界超出 prediction scenario origins。")
    return split


@dataclass(frozen=True)
class FormalTrainingDryRunResult:
    protocol: FormalTrainingProtocol
    origins_contiguous: bool
    slots_strictly_increasing: bool
    times_strictly_increasing: bool
    minibatch_divisible: bool
    reset_state_resolved: bool
    formal_rollout_executor_available: bool
    ppo_update_executor_available: bool
    formal_training_api_available: bool
    held_out_test_evaluation_api_available: bool
    sample_num_episodes: int
    planned_environment_steps: int
    planned_optimizer_steps: int
    planned_backward_calls: int
    optimizer_instances_planned: int
    executor_dry_run_only: bool
    formal_validation: bool
    environment_steps_executed: int
    backward_calls_executed: int
    optimizer_steps_executed: int
    blocker: str
    credit_assignment: str = ppo_update.LEGACY_CREDIT_ASSIGNMENT
    planned_q_optimizer_steps: int = 0
    q_optimizer_instances_planned: int = 0
    credit_phase: str = "legacy"
    planned_actor_optimizer_steps: int = 0
    actor_optimizer_instances_planned: int = 1
    critic_optimizer_instances_planned: int = 0
    cf_samples_per_episode: int = 0
    planned_counterfactual_branch_steps: int = 0
    cf_sampling_strategy: str = "uniform"
    cf_batch_episodes: int = 1
    gate_cf_samples_per_episode: int = 0
    planned_gate_counterfactual_branch_steps: int = 0
    planned_gate_optimizer_steps: int = 0
    gate_optimizer_instances_planned: int = 0
    # Human-readable algorithm identity/semantics for CLI and experiment logs.
    # This is reporting metadata only; training dispatch continues to use the
    # existing credit_assignment/method helpers below.
    algorithm_reporting: dict[str, Any] | None = None
    planned_total_optimizer_steps: int | None = None
    planned_total_backward_calls: int | None = None


@dataclass(frozen=True)
class PPOEpochSummary:
    """单个PPO epoch的轻量诊断摘要，不保存transition或tensor。"""

    epoch_index: int
    policy_loss_mean: float
    value_loss_mean: float
    entropy_mean: float
    total_loss_mean: float
    approx_kl_mean: float
    clip_fraction_mean: float
    ratio_mean: float
    ratio_min: float
    ratio_max: float
    pre_clipping_grad_norm_mean: float
    pre_clipping_grad_norm_max: float
    minibatch_count: int
    transition_count: int
    optimizer_steps_cumulative: int


@dataclass(frozen=True)
class FormalPPOUpdateSummary:
    """单个完整 rollout 的正式 PPO update 汇总。

    该结构只记录聚合指标，不保存候选配置轨迹或大体量中间数据。
    """

    update_epochs: int
    num_minibatches: int
    minibatch_size: int
    optimizer_steps: int
    policy_loss_mean: float
    value_loss_mean: float
    entropy_mean: float
    approx_kl_mean: float
    clip_fraction_mean: float
    gradient_norm_mean: float
    gradient_finite: bool
    old_log_probs_frozen: bool
    old_values_frozen: bool
    advantages_frozen: bool
    returns_frozen: bool
    each_transition_once_per_epoch: bool
    epoch_summaries: tuple[PPOEpochSummary, ...]
    # per-trigger Q mode 的独立 critic diagnostics；legacy mode 保持 None/0。
    q_loss_mean: float | None = None
    q_gradient_norm_mean: float | None = None
    q_gradient_finite: bool | None = None
    counterfactual_advantage_mean: float | None = None
    counterfactual_advantage_std: float | None = None
    counterfactual_advantage_abs_mean: float | None = None
    free_trigger_decision_count: int | None = None
    q_optimizer_steps: int = 0
    credit_phase: str = "legacy"
    critic_loss_before: float | None = None
    critic_loss_after: float | None = None
    critic_gradient_norm_mean: float | None = None
    critic_gradient_finite: bool | None = None
    critic_action_gap_abs_mean: float | None = None
    critic_action_gap_std: float | None = None
    critic_action_gap_nonzero_fraction: float | None = None
    actor_update_executed: bool = True
    critic_replay_sample_count: int = 0
    # V3 sparse Direct-CF episode diagnostics；legacy/V1/V2 保持默认值。
    cf_sample_count: int = 0
    cf_delta_mean: float | None = None
    cf_delta_std: float | None = None
    cf_delta_abs_mean: float | None = None
    cf_delta_positive_fraction: float | None = None
    cf_advantage_mean: float | None = None
    cf_advantage_std: float | None = None
    cf_advantage_abs_mean: float | None = None
    direct_cf_local_objective_change_proxy: float | None = None
    branch_environment_steps: int = 0
    actor_gradient_finite: bool = True
    # 仅用于V3 reporting；不进入Actor loss、gradient或optimizer语义。
    policy_entropy: float | None = None
    probability_drift_mean_abs: float | None = None
    probability_drift_p95: float | None = None
    probability_drift_p99: float | None = None
    probability_drift_max: float | None = None
    encoder_parameter_delta_norm: float | None = None
    trigger_heads_parameter_delta_norm: float | None = None
    autoregressive_parameter_delta_norm: float | None = None
    value_head_parameter_delta_norm: float | None = None
    logit_head_parameter_delta_norm: float | None = None
    actor_batch_sample_count: int | None = None
    cf_batch_vshift_sum: float | None = None
    cf_batch_vshift_mean: float | None = None
    cf_batch_vshift_positive_fraction: float | None = None
    # Reference-state functional drift gate diagnostics.
    reference_raw_drift_mean: float | None = None
    reference_raw_drift_median: float | None = None
    reference_raw_drift_p95: float | None = None
    reference_raw_drift_p99: float | None = None
    reference_raw_drift_max: float | None = None
    reference_constrained_drift_mean: float | None = None
    reference_constrained_drift_median: float | None = None
    reference_constrained_drift_p95: float | None = None
    reference_constrained_drift_p99: float | None = None
    reference_constrained_drift_max: float | None = None
    selected_drift_mean: float | None = None
    selected_drift_p95: float | None = None
    selected_drift_max: float | None = None
    reference_drift_alpha: float | None = None
    reference_drift_constraint_active: bool | None = None
    reference_probability_drift_cap: float | None = None
    reference_state_count: int | None = None
    reference_unselected_count: int | None = None
    reference_selected_count: int | None = None
    reference_selected_intersection_count: int | None = None
    reference_drift_by_trigger: dict[str, dict[str, float | int]] | None = None
    old_p_reconstruction_max_error: float | None = None
    old_p_reconstruction_checked: bool | None = None
    old_p_reconstruction_tolerance: float | None = None
    probability_dtype: str | None = None
    reference_count_checked: int | None = None
    collection_policy_version: int | None = None
    direct_cf_raw_vshift_sum: float | None = None
    direct_cf_raw_vshift_mean: float | None = None
    direct_cf_raw_vshift_positive_fraction: float | None = None
    direct_cf_constrained_vshift_sum: float | None = None
    direct_cf_constrained_vshift_mean: float | None = None
    direct_cf_constrained_vshift_positive_fraction: float | None = None
    direct_cf_vshift_retention_ratio: float | None = None
    # Hierarchical WHEN Gate update diagnostics.  The existing WHAT fields
    # above retain their Direct-CF meaning and remain independent of these.
    gate_optimizer_steps: int = 0
    gate_loss_mean: float | None = None
    gate_gradient_norm_mean: float | None = None
    gate_gradient_finite: bool | None = None
    gate_probability_shift_mean: float | None = None
    gate_probability_shift_max: float | None = None
    gate_delta_abs_mean: float | None = None
    gate_delta_positive_fraction: float | None = None
    gate_sample_count: int = 0
    gate_branch_environment_steps: int = 0
    gate_parameter_delta_max: float | None = None
    gate_subpolicy_parameter_delta_max: float | None = None
    gate_collection_policy_version: int | None = None


@dataclass(frozen=True)
class TrainingEpisodeSummary:
    """正式训练 episode 级 summary；一个完整rollout对应一次policy_version递增。"""

    episode_index: int
    policy_version_before: int
    policy_version_after: int
    rollout_reward_sum: float
    rollout_reward_mean: float
    # 训练batch仍保持float32；以下字段只诊断float32累计和raw累计的差异。
    batch_reward_sum_float32: float
    batch_vs_raw_reward_sum_error: float
    actual_q_sum: float
    cache_cost_sum: float
    movement_cost_sum: float
    policy_loss_mean: float
    value_loss_mean: float
    entropy_mean: float
    approx_kl_mean: float
    clip_fraction_mean: float
    gradient_norm_mean: float
    optimizer_steps: int
    credit_phase: str = "legacy"
    critic_loss_before: float | None = None
    critic_loss_after: float | None = None
    critic_gradient_norm_mean: float | None = None
    critic_gradient_finite: bool | None = None
    critic_action_gap_abs_mean: float | None = None
    critic_action_gap_std: float | None = None
    counterfactual_advantage_abs_mean: float | None = None
    actor_update_executed: bool = True
    critic_replay_sample_count: int = 0
    cf_sample_count: int = 0
    cf_delta_mean: float | None = None
    cf_delta_std: float | None = None
    cf_delta_abs_mean: float | None = None
    cf_delta_positive_fraction: float | None = None
    cf_advantage_mean: float | None = None
    cf_advantage_std: float | None = None
    cf_advantage_abs_mean: float | None = None
    direct_cf_local_objective_change_proxy: float | None = None
    branch_environment_steps: int = 0
    actor_gradient_finite: bool = True
    # 原Episode1~5没有可靠持久化时保持None；Episode6起写入真实reporting值。
    policy_entropy: float | None = None
    probability_drift_mean_abs: float | None = None
    probability_drift_p95: float | None = None
    probability_drift_p99: float | None = None
    probability_drift_max: float | None = None
    encoder_parameter_delta_norm: float | None = None
    trigger_heads_parameter_delta_norm: float | None = None
    autoregressive_parameter_delta_norm: float | None = None
    value_head_parameter_delta_norm: float | None = None
    # 仅用于K覆盖reporting；不参与reservoir、loss或optimizer。
    cf_trigger_sample_counts: dict[str, int] | list[int] | tuple[int, ...] | None = None
    # Ordinary rollout action=1 counts; reporting-only and RNG/behavior neutral.
    trigger_action1_counts: dict[str, int] | list[int] | tuple[int, ...] | None = None
    # V3 sampling reporting；旧uniform history缺失时按兼容默认值恢复。
    cf_sampling_strategy: str = "uniform"
    cf_effective_sample_count: int = 0
    cf_trigger_free_counts: list[int] | tuple[int, ...] | None = None
    cf_trigger_sampling_weights: list[float] | tuple[float, ...] | None = None
    # Frozen-policy Direct-CF batch reporting.  Defaults preserve old history
    # and the legacy/V1/V2 constructors.
    cf_batch_id: int | None = None
    cf_batch_episode_index: int | None = None
    cf_batch_pending_sample_count: int = 0
    collection_policy_version: int | None = None
    actor_updated_this_episode: bool = True
    actor_optimizer_steps_this_episode: int = 0
    actor_batch_sample_count: int | None = None
    pre_clip_gradient_norm: float | None = None
    cf_batch_vshift_sum: float | None = None
    cf_batch_vshift_mean: float | None = None
    cf_batch_vshift_positive_fraction: float | None = None
    logit_head_parameter_delta_norm: float | None = None
    reference_raw_drift_mean: float | None = None
    reference_raw_drift_median: float | None = None
    reference_raw_drift_p95: float | None = None
    reference_raw_drift_p99: float | None = None
    reference_raw_drift_max: float | None = None
    reference_constrained_drift_mean: float | None = None
    reference_constrained_drift_median: float | None = None
    reference_constrained_drift_p95: float | None = None
    reference_constrained_drift_p99: float | None = None
    reference_constrained_drift_max: float | None = None
    selected_drift_mean: float | None = None
    selected_drift_p95: float | None = None
    selected_drift_max: float | None = None
    reference_drift_alpha: float | None = None
    reference_drift_constraint_active: bool | None = None
    reference_probability_drift_cap: float | None = None
    reference_state_count: int | None = None
    reference_unselected_count: int | None = None
    reference_selected_count: int | None = None
    reference_selected_intersection_count: int | None = None
    reference_drift_by_trigger: dict[str, dict[str, float | int]] | None = None
    old_p_reconstruction_max_error: float | None = None
    old_p_reconstruction_checked: bool | None = None
    old_p_reconstruction_tolerance: float | None = None
    probability_dtype: str | None = None
    reference_count_checked: int | None = None
    direct_cf_raw_vshift_sum: float | None = None
    direct_cf_raw_vshift_mean: float | None = None
    direct_cf_raw_vshift_positive_fraction: float | None = None
    direct_cf_constrained_vshift_sum: float | None = None
    direct_cf_constrained_vshift_mean: float | None = None
    direct_cf_constrained_vshift_positive_fraction: float | None = None
    direct_cf_vshift_retention_ratio: float | None = None
    # Per-episode hierarchical Gate reporting; all are reporting-only and do
    # not redefine the existing WHAT Direct-CF metrics.
    gate_action1_count: int = 0
    gate_action0_count: int = 0
    gate_action1_ratio: float | None = None
    mean_gate_probability: float | None = None
    gate_cf_sample_count: int = 0
    gate_cf_delta_mean: float | None = None
    gate_cf_delta_abs_mean: float | None = None
    gate_cf_delta_positive_fraction: float | None = None
    gate_loss_mean: float | None = None
    gate_gradient_norm_mean: float | None = None
    gate_gradient_finite: bool | None = None
    gate_probability_shift_mean: float | None = None
    gate_probability_shift_max: float | None = None
    gate_optimizer_steps_this_episode: int = 0
    gate_optimizer_step_count: int = 0
    gate_policy_version_before: int | None = None
    gate_policy_version_after: int | None = None
    actor_policy_version_before: int | None = None
    actor_policy_version_after: int | None = None
    what_cf_coverage: float | None = None


@dataclass(frozen=True)
class FormalTrainingRunPlan:
    """正式训练run计划；execute=False时只返回该计划，不执行环境或优化器。"""

    num_episodes: int
    steps_per_episode: int
    planned_environment_steps: int
    update_epochs: int
    num_minibatches: int
    minibatch_size: int
    expected_optimizer_steps_per_episode: int
    planned_optimizer_steps: int
    planned_backward_calls: int
    optimizer_instances_planned: int
    execute: bool
    single_trace_training: bool
    no_default_long_training_budget: bool
    train_origin_start: int
    train_origin_end: int
    test_origin_start: int
    test_origin_end: int
    credit_assignment: str = ppo_update.LEGACY_CREDIT_ASSIGNMENT
    planned_q_optimizer_steps: int = 0
    planned_q_backward_calls: int = 0
    q_optimizer_instances_planned: int = 0
    credit_phase: str = "legacy"
    planned_actor_optimizer_steps: int = 0
    planned_critic_optimizer_steps: int = 0
    actor_optimizer_instances_planned: int = 1
    critic_optimizer_instances_planned: int = 0
    cf_samples_per_episode: int = 0
    planned_counterfactual_branch_steps: int = 0
    cf_sampling_strategy: str = "uniform"
    cf_batch_episodes: int = 1
    gate_cf_samples_per_episode: int = 0
    planned_gate_counterfactual_branch_steps: int = 0
    planned_gate_optimizer_steps: int = 0
    planned_gate_backward_calls: int = 0


@dataclass(frozen=True)
class FormalTrainingRunResult:
    """正式训练执行结果；本阶段不会在CLI或dry-run中创建。"""

    plan: FormalTrainingRunPlan
    episode_summaries: list[TrainingEpisodeSummary]
    final_policy_version: int
    optimizer_step_count: int
    # 本次调用新完成的episode仍放在episode_summaries；以下字段用于明确
    # resume后的全局状态，不保存RolloutBatch等大对象。
    completed_episodes: int = 0
    training_history: dict[str, Any] | None = None
    resumed: bool = False
    q_optimizer_step_count: int = 0
    gate_policy_version: int = 0
    gate_optimizer_step_count: int = 0


@dataclass(frozen=True)
class DeterministicEvaluationSummary:
    """held-out test deterministic evaluation 的区间级指标。"""

    num_steps: int
    evaluation_origin_start: int
    evaluation_origin_end: int
    evaluation_slot_start: int
    evaluation_slot_end: int
    cumulative_system_utility: float
    mean_system_utility: float
    cumulative_actual_task_utility: float
    cumulative_cache_cost: float
    cumulative_movement_cost: float
    cache_trigger_counts: dict[str, int]
    uav_movement_trigger_counts: dict[str, int]
    arrived_task_count: int
    cache_service_available_task_count: int
    cache_service_availability_rate: float | None
    cloud_execution_task_count: int
    cloud_execution_ratio: float | None
    execution_location_task_counts: dict[str, int]
    execution_location_ratios: dict[str, float]
    all_steps_feasible: bool
    reward_identity_error: float
    evaluation_semantics: str


def _is_counterfactual_mode(project_cfg: dict[str, Any]) -> bool:
    return ppo_update.credit_assignment_from_project(project_cfg) == (
        ppo_update.PER_TRIGGER_Q_CREDIT_ASSIGNMENT
    )


def _is_centralized_mode(project_cfg: dict[str, Any]) -> bool:
    return ppo_update.credit_assignment_from_project(project_cfg) == (
        ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT
    )


def _is_sparse_direct_cf_mode(project_cfg: dict[str, Any]) -> bool:
    return ppo_update.credit_assignment_from_project(project_cfg) == (
        ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT
    )


def _is_hierarchical_gate_mode(project_cfg: dict[str, Any]) -> bool:
    return ppo_update.credit_assignment_from_project(project_cfg) == (
        ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT
    )


def _is_frozen_policy_batch_mode(project_cfg: dict[str, Any]) -> bool:
    """Whether the explicit V3 frozen-policy multi-episode batch method is active."""

    return _is_sparse_direct_cf_mode(project_cfg) and str(
        project_config.get_config_value(
            project_cfg,
            "algorithm.ppo.method",
            "",
        )
    ) == ppo_update.FROZEN_POLICY_BATCH_METHOD


def _is_reference_drift_mode(project_cfg: dict[str, Any]) -> bool:
    """Whether the fixed-cap reference-state constrained batch method is active."""

    return _is_sparse_direct_cf_mode(project_cfg) and str(
        project_config.get_config_value(project_cfg, "algorithm.ppo.method", "")
    ) == ppo_update.REFERENCE_DRIFT_FROZEN_POLICY_BATCH_METHOD


def _is_what_reference_drift_mode(project_cfg: dict[str, Any]) -> bool:
    """WHAT Actor uses the existing RefDrift gate in hierarchical mode too."""

    return _is_reference_drift_mode(project_cfg) or _is_hierarchical_gate_mode(project_cfg)


def _is_frozen_batch_family(project_cfg: dict[str, Any]) -> bool:
    return _is_frozen_policy_batch_mode(project_cfg) or _is_reference_drift_mode(project_cfg)


def _configured_ppo_method_id(
    project_cfg: dict[str, Any],
    credit_assignment: str,
) -> str:
    """Resolve the method identity used by dry-run reporting.

    Older configs may omit ``algorithm.ppo.method`` and rely on the
    credit-assignment fallback.  Keep that fallback aligned with the actual
    dispatch instead of inventing a V3 method from ``cf_batch_episodes``.
    """

    configured = project_config.get_config_value(project_cfg, "algorithm.ppo.method")
    if configured is not None:
        return str(configured)
    return {
        ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT: hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_METHOD,
        ppo_update.PER_TRIGGER_Q_CREDIT_ASSIGNMENT: "proposed_per_trigger_q",
        ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT: "proposed_centralized_cf_q",
        ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT: "proposed_sparse_direct_cf_ppo",
    }.get(credit_assignment, "trigger_actor_critic")


def _dry_run_algorithm_reporting(
    project_cfg: dict[str, Any],
    credit_assignment: str,
) -> dict[str, Any]:
    """Return accurate method/optimizer/credit labels without changing dispatch."""

    method_id = _configured_ppo_method_id(project_cfg, credit_assignment)
    configured_actor_architecture = str(
        project_config.get_config_value(
            project_cfg,
            "algorithm.ppo.actor_architecture",
            ppo_trigger_policy.LEGACY_SHARED_HEAD_ARCHITECTURE_ID,
        )
    )

    if credit_assignment == ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT:
        # The top-level policy is hierarchical; its WHAT sub-policy is the
        # existing Full Shared Actor.  Report both identities explicitly.
        return {
            "method_id": hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_METHOD,
            "architecture_id": hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_ARCHITECTURE_ID,
            "actor_architecture_id": ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID,
            "actor_optimizer_mode": "independent_gate_and_what_optimizers",
            "actor_credit_semantics": (
                "WHAT Direct-CF A_cf=(a-p_old)*Delta_G; "
                "Gate PPO A_gate=(g-p_old_gate)*Delta_gate"
            ),
            "gate_optimizer_mode": "gate_first_then_what_direct_cf",
            "gate_credit_semantics": "A_gate=(g-p_old_gate)*Delta_gate",
            "what_credit_semantics": "A_cf=(a-p_old)*Delta_G",
            "learned_critic_for_actor_credit": False,
            "value_loss_in_actor_update": False,
            "direct_cf_minibatches_per_epoch": 1,
            "direct_cf_expected_optimizer_steps_per_batch": 1,
            "direct_cf_uses_scalar_gae_advantage": False,
        }

    if credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT:
        actor_architecture = (
            ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID
            if _is_frozen_batch_family(project_cfg)
            else configured_actor_architecture
        )
        return {
            "method_id": method_id,
            "architecture_id": actor_architecture,
            "actor_architecture_id": actor_architecture,
            "actor_optimizer_mode": (
                "actor_only_full_shared_encoder_autoregressive_shared_head_frozen_value"
                if actor_architecture == ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID
                else "actor_only_trigger_heads_encoder_frozen_autoregressive_value"
            ),
            "actor_credit_semantics": "sparse_direct_cf A_cf=(a-p_old)*Delta_G",
            "gate_optimizer_mode": None,
            "gate_credit_semantics": None,
            "what_credit_semantics": "A_cf=(a-p_old)*Delta_G",
            "learned_critic_for_actor_credit": False,
            "value_loss_in_actor_update": False,
            "direct_cf_minibatches_per_epoch": 1,
            "direct_cf_expected_optimizer_steps_per_batch": 1,
            "direct_cf_uses_scalar_gae_advantage": False,
        }

    if credit_assignment == ppo_update.PER_TRIGGER_Q_CREDIT_ASSIGNMENT:
        return {
            "method_id": method_id,
            "architecture_id": configured_actor_architecture,
            "actor_architecture_id": configured_actor_architecture,
            "actor_optimizer_mode": "actor_plus_independent_per_trigger_q_optimizer",
            "actor_credit_semantics": "per_trigger_q_conditional_ppo",
            "gate_optimizer_mode": None,
            "gate_credit_semantics": None,
            "what_credit_semantics": None,
            "learned_critic_for_actor_credit": True,
            "value_loss_in_actor_update": None,
            "direct_cf_minibatches_per_epoch": None,
            "direct_cf_expected_optimizer_steps_per_batch": None,
            "direct_cf_uses_scalar_gae_advantage": None,
        }

    if credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT:
        return {
            "method_id": method_id,
            "architecture_id": configured_actor_architecture,
            "actor_architecture_id": configured_actor_architecture,
            "actor_optimizer_mode": "actor_plus_independent_centralized_q_optimizer",
            "actor_credit_semantics": "centralized_counterfactual_q",
            "gate_optimizer_mode": None,
            "gate_credit_semantics": None,
            "what_credit_semantics": None,
            "learned_critic_for_actor_credit": True,
            "value_loss_in_actor_update": None,
            "direct_cf_minibatches_per_epoch": None,
            "direct_cf_expected_optimizer_steps_per_batch": None,
            "direct_cf_uses_scalar_gae_advantage": None,
        }

    return {
        "method_id": method_id,
        "architecture_id": configured_actor_architecture,
        "actor_architecture_id": configured_actor_architecture,
        "actor_optimizer_mode": "legacy_actor_critic_optimizer",
        "actor_credit_semantics": "legacy_scalar_gae",
        "gate_optimizer_mode": None,
        "gate_credit_semantics": None,
        "what_credit_semantics": None,
        "learned_critic_for_actor_credit": None,
        "value_loss_in_actor_update": True,
        "direct_cf_minibatches_per_epoch": None,
        "direct_cf_expected_optimizer_steps_per_batch": None,
        "direct_cf_uses_scalar_gae_advantage": None,
    }


def _cf_batch_episodes(project_cfg: dict[str, Any]) -> int:
    value = project_config.get_config_value(
        project_cfg,
        "algorithm.ppo.cf_batch_episodes",
        1,
    )
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError("algorithm.ppo.cf_batch_episodes 必须是正整数。")
    result = int(value)
    if (_is_frozen_batch_family(project_cfg) or _is_hierarchical_gate_mode(project_cfg)) and result != 4:
        raise ValueError(
            "frozen-policy batch / hierarchical Gate 当前固定要求cf_batch_episodes=4。"
        )
    return result


def _cf_parallel_workers(project_cfg: dict[str, Any]) -> int:
    value = project_config.get_config_value(
        project_cfg,
        "algorithm.ppo.cf_parallel_workers",
        8,
    )
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError("algorithm.ppo.cf_parallel_workers 必须是正整数。")
    return int(value)


def _cf_sampling_strategy(project_cfg: dict[str, Any]) -> str:
    """读取V3 Direct-CF采样策略；缺省保持旧uniform语义。"""

    strategy = str(
        project_config.get_config_value(
            project_cfg,
            "algorithm.ppo.cf_sampling_strategy",
            "uniform",
        )
    )
    if strategy not in {"uniform", "trigger_stratified"}:
        raise ValueError(
            "algorithm.ppo.cf_sampling_strategy 必须是 uniform 或 trigger_stratified。"
        )
    return strategy


def _is_q_mode(project_cfg: dict[str, Any]) -> bool:
    return _is_counterfactual_mode(project_cfg) or _is_centralized_mode(project_cfg)


def _cf_samples_per_episode(project_cfg: dict[str, Any]) -> int:
    default_value = HIERARCHICAL_GATE_CF_SAMPLES_PER_EPISODE if _is_hierarchical_gate_mode(project_cfg) else 0
    value = project_config.get_config_value(
        project_cfg,
        "algorithm.ppo.cf_samples_per_episode",
        default_value,
    )
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError("algorithm.ppo.cf_samples_per_episode 必须是正整数。")
    return int(value)


def _trigger_count_list(
    value: dict[str, int] | list[int] | tuple[int, ...] | None,
    name: str,
    labels: tuple[str, ...] | list[str] | None = None,
) -> list[int] | None:
    """规范化 Trigger count reporting，顺序由 dynamic action spec 或输入映射决定。"""

    if value is None:
        return None
    expected_labels = None if labels is None else tuple(str(label) for label in labels)
    if isinstance(value, dict):
        mapping_labels = tuple(str(label) for label in value)
        if expected_labels is not None and set(mapping_labels) != set(expected_labels):
            raise ValueError(f"{name} 必须覆盖当前 dynamic action spec 的全部 Trigger。")
        ordered_labels = expected_labels or mapping_labels
        result = [int(value[label]) for label in ordered_labels]
    else:
        result = [int(item) for item in value]
    if expected_labels is not None and len(result) != len(expected_labels):
        raise ValueError(f"{name} 长度必须与当前 dynamic action spec 一致。")
    if not result or any(item < 0 for item in result):
        raise ValueError(f"{name} 必须是非空非负整数列表。")
    return result


def _trigger_weight_list(
    value: list[float] | tuple[float, ...] | None,
    name: str,
    expected_count: int | None = None,
) -> list[float] | None:
    """规范化 dynamic Trigger sampling weights。"""

    if value is None:
        return None
    result = [float(item) for item in value]
    if (expected_count is not None and len(result) != int(expected_count)) or not result or any(
        not math.isfinite(item) or item < 0.0 for item in result
    ):
        raise ValueError(f"{name} 必须是与当前 Trigger 数量一致的 finite 非负列表。")
    return result


def _empty_training_history(
    credit_assignment: str = ppo_update.LEGACY_CREDIT_ASSIGNMENT,
    credit_phase: str | None = None,
    critic_return_scale: float | None = None,
    cf_sampling_strategy: str = "uniform",
    method_id: str | None = None,
    architecture_id: str | None = None,
    cf_batch_episodes: int = 1,
) -> dict[str, Any]:
    """创建与算法 mode 对应的轻量训练历史。"""

    if credit_assignment == ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT:
        if method_id != hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_METHOD:
            raise ValueError("hierarchical Gate training_history method_id不兼容。")
        if int(cf_batch_episodes) != HIERARCHICAL_GATE_CF_BATCH_EPISODES:
            raise ValueError("hierarchical Gate training_history必须使用cf_batch_episodes=4。")
        return {
            "schema_version": HIERARCHICAL_GATE_TRAINING_HISTORY_SCHEMA_VERSION,
            "algorithm_schema": hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_ALGORITHM_SCHEMA,
            "method_id": str(method_id),
            "architecture_id": str(
                architecture_id or ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID
            ),
            "credit_assignment": credit_assignment,
            "cf_sampling_strategy": str(cf_sampling_strategy),
            "gate_sampling_strategy": "uniform_slots",
            "cf_samples_per_episode": 0,
            "gate_cf_samples_per_episode": HIERARCHICAL_GATE_CF_SAMPLES_PER_EPISODE,
            "cf_batch_episodes": HIERARCHICAL_GATE_CF_BATCH_EPISODES,
            "checkpoint_save_semantics": "episode_atomic_with_pending_batch",
            "causal_artifact_schema": causal_artifacts.CAUSAL_ARTIFACT_SCHEMA_ID,
            "hierarchical_gate_artifact_schema": causal_artifacts.HIERARCHICAL_GATE_ARTIFACT_SCHEMA_ID,
            "causal_update_artifact_index": 0,
            "reference_probability_drift_cap": 0.005,
            "reference_constraint_method": reference_drift.REFERENCE_DRIFT_CONSTRAINT_METHOD,
            "reference_constraint_version": reference_drift.REFERENCE_DRIFT_CONSTRAINT_VERSION,
            "completed_episodes": 0,
            "policy_version": 0,
            "optimizer_step_count": 0,
            "actor_policy_version": 0,
            "actor_optimizer_step_count": 0,
            "gate_policy_version": 0,
            "gate_optimizer_step_count": 0,
            "batch_id": 0,
            "batch_episode_index": 0,
            "pending_cf_sample_count": 0,
            "pending_gate_cf_sample_count": 0,
            "pending_reference_state_count": 0,
            "pending_collection_policy_version": None,
            "episodes": [],
        }
    if (
        credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT
        and method_id == ppo_update.FROZEN_POLICY_BATCH_METHOD
    ):
        return {
            "schema_version": FROZEN_POLICY_BATCH_TRAINING_HISTORY_SCHEMA_VERSION,
            "algorithm_schema": "proposed_sparse_direct_cf_frozen_policy_batch_v1",
            "method_id": str(method_id),
            "architecture_id": str(architecture_id or ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID),
            "credit_assignment": credit_assignment,
            "cf_sampling_strategy": str(cf_sampling_strategy),
            "cf_samples_per_episode": 0,
            "cf_batch_episodes": int(cf_batch_episodes),
            "checkpoint_save_semantics": "batch_boundary_only",
            "completed_episodes": 0,
            "policy_version": 0,
            "optimizer_step_count": 0,
            "actor_optimizer_step_count": 0,
            "batch_id": 0,
            "batch_episode_index": 0,
            "pending_cf_sample_count": 0,
            "pending_collection_policy_version": None,
            "episodes": [],
        }
    if (
        credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT
        and method_id == ppo_update.REFERENCE_DRIFT_FROZEN_POLICY_BATCH_METHOD
    ):
        return {
            "schema_version": REFERENCE_DRIFT_TRAINING_HISTORY_SCHEMA_VERSION,
            "algorithm_schema": ppo_update.REFERENCE_DRIFT_ALGORITHM_SCHEMA,
            "method_id": str(method_id),
            "architecture_id": str(
                architecture_id or ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID
            ),
            "credit_assignment": credit_assignment,
            "cf_sampling_strategy": str(cf_sampling_strategy),
            "cf_samples_per_episode": 0,
            "cf_batch_episodes": int(cf_batch_episodes),
            "checkpoint_save_semantics": "episode_atomic_with_pending_batch",
            "causal_artifact_schema": causal_artifacts.CAUSAL_ARTIFACT_SCHEMA_ID,
            "causal_update_artifact_index": 0,
            "reference_probability_drift_cap": 0.005,
            "reference_constraint_method": reference_drift.REFERENCE_DRIFT_CONSTRAINT_METHOD,
            "reference_constraint_version": reference_drift.REFERENCE_DRIFT_CONSTRAINT_VERSION,
            "completed_episodes": 0,
            "policy_version": 0,
            "optimizer_step_count": 0,
            "actor_optimizer_step_count": 0,
            "batch_id": 0,
            "batch_episode_index": 0,
            "pending_cf_sample_count": 0,
            "pending_reference_state_count": 0,
            "pending_collection_policy_version": None,
            "episodes": [],
        }
    if credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT:
        return {
            "schema_version": SPARSE_DIRECT_CF_TRAINING_HISTORY_SCHEMA_VERSION,
            "algorithm_schema": "proposed_sparse_direct_cf_ppo_v3",
            "credit_assignment": credit_assignment,
            "cf_sampling_strategy": str(cf_sampling_strategy),
            "completed_episodes": 0,
            "policy_version": 0,
            "optimizer_step_count": 0,
            "actor_optimizer_step_count": 0,
            "episodes": [],
        }
    if credit_assignment == ppo_update.PER_TRIGGER_Q_CREDIT_ASSIGNMENT:
        return {
            "schema_version": COUNTERFACTUAL_TRAINING_HISTORY_SCHEMA_VERSION,
            "credit_assignment": credit_assignment,
            "q_target": ppo_update.MONTE_CARLO_Q_TARGET,
            "completed_episodes": 0,
            "policy_version": 0,
            "optimizer_step_count": 0,
            "q_optimizer_step_count": 0,
            "episodes": [],
        }
    if credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT:
        phase = "critic_warmup" if credit_phase is None else str(credit_phase)
        if phase not in {"critic_warmup", "joint_training"}:
            raise ValueError(f"未知 credit_phase：{phase}")
        if critic_return_scale is None:
            raise ValueError("centralized training_history 必须记录 critic_return_scale。")
        scale = float(critic_return_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("critic_return_scale 必须是有限正数。")
        return {
            "schema_version": CENTRALIZED_COUNTERFACTUAL_TRAINING_HISTORY_SCHEMA_VERSION,
            "credit_assignment": credit_assignment,
            "q_target": ppo_update.MONTE_CARLO_Q_TARGET,
            "credit_phase": phase,
            "critic_return_scale": scale,
            "completed_episodes": 0,
            "policy_version": 0,
            "optimizer_step_count": 0,
            "critic_update_count": 0,
            "critic_optimizer_step_count": 0,
            "episodes": [],
        }
    if credit_assignment != ppo_update.LEGACY_CREDIT_ASSIGNMENT:
        raise ValueError(f"未知 credit_assignment：{credit_assignment}")
    return {
        "schema_version": TRAINING_HISTORY_SCHEMA_VERSION,
        "completed_episodes": 0,
        "policy_version": 0,
        "optimizer_step_count": 0,
        "episodes": [],
    }


def _training_episode_history_record(
    episode_summary: TrainingEpisodeSummary,
    update_summary: FormalPPOUpdateSummary,
    credit_assignment: str = ppo_update.LEGACY_CREDIT_ASSIGNMENT,
) -> dict[str, Any]:
    """把一个完整episode的两个正式summary转换为轻量builtin记录。"""

    if int(episode_summary.episode_index) < 0:
        raise ValueError("TrainingEpisodeSummary.episode_index 不能为负。")
    if credit_assignment == ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT:
        return {
            "episode_index": int(episode_summary.episode_index),
            "training_episode_summary": summary_to_builtin(episode_summary),
            "ppo_update_summary": summary_to_builtin(update_summary),
        }
    if credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT:
        return {
            "episode_index": int(episode_summary.episode_index),
            "policy_version_before": int(episode_summary.policy_version_before),
            "policy_version_after": int(episode_summary.policy_version_after),
            "rollout_reward_sum": float(episode_summary.rollout_reward_sum),
            "rollout_reward_mean": float(episode_summary.rollout_reward_mean),
            "actual_q_sum": float(episode_summary.actual_q_sum),
            "cache_update_cost_sum": float(episode_summary.cache_cost_sum),
            "movement_cost_sum": float(episode_summary.movement_cost_sum),
            "cf_sample_count": int(episode_summary.cf_sample_count),
            "cf_sampling_strategy": str(episode_summary.cf_sampling_strategy),
            "cf_effective_sample_count": int(episode_summary.cf_effective_sample_count),
            "cf_trigger_free_counts": _trigger_count_list(
                episode_summary.cf_trigger_free_counts,
                "cf_trigger_free_counts",
            ),
            "cf_delta_mean": episode_summary.cf_delta_mean,
            "cf_delta_std": episode_summary.cf_delta_std,
            "cf_delta_abs_mean": episode_summary.cf_delta_abs_mean,
            "cf_delta_positive_fraction": episode_summary.cf_delta_positive_fraction,
            "cf_advantage_mean": episode_summary.cf_advantage_mean,
            "cf_advantage_std": episode_summary.cf_advantage_std,
            "cf_advantage_abs_mean": episode_summary.cf_advantage_abs_mean,
            "actor_loss_mean": float(episode_summary.policy_loss_mean),
            "actor_gradient_norm_mean": float(episode_summary.gradient_norm_mean),
            "pre_clip_gradient_norm": episode_summary.pre_clip_gradient_norm,
            "actor_gradient_finite": bool(episode_summary.actor_gradient_finite),
            "optimizer_steps": int(episode_summary.optimizer_steps),
            "actor_updated_this_episode": bool(episode_summary.actor_updated_this_episode),
            "actor_optimizer_steps_this_episode": int(episode_summary.actor_optimizer_steps_this_episode),
            "cf_batch_id": episode_summary.cf_batch_id,
            "cf_batch_episode_index": episode_summary.cf_batch_episode_index,
            "cf_batch_pending_sample_count": int(episode_summary.cf_batch_pending_sample_count),
            "collection_policy_version": episode_summary.collection_policy_version,
            "actor_batch_sample_count": episode_summary.actor_batch_sample_count,
            "cf_batch_vshift_sum": episode_summary.cf_batch_vshift_sum,
            "cf_batch_vshift_mean": episode_summary.cf_batch_vshift_mean,
            "cf_batch_vshift_positive_fraction": episode_summary.cf_batch_vshift_positive_fraction,
            "reference_raw_drift_mean": episode_summary.reference_raw_drift_mean,
            "reference_raw_drift_median": episode_summary.reference_raw_drift_median,
            "reference_raw_drift_p95": episode_summary.reference_raw_drift_p95,
            "reference_raw_drift_p99": episode_summary.reference_raw_drift_p99,
            "reference_raw_drift_max": episode_summary.reference_raw_drift_max,
            "reference_constrained_drift_mean": episode_summary.reference_constrained_drift_mean,
            "reference_constrained_drift_median": episode_summary.reference_constrained_drift_median,
            "reference_constrained_drift_p95": episode_summary.reference_constrained_drift_p95,
            "reference_constrained_drift_p99": episode_summary.reference_constrained_drift_p99,
            "reference_constrained_drift_max": episode_summary.reference_constrained_drift_max,
            "selected_drift_mean": episode_summary.selected_drift_mean,
            "selected_drift_p95": episode_summary.selected_drift_p95,
            "selected_drift_max": episode_summary.selected_drift_max,
            "reference_drift_alpha": episode_summary.reference_drift_alpha,
            "reference_drift_constraint_active": episode_summary.reference_drift_constraint_active,
            "reference_probability_drift_cap": episode_summary.reference_probability_drift_cap,
            "reference_state_count": episode_summary.reference_state_count,
            "reference_unselected_count": episode_summary.reference_unselected_count,
            "reference_selected_count": episode_summary.reference_selected_count,
            "reference_selected_intersection_count": episode_summary.reference_selected_intersection_count,
            "reference_drift_by_trigger": episode_summary.reference_drift_by_trigger,
            "old_p_reconstruction_max_error": episode_summary.old_p_reconstruction_max_error,
            "old_p_reconstruction_checked": episode_summary.old_p_reconstruction_checked,
            "old_p_reconstruction_tolerance": episode_summary.old_p_reconstruction_tolerance,
            "probability_dtype": episode_summary.probability_dtype,
            "reference_count_checked": episode_summary.reference_count_checked,
            "direct_cf_raw_vshift_sum": episode_summary.direct_cf_raw_vshift_sum,
            "direct_cf_raw_vshift_mean": episode_summary.direct_cf_raw_vshift_mean,
            "direct_cf_raw_vshift_positive_fraction": episode_summary.direct_cf_raw_vshift_positive_fraction,
            "direct_cf_constrained_vshift_sum": episode_summary.direct_cf_constrained_vshift_sum,
            "direct_cf_constrained_vshift_mean": episode_summary.direct_cf_constrained_vshift_mean,
            "direct_cf_constrained_vshift_positive_fraction": episode_summary.direct_cf_constrained_vshift_positive_fraction,
            "direct_cf_vshift_retention_ratio": episode_summary.direct_cf_vshift_retention_ratio,
            "approx_kl_mean": float(episode_summary.approx_kl_mean),
            "clip_fraction_mean": float(episode_summary.clip_fraction_mean),
            "policy_entropy": episode_summary.policy_entropy,
            "probability_drift_mean_abs": episode_summary.probability_drift_mean_abs,
            "probability_drift_p95": episode_summary.probability_drift_p95,
            "probability_drift_p99": episode_summary.probability_drift_p99,
            "probability_drift_max": episode_summary.probability_drift_max,
            # 正式history字段名；保留上面的短名以兼容本轮已写入的报告。
            "policy_probability_change_mean_abs": episode_summary.probability_drift_mean_abs,
            "policy_probability_change_p95": episode_summary.probability_drift_p95,
            "policy_probability_change_p99": episode_summary.probability_drift_p99,
            "policy_probability_change_max": episode_summary.probability_drift_max,
            "encoder_parameter_delta_norm": episode_summary.encoder_parameter_delta_norm,
            "trigger_heads_parameter_delta_norm": episode_summary.trigger_heads_parameter_delta_norm,
            "autoregressive_parameter_delta_norm": episode_summary.autoregressive_parameter_delta_norm,
            "value_head_parameter_delta_norm": episode_summary.value_head_parameter_delta_norm,
            "logit_head_parameter_delta_norm": episode_summary.logit_head_parameter_delta_norm,
            "cf_trigger_sample_counts": _trigger_count_list(
                episode_summary.cf_trigger_sample_counts,
                "cf_trigger_sample_counts",
            ),
            "trigger_action1_counts": _trigger_count_list(
                episode_summary.trigger_action1_counts,
                "trigger_action1_counts",
            ),
            "cf_trigger_sampling_weights": _trigger_weight_list(
                episode_summary.cf_trigger_sampling_weights,
                "cf_trigger_sampling_weights",
            ),
            "direct_cf_local_objective_change_proxy": episode_summary.direct_cf_local_objective_change_proxy,
            "branch_environment_steps": int(episode_summary.branch_environment_steps),
        }
    episode_builtin = summary_to_builtin(episode_summary)
    for key in (
        "cf_sample_count",
        "cf_delta_mean",
        "cf_delta_std",
        "cf_delta_abs_mean",
        "cf_delta_positive_fraction",
        "cf_advantage_mean",
        "cf_advantage_std",
        "cf_advantage_abs_mean",
        "direct_cf_local_objective_change_proxy",
        "branch_environment_steps",
        "actor_gradient_finite",
    ):
        episode_builtin.pop(key, None)
    update_builtin = summary_to_builtin(update_summary)
    for key in (
        "cf_sample_count",
        "cf_delta_mean",
        "cf_delta_std",
        "cf_delta_abs_mean",
        "cf_delta_positive_fraction",
        "cf_advantage_mean",
        "cf_advantage_std",
        "cf_advantage_abs_mean",
        "direct_cf_local_objective_change_proxy",
        "branch_environment_steps",
        "actor_gradient_finite",
    ):
        update_builtin.pop(key, None)
    if credit_assignment == ppo_update.LEGACY_CREDIT_ASSIGNMENT:
        for key in (
            "q_loss_mean",
            "q_gradient_norm_mean",
            "q_gradient_finite",
            "counterfactual_advantage_mean",
            "counterfactual_advantage_std",
            "counterfactual_advantage_abs_mean",
            "free_trigger_decision_count",
            "q_optimizer_steps",
        ):
            update_builtin.pop(key, None)
    elif credit_assignment not in {
        ppo_update.PER_TRIGGER_Q_CREDIT_ASSIGNMENT,
        ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT,
    }:
        raise ValueError(f"未知 credit_assignment：{credit_assignment}")
    return {
        "episode_index": int(episode_summary.episode_index),
        "training_episode_summary": episode_builtin,
        "ppo_update_summary": update_builtin,
    }


def _direct_cf_trigger_sample_counts(
    direct_cf_samples: list[ppo_rollout.DirectCFCreditSample],
    trigger_labels: tuple[str, ...] | list[str],
) -> dict[str, int]:
    """按 dynamic action spec 顺序统计 Direct-CF 样本。"""

    labels = tuple(str(label) for label in trigger_labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Direct-CF trigger_labels 必须非空且不重复。")
    counts = {label: 0 for label in labels}
    for sample in direct_cf_samples:
        label = str(sample.trigger_label)
        if label not in counts:
            raise RuntimeError(f"Direct-CF样本包含未知Trigger：{label}")
        counts[label] += 1
    return counts


def _build_direct_cf_reservoir(
    action_spec: ppo_trigger_policy.TriggerActionSpec,
    capacity: int,
    sampling_rng: np.random.Generator,
    sampling_strategy: str,
) -> ppo_rollout.DirectCFReservoir:
    """按配置选择V3 reservoir；两条路径共享同一snapshot/branch接口。"""

    if sampling_strategy == "trigger_stratified":
        if int(capacity) != int(action_spec.action_dim):
            raise ValueError(
                "trigger_stratified 当前要求cf_samples_per_episode等于 dynamic Trigger 数量。"
            )
        return ppo_rollout.TriggerStratifiedDirectCFReservoir(
            tuple(action_spec.labels),
            sampling_rng,
        )
    if sampling_strategy == "uniform":
        return ppo_rollout.DirectCFReservoir(
            capacity,
            sampling_rng,
            trigger_count=int(action_spec.action_dim),
            trigger_labels=tuple(action_spec.labels),
        )
    raise ValueError(f"未知Direct-CF sampling strategy：{sampling_strategy}")


def _direct_cf_reservoir_reporting(
    reservoir: ppo_rollout.DirectCFReservoir,
) -> tuple[int, list[int], list[int], list[float]]:
    """返回 reservoir action-spec 顺序的 K_eff、N_i、sample count 和 N_i/N。"""

    expected_count = int(reservoir.trigger_count)
    if int(reservoir.trigger_count) != expected_count:
        raise ValueError("Direct-CF reservoir trigger数量必须为正整数。")
    free_counts = [int(value) for value in reservoir.trigger_free_counts]
    sample_counts = [int(value) for value in reservoir.trigger_sample_counts]
    weights = [float(value) for value in reservoir.trigger_sampling_weights]
    if len(free_counts) != expected_count or len(sample_counts) != expected_count or len(weights) != expected_count:
        raise ValueError("Direct-CF reservoir reporting长度必须与 dynamic action spec 一致。")
    if int(reservoir.total_seen) != sum(free_counts):
        raise AssertionError("Direct-CF reservoir total_seen与Trigger free count不一致。")
    if not math.isclose(math.fsum(weights), 1.0, rel_tol=1e-12, abs_tol=1e-12) and reservoir.total_seen > 0:
        raise AssertionError("Direct-CF Trigger sampling weights之和必须为1。")
    if sum(sample_counts) != int(reservoir.sample_count):
        raise AssertionError("Direct-CF Trigger sample count与reservoir不一致。")
    return int(reservoir.sample_count), free_counts, sample_counts, weights


def _build_training_history(
    records: list[dict[str, Any]],
    policy_version: int,
    optimizer_step_count: int,
    credit_assignment: str = ppo_update.LEGACY_CREDIT_ASSIGNMENT,
    q_optimizer_step_count: int = 0,
    credit_phase: str | None = None,
    critic_optimizer_step_count: int | None = None,
    critic_return_scale: float | None = None,
    cf_sampling_strategy: str = "uniform",
    method_id: str | None = None,
    architecture_id: str | None = None,
    cf_batch_episodes: int = 1,
    cf_samples_per_episode: int = 0,
    batch_id: int = 0,
    batch_episode_index: int = 0,
    pending_cf_sample_count: int = 0,
    pending_collection_policy_version: int | None = None,
    reference_probability_drift_cap: float | None = None,
    pending_reference_state_count: int = 0,
    gate_policy_version: int | None = None,
    gate_optimizer_step_count: int = 0,
    pending_gate_cf_sample_count: int = 0,
) -> dict[str, Any]:
    """构造checkpoint和JSON共用的canonical training history。"""

    if credit_assignment == ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT:
        if method_id != hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_METHOD:
            raise ValueError("hierarchical Gate training_history method_id不兼容。")
        if int(cf_batch_episodes) != HIERARCHICAL_GATE_CF_BATCH_EPISODES:
            raise ValueError("hierarchical Gate training_history必须使用cf_batch_episodes=4。")
        if int(pending_cf_sample_count) < 0 or int(pending_gate_cf_sample_count) < 0:
            raise ValueError("hierarchical Gate pending counts不能为负。")
        cap = 0.005 if reference_probability_drift_cap is None else float(reference_probability_drift_cap)
        if cap != 0.005:
            raise ValueError("hierarchical Gate training_history固定要求reference drift cap=0.005。")
        gate_version = int(policy_version if gate_policy_version is None else gate_policy_version)
        if gate_version < 0 or int(gate_optimizer_step_count) < 0:
            raise ValueError("hierarchical Gate policy/optimizer counters不能为负。")
        if gate_version != int(policy_version):
            raise ValueError("hierarchical Gate 当前要求Gate/WHAT policy version同步。")
        history = {
            "schema_version": HIERARCHICAL_GATE_TRAINING_HISTORY_SCHEMA_VERSION,
            "algorithm_schema": hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_ALGORITHM_SCHEMA,
            "method_id": str(method_id),
            "architecture_id": str(
                architecture_id or ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID
            ),
            "credit_assignment": credit_assignment,
            "cf_sampling_strategy": str(cf_sampling_strategy),
            "gate_sampling_strategy": "uniform_slots",
            "cf_samples_per_episode": int(cf_samples_per_episode),
            "gate_cf_samples_per_episode": HIERARCHICAL_GATE_CF_SAMPLES_PER_EPISODE,
            "cf_batch_episodes": HIERARCHICAL_GATE_CF_BATCH_EPISODES,
            "batch_sample_count": int(cf_samples_per_episode) * HIERARCHICAL_GATE_CF_BATCH_EPISODES,
            "gate_batch_sample_count": HIERARCHICAL_GATE_CF_SAMPLES_PER_EPISODE * HIERARCHICAL_GATE_CF_BATCH_EPISODES,
            "checkpoint_save_semantics": "episode_atomic_with_pending_batch",
            "causal_artifact_schema": causal_artifacts.CAUSAL_ARTIFACT_SCHEMA_ID,
            "hierarchical_gate_artifact_schema": causal_artifacts.HIERARCHICAL_GATE_ARTIFACT_SCHEMA_ID,
            "causal_update_artifact_index": int(optimizer_step_count),
            "reference_probability_drift_cap": cap,
            "reference_constraint_method": reference_drift.REFERENCE_DRIFT_CONSTRAINT_METHOD,
            "reference_constraint_version": reference_drift.REFERENCE_DRIFT_CONSTRAINT_VERSION,
            "completed_episodes": int(len(records)),
            "policy_version": int(policy_version),
            "optimizer_step_count": int(optimizer_step_count),
            "actor_policy_version": int(policy_version),
            "actor_optimizer_step_count": int(optimizer_step_count),
            "gate_policy_version": gate_version,
            "gate_optimizer_step_count": int(gate_optimizer_step_count),
            "batch_id": int(batch_id),
            "batch_episode_index": int(batch_episode_index),
            "pending_cf_sample_count": int(pending_cf_sample_count),
            "pending_gate_cf_sample_count": int(pending_gate_cf_sample_count),
            "pending_reference_state_count": int(pending_reference_state_count),
            "pending_collection_policy_version": pending_collection_policy_version,
            "episodes": records,
        }
    elif (
        credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT
        and method_id == ppo_update.FROZEN_POLICY_BATCH_METHOD
    ):
        if int(cf_batch_episodes) != 4:
            raise ValueError("frozen_policy_batch training_history 必须使用cf_batch_episodes=4。")
        if int(pending_cf_sample_count) != 0:
            raise ValueError(
                "frozen_policy_batch checkpoint/history只允许在batch boundary持久化，pending必须为0。"
            )
        history = {
            "schema_version": FROZEN_POLICY_BATCH_TRAINING_HISTORY_SCHEMA_VERSION,
            "algorithm_schema": "proposed_sparse_direct_cf_frozen_policy_batch_v1",
            "method_id": str(method_id),
            "architecture_id": str(
                architecture_id or ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID
            ),
            "credit_assignment": credit_assignment,
            "cf_sampling_strategy": str(cf_sampling_strategy),
            "cf_samples_per_episode": int(cf_samples_per_episode),
            "cf_batch_episodes": int(cf_batch_episodes),
            "checkpoint_save_semantics": "batch_boundary_only",
            "completed_episodes": int(len(records)),
            "policy_version": int(policy_version),
            "optimizer_step_count": int(optimizer_step_count),
            "actor_optimizer_step_count": int(optimizer_step_count),
            "batch_id": int(batch_id),
            "batch_episode_index": int(batch_episode_index),
            "pending_cf_sample_count": int(pending_cf_sample_count),
            "pending_collection_policy_version": pending_collection_policy_version,
            "reporting_metadata": {
                "policy_entropy": "persisted in each episode record",
                "probability_drift": "batch-boundary update reporting; collection episodes are null",
                "policy_probability_change": "batch-boundary update reporting; collection episodes are null",
            },
            "episodes": records,
        }
    elif (
        credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT
        and method_id == ppo_update.REFERENCE_DRIFT_FROZEN_POLICY_BATCH_METHOD
    ):
        if int(cf_batch_episodes) != 4:
            raise ValueError("refdrift training_history 必须使用cf_batch_episodes=4。")
        cap = (
            0.005
            if reference_probability_drift_cap is None
            else float(reference_probability_drift_cap)
        )
        if cap != 0.005:
            raise ValueError("refdrift training_history固定要求reference drift cap=0.005。")
        if int(pending_cf_sample_count) < 0 or int(pending_reference_state_count) < 0:
            raise ValueError("refdrift pending counts不能为负。")
        history = {
            "schema_version": REFERENCE_DRIFT_TRAINING_HISTORY_SCHEMA_VERSION,
            "algorithm_schema": ppo_update.REFERENCE_DRIFT_ALGORITHM_SCHEMA,
            "method_id": str(method_id),
            "architecture_id": str(
                architecture_id or ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID
            ),
            "credit_assignment": credit_assignment,
            "cf_sampling_strategy": str(cf_sampling_strategy),
            "cf_samples_per_episode": int(cf_samples_per_episode),
            "cf_batch_episodes": int(cf_batch_episodes),
            "checkpoint_save_semantics": "episode_atomic_with_pending_batch",
            "causal_artifact_schema": causal_artifacts.CAUSAL_ARTIFACT_SCHEMA_ID,
            # batch_id identifies the currently open collection batch; the
            # artifact index counts only durable completed updates so resume
            # from an episode inside a pending batch remains valid.
            "causal_update_artifact_index": int(optimizer_step_count),
            "reference_probability_drift_cap": cap,
            "reference_constraint_method": reference_drift.REFERENCE_DRIFT_CONSTRAINT_METHOD,
            "reference_constraint_version": reference_drift.REFERENCE_DRIFT_CONSTRAINT_VERSION,
            "completed_episodes": int(len(records)),
            "policy_version": int(policy_version),
            "optimizer_step_count": int(optimizer_step_count),
            "actor_optimizer_step_count": int(optimizer_step_count),
            "batch_id": int(batch_id),
            "batch_episode_index": int(batch_episode_index),
            "pending_cf_sample_count": int(pending_cf_sample_count),
            "pending_reference_state_count": int(pending_reference_state_count),
            "pending_collection_policy_version": pending_collection_policy_version,
            "episodes": records,
        }
    elif credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT:
        history = {
            "schema_version": SPARSE_DIRECT_CF_TRAINING_HISTORY_SCHEMA_VERSION,
            "algorithm_schema": "proposed_sparse_direct_cf_ppo_v3",
            "credit_assignment": credit_assignment,
            "cf_sampling_strategy": str(cf_sampling_strategy),
            "completed_episodes": int(len(records)),
            "policy_version": int(policy_version),
            "optimizer_step_count": int(optimizer_step_count),
            "actor_optimizer_step_count": int(optimizer_step_count),
            "reporting_metadata": {
                "policy_entropy": "Episode records written before the reporting persistence fix may be null: not_persisted_in_original_episode.",
                "probability_drift": "Episode records written before the reporting persistence fix may be null: not_persisted_in_original_episode.",
                "policy_probability_change": "Episode records written before the reporting persistence fix may be null: not_persisted_in_original_episode.",
            },
            "episodes": records,
        }
    elif credit_assignment == ppo_update.PER_TRIGGER_Q_CREDIT_ASSIGNMENT:
        history = {
            "schema_version": COUNTERFACTUAL_TRAINING_HISTORY_SCHEMA_VERSION,
            "credit_assignment": credit_assignment,
            "q_target": ppo_update.MONTE_CARLO_Q_TARGET,
            "completed_episodes": int(len(records)),
            "policy_version": int(policy_version),
            "optimizer_step_count": int(optimizer_step_count),
            "q_optimizer_step_count": int(q_optimizer_step_count),
            "episodes": records,
        }
    elif credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT:
        phase = "critic_warmup" if credit_phase is None else str(credit_phase)
        if phase not in {"critic_warmup", "joint_training"}:
            raise ValueError(f"未知 credit_phase：{phase}")
        if critic_return_scale is None:
            raise ValueError("centralized training_history 必须记录 critic_return_scale。")
        scale = float(critic_return_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("critic_return_scale 必须是有限正数。")
        critic_steps = (
            int(q_optimizer_step_count)
            if critic_optimizer_step_count is None
            else int(critic_optimizer_step_count)
        )
        history = {
            "schema_version": CENTRALIZED_COUNTERFACTUAL_TRAINING_HISTORY_SCHEMA_VERSION,
            "credit_assignment": credit_assignment,
            "q_target": ppo_update.MONTE_CARLO_Q_TARGET,
            "credit_phase": phase,
            "critic_return_scale": scale,
            "completed_episodes": int(len(records)),
            "policy_version": int(policy_version),
            "optimizer_step_count": int(optimizer_step_count),
            "critic_update_count": int(len(records)),
            "critic_optimizer_step_count": critic_steps,
            "episodes": records,
        }
    elif credit_assignment == ppo_update.LEGACY_CREDIT_ASSIGNMENT:
        history = {
            "schema_version": TRAINING_HISTORY_SCHEMA_VERSION,
            "completed_episodes": int(len(records)),
            "policy_version": int(policy_version),
            "optimizer_step_count": int(optimizer_step_count),
            "episodes": records,
        }
    else:
        raise ValueError(f"未知 credit_assignment：{credit_assignment}")
    return summary_to_builtin(history)


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须是object/dict。")
    return value


def _training_episode_summary_from_builtin(value: Any) -> TrainingEpisodeSummary:
    """从JSON/checkpoint builtin重建TrainingEpisodeSummary。"""

    item = _as_mapping(value, "training_episode_summary")
    integer_fields = {
        "episode_index",
        "policy_version_before",
        "policy_version_after",
        "optimizer_steps",
        "collection_policy_version",
        "reference_count_checked",
        "gate_action1_count",
        "gate_action0_count",
        "gate_cf_sample_count",
        "gate_optimizer_steps_this_episode",
        "gate_optimizer_step_count",
        "gate_policy_version_before",
        "gate_policy_version_after",
        "actor_policy_version_before",
        "actor_policy_version_after",
    }
    kwargs: dict[str, Any] = {}
    bool_fields = {
        "critic_gradient_finite",
        "actor_update_executed",
        "actor_gradient_finite",
        "old_p_reconstruction_checked",
        "gate_gradient_finite",
    }
    string_fields = {"credit_phase", "probability_dtype", "cf_sampling_strategy"}
    count_fields = {
        "cf_trigger_sample_counts",
        "trigger_action1_counts",
        "cf_trigger_free_counts",
    }
    weight_fields = {"cf_trigger_sampling_weights"}
    mapping_fields = {"reference_drift_by_trigger"}
    for field in fields(TrainingEpisodeSummary):
        if field.name not in item:
            if field.default is not MISSING:
                kwargs[field.name] = field.default
                continue
            raise ValueError(f"training_episode_summary 缺少字段：{field.name}")
        raw = item[field.name]
        if field.name in integer_fields:
            kwargs[field.name] = None if raw is None else int(raw)
        elif field.name in bool_fields:
            kwargs[field.name] = None if raw is None else bool(raw)
        elif field.name in string_fields:
            kwargs[field.name] = None if raw is None else str(raw)
        elif field.name in count_fields:
            kwargs[field.name] = (
                None
                if raw is None
                else _trigger_count_list(raw, f"training_episode_summary.{field.name}")
            )
        elif field.name in weight_fields:
            if raw is None:
                kwargs[field.name] = None
            else:
                values = [float(item) for item in raw]
                if not values or not all(
                    math.isfinite(item) and item >= 0.0 for item in values
                ):
                    raise ValueError(
                        "training_episode_summary.cf_trigger_sampling_weights必须是非空finite非负列表。"
                    )
                kwargs[field.name] = values
        elif field.name in mapping_fields:
            kwargs[field.name] = None if raw is None else dict(raw)
        else:
            kwargs[field.name] = None if raw is None else float(raw)
    return TrainingEpisodeSummary(**kwargs)


def _ppo_epoch_summary_from_builtin(value: Any) -> PPOEpochSummary:
    item = _as_mapping(value, "ppo_epoch_summary")
    integer_fields = {"epoch_index", "minibatch_count", "transition_count", "optimizer_steps_cumulative"}
    kwargs: dict[str, Any] = {}
    for field in fields(PPOEpochSummary):
        if field.name not in item:
            raise ValueError(f"ppo_epoch_summary 缺少字段：{field.name}")
        raw = item[field.name]
        kwargs[field.name] = int(raw) if field.name in integer_fields else float(raw)
    return PPOEpochSummary(**kwargs)


def _formal_update_summary_from_builtin(value: Any) -> FormalPPOUpdateSummary:
    """从JSON/checkpoint builtin重建FormalPPOUpdateSummary。"""

    item = _as_mapping(value, "ppo_update_summary")
    required = {
        field.name
        for field in fields(FormalPPOUpdateSummary)
        if field.name not in {
            "epoch_summaries",
            "q_loss_mean",
            "q_gradient_norm_mean",
            "q_gradient_finite",
            "counterfactual_advantage_mean",
            "counterfactual_advantage_std",
            "counterfactual_advantage_abs_mean",
            "free_trigger_decision_count",
            "q_optimizer_steps",
            "credit_phase",
            "critic_loss_before",
            "critic_loss_after",
            "critic_gradient_norm_mean",
            "critic_gradient_finite",
            "critic_action_gap_abs_mean",
            "critic_action_gap_std",
            "critic_action_gap_nonzero_fraction",
            "actor_update_executed",
            "critic_replay_sample_count",
            "policy_entropy",
            "probability_drift_mean_abs",
            "probability_drift_p95",
            "probability_drift_p99",
            "probability_drift_max",
            "encoder_parameter_delta_norm",
            "trigger_heads_parameter_delta_norm",
            "autoregressive_parameter_delta_norm",
            "value_head_parameter_delta_norm",
            "logit_head_parameter_delta_norm",
             "actor_batch_sample_count",
             "cf_batch_vshift_sum",
             "cf_batch_vshift_mean",
             "cf_batch_vshift_positive_fraction",
             "reference_raw_drift_mean",
             "reference_raw_drift_median",
             "reference_raw_drift_p95",
             "reference_raw_drift_p99",
             "reference_raw_drift_max",
             "reference_constrained_drift_mean",
             "reference_constrained_drift_median",
             "reference_constrained_drift_p95",
             "reference_constrained_drift_p99",
             "reference_constrained_drift_max",
             "selected_drift_mean",
             "selected_drift_p95",
             "selected_drift_max",
             "reference_drift_alpha",
             "reference_drift_constraint_active",
             "reference_probability_drift_cap",
             "reference_state_count",
             "reference_unselected_count",
              "reference_selected_count",
              "reference_selected_intersection_count",
              "reference_drift_by_trigger",
              "old_p_reconstruction_max_error",
              "old_p_reconstruction_checked",
              "old_p_reconstruction_tolerance",
              "probability_dtype",
              "reference_count_checked",
              "collection_policy_version",
              "direct_cf_raw_vshift_sum",
             "direct_cf_raw_vshift_mean",
             "direct_cf_raw_vshift_positive_fraction",
             "direct_cf_constrained_vshift_sum",
             "direct_cf_constrained_vshift_mean",
             "direct_cf_constrained_vshift_positive_fraction",
             "direct_cf_vshift_retention_ratio",
             "gate_optimizer_steps",
             "gate_loss_mean",
             "gate_gradient_norm_mean",
             "gate_gradient_finite",
             "gate_probability_shift_mean",
             "gate_probability_shift_max",
             "gate_delta_abs_mean",
             "gate_delta_positive_fraction",
             "gate_sample_count",
             "gate_branch_environment_steps",
             "gate_parameter_delta_max",
             "gate_subpolicy_parameter_delta_max",
             "gate_collection_policy_version",
         }
    }
    missing = sorted(required - set(item))
    if missing:
        raise ValueError(f"ppo_update_summary 缺少字段：{missing}")
    epoch_values = item.get("epoch_summaries")
    if not isinstance(epoch_values, list):
        raise ValueError("ppo_update_summary.epoch_summaries 必须是数组。")
    return FormalPPOUpdateSummary(
        update_epochs=int(item["update_epochs"]),
        num_minibatches=int(item["num_minibatches"]),
        minibatch_size=int(item["minibatch_size"]),
        optimizer_steps=int(item["optimizer_steps"]),
        policy_loss_mean=float(item["policy_loss_mean"]),
        value_loss_mean=float(item["value_loss_mean"]),
        entropy_mean=float(item["entropy_mean"]),
        approx_kl_mean=float(item["approx_kl_mean"]),
        clip_fraction_mean=float(item["clip_fraction_mean"]),
        gradient_norm_mean=float(item["gradient_norm_mean"]),
        gradient_finite=bool(item["gradient_finite"]),
        old_log_probs_frozen=bool(item["old_log_probs_frozen"]),
        old_values_frozen=bool(item["old_values_frozen"]),
        advantages_frozen=bool(item["advantages_frozen"]),
        returns_frozen=bool(item["returns_frozen"]),
        each_transition_once_per_epoch=bool(item["each_transition_once_per_epoch"]),
        epoch_summaries=tuple(_ppo_epoch_summary_from_builtin(entry) for entry in epoch_values),
        q_loss_mean=None if item.get("q_loss_mean") is None else float(item["q_loss_mean"]),
        q_gradient_norm_mean=(
            None if item.get("q_gradient_norm_mean") is None else float(item["q_gradient_norm_mean"])
        ),
        q_gradient_finite=(
            None if item.get("q_gradient_finite") is None else bool(item["q_gradient_finite"])
        ),
        counterfactual_advantage_mean=(
            None
            if item.get("counterfactual_advantage_mean") is None
            else float(item["counterfactual_advantage_mean"])
        ),
        counterfactual_advantage_std=(
            None
            if item.get("counterfactual_advantage_std") is None
            else float(item["counterfactual_advantage_std"])
        ),
        counterfactual_advantage_abs_mean=(
            None
            if item.get("counterfactual_advantage_abs_mean") is None
            else float(item["counterfactual_advantage_abs_mean"])
        ),
        free_trigger_decision_count=(
            None
            if item.get("free_trigger_decision_count") is None
            else int(item["free_trigger_decision_count"])
        ),
        q_optimizer_steps=int(item.get("q_optimizer_steps", 0)),
        credit_phase=str(item.get("credit_phase", "legacy")),
        critic_loss_before=(
            None if item.get("critic_loss_before") is None else float(item["critic_loss_before"])
        ),
        critic_loss_after=(
            None if item.get("critic_loss_after") is None else float(item["critic_loss_after"])
        ),
        critic_gradient_norm_mean=(
            None
            if item.get("critic_gradient_norm_mean") is None
            else float(item["critic_gradient_norm_mean"])
        ),
        critic_gradient_finite=(
            None
            if item.get("critic_gradient_finite") is None
            else bool(item["critic_gradient_finite"])
        ),
        critic_action_gap_abs_mean=(
            None
            if item.get("critic_action_gap_abs_mean") is None
            else float(item["critic_action_gap_abs_mean"])
        ),
        critic_action_gap_std=(
            None
            if item.get("critic_action_gap_std") is None
            else float(item["critic_action_gap_std"])
        ),
        critic_action_gap_nonzero_fraction=(
            None
            if item.get("critic_action_gap_nonzero_fraction") is None
            else float(item["critic_action_gap_nonzero_fraction"])
        ),
        actor_update_executed=bool(item.get("actor_update_executed", True)),
        critic_replay_sample_count=int(item.get("critic_replay_sample_count", 0)),
        encoder_parameter_delta_norm=(
            None
            if item.get("encoder_parameter_delta_norm") is None
            else float(item["encoder_parameter_delta_norm"])
        ),
        trigger_heads_parameter_delta_norm=(
            None
            if item.get("trigger_heads_parameter_delta_norm") is None
            else float(item["trigger_heads_parameter_delta_norm"])
        ),
        autoregressive_parameter_delta_norm=(
            None
            if item.get("autoregressive_parameter_delta_norm") is None
            else float(item["autoregressive_parameter_delta_norm"])
        ),
        value_head_parameter_delta_norm=(
            None
            if item.get("value_head_parameter_delta_norm") is None
            else float(item["value_head_parameter_delta_norm"])
        ),
        logit_head_parameter_delta_norm=(
            None
            if item.get("logit_head_parameter_delta_norm") is None
            else float(item["logit_head_parameter_delta_norm"])
        ),
        actor_batch_sample_count=(
            None
            if item.get("actor_batch_sample_count") is None
            else int(item["actor_batch_sample_count"])
        ),
        cf_batch_vshift_sum=(
            None if item.get("cf_batch_vshift_sum") is None else float(item["cf_batch_vshift_sum"])
        ),
        cf_batch_vshift_mean=(
            None if item.get("cf_batch_vshift_mean") is None else float(item["cf_batch_vshift_mean"])
        ),
        cf_batch_vshift_positive_fraction=(
            None
            if item.get("cf_batch_vshift_positive_fraction") is None
            else float(item["cf_batch_vshift_positive_fraction"])
        ),
        reference_raw_drift_mean=(
            None if item.get("reference_raw_drift_mean") is None else float(item["reference_raw_drift_mean"])
        ),
        reference_raw_drift_median=(
            None if item.get("reference_raw_drift_median") is None else float(item["reference_raw_drift_median"])
        ),
        reference_raw_drift_p95=(
            None if item.get("reference_raw_drift_p95") is None else float(item["reference_raw_drift_p95"])
        ),
        reference_raw_drift_p99=(
            None if item.get("reference_raw_drift_p99") is None else float(item["reference_raw_drift_p99"])
        ),
        reference_raw_drift_max=(
            None if item.get("reference_raw_drift_max") is None else float(item["reference_raw_drift_max"])
        ),
        reference_constrained_drift_mean=(
            None if item.get("reference_constrained_drift_mean") is None else float(item["reference_constrained_drift_mean"])
        ),
        reference_constrained_drift_median=(
            None if item.get("reference_constrained_drift_median") is None else float(item["reference_constrained_drift_median"])
        ),
        reference_constrained_drift_p95=(
            None if item.get("reference_constrained_drift_p95") is None else float(item["reference_constrained_drift_p95"])
        ),
        reference_constrained_drift_p99=(
            None if item.get("reference_constrained_drift_p99") is None else float(item["reference_constrained_drift_p99"])
        ),
        reference_constrained_drift_max=(
            None if item.get("reference_constrained_drift_max") is None else float(item["reference_constrained_drift_max"])
        ),
        selected_drift_mean=(
            None if item.get("selected_drift_mean") is None else float(item["selected_drift_mean"])
        ),
        selected_drift_p95=(
            None if item.get("selected_drift_p95") is None else float(item["selected_drift_p95"])
        ),
        selected_drift_max=(
            None if item.get("selected_drift_max") is None else float(item["selected_drift_max"])
        ),
        reference_drift_alpha=(
            None if item.get("reference_drift_alpha") is None else float(item["reference_drift_alpha"])
        ),
        reference_drift_constraint_active=(
            None
            if item.get("reference_drift_constraint_active") is None
            else bool(item["reference_drift_constraint_active"])
        ),
        reference_probability_drift_cap=(
            None
            if item.get("reference_probability_drift_cap") is None
            else float(item["reference_probability_drift_cap"])
        ),
        reference_state_count=(
            None if item.get("reference_state_count") is None else int(item["reference_state_count"])
        ),
        reference_unselected_count=(
            None if item.get("reference_unselected_count") is None else int(item["reference_unselected_count"])
        ),
        reference_selected_count=(
            None if item.get("reference_selected_count") is None else int(item["reference_selected_count"])
        ),
        reference_selected_intersection_count=(
            None
            if item.get("reference_selected_intersection_count") is None
            else int(item["reference_selected_intersection_count"])
        ),
        reference_drift_by_trigger=item.get("reference_drift_by_trigger"),
        old_p_reconstruction_max_error=(
            None
            if item.get("old_p_reconstruction_max_error") is None
            else float(item["old_p_reconstruction_max_error"])
        ),
        old_p_reconstruction_checked=(
            None
            if item.get("old_p_reconstruction_checked") is None
            else bool(item["old_p_reconstruction_checked"])
        ),
        old_p_reconstruction_tolerance=(
            None
            if item.get("old_p_reconstruction_tolerance") is None
            else float(item["old_p_reconstruction_tolerance"])
        ),
        probability_dtype=(
            None
            if item.get("probability_dtype") is None
            else str(item["probability_dtype"])
        ),
        reference_count_checked=(
            None
            if item.get("reference_count_checked") is None
            else int(item["reference_count_checked"])
        ),
        collection_policy_version=(
            None
            if item.get("collection_policy_version") is None
            else int(item["collection_policy_version"])
        ),
        direct_cf_raw_vshift_sum=(
            None if item.get("direct_cf_raw_vshift_sum") is None else float(item["direct_cf_raw_vshift_sum"])
        ),
        direct_cf_raw_vshift_mean=(
            None if item.get("direct_cf_raw_vshift_mean") is None else float(item["direct_cf_raw_vshift_mean"])
        ),
        direct_cf_raw_vshift_positive_fraction=(
            None
            if item.get("direct_cf_raw_vshift_positive_fraction") is None
            else float(item["direct_cf_raw_vshift_positive_fraction"])
        ),
        direct_cf_constrained_vshift_sum=(
            None
            if item.get("direct_cf_constrained_vshift_sum") is None
            else float(item["direct_cf_constrained_vshift_sum"])
        ),
        direct_cf_constrained_vshift_mean=(
            None
            if item.get("direct_cf_constrained_vshift_mean") is None
            else float(item["direct_cf_constrained_vshift_mean"])
        ),
        direct_cf_constrained_vshift_positive_fraction=(
            None
            if item.get("direct_cf_constrained_vshift_positive_fraction") is None
            else float(item["direct_cf_constrained_vshift_positive_fraction"])
        ),
        direct_cf_vshift_retention_ratio=(
            None
            if item.get("direct_cf_vshift_retention_ratio") is None
            else float(item["direct_cf_vshift_retention_ratio"])
        ),
        gate_optimizer_steps=int(item.get("gate_optimizer_steps", 0)),
        gate_loss_mean=(
            None if item.get("gate_loss_mean") is None else float(item["gate_loss_mean"])
        ),
        gate_gradient_norm_mean=(
            None
            if item.get("gate_gradient_norm_mean") is None
            else float(item["gate_gradient_norm_mean"])
        ),
        gate_gradient_finite=(
            None
            if item.get("gate_gradient_finite") is None
            else bool(item["gate_gradient_finite"])
        ),
        gate_probability_shift_mean=(
            None
            if item.get("gate_probability_shift_mean") is None
            else float(item["gate_probability_shift_mean"])
        ),
        gate_probability_shift_max=(
            None
            if item.get("gate_probability_shift_max") is None
            else float(item["gate_probability_shift_max"])
        ),
        gate_delta_abs_mean=(
            None
            if item.get("gate_delta_abs_mean") is None
            else float(item["gate_delta_abs_mean"])
        ),
        gate_delta_positive_fraction=(
            None
            if item.get("gate_delta_positive_fraction") is None
            else float(item["gate_delta_positive_fraction"])
        ),
        gate_sample_count=int(item.get("gate_sample_count", 0)),
        gate_branch_environment_steps=int(item.get("gate_branch_environment_steps", 0)),
        gate_parameter_delta_max=(
            None
            if item.get("gate_parameter_delta_max") is None
            else float(item["gate_parameter_delta_max"])
        ),
        gate_subpolicy_parameter_delta_max=(
            None
            if item.get("gate_subpolicy_parameter_delta_max") is None
            else float(item["gate_subpolicy_parameter_delta_max"])
        ),
        gate_collection_policy_version=(
            None
            if item.get("gate_collection_policy_version") is None
            else int(item["gate_collection_policy_version"])
        ),
    )


def _sparse_direct_cf_episode_summary_from_builtin(
    value: Any,
    expected_index: int,
) -> TrainingEpisodeSummary:
    """把 V3 flat history record 转为绘图/报告共用的episode summary。"""

    item = _as_mapping(value, f"sparse_direct_cf episode[{expected_index}]")
    required = {
        "episode_index",
        "policy_version_before",
        "policy_version_after",
        "rollout_reward_sum",
        "rollout_reward_mean",
        "actual_q_sum",
        "cache_update_cost_sum",
        "movement_cost_sum",
        "cf_sample_count",
        "cf_delta_mean",
        "cf_delta_std",
        "cf_delta_abs_mean",
        "cf_delta_positive_fraction",
        "cf_advantage_mean",
        "cf_advantage_std",
        "cf_advantage_abs_mean",
        "actor_loss_mean",
        "actor_gradient_norm_mean",
        "actor_gradient_finite",
        "approx_kl_mean",
        "clip_fraction_mean",
        "direct_cf_local_objective_change_proxy",
        "branch_environment_steps",
    }
    missing = sorted(required - set(item))
    if missing:
        raise ValueError(f"sparse_direct_cf history缺少字段：{missing}")
    episode_index = int(item["episode_index"])
    if episode_index != int(expected_index):
        raise ValueError("sparse_direct_cf episode_index 必须从0连续递增。")

    def optional_float(name: str) -> float | None:
        raw = item[name]
        if raw is None:
            return None
        result = float(raw)
        if not math.isfinite(result):
            raise ValueError(f"sparse_direct_cf {name} 必须是finite数值。")
        return result

    float_values = {
        name: float(item[name])
        for name in (
            "rollout_reward_sum",
            "rollout_reward_mean",
            "actual_q_sum",
            "cache_update_cost_sum",
            "movement_cost_sum",
            "actor_loss_mean",
            "actor_gradient_norm_mean",
            "approx_kl_mean",
            "clip_fraction_mean",
        )
    }
    if not all(math.isfinite(value) for value in float_values.values()):
        raise ValueError("sparse_direct_cf history包含非finite episode数值。")
    policy_entropy = (
        None
        if item.get("policy_entropy") is None
        else optional_float("policy_entropy")
    )
    def optional_reporting_float(primary_name: str, compatibility_name: str) -> float | None:
        """读取正式probability-change名称，并兼容本轮先写入的短名称。"""

        if item.get(primary_name) is not None:
            return optional_float(primary_name)
        if item.get(compatibility_name) is not None:
            return optional_float(compatibility_name)
        return None

    drift_mean = optional_reporting_float(
        "policy_probability_change_mean_abs",
        "probability_drift_mean_abs",
    )
    drift_p95 = optional_reporting_float(
        "policy_probability_change_p95",
        "probability_drift_p95",
    )
    drift_p99 = optional_reporting_float(
        "policy_probability_change_p99",
        "probability_drift_p99",
    )
    drift_max = optional_reporting_float(
        "policy_probability_change_max",
        "probability_drift_max",
    )
    encoder_parameter_delta_norm = optional_float("encoder_parameter_delta_norm") if item.get("encoder_parameter_delta_norm") is not None else None
    trigger_heads_parameter_delta_norm = optional_float("trigger_heads_parameter_delta_norm") if item.get("trigger_heads_parameter_delta_norm") is not None else None
    autoregressive_parameter_delta_norm = optional_float("autoregressive_parameter_delta_norm") if item.get("autoregressive_parameter_delta_norm") is not None else None
    value_head_parameter_delta_norm = optional_float("value_head_parameter_delta_norm") if item.get("value_head_parameter_delta_norm") is not None else None
    raw_trigger_counts = item.get("cf_trigger_sample_counts")
    trigger_counts: dict[str, int] | None
    if raw_trigger_counts is None:
        trigger_counts = None
    elif isinstance(raw_trigger_counts, dict):
        trigger_counts_mapping = _as_mapping(
            raw_trigger_counts,
            f"sparse_direct_cf episode[{expected_index}].cf_trigger_sample_counts",
        )
        trigger_counts = {}
        for label in trigger_counts_mapping:
            raw_count = trigger_counts_mapping[label]
            if isinstance(raw_count, bool) or int(raw_count) != raw_count or int(raw_count) < 0:
                raise ValueError(f"cf_trigger_sample_counts[{label}] 必须是非负整数。")
            trigger_counts[str(label)] = int(raw_count)
    else:
        count_list = _trigger_count_list(
            raw_trigger_counts,
            f"sparse_direct_cf episode[{expected_index}].cf_trigger_sample_counts",
        )
        trigger_counts = {
            f"trigger_{index}": int(value)
            for index, value in enumerate(count_list)
        }
    free_counts_raw = item.get("cf_trigger_free_counts")
    free_counts = _trigger_count_list(
        free_counts_raw,
        f"sparse_direct_cf episode[{expected_index}].cf_trigger_free_counts",
    )
    sampling_weights = _trigger_weight_list(
        item.get("cf_trigger_sampling_weights"),
        f"sparse_direct_cf episode[{expected_index}].cf_trigger_sampling_weights",
    )
    sampling_strategy = str(item.get("cf_sampling_strategy", "uniform"))
    if sampling_strategy not in {"uniform", "trigger_stratified"}:
        raise ValueError("sparse_direct_cf cf_sampling_strategy 不支持。")
    def optional_int(name: str) -> int | None:
        raw = item.get(name)
        if raw is None:
            return None
        if isinstance(raw, bool) or int(raw) != raw:
            raise ValueError(f"sparse_direct_cf {name} 必须是整数。")
        return int(raw)

    optimizer_steps = int(item.get("optimizer_steps", 0))
    if optimizer_steps < 0:
        raise ValueError("sparse_direct_cf optimizer_steps 不能为负。")
    actor_optimizer_steps_this_episode = int(
        item.get("actor_optimizer_steps_this_episode", optimizer_steps)
    )
    if actor_optimizer_steps_this_episode < 0:
        raise ValueError("sparse_direct_cf actor_optimizer_steps_this_episode 不能为负。")
    cf_batch_pending_sample_count = int(item.get("cf_batch_pending_sample_count", 0))
    if cf_batch_pending_sample_count < 0:
        raise ValueError("sparse_direct_cf cf_batch_pending_sample_count 不能为负。")
    actor_batch_sample_count = optional_int("actor_batch_sample_count")
    pre_clip_gradient_norm = (
        optional_float("pre_clip_gradient_norm")
        if item.get("pre_clip_gradient_norm") is not None
        else None
    )
    logit_head_parameter_delta_norm = (
        optional_float("logit_head_parameter_delta_norm")
        if item.get("logit_head_parameter_delta_norm") is not None
        else None
    )
    old_p_reconstruction_max_error = (
        optional_float("old_p_reconstruction_max_error")
        if item.get("old_p_reconstruction_max_error") is not None
        else None
    )
    old_p_reconstruction_checked = item.get("old_p_reconstruction_checked")
    if old_p_reconstruction_checked is not None:
        old_p_reconstruction_checked = bool(old_p_reconstruction_checked)
    reference_count_checked = optional_int("reference_count_checked")
    reference_float_names = (
        "reference_raw_drift_mean",
        "reference_raw_drift_median",
        "reference_raw_drift_p95",
        "reference_raw_drift_p99",
        "reference_raw_drift_max",
        "reference_constrained_drift_mean",
        "reference_constrained_drift_median",
        "reference_constrained_drift_p95",
        "reference_constrained_drift_p99",
        "reference_constrained_drift_max",
        "selected_drift_mean",
        "selected_drift_p95",
        "selected_drift_max",
        "reference_drift_alpha",
        "reference_probability_drift_cap",
        "old_p_reconstruction_tolerance",
        "direct_cf_raw_vshift_sum",
        "direct_cf_raw_vshift_mean",
        "direct_cf_raw_vshift_positive_fraction",
        "direct_cf_constrained_vshift_sum",
        "direct_cf_constrained_vshift_mean",
        "direct_cf_constrained_vshift_positive_fraction",
        "direct_cf_vshift_retention_ratio",
    )
    reference_float_values = {
        name: optional_float(name) if item.get(name) is not None else None
        for name in reference_float_names
    }
    reference_drift_by_trigger = item.get("reference_drift_by_trigger")
    if reference_drift_by_trigger is not None:
        reference_drift_by_trigger = _as_mapping(
            reference_drift_by_trigger,
            f"sparse_direct_cf episode[{expected_index}].reference_drift_by_trigger",
        )
    reference_constraint_active = item.get("reference_drift_constraint_active")
    if reference_constraint_active is not None:
        reference_constraint_active = bool(reference_constraint_active)
    probability_dtype = item.get("probability_dtype")
    if probability_dtype is not None:
        probability_dtype = str(probability_dtype)
    return TrainingEpisodeSummary(
        episode_index=episode_index,
        policy_version_before=int(item["policy_version_before"]),
        policy_version_after=int(item["policy_version_after"]),
        rollout_reward_sum=float_values["rollout_reward_sum"],
        rollout_reward_mean=float_values["rollout_reward_mean"],
        batch_reward_sum_float32=float_values["rollout_reward_sum"],
        batch_vs_raw_reward_sum_error=0.0,
        actual_q_sum=float_values["actual_q_sum"],
        cache_cost_sum=float_values["cache_update_cost_sum"],
        movement_cost_sum=float_values["movement_cost_sum"],
        policy_loss_mean=float_values["actor_loss_mean"],
        value_loss_mean=0.0,
        entropy_mean=0.0 if policy_entropy is None else float(policy_entropy),
        approx_kl_mean=float_values["approx_kl_mean"],
        clip_fraction_mean=float_values["clip_fraction_mean"],
        gradient_norm_mean=float_values["actor_gradient_norm_mean"],
        optimizer_steps=optimizer_steps,
        credit_phase="sparse_direct_cf",
        actor_update_executed=bool(item.get("actor_updated_this_episode", True)),
        actor_updated_this_episode=bool(item.get("actor_updated_this_episode", True)),
        actor_optimizer_steps_this_episode=actor_optimizer_steps_this_episode,
        cf_sample_count=int(item["cf_sample_count"]),
        cf_sampling_strategy=sampling_strategy,
        cf_effective_sample_count=int(
            item.get("cf_effective_sample_count", item["cf_sample_count"])
        ),
        cf_trigger_free_counts=free_counts,
        cf_trigger_sampling_weights=sampling_weights,
        cf_delta_mean=optional_float("cf_delta_mean"),
        cf_delta_std=optional_float("cf_delta_std"),
        cf_delta_abs_mean=optional_float("cf_delta_abs_mean"),
        cf_delta_positive_fraction=optional_float("cf_delta_positive_fraction"),
        cf_advantage_mean=optional_float("cf_advantage_mean"),
        cf_advantage_std=optional_float("cf_advantage_std"),
        cf_advantage_abs_mean=optional_float("cf_advantage_abs_mean"),
        direct_cf_local_objective_change_proxy=optional_float(
            "direct_cf_local_objective_change_proxy"
        ),
        branch_environment_steps=int(item["branch_environment_steps"]),
        actor_gradient_finite=bool(item["actor_gradient_finite"]),
        policy_entropy=policy_entropy,
        probability_drift_mean_abs=drift_mean,
        probability_drift_p95=drift_p95,
        probability_drift_p99=drift_p99,
        probability_drift_max=drift_max,
        encoder_parameter_delta_norm=encoder_parameter_delta_norm,
        trigger_heads_parameter_delta_norm=trigger_heads_parameter_delta_norm,
        autoregressive_parameter_delta_norm=autoregressive_parameter_delta_norm,
        value_head_parameter_delta_norm=value_head_parameter_delta_norm,
        cf_trigger_sample_counts=trigger_counts,
        cf_batch_id=optional_int("cf_batch_id"),
        cf_batch_episode_index=optional_int("cf_batch_episode_index"),
        cf_batch_pending_sample_count=cf_batch_pending_sample_count,
        collection_policy_version=optional_int("collection_policy_version"),
        actor_batch_sample_count=actor_batch_sample_count,
        pre_clip_gradient_norm=pre_clip_gradient_norm,
        cf_batch_vshift_sum=(
            optional_float("cf_batch_vshift_sum")
            if item.get("cf_batch_vshift_sum") is not None
            else None
        ),
        cf_batch_vshift_mean=(
            optional_float("cf_batch_vshift_mean")
            if item.get("cf_batch_vshift_mean") is not None
            else None
        ),
        cf_batch_vshift_positive_fraction=(
            optional_float("cf_batch_vshift_positive_fraction")
            if item.get("cf_batch_vshift_positive_fraction") is not None
            else None
        ),
        logit_head_parameter_delta_norm=logit_head_parameter_delta_norm,
        **reference_float_values,
        reference_drift_constraint_active=reference_constraint_active,
        reference_state_count=optional_int("reference_state_count"),
        reference_unselected_count=optional_int("reference_unselected_count"),
        reference_selected_count=optional_int("reference_selected_count"),
        reference_selected_intersection_count=optional_int(
            "reference_selected_intersection_count"
        ),
        reference_drift_by_trigger=reference_drift_by_trigger,
        old_p_reconstruction_max_error=old_p_reconstruction_max_error,
        old_p_reconstruction_checked=old_p_reconstruction_checked,
        probability_dtype=probability_dtype,
        reference_count_checked=reference_count_checked,
    )


def _canonical_training_history(value: Any) -> dict[str, Any]:
    """严格校验并规范化history，避免JSON mirror成为resume权威。"""

    source = _as_mapping(value, "training_history")
    schema_version = int(source.get("schema_version", -1))
    credit_assignment = str(
        source.get("credit_assignment", ppo_update.LEGACY_CREDIT_ASSIGNMENT)
    )
    frozen_policy_batch_history = (
        credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT
        and source.get("algorithm_schema")
        == "proposed_sparse_direct_cf_frozen_policy_batch_v1"
    )
    reference_drift_history = (
        credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT
        and source.get("algorithm_schema") == ppo_update.REFERENCE_DRIFT_ALGORITHM_SCHEMA
    )
    hierarchical_gate_history = (
        credit_assignment == ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT
        and source.get("algorithm_schema")
        == hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_ALGORITHM_SCHEMA
    )
    if hierarchical_gate_history:
        expected_schema = HIERARCHICAL_GATE_TRAINING_HISTORY_SCHEMA_VERSION
    elif reference_drift_history:
        expected_schema = REFERENCE_DRIFT_TRAINING_HISTORY_SCHEMA_VERSION
    elif frozen_policy_batch_history:
        expected_schema = FROZEN_POLICY_BATCH_TRAINING_HISTORY_SCHEMA_VERSION
    elif credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT:
        expected_schema = SPARSE_DIRECT_CF_TRAINING_HISTORY_SCHEMA_VERSION
    elif credit_assignment == ppo_update.PER_TRIGGER_Q_CREDIT_ASSIGNMENT:
        expected_schema = COUNTERFACTUAL_TRAINING_HISTORY_SCHEMA_VERSION
    elif credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT:
        expected_schema = CENTRALIZED_COUNTERFACTUAL_TRAINING_HISTORY_SCHEMA_VERSION
    else:
        expected_schema = TRAINING_HISTORY_SCHEMA_VERSION
    if schema_version != expected_schema:
        raise ValueError("training_history schema_version 不兼容。")
    if credit_assignment not in {
        ppo_update.LEGACY_CREDIT_ASSIGNMENT,
        ppo_update.PER_TRIGGER_Q_CREDIT_ASSIGNMENT,
        ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT,
        ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT,
        ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT,
        ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT,
    }:
        raise ValueError(f"training_history credit_assignment 不支持：{credit_assignment}")
    if hierarchical_gate_history:
        if source.get("method_id") != hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_METHOD:
            raise ValueError("hierarchical Gate training_history method_id 不兼容。")
        if source.get("architecture_id") != ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID:
            raise ValueError("hierarchical Gate training_history architecture_id 不兼容。")
        if source.get("causal_artifact_schema") != causal_artifacts.CAUSAL_ARTIFACT_SCHEMA_ID:
            raise ValueError("hierarchical Gate training_history causal artifact schema 不兼容。")
        if source.get("hierarchical_gate_artifact_schema") != causal_artifacts.HIERARCHICAL_GATE_ARTIFACT_SCHEMA_ID:
            raise ValueError("hierarchical Gate artifact schema 不兼容。")
        if int(source.get("cf_batch_episodes", -1)) != HIERARCHICAL_GATE_CF_BATCH_EPISODES:
            raise ValueError("hierarchical Gate training_history cf_batch_episodes必须为4。")
        if int(source.get("gate_cf_samples_per_episode", -1)) != HIERARCHICAL_GATE_CF_SAMPLES_PER_EPISODE:
            raise ValueError("hierarchical Gate training_history gate K必须为8。")
        if source.get("checkpoint_save_semantics") != "episode_atomic_with_pending_batch":
            raise ValueError("hierarchical Gate training_history checkpoint语义不兼容。")
        if float(source.get("reference_probability_drift_cap", -1.0)) != 0.005:
            raise ValueError("hierarchical Gate training_history drift cap必须为0.005。")
        if source.get("reference_constraint_method") != reference_drift.REFERENCE_DRIFT_CONSTRAINT_METHOD:
            raise ValueError("hierarchical Gate training_history constraint method不兼容。")
        if source.get("reference_constraint_version") != reference_drift.REFERENCE_DRIFT_CONSTRAINT_VERSION:
            raise ValueError("hierarchical Gate training_history constraint version不兼容。")
        if str(source.get("gate_sampling_strategy")) != "uniform_slots":
            raise ValueError("hierarchical Gate training_history采样策略不兼容。")
        for name in (
            "pending_cf_sample_count",
            "pending_gate_cf_sample_count",
            "pending_reference_state_count",
            "gate_optimizer_step_count",
            "gate_policy_version",
            "actor_policy_version",
            "actor_optimizer_step_count",
        ):
            if int(source.get(name, -1)) < 0:
                raise ValueError(f"hierarchical Gate training_history {name}非法。")
        if int(source.get("actor_policy_version")) != int(source.get("policy_version")):
            raise ValueError("hierarchical Gate actor_policy_version与policy_version不一致。")
        if int(source.get("gate_policy_version")) != int(source.get("policy_version")):
            raise ValueError("hierarchical Gate gate_policy_version与policy_version不一致。")
        if int(source.get("actor_optimizer_step_count")) != int(source.get("optimizer_step_count")):
            raise ValueError("hierarchical Gate actor optimizer step与optimizer_step_count不一致。")
        if int(source.get("gate_optimizer_step_count")) != int(source.get("optimizer_step_count")):
            raise ValueError("hierarchical Gate gate optimizer step与optimizer_step_count不一致。")
        if int(source.get("causal_update_artifact_index", -1)) != int(source.get("optimizer_step_count", -2)):
            raise ValueError("hierarchical Gate causal artifact index必须等于已完成optimizer steps。")
        history_sampling_strategy = str(source.get("cf_sampling_strategy", "uniform"))
        if history_sampling_strategy not in {"uniform", "trigger_stratified"}:
            raise ValueError("hierarchical Gate WHAT cf_sampling_strategy不支持。")
    elif credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT:
        if reference_drift_history:
            if source.get("method_id") != ppo_update.REFERENCE_DRIFT_FROZEN_POLICY_BATCH_METHOD:
                raise ValueError("refdrift training_history method_id 不兼容。")
            if source.get("architecture_id") != ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID:
                raise ValueError("refdrift training_history architecture_id 不兼容。")
            if source.get("causal_artifact_schema") != causal_artifacts.CAUSAL_ARTIFACT_SCHEMA_ID:
                raise ValueError("refdrift training_history causal artifact schema 不兼容。")
            if int(source.get("cf_batch_episodes", -1)) != 4:
                raise ValueError("refdrift training_history cf_batch_episodes 必须为4。")
            if source.get("checkpoint_save_semantics") != "episode_atomic_with_pending_batch":
                raise ValueError("refdrift training_history checkpoint语义不兼容。")
            if float(source.get("reference_probability_drift_cap", -1.0)) != 0.005:
                raise ValueError("refdrift training_history drift cap必须为0.005。")
            if source.get("reference_constraint_method") != reference_drift.REFERENCE_DRIFT_CONSTRAINT_METHOD:
                raise ValueError("refdrift training_history constraint method不兼容。")
            if source.get("reference_constraint_version") != reference_drift.REFERENCE_DRIFT_CONSTRAINT_VERSION:
                raise ValueError("refdrift training_history constraint version不兼容。")
            if int(source.get("pending_cf_sample_count", -1)) < 0:
                raise ValueError("refdrift pending_cf_sample_count非法。")
            if int(source.get("pending_reference_state_count", -1)) < 0:
                raise ValueError("refdrift pending_reference_state_count非法。")
            if int(source.get("causal_update_artifact_index", -1)) != int(
                source.get("optimizer_step_count", -2)
            ):
                raise ValueError(
                    "refdrift training_history causal update artifact index必须等于已完成optimizer steps。"
                )
        elif frozen_policy_batch_history:
            if source.get("method_id") != ppo_update.FROZEN_POLICY_BATCH_METHOD:
                raise ValueError("frozen_policy_batch training_history method_id 不兼容。")
            if source.get("architecture_id") != ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID:
                raise ValueError("frozen_policy_batch training_history architecture_id 不兼容。")
            if int(source.get("cf_batch_episodes", -1)) != 4:
                raise ValueError("frozen_policy_batch training_history cf_batch_episodes 必须为4。")
            if source.get("checkpoint_save_semantics") != "batch_boundary_only":
                raise ValueError("frozen_policy_batch training_history checkpoint语义不兼容。")
            if int(source.get("pending_cf_sample_count", -1)) != 0:
                raise ValueError("frozen_policy_batch持久化history不得包含pending samples。")
        elif source.get("algorithm_schema") != "proposed_sparse_direct_cf_ppo_v3":
            raise ValueError("sparse_direct_cf training_history algorithm_schema 不兼容。")
        if int(source.get("actor_optimizer_step_count", -1)) < 0:
            raise ValueError("sparse_direct_cf actor_optimizer_step_count 非法。")
        history_sampling_strategy = str(source.get("cf_sampling_strategy", "uniform"))
        if history_sampling_strategy not in {"uniform", "trigger_stratified"}:
            raise ValueError("sparse_direct_cf training_history cf_sampling_strategy 不支持。")
    elif credit_assignment == ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT:
        history_sampling_strategy = str(source.get("cf_sampling_strategy", "uniform"))
    else:
        history_sampling_strategy = "uniform"
    if credit_assignment == ppo_update.PER_TRIGGER_Q_CREDIT_ASSIGNMENT:
        if str(source.get("q_target")) != ppo_update.MONTE_CARLO_Q_TARGET:
            raise ValueError("counterfactual training_history q_target 不兼容。")
        if int(source.get("q_optimizer_step_count", -1)) < 0:
            raise ValueError("counterfactual training_history q_optimizer_step_count 非法。")
    if credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT:
        if str(source.get("q_target")) != ppo_update.MONTE_CARLO_Q_TARGET:
            raise ValueError("centralized counterfactual training_history q_target 不兼容。")
        if str(source.get("credit_phase")) not in {"critic_warmup", "joint_training"}:
            raise ValueError("centralized counterfactual training_history credit_phase 非法。")
        if "critic_return_scale" not in source:
            raise ValueError("centralized counterfactual training_history 缺少 critic_return_scale。")
        history_scale = float(source["critic_return_scale"])
        if not math.isfinite(history_scale) or history_scale <= 0.0:
            raise ValueError("centralized training_history critic_return_scale 非法。")
        if int(source.get("critic_optimizer_step_count", -1)) < 0:
            raise ValueError("centralized training_history critic_optimizer_step_count 非法。")
        if int(source.get("critic_update_count", -1)) < 0:
            raise ValueError("centralized training_history critic_update_count 非法。")
    raw_episodes = source.get("episodes")
    if not isinstance(raw_episodes, list):
        raise ValueError("training_history.episodes 必须是数组。")

    records: list[dict[str, Any]] = []
    if credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT:
        for expected_index, raw_record in enumerate(raw_episodes):
            summary = _sparse_direct_cf_episode_summary_from_builtin(
                raw_record,
                expected_index,
            )
            records.append(_training_episode_history_record(
                summary,
                FormalPPOUpdateSummary(
                    update_epochs=0,
                    num_minibatches=0,
                    minibatch_size=0,
                    optimizer_steps=0,
                    policy_loss_mean=summary.policy_loss_mean,
                    value_loss_mean=0.0,
                    entropy_mean=0.0,
                    approx_kl_mean=summary.approx_kl_mean,
                    clip_fraction_mean=summary.clip_fraction_mean,
                    gradient_norm_mean=summary.gradient_norm_mean,
                    gradient_finite=summary.actor_gradient_finite,
                    epoch_summaries=tuple(),
                    old_log_probs_frozen=True,
                    old_values_frozen=True,
                    advantages_frozen=True,
                    returns_frozen=True,
                    each_transition_once_per_epoch=True,
                ),
                credit_assignment,
            ))
        completed_episodes = int(source.get("completed_episodes", -1))
        policy_version = int(source.get("policy_version", -1))
        optimizer_step_count = int(source.get("optimizer_step_count", -1))
        if completed_episodes != len(records):
            raise ValueError("training_history.completed_episodes 与episodes数量不一致。")
        if completed_episodes < 0 or policy_version < 0 or optimizer_step_count < 0:
            raise ValueError("training_history counters 不能为负。")
        return _build_training_history(
            records,
            policy_version,
            optimizer_step_count,
            credit_assignment,
            cf_sampling_strategy=history_sampling_strategy,
            method_id=(
                ppo_update.REFERENCE_DRIFT_FROZEN_POLICY_BATCH_METHOD
                if reference_drift_history
                else (ppo_update.FROZEN_POLICY_BATCH_METHOD if frozen_policy_batch_history else None)
            ),
            architecture_id=(
                source.get("architecture_id")
                if reference_drift_history or frozen_policy_batch_history
                else None
            ),
            cf_batch_episodes=(
                int(source.get("cf_batch_episodes", 1))
                if reference_drift_history or frozen_policy_batch_history
                else 1
            ),
            cf_samples_per_episode=(
                int(source.get("cf_samples_per_episode", 0))
                if reference_drift_history or frozen_policy_batch_history
                else 0
            ),
            batch_id=(
                int(source.get("batch_id", 0))
                if reference_drift_history or frozen_policy_batch_history
                else 0
            ),
            batch_episode_index=(
                int(source.get("batch_episode_index", 0))
                if reference_drift_history or frozen_policy_batch_history
                else 0
            ),
            pending_cf_sample_count=(
                int(source.get("pending_cf_sample_count", 0))
                if reference_drift_history or frozen_policy_batch_history
                else 0
            ),
            pending_collection_policy_version=(
                None
                if source.get("pending_collection_policy_version") is None
                else int(source["pending_collection_policy_version"])
            ),
            reference_probability_drift_cap=(
                float(source.get("reference_probability_drift_cap", 0.005))
                if reference_drift_history
                else None
            ),
            pending_reference_state_count=(
                int(source.get("pending_reference_state_count", 0))
                if reference_drift_history
                else 0
            ),
        )
    for expected_index, raw_record in enumerate(raw_episodes):
        record = _as_mapping(raw_record, f"training_history.episodes[{expected_index}]")
        episode_summary = _training_episode_summary_from_builtin(record.get("training_episode_summary"))
        update_summary = _formal_update_summary_from_builtin(record.get("ppo_update_summary"))
        record_index = int(record.get("episode_index", episode_summary.episode_index))
        if record_index != expected_index or episode_summary.episode_index != expected_index:
            raise ValueError("training_history episode_index 必须从0连续递增。")
        records.append(
            _training_episode_history_record(
                episode_summary,
                update_summary,
                credit_assignment,
            )
        )

    completed_episodes = int(source.get("completed_episodes", -1))
    policy_version = int(source.get("policy_version", -1))
    optimizer_step_count = int(source.get("optimizer_step_count", -1))
    if completed_episodes != len(records):
        raise ValueError("training_history.completed_episodes 与episodes数量不一致。")
    if completed_episodes < 0 or policy_version < 0 or optimizer_step_count < 0:
        raise ValueError("training_history counters 不能为负。")
    return _build_training_history(
        records,
        policy_version,
        optimizer_step_count,
        credit_assignment,
        int(source.get("q_optimizer_step_count", 0)),
        str(source.get("credit_phase", "critic_warmup")),
        int(source.get("critic_optimizer_step_count", source.get("q_optimizer_step_count", 0))),
        None if credit_assignment != ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT
        else float(source["critic_return_scale"]),
        history_sampling_strategy,
        method_id=(
            hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_METHOD
            if hierarchical_gate_history
            else None
        ),
        architecture_id=(
            ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID
            if hierarchical_gate_history
            else None
        ),
        cf_batch_episodes=(
            int(source.get("cf_batch_episodes", HIERARCHICAL_GATE_CF_BATCH_EPISODES))
            if hierarchical_gate_history
            else 1
        ),
        cf_samples_per_episode=(
            int(source.get("cf_samples_per_episode", 0))
            if hierarchical_gate_history
            else 0
        ),
        batch_id=int(source.get("batch_id", 0)) if hierarchical_gate_history else 0,
        batch_episode_index=(
            int(source.get("batch_episode_index", 0))
            if hierarchical_gate_history
            else 0
        ),
        pending_cf_sample_count=(
            int(source.get("pending_cf_sample_count", 0))
            if hierarchical_gate_history
            else 0
        ),
        pending_collection_policy_version=(
            None
            if source.get("pending_collection_policy_version") is None
            else int(source["pending_collection_policy_version"])
        ),
        reference_probability_drift_cap=(
            float(source.get("reference_probability_drift_cap", 0.005))
            if hierarchical_gate_history
            else None
        ),
        pending_reference_state_count=(
            int(source.get("pending_reference_state_count", 0))
            if hierarchical_gate_history
            else 0
        ),
        gate_policy_version=(
            int(source.get("gate_policy_version", policy_version))
            if hierarchical_gate_history
            else None
        ),
        gate_optimizer_step_count=(
            int(source.get("gate_optimizer_step_count", optimizer_step_count))
            if hierarchical_gate_history
            else 0
        ),
        pending_gate_cf_sample_count=(
            int(source.get("pending_gate_cf_sample_count", 0))
            if hierarchical_gate_history
            else 0
        ),
    )


def _history_episode_summaries(history: dict[str, Any]) -> tuple[TrainingEpisodeSummary, ...]:
    """从canonical history提取Figure 1可直接消费的summary序列。"""

    if history.get("credit_assignment") == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT:
        return tuple(
            _sparse_direct_cf_episode_summary_from_builtin(record, index)
            for index, record in enumerate(history["episodes"])
        )
    return tuple(
        _training_episode_summary_from_builtin(record["training_episode_summary"])
        for record in history["episodes"]
    )


def load_training_episode_summaries(run_dir: str | Path) -> tuple[TrainingEpisodeSummary, ...]:
    """读取training_history.json并返回Figure 1可直接消费的summary。"""

    history_path = Path(run_dir) / TRAINING_HISTORY_FILENAME
    if not history_path.exists():
        raise FileNotFoundError(f"训练历史文件不存在：{history_path}")
    try:
        with history_path.open("r", encoding="utf-8") as stream:
            raw_history = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取训练历史文件：{history_path}") from exc
    history = _canonical_training_history(raw_history)
    return _history_episode_summaries(history)


def _plot_training_convergence_after_persistence(
    run_dir: Path,
    history: dict[str, Any],
) -> None:
    """在history已成功进入正式持久化链路后更新Figure 1。

    绘图只消费canonical history中的episode summary，不重新执行environment、
    policy forward、sampling或optimizer。绘图失败只发出warning，不影响已经
    成功保存的checkpoint/history。
    """

    try:
        from experiment_plotting import plot_training_convergence

        summaries = _history_episode_summaries(history)
        if not summaries:
            raise ValueError("training history中没有可绘制的episode。")
        plot_training_convergence(
            summaries,
            run_dir / "figures" / "01_training_convergence.png",
            moving_average_window=10,
        )
    except Exception as exc:  # reporting故障不能回滚正式训练持久化结果
        warnings.warn(
            f"PPO_TRAINING_CONVERGENCE_PLOT_FAILED: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def _run_dir_paths(run_dir: str | Path) -> tuple[Path, Path, Path]:
    path = Path(run_dir)
    if path.exists() and not path.is_dir():
        raise ValueError(f"run_dir 不是目录：{path}")
    return path, path / LATEST_CHECKPOINT_FILENAME, path / TRAINING_HISTORY_FILENAME


def _prepare_formal_run_dir(run_dir: str | Path, resume: bool) -> tuple[Path, Path, Path]:
    """准备显式run_dir；resume=False不覆盖已有正式run。"""

    path, checkpoint_path, history_path = _run_dir_paths(run_dir)
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"resume=True 但权威checkpoint不存在：{checkpoint_path}"
            )
        return path, checkpoint_path, history_path
    if checkpoint_path.exists() or history_path.exists():
        raise FileExistsError(
            "RUN_DIRECTORY_NOT_EMPTY: run_dir 已包含latest_checkpoint.pt或training_history.json，"
            "请使用resume=True或更换run_dir。"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path, checkpoint_path, history_path


def _atomic_torch_save(payload: dict[str, Any], final_path: Path) -> None:
    """在同目录临时文件完成torch.save后再原子替换。"""

    _require_torch()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{final_path.name}.",
        suffix=".tmp",
        dir=str(final_path.parent),
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, final_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_json_save(payload: dict[str, Any], final_path: Path) -> None:
    """在同目录临时JSON完成flush/fsync后再原子替换。"""

    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{final_path.name}.",
        suffix=".tmp",
        dir=str(final_path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = -1
            json.dump(
                summary_to_builtin(payload),
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, final_path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path.exists():
            temporary_path.unlink()


def _synchronize_training_history_json(
    history_path: Path,
    checkpoint_history: dict[str, Any],
) -> None:
    """以checkpoint history为准，必要时重写JSON mirror。"""

    rewrite = not history_path.exists()
    if history_path.exists():
        try:
            with history_path.open("r", encoding="utf-8") as stream:
                json_history = json.load(stream)
            rewrite = json_history != checkpoint_history
            if rewrite:
                warnings.warn(
                    "TRAINING_HISTORY_JSON_INCONSISTENT: 以checkpoint history为准并重新同步JSON。",
                    RuntimeWarning,
                    stacklevel=2,
                )
        except (OSError, json.JSONDecodeError, ValueError):
            rewrite = True
            warnings.warn(
                "TRAINING_HISTORY_JSON_INVALID: 以checkpoint history为准并重新生成JSON。",
                RuntimeWarning,
                stacklevel=2,
            )
    if rewrite:
        _atomic_json_save(checkpoint_history, history_path)


def _policy_device(policy: Any) -> Any:
    _require_torch()
    try:
        return next(policy.parameters()).device
    except StopIteration as exc:
        raise ValueError("policy 必须包含至少一个参数。") from exc


def _capture_rng_states(
    policy: Any,
    rng_generator: Any,
    critic_rng_generator: Any | None = None,
) -> dict[str, Any]:
    """保存正式训练实际会用到的Python/NumPy/torch RNG状态。"""

    _require_torch()
    states: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state().clone(),
        "torch_generator_state": None,
        "torch_generator_device": None,
        "torch_cuda_rng_state_all": None,
    }
    if rng_generator is not None:
        if not hasattr(rng_generator, "get_state") or not hasattr(rng_generator, "set_state"):
            raise TypeError("rng_generator 必须是支持get_state/set_state的torch.Generator。")
        states["torch_generator_state"] = rng_generator.get_state().clone()
        states["torch_generator_device"] = str(getattr(rng_generator, "device", "cpu"))
    if critic_rng_generator is not None:
        if not hasattr(critic_rng_generator, "get_state") or not hasattr(critic_rng_generator, "set_state"):
            raise TypeError("critic_rng_generator 必须是支持get_state/set_state的torch.Generator。")
        states["critic_torch_generator_state"] = critic_rng_generator.get_state().clone()
        states["critic_torch_generator_device"] = str(
            getattr(critic_rng_generator, "device", "cpu")
        )

    device = _policy_device(policy)
    if getattr(device, "type", str(device)) == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("policy位于CUDA但当前torch.cuda不可用，无法保存CUDA RNG。")
        states["torch_cuda_rng_state_all"] = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return states


def _restore_rng_states(
    policy: Any,
    rng_generator: Any,
    states: dict[str, Any],
    critic_rng_generator: Any | None = None,
) -> None:
    """在policy/optimizer加载完成后恢复RNG，之后才允许采样或shuffle。"""

    _require_torch()
    required = {"python_random_state", "numpy_random_state", "torch_cpu_rng_state"}
    missing = sorted(required - set(states))
    if missing:
        raise RuntimeError(f"CHECKPOINT_COMPATIBILITY_ERROR: checkpoint缺少RNG字段：{missing}")
    random.setstate(states["python_random_state"])
    np.random.set_state(states["numpy_random_state"])
    torch.set_rng_state(states["torch_cpu_rng_state"])

    cuda_states = states.get("torch_cuda_rng_state_all")
    device = _policy_device(policy)
    if cuda_states is not None and getattr(device, "type", str(device)) == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: checkpoint需要CUDA RNG但当前不可用。")
        torch.cuda.set_rng_state_all(cuda_states)

    generator_state = states.get("torch_generator_state")
    if generator_state is not None:
        if rng_generator is None or not hasattr(rng_generator, "set_state"):
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: checkpoint需要恢复torch Generator。")
        rng_generator.set_state(generator_state)
    critic_generator_state = states.get("critic_torch_generator_state")
    if critic_generator_state is not None:
        if critic_rng_generator is None or not hasattr(critic_rng_generator, "set_state"):
            raise RuntimeError(
                "CHECKPOINT_COMPATIBILITY_ERROR: checkpoint需要恢复central critic Generator。"
            )
        critic_rng_generator.set_state(critic_generator_state)


def _seed_new_formal_run(seed: int, rng_generator: Any) -> None:
    """新run自动构造policy时使用配置seed；不在模块import时改全局RNG。"""

    random.seed(int(seed))
    np.random.seed(int(seed))
    ppo_trigger_policy.set_torch_seed(int(seed))
    rng_generator.manual_seed(int(seed))


def _build_formal_policy_only(ctx: TrainingSanityContext) -> Any:
    """按当前配置构造 policy，不创建 optimizer。"""

    _require_torch()
    network_cfg = ppo_trigger_policy.ppo_network_config_from_project(ctx.cfg)
    configured_architecture = str(
        project_config.get_config_value(
            ctx.cfg,
            "algorithm.ppo.actor_architecture",
            ppo_trigger_policy.TRIGGER_HEADS_ARCHITECTURE_ID,
        )
    )
    if _is_hierarchical_gate_mode(ctx.cfg):
        if configured_architecture != ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID:
            raise RuntimeError(
                "hierarchical Gate held-out test policy必须使用现有 Full Shared Actor："
                f"{ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID}，"
                f"当前为{configured_architecture}。"
            )
        return hierarchical_reconfiguration_gate.build_hierarchical_policy(
            ctx.observation_spec.observation_dim,
            ctx.action_spec,
            subpolicy_hidden_sizes=tuple(int(value) for value in network_cfg["hidden_sizes"]),
            autoregressive_hidden_size=int(network_cfg["autoregressive_hidden_size"]),
        )

    policy_class = (
        ppo_trigger_policy.FullSharedActorCritic
        if configured_architecture == ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID
        else ppo_trigger_policy.ConditionalTriggerActorCritic
    )
    policy = policy_class(
        ctx.observation_spec.observation_dim,
        ctx.action_spec,
        **network_cfg,
    )
    if _is_sparse_direct_cf_mode(ctx.cfg):
        expected_architecture = (
            ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID
            if _is_frozen_batch_family(ctx.cfg)
            else ppo_trigger_policy.TRIGGER_HEADS_ARCHITECTURE_ID
        )
        if configured_architecture != expected_architecture:
            raise RuntimeError(
                "sparse_direct_cf正式Actor architecture与当前method不匹配，必须显式配置"
                f"actor_architecture={expected_architecture}，"
                f"当前为{configured_architecture}。"
            )
    return policy


def _build_formal_training_components(
    ctx: TrainingSanityContext,
) -> tuple[Any, Any, Any | None, Any | None]:
    """按当前 mode 构造 Actor/scalar-V 与独立 Q optimizer。"""

    _require_torch()
    training_cfg = ppo_update.ppo_training_config_status(ctx.cfg)
    if not training_cfg.can_run_optimizer_step or training_cfg.learning_rate is None:
        raise RuntimeError(f"PPO update超参数缺失：{training_cfg.missing_required_for_step}")
    policy = _build_formal_policy_only(ctx)
    if _is_sparse_direct_cf_mode(ctx.cfg):
        actor_parameters = ppo_trigger_policy.sparse_direct_cf_actor_parameters(policy)
        if not actor_parameters:
            raise RuntimeError("sparse_direct_cf Actor optimizer参数为空。")
        actor_optimizer = torch.optim.Adam(actor_parameters, lr=float(training_cfg.learning_rate))
    else:
        actor_optimizer = torch.optim.Adam(policy.parameters(), lr=float(training_cfg.learning_rate))
    if not _is_q_mode(ctx.cfg):
        return policy, actor_optimizer, None, None
    if _is_centralized_mode(ctx.cfg):
        q_critic = ppo_trigger_policy.build_centralized_joint_action_critic_preserving_rng(
            ctx.observation_spec.observation_dim,
            ctx.action_spec.action_dim,
            tuple(ppo_trigger_policy.ppo_network_config_from_project(ctx.cfg)["hidden_sizes"]),
            device=next(policy.parameters()).device,
        )
    else:
        q_critic = ppo_trigger_policy.build_per_trigger_q_critic_preserving_rng(
            ctx.action_spec.action_dim,
            int(getattr(policy, "autoregressive_hidden_size")),
            device=next(policy.parameters()).device,
        )
    q_optimizer = torch.optim.Adam(q_critic.parameters(), lr=float(training_cfg.learning_rate))
    return policy, actor_optimizer, q_critic, q_optimizer


def _build_hierarchical_training_components(
    ctx: TrainingSanityContext,
) -> tuple[hierarchical_reconfiguration_gate.HierarchicalReconfigurationGatePolicy, Any, Any]:
    """Build the independent Gate + existing Full Shared WHAT optimizers."""

    _require_torch()
    training_cfg = ppo_update.ppo_training_config_status(ctx.cfg)
    if not training_cfg.can_run_optimizer_step or training_cfg.learning_rate is None:
        raise RuntimeError(
            f"hierarchical Gate PPO update超参数缺失：{training_cfg.missing_required_for_step}"
        )
    configured_architecture = str(
        project_config.get_config_value(
            ctx.cfg,
            "algorithm.ppo.actor_architecture",
            ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID,
        )
    )
    if configured_architecture != ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID:
        raise RuntimeError(
            "hierarchical Gate formal WHAT policy必须使用现有 Full Shared Actor："
            f"{ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID}，当前为{configured_architecture}。"
        )
    network_cfg = ppo_trigger_policy.ppo_network_config_from_project(ctx.cfg)
    policy = hierarchical_reconfiguration_gate.build_hierarchical_policy(
        ctx.observation_spec.observation_dim,
        ctx.action_spec,
        subpolicy_hidden_sizes=tuple(int(value) for value in network_cfg["hidden_sizes"]),
        autoregressive_hidden_size=int(network_cfg["autoregressive_hidden_size"]),
    )
    optimizers = hierarchical_reconfiguration_gate.build_independent_optimizers(
        policy,
        float(training_cfg.learning_rate),
    )
    isolation = hierarchical_reconfiguration_gate.optimizer_parameter_isolation(policy, optimizers)
    if not all(isolation.values()):
        raise AssertionError(f"hierarchical Gate optimizer参数隔离失败：{isolation}")
    return policy, optimizers.subpolicy_optimizer, optimizers.gate_optimizer


def _build_formal_policy_and_optimizer(ctx: TrainingSanityContext) -> tuple[Any, Any]:
    """legacy兼容wrapper：只返回 Actor/scalar-V policy 和 optimizer。"""

    policy, optimizer, _, _ = _build_formal_training_components(ctx)
    return policy, optimizer


def _formal_compatibility_metadata(
    ctx: TrainingSanityContext,
    protocol: FormalTrainingProtocol,
    policy: Any,
) -> dict[str, Any]:
    """构造最小但足够阻止错误resume的结构兼容metadata。"""

    ppo_cfg = ppo_update.ppo_training_config_status(ctx.cfg)
    if not ppo_cfg.can_run_optimizer_step or ppo_cfg.max_grad_norm is None:
        raise RuntimeError(f"PPO update超参数缺失：{ppo_cfg.missing_required_for_step}")
    network_cfg = ppo_trigger_policy.ppo_network_config_from_project(ctx.cfg)
    what_policy = getattr(policy, "sub_policy", policy)
    compatibility = {
        "action": {
            "labels": list(ctx.action_spec.labels),
            "action_dim": int(ctx.action_spec.action_dim),
            "cache_node_ids": list(ctx.action_spec.cache_node_ids),
            "uav_ids": list(ctx.action_spec.uav_ids),
            "ordering": "cache_triggers_then_uav_movement_triggers",
        },
        "observation": {
            "observation_dim": int(ctx.observation_spec.observation_dim),
            "cache_node_ids": list(ctx.observation_spec.cache_node_ids),
            "service_ids": list(ctx.observation_spec.service_ids),
            "uav_ids": list(ctx.observation_spec.uav_ids),
            "vehicle_ids": list(ctx.observation_spec.vehicle_ids),
        },
        "formal_protocol": {
            "data_protocol_schema": "ppo_train_test_origin_split_v1",
            "train_origin_range": [
                int(protocol.origin_split.train_origin_start),
                int(protocol.origin_split.train_origin_end),
            ],
            "test_origin_range": [
                int(protocol.origin_split.test_origin_start),
                int(protocol.origin_split.test_origin_end),
            ],
            "train_origin_start": int(protocol.origin_split.train_origin_start),
            "train_origin_end": int(protocol.origin_split.train_origin_end),
            "test_origin_start": int(protocol.origin_split.test_origin_start),
            "test_origin_end": int(protocol.origin_split.test_origin_end),
            "test_steps_per_evaluation": int(protocol.test_steps_per_evaluation),
            "steps_per_episode": int(protocol.steps_per_episode),
            "first_origin": int(protocol.episode_origin_indices[0]),
            "last_origin": int(protocol.episode_origin_indices[-1]),
            "first_slot": int(protocol.episode_slots[0]),
            "last_slot": int(protocol.episode_slots[-1]),
            "first_time_s": float(protocol.episode_times_s[0]),
            "last_time_s": float(protocol.episode_times_s[-1]),
            "update_epochs": int(protocol.update_epochs),
            "num_minibatches": int(protocol.num_minibatches),
            "minibatch_size": int(protocol.minibatch_size),
            "test_true_terminal_origin": int(protocol.test_true_terminal_origin),
            "test_true_terminal_slot": int(protocol.test_true_terminal_slot),
            "test_true_terminal_time_s": float(protocol.test_true_terminal_time_s),
        },
        "ppo_hyperparameters": {
            "gamma": float(ctx.gamma),
            "gae_lambda": float(ctx.gae_lambda),
            "clip_epsilon": float(ppo_cfg.clip_epsilon),
            "learning_rate": float(ppo_cfg.learning_rate),
            "value_loss_coefficient": float(ppo_cfg.value_loss_coefficient),
            "entropy_coefficient": float(ppo_cfg.entropy_coefficient),
            "max_grad_norm": float(ppo_cfg.max_grad_norm),
            "update_epochs": int(protocol.update_epochs),
            "num_minibatches": int(protocol.num_minibatches),
        },
        "experiment_seed": int(project_config.get_config_value(ctx.cfg, "experiment.seed")),
        "initial_cache": {
            "source": protocol.initial_cache_source,
            "seed": int(ctx.cache_cfg.initial_cache_seed),
            "sha256": hashlib.sha256(
                np.asarray(ctx.initial_cache, dtype=np.int8).tobytes()
            ).hexdigest(),
        },
        "policy_network": {
            **ppo_trigger_policy.actor_architecture_metadata(what_policy),
            "observation_dim": int(getattr(what_policy, "observation_dim")),
            "hidden_sizes": [int(value) for value in network_cfg["hidden_sizes"]],
            "autoregressive_hidden_size": int(network_cfg["autoregressive_hidden_size"]),
        },
    }
    if _is_hierarchical_gate_mode(ctx.cfg):
        if not isinstance(
            policy,
            hierarchical_reconfiguration_gate.HierarchicalReconfigurationGatePolicy,
        ):
            raise TypeError("hierarchical Gate compatibility metadata需要分层policy。")
        configured_cap = float(
            project_config.get_config_value(
                ctx.cfg,
                "algorithm.ppo.reference_probability_drift_cap",
                0.005,
            )
        )
        if configured_cap != 0.005:
            raise ValueError("hierarchical Gate formal implementation固定要求reference drift cap=0.005。")
        compatibility["hierarchical_gate"] = hierarchical_reconfiguration_gate.hierarchical_policy_metadata(policy)
        compatibility["algorithm"] = {
            "method_label": hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_METHOD,
            "method_id": hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_METHOD,
            "credit_assignment": ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT,
            "algorithm_schema": hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_ALGORITHM_SCHEMA,
            "actor_architecture_id": ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID,
            "cf_samples_per_episode": _cf_samples_per_episode(ctx.cfg),
            "gate_cf_samples_per_episode": HIERARCHICAL_GATE_CF_SAMPLES_PER_EPISODE,
            "cf_batch_episodes": HIERARCHICAL_GATE_CF_BATCH_EPISODES,
            "batch_sample_count": _cf_samples_per_episode(ctx.cfg) * HIERARCHICAL_GATE_CF_BATCH_EPISODES,
            "gate_batch_sample_count": HIERARCHICAL_GATE_CF_SAMPLES_PER_EPISODE * HIERARCHICAL_GATE_CF_BATCH_EPISODES,
            "counterfactual_source": "same_state_frozen_hierarchical_policy_joint_null_continuation",
            "policy_update_semantics": "Gate_first_then_existing_WHAT_Direct_CF_after_four_frozen_policy_episodes",
            "actor_loss": "existing_sparse_per_trigger_direct_cf_clipped_ppo",
            "actor_credit": "A_cf=(a-p_old)*Delta_G",
            "gate_loss": "clipped_ppo_on_A_gate",
            "gate_credit": "A_gate=(g-p_old_gate)*Delta_gate",
            "gate_refdrift": "excluded",
            "learned_critic_for_actor_credit": False,
            "value_head_in_actor_optimizer": False,
            "value_loss_in_actor_update": False,
            "cf_sampling_strategy": _cf_sampling_strategy(ctx.cfg),
            "gate_sampling_strategy": "uniform_slots",
            "gate_counterfactual_rng": "independent_numpy_generator",
            "counterfactual_rng": "independent_numpy_generator",
            "checkpoint_save_semantics": "episode_atomic_with_pending_batch",
            "causal_artifact_schema": causal_artifacts.CAUSAL_ARTIFACT_SCHEMA_ID,
            "hierarchical_gate_artifact_schema": causal_artifacts.HIERARCHICAL_GATE_ARTIFACT_SCHEMA_ID,
            "reference_probability_drift_cap": 0.005,
            "reference_constraint_method": reference_drift.REFERENCE_DRIFT_CONSTRAINT_METHOD,
            "reference_constraint_version": reference_drift.REFERENCE_DRIFT_CONSTRAINT_VERSION,
            "reference_gate": "WHAT_unselected_current_batch_masked_conditional_probability_backtracking_bisection",
            "optimizer_isolation": "independent_gate_and_what_optimizers",
        }
    elif _is_counterfactual_mode(ctx.cfg):
        q_target = ppo_update.q_target_from_project(ctx.cfg)
        if q_target != ppo_update.MONTE_CARLO_Q_TARGET:
            raise ValueError(
                "per_trigger_q 模式第一版只支持 algorithm.ppo.q_target=monte_carlo。"
            )
        compatibility["algorithm"] = {
            "method_label": "proposed_per_trigger_q",
            "credit_assignment": ppo_update.PER_TRIGGER_Q_CREDIT_ASSIGNMENT,
            "q_target": ppo_update.MONTE_CARLO_Q_TARGET,
            "q_critic_schema": "per_trigger_q_v1",
            "gradient_isolation": "detach_conditional_hidden",
            "actor_loss": "per_trigger_conditional_ppo",
        }
    elif _is_centralized_mode(ctx.cfg):
        q_target = ppo_update.q_target_from_project(ctx.cfg)
        if q_target != ppo_update.MONTE_CARLO_Q_TARGET:
            raise ValueError(
                "centralized_counterfactual_q 模式只支持 algorithm.ppo.q_target=monte_carlo。"
            )
        compatibility["algorithm"] = {
            "method_label": "proposed_centralized_cf_q",
            "credit_assignment": ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT,
            "q_target": ppo_update.MONTE_CARLO_Q_TARGET,
            "credit_phase": ppo_update.credit_phase_from_project(ctx.cfg),
            "critic_return_scale": float(protocol.critic_return_scale),
            "q_critic_schema": "centralized_joint_action_q_v2",
            "q_input": "observation_plus_full_joint_action",
            "q_decomposition": "V_c_observation_plus_D_c_observation_action",
            "gradient_isolation": "independent_critic_and_frozen_counterfactual_targets",
            "actor_loss": "per_trigger_conditional_ppo",
        }
    elif _is_sparse_direct_cf_mode(ctx.cfg):
        if _is_reference_drift_mode(ctx.cfg):
            compatibility["algorithm"] = {
                "method_label": ppo_update.REFERENCE_DRIFT_FROZEN_POLICY_BATCH_METHOD,
                "method_id": ppo_update.REFERENCE_DRIFT_FROZEN_POLICY_BATCH_METHOD,
                "credit_assignment": ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT,
                "algorithm_schema": ppo_update.REFERENCE_DRIFT_ALGORITHM_SCHEMA,
                "actor_architecture_id": ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID,
                "cf_samples_per_episode": _cf_samples_per_episode(ctx.cfg),
                "cf_batch_episodes": _cf_batch_episodes(ctx.cfg),
                "batch_sample_count": _cf_samples_per_episode(ctx.cfg) * _cf_batch_episodes(ctx.cfg),
                "counterfactual_source": "same_state_frozen_collection_policy_continuation",
                "policy_update_semantics": "one_full_batch_update_after_four_frozen_policy_episodes_then_reference_drift_gate",
                "actor_loss": "sparse_per_trigger_direct_cf_clipped_ppo",
                "actor_credit": "A_cf=(a-p_old)*Delta_G",
                "learned_critic_for_actor_credit": False,
                "value_head_in_actor_optimizer": False,
                "value_loss_in_actor_update": False,
                "cf_sampling_strategy": _cf_sampling_strategy(ctx.cfg),
                "reservoir_sampling": (
                    "uniform_without_replacement_over_free_decisions"
                    if _cf_sampling_strategy(ctx.cfg) == "uniform"
                    else "one_capacity_one_reservoir_per_trigger_over_free_decisions"
                ),
                "counterfactual_rng": "independent_numpy_generator",
                "checkpoint_save_semantics": "episode_atomic_with_pending_batch",
                "causal_artifact_schema": causal_artifacts.CAUSAL_ARTIFACT_SCHEMA_ID,
                "reference_probability_drift_cap": 0.005,
                "reference_constraint_method": reference_drift.REFERENCE_DRIFT_CONSTRAINT_METHOD,
                "reference_constraint_version": reference_drift.REFERENCE_DRIFT_CONSTRAINT_VERSION,
                "reference_gate": "unselected_current_batch_masked_conditional_probability_backtracking_bisection",
                "adam_state_semantics": "single_raw_adam_step_post_state_preserved_after_parameter_displacement_scaling",
            }
        elif _is_frozen_batch_family(ctx.cfg):
            compatibility["algorithm"] = {
                "method_label": ppo_update.FROZEN_POLICY_BATCH_METHOD,
                "method_id": ppo_update.FROZEN_POLICY_BATCH_METHOD,
                "credit_assignment": ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT,
                "algorithm_schema": "proposed_sparse_direct_cf_frozen_policy_batch_v1",
                "actor_architecture_id": ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID,
                "cf_samples_per_episode": _cf_samples_per_episode(ctx.cfg),
                "cf_batch_episodes": _cf_batch_episodes(ctx.cfg),
                "batch_sample_count": _cf_samples_per_episode(ctx.cfg) * _cf_batch_episodes(ctx.cfg),
                "counterfactual_source": "same_state_frozen_collection_policy_continuation",
                "policy_update_semantics": "one_full_batch_update_after_four_frozen_policy_episodes",
                "actor_loss": "sparse_per_trigger_direct_cf_clipped_ppo",
                "actor_credit": "A_cf=(a-p_old)*Delta_G",
                "learned_critic_for_actor_credit": False,
                "value_head_in_actor_optimizer": False,
                "value_loss_in_actor_update": False,
                "cf_sampling_strategy": _cf_sampling_strategy(ctx.cfg),
                "reservoir_sampling": (
                    "uniform_without_replacement_over_free_decisions"
                    if _cf_sampling_strategy(ctx.cfg) == "uniform"
                    else "one_capacity_one_reservoir_per_trigger_over_free_decisions"
                ),
                "counterfactual_rng": "independent_numpy_generator",
                "checkpoint_save_semantics": "batch_boundary_only",
            }
        else:
            compatibility["algorithm"] = {
                "method_label": "proposed_sparse_direct_cf_ppo",
                "credit_assignment": ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT,
                "algorithm_schema": "proposed_sparse_direct_cf_ppo_v3",
                "cf_samples_per_episode": _cf_samples_per_episode(ctx.cfg),
                "counterfactual_source": "same_state_frozen_old_policy_continuation",
                "actor_loss": "sparse_per_trigger_direct_cf_clipped_ppo",
                "actor_credit": "A_cf=(a-p_old)*Delta_G",
                "learned_critic_for_actor_credit": False,
                "value_head_in_actor_optimizer": False,
                "value_loss_in_actor_update": False,
                "cf_sampling_strategy": _cf_sampling_strategy(ctx.cfg),
                "reservoir_sampling": (
                    "uniform_without_replacement_over_free_decisions"
                    if _cf_sampling_strategy(ctx.cfg) == "uniform"
                    else "one_capacity_one_reservoir_per_trigger_over_free_decisions"
                ),
                "counterfactual_rng": "independent_numpy_generator",
            }
    return compatibility


def _metadata_mismatch(expected: Any, actual: Any, path: str = "metadata") -> str | None:
    """返回第一个兼容性差异；浮点参数使用极小容差比较。"""

    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return f"{path} keys不一致：expected={sorted(expected)}, actual={sorted(actual)}"
        for key in expected:
            mismatch = _metadata_mismatch(expected[key], actual[key], f"{path}.{key}")
            if mismatch:
                return mismatch
        return None
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return f"{path}长度不一致：expected={len(expected)}, actual={len(actual)}"
        for index, (left, right) in enumerate(zip(expected, actual)):
            mismatch = _metadata_mismatch(left, right, f"{path}[{index}]")
            if mismatch:
                return mismatch
        return None
    if isinstance(expected, (int, float)) and not isinstance(expected, bool) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if not math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=1e-12):
            return f"{path}不一致：expected={expected}, actual={actual}"
        return None
    if expected != actual:
        return f"{path}不一致：expected={expected}, actual={actual}"
    return None


def _checkpoint_payload(
    policy: Any,
    optimizer: Any,
    policy_version: int,
    optimizer_step_count: int,
    history: dict[str, Any],
    compatibility: dict[str, Any],
    rng_states: dict[str, Any],
    credit_assignment: str = ppo_update.LEGACY_CREDIT_ASSIGNMENT,
    q_critic: Any | None = None,
    q_optimizer: Any | None = None,
    q_optimizer_step_count: int = 0,
    critic_replay: ppo_rollout.CriticReplayBuffer | None = None,
    cf_sampling_rng: np.random.Generator | None = None,
    pending_direct_cf_samples: list[ppo_rollout.DirectCFCreditSample] | None = None,
    pending_reference_records: list[ppo_rollout.ReferenceStateRecord] | None = None,
    gate_optimizer: Any | None = None,
    gate_sampling_rng: np.random.Generator | None = None,
    pending_gate_cf_samples: list[Any] | None = None,
    gate_policy_version: int | None = None,
    gate_optimizer_step_count: int | None = None,
) -> dict[str, Any]:
    """构造不含RolloutBatch/梯度/候选trace的权威checkpoint。"""

    if credit_assignment == ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT:
        if not isinstance(
            policy,
            hierarchical_reconfiguration_gate.HierarchicalReconfigurationGatePolicy,
        ):
            raise TypeError("hierarchical Gate checkpoint需要分层policy。")
        if gate_optimizer is None or gate_sampling_rng is None:
            raise ValueError("hierarchical Gate checkpoint缺少Gate optimizer/RNG。")
        if not isinstance(gate_sampling_rng, np.random.Generator):
            raise TypeError("hierarchical Gate gate_sampling_rng必须是独立np.random.Generator。")
        if not isinstance(cf_sampling_rng, np.random.Generator):
            raise TypeError("hierarchical WHAT cf_sampling_rng必须是独立np.random.Generator。")
        if compatibility.get("algorithm", {}).get("method_id") != hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_METHOD:
            raise ValueError("hierarchical Gate checkpoint compatibility method_id不一致。")
        pending_samples = [] if pending_direct_cf_samples is None else pending_direct_cf_samples
        pending_gate = [] if pending_gate_cf_samples is None else pending_gate_cf_samples
        pending_references = [] if pending_reference_records is None else pending_reference_records
        if int(history.get("pending_cf_sample_count", -1)) != len(pending_samples):
            raise ValueError("hierarchical Gate checkpoint pending WHAT count不一致。")
        if int(history.get("pending_gate_cf_sample_count", -1)) != len(pending_gate):
            raise ValueError("hierarchical Gate checkpoint pending Gate count不一致。")
        if int(history.get("pending_reference_state_count", -1)) != len(pending_references):
            raise ValueError("hierarchical Gate checkpoint pending reference count不一致。")
        gate_version = int(policy_version if gate_policy_version is None else gate_policy_version)
        gate_steps = int(optimizer_step_count if gate_optimizer_step_count is None else gate_optimizer_step_count)
        if gate_version != int(policy_version) or gate_steps != int(optimizer_step_count):
            raise ValueError("hierarchical Gate checkpoint要求WHAT/Gate policy与optimizer counters同步。")
        return {
            "checkpoint_version": HIERARCHICAL_GATE_CHECKPOINT_VERSION,
            "algorithm_schema": hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_ALGORITHM_SCHEMA,
            "method_id": hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_METHOD,
            "architecture_id": ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID,
            "hierarchical_gate_artifact_schema": causal_artifacts.HIERARCHICAL_GATE_ARTIFACT_SCHEMA_ID,
            "cf_sampling_strategy": str(compatibility["algorithm"].get("cf_sampling_strategy", "uniform")),
            "gate_sampling_strategy": "uniform_slots",
            "cf_samples_per_episode": int(history["cf_samples_per_episode"]),
            "gate_cf_samples_per_episode": HIERARCHICAL_GATE_CF_SAMPLES_PER_EPISODE,
            "cf_batch_episodes": HIERARCHICAL_GATE_CF_BATCH_EPISODES,
            "batch_sample_count": int(compatibility["algorithm"]["batch_sample_count"]),
            "gate_batch_sample_count": int(compatibility["algorithm"]["gate_batch_sample_count"]),
            "update_epochs": int(compatibility["formal_protocol"]["update_epochs"]),
            "batch_id": int(history.get("batch_id", 0)),
            "completed_batch_count": int(history.get("causal_update_artifact_index", optimizer_step_count)),
            "causal_update_artifact_index": int(history.get("causal_update_artifact_index", optimizer_step_count)),
            "batch_episode_index": int(history.get("batch_episode_index", 0)),
            "completed_episodes": int(history["completed_episodes"]),
            "pending_cf_samples": [ppo_rollout.direct_cf_sample_to_checkpoint_dict(item) for item in pending_samples],
            "pending_cf_sample_count": len(pending_samples),
            "pending_gate_cf_samples": [
                hierarchical_reconfiguration_gate.gate_counterfactual_sample_to_dict(item)
                for item in pending_gate
            ],
            "pending_gate_cf_sample_count": len(pending_gate),
            "pending_reference_states": [
                ppo_rollout.reference_state_to_checkpoint_dict(item) for item in pending_references
            ],
            "pending_reference_state_count": len(pending_references),
            "pending_collection_policy_version": history.get("pending_collection_policy_version"),
            "reference_probability_drift_cap": 0.005,
            "reference_constraint_method": reference_drift.REFERENCE_DRIFT_CONSTRAINT_METHOD,
            "reference_constraint_version": reference_drift.REFERENCE_DRIFT_CONSTRAINT_VERSION,
            "causal_artifact_schema": causal_artifacts.CAUSAL_ARTIFACT_SCHEMA_ID,
            "checkpoint_save_semantics": "episode_atomic_with_pending_batch",
            "policy_state_dict": policy.state_dict(),
            "actor_policy_state_dict": policy.sub_policy.state_dict(),
            "gate_state_dict": policy.gate.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "gate_optimizer_state_dict": gate_optimizer.state_dict(),
            "policy_version": int(policy_version),
            "actor_policy_version": int(policy_version),
            "gate_policy_version": gate_version,
            "optimizer_step_count": int(optimizer_step_count),
            "actor_optimizer_step_count": int(optimizer_step_count),
            "gate_optimizer_step_count": gate_steps,
            "training_history": summary_to_builtin(history),
            "compatibility": summary_to_builtin(compatibility),
            "rng_states": rng_states,
            "actor_sampling_rng_state": copy.deepcopy({
                key: rng_states.get(key)
                for key in (
                    "python_random_state",
                    "numpy_random_state",
                    "torch_cpu_rng_state",
                    "torch_cuda_rng_state_all",
                )
            }),
            "cf_sampling_rng_state": (
                None
                if cf_sampling_rng is None
                else copy.deepcopy(cf_sampling_rng.bit_generator.state)
            ),
            "gate_sampling_rng_state": copy.deepcopy(gate_sampling_rng.bit_generator.state),
            "episode_atomic_commit": True,
        }
    if (
        credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT
        and compatibility.get("algorithm", {}).get("method_id")
        == ppo_update.REFERENCE_DRIFT_FROZEN_POLICY_BATCH_METHOD
    ):
        if cf_sampling_rng is None:
            raise ValueError("refdrift checkpoint缺少独立cf_sampling_rng。")
        pending_samples = [] if pending_direct_cf_samples is None else pending_direct_cf_samples
        pending_references = [] if pending_reference_records is None else pending_reference_records
        cap = float(compatibility["algorithm"].get("reference_probability_drift_cap", -1.0))
        if cap != 0.005:
            raise ValueError("refdrift checkpoint固定要求reference_probability_drift_cap=0.005。")
        if int(history.get("pending_cf_sample_count", -1)) != len(pending_samples):
            raise ValueError("refdrift checkpoint pending_cf_sample_count不一致。")
        if int(history.get("pending_reference_state_count", -1)) != len(pending_references):
            raise ValueError("refdrift checkpoint pending_reference_state_count不一致。")
        return {
            "checkpoint_version": REFERENCE_DRIFT_CHECKPOINT_VERSION,
            "algorithm_schema": ppo_update.REFERENCE_DRIFT_ALGORITHM_SCHEMA,
            "method_id": ppo_update.REFERENCE_DRIFT_FROZEN_POLICY_BATCH_METHOD,
            "architecture_id": ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID,
            "cf_sampling_strategy": str(
                compatibility["algorithm"].get("cf_sampling_strategy", "uniform")
            ),
            "cf_samples_per_episode": int(history["cf_samples_per_episode"]),
            "cf_batch_episodes": int(history["cf_batch_episodes"]),
            "batch_sample_count": int(
                history["cf_samples_per_episode"] * history["cf_batch_episodes"]
            ),
            "update_epochs": int(compatibility["formal_protocol"]["update_epochs"]),
            "batch_id": int(history.get("batch_id", 0)),
            "completed_batch_count": int(
                history.get("causal_update_artifact_index", history.get("optimizer_step_count", 0))
            ),
            "causal_update_artifact_index": int(
                history.get("causal_update_artifact_index", history.get("optimizer_step_count", 0))
            ),
            "batch_episode_index": int(history.get("batch_episode_index", 0)),
            "completed_episodes": int(history["completed_episodes"]),
            "pending_cf_samples": [
                ppo_rollout.direct_cf_sample_to_checkpoint_dict(item)
                for item in pending_samples
            ],
            "pending_cf_sample_count": len(pending_samples),
            "pending_reference_states": [
                ppo_rollout.reference_state_to_checkpoint_dict(item)
                for item in pending_references
            ],
            "pending_reference_state_count": len(pending_references),
            "pending_collection_policy_version": history.get(
                "pending_collection_policy_version"
            ),
            "reference_probability_drift_cap": cap,
            "reference_constraint_method": reference_drift.REFERENCE_DRIFT_CONSTRAINT_METHOD,
            "reference_constraint_version": reference_drift.REFERENCE_DRIFT_CONSTRAINT_VERSION,
            "causal_artifact_schema": causal_artifacts.CAUSAL_ARTIFACT_SCHEMA_ID,
            "checkpoint_save_semantics": "episode_atomic_with_pending_batch",
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "policy_version": int(policy_version),
            "optimizer_step_count": int(optimizer_step_count),
            "training_history": summary_to_builtin(history),
            "compatibility": summary_to_builtin(compatibility),
            "rng_states": rng_states,
            "actor_sampling_rng_state": copy.deepcopy({
                key: rng_states.get(key)
                for key in (
                    "python_random_state",
                    "numpy_random_state",
                    "torch_cpu_rng_state",
                    "torch_cuda_rng_state_all",
                )
            }),
            "cf_sampling_rng_state": copy.deepcopy(cf_sampling_rng.bit_generator.state),
            "episode_atomic_commit": True,
        }
    if (
        credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT
        and compatibility.get("algorithm", {}).get("method_id")
        == ppo_update.FROZEN_POLICY_BATCH_METHOD
    ):
        if cf_sampling_rng is None:
            raise ValueError("frozen_policy_batch checkpoint缺少独立cf_sampling_rng。")
        if int(history.get("pending_cf_sample_count", 0)) != 0:
            raise ValueError("frozen_policy_batch checkpoint只能在batch boundary保存。")
        if int(history.get("batch_episode_index", 0)) != int(history.get("cf_batch_episodes", 0)):
            raise ValueError("frozen_policy_batch checkpoint必须位于完整batch boundary。")
        return {
            "checkpoint_version": FROZEN_POLICY_BATCH_CHECKPOINT_VERSION,
            "algorithm_schema": "proposed_sparse_direct_cf_frozen_policy_batch_v1",
            "method_id": ppo_update.FROZEN_POLICY_BATCH_METHOD,
            "architecture_id": ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID,
            "cf_sampling_strategy": str(
                compatibility["algorithm"].get("cf_sampling_strategy", "uniform")
            ),
            "cf_samples_per_episode": int(history["cf_samples_per_episode"]),
            "cf_batch_episodes": int(history["cf_batch_episodes"]),
            "batch_sample_count": int(
                history["cf_samples_per_episode"] * history["cf_batch_episodes"]
            ),
            "update_epochs": int(compatibility["formal_protocol"]["update_epochs"]),
            "batch_id": int(history.get("batch_id", 0)),
            "completed_batch_count": int(history.get("batch_id", 0)),
            "batch_episode_index": int(history.get("batch_episode_index", 0)),
            "completed_episodes": int(history["completed_episodes"]),
            "pending_cf_samples": [],
            "pending_cf_sample_count": 0,
            "pending_collection_policy_version": None,
            "checkpoint_save_semantics": "batch_boundary_only",
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "policy_version": int(policy_version),
            "optimizer_step_count": int(optimizer_step_count),
            "training_history": summary_to_builtin(history),
            "compatibility": summary_to_builtin(compatibility),
            "rng_states": rng_states,
            "actor_sampling_rng_state": copy.deepcopy({
                key: rng_states.get(key)
                for key in (
                    "python_random_state",
                    "numpy_random_state",
                    "torch_cpu_rng_state",
                    "torch_cuda_rng_state_all",
                )
            }),
            "cf_sampling_rng_state": copy.deepcopy(cf_sampling_rng.bit_generator.state),
            "episode_atomic_commit": True,
        }
    if credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT:
        if cf_sampling_rng is None:
            raise ValueError("sparse_direct_cf checkpoint缺少独立cf_sampling_rng。")
        return {
            "checkpoint_version": SPARSE_DIRECT_CF_CHECKPOINT_VERSION,
            "algorithm_schema": "proposed_sparse_direct_cf_ppo_v3",
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "policy_version": int(policy_version),
            "optimizer_step_count": int(optimizer_step_count),
            "completed_episodes": int(history["completed_episodes"]),
            "cf_sampling_strategy": str(
                compatibility["algorithm"].get("cf_sampling_strategy", "uniform")
            ),
            "training_history": summary_to_builtin(history),
            "compatibility": summary_to_builtin(compatibility),
            "rng_states": rng_states,
            "actor_sampling_rng_state": copy.deepcopy({
                key: rng_states.get(key)
                for key in (
                    "python_random_state",
                    "numpy_random_state",
                    "torch_cpu_rng_state",
                    "torch_cuda_rng_state_all",
                )
            }),
            "cf_sampling_rng_state": copy.deepcopy(cf_sampling_rng.bit_generator.state),
            "episode_atomic_commit": True,
        }
    if credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT:
        if q_critic is None or q_optimizer is None or critic_replay is None:
            raise ValueError("centralized counterfactual checkpoint 缺少 critic/replay。")
        return {
            "checkpoint_version": CENTRALIZED_COUNTERFACTUAL_CHECKPOINT_VERSION,
            "algorithm_schema": "proposed_centralized_cf_q_v2",
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "critic_state_dict": q_critic.state_dict(),
            "critic_optimizer_state_dict": q_optimizer.state_dict(),
            "critic_replay_state": critic_replay.state_dict(),
            "policy_version": int(policy_version),
            "optimizer_step_count": int(optimizer_step_count),
            "critic_update_count": int(history.get("critic_update_count", history["completed_episodes"])),
            "critic_optimizer_step_count": int(q_optimizer_step_count),
            "completed_episodes": int(history["completed_episodes"]),
            "critic_return_scale": float(compatibility["algorithm"]["critic_return_scale"]),
            "training_history": summary_to_builtin(history),
            "compatibility": summary_to_builtin(compatibility),
            "rng_states": rng_states,
        }
    if credit_assignment == ppo_update.PER_TRIGGER_Q_CREDIT_ASSIGNMENT:
        if q_critic is None or q_optimizer is None:
            raise ValueError("per_trigger_q checkpoint 缺少 q_critic/q_optimizer。")
        return {
            "checkpoint_version": COUNTERFACTUAL_CHECKPOINT_VERSION,
            "algorithm_schema": "proposed_per_trigger_q_v1",
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "q_critic_state_dict": q_critic.state_dict(),
            "q_optimizer_state_dict": q_optimizer.state_dict(),
            "policy_version": int(policy_version),
            "optimizer_step_count": int(optimizer_step_count),
            "q_optimizer_step_count": int(q_optimizer_step_count),
            "completed_episodes": int(history["completed_episodes"]),
            "training_history": summary_to_builtin(history),
            "compatibility": summary_to_builtin(compatibility),
            "rng_states": rng_states,
        }
    if credit_assignment != ppo_update.LEGACY_CREDIT_ASSIGNMENT:
        raise ValueError(f"未知 credit_assignment：{credit_assignment}")
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "policy_version": int(policy_version),
        "optimizer_step_count": int(optimizer_step_count),
        "completed_episodes": int(history["completed_episodes"]),
        "training_history": summary_to_builtin(history),
        "compatibility": summary_to_builtin(compatibility),
        "rng_states": rng_states,
    }


def _load_formal_checkpoint(
    checkpoint_path: Path,
    policy: Any,
    optimizer: Any,
    rng_generator: Any,
    expected_compatibility: dict[str, Any],
    credit_assignment: str = ppo_update.LEGACY_CREDIT_ASSIGNMENT,
    q_critic: Any | None = None,
    q_optimizer: Any | None = None,
    critic_replay: ppo_rollout.CriticReplayBuffer | None = None,
    critic_rng_generator: Any | None = None,
    cf_sampling_rng: np.random.Generator | None = None,
    gate_optimizer: Any | None = None,
    gate_sampling_rng: np.random.Generator | None = None,
    hierarchical_state: dict[str, Any] | None = None,
) -> tuple[
    dict[str, Any],
    int,
    int,
    int,
    list[ppo_rollout.DirectCFCreditSample],
    list[ppo_rollout.ReferenceStateRecord],
]:
    """按policy/optimizer构造完成后加载checkpoint并恢复RNG。"""

    _require_torch()
    try:
        try:
            payload = torch.load(
                checkpoint_path,
                map_location=_policy_device(policy),
                weights_only=False,
            )
        except TypeError:  # 兼容较旧PyTorch的torch.load签名
            payload = torch.load(checkpoint_path, map_location=_policy_device(policy))
    except Exception as exc:
        raise RuntimeError(f"无法读取checkpoint：{checkpoint_path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: checkpoint顶层必须是dict。")
    reference_drift_checkpoint = (
        credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT
        and expected_compatibility.get("algorithm", {}).get("method_id")
        == ppo_update.REFERENCE_DRIFT_FROZEN_POLICY_BATCH_METHOD
    )
    hierarchical_gate_checkpoint = (
        credit_assignment == ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT
        and expected_compatibility.get("algorithm", {}).get("method_id")
        == hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_METHOD
    )
    frozen_policy_batch_checkpoint = (
        credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT
        and expected_compatibility.get("algorithm", {}).get("method_id")
        == ppo_update.FROZEN_POLICY_BATCH_METHOD
    )
    if hierarchical_gate_checkpoint:
        expected_checkpoint_version = HIERARCHICAL_GATE_CHECKPOINT_VERSION
    elif reference_drift_checkpoint:
        expected_checkpoint_version = REFERENCE_DRIFT_CHECKPOINT_VERSION
    elif frozen_policy_batch_checkpoint:
        expected_checkpoint_version = FROZEN_POLICY_BATCH_CHECKPOINT_VERSION
    elif credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT:
        expected_checkpoint_version = SPARSE_DIRECT_CF_CHECKPOINT_VERSION
    elif credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT:
        expected_checkpoint_version = CENTRALIZED_COUNTERFACTUAL_CHECKPOINT_VERSION
    elif credit_assignment == ppo_update.PER_TRIGGER_Q_CREDIT_ASSIGNMENT:
        expected_checkpoint_version = COUNTERFACTUAL_CHECKPOINT_VERSION
    else:
        expected_checkpoint_version = CHECKPOINT_VERSION
    if int(payload.get("checkpoint_version", -1)) != expected_checkpoint_version:
        raise RuntimeError(
            "CHECKPOINT_COMPATIBILITY_ERROR: checkpoint与当前credit_assignment/schema不兼容。"
        )
    if hierarchical_gate_checkpoint:
        if payload.get("algorithm_schema") != hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_ALGORITHM_SCHEMA:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical Gate algorithm schema不一致。")
        if payload.get("method_id") != hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_METHOD:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical Gate method_id不一致。")
        if payload.get("architecture_id") != ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical Gate WHAT architecture不一致。")
        if payload.get("causal_artifact_schema") != causal_artifacts.CAUSAL_ARTIFACT_SCHEMA_ID:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical Gate causal schema不一致。")
        if payload.get("hierarchical_gate_artifact_schema") != causal_artifacts.HIERARCHICAL_GATE_ARTIFACT_SCHEMA_ID:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical Gate artifact schema不一致。")
        if payload.get("checkpoint_save_semantics") != "episode_atomic_with_pending_batch":
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical Gate checkpoint语义不一致。")
        if int(payload.get("cf_batch_episodes", -1)) != HIERARCHICAL_GATE_CF_BATCH_EPISODES:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical Gate cf_batch_episodes不一致。")
        if int(payload.get("gate_cf_samples_per_episode", -1)) != HIERARCHICAL_GATE_CF_SAMPLES_PER_EPISODE:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical Gate gate K不一致。")
        if int(payload.get("cf_samples_per_episode", -1)) != int(
            expected_compatibility["algorithm"].get("cf_samples_per_episode", -2)
        ):
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical Gate WHAT K不一致。")
        if int(payload.get("gate_batch_sample_count", -1)) != int(
            expected_compatibility["algorithm"].get("gate_batch_sample_count", -2)
        ):
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical Gate batch sample count不一致。")
        if float(payload.get("reference_probability_drift_cap", -1.0)) != 0.005:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical Gate drift cap不一致。")
        if gate_optimizer is None or gate_sampling_rng is None:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical Gate缺少Gate optimizer/RNG。")
        if not isinstance(gate_sampling_rng, np.random.Generator):
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: gate_sampling_rng类型错误。")
        if cf_sampling_rng is None or not isinstance(cf_sampling_rng, np.random.Generator):
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical WHAT cf_sampling_rng缺失。")
        for field_name in (
            "cf_sampling_rng_state",
            "gate_sampling_rng_state",
            "actor_sampling_rng_state",
            "gate_optimizer_state_dict",
            "gate_state_dict",
            "pending_gate_cf_samples",
            "pending_gate_cf_sample_count",
        ):
            if field_name not in payload:
                raise RuntimeError(
                    f"CHECKPOINT_COMPATIBILITY_ERROR: hierarchical Gate checkpoint缺少{field_name}。"
                )
    elif reference_drift_checkpoint:
        if payload.get("algorithm_schema") != ppo_update.REFERENCE_DRIFT_ALGORITHM_SCHEMA:
            raise RuntimeError(
                "CHECKPOINT_COMPATIBILITY_ERROR: 缺少refdrift algorithm schema。"
            )
        if payload.get("causal_artifact_schema") != causal_artifacts.CAUSAL_ARTIFACT_SCHEMA_ID:
            raise RuntimeError(
                "CHECKPOINT_COMPATIBILITY_ERROR: 缺少causal artifact schema。"
            )
        if payload.get("method_id") != ppo_update.REFERENCE_DRIFT_FROZEN_POLICY_BATCH_METHOD:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: refdrift method_id不一致。")
        if payload.get("architecture_id") != ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: refdrift architecture_id不一致。")
        if int(payload.get("cf_batch_episodes", -1)) != int(
            expected_compatibility["algorithm"].get("cf_batch_episodes", -2)
        ):
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: refdrift cf_batch_episodes不一致。")
        if int(payload.get("cf_samples_per_episode", -1)) != int(
            expected_compatibility["algorithm"].get("cf_samples_per_episode", -2)
        ):
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: refdrift cf_samples_per_episode不一致。")
        if int(payload.get("batch_sample_count", -1)) != int(
            expected_compatibility["algorithm"].get("batch_sample_count", -2)
        ):
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: refdrift batch_sample_count不一致。")
        if float(payload.get("reference_probability_drift_cap", -1.0)) != 0.005:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: refdrift drift cap不一致。")
        if payload.get("reference_constraint_method") != reference_drift.REFERENCE_DRIFT_CONSTRAINT_METHOD:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: refdrift constraint method不一致。")
        if payload.get("reference_constraint_version") != reference_drift.REFERENCE_DRIFT_CONSTRAINT_VERSION:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: refdrift constraint version不一致。")
        if payload.get("checkpoint_save_semantics") != "episode_atomic_with_pending_batch":
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: refdrift checkpoint语义不一致。")
        if cf_sampling_rng is None:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: 当前模式缺少cf_sampling_rng。")
        if "cf_sampling_rng_state" not in payload:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: checkpoint缺少cf_sampling_rng_state。")
        if "actor_sampling_rng_state" not in payload:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: checkpoint缺少actor_sampling_rng_state。")
    elif frozen_policy_batch_checkpoint:
        if payload.get("algorithm_schema") != "proposed_sparse_direct_cf_frozen_policy_batch_v1":
            raise RuntimeError(
                "CHECKPOINT_COMPATIBILITY_ERROR: 缺少proposed_sparse_direct_cf_frozen_policy_batch_v1 schema。"
            )
        if payload.get("method_id") != ppo_update.FROZEN_POLICY_BATCH_METHOD:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: frozen_policy_batch method_id不一致。")
        if payload.get("architecture_id") != ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: frozen_policy_batch architecture_id不一致。")
        if int(payload.get("cf_batch_episodes", -1)) != int(
            expected_compatibility["algorithm"].get("cf_batch_episodes", -2)
        ):
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: cf_batch_episodes不一致。")
        if int(payload.get("cf_samples_per_episode", -1)) != int(
            expected_compatibility["algorithm"].get("cf_samples_per_episode", -2)
        ):
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: cf_samples_per_episode不一致。")
        if int(payload.get("batch_sample_count", -1)) != int(
            expected_compatibility["algorithm"].get("batch_sample_count", -2)
        ):
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: batch_sample_count不一致。")
        if int(payload.get("update_epochs", -1)) != int(
            expected_compatibility["formal_protocol"].get("update_epochs", -2)
        ):
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: update_epochs不一致。")
        if int(payload.get("pending_cf_sample_count", -1)) != 0 or payload.get("pending_cf_samples") != []:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: batch-boundary checkpoint包含pending samples。")
        if payload.get("checkpoint_save_semantics") != "batch_boundary_only":
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: checkpoint save semantics不一致。")
        if cf_sampling_rng is None:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: 当前模式缺少cf_sampling_rng。")
        if "cf_sampling_rng_state" not in payload:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: checkpoint缺少cf_sampling_rng_state。")
        if "actor_sampling_rng_state" not in payload:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: checkpoint缺少actor_sampling_rng_state。")
    elif credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT:
        if payload.get("algorithm_schema") != "proposed_sparse_direct_cf_ppo_v3":
            raise RuntimeError(
                "CHECKPOINT_COMPATIBILITY_ERROR: 缺少proposed_sparse_direct_cf_ppo_v3 schema。"
            )
        if cf_sampling_rng is None:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: 当前模式缺少cf_sampling_rng。")
        if "cf_sampling_rng_state" not in payload:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: checkpoint缺少cf_sampling_rng_state。")
        if "actor_sampling_rng_state" not in payload:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: checkpoint缺少actor_sampling_rng_state。")
    elif credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT:
        if payload.get("algorithm_schema") != "proposed_centralized_cf_q_v2":
            raise RuntimeError(
                "CHECKPOINT_COMPATIBILITY_ERROR: 缺少proposed_centralized_cf_q_v2 schema。"
            )
        if q_critic is None or q_optimizer is None or critic_replay is None:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: 当前模式缺少central critic/replay。")
        expected_scale = expected_compatibility.get("algorithm", {}).get("critic_return_scale")
        actual_scale = payload.get("critic_return_scale")
        if expected_scale is None or actual_scale is None:
            raise RuntimeError(
                "CHECKPOINT_COMPATIBILITY_ERROR: centralized checkpoint缺少critic_return_scale。"
            )
        if not math.isclose(float(actual_scale), float(expected_scale), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                "CHECKPOINT_COMPATIBILITY_ERROR: critic_return_scale与当前formal horizon不一致。"
            )
    elif credit_assignment == ppo_update.PER_TRIGGER_Q_CREDIT_ASSIGNMENT:
        if payload.get("algorithm_schema") != "proposed_per_trigger_q_v1":
            raise RuntimeError(
                "CHECKPOINT_COMPATIBILITY_ERROR: 缺少proposed_per_trigger_q_v1 schema。"
            )
        if q_critic is None or q_optimizer is None:
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: 当前模式缺少Q critic/optimizer。")
    actual_compatibility = payload.get("compatibility")
    # 旧V3 uniform checkpoint在本字段加入前没有cf_sampling_strategy；
    # 仅将缺失字段按旧uniform语义补齐，保留其余metadata严格比较。
    if (
        credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT
        and isinstance(actual_compatibility, dict)
        and isinstance(actual_compatibility.get("algorithm"), dict)
        and "cf_sampling_strategy" not in actual_compatibility["algorithm"]
    ):
        actual_compatibility = copy.deepcopy(actual_compatibility)
        actual_compatibility["algorithm"]["cf_sampling_strategy"] = "uniform"
    mismatch = _metadata_mismatch(expected_compatibility, actual_compatibility)
    if mismatch:
        raise RuntimeError(f"CHECKPOINT_COMPATIBILITY_ERROR: {mismatch}")
    if credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT:
        expected_sampling_strategy = str(
            expected_compatibility["algorithm"].get("cf_sampling_strategy", "uniform")
        )
        actual_sampling_strategy = str(payload.get("cf_sampling_strategy", "uniform"))
        if actual_sampling_strategy != expected_sampling_strategy:
            raise RuntimeError(
                "CHECKPOINT_COMPATIBILITY_ERROR: cf_sampling_strategy与当前配置不一致。"
            )
    elif hierarchical_gate_checkpoint:
        if str(payload.get("cf_sampling_strategy", "uniform")) != str(
            expected_compatibility["algorithm"].get("cf_sampling_strategy", "uniform")
        ):
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical WHAT sampling strategy不一致。")
        if str(payload.get("gate_sampling_strategy", "")) != "uniform_slots":
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical Gate sampling strategy不一致。")

    try:
        history = _canonical_training_history(payload.get("training_history"))
    except (TypeError, ValueError, KeyError) as exc:
        raise RuntimeError(f"CHECKPOINT_COMPATIBILITY_ERROR: training_history无效：{exc}") from exc
    if credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT:
        expected_sampling_strategy = str(
            expected_compatibility["algorithm"].get("cf_sampling_strategy", "uniform")
        )
        if str(history.get("cf_sampling_strategy", "uniform")) != expected_sampling_strategy:
            raise RuntimeError(
                "CHECKPOINT_COMPATIBILITY_ERROR: training_history cf_sampling_strategy不一致。"
            )
    policy_version = int(payload.get("policy_version", -1))
    optimizer_step_count = int(payload.get("optimizer_step_count", -1))
    completed_episodes = int(payload.get("completed_episodes", -1))
    if (
        policy_version != int(history["policy_version"])
        or optimizer_step_count != int(history["optimizer_step_count"])
        or completed_episodes != int(history["completed_episodes"])
    ):
        raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: checkpoint counters与training_history不一致。")
    if reference_drift_checkpoint or frozen_policy_batch_checkpoint:
        for field_name in (
            "cf_batch_episodes",
            "cf_samples_per_episode",
            "batch_id",
            "batch_episode_index",
            "pending_cf_sample_count",
        ):
            if payload.get(field_name) != history.get(field_name):
                raise RuntimeError(
                    "CHECKPOINT_COMPATIBILITY_ERROR: "
                    f"{field_name}与training_history不一致。"
                )
        expected_completed_batch_count = (
            history.get("causal_update_artifact_index")
            if reference_drift_checkpoint
            else history.get("batch_id")
        )
        if payload.get("completed_batch_count") != expected_completed_batch_count:
            raise RuntimeError(
                "CHECKPOINT_COMPATIBILITY_ERROR: completed_batch_count与training_history不一致。"
            )
        if reference_drift_checkpoint:
            if payload.get("causal_update_artifact_index") != history.get(
                "causal_update_artifact_index"
            ):
                raise RuntimeError(
                    "CHECKPOINT_COMPATIBILITY_ERROR: causal_update_artifact_index与training_history不一致。"
                )
            for field_name in (
                "pending_reference_state_count",
                "reference_probability_drift_cap",
            ):
                if payload.get(field_name) != history.get(field_name):
                    raise RuntimeError(
                        "CHECKPOINT_COMPATIBILITY_ERROR: "
                        f"{field_name}与training_history不一致。"
                    )
    if hierarchical_gate_checkpoint:
        for field_name in (
            "cf_batch_episodes",
            "cf_samples_per_episode",
            "gate_cf_samples_per_episode",
            "batch_id",
            "batch_episode_index",
            "pending_cf_sample_count",
            "pending_gate_cf_sample_count",
            "pending_reference_state_count",
            "reference_probability_drift_cap",
        ):
            if payload.get(field_name) != history.get(field_name):
                raise RuntimeError(
                    "CHECKPOINT_COMPATIBILITY_ERROR: hierarchical "
                    f"{field_name}与training_history不一致。"
                )
        if payload.get("causal_update_artifact_index") != history.get("causal_update_artifact_index"):
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical causal artifact index不一致。")
        if payload.get("completed_batch_count") != history.get("causal_update_artifact_index"):
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical completed_batch_count不一致。")
    q_optimizer_step_count = int(
        payload.get(
            "critic_optimizer_step_count"
            if credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT
            else "q_optimizer_step_count",
            0,
        )
    )
    if credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT:
        if int(payload.get("critic_update_count", -1)) != int(history.get("critic_update_count", -2)):
            raise RuntimeError(
                "CHECKPOINT_COMPATIBILITY_ERROR: critic_update_count与training_history不一致。"
            )
        if q_optimizer_step_count != int(history.get("critic_optimizer_step_count", -1)):
            raise RuntimeError(
                "CHECKPOINT_COMPATIBILITY_ERROR: critic_optimizer_step_count与training_history不一致。"
            )
    elif credit_assignment == ppo_update.PER_TRIGGER_Q_CREDIT_ASSIGNMENT:
        if q_optimizer_step_count != int(history.get("q_optimizer_step_count", -1)):
            raise RuntimeError(
                "CHECKPOINT_COMPATIBILITY_ERROR: q_optimizer_step_count与training_history不一致。"
            )

    try:
        policy.load_state_dict(payload["policy_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if hierarchical_gate_checkpoint:
            if gate_optimizer is None:
                raise RuntimeError("hierarchical Gate checkpoint缺少gate_optimizer。")
            policy.gate.load_state_dict(payload["gate_state_dict"])
            gate_optimizer.load_state_dict(payload["gate_optimizer_state_dict"])
        elif credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT:
            q_critic.load_state_dict(payload["critic_state_dict"])
            q_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
            critic_replay.load_state_dict(payload["critic_replay_state"])
        elif credit_assignment == ppo_update.PER_TRIGGER_Q_CREDIT_ASSIGNMENT:
            q_critic.load_state_dict(payload["q_critic_state_dict"])
            q_optimizer.load_state_dict(payload["q_optimizer_state_dict"])
        if credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT:
            # sparse V3 checkpoint 只恢复 Actor 与 actor-only optimizer。
            pass
    except Exception as exc:
        raise RuntimeError(f"CHECKPOINT_COMPATIBILITY_ERROR: policy/optimizer state不兼容：{exc}") from exc
    rng_states = payload.get("rng_states")
    if not isinstance(rng_states, dict):
        raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: rng_states缺失或类型错误。")
    _restore_rng_states(policy, rng_generator, rng_states, critic_rng_generator)
    if hierarchical_gate_checkpoint:
        try:
            cf_sampling_rng.bit_generator.state = copy.deepcopy(payload["cf_sampling_rng_state"])
            gate_sampling_rng.bit_generator.state = copy.deepcopy(payload["gate_sampling_rng_state"])
        except Exception as exc:
            raise RuntimeError(
                f"CHECKPOINT_COMPATIBILITY_ERROR: hierarchical sampling RNG无法恢复：{exc}"
            ) from exc
    elif credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT:
        try:
            cf_sampling_rng.bit_generator.state = copy.deepcopy(payload["cf_sampling_rng_state"])
        except Exception as exc:
            raise RuntimeError(
                f"CHECKPOINT_COMPATIBILITY_ERROR: cf_sampling_rng_state无法恢复：{exc}"
            ) from exc
    pending_samples: list[ppo_rollout.DirectCFCreditSample] = []
    pending_references: list[ppo_rollout.ReferenceStateRecord] = []
    pending_gate_samples: list[Any] = []
    if hierarchical_gate_checkpoint:
        raw_samples = payload.get("pending_cf_samples")
        raw_gate_samples = payload.get("pending_gate_cf_samples")
        raw_references = payload.get("pending_reference_states")
        if not isinstance(raw_samples, list) or not isinstance(raw_gate_samples, list) or not isinstance(raw_references, list):
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical pending data必须为list。")
        try:
            pending_samples = [
                ppo_rollout.direct_cf_sample_from_checkpoint_dict(item)
                for item in raw_samples
            ]
            pending_gate_samples = [
                hierarchical_reconfiguration_gate.gate_counterfactual_sample_from_dict(item)
                for item in raw_gate_samples
            ]
            pending_references = [
                ppo_rollout.reference_state_from_checkpoint_dict(item)
                for item in raw_references
            ]
        except (TypeError, KeyError, ValueError) as exc:
            raise RuntimeError(
                f"CHECKPOINT_COMPATIBILITY_ERROR: hierarchical pending data无法恢复：{exc}"
            ) from exc
        if len(pending_samples) != int(history.get("pending_cf_sample_count", -1)):
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical pending WHAT count不一致。")
        if len(pending_gate_samples) != int(history.get("pending_gate_cf_sample_count", -1)):
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical pending Gate count不一致。")
        if len(pending_references) != int(history.get("pending_reference_state_count", -1)):
            raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical pending reference count不一致。")
        pending_version = history.get("pending_collection_policy_version")
        if pending_version is not None:
            pending_version = int(pending_version)
            versions = (
                {int(item.collection_policy_version) for item in pending_samples}
                | {int(item.collection_policy_version) for item in pending_gate_samples}
                | {int(item.collection_policy_version) for item in pending_references}
            )
            if versions != {pending_version}:
                raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: hierarchical pending collection version不一致。")
        if hierarchical_state is not None:
            hierarchical_state.clear()
            hierarchical_state.update(
                {
                    "actor_policy_version": int(payload.get("actor_policy_version", policy_version)),
                    "gate_policy_version": int(payload.get("gate_policy_version", policy_version)),
                    "actor_optimizer_step_count": int(payload.get("actor_optimizer_step_count", optimizer_step_count)),
                    "gate_optimizer_step_count": int(payload.get("gate_optimizer_step_count", optimizer_step_count)),
                    "pending_gate_cf_samples": pending_gate_samples,
                }
            )
    elif reference_drift_checkpoint:
        raw_samples = payload.get("pending_cf_samples")
        raw_references = payload.get("pending_reference_states")
        if not isinstance(raw_samples, list) or not isinstance(raw_references, list):
            raise RuntimeError(
                "CHECKPOINT_COMPATIBILITY_ERROR: refdrift pending samples/reference states必须为list。"
            )
        try:
            pending_samples = [
                ppo_rollout.direct_cf_sample_from_checkpoint_dict(item)
                for item in raw_samples
            ]
            pending_references = [
                ppo_rollout.reference_state_from_checkpoint_dict(item)
                for item in raw_references
            ]
        except (TypeError, KeyError, ValueError) as exc:
            raise RuntimeError(
                f"CHECKPOINT_COMPATIBILITY_ERROR: refdrift pending data无法恢复：{exc}"
            ) from exc
        if len(pending_samples) != int(history.get("pending_cf_sample_count", -1)):
            raise RuntimeError(
                "CHECKPOINT_COMPATIBILITY_ERROR: 恢复后的pending CF count不一致。"
            )
        if len(pending_references) != int(history.get("pending_reference_state_count", -1)):
            raise RuntimeError(
                "CHECKPOINT_COMPATIBILITY_ERROR: 恢复后的pending reference count不一致。"
            )
        selected_ids = [str(item.decision_id) for item in pending_samples]
        try:
            selected_records, _, partition = reference_drift.validate_reference_partition(
                pending_references,
                selected_ids,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"CHECKPOINT_COMPATIBILITY_ERROR: pending reference partition无效：{exc}"
            ) from exc
        if partition["selected_count"] != len(pending_samples) or partition["intersection_count"] != 0:
            raise RuntimeError(
                "CHECKPOINT_COMPATIBILITY_ERROR: pending selected/reference partition不一致。"
            )
        selected_flags = {
            str(item.decision_id)
            for item in pending_references
            if bool(item.selected_for_direct_cf)
        }
        if selected_flags != set(selected_ids):
            raise RuntimeError(
                "CHECKPOINT_COMPATIBILITY_ERROR: pending reference selected flags不一致。"
            )
        pending_version = history.get("pending_collection_policy_version")
        if pending_version is not None:
            pending_version = int(pending_version)
            sample_versions = {int(item.collection_policy_version) for item in pending_samples}
            reference_versions = {int(item.collection_policy_version) for item in pending_references}
            if sample_versions != {pending_version} or reference_versions != {pending_version}:
                raise RuntimeError(
                    "CHECKPOINT_COMPATIBILITY_ERROR: pending collection_policy_version不一致。"
                )
    return (
        history,
        policy_version,
        optimizer_step_count,
        q_optimizer_step_count,
        pending_samples,
        pending_references,
    )


def _require_torch() -> None:
    if torch is None:
        raise ImportError("ppo_training.py 需要 PyTorch。")


def _guard_counts(rollouts: int, env_steps: int, optimizer_steps: int) -> None:
    if rollouts > SANITY_MAX_ROLLOUTS:
        raise RuntimeError("超出 SANITY_MAX_ROLLOUTS。")
    if env_steps > SANITY_MAX_TOTAL_ENV_STEPS:
        raise RuntimeError("超出 SANITY_MAX_TOTAL_ENV_STEPS。")
    if optimizer_steps > SANITY_MAX_OPTIMIZER_STEPS:
        raise RuntimeError("超出 SANITY_MAX_OPTIMIZER_STEPS。")


def _build_context(config_path: str) -> TrainingSanityContext:
    cfg = project_config.load_config(config_path)
    project_config.validate_for_formal_experiment(cfg)
    root = Path(cfg["_meta"]["project_root"])
    cache_cfg = cache_configuration.cache_config_from_project(cfg)
    scenarios = cache_configuration.load_prediction_scenarios(
        cache_configuration.resolve_path(root, project_config.get_config_value(cfg, "paths.prediction_scenarios"))
    )
    origin_split = _ppo_origin_split_from_project_config(
        cfg,
        int(scenarios["origin_slots"].shape[0]),
    )
    service_catalog = cache_configuration.load_service_catalog(
        cache_configuration.resolve_path(root, project_config.get_config_value(cfg, "paths.service_catalog"))
    )
    first_origin = cache_configuration.scenario_at_origin(
        scenarios,
        origin_split.train_origin_start,
    )
    cache_nodes = cache_configuration.build_cache_nodes(first_origin, cache_cfg)
    local_cache = task_scenario.load_tv_local_cache(
        cache_configuration.resolve_path(root, project_config.get_config_value(cfg, "paths.tv_local_cache"))
    )
    action_spec = ppo_trigger_policy.TriggerActionSpec.from_cache_nodes_and_project(cache_nodes, cfg)
    observation_spec = ppo_trigger_policy.ConfigurationObservationSpec.from_cache_nodes(
        cache_nodes,
        service_catalog["service_ids"].tolist(),
        action_spec.uav_ids,
        [str(value) for value in first_origin["vehicle_ids"].tolist()],
    )
    gamma, gae_lambda = ppo_rollout.ppo_gae_config_from_project(cfg)
    initial_cache = cache_configuration.popularity_aware_initial_cache(
        service_catalog,
        cache_nodes,
        cache_cfg,
    )
    initial_uav = uav_configuration.initial_uav_positions_from_project(cfg)
    return TrainingSanityContext(
        cfg=cfg,
        cache_cfg=cache_cfg,
        scenarios=scenarios,
        service_catalog=service_catalog,
        cache_nodes=cache_nodes,
        local_cache=local_cache,
        flight_boundary=uav_configuration.boundary_from_project(cfg),
        action_spec=action_spec,
        observation_spec=observation_spec,
        gamma=gamma,
        gae_lambda=gae_lambda,
        terminal_slot=int(scenarios["origin_slots"][origin_split.train_origin_end]),
        initial_cache=initial_cache,
        initial_uav=initial_uav,
    )


def build_formal_training_protocol(
    scenarios: dict[str, Any],
    project_cfg: dict[str, Any],
) -> FormalTrainingProtocol:
    """由 prediction_scenarios metadata 构造正式 train/test PPO协议。

    纯协议函数：不创建policy、不调用execute_slot、不创建optimizer、不训练。
    """

    total_origin_count = int(scenarios["origin_slots"].shape[0])
    origin_split = _ppo_origin_split_from_project_config(
        project_cfg,
        total_origin_count,
    )
    origin_indices = np.arange(
        origin_split.train_origin_start,
        origin_split.train_origin_end + 1,
        dtype=np.int64,
    )
    slots = np.asarray(scenarios["origin_slots"], dtype=np.int64)
    times = np.asarray(scenarios["origin_times_s"], dtype=np.float64)
    test_origin_indices = np.arange(
        origin_split.test_origin_start,
        origin_split.test_origin_end + 1,
        dtype=np.int64,
    )
    if slots.shape[0] != total_origin_count or times.shape[0] != total_origin_count:
        raise ValueError("prediction_scenarios origin_slots/origin_times_s 长度不一致。")
    if not np.all(np.isfinite(times)):
        raise ValueError("prediction_scenarios origin_times_s 不能包含 NaN/Inf。")
    steps_per_episode = int(len(origin_indices))
    test_steps_per_evaluation = int(len(test_origin_indices))
    update_epochs = int(project_config.get_config_value(project_cfg, "algorithm.ppo.update_epochs"))
    num_minibatches = int(project_config.get_config_value(project_cfg, "algorithm.ppo.num_minibatches"))
    if steps_per_episode <= 0:
        raise ValueError("正式episode没有可执行configuration origin。")
    if test_steps_per_evaluation <= 0:
        raise ValueError("held-out test没有可执行origin。")
    if steps_per_episode % num_minibatches != 0:
        raise ValueError(
            f"steps_per_episode={steps_per_episode} 不能被 num_minibatches={num_minibatches} 整除。"
        )
    minibatch_size = steps_per_episode // num_minibatches
    if minibatch_size <= 0:
        raise ValueError("推导得到的 minibatch_size 必须大于0。")
    return FormalTrainingProtocol(
        origin_split=origin_split,
        episode_origin_indices=origin_indices,
        episode_slots=slots[origin_indices],
        episode_times_s=times[origin_indices],
        test_origin_indices=test_origin_indices,
        test_slots=slots[test_origin_indices],
        test_times_s=times[test_origin_indices],
        steps_per_episode=steps_per_episode,
        test_steps_per_evaluation=test_steps_per_evaluation,
        update_epochs=update_epochs,
        num_minibatches=num_minibatches,
        minibatch_size=minibatch_size,
        expected_optimizer_steps_per_rollout=update_epochs * num_minibatches,
        true_terminal_origin=int(origin_indices[-1]),
        true_terminal_slot=int(slots[origin_indices[-1]]),
        true_terminal_time_s=float(times[origin_indices[-1]]),
        test_true_terminal_origin=int(test_origin_indices[-1]),
        test_true_terminal_slot=int(slots[test_origin_indices[-1]]),
        test_true_terminal_time_s=float(times[test_origin_indices[-1]]),
        single_trace_training=True,
        initial_cache_source=(
            "cache_configuration.popularity_aware_initial_cache(service_catalog, cache_nodes, "
            "cache_cfg); fixed once per formal run"
        ),
        initial_uav_source="uav_configuration.initial_uav_positions_from_project(system_config)",
        tv_local_cache_source=str(project_config.get_config_value(project_cfg, "paths.tv_local_cache")),
        evaluation_semantics=(
            "held-out test deterministic evaluation: deterministic=True, torch.no_grad, "
            "no optimizer, reset C/Q to formal initial state, execute test origin range only, "
            "report raw G(t)"
        ),
    )


def validate_formal_training_protocol(protocol: FormalTrainingProtocol) -> dict[str, bool]:
    """静态校验正式训练协议；不执行环境。"""

    origins = protocol.episode_origin_indices
    slots = protocol.episode_slots
    times = protocol.episode_times_s
    test_origins = protocol.test_origin_indices
    test_slots = protocol.test_slots
    test_times = protocol.test_times_s
    split = protocol.origin_split
    checks = {
        "origins_contiguous": bool(
            np.array_equal(
                origins,
                np.arange(
                    split.train_origin_start,
                    split.train_origin_end + 1,
                    dtype=np.int64,
                ),
            )
        ),
        "origins_in_train_range": bool(
            np.all(
                (origins >= split.train_origin_start)
                & (origins <= split.train_origin_end)
            )
        ),
        "test_origins_contiguous": bool(
            np.array_equal(
                test_origins,
                np.arange(
                    split.test_origin_start,
                    split.test_origin_end + 1,
                    dtype=np.int64,
                ),
            )
        ),
        "test_origins_in_test_range": bool(
            np.all(
                (test_origins >= split.test_origin_start)
                & (test_origins <= split.test_origin_end)
            )
        ),
        "train_step_count_matches_range": bool(
            protocol.steps_per_episode == split.train_count == len(origins)
        ),
        "test_step_count_matches_range": bool(
            protocol.test_steps_per_evaluation == split.test_count == len(test_origins)
        ),
        "train_test_disjoint": bool(set(origins.tolist()).isdisjoint(test_origins.tolist())),
        "train_test_adjacent": bool(
            int(origins[-1]) + 1 == int(test_origins[0])
        ),
        "slots_strictly_increasing": bool(np.all(np.diff(slots) > 0)),
        "times_strictly_increasing": bool(np.all(np.diff(times) > 0.0)),
        "test_slots_strictly_increasing": bool(np.all(np.diff(test_slots) > 0)),
        "test_times_strictly_increasing": bool(np.all(np.diff(test_times) > 0.0)),
        "rollout_size_positive": bool(protocol.steps_per_episode > 0),
        "test_size_positive": bool(protocol.test_steps_per_evaluation > 0),
        "minibatch_divisible": bool(protocol.steps_per_episode % protocol.num_minibatches == 0),
        "minibatch_size_positive": bool(protocol.minibatch_size > 0),
        "true_terminal_is_last_origin": bool(protocol.true_terminal_origin == int(origins[-1])),
        "test_true_terminal_is_last_origin": bool(
            protocol.test_true_terminal_origin == int(test_origins[-1])
        ),
        "middle_origins_not_terminal": bool(np.all(origins[:-1] < protocol.true_terminal_origin)),
        "reset_state_resolved": bool(
            protocol.initial_cache_source
            and protocol.initial_uav_source
            and protocol.tv_local_cache_source
        ),
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise ValueError(f"FormalTrainingProtocol validation failed: {failed}")
    return checks


def run_training_dry_run(config_path: str = "system_config.toml") -> FormalTrainingDryRunResult:
    """正式训练协议dry-run：只读metadata和配置，不执行环境、不optimizer.step。"""

    ctx = _build_context(config_path)
    protocol = build_formal_training_protocol(ctx.scenarios, ctx.cfg)
    checks = validate_formal_training_protocol(protocol)
    credit_assignment = ppo_update.credit_assignment_from_project(ctx.cfg)
    credit_phase = (
        ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT
        if credit_assignment == ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT
        else (
            ppo_update.credit_phase_from_project(ctx.cfg)
            if credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT
            else "legacy"
        )
    )
    algorithm_reporting = _dry_run_algorithm_reporting(ctx.cfg, credit_assignment)
    hierarchical_gate_mode = _is_hierarchical_gate_mode(ctx.cfg)
    cf_samples = (
        _cf_samples_per_episode(ctx.cfg)
        if _is_sparse_direct_cf_mode(ctx.cfg) or hierarchical_gate_mode
        else 0
    )
    cf_batch_episodes = (
        _cf_batch_episodes(ctx.cfg)
        if _is_frozen_batch_family(ctx.cfg) or hierarchical_gate_mode
        else 1
    )
    cf_sampling_strategy = (
        _cf_sampling_strategy(ctx.cfg)
        if _is_sparse_direct_cf_mode(ctx.cfg) or hierarchical_gate_mode
        else "uniform"
    )
    sample_num_episodes = (
        cf_batch_episodes
        if _is_frozen_batch_family(ctx.cfg) or hierarchical_gate_mode
        else 3
    )
    planned_q_optimizer_steps = (
        sample_num_episodes * protocol.expected_optimizer_steps_per_rollout
        if credit_assignment in {
            ppo_update.PER_TRIGGER_Q_CREDIT_ASSIGNMENT,
            ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT,
        }
        else 0
    )
    planned_actor_optimizer_steps = sample_num_episodes * protocol.expected_optimizer_steps_per_rollout
    if _is_sparse_direct_cf_mode(ctx.cfg) or hierarchical_gate_mode:
        planned_actor_optimizer_steps = (
            (sample_num_episodes // cf_batch_episodes) * int(protocol.update_epochs)
            if _is_frozen_batch_family(ctx.cfg) or hierarchical_gate_mode
            else sample_num_episodes * int(protocol.update_epochs)
        )
    if (
        credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT
        and credit_phase == "critic_warmup"
    ):
        planned_actor_optimizer_steps = 0
    return FormalTrainingDryRunResult(
        protocol=protocol,
        origins_contiguous=checks["origins_contiguous"],
        slots_strictly_increasing=checks["slots_strictly_increasing"],
        times_strictly_increasing=checks["times_strictly_increasing"],
        minibatch_divisible=checks["minibatch_divisible"],
        reset_state_resolved=checks["reset_state_resolved"],
        formal_rollout_executor_available=callable(collect_formal_episode_rollout),
        ppo_update_executor_available=callable(update_policy_on_rollout),
        formal_training_api_available=callable(run_formal_ppo_training),
        held_out_test_evaluation_api_available=callable(evaluate_held_out_test),
        sample_num_episodes=sample_num_episodes,
        planned_environment_steps=sample_num_episodes * protocol.steps_per_episode,
        planned_optimizer_steps=planned_actor_optimizer_steps,
        planned_backward_calls=planned_actor_optimizer_steps,
        optimizer_instances_planned=(
            2 if planned_q_optimizer_steps > 0 or hierarchical_gate_mode else 1
        ),
        executor_dry_run_only=True,
        formal_validation=True,
        environment_steps_executed=0,
        backward_calls_executed=0,
        optimizer_steps_executed=0,
        blocker="",
        credit_assignment=credit_assignment,
        planned_q_optimizer_steps=planned_q_optimizer_steps,
        q_optimizer_instances_planned=(1 if planned_q_optimizer_steps > 0 else 0),
        credit_phase=credit_phase,
        planned_actor_optimizer_steps=planned_actor_optimizer_steps,
        actor_optimizer_instances_planned=1,
        critic_optimizer_instances_planned=(1 if planned_q_optimizer_steps > 0 else 0),
        cf_samples_per_episode=cf_samples,
        planned_counterfactual_branch_steps=(
            sample_num_episodes * cf_samples * protocol.steps_per_episode
            if _is_sparse_direct_cf_mode(ctx.cfg) or hierarchical_gate_mode
            else 0
        ),
        cf_sampling_strategy=cf_sampling_strategy,
        cf_batch_episodes=cf_batch_episodes,
        gate_cf_samples_per_episode=(
            HIERARCHICAL_GATE_CF_SAMPLES_PER_EPISODE if hierarchical_gate_mode else 0
        ),
        planned_gate_counterfactual_branch_steps=(
            sample_num_episodes
            * HIERARCHICAL_GATE_CF_SAMPLES_PER_EPISODE
            * protocol.steps_per_episode
            if hierarchical_gate_mode
            else 0
        ),
        planned_gate_optimizer_steps=(
            planned_actor_optimizer_steps if hierarchical_gate_mode else 0
        ),
        gate_optimizer_instances_planned=(1 if hierarchical_gate_mode else 0),
        algorithm_reporting=algorithm_reporting,
        planned_total_optimizer_steps=(
            planned_actor_optimizer_steps
            + planned_q_optimizer_steps
            + (
                planned_actor_optimizer_steps
                if hierarchical_gate_mode
                else 0
            )
        ),
        planned_total_backward_calls=(
            planned_actor_optimizer_steps
            + planned_q_optimizer_steps
            + (
                planned_actor_optimizer_steps
                if hierarchical_gate_mode
                else 0
            )
        ),
    )


def _formal_aligned_steps(
    ctx: TrainingSanityContext,
    protocol: FormalTrainingProtocol,
) -> list[ppo_rollout.AlignedRolloutStep]:
    """构造 train origin 区间的完整正式episode输入。"""

    root = Path(ctx.cfg["_meta"]["project_root"])
    steps = ppo_rollout.prepare_aligned_rollout_steps(
        root,
        ctx.scenarios,
        int(protocol.episode_origin_indices[0]),
        int(protocol.steps_per_episode),
        root / "data/generated/vehicle_predictions.csv",
        root / "data/generated/popularity_predictions.csv",
        root / "data/generated/tasks.csv",
        root / "data/generated/trajectory.csv",
        include_bootstrap_step=False,
    )
    if len(steps) != protocol.steps_per_episode:
        raise RuntimeError(
            f"完整episode应包含 {protocol.steps_per_episode} 个可执行origin，实际得到 {len(steps)}。"
        )
    if int(steps[-1].origin_index) != int(protocol.true_terminal_origin):
        raise RuntimeError("完整episode最后一个origin不是FormalTrainingProtocol定义的true terminal。")
    expected_origins = protocol.episode_origin_indices.astype(np.int64).tolist()
    actual_origins = [int(step.origin_index) for step in steps]
    if actual_origins != expected_origins:
        raise RuntimeError(
            "正式训练 aligned steps 越过 train/test 边界："
            f"expected={expected_origins[0]}..{expected_origins[-1]}, "
            f"actual={actual_origins[0]}..{actual_origins[-1]}"
        )
    return steps


def _held_out_test_aligned_steps(
    ctx: TrainingSanityContext,
    protocol: FormalTrainingProtocol,
) -> list[ppo_rollout.AlignedRolloutStep]:
    """构造 held-out test origin 区间，不加载区间外的 actual task。"""

    root = Path(ctx.cfg["_meta"]["project_root"])
    steps = ppo_rollout.prepare_aligned_rollout_steps(
        root,
        ctx.scenarios,
        int(protocol.origin_split.test_origin_start),
        int(protocol.test_steps_per_evaluation),
        root / "data/generated/vehicle_predictions.csv",
        root / "data/generated/popularity_predictions.csv",
        root / "data/generated/tasks.csv",
        root / "data/generated/trajectory.csv",
        include_bootstrap_step=False,
    )
    expected_origins = protocol.test_origin_indices.astype(np.int64).tolist()
    actual_origins = [int(step.origin_index) for step in steps]
    if actual_origins != expected_origins:
        raise RuntimeError(
            "held-out test aligned steps 不在 test origin 区间内："
            f"expected={expected_origins[0]}..{expected_origins[-1]}, "
            f"actual={actual_origins[0]}..{actual_origins[-1]}"
        )
    if int(steps[-1].origin_index) != int(protocol.test_true_terminal_origin):
        raise RuntimeError("held-out test 最后一个origin不是test true terminal。")
    return steps


def collect_formal_episode_rollout(
    policy: Any,
    ctx: TrainingSanityContext,
    protocol: FormalTrainingProtocol,
    direct_cf_reservoir: ppo_rollout.DirectCFReservoir | None = None,
    *,
    episode_index: int = 0,
    collection_policy_version: int = 0,
    collect_reference_states: bool = False,
    gate_cf_reservoir: ppo_rollout.GateCFReservoir | None = None,
) -> ppo_rollout.RolloutBatch:
    """收集一个完整正式episode rollout。

    语义：
    1. episode开始 reset 到正式 initial Cache 和 initial UAV positions；
    2. 从 FormalTrainingProtocol 第一个 origin 连续执行到最后一个 origin；
    3. episode内部 C(t)、Q^U(t) 逐slot传播，不逐slot reset；
    4. 最后一个origin为true terminal，因此GAE bootstrap=0；
    5. policy 使用可行动作 mask；不做 sampling 后 repair/resample；mask 无法提前
       判断的 residual infeasibility 仍向上抛出 InfeasibleActionBlocker。

    注意：这是正式执行器。本阶段CLI dry-run不会调用该函数。
    """

    steps = _formal_aligned_steps(ctx, protocol)
    batch = ppo_rollout.collect_on_policy_rollout(
        policy,
        ctx.initial_cache,
        ctx.initial_uav,
        steps,
        ctx.local_cache,
        ctx.service_catalog,
        ctx.cfg,
        ctx.cache_nodes,
        ctx.cache_cfg,
        ctx.flight_boundary,
        ctx.observation_spec,
        ctx.action_spec,
        ctx.gamma,
        ctx.gae_lambda,
        ctx.terminal_slot,
        direct_cf_reservoir,
        episode_index,
        collection_policy_version,
        collect_reference_states,
        gate_cf_reservoir,
    )
    if batch.truncated or not bool(batch.metadata.get("true_terminal", False)):
        raise RuntimeError("正式完整episode rollout必须以true terminal结束，且bootstrap必须为0。")
    if not math.isclose(float(batch.final_bootstrap_value), 0.0, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("正式完整episode rollout的GAE bootstrap必须为0。")
    return batch


def _formal_raw_rollout_statistics(batch: ppo_rollout.RolloutBatch) -> dict[str, float]:
    """从原始 Python float transition 统计正式 episode summary 并校验恒等式。

    ``RolloutBatch.rewards`` 必须继续是 float32，供后续 GAE/PPO 使用；正式
    episode 的累计 G/Q/cost 则从 transition 的原始 Python float 读取，避免
    长序列 float32 累计误差决定 episode invariant。该函数不重新定义 reward。
    """

    transitions = batch.transitions
    if not transitions:
        raise ValueError("正式 rollout 不能为空。")
    raw_reward_sum = math.fsum(float(item.reward) for item in transitions)
    raw_actual_q_sum = math.fsum(float(item.actual_task_utility) for item in transitions)
    raw_cache_cost_sum = math.fsum(float(item.cache_cost) for item in transitions)
    raw_movement_cost_sum = math.fsum(float(item.movement_cost) for item in transitions)
    expected_reward_sum = raw_actual_q_sum - raw_cache_cost_sum - raw_movement_cost_sum
    if not math.isclose(
        raw_reward_sum,
        expected_reward_sum,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise AssertionError(
            "formal episode raw G identity failed: "
            f"sum_G={raw_reward_sum}, expected={expected_reward_sum}, "
            f"error={raw_reward_sum - expected_reward_sum}"
        )

    # 仅作mixed-precision诊断；不能参与正式episode PASS/FAIL判定。
    batch_reward_sum_float32 = float(np.sum(batch.rewards, dtype=np.float32))
    return {
        "raw_reward_sum": float(raw_reward_sum),
        "raw_reward_mean": float(raw_reward_sum / len(transitions)),
        "raw_actual_q_sum": float(raw_actual_q_sum),
        "raw_cache_cost_sum": float(raw_cache_cost_sum),
        "raw_movement_cost_sum": float(raw_movement_cost_sum),
        "batch_reward_sum_float32": batch_reward_sum_float32,
        "batch_vs_raw_reward_sum_error": float(batch_reward_sum_float32 - raw_reward_sum),
    }


def _rollout_trigger_action1_counts(
    batch: ppo_rollout.RolloutBatch,
    trigger_labels: tuple[str, ...] | list[str],
) -> dict[str, int]:
    """Count ordinary rollout action=1 decisions without changing behavior.

    The count is derived from transitions that the collector already produced;
    it performs no sampling, environment call, gradient calculation, or RNG
    access.  It is intentionally distinct from ``cf_trigger_sample_counts``,
    which reports selected Direct-CF samples.
    """

    labels = tuple(str(label) for label in trigger_labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("rollout trigger_labels 必须非空且不重复。")
    counts = {label: 0 for label in labels}
    for transition in batch.transitions:
        action = np.asarray(transition.action_vector, dtype=np.int8).reshape(-1)
        if action.shape != (len(labels),) or not np.all(np.isin(action, [0, 1])):
            raise ValueError(
                "ordinary rollout trigger action vector长度必须与 dynamic action spec 一致。"
            )
        for index, label in enumerate(labels):
            counts[label] += int(action[index])
    return counts


def _format_optional_float_for_reporting(value: float | None, digits: int = 4) -> str:
    """Format nullable update-only metrics without changing their JSON semantics."""

    if value is None:
        return "null"
    return f"{float(value):.{int(digits)}f}"


def _validate_k8_episode1_baseline(
    batch: ppo_rollout.RolloutBatch,
    raw_stats: dict[str, float],
) -> None:
    """在K=8 Episode1的任何CF branch前锁定正式baseline复现。"""

    if len(batch.transitions) != int(K8_EPISODE1_BASELINE["steps"]):
        raise RuntimeError(
            "K=8 Episode1 baseline steps不匹配；停止且不执行任何CF branch。"
        )
    checks = (
        ("G", raw_stats["raw_reward_sum"]),
        ("Q", raw_stats["raw_actual_q_sum"]),
        ("cache", raw_stats["raw_cache_cost_sum"]),
        ("UAV", raw_stats["raw_movement_cost_sum"]),
    )
    for name, actual in checks:
        expected = float(K8_EPISODE1_BASELINE[name])
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"K=8 Episode1 baseline {name}不匹配：actual={actual}, expected={expected}；"
                "停止且不执行任何CF branch。"
            )
    if bool(batch.truncated) or not bool(batch.metadata.get("true_terminal", False)):
        raise RuntimeError("K=8 Episode1 baseline不是true terminal；停止且不执行CF branch。")
    if float(batch.final_bootstrap_value) != 0.0:
        raise RuntimeError("K=8 Episode1 baseline bootstrap不是0；停止且不执行CF branch。")


def _slice_rollout_batch(
    batch: ppo_rollout.RolloutBatch,
    indices: np.ndarray,
) -> ppo_rollout.RolloutBatch:
    """按minibatch索引切片RolloutBatch；old policy数据保持来自原rollout。"""

    idx = np.asarray(indices, dtype=np.int64)
    return ppo_rollout.RolloutBatch(
        observations=batch.observations[idx].copy(),
        actions=batch.actions[idx].copy(),
        action_masks=batch.action_masks[idx].copy(),
        old_log_probs=batch.old_log_probs[idx].copy(),
        old_per_action_log_probs=batch.old_per_action_log_probs[idx].copy(),
        old_per_action_probabilities=batch.old_per_action_probabilities[idx].copy(),
        actor_valid_mask=batch.actor_valid_mask[idx].copy(),
        old_values=batch.old_values[idx].copy(),
        rewards=batch.rewards[idx].copy(),
        dones=batch.dones[idx].copy(),
        advantages=batch.advantages[idx].copy(),
        returns=batch.returns[idx].copy(),
        monte_carlo_returns=batch.monte_carlo_returns[idx].copy(),
        slots=batch.slots[idx].copy(),
        origin_indices=batch.origin_indices[idx].copy(),
        final_bootstrap_value=float(batch.final_bootstrap_value),
        truncated=bool(batch.truncated),
        transitions=[batch.transitions[int(i)] for i in idx.tolist()],
        reference_decisions=tuple(batch.reference_decisions),
        metadata=dict(batch.metadata),
        gate_actions=(
            None if batch.gate_actions is None else batch.gate_actions[idx].copy()
        ),
        old_gate_log_probs=(
            None
            if batch.old_gate_log_probs is None
            else batch.old_gate_log_probs[idx].copy()
        ),
        old_gate_probabilities=(
            None
            if batch.old_gate_probabilities is None
            else batch.old_gate_probabilities[idx].copy()
        ),
        subpolicy_actions=(
            None if batch.subpolicy_actions is None else batch.subpolicy_actions[idx].copy()
        ),
    )


def _torch_randperm_numpy(size: int, rng_generator: Any) -> np.ndarray:
    _require_torch()
    if rng_generator is None:
        perm = torch.randperm(size)
    else:
        perm = torch.randperm(size, generator=rng_generator)
    return perm.detach().cpu().numpy().astype(np.int64)


def _aggregate_epoch_diagnostics(
    epoch_index: int,
    minibatch_diagnostics: list[dict[str, float | int]],
    pre_clipping_grad_norms: list[float],
    optimizer_steps_cumulative: int,
) -> PPOEpochSummary:
    """仅用当前epoch已产生的scalar诊断值构造摘要。"""

    if not minibatch_diagnostics:
        raise ValueError("epoch诊断不能为空。")
    if len(minibatch_diagnostics) != len(pre_clipping_grad_norms):
        raise ValueError("epoch minibatch诊断与gradient norm数量不一致。")

    transition_count = sum(int(item["transition_count"]) for item in minibatch_diagnostics)
    if transition_count <= 0:
        raise ValueError("epoch transition_count必须大于0。")

    def weighted_mean(field: str) -> float:
        return float(
            math.fsum(
                float(item[field]) * int(item["transition_count"])
                for item in minibatch_diagnostics
            )
            / transition_count
        )

    for item in minibatch_diagnostics:
        for field in (
            "policy_loss",
            "value_loss",
            "entropy",
            "total_loss",
            "approx_kl",
            "clip_fraction",
            "ratio_mean",
            "ratio_min",
            "ratio_max",
        ):
            if not math.isfinite(float(item[field])):
                raise ValueError(f"epoch诊断字段 {field} 出现非finite值。")
    if not all(math.isfinite(float(value)) for value in pre_clipping_grad_norms):
        raise ValueError("epoch pre-clipping gradient norm出现非finite值。")

    return PPOEpochSummary(
        epoch_index=int(epoch_index),
        policy_loss_mean=weighted_mean("policy_loss"),
        value_loss_mean=weighted_mean("value_loss"),
        entropy_mean=weighted_mean("entropy"),
        total_loss_mean=weighted_mean("total_loss"),
        approx_kl_mean=weighted_mean("approx_kl"),
        clip_fraction_mean=weighted_mean("clip_fraction"),
        ratio_mean=weighted_mean("ratio_mean"),
        ratio_min=float(min(float(item["ratio_min"]) for item in minibatch_diagnostics)),
        ratio_max=float(max(float(item["ratio_max"]) for item in minibatch_diagnostics)),
        pre_clipping_grad_norm_mean=float(math.fsum(pre_clipping_grad_norms) / len(pre_clipping_grad_norms)),
        pre_clipping_grad_norm_max=float(max(pre_clipping_grad_norms)),
        minibatch_count=int(len(minibatch_diagnostics)),
        transition_count=int(transition_count),
        optimizer_steps_cumulative=int(optimizer_steps_cumulative),
    )


def _update_policy_on_rollout_per_trigger_q(
    policy: Any,
    optimizer: Any,
    q_critic: Any,
    q_optimizer: Any,
    rollout_batch: ppo_rollout.RolloutBatch,
    project_cfg: dict[str, Any],
    rng_generator: Any,
) -> FormalPPOUpdateSummary:
    """执行第一版 per-trigger Q + counterfactual PPO update。

    更新顺序固定为：
    1. 用更新前 Actor/Q 计算并冻结 Q_old、baseline_cf、A_cf；
    2. 用 full terminal Monte-Carlo return 独立更新 Q critic；
    3. 用冻结 A_cf、old per-action log-prob 和真实 action prefix 更新
       Actor/scalar-V。Q 参数不进入 Actor optimizer 或 Actor gradient clip。
    """

    _require_torch()
    training_cfg = ppo_update.ppo_training_config_status(project_cfg)
    if not training_cfg.can_run_optimizer_step:
        raise RuntimeError(f"PPO update超参数缺失：{training_cfg.missing_required_for_step}")
    if training_cfg.max_grad_norm is None or training_cfg.learning_rate is None:
        raise RuntimeError("per_trigger_q update 缺少 max_grad_norm/learning_rate。")
    if ppo_update.q_target_from_project(project_cfg) != ppo_update.MONTE_CARLO_Q_TARGET:
        raise RuntimeError("per_trigger_q 第一版必须使用 q_target=monte_carlo。")
    if rollout_batch.monte_carlo_returns.shape != rollout_batch.rewards.shape:
        raise ValueError("monte_carlo_returns shape 必须与 rewards 一致。")
    if not bool(rollout_batch.metadata.get("true_terminal", False)):
        raise RuntimeError("per_trigger_q 的 full MC target 要求 true-terminal rollout。")

    update_epochs = int(project_config.get_config_value(project_cfg, "algorithm.ppo.update_epochs"))
    num_minibatches = int(project_config.get_config_value(project_cfg, "algorithm.ppo.num_minibatches"))
    batch_size = int(rollout_batch.observations.shape[0])
    if batch_size <= 0 or batch_size % num_minibatches != 0:
        raise ValueError("rollout batch size 必须为正且可被 num_minibatches 整除。")
    minibatch_size = batch_size // num_minibatches

    targets = ppo_update.prepare_counterfactual_targets(policy, q_critic, rollout_batch)
    frozen_counterfactual_advantages = targets.advantages.detach().clone()
    frozen_actor_valid_mask = targets.actor_valid_mask.detach().clone()
    free_advantages = frozen_counterfactual_advantages[frozen_actor_valid_mask]
    if int(free_advantages.numel()) > 0:
        cf_adv_mean = float(torch.mean(free_advantages).detach().cpu())
        cf_adv_std = float(torch.std(free_advantages, unbiased=False).detach().cpu())
        cf_adv_abs_mean = float(torch.mean(torch.abs(free_advantages)).detach().cpu())
    else:
        cf_adv_mean = 0.0
        cf_adv_std = 0.0
        cf_adv_abs_mean = 0.0

    # 每个 epoch 只抽一次 permutation，并复用于 Q/Actor 两阶段，避免为新 Q
    # optimizer 无意义地增加一套 RNG 语义；旧 legacy mode 的 shuffle 路径不变。
    epoch_shuffles = [
        _torch_randperm_numpy(batch_size, rng_generator)
        for _ in range(update_epochs)
    ]

    q_losses: list[float] = []
    q_gradient_norms: list[float] = []
    q_gradient_finite_all = True
    q_optimizer_steps = 0

    for shuffled in epoch_shuffles:
        for start in range(0, batch_size, minibatch_size):
            minibatch_indices = shuffled[start : start + minibatch_size]
            if len(minibatch_indices) != minibatch_size:
                raise RuntimeError("Q critic minibatch划分不完整。")
            indices_t = torch.as_tensor(
                minibatch_indices,
                dtype=torch.long,
                device=targets.conditional_hidden_states.device,
            )
            hidden_mb = targets.conditional_hidden_states.index_select(0, indices_t)
            actions_mb = rollout_batch.actions[minibatch_indices]
            mc_returns_mb = rollout_batch.monte_carlo_returns[minibatch_indices]
            actor_valid_mb = rollout_batch.actor_valid_mask[minibatch_indices]
            q_optimizer.zero_grad(set_to_none=True)
            q_loss_result = ppo_update.compute_per_trigger_q_loss(
                q_critic,
                hidden_mb,
                actions_mb,
                mc_returns_mb,
                actor_valid_mb,
            )
            if not torch.isfinite(q_loss_result.q_loss):
                raise FloatingPointError("Q critic loss 非finite。")
            q_loss_result.q_loss.backward()
            q_grads = [
                parameter.grad.detach()
                for parameter in q_critic.parameters()
                if parameter.grad is not None
            ]
            q_gradient_finite = bool(q_grads) and bool(
                all(torch.isfinite(grad).all().item() for grad in q_grads)
            )
            q_gradient_finite_all = q_gradient_finite_all and q_gradient_finite
            if not q_gradient_finite:
                raise FloatingPointError("Q critic gradient 出现 NaN/Inf。")
            q_grad_norm = torch.nn.utils.clip_grad_norm_(
                q_critic.parameters(),
                float(training_cfg.max_grad_norm),
            )
            q_gradient_norms.append(float(q_grad_norm.detach().cpu()))
            q_losses.append(float(q_loss_result.q_loss.detach().cpu()))
            q_optimizer.step()
            q_optimizer_steps += 1

    # 清除 Q optimizer 上一阶段的梯度，确保后续 Actor backward 不会误留下
    # Q gradient；Q 参数从未加入 Actor optimizer。
    q_optimizer.zero_grad(set_to_none=True)

    frozen_old_log_probs = rollout_batch.old_log_probs.copy()
    frozen_old_per_action_log_probs = rollout_batch.old_per_action_log_probs.copy()
    frozen_old_per_action_probabilities = rollout_batch.old_per_action_probabilities.copy()
    frozen_old_values = rollout_batch.old_values.copy()
    frozen_returns = rollout_batch.returns.copy()
    frozen_mc_returns = rollout_batch.monte_carlo_returns.copy()
    frozen_actions = rollout_batch.actions.copy()
    frozen_masks = rollout_batch.action_masks.copy()

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approx_kls: list[float] = []
    clip_fractions: list[float] = []
    gradient_norms: list[float] = []
    epoch_summaries: list[PPOEpochSummary] = []
    gradient_finite_all = True
    each_transition_once_per_epoch = True
    optimizer_steps = 0

    all_indices = np.arange(batch_size, dtype=np.int64)
    for epoch_index, shuffled in enumerate(epoch_shuffles, start=1):
        each_transition_once_per_epoch = each_transition_once_per_epoch and bool(
            np.array_equal(np.sort(shuffled), all_indices)
        )
        epoch_minibatch_diagnostics: list[dict[str, float | int]] = []
        epoch_pre_clipping_grad_norms: list[float] = []
        for start in range(0, batch_size, minibatch_size):
            minibatch_indices = shuffled[start : start + minibatch_size]
            if len(minibatch_indices) != minibatch_size:
                raise RuntimeError("Actor minibatch划分不完整。")
            minibatch = _slice_rollout_batch(rollout_batch, minibatch_indices)
            frozen_advantage_mb = frozen_counterfactual_advantages[minibatch_indices]
            optimizer.zero_grad(set_to_none=True)
            loss = ppo_update.compute_per_trigger_ppo_loss(
                policy,
                minibatch,
                frozen_advantage_mb,
                float(training_cfg.clip_epsilon),
                float(training_cfg.value_loss_coefficient),
                float(training_cfg.entropy_coefficient),
            )
            if not torch.isfinite(loss.total_loss):
                raise FloatingPointError("per-trigger PPO total loss 非finite。")
            loss.total_loss.backward()
            grads = [
                parameter.grad.detach()
                for parameter in policy.parameters()
                if parameter.grad is not None
            ]
            gradient_finite = bool(grads) and bool(
                all(torch.isfinite(grad).all().item() for grad in grads)
            )
            gradient_finite_all = gradient_finite_all and gradient_finite
            if not gradient_finite:
                raise FloatingPointError("Actor/scalar-V gradient 出现 NaN/Inf。")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                float(training_cfg.max_grad_norm),
            )
            pre_clipping_grad_norm = float(grad_norm.detach().cpu())
            epoch_pre_clipping_grad_norms.append(pre_clipping_grad_norm)
            gradient_norms.append(pre_clipping_grad_norm)
            minibatch_diagnostics = {
                "transition_count": int(len(minibatch_indices)),
                "policy_loss": float(loss.policy_loss.detach().cpu()),
                "value_loss": float(loss.value_loss.detach().cpu()),
                "entropy": float(loss.entropy.detach().cpu()),
                "total_loss": float(loss.total_loss.detach().cpu()),
                "approx_kl": float(loss.approx_kl),
                "clip_fraction": float(loss.clip_fraction),
                "ratio_mean": float(loss.ratio_mean),
                "ratio_min": float(loss.ratio_min),
                "ratio_max": float(loss.ratio_max),
            }
            epoch_minibatch_diagnostics.append(minibatch_diagnostics)
            optimizer.step()
            optimizer_steps += 1
            policy_losses.append(minibatch_diagnostics["policy_loss"])
            value_losses.append(minibatch_diagnostics["value_loss"])
            entropies.append(minibatch_diagnostics["entropy"])
            approx_kls.append(minibatch_diagnostics["approx_kl"])
            clip_fractions.append(minibatch_diagnostics["clip_fraction"])
        epoch_summaries.append(
            _aggregate_epoch_diagnostics(
                epoch_index=epoch_index,
                minibatch_diagnostics=epoch_minibatch_diagnostics,
                pre_clipping_grad_norms=epoch_pre_clipping_grad_norms,
                optimizer_steps_cumulative=optimizer_steps,
            )
        )

    if not np.array_equal(rollout_batch.old_log_probs, frozen_old_log_probs):
        raise AssertionError("joint old_log_probs 在per-trigger update期间被修改。")
    if not np.array_equal(rollout_batch.old_per_action_log_probs, frozen_old_per_action_log_probs):
        raise AssertionError("old per-action log-probs 在update期间被修改。")
    if not np.array_equal(rollout_batch.old_per_action_probabilities, frozen_old_per_action_probabilities):
        raise AssertionError("old per-action probabilities 在update期间被修改。")
    if not np.array_equal(rollout_batch.old_values, frozen_old_values):
        raise AssertionError("old_values 在per-trigger update期间被修改。")
    if not np.array_equal(rollout_batch.returns, frozen_returns):
        raise AssertionError("GAE returns 在per-trigger update期间被修改。")
    if not np.array_equal(rollout_batch.monte_carlo_returns, frozen_mc_returns):
        raise AssertionError("Monte-Carlo returns 在per-trigger update期间被修改。")
    if not np.array_equal(rollout_batch.actions, frozen_actions):
        raise AssertionError("rollout actions 在per-trigger update期间被修改。")
    if not np.array_equal(rollout_batch.action_masks, frozen_masks):
        raise AssertionError("rollout action masks 在per-trigger update期间被修改。")
    if not torch.equal(targets.advantages, frozen_counterfactual_advantages):
        raise AssertionError("counterfactual advantages 在PPO multi-epoch期间被修改。")
    if not torch.equal(targets.actor_valid_mask, frozen_actor_valid_mask):
        raise AssertionError("actor_valid_mask 在PPO multi-epoch期间被修改。")

    update_summary = FormalPPOUpdateSummary(
        update_epochs=update_epochs,
        num_minibatches=num_minibatches,
        minibatch_size=minibatch_size,
        optimizer_steps=optimizer_steps,
        policy_loss_mean=float(np.mean(policy_losses)),
        value_loss_mean=float(np.mean(value_losses)),
        entropy_mean=float(np.mean(entropies)),
        approx_kl_mean=float(np.mean(approx_kls)),
        clip_fraction_mean=float(np.mean(clip_fractions)),
        gradient_norm_mean=float(np.mean(gradient_norms)),
        gradient_finite=gradient_finite_all,
        old_log_probs_frozen=True,
        old_values_frozen=True,
        advantages_frozen=True,
        returns_frozen=True,
        each_transition_once_per_epoch=each_transition_once_per_epoch,
        epoch_summaries=tuple(epoch_summaries),
        q_loss_mean=float(np.mean(q_losses)),
        q_gradient_norm_mean=float(np.mean(q_gradient_norms)),
        q_gradient_finite=q_gradient_finite_all,
        counterfactual_advantage_mean=cf_adv_mean,
        counterfactual_advantage_std=cf_adv_std,
        counterfactual_advantage_abs_mean=cf_adv_abs_mean,
        free_trigger_decision_count=int(frozen_actor_valid_mask.sum().detach().cpu()),
        q_optimizer_steps=q_optimizer_steps,
    )
    return update_summary


def _train_centralized_critic_on_replay(
    q_critic: Any,
    q_optimizer: Any,
    replay: ppo_rollout.CriticReplayBuffer,
    project_cfg: dict[str, Any],
    critic_rng_generator: Any,
    critic_return_scale: float,
    evaluation_batch: ppo_rollout.RolloutBatch | None = None,
) -> dict[str, Any]:
    """在最近完整rollout replay上训练 central critic。

    replay 是 bounded、按完整rollout保留的 supervised 数据；每个 epoch 对
    当前 replay 样本恰好无放回遍历一次。最后一个 minibatch 可以短于配置的
    num_minibatches 等分大小，这是 central critic 的实现边界，不改变 Actor
    的 on-policy update 规则。
    """

    _require_torch()
    training_cfg = ppo_update.ppo_training_config_status(project_cfg)
    if not training_cfg.can_run_optimizer_step or training_cfg.max_grad_norm is None:
        raise RuntimeError(f"central critic 超参数缺失：{training_cfg.missing_required_for_step}")
    if ppo_update.q_target_from_project(project_cfg) != ppo_update.MONTE_CARLO_Q_TARGET:
        raise RuntimeError("centralized counterfactual critic 必须使用 q_target=monte_carlo。")
    if critic_rng_generator is None:
        raise ValueError("centralized critic 必须使用独立 critic_rng_generator。")
    return_scale = float(critic_return_scale)
    if not math.isfinite(return_scale) or return_scale <= 0.0:
        raise ValueError("centralized critic_return_scale 必须是有限正数。")
    observations, actions, targets = replay.arrays()
    sample_count = int(observations.shape[0])
    update_epochs = int(project_config.get_config_value(project_cfg, "algorithm.ppo.update_epochs"))
    num_minibatches = int(project_config.get_config_value(project_cfg, "algorithm.ppo.num_minibatches"))
    if sample_count <= 0:
        raise ValueError("central critic replay 不能为空。")

    if evaluation_batch is None:
        evaluation_observations, evaluation_actions, evaluation_targets = observations, actions, targets
    else:
        evaluation_observations = evaluation_batch.observations
        evaluation_actions = evaluation_batch.actions
        evaluation_targets = evaluation_batch.monte_carlo_returns
    with torch.no_grad():
        before = ppo_update.compute_centralized_q_loss(
            q_critic,
            evaluation_observations,
            evaluation_actions,
            evaluation_targets,
            return_scale,
        ).q_loss
        critic_loss_before = float(before.detach().cpu())

    before_parameters = [parameter.detach().cpu().clone() for parameter in q_critic.parameters()]
    gradient_norms: list[float] = []
    clip_scales: list[float] = []
    gradient_finite_all = True
    optimizer_steps = 0
    all_indices = np.arange(sample_count, dtype=np.int64)
    for _epoch in range(update_epochs):
        shuffled = _torch_randperm_numpy(sample_count, critic_rng_generator)
        if not np.array_equal(np.sort(shuffled), all_indices):
            raise AssertionError("central critic replay shuffle 不是完整无放回排列。")
        for minibatch_indices in np.array_split(shuffled, num_minibatches):
            if len(minibatch_indices) == 0:
                continue
            q_optimizer.zero_grad(set_to_none=True)
            loss_result = ppo_update.compute_centralized_q_loss(
                q_critic,
                observations[minibatch_indices],
                actions[minibatch_indices],
                targets[minibatch_indices],
                return_scale,
            )
            if not torch.isfinite(loss_result.q_loss):
                raise FloatingPointError("central critic loss 非finite。")
            loss_result.q_loss.backward()
            grads = [
                parameter.grad.detach()
                for parameter in q_critic.parameters()
                if parameter.grad is not None
            ]
            gradient_finite = bool(grads) and bool(
                all(torch.isfinite(grad).all().item() for grad in grads)
            )
            gradient_finite_all = gradient_finite_all and gradient_finite
            if not gradient_finite:
                raise FloatingPointError("central critic gradient 出现 NaN/Inf。")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                q_critic.parameters(), float(training_cfg.max_grad_norm)
            )
            raw_grad_norm = float(grad_norm.detach().cpu())
            gradient_norms.append(raw_grad_norm)
            clip_scales.append(
                1.0
                if raw_grad_norm <= 0.0
                else min(1.0, float(training_cfg.max_grad_norm) / raw_grad_norm)
            )
            q_optimizer.step()
            optimizer_steps += 1

    q_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        after = ppo_update.compute_centralized_q_loss(
            q_critic,
            evaluation_observations,
            evaluation_actions,
            evaluation_targets,
            return_scale,
        ).q_loss
        critic_loss_after = float(after.detach().cpu())
    parameter_delta = math.sqrt(
        math.fsum(
            float(torch.sum((before - parameter.detach().cpu()) ** 2).item())
            for before, parameter in zip(before_parameters, q_critic.parameters())
        )
    )
    return {
        "critic_loss_before": critic_loss_before,
        "critic_loss_after": critic_loss_after,
        "critic_gradient_norm_mean": float(np.mean(gradient_norms)) if gradient_norms else 0.0,
        "critic_gradient_norm_max": float(np.max(gradient_norms)) if gradient_norms else 0.0,
        "critic_clip_scale_mean": float(np.mean(clip_scales)) if clip_scales else 1.0,
        "critic_clip_scale_min": float(np.min(clip_scales)) if clip_scales else 1.0,
        "critic_gradient_finite": bool(gradient_finite_all),
        "critic_parameter_delta": float(parameter_delta),
        "critic_optimizer_steps": int(optimizer_steps),
        "critic_replay_sample_count": sample_count,
        "critic_return_scale": return_scale,
    }


def _update_policy_on_rollout_centralized(
    policy: Any,
    optimizer: Any,
    q_critic: Any,
    q_optimizer: Any,
    rollout_batch: ppo_rollout.RolloutBatch,
    project_cfg: dict[str, Any],
    rng_generator: Any,
    critic_replay: ppo_rollout.CriticReplayBuffer,
    critic_rng_generator: Any,
    critic_return_scale: float,
    credit_phase: str,
) -> FormalPPOUpdateSummary:
    """执行 V2 centralized critic-first update。"""

    _require_torch()
    if credit_phase not in {"critic_warmup", "joint_training"}:
        raise ValueError(f"未知 credit_phase：{credit_phase}")
    if not bool(rollout_batch.metadata.get("true_terminal", False)):
        raise RuntimeError("centralized counterfactual Q 要求 full true-terminal rollout。")
    if rollout_batch.monte_carlo_returns.shape != rollout_batch.rewards.shape:
        raise ValueError("monte_carlo_returns shape 必须与 rewards 一致。")
    critic_replay.append_rollout(rollout_batch)
    critic_stats = _train_centralized_critic_on_replay(
        q_critic,
        q_optimizer,
        critic_replay,
        project_cfg,
        critic_rng_generator,
        critic_return_scale,
        evaluation_batch=rollout_batch,
    )
    gap_targets = ppo_update.prepare_centralized_counterfactual_targets(
        q_critic,
        rollout_batch,
        critic_return_scale,
    )
    gap_values = gap_targets.action_gaps.detach().cpu().numpy() * float(critic_return_scale)
    gap_abs_mean = float(np.mean(np.abs(gap_values)))
    gap_std = float(np.std(gap_values))
    gap_nonzero_fraction = float(np.mean(np.abs(gap_values) > 1.0e-12))

    update_epochs = int(project_config.get_config_value(project_cfg, "algorithm.ppo.update_epochs"))
    num_minibatches = int(project_config.get_config_value(project_cfg, "algorithm.ppo.num_minibatches"))
    batch_size = int(rollout_batch.observations.shape[0])
    if batch_size <= 0 or batch_size % num_minibatches != 0:
        raise ValueError("Actor rollout batch size 必须为正且可被 num_minibatches 整除。")
    minibatch_size = batch_size // num_minibatches
    if credit_phase == "critic_warmup":
        return FormalPPOUpdateSummary(
            update_epochs=update_epochs,
            num_minibatches=num_minibatches,
            minibatch_size=minibatch_size,
            optimizer_steps=0,
            policy_loss_mean=0.0,
            value_loss_mean=0.0,
            entropy_mean=0.0,
            approx_kl_mean=0.0,
            clip_fraction_mean=0.0,
            gradient_norm_mean=0.0,
            gradient_finite=True,
            old_log_probs_frozen=True,
            old_values_frozen=True,
            advantages_frozen=True,
            returns_frozen=True,
            each_transition_once_per_epoch=True,
            epoch_summaries=tuple(),
            q_loss_mean=critic_stats["critic_loss_after"],
            q_gradient_norm_mean=critic_stats["critic_gradient_norm_mean"],
            q_gradient_finite=critic_stats["critic_gradient_finite"],
            free_trigger_decision_count=int(gap_targets.actor_valid_mask.sum().detach().cpu()),
            q_optimizer_steps=critic_stats["critic_optimizer_steps"],
            credit_phase=credit_phase,
            critic_loss_before=critic_stats["critic_loss_before"],
            critic_loss_after=critic_stats["critic_loss_after"],
            critic_gradient_norm_mean=critic_stats["critic_gradient_norm_mean"],
            critic_gradient_finite=critic_stats["critic_gradient_finite"],
            critic_action_gap_abs_mean=gap_abs_mean,
            critic_action_gap_std=gap_std,
            critic_action_gap_nonzero_fraction=gap_nonzero_fraction,
            counterfactual_advantage_abs_mean=None,
            actor_update_executed=False,
            critic_replay_sample_count=critic_stats["critic_replay_sample_count"],
        )

    training_cfg = ppo_update.ppo_training_config_status(project_cfg)
    if not training_cfg.can_run_optimizer_step or training_cfg.max_grad_norm is None:
        raise RuntimeError(f"PPO update超参数缺失：{training_cfg.missing_required_for_step}")
    frozen_advantages = gap_targets.advantages.detach().clone()
    frozen_valid_mask = gap_targets.actor_valid_mask.detach().clone()
    free_advantages = frozen_advantages[frozen_valid_mask]
    if int(free_advantages.numel()) > 0:
        cf_adv_mean = float(torch.mean(free_advantages).detach().cpu())
        cf_adv_std = float(torch.std(free_advantages, unbiased=False).detach().cpu())
        cf_adv_abs_mean = float(torch.mean(torch.abs(free_advantages)).detach().cpu())
    else:
        cf_adv_mean = cf_adv_std = cf_adv_abs_mean = 0.0

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approx_kls: list[float] = []
    clip_fractions: list[float] = []
    gradient_norms: list[float] = []
    epoch_summaries: list[PPOEpochSummary] = []
    gradient_finite_all = True
    each_transition_once_per_epoch = True
    all_indices = np.arange(batch_size, dtype=np.int64)
    optimizer_steps = 0
    for epoch_index in range(1, update_epochs + 1):
        shuffled = _torch_randperm_numpy(batch_size, rng_generator)
        each_transition_once_per_epoch = each_transition_once_per_epoch and bool(
            np.array_equal(np.sort(shuffled), all_indices)
        )
        epoch_minibatch_diagnostics: list[dict[str, float | int]] = []
        epoch_pre_clipping_grad_norms: list[float] = []
        for start in range(0, batch_size, minibatch_size):
            minibatch_indices = shuffled[start : start + minibatch_size]
            minibatch = _slice_rollout_batch(rollout_batch, minibatch_indices)
            optimizer.zero_grad(set_to_none=True)
            loss = ppo_update.compute_per_trigger_ppo_loss(
                policy,
                minibatch,
                frozen_advantages[minibatch_indices],
                float(training_cfg.clip_epsilon),
                float(training_cfg.value_loss_coefficient),
                float(training_cfg.entropy_coefficient),
            )
            if not torch.isfinite(loss.total_loss):
                raise FloatingPointError("centralized Actor PPO total loss 非finite。")
            loss.total_loss.backward()
            grads = [
                parameter.grad.detach()
                for parameter in policy.parameters()
                if parameter.grad is not None
            ]
            gradient_finite = bool(grads) and bool(
                all(torch.isfinite(grad).all().item() for grad in grads)
            )
            gradient_finite_all = gradient_finite_all and gradient_finite
            if not gradient_finite:
                raise FloatingPointError("centralized Actor gradient 出现 NaN/Inf。")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), float(training_cfg.max_grad_norm)
            )
            pre_grad_norm = float(grad_norm.detach().cpu())
            gradient_norms.append(pre_grad_norm)
            epoch_pre_clipping_grad_norms.append(pre_grad_norm)
            diagnostics = {
                "transition_count": int(len(minibatch_indices)),
                "policy_loss": float(loss.policy_loss.detach().cpu()),
                "value_loss": float(loss.value_loss.detach().cpu()),
                "entropy": float(loss.entropy.detach().cpu()),
                "total_loss": float(loss.total_loss.detach().cpu()),
                "approx_kl": float(loss.approx_kl),
                "clip_fraction": float(loss.clip_fraction),
                "ratio_mean": float(loss.ratio_mean),
                "ratio_min": float(loss.ratio_min),
                "ratio_max": float(loss.ratio_max),
            }
            epoch_minibatch_diagnostics.append(diagnostics)
            optimizer.step()
            optimizer_steps += 1
            policy_losses.append(diagnostics["policy_loss"])
            value_losses.append(diagnostics["value_loss"])
            entropies.append(diagnostics["entropy"])
            approx_kls.append(diagnostics["approx_kl"])
            clip_fractions.append(diagnostics["clip_fraction"])
        epoch_summaries.append(
            _aggregate_epoch_diagnostics(
                epoch_index,
                epoch_minibatch_diagnostics,
                epoch_pre_clipping_grad_norms,
                optimizer_steps,
            )
        )
    q_optimizer.zero_grad(set_to_none=True)
    return FormalPPOUpdateSummary(
        update_epochs=update_epochs,
        num_minibatches=num_minibatches,
        minibatch_size=minibatch_size,
        optimizer_steps=optimizer_steps,
        policy_loss_mean=float(np.mean(policy_losses)),
        value_loss_mean=float(np.mean(value_losses)),
        entropy_mean=float(np.mean(entropies)),
        approx_kl_mean=float(np.mean(approx_kls)),
        clip_fraction_mean=float(np.mean(clip_fractions)),
        gradient_norm_mean=float(np.mean(gradient_norms)),
        gradient_finite=gradient_finite_all,
        old_log_probs_frozen=True,
        old_values_frozen=True,
        advantages_frozen=True,
        returns_frozen=True,
        each_transition_once_per_epoch=each_transition_once_per_epoch,
        epoch_summaries=tuple(epoch_summaries),
        q_loss_mean=critic_stats["critic_loss_after"],
        q_gradient_norm_mean=critic_stats["critic_gradient_norm_mean"],
        q_gradient_finite=critic_stats["critic_gradient_finite"],
        counterfactual_advantage_mean=cf_adv_mean,
        counterfactual_advantage_std=cf_adv_std,
        counterfactual_advantage_abs_mean=cf_adv_abs_mean,
        free_trigger_decision_count=int(frozen_valid_mask.sum().detach().cpu()),
        q_optimizer_steps=critic_stats["critic_optimizer_steps"],
        credit_phase=credit_phase,
        critic_loss_before=critic_stats["critic_loss_before"],
        critic_loss_after=critic_stats["critic_loss_after"],
        critic_gradient_norm_mean=critic_stats["critic_gradient_norm_mean"],
        critic_gradient_finite=critic_stats["critic_gradient_finite"],
        critic_action_gap_abs_mean=gap_abs_mean,
        critic_action_gap_std=gap_std,
        critic_action_gap_nonzero_fraction=gap_nonzero_fraction,
        actor_update_executed=True,
        critic_replay_sample_count=critic_stats["critic_replay_sample_count"],
    )


def _update_policy_on_sparse_direct_cf(
    policy: Any,
    optimizer: Any,
    direct_cf_samples: list[ppo_rollout.DirectCFCreditSample],
    project_cfg: dict[str, Any],
    expected_collection_policy_version: int | None = None,
    reference_records: list[ppo_rollout.ReferenceStateRecord] | None = None,
    reference_probability_drift_cap: float | None = None,
    gate_optimizer: Any | None = None,
    gate_cf_samples: list[Any] | None = None,
) -> FormalPPOUpdateSummary:
    """执行 V3 sparse Direct-CF Actor-only full-batch update。"""

    _require_torch()
    actor_policy = getattr(policy, "sub_policy", policy)
    training_cfg = ppo_update.ppo_training_config_status(project_cfg)
    if not training_cfg.can_run_optimizer_step or training_cfg.learning_rate is None:
        raise RuntimeError(f"sparse_direct_cf PPO超参数缺失：{training_cfg.missing_required_for_step}")
    if training_cfg.max_grad_norm is None:
        raise RuntimeError("sparse_direct_cf 缺少 max_grad_norm。")
    if not direct_cf_samples:
        raise RuntimeError("sparse_direct_cf 本episode没有成功的 FREE paired branch sample。")

    collection_versions = {int(item.collection_policy_version) for item in direct_cf_samples}
    if len(collection_versions) != 1:
        raise RuntimeError(
            "Direct-CF batch不得混合多个collection_policy_version："
            f"{sorted(collection_versions)}"
        )
    collection_policy_version = next(iter(collection_versions))
    if expected_collection_policy_version is not None and collection_policy_version != int(
        expected_collection_policy_version
    ):
        raise RuntimeError(
            "Direct-CF batch collection_policy_version与当前policy version不一致："
            f"{collection_policy_version} != {expected_collection_policy_version}"
        )

    actor_parameters = ppo_trigger_policy.sparse_direct_cf_actor_parameters(actor_policy)
    value_parameters = [
        parameter
        for name, parameter in actor_policy.named_parameters()
        if name.startswith("value_head.")
    ]
    autoregressive_parameters = [
        parameter
        for name, parameter in actor_policy.named_parameters()
        if name.startswith(
            (
                "initial_hidden.",
                "previous_action_embedding.",
                "action_index_embedding.",
                "gru_cell.",
            )
        )
    ]
    full_shared_actor = (
        ppo_trigger_policy.actor_architecture_id(actor_policy)
        == ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID
    )
    frozen_parameters = value_parameters if full_shared_actor else value_parameters + autoregressive_parameters
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if any(id(parameter) not in optimizer_parameter_ids for parameter in actor_parameters):
        raise RuntimeError("sparse_direct_cf Actor optimizer缺少shared Actor参数。")
    if any(id(parameter) in optimizer_parameter_ids for parameter in frozen_parameters):
        raise RuntimeError(
            "sparse_direct_cf frozen parameter group不得进入Actor optimizer。"
        )
    if any(parameter.requires_grad for parameter in frozen_parameters):
        raise RuntimeError("sparse_direct_cf frozen parameter group必须requires_grad=False。")

    parameters_before_update = {
        name: parameter.detach().clone()
        for name, parameter in actor_policy.named_parameters()
    }

    observations = np.stack([item.observation for item in direct_cf_samples]).astype(np.float32)
    actions = np.stack([item.action_vector for item in direct_cf_samples]).astype(np.int8)
    action_masks = np.stack([item.action_mask for item in direct_cf_samples]).astype(bool)
    target_indices = np.asarray(
        [item.trigger_index for item in direct_cf_samples],
        dtype=np.int64,
    )
    old_target_log_probs = np.asarray(
        [item.old_chosen_log_prob for item in direct_cf_samples],
        dtype=np.float32,
    )
    deltas = np.asarray([item.delta_g for item in direct_cf_samples], dtype=np.float64)
    a_cf = np.asarray([item.a_cf for item in direct_cf_samples], dtype=np.float64)
    sample_weights = np.asarray(
        [item.sampling_weight for item in direct_cf_samples],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(sample_weights)) or np.any(sample_weights < 0.0):
        raise ValueError("V3 Direct-CF sampling weights 必须是finite非负数。")
    if not math.isclose(float(np.sum(sample_weights)), 1.0, rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError("V3 Direct-CF sampling weights之和必须为1。")
    expected_a_cf = np.asarray(
        [
            (item.actual_action - item.old_p1) * item.delta_g
            for item in direct_cf_samples
        ],
        dtype=np.float64,
    )
    if not np.allclose(a_cf, expected_a_cf, rtol=1e-7, atol=1e-8):
        raise AssertionError("V3 A_cf 必须严格使用 (a-p_old)*Delta_G。")

    update_epochs = int(project_config.get_config_value(project_cfg, "algorithm.ppo.update_epochs"))
    if update_epochs < 1:
        raise ValueError("sparse_direct_cf update_epochs 必须 >= 1。")
    reference_drift_mode = _is_what_reference_drift_mode(project_cfg)
    old_p_reconstruction_max_error: float | None = None
    old_p_reconstruction_checked: bool | None = None
    old_p_reconstruction_tolerance: float | None = None
    probability_dtype: str | None = None
    reference_count_checked: int | None = None
    reference_collection_policy_version: int | None = None
    if reference_drift_mode:
        if update_epochs != 1:
            raise ValueError("refdrift正式update固定要求update_epochs=1。")
        if reference_records is None or not reference_records:
            raise ValueError("refdrift update必须提供当前batch的R_all reference records。")
        configured_cap = float(
            project_config.get_config_value(
                project_cfg,
                "algorithm.ppo.reference_probability_drift_cap",
                reference_drift.DEFAULT_REFERENCE_PROBABILITY_DRIFT_CAP,
            )
        )
        if configured_cap != 0.005:
            raise ValueError("refdrift正式实现固定要求reference_probability_drift_cap=0.005。")
        if reference_probability_drift_cap is not None and float(reference_probability_drift_cap) != configured_cap:
            raise ValueError("refdrift update cap与当前配置不一致。")
        reference_probability_drift_cap = configured_cap
        selected_ids_for_gate = [str(item.decision_id) for item in direct_cf_samples]
        _, unselected_reference_records, partition = reference_drift.validate_reference_partition(
            reference_records,
            selected_ids_for_gate,
        )
        (
            old_p_reconstruction_max_error,
            old_p_reconstruction_tolerance,
            probability_dtype,
        ) = reference_drift.old_p_reconstruction_diagnostics(
            actor_policy,
            unselected_reference_records,
        )
        old_p_reconstruction_checked = True
        reference_count_checked = int(partition["reference_count"])
        reference_collection_policy_versions = {
            int(item.collection_policy_version) for item in reference_records
        }
        if len(reference_collection_policy_versions) != 1:
            raise RuntimeError(
                "RefDrift reference records不得混合多个collection_policy_version："
                f"{sorted(reference_collection_policy_versions)}"
            )
        reference_collection_policy_version = next(iter(reference_collection_policy_versions))
        if old_p_reconstruction_max_error > old_p_reconstruction_tolerance:
            raise FloatingPointError(
                "RefDrift old-p reconstruction error超过dtype-aware一致性容差："
                f"{old_p_reconstruction_max_error} > "
                f"{old_p_reconstruction_tolerance} ({probability_dtype})"
            )
    elif reference_records is not None or reference_probability_drift_cap is not None:
        raise ValueError("reference drift records/cap只能用于refdrift正式方法。")
    sample_count = len(direct_cf_samples)
    policy_losses: list[float] = []
    approx_kls: list[float] = []
    clip_fractions: list[float] = []
    gradient_norms: list[float] = []
    epoch_summaries: list[PPOEpochSummary] = []
    gradient_finite_all = True
    for epoch_index in range(1, update_epochs + 1):
        # 只清除 Actor 参数；显式清除 value_head 既有梯度，防止任何隐式残留。
        actor_policy.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        loss = ppo_update.compute_sparse_direct_cf_actor_loss(
            actor_policy,
            observations,
            actions,
            action_masks,
            target_indices,
            old_target_log_probs,
            a_cf.astype(np.float32),
            float(training_cfg.clip_epsilon),
            sample_weights=sample_weights.astype(np.float32),
        )
        if not torch.isfinite(loss.total_loss):
            raise FloatingPointError("sparse_direct_cf Actor loss 非finite。")
        loss.total_loss.backward()
        if any(parameter.grad is not None for parameter in value_parameters):
            raise AssertionError("sparse_direct_cf value_head 不应产生gradient。")
        if not full_shared_actor and any(parameter.grad is not None for parameter in autoregressive_parameters):
            raise AssertionError("sparse_direct_cf autoregressive block不应产生gradient。")
        grads = [parameter.grad.detach() for parameter in actor_parameters if parameter.grad is not None]
        gradient_finite = bool(grads) and bool(
            all(torch.isfinite(grad).all().item() for grad in grads)
        )
        gradient_finite_all = gradient_finite_all and gradient_finite
        if not gradient_finite:
            raise FloatingPointError("sparse_direct_cf Actor gradient 出现 NaN/Inf。")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            actor_parameters,
            float(training_cfg.max_grad_norm),
        )
        pre_grad_norm = float(grad_norm.detach().cpu())
        if not math.isfinite(pre_grad_norm):
            raise FloatingPointError("sparse_direct_cf Actor gradient norm 非finite。")
        optimizer.step()
        policy_losses.append(float(loss.policy_loss.detach().cpu()))
        approx_kls.append(float(loss.approx_kl))
        clip_fractions.append(float(loss.clip_fraction))
        gradient_norms.append(pre_grad_norm)
        epoch_summaries.append(
            PPOEpochSummary(
                epoch_index=epoch_index,
                policy_loss_mean=float(loss.policy_loss.detach().cpu()),
                value_loss_mean=0.0,
                entropy_mean=0.0,
                total_loss_mean=float(loss.total_loss.detach().cpu()),
                approx_kl_mean=float(loss.approx_kl),
                clip_fraction_mean=float(loss.clip_fraction),
                ratio_mean=float(loss.ratio_mean),
                ratio_min=float(loss.ratio_min),
                ratio_max=float(loss.ratio_max),
                pre_clipping_grad_norm_mean=pre_grad_norm,
                pre_clipping_grad_norm_max=pre_grad_norm,
                minibatch_count=1,
                transition_count=sample_count,
                optimizer_steps_cumulative=epoch_index,
            )
        )

    reference_gate_result: reference_drift.ReferenceDriftGateResult | None = None
    raw_p1_after: np.ndarray
    if reference_drift_mode:
        raw_parameters = {
            name: parameter.detach().clone()
            for name, parameter in actor_policy.named_parameters()
        }
        raw_p1_after = reference_drift.conditional_probability_one(actor_policy, direct_cf_samples)
        reference_gate_result = reference_drift.constrain_after_adam_step(
            actor_policy,
            parameters_before_update,
            raw_parameters,
            reference_records or [],
            selected_ids_for_gate,
            float(reference_probability_drift_cap),
            tolerance=old_p_reconstruction_tolerance,
        )
        gate_probability_dtype = reference_gate_result.probability_dtype
        if gate_probability_dtype != probability_dtype:
            raise AssertionError(
                "RefDrift probability dtype在old-p与candidate gate间不一致："
                f"{probability_dtype} != {gate_probability_dtype}"
            )
        p1_after = reference_drift.conditional_probability_one(actor_policy, direct_cf_samples)
    else:
        with torch.no_grad():
            evaluation = actor_policy.evaluate_actions(
                torch.as_tensor(observations, dtype=torch.float32),
                torch.as_tensor(actions, dtype=torch.float32),
                action_masks=torch.as_tensor(action_masks, dtype=torch.bool),
            )
            rows = torch.arange(sample_count, device=evaluation.per_action_probabilities.device)
            p1_after = evaluation.per_action_probabilities[rows, torch.as_tensor(
                target_indices,
                dtype=torch.long,
                device=evaluation.per_action_probabilities.device,
            )].detach().cpu().numpy().astype(np.float64)
        raw_p1_after = p1_after.copy()
    p1_before = np.asarray([item.old_p1 for item in direct_cf_samples], dtype=np.float64)
    raw_vshift = (raw_p1_after - p1_before) * deltas
    vshift = (p1_after - p1_before) * deltas
    local_proxy = float(math.fsum(float(value) for value in vshift))
    cf_adv_mean = float(np.mean(a_cf))
    cf_adv_std = float(np.std(a_cf))
    cf_adv_abs_mean = float(np.mean(np.abs(a_cf)))

    def parameter_delta_norm(prefixes: tuple[str, ...]) -> float:
        squared = 0.0
        for name, parameter in actor_policy.named_parameters():
            if name.startswith(prefixes):
                delta = parameter.detach() - parameters_before_update[name]
                squared += float(torch.sum(delta * delta).cpu())
        return float(math.sqrt(squared))

    encoder_delta = parameter_delta_norm(("encoder.",))
    trigger_heads_delta = parameter_delta_norm(("trigger_heads.",))
    autoregressive_delta = parameter_delta_norm(
        (
            "initial_hidden.",
            "previous_action_embedding.",
            "action_index_embedding.",
            "gru_cell.",
        )
    )
    logit_head_delta = parameter_delta_norm(("logit_head.",))
    value_head_delta = parameter_delta_norm(("value_head.",))
    if not all(
        math.isfinite(value)
        for value in (encoder_delta, trigger_heads_delta, autoregressive_delta, logit_head_delta)
    ):
        raise FloatingPointError("sparse_direct_cf Actor trainable parameter delta 非finite。")
    if not full_shared_actor and autoregressive_delta != 0.0:
        raise AssertionError(
            "sparse_direct_cf frozen autoregressive parameter delta failed："
            f"AR={autoregressive_delta}"
        )
    if value_head_delta != 0.0:
        raise AssertionError(
            "sparse_direct_cf frozen value_head parameter delta failed："
            f"value={value_head_delta}"
        )
    if not np.all(np.isfinite(vshift)):
        raise FloatingPointError("sparse_direct_cf Vshift出现NaN或Inf。")
    if not np.all(np.isfinite(raw_vshift)):
        raise FloatingPointError("sparse_direct_cf raw Vshift出现NaN或Inf。")
    raw_vshift_sum = float(math.fsum(float(value) for value in raw_vshift))
    constrained_vshift_sum = float(math.fsum(float(value) for value in vshift))
    vshift_retention_ratio = (
        float(constrained_vshift_sum / raw_vshift_sum)
        if raw_vshift_sum != 0.0
        else None
    )
    reference_fields: dict[str, Any] = {}
    if reference_gate_result is not None:
        gate = reference_gate_result
        reference_fields = {
            "reference_raw_drift_mean": gate.raw_reference_drift["mean"],
            "reference_raw_drift_median": gate.raw_reference_drift["median"],
            "reference_raw_drift_p95": gate.raw_reference_drift["p95"],
            "reference_raw_drift_p99": gate.raw_reference_drift["p99"],
            "reference_raw_drift_max": gate.raw_reference_drift["max"],
            "reference_constrained_drift_mean": gate.constrained_reference_drift["mean"],
            "reference_constrained_drift_median": gate.constrained_reference_drift["median"],
            "reference_constrained_drift_p95": gate.constrained_reference_drift["p95"],
            "reference_constrained_drift_p99": gate.constrained_reference_drift["p99"],
            "reference_constrained_drift_max": gate.constrained_reference_drift["max"],
            "selected_drift_mean": gate.constrained_selected_drift["mean"],
            "selected_drift_p95": gate.constrained_selected_drift["p95"],
            "selected_drift_max": gate.constrained_selected_drift["max"],
            "reference_drift_alpha": gate.alpha,
            "reference_drift_constraint_active": gate.constraint_active,
            "reference_probability_drift_cap": gate.cap,
            "reference_state_count": gate.reference_count + gate.selected_count,
            "reference_unselected_count": gate.reference_count,
            "reference_selected_count": gate.selected_count,
            "reference_selected_intersection_count": gate.intersection_count,
            "reference_drift_by_trigger": gate.per_trigger,
            "old_p_reconstruction_max_error": old_p_reconstruction_max_error,
            "old_p_reconstruction_checked": old_p_reconstruction_checked,
            "old_p_reconstruction_tolerance": old_p_reconstruction_tolerance,
            "probability_dtype": probability_dtype,
            "reference_count_checked": reference_count_checked,
            "collection_policy_version": reference_collection_policy_version,
            "direct_cf_raw_vshift_sum": raw_vshift_sum,
            "direct_cf_raw_vshift_mean": float(np.mean(raw_vshift)),
            "direct_cf_raw_vshift_positive_fraction": float(np.mean(raw_vshift > 0.0)),
            "direct_cf_constrained_vshift_sum": constrained_vshift_sum,
            "direct_cf_constrained_vshift_mean": float(np.mean(vshift)),
            "direct_cf_constrained_vshift_positive_fraction": float(np.mean(vshift > 0.0)),
            "direct_cf_vshift_retention_ratio": vshift_retention_ratio,
        }
    return FormalPPOUpdateSummary(
        update_epochs=update_epochs,
        num_minibatches=1,
        minibatch_size=sample_count,
        optimizer_steps=update_epochs,
        policy_loss_mean=float(np.mean(policy_losses)),
        value_loss_mean=0.0,
        entropy_mean=0.0,
        approx_kl_mean=float(np.mean(approx_kls)),
        clip_fraction_mean=float(np.mean(clip_fractions)),
        gradient_norm_mean=float(np.mean(gradient_norms)),
        gradient_finite=gradient_finite_all,
        old_log_probs_frozen=True,
        old_values_frozen=True,
        advantages_frozen=True,
        returns_frozen=True,
        each_transition_once_per_epoch=True,
        epoch_summaries=tuple(epoch_summaries),
        q_loss_mean=None,
        q_gradient_norm_mean=None,
        q_gradient_finite=None,
        counterfactual_advantage_mean=cf_adv_mean,
        counterfactual_advantage_std=cf_adv_std,
        counterfactual_advantage_abs_mean=cf_adv_abs_mean,
        free_trigger_decision_count=sample_count,
        q_optimizer_steps=0,
        credit_phase="sparse_direct_cf",
        actor_update_executed=True,
        critic_replay_sample_count=0,
        cf_sample_count=sample_count,
        cf_delta_mean=float(np.mean(deltas)),
        cf_delta_std=float(np.std(deltas)),
        cf_delta_abs_mean=float(np.mean(np.abs(deltas))),
        cf_delta_positive_fraction=float(np.mean(deltas > 0.0)),
        cf_advantage_mean=cf_adv_mean,
        cf_advantage_std=cf_adv_std,
        cf_advantage_abs_mean=cf_adv_abs_mean,
        direct_cf_local_objective_change_proxy=local_proxy,
        branch_environment_steps=int(sum(item.branch_environment_steps for item in direct_cf_samples)),
        actor_gradient_finite=gradient_finite_all,
        encoder_parameter_delta_norm=encoder_delta,
        trigger_heads_parameter_delta_norm=trigger_heads_delta,
        autoregressive_parameter_delta_norm=autoregressive_delta,
        value_head_parameter_delta_norm=value_head_delta,
        logit_head_parameter_delta_norm=logit_head_delta,
        actor_batch_sample_count=sample_count,
        cf_batch_vshift_sum=float(math.fsum(float(value) for value in vshift)),
        cf_batch_vshift_mean=float(np.mean(vshift)),
        cf_batch_vshift_positive_fraction=float(np.mean(vshift > 0.0)),
        **reference_fields,
    )


def _sparse_direct_cf_teacher_forced_reporting(
    policy: Any,
    rollout_batch: ppo_rollout.RolloutBatch,
    *,
    validate_old_policy: bool = True,
) -> tuple[np.ndarray, float]:
    """用当前episode baseline做无副作用teacher-forced reporting。"""

    _require_torch()
    policy = getattr(policy, "sub_policy", policy)
    with torch.no_grad():
        evaluation = policy.evaluate_actions(
            torch.as_tensor(rollout_batch.observations, dtype=torch.float32),
            torch.as_tensor(rollout_batch.actions, dtype=torch.float32),
            action_masks=torch.as_tensor(rollout_batch.action_masks, dtype=torch.bool),
        )
    probabilities = evaluation.per_action_probabilities.detach().cpu().numpy().astype(np.float64)
    entropy = evaluation.entropy.detach().cpu().numpy().astype(np.float64)
    old_probability_table = np.asarray(
        rollout_batch.old_per_action_probabilities,
        dtype=np.float64,
    )
    if old_probability_table.ndim != 3 or old_probability_table.shape[-1] != 2:
        raise ValueError("V3 reporting需要[T, action_dim, 2] old probability table。")
    if probabilities.shape != old_probability_table.shape[:2]:
        raise ValueError("V3 reporting probability shape与rollout不一致。")
    if validate_old_policy and not np.allclose(
        probabilities,
        old_probability_table[..., 1],
        rtol=2e-5,
        atol=2e-6,
    ):
        raise AssertionError("V3 reporting的p_before与rollout frozen old policy不一致。")
    if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(entropy)):
        raise FloatingPointError("V3 reporting probability/entropy出现NaN或Inf。")
    policy_entropy = float(np.mean(entropy))
    if not math.isfinite(policy_entropy):
        raise FloatingPointError("V3 reporting policy_entropy非finite。")
    return probabilities, policy_entropy


def _attach_sparse_direct_cf_reporting(
    policy: Any,
    rollout_batch: ppo_rollout.RolloutBatch,
    update: FormalPPOUpdateSummary,
    probabilities_before: np.ndarray,
    policy_entropy: float,
) -> FormalPPOUpdateSummary:
    """在update前后计算V3 reporting；不采样、不反传、不改变optimizer/RNG。"""

    probabilities_after, _ = _sparse_direct_cf_teacher_forced_reporting(
        policy,
        rollout_batch,
        validate_old_policy=False,
    )
    absolute_change = np.abs(probabilities_after - probabilities_before).reshape(-1)
    if not np.all(np.isfinite(absolute_change)):
        raise FloatingPointError("V3 reporting probability drift出现NaN或Inf。")
    return replace(
        update,
        entropy_mean=float(policy_entropy),
        policy_entropy=float(policy_entropy),
        probability_drift_mean_abs=float(np.mean(absolute_change)),
        probability_drift_p95=float(np.percentile(absolute_change, 95)),
        probability_drift_p99=float(np.percentile(absolute_change, 99)),
        probability_drift_max=float(np.max(absolute_change)),
    )


def _attach_frozen_policy_batch_reporting(
    policy: Any,
    direct_cf_samples: list[ppo_rollout.DirectCFCreditSample],
    update: FormalPPOUpdateSummary,
    policy_entropy: float,
) -> FormalPPOUpdateSummary:
    """Record probability drift on the 32 Direct-CF batch decisions."""

    _require_torch()
    policy = getattr(policy, "sub_policy", policy)
    observations = np.stack([item.observation for item in direct_cf_samples]).astype(np.float32)
    actions = np.stack([item.action_vector for item in direct_cf_samples]).astype(np.float32)
    action_masks = np.stack([item.action_mask for item in direct_cf_samples]).astype(bool)
    target_indices = np.asarray([item.trigger_index for item in direct_cf_samples], dtype=np.int64)
    with torch.no_grad():
        evaluation = policy.evaluate_actions(
            torch.as_tensor(observations, dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.float32),
            action_masks=torch.as_tensor(action_masks, dtype=torch.bool),
        )
    rows = torch.arange(len(direct_cf_samples), device=evaluation.per_action_probabilities.device)
    p1_after = evaluation.per_action_probabilities[rows, torch.as_tensor(
        target_indices,
        dtype=torch.long,
        device=evaluation.per_action_probabilities.device,
    )].detach().cpu().numpy().astype(np.float64)
    p1_before = np.asarray([item.old_p1 for item in direct_cf_samples], dtype=np.float64)
    absolute_change = np.abs(p1_after - p1_before)
    if not np.all(np.isfinite(absolute_change)):
        raise FloatingPointError("frozen_policy_batch probability drift出现NaN或Inf。")
    return replace(
        update,
        entropy_mean=float(policy_entropy),
        policy_entropy=float(policy_entropy),
        probability_drift_mean_abs=float(np.mean(absolute_change)),
        probability_drift_p95=float(np.percentile(absolute_change, 95)),
        probability_drift_p99=float(np.percentile(absolute_change, 99)),
        probability_drift_max=float(np.max(absolute_change)),
    )


def _hierarchical_gate_collection_summary(
    policy: Any,
    rollout_batch: ppo_rollout.RolloutBatch,
    what_cf_samples: list[ppo_rollout.DirectCFCreditSample],
    gate_cf_samples: list[Any],
    project_cfg: dict[str, Any],
) -> FormalPPOUpdateSummary:
    """No-op episode summary while a hierarchical Batch4 is still collecting."""

    _, policy_entropy = _sparse_direct_cf_teacher_forced_reporting(policy, rollout_batch)
    what_deltas = np.asarray([item.delta_g for item in what_cf_samples], dtype=np.float64)
    what_advantages = np.asarray([item.a_cf for item in what_cf_samples], dtype=np.float64)
    gate_deltas = np.asarray([item.delta_gate for item in gate_cf_samples], dtype=np.float64)
    return FormalPPOUpdateSummary(
        update_epochs=int(project_config.get_config_value(project_cfg, "algorithm.ppo.update_epochs")),
        num_minibatches=1,
        minibatch_size=int(len(what_cf_samples)),
        optimizer_steps=0,
        policy_loss_mean=0.0,
        value_loss_mean=0.0,
        entropy_mean=float(policy_entropy),
        approx_kl_mean=0.0,
        clip_fraction_mean=0.0,
        gradient_norm_mean=0.0,
        gradient_finite=True,
        old_log_probs_frozen=True,
        old_values_frozen=True,
        advantages_frozen=True,
        returns_frozen=True,
        each_transition_once_per_epoch=True,
        epoch_summaries=tuple(),
        policy_entropy=float(policy_entropy),
        credit_phase=ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT,
        actor_update_executed=False,
        cf_sample_count=int(len(what_cf_samples)),
        cf_delta_mean=(None if not len(what_deltas) else float(np.mean(what_deltas))),
        cf_delta_std=(None if not len(what_deltas) else float(np.std(what_deltas))),
        cf_delta_abs_mean=(None if not len(what_deltas) else float(np.mean(np.abs(what_deltas)))),
        cf_delta_positive_fraction=(None if not len(what_deltas) else float(np.mean(what_deltas > 0.0))),
        cf_advantage_mean=(None if not len(what_advantages) else float(np.mean(what_advantages))),
        cf_advantage_std=(None if not len(what_advantages) else float(np.std(what_advantages))),
        cf_advantage_abs_mean=(None if not len(what_advantages) else float(np.mean(np.abs(what_advantages)))),
        branch_environment_steps=int(sum(item.branch_environment_steps for item in what_cf_samples)),
        actor_gradient_finite=True,
        gate_optimizer_steps=0,
        gate_loss_mean=None,
        gate_gradient_norm_mean=None,
        gate_gradient_finite=None,
        gate_probability_shift_mean=None,
        gate_probability_shift_max=None,
        gate_delta_abs_mean=(None if not len(gate_deltas) else float(np.mean(np.abs(gate_deltas)))),
        gate_delta_positive_fraction=(None if not len(gate_deltas) else float(np.mean(gate_deltas > 0.0))),
        gate_sample_count=int(len(gate_cf_samples)),
        gate_branch_environment_steps=int(
            sum(
                int(item.crn_metadata.get("config_branch_environment_steps", 0))
                + int(item.crn_metadata.get("null_branch_environment_steps", 0))
                for item in gate_cf_samples
            )
        ),
        gate_collection_policy_version=(
            None
            if not gate_cf_samples
            else int(gate_cf_samples[0].collection_policy_version)
        ),
    )


def _hierarchical_gate_formal_training(
    *,
    ctx: TrainingSanityContext,
    protocol: FormalTrainingProtocol,
    plan: FormalTrainingRunPlan,
    num_episodes: int,
    policy: Any | None,
    actor_optimizer: Any | None,
    gate_optimizer: Any | None,
    rng_generator: Any | None,
    cf_sampling_rng: np.random.Generator | None,
    gate_sampling_rng: np.random.Generator | None,
    run_dir: str | Path,
    resume: bool,
) -> FormalTrainingRunResult:
    """Execute the formal hierarchical WHEN/WHAT Batch4 path.

    This dispatcher is intentionally separate from the existing sparse
    Direct-CF loop.  The two paths share the WHAT branch and RefDrift
    implementation, but a hierarchical run owns an additional Gate optimizer,
    Gate sampling RNG, pending Gate-CF stream, and Gate causal artifact.
    """

    _require_torch()
    if not isinstance(num_episodes, int) or isinstance(num_episodes, bool) or num_episodes <= 0:
        raise ValueError("hierarchical Gate num_episodes必须是正整数。")
    if int(num_episodes) % HIERARCHICAL_GATE_CF_BATCH_EPISODES != 0:
        raise ValueError("hierarchical Gate正式run的num_episodes必须是4的整数倍。")
    if not isinstance(plan, FormalTrainingRunPlan) or plan.credit_assignment != ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT:
        raise ValueError("hierarchical Gate dispatcher需要匹配的FormalTrainingRunPlan。")

    formal_run_dir, checkpoint_path, history_path = _prepare_formal_run_dir(run_dir, resume)
    seed = int(project_config.get_config_value(ctx.cfg, "experiment.seed"))
    configured_cf_samples = _cf_samples_per_episode(ctx.cfg)
    sampling_strategy = _cf_sampling_strategy(ctx.cfg)
    direct_cf_steps = _formal_aligned_steps(ctx, protocol)

    if (policy is None) != (actor_optimizer is None):
        raise ValueError("hierarchical Gate execute时policy和Actor optimizer必须同时提供或同时省略。")
    if policy is None and gate_optimizer is not None:
        raise ValueError("hierarchical Gate没有policy时不能单独提供Gate optimizer。")
    if policy is not None and gate_optimizer is None:
        raise ValueError("hierarchical Gate必须显式提供独立Gate optimizer。")
    if rng_generator is None:
        rng_generator = torch.Generator()
        if not resume:
            rng_generator.manual_seed(seed)
    elif not hasattr(rng_generator, "get_state") or not hasattr(rng_generator, "set_state"):
        raise TypeError("rng_generator必须是支持get_state/set_state的torch.Generator。")

    if cf_sampling_rng is None:
        cf_sampling_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53444643]))
    elif not isinstance(cf_sampling_rng, np.random.Generator):
        raise TypeError("hierarchical WHAT cf_sampling_rng必须是独立np.random.Generator。")
    if gate_sampling_rng is None:
        gate_sampling_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x47415445]))
    elif not isinstance(gate_sampling_rng, np.random.Generator):
        raise TypeError("hierarchical Gate gate_sampling_rng必须是独立np.random.Generator。")
    if gate_sampling_rng is cf_sampling_rng:
        raise ValueError("hierarchical WHAT/Gate sampling RNG必须独立。")

    if policy is None:
        # The initial construction is the only place that seeds global policy
        # initialisation.  Resume first constructs the same shape, then the
        # checkpoint loader restores all states and RNGs.
        _seed_new_formal_run(seed, rng_generator)
        policy, actor_optimizer, gate_optimizer = _build_hierarchical_training_components(ctx)
    if not isinstance(policy, hierarchical_reconfiguration_gate.HierarchicalReconfigurationGatePolicy):
        raise TypeError("hierarchical Gate formal run需要HierarchicalReconfigurationGatePolicy。")
    if actor_optimizer is None or gate_optimizer is None:
        raise ValueError("hierarchical Gate formal run缺少独立WHAT/Gate optimizer。")

    actor_parameter_ids = {
        id(parameter)
        for group in actor_optimizer.param_groups
        for parameter in group["params"]
    }
    expected_actor_ids = {
        id(parameter)
        for parameter in ppo_trigger_policy.sparse_direct_cf_actor_parameters(policy.sub_policy)
    }
    gate_parameter_ids = {id(parameter) for parameter in policy.gate.parameters()}
    actual_gate_ids = {
        id(parameter)
        for group in gate_optimizer.param_groups
        for parameter in group["params"]
    }
    if actor_parameter_ids != expected_actor_ids:
        raise ValueError("hierarchical WHAT optimizer参数集合与existing Direct-CF Actor不一致。")
    if actual_gate_ids != gate_parameter_ids:
        raise ValueError("hierarchical Gate optimizer必须且只能包含Gate参数。")
    if actor_parameter_ids & actual_gate_ids:
        raise ValueError("hierarchical WHAT/Gate optimizer参数不得重叠。")

    compatibility = _formal_compatibility_metadata(ctx, protocol, policy)
    policy_version = 0
    optimizer_step_count = 0
    gate_policy_version = 0
    gate_optimizer_step_count = 0
    pending_direct_cf_samples: list[ppo_rollout.DirectCFCreditSample] = []
    pending_gate_cf_samples: list[Any] = []
    pending_reference_records: list[ppo_rollout.ReferenceStateRecord] = []
    hierarchical_state: dict[str, Any] = {}

    if resume:
        (
            history,
            policy_version,
            optimizer_step_count,
            _unused_q_optimizer_steps,
            pending_direct_cf_samples,
            pending_reference_records,
        ) = _load_formal_checkpoint(
            checkpoint_path,
            policy,
            actor_optimizer,
            rng_generator,
            compatibility,
            ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT,
            None,
            None,
            None,
            None,
            cf_sampling_rng,
            gate_optimizer,
            gate_sampling_rng,
            hierarchical_state,
        )
        gate_policy_version = int(hierarchical_state.get("gate_policy_version", policy_version))
        gate_optimizer_step_count = int(
            hierarchical_state.get("gate_optimizer_step_count", optimizer_step_count)
        )
        pending_gate_cf_samples = list(hierarchical_state.get("pending_gate_cf_samples", []))
        _synchronize_training_history_json(history_path, history)
        causal_artifacts.validate_resume_artifact_index(
            formal_run_dir,
            completed_batch_count=int(history.get("causal_update_artifact_index", optimizer_step_count)),
            completed_episodes=int(history.get("completed_episodes", 0)),
        )
    else:
        history = _build_training_history(
            [],
            0,
            0,
            ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT,
            credit_phase=ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT,
            cf_sampling_strategy=sampling_strategy,
            method_id=hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_METHOD,
            architecture_id=ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID,
            cf_batch_episodes=HIERARCHICAL_GATE_CF_BATCH_EPISODES,
            cf_samples_per_episode=configured_cf_samples,
            batch_id=0,
            batch_episode_index=0,
            pending_cf_sample_count=0,
            pending_collection_policy_version=None,
            reference_probability_drift_cap=0.005,
            pending_reference_state_count=0,
            gate_policy_version=0,
            gate_optimizer_step_count=0,
            pending_gate_cf_sample_count=0,
        )

    records = list(history["episodes"])
    completed_episodes = int(history["completed_episodes"])
    if completed_episodes > int(num_episodes):
        raise ValueError("resume checkpoint已超过本次目标num_episodes。")
    if int(history.get("cf_samples_per_episode", configured_cf_samples)) != configured_cf_samples:
        raise RuntimeError("hierarchical Gate WHAT cf_samples_per_episode与当前配置不一致。")
    if int(history.get("gate_cf_samples_per_episode", -1)) != HIERARCHICAL_GATE_CF_SAMPLES_PER_EPISODE:
        raise RuntimeError("hierarchical Gate gate_cf_samples_per_episode必须为8。")

    batch_position = completed_episodes % HIERARCHICAL_GATE_CF_BATCH_EPISODES
    expected_gate_pending = batch_position * HIERARCHICAL_GATE_CF_SAMPLES_PER_EPISODE
    if len(pending_gate_cf_samples) != expected_gate_pending:
        raise RuntimeError(
            "hierarchical Gate resume pending Gate-CF count不符合Batch4位置："
            f"{len(pending_gate_cf_samples)} != {expected_gate_pending}"
        )
    if len(pending_gate_cf_samples) or len(pending_direct_cf_samples) or len(pending_reference_records):
        versions = (
            {int(item.collection_policy_version) for item in pending_gate_cf_samples}
            | {int(item.collection_policy_version) for item in pending_direct_cf_samples}
            | {int(item.collection_policy_version) for item in pending_reference_records}
        )
        if versions != {int(policy_version)}:
            raise RuntimeError("hierarchical Gate resume pending data不得混合collection policy version。")
        pending_version = history.get("pending_collection_policy_version")
        if pending_version is None or int(pending_version) != int(policy_version):
            raise RuntimeError("hierarchical Gate resume pending collection version缺失或不一致。")
    elif history.get("pending_collection_policy_version") is not None:
        raise RuntimeError("hierarchical Gate空pending batch不得携带collection policy version。")

    causal_scenario_reset_identifier = {
        "prediction_scenarios_path": str(
            cache_configuration.resolve_path(
                Path(ctx.cfg["_meta"]["project_root"]),
                project_config.get_config_value(ctx.cfg, "paths.prediction_scenarios"),
            )
        ),
        "episode_origin_start": int(protocol.episode_origin_indices[0]),
        "episode_origin_end": int(protocol.episode_origin_indices[-1]),
        "episode_slot_start": int(protocol.episode_slots[0]),
        "episode_slot_end": int(protocol.episode_slots[-1]),
        "steps_per_episode": int(protocol.steps_per_episode),
        "reset_semantics": "deterministic formal reset from ctx.initial_cache and ctx.initial_uav",
        "initial_cache_source": str(protocol.initial_cache_source),
        "initial_uav_source": str(protocol.initial_uav_source),
        "tv_local_cache_source": str(protocol.tv_local_cache_source),
    }

    episode_summaries: list[TrainingEpisodeSummary] = []
    cf_parallel_workers = _cf_parallel_workers(ctx.cfg)
    print(
        f"[PPO] {'resume' if resume else 'training'} hierarchical Gate: episodes "
        f"{completed_episodes} -> {num_episodes}, run_dir={formal_run_dir}",
        flush=True,
    )

    def clone_parameters(module: Any) -> dict[str, Any]:
        return {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
        }

    def clone_optimizer_state(optimizer: Any) -> dict[str, Any]:
        return copy.deepcopy(optimizer.state_dict())

    def current_pending_version() -> int | None:
        if pending_gate_cf_samples or pending_direct_cf_samples or pending_reference_records:
            return int(policy_version)
        return None

    for episode_index in range(completed_episodes, int(num_episodes)):
        episode_start_time = time.perf_counter()
        collection_policy_version = int(policy_version)
        batch_id = int(episode_index // HIERARCHICAL_GATE_CF_BATCH_EPISODES + 1)
        batch_episode_index = int(episode_index % HIERARCHICAL_GATE_CF_BATCH_EPISODES + 1)
        causal_artifacts.persist_episode_start_artifact(
            run_dir=formal_run_dir,
            episode=episode_index + 1,
            policy_version=collection_policy_version,
            scenario_reset_identifier=causal_scenario_reset_identifier,
            rng_states=_capture_rng_states(policy, rng_generator),
            cf_sampling_rng_state=copy.deepcopy(cf_sampling_rng.bit_generator.state),
            environment_reset_rng={
                "gate_sampling_rng_state": copy.deepcopy(gate_sampling_rng.bit_generator.state),
            },
            reset_semantics="deterministic formal reset; no separate environment RNG is consumed",
            initial_dynamic_cache=np.asarray(ctx.initial_cache, dtype=np.int8),
            initial_uav_state=ctx.initial_uav,
            tv_local_cache_state=ctx.local_cache,
        )

        parameters_before_collection = clone_parameters(policy)
        direct_cf_reservoir = _build_direct_cf_reservoir(
            ctx.action_spec,
            configured_cf_samples,
            cf_sampling_rng,
            sampling_strategy,
        )
        gate_cf_reservoir = ppo_rollout.GateCFReservoir(
            HIERARCHICAL_GATE_CF_SAMPLES_PER_EPISODE,
            gate_sampling_rng,
        )
        batch = collect_formal_episode_rollout(
            policy,
            ctx,
            protocol,
            direct_cf_reservoir,
            episode_index=episode_index,
            collection_policy_version=collection_policy_version,
            collect_reference_states=True,
            gate_cf_reservoir=gate_cf_reservoir,
        )
        raw_stats = _formal_raw_rollout_statistics(batch)
        print(
            f"[PPO] episode {episode_index + 1}/{num_episodes} rollout done | "
            f"steps={len(batch.transitions)} | time={time.perf_counter() - episode_start_time:.1f}s",
            flush=True,
        )
        if gate_cf_reservoir.sample_count != HIERARCHICAL_GATE_CF_SAMPLES_PER_EPISODE:
            raise RuntimeError("hierarchical Gate每个formal episode必须选出8个Gate-CF slots。")
        gate_cf_start_time = time.perf_counter()
        gate_cf_samples = ppo_rollout.run_gate_cf_branches(
            policy,
            gate_cf_reservoir,
            direct_cf_steps,
            batch,
            ctx.local_cache,
            ctx.service_catalog,
            ctx.cfg,
            ctx.cache_nodes,
            ctx.cache_cfg,
            ctx.flight_boundary,
            ctx.observation_spec,
            ctx.action_spec,
            ctx.terminal_slot,
            collection_policy_version=collection_policy_version,
            episode_index=episode_index,
            cf_parallel_workers=cf_parallel_workers,
        )
        if len(gate_cf_samples) != HIERARCHICAL_GATE_CF_SAMPLES_PER_EPISODE:
            raise RuntimeError("hierarchical Gate-CF branch数量必须为8。")
        if len({str(item.decision_id) for item in gate_cf_samples}) != len(gate_cf_samples):
            raise RuntimeError("hierarchical Gate-CF decision_id必须唯一。")
        if {int(item.collection_policy_version) for item in gate_cf_samples} != {collection_policy_version}:
            raise RuntimeError("hierarchical Gate-CF collection_policy_version写入不一致。")
        print(
            f"[PPO] episode {episode_index + 1}/{num_episodes} Gate-CF done | "
            f"samples={len(gate_cf_samples)} | workers={min(cf_parallel_workers, len(gate_cf_samples))} | "
            f"time={time.perf_counter() - gate_cf_start_time:.1f}s",
            flush=True,
        )

        direct_cf_start_time = time.perf_counter()
        direct_cf_samples = ppo_rollout.run_direct_cf_branches(
            policy,
            direct_cf_reservoir,
            direct_cf_steps,
            batch,
            ctx.local_cache,
            ctx.service_catalog,
            ctx.cfg,
            ctx.cache_nodes,
            ctx.cache_cfg,
            ctx.flight_boundary,
            ctx.observation_spec,
            ctx.action_spec,
            ctx.terminal_slot,
            collection_policy_version=collection_policy_version,
            episode_index=episode_index,
            cf_parallel_workers=cf_parallel_workers,
        )
        if len(direct_cf_samples) != direct_cf_reservoir.sample_count:
            raise RuntimeError("hierarchical WHAT Direct-CF branch数量与reservoir不一致。")
        if direct_cf_samples and {int(item.collection_policy_version) for item in direct_cf_samples} != {collection_policy_version}:
            raise RuntimeError("hierarchical WHAT Direct-CF collection_policy_version写入不一致。")
        identity_errors = [
            abs(float(item.delta_g) - float(item.delta_q) + float(item.delta_cache) + float(item.delta_uav))
            for item in direct_cf_samples
        ]
        if identity_errors and max(identity_errors) > 1.0e-9:
            raise AssertionError("hierarchical WHAT Direct-CF Delta identity失败。")
        print(
            f"[PPO] episode {episode_index + 1}/{num_episodes} WHAT Direct-CF done | "
            f"samples={len(direct_cf_samples)} | workers={min(cf_parallel_workers, len(direct_cf_samples))} | "
            f"time={time.perf_counter() - direct_cf_start_time:.1f}s",
            flush=True,
        )

        if not batch.reference_decisions:
            raise RuntimeError("hierarchical Gate WHAT RefDrift rollout缺少R_all reference decisions。")
        selected_ids = [str(item.decision_id) for item in direct_cf_samples]
        try:
            _, _, partition = reference_drift.validate_reference_partition(
                batch.reference_decisions,
                selected_ids,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"hierarchical Gate reference partition无效：{exc}") from exc
        if partition["selected_count"] != len(selected_ids) or partition["intersection_count"] != 0:
            raise RuntimeError("hierarchical Gate selected/reference intersection必须为0。")
        selected_id_set = set(selected_ids)
        pending_reference_records.extend(
            replace(
                item,
                selected_for_direct_cf=str(item.decision_id) in selected_id_set,
            )
            for item in batch.reference_decisions
        )
        pending_direct_cf_samples.extend(direct_cf_samples)
        pending_gate_cf_samples.extend(gate_cf_samples)

        if any(
            not torch.equal(parameter.detach(), parameters_before_collection[name])
            for name, parameter in policy.named_parameters()
        ):
            raise AssertionError("hierarchical collection/CF阶段不得修改policy参数。")

        at_batch_boundary = batch_episode_index == HIERARCHICAL_GATE_CF_BATCH_EPISODES
        gate_parameters_pre: dict[str, Any] | None = None
        gate_optimizer_state_pre: dict[str, Any] | None = None
        what_artifact_handle: causal_artifacts.UpdateArtifactHandle | None = None
        batch_what_samples: list[ppo_rollout.DirectCFCreditSample] | None = None
        if at_batch_boundary:
            if len(pending_gate_cf_samples) != HIERARCHICAL_GATE_CF_BATCH_EPISODES * HIERARCHICAL_GATE_CF_SAMPLES_PER_EPISODE:
                raise RuntimeError("hierarchical Gate Batch4必须包含32个Gate-CF samples。")
            if not pending_direct_cf_samples:
                raise RuntimeError(
                    "hierarchical Gate WHAT coverage为0，无法执行existing WHAT Direct-CF update；"
                    "不会伪造g=0 WHAT samples。"
                )
            batch_what_samples = [
                replace(item, sampling_weight=1.0 / float(len(pending_direct_cf_samples)))
                for item in pending_direct_cf_samples
            ]
            if any(
                int(item.collection_policy_version) != collection_policy_version
                for item in (pending_gate_cf_samples + batch_what_samples + pending_reference_records)
            ):
                raise RuntimeError("hierarchical Batch4不得混合collection policy version。")
            gate_parameters_pre = clone_parameters(policy.gate)
            gate_optimizer_state_pre = clone_optimizer_state(gate_optimizer)
            pre_rng_states = _capture_rng_states(policy, rng_generator)
            boundary_state = {
                "completed_episode": int(episode_index) + 1,
                "episode_index": int(episode_index),
                "batch_id": int(batch_id),
                "batch_episode_index": int(batch_episode_index),
                "collection_policy_version": int(collection_policy_version),
                "policy_version_pre": int(policy_version),
                "gate_policy_version_pre": int(gate_policy_version),
                "optimizer_step_pre": int(optimizer_step_count),
                "gate_optimizer_step_pre": int(gate_optimizer_step_count),
                "last_origin_index": int(batch.origin_indices[-1]),
                "last_slot": int(batch.slots[-1]),
                "true_terminal": bool(batch.metadata.get("true_terminal", False)),
                "bootstrap_value": float(batch.final_bootstrap_value),
                "final_cache_state": np.asarray(batch.transitions[-1].next_cache_state, dtype=np.int8).copy(),
                "final_uav_positions": copy.deepcopy(batch.transitions[-1].next_uav_positions),
                "pending_cf_count": int(len(batch_what_samples)),
                "pending_gate_cf_count": int(len(pending_gate_cf_samples)),
                "pending_reference_count": int(len(pending_reference_records)),
            }
            what_artifact_handle = causal_artifacts.persist_pre_update_artifacts(
                run_dir=formal_run_dir,
                update_index=batch_id,
                batch_id=batch_id,
                completed_episodes=episode_index + 1,
                collection_policy_version=collection_policy_version,
                policy_version_pre=policy_version,
                optimizer_step_pre=optimizer_step_count,
                policy=policy.sub_policy,
                optimizer=actor_optimizer,
                rng_states=pre_rng_states,
                cf_sampling_rng_state=copy.deepcopy(cf_sampling_rng.bit_generator.state),
                samples=list(pending_direct_cf_samples),
                training_samples=batch_what_samples,
                reference_records=list(pending_reference_records),
                pending_reference_count=len(pending_reference_records),
                reference_drift_cap=0.005,
                algorithm_schema=hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_ALGORITHM_SCHEMA,
                architecture_id=ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID,
                checkpoint_schema=HIERARCHICAL_GATE_CHECKPOINT_VERSION,
                compatibility_snapshot=compatibility,
                boundary_state=boundary_state,
                expected_sample_count=len(batch_what_samples),
            )

        if at_batch_boundary:
            if batch_what_samples is None or gate_parameters_pre is None or gate_optimizer_state_pre is None:
                raise AssertionError("hierarchical Batch4 pre-update state缺失。")
            update = update_policy_on_rollout(
                policy,
                actor_optimizer,
                batch,
                ctx.cfg,
                rng_generator,
                direct_cf_samples=batch_what_samples,
                expected_collection_policy_version=collection_policy_version,
                reference_records=list(pending_reference_records),
                reference_probability_drift_cap=0.005,
                gate_optimizer=gate_optimizer,
                gate_cf_samples=list(pending_gate_cf_samples),
            )
            if update.optimizer_steps != int(protocol.update_epochs) or update.gate_optimizer_steps != 1:
                raise RuntimeError("hierarchical Batch4必须各执行一次WHAT/Gate optimizer step。")
            if not update.gradient_finite or not update.actor_gradient_finite or not update.gate_gradient_finite:
                raise FloatingPointError("hierarchical WHAT/Gate gradient必须finite。")
            if update.value_head_parameter_delta_norm != 0.0:
                raise AssertionError("hierarchical WHAT value_head delta必须严格为0。")
            if update.gate_subpolicy_parameter_delta_max != 0.0:
                raise AssertionError("hierarchical Gate update不得修改WHAT sub-policy。")
            if update.reference_selected_intersection_count != 0:
                raise AssertionError("hierarchical WHAT selected/reference intersection必须为0。")
            if (
                update.reference_constrained_drift_max is None
                or update.old_p_reconstruction_tolerance is None
                or update.reference_constrained_drift_max > 0.005 + update.old_p_reconstruction_tolerance
            ):
                raise AssertionError("hierarchical WHAT constrained reference drift超过固定cap。")

            gate_parameters_post = clone_parameters(policy.gate)
            gate_optimizer_state_post = clone_optimizer_state(gate_optimizer)
            gate_artifact_path = (
                formal_run_dir
                / causal_artifacts.CAUSAL_ARTIFACT_DIRNAME
                / f"update_{batch_id:04d}"
                / "hierarchical_gate"
            )
            causal_artifacts.persist_hierarchical_gate_update_artifact(
                gate_artifact_path,
                update_index=batch_id,
                collection_policy_version=collection_policy_version,
                gate_samples=list(pending_gate_cf_samples),
                gate_parameters_pre=gate_parameters_pre,
                gate_parameters_post=gate_parameters_post,
                gate_optimizer_state_pre=gate_optimizer_state_pre,
                gate_optimizer_state_post=gate_optimizer_state_post,
                compatibility_snapshot={
                    "formal_compatibility": compatibility,
                    "policy_version_pre": int(policy_version),
                    "policy_version_post": int(policy_version + 1),
                    "gate_policy_version_pre": int(gate_policy_version),
                    "gate_policy_version_post": int(gate_policy_version + 1),
                    "gate_optimizer_step_pre": int(gate_optimizer_step_count),
                    "gate_optimizer_step_post": int(gate_optimizer_step_count + 1),
                    "gate_sampling_rng_state": copy.deepcopy(gate_sampling_rng.bit_generator.state),
                    "raw_gate_sample_count": int(len(pending_gate_cf_samples)),
                },
                algorithm_schema=hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_ALGORITHM_SCHEMA,
                crn_metadata={
                    "batch_id": int(batch_id),
                    "gate_sample_count": int(len(pending_gate_cf_samples)),
                    "all_gate_samples_persisted": True,
                },
            )

            policy_version += 1
            gate_policy_version += 1
            optimizer_step_count += int(update.optimizer_steps)
            gate_optimizer_step_count += int(update.gate_optimizer_steps)
            if what_artifact_handle is None:
                raise AssertionError("hierarchical WHAT pre artifact handle缺失。")
            causal_artifacts.finalize_post_update_artifact(
                what_artifact_handle,
                policy_version_post=policy_version,
                optimizer_step_post=optimizer_step_count,
                policy=policy.sub_policy,
                optimizer=actor_optimizer,
                rng_states=_capture_rng_states(policy, rng_generator),
                cf_sampling_rng_state=copy.deepcopy(cf_sampling_rng.bit_generator.state),
                update_summary=update,
                critic_step_count=0,
            )
            _atomic_json_save(
                {
                    "episode_index": int(episode_index),
                    "collection_policy_version": int(collection_policy_version),
                    "gate_policy_version": int(gate_policy_version),
                    "actor_policy_version": int(policy_version),
                    "ppo_update_summary": update,
                },
                formal_run_dir / f"episode{int(episode_index) + 1}_update_summary.json",
            )
            pending_direct_cf_samples.clear()
            pending_gate_cf_samples.clear()
            pending_reference_records.clear()
        else:
            update = _hierarchical_gate_collection_summary(
                policy,
                batch,
                direct_cf_samples,
                gate_cf_samples,
                ctx.cfg,
            )
            if update.optimizer_steps != 0 or update.gate_optimizer_steps != 0 or update.actor_update_executed:
                raise AssertionError("hierarchical Batch4收集episode不得执行optimizer step。")

        what_deltas = np.asarray([item.delta_g for item in direct_cf_samples], dtype=np.float64)
        what_advantages = np.asarray([item.a_cf for item in direct_cf_samples], dtype=np.float64)
        gate_deltas = np.asarray([item.delta_gate for item in gate_cf_samples], dtype=np.float64)
        cf_effective_count, cf_free_counts, cf_sample_counts, cf_sampling_weights = _direct_cf_reservoir_reporting(
            direct_cf_reservoir
        )
        pending_count_after = len(pending_direct_cf_samples)
        pending_gate_count_after = len(pending_gate_cf_samples)
        episode_summary = TrainingEpisodeSummary(
            episode_index=episode_index,
            policy_version_before=collection_policy_version,
            policy_version_after=policy_version,
            rollout_reward_sum=raw_stats["raw_reward_sum"],
            rollout_reward_mean=raw_stats["raw_reward_mean"],
            batch_reward_sum_float32=raw_stats["batch_reward_sum_float32"],
            batch_vs_raw_reward_sum_error=raw_stats["batch_vs_raw_reward_sum_error"],
            actual_q_sum=raw_stats["raw_actual_q_sum"],
            cache_cost_sum=raw_stats["raw_cache_cost_sum"],
            movement_cost_sum=raw_stats["raw_movement_cost_sum"],
            policy_loss_mean=update.policy_loss_mean,
            value_loss_mean=update.value_loss_mean,
            entropy_mean=update.entropy_mean,
            approx_kl_mean=update.approx_kl_mean,
            clip_fraction_mean=update.clip_fraction_mean,
            gradient_norm_mean=update.gradient_norm_mean,
            optimizer_steps=update.optimizer_steps,
            credit_phase=update.credit_phase,
            actor_update_executed=update.actor_update_executed,
            actor_updated_this_episode=update.actor_update_executed,
            actor_optimizer_steps_this_episode=update.optimizer_steps,
            cf_sample_count=int(len(direct_cf_samples)),
            cf_delta_mean=(None if not len(what_deltas) else float(np.mean(what_deltas))),
            cf_delta_std=(None if not len(what_deltas) else float(np.std(what_deltas))),
            cf_delta_abs_mean=(None if not len(what_deltas) else float(np.mean(np.abs(what_deltas)))),
            cf_delta_positive_fraction=(None if not len(what_deltas) else float(np.mean(what_deltas > 0.0))),
            cf_advantage_mean=(None if not len(what_advantages) else float(np.mean(what_advantages))),
            cf_advantage_std=(None if not len(what_advantages) else float(np.std(what_advantages))),
            cf_advantage_abs_mean=(None if not len(what_advantages) else float(np.mean(np.abs(what_advantages)))),
            direct_cf_local_objective_change_proxy=update.direct_cf_local_objective_change_proxy,
            branch_environment_steps=int(sum(item.branch_environment_steps for item in direct_cf_samples)),
            actor_gradient_finite=update.actor_gradient_finite,
            policy_entropy=update.policy_entropy,
            probability_drift_mean_abs=update.probability_drift_mean_abs,
            probability_drift_p95=update.probability_drift_p95,
            probability_drift_p99=update.probability_drift_p99,
            probability_drift_max=update.probability_drift_max,
            encoder_parameter_delta_norm=update.encoder_parameter_delta_norm,
            trigger_heads_parameter_delta_norm=update.trigger_heads_parameter_delta_norm,
            autoregressive_parameter_delta_norm=update.autoregressive_parameter_delta_norm,
            value_head_parameter_delta_norm=update.value_head_parameter_delta_norm,
            logit_head_parameter_delta_norm=update.logit_head_parameter_delta_norm,
            trigger_action1_counts=_rollout_trigger_action1_counts(
                batch,
                tuple(ctx.action_spec.labels),
            ),
            cf_trigger_sample_counts={
                label: int(cf_sample_counts[index])
                for index, label in enumerate(ctx.action_spec.labels)
            },
            cf_sampling_strategy=sampling_strategy,
            cf_effective_sample_count=int(cf_effective_count),
            cf_trigger_free_counts=cf_free_counts,
            cf_trigger_sampling_weights=cf_sampling_weights,
            cf_batch_id=batch_id,
            cf_batch_episode_index=batch_episode_index,
            cf_batch_pending_sample_count=int(pending_count_after),
            collection_policy_version=collection_policy_version,
            actor_batch_sample_count=update.actor_batch_sample_count,
            pre_clip_gradient_norm=(update.gradient_norm_mean if update.optimizer_steps > 0 else None),
            cf_batch_vshift_sum=update.cf_batch_vshift_sum,
            cf_batch_vshift_mean=update.cf_batch_vshift_mean,
            cf_batch_vshift_positive_fraction=update.cf_batch_vshift_positive_fraction,
            reference_raw_drift_mean=update.reference_raw_drift_mean,
            reference_raw_drift_median=update.reference_raw_drift_median,
            reference_raw_drift_p95=update.reference_raw_drift_p95,
            reference_raw_drift_p99=update.reference_raw_drift_p99,
            reference_raw_drift_max=update.reference_raw_drift_max,
            reference_constrained_drift_mean=update.reference_constrained_drift_mean,
            reference_constrained_drift_median=update.reference_constrained_drift_median,
            reference_constrained_drift_p95=update.reference_constrained_drift_p95,
            reference_constrained_drift_p99=update.reference_constrained_drift_p99,
            reference_constrained_drift_max=update.reference_constrained_drift_max,
            selected_drift_mean=update.selected_drift_mean,
            selected_drift_p95=update.selected_drift_p95,
            selected_drift_max=update.selected_drift_max,
            reference_drift_alpha=update.reference_drift_alpha,
            reference_drift_constraint_active=update.reference_drift_constraint_active,
            reference_probability_drift_cap=update.reference_probability_drift_cap,
            reference_state_count=update.reference_state_count,
            reference_unselected_count=update.reference_unselected_count,
            reference_selected_count=update.reference_selected_count,
            reference_selected_intersection_count=update.reference_selected_intersection_count,
            reference_drift_by_trigger=update.reference_drift_by_trigger,
            old_p_reconstruction_max_error=update.old_p_reconstruction_max_error,
            old_p_reconstruction_checked=update.old_p_reconstruction_checked,
            old_p_reconstruction_tolerance=update.old_p_reconstruction_tolerance,
            probability_dtype=update.probability_dtype,
            reference_count_checked=update.reference_count_checked,
            direct_cf_raw_vshift_sum=update.direct_cf_raw_vshift_sum,
            direct_cf_raw_vshift_mean=update.direct_cf_raw_vshift_mean,
            direct_cf_raw_vshift_positive_fraction=update.direct_cf_raw_vshift_positive_fraction,
            direct_cf_constrained_vshift_sum=update.direct_cf_constrained_vshift_sum,
            direct_cf_constrained_vshift_mean=update.direct_cf_constrained_vshift_mean,
            direct_cf_constrained_vshift_positive_fraction=update.direct_cf_constrained_vshift_positive_fraction,
            direct_cf_vshift_retention_ratio=update.direct_cf_vshift_retention_ratio,
            gate_action1_count=int(batch.metadata.get("gate_action1_count", 0)),
            gate_action0_count=int(batch.metadata.get("gate_action0_count", 0)),
            gate_action1_ratio=batch.metadata.get("gate_action1_ratio"),
            mean_gate_probability=batch.metadata.get("mean_gate_probability"),
            gate_cf_sample_count=len(gate_cf_samples),
            gate_cf_delta_mean=(None if not len(gate_deltas) else float(np.mean(gate_deltas))),
            gate_cf_delta_abs_mean=(None if not len(gate_deltas) else float(np.mean(np.abs(gate_deltas)))),
            gate_cf_delta_positive_fraction=(None if not len(gate_deltas) else float(np.mean(gate_deltas > 0.0))),
            gate_loss_mean=update.gate_loss_mean,
            gate_gradient_norm_mean=update.gate_gradient_norm_mean,
            gate_gradient_finite=update.gate_gradient_finite,
            gate_probability_shift_mean=update.gate_probability_shift_mean,
            gate_probability_shift_max=update.gate_probability_shift_max,
            gate_optimizer_steps_this_episode=update.gate_optimizer_steps,
            gate_optimizer_step_count=gate_optimizer_step_count,
            gate_policy_version_before=collection_policy_version,
            gate_policy_version_after=gate_policy_version,
            actor_policy_version_before=collection_policy_version,
            actor_policy_version_after=policy_version,
            what_cf_coverage=float(len(direct_cf_samples) / float(configured_cf_samples)),
        )
        episode_summaries.append(episode_summary)
        records.append(_training_episode_history_record(episode_summary, update, ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT))
        history = _build_training_history(
            records,
            policy_version,
            optimizer_step_count,
            ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT,
            credit_phase=ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT,
            cf_sampling_strategy=sampling_strategy,
            method_id=hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_METHOD,
            architecture_id=ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID,
            cf_batch_episodes=HIERARCHICAL_GATE_CF_BATCH_EPISODES,
            cf_samples_per_episode=configured_cf_samples,
            batch_id=batch_id,
            batch_episode_index=batch_episode_index,
            pending_cf_sample_count=pending_count_after,
            pending_collection_policy_version=current_pending_version(),
            reference_probability_drift_cap=0.005,
            pending_reference_state_count=len(pending_reference_records),
            gate_policy_version=gate_policy_version,
            gate_optimizer_step_count=gate_optimizer_step_count,
            pending_gate_cf_sample_count=pending_gate_count_after,
        )
        payload = _checkpoint_payload(
            policy,
            actor_optimizer,
            policy_version,
            optimizer_step_count,
            history,
            compatibility,
            _capture_rng_states(policy, rng_generator),
            ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT,
            None,
            None,
            0,
            None,
            cf_sampling_rng,
            pending_direct_cf_samples=pending_direct_cf_samples,
            pending_reference_records=pending_reference_records,
            gate_optimizer=gate_optimizer,
            gate_sampling_rng=gate_sampling_rng,
            pending_gate_cf_samples=pending_gate_cf_samples,
            gate_policy_version=gate_policy_version,
            gate_optimizer_step_count=gate_optimizer_step_count,
        )
        _atomic_torch_save(payload, checkpoint_path)
        try:
            _atomic_json_save(history, history_path)
        except Exception as exc:
            warnings.warn(
                f"TRAINING_HISTORY_JSON_MIRROR_WRITE_FAILED: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        print(
            f"[PPO] hierarchical episode {episode_index + 1}/{num_episodes} | "
            f"mean_G={episode_summary.rollout_reward_mean:.4f} | "
            f"gate={episode_summary.gate_action1_count}/{len(batch.transitions)} "
            f"| gate_cf={len(gate_cf_samples)} | what_cf={len(direct_cf_samples)} "
            f"| policy={collection_policy_version}->{policy_version} "
            f"| batch={batch_id}:{batch_episode_index}/4 "
            f"| pending={pending_count_after}/{pending_gate_count_after} "
            f"| time={time.perf_counter() - episode_start_time:.1f}s",
            flush=True,
        )

    if int(history["completed_episodes"]) > 0:
        _plot_training_convergence_after_persistence(formal_run_dir, history)
    return FormalTrainingRunResult(
        plan=plan,
        episode_summaries=episode_summaries,
        final_policy_version=int(policy_version),
        optimizer_step_count=int(optimizer_step_count),
        completed_episodes=int(history["completed_episodes"]),
        training_history=history,
        resumed=bool(resume),
        q_optimizer_step_count=0,
        gate_policy_version=int(gate_policy_version),
        gate_optimizer_step_count=int(gate_optimizer_step_count),
    )


def _frozen_policy_batch_collection_summary(
    policy: Any,
    rollout_batch: ppo_rollout.RolloutBatch,
    direct_cf_samples: list[ppo_rollout.DirectCFCreditSample],
    project_cfg: dict[str, Any],
) -> FormalPPOUpdateSummary:
    """No-op per-episode summary while a frozen-policy batch is collecting."""

    _, policy_entropy = _sparse_direct_cf_teacher_forced_reporting(policy, rollout_batch)
    deltas = np.asarray([item.delta_g for item in direct_cf_samples], dtype=np.float64)
    a_cf = np.asarray([item.a_cf for item in direct_cf_samples], dtype=np.float64)
    sample_count = len(direct_cf_samples)
    return FormalPPOUpdateSummary(
        update_epochs=int(project_config.get_config_value(project_cfg, "algorithm.ppo.update_epochs")),
        num_minibatches=1,
        minibatch_size=sample_count,
        optimizer_steps=0,
        policy_loss_mean=0.0,
        value_loss_mean=0.0,
        entropy_mean=float(policy_entropy),
        approx_kl_mean=0.0,
        clip_fraction_mean=0.0,
        gradient_norm_mean=0.0,
        gradient_finite=True,
        old_log_probs_frozen=True,
        old_values_frozen=True,
        advantages_frozen=True,
        returns_frozen=True,
        each_transition_once_per_epoch=True,
        epoch_summaries=tuple(),
        policy_entropy=float(policy_entropy),
        credit_phase="sparse_direct_cf",
        actor_update_executed=False,
        cf_sample_count=sample_count,
        cf_delta_mean=float(np.mean(deltas)),
        cf_delta_std=float(np.std(deltas)),
        cf_delta_abs_mean=float(np.mean(np.abs(deltas))),
        cf_delta_positive_fraction=float(np.mean(deltas > 0.0)),
        cf_advantage_mean=float(np.mean(a_cf)),
        cf_advantage_std=float(np.std(a_cf)),
        cf_advantage_abs_mean=float(np.mean(np.abs(a_cf))),
        branch_environment_steps=int(sum(item.branch_environment_steps for item in direct_cf_samples)),
        actor_gradient_finite=True,
    )


def _update_policy_on_hierarchical_gate(
    policy: hierarchical_reconfiguration_gate.HierarchicalReconfigurationGatePolicy,
    actor_optimizer: Any,
    gate_optimizer: Any,
    rollout_batch: ppo_rollout.RolloutBatch,
    gate_cf_samples: list[Any],
    what_cf_samples: list[ppo_rollout.DirectCFCreditSample],
    project_cfg: dict[str, Any],
    expected_collection_policy_version: int,
    reference_records: list[ppo_rollout.ReferenceStateRecord],
    reference_probability_drift_cap: float,
) -> FormalPPOUpdateSummary:
    """Run one hierarchical Batch4 update in the mandated order.

    The Gate optimizer is stepped first from Joint-Null samples.  The existing
    WHAT Direct-CF/refdrift update then runs on the independent sub-policy
    optimizer.  Neither path shares optimizer parameters or a gradient graph.
    """

    _require_torch()
    if not isinstance(policy, hierarchical_reconfiguration_gate.HierarchicalReconfigurationGatePolicy):
        raise TypeError("hierarchical update需要HierarchicalReconfigurationGatePolicy。")
    if not gate_cf_samples:
        raise ValueError("hierarchical Gate update缺少Gate-CF samples。")
    if not what_cf_samples:
        raise ValueError("hierarchical WHAT update缺少g=1 Direct-CF samples。")
    if not isinstance(reference_records, list) or not reference_records:
        raise ValueError("hierarchical WHAT RefDrift update缺少reference records。")
    gate_versions = {int(item.collection_policy_version) for item in gate_cf_samples}
    if gate_versions != {int(expected_collection_policy_version)}:
        raise ValueError("hierarchical Gate-CF collection_policy_version不一致。")

    # Required update order: Gate first, WHAT second.
    gate_result = hierarchical_reconfiguration_gate.update_gate_on_counterfactual_batch(
        policy,
        gate_optimizer,
        list(gate_cf_samples),
        clip_epsilon=float(
            project_config.get_config_value(project_cfg, "algorithm.ppo.clip_epsilon")
        ),
        max_grad_norm=float(
            project_config.get_config_value(project_cfg, "algorithm.ppo.max_grad_norm")
        ),
        update_epochs=1,
    )
    if int(gate_result.optimizer_steps) != 1:
        raise AssertionError("hierarchical Gate 每个Batch4必须恰好执行一次optimizer step。")
    if not bool(gate_result.gradient_finite):
        raise FloatingPointError("hierarchical Gate gradient必须finite。")
    if float(gate_result.subpolicy_parameter_delta_max) != 0.0:
        raise AssertionError("Gate optimizer不得修改WHAT sub-policy参数。")

    what_update = _update_policy_on_sparse_direct_cf(
        policy,
        actor_optimizer,
        what_cf_samples,
        project_cfg,
        expected_collection_policy_version,
        reference_records,
        reference_probability_drift_cap,
    )
    if int(what_update.q_optimizer_steps) != 0:
        raise AssertionError("hierarchical WHAT update不得包含learned critic optimizer。")
    if what_update.value_head_parameter_delta_norm != 0.0:
        raise AssertionError("hierarchical WHAT value_head delta必须严格为0。")
    deltas = np.asarray([float(item.delta_gate) for item in gate_cf_samples], dtype=np.float64)
    probability_shift = np.asarray(gate_result.p_after, dtype=np.float64) - np.asarray(
        gate_result.p_before,
        dtype=np.float64,
    )
    if probability_shift.shape != deltas.shape:
        raise AssertionError("hierarchical Gate probability shift shape不一致。")
    gate_fields = {
        "credit_phase": ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT,
        "gate_optimizer_steps": int(gate_result.optimizer_steps),
        "gate_loss_mean": float(gate_result.policy_loss),
        "gate_gradient_norm_mean": float(gate_result.gradient_norm),
        "gate_gradient_finite": bool(gate_result.gradient_finite),
        "gate_probability_shift_mean": float(np.mean(probability_shift)),
        "gate_probability_shift_max": float(np.max(np.abs(probability_shift))),
        "gate_delta_abs_mean": float(np.mean(np.abs(deltas))),
        "gate_delta_positive_fraction": float(np.mean(deltas > 0.0)),
        "gate_sample_count": int(len(gate_cf_samples)),
        "gate_branch_environment_steps": int(
            sum(int(item.crn_metadata.get("config_branch_environment_steps", 0)) for item in gate_cf_samples)
            + sum(int(item.crn_metadata.get("null_branch_environment_steps", 0)) for item in gate_cf_samples)
        ),
        "gate_parameter_delta_max": float(gate_result.gate_parameter_delta_max),
        "gate_subpolicy_parameter_delta_max": float(gate_result.subpolicy_parameter_delta_max),
        "gate_collection_policy_version": int(expected_collection_policy_version),
    }
    return replace(what_update, **gate_fields)


def update_policy_on_rollout(
    policy: Any,
    optimizer: Any,
    rollout_batch: ppo_rollout.RolloutBatch,
    project_cfg: dict[str, Any],
    rng_generator: Any,
    q_critic: Any | None = None,
    q_optimizer: Any | None = None,
    critic_replay: ppo_rollout.CriticReplayBuffer | None = None,
    critic_rng_generator: Any | None = None,
    credit_phase: str | None = None,
    critic_return_scale: float | None = None,
    direct_cf_samples: list[ppo_rollout.DirectCFCreditSample] | None = None,
    expected_collection_policy_version: int | None = None,
    reference_records: list[ppo_rollout.ReferenceStateRecord] | None = None,
    reference_probability_drift_cap: float | None = None,
    gate_optimizer: Any | None = None,
    gate_cf_samples: list[Any] | None = None,
) -> FormalPPOUpdateSummary:
    """对一个完整on-policy rollout执行正式PPO更新。

    正式语义：
    - old_log_probs、old_values、advantages、returns、actions 在4个epochs期间冻结；
    - 每个epoch对全部transition无放回shuffle一次；
    - 每个epoch切成 num_minibatches 个等长minibatch；
    - 每个minibatch执行 zero_grad -> compute_ppo_loss -> backward -> finite gradient check
      -> clip_grad_norm_ -> optimizer.step；
    - Adam/optimizer生命周期由caller管理，本函数绝不创建optimizer；
    - 不做advantage normalization、reward normalization、value clipping或KL early stopping。

    注意：这是正式执行器。本阶段CLI dry-run不会调用该函数。
    """

    _require_torch()
    credit_assignment = ppo_update.credit_assignment_from_project(project_cfg)
    if credit_assignment == ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT:
        if q_critic is not None or q_optimizer is not None or critic_replay is not None or critic_rng_generator is not None:
            raise ValueError("hierarchical Gate 不接受任何learned critic/replay。")
        if not isinstance(
            policy,
            hierarchical_reconfiguration_gate.HierarchicalReconfigurationGatePolicy,
        ):
            raise TypeError("hierarchical Gate update需要分层policy。")
        if gate_optimizer is None or gate_cf_samples is None or direct_cf_samples is None:
            raise ValueError("hierarchical Gate update需要独立Gate optimizer、Gate-CF和WHAT-CF samples。")
        if expected_collection_policy_version is None:
            raise ValueError("hierarchical Gate update需要collection_policy_version。")
        if reference_records is None or reference_probability_drift_cap is None:
            raise ValueError("hierarchical Gate WHAT RefDrift需要reference records和cap。")
        if float(reference_probability_drift_cap) != 0.005:
            raise ValueError("hierarchical Gate WHAT RefDrift cap固定为0.005。")
        return _update_policy_on_hierarchical_gate(
            policy,
            optimizer,
            gate_optimizer,
            rollout_batch,
            list(gate_cf_samples),
            list(direct_cf_samples),
            project_cfg,
            int(expected_collection_policy_version),
            list(reference_records),
            float(reference_probability_drift_cap),
        )
    if credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT:
        if q_critic is not None or q_optimizer is not None or critic_replay is not None or critic_rng_generator is not None:
            raise ValueError("sparse_direct_cf 不接受任何learned critic/replay。")
        if direct_cf_samples is None:
            raise ValueError("sparse_direct_cf update 缺少paired Direct-CF samples。")
        probabilities_before, policy_entropy = _sparse_direct_cf_teacher_forced_reporting(
            policy,
            rollout_batch,
        )
        update = _update_policy_on_sparse_direct_cf(
            policy,
            optimizer,
            direct_cf_samples,
            project_cfg,
            expected_collection_policy_version,
            reference_records,
            reference_probability_drift_cap,
        )
        if _is_frozen_batch_family(project_cfg):
            return _attach_frozen_policy_batch_reporting(
                policy,
                direct_cf_samples,
                update,
                policy_entropy,
            )
        return _attach_sparse_direct_cf_reporting(
            policy,
            rollout_batch,
            update,
            probabilities_before,
            policy_entropy,
        )
    if credit_assignment == ppo_update.PER_TRIGGER_Q_CREDIT_ASSIGNMENT:
        if q_critic is None or q_optimizer is None:
            raise ValueError("per_trigger_q update 必须显式提供独立 q_critic/q_optimizer。")
        return _update_policy_on_rollout_per_trigger_q(
            policy,
            optimizer,
            q_critic,
            q_optimizer,
            rollout_batch,
            project_cfg,
            rng_generator,
        )
    if credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT:
        if q_critic is None or q_optimizer is None:
            raise ValueError(
                "centralized_counterfactual_q update 必须显式提供独立 q_critic/q_optimizer。"
            )
        if critic_replay is None or critic_rng_generator is None:
            raise ValueError(
                "centralized_counterfactual_q update 必须显式提供 critic replay 和独立 RNG。"
            )
        if critic_return_scale is None:
            raise ValueError(
                "centralized_counterfactual_q update 必须显式提供formal critic_return_scale。"
            )
        phase = ppo_update.credit_phase_from_project(project_cfg) if credit_phase is None else str(credit_phase)
        return _update_policy_on_rollout_centralized(
            policy,
            optimizer,
            q_critic,
            q_optimizer,
            rollout_batch,
            project_cfg,
            rng_generator,
            critic_replay,
            critic_rng_generator,
            float(critic_return_scale),
            phase,
        )
    if q_critic is not None or q_optimizer is not None:
        raise ValueError("legacy_scalar update 不接受 q_critic/q_optimizer。")
    training_cfg = ppo_update.ppo_training_config_status(project_cfg)
    if not training_cfg.can_run_optimizer_step:
        raise RuntimeError(f"PPO update超参数缺失：{training_cfg.missing_required_for_step}")
    if training_cfg.max_grad_norm is None:
        raise RuntimeError("algorithm.ppo.max_grad_norm 缺失，不能执行正式update。")

    update_epochs = int(project_config.get_config_value(project_cfg, "algorithm.ppo.update_epochs"))
    num_minibatches = int(project_config.get_config_value(project_cfg, "algorithm.ppo.num_minibatches"))
    batch_size = int(rollout_batch.observations.shape[0])
    if batch_size <= 0 or batch_size % num_minibatches != 0:
        raise ValueError("rollout batch size 必须为正且可被 num_minibatches 整除。")
    minibatch_size = batch_size // num_minibatches

    frozen_old_log_probs = rollout_batch.old_log_probs.copy()
    frozen_old_values = rollout_batch.old_values.copy()
    frozen_advantages = rollout_batch.advantages.copy()
    frozen_returns = rollout_batch.returns.copy()
    frozen_actions = rollout_batch.actions.copy()

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approx_kls: list[float] = []
    clip_fractions: list[float] = []
    gradient_norms: list[float] = []
    epoch_summaries: list[PPOEpochSummary] = []
    gradient_finite_all = True
    each_transition_once_per_epoch = True
    optimizer_steps = 0

    all_indices = np.arange(batch_size, dtype=np.int64)
    for _epoch in range(update_epochs):
        shuffled = _torch_randperm_numpy(batch_size, rng_generator)
        each_transition_once_per_epoch = each_transition_once_per_epoch and bool(
            np.array_equal(np.sort(shuffled), all_indices)
        )
        epoch_minibatch_diagnostics: list[dict[str, float | int]] = []
        epoch_pre_clipping_grad_norms: list[float] = []
        for start in range(0, batch_size, minibatch_size):
            minibatch_indices = shuffled[start : start + minibatch_size]
            if len(minibatch_indices) != minibatch_size:
                raise RuntimeError("minibatch划分不完整。")
            minibatch = _slice_rollout_batch(rollout_batch, minibatch_indices)
            optimizer.zero_grad(set_to_none=True)
            loss = ppo_update.compute_ppo_loss(
                policy,
                minibatch,
                float(training_cfg.clip_epsilon),
                float(training_cfg.value_loss_coefficient),
                float(training_cfg.entropy_coefficient),
            )
            if not torch.isfinite(loss.total_loss):
                raise FloatingPointError("PPO total loss 非finite。")
            loss.total_loss.backward()
            grads = [param.grad.detach() for param in policy.parameters() if param.grad is not None]
            gradient_finite = bool(grads) and bool(all(torch.isfinite(grad).all().item() for grad in grads))
            gradient_finite_all = gradient_finite_all and gradient_finite
            if not gradient_finite:
                raise FloatingPointError("gradient 出现 NaN/Inf。")
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float(training_cfg.max_grad_norm))
            pre_clipping_grad_norm = float(grad_norm.detach().cpu())
            epoch_pre_clipping_grad_norms.append(pre_clipping_grad_norm)
            gradient_norms.append(pre_clipping_grad_norm)
            minibatch_diagnostics = {
                "transition_count": int(len(minibatch_indices)),
                "policy_loss": float(loss.policy_loss.detach().cpu()),
                "value_loss": float(loss.value_loss.detach().cpu()),
                "entropy": float(loss.entropy.detach().cpu()),
                "total_loss": float(loss.total_loss.detach().cpu()),
                "approx_kl": float(loss.approx_kl),
                "clip_fraction": float(loss.clip_fraction),
                "ratio_mean": float(loss.ratio_mean),
                "ratio_min": float(loss.ratio_min),
                "ratio_max": float(loss.ratio_max),
            }
            epoch_minibatch_diagnostics.append(minibatch_diagnostics)
            optimizer.step()
            optimizer_steps += 1
            policy_losses.append(minibatch_diagnostics["policy_loss"])
            value_losses.append(minibatch_diagnostics["value_loss"])
            entropies.append(minibatch_diagnostics["entropy"])
            approx_kls.append(minibatch_diagnostics["approx_kl"])
            clip_fractions.append(minibatch_diagnostics["clip_fraction"])
        epoch_summaries.append(
            _aggregate_epoch_diagnostics(
                epoch_index=_epoch + 1,
                minibatch_diagnostics=epoch_minibatch_diagnostics,
                pre_clipping_grad_norms=epoch_pre_clipping_grad_norms,
                optimizer_steps_cumulative=optimizer_steps,
            )
        )

    old_log_probs_frozen = bool(np.array_equal(rollout_batch.old_log_probs, frozen_old_log_probs))
    old_values_frozen = bool(np.array_equal(rollout_batch.old_values, frozen_old_values))
    advantages_frozen = bool(np.array_equal(rollout_batch.advantages, frozen_advantages))
    returns_frozen = bool(np.array_equal(rollout_batch.returns, frozen_returns))
    actions_frozen = bool(np.array_equal(rollout_batch.actions, frozen_actions))
    if not actions_frozen:
        raise AssertionError("rollout actions 在PPO update期间被修改。")

    return FormalPPOUpdateSummary(
        update_epochs=update_epochs,
        num_minibatches=num_minibatches,
        minibatch_size=minibatch_size,
        optimizer_steps=optimizer_steps,
        policy_loss_mean=float(np.mean(policy_losses)),
        value_loss_mean=float(np.mean(value_losses)),
        entropy_mean=float(np.mean(entropies)),
        approx_kl_mean=float(np.mean(approx_kls)),
        clip_fraction_mean=float(np.mean(clip_fractions)),
        gradient_norm_mean=float(np.mean(gradient_norms)),
        gradient_finite=gradient_finite_all,
        old_log_probs_frozen=old_log_probs_frozen,
        old_values_frozen=old_values_frozen,
        advantages_frozen=advantages_frozen,
        returns_frozen=returns_frozen,
        each_transition_once_per_epoch=each_transition_once_per_epoch,
        epoch_summaries=tuple(epoch_summaries),
    )


def _build_formal_training_plan(
    protocol: FormalTrainingProtocol,
    num_episodes: int,
    execute: bool,
    credit_assignment: str = ppo_update.LEGACY_CREDIT_ASSIGNMENT,
    credit_phase: str | None = None,
    cf_samples_per_episode: int = 0,
    cf_sampling_strategy: str = "uniform",
    cf_batch_episodes: int = 1,
) -> FormalTrainingRunPlan:
    if num_episodes <= 0:
        raise ValueError("num_episodes 必须由caller显式传入正整数。")
    if credit_assignment not in {
        ppo_update.LEGACY_CREDIT_ASSIGNMENT,
        ppo_update.PER_TRIGGER_Q_CREDIT_ASSIGNMENT,
        ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT,
        ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT,
        ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT,
    }:
        raise ValueError(f"未知 credit_assignment：{credit_assignment}")
    phase = "legacy"
    if credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT:
        phase = "critic_warmup" if credit_phase is None else str(credit_phase)
        if phase not in {"critic_warmup", "joint_training"}:
            raise ValueError(f"未知 credit_phase：{phase}")
    elif credit_assignment == ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT:
        phase = ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT
    cf_samples = 0
    actor_steps = int(num_episodes * protocol.expected_optimizer_steps_per_rollout)
    q_steps = 0
    gate_steps = 0
    gate_cf_samples = 0
    if credit_assignment == ppo_update.PER_TRIGGER_Q_CREDIT_ASSIGNMENT:
        q_steps = actor_steps
    elif credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT:
        q_steps = actor_steps
        if phase == "critic_warmup":
            actor_steps = 0
    elif credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT:
        # V3 每个 epoch 使用一个 K-sample full batch，不沿用 legacy 的
        # 4-minibatch rollout update 计数。
        actor_steps = int(num_episodes * protocol.update_epochs)
        cf_samples = int(cf_samples_per_episode)
        if cf_samples <= 0:
            raise ValueError("sparse_direct_cf 必须提供正的 cf_samples_per_episode。")
        if cf_sampling_strategy not in {"uniform", "trigger_stratified"}:
            raise ValueError("未知sparse_direct_cf cf_sampling_strategy。")
        if int(cf_batch_episodes) <= 0:
            raise ValueError("cf_batch_episodes 必须是正整数。")
        if cf_batch_episodes > 1:
            if int(num_episodes) % int(cf_batch_episodes) != 0:
                raise ValueError(
                    "frozen-policy batch正式run的num_episodes必须是cf_batch_episodes的整数倍。"
                )
            actor_steps = int(
                (num_episodes // int(cf_batch_episodes)) * protocol.update_epochs
            )
    elif credit_assignment == ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT:
        actor_steps = int(num_episodes * protocol.update_epochs)
        cf_samples = int(cf_samples_per_episode)
        if cf_samples <= 0:
            raise ValueError("hierarchical Gate 必须提供正的 WHAT cf_samples_per_episode。")
        if int(cf_batch_episodes) != HIERARCHICAL_GATE_CF_BATCH_EPISODES:
            raise ValueError("hierarchical Gate 正式run固定要求cf_batch_episodes=4。")
        if int(num_episodes) % HIERARCHICAL_GATE_CF_BATCH_EPISODES != 0:
            raise ValueError("hierarchical Gate正式run的num_episodes必须是4的整数倍。")
        actor_steps = int(
            (num_episodes // HIERARCHICAL_GATE_CF_BATCH_EPISODES) * protocol.update_epochs
        )
        gate_steps = actor_steps
        gate_cf_samples = HIERARCHICAL_GATE_CF_SAMPLES_PER_EPISODE
    return FormalTrainingRunPlan(
        num_episodes=int(num_episodes),
        steps_per_episode=int(protocol.steps_per_episode),
        planned_environment_steps=int(num_episodes * protocol.steps_per_episode),
        update_epochs=int(protocol.update_epochs),
        num_minibatches=int(protocol.num_minibatches),
        minibatch_size=int(protocol.minibatch_size),
        expected_optimizer_steps_per_episode=(
            int(protocol.update_epochs)
            if credit_assignment in {
                ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT,
                ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT,
            }
            else int(protocol.expected_optimizer_steps_per_rollout)
        ),
        planned_optimizer_steps=actor_steps,
        planned_backward_calls=(
            int(actor_steps + gate_steps)
            if credit_assignment == ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT
            else actor_steps
        ),
        optimizer_instances_planned=(
            2
            if credit_assignment == ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT
            else (2 if q_steps > 0 else 1)
        ),
        execute=bool(execute),
        single_trace_training=bool(protocol.single_trace_training),
        no_default_long_training_budget=True,
        train_origin_start=int(protocol.origin_split.train_origin_start),
        train_origin_end=int(protocol.origin_split.train_origin_end),
        test_origin_start=int(protocol.origin_split.test_origin_start),
        test_origin_end=int(protocol.origin_split.test_origin_end),
        credit_assignment=credit_assignment,
        planned_q_optimizer_steps=q_steps,
        planned_q_backward_calls=q_steps,
        q_optimizer_instances_planned=(1 if q_steps > 0 else 0),
        credit_phase=phase,
        planned_actor_optimizer_steps=actor_steps,
        planned_critic_optimizer_steps=q_steps,
        actor_optimizer_instances_planned=1,
        critic_optimizer_instances_planned=(1 if q_steps > 0 else 0),
        cf_samples_per_episode=cf_samples,
        planned_counterfactual_branch_steps=(
            int(num_episodes * cf_samples * protocol.steps_per_episode)
            if credit_assignment in {
                ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT,
                ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT,
            }
            else 0
        ),
        cf_sampling_strategy=str(cf_sampling_strategy),
        cf_batch_episodes=int(cf_batch_episodes),
        gate_cf_samples_per_episode=int(gate_cf_samples),
        planned_gate_counterfactual_branch_steps=int(
            num_episodes * gate_cf_samples * protocol.steps_per_episode
            if credit_assignment == ppo_update.HIERARCHICAL_GATE_CREDIT_ASSIGNMENT
            else 0
        ),
        planned_gate_optimizer_steps=int(gate_steps),
        planned_gate_backward_calls=int(gate_steps),
    )


def run_formal_ppo_training(
    *,
    num_episodes: int,
    config_path: str = "system_config.toml",
    policy: Any | None = None,
    optimizer: Any | None = None,
    q_critic: Any | None = None,
    q_optimizer: Any | None = None,
    gate_optimizer: Any | None = None,
    rng_generator: Any | None = None,
    critic_rng_generator: Any | None = None,
    critic_replay: ppo_rollout.CriticReplayBuffer | None = None,
    cf_sampling_rng: np.random.Generator | None = None,
    gate_sampling_rng: np.random.Generator | None = None,
    execute: bool = False,
    run_dir: str | Path | None = None,
    resume: bool = False,
) -> FormalTrainingRunPlan | FormalTrainingRunResult:
    """正式PPO训练run入口。

    默认 execute=False：只构造并返回计划，不执行environment、不backward、不optimizer.step。
    execute=True时，run_dir必须由caller显式提供；每个完整episode完成后才保存checkpoint。
    resume=True时，num_episodes表示该run希望达到的总完成episode数，而不是额外episode数。
    若未由caller传入policy/optimizer，本函数按当前config构造它们；per_trigger_q
    模式同时构造独立 Q critic/optimizer。resume加载顺序仍为：构造全部网络/optimizer
    -> load state_dict -> 恢复RNG -> 才开始下一episode。
    """

    if execute and run_dir is None:
        raise ValueError("execute=True 时必须显式提供 run_dir；不会默认写入data/generated或当前目录。")
    if resume and not execute:
        raise ValueError("resume=True 只允许用于 execute=True 的正式训练run。")

    ctx = _build_context(config_path)
    protocol = build_formal_training_protocol(ctx.scenarios, ctx.cfg)
    validate_formal_training_protocol(protocol)
    credit_assignment = ppo_update.credit_assignment_from_project(ctx.cfg)
    credit_phase = (
        ppo_update.credit_phase_from_project(ctx.cfg)
        if credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT
        else None
    )
    hierarchical_gate_mode = _is_hierarchical_gate_mode(ctx.cfg)
    sampling_strategy = (
        _cf_sampling_strategy(ctx.cfg)
        if _is_sparse_direct_cf_mode(ctx.cfg) or hierarchical_gate_mode
        else "uniform"
    )
    configured_cf_samples = (
        _cf_samples_per_episode(ctx.cfg)
        if _is_sparse_direct_cf_mode(ctx.cfg) or hierarchical_gate_mode
        else 0
    )
    frozen_policy_batch_mode = _is_frozen_policy_batch_mode(ctx.cfg)
    reference_drift_mode = _is_reference_drift_mode(ctx.cfg)
    frozen_batch_family_mode = _is_frozen_batch_family(ctx.cfg)
    configured_cf_batch_episodes = (
        _cf_batch_episodes(ctx.cfg)
        if frozen_batch_family_mode or hierarchical_gate_mode
        else 1
    )
    if (_is_sparse_direct_cf_mode(ctx.cfg) or hierarchical_gate_mode) and sampling_strategy == "trigger_stratified":
        if configured_cf_samples != int(ctx.action_spec.action_dim):
            raise ValueError(
                "trigger_stratified 当前要求cf_samples_per_episode等于 dynamic Trigger 数量。"
            )
    plan = _build_formal_training_plan(
        protocol,
        num_episodes,
        execute,
        credit_assignment,
        credit_phase,
        configured_cf_samples,
        sampling_strategy,
        configured_cf_batch_episodes,
    )
    if not execute:
        return plan
    _require_torch()

    if hierarchical_gate_mode:
        if (
            q_critic is not None
            or q_optimizer is not None
            or critic_rng_generator is not None
            or critic_replay is not None
        ):
            raise ValueError("hierarchical Gate formal run不接受learned critic/replay/RNG。")
        return _hierarchical_gate_formal_training(
            ctx=ctx,
            protocol=protocol,
            plan=plan,
            num_episodes=int(num_episodes),
            policy=policy,
            actor_optimizer=optimizer,
            gate_optimizer=gate_optimizer,
            rng_generator=rng_generator,
            cf_sampling_rng=cf_sampling_rng,
            gate_sampling_rng=gate_sampling_rng,
            run_dir=run_dir,
            resume=resume,
        )

    _formal_run_dir, checkpoint_path, history_path = _prepare_formal_run_dir(run_dir, resume)
    seed = int(project_config.get_config_value(ctx.cfg, "experiment.seed"))

    if (policy is None) != (optimizer is None):
        raise ValueError("execute=True 时 policy 和 optimizer 必须同时提供或同时省略。")
    if (q_critic is None) != (q_optimizer is None):
        raise ValueError("q_critic 和 q_optimizer 必须同时提供或同时省略。")

    if rng_generator is None:
        rng_generator = torch.Generator()
        if not resume:
            rng_generator.manual_seed(seed)
    elif not hasattr(rng_generator, "get_state") or not hasattr(rng_generator, "set_state"):
        raise TypeError("rng_generator 必须是支持get_state/set_state的torch.Generator。")

    if credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT:
        if q_critic is not None or q_optimizer is not None or critic_rng_generator is not None or critic_replay is not None:
            raise ValueError("sparse_direct_cf 不接受 learned critic、critic replay 或 critic RNG。")
        if cf_sampling_rng is None:
            cf_sampling_rng = np.random.default_rng(
                np.random.SeedSequence([seed, 0x53444643])
            )
        elif not isinstance(cf_sampling_rng, np.random.Generator):
            raise TypeError("cf_sampling_rng 必须是独立 np.random.Generator。")
    elif cf_sampling_rng is not None:
        raise ValueError("legacy/V1/V2 模式不接受 cf_sampling_rng。")

    if credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT:
        if critic_rng_generator is None:
            critic_rng_generator = torch.Generator()
            if not resume:
                critic_rng_generator.manual_seed(seed + 1)
        elif not hasattr(critic_rng_generator, "get_state") or not hasattr(critic_rng_generator, "set_state"):
            raise TypeError("critic_rng_generator 必须是支持get_state/set_state的torch.Generator。")
        if critic_replay is None:
            critic_replay = ppo_rollout.CriticReplayBuffer()
    elif critic_rng_generator is not None or critic_replay is not None:
        raise ValueError("legacy/per_trigger_q 模式不接受 centralized critic replay/RNG。")

    if policy is None and optimizer is None:
        # 初始化可能消耗RNG；resume时此后会先load state_dict，最后再恢复checkpoint RNG。
        _seed_new_formal_run(seed, rng_generator)
        policy, optimizer, q_critic, q_optimizer = _build_formal_training_components(ctx)
    elif credit_assignment in {
        ppo_update.PER_TRIGGER_Q_CREDIT_ASSIGNMENT,
        ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT,
    }:
        if q_critic is None or q_optimizer is None:
            raise ValueError("counterfactual Q execute 必须提供独立 critic/optimizer。")
    elif q_critic is not None or q_optimizer is not None:
        raise ValueError("legacy_scalar execute 不接受 q_critic/q_optimizer。")

    compatibility = _formal_compatibility_metadata(ctx, protocol, policy)
    resumed = bool(resume)
    pending_direct_cf_samples: list[ppo_rollout.DirectCFCreditSample] = []
    pending_reference_records: list[ppo_rollout.ReferenceStateRecord] = []
    causal_update_handle: causal_artifacts.UpdateArtifactHandle | None = None
    if resume:
        (
            history,
            policy_version,
            optimizer_step_count,
            q_optimizer_step_count,
            pending_direct_cf_samples,
            pending_reference_records,
        ) = _load_formal_checkpoint(
            checkpoint_path,
            policy,
            optimizer,
            rng_generator,
            compatibility,
            credit_assignment,
            q_critic,
            q_optimizer,
            critic_replay,
            critic_rng_generator,
            cf_sampling_rng,
        )
        # JSON只是人类可读mirror；无论它落后、缺失还是声称更新，都以checkpoint为准。
        _synchronize_training_history_json(history_path, history)
        if reference_drift_mode:
            # RefDrift resume now requires every completed batch and episode
            # start snapshot to be present and complete; old v8 runs fail the
            # checkpoint schema gate before reaching this point.
            causal_artifacts.validate_resume_artifact_index(
                _formal_run_dir,
                completed_batch_count=int(
                    history.get(
                        "causal_update_artifact_index",
                        history.get("optimizer_step_count", 0),
                    )
                ),
                completed_episodes=int(history.get("completed_episodes", 0)),
            )
    else:
        history = _empty_training_history(
            credit_assignment=credit_assignment,
            credit_phase=credit_phase,
            critic_return_scale=(
                protocol.critic_return_scale
                if credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT
                else None
            ),
            cf_sampling_strategy=sampling_strategy,
            method_id=(
                ppo_update.REFERENCE_DRIFT_FROZEN_POLICY_BATCH_METHOD
                if reference_drift_mode
                else (ppo_update.FROZEN_POLICY_BATCH_METHOD if frozen_policy_batch_mode else None)
            ),
            architecture_id=(
                ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID
                if frozen_batch_family_mode
                else None
            ),
            cf_batch_episodes=configured_cf_batch_episodes,
        )
        if frozen_batch_family_mode:
            history["cf_samples_per_episode"] = int(configured_cf_samples)
        policy_version = 0
        optimizer_step_count = 0
        q_optimizer_step_count = 0

    records = list(history["episodes"])
    completed_episodes = int(history["completed_episodes"])
    episode_summaries: list[TrainingEpisodeSummary] = []
    print(
        f"[PPO] {'resume' if resume else 'training'}: episodes "
        f"{completed_episodes} -> {num_episodes}, run_dir={_formal_run_dir}",
        flush=True,
    )
    direct_cf_steps = _formal_aligned_steps(ctx, protocol) if _is_sparse_direct_cf_mode(ctx.cfg) else None
    if frozen_policy_batch_mode:
        if completed_episodes % configured_cf_batch_episodes != 0:
            raise RuntimeError(
                "frozen_policy_batch resume必须位于完整batch boundary。"
            )
        if int(history.get("pending_cf_sample_count", 0)) != 0:
            raise RuntimeError("frozen_policy_batch boundary history不得包含pending samples。")
    elif reference_drift_mode:
        expected_pending_count = (
            int(completed_episodes % configured_cf_batch_episodes) * int(configured_cf_samples)
        )
        if len(pending_direct_cf_samples) != expected_pending_count:
            raise RuntimeError(
                "refdrift resume pending CF count不符合当前batch位置："
                f"{len(pending_direct_cf_samples)} != {expected_pending_count}"
            )
        if len(pending_reference_records) != int(history.get("pending_reference_state_count", 0)):
            raise RuntimeError("refdrift resume pending reference count与history不一致。")
        if len(pending_direct_cf_samples) == 0:
            if pending_reference_records or history.get("pending_collection_policy_version") is not None:
                raise RuntimeError("refdrift空pending batch不得携带reference或collection version。")
        else:
            pending_version = history.get("pending_collection_policy_version")
            if pending_version is None or int(pending_version) != int(policy_version):
                raise RuntimeError("refdrift pending collection policy version与当前policy不一致。")
            sample_versions = {int(item.collection_policy_version) for item in pending_direct_cf_samples}
            reference_versions = {int(item.collection_policy_version) for item in pending_reference_records}
            if sample_versions != {int(policy_version)} or reference_versions != {int(policy_version)}:
                raise RuntimeError("refdrift pending data不得混合collection policy version。")
    causal_scenario_reset_identifier = None
    if reference_drift_mode:
        root = Path(ctx.cfg["_meta"]["project_root"])
        causal_scenario_reset_identifier = {
            "prediction_scenarios_path": str(
                cache_configuration.resolve_path(
                    root,
                    project_config.get_config_value(ctx.cfg, "paths.prediction_scenarios"),
                )
            ),
            "episode_origin_start": int(protocol.episode_origin_indices[0]),
            "episode_origin_end": int(protocol.episode_origin_indices[-1]),
            "episode_slot_start": int(protocol.episode_slots[0]),
            "episode_slot_end": int(protocol.episode_slots[-1]),
            "steps_per_episode": int(protocol.steps_per_episode),
            "reset_semantics": "deterministic formal reset from ctx.initial_cache and ctx.initial_uav",
            "initial_cache_source": str(protocol.initial_cache_source),
            "initial_uav_source": str(protocol.initial_uav_source),
            "tv_local_cache_source": str(protocol.tv_local_cache_source),
        }
    for episode_index in range(completed_episodes, num_episodes):
        # A handle belongs to only this episode's batch boundary.  It must not
        # leak into the next collection episode after post artifact finalization.
        causal_update_handle = None
        episode_start_time = time.perf_counter()
        collection_policy_version = int(policy_version)
        if reference_drift_mode:
            if causal_scenario_reset_identifier is None or cf_sampling_rng is None:
                raise AssertionError("refdrift causal episode-start metadata未准备。")
            causal_artifacts.persist_episode_start_artifact(
                run_dir=_formal_run_dir,
                episode=int(episode_index) + 1,
                policy_version=collection_policy_version,
                scenario_reset_identifier=causal_scenario_reset_identifier,
                rng_states=_capture_rng_states(
                    policy,
                    rng_generator,
                    critic_rng_generator,
                ),
                cf_sampling_rng_state=copy.deepcopy(cf_sampling_rng.bit_generator.state),
                environment_reset_rng=None,
                reset_semantics="deterministic formal reset; no separate environment RNG is consumed",
                initial_dynamic_cache=np.asarray(ctx.initial_cache, dtype=np.int8),
                initial_uav_state=ctx.initial_uav,
                tv_local_cache_state=ctx.local_cache,
            )
        parameters_before_collection = (
            {
                name: parameter.detach().clone()
                for name, parameter in policy.named_parameters()
            }
            if frozen_batch_family_mode
            else None
        )
        direct_cf_reservoir = (
            _build_direct_cf_reservoir(
                ctx.action_spec,
                configured_cf_samples,
                cf_sampling_rng,
                sampling_strategy,
            )
            if _is_sparse_direct_cf_mode(ctx.cfg)
            else None
        )
        batch = collect_formal_episode_rollout(
            policy,
            ctx,
            protocol,
            direct_cf_reservoir,
            episode_index=episode_index,
            collection_policy_version=collection_policy_version,
            collect_reference_states=reference_drift_mode,
        )
        raw_stats = _formal_raw_rollout_statistics(batch)
        if (
            _is_sparse_direct_cf_mode(ctx.cfg)
            and episode_index == 0
            and configured_cf_samples == 8
        ):
            _validate_k8_episode1_baseline(batch, raw_stats)
        direct_cf_samples = None
        cf_trigger_sample_counts = None
        cf_effective_sample_count = 0
        cf_trigger_free_counts = None
        cf_trigger_sampling_weights = None
        if _is_sparse_direct_cf_mode(ctx.cfg):
            if direct_cf_reservoir is None or direct_cf_steps is None:
                raise AssertionError("sparse_direct_cf reservoir/steps未构造。")
            direct_cf_samples = ppo_rollout.run_direct_cf_branches(
                policy,
                direct_cf_reservoir,
                direct_cf_steps,
                batch,
                ctx.local_cache,
                ctx.service_catalog,
                ctx.cfg,
                ctx.cache_nodes,
                ctx.cache_cfg,
                ctx.flight_boundary,
                ctx.observation_spec,
                ctx.action_spec,
                ctx.terminal_slot,
                collection_policy_version=collection_policy_version,
                episode_index=episode_index,
                cf_parallel_workers=_cf_parallel_workers(ctx.cfg),
            )
            if len(direct_cf_samples) != direct_cf_reservoir.sample_count:
                raise RuntimeError("Direct-CF branch成功样本数与reservoir数量不一致。")
            if frozen_batch_family_mode:
                if any(
                    int(item.collection_policy_version) != collection_policy_version
                    for item in direct_cf_samples
                ):
                    raise RuntimeError("Direct-CF样本collection_policy_version写入不一致。")
                identity_errors = [
                    abs(
                        float(item.delta_g)
                        - float(item.delta_q)
                        + float(item.delta_cache)
                        + float(item.delta_uav)
                    )
                    for item in direct_cf_samples
                ]
                if not identity_errors or max(identity_errors) > 1.0e-9:
                    raise AssertionError(
                        "Direct-CF Delta identity max error超过容差："
                        f"{max(identity_errors) if identity_errors else None}"
                    )
            cf_trigger_sample_counts = _direct_cf_trigger_sample_counts(
                direct_cf_samples,
                tuple(ctx.action_spec.labels),
            )
            (
                cf_effective_sample_count,
                cf_trigger_free_counts,
                reservoir_sample_counts,
                cf_trigger_sampling_weights,
            ) = _direct_cf_reservoir_reporting(direct_cf_reservoir)
            if reservoir_sample_counts != [
                int(cf_trigger_sample_counts[label])
                for label in ctx.action_spec.labels
            ]:
                raise AssertionError("Direct-CF branch样本Trigger计数与reservoir不一致。")
        if reference_drift_mode:
            if not batch.reference_decisions:
                raise RuntimeError("refdrift baseline rollout缺少reference decisions。")
            selected_ids = [str(item.decision_id) for item in direct_cf_samples or []]
            try:
                _, _, partition = reference_drift.validate_reference_partition(
                    batch.reference_decisions,
                    selected_ids,
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"refdrift当前episode reference partition无效：{exc}") from exc
            if partition["selected_count"] != len(selected_ids) or partition["intersection_count"] != 0:
                raise RuntimeError("refdrift当前episode selected/reference partition不一致。")
            if any(
                int(item.collection_policy_version) != collection_policy_version
                for item in batch.reference_decisions
            ):
                raise RuntimeError("refdrift reference collection_policy_version写入不一致。")
            selected_id_set = set(selected_ids)
            pending_reference_records.extend(
                replace(
                    item,
                    selected_for_direct_cf=str(item.decision_id) in selected_id_set,
                )
                for item in batch.reference_decisions
            )
        if frozen_batch_family_mode:
            if direct_cf_samples is None:
                raise AssertionError("frozen_policy_batch缺少当前episode Direct-CF samples。")
            current_batch_episode_index = int(
                episode_index % configured_cf_batch_episodes + 1
            )
            current_batch_id = int(episode_index // configured_cf_batch_episodes + 1)
            pending_direct_cf_samples.extend(direct_cf_samples)
            expected_pending_count = current_batch_episode_index * configured_cf_samples
            if current_batch_episode_index < configured_cf_batch_episodes:
                if len(pending_direct_cf_samples) != expected_pending_count:
                    raise RuntimeError("frozen_policy_batch pending sample count不符合协议。")
                update = _frozen_policy_batch_collection_summary(
                    policy,
                    batch,
                    direct_cf_samples,
                    ctx.cfg,
                )
                if update.optimizer_steps != 0 or update.actor_update_executed:
                    raise AssertionError("冻结收集episode不得执行Actor optimizer step。")
                if parameters_before_collection is None:
                    raise AssertionError("frozen_policy_batch缺少collection参数快照。")
                for name, parameter in policy.named_parameters():
                    if not torch.equal(parameter.detach(), parameters_before_collection[name]):
                        raise AssertionError(
                            "冻结收集episode期间Actor参数发生变化："
                            f"{name}"
                        )
            else:
                expected_batch_sample_count = configured_cf_batch_episodes * configured_cf_samples
                if len(pending_direct_cf_samples) != expected_batch_sample_count:
                    raise RuntimeError(
                        "frozen_policy_batch pending sample count不等于32："
                        f"{len(pending_direct_cf_samples)}"
                    )
                batch_versions = {
                    int(item.collection_policy_version)
                    for item in pending_direct_cf_samples
                }
                if batch_versions != {collection_policy_version}:
                    raise RuntimeError(
                        "frozen_policy_batch不得混合collection policy version："
                        f"{sorted(batch_versions)}"
                    )
                batch_samples = [
                    replace(
                        item,
                        sampling_weight=1.0 / float(expected_batch_sample_count),
                    )
                    for item in pending_direct_cf_samples
                ]
                if reference_drift_mode:
                    if cf_sampling_rng is None:
                        raise AssertionError("refdrift causal artifact缺少cf_sampling_rng。")
                    pre_rng_states = _capture_rng_states(
                        policy,
                        rng_generator,
                        critic_rng_generator,
                    )
                    boundary_state = {
                        "completed_episode": int(episode_index) + 1,
                        "episode_index": int(episode_index),
                        "batch_id": int(current_batch_id),
                        "batch_episode_index": int(current_batch_episode_index),
                        "collection_policy_version": int(collection_policy_version),
                        "policy_version_pre": int(policy_version),
                        "optimizer_step_pre": int(optimizer_step_count),
                        "last_origin_index": int(batch.origin_indices[-1]),
                        "last_slot": int(batch.slots[-1]),
                        "true_terminal": bool(batch.metadata.get("true_terminal", False)),
                        "bootstrap_value": float(batch.final_bootstrap_value),
                        "final_cache_state": np.asarray(
                            batch.transitions[-1].next_cache_state,
                            dtype=np.int8,
                        ).copy(),
                        "final_uav_positions": copy.deepcopy(
                            batch.transitions[-1].next_uav_positions
                        ),
                        "pending_cf_count": int(len(batch_samples)),
                        "pending_reference_count": int(len(pending_reference_records)),
                    }
                    causal_update_handle = causal_artifacts.persist_pre_update_artifacts(
                        run_dir=_formal_run_dir,
                        update_index=int(current_batch_id),
                        batch_id=int(current_batch_id),
                        completed_episodes=int(episode_index) + 1,
                        collection_policy_version=int(collection_policy_version),
                        policy_version_pre=int(policy_version),
                        optimizer_step_pre=int(optimizer_step_count),
                        policy=policy,
                        optimizer=optimizer,
                        rng_states=pre_rng_states,
                        cf_sampling_rng_state=copy.deepcopy(
                            cf_sampling_rng.bit_generator.state
                        ),
                        samples=list(pending_direct_cf_samples),
                        training_samples=batch_samples,
                        reference_records=list(pending_reference_records),
                        pending_reference_count=len(pending_reference_records),
                        reference_drift_cap=0.005,
                        algorithm_schema=ppo_update.REFERENCE_DRIFT_ALGORITHM_SCHEMA,
                        architecture_id=ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID,
                        checkpoint_schema=REFERENCE_DRIFT_CHECKPOINT_VERSION,
                        compatibility_snapshot=compatibility,
                        boundary_state=boundary_state,
                        expected_sample_count=expected_batch_sample_count,
                    )
                update = update_policy_on_rollout(
                    policy,
                    optimizer,
                    batch,
                    ctx.cfg,
                    rng_generator,
                    direct_cf_samples=batch_samples,
                    expected_collection_policy_version=collection_policy_version,
                    reference_records=(
                        pending_reference_records if reference_drift_mode else None
                    ),
                    reference_probability_drift_cap=(
                        0.005 if reference_drift_mode else None
                    ),
                )
                if update.optimizer_steps != int(protocol.update_epochs):
                    raise RuntimeError("frozen_policy_batch边界update optimizer steps不符合协议。")
                if len(update.epoch_summaries) != int(protocol.update_epochs):
                    raise RuntimeError("frozen_policy_batch边界update epoch summaries不完整。")
                if not update.actor_gradient_finite or not update.gradient_finite:
                    raise FloatingPointError("frozen_policy_batch gradient必须finite。")
                if update.encoder_parameter_delta_norm is None or update.encoder_parameter_delta_norm <= 0.0:
                    raise AssertionError("frozen_policy_batch encoder必须发生非零更新。")
                if update.autoregressive_parameter_delta_norm is None or update.autoregressive_parameter_delta_norm <= 0.0:
                    raise AssertionError("frozen_policy_batch autoregressive block必须发生非零更新。")
                if update.logit_head_parameter_delta_norm is None or update.logit_head_parameter_delta_norm <= 0.0:
                    raise AssertionError("frozen_policy_batch shared logit_head必须发生非零更新。")
                if update.value_head_parameter_delta_norm != 0.0:
                    raise AssertionError("frozen_policy_batch value_head delta必须严格为0。")
                if reference_drift_mode:
                    if update.reference_state_count != len(pending_reference_records):
                        raise AssertionError("refdrift reference_state_count与pending records不一致。")
                    if update.reference_selected_count != expected_batch_sample_count:
                        raise AssertionError("refdrift selected reference count不等于32。")
                    if update.reference_selected_intersection_count != 0:
                        raise AssertionError("refdrift selected/reference intersection必须为0。")
                    if (
                        update.reference_constrained_drift_max is None
                        or update.old_p_reconstruction_tolerance is None
                        or update.reference_constrained_drift_max
                        > 0.005 + update.old_p_reconstruction_tolerance
                    ):
                        raise AssertionError("refdrift constrained reference drift超过固定cap。")
        else:
            update = update_policy_on_rollout(
                policy,
                optimizer,
                batch,
                ctx.cfg,
                rng_generator,
                q_critic,
                q_optimizer,
                critic_replay,
                critic_rng_generator,
                credit_phase,
                protocol.critic_return_scale
                if credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT
                else None,
                direct_cf_samples,
            )
        expected_actor_steps = int(protocol.expected_optimizer_steps_per_rollout)
        if credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT:
            if frozen_batch_family_mode:
                expected_actor_steps = (
                    0
                    if int(episode_index % configured_cf_batch_episodes + 1)
                    < configured_cf_batch_episodes
                    else int(protocol.update_epochs)
                )
            else:
                expected_actor_steps = int(protocol.update_epochs)
        elif (
            credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT
            and credit_phase == "critic_warmup"
        ):
            expected_actor_steps = 0
        if update.optimizer_steps != expected_actor_steps:
            raise RuntimeError(
                "完整episode的PPO optimizer steps不符合FormalTrainingProtocol，"
                "不得写入completed history。"
            )
        expected_epoch_summaries = (
            0
            if (
                (
                    credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT
                    and credit_phase == "critic_warmup"
                )
                or (
                    frozen_batch_family_mode
                    and int(episode_index % configured_cf_batch_episodes + 1)
                    < configured_cf_batch_episodes
                )
            )
            else int(protocol.update_epochs)
        )
        if len(update.epoch_summaries) != expected_epoch_summaries:
            raise RuntimeError(
                "FormalPPOUpdateSummary epoch_summaries不完整，不得写入completed history。"
            )
        if _is_q_mode(ctx.cfg) and update.q_optimizer_steps != int(protocol.expected_optimizer_steps_per_rollout):
            raise RuntimeError("central/per-trigger Q critic optimizer steps不符合FormalTrainingProtocol。")
        policy_version_before = policy_version
        if update.actor_update_executed:
            policy_version += 1
        optimizer_step_count += update.optimizer_steps
        q_optimizer_step_count += int(update.q_optimizer_steps)
        if causal_update_handle is not None:
            if cf_sampling_rng is None:
                raise AssertionError("refdrift causal post artifact缺少cf_sampling_rng。")
            try:
                causal_artifacts.finalize_post_update_artifact(
                    causal_update_handle,
                    policy_version_post=int(policy_version),
                    optimizer_step_post=int(optimizer_step_count),
                    policy=policy,
                    optimizer=optimizer,
                    rng_states=_capture_rng_states(
                        policy,
                        rng_generator,
                        critic_rng_generator,
                    ),
                    cf_sampling_rng_state=copy.deepcopy(
                        cf_sampling_rng.bit_generator.state
                    ),
                    update_summary=update,
                    critic_step_count=int(q_optimizer_step_count),
                )
                # The human-readable per-update summary is written only after
                # post_update is durable; pending state is cleared afterwards.
                _atomic_json_save(
                    {
                        "episode_index": int(episode_index),
                        "collection_policy_version": int(collection_policy_version),
                        "ppo_update_summary": update,
                    },
                    _formal_run_dir / f"episode{int(episode_index) + 1}_update_summary.json",
                )
            except Exception as exc:
                if isinstance(exc, RuntimeError) and str(exc).startswith(
                    "FORMAL_ARTIFACT_PERSISTENCE_FAIL:"
                ):
                    raise
                raise RuntimeError(
                    "FORMAL_ARTIFACT_PERSISTENCE_FAIL: post/update summary持久化失败。"
                ) from exc
            pending_direct_cf_samples.clear()
            if reference_drift_mode:
                pending_reference_records.clear()
        if frozen_batch_family_mode:
            if direct_cf_samples is None:
                raise AssertionError("frozen_policy_batch episode缺少Direct-CF samples。")
            episode_cf_deltas = np.asarray(
                [item.delta_g for item in direct_cf_samples], dtype=np.float64
            )
            episode_cf_advantages = np.asarray(
                [item.a_cf for item in direct_cf_samples], dtype=np.float64
            )
            episode_cf_sample_count = int(len(direct_cf_samples))
            episode_branch_steps = int(
                sum(item.branch_environment_steps for item in direct_cf_samples)
            )
            current_batch_id = int(episode_index // configured_cf_batch_episodes + 1)
            current_batch_episode_index = int(
                episode_index % configured_cf_batch_episodes + 1
            )
        else:
            episode_cf_deltas = None
            episode_cf_advantages = None
            episode_cf_sample_count = update.cf_sample_count
            episode_branch_steps = update.branch_environment_steps
            current_batch_id = None
            current_batch_episode_index = None
        episode_summary = TrainingEpisodeSummary(
            episode_index=episode_index,
            policy_version_before=policy_version_before,
            policy_version_after=policy_version,
            rollout_reward_sum=raw_stats["raw_reward_sum"],
            rollout_reward_mean=raw_stats["raw_reward_mean"],
            batch_reward_sum_float32=raw_stats["batch_reward_sum_float32"],
            batch_vs_raw_reward_sum_error=raw_stats["batch_vs_raw_reward_sum_error"],
            actual_q_sum=raw_stats["raw_actual_q_sum"],
            cache_cost_sum=raw_stats["raw_cache_cost_sum"],
            movement_cost_sum=raw_stats["raw_movement_cost_sum"],
            policy_loss_mean=update.policy_loss_mean,
            value_loss_mean=update.value_loss_mean,
            entropy_mean=update.entropy_mean,
            approx_kl_mean=update.approx_kl_mean,
            clip_fraction_mean=update.clip_fraction_mean,
            gradient_norm_mean=update.gradient_norm_mean,
            optimizer_steps=update.optimizer_steps,
            credit_phase=update.credit_phase,
            critic_loss_before=update.critic_loss_before,
            critic_loss_after=update.critic_loss_after,
            critic_gradient_norm_mean=update.critic_gradient_norm_mean,
            critic_gradient_finite=update.critic_gradient_finite,
            critic_action_gap_abs_mean=update.critic_action_gap_abs_mean,
            critic_action_gap_std=update.critic_action_gap_std,
            counterfactual_advantage_abs_mean=update.counterfactual_advantage_abs_mean,
            actor_update_executed=update.actor_update_executed,
            actor_updated_this_episode=update.actor_update_executed,
            actor_optimizer_steps_this_episode=update.optimizer_steps,
            critic_replay_sample_count=update.critic_replay_sample_count,
            cf_sample_count=episode_cf_sample_count,
            cf_delta_mean=(
                float(np.mean(episode_cf_deltas))
                if episode_cf_deltas is not None
                else update.cf_delta_mean
            ),
            cf_delta_std=(
                float(np.std(episode_cf_deltas))
                if episode_cf_deltas is not None
                else update.cf_delta_std
            ),
            cf_delta_abs_mean=(
                float(np.mean(np.abs(episode_cf_deltas)))
                if episode_cf_deltas is not None
                else update.cf_delta_abs_mean
            ),
            cf_delta_positive_fraction=(
                float(np.mean(episode_cf_deltas > 0.0))
                if episode_cf_deltas is not None
                else update.cf_delta_positive_fraction
            ),
            cf_advantage_mean=(
                float(np.mean(episode_cf_advantages))
                if episode_cf_advantages is not None
                else update.cf_advantage_mean
            ),
            cf_advantage_std=(
                float(np.std(episode_cf_advantages))
                if episode_cf_advantages is not None
                else update.cf_advantage_std
            ),
            cf_advantage_abs_mean=(
                float(np.mean(np.abs(episode_cf_advantages)))
                if episode_cf_advantages is not None
                else update.cf_advantage_abs_mean
            ),
            direct_cf_local_objective_change_proxy=update.direct_cf_local_objective_change_proxy,
            branch_environment_steps=episode_branch_steps,
            actor_gradient_finite=update.actor_gradient_finite,
            policy_entropy=update.policy_entropy,
            probability_drift_mean_abs=update.probability_drift_mean_abs,
            probability_drift_p95=update.probability_drift_p95,
            probability_drift_p99=update.probability_drift_p99,
            probability_drift_max=update.probability_drift_max,
            encoder_parameter_delta_norm=update.encoder_parameter_delta_norm,
            trigger_heads_parameter_delta_norm=update.trigger_heads_parameter_delta_norm,
            autoregressive_parameter_delta_norm=update.autoregressive_parameter_delta_norm,
            value_head_parameter_delta_norm=update.value_head_parameter_delta_norm,
            logit_head_parameter_delta_norm=update.logit_head_parameter_delta_norm,
            trigger_action1_counts=_rollout_trigger_action1_counts(
                batch,
                tuple(ctx.action_spec.labels),
            ),
            cf_trigger_sample_counts=cf_trigger_sample_counts,
            cf_sampling_strategy=sampling_strategy,
            cf_effective_sample_count=int(cf_effective_sample_count),
            cf_trigger_free_counts=cf_trigger_free_counts,
            cf_trigger_sampling_weights=cf_trigger_sampling_weights,
            cf_batch_id=current_batch_id,
            cf_batch_episode_index=current_batch_episode_index,
            cf_batch_pending_sample_count=(
                len(pending_direct_cf_samples) if frozen_batch_family_mode else 0
            ),
            collection_policy_version=(
                collection_policy_version if frozen_batch_family_mode else None
            ),
            actor_batch_sample_count=update.actor_batch_sample_count,
            pre_clip_gradient_norm=(
                update.gradient_norm_mean if update.optimizer_steps > 0 else None
            ),
            cf_batch_vshift_sum=update.cf_batch_vshift_sum,
            cf_batch_vshift_mean=update.cf_batch_vshift_mean,
            cf_batch_vshift_positive_fraction=update.cf_batch_vshift_positive_fraction,
            reference_raw_drift_mean=update.reference_raw_drift_mean,
            reference_raw_drift_median=update.reference_raw_drift_median,
            reference_raw_drift_p95=update.reference_raw_drift_p95,
            reference_raw_drift_p99=update.reference_raw_drift_p99,
            reference_raw_drift_max=update.reference_raw_drift_max,
            reference_constrained_drift_mean=update.reference_constrained_drift_mean,
            reference_constrained_drift_median=update.reference_constrained_drift_median,
            reference_constrained_drift_p95=update.reference_constrained_drift_p95,
            reference_constrained_drift_p99=update.reference_constrained_drift_p99,
            reference_constrained_drift_max=update.reference_constrained_drift_max,
            selected_drift_mean=update.selected_drift_mean,
            selected_drift_p95=update.selected_drift_p95,
            selected_drift_max=update.selected_drift_max,
            reference_drift_alpha=update.reference_drift_alpha,
            reference_drift_constraint_active=update.reference_drift_constraint_active,
            reference_probability_drift_cap=update.reference_probability_drift_cap,
            reference_state_count=update.reference_state_count,
            reference_unselected_count=update.reference_unselected_count,
            reference_selected_count=update.reference_selected_count,
            reference_selected_intersection_count=update.reference_selected_intersection_count,
            reference_drift_by_trigger=update.reference_drift_by_trigger,
            old_p_reconstruction_max_error=update.old_p_reconstruction_max_error,
            old_p_reconstruction_checked=update.old_p_reconstruction_checked,
            old_p_reconstruction_tolerance=update.old_p_reconstruction_tolerance,
            probability_dtype=update.probability_dtype,
            reference_count_checked=update.reference_count_checked,
            direct_cf_raw_vshift_sum=update.direct_cf_raw_vshift_sum,
            direct_cf_raw_vshift_mean=update.direct_cf_raw_vshift_mean,
            direct_cf_raw_vshift_positive_fraction=update.direct_cf_raw_vshift_positive_fraction,
            direct_cf_constrained_vshift_sum=update.direct_cf_constrained_vshift_sum,
            direct_cf_constrained_vshift_mean=update.direct_cf_constrained_vshift_mean,
            direct_cf_constrained_vshift_positive_fraction=update.direct_cf_constrained_vshift_positive_fraction,
            direct_cf_vshift_retention_ratio=update.direct_cf_vshift_retention_ratio,
        )
        episode_summaries.append(episode_summary)

        # 只有完整episode、完整PPO update和summary全部成功后才进入history。
        records.append(
            _training_episode_history_record(
                episode_summary,
                update,
                credit_assignment,
            )
        )
        history = _build_training_history(
            records,
            policy_version,
            optimizer_step_count,
            credit_assignment,
            q_optimizer_step_count,
            credit_phase,
            q_optimizer_step_count,
            protocol.critic_return_scale
            if credit_assignment == ppo_update.CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT
            else None,
            sampling_strategy,
            method_id=(
                ppo_update.REFERENCE_DRIFT_FROZEN_POLICY_BATCH_METHOD
                if reference_drift_mode
                else (ppo_update.FROZEN_POLICY_BATCH_METHOD if frozen_policy_batch_mode else None)
            ),
            architecture_id=(
                ppo_trigger_policy.FULL_SHARED_ACTOR_ARCHITECTURE_ID
                if frozen_batch_family_mode
                else None
            ),
            cf_batch_episodes=(
                configured_cf_batch_episodes if frozen_batch_family_mode else 1
            ),
            cf_samples_per_episode=(
                configured_cf_samples if frozen_batch_family_mode else 0
            ),
            batch_id=(current_batch_id or 0),
            batch_episode_index=(current_batch_episode_index or 0),
            pending_cf_sample_count=(
                len(pending_direct_cf_samples) if frozen_batch_family_mode else 0
            ),
            pending_collection_policy_version=(
                collection_policy_version
                if reference_drift_mode and pending_direct_cf_samples
                else None
            ),
            reference_probability_drift_cap=(0.005 if reference_drift_mode else None),
            pending_reference_state_count=(
                len(pending_reference_records) if reference_drift_mode else 0
            ),
        )
        payload = _checkpoint_payload(
            policy,
            optimizer,
            policy_version,
            optimizer_step_count,
            history,
            compatibility,
            _capture_rng_states(policy, rng_generator, critic_rng_generator),
            credit_assignment,
            q_critic,
            q_optimizer,
            q_optimizer_step_count,
            critic_replay,
            cf_sampling_rng,
            pending_direct_cf_samples=pending_direct_cf_samples,
            pending_reference_records=pending_reference_records,
        )
        # checkpoint先写；JSON mirror失败时旧checkpoint/新checkpoint都仍是权威源。
        _atomic_torch_save(payload, checkpoint_path)
        try:
            _atomic_json_save(history, history_path)
        except Exception as exc:  # checkpoint已经成功，保留其权威性供下次resume同步
            warnings.warn(
                f"TRAINING_HISTORY_JSON_MIRROR_WRITE_FAILED: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        if frozen_policy_batch_mode and current_batch_episode_index != configured_cf_batch_episodes:
            print(
                f"[PPO] batch {current_batch_id} episode {current_batch_episode_index}/"
                f"{configured_cf_batch_episodes} ({episode_index + 1}/{num_episodes}) | "
                f"mean_G={episode_summary.rollout_reward_mean:.4f} | "
                f"Q={episode_summary.actual_q_sum:.4f} | "
                f"cache={episode_summary.cache_cost_sum:.4f} | "
                f"uav={episode_summary.movement_cost_sum:.4f} | "
                f"policy={collection_policy_version}->{policy_version} | "
                f"pending={len(pending_direct_cf_samples)} | opt=0 | "
                f"time={time.perf_counter() - episode_start_time:.1f}s",
                flush=True,
            )
        elif _is_sparse_direct_cf_mode(ctx.cfg):
            print(
                f"[PPO] episode {episode_index + 1}/{num_episodes} | "
                f"mean_G={episode_summary.rollout_reward_mean:.4f} | "
                f"Q={episode_summary.actual_q_sum:.4f} | "
                f"cache={episode_summary.cache_cost_sum:.4f} | "
                f"uav={episode_summary.movement_cost_sum:.4f} | "
                f"entropy={_format_optional_float_for_reporting(episode_summary.policy_entropy)} | "
                f"cf_abs={_format_optional_float_for_reporting(episode_summary.cf_delta_abs_mean)} | "
                f"cf_pos={_format_optional_float_for_reporting(episode_summary.cf_delta_positive_fraction)} | "
                f"loss={episode_summary.policy_loss_mean:.4f} | "
                f"grad_norm={episode_summary.gradient_norm_mean:.4f} | "
                f"kl={episode_summary.approx_kl_mean:.4f} | "
                f"clip={episode_summary.clip_fraction_mean:.4f} | "
                f"drift={_format_optional_float_for_reporting(episode_summary.probability_drift_mean_abs)}/"
                f"{_format_optional_float_for_reporting(episode_summary.probability_drift_p95)}/"
                f"{_format_optional_float_for_reporting(episode_summary.probability_drift_p99)}/"
                f"{_format_optional_float_for_reporting(episode_summary.probability_drift_max)} | "
                f"proxy={_format_optional_float_for_reporting(episode_summary.direct_cf_local_objective_change_proxy)} | "
                f"branches={episode_summary.branch_environment_steps} | "
                f"time={time.perf_counter() - episode_start_time:.1f}s",
                flush=True,
            )
        else:
            print(
                f"[PPO] episode {episode_index + 1}/{num_episodes} | "
                f"mean_G={episode_summary.rollout_reward_mean:.4f} | "
                f"Q={episode_summary.actual_q_sum:.4f} | "
                f"cache={episode_summary.cache_cost_sum:.4f} | "
                f"uav={episode_summary.movement_cost_sum:.4f} | "
                f"entropy={episode_summary.entropy_mean:.4f} | "
                f"value_loss={episode_summary.value_loss_mean:.4f} | "
                f"grad_norm={episode_summary.gradient_norm_mean:.4f} | "
                f"kl={episode_summary.approx_kl_mean:.4f} | "
                f"clip={episode_summary.clip_fraction_mean:.4f} | "
                f"time={time.perf_counter() - episode_start_time:.1f}s",
                flush=True,
            )

    # 所有本次调用目标episode完成且history已逐episode持久化后，只更新一次Figure 1。
    if not frozen_batch_family_mode:
        _plot_training_convergence_after_persistence(_formal_run_dir, history)
    print(
        f"[PPO] training complete: {int(history['completed_episodes'])}/{num_episodes} episodes",
        flush=True,
    )

    return FormalTrainingRunResult(
        plan=plan,
        episode_summaries=episode_summaries,
        final_policy_version=policy_version,
        optimizer_step_count=optimizer_step_count,
        completed_episodes=int(history["completed_episodes"]),
        training_history=history,
        resumed=resumed,
        q_optimizer_step_count=q_optimizer_step_count,
    )


def _execution_location_for_matching_target(
    target: Any,
    cache_nodes: dict[str, Any],
) -> str:
    """按最终Matching target和cache node role归并执行位置类别。"""

    target_id = str(target)
    if target_id == task_allocation.LOCAL or target_id.startswith(task_allocation.LOCAL_PREFIX):
        return "Local"
    if target_id == task_allocation.CLOUD:
        return "Cloud"

    node_ids = [str(value) for value in cache_nodes["node_ids"]]
    if target_id not in node_ids:
        raise ValueError(f"最终Matching包含未知执行节点：{target_id}")
    node_index = node_ids.index(target_id)
    role = str(cache_nodes["node_roles"][node_index])
    if role not in {"SV", "RSU", "UAV"}:
        raise ValueError(f"最终Matching包含未知edge node role：{role}")
    return role


def _execution_location_statistics(
    counts: dict[str, int],
    arrived_task_count: int,
) -> tuple[dict[str, int], dict[str, float]]:
    """校验五类最终执行位置计数，并计算原始[0,1]比例。"""

    if tuple(counts) != EXECUTION_LOCATION_ORDER:
        raise AssertionError(
            "execution location类别顺序/集合必须严格为 "
            f"{list(EXECUTION_LOCATION_ORDER)}，实际为 {list(counts)}"
        )
    if any(int(value) < 0 for value in counts.values()):
        raise AssertionError("execution location task count不能为负。")
    if sum(int(value) for value in counts.values()) != int(arrived_task_count):
        raise AssertionError(
            "execution_location_task_counts之和不等于arrived_task_count："
            f"{sum(counts.values())} != {arrived_task_count}"
        )

    if arrived_task_count <= 0:
        raise ValueError(
            "arrived_task_count必须大于0才能定义execution_location_ratios；"
            "不会静默返回未归一化比例。"
        )

    ratios = {
        location: float(int(counts[location]) / int(arrived_task_count))
        for location in EXECUTION_LOCATION_ORDER
    }
    if not math.isclose(math.fsum(ratios.values()), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(
            "execution_location_ratios之和不接近1.0："
            f"sum={math.fsum(ratios.values())}"
        )
    return dict(counts), ratios


def evaluate_held_out_test(
    policy: Any,
    ctx: TrainingSanityContext,
    protocol: FormalTrainingProtocol,
) -> DeterministicEvaluationSummary:
    """执行独立 held-out test evaluation。

    只执行 protocol 的 test origin 区间，使用 ``torch.no_grad``，不接收或调用
    optimizer，也不修改 policy。测试开始 reset 到正式共同初始 C/Q，并在 test
    区间内顺序传播状态；train origin 的 actual task/reward 不会进入本函数。

    注意：这是正式 held-out test 执行器；CLI protocol-check 不会调用环境。
    """

    _require_torch()
    steps = _held_out_test_aligned_steps(ctx, protocol)
    policy_state_before = {
        str(name): value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }
    previous_cache = np.asarray(ctx.initial_cache, dtype=np.int8).copy()
    previous_uav = {
        str(uav_id): (float(np.asarray(xy)[0]), float(np.asarray(xy)[1]))
        for uav_id, xy in ctx.initial_uav.items()
    }
    cache_trigger_counts = {label: 0 for label in ctx.action_spec.labels[: len(ctx.action_spec.cache_node_ids)]}
    uav_trigger_counts = {label: 0 for label in ctx.action_spec.labels[len(ctx.action_spec.cache_node_ids) :]}
    rewards: list[float] = []
    actual_q_values: list[float] = []
    cache_cost_values: list[float] = []
    movement_cost_values: list[float] = []
    arrived_task_count = 0
    cache_service_available_task_count = 0
    cloud_execution_task_count = 0
    execution_location_task_counts = {
        location: 0 for location in EXECUTION_LOCATION_ORDER
    }
    all_steps_feasible = True
    edge_node_mask = np.asarray(
        [str(role) in {"SV", "RSU", "UAV"} for role in ctx.cache_nodes["node_roles"]],
        dtype=bool,
    )
    if not np.any(edge_node_mask):
        raise ValueError("evaluation 缺少 SV/RSU/UAV edge cache node。")

    with torch.no_grad():
        for step in steps:
            observation = ppo_trigger_policy.build_configuration_observation(
                previous_cache,
                previous_uav,
                step.predicted_vehicle_positions,
                step.predicted_service_popularity,
                ctx.observation_spec,
                ctx.flight_boundary,
            )
            action_mask_provider = ppo_rollout.build_trigger_action_mask_provider(
                previous_cache,
                previous_uav,
                ctx.service_catalog,
                ctx.cfg,
                ctx.cache_nodes,
                ctx.flight_boundary,
                ctx.action_spec,
            )
            sample = policy.sample_action(
                observation,
                deterministic=True,
                action_mask=action_mask_provider,
            )
            result = slot_environment.execute_slot(
                previous_cache,
                previous_uav,
                sample.cache_triggers,
                sample.uav_triggers,
                step.scenario_slice,
                step.actual_tasks,
                step.actual_vehicle_positions,
                ctx.local_cache,
                ctx.service_catalog,
                ctx.cfg,
                ctx.cache_nodes,
                ctx.cache_cfg,
                ctx.flight_boundary,
                slot_id=step.slot,
                time_s=step.time_s,
            )
            if not result.step_feasible:
                all_steps_feasible = False
                raise ppo_rollout.InfeasibleActionBlocker(
                    json.dumps(
                        {
                            "message": "INFEASIBLE ACTION BLOCKER DURING DETERMINISTIC EVALUATION",
                            "origin_index": step.origin_index,
                            "slot": step.slot,
                            "time_s": step.time_s,
                            "action_vector": sample.action_vector.astype(int).tolist(),
                            "cache_triggers": sample.cache_triggers.astype(int).tolist(),
                            "uav_triggers": sample.uav_triggers,
                            "infeasible_reason": result.infeasible_reason,
                        },
                        ensure_ascii=False,
                    )
                )

            reward = float(result.system_utility)
            actual_q = float(result.actual_task_utility)
            cache_cost = float(result.cache_cost)
            movement_cost = float(result.movement_cost)
            final_cache_state = np.asarray(result.final_cache_state, dtype=np.int8)
            expected_cache_shape = (
                len(ctx.cache_nodes["node_ids"]),
                int(ctx.service_catalog["num_services"]),
            )
            if final_cache_state.shape != expected_cache_shape:
                raise ValueError(
                    "evaluation final_cache_state shape="
                    f"{final_cache_state.shape}，期望 {expected_cache_shape}。"
                )
            if not np.all(np.isin(final_cache_state, [0, 1])):
                raise ValueError("evaluation final_cache_state 必须只包含0/1。")

            # 图5的可用率只统计当前已到达任务；local cache 必须属于任务车辆自身。
            edge_cache_by_service = np.any(final_cache_state[edge_node_mask] == 1, axis=0)
            arrived_task_count += len(step.actual_tasks)
            for task in step.actual_tasks:
                service_id = int(task.service_id)
                if not 0 <= service_id < expected_cache_shape[1]:
                    raise ValueError(f"task {task.task_id} 的 service_id 非法：{service_id}")
                own_local_cache = ctx.local_cache.get(str(task.vehicle_id), set())
                if service_id in own_local_cache or bool(edge_cache_by_service[service_id]):
                    cache_service_available_task_count += 1

                if task.task_id not in result.matching:
                    raise ValueError(f"最终Matching缺少 task_id={task.task_id}。")
                execution_location = _execution_location_for_matching_target(
                    result.matching[task.task_id],
                    ctx.cache_nodes,
                )
                execution_location_task_counts[execution_location] += 1
                if execution_location == "Cloud":
                    cloud_execution_task_count += 1

            if not math.isclose(reward, actual_q - cache_cost - movement_cost, rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError("evaluation reward identity G=Q_actual-C_ca-C_q 失败。")
            rewards.append(reward)
            actual_q_values.append(actual_q)
            cache_cost_values.append(cache_cost)
            movement_cost_values.append(movement_cost)
            for index, trigger in enumerate(sample.cache_triggers.astype(int).tolist()):
                cache_trigger_counts[ctx.action_spec.labels[index]] += int(trigger)
            cache_count = len(ctx.action_spec.cache_node_ids)
            for uav_index, uav_id in enumerate(ctx.action_spec.uav_ids):
                uav_trigger_counts[ctx.action_spec.labels[cache_count + uav_index]] += int(sample.uav_triggers[uav_id])
            previous_cache = np.asarray(result.next_cache_state, dtype=np.int8).copy()
            previous_uav = dict(result.next_uav_positions or {})

    policy_state_after = {
        str(name): value.detach().cpu()
        for name, value in policy.state_dict().items()
    }
    if set(policy_state_before) != set(policy_state_after) or any(
        not torch.equal(policy_state_before[name], policy_state_after[name])
        for name in policy_state_before
    ):
        raise AssertionError(
            "held-out test 不得修改 policy 参数或 buffer；检测到 state_dict 变化。"
        )

    cumulative_g = float(math.fsum(rewards))
    actual_q_sum = float(math.fsum(actual_q_values))
    cache_cost_sum = float(math.fsum(cache_cost_values))
    movement_cost_sum = float(math.fsum(movement_cost_values))
    expected_cumulative_g = actual_q_sum - cache_cost_sum - movement_cost_sum
    identity_error = abs(cumulative_g - expected_cumulative_g)
    if not math.isclose(
        cumulative_g,
        expected_cumulative_g,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise AssertionError(
            "evaluation episode级 G 恒等式失败："
            f"sum_G={cumulative_g}, expected={expected_cumulative_g}, "
            f"error={cumulative_g - expected_cumulative_g}"
        )
    if not 0 <= cache_service_available_task_count <= arrived_task_count:
        raise AssertionError("cache_service_available_task_count 超出 arrived_task_count。")
    if not 0 <= cloud_execution_task_count <= arrived_task_count:
        raise AssertionError("cloud_execution_task_count 超出 arrived_task_count。")
    cache_service_availability_rate = (
        None
        if arrived_task_count == 0
        else float(cache_service_available_task_count / arrived_task_count)
    )
    cloud_execution_ratio = (
        None
        if arrived_task_count == 0
        else float(cloud_execution_task_count / arrived_task_count)
    )
    if execution_location_task_counts["Cloud"] != cloud_execution_task_count:
        raise AssertionError(
            "Cloud execution count与execution_location_task_counts不一致。"
        )
    execution_location_task_counts, execution_location_ratios = _execution_location_statistics(
        execution_location_task_counts,
        arrived_task_count,
    )
    return DeterministicEvaluationSummary(
        num_steps=len(rewards),
        evaluation_origin_start=int(protocol.test_origin_indices[0]),
        evaluation_origin_end=int(protocol.test_origin_indices[-1]),
        evaluation_slot_start=int(protocol.test_slots[0]),
        evaluation_slot_end=int(protocol.test_slots[-1]),
        cumulative_system_utility=cumulative_g,
        mean_system_utility=float(cumulative_g / len(rewards)),
        cumulative_actual_task_utility=float(actual_q_sum),
        cumulative_cache_cost=float(cache_cost_sum),
        cumulative_movement_cost=float(movement_cost_sum),
        cache_trigger_counts=cache_trigger_counts,
        uav_movement_trigger_counts=uav_trigger_counts,
        arrived_task_count=int(arrived_task_count),
        cache_service_available_task_count=int(cache_service_available_task_count),
        cache_service_availability_rate=cache_service_availability_rate,
        cloud_execution_task_count=int(cloud_execution_task_count),
        cloud_execution_ratio=cloud_execution_ratio,
        execution_location_task_counts=execution_location_task_counts,
        execution_location_ratios=execution_location_ratios,
        all_steps_feasible=all_steps_feasible,
        reward_identity_error=float(identity_error),
        evaluation_semantics="held-out test deterministic evaluation",
    )


def evaluate_formal_episode(
    policy: Any,
    ctx: TrainingSanityContext,
    protocol: FormalTrainingProtocol,
) -> DeterministicEvaluationSummary:
    """Compatibility alias; historical callers now execute held-out test only."""

    return evaluate_held_out_test(policy, ctx, protocol)


def evaluate_held_out_test_from_checkpoint(
    checkpoint_path: str | Path,
    config_path: str = "system_config.toml",
) -> DeterministicEvaluationSummary:
    """Load a current-compatible policy and run held-out test only.

    This entrypoint intentionally constructs no optimizer and never calls a
    training or checkpoint-resume path.  Compatibility is checked against the
    current action/observation and train/test origin metadata before weights
    are loaded, so old checkpoints fail closed.
    """

    _require_torch()
    ctx = _build_context(config_path)
    protocol = build_formal_training_protocol(ctx.scenarios, ctx.cfg)
    validate_formal_training_protocol(protocol)
    policy = _build_formal_policy_only(ctx)
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_absolute():
        checkpoint = Path(ctx.cfg["_meta"]["project_root"]) / checkpoint
    checkpoint = checkpoint.resolve(strict=False)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"held-out test checkpoint不存在：{checkpoint}")
    try:
        payload = torch.load(
            checkpoint,
            map_location=_policy_device(policy),
            weights_only=False,
        )
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        payload = torch.load(checkpoint, map_location=_policy_device(policy))
    if not isinstance(payload, dict):
        raise RuntimeError("HELD_OUT_TEST_CHECKPOINT_ERROR: checkpoint顶层必须是dict。")
    expected_compatibility = _formal_compatibility_metadata(ctx, protocol, policy)
    actual_compatibility = payload.get("compatibility")
    mismatch = _metadata_mismatch(
        expected_compatibility,
        actual_compatibility,
        "checkpoint.compatibility",
    )
    if mismatch:
        raise RuntimeError(
            "CHECKPOINT_COMPATIBILITY_ERROR: held-out test拒绝不匹配checkpoint："
            f"{mismatch}"
        )
    if "policy_state_dict" not in payload:
        raise RuntimeError("HELD_OUT_TEST_CHECKPOINT_ERROR: checkpoint缺少policy_state_dict。")
    try:
        policy.load_state_dict(payload["policy_state_dict"], strict=True)
    except Exception as exc:
        raise RuntimeError(
            f"CHECKPOINT_COMPATIBILITY_ERROR: held-out test policy state不兼容：{exc}"
        ) from exc
    return evaluate_held_out_test(policy, ctx, protocol)


def run_held_out_test_evaluation(
    checkpoint_path: str | Path,
    config_path: str = "system_config.toml",
    report_path: str | Path | None = None,
) -> DeterministicEvaluationSummary:
    """Public held-out test entrypoint with optional JSON reporting."""

    summary = evaluate_held_out_test_from_checkpoint(
        checkpoint_path,
        config_path,
    )
    if report_path is not None:
        save_held_out_test_report(summary, report_path)
    return summary


def save_held_out_test_report(
    summary: DeterministicEvaluationSummary,
    output_path: str | Path,
    figure_output_path: str | Path | None = None,
) -> Path:
    """保存 held-out test JSON，并在JSON成功后生成Figure 2。

    该函数只消费已经完成的evaluation summary；不会加载checkpoint、执行环境、
    采样动作或修改policy。绘图失败只发出warning，已保存的evaluation JSON
    仍然有效。
    """

    output = Path(output_path)
    if output.suffix.lower() != ".json":
        raise ValueError("held-out test report必须输出为.json。")
    expected_counts, expected_ratios = _execution_location_statistics(
        summary.execution_location_task_counts,
        summary.arrived_task_count,
    )
    if expected_counts != summary.execution_location_task_counts or any(
        not math.isclose(
            expected_ratios[location],
            float(summary.execution_location_ratios[location]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for location in EXECUTION_LOCATION_ORDER
    ):
        raise ValueError("held-out test execution location reporting不一致。")
    _atomic_json_save(summary_to_builtin(summary), output)

    figure_path = (
        Path(figure_output_path)
        if figure_output_path is not None
        else output.parent / "figures" / "02_execution_location_ratio.png"
    )
    try:
        from experiment_plotting import plot_execution_location_ratios

        plot_execution_location_ratios(summary.execution_location_ratios, figure_path)
    except Exception as exc:  # JSON已落盘，reporting故障不回滚evaluation
        warnings.warn(
            f"HELD_OUT_TEST_EXECUTION_LOCATION_PLOT_FAILED: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    return output


def save_deterministic_evaluation_report(
    summary: DeterministicEvaluationSummary,
    output_path: str | Path,
    figure_output_path: str | Path | None = None,
) -> Path:
    """Compatibility alias for :func:`save_held_out_test_report`."""

    return save_held_out_test_report(summary, output_path, figure_output_path)


def evaluate_held_out_test_with_trigger_diagnostics(
    policy: Any,
    ctx: TrainingSanityContext,
    protocol: FormalTrainingProtocol,
) -> tuple[DeterministicEvaluationSummary, list[dict[str, Any]]]:
    """在 held-out test evaluation 上附加轻量 Trigger 概率记录。

    该reporting wrapper不复制evaluation loop，也不改变policy action、action mask、
    environment执行或状态传播；它只记录原``sample_action()``返回的概率、logit、
    deterministic action和mask，随后仍由``evaluate_held_out_test()``完成正式执行。
    """

    steps = _held_out_test_aligned_steps(ctx, protocol)
    captured_samples: list[dict[str, Any]] = []

    class _TriggerSampleCapture:
        def __init__(self, wrapped_policy: Any) -> None:
            self.wrapped_policy = wrapped_policy

        def state_dict(self) -> Any:
            return self.wrapped_policy.state_dict()

        def sample_action(self, *args: Any, **kwargs: Any) -> Any:
            if not bool(kwargs.get("deterministic", False)):
                raise AssertionError("trigger probability diagnostic 必须使用 deterministic=True。")
            if kwargs.get("action_mask") is None:
                raise AssertionError("trigger probability diagnostic 不允许缺少 action_mask。")
            sample = self.wrapped_policy.sample_action(*args, **kwargs)
            captured_samples.append(
                {
                    "action_vector": np.asarray(sample.action_vector, dtype=np.int8).astype(int).tolist(),
                    "per_action_probabilities": np.asarray(
                        sample.per_action_probabilities, dtype=np.float64
                    ).tolist(),
                    "per_action_logits": np.asarray(sample.per_action_logits, dtype=np.float64).tolist(),
                    "per_action_masks": np.asarray(sample.per_action_masks, dtype=bool).tolist(),
                }
            )
            return sample

    summary = evaluate_held_out_test(_TriggerSampleCapture(policy), ctx, protocol)
    if len(captured_samples) != len(steps):
        raise RuntimeError(
            "trigger probability diagnostic 记录数量与formal deterministic steps不一致："
            f"{len(captured_samples)} != {len(steps)}"
        )

    trajectory: list[dict[str, Any]] = []
    for step, captured in zip(steps, captured_samples):
        action_vector = captured["action_vector"]
        probabilities = captured["per_action_probabilities"]
        logits = captured["per_action_logits"]
        masks = captured["per_action_masks"]
        if not (
            len(action_vector) == len(ctx.action_spec.labels)
            and len(probabilities) == len(ctx.action_spec.labels)
            and len(logits) == len(ctx.action_spec.labels)
            and len(masks) == len(ctx.action_spec.labels)
        ):
            raise RuntimeError("trigger probability diagnostic action维度与TriggerActionSpec不一致。")
        trigger_records: list[dict[str, Any]] = []
        for index, label in enumerate(ctx.action_spec.labels):
            mask = [bool(value) for value in masks[index]]
            if len(mask) != 2 or not any(mask):
                raise RuntimeError(f"Trigger {label} 收到非法action mask：{mask}")
            p_action1 = float(probabilities[index])
            trigger_records.append(
                {
                    "label": label,
                    "deterministic_action": int(action_vector[index]),
                    "p_action0": float(1.0 - p_action1),
                    "p_action1": p_action1,
                    "logit": float(logits[index]),
                    "mask_action0_allowed": mask[0],
                    "mask_action1_allowed": mask[1],
                    "mask_forced": (
                        "zero" if mask == [True, False] else
                        "one" if mask == [False, True] else
                        None
                    ),
                }
            )
        trajectory.append(
            {
                "origin_index": int(step.origin_index),
                "slot": int(step.slot),
                "time_s": float(step.time_s),
                "triggers": trigger_records,
            }
        )
    return summary, trajectory


def evaluate_formal_episode_with_trigger_diagnostics(
    policy: Any,
    ctx: TrainingSanityContext,
    protocol: FormalTrainingProtocol,
) -> tuple[DeterministicEvaluationSummary, list[dict[str, Any]]]:
    """Compatibility alias for held-out test trigger diagnostics."""

    return evaluate_held_out_test_with_trigger_diagnostics(policy, ctx, protocol)


def _aligned_steps(ctx: TrainingSanityContext, origin_start_index: int) -> list[ppo_rollout.AlignedRolloutStep]:
    root = Path(ctx.cfg["_meta"]["project_root"])
    return ppo_rollout.prepare_aligned_rollout_steps(
        root,
        ctx.scenarios,
        origin_start_index,
        SANITY_MAX_STEPS_PER_ROLLOUT,
        root / "data/generated/vehicle_predictions.csv",
        root / "data/generated/popularity_predictions.csv",
        root / "data/generated/tasks.csv",
        root / "data/generated/trajectory.csv",
    )


def _collect_rollout(
    policy: Any,
    ctx: TrainingSanityContext,
    origin_start_index: int,
    initial_cache: np.ndarray,
    initial_uav: dict[str, Any],
    policy_version: int,
    rollout_index: int,
) -> tuple[ppo_rollout.RolloutBatch, RolloutSanitySummary]:
    steps = _aligned_steps(ctx, origin_start_index)
    batch = ppo_rollout.collect_on_policy_rollout(
        policy,
        initial_cache,
        initial_uav,
        steps,
        ctx.local_cache,
        ctx.service_catalog,
        ctx.cfg,
        ctx.cache_nodes,
        ctx.cache_cfg,
        ctx.flight_boundary,
        ctx.observation_spec,
        ctx.action_spec,
        ctx.gamma,
        ctx.gae_lambda,
        ctx.terminal_slot,
    )
    transitions = batch.transitions
    rewards = [float(item.reward) for item in transitions]
    raw_stats = _formal_raw_rollout_statistics(batch)
    summary = RolloutSanitySummary(
        rollout_index=rollout_index,
        collected_with_policy_version=policy_version,
        origin_indices=batch.origin_indices.astype(int).tolist(),
        slots=batch.slots.astype(int).tolist(),
        times_s=[float(item.time_s) for item in transitions],
        actions=[item.action_vector.astype(int).tolist() for item in transitions],
        rewards=rewards,
        reward_sum=raw_stats["raw_reward_sum"],
        reward_mean=raw_stats["raw_reward_mean"],
        actual_q_sum=raw_stats["raw_actual_q_sum"],
        cache_cost_sum=raw_stats["raw_cache_cost_sum"],
        movement_cost_sum=raw_stats["raw_movement_cost_sum"],
        step_count=len(transitions),
        final_cache_state=transitions[-1].next_cache_state.copy(),
        final_uav_positions=dict(transitions[-1].next_uav_positions),
        truncated=batch.truncated,
        true_terminal=bool(batch.metadata["true_terminal"]),
    )
    return batch, summary


def _compute_loss_checked(policy: Any, batch: ppo_rollout.RolloutBatch, cfg: ppo_update.PPOTrainingConfigStatus) -> ppo_update.PPOLossResult:
    loss = ppo_update.compute_ppo_loss(
        policy,
        batch,
        float(cfg.clip_epsilon),
        float(cfg.value_loss_coefficient),
        float(cfg.entropy_coefficient),
    )
    if not torch.isfinite(loss.total_loss):
        raise FloatingPointError("PPO total loss 非finite。")
    if not (abs(loss.ratio_mean - 1.0) < 1e-5 and abs(loss.approx_kl) < 1e-5 and loss.clip_fraction < 1e-6):
        raise RuntimeError("on-policy pre-update ratio/KL/clip sanity failed。")
    return loss


def _single_update(
    policy: Any,
    optimizer: Any,
    batch: ppo_rollout.RolloutBatch,
    cfg: ppo_update.PPOTrainingConfigStatus,
    policy_version_before: int,
    optimizer_step_count: int,
) -> UpdateSanitySummary:
    if optimizer_step_count >= SANITY_MAX_OPTIMIZER_STEPS:
        raise RuntimeError("optimizer step 硬上限已达。")
    loss = _compute_loss_checked(policy, batch, cfg)
    before_params = [param.detach().clone() for param in policy.parameters() if param.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    loss.total_loss.backward()
    grads = [param.grad.detach() for param in policy.parameters() if param.grad is not None]
    gradient_finite = bool(grads) and bool(all(torch.isfinite(grad).all().item() for grad in grads))
    pre_clip_norm = float(torch.sqrt(sum(torch.sum(grad ** 2) for grad in grads)).detach().cpu()) if grads else 0.0
    if not gradient_finite:
        raise FloatingPointError("gradient 出现 NaN/Inf。")
    returned_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), float(cfg.max_grad_norm)).detach().cpu())
    if not math.isclose(pre_clip_norm, returned_norm, rel_tol=1e-5, abs_tol=1e-5):
        pre_clip_norm = returned_norm
    optimizer.step()
    optimizer_step_count += 1
    _guard_counts(0, 0, optimizer_step_count)
    after_params = [param.detach() for param in policy.parameters() if param.requires_grad]
    delta_norm = float(
        torch.sqrt(
            sum(torch.sum((after - before) ** 2) for before, after in zip(before_params, after_params))
        ).detach().cpu()
    )
    if not math.isfinite(delta_norm) or delta_norm <= 0.0:
        raise RuntimeError("optimizer.step 后参数没有发生有限非零变化。")
    with torch.no_grad():
        post = ppo_update.compute_ppo_loss(
            policy,
            batch,
            float(cfg.clip_epsilon),
            float(cfg.value_loss_coefficient),
            float(cfg.entropy_coefficient),
        )
    return UpdateSanitySummary(
        policy_version_before=policy_version_before,
        policy_version_after=policy_version_before + 1,
        pre_ratio_mean=loss.ratio_mean,
        pre_ratio_min=loss.ratio_min,
        pre_ratio_max=loss.ratio_max,
        pre_clip_fraction=loss.clip_fraction,
        pre_approx_kl=loss.approx_kl,
        policy_loss=float(loss.policy_loss.detach().cpu()),
        value_loss=float(loss.value_loss.detach().cpu()),
        entropy=float(loss.entropy.detach().cpu()),
        total_loss=float(loss.total_loss.detach().cpu()),
        gradient_finite=gradient_finite,
        gradient_norm=pre_clip_norm,
        parameter_delta_norm=delta_norm,
        post_ratio_mean=post.ratio_mean,
        post_ratio_min=post.ratio_min,
        post_ratio_max=post.ratio_max,
        post_clip_fraction=post.clip_fraction,
        post_approx_kl=post.approx_kl,
        optimizer_step_count_after=optimizer_step_count,
    )


def _normalized_xy(xy: Any, boundary: tuple[float, float, float, float]) -> np.ndarray:
    x_min, y_min, x_max, y_max = boundary
    array = np.asarray(xy, dtype=np.float64)
    return np.asarray(
        [
            (float(array[0]) - x_min) / (x_max - x_min),
            (float(array[1]) - y_min) / (y_max - y_min),
        ],
        dtype=np.float64,
    )


def _run_once(config_path: str) -> BoundedTrainingSanityResult:
    _require_torch()
    ctx = _build_context(config_path)
    protocol = build_formal_training_protocol(ctx.scenarios, ctx.cfg)
    protocol_checks = validate_formal_training_protocol(protocol)
    training_cfg = ppo_update.ppo_training_config_status(ctx.cfg)
    if not training_cfg.can_run_optimizer_step:
        raise RuntimeError("PPO update 超参数缺失，不能执行 bounded sanity。")
    seed = int(project_config.get_config_value(ctx.cfg, "experiment.seed"))
    ppo_trigger_policy.set_torch_seed(seed)
    np.random.seed(seed)
    policy = ppo_trigger_policy.ConditionalTriggerActorCritic(
        ctx.observation_spec.observation_dim,
        ctx.action_spec,
        **ppo_trigger_policy.ppo_network_config_from_project(ctx.cfg),
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(training_cfg.learning_rate))
    optimizer_created_once = True
    optimizer_only_policy_parameters = sum(1 for _ in optimizer.param_groups[0]["params"]) == sum(1 for _ in policy.parameters())
    policy_version = 0
    rollout_count = 0
    environment_step_count = 0
    optimizer_step_count = 0

    if rollout_count >= SANITY_MAX_ROLLOUTS:
        raise RuntimeError("rollout 硬上限已达。")
    batch1, rollout1 = _collect_rollout(
        policy,
        ctx,
        int(protocol.episode_origin_indices[0]),
        ctx.initial_cache,
        ctx.initial_uav,
        policy_version,
        1,
    )
    rollout_count += 1
    environment_step_count += rollout1.step_count
    _guard_counts(rollout_count, environment_step_count, optimizer_step_count)
    update1 = _single_update(policy, optimizer, batch1, training_cfg, policy_version, optimizer_step_count)
    optimizer_step_count = update1.optimizer_step_count_after
    policy_version = update1.policy_version_after

    rollout2 = None
    update2 = None
    cache_continuity = None
    uav_continuity = None
    if not rollout1.true_terminal:
        if rollout_count >= SANITY_MAX_ROLLOUTS:
            raise RuntimeError("rollout 硬上限已达。")
        next_origin = int(rollout1.origin_indices[-1]) + 1
        batch2, rollout2 = _collect_rollout(
            policy,
            ctx,
            next_origin,
            rollout1.final_cache_state,
            rollout1.final_uav_positions,
            policy_version,
            2,
        )
        rollout_count += 1
        environment_step_count += rollout2.step_count
        _guard_counts(rollout_count, environment_step_count, optimizer_step_count)
        first_obs = batch2.observations[0]
        cache_features = first_obs[ctx.observation_spec.cache_slice]
        cache_continuity = bool(
            np.array_equal(
                cache_features.astype(np.int8),
                rollout1.final_cache_state.reshape(-1).astype(np.int8),
            )
        )
        uav_features = first_obs[ctx.observation_spec.uav_position_slice].astype(np.float64)
        expected_uav = np.concatenate(
            [
                _normalized_xy(rollout1.final_uav_positions[uav_id], ctx.flight_boundary)
                for uav_id in ctx.action_spec.uav_ids
            ]
        )
        uav_continuity = bool(np.allclose(uav_features, expected_uav, atol=1e-6, rtol=0.0))
        update2 = _single_update(policy, optimizer, batch2, training_cfg, policy_version, optimizer_step_count)
        optimizer_step_count = update2.optimizer_step_count_after
        policy_version = update2.policy_version_after

    origins_ok = True
    if rollout2 is not None:
        origins_ok = (
            max(rollout1.origin_indices) + 1 == min(rollout2.origin_indices)
            and set(rollout1.origin_indices).isdisjoint(set(rollout2.origin_indices))
        )
    _guard_counts(rollout_count, environment_step_count, optimizer_step_count)
    return BoundedTrainingSanityResult(
        rollout_1=rollout1,
        update_1=update1,
        rollout_2=rollout2,
        update_2=update2,
        final_policy_version=policy_version,
        rollout_count=rollout_count,
        environment_step_count=environment_step_count,
        optimizer_step_count=optimizer_step_count,
        optimizer_created_once=optimizer_created_once,
        optimizer_type=type(optimizer).__name__,
        optimizer_only_policy_parameters=optimizer_only_policy_parameters,
        rollout_origins_continuous_non_overlapping=origins_ok,
        rollout_2_first_cache_matches_rollout_1_final=cache_continuity,
        rollout_2_first_uav_matches_rollout_1_final=uav_continuity,
        reward_is_system_utility=True,
        repeated_bounded_sanity_deterministic=None,
        formal_validation=bool(all(protocol_checks.values())),
        blocker="",
    )


def _fingerprint(result: BoundedTrainingSanityResult) -> dict[str, Any]:
    return {
        "r1_actions": result.rollout_1.actions,
        "r1_rewards": result.rollout_1.rewards,
        "u1_delta": result.update_1.parameter_delta_norm,
        "r2_actions": None if result.rollout_2 is None else result.rollout_2.actions,
        "r2_rewards": None if result.rollout_2 is None else result.rollout_2.rewards,
        "u2_delta": None if result.update_2 is None else result.update_2.parameter_delta_norm,
    }


def run_bounded_training_sanity(config_path: str = "system_config.toml", repeat_check: bool = True) -> BoundedTrainingSanityResult:
    """仅用于最多2个rollout、2次update的训练链路集成验证，不是正式训练入口。"""

    first = _run_once(config_path)
    if repeat_check:
        second = _run_once(config_path)
        deterministic = _fingerprint(first) == _fingerprint(second)
        return BoundedTrainingSanityResult(
            **{**first.__dict__, "repeated_bounded_sanity_deterministic": deterministic}
        )
    return first


def _update_summary(update: UpdateSanitySummary | None) -> dict[str, Any] | None:
    if update is None:
        return None
    return {
        "policy_version_before": update.policy_version_before,
        "policy_version_after": update.policy_version_after,
        "pre_ratio": [update.pre_ratio_mean, update.pre_ratio_min, update.pre_ratio_max],
        "pre_clip_fraction": update.pre_clip_fraction,
        "pre_approx_kl": update.pre_approx_kl,
        "policy_loss": update.policy_loss,
        "value_loss": update.value_loss,
        "entropy": update.entropy,
        "total_loss": update.total_loss,
        "gradient_finite": update.gradient_finite,
        "gradient_norm": update.gradient_norm,
        "parameter_delta_norm": update.parameter_delta_norm,
        "post_ratio": [update.post_ratio_mean, update.post_ratio_min, update.post_ratio_max],
        "post_clip_fraction": update.post_clip_fraction,
        "post_approx_kl": update.post_approx_kl,
        "optimizer_step_count_after": update.optimizer_step_count_after,
    }


def _rollout_summary(rollout: RolloutSanitySummary | None) -> dict[str, Any] | None:
    if rollout is None:
        return None
    return {
        "rollout_index": rollout.rollout_index,
        "policy_version": rollout.collected_with_policy_version,
        "origin_range": [min(rollout.origin_indices), max(rollout.origin_indices)],
        "slot_range": [min(rollout.slots), max(rollout.slots)],
        "step_count": rollout.step_count,
        "actions": rollout.actions,
        "rewards": rollout.rewards,
        "reward_sum": rollout.reward_sum,
        "reward_mean": rollout.reward_mean,
        "actual_q_sum": rollout.actual_q_sum,
        "cache_cost_sum": rollout.cache_cost_sum,
        "movement_cost_sum": rollout.movement_cost_sum,
        "truncated": rollout.truncated,
        "true_terminal": rollout.true_terminal,
    }


def _summary(result: BoundedTrainingSanityResult) -> dict[str, Any]:
    return {
        "limits": {
            "MAX_SANITY_ROLLOUTS": SANITY_MAX_ROLLOUTS,
            "MAX_STEPS_PER_ROLLOUT": SANITY_MAX_STEPS_PER_ROLLOUT,
            "MAX_TOTAL_ENV_STEPS": SANITY_MAX_TOTAL_ENV_STEPS,
            "MAX_OPTIMIZER_STEPS": SANITY_MAX_OPTIMIZER_STEPS,
        },
        "optimizer_created_once": result.optimizer_created_once,
        "optimizer_type": result.optimizer_type,
        "optimizer_only_policy_parameters": result.optimizer_only_policy_parameters,
        "rollout_1": _rollout_summary(result.rollout_1),
        "update_1": _update_summary(result.update_1),
        "rollout_2": _rollout_summary(result.rollout_2),
        "update_2": _update_summary(result.update_2),
        "final_policy_version": result.final_policy_version,
        "rollout_count": result.rollout_count,
        "environment_step_count": result.environment_step_count,
        "optimizer_step_count": result.optimizer_step_count,
        "rollout_origins_continuous_non_overlapping": result.rollout_origins_continuous_non_overlapping,
        "rollout_2_first_cache_matches_rollout_1_final": result.rollout_2_first_cache_matches_rollout_1_final,
        "rollout_2_first_uav_matches_rollout_1_final": result.rollout_2_first_uav_matches_rollout_1_final,
        "reward_is_system_utility": result.reward_is_system_utility,
        "repeated_bounded_sanity_deterministic": result.repeated_bounded_sanity_deterministic,
        "formal_validation": result.formal_validation,
        "blocker": result.blocker,
    }


def _dry_run_summary(result: FormalTrainingDryRunResult) -> dict[str, Any]:
    protocol = result.protocol
    algorithm_reporting = result.algorithm_reporting or {}
    formal_training_signature = inspect.signature(run_formal_ppo_training)
    num_episodes_parameter = formal_training_signature.parameters["num_episodes"]
    return {
        "environment_steps_executed": result.environment_steps_executed,
        "backward_calls_executed": result.backward_calls_executed,
        "optimizer_steps_executed": result.optimizer_steps_executed,
        "formal_rollout_executor_available": result.formal_rollout_executor_available,
        "ppo_update_executor_available": result.ppo_update_executor_available,
        "formal_training_api_available": result.formal_training_api_available,
        "held_out_test_evaluation_api_available": result.held_out_test_evaluation_api_available,
        "run_formal_ppo_training_num_episodes_required": (
            num_episodes_parameter.default is inspect.Parameter.empty
        ),
        "run_formal_ppo_training_execute_default": formal_training_signature.parameters["execute"].default,
        "sample_num_episodes_for_dry_run_only": result.sample_num_episodes,
        "planned_environment_steps": result.planned_environment_steps,
        "planned_optimizer_steps": result.planned_optimizer_steps,
        "planned_backward_calls": result.planned_backward_calls,
        "planned_total_optimizer_steps": result.planned_total_optimizer_steps,
        "planned_total_backward_calls": result.planned_total_backward_calls,
        "optimizer_plan_semantics": (
            "planned_optimizer_steps/planned_backward_calls are Actor-side counts; "
            "component and total counts are reported separately"
        ),
        "optimizer_instances_planned": result.optimizer_instances_planned,
        "credit_assignment": result.credit_assignment,
        "credit_phase": result.credit_phase,
        "planned_q_optimizer_steps": result.planned_q_optimizer_steps,
        "q_optimizer_instances_planned": result.q_optimizer_instances_planned,
        "planned_actor_optimizer_steps": result.planned_actor_optimizer_steps,
        "actor_optimizer_instances_planned": result.actor_optimizer_instances_planned,
        "critic_optimizer_instances_planned": result.critic_optimizer_instances_planned,
        "cf_samples_per_episode": result.cf_samples_per_episode,
        "cf_batch_episodes": result.cf_batch_episodes,
        "cf_sampling_strategy": result.cf_sampling_strategy,
        "planned_counterfactual_branch_steps": result.planned_counterfactual_branch_steps,
        "gate_cf_samples_per_episode": result.gate_cf_samples_per_episode,
        "planned_gate_counterfactual_branch_steps": result.planned_gate_counterfactual_branch_steps,
        "planned_gate_optimizer_steps": result.planned_gate_optimizer_steps,
        "gate_optimizer_instances_planned": result.gate_optimizer_instances_planned,
        "method_id": algorithm_reporting.get("method_id"),
        "architecture_id": algorithm_reporting.get("architecture_id"),
        "actor_architecture_id": algorithm_reporting.get("actor_architecture_id"),
        "actor_optimizer_mode": algorithm_reporting.get("actor_optimizer_mode"),
        "actor_credit_semantics": algorithm_reporting.get("actor_credit_semantics"),
        "gate_optimizer_mode": algorithm_reporting.get("gate_optimizer_mode"),
        "gate_credit_semantics": algorithm_reporting.get("gate_credit_semantics"),
        "what_credit_semantics": algorithm_reporting.get("what_credit_semantics"),
        "learned_critic_for_actor_credit": algorithm_reporting.get(
            "learned_critic_for_actor_credit"
        ),
        "value_loss_in_actor_update": algorithm_reporting.get(
            "value_loss_in_actor_update"
        ),
        "direct_cf_minibatches_per_epoch": algorithm_reporting.get(
            "direct_cf_minibatches_per_epoch"
        ),
        "direct_cf_expected_optimizer_steps_per_episode": (
            int(protocol.update_epochs)
            if (
                result.credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT
                and result.cf_batch_episodes == 1
            )
            else (
                0
                if result.credit_assignment == ppo_update.SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT
                else None
            )
        ),
        "direct_cf_expected_optimizer_steps_per_batch": algorithm_reporting.get(
            "direct_cf_expected_optimizer_steps_per_batch"
        ),
        "direct_cf_uses_scalar_gae_advantage": algorithm_reporting.get(
            "direct_cf_uses_scalar_gae_advantage"
        ),
        "executor_dry_run_only": result.executor_dry_run_only,
        "data_protocol": {
            "schema": "ppo_train_test_origin_split_v1",
            "train_origin_range": [
                int(protocol.origin_split.train_origin_start),
                int(protocol.origin_split.train_origin_end),
            ],
            "test_origin_range": [
                int(protocol.origin_split.test_origin_start),
                int(protocol.origin_split.test_origin_end),
            ],
            "train_steps_per_episode": int(protocol.steps_per_episode),
            "test_steps_per_evaluation": int(protocol.test_steps_per_evaluation),
            "validation_origin_range": None,
        },
        "episode_origin_count": int(protocol.steps_per_episode),
        "first_origin": int(protocol.episode_origin_indices[0]),
        "last_origin": int(protocol.episode_origin_indices[-1]),
        "first_slot": int(protocol.episode_slots[0]),
        "last_slot": int(protocol.episode_slots[-1]),
        "first_time_s": float(protocol.episode_times_s[0]),
        "last_time_s": float(protocol.episode_times_s[-1]),
        "steps_per_episode": int(protocol.steps_per_episode),
        "rollout_definition": "one complete environment episode",
        "update_epochs": int(protocol.update_epochs),
        "num_minibatches": int(protocol.num_minibatches),
        "minibatch_size": int(protocol.minibatch_size),
        "expected_optimizer_steps_per_rollout": int(protocol.expected_optimizer_steps_per_rollout),
        "true_terminal_origin": int(protocol.true_terminal_origin),
        "true_terminal_slot": int(protocol.true_terminal_slot),
        "true_terminal_time_s": float(protocol.true_terminal_time_s),
        "test_true_terminal_origin": int(protocol.test_true_terminal_origin),
        "test_true_terminal_slot": int(protocol.test_true_terminal_slot),
        "test_true_terminal_time_s": float(protocol.test_true_terminal_time_s),
        "single_trace_training": bool(protocol.single_trace_training),
        "initial_cache_source": protocol.initial_cache_source,
        "initial_uav_source": protocol.initial_uav_source,
        "tv_local_cache_source": protocol.tv_local_cache_source,
        "evaluation_semantics": protocol.evaluation_semantics,
        "origins_contiguous": result.origins_contiguous,
        "slots_strictly_increasing": result.slots_strictly_increasing,
        "times_strictly_increasing": result.times_strictly_increasing,
        "minibatch_divisible": result.minibatch_divisible,
        "reset_state_resolved": result.reset_state_resolved,
        "formal_validation": result.formal_validation,
        "blocker": result.blocker,
    }


def _positive_episode_count(value: str) -> int:
    """解析正式训练 episode 数，拒绝非正整数。"""

    try:
        episodes = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--episodes 必须是整数。") from exc
    if episodes <= 0:
        raise argparse.ArgumentTypeError("--episodes 必须大于 0。")
    return episodes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PPO training protocol and explicit formal training entrypoint.")
    parser.add_argument("--config", default="system_config.toml")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--bounded-sanity",
        action="store_true",
        help="Run the existing 2-rollout/2-update development sanity instead of dry-run.",
    )
    mode_group.add_argument(
        "--formal-train",
        action="store_true",
        help="Execute formal PPO training; requires --episodes and --run-dir.",
    )
    mode_group.add_argument(
        "--held-out-test",
        "--test-eval",
        dest="held_out_test",
        action="store_true",
        help="Run read-only held-out test on --checkpoint using origins from the configured test range.",
    )
    mode_group.add_argument(
        "--protocol-check",
        action="store_true",
        help="Validate the configured train/test protocol without training or environment execution.",
    )
    parser.add_argument(
        "--episodes",
        type=_positive_episode_count,
        help="Target total number of completed formal training episodes.",
    )
    parser.add_argument(
        "--run-dir",
        help="Explicit directory for formal training checkpoint and history files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume formal training from --run-dir; --episodes is the target total episode count.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Current-compatible policy checkpoint for --held-out-test.",
    )
    parser.add_argument(
        "--test-report",
        help="Optional JSON output path for --held-out-test; no checkpoint/history is written.",
    )
    parser.add_argument("--skip-repeat-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.held_out_test:
        if not args.checkpoint:
            parser.error("--held-out-test 必须同时提供 --checkpoint PATH。")
        if args.episodes is not None or args.run_dir or args.resume or args.skip_repeat_check:
            parser.error("--held-out-test 只接受 --config、--checkpoint 和可选 --test-report。")
        summary = run_held_out_test_evaluation(
            args.checkpoint,
            config_path=args.config,
            report_path=args.test_report,
        )
        output = summary_to_builtin(summary)
        if args.test_report:
            output["report_path"] = str(Path(args.test_report).resolve())
        print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
        return 0

    if args.checkpoint or args.test_report:
        parser.error("--checkpoint/--test-report 仅适用于 --held-out-test。")
    if args.formal_train:
        if args.episodes is None:
            parser.error("--formal-train 必须同时提供 --episodes N，且 N > 0。")
        if not args.run_dir:
            parser.error("--formal-train 必须同时提供 --run-dir PATH。")
        if args.skip_repeat_check:
            parser.error("--skip-repeat-check 仅适用于 --bounded-sanity。")
        result = run_formal_ppo_training(
            num_episodes=args.episodes,
            config_path=args.config,
            execute=True,
            run_dir=args.run_dir,
            resume=args.resume,
        )
        print(format_training_summary(result))
        return 0

    if args.resume:
        parser.error("--resume 仅适用于 --formal-train。")
    if args.episodes is not None:
        parser.error("--episodes 仅适用于 --formal-train。")
    if args.run_dir:
        parser.error("--run-dir 仅适用于 --formal-train。")
    if args.protocol_check:
        result = run_training_dry_run(args.config)
        print(json.dumps(_dry_run_summary(result), ensure_ascii=False, indent=2))
        return 0
    if args.bounded_sanity:
        result = run_bounded_training_sanity(args.config, repeat_check=not args.skip_repeat_check)
        print(json.dumps(_summary(result), ensure_ascii=False, indent=2))
    else:
        result = run_training_dry_run(args.config)
        print(json.dumps(_dry_run_summary(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
