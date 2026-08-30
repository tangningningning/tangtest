#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PPO clipped loss for trigger-only Actor-Critic.

本模块只实现一次 PPO clipped update 的数值 sanity 所需组件：

    rollout batch -> evaluate_actions -> joint-ratio clipped loss diagnostics

若 system_config.toml 尚未正式定义 learning_rate / clip_epsilon 等训练超参，
本模块不会擅自写入默认值，也不会执行 optimizer.step()。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common import project_config
from decision import cache_configuration, uav_configuration
from ppo import ppo_rollout, ppo_trigger_policy
from scenario import task_scenario

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PPOTrainingConfigStatus:
    clip_epsilon: float | None
    learning_rate: float | None
    value_loss_coefficient: float | None
    entropy_coefficient: float | None
    max_grad_norm: float | None
    missing_required_for_step: tuple[str, ...]

    @property
    def can_run_optimizer_step(self) -> bool:
        return len(self.missing_required_for_step) == 0


@dataclass(frozen=True)
class PPOLossResult:
    total_loss: Any
    policy_loss: Any
    value_loss: Any
    entropy: Any
    ratio_mean: float
    ratio_min: float
    ratio_max: float
    clip_fraction: float
    approx_kl: float
    new_log_probs: Any
    new_values: Any


@dataclass(frozen=True)
class QLossResult:
    """一次独立 Q critic loss 的张量和轻量统计。"""

    q_loss: Any
    q_values: Any
    chosen_q_values: Any
    free_count: int


@dataclass(frozen=True)
class CounterfactualTargets:
    """在 Q optimizer 更新前冻结的 per-trigger credit target。"""

    conditional_hidden_states: Any
    q_values: Any
    baseline: Any
    advantages: Any
    actor_valid_mask: Any


@dataclass(frozen=True)
class CentralizedCounterfactualTargets:
    """集中式 joint-action Q 生成的 per-trigger frozen actor target。"""

    chosen_q_values: Any
    q_action_zero: Any
    q_action_one: Any
    baseline: Any
    advantages: Any
    action_gaps: Any
    actor_valid_mask: Any


@dataclass(frozen=True)
class CentralizedQLossResult:
    """central critic 对真实 sampled joint action 的 MC 回归结果。"""

    q_loss: Any
    q_values: Any


@dataclass(frozen=True)
class SingleUpdateSanityResult:
    training_config: PPOTrainingConfigStatus
    loss_result: PPOLossResult | None
    artificial_clipping_sanity_error: float
    pre_update_ratio_mean: float
    pre_update_ratio_min: float
    pre_update_ratio_max: float
    pre_update_clip_fraction: float | None
    pre_update_approx_kl: float
    optimizer_step_count: int
    gradient_finite: bool | None
    gradient_norm: float | None
    parameter_delta_norm: float | None
    post_update_ratio_mean: float | None
    post_update_ratio_min: float | None
    post_update_ratio_max: float | None
    post_update_approx_kl: float | None
    post_update_clip_fraction: float | None
    old_log_probs_unchanged: bool
    old_values_unchanged: bool
    advantages_unchanged: bool
    returns_unchanged: bool
    formal_validation: bool
    blocker: str


def _require_torch() -> None:
    if torch is None:
        raise ImportError("ppo_update.py 需要 PyTorch 才能计算 PPO loss。")


def _validate_centralized_return_scale(return_scale: float) -> float:
    """校验 centralized critic 使用的固定正比例 return scale。"""

    scale = float(return_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("centralized critic return_scale 必须是有限正数。")
    return scale


def ppo_training_config_status(project_cfg: dict[str, Any]) -> PPOTrainingConfigStatus:
    """审计正式 PPO update 超参；缺失时不填默认值。"""

    get = project_config.get_config_value
    clip = get(project_cfg, "algorithm.ppo.clip_epsilon")
    lr = get(project_cfg, "algorithm.ppo.learning_rate")
    value_c = get(project_cfg, "algorithm.ppo.value_loss_coefficient")
    entropy_c = get(project_cfg, "algorithm.ppo.entropy_coefficient")
    max_grad = get(project_cfg, "algorithm.ppo.max_grad_norm")
    missing = []
    for key, value in (
        ("algorithm.ppo.clip_epsilon", clip),
        ("algorithm.ppo.learning_rate", lr),
        ("algorithm.ppo.value_loss_coefficient", value_c),
        ("algorithm.ppo.entropy_coefficient", entropy_c),
    ):
        if value is None:
            missing.append(key)
    return PPOTrainingConfigStatus(
        clip_epsilon=None if clip is None else float(clip),
        learning_rate=None if lr is None else float(lr),
        value_loss_coefficient=None if value_c is None else float(value_c),
        entropy_coefficient=None if entropy_c is None else float(entropy_c),
        max_grad_norm=None if max_grad is None else float(max_grad),
        missing_required_for_step=tuple(missing),
    )


LEGACY_CREDIT_ASSIGNMENT = "legacy_scalar"
PER_TRIGGER_Q_CREDIT_ASSIGNMENT = "per_trigger_q"
CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT = "centralized_counterfactual_q"
SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT = "sparse_direct_cf"
FROZEN_POLICY_BATCH_METHOD = "proposed_sparse_direct_cf_frozen_policy_batch"
REFERENCE_DRIFT_FROZEN_POLICY_BATCH_METHOD = (
    "proposed_sparse_direct_cf_frozen_policy_batch_refdrift"
)
REFERENCE_DRIFT_ALGORITHM_SCHEMA = (
    "proposed_sparse_direct_cf_frozen_policy_batch_refdrift_v1"
)
HIERARCHICAL_GATE_METHOD = "proposed_hierarchical_reconfiguration_gate_direct_cf_refdrift"
HIERARCHICAL_GATE_ALGORITHM_SCHEMA = (
    "proposed_hierarchical_reconfiguration_gate_direct_cf_refdrift_v1"
)
HIERARCHICAL_GATE_CREDIT_ASSIGNMENT = "hierarchical_gate_direct_cf"
MONTE_CARLO_Q_TARGET = "monte_carlo"


def credit_assignment_from_project(project_cfg: dict[str, Any]) -> str:
    """读取 PPO credit-assignment mode；旧配置缺失时保持 legacy 语义。"""

    explicit = project_config.get_config_value(project_cfg, "algorithm.ppo.credit_assignment")
    if explicit is None:
        method = project_config.get_config_value(
            project_cfg,
            "algorithm.ppo.method",
            "trigger_actor_critic",
        )
        if method == HIERARCHICAL_GATE_METHOD:
            return HIERARCHICAL_GATE_CREDIT_ASSIGNMENT
        return (
            PER_TRIGGER_Q_CREDIT_ASSIGNMENT
            if method == "proposed_per_trigger_q"
            else (
                CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT
                if method == "proposed_centralized_cf_q"
                else (
                    SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT
                    if method in {
                        "proposed_sparse_direct_cf_ppo",
                        FROZEN_POLICY_BATCH_METHOD,
                        REFERENCE_DRIFT_FROZEN_POLICY_BATCH_METHOD,
                    }
                    else LEGACY_CREDIT_ASSIGNMENT
                )
            )
        )
    mode = str(explicit)
    if mode not in {
        LEGACY_CREDIT_ASSIGNMENT,
        PER_TRIGGER_Q_CREDIT_ASSIGNMENT,
        CENTRALIZED_COUNTERFACTUAL_CREDIT_ASSIGNMENT,
        SPARSE_DIRECT_CF_CREDIT_ASSIGNMENT,
        HIERARCHICAL_GATE_CREDIT_ASSIGNMENT,
    }:
        raise ValueError(
            "algorithm.ppo.credit_assignment 必须是 legacy_scalar、per_trigger_q "
            "centralized_counterfactual_q、sparse_direct_cf 或 "
            "hierarchical_gate_direct_cf。"
        )
    return mode


def credit_phase_from_project(project_cfg: dict[str, Any]) -> str:
    """读取 V2 的显式训练阶段；不自动根据episode数切换。"""

    phase = str(
        project_config.get_config_value(
            project_cfg,
            "algorithm.ppo.credit_phase",
            "critic_warmup",
        )
    )
    if phase not in {"critic_warmup", "joint_training"}:
        raise ValueError(
            "algorithm.ppo.credit_phase 必须是 critic_warmup 或 joint_training。"
        )
    return phase


def q_target_from_project(project_cfg: dict[str, Any]) -> str | None:
    """读取 Q target；legacy scalar 模式不要求该字段。"""

    target = project_config.get_config_value(project_cfg, "algorithm.ppo.q_target")
    return None if target is None else str(target)


def clipped_surrogate(
    ratios: Any,
    advantages: Any,
    clip_epsilon: float,
) -> Any:
    """PPO clipped surrogate：mean(min(r*A, clip(r)*A))。"""

    _require_torch()
    ratio_t = torch.as_tensor(ratios, dtype=torch.float32)
    advantage_t = torch.as_tensor(advantages, dtype=torch.float32)
    clipped_ratio = torch.clamp(ratio_t, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    return torch.minimum(ratio_t * advantage_t, clipped_ratio * advantage_t).mean()


def _policy_device(policy: Any) -> Any:
    _require_torch()
    try:
        return next(policy.parameters()).device
    except StopIteration as exc:
        raise ValueError("policy 必须包含至少一个参数。") from exc


def _free_trigger_mask(rollout_batch: ppo_rollout.RolloutBatch, device: Any) -> Any:
    mask = torch.as_tensor(rollout_batch.actor_valid_mask, dtype=torch.bool, device=device)
    if mask.ndim != 2 or mask.shape != rollout_batch.actions.shape:
        raise ValueError(
            "actor_valid_mask shape="
            f"{tuple(mask.shape)}，期望 {tuple(rollout_batch.actions.shape)}。"
        )
    return mask


def prepare_counterfactual_targets(
    policy: Any,
    q_critic: Any,
    rollout_batch: ppo_rollout.RolloutBatch,
) -> CounterfactualTargets:
    """在 Q update 前计算并冻结 Q_old、baseline_cf 和 A_cf。

    Q critic 读取 ``h_i.detach()``；这里使用 no_grad 生成固定 target，整个
    后续 PPO multi-epoch 期间不得重新计算。Q target 本身是 sampled-action
    Monte-Carlo return，不是对未采样 action 的真实环境 counterfactual replay。
    """

    _require_torch()
    device = _policy_device(policy)
    observations = torch.as_tensor(rollout_batch.observations, dtype=torch.float32, device=device)
    actions = torch.as_tensor(rollout_batch.actions, dtype=torch.float32, device=device)
    old_probabilities = torch.as_tensor(
        rollout_batch.old_per_action_probabilities,
        dtype=torch.float32,
        device=device,
    ).detach()
    if old_probabilities.ndim != 3 or old_probabilities.shape[-1] != 2:
        raise ValueError("old_per_action_probabilities 必须是 [T,action_dim,2]。")
    if not torch.all(torch.isfinite(old_probabilities)):
        raise ValueError("old_per_action_probabilities 包含 NaN/Inf。")
    if not bool(torch.allclose(old_probabilities.sum(dim=-1), torch.ones_like(old_probabilities[..., 0]), atol=1e-5, rtol=1e-5)):
        raise ValueError("old_per_action_probabilities 最后一维必须归一化。")
    actor_valid_mask = _free_trigger_mask(rollout_batch, device)
    with torch.no_grad():
        conditional_hidden_states = policy.conditional_hidden_states(observations, actions).detach()
        q_values = q_critic(conditional_hidden_states).detach()
        baseline = torch.sum(old_probabilities * q_values, dim=-1)
        chosen_q_values = q_values.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
        advantages = torch.where(
            actor_valid_mask,
            chosen_q_values - baseline,
            torch.zeros_like(baseline),
        )
    return CounterfactualTargets(
        conditional_hidden_states=conditional_hidden_states,
        q_values=q_values,
        baseline=baseline,
        advantages=advantages.detach(),
        actor_valid_mask=actor_valid_mask.detach(),
    )


def compute_per_trigger_q_loss(
    q_critic: Any,
    conditional_hidden_states: Any,
    actions: Any,
    monte_carlo_returns: Any,
    actor_valid_mask: Any,
) -> QLossResult:
    """计算独立 Q critic loss；只对 FREE sampled decisions 做监督。"""

    _require_torch()
    try:
        device = next(q_critic.parameters()).device
    except StopIteration as exc:
        raise ValueError("q_critic 必须包含至少一个参数。") from exc
    hidden = torch.as_tensor(
        conditional_hidden_states,
        dtype=torch.float32,
        device=device,
    ).detach()
    action_t = torch.as_tensor(actions, dtype=torch.long, device=device)
    target_t = torch.as_tensor(monte_carlo_returns, dtype=torch.float32, device=device).reshape(-1)
    free_t = torch.as_tensor(actor_valid_mask, dtype=torch.bool, device=device)
    if hidden.ndim != 3 or action_t.ndim != 2 or free_t.shape != action_t.shape:
        raise ValueError("Q loss 的 hidden/actions/actor_valid_mask shape 不一致。")
    if target_t.shape != (action_t.shape[0],):
        raise ValueError("monte_carlo_returns shape 必须是 [T]。")
    q_values = q_critic(hidden)
    chosen_q_values = q_values.gather(-1, action_t.unsqueeze(-1)).squeeze(-1)
    free_count = int(free_t.sum().detach().cpu())
    if free_count > 0:
        q_loss = 0.5 * torch.mean(
            (chosen_q_values[free_t] - target_t.unsqueeze(1).expand_as(chosen_q_values)[free_t]) ** 2
        )
    else:
        # 保持 update 调度结构完整；该 batch 没有 free decision 时不产生有效 Q 梯度。
        q_loss = q_values.sum() * 0.0
    return QLossResult(
        q_loss=q_loss,
        q_values=q_values,
        chosen_q_values=chosen_q_values,
        free_count=free_count,
    )


def compute_centralized_q_loss(
    q_critic: Any,
    observations: Any,
    joint_actions: Any,
    monte_carlo_returns: Any,
    return_scale: float = 1.0,
) -> CentralizedQLossResult:
    """对真实 sampled joint action 做固定尺度 MC 回归。

    与 V1 不同，这里不使用 actor free mask：即使某个 trigger 被环境可行性
    mask 强制，整个 rollout transition 仍然是 central critic 的监督样本。
    central V2 的输出和 target 都使用 ``MC / return_scale``；默认值1保留
    旧调用的显式 raw-unit 语义，正式 V2 caller 必须传入formal horizon scale。
    """

    _require_torch()
    scale = _validate_centralized_return_scale(return_scale)
    try:
        device = next(q_critic.parameters()).device
    except StopIteration as exc:
        raise ValueError("q_critic 必须包含至少一个参数。") from exc
    observations_t = torch.as_tensor(observations, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(joint_actions, dtype=torch.float32, device=device)
    target_t = torch.as_tensor(monte_carlo_returns, dtype=torch.float32, device=device).reshape(-1)
    scaled_target_t = target_t / scale
    q_values = q_critic(observations_t, actions_t)
    if q_values.ndim != 1 or scaled_target_t.shape != q_values.shape:
        raise ValueError("central critic Q 与 MC target 必须都是 [T]。")
    if not torch.all(torch.isfinite(scaled_target_t)):
        raise ValueError("central critic MC target 包含 NaN/Inf。")
    q_loss = 0.5 * torch.mean((q_values - scaled_target_t) ** 2)
    return CentralizedQLossResult(q_loss=q_loss, q_values=q_values)


def prepare_centralized_counterfactual_targets(
    q_critic: Any,
    rollout_batch: ppo_rollout.RolloutBatch,
    return_scale: float = 1.0,
) -> CentralizedCounterfactualTargets:
    """固定其他7个实际 action，计算每个 trigger 的 two-action Q baseline。

    ``old_per_action_probabilities`` 来自 rollout 时冻结的旧 policy；因此该
    函数生成的 A_cf 可以在 Actor multi-epoch 更新期间 detach/freeze，而不会
    把新的 Actor probability 或 critic 梯度混入 surrogate。
    """

    _require_torch()
    scale = _validate_centralized_return_scale(return_scale)
    try:
        device = next(q_critic.parameters()).device
    except StopIteration as exc:
        raise ValueError("q_critic 必须包含至少一个参数。") from exc
    observations = torch.as_tensor(rollout_batch.observations, dtype=torch.float32, device=device)
    actions = torch.as_tensor(rollout_batch.actions, dtype=torch.float32, device=device)
    old_probabilities = torch.as_tensor(
        rollout_batch.old_per_action_probabilities,
        dtype=torch.float32,
        device=device,
    ).detach()
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("centralized counterfactual observation/action 必须是二维。")
    if old_probabilities.shape != (actions.shape[0], actions.shape[1], 2):
        raise ValueError("old_per_action_probabilities 必须是 [T,action_dim,2]。")
    if not torch.allclose(
        old_probabilities.sum(dim=-1),
        torch.ones_like(old_probabilities[..., 0]),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("old_per_action_probabilities 最后一维必须归一化。")
    actor_valid_mask = _free_trigger_mask(rollout_batch, device)
    with torch.no_grad():
        chosen_q_values = q_critic(observations, actions).detach()
        q_action_zero: list[Any] = []
        q_action_one: list[Any] = []
        for action_index in range(actions.shape[1]):
            actions_zero = actions.clone()
            actions_one = actions.clone()
            actions_zero[:, action_index] = 0.0
            actions_one[:, action_index] = 1.0
            q_action_zero.append(q_critic(observations, actions_zero).detach())
            q_action_one.append(q_critic(observations, actions_one).detach())
        q0 = torch.stack(q_action_zero, dim=1)
        q1 = torch.stack(q_action_one, dim=1)
        p_one = old_probabilities[..., 1]
        baseline = (1.0 - p_one) * q0 + p_one * q1
        advantages_scaled = chosen_q_values.unsqueeze(1) - baseline
        advantages_scaled = torch.where(
            actor_valid_mask,
            advantages_scaled,
            torch.zeros_like(advantages_scaled),
        )
        # Actor 的正式 advantage 仍恢复为原 reward 单位；q/gap本身保持scaled单位。
        advantages = scale * advantages_scaled
        action_gaps = q1 - q0
    return CentralizedCounterfactualTargets(
        chosen_q_values=chosen_q_values,
        q_action_zero=q0,
        q_action_one=q1,
        baseline=baseline,
        advantages=advantages.detach(),
        action_gaps=action_gaps.detach(),
        actor_valid_mask=actor_valid_mask.detach(),
    )


def compute_per_trigger_ppo_loss(
    policy: Any,
    rollout_batch: ppo_rollout.RolloutBatch,
    counterfactual_advantages: Any,
    clip_epsilon: float,
    value_loss_coefficient: float,
    entropy_coefficient: float,
) -> PPOLossResult:
    """计算 per-trigger conditional PPO actor + scalar-V loss。

    counterfactual mode 下 ratio、approx KL、clip fraction 都只在 FREE
    conditional decisions 上统计；它们不再是 joint-action sample 的统计量。
    新 total loss 不包含 Q loss，Q critic 使用独立 optimizer 更新。
    """

    _require_torch()
    if clip_epsilon < 0.0:
        raise ValueError("clip_epsilon 不能为负。")
    device = _policy_device(policy)
    observations = torch.as_tensor(rollout_batch.observations, dtype=torch.float32, device=device)
    actions = torch.as_tensor(rollout_batch.actions, dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(
        rollout_batch.old_per_action_log_probs,
        dtype=torch.float32,
        device=device,
    ).detach()
    advantages = torch.as_tensor(
        counterfactual_advantages,
        dtype=torch.float32,
        device=device,
    ).detach()
    returns = torch.as_tensor(rollout_batch.returns, dtype=torch.float32, device=device).detach()
    free_mask = _free_trigger_mask(rollout_batch, device)
    if old_log_probs.shape != actions.shape or advantages.shape != actions.shape:
        raise ValueError("per-trigger old_log_probs/advantages shape 必须与 actions 一致。")
    evaluation = policy.evaluate_actions(
        observations,
        actions,
        action_masks=rollout_batch.action_masks,
    )
    new_log_probs = evaluation.per_action_log_probs
    if not torch.allclose(
        old_log_probs.sum(dim=1),
        torch.as_tensor(rollout_batch.old_log_probs, dtype=torch.float32, device=device),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("per-trigger old log-prob之和与joint old_log_prob不一致。")
    ratios = torch.exp(new_log_probs - old_log_probs)
    clipped_ratios = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(ratios * advantages, clipped_ratios * advantages)
    free_count = int(free_mask.sum().detach().cpu())
    if free_count > 0:
        policy_loss = -torch.sum(surrogate[free_mask]) / float(free_count)
        ratio_for_stats = ratios[free_mask]
        old_for_stats = old_log_probs[free_mask]
        new_for_stats = new_log_probs[free_mask]
        entropy_per_action = evaluation.per_action_entropy
        entropy = torch.sum(entropy_per_action[free_mask]) / float(free_count)
        clip_fraction = torch.mean(
            (torch.abs(ratio_for_stats - 1.0) > clip_epsilon).to(torch.float32)
        )
        approx_kl = torch.mean(old_for_stats - new_for_stats)
    else:
        policy_loss = torch.sum(new_log_probs) * 0.0
        entropy = torch.sum(evaluation.per_action_entropy) * 0.0
        ratio_for_stats = torch.ones((1,), dtype=torch.float32, device=device)
        clip_fraction = torch.zeros((), dtype=torch.float32, device=device)
        approx_kl = torch.zeros((), dtype=torch.float32, device=device)
    new_values = evaluation.value
    value_loss = 0.5 * torch.mean((new_values - returns) ** 2)
    total_loss = policy_loss + value_loss_coefficient * value_loss - entropy_coefficient * entropy
    with torch.no_grad():
        ratio_mean = torch.mean(ratio_for_stats)
        ratio_min = torch.min(ratio_for_stats)
        ratio_max = torch.max(ratio_for_stats)
    return PPOLossResult(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        ratio_mean=float(ratio_mean.detach().cpu()),
        ratio_min=float(ratio_min.detach().cpu()),
        ratio_max=float(ratio_max.detach().cpu()),
        clip_fraction=float(clip_fraction.detach().cpu()),
        approx_kl=float(approx_kl.detach().cpu()),
        new_log_probs=new_log_probs,
        new_values=new_values,
    )


def compute_sparse_direct_cf_actor_loss(
    policy: Any,
    observations: Any,
    actions: Any,
    action_masks: Any,
    target_indices: Any,
    old_target_log_probs: Any,
    direct_cf_advantages: Any,
    clip_epsilon: float,
    sample_weights: Any | None = None,
) -> PPOLossResult:
    """计算 V3 sparse Direct-CF 的 target-only clipped Actor loss。

    ``direct_cf_advantages`` 必须已经是 ``(a-p_old) * Delta_G``。本函数
    绝不再次乘 ``(a-p)``，也不计算 scalar value loss、GAE loss 或 entropy
    loss；因此 shared Actor encoder 只接收 Direct-CF gradient。
    """

    _require_torch()
    if clip_epsilon < 0.0:
        raise ValueError("clip_epsilon 不能为负。")
    device = _policy_device(policy)
    observations_t = torch.as_tensor(observations, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
    masks_t = torch.as_tensor(action_masks, dtype=torch.bool, device=device)
    target_t = torch.as_tensor(target_indices, dtype=torch.long, device=device).reshape(-1)
    old_log_t = torch.as_tensor(old_target_log_probs, dtype=torch.float32, device=device).reshape(-1).detach()
    advantage_t = torch.as_tensor(direct_cf_advantages, dtype=torch.float32, device=device).reshape(-1).detach()
    if observations_t.ndim != 2 or actions_t.ndim != 2:
        raise ValueError("V3 observations/actions 必须是二维。")
    sample_count = int(observations_t.shape[0])
    if sample_count <= 0 or actions_t.shape[0] != sample_count:
        raise ValueError("V3 Actor batch 不能为空且样本数必须一致。")
    if target_t.shape != (sample_count,) or old_log_t.shape != (sample_count,) or advantage_t.shape != (sample_count,):
        raise ValueError("V3 target_indices/log_probs/advantages shape 必须是 [K]。")
    if masks_t.shape != (sample_count, int(actions_t.shape[1]), 2):
        raise ValueError("V3 action_masks 必须是 [K,action_dim,2]。")
    if not bool(torch.all((target_t >= 0) & (target_t < actions_t.shape[1])).item()):
        raise ValueError("V3 target_indices 越界。")
    target_masks = masks_t[torch.arange(sample_count, device=device), target_t]
    if not bool(torch.all(target_masks).item()):
        raise ValueError("V3 Direct-CF target 必须是 FREE decision。")
    if not torch.all(torch.isfinite(advantage_t)):
        raise ValueError("V3 Direct-CF advantage 包含 NaN/Inf。")

    evaluation = policy.evaluate_actions(
        observations_t,
        actions_t,
        action_masks=masks_t,
    )
    row_indices = torch.arange(sample_count, device=device)
    new_target_log_probs = evaluation.per_action_log_probs[row_indices, target_t]
    ratios = torch.exp(new_target_log_probs - old_log_t)
    clipped_ratios = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(ratios * advantage_t, clipped_ratios * advantage_t)
    if sample_weights is None:
        # legacy uniform reservoir：保持原K样本 arithmetic mean语义。
        policy_loss = -torch.mean(surrogate)
    else:
        weights_t = torch.as_tensor(
            sample_weights,
            dtype=torch.float32,
            device=device,
        ).reshape(-1).detach()
        if weights_t.shape != (sample_count,):
            raise ValueError("V3 sample_weights shape 必须是 [K]。")
        if not torch.all(torch.isfinite(weights_t)) or not torch.all(weights_t >= 0.0):
            raise ValueError("V3 sample_weights 必须是finite非负数。")
        if not bool(torch.isclose(
            torch.sum(weights_t),
            torch.ones((), dtype=torch.float32, device=device),
            rtol=1e-5,
            atol=1e-6,
        ).item()):
            raise ValueError("V3 sample_weights之和必须为1。")
        policy_loss = -torch.sum(weights_t * surrogate)
    zero = policy_loss.detach() * 0.0
    with torch.no_grad():
        approx_kl = torch.mean(old_log_t - new_target_log_probs)
        clip_fraction = torch.mean(
            (torch.abs(ratios - 1.0) > clip_epsilon).to(torch.float32)
        )
        ratio_mean = torch.mean(ratios)
        ratio_min = torch.min(ratios)
        ratio_max = torch.max(ratios)
    return PPOLossResult(
        total_loss=policy_loss,
        policy_loss=policy_loss,
        value_loss=zero,
        entropy=zero,
        ratio_mean=float(ratio_mean.detach().cpu()),
        ratio_min=float(ratio_min.detach().cpu()),
        ratio_max=float(ratio_max.detach().cpu()),
        clip_fraction=float(clip_fraction.detach().cpu()),
        approx_kl=float(approx_kl.detach().cpu()),
        new_log_probs=new_target_log_probs,
        new_values=evaluation.value,
    )


def compute_gate_direct_cf_actor_loss(
    gate_policy: Any,
    observations: Any,
    gate_actions: Any,
    old_gate_log_probs: Any,
    old_p_gate: Any,
    delta_gate: Any,
    clip_epsilon: float,
    sample_weights: Any | None = None,
) -> PPOLossResult:
    """Compute the independent Gate clipped-PPO Direct-CF loss.

    The frozen target is exactly ``A_gate=(g-p_old_gate)*Delta_gate``.  This
    function deliberately does not multiply by ``(g-p)`` again: the PPO
    likelihood ratio is the only policy-gradient factor after the advantage
    has been formed.  No value or entropy term is added to the Gate loss.
    """

    _require_torch()
    if clip_epsilon < 0.0 or clip_epsilon >= 1.0:
        raise ValueError("Gate clip_epsilon 必须位于[0,1)。")
    observations_t = torch.as_tensor(observations, dtype=torch.float32)
    if observations_t.ndim != 2:
        raise ValueError("Gate observations 必须是二维 [K,D]。")
    actions_t = torch.as_tensor(gate_actions, dtype=torch.float32).reshape(-1)
    old_log_t = torch.as_tensor(old_gate_log_probs, dtype=torch.float32).reshape(-1).detach()
    old_p_t = torch.as_tensor(old_p_gate, dtype=torch.float32).reshape(-1).detach()
    delta_t = torch.as_tensor(delta_gate, dtype=torch.float32).reshape(-1).detach()
    sample_count = int(observations_t.shape[0])
    if sample_count <= 0 or any(
        tensor.shape != (sample_count,)
        for tensor in (actions_t, old_log_t, old_p_t, delta_t)
    ):
        raise ValueError("Gate observations/actions/old log-prob/p/Delta shape必须一致且非空。")
    if not bool(torch.all((actions_t == 0.0) | (actions_t == 1.0)).item()):
        raise ValueError("gate_actions 必须是0/1。")
    if not bool(torch.all(torch.isfinite(old_p_t)).item()) or not bool(
        torch.all((old_p_t > 0.0) & (old_p_t < 1.0)).item()
    ):
        raise ValueError("old_p_gate 必须是(0,1)内的finite概率。")
    if not bool(torch.all(torch.isfinite(old_log_t)).item()) or not bool(
        torch.all(torch.isfinite(delta_t)).item()
    ):
        raise ValueError("Gate old log-prob/Delta必须finite。")

    evaluator = getattr(gate_policy, "evaluate_gate_actions", None)
    if evaluator is None:
        evaluator = getattr(gate_policy, "evaluate_actions", None)
    if evaluator is None:
        raise TypeError("gate_policy 必须提供 evaluate_gate_actions/evaluate_actions。")
    evaluation = evaluator(observations_t, actions_t.long())
    new_log_probs = evaluation.chosen_log_prob.reshape(-1)
    if new_log_probs.shape != (sample_count,):
        raise ValueError("Gate evaluation chosen_log_prob shape非法。")
    expected_old_log_probs = torch.where(
        actions_t == 1.0,
        torch.log(old_p_t),
        torch.log1p(-old_p_t),
    )
    if not torch.allclose(old_log_t, expected_old_log_probs, rtol=1e-5, atol=1e-6):
        raise ValueError("Gate old log-prob 与 old_p_gate/gate_action 不一致。")
    advantages = (actions_t - old_p_t) * delta_t
    ratios = torch.exp(new_log_probs - old_log_t)
    clipped_ratios = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(ratios * advantages, clipped_ratios * advantages)
    if sample_weights is None:
        policy_loss = -torch.mean(surrogate)
    else:
        weights_t = torch.as_tensor(sample_weights, dtype=torch.float32).reshape(-1).detach()
        if weights_t.shape != (sample_count,) or not bool(torch.all(torch.isfinite(weights_t)).item()):
            raise ValueError("Gate sample_weights shape/finite性非法。")
        if not bool(torch.all(weights_t >= 0.0).item()) or not bool(
            torch.isclose(torch.sum(weights_t), torch.ones((), dtype=torch.float32), atol=1e-6, rtol=1e-5).item()
        ):
            raise ValueError("Gate sample_weights 必须是非负且和为1。")
        policy_loss = -torch.sum(weights_t * surrogate)
    zero = policy_loss.detach() * 0.0
    with torch.no_grad():
        approx_kl = torch.mean(old_log_t - new_log_probs)
        clip_fraction = torch.mean((torch.abs(ratios - 1.0) > clip_epsilon).to(torch.float32))
        ratio_mean = torch.mean(ratios)
        ratio_min = torch.min(ratios)
        ratio_max = torch.max(ratios)
    return PPOLossResult(
        total_loss=policy_loss,
        policy_loss=policy_loss,
        value_loss=zero,
        entropy=zero,
        ratio_mean=float(ratio_mean.detach().cpu()),
        ratio_min=float(ratio_min.detach().cpu()),
        ratio_max=float(ratio_max.detach().cpu()),
        clip_fraction=float(clip_fraction.detach().cpu()),
        approx_kl=float(approx_kl.detach().cpu()),
        new_log_probs=new_log_probs,
        new_values=torch.zeros_like(new_log_probs),
    )


def compute_ppo_loss(
    policy: Any,
    rollout_batch: ppo_rollout.RolloutBatch,
    clip_epsilon: float,
    value_loss_coefficient: float,
    entropy_coefficient: float,
) -> PPOLossResult:
    """计算一次 full-batch PPO clipped loss；不 backward，不 optimizer.step。"""

    _require_torch()
    if clip_epsilon < 0.0:
        raise ValueError("clip_epsilon 不能为负。")
    observations = torch.as_tensor(rollout_batch.observations, dtype=torch.float32)
    actions = torch.as_tensor(rollout_batch.actions, dtype=torch.float32)
    old_log_probs = torch.as_tensor(rollout_batch.old_log_probs, dtype=torch.float32).detach()
    advantages = torch.as_tensor(rollout_batch.advantages, dtype=torch.float32).detach()
    returns = torch.as_tensor(rollout_batch.returns, dtype=torch.float32).detach()

    evaluation_kwargs = {
        "action_masks": rollout_batch.action_masks,
    }
    if getattr(rollout_batch, "gate_actions", None) is not None:
        evaluation_kwargs["gate_actions"] = rollout_batch.gate_actions
    evaluation = policy.evaluate_actions(
        observations,
        actions,
        **evaluation_kwargs,
    )
    new_log_probs = evaluation.joint_log_prob
    new_values = evaluation.value
    entropy_per_sample = (
        evaluation.subpolicy.entropy
        if hasattr(evaluation, "subpolicy")
        else evaluation.entropy
    )
    ratios = torch.exp(new_log_probs - old_log_probs)
    surrogate = torch.minimum(
        ratios * advantages,
        torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages,
    )
    policy_loss = -surrogate.mean()
    value_loss = 0.5 * torch.mean((new_values - returns) ** 2)
    entropy = entropy_per_sample.mean()
    total_loss = policy_loss + value_loss_coefficient * value_loss - entropy_coefficient * entropy
    with torch.no_grad():
        approx_kl = torch.mean(old_log_probs - new_log_probs)
        clip_fraction = torch.mean((torch.abs(ratios - 1.0) > clip_epsilon).to(torch.float32))
    return PPOLossResult(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        ratio_mean=float(torch.mean(ratios).detach().cpu()),
        ratio_min=float(torch.min(ratios).detach().cpu()),
        ratio_max=float(torch.max(ratios).detach().cpu()),
        clip_fraction=float(clip_fraction.detach().cpu()),
        approx_kl=float(approx_kl.detach().cpu()),
        new_log_probs=new_log_probs,
        new_values=new_values,
    )


def _artificial_clipping_sanity() -> float:
    _require_torch()
    ratios = torch.tensor([1.3, 1.3, 0.7, 0.7], dtype=torch.float32)
    advantages = torch.tensor([2.0, -2.0, 2.0, -2.0], dtype=torch.float32)
    epsilon = 0.2
    value = clipped_surrogate(ratios, advantages, epsilon)
    expected_terms = torch.tensor(
        [
            min(1.3 * 2.0, 1.2 * 2.0),
            min(1.3 * -2.0, 1.2 * -2.0),
            min(0.7 * 2.0, 0.8 * 2.0),
            min(0.7 * -2.0, 0.8 * -2.0),
        ],
        dtype=torch.float32,
    )
    return float(torch.abs(value - expected_terms.mean()).detach().cpu())


def _pre_update_ratio_diagnostics(policy: Any, rollout_batch: ppo_rollout.RolloutBatch) -> dict[str, float]:
    _require_torch()
    evaluation = policy.evaluate_actions(
        rollout_batch.observations,
        rollout_batch.actions,
        action_masks=rollout_batch.action_masks,
    )
    old_log_probs = torch.as_tensor(rollout_batch.old_log_probs, dtype=torch.float32)
    ratios = torch.exp(evaluation.joint_log_prob - old_log_probs)
    approx_kl = torch.mean(old_log_probs - evaluation.joint_log_prob)
    return {
        "ratio_mean": float(torch.mean(ratios).detach().cpu()),
        "ratio_min": float(torch.min(ratios).detach().cpu()),
        "ratio_max": float(torch.max(ratios).detach().cpu()),
        "approx_kl": float(approx_kl.detach().cpu()),
    }


def _collect_single_short_rollout(
    cfg: dict[str, Any],
    config_path: str,
    origin_start_index: int = 0,
    max_steps: int = 8,
) -> tuple[Any, ppo_rollout.RolloutBatch]:
    """只收集一次短rollout；不做重复确定性rollout，不进入训练循环。"""

    root = Path(cfg["_meta"]["project_root"])
    seed = int(project_config.get_config_value(cfg, "experiment.seed"))
    cache_cfg = cache_configuration.cache_config_from_project(cfg)
    scenarios = cache_configuration.load_prediction_scenarios(
        cache_configuration.resolve_path(root, project_config.get_config_value(cfg, "paths.prediction_scenarios"))
    )
    service_catalog = cache_configuration.load_service_catalog(
        cache_configuration.resolve_path(root, project_config.get_config_value(cfg, "paths.service_catalog"))
    )
    first_origin = cache_configuration.scenario_at_origin(scenarios, origin_start_index)
    cache_nodes = cache_configuration.build_cache_nodes(first_origin, cache_cfg)
    initial_cache = cache_configuration.deterministic_initial_cache(
        first_origin,
        service_catalog,
        cache_nodes,
        cache_cfg,
    )
    initial_uav = uav_configuration.initial_uav_positions_from_project(cfg)
    boundary = uav_configuration.boundary_from_project(cfg)
    local_cache = task_scenario.load_tv_local_cache(
        cache_configuration.resolve_path(root, project_config.get_config_value(cfg, "paths.tv_local_cache"))
    )
    aligned_steps = ppo_rollout.prepare_aligned_rollout_steps(
        root,
        scenarios,
        origin_start_index,
        max_steps,
        root / "data/generated/vehicle_predictions.csv",
        root / "data/generated/popularity_predictions.csv",
        root / "data/generated/tasks.csv",
        root / "data/generated/trajectory.csv",
    )
    action_spec = ppo_trigger_policy.TriggerActionSpec.from_cache_nodes_and_project(cache_nodes, cfg)
    observation_spec = ppo_trigger_policy.ConfigurationObservationSpec.from_cache_nodes(
        cache_nodes,
        service_catalog["service_ids"].tolist(),
        action_spec.uav_ids,
        [str(value) for value in first_origin["vehicle_ids"].tolist()],
    )
    gamma, gae_lambda = ppo_rollout.ppo_gae_config_from_project(cfg)
    terminal_slot = int(scenarios["origin_slots"][-1])
    ppo_trigger_policy.set_torch_seed(seed)
    np.random.seed(seed)
    policy = ppo_trigger_policy.ConditionalTriggerActorCritic(
        observation_spec.observation_dim,
        action_spec,
        **ppo_trigger_policy.ppo_network_config_from_project(cfg),
    )
    batch = ppo_rollout.collect_on_policy_rollout(
        policy,
        initial_cache,
        initial_uav,
        aligned_steps,
        local_cache,
        service_catalog,
        cfg,
        cache_nodes,
        cache_cfg,
        boundary,
        observation_spec,
        action_spec,
        gamma,
        gae_lambda,
        terminal_slot,
    )
    return policy, batch


def run_single_update_sanity(config_path: str = "system_config.toml") -> SingleUpdateSanityResult:
    """仅用于单次更新数值验证，不是训练循环。

    如果正式训练超参缺失，函数停在 loss/ratio/clipping sanity，不执行 optimizer.step。
    """

    _require_torch()
    cfg = project_config.load_config(config_path)
    project_config.validate_for_formal_experiment(cfg)
    training_cfg = ppo_training_config_status(cfg)
    policy, batch = _collect_single_short_rollout(cfg, config_path)
    old_log_probs_before = batch.old_log_probs.copy()
    old_values_before = batch.old_values.copy()
    advantages_before = batch.advantages.copy()
    returns_before = batch.returns.copy()
    pre_diag = _pre_update_ratio_diagnostics(policy, batch)
    artificial_error = _artificial_clipping_sanity()

    if not training_cfg.can_run_optimizer_step:
        return SingleUpdateSanityResult(
            training_config=training_cfg,
            loss_result=None,
            artificial_clipping_sanity_error=artificial_error,
            pre_update_ratio_mean=pre_diag["ratio_mean"],
            pre_update_ratio_min=pre_diag["ratio_min"],
            pre_update_ratio_max=pre_diag["ratio_max"],
            pre_update_clip_fraction=None,
            pre_update_approx_kl=pre_diag["approx_kl"],
            optimizer_step_count=0,
            gradient_finite=None,
            gradient_norm=None,
            parameter_delta_norm=None,
            post_update_ratio_mean=None,
            post_update_ratio_min=None,
            post_update_ratio_max=None,
            post_update_approx_kl=None,
            post_update_clip_fraction=None,
            old_log_probs_unchanged=bool(np.array_equal(batch.old_log_probs, old_log_probs_before)),
            old_values_unchanged=bool(np.array_equal(batch.old_values, old_values_before)),
            advantages_unchanged=bool(np.array_equal(batch.advantages, advantages_before)),
            returns_unchanged=bool(np.array_equal(batch.returns, returns_before)),
            formal_validation=True,
            blocker="正式PPO update超参数缺失，按要求未执行optimizer.step。",
        )

    loss = compute_ppo_loss(
        policy,
        batch,
        float(training_cfg.clip_epsilon),
        float(training_cfg.value_loss_coefficient),
        float(training_cfg.entropy_coefficient),
    )
    if not torch.isfinite(loss.total_loss):
        raise FloatingPointError("PPO total loss 非finite。")
    if not (abs(loss.ratio_mean - 1.0) < 1e-5 and abs(loss.approx_kl) < 1e-5):
        raise RuntimeError("pre-update ratio/KL sanity failed，拒绝执行optimizer.step。")

    before_params = [param.detach().clone() for param in policy.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(training_cfg.learning_rate))
    optimizer.zero_grad(set_to_none=True)
    loss.total_loss.backward()
    grads = [param.grad.detach() for param in policy.parameters() if param.grad is not None]
    gradient_finite = bool(all(torch.isfinite(grad).all().item() for grad in grads))
    grad_norm = float(torch.sqrt(sum(torch.sum(grad ** 2) for grad in grads)).detach().cpu()) if grads else 0.0
    if not gradient_finite:
        raise FloatingPointError("gradient 出现 NaN/Inf，拒绝 optimizer.step。")
    if training_cfg.max_grad_norm is not None:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), training_cfg.max_grad_norm).detach().cpu())
    step_counter = 0
    optimizer.step()
    step_counter += 1
    if step_counter > 1:
        raise RuntimeError("optimizer.step 执行次数超过1。")
    after_params = [param.detach() for param in policy.parameters() if param.requires_grad]
    delta_norm = float(
        torch.sqrt(
            sum(torch.sum((after - before) ** 2) for before, after in zip(before_params, after_params))
        ).detach().cpu()
    )
    post = compute_ppo_loss(
        policy,
        batch,
        float(training_cfg.clip_epsilon),
        float(training_cfg.value_loss_coefficient),
        float(training_cfg.entropy_coefficient),
    )
    return SingleUpdateSanityResult(
        training_config=training_cfg,
        loss_result=loss,
        artificial_clipping_sanity_error=artificial_error,
        pre_update_ratio_mean=loss.ratio_mean,
        pre_update_ratio_min=loss.ratio_min,
        pre_update_ratio_max=loss.ratio_max,
        pre_update_clip_fraction=loss.clip_fraction,
        pre_update_approx_kl=loss.approx_kl,
        optimizer_step_count=step_counter,
        gradient_finite=gradient_finite,
        gradient_norm=grad_norm,
        parameter_delta_norm=delta_norm,
        post_update_ratio_mean=post.ratio_mean,
        post_update_ratio_min=post.ratio_min,
        post_update_ratio_max=post.ratio_max,
        post_update_approx_kl=post.approx_kl,
        post_update_clip_fraction=post.clip_fraction,
        old_log_probs_unchanged=bool(np.array_equal(batch.old_log_probs, old_log_probs_before)),
        old_values_unchanged=bool(np.array_equal(batch.old_values, old_values_before)),
        advantages_unchanged=bool(np.array_equal(batch.advantages, advantages_before)),
        returns_unchanged=bool(np.array_equal(batch.returns, returns_before)),
        formal_validation=True,
        blocker="",
    )


def _summary(result: SingleUpdateSanityResult) -> dict[str, Any]:
    loss = result.loss_result
    return {
        "missing_required_for_step": list(result.training_config.missing_required_for_step),
        "clip_epsilon": result.training_config.clip_epsilon,
        "learning_rate": result.training_config.learning_rate,
        "value_loss_coefficient": result.training_config.value_loss_coefficient,
        "entropy_coefficient": result.training_config.entropy_coefficient,
        "max_grad_norm": result.training_config.max_grad_norm,
        "pre_update_ratio_mean": result.pre_update_ratio_mean,
        "pre_update_ratio_min": result.pre_update_ratio_min,
        "pre_update_ratio_max": result.pre_update_ratio_max,
        "pre_update_clip_fraction": result.pre_update_clip_fraction,
        "pre_update_approx_kl": result.pre_update_approx_kl,
        "pre_update_policy_loss": None if loss is None else float(loss.policy_loss.detach().cpu()),
        "pre_update_value_loss": None if loss is None else float(loss.value_loss.detach().cpu()),
        "pre_update_entropy": None if loss is None else float(loss.entropy.detach().cpu()),
        "pre_update_total_loss": None if loss is None else float(loss.total_loss.detach().cpu()),
        "artificial_clipping_sanity_error": result.artificial_clipping_sanity_error,
        "gradient_finite": result.gradient_finite,
        "gradient_norm": result.gradient_norm,
        "optimizer_step_count": result.optimizer_step_count,
        "parameter_delta_norm": result.parameter_delta_norm,
        "post_update_ratio_mean": result.post_update_ratio_mean,
        "post_update_ratio_min": result.post_update_ratio_min,
        "post_update_ratio_max": result.post_update_ratio_max,
        "post_update_approx_kl": result.post_update_approx_kl,
        "post_update_clip_fraction": result.post_update_clip_fraction,
        "old_log_probs_unchanged": result.old_log_probs_unchanged,
        "old_values_unchanged": result.old_values_unchanged,
        "advantages_unchanged": result.advantages_unchanged,
        "returns_unchanged": result.returns_unchanged,
        "formal_validation": result.formal_validation,
        "blocker": result.blocker,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single PPO clipped loss/update sanity; no training loop.")
    parser.add_argument("--config", default="system_config.toml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_single_update_sanity(args.config)
    print(json.dumps(_summary(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
