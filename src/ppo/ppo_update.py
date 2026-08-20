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

    evaluation = policy.evaluate_actions(
        observations,
        actions,
        action_masks=rollout_batch.action_masks,
    )
    new_log_probs = evaluation.joint_log_prob
    new_values = evaluation.value
    entropy_per_sample = evaluation.entropy
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
