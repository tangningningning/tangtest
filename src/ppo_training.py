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
import json
import inspect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import cache_configuration
import kkt_resource_allocation as kkt
import ppo_rollout
import ppo_trigger_policy
import ppo_update
import project_config
import slot_environment
import task_scenario
import uav_configuration

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SANITY_MAX_ROLLOUTS = 2
SANITY_MAX_STEPS_PER_ROLLOUT = 8
SANITY_MAX_TOTAL_ENV_STEPS = 16
SANITY_MAX_OPTIMIZER_STEPS = 2


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
class FormalTrainingProtocol:
    """正式PPO单episode训练协议定义；只描述协议，不执行训练。

    当前项目采用 SINGLE-TRACE TRAINING BASELINE：每个训练episode复用同一套
    SUMO/tasks/prediction artifacts；episode之间 reset C/Q，但 policy、optimizer
    和训练RNG在同一个training run中持续推进。
    """

    episode_origin_indices: np.ndarray
    episode_slots: np.ndarray
    episode_times_s: np.ndarray
    steps_per_episode: int
    update_epochs: int
    num_minibatches: int
    minibatch_size: int
    expected_optimizer_steps_per_rollout: int
    true_terminal_origin: int
    true_terminal_slot: int
    true_terminal_time_s: float
    single_trace_training: bool
    initial_cache_source: str
    initial_uav_source: str
    tv_local_cache_source: str
    evaluation_semantics: str


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
    deterministic_evaluation_api_available: bool
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


@dataclass(frozen=True)
class TrainingEpisodeSummary:
    """正式训练 episode 级 summary；一个完整rollout对应一次policy_version递增。"""

    episode_index: int
    policy_version_before: int
    policy_version_after: int
    rollout_reward_sum: float
    rollout_reward_mean: float
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


@dataclass(frozen=True)
class FormalTrainingRunResult:
    """正式训练执行结果；本阶段不会在CLI或dry-run中创建。"""

    plan: FormalTrainingRunPlan
    episode_summaries: list[TrainingEpisodeSummary]
    final_policy_version: int
    optimizer_step_count: int


@dataclass(frozen=True)
class DeterministicEvaluationSummary:
    """in-trace deterministic evaluation 的episode级指标。"""

    num_steps: int
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
    all_steps_feasible: bool
    reward_identity_error: float
    evaluation_semantics: str


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
    service_catalog = cache_configuration.load_service_catalog(
        cache_configuration.resolve_path(root, project_config.get_config_value(cfg, "paths.service_catalog"))
    )
    first_origin = cache_configuration.scenario_at_origin(scenarios, 0)
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
    initial_cache = cache_configuration.deterministic_initial_cache(
        first_origin,
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
        terminal_slot=int(scenarios["origin_slots"][-1]),
        initial_cache=initial_cache,
        initial_uav=initial_uav,
    )


def build_formal_training_protocol(
    scenarios: dict[str, Any],
    project_cfg: dict[str, Any],
) -> FormalTrainingProtocol:
    """由 prediction_scenarios metadata 构造正式单episode PPO训练协议。

    纯协议函数：不创建policy、不调用execute_slot、不创建optimizer、不训练。
    """

    origin_indices = np.arange(int(scenarios["origin_slots"].shape[0]), dtype=np.int64)
    slots = np.asarray(scenarios["origin_slots"], dtype=np.int64)
    times = np.asarray(scenarios["origin_times_s"], dtype=np.float64)
    steps_per_episode = int(len(origin_indices))
    update_epochs = int(project_config.get_config_value(project_cfg, "algorithm.ppo.update_epochs"))
    num_minibatches = int(project_config.get_config_value(project_cfg, "algorithm.ppo.num_minibatches"))
    if steps_per_episode <= 0:
        raise ValueError("正式episode没有可执行configuration origin。")
    if steps_per_episode % num_minibatches != 0:
        raise ValueError(
            f"steps_per_episode={steps_per_episode} 不能被 num_minibatches={num_minibatches} 整除。"
        )
    minibatch_size = steps_per_episode // num_minibatches
    if minibatch_size <= 0:
        raise ValueError("推导得到的 minibatch_size 必须大于0。")
    return FormalTrainingProtocol(
        episode_origin_indices=origin_indices,
        episode_slots=slots,
        episode_times_s=times,
        steps_per_episode=steps_per_episode,
        update_epochs=update_epochs,
        num_minibatches=num_minibatches,
        minibatch_size=minibatch_size,
        expected_optimizer_steps_per_rollout=update_epochs * num_minibatches,
        true_terminal_origin=int(origin_indices[-1]),
        true_terminal_slot=int(slots[-1]),
        true_terminal_time_s=float(times[-1]),
        single_trace_training=True,
        initial_cache_source=(
            "cache_configuration.deterministic_initial_cache(first_origin, service_catalog, "
            "cache_nodes, cache_cfg)"
        ),
        initial_uav_source="uav_configuration.initial_uav_positions_from_project(system_config)",
        tv_local_cache_source=str(project_config.get_config_value(project_cfg, "paths.tv_local_cache")),
        evaluation_semantics=(
            "in-trace deterministic evaluation: deterministic=True, no network update, "
            "reset C/Q to formal initial state, execute first to last origin, report raw G(t)"
        ),
    )


def validate_formal_training_protocol(protocol: FormalTrainingProtocol) -> dict[str, bool]:
    """静态校验正式训练协议；不执行环境。"""

    origins = protocol.episode_origin_indices
    slots = protocol.episode_slots
    times = protocol.episode_times_s
    checks = {
        "origins_contiguous": bool(np.array_equal(origins, np.arange(len(origins), dtype=np.int64))),
        "slots_strictly_increasing": bool(np.all(np.diff(slots) > 0)),
        "times_strictly_increasing": bool(np.all(np.diff(times) > 0.0)),
        "rollout_size_positive": bool(protocol.steps_per_episode > 0),
        "minibatch_divisible": bool(protocol.steps_per_episode % protocol.num_minibatches == 0),
        "minibatch_size_positive": bool(protocol.minibatch_size > 0),
        "true_terminal_is_last_origin": bool(protocol.true_terminal_origin == int(origins[-1])),
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
    sample_num_episodes = 3
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
        deterministic_evaluation_api_available=callable(evaluate_formal_episode),
        sample_num_episodes=sample_num_episodes,
        planned_environment_steps=sample_num_episodes * protocol.steps_per_episode,
        planned_optimizer_steps=sample_num_episodes * protocol.expected_optimizer_steps_per_rollout,
        planned_backward_calls=sample_num_episodes * protocol.expected_optimizer_steps_per_rollout,
        optimizer_instances_planned=1,
        executor_dry_run_only=True,
        formal_validation=True,
        environment_steps_executed=0,
        backward_calls_executed=0,
        optimizer_steps_executed=0,
        blocker="",
    )


def _formal_aligned_steps(
    ctx: TrainingSanityContext,
    protocol: FormalTrainingProtocol,
) -> list[ppo_rollout.AlignedRolloutStep]:
    """构造完整正式episode输入；只在正式执行器被显式调用时使用。"""

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
    )
    if len(steps) != protocol.steps_per_episode:
        raise RuntimeError(
            f"完整episode应包含 {protocol.steps_per_episode} 个可执行origin，实际得到 {len(steps)}。"
        )
    if int(steps[-1].origin_index) != int(protocol.true_terminal_origin):
        raise RuntimeError("完整episode最后一个origin不是FormalTrainingProtocol定义的true terminal。")
    return steps


def collect_formal_episode_rollout(
    policy: Any,
    ctx: TrainingSanityContext,
    protocol: FormalTrainingProtocol,
) -> ppo_rollout.RolloutBatch:
    """收集一个完整正式episode rollout。

    语义：
    1. episode开始 reset 到正式 initial Cache 和 initial UAV positions；
    2. 从 FormalTrainingProtocol 第一个 origin 连续执行到最后一个 origin；
    3. episode内部 C(t)、Q^U(t) 逐slot传播，不逐slot reset；
    4. 最后一个origin为true terminal，因此GAE bootstrap=0；
    5. policy动作不mask、不repair、不resample；不可行动作向上抛出 InfeasibleActionBlocker。

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
    )
    if batch.truncated or not bool(batch.metadata.get("true_terminal", False)):
        raise RuntimeError("正式完整episode rollout必须以true terminal结束，且bootstrap必须为0。")
    if not math.isclose(float(batch.final_bootstrap_value), 0.0, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("正式完整episode rollout的GAE bootstrap必须为0。")
    return batch


def _slice_rollout_batch(
    batch: ppo_rollout.RolloutBatch,
    indices: np.ndarray,
) -> ppo_rollout.RolloutBatch:
    """按minibatch索引切片RolloutBatch；old policy数据保持来自原rollout。"""

    idx = np.asarray(indices, dtype=np.int64)
    return ppo_rollout.RolloutBatch(
        observations=batch.observations[idx].copy(),
        actions=batch.actions[idx].copy(),
        old_log_probs=batch.old_log_probs[idx].copy(),
        old_values=batch.old_values[idx].copy(),
        rewards=batch.rewards[idx].copy(),
        dones=batch.dones[idx].copy(),
        advantages=batch.advantages[idx].copy(),
        returns=batch.returns[idx].copy(),
        slots=batch.slots[idx].copy(),
        origin_indices=batch.origin_indices[idx].copy(),
        final_bootstrap_value=float(batch.final_bootstrap_value),
        truncated=bool(batch.truncated),
        transitions=[batch.transitions[int(i)] for i in idx.tolist()],
        metadata=dict(batch.metadata),
    )


def _torch_randperm_numpy(size: int, rng_generator: Any) -> np.ndarray:
    _require_torch()
    if rng_generator is None:
        perm = torch.randperm(size)
    else:
        perm = torch.randperm(size, generator=rng_generator)
    return perm.detach().cpu().numpy().astype(np.int64)


def update_policy_on_rollout(
    policy: Any,
    optimizer: Any,
    rollout_batch: ppo_rollout.RolloutBatch,
    project_cfg: dict[str, Any],
    rng_generator: Any,
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
    gradient_finite_all = True
    each_transition_once_per_epoch = True
    optimizer_steps = 0

    all_indices = np.arange(batch_size, dtype=np.int64)
    for _epoch in range(update_epochs):
        shuffled = _torch_randperm_numpy(batch_size, rng_generator)
        each_transition_once_per_epoch = each_transition_once_per_epoch and bool(
            np.array_equal(np.sort(shuffled), all_indices)
        )
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
            gradient_norms.append(float(grad_norm.detach().cpu()))
            optimizer.step()
            optimizer_steps += 1
            policy_losses.append(float(loss.policy_loss.detach().cpu()))
            value_losses.append(float(loss.value_loss.detach().cpu()))
            entropies.append(float(loss.entropy.detach().cpu()))
            approx_kls.append(float(loss.approx_kl))
            clip_fractions.append(float(loss.clip_fraction))

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
    )


def _build_formal_training_plan(
    protocol: FormalTrainingProtocol,
    num_episodes: int,
    execute: bool,
) -> FormalTrainingRunPlan:
    if num_episodes <= 0:
        raise ValueError("num_episodes 必须由caller显式传入正整数。")
    return FormalTrainingRunPlan(
        num_episodes=int(num_episodes),
        steps_per_episode=int(protocol.steps_per_episode),
        planned_environment_steps=int(num_episodes * protocol.steps_per_episode),
        update_epochs=int(protocol.update_epochs),
        num_minibatches=int(protocol.num_minibatches),
        minibatch_size=int(protocol.minibatch_size),
        expected_optimizer_steps_per_episode=int(protocol.expected_optimizer_steps_per_rollout),
        planned_optimizer_steps=int(num_episodes * protocol.expected_optimizer_steps_per_rollout),
        planned_backward_calls=int(num_episodes * protocol.expected_optimizer_steps_per_rollout),
        optimizer_instances_planned=1,
        execute=bool(execute),
        single_trace_training=bool(protocol.single_trace_training),
        no_default_long_training_budget=True,
    )


def run_formal_ppo_training(
    *,
    num_episodes: int,
    config_path: str = "system_config.toml",
    policy: Any | None = None,
    optimizer: Any | None = None,
    rng_generator: Any | None = None,
    execute: bool = False,
) -> FormalTrainingRunPlan | FormalTrainingRunResult:
    """正式PPO训练run入口。

    默认 execute=False：只构造并返回计划，不执行environment、不backward、不optimizer.step。
    若未来显式 execute=True，caller必须传入已创建好的policy和optimizer；optimizer state
    跨episode/rollout/epoch/minibatch保持，不在本函数内部重新创建Adam。
    """

    ctx = _build_context(config_path)
    protocol = build_formal_training_protocol(ctx.scenarios, ctx.cfg)
    validate_formal_training_protocol(protocol)
    plan = _build_formal_training_plan(protocol, num_episodes, execute)
    if not execute:
        return plan
    _require_torch()
    if policy is None or optimizer is None:
        raise ValueError("execute=True 时必须由caller显式传入 policy 和 optimizer。")

    episode_summaries: list[TrainingEpisodeSummary] = []
    policy_version = 0
    optimizer_step_count = 0
    for episode_index in range(num_episodes):
        batch = collect_formal_episode_rollout(policy, ctx, protocol)
        update = update_policy_on_rollout(policy, optimizer, batch, ctx.cfg, rng_generator)
        policy_version_before = policy_version
        policy_version += 1
        optimizer_step_count += update.optimizer_steps
        episode_summaries.append(
            TrainingEpisodeSummary(
                episode_index=episode_index,
                policy_version_before=policy_version_before,
                policy_version_after=policy_version,
                rollout_reward_sum=float(np.sum(batch.rewards)),
                rollout_reward_mean=float(np.mean(batch.rewards)),
                actual_q_sum=float(sum(item.actual_task_utility for item in batch.transitions)),
                cache_cost_sum=float(sum(item.cache_cost for item in batch.transitions)),
                movement_cost_sum=float(sum(item.movement_cost for item in batch.transitions)),
                policy_loss_mean=update.policy_loss_mean,
                value_loss_mean=update.value_loss_mean,
                entropy_mean=update.entropy_mean,
                approx_kl_mean=update.approx_kl_mean,
                clip_fraction_mean=update.clip_fraction_mean,
                gradient_norm_mean=update.gradient_norm_mean,
                optimizer_steps=update.optimizer_steps,
            )
        )
    return FormalTrainingRunResult(
        plan=plan,
        episode_summaries=episode_summaries,
        final_policy_version=policy_version,
        optimizer_step_count=optimizer_step_count,
    )


def evaluate_formal_episode(
    policy: Any,
    ctx: TrainingSanityContext,
    protocol: FormalTrainingProtocol,
) -> DeterministicEvaluationSummary:
    """执行 in-trace deterministic evaluation。

    当前evaluation使用与single-trace training相同的external trace，因此只能称为
    in-trace deterministic evaluation，不是held-out/generalization evaluation。
    本函数不计算gradient、不调用optimizer、不更新network；episode开始 reset 到正式 C/Q。

    注意：这是正式评估执行器。本阶段CLI dry-run不会调用该函数。
    """

    _require_torch()
    steps = _formal_aligned_steps(ctx, protocol)
    previous_cache = np.asarray(ctx.initial_cache, dtype=np.int8).copy()
    previous_uav = {
        str(uav_id): (float(np.asarray(xy)[0]), float(np.asarray(xy)[1]))
        for uav_id, xy in ctx.initial_uav.items()
    }
    cache_trigger_counts = {label: 0 for label in ctx.action_spec.labels[: len(ctx.action_spec.cache_node_ids)]}
    uav_trigger_counts = {label: 0 for label in ctx.action_spec.labels[len(ctx.action_spec.cache_node_ids) :]}
    rewards: list[float] = []
    actual_q_sum = 0.0
    cache_cost_sum = 0.0
    movement_cost_sum = 0.0
    arrived_task_count = 0
    cache_service_available_task_count = 0
    cloud_execution_task_count = 0
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
            sample = policy.sample_action(observation, deterministic=True)
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
                if result.matching[task.task_id] == kkt.CLOUD:
                    cloud_execution_task_count += 1

            if not math.isclose(reward, actual_q - cache_cost - movement_cost, rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError("evaluation reward identity G=Q_actual-C_ca-C_q 失败。")
            rewards.append(reward)
            actual_q_sum += actual_q
            cache_cost_sum += cache_cost
            movement_cost_sum += movement_cost
            for index, trigger in enumerate(sample.cache_triggers.astype(int).tolist()):
                cache_trigger_counts[ctx.action_spec.labels[index]] += int(trigger)
            cache_count = len(ctx.action_spec.cache_node_ids)
            for uav_index, uav_id in enumerate(ctx.action_spec.uav_ids):
                uav_trigger_counts[ctx.action_spec.labels[cache_count + uav_index]] += int(sample.uav_triggers[uav_id])
            previous_cache = np.asarray(result.next_cache_state, dtype=np.int8).copy()
            previous_uav = dict(result.next_uav_positions or {})

    cumulative_g = float(np.sum(rewards))
    identity_error = abs(cumulative_g - (actual_q_sum - cache_cost_sum - movement_cost_sum))
    if identity_error > 1e-9:
        raise AssertionError("evaluation episode级 G 恒等式失败。")
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
    return DeterministicEvaluationSummary(
        num_steps=len(rewards),
        cumulative_system_utility=cumulative_g,
        mean_system_utility=float(np.mean(rewards)),
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
        all_steps_feasible=all_steps_feasible,
        reward_identity_error=float(identity_error),
        evaluation_semantics="in-trace deterministic evaluation",
    )


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
    summary = RolloutSanitySummary(
        rollout_index=rollout_index,
        collected_with_policy_version=policy_version,
        origin_indices=batch.origin_indices.astype(int).tolist(),
        slots=batch.slots.astype(int).tolist(),
        times_s=[float(item.time_s) for item in transitions],
        actions=[item.action_vector.astype(int).tolist() for item in transitions],
        rewards=rewards,
        reward_sum=float(np.sum(rewards)),
        reward_mean=float(np.mean(rewards)),
        actual_q_sum=float(sum(item.actual_task_utility for item in transitions)),
        cache_cost_sum=float(sum(item.cache_cost for item in transitions)),
        movement_cost_sum=float(sum(item.movement_cost for item in transitions)),
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
        0,
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
        formal_validation=True,
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
    formal_training_signature = inspect.signature(run_formal_ppo_training)
    num_episodes_parameter = formal_training_signature.parameters["num_episodes"]
    return {
        "environment_steps_executed": result.environment_steps_executed,
        "backward_calls_executed": result.backward_calls_executed,
        "optimizer_steps_executed": result.optimizer_steps_executed,
        "formal_rollout_executor_available": result.formal_rollout_executor_available,
        "ppo_update_executor_available": result.ppo_update_executor_available,
        "formal_training_api_available": result.formal_training_api_available,
        "deterministic_evaluation_api_available": result.deterministic_evaluation_api_available,
        "run_formal_ppo_training_num_episodes_required": (
            num_episodes_parameter.default is inspect.Parameter.empty
        ),
        "run_formal_ppo_training_execute_default": formal_training_signature.parameters["execute"].default,
        "sample_num_episodes_for_dry_run_only": result.sample_num_episodes,
        "planned_environment_steps": result.planned_environment_steps,
        "planned_optimizer_steps": result.planned_optimizer_steps,
        "planned_backward_calls": result.planned_backward_calls,
        "optimizer_instances_planned": result.optimizer_instances_planned,
        "executor_dry_run_only": result.executor_dry_run_only,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PPO training protocol dry-run; not a long training entrypoint.")
    parser.add_argument("--config", default="system_config.toml")
    parser.add_argument(
        "--bounded-sanity",
        action="store_true",
        help="Run the existing 2-rollout/2-update development sanity instead of dry-run.",
    )
    parser.add_argument("--skip-repeat-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bounded_sanity:
        result = run_bounded_training_sanity(args.config, repeat_check=not args.skip_repeat_check)
        print(json.dumps(_summary(result), ensure_ascii=False, indent=2))
    else:
        result = run_training_dry_run(args.config)
        print(json.dumps(_dry_run_summary(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
