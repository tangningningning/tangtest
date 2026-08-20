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
import os
import random
import tempfile
import warnings
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common import project_config
from decision import cache_configuration, slot_environment, task_allocation, uav_configuration
from ppo import ppo_rollout, ppo_trigger_policy, ppo_update
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
LATEST_CHECKPOINT_FILENAME = "latest_checkpoint.pt"
TRAINING_HISTORY_FILENAME = "training_history.json"


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
    # 本次调用新完成的episode仍放在episode_summaries；以下字段用于明确
    # resume后的全局状态，不保存RolloutBatch等大对象。
    completed_episodes: int = 0
    training_history: dict[str, Any] | None = None
    resumed: bool = False


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


def _empty_training_history() -> dict[str, Any]:
    """创建新的、只包含轻量summary的训练历史。"""

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
) -> dict[str, Any]:
    """把一个完整episode的两个正式summary转换为轻量builtin记录。"""

    if int(episode_summary.episode_index) < 0:
        raise ValueError("TrainingEpisodeSummary.episode_index 不能为负。")
    return {
        "episode_index": int(episode_summary.episode_index),
        "training_episode_summary": summary_to_builtin(episode_summary),
        "ppo_update_summary": summary_to_builtin(update_summary),
    }


def _build_training_history(
    records: list[dict[str, Any]],
    policy_version: int,
    optimizer_step_count: int,
) -> dict[str, Any]:
    """构造checkpoint和JSON共用的canonical training history。"""

    history = {
        "schema_version": TRAINING_HISTORY_SCHEMA_VERSION,
        "completed_episodes": int(len(records)),
        "policy_version": int(policy_version),
        "optimizer_step_count": int(optimizer_step_count),
        "episodes": records,
    }
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
    }
    kwargs: dict[str, Any] = {}
    for field in fields(TrainingEpisodeSummary):
        if field.name not in item:
            raise ValueError(f"training_episode_summary 缺少字段：{field.name}")
        raw = item[field.name]
        kwargs[field.name] = int(raw) if field.name in integer_fields else float(raw)
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
        if field.name != "epoch_summaries"
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
    )


def _canonical_training_history(value: Any) -> dict[str, Any]:
    """严格校验并规范化history，避免JSON mirror成为resume权威。"""

    source = _as_mapping(value, "training_history")
    if int(source.get("schema_version", -1)) != TRAINING_HISTORY_SCHEMA_VERSION:
        raise ValueError("training_history schema_version 不兼容。")
    raw_episodes = source.get("episodes")
    if not isinstance(raw_episodes, list):
        raise ValueError("training_history.episodes 必须是数组。")

    records: list[dict[str, Any]] = []
    for expected_index, raw_record in enumerate(raw_episodes):
        record = _as_mapping(raw_record, f"training_history.episodes[{expected_index}]")
        episode_summary = _training_episode_summary_from_builtin(record.get("training_episode_summary"))
        update_summary = _formal_update_summary_from_builtin(record.get("ppo_update_summary"))
        record_index = int(record.get("episode_index", episode_summary.episode_index))
        if record_index != expected_index or episode_summary.episode_index != expected_index:
            raise ValueError("training_history episode_index 必须从0连续递增。")
        records.append(_training_episode_history_record(episode_summary, update_summary))

    completed_episodes = int(source.get("completed_episodes", -1))
    policy_version = int(source.get("policy_version", -1))
    optimizer_step_count = int(source.get("optimizer_step_count", -1))
    if completed_episodes != len(records):
        raise ValueError("training_history.completed_episodes 与episodes数量不一致。")
    if completed_episodes < 0 or policy_version < 0 or optimizer_step_count < 0:
        raise ValueError("training_history counters 不能为负。")
    return _build_training_history(records, policy_version, optimizer_step_count)


def _history_episode_summaries(history: dict[str, Any]) -> tuple[TrainingEpisodeSummary, ...]:
    """从canonical history提取Figure 1可直接消费的summary序列。"""

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


def _capture_rng_states(policy: Any, rng_generator: Any) -> dict[str, Any]:
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

    device = _policy_device(policy)
    if getattr(device, "type", str(device)) == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("policy位于CUDA但当前torch.cuda不可用，无法保存CUDA RNG。")
        states["torch_cuda_rng_state_all"] = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return states


def _restore_rng_states(policy: Any, rng_generator: Any, states: dict[str, Any]) -> None:
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


def _seed_new_formal_run(seed: int, rng_generator: Any) -> None:
    """新run自动构造policy时使用配置seed；不在模块import时改全局RNG。"""

    random.seed(int(seed))
    np.random.seed(int(seed))
    ppo_trigger_policy.set_torch_seed(int(seed))
    rng_generator.manual_seed(int(seed))


def _build_formal_policy_and_optimizer(ctx: TrainingSanityContext) -> tuple[Any, Any]:
    """按当前config构造policy和Adam；不加载checkpoint、不执行step。"""

    _require_torch()
    training_cfg = ppo_update.ppo_training_config_status(ctx.cfg)
    if not training_cfg.can_run_optimizer_step or training_cfg.learning_rate is None:
        raise RuntimeError(f"PPO update超参数缺失：{training_cfg.missing_required_for_step}")
    policy = ppo_trigger_policy.ConditionalTriggerActorCritic(
        ctx.observation_spec.observation_dim,
        ctx.action_spec,
        **ppo_trigger_policy.ppo_network_config_from_project(ctx.cfg),
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(training_cfg.learning_rate))
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
    return {
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
        "policy_network": {
            "class": type(policy).__name__,
            "observation_dim": int(getattr(policy, "observation_dim")),
            "action_dim": int(getattr(policy, "action_dim")),
            "hidden_sizes": [int(value) for value in network_cfg["hidden_sizes"]],
            "autoregressive_hidden_size": int(network_cfg["autoregressive_hidden_size"]),
        },
    }


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
) -> dict[str, Any]:
    """构造不含RolloutBatch/梯度/候选trace的权威checkpoint。"""

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
) -> tuple[dict[str, Any], int, int]:
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
    if int(payload.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise RuntimeError(
            "CHECKPOINT_COMPATIBILITY_ERROR: checkpoint_version不支持。"
        )
    mismatch = _metadata_mismatch(expected_compatibility, payload.get("compatibility"))
    if mismatch:
        raise RuntimeError(f"CHECKPOINT_COMPATIBILITY_ERROR: {mismatch}")

    try:
        history = _canonical_training_history(payload.get("training_history"))
    except (TypeError, ValueError, KeyError) as exc:
        raise RuntimeError(f"CHECKPOINT_COMPATIBILITY_ERROR: training_history无效：{exc}") from exc
    policy_version = int(payload.get("policy_version", -1))
    optimizer_step_count = int(payload.get("optimizer_step_count", -1))
    completed_episodes = int(payload.get("completed_episodes", -1))
    if (
        policy_version != int(history["policy_version"])
        or optimizer_step_count != int(history["optimizer_step_count"])
        or completed_episodes != int(history["completed_episodes"])
    ):
        raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: checkpoint counters与training_history不一致。")

    try:
        policy.load_state_dict(payload["policy_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    except Exception as exc:
        raise RuntimeError(f"CHECKPOINT_COMPATIBILITY_ERROR: policy/optimizer state不兼容：{exc}") from exc
    rng_states = payload.get("rng_states")
    if not isinstance(rng_states, dict):
        raise RuntimeError("CHECKPOINT_COMPATIBILITY_ERROR: rng_states缺失或类型错误。")
    _restore_rng_states(policy, rng_generator, rng_states)
    return history, policy_version, optimizer_step_count


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
    run_dir: str | Path | None = None,
    resume: bool = False,
) -> FormalTrainingRunPlan | FormalTrainingRunResult:
    """正式PPO训练run入口。

    默认 execute=False：只构造并返回计划，不执行environment、不backward、不optimizer.step。
    execute=True时，run_dir必须由caller显式提供；每个完整episode完成后才保存checkpoint。
    resume=True时，num_episodes表示该run希望达到的总完成episode数，而不是额外episode数。
    若未由caller传入policy/optimizer，本函数按当前config构造它们；resume加载顺序仍为：
    构造policy/optimizer -> load state_dict -> 恢复RNG -> 才开始下一episode。
    """

    if execute and run_dir is None:
        raise ValueError("execute=True 时必须显式提供 run_dir；不会默认写入data/generated或当前目录。")
    if resume and not execute:
        raise ValueError("resume=True 只允许用于 execute=True 的正式训练run。")

    ctx = _build_context(config_path)
    protocol = build_formal_training_protocol(ctx.scenarios, ctx.cfg)
    validate_formal_training_protocol(protocol)
    plan = _build_formal_training_plan(protocol, num_episodes, execute)
    if not execute:
        return plan
    _require_torch()

    _formal_run_dir, checkpoint_path, history_path = _prepare_formal_run_dir(run_dir, resume)
    seed = int(project_config.get_config_value(ctx.cfg, "experiment.seed"))

    if (policy is None) != (optimizer is None):
        raise ValueError("execute=True 时 policy 和 optimizer 必须同时提供或同时省略。")

    if rng_generator is None:
        rng_generator = torch.Generator()
        if not resume:
            rng_generator.manual_seed(seed)
    elif not hasattr(rng_generator, "get_state") or not hasattr(rng_generator, "set_state"):
        raise TypeError("rng_generator 必须是支持get_state/set_state的torch.Generator。")

    if policy is None and optimizer is None:
        # 初始化可能消耗RNG；resume时此后会先load state_dict，最后再恢复checkpoint RNG。
        _seed_new_formal_run(seed, rng_generator)
        policy, optimizer = _build_formal_policy_and_optimizer(ctx)

    compatibility = _formal_compatibility_metadata(ctx, protocol, policy)
    resumed = bool(resume)
    if resume:
        history, policy_version, optimizer_step_count = _load_formal_checkpoint(
            checkpoint_path,
            policy,
            optimizer,
            rng_generator,
            compatibility,
        )
        # JSON只是人类可读mirror；无论它落后、缺失还是声称更新，都以checkpoint为准。
        _synchronize_training_history_json(history_path, history)
    else:
        history = _empty_training_history()
        policy_version = 0
        optimizer_step_count = 0

    records = list(history["episodes"])
    completed_episodes = int(history["completed_episodes"])
    episode_summaries: list[TrainingEpisodeSummary] = []
    for episode_index in range(completed_episodes, num_episodes):
        batch = collect_formal_episode_rollout(policy, ctx, protocol)
        raw_stats = _formal_raw_rollout_statistics(batch)
        update = update_policy_on_rollout(policy, optimizer, batch, ctx.cfg, rng_generator)
        if update.optimizer_steps != int(protocol.expected_optimizer_steps_per_rollout):
            raise RuntimeError(
                "完整episode的PPO optimizer steps不符合FormalTrainingProtocol，"
                "不得写入completed history。"
            )
        if len(update.epoch_summaries) != int(protocol.update_epochs):
            raise RuntimeError(
                "FormalPPOUpdateSummary epoch_summaries不完整，不得写入completed history。"
            )
        policy_version_before = policy_version
        policy_version += 1
        optimizer_step_count += update.optimizer_steps
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
        )
        episode_summaries.append(episode_summary)

        # 只有完整episode、完整PPO update和summary全部成功后才进入history。
        records.append(_training_episode_history_record(episode_summary, update))
        history = _build_training_history(records, policy_version, optimizer_step_count)
        payload = _checkpoint_payload(
            policy,
            optimizer,
            policy_version,
            optimizer_step_count,
            history,
            compatibility,
            _capture_rng_states(policy, rng_generator),
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

    return FormalTrainingRunResult(
        plan=plan,
        episode_summaries=episode_summaries,
        final_policy_version=policy_version,
        optimizer_step_count=optimizer_step_count,
        completed_episodes=int(history["completed_episodes"]),
        training_history=history,
        resumed=resumed,
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
    actual_q_values: list[float] = []
    cache_cost_values: list[float] = []
    movement_cost_values: list[float] = []
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
                if result.matching[task.task_id] == task_allocation.CLOUD:
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
    return DeterministicEvaluationSummary(
        num_steps=len(rewards),
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
