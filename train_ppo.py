"""SB3 PPO 的受控训练入口。

Stage15 提供显式启动的 25600-step 正式入口；默认 CLI 只做配置检查，不启动训练。
Stage13/14 的 5120-step 入口和第十一阶段的 ``run_controlled_training`` 作为历史受控入口保留。
训练仍然只使用官方 ``stable_baselines3.PPO`` 的默认 ``MlpPolicy``，底层
任务执行、缓存、UAV、Matching、KKT 和真实 system reward 全部复用现有环境。

入口职责固定为：

``config -> MECEnvironment -> AdaptiveCachePPOEnv -> VecNormalize -> SB3 PPO``

正式产物只写入新的 ``outputs/ppo_training/<run_id>/``；Reference 配置文件仍
保持 ``cache_controller.mode=reference``，不会因为 PPO 训练而改变默认闭环。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:  # pragma: no cover - 目标环境在正式训练前必须安装 SB3
    BaseCallback = object  # type: ignore[assignment,misc]

try:  # 支持从项目根目录直接运行，也支持作为包模块导入。
    from .data import load_config
    from .mobility import sumo_start_count
    from .environment import MECEnvironment
    from .ppo_env import (
        AdaptiveCachePPOEnv,
        PPO_CACHE_FEATURE_NAMES,
        runtime_diagnostics,
        sb3_check_env,
    )
except ImportError:  # pragma: no cover - 取决于调用方式
    from data import load_config
    from mobility import sumo_start_count
    from environment import MECEnvironment
    from ppo_env import (
        AdaptiveCachePPOEnv,
        PPO_CACHE_FEATURE_NAMES,
        runtime_diagnostics,
        sb3_check_env,
    )


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_SEED_BASE = 20260811
DEFAULT_EVAL_SEED = 20261811
REQUIRED_PPO_METRICS = (
    "approx_kl",
    "clip_fraction",
    "entropy_loss",
    "explained_variance",
    "policy_gradient_loss",
    "value_loss",
)


def prepare_ppo_config(
    config: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """读取配置副本并切换 PPO 模式，不改写项目默认 Reference 配置。"""

    if isinstance(config, Mapping):
        prepared = copy.deepcopy(dict(config))
    else:
        prepared = load_config() if config is None else load_config(config)
        prepared = copy.deepcopy(prepared)
    cache_controller = prepared.get("cache_controller", {})
    if not isinstance(cache_controller, Mapping):
        cache_controller = {}
    cache_controller = dict(cache_controller)
    cache_controller["mode"] = "ppo"
    prepared["cache_controller"] = cache_controller
    return prepared


def build_ppo_environment(
    config: Mapping[str, Any] | str | Path | None = None,
    *,
    episode_seed_base: int | None = None,
    seed_mode: str = "sequential",
    fixed_seed: int | None = None,
) -> AdaptiveCachePPOEnv:
    """创建 PPO wrapper；PPO 只输出 KEEP/UPDATE 两个高层动作。"""

    prepared = prepare_ppo_config(config)
    physical_environment = MECEnvironment(prepared)
    return AdaptiveCachePPOEnv(
        physical_environment,
        episode_seed_base=episode_seed_base,
        seed_mode=seed_mode,
        fixed_seed=fixed_seed,
    )


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, Mapping) else {}


def _sb3_device(config: Mapping[str, Any]) -> str:
    """读取设备配置；本阶段的正式配置固定为 CPU。"""

    ppo_section = _section(config, "ppo")
    training_section = _section(config, "training")
    return str(ppo_section.get("device", training_section.get("device", "cpu")))


def build_sb3_ppo(environment: Any, config: Mapping[str, Any] | None = None) -> Any:
    """构造官方 SB3 PPO；不在此函数中调用 ``learn``。"""

    try:
        from stable_baselines3 import PPO
    except ImportError as error:  # pragma: no cover - 依赖检查路径
        raise RuntimeError(
            "当前项目环境缺少 stable-baselines3，请在 adaptive_cache_mec 环境安装。"
        ) from error

    source_config = {} if config is None else config
    ppo_section = _section(source_config, "ppo")
    project_section = _section(source_config, "project")
    return PPO(
        "MlpPolicy",
        environment,
        learning_rate=float(ppo_section.get("learning_rate", 3.0e-4)),
        n_steps=int(ppo_section.get("n_steps", 256)),
        batch_size=int(ppo_section.get("batch_size", 64)),
        n_epochs=int(ppo_section.get("n_epochs", 10)),
        gamma=float(ppo_section.get("gamma", 0.99)),
        gae_lambda=float(ppo_section.get("gae_lambda", 0.95)),
        clip_range=float(ppo_section.get("clip_range", 0.2)),
        ent_coef=float(ppo_section.get("ent_coef", 0.0)),
        vf_coef=float(ppo_section.get("vf_coef", 0.5)),
        max_grad_norm=float(ppo_section.get("max_grad_norm", 0.5)),
        seed=int(ppo_section.get("seed", project_section.get("seed", 0))),
        verbose=0,
        device=_sb3_device(source_config),
        target_kl=None,
    )


def run_interface_smoke(
    config: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """执行不训练的最小接口检查，供环境准备阶段复用。"""

    prepared = prepare_ppo_config(config)
    environment = build_ppo_environment(prepared)
    try:
        check_status = sb3_check_env(environment)
        if check_status != "PASS":
            raise RuntimeError(f"Stable-Baselines3 check_env 未通过：{check_status}")

        model = build_sb3_ppo(environment, prepared)
        seed = int(_section(prepared, "ppo").get("seed", 0))
        observation, _reset_info = environment.reset(seed=seed)
        action, _state = model.predict(observation, deterministic=True)
        action_value = int(np.asarray(action).reshape(-1)[0])
        next_observation, reward, terminated, truncated, _step_info = environment.step(
            action_value
        )
        if observation.shape != (8,) or observation.dtype != np.dtype(np.float32):
            raise AssertionError("PPO observation 必须是 shape=(8,) 的 float32。")
        if not np.all(np.isfinite(observation)) or not np.all(
            np.isfinite(next_observation)
        ):
            raise AssertionError("PPO observation 不能包含 NaN/Inf。")
        if not np.isfinite(float(reward)):
            raise AssertionError("PPO reward 必须是有限值。")
        return {
            "check_env": check_status,
            "action_space": str(environment.action_space),
            "observation_shape": tuple(int(item) for item in observation.shape),
            "observation_dtype": str(observation.dtype),
            "action": action_value,
            "reward_finite": True,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "next_observation_finite": True,
            "model_device": str(model.device),
            "model_learn_called": 0,
        }
    finally:
        environment.close()


def _new_metric_accumulator() -> dict[str, Any]:
    return {
        "steps": 0,
        "reward": 0.0,
        "action_counts": {"KEEP": 0, "UPDATE": 0},
        "ppo_effective_update_count": 0,
        "ppo_update_noop_count": 0,
        "ppo_action_mismatch_count": 0,
        "cache_changed_slots": 0,
        "cache_changed_services": 0,
        "cache_switch_cost": 0.0,
        "uav_moved_slots": 0,
        "uav_move_cost": 0.0,
        "execution_counts": {
            "local": 0,
            "service_vehicle": 0,
            "rsu": 0,
            "uav": 0,
            "cloud": 0,
            "failed": 0,
        },
        "failed_tasks": 0,
        "future_request_leakage": 0,
        "future_trajectory_leakage": 0,
        "capacity_violations": 0,
        "compute_violations": 0,
        "bandwidth_violations": 0,
        "uav_violations": 0,
        "invalid_service_ids": 0,
        "invalid_node_ids": 0,
        "nan_inf_count": 0,
        "kkt_failure_count": 0,
        "total_delay": 0.0,
        "total_energy": 0.0,
    }


def _accumulate_step(
    accumulator: dict[str, Any],
    *,
    action: int | None,
    reward: float,
    info: Mapping[str, Any],
    observation: Any | None = None,
) -> None:
    """汇总一个 slot 的真实 reward、动作和环境诊断字段。"""

    accumulator["steps"] += 1
    accumulator["reward"] += float(reward)
    if action is not None:
        action_name = "UPDATE" if int(action) == 1 else "KEEP"
        accumulator["action_counts"][action_name] += 1
    cache_changed = np.asarray(info.get("cache_state_changed", False), dtype=bool)
    if bool(np.any(cache_changed)):
        accumulator["ppo_effective_update_count"] += 1
    accumulator["ppo_update_noop_count"] += int(info.get("ppo_update_noop", False))
    accumulator["ppo_action_mismatch_count"] += int(
        info.get("ppo_action_mismatch", 0)
    )
    accumulator["cache_changed_slots"] += int(info.get("cache_changed_slots", 0))
    accumulator["cache_changed_services"] += int(
        info.get("cache_changed_services", 0)
    )
    accumulator["cache_switch_cost"] += float(info.get("cache_cost", 0.0))
    accumulator["uav_moved_slots"] += int(info.get("uav_moved_slots", 0))
    accumulator["uav_move_cost"] += float(info.get("uav_move_cost", 0.0))
    execution_counts = info.get("execution_counts", {})
    if isinstance(execution_counts, Mapping):
        for key in accumulator["execution_counts"]:
            accumulator["execution_counts"][key] += int(execution_counts.get(key, 0))
    accumulator["failed_tasks"] += int(info.get("failed_tasks", 0))
    accumulator["future_request_leakage"] += int(
        info.get("future_request_leakage", 0)
    )
    accumulator["future_trajectory_leakage"] += int(
        info.get("future_trajectory_leakage", 0)
    )
    accumulator["capacity_violations"] += int(info.get("capacity_violations", 0))
    accumulator["compute_violations"] += int(info.get("compute_violations", 0))
    accumulator["bandwidth_violations"] += int(
        info.get("bandwidth_violations", 0)
    )
    accumulator["uav_violations"] += int(info.get("uav_violations", 0))
    accumulator["invalid_service_ids"] += int(
        info.get("invalid_service_ids", 0)
    )
    accumulator["invalid_node_ids"] += int(info.get("invalid_node_ids", 0))
    accumulator["kkt_failure_count"] += int(info.get("kkt_failure_count", 0))
    accumulator["total_delay"] += float(info.get("total_delay", 0.0))
    accumulator["total_energy"] += float(info.get("total_energy", 0.0))
    finite = np.isfinite(float(reward))
    if observation is not None:
        finite = bool(finite and np.all(np.isfinite(np.asarray(observation))))
    accumulator["nan_inf_count"] += int(
        (not finite) or int(info.get("nan_inf_count", 0)) > 0
    )


def _finalize_accumulator(accumulator: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(accumulator))
    result["mean_reward"] = (
        float(result["reward"]) / int(result["steps"])
        if int(result["steps"])
        else 0.0
    )
    result["task_total"] = int(sum(result["execution_counts"].values()))
    result["system_reward"] = float(result["reward"])
    result["mean_delay"] = (
        float(result["total_delay"]) / int(result["task_total"])
        if int(result["task_total"])
        else 0.0
    )
    result["mean_energy"] = (
        float(result["total_energy"]) / int(result["task_total"])
        if int(result["task_total"])
        else 0.0
    )
    result["cloud_rate"] = (
        float(result["execution_counts"]["cloud"]) / int(result["task_total"])
        if int(result["task_total"])
        else 0.0
    )
    return result


def _accumulator_delta(
    current: Mapping[str, Any],
    previous: Mapping[str, Any],
) -> dict[str, Any]:
    """计算相邻 PPO update 的动作/环境累计量差值。"""

    result = _new_metric_accumulator()
    scalar_keys = (
        "steps",
        "reward",
        "ppo_effective_update_count",
        "ppo_update_noop_count",
        "ppo_action_mismatch_count",
        "cache_changed_slots",
        "cache_changed_services",
        "cache_switch_cost",
        "uav_moved_slots",
        "uav_move_cost",
        "failed_tasks",
        "future_request_leakage",
        "future_trajectory_leakage",
        "capacity_violations",
        "compute_violations",
        "bandwidth_violations",
        "uav_violations",
        "invalid_service_ids",
        "invalid_node_ids",
        "nan_inf_count",
        "kkt_failure_count",
        "total_delay",
        "total_energy",
    )
    for key in scalar_keys:
        result[key] = current[key] - previous[key]
    for key in result["action_counts"]:
        result["action_counts"][key] = (
            current["action_counts"][key] - previous["action_counts"][key]
        )
    for key in result["execution_counts"]:
        result["execution_counts"][key] = (
            current["execution_counts"][key] - previous["execution_counts"][key]
        )
    return _finalize_accumulator(result)


def _new_online_scalar_stats() -> dict[str, Any]:
    """创建不保存 transition 的单变量在线统计器。"""

    return {
        "count": 0,
        "sum": 0.0,
        "squared_sum": 0.0,
        "min": math.inf,
        "max": -math.inf,
        "nonfinite_count": 0,
    }


def _update_online_scalar_stats(stats: dict[str, Any], value: float) -> None:
    value = float(value)
    if not math.isfinite(value):
        stats["nonfinite_count"] += 1
        return
    stats["count"] += 1
    stats["sum"] += value
    stats["squared_sum"] += value * value
    stats["min"] = min(float(stats["min"]), value)
    stats["max"] = max(float(stats["max"]), value)


def _finalize_online_scalar_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    count = int(stats["count"])
    if count <= 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "nonfinite_count": int(stats["nonfinite_count"]),
        }
    mean = float(stats["sum"]) / count
    variance = max(float(stats["squared_sum"]) / count - mean * mean, 0.0)
    return {
        "count": count,
        "mean": mean,
        "std": float(math.sqrt(variance)),
        "min": float(stats["min"]),
        "max": float(stats["max"]),
        "nonfinite_count": int(stats["nonfinite_count"]),
    }


class ControlledTrainingCallback(BaseCallback):
    """SB3 callback，记录最小必要的动作/环境诊断，不记录内部 tensor。"""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.aggregate = _new_metric_accumulator()
        self.current = _new_metric_accumulator()
        self.episodes: list[dict[str, Any]] = []
        self.begin_rollout_diagnostics()

    def begin_rollout_diagnostics(self) -> None:
        """只重置当前 rollout 的在线统计，不保存 transition。"""

        feature_count = len(PPO_CACHE_FEATURE_NAMES)
        self._feature_count = 0
        self._feature_sum = np.zeros(feature_count, dtype=float)
        self._feature_squared_sum = np.zeros(feature_count, dtype=float)
        self._feature_min = np.full(feature_count, np.inf, dtype=float)
        self._feature_max = np.full(feature_count, -np.inf, dtype=float)
        self._raw_reward_stats = _new_online_scalar_stats()
        self._normalized_reward_stats = _new_online_scalar_stats()
        self.last_rollout_feature_stats: dict[str, Any] | None = None

    def finish_rollout_diagnostics(self) -> dict[str, Any]:
        """完成一行 rollout 状态范围统计；不把原始状态写入结果文件。"""

        if self._feature_count <= 0:
            raise RuntimeError("当前 PPO rollout 没有收到原始8维状态。")
        count = float(self._feature_count)
        mean = self._feature_sum / count
        variance = np.maximum(self._feature_squared_sum / count - mean**2, 0.0)
        result = {
            name: {
                "mean": float(mean[index]),
                "std": float(np.sqrt(variance[index])),
                "min": float(self._feature_min[index]),
                "max": float(self._feature_max[index]),
            }
            for index, name in enumerate(PPO_CACHE_FEATURE_NAMES)
        }
        self.last_rollout_feature_stats = result
        return result

    def finish_rollout_reward_diagnostics(self) -> dict[str, Any]:
        """返回 raw/system reward 与 PPO normalized reward 的在线统计。"""

        return {
            "raw": _finalize_online_scalar_stats(self._raw_reward_stats),
            "normalized": _finalize_online_scalar_stats(
                self._normalized_reward_stats
            ),
        }

    def _on_step(self) -> bool:
        actions = np.asarray(self.locals.get("actions", [0])).reshape(-1)
        rewards = np.asarray(self.locals.get("rewards", [0.0])).reshape(-1)
        dones = np.asarray(self.locals.get("dones", [False])).reshape(-1)
        infos = self.locals.get("infos", [])
        new_obs = self.locals.get("new_obs")
        for index, info in enumerate(infos):
            if not isinstance(info, Mapping):
                info = {}
            action = int(actions[min(index, len(actions) - 1)])
            normalized_reward = float(rewards[min(index, len(rewards) - 1)])
            raw_reward = float(info.get("ppo_raw_reward", normalized_reward))
            _update_online_scalar_stats(
                self._raw_reward_stats,
                raw_reward,
            )
            _update_online_scalar_stats(
                self._normalized_reward_stats,
                normalized_reward,
            )
            observation = None
            if new_obs is not None:
                observation = np.asarray(new_obs)[min(index, len(new_obs) - 1)]
            raw_observation = info.get("ppo_raw_observation")
            if raw_observation is not None:
                raw_array = np.asarray(raw_observation, dtype=float)
                if raw_array.shape == (len(PPO_CACHE_FEATURE_NAMES),) and np.all(
                    np.isfinite(raw_array)
                ):
                    self._feature_count += 1
                    self._feature_sum += raw_array
                    self._feature_squared_sum += raw_array**2
                    self._feature_min = np.minimum(self._feature_min, raw_array)
                    self._feature_max = np.maximum(self._feature_max, raw_array)
            _accumulate_step(
                self.aggregate,
                action=action,
                reward=raw_reward,
                info=info,
                observation=observation,
            )
            _accumulate_step(
                self.current,
                action=action,
                reward=raw_reward,
                info=info,
                observation=observation,
            )
            if bool(dones[min(index, len(dones) - 1)]):
                episode = _finalize_accumulator(self.current)
                episode["episode_seed"] = info.get("episode_seed")
                self.episodes.append(episode)
                self.current = _new_metric_accumulator()
        return True


def _make_auxiliary_action(environment: MECEnvironment) -> dict[str, np.ndarray]:
    """构造固定底层辅助动作；PPO 只控制高层 KEEP/UPDATE。"""

    node_count = int(environment._cache.shape[0])
    service_count = int(environment._service_sizes.size)
    uav_count = int(environment._uav_positions.shape[0])
    return {
        "cache_triggers": np.ones(node_count, dtype=float),
        "cache_scores": np.full((node_count, service_count), 0.9, dtype=float),
        "uav_triggers": np.ones(uav_count, dtype=float),
        "uav_displacements": np.zeros((uav_count, 2), dtype=float),
    }


def _run_reference_episode(
    config: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """在 Reference 模式运行一个完整 episode，作为同 seed 对照。"""

    reference_config = copy.deepcopy(dict(config))
    cache_controller = reference_config.get("cache_controller", {})
    if not isinstance(cache_controller, Mapping):
        cache_controller = {}
    reference_config["cache_controller"] = dict(cache_controller)
    reference_config["cache_controller"]["mode"] = "reference"
    environment = MECEnvironment(reference_config)
    metrics = _new_metric_accumulator()
    try:
        environment.reset(seed=seed)
        action = _make_auxiliary_action(environment)
        for _ in range(int(reference_config["simulation"]["slot_count"])):
            previous_uav_positions = np.asarray(
                environment._uav_positions, dtype=float
            ).copy()
            _next_state, reward, done, info = environment.step(action)
            info = dict(info)
            info.update(
                runtime_diagnostics(
                    environment,
                    previous_uav_positions=previous_uav_positions,
                    reward=reward,
                    info=info,
                )
            )
            _accumulate_step(metrics, action=None, reward=reward, info=info)
            if done:
                break
    finally:
        environment.close()
    return _finalize_accumulator(metrics)


def _run_reference_short_replay(
    config: Mapping[str, Any],
    *,
    seed: int = DEFAULT_TRAIN_SEED_BASE,
) -> dict[str, Any]:
    """复用既有固定配置做训练后的 20-slot Reference 回归。"""

    short_config = copy.deepcopy(dict(config))
    short_config["project"] = dict(_section(short_config, "project"))
    short_config["project"]["seed"] = int(seed)
    short_config["simulation"] = dict(_section(short_config, "simulation"))
    short_config["simulation"]["slot_count"] = 20
    short_config["simulation"]["optimization_end_s"] = float(
        short_config["simulation"]["optimization_begin_s"]
    ) + 20.0 * float(short_config["simulation"]["decision_interval_s"])
    short_config["services"] = dict(_section(short_config, "services"))
    short_config["services"]["count"] = 20
    short_config["uav"] = dict(_section(short_config, "uav"))
    short_config["uav"]["initial_positions_m"] = copy.deepcopy(
        short_config["uav"]["initial_positions_m"][:1]
    )
    return _run_reference_episode(short_config, seed=int(seed))


def _run_ppo_episode(
    model: Any,
    vector_environment: Any,
    *,
    expected_steps: int,
    reset_seed: int | None = None,
) -> dict[str, Any]:
    """使用已加载 checkpoint 做 deterministic held-out episode。"""

    metrics = _new_metric_accumulator()
    # SB3.load 可能在 set_env 时写入模型自身 seed；held-out 评估必须在
    # reset 前显式覆盖它，确保每一行确实使用指定的未见 seed。
    if reset_seed is not None:
        seed_method = getattr(vector_environment, "seed", None)
        if not callable(seed_method):
            raise RuntimeError("评估 VecEnv 不支持显式 seed。")
        seed_method(int(reset_seed))
    observation = vector_environment.reset()
    for _ in range(expected_steps):
        action, _state = model.predict(observation, deterministic=True)
        action_value = int(np.asarray(action).reshape(-1)[0])
        observation, rewards, dones, infos = vector_environment.step(action)
        info = infos[0] if infos else {}
        if not isinstance(info, Mapping):
            info = {}
        raw_reward = float(
            info.get(
                "ppo_raw_reward",
                float(np.asarray(rewards).reshape(-1)[0]),
            )
        )
        _accumulate_step(
            metrics,
            action=action_value,
            reward=raw_reward,
            info=info,
            observation=np.asarray(observation)[0],
        )
        if bool(np.asarray(dones).reshape(-1)[0]):
            break
    return _finalize_accumulator(metrics)


def _aggregate_evaluations(
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """汇总多 seed 的核心结果，并计算 PPO-Reference reward 差值。"""

    if not evaluations:
        raise ValueError("至少需要一个 held-out evaluation。")
    rewards = np.asarray([float(item["reward_delta"]) for item in evaluations])
    reference_rewards = np.asarray(
        [float(item["reference"]["system_reward"]) for item in evaluations]
    )
    ppo_rewards = np.asarray(
        [float(item["ppo"]["system_reward"]) for item in evaluations]
    )
    def mean_metric(path: tuple[str, str]) -> float:
        return float(np.mean([float(item[path[0]][path[1]]) for item in evaluations]))

    return {
        "seed_count": len(evaluations),
        "mean_reference_reward": float(np.mean(reference_rewards)),
        "mean_ppo_reward": float(np.mean(ppo_rewards)),
        "mean_reward_delta": float(np.mean(rewards)),
        "std_reward_delta": float(np.std(rewards)),
        "ppo_wins": int(np.sum(rewards > 0.0)),
        "reference": {
            "mean_delay": mean_metric(("reference", "mean_delay")),
            "mean_energy": mean_metric(("reference", "mean_energy")),
            "cloud_rate": mean_metric(("reference", "cloud_rate")),
            "cache_changed_slots": mean_metric(
                ("reference", "cache_changed_slots")
            ),
            "cache_switch_cost": mean_metric(
                ("reference", "cache_switch_cost")
            ),
            "uav_moved_slots": mean_metric(("reference", "uav_moved_slots")),
            "uav_move_cost": mean_metric(("reference", "uav_move_cost")),
        },
        "ppo": {
            "mean_delay": mean_metric(("ppo", "mean_delay")),
            "mean_energy": mean_metric(("ppo", "mean_energy")),
            "cloud_rate": mean_metric(("ppo", "cloud_rate")),
            "cache_changed_slots": mean_metric(("ppo", "cache_changed_slots")),
            "cache_changed_services": mean_metric(
                ("ppo", "cache_changed_services")
            ),
            "cache_switch_cost": mean_metric(("ppo", "cache_switch_cost")),
            "uav_moved_slots": mean_metric(("ppo", "uav_moved_slots")),
            "uav_move_cost": mean_metric(("ppo", "uav_move_cost")),
        },
    }


def _collect_representative_states(
    config: Mapping[str, Any],
    *,
    seed: int,
    max_slots: int,
) -> list[dict[str, Any]]:
    """从一个训练 seed 的自然状态中挑选 3--5 个代表状态。"""

    environment = build_ppo_environment(config, episode_seed_base=seed)
    records: list[dict[str, Any]] = []
    try:
        observation, info = environment.reset(seed=seed)
        for slot in range(max_slots):
            preview = info.get("ppo_preview", {}) if isinstance(info, Mapping) else {}
            delta = float(preview.get("ppo_delta_predicted_utility", 0.0))
            records.append(
                {
                    "slot": int(slot),
                    "observation": np.asarray(observation, dtype=np.float32).copy(),
                    "episode_progress": float(np.asarray(observation)[2]),
                    "delta_predicted_utility": delta,
                }
            )
            observation, _reward, done, _truncated, info = environment.step(0)
            if done:
                break
    finally:
        environment.close()

    if not records:
        raise RuntimeError("无法收集 PPO 代表状态。")

    # 新状态的 probe 按 episode 起点、中段和末段均匀抽取；delta_J 仅作为
    # 诊断元数据保留，不再参与 observation，也不再作为状态类别标签。
    chosen = [
        int(index)
        for index in np.linspace(
            0,
            len(records) - 1,
            num=min(5, len(records)),
            dtype=int,
        )
    ]
    return [
        {
            "state": (
                f"slot_{records[index]['slot']}_progress_"
                f"{float(records[index]['episode_progress']):.3f}"
            ),
            "slot": int(records[index]["slot"]),
            "observation": records[index]["observation"],
            "episode_progress": float(records[index]["episode_progress"]),
            "delta_predicted_utility": float(
                records[index]["delta_predicted_utility"]
            ),
        }
        for index in chosen
    ]


def _policy_probabilities(
    model: Any,
    vector_environment: Any,
    observations: Sequence[np.ndarray],
    state_dict: Mapping[str, Any],
) -> list[dict[str, float]]:
    """在同一归一化状态上读取训练前/后的 Discrete(2) 策略概率。"""

    saved_state = copy.deepcopy(model.policy.state_dict())
    saved_training = bool(vector_environment.training)
    try:
        model.policy.load_state_dict(state_dict)
        vector_environment.training = False
        raw = np.asarray(observations, dtype=np.float32)
        normalized = vector_environment.normalize_obs(raw.copy())
        observation_tensor, _ = model.policy.obs_to_tensor(normalized)
        distribution = model.policy.get_distribution(observation_tensor).distribution
        probabilities = distribution.probs.detach().cpu().numpy()
        if probabilities.shape != (len(observations), 2) or not np.all(
            np.isfinite(probabilities)
        ):
            raise RuntimeError("PPO action probabilities 不是有限的 (N,2) 数组。")
        return [
            {
                "P_KEEP": float(row[0]),
                "P_UPDATE": float(row[1]),
            }
            for row in probabilities
        ]
    finally:
        model.policy.load_state_dict(saved_state)
        vector_environment.training = saved_training


def _policy_snapshot(
    model: Any,
    vector_environment: Any,
    states: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """读取固定 raw probe state 上的 P(KEEP)、P(UPDATE) 和 V(s)。"""

    saved_training = bool(vector_environment.training)
    try:
        vector_environment.training = False
        raw = np.asarray(
            [np.asarray(state["observation"], dtype=np.float32) for state in states],
            dtype=np.float32,
        )
        normalized = vector_environment.normalize_obs(raw.copy())
        observation_tensor, _ = model.policy.obs_to_tensor(normalized)
        distribution = model.policy.get_distribution(observation_tensor).distribution
        probabilities = distribution.probs.detach().cpu().numpy()
        values = model.policy.predict_values(observation_tensor).detach().cpu().numpy()
        values = np.asarray(values, dtype=float).reshape(-1)
        if probabilities.shape != (len(states), 2) or values.shape != (len(states),):
            raise RuntimeError("probe state 的 PPO 输出形状不正确。")
        if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(values)):
            raise RuntimeError("probe state 的策略概率或 V(s) 包含 NaN/Inf。")
        return [
            {
                "state": str(state["state"]),
                "slot": int(state["slot"]),
                "episode_progress": float(state["episode_progress"]),
                "delta_predicted_utility": float(
                    state["delta_predicted_utility"]
                ),
                "P_KEEP": float(probabilities[index, 0]),
                "P_UPDATE": float(probabilities[index, 1]),
                "V": float(values[index]),
            }
            for index, state in enumerate(states)
        ]
    finally:
        vector_environment.training = saved_training


def _json_safe(value: Any) -> Any:
    """把配置/诊断中的 numpy 标量和路径转换成 JSON 基础类型。"""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _finite_metrics(logger_values: Mapping[str, Any]) -> dict[str, float]:
    """提取 SB3 最小核心指标，并拒绝 NaN/Inf。"""

    result: dict[str, float] = {}
    for name in REQUIRED_PPO_METRICS:
        key = f"train/{name}"
        if key not in logger_values:
            raise RuntimeError(f"SB3 日志缺少核心指标：{key}")
        value = float(logger_values[key])
        if not math.isfinite(value):
            raise RuntimeError(f"SB3 指标 {key} 为 NaN/Inf。")
        result[key] = value
    for key in ("rollout/ep_rew_mean", "rollout/ep_len_mean"):
        if key in logger_values:
            value = float(logger_values[key])
            if not math.isfinite(value):
                raise RuntimeError(f"SB3 指标 {key} 为 NaN/Inf。")
            result[key] = value
    return result


def _pearson_or_undefined(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[float | None, str]:
    """计算 Pearson；任一方无方差时返回 None，禁止 NaN 进入诊断结果。"""

    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    if left.size < 2 or right.size != left.size:
        return None, "undefined_insufficient_samples"
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std <= 1.0e-12 or right_std <= 1.0e-12:
        return None, "undefined_zero_variance"
    value = float(np.corrcoef(left, right)[0, 1])
    if not math.isfinite(value):
        return None, "undefined_nonfinite"
    return value, "finite"


def _critic_rollout_diagnostics(
    model: Any,
    *,
    sb3_metrics: Mapping[str, Any],
    reward_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """从 SB3 rollout buffer 提取一行 Critic 诊断，不复制完整 transition。"""

    buffer = getattr(model, "rollout_buffer", None)
    if buffer is None:
        raise RuntimeError("SB3 rollout buffer 不可用，无法进行 Critic 诊断。")
    rewards = np.asarray(buffer.rewards, dtype=float).reshape(-1)
    returns = np.asarray(buffer.returns, dtype=float).reshape(-1)
    values = np.asarray(buffer.values, dtype=float).reshape(-1)
    if rewards.size == 0 or returns.size != values.size or returns.size != rewards.size:
        raise RuntimeError("rollout buffer 的 reward/return/value 形状不一致。")
    if not (
        np.all(np.isfinite(rewards))
        and np.all(np.isfinite(returns))
        and np.all(np.isfinite(values))
    ):
        raise RuntimeError("rollout buffer 的 reward/return/value 包含 NaN/Inf。")
    pearson, pearson_status = _pearson_or_undefined(values, returns)
    error = values - returns
    normalized_stats = (
        reward_diagnostics.get("normalized", {})
        if isinstance(reward_diagnostics, Mapping)
        else {}
    )
    raw_stats = (
        reward_diagnostics.get("raw", {})
        if isinstance(reward_diagnostics, Mapping)
        else {}
    )
    normalized_reward_mean = float(
        normalized_stats.get("mean")
        if normalized_stats.get("mean") is not None
        else np.mean(rewards)
    )
    normalized_reward_std = float(
        normalized_stats.get("std")
        if normalized_stats.get("std") is not None
        else np.std(rewards)
    )
    normalized_reward_min = float(
        normalized_stats.get("min")
        if normalized_stats.get("min") is not None
        else np.min(rewards)
    )
    normalized_reward_max = float(
        normalized_stats.get("max")
        if normalized_stats.get("max") is not None
        else np.max(rewards)
    )
    raw_reward_mean = (
        raw_stats.get("mean")
        if raw_stats.get("mean") is not None
        else float(np.mean(rewards))
    )
    raw_reward_std = (
        raw_stats.get("std")
        if raw_stats.get("std") is not None
        else float(np.std(rewards))
    )
    return_std = float(np.std(returns))
    return {
        # reward_mean/std 保留为兼容字段；新增显式字段区分两种尺度。
        "reward_mean": float(raw_reward_mean),
        "reward_std": float(raw_reward_std),
        "raw_reward_mean": raw_stats.get("mean"),
        "raw_reward_std": raw_stats.get("std"),
        "raw_reward_min": raw_stats.get("min"),
        "raw_reward_max": raw_stats.get("max"),
        "normalized_reward_mean": normalized_reward_mean,
        "normalized_reward_std": normalized_reward_std,
        "normalized_reward_min": normalized_reward_min,
        "normalized_reward_max": normalized_reward_max,
        "return_mean": float(np.mean(returns)),
        "return_std": return_std,
        "value_mean": float(np.mean(values)),
        "value_std": float(np.std(values)),
        "value_return_rmse": float(np.sqrt(np.mean(error**2))),
        "value_return_pearson": pearson,
        "value_return_pearson_status": pearson_status,
        "value_return_scale_ratio": float(
            np.std(values) / (return_std + 1.0e-12)
        ),
        "explained_variance": float(sb3_metrics["train/explained_variance"]),
        "value_loss": float(sb3_metrics["train/value_loss"]),
    }


def _new_run_dir(config: Mapping[str, Any]) -> Path:
    output_section = _section(config, "output")
    root = PROJECT_ROOT / str(output_section.get("root", "outputs")) / "ppo_training"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / stamp
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{stamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir()
    return run_dir


def _make_training_vector_environment(
    config: Mapping[str, Any],
    *,
    seed_base: int,
    norm_reward: bool | None = None,
) -> Any:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

    raw = DummyVecEnv(
        [
            lambda: build_ppo_environment(
                config,
                episode_seed_base=seed_base,
            )
        ]
    )
    monitored = VecMonitor(raw)
    if norm_reward is None:
        norm_reward = bool(_section(config, "ppo").get("norm_reward", False))
    ppo_section = _section(config, "ppo")
    return VecNormalize(
        monitored,
        norm_obs=bool(ppo_section.get("norm_obs", True)),
        norm_reward=bool(norm_reward),
        clip_obs=float(ppo_section.get("clip_obs", 10.0)),
    )


def _make_evaluation_vector_environment(
    config: Mapping[str, Any],
    *,
    seed_base: int,
    vecnormalize_path: Path,
) -> Any:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

    raw = DummyVecEnv(
        [
            lambda: build_ppo_environment(
                config,
                seed_mode="fixed",
                fixed_seed=seed_base,
            )
        ]
    )
    monitored = VecMonitor(raw)
    normalized = VecNormalize.load(str(vecnormalize_path), monitored)
    normalized.training = False
    # held-out evaluation always返回原始 system reward；即使训练快照保存了
    # norm_reward=True，也不能让归一化尺度污染系统性能比较。
    normalized.norm_reward = False
    return normalized


STAGE15_HARD_CONDITION_KEYS = (
    "future_request_leakage",
    "future_trajectory_leakage",
    "capacity_violations",
    "compute_violations",
    "bandwidth_violations",
    "uav_violations",
    "invalid_service_ids",
    "invalid_node_ids",
    "nan_inf_count",
    "kkt_failure_count",
    "ppo_action_mismatch_count",
)


def _load_config_copy(config: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    """读取配置副本；Stage15 不在训练过程中改写项目配置。"""

    if isinstance(config, Mapping):
        return copy.deepcopy(dict(config))
    return load_config() if config is None else load_config(config)


def validate_stage15_configuration(
    config: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """只解析并校验 Stage15 参数，不创建环境、不调用 ``model.learn``。"""

    base_config = _load_config_copy(config)
    ppo_config = prepare_ppo_config(base_config)
    ppo_section = _section(ppo_config, "ppo")
    simulation = _section(ppo_config, "simulation")
    mobility = _section(ppo_config, "mobility")
    if str(mobility.get("source", "")).strip().lower() != "trace":
        raise ValueError("Stage15 formal training requires mobility.source=trace")
    if not str(mobility.get("trace_root", "")).strip():
        raise ValueError("Stage15 formal training requires mobility.trace_root")
    total_timesteps = int(ppo_section.get("total_timesteps", 0))
    n_steps = int(ppo_section.get("n_steps", 0))
    eval_freq = int(ppo_section.get("eval_freq", 0))
    if total_timesteps != 25600:
        raise ValueError("Stage15 要求 total_timesteps=25600。")
    if n_steps != 256 or total_timesteps % n_steps != 0:
        raise ValueError("Stage15 要求 n_steps=256 且 total_timesteps 可整除 n_steps。")
    if eval_freq != 5120 or eval_freq % n_steps != 0:
        raise ValueError("Stage15 要求 eval_freq=5120 且与 n_steps 对齐。")
    if int(simulation.get("slot_count", 0)) != 100:
        raise ValueError("Stage15 要求每个 episode 为 100 slots。")
    if not bool(ppo_section.get("norm_obs", False)):
        raise ValueError("Stage15 必须使用 norm_obs=true。")
    if not bool(ppo_section.get("norm_reward", False)):
        raise ValueError("Stage15 正式训练必须使用 norm_reward=true。")
    if abs(float(ppo_section.get("clip_obs", 10.0)) - 10.0) > 1.0e-12:
        raise ValueError("Stage15 要求 clip_obs=10.0。")
    expected_parameters = {
        "learning_rate": 3.0e-4,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }
    for name, expected in expected_parameters.items():
        actual = float(ppo_section.get(name, expected + 1.0))
        if abs(actual - expected) > 1.0e-12:
            raise ValueError(f"Stage15 参数 {name} 必须保持为 {expected}。")
    if _sb3_device(ppo_config).lower() != "cpu":
        raise ValueError("Stage15 正式配置固定使用 device=cpu。")
    if tuple(PPO_CACHE_FEATURE_NAMES) != (
        "J_keep",
        "J_update",
        "episode_progress",
        "candidate_cache_cost",
        "changed_service_ratio",
        "demand_shift",
        "cache_age",
        "mobility_index",
    ):
        raise ValueError("Stage15 observation 必须保持冻结的 8 维状态。")
    if int(len(PPO_CACHE_FEATURE_NAMES)) != 8:
        raise ValueError("Stage15 state_dim 必须为 8。")

    episodes_started = int(math.ceil(total_timesteps / int(simulation["slot_count"])))
    train_seed_base = int(ppo_section.get("train_seed_base", DEFAULT_TRAIN_SEED_BASE))
    train_seeds = [train_seed_base + index for index in range(episodes_started)]

    def seed_list(name: str, expected_count: int) -> list[int]:
        value = ppo_section.get(name)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"Stage15 缺少 {name} 列表。")
        result = [int(seed) for seed in value]
        if len(result) != expected_count or len(set(result)) != expected_count:
            raise ValueError(f"Stage15 {name} 必须包含 {expected_count} 个不重复 seed。")
        return result

    validation_seeds = seed_list("validation_seeds", 5)
    test_seeds = seed_list("test_seeds", 10)
    train_set = set(train_seeds)
    validation_set = set(validation_seeds)
    test_set = set(test_seeds)
    if train_set.intersection(validation_set) or train_set.intersection(test_set):
        raise ValueError("Stage15 training seed 不得与 validation/test seed 重叠。")
    if validation_set.intersection(test_set):
        raise ValueError("Stage15 validation seed 不得与 test seed 重叠。")
    checkpoint_updates = [
        int(update) for update in ppo_section.get("checkpoint_updates", [])
    ]
    expected_checkpoints = [20, 40, 60, 80, 100]
    if checkpoint_updates != expected_checkpoints:
        raise ValueError("Stage15 checkpoint_updates 必须为 [20,40,60,80,100]。")
    updates = total_timesteps // n_steps
    if updates != 100 or checkpoint_updates[-1] != updates:
        raise ValueError("Stage15 必须对应 100 个 PPO updates。")
    if str(ppo_section.get("best_model_criterion", "")) != (
        "mean_original_validation_system_reward"
    ):
        raise ValueError("Stage15 best model 必须按原始 validation system reward 选择。")
    expected_hard_stops = {
        "future_request_leakage",
        "future_trajectory_leakage",
        "cache_capacity_violations",
        "compute_violations",
        "bandwidth_violations",
        "uav_violations",
        "invalid_service_ids",
        "invalid_node_ids",
        "nan_inf_count",
        "kkt_failure_count",
        "action_mismatch",
    }
    configured_hard_stops = {
        str(item) for item in ppo_section.get("hard_stop_on", [])
    }
    if configured_hard_stops != expected_hard_stops:
        raise ValueError("Stage15 hard_stop_on 配置不完整或包含未声明项目。")
    if not bool(ppo_section.get("test_after_best_only", False)):
        raise ValueError("Stage15 test_after_best_only 必须为 true。")
    return {
        "config": ppo_config,
        "total_timesteps": total_timesteps,
        "n_steps": n_steps,
        "updates": updates,
        "eval_freq": eval_freq,
        "episodes_started": episodes_started,
        "train_seed_base": train_seed_base,
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "test_seeds": test_seeds,
        "checkpoint_updates": checkpoint_updates,
        "state_features": list(PPO_CACHE_FEATURE_NAMES),
    }


def _run_stage15_trace_preflight(
    config: Mapping[str, Any],
    *,
    preflight_seed: int,
) -> dict[str, Any]:
    """使用合法 training seed 做独立 Trace 环境轻量检查，不创建 PPO 模型。"""

    seed = int(preflight_seed)
    if seed == 0:
        raise ValueError("Stage15 Trace preflight 不允许使用 seed=0。")

    # SB3 官方 check_env 会主动 reset(seed=0)，有限离线 Trace 集合不包含该 seed。
    # 因此 Stage15 只用配置中的首个合法 training seed 做独立 reset/step 检查；
    # 通用 sb3_check_env 仍保留在 ppo_env.py，供开发接口测试使用。
    sumo_starts_before = sumo_start_count()
    environment = build_ppo_environment(config)
    observed_reset_seeds: list[int] = []
    step_count = 0
    try:
        observation, reset_info = environment.reset(seed=seed)
        del reset_info
        observed_reset_seeds.append(
            int(environment.last_episode_seed)
            if environment.last_episode_seed is not None
            else -1
        )
        observation_array = np.asarray(observation)
        observation_in_space = bool(environment.observation_space.contains(observation))
        if observation_array.shape != (8,):
            raise RuntimeError(
                f"Stage15 Trace preflight observation shape={observation_array.shape}, expected (8,)"
            )
        if observation_array.dtype != np.float32:
            raise RuntimeError(
                f"Stage15 Trace preflight observation dtype={observation_array.dtype}, expected float32"
            )
        if not np.all(np.isfinite(observation_array)):
            raise RuntimeError("Stage15 Trace preflight observation contains NaN/Inf")
        if not observation_in_space:
            raise RuntimeError("Stage15 Trace preflight observation is outside observation_space")
        if not environment.action_space.contains(0):
            raise RuntimeError("Stage15 Trace preflight action_space does not contain KEEP=0")

        step_result = environment.step(0)
        if not isinstance(step_result, tuple) or len(step_result) != 5:
            raise RuntimeError(
                "Stage15 Trace preflight step API must return obs, reward, terminated, truncated, info"
            )
        next_observation, reward, terminated, truncated, info = step_result
        step_count = 1
        next_observation_array = np.asarray(next_observation)
        if next_observation_array.shape != (8,) or next_observation_array.dtype != np.float32:
            raise RuntimeError("Stage15 Trace preflight next observation has invalid shape/dtype")
        if not np.all(np.isfinite(next_observation_array)):
            raise RuntimeError("Stage15 Trace preflight next observation contains NaN/Inf")
        if not environment.observation_space.contains(next_observation):
            raise RuntimeError("Stage15 Trace preflight next observation is outside observation_space")
        if not math.isfinite(float(reward)):
            raise RuntimeError("Stage15 Trace preflight reward contains NaN/Inf")
        if not isinstance(terminated, (bool, np.bool_)) or not isinstance(
            truncated, (bool, np.bool_)
        ):
            raise RuntimeError("Stage15 Trace preflight terminated/truncated must be bool")
        if not isinstance(info, Mapping):
            raise RuntimeError("Stage15 Trace preflight info must be a mapping")
    finally:
        environment.close()

    sumo_starts = sumo_start_count() - sumo_starts_before
    if sumo_starts != 0:
        raise RuntimeError(
            f"Stage15 Trace preflight unexpectedly started SUMO {sumo_starts} times"
        )
    if observed_reset_seeds != [seed]:
        raise RuntimeError(
            f"Stage15 Trace preflight reset seeds={observed_reset_seeds}, expected [{seed}]"
        )

    return {
        "official_check_env_skipped": True,
        "official_check_env_skipped_reason": (
            "offline_trace_environment_requires_prepared_seed_set"
        ),
        "preflight_seed": seed,
        "reset_seeds_observed": observed_reset_seeds,
        "seed_0_accesses": 0,
        "observation_shape": [8],
        "observation_dtype": "float32",
        "observation_finite": True,
        "observation_in_observation_space": True,
        "action_space": str(environment.action_space),
        "step_api": "obs,reward,terminated,truncated,info",
        "steps": step_count,
        "sumo_starts": int(sumo_starts),
        "independent_environment": True,
    }


def _save_model_vecnormalize_pair(
    model: Any,
    vector_environment: Any,
    checkpoint_dir: Path,
) -> None:
    """成对保存 model.zip 与 VecNormalize 统计，禁止混合 checkpoint。"""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint_dir / "model"))
    vector_environment.save(str(checkpoint_dir / "vecnormalize.pkl"))
    if not (checkpoint_dir / "model.zip").is_file() or not (
        checkpoint_dir / "vecnormalize.pkl"
    ).is_file():
        raise RuntimeError(f"checkpoint pair 保存失败：{checkpoint_dir}")


def _copy_model_vecnormalize_pair(source_dir: Path, target_dir: Path) -> None:
    """复制完整 checkpoint pair，用于 best/final 固化。"""

    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("model.zip", "vecnormalize.pkl"):
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"checkpoint pair 缺少 {source}")
        shutil.copy2(source, target_dir / name)


def _evaluate_stage15_checkpoint(
    config: Mapping[str, Any],
    checkpoint_dir: Path,
    seeds: Sequence[int],
    *,
    reference_cache: dict[int, dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    """在指定 split 上用 raw system reward 评估一个已保存的 checkpoint pair。"""

    from stable_baselines3 import PPO

    model_path = checkpoint_dir / "model.zip"
    vecnormalize_path = checkpoint_dir / "vecnormalize.pkl"
    if not model_path.is_file() or not vecnormalize_path.is_file():
        raise FileNotFoundError(f"{split} checkpoint pair 不完整：{checkpoint_dir}")
    simulation = _section(config, "simulation")
    expected_steps = int(simulation["slot_count"])
    rows: list[dict[str, Any]] = []
    for seed_value in seeds:
        seed = int(seed_value)
        if seed not in reference_cache:
            reference_cache[seed] = _run_reference_episode(config, seed=seed)
        reference_summary = reference_cache[seed]
        evaluation_environment = _make_evaluation_vector_environment(
            config,
            seed_base=seed,
            vecnormalize_path=vecnormalize_path,
        )
        try:
            loaded_model = PPO.load(
                str(model_path),
                env=evaluation_environment,
                device="cpu",
            )
            ppo_summary = _run_ppo_episode(
                loaded_model,
                evaluation_environment,
                expected_steps=expected_steps,
                reset_seed=seed,
            )
        finally:
            evaluation_environment.close()
        rows.append(
            {
                "seed": seed,
                "reference": reference_summary,
                "ppo": ppo_summary,
                "reward_delta": float(
                    ppo_summary["system_reward"]
                    - reference_summary["system_reward"]
                ),
            }
        )
    aggregate = _aggregate_evaluations(rows)
    return {
        "split": split,
        "checkpoint": checkpoint_dir.name,
        "eval_seeds": [int(seed) for seed in seeds],
        "per_seed": rows,
        "aggregate": aggregate,
        "criterion_value": float(aggregate["mean_ppo_reward"]),
        "criterion": "mean original validation system reward"
        if split == "validation"
        else "reported only after best checkpoint selection",
    }


def _assert_stage15_evaluation_hard_zero(
    evaluation: Mapping[str, Any],
    *,
    split: str,
) -> None:
    """validation/test 也必须满足所有硬正确性计数为 0。"""

    failures: dict[str, int] = {}
    for row in evaluation.get("per_seed", []):
        for controller in ("reference", "ppo"):
            metrics = row.get(controller, {})
            for key in STAGE15_HARD_CONDITION_KEYS:
                value = int(metrics.get(key, 0))
                if value:
                    failures[f"{split}:{row.get('seed')}:{controller}:{key}"] = value
    if failures:
        raise RuntimeError(f"Stage15 hard-stop in {split} evaluation: {failures}")


def _stage15_hard_condition_counts(
    aggregate: Mapping[str, Any],
    reward_diagnostics: Mapping[str, Any],
) -> dict[str, int]:
    counts = {
        key: int(aggregate.get(key, 0)) for key in STAGE15_HARD_CONDITION_KEYS
    }
    counts["nan_inf_count"] += sum(
        int(reward_diagnostics.get(scale, {}).get("nonfinite_count", 0))
        for scale in ("raw", "normalized")
        if isinstance(reward_diagnostics.get(scale, {}), Mapping)
    )
    return counts


def _stage15_display_value(value: Any, *, digits: int = 3) -> str:
    """将控制台数值安全格式化；暂不可用的诊断统一显示为 NA。"""

    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(numeric):
        return "NA"
    return f"{numeric:.{int(digits)}f}"


def _stage15_display_count(value: Any) -> str:
    """将动作/episode 计数安全格式化，缺失值不伪造为 0。"""

    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(numeric) or numeric < 0 or numeric != int(numeric):
        return "NA"
    return str(int(numeric))


def _stage15_trace_console_partition(settings: Mapping[str, Any]) -> dict[str, Any]:
    """按单 DummyVecEnv 的实际 rollout 边界计算启动信息中的 Trace 分区。"""

    config = settings["config"]
    simulation = _section(config, "simulation")
    total_timesteps = int(settings["total_timesteps"])
    n_steps = int(settings["n_steps"])
    horizon = int(simulation.get("slot_count", 0))
    effective_env_steps = int(math.ceil(total_timesteps / n_steps) * n_steps)
    formal_count = int(settings["episodes_started"])
    reserve_count = int(
        effective_env_steps > 0 and effective_env_steps % horizon == 0
    )
    reserve_seed = (
        int(settings["train_seed_base"]) + formal_count
        if reserve_count
        else None
    )
    return {
        "formal_training_count": formal_count,
        "training_reset_reserve_count": reserve_count,
        "training_reset_reserve_seed": reserve_seed,
        "effective_env_steps": effective_env_steps,
    }


def _stage15_startup_text(settings: Mapping[str, Any]) -> str:
    """生成 Stage15 正式训练前的固定摘要，不创建环境也不调用 learn。"""

    config = settings["config"]
    mobility = _section(config, "mobility")
    partition = _stage15_trace_console_partition(settings)
    lines = [
        "===== Stage15 PPO Training =====",
        "",
        f"mobility source = {str(mobility.get('source', 'trace')).lower()}",
        f"formal training traces = {partition['formal_training_count']}",
        f"training reset reserve = {partition['training_reset_reserve_count']}",
        f"validation traces = {len(settings['validation_seeds'])}",
        f"test traces = {len(settings['test_seeds'])}",
        "",
        f"state_dim = {len(settings['state_features'])}",
        "norm_obs = true",
        "norm_reward = true",
        "",
        f"total_timesteps = {settings['total_timesteps']}",
        f"n_steps = {settings['n_steps']}",
        f"expected PPO updates = {settings['updates']}",
        f"episode horizon = {_section(config, 'simulation').get('slot_count')}",
        "",
        f"device = {_sb3_device(config)}",
        "",
        f"validation interval = {settings['eval_freq']}",
        "best criterion = mean original validation system reward",
        "",
        "formal training starting...",
    ]
    return "\n".join(lines)


def _print_stage15_startup(settings: Mapping[str, Any]) -> None:
    """打印 Stage15 启动摘要；flush 保证长训练日志实时可见。"""

    print(_stage15_startup_text(settings), flush=True)


def _stage15_update_text(
    record: Mapping[str, Any],
    *,
    completed_episodes: int,
    expected_updates: int,
    recent_raw_reward: Any,
) -> str:
    """生成单行 PPO update 进度，严格区分 raw_reward 与内部 normalized reward。"""

    metrics = record.get("metrics", {})
    critic = record.get("critic", {})
    update = record.get("update", {})
    action_counts = update.get("action_counts", {}) if isinstance(update, Mapping) else {}
    explained_variance = critic.get("explained_variance")
    if explained_variance is None and isinstance(metrics, Mapping):
        explained_variance = metrics.get("explained_variance")
    approx_kl = metrics.get("approx_kl") if isinstance(metrics, Mapping) else None
    pearson = critic.get("value_return_pearson") if isinstance(critic, Mapping) else None
    return (
        f"Update {_stage15_display_count(record.get('update_index'))}/{int(expected_updates)}"
        f" | steps={_stage15_display_count(record.get('timesteps'))}"
        f" | episodes={_stage15_display_count(completed_episodes)}"
        f" | raw_reward={_stage15_display_value(recent_raw_reward)}"
        f" | EV={_stage15_display_value(explained_variance)}"
        f" | Pearson={_stage15_display_value(pearson)}"
        f" | KL={_stage15_display_value(approx_kl)}"
        f" | KEEP={_stage15_display_count(action_counts.get('KEEP'))}"
        f" | UPDATE={_stage15_display_count(action_counts.get('UPDATE'))}"
        f" | effective UPDATE={_stage15_display_count(update.get('ppo_effective_update_count'))}"
        f" | noop={_stage15_display_count(update.get('ppo_update_noop_count'))}"
    )


def _print_stage15_update(
    record: Mapping[str, Any],
    *,
    completed_episodes: int,
    expected_updates: int,
    recent_raw_reward: Any,
) -> None:
    """打印单个 update 的诊断摘要，不打印 slot 级日志。"""

    print(
        _stage15_update_text(
            record,
            completed_episodes=completed_episodes,
            expected_updates=expected_updates,
            recent_raw_reward=recent_raw_reward,
        ),
        flush=True,
    )


def _stage15_validation_text(
    validation: Mapping[str, Any],
    *,
    update_index: int,
    current_best_update: int | None,
    new_best: bool,
) -> str:
    """生成 validation 控制台摘要；详细结果仍完整写入 JSON。"""

    aggregate = validation.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        aggregate = {}
    lines = [
        f"----- Validation @ Update {int(update_index)} -----",
        "Reference mean raw reward = "
        f"{_stage15_display_value(aggregate.get('mean_reference_reward'))}",
        "PPO mean raw reward       = "
        f"{_stage15_display_value(aggregate.get('mean_ppo_reward'))}",
        "Delta                     = "
        f"{_stage15_display_value(aggregate.get('mean_reward_delta'))}",
        "Current best update       = "
        f"{_stage15_display_count(current_best_update)}",
    ]
    per_seed = validation.get("per_seed", [])
    if isinstance(per_seed, Sequence) and not isinstance(per_seed, (str, bytes)):
        for row in per_seed:
            if not isinstance(row, Mapping):
                continue
            reference = row.get("reference", {})
            ppo = row.get("ppo", {})
            if not isinstance(reference, Mapping):
                reference = {}
            if not isinstance(ppo, Mapping):
                ppo = {}
            lines.append(
                f"VAL seed={row.get('seed', 'NA')}"
                f" | Ref={_stage15_display_value(reference.get('system_reward'))}"
                f" | PPO={_stage15_display_value(ppo.get('system_reward'))}"
                f" | Delta={_stage15_display_value(row.get('reward_delta'))}"
            )
    if new_best:
        lines.append("NEW BEST CHECKPOINT")
    return "\n".join(lines)


def _print_stage15_validation(
    validation: Mapping[str, Any],
    *,
    update_index: int,
    current_best_update: int | None,
    new_best: bool,
) -> None:
    """打印 checkpoint validation 摘要并立即 flush。"""

    print(
        _stage15_validation_text(
            validation,
            update_index=update_index,
            current_best_update=current_best_update,
            new_best=new_best,
        ),
        flush=True,
    )


def _print_stage15_hard_stop(reason: Any, *, update: Any, timesteps: Any) -> None:
    """在异常真正抛出前输出可定位的 hard-stop 原因。"""

    print(f"HARD STOP: {reason}", flush=True)
    print(f"update = {_stage15_display_count(update)}", flush=True)
    print(f"timesteps = {_stage15_display_count(timesteps)}", flush=True)


def _stage15_test_execution_rates(evaluation: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """为最终控制台摘要计算总任务分母下的 Reference/PPO 执行率。"""

    keys = ("local", "service_vehicle", "rsu", "uav", "cloud", "failed")
    result: dict[str, dict[str, float]] = {}
    rows = evaluation.get("per_seed", [])
    for controller in ("reference", "ppo"):
        counts = {key: 0 for key in keys}
        total_tasks = 0
        for row in rows:
            metrics = row.get(controller, {})
            execution = metrics.get("execution_counts", {})
            for key in keys:
                if key == "failed" and key not in execution:
                    value = metrics.get("failed_tasks", 0)
                else:
                    value = execution.get(key, 0)
                counts[key] += int(value)
            total_tasks += int(metrics.get("task_total", sum(counts.values())))
        if total_tasks <= 0:
            raise ValueError(f"Stage15 test {controller} total task count is zero")
        result[controller] = {
            key: 100.0 * float(value) / float(total_tasks)
            for key, value in counts.items()
        }
    return result


def _stage15_hard_stop_violation_total(summary: Mapping[str, Any]) -> int:
    """汇总训练和最终 test 的硬约束违规数，供成功摘要展示。"""

    correctness = summary.get("correctness", {})
    total = 0
    if isinstance(correctness, Mapping):
        training_conditions = correctness.get("training_hard_conditions", {})
        if isinstance(training_conditions, Mapping):
            total += sum(int(value) for value in training_conditions.values())
    test_summary = summary.get("test", {})
    if isinstance(test_summary, Mapping):
        for row in test_summary.get("per_seed", []):
            if not isinstance(row, Mapping):
                continue
            for controller in ("reference", "ppo"):
                metrics = row.get(controller, {})
                if isinstance(metrics, Mapping):
                    total += sum(
                        int(metrics.get(key, 0)) for key in STAGE15_HARD_CONDITION_KEYS
                    )
    return int(total)


def _generate_stage15_figures(
    run_dir: Path,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """从已落盘 summary 后处理绘图；绘图失败不回滚训练产物。"""

    del summary  # 重新从磁盘读，确保绘图只依赖已保存产物。
    result: dict[str, Any] = {"status": "FAIL", "files": {}}
    result: dict[str, Any] = {"status": "FAIL", "files": {}}
    try:
        saved_summary_path = run_dir / "training_summary.json"
        with saved_summary_path.open("r", encoding="utf-8") as handle:
            saved_summary = json.load(handle)
        from plot_results import plot_stage15_run

        figure_paths = plot_stage15_run(saved_summary, run_dir / "figures")
        files: dict[str, list[str]] = {}
        for name, paths in figure_paths.items():
            checked: list[str] = []
            for path_value in paths:
                path = Path(path_value)
                if not path.is_file() or path.stat().st_size <= 0:
                    raise RuntimeError(f"Stage15 figure 未生成或为空: {path}")
                checked.append(str(path.relative_to(run_dir)))
            files[name] = checked
        result = {"status": "PASS", "files": files}
    except Exception as error:  # 绘图是后处理，不能覆盖已完成训练。
        result["error"] = repr(error)
        result["error"] = repr(error)
    return result


def run_stage15_training(
    config: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """执行 Stage15 的 25600-step 正式 PPO 训练；只由显式 CLI 命令调用。"""

    settings = validate_stage15_configuration(config)
    ppo_config = settings["config"]
    trace_preflight = _run_stage15_trace_preflight(
        ppo_config,
        preflight_seed=int(settings["train_seeds"][0]),
    )
    # 启动摘要放在 Trace preflight 之后、正式环境创建之前；本摘要不调用 model.learn。
    _print_stage15_startup(settings)
    mobility_sumo_starts_before = sumo_start_count()
    run_dir = _new_run_dir(ppo_config)
    vector_environment = None
    reference_cache: dict[int, dict[str, Any]] = {}
    update_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    best_checkpoint_dir: Path | None = None
    best_update: int | None = None
    best_validation_reward = -math.inf
    callback_state: ControlledTrainingCallback | None = None
    try:
        with (run_dir / "config_snapshot.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(ppo_config), handle, ensure_ascii=False, indent=2)

        # Stage15 Trace 已在独立合法 seed 上完成轻量 preflight；不再触发
        # SB3 check_env 内部固定的 reset(seed=0)，避免访问不存在的 seed_0.npz。
        check_status = (
            "skipped:offline_trace_environment_requires_prepared_seed_set"
        )

        vector_environment = _make_training_vector_environment(
            ppo_config,
            seed_base=int(settings["train_seed_base"]),
            norm_reward=True,
        )
        model = build_sb3_ppo(vector_environment, ppo_config)
        initial_observation = vector_environment.reset()
        model._last_obs = np.asarray(initial_observation).copy()
        model._last_episode_starts = np.ones(
            (vector_environment.num_envs,), dtype=bool
        )
        if hasattr(vector_environment, "get_original_obs"):
            model._last_original_obs = np.asarray(
                vector_environment.get_original_obs()
            ).copy()

        callback_state = ControlledTrainingCallback()
        previous_aggregate = _new_metric_accumulator()
        checkpoint_updates = set(int(item) for item in settings["checkpoint_updates"])
        for update_index in range(1, int(settings["updates"]) + 1):
            callback_state.begin_rollout_diagnostics()
            model.learn(
                total_timesteps=int(settings["n_steps"]),
                callback=callback_state,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            sb3_metrics = _finite_metrics(model.logger.name_to_value)
            feature_stats = callback_state.finish_rollout_diagnostics()
            reward_diagnostics = callback_state.finish_rollout_reward_diagnostics()
            critic_diagnostics = _critic_rollout_diagnostics(
                model,
                sb3_metrics=sb3_metrics,
                reward_diagnostics=reward_diagnostics,
            )
            current_aggregate = _finalize_accumulator(callback_state.aggregate)
            update_metrics = _accumulator_delta(
                callback_state.aggregate,
                previous_aggregate,
            )
            record = {
                "update_index": update_index,
                "timesteps": int(model.num_timesteps),
                "metrics": {
                    name: float(sb3_metrics[f"train/{name}"])
                    for name in REQUIRED_PPO_METRICS
                },
                "critic": critic_diagnostics,
                "reward_diagnostics": reward_diagnostics,
                "feature_stats": feature_stats,
                "update": update_metrics,
                "cumulative": current_aggregate,
            }
            update_records.append(record)
            recent_episode = (
                callback_state.episodes[-1] if callback_state.episodes else {}
            )
            _print_stage15_update(
                record,
                completed_episodes=len(callback_state.episodes),
                expected_updates=int(settings["updates"]),
                recent_raw_reward=recent_episode.get("system_reward"),
            )
            hard_counts = _stage15_hard_condition_counts(
                current_aggregate,
                reward_diagnostics,
            )
            if any(value != 0 for value in hard_counts.values()):
                reason = ", ".join(
                    f"{key}={value}"
                    for key, value in hard_counts.items()
                    if int(value) != 0
                )
                _print_stage15_hard_stop(
                    reason,
                    update=update_index,
                    timesteps=model.num_timesteps,
                )
                raise RuntimeError(
                    f"Stage15 hard-stop at update {update_index}: {hard_counts}"
                )
            previous_aggregate = copy.deepcopy(callback_state.aggregate)

            if update_index in checkpoint_updates:
                checkpoint_dir = run_dir / f"checkpoint_{update_index}"
                _save_model_vecnormalize_pair(
                    model,
                    vector_environment,
                    checkpoint_dir,
                )
                validation = _evaluate_stage15_checkpoint(
                    ppo_config,
                    checkpoint_dir,
                    settings["validation_seeds"],
                    reference_cache=reference_cache,
                    split="validation",
                )
                try:
                    _assert_stage15_evaluation_hard_zero(
                        validation,
                        split="validation",
                    )
                except Exception as error:
                    _print_stage15_hard_stop(
                        f"validation evaluation: {error}",
                        update=update_index,
                        timesteps=model.num_timesteps,
                    )
                    raise
                validation["update_index"] = int(update_index)
                validation_records.append(validation)
                validation_path = (
                    run_dir / "validation" / f"update_{update_index:03d}.json"
                )
                validation_path.parent.mkdir(parents=True, exist_ok=True)
                with validation_path.open("w", encoding="utf-8") as handle:
                    json.dump(_json_safe(validation), handle, ensure_ascii=False, indent=2)
                validation_reward = float(validation["criterion_value"])
                new_best = False
                if validation_reward > best_validation_reward:
                    best_validation_reward = validation_reward
                    best_update = update_index
                    new_best = True
                    best_checkpoint_dir = run_dir / "best"
                    _copy_model_vecnormalize_pair(checkpoint_dir, best_checkpoint_dir)
                    with (run_dir / "best_selection.json").open(
                        "w", encoding="utf-8"
                    ) as handle:
                        json.dump(
                            _json_safe(
                                {
                                    "criterion": "mean original validation system reward",
                                    "best_update": best_update,
                                    "best_validation_reward": best_validation_reward,
                                    "source_checkpoint": checkpoint_dir.name,
                                }
                            ),
                            handle,
                            ensure_ascii=False,
                            indent=2,
                        )
                _print_stage15_validation(
                    validation,
                    update_index=update_index,
                    current_best_update=best_update,
                    new_best=new_best,
                )

        if int(model.num_timesteps) != int(settings["total_timesteps"]):
            raise RuntimeError(
                f"Stage15 timesteps={model.num_timesteps}，预期 {settings['total_timesteps']}。"
            )
        if len(update_records) != int(settings["updates"]):
            raise RuntimeError("Stage15 未得到完整的 100 个 PPO update 记录。")
        if best_checkpoint_dir is None or best_update is None:
            raise RuntimeError("Stage15 未产生 best checkpoint。")

        mobility_sumo_starts = sumo_start_count() - mobility_sumo_starts_before
        if mobility_sumo_starts != 0:
            raise RuntimeError(
                "Stage15 trace-mode run unexpectedly started SUMO "
                f"{mobility_sumo_starts} times"
            )

        final_checkpoint_dir = run_dir / "final"
        _save_model_vecnormalize_pair(
            model,
            vector_environment,
            final_checkpoint_dir,
        )
        vector_environment.close()
        vector_environment = None

        # Test 只在 100 updates 完成且 best 已确定后运行一次；不参与选择。
        test_summary = _evaluate_stage15_checkpoint(
            ppo_config,
            best_checkpoint_dir,
            settings["test_seeds"],
            reference_cache=reference_cache,
            split="test",
        )
        try:
            _assert_stage15_evaluation_hard_zero(test_summary, split="test")
        except Exception as error:
            _print_stage15_hard_stop(
                f"test evaluation: {error}",
                update=settings["updates"],
                timesteps=model.num_timesteps,
            )
            raise
        with (run_dir / "test_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(test_summary), handle, ensure_ascii=False, indent=2)

        _write_update_metrics_artifacts(run_dir, update_records)
        final_training_metrics = _finite_metrics(model.logger.name_to_value)
        training_rollout = _finalize_accumulator(callback_state.aggregate)
        hard_conditions = _stage15_hard_condition_counts(
            training_rollout,
            {
                "raw": {"nonfinite_count": 0},
                "normalized": {"nonfinite_count": 0},
            },
        )
        hard_zero = all(value == 0 for value in hard_conditions.values())
        test_hard_zero = all(
            int(item["ppo"].get(key, 0)) == 0
            and int(item["reference"].get(key, 0)) == 0
            for item in test_summary["per_seed"]
            for key in STAGE15_HARD_CONDITION_KEYS
        )
        training_summary = {
            "stage": 15,
            "status": "PASS" if hard_zero and test_hard_zero else "FAIL",
            "run_id": run_dir.name,
            "training": {
                "total_timesteps": int(settings["total_timesteps"]),
                "ppo_updates": int(settings["updates"]),
                "model_learn_calls": int(settings["updates"]),
                "episodes_started": int(settings["episodes_started"]),
                "episodes_completed": len(callback_state.episodes),
                "train_seed_base": int(settings["train_seed_base"]),
                "train_seed_rule": "train_seed_base + episode_index",
                "train_episode_seeds": settings["train_seeds"],
                "validation_seeds": settings["validation_seeds"],
                "test_seeds": settings["test_seeds"],
                "device": "cpu",
                "policy": "MlpPolicy",
                "action_space": "Discrete(2)",
                "observation_shape": [8],
                "state_features": list(PPO_CACHE_FEATURE_NAMES),
                "vecnormalize": {
                    "norm_obs": True,
                    "norm_reward": True,
                    "clip_obs": 10.0,
                    "system_reward_formula_unchanged": True,
                },
                "eval_freq": int(settings["eval_freq"]),
                "checkpoint_updates": settings["checkpoint_updates"],
                "best_model_criterion": "mean original validation system reward",
                "hard_stop_on": list(_section(ppo_config, "ppo").get("hard_stop_on", [])),
                "test_after_best_only": True,
                "mobility_source": "trace",
                "sumo_starts_during_trace_mode": int(mobility_sumo_starts),
                "trace_preflight": trace_preflight,
                "trace_partition": _stage15_trace_console_partition(settings),
            },
            "training_metrics": final_training_metrics,
            "training_metrics_by_update": update_records,
            "critic_trend": _critic_trend(update_records),
            "training_rollout": training_rollout,
            "training_episodes": callback_state.episodes,
            "validation": validation_records,
            "best": {
                "update": best_update,
                "validation_mean_original_system_reward": best_validation_reward,
                "criterion": "mean original validation system reward",
                "path": "best",
            },
            "test": test_summary,
            "correctness": {
                "check_env": check_status,
                "trace_preflight": trace_preflight,
                "model_learn_called": int(settings["updates"]),
                "optimizer_step_called_directly": 0,
                "training_hard_conditions": hard_conditions,
                "training_hard_conditions_zero": hard_zero,
                "test_hard_conditions_zero": test_hard_zero,
                "test_runs_after_best_selection": True,
                "sumo_starts_during_trace_mode": int(mobility_sumo_starts),
            },
            "artifacts": {
                "config_snapshot": "config_snapshot.json",
                "update_metrics_json": "update_metrics.json",
                "update_metrics_csv": "update_metrics.csv",
                "validation": "validation/update_<update>.json",
                "best": "best/model.zip + best/vecnormalize.pkl",
                "final": "final/model.zip + final/vecnormalize.pkl",
                "best_selection": "best_selection.json",
                "test": "test_summary.json",
                "training_summary": "training_summary.json",
            },
        }
        with (run_dir / "training_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(_json_safe(training_summary), handle, ensure_ascii=False, indent=2)
        # 先保存训练、checkpoint 和 test，再从磁盘 summary 后处理论文图；绘图异常不改变训练状态。
        plotting = _generate_stage15_figures(run_dir, training_summary)
        training_summary["plotting"] = plotting
        training_summary["artifacts"]["figures"] = (
            "figures/stage15_reward_convergence.png/.pdf; "
            "figures/stage15_validation_reward.png/.pdf; "
            "figures/stage15_execution_rate.png/.pdf"
        )
        with (run_dir / "training_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(_json_safe(training_summary), handle, ensure_ascii=False, indent=2)
        return training_summary
    except Exception as error:
        if vector_environment is not None:
            vector_environment.close()
        with (run_dir / "failure.json").open("w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(
                    {
                        "stage": 15,
                        "status": "FAIL",
                        "error": repr(error),
                        "updates_completed": len(update_records),
                        "formal_training_started": True,
                    }
                ),
                handle,
                ensure_ascii=False,
                indent=2,
            )
        raise


def run_controlled_training(
    config: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """执行第十一阶段的 512-step 受控训练并保存唯一正式 checkpoint。"""

    base_config = load_config() if config is None or not isinstance(config, Mapping) else copy.deepcopy(dict(config))
    if not isinstance(config, Mapping) and config is not None:
        base_config = load_config(config)
    ppo_config = prepare_ppo_config(base_config)
    ppo_section = _section(ppo_config, "ppo")
    simulation = _section(ppo_config, "simulation")
    total_timesteps = int(ppo_section.get("total_timesteps", 512))
    n_steps = int(ppo_section.get("n_steps", 256))
    if total_timesteps != 512 or n_steps != 256:
        raise ValueError("第十一阶段固定要求 total_timesteps=512、n_steps=256。")
    if int(simulation.get("slot_count", 0)) != 100:
        raise ValueError("第十一阶段 held-out 评估要求 simulation.slot_count=100。")
    train_seed_base = int(ppo_section.get("train_seed_base", DEFAULT_TRAIN_SEED_BASE))
    eval_seed = int(ppo_section.get("eval_seed", DEFAULT_EVAL_SEED))
    if eval_seed in {train_seed_base, train_seed_base + 1}:
        raise ValueError("eval_seed 必须与训练 seed 序列分离。")
    if _sb3_device(ppo_config).lower() != "cpu":
        raise ValueError("第十一阶段第一次受控训练必须使用 device=cpu。")

    run_dir = _new_run_dir(ppo_config)
    try:
        # 先保存同一 held-out seed 的 Reference 对照，再开始 PPO 更新。
        reference_summary = _run_reference_episode(base_config, seed=eval_seed)
        representative_states = _collect_representative_states(
            ppo_config,
            seed=train_seed_base,
            max_slots=int(simulation["slot_count"]),
        )

        # 官方 check_env 使用独立环境，避免检查过程污染训练环境的 SUMO 状态。
        check_environment = build_ppo_environment(ppo_config)
        try:
            check_status = sb3_check_env(check_environment)
        finally:
            check_environment.close()
        if check_status != "PASS":
            raise RuntimeError(f"官方 SB3 check_env 未通过：{check_status}")

        vector_environment = _make_training_vector_environment(
            ppo_config,
            seed_base=train_seed_base,
        )
        model = build_sb3_ppo(vector_environment, ppo_config)
        initial_policy_state = copy.deepcopy(model.policy.state_dict())
        callback_state = ControlledTrainingCallback()
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_state,
            reset_num_timesteps=True,
            progress_bar=False,
        )

        logger_values = dict(model.logger.name_to_value)
        sb3_metrics = _finite_metrics(logger_values)
        # SB3 的最终 logger 可能在最后一次 dump 后只保留 train/*；episode
        # 汇总仍由 Monitor/回调真实采集，因此补齐同名 rollout 指标而不改 reward。
        if callback_state.episodes:
            if "rollout/ep_rew_mean" not in sb3_metrics:
                sb3_metrics["rollout/ep_rew_mean"] = float(
                    np.mean([item["reward"] for item in callback_state.episodes])
                )
            if "rollout/ep_len_mean" not in sb3_metrics:
                sb3_metrics["rollout/ep_len_mean"] = float(
                    np.mean([item["steps"] for item in callback_state.episodes])
                )
        for key in ("rollout/ep_rew_mean", "rollout/ep_len_mean"):
            if key not in sb3_metrics or not math.isfinite(float(sb3_metrics[key])):
                raise RuntimeError(f"无法得到有限的 SB3 指标：{key}")
        model_base = run_dir / "model"
        vecnormalize_path = run_dir / "vecnormalize.pkl"
        model.save(str(model_base))
        vector_environment.save(str(vecnormalize_path))
        vector_environment.close()

        model_path = run_dir / "model.zip"
        if not model_path.is_file() or not vecnormalize_path.is_file():
            raise RuntimeError("受控训练 checkpoint 或 VecNormalize 统计未保存。")

        # 用刚保存的 model.zip + vecnormalize.pkl 做唯一 held-out PPO 评估。
        evaluation_environment = _make_evaluation_vector_environment(
            ppo_config,
            seed_base=eval_seed,
            vecnormalize_path=vecnormalize_path,
        )
        try:
            from stable_baselines3 import PPO

            loaded_model = PPO.load(
                str(model_base),
                env=evaluation_environment,
                device="cpu",
            )
            ppo_summary = _run_ppo_episode(
                loaded_model,
                evaluation_environment,
                expected_steps=int(simulation["slot_count"]),
            )
        finally:
            evaluation_environment.close()

        reference_replay = _run_reference_short_replay(
            base_config,
            seed=train_seed_base,
        )
        expected_replay_counts = {
            "local": 0,
            "service_vehicle": 0,
            "rsu": 15,
            "uav": 7,
            "cloud": 84,
            "failed": 0,
        }
        if (
            reference_replay["task_total"] != 106
            or reference_replay["execution_counts"] != expected_replay_counts
            or reference_replay["nan_inf_count"] != 0
            or reference_replay["kkt_failure_count"] != 0
        ):
            raise RuntimeError(
                "训练后的 20-slot Reference replay 未复现既定结果，拒绝交付。"
            )

        raw_observations = [item["observation"] for item in representative_states]
        # 使用刚保存的训练统计，保证训练前后概率在同一 norm_obs 坐标系中比较。
        probability_environment = _make_evaluation_vector_environment(
            ppo_config,
            seed_base=eval_seed,
            vecnormalize_path=vecnormalize_path,
        )
        try:
            before_probabilities = _policy_probabilities(
                model,
                probability_environment,
                raw_observations,
                initial_policy_state,
            )
            final_policy_state = copy.deepcopy(model.policy.state_dict())
            after_probabilities = _policy_probabilities(
                model,
                probability_environment,
                raw_observations,
                final_policy_state,
            )
        finally:
            probability_environment.close()

        episodes_started = int(math.ceil(total_timesteps / int(simulation["slot_count"])))
        training_summary = {
            "stage": 11,
            "status": "PASS",
            "run_id": run_dir.name,
            "training": {
                "total_timesteps": total_timesteps,
                "ppo_updates": total_timesteps // n_steps,
                "episodes_started": episodes_started,
                "episodes_completed": len(callback_state.episodes),
                "train_seed_base": train_seed_base,
                "train_episode_seeds": [
                    train_seed_base + index for index in range(episodes_started)
                ],
                "eval_seed": eval_seed,
                "device": "cpu",
                "policy": "MlpPolicy",
                "action_space": "Discrete(2)",
                "observation_shape": [8],
                "vecnormalize": {
                    "norm_obs": True,
                    "norm_reward": False,
                    "clip_obs": 10.0,
                },
                "learning_rate": float(ppo_section.get("learning_rate", 3.0e-4)),
                "n_steps": n_steps,
                "batch_size": int(ppo_section.get("batch_size", 64)),
                "n_epochs": int(ppo_section.get("n_epochs", 10)),
                "gamma": float(ppo_section.get("gamma", 0.99)),
                "gae_lambda": float(ppo_section.get("gae_lambda", 0.95)),
                "clip_range": float(ppo_section.get("clip_range", 0.2)),
                "ent_coef": float(ppo_section.get("ent_coef", 0.0)),
                "vf_coef": float(ppo_section.get("vf_coef", 0.5)),
                "max_grad_norm": float(ppo_section.get("max_grad_norm", 0.5)),
                "target_kl": None,
            },
            "training_metrics": sb3_metrics,
            "training_rollout": _finalize_accumulator(callback_state.aggregate),
            "training_episodes": callback_state.episodes,
            "representative_policy_probabilities": [
                {
                    "state": state["state"],
                    "slot": state["slot"],
                    "delta_predicted_utility": state["delta_predicted_utility"],
                    "before": before,
                    "after": after,
                }
                for state, before, after in zip(
                    representative_states,
                    before_probabilities,
                    after_probabilities,
                )
            ],
            "held_out": {
                "seed": eval_seed,
                "reference": reference_summary,
                "ppo": ppo_summary,
                "interpretation": "单一 held-out seed 的技术方向性比较，不代表正式实验优劣结论。",
            },
            "correctness": {
                "check_env": check_status,
                "model_learn_called": 1,
                "optimizer_step_called_directly": 0,
                "future_request_leakage": int(
                    callback_state.aggregate["future_request_leakage"]
                ),
                "future_trajectory_leakage": int(
                    callback_state.aggregate["future_trajectory_leakage"]
                ),
                "capacity_violations": int(
                    callback_state.aggregate["capacity_violations"]
                ),
                "uav_violations": int(callback_state.aggregate["uav_violations"]),
                "nan_inf_count": int(callback_state.aggregate["nan_inf_count"]),
                "kkt_failure_count": int(
                    callback_state.aggregate["kkt_failure_count"]
                ),
                "reference_replay_20_slot": {
                    "status": "PASS",
                    "tasks": int(reference_replay["task_total"]),
                    "execution_counts": reference_replay["execution_counts"],
                    "nan_inf_count": int(reference_replay["nan_inf_count"]),
                    "kkt_failure_count": int(reference_replay["kkt_failure_count"]),
                },
            },
            "artifacts": {
                "model": "model.zip",
                "vecnormalize": "vecnormalize.pkl",
                "config_snapshot": "config_snapshot.json",
                "training_summary": "training_summary.json",
            },
        }
        with (run_dir / "config_snapshot.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(ppo_config), handle, ensure_ascii=False, indent=2)
        with (run_dir / "training_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(_json_safe(training_summary), handle, ensure_ascii=False, indent=2)
        return training_summary
    except Exception:
        # 失败的临时 run 不属于有效训练产物，避免留下空目录/半成品误用。
        shutil.rmtree(run_dir, ignore_errors=True)
        raise


def _write_update_metrics_artifacts(
    run_dir: Path,
    update_records: Sequence[Mapping[str, Any]],
) -> None:
    """保存每个 PPO update 的可审计指标，不写入临时日志。"""

    with (run_dir / "update_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(list(update_records)), handle, ensure_ascii=False, indent=2)

    metric_names = tuple(name for name in REQUIRED_PPO_METRICS)
    fieldnames = ["update_index", "timesteps", *metric_names]
    critic_names = (
        "reward_mean",
        "reward_std",
        "raw_reward_mean",
        "raw_reward_std",
        "raw_reward_min",
        "raw_reward_max",
        "normalized_reward_mean",
        "normalized_reward_std",
        "normalized_reward_min",
        "normalized_reward_max",
        "return_mean",
        "return_std",
        "value_mean",
        "value_std",
        "value_return_rmse",
        "value_return_pearson",
        "value_return_pearson_status",
        "value_return_scale_ratio",
        "critic_explained_variance",
        "critic_value_loss",
    )
    fieldnames.extend(critic_names)
    for feature_name in PPO_CACHE_FEATURE_NAMES:
        for statistic in ("mean", "std", "min", "max"):
            fieldnames.append(f"feature_{feature_name}_{statistic}")
    for prefix in ("cumulative", "update"):
        fieldnames.extend(
            [
                f"{prefix}_steps",
                f"{prefix}_reward",
                f"{prefix}_KEEP",
                f"{prefix}_UPDATE",
                f"{prefix}_effective_UPDATE",
                f"{prefix}_UPDATE_noop",
                f"{prefix}_action_mismatch",
                f"{prefix}_cache_changed_slots",
                f"{prefix}_cache_changed_services",
                f"{prefix}_cache_switch_cost",
                f"{prefix}_uav_moved_slots",
                f"{prefix}_uav_move_cost",
                f"{prefix}_compute_violations",
                f"{prefix}_bandwidth_violations",
                f"{prefix}_capacity_violations",
                f"{prefix}_uav_violations",
                f"{prefix}_invalid_service_ids",
                f"{prefix}_invalid_node_ids",
                f"{prefix}_nan_inf_count",
                f"{prefix}_kkt_failure_count",
                f"{prefix}_local",
                f"{prefix}_service_vehicle",
                f"{prefix}_rsu",
                f"{prefix}_uav",
                f"{prefix}_cloud",
                f"{prefix}_failed",
                f"{prefix}_failed_tasks",
                f"{prefix}_total_delay",
                f"{prefix}_total_energy",
                f"{prefix}_mean_delay",
                f"{prefix}_mean_energy",
                f"{prefix}_system_reward",
            ]
        )

    def source_value(source: Mapping[str, Any], key: str) -> Any:
        if key == "KEEP":
            return source["action_counts"]["KEEP"]
        if key == "UPDATE":
            return source["action_counts"]["UPDATE"]
        if key == "effective_UPDATE":
            return source["ppo_effective_update_count"]
        if key == "UPDATE_noop":
            return source["ppo_update_noop_count"]
        if key == "action_mismatch":
            return source["ppo_action_mismatch_count"]
        if key in source.get("execution_counts", {}):
            return source["execution_counts"][key]
        return source[key]

    with (run_dir / "update_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in update_records:
            row: dict[str, Any] = {
                "update_index": int(record["update_index"]),
                "timesteps": int(record["timesteps"]),
            }
            for name in metric_names:
                row[name] = float(record["metrics"][name])
            critic = record.get("critic", {})
            for name in critic_names:
                row[name] = critic.get(
                    "explained_variance"
                    if name == "critic_explained_variance"
                    else "value_loss"
                    if name == "critic_value_loss"
                    else name
                )
            feature_stats = record.get("feature_stats", {})
            for feature_name in PPO_CACHE_FEATURE_NAMES:
                statistics = feature_stats.get(feature_name, {})
                for statistic in ("mean", "std", "min", "max"):
                    row[f"feature_{feature_name}_{statistic}"] = statistics.get(
                        statistic
                    )
            for prefix in ("cumulative", "update"):
                source = record[prefix]
                for key in (
                    "steps",
                    "reward",
                    "KEEP",
                    "UPDATE",
                    "effective_UPDATE",
                    "UPDATE_noop",
                    "action_mismatch",
                    "cache_changed_slots",
                    "cache_changed_services",
                    "cache_switch_cost",
                    "uav_moved_slots",
                    "uav_move_cost",
                    "compute_violations",
                    "bandwidth_violations",
                    "capacity_violations",
                    "uav_violations",
                    "invalid_service_ids",
                    "invalid_node_ids",
                    "nan_inf_count",
                    "kkt_failure_count",
                    "local",
                    "service_vehicle",
                    "rsu",
                    "uav",
                    "cloud",
                    "failed",
                    "failed_tasks",
                    "total_delay",
                    "total_energy",
                    "mean_delay",
                    "mean_energy",
                    "system_reward",
                ):
                    row[f"{prefix}_{key}"] = source_value(source, key)
            writer.writerow(_json_safe(row))


def _critic_trend(update_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """给出 start/mid/end 三个观察点；不设置人为 explained-variance 阈值。"""

    if not update_records:
        raise ValueError("至少需要一个 PPO update 才能计算 critic trend。")
    indexes = (0, len(update_records) // 2 - 1, len(update_records) - 1)

    def point(index: int) -> dict[str, Any]:
        record = update_records[index]
        metrics = record["metrics"]
        critic = record.get("critic", {})
        point = {
            "update_index": int(record["update_index"]),
            "explained_variance": float(
                critic.get("explained_variance", metrics["explained_variance"])
            ),
            "value_loss": float(critic.get("value_loss", metrics["value_loss"])),
            "approx_kl": float(metrics["approx_kl"]),
            "clip_fraction": float(metrics["clip_fraction"]),
            "entropy_loss": float(metrics["entropy_loss"]),
        }
        for name in (
            "reward_mean",
            "reward_std",
            "raw_reward_mean",
            "raw_reward_std",
            "raw_reward_min",
            "raw_reward_max",
            "normalized_reward_mean",
            "normalized_reward_std",
            "normalized_reward_min",
            "normalized_reward_max",
            "return_mean",
            "return_std",
            "value_mean",
            "value_std",
            "value_return_rmse",
            "value_return_pearson",
            "value_return_pearson_status",
            "value_return_scale_ratio",
        ):
            point[name] = critic.get(name)
        return point

    start = point(indexes[0])
    middle = point(indexes[1])
    end = point(indexes[2])
    return {
        "start": start,
        "mid": middle,
        "end": end,
        "explained_variance_delta_start_to_end": end["explained_variance"]
        - start["explained_variance"],
        "value_loss_delta_start_to_end": end["value_loss"] - start["value_loss"],
        "pearson_start_to_end": (
            None
            if start["value_return_pearson"] is None
            or end["value_return_pearson"] is None
            else end["value_return_pearson"] - start["value_return_pearson"]
        ),
        "all_finite": all(
            math.isfinite(float(point_value))
            for item in (start, middle, end)
            for point_value in item.values()
            if isinstance(point_value, (int, float))
        ),
    }


def _critic_comparison(
    update_records: Sequence[Mapping[str, Any]],
    previous_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """比较本次 Critic 与历史结果；缺失的历史量明确标记 unavailable。"""

    new_trend = _critic_trend(update_records)
    old_trend = previous_summary.get("critic_trend", {})
    pearsons = np.asarray(
        [
            float(record["critic"]["value_return_pearson"])
            for record in update_records
            if record["critic"].get("value_return_pearson") is not None
        ],
        dtype=float,
    )
    return_stds = np.asarray(
        [float(record["critic"]["return_std"]) for record in update_records],
        dtype=float,
    )
    value_stds = np.asarray(
        [float(record["critic"]["value_std"]) for record in update_records],
        dtype=float,
    )
    std_alignment, std_alignment_status = _pearson_or_undefined(
        value_stds,
        return_stds,
    )
    new_start_pearson = new_trend["start"].get("value_return_pearson")
    new_end_pearson = new_trend["end"].get("value_return_pearson")
    old_start_ev = old_trend.get("start", {}).get("explained_variance")
    old_end_ev = old_trend.get("end", {}).get("explained_variance")
    new_end_ev = new_trend["end"]["explained_variance"]
    flags = {
        "pearson_positive_trend": bool(
            new_start_pearson is not None
            and new_end_pearson is not None
            and new_end_pearson > 0.0
            and new_end_pearson > new_start_pearson
        ),
        "explained_variance_improved_vs_old": bool(
            old_end_ev is not None and new_end_ev > float(old_end_ev)
        ),
        "value_std_follows_return_std": bool(
            std_alignment is not None and std_alignment > 0.5
        ),
    }

    def old_value(point_name: str, key: str) -> Any:
        point = old_trend.get(point_name, {})
        value = point.get(key)
        if value is not None:
            return value
        if key == "value_return_scale_ratio":
            value_std = point.get("value_std")
            return_std = point.get("return_std")
            if value_std is not None and return_std is not None:
                return float(value_std) / (float(return_std) + 1.0e-12)
        return None

    old_pearson_start = old_value("start", "value_return_pearson")
    old_pearson_end = old_value("end", "value_return_pearson")
    old_return_std_start = old_value("start", "return_std")
    old_return_std_mid = old_value("mid", "return_std")
    old_return_std_end = old_value("end", "return_std")
    old_value_std_start = old_value("start", "value_std")
    old_value_std_mid = old_value("mid", "value_std")
    old_value_std_end = old_value("end", "value_std")
    old_scale_start = old_value("start", "value_return_scale_ratio")
    old_scale_mid = old_value("mid", "value_return_scale_ratio")
    old_scale_end = old_value("end", "value_return_scale_ratio")
    return {
        "old": {
            "explained_variance_start": old_trend.get("start", {}).get("explained_variance"),
            "explained_variance_end": old_end_ev,
            "value_loss_start": old_trend.get("start", {}).get("value_loss"),
            "value_loss_end": old_trend.get("end", {}).get("value_loss"),
            "value_return_pearson_start": old_pearson_start,
            "value_return_pearson_end": old_pearson_end,
            "return_std_start": old_return_std_start,
            "return_std_mid": old_return_std_mid,
            "return_std_end": old_return_std_end,
            "value_std_start": old_value_std_start,
            "value_std_mid": old_value_std_mid,
            "value_std_end": old_value_std_end,
            "value_return_scale_ratio_start": old_scale_start,
            "value_return_scale_ratio_mid": old_scale_mid,
            "value_return_scale_ratio_end": old_scale_end,
            "unavailable_reason": (
                None
                if all(
                    item is not None
                    for item in (
                        old_pearson_start,
                        old_pearson_end,
                        old_return_std_start,
                        old_value_std_start,
                    )
                )
                else "历史结果未保存完整的 rollout buffer return/value 诊断。"
            ),
        },
        "new": {
            "trend": new_trend,
            "value_return_pearson_start": new_start_pearson,
            "value_return_pearson_end": new_end_pearson,
            "return_std_start": new_trend["start"].get("return_std"),
            "return_std_mid": new_trend["mid"].get("return_std"),
            "return_std_end": new_trend["end"].get("return_std"),
            "value_std_start": new_trend["start"].get("value_std"),
            "value_std_mid": new_trend["mid"].get("value_std"),
            "value_std_end": new_trend["end"].get("value_std"),
            "value_return_scale_ratio_start": new_trend["start"].get(
                "value_return_scale_ratio"
            ),
            "value_return_scale_ratio_mid": new_trend["mid"].get(
                "value_return_scale_ratio"
            ),
            "value_return_scale_ratio_end": new_trend["end"].get(
                "value_return_scale_ratio"
            ),
        },
        "value_std_return_std_pearson": std_alignment,
        "value_std_return_std_pearson_status": std_alignment_status,
        "available_pearson_updates": int(pearsons.size),
        "improvement_flags": flags,
        "critic_improved": bool(any(flags.values())),
        "assessment": "明显改善" if any(flags.values()) else "无改善",
    }


def _critic_series(
    summary_or_records: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    key: str,
) -> np.ndarray:
    """读取每个 update 的 Critic 序列，并兼容 Stage13 未保存的 ratio。"""

    if isinstance(summary_or_records, Mapping):
        records = summary_or_records.get("training_metrics_by_update", [])
    else:
        records = summary_or_records
    values: list[float] = []
    for record in records:
        critic = record.get("critic", {})
        value = critic.get(key)
        if value is None and key == "value_return_scale_ratio":
            value_std = critic.get("value_std")
            return_std = critic.get("return_std")
            if value_std is not None and return_std is not None:
                value = float(value_std) / (float(return_std) + 1.0e-12)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def _checkpoint_values(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return {"start": None, "mid": None, "end": None}
    middle_index = values.size // 2 - 1
    return {
        "start": float(values[0]),
        "mid": float(values[max(0, middle_index)]),
        "end": float(values[-1]),
    }


def _normalization_decision(
    update_records: Sequence[Mapping[str, Any]],
    previous_summary: Mapping[str, Any],
    evaluation_summary: Mapping[str, Any],
    *,
    technical_correctness_pass: bool,
    training_rollout: Mapping[str, Any],
) -> dict[str, Any]:
    """按预先声明的证据规则决定是否保留 norm_reward=True。"""

    new_pearson = _critic_series(update_records, "value_return_pearson")
    old_pearson = _critic_series(previous_summary, "value_return_pearson")
    new_ev = _critic_series(update_records, "explained_variance")
    old_ev = _critic_series(previous_summary, "explained_variance")
    new_scale = _critic_series(update_records, "value_return_scale_ratio")
    old_scale = _critic_series(previous_summary, "value_return_scale_ratio")

    def checkpoint(values: np.ndarray, index: int = -1) -> float | None:
        if values.size == 0:
            return None
        return float(values[index])

    new_pearson_checkpoints = _checkpoint_values(new_pearson)
    old_pearson_checkpoints = _checkpoint_values(old_pearson)
    new_ev_checkpoints = _checkpoint_values(new_ev)
    old_ev_checkpoints = _checkpoint_values(old_ev)
    new_scale_checkpoints = _checkpoint_values(new_scale)
    old_scale_checkpoints = _checkpoint_values(old_scale)

    def nonempty_pair(left: np.ndarray, right: np.ndarray) -> bool:
        return left.size > 0 and right.size > 0

    pearson_end_not_lower = bool(
        nonempty_pair(new_pearson, old_pearson)
        and new_pearson_checkpoints["end"] is not None
        and old_pearson_checkpoints["end"] is not None
        and new_pearson_checkpoints["end"]
        >= old_pearson_checkpoints["end"] - 1.0e-12
    )
    pearson_stability_not_worse = bool(
        nonempty_pair(new_pearson, old_pearson)
        and float(np.std(new_pearson)) <= float(np.std(old_pearson)) + 1.0e-12
    )
    ev_clearly_better = bool(
        nonempty_pair(new_ev, old_ev)
        and new_ev_checkpoints["end"] is not None
        and old_ev_checkpoints["end"] is not None
        and new_ev_checkpoints["end"] > old_ev_checkpoints["end"]
        and float(np.mean(new_ev)) > float(np.mean(old_ev))
    )

    def scale_distance(values: np.ndarray) -> float:
        if values.size == 0:
            return math.inf
        return float(np.mean(np.abs(np.log(np.maximum(values, 1.0e-12)))))

    scale_mean_distance_improved = bool(
        nonempty_pair(new_scale, old_scale)
        and scale_distance(new_scale) < scale_distance(old_scale)
    )
    scale_end_distance_improved = bool(
        new_scale_checkpoints["end"] is not None
        and old_scale_checkpoints["end"] is not None
        and abs(float(new_scale_checkpoints["end"]) - 1.0)
        < abs(float(old_scale_checkpoints["end"]) - 1.0)
    )
    scale_clearly_improved = bool(
        scale_mean_distance_improved and scale_end_distance_improved
    )

    keep_count = int(training_rollout.get("action_counts", {}).get("KEEP", 0))
    update_count = int(training_rollout.get("action_counts", {}).get("UPDATE", 0))
    action_total = keep_count + update_count
    actor_no_collapse = bool(
        action_total > 0
        and keep_count / action_total > 0.01
        and update_count / action_total > 0.01
        and int(training_rollout.get("ppo_action_mismatch_count", 0)) == 0
    )

    new_aggregate = evaluation_summary.get("aggregate", {})
    previous_aggregate = evaluation_summary.get("previous_ppo_aggregate", {})
    new_reward = new_aggregate.get("mean_ppo_reward")
    old_reward = previous_aggregate.get("mean_ppo_reward")
    new_delta = new_aggregate.get("mean_reward_delta")
    old_delta = previous_aggregate.get("mean_reward_delta")
    raw_reward_no_degradation = bool(
        new_reward is not None
        and old_reward is not None
        and new_delta is not None
        and old_delta is not None
        and float(new_reward) >= float(old_reward) - 1.0e-12
        and float(new_delta) >= float(old_delta) - 1.0e-12
    )

    criteria = {
        "pearson_end_not_lower": pearson_end_not_lower,
        "pearson_stability_not_worse": pearson_stability_not_worse,
        "explained_variance_clearly_better": ev_clearly_better,
        "value_return_scale_ratio_clearly_improved": scale_clearly_improved,
        "actor_no_collapse": actor_no_collapse,
        "raw_heldout_reward_no_system_degradation": raw_reward_no_degradation,
        "all_correctness_zero": bool(technical_correctness_pass),
    }
    retain = bool(all(criteria.values()))
    return {
        "tested_variable": {
            "norm_obs": True,
            "norm_reward_before": False,
            "norm_reward_after": True,
            "clip_obs": 10.0,
        },
        "criteria": criteria,
        "retain_norm_reward": retain,
        "recommendation": (
            "retain_norm_reward_true" if retain else "restore_norm_reward_false"
        ),
        "pearson": {
            "stage13_start_mid_end": old_pearson_checkpoints,
            "stage14_start_mid_end": new_pearson_checkpoints,
            "stage13_mean": float(np.mean(old_pearson)) if old_pearson.size else None,
            "stage14_mean": float(np.mean(new_pearson)) if new_pearson.size else None,
            "stage13_std": float(np.std(old_pearson)) if old_pearson.size else None,
            "stage14_std": float(np.std(new_pearson)) if new_pearson.size else None,
        },
        "explained_variance": {
            "stage13_start_mid_end": old_ev_checkpoints,
            "stage14_start_mid_end": new_ev_checkpoints,
            "stage13_mean": float(np.mean(old_ev)) if old_ev.size else None,
            "stage14_mean": float(np.mean(new_ev)) if new_ev.size else None,
        },
        "value_return_scale_ratio": {
            "stage13_start_mid_end": old_scale_checkpoints,
            "stage14_start_mid_end": new_scale_checkpoints,
            "stage13_log_distance_to_one": scale_distance(old_scale),
            "stage14_log_distance_to_one": scale_distance(new_scale),
        },
        "heldout_raw_reward": {
            "stage13_mean_reward": old_reward,
            "stage14_mean_reward": new_reward,
            "stage13_mean_delta": old_delta,
            "stage14_mean_delta": new_delta,
        },
    }


def _reuse_stage13_probe_states(
    config: Mapping[str, Any],
    previous_summary: Mapping[str, Any],
    *,
    seed: int,
    max_slots: int,
) -> list[dict[str, Any]]:
    """复用 Stage13 的五个固定 probe；历史未存 raw 数组时做确定性重构并核验。"""

    expected = previous_summary.get("probe_states", {}).get("update_0", [])
    if not isinstance(expected, list) or len(expected) != 5:
        raise RuntimeError("Stage13 未保存完整的五个固定 probe state。")
    reconstructed = _collect_representative_states(
        config,
        seed=seed,
        max_slots=max_slots,
    )
    if len(reconstructed) != len(expected):
        raise RuntimeError("Stage13 probe 数量与确定性重构结果不一致。")
    for old, current in zip(expected, reconstructed):
        if int(old["slot"]) != int(current["slot"]):
            raise RuntimeError("Stage13 probe slot 不一致，拒绝替换固定 probe。")
        if not math.isclose(
            float(old["episode_progress"]),
            float(current["episode_progress"]),
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ) or not math.isclose(
            float(old["delta_predicted_utility"]),
            float(current["delta_predicted_utility"]),
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ):
            raise RuntimeError("Stage13 probe 元数据不一致，拒绝替换固定 probe。")
    return reconstructed


def run_medium_training(
    config: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """执行 Stage13/14 的唯一 5120-step 受控复验入口。"""

    base_config = (
        load_config()
        if config is None or not isinstance(config, Mapping)
        else copy.deepcopy(dict(config))
    )
    if not isinstance(config, Mapping) and config is not None:
        base_config = load_config(config)
    ppo_config = prepare_ppo_config(base_config)
    ppo_section = _section(ppo_config, "ppo")
    simulation = _section(ppo_config, "simulation")
    norm_reward = bool(ppo_section.get("norm_reward", False))
    stage = 14 if norm_reward else 13
    previous_stage = stage - 1
    total_timesteps = int(ppo_section.get("total_timesteps", 5120))
    n_steps = int(ppo_section.get("n_steps", 256))
    if total_timesteps != 5120 or n_steps != 256:
        raise ValueError("Stage13/14 固定要求 total_timesteps=5120、n_steps=256。")
    if int(simulation.get("slot_count", 0)) != 100:
        raise ValueError("Stage13/14 要求训练和 held-out episode 均为 100 slots。")
    train_seed_base = int(
        ppo_section.get("train_seed_base", DEFAULT_TRAIN_SEED_BASE)
    )
    episodes_started = int(math.ceil(total_timesteps / int(simulation["slot_count"])))
    train_episode_seeds = [
        train_seed_base + index for index in range(episodes_started)
    ]
    configured_eval_seeds = ppo_section.get("eval_seeds")
    if isinstance(configured_eval_seeds, Sequence) and not isinstance(
        configured_eval_seeds, (str, bytes)
    ):
        eval_seeds = [int(seed) for seed in configured_eval_seeds]
    else:
        first_eval_seed = int(ppo_section.get("eval_seed", DEFAULT_EVAL_SEED))
        eval_seeds = [first_eval_seed + index for index in range(5)]
    if len(eval_seeds) != 5 or len(set(eval_seeds)) != 5:
        raise ValueError("Stage13/14 必须使用五个互不重复的 held-out seed。")
    if set(eval_seeds).intersection(train_episode_seeds):
        raise ValueError("held-out seed 不得与训练 episode seed 序列重叠。")
    if _sb3_device(ppo_config).lower() != "cpu":
        raise ValueError("Stage13/14 固定使用 device=cpu。")

    # 旧模型只作为历史对照读取；第三维语义已从 delta_J 变为时间进度，
    # 因此本阶段下面的 model 仍由 build_sb3_ppo 从头创建，绝不 PPO.load 续训。
    output_root = (
        PROJECT_ROOT
        / str(_section(ppo_config, "output").get("root", "outputs"))
        / "ppo_training"
    )
    previous_result: tuple[Path, dict[str, Any], dict[str, Any]] | None = None
    for candidate in sorted(output_root.glob("*"), reverse=True):
        summary_path = candidate / "training_summary.json"
        evaluation_path = candidate / "evaluation_summary.json"
        if not candidate.is_dir() or not summary_path.is_file() or not evaluation_path.is_file():
            continue
        try:
            candidate_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            candidate_evaluation = json.loads(
                evaluation_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        if (
            candidate_summary.get("stage") == previous_stage
            and candidate_summary.get("status") == "PASS"
            and candidate_summary.get("training", {}).get("total_timesteps")
            == total_timesteps
            and candidate_evaluation.get("eval_seeds") == eval_seeds
        ):
            previous_result = (
                candidate,
                candidate_summary,
                candidate_evaluation,
            )
            break
    if previous_result is None:
        raise FileNotFoundError(
            f"找不到与 Stage{stage} 配置和五个 held-out seed 完全匹配的 Stage{previous_stage} 结果。"
        )
    previous_stage_dir, previous_stage_summary, previous_stage_evaluation = (
        previous_result
    )

    run_dir = _new_run_dir(ppo_config)
    vector_environment = None
    try:
        if stage == 14:
            representative_states = _reuse_stage13_probe_states(
                ppo_config,
                previous_stage_summary,
                seed=train_seed_base,
                max_slots=int(simulation["slot_count"]),
            )
        else:
            representative_states = _collect_representative_states(
                ppo_config,
                seed=train_seed_base,
                max_slots=int(simulation["slot_count"]),
            )

        # 训练前先单独通过官方 check_env，避免把检查过程混入训练 rollout。
        check_environment = build_ppo_environment(ppo_config)
        try:
            check_status = sb3_check_env(check_environment)
        finally:
            check_environment.close()
        if check_status != "PASS":
            raise RuntimeError(f"官方 SB3 check_env 未通过：{check_status}")

        vector_environment = _make_training_vector_environment(
            ppo_config,
            seed_base=train_seed_base,
            norm_reward=norm_reward,
        )
        model = build_sb3_ppo(vector_environment, ppo_config)

        # 连续调用 20 次 learn 只为记录 update；手动设置首次 observation，
        # 并使用 reset_num_timesteps=False，保证 episode/seed 不被意外重置。
        initial_observation = vector_environment.reset()
        model._last_obs = np.asarray(initial_observation).copy()
        model._last_episode_starts = np.ones(
            (vector_environment.num_envs,), dtype=bool
        )
        if hasattr(vector_environment, "get_original_obs"):
            model._last_original_obs = np.asarray(
                vector_environment.get_original_obs()
            ).copy()

        probe_snapshots: dict[str, list[dict[str, Any]]] = {
            "update_0": _policy_snapshot(
                model, vector_environment, representative_states
            )
        }
        callback_state = ControlledTrainingCallback()
        update_records: list[dict[str, Any]] = []
        previous_aggregate = _new_metric_accumulator()
        for update_index in range(1, 21):
            callback_state.begin_rollout_diagnostics()
            model.learn(
                total_timesteps=n_steps,
                callback=callback_state,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            sb3_metrics = _finite_metrics(model.logger.name_to_value)
            feature_stats = callback_state.finish_rollout_diagnostics()
            reward_diagnostics = callback_state.finish_rollout_reward_diagnostics()
            critic_diagnostics = _critic_rollout_diagnostics(
                model,
                sb3_metrics=sb3_metrics,
                reward_diagnostics=reward_diagnostics,
            )
            current_aggregate = _finalize_accumulator(callback_state.aggregate)
            update_metrics = _accumulator_delta(
                callback_state.aggregate,
                previous_aggregate,
            )
            update_records.append(
                {
                    "update_index": update_index,
                    "timesteps": int(model.num_timesteps),
                    "metrics": {
                        name: float(sb3_metrics[f"train/{name}"])
                        for name in REQUIRED_PPO_METRICS
                    },
                    "critic": critic_diagnostics,
                    "reward_diagnostics": reward_diagnostics,
                    "feature_stats": feature_stats,
                    "update": update_metrics,
                    "cumulative": current_aggregate,
                }
            )
            if update_index in {5, 10, 15, 20}:
                probe_snapshots[f"update_{update_index}"] = _policy_snapshot(
                    model,
                    vector_environment,
                    representative_states,
                )
            previous_aggregate = copy.deepcopy(callback_state.aggregate)

        if int(model.num_timesteps) != total_timesteps:
            raise RuntimeError(
                f"分段 learn 后 timesteps={model.num_timesteps}，预期 {total_timesteps}。"
            )
        if len(update_records) != 20:
            raise RuntimeError("Stage13/14 未得到完整的 20 个 PPO update 记录。")

        final_training_metrics = _finite_metrics(model.logger.name_to_value)
        if callback_state.episodes:
            final_training_metrics.setdefault(
                "rollout/ep_rew_mean",
                float(np.mean([item["reward"] for item in callback_state.episodes])),
            )
            final_training_metrics.setdefault(
                "rollout/ep_len_mean",
                float(np.mean([item["steps"] for item in callback_state.episodes])),
            )
        for key in ("rollout/ep_rew_mean", "rollout/ep_len_mean"):
            if key not in final_training_metrics or not math.isfinite(
                float(final_training_metrics[key])
            ):
                raise RuntimeError(f"缺少有限的 SB3 指标：{key}")

        model_base = run_dir / "model"
        vecnormalize_path = run_dir / "vecnormalize.pkl"
        model.save(str(model_base))
        vector_environment.save(str(vecnormalize_path))
        vector_environment.close()
        vector_environment = None
        model_path = run_dir / "model.zip"
        if not model_path.is_file() or not vecnormalize_path.is_file():
            raise RuntimeError("model.zip 或 vecnormalize.pkl 未成功保存。")

        # Reference 与旧 PPO 均直接复用第十二阶段同 seed 结果；本阶段只
        # 新运行 Time-aware PPO，避免重复计算并保持受控比较。
        old_rows_by_seed = {
            int(item["seed"]): item
            for item in previous_stage_evaluation["per_seed"]
        }
        if set(old_rows_by_seed) != set(eval_seeds):
            raise RuntimeError(
                f"Stage{previous_stage} held-out seed 集合与当前配置不一致。"
            )
        evaluations: list[dict[str, Any]] = []
        old_comparison_rows: list[dict[str, Any]] = []
        from stable_baselines3 import PPO

        for seed in eval_seeds:
            old_row = old_rows_by_seed[int(seed)]
            reference_summary = old_row["reference"]
            old_ppo_summary = old_row["ppo"]
            evaluation_environment = _make_evaluation_vector_environment(
                ppo_config,
                seed_base=seed,
                vecnormalize_path=vecnormalize_path,
            )
            try:
                loaded_model = PPO.load(
                    str(model_base),
                    env=evaluation_environment,
                    device="cpu",
                )
                ppo_summary = _run_ppo_episode(
                    loaded_model,
                    evaluation_environment,
                    expected_steps=int(simulation["slot_count"]),
                    reset_seed=int(seed),
                )
            finally:
                evaluation_environment.close()
            evaluations.append(
                {
                    "seed": int(seed),
                    "reference": reference_summary,
                    "previous_ppo": old_ppo_summary,
                    "old_ppo": old_ppo_summary,
                    "stage13_ppo": old_ppo_summary if stage == 14 else None,
                    "ppo": ppo_summary,
                    "reward_delta": float(
                        ppo_summary["system_reward"]
                        - reference_summary["system_reward"]
                    ),
                    "previous_reward_delta": float(
                        old_ppo_summary["system_reward"]
                        - reference_summary["system_reward"]
                    ),
                    "old_reward_delta": float(
                        old_ppo_summary["system_reward"]
                        - reference_summary["system_reward"]
                    ),
                }
            )
            old_comparison_rows.append(
                {
                    "seed": int(seed),
                    "reference": reference_summary,
                    "ppo": old_ppo_summary,
                    "reward_delta": float(
                        old_ppo_summary["system_reward"]
                        - reference_summary["system_reward"]
                    ),
                }
            )
        evaluation_aggregate = _aggregate_evaluations(evaluations)
        old_evaluation_aggregate = _aggregate_evaluations(old_comparison_rows)
        evaluation_summary = {
            "stage": stage,
            "model": "deterministic Time-aware PPO checkpoint",
            "vecnormalize": {
                "training": False,
                "norm_reward": False,
                "source": "vecnormalize.pkl",
            },
            "eval_seeds": eval_seeds,
            "per_seed": evaluations,
            "aggregate": evaluation_aggregate,
            "previous_ppo_aggregate": old_evaluation_aggregate,
            "old_ppo_aggregate": old_evaluation_aggregate,
            "reference_source": str(
                previous_stage_dir / "evaluation_summary.json"
            ),
            "old_ppo_source": str(
                previous_stage_dir / "evaluation_summary.json"
            ),
            "previous_ppo_source": str(
                previous_stage_dir / "evaluation_summary.json"
            ),
            "interpretation": (
                f"Reference 和 Stage{previous_stage} PPO 复用同 seed 结果；"
                f"Stage{stage} 只新增 norm_reward={norm_reward} 的 Time-aware PPO，"
                "系统 reward 评估保持 raw。"
            ),
        }
        with (run_dir / "evaluation_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(_json_safe(evaluation_summary), handle, ensure_ascii=False, indent=2)

        # Observation 语义变更不影响 Reference；复用第十二阶段已验收的
        # 20-slot 回归，避免无意义地再次运行确定性底层闭环。
        previous_reference_replay = previous_stage_summary["correctness"][
            "reference_replay_20_slot"
        ]
        reference_replay = {
            "task_total": int(previous_reference_replay["tasks"]),
            "execution_counts": previous_reference_replay["execution_counts"],
            "nan_inf_count": int(previous_reference_replay["nan_inf_count"]),
            "kkt_failure_count": int(previous_reference_replay["kkt_failure_count"]),
        }
        reference_replay_status = previous_reference_replay.get("status") == "PASS"

        training_rollout = _finalize_accumulator(callback_state.aggregate)
        hard_condition_keys = (
            "ppo_action_mismatch_count",
            "future_request_leakage",
            "future_trajectory_leakage",
            "capacity_violations",
            "compute_violations",
            "bandwidth_violations",
            "uav_violations",
            "invalid_service_ids",
            "invalid_node_ids",
            "nan_inf_count",
            "kkt_failure_count",
        )
        training_hard_conditions = {
            key: int(training_rollout[key]) for key in hard_condition_keys
        }
        reward_nonfinite_count = int(
            sum(
                int(
                    record.get("reward_diagnostics", {})
                    .get(scale, {})
                    .get("nonfinite_count", 0)
                )
                for record in update_records
                for scale in ("raw", "normalized")
            )
        )
        training_hard_conditions["nan_inf_count"] += reward_nonfinite_count
        held_out_hard_conditions = {
            "reference": [
                {
                    key: int(item["reference"].get(key, 0))
                    for key in hard_condition_keys
                }
                for item in evaluations
            ],
            "ppo": [
                {
                    key: int(item["ppo"].get(key, 0))
                    for key in hard_condition_keys
                }
                for item in evaluations
            ],
        }
        evaluation_hard_zero = all(
            value == 0
            for group in held_out_hard_conditions.values()
            for item in group
            for value in item.values()
        )
        training_hard_zero = all(value == 0 for value in training_hard_conditions.values())
        critic_comparison = _critic_comparison(
            update_records,
            previous_stage_summary,
        )
        technical_correctness_pass = (
            training_hard_zero and evaluation_hard_zero and reference_replay_status
        )
        normalization_decision = _normalization_decision(
            update_records,
            previous_stage_summary,
            evaluation_summary,
            technical_correctness_pass=technical_correctness_pass,
            training_rollout=training_rollout,
        )
        # Stage14 PASS 表示受控实验本身完整且无硬错误；是否保留
        # norm_reward=True 由 normalization_decision 单独给出。
        status = "PASS" if technical_correctness_pass else "FAIL"

        _write_update_metrics_artifacts(run_dir, update_records)
        config_snapshot = run_dir / "config_snapshot.json"
        with config_snapshot.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(ppo_config), handle, ensure_ascii=False, indent=2)

        training_summary = {
            "stage": stage,
            "status": status,
            "technical_correctness": "PASS" if technical_correctness_pass else "FAIL",
            "run_id": run_dir.name,
            "norm_reward": norm_reward,
            "previous_stage_run": str(previous_stage_dir),
            "previous_stage13_run": (
                str(previous_stage_dir) if stage == 14 else None
            ),
            "previous_stage12_run": (
                str(previous_stage_dir) if stage == 13 else None
            ),
            "training": {
                "total_timesteps": total_timesteps,
                "ppo_updates": len(update_records),
                "episodes_started": episodes_started,
                "episodes_completed": len(callback_state.episodes),
                "train_seed_base": train_seed_base,
                "train_episode_seeds": train_episode_seeds,
                "eval_seeds": eval_seeds,
                "device": "cpu",
                "policy": "MlpPolicy",
                "action_space": "Discrete(2)",
                "observation_shape": [8],
                "state_features": list(PPO_CACHE_FEATURE_NAMES),
                "state_change": "delta_J -> episode_progress at feature index 2",
                "episode_horizon": int(simulation["slot_count"]),
                "episode_termination": {
                    "terminated": True,
                    "truncated": False,
                },
                "vecnormalize": {
                    "norm_obs": True,
                    "norm_reward": norm_reward,
                    "clip_obs": 10.0,
                },
                "learning_rate": float(ppo_section.get("learning_rate", 3.0e-4)),
                "n_steps": n_steps,
                "batch_size": int(ppo_section.get("batch_size", 64)),
                "n_epochs": int(ppo_section.get("n_epochs", 10)),
                "gamma": float(ppo_section.get("gamma", 0.99)),
                "gae_lambda": float(ppo_section.get("gae_lambda", 0.95)),
                "clip_range": float(ppo_section.get("clip_range", 0.2)),
                "ent_coef": float(ppo_section.get("ent_coef", 0.0)),
                "vf_coef": float(ppo_section.get("vf_coef", 0.5)),
                "max_grad_norm": float(ppo_section.get("max_grad_norm", 0.5)),
                "target_kl": None,
            },
            "training_metrics": final_training_metrics,
            "training_metrics_by_update": update_records,
            "critic_trend": _critic_trend(update_records),
            "critic_comparison": critic_comparison,
            "normalization_decision": normalization_decision,
            "training_rollout": training_rollout,
            "training_episodes": callback_state.episodes,
            "probe_states": probe_snapshots,
            "held_out": evaluation_summary,
            "correctness": {
                "check_env": check_status,
                "model_learn_called": len(update_records),
                "optimizer_step_called_directly": 0,
                "hard_conditions_zero": training_hard_zero,
                "technical_correctness_pass": technical_correctness_pass,
                "training": training_hard_conditions,
                "held_out": held_out_hard_conditions,
                "held_out_hard_conditions_zero": evaluation_hard_zero,
                "reference_replay_20_slot": {
                    "status": "PASS" if reference_replay_status else "FAIL",
                    "tasks": int(reference_replay["task_total"]),
                    "execution_counts": reference_replay["execution_counts"],
                    "nan_inf_count": int(reference_replay["nan_inf_count"]),
                    "kkt_failure_count": int(reference_replay["kkt_failure_count"]),
                },
            },
            "artifacts": {
                "model": "model.zip",
                "vecnormalize": "vecnormalize.pkl",
                "config_snapshot": "config_snapshot.json",
                "training_summary": "training_summary.json",
                "update_metrics_json": "update_metrics.json",
                "update_metrics_csv": "update_metrics.csv",
                "evaluation_summary": "evaluation_summary.json",
            },
        }
        with (run_dir / "training_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(_json_safe(training_summary), handle, ensure_ascii=False, indent=2)
        return training_summary
    except Exception:
        if vector_environment is not None:
            vector_environment.close()
        shutil.rmtree(run_dir, ignore_errors=True)
        raise


def _print_training_summary(summary: Mapping[str, Any]) -> None:
    training = summary["training"]
    rollout = summary["training_rollout"]
    correctness = summary["correctness"]
    held_out = summary["held_out"]
    print("training_result = PASS")
    print(f"total_timesteps = {training['total_timesteps']}")
    print(f"PPO updates = {training['ppo_updates']}")
    print(f"episodes = {training['episodes_started']}")
    print(f"device = {training['device']}")
    print(f"KEEP = {rollout['action_counts']['KEEP']}")
    print(f"UPDATE = {rollout['action_counts']['UPDATE']}")
    print(f"effective UPDATE = {rollout['ppo_effective_update_count']}")
    print(f"UPDATE noop = {rollout['ppo_update_noop_count']}")
    print(f"action mismatch = {rollout['ppo_action_mismatch_count']}")
    for key in REQUIRED_PPO_METRICS:
        print(f"{key} = {summary['training_metrics'][f'train/{key}']}")
    for key in (
        "nan_inf_count",
        "future_request_leakage",
        "future_trajectory_leakage",
        "capacity_violations",
        "uav_violations",
        "kkt_failure_count",
    ):
        print(f"{key} = {correctness[key]}")
    print(f"Reference reward = {held_out['reference']['reward']}")
    print(f"PPO reward = {held_out['ppo']['reward']}")
    print(f"PPO cache_changed_slots = {held_out['ppo']['cache_changed_slots']}")
    print(f"PPO uav_moved_slots = {held_out['ppo']['uav_moved_slots']}")
    print(f"artifacts = {summary['artifacts']}")
    print("long_training_started = 0")


def _print_medium_summary(summary: Mapping[str, Any]) -> None:
    """Stage13/14 命令行只打印验收摘要，详细内容写入 JSON/CSV。"""

    training = summary["training"]
    rollout = summary["training_rollout"]
    correctness = summary["correctness"]
    aggregate = summary["held_out"]["aggregate"]
    print(f"training_result = {summary['status']}")
    print(f"total_timesteps = {training['total_timesteps']}")
    print(f"PPO updates = {training['ppo_updates']}")
    print(f"episodes_started = {training['episodes_started']}")
    print(f"episodes_completed = {training['episodes_completed']}")
    print(f"device = {training['device']}")
    print(f"KEEP = {rollout['action_counts']['KEEP']}")
    print(f"UPDATE = {rollout['action_counts']['UPDATE']}")
    print(f"effective UPDATE = {rollout['ppo_effective_update_count']}")
    print(f"UPDATE noop = {rollout['ppo_update_noop_count']}")
    print(f"action mismatch = {rollout['ppo_action_mismatch_count']}")
    for key in REQUIRED_PPO_METRICS:
        print(f"{key} = {summary['training_metrics'][f'train/{key}']}")
    for key, value in correctness["training"].items():
        print(f"{key} = {value}")
    print(f"held_out_mean_reward_delta = {aggregate['mean_reward_delta']}")
    print(f"held_out_std_reward_delta = {aggregate['std_reward_delta']}")
    print(f"held_out_ppo_wins = {aggregate['ppo_wins']}")
    print(f"artifacts = {summary['artifacts']}")
    print("long_training_started = 0")


def _print_stage15_preparation(settings: Mapping[str, Any]) -> None:
    """默认 CLI 只做 Stage15 参数检查，绝不创建训练环境或调用 learn。"""

    print("stage15_configuration = PASS")
    print(f"state_dim = {len(settings['state_features'])}")
    print("norm_obs = true")
    print("norm_reward = true")
    print(f"total_timesteps = {settings['total_timesteps']}")
    print(f"updates = {settings['updates']}")
    print(f"validation_seeds = {settings['validation_seeds']}")
    print(f"test_seeds = {settings['test_seeds']}")
    print(f"eval_freq = {settings['eval_freq']}")
    print("model.learn calls = 0")
    print("formal training started = NO")


def _print_stage15_summary(summary: Mapping[str, Any]) -> None:
    """Stage15 完成后只打印 raw validation/test 摘要。"""

    training = summary["training"]
    best_update = summary.get("best", {}).get("update")
    best_checkpoint = (
        "NA" if best_update is None else f"update{int(best_update):02d}"
    )
    rollout = summary.get("training_rollout", {})
    action_counts = rollout.get("action_counts", {})
    print("===== Stage15 Completed =====", flush=True)
    print(f"timesteps = {training.get('total_timesteps', 'NA')}", flush=True)
    print(f"updates = {training.get('ppo_updates', 'NA')}", flush=True)
    print(
        f"completed episodes = {training.get('episodes_completed', 'NA')}",
        flush=True,
    )
    print(f"best checkpoint = {best_checkpoint}", flush=True)
    print(
        "best validation raw reward = "
        f"{_stage15_display_value(summary.get('best', {}).get('validation_mean_original_system_reward'))}",
        flush=True,
    )
    print(
        f"KEEP = {_stage15_display_count(action_counts.get('KEEP'))}",
        flush=True,
    )
    print(
        f"UPDATE = {_stage15_display_count(action_counts.get('UPDATE'))}",
        flush=True,
    )
    print(
        "effective UPDATE = "
        f"{_stage15_display_count(rollout.get('ppo_effective_update_count'))}",
        flush=True,
    )
    print(
        f"noop = {_stage15_display_count(rollout.get('ppo_update_noop_count'))}",
        flush=True,
    )
    print(
        f"action mismatch = {_stage15_display_count(rollout.get('ppo_action_mismatch_count', 0))}",
        flush=True,
    )
    print(
        f"hard-stop violations = {_stage15_display_count(_stage15_hard_stop_violation_total(summary))}",
        flush=True,
    )

    test_summary = summary["test"]
    aggregate = test_summary["aggregate"]
    test_seed_count = len(test_summary.get("eval_seeds", []))
    print("===== Final Test =====", flush=True)
    print(
        "Reference mean raw reward = "
        f"{_stage15_display_value(aggregate.get('mean_reference_reward'))}",
        flush=True,
    )
    print(
        "PPO mean raw reward = "
        f"{_stage15_display_value(aggregate.get('mean_ppo_reward'))}",
        flush=True,
    )
    print(
        "mean delta = "
        f"{_stage15_display_value(aggregate.get('mean_reward_delta'))}",
        flush=True,
    )
    print(
        "std delta = "
        f"{_stage15_display_value(aggregate.get('std_reward_delta'))}",
        flush=True,
    )
    print(
        f"PPO wins = {_stage15_display_count(aggregate.get('ppo_wins'))}/{test_seed_count}",
        flush=True,
    )
    rates = _stage15_test_execution_rates(test_summary)
    for label, key in (
        ("Local", "local"),
        ("SV", "service_vehicle"),
        ("RSU", "rsu"),
        ("UAV", "uav"),
        ("Cloud", "cloud"),
        ("Failed", "failed"),
    ):
        print(
            f"Reference {label} rate = {_stage15_display_value(rates['reference'][key], digits=2)}%",
            flush=True,
        )
        print(
            f"PPO {label} rate = {_stage15_display_value(rates['ppo'][key], digits=2)}%",
            flush=True,
        )
    plotting = summary.get("plotting", {})
    plotting_status = (
        plotting.get("status", "NA") if isinstance(plotting, Mapping) else "NA"
    )
    print(f"training = {summary.get('status', 'NA')}", flush=True)
    print(f"plotting = {plotting_status}", flush=True)
    print(f"artifacts = {summary.get('artifacts', {})}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    """默认只检查参数；只有显式 --stage15 才允许用户启动正式训练。"""

    parser = argparse.ArgumentParser(description="Adaptive cache PPO entry")
    parser.add_argument(
        "--stage15",
        action="store_true",
        help="执行用户明确启动的 Stage15 25600-step 正式训练",
    )
    args = parser.parse_args(argv)
    if not args.stage15:
        _print_stage15_preparation(validate_stage15_configuration())
        return
    summary = run_stage15_training()
    _print_stage15_summary(summary)


if __name__ == "__main__":  # pragma: no cover - 命令行入口
    main()


__all__ = [
    "build_ppo_environment",
    "build_sb3_ppo",
    "prepare_ppo_config",
    "run_controlled_training",
    "run_medium_training",
    "run_stage15_training",
    "validate_stage15_configuration",
    "run_interface_smoke",
]
