"""当前第四章 MultiBinary 配置触发器的标准 SB3 PPO 入口。

本文件只做项目适配：环境构造、VecNormalize 和配置读取。策略分布、GAE、
clip loss、entropy 与优化过程全部直接复用 stable_baselines3.PPO。
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:  # pragma: no cover - 仅允许无SB3环境导入本模块
    class BaseCallback:  # type: ignore[no-redef]
        """让配置/静态检查在未安装 SB3 时仍可导入 train_ppo。"""

        def __init__(self, verbose: int = 0) -> None:
            self.verbose = int(verbose)

try:
    from .data import load_config
    from .environment import MECEnvironment
    from .ppo_env import AdaptiveCachePPOEnv, sb3_check_env
except ImportError:  # pragma: no cover
    from data import load_config
    from environment import MECEnvironment
    from ppo_env import AdaptiveCachePPOEnv, sb3_check_env


PROJECT_ROOT = Path(__file__).resolve().parent


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, Mapping) else {}


def prepare_ppo_config(
    config: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """返回 PPO 模式配置副本，不改写默认 Reference 配置文件。"""

    if isinstance(config, Mapping):
        prepared = copy.deepcopy(dict(config))
    else:
        prepared = load_config() if config is None else load_config(config)
        prepared = copy.deepcopy(prepared)
    controller = dict(_section(prepared, "cache_controller"))
    controller["mode"] = "ppo"
    prepared["cache_controller"] = controller
    return prepared


def build_ppo_environment(
    config: Mapping[str, Any] | str | Path | None = None,
    *,
    episode_seed_base: int | None = None,
    seed_mode: str = "sequential",
    fixed_seed: int | None = None,
) -> AdaptiveCachePPOEnv:
    prepared = prepare_ppo_config(config)
    return AdaptiveCachePPOEnv(
        MECEnvironment(prepared),
        episode_seed_base=episode_seed_base,
        seed_mode=seed_mode,
        fixed_seed=fixed_seed,
    )


def build_sb3_ppo(environment: Any, config: Mapping[str, Any] | None = None) -> Any:
    """构造官方 MlpPolicy PPO；MultiBinary 使用 SB3 原生 Bernoulli 分布。"""

    try:
        from stable_baselines3 import PPO
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("当前 conda 环境缺少 stable-baselines3。") from error
    source = {} if config is None else config
    ppo = _section(source, "ppo")
    project = _section(source, "project")
    return PPO(
        "MlpPolicy",
        environment,
        learning_rate=float(ppo.get("learning_rate", 3.0e-4)),
        n_steps=int(ppo.get("n_steps", 256)),
        batch_size=int(ppo.get("batch_size", 64)),
        n_epochs=int(ppo.get("n_epochs", 10)),
        gamma=float(ppo.get("gamma", 0.99)),
        gae_lambda=float(ppo.get("gae_lambda", 0.95)),
        clip_range=float(ppo.get("clip_range", 0.2)),
        ent_coef=float(ppo.get("ent_coef", 0.0)),
        vf_coef=float(ppo.get("vf_coef", 0.5)),
        max_grad_norm=float(ppo.get("max_grad_norm", 0.5)),
        seed=int(ppo.get("seed", project.get("seed", 0))),
        device=str(ppo.get("device", "cpu")),
        verbose=0,
    )


def build_vecnormalize_environment(
    config: Mapping[str, Any] | str | Path | None = None,
    *,
    episode_seed_base: int | None = None,
) -> Any:
    """直接复用 SB3 DummyVecEnv/VecMonitor/VecNormalize。"""

    try:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("当前 conda 环境缺少 stable-baselines3。") from error
    prepared = prepare_ppo_config(config)

    def factory() -> Any:
        return build_ppo_environment(prepared, episode_seed_base=episode_seed_base)

    vector = VecMonitor(DummyVecEnv([factory]))
    ppo = _section(prepared, "ppo")
    return VecNormalize(
        vector,
        norm_obs=bool(ppo.get("norm_obs", True)),
        norm_reward=bool(ppo.get("norm_reward", True)),
        clip_obs=float(ppo.get("clip_obs", 10.0)),
    )


def run_interface_smoke(
    config: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """创建 model 并 predict 一次，不调用 learn。"""

    prepared = prepare_ppo_config(config)
    seed = int(_section(prepared, "ppo").get("seed", 0))
    # check_env 会自动多次 reset；固定 seed 避免把接口检查误当成正式 episode
    # 顺序推进，也避免离线 Trace reserve 语义被测试消耗。
    environment = build_ppo_environment(
        prepared, seed_mode="fixed", fixed_seed=seed
    )
    try:
        status = sb3_check_env(environment)
        model = build_sb3_ppo(environment, prepared)
        observation, reset_info = environment.reset(seed=seed)
        action, _state = model.predict(observation, deterministic=True)
        next_observation, reward, terminated, truncated, info = environment.step(action)
        return {
            "check_env": status,
            "action_space": str(environment.action_space),
            "action_layout": reset_info["action_layout"],
            "observation_shape": tuple(int(item) for item in observation.shape),
            "observation_dtype": str(observation.dtype),
            "observation_finite": bool(np.all(np.isfinite(observation))),
            "next_observation_finite": bool(np.all(np.isfinite(next_observation))),
            "reward_finite": bool(np.isfinite(reward)),
            "formula_residual": float(
                reward
                - (
                    float(info["system_utility"]["task_service_utility"])
                    - float(info["cache_cost"])
                    - float(info["uav_move_cost"])
                )
            ),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "model_device": str(model.device),
            "model_learn_calls": 0,
        }
    finally:
        environment.close()


def run_semantic_smoke(
    config: Mapping[str, Any] | str | Path | None = None,
    *,
    seed: int = 20271811,
) -> dict[str, Any]:
    """执行节点级 trigger 与 10-slot 约束验收，不创建训练 rollout。"""

    prepared = prepare_ppo_config(config)

    def one_step(action_builder: Any) -> dict[str, Any]:
        environment = build_ppo_environment(
            prepared, seed_mode="fixed", fixed_seed=int(seed)
        )
        try:
            _observation, _reset_info = environment.reset(seed=int(seed))
            old_cache = np.asarray(environment.environment._cache, dtype=float).copy()
            old_uav = np.asarray(environment.environment._uav_positions, dtype=float).copy()
            action = np.asarray(action_builder(environment), dtype=np.int8)
            _next, reward, _done, _truncated, info = environment.step(action)
            return {
                "action": action,
                "old_cache": old_cache,
                "new_cache": np.asarray(environment.environment._cache, dtype=float).copy(),
                "old_uav": old_uav,
                "new_uav": np.asarray(environment.environment._uav_positions, dtype=float).copy(),
                "reward": float(reward),
                "info": info,
            }
        finally:
            environment.close()

    def zeros(environment: AdaptiveCachePPOEnv) -> np.ndarray:
        return np.zeros(environment.action_space.shape, dtype=np.int8)

    keep = one_step(zeros)
    cache_only = one_step(
        lambda environment: np.eye(
            1,
            environment.action_space.shape[0],
            k=2,
            dtype=np.int8,
        )[0]
    )
    uav_only = one_step(
        lambda environment: np.eye(
            1,
            environment.action_space.shape[0],
            k=environment._cache_node_count,
            dtype=np.int8,
        )[0]
    )
    mixed = one_step(
        lambda environment: np.asarray(
            [
                int(index in {2, environment._cache_node_count})
                for index in range(environment.action_space.shape[0])
            ],
            dtype=np.int8,
        )
    )

    cache_other_rows_unchanged = bool(
        np.array_equal(cache_only["old_cache"][:2], cache_only["new_cache"][:2])
        and np.array_equal(cache_only["old_cache"][3:], cache_only["new_cache"][3:])
    )
    uav_cache_unchanged = bool(
        np.array_equal(uav_only["old_cache"], uav_only["new_cache"])
    )
    uav_other_unchanged = bool(
        np.array_equal(uav_only["old_uav"][1:], uav_only["new_uav"][1:])
    )

    # 直接复用正式确定性函数构造两个精确 no-op 边界，不读取真实任务。
    try:
        from .solver import solve_cooperative_cache, select_triggered_uav_positions
    except ImportError:  # pragma: no cover
        from solver import solve_cooperative_cache, select_triggered_uav_positions
    physical = MECEnvironment(prepared)
    try:
        physical.reset(seed=int(seed))
        trigger = np.zeros(physical._cache.shape[0], dtype=bool)
        trigger[2] = True
        first = solve_cooperative_cache(
            physical._cache,
            trigger,
            physical._service_sizes,
            physical._cache_capacities,
            cache_prediction=physical._last_node_prediction,
            node_types=physical._service_node_types,
            environment=physical,
            integer_scale=1,
        )
        candidate = np.asarray(first["candidate_cache"], dtype=float)
        second = solve_cooperative_cache(
            candidate,
            trigger,
            physical._service_sizes,
            physical._cache_capacities,
            cache_prediction=physical._last_node_prediction,
            node_types=physical._service_node_types,
            environment=physical,
            integer_scale=1,
        )
        second_cache = np.asarray(second["candidate_cache"], dtype=float)
        cache_noop = bool(np.array_equal(candidate, second_cache))
        cache_noop_cost = float(
            physical.cache_update_cost(
                candidate[2],
                second_cache[2],
                node_type=physical._service_node_types[2],
                service_sizes=physical._service_sizes,
            )["total_cost"]
        )
        uav_trigger = np.zeros(len(physical._uav_positions), dtype=bool)
        uav_trigger[0] = True
        uav_noop_result = select_triggered_uav_positions(
            physical._uav_positions,
            previous_cache=physical._cache,
            requested_uav_triggers=uav_trigger,
            cache_prediction=None,
            node_types=physical._service_node_types,
            uav_service_indices=physical._uav_service_indices,
            environment=physical,
            max_speed_mps=physical.config["uav"]["max_speed_mps"],
            slot_duration_s=physical.config["simulation"]["decision_interval_s"],
            region=physical.config["uav"]["region"],
            minimum_separation_m=physical.config["uav"]["minimum_safety_distance_m"],
        )
        uav_noop = not bool(np.any(uav_noop_result["actual_movement"]))
        uav_noop_cost = float(
            physical.uav_movement_cost(
                physical._uav_positions[0],
                uav_noop_result["final_positions"][0],
                movement_epsilon_m=1.0e-9,
            )["total_cost"]
        )
    finally:
        physical.close()

    smoke_environment = build_ppo_environment(
        prepared, seed_mode="fixed", fixed_seed=int(seed)
    )
    formula_residuals: list[float] = []
    reward_components = {"Q": 0.0, "cache_cost": 0.0, "uav_cost": 0.0, "reward": 0.0}
    totals = {
        "future_request_leakage": 0,
        "future_trajectory_leakage": 0,
        "capacity_violations": 0,
        "compute_violations": 0,
        "bandwidth_violations": 0,
        "uav_speed_violations": 0,
        "uav_region_violations": 0,
        "uav_safety_violations": 0,
        "nan_inf_count": 0,
        "kkt_failure_count": 0,
    }
    try:
        smoke_environment.reset(seed=int(seed))
        for slot in range(10):
            action = np.zeros(smoke_environment.action_space.shape, dtype=np.int8)
            if slot % 4 == 1:
                action[2] = 1
            elif slot % 4 == 2:
                action[smoke_environment._cache_node_count] = 1
            elif slot % 4 == 3:
                action[[2, smoke_environment._cache_node_count]] = 1
            _obs, reward, done, _truncated, info = smoke_environment.step(action)
            formula_residuals.append(
                float(
                    reward
                    - (
                        info["system_utility"]["task_service_utility"]
                        - info["cache_cost"]
                        - info["uav_move_cost"]
                    )
                )
            )
            reward_components["Q"] += float(
                info["system_utility"]["task_service_utility"]
            )
            reward_components["cache_cost"] += float(info["cache_cost"])
            reward_components["uav_cost"] += float(info["uav_move_cost"])
            reward_components["reward"] += float(reward)
            for key in totals:
                totals[key] += int(info.get(key, 0))
            if done:
                break
    finally:
        smoke_environment.close()

    return {
        "keep_all": {
            "cache_exact": bool(np.array_equal(keep["old_cache"], keep["new_cache"])),
            "uav_exact": bool(np.array_equal(keep["old_uav"], keep["new_uav"])),
            "cache_cost": float(keep["info"]["cache_cost"]),
            "uav_cost": float(keep["info"]["uav_move_cost"]),
        },
        "cache_trigger_one": {"other_rows_exact": cache_other_rows_unchanged},
        "uav_trigger_one": {
            "cache_exact": uav_cache_unchanged,
            "other_uavs_exact": uav_other_unchanged,
        },
        "mixed_trigger_executed": True,
        "cache_trigger_noop": {"actual_update": not cache_noop, "cost": cache_noop_cost},
        "uav_trigger_noop": {"actual_update": not uav_noop, "cost": uav_noop_cost},
        "formula_residual_max": float(max(abs(item) for item in formula_residuals)),
        "reward_components_10_slots": reward_components,
        "diagnostics": totals,
        "steps": len(formula_residuals),
        "model_learn_calls": 0,
    }


# ---------------------------------------------------------------------------
# 5120-step learning-effectiveness check
# ---------------------------------------------------------------------------

LEARNING_CHECK_TIMESTEPS = 5120
LEARNING_CHECK_EXPECTED_UPDATES = 20
LEARNING_CHECK_HARD_KEYS = (
    "future_request_leakage",
    "future_trajectory_leakage",
    "future_mobility_leakage",
    "capacity_violations",
    "compute_violations",
    "bandwidth_violations",
    "uav_speed_violations",
    "uav_region_violations",
    "uav_safety_violations",
    "invalid_service_ids",
    "invalid_node_ids",
    "nan_inf_count",
    "kkt_failure_count",
)
LEARNING_CHECK_EXECUTION_KEYS = (
    ("local", "Local"),
    ("service_vehicle", "SV"),
    ("rsu", "RSU"),
    ("uav", "UAV"),
    ("cloud", "Cloud"),
)


def _json_safe(value: Any) -> Any:
    """把配置、诊断和 numpy 标量转换为可审计的 JSON 值。"""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple, np.ndarray)):
        return json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False)
    if isinstance(value, (np.generic, Path)):
        return _json_safe(value)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """写入动态布局的审计 CSV，字段顺序由第一条记录及其后续字段决定。"""

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            name = str(key)
            if name not in seen:
                seen.add(name)
                fieldnames.append(name)
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _bool_vector(value: Any, length: int) -> np.ndarray:
    if value is None:
        return np.zeros(int(length), dtype=bool)
    array = np.asarray(value, dtype=bool).reshape(-1)
    if array.size != int(length):
        raise ValueError(
            f"learning-check动作诊断长度错误：expected={length}, got={array.size}"
        )
    return array.copy()


def _layout_copy(layout: Mapping[str, Any]) -> dict[str, Any]:
    """只从 environment.action_layout 读取 bit/node 映射。"""

    copied = {
        "cache_node_ids": tuple(str(item) for item in layout["cache_node_ids"]),
        "cache_node_types": tuple(str(item) for item in layout["cache_node_types"]),
        "uav_ids": tuple(str(item) for item in layout["uav_ids"]),
        "cache_trigger_indices": tuple(
            int(item) for item in layout["cache_trigger_indices"]
        ),
        "uav_position_trigger_indices": tuple(
            int(item) for item in layout["uav_position_trigger_indices"]
        ),
        "action_length": int(layout["action_length"]),
    }
    if len(copied["cache_node_ids"]) != len(copied["cache_trigger_indices"]):
        raise ValueError("action_layout的cache节点与trigger索引数量不一致。")
    if len(copied["uav_ids"]) != len(copied["uav_position_trigger_indices"]):
        raise ValueError("action_layout的UAV节点与trigger索引数量不一致。")
    if copied["action_length"] != (
        len(copied["cache_trigger_indices"])
        + len(copied["uav_position_trigger_indices"])
    ):
        raise ValueError("action_layout.action_length与显式索引不一致。")
    return copied


class _MetricAccumulator:
    """按 slot 累计原始 G、执行指标和 MultiBinary 动作统计。"""

    def __init__(self, layout: Mapping[str, Any]) -> None:
        self.layout = _layout_copy(layout)
        self.cache_count = len(self.layout["cache_node_ids"])
        self.uav_count = len(self.layout["uav_ids"])
        self.action_length = int(self.layout["action_length"])
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.raw_g_sum = 0.0
        self.q_sum = 0.0
        self.cache_cost_sum = 0.0
        self.uav_cost_sum = 0.0
        self.delay_sum = 0.0
        self.energy_sum = 0.0
        self.completed_count = 0
        self.failed_count = 0
        self.arrival_count = 0
        self.edge_cache_available_count = 0
        self.execution_counts = {key: 0 for key, _label in LEARNING_CHECK_EXECUTION_KEYS}
        self.cache_requested = np.zeros(self.cache_count, dtype=np.int64)
        self.cache_effective_update = np.zeros(self.cache_count, dtype=np.int64)
        self.cache_noop = np.zeros(self.cache_count, dtype=np.int64)
        self.cache_services_changed = np.zeros(self.cache_count, dtype=np.int64)
        self.uav_requested = np.zeros(self.uav_count, dtype=np.int64)
        self.uav_movement = np.zeros(self.uav_count, dtype=np.int64)
        self.uav_noop = np.zeros(self.uav_count, dtype=np.int64)
        self.diagnostics = {key: 0 for key in LEARNING_CHECK_HARD_KEYS}

    def add(self, info: Mapping[str, Any]) -> None:
        utility = info.get("system_utility", {})
        if not isinstance(utility, Mapping):
            utility = {}
        raw_g = _finite_float(info.get("ppo_raw_reward", utility.get("reward", 0.0)))
        q_value = _finite_float(utility.get("task_service_utility", 0.0))
        self.steps += 1
        self.raw_g_sum += raw_g
        self.q_sum += q_value
        self.cache_cost_sum += _finite_float(info.get("cache_cost", 0.0))
        self.uav_cost_sum += _finite_float(info.get("uav_move_cost", 0.0))
        self.delay_sum += _finite_float(info.get("total_delay", 0.0))
        self.energy_sum += _finite_float(info.get("total_energy", 0.0))

        raw_counts = info.get("execution_counts", {})
        if not isinstance(raw_counts, Mapping):
            raw_counts = {}
        for key, _label in LEARNING_CHECK_EXECUTION_KEYS:
            self.execution_counts[key] += int(max(0, int(raw_counts.get(key, 0))))
        # 当前 slot 使用原始 info 计数，保证 Failed 与到达任务闭合。
        current_completed = sum(
            max(0, int(raw_counts.get(key, 0)))
            for key, _label in LEARNING_CHECK_EXECUTION_KEYS
        )
        self.completed_count += current_completed
        current_failed = max(0, int(info.get("failed_tasks", 0)))
        self.failed_count += current_failed
        self.arrival_count += current_completed + current_failed
        raw_edge_available = info.get("edge_cache_available")
        if raw_edge_available is not None:
            edge_available = np.asarray(raw_edge_available, dtype=bool).reshape(-1)
            self.edge_cache_available_count += int(np.sum(edge_available))
        else:
            self.edge_cache_available_count += max(
                0, int(info.get("edge_cache_availability_count", 0))
            )

        cache_trigger = _bool_vector(
            info.get("requested_cache_triggers"), self.cache_count
        )
        cache_update = _bool_vector(
            info.get("actual_cache_update", info.get("cache_changed")),
            self.cache_count,
        )
        self.cache_requested += cache_trigger.astype(np.int64)
        self.cache_effective_update += cache_update.astype(np.int64)
        self.cache_noop += (cache_trigger & ~cache_update).astype(np.int64)
        services_changed = info.get("services_changed", ())
        if isinstance(services_changed, (list, tuple)):
            for index in range(min(self.cache_count, len(services_changed))):
                item = services_changed[index]
                if isinstance(item, (list, tuple, np.ndarray)):
                    self.cache_services_changed[index] += int(len(item))

        uav_trigger = _bool_vector(
            info.get("requested_uav_triggers"), self.uav_count
        )
        uav_movement = _bool_vector(
            info.get("actual_uav_movement"), self.uav_count
        )
        self.uav_requested += uav_trigger.astype(np.int64)
        self.uav_movement += uav_movement.astype(np.int64)
        self.uav_noop += (uav_trigger & ~uav_movement).astype(np.int64)

        for key in LEARNING_CHECK_HARD_KEYS:
            self.diagnostics[key] += max(0, int(info.get(key, 0)))

    def _rate(self, value: np.ndarray | int | float, denominator: int | None = None) -> Any:
        divisor = self.steps if denominator is None else int(denominator)
        if isinstance(value, np.ndarray):
            return np.divide(
                value.astype(float),
                float(divisor),
                out=np.zeros(value.shape, dtype=float),
                where=divisor > 0,
            )
        return float(value) / float(divisor) if divisor > 0 else 0.0

    def row(self, *, mode: str) -> dict[str, Any]:
        if mode not in {"episode", "update"}:
            raise ValueError("MetricAccumulator mode必须是episode或update。")
        cost_divisor = 1.0 if mode == "episode" else float(max(1, self.steps))
        arrival_divisor = max(1, self.arrival_count)
        completed_divisor = max(1, self.completed_count)
        non_cloud_count = sum(
            int(self.execution_counts[key])
            for key in ("local", "service_vehicle", "rsu", "uav")
        )
        row: dict[str, Any] = {
            "steps": int(self.steps),
            "sum_raw_G": float(self.raw_g_sum),
            "mean_raw_G": float(self.raw_g_sum / max(1, self.steps)),
            "sum_Q": float(self.q_sum),
            "mean_Q": float(self.q_sum / max(1, self.steps)),
            "cache_cost": float(self.cache_cost_sum / cost_divisor),
            "uav_cost": float(self.uav_cost_sum / cost_divisor),
            "sum_cache_cost": float(self.cache_cost_sum),
            "sum_uav_cost": float(self.uav_cost_sum),
            "mean_delay": float(self.delay_sum / completed_divisor),
            "mean_vehicle_energy": float(self.energy_sum / completed_divisor),
            "completed_task_count": int(self.completed_count),
            "arrival_count": int(self.arrival_count),
            "failed_count": int(self.failed_count),
            # non_cloud/执行率按已完成任务统计；Failed 不进入该分母。
            "non_cloud_rate": float(non_cloud_count / float(completed_divisor)),
            "edge_cache_availability_rate": float(
                self.edge_cache_available_count / float(arrival_divisor)
            ),
            # 兼容旧汇总字段，但语义与 edge_cache_availability_rate 相同。
            "cache_service_availability": float(
                self.edge_cache_available_count / float(arrival_divisor)
            ),
            "edge_execution_count": int(
                self.execution_counts["service_vehicle"]
                + self.execution_counts["rsu"]
                + self.execution_counts["uav"]
            ),
            "edge_execution_rate": float(
                (
                    self.execution_counts["service_vehicle"]
                    + self.execution_counts["rsu"]
                    + self.execution_counts["uav"]
                )
                / float(completed_divisor)
            ),
        }
        for key, label in LEARNING_CHECK_EXECUTION_KEYS:
            count = int(self.execution_counts[key])
            row[f"{label}_count"] = count
            row[f"{label}_rate"] = float(count / float(completed_divisor))

        for index in range(self.cache_count):
            prefix = f"cache_node_{index}"
            bit = int(self.layout["cache_trigger_indices"][index])
            row[f"{prefix}_requested_trigger_count"] = int(self.cache_requested[index])
            row[f"{prefix}_requested_trigger_rate"] = float(
                self.cache_requested[index] / float(max(1, self.steps))
            )
            row[f"{prefix}_effective_update_count"] = int(
                self.cache_effective_update[index]
            )
            row[f"{prefix}_effective_update_rate"] = float(
                self.cache_effective_update[index] / float(max(1, self.steps))
            )
            row[f"{prefix}_noop_trigger_count"] = int(self.cache_noop[index])
            row[f"{prefix}_services_changed_count"] = int(
                self.cache_services_changed[index]
            )
            row[f"bit_{bit}_requested_trigger_count"] = int(self.cache_requested[index])
            row[f"bit_{bit}_requested_trigger_rate"] = float(
                self.cache_requested[index] / float(max(1, self.steps))
            )
        for index in range(self.uav_count):
            prefix = f"uav_node_{index}"
            bit = int(self.layout["uav_position_trigger_indices"][index])
            row[f"{prefix}_requested_move_trigger_count"] = int(self.uav_requested[index])
            row[f"{prefix}_requested_move_trigger_rate"] = float(
                self.uav_requested[index] / float(max(1, self.steps))
            )
            row[f"{prefix}_actual_movement_count"] = int(self.uav_movement[index])
            row[f"{prefix}_actual_movement_rate"] = float(
                self.uav_movement[index] / float(max(1, self.steps))
            )
            row[f"{prefix}_noop_trigger_count"] = int(self.uav_noop[index])
            row[f"bit_{bit}_requested_trigger_count"] = int(self.uav_requested[index])
            row[f"bit_{bit}_requested_trigger_rate"] = float(
                self.uav_requested[index] / float(max(1, self.steps))
            )
        for key, value in self.diagnostics.items():
            row[key] = int(value)
        return row

    def validation_row(self, seed: int) -> dict[str, Any]:
        row = self.row(mode="episode")
        row.update(
            {
                "seed": int(seed),
                "seed_mode": "fixed",
                "raw_G": float(self.raw_g_sum),
                "Q": float(self.q_sum),
                "delay": float(self.delay_sum / max(1, self.completed_count)),
                "energy": float(self.energy_sum / max(1, self.completed_count)),
            }
        )
        return row


class LearningCheckCallback(BaseCallback):
    """记录 5120-step raw system utility 和 PPO update 诊断。

    该 callback 不改变 action、observation、reward 或 solver，只读取 step info。
    PPO 的 train/approx_kl 等数值在下一次 rollout 开始前已由 SB3 写入 logger，
    因此 update 1--20 都能和对应的 256-step rollout 对齐。
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        action_layout: Mapping[str, Any] | None,
        expected_updates: int = LEARNING_CHECK_EXPECTED_UPDATES,
    ) -> None:
        super().__init__(verbose=0)
        self.output_dir = Path(output_dir)
        # DummyVecEnv 的真正 reset 发生在 model.learn 的 _setup_learn 中；
        # 构造器阶段的 MECEnvironment 尚未建立车辆角色，不能提前读取布局。
        self.layout: dict[str, Any] | None = (
            None if action_layout is None else _layout_copy(action_layout)
        )
        self.expected_updates = int(expected_updates)
        self.episode_rows: list[dict[str, Any]] = []
        self.update_rows: list[dict[str, Any]] = []
        self.action_rows: list[dict[str, Any]] = []
        self._episode_metrics: _MetricAccumulator | None = None
        self._rollout_metrics: _MetricAccumulator | None = None
        self._pending_rollout: _MetricAccumulator | None = None
        self._current_episode_seed: int | None = None
        self._completed_episodes = 0
        self._logged_updates = 0
        self.hard_diagnostics = {key: 0 for key in LEARNING_CHECK_HARD_KEYS}

    def _on_training_start(self) -> None:
        # reset 已由 SB3 在 callback.on_training_start 前完成；这里重新读取
        # action_layout 只用于确认训练时没有发生隐式维度/节点映射变化。
        runtime_layout = self.training_env.get_attr("action_layout")[0]
        runtime_layout = _layout_copy(runtime_layout)
        if self.layout is not None and runtime_layout != self.layout:
            raise RuntimeError("训练开始时 environment.action_layout 发生漂移。")
        self.layout = runtime_layout
        self._episode_metrics = _MetricAccumulator(self.layout)
        self._rollout_metrics = _MetricAccumulator(self.layout)

    def _finish_episode(self, *, completed: bool) -> None:
        if self._episode_metrics is None or self._episode_metrics.steps <= 0:
            return
        row = self._episode_metrics.row(mode="episode")
        row.update(
            {
                "episode": int(len(self.episode_rows) + 1),
                "episode_seed": self._current_episode_seed,
                "completed": bool(completed),
            }
        )
        self.episode_rows.append(row)
        if completed:
            self._completed_episodes += 1
        assert self.layout is not None
        self._episode_metrics = _MetricAccumulator(self.layout)
        self._current_episode_seed = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if len(infos) != 1:
            raise RuntimeError("learning-check只支持单环境，无法安全聚合episode指标。")
        info = infos[0]
        if not isinstance(info, Mapping):
            raise RuntimeError("SB3 step info不是mapping，无法记录原始系统指标。")
        if self._current_episode_seed is None:
            seed = info.get("episode_seed")
            self._current_episode_seed = None if seed is None else int(seed)
        if self._episode_metrics is None or self._rollout_metrics is None:
            raise RuntimeError("learning-check callback尚未完成training start初始化。")
        self._episode_metrics.add(info)
        self._rollout_metrics.add(info)
        for key in LEARNING_CHECK_HARD_KEYS:
            self.hard_diagnostics[key] += max(0, int(info.get(key, 0)))
        dones = np.asarray(self.locals.get("dones", [False]), dtype=bool).reshape(-1)
        if bool(dones[0]):
            self._finish_episode(completed=True)
        return True

    def _on_rollout_end(self) -> None:
        if self._rollout_metrics is None or self._rollout_metrics.steps <= 0:
            return
        self._pending_rollout = self._rollout_metrics
        assert self.layout is not None
        self._rollout_metrics = _MetricAccumulator(self.layout)

    def _logger_float(self, key: str) -> float:
        values = getattr(getattr(self, "model", None), "logger", None)
        values = {} if values is None else getattr(values, "name_to_value", {})
        value = values.get(key, float("nan"))
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def _append_action_rows(
        self,
        *,
        update: int,
        timesteps: int,
        metrics: _MetricAccumulator,
    ) -> None:
        cache_types = self.layout["cache_node_types"]
        for index, node_id in enumerate(self.layout["cache_node_ids"]):
            self.action_rows.append(
                {
                    "update": int(update),
                    "timesteps": int(timesteps),
                    "scope": "cache",
                    "action_index": int(self.layout["cache_trigger_indices"][index]),
                    "node_id": node_id,
                    "node_type": cache_types[index],
                    "requested_trigger_count": int(metrics.cache_requested[index]),
                    "requested_trigger_rate": float(
                        metrics.cache_requested[index] / float(max(1, metrics.steps))
                    ),
                    "effective_update_count": int(metrics.cache_effective_update[index]),
                    "effective_update_rate": float(
                        metrics.cache_effective_update[index] / float(max(1, metrics.steps))
                    ),
                    "noop_trigger_count": int(metrics.cache_noop[index]),
                    "services_changed_count": int(metrics.cache_services_changed[index]),
                    "actual_movement_count": 0,
                    "actual_movement_rate": 0.0,
                }
            )
        for index, node_id in enumerate(self.layout["uav_ids"]):
            self.action_rows.append(
                {
                    "update": int(update),
                    "timesteps": int(timesteps),
                    "scope": "uav",
                    "action_index": int(
                        self.layout["uav_position_trigger_indices"][index]
                    ),
                    "node_id": node_id,
                    "node_type": "uav",
                    "requested_trigger_count": int(metrics.uav_requested[index]),
                    "requested_trigger_rate": float(
                        metrics.uav_requested[index] / float(max(1, metrics.steps))
                    ),
                    "effective_update_count": 0,
                    "effective_update_rate": 0.0,
                    "noop_trigger_count": int(metrics.uav_noop[index]),
                    "services_changed_count": 0,
                    "actual_movement_count": int(metrics.uav_movement[index]),
                    "actual_movement_rate": float(
                        metrics.uav_movement[index] / float(max(1, metrics.steps))
                    ),
                }
            )

    def _record_update(self) -> None:
        metrics = self._pending_rollout
        if metrics is None:
            return
        if self.layout is None:
            raise RuntimeError("learning-check缺少运行时action_layout。")
        if self._logged_updates >= self.expected_updates:
            self._pending_rollout = None
            return
        self._logged_updates += 1
        update = self._logged_updates
        timesteps = int(getattr(self.model, "num_timesteps", 0))
        row = metrics.row(mode="update")
        row.update(
            {
                "update": int(update),
                "timesteps": timesteps,
                "completed_episodes": int(self._completed_episodes),
                "approx_kl": self._logger_float("train/approx_kl"),
                "clip_fraction": self._logger_float("train/clip_fraction"),
                "entropy_loss": self._logger_float("train/entropy_loss"),
                "policy_gradient_loss": self._logger_float(
                    "train/policy_gradient_loss"
                ),
                "value_loss": self._logger_float("train/value_loss"),
                "explained_variance": self._logger_float("train/explained_variance"),
            }
        )
        self.update_rows.append(row)
        self._append_action_rows(update=update, timesteps=timesteps, metrics=metrics)
        cache_rate = ", ".join(
            f"{node_id}={metrics.cache_effective_update[index] / max(1, metrics.steps):.3f}"
            for index, node_id in enumerate(self.layout["cache_node_ids"])
        )
        uav_rate = ", ".join(
            f"{node_id}={metrics.uav_movement[index] / max(1, metrics.steps):.3f}"
            for index, node_id in enumerate(self.layout["uav_ids"])
        )
        cache_trigger_rate = ", ".join(
            f"bit{self.layout['cache_trigger_indices'][index]}="
            f"{metrics.cache_requested[index] / max(1, metrics.steps):.3f}"
            for index in range(len(self.layout["cache_node_ids"]))
        )
        uav_trigger_rate = ", ".join(
            f"bit{self.layout['uav_position_trigger_indices'][index]}="
            f"{metrics.uav_requested[index] / max(1, metrics.steps):.3f}"
            for index in range(len(self.layout["uav_ids"]))
        )
        print(f"Update {update}/{self.expected_updates}")
        print(
            f"timesteps={timesteps} completed episodes={self._completed_episodes} "
            f"recent mean raw G={row['mean_raw_G']:.6f} recent Q={row['mean_Q']:.6f} "
            f"cache cost={row['cache_cost']:.6f} UAV cost={row['uav_cost']:.6f}"
        )
        print(f"cache trigger rates: {cache_trigger_rate}")
        print(f"cache actual update rates: {cache_rate}")
        print(f"UAV trigger rates: {uav_trigger_rate}")
        print(f"UAV actual movement rates: {uav_rate}")
        print(
            "approx_kl={:.6g} clip_fraction={:.6g} entropy_loss={:.6g} "
            "policy_gradient_loss={:.6g} value_loss={:.6g} explained_variance={:.6g}".format(
                row["approx_kl"],
                row["clip_fraction"],
                row["entropy_loss"],
                row["policy_gradient_loss"],
                row["value_loss"],
                row["explained_variance"],
            )
        )
        self._pending_rollout = None

    def _on_rollout_start(self) -> None:
        # 第一个 rollout 没有 PPO train 日志；之后每次这里对应刚完成的一个
        # 256-step rollout 和一个 PPO update。最后一个 update 在 training_end补齐。
        self._record_update()

    def _on_training_end(self) -> None:
        self._record_update()
        # 5120 / 100 不是整数；最后的未终止片段也保留在 episodes.csv，
        # 但不计入 completed episodes，避免把 partial episode 当作完整样本。
        self._finish_episode(completed=False)

    def write_outputs(self) -> None:
        _write_csv(self.output_dir / "training_updates.csv", self.update_rows)
        _write_csv(self.output_dir / "training_episodes.csv", self.episode_rows)
        _write_csv(self.output_dir / "action_trigger_rates.csv", self.action_rows)

    @property
    def logged_updates(self) -> int:
        return int(self._logged_updates)

    @property
    def completed_episodes(self) -> int:
        return int(self._completed_episodes)


def _select_best_validation_record(
    records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    valid = []
    for record in records:
        try:
            score = float(record["mean_original_validation_system_reward"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(score):
            valid.append((score, int(record.get("timesteps", 0)), record))
    if not valid:
        return None
    valid.sort(key=lambda item: (-item[0], item[1]))
    return valid[0][2]


class FormalTrainingCallback(LearningCheckCallback):
    """在固定 eval 频率保存 validation 最优的 model/VecNormalize 对。"""

    def __init__(
        self,
        *,
        output_dir: Path,
        config: Mapping[str, Any],
        action_layout: Mapping[str, Any] | None,
        eval_freq: int,
        expected_updates: int,
    ) -> None:
        super().__init__(
            output_dir=output_dir,
            action_layout=action_layout,
            expected_updates=expected_updates,
        )
        self.config = config
        self.eval_freq = int(eval_freq)
        if self.eval_freq <= 0:
            raise ValueError("eval_freq must be positive")
        self.validation_rows: list[dict[str, Any]] = []
        self.validation_history: list[dict[str, Any]] = []
        self.validation_diagnostics = {key: 0 for key in LEARNING_CHECK_HARD_KEYS}
        self.evaluated_timesteps: set[int] = set()
        self.best_score = -math.inf
        self.best_timesteps: int | None = None
        self.best_update: int | None = None
        self.best_model_path: Path | None = None
        self.best_vecnormalize_path: Path | None = None

    def _evaluate_checkpoint(self, timesteps: int) -> None:
        step = int(timesteps)
        if step <= 0 or step in self.evaluated_timesteps:
            return
        if self.layout is None:
            raise RuntimeError("formal callback has no action layout")
        self.evaluated_timesteps.add(step)
        rows, diagnostics = _run_deterministic_validation(
            self.model,
            self.config,
            training_vector=self.training_env,
            action_layout=self.layout,
        )
        aggregate = _validation_aggregate(rows)
        score = float(aggregate.get("mean", {}).get("mean_raw_G", -math.inf))
        update = int(step // max(1, int(_section(self.config, "ppo").get("n_steps", 256))))
        for row in rows:
            enriched = dict(row)
            enriched.update(
                {
                    "timesteps": step,
                    "update": update,
                    "mean_original_validation_system_reward": score,
                    "best_criterion": "mean_original_validation_system_reward",
                }
            )
            self.validation_rows.append(enriched)
        self.validation_history.append(
            {
                "timesteps": step,
                "update": update,
                "mean_original_validation_system_reward": score,
                "validation_std_raw_G": float(
                    aggregate.get("std", {}).get("mean_raw_G", 0.0)
                ),
                "validation_seed_count": len(rows),
            }
        )
        for key in LEARNING_CHECK_HARD_KEYS:
            self.validation_diagnostics[key] += max(0, int(diagnostics.get(key, 0)))
        if score > self.best_score:
            self.best_score = score
            self.best_timesteps = step
            self.best_update = update
            self.best_model_path = self.output_dir / "best_model"
            self.best_vecnormalize_path = self.output_dir / "best_vecnormalize.pkl"
            self.model.save(self.best_model_path)
            self.training_env.save(self.best_vecnormalize_path)

    def _on_rollout_start(self) -> None:
        super()._on_rollout_start()
        step = int(getattr(self.model, "num_timesteps", 0))
        if step > 0 and step % self.eval_freq == 0:
            self._evaluate_checkpoint(step)

    def _on_training_end(self) -> None:
        super()._on_training_end()
        step = int(getattr(self.model, "num_timesteps", 0))
        if step > 0:
            self._evaluate_checkpoint(step)

    def write_outputs(self) -> None:
        super().write_outputs()
        _write_csv(self.output_dir / "validation_results.csv", self.validation_rows)
        _write_csv(self.output_dir / "validation_history.csv", self.validation_history)


def _learning_check_output_dir(
    config: Mapping[str, Any], output_dir: str | Path | None = None
) -> Path:
    if output_dir is not None:
        target = Path(output_dir).expanduser()
        target.mkdir(parents=True, exist_ok=False)
        return target
    configured_root = _section(config, "output").get("root", "outputs")
    root = Path(str(configured_root)).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    base = root / "ppo_learning_check"
    base.mkdir(parents=True, exist_ok=True)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    target = base / run_id
    suffix = 1
    while target.exists():
        suffix += 1
        target = base / f"{run_id}_{suffix:02d}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def _formal_output_dir(
    config: Mapping[str, Any], output_dir: str | Path | None = None
) -> Path:
    if output_dir is not None:
        target = Path(output_dir).expanduser()
        target.mkdir(parents=True, exist_ok=False)
        return target
    configured_root = _section(config, "output").get("root", "outputs")
    root = Path(str(configured_root)).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    base = root / "ppo_training"
    base.mkdir(parents=True, exist_ok=True)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    target = base / run_id
    suffix = 1
    while target.exists():
        suffix += 1
        target = base / f"{run_id}_{suffix:02d}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def _formal_training_trace_seeds(
    config: Mapping[str, Any], total_timesteps: int | None = None
) -> dict[str, Any]:
    ppo = _section(config, "ppo")
    simulation = _section(config, "simulation")
    n_steps = int(ppo.get("n_steps", 256))
    requested_steps = int(
        ppo.get("total_timesteps", 25600)
        if total_timesteps is None
        else total_timesteps
    )
    horizon = int(simulation.get("slot_count", 100))
    if n_steps <= 0 or requested_steps <= 0 or horizon <= 0:
        raise ValueError("formal PPO steps and episode horizon must be positive")
    effective_steps = int(math.ceil(requested_steps / n_steps) * n_steps)
    formal_episodes = int(effective_steps // horizon)
    trace_count = int(math.floor(effective_steps / horizon) + 1)
    base = int(ppo.get("train_seed_base", ppo.get("seed", 0)))
    return {
        "requested_env_steps": requested_steps,
        "effective_env_steps": effective_steps,
        "episode_horizon": horizon,
        "n_steps": n_steps,
        "formal_training_episodes": formal_episodes,
        "training_reset_reserve": trace_count - formal_episodes,
        "training_trace_seeds": [base + offset for offset in range(trace_count)],
    }


def preflight_training_traces(
    config: Mapping[str, Any],
    *,
    total_timesteps: int | None = None,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """只检查离线 trace 的覆盖、schema、slot 数和固定研究车辆角色。"""

    mobility = _section(config, "mobility")
    if str(mobility.get("source", "sumo")).strip().lower() != "trace":
        raise RuntimeError("formal PPO training requires mobility.source=trace")
    try:
        from .mobility import TraceMobilityProvider, trace_path
    except ImportError:  # pragma: no cover
        from mobility import TraceMobilityProvider, trace_path

    roles = _section(config, "vehicle_roles")
    service_ids = tuple(str(item) for item in roles.get("service_vehicle_ids", ()))
    task_ids = tuple(str(item) for item in roles.get("task_vehicle_ids", ()))
    fixed_ids = service_ids + task_ids
    if not fixed_ids:
        raise RuntimeError("fixed_episode preflight requires explicit research vehicle IDs")
    trace_plan = _formal_training_trace_seeds(config, total_timesteps)
    validation_seeds = _formal_validation_seeds(config)
    test_seeds = _formal_test_seeds(config)
    categories = {
        "training": list(trace_plan["training_trace_seeds"]),
        "validation": validation_seeds,
        "test": test_seeds,
    }
    horizon = int(trace_plan["episode_horizon"])
    coverage: dict[str, Any] = {}
    missing: dict[str, list[int]] = {key: [] for key in categories}
    invalid: list[dict[str, Any]] = []
    for category, seeds in categories.items():
        for seed in seeds:
            path = trace_path(config, seed=int(seed))
            if not path.is_file():
                missing[category].append(int(seed))
                continue
            provider = None
            try:
                provider = TraceMobilityProvider(config, seed=int(seed))
                slot_count = int(provider._times.size)
                if slot_count != horizon:
                    raise ValueError(
                        f"slot_count={slot_count}, expected {horizon}"
                    )
                provider.validate_fixed_episode_roles(fixed_ids, slot_count=horizon)
            except Exception as error:
                invalid.append(
                    {
                        "category": category,
                        "seed": int(seed),
                        "path": str(path),
                        "error": str(error),
                    }
                )
            finally:
                if provider is not None:
                    provider.close()
        coverage[category] = {
            "count": len(seeds),
            "first_seed": None if not seeds else int(seeds[0]),
            "last_seed": None if not seeds else int(seeds[-1]),
            "missing": list(missing[category]),
        }
    missing_flat = {
        key: values for key, values in missing.items() if values
    }
    if missing_flat:
        print(f"Trace preflight missing seeds: {missing_flat}")
    if invalid:
        print(f"Trace preflight invalid traces: {invalid}")
    result = {
        **trace_plan,
        "formal_training_traces": len(categories["training"]),
        "validation_traces": len(validation_seeds),
        "test_traces": len(test_seeds),
        "total_required": sum(len(values) for values in categories.values()),
        "coverage": coverage,
        "missing": missing,
        "missing_seeds": sorted(
            seed for values in missing.values() for seed in values
        ),
        "invalid": invalid,
        "fixed_role_ids": fixed_ids,
        "passed": not missing_flat and not invalid,
    }
    if raise_on_error and not result["passed"]:
        raise RuntimeError(
            "formal training trace preflight failed: "
            f"missing={missing_flat}, invalid={invalid}"
        )
    return result


def _formal_startup_checks(
    config: Mapping[str, Any], *, first_training_seed: int
) -> dict[str, Any]:
    environment = build_ppo_environment(
        config, seed_mode="fixed", fixed_seed=int(first_training_seed)
    )
    try:
        observation, reset_info = environment.reset(seed=int(first_training_seed))
        expected_action_length = int(environment.action_space.shape[0])
        layout = _layout_copy(reset_info["action_layout"])
        if int(layout["action_length"]) != expected_action_length:
            raise RuntimeError("formal startup action layout dimension mismatch")
        if observation.shape != environment.observation_space.shape:
            raise RuntimeError("formal startup observation dimension mismatch")
        check_status = sb3_check_env(environment)
        return {
            "check_env": check_status,
            "action_space": str(environment.action_space),
            "action_layout": layout,
            "observation_dimension": int(observation.shape[0]),
            "observation_finite": bool(np.all(np.isfinite(observation))),
            "task_vehicle_ids": tuple(environment.environment._task_vehicle_ids),
            "service_vehicle_ids": tuple(environment.environment._service_vehicle_ids),
        }
    finally:
        environment.close()


def _learning_check_validation_seeds(config: Mapping[str, Any]) -> list[int]:
    values = _section(config, "ppo").get("validation_seeds")
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("learning-check需要config.yaml中的非空validation_seeds。")
    seeds = [int(item) for item in values]
    if len(set(seeds)) != len(seeds):
        raise ValueError("validation_seeds不能重复。")
    return seeds


def _configured_seed_list(config: Mapping[str, Any], name: str) -> list[int]:
    values = _section(config, "ppo").get(name)
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"ppo.{name} must be a non-empty list")
    seeds = [int(item) for item in values]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"ppo.{name} cannot contain duplicates")
    return seeds


def _formal_validation_seeds(config: Mapping[str, Any]) -> list[int]:
    return _configured_seed_list(config, "validation_seeds")


def _formal_test_seeds(config: Mapping[str, Any]) -> list[int]:
    return _configured_seed_list(config, "test_seeds")


def _build_validation_vector(
    config: Mapping[str, Any],
    *,
    seed: int,
    training_vector: Any,
) -> Any:
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("当前 conda 环境缺少 stable-baselines3。") from error

    def factory() -> Any:
        # validation 只接受 fixed validation seed；不复用 training sequential
        # resolver，也不触碰 test_seeds。
        return build_ppo_environment(
            config,
            seed_mode="fixed",
            fixed_seed=int(seed),
        )

    vector = VecMonitor(DummyVecEnv([factory]))
    ppo = _section(config, "ppo")
    evaluation = VecNormalize(
        vector,
        norm_obs=bool(ppo.get("norm_obs", True)),
        norm_reward=False,
        clip_obs=float(ppo.get("clip_obs", 10.0)),
    )
    if bool(ppo.get("norm_obs", True)) and getattr(training_vector, "obs_rms", None) is not None:
        evaluation.obs_rms = copy.deepcopy(training_vector.obs_rms)
    evaluation.training = False
    evaluation.norm_reward = False
    return evaluation


def _run_deterministic_validation(
    model: Any,
    config: Mapping[str, Any],
    *,
    training_vector: Any,
    action_layout: Mapping[str, Any],
    seeds: Sequence[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    results: list[dict[str, Any]] = []
    diagnostics = {key: 0 for key in LEARNING_CHECK_HARD_KEYS}
    evaluation_seeds = (
        _formal_validation_seeds(config)
        if seeds is None
        else [int(item) for item in seeds]
    )
    for seed in evaluation_seeds:
        vector = _build_validation_vector(
            config,
            seed=int(seed),
            training_vector=training_vector,
        )
        metrics = _MetricAccumulator(action_layout)
        terminated = False
        try:
            observation = vector.reset()
            # 工作负载长度由当前 trace 决定；10000 只是防止验证入口在
            # 意外的 done 语义错误下无限循环，不改变 episode horizon。
            for _step in range(10000):
                action, _state = model.predict(observation, deterministic=True)
                observation, _reward, dones, infos = vector.step(action)
                info = infos[0]
                if not isinstance(info, Mapping):
                    raise RuntimeError("validation step info不是mapping。")
                metrics.add(info)
                for key in LEARNING_CHECK_HARD_KEYS:
                    diagnostics[key] += max(0, int(info.get(key, 0)))
                if bool(np.asarray(dones, dtype=bool).reshape(-1)[0]):
                    terminated = True
                    break
            if not terminated:
                raise RuntimeError(f"validation seed {seed} 未在episode horizon内结束。")
        finally:
            vector.close()
        row = metrics.validation_row(int(seed))
        results.append(row)
    return results, diagnostics


def _validation_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"mean": {}, "std": {}}
    excluded = {"seed", "seed_mode"}
    fields: list[str] = []
    for key in rows[0]:
        if key in excluded:
            continue
        values = []
        valid = True
        for row in rows:
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                valid = False
                break
            if not math.isfinite(value):
                valid = False
                break
            values.append(value)
        if valid and values:
            fields.append(key)
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for key in fields:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        mean[key] = float(np.mean(values))
        std[key] = float(np.std(values))
    return {"mean": mean, "std": std}


def _mean_update_field(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return None if not values else float(np.mean(np.asarray(values, dtype=float)))


def _early_late_summary(
    updates: Sequence[Mapping[str, Any]], action_layout: Mapping[str, Any]
) -> dict[str, Any]:
    early = list(updates[:10])
    late = list(updates[10:20])
    fields = [
        "mean_raw_G",
        "mean_Q",
        "cache_cost",
        "uav_cost",
        "mean_delay",
        "mean_vehicle_energy",
        "edge_cache_availability_rate",
        "edge_execution_rate",
    ]
    fields.extend(f"{label}_rate" for _key, label in LEARNING_CHECK_EXECUTION_KEYS)
    for index in range(len(action_layout["cache_node_ids"])):
        fields.extend(
            [
                f"cache_node_{index}_requested_trigger_rate",
                f"cache_node_{index}_effective_update_rate",
            ]
        )
    for index in range(len(action_layout["uav_ids"])):
        fields.extend(
            [
                f"uav_node_{index}_requested_move_trigger_rate",
                f"uav_node_{index}_actual_movement_rate",
            ]
        )
    early_values = {field: _mean_update_field(early, field) for field in fields}
    late_values = {field: _mean_update_field(late, field) for field in fields}
    delta: dict[str, float | None] = {}
    for field in fields:
        before, after = early_values[field], late_values[field]
        delta[field] = None if before is None or after is None else float(after - before)
    return {
        "early_updates": "1-10",
        "late_updates": "11-20",
        "early": early_values,
        "late": late_values,
        "late_minus_early": delta,
        "available_update_count": len(updates),
    }


def _write_learning_check_figures(
    output_dir: Path,
    episodes: Sequence[Mapping[str, Any]],
    updates: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    action_layout: Mapping[str, Any],
) -> dict[str, Any]:
    """生成正式主图和诊断图，并按结果目录分层保存。"""

    figure_dir = output_dir / "figures"
    diagnostics_dir = figure_dir / "diagnostics"
    figure_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - 训练环境通常已安装matplotlib
        return {"generated": False, "reason": "matplotlib_missing", "files": []}

    files: list[str] = []
    main_files: list[str] = []
    diagnostic_files: list[str] = []
    primary_error: str | None = None
    try:
        try:
            from .plot_results import plot_learning_check_run
        except ImportError:  # pragma: no cover - direct script import
            from plot_results import plot_learning_check_run
        primary = plot_learning_check_run(
            output_dir / "training_episodes.csv",
            output_dir / "validation_results.csv",
            figure_dir,
        )
        for name in ("reward_convergence", "final_execution_rate"):
            for path in primary[name]:
                relative = str(Path(path).relative_to(output_dir))
                files.append(relative)
                main_files.append(relative)
    except Exception as error:  # pragma: no cover - 主图失败不应掩盖训练结果
        primary_error = repr(error)
        print(f"Skipped formal learning-check figures: {error}")
    try:
        completed = [row for row in episodes if bool(row.get("completed", True))]
        figure = plt.figure(figsize=(7.0, 4.0))
        axis = figure.add_subplot(1, 1, 1)
        axis.plot(
            [int(row["episode"]) for row in completed],
            [float(row["mean_raw_G"]) for row in completed],
            marker="o",
            linewidth=1.0,
        )
        axis.set_xlabel("Episode")
        axis.set_ylabel("mean raw G")
        axis.set_title("Learning check: raw mean G")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        path = diagnostics_dir / "raw_mean_G_episode.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        relative = str(path.relative_to(output_dir))
        files.append(relative)
        diagnostic_files.append(relative)

        figure = plt.figure(figsize=(7.0, 4.0))
        axis = figure.add_subplot(1, 1, 1)
        x = [int(row["update"]) for row in updates]
        for _key, label in LEARNING_CHECK_EXECUTION_KEYS:
            axis.plot(x, [float(row[f"{label}_rate"]) for row in updates], label=label)
        axis.plot(x, [float(row["failed_count"]) / max(1, int(row["arrival_count"])) for row in updates], label="Failed")
        axis.set_xlabel("PPO update")
        axis.set_ylabel("execution rate")
        axis.set_title("Learning check: execution rates")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, fontsize=8)
        figure.tight_layout()
        path = diagnostics_dir / "execution_rates.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        relative = str(path.relative_to(output_dir))
        files.append(relative)
        diagnostic_files.append(relative)

        figure = plt.figure(figsize=(9.0, 5.0))
        axis = figure.add_subplot(1, 1, 1)
        update_values = sorted({int(row["update"]) for row in action_rows})
        for scope in ("cache", "uav"):
            scoped = [row for row in action_rows if str(row["scope"]) == scope]
            for node_id in sorted({str(row["node_id"]) for row in scoped}):
                node_rows = [row for row in scoped if str(row["node_id"]) == node_id]
                node_rows.sort(key=lambda row: int(row["update"]))
                label = node_id
                axis.plot(
                    [int(row["update"]) for row in node_rows],
                    [float(row["requested_trigger_rate"]) for row in node_rows],
                    linestyle="--",
                    linewidth=0.9,
                    label=f"req {label}",
                )
                actual_field = (
                    "effective_update_rate" if scope == "cache" else "actual_movement_rate"
                )
                axis.plot(
                    [int(row["update"]) for row in node_rows],
                    [float(row[actual_field]) for row in node_rows],
                    linewidth=1.1,
                    label=f"actual {label}",
                )
        del update_values  # layout仅用于保证图线来自显式 action_layout 映射
        axis.set_xlabel("PPO update")
        axis.set_ylabel("rate")
        axis.set_title("Learning check: trigger vs actual update/movement")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=7)
        figure.tight_layout()
        path = diagnostics_dir / "trigger_actual_rates.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        relative = str(path.relative_to(output_dir))
        files.append(relative)
        diagnostic_files.append(relative)
    except Exception as error:  # pragma: no cover - 不影响核心数值审计
        try:
            plt.close("all")
        except Exception:
            pass
        print(f"Skipped diagnostic learning-check figures: {error}")
        if primary_error is None:
            primary_error = repr(error)
    print("Generated main figures:")
    for name in main_files:
        print(f"- {output_dir / name}")
    if diagnostic_files:
        print("Generated diagnostic figures:")
        for name in diagnostic_files:
            print(f"- {output_dir / name}")
    return {
        "generated": primary_error is None,
        "reason": primary_error,
        "files": files,
        "main_files": main_files,
        "diagnostic_files": diagnostic_files,
    }


def run_learning_check(
    config: Mapping[str, Any] | str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """运行时入口：5120 steps + validation；正式长训练由用户显式启动。"""

    prepared = prepare_ppo_config(config)
    ppo = _section(prepared, "ppo")
    n_steps = int(ppo.get("n_steps", 256))
    if n_steps != 256:
        raise ValueError(f"learning-check要求n_steps=256，当前为{n_steps}。")
    if LEARNING_CHECK_TIMESTEPS % n_steps != 0:
        raise ValueError("5120必须能被n_steps=256整除，才能得到20个PPO updates。")
    expected_updates = LEARNING_CHECK_TIMESTEPS // n_steps
    if expected_updates != LEARNING_CHECK_EXPECTED_UPDATES:
        raise RuntimeError("learning-check预期update数量计算错误。")
    validation_seeds = _learning_check_validation_seeds(prepared)
    target = _learning_check_output_dir(prepared, output_dir)

    vector = None
    model = None
    callback = None
    try:
        vector = build_vecnormalize_environment(
            prepared,
            episode_seed_base=int(ppo.get("train_seed_base", ppo.get("seed", 0))),
        )
        model = build_sb3_ppo(vector, prepared)
        callback = LearningCheckCallback(
            output_dir=target,
            action_layout=None,
            expected_updates=expected_updates,
        )
        # 这是本入口唯一的 learn 调用；Codex 的本轮 smoke 不调用本函数。
        model.learn(total_timesteps=LEARNING_CHECK_TIMESTEPS, callback=callback)
        if callback.layout is None:
            raise RuntimeError("learning-check训练结束后未获得environment.action_layout。")
        action_layout = callback.layout
        snapshot_config = copy.deepcopy(prepared)
        snapshot_ppo = dict(_section(snapshot_config, "ppo"))
        # 记录配置但不读取/枚举正式 test seed；运行路径只有 training 与 validation。
        snapshot_ppo["test_seeds"] = "NOT_ACCESSED"
        snapshot_config["ppo"] = snapshot_ppo
        _write_json(
            target / "config_snapshot.json",
            {
                "config": snapshot_config,
                "action_layout": action_layout,
                "learning_check": {
                    "total_timesteps": LEARNING_CHECK_TIMESTEPS,
                    "n_steps": n_steps,
                    "expected_updates": expected_updates,
                    "training_seed_mode": "sequential",
                    "training_seed_base": int(
                        ppo.get("train_seed_base", ppo.get("seed", 0))
                    ),
                    "validation_seed_mode": "fixed",
                    "validation_seeds": validation_seeds,
                    "test_seeds_accessed": False,
                },
            },
        )
        callback.write_outputs()
        model.save(target / "model")
        vector.save(target / "vecnormalize.pkl")

        validation_rows, validation_diagnostics = _run_deterministic_validation(
            model,
            prepared,
            training_vector=vector,
            action_layout=action_layout,
        )
        _write_csv(target / "validation_results.csv", validation_rows)
        figures = _write_learning_check_figures(
            target,
            callback.episode_rows,
            callback.update_rows,
            callback.action_rows,
            action_layout,
        )
        hard_diagnostics = {
            key: int(callback.hard_diagnostics[key] + validation_diagnostics[key])
            for key in LEARNING_CHECK_HARD_KEYS
        }
        delta = _early_late_summary(callback.update_rows, action_layout)[
            "late_minus_early"
        ]
        delta_g = delta.get("mean_raw_G")
        delta_q = delta.get("mean_Q")
        delta_cache_cost = delta.get("cache_cost")
        delta_uav_cost = delta.get("uav_cost")
        summary = {
            "learning_check_status": "PASS"
            if callback.logged_updates == expected_updates
            and all(value == 0 for value in hard_diagnostics.values())
            else "FAIL",
            "total_timesteps": LEARNING_CHECK_TIMESTEPS,
            "n_steps": n_steps,
            "expected_updates": expected_updates,
            "logged_updates": callback.logged_updates,
            "model_learn_calls": 1,
            "formal_training_restarted": False,
            "action_layout": action_layout,
            "observation_dimension": int(vector.observation_space.shape[0]),
            "training_seed_mode": "sequential",
            "training_seed_base": int(
                ppo.get("train_seed_base", ppo.get("seed", 0))
            ),
            "validation_seed_mode": "fixed",
            "validation_seeds": validation_seeds,
            "test_seeds_accessed": False,
            "completed_training_episodes": callback.completed_episodes,
            "training_episode_rows": len(callback.episode_rows),
            "hard_constraints": hard_diagnostics,
            "hard_constraints_pass": all(
                value == 0 for value in hard_diagnostics.values()
            ),
            "early_late": _early_late_summary(callback.update_rows, action_layout),
            "diagnostic_deltas": {
                "Delta_Q": delta_q,
                "Delta_cache_cost": delta_cache_cost,
                "Delta_UAV_cost": delta_uav_cost,
                "Delta_G": delta_g,
            },
            "learning_effectiveness_interpretation": (
                "Delta_G > 0"
                if delta_g is not None and delta_g > 0.0
                else "Delta_G <= 0; inspect trigger policy and cost trade-off"
            ),
            "validation": {
                "results": validation_rows,
                "aggregate": _validation_aggregate(validation_rows),
            },
            "figures": figures,
            "artifacts": {
                "config_snapshot": "config_snapshot.json",
                "training_updates": "training_updates.csv",
                "training_episodes": "training_episodes.csv",
                "action_trigger_rates": "action_trigger_rates.csv",
                "validation_results": "validation_results.csv",
                "learning_summary": "learning_summary.json",
                "final_summary": "final_summary.json",
                "model": "model.zip",
                "vecnormalize": "vecnormalize.pkl",
            },
        }
        _write_json(target / "learning_summary.json", summary)
        # Proposed 结果目录的通用最终汇总名称；保留 learning_summary.json 兼容
        # 既有 learning-check 后处理入口。
        _write_json(target / "final_summary.json", summary)
        return {"output_dir": str(target), **summary}
    finally:
        if vector is not None:
            vector.close()


def _build_loaded_vecnormalize(
    config: Mapping[str, Any], *, seed: int, path: Path
) -> Any:
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("stable-baselines3 is required for formal evaluation") from error

    def factory() -> Any:
        return build_ppo_environment(
            config,
            seed_mode="fixed",
            fixed_seed=int(seed),
        )

    vector = VecMonitor(DummyVecEnv([factory]))
    loaded = VecNormalize.load(str(path), vector)
    loaded.training = False
    loaded.norm_reward = False
    return loaded


def _formal_plot_outputs(
    output_dir: Path,
    training_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        from .plot_results import plot_final_execution_rate, plot_training_reward_curve
    except ImportError:  # pragma: no cover
        from plot_results import plot_final_execution_rate, plot_training_reward_curve
    complete_training = [
        row for row in training_rows if bool(row.get("completed", True))
    ]
    if not complete_training:
        raise RuntimeError("formal training produced no completed episode rows")
    episodes = [int(row["episode"]) for row in complete_training]
    rewards = [float(row["sum_raw_G"]) for row in complete_training]
    steps = [int(row["steps"]) for row in complete_training]
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    reward_paths = plot_training_reward_curve(
        episodes,
        rewards,
        steps,
        figure_dir,
        moving_average_window=10,
        file_stem="reward_convergence",
    )
    counts = {
        "local": sum(int(row.get("Local_count", 0)) for row in test_rows),
        "service_vehicle": sum(int(row.get("SV_count", 0)) for row in test_rows),
        "rsu": sum(int(row.get("RSU_count", 0)) for row in test_rows),
        "uav": sum(int(row.get("UAV_count", 0)) for row in test_rows),
        "cloud": sum(int(row.get("Cloud_count", 0)) for row in test_rows),
        "failed": sum(int(row.get("failed_count", 0)) for row in test_rows),
    }
    execution_paths = plot_final_execution_rate(
        counts["local"],
        counts["service_vehicle"],
        counts["rsu"],
        counts["uav"],
        counts["cloud"],
        counts["failed"],
        figure_dir,
        file_stem="final_execution_rate",
    )
    return {
        "reward_convergence": {
            "png": str(reward_paths[0]),
            "pdf": str(reward_paths[1]),
            "source": "training raw G per slot",
        },
        "final_execution_rate": {
            "png": str(execution_paths[0]),
            "pdf": str(execution_paths[1]),
            "source": "best-model final test",
        },
        "execution_counts": counts,
    }


def run_training(
    config: Mapping[str, Any] | str | Path | None = None,
    *,
    total_timesteps: int | None = None,
    output_dir: str | Path | None = None,
) -> Any:
    """正式 PPO 入口：trace 预检、validation 选优、best checkpoint 后 test。"""

    prepared = prepare_ppo_config(config)
    ppo = _section(prepared, "ppo")
    requested_steps = int(
        ppo.get("total_timesteps", 25600)
        if total_timesteps is None
        else total_timesteps
    )
    trace_preflight = preflight_training_traces(
        prepared, total_timesteps=requested_steps
    )
    startup = _formal_startup_checks(
        prepared,
        first_training_seed=int(trace_preflight["training_trace_seeds"][0]),
    )
    n_steps = int(ppo.get("n_steps", 256))
    eval_freq = int(ppo.get("eval_freq", 5120))
    if eval_freq <= 0 or eval_freq % n_steps != 0:
        raise ValueError("formal eval_freq must be a positive multiple of n_steps")
    expected_updates = int(trace_preflight["effective_env_steps"] // n_steps)
    target = _formal_output_dir(prepared, output_dir)
    snapshot = copy.deepcopy(prepared)
    _write_json(
        target / "config_snapshot.json",
        {
            "config": snapshot,
            "formal_training": {
                "requested_timesteps": requested_steps,
                "effective_env_steps": trace_preflight["effective_env_steps"],
                "n_steps": n_steps,
                "eval_freq": eval_freq,
                "training_seed_mode": "sequential",
                "validation_seed_mode": "fixed",
                "test_after_best_only": True,
                "preflight": trace_preflight,
                "startup": startup,
            },
        },
    )

    vector = None
    model = None
    callback = None
    best_vector = None
    try:
        vector = build_vecnormalize_environment(
            prepared,
            episode_seed_base=int(ppo.get("train_seed_base", ppo.get("seed", 0))),
        )
        model = build_sb3_ppo(vector, prepared)
        callback = FormalTrainingCallback(
            output_dir=target,
            config=prepared,
            action_layout=startup["action_layout"],
            eval_freq=eval_freq,
            expected_updates=expected_updates,
        )
        model.learn(total_timesteps=requested_steps, callback=callback)
        callback.write_outputs()
        if callback.best_model_path is None or callback.best_vecnormalize_path is None:
            raise RuntimeError("formal training did not produce a best model/VecNormalize pair")

        # 只有在 best validation checkpoint 已确定后，才读取并评估 test seeds。
        test_seeds = _formal_test_seeds(prepared)
        best_vector = _build_loaded_vecnormalize(
            prepared,
            seed=int(test_seeds[0]),
            path=callback.best_vecnormalize_path,
        )
        try:
            from stable_baselines3 import PPO
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("stable-baselines3 is required for formal evaluation") from error
        best_model = PPO.load(str(callback.best_model_path), env=best_vector)
        test_rows, test_diagnostics = _run_deterministic_validation(
            best_model,
            prepared,
            training_vector=best_vector,
            action_layout=startup["action_layout"],
            seeds=test_seeds,
        )
        _write_csv(target / "test_results.csv", test_rows)
        figures = _formal_plot_outputs(target, callback.episode_rows, test_rows)
        diagnostics = {
            "training": callback.hard_diagnostics,
            "validation": callback.validation_diagnostics,
            "test": test_diagnostics,
            "preflight": trace_preflight,
        }
        diagnostics_dir = target / "figures" / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        _write_json(diagnostics_dir / "formal_diagnostics.json", diagnostics)
        test_aggregate = _validation_aggregate(test_rows)
        summary = {
            "status": "PASS",
            "requested_timesteps": requested_steps,
            "effective_env_steps": trace_preflight["effective_env_steps"],
            "n_steps": n_steps,
            "eval_freq": eval_freq,
            "expected_updates": expected_updates,
            "logged_updates": callback.logged_updates,
            "best_criterion": "mean_original_validation_system_reward",
            "best_score": callback.best_score,
            "best_timesteps": callback.best_timesteps,
            "best_update": callback.best_update,
            "best_model": str(callback.best_model_path.with_suffix(".zip")),
            "best_vecnormalize": str(callback.best_vecnormalize_path),
            "training_seed_mode": "sequential",
            "validation_seed_mode": "fixed",
            "validation_seeds": _formal_validation_seeds(prepared),
            "test_seed_mode": "fixed",
            "test_seeds": test_seeds,
            "test": {
                "results": test_rows,
                "aggregate": test_aggregate,
                "mean_raw_G": float(test_aggregate["mean"].get("mean_raw_G", 0.0)),
                "std_raw_G": float(test_aggregate["std"].get("mean_raw_G", 0.0)),
            },
            "action_layout": startup["action_layout"],
            "observation_dimension": startup["observation_dimension"],
            "hard_constraints": diagnostics,
            "figures": figures,
            "artifacts": {
                "config_snapshot": "config_snapshot.json",
                "training_episodes": "training_episodes.csv",
                "training_updates": "training_updates.csv",
                "validation_results": "validation_results.csv",
                "validation_history": "validation_history.csv",
                "test_results": "test_results.csv",
                "best_model": "best_model.zip",
                "best_vecnormalize": "best_vecnormalize.pkl",
                "figures": "figures",
            },
        }
        _write_json(target / "final_summary.json", summary)
        return {"output_dir": str(target), **summary}
    finally:
        if best_vector is not None:
            best_vector.close()
        if vector is not None:
            vector.close()


def main(argv: Sequence[str] | None = None) -> None:
    """统一 CLI 入口；无参数与 ``--train`` 都进入正式论文训练。"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-check",
        action="store_true",
        help="只解析并检查 config.yaml，不启动任何训练。",
    )
    parser.add_argument("--interface-smoke", action="store_true")
    parser.add_argument("--learning-check", action="store_true")
    parser.add_argument("--learning-check-output", default=None)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    mode_flags = {
        "--config-check": bool(args.config_check),
        "--interface-smoke": bool(args.interface_smoke),
        "--learning-check": bool(args.learning_check),
        "--train": bool(args.train),
    }
    selected_modes = [name for name, selected in mode_flags.items() if selected]
    if len(selected_modes) > 1:
        parser.error(
            "运行模式互斥，不能同时使用：" + ", ".join(selected_modes)
        )

    if args.learning_check_output is not None and not args.learning_check:
        parser.error("--learning-check-output只能与--learning-check一起使用。")
    if args.learning_check and (
        args.total_timesteps is not None or args.output_dir is not None
    ):
        parser.error("--learning-check不能与--total-timesteps或--output-dir一起使用。")
    if args.interface_smoke and (
        args.total_timesteps is not None or args.output_dir is not None
    ):
        parser.error("--interface-smoke不能与--total-timesteps或--output-dir一起使用。")
    if args.config_check and (
        args.total_timesteps is not None or args.output_dir is not None
    ):
        parser.error("--config-check不能与--total-timesteps或--output-dir一起使用。")

    if args.config_check:
        # 配置检查是唯一的显式“只检查不训练”入口；默认入口不能走这里。
        load_config()
        print(
            json.dumps(
                {"config_parse": "PASS", "training_started": False},
                ensure_ascii=False,
            )
        )
        return
    if args.learning_check:
        result = run_learning_check(output_dir=args.learning_check_output)
        print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))
        return
    if args.interface_smoke:
        print(json.dumps(_json_safe(run_interface_smoke()), ensure_ascii=False, indent=2))
        return

    # 无参数和显式 --train 共用唯一正式论文训练实现；默认读取 config.yaml
    # 中的 total_timesteps/eval_freq，并在 pipeline 内完成 preflight、validation、
    # best model + VecNormalize 保存、final test 和正式主图生成。
    run_training(total_timesteps=args.total_timesteps, output_dir=args.output_dir)


if __name__ == "__main__":
    main()


__all__ = [
    "build_ppo_environment",
    "build_sb3_ppo",
    "build_vecnormalize_environment",
    "prepare_ppo_config",
    "run_interface_smoke",
    "run_semantic_smoke",
    "run_learning_check",
    "preflight_training_traces",
    "FormalTrainingCallback",
    "run_training",
]
