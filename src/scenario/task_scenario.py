#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成真实动态服务流行度和 TV 任务环境数据。

本模块只负责：

    rho_star(t) -> Bernoulli 到达 -> Categorical 服务请求 -> D/L 采样

它不读取任务效用、缓存、PPO 或预测结果，也不把 rho_obs 当成 rho_star。
``time_s`` 使用 SUMO 的绝对仿真时间，``slot`` 使用正式 episode 内的
0-based 时隙，因此任务行可以和 trajectory.csv 的正式时刻逐一对齐。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# ============================================================================
# 本阶段集中配置区
# ============================================================================


@dataclass(frozen=True)
class TaskConfig:
    """任务环境生成参数。

    数值默认值只用于无 TOML 场景下的函数级验证；正式 entrypoint 从
    system_config.toml 读取参数，vehicle_roles.json 仅用于检查上游时间轴漂移。
    """

    num_services: int = 10
    arrival_probability: float = 0.5
    zipf_exponent: float = 0.8
    popularity_epoch_length: int = 20
    popularity_changes_per_epoch: int = 1
    popularity_history_length: int = 20
    input_relative_jitter: float = 0.20
    load_relative_jitter: float = 0.10

    # 仅当角色文件缺少元数据时使用的显式回退值。
    fallback_dt_s: float = 1.0
    fallback_warmup_time_s: float = 100.0
    fallback_episode_time_s: float = 300.0
    fallback_master_seed: int = 4401

    # 这些子 seed 由 master_seed 的 SeedSequence 派生，不作为独立随机流使用。
    rng_stream_names: tuple[str, ...] = (
        "service_catalog",
        "popularity_evolution",
        "task_arrival",
        "service_request",
        "workload_variation",
    )

    # PROVISIONAL service heterogeneity envelope. These ranges preserve the
    # existing experiment scale and are not claimed as final
    # literature-calibrated values.
    program_size_min_mb: float = 1000.0
    program_size_max_mb: float = 2000.0
    mean_input_size_min_mbit: float = 0.5
    mean_input_size_max_mbit: float = 1.5
    cycles_per_bit_min: float = 200.0
    cycles_per_bit_max: float = 1000.0

    # implementation provisional defaults：参数范围由 system_config.toml
    # 正式入口统一读取；以下值只用于无 TOML 的函数级回退。
    program_size_unit: str = "MB"
    input_size_unit: str = "bit"
    cpu_cycles_unit: str = "cycle"
    cycles_per_bit_unit: str = "cycle/bit"

    def validate(self, num_tv: int) -> None:
        if self.num_services <= 0:
            raise ValueError("F/num_services 必须大于 0。")
        if not 0.0 <= self.arrival_probability <= 1.0:
            raise ValueError("arrival_probability 必须位于 [0, 1]。")
        if self.zipf_exponent <= 0.0:
            raise ValueError("zipf_exponent 必须大于 0。")
        if self.popularity_epoch_length <= 0:
            raise ValueError("popularity_epoch_length 必须大于 0。")
        if self.popularity_changes_per_epoch < 0:
            raise ValueError("popularity_changes_per_epoch 不能为负数。")
        if self.popularity_history_length < 1:
            raise ValueError("popularity_history_length 必须 >= 1。")
        if not 0.0 <= self.input_relative_jitter < 1.0:
            raise ValueError("input_relative_jitter 必须位于 [0, 1)。")
        if not 0.0 <= self.load_relative_jitter < 1.0:
            raise ValueError("load_relative_jitter 必须位于 [0, 1)。")
        if num_tv <= 0:
            raise ValueError("task_vehicles 不能为空。")
        parameter_ranges = (
            ("program_size", self.program_size_min_mb, self.program_size_max_mb),
            (
                "mean_input_size",
                self.mean_input_size_min_mbit,
                self.mean_input_size_max_mbit,
            ),
            ("cycles_per_bit", self.cycles_per_bit_min, self.cycles_per_bit_max),
        )
        for name, lower, upper in parameter_ranges:
            if not math.isfinite(float(lower)) or not math.isfinite(float(upper)):
                raise ValueError(f"{name} 参数范围不能是 NaN/Inf。")
            if float(lower) <= 0.0:
                raise ValueError(f"{name}_min 必须大于 0。")
            if float(upper) < float(lower):
                raise ValueError(f"{name}_max 必须大于等于 {name}_min。")


DEFAULT_CONFIG = TaskConfig()

SERVICE_CATALOG_FIELDS = (
    "service_id",
    "program_size",
    "mean_input_size",
    "cycles_per_bit",
)

POPULARITY_FIELDS = (
    "time_s",
    "slot",
    "epoch",
    "service_id",
    "rank",
    "rho_star",
    "request_count",
    "rho_obs",
)

POPULARITY_HISTORY_FIELDS = (
    "history_index",
    "formal_slot",
    "time_s",
    "epoch",
    "service_id",
    "rank",
    "rho_star",
    "request_count",
    "rho_obs",
)

TASK_FIELDS = (
    "time_s",
    "slot",
    "episode_time_s",
    "vehicle_id",
    "arrival",
    "service_id",
    "input_size",
    "cpu_cycles",
)

TASK_TYPE_DEFINITION = "service_id"
SERVICE_GENERATION_MODE = "stratified_uniform_per_dimension"
SERVICE_PARAMETER_ASSIGNMENT = "independent_permutation_across_dimensions"


# ============================================================================
# 输入文件和时间轴
# ============================================================================


def find_project_root() -> Path:
    candidates = (
        Path(__file__).resolve().parents[2],
        Path.cwd().resolve(),
    )
    for candidate in candidates:
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("无法找到包含 data/ 的项目根目录。")


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except OSError as exc:
        raise FileNotFoundError(f"无法读取 JSON 文件：{path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON 根对象必须是 object：{path}")
    return value


def load_roles_and_tv(
    roles_file: Path,
) -> tuple[dict[str, Any], list[str], list[str]]:
    roles = load_json(roles_file)
    task_vehicles = [str(vehicle_id) for vehicle_id in roles.get("task_vehicles", [])]
    service_vehicles = [
        str(vehicle_id) for vehicle_id in roles.get("service_vehicles", [])
    ]
    background_vehicles = {
        str(vehicle_id) for vehicle_id in roles.get("background_vehicles", [])
    }

    if not task_vehicles:
        raise ValueError("vehicle_roles.json 中没有 task_vehicles。")
    if len(task_vehicles) != len(set(task_vehicles)):
        raise ValueError("task_vehicles 中存在重复 vehicle_id。")
    if set(task_vehicles) & set(service_vehicles):
        raise ValueError("task_vehicles 与 service_vehicles 存在交集。")
    if set(task_vehicles) & background_vehicles:
        raise ValueError("task_vehicles 与 background_vehicles 存在交集。")

    declared_count = roles.get("num_task_vehicles")
    if declared_count is not None and int(declared_count) != len(task_vehicles):
        raise ValueError(
            "vehicle_roles.json 的 num_task_vehicles 与 task_vehicles 长度不一致。"
        )
    return roles, task_vehicles, service_vehicles


def resolve_time_metadata(
    roles: dict[str, Any],
    args: argparse.Namespace,
    defaults: TaskConfig,
    project_config: dict[str, Any] | None = None,
) -> tuple[float, float, float, int]:
    """读取正式时间轴；TOML 默认值和 CLI 覆盖优先于 artifact 元数据。"""
    if project_config is not None:
        from common.project_config import get_config_value

        configured_dt = float(get_config_value(project_config, "simulation.dt_s"))
        configured_warmup = float(
            get_config_value(project_config, "simulation.warmup_time_s")
        )
        configured_episode = float(
            get_config_value(project_config, "simulation.episode_time_s")
        )
        configured_seed = int(get_config_value(project_config, "task.master_seed"))
    else:
        configured_dt = defaults.fallback_dt_s
        configured_warmup = defaults.fallback_warmup_time_s
        configured_episode = defaults.fallback_episode_time_s
        configured_seed = defaults.fallback_master_seed

    dt_s = float(
        args.dt
        if args.dt is not None
        else configured_dt
    )
    warmup_s = float(
        args.warmup_time
        if args.warmup_time is not None
        else configured_warmup
    )
    episode_s = float(
        args.episode_time
        if args.episode_time is not None
        else configured_episode
    )
    master_seed = int(
        args.master_seed
        if args.master_seed is not None
        else configured_seed
    )

    # 角色文件是由上游 SUMO 生成的事实记录；它用于发现漂移，不再悄悄
    # 覆盖 system_config.toml 的正式默认值。
    if project_config is not None:
        for label, selected, key in (
            ("dt", dt_s, "dt"),
            ("warmup_time", warmup_s, "warmup_time"),
            ("episode_time", episode_s, "episode_time"),
        ):
            if key in roles and not math.isclose(
                float(roles[key]), selected, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(
                    f"vehicle_roles.json 的 {label}={roles[key]} 与正式配置 {selected} 不一致。"
                )
        if "seed" in roles and int(roles["seed"]) != master_seed:
            raise ValueError(
                f"vehicle_roles.json 的 seed={roles['seed']} 与正式 task.master_seed="
                f"{master_seed} 不一致。"
            )

    if dt_s <= 0.0 or warmup_s < 0.0 or episode_s <= 0.0:
        raise ValueError("dt/warmup_time/episode_time 的数值范围无效。")
    for name, value in (("warmup_time", warmup_s), ("episode_time", episode_s)):
        steps = value / dt_s
        if not math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"{name}={value} 不是 dt={dt_s} 的整数倍，无法建立正式时隙。"
            )
    return dt_s, warmup_s, episode_s, master_seed


def read_formal_trajectory_alignment(
    trajectory_file: Path,
    task_vehicles: list[str],
    dt_s: float,
    warmup_s: float,
    episode_s: float,
) -> tuple[int, list[float]]:
    """确认所有 TV 在每个正式时隙有且仅有一条 SUMO 轨迹记录。"""
    steps = int(round(episode_s / dt_s))
    expected_times = [
        round(warmup_s + slot * dt_s, 9)
        for slot in range(steps)
    ]
    expected_step_set = set(range(steps))
    task_set = set(task_vehicles)
    seen: dict[str, dict[int, int]] = {vehicle_id: {} for vehicle_id in task_vehicles}

    try:
        stream = trajectory_file.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise FileNotFoundError(f"无法读取 trajectory.csv：{trajectory_file}") from exc

    with stream:
        reader = csv.DictReader(stream)
        required = {"time_s", "vehicle_id"}
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError("trajectory.csv 缺少 time_s 或 vehicle_id 字段。")
        for row in reader:
            vehicle_id = str(row["vehicle_id"])
            if vehicle_id not in task_set:
                continue
            time_s = float(row["time_s"])
            relative_step = int(round((time_s - warmup_s) / dt_s))
            if relative_step not in expected_step_set:
                continue
            expected_time = expected_times[relative_step]
            if not math.isclose(time_s, expected_time, rel_tol=0.0, abs_tol=1e-6):
                continue
            per_vehicle = seen[vehicle_id]
            per_vehicle[relative_step] = per_vehicle.get(relative_step, 0) + 1

    for vehicle_id in task_vehicles:
        per_vehicle = seen[vehicle_id]
        if set(per_vehicle) != expected_step_set or any(
            count != 1 for count in per_vehicle.values()
        ):
            missing = sorted(expected_step_set - set(per_vehicle))[:5]
            raise RuntimeError(
                f"TV {vehicle_id} 未与 trajectory.csv 正式时隙完全对齐；"
                f"缺失示例 slot={missing}。"
            )
    return steps, expected_times


# ============================================================================
# 独立随机流和服务目录
# ============================================================================


def make_rngs(
    master_seed: int,
    stream_names: tuple[str, ...],
) -> tuple[dict[str, np.random.Generator], dict[str, int]]:
    """用 SeedSequence 派生互不耦合的 numpy Generator。"""
    root = np.random.SeedSequence(master_seed)
    children = root.spawn(len(stream_names))
    rngs: dict[str, np.random.Generator] = {}
    child_seeds: dict[str, int] = {}
    for name, child in zip(stream_names, children):
        rngs[name] = np.random.default_rng(child)
        child_seeds[name] = int(child.generate_state(1, dtype=np.uint64)[0])
    return rngs, child_seeds


def stratified_uniform_samples(
    rng: np.random.Generator,
    count: int,
    lower: float,
    upper: float,
) -> np.ndarray:
    """在闭区间 ``[lower, upper]`` 的每个等宽 strata 中采样一次。

    返回数组按 stratum 顺序排列；调用方应再使用独立 permutation 将
    各维度样本随机分配给 service_id。这样每个维度都覆盖整个 provisional
    envelope，同时不把 stratum 顺序赋予 service_id。
    """
    if isinstance(count, (bool, np.bool_)) or int(count) != count:
        raise ValueError("stratified sampling 的 count 必须是正整数。")
    count = int(count)
    if count <= 0:
        raise ValueError("stratified sampling 的 count 必须大于 0。")

    lower = float(lower)
    upper = float(upper)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("stratified sampling 的上下界不能是 NaN/Inf。")
    if upper < lower:
        raise ValueError("stratified sampling 要求 upper >= lower。")
    if upper == lower:
        return np.full(count, lower, dtype=np.float64)

    # u_k=(k+r_k)/F，r_k~U(0,1)；NumPy Generator.random 使用半开区间
    # [0,1)，因此样本不会因 r_k=1 落入下一个 stratum。
    jitter = rng.random(count)
    normalized = (np.arange(count, dtype=np.float64) + jitter) / count
    samples = lower + normalized * (upper - lower)
    # 仅消除浮点舍入造成的极小越界，不改变 strata 分配语义。
    return np.clip(samples, lower, upper).astype(np.float64, copy=False)


def make_service_catalog(
    config: TaskConfig,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """生成 service_id 对应的 task type 及其三维服务参数。

    每个参数维度单独分层采样，再由独立 permutation 随机分配给 service
    identity；因此 service_id 不携带 workload 大小排序语义。
    """
    program_samples = stratified_uniform_samples(
        rng,
        config.num_services,
        config.program_size_min_mb,
        config.program_size_max_mb,
    )
    input_samples_mbit = stratified_uniform_samples(
        rng,
        config.num_services,
        config.mean_input_size_min_mbit,
        config.mean_input_size_max_mbit,
    )
    cycle_samples = stratified_uniform_samples(
        rng,
        config.num_services,
        config.cycles_per_bit_min,
        config.cycles_per_bit_max,
    )

    # 三次 permutation 都从同一个 service_catalog child RNG 独立抽取；
    # 这里只共享随机流，不共享维度顺序。
    program_permutation = rng.permutation(config.num_services)
    input_permutation = rng.permutation(config.num_services)
    cycle_permutation = rng.permutation(config.num_services)

    catalog: list[dict[str, Any]] = []
    for service_id in range(config.num_services):
        catalog.append(
            {
                "service_id": service_id,
                "program_size": float(program_samples[program_permutation[service_id]]),
                "mean_input_size": float(
                    input_samples_mbit[input_permutation[service_id]]
                    * 1_000_000.0
                ),
                "cycles_per_bit": float(cycle_samples[cycle_permutation[service_id]]),
            }
        )
    return catalog


def catalog_by_id(catalog: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result = {int(row["service_id"]): row for row in catalog}
    if set(result) != set(range(len(catalog))):
        raise ValueError("service_id 必须覆盖 0 到 F-1。")
    return result


def service_parameter_stratum_summary(
    catalog: list[dict[str, Any]],
    config: TaskConfig,
) -> dict[str, Any]:
    """返回 service catalog 三个参数维度的 stratum 覆盖诊断。

    stratum 只属于生成 metadata/validation，不写入正式 service/task 行。
    ``mean_input_size`` 在 catalog 中是 bit，因此校验时先还原为 Mbit。
    """
    by_id = catalog_by_id(catalog)
    dimensions = (
        (
            "program_size_mb",
            "program_size",
            config.program_size_min_mb,
            config.program_size_max_mb,
            1.0,
        ),
        (
            "mean_input_size_mbit",
            "mean_input_size",
            config.mean_input_size_min_mbit,
            config.mean_input_size_max_mbit,
            1.0e-6,
        ),
        (
            "cycles_per_bit",
            "cycles_per_bit",
            config.cycles_per_bit_min,
            config.cycles_per_bit_max,
            1.0,
        ),
    )
    stratum_vectors: list[np.ndarray] = []
    parameter_values: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {}
    for label, field, lower, upper, scale in dimensions:
        values = np.asarray(
            [float(by_id[index][field]) * scale for index in range(config.num_services)],
            dtype=np.float64,
        )
        parameter_values[label] = values
        lower = float(lower)
        upper = float(upper)
        if np.any(values < lower) or np.any(values > upper):
            raise AssertionError(f"{label} 超出配置的 service generation envelope。")

        if upper == lower:
            # 零宽区间无法从数值上区分 strata；TaskConfig 仍允许它作为
            # 合法的退化范围，常规 provisional envelope 是非退化区间。
            stratum_indices = np.zeros(config.num_services, dtype=np.int64)
            counts = np.bincount(
                stratum_indices,
                minlength=config.num_services,
            )
            covered = config.num_services == 1
        else:
            normalized = (values - lower) / (upper - lower)
            stratum_indices = np.floor(
                normalized * config.num_services
            ).astype(np.int64)
            # upper 端点应归入最后一个 strata；同时防止浮点舍入越界。
            stratum_indices = np.clip(
                stratum_indices,
                0,
                config.num_services - 1,
            )
            counts = np.bincount(
                stratum_indices,
                minlength=config.num_services,
            )
            covered = bool(np.all(counts == 1))
        if not covered:
            raise AssertionError(
                f"{label} 未做到每个 stratum 恰好一个 service：{counts.tolist()}"
            )
        stratum_vectors.append(stratum_indices)
        summary[label] = {
            "stratum_count": config.num_services,
            "stratum_counts": counts.astype(int).tolist(),
            "stratum_index_by_service_id": stratum_indices.astype(int).tolist(),
        }

    all_orders_identical = bool(
        len(stratum_vectors) == 3
        and np.array_equal(stratum_vectors[0], stratum_vectors[1])
        and np.array_equal(stratum_vectors[0], stratum_vectors[2])
    )
    if config.num_services > 1 and all_orders_identical:
        raise AssertionError(
            "三个 service 参数维度的 stratum assignment 不能全部使用同一顺序。"
        )
    summary["independent_dimension_orders"] = (
        not all_orders_identical or config.num_services <= 1
    )

    def safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
        if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
            return None
        value = float(np.corrcoef(left, right)[0, 1])
        return value if math.isfinite(value) else None

    summary["pairwise_correlations"] = {
        "program_size_vs_mean_input_size": safe_correlation(
            parameter_values["program_size_mb"],
            parameter_values["mean_input_size_mbit"],
        ),
        "program_size_vs_cycles_per_bit": safe_correlation(
            parameter_values["program_size_mb"],
            parameter_values["cycles_per_bit"],
        ),
        "mean_input_size_vs_cycles_per_bit": safe_correlation(
            parameter_values["mean_input_size_mbit"],
            parameter_values["cycles_per_bit"],
        ),
    }
    return summary


# ============================================================================
# 动态 Zipf 真实流行度 rho_star
# ============================================================================


def zipf_probabilities(
    ranking: list[int],
    num_services: int,
    zeta: float,
) -> np.ndarray:
    """按排名计算 rho_star_f = rank_f^(-zeta) / sum_j rank_j^(-zeta)。"""
    probabilities = np.zeros(num_services, dtype=float)
    ranks = np.arange(1, num_services + 1, dtype=float)
    weights = ranks ** (-zeta)
    weights /= weights.sum()
    for rank_index, service_id in enumerate(ranking):
        probabilities[service_id] = weights[rank_index]
    return probabilities


def evolve_ranking(
    old_ranking: list[int],
    rng: np.random.Generator,
    changes_per_epoch: int,
) -> list[int]:
    """只做少量相邻交换/局部移动，禁止每个 epoch 全量随机重排。"""
    ranking = list(old_ranking)
    size = len(ranking)
    for _ in range(changes_per_epoch):
        if size < 2:
            break
        index = int(rng.integers(0, size))
        if index == 0:
            neighbour = 1
        elif index == size - 1:
            neighbour = size - 2
        else:
            neighbour = index + (1 if int(rng.integers(0, 2)) == 1 else -1)

        if int(rng.integers(0, 2)) == 0:
            # pairwise rank swap
            ranking[index], ranking[neighbour] = ranking[neighbour], ranking[index]
        else:
            # local rank shift：只将一个服务向相邻 rank 移动一格。
            service_id = ranking.pop(index)
            insert_at = neighbour
            ranking.insert(max(0, insert_at), service_id)

    if sorted(ranking) != list(range(size)):
        raise AssertionError("动态排名演化破坏了 service_id 排列。")
    return ranking


def make_popularity_trace(
    num_services: int,
    num_slots: int,
    dt_s: float,
    time_origin_s: float,
    config: TaskConfig,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    """兼容接口：只生成没有 prehistory 的独立序列。"""
    formal_rows, _, probabilities_by_slot, _, schedule = make_popularity_trace_with_history(
        num_services,
        num_slots,
        0,
        dt_s,
        time_origin_s,
        config,
        rng,
    )
    return formal_rows, probabilities_by_slot, schedule


def make_popularity_trace_with_history(
    num_services: int,
    num_slots: int,
    history_slots: int,
    dt_s: float,
    time_origin_s: float,
    config: TaskConfig,
    rng: np.random.Generator,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """连续生成 prehistory -> formal 的动态 Zipf 序列。

    ``formal_slot`` 是正式 episode 的坐标，prehistory 使用负值；
    ``history_index`` 只在 history CSV 中使用，绝不伪装成 formal slot。
    """
    if num_slots <= 0 or history_slots < 0:
        raise ValueError("num_slots 必须大于 0，history_slots 不能为负。")
    total_slots = history_slots + num_slots
    ranking = [int(value) for value in rng.permutation(num_services)]
    combined_probabilities = np.zeros((total_slots, num_services), dtype=float)
    formal_probabilities = np.zeros((num_slots, num_services), dtype=float)
    formal_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    schedule_changes: list[dict[str, Any]] = []
    formal_initial_ranking: list[int] | None = None

    for sequence_slot in range(total_slots):
        formal_slot = sequence_slot - history_slots
        epoch = formal_slot // config.popularity_epoch_length
        # epoch 坐标以 formal_slot=0 为基准，而不是以 combined 数组的
        # history 长度为基准；这样 formal epoch 0...19 内不会因 L_rho
        # 不是 epoch_length 的整数倍而发生中途 rank 变化。prehistory 仍
        # 共享同一条 ranking 状态，并在负 formal epoch 边界连续演化。
        if sequence_slot > 0 and formal_slot % config.popularity_epoch_length == 0:
            previous = list(ranking)
            ranking = evolve_ranking(
                ranking,
                rng,
                config.popularity_changes_per_epoch,
            )
            schedule_changes.append(
                {
                    "epoch": epoch,
                    "slot": formal_slot,
                    "formal_slot": formal_slot,
                    "history_index": (
                        sequence_slot if formal_slot < 0 else None
                    ),
                    "previous_ranking": previous,
                    "new_ranking": list(ranking),
                }
            )
        if formal_slot == 0:
            formal_initial_ranking = list(ranking)

        probabilities = zipf_probabilities(
            ranking,
            num_services,
            config.zipf_exponent,
        )
        combined_probabilities[sequence_slot] = probabilities
        if formal_slot >= 0:
            formal_probabilities[formal_slot] = probabilities
        rank_by_service = {service_id: rank for rank, service_id in enumerate(ranking, 1)}
        time_s = round(time_origin_s + formal_slot * dt_s, 9)
        for service_id in range(num_services):
            if formal_slot < 0:
                history_rows.append(
                    {
                        "history_index": sequence_slot,
                        "formal_slot": formal_slot,
                        "time_s": time_s,
                        "epoch": epoch,
                        "service_id": service_id,
                        "rank": rank_by_service[service_id],
                        "rho_star": float(probabilities[service_id]),
                        "request_count": 0,
                        "rho_obs": "",
                    }
                )
            else:
                formal_rows.append(
                    {
                        "time_s": time_s,
                        "slot": formal_slot,
                        "epoch": epoch,
                        "service_id": service_id,
                        "rank": rank_by_service[service_id],
                        "rho_star": float(probabilities[service_id]),
                        "request_count": 0,
                        "rho_obs": "",
                    }
                )

    formal_epoch_count = int(
        math.ceil(num_slots / config.popularity_epoch_length)
    )
    schedule = {
        "initial_ranking": formal_initial_ranking or [],
        "prehistory_initial_ranking": (
            [int(value) for value in combined_probabilities[0].argsort()[::-1]]
            if history_slots > 0
            else formal_initial_ranking or []
        ),
        "epoch_changes": schedule_changes,
        "epoch_count": formal_epoch_count,
        "history_slots": history_slots,
        "formal_slots": num_slots,
    }
    return (
        formal_rows,
        history_rows,
        formal_probabilities,
        combined_probabilities,
        schedule,
    )


# ============================================================================
# 真实任务生成
# ============================================================================


def generate_tasks(
    task_vehicles: list[str],
    catalog: list[dict[str, Any]],
    probabilities_by_slot: np.ndarray,
    num_slots: int,
    dt_s: float,
    warmup_s: float,
    config: TaskConfig,
    arrival_rng: np.random.Generator,
    request_rng: np.random.Generator,
    workload_rng: np.random.Generator,
    slot_offset: int = 0,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """按 TV×slot 生成一行状态；arrival=0 不构造伪任务。

    ``slot_offset`` 仅用于把连续 prehistory 标成负 formal slot；返回的
    request_counts 仍按本次传入的局部数组顺序排列。
    """
    by_service = catalog_by_id(catalog)
    request_counts = np.zeros((num_slots, config.num_services), dtype=int)
    rows: list[dict[str, Any]] = []

    for slot in range(num_slots):
        probabilities = probabilities_by_slot[slot]
        formal_slot = slot + slot_offset
        arrivals = arrival_rng.random(len(task_vehicles)) < config.arrival_probability
        for vehicle_index, vehicle_id in enumerate(task_vehicles):
            arrival = int(bool(arrivals[vehicle_index]))
            time_s = round(warmup_s + slot * dt_s, 9)
            episode_time_s = round(formal_slot * dt_s, 9)

            if arrival == 0:
                rows.append(
                    {
                        "time_s": time_s,
                        "slot": formal_slot,
                        "episode_time_s": episode_time_s,
                        "vehicle_id": vehicle_id,
                        "arrival": 0,
                        "service_id": "",
                        "input_size": 0.0,
                        "cpu_cycles": 0.0,
                    }
                )
                continue

            service_id = int(
                request_rng.choice(config.num_services, p=probabilities)
            )
            service = by_service[service_id]
            input_factor = 1.0 + float(
                workload_rng.uniform(
                    -config.input_relative_jitter,
                    config.input_relative_jitter,
                )
            )
            load_factor = 1.0 + float(
                workload_rng.uniform(
                    -config.load_relative_jitter,
                    config.load_relative_jitter,
                )
            )
            input_size = float(service["mean_input_size"]) * input_factor
            cpu_cycles = (
                float(service["cycles_per_bit"])
                * input_size
                * load_factor
            )
            if input_size <= 0.0 or cpu_cycles <= 0.0:
                raise AssertionError("有效任务的 input_size/cpu_cycles 必须大于 0。")

            request_counts[slot, service_id] += 1
            rows.append(
                {
                    "time_s": time_s,
                    "slot": formal_slot,
                    "episode_time_s": episode_time_s,
                    "vehicle_id": vehicle_id,
                    "arrival": 1,
                    "service_id": service_id,
                    "input_size": input_size,
                    "cpu_cycles": cpu_cycles,
                }
            )

    expected_rows = len(task_vehicles) * num_slots
    if len(rows) != expected_rows:
        raise AssertionError(
            f"tasks.csv 行数错误：{len(rows)}，期望 {expected_rows}。"
        )
    return rows, request_counts


def build_observed_popularity(request_counts: np.ndarray) -> np.ndarray:
    """把逐时隙请求次数转换为因果 observed popularity。

    有请求的时隙使用当前时隙的请求比例；空时隙沿用上一时隙的比例。
    连续序列第一个时隙若为空，则从均匀分布初始化。该函数只读取当前
    及此前的 request_counts，不读取 rho_star 或未来时隙。
    """
    counts = np.asarray(request_counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] <= 0:
        raise ValueError("request_counts 必须是 [num_slots, num_services] 的非空二维数组。")
    if not np.isfinite(counts).all():
        raise ValueError("request_counts 不能包含 NaN/Inf。")
    if np.any(counts < 0.0):
        raise ValueError("request_counts 不能为负数。")

    num_slots, num_services = counts.shape
    previous = np.full(num_services, 1.0 / num_services, dtype=np.float64)
    rho_obs = np.zeros((num_slots, num_services), dtype=np.float64)
    for slot in range(num_slots):
        total = float(counts[slot].sum())
        if total > 0.0:
            previous = counts[slot] / total
        rho_obs[slot] = previous

    if not np.isfinite(rho_obs).all() or not np.allclose(
        rho_obs.sum(axis=1),
        1.0,
        atol=1e-10,
        rtol=0.0,
    ):
        raise AssertionError("observed popularity 未逐时隙归一化或包含 NaN/Inf。")
    return rho_obs


def attach_observed_counts(
    popularity_rows: list[dict[str, Any]],
    request_counts: np.ndarray,
    rho_obs: np.ndarray,
) -> None:
    """把 formal request_count 和已计算的 observed popularity 写入行。"""
    counts = np.asarray(request_counts, dtype=np.float64)
    observed = np.asarray(rho_obs, dtype=np.float64)
    if counts.ndim != 2 or observed.shape != counts.shape:
        raise ValueError("request_counts 与 rho_obs 的 shape 必须一致。")
    num_services = counts.shape[1]
    for row in popularity_rows:
        slot = int(row["slot"])
        service_id = int(row["service_id"])
        count = int(counts[slot, service_id])
        row["request_count"] = count
        row["rho_obs"] = float(observed[slot, service_id])
    expected_rows = request_counts.shape[0] * num_services
    if len(popularity_rows) != expected_rows:
        raise AssertionError("popularity 行数与 slot×service 数量不一致。")


def attach_history_observed_counts(
    popularity_history_rows: list[dict[str, Any]],
    request_counts: np.ndarray,
    rho_obs: np.ndarray,
) -> None:
    """把 history request_count 和连续 observed popularity 写入 history 行。"""
    counts = np.asarray(request_counts, dtype=np.float64)
    observed = np.asarray(rho_obs, dtype=np.float64)
    if counts.ndim != 2 or observed.shape != counts.shape:
        raise ValueError("request_counts 与 rho_obs 的 shape 必须一致。")
    num_services = counts.shape[1]
    for row in popularity_history_rows:
        history_index = int(row["history_index"])
        service_id = int(row["service_id"])
        if not 0 <= history_index < request_counts.shape[0]:
            raise AssertionError("history_index 超出 prehistory request_counts。")
        count = int(counts[history_index, service_id])
        row["request_count"] = count
        row["rho_obs"] = float(observed[history_index, service_id])
    expected_rows = request_counts.shape[0] * num_services
    if len(popularity_history_rows) != expected_rows:
        raise AssertionError("popularity_history 行数与 history slot×service 数量不一致。")


# ============================================================================
# 校验、输出和可复现性
# ============================================================================


def json_canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def service_size_by_id(service_catalog: dict[str, Any]) -> dict[int, float]:
    """从正式 service_catalog.json 读取 service_id -> program_size(MB)。"""

    services = service_catalog.get("services")
    if not isinstance(services, list) or not services:
        raise ValueError("service_catalog.json 缺少非空 services。")
    result: dict[int, float] = {}
    for row in services:
        service_id = int(row["service_id"])
        size = float(row["program_size"])
        if size <= 0.0:
            raise ValueError("service program_size 必须大于 0。")
        if service_id in result:
            raise ValueError(f"重复 service_id：{service_id}")
        result[service_id] = size
    return result


def service_size_sanity(
    service_sizes_mb: dict[int, float],
    local_cache_capacity_mb: float,
) -> dict[str, Any]:
    """验证 local cache capacity 位于 max single 与 min two-service sum 之间。"""

    if local_cache_capacity_mb <= 0.0:
        raise ValueError("TV local cache capacity 必须大于 0。")
    ordered = sorted(service_sizes_mb.items(), key=lambda item: (item[1], item[0]))
    if len(ordered) < 2:
        raise ValueError("至少需要两个 service 才能验证 min_two_service_sum。")
    max_single = max(service_sizes_mb.values())
    min_two = ordered[0][1] + ordered[1][1]
    if not (max_single < local_cache_capacity_mb < min_two):
        raise ValueError(
            "TV local cache capacity 必须满足 "
            "max_single_service < capacity < min_two_service_sum。"
        )
    return {
        "service_count": len(service_sizes_mb),
        "min_service": {"service_id": ordered[0][0], "size_mb": ordered[0][1]},
        "second_min_service": {"service_id": ordered[1][0], "size_mb": ordered[1][1]},
        "min_service_size_mb": ordered[0][1],
        "max_service_size_mb": max_single,
        "min_two_service_sum_mb": min_two,
        "total_service_size_mb": float(sum(service_sizes_mb.values())),
        "single_services_fit_count": int(
            sum(1 for size in service_sizes_mb.values() if size <= local_cache_capacity_mb)
        ),
        "all_pairs_exceed_capacity": True,
    }


def generate_tv_local_cache(
    task_vehicle_ids: list[str],
    service_catalog: dict[str, Any],
    local_cache_capacity_mb: float,
    seed: int,
) -> dict[str, Any]:
    """为每个TV生成实验前固定的本地缓存初始状态。

    生成规则严格为 uniform one-service-per-TV，不读取 tasks.csv/popularity/future
    request，因此不泄漏未来ground truth。相同 TV 集合、service catalog 和 seed
    会得到完全相同 artifact。
    """

    service_sizes = service_size_by_id(service_catalog)
    sanity = service_size_sanity(service_sizes, local_cache_capacity_mb)
    tv_ids = sorted([str(vehicle_id) for vehicle_id in task_vehicle_ids], key=_stable_id_key)
    if len(tv_ids) != len(set(tv_ids)):
        raise ValueError("task_vehicle_ids 存在重复。")
    service_ids = sorted(service_sizes)
    rng = np.random.default_rng(int(seed))
    vehicles: dict[str, dict[str, Any]] = {}
    for tv_id in tv_ids:
        service_id = int(rng.choice(service_ids))
        used = float(service_sizes[service_id])
        if used > local_cache_capacity_mb:
            raise AssertionError("抽中的单个service超过TV local cache容量。")
        vehicles[tv_id] = {
            "cached_service_ids": [service_id],
            "used_capacity_mb": used,
        }
    return {
        "schema_version": 1,
        "metadata": {
            "seed": int(seed),
            "capacity_mb": float(local_cache_capacity_mb),
            "generation_rule": "uniform_one_service_per_tv",
            "service_count": len(service_ids),
            "service_ids": service_ids,
            "tv_count": len(tv_ids),
            "tv_ids": tv_ids,
            "fixed_over_episode": True,
            "uses_future_requests": False,
            "parameter_basis": "system-scale experimental baseline derived from current service catalog",
            "service_size_sanity": sanity,
        },
        "vehicles": vehicles,
    }


def validate_tv_local_cache(
    tv_local_cache: dict[str, Any],
    task_vehicle_ids: list[str],
    service_catalog: dict[str, Any],
    local_cache_capacity_mb: float,
) -> dict[str, Any]:
    service_sizes = service_size_by_id(service_catalog)
    sanity = service_size_sanity(service_sizes, local_cache_capacity_mb)
    valid_services = set(service_sizes)
    expected_tv = {str(vehicle_id) for vehicle_id in task_vehicle_ids}
    vehicles = tv_local_cache.get("vehicles")
    if not isinstance(vehicles, dict):
        raise ValueError("tv_local_cache.json 缺少 vehicles object。")
    actual_tv = set(str(vehicle_id) for vehicle_id in vehicles)
    if actual_tv != expected_tv:
        raise ValueError("tv_local_cache TV集合与vehicle_roles task_vehicles不一致。")
    for tv_id, row in vehicles.items():
        cached = [int(service_id) for service_id in row.get("cached_service_ids", [])]
        if len(cached) != 1:
            raise ValueError(f"TV {tv_id} 必须恰好缓存1个service。")
        if cached[0] not in valid_services:
            raise ValueError(f"TV {tv_id} 缓存了非法service_id。")
        used = sum(float(service_sizes[service_id]) for service_id in cached)
        if used > local_cache_capacity_mb + 1e-9:
            raise ValueError(f"TV {tv_id} used_capacity 超过容量。")
        if not math.isclose(float(row["used_capacity_mb"]), used, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"TV {tv_id} used_capacity_mb 与service size不一致。")
    return {
        "tv_count": len(actual_tv),
        "service_count": len(valid_services),
        "all_tv_exactly_once": True,
        "all_cached_services_valid": True,
        "all_used_capacity_within_limit": True,
        "service_size_sanity": sanity,
    }


def load_tv_local_cache(path: Path) -> dict[str, set[int]]:
    """为后续Matching提供轻量读取接口：TV ID -> cached service set。"""

    raw = load_json(path)
    vehicles = raw.get("vehicles")
    if not isinstance(vehicles, dict):
        raise ValueError("tv_local_cache.json 缺少 vehicles object。")
    return {
        str(tv_id): {int(service_id) for service_id in row.get("cached_service_ids", [])}
        for tv_id, row in vehicles.items()
    }


def build_artifacts(
    task_vehicles: list[str],
    dt_s: float,
    warmup_s: float,
    episode_s: float,
    master_seed: int,
    config: TaskConfig,
) -> dict[str, Any]:
    num_slots = int(round(episode_s / dt_s))
    history_slots = int(config.popularity_history_length)
    rngs, child_seeds = make_rngs(master_seed, config.rng_stream_names)
    catalog = make_service_catalog(config, rngs["service_catalog"])
    (
        popularity_rows,
        popularity_history_rows,
        probabilities,
        combined_probabilities,
        schedule,
    ) = make_popularity_trace_with_history(
        config.num_services,
        num_slots,
        history_slots,
        dt_s,
        warmup_s,
        config,
        rngs["popularity_evolution"],
    )
    combined_task_rows, combined_request_counts = generate_tasks(
        task_vehicles,
        catalog,
        combined_probabilities,
        history_slots + num_slots,
        dt_s,
        warmup_s - history_slots * dt_s,
        config,
        rngs["task_arrival"],
        rngs["service_request"],
        rngs["workload_variation"],
        slot_offset=-history_slots,
    )
    task_rows = [row for row in combined_task_rows if int(row["slot"]) >= 0]
    history_request_counts = combined_request_counts[:history_slots].copy()
    request_counts = combined_request_counts[history_slots:].copy()
    # history 与 formal 必须共享同一条连续 observed popularity；否则 formal
    # slot 0 的空请求无法正确继承 prehistory slot -1 的观测比例。
    combined_rho_obs = build_observed_popularity(combined_request_counts)
    history_rho_obs = combined_rho_obs[:history_slots].copy()
    rho_obs = combined_rho_obs[history_slots:].copy()
    attach_observed_counts(popularity_rows, request_counts, rho_obs)
    attach_history_observed_counts(
        popularity_history_rows,
        history_request_counts,
        history_rho_obs,
    )

    return {
        "catalog": catalog,
        "popularity_rows": popularity_rows,
        "popularity_history_rows": popularity_history_rows,
        "task_rows": task_rows,
        "probabilities": probabilities,
        "combined_probabilities": combined_probabilities,
        "combined_rho_obs": combined_rho_obs,
        "request_counts": request_counts,
        "history_request_counts": history_request_counts,
        "schedule": schedule,
        "child_seeds": child_seeds,
        "num_slots": num_slots,
        "history_slots": history_slots,
    }


def validate_artifacts(
    artifacts: dict[str, Any],
    task_vehicles: list[str],
    config: TaskConfig,
    expected_lambda: float,
) -> dict[str, Any]:
    catalog = artifacts["catalog"]
    popularity_rows = artifacts["popularity_rows"]
    task_rows = artifacts["task_rows"]
    probabilities = artifacts["probabilities"]
    request_counts = artifacts["request_counts"]
    popularity_history_rows = artifacts["popularity_history_rows"]
    history_request_counts = artifacts["history_request_counts"]
    schedule = artifacts["schedule"]
    num_slots = int(artifacts["num_slots"])
    history_slots = int(artifacts["history_slots"])

    if len(catalog) != config.num_services:
        raise AssertionError("service catalog 数量错误。")
    catalog_ids = {int(row["service_id"]) for row in catalog}
    expected_ids = set(range(config.num_services))
    if catalog_ids != expected_ids:
        raise AssertionError("service catalog 的 service_id 不完整。")
    for row in catalog:
        for field in ("program_size", "mean_input_size", "cycles_per_bit"):
            if float(row[field]) <= 0.0:
                raise AssertionError(f"服务参数 {field} 必须大于 0。")
    catalog_parameter_summary = service_parameter_stratum_summary(catalog, config)

    if len(popularity_rows) != num_slots * config.num_services:
        raise AssertionError("popularity.csv 行数错误。")
    if len(popularity_history_rows) != history_slots * config.num_services:
        raise AssertionError("popularity_history.csv 行数错误。")
    expected_history_slots = list(range(-history_slots, 0))
    actual_history_slots = sorted(
        {int(row["formal_slot"]) for row in popularity_history_rows}
    )
    if actual_history_slots != expected_history_slots:
        raise AssertionError("prehistory formal_slot 没有覆盖 -L_rho ... -1。")
    actual_history_indices = sorted(
        {int(row["history_index"]) for row in popularity_history_rows}
    )
    if actual_history_indices != list(range(history_slots)):
        raise AssertionError("prehistory history_index 没有从 0 连续编号。")
    for slot in range(num_slots):
        slot_rows = [row for row in popularity_rows if int(row["slot"]) == slot]
        if len(slot_rows) != config.num_services:
            raise AssertionError(f"slot={slot} 的服务流行度行数错误。")
        if not math.isclose(
            sum(float(row["rho_star"]) for row in slot_rows),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise AssertionError(f"slot={slot} 的 rho_star 未归一化。")
        ranks = sorted(int(row["rank"]) for row in slot_rows)
        if ranks != list(range(1, config.num_services + 1)):
            raise AssertionError(f"slot={slot} 的 rank 不是完整排列。")

    expected_task_rows = len(task_vehicles) * num_slots
    if len(task_rows) != expected_task_rows:
        raise AssertionError("tasks.csv 没有覆盖全部 TV×slot。")
    allowed_tv = set(task_vehicles)
    arrivals = []
    observed_requests = np.zeros(config.num_services, dtype=int)
    per_slot_vehicle: dict[int, set[str]] = {slot: set() for slot in range(num_slots)}
    for row in task_rows:
        vehicle_id = str(row["vehicle_id"])
        slot = int(row["slot"])
        arrival = int(row["arrival"])
        if vehicle_id not in allowed_tv:
            raise AssertionError("tasks.csv 出现非 TV 车辆。")
        if slot not in per_slot_vehicle:
            raise AssertionError("tasks.csv 出现越界 slot。")
        if arrival not in (0, 1):
            raise AssertionError("arrival 不属于 {0,1}。")
        per_slot_vehicle[slot].add(vehicle_id)
        arrivals.append(arrival)
        service_value = row["service_id"]
        input_size = float(row["input_size"])
        cpu_cycles = float(row["cpu_cycles"])
        if arrival == 0:
            if service_value not in ("", None) or input_size != 0.0 or cpu_cycles != 0.0:
                raise AssertionError("arrival=0 时存在伪任务属性。")
        else:
            service_id = int(service_value)
            if service_id not in expected_ids:
                raise AssertionError("arrival=1 时 service_id 非法。")
            if input_size <= 0.0 or cpu_cycles <= 0.0:
                raise AssertionError("arrival=1 时 input_size/cpu_cycles 必须大于 0。")
            observed_requests[service_id] += 1

    for slot, vehicles in per_slot_vehicle.items():
        if vehicles != allowed_tv:
            raise AssertionError(f"slot={slot} 未覆盖全部 TV。")

    total_opportunities = len(task_rows)
    total_arrivals = int(sum(arrivals))
    empirical_lambda = total_arrivals / total_opportunities
    standard_error = math.sqrt(
        max(expected_lambda * (1.0 - expected_lambda), 1e-12)
        / total_opportunities
    )
    lambda_tolerance = max(0.08, 4.0 * standard_error)
    if abs(empirical_lambda - expected_lambda) > lambda_tolerance:
        raise AssertionError(
            f"empirical arrival rate={empirical_lambda:.4f} 与 lambda="
            f"{expected_lambda:.4f} 偏差过大。"
        )

    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10, rtol=0.0):
        raise AssertionError("rho_star 概率矩阵存在未归一化时隙。")
    if history_request_counts.shape != (history_slots, config.num_services):
        raise AssertionError("prehistory request_counts shape 错误。")
    if request_counts.shape != (num_slots, config.num_services):
        raise AssertionError("formal request_counts shape 错误。")

    # 一个 epoch 内排名必须完全不变；epoch 边界只允许少量 rank 变化。
    epoch_length = config.popularity_epoch_length
    for start in range(0, num_slots, epoch_length):
        end = min(num_slots, start + epoch_length)
        ranks_in_epoch = [
            tuple(
                int(row["rank"])
                for row in sorted(
                    popularity_rows,
                    key=lambda item: int(item["service_id"]),
                )
                if int(row["slot"]) == slot
            )
            for slot in range(start, end)
        ]
        if ranks_in_epoch and any(signature != ranks_in_epoch[0] for signature in ranks_in_epoch[1:]):
            raise AssertionError(f"epoch {start // epoch_length} 内 rank 发生非预期变化。")

    adjacent_same_fraction = float(
        np.mean(
            [
                np.array_equal(probabilities[slot], probabilities[slot - 1])
                for slot in range(1, num_slots)
            ]
        )
    ) if num_slots > 1 else 1.0
    if adjacent_same_fraction < 0.5:
        raise AssertionError("动态 Zipf 变化过于频繁，缺少时间相关性。")

    average_rho = probabilities.mean(axis=0)
    empirical_service = observed_requests / max(total_arrivals, 1)
    if total_arrivals > 0 and config.num_services > 1:
        correlation = float(np.corrcoef(average_rho, empirical_service)[0, 1])
        if not math.isfinite(correlation) or correlation < 0.2:
            raise AssertionError(
                f"请求服务分布与 rho_star 方向相关性过低：{correlation:.3f}。"
            )
    else:
        correlation = float("nan")

    if int(request_counts.sum()) != total_arrivals:
        raise AssertionError("request_counts 与 tasks.csv 的 arrival 不一致。")

    return {
        "num_slots": num_slots,
        "num_tv": len(task_vehicles),
        "task_rows": len(task_rows),
        "total_arrivals": total_arrivals,
        "empirical_arrival_rate": empirical_lambda,
        "lambda_tolerance": lambda_tolerance,
        "rho_star_normalized": True,
        "adjacent_same_probability_fraction": adjacent_same_fraction,
        "service_distribution_correlation": correlation,
        "service_parameter_generation": catalog_parameter_summary,
        "epoch_count": int(schedule["epoch_count"]),
        "history_slots": history_slots,
        "history_request_total": int(history_request_counts.sum()),
    }


def write_csv_atomic(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def make_service_catalog_output(
    artifacts: dict[str, Any],
    config: TaskConfig,
    dt_s: float,
    warmup_s: float,
    episode_s: float,
    master_seed: int,
    child_seeds: dict[str, int],
    task_vehicles: list[str],
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "service_count": config.num_services,
        "units": {
            "program_size": config.program_size_unit,
            "mean_input_size": config.input_size_unit,
            "cycles_per_bit": config.cycles_per_bit_unit,
            "tasks.input_size": config.input_size_unit,
            "tasks.cpu_cycles": config.cpu_cycles_unit,
        },
        "provisional_defaults": True,
        "task_type_definition": TASK_TYPE_DEFINITION,
        "service_generation_mode": SERVICE_GENERATION_MODE,
        "parameter_assignment": SERVICE_PARAMETER_ASSIGNMENT,
        "parameter_ranges": {
            "program_size_mb": [
                float(config.program_size_min_mb),
                float(config.program_size_max_mb),
            ],
            "mean_input_size_mbit": [
                float(config.mean_input_size_min_mbit),
                float(config.mean_input_size_max_mbit),
            ],
            "cycles_per_bit": [
                float(config.cycles_per_bit_min),
                float(config.cycles_per_bit_max),
            ],
        },
        "strata_count": config.num_services,
        "parameter_ranges_status": (
            "PROVISIONAL service heterogeneity envelope; these ranges preserve "
            "the existing experiment scale and are not claimed as final "
            "literature-calibrated values."
        ),
        "services": artifacts["catalog"],
        "generation_config": {
            "master_seed": master_seed,
            "child_seeds": child_seeds,
            "num_task_vehicles": len(task_vehicles),
            "task_vehicle_ids": task_vehicles,
            "dt_s": dt_s,
            "warmup_time_s": warmup_s,
            "episode_time_s": episode_s,
            "num_slots": artifacts["num_slots"],
            "popularity_history_length": artifacts["history_slots"],
            "arrival_probability": config.arrival_probability,
            "zipf_exponent": config.zipf_exponent,
            "popularity_epoch_length": config.popularity_epoch_length,
            "popularity_changes_per_epoch": config.popularity_changes_per_epoch,
            "input_relative_jitter": config.input_relative_jitter,
            "load_relative_jitter": config.load_relative_jitter,
        },
        "popularity_generation": artifacts["schedule"],
        "validation_summary": validation,
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def output_artifacts(
    output_dir: Path,
    catalog_output: dict[str, Any],
    popularity_history_rows: list[dict[str, Any]],
    popularity_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_file = output_dir / "service_catalog.json"
    popularity_history_file = output_dir / "popularity_history.csv"
    popularity_file = output_dir / "popularity.csv"
    tasks_file = output_dir / "tasks.csv"
    write_json_atomic(catalog_file, catalog_output)
    write_csv_atomic(
        popularity_history_file,
        POPULARITY_HISTORY_FIELDS,
        popularity_history_rows,
    )
    write_csv_atomic(popularity_file, POPULARITY_FIELDS, popularity_rows)
    write_csv_atomic(tasks_file, TASK_FIELDS, task_rows)
    return catalog_file, popularity_history_file, popularity_file, tasks_file


def task_config_from_project_config(
    project_config: dict[str, Any],
    fallback: TaskConfig | None = None,
) -> TaskConfig:
    """把 system_config.toml 的正式任务参数映射到生成器配置。"""
    from common.project_config import get_config_value

    base = fallback or DEFAULT_CONFIG
    return TaskConfig(
        num_services=int(
            get_config_value(project_config, "service.count", base.num_services)
        ),
        program_size_min_mb=float(
            get_config_value(
                project_config,
                "service.generation.program_size_min_mb",
                base.program_size_min_mb,
            )
        ),
        program_size_max_mb=float(
            get_config_value(
                project_config,
                "service.generation.program_size_max_mb",
                base.program_size_max_mb,
            )
        ),
        mean_input_size_min_mbit=float(
            get_config_value(
                project_config,
                "service.generation.mean_input_size_min_mbit",
                base.mean_input_size_min_mbit,
            )
        ),
        mean_input_size_max_mbit=float(
            get_config_value(
                project_config,
                "service.generation.mean_input_size_max_mbit",
                base.mean_input_size_max_mbit,
            )
        ),
        cycles_per_bit_min=float(
            get_config_value(
                project_config,
                "service.generation.cycles_per_bit_min",
                base.cycles_per_bit_min,
            )
        ),
        cycles_per_bit_max=float(
            get_config_value(
                project_config,
                "service.generation.cycles_per_bit_max",
                base.cycles_per_bit_max,
            )
        ),
        arrival_probability=float(
            get_config_value(
                project_config,
                "task.arrival_probability",
                base.arrival_probability,
            )
        ),
        zipf_exponent=float(
            get_config_value(project_config, "task.popularity_zeta", base.zipf_exponent)
        ),
        popularity_epoch_length=int(
            get_config_value(
                project_config,
                "task.popularity_epoch_length",
                base.popularity_epoch_length,
            )
        ),
        popularity_changes_per_epoch=int(
            get_config_value(
                project_config,
                "task.popularity_changes_per_epoch",
                base.popularity_changes_per_epoch,
            )
        ),
        popularity_history_length=int(
            get_config_value(
                project_config,
                "prediction.popularity.history_length",
                base.popularity_history_length,
            )
        ),
        input_relative_jitter=float(
            get_config_value(
                project_config,
                "task.input_relative_jitter",
                base.input_relative_jitter,
            )
        ),
        load_relative_jitter=float(
            get_config_value(
                project_config,
                "task.load_relative_jitter",
                base.load_relative_jitter,
            )
        ),
        fallback_dt_s=float(
            get_config_value(project_config, "simulation.dt_s", base.fallback_dt_s)
        ),
        fallback_warmup_time_s=float(
            get_config_value(
                project_config,
                "simulation.warmup_time_s",
                base.fallback_warmup_time_s,
            )
        ),
        fallback_episode_time_s=float(
            get_config_value(
                project_config,
                "simulation.episode_time_s",
                base.fallback_episode_time_s,
            )
        ),
        fallback_master_seed=int(
            get_config_value(project_config, "task.master_seed", base.fallback_master_seed)
        ),
        rng_stream_names=base.rng_stream_names,
        program_size_unit=base.program_size_unit,
        input_size_unit=base.input_size_unit,
        cpu_cycles_unit=base.cpu_cycles_unit,
        cycles_per_bit_unit=base.cycles_per_bit_unit,
    )


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成真实动态 rho_star、TV 任务到达和服务相关 D/L。"
    )
    parser.add_argument("--roles-file", default=None)
    parser.add_argument("--trajectory-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-services", type=int, default=None)
    parser.add_argument(
        "--arrival-probability",
        type=float,
        default=None,
    )
    parser.add_argument("--zipf-exponent", type=float, default=None)
    parser.add_argument(
        "--popularity-epoch-length",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--popularity-changes-per-epoch",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--input-relative-jitter",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--load-relative-jitter",
        type=float,
        default=None,
    )
    parser.add_argument("--popularity-history-length", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--warmup-time", type=float, default=None)
    parser.add_argument("--episode-time", type=float, default=None)
    parser.add_argument("--master-seed", type=int, default=None)
    return parser


def config_from_args(
    args: argparse.Namespace,
    project_config: dict[str, Any] | None = None,
) -> TaskConfig:
    base = (
        task_config_from_project_config(project_config)
        if project_config is not None
        else DEFAULT_CONFIG
    )
    return TaskConfig(
        num_services=(
            base.num_services if args.num_services is None else args.num_services
        ),
        program_size_min_mb=base.program_size_min_mb,
        program_size_max_mb=base.program_size_max_mb,
        mean_input_size_min_mbit=base.mean_input_size_min_mbit,
        mean_input_size_max_mbit=base.mean_input_size_max_mbit,
        cycles_per_bit_min=base.cycles_per_bit_min,
        cycles_per_bit_max=base.cycles_per_bit_max,
        arrival_probability=(
            base.arrival_probability
            if args.arrival_probability is None
            else args.arrival_probability
        ),
        zipf_exponent=(
            base.zipf_exponent if args.zipf_exponent is None else args.zipf_exponent
        ),
        popularity_epoch_length=(
            base.popularity_epoch_length
            if args.popularity_epoch_length is None
            else args.popularity_epoch_length
        ),
        popularity_changes_per_epoch=(
            base.popularity_changes_per_epoch
            if args.popularity_changes_per_epoch is None
            else args.popularity_changes_per_epoch
        ),
        popularity_history_length=(
            base.popularity_history_length
            if args.popularity_history_length is None
            else args.popularity_history_length
        ),
        input_relative_jitter=(
            base.input_relative_jitter
            if args.input_relative_jitter is None
            else args.input_relative_jitter
        ),
        load_relative_jitter=(
            base.load_relative_jitter
            if args.load_relative_jitter is None
            else args.load_relative_jitter
        ),
        fallback_dt_s=base.fallback_dt_s,
        fallback_warmup_time_s=base.fallback_warmup_time_s,
        fallback_episode_time_s=base.fallback_episode_time_s,
        fallback_master_seed=base.fallback_master_seed,
        rng_stream_names=base.rng_stream_names,
        program_size_unit=base.program_size_unit,
        input_size_unit=base.input_size_unit,
        cpu_cycles_unit=base.cpu_cycles_unit,
        cycles_per_bit_unit=base.cycles_per_bit_unit,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        project_root = find_project_root()
        from common import project_config as project_config_module

        formal_config = project_config_module.load_config()
        config = config_from_args(args, formal_config)
        roles_file = resolve_path(
            project_root,
            args.roles_file or "data/generated/vehicle_roles.json",
        )
        if args.roles_file is None:
            roles_file = project_config_module.resolve_path(
                project_root,
                project_config_module.get_config_value(
                    formal_config, "paths.vehicle_roles"
                ),
            )
        trajectory_file = resolve_path(
            project_root,
            args.trajectory_file
            or project_config_module.get_config_value(
                formal_config,
                "paths.trajectory",
                "data/generated/trajectory.csv",
            ),
        )
        output_dir = resolve_path(
            project_root,
            args.output_dir
            or Path(
                project_config_module.get_config_value(
                    formal_config,
                    "paths.service_catalog",
                    "data/generated/service_catalog.json",
                )
            ).parent,
        )

        roles, task_vehicles, service_vehicles = load_roles_and_tv(roles_file)
        expected_tv = int(
            project_config_module.get_config_value(
                formal_config,
                "simulation.num_task_vehicles",
            )
        )
        expected_sv = int(
            project_config_module.get_config_value(
                formal_config,
                "simulation.num_service_vehicles",
            )
        )
        if len(task_vehicles) != expected_tv or len(service_vehicles) != expected_sv:
            raise ValueError(
                "vehicle_roles.json 的 TV/SV 数量与 system_config.toml 不一致。"
            )
        config.validate(len(task_vehicles))
        dt_s, warmup_s, episode_s, master_seed = resolve_time_metadata(
            roles,
            args,
            config,
            formal_config,
        )
        num_slots, _ = read_formal_trajectory_alignment(
            trajectory_file,
            task_vehicles,
            dt_s,
            warmup_s,
            episode_s,
        )
        declared_slots = roles.get("formal_time_slot_count")
        if declared_slots is not None and int(declared_slots) != num_slots:
            raise ValueError(
                "角色元数据 formal_time_slot_count 与 dt/warmup/episode 不一致。"
            )

        artifacts = build_artifacts(
            task_vehicles,
            dt_s,
            warmup_s,
            episode_s,
            master_seed,
            config,
        )
        validation = validate_artifacts(
            artifacts,
            task_vehicles,
            config,
            config.arrival_probability,
        )

        # 不重新写文件，只在内存中重复生成一次，验证所有独立 RNG 的完全复现。
        repeated = build_artifacts(
            task_vehicles,
            dt_s,
            warmup_s,
            episode_s,
            master_seed,
            config,
        )
        if json_canonical(artifacts["catalog"]) != json_canonical(repeated["catalog"]):
            raise AssertionError("相同 seed 重复生成的 service catalog 不一致。")
        if json_canonical(artifacts["popularity_rows"]) != json_canonical(
            repeated["popularity_rows"]
        ):
            raise AssertionError("相同 seed 重复生成的 popularity 不一致。")
        if json_canonical(artifacts["task_rows"]) != json_canonical(repeated["task_rows"]):
            raise AssertionError("相同 seed 重复生成的 tasks 不一致。")
        if json_canonical(artifacts["popularity_history_rows"]) != json_canonical(
            repeated["popularity_history_rows"]
        ):
            raise AssertionError("相同 seed 重复生成的 popularity history 不一致。")

        rngs, child_seeds = make_rngs(master_seed, config.rng_stream_names)
        del rngs
        catalog_output = make_service_catalog_output(
            artifacts,
            config,
            dt_s,
            warmup_s,
            episode_s,
            master_seed,
            child_seeds,
            task_vehicles,
            validation,
        )
        local_cache_capacity_mb = float(
            project_config_module.get_config_value(
                formal_config,
                "nodes.vehicle.local_cache_capacity_mb",
            )
        )
        tv_local_cache_path = project_config_module.resolve_path(
            project_root,
            project_config_module.get_config_value(
                formal_config,
                "paths.tv_local_cache",
            ),
        )
        tv_local_cache = generate_tv_local_cache(
            task_vehicles,
            catalog_output,
            local_cache_capacity_mb,
            master_seed,
        )
        tv_local_cache_validation = validate_tv_local_cache(
            tv_local_cache,
            task_vehicles,
            catalog_output,
            local_cache_capacity_mb,
        )
        write_json_atomic(tv_local_cache_path, tv_local_cache)
        files = output_artifacts(
            output_dir,
            catalog_output,
            artifacts["popularity_history_rows"],
            artifacts["popularity_rows"],
            artifacts["task_rows"],
        )

        print("任务环境生成完成：")
        print(f"  TV={len(task_vehicles)}，SV={len(service_vehicles)}（SV 不生成任务）")
        print(f"  F={config.num_services}，formal slots={num_slots}")
        print(f"  popularity prehistory slots={artifacts['history_slots']}")
        print(f"  formal rho_star epoch 数={validation['epoch_count']}")
        print(f"  tasks rows={validation['task_rows']}，arrivals={validation['total_arrivals']}")
        print(f"  empirical lambda={validation['empirical_arrival_rate']:.6f}")
        print(
            "  rho_star normalized=True，"
            f"same-seed reproducible=True，"
            f"adjacent-same={validation['adjacent_same_probability_fraction']:.6f}"
        )
        print("  TV local cache generated")
        print(f"  TV count = {tv_local_cache_validation['tv_count']}")
        print(
            "  one service per TV = "
            f"{tv_local_cache_validation['all_tv_exactly_once']}"
        )
        print(
            "  fixed over episode = "
            f"{tv_local_cache['metadata']['fixed_over_episode']}"
        )
        print(f"  output: {tv_local_cache_path}")
        for path in files:
            print(f"  output: {path}")
        return 0
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
