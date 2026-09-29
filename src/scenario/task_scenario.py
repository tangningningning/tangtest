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

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scenario.realization import derive_episode_seeds
except ModuleNotFoundError:  # direct ``python src/scenario/task_scenario.py``
    from realization import derive_episode_seeds


# ============================================================================
# 本阶段集中配置区
# ============================================================================


@dataclass(frozen=True)
class TaskConfig:
    """任务环境生成参数。

    数值默认值只用于无 TOML 场景下的函数级验证；正式 entrypoint 从
    configs/proposed/system_config.toml 读取参数，vehicle_roles.json 仅用于检查上游时间轴漂移。
    """

    num_services: int = 10
    arrival_probability: float = 0.7
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

    # 服务大小以 GB 配置，生成后同时保存 GB 展示值和运行时所需的 MB 值。
    service_size_min_gb: float = 2.0
    service_size_max_gb: float = 20.0
    # 任务输入范围以 KB 配置，目录和任务行仍保存 bit，避免改变既有计算公式接口。
    input_size_min_kb: float = 500.0
    input_size_max_kb: float = 5000.0
    cycles_per_bit_min: float = 400.0
    cycles_per_bit_max: float = 1000.0

    # 实现层回退默认值：参数范围由 configs/proposed/system_config.toml
    # 正式入口统一读取；以下值只用于无 TOML 的函数级回退。
    service_size_unit: str = "GB"
    program_size_unit: str = "MB"
    input_size_range_unit: str = "KB"
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
            ("service_size", self.service_size_min_gb, self.service_size_max_gb),
            ("input_size", self.input_size_min_kb, self.input_size_max_kb),
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

# 服务缓存容量以 MB 参与约束；这里使用十进制单位完成 GB 到 MB 的配置换算。
SERVICE_SIZE_MB_PER_GB = 1000.0
# 任务输入以 bit 进入既有计算公式；这里使用十进制 KB 完成单位换算。
INPUT_SIZE_BITS_PER_KB = 8000.0

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








# ============================================================================
# 独立随机流和服务目录
# ============================================================================




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
    ``mean_input_size`` 在 catalog 中是 bit，因此校验时先还原为 KB。
    """
    by_id = catalog_by_id(catalog)
    dimensions = (
        (
            "service_size_gb",
            "service_size_gb",
            config.service_size_min_gb,
            config.service_size_max_gb,
            1.0,
        ),
        (
            "input_size_kb",
            "mean_input_size",
            config.input_size_min_kb,
            config.input_size_max_kb,
            1.0 / INPUT_SIZE_BITS_PER_KB,
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
        "service_size_vs_input_size": safe_correlation(
            parameter_values["service_size_gb"],
            parameter_values["input_size_kb"],
        ),
        "service_size_vs_cycles_per_bit": safe_correlation(
            parameter_values["service_size_gb"],
            parameter_values["cycles_per_bit"],
        ),
        "input_size_vs_cycles_per_bit": safe_correlation(
            parameter_values["input_size_kb"],
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
    """检查 TV 容量可容纳的完整服务集合及其单服务可行性。"""

    if local_cache_capacity_mb <= 0.0:
        raise ValueError("TV local cache capacity 必须大于 0。")
    ordered = sorted(service_sizes_mb.items(), key=lambda item: (item[1], item[0]))
    if len(ordered) < 2:
        raise ValueError("至少需要两个 service 才能验证 min_two_service_sum。")
    max_single = max(service_sizes_mb.values())
    min_two = ordered[0][1] + ordered[1][1]
    fitting_service_ids = [
        service_id
        for service_id, size_mb in ordered
        if size_mb <= local_cache_capacity_mb + 1.0e-9
    ]
    if not fitting_service_ids:
        raise ValueError(
            "TV local cache capacity 至少应能容纳一个完整 service。"
        )
    return {
        "service_count": len(service_sizes_mb),
        "min_service": {"service_id": ordered[0][0], "size_mb": ordered[0][1]},
        "second_min_service": {"service_id": ordered[1][0], "size_mb": ordered[1][1]},
        "min_service_size_mb": ordered[0][1],
        "max_service_size_mb": max_single,
        "min_two_service_sum_mb": min_two,
        "total_service_size_mb": float(sum(service_sizes_mb.values())),
        "single_services_fit_count": len(fitting_service_ids),
        "single_service_ids_fit": fitting_service_ids,
        "all_services_fit": len(fitting_service_ids) == len(service_sizes_mb),
        "all_pairs_exceed_capacity": bool(local_cache_capacity_mb < min_two),
    }


def generate_tv_local_cache(
    task_vehicle_ids: list[str],
    service_catalog: dict[str, Any],
    local_cache_capacity_mb: float,
    seed: int,
) -> dict[str, Any]:
    """为每个TV生成实验前固定的本地缓存初始状态。

    生成规则为对“能够完整放入 TV 容量”的服务均匀执行 one-service-per-TV，
    不读取 tasks.csv/popularity/future request，因此不泄漏未来 ground truth。
    相同 TV 集合、service catalog 和 seed 会得到完全相同 artifact。
    """

    service_sizes = service_size_by_id(service_catalog)
    sanity = service_size_sanity(service_sizes, local_cache_capacity_mb)
    tv_ids = sorted([str(vehicle_id) for vehicle_id in task_vehicle_ids], key=_stable_id_key)
    if len(tv_ids) != len(set(tv_ids)):
        raise ValueError("task_vehicle_ids 存在重复。")
    service_ids = sorted(service_sizes)
    eligible_service_ids = list(sanity["single_service_ids_fit"])
    rng = np.random.default_rng(int(seed))
    vehicles: dict[str, dict[str, Any]] = {}
    for tv_id in tv_ids:
        service_id = int(rng.choice(eligible_service_ids))
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
            "generation_rule": "uniform_one_service_per_tv_among_fitting_services",
            "service_count": len(service_ids),
            "service_ids": service_ids,
            "eligible_service_ids": eligible_service_ids,
            "tv_count": len(tv_ids),
            "tv_ids": tv_ids,
            "fixed_over_episode": True,
            "uses_future_requests": False,
            "parameter_basis": "system-scale parameters derived from current service catalog",
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
    valid_services = set(sanity["single_service_ids_fit"])
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
        "service_count": len(service_sizes),
        "eligible_service_count": len(valid_services),
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




def build_episode_artifacts(
    task_vehicles: list[str],
    service_catalog: dict[str, Any],
    dt_s: float,
    warmup_s: float,
    episode_s: float,
    config: TaskConfig,
    named_seeds: dict[str, int],
) -> dict[str, Any]:
    """Generate one episode's task data from explicit named streams.

    The service catalog is read-only input.  Popularity prehistory and the formal
    episode are generated as one continuous process.
    """

    required = {
        "popularity_evolution",
        "task_arrival",
        "service_request",
        "workload_variation",
    }
    missing = sorted(required - set(named_seeds))
    if missing:
        raise ValueError("missing named task seed streams: " + ", ".join(missing))
    catalog = service_catalog.get("services")
    if not isinstance(catalog, list):
        raise ValueError("service_catalog.json 缺少 services list。")
    catalog = [dict(row) for row in catalog]
    if len(catalog) != config.num_services:
        raise ValueError("service catalog 的 service_count 与配置不一致。")

    num_slots = int(round(episode_s / dt_s))
    history_slots = int(config.popularity_history_length)
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
        np.random.default_rng(int(named_seeds["popularity_evolution"])),
    )
    combined_task_rows, combined_request_counts = generate_tasks(
        task_vehicles,
        catalog,
        combined_probabilities,
        history_slots + num_slots,
        dt_s,
        warmup_s - history_slots * dt_s,
        config,
        np.random.default_rng(int(named_seeds["task_arrival"])),
        np.random.default_rng(int(named_seeds["service_request"])),
        np.random.default_rng(int(named_seeds["workload_variation"])),
        slot_offset=-history_slots,
    )
    task_rows = [row for row in combined_task_rows if int(row["slot"]) >= 0]
    history_request_counts = combined_request_counts[:history_slots].copy()
    request_counts = combined_request_counts[history_slots:].copy()
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
        "child_seeds": {name: int(named_seeds[name]) for name in sorted(required)},
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




def output_realization_artifacts(
    output_dir: Path,
    popularity_history_rows: list[dict[str, Any]],
    popularity_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    """Write only realization-varying outputs; the catalog remains shared."""

    output_dir.mkdir(parents=True, exist_ok=True)
    popularity_history_file = output_dir / "popularity_history.csv"
    popularity_file = output_dir / "popularity.csv"
    tasks_file = output_dir / "tasks.csv"
    write_csv_atomic(
        popularity_history_file,
        POPULARITY_HISTORY_FIELDS,
        popularity_history_rows,
    )
    write_csv_atomic(popularity_file, POPULARITY_FIELDS, popularity_rows)
    write_csv_atomic(tasks_file, TASK_FIELDS, task_rows)
    return popularity_history_file, popularity_file, tasks_file


def task_config_from_project_config(
    project_config: dict[str, Any],
    fallback: TaskConfig | None = None,
) -> TaskConfig:
    """把 configs/proposed/system_config.toml 的正式任务参数映射到生成器配置。"""
    from common.project_config import get_config_value

    base = fallback or DEFAULT_CONFIG
    return TaskConfig(
        num_services=int(
            get_config_value(project_config, "service.count", base.num_services)
        ),
        service_size_min_gb=float(
            get_config_value(
                project_config,
                "service.generation.service_size_min_gb",
                base.service_size_min_gb,
            )
        ),
        service_size_max_gb=float(
            get_config_value(
                project_config,
                "service.generation.service_size_max_gb",
                base.service_size_max_gb,
            )
        ),
        input_size_min_kb=float(
            get_config_value(
                project_config,
                "service.generation.input_size_min_kb",
                base.input_size_min_kb,
            )
        ),
        input_size_max_kb=float(
            get_config_value(
                project_config,
                "service.generation.input_size_max_kb",
                base.input_size_max_kb,
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
        service_size_unit=base.service_size_unit,
        program_size_unit=base.program_size_unit,
        input_size_range_unit=base.input_size_range_unit,
        input_size_unit=base.input_size_unit,
        cpu_cycles_unit=base.cpu_cycles_unit,
        cycles_per_bit_unit=base.cycles_per_bit_unit,
    )


# ============================================================================
# CLI
# ============================================================================
