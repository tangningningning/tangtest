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

    数值默认值用于实现和可复现验证，均不是论文最终冻结参数。已有
    vehicle_roles.json 中的 dt/warmup/episode/seed 会优先覆盖时间和主 seed。
    """

    num_services: int = 10
    arrival_probability: float = 0.5
    zipf_exponent: float = 0.8
    popularity_epoch_length: int = 20
    popularity_changes_per_epoch: int = 1
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

    # implementation provisional defaults：参考现有项目历史配置的量纲，
    # 具体区间仍需结合论文和文献统一校准。
    program_size_unit: str = "MB"
    input_size_unit: str = "bit"
    cpu_cycles_unit: str = "cycle"
    cycles_per_bit_unit: str = "cycle/bit"

    # 不同 workload_class 的范围只用于构造异质服务标签，不是系统决策变量。
    class_ranges: tuple[tuple[str, float, float, float, float, float, float], ...] = (
        # class, program_min_MB, program_max_MB, D_min_Mbit, D_max_Mbit,
        # cycles_per_bit_min, cycles_per_bit_max
        ("data_intensive", 1200.0, 2000.0, 1.00, 1.50, 200.0, 500.0),
        ("compute_intensive", 1000.0, 1800.0, 0.50, 1.00, 700.0, 1000.0),
        ("balanced", 1000.0, 1900.0, 0.75, 1.25, 450.0, 750.0),
    )

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
        if not 0.0 <= self.input_relative_jitter < 1.0:
            raise ValueError("input_relative_jitter 必须位于 [0, 1)。")
        if not 0.0 <= self.load_relative_jitter < 1.0:
            raise ValueError("load_relative_jitter 必须位于 [0, 1)。")
        if num_tv <= 0:
            raise ValueError("task_vehicles 不能为空。")
        if not self.class_ranges:
            raise ValueError("至少需要一个 workload_class 参数范围。")


DEFAULT_CONFIG = TaskConfig()

SERVICE_CATALOG_FIELDS = (
    "service_id",
    "program_size",
    "mean_input_size",
    "cycles_per_bit",
    "workload_class",
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

TASK_FIELDS = (
    "time_s",
    "slot",
    "episode_time_s",
    "vehicle_id",
    "arrival",
    "service_id",
    "input_size",
    "cpu_cycles",
    "workload_class",
)


# ============================================================================
# 输入文件和时间轴
# ============================================================================


def find_project_root() -> Path:
    candidates = (
        Path(__file__).resolve().parents[1],
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
) -> tuple[float, float, float, int]:
    """优先读取角色文件中的时间元数据，CLI 显式值具有最高优先级。"""
    dt_s = float(
        args.dt
        if args.dt is not None
        else roles.get("dt", defaults.fallback_dt_s)
    )
    warmup_s = float(
        args.warmup_time
        if args.warmup_time is not None
        else roles.get("warmup_time", defaults.fallback_warmup_time_s)
    )
    episode_s = float(
        args.episode_time
        if args.episode_time is not None
        else roles.get("episode_time", defaults.fallback_episode_time_s)
    )
    master_seed = int(
        args.master_seed
        if args.master_seed is not None
        else roles.get("seed", defaults.fallback_master_seed)
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


def make_service_catalog(
    config: TaskConfig,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """生成服务异质性参数；workload_class 只是数据标签。"""
    catalog: list[dict[str, Any]] = []
    for service_id in range(config.num_services):
        class_name, p_min, p_max, d_min, d_max, c_min, c_max = config.class_ranges[
            service_id % len(config.class_ranges)
        ]
        catalog.append(
            {
                "service_id": service_id,
                "program_size": float(rng.uniform(p_min, p_max)),
                "mean_input_size": float(
                    rng.uniform(d_min, d_max) * 1_000_000.0
                ),
                "cycles_per_bit": float(rng.uniform(c_min, c_max)),
                "workload_class": class_name,
            }
        )
    return catalog


def catalog_by_id(catalog: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result = {int(row["service_id"]): row for row in catalog}
    if set(result) != set(range(len(catalog))):
        raise ValueError("service_id 必须覆盖 0 到 F-1。")
    return result


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
    """生成按 epoch 稳定、边界缓慢变化的 rho_star。"""
    ranking = [int(value) for value in rng.permutation(num_services)]
    probabilities_by_slot = np.zeros((num_slots, num_services), dtype=float)
    rows: list[dict[str, Any]] = []
    schedule_changes: list[dict[str, Any]] = []
    rankings_by_epoch: list[list[int]] = []

    for slot in range(num_slots):
        epoch = slot // config.popularity_epoch_length
        if slot % config.popularity_epoch_length == 0:
            previous = list(ranking)
            if slot > 0:
                ranking = evolve_ranking(
                    ranking,
                    rng,
                    config.popularity_changes_per_epoch,
                )
                schedule_changes.append(
                    {
                        "epoch": epoch,
                        "slot": slot,
                        "previous_ranking": previous,
                        "new_ranking": list(ranking),
                    }
                )
            rankings_by_epoch.append(list(ranking))

        probabilities = zipf_probabilities(
            ranking,
            num_services,
            config.zipf_exponent,
        )
        probabilities_by_slot[slot] = probabilities
        rank_by_service = {service_id: rank for rank, service_id in enumerate(ranking, 1)}
        time_s = round(time_origin_s + slot * dt_s, 9)
        for service_id in range(num_services):
            rows.append(
                {
                    "time_s": time_s,
                    "slot": slot,
                    "epoch": epoch,
                    "service_id": service_id,
                    "rank": rank_by_service[service_id],
                    "rho_star": float(probabilities[service_id]),
                    "request_count": 0,
                    "rho_obs": "",
                }
            )

    schedule = {
        "initial_ranking": rankings_by_epoch[0] if rankings_by_epoch else [],
        "epoch_changes": schedule_changes,
        "epoch_count": len(rankings_by_epoch),
    }
    return rows, probabilities_by_slot, schedule


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
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """按 TV×slot 生成一行状态；arrival=0 不构造伪任务。"""
    by_service = catalog_by_id(catalog)
    request_counts = np.zeros((num_slots, config.num_services), dtype=int)
    rows: list[dict[str, Any]] = []

    for slot in range(num_slots):
        probabilities = probabilities_by_slot[slot]
        arrivals = arrival_rng.random(len(task_vehicles)) < config.arrival_probability
        for vehicle_index, vehicle_id in enumerate(task_vehicles):
            arrival = int(bool(arrivals[vehicle_index]))
            time_s = round(warmup_s + slot * dt_s, 9)
            episode_time_s = round(slot * dt_s, 9)

            if arrival == 0:
                rows.append(
                    {
                        "time_s": time_s,
                        "slot": slot,
                        "episode_time_s": episode_time_s,
                        "vehicle_id": vehicle_id,
                        "arrival": 0,
                        "service_id": "",
                        "input_size": 0.0,
                        "cpu_cycles": 0.0,
                        "workload_class": "",
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
                    "slot": slot,
                    "episode_time_s": episode_time_s,
                    "vehicle_id": vehicle_id,
                    "arrival": 1,
                    "service_id": service_id,
                    "input_size": input_size,
                    "cpu_cycles": cpu_cycles,
                    "workload_class": service["workload_class"],
                }
            )

    expected_rows = len(task_vehicles) * num_slots
    if len(rows) != expected_rows:
        raise AssertionError(
            f"tasks.csv 行数错误：{len(rows)}，期望 {expected_rows}。"
        )
    return rows, request_counts


def attach_observed_counts(
    popularity_rows: list[dict[str, Any]],
    request_counts: np.ndarray,
) -> None:
    """保存 request_count/rho_obs，但不修改 rho_star。"""
    num_services = request_counts.shape[1]
    for row in popularity_rows:
        slot = int(row["slot"])
        service_id = int(row["service_id"])
        count = int(request_counts[slot, service_id])
        total = int(request_counts[slot].sum())
        row["request_count"] = count
        row["rho_obs"] = float(count / total) if total > 0 else ""
    expected_rows = request_counts.shape[0] * num_services
    if len(popularity_rows) != expected_rows:
        raise AssertionError("popularity 行数与 slot×service 数量不一致。")


# ============================================================================
# 校验、输出和可复现性
# ============================================================================


def json_canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_artifacts(
    task_vehicles: list[str],
    dt_s: float,
    warmup_s: float,
    episode_s: float,
    master_seed: int,
    config: TaskConfig,
) -> dict[str, Any]:
    num_slots = int(round(episode_s / dt_s))
    rngs, child_seeds = make_rngs(master_seed, config.rng_stream_names)
    catalog = make_service_catalog(config, rngs["service_catalog"])
    popularity_rows, probabilities, schedule = make_popularity_trace(
        config.num_services,
        num_slots,
        dt_s,
        warmup_s,
        config,
        rngs["popularity_evolution"],
    )
    task_rows, request_counts = generate_tasks(
        task_vehicles,
        catalog,
        probabilities,
        num_slots,
        dt_s,
        warmup_s,
        config,
        rngs["task_arrival"],
        rngs["service_request"],
        rngs["workload_variation"],
    )
    attach_observed_counts(popularity_rows, request_counts)

    return {
        "catalog": catalog,
        "popularity_rows": popularity_rows,
        "task_rows": task_rows,
        "probabilities": probabilities,
        "request_counts": request_counts,
        "schedule": schedule,
        "child_seeds": child_seeds,
        "num_slots": num_slots,
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
    schedule = artifacts["schedule"]
    num_slots = int(artifacts["num_slots"])

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

    if len(popularity_rows) != num_slots * config.num_services:
        raise AssertionError("popularity.csv 行数错误。")
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
        "epoch_count": int(schedule["epoch_count"]),
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
            "arrival_probability": config.arrival_probability,
            "zipf_exponent": config.zipf_exponent,
            "popularity_epoch_length": config.popularity_epoch_length,
            "popularity_changes_per_epoch": config.popularity_changes_per_epoch,
            "input_relative_jitter": config.input_relative_jitter,
            "load_relative_jitter": config.load_relative_jitter,
            "class_ranges": [
                {
                    "workload_class": row[0],
                    "program_size_min_MB": row[1],
                    "program_size_max_MB": row[2],
                    "mean_input_size_min_Mbit": row[3],
                    "mean_input_size_max_Mbit": row[4],
                    "cycles_per_bit_min": row[5],
                    "cycles_per_bit_max": row[6],
                }
                for row in config.class_ranges
            ],
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
    popularity_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_file = output_dir / "service_catalog.json"
    popularity_file = output_dir / "popularity.csv"
    tasks_file = output_dir / "tasks.csv"
    write_json_atomic(catalog_file, catalog_output)
    write_csv_atomic(popularity_file, POPULARITY_FIELDS, popularity_rows)
    write_csv_atomic(tasks_file, TASK_FIELDS, task_rows)
    return catalog_file, popularity_file, tasks_file


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成真实动态 rho_star、TV 任务到达和服务相关 D/L。"
    )
    parser.add_argument("--roles-file", default="data/generated/vehicle_roles.json")
    parser.add_argument("--trajectory-file", default="data/generated/trajectory.csv")
    parser.add_argument("--output-dir", default="data/generated")
    parser.add_argument("--num-services", type=int, default=DEFAULT_CONFIG.num_services)
    parser.add_argument(
        "--arrival-probability",
        type=float,
        default=DEFAULT_CONFIG.arrival_probability,
    )
    parser.add_argument("--zipf-exponent", type=float, default=DEFAULT_CONFIG.zipf_exponent)
    parser.add_argument(
        "--popularity-epoch-length",
        type=int,
        default=DEFAULT_CONFIG.popularity_epoch_length,
    )
    parser.add_argument(
        "--popularity-changes-per-epoch",
        type=int,
        default=DEFAULT_CONFIG.popularity_changes_per_epoch,
    )
    parser.add_argument(
        "--input-relative-jitter",
        type=float,
        default=DEFAULT_CONFIG.input_relative_jitter,
    )
    parser.add_argument(
        "--load-relative-jitter",
        type=float,
        default=DEFAULT_CONFIG.load_relative_jitter,
    )
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--warmup-time", type=float, default=None)
    parser.add_argument("--episode-time", type=float, default=None)
    parser.add_argument("--master-seed", type=int, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> TaskConfig:
    return TaskConfig(
        num_services=args.num_services,
        arrival_probability=args.arrival_probability,
        zipf_exponent=args.zipf_exponent,
        popularity_epoch_length=args.popularity_epoch_length,
        popularity_changes_per_epoch=args.popularity_changes_per_epoch,
        input_relative_jitter=args.input_relative_jitter,
        load_relative_jitter=args.load_relative_jitter,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        project_root = find_project_root()
        config = config_from_args(args)
        roles_file = resolve_path(project_root, args.roles_file)
        trajectory_file = resolve_path(project_root, args.trajectory_file)
        output_dir = resolve_path(project_root, args.output_dir)

        roles, task_vehicles, service_vehicles = load_roles_and_tv(roles_file)
        config.validate(len(task_vehicles))
        dt_s, warmup_s, episode_s, master_seed = resolve_time_metadata(
            roles,
            args,
            config,
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
        files = output_artifacts(
            output_dir,
            catalog_output,
            artifacts["popularity_rows"],
            artifacts["task_rows"],
        )

        print("任务环境生成完成：")
        print(f"  TV={len(task_vehicles)}，SV={len(service_vehicles)}（SV 不生成任务）")
        print(f"  F={config.num_services}，formal slots={num_slots}")
        print(f"  rho_star epoch 数={validation['epoch_count']}")
        print(f"  tasks rows={validation['task_rows']}，arrivals={validation['total_arrivals']}")
        print(f"  empirical lambda={validation['empirical_arrival_rate']:.6f}")
        print(
            "  rho_star normalized=True，"
            f"same-seed reproducible=True，"
            f"adjacent-same={validation['adjacent_same_probability_fraction']:.6f}"
        )
        for path in files:
            print(f"  output: {path}")
        return 0
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
