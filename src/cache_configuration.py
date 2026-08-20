#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trigger-conditioned cache configuration optimization.

本阶段只解决：给定上一时隙缓存状态、cache trigger 和预测场景后，选择新的
动态缓存配置。这里不训练 PPO，不调用 Matching/KKT，也不读取任何未来真实
trajectory/tasks/popularity 标签。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import project_config
import system_model


# ============================================================================
# 集中配置区：均为 provisional defaults，不是论文最终参数
# ============================================================================


@dataclass(frozen=True)
class CacheOptimizationConfig:
    # 所有系统物理参数必须由 system_config.toml 构造，不在算法文件中另设一套。
    num_rsu: int
    num_uav: int
    rsu_instances: tuple[tuple[str, float, float], ...]
    uav_instances: tuple[tuple[str, float, float], ...]
    sv_capacity_mb: float
    rsu_capacity_mb: float
    uav_capacity_mb: float
    sv_eta_ca: float
    rsu_eta_ca: float
    uav_eta_ca: float
    sv_xi_ca: float
    rsu_xi_ca: float
    uav_xi_ca: float
    lambda_cache: float
    beta_discount: float
    omega_t: float
    omega_e: float
    time_ref_s: float
    energy_ref_j: float
    local_cpu_cycles_per_s: float
    sv_cpu_cycles_per_s: float
    rsu_cpu_cycles_per_s: float
    uav_cpu_cycles_per_s: float
    cloud_cpu_cycles_per_s: float
    local_energy_per_cycle_j: float
    tx_power_w: float
    sv_bandwidth_hz: float
    rsu_bandwidth_hz: float
    uav_bandwidth_hz: float
    noise_power_w: float
    channel_gain_at_1m: float
    pathloss_exponent: float
    uav_altitude_m: float
    air_eta1: float
    air_eta2: float
    air_beta_los: float
    air_beta_nlos: float
    air_alpha_los: float
    air_alpha_nlos: float
    cloud_uplink_rate_bps: float
    cloud_delay_factor: float
    sv_range_m: float
    rsu_range_m: float
    uav_range_m: float
    tolerance: float
    max_coordinate_rounds: int
    initial_cache_seed: int

    def validate(self) -> None:
        values = [
            self.num_rsu,
            self.num_uav,
            len(self.rsu_instances),
            len(self.uav_instances),
            self.sv_capacity_mb,
            self.rsu_capacity_mb,
            self.uav_capacity_mb,
            self.time_ref_s,
            self.energy_ref_j,
            self.local_cpu_cycles_per_s,
            self.sv_bandwidth_hz,
            self.rsu_bandwidth_hz,
            self.uav_bandwidth_hz,
            self.noise_power_w,
            self.max_coordinate_rounds,
        ]
        if any(float(value) <= 0.0 for value in values):
            raise ValueError("缓存优化配置存在非正数参数。")
        if not 0.0 < self.beta_discount <= 1.0:
            raise ValueError("beta_discount 必须位于 (0, 1]。")
        if self.tolerance < 0.0:
            raise ValueError("tolerance 不能为负数。")
        if self.num_rsu != len(self.rsu_instances):
            raise ValueError("RSU instance 数量必须等于 num_rsu。")
        if self.num_uav != len(self.uav_instances):
            raise ValueError("UAV instance 数量必须等于 num_uav。")


DEFAULT_CONFIG: CacheOptimizationConfig | None = None


def cache_config_from_project(config: dict[str, Any]) -> CacheOptimizationConfig:
    """从 system_config.toml 的解析结果构造 cache optimizer 内部配置。"""
    get = project_config.get_config_value
    rsu_instances = tuple(
        (
            str(item["id"]),
            float(item["x_m"]),
            float(item["y_m"]),
        )
        for item in get(config, "nodes.rsu.instances", [])
    )
    uav_instances = tuple(
        (
            str(item["id"]),
            float(item["initial_x_m"]),
            float(item["initial_y_m"]),
        )
        for item in get(config, "nodes.uav.instances", [])
    )
    return CacheOptimizationConfig(
        num_rsu=int(get(config, "nodes.rsu.count")),
        num_uav=int(get(config, "nodes.uav.count")),
        rsu_instances=rsu_instances,
        uav_instances=uav_instances,
        sv_capacity_mb=float(get(config, "nodes.sv.cache_capacity_mb")),
        rsu_capacity_mb=float(get(config, "nodes.rsu.cache_capacity_mb")),
        uav_capacity_mb=float(get(config, "nodes.uav.cache_capacity_mb")),
        sv_eta_ca=float(get(config, "nodes.sv.eta_ca")),
        rsu_eta_ca=float(get(config, "nodes.rsu.eta_ca")),
        uav_eta_ca=float(get(config, "nodes.uav.eta_ca")),
        sv_xi_ca=float(get(config, "nodes.sv.xi_ca")),
        rsu_xi_ca=float(get(config, "nodes.rsu.xi_ca")),
        uav_xi_ca=float(get(config, "nodes.uav.xi_ca")),
        lambda_cache=float(get(config, "objective.lambda_cache")),
        beta_discount=float(get(config, "cache.discount_beta")),
        omega_t=float(get(config, "objective.omega_delay")),
        omega_e=float(get(config, "objective.omega_energy")),
        time_ref_s=float(get(config, "objective.T_ref")),
        energy_ref_j=float(get(config, "objective.E_ref")),
        local_cpu_cycles_per_s=float(get(config, "nodes.vehicle.local_cpu_cycles_per_s")),
        sv_cpu_cycles_per_s=float(get(config, "nodes.sv.cpu_cycles_per_s")),
        rsu_cpu_cycles_per_s=float(get(config, "nodes.rsu.cpu_cycles_per_s")),
        uav_cpu_cycles_per_s=float(get(config, "nodes.uav.cpu_cycles_per_s")),
        cloud_cpu_cycles_per_s=float(get(config, "nodes.cloud.cpu_cycles_per_s")),
        local_energy_per_cycle_j=float(get(config, "nodes.vehicle.kappa")),
        tx_power_w=float(get(config, "nodes.vehicle.transmit_power_w")),
        sv_bandwidth_hz=float(get(config, "nodes.sv.bandwidth_hz")),
        rsu_bandwidth_hz=float(get(config, "nodes.rsu.bandwidth_hz")),
        uav_bandwidth_hz=float(get(config, "nodes.uav.bandwidth_hz")),
        noise_power_w=float(get(config, "communication.noise.noise_power_w")),
        channel_gain_at_1m=float(get(config, "communication.ground.channel_gain_at_1m")),
        pathloss_exponent=float(get(config, "communication.ground.pathloss_exponent")),
        uav_altitude_m=float(get(config, "nodes.uav.altitude_m")),
        air_eta1=float(get(config, "communication.air.eta1")),
        air_eta2=float(get(config, "communication.air.eta2")),
        air_beta_los=float(get(config, "communication.air.beta_los")),
        air_beta_nlos=float(get(config, "communication.air.beta_nlos")),
        air_alpha_los=float(get(config, "communication.air.alpha_los")),
        air_alpha_nlos=float(get(config, "communication.air.alpha_nlos")),
        cloud_uplink_rate_bps=float(get(config, "nodes.cloud.uplink_rate_bps")),
        cloud_delay_factor=float(get(config, "nodes.cloud.cloud_delay_factor")),
        sv_range_m=float(get(config, "nodes.sv.range_m")),
        rsu_range_m=float(get(config, "nodes.rsu.range_m")),
        uav_range_m=float(get(config, "nodes.uav.range_m")),
        tolerance=float(get(config, "cache.local_search_tolerance")),
        max_coordinate_rounds=int(get(config, "cache.max_coordinate_rounds")),
        initial_cache_seed=int(get(config, "experiment.seed")),
    )


# ============================================================================
# 文件读取与场景切片
# ============================================================================


def find_project_root() -> Path:
    candidates = (Path(__file__).resolve().parents[1], Path.cwd().resolve())
    for candidate in candidates:
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("无法找到 adaptive_cache_mec 项目根目录。")


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 根对象必须是 object：{path}")
    return value


def load_service_catalog(path: Path) -> dict[str, Any]:
    catalog = load_json(path)
    services = sorted(catalog["services"], key=lambda row: int(row["service_id"]))
    service_ids = [int(row["service_id"]) for row in services]
    if service_ids != list(range(len(services))):
        raise ValueError("service_catalog.json 的 service_id 必须从 0 连续编号。")
    return {
        "num_services": len(services),
        "service_ids": np.asarray(service_ids, dtype=np.int64),
        "service_sizes_mb": np.asarray(
            [float(row["program_size"]) for row in services],
            dtype=np.float64,
        ),
        "mean_input_size_bit": np.asarray(
            [float(row["mean_input_size"]) for row in services],
            dtype=np.float64,
        ),
        "cycles_per_bit": np.asarray(
            [float(row["cycles_per_bit"]) for row in services],
            dtype=np.float64,
        ),
        "raw": catalog,
    }


def load_prediction_scenarios(path: Path) -> dict[str, Any]:
    loaded = np.load(path, allow_pickle=False)
    try:
        result = {key: np.asarray(loaded[key]) for key in loaded.files if key != "metadata_json"}
        result["metadata"] = json.loads(str(loaded["metadata_json"].item()))
        return result
    finally:
        loaded.close()


def scenario_at_origin(scenarios: dict[str, Any], origin_index: int) -> dict[str, Any]:
    n_origin = int(scenarios["origin_slots"].shape[0])
    if origin_index < 0 or origin_index >= n_origin:
        raise IndexError("origin_index 越界。")
    return {
        "origin_slot": int(scenarios["origin_slots"][origin_index]),
        "origin_time_s": float(scenarios["origin_times_s"][origin_index]),
        "scenario_ids": scenarios["scenario_ids"],
        "horizons": scenarios["horizons"],
        "vehicle_ids": scenarios["vehicle_ids"].astype(str),
        "vehicle_roles": scenarios["vehicle_roles"].astype(str),
        "task_vehicle_ids": scenarios["task_vehicle_ids"].astype(str),
        "service_vehicle_ids": scenarios["service_vehicle_ids"].astype(str),
        "positions_xy": scenarios["positions_xy"][origin_index].astype(np.float64),
        "arrival": scenarios["arrival"][origin_index].astype(np.int8),
        "service_id": scenarios["service_id"][origin_index].astype(np.int64),
        "input_size": scenarios["input_size"][origin_index].astype(np.float64),
        "cpu_cycles": scenarios["cpu_cycles"][origin_index].astype(np.float64),
        "rho_hat": scenarios["rho_hat"][origin_index].astype(np.float64),
        "metadata": scenarios["metadata"],
    }


# ============================================================================
# cache nodes and provisional infrastructure positions
# ============================================================================


def build_cache_nodes(
    origin_scenario: dict[str, Any],
    config: CacheOptimizationConfig,
) -> dict[str, Any]:
    sv_ids = [str(value) for value in origin_scenario["service_vehicle_ids"].tolist()]
    node_ids = list(sv_ids)
    node_roles = ["SV"] * len(sv_ids)
    for instance in config.rsu_instances:
        node_ids.append(instance[0])
        node_roles.append("RSU")
    for instance in config.uav_instances:
        node_ids.append(instance[0])
        node_roles.append("UAV")

    capacities = []
    eta = []
    xi = []
    for role in node_roles:
        if role == "SV":
            capacities.append(config.sv_capacity_mb)
            eta.append(config.sv_eta_ca)
            xi.append(config.sv_xi_ca)
        elif role == "RSU":
            capacities.append(config.rsu_capacity_mb)
            eta.append(config.rsu_eta_ca)
            xi.append(config.rsu_xi_ca)
        elif role == "UAV":
            capacities.append(config.uav_capacity_mb)
            eta.append(config.uav_eta_ca)
            xi.append(config.uav_xi_ca)
        else:
            raise AssertionError("未知 cache node role。")

    fixed_positions = {
        instance[0]: np.asarray([instance[1], instance[2]], dtype=np.float64)
        for instance in (*config.rsu_instances, *config.uav_instances)
    }

    return {
        "node_ids": node_ids,
        "node_roles": node_roles,
        "capacities_mb": np.asarray(capacities, dtype=np.float64),
        "eta_ca": np.asarray(eta, dtype=np.float64),
        "xi_ca": np.asarray(xi, dtype=np.float64),
        "fixed_positions": fixed_positions,
    }


def node_position(
    origin_scenario: dict[str, Any],
    cache_nodes: dict[str, Any],
    node_index: int,
    scenario_index: int,
    horizon: int,
) -> np.ndarray:
    role = cache_nodes["node_roles"][node_index]
    node_id = cache_nodes["node_ids"][node_index]
    if role == "SV":
        vehicle_ids = [str(value) for value in origin_scenario["vehicle_ids"].tolist()]
        vehicle_index = vehicle_ids.index(node_id)
        return origin_scenario["positions_xy"][scenario_index, horizon, vehicle_index]
    return cache_nodes["fixed_positions"][node_id]


def tv_position(
    origin_scenario: dict[str, Any],
    tv_id: str,
    scenario_index: int,
    horizon: int,
) -> np.ndarray:
    vehicle_ids = [str(value) for value in origin_scenario["vehicle_ids"].tolist()]
    vehicle_index = vehicle_ids.index(tv_id)
    return origin_scenario["positions_xy"][scenario_index, horizon, vehicle_index]


# ============================================================================
# surrogate objective
# ============================================================================


def role_range_m(role: str, config: CacheOptimizationConfig) -> float:
    if role == "SV":
        return config.sv_range_m
    if role == "RSU":
        return config.rsu_range_m
    if role == "UAV":
        return config.uav_range_m
    raise AssertionError("未知 role。")


def role_cpu_cycles_per_s(role: str, config: CacheOptimizationConfig) -> float:
    if role == "SV":
        return config.sv_cpu_cycles_per_s
    if role == "RSU":
        return config.rsu_cpu_cycles_per_s
    if role == "UAV":
        return config.uav_cpu_cycles_per_s
    raise AssertionError("未知 role。")


def role_bandwidth_hz(role: str, config: CacheOptimizationConfig) -> float:
    if role == "SV":
        return config.sv_bandwidth_hz
    if role == "RSU":
        return config.rsu_bandwidth_hz
    if role == "UAV":
        return config.uav_bandwidth_hz
    raise AssertionError("未知 role。")


def best_task_surrogate(
    cache: np.ndarray,
    origin_scenario: dict[str, Any],
    cache_nodes: dict[str, Any],
    service_id: int,
    input_bits: float,
    cpu_cycles: float,
    tv_id: str,
    scenario_index: int,
    horizon: int,
    config: CacheOptimizationConfig,
) -> float:
    # Local fallback：不需要缓存。
    local = system_model.local_execution(
        cpu_cycles,
        config.local_cpu_cycles_per_s,
        config.local_energy_per_cycle_j,
    )
    best = system_model.task_utility(
        local["delay_s"],
        local["vehicle_energy_j"],
        config.time_ref_s,
        config.energy_ref_j,
        config.omega_t,
        config.omega_e,
    )

    # Cloud fallback：不需要缓存。
    cloud = system_model.cloud_execution(
        input_bits,
        cpu_cycles,
        config.cloud_uplink_rate_bps,
        config.cloud_cpu_cycles_per_s,
        config.cloud_delay_factor,
        config.tx_power_w,
    )
    best = max(
        best,
        system_model.task_utility(
            cloud["total_delay_s"],
            cloud["vehicle_energy_j"],
            config.time_ref_s,
            config.energy_ref_j,
            config.omega_t,
            config.omega_e,
        ),
    )

    tv_xy = tv_position(origin_scenario, tv_id, scenario_index, horizon)
    for node_index, role in enumerate(cache_nodes["node_roles"]):
        if cache[node_index, service_id] == 0:
            continue
        node_xy = node_position(
            origin_scenario,
            cache_nodes,
            node_index,
            scenario_index,
            horizon,
        )
        distance = float(np.linalg.norm(tv_xy - node_xy))
        if distance > role_range_m(role, config):
            continue
        if role == "UAV":
            gain = system_model.air_channel_gain(
                distance,
                config.uav_altitude_m,
                config.air_eta1,
                config.air_eta2,
                config.air_beta_los,
                config.air_beta_nlos,
                config.air_alpha_los,
                config.air_alpha_nlos,
            )["channel_gain"]
        else:
            gain = system_model.ground_channel_gain(
                max(distance, 1e-9),
                config.channel_gain_at_1m,
                config.pathloss_exponent,
            )
        edge = system_model.edge_execution(
            input_bits,
            cpu_cycles,
            role_bandwidth_hz(role, config),
            bandwidth_fraction=1.0,
            transmit_power_w=config.tx_power_w,
            channel_gain=gain,
            noise_power_w=config.noise_power_w,
            node_cpu_cycles_per_s=role_cpu_cycles_per_s(role, config),
            cpu_fraction=1.0,
        )
        best = max(
            best,
            system_model.task_utility(
                edge["total_delay_s"],
                edge["vehicle_energy_j"],
                config.time_ref_s,
                config.energy_ref_j,
                config.omega_t,
                config.omega_e,
            ),
        )
    return float(best)


def predicted_service_utility(
    cache: np.ndarray,
    origin_scenario: dict[str, Any],
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
) -> float:
    arrival = origin_scenario["arrival"]
    service_id = origin_scenario["service_id"]
    input_size = origin_scenario["input_size"]
    cpu_cycles = origin_scenario["cpu_cycles"]
    tv_ids = [str(value) for value in origin_scenario["task_vehicle_ids"].tolist()]
    num_scenarios, horizon_steps, num_tv = arrival.shape
    total = 0.0
    for scenario_index in range(num_scenarios):
        for horizon in range(horizon_steps):
            discount = config.beta_discount ** horizon
            for tv_index in range(num_tv):
                if int(arrival[scenario_index, horizon, tv_index]) == 0:
                    continue
                service = int(service_id[scenario_index, horizon, tv_index])
                if service < 0:
                    raise ValueError("arrival=1 时 service_id 无效。")
                total += discount * best_task_surrogate(
                    cache,
                    origin_scenario,
                    cache_nodes,
                    service,
                    float(input_size[scenario_index, horizon, tv_index]),
                    float(cpu_cycles[scenario_index, horizon, tv_index]),
                    tv_ids[tv_index],
                    scenario_index,
                    horizon,
                    config,
                )
    return float(total / max(num_scenarios, 1))


def cache_update_energy(
    previous_cache: np.ndarray,
    new_cache: np.ndarray,
    service_sizes_mb: np.ndarray,
    cache_nodes: dict[str, Any],
) -> np.ndarray:
    result = system_model.cache_update_cost(
        previous_cache,
        new_cache,
        np.zeros(previous_cache.shape[0], dtype=np.int8),
        service_sizes_mb,
        cache_nodes["eta_ca"],
        cache_nodes["xi_ca"],
        lambda_ca=0.0,
    )
    return np.asarray(result["loading_energy"], dtype=np.float64)


def cache_update_cost(
    previous_cache: np.ndarray,
    new_cache: np.ndarray,
    cache_trigger: np.ndarray,
    service_sizes_mb: np.ndarray,
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
) -> tuple[float, np.ndarray]:
    result = system_model.cache_update_cost(
        previous_cache,
        new_cache,
        cache_trigger,
        service_sizes_mb,
        cache_nodes["eta_ca"],
        cache_nodes["xi_ca"],
        config.lambda_cache,
    )
    return float(result["total_cost"]), np.asarray(result["loading_energy"], dtype=np.float64)


def objective(
    cache: np.ndarray,
    previous_cache: np.ndarray,
    cache_trigger: np.ndarray,
    origin_scenario: dict[str, Any],
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
) -> float:
    utility = predicted_service_utility(cache, origin_scenario, cache_nodes, config)
    update_cost, _ = cache_update_cost(
        previous_cache,
        cache,
        cache_trigger,
        service_catalog["service_sizes_mb"],
        cache_nodes,
        config,
    )
    return float(utility - update_cost)


def evaluate_fixed_configuration(
    cache: np.ndarray,
    origin_scenario: dict[str, Any],
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
) -> dict[str, float]:
    """评估给定 cache + 节点位置下的预测 utility。

    这是给 Cache/UAV 配置模块共用的纯 evaluator：它不触发 cache local
    search，不修改 cache，也不读取未来真实 trajectory/tasks。UAV 配置优化只
    需要在不同 UAV fixed_positions 下复用同一套 surrogate 口径。
    """

    utility = predicted_service_utility(cache, origin_scenario, cache_nodes, config)
    if not math.isfinite(utility):
        raise AssertionError("fixed configuration predicted utility 非 finite。")
    return {"predicted_utility": float(utility)}


# ============================================================================
# capacity, initialization, and neighborhoods
# ============================================================================


def node_used_capacity_mb(node_cache: np.ndarray, service_sizes_mb: np.ndarray) -> float:
    return float(np.dot(node_cache.astype(np.float64), service_sizes_mb))


def is_node_feasible(
    node_cache: np.ndarray,
    service_sizes_mb: np.ndarray,
    capacity_mb: float,
) -> bool:
    return node_used_capacity_mb(node_cache, service_sizes_mb) <= capacity_mb + 1e-9


def validate_cache_shape_and_capacity(
    cache: np.ndarray,
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
) -> None:
    if cache.shape != (len(cache_nodes["node_ids"]), service_catalog["num_services"]):
        raise ValueError("cache shape 与 cache nodes/service count 不一致。")
    if not np.all(np.isin(cache, [0, 1])):
        raise ValueError("cache 必须是 0/1 矩阵。")
    for node_index in range(cache.shape[0]):
        if not is_node_feasible(
            cache[node_index],
            service_catalog["service_sizes_mb"],
            cache_nodes["capacities_mb"][node_index],
        ):
            raise ValueError(f"cache node {node_index} 容量约束违反。")


def service_values_from_scenarios(
    origin_scenario: dict[str, Any],
    config: CacheOptimizationConfig,
) -> np.ndarray:
    num_services = origin_scenario["rho_hat"].shape[-1]
    values = np.zeros(num_services, dtype=np.float64)
    arrival = origin_scenario["arrival"]
    service_id = origin_scenario["service_id"]
    input_size = origin_scenario["input_size"]
    cpu_cycles = origin_scenario["cpu_cycles"]
    num_scenarios, horizon_steps, num_tv = arrival.shape
    for scenario_index in range(num_scenarios):
        for horizon in range(horizon_steps):
            discount = config.beta_discount ** horizon
            for tv_index in range(num_tv):
                if int(arrival[scenario_index, horizon, tv_index]) == 0:
                    continue
                service = int(service_id[scenario_index, horizon, tv_index])
                values[service] += discount * (
                    1.0
                    + float(input_size[scenario_index, horizon, tv_index]) / 1.0e6
                    + float(cpu_cycles[scenario_index, horizon, tv_index]) / 1.0e9
                )
    return values / max(num_scenarios, 1)


def knapsack_initial_node(
    values: np.ndarray,
    sizes_mb: np.ndarray,
    capacity_mb: float,
) -> np.ndarray:
    """F=10 小规模 0-1 knapsack：只用于初始化，不作为最终 solver。"""
    num_services = len(values)
    best_value = -float("inf")
    best_mask = 0
    for mask in range(1 << num_services):
        chosen = np.asarray([(mask >> f) & 1 for f in range(num_services)], dtype=np.int8)
        used = node_used_capacity_mb(chosen, sizes_mb)
        if used > capacity_mb + 1e-9:
            continue
        value = float(np.dot(chosen, values))
        if value > best_value + 1e-12:
            best_value = value
            best_mask = mask
    return np.asarray([(best_mask >> f) & 1 for f in range(num_services)], dtype=np.int8)


def generate_node_neighbors(
    node_cache: np.ndarray,
    service_sizes_mb: np.ndarray,
    capacity_mb: float,
) -> list[tuple[str, np.ndarray]]:
    neighbors: list[tuple[str, np.ndarray]] = []
    num_services = len(node_cache)
    for service in range(num_services):
        if node_cache[service] == 0:
            candidate = node_cache.copy()
            candidate[service] = 1
            if is_node_feasible(candidate, service_sizes_mb, capacity_mb):
                neighbors.append(("add", candidate))
        else:
            candidate = node_cache.copy()
            candidate[service] = 0
            neighbors.append(("delete", candidate))
    ones = [index for index in range(num_services) if node_cache[index] == 1]
    zeros = [index for index in range(num_services) if node_cache[index] == 0]
    for remove_service in ones:
        for add_service in zeros:
            candidate = node_cache.copy()
            candidate[remove_service] = 0
            candidate[add_service] = 1
            if is_node_feasible(candidate, service_sizes_mb, capacity_mb):
                neighbors.append(("swap", candidate))
    return neighbors


def best_distinct_neighbor(
    cache: np.ndarray,
    node_index: int,
    previous_cache: np.ndarray,
    cache_trigger: np.ndarray,
    origin_scenario: dict[str, Any],
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
) -> tuple[str, np.ndarray, float] | None:
    best: tuple[str, np.ndarray, float] | None = None
    for move, node_candidate in generate_node_neighbors(
        previous_cache[node_index],
        service_catalog["service_sizes_mb"],
        cache_nodes["capacities_mb"][node_index],
    ):
        candidate_cache = cache.copy()
        candidate_cache[node_index] = node_candidate
        value = objective(
            candidate_cache,
            previous_cache,
            cache_trigger,
            origin_scenario,
            service_catalog,
            cache_nodes,
            config,
        )
        if best is None or value > best[2] + 1e-12:
            best = (move, node_candidate, value)
    return best


def initialize_cache_for_triggers(
    previous_cache: np.ndarray,
    cache_trigger: np.ndarray,
    origin_scenario: dict[str, Any],
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = service_values_from_scenarios(origin_scenario, config)
    cache = previous_cache.copy()
    trigger_feasible = np.ones(len(cache_trigger), dtype=bool)
    notes: list[str] = []
    for node_index, triggered in enumerate(cache_trigger.tolist()):
        if not triggered:
            continue
        init = knapsack_initial_node(
            values,
            service_catalog["service_sizes_mb"],
            cache_nodes["capacities_mb"][node_index],
        )
        if not np.array_equal(init, previous_cache[node_index]):
            cache[node_index] = init
            continue
        distinct = best_distinct_neighbor(
            cache,
            node_index,
            previous_cache,
            cache_trigger,
            origin_scenario,
            service_catalog,
            cache_nodes,
            config,
        )
        if distinct is None:
            trigger_feasible[node_index] = False
            notes.append(f"node {cache_nodes['node_ids'][node_index]} has no distinct feasible config")
        else:
            cache[node_index] = distinct[1]
            notes.append(
                f"node {cache_nodes['node_ids'][node_index]} knapsack equaled previous; "
                f"used best distinct {distinct[0]} neighbor"
            )
    return cache, {"trigger_feasible_by_node": trigger_feasible, "initialization_notes": notes}


# ============================================================================
# optimizer API
# ============================================================================


def optimize_cache(
    previous_cache: np.ndarray,
    cache_trigger: np.ndarray,
    predicted_scenarios: dict[str, Any],
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig | None = None,
    initial_cache_state: np.ndarray | None = None,
) -> dict[str, Any]:
    """Optimize cache configuration under a fixed trigger vector.

    ``cache_trigger[m]=0`` 严格固定该节点缓存；``cache_trigger[m]=1`` 严格要求
    最终配置与上一时隙不同，除非该节点不存在任何 distinct feasible config。

    ``previous_cache`` 是上一时隙真实 reference state，始终用于 trigger 语义
    和 0->1 loading cost。``initial_cache_state`` 只作为外层 coordinate
    refinement 的搜索起点；省略时保持原来的 trigger-conditioned 初始化逻辑。
    """
    if config is None:
        config = cache_config_from_project(project_config.load_config())
    config.validate()
    previous_cache = np.asarray(previous_cache, dtype=np.int8)
    cache_trigger = np.asarray(cache_trigger, dtype=np.int8)
    validate_cache_shape_and_capacity(previous_cache, service_catalog, cache_nodes)
    if cache_trigger.shape != (previous_cache.shape[0],):
        raise ValueError("cache_trigger shape 错误。")
    if not np.all(np.isin(cache_trigger, [0, 1])):
        raise ValueError("cache_trigger 必须是 0/1 向量。")

    if int(cache_trigger.sum()) == 0:
        unchanged = previous_cache.copy()
        utility = predicted_service_utility(unchanged, predicted_scenarios, cache_nodes, config)
        update_cost, energy = cache_update_cost(
            previous_cache,
            unchanged,
            cache_trigger,
            service_catalog["service_sizes_mb"],
            cache_nodes,
            config,
        )
        return {
            "new_cache": unchanged,
            "trigger_feasible": True,
            "trigger_feasible_by_node": np.ones_like(cache_trigger, dtype=bool),
            "objective": float(utility - update_cost),
            "predicted_utility": utility,
            "cache_update_energy": energy,
            "cache_update_cost": update_cost,
            "statistics": {
                "initial_objective": float(utility - update_cost),
                "final_objective": float(utility - update_cost),
                "num_coordinate_rounds": 0,
                "num_accepted_moves": 0,
                "num_add": 0,
                "num_delete": 0,
                "num_swap": 0,
                "converged_by_no_improvement": True,
                "hit_max_rounds": False,
                "initialization_notes": ["all triggers are zero; cache unchanged"],
            },
        }

    if initial_cache_state is None:
        current_cache, init_info = initialize_cache_for_triggers(
            previous_cache,
            cache_trigger,
            predicted_scenarios,
            service_catalog,
            cache_nodes,
            config,
        )
        trigger_feasible_by_node = init_info["trigger_feasible_by_node"]
    else:
        current_cache = np.asarray(initial_cache_state, dtype=np.int8).copy()
        validate_cache_shape_and_capacity(current_cache, service_catalog, cache_nodes)
        if current_cache.shape != previous_cache.shape:
            raise ValueError("initial_cache_state shape 必须与 previous_cache 一致。")
        trigger_feasible_by_node = np.ones_like(cache_trigger, dtype=bool)
        notes = ["started from external initial_cache_state; costs still reference previous_cache"]
        for node_index, triggered in enumerate(cache_trigger.tolist()):
            if triggered == 0 and not np.array_equal(
                current_cache[node_index],
                previous_cache[node_index],
            ):
                raise ValueError("initial_cache_state 违反 z=0 reference cache 不变语义。")
            if triggered == 1 and np.array_equal(
                current_cache[node_index],
                previous_cache[node_index],
            ):
                trigger_feasible_by_node[node_index] = False
                notes.append(
                    f"node {cache_nodes['node_ids'][node_index]} initial state is not distinct from reference"
                )
        init_info = {
            "trigger_feasible_by_node": trigger_feasible_by_node,
            "initialization_notes": notes,
        }
    if not bool(np.all(trigger_feasible_by_node[cache_trigger.astype(bool)])):
        update_cost, energy = cache_update_cost(
            previous_cache,
            previous_cache,
            cache_trigger,
            service_catalog["service_sizes_mb"],
            cache_nodes,
            config,
        )
        return {
            "new_cache": previous_cache.copy(),
            "trigger_feasible": False,
            "trigger_feasible_by_node": trigger_feasible_by_node,
            "objective": float("-inf"),
            "predicted_utility": predicted_service_utility(
                previous_cache,
                predicted_scenarios,
                cache_nodes,
                config,
            ),
            "cache_update_energy": energy,
            "cache_update_cost": update_cost,
            "statistics": {
                "initial_objective": float("-inf"),
                "final_objective": float("-inf"),
                "num_coordinate_rounds": 0,
                "num_accepted_moves": 0,
                "num_add": 0,
                "num_delete": 0,
                "num_swap": 0,
                "converged_by_no_improvement": False,
                "hit_max_rounds": False,
                "initialization_notes": init_info["initialization_notes"],
            },
        }

    initial_objective = objective(
        current_cache,
        previous_cache,
        cache_trigger,
        predicted_scenarios,
        service_catalog,
        cache_nodes,
        config,
    )
    current_objective = initial_objective
    counts = {"add": 0, "delete": 0, "swap": 0}
    accepted_moves = 0
    hit_max_rounds = False
    converged = False
    completed_rounds = 0

    triggered_indices = [index for index, value in enumerate(cache_trigger.tolist()) if value == 1]
    for round_index in range(1, config.max_coordinate_rounds + 1):
        round_improved = False
        completed_rounds = round_index
        for node_index in triggered_indices:
            best_move: tuple[str, np.ndarray, float] | None = None
            for move, node_candidate in generate_node_neighbors(
                current_cache[node_index],
                service_catalog["service_sizes_mb"],
                cache_nodes["capacities_mb"][node_index],
            ):
                candidate_cache = current_cache.copy()
                candidate_cache[node_index] = node_candidate
                # z=1 节点最终必须不同；local search 中也不接受回到 previous。
                if np.array_equal(candidate_cache[node_index], previous_cache[node_index]):
                    continue
                value = objective(
                    candidate_cache,
                    previous_cache,
                    cache_trigger,
                    predicted_scenarios,
                    service_catalog,
                    cache_nodes,
                    config,
                )
                if value > current_objective + config.tolerance and (
                    best_move is None or value > best_move[2] + config.tolerance
                ):
                    best_move = (move, node_candidate, value)
            if best_move is not None:
                move, node_candidate, value = best_move
                current_cache[node_index] = node_candidate
                current_objective = value
                counts[move] += 1
                accepted_moves += 1
                round_improved = True
        if not round_improved:
            converged = True
            break
    else:
        hit_max_rounds = True

    for node_index in triggered_indices:
        if np.array_equal(current_cache[node_index], previous_cache[node_index]):
            distinct = best_distinct_neighbor(
                current_cache,
                node_index,
                previous_cache,
                cache_trigger,
                predicted_scenarios,
                service_catalog,
                cache_nodes,
                config,
            )
            if distinct is None:
                trigger_feasible_by_node[node_index] = False
            else:
                current_cache[node_index] = distinct[1]
                current_objective = objective(
                    current_cache,
                    previous_cache,
                    cache_trigger,
                    predicted_scenarios,
                    service_catalog,
                    cache_nodes,
                    config,
                )

    validate_cache_shape_and_capacity(current_cache, service_catalog, cache_nodes)
    for node_index, triggered in enumerate(cache_trigger.tolist()):
        if triggered == 0 and not np.array_equal(current_cache[node_index], previous_cache[node_index]):
            raise AssertionError("z=0 节点 cache 被改变。")
        if triggered == 1 and trigger_feasible_by_node[node_index] and np.array_equal(
            current_cache[node_index],
            previous_cache[node_index],
        ):
            raise AssertionError("z=1 且 feasible 的节点未发生配置改变。")

    predicted_utility = predicted_service_utility(
        current_cache,
        predicted_scenarios,
        cache_nodes,
        config,
    )
    update_cost, energy = cache_update_cost(
        previous_cache,
        current_cache,
        cache_trigger,
        service_catalog["service_sizes_mb"],
        cache_nodes,
        config,
    )
    final_objective = float(predicted_utility - update_cost)
    if not math.isfinite(final_objective):
        raise AssertionError("objective 非 finite。")
    return {
        "new_cache": current_cache,
        "trigger_feasible": bool(np.all(trigger_feasible_by_node[cache_trigger.astype(bool)])),
        "trigger_feasible_by_node": trigger_feasible_by_node,
        "objective": final_objective,
        "predicted_utility": predicted_utility,
        "cache_update_energy": energy,
        "cache_update_cost": update_cost,
        "statistics": {
            "initial_objective": float(initial_objective),
            "final_objective": final_objective,
            "num_coordinate_rounds": completed_rounds,
            "num_accepted_moves": accepted_moves,
            "num_add": counts["add"],
            "num_delete": counts["delete"],
            "num_swap": counts["swap"],
            "converged_by_no_improvement": converged,
            "hit_max_rounds": hit_max_rounds,
            "initialization_notes": init_info["initialization_notes"],
        },
    }


# ============================================================================
# Deterministic previous cache for minimal offline validation
# ============================================================================


def deterministic_initial_cache(
    origin_scenario: dict[str, Any],
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
) -> np.ndarray:
    values = service_values_from_scenarios(origin_scenario, config)
    cache = np.zeros(
        (len(cache_nodes["node_ids"]), service_catalog["num_services"]),
        dtype=np.int8,
    )
    for node_index in range(cache.shape[0]):
        cache[node_index] = knapsack_initial_node(
            values,
            service_catalog["service_sizes_mb"],
            cache_nodes["capacities_mb"][node_index],
        )
    validate_cache_shape_and_capacity(cache, service_catalog, cache_nodes)
    return cache


# ============================================================================
# Minimal validation
# ============================================================================


def validate_optimizer_once(
    origin_scenario: dict[str, Any],
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
) -> dict[str, Any]:
    previous_cache = deterministic_initial_cache(
        origin_scenario,
        service_catalog,
        cache_nodes,
        config,
    )
    z0 = np.zeros(previous_cache.shape[0], dtype=np.int8)
    result_z0 = optimize_cache(
        previous_cache,
        z0,
        origin_scenario,
        service_catalog,
        cache_nodes,
        config,
    )
    if not np.array_equal(result_z0["new_cache"], previous_cache):
        raise AssertionError("z=0 验证失败：cache 发生变化。")

    z1 = np.zeros(previous_cache.shape[0], dtype=np.int8)
    z1[0] = 1
    result_single = optimize_cache(
        previous_cache,
        z1,
        origin_scenario,
        service_catalog,
        cache_nodes,
        config,
    )
    if result_single["trigger_feasible"] and np.array_equal(
        result_single["new_cache"][0],
        previous_cache[0],
    ):
        raise AssertionError("单节点 z=1 验证失败：cache 未改变。")

    zmulti = np.zeros(previous_cache.shape[0], dtype=np.int8)
    zmulti[0] = 1
    zmulti[-1] = 1
    result_multi = optimize_cache(
        previous_cache,
        zmulti,
        origin_scenario,
        service_catalog,
        cache_nodes,
        config,
    )
    for node_index, triggered in enumerate(zmulti.tolist()):
        if triggered == 0 and not np.array_equal(
            result_multi["new_cache"][node_index],
            previous_cache[node_index],
        ):
            raise AssertionError("多节点验证失败：z=0 节点发生变化。")
        if triggered == 1 and result_multi["trigger_feasible_by_node"][node_index]:
            if np.array_equal(result_multi["new_cache"][node_index], previous_cache[node_index]):
                raise AssertionError("多节点验证失败：z=1 节点未改变。")

    repeat = optimize_cache(
        previous_cache,
        zmulti,
        origin_scenario,
        service_catalog,
        cache_nodes,
        config,
    )
    if not np.array_equal(result_multi["new_cache"], repeat["new_cache"]):
        raise AssertionError("相同输入未得到相同 cache 配置。")

    # loading energy 只统计 0->1；构造纯 delete candidate 验证删除不产生成本。
    delete_cache = previous_cache.copy()
    node = 0
    ones = np.where(delete_cache[node] == 1)[0]
    if len(ones):
        delete_cache[node, int(ones[0])] = 0
        delete_energy = cache_update_energy(
            previous_cache,
            delete_cache,
            service_catalog["service_sizes_mb"],
            cache_nodes,
        )
        if delete_energy[node] != 0.0:
            raise AssertionError("Delete 错误地产生了 loading energy。")

    for cache in (previous_cache, result_z0["new_cache"], result_single["new_cache"], result_multi["new_cache"]):
        validate_cache_shape_and_capacity(cache, service_catalog, cache_nodes)

    return {
        "previous_cache": previous_cache,
        "z0": result_z0,
        "single": result_single,
        "multi": result_multi,
        "same_input_reproducible": True,
        "delete_loading_energy_zero": True,
    }


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="给定 cache trigger 和预测场景，优化动态缓存配置。"
    )
    parser.add_argument("--config", default="system_config.toml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--prediction-scenarios", default=None)
    parser.add_argument("--service-catalog", default=None)
    parser.add_argument("--origin-index", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        system_config = project_config.apply_overrides(
            project_config.load_config(args.config),
            args.overrides,
        )
        config = cache_config_from_project(system_config)
        config.validate()
        project_root = find_project_root()
        scenario_path_value = args.prediction_scenarios or project_config.get_config_value(
            system_config,
            "paths.prediction_scenarios",
            "data/generated/prediction_scenarios.npz",
        )
        catalog_path_value = args.service_catalog or project_config.get_config_value(
            system_config,
            "paths.service_catalog",
            "data/generated/service_catalog.json",
        )
        scenarios = load_prediction_scenarios(
            resolve_path(project_root, scenario_path_value)
        )
        origin_scenario = scenario_at_origin(scenarios, args.origin_index)
        service_catalog = load_service_catalog(resolve_path(project_root, catalog_path_value))
        cache_nodes = build_cache_nodes(origin_scenario, config)
        validation = validate_optimizer_once(
            origin_scenario,
            service_catalog,
            cache_nodes,
            config,
        )
        multi = validation["multi"]
        print("缓存配置优化最小验证完成：")
        print(f"  config_hash={project_config.config_hash(system_config)}")
        print(f"  origin_slot={origin_scenario['origin_slot']}")
        print(f"  cache nodes={cache_nodes['node_ids']}")
        print(f"  z=0 objective={validation['z0']['objective']:.6f}")
        print(
            f"  single-trigger objective={validation['single']['objective']:.6f}, "
            f"cost={validation['single']['cache_update_cost']:.6f}"
        )
        print(
            f"  multi-trigger initial/final="
            f"{multi['statistics']['initial_objective']:.6f}/"
            f"{multi['statistics']['final_objective']:.6f}"
        )
        print(
            f"  multi-trigger cost={multi['cache_update_cost']:.6f}, "
            f"energy={multi['cache_update_energy'].tolist()}"
        )
        print(f"  stats={json.dumps(multi['statistics'], ensure_ascii=False)}")
        print("  capacity_ok=True, same_input_reproducible=True, future_ground_truth_read=False")
        return 0
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
