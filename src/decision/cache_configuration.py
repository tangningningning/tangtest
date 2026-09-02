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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from common import project_config, system_model


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
    candidates = (Path(__file__).resolve().parents[2], Path.cwd().resolve())
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


@dataclass(frozen=True)
class SurrogateEvent:
    """当前 configuration slot 中一个 arrival=1 的预测任务事件。"""

    scenario_index: int
    horizon: int
    tv_index: int
    tv_id: str
    service_id: int
    input_bits: float
    cpu_cycles: float
    discount: float
    num_scenarios: int
    predicted_tv_position: np.ndarray


@dataclass
class SurrogateEvaluationContext:
    """slot-scoped surrogate 预计算结果和 UAV lazy position cache。

    该对象只由当前 scenario slice 生命周期内的调用显式传递，不使用
    module global cache。ground_edge_utility 对 SV/RSU 固定部分预计算一次，
    UAV 部分按 ``(uav_id, exact_x, exact_y)`` 惰性缓存。
    """

    events: tuple[SurrogateEvent, ...]
    num_scenarios: int
    node_roles: tuple[str, ...]
    node_ids: tuple[str, ...]
    uav_node_indices: dict[str, int]
    base_utility: np.ndarray
    ground_edge_utility: np.ndarray
    uav_utility_cache: dict[tuple[str, float, float], np.ndarray] = field(default_factory=dict)


def _task_utility_from_execution(
    execution: dict[str, float],
    config: CacheOptimizationConfig,
) -> float:
    """只负责复用统一 system_model.task_utility，不重写物理公式。"""

    return float(
        system_model.task_utility(
            execution["total_delay_s"],
            execution["vehicle_energy_j"],
            config.time_ref_s,
            config.energy_ref_j,
            config.omega_t,
            config.omega_e,
        )
    )


def _local_task_utility(
    event: SurrogateEvent,
    config: CacheOptimizationConfig,
) -> float:
    execution = system_model.local_execution(
        event.cpu_cycles,
        config.local_cpu_cycles_per_s,
        config.local_energy_per_cycle_j,
    )
    return float(
        system_model.task_utility(
            execution["delay_s"],
            execution["vehicle_energy_j"],
            config.time_ref_s,
            config.energy_ref_j,
            config.omega_t,
            config.omega_e,
        )
    )


def _cloud_task_utility(
    event: SurrogateEvent,
    config: CacheOptimizationConfig,
) -> float:
    execution = system_model.cloud_execution(
        event.input_bits,
        event.cpu_cycles,
        config.cloud_uplink_rate_bps,
        config.cloud_cpu_cycles_per_s,
        config.cloud_delay_factor,
        config.tx_power_w,
    )
    return _task_utility_from_execution(
        {
            "total_delay_s": execution["total_delay_s"],
            "vehicle_energy_j": execution["vehicle_energy_j"],
        },
        config,
    )


def _ground_edge_task_utility(
    event: SurrogateEvent,
    role: str,
    distance: float,
    config: CacheOptimizationConfig,
) -> float:
    gain = system_model.ground_channel_gain(
        max(distance, 1e-9),
        config.channel_gain_at_1m,
        config.pathloss_exponent,
    )
    execution = system_model.edge_execution(
        event.input_bits,
        event.cpu_cycles,
        role_bandwidth_hz(role, config),
        bandwidth_fraction=1.0,
        transmit_power_w=config.tx_power_w,
        channel_gain=gain,
        noise_power_w=config.noise_power_w,
        node_cpu_cycles_per_s=role_cpu_cycles_per_s(role, config),
        cpu_fraction=1.0,
    )
    return _task_utility_from_execution(execution, config)


def build_surrogate_evaluation_context(
    origin_scenario: dict[str, Any],
    local_cache_state: dict[str, set[int]],
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
) -> SurrogateEvaluationContext:
    """为一个 scenario slice 预计算不随 cache/UAV candidate 改变的量。"""

    arrival = origin_scenario["arrival"]
    service_id = origin_scenario["service_id"]
    input_size = origin_scenario["input_size"]
    cpu_cycles = origin_scenario["cpu_cycles"]
    tv_ids = [str(value) for value in origin_scenario["task_vehicle_ids"].tolist()]
    num_scenarios, horizon_steps, num_tv = arrival.shape
    local_services = {
        tv_id: _local_services_for_tv(local_cache_state, tv_id)
        for tv_id in tv_ids
    }
    events: list[SurrogateEvent] = []
    for scenario_index in range(num_scenarios):
        for horizon in range(horizon_steps):
            discount = config.beta_discount ** horizon
            for tv_index in range(num_tv):
                if int(arrival[scenario_index, horizon, tv_index]) == 0:
                    continue
                service = int(service_id[scenario_index, horizon, tv_index])
                if service < 0:
                    raise ValueError("arrival=1 时 service_id 无效。")
                events.append(
                    SurrogateEvent(
                        scenario_index=scenario_index,
                        horizon=horizon,
                        tv_index=tv_index,
                        tv_id=tv_ids[tv_index],
                        service_id=service,
                        input_bits=float(input_size[scenario_index, horizon, tv_index]),
                        cpu_cycles=float(cpu_cycles[scenario_index, horizon, tv_index]),
                        discount=float(discount),
                        num_scenarios=int(num_scenarios),
                        predicted_tv_position=np.asarray(
                            tv_position(
                                origin_scenario,
                                tv_ids[tv_index],
                                scenario_index,
                                horizon,
                            ),
                            dtype=np.float64,
                        ).copy(),
                    )
                )

    event_count = len(events)
    node_roles = tuple(str(role) for role in cache_nodes["node_roles"])
    node_ids = tuple(str(node_id) for node_id in cache_nodes["node_ids"])
    base_utility = np.full(event_count, -float("inf"), dtype=np.float64)
    ground_edge_utility = np.full(
        (event_count, len(node_roles)),
        -float("inf"),
        dtype=np.float64,
    )
    for event_index, event in enumerate(events):
        local = -float("inf")
        if event.service_id in local_services[event.tv_id]:
            local = _local_task_utility(event, config)
        cloud = _cloud_task_utility(event, config)
        base_utility[event_index] = max(local, cloud)

        for node_index, role in enumerate(node_roles):
            if role == "UAV":
                continue
            node_xy = node_position(
                origin_scenario,
                cache_nodes,
                node_index,
                event.scenario_index,
                event.horizon,
            )
            distance = float(np.linalg.norm(event.predicted_tv_position - node_xy))
            if distance > role_range_m(role, config):
                continue
            ground_edge_utility[event_index, node_index] = _ground_edge_task_utility(
                event,
                role,
                distance,
                config,
            )

    return SurrogateEvaluationContext(
        events=tuple(events),
        num_scenarios=int(num_scenarios),
        node_roles=node_roles,
        node_ids=node_ids,
        uav_node_indices={
            node_id: index
            for index, (node_id, role) in enumerate(zip(node_ids, node_roles))
            if role == "UAV"
        },
        base_utility=base_utility,
        ground_edge_utility=ground_edge_utility,
    )


def _uav_edge_utility_vector(
    context: SurrogateEvaluationContext,
    uav_id: str,
    position: Any,
    config: CacheOptimizationConfig,
) -> np.ndarray:
    xy = np.asarray(position, dtype=np.float64)
    if xy.shape != (2,) or not np.all(np.isfinite(xy)):
        raise ValueError(f"UAV 位置必须是 finite 二维坐标：{uav_id}")
    key = (str(uav_id), float(xy[0]), float(xy[1]))
    cached = context.uav_utility_cache.get(key)
    if cached is not None:
        return cached

    values = np.full(len(context.events), -float("inf"), dtype=np.float64)
    for event_index, event in enumerate(context.events):
        distance = float(np.linalg.norm(event.predicted_tv_position - xy))
        if distance > config.uav_range_m:
            continue
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
        execution = system_model.edge_execution(
            event.input_bits,
            event.cpu_cycles,
            config.uav_bandwidth_hz,
            bandwidth_fraction=1.0,
            transmit_power_w=config.tx_power_w,
            channel_gain=gain,
            noise_power_w=config.noise_power_w,
            node_cpu_cycles_per_s=config.uav_cpu_cycles_per_s,
            cpu_fraction=1.0,
        )
        values[event_index] = _task_utility_from_execution(execution, config)
    context.uav_utility_cache[key] = values
    return values


def evaluate_surrogate_context(
    cache: np.ndarray,
    context: SurrogateEvaluationContext,
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
) -> float:
    """按旧 scenario→horizon→tv、node 顺序累计预计算 surrogate。"""

    uav_positions = {
        uav_id: cache_nodes["fixed_positions"][uav_id]
        for uav_id in context.uav_node_indices
    }
    total = 0.0
    for event_index, event in enumerate(context.events):
        best = float(context.base_utility[event_index])
        for node_index, role in enumerate(context.node_roles):
            if cache[node_index, event.service_id] == 0:
                continue
            if role == "UAV":
                edge = _uav_edge_utility_vector(
                    context,
                    context.node_ids[node_index],
                    uav_positions[context.node_ids[node_index]],
                    config,
                )
            else:
                edge = float(context.ground_edge_utility[event_index, node_index])
            if isinstance(edge, np.ndarray):
                edge = float(edge[event_index])
            if edge > best:
                best = edge
        total += event.discount * best
    return float(total / max(context.num_scenarios, 1))


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


def _local_services_for_tv(
    local_cache_state: dict[str, set[int]],
    tv_id: str,
) -> set[int]:
    """读取显式传入的 TV fixed local cache，不在核心 optimizer 内读文件。"""

    if tv_id not in local_cache_state:
        raise ValueError(f"local_cache_state 缺少 TV：{tv_id}")
    return {int(service_id) for service_id in local_cache_state[tv_id]}


def best_task_surrogate(
    cache: np.ndarray,
    origin_scenario: dict[str, Any],
    cache_nodes: dict[str, Any],
    service_id: int,
    input_bits: float,
    cpu_cycles: float,
    tv_id: str,
    local_cache_state: dict[str, set[int]],
    scenario_index: int,
    horizon: int,
    config: CacheOptimizationConfig,
) -> float:
    best = -float("inf")

    # Local 只有在该 TV 的 fixed local cache 中包含 service 时才可用。
    if service_id in _local_services_for_tv(local_cache_state, tv_id):
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
    local_cache_state: dict[str, set[int]],
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
    evaluation_context: SurrogateEvaluationContext | None = None,
) -> float:
    context = evaluation_context
    if context is None:
        context = build_surrogate_evaluation_context(
            origin_scenario,
            local_cache_state,
            cache_nodes,
            config,
        )
    return evaluate_surrogate_context(cache, context, cache_nodes, config)


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
    local_cache_state: dict[str, set[int]],
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
    evaluation_context: SurrogateEvaluationContext | None = None,
) -> float:
    utility = predicted_service_utility(
        cache,
        origin_scenario,
        local_cache_state,
        cache_nodes,
        config,
        evaluation_context=evaluation_context,
    )
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
    local_cache_state: dict[str, set[int]],
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
    evaluation_context: SurrogateEvaluationContext | None = None,
) -> dict[str, float]:
    """评估给定 cache + 节点位置下的预测 utility。

    这是给 Cache/UAV 配置模块共用的纯 evaluator：它不触发 cache local
    search，不修改 cache，也不读取未来真实 trajectory/tasks。UAV 配置优化只
    需要在不同 UAV fixed_positions 下复用同一套 surrogate 口径。
    """

    utility = predicted_service_utility(
        cache,
        origin_scenario,
        local_cache_state,
        cache_nodes,
        config,
        evaluation_context=evaluation_context,
    )
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
    local_cache_state: dict[str, set[int]],
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
    evaluation_context: SurrogateEvaluationContext | None = None,
) -> tuple[str, np.ndarray, float] | None:
    if evaluation_context is None:
        evaluation_context = build_surrogate_evaluation_context(
            origin_scenario,
            local_cache_state,
            cache_nodes,
            config,
        )
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
            local_cache_state,
            cache_nodes,
            config,
            evaluation_context=evaluation_context,
        )
        if best is None or value > best[2] + 1e-12:
            best = (move, node_candidate, value)
    return best


def initialize_cache_for_triggers(
    previous_cache: np.ndarray,
    cache_trigger: np.ndarray,
    origin_scenario: dict[str, Any],
    service_catalog: dict[str, Any],
    local_cache_state: dict[str, set[int]],
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
    evaluation_context: SurrogateEvaluationContext | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if evaluation_context is None:
        evaluation_context = build_surrogate_evaluation_context(
            origin_scenario,
            local_cache_state,
            cache_nodes,
            config,
        )
    values = service_values_from_scenarios(origin_scenario, config)
    cache = previous_cache.copy()
    trigger_feasible = np.ones(len(cache_trigger), dtype=bool)
    notes: list[str] = []
    current_objective = objective(
        cache,
        previous_cache,
        cache_trigger,
        origin_scenario,
        service_catalog,
        local_cache_state,
        cache_nodes,
        config,
        evaluation_context=evaluation_context,
    )
    for node_index, triggered in enumerate(cache_trigger.tolist()):
        if not triggered:
            continue
        init = knapsack_initial_node(
            values,
            service_catalog["service_sizes_mb"],
            cache_nodes["capacities_mb"][node_index],
        )
        candidate_cache = cache.copy()
        candidate_cache[node_index] = init
        candidate_objective = objective(
            candidate_cache,
            previous_cache,
            cache_trigger,
            origin_scenario,
            service_catalog,
            local_cache_state,
            cache_nodes,
            config,
            evaluation_context=evaluation_context,
        )
        if candidate_objective > current_objective + config.tolerance:
            cache = candidate_cache
            current_objective = candidate_objective
            notes.append(
                f"node {cache_nodes['node_ids'][node_index]} accepted knapsack candidate"
            )
        else:
            notes.append(
                f"node {cache_nodes['node_ids'][node_index]} knapsack candidate not better; "
                "kept current cache"
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
    local_cache_state: dict[str, set[int]],
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig | None = None,
    initial_cache_state: np.ndarray | None = None,
    evaluation_context: SurrogateEvaluationContext | None = None,
) -> dict[str, Any]:
    """Optimize cache configuration under a fixed trigger vector.

    ``cache_trigger[m]`` 表示上层缓存更新意图：0 严格固定该节点缓存，1 允许
    重新优化。实际更新由最终配置与上一时隙缓存的差异决定。

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
    if evaluation_context is None:
        evaluation_context = build_surrogate_evaluation_context(
            predicted_scenarios,
            local_cache_state,
            cache_nodes,
            config,
        )

    if int(cache_trigger.sum()) == 0:
        unchanged = previous_cache.copy()
        utility = predicted_service_utility(
            unchanged,
            predicted_scenarios,
            local_cache_state,
            cache_nodes,
            config,
            evaluation_context=evaluation_context,
        )
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
            "actual_cache_trigger": np.any(
                unchanged != previous_cache,
                axis=1,
            ).astype(np.int8),
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
            local_cache_state,
            cache_nodes,
            config,
            evaluation_context=evaluation_context,
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
                notes.append(
                    f"node {cache_nodes['node_ids'][node_index]} initial state equals reference; "
                    "actual update may remain zero"
                )
        init_info = {
            "trigger_feasible_by_node": trigger_feasible_by_node,
            "initialization_notes": notes,
        }

    initial_objective = objective(
        current_cache,
        previous_cache,
        cache_trigger,
        predicted_scenarios,
        service_catalog,
        local_cache_state,
        cache_nodes,
        config,
        evaluation_context=evaluation_context,
    )
    if initial_cache_state is not None:
        previous_objective = objective(
            previous_cache,
            previous_cache,
            cache_trigger,
            predicted_scenarios,
            service_catalog,
            local_cache_state,
            cache_nodes,
            config,
            evaluation_context=evaluation_context,
        )
        if initial_objective <= previous_objective + config.tolerance:
            current_cache = previous_cache.copy()
            initial_objective = previous_objective
            init_info["initialization_notes"].append(
                "previous cache objective was not worse; kept previous cache"
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
                value = objective(
                    candidate_cache,
                    previous_cache,
                    cache_trigger,
                    predicted_scenarios,
                    service_catalog,
                    local_cache_state,
                    cache_nodes,
                    config,
                    evaluation_context=evaluation_context,
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

    validate_cache_shape_and_capacity(current_cache, service_catalog, cache_nodes)
    for node_index, triggered in enumerate(cache_trigger.tolist()):
        if triggered == 0 and not np.array_equal(current_cache[node_index], previous_cache[node_index]):
            raise AssertionError("z=0 节点 cache 被改变。")

    predicted_utility = predicted_service_utility(
        current_cache,
        predicted_scenarios,
        local_cache_state,
        cache_nodes,
        config,
        evaluation_context=evaluation_context,
    )
    update_cost, energy = cache_update_cost(
        previous_cache,
        current_cache,
        cache_trigger,
        service_catalog["service_sizes_mb"],
        cache_nodes,
        config,
    )
    actual_cache_trigger = np.any(
        current_cache != previous_cache,
        axis=1,
    ).astype(np.int8)
    final_objective = float(predicted_utility - update_cost)
    if not math.isfinite(final_objective):
        raise AssertionError("objective 非 finite。")
    return {
        "new_cache": current_cache,
        "trigger_feasible": True,
        "trigger_feasible_by_node": trigger_feasible_by_node,
        "actual_cache_trigger": actual_cache_trigger,
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


def random_feasible_initial_cache(
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
    seed: int | None = None,
) -> np.ndarray:
    """生成正式 episode 的固定 seed 随机可行初始动态缓存 C(0)。

    该函数只访问 service catalog、节点容量和显式 seed；不读取 prediction
    scenarios、future tasks、popularity，也不调用任何 utility/objective。
    ``default_rng`` 是局部 RNG，不改变全局随机状态。正式 caller 应只在
    context 构造阶段调用一次，并在每个 episode reset 时复制返回状态。
    """

    config.validate()
    num_services = int(service_catalog["num_services"])
    service_sizes = np.asarray(service_catalog["service_sizes_mb"], dtype=np.float64)
    if service_sizes.shape != (num_services,) or np.any(~np.isfinite(service_sizes)):
        raise ValueError("service_catalog service_sizes_mb 形状或数值非法。")
    if np.any(service_sizes < 0.0):
        raise ValueError("service_sizes_mb 不能为负。")
    if len(cache_nodes["node_ids"]) != len(cache_nodes["capacities_mb"]):
        raise ValueError("cache_nodes node_ids/capacities_mb 长度不一致。")

    rng = np.random.default_rng(
        int(config.initial_cache_seed if seed is None else seed)
    )
    cache = np.zeros(
        (len(cache_nodes["node_ids"]), num_services),
        dtype=np.int8,
    )
    for node_index, capacity in enumerate(cache_nodes["capacities_mb"]):
        remaining = float(capacity)
        service_order = rng.permutation(num_services)
        for service_id in service_order.tolist():
            size = float(service_sizes[service_id])
            if size <= remaining + 1.0e-9:
                cache[node_index, service_id] = 1
                remaining -= size

        used = node_used_capacity_mb(cache[node_index], service_sizes)
        remaining_capacity = float(capacity) - used
        for service_id in range(num_services):
            if (
                cache[node_index, service_id] == 0
                and service_sizes[service_id] <= remaining_capacity + 1.0e-9
            ):
                raise AssertionError(
                    f"node {cache_nodes['node_ids'][node_index]} "
                    "initial cache is not maximal feasible"
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
    local_cache_state: dict[str, set[int]],
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
        local_cache_state,
        cache_nodes,
        config,
    )
    if not np.array_equal(result_z0["new_cache"], previous_cache):
        raise AssertionError("z=0 验证失败：cache 发生变化。")

    def assert_actual_trigger(result: dict[str, Any], label: str) -> None:
        expected = np.any(
            result["new_cache"] != previous_cache,
            axis=1,
        ).astype(np.int8)
        if not np.array_equal(result["actual_cache_trigger"], expected):
            raise AssertionError(f"{label} actual_cache_trigger 与缓存差异不一致。")
        if not np.any(expected):
            if not math.isclose(
                float(result["cache_update_cost"]),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ) or not np.allclose(result["cache_update_energy"], 0.0):
                raise AssertionError(f"{label} cache 未变化却产生了更新成本或能耗。")

    assert_actual_trigger(result_z0, "z=0")

    z1 = np.zeros(previous_cache.shape[0], dtype=np.int8)
    z1[0] = 1
    result_single = optimize_cache(
        previous_cache,
        z1,
        origin_scenario,
        service_catalog,
        local_cache_state,
        cache_nodes,
        config,
    )
    assert_actual_trigger(result_single, "单节点 z=1")

    zmulti = np.zeros(previous_cache.shape[0], dtype=np.int8)
    zmulti[0] = 1
    zmulti[-1] = 1
    result_multi = optimize_cache(
        previous_cache,
        zmulti,
        origin_scenario,
        service_catalog,
        local_cache_state,
        cache_nodes,
        config,
    )
    for node_index, triggered in enumerate(zmulti.tolist()):
        if triggered == 0 and not np.array_equal(
            result_multi["new_cache"][node_index],
            previous_cache[node_index],
        ):
            raise AssertionError("多节点验证失败：z=0 节点发生变化。")
    assert_actual_trigger(result_multi, "多节点")

    repeat = optimize_cache(
        previous_cache,
        zmulti,
        origin_scenario,
        service_catalog,
        local_cache_state,
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
        from scenario import task_scenario

        local_cache_state = task_scenario.load_tv_local_cache(
            resolve_path(
                project_root,
                project_config.get_config_value(system_config, "paths.tv_local_cache"),
            )
        )
        validation = validate_optimizer_once(
            origin_scenario,
            service_catalog,
            cache_nodes,
            config,
            local_cache_state,
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
