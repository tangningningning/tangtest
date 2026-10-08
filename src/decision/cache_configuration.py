#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trigger-conditioned cache configuration optimization.

本阶段只解决：给定上一时隙缓存状态、cache trigger 和预测场景后，选择新的
动态缓存配置。预测候选复用实际执行侧的 Matching/KKT 和固定每任务 Cloud 模型，
但不训练 PPO，也不读取任何未来真实 trajectory/tasks/popularity 标签；正式
episode 初始缓存只使用 service catalog 中的 service size、节点容量和显式 seed。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
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
    # Cloud 两组字段保持兼容，但正式语义是每任务固定资源。
    cloud_cpu_cycles_per_s: float
    cloud_total_cpu_cycles_per_s: float
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
    # Cloud 两组字段保持兼容，但正式语义是每任务固定等效上行速率。
    cloud_uplink_rate_bps: float
    cloud_total_bandwidth_bps: float
    cloud_delay_factor: float
    cloud_resource_sharing: bool
    sv_range_m: float
    rsu_range_m: float
    uav_range_m: float
    tolerance: float
    max_coordinate_rounds: int
    initial_cache_seed: int
    matching_max_updates: int
    matching_objective_tolerance: float
    kkt_min_cpu_fraction: float
    kkt_min_bandwidth_fraction: float
    kkt_bisection_tolerance: float
    kkt_max_bisection_iterations: int
    cache_cost_weight: float = 1.0
    # Proposed-policy target configuration proposal controls.  These are
    # engineering bounds/coverage knobs, not physical system parameters.
    target_candidate_capacity: int = 64
    target_intermediate_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)

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
            self.cloud_total_cpu_cycles_per_s,
            self.cloud_total_bandwidth_bps,
            self.sv_bandwidth_hz,
            self.rsu_bandwidth_hz,
            self.uav_bandwidth_hz,
            self.noise_power_w,
            self.max_coordinate_rounds,
            self.matching_max_updates,
            self.kkt_min_cpu_fraction,
            self.kkt_min_bandwidth_fraction,
            self.kkt_bisection_tolerance,
            self.kkt_max_bisection_iterations,
        ]
        if any(float(value) <= 0.0 for value in values):
            raise ValueError("缓存优化配置存在非正数参数。")
        if not 0.0 < self.beta_discount <= 1.0:
            raise ValueError("beta_discount 必须位于 (0, 1]。")
        if self.tolerance < 0.0:
            raise ValueError("tolerance 不能为负数。")
        if not math.isfinite(self.cache_cost_weight) or self.cache_cost_weight < 0.0:
            raise ValueError("cache_cost_weight 必须是非负有限数。")
        if int(self.target_candidate_capacity) < 1:
            raise ValueError("target_candidate_capacity 必须 >= 1。")
        if not self.target_intermediate_fractions or any(
            not math.isfinite(float(value)) or not 0.0 < float(value) < 1.0
            for value in self.target_intermediate_fractions
        ):
            raise ValueError(
                "target_intermediate_fractions 必须是 (0,1) 内的有限非空序列。"
            )
        if self.num_rsu != len(self.rsu_instances):
            raise ValueError("RSU instance 数量必须等于 num_rsu。")
        if self.num_uav != len(self.uav_instances):
            raise ValueError("UAV instance 数量必须等于 num_uav。")
        if not isinstance(self.cloud_resource_sharing, bool):
            raise ValueError("cloud_resource_sharing 必须是显式 bool。")
        if self.cloud_resource_sharing:
            raise ValueError(
                "当前正式模型要求 Cloud 固定每任务资源，"
                "cloud_resource_sharing 必须为 false。"
            )


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
    cloud_resource_sharing = get(config, "nodes.cloud.resource_sharing")
    if not isinstance(cloud_resource_sharing, bool):
        raise ValueError(
            "缺少或非法的 nodes.cloud.resource_sharing；"
            "正式模型不允许静默回退到 Cloud 共享资源。"
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
        cloud_total_cpu_cycles_per_s=float(get(config, "nodes.cloud.total_cpu_cycles_per_s")),
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
        cloud_total_bandwidth_bps=float(get(config, "nodes.cloud.total_bandwidth_bps")),
        cloud_delay_factor=float(get(config, "nodes.cloud.cloud_delay_factor")),
        cloud_resource_sharing=cloud_resource_sharing,
        sv_range_m=float(get(config, "nodes.sv.range_m")),
        rsu_range_m=float(get(config, "nodes.rsu.range_m")),
        uav_range_m=float(get(config, "nodes.uav.range_m")),
        tolerance=float(get(config, "algorithm.cache.tolerance")),
        max_coordinate_rounds=int(get(config, "algorithm.cache.max_coordinate_rounds")),
        initial_cache_seed=int(get(config, "experiment.seed")),
        matching_max_updates=int(get(config, "algorithm.matching.max_updates")),
        matching_objective_tolerance=float(
            get(config, "algorithm.matching.objective_tolerance")
        ),
        kkt_min_cpu_fraction=float(get(config, "algorithm.kkt.min_cpu_fraction")),
        kkt_min_bandwidth_fraction=float(
            get(config, "algorithm.kkt.min_bandwidth_fraction")
        ),
        kkt_bisection_tolerance=float(
            get(config, "algorithm.kkt.bisection_tolerance")
        ),
        kkt_max_bisection_iterations=int(
            get(config, "algorithm.kkt.max_bisection_iterations")
        ),
        cache_cost_weight=float(get(config, "objective.cache_cost_weight", 1.0)),
        target_candidate_capacity=int(
            get(config, "algorithm.cache.target_candidate_capacity", 64)
        ),
        target_intermediate_fractions=tuple(
            float(value)
            for value in get(
                config,
                "algorithm.cache.target_intermediate_fractions",
                [0.25, 0.5, 0.75],
            )
        ),
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






# ============================================================================
# Cache candidate utility helpers
# ============================================================================


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
        cache_cost_weight=config.cache_cost_weight,
    )
    return float(result["total_cost"]), np.asarray(result["loading_energy"], dtype=np.float64)












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


def _serviceability_weight(
    distance_m: float,
    role: str,
    config: CacheOptimizationConfig,
) -> float:
    """Return a bounded geometric serviceability weight for proposal features.

    This is deliberately only a proposal signal.  The actual communication,
    Matching, KKT, and reward paths remain in their existing modules.
    """

    ranges = {
        "SV": float(config.sv_range_m),
        "RSU": float(config.rsu_range_m),
        "UAV": float(config.uav_range_m),
    }
    radius = ranges[str(role).upper()]
    if distance_m >= radius:
        return 0.0
    return float(max(0.0, 1.0 - distance_m / max(radius, 1.0e-12)))


def normalize_prediction_horizon_mask(
    prediction_horizon_mask: Any,
    horizon_count: int,
) -> np.ndarray:
    """Validate and normalize the rollout's existing prediction horizon mask."""

    count = int(horizon_count)
    if count < 0:
        raise ValueError("prediction horizon count must be nonnegative")
    if prediction_horizon_mask is None:
        return np.ones(count, dtype=bool)
    raw = np.asarray(prediction_horizon_mask).reshape(-1)
    if raw.shape != (count,) or not np.all(np.isin(raw, [0, 1])):
        raise ValueError(
            "prediction_horizon_mask must be a binary array aligned with scenario horizons"
        )
    return raw.astype(bool, copy=False)


def predicted_vehicle_service_demand_by_horizon(
    origin_scenario: dict[str, Any],
    num_services: int,
    beta_discount: float = 1.0,
    prediction_horizon_mask: Any = None,
) -> np.ndarray:
    """Build a vehicle-specific demand tensor from prediction scenarios only.

    ``arrival``/``service_id`` are sampled prediction scenarios, not the
    realized future task stream.  If a scenario artifact does not contain
    usable per-vehicle events, the available predicted popularity is distributed
    uniformly over vehicles as a finite fallback.
    """

    arrival = np.asarray(origin_scenario.get("arrival"), dtype=np.float64)
    service_id = np.asarray(origin_scenario.get("service_id"), dtype=np.int64)
    positions = np.asarray(origin_scenario.get("positions_xy"), dtype=np.float64)
    if arrival.ndim != 3 or service_id.shape != arrival.shape:
        raise ValueError("prediction scenario arrival/service_id must be [S,H,V]")
    scenarios, horizons, task_vehicle_count = [int(value) for value in arrival.shape]
    valid_horizon_mask = normalize_prediction_horizon_mask(
        prediction_horizon_mask, horizons
    )
    if positions.ndim != 4 or positions.shape[:2] != (scenarios, horizons):
        raise ValueError("prediction scenario positions_xy must align with [S,H,V,2]")
    vehicles = int(positions.shape[2])
    if positions.shape[3] != 2:
        raise ValueError("prediction scenario positions_xy must end in coordinate dimension 2")
    vehicle_ids = [str(value) for value in origin_scenario["vehicle_ids"].tolist()]
    raw_task_vehicle_ids = origin_scenario.get("task_vehicle_ids")
    task_vehicle_ids = (
        [str(value) for value in raw_task_vehicle_ids.tolist()]
        if raw_task_vehicle_ids is not None
        else []
    )
    if len(vehicle_ids) != vehicles or len(task_vehicle_ids) != task_vehicle_count:
        raise ValueError("prediction vehicle ID arrays do not align with scenario tensors")
    vehicle_index = {vehicle_id: index for index, vehicle_id in enumerate(vehicle_ids)}
    task_to_vehicle = [vehicle_index[vehicle_id] for vehicle_id in task_vehicle_ids]
    demand = np.zeros((scenarios, horizons, vehicles, int(num_services)), dtype=np.float64)
    for scenario_index in range(scenarios):
        for horizon in range(horizons):
            if not valid_horizon_mask[horizon]:
                continue
            discount = float(beta_discount) ** int(horizon)
            for task_index, full_vehicle_index in enumerate(task_to_vehicle):
                if arrival[scenario_index, horizon, task_index] <= 0.0:
                    continue
                service = int(service_id[scenario_index, horizon, task_index])
                if 0 <= service < int(num_services):
                    demand[scenario_index, horizon, full_vehicle_index, service] += (
                        discount * arrival[scenario_index, horizon, task_index]
                    )
    if not float(np.sum(demand)) > 1.0e-12:
        rho_hat = np.asarray(origin_scenario.get("rho_hat"), dtype=np.float64)
        if rho_hat.ndim == 2 and rho_hat.shape[-1] == int(num_services):
            if rho_hat.shape[0] == horizons:
                valid_rho_hat = rho_hat[valid_horizon_mask]
                popularity = (
                    np.mean(valid_rho_hat, axis=0)
                    if len(valid_rho_hat)
                    else np.zeros(int(num_services), dtype=np.float64)
                )
            else:
                popularity = np.mean(rho_hat, axis=0)
        elif rho_hat.ndim == 3 and rho_hat.shape[-1] == int(num_services):
            if rho_hat.shape[1] == horizons:
                valid_rho_hat = rho_hat[:, valid_horizon_mask, :]
                popularity = (
                    np.mean(valid_rho_hat, axis=(0, 1))
                    if valid_rho_hat.shape[1]
                    else np.zeros(int(num_services), dtype=np.float64)
                )
            else:
                popularity = np.mean(rho_hat, axis=tuple(range(rho_hat.ndim - 1)))
        else:
            popularity = np.full(int(num_services), 1.0 / max(int(num_services), 1))
        demand[:] = 0.0
        fallback_indices = task_to_vehicle or list(range(vehicles))
        fallback_share = popularity.reshape(1, 1, -1) / max(len(fallback_indices), 1)
        for horizon in np.flatnonzero(valid_horizon_mask):
            demand[:, int(horizon), fallback_indices, :] = fallback_share
    return np.mean(demand, axis=0)


def predicted_vehicle_service_demand(
    origin_scenario: dict[str, Any],
    num_services: int,
    beta_discount: float = 1.0,
    prediction_horizon_mask: Any = None,
) -> np.ndarray:
    """Return discounted predicted demand per vehicle/service, averaged over S."""

    by_horizon = predicted_vehicle_service_demand_by_horizon(
        origin_scenario,
        num_services,
        beta_discount=beta_discount,
        prediction_horizon_mask=prediction_horizon_mask,
    )
    return np.sum(by_horizon, axis=0)


def _node_predicted_serviceability_weights(
    origin_scenario: dict[str, Any],
    node_index: int,
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
) -> np.ndarray:
    """Return the existing prediction-only node/vehicle coverage weights."""

    positions = np.asarray(origin_scenario["positions_xy"], dtype=np.float64)
    if positions.ndim != 4 or positions.shape[-1] != 2:
        raise ValueError("prediction scenario positions_xy must be [S,H,V,2]")
    _, horizons, vehicles, _ = positions.shape
    vehicle_ids = [str(value) for value in origin_scenario["vehicle_ids"].tolist()]
    role = str(cache_nodes["node_roles"][int(node_index)]).upper()
    node_id = str(cache_nodes["node_ids"][int(node_index)])
    node_vehicle_index = vehicle_ids.index(node_id) if role == "SV" else None
    mean_positions = np.mean(positions, axis=0)
    weights = np.zeros((horizons, vehicles), dtype=np.float64)
    for horizon in range(horizons):
        if role == "SV":
            node_xy = mean_positions[horizon, int(node_vehicle_index)]
        else:
            node_xy = np.asarray(
                cache_nodes["fixed_positions"][node_id], dtype=np.float64
            )
        for vehicle_index in range(vehicles):
            distance = float(
                np.linalg.norm(mean_positions[horizon, vehicle_index] - node_xy)
            )
            weights[horizon, vehicle_index] = _serviceability_weight(
                distance, role, config
            )
    return weights


def node_specific_predicted_demand(
    origin_scenario: dict[str, Any],
    node_index: int,
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
    *,
    prediction_horizon_mask: Any = None,
) -> np.ndarray:
    """Estimate ``D_pred[m,f]`` using predicted vehicles serviceable by node m."""

    num_services = int(origin_scenario["rho_hat"].shape[-1])
    arrival = np.asarray(origin_scenario["arrival"], dtype=np.float64)
    demand = predicted_vehicle_service_demand_by_horizon(
        origin_scenario,
        num_services,
        beta_discount=config.beta_discount,
        prediction_horizon_mask=prediction_horizon_mask,
    )
    weights = _node_predicted_serviceability_weights(
        origin_scenario, node_index, cache_nodes, config
    )
    if weights.shape != demand.shape[:2]:
        raise ValueError("predicted demand and serviceability horizons do not align")
    result = np.einsum("hv,hvf->f", weights, demand)
    if not np.all(np.isfinite(result)) or np.any(result < -1.0e-12):
        raise FloatingPointError("node-specific predicted demand is invalid")
    return np.maximum(result, 0.0).astype(np.float64)


def node_local_predicted_service_workload(
    origin_scenario: dict[str, Any],
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
    service_catalog: dict[str, Any],
    *,
    input_relative_jitter: float = 0.0,
    load_relative_jitter: float = 0.0,
    prediction_horizon_mask: Any = None,
) -> np.ndarray:
    """Build normalized ``[cache_head, service, (request,bits,cycles)]`` data.

    The request-mass channel calls ``node_specific_predicted_demand`` directly,
    preserving its prediction-scenario, horizon-discount, geometry and
    node-serviceability semantics.  Input and CPU work use the same scenario
    events and coverage weights.  Normalization scales are fixed by the
    configured forecast dimensions, service catalog, and jitter limits.
    """

    arrival = np.asarray(origin_scenario.get("arrival"), dtype=np.float64)
    service_id = np.asarray(origin_scenario.get("service_id"), dtype=np.int64)
    input_bits = np.asarray(origin_scenario.get("input_size"), dtype=np.float64)
    cpu_cycles = np.asarray(origin_scenario.get("cpu_cycles"), dtype=np.float64)
    positions = np.asarray(origin_scenario.get("positions_xy"), dtype=np.float64)
    if arrival.ndim != 3 or any(
        value.shape != arrival.shape
        for value in (service_id, input_bits, cpu_cycles)
    ):
        raise ValueError(
            "prediction scenario arrival/service_id/input_size/cpu_cycles "
            "must share [S,H,T] shape"
        )
    if positions.ndim != 4 or positions.shape[:2] != arrival.shape[:2]:
        raise ValueError("prediction positions_xy must align with [S,H]")
    if positions.shape[-1] != 2:
        raise ValueError("prediction positions_xy must end in coordinate dimension 2")
    if not 0.0 <= float(input_relative_jitter) < 1.0:
        raise ValueError("input_relative_jitter must be in [0,1)")
    if not 0.0 <= float(load_relative_jitter) < 1.0:
        raise ValueError("load_relative_jitter must be in [0,1)")

    scenario_count, horizon_count, task_vehicle_count = arrival.shape
    valid_horizon_mask = normalize_prediction_horizon_mask(
        prediction_horizon_mask, horizon_count
    )
    service_count = int(service_catalog["num_services"])
    vehicle_ids = [str(value) for value in origin_scenario["vehicle_ids"].tolist()]
    task_vehicle_ids_raw = origin_scenario.get("task_vehicle_ids")
    task_vehicle_ids = (
        [str(value) for value in task_vehicle_ids_raw.tolist()]
        if task_vehicle_ids_raw is not None
        else []
    )
    if positions.shape[2] != len(vehicle_ids) or len(task_vehicle_ids) != task_vehicle_count:
        raise ValueError("prediction scenario vehicle ID arrays do not align")
    vehicle_index = {vehicle_id: index for index, vehicle_id in enumerate(vehicle_ids)}
    try:
        task_to_vehicle = [vehicle_index[value] for value in task_vehicle_ids]
    except KeyError as exc:
        raise ValueError("prediction task references an unknown vehicle") from exc

    mean_input_by_service = np.asarray(
        service_catalog["mean_input_size_bit"], dtype=np.float64
    ).reshape(-1)
    cycles_per_bit = np.asarray(
        service_catalog["cycles_per_bit"], dtype=np.float64
    ).reshape(-1)
    if mean_input_by_service.shape != (service_count,) or cycles_per_bit.shape != (
        service_count,
    ):
        raise ValueError("service workload catalog dimensions are inconsistent")
    if np.any(mean_input_by_service < 0.0) or np.any(cycles_per_bit < 0.0):
        raise ValueError("service workload catalog scales must be nonnegative")

    horizon_discount = np.asarray(
        [float(config.beta_discount) ** horizon for horizon in range(horizon_count)],
        dtype=np.float64,
    )
    request_scale = max(
        float(task_vehicle_count) * float(np.sum(horizon_discount)), 1.0
    )
    max_input = float(np.max(mean_input_by_service)) * (
        1.0 + float(input_relative_jitter)
    )
    max_cycles_per_bit = float(np.max(cycles_per_bit)) * (
        1.0 + float(load_relative_jitter)
    )
    fixed_scales = np.asarray(
        [
            request_scale,
            request_scale * max(max_input, 1.0),
            request_scale * max(max_input * max_cycles_per_bit, 1.0),
        ],
        dtype=np.float64,
    )

    vehicle_workload_by_horizon = np.zeros(
        (horizon_count, len(vehicle_ids), service_count, 2), dtype=np.float64
    )
    for scenario_index in range(scenario_count):
        for horizon in range(horizon_count):
            if not valid_horizon_mask[horizon]:
                continue
            discount = horizon_discount[horizon]
            for task_index, full_vehicle_index in enumerate(task_to_vehicle):
                if arrival[scenario_index, horizon, task_index] <= 0.0:
                    continue
                service = int(service_id[scenario_index, horizon, task_index])
                if 0 <= service < service_count:
                    scale = discount / max(float(scenario_count), 1.0)
                    vehicle_workload_by_horizon[horizon, full_vehicle_index, service, 0] += (
                        scale * input_bits[scenario_index, horizon, task_index]
                    )
                    vehicle_workload_by_horizon[horizon, full_vehicle_index, service, 1] += (
                        scale * cpu_cycles[scenario_index, horizon, task_index]
                    )

    result = np.zeros(
        (len(cache_nodes["node_ids"]), service_count, 3), dtype=np.float64
    )
    for node_index in range(len(cache_nodes["node_ids"])):
        # Reuse the formal candidate demand helper for the count channel.
        result[node_index, :, 0] = node_specific_predicted_demand(
            origin_scenario,
            node_index,
            cache_nodes,
            config,
            prediction_horizon_mask=valid_horizon_mask,
        ) / fixed_scales[0]
        weights = _node_predicted_serviceability_weights(
            origin_scenario, node_index, cache_nodes, config
        )
        aggregated_work = np.einsum(
            "hv,hvfc->fc", weights, vehicle_workload_by_horizon
        )
        if not np.any(vehicle_workload_by_horizon):
            # Match the formal predicted-demand fallback when a prediction
            # artifact has no usable event records; catalog means provide the
            # fixed expected workload scale without using realized tasks.
            request_mass = result[node_index, :, 0] * fixed_scales[0]
            aggregated_work[:, 0] = request_mass * mean_input_by_service
            aggregated_work[:, 1] = (
                aggregated_work[:, 0] * cycles_per_bit
            )
        result[node_index, :, 1:] = aggregated_work / fixed_scales[1:]

    if not np.all(np.isfinite(result)) or np.any(result < -1.0e-12):
        raise FloatingPointError("node-local predicted workload is invalid")
    return np.maximum(result, 0.0).astype(np.float32)


def observed_local_service_usage(
    local_cache_state: dict[str, set[int]],
    num_services: int,
) -> np.ndarray:
    """Return a transparent observed local-replica usage proxy.

    The current environment does not expose a historical edge-node utility
    decomposition before proposal time.  This statistic therefore uses only
    observed vehicle-local replicas; it is a usage/availability proxy, not an
    exact counterfactual gain and is never used in the reward or Exact-Q.
    """

    vehicle_count = max(len(local_cache_state), 1)
    values = np.zeros(int(num_services), dtype=np.float64)
    for services in local_cache_state.values():
        for service in services:
            service_index = int(service)
            if 0 <= service_index < int(num_services):
                values[service_index] += 1.0
    return values / float(vehicle_count)


@dataclass(frozen=True)
class CacheTargetCandidate:
    """One deterministic, feasible target cache configuration for one node."""

    node_index: int
    node_id: str
    before_node_cache: np.ndarray
    target_node_cache: np.ndarray
    proposal_source: str


def has_legal_cache_target_configuration(
    current_cache_state: np.ndarray,
    node_index: int,
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
) -> bool:
    """Return whether one different capacity-feasible binary target exists.

    This is the cheap online Trigger mask check.  It is linear in the number
    of services and deliberately does not enumerate targets, inspect demand,
    call the surrogate, or evaluate the environment.  For a non-empty
    feasible row, removing any cached service is legal.  For an empty row, a
    different target exists exactly when at least one service fits.
    """

    raw_cache = np.asarray(current_cache_state)
    service_count = int(service_catalog["num_services"])
    node_count = len(cache_nodes["node_ids"])
    if raw_cache.shape != (node_count, service_count):
        raise ValueError("cache state shape does not match nodes/services")
    index = int(node_index)
    if index < 0 or index >= node_count:
        raise IndexError("cache target node_index 越界")
    before = np.asarray(raw_cache[index])
    if not np.all(np.isin(before, [0, 1])):
        raise ValueError("cache target row must be binary")
    if bool(np.any(before)):
        return True
    sizes = np.asarray(service_catalog["service_sizes_mb"], dtype=np.float64)
    if sizes.shape != before.shape or not np.all(np.isfinite(sizes)):
        raise ValueError("service sizes are not finite or shape-aligned")
    capacity = float(cache_nodes["capacities_mb"][index])
    if not math.isfinite(capacity) or capacity < 0.0:
        raise ValueError("cache capacity must be finite and non-negative")
    return bool(np.any(sizes <= capacity + 1.0e-9))


def _greedy_target_configuration(
    scores: np.ndarray,
    service_sizes_mb: np.ndarray,
    capacity_mb: float,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    sizes = np.asarray(service_sizes_mb, dtype=np.float64).reshape(-1)
    if scores.shape != sizes.shape or not np.all(np.isfinite(scores)):
        raise ValueError("target proposal scores/service sizes are not aligned")
    density = scores / np.maximum(sizes, 1.0e-12)
    order = sorted(
        range(len(scores)),
        key=lambda index: (-float(density[index]), -float(scores[index]), int(index)),
    )
    target = np.zeros(len(scores), dtype=np.int8)
    used = 0.0
    for service in order:
        size = float(sizes[service])
        if size >= 0.0 and used + size <= float(capacity_mb) + 1.0e-9:
            target[service] = 1
            used += size
    return target


def enumerate_all_cache_target_configurations(
    current_cache_state: np.ndarray,
    node_index: int,
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    origin_scenario: dict[str, Any],
    local_cache_state: dict[str, set[int]],
    config: CacheOptimizationConfig,
    *,
    prediction_horizon_mask: Any = None,
) -> list[CacheTargetCandidate]:
    """Enumerate every non-KEEP capacity-feasible binary target configuration.

    This routine performs only bit enumeration and capacity checks. It does not
    evaluate the environment, Matching, KKT, or Exact-Q.
    """

    current_cache = np.asarray(current_cache_state, dtype=np.int8)
    validate_cache_shape_and_capacity(current_cache, service_catalog, cache_nodes)
    index = int(node_index)
    if index < 0 or index >= current_cache.shape[0]:
        raise IndexError("cache target node_index 越界")
    sizes = np.asarray(service_catalog["service_sizes_mb"], dtype=np.float64)
    before = current_cache[index].copy()
    capacity = float(cache_nodes["capacities_mb"][index])
    demand = node_specific_predicted_demand(
        origin_scenario,
        index,
        cache_nodes,
        config,
        prediction_horizon_mask=prediction_horizon_mask,
    )
    service_count = int(service_catalog["num_services"])
    if service_count > 20:
        raise ValueError("exhaustive Cache proposal enumeration is capped at 20 services")
    observed_usage = observed_local_service_usage(local_cache_state, service_count)
    replica_counts = np.sum(current_cache, axis=0).astype(np.float64)
    mean_demand = float(np.mean(demand)) if len(demand) else 0.0
    prediction_target = _greedy_target_configuration(demand, sizes, capacity)
    replica_scores = demand + (mean_demand + 1.0e-12) / (1.0 + replica_counts)
    replica_target = _greedy_target_configuration(replica_scores, sizes, capacity)
    anchor_sources: dict[bytes, str] = {}

    def add_anchor(source: str, target: np.ndarray) -> None:
        candidate = np.asarray(target, dtype=np.int8).reshape(-1)
        if candidate.shape != before.shape or not np.all(np.isin(candidate, [0, 1])):
            raise ValueError("target cache proposal must be binary and shape-aligned")
        if np.array_equal(candidate, before) or not is_node_feasible(candidate, sizes, capacity):
            return
        anchor_sources.setdefault(np.ascontiguousarray(candidate).tobytes(), source)

    add_anchor("prediction", prediction_target)
    add_anchor("coverage_anchor", replica_target)
    for fraction in config.target_intermediate_fractions:
        blended_scores = (
            float(fraction) * demand
            + (1.0 - float(fraction)) * (before.astype(np.float64) * (mean_demand + 1.0e-12))
        )
        add_anchor(
            "intermediate",
            _greedy_target_configuration(blended_scores, sizes, capacity),
        )
    for _, target in generate_node_neighbors(before, sizes, capacity):
        add_anchor("low_reconfiguration", np.asarray(target, dtype=np.int8))

    candidates: list[CacheTargetCandidate] = []
    for bitset in range(1 << service_count):
        target = np.fromiter(
            (((bitset >> service) & 1) for service in range(service_count)),
            dtype=np.int8,
            count=service_count,
        )
        if np.array_equal(target, before) or not is_node_feasible(target, sizes, capacity):
            continue
        key = np.ascontiguousarray(target).tobytes()
        candidates.append(
            CacheTargetCandidate(
                node_index=index,
                node_id=str(cache_nodes["node_ids"][index]),
                before_node_cache=before.copy(),
                target_node_cache=target,
                proposal_source=anchor_sources.get(key, "exhaustive_feasible"),
            )
        )
    return candidates


def select_cache_target_configurations(
    candidates: list[CacheTargetCandidate],
    current_node_cache: np.ndarray,
    capacity: int,
    *,
    predicted_delta_q: np.ndarray | None = None,
    predicted_sigma: np.ndarray | None = None,
    diversity_reserve: int = 0,
    required_indices: tuple[int, ...] = (),
) -> list[CacheTargetCandidate]:
    """Select a bounded proposal set with anchors, surrogate ranks, and diversity."""

    source = list(candidates)
    if not source:
        return []
    limit = int(capacity)
    if limit <= 0:
        raise ValueError("target candidate capacity must be positive")
    reserve = int(diversity_reserve)
    if reserve < 0 or reserve >= limit:
        raise ValueError("candidate diversity reserve must be in [0, capacity)")
    if predicted_delta_q is not None:
        means = np.asarray(predicted_delta_q, dtype=np.float64).reshape(-1)
        if means.shape != (len(source),) or not np.all(np.isfinite(means)):
            raise ValueError("surrogate candidate means are not aligned and finite")
    else:
        means = None
    if predicted_sigma is not None:
        sigmas = np.asarray(predicted_sigma, dtype=np.float64).reshape(-1)
        if sigmas.shape != (len(source),) or not np.all(np.isfinite(sigmas)):
            raise ValueError("surrogate candidate uncertainty is not aligned and finite")
        sigmas = np.maximum(sigmas, 0.0)
    else:
        sigmas = None
    current = np.asarray(current_node_cache, dtype=np.int8).reshape(-1)
    if any(np.asarray(item.target_node_cache).shape != current.shape for item in source):
        raise ValueError("candidate target width does not match current node cache")

    selected: dict[int, str] = {}

    def add(index: int, label: str | None = None) -> None:
        if index < 0 or index >= len(source):
            raise IndexError("required Cache candidate index is out of range")
        if len(selected) >= limit or index in selected:
            return
        selected[index] = label or source[index].proposal_source

    # Hybrid-confirmed and production-selected targets cannot be displaced by
    # anchors when the policy tensor reaches capacity.
    for index in dict.fromkeys(int(value) for value in required_indices):
        add(index, "required")

    # Preserve all demand/replica/intermediate anchors before filling the set.
    anchor_priority = {"prediction": 0, "coverage_anchor": 1, "intermediate": 2}
    anchors = sorted(
        (
            index
            for index, item in enumerate(source)
            if item.proposal_source in anchor_priority
        ),
        key=lambda index: (
            anchor_priority[source[index].proposal_source],
            np.ascontiguousarray(source[index].target_node_cache).tobytes(),
        ),
    )
    for index in anchors:
        add(index)

    # Protect the eight closest feasible targets from the reconfiguration cost.
    hamming = np.asarray(
        [np.count_nonzero(np.asarray(item.target_node_cache) != current) for item in source],
        dtype=np.int64,
    )
    low_rank = sorted(
        range(len(source)),
        key=lambda index: (
            int(hamming[index]),
            np.ascontiguousarray(source[index].target_node_cache).tobytes(),
        ),
    )
    for index in low_rank[: min(8, limit)]:
        add(index, "low_reconfiguration")

    if means is not None and sigmas is not None:
        high_value_count = min(36, max(1, limit // 2 + 4 - reserve))
        ucb_count = min(8, max(1, limit // 8))
        uncertainty_count = min(8, max(1, limit // 8))
        ucb = means + 2.0 * sigmas
        for index in np.argsort(-means, kind="mergesort")[:high_value_count]:
            add(int(index), "surrogate_high_value")
        for index in np.argsort(-ucb, kind="mergesort")[:ucb_count]:
            add(int(index), "surrogate_ucb")
        for index in np.argsort(-sigmas, kind="mergesort")[:uncertainty_count]:
            add(int(index), "surrogate_uncertainty")

        # Keep one surrogate-ranked target from each feasible Hamming shell.
        # This protects multi-service replacements that can rank outside the
        # global top-μ/UCB lists while leaving remaining slots for farthest-point
        # completion below.
        for distance in sorted(set(int(value) for value in hamming if value > 0)):
            shell = np.flatnonzero(hamming == distance)
            if not len(shell):
                continue
            shell_best = max(
                (int(index) for index in shell),
                key=lambda index: (
                    float(means[index]),
                    tuple(-int(value) for value in source[index].target_node_cache),
                ),
            )
            add(shell_best, "surrogate_hamming_shell")

    # Deterministic Hamming farthest-point completion is policy-independent.
    while len(selected) < min(limit, len(source)):
        available = [index for index in range(len(source)) if index not in selected]
        if not available:
            break
        selected_targets = [source[index].target_node_cache for index in selected]
        next_index = max(
            available,
            key=lambda index: (
                min(
                    int(np.count_nonzero(source[index].target_node_cache != target))
                    for target in selected_targets
                ),
                tuple(-int(value) for value in source[index].target_node_cache),
            ),
        )
        add(next_index, "hamming_diversity")

    return [replace(source[index], proposal_source=label) for index, label in selected.items()]




def apply_cache_target_configuration(
    current_cache_state: np.ndarray,
    node_index: int,
    target_node_cache: np.ndarray,
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
) -> dict[str, Any]:
    """Validate/apply a target cache configuration selected after UPDATE."""

    current_cache = np.asarray(current_cache_state, dtype=np.int8)
    validate_cache_shape_and_capacity(current_cache, service_catalog, cache_nodes)
    index = int(node_index)
    before = current_cache[index].copy()
    target = np.asarray(target_node_cache, dtype=np.int8).reshape(-1)
    if target.shape != before.shape or not np.all(np.isin(target, [0, 1])):
        raise ValueError("target cache configuration must be binary and shape-aligned")
    if np.array_equal(target, before):
        raise ValueError("UPDATE target configuration must differ from KEEP state")
    if not is_node_feasible(
        target,
        np.asarray(service_catalog["service_sizes_mb"], dtype=np.float64),
        float(cache_nodes["capacities_mb"][index]),
    ):
        raise ValueError("target cache configuration violates node capacity")
    current_cache[index] = target
    return {
        "node_index": index,
        "node_id": str(cache_nodes["node_ids"][index]),
        "before_node_cache": before,
        "target_node_cache": current_cache[index].copy(),
        "actual_changed": True,
        "conditional_action": "TARGET_CONFIGURATION",
    }










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






# ============================================================================
# optimizer API
# ============================================================================




# ============================================================================
# Deterministic previous cache for minimal offline validation
# ============================================================================




def popularity_aware_initial_cache(
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
) -> np.ndarray:
    """生成正式 episode 的 uniform feasible 初始缓存 C(0)。

    保留该函数名以兼容现有 formal caller；其当前语义是均匀可行初始化，
    不再读取 popularity、prediction 或 request。每个节点使用 seed 驱动的
    局部 RNG 对全部 service 做均匀随机排列，然后按容量逐个装入；因此每
    个 service 在选择顺序中具有相同概率，并且最终 cache 严格满足容量约束。
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

    rng = np.random.default_rng(int(config.initial_cache_seed))
    cache = np.zeros(
        (len(cache_nodes["node_ids"]), num_services),
        dtype=np.int8,
    )
    for node_index, capacity in enumerate(cache_nodes["capacities_mb"]):
        remaining = float(capacity)
        for service_id in rng.permutation(num_services).tolist():
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
                    "uniform initial cache is not maximal feasible"
                )

    validate_cache_shape_and_capacity(cache, service_catalog, cache_nodes)
    return cache


# ============================================================================
# Minimal validation
# ============================================================================




# ============================================================================
# CLI
# ============================================================================
