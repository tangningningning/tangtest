#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trigger-conditioned cache configuration optimization.

本阶段只解决：给定上一时隙缓存状态、cache trigger 和预测场景后，选择新的
动态缓存配置。预测候选复用实际执行侧的 Matching/KKT 和所配置的 Cloud 资源合同，
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
    sv_cache_backhaul_rate_bps: float
    rsu_cache_backhaul_rate_bps: float
    uav_cache_backhaul_rate_bps: float
    beta_discount: float
    omega_t: float
    omega_e: float
    time_ref_s: float
    energy_ref_j: float
    sv_cpu_cycles_per_s: float
    rsu_cpu_cycles_per_s: float
    uav_cpu_cycles_per_s: float
    # Legacy Cloud CPU aliases remain available for frozen run configs.
    cloud_cpu_cycles_per_s: float
    cloud_total_cpu_cycles_per_s: float
    cloud_resource_mode: str
    cloud_cpu_capacity_cycles_per_s: float
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
    # Compatibility fields: Matching uses cloud_total_bandwidth_bps as the
    # shared per-slot equivalent rate cap; cloud_uplink_rate_bps is its alias.
    cloud_uplink_rate_bps: float
    cloud_total_bandwidth_bps: float
    cloud_delay_factor: float
    # Retained as a legacy config field; the shared total-rate constraint is
    # enforced regardless of this flag's value.
    cloud_resource_sharing: bool
    cloud_fallback_mode: str
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
    # Proposed-policy target configuration proposal controls.  These are
    # engineering bounds/coverage knobs, not physical system parameters.
    target_candidate_capacity: int = 64
    target_intermediate_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    noise_model: str = system_model.FIXED_TOTAL_NOISE_V0
    noise_psd_w_per_hz: float | None = None
    receiver_noise_figure_db: float | None = None

    def validate(self) -> None:
        values = [
            self.num_rsu,
            self.num_uav,
            len(self.rsu_instances),
            len(self.uav_instances),
            self.sv_capacity_mb,
            self.rsu_capacity_mb,
            self.uav_capacity_mb,
            self.sv_cache_backhaul_rate_bps,
            self.rsu_cache_backhaul_rate_bps,
            self.uav_cache_backhaul_rate_bps,
            self.time_ref_s,
            self.energy_ref_j,
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
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in values
        ):
            raise ValueError("缓存优化配置存在非正数或非有限参数。")
        if not 0.0 < self.beta_discount <= 1.0:
            raise ValueError("beta_discount 必须位于 (0, 1]。")
        if self.tolerance < 0.0:
            raise ValueError("tolerance 不能为负数。")
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
        if self.cloud_uplink_rate_bps != self.cloud_total_bandwidth_bps:
            raise ValueError(
                "Cloud uplink alias 必须与每时隙共享总速率上限保持数值一致。"
            )
        if not isinstance(self.cloud_resource_sharing, bool):
            raise ValueError("cloud_resource_sharing 必须是显式 bool。")
        if self.cloud_resource_mode not in {"legacy_unlimited", "finite_pool"}:
            raise ValueError(
                "cloud_resource_mode 必须是 legacy_unlimited 或 finite_pool。"
            )
        if self.cloud_resource_mode == "finite_pool" and (
            not math.isfinite(self.cloud_cpu_capacity_cycles_per_s)
            or self.cloud_cpu_capacity_cycles_per_s <= 0.0
        ):
            raise ValueError(
                "finite_pool 模式要求正的 cloud_cpu_capacity_cycles_per_s。"
            )
        if (
            self.cloud_resource_mode == "finite_pool"
            and self.cloud_cpu_capacity_cycles_per_s
            != self.cloud_total_cpu_cycles_per_s
        ):
            raise ValueError(
                "finite_pool 模式下 Cloud CPU 容量必须与兼容 total_cpu 字段一致。"
            )
        if self.cloud_fallback_mode not in {
            "LEGACY_EDGE_GATE",
            "RESOURCE_AWARE_FALLBACK",
        }:
            raise ValueError("cloud_fallback_mode 无效。")
        if self.noise_model not in {
            system_model.FIXED_TOTAL_NOISE_V0,
            system_model.BANDWIDTH_DEPENDENT_NOISE_V1,
        }:
            raise ValueError(f"noise_model 无效：{self.noise_model!r}")
        if self.noise_model == system_model.BANDWIDTH_DEPENDENT_NOISE_V1:
            if (
                self.noise_psd_w_per_hz is None
                or not math.isfinite(float(self.noise_psd_w_per_hz))
                or float(self.noise_psd_w_per_hz) <= 0.0
            ):
                raise ValueError(
                    "BANDWIDTH_DEPENDENT_NOISE_V1 requires a positive finite noise PSD"
                )
            if (
                self.receiver_noise_figure_db is None
                or not math.isfinite(float(self.receiver_noise_figure_db))
                or float(self.receiver_noise_figure_db) < 0.0
            ):
                raise ValueError(
                    "BANDWIDTH_DEPENDENT_NOISE_V1 requires a finite nonnegative receiver NF"
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
            "缺少或非法的兼容字段 nodes.cloud.resource_sharing；"
            "Cloud 上行总池始终按共享容量分配，该字段不控制上行共享。"
        )
    backhaul_rates = {}
    for role, key in (
        ("SV", "nodes.sv.cache_backhaul_rate_bps"),
        ("RSU", "nodes.rsu.cache_backhaul_rate_bps"),
        ("UAV", "nodes.uav.cache_backhaul_rate_bps"),
    ):
        raw_rate = get(config, key)
        if raw_rate is None:
            raise ValueError(
                f"缺少必需配置 {key}；Cache 下载时延成本不允许静默默认速率。"
            )
        rate = float(raw_rate)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError(f"{key} 必须是正的有限 bit/s 数值。")
        backhaul_rates[role] = rate
    return CacheOptimizationConfig(
        num_rsu=int(get(config, "nodes.rsu.count")),
        num_uav=int(get(config, "nodes.uav.count")),
        rsu_instances=rsu_instances,
        uav_instances=uav_instances,
        sv_capacity_mb=float(get(config, "nodes.sv.cache_capacity_mb")),
        rsu_capacity_mb=float(get(config, "nodes.rsu.cache_capacity_mb")),
        uav_capacity_mb=float(get(config, "nodes.uav.cache_capacity_mb")),
        sv_cache_backhaul_rate_bps=backhaul_rates["SV"],
        rsu_cache_backhaul_rate_bps=backhaul_rates["RSU"],
        uav_cache_backhaul_rate_bps=backhaul_rates["UAV"],
        beta_discount=float(get(config, "cache.discount_beta")),
        omega_t=float(get(config, "objective.omega_delay")),
        omega_e=float(get(config, "objective.omega_energy")),
        time_ref_s=float(get(config, "objective.T_ref")),
        energy_ref_j=float(get(config, "objective.E_ref")),
        sv_cpu_cycles_per_s=float(get(config, "nodes.sv.cpu_cycles_per_s")),
        rsu_cpu_cycles_per_s=float(get(config, "nodes.rsu.cpu_cycles_per_s")),
        uav_cpu_cycles_per_s=float(get(config, "nodes.uav.cpu_cycles_per_s")),
        cloud_cpu_cycles_per_s=float(get(config, "nodes.cloud.cpu_cycles_per_s")),
        cloud_total_cpu_cycles_per_s=float(get(config, "nodes.cloud.total_cpu_cycles_per_s")),
        cloud_resource_mode=str(
            get(config, "nodes.cloud.resource_mode", "legacy_unlimited")
        ),
        cloud_cpu_capacity_cycles_per_s=float(
            get(config, "nodes.cloud.cpu_capacity_cycles_per_s", 0.0)
        ),
        tx_power_w=float(get(config, "nodes.vehicle.transmit_power_w")),
        sv_bandwidth_hz=float(get(config, "nodes.sv.bandwidth_hz")),
        rsu_bandwidth_hz=float(get(config, "nodes.rsu.bandwidth_hz")),
        uav_bandwidth_hz=float(get(config, "nodes.uav.bandwidth_hz")),
        noise_power_w=float(get(config, "communication.noise.noise_power_w")),
        noise_model=str(
            get(
                config,
                "communication.noise.model",
                system_model.FIXED_TOTAL_NOISE_V0,
            )
        ),
        noise_psd_w_per_hz=(
            None
            if get(config, "communication.noise.psd_dbm_per_hz") is None
            else system_model.dbm_per_hz_to_w_per_hz(
                float(get(config, "communication.noise.psd_dbm_per_hz"))
            )
        ),
        receiver_noise_figure_db=(
            None
            if get(config, "communication.noise.receiver_noise_figure_db") is None
            else float(get(config, "communication.noise.receiver_noise_figure_db"))
        ),
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
        cloud_fallback_mode=str(
            get(
                config,
                "algorithm.matching.cloud_fallback_mode",
                "LEGACY_EDGE_GATE",
            )
        ),
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
    backhaul_rates = []
    for role in node_roles:
        if role == "SV":
            capacities.append(config.sv_capacity_mb)
            backhaul_rates.append(config.sv_cache_backhaul_rate_bps)
        elif role == "RSU":
            capacities.append(config.rsu_capacity_mb)
            backhaul_rates.append(config.rsu_cache_backhaul_rate_bps)
        elif role == "UAV":
            capacities.append(config.uav_capacity_mb)
            backhaul_rates.append(config.uav_cache_backhaul_rate_bps)
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
        "cache_backhaul_rate_bps": np.asarray(backhaul_rates, dtype=np.float64),
        "fixed_positions": fixed_positions,
    }


def initialize_cache_cost_normalization(
    cache_nodes: dict[str, Any],
    service_sizes_mb: np.ndarray,
    reference_s_per_node_s: float,
) -> dict[str, Any]:
    """Fix a positive per-node cache-delay design scale at environment setup.

    Every node in ``cache_nodes`` is a policy-controlled cache head in the
    formal multi-node action schema. Each node uses the configured fixed
    reference scale, independent of capacity, policy actions, and Slot state.
    This is a normalization scale, not a physical upper bound on download delay.
    """

    sizes = np.asarray(service_sizes_mb, dtype=np.float64).reshape(-1)
    capacities = np.asarray(cache_nodes["capacities_mb"], dtype=np.float64).reshape(-1)
    rates = np.asarray(
        cache_nodes["cache_backhaul_rate_bps"], dtype=np.float64
    ).reshape(-1)
    node_ids = list(cache_nodes["node_ids"])
    if sizes.ndim != 1 or not np.isfinite(sizes).all():
        raise ValueError("service_sizes_mb 必须是有限的一维数组。")
    if np.any(sizes < 0.0):
        raise ValueError("service_sizes_mb 不能为负数。")
    if (
        capacities.shape != (len(node_ids),)
        or rates.shape != (len(node_ids),)
        or not np.isfinite(capacities).all()
        or not np.isfinite(rates).all()
        or np.any(capacities <= 0.0)
        or np.any(rates <= 0.0)
    ):
        raise ValueError("Cache 节点容量必须为正，实际回传速率必须为正有限数。")

    reference_scale_s = float(reference_s_per_node_s)
    if not math.isfinite(reference_scale_s) or reference_scale_s <= 0.0:
        raise ValueError("Cache 每节点归一化参考时延必须为有限正数。")

    reference_s_by_node = np.full(len(node_ids), reference_scale_s, dtype=np.float64)
    capacity_upper_bound_s_by_node = capacities * 8.0e6 / rates
    reference_s_total = float(np.sum(reference_s_by_node))
    reference_kind = "fixed_per_node_design_normalization_scale"
    if (
        not np.isfinite(reference_s_by_node).all()
        or np.any(reference_s_by_node <= 0.0)
        or not np.isfinite(capacity_upper_bound_s_by_node).all()
        or np.any(capacity_upper_bound_s_by_node <= 0.0)
        or not math.isfinite(reference_s_total)
        or reference_s_total <= 0.0
    ):
        raise FloatingPointError("per-node cache delay references are invalid")
    metadata = {
        "reference_s_by_node": [float(value) for value in reference_s_by_node],
        "reference_s_per_node": reference_scale_s,
        "capacity_upper_bound_s_by_node": [
            float(value) for value in capacity_upper_bound_s_by_node
        ],
        # Aggregate retained for diagnostics only; every node has its own divisor.
        "reference_s_total": reference_s_total,
        "reference_kind": reference_kind,
        "formula": "T_ca_m_ref_s=objective.cost_normalization.cache_reference_s_per_node",
        "cost_formula": "C_ca=sum_m(T_ca_m_s/T_ca_m_ref_s)",
        "additional_cost_multiplier": "none",
        "service_count": int(sizes.size),
        "cache_node_count": int(len(node_ids)),
        "unit": "s",
    }
    cache_nodes["cache_cost_normalization"] = metadata
    return dict(metadata)






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
    result = cache_update_cost_details(
        previous_cache,
        new_cache,
        cache_trigger,
        service_sizes_mb,
        cache_nodes,
        config,
    )
    return float(result["cache_cost"]), np.asarray(
        result["cache_cost_by_node"], dtype=np.float64
    )


def cache_update_cost_details(
    previous_cache: np.ndarray,
    new_cache: np.ndarray,
    cache_trigger: np.ndarray,
    service_sizes_mb: np.ndarray,
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
) -> dict[str, Any]:
    """Expose download delays and costs without duplicating the formal formula."""

    normalization = cache_nodes.get("cache_cost_normalization")
    if not isinstance(normalization, dict):
        raise RuntimeError(
            "Cache 成本归一化参考量必须在环境初始化时固定。"
        )

    return system_model.cache_update_cost(
        previous_cache,
        new_cache,
        cache_trigger,
        service_sizes_mb,
        cache_nodes["cache_backhaul_rate_bps"],
        np.asarray(normalization["reference_s_by_node"], dtype=np.float64),
    )












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


def predicted_cache_coverage_ratio(
    cache_state: Any,
    origin_scenario: dict[str, Any],
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
    *,
    prediction_horizon_mask: Any = None,
) -> float | None:
    """计算预测场景中的可服务缓存需求覆盖率，不代表实际任务命中率。"""

    cache = np.asarray(cache_state, dtype=np.int8)
    service_count = int(np.asarray(origin_scenario["rho_hat"]).shape[-1])
    if cache.shape != (len(cache_nodes["node_ids"]), service_count) or not np.all(
        np.isin(cache, [0, 1])
    ):
        raise ValueError("缓存状态必须与预测场景节点/服务维度一致")
    demand = predicted_vehicle_service_demand_by_horizon(
        origin_scenario,
        service_count,
        beta_discount=config.beta_discount,
        prediction_horizon_mask=prediction_horizon_mask,
    )
    if demand.ndim != 3 or demand.shape[-1] != service_count:
        raise ValueError("预测服务需求张量形状非法")
    coverage_weight = np.zeros_like(demand, dtype=np.float64)
    for node_index in range(cache.shape[0]):
        serviceability = _node_predicted_serviceability_weights(
            origin_scenario,
            node_index,
            cache_nodes,
            config,
        )
        node_weight = serviceability[:, :, None] * cache[node_index][None, None, :]
        coverage_weight = np.maximum(coverage_weight, node_weight)
    total_demand = float(np.sum(demand))
    if total_demand <= 1.0e-12:
        return None
    covered_demand = float(np.sum(demand * np.clip(coverage_weight, 0.0, 1.0)))
    return covered_demand / total_demand


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


@dataclass(frozen=True)
class CacheTargetCandidate:
    """One deterministic, feasible target cache configuration for one node."""

    node_index: int
    node_id: str
    before_node_cache: np.ndarray
    target_node_cache: np.ndarray
    proposal_source: str


def is_legal_cache_replacement(before_node_cache: Any, target_node_cache: Any) -> bool:
    """Return whether a target both removes an old service and adds a new one."""

    before = np.asarray(before_node_cache)
    target = np.asarray(target_node_cache)
    if (
        before.shape != target.shape
        or not np.all(np.isin(before, [0, 1]))
        or not np.all(np.isin(target, [0, 1]))
    ):
        return False
    removed = np.any((before == 1) & (target == 0))
    added = np.any((before == 0) & (target == 1))
    return bool(removed and added)


def has_legal_cache_target_configuration(
    current_cache_state: np.ndarray,
    node_index: int,
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    *,
    locked_services: Any = None,
) -> bool:
    """Return whether at least one capacity-feasible replacement preserves locks."""

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
    sizes = np.asarray(service_catalog["service_sizes_mb"], dtype=np.float64)
    if sizes.shape != before.shape or not np.all(np.isfinite(sizes)):
        raise ValueError("service sizes are not finite or shape-aligned")
    capacity = float(cache_nodes["capacities_mb"][index])
    if not math.isfinite(capacity) or capacity < 0.0:
        raise ValueError("cache capacity must be finite and non-negative")
    if not is_node_feasible(before, sizes, capacity):
        raise ValueError("current Cache row violates node capacity")
    if locked_services is None:
        locked = np.zeros(before.shape, dtype=bool)
    else:
        raw_locked = np.asarray(locked_services)
        if raw_locked.shape != before.shape or not np.all(np.isin(raw_locked, [0, 1, False, True])):
            raise ValueError("locked services must be a binary vector aligned with catalog order")
        locked = raw_locked.astype(bool, copy=False)
    if np.any(locked & (before != 1)):
        raise ValueError("an active service lock is absent from the current Cache row")
    if not bool(np.any((before == 1) & ~locked)):
        return False
    used_locked = float(np.sum(sizes[locked]))
    remaining = capacity - used_locked
    return bool(
        np.any(
            (before == 0)
            & (sizes <= remaining + 1.0e-9)
        )
    )


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
    config: CacheOptimizationConfig,
    *,
    prediction_horizon_mask: Any = None,
    locked_services: Any = None,
    generation_diagnostics: dict[str, Any] | None = None,
) -> list[CacheTargetCandidate]:
    """Enumerate every capacity-feasible, lock-preserving legal REPLACE target.

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
    service_count = int(service_catalog["num_services"])
    if locked_services is None:
        locked = np.zeros(before.shape, dtype=bool)
    else:
        raw_locked = np.asarray(locked_services)
        if raw_locked.shape != before.shape or not np.all(np.isin(raw_locked, [0, 1, False, True])):
            raise ValueError("locked services must be a binary vector aligned with catalog order")
        locked = raw_locked.astype(bool, copy=False)
    if np.any(locked & (before != 1)):
        raise ValueError("an active service lock is absent from the current Cache row")

    # Establish the complete legal target set before demand features, candidate
    # scoring, sorting, or bounded candidate selection are performed.
    legal_targets: dict[bytes, np.ndarray] = {}
    capacity_feasible_count = 0
    legal_replace_count = 0
    for bitset in range(1 << service_count):
        target = np.fromiter(
            (((bitset >> service) & 1) for service in range(service_count)),
            dtype=np.int8,
            count=service_count,
        )
        capacity_feasible = is_node_feasible(target, sizes, capacity)
        capacity_feasible_count += int(capacity_feasible)
        if (
            not capacity_feasible
            or np.any(locked & (target != 1))
            or not is_legal_cache_replacement(before, target)
        ):
            continue
        legal_replace_count += 1
        legal_targets[np.ascontiguousarray(target).tobytes()] = target
    if generation_diagnostics is not None:
        generation_diagnostics.update(
            {
                "raw_combination_count": int(1 << service_count),
                "capacity_feasible_count": int(capacity_feasible_count),
                "legal_replace_count": int(legal_replace_count),
                "unique_legal_replace_count": int(len(legal_targets)),
                "deduplicated_legal_replace_count": int(
                    legal_replace_count - len(legal_targets)
                ),
                "truncated_legal_replace_count": 0,
                "enumeration_complete": True,
            }
        )
    if not legal_targets:
        return []

    demand = node_specific_predicted_demand(
        origin_scenario,
        index,
        cache_nodes,
        config,
        prediction_horizon_mask=prediction_horizon_mask,
    )
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
        key = np.ascontiguousarray(candidate).tobytes()
        if key not in legal_targets:
            return
        anchor_sources.setdefault(key, source)

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
    for key, target in legal_targets.items():
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


def generate_bounded_cache_target_configurations(
    current_cache_state: np.ndarray,
    node_index: int,
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    origin_scenario: dict[str, Any],
    config: CacheOptimizationConfig,
    *,
    prediction_horizon_mask: Any = None,
    locked_services: Any = None,
    max_candidates: int | None = None,
) -> list[CacheTargetCandidate]:
    """Generate bounded feasible replacement proposals without enumerating 2**F states.

    The proposal budget is O(K) targets and O(KF) descriptor work (plus an
    O(F log F) demand sort), where K is the configured per-node candidate
    capacity. It keeps demand/replica anchors and a bounded set of
    high-demand-for-low-demand replacements. KEEP is supplied separately by
    the policy action builder. This is a candidate generator, not an optimizer
    over every feasible cache configuration.
    """

    current = np.asarray(current_cache_state, dtype=np.int8)
    validate_cache_shape_and_capacity(current, service_catalog, cache_nodes)
    index = int(node_index)
    if index < 0 or index >= current.shape[0]:
        raise IndexError("cache target node_index 越界")
    before = current[index].copy()
    sizes = np.asarray(service_catalog["service_sizes_mb"], dtype=np.float64).reshape(-1)
    service_count = int(service_catalog["num_services"])
    if sizes.shape != (service_count,) or not np.isfinite(sizes).all() or np.any(sizes < 0.0):
        raise ValueError("service sizes must be finite, nonnegative, and catalog-aligned")
    limit = int(config.target_candidate_capacity if max_candidates is None else max_candidates)
    if limit <= 0:
        raise ValueError("bounded candidate budget must be positive")
    limit = min(limit, int(config.target_candidate_capacity))
    capacity = float(cache_nodes["capacities_mb"][index])
    if locked_services is None:
        locked = np.zeros(service_count, dtype=bool)
    else:
        raw_locked = np.asarray(locked_services)
        if raw_locked.shape != (service_count,) or not np.all(np.isin(raw_locked, [0, 1, False, True])):
            raise ValueError("locked services must be a binary vector aligned with catalog order")
        locked = raw_locked.astype(bool, copy=False)
    if np.any(locked & (before != 1)):
        raise ValueError("an active service lock is absent from the current Cache row")

    demand = np.asarray(
        node_specific_predicted_demand(
            origin_scenario,
            index,
            cache_nodes,
            config,
            prediction_horizon_mask=prediction_horizon_mask,
        ),
        dtype=np.float64,
    ).reshape(-1)
    if demand.shape != (service_count,) or not np.isfinite(demand).all() or np.any(demand < 0.0):
        raise ValueError("predicted per-service demand is invalid")
    replica_counts = np.sum(current, axis=0).astype(np.float64)
    mean_demand = float(np.mean(demand)) if len(demand) else 0.0
    used_locked = float(np.sum(sizes[locked]))
    if used_locked > capacity + 1.0e-9:
        raise ValueError("locked services exceed node cache capacity")

    def greedy_target(scores: np.ndarray) -> np.ndarray:
        target = locked.astype(np.int8, copy=True)
        used = used_locked
        candidates = [service for service in range(service_count) if not locked[service]]
        candidates.sort(
            key=lambda service: (
                -float(scores[service] / max(float(sizes[service]), 1.0e-12)),
                -float(scores[service]),
                int(service),
            )
        )
        for service in candidates:
            if used + float(sizes[service]) <= capacity + 1.0e-9:
                target[service] = 1
                used += float(sizes[service])
        return target

    proposals: list[CacheTargetCandidate] = []
    seen: set[bytes] = set()

    def add_target(target: Any, source: str) -> None:
        if len(proposals) >= limit:
            return
        value = np.asarray(target, dtype=np.int8).reshape(-1)
        if (
            value.shape != before.shape
            or not np.all(np.isin(value, [0, 1]))
            or np.any(locked & (value != 1))
            or not is_legal_cache_replacement(before, value)
            or not is_node_feasible(value, sizes, capacity)
        ):
            return
        key = np.ascontiguousarray(value).tobytes()
        if key in seen:
            return
        seen.add(key)
        proposals.append(
            CacheTargetCandidate(
                node_index=index,
                node_id=str(cache_nodes["node_ids"][index]),
                before_node_cache=before.copy(),
                target_node_cache=value.copy(),
                proposal_source=source,
            )
        )

    add_target(greedy_target(demand), "prediction")
    replica_scores = demand + (mean_demand + 1.0e-12) / (1.0 + replica_counts)
    add_target(greedy_target(replica_scores), "coverage_anchor")
    for fraction in config.target_intermediate_fractions:
        if len(proposals) >= limit:
            break
        blended = float(fraction) * demand + (1.0 - float(fraction)) * (
            before.astype(np.float64) * (mean_demand + 1.0e-12)
        )
        add_target(greedy_target(blended), "intermediate")

    removable = [service for service in range(service_count) if before[service] and not locked[service]]
    removable.sort(key=lambda service: (float(demand[service]), -float(sizes[service]), int(service)))
    addable = [service for service in range(service_count) if not before[service] and sizes[service] <= capacity - used_locked + 1.0e-9]
    addable.sort(
        key=lambda service: (
            -float(demand[service] / max(float(sizes[service]), 1.0e-12)),
            -float(demand[service]),
            int(service),
        )
    )
    swap_budget = max(1, min(8, limit))
    for remove_service in removable[:swap_budget]:
        for add_service in addable[:swap_budget]:
            if len(proposals) >= limit:
                break
            target = before.copy()
            target[remove_service] = 0
            target[add_service] = 1
            add_target(target, "prediction_swap")
        if len(proposals) >= limit:
            break
    return proposals


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
    if not is_legal_cache_replacement(before, target):
        raise ValueError(
            "UPDATE target configuration must remove at least one cached service "
            "and add at least one previously uncached service"
        )
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
