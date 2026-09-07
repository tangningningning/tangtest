#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trigger-conditioned cache configuration optimization.

本阶段只解决：给定上一时隙缓存状态、cache trigger 和预测场景后，选择新的
动态缓存配置。预测候选复用实际执行侧的 Matching/KKT 和固定每任务 Cloud 模型，
但不训练 PPO，也不读取任何未来真实 trajectory/tasks/popularity 标签；正式
episode 初始缓存只使用 service catalog 中的 service size、节点容量和显式 seed。
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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


_CONFIGURATION_PROFILE_SOURCES = ("cache", "uav", "joint")


def _new_configuration_profile_source_stats() -> dict[str, dict[str, int]]:
    return {
        source: {
            "requests": 0,
            "unique": 0,
            "hits": 0,
            "misses": 0,
            "predictive_batches": 0,
            "matching_calls": 0,
            "candidate_evaluations_total": 0,
        }
        for source in _CONFIGURATION_PROFILE_SOURCES
    }


@dataclass
class ConfigurationEvaluationProfile:
    """Optional observation-only counters for one trigger configuration run."""

    utility_requests: int = 0
    unique_configurations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    predictive_batch_evaluations: int = 0
    predictive_matching_calls: int = 0
    matching_candidate_evaluations_total: int = 0
    utility_evaluation_seconds: float = 0.0
    matching_seconds: float = 0.0
    prediction_batch_mode: str = "serial"
    prediction_batch_workers: int = 1
    prediction_batch_requested_workers: int = 1
    batch_results: list[tuple[int, float, int]] = field(default_factory=list)
    source_stats: dict[str, dict[str, int]] = field(
        default_factory=_new_configuration_profile_source_stats
    )

    def source_bucket(self, source: str | None) -> dict[str, int]:
        source_key = str(source or "joint").lower()
        if source_key not in _CONFIGURATION_PROFILE_SOURCES:
            source_key = "joint"
        return self.source_stats[source_key]


@dataclass
class SurrogateEvaluationContext:
    """时隙级预测评价上下文。

    候选缓存或 UAV 位置的预测效用不再使用逐任务独占资源近似，而是按每个
    ``场景 × 预测步`` 组成完整任务批次，并复用实际执行侧的
    Transfer-Exchange Matching 与 KKT 资源分配。缓存仅保存同一时隙内已经
    完整评价过的候选，避免重复计算。
    """

    events: tuple[SurrogateEvent, ...]
    batches: tuple["SurrogateBatch", ...]
    num_scenarios: int
    node_roles: tuple[str, ...]
    node_ids: tuple[str, ...]
    local_cache_state: dict[str, frozenset[int]]
    matching_project_config: dict[str, Any]
    configuration_utility_cache: dict[tuple[bytes, tuple[tuple[str, float, float], ...]], float] = field(
        default_factory=dict
    )
    profile: ConfigurationEvaluationProfile | None = None
    base_cache_nodes: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    prediction_batch_parallel_workers: int = field(
        default=1,
        repr=False,
        compare=False,
    )
    prediction_batch_runtime: "PredictionBatchRuntime | None" = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class SurrogateBatch:
    """一个预测场景步中的并发任务及对应车辆位置。"""

    scenario_index: int
    horizon: int
    discount: float
    events: tuple[SurrogateEvent, ...]
    vehicle_positions: dict[str, np.ndarray]


_PREDICTION_BATCH_WORKER_STATE: dict[str, Any] | None = None


def _normalize_prediction_batch_workers(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("prediction batch worker 数必须是正整数。")
    try:
        workers = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("prediction batch worker 数必须是正整数。") from exc
    if workers != value or workers <= 0:
        raise ValueError("prediction batch worker 数必须是正整数。")
    return workers


def prediction_batch_parallel_workers(project_cfg: dict[str, Any]) -> int:
    """读取 prediction batch 的执行 worker 数。

    当前不新增 formal 配置字段，复用已有的 execution-only
    ``algorithm.ppo.cf_parallel_workers``。外层 CF worker 会由
    ``PredictionBatchRuntime`` 强制退回单进程，避免 nested multiprocessing。
    """

    configured = project_config.get_config_value(
        project_cfg,
        "algorithm.configuration.prediction_batch_parallel_workers",
        None,
    )
    if configured is None:
        configured = project_config.get_config_value(
            project_cfg,
            "algorithm.ppo.cf_parallel_workers",
            1,
        )
    return _normalize_prediction_batch_workers(configured)


@dataclass
class PredictionBatchRuntime:
    """Runtime-only holder for one context's reusable prediction batch pool."""

    requested_workers: int
    effective_workers: int
    mode: str
    executor: ProcessPoolExecutor | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @classmethod
    def for_batch_count(cls, requested_workers: Any, batch_count: int) -> "PredictionBatchRuntime":
        requested = _normalize_prediction_batch_workers(requested_workers)
        # ProcessPoolExecutor cannot be nested safely under the existing CF
        # worker pool.  All non-MainProcess contexts therefore stay serial.
        in_child_process = multiprocessing.current_process().name != "MainProcess"
        if in_child_process or requested <= 1 or int(batch_count) <= 1:
            return cls(requested, 1, "serial")
        return cls(requested, min(requested, int(batch_count)), "parallel")

    def bind_profile(self, profile: ConfigurationEvaluationProfile | None) -> None:
        if profile is None:
            return
        profile.prediction_batch_mode = self.mode
        profile.prediction_batch_workers = self.effective_workers
        profile.prediction_batch_requested_workers = self.requested_workers

    def ensure_executor(
        self,
        context: SurrogateEvaluationContext,
        config: CacheOptimizationConfig,
    ) -> ProcessPoolExecutor | None:
        if self.mode != "parallel":
            return None
        if self.executor is None:
            base_cache_nodes = context.base_cache_nodes
            if not base_cache_nodes:
                raise ValueError("prediction batch pool 缺少 base cache nodes。")
            worker_payload = {
                "batches": context.batches,
                "local_cache_state": context.local_cache_state,
                "matching_project_config": context.matching_project_config,
                "base_cache_nodes": base_cache_nodes,
                "config": config,
            }
            self.executor = ProcessPoolExecutor(
                max_workers=self.effective_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_prediction_batch_worker,
                initargs=(worker_payload,),
            )
        return self.executor

    def close(self) -> None:
        executor = self.executor
        self.executor = None
        if executor is not None:
            executor.shutdown(wait=True)

    def __getstate__(self) -> dict[str, Any]:
        # The executor is execution-only state and must never be serialized into
        # a checkpoint, worker payload, or history object.
        state = dict(self.__dict__)
        state["executor"] = None
        return state


def _initialize_prediction_batch_worker(worker_payload: dict[str, Any]) -> None:
    """Initialize one Windows-spawned prediction batch worker."""

    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass
    global _PREDICTION_BATCH_WORKER_STATE
    _PREDICTION_BATCH_WORKER_STATE = worker_payload


def _evaluate_prediction_batch(
    batch: SurrogateBatch,
    candidate_cache: np.ndarray,
    cache_nodes: dict[str, Any],
    uav_position_map: dict[str, np.ndarray],
    local_cache: dict[str, set[int]],
    matching_project_config: dict[str, Any],
    config: CacheOptimizationConfig,
) -> tuple[float, bool, int]:
    """Evaluate one prediction batch with the unchanged Matching/KKT path."""

    from decision import task_allocation

    tasks = [
        task_allocation.Task(
            task_id=event.tv_id,
            vehicle_id=event.tv_id,
            service_id=event.service_id,
            input_bits=event.input_bits,
            cpu_cycles=event.cpu_cycles,
        )
        for event in batch.events
    ]
    service_positions = task_allocation.build_service_node_positions(
        cache_nodes,
        batch.vehicle_positions,
        uav_position_map,
    )
    result = task_allocation.optimize_task_matching(
        tasks,
        batch.vehicle_positions,
        service_positions,
        local_cache,
        candidate_cache,
        matching_project_config,
        cache_nodes,
        config,
    )
    return (
        float(result.task_utility_sum),
        bool(result.resource_feasible),
        int(result.candidate_evaluations_total),
    )


def _prediction_batch_worker_job(
    job: tuple[int, np.ndarray, tuple[tuple[str, float, float], ...]]
) -> tuple[int, float, bool, int, float]:
    """Evaluate one batch in a persistent worker and return compact results."""

    state = _PREDICTION_BATCH_WORKER_STATE
    if state is None:
        raise RuntimeError("prediction batch worker 未初始化。")
    batch_index, candidate_cache, uav_positions = job
    from decision import uav_configuration

    uav_position_map = {
        node_id: np.asarray([x_m, y_m], dtype=np.float64)
        for node_id, x_m, y_m in uav_positions
    }
    candidate_nodes = uav_configuration.cache_nodes_with_uav_positions(
        state["base_cache_nodes"],
        uav_position_map,
    )
    local_cache = {
        vehicle_id: set(services)
        for vehicle_id, services in state["local_cache_state"].items()
    }
    started_at = time.perf_counter()
    utility, feasible, candidate_evaluations = _evaluate_prediction_batch(
        state["batches"][batch_index],
        candidate_cache,
        candidate_nodes,
        uav_position_map,
        local_cache,
        state["matching_project_config"],
        state["config"],
    )
    return (
        int(batch_index),
        utility,
        feasible,
        candidate_evaluations,
        time.perf_counter() - started_at,
    )


def _matching_project_config(config: CacheOptimizationConfig) -> dict[str, Any]:
    """从统一配置对象提取 Matching/KKT 所需参数，避免读取第二套配置。"""

    return {
        "algorithm": {
            "matching": {
                "max_updates": int(config.matching_max_updates),
                "objective_tolerance": float(config.matching_objective_tolerance),
            },
            "kkt": {
                "min_cpu_fraction": float(config.kkt_min_cpu_fraction),
                "min_bandwidth_fraction": float(config.kkt_min_bandwidth_fraction),
                "bisection_tolerance": float(config.kkt_bisection_tolerance),
                "max_bisection_iterations": int(config.kkt_max_bisection_iterations),
            },
        }
    }


def build_surrogate_evaluation_context(
    origin_scenario: dict[str, Any],
    local_cache_state: dict[str, set[int]],
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
    profile: ConfigurationEvaluationProfile | None = None,
    prediction_batch_parallel_workers: int = 1,
) -> SurrogateEvaluationContext:
    """把预测切片整理成与实际执行模型一致的并发任务批次。"""

    arrival = origin_scenario["arrival"]
    service_id = origin_scenario["service_id"]
    input_size = origin_scenario["input_size"]
    cpu_cycles = origin_scenario["cpu_cycles"]
    tv_ids = [str(value) for value in origin_scenario["task_vehicle_ids"].tolist()]
    num_scenarios, horizon_steps, num_tv = arrival.shape
    vehicle_ids = [str(value) for value in origin_scenario["vehicle_ids"].tolist()]
    events: list[SurrogateEvent] = []
    batches: list[SurrogateBatch] = []
    for scenario_index in range(num_scenarios):
        for horizon in range(horizon_steps):
            discount = config.beta_discount ** horizon
            batch_events: list[SurrogateEvent] = []
            for tv_index in range(num_tv):
                if int(arrival[scenario_index, horizon, tv_index]) == 0:
                    continue
                service = int(service_id[scenario_index, horizon, tv_index])
                if service < 0:
                    raise ValueError("arrival=1 时 service_id 无效。")
                event = SurrogateEvent(
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
                events.append(event)
                batch_events.append(event)
            # 即使当前预测步没有任务，也保留空批次以保持预测时域结构。
            # 实际执行侧需要同一预测步内全部研究车辆的位置，其中 SV 位置
            # 用于构造移动边缘节点，TV 位置用于无线链路计算。
            positions = {
                vehicle_id: np.asarray(
                    origin_scenario["positions_xy"][
                        scenario_index,
                        horizon,
                        vehicle_index,
                    ],
                    dtype=np.float64,
                ).copy()
                for vehicle_index, vehicle_id in enumerate(vehicle_ids)
            }
            batches.append(
                SurrogateBatch(
                    scenario_index=scenario_index,
                    horizon=horizon,
                    discount=float(discount),
                    events=tuple(batch_events),
                    vehicle_positions=positions,
                )
            )

    node_roles = tuple(str(role) for role in cache_nodes["node_roles"])
    node_ids = tuple(str(node_id) for node_id in cache_nodes["node_ids"])
    context = SurrogateEvaluationContext(
        events=tuple(events),
        batches=tuple(batches),
        num_scenarios=int(num_scenarios),
        node_roles=node_roles,
        node_ids=node_ids,
        local_cache_state={
            str(vehicle_id): frozenset(int(service_id) for service_id in services)
            for vehicle_id, services in local_cache_state.items()
        },
        matching_project_config=_matching_project_config(config),
        profile=profile,
        base_cache_nodes=cache_nodes,
        prediction_batch_parallel_workers=_normalize_prediction_batch_workers(
            prediction_batch_parallel_workers
        ),
    )
    context.prediction_batch_runtime = PredictionBatchRuntime.for_batch_count(
        context.prediction_batch_parallel_workers,
        len(context.batches),
    )
    context.prediction_batch_runtime.bind_profile(profile)
    return context


def close_prediction_batch_pool(context: SurrogateEvaluationContext | None) -> None:
    """Close one context's execution-only prediction batch pool."""

    if context is not None and context.prediction_batch_runtime is not None:
        context.prediction_batch_runtime.close()


def evaluate_surrogate_context(
    cache: np.ndarray,
    context: SurrogateEvaluationContext,
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
    profile_source: str = "joint",
) -> float:
    """用实际 Matching/KKT 链评价候选配置的预测效用。"""

    profile = context.profile
    source_bucket = profile.source_bucket(profile_source) if profile is not None else None
    utility_started_at = time.perf_counter() if profile is not None else 0.0
    if profile is not None and source_bucket is not None:
        profile.utility_requests += 1
        source_bucket["requests"] += 1

    candidate_cache = np.ascontiguousarray(cache, dtype=np.int8)
    uav_positions = tuple(
        (
            node_id,
            float(np.asarray(cache_nodes["fixed_positions"][node_id])[0]),
            float(np.asarray(cache_nodes["fixed_positions"][node_id])[1]),
        )
        for node_id, role in zip(context.node_ids, context.node_roles)
        if role == "UAV"
    )
    cache_key = (candidate_cache.tobytes(), uav_positions)
    cached = context.configuration_utility_cache.get(cache_key)
    if cached is not None:
        if profile is not None and source_bucket is not None:
            profile.cache_hits += 1
            source_bucket["hits"] += 1
            profile.utility_evaluation_seconds += time.perf_counter() - utility_started_at

        return float(cached)

    if profile is not None and source_bucket is not None:
        profile.cache_misses += 1
        profile.unique_configurations += 1
        source_bucket["misses"] += 1
        source_bucket["unique"] += 1

    local_cache = {
        vehicle_id: set(services)
        for vehicle_id, services in context.local_cache_state.items()
    }
    uav_position_map = {
        node_id: np.asarray([x_m, y_m], dtype=np.float64)
        for node_id, x_m, y_m in uav_positions
    }
    runtime = context.prediction_batch_runtime
    if runtime is None:
        runtime = PredictionBatchRuntime.for_batch_count(
            context.prediction_batch_parallel_workers,
            len(context.batches),
        )
        context.prediction_batch_runtime = runtime
        runtime.bind_profile(profile)

    executor = runtime.ensure_executor(context, config)
    total = 0.0
    if executor is None:
        batch_results = []
        for batch_index, batch in enumerate(context.batches):
            matching_started_at = time.perf_counter()
            result = _evaluate_prediction_batch(
                batch,
                candidate_cache,
                cache_nodes,
                uav_position_map,
                local_cache,
                context.matching_project_config,
                config,
            )
            batch_results.append(
                (
                    batch_index,
                    result[0],
                    result[1],
                    result[2],
                    time.perf_counter() - matching_started_at,
                )
            )
    else:
        jobs = [
            (batch_index, candidate_cache, uav_positions)
            for batch_index in range(len(context.batches))
        ]
        # executor.map preserves the original batch order; the parent performs
        # the floating-point reduction below in that same order.
        batch_results = list(executor.map(_prediction_batch_worker_job, jobs, chunksize=1))

    for batch_index, raw_utility, resource_feasible, candidate_evaluations, matching_seconds in batch_results:
        batch = context.batches[batch_index]
        if profile is not None and source_bucket is not None:
            profile.predictive_batch_evaluations += 1
            profile.predictive_matching_calls += 1
            profile.matching_seconds += float(matching_seconds)
            source_bucket["predictive_batches"] += 1
            source_bucket["matching_calls"] += 1
            profile.matching_candidate_evaluations_total += int(candidate_evaluations)
            source_bucket["candidate_evaluations_total"] += int(candidate_evaluations)
            profile.batch_results.append(
                (int(batch_index), float(raw_utility), int(candidate_evaluations))
            )
        if not resource_feasible:
            raise AssertionError(
                "预测配置的实际 Matching/KKT 评价不可行："
                f"scenario={batch.scenario_index}, horizon={batch.horizon}"
            )
        total += batch.discount * float(raw_utility)
    utility = float(total / max(context.num_scenarios, 1))
    context.configuration_utility_cache[cache_key] = utility
    if profile is not None:
        profile.utility_evaluation_seconds += time.perf_counter() - utility_started_at
    return utility


def print_configuration_profile(profile: ConfigurationEvaluationProfile) -> None:
    """Print compact terminal-only counters for one bounded configuration run."""

    hit_rate = (
        100.0 * profile.cache_hits / profile.utility_requests
        if profile.utility_requests
        else 0.0
    )
    print("[CONFIG PROFILE]", flush=True)
    print(
        f"utility_requests={profile.utility_requests} | "
        f"unique_configs={profile.unique_configurations} | "
        f"cache_hits={profile.cache_hits} | "
        f"cache_misses={profile.cache_misses} | "
        f"cache_hit_rate={hit_rate:.1f}% | "
        f"predictive_batches={profile.predictive_batch_evaluations} | "
        f"predictive_matching_calls={profile.predictive_matching_calls} | "
        f"matching_candidate_evals={profile.matching_candidate_evaluations_total} | "
        f"prediction_batch_mode={profile.prediction_batch_mode} | "
        f"prediction_batch_workers={profile.prediction_batch_workers} | "
        f"utility_time={profile.utility_evaluation_seconds:.3f}s | "
        f"matching_time={profile.matching_seconds:.3f}s",
        flush=True,
    )
    for label, source in (("CACHE", "cache"), ("UAV", "uav"), ("JOINT", "joint")):
        stats = profile.source_stats[source]
        print(
            f"[{label} PROFILE] requests={stats['requests']} | "
            f"unique={stats['unique']} | "
            f"matching_calls={stats['matching_calls']} | "
            f"candidate_evals={stats['candidate_evaluations_total']}",
            flush=True,
        )


def predicted_service_utility(
    cache: np.ndarray,
    origin_scenario: dict[str, Any],
    local_cache_state: dict[str, set[int]],
    cache_nodes: dict[str, Any],
    config: CacheOptimizationConfig,
    evaluation_context: SurrogateEvaluationContext | None = None,
    profile_source: str = "joint",
) -> float:
    context = evaluation_context
    if context is None:
        context = build_surrogate_evaluation_context(
            origin_scenario,
            local_cache_state,
            cache_nodes,
            config,
        )
    return evaluate_surrogate_context(
        cache,
        context,
        cache_nodes,
        config,
        profile_source=profile_source,
    )


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
    profile_source: str = "cache",
) -> float:
    utility = predicted_service_utility(
        cache,
        origin_scenario,
        local_cache_state,
        cache_nodes,
        config,
        evaluation_context=evaluation_context,
        profile_source=profile_source,
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
    profile_source: str = "joint",
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
        profile_source=profile_source,
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
    """按当前 service 数量执行 0-1 knapsack：只用于初始化，不作为最终 solver。"""
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
            profile_source="cache",
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
        profile_source="cache",
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
