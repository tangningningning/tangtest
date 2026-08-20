#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预测场景生成器：车辆位置不确定性 + 服务流行度 + 未来任务。

本模块只生成后续优化器可直接使用的预测输入，不保存未来 ground truth。
车辆 nominal 预测复用 ``vehicle_prediction.py`` 的 SUMO route geometry、
noisy observation replay、Kalman replay 和 ``predict_nominal``；服务流行度
预测复用 ``popularity_prediction.py`` 的 EWMA rho_obs、GRU checkpoint 加载
和推理；任务负载采样沿用 ``task_scenario.py`` 中的 service catalog 与
workload jitter 语义。
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
import torch

import popularity_prediction as pop
import vehicle_prediction as veh
from task_scenario import TaskConfig, catalog_by_id, load_json


# ============================================================================
# 集中配置区：均为 provisional defaults
# ============================================================================


@dataclass(frozen=True)
class ScenarioConfig:
    horizon_steps: int = 5
    num_scenarios: int = 20
    scenario_seed: int = 20260820
    max_resample_attempts: int = 20
    road_distance_tolerance_m: float = 15.0
    v2v_range_m: float = 100.0
    device: str = "cpu"

    def validate(self) -> None:
        if self.horizon_steps <= 0:
            raise ValueError("H_p 必须大于 0。")
        if self.num_scenarios <= 0:
            raise ValueError("S/num_scenarios 必须大于 0。")
        if self.max_resample_attempts <= 0:
            raise ValueError("max_resample_attempts 必须大于 0。")
        if self.road_distance_tolerance_m <= 0.0:
            raise ValueError("road_distance_tolerance_m 必须大于 0。")
        if self.v2v_range_m <= 0.0:
            raise ValueError("v2v_range_m 必须大于 0。")


DEFAULT_CONFIG = ScenarioConfig()


# ============================================================================
# 文件和元数据读取
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


def load_roles(path: Path) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    roles = load_json(path)
    sv_ids = [str(vehicle_id) for vehicle_id in roles.get("service_vehicles", [])]
    tv_ids = [str(vehicle_id) for vehicle_id in roles.get("task_vehicles", [])]
    research_ids = list(dict.fromkeys(sv_ids + tv_ids))
    declared = {str(vehicle_id) for vehicle_id in roles.get("research_vehicles", [])}
    if set(research_ids) != declared:
        raise ValueError("research_vehicles 与 service_vehicles + task_vehicles 不一致。")
    if set(sv_ids) & set(tv_ids):
        raise ValueError("SV 与 TV 集合存在交集。")
    return roles, research_ids, sv_ids, tv_ids


def load_residual_bank(path: Path) -> dict[str, Any]:
    loaded = np.load(path, allow_pickle=False)
    try:
        metadata = json.loads(str(loaded["metadata_json"].item()))
        return {
            "residual_blocks": np.asarray(loaded["residual_blocks"], dtype=np.float64),
            "vehicle_ids": np.asarray(loaded["vehicle_ids"]).astype(str),
            "origin_slots": np.asarray(loaded["origin_slots"], dtype=np.int64),
            "H_p": int(np.asarray(loaded["H_p"]).item()),
            "dt": float(np.asarray(loaded["dt"]).item()),
            "Q": np.asarray(loaded["Q"], dtype=np.float64),
            "R": np.asarray(loaded["R"], dtype=np.float64),
            "noise_sigma_q": float(np.asarray(loaded["noise_sigma_q"]).item()),
            "noise_sigma_u": float(np.asarray(loaded["noise_sigma_u"]).item()),
            "observation_noise_seed": int(
                np.asarray(loaded["observation_noise_seed"]).item()
            ),
            "metadata": metadata,
        }
    finally:
        loaded.close()


def read_request_counts(path: Path) -> tuple[np.ndarray, list[float]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"time_s", "slot", "service_id", "request_count"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("popularity.csv 缺少 request_count 所需字段。")
        rows.extend(reader)
    slots = sorted({int(row["slot"]) for row in rows})
    services = sorted({int(row["service_id"]) for row in rows})
    if slots != list(range(len(slots))) or services != list(range(len(services))):
        raise ValueError("popularity.csv 的 slot/service_id 必须连续。")
    counts = np.zeros((len(slots), len(services)), dtype=np.float64)
    times = [0.0 for _ in slots]
    for row in rows:
        slot = int(row["slot"])
        service_id = int(row["service_id"])
        counts[slot, service_id] = float(row["request_count"])
        times[slot] = float(row["time_s"])
    return counts, times


# ============================================================================
# 车辆 nominal prediction
# ============================================================================


def build_vehicle_nominal_predictions(
    project_root: Path,
    roles: dict[str, Any],
    research_ids: list[str],
    residual_bank: dict[str, Any],
    trajectory_file: Path,
    net_file: str | None,
    sumo_home_arg: str | None,
    origin_slots: np.ndarray,
    horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, veh.RouteGeometry], dict[str, dict[int, veh.TruthSample]]]:
    dt_s = float(roles.get("dt", residual_bank["dt"]))
    warmup_s = float(roles.get("warmup_time", 100.0))
    episode_s = float(roles.get("episode_time", 300.0))
    num_slots = int(round(episode_s / dt_s))

    pred_config = veh.PredictionConfig(
        horizon_steps=horizon_steps,
        sigma_position_m=float(residual_bank["noise_sigma_q"]),
        sigma_speed_mps=float(residual_bank["noise_sigma_u"]),
        observation_noise_seed=int(residual_bank["observation_noise_seed"]),
    )
    planned_routes = veh.load_planned_routes(roles, research_ids)
    truth = veh.load_truth_trajectories(trajectory_file, research_ids, dt_s)
    _, truth_index = veh.formal_truth_index(
        truth,
        research_ids,
        warmup_s,
        episode_s,
        dt_s,
    )

    sumo_home = veh.find_sumo_home(sumo_home_arg)
    sumolib = veh.load_sumolib(sumo_home)
    network_file = veh.find_network(project_root, net_file)
    net = sumolib.net.readNet(str(network_file), withConnections=True, withInternal=True)
    route_geometries = {
        vehicle_id: veh.build_route_geometry(vehicle_id, planned_routes[vehicle_id], net)
        for vehicle_id in research_ids
    }
    veh.map_truth_to_route_s(truth, route_geometries, pred_config)
    veh.validate_route_coordinates(truth, route_geometries)
    observations = veh.build_noisy_observations(truth, route_geometries, pred_config)
    replay = veh.run_kalman_replay(
        observations,
        residual_bank["Q"],
        residual_bank["R"],
        dt_s,
        pred_config,
    )

    nominal_xy = np.zeros(
        (len(origin_slots), horizon_steps, len(research_ids), 2),
        dtype=np.float64,
    )
    tangents = np.zeros_like(nominal_xy)
    warmup_step = int(round(warmup_s / dt_s))
    for origin_index, origin_slot in enumerate(origin_slots.tolist()):
        if origin_slot + horizon_steps > num_slots:
            raise ValueError("origin_slot 越界，无法生成完整 H_p。")
        cutoff_abs_step = warmup_step + origin_slot - 1
        for vehicle_index, vehicle_id in enumerate(research_ids):
            posterior = veh.state_at_cutoff(
                replay[vehicle_id],
                cutoff_abs_step,
                dt_s,
            )
            nominal = veh.predict_nominal(
                posterior,
                route_geometries[vehicle_id],
                horizon_steps,
                dt_s,
            )
            for horizon, item in enumerate(nominal):
                nominal_xy[origin_index, horizon, vehicle_index] = [
                    float(item["x_pred"]),
                    float(item["y_pred"]),
                ]
                tangents[origin_index, horizon, vehicle_index] = np.asarray(
                    item["tangent"],
                    dtype=np.float64,
                )
    return nominal_xy, tangents, route_geometries, truth_index


# ============================================================================
# residual block sampling 和道路合法性
# ============================================================================


def build_residual_indices(
    residual_blocks: np.ndarray,
    vehicle_ids: np.ndarray,
    research_ids: list[str],
) -> tuple[dict[str, np.ndarray], list[str]]:
    pooled = np.arange(len(residual_blocks), dtype=np.int64)
    by_vehicle: dict[str, np.ndarray] = {}
    warnings: list[str] = []
    for vehicle_id in research_ids:
        indices = np.where(vehicle_ids == vehicle_id)[0].astype(np.int64)
        if len(indices) > 0:
            by_vehicle[vehicle_id] = indices
        else:
            by_vehicle[vehicle_id] = pooled
            warnings.append(f"vehicle {vehicle_id} residual bank fallback to pooled")
    return by_vehicle, warnings


def point_route_distance(
    geometry: veh.RouteGeometry,
    point: np.ndarray,
    previous_s: float | None,
) -> tuple[np.ndarray, float, float]:
    s_value, distance = geometry.project_xy_to_s(
        float(point[0]),
        float(point[1]),
        previous_s,
        max(geometry.total_length_m, 1.0),
        max(geometry.total_length_m, 1.0),
    )
    route_point, _ = geometry.point_and_tangent(s_value)
    return route_point, float(distance), float(s_value)


def apply_residual_block(
    nominal_xy: np.ndarray,
    tangents: np.ndarray,
    residual_block: np.ndarray,
) -> np.ndarray:
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=-1)
    return nominal_xy + residual_block[:, 0:1] * tangents + residual_block[:, 1:2] * normals


def block_is_road_valid(
    points: np.ndarray,
    geometry: veh.RouteGeometry,
    tolerance_m: float,
) -> bool:
    previous_s: float | None = None
    for point in points:
        _, distance, previous_s = point_route_distance(geometry, point, previous_s)
        if distance > tolerance_m:
            return False
    return True


def project_block_to_route(
    points: np.ndarray,
    geometry: veh.RouteGeometry,
) -> np.ndarray:
    projected = np.zeros_like(points)
    previous_s: float | None = None
    for index, point in enumerate(points):
        route_point, _, previous_s = point_route_distance(geometry, point, previous_s)
        projected[index] = route_point
    return projected


def sample_positions(
    nominal_xy: np.ndarray,
    tangents: np.ndarray,
    route_geometries: dict[str, veh.RouteGeometry],
    residual_bank: dict[str, Any],
    research_ids: list[str],
    config: ScenarioConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    residual_blocks = residual_bank["residual_blocks"]
    by_vehicle, warnings = build_residual_indices(
        residual_blocks,
        residual_bank["vehicle_ids"],
        research_ids,
    )
    n_origin, horizon_steps, n_vehicle, _ = nominal_xy.shape
    positions = np.zeros(
        (n_origin, config.num_scenarios, horizon_steps, n_vehicle, 2),
        dtype=np.float32,
    )
    raw_attempt_count = 0
    raw_invalid_count = 0
    resampled_count = 0
    fallback_projection_count = 0
    used_own_bank_count = 0
    used_pooled_bank_count = 0

    for origin_index in range(n_origin):
        for scenario_index in range(config.num_scenarios):
            for vehicle_index, vehicle_id in enumerate(research_ids):
                geometry = route_geometries[vehicle_id]
                candidate_points: np.ndarray | None = None
                indices = by_vehicle[vehicle_id]
                uses_own_bank = bool(np.any(residual_bank["vehicle_ids"][indices] == vehicle_id))
                for attempt in range(config.max_resample_attempts):
                    raw_attempt_count += 1
                    block_index = int(indices[int(rng.integers(0, len(indices)))])
                    residual_block = residual_blocks[block_index]
                    points = apply_residual_block(
                        nominal_xy[origin_index, :, vehicle_index, :],
                        tangents[origin_index, :, vehicle_index, :],
                        residual_block,
                    )
                    # e_perp 是相对 planned route 切线法向的横向残差。第一版
                    # 用整段 block 的最大横向偏离作为道路可行性 reject 条件；
                    # 这样保留 block 时序连续性，fallback 再回到 route reference。
                    if (
                        np.all(np.isfinite(points))
                        and float(np.max(np.abs(residual_block[:, 1])))
                        <= config.road_distance_tolerance_m
                    ):
                        candidate_points = points
                        break
                    raw_invalid_count += 1
                    resampled_count += 1
                if candidate_points is None:
                    fallback_projection_count += horizon_steps
                    candidate_points = nominal_xy[origin_index, :, vehicle_index, :]
                if uses_own_bank:
                    used_own_bank_count += 1
                else:
                    used_pooled_bank_count += 1
                positions[origin_index, scenario_index, :, vehicle_index, :] = (
                    candidate_points.astype(np.float32)
                )

    total_vehicle_scenarios = n_origin * config.num_scenarios * n_vehicle
    stats = {
        "raw_attempt_count": raw_attempt_count,
        "raw_invalid_count": raw_invalid_count,
        "resampled_count": resampled_count,
        "fallback_projection_count": fallback_projection_count,
        "raw_invalid_rate": (
            float(raw_invalid_count / raw_attempt_count) if raw_attempt_count else 0.0
        ),
        "fallback_projection_rate": float(
            fallback_projection_count
            / max(n_origin * config.num_scenarios * horizon_steps * n_vehicle, 1)
        ),
        "used_own_vehicle_bank_count": used_own_bank_count,
        "used_pooled_bank_count": used_pooled_bank_count,
        "residual_bank_warnings": warnings,
        "residual_block_sampling": "whole_block_per_vehicle_scenario",
        "cross_vehicle_residual_sampling": "independent",
    }
    if used_own_bank_count + used_pooled_bank_count != total_vehicle_scenarios:
        raise AssertionError("residual bank 使用计数不一致。")
    return positions, stats


# ============================================================================
# GRU 流行度预测和任务场景
# ============================================================================


def predict_rho_hat(
    checkpoint_file: Path,
    request_counts: np.ndarray,
    origin_slots: np.ndarray,
    config: ScenarioConfig,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    model, checkpoint_metadata = pop.load_checkpoint_model(checkpoint_file, device)
    checkpoint_hp = int(checkpoint_metadata["H_p"])
    if checkpoint_hp != config.horizon_steps:
        raise ValueError("GRU checkpoint H_p 与 scenario H_p 不一致。")
    history_length = int(checkpoint_metadata["L_rho"])
    beta_rho = float(checkpoint_metadata["beta_rho"])
    alpha_rho = float(checkpoint_metadata["alpha_rho"])
    rho_obs = pop.build_ewma_rho_obs(request_counts, beta_rho, alpha_rho)
    inputs = np.stack(
        [
            rho_obs[origin - history_length : origin]
            for origin in origin_slots.tolist()
        ],
        axis=0,
    ).astype(np.float32)
    rho_hat = pop.predict_probabilities(
        model,
        inputs,
        batch_size=256,
        device=device,
    )
    if not np.allclose(rho_hat.sum(axis=-1), 1.0, atol=1e-5, rtol=0.0):
        raise AssertionError("rho_hat 每个 horizon 概率和不为 1。")
    return rho_hat.astype(np.float32), checkpoint_metadata


def task_config_from_catalog(catalog_json: dict[str, Any]) -> TaskConfig:
    generation = catalog_json.get("generation_config", {})
    return TaskConfig(
        num_services=int(catalog_json["service_count"]),
        arrival_probability=float(generation.get("arrival_probability", 0.5)),
        zipf_exponent=float(generation.get("zipf_exponent", 0.8)),
        popularity_epoch_length=int(generation.get("popularity_epoch_length", 20)),
        popularity_changes_per_epoch=int(
            generation.get("popularity_changes_per_epoch", 1)
        ),
        input_relative_jitter=float(generation.get("input_relative_jitter", 0.2)),
        load_relative_jitter=float(generation.get("load_relative_jitter", 0.1)),
    )


def make_rng_streams(seed: int, origin_slot: int) -> dict[str, np.random.Generator]:
    root = np.random.SeedSequence([int(seed), int(origin_slot)])
    names = ("residual", "arrival", "service_request", "workload")
    return {
        name: np.random.default_rng(child)
        for name, child in zip(names, root.spawn(len(names)))
    }


def sample_tasks(
    rho_hat: np.ndarray,
    tv_ids: list[str],
    catalog_json: dict[str, Any],
    task_config: TaskConfig,
    config: ScenarioConfig,
    rngs_by_origin: list[dict[str, np.random.Generator]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    services = catalog_by_id(catalog_json["services"])
    n_origin, horizon_steps, num_services = rho_hat.shape
    n_tv = len(tv_ids)
    arrival = np.zeros(
        (n_origin, config.num_scenarios, horizon_steps, n_tv),
        dtype=np.int8,
    )
    service_id = np.full(arrival.shape, -1, dtype=np.int16)
    input_size = np.zeros(arrival.shape, dtype=np.float64)
    cpu_cycles = np.zeros(arrival.shape, dtype=np.float64)

    for origin_index in range(n_origin):
        arrival_rng = rngs_by_origin[origin_index]["arrival"]
        request_rng = rngs_by_origin[origin_index]["service_request"]
        workload_rng = rngs_by_origin[origin_index]["workload"]
        for scenario_index in range(config.num_scenarios):
            for horizon in range(horizon_steps):
                probabilities = rho_hat[origin_index, horizon].astype(np.float64)
                probabilities = probabilities / probabilities.sum()
                arrivals = (
                    arrival_rng.random(n_tv) < task_config.arrival_probability
                )
                arrival[origin_index, scenario_index, horizon, :] = arrivals.astype(
                    np.int8
                )
                for tv_index, has_task in enumerate(arrivals.tolist()):
                    if not has_task:
                        continue
                    chosen = int(request_rng.choice(num_services, p=probabilities))
                    service = services[chosen]
                    input_factor = 1.0 + float(
                        workload_rng.uniform(
                            -task_config.input_relative_jitter,
                            task_config.input_relative_jitter,
                        )
                    )
                    load_factor = 1.0 + float(
                        workload_rng.uniform(
                            -task_config.load_relative_jitter,
                            task_config.load_relative_jitter,
                        )
                    )
                    sampled_input = float(service["mean_input_size"]) * input_factor
                    sampled_cycles = (
                        float(service["cycles_per_bit"]) * sampled_input * load_factor
                    )
                    service_id[origin_index, scenario_index, horizon, tv_index] = chosen
                    input_size[origin_index, scenario_index, horizon, tv_index] = (
                        sampled_input
                    )
                    cpu_cycles[origin_index, scenario_index, horizon, tv_index] = (
                        sampled_cycles
                    )

    stats = {
        "predicted_arrival_empirical_rate": float(np.mean(arrival)),
        "arrival_probability_lambda": float(task_config.arrival_probability),
    }
    return arrival, service_id, input_size, cpu_cycles, stats


# ============================================================================
# 验证、relative distance sanity 和输出
# ============================================================================


def relative_distance_sanity(
    positions_xy: np.ndarray,
    origin_slots: np.ndarray,
    research_ids: list[str],
    sv_ids: list[str],
    tv_ids: list[str],
    truth_index: dict[str, dict[int, veh.TruthSample]],
    v2v_range_m: float,
) -> dict[str, float]:
    vehicle_index = {vehicle_id: index for index, vehicle_id in enumerate(research_ids)}
    pred_distances: list[float] = []
    true_distances: list[float] = []
    for origin_index, origin_slot in enumerate(origin_slots.tolist()):
        for horizon in range(positions_xy.shape[2]):
            target_slot = origin_slot + horizon
            for tv_id in tv_ids:
                tv_pred = positions_xy[
                    origin_index, :, horizon, vehicle_index[tv_id], :
                ].astype(np.float64)
                tv_true = np.asarray(
                    [
                        truth_index[tv_id][target_slot].x_m,
                        truth_index[tv_id][target_slot].y_m,
                    ],
                    dtype=np.float64,
                )
                for sv_id in sv_ids:
                    sv_pred = positions_xy[
                        origin_index, :, horizon, vehicle_index[sv_id], :
                    ].astype(np.float64)
                    sv_true = np.asarray(
                        [
                            truth_index[sv_id][target_slot].x_m,
                            truth_index[sv_id][target_slot].y_m,
                        ],
                        dtype=np.float64,
                    )
                    pred_distances.extend(
                        np.linalg.norm(tv_pred - sv_pred, axis=1).tolist()
                    )
                    true_distances.append(float(np.linalg.norm(tv_true - sv_true)))
    pred_array = np.asarray(pred_distances, dtype=np.float64)
    true_array = np.asarray(true_distances, dtype=np.float64)
    return {
        "pred_mean_relative_distance_m": float(np.mean(pred_array)),
        "pred_std_relative_distance_m": float(np.std(pred_array)),
        "true_mean_relative_distance_m": float(np.mean(true_array)),
        "true_std_relative_distance_m": float(np.std(true_array)),
        "pred_v2v_reachable_rate": float(np.mean(pred_array <= v2v_range_m)),
        "true_v2v_reachable_rate": float(np.mean(true_array <= v2v_range_m)),
        "v2v_range_m": float(v2v_range_m),
    }


def verify_scenarios(
    positions_xy: np.ndarray,
    arrival: np.ndarray,
    service_id: np.ndarray,
    input_size: np.ndarray,
    cpu_cycles: np.ndarray,
    rho_hat: np.ndarray,
    task_config: TaskConfig,
    num_services: int,
) -> None:
    arrays = (positions_xy, arrival, service_id, input_size, cpu_cycles, rho_hat)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise AssertionError("预测场景数组存在 NaN/Inf。")
    if not np.all(np.isin(arrival, [0, 1])):
        raise AssertionError("arrival 不属于 {0,1}。")
    if not np.allclose(rho_hat.sum(axis=-1), 1.0, atol=1e-5, rtol=0.0):
        raise AssertionError("rho_hat 概率和不为 1。")
    no_arrival = arrival == 0
    has_arrival = arrival == 1
    if not np.all(service_id[no_arrival] == -1):
        raise AssertionError("arrival=0 时 service_id 应为 -1。")
    if not np.all(input_size[no_arrival] == 0.0) or not np.all(
        cpu_cycles[no_arrival] == 0.0
    ):
        raise AssertionError("arrival=0 时 D/L 应为 0。")
    if np.any(has_arrival):
        if np.any(service_id[has_arrival] < 0) or np.any(
            service_id[has_arrival] >= num_services
        ):
            raise AssertionError("arrival=1 时 service_id 越界。")
        if np.any(input_size[has_arrival] <= 0.0) or np.any(
            cpu_cycles[has_arrival] <= 0.0
        ):
            raise AssertionError("arrival=1 时 D/L 必须大于 0。")
    empirical = float(np.mean(arrival))
    tolerance = max(
        0.05,
        4.0
        * math.sqrt(
            max(task_config.arrival_probability * (1.0 - task_config.arrival_probability), 1e-12)
            / arrival.size
        ),
    )
    if abs(empirical - task_config.arrival_probability) > tolerance:
        raise AssertionError("预测 arrival rate 与 lambda 偏差过大。")


def save_npz(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    try:
        np.savez_compressed(temporary, **payload)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 S 个未来预测场景 Xi_pred。")
    parser.add_argument("--roles-file", default="data/generated/vehicle_roles.json")
    parser.add_argument("--trajectory-file", default="data/generated/trajectory.csv")
    parser.add_argument("--popularity-file", default="data/generated/popularity.csv")
    parser.add_argument("--service-catalog-file", default="data/generated/service_catalog.json")
    parser.add_argument("--residual-file", default="data/generated/residual_blocks.npz")
    parser.add_argument("--gru-checkpoint", default="data/generated/popularity_gru.pt")
    parser.add_argument("--output-file", default="data/generated/prediction_scenarios.npz")
    parser.add_argument("--net-file", default=None)
    parser.add_argument("--sumo-home", default=None)
    parser.add_argument("--horizon", type=int, default=DEFAULT_CONFIG.horizon_steps)
    parser.add_argument("--num-scenarios", type=int, default=DEFAULT_CONFIG.num_scenarios)
    parser.add_argument("--scenario-seed", type=int, default=DEFAULT_CONFIG.scenario_seed)
    parser.add_argument(
        "--max-resample-attempts",
        type=int,
        default=DEFAULT_CONFIG.max_resample_attempts,
    )
    parser.add_argument(
        "--road-distance-tolerance-m",
        type=float,
        default=DEFAULT_CONFIG.road_distance_tolerance_m,
    )
    parser.add_argument("--v2v-range-m", type=float, default=DEFAULT_CONFIG.v2v_range_m)
    parser.add_argument("--device", default=DEFAULT_CONFIG.device, choices=("cpu", "cuda"))
    return parser


def config_from_args(args: argparse.Namespace) -> ScenarioConfig:
    return ScenarioConfig(
        horizon_steps=args.horizon,
        num_scenarios=args.num_scenarios,
        scenario_seed=args.scenario_seed,
        max_resample_attempts=args.max_resample_attempts,
        road_distance_tolerance_m=args.road_distance_tolerance_m,
        v2v_range_m=args.v2v_range_m,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = config_from_args(args)
        config.validate()
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求 CUDA，但当前 PyTorch CUDA 不可用。")
        device = torch.device(config.device)

        project_root = find_project_root()
        roles_file = resolve_path(project_root, args.roles_file)
        trajectory_file = resolve_path(project_root, args.trajectory_file)
        popularity_file = resolve_path(project_root, args.popularity_file)
        catalog_file = resolve_path(project_root, args.service_catalog_file)
        residual_file = resolve_path(project_root, args.residual_file)
        checkpoint_file = resolve_path(project_root, args.gru_checkpoint)
        output_file = resolve_path(project_root, args.output_file)

        roles, research_ids, sv_ids, tv_ids = load_roles(roles_file)
        residual_bank = load_residual_bank(residual_file)
        if int(residual_bank["H_p"]) != config.horizon_steps:
            raise ValueError("residual_blocks H_p 与 scenario H_p 不一致。")

        service_catalog = load_json(catalog_file)
        task_config = task_config_from_catalog(service_catalog)
        task_config.validate(len(tv_ids))
        request_counts, times = read_request_counts(popularity_file)
        rho_history_length = int(
            torch.load(checkpoint_file, map_location="cpu", weights_only=False)[
                "metadata"
            ]["L_rho"]
        )
        checkpoint_hp = int(
            torch.load(checkpoint_file, map_location="cpu", weights_only=False)[
                "metadata"
            ]["H_p"]
        )
        if checkpoint_hp != config.horizon_steps:
            raise ValueError("GRU checkpoint H_p 与 scenario H_p 不一致。")

        num_slots = request_counts.shape[0]
        max_origin = num_slots - config.horizon_steps
        origin_slots = np.arange(rho_history_length, max_origin + 1, dtype=np.int64)
        if len(origin_slots) <= 0:
            raise ValueError("没有满足 L_rho 和 H_p 的合法 prediction origin。")
        origin_times = np.asarray([times[int(slot)] for slot in origin_slots], dtype=np.float64)
        scenario_ids = np.arange(config.num_scenarios, dtype=np.int64)
        horizons = np.arange(config.horizon_steps, dtype=np.int64)

        nominal_xy, tangents, route_geometries, truth_index = build_vehicle_nominal_predictions(
            project_root,
            roles,
            research_ids,
            residual_bank,
            trajectory_file,
            args.net_file,
            args.sumo_home,
            origin_slots,
            config.horizon_steps,
        )

        rho_hat, checkpoint_metadata = predict_rho_hat(
            checkpoint_file,
            request_counts,
            origin_slots,
            config,
            device,
        )

        rngs_by_origin = [
            make_rng_streams(config.scenario_seed, int(origin_slot))
            for origin_slot in origin_slots.tolist()
        ]
        positions_xy, residual_stats = sample_positions(
            nominal_xy,
            tangents,
            route_geometries,
            residual_bank,
            research_ids,
            config,
            rng=np.random.default_rng(
                np.random.SeedSequence([config.scenario_seed, 777])
            ),
        )
        arrival, service_id, input_size, cpu_cycles, task_stats = sample_tasks(
            rho_hat,
            tv_ids,
            service_catalog,
            task_config,
            config,
            rngs_by_origin,
        )
        distance_stats = relative_distance_sanity(
            positions_xy,
            origin_slots,
            research_ids,
            sv_ids,
            tv_ids,
            truth_index,
            config.v2v_range_m,
        )
        verify_scenarios(
            positions_xy,
            arrival,
            service_id,
            input_size,
            cpu_cycles,
            rho_hat,
            task_config,
            int(service_catalog["service_count"]),
        )

        # 同 seed 完全重放：重新生成 residual 和任务随机过程，检查 CRN 接口稳定。
        repeat_positions, _ = sample_positions(
            nominal_xy,
            tangents,
            route_geometries,
            residual_bank,
            research_ids,
            config,
            rng=np.random.default_rng(
                np.random.SeedSequence([config.scenario_seed, 777])
            ),
        )
        repeat_rngs_by_origin = [
            make_rng_streams(config.scenario_seed, int(origin_slot))
            for origin_slot in origin_slots.tolist()
        ]
        repeat_arrival, repeat_service, repeat_input, repeat_cycles, _ = sample_tasks(
            rho_hat,
            tv_ids,
            service_catalog,
            task_config,
            config,
            repeat_rngs_by_origin,
        )
        if not np.array_equal(positions_xy, repeat_positions):
            raise AssertionError("相同 scenario_seed 未复现 positions_xy。")
        if not (
            np.array_equal(arrival, repeat_arrival)
            and np.array_equal(service_id, repeat_service)
            and np.array_equal(input_size, repeat_input)
            and np.array_equal(cpu_cycles, repeat_cycles)
        ):
            raise AssertionError("相同 scenario_seed 未复现任务场景。")

        vehicle_roles = np.asarray(
            ["SV" if vehicle_id in set(sv_ids) else "TV" for vehicle_id in research_ids],
            dtype="U8",
        )
        metadata = {
            "schema_version": 1,
            "S": config.num_scenarios,
            "H_p": config.horizon_steps,
            "F": int(service_catalog["service_count"]),
            "TV IDs": tv_ids,
            "SV IDs": sv_ids,
            "research_vehicle_ids": research_ids,
            "scenario_seed": config.scenario_seed,
            "lambda": task_config.arrival_probability,
            "residual_bank_source": "provisional_single_episode",
            "cross_vehicle_residual_sampling": "independent",
            "residual_block_sampling": "whole_block_per_vehicle_scenario",
            "vehicle_residual_bank_preference": "same_vehicle_first_else_pooled",
            "max_resample_attempts": config.max_resample_attempts,
            "road_distance_tolerance_m": config.road_distance_tolerance_m,
            "noise_related_metadata_reference": residual_bank["metadata"].get(
                "observation_noise",
                {},
            ),
            "GRU checkpoint metadata reference": {
                "H_p": checkpoint_metadata["H_p"],
                "L_rho": checkpoint_metadata["L_rho"],
                "hidden_size": checkpoint_metadata["hidden_size"],
                "beta_rho": checkpoint_metadata["beta_rho"],
                "alpha_rho": checkpoint_metadata["alpha_rho"],
                "best_validation_loss": checkpoint_metadata["training_summary"][
                    "best_validation_loss"
                ],
            },
            "online_information_boundary": {
                "vehicle_prediction": "observations <= origin_slot - 1",
                "popularity_prediction": "request_count <= origin_slot - 1",
                "tasks": "sampled from rho_hat, not rho_star",
                "future_ground_truth_saved": False,
            },
            "residual_stats": residual_stats,
            "task_stats": task_stats,
            "relative_distance_sanity": distance_stats,
            "same_seed_reproducible": True,
        }

        payload = {
            "origin_slots": origin_slots,
            "origin_times_s": origin_times,
            "scenario_ids": scenario_ids,
            "horizons": horizons,
            "vehicle_ids": np.asarray(research_ids, dtype="U32"),
            "vehicle_roles": vehicle_roles,
            "task_vehicle_ids": np.asarray(tv_ids, dtype="U32"),
            "service_vehicle_ids": np.asarray(sv_ids, dtype="U32"),
            "positions_xy": positions_xy,
            "arrival": arrival,
            "service_id": service_id,
            "input_size": input_size,
            "cpu_cycles": cpu_cycles,
            "rho_hat": rho_hat,
            "metadata_json": np.asarray(
                json.dumps(metadata, ensure_ascii=False, allow_nan=False)
            ),
        }
        save_npz(output_file, payload)

        print("预测场景生成完成：")
        print(f"  origins={len(origin_slots)}, S={config.num_scenarios}, H_p={config.horizon_steps}")
        print(f"  positions_xy shape={positions_xy.shape}")
        print(f"  arrival shape={arrival.shape}, rho_hat shape={rho_hat.shape}")
        print(
            f"  raw_invalid_rate={residual_stats['raw_invalid_rate']:.6f}, "
            f"fallback_projection_rate={residual_stats['fallback_projection_rate']:.6f}"
        )
        print(
            f"  predicted arrival rate="
            f"{task_stats['predicted_arrival_empirical_rate']:.6f}"
        )
        print("  same-seed reproducible=True, future_ground_truth_saved=False")
        print(f"  output: {output_file}")
        return 0
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
