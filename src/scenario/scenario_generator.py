#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预测场景生成器：车辆位置不确定性 + 服务流行度 + 未来任务。

本模块只生成后续优化器可直接使用的预测输入，不保存未来 ground truth。
车辆 nominal 预测复用 ``vehicle_prediction.py`` 的 SUMO route geometry、
noisy observation replay、Kalman replay 和 ``predict_nominal``；服务流行度
预测复用 ``popularity_prediction.py`` 的 observed rho_obs、GRU checkpoint 加载
和推理；任务负载采样沿用 ``task_scenario.py`` 中的 service catalog 与
workload jitter 语义。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import project_config as project_config_module
from scenario import popularity_prediction as pop
from scenario import vehicle_prediction as veh
from scenario.vehicle_order import (
    RESIDUAL_SAMPLING_CONTRACT,
)
from scenario.task_scenario import (
    TaskConfig,
    catalog_by_id,
    load_json,
    task_config_from_project_config,
)
try:
    from scenario.realization import derive_episode_seeds
except ModuleNotFoundError:  # direct ``python src/scenario/scenario_generator.py``
    from realization import derive_episode_seeds


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
    vehicle_observation_seed: int | None = None

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


def load_roles(path: Path) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    roles = load_json(path)

    def get_role_ids(field: str) -> tuple[str, ...]:
        value = roles.get(field)
        if not isinstance(value, list):
            raise ValueError(f"vehicle_roles.json {field} must be a list of IDs")
        ids = tuple(str(item) for item in value)
        if any(not vehicle_id.strip() for vehicle_id in ids):
            raise ValueError(f"vehicle_roles.json {field} contains empty IDs")
        if len(set(ids)) != len(ids):
            raise ValueError(f"vehicle_roles.json {field} contains duplicate IDs")
        return ids

    sv_ids = get_role_ids("service_vehicles")
    tv_ids = get_role_ids("task_vehicles")
    research_ids = get_role_ids("research_vehicles")
    if (
        set(sv_ids) & set(tv_ids)
        or set(research_ids) != set(sv_ids) | set(tv_ids)
    ):
        raise ValueError("vehicle_roles.json has conflicting research/SV/TV ID sets")
    # SV and TV blocks preserve their serialized role-specific sequence. The
    # matching IDs travel with each NPZ axis and are joined explicitly by ID.
    return roles, list(sv_ids + tv_ids), list(sv_ids), list(tv_ids)


def load_residual_bank(path: Path) -> dict[str, Any]:
    loaded = np.load(path, allow_pickle=False)
    try:
        metadata = json.loads(str(loaded["metadata_json"].item()))
        schema_version = int(metadata.get("schema_version", 0))
        calibration_scope = str(metadata.get("calibration_scope", ""))
        if (
            schema_version < 2
            or metadata.get("artifact_type") != "vehicle_calibration_residual_bank"
            or calibration_scope != "independent_calibration_trajectory"
        ):
            raise ValueError(
                "residual bank 不是独立 calibration artifact；"
                "禁止把 formal episode 单集 residual bank 用于正式场景。"
            )
        source_metadata = metadata.get("source")
        if not isinstance(source_metadata, dict) or not source_metadata.get(
            "roles_path"
        ) or not source_metadata.get("trajectory_path"):
            raise ValueError("独立 residual bank metadata 缺少 calibration source 路径。")
        residual_blocks = np.asarray(loaded["residual_blocks"], dtype=np.float64)
        horizon_steps = int(np.asarray(loaded["H_p"]).item())
        vehicle_ids = np.asarray(loaded["vehicle_ids"]).astype(str)
        origin_slots = np.asarray(loaded["origin_slots"], dtype=np.int64)
        q = np.asarray(loaded["Q"], dtype=np.float64)
        r = np.asarray(loaded["R"], dtype=np.float64)
        if residual_blocks.ndim != 3 or residual_blocks.shape[2] != 2:
            raise ValueError("独立 residual bank 的 residual_blocks shape 无效。")
        if horizon_steps <= 0 or residual_blocks.shape[1] != horizon_steps:
            raise ValueError("独立 residual bank 的 H_p 与 residual_blocks shape 不一致。")
        if len(vehicle_ids) != len(residual_blocks) or len(origin_slots) != len(residual_blocks):
            raise ValueError("独立 residual bank 的 block 元数据长度不一致。")
        if q.shape != (3, 3) or r.shape != (2, 2):
            raise ValueError("独立 residual bank 的 Q/R shape 无效。")
        if not np.all(np.isfinite(residual_blocks)) or not np.all(
            np.isfinite(q)
        ) or not np.all(np.isfinite(r)):
            raise ValueError("独立 residual bank 的 Q/R/residual_blocks 含 NaN/Inf。")
        return {
            "residual_blocks": residual_blocks,
            "vehicle_ids": vehicle_ids,
            "origin_slots": origin_slots,
            "H_p": horizon_steps,
            "dt": float(np.asarray(loaded["dt"]).item()),
            "Q": q,
            "R": r,
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


def read_history_request_counts(path: Path) -> tuple[np.ndarray, list[int]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"history_index", "formal_slot", "service_id", "request_count"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("popularity_history.csv 缺少 request_count 所需字段。")
        rows.extend(reader)
    history_indices = sorted({int(row["history_index"]) for row in rows})
    formal_slots = sorted({int(row["formal_slot"]) for row in rows})
    services = sorted({int(row["service_id"]) for row in rows})
    if history_indices != list(range(len(history_indices))):
        raise ValueError("popularity_history.csv 的 history_index 必须从 0 连续编号。")
    if formal_slots != [index - len(history_indices) for index in history_indices]:
        raise ValueError("popularity_history.csv 的 formal_slot 与 history_index 不一致。")
    if services != list(range(len(services))):
        raise ValueError("popularity_history.csv 的 service_id 必须从 0 连续编号。")
    counts = np.zeros((len(history_indices), len(services)), dtype=np.float64)
    seen: set[tuple[int, int]] = set()
    for row in rows:
        history_index = int(row["history_index"])
        service_id = int(row["service_id"])
        key = (history_index, service_id)
        if key in seen:
            raise ValueError("popularity_history.csv 存在重复 history_index/service_id。")
        seen.add(key)
        counts[history_index, service_id] = float(row["request_count"])
    if len(seen) != len(history_indices) * len(services):
        raise ValueError("popularity_history.csv 没有覆盖完整 history x service 网格。")
    return counts, formal_slots


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
    prediction_rows: list[dict[str, Any]] | None = None,
    observation_noise_seed: int | None = None,
    execution_horizon_slots: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, veh.RouteGeometry], dict[str, dict[int, veh.TruthSample]]]:
    dt_s = float(roles.get("dt", residual_bank["dt"]))
    warmup_s = float(roles.get("warmup_time", 100.0))
    episode_s = float(roles.get("episode_time", 300.0))
    num_slots = (
        int(round(episode_s / dt_s))
        if execution_horizon_slots is None
        else int(execution_horizon_slots)
    )

    pred_config = veh.PredictionConfig(
        horizon_steps=horizon_steps,
        sigma_position_m=float(residual_bank["noise_sigma_q"]),
        sigma_speed_mps=float(residual_bank["noise_sigma_u"]),
        observation_noise_seed=int(
            residual_bank["observation_noise_seed"]
            if observation_noise_seed is None
            else observation_noise_seed
        ),
    )
    planned_routes = veh.load_planned_routes(roles, research_ids)
    truth = veh.load_truth_trajectories(trajectory_file, research_ids, dt_s)
    _, truth_index = veh.formal_truth_index(
        truth,
        research_ids,
        warmup_s,
        num_slots * dt_s,
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
        if origin_slot < 0 or origin_slot >= num_slots:
            raise ValueError("origin_slot 不在 formal episode 范围内。")
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
                if prediction_rows is not None:
                    target_slot = int(origin_slot + horizon)
                    prediction_rows.append(
                        {
                            "origin_slot": int(origin_slot),
                            "origin_time_s": round(
                                warmup_s + origin_slot * dt_s,
                                9,
                            ),
                            "observation_cutoff_slot": int(origin_slot - 1),
                            "observation_cutoff_time_s": round(
                                warmup_s + (origin_slot - 1) * dt_s,
                                9,
                            ),
                            "vehicle_id": vehicle_id,
                            "horizon": int(horizon),
                            "target_slot": target_slot,
                            "target_time_s": round(
                                warmup_s + target_slot * dt_s,
                                9,
                            ),
                            "s_pred": float(item["s_pred"]),
                            "x_pred": float(item["x_pred"]),
                            "y_pred": float(item["y_pred"]),
                            # 这是供正式执行器读取的 causal prediction
                            # artifact；不把 formal future truth 写入其中。
                            "s_true": "",
                            "x_true": "",
                            "y_true": "",
                            "longitudinal_error": "",
                            "lateral_error": "",
                            "e_parallel": "",
                            "e_perp": "",
                        }
                    )
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
    blocks = np.asarray(residual_blocks)
    # Calibration IDs are provenance labels, not persistent Training vehicle identities.
    ids = np.asarray(vehicle_ids).astype(str)
    if (
        blocks.ndim != 3
        or blocks.shape[0] <= 0
        or blocks.shape[1] <= 0
        or blocks.shape[2] != 2
        or ids.shape != (blocks.shape[0],)
        or not np.all(np.isfinite(blocks))
    ):
        raise ValueError("residual bank blocks/vehicle IDs have invalid shape or values")
    if len(set(str(value) for value in research_ids)) != len(research_ids):
        raise ValueError("research_ids must not contain duplicates")
    by_vehicle = {str(vehicle_id): pooled for vehicle_id in research_ids}
    warnings: list[str] = []
    return by_vehicle, warnings


def point_route_distance(
    geometry: veh.RouteGeometry,
    point: np.ndarray,
    previous_s: float | None,
) -> tuple[np.ndarray, float, float]:
    if previous_s is None:
        projection_window_m = max(geometry.total_length_m, 1.0)
        projection_backtrack_m = max(geometry.total_length_m, 1.0)
    else:
        projection_window_m = veh.DEFAULT_CONFIG.projection_window_m
        projection_backtrack_m = veh.DEFAULT_CONFIG.projection_backtrack_m
    s_value, distance = geometry.project_xy_to_s(
        float(point[0]),
        float(point[1]),
        previous_s,
        projection_window_m,
        projection_backtrack_m,
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




def sample_positions(
    nominal_xy: np.ndarray,
    tangents: np.ndarray,
    route_geometries: dict[str, veh.RouteGeometry],
    residual_bank: dict[str, Any],
    research_ids: list[str],
    config: ScenarioConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    residual_blocks = np.asarray(residual_bank["residual_blocks"], dtype=np.float64)
    nominal = np.asarray(nominal_xy, dtype=np.float64)
    tangent_values = np.asarray(tangents, dtype=np.float64)
    if (
        nominal.ndim != 4
        or nominal.shape[-1] != 2
        or tangent_values.shape != nominal.shape
        or not np.all(np.isfinite(nominal))
        or not np.all(np.isfinite(tangent_values))
    ):
        raise ValueError("nominal positions/tangents have invalid shape or values")
    by_vehicle, warnings = build_residual_indices(
        residual_blocks,
        residual_bank["vehicle_ids"],
        research_ids,
    )
    n_origin, horizon_steps, n_vehicle, _ = nominal.shape
    if len(research_ids) != n_vehicle:
        raise ValueError("research_ids length does not match nominal vehicle axis")
    if residual_blocks.shape[1] != horizon_steps:
        raise ValueError("residual block horizon does not match nominal positions")
    positions = np.zeros(
        (n_origin, config.num_scenarios, horizon_steps, n_vehicle, 2),
        dtype=np.float32,
    )
    raw_attempt_count = 0
    raw_invalid_count = 0
    resampled_count = 0
    fallback_projection_count = 0
    used_global_pooled_bank_count = 0

    for origin_index in range(n_origin):
        for scenario_index in range(config.num_scenarios):
            for vehicle_index, vehicle_id in enumerate(research_ids):
                geometry = route_geometries[vehicle_id]
                candidate_points: np.ndarray | None = None
                indices = by_vehicle[vehicle_id]
                for attempt in range(config.max_resample_attempts):
                    raw_attempt_count += 1
                    block_index = int(indices[int(rng.integers(0, len(indices)))])
                    residual_block = residual_blocks[block_index]
                    points = apply_residual_block(
                        nominal[origin_index, :, vehicle_index, :],
                        tangent_values[origin_index, :, vehicle_index, :],
                        residual_block,
                    )
                    # residual 加到 nominal trajectory 后，使用 block_is_road_valid()
                    # 检查整个 H_p block 与 planned route 的距离是否满足容差。
                    # nominal 点本身由 route geometry 生成，因此当每个残差点
                    # 到对应 nominal route 点的距离都不超过容差时，候选必然
                    # 满足同一合法性条件；只有可能超出容差的 block 才需要
                    # 进入逐段几何投影。该短路不改变判定结果，只避免扩容后
                    # 对相同的已知合法 nominal route 重复扫描几何线段。
                    residual_distance = np.linalg.norm(
                        points - nominal[origin_index, :, vehicle_index, :],
                        axis=1,
                    )
                    if (
                        np.all(np.isfinite(points))
                        and (
                            float(np.max(residual_distance))
                            <= config.road_distance_tolerance_m
                            or block_is_road_valid(
                                points,
                                geometry,
                                config.road_distance_tolerance_m,
                            )
                        )
                    ):
                        candidate_points = points
                        break
                    raw_invalid_count += 1
                    resampled_count += 1
                if candidate_points is None:
                    fallback_projection_count += horizon_steps
                    candidate_points = nominal[origin_index, :, vehicle_index, :]
                used_global_pooled_bank_count += 1
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
        # Keep the historical count key at zero for readers of old diagnostics.
        "used_own_vehicle_bank_count": 0,
        "used_pooled_bank_count": used_global_pooled_bank_count,
        "used_global_pooled_bank_count": used_global_pooled_bank_count,
        "residual_bank_warnings": warnings,
        "residual_block_sampling": "whole_block_per_vehicle_scenario",
        "cross_vehicle_residual_sampling": "independent",
        "residual_sampling_contract": RESIDUAL_SAMPLING_CONTRACT,
    }
    if used_global_pooled_bank_count != total_vehicle_scenarios:
        raise AssertionError("residual bank 使用计数不一致。")
    return positions, stats


# ============================================================================
# GRU 流行度预测和任务场景
# ============================================================================


def predict_rho_hat(
    checkpoint_file: Path,
    history_request_counts: np.ndarray,
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
    inputs, _ = pop.build_causal_formal_inputs(
        history_request_counts,
        request_counts,
        history_length,
        origin_slots,
    )
    rho_hat = pop.predict_probabilities(
        model,
        inputs,
        batch_size=256,
        device=device,
    )
    if not np.allclose(rho_hat.sum(axis=-1), 1.0, atol=1e-5, rtol=0.0):
        raise AssertionError("rho_hat 每个 horizon 概率和不为 1。")
    return rho_hat.astype(np.float32), checkpoint_metadata


def task_config_from_catalog(
    catalog_json: dict[str, Any],
    project_config: dict[str, Any] | None = None,
) -> TaskConfig:
    if project_config is not None:
        config = task_config_from_project_config(project_config)
        if int(catalog_json["service_count"]) != config.num_services:
            raise ValueError("service_catalog.json 的 service_count 与 configs/dqn/online_joint_dqn.toml 不一致。")
        return config
    generation = catalog_json.get("generation_config", {})
    return TaskConfig(
        num_services=int(catalog_json["service_count"]),
        arrival_probability=float(generation.get("arrival_probability", 0.5)),
        zipf_exponent=float(generation.get("zipf_exponent", 0.8)),
        popularity_epoch_length=int(generation.get("popularity_epoch_length", 20)),
        popularity_changes_per_epoch=int(
            generation.get("popularity_changes_per_epoch", 1)
        ),
        popularity_history_length=int(
            generation.get("popularity_history_length", 20)
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
    *,
    origin_slots: np.ndarray | None = None,
    arrival_horizon_slots: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    services = catalog_by_id(catalog_json["services"])
    n_origin, horizon_steps, num_services = rho_hat.shape
    origins = (
        np.arange(n_origin, dtype=np.int64)
        if origin_slots is None
        else np.asarray(origin_slots, dtype=np.int64)
    )
    if origins.shape != (n_origin,):
        raise ValueError("origin_slots shape does not match rho_hat.")
    arrival_limit = n_origin if arrival_horizon_slots is None else int(arrival_horizon_slots)
    if arrival_limit < 0:
        raise ValueError("arrival_horizon_slots cannot be negative.")
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
                target_slot = int(origins[origin_index]) + horizon
                if target_slot >= arrival_limit:
                    continue
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

    valid_targets = (
        origins[:, None] + np.arange(horizon_steps, dtype=np.int64)[None, :]
    ) < arrival_limit
    valid_arrivals = arrival[
        np.broadcast_to(
            valid_targets[:, None, :, None],
            (n_origin, config.num_scenarios, horizon_steps, n_tv),
        )
    ]
    stats = {
        "predicted_arrival_empirical_rate": float(np.mean(valid_arrivals))
        if valid_arrivals.size
        else 0.0,
        "arrival_probability_lambda": float(task_config.arrival_probability),
        "arrival_samples_after_arrival_horizon": int(
            np.count_nonzero(arrival[~np.broadcast_to(
                valid_targets[:, None, :, None],
                (n_origin, config.num_scenarios, horizon_steps, n_tv),
            )])
        ),
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
                for sv_id in sv_ids:
                    sv_pred = positions_xy[
                        origin_index, :, horizon, vehicle_index[sv_id], :
                    ].astype(np.float64)
                    pred_distances.extend(
                        np.linalg.norm(tv_pred - sv_pred, axis=1).tolist()
                    )
                    tv_truth = truth_index[tv_id].get(target_slot)
                    sv_truth = truth_index[sv_id].get(target_slot)
                    if tv_truth is not None and sv_truth is not None:
                        tv_true = np.asarray(
                            [tv_truth.x_m, tv_truth.y_m],
                            dtype=np.float64,
                        )
                        sv_true = np.asarray(
                            [sv_truth.x_m, sv_truth.y_m],
                            dtype=np.float64,
                        )
                        true_distances.append(
                            float(np.linalg.norm(tv_true - sv_true))
                        )
    pred_array = np.asarray(pred_distances, dtype=np.float64)
    true_array = np.asarray(true_distances, dtype=np.float64)
    if pred_array.size == 0:
        raise AssertionError("relative distance sanity 没有预测距离样本。")
    return {
        "pred_mean_relative_distance_m": float(np.mean(pred_array)),
        "pred_std_relative_distance_m": float(np.std(pred_array)),
        "true_mean_relative_distance_m": float(np.mean(true_array))
        if true_array.size
        else 0.0,
        "true_std_relative_distance_m": float(np.std(true_array))
        if true_array.size
        else 0.0,
        "pred_v2v_reachable_rate": float(np.mean(pred_array <= v2v_range_m)),
        "true_v2v_reachable_rate": float(np.mean(true_array <= v2v_range_m))
        if true_array.size
        else 0.0,
        "true_distance_sample_count": float(true_array.size),
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
    arrival_horizon_slots: int | None = None,
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
    arrival_limit = arrival.shape[0] if arrival_horizon_slots is None else int(arrival_horizon_slots)
    origin_slots = np.arange(arrival.shape[0], dtype=np.int64)
    valid_targets = (
        origin_slots[:, None] + np.arange(arrival.shape[2], dtype=np.int64)[None, :]
    ) < arrival_limit
    valid_mask = np.broadcast_to(
        valid_targets[:, None, :, None], arrival.shape
    )
    if np.any(arrival[~valid_mask]):
        raise AssertionError("execution-tail scenario arrivals must be zero.")
    empirical = float(np.mean(arrival[valid_mask])) if np.any(valid_mask) else 0.0
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
    parser = argparse.ArgumentParser(description="为一个正式 episode 生成预测场景。")
    parser.add_argument("--config", default="configs/dqn/online_joint_dqn.toml")
    parser.add_argument("--episode-seed", type=int, required=True)
    parser.add_argument("--roles-file", default=None)
    parser.add_argument("--trajectory-file", default=None)
    parser.add_argument("--popularity-file", default=None)
    parser.add_argument("--popularity-history-file", default=None)
    parser.add_argument("--service-catalog-file", default=None)
    parser.add_argument("--residual-file", default=None)
    parser.add_argument("--gru-checkpoint", default=None)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--vehicle-prediction-file", default=None)
    parser.add_argument("--net-file", default=None)
    parser.add_argument("--sumo-home", default=None)
    parser.add_argument(
        "--origin-count",
        type=int,
        default=None,
        help="formal decision origins; can include the no-arrival execution tail",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    return parser


def config_from_project_config(
    project_config: dict[str, Any],
    device: str,
) -> ScenarioConfig:
    get_value = project_config_module.get_config_value
    return ScenarioConfig(
        horizon_steps=int(
            get_value(
                project_config,
                "prediction.vehicle.horizon",
                DEFAULT_CONFIG.horizon_steps,
            )
        ),
        num_scenarios=int(
            get_value(
                project_config,
                "scenario.num_scenarios",
                DEFAULT_CONFIG.num_scenarios,
            )
        ),
        scenario_seed=int(
            get_value(
                project_config,
                "scenario.master_seed",
                DEFAULT_CONFIG.scenario_seed,
            )
        ),
        max_resample_attempts=int(
            get_value(
                project_config,
                "scenario.max_resample_attempts",
                DEFAULT_CONFIG.max_resample_attempts,
            )
        ),
        road_distance_tolerance_m=DEFAULT_CONFIG.road_distance_tolerance_m,
        v2v_range_m=DEFAULT_CONFIG.v2v_range_m,
        device=str(device),
        vehicle_observation_seed=None,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        project_root = find_project_root()
        formal_config = project_config_module.load_config(
            args.config,
            require_artifacts=False,
        )
        config = config_from_project_config(formal_config, args.device)
        episode_seeds = derive_episode_seeds(args.episode_seed)
        config = replace(
            config,
            scenario_seed=episode_seeds["scenario_seed"],
            vehicle_observation_seed=episode_seeds["vehicle_observation_seed"],
        )
        config.validate()
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求 CUDA，但当前 PyTorch CUDA 不可用。")
        device = torch.device(config.device)

        get_value = project_config_module.get_config_value
        roles_file = resolve_path(
            project_root,
            args.roles_file
            or get_value(formal_config, "paths.vehicle_roles", "data/generated/vehicle_roles.json"),
        )
        trajectory_file = resolve_path(
            project_root,
            args.trajectory_file
            or get_value(
                formal_config,
                "paths.trajectory",
                "data/generated/trajectory.csv",
            ),
        )
        popularity_file = resolve_path(
            project_root,
            args.popularity_file
            or get_value(
                formal_config,
                "paths.popularity",
                "data/generated/popularity.csv",
            ),
        )
        popularity_history_file = resolve_path(
            project_root,
            args.popularity_history_file
            or get_value(
                formal_config,
                "paths.popularity_history",
                "data/generated/popularity_history.csv",
            ),
        )
        catalog_file = resolve_path(
            project_root,
            args.service_catalog_file
            or get_value(formal_config, "paths.service_catalog", "data/generated/service_catalog.json"),
        )
        residual_file = resolve_path(
            project_root,
            args.residual_file
            or get_value(
                formal_config,
                "paths.vehicle_residual_bank",
                "data/calibration/residual_blocks.npz",
            ),
        )
        checkpoint_file = resolve_path(
            project_root,
            args.gru_checkpoint
            or get_value(
                formal_config,
                "paths.popularity_gru",
                "data/generated/popularity_gru.pt",
            ),
        )
        output_file = resolve_path(
            project_root,
            args.output_file
            or get_value(
                formal_config,
                "paths.prediction_scenarios",
                "data/generated/prediction_scenarios.npz",
            ),
        )

        roles, research_ids, sv_ids, tv_ids = load_roles(roles_file)
        expected_research = int(get_value(formal_config, "simulation.num_research_vehicles"))
        expected_sv = int(get_value(formal_config, "simulation.num_service_vehicles"))
        expected_tv = int(get_value(formal_config, "simulation.num_task_vehicles"))
        if (len(research_ids), len(sv_ids), len(tv_ids)) != (
            expected_research,
            expected_sv,
            expected_tv,
        ):
            raise ValueError(
                "vehicle_roles.json 角色数量与 configs/dqn/online_joint_dqn.toml 的正式数量不一致。"
            )
        protocol = project_config_module.get_episode_data_protocol(formal_config)
        execution_slots = int(protocol["execution_horizon_slots"])
        arrival_slots = int(protocol["arrival_horizon_slots"])
        for role_key, config_key in (
            ("dt", "simulation.dt_s"),
            ("warmup_time", "simulation.warmup_time_s"),
        ):
            configured = float(get_value(formal_config, config_key))
            if role_key in roles and not math.isclose(
                float(roles[role_key]), configured, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(
                    f"vehicle_roles.json 的 {role_key} 与 configs/dqn/online_joint_dqn.toml 不一致。"
                )
        expected_execution_time_s = execution_slots * float(
            get_value(formal_config, "simulation.dt_s")
        )
        if "execution_horizon_slots" in roles:
            if (
                int(roles.get("arrival_horizon_slots", -1)) != arrival_slots
                or int(roles.get("execution_horizon_slots", -1)) != execution_slots
                or not math.isclose(
                    float(roles.get("episode_time", math.nan)),
                    expected_execution_time_s,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(
                    "vehicle_roles.json arrival/execution time contract differs from formal config."
                )
        elif "episode_time" in roles and not math.isclose(
            float(roles["episode_time"]),
            float(get_value(formal_config, "simulation.episode_time_s")),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "vehicle_roles.json 的 episode_time 与 configs/dqn/online_joint_dqn.toml 不一致。"
            )
        residual_bank = load_residual_bank(residual_file)
        if int(residual_bank["H_p"]) != config.horizon_steps:
            raise ValueError("residual_blocks H_p 与 scenario H_p 不一致。")
        configured_dt = float(get_value(formal_config, "simulation.dt_s"))
        if not math.isclose(
            float(residual_bank["dt"]), configured_dt, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("calibration residual bank dt 与 configs/dqn/online_joint_dqn.toml 不一致。")

        service_catalog = load_json(catalog_file)
        task_config = task_config_from_catalog(service_catalog, formal_config)
        task_config.validate(len(tv_ids))
        request_counts, times = read_request_counts(popularity_file)
        history_request_counts, history_formal_slots = read_history_request_counts(
            popularity_history_file
        )
        checkpoint_payload = torch.load(
            checkpoint_file,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_metadata = checkpoint_payload["metadata"]
        rho_history_length = int(checkpoint_metadata["L_rho"])
        checkpoint_hp = int(checkpoint_metadata["H_p"])
        if checkpoint_hp != config.horizon_steps:
            raise ValueError("GRU checkpoint H_p 与 scenario H_p 不一致。")

        num_slots = request_counts.shape[0]
        if len(history_request_counts) != rho_history_length:
            raise ValueError("prehistory 长度与 GRU checkpoint L_rho 不一致。")
        if history_formal_slots != list(range(-rho_history_length, 0)):
            raise ValueError("prehistory 未覆盖 formal slot -L_rho ... -1。")
        configured_num_slots = int(
            round(
                float(get_value(formal_config, "simulation.episode_time_s"))
                / float(get_value(formal_config, "simulation.dt_s"))
            )
        )
        if num_slots != configured_num_slots:
            raise ValueError(
                f"formal popularity slots={num_slots} 与 configs/dqn/online_joint_dqn.toml={configured_num_slots} 不一致。"
            )
        origin_count = num_slots if args.origin_count is None else int(args.origin_count)
        if not num_slots <= origin_count <= execution_slots:
            raise ValueError(
                "origin-count must cover the arrival horizon and not exceed execution horizon: "
                f"arrival={num_slots}, requested={origin_count}, execution={execution_slots}"
            )
        prediction_request_counts = request_counts
        if origin_count > num_slots:
            # No tasks arrive in the execution tail, so zero counts are known
            # causal observations; no rho_star labels are appended.
            prediction_request_counts = np.concatenate(
                (
                    request_counts,
                    np.zeros(
                        (origin_count - num_slots, request_counts.shape[1]),
                        dtype=request_counts.dtype,
                    ),
                ),
                axis=0,
            )
        origin_slots = np.arange(origin_count, dtype=np.int64)
        dt_s = float(get_value(formal_config, "simulation.dt_s"))
        origin_times = np.asarray(
            [
                times[int(slot)]
                if int(slot) < len(times)
                else float(times[-1] + (int(slot) - len(times) + 1) * dt_s)
                for slot in origin_slots
            ],
            dtype=np.float64,
        )
        scenario_ids = np.arange(config.num_scenarios, dtype=np.int64)
        horizons = np.arange(config.horizon_steps, dtype=np.int64)

        formal_vehicle_prediction_file = resolve_path(
            project_root,
            args.vehicle_prediction_file
            or get_value(
                formal_config,
                "paths.vehicle_predictions",
                "data/generated/vehicle_predictions.csv",
            ),
        )
        formal_vehicle_prediction_rows: list[dict[str, Any]] = []
        nominal_xy, tangents, route_geometries, truth_index = build_vehicle_nominal_predictions(
            project_root,
            roles,
            research_ids,
            residual_bank,
            trajectory_file,
            args.net_file or get_value(formal_config, "paths.sumo_net"),
            args.sumo_home,
            origin_slots,
            config.horizon_steps,
            formal_vehicle_prediction_rows,
            config.vehicle_observation_seed,
            execution_horizon_slots=execution_slots,
        )
        expected_vehicle_prediction_rows = (
            len(origin_slots)
            * config.horizon_steps
            * len(research_ids)
        )
        if len(formal_vehicle_prediction_rows) != expected_vehicle_prediction_rows:
            raise AssertionError(
                "formal vehicle_predictions 行数与 origin×H_p×research 不一致。"
            )

        rho_hat, checkpoint_metadata = predict_rho_hat(
            checkpoint_file,
            history_request_counts,
            prediction_request_counts,
            origin_slots,
            config,
            device,
        )

        rngs_by_origin = [
            make_rng_streams(config.scenario_seed, int(origin_slot))
            for origin_slot in origin_slots.tolist()
        ]
        position_seed = config.scenario_seed
        positions_xy, residual_stats = sample_positions(
            nominal_xy,
            tangents,
            route_geometries,
            residual_bank,
            research_ids,
            config,
            rng=np.random.default_rng(position_seed),
        )
        arrival, service_id, input_size, cpu_cycles, task_stats = sample_tasks(
            rho_hat,
            tv_ids,
            service_catalog,
            task_config,
            config,
            rngs_by_origin,
            origin_slots=origin_slots,
            arrival_horizon_slots=num_slots,
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
            arrival_horizon_slots=num_slots,
        )

        # 同 seed 完全重放：重新生成 residual 和任务随机过程，检查 CRN 接口稳定。
        repeat_position_seed = config.scenario_seed
        repeat_positions, _ = sample_positions(
            nominal_xy,
            tangents,
            route_geometries,
            residual_bank,
            research_ids,
            config,
            rng=np.random.default_rng(repeat_position_seed),
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
            origin_slots=origin_slots,
            arrival_horizon_slots=num_slots,
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
            "schema_version": 3,
            "S": config.num_scenarios,
            "H_p": config.horizon_steps,
            "F": int(service_catalog["service_count"]),
            "TV IDs": tv_ids,
            "SV IDs": sv_ids,
            "research_vehicle_ids": research_ids,
            "scenario_seed": config.scenario_seed,
            "vehicle_observation_seed": config.vehicle_observation_seed,
            "lambda": task_config.arrival_probability,
            "residual_bank_source": "independent_calibration_artifact",
            "arrival_horizon_slots": int(num_slots),
            "execution_horizon_slots": int(execution_slots),
            "formal_time_slot_count": int(origin_count),
            "formal_origin_count": int(len(origin_slots)),
            "execution_tail_request_observation": (
                "known_zero_counts_under_no_new_task_arrivals_contract"
                if origin_count > num_slots
                else "not_requested"
            ),
            "prediction_tail_origins": [
                int(origin)
                for origin in origin_slots
                if int(origin) + config.horizon_steps > num_slots
            ],
            "formal_truth_used_for_diagnostics_only": True,
            "formal_vehicle_prediction_file": str(formal_vehicle_prediction_file),
            "calibration_artifact_metadata_reference": residual_bank["metadata"],
            "cross_vehicle_residual_sampling": "independent",
            "residual_block_sampling": "whole_block_per_vehicle_scenario",
            "residual_sampling_contract": RESIDUAL_SAMPLING_CONTRACT,
            "vehicle_residual_bank_preference": "global_pooled_no_vehicle_id_matching",
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
                "observation_definition": checkpoint_metadata[
                    "observation_definition"
                ],
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
        veh.write_predictions_csv(
            formal_vehicle_prediction_file,
            formal_vehicle_prediction_rows,
        )
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
