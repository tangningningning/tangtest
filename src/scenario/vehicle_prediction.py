#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""道路约束纵向 Kalman replay、因果位置预测和 residual trajectory blocks。

本文件只新增本阶段的预测数据管线，不修改 SUMO 场景、任务或角色生成逻辑。
路网几何使用 SUMO 官方 ``sumolib.net.readNet``、Edge/Lane ``getShape``、
``getLength``、``getOutgoing`` 和 ``getInternalPath`` API：

* https://github.com/eclipse-sumo/sumo/tree/main/tools/sumolib
* https://sumo.dlr.de/docs/Tools/Sumolib.html
* https://sumo.dlr.de/pydoc/sumolib/net.html

本项目没有复制 SUMO 源码；上述 API 属于 Eclipse SUMO（EPL-2.0）官方实现。
Q/R 和 residual bank 只能从显式指定的独立 calibration trajectory 生成；
正式 episode 只读取冻结的 calibration artifact。
"""

from __future__ import annotations

import bisect
import csv
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# ============================================================================
# 集中配置区：均为实现阶段 provisional defaults
# ============================================================================


@dataclass(frozen=True)
class PredictionConfig:
    horizon_steps: int = 5
    sigma_position_m: float = 3.0
    sigma_speed_mps: float = 1.0
    observation_noise_seed: int = 9101
    projection_window_m: float = 200.0
    projection_backtrack_m: float = 50.0
    initial_position_variance_m2: float = 100.0
    initial_speed_variance_m2ps2: float = 25.0
    initial_acceleration_variance_m2ps4: float = 4.0
    covariance_floor: float = 1e-8

    def validate(self) -> None:
        if self.horizon_steps <= 0:
            raise ValueError("H_p 必须大于 0。")
        if self.sigma_position_m < 0.0 or self.sigma_speed_mps < 0.0:
            raise ValueError("观测噪声标准差不能为负数。")
        if self.projection_window_m <= 0.0 or self.projection_backtrack_m < 0.0:
            raise ValueError("route projection window 参数无效。")
        if self.covariance_floor <= 0.0:
            raise ValueError("covariance_floor 必须大于 0。")


DEFAULT_CONFIG = PredictionConfig()

PREDICTION_FIELDS = (
    "origin_slot",
    "origin_time_s",
    "observation_cutoff_slot",
    "observation_cutoff_time_s",
    "vehicle_id",
    "horizon",
    "target_slot",
    "target_time_s",
    "s_pred",
    "x_pred",
    "y_pred",
    "s_true",
    "x_true",
    "y_true",
    "longitudinal_error",
    "lateral_error",
    "e_parallel",
    "e_perp",
)


@dataclass
class TruthSample:
    abs_step: int
    time_s: float
    x_m: float
    y_m: float
    speed_mps: float
    acceleration_mps2: float
    distance_m: float
    road_id: str
    lane_position_m: float
    s_true: float = 0.0
    route_edge_matched: bool = False


@dataclass
class Observation:
    abs_step: int
    time_s: float
    x_obs_m: float
    y_obs_m: float
    s_obs: float
    speed_obs_mps: float
    projection_distance_m: float


@dataclass
class RouteSegment:
    edge_id: str
    edge: Any
    start_s: float
    length_m: float
    shape: np.ndarray
    shape_cumulative_m: np.ndarray
    shape_length_m: float


@dataclass
class RouteGeometry:
    """一个研究车辆 planned route 的连续几何坐标。"""

    vehicle_id: str
    segments: list[RouteSegment]
    total_length_m: float
    edge_to_segments: dict[str, list[int]]

    def point_and_tangent(self, s_m: float) -> tuple[np.ndarray, np.ndarray]:
        """将 route s 映射到合法 nominal x/y，并返回局部切向量。"""
        if not self.segments:
            raise ValueError(f"车辆 {self.vehicle_id} 没有 route geometry。")
        clipped_s = float(np.clip(s_m, 0.0, self.total_length_m))
        starts = [segment.start_s for segment in self.segments]
        index = bisect.bisect_right(starts, clipped_s) - 1
        index = min(max(index, 0), len(self.segments) - 1)
        segment = self.segments[index]
        local_s = float(np.clip(clipped_s - segment.start_s, 0.0, segment.length_m))
        shape_offset = 0.0
        if segment.length_m > 0.0 and segment.shape_length_m > 0.0:
            shape_offset = local_s / segment.length_m * segment.shape_length_m
        return polyline_point_tangent(
            segment.shape,
            segment.shape_cumulative_m,
            shape_offset,
        )

    def candidate_s_for_edge(self, edge_id: str, lane_position_m: float) -> list[float]:
        """返回当前 road_id 在 planned route 中所有可能的连续 s。"""
        candidates: list[float] = []
        for index in self.edge_to_segments.get(edge_id, []):
            segment = self.segments[index]
            local = float(np.clip(lane_position_m, 0.0, segment.length_m))
            candidates.append(segment.start_s + local)
        return candidates

    def project_xy_to_s(
        self,
        x_m: float,
        y_m: float,
        previous_s_m: float | None,
        window_m: float,
        backtrack_m: float,
        reference_s_m: float | None = None,
    ) -> tuple[float, float]:
        """将 noisy GPS 点投影到 planned route，并用上一步 s 限制跳跃。"""
        point = np.asarray([x_m, y_m], dtype=float)
        if reference_s_m is not None:
            lower = max(0.0, reference_s_m - backtrack_m)
            upper = min(self.total_length_m, reference_s_m + window_m)
        elif previous_s_m is None:
            lower, upper = 0.0, self.total_length_m
        else:
            lower = max(0.0, previous_s_m - backtrack_m)
            upper = min(self.total_length_m, previous_s_m + window_m)

        continuity_s = previous_s_m if previous_s_m is not None else reference_s_m
        best = self._project_in_range(point, lower, upper, continuity_s)
        if best is None and reference_s_m is None:
            # 受限范围没有几何候选时，回退到整条 planned route，而不是返回 NaN。
            best = self._project_in_range(point, 0.0, self.total_length_m, previous_s_m)
        if best is None:
            raise RuntimeError(f"车辆 {self.vehicle_id} 的 noisy 点无法投影到 route。")
        return best

    def _project_in_range(
        self,
        point: np.ndarray,
        lower_s: float,
        upper_s: float,
        previous_s_m: float | None,
    ) -> tuple[float, float] | None:
        best_key: tuple[float, float] | None = None
        best_value: tuple[float, float] | None = None
        for segment in self.segments:
            segment_end = segment.start_s + segment.length_m
            if segment_end < lower_s - 1e-9 or segment.start_s > upper_s + 1e-9:
                continue
            if segment.shape.shape[0] == 0:
                continue
            for index in range(len(segment.shape) - 1):
                start = segment.shape[index]
                end = segment.shape[index + 1]
                vector = end - start
                denominator = float(np.dot(vector, vector))
                if denominator <= 1e-12:
                    alpha = 0.0
                else:
                    alpha = float(np.clip(np.dot(point - start, vector) / denominator, 0.0, 1.0))
                projected = start + alpha * vector
                distance_sq = float(np.dot(point - projected, point - projected))
                shape_offset = float(segment.shape_cumulative_m[index])
                shape_piece = float(segment.shape_cumulative_m[index + 1] - shape_offset)
                local_shape_offset = shape_offset + alpha * shape_piece
                if segment.shape_length_m > 0.0:
                    local_s = local_shape_offset / segment.shape_length_m * segment.length_m
                else:
                    local_s = 0.0
                candidate_s = segment.start_s + local_s
                if candidate_s < lower_s - 1e-7 or candidate_s > upper_s + 1e-7:
                    continue
                # 距离优先；相同距离时优先接近上一步，避免平行 route 跳跃。
                progress_distance = (
                    abs(candidate_s - previous_s_m) if previous_s_m is not None else 0.0
                )
                key = (distance_sq, progress_distance)
                if best_key is None or key < best_key:
                    best_key = key
                    best_value = (candidate_s, math.sqrt(distance_sq))
        return best_value


# ============================================================================
# 项目文件、SUMO 官方库和路网 geometry
# ============================================================================








def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def find_network(project_root: Path, explicit: str | None) -> Path:
    if explicit:
        path = resolve_path(project_root, explicit)
        if not path.is_file():
            raise FileNotFoundError(f"指定的 SUMO 路网不存在：{path}")
        return path
    candidates = sorted(
        [
            path
            for path in (project_root / "data").rglob("*")
            if path.is_file()
            and (
                path.name.lower().endswith(".net.xml")
                or path.name.lower().endswith(".net.xml.gz")
            )
        ],
        key=lambda path: path.as_posix().lower(),
    )
    if not candidates:
        raise FileNotFoundError("data/ 下没有 *.net.xml 或 *.net.xml.gz。")
    return candidates[0]


def find_sumo_home(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.environ.get("SUMO_HOME"):
        candidates.append(Path(os.environ["SUMO_HOME"]).expanduser())
    sumo_exe = shutil.which("sumo")
    if sumo_exe:
        executable = Path(sumo_exe).resolve(strict=False)
        candidates.append(executable.parent.parent if executable.parent.name.lower() == "bin" else executable.parent)
    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        if (candidate / "tools" / "sumolib").is_dir():
            return candidate
    raise FileNotFoundError("找不到 SUMO_HOME，请设置环境变量或传入 --sumo-home。")


def load_sumolib(sumo_home: Path) -> Any:
    tools_dir = str(sumo_home / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    try:
        import sumolib  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(f"无法导入 SUMO 官方 sumolib：{tools_dir}") from exc
    return sumolib


def polyline_shape(edge: Any) -> np.ndarray:
    """读取 SUMO Edge geometry；includeJunctions 保持 junction 处几何连续。"""
    shape = edge.getShape(includeJunctions=True)
    if not shape:
        shape = [edge.getFromNode().getCoord(), edge.getToNode().getCoord()]
    array = np.asarray([[float(point[0]), float(point[1])] for point in shape], dtype=float)
    if len(array) == 1:
        array = np.vstack([array, array])
    return array


def polyline_cumulative(shape: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(shape, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(lengths)))


def polyline_point_tangent(
    shape: np.ndarray,
    cumulative_m: np.ndarray,
    offset_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """在 polyline 上插值，并返回单位切向量。"""
    if len(shape) < 2:
        return shape[0].copy(), np.asarray([1.0, 0.0], dtype=float)
    total = float(cumulative_m[-1])
    target = float(np.clip(offset_m, 0.0, total))
    index = int(np.searchsorted(cumulative_m, target, side="right") - 1)
    index = min(max(index, 0), len(shape) - 2)
    start = shape[index]
    end = shape[index + 1]
    piece = float(cumulative_m[index + 1] - cumulative_m[index])
    alpha = 0.0 if piece <= 1e-12 else (target - cumulative_m[index]) / piece
    point = start + alpha * (end - start)
    tangent = end - start
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-12:
        tangent = np.asarray([1.0, 0.0], dtype=float)
    else:
        tangent = tangent / norm
    return point, tangent


def resolve_edge(net: Any, edge_id: str) -> Any:
    try:
        return net.getEdge(edge_id)
    except Exception:
        # 兼容少量外部输入把反向 edge 写成 -<base> 的情况；当前 SUMO 路网
        # 通常已经为反向 edge 保存了带 '-' 的真实 ID，因此优先使用上面的查找。
        if edge_id.startswith("-"):
            return net.getEdge(edge_id[1:])
        raise


def build_route_geometry(
    vehicle_id: str,
    route_edges: list[str],
    net: Any,
) -> RouteGeometry:
    if not route_edges:
        raise ValueError(f"车辆 {vehicle_id} planned route 为空。")

    segments_raw: list[tuple[str, Any]] = []
    resolved_edges = [resolve_edge(net, edge_id) for edge_id in route_edges]
    for index, edge in enumerate(resolved_edges):
        segments_raw.append((edge.getID(), edge))
        if index >= len(resolved_edges) - 1:
            continue
        next_edge = resolved_edges[index + 1]
        connections = edge.getOutgoing().get(next_edge, [])
        if connections:
            # SUMO 官方 getInternalPath 接受从当前 edge 到下一 edge 的
            # Connection 列表，并返回 junction 内部 edge 路径。
            internal_path, _ = net.getInternalPath(connections)
            for internal_edge in internal_path or []:
                segments_raw.append((internal_edge.getID(), internal_edge))

    segments: list[RouteSegment] = []
    edge_to_segments: dict[str, list[int]] = {}
    cumulative_s = 0.0
    for edge_id, edge in segments_raw:
        length_m = max(float(edge.getLength()), 0.0)
        shape = polyline_shape(edge)
        shape_cumulative = polyline_cumulative(shape)
        segment = RouteSegment(
            edge_id=edge_id,
            edge=edge,
            start_s=cumulative_s,
            length_m=length_m,
            shape=shape,
            shape_cumulative_m=shape_cumulative,
            shape_length_m=float(shape_cumulative[-1]),
        )
        edge_to_segments.setdefault(edge_id, []).append(len(segments))
        segments.append(segment)
        cumulative_s += length_m

    if cumulative_s <= 0.0:
        raise ValueError(f"车辆 {vehicle_id} planned route 的累计长度为 0。")
    return RouteGeometry(vehicle_id, segments, cumulative_s, edge_to_segments)




# ============================================================================
# 轨迹读取和连续 route s 构造
# ============================================================================


def load_planned_routes(
    roles: dict[str, Any],
    research_ids: list[str],
) -> dict[str, list[str]]:
    planned = roles.get("planned_routes", {})
    result: dict[str, list[str]] = {}
    for vehicle_id in research_ids:
        route = planned.get(vehicle_id, {})
        edges = [str(edge) for edge in route.get("route_edges", [])]
        if not edges:
            raise ValueError(f"vehicle_roles.json 缺少车辆 {vehicle_id} 的 planned route edges。")
        result[vehicle_id] = edges
    return result


def make_truth_sample(row: dict[str, str], dt_s: float) -> TruthSample:
    time_s = float(row["time_s"])
    abs_step_float = time_s / dt_s
    if not math.isclose(abs_step_float, round(abs_step_float), rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"trajectory time_s={time_s} 无法映射到 dt={dt_s} 的整数 step。")
    return TruthSample(
        abs_step=int(round(abs_step_float)),
        time_s=time_s,
        x_m=float(row["x_m"]),
        y_m=float(row["y_m"]),
        speed_mps=float(row["speed_mps"]),
        acceleration_mps2=float(row["acceleration_mps2"]),
        distance_m=float(row["distance_m"]),
        road_id=str(row["road_id"]),
        lane_position_m=float(row["lane_position_m"]),
    )


def load_truth_trajectories(
    trajectory_file: Path,
    research_ids: list[str],
    dt_s: float,
) -> dict[str, list[TruthSample]]:
    truth = {vehicle_id: [] for vehicle_id in research_ids}
    allowed = set(research_ids)
    with trajectory_file.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "time_s",
            "vehicle_id",
            "x_m",
            "y_m",
            "speed_mps",
            "acceleration_mps2",
            "road_id",
            "lane_position_m",
            "distance_m",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"trajectory.csv 缺少字段：{sorted(missing)}")
        for row in reader:
            vehicle_id = str(row["vehicle_id"])
            if vehicle_id in allowed:
                truth[vehicle_id].append(make_truth_sample(row, dt_s))

    for vehicle_id, samples in truth.items():
        samples.sort(key=lambda sample: sample.abs_step)
        if not samples:
            raise ValueError(f"trajectory.csv 没有研究车辆 {vehicle_id} 的记录。")
        if len({sample.abs_step for sample in samples}) != len(samples):
            raise ValueError(f"车辆 {vehicle_id} 的 trajectory step 存在重复。")
    return truth


def map_truth_to_route_s(
    truth: dict[str, list[TruthSample]],
    route_geometries: dict[str, RouteGeometry],
    config: PredictionConfig,
) -> dict[str, float]:
    """用 road_id/lane_position 映射 route s，junction 内部 edge 由官方拓扑补入。

    distance_m 只作为连续性校验和 road_id 不可匹配时的回退，不直接替代
    planned route cumulative coordinate。这样 lane_position_m 不会被误当成全程 s。
    """
    diagnostics: dict[str, float] = {
        "matched_samples": 0.0,
        "fallback_samples": 0.0,
        "distance_residual_sum": 0.0,
        "distance_residual_sq_sum": 0.0,
        "distance_residual_count": 0.0,
    }
    for vehicle_id, samples in truth.items():
        geometry = route_geometries[vehicle_id]
        previous_s: float | None = None
        distance_offset: float | None = None
        for sample in samples:
            candidates = geometry.candidate_s_for_edge(
                sample.road_id,
                sample.lane_position_m,
            )
            candidate_s: float | None = None
            if candidates:
                if previous_s is None:
                    candidate_s = candidates[0]
                else:
                    forward_limit = previous_s + max(
                        config.projection_window_m,
                        abs(sample.speed_mps) * 4.0,
                    )
                    feasible = [
                        value
                        for value in candidates
                        if value >= previous_s - config.projection_backtrack_m
                        and value <= forward_limit
                    ]
                    candidate_s = min(
                        feasible or candidates,
                        key=lambda value: abs(value - previous_s),
                    )

            if candidate_s is not None:
                sample.s_true = float(np.clip(candidate_s, 0.0, geometry.total_length_m))
                sample.route_edge_matched = True
                diagnostics["matched_samples"] += 1.0
                if distance_offset is None and math.isfinite(sample.distance_m):
                    distance_offset = sample.s_true - sample.distance_m
                if distance_offset is not None and math.isfinite(sample.distance_m):
                    residual = sample.s_true - (sample.distance_m + distance_offset)
                    diagnostics["distance_residual_sum"] += residual
                    diagnostics["distance_residual_sq_sum"] += residual * residual
                    diagnostics["distance_residual_count"] += 1.0
            else:
                diagnostics["fallback_samples"] += 1.0
                if distance_offset is not None and math.isfinite(sample.distance_m):
                    candidate_s = sample.distance_m + distance_offset
                elif previous_s is not None:
                    candidate_s = previous_s + max(sample.speed_mps, 0.0)
                else:
                    candidate_s = sample.distance_m
                sample.s_true = float(np.clip(candidate_s, 0.0, geometry.total_length_m))
                sample.route_edge_matched = False

            previous_s = sample.s_true

    return diagnostics


def validate_route_coordinates(
    truth: dict[str, list[TruthSample]],
    route_geometries: dict[str, RouteGeometry],
) -> dict[str, float]:
    """检查连续 route-s 和 nominal geometry 的基本合法性。"""
    all_s: list[float] = []
    nonmonotonic_count = 0
    for vehicle_id, samples in truth.items():
        geometry = route_geometries[vehicle_id]
        values = np.asarray([sample.s_true for sample in samples], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"车辆 {vehicle_id} 的 route-s 包含 NaN/Inf。")
        if np.any(values < -1e-7) or np.any(values > geometry.total_length_m + 1e-7):
            raise ValueError(f"车辆 {vehicle_id} 的 route-s 超出 planned route 范围。")
        nonmonotonic_count += int(np.sum(np.diff(values) < -1e-7))
        all_s.extend(values.tolist())
        for endpoint in (0.0, geometry.total_length_m):
            point, tangent = geometry.point_and_tangent(endpoint)
            if not np.all(np.isfinite(point)) or not np.all(np.isfinite(tangent)):
                raise ValueError(f"车辆 {vehicle_id} 的 route nominal geometry 无效。")
    if nonmonotonic_count:
        raise ValueError(f"连续 route-s 出现 {nonmonotonic_count} 次反向跳变。")
    return {
        "route_s_min_m": float(min(all_s)) if all_s else 0.0,
        "route_s_max_m": float(max(all_s)) if all_s else 0.0,
        "route_nonmonotonic_count": float(nonmonotonic_count),
        "nominal_geometry_checked": 1.0,
    }


def formal_truth_index(
    truth: dict[str, list[TruthSample]],
    research_ids: list[str],
    warmup_s: float,
    episode_s: float,
    dt_s: float,
) -> tuple[int, dict[str, dict[int, TruthSample]]]:
    num_slots = int(round(episode_s / dt_s))
    warmup_step = int(round(warmup_s / dt_s))
    indexed: dict[str, dict[int, TruthSample]] = {vehicle_id: {} for vehicle_id in research_ids}
    for vehicle_id in research_ids:
        for sample in truth[vehicle_id]:
            relative_slot = sample.abs_step - warmup_step
            if 0 <= relative_slot < num_slots:
                indexed[vehicle_id][relative_slot] = sample
        if set(indexed[vehicle_id]) != set(range(num_slots)):
            raise RuntimeError(f"车辆 {vehicle_id} 的正式 episode 轨迹不完整。")
    return num_slots, indexed


# ============================================================================
# 噪声、Q/R provisional calibration 和 Kalman replay
# ============================================================================


def build_noisy_observations(
    truth: dict[str, list[TruthSample]],
    route_geometries: dict[str, RouteGeometry],
    config: PredictionConfig,
) -> dict[str, list[Observation]]:
    """由当前时刻的 ground truth 模拟传感器；replay 只按 cutoff 读取这些观测。"""
    rng = np.random.default_rng(config.observation_noise_seed)
    observations: dict[str, list[Observation]] = {}
    for vehicle_id in sorted(truth):
        geometry = route_geometries[vehicle_id]
        previous_s: float | None = None
        vehicle_observations: list[Observation] = []
        for sample in truth[vehicle_id]:
            noisy_xy = np.asarray([sample.x_m, sample.y_m]) + rng.normal(
                0.0,
                config.sigma_position_m,
                size=2,
            )
            noisy_speed = sample.speed_mps + float(
                rng.normal(0.0, config.sigma_speed_mps)
            )
            candidates = geometry.candidate_s_for_edge(
                sample.road_id,
                sample.lane_position_m,
            )
            reference_s: float | None = None
            if candidates:
                if previous_s is None:
                    reference_s = candidates[0]
                else:
                    lower = previous_s - config.projection_backtrack_m
                    upper = previous_s + config.projection_window_m
                    feasible = [
                        value
                        for value in candidates
                        if lower <= value <= upper
                    ]
                    reference_s = min(
                        feasible or candidates,
                        key=lambda value: abs(value - previous_s),
                    )
            s_obs, projection_distance = geometry.project_xy_to_s(
                float(noisy_xy[0]),
                float(noisy_xy[1]),
                previous_s,
                config.projection_window_m,
                config.projection_backtrack_m,
                reference_s_m=reference_s,
            )
            previous_s = s_obs
            vehicle_observations.append(
                Observation(
                    abs_step=sample.abs_step,
                    time_s=sample.time_s,
                    x_obs_m=float(noisy_xy[0]),
                    y_obs_m=float(noisy_xy[1]),
                    s_obs=float(s_obs),
                    speed_obs_mps=float(noisy_speed),
                    projection_distance_m=float(projection_distance),
                )
            )
        observations[vehicle_id] = vehicle_observations
    return observations




def state_transition(dt_s: float) -> np.ndarray:
    return np.asarray(
        [
            [1.0, dt_s, 0.5 * dt_s * dt_s],
            [0.0, 1.0, dt_s],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )




def initial_covariance(config: PredictionConfig) -> np.ndarray:
    return np.diag(
        [
            config.initial_position_variance_m2,
            config.initial_speed_variance_m2ps2,
            config.initial_acceleration_variance_m2ps4,
        ]
    ).astype(float)


def kalman_update(
    predicted_state: np.ndarray,
    predicted_covariance: np.ndarray,
    observation: Observation,
    q_measurement: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    measurement_matrix = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    measurement = np.asarray([observation.s_obs, observation.speed_obs_mps], dtype=float)
    innovation = measurement - measurement_matrix @ predicted_state
    innovation_covariance = (
        measurement_matrix @ predicted_covariance @ measurement_matrix.T + q_measurement
    )
    gain = np.linalg.solve(innovation_covariance, measurement_matrix @ predicted_covariance).T
    updated_state = predicted_state + gain @ innovation
    identity = np.eye(3)
    updated_covariance = (identity - gain @ measurement_matrix) @ predicted_covariance
    updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
    return updated_state, updated_covariance


def run_kalman_replay(
    observations: dict[str, list[Observation]],
    q: np.ndarray,
    r: np.ndarray,
    dt_s: float,
    config: PredictionConfig,
) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """从首个可用观测开始顺序 replay，保存每个 cutoff 的 posterior state。"""
    transition = state_transition(dt_s)
    replay: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for vehicle_id in sorted(observations):
        by_step = {observation.abs_step: observation for observation in observations[vehicle_id]}
        all_steps = sorted(by_step)
        if not all_steps:
            raise ValueError(f"车辆 {vehicle_id} 没有可用 noisy observation。")
        state: np.ndarray | None = None
        covariance: np.ndarray | None = None
        vehicle_replay: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        previous_step: int | None = None
        for abs_step in range(all_steps[0], all_steps[-1] + 1):
            observation = by_step.get(abs_step)
            if state is None:
                if observation is None:
                    continue
                # 初始加速度不使用真实 acceleration；它只是 filter 初值。
                state = np.asarray([observation.s_obs, observation.speed_obs_mps, 0.0])
                covariance = initial_covariance(config)
            else:
                state = transition @ state
                covariance = transition @ covariance @ transition.T + q
                if observation is not None:
                    state, covariance = kalman_update(state, covariance, observation, r)
            if previous_step is not None and abs_step - previous_step > 1:
                # 上面的循环会逐步 predict，这里只用于可读性保留 step 轨迹。
                pass
            vehicle_replay[abs_step] = (state.copy(), covariance.copy())
            previous_step = abs_step
        replay[vehicle_id] = vehicle_replay
    return replay


def state_at_cutoff(
    vehicle_replay: dict[int, tuple[np.ndarray, np.ndarray]],
    cutoff_step: int,
    dt_s: float,
) -> np.ndarray:
    """取得 <= cutoff 的 posterior，并只用 A 向前预测到 cutoff。"""
    available = [step for step in vehicle_replay if step <= cutoff_step]
    if not available:
        raise RuntimeError(f"cutoff={cutoff_step} 之前没有可用观测。")
    latest = max(available)
    state = vehicle_replay[latest][0].copy()
    transition = state_transition(dt_s)
    for _ in range(cutoff_step - latest):
        state = transition @ state
    return state


# ============================================================================
# 因果预测、离线 evaluation 和 residual blocks
# ============================================================================


def predict_nominal(
    posterior_state: np.ndarray,
    geometry: RouteGeometry,
    horizon_steps: int,
    dt_s: float,
) -> list[dict[str, Any]]:
    """只由 x_hat(t-1|t-1) 预测；这里不接收任何未来 truth 参数。"""
    transition = state_transition(dt_s)
    state = posterior_state.copy()
    predictions: list[dict[str, Any]] = []
    for horizon in range(horizon_steps):
        state = transition @ state
        raw_s = float(state[0])
        s_pred = float(np.clip(raw_s, 0.0, geometry.total_length_m))
        point, tangent = geometry.point_and_tangent(s_pred)
        if not np.all(np.isfinite(point)) or not np.all(np.isfinite(tangent)):
            raise ValueError(f"车辆 {geometry.vehicle_id} 的 nominal prediction geometry 无效。")
        predictions.append(
            {
                "horizon": horizon,
                "s_pred": s_pred,
                "u_pred": float(state[1]),
                "a_pred": float(state[2]),
                "x_pred": float(point[0]),
                "y_pred": float(point[1]),
                "tangent": tangent.copy(),
                "raw_s_pred": raw_s,
            }
        )
    return predictions










# ============================================================================
# 输出和最小验证
# ============================================================================


def write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()






# ============================================================================
# CLI
# ============================================================================
