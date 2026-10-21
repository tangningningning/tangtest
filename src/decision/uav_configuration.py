#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 Trigger 固定条件下优化 UAV 配置。

本模块只解决冻结框架中的 WHERE：

给定外部 trigger ``z_u^q(t)``、上一时隙 UAV 水平位置、固定 cache
configuration 和同一批 prediction scenarios，选择当前时隙 UAV 二维水平位置。

注意：

- 不决定 trigger/WHEN，不训练 PPO，不调用 Matching/KKT；
- 不做 UAV 轨迹规划，高度固定且只优化 ``[x, y]``；
- ``z=0`` 时严格保持原位置；
- ``z=1`` 时允许重新搜索位置，但如果没有严格更优候选则保持原位置；
- movement cost 只由最终 ``q_old -> q_final`` 决定，不累计搜索过程访问路径；
- candidate 在整个 ``H_p`` 预测窗口内保持固定，并复用同一 scenario slice
  进行 common random numbers 比较。
"""

from __future__ import annotations

import gzip
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common import project_config, system_model
from decision import cache_configuration


@dataclass(frozen=True)
class UavSearchConfig:
    """UAV 配置优化算法超参数。"""

    coarse_num_angles: int
    coarse_radius_fractions: tuple[float, ...]
    refine_rounds: int
    refine_num_angles: int
    refine_radius_shrink: float
    exploration_num_angles: int
    exploration_radius_fractions: tuple[float, ...]
    demand_anchor_scales: tuple[float, ...]
    target_candidate_capacity: int
    max_coordinate_rounds: int
    position_tolerance_m: float
    objective_tolerance: float
    dt_s: float
    max_horizontal_speed_mps: float
    min_safety_distance_m: float
    movement_energy_coefficient_j_per_m: float
    movement_fixed_cost: float
    lambda_movement: float
    movement_energy_reference_j: float
    beta_movement: float

    @property
    def max_move_distance_m(self) -> float:
        return self.dt_s * self.max_horizontal_speed_mps




def uav_search_config_from_project(config: dict[str, Any]) -> UavSearchConfig:
    """从统一 TOML 配置构造 UAV 搜索配置。"""

    get = project_config.get_config_value
    lambda_movement = float(get(config, "objective.lambda_movement"))
    cost_normalization = get(config, "_meta.reward_cost_normalization", {})
    movement_normalization = (
        cost_normalization.get("movement", {})
        if isinstance(cost_normalization, dict)
        else {}
    )
    reference_value = movement_normalization.get("reference_j")
    movement_energy_reference_j = (
        movement_energy_upper_bound_j(config)
        if reference_value is None
        else float(reference_value)
    )
    beta_value = movement_normalization.get("beta_movement")
    beta_movement = (
        lambda_movement * movement_energy_reference_j
        if beta_value is None
        else float(beta_value)
    )
    return UavSearchConfig(
        coarse_num_angles=int(get(config, "algorithm.uav.coarse_num_angles")),
        coarse_radius_fractions=tuple(
            float(value) for value in get(config, "algorithm.uav.coarse_radius_fractions")
        ),
        refine_rounds=int(get(config, "algorithm.uav.refine_rounds")),
        refine_num_angles=int(get(config, "algorithm.uav.refine_num_angles")),
        refine_radius_shrink=float(get(config, "algorithm.uav.refine_radius_shrink")),
        exploration_num_angles=int(get(config, "algorithm.uav.exploration_num_angles")),
        exploration_radius_fractions=tuple(
            float(value)
            for value in get(config, "algorithm.uav.exploration_radius_fractions")
        ),
        demand_anchor_scales=tuple(
            float(value)
            for value in get(config, "algorithm.uav.demand_anchor_scales")
        ),
        target_candidate_capacity=int(get(config, "algorithm.uav.target_candidate_capacity")),
        max_coordinate_rounds=int(get(config, "algorithm.uav.max_coordinate_rounds")),
        position_tolerance_m=float(get(config, "algorithm.uav.position_tolerance_m")),
        objective_tolerance=float(get(config, "algorithm.uav.objective_tolerance")),
        dt_s=float(get(config, "simulation.dt_s")),
        max_horizontal_speed_mps=float(get(config, "nodes.uav.max_horizontal_speed_mps")),
        min_safety_distance_m=float(get(config, "nodes.uav.min_safety_distance_m")),
        movement_energy_coefficient_j_per_m=float(
            get(config, "nodes.uav.movement.movement_energy_coefficient_j_per_m")
        ),
        movement_fixed_cost=float(get(config, "nodes.uav.movement.movement_fixed_cost")),
        lambda_movement=lambda_movement,
        movement_energy_reference_j=movement_energy_reference_j,
        beta_movement=beta_movement,
    )


def movement_energy_upper_bound_j(config: dict[str, Any]) -> float:
    """Return the fixed single-UAV speed-limited system energy upper bound."""

    get = project_config.get_config_value
    uav_count = int(get(config, "nodes.uav.count"))
    dt_s = float(get(config, "simulation.dt_s"))
    max_speed_mps = float(get(config, "nodes.uav.max_horizontal_speed_mps"))
    xi_j_per_m = float(
        get(config, "nodes.uav.movement.movement_energy_coefficient_j_per_m")
    )
    if (
        uav_count < 0
        or not all(
            math.isfinite(value)
            for value in (dt_s, max_speed_mps, xi_j_per_m)
        )
        or dt_s <= 0.0
        or max_speed_mps < 0.0
        or xi_j_per_m < 0.0
    ):
        raise ValueError("UAV 移动能耗参考量参数非法。")
    return float(uav_count * xi_j_per_m * max_speed_mps * dt_s)


def uav_ids_from_project(config: dict[str, Any]) -> list[str]:
    """按配置顺序读取 UAV ID，不把 UAV 数量写死为 2。"""

    instances = project_config.get_config_value(config, "nodes.uav.instances", [])
    return [str(item["id"]) for item in instances]


def initial_uav_positions_from_project(config: dict[str, Any]) -> dict[str, np.ndarray]:
    """读取上一时隙/初始 UAV 水平位置；CLI 最小验证使用。"""

    positions: dict[str, np.ndarray] = {}
    for item in project_config.get_config_value(config, "nodes.uav.instances", []):
        positions[str(item["id"])] = np.asarray(
            [float(item["initial_x_m"]), float(item["initial_y_m"])],
            dtype=np.float64,
        )
    return positions


def read_sumo_conv_boundary(net_path: Path) -> tuple[float, float, float, float]:
    """读取 SUMO net.xml(.gz) 的 convBoundary，复用 map_stratified 的地图语义。"""

    opener = gzip.open if net_path.suffix == ".gz" else open
    with opener(net_path, "rb") as stream:
        for _, element in ET.iterparse(stream, events=("start",)):
            if element.tag.endswith("location"):
                raw = element.attrib.get("convBoundary") or element.attrib.get("origBoundary")
                if raw is None:
                    raise ValueError(f"SUMO net 缺少 convBoundary/origBoundary：{net_path}")
                values = [float(part) for part in raw.split(",")]
                if len(values) != 4:
                    raise ValueError(f"SUMO boundary 格式错误：{raw}")
                x_min, y_min, x_max, y_max = values
                if x_min >= x_max or y_min >= y_max:
                    raise ValueError(f"SUMO boundary 非法：{raw}")
                return x_min, y_min, x_max, y_max
    raise ValueError(f"无法读取 SUMO net location boundary：{net_path}")


def boundary_from_project(config: dict[str, Any]) -> tuple[float, float, float, float]:
    """优先从统一 SUMO net 路径读取 UAV 允许水平区域。"""

    project_root = Path(config.get("_meta", {}).get("project_root", cache_configuration.find_project_root()))
    net_path = cache_configuration.resolve_path(
        project_root,
        project_config.get_config_value(config, "paths.sumo_net"),
    )
    return read_sumo_conv_boundary(net_path)


def cache_nodes_with_uav_positions(
    cache_nodes: dict[str, Any],
    uav_positions: dict[str, np.ndarray],
) -> dict[str, Any]:
    """返回替换 UAV fixed_positions 后的 cache_nodes 浅拷贝。"""

    copied = dict(cache_nodes)
    fixed_positions = {
        str(node_id): np.asarray(position, dtype=np.float64).copy()
        for node_id, position in cache_nodes["fixed_positions"].items()
    }
    for uav_id, xy in uav_positions.items():
        fixed_positions[str(uav_id)] = np.asarray(xy, dtype=np.float64).copy()
    copied["fixed_positions"] = fixed_positions
    return copied


def _as_position_dict(
    positions: dict[str, Any],
    uav_ids: list[str],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for uav_id in uav_ids:
        if uav_id not in positions:
            raise ValueError(f"缺少 UAV 位置：{uav_id}")
        xy = np.asarray(positions[uav_id], dtype=np.float64)
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            raise ValueError(f"UAV 位置必须是 finite 二维坐标：{uav_id}")
        result[uav_id] = xy.copy()
    return result


def _trigger_dict(uav_triggers: dict[str, int], uav_ids: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for uav_id in uav_ids:
        value = int(uav_triggers.get(uav_id, 0))
        if value not in (0, 1):
            raise ValueError(f"UAV trigger 必须是 0/1：{uav_id}")
        result[uav_id] = value
    unknown = sorted(set(uav_triggers) - set(uav_ids))
    if unknown:
        raise ValueError(f"trigger 中包含未知 UAV ID：{unknown}")
    return result


def _same_position(left: np.ndarray, right: np.ndarray, tolerance_m: float) -> bool:
    return float(np.linalg.norm(left - right)) <= tolerance_m


def _inside_boundary(xy: np.ndarray, boundary: tuple[float, float, float, float]) -> bool:
    x_min, y_min, x_max, y_max = boundary
    return bool(x_min <= xy[0] <= x_max and y_min <= xy[1] <= y_max)


def _satisfies_min_safety_distance(
    uav_id: str,
    xy: np.ndarray,
    positions: dict[str, np.ndarray],
    search_config: UavSearchConfig,
) -> bool:
    """检查 candidate 与当前 trial_positions 中其他 UAV 的物理安全距离。"""

    for other_id, other_xy in positions.items():
        if other_id == uav_id:
            continue
        distance = float(np.linalg.norm(xy - other_xy))
        if distance + search_config.position_tolerance_m < search_config.min_safety_distance_m:
            return False
    return True


def _candidate_is_feasible(
    uav_id: str,
    candidate: np.ndarray,
    old_positions: dict[str, np.ndarray],
    trial_positions: dict[str, np.ndarray],
    triggers: dict[str, int],
    boundary: tuple[float, float, float, float],
    search_config: UavSearchConfig,
) -> bool:
    if candidate.shape != (2,) or not np.all(np.isfinite(candidate)):
        return False
    if not _inside_boundary(candidate, boundary):
        return False
    old_xy = old_positions[uav_id]
    displacement = float(np.linalg.norm(candidate - old_xy))
    if displacement > search_config.max_move_distance_m + search_config.position_tolerance_m:
        return False
    if not _satisfies_min_safety_distance(uav_id, candidate, trial_positions, search_config):
        return False
    return True




def _polar_candidates(
    center: np.ndarray,
    radii: list[float],
    num_angles: int,
) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for radius in radii:
        if radius <= 0.0:
            continue
        for angle_index in range(num_angles):
            theta = 2.0 * math.pi * angle_index / num_angles
            candidates.append(
                center
                + radius
                * np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float64)
            )
    return candidates






@dataclass(frozen=True)
class UavTargetCandidate:
    """One deterministic feasible MOVE target supplied to the policy."""

    uav_id: str
    before_position: np.ndarray
    target_position: np.ndarray
    proposal_source: str


def _project_target_to_move_ball(
    old_xy: np.ndarray,
    target_xy: np.ndarray,
    max_move_distance_m: float,
) -> np.ndarray:
    old = np.asarray(old_xy, dtype=np.float64).reshape(2)
    target = np.asarray(target_xy, dtype=np.float64).reshape(2)
    delta = target - old
    distance = float(np.linalg.norm(delta))
    if distance <= float(max_move_distance_m):
        return target.copy()
    if distance <= 1.0e-12:
        return old.copy()
    return old + delta * (float(max_move_distance_m) / distance)


def _service_aware_uav_demand_weights(
    scenario_slice: dict[str, Any],
    cache_state: np.ndarray,
    cache_nodes: dict[str, Any],
    uav_id: str,
    beta_discount: float = 1.0,
    prediction_horizon_mask: Any = None,
) -> tuple[tuple[str, ...], np.ndarray]:
    service_count = int(np.asarray(cache_state).shape[1])
    vehicle_ids = tuple(str(value) for value in scenario_slice["vehicle_ids"].tolist())
    demand = cache_configuration.predicted_vehicle_service_demand(
        scenario_slice,
        service_count,
        beta_discount=float(beta_discount),
        prediction_horizon_mask=prediction_horizon_mask,
    )
    node_ids = [str(value) for value in cache_nodes["node_ids"]]
    if str(uav_id) not in node_ids:
        raise KeyError(f"UAV cache node not found for demand-aware proposal: {uav_id}")
    node_index = node_ids.index(str(uav_id))
    cache_vector = np.asarray(cache_state[node_index], dtype=np.float64)
    weights = np.asarray(demand @ cache_vector, dtype=np.float64)
    if weights.shape != (len(vehicle_ids),) or not np.all(np.isfinite(weights)):
        raise FloatingPointError("service-aware UAV vehicle weights are invalid")
    return vehicle_ids, np.maximum(weights, 0.0)


def enumerate_uav_target_candidates(
    current_uav_positions: dict[str, Any],
    previous_uav_positions: dict[str, Any],
    uav_id: str,
    uav_triggers: dict[str, int],
    project_cfg: dict[str, Any],
    flight_boundary: tuple[float, float, float, float],
    *,
    scenario_slice: dict[str, Any],
    cache_state: np.ndarray,
    cache_nodes: dict[str, Any],
    predicted_vehicle_positions: dict[str, Any],
    prediction_horizon_mask: Any = None,
) -> list[UavTargetCandidate]:
    """Return demand-aware MOVE targets plus a small deterministic geometry set.

    The candidate set excludes STAY; STAY remains exclusively the high-level
    operation head.  Every returned point is checked against the existing
    boundary, one-slot displacement, and sequential multi-UAV safety rules.
    ``prediction_horizon_mask`` limits demand weights when supplied; ``None``
    preserves the original all-horizon semantics for other callers.
    """

    uav_ids = uav_ids_from_project(project_cfg)
    current = _as_position_dict(current_uav_positions, uav_ids)
    previous = _as_position_dict(previous_uav_positions, uav_ids)
    triggers = _trigger_dict(uav_triggers, uav_ids)
    target_id = str(uav_id)
    if target_id not in current:
        raise KeyError(f"未知 UAV ID：{target_id}")
    if triggers[target_id] != 1:
        raise ValueError("只有 MOVE trigger=1 的 UAV 才枚举 target positions")
    search_config = uav_search_config_from_project(project_cfg)
    before = current[target_id].copy()
    vehicle_ids, vehicle_weights = _service_aware_uav_demand_weights(
        scenario_slice,
        cache_state,
        cache_nodes,
        target_id,
        beta_discount=float(
            project_config.get_config_value(project_cfg, "cache.discount_beta", 1.0)
        ),
        prediction_horizon_mask=prediction_horizon_mask,
    )
    ordered_positions = np.asarray(
        [predicted_vehicle_positions[vehicle_id] for vehicle_id in vehicle_ids],
        dtype=np.float64,
    )
    if ordered_positions.shape != (len(vehicle_ids), 2) or not np.all(
        np.isfinite(ordered_positions)
    ):
        raise ValueError("predicted_vehicle_positions must be finite and aligned")
    total_weight = float(np.sum(vehicle_weights))
    anchors: list[tuple[str, np.ndarray]] = []
    if total_weight > 1.0e-12:
        centroid = np.sum(
            ordered_positions * vehicle_weights[:, None],
            axis=0,
        ) / total_weight
        centroid = _project_target_to_move_ball(
            before,
            centroid,
            search_config.max_move_distance_m,
        )
        anchors.append(("demand_centroid", centroid))
        for scale in search_config.demand_anchor_scales:
            if not math.isfinite(float(scale)) or float(scale) <= 0.0:
                continue
            projected = before + float(scale) * (centroid - before)
            anchors.append(("projected_demand", projected))

    exploration_radii = [
        float(fraction) * search_config.max_move_distance_m
        for fraction in search_config.exploration_radius_fractions
        if math.isfinite(float(fraction)) and float(fraction) > 0.0
    ]
    raw_candidates: list[tuple[str, np.ndarray]] = anchors
    raw_candidates.extend(
        (
            "geometry_exploration",
            candidate,
        )
        for candidate in _polar_candidates(
            before,
            exploration_radii,
            search_config.exploration_num_angles,
        )
    )
    candidates: list[UavTargetCandidate] = []
    seen: set[bytes] = set()
    for source, raw_candidate in raw_candidates:
        candidate = np.asarray(raw_candidate, dtype=np.float64).reshape(-1)
        if candidate.shape != (2,) or not np.all(np.isfinite(candidate)):
            continue
        if _same_position(candidate, before, search_config.position_tolerance_m):
            continue
        trial = {key: xy.copy() for key, xy in current.items()}
        trial[target_id] = candidate.copy()
        if not _candidate_is_feasible(
            target_id,
            candidate,
            previous,
            trial,
            triggers,
            flight_boundary,
            search_config,
        ):
            continue
        key = np.ascontiguousarray(candidate, dtype=np.float64).tobytes()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            UavTargetCandidate(
                uav_id=target_id,
                before_position=before.copy(),
                target_position=candidate.copy(),
                proposal_source=str(source),
            )
        )
    if len(candidates) > int(search_config.target_candidate_capacity):
        raise RuntimeError(
            "UAV target proposal exceeded configured engineering capacity: "
            f"count={len(candidates)}, capacity={search_config.target_candidate_capacity}"
        )
    return candidates


def apply_uav_target_position(
    current_uav_positions: dict[str, np.ndarray],
    previous_uav_positions: dict[str, Any],
    uav_id: str,
    target_position: np.ndarray,
    uav_triggers: dict[str, int],
    project_cfg: dict[str, Any],
    flight_boundary: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Validate/apply a policy-selected MOVE target position."""

    target = np.asarray(target_position, dtype=np.float64).reshape(-1)
    current = _as_position_dict(current_uav_positions, uav_ids_from_project(project_cfg))
    before = current[str(uav_id)].copy()
    search_config = uav_search_config_from_project(project_cfg)
    if _same_position(target, before, search_config.position_tolerance_m):
        raise ValueError("MOVE target position must differ from current position")
    applied = apply_uav_candidate(
        current_uav_positions,
        previous_uav_positions,
        uav_id,
        target,
        uav_triggers,
        project_cfg,
        flight_boundary,
    )
    applied["conditional_action"] = "TARGET_POSITION"
    applied["target_position"] = np.asarray(applied["after_position"], dtype=np.float64).copy()
    return applied


def apply_uav_candidate(
    current_uav_positions: dict[str, np.ndarray],
    previous_uav_positions: dict[str, Any],
    uav_id: str,
    candidate_position: np.ndarray,
    uav_triggers: dict[str, int],
    project_cfg: dict[str, Any],
    flight_boundary: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Validate and apply one selected position to the mutable UAV state."""

    uav_ids = uav_ids_from_project(project_cfg)
    if list(current_uav_positions) != uav_ids:
        raise ValueError("current_uav_positions 必须保持 configured UAV ID 顺序。")
    previous = _as_position_dict(previous_uav_positions, uav_ids)
    current = _as_position_dict(current_uav_positions, uav_ids)
    triggers = _trigger_dict(uav_triggers, uav_ids)
    target = str(uav_id)
    if target not in current:
        raise KeyError(f"未知 UAV ID：{target}")
    candidate = np.asarray(candidate_position, dtype=np.float64).reshape(-1)
    trial = {key: value.copy() for key, value in current.items()}
    if candidate.shape == (2,):
        trial[target] = candidate.copy()
    search_config = uav_search_config_from_project(project_cfg)
    if not _candidate_is_feasible(
        target,
        candidate,
        previous,
        trial,
        triggers,
        flight_boundary,
        search_config,
    ):
        raise ValueError(
            f"UAV candidate 违反 boundary/displacement/separation 约束：{target}"
        )
    before = current[target].copy()
    current_uav_positions[target] = candidate.copy()
    return {
        "uav_id": target,
        "operation_type": (
            "STAY"
            if _same_position(candidate, before, search_config.position_tolerance_m)
            else "MOVE"
        ),
        "before_position": before,
        "after_position": candidate.copy(),
        "distance_m": float(np.linalg.norm(candidate - before)),
        "actual_moved": bool(
            not _same_position(candidate, before, search_config.position_tolerance_m)
        ),
    }








def validate_uav_configuration_transition(
    previous_uav_positions: dict[str, Any],
    final_uav_positions: dict[str, Any],
    uav_triggers: dict[str, int],
    project_cfg: dict[str, Any],
    flight_boundary: tuple[float, float, float, float],
) -> None:
    """Validate a complete one-slot UAV transition with production constraints."""

    uav_ids = uav_ids_from_project(project_cfg)
    previous = _as_position_dict(previous_uav_positions, uav_ids)
    final = _as_position_dict(final_uav_positions, uav_ids)
    triggers = _trigger_dict(uav_triggers, uav_ids)
    search_config = uav_search_config_from_project(project_cfg)
    for uav_id in uav_ids:
        if triggers[uav_id] == 0 and not _same_position(
            previous[uav_id], final[uav_id], search_config.position_tolerance_m
        ):
            raise ValueError(f"UAV {uav_id} trigger=0 但位置发生变化。")
        if not _candidate_is_feasible(
            uav_id,
            final[uav_id],
            previous,
            final,
            triggers,
            flight_boundary,
            search_config,
        ):
            raise ValueError(f"UAV {uav_id} final position 违反正式移动约束。")


def movement_summary(
    old_positions: dict[str, np.ndarray],
    new_positions: dict[str, np.ndarray],
    triggers: dict[str, int],
    search_config: UavSearchConfig,
) -> tuple[
    float,
    float,
    dict[str, float],
    dict[str, float],
    dict[str, float],
    float,
]:
    """用 system_model.uav_movement_cost 计算一次性最终移动成本。"""

    total_energy = 0.0
    total_cost = 0.0
    displacement: dict[str, float] = {}
    energy_by_uav: dict[str, float] = {}
    cost_by_uav: dict[str, float] = {}
    normalized_energy = 0.0
    for uav_id in old_positions:
        result = system_model.uav_movement_cost(
            old_positions[uav_id],
            new_positions[uav_id],
            triggers[uav_id],
            search_config.dt_s,
            search_config.max_horizontal_speed_mps,
            search_config.movement_energy_coefficient_j_per_m,
            search_config.movement_fixed_cost,
            search_config.movement_energy_reference_j,
            search_config.beta_movement,
            tolerance_m=search_config.position_tolerance_m,
        )
        displacement[uav_id] = float(result["distance_m"])
        energy_by_uav[uav_id] = float(result["movement_energy_j"])
        cost_by_uav[uav_id] = float(result["movement_cost"])
        normalized_energy += float(result["movement_energy_normalized"])
        total_energy += energy_by_uav[uav_id]
        total_cost += cost_by_uav[uav_id]
    energy_tolerance_j = max(
        1.0e-12, search_config.movement_energy_reference_j * 1.0e-12
    )
    if total_energy > search_config.movement_energy_reference_j + energy_tolerance_j:
        raise AssertionError(
            "joint actual UAV movement energy exceeds the fixed normalization reference"
        )
    return (
        total_energy,
        total_cost,
        displacement,
        energy_by_uav,
        cost_by_uav,
        normalized_energy,
    )
