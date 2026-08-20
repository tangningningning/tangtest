#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trigger-conditioned UAV configuration optimization.

本模块只解决冻结框架中的 WHERE：

给定外部 trigger ``z_u^q(t)``、上一时隙 UAV 水平位置、固定 cache
configuration 和同一批 prediction scenarios，选择当前时隙 UAV 二维水平位置。

注意：

- 不决定 trigger/WHEN，不训练 PPO，不调用 Matching/KKT；
- 不做 UAV 轨迹规划，高度固定且只优化 ``[x, y]``；
- ``z=0`` 时严格保持原位置；
- ``z=1`` 时 trigger 已经发生，必须在 feasible-different 候选中选一个位置，
  即使预测 objective 比旧位置更差；
- movement cost 只由最终 ``q_old -> q_final`` 决定，不累计搜索过程访问路径；
- candidate 在整个 ``H_p`` 预测窗口内保持固定，并复用同一 scenario slice
  进行 common random numbers 比较。
"""

from __future__ import annotations

import argparse
import gzip
import json
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
    max_coordinate_rounds: int
    position_tolerance_m: float
    objective_tolerance: float
    dt_s: float
    max_horizontal_speed_mps: float
    movement_energy_coefficient_j_per_m: float
    movement_fixed_cost: float
    lambda_movement: float

    @property
    def max_move_distance_m(self) -> float:
        return self.dt_s * self.max_horizontal_speed_mps


@dataclass(frozen=True)
class UavOptimizationResult:
    """UAV optimizer 的紧凑返回结构。"""

    final_positions: dict[str, tuple[float, float]]
    trigger_feasible: bool
    trigger_feasible_by_uav: dict[str, bool]
    reason: str
    predictive_utility: float
    movement_energy: float
    movement_cost: float
    objective_value: float
    per_uav_displacement_m: dict[str, float]
    per_uav_movement_energy: dict[str, float]
    per_uav_movement_cost: dict[str, float]
    search_rounds: int
    accepted_improvements: int
    converged: bool
    stop_reason: str


def uav_search_config_from_project(config: dict[str, Any]) -> UavSearchConfig:
    """从统一 TOML 配置构造 UAV 搜索配置。"""

    get = project_config.get_config_value
    return UavSearchConfig(
        coarse_num_angles=int(get(config, "algorithm.uav.coarse_num_angles")),
        coarse_radius_fractions=tuple(
            float(value) for value in get(config, "algorithm.uav.coarse_radius_fractions")
        ),
        refine_rounds=int(get(config, "algorithm.uav.refine_rounds")),
        refine_num_angles=int(get(config, "algorithm.uav.refine_num_angles")),
        refine_radius_shrink=float(get(config, "algorithm.uav.refine_radius_shrink")),
        max_coordinate_rounds=int(get(config, "algorithm.uav.max_coordinate_rounds")),
        position_tolerance_m=float(get(config, "algorithm.uav.position_tolerance_m")),
        objective_tolerance=float(get(config, "algorithm.uav.objective_tolerance")),
        dt_s=float(get(config, "simulation.dt_s")),
        max_horizontal_speed_mps=float(get(config, "nodes.uav.max_horizontal_speed_mps")),
        movement_energy_coefficient_j_per_m=float(
            get(config, "nodes.uav.movement.movement_energy_coefficient_j_per_m")
        ),
        movement_fixed_cost=float(get(config, "nodes.uav.movement.movement_fixed_cost")),
        lambda_movement=float(get(config, "objective.lambda_movement")),
    )


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


def _no_exact_overlap(
    uav_id: str,
    xy: np.ndarray,
    positions: dict[str, np.ndarray],
    tolerance_m: float,
) -> bool:
    """当前没有统一 safety-distance 数值；至少禁止两个 UAV 完全重合。"""

    for other_id, other_xy in positions.items():
        if other_id == uav_id:
            continue
        if _same_position(xy, other_xy, tolerance_m):
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
    if triggers[uav_id] == 1 and _same_position(candidate, old_xy, search_config.position_tolerance_m):
        return False
    if not _no_exact_overlap(uav_id, candidate, trial_positions, search_config.position_tolerance_m):
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


def initialization_candidates(
    old_xy: np.ndarray,
    search_config: UavSearchConfig,
) -> list[np.ndarray]:
    """trigger-conditioned 初始化只看基于 q_old 的可行移动圆。"""

    radii = [
        fraction * search_config.max_move_distance_m
        for fraction in search_config.coarse_radius_fractions
    ]
    return _polar_candidates(old_xy, radii, search_config.coarse_num_angles)


def coordinate_candidates(
    old_xy: np.ndarray,
    current_xy: np.ndarray,
    search_config: UavSearchConfig,
) -> list[np.ndarray]:
    """生成 coarse + refine 候选；最终 feasibility 仍按 q_old 检查真实单 slot 位移。"""

    candidates = initialization_candidates(old_xy, search_config)
    radius = search_config.max_move_distance_m
    for _ in range(search_config.refine_rounds):
        radius *= search_config.refine_radius_shrink
        candidates.extend(_polar_candidates(current_xy, [radius], search_config.refine_num_angles))
    return candidates


def movement_summary(
    old_positions: dict[str, np.ndarray],
    new_positions: dict[str, np.ndarray],
    triggers: dict[str, int],
    search_config: UavSearchConfig,
) -> tuple[float, float, dict[str, float], dict[str, float], dict[str, float]]:
    """用 system_model.uav_movement_cost 计算一次性最终移动成本。"""

    total_energy = 0.0
    total_cost = 0.0
    displacement: dict[str, float] = {}
    energy_by_uav: dict[str, float] = {}
    cost_by_uav: dict[str, float] = {}
    for uav_id in old_positions:
        result = system_model.uav_movement_cost(
            old_positions[uav_id],
            new_positions[uav_id],
            triggers[uav_id],
            search_config.dt_s,
            search_config.max_horizontal_speed_mps,
            search_config.movement_energy_coefficient_j_per_m,
            search_config.movement_fixed_cost,
            search_config.lambda_movement,
            tolerance_m=search_config.position_tolerance_m,
        )
        displacement[uav_id] = float(result["distance_m"])
        energy_by_uav[uav_id] = float(result["movement_energy_j"])
        cost_by_uav[uav_id] = float(result["movement_cost"])
        total_energy += energy_by_uav[uav_id]
        total_cost += cost_by_uav[uav_id]
    return total_energy, total_cost, displacement, energy_by_uav, cost_by_uav


def evaluate_uav_configuration(
    positions: dict[str, np.ndarray],
    old_positions: dict[str, np.ndarray],
    triggers: dict[str, int],
    scenario_slice: dict[str, Any],
    cache_state: np.ndarray,
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    search_config: UavSearchConfig,
    cache_cost_constant: float = 0.0,
) -> dict[str, Any]:
    """评估固定 UAV configuration 的预测目标。

    使用 cache_configuration.evaluate_fixed_configuration 复用 Cache surrogate：
    Local/SV/RSU/UAV/Cloud fallback、cache availability、range surrogate、
    bandwidth/CPU 近似和 UAV air channel 全部保持同一口径。所有候选共享同一
    scenario_slice，不重新采样，满足 common random numbers。
    """

    nodes = cache_nodes_with_uav_positions(cache_nodes, positions)
    utility = cache_configuration.evaluate_fixed_configuration(
        cache_state,
        scenario_slice,
        nodes,
        cache_config,
    )["predicted_utility"]
    energy, move_cost, displacement, energy_by_uav, cost_by_uav = movement_summary(
        old_positions,
        positions,
        triggers,
        search_config,
    )
    objective = float(utility - move_cost - cache_cost_constant)
    if not math.isfinite(objective):
        raise AssertionError("UAV objective 非 finite。")
    return {
        "predictive_utility": float(utility),
        "movement_energy": float(energy),
        "movement_cost": float(move_cost),
        "objective_value": objective,
        "per_uav_displacement_m": displacement,
        "per_uav_movement_energy": energy_by_uav,
        "per_uav_movement_cost": cost_by_uav,
    }


def _result_from_positions(
    positions: dict[str, np.ndarray],
    old_positions: dict[str, np.ndarray],
    triggers: dict[str, int],
    scenario_slice: dict[str, Any],
    cache_state: np.ndarray,
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    search_config: UavSearchConfig,
    feasible_by_uav: dict[str, bool],
    reason: str,
    search_rounds: int,
    accepted_improvements: int,
    converged: bool,
    stop_reason: str,
    cache_cost_constant: float,
) -> UavOptimizationResult:
    evaluated = evaluate_uav_configuration(
        positions,
        old_positions,
        triggers,
        scenario_slice,
        cache_state,
        cache_nodes,
        cache_config,
        search_config,
        cache_cost_constant,
    )
    return UavOptimizationResult(
        final_positions={
            uav_id: (float(xy[0]), float(xy[1])) for uav_id, xy in positions.items()
        },
        trigger_feasible=bool(all(feasible_by_uav[uav_id] for uav_id, z in triggers.items() if z == 1)),
        trigger_feasible_by_uav=dict(feasible_by_uav),
        reason=reason,
        predictive_utility=float(evaluated["predictive_utility"]),
        movement_energy=float(evaluated["movement_energy"]),
        movement_cost=float(evaluated["movement_cost"]),
        objective_value=float(evaluated["objective_value"]),
        per_uav_displacement_m=dict(evaluated["per_uav_displacement_m"]),
        per_uav_movement_energy=dict(evaluated["per_uav_movement_energy"]),
        per_uav_movement_cost=dict(evaluated["per_uav_movement_cost"]),
        search_rounds=search_rounds,
        accepted_improvements=accepted_improvements,
        converged=converged,
        stop_reason=stop_reason,
    )


def optimize_uav_configuration(
    current_positions: dict[str, Any],
    uav_triggers: dict[str, int],
    scenario_slice: dict[str, Any],
    cache_state: np.ndarray,
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    flight_boundary: tuple[float, float, float, float],
    cache_cost_constant: float = 0.0,
    initial_candidate_positions: dict[str, Any] | None = None,
) -> UavOptimizationResult:
    """给定 trigger 后优化 UAV 当前时隙二维位置配置。

    核心语义：

    1. ``z=0`` 的 UAV 不进入搜索，最终位置必须等于旧位置；
    2. ``z=1`` 的 UAV 先执行 trigger-conditioned feasible initialization：
       候选集合不包含旧位置，且即使所有候选 objective 都比旧配置差，也必须选
       best feasible different candidate；
    3. 初始化完成后才做 coordinate local search，且只接受 strict improvement；
    4. 多 UAV 按固定 ID 顺序逐架更新，不做 Cartesian product 爆炸搜索；
    5. movement constraint 和 energy/cost 均基于最终 ``q_old -> q_final``。

    ``current_positions`` 是上一时隙真实 reference positions。外层 joint
    coordinate refinement 可通过 ``initial_candidate_positions`` 指定搜索起点；
    但所有速度约束、movement cost 和 z=1 different 检查仍相对 reference。
    """

    uav_ids = uav_ids_from_project(project_cfg)
    search_config = uav_search_config_from_project(project_cfg)
    old_positions = _as_position_dict(current_positions, uav_ids)
    triggers = _trigger_dict(uav_triggers, uav_ids)
    if initial_candidate_positions is None:
        final_positions = {uav_id: xy.copy() for uav_id, xy in old_positions.items()}
    else:
        final_positions = _as_position_dict(initial_candidate_positions, uav_ids)
    feasible_by_uav = {uav_id: True for uav_id in uav_ids}
    triggered_ids = [uav_id for uav_id in uav_ids if triggers[uav_id] == 1]

    if not triggered_ids:
        return _result_from_positions(
            final_positions,
            old_positions,
            triggers,
            scenario_slice,
            cache_state,
            cache_nodes,
            cache_config,
            search_config,
            feasible_by_uav,
            "all_triggers_zero",
            0,
            0,
            True,
            "no_trigger_no_search",
            cache_cost_constant,
        )

    if initial_candidate_positions is None:
        # Trigger 初始化：old position 只可作为 baseline/reference，不允许作为 z=1 最终候选。
        for uav_id in triggered_ids:
            best_position: np.ndarray | None = None
            best_objective = float("-inf")
            for candidate in initialization_candidates(old_positions[uav_id], search_config):
                trial_positions = {key: value.copy() for key, value in final_positions.items()}
                trial_positions[uav_id] = candidate
                if not _candidate_is_feasible(
                    uav_id,
                    candidate,
                    old_positions,
                    trial_positions,
                    triggers,
                    flight_boundary,
                    search_config,
                ):
                    continue
                value = evaluate_uav_configuration(
                    trial_positions,
                    old_positions,
                    triggers,
                    scenario_slice,
                    cache_state,
                    cache_nodes,
                    cache_config,
                    search_config,
                    cache_cost_constant,
                )["objective_value"]
                if value > best_objective + search_config.objective_tolerance:
                    best_objective = float(value)
                    best_position = candidate.copy()
            if best_position is None:
                feasible_by_uav[uav_id] = False
                return _result_from_positions(
                    final_positions,
                    old_positions,
                    triggers,
                    scenario_slice,
                    cache_state,
                    cache_nodes,
                    cache_config,
                    search_config,
                    feasible_by_uav,
                    f"{uav_id}: no different feasible candidate",
                    0,
                    0,
                    False,
                    "infeasible_trigger",
                    cache_cost_constant,
                )
            final_positions[uav_id] = best_position
    else:
        # 外层 refinement 起点必须已经满足 trigger-conditioned 语义；
        # 这里不重新做强制初始化，但仍按 Q_prev 检查不同性、边界和移动距离。
        for uav_id in uav_ids:
            if triggers[uav_id] == 0 and not _same_position(
                final_positions[uav_id],
                old_positions[uav_id],
                search_config.position_tolerance_m,
            ):
                raise ValueError("initial_candidate_positions 违反 z=0 reference position 不变语义。")
            if triggers[uav_id] == 1 and not _candidate_is_feasible(
                uav_id,
                final_positions[uav_id],
                old_positions,
                final_positions,
                triggers,
                flight_boundary,
                search_config,
            ):
                feasible_by_uav[uav_id] = False
        if not bool(all(feasible_by_uav[uav_id] for uav_id in triggered_ids)):
            return _result_from_positions(
                final_positions,
                old_positions,
                triggers,
                scenario_slice,
                cache_state,
                cache_nodes,
                cache_config,
                search_config,
                feasible_by_uav,
                "initial_candidate_positions infeasible relative to reference",
                0,
                0,
                False,
                "infeasible_initial_candidate",
                cache_cost_constant,
            )

    current_eval = evaluate_uav_configuration(
        final_positions,
        old_positions,
        triggers,
        scenario_slice,
        cache_state,
        cache_nodes,
        cache_config,
        search_config,
        cache_cost_constant,
    )
    current_objective = float(current_eval["objective_value"])
    accepted = 0
    completed_rounds = 0

    for round_index in range(1, search_config.max_coordinate_rounds + 1):
        completed_rounds = round_index
        round_improved = False
        for uav_id in triggered_ids:
            best_position: np.ndarray | None = None
            best_objective = current_objective
            for candidate in coordinate_candidates(
                old_positions[uav_id],
                final_positions[uav_id],
                search_config,
            ):
                trial_positions = {key: value.copy() for key, value in final_positions.items()}
                trial_positions[uav_id] = candidate
                if not _candidate_is_feasible(
                    uav_id,
                    candidate,
                    old_positions,
                    trial_positions,
                    triggers,
                    flight_boundary,
                    search_config,
                ):
                    continue
                value = float(
                    evaluate_uav_configuration(
                        trial_positions,
                        old_positions,
                        triggers,
                        scenario_slice,
                        cache_state,
                        cache_nodes,
                        cache_config,
                        search_config,
                        cache_cost_constant,
                    )["objective_value"]
                )
                if value > best_objective + search_config.objective_tolerance:
                    best_objective = value
                    best_position = candidate.copy()
            if best_position is not None:
                final_positions[uav_id] = best_position
                current_objective = best_objective
                accepted += 1
                round_improved = True
        if not round_improved:
            return _result_from_positions(
                final_positions,
                old_positions,
                triggers,
                scenario_slice,
                cache_state,
                cache_nodes,
                cache_config,
                search_config,
                feasible_by_uav,
                "feasible",
                completed_rounds,
                accepted,
                True,
                "converged_by_no_improvement",
                cache_cost_constant,
            )

    return _result_from_positions(
        final_positions,
        old_positions,
        triggers,
        scenario_slice,
        cache_state,
        cache_nodes,
        cache_config,
        search_config,
        feasible_by_uav,
        "feasible",
        completed_rounds,
        accepted,
        False,
        "stopped_by_max_rounds",
        cache_cost_constant,
    )


def _result_to_dict(result: UavOptimizationResult) -> dict[str, Any]:
    return {
        "final_positions": result.final_positions,
        "trigger_feasible": result.trigger_feasible,
        "trigger_feasible_by_uav": result.trigger_feasible_by_uav,
        "reason": result.reason,
        "predictive_utility": result.predictive_utility,
        "movement_energy": result.movement_energy,
        "movement_cost": result.movement_cost,
        "objective_value": result.objective_value,
        "per_uav_displacement_m": result.per_uav_displacement_m,
        "per_uav_movement_energy": result.per_uav_movement_energy,
        "per_uav_movement_cost": result.per_uav_movement_cost,
        "search_rounds": result.search_rounds,
        "accepted_improvements": result.accepted_improvements,
        "converged": result.converged,
        "stop_reason": result.stop_reason,
    }


def validate_uav_optimizer_once(
    origin_scenario: dict[str, Any],
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    project_cfg: dict[str, Any],
    flight_boundary: tuple[float, float, float, float],
) -> dict[str, UavOptimizationResult]:
    """只用一个 origin 做四种 trigger pattern 的最小验证。"""

    previous_cache = cache_configuration.deterministic_initial_cache(
        origin_scenario,
        service_catalog,
        cache_nodes,
        cache_config,
    )
    current_positions = initial_uav_positions_from_project(project_cfg)
    uav_ids = uav_ids_from_project(project_cfg)
    search_config = uav_search_config_from_project(project_cfg)
    patterns = {
        "00": {uav_id: 0 for uav_id in uav_ids},
        "10": {uav_id: int(index == 0) for index, uav_id in enumerate(uav_ids)},
        "01": {uav_id: int(index == 1) for index, uav_id in enumerate(uav_ids)},
        "11": {uav_id: 1 for uav_id in uav_ids},
    }
    results: dict[str, UavOptimizationResult] = {}
    for label, trigger in patterns.items():
        result = optimize_uav_configuration(
            current_positions,
            trigger,
            origin_scenario,
            previous_cache,
            project_cfg,
            cache_nodes,
            cache_config,
            flight_boundary,
        )
        results[label] = result
        if not result.trigger_feasible:
            raise AssertionError(f"最小验证 trigger={label} unexpectedly infeasible: {result.reason}")

        final_np = {
            uav_id: np.asarray(result.final_positions[uav_id], dtype=np.float64)
            for uav_id in uav_ids
        }
        for uav_id in uav_ids:
            old_xy = current_positions[uav_id]
            new_xy = final_np[uav_id]
            moved = not _same_position(old_xy, new_xy, search_config.position_tolerance_m)
            if trigger[uav_id] == 0 and moved:
                raise AssertionError(f"trigger={label} 中 {uav_id} z=0 但发生移动。")
            if trigger[uav_id] == 1 and not moved:
                raise AssertionError(f"trigger={label} 中 {uav_id} z=1 但未移动。")
            if result.per_uav_displacement_m[uav_id] > search_config.max_move_distance_m + search_config.position_tolerance_m:
                raise AssertionError(f"trigger={label} 中 {uav_id} 超过单 slot 最大移动距离。")
            if not _inside_boundary(new_xy, flight_boundary):
                raise AssertionError(f"trigger={label} 中 {uav_id} 越界。")
            # 与 system_model 的一次性 movement cost 交叉检查。
            expected = system_model.uav_movement_cost(
                old_xy,
                new_xy,
                trigger[uav_id],
                search_config.dt_s,
                search_config.max_horizontal_speed_mps,
                search_config.movement_energy_coefficient_j_per_m,
                search_config.movement_fixed_cost,
                search_config.lambda_movement,
                tolerance_m=search_config.position_tolerance_m,
            )
            if not math.isclose(
                float(expected["movement_cost"]),
                result.per_uav_movement_cost[uav_id],
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise AssertionError(f"trigger={label} 中 {uav_id} movement cost 不一致。")

        repeated = optimize_uav_configuration(
            current_positions,
            trigger,
            origin_scenario,
            previous_cache,
            project_cfg,
            cache_nodes,
            cache_config,
            flight_boundary,
        )
        if _result_to_dict(result) != _result_to_dict(repeated):
            raise AssertionError(f"trigger={label} 重复运行不确定。")

        direct_eval = evaluate_uav_configuration(
            final_np,
            current_positions,
            trigger,
            origin_scenario,
            previous_cache,
            cache_nodes,
            cache_config,
            search_config,
        )
        if not math.isclose(
            direct_eval["objective_value"],
            result.objective_value,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise AssertionError(f"trigger={label} direct evaluator objective 不一致。")

    # 当前没有统一 safety-distance 数值，因此只检查 UAV 不完全重合。
    for label, result in results.items():
        final_np = {
            uav_id: np.asarray(result.final_positions[uav_id], dtype=np.float64)
            for uav_id in uav_ids
        }
        for index, left_id in enumerate(uav_ids):
            for right_id in uav_ids[index + 1 :]:
                if _same_position(
                    final_np[left_id],
                    final_np[right_id],
                    search_config.position_tolerance_m,
                ):
                    raise AssertionError(f"trigger={label} 中 UAV 最终位置重合。")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="给定 UAV trigger，优化 UAV 当前位置配置。")
    parser.add_argument("--config", default="system_config.toml")
    parser.add_argument("--origin-index", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = project_config.load_config(args.config)
    project_config.validate_config(cfg)
    project_config.validate_for_formal_experiment(cfg)
    project_root = Path(cfg["_meta"]["project_root"])
    cache_cfg = cache_configuration.cache_config_from_project(cfg)
    scenario_path = cache_configuration.resolve_path(
        project_root,
        project_config.get_config_value(cfg, "paths.prediction_scenarios"),
    )
    catalog_path = cache_configuration.resolve_path(
        project_root,
        project_config.get_config_value(cfg, "paths.service_catalog"),
    )
    scenarios = cache_configuration.load_prediction_scenarios(scenario_path)
    origin_scenario = cache_configuration.scenario_at_origin(scenarios, args.origin_index)
    service_catalog = cache_configuration.load_service_catalog(catalog_path)
    cache_nodes = cache_configuration.build_cache_nodes(origin_scenario, cache_cfg)
    boundary = boundary_from_project(cfg)
    results = validate_uav_optimizer_once(
        origin_scenario,
        service_catalog,
        cache_nodes,
        cache_cfg,
        cfg,
        boundary,
    )
    old_positions = initial_uav_positions_from_project(cfg)
    print("UAV configuration minimal validation PASS")
    print(f"origin_slot={origin_scenario['origin_slot']}, boundary={boundary}")
    for label, result in results.items():
        print(f"trigger={label}")
        print(f"  old_positions={json.dumps({k: old_positions[k].tolist() for k in old_positions}, ensure_ascii=False)}")
        print(f"  new_positions={json.dumps(result.final_positions, ensure_ascii=False)}")
        print(f"  displacement_m={json.dumps(result.per_uav_displacement_m, ensure_ascii=False)}")
        print(f"  movement_energy_j={result.movement_energy:.6f}")
        print(f"  movement_cost={result.movement_cost:.6f}")
        print(f"  predictive_utility={result.predictive_utility:.6f}")
        print(f"  objective={result.objective_value:.6f}")
        print(f"  trigger_feasible={result.trigger_feasible}")
        print(
            "  search_rounds="
            f"{result.search_rounds}, accepted_improvements={result.accepted_improvements}, "
            f"stop_reason={result.stop_reason}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
