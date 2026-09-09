"""Minimal environment interface used by the standalone PPO v2 path.

This module contains only the state/action contract and the artifact-alignment
helpers needed to connect PPO v2 to the existing environment. Cache/UAV
configuration, task matching, KKT allocation, and reward calculation remain
owned by the existing decision modules.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common import project_config
from decision import cache_configuration, task_allocation, uav_configuration


def _validate_binary_vector(value: Any, expected_length: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.int8).reshape(-1)
    if vector.shape != (expected_length,):
        raise ValueError(f"{name} shape={vector.shape}，期望 {(expected_length,)}。")
    if not np.all(np.isin(vector, [0, 1])):
        raise ValueError(f"{name} 必须是 0/1 向量。")
    return vector


@dataclass(frozen=True)
class TriggerActionSpec:
    """Trigger action 的唯一权威顺序。"""

    cache_node_ids: tuple[str, ...]
    uav_ids: tuple[str, ...]
    labels: tuple[str, ...]
    action_dim: int

    @classmethod
    def from_cache_nodes_and_project(
        cls,
        cache_nodes: dict[str, Any],
        project_cfg: dict[str, Any],
    ) -> "TriggerActionSpec":
        cache_node_ids = tuple(str(value) for value in cache_nodes["node_ids"])
        uav_ids = tuple(
            str(item["id"])
            for item in project_config.get_config_value(project_cfg, "nodes.uav.instances", [])
        )
        if not cache_node_ids:
            raise ValueError("TriggerActionSpec 需要至少一个 dynamic cache node。")
        if not uav_ids:
            raise ValueError("TriggerActionSpec 需要至少一个 UAV。")
        labels = tuple(
            [f"z_{node_id}^ca" for node_id in cache_node_ids]
            + [f"z_{uav_id}^q" for uav_id in uav_ids]
        )
        return cls(
            cache_node_ids=cache_node_ids,
            uav_ids=uav_ids,
            labels=labels,
            action_dim=len(labels),
        )

    def action_vector_to_triggers(self, action_vector: Any) -> tuple[np.ndarray, dict[str, int]]:
        vector = _validate_binary_vector(action_vector, self.action_dim, "action_vector")
        cache_count = len(self.cache_node_ids)
        cache_triggers = vector[:cache_count].astype(np.int8).copy()
        uav_triggers = {
            uav_id: int(vector[cache_count + index])
            for index, uav_id in enumerate(self.uav_ids)
        }
        return cache_triggers, uav_triggers

    def triggers_to_action_vector(
        self,
        cache_triggers: Any,
        uav_triggers: dict[str, int],
    ) -> np.ndarray:
        cache_vector = _validate_binary_vector(
            cache_triggers,
            len(self.cache_node_ids),
            "cache_triggers",
        )
        if set(uav_triggers) != set(self.uav_ids):
            raise ValueError(
                f"uav_triggers IDs={sorted(uav_triggers)} 与 spec={list(self.uav_ids)} 不一致。"
            )
        uav_vector = np.asarray(
            [int(uav_triggers[uav_id]) for uav_id in self.uav_ids],
            dtype=np.int8,
        )
        if not np.all(np.isin(uav_vector, [0, 1])):
            raise ValueError("uav_triggers 必须全部为 0/1。")
        return np.concatenate([cache_vector, uav_vector]).astype(np.int8)


@dataclass(frozen=True)
class ConfigurationObservationSpec:
    """PPO configuration observation 的固定结构。"""

    cache_node_ids: tuple[str, ...]
    service_ids: tuple[int, ...]
    uav_ids: tuple[str, ...]
    vehicle_ids: tuple[str, ...]
    cache_slice: slice
    uav_position_slice: slice
    predicted_vehicle_slice: slice
    popularity_slice: slice
    time_to_go_slice: slice | None
    observation_dim: int

    @classmethod
    def create(
        cls,
        cache_node_ids: list[str] | tuple[str, ...],
        service_ids: list[int] | tuple[int, ...],
        uav_ids: list[str] | tuple[str, ...],
        vehicle_ids: list[str] | tuple[str, ...],
        *,
        include_time_to_go: bool = False,
    ) -> "ConfigurationObservationSpec":
        cache_node_ids_t = tuple(str(value) for value in cache_node_ids)
        service_ids_t = tuple(int(value) for value in service_ids)
        uav_ids_t = tuple(str(value) for value in uav_ids)
        vehicle_ids_t = tuple(str(value) for value in vehicle_ids)
        if len(set(cache_node_ids_t)) != len(cache_node_ids_t) or not cache_node_ids_t:
            raise ValueError("cache_node_ids 不能为空且不能重复。")
        if service_ids_t != tuple(range(len(service_ids_t))):
            raise ValueError("service_ids 必须从 0 连续排列。")
        if len(set(uav_ids_t)) != len(uav_ids_t) or not uav_ids_t:
            raise ValueError("uav_ids 不能为空且不能重复。")
        if len(set(vehicle_ids_t)) != len(vehicle_ids_t) or not vehicle_ids_t:
            raise ValueError("vehicle_ids 不能为空且不能重复。")

        start = 0
        cache_len = len(cache_node_ids_t) * len(service_ids_t)
        cache_slice = slice(start, start + cache_len)
        start += cache_len
        uav_len = len(uav_ids_t) * 2
        uav_position_slice = slice(start, start + uav_len)
        start += uav_len
        vehicle_len = len(vehicle_ids_t) * 2
        predicted_vehicle_slice = slice(start, start + vehicle_len)
        start += vehicle_len
        popularity_len = len(service_ids_t)
        popularity_slice = slice(start, start + popularity_len)
        start += popularity_len
        time_to_go_slice = None
        if include_time_to_go:
            time_to_go_slice = slice(start, start + 1)
            start += 1
        return cls(
            cache_node_ids=cache_node_ids_t,
            service_ids=service_ids_t,
            uav_ids=uav_ids_t,
            vehicle_ids=vehicle_ids_t,
            cache_slice=cache_slice,
            uav_position_slice=uav_position_slice,
            predicted_vehicle_slice=predicted_vehicle_slice,
            popularity_slice=popularity_slice,
            time_to_go_slice=time_to_go_slice,
            observation_dim=start,
        )

    @classmethod
    def from_cache_nodes(
        cls,
        cache_nodes: dict[str, Any],
        service_ids: list[int] | tuple[int, ...],
        uav_ids: list[str] | tuple[str, ...],
        vehicle_ids: list[str] | tuple[str, ...],
        *,
        include_time_to_go: bool = False,
    ) -> "ConfigurationObservationSpec":
        return cls.create(
            cache_nodes["node_ids"],
            service_ids,
            uav_ids,
            vehicle_ids,
            include_time_to_go=include_time_to_go,
        )


def _normalize_xy(
    xy: Any,
    boundary: tuple[float, float, float, float],
    label: str,
) -> np.ndarray:
    x_min, y_min, x_max, y_max = [float(value) for value in boundary]
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("flight_boundary 非法。")
    array = np.asarray(xy, dtype=np.float64)
    if array.shape != (2,):
        raise ValueError(f"{label} 坐标 shape={array.shape}，期望 (2,)。")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} 坐标包含 NaN/Inf。")
    normalized = np.asarray(
        [
            (float(array[0]) - x_min) / (x_max - x_min),
            (float(array[1]) - y_min) / (y_max - y_min),
        ],
        dtype=np.float64,
    )
    if np.any(normalized < -1e-6) or np.any(normalized > 1.0 + 1e-6):
        raise ValueError(f"{label} 归一化坐标越界：{normalized.tolist()}。")
    return np.clip(normalized, 0.0, 1.0)


def _ordered_position_array(
    positions: dict[str, Any] | Any,
    ids: tuple[str, ...],
    name: str,
) -> np.ndarray:
    if isinstance(positions, dict):
        if set(str(key) for key in positions) != set(ids):
            raise ValueError(f"{name} IDs 与 observation spec 不一致。")
        return np.asarray([positions[vehicle_id] for vehicle_id in ids], dtype=np.float64)
    array = np.asarray(positions, dtype=np.float64)
    if array.shape != (len(ids), 2):
        raise ValueError(f"{name} shape={array.shape}，期望 {(len(ids), 2)}。")
    return array


def _ordered_popularity_array(
    popularity: dict[int | str, float] | Any,
    service_ids: tuple[int, ...],
) -> np.ndarray:
    if isinstance(popularity, dict):
        normalized_dict = {int(key): float(value) for key, value in popularity.items()}
        if set(normalized_dict) != set(service_ids):
            raise ValueError("predicted_service_popularity service IDs 与 spec 不一致。")
        values = np.asarray(
            [normalized_dict[service_id] for service_id in service_ids],
            dtype=np.float64,
        )
    else:
        values = np.asarray(popularity, dtype=np.float64).reshape(-1)
        if values.shape != (len(service_ids),):
            raise ValueError(
                f"predicted_service_popularity shape={values.shape}，期望 {(len(service_ids),)}。"
            )
    if not np.all(np.isfinite(values)):
        raise ValueError("predicted_service_popularity 包含 NaN/Inf。")
    if np.any(values < -1e-8):
        raise ValueError("predicted_service_popularity 不能为负。")
    if not math.isclose(float(values.sum()), 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError("predicted_service_popularity 必须已经归一化为概率分布。")
    return values.astype(np.float64)


def normalized_time_to_go(
    origin_index: int,
    train_origin_start: int,
    train_origin_end: int,
) -> float:
    """Return the finite-horizon state feature (end - origin) / (end - start)."""

    start = int(train_origin_start)
    end = int(train_origin_end)
    origin = int(origin_index)
    if end <= start:
        raise ValueError("time_to_go requires train_origin_end > train_origin_start")
    if origin < start or origin > end:
        raise ValueError("time_to_go origin is outside the formal train range")
    value = (end - origin) / (end - start)
    if not 0.0 <= value <= 1.0:
        raise AssertionError("time_to_go must be within [0, 1]")
    return float(value)


def build_configuration_observation(
    previous_cache_state: Any,
    previous_uav_positions: dict[str, Any] | Any,
    predicted_vehicle_positions: dict[str, Any] | Any,
    predicted_service_popularity: dict[int | str, float] | Any,
    observation_spec: ConfigurationObservationSpec,
    flight_boundary: tuple[float, float, float, float],
    *,
    time_to_go: float | None = None,
) -> np.ndarray:
    """构造只包含 state/prediction 字段的 configuration observation。"""

    m = len(observation_spec.cache_node_ids)
    f = len(observation_spec.service_ids)
    previous_cache = np.asarray(previous_cache_state, dtype=np.int8)
    if previous_cache.shape != (m, f):
        raise ValueError(f"previous_cache_state shape={previous_cache.shape}，期望 {(m, f)}。")
    if not np.all(np.isin(previous_cache, [0, 1])):
        raise ValueError("previous_cache_state 必须只包含 0/1。")

    uav_positions = _ordered_position_array(
        previous_uav_positions,
        observation_spec.uav_ids,
        "previous_uav_positions",
    )
    vehicle_positions = _ordered_position_array(
        predicted_vehicle_positions,
        observation_spec.vehicle_ids,
        "predicted_vehicle_positions",
    )
    popularity = _ordered_popularity_array(
        predicted_service_popularity,
        observation_spec.service_ids,
    )
    cache_features = previous_cache.reshape(-1).astype(np.float64)
    uav_features = np.concatenate(
        [
            _normalize_xy(uav_positions[index], flight_boundary, f"UAV {uav_id}")
            for index, uav_id in enumerate(observation_spec.uav_ids)
        ]
    )
    vehicle_features = np.concatenate(
        [
            _normalize_xy(vehicle_positions[index], flight_boundary, f"vehicle {vehicle_id}")
            for index, vehicle_id in enumerate(observation_spec.vehicle_ids)
        ]
    )
    if observation_spec.time_to_go_slice is None:
        if time_to_go is not None:
            raise ValueError("v1 observation must not receive time_to_go")
        time_features = np.empty(0, dtype=np.float64)
    else:
        if time_to_go is None:
            raise ValueError("v2 observation requires time_to_go")
        time_value = float(time_to_go)
        if not np.isfinite(time_value) or not 0.0 <= time_value <= 1.0:
            raise ValueError("time_to_go must be finite and within [0, 1]")
        time_features = np.asarray([time_value], dtype=np.float64)

    observation = np.concatenate(
        [cache_features, uav_features, vehicle_features, popularity, time_features]
    )
    if observation.shape != (observation_spec.observation_dim,):
        raise AssertionError("observation length 与 spec 不一致。")
    if not np.all(np.isfinite(observation)):
        raise ValueError("observation 包含 NaN/Inf。")
    return observation.astype(np.float32)


def build_trigger_action_mask_provider(
    previous_cache_state: np.ndarray,
    previous_uav_positions: dict[str, Any],
    service_catalog: dict[str, Any],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    flight_boundary: tuple[float, float, float, float],
    action_spec: TriggerActionSpec,
) -> Callable[[int, np.ndarray], np.ndarray]:
    """构造当前 state 下的 binary request mask。"""

    previous_cache = np.asarray(previous_cache_state, dtype=np.int8)
    cache_node_ids = [str(value) for value in cache_nodes["node_ids"]]
    if tuple(cache_node_ids) != tuple(action_spec.cache_node_ids):
        raise ValueError("action_spec.cache_node_ids 与 cache_nodes 不一致。")
    if previous_cache.shape != (
        len(cache_node_ids),
        int(service_catalog["num_services"]),
    ):
        raise ValueError("previous_cache_state shape 与 cache_nodes/service catalog 不一致。")
    if not np.all(np.isin(previous_cache, [0, 1])):
        raise ValueError("previous_cache_state 必须是 0/1。")

    previous_uav = {
        str(uav_id): np.asarray(xy, dtype=np.float64).copy()
        for uav_id, xy in previous_uav_positions.items()
    }
    cache_count = len(action_spec.cache_node_ids)

    def provider(action_index: int, prefix_actions: np.ndarray) -> np.ndarray:
        index = int(action_index)
        prefix = np.asarray(prefix_actions, dtype=np.int8).reshape(-1)
        if index < 0 or index >= action_spec.action_dim or len(prefix) != index:
            raise ValueError("action mask provider 收到非法 action index/prefix。")
        if not np.all(np.isin(prefix, [0, 1])):
            raise ValueError("action mask prefix 必须是 0/1。")
        if index < cache_count:
            return np.asarray([True, True], dtype=bool)

        uav_index = index - cache_count
        triggers = {uav_id: 0 for uav_id in action_spec.uav_ids}
        for prior_index in range(uav_index):
            triggers[action_spec.uav_ids[prior_index]] = int(
                prefix[cache_count + prior_index]
            )
        current_uav_id = action_spec.uav_ids[uav_index]
        triggers[current_uav_id] = 1
        can_trigger = uav_configuration.has_feasible_trigger_assignment(
            previous_uav,
            triggers,
            project_cfg,
            flight_boundary,
        )
        return np.asarray([True, bool(can_trigger)], dtype=bool)

    return provider


def _copy_uav_positions(positions: dict[str, Any]) -> dict[str, tuple[float, float]]:
    return {
        str(uav_id): (float(np.asarray(xy)[0]), float(np.asarray(xy)[1]))
        for uav_id, xy in positions.items()
    }


@dataclass(frozen=True)
class AlignedRolloutStep:
    """已经按同一个真实 slot/time 对齐的一步输入。"""

    origin_index: int
    slot: int
    time_s: float
    scenario_slice: dict[str, Any]
    predicted_vehicle_positions: dict[str, np.ndarray]
    predicted_service_popularity: dict[int, float]
    actual_tasks: list[task_allocation.Task]
    actual_vehicle_positions: dict[str, np.ndarray]


def _load_vehicle_prediction_h0(
    path: Path,
) -> dict[int, dict[str, tuple[float, float, int, float]]]:
    by_origin: dict[int, dict[str, tuple[float, float, int, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if int(row["horizon"]) != 0:
                continue
            origin = int(row["origin_slot"])
            by_origin.setdefault(origin, {})[str(row["vehicle_id"])] = (
                float(row["x_pred"]),
                float(row["y_pred"]),
                int(row["target_slot"]),
                float(row["target_time_s"]),
            )
    return by_origin


def _load_popularity_prediction_h0(
    path: Path,
) -> dict[int, dict[int, tuple[float, int, float]]]:
    by_origin: dict[int, dict[int, tuple[float, int, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if int(row["horizon"]) != 0:
                continue
            origin = int(row["origin_slot"])
            service_id = int(row["service_id"])
            by_origin.setdefault(origin, {})[service_id] = (
                float(row["rho_pred"]),
                int(row["target_slot"]),
                float(row["target_time_s"]),
            )
    return by_origin


def _load_task_time_by_slot(path: Path) -> dict[int, float]:
    result: dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            slot = int(row["slot"])
            time_s = float(row["time_s"])
            if slot in result and not math.isclose(
                result[slot], time_s, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(f"tasks.csv slot={slot} time_s 不一致。")
            result[slot] = time_s
    return result


def prepare_aligned_rollout_steps(
    project_root: Path,
    scenarios: dict[str, Any],
    origin_start_index: int,
    max_steps: int,
    vehicle_prediction_file: Path,
    popularity_prediction_file: Path,
    tasks_file: Path,
    trajectory_file: Path,
    include_bootstrap_step: bool = True,
) -> list[AlignedRolloutStep]:
    """从正式 artifact 构造已对齐 steps；不重新预测或生成场景。"""

    del project_root  # retained for the stable caller-facing interface
    if max_steps <= 0:
        raise ValueError("max_steps 必须大于 0。")
    vehicle_pred = _load_vehicle_prediction_h0(vehicle_prediction_file)
    popularity_pred = _load_popularity_prediction_h0(popularity_prediction_file)
    task_time_by_slot = _load_task_time_by_slot(tasks_file)
    n_origin = int(scenarios["origin_slots"].shape[0])
    if origin_start_index < 0 or origin_start_index >= n_origin:
        raise IndexError("origin_start_index 越界。")

    extra_step = 1 if include_bootstrap_step else 0
    stop = min(n_origin, origin_start_index + max_steps + extra_step)
    steps: list[AlignedRolloutStep] = []
    for origin_index in range(origin_start_index, stop):
        scenario = cache_configuration.scenario_at_origin(scenarios, origin_index)
        slot = int(scenario["origin_slot"])
        time_s = float(scenario["origin_time_s"])
        if not math.isclose(
            float(task_time_by_slot.get(slot, float("nan"))),
            time_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"origin_index={origin_index} 的 scenario slot/time 与 tasks.csv 不一致。")
        if int(scenario["horizons"][0]) != 0:
            raise ValueError("scenario horizons[0] 必须为0，表示当前调度slot。")

        vehicle_ids = [str(value) for value in scenario["vehicle_ids"].tolist()]
        raw_vehicle = vehicle_pred.get(slot)
        if raw_vehicle is None or set(raw_vehicle) != set(vehicle_ids):
            raise ValueError(f"vehicle_predictions.csv 缺少 origin slot={slot} 的完整车辆预测。")
        predicted_positions: dict[str, np.ndarray] = {}
        for vehicle_id in vehicle_ids:
            x_pred, y_pred, target_slot, target_time = raw_vehicle[vehicle_id]
            if target_slot != slot or not math.isclose(
                target_time,
                time_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(f"vehicle prediction target 与 slot/time 不一致：slot={slot}")
            predicted_positions[vehicle_id] = np.asarray([x_pred, y_pred], dtype=np.float64)

        service_ids = list(range(int(scenario["rho_hat"].shape[1])))
        raw_popularity = popularity_pred.get(slot)
        if raw_popularity is None or set(raw_popularity) != set(service_ids):
            raise ValueError(f"popularity_predictions.csv 缺少 origin slot={slot} 的完整服务预测。")
        predicted_popularity: dict[int, float] = {}
        for service_id in service_ids:
            rho, target_slot, target_time = raw_popularity[service_id]
            if target_slot != slot or not math.isclose(
                target_time,
                time_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(f"popularity prediction target 与 slot/time 不一致：slot={slot}")
            predicted_popularity[service_id] = rho
        if not math.isclose(
            sum(predicted_popularity.values()),
            1.0,
            rel_tol=1e-5,
            abs_tol=1e-5,
        ):
            raise ValueError(f"slot={slot} popularity rho_pred 未归一化。")

        actual_tasks = task_allocation.load_actual_tasks(tasks_file, slot)
        sv_ids = {str(value) for value in scenario["service_vehicle_ids"].tolist()}
        tv_ids = {task.vehicle_id for task in actual_tasks}
        actual_positions = task_allocation.load_vehicle_positions(
            trajectory_file,
            time_s,
            sv_ids | tv_ids,
        )
        steps.append(
            AlignedRolloutStep(
                origin_index=origin_index,
                slot=slot,
                time_s=time_s,
                scenario_slice=scenario,
                predicted_vehicle_positions=predicted_positions,
                predicted_service_popularity=predicted_popularity,
                actual_tasks=actual_tasks,
                actual_vehicle_positions=actual_positions,
            )
        )
    for left, right in zip(steps, steps[1:]):
        if right.origin_index <= left.origin_index or right.slot <= left.slot or right.time_s <= left.time_s:
            raise ValueError("aligned steps 的 origin/slot/time 必须严格递增。")
    return steps
