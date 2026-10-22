"""互斥缓存与 UAV 候选的稳定表示及特征编码。"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from common import project_config
from decision import cache_configuration, uav_configuration
from online_joint_optimizer.candidate_evaluator import SlotInput, copy_uav_positions


OPERATIONS = ("KEEP_STAY", "CACHE_UPDATE", "UAV_MOVE")


def _state_fingerprint(
    cache: np.ndarray,
    positions: Mapping[str, Any],
    slot_input: SlotInput | None = None,
) -> str:
    hasher = hashlib.sha256()
    array = np.ascontiguousarray(cache, dtype=np.int8)
    hasher.update(str(array.shape).encode("ascii"))
    hasher.update(array.tobytes())
    for uav_id, xy in positions.items():
        hasher.update(str(uav_id).encode("utf-8"))
        hasher.update(np.ascontiguousarray(xy, dtype=np.float64).tobytes())
    if slot_input is not None:
        hasher.update(
            pickle.dumps(
                (
                    slot_input.slot_id,
                    slot_input.time_s,
                    slot_input.scenario_id,
                    slot_input.scenario_slice,
                    slot_input.tasks,
                    slot_input.vehicle_positions,
                    slot_input.predicted_vehicle_positions,
                    slot_input.prediction_horizon_mask,
                ),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        )
    return hasher.hexdigest()


@dataclass(frozen=True)
class ActionCandidate:
    """保存动作语义和完整目标配置，供回放与复现使用。"""

    operation: str
    candidate_id: str
    source_state_fingerprint: str
    target_cache_state: tuple[tuple[int, ...], ...]
    target_uav_positions: tuple[tuple[str, float, float], ...]
    changed_node_ids: tuple[str, ...]
    moved_uav_ids: tuple[str, ...]
    proposal_source: str

    @classmethod
    def from_states(
        cls,
        *,
        previous_cache_state: Any,
        previous_uav_positions: Mapping[str, Any],
        target_cache_state: Any,
        target_uav_positions: Mapping[str, Any],
        slot_input: SlotInput,
        proposal_source: str,
    ) -> "ActionCandidate":
        """根据正式候选生成器的输出建立并验证一个动作。"""

        before_cache = np.asarray(previous_cache_state, dtype=np.int8).copy()
        after_cache = np.asarray(target_cache_state, dtype=np.int8).copy()
        before_uav = copy_uav_positions(previous_uav_positions)
        after_uav = copy_uav_positions(target_uav_positions)
        if before_cache.shape != after_cache.shape:
            raise ValueError("候选缓存矩阵必须与当前缓存形状一致")
        if tuple(before_uav) != tuple(after_uav):
            raise ValueError("候选 UAV 顺序必须与当前状态一致")
        node_ids = tuple(str(value) for value in slot_input.model.cache_nodes["node_ids"])
        if len(node_ids) != before_cache.shape[0]:
            raise ValueError("候选缓存行与正式缓存节点顺序不匹配")
        tolerance = float(
            project_config.get_config_value(
                slot_input.model.project_cfg,
                "algorithm.uav.position_tolerance_m",
                1.0e-6,
            )
        )
        changed_node_ids = tuple(
            node_ids[index]
            for index in range(len(node_ids))
            if not np.array_equal(before_cache[index], after_cache[index])
        )
        moved_uav_ids = tuple(
            uav_id
            for uav_id in before_uav
            if float(np.linalg.norm(after_uav[uav_id] - before_uav[uav_id])) > tolerance
        )
        cache_changed = bool(changed_node_ids)
        uav_moved = bool(moved_uav_ids)
        if cache_changed and uav_moved:
            raise ValueError("一个当前动作不能同时更新缓存和移动 UAV")
        if not cache_changed and not uav_moved:
            operation = "KEEP_STAY"
        elif cache_changed:
            operation = "CACHE_UPDATE"
        else:
            operation = "UAV_MOVE"
        for index, node_id in enumerate(node_ids):
            if not np.array_equal(before_cache[index], after_cache[index]) and not cache_configuration.is_legal_cache_replacement(
                before_cache[index], after_cache[index]
            ):
                raise ValueError(f"缓存候选 {node_id} 不符合正式 REPLACE 规则")
        triggers = {uav_id: int(uav_id in moved_uav_ids) for uav_id in before_uav}
        uav_configuration.validate_uav_configuration_transition(
            before_uav,
            after_uav,
            triggers,
            dict(slot_input.model.project_cfg),
            tuple(float(value) for value in slot_input.model.flight_boundary),
        )
        state_hash = _state_fingerprint(before_cache, before_uav, slot_input)
        target_cache_tuple = tuple(
            tuple(int(value) for value in row) for row in after_cache.tolist()
        )
        target_uav_tuple = tuple(
            (uav_id, float(after_uav[uav_id][0]), float(after_uav[uav_id][1]))
            for uav_id in after_uav
        )
        identity = json.dumps(
            {
                "operation": operation,
                "source_state": state_hash,
                "cache": target_cache_tuple,
                "uav": target_uav_tuple,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        candidate_id = hashlib.sha256(identity).hexdigest()
        return cls(
            operation=operation,
            candidate_id=candidate_id,
            source_state_fingerprint=state_hash,
            target_cache_state=target_cache_tuple,
            target_uav_positions=target_uav_tuple,
            changed_node_ids=changed_node_ids,
            moved_uav_ids=moved_uav_ids,
            proposal_source=str(proposal_source),
        )

    def cache_array(self) -> np.ndarray:
        return np.asarray(self.target_cache_state, dtype=np.int8).copy()

    def uav_dict(self) -> dict[str, np.ndarray]:
        return {
            uav_id: np.asarray((x, y), dtype=np.float64)
            for uav_id, x, y in self.target_uav_positions
        }

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 可序列化的动作语义。"""

        return {
            "operation": self.operation,
            "candidate_id": self.candidate_id,
            "source_state_fingerprint": self.source_state_fingerprint,
            "target_cache_state": [list(row) for row in self.target_cache_state],
            "target_uav_positions": [list(row) for row in self.target_uav_positions],
            "changed_node_ids": list(self.changed_node_ids),
            "moved_uav_ids": list(self.moved_uav_ids),
            "proposal_source": self.proposal_source,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionCandidate":
        """从序列化动作恢复稳定表示。"""

        return cls(
            operation=str(value["operation"]),
            candidate_id=str(value["candidate_id"]),
            source_state_fingerprint=str(value["source_state_fingerprint"]),
            target_cache_state=tuple(
                tuple(int(item) for item in row) for row in value["target_cache_state"]
            ),
            target_uav_positions=tuple(
                (str(row[0]), float(row[1]), float(row[2]))
                for row in value["target_uav_positions"]
            ),
            changed_node_ids=tuple(str(item) for item in value["changed_node_ids"]),
            moved_uav_ids=tuple(str(item) for item in value["moved_uav_ids"]),
            proposal_source=str(value.get("proposal_source", "replay")),
        )


@dataclass(frozen=True)
class CandidateOption:
    """一个合法候选及其在所属状态下计算出的特征。"""

    action: ActionCandidate
    features: tuple[float, ...]

    @classmethod
    def create(cls, action: ActionCandidate, features: Any) -> "CandidateOption":
        vector = np.asarray(features, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(vector)):
            raise ValueError("候选动作特征必须为有限数值")
        return cls(action, tuple(float(value) for value in vector))

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action.to_dict(), "features": list(self.features)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateOption":
        return cls.create(ActionCandidate.from_dict(value["action"]), value["features"])


class CandidateFeatureEncoder:
    """将正式 Slot 状态和合法目标配置编码为固定维度向量。"""

    def __init__(self, slot_input: SlotInput) -> None:
        self.node_ids = tuple(str(value) for value in slot_input.model.cache_nodes["node_ids"])
        self.service_ids = tuple(
            int(value) for value in np.asarray(slot_input.model.service_catalog["service_ids"]).tolist()
        )
        self.service_index = {service_id: index for index, service_id in enumerate(self.service_ids)}
        self.uav_ids = tuple(str(value) for value in slot_input.vehicle_positions if str(value).startswith("UAV_"))
        configured_uavs = tuple(uav_configuration.uav_ids_from_project(dict(slot_input.model.project_cfg)))
        if self.uav_ids != configured_uavs:
            self.uav_ids = configured_uavs
        self.vehicle_ids = tuple(str(value) for value in slot_input.predicted_vehicle_positions)
        self.bounds = tuple(float(value) for value in slot_input.model.flight_boundary)
        scenario = slot_input.scenario_slice
        rho = np.asarray(scenario.get("rho_hat", np.zeros((1, len(self.service_ids)))), dtype=np.float64)
        self.horizon_count = int(rho.shape[0]) if rho.ndim == 2 else int(len(scenario.get("horizons", ())))
        if self.horizon_count <= 0:
            self.horizon_count = 1
        self.task_vehicle_capacity = max(1, len(tuple(scenario.get("task_vehicle_ids", ()))))
        service_cfg = slot_input.model.project_cfg.get("service", {}).get("generation", {})
        task_cfg = slot_input.model.project_cfg.get("task", {})
        self.max_input_bits = max(
            1.0,
            float(service_cfg.get("input_size_max_kb", 1000.0)) * 8000.0,
        )
        self.max_cpu_cycles = max(
            1.0,
            self.max_input_bits
            * float(service_cfg.get("cycles_per_bit_max", 1000.0))
            * (1.0 + float(task_cfg.get("load_relative_jitter", 0.0))),
        )
        self.episode_time_s = max(
            1.0,
            float(project_config.get_config_value(slot_input.model.project_cfg, "simulation.episode_time_s", 1.0)),
        )
        self.service_sizes_mb = np.asarray(
            slot_input.model.service_catalog["service_sizes_mb"], dtype=np.float64
        ).reshape(-1)
        self.total_service_size_mb = max(1.0, float(np.sum(self.service_sizes_mb)))
        self.map_diagonal_m = max(
            1.0,
            math.hypot(self.bounds[1] - self.bounds[0], self.bounds[3] - self.bounds[2]),
        )
        search = uav_configuration.uav_search_config_from_project(dict(slot_input.model.project_cfg))
        self.max_move_distance_m = max(
            1.0, float(search.max_horizontal_speed_mps) * float(search.dt_s)
        )
        node_count, service_count, uav_count = len(self.node_ids), len(self.service_ids), len(self.uav_ids)
        self.state_dim = (
            node_count * service_count
            + 2 * uav_count
            + 2
            + service_count
            + 2
            + service_count
            + 2 * len(self.vehicle_ids)
            + self.horizon_count
        )
        self.candidate_dim = (
            3 + 2 + node_count * service_count + node_count
            + 2 * node_count * service_count + 4 + 8 * uav_count
        )

    def encode_state(
        self,
        slot_input: SlotInput,
        cache_state: Any,
        uav_positions: Mapping[str, Any],
    ) -> np.ndarray:
        cache = np.asarray(cache_state, dtype=np.float32)
        if cache.shape != (len(self.node_ids), len(self.service_ids)):
            raise ValueError("缓存状态形状与特征编码器不一致")
        positions = copy_uav_positions(uav_positions)
        if tuple(positions) != self.uav_ids:
            raise ValueError("UAV 状态顺序与特征编码器不一致")
        features: list[float] = cache.reshape(-1).tolist()
        features.extend(self._positions_xy(positions))
        features.append(float(np.clip(float(slot_input.time_s or 0.0) / self.episode_time_s, 0.0, 1.0)))
        tasks = tuple(slot_input.tasks)
        features.append(float(len(tasks) / self.task_vehicle_capacity))
        histogram = np.zeros(len(self.service_ids), dtype=np.float32)
        input_values: list[float] = []
        cpu_values: list[float] = []
        for task in tasks:
            try:
                histogram[self.service_index[int(task.service_id)]] += 1.0
            except KeyError as exc:
                raise ValueError(f"任务服务 ID 不在正式服务目录中：{task.service_id}") from exc
            input_values.append(float(task.input_bits) / self.max_input_bits)
            cpu_values.append(float(task.cpu_cycles) / self.max_cpu_cycles)
        if len(tasks):
            histogram /= float(len(tasks))
        features.extend(histogram.tolist())
        features.extend(
            [
                float(np.mean(input_values)) if input_values else 0.0,
                float(np.mean(cpu_values)) if cpu_values else 0.0,
            ]
        )
        rho = np.asarray(slot_input.scenario_slice.get("rho_hat", ()), dtype=np.float32)
        if rho.ndim == 2 and rho.shape[0] and rho.shape[1] == len(self.service_ids):
            popularity = np.clip(rho[0], 0.0, None)
            denominator = float(popularity.sum())
            if denominator > 0.0:
                popularity = popularity / denominator
            features.extend(popularity.tolist())
        else:
            features.extend([0.0] * len(self.service_ids))
        predicted = slot_input.predicted_vehicle_positions
        for vehicle_id in self.vehicle_ids:
            if vehicle_id not in predicted:
                raise ValueError(f"预测车辆位置缺少配置车辆：{vehicle_id}")
            features.extend(self._normalize_xy(predicted[vehicle_id]))
        mask = np.asarray(slot_input.prediction_horizon_mask, dtype=np.float32).reshape(-1)
        if mask.size < self.horizon_count:
            mask = np.pad(mask, (0, self.horizon_count - mask.size))
        features.extend(mask[: self.horizon_count].tolist())
        result = np.asarray(features, dtype=np.float32)
        if result.shape != (self.state_dim,) or not np.all(np.isfinite(result)):
            raise ValueError("状态特征维度或数值无效")
        return result

    def encode_candidate(
        self,
        action: ActionCandidate,
        *,
        previous_cache_state: Any,
        previous_uav_positions: Mapping[str, Any],
        slot_input: SlotInput,
    ) -> np.ndarray:
        before_cache = np.asarray(previous_cache_state, dtype=np.int8)
        after_cache = action.cache_array()
        before_uav = copy_uav_positions(previous_uav_positions)
        after_uav = action.uav_dict()
        if action.operation not in OPERATIONS:
            raise ValueError("候选操作类型非法")
        if action.operation == "KEEP_STAY" and (action.changed_node_ids or action.moved_uav_ids):
            raise ValueError("KEEP_STAY 必须保持全部配置不变")
        if action.operation == "CACHE_UPDATE" and (not action.changed_node_ids or action.moved_uav_ids):
            raise ValueError("CACHE_UPDATE 必须只改变缓存")
        if action.operation == "UAV_MOVE" and (action.changed_node_ids or not action.moved_uav_ids):
            raise ValueError("UAV_MOVE 必须只改变 UAV 位置")
        if bool(action.changed_node_ids) and bool(action.moved_uav_ids):
            raise ValueError("候选动作违反缓存与 UAV 互斥约束")
        node_count, service_count = before_cache.shape
        features = [float(action.operation == op) for op in OPERATIONS]
        features.extend([float(bool(action.changed_node_ids)), float(bool(action.moved_uav_ids))])
        features.extend(after_cache.astype(np.float32).reshape(-1).tolist())
        changed_node_mask = [float(node_id in action.changed_node_ids) for node_id in self.node_ids]
        features.extend(changed_node_mask)
        removed = np.maximum(before_cache - after_cache, 0).astype(np.float32)
        added = np.maximum(after_cache - before_cache, 0).astype(np.float32)
        features.extend(removed.reshape(-1).tolist())
        features.extend(added.reshape(-1).tolist())
        added_mb = float(added.sum(axis=0) @ self.service_sizes_mb)
        trigger = np.asarray(changed_node_mask, dtype=np.int8)
        cache_details = cache_configuration.cache_update_cost_details(
            before_cache,
            after_cache,
            trigger,
            self.service_sizes_mb,
            dict(slot_input.model.cache_nodes),
            slot_input.model.cache_config,
        )
        features.extend(
            [
                added_mb / self.total_service_size_mb,
                float(cache_details["cache_cost"]),
                float(cache_details["cache_download_delay_normalized"]),
            ]
        )
        for positions in (before_uav, after_uav):
            for uav_id in self.uav_ids:
                features.extend(self._normalize_xy(positions[uav_id]))
        moved_mask = [float(uav_id in action.moved_uav_ids) for uav_id in self.uav_ids]
        distances = [
            float(np.linalg.norm(after_uav[uav_id] - before_uav[uav_id]))
            for uav_id in self.uav_ids
        ]
        features.extend(moved_mask)
        features.extend([min(distance / self.max_move_distance_m, 1.0) for distance in distances])
        search = uav_configuration.uav_search_config_from_project(dict(slot_input.model.project_cfg))
        uav_triggers = {uav_id: int(uav_id in action.moved_uav_ids) for uav_id in self.uav_ids}
        movement = uav_configuration.movement_summary(before_uav, after_uav, uav_triggers, search)
        features.append(float(movement[1]))
        demand_xy = [
            np.asarray(slot_input.vehicle_positions[task.vehicle_id], dtype=np.float64)
            for task in slot_input.tasks
            if task.vehicle_id in slot_input.vehicle_positions
        ]
        for positions in (before_uav, after_uav):
            for uav_id in self.uav_ids:
                if demand_xy:
                    mean_distance = float(
                        np.mean([np.linalg.norm(positions[uav_id] - vehicle_xy) for vehicle_xy in demand_xy])
                    )
                    features.append(min(mean_distance / self.map_diagonal_m, 1.0))
                else:
                    features.append(0.0)
        result = np.asarray(features, dtype=np.float32)
        if result.shape != (self.candidate_dim,) or not np.all(np.isfinite(result)):
            raise ValueError("候选动作特征维度或数值无效")
        return result

    def _normalize_xy(self, position: Any) -> list[float]:
        x, y = np.asarray(position, dtype=np.float64).reshape(2)
        x0, x1, y0, y1 = self.bounds
        return [
            float(np.clip((x - x0) / max(x1 - x0, 1.0), 0.0, 1.0)),
            float(np.clip((y - y0) / max(y1 - y0, 1.0), 0.0, 1.0)),
        ]

    def _positions_xy(self, positions: Mapping[str, Any]) -> list[float]:
        return [value for uav_id in self.uav_ids for value in self._normalize_xy(positions[uav_id])]


__all__ = ["ActionCandidate", "CandidateOption", "CandidateFeatureEncoder", "OPERATIONS"]
