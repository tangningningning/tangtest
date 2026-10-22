"""互斥缓存与 UAV 候选的稳定表示及特征编码。"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from common import project_config
from decision import cache_configuration, uav_configuration
from online_joint_optimizer.candidate_evaluator import SlotInput, copy_uav_positions


OPERATIONS = ("KEEP_STAY", "CACHE_UPDATE", "UAV_MOVE")
STATE_FEATURE_CONTRACT_VERSION = "formal_observation_horizon_workload_candidate_demand_v3"
CANDIDATE_FEATURE_CONTRACT_VERSION = "prediction_only_exclusive_candidate_v2"
NETWORK_ARCHITECTURE_VERSION = "legal_candidate_mean_centered_dueling_q_v3"


@dataclass(frozen=True)
class DecisionForecast:
    """只包含正式决策观测可见的预测字段，不携带真实任务或真实轨迹。"""

    origin_index: int
    slot: int
    predicted_vehicle_positions: Mapping[str, Any]
    predicted_service_popularity: Mapping[int, float]
    predicted_vehicle_positions_horizon: tuple[Mapping[str, Any], ...]
    predicted_service_popularity_horizon: tuple[Mapping[int, float], ...]
    prediction_horizon_mask: Any | None

    @classmethod
    def from_aligned_step(cls, step: Any) -> "DecisionForecast":
        """从正式对齐步骤中只复制预测接口字段。"""

        vehicle_horizon = tuple(
            {
                str(vehicle_id): np.asarray(position, dtype=np.float64).reshape(2).copy()
                for vehicle_id, position in positions.items()
            }
            for positions in step.predicted_vehicle_positions_horizon
        )
        popularity_horizon = tuple(
            {int(service_id): float(value) for service_id, value in popularity.items()}
            for popularity in step.predicted_service_popularity_horizon
        )
        mask = (
            None
            if step.prediction_horizon_mask is None
            else np.asarray(step.prediction_horizon_mask, dtype=np.int8).reshape(-1).copy()
        )
        return cls(
            origin_index=int(step.origin_index),
            slot=int(step.slot),
            predicted_vehicle_positions={
                str(vehicle_id): np.asarray(position, dtype=np.float64).reshape(2).copy()
                for vehicle_id, position in step.predicted_vehicle_positions.items()
            },
            predicted_service_popularity={
                int(service_id): float(value)
                for service_id, value in step.predicted_service_popularity.items()
            },
            predicted_vehicle_positions_horizon=vehicle_horizon,
            predicted_service_popularity_horizon=popularity_horizon,
            prediction_horizon_mask=mask,
        )


def _state_fingerprint(
    cache: np.ndarray,
    positions: Mapping[str, Any],
    decision_features: Any | None = None,
    prediction_context: Any | None = None,
) -> str:
    hasher = hashlib.sha256()
    array = np.ascontiguousarray(cache, dtype=np.int8)
    hasher.update(str(array.shape).encode("ascii"))
    hasher.update(array.tobytes())
    for uav_id, xy in positions.items():
        hasher.update(str(uav_id).encode("utf-8"))
        hasher.update(np.ascontiguousarray(xy, dtype=np.float64).tobytes())
    if decision_features is not None:
        features = np.ascontiguousarray(decision_features, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(features)):
            raise ValueError("状态指纹只能使用有限的决策特征")
        hasher.update(
            b"decision_features_v2"
            + np.asarray([features.size], dtype=np.int64).tobytes()
            + features.tobytes()
        )
    if prediction_context is not None:
        # 候选生成器会读取正式预测场景；把预测来源纳入身份，但不写入真实任务/轨迹。
        hasher.update(pickle.dumps(prediction_context, protocol=pickle.HIGHEST_PROTOCOL))
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
        source_state_fingerprint: str,
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
        state_hash = str(source_state_fingerprint)
        if not state_hash:
            raise ValueError("候选必须绑定到当前合法决策状态指纹")
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

    def __init__(
        self,
        slot_input: SlotInput,
        *,
        observation_spec: Any,
        train_origin_start: int,
        train_origin_end: int,
    ) -> None:
        self.node_ids = tuple(str(value) for value in observation_spec.cache_node_ids)
        configured_node_ids = tuple(str(value) for value in slot_input.model.cache_nodes["node_ids"])
        self.service_ids = tuple(int(value) for value in observation_spec.service_ids)
        configured_service_ids = tuple(
            int(value)
            for value in np.asarray(slot_input.model.service_catalog["service_ids"]).tolist()
        )
        if self.node_ids != configured_node_ids or self.service_ids != configured_service_ids:
            raise ValueError("DQN 特征顺序必须与正式 observation spec 和服务目录一致")
        self.observation_spec = observation_spec
        self.train_origin_start = int(train_origin_start)
        self.train_origin_end = int(train_origin_end)
        self.state_feature_contract_version = STATE_FEATURE_CONTRACT_VERSION
        self.candidate_feature_contract_version = CANDIDATE_FEATURE_CONTRACT_VERSION
        self.network_architecture_version = NETWORK_ARCHITECTURE_VERSION
        self.service_index = {service_id: index for index, service_id in enumerate(self.service_ids)}
        self.uav_ids = tuple(str(value) for value in observation_spec.uav_ids)
        self.vehicle_ids = tuple(str(value) for value in observation_spec.vehicle_ids)
        self.task_vehicle_ids = tuple(str(value) for value in observation_spec.task_vehicle_ids)
        configured_uavs = tuple(uav_configuration.uav_ids_from_project(dict(slot_input.model.project_cfg)))
        if self.uav_ids != configured_uavs:
            raise ValueError("正式 observation spec 的 UAV 顺序与配置不一致")
        self.bounds = tuple(float(value) for value in slot_input.model.flight_boundary)
        self.cache_nodes = deepcopy(dict(slot_input.model.cache_nodes))
        self.service_catalog = deepcopy(dict(slot_input.model.service_catalog))
        self.cache_config = slot_input.model.cache_config
        self.prediction_demand_beta = float(
            project_config.get_config_value(
                slot_input.model.project_cfg, "cache.discount_beta", 1.0
            )
        )
        if not 0.0 < self.prediction_demand_beta <= 1.0:
            raise ValueError("缓存预测需求折扣必须位于 (0,1]")
        self.horizon_count = int(observation_spec.prediction_horizon)
        configured_horizon = int(
            project_config.get_config_value(slot_input.model.project_cfg, "prediction.vehicle.horizon")
        )
        if self.horizon_count != configured_horizon:
            raise ValueError("DQN 预测时域必须与正式配置和 observation spec 一致")
        task_cfg = slot_input.model.project_cfg.get("task", {})
        service_cfg = slot_input.model.project_cfg.get("service", {}).get("generation", {})
        self.input_relative_jitter = float(task_cfg.get("input_relative_jitter", 0.0))
        self.load_relative_jitter = float(task_cfg.get("load_relative_jitter", 0.0))
        if not 0.0 <= self.input_relative_jitter < 1.0:
            raise ValueError("正式输入大小扰动范围必须位于 [0,1)")
        if not 0.0 <= self.load_relative_jitter < 1.0:
            raise ValueError("正式计算负载扰动范围必须位于 [0,1)")
        input_min_kb = float(service_cfg["input_size_min_kb"])
        input_max_kb = float(service_cfg["input_size_max_kb"])
        if input_min_kb <= 0.0 or input_max_kb < input_min_kb:
            raise ValueError("正式服务输入大小范围非法")
        base_input_min_bits = input_min_kb * 8000.0
        base_input_max_bits = input_max_kb * 8000.0
        self.service_mean_input_bits = np.asarray(
            slot_input.model.service_catalog["mean_input_size_bit"], dtype=np.float64
        ).reshape(-1)
        self.service_cycles_per_bit = np.asarray(
            slot_input.model.service_catalog["cycles_per_bit"], dtype=np.float64
        ).reshape(-1)
        if (
            self.service_mean_input_bits.shape != (len(self.service_ids),)
            or self.service_cycles_per_bit.shape != (len(self.service_ids),)
            or not np.isfinite(self.service_mean_input_bits).all()
            or not np.isfinite(self.service_cycles_per_bit).all()
            or np.any(self.service_mean_input_bits <= 0.0)
            or np.any(self.service_cycles_per_bit <= 0.0)
        ):
            raise ValueError("服务目录的输入大小/计算密度必须有限且与服务顺序一致")
        range_tolerance = 1.0e-6 * max(1.0, base_input_max_bits)
        if np.any(self.service_mean_input_bits < base_input_min_bits - range_tolerance) or np.any(
            self.service_mean_input_bits > base_input_max_bits + range_tolerance
        ):
            raise ValueError("服务目录输入基准超出正式配置声明的 KB 范围")
        # service_catalog 的 mean_input_size 单位是 bit；配置范围以 KB 给出，1 KB=8000 bit。
        self.base_max_input_bits = base_input_max_bits
        self.max_input_bits = base_input_max_bits * (1.0 + self.input_relative_jitter)
        self.max_cpu_cycles = float(
            np.max(base_input_max_bits * self.service_cycles_per_bit)
            * (1.0 + self.input_relative_jitter)
            * (1.0 + self.load_relative_jitter)
        )
        self.service_sizes_mb = np.asarray(
            slot_input.model.service_catalog["service_sizes_mb"], dtype=np.float64
        ).reshape(-1)
        self.total_service_size_mb = max(1.0, float(np.sum(self.service_sizes_mb)))
        self.map_diagonal_m = max(
            1.0,
            math.hypot(self.bounds[2] - self.bounds[0], self.bounds[3] - self.bounds[1]),
        )
        search = uav_configuration.uav_search_config_from_project(dict(slot_input.model.project_cfg))
        self.max_move_distance_m = max(
            1.0, float(search.max_horizontal_speed_mps) * float(search.dt_s)
        )
        node_count, service_count, uav_count = len(self.node_ids), len(self.service_ids), len(self.uav_ids)
        self.state_prediction_demand_dim = (
            len(self.vehicle_ids) * len(self.service_ids)
            + len(self.node_ids) * len(self.service_ids)
        )
        self.state_dim = (
            int(observation_spec.observation_dim)
            + 2 * self.horizon_count
            + self.state_prediction_demand_dim
        )
        self.candidate_dim = (
            3 + 2 + node_count * service_count + node_count
            + 2 * node_count * service_count + 4 + 8 * uav_count
        )

    def encode_state(
        self,
        formal_observation: Any,
        decision_forecast: DecisionForecast,
        prediction_scenario: Mapping[str, Any],
    ) -> np.ndarray:
        observation = np.asarray(formal_observation, dtype=np.float32).reshape(-1)
        if observation.shape != (int(self.observation_spec.observation_dim),):
            raise ValueError("正式 MAPPO observation 维度与 DQN observation spec 不一致")
        if not np.all(np.isfinite(observation)):
            raise ValueError("正式观测包含非有限值")
        mask = self._forecast_mask(decision_forecast)
        if self.observation_spec.horizon_mask_slice is not None:
            observed_mask = observation[self.observation_spec.horizon_mask_slice]
            if not np.array_equal(observed_mask.astype(np.int8), mask):
                raise ValueError("正式观测与预测时域 mask 不一致")
        expected_input: list[float] = []
        expected_cycles: list[float] = []
        for horizon_index, valid in enumerate(mask):
            if not valid:
                expected_input.append(0.0)
                expected_cycles.append(0.0)
                continue
            popularity = self._ordered_popularity(
                decision_forecast.predicted_service_popularity_horizon[horizon_index]
            )
            expected_input_bits = float(popularity @ self.service_mean_input_bits)
            expected_cpu_cycles = float(
                popularity @ (self.service_mean_input_bits * self.service_cycles_per_bit)
            )
            expected_input.append(self._normalize_workload(expected_input_bits, self.max_input_bits, "预测输入大小"))
            expected_cycles.append(self._normalize_workload(expected_cpu_cycles, self.max_cpu_cycles, "预测计算量"))
        scenario_demand_features = self._encode_candidate_generation_demand(
            prediction_scenario, mask
        )
        result = np.concatenate(
            (
                observation,
                np.asarray(expected_input, dtype=np.float32),
                np.asarray(expected_cycles, dtype=np.float32),
                scenario_demand_features,
            )
        )
        if result.shape != (self.state_dim,) or not np.all(np.isfinite(result)):
            raise ValueError("状态特征维度或数值无效")
        return result

    def _encode_candidate_generation_demand(
        self,
        prediction_scenario: Mapping[str, Any],
        horizon_mask: np.ndarray,
    ) -> np.ndarray:
        """编码候选生成器实际使用的预测需求充分统计量。"""

        scenario = dict(prediction_scenario)
        scenario_vehicle_ids = tuple(
            str(value) for value in np.asarray(scenario["vehicle_ids"]).tolist()
        )
        scenario_task_vehicle_ids = tuple(
            str(value)
            for value in np.asarray(scenario["task_vehicle_ids"]).tolist()
        )
        if scenario_vehicle_ids != self.vehicle_ids:
            raise ValueError("预测场景车辆顺序与 DQN 状态顺序不一致")
        if scenario_task_vehicle_ids != self.task_vehicle_ids:
            raise ValueError("预测场景任务车辆顺序与 DQN 状态顺序不一致")
        by_horizon = cache_configuration.predicted_vehicle_service_demand_by_horizon(
            scenario,
            len(self.service_ids),
            beta_discount=self.prediction_demand_beta,
            prediction_horizon_mask=horizon_mask,
        )
        expected_shape = (
            self.horizon_count,
            len(self.vehicle_ids),
            len(self.service_ids),
        )
        if by_horizon.shape != expected_shape or not np.isfinite(by_horizon).all():
            raise ValueError("预测场景的车辆-服务需求统计维度或数值非法")
        vehicle_service_demand = np.sum(by_horizon, axis=0)
        cache_node_demand = np.stack(
            [
                cache_configuration.node_specific_predicted_demand(
                    scenario,
                    node_index,
                    self.cache_nodes,
                    self.cache_config,
                    prediction_horizon_mask=horizon_mask,
                )
                for node_index in range(len(self.node_ids))
            ],
            axis=0,
        )
        max_vehicle_demand = float(self.horizon_count)
        max_node_demand = float(
            self.horizon_count * max(1, len(self.task_vehicle_ids))
        )
        normalized_vehicle_demand = vehicle_service_demand / max_vehicle_demand
        normalized_node_demand = cache_node_demand / max_node_demand
        normalized = np.concatenate(
            (normalized_vehicle_demand.reshape(-1), normalized_node_demand.reshape(-1))
        ).astype(np.float32)
        if normalized.shape != (self.state_prediction_demand_dim,):
            raise ValueError("候选生成预测需求特征维度错误")
        if not np.isfinite(normalized).all() or np.any(normalized < 0.0) or np.any(
            normalized > 1.0 + 1.0e-6
        ):
            raise ValueError("候选生成预测需求特征超出固定归一化范围")
        return normalized

    def encode_candidate(
        self,
        action: ActionCandidate,
        *,
        previous_cache_state: Any,
        previous_uav_positions: Mapping[str, Any],
        slot_input: SlotInput,
        decision_forecast: DecisionForecast,
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
        horizon_mask = self._forecast_mask(decision_forecast)
        demand_xy = [
            np.asarray(decision_forecast.predicted_vehicle_positions_horizon[horizon][vehicle_id], dtype=np.float64)
            for horizon, valid in enumerate(horizon_mask)
            if valid
            for vehicle_id in self.task_vehicle_ids
            if vehicle_id in decision_forecast.predicted_vehicle_positions_horizon[horizon]
        ]
        if not demand_xy:
            raise ValueError("合法预测时域中缺少任务车辆预测位置")
        for positions in (before_uav, after_uav):
            for uav_id in self.uav_ids:
                mean_distance = float(
                    np.mean([np.linalg.norm(positions[uav_id] - vehicle_xy) for vehicle_xy in demand_xy])
                )
                normalized_distance = mean_distance / self.map_diagonal_m
                if not np.isfinite(normalized_distance) or normalized_distance > 1.0 + 1.0e-6:
                    raise ValueError("预测需求距离超出地图对角线归一化范围")
                features.append(min(normalized_distance, 1.0))
        result = np.asarray(features, dtype=np.float32)
        if result.shape != (self.candidate_dim,) or not np.all(np.isfinite(result)):
            raise ValueError("候选动作特征维度或数值无效")
        return result

    def _normalize_xy(self, position: Any) -> list[float]:
        x, y = np.asarray(position, dtype=np.float64).reshape(2)
        x_min, y_min, x_max, y_max = self.bounds
        return [
            float(np.clip((x - x_min) / max(x_max - x_min, 1.0), 0.0, 1.0)),
            float(np.clip((y - y_min) / max(y_max - y_min, 1.0), 0.0, 1.0)),
        ]

    def _positions_xy(self, positions: Mapping[str, Any]) -> list[float]:
        return [value for uav_id in self.uav_ids for value in self._normalize_xy(positions[uav_id])]

    def _forecast_mask(self, forecast: DecisionForecast) -> np.ndarray:
        if len(forecast.predicted_vehicle_positions_horizon) != self.horizon_count or len(
            forecast.predicted_service_popularity_horizon
        ) != self.horizon_count:
            raise ValueError("预测车辆/服务流行度时域数量与正式配置不一致")
        if forecast.prediction_horizon_mask is None:
            if self.horizon_count != 1:
                raise ValueError("多时域决策必须提供正式预测有效 mask")
            mask = np.ones(1, dtype=np.int8)
        else:
            mask = np.asarray(forecast.prediction_horizon_mask, dtype=np.int8).reshape(-1)
        if mask.shape != (self.horizon_count,) or not np.all(np.isin(mask, [0, 1])):
            raise ValueError("预测有效 mask 必须是与时域对齐的二进制向量")
        if not np.any(mask):
            raise ValueError("正式决策至少需要一个有效预测时域")
        for horizon, valid in enumerate(mask):
            if valid:
                positions = forecast.predicted_vehicle_positions_horizon[horizon]
                missing = set(self.vehicle_ids) - set(str(value) for value in positions)
                if missing:
                    raise ValueError(f"预测时域 {horizon} 缺少车辆位置：{sorted(missing)}")
                if not set(self.task_vehicle_ids).issubset(positions):
                    raise ValueError(f"预测时域 {horizon} 缺少任务车辆位置")
        return mask

    def _ordered_popularity(self, popularity: Mapping[int, float]) -> np.ndarray:
        values = np.asarray([float(popularity[service_id]) for service_id in self.service_ids], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("预测服务流行度必须有限且非负")
        if not math.isclose(float(values.sum()), 1.0, rel_tol=1.0e-5, abs_tol=1.0e-5):
            raise ValueError("预测服务流行度必须符合正式归一化合同")
        return values

    @staticmethod
    def _normalize_workload(value: float, upper_bound: float, label: str) -> float:
        if not np.isfinite(value) or value < 0.0 or upper_bound <= 0.0:
            raise ValueError(f"{label}或其固定归一化上界非法")
        normalized = float(value / upper_bound)
        if normalized > 1.0:
            raise ValueError(f"{label}超过正式配置推导的固定上界")
        return normalized


__all__ = [
    "ActionCandidate",
    "CandidateOption",
    "CandidateFeatureEncoder",
    "DecisionForecast",
    "STATE_FEATURE_CONTRACT_VERSION",
    "CANDIDATE_FEATURE_CONTRACT_VERSION",
    "NETWORK_ARCHITECTURE_VERSION",
    "OPERATIONS",
]
