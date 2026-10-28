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
from .candidate_evaluator import SlotInput, copy_uav_positions


OPERATIONS = ("KEEP_STAY", "CACHE_UPDATE", "UAV_MOVE")
STATE_FEATURE_CONTRACT_VERSION = "formal_observation_spatial_role_slots_active_task_energy_state_v7"
CANDIDATE_FEATURE_CONTRACT_VERSION = "prediction_only_cache_spatial_role_slots_v4"
NETWORK_ARCHITECTURE_VERSION = "legal_candidate_mean_centered_dueling_q_active_task_energy_state_v8"


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
        changed_node_indices = np.flatnonzero(
            np.any(before_cache != after_cache, axis=1)
        )
        changed_node_ids = tuple(
            node_ids[int(index)] for index in changed_node_indices
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
        for raw_index in changed_node_indices:
            index = int(raw_index)
            node_id = node_ids[index]
            if not cache_configuration.is_legal_cache_replacement(
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

    @classmethod
    def from_cache_states(
        cls,
        *,
        previous_cache_state: Any,
        previous_uav_positions: Mapping[str, Any],
        target_cache_states: Sequence[Any],
        proposal_sources: Sequence[str],
        slot_input: SlotInput,
        source_state_fingerprint: str,
    ) -> tuple["ActionCandidate", ...]:
        """Build cache-only actions while sharing identical UAV validation."""

        before_cache = np.asarray(previous_cache_state, dtype=np.int8).copy()
        targets = np.asarray(target_cache_states, dtype=np.int8)
        if targets.ndim != 3 or targets.shape[1:] != before_cache.shape:
            raise ValueError("候选缓存矩阵必须与当前缓存形状一致")
        if len(targets) != len(proposal_sources):
            raise ValueError("候选缓存目标与 proposal source 数量不一致")
        if not np.all((before_cache == 0) | (before_cache == 1)) or not np.all(
            (targets == 0) | (targets == 1)
        ):
            raise ValueError("候选缓存矩阵必须只包含 0 和 1")
        before_uav = copy_uav_positions(previous_uav_positions)
        after_uav = copy_uav_positions(previous_uav_positions)
        if not before_uav or tuple(before_uav) != tuple(after_uav):
            raise ValueError("候选 UAV 顺序必须与当前状态一致")
        node_ids = tuple(
            str(value) for value in slot_input.model.cache_nodes["node_ids"]
        )
        if len(node_ids) != before_cache.shape[0]:
            raise ValueError("候选缓存行与正式缓存节点顺序不匹配")
        if not str(source_state_fingerprint):
            raise ValueError("候选必须绑定到当前合法决策状态指纹")

        # Every cache-only candidate has the same unchanged UAV transition.
        # Validate that shared transition once; cache legality is still checked
        # independently for every changed node in every target.
        triggers = {uav_id: 0 for uav_id in before_uav}
        uav_configuration.validate_uav_configuration_transition(
            before_uav,
            after_uav,
            triggers,
            dict(slot_input.model.project_cfg),
            tuple(float(value) for value in slot_input.model.flight_boundary),
        )
        removed_by_node = np.any(
            (before_cache[None, :, :] == 1) & (targets == 0), axis=2
        )
        added_by_node = np.any(
            (before_cache[None, :, :] == 0) & (targets == 1), axis=2
        )
        legal_changed_nodes = removed_by_node & added_by_node
        changed_masks = np.any(before_cache[None, :, :] != targets, axis=2)
        illegal_rows = np.flatnonzero(
            np.any(changed_masks & ~legal_changed_nodes, axis=1)
        )
        if illegal_rows.size:
            row_index = int(illegal_rows[0])
            invalid_nodes = np.flatnonzero(
                changed_masks[row_index] & ~legal_changed_nodes[row_index]
            )
            node_id = node_ids[int(invalid_nodes[0])]
            raise ValueError(f"缓存候选 {node_id} 不符合正式 REPLACE 规则")
        target_uav_tuple = tuple(
            (uav_id, float(after_uav[uav_id][0]), float(after_uav[uav_id][1]))
            for uav_id in after_uav
        )
        actions: list[ActionCandidate] = []
        state_hash = str(source_state_fingerprint)
        for target, changed_mask, proposal_source in zip(
            targets, changed_masks, proposal_sources, strict=True
        ):
            changed_indices = np.flatnonzero(changed_mask)
            if not changed_indices.size:
                raise ValueError("CACHE_UPDATE 必须改变至少一个缓存节点")
            changed_node_ids = tuple(node_ids[int(index)] for index in changed_indices)
            target_cache_tuple = tuple(
                tuple(row) for row in target.tolist()
            )
            identity = json.dumps(
                {
                    "operation": "CACHE_UPDATE",
                    "source_state": state_hash,
                    "cache": target_cache_tuple,
                    "uav": target_uav_tuple,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            actions.append(
                cls(
                    operation="CACHE_UPDATE",
                    candidate_id=hashlib.sha256(identity).hexdigest(),
                    source_state_fingerprint=state_hash,
                    target_cache_state=target_cache_tuple,
                    target_uav_positions=target_uav_tuple,
                    changed_node_ids=changed_node_ids,
                    moved_uav_ids=(),
                    proposal_source=str(proposal_source),
                )
            )
        return tuple(actions)

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

    @classmethod
    def create_many(
        cls, actions: Sequence[ActionCandidate], features: Any
    ) -> tuple["CandidateOption", ...]:
        """Validate a feature matrix once and materialize same-state options."""

        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(actions):
            raise ValueError("批量候选特征矩阵与动作数量不一致")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("候选动作特征必须为有限数值")
        return tuple(
            cls(action, tuple(map(float, row)))
            for action, row in zip(actions, matrix, strict=True)
        )

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action.to_dict(), "features": list(self.features)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateOption":
        return cls.create(ActionCandidate.from_dict(value["action"]), value["features"])


@dataclass(frozen=True)
class _CacheComplementarityContext:
    """从决策时可用预测构造、供同一候选集共享的缓存需求统计。"""

    predicted_demand_by_horizon: np.ndarray
    node_serviceability_by_horizon: np.ndarray
    uav_local_demand_by_service: np.ndarray
    uav_node_indices: tuple[int, ...]
    fixed_node_indices: tuple[int, ...]
    popularity_by_service: np.ndarray
    demand_normalizer: float
    cache_coverage_before: float | None
    cache_coverage_available: bool
    cache_coverage_before_by_horizon: np.ndarray
    cache_coverage_top_values: np.ndarray
    cache_coverage_top_node_indices: np.ndarray
    fixed_cache_coverage_before: np.ndarray
    fixed_cache_coverage_top_values: np.ndarray
    fixed_cache_coverage_top_node_indices: np.ndarray
    prediction_forecast: DecisionForecast
    task_vehicle_predicted_positions: tuple[np.ndarray, ...]
    current_uav_positions: Mapping[str, tuple[float, float]]
    current_uav_mean_task_distance_normalized: Mapping[str, float]
    previous_cache_state_source: np.ndarray
    unchanged_cache_cost: float | None
    unchanged_cache_download_delay_normalized: float | None


class CandidateFeatureEncoder:
    """将正式 Slot 状态和合法目标配置编码为固定维度向量。"""

    def __init__(
        self,
        slot_input: SlotInput,
        *,
        observation_spec: Any,
        train_origin_start: int,
        train_origin_end: int,
        k_max_slots: int,
        canonical_vehicle_ids: Sequence[str] | None = None,
        canonical_cache_node_ids: Sequence[str] | None = None,
    ) -> None:
        source_node_ids = tuple(str(value) for value in observation_spec.cache_node_ids)
        configured_node_ids = tuple(str(value) for value in slot_input.model.cache_nodes["node_ids"])
        self.service_ids = tuple(int(value) for value in observation_spec.service_ids)
        configured_service_ids = tuple(
            int(value)
            for value in np.asarray(slot_input.model.service_catalog["service_ids"]).tolist()
        )
        if source_node_ids != configured_node_ids or self.service_ids != configured_service_ids:
            raise ValueError("DQN 特征顺序必须与正式 observation spec 和服务目录一致")
        self.observation_spec = observation_spec
        self.train_origin_start = int(train_origin_start)
        self.train_origin_end = int(train_origin_end)
        if (
            isinstance(k_max_slots, bool)
            or int(k_max_slots) != k_max_slots
            or int(k_max_slots) <= 0
        ):
            raise ValueError("K_max must be a positive integer for active-task features")
        self.k_max_slots = int(k_max_slots)
        self.source_node_ids = source_node_ids
        self.node_ids = tuple(
            source_node_ids
            if canonical_cache_node_ids is None
            else (str(value) for value in canonical_cache_node_ids)
        )
        if len(self.node_ids) != len(source_node_ids) or set(self.node_ids) != set(source_node_ids):
            raise ValueError("规范缓存节点槽位必须与本 Episode 的真实节点一一对应")
        self.cache_node_reorder_indices = tuple(
            source_node_ids.index(node_id) for node_id in self.node_ids
        )
        self.state_feature_contract_version = STATE_FEATURE_CONTRACT_VERSION
        self.candidate_feature_contract_version = CANDIDATE_FEATURE_CONTRACT_VERSION
        self.network_architecture_version = NETWORK_ARCHITECTURE_VERSION
        self.service_index = {service_id: index for index, service_id in enumerate(self.service_ids)}
        self.uav_ids = tuple(str(value) for value in observation_spec.uav_ids)
        self.source_vehicle_ids = tuple(str(value) for value in observation_spec.vehicle_ids)
        self.source_task_vehicle_ids = tuple(
            str(value) for value in observation_spec.task_vehicle_ids
        )
        self.vehicle_ids = tuple(
            self.source_vehicle_ids
            if canonical_vehicle_ids is None
            else (str(value) for value in canonical_vehicle_ids)
        )
        if len(self.vehicle_ids) != len(self.source_vehicle_ids) or set(self.vehicle_ids) != set(self.source_vehicle_ids):
            raise ValueError("规范车辆槽位必须与本 Episode 的真实车辆一一对应")
        task_vehicle_set = set(self.source_task_vehicle_ids)
        if tuple(value for value in self.vehicle_ids if value not in task_vehicle_set) + tuple(
            value for value in self.vehicle_ids if value in task_vehicle_set
        ) != self.vehicle_ids:
            raise ValueError("规范车辆槽位必须先排列 SV，再排列 TV")
        self.vehicle_reorder_indices = tuple(
            self.source_vehicle_ids.index(vehicle_id) for vehicle_id in self.vehicle_ids
        )
        self.task_vehicle_ids = tuple(
            value for value in self.vehicle_ids if value in task_vehicle_set
        )
        self.execution_node_ids = (*self.node_ids, "CLOUD", "UNASSIGNED")
        configured_uavs = tuple(uav_configuration.uav_ids_from_project(dict(slot_input.model.project_cfg)))
        if self.uav_ids != configured_uavs:
            raise ValueError("正式 observation spec 的 UAV 顺序与配置不一致")
        self.bounds = tuple(float(value) for value in slot_input.model.flight_boundary)
        self.cache_nodes = deepcopy(dict(slot_input.model.cache_nodes))
        self.cache_nodes["node_ids"] = list(self.node_ids)
        for field in ("node_roles", "capacities_mb", "cache_backhaul_rate_bps"):
            values = slot_input.model.cache_nodes[field]
            self.cache_nodes[field] = [
                deepcopy(values[index]) for index in self.cache_node_reorder_indices
            ]
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
        self.upload_energy_reference_j = float(
            project_config.get_config_value(slot_input.model.project_cfg, "objective.E_ref")
        )
        if (
            not math.isfinite(self.upload_energy_reference_j)
            or self.upload_energy_reference_j <= 0.0
        ):
            raise ValueError("正式上传能耗参考尺度 objective.E_ref 必须为正且有限")
        self.service_sizes_mb = np.asarray(
            slot_input.model.service_catalog["service_sizes_mb"], dtype=np.float64
        ).reshape(-1)
        self.total_service_size_mb = max(1.0, float(np.sum(self.service_sizes_mb)))
        self.map_diagonal_m = max(
            1.0,
            math.hypot(self.bounds[2] - self.bounds[0], self.bounds[3] - self.bounds[1]),
        )
        search = uav_configuration.uav_search_config_from_project(dict(slot_input.model.project_cfg))
        self.uav_search_config = search
        self.max_move_distance_m = max(
            1.0, float(search.max_horizontal_speed_mps) * float(search.dt_s)
        )
        node_count, service_count, uav_count = len(self.node_ids), len(self.service_ids), len(self.uav_ids)
        self.state_prediction_demand_dim = (
            len(self.vehicle_ids) * len(self.service_ids)
            + len(self.node_ids) * len(self.service_ids)
        )
        self.active_task_feature_order = (
            "active",
            "stage_uploading",
            "stage_computing",
            "remaining_upload_workload_over_configured_max",
            "remaining_compute_workload_over_configured_max",
            "remaining_deadline_slots",
            "cumulative_upload_energy_over_E_ref",
            "remaining_upload_fraction",
            "remaining_compute_fraction",
            "remaining_deadline_fraction",
            *(f"service_id={value}" for value in self.service_ids),
            *(f"execution_node={value}" for value in self.execution_node_ids),
        )
        self.active_task_feature_width = len(self.active_task_feature_order)
        self.active_task_state_dim = (
            len(self.task_vehicle_ids)
            * self.k_max_slots
            * self.active_task_feature_width
        )
        self.state_dim = (
            int(observation_spec.observation_dim)
            + 2 * self.horizon_count
            + self.state_prediction_demand_dim
            + self.active_task_state_dim
        )
        self.candidate_dim = (
            3 + 2 + node_count * service_count + node_count
            + 2 * node_count * service_count + 4 + 8 * uav_count
            + 7 * service_count + 4
        )
        self._candidate_generation_demand_cache_scenario: Mapping[str, Any] | None = None
        self._candidate_generation_demand_cache_horizon_mask: bytes | None = None
        self._candidate_generation_demand_cache_node_values: np.ndarray | None = None
        self._candidate_generation_serviceability_cache_node_values: np.ndarray | None = None
        self._candidate_generation_demand_cache_hits = 0
        self._candidate_generation_demand_cache_misses = 0
        self._candidate_generation_serviceability_cache_hits = 0
        self._candidate_generation_serviceability_cache_misses = 0

    def encode_state(
        self,
        formal_observation: Any,
        decision_forecast: DecisionForecast,
        prediction_scenario: Mapping[str, Any],
        active_task_states: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        source_observation = np.asarray(formal_observation, dtype=np.float32).reshape(-1)
        if source_observation.shape != (int(self.observation_spec.observation_dim),):
            raise ValueError("environment observation 维度与 DQN observation spec 不一致")
        if not np.all(np.isfinite(source_observation)):
            raise ValueError("正式观测包含非有限值")
        mask = self._forecast_mask(decision_forecast)
        if self.observation_spec.horizon_mask_slice is not None:
            observed_mask = source_observation[self.observation_spec.horizon_mask_slice]
            if not np.array_equal(observed_mask.astype(np.int8), mask):
                raise ValueError("正式观测与预测时域 mask 不一致")
        observation = source_observation.copy()
        cache_rows = source_observation[self.observation_spec.cache_slice].reshape(
            len(self.source_node_ids), len(self.service_ids)
        )
        observation[self.observation_spec.cache_slice] = cache_rows[
            list(self.cache_node_reorder_indices)
        ].reshape(-1)
        vehicle_positions = source_observation[
            self.observation_spec.predicted_vehicle_slice
        ].reshape(self.horizon_count, len(self.source_vehicle_ids), 2)
        observation[self.observation_spec.predicted_vehicle_slice] = vehicle_positions[
            :, list(self.vehicle_reorder_indices), :
        ].reshape(-1)
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
        active_task_features = self._encode_active_task_states(active_task_states)
        result = np.concatenate(
            (
                observation,
                np.asarray(expected_input, dtype=np.float32),
                np.asarray(expected_cycles, dtype=np.float32),
                scenario_demand_features,
                active_task_features,
            )
        )
        if result.shape != (self.state_dim,) or not np.all(np.isfinite(result)):
            raise ValueError("状态特征维度或数值无效")
        return result

    def _encode_active_task_states(
        self, active_task_states: Sequence[Mapping[str, Any]]
    ) -> np.ndarray:
        """Encode tasks by canonical TV then arrival order, with K_max padded slots."""

        if isinstance(active_task_states, (str, bytes)):
            raise TypeError("active_task_states must be a sequence of task snapshots")
        snapshots = [dict(state) for state in active_task_states]
        by_vehicle: dict[str, list[dict[str, Any]]] = {
            vehicle_id: [] for vehicle_id in self.task_vehicle_ids
        }
        for state in snapshots:
            vehicle_id = str(state.get("vehicle_id", ""))
            if vehicle_id not in by_vehicle:
                raise ValueError(f"active task has unknown or non-TV vehicle {vehicle_id!r}")
            by_vehicle[vehicle_id].append(state)

        features: list[float] = []
        service_index = self.service_index
        node_index = {node_id: index for index, node_id in enumerate(self.execution_node_ids)}
        for vehicle_id in self.task_vehicle_ids:
            tasks = sorted(
                by_vehicle[vehicle_id],
                key=lambda item: (int(item["arrival_slot"]), str(item["task_id"])),
            )
            if len(tasks) > self.k_max_slots:
                raise ValueError(
                    f"TV {vehicle_id} has {len(tasks)} active tasks, exceeding K_max="
                    f"{self.k_max_slots}"
                )
            for state in tasks:
                service_id = int(state["service_id"])
                execution_node = state.get("execution_node")
                stage = str(state.get("execution_stage", "")).upper()
                original_bits = float(state["input_bits"])
                remaining_bits = float(state["remaining_upload_bits"])
                original_cycles = float(state["cpu_cycles"])
                remaining_cycles = float(state["remaining_compute_cycles"])
                remaining_slots = int(state["remaining_slots"])
                cumulative_upload_energy = float(state["cumulative_upload_energy_j"])
                if service_id not in service_index:
                    raise ValueError(f"active task references unknown service {service_id}")
                node_key = (
                    "UNASSIGNED" if execution_node is None else str(execution_node)
                )
                if node_key not in node_index:
                    raise ValueError(
                        f"active task {state.get('task_id')!r} has unknown execution node "
                        f"{execution_node!r}"
                    )
                if stage not in {"UPLOADING", "COMPUTING"}:
                    raise ValueError(f"unsupported active task stage {stage!r}")
                numeric = (
                    original_bits,
                    remaining_bits,
                    original_cycles,
                    remaining_cycles,
                    cumulative_upload_energy,
                )
                if (
                    not all(math.isfinite(value) for value in numeric)
                    or original_bits <= 0.0
                    or original_cycles <= 0.0
                    or cumulative_upload_energy < 0.0
                    or not 0.0 <= remaining_bits <= original_bits
                    or not 0.0 <= remaining_cycles <= original_cycles
                    or not 0 < remaining_slots <= self.k_max_slots
                ):
                    raise ValueError(f"active task {state.get('task_id')!r} has invalid work/deadline")
                if (stage == "UPLOADING") != (remaining_bits > 0.0):
                    raise ValueError(
                        f"active task {state.get('task_id')!r} stage disagrees with remaining upload"
                    )
                record = [
                    1.0,
                    float(stage == "UPLOADING"),
                    float(stage == "COMPUTING"),
                    self._normalize_workload(
                        remaining_bits,
                        self.max_input_bits,
                        "活动任务剩余上传量",
                    ),
                    self._normalize_workload(
                        remaining_cycles,
                        self.max_cpu_cycles,
                        "活动任务剩余计算量",
                    ),
                    float(remaining_slots),
                    cumulative_upload_energy / self.upload_energy_reference_j,
                    remaining_bits / original_bits,
                    remaining_cycles / original_cycles,
                    remaining_slots / self.k_max_slots,
                ]
                record.extend(
                    float(index == service_index[service_id])
                    for index in range(len(self.service_ids))
                )
                record.extend(
                    float(index == node_index[node_key])
                    for index in range(len(self.execution_node_ids))
                )
                features.extend(record)
            for _ in range(self.k_max_slots - len(tasks)):
                features.extend([0.0] * self.active_task_feature_width)
        result = np.asarray(features, dtype=np.float32)
        if result.shape != (self.active_task_state_dim,) or not np.isfinite(result).all():
            raise ValueError("active-task features have invalid shape or non-finite values")
        return result

    def restore_source_formal_observation(self, canonical_observation: Any) -> np.ndarray:
        """将保存的规范槽位观测还原为正式 ObservationSpec 的源数组顺序。"""

        observation = np.asarray(canonical_observation, dtype=np.float32).reshape(-1).copy()
        if observation.shape != (int(self.observation_spec.observation_dim),):
            raise ValueError("规范观测维度与正式 ObservationSpec 不一致")
        cache_rows = observation[self.observation_spec.cache_slice].reshape(
            len(self.node_ids), len(self.service_ids)
        )
        observation[self.observation_spec.cache_slice] = cache_rows[
            np.argsort(self.cache_node_reorder_indices).tolist()
        ].reshape(-1)
        positions = observation[
            self.observation_spec.predicted_vehicle_slice
        ].reshape(self.horizon_count, len(self.vehicle_ids), 2)
        observation[self.observation_spec.predicted_vehicle_slice] = positions[
            :, np.argsort(self.vehicle_reorder_indices).tolist(), :
        ].reshape(-1)
        return observation

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
        if scenario_vehicle_ids != self.source_vehicle_ids:
            raise ValueError("预测场景车辆顺序与正式 ObservationSpec 不一致")
        if set(scenario_task_vehicle_ids) != set(self.task_vehicle_ids):
            raise ValueError("预测场景任务车辆身份与正式 ObservationSpec 不一致")
        source_by_horizon = cache_configuration.predicted_vehicle_service_demand_by_horizon(
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
        if (
            source_by_horizon.shape != expected_shape
            or not np.isfinite(source_by_horizon).all()
        ):
            raise ValueError("预测场景的车辆-服务需求统计维度或数值非法")
        by_horizon = source_by_horizon[:, list(self.vehicle_reorder_indices), :]
        vehicle_service_demand = np.sum(by_horizon, axis=0)
        cache_node_serviceability = np.stack(
            [
                cache_configuration._node_predicted_serviceability_weights(
                    scenario,
                    node_index,
                    self.cache_nodes,
                    self.cache_config,
                )
                for node_index in range(len(self.node_ids))
            ],
            axis=0,
        )
        cache_node_demand = np.stack(
            [
                cache_configuration.node_specific_predicted_demand(
                    scenario,
                    node_index,
                    self.cache_nodes,
                    self.cache_config,
                    prediction_horizon_mask=horizon_mask,
                    precomputed_vehicle_service_demand_by_horizon=source_by_horizon,
                    precomputed_serviceability_weights=(
                        cache_node_serviceability[node_index]
                    ),
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
        cache_node_demand.setflags(write=False)
        cache_node_serviceability.setflags(write=False)
        self._candidate_generation_demand_cache_scenario = prediction_scenario
        self._candidate_generation_demand_cache_horizon_mask = np.ascontiguousarray(
            horizon_mask, dtype=np.int8
        ).tobytes()
        self._candidate_generation_demand_cache_node_values = cache_node_demand
        self._candidate_generation_serviceability_cache_node_values = (
            cache_node_serviceability
        )
        return normalized

    def cached_candidate_generation_node_demands(
        self,
        prediction_scenario: Mapping[str, Any],
        horizon_mask: Any,
    ) -> np.ndarray | None:
        """Reuse demand rows computed moments earlier for this same scenario object."""

        mask_key = np.ascontiguousarray(horizon_mask, dtype=np.int8).tobytes()
        if (
            prediction_scenario
            is not self._candidate_generation_demand_cache_scenario
            or mask_key != self._candidate_generation_demand_cache_horizon_mask
            or self._candidate_generation_demand_cache_node_values is None
        ):
            self._candidate_generation_demand_cache_misses = int(
                getattr(self, "_candidate_generation_demand_cache_misses", 0)
            ) + 1
            return None
        self._candidate_generation_demand_cache_hits = int(
            getattr(self, "_candidate_generation_demand_cache_hits", 0)
        ) + 1
        source_order = np.argsort(self.cache_node_reorder_indices)
        return self._candidate_generation_demand_cache_node_values[source_order]

    def cached_candidate_generation_serviceability_weights(
        self,
        prediction_scenario: Mapping[str, Any],
        horizon_mask: Any,
    ) -> np.ndarray | None:
        """Reuse exact node geometry weights for the same scenario and mask."""

        mask_key = np.ascontiguousarray(horizon_mask, dtype=np.int8).tobytes()
        cached_weights = getattr(
            self, "_candidate_generation_serviceability_cache_node_values", None
        )
        if (
            prediction_scenario
            is not getattr(self, "_candidate_generation_demand_cache_scenario", None)
            or mask_key
            != getattr(self, "_candidate_generation_demand_cache_horizon_mask", None)
            or cached_weights is None
        ):
            self._candidate_generation_serviceability_cache_misses = int(
                getattr(self, "_candidate_generation_serviceability_cache_misses", 0)
            ) + 1
            return None
        self._candidate_generation_serviceability_cache_hits = int(
            getattr(self, "_candidate_generation_serviceability_cache_hits", 0)
        ) + 1
        # The stored rows already use canonical ``self.node_ids`` order, which
        # is also the order consumed by prepare_cache_complementarity_context.
        return cached_weights

    def encode_candidate(
        self,
        action: ActionCandidate,
        *,
        previous_cache_state: Any,
        previous_uav_positions: Mapping[str, Any],
        slot_input: SlotInput,
        decision_forecast: DecisionForecast,
        cache_complementarity_context: _CacheComplementarityContext | None = None,
    ) -> np.ndarray:
        before_cache_source = np.asarray(previous_cache_state, dtype=np.int8)
        after_cache_source = action.cache_array()
        expected_cache_shape = (len(self.source_node_ids), len(self.service_ids))
        if before_cache_source.shape != expected_cache_shape or after_cache_source.shape != expected_cache_shape:
            raise ValueError("候选缓存状态与本 Episode 的真实缓存节点顺序不匹配")
        before_cache = before_cache_source[list(self.cache_node_reorder_indices)].copy()
        after_cache = after_cache_source[list(self.cache_node_reorder_indices)].copy()
        cache_unchanged = np.array_equal(before_cache_source, after_cache_source)
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
        complementarity = cache_complementarity_context
        if complementarity is None:
            complementarity = self.prepare_cache_complementarity_context(
                previous_cache_state=before_cache,
                previous_uav_positions=before_uav,
                slot_input=slot_input,
                decision_forecast=decision_forecast,
            )
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
        if (
            cache_unchanged
            and complementarity.unchanged_cache_cost is not None
            and complementarity.unchanged_cache_download_delay_normalized is not None
            and np.array_equal(
                before_cache_source,
                complementarity.previous_cache_state_source,
            )
        ):
            # KEEP_STAY and UAV_MOVE repeatedly ask the formal cost routine to
            # validate and reduce the same unchanged cache. The state context
            # already ran that exact routine once for this immutable cache.
            cache_cost = complementarity.unchanged_cache_cost
            cache_download_delay_normalized = (
                complementarity.unchanged_cache_download_delay_normalized
            )
        else:
            actual_changed_node_mask = np.asarray(
                [float(node_id in action.changed_node_ids) for node_id in self.source_node_ids],
                dtype=np.int8,
            )
            cache_details = cache_configuration.cache_update_cost_details(
                before_cache_source,
                after_cache_source,
                actual_changed_node_mask,
                self.service_sizes_mb,
                dict(slot_input.model.cache_nodes),
                slot_input.model.cache_config,
            )
            cache_cost = float(cache_details["cache_cost"])
            cache_download_delay_normalized = float(
                cache_details["cache_download_delay_normalized"]
            )
        features.extend(
            [
                added_mb / self.total_service_size_mb,
                cache_cost,
                cache_download_delay_normalized,
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
        uav_triggers = {uav_id: int(uav_id in action.moved_uav_ids) for uav_id in self.uav_ids}
        unchanged_uav_positions = not action.moved_uav_ids and all(
            np.array_equal(before_uav[uav_id], after_uav[uav_id])
            for uav_id in self.uav_ids
        )
        if unchanged_uav_positions:
            movement_cost = 0.0
        else:
            movement = uav_configuration.movement_summary(
                before_uav, after_uav, uav_triggers, self.uav_search_config
            )
            movement_cost = float(movement[1])
        features.append(movement_cost)
        forecast_matches_context = (
            decision_forecast is complementarity.prediction_forecast
        )
        horizon_mask = (
            None
            if forecast_matches_context
            else self._forecast_mask(decision_forecast)
        )
        demand_xy = (
            complementarity.task_vehicle_predicted_positions
            if forecast_matches_context
            else tuple(
                np.asarray(
                    decision_forecast.predicted_vehicle_positions_horizon[horizon][vehicle_id],
                    dtype=np.float64,
                )
                for horizon, valid in enumerate(horizon_mask)
                if valid
                for vehicle_id in self.task_vehicle_ids
                if vehicle_id in decision_forecast.predicted_vehicle_positions_horizon[horizon]
            )
        )
        if not demand_xy:
            raise ValueError("合法预测时域中缺少任务车辆预测位置")
        cached_positions_match = forecast_matches_context and all(
            tuple(float(value) for value in before_uav[uav_id])
            == complementarity.current_uav_positions[uav_id]
            for uav_id in self.uav_ids
        )
        for position_index, positions in enumerate((before_uav, after_uav)):
            for uav_id in self.uav_ids:
                if (
                    cached_positions_match
                    and position_index == 0
                ):
                    normalized_distance = complementarity.current_uav_mean_task_distance_normalized[
                        uav_id
                    ]
                elif (
                    cached_positions_match
                    and np.array_equal(positions[uav_id], before_uav[uav_id])
                ):
                    normalized_distance = complementarity.current_uav_mean_task_distance_normalized[
                        uav_id
                    ]
                else:
                    mean_distance = float(
                        np.mean(
                            [
                                np.linalg.norm(positions[uav_id] - vehicle_xy)
                                for vehicle_xy in demand_xy
                            ]
                        )
                    )
                    normalized_distance = mean_distance / self.map_diagonal_m
                    if (
                        not np.isfinite(normalized_distance)
                        or normalized_distance > 1.0 + 1.0e-6
                    ):
                        raise ValueError("预测需求距离超出地图对角线归一化范围")
                    normalized_distance = min(normalized_distance, 1.0)
                features.append(float(normalized_distance))
        if cache_unchanged:
            unchanged_coverage = float(
                complementarity.cache_coverage_before
                if complementarity.cache_coverage_before is not None
                else 0.0
            )
            features.extend([0.0] * (7 * service_count))
            features.extend(
                [
                    float(complementarity.cache_coverage_available),
                    unchanged_coverage,
                    unchanged_coverage,
                    0.0,
                ]
            )
        else:
            features.extend(
                self._encode_cache_complementarity_candidate(
                    before_cache, after_cache, complementarity
                )
            )
        result = np.asarray(features, dtype=np.float32)
        if result.shape != (self.candidate_dim,) or not np.all(np.isfinite(result)):
            raise ValueError("候选动作特征维度或数值无效")
        return result

    def encode_candidates(
        self,
        actions: Sequence[ActionCandidate],
        *,
        previous_cache_state: Any,
        previous_uav_positions: Mapping[str, Any],
        slot_input: SlotInput,
        decision_forecast: DecisionForecast,
        cache_complementarity_context: _CacheComplementarityContext | None = None,
    ) -> np.ndarray:
        """Encode a same-state CACHE_UPDATE group with shared invariant work."""

        if not actions:
            return np.empty((0, self.candidate_dim), dtype=np.float32)
        if any(action.operation != "CACHE_UPDATE" for action in actions):
            return np.asarray(
                tuple(
                    self.encode_candidate(
                        action,
                        previous_cache_state=previous_cache_state,
                        previous_uav_positions=previous_uav_positions,
                        slot_input=slot_input,
                        decision_forecast=decision_forecast,
                        cache_complementarity_context=cache_complementarity_context,
                    )
                    for action in actions
                ),
                dtype=np.float32,
            )
        before_source = np.asarray(previous_cache_state, dtype=np.int8)
        expected_shape = (len(self.source_node_ids), len(self.service_ids))
        if before_source.shape != expected_shape:
            raise ValueError("候选缓存状态与本 Episode 的真实缓存节点顺序不匹配")
        after_source = np.asarray(
            [action.target_cache_state for action in actions], dtype=np.int8
        )
        if after_source.shape != (len(actions), *expected_shape) or not np.all(
            (after_source == 0) | (after_source == 1)
        ):
            raise ValueError("批量候选缓存状态形状或二进制取值非法")
        before_cache = before_source[list(self.cache_node_reorder_indices)]
        after_cache = after_source[:, list(self.cache_node_reorder_indices), :]
        if any(action.changed_node_ids == () or action.moved_uav_ids for action in actions):
            raise ValueError("CACHE_UPDATE 必须只改变缓存")
        changed_node_mask = np.any(
            after_cache != before_cache[None, :, :], axis=2
        )
        if any(
            {
                self.node_ids[index]
                for index in np.flatnonzero(changed_node_mask[row])
            }
            != set(action.changed_node_ids)
            for row, action in enumerate(actions)
        ):
            raise ValueError("批量 CACHE_UPDATE 动作识别结果与目标缓存不一致")

        complementarity = cache_complementarity_context
        if complementarity is None:
            complementarity = self.prepare_cache_complementarity_context(
                previous_cache_state=before_cache,
                previous_uav_positions=previous_uav_positions,
                slot_input=slot_input,
                decision_forecast=decision_forecast,
            )
        before_uav = copy_uav_positions(previous_uav_positions)
        forecast_matches_context = (
            decision_forecast is complementarity.prediction_forecast
        )
        if forecast_matches_context:
            demand_xy = complementarity.task_vehicle_predicted_positions
            cached_positions_match = all(
                tuple(float(value) for value in before_uav[uav_id])
                == complementarity.current_uav_positions[uav_id]
                for uav_id in self.uav_ids
            )
        else:
            horizon_mask = self._forecast_mask(decision_forecast)
            demand_xy = tuple(
                np.asarray(
                    decision_forecast.predicted_vehicle_positions_horizon[horizon][vehicle_id],
                    dtype=np.float64,
                )
                for horizon, valid in enumerate(horizon_mask)
                if valid
                for vehicle_id in self.task_vehicle_ids
                if vehicle_id in decision_forecast.predicted_vehicle_positions_horizon[horizon]
            )
            cached_positions_match = False
        if not demand_xy:
            raise ValueError("合法预测时域中缺少任务车辆预测位置")

        before_position_features = [
            self._normalize_xy(before_uav[uav_id]) for uav_id in self.uav_ids
        ]
        before_distances: list[float] = []
        for uav_id in self.uav_ids:
            if cached_positions_match:
                normalized_distance = complementarity.current_uav_mean_task_distance_normalized[
                    uav_id
                ]
            else:
                mean_distance = float(
                    np.mean(
                        [
                            np.linalg.norm(before_uav[uav_id] - vehicle_xy)
                            for vehicle_xy in demand_xy
                        ]
                    )
                )
                normalized_distance = mean_distance / self.map_diagonal_m
                if (
                    not np.isfinite(normalized_distance)
                    or normalized_distance > 1.0 + 1.0e-6
                ):
                    raise ValueError("预测需求距离超出地图对角线归一化范围")
                normalized_distance = min(normalized_distance, 1.0)
            before_distances.append(float(normalized_distance))

        added_rows = np.maximum(
            after_cache - before_cache[None, :, :], 0
        ).astype(np.float32)
        removed_rows = np.maximum(
            before_cache[None, :, :] - after_cache, 0
        ).astype(np.float32)
        cache_nodes = dict(slot_input.model.cache_nodes)
        normalization = cache_nodes.get("cache_cost_normalization")
        if not isinstance(normalization, dict):
            raise RuntimeError("Cache 成本归一化参考量必须在环境初始化时固定。")
        cache_backhaul_rates = np.asarray(
            cache_nodes["cache_backhaul_rate_bps"], dtype=np.float64
        )
        cache_cost_references = np.asarray(
            normalization["reference_s_by_node"], dtype=np.float64
        )
        if (
            cache_backhaul_rates.shape != (len(self.source_node_ids),)
            or cache_cost_references.shape != (len(self.source_node_ids),)
            or not np.isfinite(cache_backhaul_rates).all()
            or not np.isfinite(cache_cost_references).all()
            or np.any(cache_backhaul_rates <= 0.0)
            or np.any(cache_cost_references <= 0.0)
        ):
            raise ValueError("Cache 成本速率或归一化参考量无效")
        cache_sizes = np.asarray(self.service_sizes_mb, dtype=np.float64)
        complementarity_features = self._encode_cache_complementarity_candidates(
            before_cache, after_cache, complementarity
        )
        candidate_count = len(actions)
        feature_matrix = np.empty(
            (candidate_count, self.candidate_dim), dtype=np.float32
        )
        cursor = 0
        feature_matrix[:, cursor : cursor + 5] = np.asarray(
            [0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32
        )
        cursor += 5
        cache_width = len(self.node_ids) * len(self.service_ids)
        feature_matrix[:, cursor : cursor + cache_width] = after_cache.astype(
            np.float32
        ).reshape(candidate_count, cache_width)
        cursor += cache_width
        node_width = len(self.node_ids)
        feature_matrix[:, cursor : cursor + node_width] = changed_node_mask.astype(
            np.float32
        )
        cursor += node_width
        feature_matrix[:, cursor : cursor + cache_width] = removed_rows.reshape(
            candidate_count, cache_width
        )
        cursor += cache_width
        feature_matrix[:, cursor : cursor + cache_width] = added_rows.reshape(
            candidate_count, cache_width
        )
        cursor += cache_width
        added_fraction_column = cursor
        cache_cost_column = cursor + 1
        cursor += 3
        position_row = np.asarray(
            [value for position in before_position_features for value in position],
            dtype=np.float32,
        )
        uav_position_width = len(self.uav_ids) * 2
        feature_matrix[:, cursor : cursor + uav_position_width] = position_row
        cursor += uav_position_width
        feature_matrix[:, cursor : cursor + uav_position_width] = position_row
        cursor += uav_position_width
        feature_matrix[:, cursor : cursor + (2 * len(self.uav_ids) + 1)] = 0.0
        cursor += 2 * len(self.uav_ids) + 1
        before_distance_row = np.asarray(before_distances, dtype=np.float32)
        feature_matrix[:, cursor : cursor + len(self.uav_ids)] = before_distance_row
        cursor += len(self.uav_ids)
        feature_matrix[:, cursor : cursor + len(self.uav_ids)] = before_distance_row
        cursor += len(self.uav_ids)
        complementarity_width = 7 * len(self.service_ids) + 4
        feature_matrix[:, cursor : cursor + complementarity_width] = (
            complementarity_features
        )
        cursor += complementarity_width
        if cursor != self.candidate_dim:
            raise RuntimeError("批量候选特征布局与合同维度不一致")
        for row_index, (after_source_row, added) in enumerate(
            zip(after_source, added_rows, strict=True)
        ):
            added_mb = float(added.sum(axis=0) @ self.service_sizes_mb)
            additions_source = np.maximum(
                after_source_row - before_source, 0
            )
            loaded_mb = additions_source @ cache_sizes
            loaded_bits = loaded_mb * 8.0e6
            download_delay_s = loaded_bits / cache_backhaul_rates
            normalized_delay = download_delay_s / cache_cost_references
            if (
                not np.isfinite(normalized_delay).all()
                or np.any(normalized_delay < 0.0)
            ):
                raise FloatingPointError("normalized cache cost is not finite")
            cache_cost = float(np.sum(normalized_delay))
            feature_matrix[row_index, added_fraction_column] = (
                added_mb / self.total_service_size_mb
            )
            feature_matrix[row_index, cache_cost_column] = cache_cost
            feature_matrix[row_index, cache_cost_column + 1] = cache_cost
        if not np.all(np.isfinite(feature_matrix)):
            raise ValueError("批量候选动作特征维度或数值无效")
        return feature_matrix

    def prepare_cache_complementarity_context(
        self,
        *,
        previous_cache_state: Any,
        previous_uav_positions: Mapping[str, Any],
        slot_input: SlotInput,
        decision_forecast: DecisionForecast,
        prepare_unchanged_cache_cost: bool = True,
    ) -> _CacheComplementarityContext:
        """预先计算共享预测统计，避免对每个缓存候选重复做几何归约。"""

        scenario = dict(slot_input.scenario_slice)
        required = {
            "vehicle_ids",
            "task_vehicle_ids",
            "positions_xy",
            "arrival",
            "service_id",
            "rho_hat",
        }
        missing = sorted(required - set(scenario))
        if missing:
            raise ValueError("缓存互补性特征缺少预测字段：" + ", ".join(missing))
        mask = self._forecast_mask(decision_forecast).astype(bool, copy=False)
        if slot_input.prediction_horizon_mask is not None and not np.array_equal(
            mask.astype(np.int8),
            np.asarray(slot_input.prediction_horizon_mask, dtype=np.int8).reshape(-1),
        ):
            raise ValueError("缓存互补性预测 mask 与 SlotInput 不一致")
        scenario_vehicle_ids = tuple(
            str(value) for value in np.asarray(scenario["vehicle_ids"]).tolist()
        )
        scenario_task_vehicle_ids = tuple(
            str(value) for value in np.asarray(scenario["task_vehicle_ids"]).tolist()
        )
        if scenario_vehicle_ids != self.source_vehicle_ids:
            raise ValueError("缓存互补性预测车辆顺序与正式 ObservationSpec 不一致")
        if set(scenario_task_vehicle_ids) != set(self.task_vehicle_ids):
            raise ValueError("缓存互补性预测任务车辆身份与正式 ObservationSpec 不一致")

        cached_serviceability = self.cached_candidate_generation_serviceability_weights(
            slot_input.scenario_slice, mask
        )

        demand = cache_configuration.predicted_vehicle_service_demand_by_horizon(
            scenario,
            len(self.service_ids),
            beta_discount=self.prediction_demand_beta,
            prediction_horizon_mask=mask,
        )
        expected_demand_shape = (
            self.horizon_count,
            len(self.vehicle_ids),
            len(self.service_ids),
        )
        if demand.shape != expected_demand_shape or not np.isfinite(demand).all():
            raise ValueError("缓存互补性预测需求形状或数值非法")
        demand = demand[:, list(self.vehicle_reorder_indices), :]

        # UAV 使用决策时的当前位置；预测时域内不假设 UAV 已经移动。
        feature_cache_nodes = deepcopy(self.cache_nodes)
        current_uav = copy_uav_positions(previous_uav_positions)
        for node_index, node_id in enumerate(self.node_ids):
            if str(feature_cache_nodes["node_roles"][node_index]).upper() == "UAV":
                if node_id not in current_uav:
                    raise ValueError(f"缓存互补性特征缺少 UAV {node_id} 的当前位置")
                feature_cache_nodes["fixed_positions"][node_id] = current_uav[
                    node_id
                ].copy()

        node_weights: list[np.ndarray] = []
        for node_index in range(len(self.node_ids)):
            cache_static_geometry = cached_serviceability is not None
            if (
                cache_static_geometry
                and str(feature_cache_nodes["node_roles"][node_index]).upper()
                == "UAV"
            ):
                node_id = self.node_ids[node_index]
                cache_static_geometry = np.array_equal(
                    np.asarray(
                        self.cache_nodes["fixed_positions"][node_id],
                        dtype=np.float64,
                    ),
                    np.asarray(
                        feature_cache_nodes["fixed_positions"][node_id],
                        dtype=np.float64,
                    ),
                )
            if cache_static_geometry:
                assert cached_serviceability is not None
                weights = cached_serviceability[node_index]
            else:
                weights = cache_configuration._node_predicted_serviceability_weights(
                    scenario,
                    node_index,
                    feature_cache_nodes,
                    self.cache_config,
                )
            if weights.shape != (self.horizon_count, len(self.vehicle_ids)):
                raise ValueError("缓存互补性节点服务性权重维度非法")
            # 无效时域即使含占位坐标也不参与任何需求统计。
            weights = np.where(mask[:, None], weights, 0.0)
            if (
                not np.isfinite(weights).all()
                or np.any(weights < 0.0)
                or np.any(weights > 1.0)
            ):
                raise ValueError("缓存互补性节点服务性权重非法")
            node_weights.append(weights[:, list(self.vehicle_reorder_indices)])
        node_weights_array = np.stack(node_weights, axis=0)
        uav_indices = tuple(
            index
            for index, role in enumerate(feature_cache_nodes["node_roles"])
            if str(role).upper() == "UAV"
        )
        fixed_indices = tuple(index for index in range(len(self.node_ids)) if index not in uav_indices)
        if tuple(self.node_ids[index] for index in uav_indices) != self.uav_ids:
            raise ValueError("缓存节点中的 UAV 顺序与正式 observation spec 不一致")
        uav_local_demand = np.stack(
            [
                np.einsum(
                    "hv,hvf->f",
                    node_weights_array[node_index],
                    demand,
                )
                for node_index in uav_indices
            ],
            axis=0,
        )
        popularity = np.zeros(len(self.service_ids), dtype=np.float64)
        for horizon, valid in enumerate(mask):
            if valid:
                popularity += self._ordered_popularity(
                    decision_forecast.predicted_service_popularity_horizon[horizon]
                )
        popularity /= float(np.count_nonzero(mask))

        cache_state_source = np.asarray(previous_cache_state, dtype=np.int8)
        if cache_state_source.shape != (len(self.source_node_ids), len(self.service_ids)):
            raise ValueError("缓存互补性输入缓存矩阵维度错误")
        previous_cache_state_source = cache_state_source.copy()
        previous_cache_state_source.setflags(write=False)
        unchanged_cache_cost: float | None = None
        unchanged_cache_download_delay_normalized: float | None = None
        if prepare_unchanged_cache_cost:
            unchanged_details = cache_configuration.cache_update_cost_details(
                cache_state_source,
                cache_state_source,
                np.zeros(len(self.source_node_ids), dtype=np.int8),
                self.service_sizes_mb,
                dict(slot_input.model.cache_nodes),
                slot_input.model.cache_config,
            )
            unchanged_cache_cost = float(unchanged_details["cache_cost"])
            unchanged_cache_download_delay_normalized = float(
                unchanged_details["cache_download_delay_normalized"]
            )
        cache_state = cache_state_source[list(self.cache_node_reorder_indices)].copy()
        total_predicted_demand = float(np.sum(demand))
        coverage_available = total_predicted_demand > 1.0e-12
        cache_coverage_before: float | None = None
        node_coverage = (
            node_weights_array[:, :, :, None]
            * cache_state[:, None, None, :]
        )
        coverage = np.zeros_like(demand)
        for node_index in range(len(self.node_ids)):
            coverage = np.maximum(coverage, node_coverage[node_index])
        if coverage_available:
            cache_coverage_before = float(
                np.sum(demand * np.clip(coverage, 0.0, 1.0))
                / total_predicted_demand
            )
        cache_coverage_top_values, cache_coverage_top_node_indices = (
            self._top_node_coverage(node_coverage, tuple(range(len(self.node_ids))))
        )
        fixed_cache_coverage = np.zeros_like(demand)
        for node_index in fixed_indices:
            fixed_cache_coverage = np.maximum(
                fixed_cache_coverage, node_coverage[node_index]
            )
        (
            fixed_cache_coverage_top_values,
            fixed_cache_coverage_top_node_indices,
        ) = self._top_node_coverage(node_coverage, fixed_indices)
        task_predicted_positions = tuple(
            np.asarray(
                decision_forecast.predicted_vehicle_positions_horizon[horizon][vehicle_id],
                dtype=np.float64,
            ).reshape(2).copy()
            for horizon, valid in enumerate(mask)
            if valid
            for vehicle_id in self.task_vehicle_ids
            if vehicle_id in decision_forecast.predicted_vehicle_positions_horizon[horizon]
        )
        if not task_predicted_positions:
            raise ValueError("合法预测时域中缺少任务车辆预测位置")
        uav_position_tuples = {
            uav_id: tuple(float(value) for value in current_uav[uav_id])
            for uav_id in self.uav_ids
        }
        uav_mean_distance_normalized: dict[str, float] = {}
        for uav_id in self.uav_ids:
            mean_distance = float(
                np.mean(
                    [
                        np.linalg.norm(current_uav[uav_id] - vehicle_xy)
                        for vehicle_xy in task_predicted_positions
                    ]
                )
            )
            normalized_distance = mean_distance / self.map_diagonal_m
            if not np.isfinite(normalized_distance) or normalized_distance > 1.0 + 1.0e-6:
                raise ValueError("预测需求距离超出地图对角线归一化范围")
            uav_mean_distance_normalized[uav_id] = min(normalized_distance, 1.0)
        task_vehicle_count = max(1, len(self.task_vehicle_ids))
        demand_normalizer = float(max(1, self.horizon_count * task_vehicle_count))
        for array in (
            demand,
            node_weights_array,
            uav_local_demand,
            popularity,
            coverage,
            cache_coverage_top_values,
            cache_coverage_top_node_indices,
            fixed_cache_coverage,
            fixed_cache_coverage_top_values,
            fixed_cache_coverage_top_node_indices,
        ):
            array.setflags(write=False)
        for array in task_predicted_positions:
            array.setflags(write=False)
        return _CacheComplementarityContext(
            predicted_demand_by_horizon=demand,
            node_serviceability_by_horizon=node_weights_array,
            uav_local_demand_by_service=uav_local_demand,
            uav_node_indices=uav_indices,
            fixed_node_indices=fixed_indices,
            popularity_by_service=popularity,
            demand_normalizer=demand_normalizer,
            cache_coverage_before=cache_coverage_before,
            cache_coverage_available=coverage_available,
            cache_coverage_before_by_horizon=coverage,
            cache_coverage_top_values=cache_coverage_top_values,
            cache_coverage_top_node_indices=cache_coverage_top_node_indices,
            fixed_cache_coverage_before=fixed_cache_coverage,
            fixed_cache_coverage_top_values=fixed_cache_coverage_top_values,
            fixed_cache_coverage_top_node_indices=fixed_cache_coverage_top_node_indices,
            prediction_forecast=decision_forecast,
            task_vehicle_predicted_positions=task_predicted_positions,
            current_uav_positions=uav_position_tuples,
            current_uav_mean_task_distance_normalized=uav_mean_distance_normalized,
            previous_cache_state_source=previous_cache_state_source,
            unchanged_cache_cost=unchanged_cache_cost,
            unchanged_cache_download_delay_normalized=(
                unchanged_cache_download_delay_normalized
            ),
        )

    @staticmethod
    def _top_node_coverage(
        node_coverage: np.ndarray, node_indices: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Keep the three strongest node contributions for exact small exclusions."""

        indices = np.asarray(node_indices, dtype=np.intp)
        contributions = node_coverage[indices]
        top_count = min(3, len(indices))
        top_local_indices = np.argpartition(
            contributions, kth=len(indices) - top_count, axis=0
        )[-top_count:]
        top_values = np.take_along_axis(
            contributions, top_local_indices, axis=0
        )
        top_node_indices = indices[top_local_indices]
        return top_values, top_node_indices

    @staticmethod
    def _coverage_without_changed_nodes(
        top_values: np.ndarray,
        top_node_indices: np.ndarray,
        changed_node_indices: Sequence[int],
        *,
        total_node_count: int,
    ) -> np.ndarray | None:
        """Return exact max coverage after excluding at most two old node rows."""

        if len(changed_node_indices) > 2 or (
            len(changed_node_indices) == top_values.shape[0]
            and top_values.shape[0] < total_node_count
        ):
            return None
        eligible = np.ones(top_node_indices.shape, dtype=bool)
        for node_index in changed_node_indices:
            eligible &= top_node_indices != int(node_index)
        return np.max(np.where(eligible, top_values, 0.0), axis=0)

    def _encode_cache_complementarity_candidate(
        self,
        before_cache: np.ndarray,
        after_cache: np.ndarray,
        context: _CacheComplementarityContext,
    ) -> list[float]:
        """按固定次序编码 UAV 局部需求、固定覆盖缺口、流行度和副本变化。

        每组服务向量依次为新增 UAV 本地需求、其中固定节点可覆盖量、
        固定节点未覆盖量、删除服务本地需求、全网流行度净变化、
        变更服务的原有副本比例、候选副本数变化；末尾为覆盖率可用标记、
        更新前后覆盖率及差值。覆盖量使用正式预测服务性权重，不代表
        Matching 的实际分配或任务命中。
        """

        service_count = len(self.service_ids)
        changed_node_indices = np.flatnonzero(
            np.any(after_cache != before_cache, axis=1)
        )
        added = np.maximum(after_cache - before_cache, 0).astype(np.float64)
        removed = np.maximum(before_cache - after_cache, 0).astype(np.float64)
        uav_added_demand = np.zeros(service_count, dtype=np.float64)
        uav_removed_demand = np.zeros(service_count, dtype=np.float64)
        fixed_available_added_demand = np.zeros(service_count, dtype=np.float64)
        uav_replica_before = np.zeros(service_count, dtype=np.float64)
        replicas_before = np.sum(before_cache, axis=0, dtype=np.float64)

        changed_uav_services = np.zeros(service_count, dtype=np.float64)
        for uav_index, node_index in enumerate(context.uav_node_indices):
            changed_uav_services = np.maximum(
                changed_uav_services,
                np.maximum(added[node_index], removed[node_index]),
            )
            local_by_service = context.uav_local_demand_by_service[uav_index]
            uav_added_demand += local_by_service * added[node_index]
            uav_removed_demand += local_by_service * removed[node_index]
            uav_replica_before += (
                replicas_before
                * np.maximum(added[node_index], removed[node_index])
                / max(1, len(self.node_ids))
            )

        fixed_coverage = np.zeros_like(context.predicted_demand_by_horizon)
        if np.any(changed_uav_services > 0.0):
            changed_fixed_indices = tuple(
                int(index)
                for index in changed_node_indices
                if int(index) in context.fixed_node_indices
            )
            if not changed_fixed_indices:
                fixed_coverage = context.fixed_cache_coverage_before
            else:
                base_fixed_coverage = self._coverage_without_changed_nodes(
                    context.fixed_cache_coverage_top_values,
                    context.fixed_cache_coverage_top_node_indices,
                    changed_fixed_indices,
                    total_node_count=len(context.fixed_node_indices),
                )
                if base_fixed_coverage is None:
                    for node_index in context.fixed_node_indices:
                        fixed_coverage = np.maximum(
                            fixed_coverage,
                            context.node_serviceability_by_horizon[
                                node_index, :, :, None
                            ]
                            * after_cache[node_index][None, None, :],
                        )
                else:
                    fixed_coverage = np.zeros_like(
                        context.predicted_demand_by_horizon
                    )
                    fixed_coverage[...] = base_fixed_coverage
                    for node_index in changed_fixed_indices:
                        fixed_coverage = np.maximum(
                            fixed_coverage,
                            context.node_serviceability_by_horizon[
                                node_index, :, :, None
                            ]
                            * after_cache[node_index][None, None, :],
                        )
            for node_index in context.uav_node_indices:
                added_services = added[node_index]
                if not np.any(added_services):
                    continue
                uav_weights = context.node_serviceability_by_horizon[node_index]
                local_weights = uav_weights[:, :, None] * context.predicted_demand_by_horizon
                fixed_available_by_service = np.sum(
                    local_weights * np.clip(fixed_coverage, 0.0, 1.0), axis=(0, 1)
                )
                fixed_available_added_demand += (
                    fixed_available_by_service * added_services
                )

        uav_unavailable_added_demand = np.maximum(
            uav_added_demand - fixed_available_added_demand, 0.0
        )
        popularity_delta = (
            np.sum(after_cache - before_cache, axis=0, dtype=np.float64)
            * context.popularity_by_service
            / max(1, len(self.node_ids))
        )
        replica_delta = (
            np.sum(after_cache - before_cache, axis=0, dtype=np.float64)
            / max(1, len(self.node_ids))
        )

        total_demand = float(np.sum(context.predicted_demand_by_horizon))
        coverage_before = context.cache_coverage_before
        coverage_after: float | None = None
        if context.cache_coverage_available:
            if not changed_node_indices.size:
                coverage_after = coverage_before
            else:
                base_coverage = self._coverage_without_changed_nodes(
                    context.cache_coverage_top_values,
                    context.cache_coverage_top_node_indices,
                    changed_node_indices,
                    total_node_count=len(self.node_ids),
                )
                if base_coverage is None:
                    coverage = np.zeros_like(context.predicted_demand_by_horizon)
                    for node_index in range(len(self.node_ids)):
                        coverage = np.maximum(
                            coverage,
                            context.node_serviceability_by_horizon[
                                node_index, :, :, None
                            ]
                            * after_cache[node_index][None, None, :],
                        )
                else:
                    coverage = np.zeros_like(
                        context.predicted_demand_by_horizon
                    )
                    coverage[...] = base_coverage
                    for node_index in changed_node_indices:
                        coverage = np.maximum(
                            coverage,
                            context.node_serviceability_by_horizon[
                                node_index, :, :, None
                            ]
                            * after_cache[node_index][None, None, :],
                        )
                coverage_after = float(
                    np.sum(
                        context.predicted_demand_by_horizon
                        * np.clip(coverage, 0.0, 1.0)
                    )
                    / total_demand
                )
        coverage_available = float(context.cache_coverage_available)
        coverage_before_value = float(coverage_before) if coverage_before is not None else 0.0
        coverage_after_value = float(coverage_after) if coverage_after is not None else 0.0
        coverage_delta = (
            coverage_after_value - coverage_before_value
            if context.cache_coverage_available
            else 0.0
        )
        values = np.concatenate(
            (
                uav_added_demand / context.demand_normalizer,
                fixed_available_added_demand / context.demand_normalizer,
                uav_unavailable_added_demand / context.demand_normalizer,
                uav_removed_demand / context.demand_normalizer,
                popularity_delta,
                uav_replica_before,
                replica_delta,
                np.asarray(
                    [
                        coverage_available,
                        coverage_before_value,
                        coverage_after_value,
                        coverage_delta,
                    ],
                    dtype=np.float64,
                ),
            )
        )
        if values.shape != (7 * service_count + 4,) or not np.isfinite(values).all():
            raise ValueError("缓存互补性候选特征维度或数值非法")
        return values.astype(np.float32).tolist()

    def _encode_cache_complementarity_candidates(
        self,
        before_cache: np.ndarray,
        after_cache: np.ndarray,
        context: _CacheComplementarityContext,
        *,
        chunk_size: int = 512,
    ) -> np.ndarray:
        """Encode cache complementarity in bounded candidate batches.

        The candidate axis is vectorized while horizon, vehicle, node, and
        service reductions retain the scalar encoder's axes and ordering.
        """

        before = np.asarray(before_cache, dtype=np.int8)
        after = np.asarray(after_cache, dtype=np.int8)
        if (
            before.ndim != 2
            or after.ndim != 3
            or after.shape[1:] != before.shape
            or before.shape != (len(self.node_ids), len(self.service_ids))
        ):
            raise ValueError("批量互补性缓存矩阵形状无效")
        if int(chunk_size) <= 0:
            raise ValueError("互补性候选 chunk_size 必须为正数")
        if not len(after):
            return np.empty((0, 7 * len(self.service_ids) + 4), dtype=np.float32)

        chunks: list[np.ndarray] = []
        for start in range(0, len(after), int(chunk_size)):
            chunks.append(
                self._encode_cache_complementarity_candidate_chunk(
                    before, after[start : start + int(chunk_size)], context
                )
            )
        return np.concatenate(chunks, axis=0)

    def _encode_cache_complementarity_candidate_chunk(
        self,
        before_cache: np.ndarray,
        after_cache: np.ndarray,
        context: _CacheComplementarityContext,
    ) -> np.ndarray:
        """Vectorized implementation of the scalar cache complementarity rule."""

        candidate_count = len(after_cache)
        service_count = len(self.service_ids)
        node_count = len(self.node_ids)
        delta = after_cache - before_cache[None, :, :]
        changed_node_mask = np.any(delta != 0, axis=2)
        changed_node_counts = np.count_nonzero(changed_node_mask, axis=1)
        added = np.maximum(delta, 0).astype(np.float64)
        removed = np.maximum(-delta, 0).astype(np.float64)
        replicas_before = np.sum(before_cache, axis=0, dtype=np.float64)
        uav_added_demand = np.zeros((candidate_count, service_count), dtype=np.float64)
        uav_removed_demand = np.zeros_like(uav_added_demand)
        fixed_available_added_demand = np.zeros_like(uav_added_demand)
        uav_replica_before = np.zeros_like(uav_added_demand)
        changed_uav_services = np.zeros_like(uav_added_demand)
        replica_normalizer = float(max(1, node_count))

        for uav_index, node_index in enumerate(context.uav_node_indices):
            uav_added = added[:, node_index]
            uav_removed = removed[:, node_index]
            changed_uav_services = np.maximum(
                changed_uav_services, np.maximum(uav_added, uav_removed)
            )
            local_by_service = context.uav_local_demand_by_service[uav_index]
            uav_added_demand += local_by_service[None, :] * uav_added
            uav_removed_demand += local_by_service[None, :] * uav_removed
            uav_replica_before += (
                replicas_before[None, :]
                * np.maximum(uav_added, uav_removed)
                / replica_normalizer
            )

        uav_changed_rows = np.any(changed_uav_services > 0.0, axis=1)
        fixed_coverage = np.zeros(
            (candidate_count, *context.predicted_demand_by_horizon.shape),
            dtype=np.float64,
        )
        fixed_node_indices = tuple(int(index) for index in context.fixed_node_indices)
        fixed_count = len(fixed_node_indices)
        if fixed_count and np.any(uav_changed_rows):
            changed_fixed_counts = np.count_nonzero(
                changed_node_mask[:, fixed_node_indices], axis=1
            )
            unchanged_fixed_rows = uav_changed_rows & (changed_fixed_counts == 0)
            fixed_coverage[unchanged_fixed_rows] = (
                context.fixed_cache_coverage_before
            )

            top_values = context.fixed_cache_coverage_top_values
            top_node_indices = context.fixed_cache_coverage_top_node_indices
            top_count = top_values.shape[0]
            fallback_fixed_rows = uav_changed_rows & (
                (changed_fixed_counts > 2)
                | (
                    (changed_fixed_counts == top_count)
                    & (top_count < fixed_count)
                )
            )
            regular_fixed_rows = uav_changed_rows & (changed_fixed_counts > 0) & (
                ~fallback_fixed_rows
            )
            if np.any(regular_fixed_rows):
                eligible = np.ones(
                    (candidate_count, *top_node_indices.shape), dtype=bool
                )
                for node_index in fixed_node_indices:
                    eligible &= ~(
                        changed_node_mask[:, node_index, None, None, None, None]
                        & (top_node_indices[None, ...] == node_index)
                    )
                base_fixed_coverage = np.max(
                    np.where(
                        eligible,
                        top_values[None, ...],
                        0.0,
                    ),
                    axis=1,
                )
                fixed_coverage[regular_fixed_rows] = base_fixed_coverage[
                    regular_fixed_rows
                ]
                for node_index in fixed_node_indices:
                    rows = regular_fixed_rows & changed_node_mask[:, node_index]
                    if not np.any(rows):
                        continue
                    changed_node_coverage = (
                        context.node_serviceability_by_horizon[
                            node_index, :, :, None
                        ]
                        * after_cache[rows, node_index, None, None, :]
                    )
                    fixed_coverage[rows] = np.maximum(
                        fixed_coverage[rows], changed_node_coverage
                    )

            if np.any(fallback_fixed_rows):
                fixed_coverage[fallback_fixed_rows] = 0.0
                for node_index in fixed_node_indices:
                    node_coverage = (
                        context.node_serviceability_by_horizon[
                            node_index, :, :, None
                        ]
                        * after_cache[
                            fallback_fixed_rows, node_index, None, None, :
                        ]
                    )
                    fixed_coverage[fallback_fixed_rows] = np.maximum(
                        fixed_coverage[fallback_fixed_rows], node_coverage
                    )

            for uav_index, node_index in enumerate(context.uav_node_indices):
                added_services = added[:, node_index]
                if not np.any(added_services):
                    continue
                uav_weights = context.node_serviceability_by_horizon[node_index]
                local_weights = (
                    uav_weights[:, :, None]
                    * context.predicted_demand_by_horizon
                )
                fixed_available_by_service = np.sum(
                    local_weights[None, ...]
                    * np.clip(fixed_coverage, 0.0, 1.0),
                    axis=(1, 2),
                )
                fixed_available_added_demand += (
                    fixed_available_by_service * added_services
                )

        uav_unavailable_added_demand = np.maximum(
            uav_added_demand - fixed_available_added_demand, 0.0
        )
        cache_delta_by_service = np.sum(delta, axis=1, dtype=np.float64)
        replica_normalizer = float(max(1, node_count))
        popularity_delta = (
            cache_delta_by_service * context.popularity_by_service[None, :]
            / replica_normalizer
        )
        replica_delta = cache_delta_by_service / replica_normalizer

        coverage_before_value = float(
            context.cache_coverage_before
            if context.cache_coverage_before is not None
            else 0.0
        )
        coverage_after = np.zeros(candidate_count, dtype=np.float64)
        coverage_delta = np.zeros(candidate_count, dtype=np.float64)
        if context.cache_coverage_available:
            unchanged_rows = changed_node_counts == 0
            coverage_after[unchanged_rows] = coverage_before_value
            total_demand = float(np.sum(context.predicted_demand_by_horizon))
            for candidate_index in np.flatnonzero(~unchanged_rows):
                changed_nodes = np.flatnonzero(
                    changed_node_mask[candidate_index]
                )
                base_coverage = self._coverage_without_changed_nodes(
                    context.cache_coverage_top_values,
                    context.cache_coverage_top_node_indices,
                    changed_nodes,
                    total_node_count=node_count,
                )
                if base_coverage is None:
                    candidate_coverage = np.zeros_like(
                        context.predicted_demand_by_horizon
                    )
                    for node_index in range(node_count):
                        candidate_coverage = np.maximum(
                            candidate_coverage,
                            context.node_serviceability_by_horizon[
                                node_index, :, :, None
                            ]
                            * after_cache[
                                candidate_index, node_index
                            ][None, None, :],
                        )
                else:
                    candidate_coverage = np.zeros_like(
                        context.predicted_demand_by_horizon
                    )
                    candidate_coverage[...] = base_coverage
                    for raw_node_index in changed_nodes:
                        node_index = int(raw_node_index)
                        candidate_coverage = np.maximum(
                            candidate_coverage,
                            context.node_serviceability_by_horizon[
                                node_index, :, :, None
                            ]
                            * after_cache[
                                candidate_index, node_index
                            ][None, None, :],
                        )
                if np.array_equal(
                    candidate_coverage,
                    context.cache_coverage_before_by_horizon,
                ):
                    coverage_after[candidate_index] = coverage_before_value
                else:
                    # Preserve the scalar encoder's exact reduction order for
                    # the coverage delta while vectorizing the service vectors.
                    coverage_after[candidate_index] = float(
                        np.sum(
                            context.predicted_demand_by_horizon
                            * np.clip(candidate_coverage, 0.0, 1.0)
                        )
                        / total_demand
                    )
            coverage_delta = coverage_after - coverage_before_value

        coverage_available = float(context.cache_coverage_available)
        coverage_features = np.column_stack(
            (
                np.full(candidate_count, coverage_available, dtype=np.float64),
                np.full(
                    candidate_count, coverage_before_value, dtype=np.float64
                ),
                coverage_after,
                coverage_delta,
            )
        )
        values = np.concatenate(
            (
                uav_added_demand / context.demand_normalizer,
                fixed_available_added_demand / context.demand_normalizer,
                uav_unavailable_added_demand / context.demand_normalizer,
                uav_removed_demand / context.demand_normalizer,
                popularity_delta,
                uav_replica_before,
                replica_delta,
                coverage_features,
            ),
            axis=1,
        )
        expected_shape = (candidate_count, 7 * service_count + 4)
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError("批量缓存互补性候选特征维度或数值非法")
        return values.astype(np.float32)

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
                missing = set(self.source_vehicle_ids) - set(str(value) for value in positions)
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
