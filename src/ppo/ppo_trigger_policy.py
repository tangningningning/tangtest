#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trigger-only PPO policy building blocks.

本模块只实现冻结主线中的：

    s_cfg(t) -> conditionally-factorized trigger policy + value

它不调用 configuration optimizer、slot environment、Matching 或 KKT；也不读取
prediction_scenarios.npz、tasks.csv、trajectory.csv 等文件。上游负责把
``C(t-1)``, ``Q^U(t-1)``, ``q_hat^V(t)``, ``rho_hat(t)`` 准备好后传入纯函数。

PyTorch 部分仅使用官方成熟 API，没有复制 PyTorch 源码：
https://docs.pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
https://docs.pytorch.org/docs/stable/distributions.html#categorical
"""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

try:  # 允许在没有 torch 的环境中导入 action/observation 纯函数。
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - 当前运行环境可能未安装 torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from common import project_config


LEGACY_SHARED_HEAD_ARCHITECTURE_ID = "v3_shared_logit_head"
TRIGGER_HEADS_ARCHITECTURE_ID = "v3_encoder_trigger_heads_frozen_ar_v1"
TRIGGER_HEADS_ARCHITECTURE_VERSION = 1
FULL_SHARED_ACTOR_ARCHITECTURE_ID = "v3_full_shared_actor_frozen_policy_batch_v1"
FULL_SHARED_ACTOR_ARCHITECTURE_VERSION = 1


@dataclass(frozen=True)
class TriggerActionSpec:
    """Trigger action 的唯一权威顺序。

    顺序固定为：先所有 dynamic cache node trigger，再所有 UAV movement trigger。
    ``action_vector`` 中每个元素都是 0/1，且可直接转换为 ``execute_slot()`` 需要的
    ``cache_triggers`` 和 ``uav_triggers``。
    """

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
        labels = tuple([f"z_{node_id}^ca" for node_id in cache_node_ids] + [f"z_{uav_id}^q" for uav_id in uav_ids])
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
        uav_vector = np.asarray([int(uav_triggers[uav_id]) for uav_id in self.uav_ids], dtype=np.int8)
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
    observation_dim: int

    @classmethod
    def create(
        cls,
        cache_node_ids: list[str] | tuple[str, ...],
        service_ids: list[int] | tuple[int, ...],
        uav_ids: list[str] | tuple[str, ...],
        vehicle_ids: list[str] | tuple[str, ...],
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
        return cls(
            cache_node_ids=cache_node_ids_t,
            service_ids=service_ids_t,
            uav_ids=uav_ids_t,
            vehicle_ids=vehicle_ids_t,
            cache_slice=cache_slice,
            uav_position_slice=uav_position_slice,
            predicted_vehicle_slice=predicted_vehicle_slice,
            popularity_slice=popularity_slice,
            observation_dim=start,
        )

    @classmethod
    def from_cache_nodes(
        cls,
        cache_nodes: dict[str, Any],
        service_ids: list[int] | tuple[int, ...],
        uav_ids: list[str] | tuple[str, ...],
        vehicle_ids: list[str] | tuple[str, ...],
    ) -> "ConfigurationObservationSpec":
        return cls.create(cache_nodes["node_ids"], service_ids, uav_ids, vehicle_ids)


@dataclass(frozen=True)
class TriggerActionSample:
    action_vector: np.ndarray
    cache_triggers: np.ndarray
    uav_triggers: dict[str, int]
    joint_log_prob: float
    conditional_entropy: float
    value: float
    per_action_probabilities: np.ndarray
    per_action_probability_table: np.ndarray
    per_action_log_probs: np.ndarray
    per_action_logits: np.ndarray
    per_action_masks: np.ndarray


@dataclass(frozen=True)
class TriggerEvaluation:
    joint_log_prob: Any
    entropy: Any
    value: Any
    per_action_logits: Any
    per_action_probabilities: Any
    per_action_probability_table: Any
    per_action_log_probs: Any
    per_action_entropy: Any


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError(
            "当前 Python 环境未安装 PyTorch；ppo_trigger_policy 的网络采样/"
            "evaluate_actions 需要 torch。"
        )


def _validate_binary_vector(value: Any, expected_length: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.int8).reshape(-1)
    if vector.shape != (expected_length,):
        raise ValueError(f"{name} shape={vector.shape}，期望 {(expected_length,)}。")
    if not np.all(np.isin(vector, [0, 1])):
        raise ValueError(f"{name} 必须是 0/1 向量。")
    return vector


def _normalize_action_masks(
    action_masks: Any,
    batch_size: int,
    action_dim: int,
    device: Any,
) -> Any:
    """规范化为 ``[B, action_dim, 2]`` 的有效 binary action mask。"""

    if action_masks is None:
        return torch.ones(
            (batch_size, action_dim, 2),
            dtype=torch.bool,
            device=device,
        )
    masks = torch.as_tensor(action_masks, dtype=torch.bool, device=device)
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    if masks.shape != (batch_size, action_dim, 2):
        raise ValueError(
            f"action_masks shape={tuple(masks.shape)}，期望 "
            f"{(batch_size, action_dim, 2)}。"
        )
    if not bool(torch.all(torch.any(masks, dim=-1)).item()):
        raise ValueError("action_masks 不能把某个 binary action 的 0/1 两个值同时屏蔽。")
    return masks


def _resolve_sampling_mask(
    action_mask: Any,
    action_index: int,
    prefix_actions: np.ndarray,
    action_dim: int,
) -> np.ndarray:
    if action_mask is None:
        return np.ones(2, dtype=bool)
    if callable(action_mask):
        raw = action_mask(int(action_index), prefix_actions.copy())
    else:
        raw = np.asarray(action_mask)
        if raw.shape == (action_dim, 2):
            raw = raw[action_index]
    mask = np.asarray(raw, dtype=bool).reshape(-1)
    if mask.shape != (2,) or not np.any(mask):
        raise ValueError(f"action mask[{action_index}] 必须是至少允许一个值的 [2] mask。")
    return mask


def _normalize_xy(xy: Any, boundary: tuple[float, float, float, float], label: str) -> np.ndarray:
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
        values = np.asarray([normalized_dict[service_id] for service_id in service_ids], dtype=np.float64)
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


def build_configuration_observation(
    previous_cache_state: Any,
    previous_uav_positions: dict[str, Any] | Any,
    predicted_vehicle_positions: dict[str, Any] | Any,
    predicted_service_popularity: dict[int | str, float] | Any,
    observation_spec: ConfigurationObservationSpec,
    flight_boundary: tuple[float, float, float, float],
) -> np.ndarray:
    """构造 ``s_cfg(t)`` 的纯函数。

    只包含 ``C(t-1)``, ``Q^U(t-1)``, ``q_hat^V(t)``, ``rho_hat(t)``；
    不包含 TV fixed local cache、slot t 真实任务、slot t 真实车辆位置，也不展开
    stochastic ``S × H_p`` scenarios。
    """

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
    observation = np.concatenate([cache_features, uav_features, vehicle_features, popularity])
    if observation.shape != (observation_spec.observation_dim,):
        raise AssertionError("observation length 与 spec 不一致。")
    if not np.all(np.isfinite(observation)):
        raise ValueError("observation 包含 NaN/Inf。")
    return observation.astype(np.float32)


def ppo_network_config_from_project(project_cfg: dict[str, Any]) -> dict[str, Any]:
    """读取本阶段唯一需要的 PPO 网络结构参数。"""

    hidden_sizes = project_config.get_config_value(project_cfg, "algorithm.ppo.hidden_sizes", [128, 128])
    ar_hidden = int(project_config.get_config_value(project_cfg, "algorithm.ppo.autoregressive_hidden_size", 64))
    return {
        "hidden_sizes": tuple(int(value) for value in hidden_sizes),
        "autoregressive_hidden_size": ar_hidden,
    }


def set_torch_seed(seed: int) -> None:
    """由 caller 显式控制 torch seed；模块 import 时不修改全局 RNG。"""

    _require_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class PerTriggerQCritic(nn.Module if nn is not None else object):
    """独立的 per-trigger conditional Q critic。

    Q critic 只接收已经由 caller detach 的 autoregressive conditional hidden。
    因此本模块的参数不会天然把梯度传回 Actor；正式 update 仍会显式检查
    ``h_i.detach()``。未采样 action 没有直接环境监督，第一版只对 sampled
    action 的 Monte-Carlo suffix return 做监督，另一 action 的 Q 依赖跨样本
    泛化，不能宣称是真实 counterfactual replay Q。
    """

    def __init__(self, action_dim: int, hidden_size: int) -> None:
        _require_torch()
        super().__init__()
        if int(action_dim) <= 0 or int(hidden_size) <= 0:
            raise ValueError("PerTriggerQCritic 的 action_dim/hidden_size 必须大于0。")
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)
        self.q_head = nn.Linear(self.hidden_size, 2)

    def forward(self, conditional_hidden_states: Any) -> Any:
        hidden = torch.as_tensor(conditional_hidden_states, dtype=torch.float32)
        if hidden.ndim != 3 or tuple(hidden.shape[1:]) != (self.action_dim, self.hidden_size):
            raise ValueError(
                "conditional_hidden_states shape="
                f"{tuple(hidden.shape)}，期望 [B,{self.action_dim},{self.hidden_size}]。"
            )
        if not torch.all(torch.isfinite(hidden)):
            raise ValueError("conditional_hidden_states 包含 NaN/Inf。")
        return self.q_head(hidden)


def build_per_trigger_q_critic_preserving_rng(
    action_dim: int,
    hidden_size: int,
    device: Any | None = None,
) -> PerTriggerQCritic:
    """初始化 Q critic，同时恢复初始化前的 global torch RNG。

    新增 Q 层不能改变旧 Actor/scalar-V 的 Episode 1 sampling trajectory。
    这里使用官方 torch RNG state API 保护 CPU/CUDA 状态；Q critic 参数本身
    仍使用 PyTorch 默认初始化方式，不引入新的超参数或自定义初始化算法。
    """

    _require_torch()
    cpu_state = torch.get_rng_state().clone()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
    critic = PerTriggerQCritic(action_dim, hidden_size)
    if device is not None:
        critic = critic.to(device)
    torch.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    return critic


class CentralizedJointActionCritic(nn.Module if nn is not None else object):
    """集中式联合动作 critic：``Q_c(s,a)=V_c(s)+D_c(s,a)``。

    ``V_c`` 只接收 observation，``D_c`` 接收 observation 与完整的 dynamic
    joint action vector
    joint action。两条分支不与 Actor 共享 encoder；action-dependent 分支的
    最终 head 显式零初始化，因此新 critic 初始时对不同 joint action 给出相同
    Q 值，避免凭随机初始化制造 counterfactual preference。
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        _require_torch()
        super().__init__()
        if int(observation_dim) <= 0 or int(action_dim) <= 0:
            raise ValueError("CentralizedJointActionCritic 的维度必须大于0。")
        if not hidden_sizes or any(int(size) <= 0 for size in hidden_sizes):
            raise ValueError("CentralizedJointActionCritic hidden_sizes 必须是正整数列表。")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        sizes = tuple(int(size) for size in hidden_sizes)

        state_layers: list[Any] = []
        state_last = self.observation_dim
        for hidden in sizes:
            state_layers.extend([nn.Linear(state_last, hidden), nn.ReLU()])
            state_last = hidden
        self.state_encoder = nn.Sequential(*state_layers)
        self.value_head = nn.Linear(state_last, 1)

        action_layers: list[Any] = []
        action_last = self.observation_dim + self.action_dim
        for hidden in sizes:
            action_layers.extend([nn.Linear(action_last, hidden), nn.ReLU()])
            action_last = hidden
        self.action_encoder = nn.Sequential(*action_layers)
        self.delta_head = nn.Linear(action_last, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, observations: Any, joint_actions: Any) -> Any:
        _require_torch()
        device = next(self.parameters()).device
        observations_t = torch.as_tensor(observations, dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(joint_actions, dtype=torch.float32, device=device)
        if observations_t.ndim == 1:
            observations_t = observations_t.unsqueeze(0)
        if actions_t.ndim == 1:
            actions_t = actions_t.unsqueeze(0)
        expected_obs = (observations_t.shape[0], self.observation_dim)
        expected_action = (observations_t.shape[0], self.action_dim)
        if tuple(observations_t.shape) != expected_obs:
            raise ValueError(
                f"observations shape={tuple(observations_t.shape)}，期望 [B,{self.observation_dim}]。"
            )
        if tuple(actions_t.shape) != expected_action:
            raise ValueError(
                f"joint_actions shape={tuple(actions_t.shape)}，期望 [B,{self.action_dim}]。"
            )
        if not torch.all((actions_t == 0) | (actions_t == 1)):
            raise ValueError("joint_actions 必须是0/1。")
        if not torch.all(torch.isfinite(observations_t)):
            raise ValueError("observations 包含 NaN/Inf。")
        state_value = self.value_head(self.state_encoder(observations_t)).squeeze(-1)
        action_delta = self.delta_head(
            self.action_encoder(torch.cat([observations_t, actions_t], dim=-1))
        ).squeeze(-1)
        return state_value + action_delta


def build_centralized_joint_action_critic_preserving_rng(
    observation_dim: int,
    action_dim: int,
    hidden_sizes: tuple[int, ...] = (128, 128),
    device: Any | None = None,
) -> CentralizedJointActionCritic:
    """初始化集中式 critic，同时恢复 Actor 使用的 global torch RNG。

    这是对官方 ``torch.get_rng_state``/``torch.set_rng_state`` API 的轻量封装；
    目的只是让新增 critic 的参数初始化不改变既有 Actor sampling trajectory。
    """

    _require_torch()
    cpu_state = torch.get_rng_state().clone()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
    critic = CentralizedJointActionCritic(observation_dim, action_dim, hidden_sizes)
    if device is not None:
        critic = critic.to(device)
    torch.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    return critic


class ConditionalTriggerActorCritic(nn.Module if nn is not None else object):
    """Shared encoder/AR block + trigger-specific binary heads + scalar critic.

    One head per dynamic trigger is initialized as an exact copy of the historical shared
    ``logit_head`` initialization.  A temporary single Linear is used only to
    preserve the old RNG draw order; it is not registered as a model
    parameter.  The formal V3 sparse Direct-CF optimizer freezes the
    autoregressive block and value head and trains the encoder plus these
    trigger-specific heads.
    """

    architecture_id = TRIGGER_HEADS_ARCHITECTURE_ID
    architecture_version = TRIGGER_HEADS_ARCHITECTURE_VERSION
    uses_shared_logit_head = False

    def __init__(
        self,
        observation_dim: int,
        action_spec: TriggerActionSpec,
        hidden_sizes: tuple[int, ...] = (128, 128),
        autoregressive_hidden_size: int = 64,
        action_embedding_dim: int = 8,
        index_embedding_dim: int = 8,
    ) -> None:
        _require_torch()
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim 必须大于 0。")
        if action_spec.action_dim <= 0:
            raise ValueError("action_dim 必须大于 0。")
        if not hidden_sizes or any(int(size) <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes 必须是正整数列表。")
        if autoregressive_hidden_size <= 0:
            raise ValueError("autoregressive_hidden_size 必须大于 0。")
        self.observation_dim = int(observation_dim)
        self.action_spec = action_spec
        self.action_dim = int(action_spec.action_dim)
        self.autoregressive_hidden_size = int(autoregressive_hidden_size)

        layers: list[Any] = []
        last_dim = self.observation_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_dim, int(hidden)))
            layers.append(nn.ReLU())
            last_dim = int(hidden)
        self.encoder = nn.Sequential(*layers)
        self.initial_hidden = nn.Linear(last_dim, self.autoregressive_hidden_size)
        self.previous_action_embedding = nn.Embedding(3, action_embedding_dim)  # 0/1/START
        self.action_index_embedding = nn.Embedding(self.action_dim, index_embedding_dim)
        self.gru_cell = nn.GRUCell(
            input_size=last_dim + action_embedding_dim + index_embedding_dim,
            hidden_size=self.autoregressive_hidden_size,
        )
        if self.uses_shared_logit_head:
            # This is the historical V3 Full Actor projection.  It consumes
            # exactly one Linear initialization draw before value_head.
            self.logit_head = nn.Linear(self.autoregressive_hidden_size, 1)
        else:
            # Keep the trigger-head initialization draw exactly once, then
            # make one bitwise copy per dynamic action without consuming additional RNG.
            shared_logit_head = nn.Linear(self.autoregressive_hidden_size, 1)
            self.trigger_heads = nn.ModuleList(
                [copy.deepcopy(shared_logit_head) for _ in range(self.action_dim)]
            )
        self.value_head = nn.Linear(last_dim, 1)

    def _as_observation_batch(self, observations: Any) -> tuple[Any, bool]:
        obs = torch.as_tensor(observations, dtype=torch.float32)
        single = False
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
            single = True
        if obs.ndim != 2 or obs.shape[1] != self.observation_dim:
            raise ValueError(
                f"observations shape={tuple(obs.shape)}，期望 [D] 或 [B,{self.observation_dim}]。"
            )
        return obs, single

    def _encode(self, observations: Any) -> tuple[Any, Any]:
        latent = self.encoder(observations)
        value = self.value_head(latent).squeeze(-1)
        return latent, value

    def _logit_head_for_action(self, action_index: int) -> Any:
        if self.uses_shared_logit_head:
            return self.logit_head
        return self.trigger_heads[action_index]

    def _teacher_forced_outputs(
        self,
        observations: Any,
        action_vectors: Any,
    ) -> tuple[Any, Any, Any]:
        obs, _ = self._as_observation_batch(observations)
        actions = torch.as_tensor(action_vectors, dtype=torch.float32, device=obs.device)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        if actions.shape != (obs.shape[0], self.action_dim):
            raise ValueError(f"action_vectors shape={tuple(actions.shape)}，期望 {(obs.shape[0], self.action_dim)}。")
        if not torch.all((actions == 0) | (actions == 1)):
            raise ValueError("action_vectors 必须是 0/1。")

        latent, value = self._encode(obs)
        hidden = torch.tanh(self.initial_hidden(latent))
        previous_action = torch.full(
            (obs.shape[0],),
            2,
            dtype=torch.long,
            device=obs.device,
        )
        logits = []
        conditional_hidden_states = []
        for action_index in range(self.action_dim):
            index_tensor = torch.full(
                (obs.shape[0],),
                action_index,
                dtype=torch.long,
                device=obs.device,
            )
            step_input = torch.cat(
                [
                    latent,
                    self.previous_action_embedding(previous_action),
                    self.action_index_embedding(index_tensor),
                ],
                dim=-1,
            )
            hidden = self.gru_cell(step_input, hidden)
            # 这是读取当前 a_i 之前、已经条件于真实 prefix a_<i 的 hidden。
            conditional_hidden_states.append(hidden)
            logits.append(self._logit_head_for_action(action_index)(hidden).squeeze(-1))
            previous_action = actions[:, action_index].long()
        return (
            torch.stack(logits, dim=1),
            value,
            torch.stack(conditional_hidden_states, dim=1),
        )

    def _teacher_forced_logits(self, observations: Any, action_vectors: Any) -> tuple[Any, Any]:
        logits, value, _ = self._teacher_forced_outputs(observations, action_vectors)
        return logits, value

    def conditional_hidden_states(self, observations: Any, action_vectors: Any) -> Any:
        """返回每个 trigger 当前 action 决策前的 teacher-forced hidden。

        返回形状固定为 ``[B, action_dim, autoregressive_hidden_size]``。
        hidden 在输入 Q critic 前必须由 caller 显式 ``detach``；该 API 本身
        保留计算图，便于 synthetic gradient-isolation 验证发现错误连接。
        """

        _, _, hidden_states = self._teacher_forced_outputs(observations, action_vectors)
        return hidden_states

    def evaluate_actions(
        self,
        observations: Any,
        action_vectors: Any,
        action_masks: Any = None,
    ) -> TriggerEvaluation:
        """Teacher-forced 复算 masked ``log π(a|s)``，原生支持 single/batch。

        ``action_masks`` 必须来自同一个状态和 autoregressive prefix；rollout
        会把采样时的 mask 保存到 ``RolloutBatch``，供 PPO update 原样复用。
        """

        logits, value, _ = self._teacher_forced_outputs(observations, action_vectors)
        actions = torch.as_tensor(action_vectors, dtype=torch.float32, device=logits.device)
        single_action = False
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
            single_action = True
        masks = _normalize_action_masks(
            action_masks,
            int(actions.shape[0]),
            self.action_dim,
            logits.device,
        )
        categorical_logits = torch.stack(
            [torch.zeros_like(logits), logits],
            dim=-1,
        )
        categorical_logits = categorical_logits.masked_fill(~masks, -float("inf"))
        dist = torch.distributions.Categorical(logits=categorical_logits)
        action_indices = actions.long()
        action_is_valid = masks.gather(-1, action_indices.unsqueeze(-1)).squeeze(-1)
        if not bool(torch.all(action_is_valid).item()):
            raise ValueError("action_vectors 包含被 action mask 屏蔽的 binary action。")
        per_action_log_probs = dist.log_prob(action_indices)
        joint_log_prob = per_action_log_probs.sum(dim=1)
        entropy_per_action = dist.entropy()
        entropy_per_action = torch.where(
            masks.sum(dim=-1) == 1,
            torch.zeros_like(entropy_per_action),
            entropy_per_action,
        )
        entropy = entropy_per_action.sum(dim=1)
        probabilities = dist.probs[..., 1]
        probability_table = dist.probs
        if single_action:
            joint_log_prob = joint_log_prob.squeeze(0)
            entropy = entropy.squeeze(0)
            value = value.squeeze(0)
            logits = logits.squeeze(0)
            probabilities = probabilities.squeeze(0)
            probability_table = probability_table.squeeze(0)
            per_action_log_probs = per_action_log_probs.squeeze(0)
            entropy_per_action = entropy_per_action.squeeze(0)
        return TriggerEvaluation(
            joint_log_prob=joint_log_prob,
            entropy=entropy,
            value=value,
            per_action_logits=logits,
            per_action_probabilities=probabilities,
            per_action_probability_table=probability_table,
            per_action_log_probs=per_action_log_probs,
            per_action_entropy=entropy_per_action,
        )

    def sample_action(
        self,
        observation: Any,
        deterministic: bool = False,
        action_mask: Callable[[int, np.ndarray], Any] | Any = None,
        force_action_index: int | None = None,
        force_action_value: int | None = None,
        fixed_action_prefix: Any = None,
    ) -> TriggerActionSample:
        """按固定 action 顺序自回归采样一个 masked trigger vector。

        callable mask 接收 ``(action_index, prefix_actions)``，因此后续 UAV
        trigger 可以依据前面已采样的 trigger 和当前几何约束动态决定。

        Direct-CF branch 可以指定一个目标 action。目标位置仍先执行一次
        stochastic draw，再替换其取值，以保持后续 policy RNG 消费数量与
        baseline 一致；普通调用不传这两个参数，行为完全不变。
        """

        obs, single = self._as_observation_batch(observation)
        if not single or obs.shape[0] != 1:
            raise ValueError("sample_action 当前只接受单个 observation [D]。")
        if (force_action_index is None) != (force_action_value is None):
            raise ValueError("force_action_index 和 force_action_value 必须同时提供或同时省略。")
        if force_action_index is not None:
            force_action_index = int(force_action_index)
            force_action_value = int(force_action_value)
            if not 0 <= force_action_index < self.action_dim:
                raise ValueError("force_action_index 超出 action_dim。")
            if force_action_value not in {0, 1}:
                raise ValueError("force_action_value 必须是0或1。")
        fixed_prefix = np.asarray(
            [] if fixed_action_prefix is None else fixed_action_prefix,
            dtype=np.int8,
        ).reshape(-1)
        if not np.all(np.isin(fixed_prefix, [0, 1])):
            raise ValueError("fixed_action_prefix 必须是0/1向量。")
        if force_action_index is not None and len(fixed_prefix) > force_action_index:
            raise ValueError("fixed_action_prefix 不能覆盖 force_action_index。")
        latent, value = self._encode(obs)
        hidden = torch.tanh(self.initial_hidden(latent))
        previous_action = torch.full((1,), 2, dtype=torch.long, device=obs.device)

        actions = []
        logits = []
        probs = []
        probability_tables = []
        log_probs = []
        entropies = []
        masks = []
        for action_index in range(self.action_dim):
            index_tensor = torch.full((1,), action_index, dtype=torch.long, device=obs.device)
            step_input = torch.cat(
                [
                    latent,
                    self.previous_action_embedding(previous_action),
                    self.action_index_embedding(index_tensor),
                ],
                dim=-1,
            )
            hidden = self.gru_cell(step_input, hidden)
            logit = self._logit_head_for_action(action_index)(hidden).squeeze(-1)
            mask = _resolve_sampling_mask(
                action_mask,
                action_index,
                np.asarray(
                    [int(item.detach().cpu().item()) for item in actions],
                    dtype=np.int8,
                ),
                self.action_dim,
            )
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=obs.device)
            categorical_logits = torch.stack(
                [torch.zeros_like(logit), logit],
                dim=-1,
            ).masked_fill(~mask_tensor, -float("inf"))
            dist = torch.distributions.Categorical(logits=categorical_logits)
            if deterministic:
                action = torch.argmax(dist.probs, dim=-1).to(torch.float32)
            else:
                # 所有非 deterministic 决策都执行一次 draw。Direct-CF 的固定
                # prefix/forced target 只替换 draw 的结果，不改变 RNG 消费数量。
                sampled_action = dist.sample().to(torch.float32)
                if action_index < len(fixed_prefix):
                    fixed_value = int(fixed_prefix[action_index])
                    if not bool(mask_tensor[fixed_value].item()):
                        raise ValueError("Direct-CF fixed prefix 被当前 action mask 屏蔽。")
                    action = torch.full_like(sampled_action, float(fixed_value))
                elif force_action_index is not None and action_index == force_action_index:
                    if not bool(mask_tensor[force_action_value].item()):
                        raise ValueError("Direct-CF force action 被当前 action mask 屏蔽。")
                    action = torch.full_like(sampled_action, float(force_action_value))
                else:
                    action = sampled_action
            actions.append(action)
            logits.append(logit)
            probs.append(dist.probs[..., 1])
            probability_tables.append(dist.probs)
            log_probs.append(dist.log_prob(action.long()))
            entropy = dist.entropy()
            entropies.append(torch.where(mask_tensor.sum() == 1, torch.zeros_like(entropy), entropy))
            masks.append(mask)
            previous_action = action.long()

        action_tensor = torch.stack(actions, dim=1).to(torch.int64).squeeze(0)
        logit_tensor = torch.stack(logits, dim=1).squeeze(0)
        prob_tensor = torch.stack(probs, dim=1).squeeze(0)
        probability_table_tensor = torch.stack(probability_tables, dim=1).squeeze(0)
        log_prob_tensor = torch.stack(log_probs, dim=1).squeeze(0)
        action_vector = action_tensor.detach().cpu().numpy().astype(np.int8)
        cache_triggers, uav_triggers = self.action_spec.action_vector_to_triggers(action_vector)
        return TriggerActionSample(
            action_vector=action_vector,
            cache_triggers=cache_triggers,
            uav_triggers=uav_triggers,
            joint_log_prob=float(torch.stack(log_probs).sum().detach().cpu()),
            conditional_entropy=float(torch.stack(entropies).sum().detach().cpu()),
            value=float(value.squeeze(0).detach().cpu()),
            per_action_probabilities=prob_tensor.detach().cpu().numpy().astype(np.float64),
            per_action_probability_table=probability_table_tensor.detach().cpu().numpy().astype(np.float64),
            per_action_log_probs=log_prob_tensor.detach().cpu().numpy().astype(np.float64),
            per_action_logits=logit_tensor.detach().cpu().numpy().astype(np.float64),
            per_action_masks=np.asarray(masks, dtype=bool),
        )


class FullSharedActorCritic(ConditionalTriggerActorCritic):
    """V3 Full Shared Actor used by frozen-policy multi-episode batching.

    The historical shared encoder, autoregressive block and single
    ``logit_head`` are active.  ``value_head`` remains part of the Actor
    object for compatibility but is excluded from the Direct-CF optimizer.
    The trigger-head class above remains available for historical checkpoint
    loading and read-only diagnostics; it is not this candidate's Actor.
    """

    architecture_id = FULL_SHARED_ACTOR_ARCHITECTURE_ID
    architecture_version = FULL_SHARED_ACTOR_ARCHITECTURE_VERSION
    uses_shared_logit_head = True


def actor_architecture_id(policy: Any) -> str:
    """Return the explicit architecture identifier used by checkpoint metadata."""

    value = getattr(policy, "architecture_id", None)
    if value is None:
        return LEGACY_SHARED_HEAD_ARCHITECTURE_ID
    return str(value)


def actor_architecture_metadata(policy: Any) -> dict[str, Any]:
    """Describe the formal Actor variant without inspecting optimizer state."""

    architecture = actor_architecture_id(policy)
    if architecture == FULL_SHARED_ACTOR_ARCHITECTURE_ID:
        trainable_groups = ["encoder", "autoregressive_block", "logit_head"]
        frozen_groups = ["value_head"]
    elif architecture == TRIGGER_HEADS_ARCHITECTURE_ID:
        trainable_groups = ["encoder", "trigger_heads"]
        frozen_groups = ["autoregressive_block", "value_head"]
    else:
        trainable_groups = ["encoder", "autoregressive_block", "logit_head"]
        frozen_groups = ["value_head"]
    trigger_heads = getattr(policy, "trigger_heads", None)
    trigger_head_count = 0 if trigger_heads is None else int(len(trigger_heads))
    return {
        "architecture_id": architecture,
        "architecture_version": int(getattr(policy, "architecture_version", 0)),
        "class": type(policy).__name__,
        "action_dim": int(getattr(policy, "action_dim")),
        "trigger_head_count": trigger_head_count,
        "trainable_sparse_direct_cf_groups": trainable_groups,
        "frozen_sparse_direct_cf_groups": frozen_groups,
    }


def configure_sparse_direct_cf_actor_trainability(policy: Any) -> None:
    """Apply the architecture-specific formal V3 Direct-CF freeze contract."""

    architecture = actor_architecture_id(policy)
    if architecture not in {
        TRIGGER_HEADS_ARCHITECTURE_ID,
        FULL_SHARED_ACTOR_ARCHITECTURE_ID,
    }:
        raise RuntimeError(
            "sparse_direct_cf正式Actor architecture 不受支持："
            f"{architecture}。"
        )
    if architecture == TRIGGER_HEADS_ARCHITECTURE_ID:
        trainable_prefixes = ("encoder.", "trigger_heads.")
    else:
        trainable_prefixes = (
            "encoder.",
            "initial_hidden.",
            "previous_action_embedding.",
            "action_index_embedding.",
            "gru_cell.",
            "logit_head.",
        )
    for name, parameter in policy.named_parameters():
        trainable = name.startswith(trainable_prefixes)
        parameter.requires_grad_(trainable)
    trainable_names = [
        name
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_names or not all(name.startswith(trainable_prefixes) for name in trainable_names):
        raise AssertionError("V3 sparse_direct_cf optimizer trainability范围不符合当前Actor architecture。")


def sparse_direct_cf_actor_parameters(policy: Any) -> list[Any]:
    """Return only formal V3 trainable Actor parameters after freeze setup."""

    configure_sparse_direct_cf_actor_trainability(policy)
    architecture = actor_architecture_id(policy)
    if architecture == TRIGGER_HEADS_ARCHITECTURE_ID:
        trainable_prefixes = ("encoder.", "trigger_heads.")
    elif architecture == FULL_SHARED_ACTOR_ARCHITECTURE_ID:
        trainable_prefixes = (
            "encoder.",
            "initial_hidden.",
            "previous_action_embedding.",
            "action_index_embedding.",
            "gru_cell.",
            "logit_head.",
        )
    else:
        raise RuntimeError(f"未知 sparse_direct_cf Actor architecture：{architecture}")
    parameters = [
        parameter for name, parameter in policy.named_parameters()
        if name.startswith(trainable_prefixes)
    ]
    if not parameters:
        raise RuntimeError("V3 sparse_direct_cf Actor optimizer参数为空。")
    return parameters


def load_legacy_shared_head_state_dict(
    policy: Any,
    legacy_state_dict: dict[str, Any],
    *,
    strict: bool = True,
) -> Any:
    """Explicitly map a historical shared-head state into all new heads.

    This helper is for read-only historical analysis.  Formal checkpoint
    loading still requires matching architecture metadata and therefore will
    not silently resume a legacy shared-head run.
    """

    if actor_architecture_id(policy) != TRIGGER_HEADS_ARCHITECTURE_ID:
        raise RuntimeError("legacy shared-head mapping的目标policy不是trigger-head Actor。")
    source_weight = legacy_state_dict.get("logit_head.weight")
    source_bias = legacy_state_dict.get("logit_head.bias")
    if source_weight is None or source_bias is None:
        raise KeyError("legacy shared-head state_dict缺少logit_head.weight/bias。")
    mapped = {
        key: value
        for key, value in legacy_state_dict.items()
        if key not in {"logit_head.weight", "logit_head.bias"}
    }
    for index in range(int(policy.action_dim)):
        mapped[f"trigger_heads.{index}.weight"] = copy.deepcopy(source_weight)
        mapped[f"trigger_heads.{index}.bias"] = copy.deepcopy(source_bias)
    return policy.load_state_dict(mapped, strict=strict)
