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


class ConditionalTriggerActorCritic(nn.Module if nn is not None else object):
    """Shared MLP encoder + autoregressive GRUCell binary actor + scalar critic."""

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
        self.logit_head = nn.Linear(self.autoregressive_hidden_size, 1)
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
            logits.append(self.logit_head(hidden).squeeze(-1))
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
    ) -> TriggerActionSample:
        """按固定 action 顺序自回归采样一个 masked trigger vector。

        callable mask 接收 ``(action_index, prefix_actions)``，因此后续 UAV
        trigger 可以依据前面已采样的 trigger 和当前几何约束动态决定。
        """

        obs, single = self._as_observation_batch(observation)
        if not single or obs.shape[0] != 1:
            raise ValueError("sample_action 当前只接受单个 observation [D]。")
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
            logit = self.logit_head(hidden).squeeze(-1)
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
                action = dist.sample().to(torch.float32)
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
