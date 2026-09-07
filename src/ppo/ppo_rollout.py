#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""On-policy rollout collection and GAE for trigger-only PPO.

本模块只实现：

    s_t -> sample Z_t -> execute_slot -> G(t) -> next C/Q -> GAE

不实现 PPO loss、optimizer.step、minibatch update、checkpoint 或完整训练循环。
核心 collector 不读磁盘；磁盘 artifact 读取只放在薄 helper/CLI 中。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common import project_config
from decision import cache_configuration, task_allocation, slot_environment, uav_configuration
from ppo import ppo_trigger_policy
from scenario import task_scenario

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _sample_action_without_grad(policy: Any, *args: Any, **kwargs: Any) -> Any:
    """Call rollout sampling in inference mode without changing RNG semantics."""

    if torch is None:
        return policy.sample_action(*args, **kwargs)
    with torch.no_grad():
        return policy.sample_action(*args, **kwargs)


def _is_hierarchical_policy(policy: Any) -> bool:
    """判断当前策略是否为正式的分层 Gate + WHAT 策略。"""

    from ppo import hierarchical_reconfiguration_gate

    return isinstance(
        policy,
        hierarchical_reconfiguration_gate.HierarchicalReconfigurationGatePolicy,
    )


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


@dataclass(frozen=True)
class RolloutTransition:
    slot: int
    time_s: float
    origin_index: int
    observation: np.ndarray
    action_vector: np.ndarray
    cache_triggers: np.ndarray
    uav_triggers: dict[str, int]
    old_log_prob: float
    old_per_action_log_probs: np.ndarray
    old_per_action_probability_table: np.ndarray
    old_value: float
    reward: float
    done: bool
    step_feasible: bool
    infeasible_reason: str
    actual_task_utility: float
    cache_cost: float
    movement_cost: float
    next_cache_state: np.ndarray
    next_uav_positions: dict[str, tuple[float, float]]
    # Hierarchical WHEN-policy fields.  They are optional so historical
    # trigger-only rollout fixtures keep their exact schema/constructor.
    gate_action: int | None = None
    old_gate_probability: float | None = None
    old_gate_log_prob: float | None = None
    subpolicy_action_vector: np.ndarray | None = None


@dataclass(frozen=True)
class RolloutBatch:
    observations: np.ndarray
    actions: np.ndarray
    action_masks: np.ndarray
    old_log_probs: np.ndarray
    old_per_action_log_probs: np.ndarray
    old_per_action_probabilities: np.ndarray
    actor_valid_mask: np.ndarray
    old_values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    monte_carlo_returns: np.ndarray
    slots: np.ndarray
    origin_indices: np.ndarray
    final_bootstrap_value: float
    truncated: bool
    transitions: list[RolloutTransition]
    reference_decisions: tuple["ReferenceStateRecord", ...]
    metadata: dict[str, Any]
    # Present only for the hierarchical Gate formal path.
    gate_actions: np.ndarray | None = None
    old_gate_log_probs: np.ndarray | None = None
    old_gate_probabilities: np.ndarray | None = None
    subpolicy_actions: np.ndarray | None = None


@dataclass(frozen=True)
class ReferenceStateRecord:
    """最小的 frozen-policy conditional reference state。

    不保存 environment object、cache/UAV solver state 或完整 transition；这些
    字段足以在同一个 Actor 上重算 masked conditional p(action=1)。
    """

    decision_id: str
    episode_index: int
    step_index: int
    slot: int
    origin_index: int
    trigger_index: int
    trigger_label: str
    observation: np.ndarray
    action_prefix: np.ndarray
    action_vector: np.ndarray
    action_mask: np.ndarray
    p_old: float
    collection_policy_version: int
    selected_for_direct_cf: bool = False


@dataclass(frozen=True)
class DirectCFDecisionSnapshot:
    """reservoir 中保存的单个 FREE decision pre-action snapshot。"""

    step_index: int
    slot: int
    origin_index: int
    trigger_index: int
    trigger_label: str
    decision_id: str
    previous_cache_state: np.ndarray
    previous_uav_positions: dict[str, tuple[float, float]]
    observation: np.ndarray
    action_vector: np.ndarray
    action_prefix: np.ndarray
    old_p1: float
    old_chosen_log_prob: float
    action_mask: np.ndarray
    baseline_rng_before: dict[str, Any]
    baseline_rng_after_slot: dict[str, Any]


@dataclass(frozen=True)
class DirectCFCreditSample:
    """一次 paired branch 完成后供 V3 Actor 使用的轻量样本。"""

    slot: int
    origin_index: int
    trigger_index: int
    trigger_label: str
    decision_id: str
    observation: np.ndarray
    action_vector: np.ndarray
    action_prefix: np.ndarray
    action_mask: np.ndarray
    actual_action: int
    old_p1: float
    old_chosen_log_prob: float
    delta_g: float
    delta_q: float
    delta_cache: float
    delta_uav: float
    a_cf: float
    baseline_suffix_g: float
    branch_suffix_g: float
    branch_environment_steps: int
    crn_target_step_rng_aligned: bool
    # 分层采样的无偏估计权重；uniform模式为每个已选样本的1/K。
    sampling_weight: float = 0.0
    trigger_free_count: int = 0
    # Frozen-policy batch runtime guard；旧单episode样本默认属于version 0。
    collection_policy_version: int = 0
    # Persisted causal artifacts need the originating formal episode explicitly.
    # The default preserves compatibility with older recovered sample fixtures.
    episode_index: int = 0
    # Optional recovery-artifact field.  Formal collection fills this with the
    # selected-sample ordinal; older raw fixtures may omit it and load as 0.
    sample_index_in_episode: int = 0
    # Direct-CF branch origin within the aligned episode; older fixtures may
    # omit it because origin_index/slot were historically sufficient for loss.
    step_index: int = 0


@dataclass(frozen=True)
class GateDecisionSnapshot:
    """Pre-slot snapshot used by the uniform Gate-CF slot reservoir."""

    step_index: int
    slot: int
    origin_index: int
    decision_id: str
    previous_cache_state: np.ndarray
    previous_uav_positions: dict[str, tuple[float, float]]
    observation: np.ndarray
    gate_action: int
    old_gate_probability: float
    old_gate_log_prob: float
    final_action_vector: np.ndarray
    baseline_rng_before: dict[str, Any]
    baseline_rng_after_slot: dict[str, Any]


@dataclass(frozen=True)
class _CFSuffixBaselineTransition:
    """Pickle-minimal baseline metrics needed by a CF suffix worker."""

    reward: float
    actual_task_utility: float
    cache_cost: float
    movement_cost: float


@dataclass(frozen=True)
class _CFSuffixBaselineBatch:
    """Pickle-minimal baseline batch facade used by existing CF helpers."""

    transitions: tuple[_CFSuffixBaselineTransition, ...]


# Windows ``spawn`` workers receive a compact, read-only context through the
# initializer.  The live training policy and parent RNG are never shared.
_CF_WORKER_STATE: dict[str, Any] | None = None


def _record_cf_execution_status(
    status: dict[str, Any] | None,
    *,
    mode: str,
    workers: int,
) -> None:
    if status is not None:
        status.clear()
        status.update({"mode": str(mode), "workers": int(workers)})


class GateCFReservoir:
    """Uniform sparse reservoir over legal formal slots.

    Every formal slot is legal for the Gate decision: NULL is represented by
    the strict all-zero action and CONFIG is sampled from the frozen WHAT
    policy under that slot's mask.  The independent NumPy generator is saved
    by the formal checkpoint so resume does not alter slot selection.
    """

    sampling_strategy = "uniform_slots"

    def __init__(self, capacity: int, sampling_rng: Any) -> None:
        if isinstance(capacity, bool) or int(capacity) <= 0:
            raise ValueError("GateCFReservoir capacity 必须是正整数。")
        if not isinstance(sampling_rng, np.random.Generator):
            raise TypeError("GateCFReservoir 必须使用独立 np.random.Generator。")
        self.capacity = int(capacity)
        self.sampling_rng = sampling_rng
        self.items: list[GateDecisionSnapshot] = []
        self.total_seen = 0

    def consider(self, snapshot: GateDecisionSnapshot) -> None:
        if not isinstance(snapshot, GateDecisionSnapshot):
            raise TypeError("GateCFReservoir 只能接收 GateDecisionSnapshot。")
        self.total_seen += 1
        if len(self.items) < self.capacity:
            self.items.append(snapshot)
            return
        replacement_index = int(self.sampling_rng.integers(0, self.total_seen))
        if replacement_index < self.capacity:
            self.items[replacement_index] = snapshot

    @property
    def sample_count(self) -> int:
        return len(self.items)

    def rng_state(self) -> dict[str, Any]:
        return copy.deepcopy(self.sampling_rng.bit_generator.state)


def reference_state_to_checkpoint_dict(record: ReferenceStateRecord) -> dict[str, Any]:
    """Serialize only the conditional-input fields needed by refdrift resume."""

    return {
        "decision_id": str(record.decision_id),
        "episode_index": int(record.episode_index),
        "step_index": int(record.step_index),
        "slot": int(record.slot),
        "origin_index": int(record.origin_index),
        "trigger_index": int(record.trigger_index),
        "trigger_label": str(record.trigger_label),
        "observation": np.asarray(record.observation, dtype=np.float32).tolist(),
        "action_prefix": np.asarray(record.action_prefix, dtype=np.int8).tolist(),
        "action_vector": np.asarray(record.action_vector, dtype=np.int8).tolist(),
        "action_mask": np.asarray(record.action_mask, dtype=bool).tolist(),
        "p_old": float(record.p_old),
        "collection_policy_version": int(record.collection_policy_version),
        "selected_for_direct_cf": bool(record.selected_for_direct_cf),
    }


def reference_state_from_checkpoint_dict(value: dict[str, Any]) -> ReferenceStateRecord:
    """Restore one compact reference state with strict shape/value checks."""

    observation = np.asarray(value["observation"], dtype=np.float32)
    action_prefix = np.asarray(value["action_prefix"], dtype=np.int8)
    action_vector = np.asarray(value["action_vector"], dtype=np.int8)
    action_mask = np.asarray(value["action_mask"], dtype=bool)
    if observation.ndim != 1 or action_vector.ndim != 1 or action_prefix.ndim != 1:
        raise ValueError("reference checkpoint observation/action必须是一维。")
    if action_mask.ndim != 2 or action_mask.shape != (action_vector.shape[0], 2):
        raise ValueError("reference checkpoint action_mask shape非法。")
    if action_prefix.shape[0] > action_vector.shape[0]:
        raise ValueError("reference checkpoint action_prefix长度非法。")
    if not np.all(np.isin(action_vector, [0, 1])) or not np.all(np.isin(action_prefix, [0, 1])):
        raise ValueError("reference checkpoint action必须是0/1。")
    p_old = float(value["p_old"])
    if not math.isfinite(p_old) or not 0.0 <= p_old <= 1.0:
        raise ValueError("reference checkpoint p_old非法。")
    return ReferenceStateRecord(
        decision_id=str(value["decision_id"]),
        episode_index=int(value["episode_index"]),
        step_index=int(value["step_index"]),
        slot=int(value["slot"]),
        origin_index=int(value["origin_index"]),
        trigger_index=int(value["trigger_index"]),
        trigger_label=str(value["trigger_label"]),
        observation=observation.copy(),
        action_prefix=action_prefix.copy(),
        action_vector=action_vector.copy(),
        action_mask=action_mask.copy(),
        p_old=p_old,
        collection_policy_version=int(value["collection_policy_version"]),
        selected_for_direct_cf=bool(value.get("selected_for_direct_cf", False)),
    )


def direct_cf_sample_to_checkpoint_dict(sample: DirectCFCreditSample) -> dict[str, Any]:
    """Serialize a pending Direct-CF sample without environment state."""

    return {
        "episode": int(sample.episode_index),
        "episode_index": int(sample.episode_index),
        "sample_index_in_episode": int(sample.sample_index_in_episode),
        "step_index": int(sample.step_index),
        "slot": int(sample.slot),
        "origin_index": int(sample.origin_index),
        "trigger_index": int(sample.trigger_index),
        "trigger_label": str(sample.trigger_label),
        "decision_id": str(sample.decision_id),
        "observation": np.asarray(sample.observation, dtype=np.float32).tolist(),
        "action_vector": np.asarray(sample.action_vector, dtype=np.int8).tolist(),
        "action_prefix": np.asarray(sample.action_prefix, dtype=np.int8).tolist(),
        "action_mask": np.asarray(sample.action_mask, dtype=bool).tolist(),
        "actual_action": int(sample.actual_action),
        "old_p1": float(sample.old_p1),
        "old_chosen_log_prob": float(sample.old_chosen_log_prob),
        "delta_g": float(sample.delta_g),
        "delta_q": float(sample.delta_q),
        "delta_cache": float(sample.delta_cache),
        "delta_uav": float(sample.delta_uav),
        "a_cf": float(sample.a_cf),
        "baseline_suffix_g": float(sample.baseline_suffix_g),
        "branch_suffix_g": float(sample.branch_suffix_g),
        "branch_environment_steps": int(sample.branch_environment_steps),
        "crn_target_step_rng_aligned": bool(sample.crn_target_step_rng_aligned),
        "sampling_weight": float(sample.sampling_weight),
        "trigger_free_count": int(sample.trigger_free_count),
        "collection_policy_version": int(sample.collection_policy_version),
    }


def direct_cf_sample_from_checkpoint_dict(value: dict[str, Any]) -> DirectCFCreditSample:
    """Restore one pending Direct-CF sample with finite scalar validation."""

    float_fields = (
        "old_p1",
        "old_chosen_log_prob",
        "delta_g",
        "delta_q",
        "delta_cache",
        "delta_uav",
        "a_cf",
        "baseline_suffix_g",
        "branch_suffix_g",
        "sampling_weight",
    )
    scalars = {name: float(value[name]) for name in float_fields}
    if not all(math.isfinite(item) for item in scalars.values()):
        raise ValueError("pending Direct-CF sample包含NaN/Inf。")
    observation = np.asarray(value["observation"], dtype=np.float32)
    action_vector = np.asarray(value["action_vector"], dtype=np.int8)
    action_prefix = np.asarray(value["action_prefix"], dtype=np.int8)
    action_mask = np.asarray(value["action_mask"], dtype=bool)
    if observation.ndim != 1 or action_vector.ndim != 1 or action_prefix.ndim != 1:
        raise ValueError("pending Direct-CF sample observation/action必须是一维。")
    if action_mask.ndim != 2 or action_mask.shape != (action_vector.shape[0], 2):
        raise ValueError("pending Direct-CF sample action_mask shape非法。")
    if action_prefix.shape[0] > action_vector.shape[0]:
        raise ValueError("pending Direct-CF sample action_prefix长度非法。")
    if not np.all(np.isin(action_vector, [0, 1])) or not np.all(np.isin(action_prefix, [0, 1])):
        raise ValueError("pending Direct-CF sample action必须是0/1。")
    trigger_index = int(value["trigger_index"])
    if trigger_index < 0 or trigger_index >= action_vector.shape[0]:
        raise ValueError("pending Direct-CF sample trigger_index非法。")
    actual_action = int(value["actual_action"])
    if actual_action not in {0, 1} or actual_action != int(action_vector[trigger_index]):
        raise ValueError("pending Direct-CF sample actual_action与action_vector不一致。")
    episode_index = int(value.get("episode_index", value.get("episode", 0)))
    if episode_index < 0:
        raise ValueError("pending Direct-CF sample episode_index不能为负。")
    branch_steps = int(value["branch_environment_steps"])
    if branch_steps < 0:
        raise ValueError("pending Direct-CF sample branch_environment_steps不能为负。")
    sampling_weight = scalars["sampling_weight"]
    trigger_free_count = int(value["trigger_free_count"])
    collection_policy_version = int(value["collection_policy_version"])
    if sampling_weight < 0.0 or trigger_free_count < 0 or collection_policy_version < 0:
        raise ValueError("pending Direct-CF sample count/version/weight非法。")
    decision_id = value.get("decision_id")
    if decision_id is None:
        # The first recovered Update2 raw-sample fixture predates persisted
        # decision IDs.  Reconstruct the exact ID format used by collection;
        # this is metadata only and does not enter the Direct-CF loss.
        decision_id = (
            f"policy{collection_policy_version}:episode{episode_index}:"
            f"origin{int(value['origin_index'])}:slot{int(value['slot'])}:"
            f"trigger{trigger_index}"
        )
    sample_index_in_episode = int(value.get("sample_index_in_episode", 0))
    if sample_index_in_episode < 0:
        raise ValueError("pending Direct-CF sample sample_index_in_episode不能为负。")
    step_index = int(value.get("step_index", 0))
    if step_index < 0:
        raise ValueError("pending Direct-CF sample step_index不能为负。")
    return DirectCFCreditSample(
        episode_index=episode_index,
        sample_index_in_episode=sample_index_in_episode,
        step_index=step_index,
        slot=int(value["slot"]),
        origin_index=int(value["origin_index"]),
        trigger_index=trigger_index,
        trigger_label=str(value["trigger_label"]),
        decision_id=str(decision_id),
        observation=observation.copy(),
        action_vector=action_vector.copy(),
        action_prefix=action_prefix.copy(),
        action_mask=action_mask.copy(),
        actual_action=actual_action,
        old_p1=scalars["old_p1"],
        old_chosen_log_prob=scalars["old_chosen_log_prob"],
        delta_g=scalars["delta_g"],
        delta_q=scalars["delta_q"],
        delta_cache=scalars["delta_cache"],
        delta_uav=scalars["delta_uav"],
        a_cf=scalars["a_cf"],
        baseline_suffix_g=scalars["baseline_suffix_g"],
        branch_suffix_g=scalars["branch_suffix_g"],
        branch_environment_steps=branch_steps,
        crn_target_step_rng_aligned=bool(value["crn_target_step_rng_aligned"]),
        sampling_weight=sampling_weight,
        trigger_free_count=trigger_free_count,
        collection_policy_version=collection_policy_version,
    )


class DirectCFReservoir:
    """使用独立 Generator 对所有 FREE decisions 做均匀 reservoir sampling。"""

    sampling_strategy = "uniform"

    def __init__(
        self,
        capacity: int,
        sampling_rng: Any,
        trigger_count: int = 8,
        trigger_labels: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        if isinstance(capacity, bool) or int(capacity) != capacity or int(capacity) <= 0:
            raise ValueError("DirectCFReservoir capacity 必须是正整数。")
        if not isinstance(sampling_rng, np.random.Generator):
            raise TypeError("DirectCFReservoir 必须使用独立 np.random.Generator。")
        self.capacity = int(capacity)
        self.sampling_rng = sampling_rng
        self.items: list[DirectCFDecisionSnapshot] = []
        self.total_seen = 0
        self.trigger_count = int(trigger_count)
        if self.trigger_count <= 0:
            raise ValueError("DirectCFReservoir trigger_count 必须是正整数。")
        self.trigger_labels = tuple(
            str(label) for label in (trigger_labels or tuple(str(i) for i in range(self.trigger_count)))
        )
        if len(self.trigger_labels) != self.trigger_count:
            raise ValueError("DirectCFReservoir trigger_labels长度必须等于trigger_count。")
        self._trigger_free_counts = [0 for _ in range(self.trigger_count)]

    def consider(self, snapshot: DirectCFDecisionSnapshot) -> None:
        self.total_seen += 1
        trigger_index = int(snapshot.trigger_index)
        if not 0 <= trigger_index < self.trigger_count:
            raise ValueError("DirectCF snapshot trigger_index超出reservoir范围。")
        self._trigger_free_counts[trigger_index] += 1
        if len(self.items) < self.capacity:
            self.items.append(snapshot)
            return
        replacement_index = int(self.sampling_rng.integers(0, self.total_seen))
        if replacement_index < self.capacity:
            self.items[replacement_index] = snapshot

    @property
    def sample_count(self) -> int:
        return len(self.items)

    @property
    def trigger_free_counts(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self._trigger_free_counts)

    @property
    def trigger_sample_counts(self) -> tuple[int, ...]:
        counts = [0 for _ in range(self.trigger_count)]
        for item in self.items:
            counts[int(item.trigger_index)] += 1
        return tuple(counts)

    @property
    def trigger_sampling_weights(self) -> tuple[float, ...]:
        if self.total_seen <= 0:
            return tuple(0.0 for _ in range(self.trigger_count))
        return tuple(float(value / self.total_seen) for value in self._trigger_free_counts)

    def sampling_weight_for(self, trigger_index: int) -> float:
        """返回单个已选样本用于Actor估计的权重。"""

        if self.sample_count <= 0:
            raise ValueError("DirectCF reservoir为空，无法计算sampling weight。")
        if not 0 <= int(trigger_index) < self.trigger_count:
            raise ValueError("trigger_index超出reservoir范围。")
        # uniform旧路径保持原来的K样本 arithmetic mean。
        return float(1.0 / self.sample_count)

    def rng_state(self) -> dict[str, Any]:
        return copy.deepcopy(self.sampling_rng.bit_generator.state)


class TriggerStratifiedDirectCFReservoir(DirectCFReservoir):
    """每个Trigger独立容量为1的均匀分层reservoir。

    对于每个Trigger的N_i个FREE decision，使用独立Generator执行容量1
    reservoir sampling，因此最终样本在该Trigger内部严格等概率；不同Trigger
    之间不做replacement，也不从其他stratum补样本。
    """

    sampling_strategy = "trigger_stratified"

    def __init__(self, trigger_labels: tuple[str, ...] | list[str], sampling_rng: Any) -> None:
        labels = tuple(str(label) for label in trigger_labels)
        if not labels:
            raise ValueError("TriggerStratifiedDirectCFReservoir 至少需要一个Trigger。")
        super().__init__(
            len(labels),
            sampling_rng,
            trigger_count=len(labels),
            trigger_labels=labels,
        )
        self._items_by_trigger: list[DirectCFDecisionSnapshot | None] = [
            None for _ in labels
        ]
        self._trigger_seen = [0 for _ in labels]
        self.items = []

    def consider(self, snapshot: DirectCFDecisionSnapshot) -> None:
        trigger_index = int(snapshot.trigger_index)
        if not 0 <= trigger_index < self.trigger_count:
            raise ValueError("DirectCF snapshot trigger_index超出分层reservoir范围。")
        self.total_seen += 1
        self._trigger_seen[trigger_index] += 1
        self._trigger_free_counts[trigger_index] += 1
        seen = self._trigger_seen[trigger_index]
        current = self._items_by_trigger[trigger_index]
        if current is None or int(self.sampling_rng.integers(0, seen)) == 0:
            self._items_by_trigger[trigger_index] = snapshot
        self.items = [item for item in self._items_by_trigger if item is not None]

    @property
    def sample_count(self) -> int:
        return int(sum(item is not None for item in self._items_by_trigger))

    @property
    def trigger_sample_counts(self) -> tuple[int, ...]:
        return tuple(int(item is not None) for item in self._items_by_trigger)

    def sampling_weight_for(self, trigger_index: int) -> float:
        if not 0 <= int(trigger_index) < self.trigger_count:
            raise ValueError("trigger_index超出分层reservoir范围。")
        if self.total_seen <= 0:
            raise ValueError("分层reservoir为空，无法计算sampling weight。")
        return float(self._trigger_free_counts[int(trigger_index)] / self.total_seen)


def capture_direct_cf_rng_state() -> dict[str, Any]:
    """捕获 branch CRN 所需的全局 Python/NumPy/Torch RNG 状态。"""

    state: dict[str, Any] = {
        "python_random_state": copy.deepcopy(random.getstate()),
        "numpy_random_state": copy.deepcopy(np.random.get_state()),
        "torch_cpu_rng_state": None,
        "torch_cuda_rng_state_all": None,
    }
    if torch is not None:
        state["torch_cpu_rng_state"] = torch.get_rng_state().clone()
        if torch.cuda.is_available():
            state["torch_cuda_rng_state_all"] = [
                item.clone() for item in torch.cuda.get_rng_state_all()
            ]
    return state


def restore_direct_cf_rng_state(state: dict[str, Any]) -> None:
    """恢复 Direct-CF branch 的全局 RNG 状态。"""

    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    if torch is not None and state.get("torch_cpu_rng_state") is not None:
        torch.set_rng_state(state["torch_cpu_rng_state"])
        cuda_states = state.get("torch_cuda_rng_state_all")
        if cuda_states is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("Direct-CF RNG state需要CUDA，但当前CUDA不可用。")
            torch.cuda.set_rng_state_all(cuda_states)


def direct_cf_rng_states_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """比较 branch 与 baseline 的 RNG 状态，供 CRN 诊断使用。"""

    if left["python_random_state"] != right["python_random_state"]:
        return False
    left_np = left["numpy_random_state"]
    right_np = right["numpy_random_state"]
    if left_np[0] != right_np[0] or not np.array_equal(left_np[1], right_np[1]):
        return False
    if tuple(left_np[2:]) != tuple(right_np[2:]):
        return False
    if torch is None:
        return True
    left_torch = left.get("torch_cpu_rng_state")
    right_torch = right.get("torch_cpu_rng_state")
    if left_torch is None or right_torch is None:
        return left_torch is right_torch
    if not bool(torch.equal(left_torch, right_torch)):
        return False
    left_cuda = left.get("torch_cuda_rng_state_all")
    right_cuda = right.get("torch_cuda_rng_state_all")
    if left_cuda is None or right_cuda is None:
        return left_cuda is right_cuda
    return len(left_cuda) == len(right_cuda) and all(
        bool(torch.equal(a, b)) for a, b in zip(left_cuda, right_cuda)
    )


def _normalize_cf_parallel_workers(value: Any) -> int:
    """Validate the CPU CF worker count without changing config semantics."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("cf_parallel_workers 必须是正整数。")
    workers = int(value)
    if workers <= 0:
        raise ValueError("cf_parallel_workers 必须是正整数。")
    return workers


def _cf_policy_worker_spec(policy: Any) -> dict[str, Any] | None:
    """Build a minimal CPU policy spec for a Windows spawn worker."""

    if torch is None or not hasattr(policy, "state_dict") or not hasattr(policy, "parameters"):
        return None
    parameters = list(policy.parameters())
    if not parameters or any(parameter.device.type != "cpu" for parameter in parameters):
        # The formal project is CPU-only.  Do not silently move a live GPU
        # policy and change its numerical execution path.
        return None
    what_policy = getattr(policy, "sub_policy", policy)
    encoder = getattr(what_policy, "encoder", None)
    hidden_sizes = tuple(
        int(layer.out_features)
        for layer in ([] if encoder is None else encoder)
        if isinstance(layer, torch.nn.Linear)
    )
    if not hidden_sizes:
        return None
    state_dict = {
        str(name): value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }
    spec: dict[str, Any] = {
        "observation_dim": int(getattr(policy, "observation_dim")),
        "action_spec": getattr(policy, "action_spec"),
        "hidden_sizes": hidden_sizes,
        "autoregressive_hidden_size": int(getattr(what_policy, "autoregressive_hidden_size")),
        "action_embedding_dim": int(
            getattr(getattr(what_policy, "previous_action_embedding"), "embedding_dim", 8)
        ),
        "index_embedding_dim": int(
            getattr(getattr(what_policy, "action_index_embedding"), "embedding_dim", 8)
        ),
        "state_dict": state_dict,
        "training": bool(policy.training),
    }
    if _is_hierarchical_policy(policy):
        spec.update(
            {
                "kind": "hierarchical",
                "gate_hidden_sizes": tuple(int(value) for value in policy.gate.hidden_sizes),
            }
        )
    elif isinstance(policy, ppo_trigger_policy.FullSharedActorCritic):
        spec["kind"] = "full_shared"
    elif isinstance(policy, ppo_trigger_policy.ConditionalTriggerActorCritic):
        spec["kind"] = "conditional"
    else:
        return None
    return spec


def _rebuild_cf_worker_policy(spec: dict[str, Any]) -> Any:
    """Rebuild a frozen CPU policy from the explicit worker spec."""

    kind = str(spec["kind"])
    if kind == "hierarchical":
        from ppo import hierarchical_reconfiguration_gate

        policy = hierarchical_reconfiguration_gate.build_hierarchical_policy(
            int(spec["observation_dim"]),
            spec["action_spec"],
            subpolicy_hidden_sizes=tuple(int(value) for value in spec["hidden_sizes"]),
            autoregressive_hidden_size=int(spec["autoregressive_hidden_size"]),
            gate_hidden_sizes=tuple(int(value) for value in spec["gate_hidden_sizes"]),
        )
    else:
        policy_class = (
            ppo_trigger_policy.FullSharedActorCritic
            if kind == "full_shared"
            else ppo_trigger_policy.ConditionalTriggerActorCritic
        )
        policy = policy_class(
            int(spec["observation_dim"]),
            spec["action_spec"],
            hidden_sizes=tuple(int(value) for value in spec["hidden_sizes"]),
            autoregressive_hidden_size=int(spec["autoregressive_hidden_size"]),
            action_embedding_dim=int(spec["action_embedding_dim"]),
            index_embedding_dim=int(spec["index_embedding_dim"]),
        )
    policy.load_state_dict(spec["state_dict"], strict=True)
    policy.train(bool(spec["training"]))
    return policy


def _initialize_cf_worker(worker_payload: dict[str, Any]) -> None:
    """Initialize one isolated CPU worker; never starts the training CLI."""

    global _CF_WORKER_STATE
    if torch is not None:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Some embedded runtimes initialize inter-op threads before the
            # worker initializer.  In that case the existing setting is safe.
            pass
    _CF_WORKER_STATE = {
        "policy": _rebuild_cf_worker_policy(worker_payload["policy_spec"]),
        "context": worker_payload,
    }


def _worker_context() -> tuple[Any, dict[str, Any]]:
    if _CF_WORKER_STATE is None:
        raise RuntimeError("CF worker尚未完成initializer。")
    return _CF_WORKER_STATE["policy"], _CF_WORKER_STATE["context"]


def _single_direct_cf_reservoir(
    snapshot: DirectCFDecisionSnapshot,
    action_spec: ppo_trigger_policy.TriggerActionSpec,
    total_seen: int,
    trigger_free_counts: tuple[int, ...],
) -> DirectCFReservoir:
    """Create a one-item facade while reusing the serial CF helper exactly."""

    reservoir = DirectCFReservoir(
        1,
        np.random.default_rng(0),
        trigger_count=int(action_spec.action_dim),
        trigger_labels=tuple(action_spec.labels),
    )
    reservoir.items = [snapshot]
    reservoir.total_seen = int(total_seen)
    reservoir._trigger_free_counts = [int(value) for value in trigger_free_counts]
    return reservoir


def _cf_worker_direct_job(sample_index: int) -> DirectCFCreditSample:
    """ProcessPool job for one WHAT Direct-CF sample."""

    policy, context = _worker_context()
    action_spec = context["action_spec"]
    snapshot = context["snapshots"][int(sample_index)]
    reservoir = _single_direct_cf_reservoir(
        snapshot,
        action_spec,
        int(context["direct_total_seen"]),
        tuple(context["direct_trigger_free_counts"]),
    )
    samples = run_direct_cf_branches(
        policy,
        reservoir,
        context["aligned_steps"],
        context["baseline_batch"],
        context["local_cache_state"],
        context["service_catalog"],
        context["project_cfg"],
        context["cache_nodes"],
        context["cache_config"],
        context["flight_boundary"],
        context["observation_spec"],
        action_spec,
        int(context["terminal_slot"]),
        collection_policy_version=int(context["collection_policy_version"]),
        episode_index=int(context["episode_index"]),
    )
    if len(samples) != 1:
        raise RuntimeError("Direct-CF worker必须恰好返回一个sample。")
    original_index = int(sample_index)
    snapshot_trigger = int(snapshot.trigger_index)
    return replace(
        samples[0],
        sample_index_in_episode=original_index,
        sampling_weight=float(context["direct_sampling_weights"][original_index]),
        trigger_free_count=int(context["direct_trigger_free_counts"][snapshot_trigger]),
    )


def _cf_worker_gate_job(sample_index: int) -> Any:
    """ProcessPool job for one paired Gate CONFIG/NULL sample."""

    policy, context = _worker_context()
    reservoir = GateCFReservoir(1, np.random.default_rng(0))
    reservoir.items = [context["snapshots"][int(sample_index)]]
    samples = run_gate_cf_branches(
        policy,
        reservoir,
        context["aligned_steps"],
        context["baseline_batch"],
        context["local_cache_state"],
        context["service_catalog"],
        context["project_cfg"],
        context["cache_nodes"],
        context["cache_config"],
        context["flight_boundary"],
        context["observation_spec"],
        context["action_spec"],
        int(context["terminal_slot"]),
        collection_policy_version=int(context["collection_policy_version"]),
        episode_index=int(context["episode_index"]),
    )
    if len(samples) != 1:
        raise RuntimeError("Gate-CF worker必须恰好返回一个paired sample。")
    return samples[0]


def _compact_cf_baseline_batch(
    baseline_batch: RolloutBatch,
) -> _CFSuffixBaselineBatch:
    return _CFSuffixBaselineBatch(
        transitions=tuple(
            _CFSuffixBaselineTransition(
                reward=float(item.reward),
                actual_task_utility=float(item.actual_task_utility),
                cache_cost=float(item.cache_cost),
                movement_cost=float(item.movement_cost),
            )
            for item in baseline_batch.transitions
        )
    )


def _run_cf_process_pool(
    *,
    stage: str,
    policy: Any,
    snapshots: list[Any],
    aligned_steps: list[AlignedRolloutStep],
    baseline_batch: RolloutBatch,
    local_cache_state: dict[str, set[int]],
    service_catalog: dict[str, Any],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    flight_boundary: tuple[float, float, float, float],
    observation_spec: ppo_trigger_policy.ConfigurationObservationSpec,
    action_spec: ppo_trigger_policy.TriggerActionSpec,
    terminal_slot: int,
    collection_policy_version: int,
    episode_index: int,
    parallel_workers: int,
    execution_status: dict[str, Any] | None = None,
    direct_total_seen: int = 0,
    direct_trigger_free_counts: tuple[int, ...] = (),
    direct_sampling_weights: tuple[float, ...] = (),
) -> list[Any] | None:
    """Run independent CF jobs in input order; return None when CPU parallelism is unavailable."""

    workers = _normalize_cf_parallel_workers(parallel_workers)
    if workers <= 1 or len(snapshots) <= 1:
        _record_cf_execution_status(execution_status, mode="serial", workers=1)
        return None
    policy_spec = _cf_policy_worker_spec(policy)
    if policy_spec is None:
        _record_cf_execution_status(execution_status, mode="serial", workers=1)
        return None
    if stage not in {"direct", "gate"}:
        raise ValueError(f"未知 CF parallel stage：{stage}")
    worker_payload = {
        "stage": stage,
        "policy_spec": policy_spec,
        "snapshots": tuple(snapshots),
        "aligned_steps": tuple(aligned_steps),
        "baseline_batch": _compact_cf_baseline_batch(baseline_batch),
        "local_cache_state": local_cache_state,
        "service_catalog": service_catalog,
        "project_cfg": project_cfg,
        "cache_nodes": cache_nodes,
        "cache_config": cache_config,
        "flight_boundary": flight_boundary,
        "observation_spec": observation_spec,
        "action_spec": action_spec,
        "terminal_slot": int(terminal_slot),
        "collection_policy_version": int(collection_policy_version),
        "episode_index": int(episode_index),
        "direct_total_seen": int(direct_total_seen),
        "direct_trigger_free_counts": tuple(int(value) for value in direct_trigger_free_counts),
        "direct_sampling_weights": tuple(float(value) for value in direct_sampling_weights),
    }
    worker_count = min(workers, len(snapshots))
    _record_cf_execution_status(execution_status, mode="parallel", workers=worker_count)
    caller_rng_state = capture_direct_cf_rng_state()
    try:
        mp_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp_context,
            initializer=_initialize_cf_worker,
            initargs=(worker_payload,),
        ) as executor:
            worker = _cf_worker_direct_job if stage == "direct" else _cf_worker_gate_job
            # executor.map guarantees input-order results even when jobs finish
            # out of order; chunksize=1 keeps the sample boundary explicit.
            return list(executor.map(worker, range(len(snapshots)), chunksize=1))
    finally:
        # Child processes have isolated RNGs; restore the parent explicitly so
        # pool startup/serialization can never alter training RNG semantics.
        restore_direct_cf_rng_state(caller_rng_state)


# 这是实现边界而非待调优的训练超参数：central critic 最多保留最近两个完整
# rollout，正式episode长度为276时最多552条 (observation, action, MC return) 样本。
CRITIC_REPLAY_MAX_ROLLOUTS = 2


class CriticReplayBuffer:
    """只保存完整rollout的 central-critic supervised replay。

    Replay 中不保存 transition、Matching、actor ratio 或 old policy tensor；
    Actor/scalar-V PPO 永远只使用当前 on-policy rollout。
    """

    def __init__(self, max_rollouts: int = CRITIC_REPLAY_MAX_ROLLOUTS) -> None:
        if int(max_rollouts) != CRITIC_REPLAY_MAX_ROLLOUTS:
            raise ValueError(
                "CriticReplayBuffer 当前固定保留最近两个完整rollout；"
                "该实现边界不是可调训练超参数。"
            )
        self.max_rollouts = int(max_rollouts)
        self._rollouts: list[dict[str, np.ndarray]] = []

    def append_rollout(self, rollout_batch: RolloutBatch) -> None:
        observations = np.asarray(rollout_batch.observations, dtype=np.float32)
        actions = np.asarray(rollout_batch.actions, dtype=np.int8)
        targets = np.asarray(rollout_batch.monte_carlo_returns, dtype=np.float32)
        if observations.ndim != 2 or actions.ndim != 2 or targets.ndim != 1:
            raise ValueError("critic replay 的 observation/action/target 维度非法。")
        if observations.shape[0] != actions.shape[0] or targets.shape[0] != observations.shape[0]:
            raise ValueError("critic replay 的 observation/action/target 样本数必须一致。")
        if observations.shape[0] <= 0:
            raise ValueError("critic replay 不接受空 rollout。")
        if not np.all(np.isin(actions, [0, 1])):
            raise ValueError("critic replay action 必须是0/1。")
        self._rollouts.append(
            {
                "observations": observations.copy(),
                "actions": actions.copy(),
                "monte_carlo_returns": targets.copy(),
            }
        )
        if len(self._rollouts) > self.max_rollouts:
            self._rollouts = self._rollouts[-self.max_rollouts :]

    @property
    def rollout_count(self) -> int:
        return len(self._rollouts)

    @property
    def sample_count(self) -> int:
        return int(sum(item["observations"].shape[0] for item in self._rollouts))

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._rollouts:
            raise ValueError("critic replay 为空。")
        return (
            np.concatenate([item["observations"] for item in self._rollouts], axis=0),
            np.concatenate([item["actions"] for item in self._rollouts], axis=0),
            np.concatenate([item["monte_carlo_returns"] for item in self._rollouts], axis=0),
        )

    def state_dict(self) -> dict[str, Any]:
        """返回可直接放入torch checkpoint的轻量数组状态。"""

        return {
            "max_rollouts": int(self.max_rollouts),
            "rollouts": [
                {
                    "observations": item["observations"].copy(),
                    "actions": item["actions"].copy(),
                    "monte_carlo_returns": item["monte_carlo_returns"].copy(),
                }
                for item in self._rollouts
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("critic replay state 必须是dict。")
        saved_max = int(state.get("max_rollouts", self.max_rollouts))
        if saved_max != self.max_rollouts:
            raise ValueError("critic replay max_rollouts 与当前实现边界不一致。")
        raw_rollouts = state.get("rollouts")
        if not isinstance(raw_rollouts, list) or len(raw_rollouts) > self.max_rollouts:
            raise ValueError("critic replay rollouts 数量非法。")
        restored: list[dict[str, np.ndarray]] = []
        for raw in raw_rollouts:
            if not isinstance(raw, dict):
                raise ValueError("critic replay rollout item 必须是dict。")
            observations = np.asarray(raw["observations"], dtype=np.float32)
            actions = np.asarray(raw["actions"], dtype=np.int8)
            targets = np.asarray(raw["monte_carlo_returns"], dtype=np.float32)
            if (
                observations.ndim != 2
                or actions.ndim != 2
                or targets.ndim != 1
                or observations.shape[0] != actions.shape[0]
                or targets.shape[0] != observations.shape[0]
                or not np.all(np.isin(actions, [0, 1]))
            ):
                raise ValueError("checkpoint中的critic replay item形状或action非法。")
            restored.append(
                {
                    "observations": observations.copy(),
                    "actions": actions.copy(),
                    "monte_carlo_returns": targets.copy(),
                }
            )
        self._rollouts = restored


class InfeasibleActionBlocker(RuntimeError):
    """Policy 采样出不可行动作时立即停止；不做 resample/fallback。"""


def ppo_gae_config_from_project(project_cfg: dict[str, Any]) -> tuple[float, float]:
    """读取 PPO discounted training/GAE 参数；正式性能另以 raw G 累计评价。"""

    return (
        float(project_config.get_config_value(project_cfg, "algorithm.ppo.gamma")),
        float(project_config.get_config_value(project_cfg, "algorithm.ppo.gae_lambda")),
    )


def compute_gae(
    rewards: Any,
    values: Any,
    dones: Any,
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """纯函数 GAE：不依赖 policy、environment 或任何文件。"""

    rewards_a = np.asarray(rewards, dtype=np.float64).reshape(-1)
    values_a = np.asarray(values, dtype=np.float64).reshape(-1)
    dones_a = np.asarray(dones, dtype=np.float64).reshape(-1)
    if not (rewards_a.shape == values_a.shape == dones_a.shape):
        raise ValueError("rewards/values/dones shape 必须一致。")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma 和 gae_lambda 必须位于 [0,1]。")
    if not np.all(np.isfinite(rewards_a)) or not np.all(np.isfinite(values_a)):
        raise ValueError("rewards/values 包含 NaN/Inf。")
    if not np.all(np.isin(dones_a, [0.0, 1.0])):
        raise ValueError("dones 必须是 0/1。")
    advantages = np.zeros_like(rewards_a, dtype=np.float64)
    next_advantage = 0.0
    next_value = float(bootstrap_value)
    for index in range(len(rewards_a) - 1, -1, -1):
        non_terminal = 1.0 - float(dones_a[index])
        delta = rewards_a[index] + gamma * non_terminal * next_value - values_a[index]
        next_advantage = delta + gamma * gae_lambda * non_terminal * next_advantage
        advantages[index] = next_advantage
        next_value = values_a[index]
    returns = advantages + values_a
    return advantages.astype(np.float32), returns.astype(np.float32)


def compute_monte_carlo_returns(
    rewards: Any,
    dones: Any,
    gamma: float,
) -> np.ndarray:
    """计算已经收集的 on-policy reward sequence 的 suffix Monte-Carlo return。

    该函数只读取 rollout 内已有 rewards/dones，不访问未来场景或环境状态。
    terminal transition 的 return 就是其自身 reward；多段数据若出现 done，
    done 之后的 transition 不会泄漏到 done 之前的 episode。
    """

    rewards_a = np.asarray(rewards, dtype=np.float64).reshape(-1)
    dones_a = np.asarray(dones, dtype=np.float64).reshape(-1)
    if rewards_a.shape != dones_a.shape:
        raise ValueError("rewards/dones shape 必须一致。")
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma 必须位于 [0,1]。")
    if not np.all(np.isfinite(rewards_a)):
        raise ValueError("rewards 包含 NaN/Inf。")
    if not np.all(np.isin(dones_a, [0.0, 1.0])):
        raise ValueError("dones 必须是 0/1。")
    returns = np.zeros_like(rewards_a, dtype=np.float64)
    next_return = 0.0
    for index in range(len(rewards_a) - 1, -1, -1):
        if dones_a[index] == 1.0:
            next_return = 0.0
        returns[index] = rewards_a[index] + float(gamma) * next_return
        next_return = returns[index]
    if not np.all(np.isfinite(returns)):
        raise ValueError("Monte-Carlo returns 包含 NaN/Inf。")
    return returns.astype(np.float32)


def _value_only(policy: Any, observation: np.ndarray) -> float:
    zero_action = np.zeros(policy.action_dim, dtype=np.int8)
    evaluation = policy.evaluate_actions(observation, zero_action)
    return float(evaluation.value.detach().cpu())


def _check_reward_invariant(result: slot_environment.SlotExecutionResult) -> None:
    if result.system_utility is None:
        raise AssertionError("feasible step 缺少 system_utility。")
    expected = float(result.actual_task_utility) - float(result.cache_cost) - float(result.movement_cost)
    if not math.isclose(float(result.system_utility), expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError("reward invariant G=Q_actual-cache_cost-movement_cost 失败。")


def _changed_uav(previous: dict[str, Any], final: dict[str, Any], uav_id: str) -> bool:
    return float(np.linalg.norm(np.asarray(final[uav_id]) - np.asarray(previous[uav_id]))) > 1e-6


def _check_trigger_semantics(
    previous_cache: np.ndarray,
    previous_uav: dict[str, Any],
    result: slot_environment.SlotExecutionResult,
    action_spec: ppo_trigger_policy.TriggerActionSpec,
) -> None:
    final_cache = np.asarray(result.next_cache_state, dtype=np.int8)
    for index, trigger in enumerate(result.input_cache_triggers):
        equal = np.array_equal(previous_cache[index], final_cache[index])
        if int(trigger) == 0 and not equal:
            raise AssertionError(f"z_ca=0 但 cache node {index} 改变。")
    final_uav = result.next_uav_positions or {}
    for uav_id in action_spec.uav_ids:
        trigger = int(result.input_uav_triggers[uav_id])
        changed = _changed_uav(previous_uav, final_uav, uav_id)
        if trigger == 0 and changed:
            raise AssertionError(f"z_q=0 但 {uav_id} 移动。")


def _make_observation(
    previous_cache: np.ndarray,
    previous_uav: dict[str, Any],
    step: AlignedRolloutStep,
    observation_spec: ppo_trigger_policy.ConfigurationObservationSpec,
    flight_boundary: tuple[float, float, float, float],
) -> np.ndarray:
    return ppo_trigger_policy.build_configuration_observation(
        previous_cache,
        previous_uav,
        step.predicted_vehicle_positions,
        step.predicted_service_popularity,
        observation_spec,
        flight_boundary,
    )


def build_trigger_action_mask_provider(
    previous_cache_state: np.ndarray,
    previous_uav_positions: dict[str, Any],
    service_catalog: dict[str, Any],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    flight_boundary: tuple[float, float, float, float],
    action_spec: ppo_trigger_policy.TriggerActionSpec,
) -> Callable[[int, np.ndarray], np.ndarray]:
    """构造当前状态下的 autoregressive binary trigger mask。

    Cache trigger 的 0/1 action 都对当前合法 previous cache 开放；UAV trigger
    则按 policy 当前的 UAV 顺序逐步检查几何 candidate assignment。这里不调用
    objective，因此 objective 选择造成的 residual infeasibility 仍由
    InfeasibleActionBlocker 负责暴露。
    """

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


def collect_on_policy_rollout(
    policy: Any,
    initial_cache_state: np.ndarray,
    initial_uav_positions: dict[str, Any],
    aligned_steps: list[AlignedRolloutStep],
    local_cache_state: dict[str, set[int]],
    service_catalog: dict[str, Any],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    flight_boundary: tuple[float, float, float, float],
    observation_spec: ppo_trigger_policy.ConfigurationObservationSpec,
    action_spec: ppo_trigger_policy.TriggerActionSpec,
    gamma: float,
    gae_lambda: float,
    terminal_slot: int,
    direct_cf_reservoir: DirectCFReservoir | None = None,
    episode_index: int = 0,
    collection_policy_version: int = 0,
    collect_reference_states: bool = False,
    gate_cf_reservoir: GateCFReservoir | None = None,
    forced_gate_action: int | None = None,
    force_noop_configuration: bool = False,
) -> RolloutBatch:
    """收集连续 on-policy rollout；核心逻辑不读磁盘、不重置 RNG。"""

    if not aligned_steps:
        raise ValueError("aligned_steps 不能为空。")
    previous_cache = np.asarray(initial_cache_state, dtype=np.int8).copy()
    previous_uav = {
        str(uav_id): (float(np.asarray(xy)[0]), float(np.asarray(xy)[1]))
        for uav_id, xy in initial_uav_positions.items()
    }
    transitions: list[RolloutTransition] = []
    action_masks: list[np.ndarray] = []
    old_per_action_log_probs: list[np.ndarray] = []
    old_per_action_probability_tables: list[np.ndarray] = []
    reference_decisions: list[ReferenceStateRecord] = []
    state_propagation_ok = True
    if direct_cf_reservoir is not None and not isinstance(direct_cf_reservoir, DirectCFReservoir):
        raise TypeError("direct_cf_reservoir 必须是 DirectCFReservoir 或 None。")
    if gate_cf_reservoir is not None and not isinstance(gate_cf_reservoir, GateCFReservoir):
        raise TypeError("gate_cf_reservoir 必须是 GateCFReservoir 或 None。")
    hierarchical_rollout = _is_hierarchical_policy(policy)
    if forced_gate_action is not None:
        if not hierarchical_rollout:
            raise TypeError("forced_gate_action 只能用于分层 Gate 策略。")
        if int(forced_gate_action) not in (0, 1):
            raise ValueError("forced_gate_action 必须为0或1。")
    if force_noop_configuration and int(forced_gate_action or 0) != 1:
        raise ValueError("force_noop_configuration 只允许与 forced_gate_action=1 一起使用。")

    # aligned_steps 可包含最后一个只用于 truncation bootstrap 的 next step。
    # 若最后一个 step 已是真正 terminal，则全部可执行；否则保留最后一个不采样动作。
    execution_steps = (
        aligned_steps
        if int(aligned_steps[-1].slot) >= int(terminal_slot)
        else aligned_steps[:-1]
    )
    if not execution_steps:
        raise ValueError("aligned_steps 至少需要一个可执行 step。")

    for step_index, step in enumerate(execution_steps):
        if step_index > 0:
            prev = transitions[-1]
            state_propagation_ok = state_propagation_ok and np.array_equal(
                prev.next_cache_state,
                previous_cache,
            )
            for uav_id in action_spec.uav_ids:
                state_propagation_ok = state_propagation_ok and np.allclose(
                    np.asarray(prev.next_uav_positions[uav_id]),
                    np.asarray(previous_uav[uav_id]),
                    atol=0.0,
                    rtol=0.0,
                )
        observation = _make_observation(
            previous_cache,
            previous_uav,
            step,
            observation_spec,
            flight_boundary,
        )
        action_mask_provider = build_trigger_action_mask_provider(
            previous_cache,
            previous_uav,
            service_catalog,
            project_cfg,
            cache_nodes,
            flight_boundary,
            action_spec,
        )
        if force_noop_configuration:
            original_mask_provider = action_mask_provider

            def action_mask_provider(
                action_index: int,
                prefix: np.ndarray,
            ) -> np.ndarray:
                """保留动态合法性检查，并在验证分支中强制 WHAT 为全零。"""

                dynamic_mask = np.asarray(
                    original_mask_provider(action_index, prefix),
                    dtype=bool,
                )
                if not bool(dynamic_mask[0]):
                    raise AssertionError("当前动态动作约束不允许合法的零动作。")
                return np.asarray([True, False], dtype=bool)
        previous_cache_before_action = previous_cache.copy()
        previous_uav_before_action = {
            str(uav_id): (
                float(np.asarray(xy)[0]),
                float(np.asarray(xy)[1]),
            )
            for uav_id, xy in previous_uav.items()
        }
        baseline_rng_before = (
            capture_direct_cf_rng_state()
            if direct_cf_reservoir is not None or gate_cf_reservoir is not None
            else None
        )
        sample_kwargs: dict[str, Any] = {
            "deterministic": False,
            "action_mask": action_mask_provider,
        }
        if forced_gate_action is not None:
            sample_kwargs["force_gate_action"] = int(forced_gate_action)
        sample = _sample_action_without_grad(policy, observation, **sample_kwargs)
        result = slot_environment.execute_slot(
            previous_cache,
            previous_uav,
            sample.cache_triggers,
            sample.uav_triggers,
            step.scenario_slice,
            step.actual_tasks,
            step.actual_vehicle_positions,
            local_cache_state,
            service_catalog,
            project_cfg,
            cache_nodes,
            cache_config,
            flight_boundary,
            slot_id=step.slot,
            time_s=step.time_s,
        )
        if not result.step_feasible:
            raise InfeasibleActionBlocker(
                json.dumps(
                    {
                        "message": "INFEASIBLE ACTION BLOCKER",
                        "origin_index": step.origin_index,
                        "slot": step.slot,
                        "time_s": step.time_s,
                        "action_vector": sample.action_vector.astype(int).tolist(),
                        "cache_triggers": sample.cache_triggers.astype(int).tolist(),
                        "uav_triggers": sample.uav_triggers,
                        "infeasible_reason": result.infeasible_reason,
                    },
                    ensure_ascii=False,
                )
            )
        _check_reward_invariant(result)
        _check_trigger_semantics(previous_cache, previous_uav, result, action_spec)
        baseline_rng_after_slot = (
            capture_direct_cf_rng_state()
            if direct_cf_reservoir is not None or gate_cf_reservoir is not None
            else None
        )
        if direct_cf_reservoir is not None:
            if baseline_rng_before is None or baseline_rng_after_slot is None:
                raise AssertionError("Direct-CF reservoir缺少baseline RNG snapshot。")
            for action_index in range(action_spec.action_dim):
                decision_id = (
                    f"policy{int(collection_policy_version)}:episode{int(episode_index)}:"
                    f"origin{int(step.origin_index)}:slot{int(step.slot)}:trigger{int(action_index)}"
                )
                # A NULL Gate action is not a sampled WHAT decision.  Do not
                # manufacture an all-zero conditional trigger sample or a
                # WHAT reference state from it.
                if collect_reference_states and int(getattr(sample, "gate_action", 1)) == 1:
                    reference_decisions.append(
                        ReferenceStateRecord(
                            decision_id=decision_id,
                            episode_index=int(episode_index),
                            step_index=int(step_index),
                            slot=int(step.slot),
                            origin_index=int(step.origin_index),
                            trigger_index=int(action_index),
                            trigger_label=str(action_spec.labels[action_index]),
                            observation=np.asarray(observation, dtype=np.float32).copy(),
                            action_prefix=np.asarray(
                                sample.action_vector[:action_index], dtype=np.int8
                            ).copy(),
                            action_vector=np.asarray(sample.action_vector, dtype=np.int8).copy(),
                            action_mask=np.asarray(sample.per_action_masks, dtype=bool).copy(),
                            p_old=float(sample.per_action_probabilities[action_index]),
                            collection_policy_version=int(collection_policy_version),
                        )
                    )
                if int(getattr(sample, "gate_action", 1)) == 1 and bool(
                    np.all(sample.per_action_masks[action_index])
                ):
                    direct_cf_reservoir.consider(
                        DirectCFDecisionSnapshot(
                            step_index=int(step_index),
                            slot=int(step.slot),
                            origin_index=int(step.origin_index),
                            trigger_index=int(action_index),
                            trigger_label=str(action_spec.labels[action_index]),
                            decision_id=decision_id,
                            previous_cache_state=previous_cache_before_action.copy(),
                            previous_uav_positions=previous_uav_before_action.copy(),
                            observation=np.asarray(observation, dtype=np.float32).copy(),
                            action_vector=np.asarray(sample.action_vector, dtype=np.int8).copy(),
                            action_prefix=np.asarray(
                                sample.action_vector[:action_index], dtype=np.int8
                            ).copy(),
                            old_p1=float(sample.per_action_probabilities[action_index]),
                            old_chosen_log_prob=float(sample.per_action_log_probs[action_index]),
                            action_mask=np.asarray(sample.per_action_masks, dtype=bool).copy(),
                            baseline_rng_before=baseline_rng_before,
                            baseline_rng_after_slot=baseline_rng_after_slot,
                        )
                    )
        if gate_cf_reservoir is not None:
            if baseline_rng_before is None or baseline_rng_after_slot is None:
                raise AssertionError("Gate-CF reservoir缺少baseline RNG snapshot。")
            if not hasattr(sample, "gate_action"):
                raise TypeError("gate_cf_reservoir 只能用于Hierarchical Gate policy。")
            decision_id = (
                f"gatepolicy{int(collection_policy_version)}:episode{int(episode_index)}:"
                f"origin{int(step.origin_index)}:slot{int(step.slot)}"
            )
            gate_cf_reservoir.consider(
                GateDecisionSnapshot(
                    step_index=int(step_index),
                    slot=int(step.slot),
                    origin_index=int(step.origin_index),
                    decision_id=decision_id,
                    previous_cache_state=previous_cache_before_action.copy(),
                    previous_uav_positions=previous_uav_before_action.copy(),
                    observation=np.asarray(observation, dtype=np.float32).copy(),
                    gate_action=int(sample.gate_action),
                    old_gate_probability=float(sample.p_old_gate),
                    old_gate_log_prob=float(sample.gate_old_logprob),
                    final_action_vector=np.asarray(sample.action_vector, dtype=np.int8).copy(),
                    baseline_rng_before=baseline_rng_before,
                    baseline_rng_after_slot=baseline_rng_after_slot,
                )
            )
        done = bool(step.slot >= terminal_slot)
        reward = float(result.system_utility)
        transition = RolloutTransition(
            slot=int(step.slot),
            time_s=float(step.time_s),
            origin_index=int(step.origin_index),
            observation=observation.copy(),
            action_vector=sample.action_vector.astype(np.int8).copy(),
            cache_triggers=sample.cache_triggers.astype(np.int8).copy(),
            uav_triggers=dict(sample.uav_triggers),
            old_log_prob=float(sample.joint_log_prob),
            old_per_action_log_probs=np.asarray(
                sample.per_action_log_probs,
                dtype=np.float32,
            ).copy(),
            old_per_action_probability_table=np.asarray(
                sample.per_action_probability_table,
                dtype=np.float32,
            ).copy(),
            old_value=float(sample.value),
            reward=reward,
            done=done,
            step_feasible=True,
            infeasible_reason="",
            actual_task_utility=float(result.actual_task_utility),
            cache_cost=float(result.cache_cost),
            movement_cost=float(result.movement_cost),
            next_cache_state=np.asarray(result.next_cache_state, dtype=np.int8).copy(),
            next_uav_positions=dict(result.next_uav_positions or {}),
            gate_action=(
                None if not hasattr(sample, "gate_action") else int(sample.gate_action)
            ),
            old_gate_probability=(
                None if not hasattr(sample, "p_old_gate") else float(sample.p_old_gate)
            ),
            old_gate_log_prob=(
                None if not hasattr(sample, "gate_old_logprob") else float(sample.gate_old_logprob)
            ),
            subpolicy_action_vector=(
                None
                if not hasattr(sample, "subpolicy_action_vector")
                else np.asarray(sample.subpolicy_action_vector, dtype=np.int8).copy()
            ),
        )
        transitions.append(transition)
        action_masks.append(sample.per_action_masks.copy())
        old_per_action_log_probs.append(transition.old_per_action_log_probs.copy())
        old_per_action_probability_tables.append(transition.old_per_action_probability_table.copy())
        previous_cache = transition.next_cache_state.copy()
        previous_uav = dict(transition.next_uav_positions)
        if done:
            break

    true_terminal = bool(transitions[-1].done)
    truncated = not true_terminal
    if true_terminal:
        bootstrap_value = 0.0
    else:
        used_count = len(transitions)
        if used_count >= len(aligned_steps):
            raise ValueError("rollout 截断需要额外 next aligned step 计算 bootstrap value。")
        next_step = aligned_steps[used_count]
        next_observation = _make_observation(
            previous_cache,
            previous_uav,
            next_step,
            observation_spec,
            flight_boundary,
        )
        bootstrap_value = _value_only(policy, next_observation)

    observations = np.stack([item.observation for item in transitions]).astype(np.float32)
    actions = np.stack([item.action_vector for item in transitions]).astype(np.int8)
    old_log_probs = np.asarray([item.old_log_prob for item in transitions], dtype=np.float32)
    old_per_action_log_probs_array = np.stack(old_per_action_log_probs).astype(np.float32)
    old_per_action_probabilities = np.stack(old_per_action_probability_tables).astype(np.float32)
    action_masks_array = np.stack(action_masks).astype(bool)
    actor_valid_mask = np.logical_and(
        action_masks_array[..., 0],
        action_masks_array[..., 1],
    )
    if hierarchical_rollout:
        gate_actions = np.asarray(
            [int(item.gate_action) for item in transitions], dtype=np.int8
        )
        old_gate_log_probs = np.asarray(
            [float(item.old_gate_log_prob) for item in transitions], dtype=np.float32
        )
        old_gate_probabilities = np.asarray(
            [float(item.old_gate_probability) for item in transitions], dtype=np.float32
        )
        if not np.all(np.isin(gate_actions, [0, 1])):
            raise AssertionError("hierarchical rollout gate_action 必须为0/1。")
        if not np.all(np.isfinite(old_gate_log_probs)) or not np.all(
            np.isfinite(old_gate_probabilities)
        ):
            raise AssertionError("hierarchical rollout Gate old probability/log-prob必须finite。")
        if not np.allclose(
            np.sum(old_per_action_log_probs_array, axis=1) + old_gate_log_probs,
            old_log_probs,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise AssertionError("Gate+WHAT chosen log-prob之和与joint old_log_prob不一致。")
        actor_valid_mask = np.logical_and(actor_valid_mask, gate_actions[:, None].astype(bool))
    else:
        gate_actions = None
        old_gate_log_probs = None
        old_gate_probabilities = None
        if not np.allclose(
            np.sum(old_per_action_log_probs_array, axis=1),
            old_log_probs,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise AssertionError("per-action chosen log-prob之和与joint old_log_prob不一致。")
    if not np.all(np.isfinite(old_per_action_probabilities)):
        raise ValueError("rollout per-action probability table 包含 NaN/Inf。")
    if not np.allclose(
        np.sum(old_per_action_probabilities, axis=-1),
        1.0,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise AssertionError("rollout per-action probability table 未归一化。")
    old_values = np.asarray([item.old_value for item in transitions], dtype=np.float32)
    rewards = np.asarray([item.reward for item in transitions], dtype=np.float32)
    dones = np.asarray([float(item.done) for item in transitions], dtype=np.float32)
    advantages, returns = compute_gae(
        rewards,
        old_values,
        dones,
        bootstrap_value,
        gamma,
        gae_lambda,
    )
    monte_carlo_returns = compute_monte_carlo_returns(rewards, dones, gamma)
    return RolloutBatch(
        observations=observations,
        actions=actions,
        action_masks=action_masks_array,
        old_log_probs=old_log_probs,
        old_per_action_log_probs=old_per_action_log_probs_array,
        old_per_action_probabilities=old_per_action_probabilities,
        actor_valid_mask=actor_valid_mask.astype(bool),
        old_values=old_values,
        rewards=rewards,
        dones=dones,
        advantages=advantages,
        returns=returns,
        monte_carlo_returns=monte_carlo_returns,
        slots=np.asarray([item.slot for item in transitions], dtype=np.int64),
        origin_indices=np.asarray([item.origin_index for item in transitions], dtype=np.int64),
        final_bootstrap_value=float(bootstrap_value),
        truncated=truncated,
        transitions=transitions,
        metadata={
            "true_terminal": true_terminal,
            "state_propagation_ok": state_propagation_ok,
            "reward_invariant_ok": True,
            "old_per_action_log_prob_consistency_ok": True,
            "actor_valid_trigger_count": int(np.sum(actor_valid_mask)),
            "direct_cf_total_free_decisions": (
                0 if direct_cf_reservoir is None else int(direct_cf_reservoir.total_seen)
            ),
            "direct_cf_sample_count": (
                0 if direct_cf_reservoir is None else int(direct_cf_reservoir.sample_count)
            ),
            "hierarchical_gate_rollout": bool(hierarchical_rollout),
            "gate_action1_count": (
                0 if gate_actions is None else int(np.sum(gate_actions == 1))
            ),
            "gate_action0_count": (
                0 if gate_actions is None else int(np.sum(gate_actions == 0))
            ),
            "gate_action1_ratio": (
                None if gate_actions is None else float(np.mean(gate_actions == 1))
            ),
            "mean_gate_probability": (
                None
                if old_gate_probabilities is None
                else float(np.mean(old_gate_probabilities))
            ),
            "gate_cf_total_legal_slots": (
                0 if gate_cf_reservoir is None else int(gate_cf_reservoir.total_seen)
            ),
            "gate_cf_sample_count": (
                0 if gate_cf_reservoir is None else int(gate_cf_reservoir.sample_count)
            ),
            "direct_cf_reservoir": direct_cf_reservoir,
            "reference_state_count": int(len(reference_decisions)),
        },
        reference_decisions=tuple(reference_decisions),
        gate_actions=None if gate_actions is None else gate_actions.copy(),
        old_gate_log_probs=(
            None if old_gate_log_probs is None else old_gate_log_probs.copy()
        ),
        old_gate_probabilities=(
            None if old_gate_probabilities is None else old_gate_probabilities.copy()
        ),
        subpolicy_actions=(
            None
            if not hierarchical_rollout
            else np.stack(
                [
                    np.asarray(item.subpolicy_action_vector, dtype=np.int8)
                    for item in transitions
                ]
            ).astype(np.int8)
        ),
    )


def run_direct_cf_branches(
    policy: Any,
    reservoir: DirectCFReservoir,
    aligned_steps: list[AlignedRolloutStep],
    baseline_batch: RolloutBatch,
    local_cache_state: dict[str, set[int]],
    service_catalog: dict[str, Any],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    flight_boundary: tuple[float, float, float, float],
    observation_spec: ppo_trigger_policy.ConfigurationObservationSpec,
    action_spec: ppo_trigger_policy.TriggerActionSpec,
    terminal_slot: int,
    collection_policy_version: int = 0,
    episode_index: int = 0,
    cf_parallel_workers: int = 1,
    execution_status: dict[str, Any] | None = None,
) -> list[DirectCFCreditSample]:
    """执行 selected FREE decisions 的 same-state frozen-old-policy continuation。

    baseline 的 actual-action suffix 直接读取 ``baseline_batch.transitions``；每条
    branch 只从 selected decision 的 pre-action snapshot 开始重放到 true terminal。
    当前 slot 的 target 前缀固定为 baseline prefix，target action 强制翻转，target
    后缀及未来 slot 由未更新的 old policy 重新 masked stochastic sampling。每次
    branch 开始前恢复该 snapshot 的 RNG，完成后恢复调用者 RNG，避免改变正式
    baseline/update 的随机数语义。
    """

    if not isinstance(reservoir, DirectCFReservoir):
        raise TypeError("run_direct_cf_branches 需要 DirectCFReservoir。")
    if not aligned_steps or len(baseline_batch.transitions) != len(aligned_steps):
        raise ValueError("Direct-CF branch 需要完整 aligned_steps 与 baseline transitions。")
    if int(aligned_steps[-1].slot) != int(terminal_slot):
        raise ValueError("Direct-CF branch 必须运行到 FormalTrainingProtocol true terminal。")
    collection_policy_version = int(collection_policy_version)
    if collection_policy_version < 0:
        raise ValueError("collection_policy_version 不能为负。")
    episode_index = int(episode_index)
    if episode_index < 0:
        raise ValueError("episode_index 不能为负。")

    # Hierarchical Gate collection selects WHAT decisions only from g=1
    # slots.  Its Direct-CF branch must therefore operate on the frozen WHAT
    # Actor directly; sampling the outer Gate again would be both semantically
    # wrong and capable of turning a valid target into NULL/all-zero.
    what_policy = getattr(policy, "sub_policy", policy)

    parallel_samples = _run_cf_process_pool(
        stage="direct",
        policy=policy,
        snapshots=list(reservoir.items),
        aligned_steps=aligned_steps,
        baseline_batch=baseline_batch,
        local_cache_state=local_cache_state,
        service_catalog=service_catalog,
        project_cfg=project_cfg,
        cache_nodes=cache_nodes,
        cache_config=cache_config,
        flight_boundary=flight_boundary,
        observation_spec=observation_spec,
        action_spec=action_spec,
        terminal_slot=terminal_slot,
        collection_policy_version=collection_policy_version,
        episode_index=episode_index,
        parallel_workers=cf_parallel_workers,
        execution_status=execution_status,
        direct_total_seen=int(reservoir.total_seen),
        direct_trigger_free_counts=reservoir.trigger_free_counts,
        direct_sampling_weights=tuple(
            float(reservoir.sampling_weight_for(snapshot.trigger_index))
            for snapshot in reservoir.items
        ),
    )
    if parallel_samples is not None:
        return [sample for sample in parallel_samples]

    caller_rng_state = capture_direct_cf_rng_state()
    samples: list[DirectCFCreditSample] = []

    def suffix_metric(start: int, field: str) -> float:
        return float(math.fsum(float(getattr(item, field)) for item in baseline_batch.transitions[start:]))

    try:
        for sample_index_in_episode, snapshot in enumerate(reservoir.items):
            if not bool(np.all(snapshot.action_mask[snapshot.trigger_index])):
                raise ValueError("Direct-CF reservoir target不是FREE decision。")
            if snapshot.step_index < 0 or snapshot.step_index >= len(aligned_steps):
                raise ValueError("Direct-CF snapshot step_index越界。")

            restore_direct_cf_rng_state(snapshot.baseline_rng_before)
            previous_cache = np.asarray(snapshot.previous_cache_state, dtype=np.int8).copy()
            previous_uav = {
                str(uav_id): (float(xy[0]), float(xy[1]))
                for uav_id, xy in snapshot.previous_uav_positions.items()
            }
            branch_g: list[float] = []
            branch_q: list[float] = []
            branch_cache: list[float] = []
            branch_uav: list[float] = []
            target_rng_aligned = False
            branch_steps = 0

            for step_index in range(snapshot.step_index, len(aligned_steps)):
                step = aligned_steps[step_index]
                observation = _make_observation(
                    previous_cache,
                    previous_uav,
                    step,
                    observation_spec,
                    flight_boundary,
                )
                mask_provider = build_trigger_action_mask_provider(
                    previous_cache,
                    previous_uav,
                    service_catalog,
                    project_cfg,
                    cache_nodes,
                    flight_boundary,
                    action_spec,
                )
                if step_index == snapshot.step_index:
                    sample = _sample_action_without_grad(
                        what_policy,
                        observation,
                        deterministic=False,
                        action_mask=mask_provider,
                        force_action_index=int(snapshot.trigger_index),
                        force_action_value=1 - int(snapshot.action_vector[snapshot.trigger_index]),
                        fixed_action_prefix=snapshot.action_prefix,
                    )
                else:
                    sample = _sample_action_without_grad(
                        what_policy,
                        observation,
                        deterministic=False,
                        action_mask=mask_provider,
                    )
                if step_index == snapshot.step_index:
                    expected_action = 1 - int(snapshot.action_vector[snapshot.trigger_index])
                    if int(sample.action_vector[snapshot.trigger_index]) != expected_action:
                        raise RuntimeError("Direct-CF branch target action未按要求翻转。")

                result = slot_environment.execute_slot(
                    previous_cache,
                    previous_uav,
                    sample.cache_triggers,
                    sample.uav_triggers,
                    step.scenario_slice,
                    step.actual_tasks,
                    step.actual_vehicle_positions,
                    local_cache_state,
                    service_catalog,
                    project_cfg,
                    cache_nodes,
                    cache_config,
                    flight_boundary,
                    slot_id=step.slot,
                    time_s=step.time_s,
                )
                if not result.step_feasible:
                    raise InfeasibleActionBlocker(
                        json.dumps(
                            {
                                "message": "INFEASIBLE DIRECT-CF BRANCH",
                                "slot": int(step.slot),
                                "origin_index": int(step.origin_index),
                                "trigger_label": snapshot.trigger_label,
                                "infeasible_reason": result.infeasible_reason,
                            },
                            ensure_ascii=False,
                        )
                    )
                _check_reward_invariant(result)
                _check_trigger_semantics(previous_cache, previous_uav, result, action_spec)
                if step_index == snapshot.step_index:
                    target_rng_aligned = direct_cf_rng_states_equal(
                        capture_direct_cf_rng_state(),
                        snapshot.baseline_rng_after_slot,
                    )
                branch_g.append(float(result.system_utility))
                branch_q.append(float(result.actual_task_utility))
                branch_cache.append(float(result.cache_cost))
                branch_uav.append(float(result.movement_cost))
                previous_cache = np.asarray(result.next_cache_state, dtype=np.int8).copy()
                previous_uav = {
                    str(uav_id): (float(xy[0]), float(xy[1]))
                    for uav_id, xy in (result.next_uav_positions or {}).items()
                }
                branch_steps += 1
                if int(step.slot) >= int(terminal_slot):
                    break

            if branch_steps != len(aligned_steps) - snapshot.step_index:
                raise RuntimeError("Direct-CF branch未运行到true terminal。")
            baseline_suffix_g = suffix_metric(snapshot.step_index, "reward")
            baseline_suffix_q = suffix_metric(snapshot.step_index, "actual_task_utility")
            baseline_suffix_cache = suffix_metric(snapshot.step_index, "cache_cost")
            baseline_suffix_uav = suffix_metric(snapshot.step_index, "movement_cost")
            branch_suffix_g = float(math.fsum(branch_g))
            branch_suffix_q = float(math.fsum(branch_q))
            branch_suffix_cache = float(math.fsum(branch_cache))
            branch_suffix_uav = float(math.fsum(branch_uav))
            delta_g = branch_suffix_g - baseline_suffix_g
            delta_q = branch_suffix_q - baseline_suffix_q
            delta_cache = branch_suffix_cache - baseline_suffix_cache
            delta_uav = branch_suffix_uav - baseline_suffix_uav
            identity_error = delta_g - delta_q + delta_cache + delta_uav
            if not math.isclose(identity_error, 0.0, rel_tol=1e-9, abs_tol=1e-9):
                raise AssertionError(
                    "Direct-CF Delta reward identity failed: "
                    f"slot={snapshot.slot}, error={identity_error}"
                )
            a_cf = (float(snapshot.action_vector[snapshot.trigger_index]) - float(snapshot.old_p1)) * delta_g
            samples.append(
                DirectCFCreditSample(
                    episode_index=episode_index,
                    sample_index_in_episode=int(sample_index_in_episode),
                    step_index=int(snapshot.step_index),
                    slot=int(snapshot.slot),
                    origin_index=int(snapshot.origin_index),
                    trigger_index=int(snapshot.trigger_index),
                    trigger_label=str(snapshot.trigger_label),
                    decision_id=str(snapshot.decision_id),
                    observation=snapshot.observation.copy(),
                    action_vector=snapshot.action_vector.copy(),
                    action_prefix=snapshot.action_prefix.copy(),
                    action_mask=snapshot.action_mask.copy(),
                    actual_action=int(snapshot.action_vector[snapshot.trigger_index]),
                    old_p1=float(snapshot.old_p1),
                    old_chosen_log_prob=float(snapshot.old_chosen_log_prob),
                    delta_g=float(delta_g),
                    delta_q=float(delta_q),
                    delta_cache=float(delta_cache),
                    delta_uav=float(delta_uav),
                    a_cf=float(a_cf),
                    baseline_suffix_g=float(baseline_suffix_g),
                    branch_suffix_g=float(branch_suffix_g),
                    branch_environment_steps=int(branch_steps),
                    crn_target_step_rng_aligned=bool(target_rng_aligned),
                    sampling_weight=float(reservoir.sampling_weight_for(snapshot.trigger_index)),
                    trigger_free_count=int(reservoir.trigger_free_counts[snapshot.trigger_index]),
                    collection_policy_version=collection_policy_version,
                )
            )
    finally:
        restore_direct_cf_rng_state(caller_rng_state)
    return samples


def run_gate_cf_branches(
    policy: Any,
    reservoir: GateCFReservoir,
    aligned_steps: list[AlignedRolloutStep],
    baseline_batch: RolloutBatch,
    local_cache_state: dict[str, set[int]],
    service_catalog: dict[str, Any],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    flight_boundary: tuple[float, float, float, float],
    observation_spec: ppo_trigger_policy.ConfigurationObservationSpec,
    action_spec: ppo_trigger_policy.TriggerActionSpec,
    terminal_slot: int,
    collection_policy_version: int = 0,
    episode_index: int = 0,
    cf_parallel_workers: int = 1,
    execution_status: dict[str, Any] | None = None,
) -> list[Any]:
    """Run the formal Gate Joint-Null paired branches.

    The reservoir contains pre-slot snapshots captured by the real
    hierarchical rollout.  Each selected slot is replayed twice from the
    same pre-action state: CONFIG forces ``g=1`` and samples the frozen WHAT
    policy under the saved dynamic mask; NULL forces ``g=0`` and exposes the
    strict all-zero action.  Both branches own an isolated local-cache copy,
    and all policy/environment suffix draws are restored from the snapshot's
    baseline RNG checkpoint so this helper cannot advance the caller's
    training RNG.
    """

    from ppo import hierarchical_reconfiguration_gate

    if not isinstance(reservoir, GateCFReservoir):
        raise TypeError("run_gate_cf_branches 需要 GateCFReservoir。")
    if not isinstance(policy, hierarchical_reconfiguration_gate.HierarchicalReconfigurationGatePolicy):
        raise TypeError("run_gate_cf_branches 需要 HierarchicalReconfigurationGatePolicy。")
    if not aligned_steps or len(baseline_batch.transitions) != len(aligned_steps):
        raise ValueError("Gate-CF branch 需要完整 aligned_steps 与 baseline transitions。")
    if int(aligned_steps[-1].slot) != int(terminal_slot):
        raise ValueError("Gate-CF branch 必须运行到 FormalTrainingProtocol true terminal。")
    collection_policy_version = int(collection_policy_version)
    episode_index = int(episode_index)
    if collection_policy_version < 0 or episode_index < 0:
        raise ValueError("Gate-CF collection policy version/episode index不能为负。")

    parallel_samples = _run_cf_process_pool(
        stage="gate",
        policy=policy,
        snapshots=list(reservoir.items),
        aligned_steps=aligned_steps,
        baseline_batch=baseline_batch,
        local_cache_state=local_cache_state,
        service_catalog=service_catalog,
        project_cfg=project_cfg,
        cache_nodes=cache_nodes,
        cache_config=cache_config,
        flight_boundary=flight_boundary,
        observation_spec=observation_spec,
        action_spec=action_spec,
        terminal_slot=terminal_slot,
        collection_policy_version=collection_policy_version,
        episode_index=episode_index,
        parallel_workers=cf_parallel_workers,
        execution_status=execution_status,
    )
    if parallel_samples is not None:
        return [sample for sample in parallel_samples]

    caller_rng_state = capture_direct_cf_rng_state()
    samples: list[Any] = []

    def suffix_metric(start: int, field: str) -> float:
        return float(
            math.fsum(float(getattr(item, field)) for item in baseline_batch.transitions[start:])
        )

    def clone_local_cache() -> dict[str, set[int]]:
        return {
            str(vehicle_id): {int(service_id) for service_id in services}
            for vehicle_id, services in local_cache_state.items()
        }

    try:
        for sample_index_in_episode, snapshot in enumerate(reservoir.items):
            if snapshot.step_index < 0 or snapshot.step_index >= len(aligned_steps):
                raise ValueError("Gate-CF snapshot step_index越界。")
            target_step = aligned_steps[snapshot.step_index]
            target_mask_provider = build_trigger_action_mask_provider(
                snapshot.previous_cache_state,
                snapshot.previous_uav_positions,
                service_catalog,
                project_cfg,
                cache_nodes,
                flight_boundary,
                action_spec,
            )
            branch_metrics: dict[str, hierarchical_reconfiguration_gate.GateBranchReturn] = {}
            target_rng_aligned: dict[str, bool] = {}

            def continuation(
                branch_name: str,
                first_sample: Any,
            ) -> hierarchical_reconfiguration_gate.GateBranchReturn:
                previous_cache = np.asarray(snapshot.previous_cache_state, dtype=np.int8).copy()
                previous_uav = {
                    str(uav_id): (float(xy[0]), float(xy[1]))
                    for uav_id, xy in snapshot.previous_uav_positions.items()
                }
                branch_local_cache = clone_local_cache()
                branch_g: list[float] = []
                branch_q: list[float] = []
                branch_cache: list[float] = []
                branch_uav: list[float] = []
                branch_steps = 0
                for step_index in range(snapshot.step_index, len(aligned_steps)):
                    step = aligned_steps[step_index]
                    observation = _make_observation(
                        previous_cache,
                        previous_uav,
                        step,
                        observation_spec,
                        flight_boundary,
                    )
                    mask_provider = build_trigger_action_mask_provider(
                        previous_cache,
                        previous_uav,
                        service_catalog,
                        project_cfg,
                        cache_nodes,
                        flight_boundary,
                        action_spec,
                    )
                    is_target_step = step_index == snapshot.step_index
                    if is_target_step:
                        sample = first_sample
                        if not np.array_equal(
                            np.asarray(observation, dtype=np.float32),
                            np.asarray(snapshot.observation, dtype=np.float32),
                        ):
                            raise AssertionError("Gate-CF snapshot observation与branch state不一致。")
                    else:
                        sample = _sample_action_without_grad(
                            policy,
                            observation,
                            deterministic=False,
                            action_mask=mask_provider,
                            consume_subpolicy_draw_for_null=True,
                        )
                    if is_target_step:
                        if branch_name == "NULL":
                            if int(sample.gate_action) != hierarchical_reconfiguration_gate.GATE_ACTION_NULL:
                                raise AssertionError("NULL Gate-CF branch未强制g=0。")
                            if not np.array_equal(
                                np.asarray(sample.action_vector, dtype=np.int8),
                                np.zeros(int(action_spec.action_dim), dtype=np.int8),
                            ):
                                raise AssertionError("NULL Gate-CF branch action不是严格全零。")
                        elif branch_name == "CONFIG":
                            if int(sample.gate_action) != hierarchical_reconfiguration_gate.GATE_ACTION_CONFIGURE:
                                raise AssertionError("CONFIG Gate-CF branch未强制g=1。")
                        else:
                            raise ValueError(f"未知 Gate-CF branch：{branch_name}")
                    result = slot_environment.execute_slot(
                        previous_cache,
                        previous_uav,
                        sample.cache_triggers,
                        sample.uav_triggers,
                        step.scenario_slice,
                        step.actual_tasks,
                        step.actual_vehicle_positions,
                        branch_local_cache,
                        service_catalog,
                        project_cfg,
                        cache_nodes,
                        cache_config,
                        flight_boundary,
                        slot_id=step.slot,
                        time_s=step.time_s,
                    )
                    if not result.step_feasible:
                        raise InfeasibleActionBlocker(
                            json.dumps(
                                {
                                    "message": "INFEASIBLE GATE-CF BRANCH",
                                    "branch": branch_name,
                                    "slot": int(step.slot),
                                    "origin_index": int(step.origin_index),
                                    "infeasible_reason": result.infeasible_reason,
                                },
                                ensure_ascii=False,
                            )
                        )
                    _check_reward_invariant(result)
                    _check_trigger_semantics(previous_cache, previous_uav, result, action_spec)
                    if step_index == snapshot.step_index:
                        target_rng_aligned[branch_name] = direct_cf_rng_states_equal(
                            capture_direct_cf_rng_state(),
                            snapshot.baseline_rng_after_slot,
                        )
                    branch_g.append(float(result.system_utility))
                    branch_q.append(float(result.actual_task_utility))
                    branch_cache.append(float(result.cache_cost))
                    branch_uav.append(float(result.movement_cost))
                    previous_cache = np.asarray(result.next_cache_state, dtype=np.int8).copy()
                    previous_uav = {
                        str(uav_id): (float(xy[0]), float(xy[1]))
                        for uav_id, xy in (result.next_uav_positions or {}).items()
                    }
                    branch_steps += 1
                    if int(step.slot) >= int(terminal_slot):
                        break
                expected_steps = len(aligned_steps) - snapshot.step_index
                if branch_steps != expected_steps:
                    raise RuntimeError("Gate-CF branch未运行到true terminal。")
                return hierarchical_reconfiguration_gate.GateBranchReturn(
                    branch=branch_name,
                    return_g=float(math.fsum(branch_g)),
                    return_q=float(math.fsum(branch_q)),
                    return_cache=float(math.fsum(branch_cache)),
                    return_uav=float(math.fsum(branch_uav)),
                    environment_steps=int(branch_steps),
                )

            restore_direct_cf_rng_state(snapshot.baseline_rng_before)
            try:
                joint = hierarchical_reconfiguration_gate.run_joint_null_branches(
                    policy,
                    snapshot.observation,
                    continuation,
                    action_mask=target_mask_provider,
                    crn_metadata={
                        "decision_id": str(snapshot.decision_id),
                        "episode_index": int(episode_index),
                        "step_index": int(snapshot.step_index),
                        "slot": int(snapshot.slot),
                        "origin_index": int(snapshot.origin_index),
                        "baseline_gate_action": int(snapshot.gate_action),
                    },
                )
            finally:
                restore_direct_cf_rng_state(snapshot.baseline_rng_before)
            config_return = joint.config_return
            null_return = joint.null_return
            gate_crn_metadata = copy.deepcopy(joint.crn_metadata)
            gate_crn_metadata.update(
                {
                    "config_branch_environment_steps": int(config_return.environment_steps),
                    "null_branch_environment_steps": int(null_return.environment_steps),
                    "config_target_step_rng_aligned": bool(
                        target_rng_aligned.get("CONFIG", False)
                    ),
                    "null_target_step_rng_aligned": bool(
                        target_rng_aligned.get("NULL", False)
                    ),
                }
            )
            delta_q = float(config_return.return_q - null_return.return_q)
            delta_cache = float(config_return.return_cache - null_return.return_cache)
            delta_uav = float(config_return.return_uav - null_return.return_uav)
            delta_gate = float(config_return.return_g - null_return.return_g)
            if not math.isclose(
                delta_gate,
                delta_q - delta_cache - delta_uav,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise AssertionError("Gate-CF Delta decomposition identity failed。")
            samples.append(
                hierarchical_reconfiguration_gate.build_gate_counterfactual_sample(
                    decision_id=str(snapshot.decision_id),
                    episode_index=episode_index,
                    step_index=int(snapshot.step_index),
                    slot=int(snapshot.slot),
                    origin_index=int(snapshot.origin_index),
                    gate_observation=snapshot.observation,
                    gate_action=int(snapshot.gate_action),
                    p_old_gate=float(snapshot.old_gate_probability),
                    gate_old_logprob=float(snapshot.old_gate_log_prob),
                    config_joint_action=joint.config_sample.action_vector,
                    null_joint_action=joint.null_sample.action_vector,
                    return_config=float(config_return.return_g),
                    return_null=float(null_return.return_g),
                    crn_metadata=gate_crn_metadata,
                    collection_policy_version=collection_policy_version,
                    delta_q=delta_q,
                    delta_cache=delta_cache,
                    delta_uav=delta_uav,
                )
            )
    finally:
        restore_direct_cf_rng_state(caller_rng_state)
    return samples


def _load_vehicle_prediction_h0(path: Path) -> dict[int, dict[str, tuple[float, float, int, float]]]:
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


def _load_popularity_prediction_h0(path: Path) -> dict[int, dict[int, tuple[float, int, float]]]:
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
            if slot in result and not math.isclose(result[slot], time_s, rel_tol=0.0, abs_tol=1e-9):
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
    """从正式 artifact 构造已对齐 steps；不重新预测/生成场景。

    ``include_bootstrap_step`` 保留旧的 truncation bootstrap 语义。正式
    train/test 区间使用 ``False``，避免为了 bootstrap 读取或执行区间外的
    actual task/reward。
    """

    if max_steps <= 0:
        raise ValueError("max_steps 必须大于 0。")
    vehicle_pred = _load_vehicle_prediction_h0(vehicle_prediction_file)
    popularity_pred = _load_popularity_prediction_h0(popularity_prediction_file)
    task_time_by_slot = _load_task_time_by_slot(tasks_file)
    n_origin = int(scenarios["origin_slots"].shape[0])
    if origin_start_index < 0 or origin_start_index >= n_origin:
        raise IndexError("origin_start_index 越界。")

    # 多准备一步，用于 rollout truncation bootstrap。
    extra_step = 1 if include_bootstrap_step else 0
    stop = min(n_origin, origin_start_index + max_steps + extra_step)
    steps: list[AlignedRolloutStep] = []
    for origin_index in range(origin_start_index, stop):
        scenario = cache_configuration.scenario_at_origin(scenarios, origin_index)
        slot = int(scenario["origin_slot"])
        time_s = float(scenario["origin_time_s"])
        if not math.isclose(float(task_time_by_slot.get(slot, float("nan"))), time_s, rel_tol=0.0, abs_tol=1e-9):
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
            if target_slot != slot or not math.isclose(target_time, time_s, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"vehicle prediction target 与 slot/time 不一致：slot={slot}")
            predicted_positions[vehicle_id] = np.asarray([x_pred, y_pred], dtype=np.float64)

        service_ids = list(range(int(scenario["rho_hat"].shape[1])))
        raw_popularity = popularity_pred.get(slot)
        if raw_popularity is None or set(raw_popularity) != set(service_ids):
            raise ValueError(f"popularity_predictions.csv 缺少 origin slot={slot} 的完整服务预测。")
        predicted_popularity: dict[int, float] = {}
        for service_id in service_ids:
            rho, target_slot, target_time = raw_popularity[service_id]
            if target_slot != slot or not math.isclose(target_time, time_s, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"popularity prediction target 与 slot/time 不一致：slot={slot}")
            predicted_popularity[service_id] = rho
        if not math.isclose(sum(predicted_popularity.values()), 1.0, rel_tol=1e-5, abs_tol=1e-5):
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


def _manual_gae_sanity() -> float:
    rewards = np.asarray([1.0, 0.5, -0.2], dtype=np.float64)
    values = np.asarray([0.3, 0.1, -0.1], dtype=np.float64)
    dones = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    bootstrap = 0.7
    gamma = 0.9
    lam = 0.8
    adv, _ = compute_gae(rewards, values, dones, bootstrap, gamma, lam)
    expected = np.zeros_like(rewards)
    next_adv = 0.0
    next_value = bootstrap
    for index in range(len(rewards) - 1, -1, -1):
        non_terminal = 1.0 - dones[index]
        delta = rewards[index] + gamma * non_terminal * next_value - values[index]
        next_adv = delta + gamma * lam * non_terminal * next_adv
        expected[index] = next_adv
        next_value = values[index]
    return float(np.max(np.abs(adv.astype(np.float64) - expected)))


def _batch_consistency(policy: Any, batch: RolloutBatch) -> tuple[float, float]:
    evaluation_kwargs = {
        "action_masks": batch.action_masks,
    }
    if getattr(batch, "gate_actions", None) is not None:
        evaluation_kwargs["gate_actions"] = batch.gate_actions
    evaluation = policy.evaluate_actions(
        batch.observations,
        batch.actions,
        **evaluation_kwargs,
    )
    log_probs = evaluation.joint_log_prob.detach().cpu().numpy()
    values = evaluation.value.detach().cpu().numpy()
    return (
        float(np.max(np.abs(log_probs - batch.old_log_probs))),
        float(np.max(np.abs(values - batch.old_values))),
    )


def _build_rollout_policy_from_config(
    cfg: dict[str, Any],
    observation_dim: int,
    action_spec: ppo_trigger_policy.TriggerActionSpec,
) -> Any:
    """按照统一配置构造 rollout 策略，不创建优化器。"""

    method = str(project_config.get_config_value(cfg, "algorithm.ppo.method"))
    network_cfg = ppo_trigger_policy.ppo_network_config_from_project(cfg)
    from ppo import hierarchical_reconfiguration_gate

    if method == hierarchical_reconfiguration_gate.HIERARCHICAL_GATE_METHOD:
        return hierarchical_reconfiguration_gate.build_hierarchical_policy(
            observation_dim,
            action_spec,
            subpolicy_hidden_sizes=tuple(
                int(value) for value in network_cfg["hidden_sizes"]
            ),
            autoregressive_hidden_size=int(
                network_cfg["autoregressive_hidden_size"]
            ),
        )
    return ppo_trigger_policy.ConditionalTriggerActorCritic(
        observation_dim,
        action_spec,
        **network_cfg,
    )


def run_minimal_sanity(
    config_path: str = "system_config.toml",
    origin_start_index: int = 0,
    max_steps: int = 8,
    *,
    forced_gate_action: int | None = None,
    force_noop_configuration: bool = False,
    repeat_check: bool = True,
) -> dict[str, Any]:
    cfg = project_config.load_config(config_path)
    project_config.validate_for_formal_experiment(cfg)
    root = Path(cfg["_meta"]["project_root"])
    seed = int(project_config.get_config_value(cfg, "experiment.seed"))

    cache_cfg = cache_configuration.cache_config_from_project(cfg)
    scenarios = cache_configuration.load_prediction_scenarios(
        cache_configuration.resolve_path(root, project_config.get_config_value(cfg, "paths.prediction_scenarios"))
    )
    service_catalog = cache_configuration.load_service_catalog(
        cache_configuration.resolve_path(root, project_config.get_config_value(cfg, "paths.service_catalog"))
    )
    first_origin = cache_configuration.scenario_at_origin(scenarios, origin_start_index)
    cache_nodes = cache_configuration.build_cache_nodes(first_origin, cache_cfg)
    initial_cache = cache_configuration.deterministic_initial_cache(
        first_origin,
        service_catalog,
        cache_nodes,
        cache_cfg,
    )
    initial_uav = uav_configuration.initial_uav_positions_from_project(cfg)
    boundary = uav_configuration.boundary_from_project(cfg)
    local_cache = task_scenario.load_tv_local_cache(
        cache_configuration.resolve_path(root, project_config.get_config_value(cfg, "paths.tv_local_cache"))
    )
    aligned_steps = prepare_aligned_rollout_steps(
        root,
        scenarios,
        origin_start_index,
        max_steps,
        root / "data/generated/vehicle_predictions.csv",
        root / "data/generated/popularity_predictions.csv",
        root / "data/generated/tasks.csv",
        root / "data/generated/trajectory.csv",
    )
    action_spec = ppo_trigger_policy.TriggerActionSpec.from_cache_nodes_and_project(cache_nodes, cfg)
    observation_spec = ppo_trigger_policy.ConfigurationObservationSpec.from_cache_nodes(
        cache_nodes,
        service_catalog["service_ids"].tolist(),
        action_spec.uav_ids,
        [str(value) for value in first_origin["vehicle_ids"].tolist()],
    )
    gamma, gae_lambda = ppo_gae_config_from_project(cfg)
    terminal_slot = int(scenarios["origin_slots"][-1])

    ppo_trigger_policy.set_torch_seed(seed)
    np.random.seed(seed)
    policy = _build_rollout_policy_from_config(
        cfg,
        observation_spec.observation_dim,
        action_spec,
    )
    batch = collect_on_policy_rollout(
        policy,
        initial_cache,
        initial_uav,
        aligned_steps,
        local_cache,
        service_catalog,
        cfg,
        cache_nodes,
        cache_cfg,
        boundary,
        observation_spec,
        action_spec,
        gamma,
        gae_lambda,
        terminal_slot,
        forced_gate_action=forced_gate_action,
        force_noop_configuration=force_noop_configuration,
    )
    logprob_error, value_error = _batch_consistency(policy, batch)

    repeated_ok: bool | None = None
    if repeat_check:
        # 重复 sanity：重新初始化同一策略参数和随机数，动作及 reward 应完全一致。
        ppo_trigger_policy.set_torch_seed(seed)
        np.random.seed(seed)
        policy_repeat = _build_rollout_policy_from_config(
            cfg,
            observation_spec.observation_dim,
            action_spec,
        )
        repeat = collect_on_policy_rollout(
            policy_repeat,
            initial_cache,
            initial_uav,
            aligned_steps,
            local_cache,
            service_catalog,
            cfg,
            cache_nodes,
            cache_cfg,
            boundary,
            observation_spec,
            action_spec,
            gamma,
            gae_lambda,
            terminal_slot,
            forced_gate_action=forced_gate_action,
            force_noop_configuration=force_noop_configuration,
        )
        repeated_ok = bool(
            np.array_equal(batch.actions, repeat.actions)
            and np.allclose(batch.rewards, repeat.rewards, atol=0.0, rtol=0.0)
        )
    return {
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "batch": batch,
        "action_spec": action_spec,
        "observation_spec": observation_spec,
        "manual_gae_max_error": _manual_gae_sanity(),
        "old_logprob_recompute_max_error": logprob_error,
        "old_value_recompute_max_error": value_error,
        "repeated_rollout_deterministic": repeated_ok,
        "policy": policy,
        "policy_type": type(policy).__name__,
        "formal_validation": True,
        "alignment_steps_prepared": len(aligned_steps),
    }


def run_hierarchical_minimal_e2e_validation(
    config_path: str = "system_config.toml",
    origin_start_index: int = 0,
) -> dict[str, Any]:
    """执行一次无训练的分层策略最小端到端验证。"""

    from ppo import hierarchical_reconfiguration_gate

    null_payload = run_minimal_sanity(
        config_path,
        origin_start_index,
        1,
        forced_gate_action=hierarchical_reconfiguration_gate.GATE_ACTION_NULL,
        repeat_check=False,
    )
    configure_payload = run_minimal_sanity(
        config_path,
        origin_start_index,
        1,
        forced_gate_action=hierarchical_reconfiguration_gate.GATE_ACTION_CONFIGURE,
        force_noop_configuration=True,
        repeat_check=False,
    )
    policy = null_payload["policy"]
    if not isinstance(
        policy,
        hierarchical_reconfiguration_gate.HierarchicalReconfigurationGatePolicy,
    ):
        raise TypeError("当前正式配置没有构造 Hierarchical Gate 策略。")
    null_batch: RolloutBatch = null_payload["batch"]
    configure_batch: RolloutBatch = configure_payload["batch"]
    null_transition = null_batch.transitions[0]
    configure_transition = configure_batch.transitions[0]
    if (
        null_batch.gate_actions is None
        or null_batch.subpolicy_actions is None
        or configure_batch.gate_actions is None
        or configure_batch.subpolicy_actions is None
    ):
        raise AssertionError("分层 rollout 未保存 Gate/WHAT 动作字段。")
    null_action_ok = bool(
        int(null_transition.gate_action)
        == hierarchical_reconfiguration_gate.GATE_ACTION_NULL
        and not np.any(null_transition.action_vector)
    )
    configure_mapping_ok = bool(
        int(configure_transition.gate_action)
        == hierarchical_reconfiguration_gate.GATE_ACTION_CONFIGURE
        and np.array_equal(
            configure_transition.action_vector,
            configure_transition.subpolicy_action_vector,
        )
        and not np.any(configure_transition.action_vector)
    )
    finite_rollout_ok = bool(
        all(
            np.all(np.isfinite(array))
            for batch in (null_batch, configure_batch)
            for array in (
                batch.observations,
                batch.old_log_probs,
                batch.old_values,
                batch.rewards,
            )
        )
    )
    return {
        "policy_type": type(policy).__name__,
        "method": str(
            project_config.get_config_value(
                project_config.load_config(config_path),
                "algorithm.ppo.method",
            )
        ),
        "observation_dim": int(null_payload["observation_spec"].observation_dim),
        "action_dim": int(null_payload["action_spec"].action_dim),
        "executed_slots_per_branch": 1,
        "environment_steps_total": 2,
        "null_gate_action": int(null_transition.gate_action),
        "configure_gate_action": int(configure_transition.gate_action),
        "null_final_action": null_transition.action_vector.astype(int).tolist(),
        "configure_final_action": configure_transition.action_vector.astype(int).tolist(),
        "null_reward": float(null_transition.reward),
        "configure_noop_reward": float(configure_transition.reward),
        "same_noop_environment_result": bool(
            math.isclose(
                float(null_transition.reward),
                float(configure_transition.reward),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and np.array_equal(
                null_transition.next_cache_state,
                configure_transition.next_cache_state,
            )
            and null_transition.next_uav_positions
            == configure_transition.next_uav_positions
        ),
        "finite_rollout_ok": finite_rollout_ok,
        "null_maps_to_all_zero": null_action_ok,
        "configure_maps_to_what": configure_mapping_ok,
        "gate_fields_preserved": bool(
            null_batch.metadata["hierarchical_gate_rollout"]
            and configure_batch.metadata["hierarchical_gate_rollout"]
        ),
        "old_logprob_recompute_max_error": float(
            max(
                null_payload["old_logprob_recompute_max_error"],
                configure_payload["old_logprob_recompute_max_error"],
            )
        ),
        "old_value_recompute_max_error": float(
            max(
                null_payload["old_value_recompute_max_error"],
                configure_payload["old_value_recompute_max_error"],
            )
        ),
        "state_propagation_ok": bool(
            null_batch.metadata["state_propagation_ok"]
            and configure_batch.metadata["state_propagation_ok"]
        ),
        "optimizer_steps": 0,
        "backward_calls": 0,
    }


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    batch: RolloutBatch = payload["batch"]
    return {
        "gamma": payload["gamma"],
        "gae_lambda": payload["gae_lambda"],
        "collected_steps": int(len(batch.transitions)),
        "alignment_steps_prepared": payload["alignment_steps_prepared"],
        "observation_shape": list(batch.observations.shape),
        "actions_shape": list(batch.actions.shape),
        "old_log_probs_shape": list(batch.old_log_probs.shape),
        "old_values_shape": list(batch.old_values.shape),
        "rewards_shape": list(batch.rewards.shape),
        "dones_shape": list(batch.dones.shape),
        "advantages_shape": list(batch.advantages.shape),
        "returns_shape": list(batch.returns.shape),
        "origin_indices": batch.origin_indices.astype(int).tolist(),
        "slots": batch.slots.astype(int).tolist(),
        "times_s": [transition.time_s for transition in batch.transitions],
        "actions": [transition.action_vector.astype(int).tolist() for transition in batch.transitions],
        "rewards": batch.rewards.astype(float).tolist(),
        "actual_task_utility": [transition.actual_task_utility for transition in batch.transitions],
        "cache_cost": [transition.cache_cost for transition in batch.transitions],
        "movement_cost": [transition.movement_cost for transition in batch.transitions],
        "step_feasible": [transition.step_feasible for transition in batch.transitions],
        "truncated": batch.truncated,
        "true_terminal": batch.metadata["true_terminal"],
        "bootstrap_value": batch.final_bootstrap_value,
        "advantages": batch.advantages.astype(float).tolist(),
        "returns": batch.returns.astype(float).tolist(),
        "manual_gae_max_error": payload["manual_gae_max_error"],
        "old_logprob_recompute_max_error": payload["old_logprob_recompute_max_error"],
        "old_value_recompute_max_error": payload["old_value_recompute_max_error"],
        "state_propagation_ok": batch.metadata["state_propagation_ok"],
        "reward_invariant_ok": batch.metadata["reward_invariant_ok"],
        "repeated_rollout_deterministic": payload["repeated_rollout_deterministic"],
        "formal_validation": payload["formal_validation"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect a small on-policy rollout and compute GAE.")
    parser.add_argument("--config", default="system_config.toml")
    parser.add_argument("--origin-start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--hierarchical-e2e",
        action="store_true",
        help="仅执行一次分层策略最小端到端验证，不训练。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.hierarchical_e2e:
            summary = run_hierarchical_minimal_e2e_validation(
                args.config,
                args.origin_start_index,
            )
        else:
            payload = run_minimal_sanity(
                args.config,
                args.origin_start_index,
                args.max_steps,
            )
            summary = _summary(payload)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except InfeasibleActionBlocker as exc:
        print(str(exc))
        return 2
    except Exception as exc:
        print(f"错误：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
