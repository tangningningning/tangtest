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
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common import project_config
from decision import cache_configuration, task_allocation, slot_environment, uav_configuration
from ppo import ppo_trigger_policy
from scenario import task_scenario


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


@dataclass(frozen=True)
class RolloutBatch:
    observations: np.ndarray
    actions: np.ndarray
    action_masks: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    slots: np.ndarray
    origin_indices: np.ndarray
    final_bootstrap_value: float
    truncated: bool
    transitions: list[RolloutTransition]
    metadata: dict[str, Any]


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
        if int(trigger) == 1 and equal:
            raise AssertionError(f"z_ca=1 但 cache node {index} 未改变。")
    final_uav = result.next_uav_positions or {}
    for uav_id in action_spec.uav_ids:
        trigger = int(result.input_uav_triggers[uav_id])
        changed = _changed_uav(previous_uav, final_uav, uav_id)
        if trigger == 0 and changed:
            raise AssertionError(f"z_q=0 但 {uav_id} 移动。")
        if trigger == 1 and not changed:
            raise AssertionError(f"z_q=1 但 {uav_id} 未移动。")


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

    Cache trigger 的 distinct feasibility 只依赖本节点上一状态和容量；UAV
    trigger 则按 policy 当前的 UAV 顺序逐步检查几何 candidate assignment。
    这里不调用 objective，因此 objective 选择造成的 residual infeasibility
    仍由 InfeasibleActionBlocker 负责暴露。
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

    cache_trigger_allowed = np.ones(len(cache_node_ids), dtype=bool)
    service_sizes = np.asarray(service_catalog["service_sizes_mb"], dtype=np.float64)
    for node_index in range(len(cache_node_ids)):
        neighbors = cache_configuration.generate_node_neighbors(
            previous_cache[node_index],
            service_sizes,
            float(cache_nodes["capacities_mb"][node_index]),
        )
        cache_trigger_allowed[node_index] = any(
            not np.array_equal(candidate, previous_cache[node_index])
            for _, candidate in neighbors
        )

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
            return np.asarray([True, bool(cache_trigger_allowed[index])], dtype=bool)

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
    state_propagation_ok = True

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
        sample = policy.sample_action(
            observation,
            deterministic=False,
            action_mask=action_mask_provider,
        )
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
        )
        transitions.append(transition)
        action_masks.append(sample.per_action_masks.copy())
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
    return RolloutBatch(
        observations=observations,
        actions=actions,
        action_masks=np.stack(action_masks).astype(bool),
        old_log_probs=old_log_probs,
        old_values=old_values,
        rewards=rewards,
        dones=dones,
        advantages=advantages,
        returns=returns,
        slots=np.asarray([item.slot for item in transitions], dtype=np.int64),
        origin_indices=np.asarray([item.origin_index for item in transitions], dtype=np.int64),
        final_bootstrap_value=float(bootstrap_value),
        truncated=truncated,
        transitions=transitions,
        metadata={
            "true_terminal": true_terminal,
            "state_propagation_ok": state_propagation_ok,
            "reward_invariant_ok": True,
        },
    )


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
) -> list[AlignedRolloutStep]:
    """从正式 artifact 构造已对齐 steps；不重新预测/生成场景。"""

    if max_steps <= 0:
        raise ValueError("max_steps 必须大于 0。")
    vehicle_pred = _load_vehicle_prediction_h0(vehicle_prediction_file)
    popularity_pred = _load_popularity_prediction_h0(popularity_prediction_file)
    task_time_by_slot = _load_task_time_by_slot(tasks_file)
    n_origin = int(scenarios["origin_slots"].shape[0])
    if origin_start_index < 0 or origin_start_index >= n_origin:
        raise IndexError("origin_start_index 越界。")

    # 多准备一步，用于 rollout truncation bootstrap。
    stop = min(n_origin, origin_start_index + max_steps + 1)
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
    evaluation = policy.evaluate_actions(
        batch.observations,
        batch.actions,
        action_masks=batch.action_masks,
    )
    log_probs = evaluation.joint_log_prob.detach().cpu().numpy()
    values = evaluation.value.detach().cpu().numpy()
    return (
        float(np.max(np.abs(log_probs - batch.old_log_probs))),
        float(np.max(np.abs(values - batch.old_values))),
    )


def run_minimal_sanity(config_path: str = "system_config.toml", origin_start_index: int = 0, max_steps: int = 8) -> dict[str, Any]:
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
    policy = ppo_trigger_policy.ConditionalTriggerActorCritic(
        observation_spec.observation_dim,
        action_spec,
        **ppo_trigger_policy.ppo_network_config_from_project(cfg),
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
    )
    logprob_error, value_error = _batch_consistency(policy, batch)

    # 重复 sanity：重新初始化同一 policy weights + 同一 RNG，动作和reward应一致。
    ppo_trigger_policy.set_torch_seed(seed)
    np.random.seed(seed)
    policy_repeat = ppo_trigger_policy.ConditionalTriggerActorCritic(
        observation_spec.observation_dim,
        action_spec,
        **ppo_trigger_policy.ppo_network_config_from_project(cfg),
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
        "formal_validation": True,
        "alignment_steps_prepared": len(aligned_steps),
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run_minimal_sanity(args.config, args.origin_start_index, args.max_steps)
        print(json.dumps(_summary(payload), ensure_ascii=False, indent=2))
        return 0
    except InfeasibleActionBlocker as exc:
        print(str(exc))
        return 2
    except Exception as exc:
        print(f"错误：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
