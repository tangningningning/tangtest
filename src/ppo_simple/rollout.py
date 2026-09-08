"""Environment-facing rollout and GAE for the minimal PPO v2 trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from decision import slot_environment
from ppo import ppo_rollout, ppo_trigger_policy

from .policy import BernoulliActorCritic


@dataclass(frozen=True)
class SimpleTransition:
    observation: np.ndarray
    action: np.ndarray
    action_mask: np.ndarray
    old_log_prob: float
    old_value: float
    reward: float
    done: bool
    slot: int
    origin_index: int
    actual_task_utility: float
    cache_cost: float
    movement_cost: float
    actual_cache_update_count: int
    actual_uav_movement_count: int
    next_cache_state: np.ndarray
    next_uav_positions: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class SimpleRollout:
    transitions: tuple[SimpleTransition, ...]
    observations: np.ndarray
    actions: np.ndarray
    action_masks: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    next_cache_state: np.ndarray
    next_uav_positions: dict[str, tuple[float, float]]
    bootstrap_observation: np.ndarray | None


def _copy_uav_positions(positions: dict[str, Any]) -> dict[str, tuple[float, float]]:
    return {
        str(uav_id): (float(np.asarray(xy)[0]), float(np.asarray(xy)[1]))
        for uav_id, xy in positions.items()
    }


def _observation(
    previous_cache: np.ndarray,
    previous_uav: dict[str, Any],
    step: Any,
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


def _check_reward(result: slot_environment.SlotExecutionResult) -> float:
    if result.system_utility is None:
        raise AssertionError("feasible step 缺少 system_utility。")
    expected = float(result.actual_task_utility) - float(result.cache_cost) - float(result.movement_cost)
    if not np.isclose(float(result.system_utility), expected, rtol=1e-12, atol=1e-12):
        raise AssertionError("PPO v2 reward 不是 execute_slot().system_utility。")
    return float(result.system_utility)


def _check_request_contract(
    previous_cache: np.ndarray,
    previous_uav: dict[str, Any],
    result: slot_environment.SlotExecutionResult,
    action_spec: ppo_trigger_policy.TriggerActionSpec,
) -> None:
    if result.next_cache_state is None or result.next_uav_positions is None:
        raise AssertionError("可行 environment transition 缺少 next state。")
    next_cache = np.asarray(result.next_cache_state, dtype=np.int8)
    for node_index, request in enumerate(result.input_cache_triggers):
        if int(request) == 0 and not np.array_equal(previous_cache[node_index], next_cache[node_index]):
            raise AssertionError("request=0 但 cache actual state 发生变化。")
    for uav_id in action_spec.uav_ids:
        if int(result.input_uav_triggers[uav_id]) != 0:
            continue
        before = np.asarray(previous_uav[uav_id], dtype=np.float64)
        after = np.asarray(result.next_uav_positions[uav_id], dtype=np.float64)
        if not np.allclose(before, after, rtol=0.0, atol=1.0e-6):
            raise AssertionError("request=0 但 UAV actual state 发生变化。")


def collect_rollout(
    policy: BernoulliActorCritic,
    context: Any,
    aligned_steps: list[Any],
    *,
    initial_cache: np.ndarray | None = None,
    initial_uav: dict[str, Any] | None = None,
    bootstrap_step: Any | None = None,
    terminal_origin_index: int | None = None,
) -> SimpleRollout:
    """Collect one contiguous request rollout from a supplied environment state.

    A rollout chunk is a truncation boundary, not an environment reset.  The
    caller supplies the state produced by the previous chunk and, when the
    chunk is not terminal, the next aligned step used to build the critic
    bootstrap observation.
    """

    if not aligned_steps:
        raise ValueError("aligned_steps 不能为空。")
    execution_steps = aligned_steps
    if terminal_origin_index is None:
        terminal_origin_index = int(execution_steps[-1].origin_index)
    terminal_origin_index = int(terminal_origin_index)
    if int(execution_steps[-1].origin_index) > terminal_origin_index:
        raise ValueError("aligned_steps 不能超过 true episode terminal。")

    cache_source = context.initial_cache if initial_cache is None else initial_cache
    previous_cache = np.asarray(cache_source, dtype=np.int8).copy()
    if previous_cache.shape != np.asarray(context.initial_cache).shape:
        raise ValueError("initial_cache shape 与 environment context 不一致。")
    uav_source = context.initial_uav if initial_uav is None else initial_uav
    previous_uav = _copy_uav_positions(uav_source)
    transitions: list[SimpleTransition] = []
    for step_index, step in enumerate(execution_steps):
        observation = _observation(
            previous_cache,
            previous_uav,
            step,
            context.observation_spec,
            context.flight_boundary,
        )
        if observation.shape != (int(context.observation_spec.observation_dim),):
            raise AssertionError("PPO v2 observation 不符合 observation_spec。")
        action_mask_provider = ppo_rollout.build_trigger_action_mask_provider(
            previous_cache,
            previous_uav,
            context.service_catalog,
            context.cfg,
            context.cache_nodes,
            context.flight_boundary,
            context.action_spec,
        )
        sample = policy.sample(observation, action_mask=action_mask_provider)
        if sample.action.shape != (int(context.action_spec.action_dim),):
            raise AssertionError("PPO v2 action 不符合 action_spec。")
        cache_triggers, uav_triggers = context.action_spec.action_vector_to_triggers(sample.action)
        result = slot_environment.execute_slot(
            previous_cache,
            previous_uav,
            cache_triggers,
            uav_triggers,
            step.scenario_slice,
            step.actual_tasks,
            step.actual_vehicle_positions,
            context.local_cache,
            context.service_catalog,
            context.cfg,
            context.cache_nodes,
            context.cache_cfg,
            context.flight_boundary,
            slot_id=step.slot,
            time_s=step.time_s,
        )
        if not result.step_feasible:
            raise RuntimeError(
                "PPO v2 request produced an infeasible environment transition: "
                f"origin={step.origin_index}, slot={step.slot}, reason={result.infeasible_reason}"
            )
        _check_request_contract(previous_cache, previous_uav, result, context.action_spec)
        reward = _check_reward(result)
        done = int(step.origin_index) == terminal_origin_index
        if done and step_index != len(execution_steps) - 1:
            raise ValueError("true episode terminal 必须是 rollout 的最后一步。")
        next_cache = np.asarray(result.next_cache_state, dtype=np.int8).copy()
        next_uav = _copy_uav_positions(result.next_uav_positions)
        actual_cache_update_count = int(
            np.any(next_cache != previous_cache, axis=1).sum()
        )
        actual_uav_movement_count = sum(
            int(
                np.linalg.norm(
                    np.asarray(next_uav[uav_id], dtype=np.float64)
                    - np.asarray(previous_uav[uav_id], dtype=np.float64)
                )
                > 1.0e-6
            )
            for uav_id in context.action_spec.uav_ids
        )
        transitions.append(
            SimpleTransition(
                observation=np.asarray(observation, dtype=np.float32).copy(),
                action=np.asarray(sample.action, dtype=np.int8).copy(),
                action_mask=np.asarray(sample.action_mask, dtype=bool).copy(),
                old_log_prob=float(sample.log_prob.detach().cpu()),
                old_value=float(sample.value.detach().cpu()),
                reward=reward,
                done=done,
                slot=int(step.slot),
                origin_index=int(step.origin_index),
                actual_task_utility=float(result.actual_task_utility),
                cache_cost=float(result.cache_cost),
                movement_cost=float(result.movement_cost),
                actual_cache_update_count=actual_cache_update_count,
                actual_uav_movement_count=actual_uav_movement_count,
                next_cache_state=next_cache.copy(),
                next_uav_positions=dict(next_uav),
            )
        )
        previous_cache = next_cache
        previous_uav = next_uav

    bootstrap_observation: np.ndarray | None = None
    if not transitions[-1].done:
        if bootstrap_step is None:
            raise ValueError("非 terminal rollout chunk 缺少 bootstrap step。")
        if int(bootstrap_step.origin_index) != int(execution_steps[-1].origin_index) + 1:
            raise ValueError("bootstrap step 必须紧跟 rollout chunk 的最后 origin。")
        bootstrap_observation = _observation(
            previous_cache,
            previous_uav,
            bootstrap_step,
            context.observation_spec,
            context.flight_boundary,
        )

    return SimpleRollout(
        transitions=tuple(transitions),
        observations=np.stack([item.observation for item in transitions]).astype(np.float32),
        actions=np.stack([item.action for item in transitions]).astype(np.int8),
        action_masks=np.stack([item.action_mask for item in transitions]).astype(bool),
        old_log_probs=np.asarray([item.old_log_prob for item in transitions], dtype=np.float32),
        old_values=np.asarray([item.old_value for item in transitions], dtype=np.float32),
        rewards=np.asarray([item.reward for item in transitions], dtype=np.float32),
        dones=np.asarray([item.done for item in transitions], dtype=bool),
        next_cache_state=previous_cache.copy(),
        next_uav_positions=dict(previous_uav),
        bootstrap_observation=bootstrap_observation,
    )


def compute_gae(
    rewards: Any,
    values: Any,
    dones: Any,
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute standard GAE and returns without any environment-specific terms."""

    rewards_a = np.asarray(rewards, dtype=np.float32).reshape(-1)
    values_a = np.asarray(values, dtype=np.float32).reshape(-1)
    dones_a = np.asarray(dones, dtype=bool).reshape(-1)
    if not (len(rewards_a) == len(values_a) == len(dones_a)) or not len(rewards_a):
        raise ValueError("GAE input 长度必须一致且非空。")
    if not np.all(np.isfinite(rewards_a)) or not np.all(np.isfinite(values_a)):
        raise ValueError("GAE input 包含 NaN/Inf。")
    advantages = np.zeros_like(rewards_a, dtype=np.float32)
    next_advantage = 0.0
    for index in range(len(rewards_a) - 1, -1, -1):
        next_value = float(bootstrap_value) if index == len(rewards_a) - 1 else float(values_a[index + 1])
        nonterminal = 0.0 if bool(dones_a[index]) else 1.0
        delta = float(rewards_a[index]) + float(gamma) * nonterminal * next_value - float(values_a[index])
        next_advantage = delta + float(gamma) * float(gae_lambda) * nonterminal * next_advantage
        advantages[index] = next_advantage
    returns = advantages + values_a
    if not np.all(np.isfinite(advantages)) or not np.all(np.isfinite(returns)):
        raise ValueError("GAE result 包含 NaN/Inf。")
    return advantages.astype(np.float32), returns.astype(np.float32)
