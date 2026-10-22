"""独立的候选条件化 Double DQN 训练循环。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

from .agent import DoubleDQNAgent, linear_epsilon_schedule
from .environment_adapter import ExclusiveDQNEnvironment
from .replay_buffer import ReplayBuffer


def _summarize_update_metrics(
    update_rows: list[dict[str, float]],
) -> dict[str, Any]:
    """汇总可用的数值指标，并显式保留缺项和非有限值数量。"""

    metric_names = (
        "td_loss",
        "gradient_norm_before_clip",
        "q_value_mean",
        "td_target_mean",
    )
    summary: dict[str, Any] = {"update_count": len(update_rows)}
    for name in metric_names:
        values: list[float] = []
        missing_count = 0
        nonfinite_count = 0
        for row in update_rows:
            if name not in row or row[name] is None:
                missing_count += 1
                continue
            value = float(row[name])
            if not math.isfinite(value):
                nonfinite_count += 1
                continue
            values.append(value)
        summary[name] = {
            "count": len(values),
            "missing_count": missing_count,
            "nonfinite_count": nonfinite_count,
            "mean": sum(values) / len(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    sync_values = [row.get("target_network_synced") for row in update_rows]
    observed_sync_values = [value for value in sync_values if value is not None]
    summary["target_network_sync_count"] = (
        sum(bool(value) for value in observed_sync_values)
        if observed_sync_values
        else None
    )
    summary["target_network_sync_missing_count"] = (
        len(sync_values) - len(observed_sync_values)
    )
    summary["target_network_sync_gradient_steps"] = [
        int(row["gradient_step"])
        for row in update_rows
        if bool(row.get("target_network_synced", False))
        and row.get("gradient_step") is not None
    ]
    return summary


def _summarize_step_durations(step_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """只统计实际记录的转移时长，不为缺失字段补默认值。"""

    durations: list[int] = []
    missing_count = 0
    for row in step_rows:
        if "duration_slots" not in row or row["duration_slots"] is None:
            missing_count += 1
            continue
        value = row["duration_slots"]
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ValueError(f"训练记录中的 duration_slots 非法：{value}")
        durations.append(int(value))
    return {
        "recorded_count": len(durations),
        "missing_count": missing_count,
        "total_slots": sum(durations) if durations else None,
        "mean_slots": sum(durations) / len(durations) if durations else None,
    }


@dataclass(frozen=True)
class TrainerConfig:
    """单独于正式 MAPPO 配置的 DQN 训练控制项。"""

    batch_size: int = 64
    learning_starts: int = 1000
    train_frequency: int = 1
    gradient_updates_per_step: int = 1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10000

    def __post_init__(self) -> None:
        if min(
            int(self.batch_size),
            int(self.train_frequency),
            int(self.gradient_updates_per_step),
            int(self.epsilon_decay_steps),
        ) <= 0:
            raise ValueError("训练批量、更新间隔、更新次数和探索衰减步数必须为正")
        if int(self.learning_starts) < 0:
            raise ValueError("learning_starts 不能为负数")
        if not 0.0 <= float(self.epsilon_end) <= float(self.epsilon_start) <= 1.0:
            raise ValueError("epsilon 配置不合法")


def train_episode(
    environment: ExclusiveDQNEnvironment,
    agent: DoubleDQNAgent,
    replay_buffer: ReplayBuffer,
    *,
    config: TrainerConfig | None = None,
    global_step_start: int = 0,
    on_step: Callable[[dict[str, Any]], None] | None = None,
    on_update: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, Any]:
    """在一个真实 Episode 窗口中采集实际奖励并按回放批次更新。"""

    settings = config or TrainerConfig()
    environment.reset()
    global_step = int(global_step_start)
    replay_writes_start = int(replay_buffer.write_count)
    replay_samples_start = int(replay_buffer.sample_count)
    replay_sample_calls_start = int(replay_buffer.sample_calls)
    episode_reward = 0.0
    operation_names = ("KEEP_STAY", "CACHE_UPDATE", "UAV_MOVE")
    action_counts = {operation: 0 for operation in operation_names}
    exploration_counts = {operation: 0 for operation in operation_names}
    greedy_selection_counts = {operation: 0 for operation in operation_names}
    execution_counts = {operation: 0 for operation in operation_names}
    update_rows: list[dict[str, float]] = []
    step_rows: list[dict[str, Any]] = []
    while environment.can_act:
        step_started = time.perf_counter()
        state, candidates = environment.observe()
        epsilon = linear_epsilon_schedule(
            settings.epsilon_start,
            settings.epsilon_end,
            settings.epsilon_decay_steps,
            global_step,
        )
        selected = agent.select_action(
            state,
            candidates,
            epsilon=epsilon,
            explore=True,
        )
        selected_operation = selected.candidate.action.operation
        if selected_operation not in action_counts:
            raise RuntimeError(f"智能体返回未知操作类别：{selected_operation}")
        candidate_counts = {
            operation: sum(
                candidate.action.operation == operation for candidate in candidates
            )
            for operation in operation_names
        }
        if not candidate_counts["KEEP_STAY"]:
            raise RuntimeError("当前合法候选缺少 KEEP_STAY")
        if selected.explored:
            exploration_counts[selected_operation] += 1
        else:
            greedy_selection_counts[selected_operation] += 1
        previous_cache = environment.cache_state.copy()
        previous_uav = {
            key: value.copy() for key, value in environment.uav_positions.items()
        }
        action_cache = selected.candidate.action.cache_array()
        action_uav = selected.candidate.action.uav_dict()
        service_ids = tuple(
            int(value) for value in environment.model.service_catalog["service_ids"]
        )
        node_ids = tuple(
            str(value) for value in environment.model.cache_nodes["node_ids"]
        )
        cache_changes_by_node = {}
        for node_index, node_id in enumerate(node_ids):
            removed_services = [
                service_ids[index]
                for index in range(len(service_ids))
                if previous_cache[node_index, index] and not action_cache[node_index, index]
            ]
            added_services = [
                service_ids[index]
                for index in range(len(service_ids))
                if action_cache[node_index, index] and not previous_cache[node_index, index]
            ]
            if removed_services or added_services:
                cache_changes_by_node[node_id] = {
                    "removed_service_ids": removed_services,
                    "added_service_ids": added_services,
                }
        cache_removed = int((previous_cache > action_cache).sum())
        cache_added = int((action_cache > previous_cache).sum())
        uav_move_distances = {
            key: float(
                math.dist(previous_uav[key].tolist(), action_uav[key].tolist())
            )
            for key in previous_uav
        }
        transition = environment.step(selected.candidate)
        if (
            transition.candidate_id != selected.candidate.action.candidate_id
            or transition.operation != selected_operation
        ):
            raise RuntimeError("实际执行转移与选择的合法候选不一致")
        replay_buffer.add(
            state_features=state,
            action=selected.candidate.action,
            action_features=selected.candidate.features,
            reward=transition.reward,
            next_state_features=transition.next_state_features,
            next_state_fingerprint=transition.next_state_fingerprint,
            next_legal_candidates=transition.next_legal_candidates,
            terminated=transition.terminated,
            duration_slots=transition.duration_slots,
        )
        next_candidate_counts = {
            operation: sum(
                item.action.operation == operation
                for item in transition.next_legal_candidates
            )
            for operation in operation_names
        }
        episode_reward += transition.reward
        action_counts[transition.operation] += 1
        execution_counts[transition.operation] += 1
        step_rows.append(
            {
                "slot_id": int(transition.diagnostics["slot_id"]),
                "operation": transition.operation,
                "candidate_id": transition.candidate_id,
                "state_fingerprint": selected.candidate.action.source_state_fingerprint,
                "next_state_fingerprint": transition.next_state_fingerprint,
                "next_candidate_counts_by_operation": next_candidate_counts,
                "candidate_count": sum(candidate_counts.values()),
                "reward": transition.reward,
                "duration_slots": int(transition.duration_slots),
                "epsilon": epsilon,
                "global_step": global_step,
                "candidate_counts_by_operation": candidate_counts,
                "candidate_q_statistics_by_operation": selected.q_statistics_by_operation,
                "selected_q_value": selected.q_value,
                "selected_candidate_action": selected.candidate.action.to_dict(),
                "explored": bool(selected.explored),
                "selection_mode": "exploration" if selected.explored else "greedy",
                "cache_changed_node_ids": list(selected.candidate.action.changed_node_ids),
                "cache_changes_by_node": cache_changes_by_node,
                "cache_services_removed": cache_removed,
                "cache_services_added": cache_added,
                "cache_service_replacements": min(cache_removed, cache_added),
                "uav_move_distances_m": uav_move_distances,
                "selected_target_uav_positions": {
                    key: value.tolist() for key, value in action_uav.items()
                },
                "uav_total_move_distance_m": sum(uav_move_distances.values()),
                "cache_download_cost": float(transition.diagnostics["cache_cost"]),
                "uav_movement_cost": float(transition.diagnostics["movement_cost"]),
                "decision_wall_time_s": 0.0,
                "executor_calls": transition.executor_calls,
                "matching_calls": transition.matching_calls,
                "kkt_solver_calls": transition.kkt_solver_calls,
                "executor_performance_profile": transition.diagnostics.get(
                    "executor_performance_profile", {}
                ),
                "cache_candidate_generation_diagnostics": transition.diagnostics.get(
                    "cache_candidate_generation_diagnostics", {}
                ),
                "uav_candidate_generation_diagnostics": transition.diagnostics.get(
                    "uav_candidate_generation_diagnostics", {}
                ),
                "candidate_generation_complete_within_bounded_proposal_policy": transition.diagnostics.get(
                    "candidate_generation_complete_within_bounded_proposal_policy"
                ),
                "candidate_legal_action_space_exhaustive": transition.diagnostics.get(
                    "candidate_legal_action_space_exhaustive"
                ),
                "dqn_adapter_additional_candidate_truncation": transition.diagnostics.get(
                    "dqn_adapter_additional_candidate_truncation"
                ),
                "mutual_exclusion_pass": transition.mutual_exclusion_pass,
                "state_isolation_pass": transition.state_isolation_pass,
                "state_features_used_future_actual_data": transition.diagnostics.get(
                    "state_features_used_future_actual_data"
                ),
                "replay_size": len(replay_buffer),
                "replay_capacity": replay_buffer.capacity,
                "replay_write_count": replay_buffer.write_count,
                "replay_sample_calls": replay_buffer.sample_calls,
                "replay_samples_drawn": replay_buffer.sample_count,
                "nonfinite_value_detected": False,
                "illegal_action_detected": False,
            }
        )
        if (
            global_step >= int(settings.learning_starts)
            and len(replay_buffer) >= int(settings.batch_size)
            and global_step % int(settings.train_frequency) == 0
        ):
            for _ in range(int(settings.gradient_updates_per_step)):
                update = agent.update(replay_buffer.sample(int(settings.batch_size)))
                update.update(
                    {
                        "global_step": global_step,
                        "slot_id": int(transition.diagnostics["slot_id"]),
                        "epsilon": float(epsilon),
                        "replay_size": len(replay_buffer),
                        "replay_sample_count": replay_buffer.sample_count,
                    }
                )
                update_rows.append(update)
                if any(
                    not math.isfinite(float(update[name]))
                    for name in (
                        "td_loss",
                        "gradient_norm_before_clip",
                        "q_value_mean",
                        "td_target_mean",
                    )
                ):
                    raise FloatingPointError("Double DQN 更新产生非有限指标")
                if on_update is not None:
                    on_update(update)
        step_rows[-1]["decision_wall_time_s"] = float(
            time.perf_counter() - step_started
        )
        step_rows[-1]["gradient_update_count"] = int(agent.gradient_steps)
        step_rows[-1]["target_network_sync_count"] = int(
            agent.gradient_steps // agent.target_network_frequency
        )
        step_rows[-1]["target_network_sync_mode"] = (
            "polyak_interpolation" if agent.target_tau < 1.0 else "hard_copy"
        )
        step_rows[-1]["target_network_sync_gradient_step_latest"] = (
            int(agent.gradient_steps // agent.target_network_frequency)
            * agent.target_network_frequency
        )
        if on_step is not None:
            on_step(step_rows[-1])
        global_step += 1
    selection_counts = {
        operation: exploration_counts[operation]
        + greedy_selection_counts[operation]
        for operation in operation_names
    }
    update_summary = _summarize_update_metrics(update_rows)
    duration_summary = _summarize_step_durations(step_rows)
    slot_count = len(step_rows)
    episode_summary = {
        "cumulative_system_utility": float(episode_reward),
        "mean_system_utility_per_slot": (
            float(episode_reward) / slot_count if slot_count else None
        ),
        "cache_update_count": execution_counts["CACHE_UPDATE"],
        "cache_changed_node_count": sum(
            len(row["cache_changed_node_ids"]) for row in step_rows
        ),
        "cache_service_replacement_count": sum(
            int(row["cache_service_replacements"]) for row in step_rows
        ),
        "cache_services_removed": sum(
            int(row["cache_services_removed"]) for row in step_rows
        ),
        "cache_services_added": sum(
            int(row["cache_services_added"]) for row in step_rows
        ),
        "uav_move_count": execution_counts["UAV_MOVE"],
        "uav_total_move_distance_m": sum(
            float(row["uav_total_move_distance_m"]) for row in step_rows
        ),
        "cache_download_cost": sum(
            float(row["cache_download_cost"]) for row in step_rows
        ),
        "uav_movement_cost": sum(
            float(row["uav_movement_cost"]) for row in step_rows
        ),
        "decision_wall_time_s": sum(
            float(row["decision_wall_time_s"]) for row in step_rows
        ),
        "executor_calls": sum(int(row["executor_calls"]) for row in step_rows),
        "matching_calls": sum(int(row["matching_calls"]) for row in step_rows),
        "kkt_solver_calls": sum(int(row["kkt_solver_calls"]) for row in step_rows),
        "mutual_exclusion_violations": sum(
            not bool(row["mutual_exclusion_pass"]) for row in step_rows
        ),
        "state_isolation_failures": sum(
            not bool(row["state_isolation_pass"]) for row in step_rows
        ),
        "nonfinite_update_metric_count": sum(
            int(
                any(
                    not math.isfinite(float(row[name]))
                    for name in (
                        "td_loss",
                        "gradient_norm_before_clip",
                        "q_value_mean",
                        "td_target_mean",
                    )
                )
            )
            for row in update_rows
        ),
    }
    return {
        "episode_reward": episode_reward,
        "step_count": len(step_rows),
        "action_counts": action_counts,
        "exploration_counts_by_operation": exploration_counts,
        "greedy_selection_counts_by_operation": greedy_selection_counts,
        "selection_counts_by_operation": selection_counts,
        "execution_counts_by_operation": execution_counts,
        "steps": step_rows,
        "updates": update_rows,
        "update_summary": update_summary,
        "target_network_sync_policy": {
            "counter_unit": "gradient_update",
            "interval": int(agent.target_network_frequency),
            "tau": float(agent.target_tau),
            "mode": "polyak_interpolation" if agent.target_tau < 1.0 else "hard_copy",
        },
        "duration_summary": duration_summary,
        "episode_summary": episode_summary,
        "epsilon_schedule": {
            "start": float(settings.epsilon_start),
            "end": float(settings.epsilon_end),
            "decay_steps": int(settings.epsilon_decay_steps),
            "counter_unit": "environment_decision_slot",
            "global_step_start": int(global_step_start),
            "global_step_end": global_step,
            "last_action_epsilon": (
                float(step_rows[-1]["epsilon"]) if step_rows else None
            ),
        },
        "global_step_end": global_step,
        "replay_size": len(replay_buffer),
        "replay_capacity": replay_buffer.capacity,
        "replay_write_count": replay_buffer.write_count,
        "replay_writes_this_episode": replay_buffer.write_count - replay_writes_start,
        "replay_sample_calls": replay_buffer.sample_calls,
        "replay_sample_calls_this_episode": replay_buffer.sample_calls
        - replay_sample_calls_start,
        "replay_samples_drawn": replay_buffer.sample_count,
        "replay_samples_drawn_this_episode": replay_buffer.sample_count
        - replay_samples_start,
        "executor_calls": environment.executor_calls,
        "matching_calls": environment.matching_calls,
        "kkt_solver_calls": environment.kkt_solver_calls,
        "training_status": (
            "learning_updates_performed" if update_rows else "transitions_collected_no_gradient_update"
        ),
        "episode_terminated": environment.terminated,
        "window_truncated": not environment.terminated,
    }


__all__ = ["TrainerConfig", "train_episode"]
