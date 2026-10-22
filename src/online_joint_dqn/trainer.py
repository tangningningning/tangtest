"""独立的候选条件化 Double DQN 训练循环。"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .agent import DoubleDQNAgent, linear_epsilon_schedule
from .environment_adapter import ExclusiveDQNEnvironment
from .replay_buffer import ReplayBuffer


def _summarize_update_metrics(
    update_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """汇总可用的数值指标，并显式保留缺项和非有限值数量。"""

    metric_names = (
        "td_loss",
        "category_td_loss",
        "candidate_advantage_loss",
        "gradient_norm_before_clip",
        "q_value_mean",
        "category_q_value_mean",
        "candidate_q_value_mean",
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


def _numeric_summary(values: list[Any]) -> dict[str, Any]:
    recorded: list[float] = []
    missing_count = 0
    nonfinite_count = 0
    for raw in values:
        if raw is None:
            missing_count += 1
            continue
        value = float(raw)
        if not math.isfinite(value):
            nonfinite_count += 1
            continue
        recorded.append(value)
    return {
        "count": len(recorded),
        "missing_count": missing_count,
        "nonfinite_count": nonfinite_count,
        "mean": float(np.mean(recorded)) if recorded else None,
        "std": float(np.std(recorded)) if recorded else None,
        "min": min(recorded) if recorded else None,
        "max": max(recorded) if recorded else None,
    }


def _aggregate_candidate_q_statistics(
    step_rows: list[dict[str, Any]], operation_names: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for operation in operation_names:
        count = 0
        sum_values = 0.0
        sum_squares = 0.0
        missing_states = 0
        minima: list[float] = []
        maxima: list[float] = []
        for row in step_rows:
            stats = row.get("candidate_q_statistics_by_operation", {}).get(operation)
            if not isinstance(stats, Mapping) or not int(stats.get("count", 0)):
                missing_states += 1
                continue
            state_count = int(stats["count"])
            mean = float(stats["mean"])
            std = float(stats["std"])
            count += state_count
            sum_values += state_count * mean
            sum_squares += state_count * (std * std + mean * mean)
            minima.append(float(stats["min"]))
            maxima.append(float(stats["max"]))
        mean_value = sum_values / count if count else None
        variance = (
            max(0.0, sum_squares / count - mean_value * mean_value)
            if count and mean_value is not None
            else None
        )
        result[operation] = {
            "candidate_state_count": count,
            "state_count_without_candidates": missing_states,
            "mean": mean_value,
            "std": math.sqrt(variance) if variance is not None else None,
            "min": min(minima) if minima else None,
            "max": max(maxima) if maxima else None,
        }
    return result


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
    on_update: Callable[[dict[str, Any]], None] | None = None,
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
    update_rows: list[dict[str, Any]] = []
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
        action_uav = selected.candidate.action.uav_dict()
        service_ids = tuple(
            int(value) for value in environment.model.service_catalog["service_ids"]
        )
        node_ids = tuple(
            str(value) for value in environment.model.cache_nodes["node_ids"]
        )
        transition = environment.step(selected.candidate)
        if (
            transition.candidate_id != selected.candidate.action.candidate_id
            or transition.operation != selected_operation
        ):
            raise RuntimeError("实际执行转移与选择的合法候选不一致")
        actual_cache = np.asarray(transition.final_cache_state, dtype=np.int8)
        cache_changes_by_node: dict[str, dict[str, list[int]]] = {}
        cache_services_added: dict[str, list[int]] = {}
        cache_services_removed: dict[str, list[int]] = {}
        changed_node_ids: list[str] = []
        for node_index, node_id in enumerate(node_ids):
            removed_services = [
                service_ids[index]
                for index in range(len(service_ids))
                if previous_cache[node_index, index] and not actual_cache[node_index, index]
            ]
            added_services = [
                service_ids[index]
                for index in range(len(service_ids))
                if actual_cache[node_index, index] and not previous_cache[node_index, index]
            ]
            if removed_services or added_services:
                changed_node_ids.append(node_id)
                cache_services_removed[node_id] = removed_services
                cache_services_added[node_id] = added_services
                cache_changes_by_node[node_id] = {
                    "removed_service_ids": removed_services,
                    "added_service_ids": added_services,
                }
        cache_removed = sum(len(values) for values in cache_services_removed.values())
        cache_added = sum(len(values) for values in cache_services_added.values())
        cache_state_changed = not np.array_equal(previous_cache, actual_cache)

        actual_uav_positions = transition.final_uav_positions
        uav_movement_details: dict[str, dict[str, Any]] = {}
        uav_move_distances: dict[str, float] = {}
        for uav_id in previous_uav:
            before = np.asarray(previous_uav[uav_id], dtype=np.float64)
            after = np.asarray(actual_uav_positions[uav_id], dtype=np.float64)
            distance = float(math.dist(before.tolist(), after.tolist()))
            moved = not np.array_equal(before, after)
            uav_move_distances[uav_id] = distance
            uav_movement_details[uav_id] = {
                "from_position_m": before.tolist(),
                "to_position_m": after.tolist(),
                "distance_m": distance,
                "state_changed": moved,
            }
        uav_moved_count = sum(
            bool(details["state_changed"])
            for details in uav_movement_details.values()
        )
        uav_state_changed = uav_moved_count > 0
        uav_total_distance = sum(uav_move_distances.values())

        cache_cost_available = bool(
            transition.diagnostics.get("cache_update_cost_available", False)
        )
        cache_update_cost = transition.diagnostics.get("cache_cost")
        if cache_cost_available and cache_update_cost is not None:
            cache_update_cost = float(cache_update_cost)
            if not math.isfinite(cache_update_cost):
                raise FloatingPointError("正式执行器返回了非有限缓存更新成本")
        else:
            cache_update_cost = None
            cache_cost_available = False
        movement_cost_available = bool(
            transition.diagnostics.get("uav_movement_cost_available", False)
        )
        uav_movement_cost = transition.diagnostics.get("movement_cost")
        if movement_cost_available and uav_movement_cost is not None:
            uav_movement_cost = float(uav_movement_cost)
            if not math.isfinite(uav_movement_cost):
                raise FloatingPointError("正式执行器返回了非有限 UAV 移动成本")
        else:
            uav_movement_cost = None
            movement_cost_available = False
        replay_buffer.add(
            state_features=state,
            action=selected.candidate.action,
            action_features=selected.candidate.features,
            reward=transition.reward,
            next_state_features=transition.next_state_features,
            next_state_fingerprint=transition.next_state_fingerprint,
            next_legal_candidates=transition.next_legal_candidates,
            next_candidate_rebuild_context=(
                transition.next_candidate_rebuild_context
            ),
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
                "selected_action_type": transition.operation,
                "selected_candidate_id": transition.candidate_id,
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
                "selected_category_q": selected.category_q_value,
                "selected_candidate_q": selected.candidate_q_value,
                "q_values_by_action_category": dict(
                    selected.q_values_by_action_category
                ),
                "selected_candidate_action": selected.candidate.action.to_dict(),
                "explored": bool(selected.explored),
                "selection_mode": "exploration" if selected.explored else "greedy",
                "cache_state_changed": cache_state_changed,
                "actual_state_changed": bool(
                    cache_state_changed or uav_state_changed
                ),
                "cache_changed_node_ids": changed_node_ids,
                "cache_changed_node_count": len(changed_node_ids),
                "cache_changes_by_node": cache_changes_by_node,
                "cache_services_added": cache_services_added,
                "cache_services_removed": cache_services_removed,
                "cache_services_added_count": cache_added,
                "cache_services_removed_count": cache_removed,
                "cache_service_replacements": min(cache_removed, cache_added),
                "uav_move_distances_m": uav_move_distances,
                "uav_state_changed": uav_state_changed,
                "uav_movement_details": uav_movement_details,
                "uav_moved_count": uav_moved_count,
                "uav_total_distance": uav_total_distance,
                "selected_target_uav_positions": {
                    key: value.tolist() for key, value in action_uav.items()
                },
                "uav_total_move_distance_m": uav_total_distance,
                "cache_update_cost": cache_update_cost,
                "cache_update_cost_available": cache_cost_available,
                "cache_download_cost": cache_update_cost,
                "uav_movement_cost": uav_movement_cost,
                "uav_movement_cost_available": movement_cost_available,
                "decision_wall_time_s": 0.0,
                "executor_calls": transition.executor_calls,
                "matching_calls": transition.matching_calls,
                "kkt_solver_calls": transition.kkt_solver_calls,
                "actual_execution_node_counts": transition.diagnostics.get(
                    "actual_execution_node_counts"
                ),
                "actual_execution_nodes": transition.diagnostics.get(
                    "actual_execution_nodes"
                ),
                "actual_execution_node_data_available": transition.diagnostics.get(
                    "actual_execution_node_data_available", False
                ),
                "actual_system_utility": transition.diagnostics.get(
                    "actual_system_utility", transition.reward
                ),
                "actual_task_utility": transition.diagnostics.get(
                    "actual_task_utility"
                ),
                "actual_task_count": transition.diagnostics.get(
                    "actual_task_count"
                ),
                "actual_matching_node_task_counts": transition.diagnostics.get(
                    "actual_matching_node_task_counts"
                ),
                "per_task_delay_s": transition.diagnostics.get(
                    "per_task_delay_s"
                ),
                "per_task_vehicle_energy_j": transition.diagnostics.get(
                    "per_task_vehicle_energy_j"
                ),
                "per_task_utility": transition.diagnostics.get(
                    "per_task_utility"
                ),
                "actual_total_task_delay_s": transition.diagnostics.get(
                    "actual_total_task_delay_s"
                ),
                "actual_total_vehicle_energy_j": transition.diagnostics.get(
                    "actual_total_vehicle_energy_j"
                ),
                "cache_download_delay_s": transition.diagnostics.get(
                    "cache_download_delay_s"
                ),
                "uav_movement_energy_j": transition.diagnostics.get(
                    "uav_movement_energy_j"
                ),
                "predicted_cache_coverage_before": transition.diagnostics.get(
                    "predicted_cache_coverage_before"
                ),
                "predicted_cache_coverage_after": transition.diagnostics.get(
                    "predicted_cache_coverage_after"
                ),
                "predicted_cache_coverage_delta": transition.diagnostics.get(
                    "predicted_cache_coverage_delta"
                ),
                "predicted_cache_coverage_metric": transition.diagnostics.get(
                    "predicted_cache_coverage_metric"
                ),
                "predicted_cache_coverage_error": transition.diagnostics.get(
                    "predicted_cache_coverage_error"
                ),
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
                "state_features_used_current_actual_task_data": transition.diagnostics.get(
                    "state_features_used_current_actual_task_data"
                ),
                "state_features_used_current_actual_vehicle_positions": transition.diagnostics.get(
                    "state_features_used_current_actual_vehicle_positions"
                ),
                "candidate_generation_received_actual_tasks": transition.diagnostics.get(
                    "candidate_generation_received_actual_tasks"
                ),
                "candidate_generation_received_actual_vehicle_positions": transition.diagnostics.get(
                    "candidate_generation_received_actual_vehicle_positions"
                ),
                "candidate_features_used_current_actual_vehicle_positions": transition.diagnostics.get(
                    "candidate_features_used_current_actual_vehicle_positions"
                ),
                "state_feature_contract_version": transition.diagnostics.get(
                    "decision_state_feature_contract_version"
                ),
                "candidate_feature_contract_version": transition.diagnostics.get(
                    "decision_candidate_feature_contract_version"
                ),
                "prediction_horizon_count": transition.diagnostics.get(
                    "prediction_horizon_count"
                ),
                "prediction_horizon_mask": transition.diagnostics.get(
                    "prediction_horizon_mask"
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
                update = agent.update(
                    replay_buffer.sample(int(settings.batch_size)),
                    candidate_rebuilder=environment.rebuild_candidates,
                )
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
                        "category_td_loss",
                        "candidate_advantage_loss",
                        "gradient_norm_before_clip",
                        "q_value_mean",
                        "category_q_value_mean",
                        "candidate_q_value_mean",
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
    execution_node_counts: Counter[str] = Counter()
    execution_node_data_available_slots = 0
    for row in step_rows:
        counts = row.get("actual_execution_node_counts")
        if (
            row.get("actual_execution_node_data_available") is True
            and isinstance(counts, Mapping)
        ):
            execution_node_data_available_slots += 1
            for node_id, count in counts.items():
                execution_node_counts[str(node_id)] += int(count)
    execution_node_total = sum(execution_node_counts.values())
    execution_node_data_available = (
        slot_count > 0 and execution_node_data_available_slots == slot_count
    )
    execution_node_shares = {
        node_id: count / execution_node_total
        for node_id, count in sorted(execution_node_counts.items())
    } if execution_node_total else {}
    most_common_execution_node = (
        min(
            execution_node_counts,
            key=lambda node_id: (-execution_node_counts[node_id], node_id),
        )
        if execution_node_data_available and execution_node_total
        else None
    )
    cache_node_update_slot_counts: Counter[str] = Counter()
    cache_services_added_by_service: Counter[str] = Counter()
    cache_services_removed_by_service: Counter[str] = Counter()
    cache_services_added_by_node: Counter[str] = Counter()
    cache_services_removed_by_node: Counter[str] = Counter()
    uav_distance_by_uav: Counter[str] = Counter()
    for row in step_rows:
        for node_id in row["cache_changed_node_ids"]:
            cache_node_update_slot_counts[str(node_id)] += 1
        for node_id, service_values in row["cache_services_added"].items():
            cache_services_added_by_node[str(node_id)] += len(service_values)
            cache_services_added_by_service.update(
                str(service_id) for service_id in service_values
            )
        for node_id, service_values in row["cache_services_removed"].items():
            cache_services_removed_by_node[str(node_id)] += len(service_values)
            cache_services_removed_by_service.update(
                str(service_id) for service_id in service_values
            )
        for uav_id, distance in row["uav_move_distances_m"].items():
            uav_distance_by_uav[str(uav_id)] += float(distance)
    actual_cache_update_count = sum(
        bool(row["cache_state_changed"]) for row in step_rows
    )
    cache_changed_node_count = sum(
        int(row["cache_changed_node_count"]) for row in step_rows
    )
    uav_movement_count = sum(int(row["uav_moved_count"]) for row in step_rows)
    uav_movement_slot_count = sum(
        bool(row["uav_state_changed"]) for row in step_rows
    )
    cache_cost_values = [
        float(row["cache_update_cost"])
        for row in step_rows
        if row["cache_update_cost_available"]
    ]
    uav_cost_values = [
        float(row["uav_movement_cost"])
        for row in step_rows
        if row["uav_movement_cost_available"]
    ]
    cache_cost_complete = bool(slot_count) and len(cache_cost_values) == slot_count
    uav_cost_complete = bool(slot_count) and len(uav_cost_values) == slot_count
    cache_cost_known_subtotal = sum(cache_cost_values) if cache_cost_values else None
    uav_cost_known_subtotal = sum(uav_cost_values) if uav_cost_values else None
    actual_uav_total_distance = sum(
        float(row["uav_total_distance"]) for row in step_rows
    )
    actual_state_change_counts_by_operation = {
        "KEEP_STAY": sum(
            row["operation"] == "KEEP_STAY" and not row["actual_state_changed"]
            for row in step_rows
        ),
        "CACHE_UPDATE": sum(
            row["operation"] == "CACHE_UPDATE" and row["cache_state_changed"]
            for row in step_rows
        ),
        "UAV_MOVE": sum(
            row["operation"] == "UAV_MOVE" and row["uav_state_changed"]
            for row in step_rows
        ),
    }
    category_q_statistics: dict[str, dict[str, Any]] = {}
    candidate_q_statistics: dict[str, dict[str, Any]] = {}
    candidate_count_statistics: dict[str, dict[str, Any]] = {}
    for operation in operation_names:
        category_q_statistics[operation] = _numeric_summary(
            [
                row.get("q_values_by_action_category", {}).get(operation)
                for row in step_rows
            ]
        )
        candidate_q_statistics[operation] = _numeric_summary(
            [
                row.get("selected_candidate_q")
                if row.get("selected_action_type") == operation
                else None
                for row in step_rows
            ]
        )
        candidate_count_statistics[operation] = _numeric_summary(
            [
                row.get("candidate_counts_by_operation", {}).get(operation)
                for row in step_rows
            ]
        )
    candidate_population_q_statistics = _aggregate_candidate_q_statistics(
        step_rows, operation_names
    )
    cache_update_action_rows = [
        row for row in step_rows if row["selected_action_type"] == "CACHE_UPDATE"
    ]
    cache_changed_rows = [row for row in step_rows if row["cache_state_changed"]]
    cache_unchanged_rows = [row for row in step_rows if not row["cache_state_changed"]]
    predicted_coverage_delta_by_cache_action = _numeric_summary(
        [row.get("predicted_cache_coverage_delta") for row in cache_update_action_rows]
    )
    cache_update_actual_utility = _numeric_summary(
        [row.get("actual_system_utility") for row in cache_update_action_rows]
    )
    cache_changed_actual_utility = _numeric_summary(
        [row.get("actual_system_utility") for row in cache_changed_rows]
    )
    cache_unchanged_actual_utility = _numeric_summary(
        [row.get("actual_system_utility") for row in cache_unchanged_rows]
    )
    episode_summary = {
        "cumulative_system_utility": float(episode_reward),
        "mean_system_utility_per_slot": (
            float(episode_reward) / slot_count if slot_count else None
        ),
        "cache_update_count": actual_cache_update_count,
        "actual_cache_update_count": actual_cache_update_count,
        "cache_update_action_count": execution_counts["CACHE_UPDATE"],
        "cache_update_action_selected_q_statistics": _numeric_summary(
            [row["selected_q_value"] for row in cache_update_action_rows]
        ),
        "cache_update_action_category_q_statistics": _numeric_summary(
            [row["selected_category_q"] for row in cache_update_action_rows]
        ),
        "cache_update_action_candidate_q_statistics": _numeric_summary(
            [row["selected_candidate_q"] for row in cache_update_action_rows]
        ),
        "cache_changed_slot_count": actual_cache_update_count,
        "actual_cache_changed_slot_count": actual_cache_update_count,
        "cache_changed_node_count": cache_changed_node_count,
        "actual_cache_changed_node_count": cache_changed_node_count,
        "cache_unique_changed_node_count": len(cache_node_update_slot_counts),
        "cache_node_update_slot_counts": dict(
            sorted(cache_node_update_slot_counts.items())
        ),
        "cache_node_update_slot_shares": {
            node_id: count / slot_count
            for node_id, count in sorted(cache_node_update_slot_counts.items())
        } if slot_count else {},
        "cache_service_replacement_count": sum(
            int(row["cache_service_replacements"]) for row in step_rows
        ),
        "cache_services_removed": sum(
            int(row["cache_services_removed_count"]) for row in step_rows
        ),
        "cache_services_added": sum(
            int(row["cache_services_added_count"]) for row in step_rows
        ),
        "actual_cache_services_added_count": sum(
            int(row["cache_services_added_count"]) for row in step_rows
        ),
        "actual_cache_services_removed_count": sum(
            int(row["cache_services_removed_count"]) for row in step_rows
        ),
        "cache_services_added_by_node": dict(
            sorted(cache_services_added_by_node.items())
        ),
        "cache_services_removed_by_node": dict(
            sorted(cache_services_removed_by_node.items())
        ),
        "cache_services_added_by_service": dict(
            sorted(cache_services_added_by_service.items())
        ),
        "cache_services_removed_by_service": dict(
            sorted(cache_services_removed_by_service.items())
        ),
        "cache_update_cost": sum(cache_cost_values) if cache_cost_complete else None,
        "cache_update_cost_known_subtotal": cache_cost_known_subtotal,
        "cache_update_cost_recorded_slot_count": len(cache_cost_values),
        "cache_update_cost_unavailable_slot_count": slot_count - len(cache_cost_values),
        "cache_download_cost": sum(cache_cost_values) if cache_cost_complete else None,
        "predicted_cache_coverage_delta_by_cache_update_action": (
            predicted_coverage_delta_by_cache_action
        ),
        "predicted_cache_coverage_metric_is_not_actual_hit_rate": True,
        "cache_update_action_actual_system_utility": cache_update_actual_utility,
        "cache_changed_actual_system_utility": cache_changed_actual_utility,
        "cache_unchanged_actual_system_utility": cache_unchanged_actual_utility,
        "cache_update_action_total_task_delay_s": _numeric_summary(
            [row.get("actual_total_task_delay_s") for row in cache_update_action_rows]
        ),
        "cache_update_action_total_vehicle_energy_j": _numeric_summary(
            [row.get("actual_total_vehicle_energy_j") for row in cache_update_action_rows]
        ),
        "cache_update_action_cache_download_delay_s": _numeric_summary(
            [row.get("cache_download_delay_s") for row in cache_update_action_rows]
        ),
        "actual_state_change_counts_by_operation": (
            actual_state_change_counts_by_operation
        ),
        "q_values_by_action_category_statistics": category_q_statistics,
        "candidate_q_population_statistics_by_operation": (
            candidate_population_q_statistics
        ),
        "selected_candidate_q_statistics_by_operation": candidate_q_statistics,
        "candidate_count_statistics_by_operation": candidate_count_statistics,
        "uav_move_count": uav_movement_slot_count,
        "uav_move_action_count": execution_counts["UAV_MOVE"],
        "uav_movement_slot_count": uav_movement_slot_count,
        "actual_uav_movement_slot_count": uav_movement_slot_count,
        "uav_movement_count": uav_movement_count,
        "actual_uav_movement_count": uav_movement_count,
        "uav_moved_uav_count": uav_movement_count,
        "uav_distance_by_uav_m": dict(sorted(uav_distance_by_uav.items())),
        "actual_uav_distance_by_uav_m": dict(
            sorted(uav_distance_by_uav.items())
        ),
        "uav_total_distance": actual_uav_total_distance,
        "actual_uav_total_distance_m": actual_uav_total_distance,
        "uav_total_move_distance_m": actual_uav_total_distance,
        "uav_movement_cost": sum(uav_cost_values) if uav_cost_complete else None,
        "uav_movement_cost_known_subtotal": uav_cost_known_subtotal,
        "uav_movement_cost_recorded_slot_count": len(uav_cost_values),
        "uav_movement_cost_unavailable_slot_count": slot_count - len(uav_cost_values),
        "decision_wall_time_s": sum(
            float(row["decision_wall_time_s"]) for row in step_rows
        ),
        "executor_calls": sum(int(row["executor_calls"]) for row in step_rows),
        "matching_calls": sum(int(row["matching_calls"]) for row in step_rows),
        "kkt_solver_calls": sum(int(row["kkt_solver_calls"]) for row in step_rows),
        "actual_execution_node_data_available": execution_node_data_available,
        "actual_execution_node_data_available_slot_count": (
            execution_node_data_available_slots
        ),
        "actual_execution_node_data_unavailable_slot_count": (
            slot_count - execution_node_data_available_slots
        ),
        "actual_execution_node_task_count": execution_node_total,
        "actual_execution_node_counts": dict(sorted(execution_node_counts.items())),
        "actual_execution_node_shares": execution_node_shares,
        "most_common_actual_execution_node": most_common_execution_node,
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
