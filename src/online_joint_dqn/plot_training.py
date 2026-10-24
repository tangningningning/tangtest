"""从现有 DQN JSONL 日志生成非交互式训练图表。"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np


def _iter_jsonl_snapshot(path: Path, stats: dict[str, int]):
    """流式读取日志；只忽略并计数仍在写入的末尾半行。"""

    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith(("\n", "\r")):
                    stats["incomplete_tail_count"] = (
                        stats.get("incomplete_tail_count", 0) + 1
                    )
                    continue
                raise
            if isinstance(value, dict):
                yield value


def _read_small_jsonl_snapshot(path: Path) -> tuple[list[dict[str, Any]], int]:
    """读取通常只有数百行的 Episode 汇总。"""

    stats: dict[str, int] = {"incomplete_tail_count": 0}
    rows = list(_iter_jsonl_snapshot(path, stats))
    return rows, stats["incomplete_tail_count"]


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _coordinate_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    x_value = _finite_number(value[0])
    y_value = _finite_number(value[1])
    if x_value is None or y_value is None:
        return None
    return x_value, y_value


def _contiguous_moving_average(
    points: list[tuple[int, float]], window: int = 10
) -> list[tuple[int, float]]:
    """只对连续且有观测的索引窗口计算移动平均。"""

    output: list[tuple[int, float]] = []
    ordered = sorted(points)
    for start in range(max(0, len(ordered) - window + 1)):
        sample = ordered[start : start + window]
        if len(sample) != window:
            continue
        if any(sample[index + 1][0] != sample[index][0] + 1 for index in range(window - 1)):
            continue
        output.append((sample[-1][0], sum(value for _, value in sample) / window))
    return output


def _save_figure(figure: Any, path: Path) -> None:
    if getattr(figure, "_suptitle", None) is not None:
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    else:
        figure.tight_layout()
    temporary = path.with_name(f"{path.stem}.{uuid4().hex}.tmp{path.suffix}")
    try:
        figure.savefig(
            temporary,
            dpi=150,
            bbox_inches="tight",
            format=path.suffix.lstrip("."),
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _completed_attempts(
    episode_rows: list[dict[str, Any]],
) -> dict[int, int]:
    attempts: dict[int, int] = {}
    for row in episode_rows:
        if row.get("episode_index") is None:
            continue
        attempts[int(row["episode_index"])] = int(row.get("attempt", 1))
    return attempts


def generate_training_plots(
    run_dir: str | Path, *, snapshot_episode: int | None = None
) -> dict[str, Any]:
    """按已有记录生成图表，并返回可写入运行汇总的状态。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(run_dir).resolve()
    if snapshot_episode is not None and int(snapshot_episode) <= 0:
        raise ValueError("snapshot_episode 必须是正整数")
    if snapshot_episode is not None:
        snapshot_episode = int(snapshot_episode)
    output_dir = root / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_rows, incomplete_episode_tails = _read_small_jsonl_snapshot(
        root / "episode_metrics.jsonl"
    )
    if snapshot_episode is not None:
        episode_rows = [
            row
            for row in episode_rows
            if row.get("episode_index") is not None
            and int(row["episode_index"]) + 1 <= snapshot_episode
        ]
    completed_attempts = _completed_attempts(episode_rows)
    update_stats: dict[str, int] = {"incomplete_tail_count": 0}
    step_stats: dict[str, int] = {"incomplete_tail_count": 0}
    update_row_count = 0
    step_row_count = 0
    generated: list[str] = []
    skipped: dict[str, str] = {}
    failed: dict[str, str] = {}
    latest_plot_paths: list[Path] = []

    def save_plot(name: str, callback: Any) -> None:
        figure = None
        try:
            figure = callback()
            figure.suptitle(
                f"Completed Episodes: {len(episode_rows)}", fontsize="small"
            )
            target = output_dir / name
            _save_figure(figure, target)
            generated.append(str(target))
            latest_plot_paths.append(target)
            if snapshot_episode is not None:
                snapshot = output_dir / (
                    f"{target.stem}_episode_{snapshot_episode:04d}{target.suffix}"
                )
                _save_figure(figure, snapshot)
                generated.append(str(snapshot))
        except Exception as exc:
            failed[name] = f"{type(exc).__name__}: {exc}"
        finally:
            if figure is not None:
                plt.close(figure)

    # 系统效用：Return 使用左轴，单 Slot 均值及其 10-Episode 移动平均使用右轴。
    utility_points: list[tuple[int, float, float]] = []
    for fallback_index, row in enumerate(episode_rows, start=1):
        summary = row.get("episode_summary", {})
        episode_index = row.get("episode_index", fallback_index - 1)
        cumulative = _finite_number(
            summary.get("cumulative_system_utility", row.get("episode_reward"))
        )
        mean_slot = _finite_number(summary.get("mean_system_utility_per_slot"))
        if cumulative is not None and mean_slot is not None:
            utility_points.append((int(episode_index) + 1, cumulative, mean_slot))
    if utility_points:
        def draw_utility() -> Any:
            figure, utility_axis = plt.subplots(figsize=(10, 5.5))
            x = [point[0] for point in utility_points]
            cumulative_values = [point[1] for point in utility_points]
            mean_values = [point[2] for point in utility_points]
            utility_axis.plot(x, cumulative_values, color="#1f77b4", label="Cumulative utility")
            utility_axis.set_xlabel("Episode")
            utility_axis.set_ylabel("Cumulative system utility")
            mean_axis = utility_axis.twinx()
            mean_axis.plot(x, mean_values, color="#ff7f0e", alpha=0.65, label="Mean utility / Slot")
            smooth = _contiguous_moving_average(
                list(zip(x, mean_values, strict=True)), 10
            )
            if smooth:
                mean_axis.plot(
                    [point[0] for point in smooth],
                    [point[1] for point in smooth],
                    color="#d62728",
                    linewidth=2.0,
                    label="10-Episode mean",
                )
            mean_axis.set_ylabel("System utility per Slot")
            left_handles, left_labels = utility_axis.get_legend_handles_labels()
            right_handles, right_labels = mean_axis.get_legend_handles_labels()
            utility_axis.legend(left_handles + right_handles, left_labels + right_labels, loc="best")
            utility_axis.set_title("DQN System Utility")
            utility_axis.grid(alpha=0.2)
            return figure

        save_plot("system_utility.png", draw_utility)
    else:
        skipped["system_utility.png"] = "没有同时包含累计效用和每 Slot 效用均值的完整 Episode 记录。"

    # TD Loss：不补齐缺失更新；至少有 20 个点时增加 20-update 平滑线。
    loss_points: list[tuple[int, float]] = []
    q_points: list[tuple[int, float | None, float | None]] = []
    gradient_norm_points: list[tuple[int, float]] = []
    gradient_clip_points: list[tuple[int, float]] = []
    clipped_gradient_points: list[tuple[int, float]] = []
    target_sync_steps: list[int] = []
    for index, row in enumerate(
        _iter_jsonl_snapshot(root / "update_metrics.jsonl", update_stats),
        start=1,
    ):
        if row.get("episode_index") is not None:
            episode_index = int(row["episode_index"])
            if (
                episode_index not in completed_attempts
                or int(row.get("attempt", 1)) != completed_attempts[episode_index]
            ):
                continue
        update_row_count += 1
        gradient_step = int(row.get("gradient_step", index))
        loss_value = _finite_number(row.get("td_loss"))
        if loss_value is not None:
            loss_points.append((gradient_step, loss_value))
        q_value = _finite_number(row.get("q_value_mean"))
        target_value = _finite_number(row.get("td_target_mean"))
        if q_value is not None or target_value is not None:
            q_points.append((gradient_step, q_value, target_value))
        gradient_norm = _finite_number(row.get("gradient_norm_before_clip"))
        if gradient_norm is not None:
            gradient_norm_points.append((gradient_step, gradient_norm))
            threshold = _finite_number(row.get("gradient_clip_threshold"))
            clipped = row.get("gradient_was_clipped")
            if not isinstance(clipped, bool) and threshold is not None:
                clipped = gradient_norm > threshold
            if threshold is not None:
                gradient_clip_points.append((gradient_step, threshold))
            if clipped is True:
                clipped_gradient_points.append((gradient_step, gradient_norm))
        if bool(row.get("target_network_synced", False)):
            target_sync_steps.append(gradient_step)
    if loss_points:
        def draw_loss() -> Any:
            figure, axis = plt.subplots(figsize=(10, 5.5))
            x = [point[0] for point in loss_points]
            values = [point[1] for point in loss_points]
            axis.plot(x, values, color="#1f77b4", alpha=0.55, label="TD loss")
            smooth = _contiguous_moving_average(
                list(zip(x, values, strict=True)), 20
            )
            if smooth:
                axis.plot(
                    [point[0] for point in smooth],
                    [point[1] for point in smooth],
                    color="#d62728",
                    linewidth=2,
                    label="20-update moving average",
                )
            axis.set_xlabel("Gradient update")
            axis.set_ylabel("TD loss")
            axis.set_title("Double DQN TD Loss")
            axis.grid(alpha=0.2)
            axis.legend()
            return figure

        save_plot("td_loss.png", draw_loss)
    else:
        skipped["td_loss.png"] = "update_metrics.jsonl 中没有有限的 TD Loss 记录。"

    if q_points:
        def draw_q_targets() -> Any:
            figure, axis = plt.subplots(figsize=(10, 5.5))
            q_x = [point[0] for point in q_points if point[1] is not None]
            q_y = [float(point[1]) for point in q_points if point[1] is not None]
            target_x = [point[0] for point in q_points if point[2] is not None]
            target_y = [float(point[2]) for point in q_points if point[2] is not None]
            if q_y:
                axis.plot(q_x, q_y, label="Online Q mean", color="#1f77b4")
            if target_y:
                axis.plot(target_x, target_y, label="TD target mean", color="#ff7f0e")
            axis.set_xlabel("Gradient update")
            axis.set_ylabel("Value")
            axis.set_title("Online Q Values and Double DQN TD Targets")
            axis.grid(alpha=0.2)
            axis.legend()
            return figure

        save_plot("q_value_td_target.png", draw_q_targets)
    else:
        skipped["q_value_td_target.png"] = "没有有限的在线 Q 值或 TD 目标记录。"

    epsilon_stats: dict[str, int] = {"incomplete_tail_count": 0}
    epsilon_points: list[tuple[int, float]] = []
    for row in _iter_jsonl_snapshot(root / "step_metrics.jsonl", epsilon_stats):
        if row.get("episode_index") is not None:
            episode_index = int(row["episode_index"])
            if (
                episode_index not in completed_attempts
                or int(row.get("attempt", 1)) != completed_attempts[episode_index]
            ):
                continue
        global_step = row.get("global_step")
        epsilon = _finite_number(row.get("epsilon"))
        if isinstance(global_step, int) and not isinstance(global_step, bool) and epsilon is not None:
            epsilon_points.append((global_step, epsilon))

    if gradient_norm_points or epsilon_points or target_sync_steps:
        def draw_learning_stability() -> Any:
            figure, (gradient_axis, epsilon_axis) = plt.subplots(
                2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1.3, 1]}
            )
            if gradient_norm_points:
                x = [point[0] for point in gradient_norm_points]
                y = [point[1] for point in gradient_norm_points]
                gradient_axis.plot(x, y, color="#1f77b4", label="Gradient norm before clipping")
                if gradient_clip_points:
                    thresholds = {point[1] for point in gradient_clip_points}
                    if len(thresholds) == 1:
                        gradient_axis.axhline(
                            next(iter(thresholds)),
                            color="#d62728",
                            linestyle="--",
                            label="Gradient clip threshold",
                        )
                    else:
                        threshold_x = [point[0] for point in gradient_clip_points]
                        threshold_y = [point[1] for point in gradient_clip_points]
                        gradient_axis.plot(
                            threshold_x,
                            threshold_y,
                            color="#d62728",
                            linestyle="--",
                            label="Gradient clip threshold",
                        )
                    if clipped_gradient_points:
                        gradient_axis.scatter(
                            [point[0] for point in clipped_gradient_points],
                            [point[1] for point in clipped_gradient_points],
                            color="#ff7f0e",
                            marker="x",
                            label="Clipped updates",
                        )
                if target_sync_steps:
                    for sync_index, step in enumerate(target_sync_steps):
                        gradient_axis.axvline(
                            step,
                            color="#9467bd",
                            alpha=0.35,
                            linestyle=":",
                            label="Target network sync" if sync_index == 0 else None,
                        )
                gradient_axis.set_xlabel("Gradient update")
                gradient_axis.set_ylabel("Gradient norm")
                gradient_axis.set_title(
                    f"Gradient clipping and target sync positions ({len(target_sync_steps)} syncs)"
                )
                gradient_axis.legend()
                gradient_axis.grid(alpha=0.2)
            else:
                gradient_axis.text(0.5, 0.5, "Gradient/update sync metrics unavailable", ha="center", va="center")
                gradient_axis.set_axis_off()
            if epsilon_points:
                epsilon_axis.plot(
                    [point[0] for point in epsilon_points],
                    [point[1] for point in epsilon_points],
                    color="#2ca02c",
                    label="Epsilon",
                )
                epsilon_axis.set_xlabel("Environment decision Slot")
                epsilon_axis.set_ylabel("Epsilon")
                epsilon_axis.set_title("Exploration schedule")
                epsilon_axis.set_ylim(bottom=0.0)
                epsilon_axis.grid(alpha=0.2)
                epsilon_axis.legend()
            else:
                epsilon_axis.text(0.5, 0.5, "Epsilon records unavailable", ha="center", va="center")
                epsilon_axis.set_axis_off()
            return figure

        save_plot("dqn_learning_stability.png", draw_learning_stability)
    else:
        skipped["dqn_learning_stability.png"] = (
            "没有可用的梯度范数、目标网络同步或环境步 epsilon 记录。"
        )

    operations = ("KEEP_STAY", "CACHE_UPDATE", "UAV_MOVE")
    action_points: list[tuple[int, dict[str, int]]] = []
    selection_mode_points: list[
        tuple[int, dict[str, int], dict[str, int]]
    ] = []
    for fallback_index, row in enumerate(episode_rows, start=1):
        counts = row.get("execution_counts_by_operation", row.get("action_counts"))
        if not isinstance(counts, Mapping):
            continue
        values: dict[str, int] = {}
        complete = True
        for operation in operations:
            value = counts.get(operation)
            if isinstance(value, bool) or not isinstance(value, int):
                complete = False
                break
            values[operation] = int(value)
        if complete:
            episode_index = int(row.get("episode_index", fallback_index - 1)) + 1
            action_points.append((episode_index, values))
            exploration = row.get("exploration_counts_by_operation")
            greedy = row.get("greedy_selection_counts_by_operation")
            if isinstance(exploration, Mapping) and isinstance(greedy, Mapping):
                exploration_values: dict[str, int] = {}
                greedy_values: dict[str, int] = {}
                for operation in operations:
                    explore_count = exploration.get(operation)
                    greedy_count = greedy.get(operation)
                    if (
                        isinstance(explore_count, bool)
                        or not isinstance(explore_count, int)
                        or explore_count < 0
                        or isinstance(greedy_count, bool)
                        or not isinstance(greedy_count, int)
                        or greedy_count < 0
                    ):
                        break
                    exploration_values[operation] = int(explore_count)
                    greedy_values[operation] = int(greedy_count)
                if (
                    len(exploration_values) == len(operations)
                    and len(greedy_values) == len(operations)
                ):
                    selection_mode_points.append(
                        (episode_index, exploration_values, greedy_values)
                    )
    if action_points:
        def draw_actions() -> Any:
            panel_count = 3 if selection_mode_points else 2
            figure, axes = plt.subplots(
                panel_count,
                1,
                figsize=(11, 3.8 * panel_count),
                squeeze=False,
            )
            count_axis = axes[0, 0]
            x = [point[0] for point in action_points]
            width = 0.25
            colors = {"KEEP_STAY": "#7f7f7f", "CACHE_UPDATE": "#2ca02c", "UAV_MOVE": "#9467bd"}
            offsets = (-width, 0.0, width)
            for operation, offset in zip(operations, offsets, strict=True):
                count_axis.bar(
                    [value + offset for value in x],
                    [point[1][operation] for point in action_points],
                    width=width,
                    label=operation,
                    color=colors[operation],
                )
            count_axis.set_ylabel("Executed decisions")
            count_axis.set_title("Action counts by Episode")
            count_axis.legend(ncol=3)
            count_axis.grid(axis="y", alpha=0.2)

            share_axis = axes[1, 0]
            share_points = [
                (
                    episode,
                    {
                        operation: count / sum(counts.values())
                        for operation, count in counts.items()
                    },
                )
                for episode, counts in action_points
                if sum(counts.values()) > 0
            ]
            for operation in operations:
                share_axis.plot(
                    [point[0] for point in share_points],
                    [100.0 * point[1][operation] for point in share_points],
                    marker=".",
                    color=colors[operation],
                    label=operation,
                )
            share_axis.set_xlabel("Episode")
            share_axis.set_ylabel("Executed decisions (%)")
            share_axis.set_title("Action shares by Episode")
            share_axis.set_ylim(0.0, 100.0)
            share_axis.legend(ncol=3)
            share_axis.grid(alpha=0.2)

            if selection_mode_points:
                mode_axis = axes[2, 0]
                styles = (("exploration", "--"), ("greedy", "-"))
                for operation in operations:
                    for mode_index, (mode, line_style) in enumerate(styles):
                        values = [
                            point[mode_index + 1][operation]
                            for point in selection_mode_points
                        ]
                        mode_axis.plot(
                            [point[0] for point in selection_mode_points],
                            values,
                            marker=".",
                            linestyle=line_style,
                            color=colors[operation],
                            label=f"{operation} {mode}",
                        )
                mode_axis.set_xlabel("Episode")
                mode_axis.set_ylabel("Selections")
                mode_axis.set_title("Exploration and greedy selections")
                mode_axis.legend(ncol=3, fontsize="small")
                mode_axis.grid(alpha=0.2)
            return figure

        save_plot("action_distribution.png", draw_actions)
    else:
        skipped["action_distribution.png"] = "没有完整记录三类动作实际执行次数的 Episode。"

    cache_frequency_points: list[tuple[int, int, int]] = []
    cache_node_totals: Counter[str] = Counter()
    cache_node_metric_episode_ids: set[int] = set()
    cache_service_points: list[tuple[int, int, int]] = []
    uav_frequency_points: list[tuple[int, int, int]] = []
    uav_distance_points: list[tuple[int, dict[str, float]]] = []
    action_state_change_points: list[
        tuple[int, int, int, int, int, float]
    ] = []
    for fallback_index, row in enumerate(episode_rows, start=1):
        episode_index = int(row.get("episode_index", fallback_index - 1)) + 1
        summary = row.get("episode_summary", {})
        action_counts = row.get(
            "execution_counts_by_operation", row.get("action_counts")
        )
        actual_changes = summary.get("actual_state_change_counts_by_operation")
        actual_distance = _finite_number(summary.get("actual_uav_total_distance_m"))
        if isinstance(action_counts, Mapping) and isinstance(actual_changes, Mapping):
            values = (
                action_counts.get("CACHE_UPDATE"),
                summary.get("actual_cache_update_count"),
                action_counts.get("UAV_MOVE"),
                actual_changes.get("UAV_MOVE"),
            )
            if (
                all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                    for value in values
                )
                and actual_distance is not None
                and actual_distance >= 0.0
            ):
                action_state_change_points.append(
                    (
                        episode_index,
                        int(values[0]),
                        int(values[1]),
                        int(values[2]),
                        int(values[3]),
                        actual_distance,
                    )
                )
        update_count = summary.get("actual_cache_update_count")
        changed_slots = summary.get("actual_cache_changed_slot_count")
        if (
            isinstance(update_count, int)
            and not isinstance(update_count, bool)
            and isinstance(changed_slots, int)
            and not isinstance(changed_slots, bool)
            and update_count >= 0
            and changed_slots >= 0
        ):
            cache_frequency_points.append(
                (episode_index, update_count, changed_slots)
            )
        node_counts = summary.get("cache_node_update_slot_counts")
        if isinstance(node_counts, Mapping) and all(
            isinstance(count, int) and not isinstance(count, bool) and count >= 0
            for count in node_counts.values()
        ):
            cache_node_metric_episode_ids.add(episode_index)
            for node_id, count in node_counts.items():
                cache_node_totals[str(node_id)] += count
        added_count = summary.get("actual_cache_services_added_count")
        removed_count = summary.get("actual_cache_services_removed_count")
        if (
            isinstance(added_count, int)
            and not isinstance(added_count, bool)
            and isinstance(removed_count, int)
            and not isinstance(removed_count, bool)
            and added_count >= 0
            and removed_count >= 0
        ):
            cache_service_points.append(
                (episode_index, added_count, removed_count)
            )
        movement_count = summary.get("actual_uav_movement_count")
        movement_slots = summary.get("actual_uav_movement_slot_count")
        if (
            isinstance(movement_count, int)
            and not isinstance(movement_count, bool)
            and isinstance(movement_slots, int)
            and not isinstance(movement_slots, bool)
            and movement_count >= 0
            and movement_slots >= 0
        ):
            uav_frequency_points.append(
                (episode_index, movement_count, movement_slots)
            )
        distance_by_uav = summary.get("actual_uav_distance_by_uav_m")
        if isinstance(distance_by_uav, Mapping) and distance_by_uav:
            distances = {
                str(uav_id): _finite_number(distance)
                for uav_id, distance in distance_by_uav.items()
            }
            if all(value is not None and value >= 0 for value in distances.values()):
                uav_distance_points.append(
                    (episode_index, {key: float(value) for key, value in distances.items()})
                )

    if action_state_change_points:
        def draw_action_state_changes() -> Any:
            figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
            x = [point[0] for point in action_state_change_points]
            width = 0.36
            axes[0].bar(
                [value - width / 2 for value in x],
                [point[1] for point in action_state_change_points],
                width=width,
                label="CACHE_UPDATE selected",
                color="#2ca02c",
            )
            axes[0].bar(
                [value + width / 2 for value in x],
                [point[2] for point in action_state_change_points],
                width=width,
                label="Cache state changed",
                color="#98df8a",
            )
            axes[0].set_ylabel("Slots")
            axes[0].set_title("Cache action selections vs actual cache changes")
            axes[0].legend()
            axes[0].grid(axis="y", alpha=0.2)
            axes[1].bar(
                [value - width / 2 for value in x],
                [point[3] for point in action_state_change_points],
                width=width,
                label="UAV_MOVE selected",
                color="#9467bd",
            )
            axes[1].bar(
                [value + width / 2 for value in x],
                [point[4] for point in action_state_change_points],
                width=width,
                label="UAV position changed",
                color="#c5b0d5",
            )
            axes[1].set_ylabel("Moves / Slots")
            axes[1].set_title("UAV action selections vs actual movements")
            axes[1].legend()
            axes[1].grid(axis="y", alpha=0.2)
            axes[2].plot(
                x,
                [point[5] for point in action_state_change_points],
                marker=".",
                color="#8c564b",
                label="Actual UAV distance",
            )
            axes[2].set_xlabel("Episode")
            axes[2].set_ylabel("Distance (m)")
            axes[2].set_title("Actual UAV movement distance")
            axes[2].legend()
            axes[2].grid(alpha=0.2)
            return figure

        save_plot("selected_actions_vs_actual_changes.png", draw_action_state_changes)
    else:
        skipped["selected_actions_vs_actual_changes.png"] = (
            "缺少完整的动作计数、实际缓存/UAV 状态变化或实际移动距离记录。"
        )

    if cache_frequency_points:
        def draw_cache_frequency() -> Any:
            figure, axis = plt.subplots(figsize=(10, 5.5))
            x = [point[0] for point in cache_frequency_points]
            width = 0.38
            axis.bar(
                [value - width / 2 for value in x],
                [point[1] for point in cache_frequency_points],
                width=width,
                label="Actual cache update Slots",
                color="#2ca02c",
            )
            axis.bar(
                [value + width / 2 for value in x],
                [point[2] for point in cache_frequency_points],
                width=width,
                label="Slots with cache state changes",
                color="#98df8a",
            )
            moving_average = _contiguous_moving_average(
                [(point[0], float(point[1])) for point in cache_frequency_points]
            )
            if moving_average:
                axis.plot(
                    [point[0] for point in moving_average],
                    [point[1] for point in moving_average],
                    color="#d62728",
                    linewidth=2,
                    label="10-Episode moving average",
                )
            axis.set_xlabel("Episode")
            axis.set_ylabel("Actual cache changes")
            axis.set_title("Actual Cache Update Frequency")
            axis.legend()
            axis.grid(axis="y", alpha=0.2)
            return figure

        save_plot("cache_update_frequency.png", draw_cache_frequency)
    else:
        skipped["cache_update_frequency.png"] = (
            "Episode 日志缺少基于执行前后缓存状态差异的 actual_cache_update_count。"
        )

    if cache_node_metric_episode_ids:
        def draw_cache_nodes() -> Any:
            figure, axis = plt.subplots(figsize=(10, 5.5))
            if cache_node_totals:
                nodes = sorted(cache_node_totals, key=lambda node_id: (-cache_node_totals[node_id], node_id))
                axis.barh(
                    nodes[::-1],
                    [cache_node_totals[node_id] for node_id in nodes[::-1]],
                    color="#17becf",
                )
                axis.set_xlabel("Slots with an actual cache change on node")
                axis.grid(axis="x", alpha=0.2)
            else:
                axis.text(0.5, 0.5, "No cache node changes in complete recorded Episodes", ha="center", va="center")
                axis.set_axis_off()
            axis.set_title("Actual Cache Node Change Frequency")
            return figure

        save_plot("cache_node_changes.png", draw_cache_nodes)
    else:
        skipped["cache_node_changes.png"] = (
            "Episode 日志缺少基于真实缓存状态差异的节点更新频次。"
        )

    if cache_service_points:
        def draw_cache_services() -> Any:
            figure, axis = plt.subplots(figsize=(10, 5.5))
            x = [point[0] for point in cache_service_points]
            width = 0.38
            axis.bar(
                [value - width / 2 for value in x],
                [point[1] for point in cache_service_points],
                width=width,
                label="Services added",
                color="#2ca02c",
            )
            axis.bar(
                [value + width / 2 for value in x],
                [point[2] for point in cache_service_points],
                width=width,
                label="Services removed",
                color="#d62728",
            )
            axis.set_xlabel("Episode")
            axis.set_ylabel("Actual service entries changed")
            axis.set_title("Actual Cache Service Changes")
            axis.legend()
            axis.grid(axis="y", alpha=0.2)
            return figure

        save_plot("cache_service_changes.png", draw_cache_services)
    else:
        skipped["cache_service_changes.png"] = (
            "Episode 日志缺少基于实际缓存状态差异的服务增删计数。"
        )

    if uav_frequency_points:
        def draw_uav_frequency() -> Any:
            figure, axis = plt.subplots(figsize=(10, 5.5))
            x = [point[0] for point in uav_frequency_points]
            axis.plot(
                x,
                [point[1] for point in uav_frequency_points],
                marker="o",
                label="Actual UAV movements",
                color="#9467bd",
            )
            axis.plot(
                x,
                [point[2] for point in uav_frequency_points],
                marker="s",
                label="Slots with UAV movement",
                color="#8c564b",
            )
            moving_average = _contiguous_moving_average(
                [(point[0], float(point[1])) for point in uav_frequency_points]
            )
            if moving_average:
                axis.plot(
                    [point[0] for point in moving_average],
                    [point[1] for point in moving_average],
                    color="#e377c2",
                    linewidth=2,
                    label="10-Episode moving average",
                )
            axis.set_xlabel("Episode")
            axis.set_ylabel("Actual UAV moves")
            axis.set_title("Actual UAV Movement Frequency")
            axis.legend()
            axis.grid(alpha=0.2)
            return figure

        save_plot("uav_movement_frequency.png", draw_uav_frequency)
    else:
        skipped["uav_movement_frequency.png"] = (
            "Episode 日志缺少基于实际 UAV 位置差异的移动次数和移动 Slot 数。"
        )

    distance_uav_ids = (
        set(uav_distance_points[0][1]) if uav_distance_points else set()
    )
    distance_points = [
        point
        for point in uav_distance_points
        if set(point[1]) == distance_uav_ids
    ]
    if distance_points and distance_uav_ids:
        def draw_uav_distance() -> Any:
            figure, (episode_axis, cumulative_axis) = plt.subplots(
                2, 1, figsize=(10, 8), sharex=True
            )
            x = [point[0] for point in distance_points]
            for uav_id in sorted(distance_uav_ids):
                distances = [point[1][uav_id] for point in distance_points]
                episode_axis.plot(x, distances, marker=".", label=uav_id)
                continuous_count = 0
                cumulative_values: list[float] = []
                cumulative_x: list[int] = []
                running_total = 0.0
                previous_episode: int | None = None
                for episode, distance in zip(x, distances, strict=True):
                    if previous_episode is not None and episode != previous_episode + 1:
                        break
                    running_total += distance
                    cumulative_values.append(running_total)
                    cumulative_x.append(episode)
                    previous_episode = episode
                    continuous_count += 1
                if cumulative_x:
                    cumulative_axis.plot(
                        cumulative_x,
                        cumulative_values,
                        marker=".",
                        label=f"{uav_id} (first {continuous_count} contiguous episodes)",
                    )
            episode_axis.set_ylabel("Actual distance per Episode (m)")
            episode_axis.set_title("Actual UAV Movement Distance")
            episode_axis.legend()
            episode_axis.grid(alpha=0.2)
            cumulative_axis.set_xlabel("Episode")
            cumulative_axis.set_ylabel("Cumulative actual distance (m)")
            cumulative_axis.legend()
            cumulative_axis.grid(alpha=0.2)
            return figure

        save_plot("uav_movement_distance.png", draw_uav_distance)
    else:
        skipped["uav_movement_distance.png"] = (
            "Episode 日志缺少完整的实际 UAV 分机距离汇总。"
        )

    # 仅使用明确标记可用且属于已完成 Episode/attempt 的正式 Matching 结果。
    expected_slots_by_episode = {
        int(row["episode_index"]): int(row["step_count"])
        for row in episode_rows
        if row.get("episode_index") is not None and row.get("step_count") is not None
    }
    node_counts_by_episode: dict[int, Counter[str]] = defaultdict(Counter)
    node_available_by_episode: dict[int, bool] = defaultdict(lambda: True)
    node_seen_episode_ids: set[int] = set()
    node_seen_slots_by_episode: Counter[int] = Counter()
    known_node_counts: Counter[str] = Counter()
    known_slots = 0
    unavailable_slots = 0
    cost_rows_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    trajectory_by_episode: dict[int, dict[str, list[tuple[float, float]]]] = defaultdict(dict)
    trajectory_valid_by_episode: dict[int, bool] = defaultdict(lambda: True)
    trajectory_seen_slots: Counter[int] = Counter()
    trajectory_uav_ids_by_episode: dict[int, set[str]] = {}
    for row in _iter_jsonl_snapshot(root / "step_metrics.jsonl", step_stats):
        if row.get("episode_index") is None:
            continue
        episode_index = int(row["episode_index"])
        if (
            episode_index not in completed_attempts
            or int(row.get("attempt", 1)) != completed_attempts[episode_index]
        ):
            continue
        step_row_count += 1
        cost_rows_by_episode[episode_index].append(row)
        trajectory_seen_slots[episode_index] += 1
        movement_details = row.get("uav_movement_details")
        if (
            not isinstance(row.get("uav_state_changed"), bool)
            or not isinstance(row.get("uav_moved_count"), int)
            or isinstance(row.get("uav_moved_count"), bool)
            or not isinstance(movement_details, Mapping)
            or not movement_details
        ):
            trajectory_valid_by_episode[episode_index] = False
        elif trajectory_valid_by_episode[episode_index]:
            current_uav_ids = {str(uav_id) for uav_id in movement_details}
            expected_uav_ids = trajectory_uav_ids_by_episode.get(episode_index)
            if expected_uav_ids is None:
                trajectory_uav_ids_by_episode[episode_index] = current_uav_ids
            elif current_uav_ids != expected_uav_ids:
                trajectory_valid_by_episode[episode_index] = False
            if trajectory_valid_by_episode[episode_index]:
                for raw_uav_id, details in movement_details.items():
                    uav_id = str(raw_uav_id)
                    if not isinstance(details, Mapping):
                        trajectory_valid_by_episode[episode_index] = False
                        break
                    start_position = _coordinate_pair(details.get("from_position_m"))
                    end_position = _coordinate_pair(details.get("to_position_m"))
                    recorded_distance = _finite_number(details.get("distance_m"))
                    if (
                        start_position is None
                        or end_position is None
                        or recorded_distance is None
                        or recorded_distance < 0
                        or not isinstance(details.get("state_changed"), bool)
                        or not math.isclose(
                            math.dist(start_position, end_position),
                            recorded_distance,
                            rel_tol=1.0e-7,
                            abs_tol=1.0e-7,
                        )
                    ):
                        trajectory_valid_by_episode[episode_index] = False
                        break
                    path = trajectory_by_episode[episode_index].setdefault(uav_id, [])
                    if path and not np.allclose(
                        path[-1], start_position, rtol=1.0e-9, atol=1.0e-6
                    ):
                        trajectory_valid_by_episode[episode_index] = False
                        break
                    if not path:
                        path.append(start_position)
                    path.append(end_position)
                if trajectory_valid_by_episode[episode_index]:
                    actual_moved_count = sum(
                        bool(details.get("state_changed"))
                        for details in movement_details.values()
                    )
                    if (
                        actual_moved_count != int(row["uav_moved_count"])
                        or bool(actual_moved_count)
                        != bool(row["uav_state_changed"])
                    ):
                        trajectory_valid_by_episode[episode_index] = False
        node_seen_episode_ids.add(episode_index)
        node_seen_slots_by_episode[episode_index] += 1
        counts = row.get("actual_execution_node_counts")
        available = (
            row.get("actual_execution_node_data_available") is True
            and isinstance(counts, Mapping)
        )
        if not available:
            node_available_by_episode[episode_index] = False
            unavailable_slots += 1
            continue
        known_slots += 1
        for node_id, count in counts.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                node_available_by_episode[episode_index] = False
                continue
            node_counts_by_episode[episode_index][str(node_id)] += count
            known_node_counts[str(node_id)] += count
    complete_node_episode_ids = sorted(
        episode_index
        for episode_index in node_seen_episode_ids
        if (
            node_available_by_episode[episode_index]
            and node_seen_slots_by_episode[episode_index]
            == expected_slots_by_episode.get(episode_index)
        )
    )
    node_task_total = sum(known_node_counts.values())
    if node_task_total:
        def draw_nodes() -> Any:
            figure, (total_axis, episode_axis) = plt.subplots(
                2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [1, 1.2]}
            )
            sorted_nodes = sorted(known_node_counts, key=lambda name: (-known_node_counts[name], name))
            total_axis.barh(
                sorted_nodes[::-1],
                [known_node_counts[node_id] for node_id in sorted_nodes[::-1]],
                color="#17becf",
            )
            for bar, node_id in zip(
                total_axis.patches, sorted_nodes[::-1], strict=True
            ):
                count = known_node_counts[node_id]
                share = count / node_task_total * 100.0
                total_axis.text(
                    bar.get_width(),
                    bar.get_y() + bar.get_height() / 2,
                    f"  {count} ({share:.1f}%)",
                    va="center",
                    fontsize="small",
                )
            total_axis.set_xlabel("Recorded assigned tasks (count and cumulative share)")
            total_axis.set_title(
                f"Actual Matching Assignments (known slots {known_slots}, unavailable {unavailable_slots})"
            )
            if complete_node_episode_ids:
                nodes = sorted(known_node_counts)
                x = [index + 1 for index in complete_node_episode_ids]
                bottom = np.zeros(len(x), dtype=np.float64)
                # 仅对每个 Slot 节点数据完整的 Episode 绘图；未出现的节点代表观测到的零分配。
                for node_id in nodes:
                    values = [
                        node_counts_by_episode[episode_index].get(node_id, 0)
                        for episode_index in complete_node_episode_ids
                    ]
                    episode_axis.bar(x, values, bottom=bottom, label=node_id)
                    bottom += np.asarray(values, dtype=np.float64)
                episode_axis.set_xlabel("Episode index")
                episode_axis.set_ylabel("Recorded assigned tasks")
                episode_axis.set_title("Actual Execution Node Counts by Episode")
                episode_axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="small")
            else:
                episode_axis.text(0.5, 0.5, "No fully recorded Episode node assignments", ha="center", va="center")
                episode_axis.set_axis_off()
            if len(complete_node_episode_ids) < len(completed_attempts):
                total_axis.text(
                    0.99,
                    0.02,
                    "Partial: some completed Episodes lack complete node records",
                    ha="right",
                    transform=total_axis.transAxes,
                    fontsize="small",
                )
            return figure

        save_plot("actual_execution_node_distribution.png", draw_nodes)
    else:
        if known_slots and unavailable_slots == 0:
            skipped["actual_execution_node_distribution.png"] = (
                "存在完整的真实 Matching 记录，但所有已记录 Slot 均无任务分配；节点任务数为零。"
            )
        else:
            skipped["actual_execution_node_distribution.png"] = (
                "逐 Slot 日志没有带 actual_execution_node_data_available=true 的真实 Matching 节点计数；"
                "不从缓存、动作或预测字段推断执行节点。"
            )

    cost_series: dict[str, list[tuple[int, float]]] = {
        "task_utility": [],
        "cache_cost_penalty": [],
        "uav_movement_cost_penalty": [],
        "system_utility": [],
        "training_reward": [],
    }
    cost_complete_episode_ids: set[int] = set()
    for episode_index, expected_slots in expected_slots_by_episode.items():
        rows = cost_rows_by_episode.get(episode_index, [])
        slot_ids = [row.get("slot_id") for row in rows]
        if (
            expected_slots <= 0
            or len(rows) != expected_slots
            or any(not isinstance(slot_id, int) or isinstance(slot_id, bool) for slot_id in slot_ids)
            or len(set(slot_ids)) != expected_slots
        ):
            continue
        cost_complete_episode_ids.add(episode_index)
        episode_number = episode_index + 1

        def sum_finite_field(field: str) -> float | None:
            values = [_finite_number(row.get(field)) for row in rows]
            if any(value is None for value in values):
                return None
            return math.fsum(float(value) for value in values if value is not None)

        task_utility = sum_finite_field("actual_task_utility")
        if task_utility is not None:
            cost_series["task_utility"].append((episode_number, task_utility))
        final_utility = sum_finite_field("actual_system_utility")
        if final_utility is not None:
            cost_series["system_utility"].append((episode_number, final_utility))
        training_reward = sum_finite_field("reward")
        if training_reward is not None:
            cost_series["training_reward"].append((episode_number, training_reward))

        for field, availability_field, beta_field, old_destination, destination in (
            (
                "cache_update_cost",
                "cache_update_cost_available",
                "beta_ca",
                "cache_cost_weighted",
                "cache_cost_penalty",
            ),
            (
                "uav_movement_cost",
                "uav_movement_cost_available",
                "beta_q",
                "uav_movement_cost_weighted",
                "uav_movement_cost_penalty",
            ),
        ):
            penalty_values: list[float] = []
            complete = True
            for row in rows:
                raw_cost = _finite_number(row.get(field))
                beta = _finite_number(row.get(beta_field))
                penalty = _finite_number(row.get(destination))
                if penalty is None and row.get("reward_formula") == "G(t)=Q(t)-C_ca(t)-C_q(t)":
                    penalty = raw_cost
                old_weighted = _finite_number(row.get(old_destination))
                if penalty is None:
                    if (
                        row.get(availability_field) is True
                        and raw_cost is not None
                        and old_weighted is not None
                    ):
                        penalty = old_weighted
                    elif (
                        row.get(availability_field) is True
                        and raw_cost is not None
                        and beta is not None
                    ):
                        penalty = beta * raw_cost
                if (
                    row.get(availability_field) is not True
                    or penalty is None
                ):
                    complete = False
                    break
                penalty_values.append(penalty)
            if complete:
                cost_series[destination].append(
                    (episode_number, math.fsum(penalty_values))
                )

    if any(cost_series.values()):
        reward_matches_system_utility = bool(cost_series["system_utility"])
        reward_values_by_episode = dict(cost_series["training_reward"])
        for episode_number, system_value in cost_series["system_utility"]:
            reward_value = reward_values_by_episode.get(episode_number)
            if reward_value is None or not math.isclose(
                system_value, reward_value, rel_tol=1.0e-12, abs_tol=1.0e-9
            ):
                reward_matches_system_utility = False
                break

        def draw_cost_utility() -> Any:
            figure, (utility_axis, cost_axis) = plt.subplots(
                2, 1, figsize=(11, 8), sharex=True
            )
            for field, label, color in (
                ("task_utility", "Task system utility Q", "#1f77b4"),
                ("system_utility", "Final system utility G", "#2ca02c"),
            ):
                points = cost_series[field]
                if points:
                    utility_axis.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        marker=".",
                        label=label,
                        color=color,
                    )
            if reward_matches_system_utility:
                utility_axis.text(
                    0.01,
                    0.02,
                    "Training reward equals final system utility in the recorded DQN contract.",
                    transform=utility_axis.transAxes,
                    fontsize="small",
                )
            elif cost_series["training_reward"]:
                points = cost_series["training_reward"]
                utility_axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker=".",
                    linestyle="--",
                    label="Training reward",
                    color="#d62728",
                )
            utility_axis.set_ylabel("System utility / reward")
            utility_axis.set_title("Task utility and final utility")
            utility_axis.grid(alpha=0.2)
            if utility_axis.lines:
                utility_axis.legend()
            else:
                utility_axis.text(0.5, 0.5, "Utility records unavailable", ha="center", va="center")
            for field, label, color in (
                ("cache_cost_penalty", "Cache cost C_ca", "#ff7f0e"),
                ("uav_movement_cost_penalty", "UAV movement cost C_q", "#9467bd"),
            ):
                points = cost_series[field]
                if points:
                    cost_axis.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        marker=".",
                        label=label,
                        color=color,
                    )
            cost_axis.set_xlabel("Episode")
            cost_axis.set_ylabel("Cost penalty")
            cost_axis.set_title("Recorded cost penalties (missing values omitted)")
            cost_axis.grid(alpha=0.2)
            if cost_axis.lines:
                cost_axis.legend()
            else:
                cost_axis.text(0.5, 0.5, "Weighted cost records unavailable", ha="center", va="center")
            return figure

        save_plot("cost_utility_breakdown.png", draw_cost_utility)
    else:
        skipped["cost_utility_breakdown.png"] = (
            "没有完整 Episode 的逐 Slot 任务效用、最终效用或成本分解记录。"
        )

    complete_trajectory_episode_ids = sorted(
        episode_index
        for episode_index, expected_slots in expected_slots_by_episode.items()
        if (
            expected_slots > 0
            and trajectory_valid_by_episode[episode_index]
            and trajectory_seen_slots[episode_index] == expected_slots
            and trajectory_by_episode[episode_index]
            and all(
                len(path) == expected_slots + 1
                for path in trajectory_by_episode[episode_index].values()
            )
            and set(trajectory_by_episode[episode_index])
            == trajectory_uav_ids_by_episode.get(episode_index, set())
        )
    )
    if complete_trajectory_episode_ids:
        def draw_uav_trajectory() -> Any:
            figure, axis = plt.subplots(figsize=(9, 7))
            colors = plt.get_cmap("tab10")
            uav_ids = sorted(
                {
                    uav_id
                    for episode_index in complete_trajectory_episode_ids
                    for uav_id in trajectory_by_episode[episode_index]
                }
            )
            for uav_index, uav_id in enumerate(uav_ids):
                color = colors(uav_index % 10)
                first_episode = complete_trajectory_episode_ids[0]
                last_episode = complete_trajectory_episode_ids[-1]
                for episode_index in complete_trajectory_episode_ids:
                    path = trajectory_by_episode[episode_index][uav_id]
                    axis.plot(
                        [position[0] for position in path],
                        [position[1] for position in path],
                        color=color,
                        alpha=0.22,
                        linewidth=0.8,
                        label=uav_id if episode_index == first_episode else None,
                    )
                    axis.scatter(
                        path[0][0],
                        path[0][1],
                        color=color,
                        marker="o",
                        s=14,
                        alpha=0.35,
                        label="Episode starts" if episode_index == first_episode and uav_index == 0 else None,
                    )
                    axis.scatter(
                        path[-1][0],
                        path[-1][1],
                        color=color,
                        marker="x",
                        s=18,
                        alpha=0.45,
                        label="Episode ends" if episode_index == first_episode and uav_index == 0 else None,
                    )
                first_path = trajectory_by_episode[first_episode][uav_id]
                last_path = trajectory_by_episode[last_episode][uav_id]
                axis.annotate(
                    f"{uav_id} start",
                    first_path[0],
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize="small",
                )
                axis.annotate(
                    f"{uav_id} end",
                    last_path[-1],
                    xytext=(4, -10),
                    textcoords="offset points",
                    fontsize="small",
                )
            axis.set_xlabel("X coordinate (m)")
            axis.set_ylabel("Y coordinate (m)")
            axis.set_title("Actual UAV Trajectories (complete recorded Episodes)")
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(alpha=0.2)
            axis.legend()
            return figure

        save_plot("uav_trajectory.png", draw_uav_trajectory)
    else:
        skipped["uav_trajectory.png"] = (
            "没有完整且连续的逐 Slot 实际 UAV 起止坐标记录；未根据候选目标位置补绘轨迹。"
        )

    if latest_plot_paths:
        def draw_progress_overview() -> Any:
            images = [path for path in latest_plot_paths if path.is_file()]
            columns = 2
            rows = max(1, math.ceil(len(images) / columns))
            figure, axes = plt.subplots(
                rows,
                columns,
                figsize=(14, 5.2 * rows),
                squeeze=False,
            )
            for axis, image_path in zip(
                axes.flat, images, strict=False
            ):
                axis.imshow(plt.imread(image_path))
                axis.set_title(image_path.stem.replace("_", " "))
                axis.set_axis_off()
            for axis in list(axes.flat)[len(images):]:
                axis.set_axis_off()
            figure.suptitle(
                f"DQN Training Progress — {len(episode_rows)} completed Episodes"
            )
            return figure

        overview = None
        try:
            overview = draw_progress_overview()
            latest_overview = output_dir / "training_progress_latest.png"
            _save_figure(overview, latest_overview)
            generated.append(str(latest_overview))
            if snapshot_episode is not None:
                snapshot_overview = output_dir / f"progress_episode_{snapshot_episode:04d}.png"
                _save_figure(overview, snapshot_overview)
                generated.append(str(snapshot_overview))
        except Exception as exc:
            failed["training_progress_overview"] = (
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            if overview is not None:
                plt.close(overview)
    else:
        skipped["training_progress_latest.png"] = (
            "没有任何图表成功生成，无法组合训练总览。"
        )

    result = {
        "status": "FAILED" if failed and not generated else "PARTIAL" if failed or skipped else "COMPLETED",
        "run_dir": str(root),
        "plot_dir": str(output_dir),
        "snapshot_episode": snapshot_episode,
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "data_rows": {
            "episodes": len(episode_rows),
            "actual_cache_frequency_episodes": len(cache_frequency_points),
            "actual_cache_node_metric_episodes": len(cache_node_metric_episode_ids),
            "actual_cache_service_metric_episodes": len(cache_service_points),
            "actual_uav_movement_metric_episodes": len(uav_frequency_points),
            "actual_uav_distance_metric_episodes": len(uav_distance_points),
            "updates": update_row_count,
            "slots": step_row_count,
            "completed_episode_attempts": len(completed_attempts),
            "complete_slot_record_episodes": len(cost_complete_episode_ids),
            "task_utility_points": len(cost_series["task_utility"]),
            "cache_cost_penalty_points": len(cost_series["cache_cost_penalty"]),
            "uav_movement_cost_penalty_points": len(cost_series["uav_movement_cost_penalty"]),
            "cost_metric_missing_episode_points": {
                key: max(0, len(cost_complete_episode_ids) - len(points))
                for key, points in cost_series.items()
            },
            "gradient_norm_updates": len(gradient_norm_points),
            "gradient_clipped_updates": len(clipped_gradient_points),
            "target_network_sync_events": len(target_sync_steps),
            "epsilon_slots": len(epsilon_points),
            "actual_execution_node_known_slots": known_slots,
            "actual_execution_node_unavailable_slots": unavailable_slots,
            "complete_actual_uav_trajectory_episodes": len(
                complete_trajectory_episode_ids
            ),
        },
        "incomplete_trailing_jsonl_lines_ignored": {
            "episodes": incomplete_episode_tails,
            "updates": update_stats["incomplete_tail_count"],
            "slots": max(
                step_stats["incomplete_tail_count"],
                epsilon_stats["incomplete_tail_count"],
            ),
        },
    }
    manifest_paths = [output_dir / "plot_summary.json"]
    if snapshot_episode is not None:
        manifest_paths.append(
            output_dir / f"plot_summary_episode_{snapshot_episode:04d}.json"
        )
    manifest_text = json.dumps(
        result, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    for manifest_path in manifest_paths:
        temporary_path = manifest_path.with_name(
            f"{manifest_path.name}.{uuid4().hex}.tmp"
        )
        try:
            temporary_path.write_text(manifest_text, encoding="utf-8")
            os.replace(temporary_path, manifest_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--snapshot-episode", type=int, default=None)
    args = parser.parse_args()
    result = generate_training_plots(
        args.run_dir, snapshot_episode=args.snapshot_episode
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
