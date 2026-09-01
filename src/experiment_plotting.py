# -*- coding: utf-8 -*-
"""正式实验图绘制接口。

本模块只负责把已经存在的 Python 结果对象绘制为 PNG，不读取训练日志，
也不执行 environment、Matching、KKT 或 PPO。正式结果只包含：
1. PPO Training Convergence；
2. Task Execution Ratio by Execution Location。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

# 正式绘图只输出 PNG；Agg 使无GUI环境下的论文图生成保持确定可用。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


__all__ = [
    "plot_training_convergence",
    "plot_execution_location_ratios",
]


def _field(record: Any, name: str, record_index: int) -> Any:
    """读取 dataclass/object 或 mapping 中的同名字段。"""

    if isinstance(record, Mapping):
        if name not in record:
            raise ValueError(f"第{record_index}条结果缺少字段：{name}")
        return record[name]
    if not hasattr(record, name):
        raise ValueError(f"第{record_index}条结果缺少字段：{name}")
    return getattr(record, name)


def _records(value: Any, name: str) -> list[Any]:
    if value is None or isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{name} 必须是非空结果序列。")
    try:
        records = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} 必须是非空结果序列。") from exc
    if not records:
        raise ValueError(f"{name} 不能为空。")
    return records


def _png_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".png":
        raise ValueError("正式绘图只支持 .png 输出，不生成 PDF。")
    if path.name in {"", ".png"}:
        raise ValueError("output_path 必须包含有效的 PNG 文件名。")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _finite_float(value: Any, field_name: str, record_index: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"第{record_index}条结果的 {field_name} 不是数值。") from exc
    if not np.isfinite(result):
        raise ValueError(f"第{record_index}条结果的 {field_name} 必须是 finite 数值。")
    return result


def plot_training_convergence(
    episode_summaries: Sequence[Any],
    output_path: str | Path,
    moving_average_window: int = 10,
) -> Path:
    """绘制训练 episode 平均真实系统效用及其滑动平均。

    ``rollout_reward_mean`` 已由训练 episode 汇总定义为
    ``mean_t G(t)``，其中 ``G(t)=Q_actual-C_ca-C_q``；本函数直接使用该
    字段，不重新计算 reward，也不把 PPO loss/value loss 放到图1中。
    moving average 只用于可视化，不是 system_config 或 PPO 训练参数。
    """

    records = _records(episode_summaries, "episode_summaries")
    output = _png_path(output_path)
    if isinstance(moving_average_window, bool):
        raise ValueError("moving_average_window 必须是大于等于1的整数。")
    try:
        window = int(moving_average_window)
    except (TypeError, ValueError) as exc:
        raise ValueError("moving_average_window 必须是大于等于1的整数。") from exc
    if window < 1 or window != moving_average_window:
        raise ValueError("moving_average_window 必须是大于等于1的整数。")
    # 第一阶段可能只有少于10轮；使用已有episode数作为有效窗口，仍保留
    # 原始曲线和moving average，不让绘图阻断训练完成后的reporting。
    window = min(window, len(records))

    episode_indices = np.asarray(
        [_finite_float(_field(record, "episode_index", index), "episode_index", index)
         for index, record in enumerate(records)],
        dtype=np.float64,
    )
    if not np.all(np.equal(episode_indices, np.floor(episode_indices))):
        raise ValueError("episode_index 必须是整数。")
    episode_indices = episode_indices.astype(np.int64)
    if len(np.unique(episode_indices)) != len(episode_indices):
        raise ValueError("episode_index 必须唯一。")
    if len(episode_indices) > 1 and not np.all(np.diff(episode_indices) > 0):
        raise ValueError("episode_index 必须严格递增。")

    rewards = np.asarray(
        [_finite_float(_field(record, "rollout_reward_mean", index), "rollout_reward_mean", index)
         for index, record in enumerate(records)],
        dtype=np.float64,
    )
    prefix = np.concatenate(([0.0], np.cumsum(rewards, dtype=np.float64)))
    moving_average = (prefix[window:] - prefix[:-window]) / float(window)
    moving_episode_indices = episode_indices[window - 1:]

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    try:
        axis.plot(
            episode_indices,
            rewards,
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            label="Mean System Utility G",
        )
        axis.plot(
            moving_episode_indices,
            moving_average,
            linewidth=2.0,
            label=f"{window}-Episode Moving Average",
        )
        axis.set_title("PPO Training Convergence")
        axis.set_xlabel("Training Episode")
        axis.set_ylabel("Mean System Utility G")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.legend()
        figure.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
    finally:
        plt.close(figure)
    return output


EXECUTION_LOCATION_ORDER = ("Local", "SV", "RSU", "UAV", "Cloud")


def plot_execution_location_ratios(
    execution_location_ratios: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """绘制最终Actual-State Matching的五类任务执行位置比例。"""

    if not isinstance(execution_location_ratios, Mapping):
        raise ValueError("execution_location_ratios 必须是mapping。")
    output = _png_path(output_path)
    expected = set(EXECUTION_LOCATION_ORDER)
    actual = set(str(key) for key in execution_location_ratios)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            "execution_location_ratios 类别必须严格为 "
            f"{list(EXECUTION_LOCATION_ORDER)}；missing={missing}, extra={extra}"
        )

    ratios: list[float] = []
    for index, location in enumerate(EXECUTION_LOCATION_ORDER):
        ratios.append(
            _finite_float(
                execution_location_ratios[location],
                f"execution_location_ratios[{location}]",
                index,
            )
        )
        if not 0.0 <= ratios[-1] <= 1.0:
            raise ValueError(f"execution_location_ratios[{location}] 必须位于 [0,1]。")

    positions = np.arange(len(EXECUTION_LOCATION_ORDER), dtype=np.float64)
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    try:
        axis.bar(positions, np.asarray(ratios, dtype=np.float64), width=0.62)
        axis.set_title("Task Execution Ratio by Execution Location")
        axis.set_xlabel("Execution Location")
        axis.set_ylabel("Task Execution Ratio")
        axis.set_xticks(positions)
        axis.set_xticklabels(EXECUTION_LOCATION_ORDER)
        axis.set_ylim(0.0, 1.0)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        figure.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
    finally:
        plt.close(figure)
    return output

