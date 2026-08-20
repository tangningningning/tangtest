# -*- coding: utf-8 -*-
"""正式实验图绘制接口。

本模块只负责把已经存在的 Python 结果对象绘制为 PNG，不读取训练日志，
也不执行 environment、Matching、KKT 或 PPO。图5的两个比例是独立指标：
缓存服务可用率表示服务程序是否存在于任务车辆自己的固定 local cache 或
当前 slot 的 SV/RSU/UAV cache；Cloud execution ratio 则来自最终 Matching。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

# 正式绘图只输出 PNG；Agg 使无GUI环境下的论文图生成保持确定可用。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


__all__ = [
    "plot_training_convergence",
    "plot_cache_availability_cloud_ratio",
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
    moving_average_window: int,
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
    if window > len(records):
        raise ValueError("moving_average_window 不能大于 episode 数量。")

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
            label="Episode Mean Reward",
        )
        axis.plot(
            moving_episode_indices,
            moving_average,
            linewidth=2.0,
            label="Moving Average",
        )
        axis.set_xlabel("Training Episode")
        axis.set_ylabel("Mean Reward per Slot")
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


def plot_cache_availability_cloud_ratio(
    algorithm_results: Sequence[Any],
    output_path: str | Path,
) -> Path:
    """绘制缓存服务可用率与云端执行比例的并列柱状图。

    缓存服务可用率按到达任务统计：任务请求服务若存在于该任务车辆自己
    的 fixed local cache，或存在于当前最终 configuration 的任一 SV/RSU/UAV
    cache，则记为可用。该指标不是 cache hit rate，服务可用也不代表最终
    Matching 一定选择 Local/Edge 执行。Cloud execution ratio 严格读取最终
    Actual-State Matching 中 target == CLOUD 的任务数。两个比例不互补，
    因此使用 grouped bar 而不是 stacked bar，并共享 0--100% Y 轴。
    """

    records = _records(algorithm_results, "algorithm_results")
    output = _png_path(output_path)
    names: list[str] = []
    availability: list[float] = []
    cloud_ratio: list[float] = []
    for index, record in enumerate(records):
        raw_name = _field(record, "algorithm_name", index)
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"第{index}条结果的 algorithm_name 不能为空。")
        name = raw_name.strip()
        if name in names:
            raise ValueError(f"algorithm_name 不能重复：{name}")
        names.append(name)

        availability_value = _finite_float(
            _field(record, "cache_service_availability_rate", index),
            "cache_service_availability_rate",
            index,
        )
        cloud_value = _finite_float(
            _field(record, "cloud_execution_ratio", index),
            "cloud_execution_ratio",
            index,
        )
        if not 0.0 <= availability_value <= 1.0:
            raise ValueError("cache_service_availability_rate 必须位于 [0,1]。")
        if not 0.0 <= cloud_value <= 1.0:
            raise ValueError("cloud_execution_ratio 必须位于 [0,1]。")
        availability.append(availability_value)
        cloud_ratio.append(cloud_value)

    positions = np.arange(len(names), dtype=np.float64)
    width = 0.38
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    try:
        # 两个比例含义不同，不做 stacked bar，也不假设二者和为1。
        axis.bar(
            positions - width / 2.0,
            np.asarray(availability) * 100.0,
            width,
            label="Cache Service Availability",
        )
        axis.bar(
            positions + width / 2.0,
            np.asarray(cloud_ratio) * 100.0,
            width,
            label="Cloud Execution",
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(names)
        axis.set_ylabel("Ratio (%)")
        axis.set_ylim(0.0, 100.0)
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
