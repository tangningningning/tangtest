"""训练/评估结果后处理绘图入口。

本文件只读取已经保存的 CSV/JSON，不重新执行 environment、solver、matching、
KKT 或 PPO。默认正式主图统一为 ``reward_convergence`` 和
``final_execution_rate``；旧 Stage15 对照图和 learning-check 诊断图继续保留。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
MPL_CONFIG_DIR = PROJECT_ROOT / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
# 必须在导入 pyplot 前固定到项目内可写目录，避免依赖用户手工设置权限或环境变量。
os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG_DIR)

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_METRIC_FIELDS = {
    "phase",
    "episode",
    "episode_reward_mean",
    "total_task_count",
    "completed_count",
    "failed_count",
    "local_count",
    "sv_count",
    "rsu_count",
    "uav_count",
    "cloud_count",
}
EXECUTION_COUNT_KEYS = (
    "local",
    "service_vehicle",
    "rsu",
    "uav",
    "cloud",
)
METRIC_COUNT_FIELDS = {
    "local": "local_count",
    "service_vehicle": "sv_count",
    "rsu": "rsu_count",
    "uav": "uav_count",
    "cloud": "cloud_count",
}
STAGE15_EXECUTION_KEYS = (
    "local",
    "service_vehicle",
    "rsu",
    "uav",
    "cloud",
    "failed",
)
STAGE15_EXECUTION_LABELS = (
    "Local",
    "Service Vehicle",
    "RSU",
    "UAV",
    "Cloud",
    "Failed",
)

MAIN_REWARD_RAW_COLOR = "#78B4D4"
MAIN_REWARD_MA_COLOR = "#F28E2B"
MAIN_EXECUTION_BAR_COLOR = "#5B9BD5"
MAIN_EXECUTION_LABELS = (
    "Local",
    "Service Vehicle",
    "RSU",
    "UAV",
    "Cloud",
)


def _read_metrics(metrics_file: str | Path) -> list[dict[str, str]]:
    """读取并检查 episode-level metrics.csv。"""

    path = Path(metrics_file).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"metrics.csv 不存在：{path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = REQUIRED_METRIC_FIELDS.difference(fields)
        if missing:
            raise ValueError(f"metrics.csv 缺少字段：{sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("metrics.csv 没有 episode 记录。")
    return rows


def _as_float(row: Mapping[str, str], field: str, row_index: int) -> float:
    """把 CSV 数值字段转换为有限 float。"""

    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"metrics.csv 第 {row_index} 行字段 {field} 不是数值。") from error
    if not np.isfinite(value):
        raise ValueError(f"metrics.csv 第 {row_index} 行字段 {field} 为 NaN/Inf。")
    return value


def _as_nonnegative_int(row: Mapping[str, str], field: str, row_index: int) -> int:
    """把 CSV 计数字段转换为非负整数。"""

    value = _as_float(row, field, row_index)
    numeric = int(value)
    if numeric < 0 or float(numeric) != value:
        raise ValueError(f"metrics.csv 第 {row_index} 行字段 {field} 必须是非负整数。")
    return numeric


def _final_eval_execution_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    """只聚合 final_eval 行，并验证 completed/failed/total 计数闭合。"""

    final_rows = [
        row for row in rows if str(row.get("phase", "")).strip() == "final_eval"
    ]
    if not final_rows:
        raise ValueError("metrics.csv 中没有 phase=final_eval，禁止生成 Figure 2。")
    totals = {key: 0 for key in EXECUTION_COUNT_KEYS}
    failed_total = 0
    for row_index, row in enumerate(final_rows, start=1):
        counts = {
            key: _as_nonnegative_int(row, field, row_index)
            for key, field in METRIC_COUNT_FIELDS.items()
        }
        completed = _as_nonnegative_int(row, "completed_count", row_index)
        failed = _as_nonnegative_int(row, "failed_count", row_index)
        total = _as_nonnegative_int(row, "total_task_count", row_index)
        if sum(counts.values()) != completed:
            raise ValueError(
                "final_eval 五类执行计数之和不等于 completed_count："
                f"sum={sum(counts.values())}, completed={completed}"
            )
        if completed + failed != total:
            raise ValueError(
                "final_eval completed_count + failed_count 不等于 total_task_count："
                f"completed={completed}, failed={failed}, total={total}"
            )
        for key, value in counts.items():
            totals[key] += value
        failed_total += failed
    totals["failed"] = failed_total
    return totals


def _train_reward_series(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """只抽取 train phase 的 slot 平均 reward，拒绝使用总 reward。"""

    train_rows = [row for row in rows if str(row.get("phase", "")).strip() == "train"]
    if not train_rows:
        raise ValueError(
            "metrics.csv 中没有 phase=train 记录，无法绘制 Figure 1；"
            "请先记录完整 train episode。"
        )
    indexed = []
    for row_index, row in enumerate(train_rows, start=1):
        try:
            episode = int(row["episode"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"train metrics 第 {row_index} 行 episode 不是整数。") from error
        indexed.append((episode, _as_float(row, "episode_reward_mean", row_index)))
    indexed.sort(key=lambda item: item[0])
    episodes = np.asarray([item[0] for item in indexed], dtype=np.int64)
    rewards = np.asarray([item[1] for item in indexed], dtype=np.float64)
    if np.any(np.diff(episodes) <= 0):
        raise ValueError("phase=train 的 episode 必须严格递增且不能重复。")
    return episodes, rewards


def _read_csv_rows(
    csv_file: str | Path,
    *,
    required_fields: Sequence[str],
    description: str,
) -> tuple[Path, list[dict[str, str]]]:
    """读取绘图 CSV，并对缺失列给出明确错误。"""

    path = Path(csv_file).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{description}不存在：{path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = sorted(set(required_fields).difference(fields))
        if missing:
            raise ValueError(
                f"{description}缺少列：{missing}；当前列为：{sorted(fields)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{description}没有数据行：{path}")
    return path, rows


def _learning_check_training_series(
    training_episodes_file: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 learning-check episode CSV 的 reward sum/steps。"""

    path, rows = _read_csv_rows(
        training_episodes_file,
        required_fields=("episode", "sum_raw_G", "steps"),
        description="training_episodes.csv",
    )
    episodes: list[int] = []
    rewards: list[float] = []
    steps: list[int] = []
    for index, row in enumerate(rows, start=2):
        # SB3 terminal后的最后一个未完成片段只用于审计，不作为正式
        # episode convergence 点，避免短片段把曲线末端人为拉高/拉低。
        if str(row.get("completed", "true")).strip().lower() in {
            "false",
            "0",
            "no",
        }:
            continue
        try:
            episode = int(row["episode"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}第{index}行episode不是整数。") from error
        episodes.append(episode)
        rewards.append(_as_float(row, "sum_raw_G", index))
        steps.append(_as_nonnegative_int(row, "steps", index))
        if steps[-1] <= 0:
            raise ValueError(f"{path}第{index}行steps必须为正整数。")
    episode_array = np.asarray(episodes, dtype=np.int64)
    reward_array = np.asarray(rewards, dtype=np.float64)
    step_array = np.asarray(steps, dtype=np.int64)
    order = np.argsort(episode_array, kind="stable")
    episode_array = episode_array[order]
    reward_array = reward_array[order]
    step_array = step_array[order]
    if np.any(np.diff(episode_array) <= 0):
        raise ValueError(f"{path}的episode必须严格递增且不能重复。")
    _reward_per_slot_series(episode_array, reward_array, step_array)
    return episode_array, reward_array, step_array


def _find_count_field(
    fields: set[str], aliases: Sequence[str], *, label: str, csv_file: Path
) -> str:
    for alias in aliases:
        if alias in fields:
            return alias
    raise ValueError(
        f"{csv_file}缺少{label}列；候选列为：{list(aliases)}；"
        f"当前列为：{sorted(fields)}"
    )


def _learning_check_execution_counts(
    validation_results_file: str | Path,
) -> dict[str, int]:
    """聚合 validation seeds 的五类 completed 与 Failed 计数。"""

    path, rows = _read_csv_rows(
        validation_results_file,
        required_fields=(),
        description="validation_results.csv",
    )
    fields = set(rows[0].keys())
    aliases = {
        "local": ("Local_count", "local_count", "local_tasks"),
        "service_vehicle": (
            "SV_count",
            "sv_count",
            "service_vehicle_count",
            "service_vehicle_tasks",
        ),
        "rsu": ("RSU_count", "rsu_count", "rsu_tasks"),
        "uav": ("UAV_count", "uav_count", "uav_tasks"),
        "cloud": ("Cloud_count", "cloud_count", "cloud_tasks"),
        "failed": ("Failed_count", "failed_count", "failed_tasks"),
    }
    columns = {
        key: _find_count_field(fields, names, label=key, csv_file=path)
        for key, names in aliases.items()
    }
    totals = {key: 0 for key in aliases}
    for row_index, row in enumerate(rows, start=2):
        for key, column in columns.items():
            totals[key] += _as_nonnegative_int(row, column, row_index)
    return totals


def plot_learning_check_run(
    training_episodes_file: str | Path,
    validation_results_file: str | Path,
    output_dir: str | Path,
    *,
    moving_average_window: int = 10,
) -> dict[str, tuple[Path, Path]]:
    """从 learning-check 已保存 CSV 生成两张正式主图。"""

    episodes, raw_rewards, steps = _learning_check_training_series(
        training_episodes_file
    )
    reward_paths = plot_training_reward_curve(
        episodes,
        raw_rewards,
        steps,
        output_dir,
        moving_average_window=moving_average_window,
        file_stem="reward_convergence",
    )
    counts = _learning_check_execution_counts(validation_results_file)
    execution_paths = plot_final_execution_rate(
        counts["local"],
        counts["service_vehicle"],
        counts["rsu"],
        counts["uav"],
        counts["cloud"],
        counts["failed"],
        output_dir,
        file_stem="final_execution_rate",
    )
    return {
        "reward_convergence": reward_paths,
        "final_execution_rate": execution_paths,
    }


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """实现 rolling mean，使用 min_periods=1 处理 episode 少于 window 的情况。"""

    if int(window) <= 0:
        raise ValueError("moving average window 必须为正整数。")
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    for index in range(values.size):
        start = max(0, index - int(window) + 1)
        result[index] = float(np.mean(values[start : index + 1]))
    return result


def _save_main_figure(
    figure: Any,
    output_dir: str | Path,
    file_stem: str,
    *,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """以统一的 PNG/PDF 命名保存正式展示图。"""

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    png_path = destination / f"{file_stem}.png"
    pdf_path = destination / f"{file_stem}.pdf"
    try:
        figure.tight_layout()
        figure.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
        figure.savefig(pdf_path, bbox_inches="tight")
    finally:
        plt.close(figure)
    return png_path, pdf_path


def _reward_per_slot_series(
    episodes: Sequence[Any],
    raw_rewards: Sequence[Any],
    steps: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """把 episode 原始 reward sum 严格转换成每 slot 平均 reward。"""

    episode_array = np.asarray(episodes, dtype=np.int64).reshape(-1)
    reward_array = np.asarray(raw_rewards, dtype=np.float64).reshape(-1)
    step_array = np.asarray(steps, dtype=np.float64).reshape(-1)
    if not (episode_array.size == reward_array.size == step_array.size):
        raise ValueError(
            "训练奖励绘图输入长度不一致：episodes、raw_rewards、steps必须逐episode对应。"
        )
    if episode_array.size == 0:
        raise ValueError("训练奖励绘图没有episode记录。")
    if np.any(~np.isfinite(reward_array)) or np.any(~np.isfinite(step_array)):
        raise ValueError("训练奖励绘图输入不能包含NaN/Inf。")
    if np.any(step_array <= 0.0) or np.any(step_array != np.floor(step_array)):
        raise ValueError("训练奖励绘图的steps必须是正整数。")
    order = np.argsort(episode_array, kind="stable")
    episode_array = episode_array[order]
    reward_array = reward_array[order]
    step_array = step_array[order]
    if np.any(np.diff(episode_array) <= 0):
        raise ValueError("训练奖励绘图的episode必须严格递增且不能重复。")
    return episode_array, reward_array / step_array


def plot_training_reward_curve(
    episodes: Sequence[Any],
    raw_rewards: Sequence[Any],
    steps: Sequence[Any],
    output_dir: str | Path,
    *,
    moving_average_window: int = 10,
    file_stem: str = "reward_convergence",
) -> tuple[Path, Path]:
    """绘制统一主图：Average Reward per Slot 与 Raw/MA10。"""

    episode_array, rewards_per_slot = _reward_per_slot_series(
        episodes,
        raw_rewards,
        steps,
    )
    moving_average = _moving_average(rewards_per_slot, moving_average_window)
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(
        episode_array,
        rewards_per_slot,
        color=MAIN_REWARD_RAW_COLOR,
        linewidth=1.2,
        alpha=0.85,
        label="Raw Average Reward",
    )
    axis.plot(
        episode_array,
        moving_average,
        color=MAIN_REWARD_MA_COLOR,
        linewidth=2.2,
        label=f"MA{int(moving_average_window)}",
    )
    axis.set_title("Average Reward per Slot")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Average Reward per Slot")
    axis.grid(True, alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right")
    return _save_main_figure(figure, output_dir, file_stem)


def _nonnegative_count(value: Any, name: str) -> int:
    """验证绘图使用的执行计数是非负整数。"""

    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}必须是非负整数。") from error
    if not np.isfinite(numeric) or numeric < 0.0 or numeric != float(int(numeric)):
        raise ValueError(f"{name}必须是非负整数。")
    return int(numeric)


def plot_final_execution_rate(
    local_count: Any,
    service_vehicle_count: Any,
    rsu_count: Any,
    uav_count: Any,
    cloud_count: Any,
    failed_count: Any,
    output_dir: str | Path,
    *,
    file_stem: str = "final_execution_rate",
) -> tuple[Path, Path]:
    """绘制五类最终执行率；Failed只显示在右上角，不占柱子。"""

    counts = np.asarray(
        [
            _nonnegative_count(local_count, "local_count"),
            _nonnegative_count(service_vehicle_count, "service_vehicle_count"),
            _nonnegative_count(rsu_count, "rsu_count"),
            _nonnegative_count(uav_count, "uav_count"),
            _nonnegative_count(cloud_count, "cloud_count"),
        ],
        dtype=np.int64,
    )
    failed = _nonnegative_count(failed_count, "failed_count")
    completed = int(np.sum(counts))
    rates = (
        np.zeros(5, dtype=np.float64)
        if completed == 0
        else 100.0 * counts.astype(np.float64) / float(completed)
    )
    if completed > 0 and not np.isclose(
        float(np.sum(rates)), 100.0, rtol=0.0, atol=1.0e-10
    ):
        raise RuntimeError("五类最终 execution rate 未归一到100%。")

    positions = np.arange(len(MAIN_EXECUTION_LABELS), dtype=np.float64)
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    bars = axis.bar(
        positions,
        rates,
        color=MAIN_EXECUTION_BAR_COLOR,
        width=0.62,
    )
    axis.set_title("Final Execution Rate")
    axis.set_xlabel("Execution Node")
    axis.set_ylabel("Execution Rate (%)")
    axis.set_xticks(positions)
    axis.set_xticklabels(MAIN_EXECUTION_LABELS)
    axis.set_ylim(0.0, max(100.0, float(np.max(rates)) + 12.0))
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.text(
        0.98,
        0.97,
        f"Completed={completed}\nFailed={failed}",
        transform=axis.transAxes,
        ha="right",
        va="top",
    )
    label_offset = max(1.0, float(np.max(rates)) * 0.015)
    for bar, rate in zip(bars, rates):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(rate) + label_offset,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
        )
    return _save_main_figure(figure, output_dir, file_stem)


def _stage15_finite_float(value: Any, name: str) -> float:
    """读取 Stage15 原始汇总中的有限浮点数。"""

    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Stage15 {name} 不是数值") from error
    if not np.isfinite(numeric):
        raise ValueError(f"Stage15 {name} 不能是 NaN/Inf")
    return numeric


def _stage15_raw_episode_series(
    summary: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """只读取已保存的 episode-level 原始 system reward。"""

    records = summary.get("training_episodes", [])
    if not isinstance(records, list) or not records:
        raise ValueError("training_summary 缺少 training_episodes 原始序列")
    episodes: list[int] = []
    rewards: list[float] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"training_episodes[{index - 1}] 不是对象")
        raw_value = record.get("system_reward")
        if raw_value is None:
            raw_value = record.get("raw_reward", record.get("reward"))
        # system_reward/reward 都来自原始 reward 累计；明确排除 normalized reward。
        raw_value = record.get("system_reward")
        if raw_value is None:
            raw_value = record.get("raw_reward", record.get("reward"))
        episodes.append(index)
        rewards.append(
            _stage15_finite_float(raw_value, f"training_episodes[{index - 1}].raw_reward")
        )
    return (
        np.asarray(episodes, dtype=np.int64),
        np.asarray(rewards, dtype=np.float64),
    )


def _stage15_training_series(
    summary: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 Stage15 的原始 reward sum 和 episode steps。"""

    records = summary.get("training_episodes", [])
    if not isinstance(records, list) or not records:
        raise ValueError("training_summary 缺少 training_episodes 原始序列")
    episodes: list[int] = []
    rewards: list[float] = []
    steps: list[int] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"training_episodes[{index - 1}] 不是对象")
        raw_value = record.get("system_reward")
        if raw_value is None:
            raw_value = record.get("raw_reward", record.get("reward"))
        if raw_value is None:
            raise ValueError(f"training_episodes[{index - 1}] 缺少原始 reward 列")
        step_value = record.get("steps")
        if step_value is None:
            raise ValueError(
                f"training_episodes[{index - 1}] 缺少 steps；"
                "Average Reward per Slot不能由累计reward替代。"
            )
        episodes.append(index)
        rewards.append(
            _stage15_finite_float(raw_value, f"training_episodes[{index - 1}].raw_reward")
        )
        steps.append(_nonnegative_count(step_value, f"training_episodes[{index - 1}].steps"))
        if steps[-1] <= 0:
            raise ValueError(f"training_episodes[{index - 1}].steps必须为正整数")
    return (
        np.asarray(episodes, dtype=np.int64),
        np.asarray(rewards, dtype=np.float64),
        np.asarray(steps, dtype=np.int64),
    )


def _stage15_validation_series(
    summary: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从保存的 validation summary 提取 update/Reference/PPO 原始 reward。"""

    records = summary.get("validation", [])
    if not isinstance(records, list) or not records:
        raise ValueError("training_summary 缺少 validation summary")
    updates: list[int] = []
    reference_rewards: list[float] = []
    ppo_rewards: list[float] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"validation[{index - 1}] 不是对象")
        update_value = record.get("update_index")
        if update_value is None:
            match = re.search(r"(\d+)", str(record.get("checkpoint", "")))
            if match is None:
                raise ValueError(f"validation[{index - 1}] 缺少 update index")
            update_value = match.group(1)
        try:
            update = int(update_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"validation[{index - 1}] update index 无效") from error
        aggregate = record.get("aggregate", {})
        if not isinstance(aggregate, Mapping):
            raise ValueError(f"validation[{index - 1}] 缺少 aggregate")
        updates.append(update)
        reference_rewards.append(
            _stage15_finite_float(
                aggregate.get("mean_reference_reward"),
                f"validation[{index - 1}].mean_reference_reward",
            )
        )
        ppo_rewards.append(
            _stage15_finite_float(
                aggregate.get("mean_ppo_reward"),
                f"validation[{index - 1}].mean_ppo_reward",
            )
        )
    order = np.argsort(np.asarray(updates, dtype=np.int64))
    updates_array = np.asarray(updates, dtype=np.int64)[order]
    if np.any(np.diff(updates_array) <= 0):
        raise ValueError("validation update index 必须严格递增")
    return (
        updates_array,
        np.asarray(reference_rewards, dtype=np.float64)[order],
        np.asarray(ppo_rewards, dtype=np.float64)[order],
    )


def _stage15_nonnegative_count(value: Any, name: str) -> int:
    """校验 Stage15 test execution count。"""

    numeric = _stage15_finite_float(value, name)
    integer = int(numeric)
    if integer < 0 or float(integer) != numeric:
        raise ValueError(f"Stage15 {name} 必须是非负整数")
    return integer


def _stage15_controller_counts(
    evaluation: Mapping[str, Any],
    controller: str,
) -> tuple[dict[str, int], int]:
    """按总任务数汇总 test seeds 的 Reference/PPO 执行位置。"""

    rows = evaluation.get("per_seed", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("Stage15 test summary 缺少 per_seed")
    counts = {key: 0 for key in STAGE15_EXECUTION_KEYS}
    total_tasks = 0
    for row_index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"test per_seed[{row_index - 1}] 不是对象")
        metrics = row.get(controller, {})
        if not isinstance(metrics, Mapping):
            raise ValueError(f"test per_seed[{row_index - 1}] 缺少 {controller}")
        execution = metrics.get("execution_counts", {})
        if not isinstance(execution, Mapping):
            raise ValueError(f"test per_seed[{row_index - 1}] execution_counts 无效")
        row_counts = {
            key: _stage15_nonnegative_count(
                execution.get(key, 0),
                f"test per_seed[{row_index - 1}].{controller}.{key}",
            )
            for key in STAGE15_EXECUTION_KEYS
        }
        if "failed" not in execution:
            row_counts["failed"] = _stage15_nonnegative_count(
                metrics.get("failed_tasks", 0),
                f"test per_seed[{row_index - 1}].{controller}.failed_tasks",
            )
        row_total = metrics.get("task_total")
        if row_total is None:
            row_total_value = sum(row_counts.values())
        else:
            row_total_value = _stage15_nonnegative_count(
                row_total,
                f"test per_seed[{row_index - 1}].{controller}.task_total",
            )
        if sum(row_counts.values()) != row_total_value:
            raise ValueError(
                f"test per_seed[{row_index - 1}] {controller} 执行计数不等于总任务数"
            )
        for key, value in row_counts.items():
            counts[key] += value
        total_tasks += row_total_value
    if total_tasks <= 0:
        raise ValueError(f"Stage15 test {controller} 没有总任务")
    return counts, total_tasks


def stage15_execution_rate_data(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """返回 Stage15 Figure 2 使用的总任务分母和 Reference/PPO 比例。"""

    evaluation = summary.get("test", summary)
    if not isinstance(evaluation, Mapping):
        raise ValueError("Stage15 summary 缺少 test summary")
    result: dict[str, Any] = {
        "categories": list(STAGE15_EXECUTION_KEYS),
        "labels": list(STAGE15_EXECUTION_LABELS),
    }
    for controller in ("reference", "ppo"):
        counts, total_tasks = _stage15_controller_counts(evaluation, controller)
        rates = {
            key: 100.0 * float(counts[key]) / float(total_tasks)
            for key in STAGE15_EXECUTION_KEYS
        }
        if not np.isclose(sum(rates.values()), 100.0, rtol=0.0, atol=1.0e-10):
            raise RuntimeError(f"Stage15 {controller} execution rate 未归一到 100%")
        result[controller] = {
            "counts": counts,
            "total_tasks": total_tasks,
            "rates": rates,
        }
    return result


def _save_stage15_figure(
    figure: Any,
    destination: Path,
    stem: str,
) -> tuple[Path, Path]:
    """统一保存 Stage15 PNG/PDF，并及时关闭 Figure。"""

    destination.mkdir(parents=True, exist_ok=True)
    png_path = destination / f"{stem}.png"
    pdf_path = destination / f"{stem}.pdf"
    try:
        figure.tight_layout()
        figure.savefig(png_path, dpi=300)
        figure.savefig(pdf_path)
    finally:
        plt.close(figure)
    return png_path, pdf_path


def plot_stage15_reward_convergence(
    summary: Mapping[str, Any],
    output_dir: str | Path,
    *,
    moving_average_window: int = 10,
) -> tuple[Path, Path]:
    """兼容旧 Stage15 API，但输出统一的正式主图名称。"""

    episodes, rewards, steps = _stage15_training_series(summary)
    return plot_training_reward_curve(
        episodes,
        rewards,
        steps,
        output_dir,
        moving_average_window=moving_average_window,
        file_stem="reward_convergence",
    )


def plot_stage15_final_execution_rate(
    summary: Mapping[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """从 Stage15 最终 test summary 的 PPO 结果生成五柱正式主图。"""

    data = stage15_execution_rate_data(summary)
    counts = data["ppo"]["counts"]
    return plot_final_execution_rate(
        counts["local"],
        counts["service_vehicle"],
        counts["rsu"],
        counts["uav"],
        counts["cloud"],
        counts["failed"],
        output_dir,
        file_stem="final_execution_rate",
    )


def plot_stage15_validation_reward(
    summary: Mapping[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """绘制五个 checkpoint 的 Reference/PPO 原始 validation reward。"""

    updates, reference_rewards, ppo_rewards = _stage15_validation_series(summary)
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(updates, reference_rewards, marker="o", label="Reference mean raw reward")
    axis.plot(updates, ppo_rewards, marker="s", label="PPO mean raw reward")
    axis.set_xlabel("PPO Update")
    axis.set_ylabel("Raw System Reward")
    axis.set_title("Stage15 Validation Reward")
    axis.set_xticks(updates.tolist())
    axis.grid(True, alpha=0.25)
    axis.legend()
    return _save_stage15_figure(
        figure,
        Path(output_dir).expanduser(),
        "stage15_validation_reward",
    )


def plot_stage15_execution_rate(
    summary: Mapping[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """绘制最终十个 test seed 的总任务执行率，包含 Failed。"""

    data = stage15_execution_rate_data(summary)
    categories = data["categories"]
    reference_rates = [data["reference"]["rates"][key] for key in categories]
    ppo_rates = [data["ppo"]["rates"][key] for key in categories]
    positions = np.arange(len(categories), dtype=np.float64)
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.bar(positions - width / 2.0, reference_rates, width, label="Reference")
    axis.bar(positions + width / 2.0, ppo_rates, width, label="PPO")
    axis.set_xticks(positions)
    axis.set_xticklabels(data["labels"])
    axis.set_xlabel("Execution Node")
    axis.set_ylabel("Rate (%)")
    axis.set_title("Stage15 Final Test Execution Rate")
    axis.set_ylim(0.0, max(100.0, max(reference_rates + ppo_rates) + 10.0))
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.text(
        0.99,
        0.98,
        "Denominator: total tasks\\n"
        f"Reference={data['reference']['total_tasks']}  "
        f"PPO={data['ppo']['total_tasks']}",
        transform=axis.transAxes,
        ha="right",
        va="top",
    )
    return _save_stage15_figure(
        figure,
        Path(output_dir).expanduser(),
        "stage15_execution_rate",
    )


def plot_stage15_run(
    summary: Mapping[str, Any],
    output_dir: str | Path,
    *,
    moving_average_window: int = 10,
) -> dict[str, tuple[Path, Path]]:
    """后处理 Stage15 主图和保留的 validation/Reference 对照诊断图。"""

    return {
        "reward_convergence": plot_stage15_reward_convergence(
            summary,
            output_dir,
            moving_average_window=moving_average_window,
        ),
        "final_execution_rate": plot_stage15_final_execution_rate(
            summary,
            output_dir,
        ),
        "validation_reward": plot_stage15_validation_reward(summary, output_dir),
        "execution_rate": plot_stage15_execution_rate(summary, output_dir),
    }


def plot_reward_convergence(
    metrics_file: str | Path,
    output_dir: str | Path,
    *,
    moving_average_window: int = 10,
) -> tuple[Path, Path]:
    """兼容 legacy metrics.csv 的统一 reward 主图入口。"""

    rows = _read_metrics(metrics_file)
    train_rows = [row for row in rows if str(row.get("phase", "")).strip() == "train"]
    if not train_rows:
        raise ValueError("metrics.csv中没有phase=train记录，无法生成reward_convergence。")
    episodes: list[int] = []
    raw_rewards: list[float] = []
    steps: list[int] = []
    for row_index, row in enumerate(train_rows, start=2):
        episodes.append(int(row["episode"]))
        if "episode_reward_sum" in row and "episode_steps" in row:
            raw_rewards.append(_as_float(row, "episode_reward_sum", row_index))
            steps.append(_as_nonnegative_int(row, "episode_steps", row_index))
        else:
            # legacy metrics 的 episode_reward_mean 已经是每 slot平均值；
            # 用 steps=1 传入统一绘图器，不把它误当累计 reward。
            raw_rewards.append(_as_float(row, "episode_reward_mean", row_index))
            steps.append(1)
    return plot_training_reward_curve(
        episodes,
        raw_rewards,
        steps,
        output_dir,
        moving_average_window=moving_average_window,
        file_stem="reward_convergence",
    )


def plot_execution_rate(
    local_tasks: int,
    sv_tasks: int,
    rsu_tasks: int,
    uav_tasks: int,
    cloud_tasks: int,
    failed_tasks: int,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """兼容 legacy 参数名的统一最终执行率主图入口。"""

    return plot_final_execution_rate(
        local_tasks,
        sv_tasks,
        rsu_tasks,
        uav_tasks,
        cloud_tasks,
        failed_tasks,
        output_dir,
        file_stem="final_execution_rate",
    )


def plot_run(
    metrics_file: str | Path,
    output_dir: str | Path,
    *,
    moving_average_window: int = 10,
) -> dict[str, tuple[Path, Path]]:
    """统一生成 Figure 1/2；Figure 2 的唯一来源是 final_eval metrics。"""

    rows = _read_metrics(metrics_file)
    execution_counts = _final_eval_execution_counts(rows)
    return {
        "reward_convergence": plot_reward_convergence(
            metrics_file,
            output_dir,
            moving_average_window=moving_average_window,
        ),
        "final_execution_rate": plot_execution_rate(
            execution_counts["local"],
            execution_counts["service_vehicle"],
            execution_counts["rsu"],
            execution_counts["uav"],
            execution_counts["cloud"],
            execution_counts["failed"],
            output_dir,
        ),
    }


def plot_saved_result_directory(
    result_dir: str | Path,
    *,
    moving_average_window: int = 10,
) -> dict[str, tuple[Path, Path]]:
    """按已有结果文件自动选择 learning-check、Stage15 或 legacy 后处理。"""

    directory = Path(result_dir).expanduser()
    learning_episodes = directory / "training_episodes.csv"
    learning_validation = directory / "validation_results.csv"
    if learning_episodes.is_file() or learning_validation.is_file():
        return plot_learning_check_run(
            learning_episodes,
            learning_validation,
            directory / "figures",
            moving_average_window=moving_average_window,
        )

    training_summary = directory / "training_summary.json"
    if training_summary.is_file():
        with training_summary.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        return plot_stage15_run(
            summary,
            directory / "figures",
            moving_average_window=moving_average_window,
        )

    metrics_file = directory / "metrics.csv"
    if metrics_file.is_file():
        return plot_run(
            metrics_file,
            directory / "figures",
            moving_average_window=moving_average_window,
        )
    raise FileNotFoundError(
        f"{directory}中没有可识别的结果文件：training_episodes.csv、"
        "training_summary.json或metrics.csv。"
    )


def _print_generated_figures(result: Mapping[str, Any]) -> None:
    """按用户要求优先打印两张正式主图，再打印附加诊断图。"""

    ordered_names = ("reward_convergence", "final_execution_rate")
    printed: set[str] = set()
    print("Generated main figures:")
    for name in ordered_names:
        if name in printed or name not in result:
            continue
        paths = result[name]
        if isinstance(paths, (tuple, list)):
            for path in paths:
                print(f"- {Path(path)}")
        else:
            print(f"- {Path(paths)}")
        printed.add(name)
    diagnostic_names = tuple(str(key) for key in result if key not in printed)
    if diagnostic_names:
        print("Generated diagnostic figures:")
        for name in diagnostic_names:
            paths = result[name]
            if isinstance(paths, (tuple, list)):
                for path in paths:
                    print(f"- {Path(path)}")
            else:
                print(f"- {Path(paths)}")


def _build_parser() -> argparse.ArgumentParser:
    """构造结果后处理命令行接口，不执行训练或环境。"""

    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="从已保存训练/评估结果生成统一主图")
    parser.add_argument(
        "--metrics",
        "--metrics-file",
        dest="metrics_file",
        type=Path,
        default=project_root / "outputs" / "metrics.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="自动识别 learning-check、Stage15 或 legacy 结果目录。",
    )
    parser.add_argument(
        "--figure2-only",
        action="store_true",
        help="只从 phase=final_eval 生成 Figure 2。",
    )
    return parser


def main() -> None:
    """只读取已保存结果生成主图，不启动训练。"""

    args = _build_parser().parse_args()
    if args.result_dir is not None:
        result = plot_saved_result_directory(args.result_dir, moving_average_window=10)
    elif args.figure2_only:
        rows = _read_metrics(args.metrics_file)
        counts = _final_eval_execution_counts(rows)
        result = {
            "final_execution_rate": plot_execution_rate(
                counts["local"],
                counts["service_vehicle"],
                counts["rsu"],
                counts["uav"],
                counts["cloud"],
                counts["failed"],
                args.output_dir,
            )
        }
    else:
        result = plot_run(
            args.metrics_file,
            args.output_dir,
            moving_average_window=10,
        )
    _print_generated_figures(result)


if __name__ == "__main__":
    main()
