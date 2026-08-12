"""论文基础结果绘图入口。

本文件只读取 episode-level ``metrics.csv``，不重新执行 environment、solver、
matching、KKT 或 PPO。所有图表都使用论文定义的最终量：

* Figure 1：``phase=train`` 的 100-slot ``episode_reward_mean`` 及其移动平均；
* Figure 2：只读取 ``phase=final_eval`` 的最终 Local/SV/RSU/UAV/Cloud 计数。
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Any, Mapping

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
    "SV",
    "RSU",
    "UAV",
    "Cloud",
    "Failed",
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
    """绘制 Stage15 episode 原始 system reward 与 MA10。"""

    episodes, rewards = _stage15_raw_episode_series(summary)
    moving_average = _moving_average(rewards, moving_average_window)
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(episodes, rewards, linewidth=1.0, alpha=0.55, label="Raw System Reward")
    axis.plot(
        episodes,
        moving_average,
        linewidth=2.0,
        label=f"Moving Average ({int(moving_average_window)})",
    )
    axis.set_xlabel("Episode")
    axis.set_ylabel("Raw System Reward")
    axis.set_title("Stage15 PPO Reward Convergence")
    axis.grid(True, alpha=0.25)
    axis.legend()
    return _save_stage15_figure(
        figure,
        Path(output_dir).expanduser(),
        "stage15_reward_convergence",
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
    """只用已保存 Stage15 summary 后处理三张核心论文图。"""

    return {
        "reward_convergence": plot_stage15_reward_convergence(
            summary,
            output_dir,
            moving_average_window=moving_average_window,
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
    """绘制 Figure 1：train slot 平均 reward 原始曲线和移动平均。"""

    rows = _read_metrics(metrics_file)
    episodes, rewards = _train_reward_series(rows)
    moving_average = _moving_average(rewards, moving_average_window)

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    png_path = destination / "figure1_reward_convergence.png"
    pdf_path = destination / "figure1_reward_convergence.pdf"

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(
        episodes,
        rewards,
        linewidth=1.0,
        alpha=0.55,
        label="Raw Average Reward",
    )
    axis.plot(
        episodes,
        moving_average,
        linewidth=2.0,
        label=f"MA{int(moving_average_window)}",
    )
    axis.set_xlabel("Episode")
    axis.set_ylabel("Average Reward per Slot")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def _nonnegative_count(value: Any, name: str) -> int:
    """验证最终执行计数是非负整数。"""

    try:
        numeric = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是非负整数。") from error
    if numeric < 0 or numeric != float(value):
        raise ValueError(f"{name} 必须是非负整数。")
    return numeric


def plot_execution_rate(
    local_tasks: int,
    sv_tasks: int,
    rsu_tasks: int,
    uav_tasks: int,
    cloud_tasks: int,
    failed_tasks: int,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """绘制 Figure 2：最终实际 assignment 的五类执行节点比例。

    ``failed_tasks`` 只参与输入校验和结果语义记录，不绘制为第六根柱子；
    分母严格为五类 completed task 计数之和。
    """

    counts = {
        "local": _nonnegative_count(local_tasks, "local_tasks"),
        "service_vehicle": _nonnegative_count(sv_tasks, "sv_tasks"),
        "rsu": _nonnegative_count(rsu_tasks, "rsu_tasks"),
        "uav": _nonnegative_count(uav_tasks, "uav_tasks"),
        "cloud": _nonnegative_count(cloud_tasks, "cloud_tasks"),
    }
    failed = _nonnegative_count(failed_tasks, "failed_tasks")
    completed = sum(counts.values())
    if completed <= 0:
        raise ValueError("最终 evaluation 没有 completed task，无法计算执行率。")

    rates = np.asarray(
        [100.0 * counts[key] / completed for key in EXECUTION_COUNT_KEYS],
        dtype=np.float64,
    )
    if not np.isclose(float(np.sum(rates)), 100.0, rtol=0.0, atol=1e-10):
        raise RuntimeError("五类最终 execution rate 未归一到 100%。")

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    png_path = destination / "figure2_execution_rate.png"
    pdf_path = destination / "figure2_execution_rate.pdf"

    labels = ("Local", "Service Vehicle", "RSU", "UAV", "Cloud")
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.bar(labels, rates)
    axis.set_title("Final Execution Rate")
    axis.set_xlabel("Execution Node")
    axis.set_ylabel("Execution Rate (%)")
    axis.set_ylim(0.0, max(100.0, float(np.max(rates)) + 10.0))
    axis.grid(axis="y", alpha=0.25)
    axis.text(
        0.99,
        0.98,
        f"Completed={completed}\nFailed={failed}",
        transform=axis.transAxes,
        ha="right",
        va="top",
    )
    for index, rate in enumerate(rates):
        axis.text(index, float(rate) + 1.0, f"{rate:.1f}%", ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(png_path, dpi=180)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


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
        "reward": plot_reward_convergence(
            metrics_file,
            output_dir,
            moving_average_window=moving_average_window,
        ),
        "execution_rate": plot_execution_rate(
            execution_counts["local"],
            execution_counts["service_vehicle"],
            execution_counts["rsu"],
            execution_counts["uav"],
            execution_counts["cloud"],
            execution_counts["failed"],
            output_dir,
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    """构造只从正式 metrics 生成 Figure 1/2 的最小命令行接口。"""

    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="从正式 metrics 绘制论文 Figure 1/2")
    parser.add_argument(
        "--metrics",
        "--metrics-file",
        dest="metrics_file",
        type=Path,
        default=project_root / "outputs" / "metrics.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs")
    parser.add_argument(
        "--figure2-only",
        action="store_true",
        help="只从 phase=final_eval 生成 Figure 2。",
    )
    return parser


def main() -> None:
    """从 phase=train/final_eval 的正式 metrics 生成两张论文图。"""

    args = _build_parser().parse_args()
    if args.figure2_only:
        rows = _read_metrics(args.metrics_file)
        counts = _final_eval_execution_counts(rows)
        result = {
            "execution_rate": plot_execution_rate(
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
    for name, paths in result.items():
        label = "Figure 1" if name == "reward" else "Figure 2"
        print(f"{label}: {paths[0]}")


if __name__ == "__main__":
    main()
