"""记录正式 PPO run 的来源信息、清单和汇总指标。"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - 无 PyTorch 环境也可检查 reporting
    torch = None  # type: ignore[assignment]

from common import project_config
from .macro_targets import (
    build_macro_n2_target,
    build_macro_smdp_lambda_return_target,
)

SLOT_METRICS_FILENAME = "slot_metrics.jsonl"
MACRO_TRANSITION_TRACE_FILENAME = "macro_transition_trace.npz"

EPISODE_METRICS_FILENAME = "episode_metrics.jsonl"

TRAINING_EPOCH_RETURN_PLOT_FILENAME = "training_epoch_return.png"

SLOT_DETAIL_FIELDS = (
    "q",
    "cache_cost",
    "movement_cost",
    "cache_trigger_count",
    "cache_change_count",
    "uav_trigger_count",
    "uav_movement_count",
)

def _load_slot_contract(run_dir: Path) -> tuple[int, int]:
    """从正式 run manifest 读取 episode slot 范围。"""

    manifest_path = Path(run_dir) / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"缺少正式 run manifest：{manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read plotting protocol manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"plotting protocol manifest must be an object: {manifest_path}")
    raw_slots = manifest.get("episode_slots")
    raw_start = manifest.get("origin_start")
    raw_end = manifest.get("origin_end")
    if raw_slots is None or raw_start is None or raw_end is None:
        raise ValueError(f"正式 run manifest 缺少 episode slot 范围：{manifest_path}")
    try:
        episode_slots = int(raw_slots)
        origin_start = int(raw_start)
        origin_end = int(raw_end)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"plotting protocol manifest has invalid slot bounds: {manifest_path}") from exc
    if episode_slots <= 0 or origin_end - origin_start + 1 != episode_slots:
        raise ValueError(f"plotting protocol manifest has inconsistent slot bounds: {manifest_path}")
    return episode_slots, origin_start

def _resolve_slot_contract(
    path: Path,
    episode_slots: int | None,
    origin_start: int | None,
) -> tuple[int, int]:
    if episode_slots is None:
        return _load_slot_contract(path.parent)
    slots = int(episode_slots)
    start = 0 if origin_start is None else int(origin_start)
    if slots <= 0:
        raise ValueError("episode_slots must be positive")
    return slots, start

def _slot_metrics_finite_float(value: Any, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"slot_metrics {field} must be numeric") from exc
    if not math.isfinite(converted):
        raise ValueError(f"slot_metrics {field} must be finite")
    return converted

def _nonnegative_int(value: Any, field: str) -> int:
    converted = _slot_metrics_finite_float(value, field)
    integer = int(converted)
    if converted != integer or integer < 0:
        raise ValueError(f"slot_metrics {field} must be a non-negative integer")
    return integer

def _read_slot_metrics(
    path: Path,
    *,
    episode_slots: int | None = None,
    origin_start: int | None = None,
) -> dict[tuple[int, int], dict[str, Any]]:
    active_slots, active_origin_start = _resolve_slot_contract(
        Path(path), episode_slots, origin_start
    )
    records: dict[tuple[int, int], dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"slot_metrics contains a blank line at {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"slot_metrics contains invalid JSON at line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"slot_metrics line {line_number} is not an object")
            missing = {
                "epoch",
                "origin",
                "global_training_slot",
                "reward",
            } - set(record)
            if missing:
                raise ValueError(
                    f"slot_metrics line {line_number} missing fields: {sorted(missing)}"
                )
            epoch = int(record["epoch"])
            origin = int(record["origin"])
            global_slot = int(record["global_training_slot"])
            if epoch < 1:
                raise ValueError(f"slot_metrics epoch must be >=1 at line {line_number}")
            origin_offset = origin - active_origin_start
            if not 0 <= origin_offset < active_slots:
                raise ValueError(
                    f"slot_metrics origin must be in "
                    f"[{active_origin_start},{active_origin_start + active_slots - 1}] "
                    f"at line {line_number}"
                )
            expected_global_slot = (epoch - 1) * active_slots + origin_offset
            if global_slot != expected_global_slot:
                raise ValueError(
                    "slot_metrics global_training_slot mismatch at "
                    f"line {line_number}: {global_slot} != {expected_global_slot}"
                )
            reward = _slot_metrics_finite_float(record["reward"], "reward")
            present_detail_fields = [field for field in SLOT_DETAIL_FIELDS if field in record]
            if present_detail_fields and len(present_detail_fields) != len(SLOT_DETAIL_FIELDS):
                missing_detail_fields = sorted(set(SLOT_DETAIL_FIELDS) - set(present_detail_fields))
                raise ValueError(
                    f"slot_metrics line {line_number} missing detail fields: "
                    f"{missing_detail_fields}"
                )
            key = (epoch, origin)
            if key in records:
                raise ValueError(
                    f"slot_metrics duplicate (epoch, origin)={key}"
                )
            parsed = {
                "epoch": epoch,
                "origin": origin,
                "global_training_slot": global_slot,
                "reward": reward,
            }
            if present_detail_fields:
                parsed.update(
                    {
                        "q": _slot_metrics_finite_float(record["q"], "q"),
                        "cache_cost": _slot_metrics_finite_float(record["cache_cost"], "cache_cost"),
                        "movement_cost": _slot_metrics_finite_float(
                            record["movement_cost"], "movement_cost"
                        ),
                        "cache_trigger_count": _nonnegative_int(
                            record["cache_trigger_count"], "cache_trigger_count"
                        ),
                        "cache_change_count": _nonnegative_int(
                            record["cache_change_count"], "cache_change_count"
                        ),
                        "uav_trigger_count": _nonnegative_int(
                            record["uav_trigger_count"], "uav_trigger_count"
                        ),
                        "uav_movement_count": _nonnegative_int(
                            record["uav_movement_count"], "uav_movement_count"
                        ),
                    }
                )
                expected_reward = (
                    parsed["q"]
                    - parsed["cache_cost"]
                    - parsed["movement_cost"]
                )
                if not bool(
                    np.isclose(reward, expected_reward, rtol=1.0e-12, atol=1.0e-9)
                ):
                    raise ValueError(
                        "slot_metrics reward contract mismatch at "
                        f"(epoch, origin)={key}: reward={reward}, "
                        f"q-cache_cost-movement_cost={expected_reward}"
                    )
            records[key] = parsed
    return records

def append_training_slot_metrics(
    path: Path,
    *,
    epoch: int,
    rewards: Iterable[float],
    episode_return: float,
    episode_slots: int | None = None,
    origin_start: int | None = None,
    q_values: Iterable[float] | None = None,
    cache_costs: Iterable[float] | None = None,
    movement_costs: Iterable[float] | None = None,
    cache_trigger_counts: Iterable[int] | None = None,
    cache_change_counts: Iterable[int] | None = None,
    uav_trigger_counts: Iterable[int] | None = None,
    uav_movement_counts: Iterable[int] | None = None,
) -> int:
    """追加一个 epoch 的 transition reward；已有记录经过校验后可安全补全。"""

    path = Path(path)
    active_slots, active_origin_start = _resolve_slot_contract(
        path, episode_slots, origin_start
    )
    epoch_i = int(epoch)
    if epoch_i < 1:
        raise ValueError("slot_metrics epoch must be >=1")
    reward_array = np.asarray(list(rewards), dtype=np.float64).reshape(-1)
    if reward_array.shape != (active_slots,):
        raise ValueError(
            "slot_metrics requires one complete episode; "
            f"received {reward_array.size} rewards"
        )
    if not np.all(np.isfinite(reward_array)):
        raise ValueError("slot_metrics rewards must be finite")
    episode_return_f = _slot_metrics_finite_float(episode_return, "episode_return")
    if not bool(
        np.isclose(
            float(reward_array.sum()),
            episode_return_f,
            rtol=1.0e-12,
            atol=1.0e-9,
        )
    ):
        raise ValueError(
            "slot reward sum does not match episode_return: "
            f"{float(reward_array.sum())} != {episode_return_f}"
        )

    detail_inputs = (
        q_values,
        cache_costs,
        movement_costs,
        cache_trigger_counts,
        cache_change_counts,
        uav_trigger_counts,
        uav_movement_counts,
    )
    detailed = any(value is not None for value in detail_inputs)
    if detailed and not all(value is not None for value in detail_inputs):
        raise ValueError(
            "slot_metrics detailed fields must be supplied together"
        )
    detail_arrays: dict[str, np.ndarray] = {}
    if detailed:
        detail_arrays = {
            "q": np.asarray(list(q_values), dtype=np.float64).reshape(-1),  # type: ignore[arg-type]
            "cache_cost": np.asarray(list(cache_costs), dtype=np.float64).reshape(-1),  # type: ignore[arg-type]
            "movement_cost": np.asarray(list(movement_costs), dtype=np.float64).reshape(-1),  # type: ignore[arg-type]
            "cache_trigger_count": np.asarray(
                list(cache_trigger_counts), dtype=np.float64  # type: ignore[arg-type]
            ).reshape(-1),
            "cache_change_count": np.asarray(
                list(cache_change_counts), dtype=np.float64  # type: ignore[arg-type]
            ).reshape(-1),
            "uav_trigger_count": np.asarray(
                list(uav_trigger_counts), dtype=np.float64  # type: ignore[arg-type]
            ).reshape(-1),
            "uav_movement_count": np.asarray(
                list(uav_movement_counts), dtype=np.float64  # type: ignore[arg-type]
            ).reshape(-1),
        }
        for field, values in detail_arrays.items():
            if values.shape != (active_slots,):
                raise ValueError(
                    f"slot_metrics {field} requires {active_slots} values; "
                    f"received {values.size}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"slot_metrics {field} must be finite")
        for field in (
            "cache_trigger_count",
            "cache_change_count",
            "uav_trigger_count",
            "uav_movement_count",
        ):
            values = detail_arrays[field]
            if np.any(values < 0.0) or np.any(values != np.floor(values)):
                raise ValueError(f"slot_metrics {field} must be non-negative integers")
        expected_rewards = (
            detail_arrays["q"]
            - detail_arrays["cache_cost"]
            - detail_arrays["movement_cost"]
        )
        if not np.allclose(reward_array, expected_rewards, rtol=1.0e-12, atol=1.0e-9):
            mismatch = int(np.flatnonzero(~np.isclose(
                reward_array, expected_rewards, rtol=1.0e-12, atol=1.0e-9
            ))[0])
            raise ValueError(
                "slot reward contract mismatch at origin "
                f"{mismatch}: reward={reward_array[mismatch]}, "
                f"q-cache_cost-movement_cost={expected_rewards[mismatch]}"
            )

    records = _read_slot_metrics(
        path,
        episode_slots=active_slots,
        origin_start=active_origin_start,
    )
    epoch_counts: dict[int, int] = {}
    for existing_epoch, _ in records:
        epoch_counts[existing_epoch] = epoch_counts.get(existing_epoch, 0) + 1
    existing_epochs = sorted(epoch_counts)
    if existing_epochs != list(range(1, len(existing_epochs) + 1)):
        raise ValueError("slot_metrics epochs are not contiguous from epoch 1")
    for existing_epoch, count in epoch_counts.items():
        if existing_epoch > epoch_i:
            raise ValueError(
                f"slot_metrics contains a future epoch {existing_epoch} while writing {epoch_i}"
            )
        if existing_epoch < epoch_i and count != active_slots:
            raise ValueError(
                f"slot_metrics epoch {existing_epoch} is incomplete ({count}/"
                f"{active_slots} records)"
            )
    to_append: list[dict[str, Any]] = []
    for origin_offset, reward in enumerate(reward_array.tolist()):
        origin = active_origin_start + origin_offset
        key = (epoch_i, origin)
        existing = records.get(key)
        if existing is not None:
            if not bool(np.isclose(existing["reward"], reward, rtol=0.0, atol=1.0e-12)):
                raise ValueError(
                    "existing slot reward conflicts with the completed episode at "
                    f"(epoch, origin)={key}"
                )
            if detailed and all(field in existing for field in SLOT_DETAIL_FIELDS):
                for field in SLOT_DETAIL_FIELDS:
                    expected = (
                        int(detail_arrays[field][origin_offset])
                        if field.endswith("_count")
                        else float(detail_arrays[field][origin_offset])
                    )
                    if existing[field] != expected:
                        raise ValueError(
                            f"existing slot {field} conflicts at (epoch, origin)={key}"
                        )
            continue
        record = {
            "epoch": epoch_i,
            "origin": origin,
            "global_training_slot": (epoch_i - 1) * active_slots + origin_offset,
            "reward": float(reward),
        }
        if detailed:
            for field in SLOT_DETAIL_FIELDS:
                value = detail_arrays[field][origin_offset]
                record[field] = int(value) if field.endswith("_count") else float(value)
        to_append.append(record)
    for record in to_append:
        _append_jsonl(path, record)
    return len(to_append)

def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")

def _finite_episode_return(value: Any) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("episode_metrics episode_return must be numeric") from exc
    if not math.isfinite(converted):
        raise ValueError("episode_metrics episode_return must be finite")
    return converted

def _read_episode_returns(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取每个连续完成 epoch 的有限原始回报。"""

    records: dict[int, float] = {}
    if not path.exists():
        raise ValueError("episode_metrics.jsonl is empty or does not exist")
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(
                    f"episode_metrics contains a blank line at {line_number}"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"episode_metrics contains invalid JSON at line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"episode_metrics line {line_number} is not an object"
                )
            if record.get("record_type", "episode") != "episode":
                continue
            if "epoch" not in record:
                raise ValueError(
                    f"episode_metrics line {line_number} is missing epoch"
                )
            raw_epoch = record["epoch"]
            try:
                epoch = int(raw_epoch)
                epoch_as_float = float(raw_epoch)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"episode_metrics line {line_number} has invalid epoch"
                ) from exc
            if (
                isinstance(raw_epoch, bool)
                or not math.isfinite(epoch_as_float)
                or epoch_as_float != epoch
                or epoch < 1
            ):
                raise ValueError(
                    f"episode_metrics line {line_number} has invalid epoch: {raw_epoch!r}"
                )
            if epoch in records:
                raise ValueError(f"episode_metrics duplicate epoch: {epoch}")
            records[epoch] = _finite_episode_return(record.get("episode_return"))

    if not records:
        raise ValueError("episode_metrics.jsonl has no episode records")
    epochs = sorted(records)
    expected_epochs = list(range(1, len(epochs) + 1))
    if epochs != expected_epochs:
        raise ValueError(
            "episode_metrics epochs are not contiguous from 1: "
            f"expected {expected_epochs}, got {epochs}"
        )
    return (
        np.asarray(epochs, dtype=np.int64),
        np.asarray([records[epoch] for epoch in epochs], dtype=np.float64),
    )

def _full_window_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if int(window) <= 0:
        raise ValueError("moving-average window must be positive")
    moving = np.full(values.shape, np.nan, dtype=np.float64)
    if values.size >= int(window):
        for index in range(int(window) - 1, values.size):
            moving[index] = float(np.mean(values[index - int(window) + 1 : index + 1]))
    return moving

def update_training_epoch_return_plot(run_dir: Path) -> Path:
    """根据 epoch 回报生成当前正式 PPO 训练图。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    run_path = Path(run_dir)
    epochs, returns = _read_episode_returns(
        run_path / EPISODE_METRICS_FILENAME
    )
    ma20 = _full_window_moving_average(returns, 20)

    figure, axis = plt.subplots(figsize=(12, 6))
    try:
        axis.plot(
            epochs,
            returns,
            label="_nolegend_",
            color="#2563eb",
            linewidth=0.8,
            alpha=0.15,
        )
        valid = np.isfinite(ma20)
        if np.any(valid):
            axis.plot(
                epochs[valid],
                ma20[valid],
                label="MA20",
                color="#2563eb",
                linewidth=2.2,
                marker="o",
                markersize=4.5,
                markevery=10,
            )
        axis.set_xlabel("Training Epoch")
        axis.set_ylabel("Episode Return")
        axis.set_title("PPO Training Return (MA20)")
        axis.grid(True, axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.legend(frameon=False)
        figure.tight_layout()
        output_path = run_path / "plots" / TRAINING_EPOCH_RETURN_PLOT_FILENAME
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=320, facecolor="white")
    finally:
        plt.close(figure)
    return output_path

def _gradient_group_norm(parameters: Any) -> float:
    """返回一个诊断参数组在梯度裁剪前的 L2 范数。"""

    squared_norm = None
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        component = torch.sum(gradient.detach() ** 2)
        squared_norm = component if squared_norm is None else squared_norm + component
    if squared_norm is None:
        return 0.0
    return float(torch.sqrt(squared_norm).detach().cpu())

def _module_parameter_delta_norm(before_parameters: dict[str, Any], module: Any) -> float:
    """返回模块具名参数变化量的范数，仅用于诊断。"""

    delta_squared = 0.0
    current_names = set()
    for name, parameter in module.named_parameters():
        key = str(name)
        current_names.add(key)
        if key not in before_parameters:
            raise RuntimeError(f"diagnostic module parameter disappeared: {key}")
        delta_squared += float(
            torch.sum((parameter.detach() - before_parameters[key]) ** 2).cpu()
        )
    if current_names != set(before_parameters):
        raise RuntimeError("diagnostic module parameter names changed during PPO update")
    return float(np.sqrt(delta_squared))

def _matching_termination_metrics(
    summaries: Any,
) -> dict[str, Any]:
    """汇总 Matching 停止原因，不改变 Matching 决策。"""

    natural_stable_count = 0
    max_updates_hit_count = 0
    update_counts: list[int] = []
    for summary in summaries:
        if not summary or not isinstance(summary, dict):
            continue
        stop_reason = str(summary.get("stop_reason", ""))
        if stop_reason == "converged_by_no_improvement":
            natural_stable_count += 1
        elif stop_reason == "stopped_by_max_updates":
            max_updates_hit_count += 1
        if "updates" not in summary:
            continue
        update_count = int(summary["updates"])
        if update_count < 0:
            raise ValueError("Matching update count must be non-negative")
        update_counts.append(update_count)
    return _matching_metrics_from_counts(
        natural_stable_count,
        max_updates_hit_count,
        update_counts,
    )

def _matching_metrics_from_counts(
    natural_stable_count: int,
    max_updates_hit_count: int,
    update_counts: Any,
) -> dict[str, Any]:
    """从 Exact-Q worker 汇总数据中构建 Matching 指标。"""

    counts_list = [int(value) for value in update_counts]
    if any(value < 0 for value in counts_list):
        raise ValueError("Matching update count must be non-negative")
    if not counts_list:
        return {
            "natural_stable_count": int(natural_stable_count),
            "max_updates_hit_count": int(max_updates_hit_count),
            "max_updates_hit_rate": 0.0,
            "update_count_mean": 0.0,
            "update_count_p95": 0.0,
            "update_count_max": 0,
        }
    counts = np.asarray(counts_list, dtype=np.float64)
    return {
        "natural_stable_count": int(natural_stable_count),
        "max_updates_hit_count": int(max_updates_hit_count),
        "max_updates_hit_rate": float(max_updates_hit_count / len(counts_list)),
        "update_count_mean": float(np.mean(counts)),
        "update_count_p95": float(np.percentile(counts, 95.0)),
        "update_count_max": int(np.max(counts)),
    }

def _kkt_metrics_from_diagnostics(records: Any) -> dict[str, Any]:
    """从 Matching 汇总数据中聚合只读 KKT 求解诊断。"""

    normalized: list[dict[str, Any]] = []
    for record in records or ():
        if not isinstance(record, dict):
            continue
        normalized.append(record)

    solve_count = len(normalized)
    converged_count = sum(
        bool(record.get("converged_by_tolerance", False))
        for record in normalized
    )
    max_iter_hit_count = sum(
        bool(record.get("hit_max_bisection_iterations", False))
        for record in normalized
    )
    finite_count = sum(bool(record.get("finite", False)) for record in normalized)

    def finite_values(field: str) -> np.ndarray:
        values = []
        for record in normalized:
            try:
                value = float(record.get(field, math.nan))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        return np.asarray(values, dtype=np.float64)

    iterations = finite_values("iterations")
    resource_residuals = finite_values("resource_sum_residual")
    stationarity_residuals = finite_values("stationarity_residual")
    complementary_residuals = finite_values("complementary_slackness_residual")

    def mean(values: np.ndarray) -> float:
        return 0.0 if values.size == 0 else float(np.mean(values))

    def maximum(values: np.ndarray) -> float:
        return math.nan if values.size == 0 else float(np.max(values))

    return {
        "solve_count": int(solve_count),
        "converged_count": int(converged_count),
        "converged_rate": (
            float(converged_count / solve_count) if solve_count else 0.0
        ),
        "max_iter_hit_count": int(max_iter_hit_count),
        "max_iter_hit_rate": (
            float(max_iter_hit_count / solve_count) if solve_count else 0.0
        ),
        "finite_count": int(finite_count),
        "finite_rate": float(finite_count / solve_count) if solve_count else 0.0,
        "iteration_mean": mean(iterations),
        "iteration_p95": (
            0.0 if iterations.size == 0 else float(np.percentile(iterations, 95.0))
        ),
        "iteration_max": int(np.max(iterations)) if iterations.size else 0,
        "resource_residual_mean": mean(resource_residuals),
        "resource_residual_max": maximum(resource_residuals),
        "stationarity_residual_mean": mean(stationarity_residuals),
        "stationarity_residual_max": maximum(stationarity_residuals),
        "complementary_slackness_residual_mean": mean(complementary_residuals),
        "complementary_slackness_residual_max": maximum(complementary_residuals),
    }

RUN_MANIFEST_FILENAME = "run_manifest.json"
RUN_SUMMARY_FILENAME = "run_summary.json"
CHECKPOINTS_DIRECTORY = "checkpoints"
PLOTS_DIRECTORY = "plots"
FRESH_RUN_ARTIFACTS = (
    RUN_MANIFEST_FILENAME,
    RUN_SUMMARY_FILENAME,
    "episode_metrics.jsonl",
    "update_metrics.jsonl",
    "critic_diagnostics.jsonl",
    "slot_metrics.jsonl",
    f"{CHECKPOINTS_DIRECTORY}/latest_checkpoint.pt",
    f"{CHECKPOINTS_DIRECTORY}/best_train_return_checkpoint.pt",
    f"plots/{TRAINING_EPOCH_RETURN_PLOT_FILENAME}",
    MACRO_TRANSITION_TRACE_FILENAME,
)


def checkpoint_path_for_write(run_dir: Path, filename: str) -> Path:
    """返回正式 checkpoint 的保存路径。"""

    return Path(run_dir) / CHECKPOINTS_DIRECTORY / filename


def checkpoint_path_for_resume(run_dir: Path, filename: str) -> Path:
    """返回正式 checkpoint 的恢复路径。"""

    return Path(run_dir) / CHECKPOINTS_DIRECTORY / filename


def resolve_checkpoint_reference(checkpoint: Path) -> Path:
    """解析 checkpoint 文件、正式 checkpoint 目录或 run 目录。"""

    path = Path(checkpoint).expanduser()
    if path.is_dir():
        if path.name == CHECKPOINTS_DIRECTORY:
            return checkpoint_path_for_resume(path.parent, "latest_checkpoint.pt")
        return checkpoint_path_for_resume(path, "latest_checkpoint.pt")
    if path.parent.name == CHECKPOINTS_DIRECTORY:
        return path
    return path.parent / CHECKPOINTS_DIRECTORY / path.name


def ensure_run_artifact_directories(run_dir: Path) -> None:
    """创建正式 checkpoint 和 plot 目录。"""

    run_path = Path(run_dir)
    (run_path / CHECKPOINTS_DIRECTORY).mkdir(parents=True, exist_ok=True)
    (run_path / PLOTS_DIRECTORY).mkdir(parents=True, exist_ok=True)


def ensure_fresh_run_dir_is_clean(run_dir: Path) -> None:
    """如果新 run 会与现有实验混合，则拒绝启动。"""

    run_path = Path(run_dir)
    if not run_path.exists():
        return
    if not run_path.is_dir():
        raise RuntimeError(
            "fresh PPO run-dir already exists but is not a directory; "
            "use a new --run-dir or --resume"
        )
    existing_artifacts = [
        relative_path
        for relative_path in FRESH_RUN_ARTIFACTS
        if (run_path / relative_path).exists()
    ]
    for checkpoint_directory in (run_path / CHECKPOINTS_DIRECTORY, run_path):
        if checkpoint_directory.is_dir():
            existing_artifacts.extend(
                str(path.relative_to(run_path))
                for path in checkpoint_directory.glob("*.pt")
                if str(path.relative_to(run_path)) not in existing_artifacts
            )
    if existing_artifacts:
        raise RuntimeError(
            "fresh PPO run-dir already contains experiment artifacts; "
            "use a new --run-dir or --resume"
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream, parse_constant=_parse_nonfinite)
    if not isinstance(payload, dict):
        raise ValueError(f"reporting JSON root must be an object: {path}")
    return payload


def _parse_nonfinite(value: str) -> float:
    return float(value)


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], bool]:
    records: list[dict[str, Any]] = []
    nan_inf_detected = False
    if not path.exists():
        return records, nan_inf_detected
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"reporting JSONL contains a blank line: {path}:{line_number}")
            try:
                record = json.loads(line, parse_constant=_parse_nonfinite)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"reporting JSONL contains invalid JSON: {path}:{line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"reporting JSONL line must be an object: {path}:{line_number}"
                )
            nan_inf_detected = nan_inf_detected or _contains_nonfinite(record)
            records.append(record)
    return records, nan_inf_detected


def _finite_float(value: Any, field: str) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"reporting field {field} must be numeric or null") from exc
    return converted if math.isfinite(converted) else None


def _git_identity(project_root: Path) -> tuple[str | None, bool | None]:
    try:
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if sha_result.returncode != 0 or dirty_result.returncode != 0:
            raise RuntimeError(
                f"git command failed: sha={sha_result.returncode}, "
                f"dirty={dirty_result.returncode}"
            )
        sha = sha_result.stdout.strip()
        if not sha:
            raise RuntimeError("git rev-parse HEAD returned an empty SHA")
        return sha, bool(dirty_result.stdout.strip())
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[REPORTING] Warning: unable to read git identity: {exc}", flush=True)
        return None, None


def _optimizer_learning_rate(optimizer: Any, label: str) -> float:
    if optimizer is None:
        raise ValueError(f"{label} optimizer is not initialized")
    groups = list(getattr(optimizer, "param_groups", ()))
    if not groups:
        raise ValueError(f"{label} optimizer has no parameter groups")
    rates = []
    for group in groups:
        rate = float(group["lr"])
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError(f"{label} optimizer learning rate is invalid: {rate}")
        rates.append(rate)
    if any(rate != rates[0] for rate in rates[1:]):
        raise ValueError(
            f"{label} optimizer has multiple learning rates; manifest field is scalar"
        )
    return rates[0]


def _actual_optimizers(trainer: Any) -> tuple[Any, Any]:
    actor_optimizer = getattr(trainer, "cache_trigger_optimizer", None)
    critic_optimizer = getattr(trainer, "critic_optimizer", None)
    return actor_optimizer, critic_optimizer


def build_run_manifest(
    *,
    config_path: str | Path,
    config_raw_sha256: str,
    config_semantic_sha256: str,
    seed: int,
    trainer: Any,
    context: Any,
    target_epochs: int,
    rollout_steps: int,
) -> dict[str, Any]:
    """从当前 trainer、策略、优化器和配置生成正式 run manifest。"""

    hyper = trainer.hyperparameters
    actor_optimizer, critic_optimizer = _actual_optimizers(trainer)
    protocol = project_config.get_ppo_data_protocol(context.cfg)
    train_origin_start = int(context.train_origin_start)
    train_origin_end = int(context.train_origin_end)
    policy_schema = str(getattr(trainer.policy, "architecture_id", context.policy_schema))
    if policy_schema != "adaptive_joint_cache_uav_ppo":
        raise ValueError("run manifest 仅支持正式 proposed_joint 策略。")
    git_sha, git_dirty = _git_identity(Path(context.root))
    resolved_config_path = Path(
        context.cfg.get("_meta", {}).get("config_path", config_path)
    ).resolve()
    learning_rate_metadata = trainer.learning_rate_metadata()
    training_provenance = context.cfg.get("training_provenance", {})
    cache_credit_manifest_fields: dict[str, Any] = {}
    if hasattr(trainer, "cache_credit_metadata"):
        cache_credit_manifest_fields = dict(trainer.cache_credit_metadata())
    return {
        "experiment_type": "main_training",
        "architecture_version": int(getattr(trainer.policy, "architecture_version", 1)),
        "checkpoint_version": int(trainer._expected_checkpoint_version()),
        "initial_actor_sha256": training_provenance.get("ACTOR_INITIAL_STATE_SHA256"),
        "initial_critic_sha256": training_provenance.get("CRITIC_INITIAL_STATE_SHA256"),
        "initial_uav_candidate_encoder_sha256": training_provenance.get(
            "INITIAL_UAV_CANDIDATE_ENCODER_SHA256"
        ),
        "uav_candidate_encoder_initialization": (
            "deterministic deepcopy of the seeded base UAV encoder before optimization"
        ),
        "checkpoint_interval_epochs": 10,
        "best_train_return_checkpoint_role": "TRAINING_DIAGNOSTIC_ONLY_NOT_FINAL_SELECTION",
        "final_checkpoint_selection_metric": "validation_mean_episode_return",
        "git_commit_sha": git_sha,
        "git_dirty": git_dirty,
        "config_path": str(resolved_config_path),
        "config_sha256": str(config_raw_sha256),
        "config_raw_sha256": str(config_raw_sha256),
        "config_semantic_sha256": str(config_semantic_sha256),
        "seed": int(seed),
        "device": str(trainer.device),
        "policy_schema": policy_schema,
        **(
            trainer.critic_bootstrap_teacher_metadata()
            if hasattr(trainer, "critic_bootstrap_teacher_metadata")
            else {}
        ),
        "configuration_control_mode": str(context.configuration_control_mode),
        "advantage_normalization_mode": str(hyper.advantage_normalization_mode),
        "operation_advantage_normalization_mode": str(
            hyper.operation_advantage_normalization_mode
        ),
        "operation_advantage_reference_scale": float(
            hyper.operation_advantage_reference_scale
        ),
        "uav_operation_advantage_source": "FULL_INFORMATION_EXACT_Q_TEACHER",
        "UAV_OPERATION_TEACHER": "FULL_INFORMATION_EXACT_Q_TEACHER",
        "uav_candidate_advantage_source": "EXACT_Q_LISTWISE",
        "UAV_CANDIDATE_TEACHER": "EXACT_Q_LISTWISE",
        "candidate_advantage_normalization": "EXACT_Q_LISTWISE",
        "critic_value_scale_mode": str(hyper.critic_value_scale_mode),
        "critic_value_scale_epsilon": float(hyper.critic_value_scale_epsilon),
        "critic_value_scale_warmup_samples": int(hyper.critic_value_scale_warmup_samples),
        "observation_dim": int(context.observation_spec.observation_dim),
        "action_dim": int(context.action_spec.action_dim),
        "data_protocol": str(protocol["protocol"]),
        "episode_slots": int(context.episode_slots),
        "origin_start": train_origin_start,
        "origin_end": train_origin_end,
        "terminal_origin": int(context.terminal_origin),
        "train_origin_start": train_origin_start,
        "train_origin_end": train_origin_end,
        "target_epochs": int(target_epochs),
        "rollout_steps": int(rollout_steps),
        "gamma": float(hyper.gamma),
        "gae_lambda": float(hyper.gae_lambda),
        "clip_epsilon": float(hyper.clip_epsilon),
        "update_epochs": int(hyper.update_epochs),
        "target_kl": hyper.target_kl,
        "cache_kl_control": str(
            getattr(trainer, "cache_kl_control", "minibatch_early_stop")
        ),
        "num_minibatches": int(hyper.num_minibatches),
        "minibatch_size": int(context.minibatch_size),
        "actor_learning_rate": _optimizer_learning_rate(
            actor_optimizer, "actor"
        ),
        "critic_learning_rate": _optimizer_learning_rate(
            critic_optimizer, "critic"
        ),
        "max_grad_norm": float(hyper.max_grad_norm),
        "entropy_coefficient": float(hyper.entropy_coefficient),
        **learning_rate_metadata,
        **cache_credit_manifest_fields,
    }


def _manifest_protocol_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    """校验并规范化正式完整 episode manifest 的协议字段。"""

    protocol = str(manifest["data_protocol"])
    if protocol != project_config.PPO_DATA_PROTOCOL_FULL_EPISODE_RANDOMIZED:
        raise ValueError(f"run manifest 使用了不支持的数据协议：{protocol}")
    origin_start = int(manifest["origin_start"])
    origin_end = int(manifest["origin_end"])
    episode_slots = int(manifest["episode_slots"])
    if episode_slots <= 0 or origin_end - origin_start + 1 != episode_slots:
        raise ValueError("run manifest 中的 episode slot 范围不一致。")
    terminal_origin = int(manifest["terminal_origin"])
    if terminal_origin != origin_end:
        raise ValueError("run manifest 的 terminal_origin 必须等于 origin_end。")
    rollout_steps = int(manifest["rollout_steps"])
    num_minibatches = int(manifest["num_minibatches"])
    minibatch_size = int(manifest["minibatch_size"])
    if rollout_steps != episode_slots:
        raise ValueError("完整 episode 的 rollout_steps 必须等于 episode_slots。")
    if num_minibatches <= 0 or episode_slots % num_minibatches != 0:
        raise ValueError("完整 episode 的 num_minibatches 必须整除 episode_slots。")
    if minibatch_size != episode_slots // num_minibatches:
        raise ValueError("run manifest 的 minibatch_size 与 episode 范围不一致。")
    return {
        "data_protocol": protocol,
        "episode_slots": episode_slots,
        "origin_start": origin_start,
        "origin_end": origin_end,
        "terminal_origin": terminal_origin,
        "rollout_steps": rollout_steps,
        "num_minibatches": num_minibatches,
        "minibatch_size": minibatch_size,
    }


def validate_run_manifest_for_resume(
    existing: dict[str, Any], current: dict[str, Any]
) -> None:
    """拒绝恢复身份字段或完整 episode 协议不一致的 run。"""

    required_match_fields = (
        "policy_schema",
        "advantage_normalization_mode",
        "critic_value_scale_mode",
        "observation_dim",
        "rollout_steps",
        "experiment_id",
        "control_mode",
        "configured_actor_learning_rate",
        "configured_critic_learning_rate",
        "effective_actor_learning_rate",
        "effective_critic_learning_rate",
        "actor_learning_rate_overridden",
        "critic_learning_rate_overridden",
    )
    mismatches = {
        field: (existing.get(field), current.get(field))
        for field in required_match_fields
        if existing.get(field) != current.get(field)
    }
    if mismatches:
        details = ", ".join(
            f"{field}: existing={old!r}, current={new!r}"
            for field, (old, new) in mismatches.items()
        )
        raise ValueError(f"PPO resume run_manifest mismatch: {details}")

    lifecycle_identity_fields = (
        "checkpoint_version",
        "critic_bootstrap_teacher_warmup_required",
        "critic_warmup_seed_namespace",
    )
    lifecycle_mismatches = {
        field: (existing.get(field), current.get(field))
        for field in lifecycle_identity_fields
        if existing.get(field) != current.get(field)
    }
    if lifecycle_mismatches:
        details = ", ".join(
            f"{field}: existing={old!r}, current={new!r}"
            for field, (old, new) in lifecycle_mismatches.items()
        )
        raise ValueError(
            f"PPO resume critic warmup manifest mismatch: {details}"
        )

    existing_protocol = _manifest_protocol_identity(existing)
    current_protocol = _manifest_protocol_identity(current)
    protocol_mismatches = {
        field: (existing_protocol[field], current_protocol[field])
        for field in (
            "data_protocol",
            "episode_slots",
            "origin_start",
            "origin_end",
            "terminal_origin",
            "rollout_steps",
            "num_minibatches",
            "minibatch_size",
        )
        if existing_protocol[field] != current_protocol[field]
    }
    if protocol_mismatches:
        details = ", ".join(
            f"{field}: existing={old!r}, current={new!r}"
            for field, (old, new) in protocol_mismatches.items()
        )
        raise ValueError(f"PPO resume protocol mismatch: {details}")

    existing_git_sha = existing.get("git_commit_sha")
    current_git_sha = current.get("git_commit_sha")
    if existing_git_sha is not None:
        existing_git_sha = str(existing_git_sha).strip() or None
    if current_git_sha is not None:
        current_git_sha = str(current_git_sha).strip() or None
    if (
        existing_git_sha is not None
        and current_git_sha is not None
        and existing_git_sha != current_git_sha
    ):
        raise ValueError(
            "PPO resume git_commit_sha mismatch; refusing cross-code-version resume"
        )
    if existing_git_sha is None or current_git_sha is None:
        print(
            "[REPORTING] Warning: Git provenance unavailable for resume; "
            "allowing resume without cross-code-version verification.",
            flush=True,
        )


def write_run_manifest(
    path: Path,
    *,
    config_path: str | Path,
    config_raw_sha256: str,
    config_semantic_sha256: str,
    seed: int,
    trainer: Any,
    context: Any,
    target_epochs: int,
    rollout_steps: int,
    resume: bool,
) -> dict[str, Any]:
    """创建正式 manifest，或校验并更新正在恢复的 run。"""

    current = build_run_manifest(
        config_path=config_path,
        config_raw_sha256=config_raw_sha256,
        config_semantic_sha256=config_semantic_sha256,
        seed=seed,
        trainer=trainer,
        context=context,
        target_epochs=target_epochs,
        rollout_steps=rollout_steps,
    )
    if resume and path.exists():
        existing = _read_json(path)
        validate_run_manifest_for_resume(existing, current)
        initial_target_epochs = int(
            existing.get("initial_target_epochs", existing.get("target_epochs", target_epochs))
        )
        merged = dict(existing)
        merged["initial_target_epochs"] = initial_target_epochs
        merged["current_target_epochs"] = int(target_epochs)
        merged["target_epochs"] = int(target_epochs)
        _write_json(path, merged)
        return merged

    current["initial_target_epochs"] = int(target_epochs)
    current["current_target_epochs"] = int(target_epochs)
    _write_json(path, current)
    return current


def update_run_manifest_teacher_state(path: Path, trainer: Any) -> dict[str, Any]:
    """持久化仅由训练数据校准的 Cache teacher 与筛选状态。"""

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"run manifest is missing: {manifest_path}")
    payload = _read_json(manifest_path)
    if hasattr(trainer, "cache_credit_metadata"):
        payload.update(trainer.cache_credit_metadata())
    if hasattr(trainer, "critic_bootstrap_teacher_metadata"):
        payload.update(trainer.critic_bootstrap_teacher_metadata())
    payload.update(
        {
            "cache_teacher_scale_calibration": dict(
                getattr(trainer, "cache_teacher_scale_calibration", {}) or {}
            ),
            "cache_surrogate_training_count": int(
                getattr(trainer, "cache_surrogate_training_count", 0)
            ),
            "cache_surrogate_screening_ready": bool(
                getattr(trainer, "cache_surrogate_screening_ready", False)
            ),
            "cache_surrogate_screening_active": bool(
                getattr(trainer, "cache_surrogate_screening_active", False)
            ),
            "cache_surrogate_selected_k": getattr(
                trainer, "cache_surrogate_selected_k", None
            ),
            "cache_surrogate_audit_statistics": getattr(
                trainer, "cache_surrogate_audit_statistics", None
            ),
            "cache_surrogate_fallback_count": int(
                getattr(trainer, "cache_surrogate_fallback_count", 0)
            ),
            "cache_surrogate_fallback_epochs": list(
                getattr(trainer, "cache_surrogate_fallback_epochs", [])
            ),
            "cache_surrogate_recovery_count": int(
                getattr(trainer, "cache_surrogate_recovery_count", 0)
            ),
        }
    )
    _write_json(manifest_path, payload)
    return payload


def _episode_records(path: Path) -> tuple[list[dict[str, Any]], bool]:
    raw_records, nan_inf_detected = _read_jsonl(path)
    by_epoch: dict[int, dict[str, Any]] = {}
    for record in raw_records:
        if record.get("record_type", "episode") != "episode":
            continue
        try:
            epoch = int(record["epoch"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"episode_metrics record has invalid epoch: {record}") from exc
        if epoch < 1:
            raise ValueError(f"episode_metrics epoch must be >=1: {epoch}")
        if epoch in by_epoch:
            raise ValueError(f"episode_metrics duplicate epoch: {epoch}")
        by_epoch[epoch] = record
    ordered = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    return ordered, nan_inf_detected


def _metric_values(
    records: list[dict[str, Any]], field: str, *, allow_null: bool = True
) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _finite_float(record.get(field), field)
        if value is None:
            if not allow_null:
                raise ValueError(f"episode_metrics field {field} is missing or null")
            continue
        values.append(value)
    return values


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _rolling_mean(
    returns: list[float], epochs: list[int], window: int
) -> tuple[float | None, float | None]:
    if not returns:
        return None, None
    actual_window = min(int(window), len(returns))
    return float(sum(returns[-actual_window:]) / actual_window), None


def _rolling_max(
    returns: list[float], epochs: list[int], window: int
) -> tuple[float | None, int | None]:
    if len(returns) < int(window):
        return None, None
    candidates = [
        (
            float(sum(returns[index - window + 1 : index + 1]) / window),
            int(epochs[index]),
        )
        for index in range(window - 1, len(returns))
    ]
    best = max(candidates, key=lambda item: item[0])
    return best


def update_run_summary(
    path: Path,
    *,
    episode_metrics_path: Path,
    slot_metrics_path: Path,
    completed_epochs: int,
    target_epochs: int,
    status: str,
    protocol_metadata: dict[str, Any],
    lifecycle_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """根据已写入的指标生成当前 run 汇总。"""

    if status not in {"running", "completed"}:
        raise ValueError(f"invalid run summary status: {status}")
    episodes, episode_nan_inf = _episode_records(episode_metrics_path)
    episode_slots = int(protocol_metadata["episode_slots"])
    origin_start = int(protocol_metadata["origin_start"])
    origin_end = int(protocol_metadata["origin_end"])
    terminal_origin = int(protocol_metadata["terminal_origin"])
    if origin_end - origin_start + 1 != episode_slots:
        raise ValueError("run summary protocol has inconsistent episode slot bounds")
    if terminal_origin != origin_end:
        raise ValueError("run summary terminal_origin must equal origin_end")
    slot_records = _read_slot_metrics(
        slot_metrics_path,
        episode_slots=episode_slots,
        origin_start=origin_start,
    )
    epochs = [int(record["epoch"]) for record in episodes]
    returns = [
        value
        for value in (_finite_float(record.get("episode_return"), "episode_return") for record in episodes)
        if value is not None
    ]
    if len(returns) != len(episodes):
        episode_nan_inf = True
    final_window_count = min(20, len(episodes))
    final_episodes = episodes[-final_window_count:] if final_window_count else []
    final_epoch = epochs[-1] if epochs else None
    final_return = returns[-1] if returns else None
    best_epoch = 0
    best_return: float | None = None
    if returns:
        best_index = max(range(len(returns)), key=lambda index: returns[index])
        best_epoch = epochs[best_index]
        best_return = returns[best_index]

    ma20, _ = _rolling_mean(returns, epochs, 20)
    ma50, _ = _rolling_mean(returns, epochs, 50)
    ma20_max, ma20_max_epoch = _rolling_max(returns, epochs, 20)
    ma50_max, ma50_max_epoch = _rolling_max(returns, epochs, 50)

    critic_ev = _metric_values(final_episodes, "critic_explained_variance")
    all_critic_ev = _metric_values(episodes, "critic_explained_variance")
    max_critic_ev = None
    max_critic_ev_epoch = None
    if all_critic_ev:
        max_index = max(
            (index for index, record in enumerate(episodes)
             if _finite_float(record.get("critic_explained_variance"), "critic_explained_variance") is not None),
            key=lambda index: float(episodes[index]["critic_explained_variance"]),
        )
        max_critic_ev = float(episodes[max_index]["critic_explained_variance"])
        max_critic_ev_epoch = int(episodes[max_index]["epoch"])

    final_epoch_set = {int(record["epoch"]) for record in final_episodes}
    final_slot_records = [
        record
        for (epoch, _), record in slot_records.items()
        if epoch in final_epoch_set
    ]

    def region_mean(start: int, end: int) -> float | None:
        values = [
            float(record["reward"])
            for record in final_slot_records
            if start <= int(record["origin"]) <= end
        ]
        return _mean(values)

    base_region_size = episode_slots // 4
    region_lengths = [base_region_size] * 3 + [
        episode_slots - (base_region_size * 3)
    ]
    region_values: dict[str, float | None] = {}
    region_start = origin_start
    for region_length in region_lengths:
        region_end = region_start + region_length - 1
        if region_length <= 0:
            value = None
        else:
            value = region_mean(region_start, region_end)
        region_values[
            f"final20_reward_origin_{region_start}_{region_end}"
        ] = value
        region_start = region_end + 1

    summary = {
        "status": status,
        "completed_epochs": int(completed_epochs),
        "target_epochs": int(target_epochs),
        "data_protocol": str(protocol_metadata["data_protocol"]),
        "episode_slots": episode_slots,
        "origin_start": origin_start,
        "origin_end": origin_end,
        "terminal_origin": terminal_origin,
        "rollout_steps": protocol_metadata.get("rollout_steps"),
        "num_minibatches": protocol_metadata.get("num_minibatches"),
        "minibatch_size": protocol_metadata.get("minibatch_size"),
        "best_epoch": int(best_epoch),
        "best_return": best_return,
        "final_epoch": final_epoch,
        "final_return": final_return,
        "final20_window_count": int(final_window_count),
        "final20_mean_return": _mean(returns[-final_window_count:]) if final_window_count else None,
        "ma20": ma20,
        "ma20_max": ma20_max,
        "ma20_max_epoch": ma20_max_epoch,
        "ma50": ma50,
        "ma50_max": ma50_max,
        "ma50_max_epoch": ma50_max_epoch,
        "final20_mean_critic_ev": _mean(critic_ev),
        "max_critic_ev": max_critic_ev,
        "max_critic_ev_epoch": max_critic_ev_epoch,
        "final20_cache_change_rate": _mean(
            _metric_values(final_episodes, "cache_actual_change_rate")
        ),
        "final20_uav_movement_rate": _mean(
            _metric_values(final_episodes, "uav_actual_movement_rate")
        ),
        "nan_inf_detected": bool(episode_nan_inf),
    }
    summary.update(region_values)
    if lifecycle_metadata is not None:
        summary.update(dict(lifecycle_metadata))
    _write_json(path, summary)
    return summary

def _cache_operation_epoch_metrics(
    transitions: Any,
    initial_cache: np.ndarray,
    cache_nodes: dict[str, Any],
    service_sizes_mb: np.ndarray,
) -> dict[str, Any]:
    """汇总 KEEP/UPDATE 行为，不影响 rollout 或 PPO 更新。"""

    operation_names = ("KEEP", "UPDATE")
    roles = ("SV", "RSU", "UAV")
    node_roles = [str(value).upper() for value in cache_nodes["node_roles"]]
    node_count = len(node_roles)
    capacities = np.asarray(cache_nodes["capacities_mb"], dtype=np.float64).reshape(-1)
    sizes = np.asarray(service_sizes_mb, dtype=np.float64).reshape(-1)
    operation_counts = {
        role: {operation: 0 for operation in operation_names} for role in roles
    }
    role_samples = {role: 0 for role in roles}
    previous_cache = np.asarray(initial_cache, dtype=np.int8).copy()
    rsu_service_count: list[float] = []
    rsu_capacity_utilization: list[float] = []
    rsu_nonempty: list[bool] = []
    rsu_requested_updates = 0
    rsu_actual_changes = 0
    rsu_samples = 0

    for transition in transitions:
        next_cache = np.asarray(transition.next_cache_state, dtype=np.int8)
        if next_cache.shape != previous_cache.shape:
            raise ValueError("cache behavior diagnostics encountered a state shape change")
        actions = {
            int(item["head_index"]): item
            for item in transition.cache_conditional_actions
            if "head_index" in item
        }
        for node_index, role in enumerate(node_roles):
            role_samples[role] = role_samples.get(role, 0) + 1
            action = actions.get(node_index)
            if action is not None and role in operation_counts:
                conditional_action = str(action.get("conditional_action", "")).upper()
                raw_operation = str(action.get("operation_type", "")).upper()
                if conditional_action == "KEEP" or raw_operation == "NO_CHANGE":
                    operation = "KEEP"
                elif conditional_action == "TARGET_CONFIGURATION" or raw_operation in {
                    "ADD",
                    "DELETE",
                    "SWAP",
                }:
                    operation = "UPDATE"
                else:
                    operation = ""
                if operation in operation_counts[role]:
                    operation_counts[role][operation] += 1
            if role != "RSU":
                continue
            row = next_cache[node_index]
            used_mb = float(np.dot(row.astype(np.float64), sizes))
            capacity = float(capacities[node_index])
            if not math.isfinite(used_mb) or not math.isfinite(capacity) or capacity <= 0.0:
                raise FloatingPointError("RSU cache behavior diagnostic is non-finite")
            rsu_service_count.append(float(row.sum()))
            rsu_capacity_utilization.append(used_mb / capacity)
            rsu_nonempty.append(bool(row.sum() > 0))
            rsu_requested_updates += int(
                np.asarray(transition.action, dtype=np.int8)[node_index] == 1
            )
            rsu_actual_changes += int(
                np.any(row != previous_cache[node_index])
            )
            rsu_samples += 1
        previous_cache = next_cache.copy()

    def _rate(count: int, denominator: int) -> float:
        return 0.0 if denominator <= 0 else float(count) / float(denominator)

    operation_rates = {
        role: {
            operation: _rate(count, role_samples[role])
            for operation, count in counts.items()
        }
        for role, counts in operation_counts.items()
    }
    return {
        "cache_operation_counts": operation_counts,
        "cache_operation_rates": operation_rates,
        "RSU_mean_cached_service_count": (
            0.0 if not rsu_service_count else float(np.mean(rsu_service_count))
        ),
        "RSU_nonempty_fraction": (
            0.0 if not rsu_nonempty else float(np.mean(rsu_nonempty))
        ),
        "RSU_cache_capacity_utilization": (
            0.0
            if not rsu_capacity_utilization
            else float(np.mean(rsu_capacity_utilization))
        ),
        "RSU_requested_update_rate": _rate(rsu_requested_updates, rsu_samples),
        "RSU_actual_change_rate": _rate(rsu_actual_changes, rsu_samples),
        "RSU_operation_sample_count": int(rsu_samples),
    }

def _uav_episode_diagnostics(
    *,
    available: bool,
    move_probabilities: list[float],
    stay_probabilities: list[float],
    logit_margins: list[float],
    move_feasible: list[bool],
    move_selected: list[bool],
    actual_movement: list[bool],
    move_advantages: list[float],
    stay_advantages: list[float],
    global_advantages: list[float],
    global_return_targets: list[float],
    rewards: list[float],
    q_values: list[float],
    movement_costs: list[float],
    delays: list[float],
    energies: list[float],
    old_log_probs: list[float],
    new_log_probs: list[float],
    ratios: list[float],
    clipped: list[bool],
    feasible_action_counts: list[int],
    move_candidate_counts: list[int],
) -> dict[str, Any]:
    """不再调用策略，汇总已采样的 UAV 诊断。按经核验的 actual_movement 标记对 MOVE/STAY 条件字段分组，并单独保留所选操作标签，供后续审计动作与物理结果是否一致；不会把动作索引当作移动结果。"""

    empty = {
        "uav_move_probability_mean": None,
        "uav_move_probability_median": None,
        "uav_move_probability_p10": None,
        "uav_move_probability_p90": None,
        "uav_move_probability_gt_0_5_rate": None,
        "uav_stay_probability_mean": None,
        "uav_head_entropy": None,
        "uav_binary_entropy_mean": None,
        "uav_stay_move_logit_margin_mean": None,
        "uav_stay_move_logit_margin_median": None,
        "uav_move_feasible_count": None,
        "uav_move_feasible_rate": None,
        "uav_move_selected_count": None,
        "uav_move_selected_rate": None,
        "uav_move_given_selected": None,
        "uav_action_advantage_logging_available": False,
        "mean_advantage_for_uav_move": None,
        "median_advantage_for_uav_move": None,
        "mean_advantage_for_uav_stay": None,
        "median_advantage_for_uav_stay": None,
        "uav_action_conditional_logging_available": False,
        "uav_move_count": None,
        "uav_stay_count": None,
        "uav_move_adv_mean": None,
        "uav_stay_adv_mean": None,
        "uav_move_adv_median": None,
        "uav_stay_adv_median": None,
        "uav_move_adv_positive_fraction": None,
        "uav_stay_adv_positive_fraction": None,
        "uav_move_reward_mean": None,
        "uav_stay_reward_mean": None,
        "uav_move_q_total_mean": None,
        "uav_stay_q_total_mean": None,
        "uav_move_movement_cost_mean": None,
        "uav_stay_movement_cost_mean": None,
        "uav_move_delay_mean": None,
        "uav_stay_delay_mean": None,
        "uav_move_energy_mean": None,
        "uav_stay_energy_mean": None,
        "uav_move_return_target_mean": None,
        "uav_stay_return_target_mean": None,
        "uav_move_old_log_prob_mean": None,
        "uav_stay_old_log_prob_mean": None,
        "uav_move_new_log_prob_mean": None,
        "uav_stay_new_log_prob_mean": None,
        "uav_move_ratio_mean": None,
        "uav_stay_ratio_mean": None,
        "uav_move_clip_fraction": None,
        "uav_stay_clip_fraction": None,
        "uav_selected_move_count": None,
        "uav_selected_stay_count": None,
        "uav_selected_move_actual_move_count": None,
        "uav_selected_move_actual_stay_count": None,
        "uav_selected_stay_actual_move_count": None,
        "uav_selected_stay_actual_stay_count": None,
        "uav_feasible_action_count_mean": None,
        "uav_feasible_action_count_min": None,
        "uav_feasible_action_count_max": None,
        "uav_move_candidate_count_mean": None,
        "uav_move_candidate_count_min": None,
        "uav_move_candidate_count_max": None,
        "uav_action_mask_summary": None,
    }
    if not available:
        return empty

    move = np.asarray(move_probabilities, dtype=np.float64)
    stay = np.asarray(stay_probabilities, dtype=np.float64)
    margins = np.asarray(logit_margins, dtype=np.float64)
    feasible = np.asarray(move_feasible, dtype=bool)
    selected = np.asarray(move_selected, dtype=bool)
    actual = np.asarray(actual_movement, dtype=bool)
    expected_length = len(move)
    if expected_length <= 0 or not all(
        array.ndim == 1 and len(array) == expected_length
        for array in (stay, margins, feasible, selected, actual)
    ):
        raise AssertionError("UAV diagnostic arrays must be aligned and non-empty")
    if not (
        np.all(np.isfinite(move))
        and np.all(np.isfinite(stay))
        and np.all(np.isfinite(margins))
    ):
        raise FloatingPointError("UAV diagnostic arrays contain NaN/Inf")
    if not (
        np.all((move >= 0.0) & (move <= 1.0))
        and np.all((stay >= 0.0) & (stay <= 1.0))
        and np.allclose(move + stay, 1.0, rtol=0.0, atol=1.0e-6)
    ):
        raise AssertionError("UAV diagnostic probabilities are invalid")

    def _aligned_float_array(values: list[float], field: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or len(array) != expected_length:
            raise AssertionError(f"{field} must align with UAV diagnostic samples")
        if not np.all(np.isfinite(array)):
            raise FloatingPointError(f"{field} contains NaN/Inf")
        return array

    advantages = _aligned_float_array(global_advantages, "global_advantages")
    return_targets = _aligned_float_array(
        global_return_targets, "global_return_targets"
    )
    rewards_a = _aligned_float_array(rewards, "rewards")
    q_values_a = _aligned_float_array(q_values, "q_values")
    movement_costs_a = _aligned_float_array(movement_costs, "movement_costs")
    delays_a = _aligned_float_array(delays, "delays")
    energies_a = _aligned_float_array(energies, "energies")
    old_log_probs_a = _aligned_float_array(old_log_probs, "old_log_probs")
    new_log_probs_a = _aligned_float_array(new_log_probs, "new_log_probs")
    ratios_a = _aligned_float_array(ratios, "ratios")
    if np.any(ratios_a < 0.0):
        raise AssertionError("UAV branch ratios must be non-negative")
    clipped_a = np.asarray(clipped, dtype=bool)
    if clipped_a.ndim != 1 or len(clipped_a) != expected_length:
        raise AssertionError("clipped must align with UAV diagnostic samples")
    feasible_counts_a = np.asarray(feasible_action_counts, dtype=np.float64)
    candidate_counts_a = np.asarray(move_candidate_counts, dtype=np.float64)
    if not (
        feasible_counts_a.ndim == candidate_counts_a.ndim == 1
        and len(feasible_counts_a) == len(candidate_counts_a) == expected_length
        and np.all(np.isfinite(feasible_counts_a))
        and np.all(np.isfinite(candidate_counts_a))
        and np.all(feasible_counts_a >= 0.0)
        and np.all(candidate_counts_a >= 0.0)
        and np.all(feasible_counts_a == np.floor(feasible_counts_a))
        and np.all(candidate_counts_a == np.floor(candidate_counts_a))
    ):
        raise AssertionError("UAV action mask counts are invalid")

    def _percentile(values: np.ndarray, percentile: float) -> float:
        return float(np.percentile(values, percentile))

    def _advantage_summary(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        values_a = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(values_a)):
            raise FloatingPointError("UAV action advantages contain NaN/Inf")
        return float(np.mean(values_a)), float(np.median(values_a))

    def _group_stats(mask: np.ndarray) -> dict[str, Any]:
        if mask.ndim != 1 or len(mask) != expected_length:
            raise AssertionError("UAV action-conditional mask is not aligned")
        if not bool(np.any(mask)):
            return {
                "count": 0,
                "adv_mean": None,
                "adv_median": None,
                "adv_positive_fraction": None,
                "reward_mean": None,
                "q_total_mean": None,
                "movement_cost_mean": None,
                "delay_mean": None,
                "energy_mean": None,
                "return_target_mean": None,
                "old_log_prob_mean": None,
                "new_log_prob_mean": None,
                "ratio_mean": None,
                "clip_fraction": None,
            }

        def _mean(values: np.ndarray) -> float:
            result = float(np.mean(values[mask]))
            if not np.isfinite(result):
                raise FloatingPointError("UAV action-conditional mean is non-finite")
            return result

        def _median(values: np.ndarray) -> float:
            result = float(np.median(values[mask]))
            if not np.isfinite(result):
                raise FloatingPointError("UAV action-conditional median is non-finite")
            return result

        return {
            "count": int(np.sum(mask)),
            "adv_mean": _mean(advantages),
            "adv_median": _median(advantages),
            "adv_positive_fraction": float(np.mean(advantages[mask] > 0.0)),
            "reward_mean": _mean(rewards_a),
            "q_total_mean": _mean(q_values_a),
            "movement_cost_mean": _mean(movement_costs_a),
            "delay_mean": _mean(delays_a),
            "energy_mean": _mean(energies_a),
            "return_target_mean": _mean(return_targets),
            "old_log_prob_mean": _mean(old_log_probs_a),
            "new_log_prob_mean": _mean(new_log_probs_a),
            "ratio_mean": _mean(ratios_a),
            "clip_fraction": float(np.mean(clipped_a[mask])),
        }

    move_advantage_mean, move_advantage_median = _advantage_summary(move_advantages)
    stay_advantage_mean, stay_advantage_median = _advantage_summary(stay_advantages)
    actual_move_stats = _group_stats(actual)
    actual_stay_stats = _group_stats(~actual)
    probability_entropy = -(
        np.where(move > 0.0, move * np.log(np.maximum(move, 1.0e-12)), 0.0)
        + np.where(stay > 0.0, stay * np.log(np.maximum(stay, 1.0e-12)), 0.0)
    )
    selected_count = int(np.sum(selected))
    return {
        "uav_move_probability_mean": float(np.mean(move)),
        "uav_move_probability_median": float(np.median(move)),
        "uav_move_probability_p10": _percentile(move, 10.0),
        "uav_move_probability_p90": _percentile(move, 90.0),
        "uav_move_probability_gt_0_5_rate": float(np.mean(move > 0.5)),
        "uav_stay_probability_mean": float(np.mean(stay)),
        "uav_head_entropy": float(np.mean(probability_entropy)),
        "uav_binary_entropy_mean": float(np.mean(probability_entropy)),
        "uav_stay_move_logit_margin_mean": float(np.mean(margins)),
        "uav_stay_move_logit_margin_median": float(np.median(margins)),
        "uav_move_feasible_count": int(np.sum(feasible)),
        "uav_move_feasible_rate": float(np.mean(feasible)),
        "uav_move_selected_count": selected_count,
        "uav_move_selected_rate": float(np.mean(selected)),
        "uav_move_given_selected": (
            None
            if selected_count == 0
            else float(np.sum(actual & selected)) / float(selected_count)
        ),
        "uav_action_advantage_logging_available": True,
        "mean_advantage_for_uav_move": move_advantage_mean,
        "median_advantage_for_uav_move": move_advantage_median,
        "mean_advantage_for_uav_stay": stay_advantage_mean,
        "median_advantage_for_uav_stay": stay_advantage_median,
        "uav_action_conditional_logging_available": True,
        "uav_move_count": actual_move_stats["count"],
        "uav_stay_count": actual_stay_stats["count"],
        "uav_move_adv_mean": actual_move_stats["adv_mean"],
        "uav_stay_adv_mean": actual_stay_stats["adv_mean"],
        "uav_move_adv_median": actual_move_stats["adv_median"],
        "uav_stay_adv_median": actual_stay_stats["adv_median"],
        "uav_move_adv_positive_fraction": actual_move_stats[
            "adv_positive_fraction"
        ],
        "uav_stay_adv_positive_fraction": actual_stay_stats[
            "adv_positive_fraction"
        ],
        "uav_move_reward_mean": actual_move_stats["reward_mean"],
        "uav_stay_reward_mean": actual_stay_stats["reward_mean"],
        "uav_move_q_total_mean": actual_move_stats["q_total_mean"],
        "uav_stay_q_total_mean": actual_stay_stats["q_total_mean"],
        "uav_move_movement_cost_mean": actual_move_stats["movement_cost_mean"],
        "uav_stay_movement_cost_mean": actual_stay_stats["movement_cost_mean"],
        "uav_move_delay_mean": actual_move_stats["delay_mean"],
        "uav_stay_delay_mean": actual_stay_stats["delay_mean"],
        "uav_move_energy_mean": actual_move_stats["energy_mean"],
        "uav_stay_energy_mean": actual_stay_stats["energy_mean"],
        "uav_move_return_target_mean": actual_move_stats["return_target_mean"],
        "uav_stay_return_target_mean": actual_stay_stats["return_target_mean"],
        "uav_move_old_log_prob_mean": actual_move_stats["old_log_prob_mean"],
        "uav_stay_old_log_prob_mean": actual_stay_stats["old_log_prob_mean"],
        "uav_move_new_log_prob_mean": actual_move_stats["new_log_prob_mean"],
        "uav_stay_new_log_prob_mean": actual_stay_stats["new_log_prob_mean"],
        "uav_move_ratio_mean": actual_move_stats["ratio_mean"],
        "uav_stay_ratio_mean": actual_stay_stats["ratio_mean"],
        "uav_move_clip_fraction": actual_move_stats["clip_fraction"],
        "uav_stay_clip_fraction": actual_stay_stats["clip_fraction"],
        "uav_selected_move_count": selected_count,
        "uav_selected_stay_count": int(expected_length - selected_count),
        "uav_selected_move_actual_move_count": int(np.sum(selected & actual)),
        "uav_selected_move_actual_stay_count": int(np.sum(selected & ~actual)),
        "uav_selected_stay_actual_move_count": int(np.sum(~selected & actual)),
        "uav_selected_stay_actual_stay_count": int(np.sum(~selected & ~actual)),
        "uav_feasible_action_count_mean": float(np.mean(feasible_counts_a)),
        "uav_feasible_action_count_min": int(np.min(feasible_counts_a)),
        "uav_feasible_action_count_max": int(np.max(feasible_counts_a)),
        "uav_move_candidate_count_mean": float(np.mean(candidate_counts_a)),
        "uav_move_candidate_count_min": int(np.min(candidate_counts_a)),
        "uav_move_candidate_count_max": int(np.max(candidate_counts_a)),
        "uav_action_mask_summary": {
            "operation_width": 2,
            "feasible_operation_count_mean": float(np.mean(feasible_counts_a)),
            "feasible_operation_count_min": int(np.min(feasible_counts_a)),
            "feasible_operation_count_max": int(np.max(feasible_counts_a)),
            "move_candidate_count_mean": float(np.mean(candidate_counts_a)),
            "move_candidate_count_min": int(np.min(candidate_counts_a)),
            "move_candidate_count_max": int(np.max(candidate_counts_a)),
        },
    }

def _binary_action_bitmask(action: Any) -> int:
    """将二元动作向量编码为紧凑且确定的整数。"""

    values = np.asarray(action, dtype=np.int8).reshape(-1)
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("binary action bitmask requires only 0/1 values")
    bitmask = 0
    for index, value in enumerate(values.tolist()):
        if int(value):
            bitmask |= 1 << int(index)
    return int(bitmask)

def _branch_update_metric_fields(update: Any) -> dict[str, Any]:
    """返回有限的分支诊断指标。"""

    values = {
        "cache_approx_kl": float(update.cache_approx_kl),
        "uav_approx_kl": float(update.uav_approx_kl),
        "cache_clip_fraction": float(update.cache_clip_fraction),
        "CACHE_PER_HEAD_OLD_NEW_KL_MEAN": float(
            update.cache_per_head_old_new_kl_mean
        ),
        "CACHE_PER_HEAD_OLD_NEW_KL_P50": float(
            update.cache_per_head_old_new_kl_p50
        ),
        "CACHE_PER_HEAD_OLD_NEW_KL_P90": float(
            update.cache_per_head_old_new_kl_p90
        ),
        "CACHE_PER_HEAD_OLD_NEW_KL_MAX": float(
            update.cache_per_head_old_new_kl_max
        ),
        "CACHE_PER_HEAD_RATIO_MEAN": float(update.cache_per_head_ratio_mean),
        "CACHE_PER_HEAD_RATIO_P90": float(update.cache_per_head_ratio_p90),
        "CACHE_PER_HEAD_RATIO_MAX": float(update.cache_per_head_ratio_max),
        "CACHE_PER_HEAD_CLIP_FRACTION": float(update.cache_per_head_clip_fraction),
        "uav_clip_fraction": float(update.uav_clip_fraction),
        "cache_entropy": float(update.cache_entropy),
        "uav_entropy": float(update.uav_entropy),
        "cache_head_gradient_norm": float(update.cache_head_gradient_norm),
        "uav_head_gradient_norm": float(update.uav_head_gradient_norm),
            "shared_actor_gradient_norm": float(update.shared_actor_gradient_norm),
        }
    fields = values if all(math.isfinite(value) for value in values.values()) else {}
    fields.update(
        {
        }
    )
    policy_losses = {
        "cache_policy_loss": float(update.cache_policy_loss),
        "cache_trigger_policy_loss": float(update.cache_trigger_policy_loss),
        "cache_candidate_listwise_loss": float(
            update.cache_candidate_policy_loss
        ),
        "uav_policy_loss": float(update.uav_policy_loss),
        "uav_operation_policy_loss": float(update.uav_operation_policy_loss),
        "uav_candidate_listwise_loss": float(update.uav_candidate_policy_loss),
        "actor_policy_loss": float(update.actor_policy_loss),
    }
    if all(math.isfinite(value) for value in policy_losses.values()):
        fields.update(policy_losses)
    candidate_metrics = {
        "candidate_loss": float(update.uav_candidate_policy_loss),
        "mean_abs_candidate_advantage": float(
            update.uav_candidate_advantage_mean_abs
        ),
        "candidate_adv_weighted_mean_residual": float(
            update.uav_candidate_advantage_weighted_mean_residual
        ),
    }
    if all(math.isfinite(value) for value in candidate_metrics.values()):
        fields.update(candidate_metrics)
        fields["candidate_advantage_normalization"] = str(
            update.uav_candidate_advantage_normalization
        )
        fields["candidate_advantage_source"] = str(
            update.uav_candidate_advantage_source
        )
    candidate_policy_metrics = {
        "UAV_CANDIDATE_POLICY_ENTROPY_MEAN": float(
            update.uav_candidate_policy_entropy_mean
        ),
        "UAV_CANDIDATE_TEACHER_ENTROPY_MEAN": float(
            update.uav_candidate_teacher_entropy_mean
        ),
        "UAV_CANDIDATE_BEST_POLICY_PROB_MEAN": float(
            update.uav_candidate_best_policy_prob_mean
        ),
        "UAV_CANDIDATE_BEST_POLICY_PROB_MEDIAN": float(
            update.uav_candidate_best_policy_prob_median
        ),
        "UAV_CANDIDATE_BEST_POLICY_RANK_MEAN": float(
            update.uav_candidate_best_policy_rank_mean
        ),
        "UAV_CANDIDATE_BEST_POLICY_TOP1_RATE": float(
            update.uav_candidate_best_policy_top1_rate
        ),
        "UAV_CANDIDATE_BEST_POLICY_TOP3_RATE": float(
            update.uav_candidate_best_policy_top3_rate
        ),
        "UAV_CANDIDATE_LOGIT_STD_MEAN": float(update.uav_candidate_logit_std_mean),
        "UAV_CANDIDATE_LOGIT_RANGE_MEAN": float(
            update.uav_candidate_logit_range_mean
        ),
    }
    if all(math.isfinite(value) for value in candidate_policy_metrics.values()):
        fields.update(candidate_policy_metrics)
    candidate_confidence_metrics = {
        "UAV_CANDIDATE_CONFIDENCE_MEAN": float(
            update.uav_candidate_confidence_mean
        ),
        "UAV_CANDIDATE_CONFIDENCE_P10": float(update.uav_candidate_confidence_p10),
        "UAV_CANDIDATE_CONFIDENCE_P50": float(update.uav_candidate_confidence_p50),
        "UAV_CANDIDATE_CONFIDENCE_P90": float(update.uav_candidate_confidence_p90),
        "UAV_CANDIDATE_CONFIDENCE_SATURATION_RATE": float(
            update.uav_candidate_confidence_saturation_rate
        ),
    }
    if all(math.isfinite(value) for value in candidate_confidence_metrics.values()):
        fields.update(candidate_confidence_metrics)
    fields.update(
        {
            "CACHE_ACTOR_GRAD_NORM_PRECLIP": [
                float(value)
                for value in update.cache_actor_grad_norm_preclip_by_update_epoch
            ],
            "CACHE_ACTOR_GRAD_CLIP_COEFFICIENT": [
                float(value)
                for value in update.cache_actor_grad_clip_coefficient_by_update_epoch
            ],
            "UAV_ACTOR_GRAD_NORM_PRECLIP": [
                float(value)
                for value in update.uav_actor_grad_norm_preclip_by_update_epoch
            ],
            "UAV_ACTOR_GRAD_CLIP_COEFFICIENT": [
                float(value)
                for value in update.uav_actor_grad_clip_coefficient_by_update_epoch
            ],
            "COMBINED_ACTOR_GRAD_NORM_POSTCLIP": [
                float(value)
                for value in update.combined_actor_grad_norm_postclip_by_update_epoch
            ],
            "UAV_CANDIDATE_SCORER_GRADIENT_NORM_BY_UPDATE_EPOCH": [
                float(value)
                for value in update.uav_candidate_scorer_gradient_norm_by_update_epoch
            ],
            "UAV_CANDIDATE_SCORER_PARAMETER_DELTA_NORM_BY_UPDATE_EPOCH": [
                float(value)
                for value in update.uav_candidate_scorer_parameter_delta_norm_by_update_epoch
            ],
            "UAV_ENCODER_GRADIENT_NORM_BY_UPDATE_EPOCH": [
                float(value)
                for value in update.uav_encoder_gradient_norm_by_update_epoch
            ],
            "UAV_OPERATION_ENCODER_GRADIENT_NORM_BY_UPDATE_EPOCH": [
                float(value)
                for value in update.uav_encoder_gradient_norm_by_update_epoch
            ],
            "UAV_ENCODER_PARAMETER_DELTA_NORM_BY_UPDATE_EPOCH": [
                float(value)
                for value in update.uav_encoder_parameter_delta_norm_by_update_epoch
            ],
            "UAV_OPERATION_ENCODER_PARAMETER_DELTA_NORM_BY_UPDATE_EPOCH": [
                float(value)
                for value in update.uav_encoder_parameter_delta_norm_by_update_epoch
            ],
            "UAV_CANDIDATE_ENCODER_GRADIENT_NORM_BY_UPDATE_EPOCH": [
                float(value)
                for value in update.uav_candidate_encoder_gradient_norm_by_update_epoch
            ],
            "UAV_CANDIDATE_ENCODER_PARAMETER_DELTA_NORM_BY_UPDATE_EPOCH": [
                float(value)
                for value in update.uav_candidate_encoder_parameter_delta_norm_by_update_epoch
            ],
            "UAV_CANDIDATE_OLD_NEW_KL_MEAN_BY_UPDATE_EPOCH": [
                float(value)
                for value in update.uav_candidate_old_new_kl_mean_by_update_epoch
            ],
            "UAV_CANDIDATE_OLD_NEW_KL_P90_BY_UPDATE_EPOCH": [
                float(value)
                for value in update.uav_candidate_old_new_kl_p90_by_update_epoch
            ],
            "UAV_CANDIDATE_OLD_NEW_KL_MAX_BY_UPDATE_EPOCH": [
                float(value)
                for value in update.uav_candidate_old_new_kl_max_by_update_epoch
            ],
            "UAV_CANDIDATE_ARGMAX_FLIP_RATE_BY_UPDATE_EPOCH": [
                float(value)
                for value in update.uav_candidate_argmax_flip_rate_by_update_epoch
            ],
            "UAV_CANDIDATE_BEST_POLICY_PROB_CHANGE_MEAN_BY_UPDATE_EPOCH": [
                float(value)
                for value in update.uav_candidate_best_policy_prob_change_mean_by_update_epoch
            ],
            "UAV_CANDIDATE_BEST_POLICY_PROB_CHANGE_P10_BY_UPDATE_EPOCH": [
                float(value)
                for value in update.uav_candidate_best_policy_prob_change_p10_by_update_epoch
            ],
        }
    )
    combined_postclip = np.asarray(
        update.combined_actor_grad_norm_postclip_by_update_epoch,
        dtype=np.float64,
    )
    if combined_postclip.size:
        fields.update(
            {
                "COMBINED_POSTCLIP_NORM_MEAN": float(np.mean(combined_postclip)),
                "COMBINED_POSTCLIP_NORM_MAX": float(np.max(combined_postclip)),
                "COMBINED_POSTCLIP_GT_0P5_FRACTION": float(
                    np.mean(combined_postclip > 0.5)
                ),
            }
        )
    candidate_drift_metrics = {
        "UAV_CANDIDATE_OLD_NEW_KL_MEAN": float(
            update.uav_candidate_old_new_kl_mean
        ),
        "UAV_CANDIDATE_OLD_NEW_KL_P50": float(update.uav_candidate_old_new_kl_p50),
        "UAV_CANDIDATE_OLD_NEW_KL_P90": float(update.uav_candidate_old_new_kl_p90),
        "UAV_CANDIDATE_OLD_NEW_KL_MAX": float(update.uav_candidate_old_new_kl_max),
        "UAV_CANDIDATE_ARGMAX_FLIP_RATE": float(
            update.uav_candidate_argmax_flip_rate
        ),
        "UAV_CANDIDATE_BEST_POLICY_PROB_CHANGE_MEAN": float(
            update.uav_candidate_best_policy_prob_change_mean
        ),
        "UAV_CANDIDATE_BEST_POLICY_PROB_CHANGE_P10": float(
            update.uav_candidate_best_policy_prob_change_p10
        ),
    }
    if all(math.isfinite(value) for value in candidate_drift_metrics.values()):
        fields.update(candidate_drift_metrics)
    candidate_gate_metrics = {
        "UAV_CANDIDATE_TARGET_KL": float(update.uav_candidate_target_kl),
    }
    if all(math.isfinite(value) for value in candidate_gate_metrics.values()):
        fields.update(candidate_gate_metrics)
    candidate_effective_kl_metrics = {
        "UAV_CANDIDATE_EFFECTIVE_KL_AT_GATE": float(
            update.uav_candidate_effective_kl_at_gate
        ),
        "UAV_CANDIDATE_EFFECTIVE_KL_IMMEDIATELY_AFTER_GATE": float(
            update.uav_candidate_effective_kl_immediately_after_gate
        ),
        "UAV_CANDIDATE_EFFECTIVE_KL_AFTER_GATE": float(
            update.uav_candidate_effective_kl_after_gate
        ),
        "UAV_CANDIDATE_POST_GATE_DRIFT_FROM_GATE": float(
            update.uav_candidate_post_gate_drift_from_gate
        ),
    }
    fields.update(
        {
            key: value
            for key, value in candidate_effective_kl_metrics.items()
            if math.isfinite(value)
        }
    )
    fields.update(
        {
            "UAV_CANDIDATE_KL_GATE_STATISTIC": str(
                update.uav_candidate_kl_gate_statistic
            ),
            "UAV_CANDIDATE_KL_GATE_TRIGGER_SOURCE": str(
                update.uav_candidate_kl_gate_trigger_source
            ),
            "UAV_CANDIDATE_KL_STOP_EPOCH": update.uav_candidate_kl_stop_epoch,
            "UAV_CANDIDATE_ACTIVE_PER_UPDATE_EPOCH": [
                bool(value) for value in update.uav_candidate_active_per_update_epoch
            ],
            "UAV_CANDIDATE_SUPPRESSED_UPDATE_EPOCHS": int(
                update.uav_candidate_suppressed_update_epochs
            ),
        }
    )
    scorer_deltas = {
        "cache_target_scorer_parameter_delta_norm": float(
            update.cache_target_scorer_parameter_delta_norm
        ),
        "uav_target_scorer_parameter_delta_norm": float(
            update.uav_target_scorer_parameter_delta_norm
        ),
    }
    if all(math.isfinite(value) for value in scorer_deltas.values()):
        fields.update(scorer_deltas)
    exact_q_quality = {
        "uav_top1_exact_q_accuracy_before": float(
            update.uav_top1_exact_q_accuracy_before
        ),
        "uav_top1_exact_q_accuracy_after": float(
            update.uav_top1_exact_q_accuracy_after
        ),
        "uav_mean_exact_q_regret_before": float(
            update.uav_mean_exact_q_regret_before
        ),
        "uav_mean_exact_q_regret_after": float(
            update.uav_mean_exact_q_regret_after
        ),
        "uav_expected_exact_q_before": float(update.uav_expected_exact_q_before),
        "uav_expected_exact_q_after": float(update.uav_expected_exact_q_after),
        "uav_expected_exact_q_delta": float(update.uav_expected_exact_q_delta),
        "uav_target_policy_kl_old_new": float(update.uav_target_policy_kl_old_new),
    }
    if all(math.isfinite(value) for value in exact_q_quality.values()):
        fields.update(exact_q_quality)
    if update.cache_proposal_diagnostics is not None:
        fields["cache_proposal_diagnostics"] = dict(update.cache_proposal_diagnostics)
    if update.uav_proposal_diagnostics is not None:
        fields["uav_proposal_diagnostics"] = dict(update.uav_proposal_diagnostics)
    fields.update(
        {
            "CACHE_TRIGGER_CREDIT_SOURCE": str(update.cache_trigger_credit_source),
            "CACHE_CANDIDATE_CREDIT_SOURCE": str(update.cache_candidate_credit_source),
            "CACHE_VALUE_EVALUATOR": str(update.cache_value_evaluator),
            "CACHE_DOWNSTREAM_MODE": str(update.cache_downstream_mode),
            "CACHE_TRIGGER_TEACHER_TEMPERATURE": float(
                update.cache_trigger_teacher_temperature
            ),
            "CACHE_CANDIDATE_TEACHER_TEMPERATURE": float(
                update.cache_candidate_teacher_temperature
            ),
            "CACHE_TRIGGER_VALID_HEAD_COUNT": int(
                update.cache_trigger_valid_head_count
            ),
            "CACHE_CANDIDATE_VALID_HEAD_COUNT": int(
                update.cache_candidate_valid_head_count
            ),
            "CACHE_Q_KEEP_MEAN": float(update.cache_q_keep_mean),
            "CACHE_Q_BEST_UPDATE_MEAN": float(update.cache_q_best_update_mean),
            "CACHE_DELTA_BEST_MEAN": float(update.cache_delta_best_mean),
            "CACHE_LOGICAL_UPDATE_CANDIDATE_COUNT": int(
                update.cache_logical_candidate_count
            ),
            "CACHE_FULL_EXACT_KEEP_BRANCH_COUNT": int(update.cache_keep_exact_count),
            "CACHE_FULL_EXACT_UPDATE_BRANCH_COUNT": int(
                update.cache_full_exact_update_branch_count
            ),
            "CACHE_DEDUPLICATED_BRANCH_COUNT": int(
                update.cache_deduplicated_branch_count
            ),
            "CACHE_COUNTERFACTUAL_ENV_EXECUTIONS": int(
                update.cache_credit_counterfactual_env_executions
            ),
            "CACHE_TEACHER_EVALUATION_RUNTIME_SECONDS": float(
                update.cache_credit_runtime_seconds
            ),
            "CACHE_EXACT_CANDIDATE_COUNT": int(
                update.cache_exact_candidate_count
            ),
            "CACHE_SURROGATE_CANDIDATE_COUNT": int(
                update.cache_surrogate_candidate_count
            ),
            "CACHE_COUNTERFACTUAL_EXECUTIONS_SAVED": int(
                update.cache_counterfactual_executions_saved
            ),
            "CACHE_COUNTERFACTUAL_EXECUTION_REDUCTION_RATIO": float(
                update.cache_counterfactual_execution_reduction_fraction
            ),
            "CACHE_SURROGATE_SCREENING_ACTIVE": bool(
                update.cache_surrogate_screening_active
            ),
            "CACHE_SURROGATE_SCREENING_READY": bool(
                update.cache_surrogate_screening_ready
            ),
            "CACHE_SURROGATE_SELECTED_K": update.cache_surrogate_selected_k,
            "CACHE_SURROGATE_DELTA_Q_MAE": float(
                update.cache_surrogate_delta_q_mae
            ),
            "CACHE_SURROGATE_DELTA_Q_RMSE": float(
                update.cache_surrogate_delta_q_rmse
            ),
            "CACHE_SURROGATE_GATE_METRICS": update.cache_surrogate_gate_metrics,
            "CACHE_TEACHER_SCALE_CALIBRATION": update.cache_teacher_scale_calibration,
            "CACHE_TRIGGER_DELTA_Q_STATISTICS": (
                update.cache_trigger_delta_best_statistics
            ),
            "CACHE_TRIGGER_TEACHER_STATISTICS": update.cache_trigger_teacher_statistics,
            "CACHE_CANDIDATE_TEACHER_STATISTICS": update.cache_candidate_teacher_statistics,
        }
    )
    if math.isfinite(float(update.uav_operation_credit_runtime_seconds)):
        fields["uav_operation_credit_runtime_seconds"] = float(
            update.uav_operation_credit_runtime_seconds
        )
        fields["uav_operation_counterfactual_env_executions"] = int(
            update.uav_operation_counterfactual_env_executions
        )
        fields.update(
            {
                "UAV_EXACT_Q_BRANCH_REQUEST_COUNT": int(
                    update.uav_exact_q_branch_request_count
                ),
                "UAV_EXACT_Q_UNIQUE_BRANCH_COUNT": int(
                    update.uav_exact_q_unique_branch_count
                ),
                "UAV_EXACT_Q_DUPLICATE_BRANCH_COUNT": int(
                    update.uav_exact_q_duplicate_branch_count
                ),
                "UAV_EXACT_Q_CACHE_HIT_COUNT": int(
                    update.uav_exact_q_cache_hit_count
                ),
                "EXECUTE_SLOT_CALL_COUNT": int(
                    update.uav_exact_q_execute_slot_call_count
                ),
                "MATCHING_CALL_COUNT": int(update.uav_exact_q_matching_call_count),
                # 该值来自 Matching/KKT 资源缓存的实际请求次数。
                "KKT_CALL_COUNT": int(update.uav_exact_q_kkt_call_count),
                "KKT_CACHE_REQUEST_COUNT": int(
                    update.uav_exact_q_kkt_cache_request_count
                ),
                "KKT_CACHE_HIT_COUNT": int(update.uav_exact_q_kkt_cache_hit_count),
                "KKT_CACHE_MISS_COUNT": int(update.uav_exact_q_kkt_cache_miss_count),
                "KKT_SOLVER_CALL_COUNT": int(
                    update.uav_exact_q_kkt_solver_call_count
                ),
                "NODE_EVAL_CACHE_REQUEST_COUNT": int(
                    update.uav_exact_q_node_eval_cache_request_count
                ),
                "NODE_EVAL_CACHE_HIT_COUNT": int(
                    update.uav_exact_q_node_eval_cache_hit_count
                ),
                "NODE_EVAL_CACHE_MISS_COUNT": int(
                    update.uav_exact_q_node_eval_cache_miss_count
                ),
                "NODE_EVAL_CACHE_HIT_RATE": float(
                    update.uav_exact_q_node_eval_cache_hit_count
                    / update.uav_exact_q_node_eval_cache_request_count
                    if update.uav_exact_q_node_eval_cache_request_count > 0
                    else 0.0
                ),
                "NODE_EVAL_FULL_EXECUTION_COUNT": int(
                    update.uav_exact_q_node_eval_full_execution_count
                ),
                "IDEAL_UTILITY_REQUEST_COUNT": int(
                    update.uav_exact_q_ideal_utility_request_count
                ),
                "IDEAL_UTILITY_CACHE_HIT_COUNT": int(
                    update.uav_exact_q_ideal_utility_cache_hit_count
                ),
                "IDEAL_UTILITY_CACHE_MISS_COUNT": int(
                    update.uav_exact_q_ideal_utility_cache_miss_count
                ),
                "IDEAL_UTILITY_CACHE_HIT_RATE": float(
                    update.uav_exact_q_ideal_utility_cache_hit_count
                    / update.uav_exact_q_ideal_utility_request_count
                    if update.uav_exact_q_ideal_utility_request_count > 0
                    else 0.0
                ),
                "IMPACT_SCOPE_REQUEST_COUNT": int(
                    update.uav_exact_q_impact_scope_request_count
                ),
                "IMPACT_SCOPE_COMPUTE_COUNT": int(
                    update.uav_exact_q_impact_scope_compute_count
                ),
                "IMPACT_SCOPE_REUSE_COUNT": int(
                    update.uav_exact_q_impact_scope_reuse_count
                ),
                "IMPACT_SCOPE_REUSE_RATE": float(
                    update.uav_exact_q_impact_scope_reuse_count
                    / update.uav_exact_q_impact_scope_request_count
                    if update.uav_exact_q_impact_scope_request_count > 0
                    else 0.0
                ),
            }
        )
        if update.uav_operation_timing is not None:
            timing_fields = {
                "EXACT_Q_MATCHING_RUNTIME": "matching_runtime",
                "MATCHING_PROFILE_TOTAL_RUNTIME": "matching_runtime",
                "MATCHING_FEASIBLE_TARGET_RUNTIME": "matching_feasible_target_runtime",
                "MATCHING_CANDIDATE_GENERATION_RUNTIME": "matching_candidate_generation_runtime",
                "MATCHING_CANDIDATE_BUILD_RUNTIME": "matching_candidate_generation_runtime",
                "MATCHING_IMPACT_SCOPE_RUNTIME": "matching_impact_scope_runtime",
                "MATCHING_UPPER_BOUND_RUNTIME": "matching_upper_bound_runtime",
                "MATCHING_IDEAL_UTILITY_RUNTIME": "matching_ideal_utility_runtime",
                "MATCHING_CHANNEL_RATE_RUNTIME": "matching_channel_rate_runtime",
                "MATCHING_EXACT_EVAL_ORCHESTRATION_RUNTIME": "matching_exact_eval_orchestration_runtime",
                "MATCHING_EXACT_EVAL_CONTROL_RUNTIME": "matching_exact_eval_orchestration_runtime",
                "MATCHING_REPLAY_TIEBREAK_RUNTIME": "matching_replay_tiebreak_runtime",
                "MATCHING_OTHER_RUNTIME": "matching_other_runtime",
                "EXACT_Q_NODE_CACHE_LOOKUP_RUNTIME": "node_cache_lookup_runtime",
                # Node 求解未命中耗时不包含单独记录的
                # KKT allocation 耗时，以保证两类统计互不重叠。
                "EXACT_Q_NODE_EVAL_RUNTIME": "node_eval_miss_runtime",
                "EXACT_Q_NODE_EVAL_MISS_RUNTIME": "node_eval_miss_runtime",
                "EXACT_Q_KKT_RUNTIME": "kkt_runtime",
                "KKT_COEFFICIENT_BUILD_RUNTIME": "kkt_coefficient_build_runtime",
                "KKT_SOLVER_CORE_RUNTIME": "kkt_solver_core_runtime",
                "KKT_RESULT_ASSEMBLY_RUNTIME": "kkt_result_assembly_runtime",
                "REWARD_DELAY_AGGREGATION_RUNTIME": "reward_delay_aggregation_runtime",
                "REWARD_ENERGY_AGGREGATION_RUNTIME": "reward_energy_aggregation_runtime",
                "REWARD_COST_AGGREGATION_RUNTIME": "reward_cost_aggregation_runtime",
                "REWARD_FINAL_COMBINATION_RUNTIME": "reward_final_combination_runtime",
                "EXACT_Q_BRANCH_STATE_BUILD_RUNTIME": "branch_state_build_runtime",
                "EXACT_Q_REWARD_ASSEMBLY_RUNTIME": "reward_assembly_runtime",
                "EXACT_Q_CRITIC_RUNTIME": "critic_runtime",
                "EXACT_Q_OTHER_RUNTIME": "other_runtime",
                "PARALLEL_WORKER_STARTUP_RUNTIME": "parallel_worker_startup_runtime",
                "WORK_ITEM_SERIALIZATION_RUNTIME": "work_item_serialization_runtime",
                "RESULT_MERGE_RUNTIME": "result_merge_runtime",
                "EXACT_Q_PARALLEL_WORKERS": "parallel_worker_count",
            }
            total_runtime = float(update.uav_operation_credit_runtime_seconds)
            if math.isfinite(total_runtime) and total_runtime > 0.0:
                for metric_name, timing_key in timing_fields.items():
                    value = float(update.uav_operation_timing.get(timing_key, 0.0))
                    if not math.isfinite(value):
                        continue
                    fields[metric_name] = value
                    fields[f"{metric_name}_FRACTION"] = value / total_runtime
                exact_eval_subhotspot_fields = {
                    "EXACT_EVAL_KKT_CONVERSION_RUNTIME": "exact_eval_kkt_conversion_runtime",
                    "EXACT_EVAL_STATE_COPY_RUNTIME": "exact_eval_state_copy_runtime",
                    "EXACT_EVAL_AFFECTED_NODE_PREP_RUNTIME": "exact_eval_affected_node_prep_runtime",
                    "EXACT_EVAL_NODE_CALL_ORCHESTRATION_RUNTIME": "exact_eval_node_call_orchestration_runtime",
                    "EXACT_EVAL_RESULT_MERGE_RUNTIME": "exact_eval_result_merge_runtime",
                    "EXACT_EVAL_NON_EDGE_PREP_RUNTIME": "exact_eval_non_edge_prep_runtime",
                    "EXACT_EVAL_TARGET_COUNT_BUILD_RUNTIME": "exact_eval_target_count_build_runtime",
                    "EXACT_EVAL_UTILITY_SUM_RUNTIME": "exact_eval_utility_sum_runtime",
                    "EXACT_EVAL_OTHER_RUNTIME": "exact_eval_other_runtime",
                }
                exact_eval_control_runtime = float(
                    update.uav_operation_timing.get(
                        "matching_exact_eval_orchestration_runtime", 0.0
                    )
                )
                classified_subhotspot_runtime = 0.0
                for metric_name, timing_key in exact_eval_subhotspot_fields.items():
                    value = float(update.uav_operation_timing.get(timing_key, 0.0))
                    if not math.isfinite(value):
                        continue
                    fields[metric_name] = value
                    if exact_eval_control_runtime > 0.0:
                        fields[f"{metric_name}_FRACTION"] = (
                            value / exact_eval_control_runtime
                        )
                    classified_subhotspot_runtime += value
                fields["EXACT_EVAL_RECORD_BOOKKEEPING_RUNTIME"] = float(
                    update.uav_operation_timing.get(
                        "exact_eval_record_bookkeeping_runtime", 0.0
                    )
                )
                fields["EXACT_EVAL_SUBHOTSPOT_CLASSIFIED_RUNTIME"] = (
                    classified_subhotspot_runtime
                )
                fields["EXACT_EVAL_UNCLASSIFIED_RUNTIME"] = (
                    exact_eval_control_runtime - classified_subhotspot_runtime
                )
                fields["CURRENT_KKT_CONVERSION_CALL_COUNT"] = int(
                    round(
                        float(
                            update.uav_operation_timing.get(
                                "current_kkt_conversion_call_count", 0.0
                            )
                        )
                    )
                )
                fields["CANDIDATE_KKT_CONVERSION_CALL_COUNT"] = int(
                    round(
                        float(
                            update.uav_operation_timing.get(
                                "candidate_kkt_conversion_call_count", 0.0
                            )
                        )
                    )
                )
                fields["EXACT_EVAL_STATE_COPY_CALL_COUNT"] = int(
                    round(
                        float(
                            update.uav_operation_timing.get(
                                "exact_eval_state_copy_call_count", 0.0
                            )
                        )
                    )
                )
                fields["CURRENT_KKT_CONTEXT_REQUEST_COUNT"] = int(
                    round(
                        float(
                            update.uav_operation_timing.get(
                                "current_kkt_context_request_count", 0.0
                            )
                        )
                    )
                )
                fields["CURRENT_KKT_CONTEXT_REUSE_COUNT"] = int(
                    round(
                        float(
                            update.uav_operation_timing.get(
                                "current_kkt_context_reuse_count", 0.0
                            )
                        )
                    )
                )
                fields["CANDIDATE_KKT_CONTEXT_REQUEST_COUNT"] = int(
                    round(
                        float(
                            update.uav_operation_timing.get(
                                "candidate_kkt_context_request_count", 0.0
                            )
                        )
                    )
                )
                fields["CANDIDATE_KKT_CONTEXT_REUSE_COUNT"] = int(
                    round(
                        float(
                            update.uav_operation_timing.get(
                                "candidate_kkt_context_reuse_count", 0.0
                            )
                        )
                    )
                )
    routing_metrics = {
        "UAV_OPERATION_FULL_ROLLOUT_KL": float(
            update.uav_operation_full_rollout_kl
        ),
        "uav_operation_head_parameter_delta_norm": float(
            update.uav_operation_head_parameter_delta_norm
        ),
        "uav_operation_encoder_parameter_delta_norm": float(
            update.uav_encoder_parameter_delta_norm
        ),
        "uav_candidate_encoder_parameter_delta_norm": float(
            update.uav_candidate_encoder_parameter_delta_norm
        ),
    }
    if all(math.isfinite(value) for value in routing_metrics.values()):
        fields.update(routing_metrics)
    fields.update(
        {
            "UAV_OPERATION_STEPS_EXECUTED": int(update.uav_operation_steps_executed),
            "UAV_OPERATION_STEPS_SUPPRESSED": int(update.uav_operation_steps_suppressed),
            "UAV_CANDIDATE_FULL_STEPS": int(update.uav_candidate_full_steps),
            "UAV_CANDIDATE_ONLY_STEPS_AFTER_OPERATION_STOP": int(
                update.uav_candidate_only_steps_after_operation_stop
            ),
            "UAV_CANDIDATE_OPTIMIZER_STEP_COUNT": int(
                update.uav_candidate_optimizer_step_count
            ),
            "UAV_CANDIDATE_SCORER_OPTIMIZER_STEP_COUNT": int(
                update.uav_candidate_optimizer_step_count
            ),
            "UAV_OPERATION_ENCODER_OPTIMIZER_STEP_COUNT": int(
                update.uav_operation_encoder_optimizer_step_count
            ),
            "UAV_CANDIDATE_ENCODER_OPTIMIZER_STEP_COUNT": int(
                update.uav_candidate_encoder_optimizer_step_count
            ),
            "UAV_CANDIDATE_SCORER_ENCODER_STEP_GAP": int(
                update.uav_candidate_scorer_encoder_step_gap
            ),
        }
    )
    if update.uav_operation_teacher_diagnostics is not None:
        fields.update(dict(update.uav_operation_teacher_diagnostics))
    if update.uav_operation_diagnostics is not None:
        fields.update(dict(update.uav_operation_diagnostics))
    encoder_gradient_norms = {
        "cache_encoder_gradient_norm": float(update.cache_encoder_gradient_norm),
        "uav_operation_encoder_gradient_norm": float(update.uav_encoder_gradient_norm),
        "uav_candidate_encoder_gradient_norm": float(
            update.uav_candidate_encoder_gradient_norm
        ),
    }
    if all(math.isfinite(value) for value in encoder_gradient_norms.values()):
        fields.update(encoder_gradient_norms)
    fields.update(
        {
            "MATCHING_NATURAL_STABLE_COUNT": int(
                update.matching_natural_stable_count
            ),
            "MATCHING_MAX_UPDATES_HIT_COUNT": int(
                update.matching_max_updates_hit_count
            ),
            "MATCHING_MAX_UPDATES_HIT_RATE": float(
                update.matching_max_updates_hit_rate
            ),
            "MATCHING_UPDATE_COUNT_MEAN": float(update.matching_update_count_mean),
            "MATCHING_UPDATE_COUNT_P95": float(update.matching_update_count_p95),
            "MATCHING_UPDATE_COUNT_MAX": int(update.matching_update_count_max),
            "MATCHING_CF_NATURAL_STABLE_COUNT": int(
                update.matching_cf_natural_stable_count
            ),
            "MATCHING_CF_MAX_UPDATES_HIT_COUNT": int(
                update.matching_cf_max_updates_hit_count
            ),
            "MATCHING_CF_MAX_UPDATES_HIT_RATE": float(
                update.matching_cf_max_updates_hit_rate
            ),
            "MATCHING_CF_UPDATE_COUNT_MEAN": float(
                update.matching_cf_update_count_mean
            ),
            "MATCHING_CF_UPDATE_COUNT_P95": float(
                update.matching_cf_update_count_p95
            ),
            "MATCHING_CF_UPDATE_COUNT_MAX": int(
                update.matching_cf_update_count_max
            ),
            "KKT_SOLVE_COUNT": int(update.kkt_solve_count),
            "KKT_CONVERGED_COUNT": int(update.kkt_converged_count),
            "KKT_CONVERGED_RATE": float(update.kkt_converged_rate),
            "KKT_MAX_ITER_HIT_COUNT": int(update.kkt_max_iter_hit_count),
            "KKT_MAX_ITER_HIT_RATE": float(update.kkt_max_iter_hit_rate),
            "KKT_FINITE_COUNT": int(update.kkt_finite_count),
            "KKT_FINITE_RATE": float(update.kkt_finite_rate),
            "KKT_ITERATION_MEAN": float(update.kkt_iteration_mean),
            "KKT_ITERATION_P95": float(update.kkt_iteration_p95),
            "KKT_ITERATION_MAX": int(update.kkt_iteration_max),
            "KKT_RESOURCE_RESIDUAL_MEAN": float(
                update.kkt_resource_residual_mean
            ),
            "KKT_RESOURCE_RESIDUAL_MAX": float(update.kkt_resource_residual_max),
            "KKT_STATIONARITY_RESIDUAL_MEAN": float(
                update.kkt_stationarity_residual_mean
            ),
            "KKT_STATIONARITY_RESIDUAL_MAX": float(
                update.kkt_stationarity_residual_max
            ),
            "KKT_COMPLEMENTARY_SLACKNESS_RESIDUAL_MEAN": float(
                update.kkt_complementary_slackness_residual_mean
            ),
            "KKT_COMPLEMENTARY_SLACKNESS_RESIDUAL_MAX": float(
                update.kkt_complementary_slackness_residual_max
            ),
            "KKT_CF_SOLVE_COUNT": int(update.matching_cf_kkt_solve_count),
            "KKT_CF_CONVERGED_COUNT": int(
                update.matching_cf_kkt_converged_count
            ),
            "KKT_CF_CONVERGED_RATE": float(
                update.matching_cf_kkt_converged_rate
            ),
            "KKT_CF_MAX_ITER_HIT_COUNT": int(
                update.matching_cf_kkt_max_iter_hit_count
            ),
            "KKT_CF_MAX_ITER_HIT_RATE": float(
                update.matching_cf_kkt_max_iter_hit_rate
            ),
            "KKT_CF_FINITE_COUNT": int(update.matching_cf_kkt_finite_count),
            "KKT_CF_FINITE_RATE": float(update.matching_cf_kkt_finite_rate),
            "KKT_CF_ITERATION_MEAN": float(
                update.matching_cf_kkt_iteration_mean
            ),
            "KKT_CF_ITERATION_P95": float(update.matching_cf_kkt_iteration_p95),
            "KKT_CF_ITERATION_MAX": int(update.matching_cf_kkt_iteration_max),
            "KKT_CF_RESOURCE_RESIDUAL_MEAN": float(
                update.matching_cf_kkt_resource_residual_mean
            ),
            "KKT_CF_RESOURCE_RESIDUAL_MAX": float(
                update.matching_cf_kkt_resource_residual_max
            ),
            "KKT_CF_STATIONARITY_RESIDUAL_MEAN": float(
                update.matching_cf_kkt_stationarity_residual_mean
            ),
            "KKT_CF_STATIONARITY_RESIDUAL_MAX": float(
                update.matching_cf_kkt_stationarity_residual_max
            ),
            "KKT_CF_COMPLEMENTARY_SLACKNESS_RESIDUAL_MEAN": float(
                update.matching_cf_kkt_complementary_slackness_residual_mean
            ),
            "KKT_CF_COMPLEMENTARY_SLACKNESS_RESIDUAL_MAX": float(
                update.matching_cf_kkt_complementary_slackness_residual_max
            ),
        }
    )
    return fields

def _critic_episode_metrics(
    old_values: list[float],
    raw_targets: list[float],
) -> tuple[float | None, float, float]:
    """将采集阶段的 Critic 价值与正式 GAE target 进行汇总比较。"""

    values = np.asarray(old_values, dtype=np.float64)
    targets = np.asarray(raw_targets, dtype=np.float64)
    if values.shape != targets.shape or values.ndim != 1 or not len(values):
        raise ValueError("critic episode metric arrays must be aligned and non-empty")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(targets)):
        raise FloatingPointError("critic episode metrics contain NaN/Inf")
    target_variance = float(np.var(targets))
    value_std = float(np.std(values))
    target_std = float(np.std(targets))
    if not all(np.isfinite(value) for value in (target_variance, value_std, target_std)):
        raise FloatingPointError("critic episode metric summary is non-finite")
    explained_variance: float | None
    if target_variance <= 1.0e-12:
        explained_variance = None
    else:
        explained_variance = float(
            1.0 - float(np.var(targets - values)) / target_variance
        )
        if not np.isfinite(explained_variance):
            raise FloatingPointError("critic explained variance is non-finite")
    return explained_variance, value_std, target_std

def _append_critic_diagnostics(
    path: Path,
    *,
    seed: int,
    epoch: int,
    chunk_origin_start: int,
    optimizer_steps: int,
    rollout: Any,
    update: Any,
    cache_count: int | None = None,
) -> None:
    """追加 PPO 更新中已计算的 Critic 原始目标值。"""

    transitions = rollout.transitions
    if not (
        len(transitions)
        == len(update.raw_advantages)
        == len(update.ppo_advantages)
        == len(update.raw_returns)
    ):
        raise AssertionError("critic diagnostics length mismatch")
    for name, values in (
        ("Cache", update.cache_q_component_diagnostics),
        ("UAV", update.uav_q_component_diagnostics),
    ):
        if values is not None and len(values) != len(transitions):
            raise AssertionError(f"{name} Q-component diagnostics length mismatch")
    for transition_index, (transition, raw_advantage, raw_return) in enumerate(zip(
        transitions,
        update.raw_advantages,
        update.raw_returns,
        strict=True,
    )):
        old_value = float(transition.old_value)
        target = float(raw_return)
        residual = old_value - target
        normalized_residual = residual / float(update.critic_return_scale)
        global_advantage = float(update.ppo_advantages[transition_index])
        values = (
            old_value,
            float(raw_advantage),
            global_advantage,
            target,
            update.bootstrap_value,
            residual,
        )
        if not all(np.isfinite(value) for value in values):
            raise FloatingPointError("critic diagnostics contain non-finite values")
        record: dict[str, Any] = {
            "record_type": "critic_transition",
            "seed": int(seed),
            "episode": int(epoch),
            "epoch": int(epoch),
            "chunk_origin_start": int(chunk_origin_start),
            "optimizer_steps": int(optimizer_steps),
            "origin": int(transition.origin_index),
            "slot": int(transition.slot),
            "done": bool(transition.done),
            "old_value": old_value,
            "raw_advantage": float(raw_advantage),
            "global_advantage": global_advantage,
            "advantage_source": "actual_PPO_advantage",
            "raw_return": target,
            "global_return_target": target,
            "value_target": target,
            "bootstrap_value": float(update.bootstrap_value),
            "value_residual": residual,
            "abs_value_residual": abs(residual),
            "critic_return_scale": float(update.critic_return_scale),
            "normalized_value_residual": normalized_residual,
        }
        if update.teacher_generation_executed:
            record.update(
                {
                    "critic_bootstrap_teacher_warmup_complete": bool(
                        update.critic_bootstrap_teacher_warmup_complete
                    ),
                    "critic_optimizer_steps_before_teacher": int(
                        update.critic_optimizer_steps_before_teacher or 0
                    ),
                    "critic_optimizer_steps_after_update": int(
                        update.critic_optimizer_steps_after_update
                    ),
                    "popart_mean_before_teacher": update.popart_mean_before_teacher,
                    "popart_std_before_teacher": update.popart_std_before_teacher,
                    "critic_parameter_digest_before_teacher": (
                        update.critic_parameter_digest_before_teacher
                    ),
                    "teacher_source": update.teacher_source,
                    "max_q_component_reconstruction_error": float(
                        update.max_q_component_reconstruction_error
                    ),
                }
            )
            if update.cache_q_component_diagnostics is not None:
                record["cache_q_components"] = update.cache_q_component_diagnostics[
                    transition_index
                ]
            if update.uav_q_component_diagnostics is not None:
                record["uav_q_components"] = update.uav_q_component_diagnostics[
                    transition_index
                ]
        if transition.uav_operation_actions is not None:
            if cache_count is None:
                raise AssertionError(
                    "cache_count is required for joint UAV action diagnostics"
                )
            if (
                transition.uav_operation_masks is None
                or transition.uav_candidate_counts is None
                or update.uav_branch_old_log_probs is None
                or update.uav_branch_new_log_probs is None
                or update.uav_branch_ratios is None
                or update.uav_branch_clipped is None
            ):
                raise AssertionError(
                    "joint UAV action diagnostics are incomplete"
                )
            operation_actions = np.asarray(
                transition.uav_operation_actions, dtype=np.int64
            ).reshape(-1)
            operation_masks = np.asarray(
                transition.uav_operation_masks, dtype=bool
            )
            candidate_counts = np.asarray(
                transition.uav_candidate_counts, dtype=np.int64
            ).reshape(-1)
            actual_flags = np.asarray(
                [
                    bool(action.get("actual_moved", False))
                    for action in transition.uav_conditional_actions
                ],
                dtype=bool,
            ).reshape(-1)
            uav_count = len(operation_actions)
            if not (
                operation_masks.shape == (uav_count, 2)
                and len(candidate_counts) == uav_count
                and len(actual_flags) == uav_count
                and np.all(operation_actions >= 0)
                and np.all(operation_actions < 2)
                and np.all(candidate_counts >= 0)
                and np.all(np.isfinite(operation_masks))
            ):
                raise AssertionError("joint UAV action diagnostic shapes are invalid")
            if int(np.sum(actual_flags)) != int(transition.actual_uav_movement_count):
                raise AssertionError(
                    "UAV actual movement flags disagree with rollout movement count"
                )
            branch_old = float(update.uav_branch_old_log_probs[transition_index])
            branch_new = float(update.uav_branch_new_log_probs[transition_index])
            branch_ratio = float(update.uav_branch_ratios[transition_index])
            branch_clipped = bool(update.uav_branch_clipped[transition_index])
            branch_values = (branch_old, branch_new, branch_ratio)
            if not all(np.isfinite(value) for value in branch_values):
                raise FloatingPointError(
                    "joint UAV branch action diagnostics are non-finite"
                )
            cache_action = np.asarray(
                transition.action[: int(cache_count)], dtype=np.int8
            ).reshape(-1)
            if len(cache_action) != int(cache_count) or not np.all(
                (cache_action == 0) | (cache_action == 1)
            ):
                raise AssertionError("selected Cache action is not binary")
            record.update(
                {
                    "selected_cache_action": [int(value) for value in cache_action],
                    "selected_uav_action": [
                        "MOVE" if int(value) == 1 else "STAY"
                        for value in operation_actions
                    ],
                    "actual_uav_move": [int(value) for value in actual_flags],
                    "old_uav_log_prob": branch_old,
                    "new_uav_log_prob": branch_new,
                    "uav_ratio": branch_ratio,
                    "uav_clipped": branch_clipped,
                    "uav_log_prob_scope": "joint_uav_branch_per_transition",
                    "feasible_uav_action_count": [
                        int(value) for value in operation_masks.sum(axis=1)
                    ],
                    "uav_action_mask": [
                        [int(value) for value in row]
                        for row in operation_masks
                    ],
                    "uav_move_candidate_count": [
                        int(value) for value in candidate_counts
                    ],
                    "reward": float(transition.reward),
                    "q_total": float(transition.actual_task_utility),
                    "movement_cost": float(transition.movement_cost),
                    "delay": float(transition.total_task_delay_s),
                    "energy": float(transition.total_vehicle_energy_j),
                }
            )
        _append_jsonl(
            path,
            record,
        )

def _branch_kl_update_metric_fields(
    update: Any, minibatch_early_stop_mode: str
) -> dict[str, Any]:
    """返回写入 update JSONL 的分支级 minibatch/epoch KL 诊断。"""

    if not update.kl_epoch_diagnostics:
        return {}
    epoch_fields = (
        "cache_without_uav_kl_trigger_count",
        "uav_only_kl_trigger_count",
        "both_kl_trigger_count",
        "neither_kl_trigger_count",
    )
    fields: dict[str, Any] = {
        "kl_minibatch_diagnostics": [
            dict(record) for record in update.kl_minibatch_diagnostics
        ],
        "kl_epoch_diagnostics": [
            dict(record) for record in update.kl_epoch_diagnostics
        ],
        **{
            field: int(
                sum(int(record[field]) for record in update.kl_epoch_diagnostics)
            )
            for field in epoch_fields
        },
        "cache_trigger_suppressed_uav_update": bool(
            any(
                record["cache_trigger_suppressed_uav_update"]
                for record in update.kl_epoch_diagnostics
            )
        ),
        "uav_trigger_suppressed_cache_update": bool(
            any(
                record["uav_trigger_suppressed_cache_update"]
                for record in update.kl_epoch_diagnostics
            )
        ),
        "CACHE_OPTIMIZER_STEPS_THIS_UPDATE": int(
            update.cache_optimizer_steps_this_update
        ),
        "UAV_OPERATION_OPTIMIZER_STEPS_THIS_UPDATE": int(
            update.uav_operation_optimizer_steps_this_update
        ),
        "UAV_CANDIDATE_OPTIMIZER_STEPS_THIS_UPDATE": int(
            update.uav_candidate_optimizer_steps_this_update
        ),
        "CACHE_MINIBATCHES_SUPPRESSED": int(update.cache_minibatches_suppressed),
        "UAV_OPERATION_MINIBATCHES_SUPPRESSED": int(
            update.uav_operation_minibatches_suppressed
        ),
        "UAV_CANDIDATE_MINIBATCHES_SUPPRESSED": int(
            update.uav_candidate_minibatches_suppressed
        ),
        "UAV_OPERATION_STOP_MINIBATCH": update.uav_operation_stop_minibatch,
        "UAV_CANDIDATE_STOP_MINIBATCH": update.uav_candidate_stop_minibatch,
        "CACHE_KL_MINIBATCH_EARLY_STOP_ACTIVE": bool(
            update.cache_kl_control == minibatch_early_stop_mode
        ),
        "uav_candidate_active_per_update_epoch": [
            bool(value) for value in update.uav_candidate_active_per_update_epoch
        ],
        "UAV_CANDIDATE_ONLY_STEPS_AFTER_OPERATION_STOP": int(
            update.uav_candidate_only_steps_after_operation_stop
        ),
    }
    if update.cache_active_per_update_epoch or update.uav_active_per_update_epoch:
        fields.update(
            {
                "kl_gate_mode": str(update.kl_gate_mode),
                "cache_active_per_update_epoch": [
                    bool(value) for value in update.cache_active_per_update_epoch
                ],
                "uav_active_per_update_epoch": [
                    bool(value) for value in update.uav_active_per_update_epoch
                ],
                "cache_stop_epoch": update.cache_stop_epoch,
                "uav_stop_epoch": update.uav_stop_epoch,
                "cache_suppressed_update_epochs": int(
                    update.cache_suppressed_update_epochs
                ),
                "uav_suppressed_update_epochs": int(
                    update.uav_suppressed_update_epochs
                ),
                "joint_kl_diagnostic": bool(update.joint_kl_diagnostic),
                "cache_independent_stop": bool(update.cache_independent_stop),
                "uav_independent_stop": bool(update.uav_independent_stop),
                "both_inactive_stop": bool(update.both_inactive_stop),
                "joint_kl_control_enabled": False,
            }
        )
    return fields


def append_dual_timescale_record(path: Path, record: dict[str, Any]) -> None:
    """追加新主线的 episode/update JSONL 记录。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _integer_histogram(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts, strict=True)}


def build_macro_transition_trace_arrays(
    episode_batches: Iterable[dict[str, Any]], *, gamma: float,
    target_mode: str = "one_step_smdp_td",
    lambda_return_lambda: float | None = None,
    expected_macro_frame_counts: dict[int, int] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Assemble and validate the minimal per-transition Macro trace."""

    batches = list(episode_batches)
    if not batches:
        raise ValueError("Macro transition trace requires at least one episode.")
    gamma_value = float(gamma)
    if not math.isfinite(gamma_value) or gamma_value < 0.0:
        raise ValueError("Macro transition trace gamma must be finite and non-negative.")
    target_mode_value = str(target_mode)
    if target_mode_value not in {
        "one_step_smdp_td",
        "two_step_smdp_td",
        "smdp_lambda_return",
    }:
        raise ValueError(f"Unsupported Macro transition trace target mode: {target_mode_value!r}.")
    lambda_value = None if lambda_return_lambda is None else float(lambda_return_lambda)
    if target_mode_value == "smdp_lambda_return":
        if lambda_value is None or not math.isfinite(lambda_value) or not 0.0 <= lambda_value <= 1.0:
            raise ValueError("SMDP lambda-return trace requires a finite lambda in [0, 1].")

    fields = (
        "observation",
        "next_observation",
        "macro_reward",
        "requested_duration",
        "actual_duration",
        "done",
        "old_value",
        "old_next_value",
        "critic_training_target",
        "realized_return_to_go",
        "macro_frame_index",
    )
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in fields}
    n2_trace_fields = (
        "n2_target",
        "n2_bootstrap_old_value",
        "n2_bootstrap_frame_index",
    )
    if target_mode_value == "two_step_smdp_td":
        chunks.update({name: [] for name in n2_trace_fields})
    lambda_trace_fields = (
        "lambda_return_target",
        "macro_critic_lambda_return_lambda",
    )
    if target_mode_value == "smdp_lambda_return":
        chunks.update({name: [] for name in lambda_trace_fields})
    chunks["episode_id"] = []
    chunks["episode_seed"] = []
    episode_transition_counts: dict[str, int] = {}
    seen_episodes: set[int] = set()
    episode_order: list[int] = []
    target_max_abs_error = 0.0
    return_max_abs_error = 0.0

    for batch in batches:
        episode_value = batch.get("episode_id", batch.get("episode"))
        if episode_value is None:
            raise ValueError("Macro transition trace episode_id is missing.")
        episode = int(episode_value)
        episode_seed = int(batch["episode_seed"])
        if episode in seen_episodes:
            raise ValueError(f"Macro transition trace repeats episode {episode}.")
        seen_episodes.add(episode)
        episode_order.append(episode)
        raw = {name: np.asarray(batch[name]) for name in fields}
        observation = np.asarray(raw["observation"], dtype=np.float32)
        next_observation = np.asarray(raw["next_observation"], dtype=np.float32)
        reward = np.asarray(raw["macro_reward"], dtype=np.float32).reshape(-1)
        requested_raw = np.asarray(raw["requested_duration"]).reshape(-1)
        duration_raw = np.asarray(raw["actual_duration"]).reshape(-1)
        if (
            not np.isfinite(requested_raw).all()
            or not np.isfinite(duration_raw).all()
            or np.any(requested_raw != np.floor(requested_raw))
            or np.any(duration_raw != np.floor(duration_raw))
        ):
            raise ValueError(f"Macro transition trace episode {episode} has non-finite/non-integer durations.")
        requested = requested_raw.astype(np.int64)
        duration = duration_raw.astype(np.int64)
        done = np.asarray(raw["done"], dtype=bool).reshape(-1)
        old_value = np.asarray(raw["old_value"], dtype=np.float32).reshape(-1)
        old_next_value = np.asarray(raw["old_next_value"], dtype=np.float32).reshape(-1)
        critic_target = np.asarray(raw["critic_training_target"], dtype=np.float32).reshape(-1)
        realized_return = np.asarray(raw["realized_return_to_go"], dtype=np.float32).reshape(-1)
        frame_index = np.asarray(raw["macro_frame_index"], dtype=np.int64).reshape(-1)
        if target_mode_value == "two_step_smdp_td":
            missing_n2_fields = [name for name in n2_trace_fields if name not in batch]
            if missing_n2_fields:
                raise ValueError(
                    f"Macro transition trace episode {episode} is missing N2 linkage fields: {missing_n2_fields}."
                )
            n2_target = np.asarray(batch["n2_target"], dtype=np.float32).reshape(-1)
            n2_bootstrap_old_value = np.asarray(
                batch["n2_bootstrap_old_value"], dtype=np.float32
            ).reshape(-1)
            n2_bootstrap_frame_index = np.asarray(
                batch["n2_bootstrap_frame_index"], dtype=np.int64
            ).reshape(-1)
        elif target_mode_value == "smdp_lambda_return":
            missing_lambda_fields = [name for name in lambda_trace_fields if name not in batch]
            if missing_lambda_fields:
                raise ValueError(
                    f"Macro transition trace episode {episode} is missing lambda-return fields: {missing_lambda_fields}."
                )
            lambda_return_target = np.asarray(
                batch["lambda_return_target"], dtype=np.float32
            ).reshape(-1)
            recorded_lambda = np.asarray(
                batch["macro_critic_lambda_return_lambda"], dtype=np.float64
            ).reshape(-1)
        count = int(reward.size)
        if count <= 0:
            raise ValueError(f"Macro transition trace episode {episode} is empty.")
        vector_fields = (requested, duration, done, old_value, old_next_value, critic_target, realized_return, frame_index)
        if target_mode_value == "two_step_smdp_td":
            vector_fields += (n2_target, n2_bootstrap_old_value, n2_bootstrap_frame_index)
        elif target_mode_value == "smdp_lambda_return":
            vector_fields += (lambda_return_target, recorded_lambda)
        if any(value.shape != (count,) for value in vector_fields):
            raise ValueError(f"Macro transition trace episode {episode} has misaligned fields.")
        if observation.shape != (count, 400) or next_observation.shape != (count, 400):
            raise ValueError(f"Macro transition trace episode {episode} must have 400D observations.")
        floats = (observation, next_observation, reward, old_value, old_next_value, critic_target, realized_return)
        if target_mode_value == "two_step_smdp_td":
            floats += (n2_target, n2_bootstrap_old_value)
        elif target_mode_value == "smdp_lambda_return":
            floats += (lambda_return_target, recorded_lambda)
        if any(not np.isfinite(value).all() for value in floats):
            raise ValueError(f"Macro transition trace episode {episode} contains NaN/Inf.")
        if np.any(requested <= 0) or np.any(duration <= 0) or np.any(duration > requested):
            raise ValueError(f"Macro transition trace episode {episode} has invalid durations.")
        if not np.array_equal(frame_index, np.arange(count, dtype=np.int64)):
            raise ValueError(f"Macro frame indexes are not continuous in episode {episode}.")
        if not bool(done[-1]) or bool(np.any(done[:-1])):
            raise ValueError(f"Terminal flag must occur only on the final transition of episode {episode}.")
        if float(old_next_value[-1]) != 0.0:
            raise ValueError(f"Terminal old_next_value must be zero in episode {episode}.")
        if np.any(next_observation[-1] != 0.0):
            raise ValueError(f"Terminal next_observation must be zero in episode {episode}.")
        if count > 1 and not np.allclose(
            next_observation[:-1], observation[1:], rtol=0.0, atol=1.0e-6
        ):
            raise ValueError(f"Macro next-observation linkage breaks within episode {episode}.")

        if target_mode_value == "two_step_smdp_td":
            expected_bootstrap_frame_index = np.full(count, -1, dtype=np.int64)
            expected_bootstrap_old_value = np.zeros(count, dtype=np.float32)
            for index in range(count):
                if done[index] or done[index + 1]:
                    continue
                expected_bootstrap_frame_index[index] = frame_index[index + 2]
                expected_bootstrap_old_value[index] = old_value[index + 2]
            if not np.array_equal(
                n2_bootstrap_frame_index, expected_bootstrap_frame_index
            ) or not np.array_equal(n2_bootstrap_old_value, expected_bootstrap_old_value):
                raise ValueError(
                    f"N2 frozen old-value k+2 linkage breaks in episode {episode}."
                )
        elif target_mode_value == "smdp_lambda_return":
            if not np.array_equal(
                recorded_lambda,
                np.full(count, lambda_value, dtype=np.float64),
            ):
                raise ValueError(
                    f"Critic lambda-return lambda changes within episode {episode}."
                )

        if target_mode_value == "one_step_smdp_td":
            target_expected = (
                reward + np.power(gamma_value, duration) * old_next_value * (~done)
            ).astype(np.float32)
        elif target_mode_value == "two_step_smdp_td":
            target_expected = build_macro_n2_target(
                reward, old_value, duration, done, gamma_value
            )
        else:
            target_expected = build_macro_smdp_lambda_return_target(
                reward,
                old_next_value,
                duration,
                done,
                gamma_value,
                lambda_value,
            )
        # The shared builder keeps all arithmetic in float64. Training then
        # casts the target to the network's float32 dtype, so compare the trace
        # against that same cast rather than counting dtype quantization as a
        # target-reconstruction error.
        target_expected_for_training = np.asarray(target_expected, dtype=np.float32)
        target_outputs = [critic_target]
        if target_mode_value == "two_step_smdp_td":
            target_outputs.append(n2_target)
        elif target_mode_value == "smdp_lambda_return":
            target_outputs.append(lambda_return_target)
        target_error = max(
            float(
                np.max(
                    np.abs(
                        target_expected_for_training.astype(np.float64)
                        - target_output.astype(np.float64)
                    )
                )
            )
            for target_output in target_outputs
        )
        if target_error > 1.0e-6:
            raise ValueError(
                f"{target_mode_value} trace reconstruction error {target_error} exceeds 1e-6 in episode {episode}."
            )
        terminal_target = (
            n2_target
            if target_mode_value == "two_step_smdp_td"
            else lambda_return_target
            if target_mode_value == "smdp_lambda_return"
            else critic_target
        )
        terminal_target_error = float(np.max(np.abs(terminal_target[done] - reward[done])))
        if terminal_target_error > 1.0e-6:
            raise ValueError(
                f"Terminal {target_mode_value} target differs from reward in episode {episode}."
            )

        return_expected = np.zeros(count, dtype=np.float64)
        running_return = 0.0
        reward64 = reward.astype(np.float64)
        for index in range(count - 1, -1, -1):
            if done[index]:
                running_return = 0.0
            running_return = float(
                reward64[index]
                + gamma_value ** int(duration[index]) * running_return
            )
            return_expected[index] = running_return
        return_error = float(
            np.max(
                np.abs(
                    return_expected.astype(np.float32).astype(np.float64)
                    - realized_return.astype(np.float64)
                )
            )
        )
        if return_error > 1.0e-6:
            raise ValueError(
                f"Realized return-to-go reconstruction error {return_error} exceeds 1e-6 in episode {episode}."
            )
        target_max_abs_error = max(target_max_abs_error, target_error)
        return_max_abs_error = max(return_max_abs_error, return_error)
        episode_transition_counts[f"E{episode}"] = count
        chunks["episode_id"].append(np.full(count, episode, dtype=np.int64))
        chunks["episode_seed"].append(np.full(count, episode_seed, dtype=np.int64))
        chunks["observation"].append(observation)
        chunks["next_observation"].append(next_observation)
        chunks["macro_reward"].append(reward)
        chunks["requested_duration"].append(requested)
        chunks["actual_duration"].append(duration)
        chunks["done"].append(done)
        chunks["old_value"].append(old_value)
        chunks["old_next_value"].append(old_next_value)
        chunks["critic_training_target"].append(critic_target)
        chunks["realized_return_to_go"].append(realized_return)
        chunks["macro_frame_index"].append(frame_index)
        if target_mode_value == "two_step_smdp_td":
            chunks["n2_target"].append(n2_target)
            chunks["n2_bootstrap_old_value"].append(n2_bootstrap_old_value)
            chunks["n2_bootstrap_frame_index"].append(n2_bootstrap_frame_index)
        elif target_mode_value == "smdp_lambda_return":
            chunks["lambda_return_target"].append(lambda_return_target)
            chunks["macro_critic_lambda_return_lambda"].append(recorded_lambda)

    if episode_order != sorted(episode_order):
        raise ValueError("Macro transition trace episodes must be ordered and unique.")
    if expected_macro_frame_counts is not None:
        expected = {int(key): int(value) for key, value in expected_macro_frame_counts.items()}
        actual = {int(key[1:]): int(value) for key, value in episode_transition_counts.items()}
        if actual != expected:
            raise ValueError(
                f"Macro transition trace frame counts disagree with episode metrics: {actual} != {expected}."
            )

    arrays = {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}
    arrays["gamma"] = np.asarray(gamma_value, dtype=np.float64)
    requested = arrays["requested_duration"]
    duration = arrays["actual_duration"]
    pair_histogram: dict[str, int] = {}
    tail_histogram: dict[str, int] = {}
    for requested_value, actual_value in zip(requested, duration, strict=True):
        pair_key = f"{int(requested_value)}->{int(actual_value)}"
        pair_histogram[pair_key] = pair_histogram.get(pair_key, 0) + 1
        if int(actual_value) < int(requested_value):
            tail_histogram[pair_key] = tail_histogram.get(pair_key, 0) + 1
    summary = {
        "macro_transition_trace_path": None,
        "transition_count": int(len(arrays["episode_id"])),
        "episode_transition_counts": episode_transition_counts,
        "requested_duration_histogram": _integer_histogram(requested),
        "actual_duration_histogram": _integer_histogram(duration),
        "requested_actual_duration_pair_histogram": pair_histogram,
        "tail_truncated_duration_counts": tail_histogram,
        "critic_training_target_mode": target_mode_value,
        "macro_critic_lambda_return_lambda": lambda_value,
        "critic_target_recompute_max_abs_error": target_max_abs_error,
        "realized_return_recompute_max_abs_error": return_max_abs_error,
        "macro_transition_trace_fields": list(arrays),
        "reporting_only_change": True,
    }
    return arrays, summary


def write_macro_transition_trace_artifact(
    run_dir: Path,
    episode_batches: Iterable[dict[str, Any]],
    *,
    gamma: float,
    target_mode: str = "one_step_smdp_td",
    lambda_return_lambda: float | None = None,
) -> dict[str, Any]:
    """Write a compressed validated Macro trace and add its index to run reports."""

    destination = Path(run_dir)
    episode_records, _ = _read_jsonl(destination / EPISODE_METRICS_FILENAME)
    expected_counts = {
        int(record["episode"]): int(record["macro_frame_count"])
        for record in episode_records
        if record.get("record_type") == "episode"
    }
    arrays, summary = build_macro_transition_trace_arrays(
        episode_batches,
        gamma=gamma,
        target_mode=target_mode,
        lambda_return_lambda=lambda_return_lambda,
        expected_macro_frame_counts=expected_counts,
    )
    trace_path = destination / MACRO_TRANSITION_TRACE_FILENAME
    try:
        with trace_path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
    except Exception:
        if trace_path.exists():
            trace_path.unlink()
        raise
    summary["macro_transition_trace_path"] = str(trace_path.resolve())
    summary["macro_transition_trace_format"] = "npz"
    summary["trace_format_version"] = (
        4
        if target_mode == "smdp_lambda_return"
        else 3
        if target_mode == "two_step_smdp_td"
        else 2
    )
    for filename in (RUN_MANIFEST_FILENAME, RUN_SUMMARY_FILENAME):
        report_path = destination / filename
        payload = _read_json(report_path)
        payload.update(summary)
        _write_json(report_path, payload)
    return summary


def write_dual_timescale_run_manifest(
    path: Path,
    *,
    config_path: str | Path,
    config_raw_sha256: str,
    config_semantic_sha256: str,
    seed: int,
    trainer: Any,
    context: Any,
    target_episodes: int,
    episode_pool: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """记录双时间尺度 checkpoint、动作和 PPO 配置合同。"""

    manifest = {
        "experiment_type": "macro_cache_smdp_micro_uav_ppo",
        "config_path": str(Path(config_path).resolve()),
        "config_raw_sha256": str(config_raw_sha256),
        "config_semantic_sha256": str(config_semantic_sha256),
        "seed": int(seed),
        "target_episodes": int(target_episodes),
        "episode_pool": None if episode_pool is None else str(Path(episode_pool).resolve()),
        "policy_schema": trainer.policy.architecture_id,
        "checkpoint_version": int(trainer.checkpoint_version),
        "macro_actor_advantage_mode": "duration_aware_smdp_gae",
        "macro_critic_target_mode": str(trainer.macro_critic_target_mode),
        "macro_critic_lambda_return_lambda": trainer.macro_critic_lambda_return_lambda,
        "long_value_enabled": bool(trainer.long_value_enabled),
        "long_value_target": str(trainer.long_value_target),
        "long_value_architecture": "observation_dim->128->128->1 (SiLU)",
        "long_value_input_transform": "raw_macro_observation",
        "long_value_optimizer": "Adam",
        "long_value_learning_rate": float(trainer.learning_rates["long_value"]),
        "long_value_optimizer_epsilon": 1.0e-5,
        "long_value_max_grad_norm": float(trainer.long_value_max_grad_norm),
        "long_value_steps_per_update": int(trainer.long_value_steps_per_update),
        "architecture_version": int(trainer.policy.architecture_version),
        "observation_dim": int(context.observation_spec.observation_dim),
        "macro_cache_candidate_feature_dim": int(trainer.policy.macro_cache_feature_dim),
        "micro_uav_candidate_feature_dim": int(trainer.policy.micro_uav_feature_dim),
        "prediction_horizon": int(trainer.policy.prediction_horizon),
        "macro_duration_candidates": list(trainer.policy.duration_candidates),
        "actor_learning_rate": float(trainer.learning_rates["actor"]),
        "macro_critic_learning_rate": float(trainer.learning_rates["macro_critic"]),
        "micro_critic_learning_rate": float(trainer.learning_rates["micro_critic"]),
        "critic_input_transform": "2*x-1",
        "critic_hidden_activation": "SiLU",
        "macro_cache_action": "(Cache configuration C_k, duration L_k)",
        "cache_candidate_zero": "current actual Cache configuration (KEEP)",
        "uav_candidate_zero": "current UAV position (STAY)",
        "uav_joint_policy": "UAV0 then UAV1 conditional on selected UAV0 position",
        "mask_rollout_contract": "behavior masks and candidate descriptors are saved per action",
        "macro_full_batch": True,
        "macro_actor_update_epochs": int(trainer.config["update_epochs"]),
        "macro_critic_steps_per_update": int(
            trainer.config.get("macro_critic_steps_per_update", 50)
        ),
        "macro_critic_replay_history_episodes": int(
            trainer.config.get("macro_critic_replay_history_episodes", 4)
        ),
        "micro_macro_buffers_separate": True,
        "exact_q_training_enabled": False,
        "critic_mc_warmup_required": False,
        "prediction_model_frozen": True,
        "matching": "Transfer-Exchange (unchanged)",
        "kkt": "existing continuous resource allocation (unchanged)",
        "physical_system_model_changed": False,
        "configuration_control_mode": context.configuration_control_mode,
        "episode_slots": int(context.episode_slots),
        "origin_start": int(context.train_origin_start),
        "origin_end": int(context.train_origin_end),
        "terminal_origin": int(context.terminal_origin),
        "training_counters": dict(trainer.training_counters),
        "resume": bool(resume),
    }
    _write_json(Path(path), manifest)
    return manifest


def append_dual_timescale_slot_metrics(
    path: Path,
    *,
    episode: int,
    transitions: Iterable[Any],
) -> int:
    """记录新主线的实际系统 slot 结果，不输出旧 Trigger/Operation 指标。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        for transition in transitions:
            expected_reward = (
                float(transition.actual_task_utility)
                - float(transition.cache_cost)
                - float(transition.movement_cost)
            )
            if not math.isclose(
                float(transition.reward), expected_reward, rel_tol=1e-12, abs_tol=1e-9
            ):
                raise ValueError("slot utility 与任务效用、Cache cost、移动 cost 不一致。")
            record = {
                "record_type": "slot",
                "episode": int(episode),
                "origin": int(transition.origin_index),
                "slot": int(transition.slot),
                "G": float(transition.reward),
                "task_utility": float(transition.actual_task_utility),
                "delay": float(transition.total_task_delay_s),
                "energy": float(transition.total_vehicle_energy_j),
                "cache_cost": float(transition.cache_cost),
                "movement_cost": float(transition.movement_cost),
                "actual_cache_update_count": int(transition.actual_cache_update_count),
                "uav0_stay": bool(int(transition.actions[0]) == 0),
                "uav1_stay": bool(int(transition.actions[1]) == 0),
                "uav0_move": bool(int(transition.actions[0]) != 0),
                "uav1_move": bool(int(transition.actions[1]) != 0),
                "matching_calls": 1,
                "kkt_calls": int(
                    transition.matching_summary.get("KKT_SOLVER_CALL_COUNT", 0)
                ),
                "slot_runtime_seconds": float(transition.slot_runtime_seconds),
            }
            json.dump(record, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write("\n")
            count += 1
    return count
