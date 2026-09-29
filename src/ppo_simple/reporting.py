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

SLOT_METRICS_FILENAME = "slot_metrics.jsonl"

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
