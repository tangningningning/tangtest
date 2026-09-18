"""Minimal training metrics logging and plotting for PPO runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SLOTS_PER_EPOCH = 240
SLOT_METRICS_FILENAME = "slot_metrics.jsonl"
EPISODE_METRICS_FILENAME = "episode_metrics.jsonl"
TRAINING_EPOCH_RETURN_PLOT_FILENAME = "training_epoch_return.png"
# Kept only as a legacy artifact name for fresh-run contamination checks.
TRAINING_SLOT_REWARD_PLOT_FILENAME = "training_slot_reward.png"
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
    """Read active episode slots/origin start, with the legacy 240 fallback."""

    manifest_path = Path(run_dir) / "run_manifest.json"
    if not manifest_path.exists():
        return SLOTS_PER_EPOCH, 0
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read plotting protocol manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"plotting protocol manifest must be an object: {manifest_path}")
    raw_slots = manifest.get(
        "episode_slots",
        manifest.get("origins_per_epoch", SLOTS_PER_EPOCH),
    )
    raw_start = manifest.get(
        "origin_start",
        manifest.get("train_origin_start", 0),
    )
    raw_end = manifest.get("origin_end", manifest.get("train_origin_end"))
    try:
        episode_slots = int(raw_slots)
        origin_start = int(raw_start)
        origin_end = (
            origin_start + episode_slots - 1
            if raw_end is None
            else int(raw_end)
        )
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


def _finite_float(value: Any, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"slot_metrics {field} must be numeric") from exc
    if not math.isfinite(converted):
        raise ValueError(f"slot_metrics {field} must be finite")
    return converted


def _nonnegative_int(value: Any, field: str) -> int:
    converted = _finite_float(value, field)
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
            reward = _finite_float(record["reward"], "reward")
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
                        "q": _finite_float(record["q"], "q"),
                        "cache_cost": _finite_float(record["cache_cost"], "cache_cost"),
                        "movement_cost": _finite_float(
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
    """Append or safely complete one epoch of raw transition rewards.

    Existing records are validated against the current episode before missing
    origins are appended.  A complete epoch is therefore idempotent, while a
    partial epoch is safely filled without creating duplicate keys.
    """

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
    episode_return_f = _finite_float(episode_return, "episode_return")
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
    """Read one finite raw return for every contiguous completed Epoch."""

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
    """Render the only supported PPO training plot from Epoch returns."""

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
