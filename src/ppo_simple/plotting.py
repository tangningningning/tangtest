"""Minimal training-slot reward logging and plotting for PPO runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SLOTS_PER_EPOCH = 240
SLOT_METRICS_FILENAME = "slot_metrics.jsonl"
TRAINING_SLOT_REWARD_PLOT_FILENAME = "training_slot_reward.png"
MA_WINDOW = 240


def _finite_float(value: Any, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"slot_metrics {field} must be numeric") from exc
    if not math.isfinite(converted):
        raise ValueError(f"slot_metrics {field} must be finite")
    return converted


def _read_slot_metrics(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
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
            if not 0 <= origin < SLOTS_PER_EPOCH:
                raise ValueError(
                    f"slot_metrics origin must be in [0,{SLOTS_PER_EPOCH - 1}] "
                    f"at line {line_number}"
                )
            expected_global_slot = (epoch - 1) * SLOTS_PER_EPOCH + origin
            if global_slot != expected_global_slot:
                raise ValueError(
                    "slot_metrics global_training_slot mismatch at "
                    f"line {line_number}: {global_slot} != {expected_global_slot}"
                )
            reward = _finite_float(record["reward"], "reward")
            key = (epoch, origin)
            if key in records:
                raise ValueError(
                    f"slot_metrics duplicate (epoch, origin)={key}"
                )
            records[key] = {
                "epoch": epoch,
                "origin": origin,
                "global_training_slot": global_slot,
                "reward": reward,
            }
    return records


def append_training_slot_metrics(
    path: Path,
    *,
    epoch: int,
    rewards: Iterable[float],
    episode_return: float,
) -> int:
    """Append or safely complete one epoch of raw transition rewards.

    Existing records are validated against the current episode before missing
    origins are appended.  A complete epoch is therefore idempotent, while a
    partial epoch is safely filled without creating duplicate keys.
    """

    epoch_i = int(epoch)
    if epoch_i < 1:
        raise ValueError("slot_metrics epoch must be >=1")
    reward_array = np.asarray(list(rewards), dtype=np.float64).reshape(-1)
    if reward_array.shape != (SLOTS_PER_EPOCH,):
        raise ValueError(
            "slot_metrics requires one complete 240-origin episode; "
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

    records = _read_slot_metrics(path)
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
        if existing_epoch < epoch_i and count != SLOTS_PER_EPOCH:
            raise ValueError(
                f"slot_metrics epoch {existing_epoch} is incomplete ({count}/"
                f"{SLOTS_PER_EPOCH} records)"
            )
    to_append: list[dict[str, Any]] = []
    for origin, reward in enumerate(reward_array.tolist()):
        key = (epoch_i, origin)
        existing = records.get(key)
        if existing is not None:
            if not bool(np.isclose(existing["reward"], reward, rtol=0.0, atol=1.0e-12)):
                raise ValueError(
                    "existing slot reward conflicts with the completed episode at "
                    f"(epoch, origin)={key}"
                )
            continue
        to_append.append(
            {
                "epoch": epoch_i,
                "origin": origin,
                "global_training_slot": (epoch_i - 1) * SLOTS_PER_EPOCH + origin,
                "reward": float(reward),
            }
        )
    for record in to_append:
        _append_jsonl(path, record)
    return len(to_append)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


def update_training_slot_reward_plot(run_dir: Path) -> Path:
    """Render the only supported PPO training plot from raw slot metrics."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    run_path = Path(run_dir)
    records = _read_slot_metrics(run_path / SLOT_METRICS_FILENAME)
    if not records:
        raise ValueError("slot_metrics.jsonl is empty or does not exist")

    ordered = sorted(records.values(), key=lambda item: item["global_training_slot"])
    global_slots = np.asarray(
        [item["global_training_slot"] for item in ordered], dtype=np.int64
    )
    rewards = np.asarray([item["reward"] for item in ordered], dtype=np.float64)
    if not np.all(np.diff(global_slots) == 1):
        raise ValueError("slot_metrics global_training_slot sequence is not contiguous")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("slot_metrics rewards contain NaN/Inf")

    cumulative = np.cumsum(rewards, dtype=np.float64)
    rolling = np.empty_like(rewards)
    for index in range(rewards.size):
        start = max(0, index + 1 - MA_WINDOW)
        total = cumulative[index] - (cumulative[start - 1] if start else 0.0)
        rolling[index] = total / float(index + 1 - start)

    figure, axis = plt.subplots(figsize=(12, 6))
    try:
        axis.plot(
            global_slots,
            rewards,
            label="Raw Slot Reward",
            linewidth=0.7,
            alpha=0.55,
        )
        axis.plot(
            global_slots,
            rolling,
            label="MA240",
            linewidth=1.8,
        )
        final_slot_exclusive = int(global_slots[-1]) + 1
        for boundary in range(
            SLOTS_PER_EPOCH,
            final_slot_exclusive + 1,
            SLOTS_PER_EPOCH,
        ):
            axis.axvline(boundary, linestyle="--", linewidth=0.7, alpha=0.35)
        axis.set_xlabel("Global Training Slot")
        axis.set_ylabel("Reward G(t)")
        axis.set_title("PPO Training Reward per Slot")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        output_path = run_path / "plots" / TRAINING_SLOT_REWARD_PLOT_FILENAME
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180)
    finally:
        plt.close(figure)
    return output_path
