"""Small, read-only state traces for adaptive cache/UAV visualizations.

The trace is deliberately built from the state returned by the environment
after a slot has been executed.  It does not sample a policy, inspect a
trigger as a proxy for a state change, or retain the complete cache content.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


STATE_TRACE_FILENAME = "state_trace.jsonl"
TRACE_FIELDS = (
    "epoch",
    "origin",
    "global_training_slot",
    "cache_updates",
    "cache_update_counts",
    "uav_positions",
)


def _position_dict(
    positions: Mapping[str, Any],
    uav_ids: tuple[str, ...],
    *,
    field: str,
) -> dict[str, tuple[float, float]]:
    if set(positions) != set(uav_ids):
        raise ValueError(
            f"{field} UAV IDs do not match the configured action spec: "
            f"expected {list(uav_ids)}, got {list(positions)}"
        )
    copied: dict[str, tuple[float, float]] = {}
    for uav_id in uav_ids:
        value = np.asarray(positions[uav_id], dtype=np.float64).reshape(-1)
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{field}[{uav_id}] must be a finite x/y pair")
        copied[uav_id] = (float(value[0]), float(value[1]))
    return copied


def build_state_trace_records(
    context: Any,
    transitions: Iterable[Any],
    *,
    initial_cache: Any,
    initial_uav: Mapping[str, Any],
    epoch: int | None = None,
    origin_start: int | None = None,
    global_training_slot_start: int = 0,
) -> list[dict[str, Any]]:
    """Build one compact record per completed transition.

    ``transition.next_cache_state`` and ``transition.next_uav_positions`` are
    the actual post-application environment states.  Cache events are derived
    by comparing each of those states with the preceding state, node by node.
    ``cache_update_counts[node_id]`` is exactly
    ``sum_f |C_after[node_id, f] - C_before[node_id, f]|``: the number of
    changed cache-state bits for that node.  It is included as a useful
    optional enhancement for plots; it is not a replacement-operation count.
    """

    transition_list = list(transitions)
    if not transition_list:
        raise ValueError("state trace requires at least one transition")

    action_spec = context.action_spec
    cache_node_ids = tuple(str(node_id) for node_id in action_spec.cache_node_ids)
    uav_ids = tuple(str(uav_id) for uav_id in action_spec.uav_ids)
    if not cache_node_ids or not uav_ids:
        raise ValueError("state trace requires dynamic cache nodes and UAVs")

    previous_cache = np.asarray(initial_cache, dtype=np.int8).copy()
    if previous_cache.ndim != 2 or previous_cache.shape[0] != len(cache_node_ids):
        raise ValueError(
            "initial_cache shape is incompatible with the configured cache nodes"
        )
    if not np.all(np.isin(previous_cache, [0, 1])):
        raise ValueError("initial_cache must be a binary cache-state matrix")
    _position_dict(initial_uav, uav_ids, field="initial_uav")

    first_origin = int(transition_list[0].origin_index)
    active_origin_start = first_origin if origin_start is None else int(origin_start)
    epoch_value = None if epoch is None else int(epoch)
    global_start = int(global_training_slot_start)
    records: list[dict[str, Any]] = []
    seen_origins: set[tuple[int | None, int]] = set()

    for transition in transition_list:
        origin = int(transition.origin_index)
        key = (epoch_value, origin)
        if key in seen_origins:
            raise ValueError(f"duplicate state trace transition: {key}")
        seen_origins.add(key)

        next_cache = np.asarray(transition.next_cache_state, dtype=np.int8).copy()
        if next_cache.shape != previous_cache.shape:
            raise ValueError(
                "transition.next_cache_state shape changed within the state trace"
            )
        if not np.all(np.isin(next_cache, [0, 1])):
            raise ValueError("transition.next_cache_state must be binary")
        changed_items = np.not_equal(previous_cache, next_cache)
        cache_updates = {
            node_id: int(np.any(changed_items[node_index]))
            for node_index, node_id in enumerate(cache_node_ids)
        }
        cache_update_counts = {
            node_id: int(np.count_nonzero(changed_items[node_index]))
            for node_index, node_id in enumerate(cache_node_ids)
        }

        next_uav = _position_dict(
            transition.next_uav_positions,
            uav_ids,
            field="transition.next_uav_positions",
        )
        # Dict insertion order follows the action spec, keeping serialized
        # traces and plots stable without hard-coding node names or counts.
        record = {
            "epoch": epoch_value,
            "origin": origin,
            "global_training_slot": global_start + origin - active_origin_start,
            "cache_updates": cache_updates,
            "cache_update_counts": cache_update_counts,
            "uav_positions": {
                uav_id: {"x": next_uav[uav_id][0], "y": next_uav[uav_id][1]}
                for uav_id in uav_ids
            },
        }
        records.append(record)
        previous_cache = next_cache

    return records


def write_state_trace(
    path: str | Path,
    context: Any,
    transitions: Iterable[Any],
    *,
    initial_cache: Any,
    initial_uav: Mapping[str, Any],
    epoch: int | None = None,
    origin_start: int | None = None,
    global_training_slot_start: int = 0,
) -> Path:
    """Serialize compact post-slot state results as JSON Lines."""

    output_path = Path(path)
    records = build_state_trace_records(
        context,
        transitions,
        initial_cache=initial_cache,
        initial_uav=initial_uav,
        epoch=epoch,
        origin_start=origin_start,
        global_training_slot_start=global_training_slot_start,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    return output_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"state trace does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"state trace contains a blank line at {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"state trace contains invalid JSON at line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"state trace line {line_number} is not an object")
            missing = {"origin", "cache_updates", "uav_positions"} - set(record)
            if missing:
                raise ValueError(
                    f"state trace line {line_number} missing fields: {sorted(missing)}"
                )
            try:
                origin_float = float(record["origin"])
                origin = int(record["origin"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"state trace line {line_number} has an invalid origin"
                ) from exc
            if (
                isinstance(record["origin"], bool)
                or not math.isfinite(origin_float)
                or origin_float != origin
            ):
                raise ValueError(
                    f"state trace line {line_number} has an invalid origin"
                )

            epoch_raw = record.get("epoch")
            epoch: int | None
            if epoch_raw is None:
                epoch = None
            else:
                try:
                    epoch_float = float(epoch_raw)
                    epoch = int(epoch_raw)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"state trace line {line_number} has an invalid epoch"
                    ) from exc
                if (
                    isinstance(epoch_raw, bool)
                    or not math.isfinite(epoch_float)
                    or epoch_float != epoch
                ):
                    raise ValueError(
                        f"state trace line {line_number} has an invalid epoch"
                    )

            updates = record["cache_updates"]
            if not isinstance(updates, dict):
                raise ValueError(
                    f"state trace line {line_number} cache_updates must be an object"
                )
            parsed_updates: dict[str, int] = {}
            for node_id, raw_value in updates.items():
                try:
                    value_float = float(raw_value)
                    value = int(raw_value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"state trace line {line_number} has an invalid cache update"
                    ) from exc
                if (
                    isinstance(raw_value, bool)
                    or not math.isfinite(value_float)
                    or value_float != value
                    or value not in (0, 1)
                ):
                    raise ValueError(
                        f"state trace line {line_number} has an invalid cache update"
                    )
                parsed_updates[str(node_id)] = value

            raw_counts = record.get("cache_update_counts")
            parsed_counts: dict[str, int] | None = None
            if raw_counts is not None:
                if not isinstance(raw_counts, dict):
                    raise ValueError(
                        f"state trace line {line_number} cache_update_counts must be an object"
                    )
                parsed_counts = {}
                for node_id, raw_value in raw_counts.items():
                    try:
                        value_float = float(raw_value)
                        value = int(raw_value)
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise ValueError(
                            f"state trace line {line_number} has an invalid cache update count"
                        ) from exc
                    if (
                        isinstance(raw_value, bool)
                        or not math.isfinite(value_float)
                        or value_float != value
                        or value < 0
                    ):
                        raise ValueError(
                            f"state trace line {line_number} has an invalid cache update count"
                        )
                    parsed_counts[str(node_id)] = value

            positions = record["uav_positions"]
            if not isinstance(positions, dict):
                raise ValueError(
                    f"state trace line {line_number} uav_positions must be an object"
                )
            parsed_positions: dict[str, dict[str, float]] = {}
            for uav_id, raw_position in positions.items():
                if not isinstance(raw_position, dict) or not {"x", "y"} <= set(raw_position):
                    raise ValueError(
                        f"state trace line {line_number} has an invalid UAV position"
                    )
                try:
                    x = float(raw_position["x"])
                    y = float(raw_position["y"])
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"state trace line {line_number} has an invalid UAV position"
                    ) from exc
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError(
                        f"state trace line {line_number} has a non-finite UAV position"
                    )
                parsed_positions[str(uav_id)] = {"x": x, "y": y}

            parsed_record = dict(record)
            parsed_record["origin"] = origin
            parsed_record["epoch"] = epoch
            parsed_record["cache_updates"] = parsed_updates
            if parsed_counts is not None:
                parsed_record["cache_update_counts"] = parsed_counts
            parsed_record["uav_positions"] = parsed_positions
            if "global_training_slot" in parsed_record:
                try:
                    global_float = float(parsed_record["global_training_slot"])
                    global_slot = int(parsed_record["global_training_slot"])
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"state trace line {line_number} has an invalid global_training_slot"
                    ) from exc
                if (
                    isinstance(parsed_record["global_training_slot"], bool)
                    or not math.isfinite(global_float)
                    or global_float != global_slot
                ):
                    raise ValueError(
                        f"state trace line {line_number} has an invalid global_training_slot"
                    )
                parsed_record["global_training_slot"] = global_slot
            records.append(parsed_record)
    if not records:
        raise ValueError(f"state trace is empty: {path}")
    return records


def read_state_trace(
    path: str | Path,
    *,
    epoch: int | None = None,
) -> list[dict[str, Any]]:
    """Read a trace and optionally select one numeric epoch."""

    records = _read_jsonl(Path(path))
    if epoch is not None:
        selected = [record for record in records if record.get("epoch") == int(epoch)]
        if not selected:
            available = sorted({record.get("epoch") for record in records}, key=str)
            raise ValueError(
                f"epoch {int(epoch)} is not present in {path}; available epochs: {available}"
            )
        records = selected
    records.sort(
        key=lambda record: (
            record.get("epoch") is not None,
            record.get("epoch") if record.get("epoch") is not None else -1,
            int(record["origin"]),
        )
    )
    return records


def trace_cache_node_ids(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return cache IDs in first-seen serialized order."""

    node_ids: list[str] = []
    seen: set[str] = set()
    for record in records:
        for node_id in record["cache_updates"]:
            node_id_s = str(node_id)
            if node_id_s not in seen:
                seen.add(node_id_s)
                node_ids.append(node_id_s)
    return node_ids


def trace_uav_ids(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return UAV IDs in first-seen serialized order."""

    uav_ids: list[str] = []
    seen: set[str] = set()
    for record in records:
        for uav_id in record["uav_positions"]:
            uav_id_s = str(uav_id)
            if uav_id_s not in seen:
                seen.add(uav_id_s)
                uav_ids.append(uav_id_s)
    return uav_ids
