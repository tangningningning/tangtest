#!/usr/bin/env python
"""Read-only, serial audit of the formal 300-realization training environment pool.

This script intentionally imports no project runtime modules. It reads the training
episodes, shared config, service catalog, and split manifest; it never opens
validation/test episode contents or writes anything except the two new report files.
"""

from __future__ import annotations

import csv
import ctypes
import gc
import hashlib
import json
import math
import os
import re
import statistics
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "environment_data" / "proposed_joint_training"
MANIFEST_PATH = ROOT / "environment_data" / "data_split_manifest.json"
CONFIG_PATH = ROOT / "configs" / "dqn" / "online_joint_dqn.toml"
CATALOG_PATH = ROOT / "data" / "generated" / "service_catalog.json"
REPORT_PATH = ROOT / "reports" / "training_environment_diversity.md"
SUMMARY_PATH = ROOT / "reports" / "training_environment_diversity_summary.csv"

SEED_START = 20001
SEED_END = 20300
EXPECTED_SEEDS = tuple(range(SEED_START, SEED_END + 1))
EXPECTED_COUNT = len(EXPECTED_SEEDS)

REQUIRED_FILES = (
    "trajectory.csv",
    "vehicle_roles.json",
    "tasks.csv",
    "popularity.csv",
    "popularity_history.csv",
    "vehicle_predictions.csv",
    "popularity_predictions.csv",
    "prediction_scenarios.npz",
    "sumo_demand.rou.xml",
    "sumo_demand.meta.json",
)

HASH_FILES = {
    "trajectory": "trajectory.csv",
    "roles": "vehicle_roles.json",
    "tasks": "tasks.csv",
    "popularity": "popularity.csv",
    "vehicle_predictions": "vehicle_predictions.csv",
    "popularity_predictions": "popularity_predictions.csv",
    "prediction_scenarios": "prediction_scenarios.npz",
}

SIGNATURE_COMPONENTS = (
    "trajectory",
    "tasks",
    "popularity",
    "roles",
)

# A diagnostic definition of "near-identical" service distributions, in bits.
# This is not a scientific quality threshold and is used only for pair counting.
NEAR_IDENTICAL_JS_THRESHOLD = 1e-4


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def lower_this_process_priority() -> None:
    """On Windows, yield CPU scheduling priority to other applications."""
    if os.name == "nt":
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
        except (AttributeError, OSError):
            pass


def as_int(value: Any) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    rounded = round(number)
    return int(rounded) if abs(number - rounded) <= 1e-9 else None


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    return {
        "n": len(finite),
        "mean": statistics.fmean(finite),
        "std": statistics.pstdev(finite),
        "min": min(finite),
        "p25": percentile(finite, 0.25),
        "median": percentile(finite, 0.50),
        "p75": percentile(finite, 0.75),
        "max": max(finite),
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}g}"
    return str(value)


def read_hashed_bytes(path: Path) -> tuple[bytes, str]:
    data = path.read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def hash_file_streaming(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_hashed_csv(
    path: Path, row_callback: Callable[[dict[str, str]], None]
) -> tuple[set[str], str]:
    """Visit CSV rows serially while hashing the exact original bytes in the same pass."""
    digest = hashlib.sha256()

    def decoded_lines() -> Iterable[str]:
        first = True
        with path.open("rb") as stream:
            for raw_line in stream:
                digest.update(raw_line)
                encoding = "utf-8-sig" if first else "utf-8"
                first = False
                yield raw_line.decode(encoding)

    reader = csv.DictReader(decoded_lines())
    fieldnames = set(reader.fieldnames or [])
    for row in reader:
        row_callback(row)
    return fieldnames, digest.hexdigest()


def normalized_columns(fieldnames: Iterable[str]) -> dict[str, str]:
    return {name.strip().lower(): name for name in fieldnames if name}


def get_config_value(tree: dict[str, Any], dotted: str) -> Any:
    value: Any = tree
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def flatten_scalars(tree: Any, prefix: str = "") -> list[tuple[str, Any]]:
    flattened: list[tuple[str, Any]] = []
    if isinstance(tree, dict):
        for key in sorted(tree):
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.extend(flatten_scalars(tree[key], name))
    elif isinstance(tree, (str, int, float, bool)) or tree is None:
        flattened.append((prefix, tree))
    return flattened


def read_roles(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    raw, digest = read_hashed_bytes(path)
    roles = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(roles, dict):
        raise ValueError("vehicle_roles.json root is not an object")
    ids = {
        name: tuple(sorted({str(item) for item in roles.get(source, [])}))
        for name, source in (
            ("research", "research_vehicles"),
            ("sv", "service_vehicles"),
            ("tv", "task_vehicles"),
        )
    }
    return roles, {**ids, "sha256": digest}


def _empty_episode_row(seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "episode_dir": f"episode_{seed}",
        "episode_found": "NO",
        "required_files_complete": "NO",
        "missing_files": ";".join(REQUIRED_FILES),
        "research_vehicle_count": None,
        "sv_count": None,
        "tv_count": None,
        "unique_research_vehicle_ids": None,
        "research_vehicle_ids": "",
        "sv_ids": "",
        "tv_ids": "",
        "vehicle_mean_speed_mps": None,
        "vehicle_std_speed_mps": None,
        "vehicle_min_speed_mps": None,
        "vehicle_max_speed_mps": None,
        "vehicle_mean_acceleration_mps2": None,
        "mean_vehicle_travel_distance_m": None,
        "mean_position_dispersion_m": None,
        "trajectory_formal_rows": 0,
        "total_task_count": None,
        "task_opportunity_rows": 0,
        "realized_arrival_rate": None,
        "mean_input_size": None,
        "std_input_size": None,
        "mean_cpu_cycles": None,
        "std_cpu_cycles": None,
        "initial_top1_service": None,
        "initial_top3_services": "",
        "top1_change_count": None,
        "top3_turnover_count": None,
        "top3_turnover_rate": None,
        "pop_rho_tv_drift_mean": None,
        "pop_rank_mean_abs_change": None,
        "popularity_formal_slots": 0,
        "popularity_prediction_mae": None,
        "popularity_prediction_rmse": None,
        "popularity_prediction_error_rows": 0,
        "position_prediction_mae_m": None,
        "position_prediction_rmse_m": None,
        "position_prediction_error_rows": 0,
        "position_prediction_unmatched_in_range_rows": 0,
        "position_prediction_alignment_bad_rows": 0,
        "position_prediction_out_of_episode_rows": 0,
        "realization_signature": "",
    }


def process_trajectory(
    path: Path,
    roles: dict[str, Any],
    research_ids: set[str],
    config: dict[str, Any],
    row_summary: dict[str, Any],
) -> tuple[str, dict[str, dict[int, tuple[float, float]]], list[float], list[float], list[float]]:
    dt = as_float(roles.get("dt")) or as_float(get_config_value(config, "simulation.dt_s")) or 1.0
    warmup = as_float(roles.get("warmup_time"))
    if warmup is None:
        warmup = as_float(get_config_value(config, "simulation.warmup_time_s")) or 0.0
    episode_time = as_float(roles.get("episode_time"))
    if episode_time is None:
        episode_time = as_float(get_config_value(config, "simulation.episode_time_s")) or 0.0
    slot_count = int(round(episode_time / dt)) if dt > 0 else 0

    speeds: list[float] = []
    accelerations: list[float] = []
    positions_by_slot: dict[int, list[tuple[float, float]]] = defaultdict(list)
    positions_by_vehicle: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    distance_by_vehicle: dict[str, list[tuple[int, float]]] = defaultdict(list)

    def visit(row: dict[str, str]) -> None:
        vehicle_id = str(row.get("vehicle_id", ""))
        if vehicle_id not in research_ids:
            return
        time_s = as_float(row.get("time_s"))
        if time_s is None or dt <= 0:
            return
        formal_slot_float = (time_s - warmup) / dt
        formal_slot = int(round(formal_slot_float))
        if (
            abs(formal_slot_float - formal_slot) > 1e-6
            or formal_slot < 0
            or formal_slot >= slot_count
        ):
            return
        speed = as_float(row.get("speed_mps"))
        if speed is not None:
            speeds.append(speed)
        acceleration = as_float(row.get("acceleration_mps2"))
        if acceleration is not None:
            accelerations.append(acceleration)
        x_value = as_float(row.get("x_m"))
        y_value = as_float(row.get("y_m"))
        if x_value is not None and y_value is not None:
            point = (x_value, y_value)
            positions_by_slot[formal_slot].append(point)
            positions_by_vehicle[vehicle_id][formal_slot] = point
        distance = as_float(row.get("distance_m"))
        if distance is not None:
            distance_by_vehicle[vehicle_id].append((formal_slot, distance))

    fieldnames, digest = read_hashed_csv(path, visit)
    missing = {"time_s", "vehicle_id", "speed_mps", "x_m", "y_m"} - fieldnames
    if missing:
        raise ValueError(f"trajectory.csv missing fields: {sorted(missing)}")

    dispersions: list[float] = []
    for points in positions_by_slot.values():
        if len(points) < 2:
            continue
        mean_x = statistics.fmean(point[0] for point in points)
        mean_y = statistics.fmean(point[1] for point in points)
        dispersions.append(
            math.sqrt(
                statistics.fmean(
                    (x_value - mean_x) ** 2 + (y_value - mean_y) ** 2
                    for x_value, y_value in points
                )
            )
        )

    travel_distances: list[float] = []
    for observations in distance_by_vehicle.values():
        if len(observations) >= 2:
            ordered = sorted(observations)
            travel_distances.append(ordered[-1][1] - ordered[0][1])

    summary = numeric_summary(speeds)
    row_summary["trajectory_formal_rows"] = sum(len(v) for v in positions_by_slot.values())
    row_summary["vehicle_mean_speed_mps"] = summary["mean"]
    row_summary["vehicle_std_speed_mps"] = summary["std"]
    row_summary["vehicle_min_speed_mps"] = summary["min"]
    row_summary["vehicle_max_speed_mps"] = summary["max"]
    row_summary["vehicle_mean_acceleration_mps2"] = (
        statistics.fmean(accelerations) if accelerations else None
    )
    row_summary["mean_vehicle_travel_distance_m"] = (
        statistics.fmean(travel_distances) if travel_distances else None
    )
    row_summary["mean_position_dispersion_m"] = (
        statistics.fmean(dispersions) if dispersions else None
    )
    return digest, positions_by_vehicle, speeds, dispersions, travel_distances


def process_tasks(
    path: Path,
    num_services: int,
    episode_slots: int,
    row_summary: dict[str, Any],
) -> tuple[str, list[int], list[float], list[float]]:
    service_counts = [0] * num_services
    input_sizes: list[float] = []
    cpu_cycles: list[float] = []
    arrivals = 0
    opportunities = 0

    def visit(row: dict[str, str]) -> None:
        nonlocal arrivals, opportunities
        slot = as_int(row.get("slot"))
        if slot is None or slot < 0 or slot >= episode_slots:
            return
        opportunities += 1
        arrival = as_int(row.get("arrival"))
        if arrival != 1:
            return
        arrivals += 1
        service_id = as_int(row.get("service_id"))
        if service_id is not None and 0 <= service_id < num_services:
            service_counts[service_id] += 1
        input_size = as_float(row.get("input_size"))
        cpu = as_float(row.get("cpu_cycles"))
        if input_size is not None:
            input_sizes.append(input_size)
        if cpu is not None:
            cpu_cycles.append(cpu)

    fieldnames, digest = read_hashed_csv(path, visit)
    missing = {"slot", "arrival", "service_id", "input_size", "cpu_cycles"} - fieldnames
    if missing:
        raise ValueError(f"tasks.csv missing fields: {sorted(missing)}")
    input_summary = numeric_summary(input_sizes)
    cpu_summary = numeric_summary(cpu_cycles)
    row_summary["total_task_count"] = arrivals
    row_summary["task_opportunity_rows"] = opportunities
    row_summary["realized_arrival_rate"] = arrivals / opportunities if opportunities else None
    row_summary["mean_input_size"] = input_summary["mean"]
    row_summary["std_input_size"] = input_summary["std"]
    row_summary["mean_cpu_cycles"] = cpu_summary["mean"]
    row_summary["std_cpu_cycles"] = cpu_summary["std"]
    return digest, service_counts, input_sizes, cpu_cycles


def process_popularity(
    path: Path,
    num_services: int,
    episode_slots: int,
    row_summary: dict[str, Any],
) -> tuple[str, list[float] | None, str | None, str | None, dict[str, Any]]:
    columns: dict[str, str] = {}
    slot_rows: dict[int, dict[int, tuple[float, float | None]]] = defaultdict(dict)

    def visit(row: dict[str, str]) -> None:
        slot = as_int(row.get(columns.get("slot", "")))
        service_id = as_int(row.get(columns.get("service_id", "")))
        rho = as_float(row.get(columns.get("rho_star", "")))
        rank = as_float(row.get(columns.get("rank", ""))) if "rank" in columns else None
        if (
            slot is None
            or service_id is None
            or rho is None
            or slot < 0
            or slot >= episode_slots
            or service_id < 0
            or service_id >= num_services
        ):
            return
        slot_rows[slot][service_id] = (rho, rank)

    # Detect the header before consuming rows so service columns stay schema-driven.
    def read_all_hashed() -> tuple[set[str], str]:
        digest = hashlib.sha256()

        def decoded_lines() -> Iterable[str]:
            first = True
            with path.open("rb") as stream:
                for raw_line in stream:
                    digest.update(raw_line)
                    yield raw_line.decode("utf-8-sig" if first else "utf-8")
                    first = False

        reader = csv.DictReader(decoded_lines())
        fieldnames = set(reader.fieldnames or [])
        columns.update(normalized_columns(fieldnames))
        for row in reader:
            visit(row)
        return fieldnames, digest.hexdigest()

    fieldnames, digest = read_all_hashed()
    missing = {"slot", "service_id", "rho_star"} - set(columns)
    if missing:
        raise ValueError(f"popularity.csv missing fields: {sorted(missing)}")

    slots = sorted(slot_rows)
    row_summary["popularity_formal_slots"] = len(slots)
    if not slots:
        return digest, None, None, None, {}

    def order_for_slot(slot: int) -> list[int]:
        values = slot_rows[slot]
        if "rank" in columns and all(values[sid][1] is not None for sid in values):
            return sorted(values, key=lambda sid: (values[sid][1], sid))
        return sorted(values, key=lambda sid: (-values[sid][0], sid))

    initial_order = order_for_slot(slots[0])
    initial_top1 = initial_order[0] if initial_order else None
    initial_top3 = initial_order[: min(3, num_services)]
    row_summary["initial_top1_service"] = initial_top1
    row_summary["initial_top3_services"] = ",".join(str(value) for value in initial_top3)

    top1_changes = 0
    top3_changed_transitions = 0
    top3_replacements = 0
    adjacent_transitions = 0
    rho_drifts: list[float] = []
    rank_drifts: list[float] = []
    exact_path: list[tuple[int, ...]] = []
    canonical_path: list[tuple[int, ...]] = []
    previous_slot: int | None = None
    previous_order: list[int] | None = None

    for slot in slots:
        order = order_for_slot(slot)
        exact_path.append(tuple(order))
        values = slot_rows[slot]
        rank_lookup = {
            sid: values[sid][1]
            if values[sid][1] is not None
            else float(order.index(sid) + 1)
            for sid in values
        }
        canonical_path.append(
            tuple(int(rank_lookup.get(sid, num_services + 1)) for sid in initial_order)
        )
        if previous_slot is not None and slot == previous_slot + 1 and previous_order is not None:
            adjacent_transitions += 1
            if order and previous_order and order[0] != previous_order[0]:
                top1_changes += 1
            previous_top3 = set(previous_order[:3])
            current_top3 = set(order[:3])
            replacement_count = len(current_top3 - previous_top3)
            top3_replacements += replacement_count
            if replacement_count:
                top3_changed_transitions += 1
            previous_values = slot_rows[previous_slot]
            if set(values) == set(range(num_services)) and set(previous_values) == set(range(num_services)):
                rho_drifts.append(
                    0.5
                    * sum(abs(values[sid][0] - previous_values[sid][0]) for sid in range(num_services))
                )
                rank_drifts.append(
                    statistics.fmean(
                        abs(float(rank_lookup[sid]) - float(
                            previous_values[sid][1]
                            if previous_values[sid][1] is not None
                            else float(previous_order.index(sid) + 1)
                        ))
                        for sid in range(num_services)
                    )
                )
        previous_slot = slot
        previous_order = order

    row_summary["top1_change_count"] = top1_changes
    row_summary["top3_turnover_count"] = top3_changed_transitions
    row_summary["top3_turnover_rate"] = (
        top3_changed_transitions / adjacent_transitions if adjacent_transitions else None
    )
    row_summary["pop_rho_tv_drift_mean"] = statistics.fmean(rho_drifts) if rho_drifts else None
    row_summary["pop_rank_mean_abs_change"] = statistics.fmean(rank_drifts) if rank_drifts else None

    exact_digest = hashlib.sha256(repr(tuple(exact_path)).encode("utf-8")).hexdigest()
    canonical_digest = hashlib.sha256(repr(tuple(canonical_path)).encode("utf-8")).hexdigest()
    complete_grid = (
        len(slots) == episode_slots
        and slots == list(range(episode_slots))
        and all(set(slot_rows[slot]) == set(range(num_services)) for slot in slots)
    )
    path_info = {
        "complete_grid": complete_grid,
        "exact_path_digest": exact_digest,
        "canonical_path_digest": canonical_digest,
        "schema": sorted(columns),
        "initial_order": initial_order,
        "initial_top3": tuple(sorted(initial_top3)),
        "rho_drifts": rho_drifts,
        "rank_drifts": rank_drifts,
        "top3_replacements": top3_replacements,
        "adjacent_transitions": adjacent_transitions,
    }
    return digest, None, exact_digest, canonical_digest, path_info


def process_popularity_predictions(
    path: Path, row_summary: dict[str, Any]
) -> tuple[str, list[float], list[float]]:
    errors: list[float] = []
    squared_errors: list[float] = []
    digest_obj = hashlib.sha256()

    def decoded_lines() -> Iterable[str]:
        first = True
        with path.open("rb") as stream:
            for raw_line in stream:
                digest_obj.update(raw_line)
                yield raw_line.decode("utf-8-sig" if first else "utf-8")
                first = False

    reader = csv.DictReader(decoded_lines())
    columns = normalized_columns(reader.fieldnames or [])
    required = {"rho_pred", "rho_star"}
    if not required.issubset(columns):
        raise ValueError(f"popularity_predictions.csv missing fields: {sorted(required - set(columns))}")
    for row in reader:
        predicted = as_float(row.get(columns["rho_pred"]))
        truth = as_float(row.get(columns["rho_star"]))
        if predicted is None or truth is None:
            continue
        error = (
            as_float(row.get(columns["abs_error"]))
            if "abs_error" in columns
            else None
        )
        if error is None:
            error = abs(predicted - truth)
        errors.append(error)
        squared_errors.append((predicted - truth) ** 2)
    row_summary["popularity_prediction_error_rows"] = len(errors)
    row_summary["popularity_prediction_mae"] = statistics.fmean(errors) if errors else None
    row_summary["popularity_prediction_rmse"] = (
        math.sqrt(statistics.fmean(squared_errors)) if squared_errors else None
    )
    return digest_obj.hexdigest(), errors, squared_errors


def process_vehicle_predictions(
    path: Path,
    truth_positions: dict[str, dict[int, tuple[float, float]]],
    episode_slots: int,
    row_summary: dict[str, Any],
) -> tuple[str, list[float], list[float], set[int]]:
    errors: list[float] = []
    squared_errors: list[float] = []
    horizons_seen: set[int] = set()
    columns: dict[str, str] = {}
    unmatched = 0
    alignment_bad = 0
    out_of_episode = 0

    def visit(row: dict[str, str]) -> None:
        nonlocal unmatched, alignment_bad, out_of_episode
        vehicle_id = str(row.get(columns.get("vehicle_id", ""), ""))
        origin = as_int(row.get(columns.get("origin_slot", "")))
        horizon = as_int(row.get(columns.get("horizon", "")))
        target = as_int(row.get(columns.get("target_slot", "")))
        if horizon is not None:
            horizons_seen.add(horizon)
        if origin is None or horizon is None or target is None:
            alignment_bad += 1
            return
        if target != origin + horizon:
            alignment_bad += 1
            return
        if target < 0 or target >= episode_slots:
            out_of_episode += 1
            return
        prediction_x = as_float(row.get(columns.get("x_pred", "")))
        prediction_y = as_float(row.get(columns.get("y_pred", "")))
        truth = truth_positions.get(vehicle_id, {}).get(target)
        if prediction_x is None or prediction_y is None or truth is None:
            unmatched += 1
            return
        error = math.hypot(prediction_x - truth[0], prediction_y - truth[1])
        errors.append(error)
        squared_errors.append(error * error)

    def decoded_lines(digest: Any) -> Iterable[str]:
        first = True
        with path.open("rb") as stream:
            for raw_line in stream:
                digest.update(raw_line)
                yield raw_line.decode("utf-8-sig" if first else "utf-8")
                first = False

    digest_obj = hashlib.sha256()
    reader = csv.DictReader(decoded_lines(digest_obj))
    fieldnames = set(reader.fieldnames or [])
    columns.update(normalized_columns(fieldnames))
    required = {"vehicle_id", "origin_slot", "horizon", "target_slot", "x_pred", "y_pred"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"vehicle_predictions.csv missing fields: {sorted(missing)}")
    for row in reader:
        visit(row)

    row_summary["position_prediction_error_rows"] = len(errors)
    row_summary["position_prediction_unmatched_in_range_rows"] = unmatched
    row_summary["position_prediction_alignment_bad_rows"] = alignment_bad
    row_summary["position_prediction_out_of_episode_rows"] = out_of_episode
    row_summary["position_prediction_mae_m"] = statistics.fmean(errors) if errors else None
    row_summary["position_prediction_rmse_m"] = (
        math.sqrt(statistics.fmean(squared_errors)) if squared_errors else None
    )
    return digest_obj.hexdigest(), errors, squared_errors, horizons_seen


def js_divergence(left: list[float], right: list[float]) -> float:
    middle = [(a + b) / 2.0 for a, b in zip(left, right)]
    result = 0.0
    for p_value, q_value, m_value in zip(left, right, middle):
        if p_value > 0.0:
            result += 0.5 * p_value * math.log2(p_value / m_value)
        if q_value > 0.0:
            result += 0.5 * q_value * math.log2(q_value / m_value)
    return max(0.0, result)


def make_seed_set(split: dict[str, Any]) -> set[int]:
    start = int(split["seed_start"])
    end = int(split["seed_end"])
    return set(range(start, end + 1))


def duplicate_groups(hash_by_seed: dict[int, str | None]) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for seed, digest in hash_by_seed.items():
        if digest:
            groups[digest].append(seed)
    return sorted(
        [sorted(seeds) for seeds in groups.values() if len(seeds) > 1],
        key=lambda seeds: seeds[0],
    )


def format_seed_groups(groups: list[list[int]]) -> str:
    if not groups:
        return "无"
    return "; ".join(", ".join(str(seed) for seed in group) for group in groups)


def format_ranges(values: Iterable[int]) -> str:
    ordered = sorted(set(values))
    if not ordered:
        return "无"
    chunks: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        chunks.append(str(start) if start == previous else f"{start}–{previous}")
        start = previous = value
    chunks.append(str(start) if start == previous else f"{start}–{previous}")
    return ", ".join(chunks)


def verify_role_metadata(
    roles: dict[str, Any], config: dict[str, Any], num_services: int
) -> list[str]:
    expected = {
        "dt": as_float(get_config_value(config, "simulation.dt_s")),
        "warmup_time": as_float(get_config_value(config, "simulation.warmup_time_s")),
        "episode_time": as_float(get_config_value(config, "simulation.episode_time_s")),
        "num_research_vehicles": as_int(get_config_value(config, "simulation.num_research_vehicles")),
        "num_service_vehicles": as_int(get_config_value(config, "simulation.num_service_vehicles")),
        "num_task_vehicles": as_int(get_config_value(config, "simulation.num_task_vehicles")),
    }
    issues: list[str] = []
    for key, expected_value in expected.items():
        actual = roles.get(key)
        if expected_value is None or actual is None:
            issues.append(f"roles metadata lacks/does not parse {key}")
            continue
        if isinstance(expected_value, float):
            actual_value = as_float(actual)
            matches = actual_value is not None and abs(actual_value - expected_value) <= 1e-9
        else:
            actual_value = as_int(actual)
            matches = actual_value == expected_value
        if not matches:
            issues.append(f"roles metadata {key}={actual!r}, config={expected_value!r}")
    dt = expected["dt"] or 1.0
    duration = expected["episode_time"] or 0.0
    expected_slots = int(round(duration / dt))
    if as_int(roles.get("formal_time_slot_count")) != expected_slots:
        issues.append(
            f"roles metadata formal_time_slot_count={roles.get('formal_time_slot_count')!r}, config-derived={expected_slots}"
        )
    role_sizes = {
        "research_vehicles": as_int(get_config_value(config, "simulation.num_research_vehicles")),
        "service_vehicles": as_int(get_config_value(config, "simulation.num_service_vehicles")),
        "task_vehicles": as_int(get_config_value(config, "simulation.num_task_vehicles")),
    }
    for key, expected_count in role_sizes.items():
        actual_values = roles.get(key)
        if not isinstance(actual_values, list) or len(actual_values) != expected_count:
            issues.append(f"roles list {key} length={len(actual_values) if isinstance(actual_values, list) else 'invalid'}, expected={expected_count}")
    if as_int(get_config_value(config, "service.count")) != num_services:
        issues.append("service count differs between config and catalog")
    return issues


def render_markdown(
    results: dict[str, Any], config: dict[str, Any], manifest: dict[str, Any], catalog: dict[str, Any]
) -> str:
    rows = results["rows"]
    found = results["found_seeds"]
    missing_seeds = results["missing_seeds"]
    unexpected = results["unexpected_seeds"]
    missing_files = results["missing_files_by_seed"]
    file_hashes = results["hashes"]

    def dist_table(title: str, data: dict[str, Any], unit: str = "") -> list[str]:
        keys = ("mean", "std", "min", "p25", "median", "p75", "max")
        cells = [fmt(data.get(key)) for key in keys]
        suffix = f" {unit}" if unit else ""
        return [
            f"| {title} | {cells[0]}{suffix} | {cells[1]}{suffix} | {cells[2]}{suffix} | {cells[3]}{suffix} | {cells[4]}{suffix} | {cells[5]}{suffix} | {cells[6]}{suffix} |",
        ]

    lines: list[str] = [
        "# 训练环境多样性检查",
        "",
        "本报告由只读、单进程 CPU 脚本生成。分析顺序处理训练 episode；没有导入项目运行时模块，没有执行训练、rollout、SUMO/TraCI、Matching、KKT、Exact-Q 或反事实环境代码。未读取 validation/test episode 内容。",
        "",
        "## 1. 数据协议",
        "",
        f"- Manifest：`environment_data/data_split_manifest.json`，protocol_version={manifest.get('protocol_version')}，split_unit=`{manifest.get('split_unit')}`，slots_per_realization={manifest.get('slots_per_realization')}。",
        f"- Train 声明：{manifest.get('train', {}).get('seed_start')}–{manifest.get('train', {}).get('seed_end')}，count={manifest.get('train', {}).get('count')}；审计范围固定为 {SEED_START}–{SEED_END}。",
        f"- Validation 声明：{manifest.get('validation', {}).get('seed_start')}–{manifest.get('validation', {}).get('seed_end')}，count={manifest.get('validation', {}).get('count')}；Test 声明：{manifest.get('test', {}).get('seed_start')}–{manifest.get('test', {}).get('seed_end')}，count={manifest.get('test', {}).get('count')}。",
        f"- Service catalog：{catalog.get('service_count')} services；manifest SHA256 匹配={fmt(results['catalog_hash_matches_manifest'])}。",
        "- Manifest 说明 training pool 是既有 300-realization 池且没有逐 episode `episode_manifest.json`；因此没有把共享配置值描述成逐 episode snapshot 验证结果。",
        "",
        "## 2. 完整性",
        "",
        f"- EXPECTED_EPISODES = {EXPECTED_COUNT}",
        f"- FOUND_EPISODES = {len(found)}",
        f"- MISSING_EPISODES = {len(missing_seeds)} ({format_ranges(missing_seeds)})",
        f"- UNEXPECTED_EPISODES = {len(unexpected)} ({format_ranges(unexpected)})",
        f"- POOL_COMPLETE = {results['pool_complete']}",
        f"- 必需文件缺失 episode 数 = {len(missing_files)}",
    ]
    if missing_files:
        lines.extend(["- 必需文件缺失明细："])
        for seed in sorted(missing_files):
            lines.append(f"  - episode_{seed}: {', '.join(missing_files[seed])}")
    lines.extend(
        [
            "",
            "## 3. 重复与 SHA256 检查",
            "",
            "表中计数的分母为 300；缺失文件没有 hash。Realization signature 是 trajectory、tasks、popularity、vehicle_roles 四个文件名及其 SHA256 按固定顺序拼接后再做 SHA256。",
            "",
            "| Artifact | Unique hashes | 重复 episode seed 组 |",
            "|---|---:|---|",
        ]
    )
    for key in HASH_FILES:
        digests = [file_hashes.get(seed, {}).get(key) for seed in EXPECTED_SEEDS]
        unique = len({value for value in digests if value})
        groups = duplicate_groups({seed: file_hashes.get(seed, {}).get(key) for seed in EXPECTED_SEEDS})
        lines.append(f"| `{HASH_FILES[key]}` | {unique}/{EXPECTED_COUNT} | {format_seed_groups(groups)} |")
    lines.append(
        f"| `REALIZATION_SIGNATURE` | {results['full_unique_count']}/{EXPECTED_COUNT} | {format_seed_groups(results['signature_duplicate_groups'])} |"
    )
    lines.extend(
        [
            "",
            f"- FULL_REALIZATION_UNIQUE_COUNT = {results['full_unique_count']}/{EXPECTED_COUNT}",
            f"- REALIZATION_DUPLICATION = {results['realization_duplication']}",
            "",
            "## 4. Vehicle diversity",
            "",
            f"- RESEARCH_VEHICLE_SET_UNIQUE_COUNT = {results['research_set_unique_count']}/{EXPECTED_COUNT}",
            f"- SV_SET_UNIQUE_COUNT = {results['sv_set_unique_count']}/{EXPECTED_COUNT}",
            f"- TV_SET_UNIQUE_COUNT = {results['tv_set_unique_count']}/{EXPECTED_COUNT}",
            f"- 每 episode formal research vehicle rows：以 roles warmup/dt 对齐正式 slot 0…N−1 后统计；position dispersion 是每 slot 研究车辆二维坐标相对该 slot 质心的 RMS 距离，再对 slot 取均值。速度/加速度取正式 300-slot 研究车辆记录。",
            "",
            "| Episode-level metric | mean | std | min | p25 | median | p75 | max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(dist_table("Mean speed (m/s)", results["vehicle_mean_speed_dist"], "m/s"))
    lines.extend(dist_table("Speed std (m/s)", results["vehicle_speed_std_dist"], "m/s"))
    lines.extend(dist_table("Position dispersion (m)", results["position_dispersion_dist"], "m"))
    lines.extend(dist_table("Mean vehicle travel distance (m)", results["travel_distance_dist"], "m"))
    lines.extend(
        [
            "",
            f"- Mean research vehicles / SV / TV per episode = {fmt(results['mean_research_count'])} / {fmt(results['mean_sv_count'])} / {fmt(results['mean_tv_count'])}。mean_vehicle_travel_distance 使用 formal window 内 `distance_m` 首末记录差；无法形成至少两个记录的车辆不计入。",
            "",
            "| Episode-level metric | mean | std | min | p25 | median | p75 | max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(dist_table("Mean acceleration (m/s²)", results["mean_acceleration_dist"], "m/s²"))
    lines.extend(dist_table("Episode min speed (m/s)", results["episode_min_speed_dist"], "m/s"))
    lines.extend(dist_table("Episode max speed (m/s)", results["episode_max_speed_dist"], "m/s"))
    lines.extend(
        [
            "",
            "## 5. Task diversity",
            "",
            f"- Config/catalog 确认 F={results['num_services']}；num_task_vehicles={fmt(get_config_value(config, 'simulation.num_task_vehicles'))}；episode slots={fmt(get_config_value(config, 'training.episode_slots') or manifest.get('slots_per_realization'))}；arrival_probability={fmt(get_config_value(config, 'task.arrival_probability'))}。",
            "- `total_task_count` 是 `arrival=1` 行数；realized arrival rate 的分母是读到的正式 TV×slot opportunity 行数。输入大小和 cpu_cycles 统计只针对到达任务。",
            "",
            "| Episode-level metric | mean | std | min | p25 | median | p75 | max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(dist_table("Task count", results["task_count_dist"]))
    lines.extend(dist_table("Arrival rate", results["arrival_rate_dist"]))
    lines.extend(dist_table("Mean input size", results["mean_input_size_dist"]))
    lines.extend(dist_table("Within-episode input-size std", results["input_size_std_dist"]))
    lines.extend(dist_table("Mean cpu_cycles", results["mean_cpu_cycles_dist"]))
    lines.extend(dist_table("Within-episode cpu_cycles std", results["cpu_cycles_std_dist"]))
    lines.extend(
        [
            "",
            f"- 300×F service-probability vectors由 `tasks.csv` 到达任务的 service_id 计数归一化；每个 episode 的概率列写入 summary CSV。TASKS_UNIQUE_COUNT = {results['hash_unique_counts']['tasks']}/{EXPECTED_COUNT}。",
            "",
            "## 6. Service request distribution diversity",
            "",
            f"- 两两 Jensen–Shannon divergence 使用 log₂，单位 bits；比较 episode empirical service probability。近似相同的诊断阈值为 JS ≤ {NEAR_IDENTICAL_JS_THRESHOLD:g} bits。该阈值仅用于 near-identical pair count，不代表外部质量标准。",
            f"- 有效 service vectors = {len(results['service_distributions'])}/{EXPECTED_COUNT}；pair count = {len(results['js_values'])}；NEAR_IDENTICAL_SERVICE_DISTRIBUTION_PAIR_COUNT = {results['near_identical_js_pair_count']}。",
            "",
            "| Pairwise JS (bits) | mean | std | min(non-identical) | p10 | p25 | median | p75 | p90 | max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    js = results["js_summary"]
    lines.append(
        "| All valid episode pairs | "
        + " | ".join(fmt(js.get(key)) for key in ("mean", "std", "min_nonidentical", "p10", "p25", "median", "p75", "p90", "max"))
        + " |"
    )
    lines.extend(
        [
            "",
            "## 7. Popularity diversity",
            "",
            f"- schema 自动识别到列：`{', '.join(results['popularity_schema'])}`；true popularity 取 `rho_star`，top order 优先按 `rank`，没有 rank 时按 rho_star 降序。",
            f"- INITIAL_TOP1_SERVICE_FREQUENCY = {json.dumps(results['initial_top1_frequency'], ensure_ascii=False, sort_keys=True)}",
            f"- INITIAL_TOP3_DIVERSITY = {results['initial_top3_diversity']} unique top-3 sets；initial top-3 覆盖服务数 = {results['initial_top3_service_coverage']}。",
            f"- Labeled popularity path unique = {results['popularity_exact_path_unique_count']}/{EXPECTED_COUNT}；按每 episode 初始 rank 重排服务 label 后的 canonical path unique = {results['popularity_canonical_path_unique_count']}/{EXPECTED_COUNT}。",
            f"- 所有 labeled path 相同 = {results['all_popularity_paths_identical']}；仅 label permutation、rank evolution 相同 = {results['same_evolution_different_label_realization']}。",
            "- `top3_turnover_count` 计 top-3 集合发生变化的相邻 slot 转移数；rate 为该数/可比较相邻转移数。`pop_rho_tv_drift` 是相邻 slot rho_star 分布的 0.5×L1/total variation distance；rank change 是跨服务平均绝对 rank 改变量。",
            "",
            "| Episode-level metric | mean | std | min | p25 | median | p75 | max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(dist_table("Top-1 change count", results["top1_change_dist"]))
    lines.extend(dist_table("Top-3 changed-transition count", results["top3_turnover_count_dist"]))
    lines.extend(dist_table("Top-3 changed-transition rate", results["top3_turnover_rate_dist"]))
    lines.extend(dist_table("Temporal rho_star TV drift", results["temporal_popularity_drift_dist"]))
    lines.extend(dist_table("Temporal mean absolute rank change", results["temporal_rank_drift_dist"]))
    lines.extend(
        [
            "",
            "## 8. Prediction difficulty",
            "",
            f"- Popularity prediction 使用已保存 `rho_pred` 与 `rho_star` truth；若 `abs_error` 有值则使用该列，否则按 |rho_pred−rho_star| 计算。统计 MAE/RMSE，共 {results['pop_prediction_error_rows']} truth-labeled rows。",
            f"- Vehicle position prediction 逐行核对 `target_slot == origin_slot + horizon`，用 `vehicle_id + target_slot` 对齐 formal trajectory `x_m/y_m` 与 `x_pred/y_pred`，误差为平面欧氏距离。有效对齐行={results['position_error_rows']}；in-range unmatched={results['position_unmatched_rows']}；key mismatch={results['position_alignment_bad_rows']}；episode 外 horizon={results['position_out_of_episode_rows']}。",
            f"- POSITION_PREDICTION_ERROR_CHECK = {results['position_prediction_check']}。",
            "",
            "| Popularity prediction MAE | mean | std | min | p25 | median | p75 | max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            "| Episode MAE | " + " | ".join(fmt(results["pop_prediction_mae_dist"].get(key)) for key in ("mean", "std", "min", "p25", "median", "p75", "max")) + " |",
            "",
            "| Popularity prediction RMSE | mean | std | min | p25 | median | p75 | max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            "| Episode RMSE | " + " | ".join(fmt(results["pop_prediction_rmse_dist"].get(key)) for key in ("mean", "std", "min", "p25", "median", "p75", "max")) + " |",
            "",
            "| Vehicle position prediction MAE (m) | mean | std | min | p25 | median | p75 | max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            "| Episode MAE | " + " | ".join(fmt(results["position_prediction_mae_dist"].get(key)) for key in ("mean", "std", "min", "p25", "median", "p75", "max")) + " |",
            "",
            "## 9. Fixed experiment contract consistency",
            "",
            f"- FIXED_EXPERIMENT_CONTRACT_CONSISTENT = {results['fixed_contract_consistent']}",
            "- 逐 episode roles metadata 与 shared formal config 中可见的 dt/warmup/episode length/车辆角色数量/slot count 对照；service count 对照 catalog；预测 horizon 对照配置。没有 episode-level snapshot 的参数标为 shared formal config 固定。",
        ]
    )
    for key, value in results["contract_values"]:
        lines.append(f"- `{key}` = `{value}` (fixed by shared formal config)")
    if results["contract_issues"]:
        lines.append("- Contract issues:")
        lines.extend(f"  - {issue}" for issue in results["contract_issues"])
    if results["config_hash_differs_from_manifest"]:
        note = manifest.get("generator_config_text_change_note", "")
        lines.append(
            f"- 当前 config 文件 SHA256 与 manifest 记录值不同；manifest note：{note or '无说明'}。解析后的字段仍按共享 config 和 roles metadata 核对。"
        )
    lines.extend(
        [
            "",
            "## 10. Train / validation / test leakage",
            "",
            f"- TRAIN_VALIDATION_OVERLAP = {format_ranges(results['train_validation_overlap'])}",
            f"- TRAIN_TEST_OVERLAP = {format_ranges(results['train_test_overlap'])}",
            f"- VALIDATION_TEST_OVERLAP = {format_ranges(results['validation_test_overlap'])}",
            f"- TRAIN_VALIDATION_TEST_SEED_LEAKAGE = {results['seed_leakage']}",
            f"- split directory names exist = {fmt(results['split_directories_exist'])}；只检查 validation/test 目录是否存在，没有列举其内容或读取其中 episode 文件。",
            "",
            "## 11. 最终结论",
            "",
            f"- VEHICLE_ENVIRONMENT_VARIATION = {results['vehicle_variation']}",
            f"- TASK_ENVIRONMENT_VARIATION = {results['task_variation']}",
            f"- POPULARITY_VARIATION = {results['popularity_variation']}",
            f"- PREDICTION_DIFFICULTY_VARIATION = {results['prediction_variation']}",
            f"- TRAIN_ENVIRONMENT_DIVERSITY = {results['overall_diversity']}",
            "",
            "本分类是对本次持久化数据的保守描述，不使用外部所谓顶刊阈值。对总分类中的 service spread，若至少一半 episode pair 落在明示的 near-identical 诊断区间，则按 LIMITED 处理；这是透明的多数描述规则，不是外部质量标准。Service JS 门槛仅按第 6 节明示用于 pair 诊断。",
            "",
            "运行约束记录：TRAINING_EXECUTED=NO；SUMO_EXECUTED=NO；MATCHING_EXECUTED=NO；KKT_EXECUTED=NO；GPU_USED=NO；MULTIPROCESSING_USED=NO；OTHER_PROCESSES_MODIFIED=NO。",
            "",
        ]
    )
    return "\n".join(lines)


def build_summary_csv(rows: list[dict[str, Any]], num_services: int) -> str:
    columns = [
        "seed",
        "episode_dir",
        "episode_found",
        "required_files_complete",
        "missing_files",
        "research_vehicle_count",
        "sv_count",
        "tv_count",
        "unique_research_vehicle_ids",
        "research_vehicle_ids",
        "sv_ids",
        "tv_ids",
        "trajectory_formal_rows",
        "vehicle_mean_speed_mps",
        "vehicle_std_speed_mps",
        "vehicle_min_speed_mps",
        "vehicle_max_speed_mps",
        "vehicle_mean_acceleration_mps2",
        "mean_vehicle_travel_distance_m",
        "mean_position_dispersion_m",
        "total_task_count",
        "task_opportunity_rows",
        "realized_arrival_rate",
        "mean_input_size",
        "std_input_size",
        "mean_cpu_cycles",
        "std_cpu_cycles",
    ]
    columns.extend(f"service_{sid}_count" for sid in range(num_services))
    columns.extend(f"service_{sid}_probability" for sid in range(num_services))
    columns.extend(
        [
            "initial_top1_service",
            "initial_top3_services",
            "top1_change_count",
            "top3_turnover_count",
            "top3_turnover_rate",
            "pop_rho_tv_drift_mean",
            "pop_rank_mean_abs_change",
            "popularity_formal_slots",
            "popularity_prediction_mae",
            "popularity_prediction_rmse",
            "popularity_prediction_error_rows",
            "position_prediction_mae_m",
            "position_prediction_rmse_m",
            "position_prediction_error_rows",
            "position_prediction_unmatched_in_range_rows",
            "position_prediction_alignment_bad_rows",
            "position_prediction_out_of_episode_rows",
            "realization_signature",
        ]
    )
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue()


def main() -> None:
    lower_this_process_priority()
    for target in (REPORT_PATH, SUMMARY_PATH):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {target}")
    for required in (TRAIN_DIR, MANIFEST_PATH, CONFIG_PATH, CATALOG_PATH):
        if not required.exists():
            raise FileNotFoundError(required)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8-sig"))
    config_bytes = CONFIG_PATH.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8-sig"))
    catalog_bytes, catalog_hash = read_hashed_bytes(CATALOG_PATH)
    catalog = json.loads(catalog_bytes.decode("utf-8-sig"))
    num_services = as_int(get_config_value(config, "service.count"))
    if num_services is None or num_services <= 0:
        raise ValueError("Could not determine service.count from formal config")
    episode_slots = as_int(manifest.get("slots_per_realization"))
    if episode_slots is None:
        episode_slots = int(round(
            (as_float(get_config_value(config, "simulation.episode_time_s")) or 0.0)
            / (as_float(get_config_value(config, "simulation.dt_s")) or 1.0)
        ))

    child_episode_ids: set[int] = set()
    unexpected_seeds: list[int] = []
    for child in TRAIN_DIR.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"episode_(\d+)", child.name)
        if not match:
            continue
        seed = int(match.group(1))
        child_episode_ids.add(seed)
        if seed not in EXPECTED_SEEDS:
            unexpected_seeds.append(seed)
    found_seeds = [seed for seed in EXPECTED_SEEDS if seed in child_episode_ids]
    missing_seeds = [seed for seed in EXPECTED_SEEDS if seed not in child_episode_ids]

    rows: list[dict[str, Any]] = []
    hashes: dict[int, dict[str, str | None]] = {}
    research_sets: set[frozenset[str]] = set()
    sv_sets: set[frozenset[str]] = set()
    tv_sets: set[frozenset[str]] = set()
    service_distributions: list[list[float]] = []
    popularity_info: dict[int, dict[str, Any]] = {}
    contract_issues: list[str] = []
    role_metadata_checked = 0
    catalog_service_ids = sorted(
        as_int(service.get("service_id"))
        for service in catalog.get("services", [])
        if isinstance(service, dict) and as_int(service.get("service_id")) is not None
    )
    expected_catalog_ids = list(range(num_services))
    catalog_hash_matches_manifest = (
        catalog_hash.lower()
        == str(manifest.get("service_catalog_sha256", "")).lower()
    )

    for seed in EXPECTED_SEEDS:
        row = _empty_episode_row(seed)
        episode_dir = TRAIN_DIR / f"episode_{seed}"
        episode_hashes: dict[str, str | None] = {key: None for key in HASH_FILES}
        hashes[seed] = episode_hashes
        if not episode_dir.is_dir():
            rows.append(row)
            continue
        row["episode_found"] = "YES"
        missing_files = [name for name in REQUIRED_FILES if not (episode_dir / name).is_file()]
        row["missing_files"] = ";".join(missing_files)
        row["required_files_complete"] = "YES" if not missing_files else "NO"

        roles: dict[str, Any] | None = None
        role_ids: dict[str, Any] = {}
        role_path = episode_dir / "vehicle_roles.json"
        if role_path.is_file():
            try:
                roles, role_ids = read_roles(role_path)
                episode_hashes["roles"] = role_ids["sha256"]
                research_ids = set(role_ids["research"])
                row["research_vehicle_count"] = len(role_ids["research"])
                row["sv_count"] = len(role_ids["sv"])
                row["tv_count"] = len(role_ids["tv"])
                row["unique_research_vehicle_ids"] = len(research_ids)
                row["research_vehicle_ids"] = ";".join(role_ids["research"])
                row["sv_ids"] = ";".join(role_ids["sv"])
                row["tv_ids"] = ";".join(role_ids["tv"])
                research_sets.add(frozenset(research_ids))
                sv_sets.add(frozenset(role_ids["sv"]))
                tv_sets.add(frozenset(role_ids["tv"]))
                role_metadata_checked += 1
                contract_issues.extend(
                    f"episode_{seed}: {issue}"
                    for issue in verify_role_metadata(roles, config, num_services)
                )
            except Exception as exc:  # keep auditing other episodes
                contract_issues.append(f"episode_{seed}: vehicle_roles.json parse error: {exc}")
                roles = None
                research_ids = set()
        else:
            research_ids = set()
        if roles is None:
            research_ids = set()

        scenarios_path = episode_dir / "prediction_scenarios.npz"
        if scenarios_path.is_file():
            try:
                episode_hashes["prediction_scenarios"] = hash_file_streaming(scenarios_path)
            except OSError as exc:
                contract_issues.append(f"episode_{seed}: prediction_scenarios.npz read error: {exc}")

        truth_positions: dict[str, dict[int, tuple[float, float]]] = {}
        trajectory_path = episode_dir / "trajectory.csv"
        if trajectory_path.is_file() and roles is not None:
            try:
                digest, positions, _, _, _ = process_trajectory(
                    trajectory_path, roles, research_ids, config, row
                )
                episode_hashes["trajectory"] = digest
                truth_positions = positions
            except Exception as exc:
                contract_issues.append(f"episode_{seed}: trajectory.csv analysis error: {exc}")
        elif trajectory_path.is_file():
            try:
                episode_hashes["trajectory"] = hash_file_streaming(trajectory_path)
            except OSError as exc:
                contract_issues.append(f"episode_{seed}: trajectory.csv hash error: {exc}")

        task_counts: list[int] | None = None
        tasks_path = episode_dir / "tasks.csv"
        if tasks_path.is_file():
            try:
                digest, task_counts, _, _ = process_tasks(
                    tasks_path, num_services, episode_slots, row
                )
                episode_hashes["tasks"] = digest
                opportunities_expected = (
                    as_int(get_config_value(config, "simulation.num_task_vehicles")) or 0
                ) * episode_slots
                if row["task_opportunity_rows"] != opportunities_expected:
                    contract_issues.append(
                        f"episode_{seed}: task opportunities={row['task_opportunity_rows']}, expected={opportunities_expected}"
                    )
                task_total = sum(task_counts)
                if task_total != row["total_task_count"]:
                    contract_issues.append(
                        f"episode_{seed}: arrived task rows with service_id in 0..F-1={task_total}, arrival rows={row['total_task_count']}"
                    )
                service_distributions.append(
                    [count / task_total for count in task_counts]
                    if task_total
                    else [0.0] * num_services
                )
                for service_id, count in enumerate(task_counts):
                    row[f"service_{service_id}_count"] = count
                    row[f"service_{service_id}_probability"] = (
                        count / task_total if task_total else None
                    )
            except Exception as exc:
                contract_issues.append(f"episode_{seed}: tasks.csv analysis error: {exc}")

        popularity_path = episode_dir / "popularity.csv"
        if popularity_path.is_file():
            try:
                digest, _, exact_path_digest, canonical_digest, info = process_popularity(
                    popularity_path, num_services, episode_slots, row
                )
                episode_hashes["popularity"] = digest
                info["exact_path_digest"] = exact_path_digest
                info["canonical_path_digest"] = canonical_digest
                popularity_info[seed] = info
                if not info.get("complete_grid"):
                    contract_issues.append(f"episode_{seed}: popularity.csv does not cover full slot × service grid")
            except Exception as exc:
                contract_issues.append(f"episode_{seed}: popularity.csv analysis error: {exc}")

        pop_prediction_path = episode_dir / "popularity_predictions.csv"
        if pop_prediction_path.is_file():
            try:
                digest, _, _ = process_popularity_predictions(pop_prediction_path, row)
                episode_hashes["popularity_predictions"] = digest
            except Exception as exc:
                contract_issues.append(f"episode_{seed}: popularity_predictions.csv analysis error: {exc}")

        vehicle_prediction_path = episode_dir / "vehicle_predictions.csv"
        if vehicle_prediction_path.is_file():
            try:
                digest, _, _, horizons = process_vehicle_predictions(
                    vehicle_prediction_path, truth_positions, episode_slots, row
                )
                episode_hashes["vehicle_predictions"] = digest
                configured_horizon = as_int(get_config_value(config, "prediction.vehicle.horizon"))
                if configured_horizon is not None and horizons and horizons != set(range(configured_horizon)):
                    contract_issues.append(
                        f"episode_{seed}: vehicle prediction horizons={sorted(horizons)}, expected=0..{configured_horizon - 1}"
                    )
            except Exception as exc:
                contract_issues.append(f"episode_{seed}: vehicle_predictions.csv analysis error: {exc}")

        if all(episode_hashes.get(key) for key in SIGNATURE_COMPONENTS):
            signature_source = "\n".join(
                f"{key}:{episode_hashes[key]}" for key in SIGNATURE_COMPONENTS
            )
            row["realization_signature"] = hashlib.sha256(
                signature_source.encode("ascii")
            ).hexdigest()
        rows.append(row)
        gc.collect()

    # Hash files excluded from numeric parsing but still required to exist.
    # popularity_history and SUMO artifacts are presence-only by task definition.
    signature_hashes = {
        seed: row["realization_signature"] or None
        for seed, row in zip(EXPECTED_SEEDS, rows)
    }
    signature_duplicate_groups = duplicate_groups(signature_hashes)
    full_unique_count = len({value for value in signature_hashes.values() if value})
    realization_duplication = "YES" if signature_duplicate_groups else "NO"

    hash_unique_counts: dict[str, int] = {}
    hash_duplicate_groups: dict[str, list[list[int]]] = {}
    for key in HASH_FILES:
        by_seed = {seed: hashes[seed].get(key) for seed in EXPECTED_SEEDS}
        hash_unique_counts[key] = len({value for value in by_seed.values() if value})
        hash_duplicate_groups[key] = duplicate_groups(by_seed)

    js_values: list[float] = []
    for left_index in range(len(service_distributions)):
        for right_index in range(left_index + 1, len(service_distributions)):
            js_values.append(js_divergence(service_distributions[left_index], service_distributions[right_index]))
    nonidentical_js = [value for value in js_values if value > 1e-15]
    js_summary = {
        "mean": statistics.fmean(js_values) if js_values else None,
        "std": statistics.pstdev(js_values) if js_values else None,
        "min_nonidentical": min(nonidentical_js) if nonidentical_js else None,
        "p10": percentile(js_values, 0.10),
        "p25": percentile(js_values, 0.25),
        "median": percentile(js_values, 0.50),
        "p75": percentile(js_values, 0.75),
        "p90": percentile(js_values, 0.90),
        "max": max(js_values) if js_values else None,
    }
    near_identical_js_pair_count = sum(
        value <= NEAR_IDENTICAL_JS_THRESHOLD for value in js_values
    )

    valid_popularity = {
        seed: info for seed, info in popularity_info.items() if info.get("complete_grid")
    }
    exact_pop_paths = {
        info["exact_path_digest"] for info in valid_popularity.values() if info.get("exact_path_digest")
    }
    canonical_pop_paths = {
        info["canonical_path_digest"]
        for info in valid_popularity.values()
        if info.get("canonical_path_digest")
    }
    initial_top1_frequency = Counter(
        str(rows[seed - SEED_START]["initial_top1_service"])
        for seed in EXPECTED_SEEDS
        if rows[seed - SEED_START].get("initial_top1_service") is not None
    )
    initial_top3_sets = {
        info.get("initial_top3")
        for info in valid_popularity.values()
        if info.get("initial_top3") is not None
    }
    initial_top3_service_coverage = len(
        {
            service_id
            for info in valid_popularity.values()
            for service_id in info.get("initial_top3", ())
        }
    )
    same_evolution_different_label_realization = (
        len(exact_pop_paths) > 1 and len(canonical_pop_paths) == 1
    )
    all_popularity_paths_identical = len(exact_pop_paths) == 1 and len(valid_popularity) == EXPECTED_COUNT

    missing_files_by_seed = {
        row["seed"]: row["missing_files"].split(";")
        for row in rows
        if row["episode_found"] == "YES" and row["missing_files"]
    }
    pool_complete = (
        len(found_seeds) == EXPECTED_COUNT
        and not missing_seeds
        and not missing_files_by_seed
    )

    train = manifest.get("train", {})
    validation = manifest.get("validation", {})
    test = manifest.get("test", {})
    train_set = make_seed_set(train)
    validation_set = make_seed_set(validation)
    test_set = make_seed_set(test)
    train_validation_overlap = sorted(train_set & validation_set)
    train_test_overlap = sorted(train_set & test_set)
    validation_test_overlap = sorted(validation_set & test_set)
    seed_leakage = "YES" if any((train_validation_overlap, train_test_overlap, validation_test_overlap)) else "NO"
    split_directories_exist = all(
        (ROOT / "environment_data" / split.get("directory", "")).is_dir()
        for split in (train, validation, test)
    )
    expected_split_ranges = {
        "train": (SEED_START, SEED_END, EXPECTED_COUNT),
        "validation": (30001, 30030, 30),
        "test": (40001, 40100, 100),
    }
    for split_name, expected_range in expected_split_ranges.items():
        split = manifest.get(split_name, {})
        observed = (split.get("seed_start"), split.get("seed_end"), split.get("count"))
        if observed != expected_range:
            contract_issues.append(
                f"manifest {split_name} range/count={observed}, expected={expected_range}"
            )
        if len(make_seed_set(split)) != split.get("count"):
            contract_issues.append(f"manifest {split_name} seed count does not match its inclusive range")
    if not split_directories_exist:
        contract_issues.append("one or more split directory names are missing")

    actual_config_hash = hashlib.sha256(config_bytes).hexdigest()
    manifest_config_hash = str(manifest.get("generator_config_sha256", "")).lower()
    config_hash_differs_from_manifest = (
        bool(manifest_config_hash) and actual_config_hash.lower() != manifest_config_hash
    )
    contract_issues = sorted(set(contract_issues))
    fixed_contract_consistent = (
        pool_complete
        and not contract_issues
        and catalog_hash_matches_manifest
        and catalog_service_ids == expected_catalog_ids
        and role_metadata_checked == EXPECTED_COUNT
    )

    vehicle_mean_speed_values = [row["vehicle_mean_speed_mps"] for row in rows if row["vehicle_mean_speed_mps"] is not None]
    vehicle_speed_std_values = [row["vehicle_std_speed_mps"] for row in rows if row["vehicle_std_speed_mps"] is not None]
    vehicle_acceleration_values = [row["vehicle_mean_acceleration_mps2"] for row in rows if row["vehicle_mean_acceleration_mps2"] is not None]
    episode_min_speed_values = [row["vehicle_min_speed_mps"] for row in rows if row["vehicle_min_speed_mps"] is not None]
    episode_max_speed_values = [row["vehicle_max_speed_mps"] for row in rows if row["vehicle_max_speed_mps"] is not None]
    position_dispersion_values = [row["mean_position_dispersion_m"] for row in rows if row["mean_position_dispersion_m"] is not None]
    travel_distance_values = [row["mean_vehicle_travel_distance_m"] for row in rows if row["mean_vehicle_travel_distance_m"] is not None]
    task_count_values = [row["total_task_count"] for row in rows if row["total_task_count"] is not None]
    arrival_rate_values = [row["realized_arrival_rate"] for row in rows if row["realized_arrival_rate"] is not None]
    input_size_values = [row["mean_input_size"] for row in rows if row["mean_input_size"] is not None]
    input_size_std_values = [row["std_input_size"] for row in rows if row["std_input_size"] is not None]
    cpu_cycle_values = [row["mean_cpu_cycles"] for row in rows if row["mean_cpu_cycles"] is not None]
    cpu_cycle_std_values = [row["std_cpu_cycles"] for row in rows if row["std_cpu_cycles"] is not None]
    top1_change_values = [row["top1_change_count"] for row in rows if row["top1_change_count"] is not None]
    top3_count_values = [row["top3_turnover_count"] for row in rows if row["top3_turnover_count"] is not None]
    top3_rate_values = [row["top3_turnover_rate"] for row in rows if row["top3_turnover_rate"] is not None]
    temporal_pop_drift_values = [row["pop_rho_tv_drift_mean"] for row in rows if row["pop_rho_tv_drift_mean"] is not None]
    temporal_rank_drift_values = [row["pop_rank_mean_abs_change"] for row in rows if row["pop_rank_mean_abs_change"] is not None]
    pop_mae_values = [row["popularity_prediction_mae"] for row in rows if row["popularity_prediction_mae"] is not None]
    pop_rmse_values = [row["popularity_prediction_rmse"] for row in rows if row["popularity_prediction_rmse"] is not None]
    position_mae_values = [row["position_prediction_mae_m"] for row in rows if row["position_prediction_mae_m"] is not None]
    position_rmse_values = [row["position_prediction_rmse_m"] for row in rows if row["position_prediction_rmse_m"] is not None]

    position_error_rows = sum(row["position_prediction_error_rows"] for row in rows)
    position_unmatched_rows = sum(row["position_prediction_unmatched_in_range_rows"] for row in rows)
    position_alignment_bad_rows = sum(row["position_prediction_alignment_bad_rows"] for row in rows)
    position_out_of_episode_rows = sum(row["position_prediction_out_of_episode_rows"] for row in rows)
    if position_error_rows and position_unmatched_rows == 0 and position_alignment_bad_rows == 0:
        position_prediction_check = "COMPUTABLE_FROM_PERSISTED_SCHEMA"
    else:
        position_prediction_check = "NOT_COMPUTABLE_FROM_PERSISTED_SCHEMA"

    complete_for_variation = pool_complete and full_unique_count == EXPECTED_COUNT
    vehicle_metrics_vary = (
        len({round(value, 9) for value in vehicle_mean_speed_values}) > 1
        or len({round(value, 9) for value in vehicle_speed_std_values}) > 1
        or len({round(value, 6) for value in position_dispersion_values}) > 1
    )
    vehicle_variation = (
        "NOT_CONFIRMED"
        if not vehicle_mean_speed_values or not position_dispersion_values
        else "CONFIRMED"
        if hash_unique_counts["trajectory"] > 1 and vehicle_metrics_vary
        else "LIMITED"
    )
    task_hashes_vary = hash_unique_counts["tasks"] > 1
    service_distribution_varies = bool(nonidentical_js)
    task_variation = (
        "NOT_CONFIRMED"
        if not task_count_values
        else "CONFIRMED"
        if task_hashes_vary and service_distribution_varies
        else "LIMITED"
    )
    if not valid_popularity:
        popularity_variation = "NOT_CONFIRMED"
    elif len(exact_pop_paths) > 1 and len(canonical_pop_paths) > 1:
        popularity_variation = "CONFIRMED"
    elif len(exact_pop_paths) > 1:
        popularity_variation = "LIMITED"
    else:
        popularity_variation = "NOT_CONFIRMED"
    if not pop_mae_values:
        prediction_variation = "NOT_COMPUTABLE"
    elif len({round(value, 12) for value in pop_mae_values}) > 1:
        prediction_variation = "CONFIRMED"
    else:
        prediction_variation = "LIMITED"

    near_fraction = (
        near_identical_js_pair_count / len(js_values) if js_values else 1.0
    )
    service_variation_reasonable = bool(js_values) and near_fraction < 0.5
    repeated_seed_count = sum(len(group) for group in signature_duplicate_groups)
    if not pool_complete or full_unique_count == 0 or not service_distributions or not valid_popularity:
        overall_diversity = "INCONCLUSIVE"
    elif repeated_seed_count >= EXPECTED_COUNT / 2:
        overall_diversity = "INSUFFICIENT"
    elif (
        realization_duplication == "NO"
        and vehicle_variation == "CONFIRMED"
        and task_variation == "CONFIRMED"
        and service_variation_reasonable
        and popularity_variation == "CONFIRMED"
        and fixed_contract_consistent
    ):
        overall_diversity = "SUFFICIENT"
    else:
        overall_diversity = "LIMITED"

    selected_contract_keys = (
        "simulation.dt_s",
        "simulation.warmup_time_s",
        "simulation.episode_time_s",
        "simulation.num_research_vehicles",
        "simulation.num_service_vehicles",
        "simulation.num_task_vehicles",
        "service.count",
        "task.arrival_probability",
        "task.popularity_zeta",
        "task.popularity_epoch_length",
        "task.popularity_changes_per_epoch",
        "prediction.vehicle.horizon",
        "prediction.vehicle.sigma_position_m",
        "prediction.vehicle.sigma_speed_mps",
        "prediction.popularity.history_length",
        "scenario.num_scenarios",
        "scenario.cross_vehicle_residual_sampling",
    )
    contract_values = [
        (key, get_config_value(config, key))
        for key in selected_contract_keys
        if get_config_value(config, key) is not None
    ]
    for key, value in flatten_scalars(get_config_value(config, "nodes"), "nodes"):
        contract_values.append((key, value))

    results = {
        "rows": rows,
        "found_seeds": found_seeds,
        "missing_seeds": missing_seeds,
        "unexpected_seeds": sorted(unexpected_seeds),
        "missing_files_by_seed": missing_files_by_seed,
        "pool_complete": "YES" if pool_complete else "NO",
        "hashes": hashes,
        "hash_unique_counts": hash_unique_counts,
        "full_unique_count": full_unique_count,
        "signature_duplicate_groups": signature_duplicate_groups,
        "realization_duplication": realization_duplication,
        "research_set_unique_count": len(research_sets),
        "sv_set_unique_count": len(sv_sets),
        "tv_set_unique_count": len(tv_sets),
        "service_distributions": service_distributions,
        "js_values": js_values,
        "js_summary": js_summary,
        "near_identical_js_pair_count": near_identical_js_pair_count,
        "initial_top1_frequency": dict(sorted(initial_top1_frequency.items(), key=lambda item: int(item[0]))),
        "initial_top3_diversity": len(initial_top3_sets),
        "initial_top3_service_coverage": initial_top3_service_coverage,
        "popularity_exact_path_unique_count": len(exact_pop_paths),
        "popularity_canonical_path_unique_count": len(canonical_pop_paths),
        "all_popularity_paths_identical": "YES" if all_popularity_paths_identical else "NO",
        "same_evolution_different_label_realization": "YES" if same_evolution_different_label_realization else "NO",
        "popularity_schema": sorted(
            {column for info in popularity_info.values() for column in info.get("schema", [])}
        ),
        "vehicle_mean_speed_dist": numeric_summary(vehicle_mean_speed_values),
        "vehicle_speed_std_dist": numeric_summary(vehicle_speed_std_values),
        "mean_acceleration_dist": numeric_summary(vehicle_acceleration_values),
        "episode_min_speed_dist": numeric_summary(episode_min_speed_values),
        "episode_max_speed_dist": numeric_summary(episode_max_speed_values),
        "position_dispersion_dist": numeric_summary(position_dispersion_values),
        "travel_distance_dist": numeric_summary(travel_distance_values),
        "task_count_dist": numeric_summary(task_count_values),
        "arrival_rate_dist": numeric_summary(arrival_rate_values),
        "mean_input_size_dist": numeric_summary(input_size_values),
        "input_size_std_dist": numeric_summary(input_size_std_values),
        "mean_cpu_cycles_dist": numeric_summary(cpu_cycle_values),
        "cpu_cycles_std_dist": numeric_summary(cpu_cycle_std_values),
        "top1_change_dist": numeric_summary(top1_change_values),
        "top3_turnover_count_dist": numeric_summary(top3_count_values),
        "top3_turnover_rate_dist": numeric_summary(top3_rate_values),
        "temporal_popularity_drift_dist": numeric_summary(temporal_pop_drift_values),
        "temporal_rank_drift_dist": numeric_summary(temporal_rank_drift_values),
        "pop_prediction_mae_dist": numeric_summary(pop_mae_values),
        "pop_prediction_rmse_dist": numeric_summary(pop_rmse_values),
        "position_prediction_mae_dist": numeric_summary(position_mae_values),
        "position_prediction_rmse_dist": numeric_summary(position_rmse_values),
        "pop_prediction_error_rows": sum(row["popularity_prediction_error_rows"] for row in rows),
        "position_error_rows": position_error_rows,
        "position_unmatched_rows": position_unmatched_rows,
        "position_alignment_bad_rows": position_alignment_bad_rows,
        "position_out_of_episode_rows": position_out_of_episode_rows,
        "position_prediction_check": position_prediction_check,
        "mean_research_count": statistics.fmean(row["research_vehicle_count"] for row in rows if row["research_vehicle_count"] is not None) if research_sets else None,
        "mean_sv_count": statistics.fmean(row["sv_count"] for row in rows if row["sv_count"] is not None) if sv_sets else None,
        "mean_tv_count": statistics.fmean(row["tv_count"] for row in rows if row["tv_count"] is not None) if tv_sets else None,
        "num_services": num_services,
        "catalog_hash_matches_manifest": "YES" if catalog_hash_matches_manifest else "NO",
        "fixed_contract_consistent": "YES" if fixed_contract_consistent else "NO",
        "contract_issues": contract_issues,
        "contract_values": contract_values,
        "config_hash_differs_from_manifest": config_hash_differs_from_manifest,
        "train_validation_overlap": train_validation_overlap,
        "train_test_overlap": train_test_overlap,
        "validation_test_overlap": validation_test_overlap,
        "seed_leakage": seed_leakage,
        "split_directories_exist": split_directories_exist,
        "vehicle_variation": vehicle_variation,
        "task_variation": task_variation,
        "popularity_variation": popularity_variation,
        "prediction_variation": prediction_variation,
        "overall_diversity": overall_diversity,
        "service_variation_reasonable": service_variation_reasonable,
    }
    report = render_markdown(results, config, manifest, catalog)
    summary = build_summary_csv(rows, num_services)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents replacing user files even if a path appears after preflight.
    created: list[Path] = []
    try:
        with REPORT_PATH.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(report)
        created.append(REPORT_PATH)
        with SUMMARY_PATH.open("x", encoding="utf-8", newline="") as stream:
            stream.write(summary)
        created.append(SUMMARY_PATH)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise

    print(f"EXPECTED_EPISODES={EXPECTED_COUNT}")
    print(f"FOUND_EPISODES={len(found_seeds)}")
    print(f"MISSING_EPISODES={len(missing_seeds)}")
    print(f"UNEXPECTED_EPISODES={len(unexpected_seeds)}")
    print(f"POOL_COMPLETE={results['pool_complete']}")
    print(f"FULL_REALIZATION_UNIQUE_COUNT={full_unique_count}/{EXPECTED_COUNT}")
    print(f"REALIZATION_DUPLICATION={realization_duplication}")
    print(f"TRAJECTORY_UNIQUE_COUNT={hash_unique_counts['trajectory']}/{EXPECTED_COUNT}")
    print(f"TASKS_UNIQUE_COUNT={hash_unique_counts['tasks']}/{EXPECTED_COUNT}")
    print(f"POPULARITY_UNIQUE_COUNT={hash_unique_counts['popularity']}/{EXPECTED_COUNT}")
    print(f"ARRIVAL_RATE_MEAN={fmt(results['arrival_rate_dist']['mean'])}")
    print(f"ARRIVAL_RATE_STD={fmt(results['arrival_rate_dist']['std'])}")
    print(f"SERVICE_JS_MEDIAN={fmt(js_summary['median'])}")
    print(f"SERVICE_JS_P90={fmt(js_summary['p90'])}")
    print(f"POPULARITY_TEMPORAL_DRIFT_MEAN={fmt(results['temporal_popularity_drift_dist']['mean'])}")
    print(f"POPULARITY_PREDICTION_MAE_MEAN={fmt(results['pop_prediction_mae_dist']['mean'])}")
    print(f"POSITION_PREDICTION_MAE_MEAN={fmt(results['position_prediction_mae_dist']['mean'])}")
    print(f"POSITION_PREDICTION_ERROR_CHECK={position_prediction_check}")
    print(f"VEHICLE_ENVIRONMENT_VARIATION={vehicle_variation}")
    print(f"TASK_ENVIRONMENT_VARIATION={task_variation}")
    print(f"POPULARITY_VARIATION={popularity_variation}")
    print(f"PREDICTION_DIFFICULTY_VARIATION={prediction_variation}")
    print(f"FIXED_EXPERIMENT_CONTRACT_CONSISTENT={results['fixed_contract_consistent']}")
    print(f"TRAIN_VALIDATION_TEST_SEED_LEAKAGE={seed_leakage}")
    print(f"TRAIN_ENVIRONMENT_DIVERSITY={overall_diversity}")
    print(f"REPORT={REPORT_PATH.relative_to(ROOT)}")
    print(f"SUMMARY={SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
