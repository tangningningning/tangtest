#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mobility trace audit for the adaptive_cache_mec project.

Place at:
    adaptive_cache_mec/data/audit_mobility.py

Default input:
    data/sumo/processed/mobility_trace.csv

Outputs:
    data/sumo/processed/mobility_audit_summary.csv
    data/sumo/processed/vehicle_speed_stats.csv

This script performs descriptive auditing only. It does NOT decide whether
a speed is "valid" or automatically modify/filter SUMO trajectories.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean


def project_root_from_script() -> Path:
    script = Path(__file__).resolve()

    if script.parent.name == "data":
        root = script.parent.parent
        if (root / "data" / "sumo").is_dir():
            return root

    cwd = Path.cwd().resolve()
    if (cwd / "data" / "sumo").is_dir():
        return cwd

    raise FileNotFoundError(
        "Cannot locate project root. Expected data/sumo."
    )


def percentile(sorted_values: list[float], q: float) -> float:
    """
    Linear-interpolated percentile, q in [0, 1].
    """
    if not sorted_values:
        raise ValueError("Cannot compute percentile of an empty list.")

    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1].")

    if len(sorted_values) == 1:
        return sorted_values[0]

    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return sorted_values[lo]

    weight = pos - lo
    return (
        sorted_values[lo] * (1.0 - weight)
        + sorted_values[hi] * weight
    )


def load_trace(path: Path) -> list[dict]:
    rows: list[dict] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "slot",
            "time_s",
            "vehicle_id",
            "x",
            "y",
            "speed_mps",
        }

        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Input file missing columns: {sorted(missing)}"
            )

        for line_no, row in enumerate(reader, start=2):
            try:
                rows.append(
                    {
                        "slot": int(row["slot"]),
                        "time_s": float(row["time_s"]),
                        "vehicle_id": str(row["vehicle_id"]),
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "speed_mps": float(row["speed_mps"]),
                    }
                )
            except ValueError as exc:
                raise ValueError(
                    f"Failed to parse line {line_no}: {row}"
                ) from exc

    if not rows:
        raise ValueError(f"Mobility trace is empty: {path}")

    return rows


def numeric_id_key(vehicle_id: str):
    try:
        return (0, int(vehicle_id))
    except ValueError:
        return (1, vehicle_id)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit processed SUMO mobility trace."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Input mobility trace. Default: "
            "data/sumo/processed/mobility_trace.csv"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing audit outputs.",
    )
    args = parser.parse_args()

    root = project_root_from_script()

    input_path = (
        args.input
        if args.input is not None
        else root / "data" / "sumo" / "processed" / "mobility_trace.csv"
    )

    if not input_path.is_absolute():
        input_path = (root / input_path).resolve()
    else:
        input_path = input_path.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input trace not found: {input_path}")

    processed_dir = root / "data" / "sumo" / "processed"
    summary_path = processed_dir / "mobility_audit_summary.csv"
    vehicle_stats_path = processed_dir / "vehicle_speed_stats.csv"

    existing = [
        p for p in (summary_path, vehicle_stats_path)
        if p.exists()
    ]

    if existing and not args.overwrite:
        raise FileExistsError(
            "Audit output already exists:\n"
            + "\n".join(str(p) for p in existing)
            + "\nUse --overwrite to regenerate."
        )

    rows = load_trace(input_path)

    speeds = sorted(row["speed_mps"] for row in rows)

    if speeds[0] < -1e-9:
        raise ValueError(
            f"Negative speed found: minimum={speeds[0]}"
        )

    vehicle_rows: dict[str, list[dict]] = {}
    for row in rows:
        vehicle_rows.setdefault(row["vehicle_id"], []).append(row)

    global_stats = {
        "records": len(rows),
        "vehicles": len(vehicle_rows),
        "slot_min": min(r["slot"] for r in rows),
        "slot_max": max(r["slot"] for r in rows),
        "x_min_m": min(r["x"] for r in rows),
        "x_max_m": max(r["x"] for r in rows),
        "y_min_m": min(r["y"] for r in rows),
        "y_max_m": max(r["y"] for r in rows),
        "speed_min_mps": speeds[0],
        "speed_p05_mps": percentile(speeds, 0.05),
        "speed_p25_mps": percentile(speeds, 0.25),
        "speed_median_mps": percentile(speeds, 0.50),
        "speed_p75_mps": percentile(speeds, 0.75),
        "speed_p95_mps": percentile(speeds, 0.95),
        "speed_p99_mps": percentile(speeds, 0.99),
        "speed_max_mps": speeds[-1],
        "speed_mean_mps": mean(speeds),
    }

    # Purely descriptive thresholds; no automatic acceptance rule.
    thresholds_mps = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]

    for threshold in thresholds_mps:
        count = sum(speed >= threshold for speed in speeds)
        global_stats[f"fraction_ge_{threshold:g}_mps"] = count / len(speeds)

    vehicle_stats = []

    for vehicle_id, vrows in vehicle_rows.items():
        vspeeds = sorted(r["speed_mps"] for r in vrows)

        vehicle_stats.append(
            {
                "vehicle_id": vehicle_id,
                "first_slot": min(r["slot"] for r in vrows),
                "last_slot": max(r["slot"] for r in vrows),
                "records": len(vrows),
                "mean_speed_mps": mean(vspeeds),
                "median_speed_mps": percentile(vspeeds, 0.50),
                "p95_speed_mps": percentile(vspeeds, 0.95),
                "max_speed_mps": vspeeds[-1],
                "mean_speed_kmh": mean(vspeeds) * 3.6,
                "p95_speed_kmh": percentile(vspeeds, 0.95) * 3.6,
                "max_speed_kmh": vspeeds[-1] * 3.6,
            }
        )

    vehicle_stats.sort(
        key=lambda r: (
            -r["max_speed_mps"],
            numeric_id_key(r["vehicle_id"]),
        )
    )

    processed_dir.mkdir(parents=True, exist_ok=True)

    with summary_path.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])

        for key, value in global_stats.items():
            if isinstance(value, float):
                writer.writerow([key, f"{value:.6f}"])
            else:
                writer.writerow([key, value])

    with vehicle_stats_path.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        fieldnames = [
            "vehicle_id",
            "first_slot",
            "last_slot",
            "records",
            "mean_speed_mps",
            "median_speed_mps",
            "p95_speed_mps",
            "max_speed_mps",
            "mean_speed_kmh",
            "p95_speed_kmh",
            "max_speed_kmh",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in vehicle_stats:
            out = dict(row)
            for key in fieldnames:
                if isinstance(out[key], float):
                    out[key] = f"{out[key]:.6f}"
            writer.writerow(out)

    print("=" * 76)
    print("Mobility audit finished")
    print("=" * 76)
    print(f"Input       : {input_path}")
    print(f"Records     : {global_stats['records']}")
    print(f"Vehicles    : {global_stats['vehicles']}")
    print(
        f"Slot range  : {global_stats['slot_min']} ... "
        f"{global_stats['slot_max']}"
    )

    print("\nGlobal speed distribution:")
    print("-" * 76)

    labels = [
        ("min", "speed_min_mps"),
        ("P05", "speed_p05_mps"),
        ("P25", "speed_p25_mps"),
        ("median", "speed_median_mps"),
        ("mean", "speed_mean_mps"),
        ("P75", "speed_p75_mps"),
        ("P95", "speed_p95_mps"),
        ("P99", "speed_p99_mps"),
        ("max", "speed_max_mps"),
    ]

    for label, key in labels:
        mps = global_stats[key]
        print(
            f"{label:>8}: "
            f"{mps:8.3f} m/s   "
            f"{mps * 3.6:8.2f} km/h"
        )

    print("\nFraction of records at or above threshold:")
    print("-" * 76)

    for threshold in thresholds_mps:
        fraction = global_stats[f"fraction_ge_{threshold:g}_mps"]
        print(
            f">= {threshold:5.1f} m/s "
            f"({threshold * 3.6:6.1f} km/h): "
            f"{fraction:8.4%}"
        )

    print("\nTop 10 vehicles by maximum speed:")
    print("-" * 76)
    print(
        f"{'vehicle_id':>12} "
        f"{'mean km/h':>12} "
        f"{'P95 km/h':>12} "
        f"{'max km/h':>12} "
        f"{'records':>9}"
    )

    for row in vehicle_stats[:10]:
        print(
            f"{row['vehicle_id']:>12} "
            f"{row['mean_speed_kmh']:>12.2f} "
            f"{row['p95_speed_kmh']:>12.2f} "
            f"{row['max_speed_kmh']:>12.2f} "
            f"{row['records']:>9}"
        )

    print("\nFiles:")
    print(f"  {summary_path}")
    print(f"  {vehicle_stats_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
