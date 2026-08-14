#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a denser and longer SUMO passenger-trip pool by calling SUMO's
official tools/randomTrips.py.

Place this file at:
    adaptive_cache_mec/data/generate_sumo_demand.py

Default input:
    data/sumo/map/osm.net.xml.gz

Default output (replaces the demand file referenced by osm.sumocfg):
    data/sumo/map/osm.passenger.trips.xml

A reproducibility record is also written to:
    data/sumo/processed/demand_generation_config.json

Important:
- This script does NOT implement its own random-trip generator.
- It calls SUMO's official randomTrips.py.
- Temporary validation files are created inside a TemporaryDirectory and
  are automatically removed when the command finishes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def find_project_root() -> Path:
    script = Path(__file__).resolve()

    if script.parent.name == "data":
        root = script.parent.parent
        if (root / "data" / "sumo").is_dir():
            return root

    cwd = Path.cwd().resolve()
    if (cwd / "data" / "sumo").is_dir():
        return cwd

    raise FileNotFoundError(
        "Cannot locate project root. Expected data/sumo under the project root."
    )


def find_sumo_home(explicit: str | None) -> Path:
    candidates: list[Path] = []

    if explicit:
        candidates.append(Path(explicit).expanduser())

    env_home = os.environ.get("SUMO_HOME")
    if env_home:
        candidates.append(Path(env_home).expanduser())

    sumo_exe = shutil.which("sumo")
    if sumo_exe:
        exe = Path(sumo_exe).resolve()
        if exe.parent.name.lower() == "bin":
            candidates.append(exe.parent.parent)
        else:
            candidates.append(exe.parent)

    for candidate in candidates:
        candidate = candidate.resolve()
        random_trips = candidate / "tools" / "randomTrips.py"
        if random_trips.is_file():
            return candidate

    raise FileNotFoundError(
        "SUMO installation not found. Set SUMO_HOME or pass --sumo-home."
    )


def load_network_info(net_file: Path, sumo_home: Path) -> dict:
    tools_dir = str(sumo_home / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)

    import sumolib  # SUMO official Python library

    net = sumolib.net.readNet(str(net_file))
    xmin, ymin, xmax, ymax = net.getBoundary()

    width = xmax - xmin
    height = ymax - ymin
    diagonal = math.hypot(width, height)

    return {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "width_m": width,
        "height_m": height,
        "diagonal_m": diagonal,
    }


def count_trips(path: Path) -> tuple[int, list[tuple[str, float]]]:
    tree = ET.parse(path)
    root = tree.getroot()

    trips = []
    for elem in root:
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag == "trip":
            trip_id = elem.get("id")
            depart = elem.get("depart")
            if trip_id is None or depart is None:
                raise ValueError("Generated <trip> is missing id or depart.")
            trips.append((trip_id, float(depart)))

    return len(trips), trips


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Call SUMO official randomTrips.py to generate a dense, "
            "longer passenger-trip pool."
        )
    )
    parser.add_argument(
        "--vehicle-count",
        type=int,
        default=60,
        help="Requested number of vehicles. Default: 60.",
    )
    parser.add_argument(
        "--begin",
        type=float,
        default=0.0,
        help="First departure time in seconds. Default: 0.",
    )
    parser.add_argument(
        "--departure-window",
        type=float,
        default=60.0,
        help=(
            "Length of departure-generation window in seconds. "
            "Default: 60.0."
        ),
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=4000.0,
        help=(
            "Minimum SUMO randomTrips geometric trip distance in meters. "
            "With intermediate waypoints this is the sum of Euclidean "
            "distances between consecutive sampled points. Default: 4000."
        ),
    )
    parser.add_argument(
        "--intermediate",
        type=int,
        default=2,
        help="Number of intermediate waypoints. Default: 2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="SUMO randomTrips seed. Default: 42.",
    )
    parser.add_argument(
        "--maxtries",
        type=int,
        default=1000,
        help=(
            "Maximum sampling attempts for a trip satisfying the distance "
            "constraint. Default: 1000."
        ),
    )
    parser.add_argument(
        "--sumo-home",
        type=str,
        default=None,
        help="SUMO installation root. Otherwise SUMO_HOME/PATH is used.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing osm.passenger.trips.xml.",
    )
    args = parser.parse_args()

    if args.vehicle_count <= 0:
        raise ValueError("--vehicle-count must be > 0.")
    if args.departure_window <= 0:
        raise ValueError("--departure-window must be > 0.")
    if args.min_distance < 0:
        raise ValueError("--min-distance must be >= 0.")
    if args.intermediate < 0:
        raise ValueError("--intermediate must be >= 0.")
    if args.maxtries <= 0:
        raise ValueError("--maxtries must be > 0.")

    root = find_project_root()
    sumo_home = find_sumo_home(args.sumo_home)

    net_file = root / "data" / "sumo" / "map" / "osm.net.xml.gz"
    target_trip_file = (
        root / "data" / "sumo" / "map" / "osm.passenger.trips.xml"
    )
    processed_dir = root / "data" / "sumo" / "processed"
    metadata_file = processed_dir / "demand_generation_config.json"

    if not net_file.is_file():
        raise FileNotFoundError(f"Network file not found: {net_file}")

    if target_trip_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"Demand file already exists: {target_trip_file}\n"
            "Use --overwrite to replace it."
        )

    random_trips = sumo_home / "tools" / "randomTrips.py"

    end = args.begin + args.departure_window

    # randomTrips generates equidistant departures with this period.
    period = args.departure_window / args.vehicle_count

    net_info = load_network_info(net_file, sumo_home)

    print("=" * 76)
    print("SUMO demand generation")
    print("=" * 76)
    print(f"SUMO_HOME          : {sumo_home}")
    print(f"Network            : {net_file}")
    print(
        "Network size       : "
        f"{net_info['width_m']:.2f} m x {net_info['height_m']:.2f} m"
    )
    print(f"Network diagonal   : {net_info['diagonal_m']:.2f} m")
    print(f"Requested vehicles : {args.vehicle_count}")
    print(f"Departure interval : [{args.begin:.2f}, {end:.2f}) s")
    print(f"Period             : {period:.6f} s/vehicle")
    print(f"Min distance       : {args.min_distance:.2f} m")
    print(f"Intermediate points: {args.intermediate}")
    print(f"Seed               : {args.seed}")
    print("=" * 76)

    # Use a temporary directory so randomTrips/duarouter validation artifacts
    # never pollute the project tree.
    with tempfile.TemporaryDirectory(prefix="sumo_randomTrips_") as tmp:
        tmp_dir = Path(tmp)
        tmp_trip_file = tmp_dir / "osm.passenger.trips.xml"
        tmp_route_file = tmp_dir / "validation.rou.xml"

        command = [
            sys.executable,
            str(random_trips),
            "-n",
            str(net_file),
            "-o",
            str(tmp_trip_file),
            "-r",
            str(tmp_route_file),
            "-b",
            str(args.begin),
            "-e",
            str(end),
            "-p",
            str(period),
            "-s",
            str(args.seed),
            "--min-distance",
            str(args.min_distance),
            "--intermediate",
            str(args.intermediate),
            "--maxtries",
            str(args.maxtries),
            "--vehicle-class",
            "passenger",
            "--validate",
        ]

        print("\nCalling SUMO official randomTrips.py ...")
        subprocess.run(command, check=True)

        if not tmp_trip_file.is_file():
            raise RuntimeError(
                "randomTrips.py finished but did not generate the trip file."
            )

        actual_count, trips = count_trips(tmp_trip_file)

        if actual_count != args.vehicle_count:
            raise RuntimeError(
                "Generated trip count does not match the requested count: "
                f"requested={args.vehicle_count}, actual={actual_count}. "
                "The existing project demand file was NOT replaced."
            )

        departs = [depart for _, depart in trips]
        first_depart = min(departs)
        last_depart = max(departs)

        target_trip_file.parent.mkdir(parents=True, exist_ok=True)

        # Replace only after official generation and our basic consistency
        # checks have succeeded.
        os.replace(tmp_trip_file, target_trip_file)

    processed_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "generator": "SUMO tools/randomTrips.py",
        "sumo_home": str(sumo_home),
        "network_file": str(net_file),
        "trip_file": str(target_trip_file),
        "vehicle_count_requested": args.vehicle_count,
        "vehicle_count_generated": actual_count,
        "begin_s": args.begin,
        "end_s": end,
        "departure_window_s": args.departure_window,
        "period_s": period,
        "first_depart_s": first_depart,
        "last_depart_s": last_depart,
        "min_distance_m": args.min_distance,
        "intermediate_waypoints": args.intermediate,
        "seed": args.seed,
        "maxtries": args.maxtries,
        "network": net_info,
    }

    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nGeneration succeeded.")
    print(f"Trip file          : {target_trip_file}")
    print(f"Generated vehicles : {actual_count}")
    print(
        f"Actual depart range: {first_depart:.2f} ... {last_depart:.2f} s"
    )
    print(f"Metadata           : {metadata_file}")

    print("\nFirst 10 trips:")
    for trip_id, depart in trips[:10]:
        print(f"  id={trip_id}, depart={depart:.2f}")

    print(
        "\nNext: run SUMO again, regenerate FCD, export the 1 s trace, "
        "prepare mobility, and rerun episode-window capacity analysis."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
