#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate continuous, raw SUMO traffic datasets for Train/Validation/Test.

This entry point deliberately stops at environment data.  It invokes the
official ``randomTrips.py`` and headless SUMO CLI, and does not import or run the
training stack, Matching, or KKT.

SUMO's saved states do not serialize all lane-change/car-following model
internals.  After an interrupted realization this generator therefore
rebuilds that realization from its recorded route file and fixed seeds instead
of claiming bit-exact TraCI/SUMO state restoration.  Completed artifacts are
published atomically and attested by per-realization manifests.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common import project_config  # noqa: E402
from scenario.realization import derive_episode_seeds  # noqa: E402


DATASET_SCHEMA_VERSION = "continuous_sumo_dataset_v1"
EPISODE_SCHEMA_VERSION = "continuous_sumo_episode_v1"
PROGRESS_SCHEMA_VERSION = "continuous_sumo_progress_v1"
TRAJECTORY_FORMAT_VERSION = "sumo_fcd_xml_gzip_v1"
ACTIVITY_FORMAT_VERSION = "vehicle_activity_csv_gzip_v1"
CHECKPOINT_PERIOD_S = 600.0
FORMAL_RANDOM_TRIPS_PERIOD_S = 6.0
FCD_SIZE_BENCHMARK_PERIOD_S = 1.0
FCD_SIZE_BENCHMARK_BYTES_PER_SIM_SECOND = 38_000
DEFAULT_CONFIG = "configs/dqn/online_joint_dqn.toml"
DEFAULT_SPLIT_MANIFEST = "environment_data/data_split_manifest.json"
DEFAULT_OUTPUT_DIR = "environment_data/continuous_v1"
VEHICLE_ACTIVITY_FIELDS = (
    "vehicle_id",
    "scheduled_departure_s",
    "first_observed_s",
    "last_observed_s",
    "tripinfo_departure_s",
    "arrival_s",
    "route_status",
    "vaporized_reason",
    "activity_at_observation_end",
)


@dataclass(frozen=True)
class EpisodeSpec:
    split: str
    episode_seed: int
    sumo_seed: int
    traffic_seed: int
    duration_s: float
    warmup_s: float
    formal_start_s: float
    formal_end_s: float
    departure_begin_s: float
    departure_end_s: float
    episode_name: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def configured_random_trips_period_s(config: dict[str, Any]) -> float:
    """Read and enforce the formal vehicle-departure interval."""
    value = project_config.get_config_value(
        config,
        "simulation.random_trips_period_s",
    )
    if isinstance(value, bool) or value is None:
        raise ValueError("simulation.random_trips_period_s must be explicitly configured")
    period_s = float(value)
    if not math.isfinite(period_s) or period_s <= 0.0:
        raise ValueError("simulation.random_trips_period_s must be finite and positive")
    if not math.isclose(
        period_s,
        FORMAL_RANDOM_TRIPS_PERIOD_S,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "formal generation requires random_trips_period_s="
            f"{FORMAL_RANDOM_TRIPS_PERIOD_S:.1f}; actual={period_s}"
        )
    return period_s


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _sumo_environment() -> tuple[Path, Path, Path, str]:
    sumo_home_value = os.environ.get("SUMO_HOME")
    if not sumo_home_value:
        raise EnvironmentError("SUMO_HOME is not set in the active environment.")
    sumo_home = Path(sumo_home_value).expanduser().resolve(strict=True)
    random_trips = sumo_home / "tools" / "randomTrips.py"
    if not random_trips.is_file():
        raise FileNotFoundError(f"SUMO randomTrips.py is missing: {random_trips}")
    candidates = (
        sumo_home / "bin" / "sumo.exe",
        sumo_home / "bin" / "sumo",
        sumo_home / "bin" / "sumo.com",
    )
    sumo_binary = next((path for path in candidates if path.is_file()), None)
    if sumo_binary is None:
        raise FileNotFoundError(f"Could not find the SUMO CLI under {sumo_home / 'bin'}")
    environment = os.environ.copy()
    environment["SUMO_HOME"] = str(sumo_home)
    result = subprocess.run(
        [str(sumo_binary), "--version"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        cwd=PROJECT_ROOT,
    )
    version = (result.stdout or result.stderr).strip().splitlines()[0]
    return sumo_home, sumo_binary, random_trips, version


def _load_source_contract(
    config_path: Path,
    split_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = project_config.parse_toml_file(config_path)
    split_manifest = _load_json(split_manifest_path)
    simulation = config.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError("Formal DQN configuration is missing [simulation].")
    network_value = project_config.get_config_value(config, "paths.sumo_net")
    if not network_value:
        raise ValueError("Formal DQN configuration is missing paths.sumo_net.")
    network_path = _resolve_project_path(str(network_value))
    if not network_path.is_file():
        raise FileNotFoundError(f"Formal SUMO network is missing: {network_path}")
    network_hash = _sha256_file(network_path)
    source_network_hash = split_manifest.get("sumo_network_sha256")
    if source_network_hash and source_network_hash != network_hash:
        raise ValueError(
            "The split manifest and active DQN configuration refer to different "
            f"SUMO networks: manifest={source_network_hash}, current={network_hash}."
        )
    dt_s = float(simulation["dt_s"])
    warmup_s = float(simulation["warmup_time_s"])
    episode_s = float(simulation["episode_time_s"])
    random_trips_period_s = configured_random_trips_period_s(config)
    episode_protocol = project_config.get_episode_data_protocol(config)
    task_timing = project_config.get_cross_slot_task_parameters(config)
    arrival_slots = int(episode_protocol["arrival_horizon_slots"])
    execution_slots = int(episode_protocol["execution_horizon_slots"])
    k_max_slots = int(task_timing["k_max_slots"])
    if execution_slots != arrival_slots + k_max_slots - 1:
        raise ValueError(
            "execution horizon must equal arrival horizon + K_max - 1: "
            f"arrival={arrival_slots}, K_max={k_max_slots}, execution={execution_slots}"
        )
    if not math.isclose(episode_s / dt_s, arrival_slots, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("simulation.episode_time_s must equal the configured arrival horizon.")
    for label, value in (("dt_s", dt_s), ("episode_time_s", episode_s)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"simulation.{label} must be finite and positive.")
    if not math.isfinite(warmup_s) or warmup_s < 0:
        raise ValueError("simulation.warmup_time_s must be finite and nonnegative.")
    if not math.isclose(warmup_s / dt_s, round(warmup_s / dt_s), abs_tol=1e-9):
        raise ValueError("Warm-up duration is not an integer number of SUMO steps.")
    if not math.isclose(episode_s / dt_s, round(episode_s / dt_s), abs_tol=1e-9):
        raise ValueError("Episode duration is not an integer number of SUMO steps.")
    for split in ("train", "validation", "test"):
        contract = split_manifest.get(split)
        if not isinstance(contract, dict):
            raise ValueError(f"Official data split manifest is missing {split!r}.")
        start, end, count = (
            int(contract["seed_start"]),
            int(contract["seed_end"]),
            int(contract["count"]),
        )
        if end < start or count <= 0 or end - start + 1 != count:
            raise ValueError(f"Invalid {split} seed/count contract in split manifest.")
    if int(split_manifest.get("slots_per_realization", -1)) <= 0:
        raise ValueError("Official split manifest is missing slots_per_realization.")
    return config, split_manifest, {
        "dt_s": dt_s,
        "warmup_s": warmup_s,
        "episode_s": episode_s,
        "random_trips_period_s": random_trips_period_s,
        "arrival_slots": arrival_slots,
        "execution_slots": execution_slots,
        "k_max_slots": k_max_slots,
        "network_path": network_path,
        "network_sha256": network_hash,
    }


def _make_specs(
    config: dict[str, Any],
    split_manifest: dict[str, Any],
    physical: dict[str, Any],
    selected_splits: Iterable[str],
) -> list[EpisodeSpec]:
    dt_s = float(physical["dt_s"])
    warmup_s = float(physical["warmup_s"])
    episode_s = float(physical["episode_s"])
    execution_tail_s = (
        int(physical["execution_slots"]) - int(physical["arrival_slots"])
    ) * dt_s
    slots_per_realization = int(split_manifest["slots_per_realization"])
    specs: list[EpisodeSpec] = []
    for split in selected_splits:
        source = split_manifest[split]
        start_seed = int(source["seed_start"])
        if split == "train":
            # One stateful SUMO stream spans the 300 sequential DQN episodes.
            seed_values = (start_seed,)
            duration_s = warmup_s + episode_s * slots_per_realization + execution_tail_s
        else:
            seed_values = range(start_seed, int(source["seed_end"]) + 1)
            duration_s = warmup_s + episode_s + execution_tail_s
        if not math.isclose(duration_s / dt_s, round(duration_s / dt_s), abs_tol=1e-9):
            raise ValueError(f"{split} duration is not aligned to the SUMO time grid.")
        for episode_seed in seed_values:
            derived = derive_episode_seeds(int(episode_seed))
            prefix = "continuous" if split == "train" else "episode"
            specs.append(
                EpisodeSpec(
                    split=split,
                    episode_seed=int(episode_seed),
                    sumo_seed=int(derived["sumo_seed"]),
                    traffic_seed=int(derived["traffic_seed"]),
                    duration_s=duration_s,
                    warmup_s=warmup_s,
                    formal_start_s=warmup_s,
                    formal_end_s=duration_s,
                    departure_begin_s=float(
                        project_config.get_config_value(
                            config, "simulation.random_trips_departure_begin_s", 0.0
                        )
                    ),
                    departure_end_s=duration_s,
                    episode_name=f"{prefix}_seed_{int(episode_seed)}",
                )
            )
    root_seeds: dict[str, set[int]] = {}
    derived_seeds: dict[str, set[int]] = {}
    for split in ("train", "validation", "test"):
        current = [spec for spec in specs if spec.split == split]
        root_seeds[split] = {spec.episode_seed for spec in current}
        derived_seeds[split] = {
            seed
            for spec in current
            for seed in (spec.sumo_seed, spec.traffic_seed)
        }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if root_seeds.get(left, set()) & root_seeds.get(right, set()):
            raise ValueError(f"Episode root seeds overlap between {left} and {right}.")
        if derived_seeds.get(left, set()) & derived_seeds.get(right, set()):
            raise ValueError(f"SUMO/traffic seeds overlap between {left} and {right}.")
    return specs


def _dataset_layout(
    output_root: Path,
    split: str,
    episode_name: str,
) -> Path:
    return output_root / split / episode_name


def _random_trips_parameters(config: dict[str, Any], spec: EpisodeSpec) -> dict[str, Any]:
    sim = config["simulation"]
    return {
        "departure_begin_s": float(
            project_config.get_config_value(
                config, "simulation.random_trips_departure_begin_s", 0.0
            )
        ),
        "departure_end_s": float(spec.departure_end_s),
        "period_s": configured_random_trips_period_s(config),
        "min_distance_m": float(
            project_config.get_config_value(config, "simulation.random_trips_min_distance_m", 4000.0)
        ),
        "intermediate": int(
            project_config.get_config_value(config, "simulation.random_trips_intermediate", 2)
        ),
        "maxtries": int(
            project_config.get_config_value(config, "simulation.random_trips_maxtries", 1000)
        ),
        "vehicle_class": "passenger",
        "sumo_seed": int(spec.sumo_seed),
        "traffic_seed": int(spec.traffic_seed),
        "configured_source_departure_end_s": float(
            sim.get("random_trips_departure_end_s", 60.0)
        ),
    }


def _run_fingerprint(
    config_sha256: str,
    network_sha256: str,
    random_trips_sha256: str,
    sumo_version: str,
    generator_sha256: str,
    spec: EpisodeSpec,
    traffic_parameters: dict[str, Any],
) -> str:
    return _fingerprint(
        {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "config_sha256": config_sha256,
            "network_sha256": network_sha256,
            "randomTrips_sha256": random_trips_sha256,
            "sumo_version": sumo_version,
            "generator_sha256": generator_sha256,
            "split": spec.split,
            "episode_seed": spec.episode_seed,
            "sumo_seed": spec.sumo_seed,
            "traffic_seed": spec.traffic_seed,
            "duration_s": spec.duration_s,
            "dt_s": traffic_parameters["dt_s"],
            "warmup_s": spec.warmup_s,
            "formal_start_s": spec.formal_start_s,
            "formal_end_s": spec.formal_end_s,
            "randomTrips": traffic_parameters,
            "trajectory_format_version": TRAJECTORY_FORMAT_VERSION,
        }
    )


def _create_route_demand(
    *,
    spec: EpisodeSpec,
    parameters: dict[str, Any],
    network_path: Path,
    random_trips: Path,
    sumo_home: Path,
    episode_dir: Path,
    allow_incomplete_rebuild: bool,
) -> dict[str, Any]:
    route_path = episode_dir / "sumo_demand.rou.xml"
    trips_path = episode_dir / "sumo_demand.trips.xml"
    metadata_path = episode_dir / "sumo_demand.meta.json"
    if route_path.exists() or trips_path.exists() or metadata_path.exists():
        if not (route_path.is_file() and trips_path.is_file() and metadata_path.is_file()):
            if not allow_incomplete_rebuild:
                raise RuntimeError(
                    f"Route-demand artifacts are incomplete; refusing to replace them: {episode_dir}"
                )
            for path in (route_path, trips_path, metadata_path):
                if path.is_file():
                    path.unlink()
        else:
            stored = _load_json(metadata_path)
            for key, expected in (
                ("run_fingerprint", parameters["run_fingerprint"]),
                ("network_sha256", parameters["network_sha256"]),
                ("episode_seed", spec.episode_seed),
                ("sumo_seed", spec.sumo_seed),
                ("traffic_seed", spec.traffic_seed),
                ("randomTrips", parameters["randomTrips"]),
            ):
                if stored.get(key) != expected:
                    raise RuntimeError(
                        f"Existing route demand does not match its contract: {episode_dir} field={key}"
                    )
            if stored.get("route_sha256") != _sha256_file(route_path):
                raise RuntimeError(f"Existing route file hash mismatch: {route_path}")
            if stored.get("trips_sha256") != _sha256_file(trips_path):
                raise RuntimeError(f"Existing trip file hash mismatch: {trips_path}")
            return stored

    work = episode_dir / ".working" / "routes"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    temporary_route = work / "sumo_demand.rou.xml"
    temporary_trips = work / "sumo_demand.trips.xml"
    command = [
        sys.executable,
        str(random_trips),
        "-n",
        str(network_path),
        "-o",
        str(temporary_trips),
        "-r",
        str(temporary_route),
        "-b",
        f"{parameters['randomTrips']['departure_begin_s']:.9g}",
        "-e",
        f"{parameters['randomTrips']['departure_end_s']:.9g}",
        "-p",
        f"{parameters['randomTrips']['period_s']:.9g}",
        "-s",
        str(spec.traffic_seed),
        "--min-distance",
        f"{parameters['randomTrips']['min_distance_m']:.9g}",
        "-i",
        str(parameters["randomTrips"]["intermediate"]),
        "--maxtries",
        str(parameters["randomTrips"]["maxtries"]),
        "--vehicle-class",
        "passenger",
    ]
    environment = os.environ.copy()
    environment["SUMO_HOME"] = str(sumo_home)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "randomTrips.py failed: "
            f"exit={completed.returncode}; stdout={completed.stdout[-1000:]}; "
            f"stderr={completed.stderr[-2000:]}"
        )
    if not temporary_route.is_file() or not temporary_trips.is_file():
        raise RuntimeError("randomTrips.py succeeded without both route and trip outputs.")
    route_count, first_departure, last_departure = _inspect_route_file(temporary_route)
    if route_count <= 0:
        raise RuntimeError(f"randomTrips.py generated no vehicles: {temporary_route}")
    if first_departure is None or not math.isclose(
        first_departure, parameters["randomTrips"]["departure_begin_s"], abs_tol=1e-6
    ):
        raise RuntimeError(
            "Generated route demand does not start at the configured departure begin: "
            f"{first_departure}"
        )
    if last_departure is None or last_departure >= spec.duration_s + 1e-7:
        raise RuntimeError(
            "Generated route demand contains a departure outside the observation horizon: "
            f"last={last_departure}, end_exclusive={spec.duration_s}"
        )
    os.replace(temporary_route, route_path)
    os.replace(temporary_trips, trips_path)
    shutil.rmtree(work, ignore_errors=True)
    metadata = {
        "schema_version": 1,
        "generator": "SUMO randomTrips.py",
        "generator_path": str(random_trips),
        "generator_sha256": parameters["randomTrips_sha256"],
        "network_path": str(network_path),
        "network_sha256": parameters["network_sha256"],
        "run_fingerprint": parameters["run_fingerprint"],
        "split": spec.split,
        "episode_seed": spec.episode_seed,
        "sumo_seed": spec.sumo_seed,
        "traffic_seed": spec.traffic_seed,
        "randomTrips": parameters["randomTrips"],
        "route_vehicle_count": route_count,
        "first_departure_s": first_departure,
        "last_departure_s": last_departure,
        "route_sha256": _sha256_file(route_path),
        "trips_sha256": _sha256_file(trips_path),
    }
    _atomic_json(metadata_path, metadata)
    return metadata


def _inspect_route_file(route_path: Path) -> tuple[int, float | None, float | None]:
    count = 0
    first_departure: float | None = None
    last_departure: float | None = None
    for _, element in ET.iterparse(route_path, events=("end",)):
        if element.tag == "vehicle":
            departure = float(element.attrib["depart"])
            if first_departure is None or departure < first_departure:
                first_departure = departure
            if last_departure is None or departure > last_departure:
                last_departure = departure
            count += 1
            element.clear()
    return count, first_departure, last_departure


def _atomic_csv_gzip(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=VEHICLE_ACTIVITY_FIELDS)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)
                        count += 1
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


def _read_summary(summary_path: Path, duration_s: float, dt_s: float) -> dict[int, dict[str, Any]]:
    expected_steps = int(round(duration_s / dt_s))
    records: dict[int, dict[str, Any]] = {}
    with gzip.open(summary_path, "rb") as stream:
        for _, element in ET.iterparse(stream, events=("end",)):
            if element.tag != "step":
                element.clear()
                continue
            time_s = float(element.attrib["time"])
            step = int(round(time_s / dt_s))
            if not math.isclose(time_s, step * dt_s, rel_tol=0.0, abs_tol=1e-6):
                raise RuntimeError(f"SUMO summary time is off-grid: {time_s}")
            if step in records:
                raise RuntimeError(f"Duplicate SUMO summary time: {time_s}")
            records[step] = {
                "time_s": time_s,
                "running": int(element.attrib["running"]),
                "loaded": int(element.attrib["loaded"]),
                "inserted": int(element.attrib["inserted"]),
                "waiting": int(element.attrib["waiting"]),
                "ended": int(element.attrib["ended"]),
                "arrived": int(element.attrib["arrived"]),
                "teleports": int(element.attrib["teleports"]),
                "halting": int(element.attrib["halting"]),
                "mean_speed_mps": float(element.attrib["meanSpeed"]),
                "mean_waiting_time_s": float(element.attrib["meanWaitingTime"]),
            }
            element.clear()
    expected = set(range(expected_steps))
    if set(records) != expected:
        missing = sorted(expected - set(records))[:10]
        extra = sorted(set(records) - expected)[:10]
        raise RuntimeError(
            "SUMO summary time axis is incomplete: "
            f"expected={expected_steps}, actual={len(records)}, missing={missing}, extra={extra}"
        )
    return records


_TIME_RE = re.compile(r'<timestep\s+time="([^"]+)"')
_ID_RE = re.compile(r'\bid="([^"]+)"')


def _validate_trajectory_and_capture_activity(
    trajectory_path: Path,
    summary: dict[int, dict[str, Any]],
    duration_s: float,
    dt_s: float,
) -> tuple[dict[str, tuple[float, float]], int, int]:
    expected_steps = int(round(duration_s / dt_s))
    first_last: dict[str, tuple[float, float]] = {}
    time_count = 0
    row_count = 0
    current_step: int | None = None
    rows_in_step = 0
    root_closed = False
    with gzip.open(trajectory_path, "rt", encoding="utf-8", newline="") as stream:
        for line in stream:
            stripped = line.lstrip()
            if stripped.startswith("<timestep "):
                if current_step is not None:
                    raise RuntimeError("Nested FCD timestep output is invalid.")
                match = _TIME_RE.search(stripped)
                if match is None:
                    raise RuntimeError("FCD timestep record has no time attribute.")
                time_s = float(match.group(1))
                current_step = int(round(time_s / dt_s))
                if not math.isclose(time_s, current_step * dt_s, abs_tol=1e-6):
                    raise RuntimeError(f"FCD time is off-grid: {time_s}")
                if current_step != time_count:
                    raise RuntimeError(
                        f"FCD timestamps are missing, duplicated or out of order at {time_s}; "
                        f"expected step {time_count}."
                    )
                if current_step >= expected_steps:
                    raise RuntimeError(f"FCD contains a time beyond its end-exclusive horizon: {time_s}")
                rows_in_step = 0
                time_count += 1
            elif stripped.startswith("<vehicle "):
                if current_step is None:
                    raise RuntimeError("FCD vehicle record is outside a timestep.")
                match = _ID_RE.search(stripped)
                if match is None:
                    raise RuntimeError("FCD vehicle record has no ID.")
                vehicle_id = match.group(1)
                time_s = current_step * dt_s
                if vehicle_id in first_last:
                    first_last[vehicle_id] = (first_last[vehicle_id][0], time_s)
                else:
                    first_last[vehicle_id] = (time_s, time_s)
                rows_in_step += 1
                row_count += 1
            elif stripped.startswith("</timestep>"):
                if current_step is None:
                    raise RuntimeError("FCD timestep close marker has no open timestep.")
                expected_running = summary[current_step]["running"]
                if rows_in_step != expected_running:
                    raise RuntimeError(
                        "FCD active vehicle rows disagree with SUMO summary: "
                        f"time={current_step * dt_s}, trajectory={rows_in_step}, "
                        f"summary_running={expected_running}"
                    )
                current_step = None
            elif stripped.startswith("</fcd-export>"):
                root_closed = True
    if current_step is not None or not root_closed or time_count != expected_steps:
        raise RuntimeError(
            "SUMO FCD file is incomplete: "
            f"steps={time_count}/{expected_steps}, open_timestep={current_step}, "
            f"root_closed={root_closed}"
        )
    return first_last, row_count, time_count


def _read_route_schedule(route_path: Path) -> dict[str, float]:
    schedules: dict[str, float] = {}
    for _, element in ET.iterparse(route_path, events=("end",)):
        if element.tag == "vehicle":
            vehicle_id = element.attrib["id"]
            if vehicle_id in schedules:
                raise RuntimeError(f"Duplicate vehicle ID in route file: {vehicle_id}")
            schedules[vehicle_id] = float(element.attrib["depart"])
            element.clear()
    return schedules


def _read_tripinfo(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    with gzip.open(path, "rb") as stream:
        for _, element in ET.iterparse(stream, events=("end",)):
            if element.tag != "tripinfo":
                element.clear()
                continue
            vehicle_id = element.attrib["id"]
            if vehicle_id in results:
                raise RuntimeError(f"Duplicate tripinfo row for vehicle {vehicle_id}")
            results[vehicle_id] = {
                "depart": _optional_float(element.attrib.get("depart")),
                "arrival": _optional_float(element.attrib.get("arrival")),
                "vaporized": element.attrib.get("vaporized", ""),
            }
            element.clear()
    return results


def _optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _write_vehicle_activity(
    activity_path: Path,
    schedules: dict[str, float],
    observed: dict[str, tuple[float, float]],
    tripinfo: dict[str, dict[str, Any]],
    end_exclusive_s: float,
    dt_s: float,
) -> dict[str, Any]:
    ids = sorted(set(schedules) | set(observed) | set(tripinfo))

    def rows() -> Iterable[dict[str, Any]]:
        for vehicle_id in ids:
            planned = schedules.get(vehicle_id)
            first_last = observed.get(vehicle_id)
            trip = tripinfo.get(vehicle_id, {})
            arrival = trip.get("arrival")
            first_seen = first_last[0] if first_last is not None else None
            last_seen = first_last[1] if first_last is not None else None
            if arrival is not None and arrival < end_exclusive_s:
                status = "arrived"
            elif first_seen is not None and last_seen is not None and math.isclose(
                last_seen, end_exclusive_s - dt_s, abs_tol=1e-6
            ):
                status = "active_at_observation_end"
            elif first_seen is not None:
                status = "left_or_teleported_before_observation_end"
            else:
                status = "not_observed_entering_by_observation_end"
            yield {
                "vehicle_id": vehicle_id,
                "scheduled_departure_s": planned,
                "first_observed_s": first_seen,
                "last_observed_s": last_seen,
                "tripinfo_departure_s": trip.get("depart"),
                "arrival_s": arrival,
                "route_status": "completed" if arrival is not None else "unfinished_at_end",
                "vaporized_reason": trip.get("vaporized", ""),
                "activity_at_observation_end": status,
            }

    row_count = _atomic_csv_gzip(activity_path, rows())
    status_counts: dict[str, int] = {}
    arrived_count = 0
    for vehicle_id in ids:
        trip = tripinfo.get(vehicle_id, {})
        arrival = trip.get("arrival")
        first_last = observed.get(vehicle_id)
        if arrival is not None and arrival < end_exclusive_s:
            status = "arrived"
            arrived_count += 1
        elif first_last is not None and math.isclose(
            first_last[1], end_exclusive_s - dt_s, abs_tol=1e-6
        ):
            status = "active_at_observation_end"
        elif first_last is not None:
            status = "left_or_teleported_before_observation_end"
        else:
            status = "not_observed_entering_by_observation_end"
        status_counts[status] = status_counts.get(status, 0) + 1
    if row_count != len(ids):
        raise RuntimeError("Vehicle activity CSV row count does not match unique vehicle IDs.")
    return {
        "vehicle_id_count": len(ids),
        "observed_vehicle_count": len(observed),
        "arrived_vehicle_count": arrived_count,
        "activity_at_observation_end_counts": status_counts,
    }


def _validate_gzip(path: Path) -> None:
    with gzip.open(path, "rb") as stream:
        while stream.read(4 * 1024 * 1024):
            pass


def _checkpoint_snapshot(checkpoint_dir: Path) -> dict[str, Any] | None:
    if not checkpoint_dir.is_dir():
        return None
    candidates: list[tuple[float, Path]] = []
    for path in checkpoint_dir.glob("sumo_*.xml.gz"):
        match = re.search(r"_([0-9]+(?:\.[0-9]+)?)\.xml\.gz$", path.name)
        if match:
            candidates.append((float(match.group(1)), path))
    for time_s, path in sorted(candidates, reverse=True):
        try:
            size_before = path.stat().st_size
            if size_before <= 0:
                continue
            with gzip.open(path, "rb") as stream:
                prefix = stream.read(512)
            size_after = path.stat().st_size
            if size_before != size_after or not prefix:
                continue
            return {
                "simulation_time_s": time_s,
                "path": str(path),
                "size_bytes": size_after,
                "sha256": _sha256_file(path),
                "gzip_prefix_valid": True,
            }
        except (OSError, EOFError, gzip.BadGzipFile):
            continue
    return None


def _sumo_command(
    *,
    sumo_binary: Path,
    network_path: Path,
    route_path: Path,
    dt_s: float,
    seed: int,
    end_s: float,
    work_dir: Path,
    checkpoint_period_s: float,
) -> list[str]:
    checkpoint_dir = work_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return [
        str(sumo_binary),
        "--net-file",
        str(network_path),
        "--route-files",
        str(route_path),
        "--begin",
        "0",
        "--end",
        f"{end_s:.9g}",
        "--step-length",
        f"{dt_s:.9g}",
        "--seed",
        str(seed),
        "--no-step-log",
        "true",
        "--fcd-output",
        str(work_dir / "trajectory.partial.xml.gz"),
        "--fcd-output.acceleration",
        "--fcd-output.distance",
        "--tripinfo-output",
        str(work_dir / "tripinfo.partial.xml.gz"),
        "--tripinfo-output.write-unfinished",
        "--tripinfo-output.write-undeparted",
        "--summary-output",
        str(work_dir / "summary.partial.xml.gz"),
        "--save-state.period",
        f"{checkpoint_period_s:.9g}",
        "--save-state.period.keep",
        "2",
        "--save-state.rng",
        "--save-state.prefix",
        str(checkpoint_dir / "sumo"),
    ]


def _run_sumo(
    *,
    command: list[str],
    work_dir: Path,
    environment: dict[str, str],
    progress_path: Path,
    progress: dict[str, Any],
    checkpoint_period_s: float,
) -> dict[str, Any]:
    log_path = work_dir / "sumo.stderr.log"
    work_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint: dict[str, Any] | None = None
    last_checkpoint_sizes: dict[str, tuple[int, float]] = {}
    started_at = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=log_stream,
        )
        try:
            while process.poll() is None:
                time.sleep(0.5)
                snapshot = _checkpoint_snapshot(work_dir / "checkpoints")
                if snapshot is not None:
                    path = Path(snapshot["path"])
                    size = int(snapshot["size_bytes"])
                    modified = path.stat().st_mtime
                    key = str(path)
                    previous = last_checkpoint_sizes.get(key)
                    if previous is not None and previous[0] == size and previous[1] == modified:
                        last_checkpoint = snapshot
                        progress.update(
                            {
                                "status": "running",
                                "latest_confirmed_sumo_checkpoint": last_checkpoint,
                                "last_confirmed_trajectory_end_exclusive_s": None,
                                "trajectory_publication": "atomic_after_SUMO_exit_and_validation",
                            }
                        )
                        _atomic_json(progress_path, progress)
                    last_checkpoint_sizes[key] = (size, modified)
            return_code = int(process.returncode or 0)
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            snapshot = _checkpoint_snapshot(work_dir / "checkpoints") or last_checkpoint
            progress.update(
                {
                    "status": "interrupted",
                    "latest_confirmed_sumo_checkpoint": snapshot,
                    "recovery_action": "rebuild_realization_from_fixed_seed",
                    "updated_at_utc": _utc_now(),
                }
            )
            _atomic_json(progress_path, progress)
            raise
    elapsed = time.perf_counter() - started_at
    if return_code != 0:
        progress.update(
            {
                "status": "failed",
                "sumo_exit_code": return_code,
                "latest_confirmed_sumo_checkpoint": _checkpoint_snapshot(
                    work_dir / "checkpoints"
                )
                or last_checkpoint,
                "recovery_action": "rebuild_realization_from_fixed_seed",
                "updated_at_utc": _utc_now(),
            }
        )
        _atomic_json(progress_path, progress)
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"SUMO failed with exit code {return_code}:\n{tail}")
    return {
        "wall_seconds": elapsed,
        "sumo_exit_code": return_code,
        "latest_confirmed_checkpoint": _checkpoint_snapshot(
            work_dir / "checkpoints"
        )
        or last_checkpoint,
        "stderr_log_path": log_path,
    }


def _read_warning_count(path: Path) -> int:
    count = 0
    if not path.is_file():
        return count
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("Warning:"):
                count += 1
    return count


def _move_completed_outputs(work_dir: Path, episode_dir: Path) -> None:
    mapping = {
        "trajectory.partial.xml.gz": "trajectory.fcd.xml.gz",
        "tripinfo.partial.xml.gz": "tripinfo.xml.gz",
        "summary.partial.xml.gz": "summary.xml.gz",
    }
    for source_name, target_name in mapping.items():
        source = work_dir / source_name
        target = episode_dir / target_name
        if not source.is_file():
            raise RuntimeError(f"SUMO output is missing: {source}")
        _validate_gzip(source)
        os.replace(source, target)
    checkpoint_dir = work_dir / "checkpoints"
    final_checkpoint_dir = episode_dir / "checkpoints"
    if final_checkpoint_dir.exists():
        shutil.rmtree(final_checkpoint_dir)
    if checkpoint_dir.is_dir():
        os.replace(checkpoint_dir, final_checkpoint_dir)
    log_path = work_dir / "sumo.stderr.log"
    if log_path.is_file():
        os.replace(log_path, episode_dir / "sumo.stderr.log")


def _normalize_checkpoint_reference(
    checkpoint: dict[str, Any] | None,
    episode_dir: Path,
) -> dict[str, Any] | None:
    if not checkpoint:
        return None
    name = Path(str(checkpoint.get("path", ""))).name
    path = episode_dir / "checkpoints" / name
    if not path.is_file():
        return {**checkpoint, "path": None, "available_after_finalize": False}
    return {
        **checkpoint,
        "path": f"checkpoints/{name}",
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "available_after_finalize": True,
    }


def _build_episode_events_and_validate(
    *,
    spec: EpisodeSpec,
    episode_dir: Path,
    dt_s: float,
) -> dict[str, Any]:
    trajectory_path = episode_dir / "trajectory.fcd.xml.gz"
    summary_path = episode_dir / "summary.xml.gz"
    tripinfo_path = episode_dir / "tripinfo.xml.gz"
    route_path = episode_dir / "sumo_demand.rou.xml"
    summary = _read_summary(summary_path, spec.duration_s, dt_s)
    observed, trajectory_rows, trajectory_steps = _validate_trajectory_and_capture_activity(
        trajectory_path,
        summary,
        spec.duration_s,
        dt_s,
    )
    schedules = _read_route_schedule(route_path)
    tripinfo = _read_tripinfo(tripinfo_path)
    if not set(tripinfo).issubset(set(schedules)):
        raise RuntimeError("SUMO tripinfo contains vehicle IDs absent from the demand route file.")
    activity_path = episode_dir / "vehicle_activity.csv.gz"
    activity = _write_vehicle_activity(
        activity_path,
        schedules,
        observed,
        tripinfo,
        spec.duration_s,
        dt_s,
    )
    final_summary = summary[trajectory_steps - 1]
    return {
        "trajectory_timestep_count": trajectory_steps,
        "trajectory_vehicle_state_row_count": trajectory_rows,
        "trajectory_unique_observed_vehicle_count": len(observed),
        "summary_timestep_count": len(summary),
        "summary_first_time_s": summary[0]["time_s"],
        "summary_last_time_s": final_summary["time_s"],
        "summary_max_running_vehicles": max(item["running"] for item in summary.values()),
        "summary_mean_running_vehicles": sum(item["running"] for item in summary.values())
        / len(summary),
        "summary_total_loaded_at_end": final_summary["loaded"],
        "summary_total_inserted_at_end": final_summary["inserted"],
        "summary_total_waiting_at_end": final_summary["waiting"],
        "summary_total_arrived_at_end": final_summary["arrived"],
        "summary_total_teleports_at_end": final_summary["teleports"],
        "summary_final_mean_speed_mps": final_summary["mean_speed_mps"],
        "summary_final_halting_vehicles": final_summary["halting"],
        **activity,
    }


def _episode_artifacts(episode_dir: Path) -> list[str]:
    required = [
        "sumo_demand.trips.xml",
        "sumo_demand.rou.xml",
        "sumo_demand.meta.json",
        "trajectory.fcd.xml.gz",
        "tripinfo.xml.gz",
        "summary.xml.gz",
        "vehicle_activity.csv.gz",
        "sumo.stderr.log",
    ]
    checkpoint_dir = episode_dir / "checkpoints"
    if checkpoint_dir.is_dir():
        required.extend(
            str(path.relative_to(episode_dir).as_posix())
            for path in sorted(checkpoint_dir.glob("sumo_*.xml.gz"))
        )
    return required


def _validate_episode_manifest(episode_dir: Path, expected_fingerprint: str | None = None) -> dict[str, Any]:
    manifest_path = episode_dir / "episode_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != EPISODE_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported episode manifest schema: {manifest_path}")
    if expected_fingerprint and manifest.get("run_fingerprint") != expected_fingerprint:
        raise RuntimeError(f"Episode manifest contract mismatch: {episode_dir}")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict):
        raise RuntimeError(f"Episode manifest has no artifact hashes: {manifest_path}")
    for name, expected_hash in hashes.items():
        path = episode_dir / name
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise RuntimeError(f"Episode artifact missing or hash mismatch: {path}")
    return manifest


def _finalize_episode(
    *,
    spec: EpisodeSpec,
    config: dict[str, Any],
    config_path: Path,
    split_manifest_path: Path,
    split_manifest: dict[str, Any],
    split_manifest_sha256: str,
    network_path: Path,
    network_sha256: str,
    random_trips: Path,
    random_trips_sha256: str,
    sumo_binary: Path,
    sumo_version: str,
    generator_sha256: str,
    run_fingerprint: str,
    parameters: dict[str, Any],
    episode_dir: Path,
    sim_wall_seconds: float,
    latest_checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = _build_episode_events_and_validate(
        spec=spec,
        episode_dir=episode_dir,
        dt_s=float(config["simulation"]["dt_s"]),
    )
    demand_metadata = _load_json(episode_dir / "sumo_demand.meta.json")
    if demand_metadata.get("run_fingerprint") != run_fingerprint:
        raise RuntimeError("Route-demand metadata run fingerprint mismatch.")
    route_count = int(demand_metadata["route_vehicle_count"])
    if metrics["summary_total_loaded_at_end"] != route_count:
        raise RuntimeError(
            "SUMO summary did not load the full generated demand: "
            f"loaded={metrics['summary_total_loaded_at_end']}, routes={route_count}"
        )
    if metrics["trajectory_unique_observed_vehicle_count"] > route_count:
        raise RuntimeError("FCD contains more unique vehicle IDs than route demand.")
    if metrics["vehicle_id_count"] != route_count:
        raise RuntimeError(
            "Vehicle activity records do not cover the generated demand: "
            f"activity_ids={metrics['vehicle_id_count']}, routes={route_count}"
        )
    if latest_checkpoint is None:
        raise RuntimeError("Completed SUMO realization has no recoverable state checkpoint.")
    artifacts = _episode_artifacts(episode_dir)
    artifact_hashes = {
        name: _sha256_file(episode_dir / name)
        for name in artifacts
        if (episode_dir / name).is_file()
    }
    manifest = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "split": spec.split,
        "episode_name": spec.episode_name,
        "episode_seed": spec.episode_seed,
        "derived_seeds": derive_episode_seeds(spec.episode_seed),
        "sumo_seed": spec.sumo_seed,
        "traffic_seed": spec.traffic_seed,
        "run_fingerprint": run_fingerprint,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "source_split_manifest_path": str(split_manifest_path),
        "source_split_manifest_sha256": split_manifest_sha256,
        "source_split_manifest_config_path": split_manifest.get("generator_config_path"),
        "source_split_manifest_config_sha256": split_manifest.get("generator_config_sha256"),
        "source_split_manifest_config_file_exists": (
            _resolve_project_path(str(split_manifest["generator_config_path"])).is_file()
            if split_manifest.get("generator_config_path")
            else False
        ),
        "network_path": str(network_path),
        "network_sha256": network_sha256,
        "randomTrips_path": str(random_trips),
        "randomTrips_sha256": random_trips_sha256,
        "sumo_binary_path": str(sumo_binary),
        "sumo_version": sumo_version,
        "platform": platform.platform(),
        "duration_s": spec.duration_s,
        "dt_s": float(config["simulation"]["dt_s"]),
        "warmup_s": spec.warmup_s,
        "formal_observation_interval_s": [spec.formal_start_s, spec.formal_end_s],
        "formal_interval_convention": "[start_s, end_exclusive_s)",
        "simulation_timestamps_convention": "0_s through duration_s-dt_s inclusive",
        "randomTrips": parameters["randomTrips"],
        "sumo_command_contract": {
            "begin_s": 0.0,
            "end_exclusive_s": spec.duration_s,
            "step_length_s": float(config["simulation"]["dt_s"]),
            "fcd_attributes": ["id", "time", "x", "y", "speed", "acceleration", "edge", "lane", "pos", "distance"],
            "vehicle_activity_status_source": "FCD active rows + tripinfo + route demand + summary",
            "sumo_defaults_changed": False,
            "teleport_policy": "SUMO 1.27.1 default; counts retained in summary output",
        },
        "checkpoint_contract": {
            "period_s": float(parameters["checkpoint_period_s"]),
            "save_rng_state": True,
            "kept_checkpoint_files": [
                name for name in artifacts if name.startswith("checkpoints/")
            ],
            "resume_strategy": "deterministic_rebuild_from_episode_seed_and_route_file",
            "state_load_claim": "not_used_for_exact_resume; SUMO does not serialize all car-following/lane-change internals",
            "latest_confirmed_checkpoint": latest_checkpoint,
        },
        "trajectory_format": TRAJECTORY_FORMAT_VERSION,
        "vehicle_activity_format": ACTIVITY_FORMAT_VERSION,
        "sumo_wall_seconds": sim_wall_seconds,
        "sumo_warning_count": _read_warning_count(episode_dir / "sumo.stderr.log"),
        "metrics": metrics,
        "artifact_sha256": artifact_hashes,
    }
    _atomic_json(episode_dir / "episode_manifest.json", manifest)
    progress = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "status": "complete",
        "episode_seed": spec.episode_seed,
        "run_fingerprint": run_fingerprint,
        "last_confirmed_trajectory_end_exclusive_s": spec.duration_s,
        "latest_confirmed_sumo_checkpoint": latest_checkpoint,
        "episode_manifest_sha256": _sha256_file(episode_dir / "episode_manifest.json"),
        "updated_at_utc": _utc_now(),
    }
    _atomic_json(episode_dir / "episode_progress.json", progress)
    return manifest


def _run_episode(
    *,
    spec: EpisodeSpec,
    config: dict[str, Any],
    config_path: Path,
    split_manifest_path: Path,
    split_manifest: dict[str, Any],
    split_manifest_sha256: str,
    network_path: Path,
    network_sha256: str,
    random_trips: Path,
    random_trips_sha256: str,
    sumo_home: Path,
    sumo_binary: Path,
    sumo_version: str,
    generator_sha256: str,
    output_root: Path,
    environment: dict[str, str],
    resume: bool,
    smoke_interrupt_at_s: float | None = None,
    smoke_duration_s: float | None = None,
) -> dict[str, Any]:
    episode_dir = _dataset_layout(output_root, spec.split, spec.episode_name)
    episode_dir.mkdir(parents=True, exist_ok=True)
    parameters = {
        "randomTrips": _random_trips_parameters(config, spec),
        "network_sha256": network_sha256,
        "randomTrips_sha256": random_trips_sha256,
        "dt_s": float(config["simulation"]["dt_s"]),
        "checkpoint_period_s": min(CHECKPOINT_PERIOD_S, max(float(config["simulation"]["dt_s"]), spec.duration_s / 2.0)),
    }
    if smoke_duration_s is not None:
        parameters["randomTrips"]["departure_end_s"] = smoke_duration_s
        parameters["checkpoint_period_s"] = min(
            15.0,
            max(float(config["simulation"]["dt_s"]), smoke_duration_s / 4.0),
        )
        spec = EpisodeSpec(
            split=spec.split,
            episode_seed=spec.episode_seed,
            sumo_seed=spec.sumo_seed,
            traffic_seed=spec.traffic_seed,
            duration_s=smoke_duration_s,
            warmup_s=min(spec.warmup_s, smoke_duration_s / 4.0),
            formal_start_s=min(spec.warmup_s, smoke_duration_s / 4.0),
            formal_end_s=smoke_duration_s,
            departure_begin_s=spec.departure_begin_s,
            departure_end_s=smoke_duration_s,
            episode_name=spec.episode_name,
        )
    run_fingerprint = _run_fingerprint(
        _sha256_file(config_path),
        network_sha256,
        random_trips_sha256,
        sumo_version,
        generator_sha256,
        spec,
        parameters,
    )
    manifest_path = episode_dir / "episode_manifest.json"
    if manifest_path.is_file():
        existing = _validate_episode_manifest(episode_dir, run_fingerprint)
        return existing

    progress_path = episode_dir / "episode_progress.json"
    previous_progress: dict[str, Any] | None = None
    previous_status: str | None = None
    if progress_path.is_file():
        previous_progress = _load_json(progress_path)
        previous_status = str(previous_progress.get("status", ""))
        if previous_progress.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(
                f"Interrupted episode contract changed; refusing to mix data: {episode_dir}"
            )
        if not resume and not smoke_interrupt_at_s:
            raise RuntimeError(
                f"Incomplete episode exists; pass --resume to rebuild it: {episode_dir}"
            )

    route_parameters = {
        **parameters,
        "run_fingerprint": run_fingerprint,
    }
    if previous_progress is None:
        progress = {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "status": "preparing",
            "episode_seed": spec.episode_seed,
            "run_fingerprint": run_fingerprint,
            "last_confirmed_trajectory_end_exclusive_s": 0.0,
            "checkpoint_period_s": parameters["checkpoint_period_s"],
            "recovery_action": None,
            "created_at_utc": _utc_now(),
        }
    else:
        progress = previous_progress
        progress.update(
            {
                "status": "rebuilding_from_fixed_seed",
                "recovery_action": "rebuild_realization_from_fixed_seed",
                "last_confirmed_trajectory_end_exclusive_s": 0.0,
                "updated_at_utc": _utc_now(),
            }
        )
        history = list(progress.get("recovery_history", []))
        history.append(
            {
                "prior_status": previous_progress.get("status"),
                "prior_latest_checkpoint": previous_progress.get(
                    "latest_confirmed_sumo_checkpoint"
                ),
                "action": "rebuild full realization from beginning using the same seed and route file",
                "at_utc": _utc_now(),
            }
        )
        progress["recovery_history"] = history
    _atomic_json(progress_path, progress)

    demand = _create_route_demand(
        spec=spec,
        parameters=route_parameters,
        network_path=network_path,
        random_trips=random_trips,
        sumo_home=sumo_home,
        episode_dir=episode_dir,
        allow_incomplete_rebuild=previous_progress is not None,
    )
    final_outputs = (
        episode_dir / "trajectory.fcd.xml.gz",
        episode_dir / "tripinfo.xml.gz",
        episode_dir / "summary.xml.gz",
        episode_dir / "sumo.stderr.log",
    )
    if (
        previous_status == "sumo_complete_pending_validation"
        and all(path.is_file() for path in final_outputs)
    ):
        latest_checkpoint = _normalize_checkpoint_reference(
            progress.get("latest_confirmed_sumo_checkpoint"),
            episode_dir,
        )
        return _finalize_episode(
            spec=spec,
            config=config,
            config_path=config_path,
            split_manifest_path=split_manifest_path,
            split_manifest=split_manifest,
            split_manifest_sha256=split_manifest_sha256,
            network_path=network_path,
            network_sha256=network_sha256,
            random_trips=random_trips,
            random_trips_sha256=random_trips_sha256,
            sumo_binary=sumo_binary,
            sumo_version=sumo_version,
            generator_sha256=generator_sha256,
            run_fingerprint=run_fingerprint,
            parameters=parameters,
            episode_dir=episode_dir,
            sim_wall_seconds=float(progress.get("sumo_wall_seconds", 0.0)),
            latest_checkpoint=latest_checkpoint,
        )
    attempt_number = int(progress.get("attempt_number", 0)) + 1
    work_dir = episode_dir / ".working" / f"attempt_{attempt_number:03d}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    progress.update(
        {
            "status": "running",
            "attempt_number": attempt_number,
            "route_vehicle_count": demand["route_vehicle_count"],
            "last_confirmed_trajectory_end_exclusive_s": 0.0,
            "trajectory_publication": "atomic_after_SUMO_exit_and_validation",
            "updated_at_utc": _utc_now(),
        }
    )
    _atomic_json(progress_path, progress)
    end_s = smoke_interrupt_at_s if smoke_interrupt_at_s is not None else spec.duration_s
    if not 0 < end_s <= spec.duration_s:
        raise ValueError("Controlled interruption time must be in (0, duration_s].")
    command = _sumo_command(
        sumo_binary=sumo_binary,
        network_path=network_path,
        route_path=episode_dir / "sumo_demand.rou.xml",
        dt_s=float(config["simulation"]["dt_s"]),
        seed=spec.sumo_seed,
        end_s=end_s,
        work_dir=work_dir,
        checkpoint_period_s=parameters["checkpoint_period_s"],
    )
    run_info = _run_sumo(
        command=command,
        work_dir=work_dir,
        environment=environment,
        progress_path=progress_path,
        progress=progress,
        checkpoint_period_s=parameters["checkpoint_period_s"],
    )
    if smoke_interrupt_at_s is not None:
        progress.update(
            {
                "status": "controlled_interruption",
                "sumo_completed_through_s": end_s,
                "latest_confirmed_sumo_checkpoint": run_info["latest_confirmed_checkpoint"],
                "updated_at_utc": _utc_now(),
            }
        )
        _atomic_json(progress_path, progress)
        return {
            "interrupted": True,
            "progress": progress,
            "work_dir": work_dir,
            "spec": spec,
            "run_fingerprint": run_fingerprint,
            "parameters": parameters,
            "demand": demand,
        }

    if end_s != spec.duration_s:
        raise RuntimeError("SUMO end time unexpectedly differs from the complete episode horizon.")
    progress.update(
        {
            "status": "sumo_complete_pending_validation",
            "sumo_wall_seconds": run_info["wall_seconds"],
            "latest_confirmed_sumo_checkpoint": run_info["latest_confirmed_checkpoint"],
            "updated_at_utc": _utc_now(),
        }
    )
    _atomic_json(progress_path, progress)
    _move_completed_outputs(work_dir, episode_dir)
    shutil.rmtree(episode_dir / ".working", ignore_errors=True)
    latest_checkpoint = _normalize_checkpoint_reference(
        run_info["latest_confirmed_checkpoint"],
        episode_dir,
    )
    progress["latest_confirmed_sumo_checkpoint"] = latest_checkpoint
    _atomic_json(progress_path, progress)
    manifest = _finalize_episode(
        spec=spec,
        config=config,
        config_path=config_path,
        split_manifest_path=split_manifest_path,
        split_manifest=split_manifest,
        split_manifest_sha256=split_manifest_sha256,
        network_path=network_path,
        network_sha256=network_sha256,
        random_trips=random_trips,
        random_trips_sha256=random_trips_sha256,
        sumo_binary=sumo_binary,
        sumo_version=sumo_version,
        generator_sha256=generator_sha256,
        run_fingerprint=run_fingerprint,
        parameters=parameters,
        episode_dir=episode_dir,
        sim_wall_seconds=run_info["wall_seconds"],
        latest_checkpoint=latest_checkpoint,
    )
    return manifest


def _split_specification(specs: list[EpisodeSpec], source_manifest: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        selected = [spec for spec in specs if spec.split == split]
        source = source_manifest[split]
        result[split] = {
            "directory": split,
            "source_seed_range": [int(source["seed_start"]), int(source["seed_end"])],
            "source_contract_count": int(source["count"]),
            "generated_episode_count": len(selected),
            "episode_seeds": [spec.episode_seed for spec in selected],
            "episode_duration_s": sorted({spec.duration_s for spec in selected}),
            "formal_interval_s": [
                [spec.formal_start_s, spec.formal_end_s] for spec in selected[:1]
            ],
            "split_unit_note": (
                "one continuous 300-round training stream; the source Training Pool's 300 independent episodes are not concatenated"
                if split == "train"
                else "independently generated complete evaluation realization"
            ),
        }
    return result


def _prepare_dataset_manifest(
    *,
    output_root: Path,
    config_path: Path,
    split_manifest_path: Path,
    config: dict[str, Any],
    split_manifest: dict[str, Any],
    split_manifest_sha256: str,
    network_path: Path,
    network_sha256: str,
    random_trips: Path,
    random_trips_sha256: str,
    sumo_binary: Path,
    sumo_version: str,
    generator_sha256: str,
    specs: list[EpisodeSpec],
    resume: bool,
) -> tuple[dict[str, Any], str]:
    config_sha256 = _sha256_file(config_path)
    dataset_identity = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "config_sha256": config_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "network_sha256": network_sha256,
        "randomTrips_sha256": random_trips_sha256,
        "sumo_binary_path": str(sumo_binary),
        "sumo_version": sumo_version,
        "generator_sha256": generator_sha256,
        "split_specs": [spec.__dict__ for spec in specs],
        "trajectory_format": TRAJECTORY_FORMAT_VERSION,
        "activity_format": ACTIVITY_FORMAT_VERSION,
    }
    dataset_fingerprint = _fingerprint(dataset_identity)
    manifest_path = output_root / "dataset_manifest.json"
    if output_root.exists():
        if not manifest_path.is_file():
            if any(output_root.iterdir()):
                raise RuntimeError(
                    f"Output directory exists without this generator's manifest; refusing to overwrite: {output_root}"
                )
            if not resume:
                raise RuntimeError("Output directory exists; pass --resume to continue it.")
        elif not resume:
            raise RuntimeError(
                f"Dataset already exists or is incomplete; pass --resume after verifying its manifest: {output_root}"
            )
        else:
            stored = _load_json(manifest_path)
            if stored.get("dataset_fingerprint") != dataset_fingerprint:
                raise RuntimeError(
                    "Existing dataset has a different configuration/code/input contract; "
                    "refusing to mix or replace it."
                )
            for relative, expected_hash in stored.get("source_snapshots", {}).items():
                snapshot = output_root / relative
                if not snapshot.is_file() or _sha256_file(snapshot) != expected_hash:
                    raise RuntimeError(f"Dataset source snapshot is missing or changed: {snapshot}")
            return stored, dataset_fingerprint
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    config_snapshot = output_root / "source_config_snapshot.toml"
    split_snapshot = output_root / "source_data_split_manifest_snapshot.json"
    _atomic_bytes(config_snapshot, config_path.read_bytes())
    _atomic_bytes(split_snapshot, split_manifest_path.read_bytes())
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "status": "in_progress",
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "dataset_fingerprint": dataset_fingerprint,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "source_snapshots": {
            config_snapshot.name: _sha256_file(config_snapshot),
            split_snapshot.name: _sha256_file(split_snapshot),
        },
        "active_config": {
            "dt_s": float(config["simulation"]["dt_s"]),
            "warmup_time_s": float(config["simulation"]["warmup_time_s"]),
            "episode_time_s": float(config["simulation"]["episode_time_s"]),
            "source_randomTrips_departure_end_s": float(
                project_config.get_config_value(config, "simulation.random_trips_departure_end_s", 60.0)
            ),
            "randomTrips_period_s": float(
                configured_random_trips_period_s(config)
            ),
            "randomTrips_min_distance_m": float(
                project_config.get_config_value(config, "simulation.random_trips_min_distance_m", 4000.0)
            ),
            "randomTrips_intermediate": int(
                project_config.get_config_value(config, "simulation.random_trips_intermediate", 2)
            ),
            "randomTrips_maxtries": int(
                project_config.get_config_value(config, "simulation.random_trips_maxtries", 1000)
            ),
            "num_research_vehicles_fixed_role_interface": int(
                project_config.get_config_value(config, "simulation.num_research_vehicles", 0)
            ),
            "num_service_vehicles_fixed_role_interface": int(
                project_config.get_config_value(config, "simulation.num_service_vehicles", 0)
            ),
            "num_task_vehicles_fixed_role_interface": int(
                project_config.get_config_value(config, "simulation.num_task_vehicles", 0)
            ),
        },
        "traffic_demand_contract": {
            "departure_end_is_extended_to_each_observation_horizon": True,
            "density_and_physical_parameters_changed": False,
            "source_demand_window_shorter_than_generated_horizons": True,
            "vehicle_class": "passenger",
            "sumo_defaults_changed": False,
        },
        "source_split_manifest_path": str(split_manifest_path),
        "source_split_manifest_sha256": split_manifest_sha256,
        "source_split_manifest_generator_config_path": split_manifest.get("generator_config_path"),
        "source_split_manifest_generator_config_sha256": split_manifest.get("generator_config_sha256"),
        "source_split_manifest_generator_config_exists": (
            _resolve_project_path(str(split_manifest["generator_config_path"])).is_file()
            if split_manifest.get("generator_config_path")
            else False
        ),
        "network_path": str(network_path),
        "network_sha256": network_sha256,
        "randomTrips_path": str(random_trips),
        "randomTrips_sha256": random_trips_sha256,
        "sumo_binary_path": str(sumo_binary),
        "sumo_version": sumo_version,
        "platform": platform.platform(),
        "generator_path": str(Path(__file__).resolve()),
        "generator_sha256": generator_sha256,
        "checkpoint_period_s": CHECKPOINT_PERIOD_S,
        "checkpoint_resume_strategy": "replay interrupted realization from its fixed episode seed and route file",
        "episode_specs": [spec.__dict__ for spec in specs],
        "splits": _split_specification(specs, split_manifest),
        "episodes": {
            f"{spec.split}/{spec.episode_name}": {
                "status": "pending",
                "episode_seed": spec.episode_seed,
            }
            for spec in specs
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest, dataset_fingerprint


def _validate_dataset(output_root: Path, expected_fingerprint: str | None = None) -> dict[str, Any]:
    manifest_path = output_root / "dataset_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported continuous dataset manifest: {manifest_path}")
    if expected_fingerprint and manifest.get("dataset_fingerprint") != expected_fingerprint:
        raise RuntimeError("Dataset-level contract fingerprint mismatch.")
    for key, entry in manifest.get("episodes", {}).items():
        if entry.get("status") != "complete":
            continue
        split, name = key.split("/", 1)
        episode_dir = output_root / split / name
        episode = _validate_episode_manifest(episode_dir)
        if entry.get("episode_manifest_sha256") != _sha256_file(
            episode_dir / "episode_manifest.json"
        ):
            raise RuntimeError(f"Dataset entry manifest hash mismatch: {key}")
        if episode.get("episode_seed") != entry.get("episode_seed"):
            raise RuntimeError(f"Dataset episode seed mismatch: {key}")
    return manifest


def _run_recovery_smoke_test(
    *,
    config: dict[str, Any],
    config_path: Path,
    split_manifest_path: Path,
    split_manifest: dict[str, Any],
    network_path: Path,
    network_sha256: str,
    sumo_home: Path,
    sumo_binary: Path,
    random_trips: Path,
    sumo_version: str,
    generator_sha256: str,
) -> dict[str, Any]:
    seed_root = 987654321
    derived = derive_episode_seeds(seed_root)
    smoke_seconds = 60.0
    interrupted_at_s = 30.0
    dt_s = float(config["simulation"]["dt_s"])
    base_spec = EpisodeSpec(
        split="train",
        episode_seed=seed_root,
        sumo_seed=int(derived["sumo_seed"]),
        traffic_seed=int(derived["traffic_seed"]),
        duration_s=smoke_seconds,
        warmup_s=min(float(config["simulation"]["warmup_time_s"]), 10.0),
        formal_start_s=min(float(config["simulation"]["warmup_time_s"]), 10.0),
        formal_end_s=smoke_seconds,
        departure_begin_s=0.0,
        departure_end_s=smoke_seconds,
        episode_name=f"smoke_seed_{seed_root}",
    )
    environment = os.environ.copy()
    environment["SUMO_HOME"] = str(sumo_home)
    scratch = Path(tempfile.mkdtemp(prefix="continuous_sumo_recovery_smoke_", dir=PROJECT_ROOT))
    try:
        reference_root = scratch / "reference"
        resumed_root = scratch / "resumed"
        reference_root.mkdir()
        resumed_root.mkdir()
        common = {
            "spec": base_spec,
            "config": config,
            "config_path": config_path,
            "split_manifest_path": split_manifest_path,
            "split_manifest": split_manifest,
            "split_manifest_sha256": _sha256_file(split_manifest_path),
            "network_path": network_path,
            "network_sha256": network_sha256,
            "random_trips": random_trips,
            "random_trips_sha256": _sha256_file(random_trips),
            "sumo_home": sumo_home,
            "sumo_binary": sumo_binary,
            "sumo_version": sumo_version,
            "generator_sha256": generator_sha256,
            "environment": environment,
            "smoke_duration_s": smoke_seconds,
        }
        reference_manifest = _run_episode(
            **common,
            output_root=reference_root,
            resume=False,
        )
        interrupted = _run_episode(
            **common,
            output_root=resumed_root,
            resume=False,
            smoke_interrupt_at_s=interrupted_at_s,
        )
        checkpoint = interrupted["progress"].get("latest_confirmed_sumo_checkpoint")
        if not checkpoint:
            raise RuntimeError("Controlled interruption produced no valid SUMO checkpoint.")
        resumed_manifest = _run_episode(
            **common,
            output_root=resumed_root,
            resume=True,
        )
        reference_dir = _dataset_layout(reference_root, "train", base_spec.episode_name)
        resumed_dir = _dataset_layout(resumed_root, "train", base_spec.episode_name)
        _validate_episode_manifest(reference_dir, reference_manifest["run_fingerprint"])
        _validate_episode_manifest(resumed_dir, resumed_manifest["run_fingerprint"])
        reference_signature = _semantic_fcd_hash(reference_dir / "trajectory.fcd.xml.gz")
        resumed_signature = _semantic_fcd_hash(resumed_dir / "trajectory.fcd.xml.gz")
        if reference_signature != resumed_signature:
            raise RuntimeError(
                "Fixed-seed recovery did not reproduce the uninterrupted FCD vehicle records."
            )
        return {
            "status": "passed",
            "smoke_duration_s": smoke_seconds,
            "controlled_interruption_at_s": interrupted_at_s,
            "checkpoint_created": True,
            "resume_mode": "replay from fixed seed and same route file",
            "reference_fcd_semantic_sha256": reference_signature,
            "resumed_fcd_semantic_sha256": resumed_signature,
            "semantic_records_match": True,
            "route_vehicle_count": reference_manifest["metrics"]["vehicle_id_count"],
            "reference_sumo_wall_s": reference_manifest["sumo_wall_seconds"],
            "resumed_sumo_wall_s": resumed_manifest["sumo_wall_seconds"],
            "summary_and_activity_validation_passed": True,
            "outputs_cleaned": True,
            "formal_environment_data_modified": False,
        }
    finally:
        resolved_scratch = scratch.resolve(strict=False)
        if not _inside(resolved_scratch, PROJECT_ROOT):
            raise RuntimeError("Recovery smoke scratch directory escaped the project workspace.")
        shutil.rmtree(resolved_scratch, ignore_errors=False)


def _semantic_fcd_hash(path: Path) -> str:
    digest = hashlib.sha256()
    current_time: str | None = None
    vehicles: list[tuple[tuple[str, str], ...]] = []
    with gzip.open(path, "rb") as stream:
        for event, element in ET.iterparse(stream, events=("start", "end")):
            if event == "start" and element.tag == "timestep":
                current_time = element.attrib["time"]
                vehicles = []
            elif event == "end" and element.tag == "vehicle":
                vehicles.append(tuple(sorted(element.attrib.items())))
                element.clear()
            elif event == "end" and element.tag == "timestep":
                payload = _canonical_json(
                    [current_time, sorted(vehicles)]
                ).encode("utf-8")
                digest.update(payload)
                digest.update(b"\n")
                current_time = None
                element.clear()
    return digest.hexdigest()


def nominal_departure_count_for_interval(
    departure_begin_s: float,
    departure_end_s: float,
    period_s: float,
) -> int:
    """Count equidistant departures at begin + n*period before end."""
    begin = float(departure_begin_s)
    end = float(departure_end_s)
    period = float(period_s)
    if not all(math.isfinite(value) for value in (begin, end, period)):
        raise ValueError("randomTrips departure bounds and period must be finite")
    if begin < 0.0 or end < begin or period <= 0.0:
        raise ValueError("randomTrips departure interval or period is invalid")
    if end == begin:
        return 0
    # randomTrips emits begin, begin+period, ... while each time is < end.
    return int(math.ceil((end - begin) / period))


def _estimate_resources(
    specs: list[EpisodeSpec],
    *,
    random_trips_period_s: float,
) -> dict[str, Any]:
    period_s = float(random_trips_period_s)
    if not math.isfinite(period_s) or period_s <= 0.0:
        raise ValueError("randomTrips_period_s must be finite and positive for resource estimation")
    total_sim_s = sum(spec.duration_s for spec in specs)
    total_vehicle_departures = sum(
        nominal_departure_count_for_interval(
            spec.departure_begin_s,
            spec.departure_end_s,
            random_trips_period_s,
        )
        for spec in specs
    )
    nominal_departure_rate_per_s = 1.0 / period_s
    benchmark_departure_rate_per_s = 1.0 / FCD_SIZE_BENCHMARK_PERIOD_S
    fcd_traffic_scale = nominal_departure_rate_per_s / benchmark_departure_rate_per_s
    estimated_fcd_bytes_per_sim_second = (
        FCD_SIZE_BENCHMARK_BYTES_PER_SIM_SECOND * fcd_traffic_scale
    )
    estimated_fcd_bytes = int(total_sim_s * estimated_fcd_bytes_per_sim_second)
    estimated_other_bytes = int(total_vehicle_departures * 900)
    estimated_final_bytes = estimated_fcd_bytes + estimated_other_bytes
    train_seconds = sum(spec.duration_s for spec in specs if spec.split == "train")
    evaluation_seconds = total_sim_s - train_seconds
    simulated_wall_s = train_seconds / 68.8 + evaluation_seconds / 281.0
    route_wall_s = total_vehicle_departures * 3.1 / 3600.0
    episode_start_overhead_s = len(specs) * 1.0
    validation_wall_s = total_sim_s * 0.003
    return {
        "total_simulation_seconds": total_sim_s,
        "formal_observation_seconds": sum(
            spec.formal_end_s - spec.formal_start_s for spec in specs
        ),
        "randomTrips_period_s": period_s,
        "nominal_departure_rate_per_s": nominal_departure_rate_per_s,
        "nominal_vehicle_departure_count": int(total_vehicle_departures),
        "fcd_size_benchmark_period_s": FCD_SIZE_BENCHMARK_PERIOD_S,
        "observed_benchmark_fcd_bytes_per_sim_second": FCD_SIZE_BENCHMARK_BYTES_PER_SIM_SECOND,
        "estimated_fcd_bytes_per_sim_second": round(estimated_fcd_bytes_per_sim_second, 3),
        "estimated_fcd_bytes": estimated_fcd_bytes,
        "estimated_final_bytes": estimated_final_bytes,
        "estimated_peak_working_bytes": 2 * estimated_final_bytes,
        "estimated_sumo_compute_minutes": round(simulated_wall_s / 60.0, 1),
        "estimated_randomTrips_compute_minutes": round(route_wall_s / 60.0, 1),
        "estimated_episode_start_overhead_minutes": round(episode_start_overhead_s / 60.0, 1),
        "estimated_validation_scan_minutes": round(validation_wall_s / 60.0, 1),
        "estimated_end_to_end_wall_minutes": round(
            (simulated_wall_s + route_wall_s + episode_start_overhead_s + validation_wall_s) / 60.0,
            1,
        ),
        "estimate_note": (
            "Nominal departures use the configured equidistant randomTrips period "
            f"of {period_s:g} s over each [departure_begin_s, departure_end_s) interval. "
            "FCD byte rate is scaled linearly from the measured 1 s-period benchmark "
            "by scheduled departure rate; this is an estimate. Route validation may "
            "discard trips and congestion can change compute and output size."
        ),
    }


def _update_dataset_episode_entry(
    output_root: Path,
    dataset_manifest: dict[str, Any],
    spec: EpisodeSpec,
    episode_manifest: dict[str, Any] | None,
    status: str,
) -> None:
    key = f"{spec.split}/{spec.episode_name}"
    entry = {
        "status": status,
        "episode_seed": spec.episode_seed,
    }
    if episode_manifest is not None:
        episode_dir = _dataset_layout(output_root, spec.split, spec.episode_name)
        entry.update(
            {
                "episode_manifest": str(
                    (episode_dir / "episode_manifest.json").relative_to(output_root)
                ),
                "episode_manifest_sha256": _sha256_file(
                    episode_dir / "episode_manifest.json"
                ),
                "run_fingerprint": episode_manifest["run_fingerprint"],
            }
        )
    dataset_manifest.setdefault("episodes", {})[key] = entry
    dataset_manifest["updated_at_utc"] = _utc_now()
    _atomic_json(output_root / "dataset_manifest.json", dataset_manifest)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate continuous raw SUMO Train/Validation/Test traffic data only."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split-manifest", default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--split",
        choices=("all", "train", "validation", "test"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--self-test-recovery", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config_path = _resolve_project_path(args.config)
    split_manifest_path = _resolve_project_path(args.split_manifest)
    if not config_path.is_file():
        raise FileNotFoundError(f"DQN configuration not found: {config_path}")
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"Data split manifest not found: {split_manifest_path}")
    output_root = _resolve_project_path(args.output_dir)
    environment_root = (PROJECT_ROOT / "environment_data").resolve(strict=False)
    if not _inside(output_root, environment_root) and not args.self_test_recovery:
        raise ValueError("Dataset output must remain under environment_data/.")
    config, split_manifest, physical = _load_source_contract(
        config_path,
        split_manifest_path,
    )
    sumo_home, sumo_binary, random_trips, sumo_version = _sumo_environment()
    network_path = physical["network_path"]
    network_sha256 = physical["network_sha256"]
    random_trips_sha256 = _sha256_file(random_trips)
    generator_sha256 = _sha256_file(Path(__file__).resolve())
    if args.self_test_recovery:
        report = _run_recovery_smoke_test(
            config=config,
            config_path=config_path,
            split_manifest_path=split_manifest_path,
            split_manifest=split_manifest,
            network_path=network_path,
            network_sha256=network_sha256,
            sumo_home=sumo_home,
            sumo_binary=sumo_binary,
            random_trips=random_trips,
            sumo_version=sumo_version,
            generator_sha256=generator_sha256,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    selected_splits = (
        ("train", "validation", "test")
        if args.split == "all"
        else (args.split,)
    )
    specs = _make_specs(config, split_manifest, physical, selected_splits)
    if args.validate_only:
        manifest = _validate_dataset(output_root)
        print(
            json.dumps(
                {
                    "dataset_status": manifest.get("status"),
                    "completed_episode_count": sum(
                        entry.get("status") == "complete"
                        for entry in manifest.get("episodes", {}).values()
                    ),
                    "manifest": str(output_root / "dataset_manifest.json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    dataset_manifest, dataset_fingerprint = _prepare_dataset_manifest(
        output_root=output_root,
        config_path=config_path,
        split_manifest_path=split_manifest_path,
        config=config,
        split_manifest=split_manifest,
        split_manifest_sha256=_sha256_file(split_manifest_path),
        network_path=network_path,
        network_sha256=network_sha256,
        random_trips=random_trips,
        random_trips_sha256=random_trips_sha256,
        sumo_binary=sumo_binary,
        sumo_version=sumo_version,
        generator_sha256=generator_sha256,
        specs=specs,
        resume=args.resume,
    )
    random_trips_period_s = configured_random_trips_period_s(config)
    if not math.isclose(
        random_trips_period_s,
        float(physical["random_trips_period_s"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("traffic interval changed after continuous dataset contract validation")
    dataset_manifest["resource_estimate"] = _estimate_resources(
        specs,
        random_trips_period_s=random_trips_period_s,
    )
    _atomic_json(output_root / "dataset_manifest.json", dataset_manifest)
    print(
        "[ContinuousSUMO] resource estimate: "
        + json.dumps(dataset_manifest["resource_estimate"], ensure_ascii=False)
    )
    environment = os.environ.copy()
    environment["SUMO_HOME"] = str(sumo_home)
    split_manifest_sha256 = _sha256_file(split_manifest_path)
    started = time.perf_counter()
    for index, spec in enumerate(specs, start=1):
        key = f"{spec.split}/{spec.episode_name}"
        print(
            f"[ContinuousSUMO] {index}/{len(specs)} split={spec.split} "
            f"seed={spec.episode_seed} duration_s={spec.duration_s:g}"
        )
        _update_dataset_episode_entry(
            output_root,
            dataset_manifest,
            spec,
            None,
            "running",
        )
        try:
            episode = _run_episode(
                spec=spec,
                config=config,
                config_path=config_path,
                split_manifest_path=split_manifest_path,
                split_manifest=split_manifest,
                split_manifest_sha256=split_manifest_sha256,
                network_path=network_path,
                network_sha256=network_sha256,
                random_trips=random_trips,
                random_trips_sha256=random_trips_sha256,
                sumo_home=sumo_home,
                sumo_binary=sumo_binary,
                sumo_version=sumo_version,
                generator_sha256=generator_sha256,
                output_root=output_root,
                environment=environment,
                resume=args.resume or index > 1,
            )
        except Exception as exc:
            dataset_manifest["status"] = "incomplete"
            dataset_manifest["last_error"] = {
                "episode": key,
                "error": f"{type(exc).__name__}: {exc}",
                "at_utc": _utc_now(),
            }
            _update_dataset_episode_entry(
                output_root,
                dataset_manifest,
                spec,
                None,
                "incomplete",
            )
            raise
        _update_dataset_episode_entry(
            output_root,
            dataset_manifest,
            spec,
            episode,
            "complete",
        )
        metrics = episode.get("metrics", {})
        print(
            f"[ContinuousSUMO] complete split={spec.split} seed={spec.episode_seed} "
            f"vehicle_rows={metrics.get('trajectory_vehicle_state_row_count')} "
            f"arrived={metrics.get('summary_total_arrived_at_end')} "
            f"waiting={metrics.get('summary_total_waiting_at_end')} "
            f"teleports={metrics.get('summary_total_teleports_at_end')}"
        )
    dataset_manifest["status"] = "complete" if len(specs) == len(
        dataset_manifest.get("episodes", {})
    ) and all(
        entry.get("status") == "complete"
        for entry in dataset_manifest.get("episodes", {}).values()
    ) else "partially_complete"
    dataset_manifest["elapsed_wall_seconds_this_invocation"] = time.perf_counter() - started
    dataset_manifest["updated_at_utc"] = _utc_now()
    _atomic_json(output_root / "dataset_manifest.json", dataset_manifest)
    _validate_dataset(output_root, dataset_fingerprint)
    print(
        json.dumps(
            {
                "dataset_status": dataset_manifest["status"],
                "output_root": str(output_root),
                "completed_episode_count": sum(
                    entry.get("status") == "complete"
                    for entry in dataset_manifest.get("episodes", {}).values()
                ),
                "elapsed_wall_seconds": round(
                    dataset_manifest["elapsed_wall_seconds_this_invocation"], 3
                ),
                "dataset_manifest": str(output_root / "dataset_manifest.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
