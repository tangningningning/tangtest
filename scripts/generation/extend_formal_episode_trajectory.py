#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministically extend one generated episode's SUMO trajectory.

The regular episode generator remains configured for the original 300 arrival
slots. This utility replays the episode's saved SUMO demand with the same
derived SUMO seed for a longer physical horizon, verifies the complete original
trajectory prefix byte-for-byte at the row level, and appends only the real
SUMO rows for the requested tail slots. It never creates task arrivals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
GENERATION_ROOT = SCRIPTS_ROOT / "generation"
for _path in (SRC_ROOT, SCRIPTS_ROOT, GENERATION_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import project_config  # noqa: E402
from generation import generate_episode_realization as realization  # noqa: E402
from scenario import sumo_scenario as sumo  # noqa: E402
from scenario.realization import derive_episode_seeds  # noqa: E402


TRAJECTORY_FIELDS = sumo.TRAJECTORY_FIELDS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TRAJECTORY_FIELDS:
            raise ValueError(f"unexpected trajectory schema: {path}")
        return list(reader)


def _prefix_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        encoded = json.dumps(
            {key: str(row[key]) for key in TRAJECTORY_FIELDS},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (float(row["time_s"]), str(row["vehicle_id"])),
    )
    temporary = path.with_name(path.name + ".extension.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=TRAJECTORY_FIELDS)
            writer.writeheader()
            writer.writerows(ordered)
            stream.flush()
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def extend_episode(
    *,
    config_file: Path,
    episode_dir: Path,
    episode_seed: int,
    split: str,
    trajectory_slots: int,
    sumo_home_arg: str | None,
) -> dict[str, Any]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported split: {split}")
    if trajectory_slots <= 0:
        raise ValueError("trajectory_slots must be positive")
    episode_dir = episode_dir.resolve(strict=True)
    config_file = config_file.resolve(strict=True)
    config = realization.load_and_validate_config(config_file)
    get = project_config.get_config_value
    dt_s = float(get(config, "simulation.dt_s"))
    warmup_s = float(get(config, "simulation.warmup_time_s"))
    original_slots_float = float(get(config, "simulation.episode_time_s")) / dt_s
    if not math.isclose(original_slots_float, round(original_slots_float), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("configured episode duration is not aligned to the slot length")
    original_slots = int(round(original_slots_float))
    if trajectory_slots <= original_slots:
        raise ValueError(
            f"requested trajectory_slots={trajectory_slots} must exceed the configured "
            f"arrival horizon={original_slots}"
        )

    manifest_path = episode_dir / "episode_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("episode_seed", -1)) != int(episode_seed):
        raise ValueError("episode manifest seed does not match requested seed")
    if manifest.get("cross_slot_trajectory_extension"):
        raise ValueError("episode already contains cross-slot trajectory extension metadata")
    route_path = episode_dir / "sumo_demand.rou.xml"
    network_path = realization.config_path(config, "paths.sumo_net")
    trajectory_path = episode_dir / "trajectory.csv"
    roles_path = episode_dir / "vehicle_roles.json"
    for required in (route_path, network_path, trajectory_path, roles_path):
        if not required.is_file():
            raise FileNotFoundError(f"required episode extension input missing: {required}")

    original_rows = _rows(trajectory_path)
    role_data = json.loads(roles_path.read_text(encoding="utf-8"))
    research_ids = [str(value) for value in role_data["research_vehicles"]]
    service_ids = [str(value) for value in role_data["service_vehicles"]]
    task_ids = [str(value) for value in role_data["task_vehicles"]]
    if (
        len(research_ids) != int(get(config, "simulation.num_research_vehicles"))
        or len(set(research_ids)) != len(research_ids)
        or set(service_ids) & set(task_ids)
        or set(service_ids) | set(task_ids) != set(research_ids)
    ):
        raise ValueError("fixed research vehicle identities/roles are inconsistent")

    seeds = derive_episode_seeds(int(episode_seed))
    base_sumo_config = realization.build_sumo_config(config, seeds)
    extended_sumo_config = replace(
        base_sumo_config,
        episode_time_s=float(trajectory_slots) * dt_s,
    )
    extended_sumo_config.validate()
    sumolib, traci = sumo.load_sumo_modules(sumo.find_sumo_home(sumo_home_arg))
    source = sumo.TrafficSource(
        network_file=network_path,
        route_files=(route_path,),
        generated_by_random_trips=True,
    )
    command = sumo.build_sumo_command(
        sumo.find_sumo_binary(sumolib, gui=False),
        source,
        extended_sumo_config,
    )
    replay_rows, _route_cache, _seen_ids, observed_times = sumo.collect_trajectories(
        traci,
        command,
        extended_sumo_config,
    )
    if not observed_times:
        raise RuntimeError("SUMO replay returned no observed simulation times")

    formal_start = max(1, int(round(warmup_s / dt_s)))
    prefix_steps = set(range(formal_start, formal_start + original_slots))
    boundary_step = formal_start + original_slots
    tail_steps = set(range(formal_start + original_slots, formal_start + trajectory_slots))

    def slot_key(row: dict[str, Any]) -> int:
        step = int(round(float(row["time_s"]) / dt_s))
        return step

    original_prefix = [
        row
        for row in original_rows
        if slot_key(row) in prefix_steps
    ]
    replay_prefix = [
        row
        for row in replay_rows
        if slot_key(row) in prefix_steps
    ]
    original_prefix.sort(key=lambda row: (float(row["time_s"]), str(row["vehicle_id"])))
    replay_prefix.sort(key=lambda row: (float(row["time_s"]), str(row["vehicle_id"])))
    normalized_original_prefix = [
        {key: str(row[key]) for key in TRAJECTORY_FIELDS}
        for row in original_prefix
    ]
    normalized_replay_prefix = [
        {key: str(row[key]) for key in TRAJECTORY_FIELDS}
        for row in replay_prefix
    ]
    if normalized_original_prefix != normalized_replay_prefix:
        raise RuntimeError(
            "deterministic SUMO replay changed the complete original trajectory prefix"
        )
    prefix_digest = _prefix_digest(original_prefix)

    original_boundary = [row for row in original_rows if slot_key(row) == boundary_step]
    replay_boundary = [row for row in replay_rows if slot_key(row) == boundary_step]
    original_boundary.sort(key=lambda row: (float(row["time_s"]), str(row["vehicle_id"])))
    replay_boundary.sort(key=lambda row: (float(row["time_s"]), str(row["vehicle_id"])))
    if [
        {key: str(row[key]) for key in TRAJECTORY_FIELDS}
        for row in original_boundary
    ] != [
        {key: str(row[key]) for key in TRAJECTORY_FIELDS}
        for row in replay_boundary
    ]:
        raise RuntimeError("deterministic SUMO replay changed the saved end-boundary state")

    replay_tail = [row for row in replay_rows if slot_key(row) in tail_steps]
    tail_by_vehicle_step: dict[tuple[str, int], dict[str, Any]] = {}
    for row in replay_tail:
        key = (str(row["vehicle_id"]), slot_key(row))
        if key in tail_by_vehicle_step:
            raise RuntimeError(f"duplicate tail trajectory row: {key}")
        if not sumo._valid_trajectory_row(row):
            raise RuntimeError(f"invalid SUMO tail trajectory row: {key}")
        tail_by_vehicle_step[key] = row

    missing = [
        (vehicle_id, step)
        for vehicle_id in research_ids
        for step in sorted(tail_steps)
        if (vehicle_id, step) not in tail_by_vehicle_step
    ]
    if missing:
        raise ValueError(
            "fixed research vehicles do not remain present for the complete extension: "
            f"missing_count={len(missing)}, sample={missing[:8]}"
        )

    net = sumolib.net.readNet(str(network_path), withInternal=False)
    x_min, y_min, x_max, y_max = net.getBoundary()
    for vehicle_id in research_ids:
        for step in sorted(tail_steps):
            row = tail_by_vehicle_step[(vehicle_id, step)]
            x_m = float(row["x_m"])
            y_m = float(row["y_m"])
            if not (
                x_min - 1e-6 <= x_m <= x_max + 1e-6
                and y_min - 1e-6 <= y_m <= y_max + 1e-6
                and str(row["road_id"]).strip()
                and str(row["lane_id"]).strip()
            ):
                raise ValueError(
                    f"fixed research vehicle position is outside the road network: "
                    f"vehicle={vehicle_id}, step={step}, xy=({x_m},{y_m})"
                )

    # The 300-slot generator already saves SUMO's end-boundary sample at
    # step 400 (slot 300). Preserve those original rows after exact replay
    # comparison, and append only newly simulated steps 401..413.
    rows_to_append = [row for row in replay_tail if slot_key(row) > boundary_step]
    original_keys = {(float(row["time_s"]), str(row["vehicle_id"])) for row in original_rows}
    for row in rows_to_append:
        key = (float(row["time_s"]), str(row["vehicle_id"]))
        if key in original_keys:
            raise RuntimeError(f"trajectory tail row collides with the original prefix: {key}")

    original_trajectory_sha256 = sha256_file(trajectory_path)
    combined_rows = [*original_rows, *rows_to_append]
    _write_rows(trajectory_path, combined_rows)
    updated_trajectory_sha256 = sha256_file(trajectory_path)

    generated_artifacts = manifest.get("generated_artifacts")
    artifact_hashes = manifest.get("generated_artifact_sha256")
    if not isinstance(generated_artifacts, list) or not isinstance(artifact_hashes, dict):
        raise ValueError("episode generator manifest has no artifact identity fields")
    artifact_hashes["trajectory.csv"] = updated_trajectory_sha256
    manifest["generated_artifact_sha256"] = artifact_hashes
    manifest["cross_slot_trajectory_extension"] = {
        "schema_version": "sumo_same_route_trajectory_extension_v1",
        "split": split,
        "scene_id": f"{split}:{int(episode_seed)}",
        "episode_seed": int(episode_seed),
        "seed_protocol": "adaptive_cache_mec_episode_seed_v1",
        "derived_sumo_seed": int(seeds["sumo_seed"]),
        "derived_traffic_seed": int(seeds["traffic_seed"]),
        "dt_s": dt_s,
        "arrival_slots": original_slots,
        "trajectory_slots": trajectory_slots,
        "tail_slots": trajectory_slots - original_slots,
        "warmup_time_s": warmup_s,
        "network_sha256": sha256_file(network_path),
        "sumo_demand_sha256": sha256_file(route_path),
        "roles_sha256": sha256_file(roles_path),
        "original_trajectory_sha256": original_trajectory_sha256,
        "original_prefix_row_count": len(original_prefix),
        "original_prefix_canonical_sha256": prefix_digest,
        "replayed_prefix_canonical_sha256": _prefix_digest(replay_prefix),
        "prefix_exact_match": True,
        "preserved_end_boundary_step": boundary_step,
        "preserved_end_boundary_row_count": len(original_boundary),
        "end_boundary_exact_match": True,
        "fixed_research_ids_present_in_every_tail_slot": True,
        "network_boundary_xy": [float(x_min), float(y_min), float(x_max), float(y_max)],
        "tail_trajectory_row_count": len(rows_to_append),
        "trajectory_sha256": updated_trajectory_sha256,
    }
    manifest["formal_pool_supplement"] = {
        "schema_version": "formal_pool_supplement_v1",
        "split": split,
        "scene_id": f"{split}:{int(episode_seed)}",
        "episode_seed": int(episode_seed),
        "task_arrival_slots": original_slots,
        "physical_trajectory_slots": trajectory_slots,
        "task_randomness_streams": [
            "task_arrival_seed",
            "service_request_seed",
            "workload_seed",
            "popularity_seed",
        ],
        "traffic_randomness_streams": ["traffic_seed", "sumo_seed"],
        "shared_network_sha256": sha256_file(network_path),
    }
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")

    # Re-check generator-level provenance after updating its trajectory hash.
    realization.validate_episode_provenance(episode_dir, config, int(episode_seed))
    return {
        "split": split,
        "episode_seed": int(episode_seed),
        "scene_id": f"{split}:{int(episode_seed)}",
        "arrival_slots": original_slots,
        "trajectory_slots": trajectory_slots,
        "prefix_match": True,
        "prefix_sha256": prefix_digest,
        "trajectory_sha256": updated_trajectory_sha256,
        "task_sha256": sha256_file(episode_dir / "tasks.csv"),
        "roles_sha256": sha256_file(roles_path),
        "network_sha256": sha256_file(network_path),
        "sumo_demand_sha256": sha256_file(route_path),
        "fixed_research_vehicle_count": len(research_ids),
        "tail_rows": len(rows_to_append),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dqn/online_joint_dqn.toml")
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--episode-seed", required=True, type=int)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--trajectory-slots", required=True, type=int)
    parser.add_argument("--sumo-home", default=None)
    args = parser.parse_args(argv)
    config_file = realization.resolve_project_path(args.config)
    episode_dir = realization.resolve_project_path(args.episode_dir)
    result = extend_episode(
        config_file=config_file,
        episode_dir=episode_dir,
        episode_seed=args.episode_seed,
        split=args.split,
        trajectory_slots=args.trajectory_slots,
        sumo_home_arg=args.sumo_home,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
