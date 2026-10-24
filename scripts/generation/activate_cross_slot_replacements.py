#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Activate complete, independently generated cross-slot replacement scenes.

This tool keeps the original episode directories in place, snapshots all
pre-change scene hashes for audit, adds accepted supplemental episode folders,
and writes explicit active-seed and dataset-identity indexes to the formal
pool manifests. Candidate acceptance is limited to data integrity and complete
314-slot fixed-research-vehicle coverage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
TRAINING_ROOT = SCRIPTS_ROOT / "training"
GENERATION_ROOT = SCRIPTS_ROOT / "generation"
for _path in (SRC_ROOT, SCRIPTS_ROOT, TRAINING_ROOT, GENERATION_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import project_config  # noqa: E402
import training_pool_contract as pool_contract  # noqa: E402
from generation import generate_episode_realization as realization  # noqa: E402


REPORT_JSON = PROJECT_ROOT / "reports" / "cross_slot_load_feasibility_2026-10-24.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "dqn" / "online_joint_dqn.toml"
CENTRAL_SPLIT_PATH = PROJECT_ROOT / "environment_data" / "data_split_manifest.json"
AUDIT_DIR = PROJECT_ROOT / "environment_data" / "cross_slot_replacement_audit_2026-10-24"
STAGING_ROOT = PROJECT_ROOT / ".cross_slot_replacements_2026-10-24"
ROAD_SHA256 = "30379989e0f94505e30ed5ec27449367615847ab4baf1bfe2340d8c405388c7d"
ARRIVAL_SLOTS = 300
TRAJECTORY_SLOTS = 314
SPLITS = {
    "train": {
        "pool_relative": Path("environment_data/proposed_joint_training"),
        "seed_start": 20001,
        "count": 300,
        "replacement_count": 13,
    },
    "validation": {
        "pool_relative": Path("environment_data/proposed_joint_validation"),
        "seed_start": 30001,
        "count": 30,
        "replacement_count": 2,
    },
    "test": {
        "pool_relative": Path("environment_data/proposed_joint_test"),
        "seed_start": 40001,
        "count": 100,
        "replacement_count": 6,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _scene_file_hashes(episode_dir: Path) -> dict[str, str]:
    return {
        path.name: _sha256(path)
        for path in sorted(episode_dir.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }


def _scene_identity(split: str, seed: int, file_hashes: dict[str, str]) -> str:
    return _canonical_sha256(
        {
            "schema_version": "formal_scene_identity_v1",
            "scene_id": f"{split}:{seed}",
            "split": split,
            "episode_seed": int(seed),
            "files": file_hashes,
        }
    )


def _dataset_identity(
    split: str,
    seeds: list[int],
    scene_hashes: dict[str, str],
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "formal_episode_dataset_identity_v1",
            "split": split,
            "network_sha256": ROAD_SHA256,
            "slot_duration_s": 1.0,
            "arrival_slots": ARRIVAL_SLOTS,
            "trajectory_slots": TRAJECTORY_SLOTS,
            "active_episode_seeds": seeds,
            "active_episode_scene_sha256": scene_hashes,
        }
    )


def _validate_candidate(
    episode_dir: Path,
    *,
    split: str,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    manifest_path = episode_dir / "episode_manifest.json"
    manifest = _read_json(manifest_path)
    supplement = manifest.get("formal_pool_supplement")
    extension = manifest.get("cross_slot_trajectory_extension")
    expected_scene_id = f"{split}:{seed}"
    if (
        manifest.get("episode_seed") != seed
        or not isinstance(supplement, dict)
        or supplement.get("schema_version") != "formal_pool_supplement_v1"
        or supplement.get("split") != split
        or supplement.get("scene_id") != expected_scene_id
        or supplement.get("task_arrival_slots") != ARRIVAL_SLOTS
        or supplement.get("physical_trajectory_slots") != TRAJECTORY_SLOTS
        or not isinstance(extension, dict)
        or extension.get("scene_id") != expected_scene_id
        or extension.get("prefix_exact_match") is not True
        or extension.get("end_boundary_exact_match") is not True
        or extension.get("fixed_research_ids_present_in_every_tail_slot") is not True
        or extension.get("trajectory_slots") != TRAJECTORY_SLOTS
        or extension.get("arrival_slots") != ARRIVAL_SLOTS
    ):
        raise ValueError(f"candidate extension identity or completeness failed: {episode_dir}")
    realization.validate_episode_provenance(episode_dir, config, seed)
    realization.validate_written_outputs(
        episode_dir,
        task_vehicle_count=int(project_config.get_config_value(config, "simulation.num_task_vehicles")),
        research_vehicle_count=int(project_config.get_config_value(config, "simulation.num_research_vehicles")),
        horizon_steps=int(project_config.get_config_value(config, "prediction.vehicle.horizon")),
    )

    roles = _read_json(episode_dir / "vehicle_roles.json")
    research_ids = [str(value) for value in roles["research_vehicles"]]
    sv_ids = [str(value) for value in roles["service_vehicles"]]
    tv_ids = [str(value) for value in roles["task_vehicles"]]
    expected_research = int(project_config.get_config_value(config, "simulation.num_research_vehicles"))
    expected_sv = int(project_config.get_config_value(config, "simulation.num_service_vehicles"))
    expected_tv = int(project_config.get_config_value(config, "simulation.num_task_vehicles"))
    if len(research_ids) != expected_research or len(set(research_ids)) != expected_research:
        raise ValueError(f"candidate research vehicle count/IDs are invalid: {episode_dir}")
    if len(sv_ids) != expected_sv or len(tv_ids) != expected_tv or set(sv_ids) & set(tv_ids):
        raise ValueError(f"candidate SV/TV roles are invalid: {episode_dir}")
    if set(sv_ids) | set(tv_ids) != set(research_ids):
        raise ValueError(f"candidate role assignment is incomplete: {episode_dir}")

    dt_s = float(project_config.get_config_value(config, "simulation.dt_s"))
    warmup_s = float(project_config.get_config_value(config, "simulation.warmup_time_s"))
    expected_times = {
        round(warmup_s + slot * dt_s, 9) for slot in range(TRAJECTORY_SLOTS)
    }
    expected_boundary = extension.get("network_boundary_xy")
    if not isinstance(expected_boundary, list) or len(expected_boundary) != 4:
        raise ValueError(f"candidate network boundary metadata is invalid: {episode_dir}")
    x_min, y_min, x_max, y_max = (float(value) for value in expected_boundary)
    by_vehicle: dict[str, list[dict[str, str]]] = {vehicle_id: [] for vehicle_id in research_ids}
    with (episode_dir / "trajectory.csv").open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            vehicle_id = str(row["vehicle_id"])
            if vehicle_id not in by_vehicle:
                continue
            time_s = float(row["time_s"])
            if time_s in expected_times:
                by_vehicle[vehicle_id].append(row)
    for vehicle_id, rows in by_vehicle.items():
        times = [float(row["time_s"]) for row in rows]
        if len(times) != TRAJECTORY_SLOTS or set(times) != expected_times:
            raise ValueError(
                f"candidate vehicle lacks unique positions over 314 slots: "
                f"seed={seed}, vehicle={vehicle_id}, rows={len(rows)}"
            )
        for row in rows:
            for key in ("x_m", "y_m", "speed_mps", "acceleration_mps2", "lane_position_m", "distance_m"):
                if not math.isfinite(float(row[key])):
                    raise ValueError(f"candidate contains non-finite trajectory data: {seed}/{vehicle_id}")
            x_m, y_m = float(row["x_m"]), float(row["y_m"])
            if not (x_min - 1e-6 <= x_m <= x_max + 1e-6 and y_min - 1e-6 <= y_m <= y_max + 1e-6):
                raise ValueError(f"candidate position outside network boundary: {seed}/{vehicle_id}")
            if not row["road_id"].strip() or not row["lane_id"].strip():
                raise ValueError(f"candidate position has no SUMO road/lane: {seed}/{vehicle_id}")

    task_slots: set[int] = set()
    with (episode_dir / "tasks.csv").open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            slot = int(row["slot"])
            if slot < 0 or slot >= ARRIVAL_SLOTS:
                raise ValueError(f"candidate task row is outside slots 0..299: {seed}/{slot}")
            if row["vehicle_id"] not in set(tv_ids):
                raise ValueError(f"candidate task is not associated with a TV: {seed}/{row['vehicle_id']}")
            task_slots.add(slot)
    if task_slots != set(range(ARRIVAL_SLOTS)):
        raise ValueError(f"candidate task data do not cover slots 0..299: {episode_dir}")
    if extension.get("network_sha256") != ROAD_SHA256:
        raise ValueError(f"candidate uses a different road network: {episode_dir}")
    return {
        "split": split,
        "episode_seed": seed,
        "scene_id": expected_scene_id,
        "episode_manifest_sha256": _sha256(manifest_path),
        "trajectory_sha256": _sha256(episode_dir / "trajectory.csv"),
        "tasks_sha256": _sha256(episode_dir / "tasks.csv"),
        "roles_sha256": _sha256(episode_dir / "vehicle_roles.json"),
        "prediction_scenarios_sha256": _sha256(episode_dir / "prediction_scenarios.npz"),
        "sumo_demand_sha256": _sha256(episode_dir / "sumo_demand.rou.xml"),
        "trajectory_prefix_sha256": extension["original_prefix_canonical_sha256"],
        "research_vehicle_count": len(research_ids),
        "tail_rows": extension["tail_trajectory_row_count"],
        "file_hashes": _scene_file_hashes(episode_dir),
    }


def _snapshot_old_episodes(
    *,
    split_report: dict[str, list[dict[str, Any]]],
    source_manifests: dict[str, dict[str, Any]],
    original_train_baseline: dict[str, Any],
) -> dict[str, Any]:
    baseline_by_seed = {
        str(row["episode_seed"]): row
        for row in original_train_baseline.get("episodes", [])
    }
    snapshot: dict[str, Any] = {
        "schema_version": "pre_cross_slot_replacement_episode_hash_snapshot_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prior_audit_report_path": REPORT_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "prior_audit_report_sha256": _sha256(REPORT_JSON),
        "road_network_sha256": ROAD_SHA256,
        "split_seed_ranges": {},
        "episodes": {},
    }
    for split, info in SPLITS.items():
        pool = PROJECT_ROOT / info["pool_relative"]
        start, end = info["seed_start"], info["seed_start"] + info["count"] - 1
        snapshot["split_seed_ranges"][split] = [start, end]
        manifest = source_manifests[split]
        report_rows = {
            int(row["episode_seed"]): row for row in split_report[split]
        }
        for seed in range(start, end + 1):
            episode_dir = pool / f"episode_{seed}"
            if not episode_dir.is_dir():
                raise FileNotFoundError(f"original episode directory missing: {episode_dir}")
            file_hashes = _scene_file_hashes(episode_dir)
            if split == "train":
                baseline_row = baseline_by_seed.get(str(seed))
                if not isinstance(baseline_row, dict):
                    raise ValueError(f"original Train baseline is missing seed {seed}")
                expected_hashes = {
                    row["relative_path"]: row["sha256"]
                    for row in baseline_row.get("files", [])
                }
                if expected_hashes != file_hashes:
                    raise ValueError(f"original Train scene differs from its baseline: seed={seed}")
            elif seed in report_rows:
                report_row = report_rows[seed]
                if file_hashes.get("trajectory.csv") != report_row.get("source_trajectory_sha256"):
                    raise ValueError(f"original {split} trajectory differs from prior audit: seed={seed}")
                if file_hashes.get("tasks.csv") != report_row.get("tasks_sha256"):
                    raise ValueError(f"original {split} tasks differ from prior audit: seed={seed}")
            invalid = not bool(report_rows[seed]["full_314_slot_research_coverage"])
            snapshot["episodes"][f"{split}:{seed}"] = {
                "split": split,
                "episode_seed": seed,
                "scene_id": f"{split}:{seed}",
                "prior_314_slot_coverage": not invalid,
                "original_directory": str(episode_dir.relative_to(PROJECT_ROOT).as_posix()),
                "episode_manifest_sha256": file_hashes.get("episode_manifest.json"),
                "trajectory_sha256": file_hashes.get("trajectory.csv"),
                "tasks_sha256": file_hashes.get("tasks.csv"),
                "file_hashes": file_hashes,
                "prior_audit_record": report_rows[seed],
                "formal_pool_source_manifest_sha256": manifest.get(
                    "source_episode_manifest_sha256", {}
                ).get(str(seed)),
            }
    return snapshot


def activate_replacements() -> dict[str, Any]:
    if AUDIT_DIR.exists():
        raise FileExistsError(f"audit destination already exists; refusing to overwrite: {AUDIT_DIR}")
    activation_report_path = PROJECT_ROOT / "reports" / "cross_slot_replacements_2026-10-24.json"
    if activation_report_path.exists():
        raise FileExistsError(
            f"activation report already exists; refusing to overwrite: {activation_report_path}"
        )
    config = realization.load_and_validate_config(CONFIG_PATH)
    prior_report = _read_json(REPORT_JSON)
    report_by_split = {
        split: [row for row in prior_report["episodes"] if row["split"] == split]
        for split in SPLITS
    }
    invalid_by_split = {
        split: sorted(
            int(row["episode_seed"])
            for row in report_by_split[split]
            if not bool(row["full_314_slot_research_coverage"])
        )
        for split in SPLITS
    }
    expected_invalid_counts = {"train": 13, "validation": 2, "test": 6}
    for split, expected_count in expected_invalid_counts.items():
        if len(invalid_by_split[split]) != expected_count:
            raise ValueError(f"prior audit reports an unexpected invalid count in {split}")

    old_manifests = {
        split: _read_json(PROJECT_ROOT / info["pool_relative"] / "formal_pool_manifest.json")
        for split, info in SPLITS.items()
    }
    old_manifest_hashes = {
        split: _sha256(PROJECT_ROOT / info["pool_relative"] / "formal_pool_manifest.json")
        for split, info in SPLITS.items()
    }
    old_data_split = _read_json(CENTRAL_SPLIT_PATH)
    old_train_baseline = _read_json(
        PROJECT_ROOT / SPLITS["train"]["pool_relative"] / "formal_pool_integrity_baseline.json"
    )
    if old_manifest_hashes["train"] != pool_contract.FORMAL_TRAINING_POOL_MANIFEST_SHA256:
        raise ValueError("original Train formal manifest identity changed before activation")
    old_baseline_path = PROJECT_ROOT / SPLITS["train"]["pool_relative"] / "formal_pool_integrity_baseline.json"
    if _sha256(old_baseline_path) != pool_contract.FORMAL_TRAINING_POOL_INTEGRITY_BASELINE_SHA256:
        raise ValueError("original Train integrity baseline identity changed before activation")
    for split, manifest in old_manifests.items():
        actual = _sha256(CENTRAL_SPLIT_PATH)
        if manifest.get("data_split_manifest_sha256") != actual:
            raise ValueError(f"original {split} pool points to a changed data split manifest")

    accepted: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for split, info in SPLITS.items():
        stage_pool = STAGING_ROOT / split
        candidates: list[tuple[Path, dict[str, Any]]] = []
        if stage_pool.is_dir():
            for episode_dir in sorted(stage_pool.glob("episode_*"), key=lambda path: path.name):
                try:
                    seed = int(episode_dir.name.removeprefix("episode_"))
                    candidate = _validate_candidate(
                        episode_dir,
                        split=split,
                        config=config,
                        seed=seed,
                    )
                except Exception as exc:
                    # A candidate that fails the physical/data completeness contract is rejected.
                    print(f"REJECT {split} {episode_dir.name}: {type(exc).__name__}: {exc}", flush=True)
                    continue
                candidates.append((episode_dir, candidate))
        if len(candidates) < info["replacement_count"]:
            raise ValueError(
                f"not enough complete candidates for {split}: "
                f"needed={info['replacement_count']}, complete={len(candidates)}"
            )
        selected = candidates[: info["replacement_count"]]
        selected_seeds = [item[1]["episode_seed"] for item in selected]
        if len(selected_seeds) != len(set(selected_seeds)):
            raise ValueError(f"duplicate accepted candidate seeds in {split}")
        accepted[split] = selected

    # Create immutable pre-change evidence before copying or editing any official data.
    snapshot = _snapshot_old_episodes(
        split_report=report_by_split,
        source_manifests=old_manifests,
        original_train_baseline=old_train_baseline,
    )
    AUDIT_DIR.mkdir(parents=True, exist_ok=False)
    audit_copies = {
        "data_split_manifest.before.json": CENTRAL_SPLIT_PATH,
        "train_formal_pool_manifest.before.json": PROJECT_ROOT / SPLITS["train"]["pool_relative"] / "formal_pool_manifest.json",
        "validation_formal_pool_manifest.before.json": PROJECT_ROOT / SPLITS["validation"]["pool_relative"] / "formal_pool_manifest.json",
        "test_formal_pool_manifest.before.json": PROJECT_ROOT / SPLITS["test"]["pool_relative"] / "formal_pool_manifest.json",
        "train_integrity_baseline.before.json": old_baseline_path,
        "prior_load_feasibility_report.json": REPORT_JSON,
        "prior_load_feasibility_report.md": REPORT_JSON.with_suffix(".md"),
    }
    for name, source in audit_copies.items():
        shutil.copy2(source, AUDIT_DIR / name)
    _write_json(AUDIT_DIR / "pre_replacement_episode_hash_snapshot.json", snapshot)

    candidate_metadata: dict[str, dict[str, dict[str, Any]]] = {split: {} for split in SPLITS}
    for split, rows in accepted.items():
        pool = PROJECT_ROOT / SPLITS[split]["pool_relative"]
        for source_dir, candidate in rows:
            seed = int(candidate["episode_seed"])
            destination = pool / f"episode_{seed}"
            if destination.exists():
                raise FileExistsError(f"refusing to overwrite an episode directory: {destination}")
            shutil.copytree(source_dir, destination)
            copied_hashes = _scene_file_hashes(destination)
            if copied_hashes != candidate["file_hashes"]:
                raise RuntimeError(f"supplement copy hash mismatch: {destination}")
            candidate_metadata[split][str(seed)] = candidate

    per_split_identity: dict[str, dict[str, Any]] = {}
    for split, info in SPLITS.items():
        pool = PROJECT_ROOT / info["pool_relative"]
        original_seeds = list(range(info["seed_start"], info["seed_start"] + info["count"]))
        excluded = invalid_by_split[split]
        replacements = sorted(int(seed) for seed in candidate_metadata[split])
        active = sorted((set(original_seeds) - set(excluded)) | set(replacements))
        if len(active) != info["count"]:
            raise AssertionError(f"active episode count changed in {split}")
        scene_hashes = {
            str(seed): pool_contract._scene_identity_sha256(
                pool / f"episode_{seed}", split=split, seed=seed
            )
            for seed in active
        }
        identity = _dataset_identity(split, active, scene_hashes)
        per_split_identity[split] = {
            "active_episode_seeds": active,
            "replacement_episode_seeds": replacements,
            "excluded_episode_seeds": excluded,
            "active_episode_scene_sha256": scene_hashes,
            "dataset_identity_sha256": identity,
        }

    data_split = old_data_split
    data_split["protocol_version"] = 3
    data_split["arrival_slots"] = ARRIVAL_SLOTS
    data_split["trajectory_slots"] = TRAJECTORY_SLOTS
    data_split["tail_slots"] = list(range(ARRIVAL_SLOTS, TRAJECTORY_SLOTS))
    data_split["replacement_generation_provenance"] = {
        "generator_entrypoint": "scripts/generation/generate_episode_realization.py",
        "trajectory_extension_tool": "scripts/generation/extend_formal_episode_trajectory.py",
        "activation_tool": "scripts/generation/activate_cross_slot_replacements.py",
        "episode_seed_protocol": "adaptive_cache_mec_episode_seed_v1",
        "seed_streams_are_domain_separated": True,
        "traffic_and_task_randomness_are_independent_streams": True,
        "network_sha256": ROAD_SHA256,
        "runtime_config_path": CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "runtime_config_sha256": _sha256(CONFIG_PATH),
        "episode_generator_sha256": _sha256(GENERATION_ROOT / "generate_episode_realization.py"),
        "trajectory_extension_tool_sha256": _sha256(GENERATION_ROOT / "extend_formal_episode_trajectory.py"),
        "activation_tool_sha256": _sha256(GENERATION_ROOT / "activate_cross_slot_replacements.py"),
        "selection_rule": "complete 314-slot fixed-research-vehicle trajectory and exact replayed prefix only",
        "algorithm_reward_or_evaluation_used_for_selection": False,
    }
    for split, identity in per_split_identity.items():
        data_split[split]["active_episode_count"] = len(identity["active_episode_seeds"])
        data_split[split]["active_episode_seeds"] = identity["active_episode_seeds"]
        data_split[split]["replacement_episode_seeds"] = identity["replacement_episode_seeds"]
        data_split[split]["excluded_incomplete_episode_seeds"] = identity["excluded_episode_seeds"]
        data_split[split]["active_episode_scene_sha256"] = identity[
            "active_episode_scene_sha256"
        ]
        data_split[split]["dataset_identity_sha256"] = identity["dataset_identity_sha256"]
    all_active = [
        seed
        for split in SPLITS
        for seed in per_split_identity[split]["active_episode_seeds"]
    ]
    if len(all_active) != len(set(all_active)):
        raise ValueError("active Train/Validation/Test seed lists overlap")
    _write_json(CENTRAL_SPLIT_PATH, data_split)
    data_split_hash = _sha256(CENTRAL_SPLIT_PATH)

    for split, info in SPLITS.items():
        pool = PROJECT_ROOT / info["pool_relative"]
        manifest_path = pool / "formal_pool_manifest.json"
        manifest = old_manifests[split]
        identity = per_split_identity[split]
        manifest["episode_selection"] = {
            "schema_version": "active_episode_seed_selection_v1",
            "source_seed_range": [info["seed_start"], info["seed_start"] + info["count"] - 1],
            "active_episode_seeds": identity["active_episode_seeds"],
            "replacement_episode_seeds": identity["replacement_episode_seeds"],
            "excluded_episode_seeds": identity["excluded_episode_seeds"],
            "arrival_slots": ARRIVAL_SLOTS,
            "trajectory_slots": TRAJECTORY_SLOTS,
            "selection_rule": "complete fixed-research-vehicle trajectory and exact SUMO replay prefix only",
        }
        manifest["active_episode_scene_sha256"] = identity[
            "active_episode_scene_sha256"
        ]
        manifest["dataset_identity_sha256"] = identity["dataset_identity_sha256"]
        manifest["data_split_manifest_sha256"] = data_split_hash
        manifest["arrival_slots"] = ARRIVAL_SLOTS
        manifest["trajectory_slots"] = TRAJECTORY_SLOTS
        manifest["episode_manifest_schemas"] = {
            "legacy_transformed_episode": "formal_task_transform_v1",
            "supplemental_episode": "episode_manifest_v2+formal_pool_supplement_v1",
        }
        manifest["required_artifacts_by_episode_schema"] = {
            "formal_task_transform_v1": list(manifest.get("required_artifacts", [])),
            "episode_manifest_v2+formal_pool_supplement_v1": [
                "popularity.csv",
                "popularity_history.csv",
                "popularity_predictions.csv",
                "prediction_scenarios.npz",
                "sumo_demand.meta.json",
                "sumo_demand.rou.xml",
                "tasks.csv",
                "trajectory.csv",
                "vehicle_predictions.csv",
                "vehicle_roles.json",
            ],
        }
        manifest["replacement_episode_manifest_sha256"] = {
            seed: candidate_metadata[split][seed]["episode_manifest_sha256"]
            for seed in sorted(candidate_metadata[split], key=int)
        }
        manifest["replacement_scene_metadata"] = {
            seed: {
                key: value
                for key, value in candidate_metadata[split][seed].items()
                if key != "file_hashes"
            }
            for seed in sorted(candidate_metadata[split], key=int)
        }
        manifest["excluded_episode_evidence"] = {
            str(seed): snapshot["episodes"][f"{split}:{seed}"]
            for seed in identity["excluded_episode_seeds"]
        }
        _write_json(manifest_path, manifest)

    train_manifest_path = PROJECT_ROOT / SPLITS["train"]["pool_relative"] / "formal_pool_manifest.json"
    train_manifest_sha = _sha256(train_manifest_path)
    train_pool = PROJECT_ROOT / SPLITS["train"]["pool_relative"]
    baseline_episodes: list[dict[str, Any]] = []
    total_file_count = 0
    for seed in per_split_identity["train"]["active_episode_seeds"]:
        episode_dir = train_pool / f"episode_{seed}"
        files = []
        total_bytes = 0
        for path in sorted(episode_dir.iterdir(), key=lambda item: item.name):
            if not path.is_file():
                continue
            stats = path.stat()
            file_record = {
                "relative_path": path.name,
                "sha256": _sha256(path),
                "size_bytes": int(stats.st_size),
            }
            files.append(file_record)
            total_bytes += int(stats.st_size)
        total_file_count += len(files)
        baseline_episodes.append(
            {
                "episode_seed": int(seed),
                "relative_directory": f"environment_data/proposed_joint_training/episode_{seed}",
                "episode_manifest_sha256": next(
                    row["sha256"] for row in files if row["relative_path"] == "episode_manifest.json"
                ),
                "file_count": len(files),
                "total_bytes": total_bytes,
                "files": files,
            }
        )
    baseline = {
        **{
            key: value
            for key, value in old_train_baseline.items()
            if key not in {"episodes", "pool_manifest_sha256", "created_at_utc", "integrity_baseline_type", "schema_version", "episode_file_count"}
        },
        "schema_version": "formal_task_data_integrity_baseline_v2",
        "integrity_baseline_type": "ACTIVE_EPISODE_SNAPSHOT",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pool_manifest_sha256": train_manifest_sha,
        "data_split_manifest_sha256": data_split_hash,
        "episode_count": SPLITS["train"]["count"],
        "episode_file_count": total_file_count,
        "active_episode_seeds": per_split_identity["train"]["active_episode_seeds"],
        "episodes": baseline_episodes,
        "pre_replacement_integrity_baseline_sha256": _sha256(AUDIT_DIR / "train_integrity_baseline.before.json"),
        "pre_replacement_episode_hash_snapshot": "environment_data/cross_slot_replacement_audit_2026-10-24/pre_replacement_episode_hash_snapshot.json",
    }
    _write_json(train_pool / "formal_pool_integrity_baseline.json", baseline)

    original_files_unchanged = True
    for scene_id, old_scene in snapshot["episodes"].items():
        split, seed_text = scene_id.split(":", 1)
        episode_dir = PROJECT_ROOT / SPLITS[split]["pool_relative"] / f"episode_{int(seed_text)}"
        if _scene_file_hashes(episode_dir) != old_scene["file_hashes"]:
            original_files_unchanged = False
            raise RuntimeError(f"an original episode file changed during activation: {scene_id}")

    report_payload = {
        "schema_version": "cross_slot_replacement_activation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "audit_directory": str(AUDIT_DIR),
        "data_split_manifest_sha256": data_split_hash,
        "train_pool_manifest_sha256": train_manifest_sha,
        "train_integrity_baseline_sha256": _sha256(train_pool / "formal_pool_integrity_baseline.json"),
        "train_episode_file_count": total_file_count,
        "road_network_sha256": ROAD_SHA256,
        "arrival_slots": ARRIVAL_SLOTS,
        "trajectory_slots": TRAJECTORY_SLOTS,
        "selection_rule": "complete 314-slot fixed-research-vehicle coverage and exact original-prefix replay only",
        "algorithm_reward_or_evaluation_used_for_selection": False,
        "all_original_430_episode_files_hashes_unchanged": original_files_unchanged,
        "splits": {
            split: {
                **per_split_identity[split],
                "replacement_scene_metadata": {
                    seed: {
                        key: value
                        for key, value in candidate_metadata[split][seed].items()
                        if key != "file_hashes"
                    }
                    for seed in sorted(candidate_metadata[split], key=int)
                },
            }
            for split in SPLITS
        },
    }
    _write_json(activation_report_path, report_payload)
    return report_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="activate the already staged complete candidates")
    args = parser.parse_args(argv)
    if not args.apply:
        parser.error("--apply is required to update the formal episode indexes")
    result = activate_replacements()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
