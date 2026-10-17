"""Single formal runner for the 500–1000 KB multi-node main experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for candidate in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

CONFIG_RELATIVE = Path("configs/proposed/system_config.toml")
POOL_RELATIVE = Path("environment_data/proposed_joint_training")
VALIDATION_POOL_RELATIVE = Path("environment_data/proposed_joint_validation")
RUN_DIRECTORY_RELATIVE = Path("runs/stageB3_10j_multi_node_v4_mainline_500_1000")
POOL_MANIFEST_NAME = "formal_pool_manifest.json"
POOL_BASELINE_NAME = "formal_pool_integrity_baseline.json"
POOL_BASELINE_SCHEMA = "formal_task_data_integrity_baseline_v1"
EPISODE_COUNT = 300
EPISODE_SEED_START = 20001
EPISODE_SEED_END = EPISODE_SEED_START + EPISODE_COUNT - 1
PPO_INITIALIZATION_SEED = 27005
VALIDATION_SEED_START = 30001
VALIDATION_EPISODE_COUNT = 30
VALIDATION_DIRECTORY_NAME = "validation"
MACRO_ACTOR_LEARNING_RATE_RUNS = {
    1.0e-4: Path("runs/stageB3_14h_fresh_macro_actor_lr_1e_minus_4_500_1000"),
    5.0e-4: Path("runs/stageB3_14h_fresh_macro_actor_lr_5e_minus_4_500_1000"),
    1.0e-3: Path("runs/stageB3_14h_fresh_macro_actor_lr_1e_minus_3_500_1000"),
    5.0e-3: Path("runs/stageB3_14h_fresh_macro_actor_lr_5e_minus_3_500_1000"),
}
TASK_DATA_MODEL = "500_1000_KB"
TASK_DATA_EPISODE_SCHEMA = "formal_task_transform_v1"
REQUIRED_TRAINING_ARTIFACTS = (
    "trajectory.csv",
    "vehicle_roles.json",
    "tasks.csv",
    "popularity.csv",
    "popularity_history.csv",
    "vehicle_predictions.csv",
    "popularity_predictions.csv",
    "prediction_scenarios.npz",
)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_file_sha256(path: Path, *, expected_size: int | None = None) -> str:
    before = path.stat()
    if expected_size is not None and before.st_size != expected_size:
        raise ValueError(f"File size changed from the expected snapshot: {path}")
    digest = _sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed while it was being hashed: {path}")
    return digest


def _assert_close(field: str, actual: Any, expected: float) -> None:
    if not math.isclose(float(actual), expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(
            f"Formal main-experiment config mismatch for {field}: "
            f"{actual!r} != {expected!r}"
        )


def _load_main_experiment_contract() -> tuple[dict[str, Any], str, str, str]:
    from common import project_config
    from generation import generate_episode_realization

    config_path = PROJECT_ROOT / CONFIG_RELATIVE
    cfg = project_config.load_config(str(config_path))
    project_config.validate_for_formal_experiment(cfg)

    expected_values = {
        "nodes.cloud.cpu_capacity_cycles_per_s": (
            cfg["nodes"]["cloud"]["cpu_capacity_cycles_per_s"],
            40.0e9,
        ),
        "nodes.cloud.cloud_delay_factor": (
            cfg["nodes"]["cloud"]["cloud_delay_factor"],
            13.0,
        ),
        "nodes.cloud.uplink_rate_bps": (
            cfg["nodes"]["cloud"]["uplink_rate_bps"],
            20.0e6,
        ),
        "nodes.cloud.total_bandwidth_bps": (
            cfg["nodes"]["cloud"]["total_bandwidth_bps"],
            20.0e6,
        ),
        "nodes.rsu.cpu_cycles_per_s": (cfg["nodes"]["rsu"]["cpu_cycles_per_s"], 10.0e9),
        "nodes.rsu.bandwidth_hz": (cfg["nodes"]["rsu"]["bandwidth_hz"], 10.0e6),
        "nodes.rsu.cache_capacity_mb": (cfg["nodes"]["rsu"]["cache_capacity_mb"], 25000.0),
        "nodes.uav.cpu_cycles_per_s": (cfg["nodes"]["uav"]["cpu_cycles_per_s"], 5.0e9),
        "nodes.uav.bandwidth_hz": (cfg["nodes"]["uav"]["bandwidth_hz"], 10.0e6),
        "nodes.uav.cache_capacity_mb": (cfg["nodes"]["uav"]["cache_capacity_mb"], 20000.0),
        "nodes.sv.cpu_cycles_per_s": (cfg["nodes"]["sv"]["cpu_cycles_per_s"], 1.0e9),
        "nodes.sv.bandwidth_hz": (cfg["nodes"]["sv"]["bandwidth_hz"], 10.0e6),
        "nodes.sv.cache_capacity_mb": (cfg["nodes"]["sv"]["cache_capacity_mb"], 12000.0),
        "communication.ground.bandwidth_hz": (
            cfg["communication"]["ground"]["bandwidth_hz"],
            10.0e6,
        ),
        "communication.ground.carrier_frequency_hz": (
            cfg["communication"]["ground"]["carrier_frequency_hz"],
            5.9e9,
        ),
        "communication.noise.psd_dbm_per_hz": (
            cfg["communication"]["noise"]["psd_dbm_per_hz"],
            -174.0,
        ),
        "communication.noise.receiver_noise_figure_db": (
            cfg["communication"]["noise"]["receiver_noise_figure_db"],
            9.0,
        ),
        "communication.cloud.uplink_rate_bps": (
            cfg["communication"]["cloud"]["uplink_rate_bps"],
            20.0e6,
        ),
        "service.generation.input_size_min_kb": (
            cfg["service"]["generation"]["input_size_min_kb"],
            500.0,
        ),
        "service.generation.input_size_max_kb": (
            cfg["service"]["generation"]["input_size_max_kb"],
            1000.0,
        ),
    }
    for field, (actual, expected) in expected_values.items():
        _assert_close(field, actual, expected)
    task_contract_values = {
        "simulation.dt_s": (cfg["simulation"]["dt_s"], 1.0),
        "simulation.episode_time_s": (cfg["simulation"]["episode_time_s"], 300.0),
        "simulation.num_task_vehicles": (cfg["simulation"]["num_task_vehicles"], 20.0),
        "simulation.num_service_vehicles": (cfg["simulation"]["num_service_vehicles"], 5.0),
        "service.count": (cfg["service"]["count"], 10.0),
        "service.generation.service_size_min_gb": (
            cfg["service"]["generation"]["service_size_min_gb"], 2.0
        ),
        "service.generation.service_size_max_gb": (
            cfg["service"]["generation"]["service_size_max_gb"], 20.0
        ),
        "service.generation.cycles_per_bit_min": (
            cfg["service"]["generation"]["cycles_per_bit_min"], 400.0
        ),
        "service.generation.cycles_per_bit_max": (
            cfg["service"]["generation"]["cycles_per_bit_max"], 1000.0
        ),
        "task.arrival_probability": (cfg["task"]["arrival_probability"], 0.7),
        "task.popularity_zeta": (cfg["task"]["popularity_zeta"], 0.8),
        "task.input_relative_jitter": (cfg["task"]["input_relative_jitter"], 0.2),
        "task.load_relative_jitter": (cfg["task"]["load_relative_jitter"], 0.1),
    }
    for field, (actual, expected) in task_contract_values.items():
        _assert_close(field, actual, expected)
    if cfg["nodes"]["cloud"].get("resource_mode") != "finite_pool":
        raise ValueError("Formal Cloud contract must use finite_pool.")
    if cfg["algorithm"]["matching"].get("cloud_fallback_mode") != "RESOURCE_AWARE_FALLBACK":
        raise ValueError("Formal Matching contract must use RESOURCE_AWARE_FALLBACK.")
    ppo = cfg["algorithm"]["ppo"]
    if ppo.get("policy_schema") != project_config.MULTI_NODE_FACTORIZED_POLICY_SCHEMA:
        raise ValueError("Formal runtime policy contract must use the multi-node v4 schema.")
    if ppo.get("cache_action_mode") != project_config.CACHE_ACTION_MODE_MULTI_NODE_FACTORIZED:
        raise ValueError("Formal runtime policy contract must use MULTI_NODE_FACTORIZED.")
    if (
        ppo.get("cache_reconfiguration_model")
        != project_config.CACHE_RECONFIGURATION_MODEL_ABSTRACT
    ):
        raise ValueError("Formal Cache contract must use abstract macro-boundary reconfiguration.")
    if (
        int(cfg["simulation"]["num_service_vehicles"]) != 5
        or int(cfg["nodes"]["rsu"]["count"]) != 2
        or int(cfg["nodes"]["uav"]["count"]) != 2
    ):
        raise ValueError("Formal main experiment requires nine Cache nodes and two UAVs.")
    if cfg["communication"]["noise"].get("model") != "BANDWIDTH_DEPENDENT_NOISE_V1":
        raise ValueError("Formal Edge communication contract must use bandwidth-dependent noise.")
    if cfg["communication"].get("resource_model") != {
        "inter_node_spectrum": "ORTHOGONAL_INDEPENDENT",
        "interference": "NONE",
    }:
        raise ValueError("Formal communication resource model does not match B3-10C.")
    if "tv_local_cache" in cfg.get("paths", {}):
        raise ValueError("Formal runtime config still declares a TV local-cache artifact.")

    if int(ppo.get("realizations_per_policy_update", 0)) != 4:
        raise ValueError("Formal PPO contract must update once per four Episodes.")
    if ppo.get("macro_actor_learning_rate_schedule") != "constant":
        raise ValueError("Formal Macro Actor learning-rate schedule must be constant.")
    if ppo.get("optimizer_learning_rate_schedule") != "constant":
        raise ValueError("All formal optimizer learning-rate schedules must be constant.")
    if EPISODE_COUNT % int(ppo["realizations_per_policy_update"]):
        raise ValueError("Training Episode count must align with the PPO update group size.")
    if int(ppo.get("data_protocol", {}).get("episode_slots", 0)) != 300:
        raise ValueError("Formal PPO contract must retain 300 Training slots per Episode.")

    raw_hash = project_config.raw_config_sha256(config_path)
    generation_fingerprint = generate_episode_realization.generation_config_fingerprint(cfg)
    semantic_hash = project_config.semantic_config_hash(cfg)
    return cfg, raw_hash, semantic_hash, generation_fingerprint


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Formal task-data {field} is missing or invalid.")
    return value


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _task_data_contract(cfg: dict[str, Any], catalog_path: str, catalog_sha256: str) -> dict[str, Any]:
    service = cfg["service"]
    generation = service["generation"]
    task = cfg["task"]
    return {
        "schema_version": "formal_task_data_contract_v1",
        "task_data_model": TASK_DATA_MODEL,
        "service_count": int(service["count"]),
        "input_size_min_kb": float(generation["input_size_min_kb"]),
        "input_size_max_kb": float(generation["input_size_max_kb"]),
        "service_size_min_gb": float(generation["service_size_min_gb"]),
        "service_size_max_gb": float(generation["service_size_max_gb"]),
        "cycles_per_bit_min": float(generation["cycles_per_bit_min"]),
        "cycles_per_bit_max": float(generation["cycles_per_bit_max"]),
        "input_relative_jitter": float(task["input_relative_jitter"]),
        "load_relative_jitter": float(task["load_relative_jitter"]),
        "arrival_probability": float(task["arrival_probability"]),
        "popularity_zeta": float(task["popularity_zeta"]),
        "episode_slots": int(round(float(cfg["simulation"]["episode_time_s"]) / float(cfg["simulation"]["dt_s"]))),
        "slot_duration_s": float(cfg["simulation"]["dt_s"]),
        "service_catalog_path": catalog_path,
        "service_catalog_sha256": catalog_sha256,
    }


def _episode_file_inventory(episode_dir: Path) -> list[tuple[str, Path, int]]:
    inventory: list[tuple[str, Path, int]] = []
    for path in sorted(
        episode_dir.rglob("*"), key=lambda item: item.relative_to(episode_dir).as_posix()
    ):
        if path.is_symlink():
            raise ValueError(f"Formal Episode data must not contain symlinks: {path}")
        if path.is_file():
            inventory.append(
                (path.relative_to(episode_dir).as_posix(), path, path.stat().st_size)
            )
        elif not path.is_dir():
            raise ValueError(f"Formal Episode contains a non-regular entry: {path}")
    return inventory


def _validate_pool_root_inventory(
    pool_path: Path, *, seed_start: int, episode_count: int, include_catalog: bool
) -> None:
    expected_episode_dirs = {
        f"episode_{seed}" for seed in range(seed_start, seed_start + episode_count)
    }
    expected_files = {POOL_MANIFEST_NAME, POOL_BASELINE_NAME}
    if include_catalog:
        expected_files.add("service_catalog_500_1000.json")
    actual_entries = {entry.name: entry for entry in pool_path.iterdir()}
    expected_names = expected_episode_dirs | expected_files
    if set(actual_entries) != expected_names:
        missing = sorted(expected_names - set(actual_entries))
        unexpected = sorted(set(actual_entries) - expected_names)
        raise ValueError(
            f"Formal {pool_path.name} root inventory mismatch; "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}."
        )
    for name, entry in actual_entries.items():
        if entry.is_symlink():
            raise ValueError(f"Formal pool entries must not be symlinks: {entry}")
        if name in expected_episode_dirs and not entry.is_dir():
            raise ValueError(f"Formal Episode entry is not a directory: {entry}")
        if name in expected_files and not entry.is_file():
            raise ValueError(f"Formal pool file entry is not a file: {entry}")


def _validate_formal_task_pool(
    pool_relative: Path,
    *,
    split: str,
    seed_start: int,
    episode_count: int,
    cfg: dict[str, Any],
    generation_fingerprint: str,
) -> Path:
    from common import project_config

    pool_path = PROJECT_ROOT / pool_relative
    seed_end = seed_start + episode_count - 1
    _validate_pool_root_inventory(
        pool_path,
        seed_start=seed_start,
        episode_count=episode_count,
        include_catalog=(split == "train"),
    )
    manifest_path = pool_path / POOL_MANIFEST_NAME
    baseline_path = pool_path / POOL_BASELINE_NAME
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Formal pool manifest root must be an object: {manifest_path}")

    config_path = PROJECT_ROOT / CONFIG_RELATIVE
    raw_config_sha256 = project_config.raw_config_sha256(config_path)
    semantic_config_sha256 = project_config.semantic_config_hash(cfg)
    catalog_relative = str(project_config.get_config_value(cfg, "paths.service_catalog"))
    catalog_path = project_config.resolve_path(PROJECT_ROOT, catalog_relative)
    catalog_sha256 = _stable_file_sha256(catalog_path)
    contract = _task_data_contract(cfg, catalog_relative, catalog_sha256)
    contract_sha256 = _json_sha256(contract)
    split_manifest_path = PROJECT_ROOT / "environment_data/data_split_manifest.json"
    split_manifest_sha256 = _stable_file_sha256(split_manifest_path)

    expected_manifest = {
        "schema_version": "formal_task_data_pool_manifest_v1",
        "task_data_model": TASK_DATA_MODEL,
        "split": split,
        "seed_start": seed_start,
        "seed_end": seed_end,
        "episode_count": episode_count,
        "config_path": CONFIG_RELATIVE.as_posix(),
        "config_sha256": raw_config_sha256,
        "semantic_config_sha256": semantic_config_sha256,
        "target_generation_config_fingerprint": generation_fingerprint,
        "task_data_contract": contract,
        "task_data_contract_sha256": contract_sha256,
        "service_catalog_path": catalog_relative,
        "service_catalog_sha256": catalog_sha256,
        "data_split_manifest_sha256": split_manifest_sha256,
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"Formal {split} Pool {field} mismatch: "
                f"{manifest.get(field)!r} != {expected!r}."
            )

    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split_manifest.get("task_data_model") != TASK_DATA_MODEL:
        raise ValueError("Formal data split manifest does not declare the current task model.")
    if split_manifest.get("task_data_contract_sha256") != contract_sha256:
        raise ValueError("Formal data split manifest task-data contract hash mismatch.")

    try:
        catalog_obj = json.loads(catalog_path.read_text(encoding="utf-8"))
        services = catalog_obj["services"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid formal service catalog: {catalog_path}") from exc
    if len(services) != 10 or [int(row["service_id"]) for row in services] != list(range(10)):
        raise ValueError("Formal 500–1000 KB service catalog must contain IDs 0–9 exactly once.")
    for row in services:
        input_kb = float(row["input_size_kb"])
        cycles_per_bit = float(row["cycles_per_bit"])
        if not 500.0 <= input_kb <= 1000.0:
            raise ValueError(
                f"Service {row['service_id']} input mean is outside 500–1000 KB: {input_kb}."
            )
        _assert_close(
            f"service[{row['service_id']}].mean_input_size",
            row["mean_input_size"],
            input_kb * 8000.0,
        )
        if not 400.0 <= cycles_per_bit <= 1000.0:
            raise ValueError(
                f"Service {row['service_id']} cycles_per_bit is outside 400–1000: {cycles_per_bit}."
            )

    baseline_bytes = baseline_path.read_bytes()
    baseline = json.loads(baseline_bytes.decode("utf-8"))
    if not isinstance(baseline, dict):
        raise ValueError(f"Formal pool integrity baseline root must be an object: {baseline_path}")
    expected_baseline = {
        "schema_version": POOL_BASELINE_SCHEMA,
        "integrity_baseline_type": "CURRENT_SNAPSHOT",
        "task_data_model": TASK_DATA_MODEL,
        "pool_manifest_sha256": manifest_sha256,
        "config_sha256": raw_config_sha256,
        "semantic_config_sha256": semantic_config_sha256,
        "target_generation_config_fingerprint": generation_fingerprint,
        "task_data_contract_sha256": contract_sha256,
        "service_catalog_path": catalog_relative,
        "service_catalog_sha256": catalog_sha256,
        "split": split,
        "seed_start": seed_start,
        "seed_end": seed_end,
        "episode_count": episode_count,
    }
    for field, expected in expected_baseline.items():
        if baseline.get(field) != expected:
            raise ValueError(
                f"Formal {split} integrity baseline {field} mismatch: "
                f"{baseline.get(field)!r} != {expected!r}."
            )
    if not isinstance(baseline.get("created_at_utc"), str) or not baseline["created_at_utc"].endswith("Z"):
        raise ValueError(f"Formal {split} baseline UTC timestamp is invalid.")
    _require_sha256(baseline.get("pool_manifest_sha256"), "pool_manifest_sha256")

    expected_seeds = list(range(seed_start, seed_end + 1))
    episode_hashes = manifest.get("episode_manifest_sha256")
    if not isinstance(episode_hashes, dict) or sorted(episode_hashes) != [str(x) for x in expected_seeds]:
        raise ValueError(f"Formal {split} Pool Manifest must enumerate every declared Episode.")
    baseline_episodes = baseline.get("episodes")
    if not isinstance(baseline_episodes, list) or len(baseline_episodes) != episode_count:
        raise ValueError(f"Formal {split} baseline does not contain {episode_count} Episodes.")
    if [record.get("episode_seed") for record in baseline_episodes if isinstance(record, dict)] != expected_seeds:
        raise ValueError(f"Formal {split} baseline Episode seeds are missing, duplicated, or unsorted.")

    required = set(REQUIRED_TRAINING_ARTIFACTS) | {"episode_manifest.json"}
    if split != "train":
        from scenario.realization import REALIZATION_ARTIFACT_FILENAMES

        required = set(REALIZATION_ARTIFACT_FILENAMES.values()) | {"episode_manifest.json"}
    expected_total_files = 0
    for index, seed in enumerate(expected_seeds):
        episode_dir = pool_path / f"episode_{seed}"
        episode_manifest_path = episode_dir / "episode_manifest.json"
        episode_manifest_hash = _stable_file_sha256(episode_manifest_path)
        if episode_manifest_hash != episode_hashes[str(seed)]:
            raise ValueError(f"Formal {split} Episode manifest hash mismatch for seed {seed}.")
        episode_manifest = json.loads(episode_manifest_path.read_text(encoding="utf-8"))
        expected_episode_fields = {
            "schema_version": TASK_DATA_EPISODE_SCHEMA,
            "task_data_model": TASK_DATA_MODEL,
            "split": split,
            "episode_seed": seed,
            "episode_slots": 300,
            "config_sha256": raw_config_sha256,
            "semantic_config_sha256": semantic_config_sha256,
            "target_generation_config_fingerprint": generation_fingerprint,
            "task_data_contract_sha256": contract_sha256,
            "service_catalog_path": catalog_relative,
            "service_catalog_sha256": catalog_sha256,
        }
        for field, expected in expected_episode_fields.items():
            if episode_manifest.get(field) != expected:
                raise ValueError(
                    f"Formal {split} Episode {seed} {field} mismatch: "
                    f"{episode_manifest.get(field)!r} != {expected!r}."
                )
        inventory = _episode_file_inventory(episode_dir)
        actual_paths = {relative for relative, _, _ in inventory}
        if not required.issubset(actual_paths):
            raise FileNotFoundError(
                f"Formal {split} Episode {seed} lacks required files: {sorted(required - actual_paths)}."
            )
        artifact_hashes = episode_manifest.get("artifact_sha256")
        artifact_paths = actual_paths - {"episode_manifest.json"}
        if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != artifact_paths:
            raise ValueError(f"Formal {split} Episode {seed} artifact inventory mismatch.")

        baseline_episode = baseline_episodes[index]
        if baseline_episode.get("relative_directory") != episode_dir.relative_to(PROJECT_ROOT).as_posix():
            raise ValueError(f"Formal {split} baseline Episode path mismatch for seed {seed}.")
        if baseline_episode.get("episode_manifest_sha256") != episode_manifest_hash:
            raise ValueError(f"Formal {split} baseline Episode manifest hash mismatch for seed {seed}.")
        baseline_files = baseline_episode.get("files")
        if not isinstance(baseline_files, list):
            raise ValueError(f"Formal {split} baseline file list is invalid for seed {seed}.")
        expected_file_map: dict[str, dict[str, Any]] = {}
        previous_path = ""
        for record in baseline_files:
            if not isinstance(record, dict):
                raise ValueError(f"Formal {split} baseline file record is invalid for seed {seed}.")
            relative_path = record.get("relative_path")
            if not isinstance(relative_path, str) or not relative_path or relative_path <= previous_path:
                raise ValueError(f"Formal {split} baseline file ordering is invalid for seed {seed}.")
            previous_path = relative_path
            if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
                raise ValueError(f"Formal {split} baseline contains unsafe path {relative_path!r}.")
            _require_sha256(record.get("sha256"), f"episode {seed} file {relative_path} sha256")
            if not isinstance(record.get("size_bytes"), int) or record["size_bytes"] < 0:
                raise ValueError(f"Formal {split} baseline file size is invalid: {relative_path}.")
            expected_file_map[relative_path] = record
        if set(expected_file_map) != actual_paths:
            raise ValueError(f"Formal {split} baseline file inventory mismatch for seed {seed}.")
        if len(baseline_files) != baseline_episode.get("file_count"):
            raise ValueError(f"Formal {split} baseline file count mismatch for seed {seed}.")
        total_bytes = 0
        for relative_path, file_path, size_bytes in inventory:
            registered = expected_file_map[relative_path]
            if size_bytes != registered["size_bytes"]:
                raise ValueError(f"Formal {split} baseline size mismatch for seed {seed}: {relative_path}.")
            digest = _stable_file_sha256(file_path, expected_size=size_bytes)
            if digest != registered["sha256"]:
                raise ValueError(f"Formal {split} data hash mismatch for seed {seed}: {relative_path}.")
            if relative_path != "episode_manifest.json" and digest != artifact_hashes[relative_path]:
                raise ValueError(f"Formal {split} Episode artifact hash mismatch: {seed}/{relative_path}.")
            total_bytes += size_bytes
        if total_bytes != baseline_episode.get("total_bytes"):
            raise ValueError(f"Formal {split} baseline byte total mismatch for seed {seed}.")
        expected_total_files += len(inventory)
    if expected_total_files != baseline.get("episode_file_count"):
        raise ValueError(f"Formal {split} baseline total Episode file count mismatch.")
    return pool_path


def _validate_training_pool(
    cfg: dict[str, Any], generation_fingerprint: str
) -> Path:
    return _validate_formal_task_pool(
        POOL_RELATIVE,
        split="train",
        seed_start=EPISODE_SEED_START,
        episode_count=EPISODE_COUNT,
        cfg=cfg,
        generation_fingerprint=generation_fingerprint,
    )


def _validate_validation_pool(
    cfg: dict[str, Any] | None = None,
    generation_fingerprint: str | None = None,
) -> Path:
    if cfg is None or generation_fingerprint is None:
        cfg, _, _, generation_fingerprint = _load_main_experiment_contract()
    return _validate_formal_task_pool(
        VALIDATION_POOL_RELATIVE,
        split="validation",
        seed_start=VALIDATION_SEED_START,
        episode_count=VALIDATION_EPISODE_COUNT,
        cfg=cfg,
        generation_fingerprint=generation_fingerprint,
    )


def _validate_test_pool(
    cfg: dict[str, Any], generation_fingerprint: str
) -> Path:
    return _validate_formal_task_pool(
        Path("environment_data/proposed_joint_test"),
        split="test",
        seed_start=40001,
        episode_count=100,
        cfg=cfg,
        generation_fingerprint=generation_fingerprint,
    )


def _build_macro_actor_mode_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or validate one formal PROPOSED Macro Actor learning-rate experiment."
    )
    parser.add_argument(
        "--mode",
        choices=("check", "train", "resume", "validate", "train-validate"),
        default="train",
        help=(
            "check static inputs, start a fresh training run, explicitly resume a checkpoint, "
            "validate a checkpoint, or train then validate."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        help=(
            "Checkpoint filename under this run's checkpoints directory, or a path inside "
            "that directory; required by resume and defaults to latest_checkpoint.pt in "
            "validate mode."
        ),
    )
    parser.add_argument(
        "--confirm-run-idle",
        action="store_true",
        help="Acknowledge that the target Run was checked and has no active training job.",
    )
    parser.add_argument(
        "--validation-episodes",
        type=int,
        default=VALIDATION_EPISODE_COUNT,
        help="Validation prefix length from seeds 30001–30030 (2–30; default 30).",
    )
    parser.add_argument(
        "--validation-name",
        help="Optional new output directory name under this run's validation directory.",
    )
    return parser


def _resolve_macro_validation_checkpoint(
    run_path: Path, checkpoint_reference: str | None
) -> Path:
    from ppo_simple.reporting import resolve_checkpoint_reference

    candidate = Path(checkpoint_reference or "latest_checkpoint.pt").expanduser()
    if not candidate.is_absolute():
        if candidate.parts and candidate.parts[0].casefold() == "checkpoints":
            candidate = run_path / candidate
        else:
            candidate = run_path / "checkpoints" / candidate
    checkpoint_path = resolve_checkpoint_reference(candidate).resolve(strict=True)
    checkpoint_root = (run_path / "checkpoints").resolve(strict=True)
    try:
        checkpoint_path.relative_to(checkpoint_root)
    except ValueError as exc:
        raise ValueError(
            "Checkpoints must belong to this learning-rate run."
        ) from exc
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Validation checkpoint is not a file: {checkpoint_path}")
    return checkpoint_path


def _macro_validation_output_path(
    run_path: Path,
    checkpoint_path: Path,
    episodes: int,
    validation_name: str | None,
) -> Path:
    name = (
        f"{checkpoint_path.stem}_{int(episodes)}ep"
        if validation_name is None
        else validation_name
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
        raise ValueError(
            "Validation name must be a single safe directory name of at most 64 characters."
        )
    output_path = run_path / VALIDATION_DIRECTORY_NAME / name
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Validation output: {output_path}"
        )
    return output_path


def _evaluate_macro_actor_checkpoint(
    *,
    checkpoint_path: Path,
    output_path: Path,
    episodes: int,
) -> int:
    from ppo_simple.evaluator import evaluate_pool

    print(f"Validation checkpoint: {checkpoint_path}", flush=True)
    print(
        f"Validation protocol: deterministic | seeds {VALIDATION_SEED_START}–"
        f"{VALIDATION_SEED_START + int(episodes) - 1} | Episodes={int(episodes)}",
        flush=True,
    )
    result = evaluate_pool(
        checkpoint_path=checkpoint_path,
        run_dir=output_path,
        config_path=PROJECT_ROOT / CONFIG_RELATIVE,
        episode_pool=PROJECT_ROOT / VALIDATION_POOL_RELATIVE,
        seed_start=VALIDATION_SEED_START,
        episodes=int(episodes),
        repeats=1,
        device="cpu",
        policy_action_mode="deterministic",
    )
    print(
        f"[VALIDATION] Completed | Episodes={result['episode_count']} | "
        f"Output={output_path.relative_to(PROJECT_ROOT).as_posix()}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the formal runner CLI without starting validation or training."""

    parser = argparse.ArgumentParser(
        description="Run the 300-Episode 500–1000 KB multi-node formal PPO main experiment."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resume",
        action="store_true",
        help="Strictly resume only a compatible run under the 500–1000 KB run directory.",
    )
    group.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate all formal task pools and manifests, then exit without training.",
    )
    return parser


def _validate_formal_run_mode(run_path: Path, *, resume: bool) -> None:
    """Reject unsafe fresh-run reuse or incomplete resume directories."""

    run_path = Path(run_path)
    if not resume:
        if run_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing formal main-experiment run: {run_path}"
            )
        return

    if not run_path.is_dir():
        raise FileNotFoundError(
            f"Cannot resume: formal run directory does not exist: {run_path}"
        )

    from ppo_simple.reporting import RUN_MANIFEST_FILENAME, checkpoint_path_for_resume
    from ppo_simple.trainer import CHECKPOINT_FILENAME

    manifest_path = run_path / RUN_MANIFEST_FILENAME
    checkpoint_path = checkpoint_path_for_resume(run_path, CHECKPOINT_FILENAME)
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Cannot resume: formal run requires run_manifest.json and "
            "checkpoints/latest_checkpoint.pt."
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg, raw_config_hash, semantic_hash, generation_fingerprint = _load_main_experiment_contract()
    pool_path = _validate_training_pool(cfg, generation_fingerprint)
    baseline_sha256 = _stable_file_sha256(pool_path / POOL_BASELINE_NAME)
    if args.preflight_only:
        _validate_validation_pool(cfg, generation_fingerprint)
        _validate_test_pool(cfg, generation_fingerprint)
        print(
            "[PREFLIGHT] PASS | TASK_DATA_MODEL=500_1000_KB | "
            "Train=300 | Validation=30 | Test=100 | No PPO training.",
            flush=True,
        )
        return 0

    run_path = PROJECT_ROOT / RUN_DIRECTORY_RELATIVE
    _validate_formal_run_mode(run_path, resume=bool(args.resume))
    os.chdir(PROJECT_ROOT)
    from ppo_simple.trainer import main as trainer_main

    print(f"Formal config: {CONFIG_RELATIVE.as_posix()}", flush=True)
    print(f"Formal config SHA-256: {raw_config_hash}", flush=True)
    print(f"Formal semantic config SHA-256: {semantic_hash}", flush=True)
    print(f"Episode generation fingerprint: {generation_fingerprint}", flush=True)
    print(f"Training Pool integrity baseline SHA-256: {baseline_sha256}", flush=True)
    print(f"Training Pool: {POOL_RELATIVE.as_posix()} (20001–20300)", flush=True)
    print(f"Run mode: {'resume' if args.resume else 'fresh'}", flush=True)
    print(
        "Macro Actor learning rate: "
        f"{cfg['algorithm']['ppo']['macro_actor_learning_rate']:.1e}",
        flush=True,
    )
    print(f"PPO initialization seed: {PPO_INITIALIZATION_SEED}", flush=True)
    print(f"Output run: {RUN_DIRECTORY_RELATIVE.as_posix()}", flush=True)
    trainer_args = [
        "--config",
        CONFIG_RELATIVE.as_posix(),
        "--epochs",
        str(EPISODE_COUNT),
        "--run-dir",
        RUN_DIRECTORY_RELATIVE.as_posix(),
        "--episode-pool",
        POOL_RELATIVE.as_posix(),
        "--validated-integrity-baseline-sha256",
        baseline_sha256,
        "--episode-seed-start",
        str(EPISODE_SEED_START),
        "--seed",
        str(PPO_INITIALIZATION_SEED),
    ]
    if args.resume:
        trainer_args.append("--resume")
    return trainer_main(trainer_args)


def main_for_macro_actor_learning_rate(
    rate: float, argv: list[str] | None = None
) -> int:
    """Dispatch one approved Macro Actor LR run through shared modes."""

    requested_rate = float(rate)
    matched_rate = next(
        (
            value
            for value in MACRO_ACTOR_LEARNING_RATE_RUNS
            if math.isclose(requested_rate, value, rel_tol=0.0, abs_tol=1.0e-15)
        ),
        None,
    )
    if matched_rate is None:
        raise ValueError(f"Unsupported formal Macro Actor learning rate: {rate!r}")

    parser = _build_macro_actor_mode_parser()
    args = parser.parse_args(argv)
    if not 2 <= int(args.validation_episodes) <= VALIDATION_EPISODE_COUNT:
        parser.error("--validation-episodes must be between 2 and 30.")
    if args.mode not in {"validate", "resume", "train-validate"}:
        if args.checkpoint is not None:
            parser.error("--checkpoint is only valid with --mode resume or validate.")
        if args.validation_name is not None:
            parser.error("--validation-name is only valid with a Validation mode.")
        if int(args.validation_episodes) != VALIDATION_EPISODE_COUNT:
            parser.error("--validation-episodes is only valid with a Validation mode.")
    if args.mode == "train-validate" and args.checkpoint is not None:
        parser.error("train-validate always evaluates the newly saved latest checkpoint.")
    if args.mode == "resume" and args.checkpoint is None:
        parser.error("--mode resume requires an explicit --checkpoint path.")
    if args.mode == "resume" and not args.confirm_run_idle:
        parser.error(
            "Before resume, verify the target Run has no active job and pass --confirm-run-idle."
        )
    if args.mode != "resume" and args.confirm_run_idle:
        parser.error("--confirm-run-idle is only valid with --mode resume.")
    if args.mode == "resume" and (
        args.validation_name is not None
        or int(args.validation_episodes) != VALIDATION_EPISODE_COUNT
    ):
        parser.error("Validation options are not valid with --mode resume.")

    run_directory = MACRO_ACTOR_LEARNING_RATE_RUNS[matched_rate]
    run_path = PROJECT_ROOT / run_directory
    (
        cfg,
        raw_config_hash,
        semantic_hash,
        generation_fingerprint,
    ) = _load_main_experiment_contract()
    needs_training = args.mode in {"check", "train", "resume", "train-validate"}
    needs_validation = args.mode in {"check", "validate", "train-validate"}
    pool_path = (
        _validate_training_pool(cfg, generation_fingerprint)
        if needs_training
        else None
    )
    validation_pool = _validate_validation_pool(cfg, generation_fingerprint) if needs_validation else None

    checkpoint_path: Path | None = None
    if args.mode in {"check", "train", "train-validate"}:
        _validate_formal_run_mode(run_path, resume=False)
    elif args.mode == "resume":
        _validate_formal_run_mode(run_path, resume=True)
        checkpoint_path = _resolve_macro_validation_checkpoint(run_path, args.checkpoint)
    elif not run_path.is_dir():
        raise FileNotFoundError(
            f"Cannot validate: learning-rate run directory does not exist: {run_path}"
        )

    validation_output: Path | None = None
    if args.mode == "validate":
        checkpoint_path = _resolve_macro_validation_checkpoint(run_path, args.checkpoint)
        validation_output = _macro_validation_output_path(
            run_path,
            checkpoint_path,
            int(args.validation_episodes),
            args.validation_name,
        )
    elif args.mode == "train-validate":
        planned_checkpoint = run_path / "checkpoints" / "latest_checkpoint.pt"
        validation_output = _macro_validation_output_path(
            run_path,
            planned_checkpoint,
            int(args.validation_episodes),
            args.validation_name,
        )

    if args.mode == "check":
        print(
            f"[CHECK] Passed | Config={CONFIG_RELATIVE.as_posix()} | "
            f"TrainingPool={POOL_RELATIVE.as_posix()} | "
            f"ValidationPool={VALIDATION_POOL_RELATIVE.as_posix()} | "
            f"Run={run_directory.as_posix()} | MacroActorLR={matched_rate:.1e}",
            flush=True,
        )
        return 0

    if args.mode == "validate":
        if checkpoint_path is None or validation_output is None:
            raise RuntimeError("Validation dispatch did not resolve its Checkpoint and output.")
        return _evaluate_macro_actor_checkpoint(
            checkpoint_path=checkpoint_path,
            output_path=validation_output,
            episodes=int(args.validation_episodes),
        )

    if args.mode == "resume" and checkpoint_path is None:
        raise RuntimeError("Resume dispatch did not resolve its Checkpoint.")
    if args.mode == "resume":
        print(f"Resume checkpoint: {checkpoint_path}", flush=True)

    if pool_path is None:
        raise RuntimeError("Training dispatch did not validate the formal Training Pool.")
    baseline_sha256 = _stable_file_sha256(pool_path / POOL_BASELINE_NAME)
    os.chdir(PROJECT_ROOT)
    from ppo_simple.trainer import main as trainer_main

    print(f"Formal config: {CONFIG_RELATIVE.as_posix()}", flush=True)
    print(f"Formal config SHA-256: {raw_config_hash}", flush=True)
    print(f"Formal semantic config SHA-256: {semantic_hash}", flush=True)
    print(f"Episode generation fingerprint: {generation_fingerprint}", flush=True)
    print(f"Training Pool integrity baseline SHA-256: {baseline_sha256}", flush=True)
    print(f"Training Pool: {POOL_RELATIVE.as_posix()} (20001–20300)", flush=True)
    print(f"Run mode: {args.mode} | Macro Actor LR override: {matched_rate:.1e}", flush=True)
    print(f"PPO initialization seed: {PPO_INITIALIZATION_SEED}", flush=True)
    print(f"Output run: {run_directory.as_posix()}", flush=True)
    trainer_args = [
        "--config",
        CONFIG_RELATIVE.as_posix(),
        "--epochs",
        str(EPISODE_COUNT),
        "--run-dir",
        run_directory.as_posix(),
        "--episode-pool",
        POOL_RELATIVE.as_posix(),
        "--validated-integrity-baseline-sha256",
        baseline_sha256,
        "--episode-seed-start",
        str(EPISODE_SEED_START),
        "--macro-actor-learning-rate",
        f"{matched_rate:.1e}",
        "--seed",
        str(PPO_INITIALIZATION_SEED),
    ]
    if args.mode == "resume":
        trainer_args.extend(["--resume", "--resume-checkpoint", str(checkpoint_path)])
    training_status = int(trainer_main(trainer_args))
    if args.mode in {"train", "resume"} or training_status != 0:
        if args.mode == "train-validate" and training_status != 0:
            print(
                f"[VALIDATION] Skipped | Training exited with status={training_status}",
                flush=True,
            )
        return training_status

    if validation_pool is None or validation_output is None:
        raise RuntimeError("train-validate did not complete Validation preflight.")
    checkpoint_path = _resolve_macro_validation_checkpoint(run_path, None)
    return _evaluate_macro_actor_checkpoint(
        checkpoint_path=checkpoint_path,
        output_path=validation_output,
        episodes=int(args.validation_episodes),
    )


if __name__ == "__main__":
    raise SystemExit(main())
