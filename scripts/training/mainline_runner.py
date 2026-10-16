"""Single formal runner for the multi-node v4 PROPOSED main experiment."""

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
RUN_DIRECTORY_RELATIVE = Path("runs/stageB3_10j_multi_node_v4_mainline")
POOL_MANIFEST_NAME = "stage6_training_pool_manifest.json"
POOL_BASELINE_NAME = "stage6_training_pool_integrity_baseline.json"
POOL_BASELINE_SCHEMA = "stage6_training_pool_integrity_baseline_v1"
EPISODE_COUNT = 300
EPISODE_SEED_START = 20001
EPISODE_SEED_END = EPISODE_SEED_START + EPISODE_COUNT - 1
PPO_INITIALIZATION_SEED = 27005
VALIDATION_SEED_START = 30001
VALIDATION_EPISODE_COUNT = 30
VALIDATION_DIRECTORY_NAME = "validation"
MACRO_ACTOR_LEARNING_RATE_RUNS = {
    1.0e-4: Path("runs/stageB3_14h_fresh_macro_actor_lr_1e_minus_4"),
    5.0e-4: Path("runs/stageB3_14h_fresh_macro_actor_lr_5e_minus_4"),
    1.0e-3: Path("runs/stageB3_14h_fresh_macro_actor_lr_1e_minus_3"),
    5.0e-3: Path("runs/stageB3_14h_fresh_macro_actor_lr_5e_minus_3"),
}
HISTORICAL_STAGE6_CONFIG_SHA256 = (
    "684bdfa4dc200db6227b2b00a7718797a98bf658e54d2d4dbef6f7d1f5babf77"
)
HISTORICAL_STAGE6_POOL_MANIFEST_SHA256 = (
    "c9c401dfe6c36b11faecfa080b86cb878d325c2799ee7e54435ad85365992d93"
)
LEGACY_OVERLAY_PROVENANCE_FIELDS = (
    "source_git_sha",
    "source_git_dirty",
    "generation_config_fields",
    "source_generated_artifact_sha256",
    "generated_artifacts",
    "generated_artifact_sha256",
)
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
            2000.0,
        ),
    }
    for field, (actual, expected) in expected_values.items():
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
        raise ValueError(f"Stage6 training pool {field} is missing or invalid.")
    return value


def _overlay_missing_provenance_fields(episode_manifest: dict[str, Any]) -> list[str]:
    return [field for field in LEGACY_OVERLAY_PROVENANCE_FIELDS if field not in episode_manifest]


def _validate_overlay_snapshot_metadata(
    episode_manifest: dict[str, Any],
    *,
    seed: int,
    pool_manifest: dict[str, Any],
    generation_fingerprint: str,
) -> None:
    """Check known overlay identities while keeping unavailable history explicit."""

    if int(episode_manifest.get("source_seed", -1)) != seed:
        raise ValueError(f"Stage6 workload overlay source seed mismatch for {seed}.")
    if episode_manifest.get("generation_config_fingerprint") != generation_fingerprint:
        raise ValueError(f"Stage6 generation fingerprint mismatch for seed {seed}.")
    if episode_manifest.get("config_sha256") != pool_manifest["config_sha256"]:
        raise ValueError(f"Stage6 workload overlay config provenance mismatch for {seed}.")
    if episode_manifest.get("semantic_config_sha256") != pool_manifest["semantic_config_sha256"]:
        raise ValueError(f"Stage6 workload overlay semantic provenance mismatch for {seed}.")
    for episode_field, pool_field in (
        ("source_config_sha256", "source_config_sha256"),
        ("source_semantic_config_sha256", "source_semantic_config_sha256"),
        ("stage6_service_catalog_sha256", "stage6_service_catalog_sha256"),
    ):
        if episode_manifest.get(episode_field) != pool_manifest.get(pool_field):
            raise ValueError(
                f"Stage6 workload overlay {episode_field} mismatch for seed {seed}."
            )
    if episode_manifest.get("source_training_split") != "train":
        raise ValueError(f"Stage6 workload overlay source split mismatch for {seed}.")
    if episode_manifest.get("source_stage6_service_catalog_path") != pool_manifest.get(
        "source_stage6_service_catalog_path"
    ):
        raise ValueError(f"Stage6 workload overlay source catalog path mismatch for {seed}.")
    if episode_manifest.get("transformed_fields") != pool_manifest.get("transformed_fields"):
        raise ValueError(f"Stage6 workload overlay transform fields mismatch for {seed}.")

    if "source_git_sha" in episode_manifest:
        source_git_sha = episode_manifest.get("source_git_sha")
        if not isinstance(source_git_sha, str) or re.fullmatch(r"[0-9a-f]{40}", source_git_sha) is None:
            raise ValueError(f"Stage6 workload overlay source Git SHA is invalid for {seed}.")
    if "source_git_dirty" in episode_manifest and not isinstance(
        episode_manifest.get("source_git_dirty"), bool
    ):
        raise ValueError(f"Stage6 workload overlay Git dirty flag is invalid for {seed}.")


def _episode_file_inventory(episode_dir: Path) -> list[tuple[str, Path, int]]:
    inventory: list[tuple[str, Path, int]] = []
    for path in sorted(episode_dir.rglob("*"), key=lambda item: item.relative_to(episode_dir).as_posix()):
        if path.is_symlink():
            raise ValueError(f"Training Episode data must not contain symlinks: {path}")
        if path.is_file():
            relative_path = path.relative_to(episode_dir).as_posix()
            inventory.append((relative_path, path, path.stat().st_size))
        elif not path.is_dir():
            raise ValueError(f"Training Episode contains a non-regular filesystem entry: {path}")
    return inventory


def _validate_training_pool_root_inventory(pool_path: Path) -> None:
    expected_episode_dirs = {
        f"episode_{seed}" for seed in range(EPISODE_SEED_START, EPISODE_SEED_END + 1)
    }
    expected_files = {POOL_MANIFEST_NAME, "service_catalog_500_2000.json"}
    baseline_path = pool_path / POOL_BASELINE_NAME
    if baseline_path.exists():
        expected_files.add(POOL_BASELINE_NAME)
    actual_entries = {entry.name: entry for entry in pool_path.iterdir()}
    expected_names = expected_episode_dirs | expected_files
    if set(actual_entries) != expected_names:
        missing = sorted(expected_names - set(actual_entries))
        unexpected = sorted(set(actual_entries) - expected_names)
        raise ValueError(
            f"Stage6 Training Pool root inventory mismatch; missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}."
        )
    for name, entry in actual_entries.items():
        if entry.is_symlink():
            raise ValueError(f"Stage6 Training Pool entries must not be symlinks: {entry}")
        if name in expected_episode_dirs and not entry.is_dir():
            raise ValueError(f"Stage6 Training Pool Episode entry is not a directory: {entry}")
        if name in expected_files and not entry.is_file():
            raise ValueError(f"Stage6 Training Pool file entry is not a file: {entry}")


def _baseline_git_identity() -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if re.fullmatch(r"[0-9a-f]{40}", head) is None:
            raise ValueError(f"git rev-parse returned an invalid HEAD: {head!r}")
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"head": head, "worktree_dirty": bool(status), "unavailable_reason": None}
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        return {
            "head": None,
            "worktree_dirty": None,
            "unavailable_reason": f"Could not read Git identity at baseline time: {exc}",
        }


def _validate_integrity_baseline_object(
    baseline: Any,
    pool_manifest: dict[str, Any],
    *,
    pool_manifest_sha256: str,
    generation_fingerprint: str,
) -> dict[str, Any]:
    if not isinstance(baseline, dict):
        raise ValueError("Stage6 Training Pool integrity baseline root must be an object.")
    expected = {
        "schema_version": POOL_BASELINE_SCHEMA,
        "integrity_baseline_type": "CURRENT_SNAPSHOT",
        "historical_generation_attestation": "UNAVAILABLE",
        "training_pool_path": POOL_RELATIVE.as_posix(),
        "split": "train",
        "seed_start": EPISODE_SEED_START,
        "seed_end": EPISODE_SEED_END,
        "episode_count": EPISODE_COUNT,
        "pool_manifest_sha256": pool_manifest_sha256,
        "pool_config_sha256": pool_manifest.get("config_sha256"),
        "pool_semantic_config_sha256": pool_manifest.get("semantic_config_sha256"),
        "generation_config_fingerprint": generation_fingerprint,
    }
    for field, value in expected.items():
        if baseline.get(field) != value:
            raise ValueError(
                f"Stage6 integrity baseline {field} mismatch: "
                f"{baseline.get(field)!r} != {value!r}."
            )
    if baseline.get("historical_source_status") != "HISTORICAL_PROVENANCE_INCOMPLETE":
        raise ValueError("Stage6 integrity baseline historical source status is invalid.")
    if baseline.get("stage6_service_catalog_path") != pool_manifest.get("stage6_service_catalog_path"):
        raise ValueError("Stage6 integrity baseline service-catalog path mismatch.")
    if baseline.get("stage6_service_catalog_sha256") != pool_manifest.get(
        "stage6_service_catalog_sha256"
    ):
        raise ValueError("Stage6 integrity baseline service-catalog hash mismatch.")
    if baseline.get("historical_source_evidence", {}).get("status") != (
        "HISTORICAL_PROVENANCE_INCOMPLETE"
    ):
        raise ValueError("Stage6 integrity baseline historical-evidence record is invalid.")
    _require_sha256(
        baseline.get("pool_manifest_sha256"), "baseline pool_manifest_sha256"
    )
    timestamp = baseline.get("created_at_utc")
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        raise ValueError("Stage6 integrity baseline UTC creation time is invalid.")
    episodes = baseline.get("episodes")
    expected_seeds = list(range(EPISODE_SEED_START, EPISODE_SEED_END + 1))
    if not isinstance(episodes, list) or len(episodes) != EPISODE_COUNT:
        raise ValueError("Stage6 integrity baseline does not contain 300 Episode records.")
    observed_seeds = [record.get("episode_seed") for record in episodes if isinstance(record, dict)]
    if observed_seeds != expected_seeds:
        raise ValueError("Stage6 integrity baseline Episode seeds are missing, duplicated, or unsorted.")
    return baseline


def _validate_integrity_baseline_header(
    pool_path: Path,
    pool_manifest: dict[str, Any],
    *,
    pool_manifest_sha256: str,
    generation_fingerprint: str,
) -> dict[str, Any]:
    baseline_path = pool_path / POOL_BASELINE_NAME
    if not baseline_path.is_file() or baseline_path.is_symlink():
        raise FileNotFoundError(
            f"Stage6 Training Pool current-snapshot integrity baseline is missing: {baseline_path}. "
            "Create it explicitly after the preflight integrity checks; training never creates it."
        )
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Stage6 Training Pool integrity baseline: {baseline_path}") from exc
    return _validate_integrity_baseline_object(
        baseline,
        pool_manifest,
        pool_manifest_sha256=pool_manifest_sha256,
        generation_fingerprint=generation_fingerprint,
    )


def _validate_baseline_episode_files(
    pool_path: Path,
    pool_manifest: dict[str, Any],
    baseline_episode: dict[str, Any],
    *,
    seed: int,
) -> int:
    episode_dir = pool_path / f"episode_{seed}"
    episode_manifest_path = episode_dir / "episode_manifest.json"
    current_manifest_hash = _sha256(episode_manifest_path)
    registered_manifest_hash = pool_manifest["episode_manifest_sha256"][str(seed)]
    if current_manifest_hash != registered_manifest_hash:
        raise ValueError(f"Stage6 Episode Manifest no longer matches Pool Manifest for seed {seed}.")
    if baseline_episode.get("relative_directory") != episode_dir.relative_to(PROJECT_ROOT).as_posix():
        raise ValueError(f"Stage6 integrity baseline Episode path mismatch for seed {seed}.")
    if baseline_episode.get("episode_manifest_sha256") != current_manifest_hash:
        raise ValueError(f"Stage6 integrity baseline Episode Manifest hash mismatch for seed {seed}.")
    if baseline_episode.get("pool_manifest_registered_episode_manifest_sha256") != registered_manifest_hash:
        raise ValueError(f"Stage6 integrity baseline Pool Manifest registration mismatch for seed {seed}.")

    episode_manifest = json.loads(episode_manifest_path.read_text(encoding="utf-8"))
    if baseline_episode.get("manifest_schema") != episode_manifest.get("schema_version"):
        raise ValueError(f"Stage6 integrity baseline Episode schema mismatch for seed {seed}.")
    baseline_files = baseline_episode.get("files")
    if not isinstance(baseline_files, list):
        raise ValueError(f"Stage6 integrity baseline file list is invalid for seed {seed}.")
    expected_file_map: dict[str, dict[str, Any]] = {}
    previous_path = ""
    for record in baseline_files:
        if not isinstance(record, dict):
            raise ValueError(f"Stage6 integrity baseline file record is invalid for seed {seed}.")
        relative_path = record.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path or relative_path <= previous_path:
            raise ValueError(f"Stage6 integrity baseline file order/path is invalid for seed {seed}.")
        previous_path = relative_path
        if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            raise ValueError(f"Stage6 integrity baseline contains an unsafe file path for seed {seed}.")
        if relative_path in expected_file_map:
            raise ValueError(f"Stage6 integrity baseline repeats a file for seed {seed}: {relative_path}.")
        _require_sha256(record.get("sha256"), f"baseline episode {seed} file {relative_path} sha256")
        size = record.get("size_bytes")
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"Stage6 integrity baseline file size is invalid for seed {seed}: {relative_path}.")
        expected_file_map[relative_path] = record
    current_inventory = _episode_file_inventory(episode_dir)
    current_paths = {relative_path for relative_path, _, _ in current_inventory}
    if current_paths != set(expected_file_map):
        raise ValueError(
            f"Stage6 Episode file set changed for seed {seed}; "
            f"missing={sorted(set(expected_file_map) - current_paths)[:8]}, "
            f"unexpected={sorted(current_paths - set(expected_file_map))[:8]}."
        )
    for relative_path, path, size in current_inventory:
        record = expected_file_map[relative_path]
        if size != record["size_bytes"] or _stable_file_sha256(
            path, expected_size=size
        ) != record["sha256"]:
            raise ValueError(f"Stage6 Training Episode file hash/size mismatch for seed {seed}: {relative_path}.")
    required = set(REQUIRED_TRAINING_ARTIFACTS) | {"episode_manifest.json"}
    if not required.issubset(current_paths):
        raise ValueError(f"Stage6 integrity baseline lacks required data files for seed {seed}.")
    if sum(record["size_bytes"] for record in baseline_files) != baseline_episode.get("total_bytes"):
        raise ValueError(f"Stage6 integrity baseline total size mismatch for seed {seed}.")
    if len(baseline_files) != baseline_episode.get("file_count"):
        raise ValueError(f"Stage6 integrity baseline file count mismatch for seed {seed}.")
    return len(current_inventory)


def _validate_overlay_episode_provenance(
    episode_dir: Path,
    episode_manifest: dict[str, Any],
    *,
    seed: int,
    pool_manifest: dict[str, Any],
    generation_fingerprint: str,
    generation_fields: dict[str, Any],
) -> None:
    """Validate a workload overlay without treating it as a raw generated Episode."""

    _validate_overlay_snapshot_metadata(
        episode_manifest,
        seed=seed,
        pool_manifest=pool_manifest,
        generation_fingerprint=generation_fingerprint,
    )

    if int(episode_manifest.get("source_seed", -1)) != seed:
        raise ValueError(f"Stage6 workload overlay source seed mismatch for {seed}.")
    if episode_manifest.get("generation_config_fingerprint") != generation_fingerprint:
        raise ValueError(f"Stage6 generation fingerprint mismatch for seed {seed}.")
    if episode_manifest.get("config_sha256") != pool_manifest["config_sha256"]:
        raise ValueError(f"Stage6 workload overlay config provenance mismatch for {seed}.")
    if episode_manifest.get("semantic_config_sha256") != pool_manifest["semantic_config_sha256"]:
        raise ValueError(f"Stage6 workload overlay semantic provenance mismatch for {seed}.")
    for episode_field, pool_field in (
        ("source_config_sha256", "source_config_sha256"),
        ("source_semantic_config_sha256", "source_semantic_config_sha256"),
        ("stage6_service_catalog_sha256", "stage6_service_catalog_sha256"),
    ):
        expected = pool_manifest.get(pool_field)
        if episode_manifest.get(episode_field) != expected:
            raise ValueError(
                f"Stage6 workload overlay {episode_field} mismatch for seed {seed}."
            )
    if episode_manifest.get("source_training_split") != "train":
        raise ValueError(f"Stage6 workload overlay source split mismatch for {seed}.")
    if episode_manifest.get("source_stage6_service_catalog_path") != pool_manifest.get(
        "source_stage6_service_catalog_path"
    ):
        raise ValueError(f"Stage6 workload overlay source catalog path mismatch for {seed}.")
    if episode_manifest.get("transformed_fields") != pool_manifest.get("transformed_fields"):
        raise ValueError(f"Stage6 workload overlay transform fields mismatch for {seed}.")
    if episode_manifest.get("source_git_dirty") is not None and not isinstance(
        episode_manifest.get("source_git_dirty"), bool
    ):
        raise ValueError(f"Stage6 workload overlay Git dirty flag is invalid for {seed}.")

    missing_provenance = [
        field
        for field in (
            "source_git_sha",
            "source_git_dirty",
            "generation_config_fields",
            "source_generated_artifact_sha256",
            "generated_artifacts",
            "generated_artifact_sha256",
        )
        if field not in episode_manifest
    ]
    if missing_provenance:
        raise ValueError(
            f"Stage6 workload overlay {seed} lacks required historical provenance "
            f"{missing_provenance}; raw source and transformed artifact integrity "
            "cannot be verified without changing the preserved manifests."
        )
    if episode_manifest.get("generation_config_fields") != generation_fields:
        raise ValueError(f"Stage6 workload overlay generation fields mismatch for {seed}.")

    source_git_sha = episode_manifest.get("source_git_sha")
    if not isinstance(source_git_sha, str) or re.fullmatch(r"[0-9a-f]{40}", source_git_sha) is None:
        raise ValueError(f"Stage6 workload overlay source Git SHA is invalid for {seed}.")
    if not isinstance(episode_manifest.get("source_git_dirty"), bool):
        raise ValueError(f"Stage6 workload overlay source Git dirty flag is invalid for {seed}.")

    generated_artifacts = episode_manifest.get("generated_artifacts")
    artifact_hashes = episode_manifest.get("generated_artifact_sha256")
    source_hashes = episode_manifest.get("source_generated_artifact_sha256")
    if not isinstance(generated_artifacts, list) or not isinstance(artifact_hashes, dict):
        raise ValueError(f"Stage6 workload overlay artifact hash table is invalid for {seed}.")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError(f"Stage6 workload overlay source artifact hashes are missing for {seed}.")
    actual_paths = sorted(
        path for path in episode_dir.iterdir()
        if path.is_file() and path.name != "episode_manifest.json"
    )
    actual_names = sorted(path.name for path in actual_paths)
    if generated_artifacts != actual_names or set(artifact_hashes) != set(actual_names):
        raise ValueError(f"Stage6 workload overlay artifact list mismatch for {seed}.")
    for path in actual_paths:
        if _sha256(path) != artifact_hashes.get(path.name):
            raise ValueError(
                f"Stage6 workload overlay artifact hash mismatch for seed {seed}: {path.name}."
            )
    for name, digest in source_hashes.items():
        _require_sha256(digest, f"episode {seed} source artifact hash {name}")


def _validate_training_pool(
    cfg: dict[str, Any],
    generation_fingerprint: str,
    *,
    allow_legacy_overlay_snapshot_preflight: bool = False,
) -> Path:
    from common import project_config
    from generation import generate_episode_realization

    pool_path = PROJECT_ROOT / POOL_RELATIVE
    pool_manifest_path = pool_path / POOL_MANIFEST_NAME
    if not pool_manifest_path.is_file():
        raise FileNotFoundError(
            f"Stage6 training pool manifest is missing: {pool_manifest_path}. "
            "Prepare it from the existing proposed_joint_training pool using the Stage6 workload overlay."
        )
    _validate_training_pool_root_inventory(pool_path)
    pool_manifest_sha256 = _stable_file_sha256(pool_manifest_path)
    if pool_manifest_sha256 != HISTORICAL_STAGE6_POOL_MANIFEST_SHA256:
        raise ValueError(
            "Stage6 Training Pool Manifest differs from the previously inspected immutable pool identity: "
            f"{pool_manifest_sha256}."
        )
    manifest = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "stage6_training_pool_manifest_v1",
        "split": "train",
        "seed_start": EPISODE_SEED_START,
        "seed_end": EPISODE_SEED_END,
        "episode_count": EPISODE_COUNT,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"Stage6 training pool {field} mismatch: {manifest.get(field)!r} != {value!r}"
            )

    historical_config_hash = _require_sha256(manifest.get("config_sha256"), "config_sha256")
    if historical_config_hash != HISTORICAL_STAGE6_CONFIG_SHA256:
        raise ValueError(
            "Stage6 training pool historical config identity does not match the "
            "declared source config."
        )
    historical_semantic_hash = _require_sha256(
        manifest.get("semantic_config_sha256"), "semantic_config_sha256"
    )
    stored_generation_fingerprint = _require_sha256(
        manifest.get("generation_config_fingerprint"), "generation_config_fingerprint"
    )
    if stored_generation_fingerprint != generation_fingerprint:
        raise ValueError(
            "Stage6 Episode generation contract mismatch: "
            f"stored={stored_generation_fingerprint!r} != current={generation_fingerprint!r}"
        )
    combined = manifest.get("generation_and_semantic_fingerprint")
    if combined != f"{stored_generation_fingerprint}:{historical_semantic_hash}":
        raise ValueError(
            "Stage6 pool generation_and_semantic_fingerprint is inconsistent with "
            "its separately recorded historical identities."
        )
    for field in (
        "source_config_sha256",
        "source_semantic_config_sha256",
        "source_data_split_manifest_sha256",
        "source_service_catalog_sha256",
        "stage6_service_catalog_sha256",
    ):
        _require_sha256(manifest.get(field), field)
    if manifest.get("config_path") != CONFIG_RELATIVE.as_posix():
        raise ValueError("Stage6 pool historical config path mismatch.")
    if manifest.get("source_training_pool") != POOL_RELATIVE.as_posix():
        raise ValueError("Stage6 pool source path identity mismatch.")

    split_path = PROJECT_ROOT / "environment_data/data_split_manifest.json"
    if _sha256(split_path) != manifest["source_data_split_manifest_sha256"]:
        raise ValueError("Stage6 Training split provenance hash mismatch.")
    service_catalog_path = project_config.resolve_path(
        PROJECT_ROOT, project_config.get_config_value(cfg, "paths.service_catalog")
    )
    if manifest.get("stage6_service_catalog_path") != project_config.get_config_value(
        cfg, "paths.service_catalog"
    ):
        raise ValueError("Stage6 service catalog path provenance mismatch.")
    if _sha256(service_catalog_path) != manifest["stage6_service_catalog_sha256"]:
        raise ValueError("Stage6 service catalog provenance hash mismatch.")
    source_catalog_path = PROJECT_ROOT / "data/generated/service_catalog.json"
    if not source_catalog_path.is_file() or _sha256(source_catalog_path) != manifest[
        "source_service_catalog_sha256"
    ]:
        raise ValueError("Stage6 source service catalog provenance hash mismatch.")
    source_stage6_catalog_relative = manifest.get("source_stage6_service_catalog_path")
    if not isinstance(source_stage6_catalog_relative, str) or not source_stage6_catalog_relative:
        raise ValueError("Stage6 source service catalog path is missing.")
    source_stage6_catalog_path = project_config.resolve_path(
        PROJECT_ROOT, source_stage6_catalog_relative
    )
    if not source_stage6_catalog_path.is_file() or _sha256(
        source_stage6_catalog_path
    ) != manifest["stage6_service_catalog_sha256"]:
        raise ValueError("Stage6 derived service catalog provenance hash mismatch.")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_split = split.get("train", {})
    validation_split = split.get("validation", {})
    test_split = split.get("test", {})
    if not (
        int(train_split.get("seed_start", -1)) <= EPISODE_SEED_START
        and int(train_split.get("seed_end", -1)) >= EPISODE_SEED_END
        and EPISODE_SEED_END < int(validation_split.get("seed_start", -1))
        and EPISODE_SEED_END < int(test_split.get("seed_start", -1))
    ):
        raise ValueError("Stage6 episode seeds are not isolated inside the declared Training split.")

    manifest_hashes = manifest.get("episode_manifest_sha256")
    expected_seeds = [str(seed) for seed in range(EPISODE_SEED_START, EPISODE_SEED_END + 1)]
    if not isinstance(manifest_hashes, dict) or sorted(manifest_hashes) != expected_seeds:
        raise ValueError("Stage6 pool manifest does not enumerate all 300 Training Episodes.")
    baseline_path = pool_path / POOL_BASELINE_NAME
    if allow_legacy_overlay_snapshot_preflight and baseline_path.exists():
        raise FileExistsError(
            "Snapshot-baseline preflight is only used before first creation; the baseline already exists."
        )
    baseline = (
        _validate_integrity_baseline_header(
            pool_path,
            manifest,
            pool_manifest_sha256=pool_manifest_sha256,
            generation_fingerprint=generation_fingerprint,
        )
        if baseline_path.exists()
        else None
    )
    baseline_file_count = 0
    for seed in range(EPISODE_SEED_START, EPISODE_SEED_END + 1):
        seed_text = str(seed)
        episode_dir = pool_path / f"episode_{seed}"
        episode_manifest_path = episode_dir / "episode_manifest.json"
        if not episode_manifest_path.is_file():
            raise FileNotFoundError(f"Stage6 Training Episode manifest missing: {episode_manifest_path}")
        if _sha256(episode_manifest_path) != manifest_hashes[seed_text]:
            raise ValueError(f"Stage6 Training Episode manifest hash mismatch for seed {seed}.")
        episode_manifest = json.loads(episode_manifest_path.read_text(encoding="utf-8"))
        if int(episode_manifest.get("episode_seed", -1)) != seed:
            raise ValueError(f"Episode manifest seed mismatch for {seed}.")
        schema = episode_manifest.get("schema_version")
        if schema == "episode_manifest_v1":
            generate_episode_realization.validate_episode_provenance(
                episode_dir, cfg, seed
            )
            if episode_manifest.get("generation_config_fields") != (
                generate_episode_realization._generation_config_payload(cfg)
            ):
                raise ValueError(f"Stage6 generation fields mismatch for seed {seed}.")
        elif schema == "stage6_workload_overlay_v1":
            missing_overlay_fields = _overlay_missing_provenance_fields(episode_manifest)
            if missing_overlay_fields:
                present_overlay_fields = sorted(
                    set(LEGACY_OVERLAY_PROVENANCE_FIELDS) - set(missing_overlay_fields)
                )
                if present_overlay_fields:
                    raise ValueError(
                        f"Stage6 workload overlay {seed} has partial historical provenance; "
                        f"missing={missing_overlay_fields}, present={present_overlay_fields}."
                    )
                _validate_overlay_snapshot_metadata(
                    episode_manifest,
                    seed=seed,
                    pool_manifest=manifest,
                    generation_fingerprint=generation_fingerprint,
                )
                if baseline is None and not allow_legacy_overlay_snapshot_preflight:
                    raise ValueError(
                        f"Stage6 workload overlay {seed} lacks historical provenance; "
                        f"the explicit current-snapshot integrity baseline is required. "
                        f"Missing fields: {missing_overlay_fields}."
                    )
            else:
                _validate_overlay_episode_provenance(
                    episode_dir,
                    episode_manifest,
                    seed=seed,
                    pool_manifest=manifest,
                    generation_fingerprint=generation_fingerprint,
                    generation_fields=generate_episode_realization._generation_config_payload(cfg),
                )
        else:
            raise ValueError(
                f"Unsupported Stage6 Episode manifest schema for seed {seed}: {schema!r}."
            )
        missing = [name for name in REQUIRED_TRAINING_ARTIFACTS if not (episode_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Stage6 Training Episode {seed} is incomplete; missing {missing}."
            )
        manifest_required_artifacts = episode_manifest.get("required_artifacts")
        if manifest_required_artifacts is not None:
            if not isinstance(manifest_required_artifacts, list) or any(
                not isinstance(name, str) for name in manifest_required_artifacts
            ):
                raise ValueError(f"Stage6 Episode {seed} required_artifacts is malformed.")
            missing_declared = [
                name for name in manifest_required_artifacts if not (episode_dir / name).is_file()
            ]
            if missing_declared:
                raise FileNotFoundError(
                    f"Stage6 Episode {seed} lacks files declared by its preserved manifest: {missing_declared}."
                )
        if baseline is not None:
            baseline_record = baseline["episodes"][seed - EPISODE_SEED_START]
            baseline_file_count += _validate_baseline_episode_files(
                pool_path, manifest, baseline_record, seed=seed
            )
    if baseline is not None and baseline_file_count != baseline.get("episode_file_count"):
        raise ValueError("Stage6 integrity baseline total Episode file count mismatch.")
    return pool_path


def _validate_validation_pool() -> Path:
    """Check the fixed Validation split and its realization files without running Episodes."""

    from scenario.realization import REALIZATION_ARTIFACT_FILENAMES

    pool_path = PROJECT_ROOT / VALIDATION_POOL_RELATIVE
    split_path = PROJECT_ROOT / "environment_data/data_split_manifest.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    validation_split = split.get("validation", {})
    expected_end = VALIDATION_SEED_START + VALIDATION_EPISODE_COUNT - 1
    if (
        int(validation_split.get("seed_start", -1)) != VALIDATION_SEED_START
        or int(validation_split.get("seed_end", -1)) != expected_end
    ):
        raise ValueError("Validation seed range does not match the formal 30001–30030 contract.")

    required_artifacts = tuple(REALIZATION_ARTIFACT_FILENAMES.values())
    for seed in range(VALIDATION_SEED_START, expected_end + 1):
        episode_dir = pool_path / f"episode_{seed}"
        if not episode_dir.is_dir():
            raise FileNotFoundError(f"Validation Episode directory missing: {episode_dir}")
        manifest_path = episode_dir / "episode_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Validation Episode manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("episode_seed", -1)) != seed:
            raise ValueError(f"Validation Episode manifest seed mismatch for {seed}.")
        if int(manifest.get("simulation", {}).get("episode_slots", -1)) != 300:
            raise ValueError(f"Validation Episode {seed} does not declare 300 slots.")
        missing = [name for name in required_artifacts if not (episode_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Validation Episode {seed} is missing required artifacts: {missing}."
            )
    return pool_path


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


def _create_training_pool_integrity_baseline() -> Path:
    """Explicit one-time baseline creation; the formal runner only reads it."""

    pool_path = PROJECT_ROOT / POOL_RELATIVE
    baseline_path = pool_path / POOL_BASELINE_NAME
    if baseline_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing integrity baseline: {baseline_path}")

    cfg, _, _, generation_fingerprint = _load_main_experiment_contract()
    _validate_training_pool(
        cfg,
        generation_fingerprint,
        allow_legacy_overlay_snapshot_preflight=True,
    )
    pool_manifest_path = pool_path / POOL_MANIFEST_NAME
    pool_manifest_bytes = pool_manifest_path.read_bytes()
    pool_manifest = json.loads(pool_manifest_bytes.decode("utf-8"))
    if hashlib.sha256(pool_manifest_bytes).hexdigest() != HISTORICAL_STAGE6_POOL_MANIFEST_SHA256:
        raise ValueError("Stage6 Training Pool Manifest changed during baseline preflight.")

    episode_records: list[dict[str, Any]] = []
    historical_missing: set[str] = set()
    total_file_count = 0
    for seed in range(EPISODE_SEED_START, EPISODE_SEED_END + 1):
        episode_dir = pool_path / f"episode_{seed}"
        manifest_path = episode_dir / "episode_manifest.json"
        episode_manifest_hash = _stable_file_sha256(manifest_path)
        registered_hash = pool_manifest["episode_manifest_sha256"][str(seed)]
        if episode_manifest_hash != registered_hash:
            raise ValueError(f"Episode Manifest changed after preflight for seed {seed}.")
        episode_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if episode_manifest.get("schema_version") == "stage6_workload_overlay_v1":
            historical_missing.update(_overlay_missing_provenance_fields(episode_manifest))

        file_records: list[dict[str, Any]] = []
        total_bytes = 0
        for relative_path, path, size_bytes in _episode_file_inventory(episode_dir):
            file_sha256 = _stable_file_sha256(path, expected_size=size_bytes)
            file_records.append(
                {
                    "relative_path": relative_path,
                    "size_bytes": size_bytes,
                    "sha256": file_sha256,
                }
            )
            total_bytes += size_bytes
        file_names = {record["relative_path"] for record in file_records}
        required = set(REQUIRED_TRAINING_ARTIFACTS) | {"episode_manifest.json"}
        if not required.issubset(file_names):
            raise FileNotFoundError(
                f"Episode {seed} is missing required snapshot files: {sorted(required - file_names)}."
            )
        if episode_manifest.get("schema_version") == "stage6_workload_overlay_v1":
            for name in episode_manifest.get("required_artifacts", []):
                if name not in file_names:
                    raise FileNotFoundError(
                        f"Episode {seed} manifest requires absent file {name!r}."
                    )
        episode_records.append(
            {
                "episode_seed": seed,
                "relative_directory": episode_dir.relative_to(PROJECT_ROOT).as_posix(),
                "episode_manifest_sha256": episode_manifest_hash,
                "pool_manifest_registered_episode_manifest_sha256": registered_hash,
                "pool_manifest_registration_matches": True,
                "manifest_schema": episode_manifest.get("schema_version"),
                "historical_provenance_missing_fields": _overlay_missing_provenance_fields(
                    episode_manifest
                )
                if episode_manifest.get("schema_version") == "stage6_workload_overlay_v1"
                else [],
                "files": file_records,
                "file_count": len(file_records),
                "total_bytes": total_bytes,
            }
        )
        total_file_count += len(file_records)
        if seed % 25 == 0:
            print(f"Hashed Training Episodes through {seed}.", flush=True)

    if len(episode_records) != EPISODE_COUNT:
        raise ValueError("Integrity baseline preflight did not enumerate exactly 300 Episodes.")
    baseline = {
        "schema_version": POOL_BASELINE_SCHEMA,
        "integrity_baseline_type": "CURRENT_SNAPSHOT",
        "historical_generation_attestation": "UNAVAILABLE",
        "historical_source_status": "HISTORICAL_PROVENANCE_INCOMPLETE",
        "training_pool_path": POOL_RELATIVE.as_posix(),
        "split": pool_manifest["split"],
        "seed_start": pool_manifest["seed_start"],
        "seed_end": pool_manifest["seed_end"],
        "episode_count": pool_manifest["episode_count"],
        "pool_manifest_sha256": hashlib.sha256(pool_manifest_bytes).hexdigest(),
        "pool_config_sha256": pool_manifest["config_sha256"],
        "pool_semantic_config_sha256": pool_manifest["semantic_config_sha256"],
        "generation_config_fingerprint": pool_manifest["generation_config_fingerprint"],
        "stage6_service_catalog_path": pool_manifest["stage6_service_catalog_path"],
        "stage6_service_catalog_sha256": pool_manifest["stage6_service_catalog_sha256"],
        "historical_source_evidence": {
            "status": "HISTORICAL_PROVENANCE_INCOMPLETE",
            "missing_fields": sorted(historical_missing),
            "explanation": (
                "The preserved stage6_workload_overlay_v1 Episode manifests omit historical "
                "source Git identity and source/generated artifact hashes. This baseline "
                "records file contents at baseline time; it does not attest to their original "
                "generation state or source revision."
            ),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "baseline_git_identity": _baseline_git_identity(),
        "episode_file_count": total_file_count,
        "episodes": episode_records,
    }
    encoded = (
        json.dumps(baseline, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary_path: Path | None = None
    created_official = False
    try:
        import tempfile

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{POOL_BASELINE_NAME}.", suffix=".tmp", dir=pool_path
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if baseline_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing integrity baseline: {baseline_path}")
        # Publish atomically without replacing a pre-existing baseline.
        if os.name == "nt":
            os.rename(temporary_path, baseline_path)
        else:
            os.link(temporary_path, baseline_path)
            temporary_path.unlink()
        created_official = True
        temporary_path = None
        _validate_training_pool(cfg, generation_fingerprint)
        print(
            f"Snapshot integrity baseline verified: {len(episode_records)} Episodes, "
            f"{total_file_count} files.",
            flush=True,
        )
        return baseline_path
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if created_official:
            baseline_path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    """Build the formal runner CLI without starting validation or training."""

    parser = argparse.ArgumentParser(
        description="Run the 300-Episode multi-node v4 formal PPO main experiment."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Strictly resume the existing formal run from latest_checkpoint.pt.",
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
    run_path = PROJECT_ROOT / RUN_DIRECTORY_RELATIVE
    _validate_formal_run_mode(run_path, resume=bool(args.resume))

    cfg, raw_config_hash, semantic_hash, generation_fingerprint = _load_main_experiment_contract()
    pool_path = _validate_training_pool(cfg, generation_fingerprint)
    baseline_sha256 = _stable_file_sha256(pool_path / POOL_BASELINE_NAME)

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
    validation_pool = _validate_validation_pool() if needs_validation else None

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
