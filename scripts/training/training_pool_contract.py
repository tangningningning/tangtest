"""Reviewed Training-Pool provenance and aligned-input helpers for research tools."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
GENERATION_ROOT = PROJECT_ROOT / "scripts" / "generation"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(GENERATION_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATION_ROOT))

from common import project_config  # noqa: E402
import generate_episode_realization  # noqa: E402
from common.environment_interface import (  # noqa: E402
    load_environment_context,
    prepare_aligned_rollout_steps,
)


CONFIG_RELATIVE = Path("configs/dqn/online_joint_dqn.toml")
# Frozen pool manifests retain their original source-path provenance.
POOL_MANIFEST_SOURCE_CONFIG_RELATIVE = Path("configs/proposed/system_config.toml")
TRAIN_POOL_RELATIVE = Path("environment_data/proposed_joint_training")
VALIDATION_POOL_RELATIVE = Path("environment_data/proposed_joint_validation")
TEST_POOL_RELATIVE = Path("environment_data/proposed_joint_test")
DEFAULT_TRAINING_SEED_START = 20001
DEFAULT_TEST_SEED_START = 40001

# The training data contract is pinned independently of runtime and reward values.
FORMAL_TRAINING_POOL_MANIFEST_SHA256 = (
    "9393eee43d1eb87dfd09c36eca149a43cf139162965cc735da50cb1180667ba6"
)
FORMAL_TRAINING_POOL_INTEGRITY_BASELINE_SHA256 = (
    "6c2309f0cc5c82e265e5c49d559b03e1ac1e9bbb32fe3cecb2f68c04251abe33"
)
FORMAL_TRAINING_POOL_EPISODE_FILE_COUNT = 3015
FORMAL_TRAINING_EPISODE_COUNT = 300
FORMAL_TRAINING_EPISODE_SLOTS = 300
FORMAL_TRAINING_ARRIVAL_HORIZON_SLOTS = FORMAL_TRAINING_EPISODE_SLOTS
FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS = 314
FORMAL_TEST_EPISODE_COUNT = 100
FORMAL_TRAINING_TASK_CONTRACT_SHA256 = (
    "dc7e83d99c69f1c3ea603c18fe9a2606f0bffe477556e75924e1d8dfd24387c3"
)
LEGACY_FORMAL_POOL_GENERATION_FINGERPRINT = (
    "4083b917948cd7203972c0edd5b40d2efefd8984dde27089a77da43d460b32f3"
)
SOURCE_CONFIG_SHA256 = "684bdfa4dc200db6227b2b00a7718797a98bf658e54d2d4dbef6f7d1f5babf77"
SOURCE_SEMANTIC_SHA256 = "10f893fc549c4016bedb1a5389ad996f9a3ff646aa012596a7230c284b13f542"
SOURCE_GENERATION_FINGERPRINT = "cfb3ee9fae004108af2d6bcb641609142c305d3ad6cc3ca74be13faaa6ff9865"
SOURCE_CATALOG_SHA256 = "3963eaa1c4cb116563fbbb9de0329dcc8c256c64fd849d5e0b75e9e51ca4c3ed"
TARGET_CATALOG_SHA256 = "6e8c497ed06d76e1251720ebc926e975a6a5c079fd224acab95ea5fb4799cf13"
VALIDATION_SOURCE_GENERATION_FINGERPRINT = "9ad3c6179be6cea67a0c95695f0cc2802e7570274c9be5446e51d996d5f7045d"
VALIDATION_SOURCE_CATALOG_PATH = "data/generated/service_catalog.json"
VALIDATION_SOURCE_CATALOG_SHA256 = "3adb64f9724a429a5cb6bf9c6eb609a54120d9918feaab1dd0475034af22e402"
SOURCE_PREDICTION_ARTIFACTS = {
    "popularity.csv",
    "popularity_history.csv",
    "popularity_predictions.csv",
    "prediction_scenarios.npz",
    "trajectory.csv",
    "vehicle_predictions.csv",
    "vehicle_roles.json",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any, *, exclusive: bool = False) -> None:
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        json.dump(
            _jsonable(payload),
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _active_episode_seeds(
    manifest: dict[str, Any],
    *,
    expected_seed_start: int,
    expected_episode_count: int,
) -> tuple[list[int], dict[str, Any]]:
    """Return the exact formal episode order, honoring explicit replacements."""

    selection = manifest.get("episode_selection")
    original = list(
        range(int(expected_seed_start), int(expected_seed_start) + int(expected_episode_count))
    )
    if selection is None:
        return original, {
            "active_episode_seeds": original,
            "replacement_episode_seeds": [],
            "excluded_episode_seeds": [],
            "trajectory_slots": FORMAL_TRAINING_EPISODE_SLOTS,
            "arrival_slots": FORMAL_TRAINING_EPISODE_SLOTS,
        }
    if not isinstance(selection, dict) or selection.get("schema_version") != "active_episode_seed_selection_v1":
        raise ValueError("Formal pool episode_selection schema is invalid.")
    expected_source_range = [
        int(expected_seed_start),
        int(expected_seed_start) + int(expected_episode_count) - 1,
    ]
    if selection.get("source_seed_range") != expected_source_range:
        raise ValueError("Formal pool episode_selection source seed range changed.")
    active = selection.get("active_episode_seeds")
    replacements = selection.get("replacement_episode_seeds")
    excluded = selection.get("excluded_episode_seeds")
    if not all(isinstance(value, list) for value in (active, replacements, excluded)):
        raise ValueError("Formal pool active/replacement/excluded seed lists are invalid.")
    try:
        active_int = [int(seed) for seed in active]
        replacements_int = [int(seed) for seed in replacements]
        excluded_int = [int(seed) for seed in excluded]
    except (TypeError, ValueError) as exc:
        raise ValueError("Formal pool seed lists must contain integer IDs.") from exc
    if (
        active_int != sorted(set(active_int))
        or replacements_int != sorted(set(replacements_int))
        or excluded_int != sorted(set(excluded_int))
        or len(active_int) != int(expected_episode_count)
    ):
        raise ValueError("Formal pool active seed list is not unique, ordered, and complete.")
    original_set = set(original)
    replacement_set = set(replacements_int)
    excluded_set = set(excluded_int)
    if (
        not excluded_set.issubset(original_set)
        or replacement_set & original_set
        or replacement_set & excluded_set
        or set(active_int) != (original_set - excluded_set) | replacement_set
    ):
        raise ValueError("Formal pool replacement selection does not balance its source range.")
    return active_int, {
        **selection,
        "active_episode_seeds": active_int,
        "replacement_episode_seeds": replacements_int,
        "excluded_episode_seeds": excluded_int,
    }


def _scene_identity_sha256(
    episode_dir: Path,
    *,
    split: str,
    seed: int,
    file_hashes: dict[str, str] | None = None,
) -> str:
    files = file_hashes or {
        path.name: _sha256(path)
        for path in sorted(episode_dir.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }
    return _canonical_sha256(
        {
            "schema_version": "formal_scene_identity_v1",
            "scene_id": f"{split}:{int(seed)}",
            "split": split,
            "episode_seed": int(seed),
            "files": files,
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_pool_config_compatibility(
    manifest: dict[str, Any],
    *,
    generation_fingerprint: str,
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    pool_fingerprint = manifest.get("target_generation_config_fingerprint")
    if pool_fingerprint == generation_fingerprint:
        policy = "CURRENT_GENERATION_CONTRACT"
        reason = "pool and runtime generation fingerprints are identical"
    else:
        legacy_payload = copy.deepcopy(
            generate_episode_realization._generation_config_payload(runtime_config)
        )
        legacy_payload["schema"] = "episode_manifest_v1"
        legacy_payload.pop("residual_sampling_contract", None)
        legacy_fingerprint = hashlib.sha256(
            json.dumps(
                legacy_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if (
            pool_fingerprint != LEGACY_FORMAL_POOL_GENERATION_FINGERPRINT
            or legacy_fingerprint != LEGACY_FORMAL_POOL_GENERATION_FINGERPRINT
        ):
            raise ValueError(
                "Training data generation contract mismatch: "
                f"pool={pool_fingerprint!r}, runtime={generation_fingerprint!r}, "
                f"reconstructed_legacy={legacy_fingerprint!r}"
            )
        policy = "LEGACY_FORMAL_POOL_V1_VERSION_COMPATIBLE"
        reason = (
            "the stored fingerprint is reproduced from the current generation "
            "payload after reverting the legacy schema marker and residual-sampling "
            "metadata; vehicle ordering is not a compatibility condition because "
            "the loader joins explicit IDs and preserves their within-role NPZ order; all "
            "generation config fields remain equal"
        )
    expected_config_path = (
        CONFIG_RELATIVE.as_posix()
        if manifest.get("schema_version") in {
            "formal_episode_pool_v2",
            "formal_episode_pool_v3",
        }
        else POOL_MANIFEST_SOURCE_CONFIG_RELATIVE.as_posix()
    )
    if manifest.get("config_path") != expected_config_path:
        raise ValueError(
            "Training Pool config_path does not identify the formal config."
        )
    return {
        "policy": policy,
        "reason": reason,
        "pool_generation_config_fingerprint": pool_fingerprint,
        "pool_config_sha256_recorded": manifest.get("config_sha256"),
        "pool_semantic_config_sha256_recorded": manifest.get(
            "semantic_config_sha256"
        ),
        "runtime_generation_config_fingerprint": generation_fingerprint,
        "historical_git_config_read": False,
    }


def _validate_pool(
    formal_root: Path,
    relative_pool: Path,
    *,
    expected_split: str,
    expected_seed_start: int,
    expected_episode_count: int,
    runtime_config: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    pool = formal_root / relative_pool
    manifest_path = pool / "formal_pool_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Formal {expected_split} pool manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "split": expected_split,
        "seed_start": expected_seed_start,
        "seed_end": expected_seed_start + expected_episode_count - 1,
        "episode_count": expected_episode_count,
        "slots_per_episode": FORMAL_TRAINING_EPISODE_SLOTS,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Formal {expected_split} pool manifest mismatch for {key}: "
                f"expected={value!r}, actual={manifest.get(key)!r}"
            )
    active_seeds, episode_selection = _active_episode_seeds(
        manifest,
        expected_seed_start=expected_seed_start,
        expected_episode_count=expected_episode_count,
    )
    split_manifest_path = formal_root / "environment_data" / "data_split_manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    split_manifest_hash = _sha256(split_manifest_path)
    if manifest.get("data_split_manifest_sha256") != split_manifest_hash:
        raise ValueError(
            f"Formal {expected_split} pool points to a different data split manifest: "
            f"expected_sha256={split_manifest_hash}, "
            f"actual={manifest.get('data_split_manifest_sha256')!r}"
        )
    split_payload = split_manifest.get(expected_split)
    if not isinstance(split_payload, dict):
        raise ValueError(f"Central data split manifest is missing {expected_split}.")
    if split_payload.get("active_episode_seeds") != active_seeds:
        raise ValueError(
            f"Formal {expected_split} active episode list differs from the central dataset identity."
        )
    if split_payload.get("dataset_identity_sha256") != manifest.get(
        "dataset_identity_sha256"
    ):
        raise ValueError(
            f"Formal {expected_split} dataset identity differs from the central manifest."
        )
    identity_payload = {
        "schema_version": "formal_episode_dataset_identity_v1",
        "split": expected_split,
        "network_sha256": split_manifest.get("sumo_network_sha256"),
        "slot_duration_s": float(manifest["task_data_contract"]["slot_duration_s"]),
        "arrival_slots": int(episode_selection["arrival_slots"]),
        "trajectory_slots": int(episode_selection["trajectory_slots"]),
        "active_episode_seeds": active_seeds,
        "active_episode_scene_sha256": manifest.get("active_episode_scene_sha256"),
    }
    if _canonical_sha256(identity_payload) != manifest.get("dataset_identity_sha256"):
        raise ValueError(f"Formal {expected_split} dataset identity hash is invalid.")
    scene_hashes = manifest.get("active_episode_scene_sha256")
    if not isinstance(scene_hashes, dict) or set(scene_hashes) != {
        str(seed) for seed in active_seeds
    }:
        raise ValueError(
            f"Formal {expected_split} scene hash index does not match its active episode list."
        )
    if split_payload.get("active_episode_scene_sha256") != scene_hashes:
        raise ValueError(
            f"Formal {expected_split} scene hash index differs from the central manifest."
        )
    if split_payload.get("active_episode_count") != expected_episode_count:
        raise ValueError(f"Central {expected_split} active episode count is invalid.")
    all_split_seeds = [
        int(seed)
        for split_name in ("train", "validation", "test")
        for seed in split_manifest.get(split_name, {}).get("active_episode_seeds", [])
    ]
    if len(all_split_seeds) != len(set(all_split_seeds)):
        raise ValueError("Formal Train/Validation/Test active seeds overlap.")
    if (
        int(episode_selection.get("arrival_slots", FORMAL_TRAINING_EPISODE_SLOTS))
        != FORMAL_TRAINING_ARRIVAL_HORIZON_SLOTS
        or int(episode_selection.get("trajectory_slots", FORMAL_TRAINING_EPISODE_SLOTS))
        != FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS
    ):
        raise ValueError("Formal cross-slot pool must define 300 arrival and 314 physical slots.")
    config_path = formal_root / CONFIG_RELATIVE
    cfg = (
        copy.deepcopy(runtime_config)
        if runtime_config is not None
        else project_config.parse_toml_file(config_path)
    )
    cfg["_meta"] = {
        "config_path": str(config_path.resolve()),
        "project_root": str(formal_root),
    }
    project_config.validate_for_formal_experiment(cfg)
    generation_fingerprint = generate_episode_realization.generation_config_fingerprint(cfg)
    catalog_relative = str(project_config.get_config_value(cfg, "paths.service_catalog"))
    catalog_path = project_config.resolve_path(formal_root, catalog_relative)
    catalog_hash = _sha256(catalog_path)
    service = cfg["service"]
    generation = service["generation"]
    task = cfg["task"]
    task_contract = {
        "schema_version": "formal_task_data_contract_v1",
        "task_data_model": "500_1000_KB",
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
        "service_catalog_path": catalog_relative,
        "service_catalog_sha256": catalog_hash,
    }
    contract_bytes = json.dumps(
        task_contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    task_contract_hash = hashlib.sha256(contract_bytes).hexdigest()
    config_compatibility = _validate_pool_config_compatibility(
        manifest,
        generation_fingerprint=generation_fingerprint,
        runtime_config=cfg,
    )
    expected_metadata = {
        "task_data_contract": task_contract,
        "task_data_contract_sha256": task_contract_hash,
        "service_catalog_path": catalog_relative,
        "service_catalog_sha256": catalog_hash,
    }
    mismatches = {
        key: {"pool": manifest.get(key), "current": value}
        for key, value in expected_metadata.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Formal {expected_split} pool metadata does not match current formal config: "
            f"{json.dumps(mismatches, ensure_ascii=False, sort_keys=True)}"
        )
    if manifest.get("task_data_model") != "500_1000_KB":
        raise ValueError("Formal task pool has an unexpected task-data model.")
    manifest_hash = _sha256(manifest_path)
    return pool, {
        "split": expected_split,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "pool_config_sha256": manifest.get("config_sha256"),
        "pool_semantic_config_sha256": manifest.get("semantic_config_sha256"),
        "config_compatibility": config_compatibility,
        "generation_config_fingerprint": generation_fingerprint,
        "pool_generation_config_fingerprint": manifest.get(
            "target_generation_config_fingerprint"
        ),
        "task_data_contract": task_contract,
        "task_data_contract_sha256": task_contract_hash,
        "generation_config_fields": generate_episode_realization._generation_config_payload(cfg),
        "service_catalog_sha256": catalog_hash,
        "task_data_model": manifest["task_data_model"],
        "active_episode_seeds": active_seeds,
        "episode_selection": episode_selection,
        "active_episode_scene_sha256": scene_hashes,
        "dataset_identity_sha256": manifest.get("dataset_identity_sha256"),
        "data_split_manifest_sha256": split_manifest_hash,
        "arrival_slots": int(episode_selection["arrival_slots"]),
        "trajectory_slots": int(episode_selection["trajectory_slots"]),
        "arrival_horizon_slots": int(episode_selection["arrival_slots"]),
        "execution_horizon_slots": int(episode_selection["trajectory_slots"]),
        "terminal_execution_slot": int(episode_selection["trajectory_slots"]) - 1,
    }


def _validate_training_integrity_baseline(
    pool: Path,
    pool_info: dict[str, Any],
) -> dict[str, Any]:
    baseline_path = pool / "formal_pool_integrity_baseline.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Training Pool integrity baseline is missing: {baseline_path}")
    baseline_hash = _sha256(baseline_path)
    pool_manifest = json.loads(
        Path(pool_info["manifest_path"]).read_text(encoding="utf-8")
    )
    if pool_manifest.get("schema_version") in {
        "formal_episode_pool_v2",
        "formal_episode_pool_v3",
    }:
        if baseline_hash != pool_manifest.get("integrity_baseline_sha256"):
            raise ValueError("New formal pool integrity baseline hash differs from its pool manifest.")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        expected_new = {
            "schema_version": "formal_task_data_integrity_baseline_v3",
            "integrity_baseline_type": "ACTIVE_EPISODE_SNAPSHOT",
            "task_data_model": "500_1000_KB",
            "split": "train",
            "seed_start": DEFAULT_TRAINING_SEED_START,
            "seed_end": DEFAULT_TRAINING_SEED_START + FORMAL_TRAINING_EPISODE_COUNT - 1,
            "episode_count": FORMAL_TRAINING_EPISODE_COUNT,
            "active_episode_seeds": pool_info["active_episode_seeds"],
            "data_split_manifest_sha256": pool_info["data_split_manifest_sha256"],
            "dataset_identity_sha256": pool_info["dataset_identity_sha256"],
            "service_catalog_sha256": pool_info["service_catalog_sha256"],
        }
        mismatches = {
            key: {"baseline": baseline.get(key), "expected": value}
            for key, value in expected_new.items()
            if baseline.get(key) != value
        }
        entries = baseline.get("episodes")
        if (
            mismatches
            or not isinstance(entries, list)
            or [int(row.get("episode_seed", -1)) for row in entries]
            != pool_info["active_episode_seeds"]
            or int(baseline.get("episode_file_count", -1))
            != int(pool_manifest.get("episode_file_count", -2))
        ):
            raise ValueError(
                "New formal pool integrity baseline metadata mismatch: "
                + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
            )
        return {
            "path": str(baseline_path),
            "sha256": baseline_hash,
            "file_count": int(baseline["episode_file_count"]),
            "active_episode_count": len(entries),
            "previous_integrity_validation_reused": False,
            "episode_files_rehashed_at_startup": False,
        }
    if baseline_hash != FORMAL_TRAINING_POOL_INTEGRITY_BASELINE_SHA256:
        raise ValueError(
            "Training Pool integrity baseline identity changed: "
            f"expected={FORMAL_TRAINING_POOL_INTEGRITY_BASELINE_SHA256}, "
            f"actual={baseline_hash}"
        )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "formal_task_data_integrity_baseline_v2",
        "integrity_baseline_type": "ACTIVE_EPISODE_SNAPSHOT",
        "task_data_model": "500_1000_KB",
        "pool_manifest_sha256": pool_info["manifest_sha256"],
        "task_data_contract_sha256": pool_info["task_data_contract_sha256"],
        "target_generation_config_fingerprint": pool_info[
            "pool_generation_config_fingerprint"
        ],
        "service_catalog_path": pool_info["task_data_contract"][
            "service_catalog_path"
        ],
        "service_catalog_sha256": pool_info["service_catalog_sha256"],
        "split": "train",
        "seed_start": DEFAULT_TRAINING_SEED_START,
        "seed_end": DEFAULT_TRAINING_SEED_START
        + FORMAL_TRAINING_EPISODE_COUNT
        - 1,
        "episode_count": FORMAL_TRAINING_EPISODE_COUNT,
        "episode_file_count": FORMAL_TRAINING_POOL_EPISODE_FILE_COUNT,
        "active_episode_seeds": pool_info["active_episode_seeds"],
    }
    mismatches = {
        key: {"baseline": baseline.get(key), "expected": value}
        for key, value in expected.items()
        if baseline.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Training Pool integrity baseline metadata mismatch: "
            f"{json.dumps(mismatches, ensure_ascii=False, sort_keys=True)}"
        )
    episode_entries = baseline.get("episodes")
    if (
        not isinstance(episode_entries, list)
        or [int(row.get("episode_seed", -1)) for row in episode_entries]
        != pool_info["active_episode_seeds"]
    ):
        raise ValueError("Training Pool active integrity baseline episode list is invalid.")
    return {
        "path": str(baseline_path),
        "sha256": baseline_hash,
        "file_count": FORMAL_TRAINING_POOL_EPISODE_FILE_COUNT,
        "active_episode_count": len(episode_entries),
        "previous_integrity_validation_reused": True,
        "episode_files_rehashed_at_startup": False,
    }


def validate_dqn_training_pool_for_run(
    *,
    formal_root: str | Path,
    config_path: str | Path,
    runtime_config: dict[str, Any],
    episode_pool: str | Path,
    episode_seed_start: int | None,
    episode_count: int,
    generate_missing: bool,
) -> dict[str, Any]:
    """Validate the frozen Training Pool against the DQN system configuration."""

    root = Path(formal_root).expanduser().resolve(strict=True)
    dqn_config = (root / CONFIG_RELATIVE).resolve(strict=True)
    actual_config = Path(config_path).expanduser().resolve(strict=True)
    if actual_config != dqn_config:
        raise ValueError(
            "DQN training requires configs/dqn/online_joint_dqn.toml: "
            f"expected={dqn_config}, actual={actual_config}"
        )
    expected_pool = (root / TRAIN_POOL_RELATIVE).resolve(strict=True)
    actual_pool = project_config.resolve_path(root, episode_pool).resolve(strict=True)
    if actual_pool != expected_pool:
        raise ValueError(
            "DQN training requires the frozen proposed_joint_training pool: "
            f"expected={expected_pool}, actual={actual_pool}"
        )
    if episode_seed_start != DEFAULT_TRAINING_SEED_START:
        raise ValueError(
            "DQN Training Pool seed start must be "
            f"{DEFAULT_TRAINING_SEED_START}, actual={episode_seed_start!r}"
        )
    if int(episode_count) != FORMAL_TRAINING_EPISODE_COUNT:
        raise ValueError(
            "DQN Training requires the complete frozen Pool episode count: "
            f"expected={FORMAL_TRAINING_EPISODE_COUNT}, actual={episode_count}"
        )
    if bool(generate_missing):
        raise ValueError("DQN Training Pool validation does not generate missing Episodes.")

    cfg = copy.deepcopy(runtime_config)
    cfg["_meta"] = {
        "config_path": str(actual_config),
        "project_root": str(root),
    }
    project_config.validate_for_formal_experiment(cfg)
    protocol = project_config.get_episode_data_protocol(cfg)
    expected_protocol = {
        "protocol": "full_episode_randomized",
        "physical_episode_slots": FORMAL_TRAINING_EPISODE_SLOTS,
        "episode_slots": FORMAL_TRAINING_EPISODE_SLOTS,
        "arrival_horizon_slots": FORMAL_TRAINING_ARRIVAL_HORIZON_SLOTS,
        "execution_horizon_slots": FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS,
        "train_origin_start": 0,
        "train_origin_end": FORMAL_TRAINING_ARRIVAL_HORIZON_SLOTS - 1,
        "rollout_steps": FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS,
        "terminal_origin": FORMAL_TRAINING_ARRIVAL_HORIZON_SLOTS - 1,
        "terminal_execution_slot": FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS - 1,
    }
    protocol_mismatches = {
        key: {"current": protocol.get(key), "expected": value}
        for key, value in expected_protocol.items()
        if protocol.get(key) != value
    }
    if protocol_mismatches:
        raise ValueError(
            "DQN Training Episode protocol mismatch: "
            f"{json.dumps(protocol_mismatches, ensure_ascii=False, sort_keys=True)}"
        )

    pool, pool_info = _validate_pool(
        root,
        TRAIN_POOL_RELATIVE,
        expected_split="train",
        expected_seed_start=DEFAULT_TRAINING_SEED_START,
        expected_episode_count=FORMAL_TRAINING_EPISODE_COUNT,
        runtime_config=cfg,
    )
    pool_manifest = json.loads(
        Path(pool_info["manifest_path"]).read_text(encoding="utf-8")
    )
    is_new_formal_pool = pool_manifest.get("schema_version") in {
        "formal_episode_pool_v2",
        "formal_episode_pool_v3",
    }
    if (
        not is_new_formal_pool
        and pool_info["manifest_sha256"] != FORMAL_TRAINING_POOL_MANIFEST_SHA256
    ):
        raise ValueError(
            "Training Pool Manifest identity changed: "
            f"expected={FORMAL_TRAINING_POOL_MANIFEST_SHA256}, "
            f"actual={pool_info['manifest_sha256']}"
        )
    if (
        not is_new_formal_pool
        and pool_info["task_data_contract_sha256"] != FORMAL_TRAINING_TASK_CONTRACT_SHA256
    ):
        raise ValueError("Training Pool task-data contract is not the reviewed contract.")
    baseline_info = _validate_training_integrity_baseline(pool, pool_info)
    return {
        **pool_info,
        "pool_path": str(pool),
        "integrity_baseline": baseline_info,
        "data_input_compatibility": {
            "policy": pool_info["config_compatibility"]["policy"],
            "reason": pool_info["config_compatibility"]["reason"],
            "pool_manifest_schema_version": pool_manifest.get("schema_version"),
            "pool_generation_config_fingerprint": pool_info[
                "pool_generation_config_fingerprint"
            ],
            "runtime_generation_config_fingerprint": pool_info[
                "generation_config_fingerprint"
            ],
            "vehicle_order_handling": (
                "VALIDATE_ROLE_ID_SETS; PRESERVE_NPZ_WITHIN_ROLE_ORDER; "
                "MAP_POSITION_AND_TASK_AXES_BY_ID; JOIN_CSV_ROWS_BY_ID"
            ),
            "saved_prediction_handling": (
                "READ_EXISTING_SCENARIOS_AND_PRESERVE_ORIGINAL_NPZ_METADATA"
            ),
            "historical_source_status": pool_manifest.get(
                "source_provenance", {}
            ).get("historical_source_status", "UNRESOLVED"),
        },
        "episode_seed_start": min(pool_info["active_episode_seeds"]),
        "episode_seed_end": max(pool_info["active_episode_seeds"]),
        "episode_count": FORMAL_TRAINING_EPISODE_COUNT,
        "episode_seeds": pool_info["active_episode_seeds"],
        "episode_slots": FORMAL_TRAINING_EPISODE_SLOTS,
        "arrival_horizon_slots": FORMAL_TRAINING_ARRIVAL_HORIZON_SLOTS,
        "execution_horizon_slots": FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS,
        "terminal_origin": FORMAL_TRAINING_ARRIVAL_HORIZON_SLOTS - 1,
        "terminal_execution_slot": FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS - 1,
        "protocol": protocol,
    }


def _validate_episode_snapshot(
    pool: Path,
    seed: int,
    pool_info: dict[str, Any],
) -> None:
    episode_dir = pool / f"episode_{int(seed)}"
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"Formal realization is missing: {episode_dir}")
    manifest_path = episode_dir / "episode_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Transformed episode manifest is missing: {manifest_path}")
    episode_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool_manifest = json.loads(
        Path(pool_info["manifest_path"]).read_text(encoding="utf-8")
    )
    if pool_manifest.get("schema_version") in {
        "formal_episode_pool_v2",
        "formal_episode_pool_v3",
    }:
        generation_fields = pool_info["generation_config_fields"]
        if (
            episode_manifest.get("schema_version") != "episode_manifest_v2"
            or int(episode_manifest.get("episode_seed", -1)) != int(seed)
            or episode_manifest.get("generation_config_fingerprint")
            != pool_info["generation_config_fingerprint"]
            or episode_manifest.get("service_catalog_sha256")
            != pool_info["service_catalog_sha256"]
            or episode_manifest.get("derived_episode_seed_streams")
            != generate_episode_realization.derive_episode_seeds(int(seed))
        ):
            raise ValueError(f"Native formal episode provenance mismatch: episode_seed={seed}")
        stored_names = episode_manifest.get("generated_artifacts")
        stored_hashes = episode_manifest.get("generated_artifact_sha256")
        actual_paths = sorted(
            path
            for path in episode_dir.iterdir()
            if path.is_file() and path.name != manifest_path.name
        )
        if (
            not isinstance(stored_names, list)
            or not isinstance(stored_hashes, dict)
            or set(stored_names) != {path.name for path in actual_paths}
            or set(stored_hashes) != set(stored_names)
        ):
            raise ValueError(f"Native formal episode artifact inventory mismatch: episode_seed={seed}")
        for artifact_path in actual_paths:
            if _sha256(artifact_path) != stored_hashes.get(artifact_path.name):
                raise ValueError(
                    f"Native formal episode artifact hash mismatch: episode_seed={seed}, artifact={artifact_path.name}"
                )
        actual_scene_hash = _sha256(manifest_path)
        expected_scene_hash = pool_info["active_episode_scene_sha256"].get(str(seed))
        if actual_scene_hash != expected_scene_hash:
            raise ValueError(f"Native formal episode scene identity mismatch: episode_seed={seed}")
        if pool_manifest.get("schema_version") == "formal_episode_pool_v3":
            source = episode_manifest.get("traffic_source_provenance")
            streams = generate_episode_realization.derive_episode_seeds(int(seed))
            simulation = generation_fields.get("simulation", {})
            expected_source = {
                "schema_version": "independent_episode_sumo_v1",
                "split": pool_info["split"],
                "episode_seed": int(seed),
                "sumo_seed": int(streams["sumo_seed"]),
                "traffic_seed": int(streams["traffic_seed"]),
                "warmup_time_s": float(simulation["warmup_time_s"]),
                "arrival_horizon_slots": FORMAL_TRAINING_ARRIVAL_HORIZON_SLOTS,
                "execution_horizon_slots": FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS,
                "simulation_end_time_s": float(simulation["warmup_time_s"])
                + FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS
                * float(simulation["dt_s"]),
                "random_trips_period_s": float(simulation["random_trips_period_s"]),
                "random_trips_departure_begin_s": float(simulation["random_trips_departure_begin_s"]),
                "random_trips_departure_end_s": float(simulation["random_trips_departure_end_s"]),
                "random_trips_min_distance_m": float(simulation["random_trips_min_distance_m"]),
                "random_trips_intermediate": int(simulation["random_trips_intermediate"]),
                "random_trips_maxtries": int(simulation["random_trips_maxtries"]),
            }
            time_contract = episode_manifest.get("simulation", {})
            if (
                source != expected_source
                or int(time_contract.get("arrival_horizon_slots", -1))
                != FORMAL_TRAINING_ARRIVAL_HORIZON_SLOTS
                or int(time_contract.get("execution_horizon_slots", -1))
                != FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS
            ):
                raise ValueError(
                    f"Independent Episode traffic/time identity mismatch: split={pool_info['split']}, seed={seed}"
                )
        elif pool_info["split"] == "train":
            source = episode_manifest.get("traffic_source_provenance")
            if (
                not isinstance(source, dict)
                or source.get("schema_version") != "continuous_train_window_source_v1"
                or int(source.get("window_index", -1)) != int(seed) - DEFAULT_TRAINING_SEED_START
                or int(source.get("physical_trajectory_slots", -1))
                != FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS
            ):
                raise ValueError(f"Native Train episode is missing its continuous-window source identity: {seed}")
        else:
            extension = episode_manifest.get("cross_slot_trajectory_extension")
            tail_predictions = episode_manifest.get("prediction_tail_regeneration")
            if (
                not isinstance(extension, dict)
                or int(extension.get("trajectory_slots", -1))
                != FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS
                or not isinstance(tail_predictions, dict)
                or int(tail_predictions.get("execution_horizon_slots", -1))
                != FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS
            ):
                raise ValueError(f"Native evaluation episode is missing its 314-slot extension/prediction identity: {seed}")
        vehicle_prediction_config = generation_fields.get("prediction", {}).get(
            "vehicle", {}
        )
        generate_episode_realization.validate_written_outputs(
            episode_dir,
            task_vehicle_count=int(generation_fields["simulation"].get("num_task_vehicles", 20)),
            research_vehicle_count=int(generation_fields["simulation"].get("num_research_vehicles", 25)),
            horizon_steps=int(vehicle_prediction_config.get("horizon", 0)),
            origin_count=FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS,
        )
        return
    supplement = episode_manifest.get("formal_pool_supplement")
    if supplement is not None:
        _validate_supplement_episode_snapshot(
            episode_dir=episode_dir,
            episode_manifest=episode_manifest,
            pool_manifest=pool_manifest,
            pool_info=pool_info,
            seed=int(seed),
            split=str(pool_info["split"]),
        )
        return
    expected_episode_values = {
        "schema_version": "formal_task_transform_v1",
        "task_data_model": pool_info["task_data_model"],
        "split": pool_info["split"],
        "episode_seed": int(seed),
        "config_sha256": pool_info["pool_config_sha256"],
        "semantic_config_sha256": pool_info["pool_semantic_config_sha256"],
        "target_generation_config_fingerprint": pool_info[
            "pool_generation_config_fingerprint"
        ],
        "task_data_contract_sha256": pool_info["task_data_contract_sha256"],
        "service_catalog_sha256": pool_info["service_catalog_sha256"],
    }
    mismatches = {
        key: {"episode": episode_manifest.get(key), "pool": expected_value}
        for key, expected_value in expected_episode_values.items()
        if episode_manifest.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(
            "Transformed episode metadata does not match its frozen pool contract: "
            f"{json.dumps(mismatches, ensure_ascii=False, sort_keys=True)}"
        )

    source_manifest_sha = pool_manifest.get("source_episode_manifest_sha256", {}).get(
        str(seed)
    )
    source_provenance = episode_manifest.get("source_generation_provenance")
    if pool_info["split"] == "train":
        expected_source_values = {
            "source_episode_manifest_sha256": source_manifest_sha,
            "source_episode_manifest_schema": "stage6_workload_overlay_v1",
            "source_generation_config_fingerprint": SOURCE_GENERATION_FINGERPRINT,
            "source_config_sha256": SOURCE_CONFIG_SHA256,
            "source_semantic_config_sha256": SOURCE_SEMANTIC_SHA256,
            "source_service_catalog_sha256": SOURCE_CATALOG_SHA256,
            "canonical_old_service_catalog_sha256": SOURCE_CATALOG_SHA256,
        }
        source_contract_valid = isinstance(source_provenance, dict) and all(
            source_provenance.get(key) == value
            for key, value in expected_source_values.items()
        )
    else:
        expected_source_values = {
            "source_episode_manifest_sha256": source_manifest_sha,
            "source_episode_manifest_schema": "episode_manifest_v1",
            "source_generation_config_fingerprint": VALIDATION_SOURCE_GENERATION_FINGERPRINT,
            "source_service_catalog_path": VALIDATION_SOURCE_CATALOG_PATH,
            "source_service_catalog_sha256": VALIDATION_SOURCE_CATALOG_SHA256,
            "canonical_old_service_catalog_sha256": SOURCE_CATALOG_SHA256,
        }
        source_contract_valid = isinstance(source_provenance, dict) and all(
            source_provenance.get(key) == value
            for key, value in expected_source_values.items()
        )
        source_fields = (
            source_provenance.get("source_generation_config_fields")
            if isinstance(source_provenance, dict)
            else None
        )
        # Test Pool 的来源是冻结的 v1 清单；逐项字段由下方固定 SHA-256
        # 精确绑定，不能将 v2 的新字段要求套用到历史 v1，也不能改写其来源。
        if (
            not isinstance(source_fields, dict)
            or source_fields.get("schema") != "episode_manifest_v1"
        ):
            source_contract_valid = False
        elif hashlib.sha256(
            json.dumps(
                source_fields,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest() != VALIDATION_SOURCE_GENERATION_FINGERPRINT:
            source_contract_valid = False
        source_git_sha = (
            source_provenance.get("source_git_sha")
            if isinstance(source_provenance, dict)
            else None
        )
        if (
            not isinstance(source_git_sha, str)
            or len(source_git_sha) != 40
            or any(character not in "0123456789abcdef" for character in source_git_sha.lower())
            or not isinstance(source_provenance.get("source_git_dirty"), bool)
        ):
            source_contract_valid = False
    if not source_contract_valid:
        raise ValueError(
            "Transformed episode prediction/workload source contract is not the reviewed snapshot: "
            f"episode_seed={seed}"
        )
    source_artifact_hashes = source_provenance.get("source_generated_artifact_sha256")
    if not isinstance(source_artifact_hashes, dict) or not SOURCE_PREDICTION_ARTIFACTS.issubset(
        source_artifact_hashes
    ) or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
        for value in source_artifact_hashes.values()
    ):
        raise ValueError(
            "Transformed episode is missing valid source prediction artifact hashes: "
            f"episode_seed={seed}"
        )

    transform = episode_manifest.get("task_data_transform")
    task_contract = pool_info["task_data_contract"]
    expected_source_catalog_path = (
        "environment_data/proposed_joint_training/service_catalog_500_2000.json"
        if pool_info["split"] == "train"
        else VALIDATION_SOURCE_CATALOG_PATH
    )
    expected_source_catalog_hash = (
        SOURCE_CATALOG_SHA256
        if pool_info["split"] == "train"
        else VALIDATION_SOURCE_CATALOG_SHA256
    )
    expected_normalization_steps = (
        ["canonical 500–2000 KB -> target 500–1000 KB using /3"]
        if pool_info["split"] == "train"
        else [
            "source catalog -> canonical old Training means",
            "canonical 500–2000 KB -> target 500–1000 KB using /3",
        ]
    )
    expected_transform_values = {
        "schema_version": "formal_task_data_transform_v1",
        "task_data_model": pool_info["task_data_model"],
        "split": pool_info["split"],
        "episode_seed": int(seed),
        "source_service_catalog_path": expected_source_catalog_path,
        "source_service_catalog_sha256": expected_source_catalog_hash,
        "canonical_old_service_catalog_sha256": SOURCE_CATALOG_SHA256,
        "target_service_catalog_sha256": TARGET_CATALOG_SHA256,
        "normalization_steps": expected_normalization_steps,
        "service_input_mapping": "D_f_new_kb = 500 + (D_f_old_kb - 500) / 3",
        "task_workload_mapping": (
            "D_i_new = D_i_source * D_f_new / D_f_source; "
            "L_i_new = L_i_source * D_f_new / D_f_source; "
            "input/load jitter factors are preserved."
        ),
        "input_relative_jitter": task_contract["input_relative_jitter"],
        "load_relative_jitter": task_contract["load_relative_jitter"],
    }
    if not isinstance(transform, dict) or any(
        transform.get(key) != value
        for key, value in expected_transform_values.items()
    ):
        raise ValueError(
            "Transformed episode task mapping is not the reviewed 500–1000 KB contract: "
            f"episode_seed={seed}"
        )

    stored_artifact_hashes = episode_manifest.get("artifact_sha256")
    generated_artifacts = episode_manifest.get("generated_artifacts")
    if (
        not isinstance(stored_artifact_hashes, dict)
        or not isinstance(generated_artifacts, list)
        or set(generated_artifacts) != set(stored_artifact_hashes)
    ):
        raise ValueError(
            "Transformed episode artifact manifest is incomplete: "
            f"episode_seed={seed}"
        )
    actual_paths = sorted(
        path
        for path in episode_dir.iterdir()
        if path.is_file() and path.name != manifest_path.name
    )
    actual_names = {path.name for path in actual_paths}
    if actual_names != set(generated_artifacts):
        raise ValueError(
            "Transformed episode artifact file set differs from its frozen manifest: "
            f"episode_seed={seed}"
        )
    legacy_required = pool_manifest.get("required_artifacts_by_episode_schema", {}).get(
        "formal_task_transform_v1", pool_manifest.get("required_artifacts", [])
    )
    if not isinstance(legacy_required, list) or not set(legacy_required).issubset(actual_names):
        raise ValueError(f"Transformed episode is missing its contracted artifacts: seed={seed}")
    for artifact_path in actual_paths:
        expected_hash = stored_artifact_hashes.get(artifact_path.name)
        if not isinstance(expected_hash, str) or _sha256(artifact_path) != expected_hash:
            raise ValueError(
                "Transformed episode artifact hash mismatch: "
                f"episode_seed={seed}, artifact={artifact_path.name}"
            )
    scene_files = {
        path.name: _sha256(path)
        for path in actual_paths
    }
    scene_files[manifest_path.name] = _sha256(manifest_path)
    expected_scene_hash = pool_info["active_episode_scene_sha256"].get(str(seed))
    actual_scene_hash = _scene_identity_sha256(
        episode_dir,
        split=str(pool_info["split"]),
        seed=int(seed),
        file_hashes=scene_files,
    )
    if expected_scene_hash != actual_scene_hash:
        raise ValueError(
            "Formal scene identity hash mismatch: "
            f"episode_seed={seed}, expected={expected_scene_hash!r}, actual={actual_scene_hash}"
        )


def _validate_supplement_episode_snapshot(
    *,
    episode_dir: Path,
    episode_manifest: dict[str, Any],
    pool_manifest: dict[str, Any],
    pool_info: dict[str, Any],
    seed: int,
    split: str,
) -> None:
    supplement = episode_manifest.get("formal_pool_supplement")
    extension = episode_manifest.get("cross_slot_trajectory_extension")
    selection = pool_info["episode_selection"]
    if seed not in selection["replacement_episode_seeds"]:
        raise ValueError(f"Non-replacement episode unexpectedly has supplement metadata: {seed}")
    expected_supplement = {
        "schema_version": "formal_pool_supplement_v1",
        "split": split,
        "scene_id": f"{split}:{seed}",
        "episode_seed": seed,
        "task_arrival_slots": pool_info["arrival_slots"],
        "physical_trajectory_slots": pool_info["trajectory_slots"],
        "task_randomness_streams": [
            "task_arrival_seed",
            "service_request_seed",
            "workload_seed",
            "popularity_seed",
        ],
        "traffic_randomness_streams": ["traffic_seed", "sumo_seed"],
    }
    if not isinstance(supplement, dict) or any(
        supplement.get(key) != value for key, value in expected_supplement.items()
    ):
        raise ValueError(f"Supplemental episode identity metadata is invalid: seed={seed}")
    if supplement.get("shared_network_sha256") != episode_manifest.get("sumo_network_sha256"):
        raise ValueError(f"Supplemental episode network identity is invalid: seed={seed}")
    expected_extension = {
        "schema_version": "sumo_same_route_trajectory_extension_v1",
        "split": split,
        "scene_id": f"{split}:{seed}",
        "episode_seed": seed,
        "dt_s": pool_info["task_data_contract"]["slot_duration_s"],
        "arrival_slots": pool_info["arrival_slots"],
        "trajectory_slots": pool_info["trajectory_slots"],
        "tail_slots": pool_info["trajectory_slots"] - pool_info["arrival_slots"],
        "network_sha256": episode_manifest.get("sumo_network_sha256"),
        "sumo_demand_sha256": episode_manifest.get("generated_artifact_sha256", {}).get(
            "sumo_demand.rou.xml"
        ),
        "roles_sha256": episode_manifest.get("generated_artifact_sha256", {}).get(
            "vehicle_roles.json"
        ),
        "trajectory_sha256": episode_manifest.get("generated_artifact_sha256", {}).get(
            "trajectory.csv"
        ),
        "prefix_exact_match": True,
        "end_boundary_exact_match": True,
        "fixed_research_ids_present_in_every_tail_slot": True,
    }
    if not isinstance(extension, dict) or any(
        extension.get(key) != value for key, value in expected_extension.items()
    ):
        raise ValueError(f"Supplemental trajectory extension metadata is invalid: seed={seed}")
    if extension.get("original_prefix_canonical_sha256") != extension.get(
        "replayed_prefix_canonical_sha256"
    ):
        raise ValueError(f"Supplemental trajectory prefix digest differs: seed={seed}")
    if episode_manifest.get("generation_config_fingerprint") != pool_info[
        "generation_config_fingerprint"
    ]:
        raise ValueError(f"Supplemental generation config differs from runtime: seed={seed}")
    if episode_manifest.get("service_catalog_sha256") != pool_info["service_catalog_sha256"]:
        raise ValueError(f"Supplemental service catalog differs from formal pool: seed={seed}")
    derived_seeds = generate_episode_realization.derive_episode_seeds(seed)
    if episode_manifest.get("derived_episode_seed_streams") != derived_seeds:
        raise ValueError(f"Supplemental episode seed streams are invalid: seed={seed}")

    generated_artifacts = episode_manifest.get("generated_artifacts")
    stored_hashes = episode_manifest.get("generated_artifact_sha256")
    if not isinstance(generated_artifacts, list) or not isinstance(stored_hashes, dict):
        raise ValueError(f"Supplemental artifact inventory is missing: seed={seed}")
    actual_paths = sorted(
        path
        for path in episode_dir.iterdir()
        if path.is_file() and path.name != "episode_manifest.json"
    )
    supplemental_required = pool_manifest.get("required_artifacts_by_episode_schema", {}).get(
        "episode_manifest_v2+formal_pool_supplement_v1"
    )
    if not isinstance(supplemental_required, list) or set(supplemental_required) != {
        path.name for path in actual_paths
    }:
        raise ValueError(f"Supplemental episode does not satisfy its artifact schema: seed={seed}")
    if {path.name for path in actual_paths} != set(generated_artifacts) or set(
        generated_artifacts
    ) != set(stored_hashes):
        raise ValueError(f"Supplemental artifact file set differs: seed={seed}")
    scene_files = {}
    for artifact_path in actual_paths:
        actual_hash = _sha256(artifact_path)
        if stored_hashes.get(artifact_path.name) != actual_hash:
            raise ValueError(
                f"Supplemental artifact hash mismatch: seed={seed}, file={artifact_path.name}"
            )
        scene_files[artifact_path.name] = actual_hash
    if extension.get("trajectory_sha256") != stored_hashes.get("trajectory.csv"):
        raise ValueError(f"Supplemental trajectory hash is invalid: seed={seed}")
    scene_files["episode_manifest.json"] = _sha256(
        episode_dir / "episode_manifest.json"
    )
    actual_scene_hash = _scene_identity_sha256(
        episode_dir,
        split=split,
        seed=seed,
        file_hashes=scene_files,
    )
    expected_scene_hash = pool_info["active_episode_scene_sha256"].get(str(seed))
    if actual_scene_hash != expected_scene_hash:
        raise ValueError(
            f"Supplemental scene identity hash mismatch: seed={seed}, "
            f"expected={expected_scene_hash!r}, actual={actual_scene_hash}"
        )


def _episode_context(
    formal_root: Path,
    pool: Path,
    seed: int,
    pool_info: dict[str, Any],
) -> Any:
    episode_dir = pool / f"episode_{int(seed)}"
    if int(seed) not in set(pool_info["active_episode_seeds"]):
        raise ValueError(
            f"Episode seed is not in the formal active set for {pool_info['split']}: {seed}"
        )
    _validate_episode_snapshot(pool, seed, pool_info)
    return load_environment_context(
        str((formal_root / CONFIG_RELATIVE).resolve()),
        realization_dir=episode_dir,
    )


def _aligned_window(context: Any, origin_start: int, slots: int) -> list[Any]:
    start = int(origin_start)
    count = int(slots)
    if start < int(context.train_origin_start) or start + count - 1 > int(context.train_origin_end):
        raise ValueError("Requested short window is outside this realization's formal origins.")
    if context.realization is None:
        raise ValueError("Stage-1B requires an explicit Training or Validation realization.")
    return prepare_aligned_rollout_steps(
        context.root,
        context.scenarios,
        start,
        count,
        context.realization.vehicle_predictions,
        context.realization.popularity_predictions,
        context.realization.tasks,
        context.realization.trajectory,
        include_bootstrap_step=True,
        prediction_horizon=int(context.observation_spec.prediction_horizon),
        prediction_dt_s=float(project_config.get_config_value(context.cfg, "simulation.dt_s")),
        prediction_terminal_origin=int(context.train_origin_end),
        expected_vehicle_ids=context.observation_spec.vehicle_ids,
        expected_task_vehicle_ids=context.observation_spec.task_vehicle_ids,
        expected_service_ids=context.observation_spec.service_ids,
        include_prediction_horizon_mask=(context.observation_spec.horizon_mask_slice is not None),
    )
