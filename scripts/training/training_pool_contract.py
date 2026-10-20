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
from ppo_simple.interface import (  # noqa: E402
    load_environment_context,
    prepare_aligned_rollout_steps,
)


CONFIG_RELATIVE = Path("configs/proposed/system_config.toml")
ALTERNATING_ACTOR_UPDATES_CONFIG_RELATIVE = Path(
    "configs/proposed/alternating_actor_updates_v1.toml"
)
TRAIN_POOL_RELATIVE = Path("environment_data/proposed_joint_training")
VALIDATION_POOL_RELATIVE = Path("environment_data/proposed_joint_validation")
DEFAULT_TRAINING_SEED_START = 20001

# The training data contract is pinned independently of runtime/reward/MAPPO values.
FORMAL_TRAINING_POOL_MANIFEST_SHA256 = (
    "b6cae236b029f4d24c7488a91f0825247a20f475ec76b2d80737de44ae491bd3"
)
FORMAL_TRAINING_POOL_INTEGRITY_BASELINE_SHA256 = (
    "43e82e19a1231ea2719541080c8b16b2137a7df5166a9a94a1ca19b9b7e01538"
)
FORMAL_TRAINING_POOL_EPISODE_FILE_COUNT = 3002
FORMAL_TRAINING_EPISODE_COUNT = 300
FORMAL_TRAINING_EPISODE_SLOTS = 300
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
    if manifest.get("config_path") != CONFIG_RELATIVE.as_posix():
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
    }


def _validate_training_integrity_baseline(
    pool: Path,
    pool_info: dict[str, Any],
) -> dict[str, Any]:
    baseline_path = pool / "formal_pool_integrity_baseline.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Training Pool integrity baseline is missing: {baseline_path}")
    baseline_hash = _sha256(baseline_path)
    if baseline_hash != FORMAL_TRAINING_POOL_INTEGRITY_BASELINE_SHA256:
        raise ValueError(
            "Training Pool integrity baseline identity changed: "
            f"expected={FORMAL_TRAINING_POOL_INTEGRITY_BASELINE_SHA256}, "
            f"actual={baseline_hash}"
        )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "formal_task_data_integrity_baseline_v1",
        "integrity_baseline_type": "CURRENT_SNAPSHOT",
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
    return {
        "path": str(baseline_path),
        "sha256": baseline_hash,
        "file_count": FORMAL_TRAINING_POOL_EPISODE_FILE_COUNT,
        "previous_integrity_validation_reused": True,
        "episode_files_rehashed_at_startup": False,
    }


def validate_formal_training_pool_for_run(
    *,
    formal_root: str | Path,
    config_path: str | Path,
    runtime_config: dict[str, Any],
    episode_pool: str | Path,
    episode_seed_start: int | None,
    episode_count: int,
    generate_missing: bool,
) -> dict[str, Any]:
    """Run the lightweight generation-contract gate for a new formal run."""

    root = Path(formal_root).expanduser().resolve(strict=True)
    alternating_config = (
        root / ALTERNATING_ACTOR_UPDATES_CONFIG_RELATIVE
    ).resolve(strict=True)
    actual_config = Path(config_path).expanduser().resolve(strict=True)
    if actual_config != alternating_config:
        raise ValueError(
            "Formal Training requires the reviewed "
            "ALTERNATING_ACTOR_UPDATES_V1 config: "
            f"expected={alternating_config}, actual={actual_config}"
        )
    actor_update_config = runtime_config["algorithm"]["ppo"]
    if (
        actor_update_config.get("actor_update_mode") != "ALTERNATING"
        or actor_update_config.get("actor_update_first") != "MICRO"
        or actor_update_config.get("actor_update_period") != 1
    ):
        raise ValueError(
            "ALTERNATING_ACTOR_UPDATES_V1 requires mode=ALTERNATING, "
            "first=MICRO, and period=1."
        )
    expected_pool = (root / TRAIN_POOL_RELATIVE).resolve(strict=True)
    actual_pool = project_config.resolve_path(root, episode_pool).resolve(strict=True)
    if actual_pool != expected_pool:
        raise ValueError(
            "Formal Training requires the frozen proposed_joint_training pool: "
            f"expected={expected_pool}, actual={actual_pool}"
        )
    if episode_seed_start != DEFAULT_TRAINING_SEED_START:
        raise ValueError(
            "Formal Training seed start must be "
            f"{DEFAULT_TRAINING_SEED_START}, actual={episode_seed_start!r}"
        )
    if int(episode_count) != FORMAL_TRAINING_EPISODE_COUNT:
        raise ValueError(
            "Formal Training requires the complete frozen Pool episode count: "
            f"expected={FORMAL_TRAINING_EPISODE_COUNT}, actual={episode_count}"
        )
    if bool(generate_missing):
        raise ValueError("Formal Training Pool gate does not permit generating missing Episodes.")

    cfg = copy.deepcopy(runtime_config)
    cfg["_meta"] = {
        "config_path": str(actual_config),
        "project_root": str(root),
    }
    project_config.validate_for_formal_experiment(cfg)
    protocol = project_config.get_ppo_data_protocol(cfg)
    expected_protocol = {
        "protocol": "full_episode_randomized",
        "physical_episode_slots": FORMAL_TRAINING_EPISODE_SLOTS,
        "episode_slots": FORMAL_TRAINING_EPISODE_SLOTS,
        "train_origin_start": 0,
        "train_origin_end": FORMAL_TRAINING_EPISODE_SLOTS - 1,
        "rollout_steps": FORMAL_TRAINING_EPISODE_SLOTS,
        "terminal_origin": FORMAL_TRAINING_EPISODE_SLOTS - 1,
    }
    protocol_mismatches = {
        key: {"current": protocol.get(key), "expected": value}
        for key, value in expected_protocol.items()
        if protocol.get(key) != value
    }
    if protocol_mismatches:
        raise ValueError(
            "Formal Training Episode protocol mismatch: "
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
    if pool_info["manifest_sha256"] != FORMAL_TRAINING_POOL_MANIFEST_SHA256:
        raise ValueError(
            "Training Pool Manifest identity changed: "
            f"expected={FORMAL_TRAINING_POOL_MANIFEST_SHA256}, "
            f"actual={pool_info['manifest_sha256']}"
        )
    if pool_info["task_data_contract_sha256"] != FORMAL_TRAINING_TASK_CONTRACT_SHA256:
        raise ValueError("Training Pool task-data contract is not the reviewed contract.")
    baseline_info = _validate_training_integrity_baseline(pool, pool_info)
    pool_manifest = json.loads(
        Path(pool_info["manifest_path"]).read_text(encoding="utf-8")
    )
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
        "episode_seed_start": DEFAULT_TRAINING_SEED_START,
        "episode_seed_end": DEFAULT_TRAINING_SEED_START
        + FORMAL_TRAINING_EPISODE_COUNT
        - 1,
        "episode_count": FORMAL_TRAINING_EPISODE_COUNT,
        "episode_slots": FORMAL_TRAINING_EPISODE_SLOTS,
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
        expected_source_fields = _jsonable(pool_info["generation_config_fields"])
        expected_source_fields["service"]["generation"]["input_size_max_kb"] = 5000.0
        if not isinstance(source_fields, dict) or source_fields != expected_source_fields:
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
    for artifact_path in actual_paths:
        expected_hash = stored_artifact_hashes.get(artifact_path.name)
        if not isinstance(expected_hash, str) or _sha256(artifact_path) != expected_hash:
            raise ValueError(
                "Transformed episode artifact hash mismatch: "
                f"episode_seed={seed}, artifact={artifact_path.name}"
            )


def _episode_context(
    formal_root: Path,
    pool: Path,
    seed: int,
    pool_info: dict[str, Any],
) -> Any:
    episode_dir = pool / f"episode_{int(seed)}"
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
