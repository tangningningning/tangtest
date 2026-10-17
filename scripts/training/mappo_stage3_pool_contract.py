"""Reviewed Training-Pool provenance and aligned-input helpers for Stage3."""

from __future__ import annotations

import hashlib
import json
import subprocess
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
TRAIN_POOL_RELATIVE = Path("environment_data/proposed_joint_training")
VALIDATION_POOL_RELATIVE = Path("environment_data/proposed_joint_validation")
DEFAULT_VALIDATION_SEED = 30014
DEFAULT_TRAINING_SEED_START = 20001
DEFAULT_OUTPUT_PARENT = PROJECT_ROOT / "runs" / "research_mappo_stage1b"

# A single, reviewed compatibility profile for the frozen 500–1000 KB pools.
# Any other hash or field difference remains a hard failure.
LEGACY_POOL_CONFIG_REVISION = "df52747091cbd255f0dc65686d6e4d34656ba8e2"
LEGACY_POOL_CONFIG_SHA256 = "80762a71c870499eb9cce6aba330531495eac88783732c2ead1b045b2ba8f7c9"
LEGACY_POOL_SEMANTIC_SHA256 = "3ff3f083562393b6d66492ec2c43be9e5e8ff0bd9ecf425823ccac0ae40f533e"
CURRENT_FORMAL_CONFIG_SHA256 = "03328b88658a6f15c0518eee9f4903d5647f12a800432e58c74b42fdef51d80c"
CURRENT_FORMAL_SEMANTIC_SHA256 = "8a95bd7925093f523bc9d6d38118c54fed240d3aab0dce3d031686cd5b94d2b7"
CURRENT_GENERATION_FINGERPRINT = "4083b917948cd7203972c0edd5b40d2efefd8984dde27089a77da43d460b32f3"
SOURCE_CONFIG_REVISION = "eaa44d9169aa12c88012c8ad74a2c342098edfae"
SOURCE_CONFIG_SHA256 = "684bdfa4dc200db6227b2b00a7718797a98bf658e54d2d4dbef6f7d1f5babf77"
SOURCE_SEMANTIC_SHA256 = "10f893fc549c4016bedb1a5389ad996f9a3ff646aa012596a7230c284b13f542"
SOURCE_GENERATION_FINGERPRINT = "cfb3ee9fae004108af2d6bcb641609142c305d3ad6cc3ca74be13faaa6ff9865"
SOURCE_CATALOG_SHA256 = "3963eaa1c4cb116563fbbb9de0329dcc8c256c64fd849d5e0b75e9e51ca4c3ed"
TARGET_CATALOG_SHA256 = "6e8c497ed06d76e1251720ebc926e975a6a5c079fd224acab95ea5fb4799cf13"
VALIDATION_SOURCE_GENERATION_FINGERPRINT = "9ad3c6179be6cea67a0c95695f0cc2802e7570274c9be5446e51d996d5f7045d"
VALIDATION_SOURCE_CATALOG_PATH = "data/generated/service_catalog.json"
VALIDATION_SOURCE_CATALOG_SHA256 = "3adb64f9724a429a5cb6bf9c6eb609a54120d9918feaab1dd0475034af22e402"
VALIDATION_SOURCE_SPLIT_SHA256 = "3ecc408fb6f1dba72f9a9d1575d51c8e308710e26f5504e81a5ea7824290c331"
VALIDATION_SPLIT_MANIFEST_SHA256 = "70e6aea73efb23c929cecd2372d9c155b4aebea25c3bcfb59750c742f686f26d"
VALIDATION_SOURCE_CONFIG_SHA256 = "09783c6b360443f18da1c96209aebf925cc080a429750350e1c6ea97471e1e5b"
VALIDATION_SOURCE_CONFIG_SHA256_AT_START = "759d42cc2b303339e5f71dded2f40a9d4858e3a1878b66104d689081644a0cb9"
LEGACY_POOL_FIELD_DIFFERENCES = {
    "algorithm.ppo.policy_schema": (
        "adaptive_dual_timescale_proposed_cache_economic_features_fixed_lr_horizon_mask_v4_multi_node_cache_factorized_v1",
        "adaptive_dual_timescale_proposed_cache_economic_features_fixed_lr_horizon_mask_v4_multi_node_cache_factorized_v1_per_slot_analytic_task_model_v1_training_p95_reward_norm_v1",
        "METADATA_ONLY",
    ),
    "algorithm.ppo.task_execution_model": (
        None,
        "PER_SLOT_ANALYTIC",
        "RUNTIME_ONLY",
    ),
    "objective.E_ref": (3.06289851662, 5.1574642506299275, "RUNTIME_ONLY"),
    "objective.T_ref": (30.6289851662, 30.473728355325758, "RUNTIME_ONLY"),
}
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


def _flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_config(value, path))
        else:
            flattened[path] = value
    return flattened


def _git_config_bytes(formal_root: Path, revision: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{revision}:{CONFIG_RELATIVE.as_posix()}"],
        cwd=formal_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(
            f"Unable to read pinned configuration {revision}:{CONFIG_RELATIVE}: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return result.stdout


def _validate_pool_config_compatibility(
    formal_root: Path,
    config_path: Path,
    manifest: dict[str, Any],
    *,
    config_hash: str,
    semantic_hash: str,
    generation_fingerprint: str,
) -> dict[str, Any]:
    pool_raw_hash = manifest.get("config_sha256")
    pool_semantic_hash = manifest.get("semantic_config_sha256")
    if (pool_raw_hash, pool_semantic_hash) == (config_hash, semantic_hash):
        return {
            "policy": "EXACT_CONFIG_MATCH",
            "pool_config_sha256": pool_raw_hash,
            "pool_semantic_config_sha256": pool_semantic_hash,
            "runtime_config_sha256": config_hash,
            "runtime_semantic_config_sha256": semantic_hash,
            "approved_field_differences": [],
        }

    expected_pool_pair = (LEGACY_POOL_CONFIG_SHA256, LEGACY_POOL_SEMANTIC_SHA256)
    expected_runtime_pair = (CURRENT_FORMAL_CONFIG_SHA256, CURRENT_FORMAL_SEMANTIC_SHA256)
    if (pool_raw_hash, pool_semantic_hash) != expected_pool_pair:
        raise ValueError(
            "Unrecognized Training-Pool configuration provenance: "
            f"pool=({pool_raw_hash!r}, {pool_semantic_hash!r}), "
            f"runtime=({config_hash!r}, {semantic_hash!r})"
        )
    if (config_hash, semantic_hash) != expected_runtime_pair:
        raise ValueError(
            "Reviewed legacy compatibility applies only to the pinned current "
            f"formal config; actual=({config_hash!r}, {semantic_hash!r})"
        )

    historical_bytes = _git_config_bytes(formal_root, LEGACY_POOL_CONFIG_REVISION)
    historical_hash = hashlib.sha256(historical_bytes).hexdigest()
    if historical_hash != LEGACY_POOL_CONFIG_SHA256:
        raise ValueError(
            "Pinned historical config bytes do not match the Training-Pool raw hash: "
            f"git={historical_hash}, manifest={LEGACY_POOL_CONFIG_SHA256}"
        )
    historical_config = tomllib.loads(historical_bytes.decode("utf-8"))
    runtime_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    historical_flat = _flatten_config(historical_config)
    runtime_flat = _flatten_config(runtime_config)
    missing = object()
    observed_differences = {
        path: (historical_flat.get(path, missing), runtime_flat.get(path, missing))
        for path in sorted(set(historical_flat) | set(runtime_flat))
        if historical_flat.get(path, missing) != runtime_flat.get(path, missing)
    }
    expected_differences = {
        path: (old_value if old_value is not None else missing, new_value)
        for path, (old_value, new_value, _category) in LEGACY_POOL_FIELD_DIFFERENCES.items()
    }
    if observed_differences != expected_differences:
        printable = {
            path: ["<MISSING>" if before is missing else before,
                   "<MISSING>" if after is missing else after]
            for path, (before, after) in observed_differences.items()
        }
        raise ValueError(
            "Legacy configuration has an unapproved field difference: "
            f"{json.dumps(printable, ensure_ascii=False, sort_keys=True)}"
        )

    if generation_fingerprint != CURRENT_GENERATION_FINGERPRINT or manifest.get(
        "target_generation_config_fingerprint"
    ) != CURRENT_GENERATION_FINGERPRINT:
        raise ValueError(
            "Legacy compatibility requires the exact frozen generation fingerprint: "
            f"pool={manifest.get('target_generation_config_fingerprint')!r}, "
            f"runtime={generation_fingerprint!r}"
        )

    source_provenance = manifest.get("source_provenance")
    split = manifest.get("split")
    source_contract_identity: dict[str, Any]
    if split == "train":
        source_config_bytes = _git_config_bytes(formal_root, SOURCE_CONFIG_REVISION)
        source_config_hash = hashlib.sha256(source_config_bytes).hexdigest()
        if source_config_hash != SOURCE_CONFIG_SHA256:
            raise ValueError(
                "Pinned source-generation config does not match its recorded raw hash: "
                f"git={source_config_hash}, expected={SOURCE_CONFIG_SHA256}"
            )
        expected_source_fields = {
            "source_pool_manifest_sha256": "c9c401dfe6c36b11faecfa080b86cb878d325c2799ee7e54435ad85365992d93",
            "source_integrity_baseline_sha256": "f25594d7da6fac36af2f8e7ce00b3f124e16e0fcc5d711d691ac99e5d0337b94",
            "source_service_catalog_sha256": SOURCE_CATALOG_SHA256,
            "canonical_old_service_catalog_path": (
                "environment_data/stage6_candidate_feasibility_25_20_12_reduced/"
                "service_catalog_500_2000.json"
            ),
            "canonical_old_service_catalog_sha256": SOURCE_CATALOG_SHA256,
        }
        expected_source_status = "HISTORICAL_GENERATION_ATTESTATION_UNAVAILABLE_SOURCE_SNAPSHOT_HASHED"
        source_catalog_relative = (
            "environment_data/stage6_candidate_feasibility_25_20_12_reduced/"
            "service_catalog_500_2000.json"
        )
        expected_source_catalog_hash = SOURCE_CATALOG_SHA256
        source_contract_identity = {
            "source_config_revision": SOURCE_CONFIG_REVISION,
            "source_config_sha256": source_config_hash,
            "source_semantic_config_sha256": SOURCE_SEMANTIC_SHA256,
            "source_generation_config_fingerprint": SOURCE_GENERATION_FINGERPRINT,
        }
    elif split == "validation":
        expected_source_fields = {
            "source_data_split_manifest_sha256": VALIDATION_SOURCE_SPLIT_SHA256,
            "source_service_catalog_path": VALIDATION_SOURCE_CATALOG_PATH,
            "source_service_catalog_sha256": VALIDATION_SOURCE_CATALOG_SHA256,
            "canonical_old_service_catalog_path": (
                "environment_data/stage6_candidate_feasibility_25_20_12_reduced/"
                "service_catalog_500_2000.json"
            ),
            "canonical_old_service_catalog_sha256": SOURCE_CATALOG_SHA256,
            "source_input_range_kb": [500.0, 5000.0],
        }
        expected_source_status = "SOURCE_GENERATION_MANIFESTS_AND_ARTIFACT_HASHES_VERIFIED"
        source_catalog_relative = VALIDATION_SOURCE_CATALOG_PATH
        expected_source_catalog_hash = VALIDATION_SOURCE_CATALOG_SHA256
        split_manifest_path = formal_root / "environment_data" / "data_split_manifest.json"
        if (
            not split_manifest_path.is_file()
            or _sha256(split_manifest_path) != VALIDATION_SPLIT_MANIFEST_SHA256
            or manifest.get("data_split_manifest_sha256")
            != VALIDATION_SPLIT_MANIFEST_SHA256
        ):
            raise ValueError("Validation source data-split manifest is missing or changed.")
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        split_source = split_manifest.get("source_generation_provenance", {})
        if (
            split_source.get("source_split_manifest_sha256")
            != VALIDATION_SOURCE_SPLIT_SHA256
            or split_source.get("source_generation_config_fingerprint")
            != VALIDATION_SOURCE_GENERATION_FINGERPRINT
            or split_source.get("source_service_catalog_sha256")
            != VALIDATION_SOURCE_CATALOG_SHA256
            or split_source.get("source_generator_config_sha256")
            != VALIDATION_SOURCE_CONFIG_SHA256
            or split_source.get("source_generator_config_sha256_at_generation_start")
            != VALIDATION_SOURCE_CONFIG_SHA256_AT_START
        ):
            raise ValueError("Validation source generation contract is not the reviewed snapshot.")
        source_contract_identity = {
            "source_data_split_manifest_sha256": VALIDATION_SOURCE_SPLIT_SHA256,
            "source_split_manifest_sha256": VALIDATION_SOURCE_SPLIT_SHA256,
            "source_generation_config_fingerprint": VALIDATION_SOURCE_GENERATION_FINGERPRINT,
            "source_config_sha256": VALIDATION_SOURCE_CONFIG_SHA256,
            "source_config_sha256_at_generation_start": VALIDATION_SOURCE_CONFIG_SHA256_AT_START,
        }
    else:
        raise ValueError(f"Unsupported source compatibility split: {split!r}")
    if not isinstance(source_provenance, dict) or any(
        source_provenance.get(key) != value
        for key, value in expected_source_fields.items()
    ):
        raise ValueError("Pool source snapshot metadata is not the reviewed contract.")
    if source_provenance.get("historical_source_status") != expected_source_status:
        raise ValueError(
            "Pool historical source status is not the reviewed split contract: "
            f"split={split!r}, "
            f"status={source_provenance.get('historical_source_status')!r}"
        )

    source_catalog_path = formal_root / Path(source_catalog_relative)
    if (
        not source_catalog_path.is_file()
        or _sha256(source_catalog_path) != expected_source_catalog_hash
    ):
        raise ValueError("The recorded canonical source service catalog is missing or changed.")

    return {
        "policy": "APPROVED_LEGACY_RUNTIME_COMPATIBILITY_V1",
        "legacy_config_revision": LEGACY_POOL_CONFIG_REVISION,
        "legacy_config_sha256": pool_raw_hash,
        "legacy_semantic_config_sha256": pool_semantic_hash,
        "runtime_config_sha256": config_hash,
        "runtime_semantic_config_sha256": semantic_hash,
        **source_contract_identity,
        "approved_field_differences": [
            {"path": path, "category": category}
            for path, (_old_value, _new_value, category)
            in LEGACY_POOL_FIELD_DIFFERENCES.items()
        ],
    }


def _validate_pool(
    formal_root: Path,
    relative_pool: Path,
    *,
    expected_split: str,
    expected_seed_start: int,
    expected_episode_count: int,
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
        "slots_per_episode": 300,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Formal {expected_split} pool manifest mismatch for {key}: "
                f"expected={value!r}, actual={manifest.get(key)!r}"
            )
    config_path = formal_root / CONFIG_RELATIVE
    config_hash = project_config.raw_config_sha256(config_path)
    cfg = project_config.parse_toml_file(config_path)
    cfg["_meta"] = {
        "config_path": str(config_path.resolve()),
        "project_root": str(formal_root),
    }
    project_config.validate_for_formal_experiment(cfg)
    semantic_hash = project_config.semantic_config_hash(cfg)
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
        formal_root,
        config_path,
        manifest,
        config_hash=config_hash,
        semantic_hash=semantic_hash,
        generation_fingerprint=generation_fingerprint,
    )
    expected_metadata = {
        "target_generation_config_fingerprint": generation_fingerprint,
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
        "runtime_config_sha256": config_hash,
        "runtime_semantic_config_sha256": semantic_hash,
        "config_compatibility": config_compatibility,
        "generation_config_fingerprint": generation_fingerprint,
        "task_data_contract": task_contract,
        "task_data_contract_sha256": task_contract_hash,
        "generation_config_fields": generate_episode_realization._generation_config_payload(cfg),
        "service_catalog_sha256": catalog_hash,
        "task_data_model": manifest["task_data_model"],
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
            "generation_config_fingerprint"
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
