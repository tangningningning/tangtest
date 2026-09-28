#!/usr/bin/env python
"""Trace a retained Cache candidate reward difference to slot execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common import project_config
from decision import cache_configuration, task_allocation
from ppo_simple import cache_credit
from ppo_simple.trainer import (
    GENERATOR_EXTRA_ARTIFACT_FILENAMES,
    PPOV2Trainer,
    REALIZATION_ARTIFACT_FILENAMES,
    _assert_realization_context_compatible,
    _config_hash,
    _ensure_episode_realization,
    _load_single_episode_generator,
    _run_with_isolated_rng,
    aligned_train_steps,
    load_environment_context,
)
from scenario.realization import derive_episode_seeds, derive_namespaced_seed


DIAGNOSTIC_NAMESPACE = "ppo_candidate_q_integrity"
ATOL = 1.0e-8
RTOL = 1.0e-10


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _digest(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise FloatingPointError(f"{name} is non-finite: {number!r}")
    return number


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _assert_all_finite_json(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_all_finite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_all_finite_json(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"non-finite JSON number at {path}")


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve(strict=False) if value.is_absolute() else (ROOT / value).resolve(strict=False)


def _provenance(source: dict[str, Any]) -> dict[str, Any]:
    provenance = source.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("result JSON does not contain provenance")
    aliases = {
        "CURRENT_HEAD": ("current_head", "CURRENT_HEAD"),
        "DIAGNOSTIC_REALIZATION_SEED": ("diagnostic_realization_seed", "DIAGNOSTIC_REALIZATION_SEED"),
        "DIAGNOSTIC_GENERATION_RNG_SEED": ("diagnostic_generation_rng_seed", "DIAGNOSTIC_GENERATION_RNG_SEED"),
        "DIAGNOSTIC_ROLLOUT_RNG_SEED": ("diagnostic_rollout_rng_seed", "DIAGNOSTIC_ROLLOUT_RNG_SEED"),
        "TRAINER_INITIALIZATION_RNG_SEED": ("trainer_initialization_rng_seed", "TRAINER_INITIALIZATION_RNG_SEED"),
        "CONFIG_SHA256": ("config_sha256", "CONFIG_SHA256"),
        "DIAGNOSTIC_NAMESPACE": ("diagnostic_namespace", "DIAGNOSTIC_NAMESPACE"),
        "DATA_PROTOCOL": ("data_protocol", "DATA_PROTOCOL"),
        "EPISODE_SLOTS": ("episode_slots", "EPISODE_SLOTS"),
    }
    result: dict[str, Any] = {}
    for output_name, names in aliases.items():
        for name in names:
            if name in provenance:
                result[output_name] = provenance[name]
                break
        if output_name not in result:
            raise ValueError(f"result JSON provenance is missing {output_name}")
    if str(result["DIAGNOSTIC_NAMESPACE"]) != DIAGNOSTIC_NAMESPACE:
        raise ValueError("saved diagnostic namespace differs from the integrity run")
    return result


def _verify_seeds_and_config(
    context: Any, config_file: Path, provenance: dict[str, Any]
) -> dict[str, int]:
    base_seed = int(project_config.get_config_value(context.cfg, "experiment.seed"))
    expected = {
        "DIAGNOSTIC_REALIZATION_SEED": derive_namespaced_seed(
            base_seed, DIAGNOSTIC_NAMESPACE, "episode_realization"
        ),
        "DIAGNOSTIC_GENERATION_RNG_SEED": derive_namespaced_seed(
            int(provenance["DIAGNOSTIC_REALIZATION_SEED"]),
            DIAGNOSTIC_NAMESPACE, "episode_generation",
        ),
        "DIAGNOSTIC_ROLLOUT_RNG_SEED": derive_namespaced_seed(
            int(provenance["DIAGNOSTIC_REALIZATION_SEED"]),
            DIAGNOSTIC_NAMESPACE, "rollout_collection",
        ),
        "TRAINER_INITIALIZATION_RNG_SEED": derive_namespaced_seed(
            base_seed, DIAGNOSTIC_NAMESPACE, "trainer_initialization"
        ),
    }
    if int(provenance["DIAGNOSTIC_REALIZATION_SEED"]) != expected["DIAGNOSTIC_REALIZATION_SEED"]:
        raise AssertionError("saved diagnostic realization seed does not match current experiment config")
    for key, value in expected.items():
        if int(provenance[key]) != int(value):
            raise AssertionError(f"saved {key} does not match deterministic seed derivation")
    if _config_hash(str(config_file)) != str(provenance["CONFIG_SHA256"]):
        raise AssertionError("current config SHA-256 differs from the retained diagnostic run")
    if str(provenance["DATA_PROTOCOL"]) != str(context.data_protocol):
        raise AssertionError("current data protocol differs from the retained diagnostic run")
    if int(provenance["EPISODE_SLOTS"]) != int(context.episode_slots):
        raise AssertionError("current episode slot count differs from the retained diagnostic run")
    formal_seeds = derive_episode_seeds(base_seed)
    return {key: int(value) for key, value in expected.items()} | {
        "FORMAL_EPISODE_SEED_STREAM_COUNT": len(formal_seeds),
    }


def _audit_existing_realizations(
    realization_seed: int, generation_rng_seed: int, config_sha256: str,
    required_files: list[str], configured_protocol: str, episode_slots: int,
    generation_config_fingerprint: str, source_hashes: dict[str, str],
    context: Any, generator_module: Any,
) -> dict[str, Any]:
    """Find complete TRAIN episodes without requiring optional episode manifests."""
    data_root = ROOT / "environment_data"
    split_path = data_root / "data_split_manifest.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"TRAIN/validation/test split manifest is missing: {split_path}")
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    pool_dirs = [
        path for path in sorted(data_root.iterdir())
        if path.is_dir() and (
            path.name == "proposed_joint_training"
            or path.name.startswith("proposed_joint_training_preexisting")
            or path.name in {"proposed_joint_validation", "proposed_joint_test"}
        )
    ]
    directories: list[dict[str, Any]] = []
    train_exact_matches = 0
    compatible_realization_paths: list[str] = []
    target_seed_records: list[dict[str, Any]] = []
    for pool in pool_dirs:
        split = (
            "TRAIN" if pool.name == "proposed_joint_training" or pool.name.startswith("proposed_joint_training_preexisting")
            else "VALIDATION" if pool.name == "proposed_joint_validation"
            else "TEST"
        )
        split_info = split_manifest.get("train" if split == "TRAIN" else split.lower(), {})
        episode_dirs = [
            path for path in sorted(pool.iterdir())
            if path.is_dir() and re.fullmatch(r"episode_\d+", path.name)
        ]
        if not episode_dirs and any((pool / name).exists() for name in required_files):
            episode_dirs = [pool]
        manifests_by_dir: dict[Path, Path] = {
            manifest.parent: manifest
            for manifest in pool.rglob("episode_manifest.json")
        }
        complete_count = 0
        readable_count = 0
        seed_matches = 0
        metadata_conflict_count = 0
        for episode_dir in episode_dirs:
            name_match = re.fullmatch(r"episode_(\d+)", episode_dir.name)
            folder_seed = int(name_match.group(1)) if name_match else None
            manifest_path = manifests_by_dir.get(episode_dir)
            manifest: dict[str, Any] = {}
            manifest_status = "ABSENT_OPTIONAL"
            if manifest_path is not None:
                try:
                    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        manifest = loaded
                        manifest_status = "READ"
                    else:
                        manifest_status = "INVALID_IGNORED"
                except (OSError, json.JSONDecodeError):
                    manifest_status = "INVALID_IGNORED"
            manifest_seed_value = manifest.get("episode_seed")
            try:
                manifest_seed = int(manifest_seed_value) if manifest_seed_value is not None else None
            except (TypeError, ValueError):
                manifest_seed = None
            seed_conflict = (
                folder_seed is not None and manifest_seed is not None
                and folder_seed != manifest_seed
            )
            episode_seed = manifest_seed if manifest_seed is not None else folder_seed
            seed_ok = episode_seed == int(realization_seed)
            seed_matches += int(seed_ok)
            missing = [name for name in required_files if not (episode_dir / name).is_file()]
            unreadable: list[str] = []
            for filename in required_files:
                file_path = episode_dir / filename
                if not file_path.is_file():
                    continue
                try:
                    with file_path.open("rb") as stream:
                        if not stream.read(1):
                            unreadable.append(filename)
                except OSError:
                    unreadable.append(filename)
            files_ok = not missing and not unreadable
            complete_count += int(not missing)
            readable_count += int(files_ok)

            stored_fingerprint = manifest.get("generation_config_fingerprint")
            fingerprint_source = "episode_manifest" if stored_fingerprint is not None else "data_split_manifest"
            if stored_fingerprint is None:
                stored_fingerprint = split_manifest.get("generation_config_fingerprint")
            config_conflicts = []
            if stored_fingerprint is not None and str(stored_fingerprint) != str(generation_config_fingerprint):
                config_conflicts.append("generation_config_fingerprint")
            run_config_hash_fields = (
                "run_config_sha256", "config_sha256", "full_config_sha256",
            )
            for field in run_config_hash_fields:
                stored_hash = manifest.get(field)
                if stored_hash is not None and str(stored_hash) != str(config_sha256):
                    config_conflicts.append(field)
            for field, expected_hash in source_hashes.items():
                stored_hash = manifest.get(field, split_manifest.get(field))
                if stored_hash is not None and str(stored_hash) != str(expected_hash):
                    config_conflicts.append(field)
            direct_protocol = manifest.get(
                "data_protocol",
                manifest.get("ppo_data_protocol", manifest.get("training_data_protocol")),
            )
            if direct_protocol is not None and str(direct_protocol) != str(configured_protocol):
                config_conflicts.append("data_protocol")
            declared_split = manifest.get("split", manifest.get("data_split"))
            split_conflict = declared_split is not None and str(declared_split).upper() != split
            if split_conflict:
                config_conflicts.append("split")
            split_slots = manifest.get("episode_slots")
            if split_slots is None and isinstance(manifest.get("simulation"), dict):
                split_slots = manifest["simulation"].get("episode_slots")
            if split_slots is None:
                split_slots = split_manifest.get("slots_per_realization")
            try:
                slots_ok = split_slots is not None and int(split_slots) == int(episode_slots)
            except (TypeError, ValueError):
                slots_ok = False
            if not slots_ok:
                config_conflicts.append("episode_slots")

            seed_range_ok = True
            if split == "TRAIN" and episode_seed is not None:
                seed_start = split_info.get("seed_start")
                seed_end = split_info.get("seed_end")
                if seed_start is not None and seed_end is not None:
                    seed_range_ok = int(seed_start) <= episode_seed <= int(seed_end)
            # If a generation RNG seed was persisted, a mismatch is explicit;
            # its absence is allowed because the split pool predates that field.
            stored_generation_seed = manifest.get(
                "generation_rng_seed", manifest.get("generation_rng_seed_used")
            )
            try:
                generation_seed_conflict = (
                    stored_generation_seed is not None
                    and int(stored_generation_seed) != int(generation_rng_seed)
                )
            except (TypeError, ValueError):
                generation_seed_conflict = True
            artifact_hash_status = "NOT_PRESENT_OPTIONAL"
            if seed_ok and manifest.get("generated_artifact_sha256"):
                mismatches = []
                for filename, expected_hash in manifest["generated_artifact_sha256"].items():
                    expected_hash = expected_hash.get("sha256") if isinstance(expected_hash, dict) else expected_hash
                    target_file = episode_dir / str(filename)
                    actual_hash = hashlib.sha256(target_file.read_bytes()).hexdigest() if target_file.is_file() else None
                    if actual_hash != expected_hash:
                        mismatches.append(str(filename))
                artifact_hash_status = "PASS" if not mismatches else "FAIL"
            slot_validation_status = "NOT_NEEDED_SEED_MISMATCH"
            slot_validation_error = None
            if seed_ok and split == "TRAIN" and files_ok and slots_ok:
                try:
                    get = project_config.get_config_value
                    generator_module.validate_written_outputs(
                        episode_dir,
                        task_vehicle_count=int(get(context.cfg, "simulation.num_task_vehicles")),
                        research_vehicle_count=int(get(context.cfg, "simulation.num_research_vehicles")),
                        horizon_steps=int(get(context.cfg, "prediction.vehicle.horizon")),
                    )
                    slot_validation_status = "PASS"
                except Exception as exc:
                    slot_validation_status = "FAIL"
                    slot_validation_error = f"{type(exc).__name__}: {exc}"
            eligible = bool(
                split == "TRAIN" and seed_ok and seed_range_ok and files_ok and slots_ok
                and slot_validation_status == "PASS"
                and not seed_conflict and not config_conflicts and not generation_seed_conflict
                and artifact_hash_status != "FAIL"
            )
            if eligible:
                train_exact_matches += 1
                compatible_realization_paths.append(str(episode_dir.relative_to(ROOT)))
            metadata_conflict_count += int(bool(config_conflicts) or seed_conflict or generation_seed_conflict)
            if seed_ok:
                target_seed_records.append({
                    "relative_path": str(episode_dir.relative_to(ROOT)),
                    "split": split,
                    "episode_seed": episode_seed,
                    "episode_manifest_status": manifest_status,
                    "generation_rng_seed": stored_generation_seed,
                    "generation_rng_seed_status": "MATCH" if stored_generation_seed is not None and not generation_seed_conflict else "NOT_RECORDED_OPTIONAL",
                    "config_fingerprint_source": fingerprint_source,
                    "generation_config_fingerprint": stored_fingerprint,
                    "config_conflicts": config_conflicts,
                    "data_protocol_from_manifest": direct_protocol,
                    "declared_split_from_manifest": declared_split,
                    "episode_slots": split_slots,
                    "required_artifacts_missing": missing,
                    "required_artifacts_unreadable": unreadable,
                    "artifact_hash_check": artifact_hash_status,
                    "slot_data_validation": slot_validation_status,
                    "slot_data_validation_error": slot_validation_error,
                    "train_seed_range_match": seed_range_ok,
                    "eligible_for_reuse": eligible,
                })
        directories.append({
            "directory": pool.name,
            "split": split,
            "manifest_count": len(manifests_by_dir),
            "episode_directory_count": len(episode_dirs),
            "target_episode_seed_match_count": seed_matches,
            "complete_artifact_directory_count": complete_count,
            "readable_artifact_directory_count": readable_count,
            "configuration_or_seed_conflict_count": metadata_conflict_count,
            "split_manifest_assignment": _json_safe(split_info),
        })
    slots_match = int(split_manifest.get("slots_per_realization", -1)) == int(episode_slots)
    return {
        "target_realization_seed": int(realization_seed),
        "target_generation_rng_seed": int(generation_rng_seed),
        "target_config_sha256": str(config_sha256),
        "target_data_protocol": str(configured_protocol),
        "target_episode_slots": int(episode_slots),
        "current_generation_config_fingerprint": str(generation_config_fingerprint),
        "current_source_hashes": _json_safe(source_hashes),
        "data_split_manifest_path": str(split_path),
        "split_manifest_generator_config_sha256": split_manifest.get("generator_config_sha256"),
        "split_manifest_generator_config_sha256_matches_full_run_config": split_manifest.get("generator_config_sha256") == str(config_sha256),
        "split_manifest_generator_config_hash_scope": "generator_config_sha256 covers generation config; it is not the full PPO run config SHA-256",
        "split_manifest_generation_config_fingerprint": split_manifest.get("generation_config_fingerprint"),
        "split_manifest_slots_per_realization": int(split_manifest.get("slots_per_realization", -1)),
        "split_manifest_train": _json_safe(split_manifest.get("train")),
        "split_manifest_validation": _json_safe(split_manifest.get("validation")),
        "split_manifest_test": _json_safe(split_manifest.get("test")),
        "split_manifest_slot_count_matches": slots_match,
        "directories": directories,
        "compatible_train_realization_paths": compatible_realization_paths,
        "target_seed_candidate_records": target_seed_records,
        "train_exact_compatible_realization_count": train_exact_matches,
        "episode_manifest_policy": "OPTIONAL_PROVENANCE_ONLY",
        "REUSE_EXISTING_REALIZATION": "YES" if train_exact_matches else "NO",
        "GENERATE_NEW_REALIZATION": "NO" if train_exact_matches else "YES",
        "manifest_is_required_for_reuse": "NO",
        "validation_or_test_used": "NO",
    }


def _task_record(task: Any) -> dict[str, Any]:
    # These are the actual Task dataclass fields; no aliases are invented.
    return {
        "task_id": str(task.task_id),
        "vehicle_id": str(task.vehicle_id),
        "service_id": int(task.service_id),
        "input_bits": float(task.input_bits),
        "cpu_cycles": float(task.cpu_cycles),
    }


def _position_record(positions: dict[str, Any]) -> dict[str, list[float]]:
    return {
        str(key): np.asarray(value, dtype=np.float64).reshape(-1).tolist()
        for key, value in sorted(positions.items(), key=lambda item: str(item[0]))
    }


def _local_cache_record(local_cache: Any) -> dict[str, list[int]]:
    return {
        str(vehicle_id): sorted(int(service_id) for service_id in services)
        for vehicle_id, services in sorted(local_cache.items(), key=lambda item: str(item[0]))
    }


def _retained_candidates(
    source: dict[str, Any], transition_index: int, head_id: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = source.get("cache_q", {}).get("all_candidate_records")
    if not isinstance(rows, list):
        raise ValueError("result JSON lacks cache_q.all_candidate_records")
    selected = [
        row for row in rows
        if int(row.get("transition_index", -1)) == int(transition_index)
        and str(row.get("head_id", "")) == str(head_id)
    ]
    if not selected:
        raise ValueError(f"no retained candidate records for transition={transition_index}, head={head_id}")
    first = selected[0]
    for row in selected:
        if int(row["origin_index"]) != int(first["origin_index"]):
            raise AssertionError("retained Candidate rows disagree on origin_index")
        if float(row["keep_reward"]) != float(first["keep_reward"]):
            raise AssertionError("retained Candidate rows disagree on KEEP reward")
    exact_best = max(selected, key=lambda row: float(row["q"]))
    reward_best = max(selected, key=lambda row: float(row["reward"]))
    return selected, {"keep": first, "exact_q_best": exact_best, "reward_best": reward_best}


def _candidate_for(
    row: dict[str, Any], node_index: int, node_id: str, before_row: np.ndarray,
    proposal_source: str,
) -> cache_configuration.CacheTargetCandidate:
    return cache_configuration.CacheTargetCandidate(
        node_index=int(node_index), node_id=str(node_id),
        before_node_cache=np.asarray(before_row, dtype=np.int8).copy(),
        target_node_cache=np.asarray(row["target_node_cache"], dtype=np.int8).copy(),
        proposal_source=str(proposal_source),
    )


def _run_branch(
    trainer: Any, transition: Any, step: Any, node_index: int,
    working_cache: np.ndarray, previous_cache: np.ndarray,
    candidate: Any | None,
) -> dict[str, Any]:
    action_map = cache_credit._production_action_map(transition)
    input_signature = {
        "actual_task_set": [_task_record(task) for task in step.actual_tasks],
        "actual_vehicle_positions": _position_record(step.actual_vehicle_positions),
        "local_tv_cache": _local_cache_record(trainer.context.local_cache),
        "previous_cache_state": np.asarray(transition.previous_cache_state, dtype=np.int8).astype(int).tolist(),
        "previous_uav_positions": _position_record(transition.previous_uav_positions),
        "other_cache_actions": [
            _json_safe(action_map[index])
            for index in sorted(action_map) if index != node_index
        ],
        "uav_actions": _json_safe(list(transition.uav_conditional_actions)),
    }
    result = cache_credit._execute_cache_branch(
        trainer, transition, step, None, node_index,
        {"working_cache": working_cache, "previous_cache": previous_cache},
        candidate,
    )
    result["_input_signature"] = input_signature
    return result


def _branch_summary(name: str, branch: dict[str, Any], config: list[int]) -> dict[str, Any]:
    result = branch["result"]
    summary = dict(result.matching_summary)
    kkt_diagnostics = summary.get("KKT_SOLVE_DIAGNOSTICS", [])
    residual_names = (
        "KKT_RESOURCE_RESIDUAL_MAX",
        "KKT_STATIONARITY_RESIDUAL_MAX",
        "KKT_COMPLEMENTARY_SLACKNESS_RESIDUAL_MAX",
    )
    kkt = {
        "KKT_SOLVER_CALL_COUNT": int(summary.get("KKT_SOLVER_CALL_COUNT", 0)),
        "KKT_SOLVE_COUNT": int(summary.get("KKT_SOLVE_COUNT", 0)),
        "KKT_CONVERGED_COUNT": int(summary.get("KKT_CONVERGED_COUNT", 0)),
        "KKT_MAX_ITER_HIT_COUNT": int(summary.get("KKT_MAX_ITER_HIT_COUNT", 0)),
        "KKT_FINITE_COUNT": int(summary.get("KKT_FINITE_COUNT", 0)),
        "KKT_RESOURCE_RESIDUAL_MAX": _float_or_none(summary.get(residual_names[0])),
        "KKT_STATIONARITY_RESIDUAL_MAX": _float_or_none(summary.get(residual_names[1])),
        "KKT_COMPLEMENTARY_SLACKNESS_RESIDUAL_MAX": _float_or_none(summary.get(residual_names[2])),
        "KKT_SOLVE_DIAGNOSTICS": _json_safe(kkt_diagnostics),
    }
    solve_count = kkt["KKT_SOLVE_COUNT"]
    bad_kkt_record = any(
        not bool(record.get("finite", False))
        or not bool(record.get("converged_by_tolerance", False))
        or bool(record.get("hit_max_bisection_iterations", False))
        for record in kkt_diagnostics
    )
    kkt_concern = bool(
        bad_kkt_record
        or kkt["KKT_FINITE_COUNT"] != solve_count
        or kkt["KKT_CONVERGED_COUNT"] != solve_count
        or kkt["KKT_MAX_ITER_HIT_COUNT"] > 0
    )
    stable = bool(summary.get("transfer_exchange_stable", False)) and bool(
        summary.get("stable_check_passed", False)
    ) and str(summary.get("stop_reason", "")) != "stopped_by_max_updates"
    numeric = {
        "REWARD": _finite(branch["reward"], f"{name}.reward"),
        "SYSTEM_UTILITY": _finite(result.system_utility, f"{name}.system_utility"),
        "ACTUAL_TASK_UTILITY": _finite(result.actual_task_utility, f"{name}.actual_task_utility"),
        "CACHE_COST": _finite(result.cache_cost, f"{name}.cache_cost"),
        "MOVEMENT_ENERGY": _finite(result.movement_energy, f"{name}.movement_energy"),
        "MOVEMENT_COST": _finite(result.movement_cost, f"{name}.movement_cost"),
    }
    identity_error = abs(
        numeric["SYSTEM_UTILITY"] - numeric["ACTUAL_TASK_UTILITY"]
        + numeric["CACHE_COST"] + numeric["MOVEMENT_COST"]
    )
    task_utility_sum = math.fsum(float(value) for value in result.per_task_utility.values())
    task_utility_error = abs(task_utility_sum - numeric["ACTUAL_TASK_UTILITY"])
    return {
        "BRANCH": name,
        "CONFIG": [int(value) for value in config],
        "STEP_FEASIBLE": bool(result.step_feasible),
        "INFEASIBLE_REASON": str(result.infeasible_reason),
        **numeric,
        "SYSTEM_UTILITY_RECONSTRUCTION_ERROR": identity_error,
        "PER_TASK_UTILITY_SUM": task_utility_sum,
        "PER_TASK_UTILITY_RECONSTRUCTION_ERROR": task_utility_error,
        "MATCHING": {str(key): str(value) for key, value in result.matching.items()},
        "CPU_ALLOCATIONS": _json_safe(result.cpu_allocations),
        "BANDWIDTH_ALLOCATIONS": _json_safe(result.bandwidth_allocations),
        "PER_TASK_DELAY_S": _json_safe(result.per_task_delay_s),
        "PER_TASK_VEHICLE_ENERGY_J": _json_safe(result.per_task_vehicle_energy_j),
        "PER_TASK_UTILITY": _json_safe(result.per_task_utility),
        "PREVIOUS_CACHE_STATE": np.asarray(result.previous_cache_state, dtype=np.int8).astype(int).tolist(),
        "FINAL_CACHE_STATE": np.asarray(result.final_cache_state, dtype=np.int8).astype(int).tolist(),
        "PREVIOUS_UAV_POSITIONS": _position_record(result.previous_uav_positions),
        "FINAL_UAV_POSITIONS": _position_record(result.final_uav_positions),
        "INPUT_CACHE_TRIGGERS": list(result.input_cache_triggers),
        "INPUT_UAV_TRIGGERS": _json_safe(result.input_uav_triggers),
        "CONFIGURATION_SUMMARY": _json_safe(result.configuration_summary),
        "MATCHING_SUMMARY": {
            key: _json_safe(summary.get(key))
            for key in (
                "stop_reason", "transfer_exchange_stable", "stable_check_passed",
                "max_remaining_delta", "updates", "transfer_updates", "exchange_updates",
                "candidate_evaluations_total", "full_evaluator_calls",
            )
        },
        **kkt,
        "MATCHING_STABLE": stable,
        "KKT_EXECUTION_CONCERN": kkt_concern,
    }


def _task_comparison(
    tasks: list[Any], keep: dict[str, Any], best: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    keep_result = keep["result"]
    best_result = best["result"]
    rows: list[dict[str, Any]] = []
    transitions: Counter[str] = Counter()
    keep_counts: Counter[str] = Counter()
    best_counts: Counter[str] = Counter()
    changed_count = 0
    for task in tasks:
        task_id = str(task.task_id)
        keep_node = str(keep_result.matching[task_id])
        best_node = str(best_result.matching[task_id])
        keep_counts[keep_node] += 1
        best_counts[best_node] += 1
        transitions[f"{keep_node} -> {best_node}"] += 1
        assignment_changed = keep_node != best_node
        changed_count += int(assignment_changed)
        keep_delay = _finite(keep_result.per_task_delay_s[task_id], f"KEEP delay {task_id}")
        best_delay = _finite(best_result.per_task_delay_s[task_id], f"BEST delay {task_id}")
        keep_energy = _finite(keep_result.per_task_vehicle_energy_j[task_id], f"KEEP energy {task_id}")
        best_energy = _finite(best_result.per_task_vehicle_energy_j[task_id], f"BEST energy {task_id}")
        keep_utility = _finite(keep_result.per_task_utility[task_id], f"KEEP utility {task_id}")
        best_utility = _finite(best_result.per_task_utility[task_id], f"BEST utility {task_id}")
        rows.append({
            **_task_record(task),
            "KEEP_matching_node": keep_node,
            "BEST_matching_node": best_node,
            "MATCHING_CHANGED": assignment_changed,
            "KEEP_delay_s": keep_delay,
            "BEST_delay_s": best_delay,
            "DELTA_delay_s": best_delay - keep_delay,
            "KEEP_vehicle_energy_j": keep_energy,
            "BEST_vehicle_energy_j": best_energy,
            "DELTA_vehicle_energy_j": best_energy - keep_energy,
            "KEEP_task_utility": keep_utility,
            "BEST_task_utility": best_utility,
            "DELTA_task_utility": best_utility - keep_utility,
            "KEEP_cpu_allocation": _float_or_none(keep_result.cpu_allocations.get(keep_node, {}).get(task_id)),
            "BEST_cpu_allocation": _float_or_none(best_result.cpu_allocations.get(best_node, {}).get(task_id)),
            "KEEP_bandwidth_allocation": _float_or_none(keep_result.bandwidth_allocations.get(keep_node, {}).get(task_id)),
            "BEST_bandwidth_allocation": _float_or_none(best_result.bandwidth_allocations.get(best_node, {}).get(task_id)),
        })
    rows.sort(key=lambda item: abs(float(item["DELTA_task_utility"])), reverse=True)
    all_abs = math.fsum(abs(float(row["DELTA_task_utility"])) for row in rows)
    shares = {}
    for count in (1, 5, 10):
        numerator = math.fsum(abs(float(row["DELTA_task_utility"])) for row in rows[:count])
        shares[f"TOP{count}_ABS_UTILITY_DELTA_SHARE"] = numerator / all_abs if all_abs > 0.0 else 0.0
    return rows, {
        "TASK_COUNT": len(rows),
        "MATCHING_CHANGED_TASK_COUNT": changed_count,
        "MATCHING_UNCHANGED_TASK_COUNT": len(rows) - changed_count,
        "KEEP_NODE_ASSIGNMENT_COUNTS": dict(sorted(keep_counts.items())),
        "EXACT_BEST_NODE_ASSIGNMENT_COUNTS": dict(sorted(best_counts.items())),
        "KEEP_CLOUD_TASK_COUNT": int(keep_counts.get(task_allocation.CLOUD, 0)),
        "EXACT_BEST_CLOUD_TASK_COUNT": int(best_counts.get(task_allocation.CLOUD, 0)),
        "KEEP_TO_BEST_NODE_TRANSITION_COUNTS": dict(sorted(transitions.items())),
        **shares,
    }


def _service_feasibility(
    tasks: list[Any], task_rows: list[dict[str, Any]], keep: dict[str, Any],
    best: dict[str, Any], context: Any, step: Any,
) -> list[dict[str, Any]]:
    changed_ids = {row["task_id"] for row in task_rows if row["MATCHING_CHANGED"]}
    if not changed_ids:
        return []
    actual_positions = {
        str(key): np.asarray(value, dtype=np.float64).copy()
        for key, value in step.actual_vehicle_positions.items()
    }
    cache_nodes = context.cache_nodes
    tasks_by_id = {str(task.task_id): task for task in tasks}
    out: list[dict[str, Any]] = []
    for label, branch in (("KEEP", keep), ("EXACT_BEST", best)):
        result = branch["result"]
        final_cache = np.asarray(result.final_cache_state, dtype=np.int8)
        service_positions = task_allocation.build_service_node_positions(
            cache_nodes, actual_positions, result.final_uav_positions,
        )
        all_targets = task_allocation.build_feasible_targets(
            tasks, context.local_cache, final_cache, cache_nodes,
            actual_positions, service_positions, context.cache_cfg,
        )
        for task_id in sorted(changed_ids):
            task = tasks_by_id[task_id]
            targets = [str(value) for value in all_targets[task_id]]
            local = task_allocation.local_target(task_id)
            edge_nodes = sorted(
                value for value in targets
                if value != task_allocation.CLOUD and value != local
            )
            row = next(item for item in task_rows if item["task_id"] == task_id)
            row.setdefault("SERVICE_FEASIBILITY", {})[label] = {
                "service_id": int(task.service_id),
                "all_feasible_targets": sorted(targets),
                "edge_service_nodes": edge_nodes,
            }
    for task_id in sorted(changed_ids):
        task = tasks_by_id[task_id]
        row = next(item for item in task_rows if item["task_id"] == task_id)
        keep_targets = row["SERVICE_FEASIBILITY"]["KEEP"]["all_feasible_targets"]
        best_targets = row["SERVICE_FEASIBILITY"]["EXACT_BEST"]["all_feasible_targets"]
        out.append({
            "task_id": task_id,
            "vehicle_id": str(task.vehicle_id),
            "service_id": int(task.service_id),
            "SERVICE_FEASIBLE_NODE_SET_KEEP": row["SERVICE_FEASIBILITY"]["KEEP"]["edge_service_nodes"],
            "SERVICE_FEASIBLE_NODE_SET_BEST": row["SERVICE_FEASIBILITY"]["EXACT_BEST"]["edge_service_nodes"],
            "ALL_FEASIBLE_TARGETS_KEEP": keep_targets,
            "ALL_FEASIBLE_TARGETS_BEST": best_targets,
        })
    return out


def _isolation(
    names: list[str], branches: dict[str, dict[str, Any]], transition: Any,
    step: Any, context: Any, node_index: int, expected_rows: dict[str, list[int]],
) -> dict[str, Any]:
    results = {name: branches[name]["result"] for name in names}
    signatures = [branches[name]["_input_signature"] for name in names]
    first_signature = signatures[0]
    checks: dict[str, bool] = {}
    checks["SAME_ACTUAL_TASK_SET"] = all(
        sorted(result.matching) == sorted(task.task_id for task in step.actual_tasks)
        and sorted(result.per_task_utility) == sorted(task.task_id for task in step.actual_tasks)
        for result in results.values()
    ) and all(_digest(first_signature["actual_task_set"]) == _digest(item["actual_task_set"]) for item in signatures)
    checks["SAME_ACTUAL_VEHICLE_POSITIONS"] = all(
        _digest(first_signature["actual_vehicle_positions"]) == _digest(item["actual_vehicle_positions"]) for item in signatures
    )
    checks["SAME_LOCAL_TV_CACHE"] = all(
        _digest(first_signature["local_tv_cache"]) == _digest(item["local_tv_cache"]) for item in signatures
    )
    checks["SAME_PREVIOUS_CACHE_STATE"] = all(
        np.array_equal(np.asarray(result.previous_cache_state), np.asarray(results[names[0]].previous_cache_state))
        for result in results.values()
    ) and all(
        _digest(first_signature["previous_cache_state"]) == _digest(item["previous_cache_state"])
        for item in signatures
    )
    checks["SAME_PREVIOUS_UAV_POSITIONS"] = all(
        _digest(_position_record(result.previous_uav_positions))
        == _digest(_position_record(results[names[0]].previous_uav_positions))
        for result in results.values()
    ) and all(
        _digest(first_signature["previous_uav_positions"]) == _digest(item["previous_uav_positions"])
        for item in signatures
    )
    checks["SAME_OTHER_CACHE_ROWS"] = all(
        np.array_equal(
            np.asarray(result.final_cache_state)[np.arange(len(result.final_cache_state)) != node_index],
            np.asarray(results[names[0]].final_cache_state)[np.arange(len(result.final_cache_state)) != node_index],
        ) for result in results.values()
    )
    checks["SAME_OTHER_CACHE_ACTIONS"] = all(
        _digest(first_signature["other_cache_actions"]) == _digest(item["other_cache_actions"])
        for item in signatures
    )
    checks["SAME_UAV_ACTIONS"] = all(
        _digest(first_signature["uav_actions"]) == _digest(item["uav_actions"])
        for item in signatures
    )
    checks["SAME_FINAL_UAV_POSITIONS"] = all(
        _digest(_position_record(result.final_uav_positions))
        == _digest(_position_record(results[names[0]].final_uav_positions))
        for result in results.values()
    )
    checks["TARGET_CACHE_ROWS_MATCH_SELECTED_CONFIG"] = all(
        np.asarray(result.final_cache_state, dtype=np.int8)[node_index].astype(int).tolist()
        == expected_rows[name]
        for name, result in results.items()
    )
    return {
        "BRANCH_INPUT_ISOLATION_GATE": "PASS" if all(checks.values()) else "FAIL",
        **{key: "YES" if value else "NO" for key, value in checks.items()},
        "OTHER_CACHE_ACTIONS_SHA256": _digest(first_signature["other_cache_actions"]),
        "UAV_ACTIONS_SHA256": _digest(first_signature["uav_actions"]),
        "ACTUAL_TASK_SET_SHA256": _digest(first_signature["actual_task_set"]),
        "ACTUAL_VEHICLE_POSITIONS_SHA256": _digest(first_signature["actual_vehicle_positions"]),
        "LOCAL_TV_CACHE_SHA256": _digest(first_signature["local_tv_cache"]),
    }


def _markdown_report(data: dict[str, Any]) -> str:
    branches = data["BRANCHES"]
    keep = branches["KEEP"]
    exact = branches["EXACT_Q_BEST"]
    best_rows = data.get("TOP20_TASK_UTILITY_CHANGES", [])
    headers = (
        "Task", "Vehicle", "Service", "KEEP node", "Best node", "Δ delay (s)",
        "Δ energy (J)", "Δ utility",
    )
    lines = [
        f"## {data['TARGET_HEAD']} Transition-{data['TARGET_TRANSITION']} Reward Root-Cause Gate", "",
        f"- Baseline HEAD: `{data['CURRENT_HEAD']}`; transition/origin: `{data['TARGET_TRANSITION']}/{data['TARGET_ORIGIN_INDEX']}`; head: `{data['TARGET_HEAD']}`.",
        f"- Reproduction: **{data['REPRODUCTION_GATE']}**; KEEP `{keep['REWARD']:.12g}`; Exact-Q best `{exact['REWARD']:.12g}`; reward-best `{branches['REWARD_BEST']['REWARD']:.12g}` (same as Exact-Q best: {data['REWARD_BEST_SAME_AS_EXACT_Q_BEST']}).",
        f"- Branch input isolation: **{data['BRANCH_INPUT_ISOLATION_GATE']}**. Task set, vehicle positions, local cache, previous cache/UAV states, other cache rows/actions, UAV actions, and final UAV positions are separately recorded in the JSON artifact.",
        "",
        "### Reward identity and first-level decomposition", "",
        "| Branch | Task utility | Cache cost | Movement energy | Movement cost | System utility / reward |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, branch in branches.items():
        lines.append(
            f"| {name} | {branch['ACTUAL_TASK_UTILITY']:.9g} | {branch['CACHE_COST']:.9g} | {branch['MOVEMENT_ENERGY']:.9g} | {branch['MOVEMENT_COST']:.9g} | {branch['SYSTEM_UTILITY']:.9g} |"
        )
    lines += [
        "",
        f"Exact-Q best minus KEEP: Δ task utility `{data['DELTA_ACTUAL_TASK_UTILITY']:.12g}`, −Δ cache cost `{data['NEG_DELTA_CACHE_COST']:.12g}`, −Δ movement cost `{data['NEG_DELTA_MOVEMENT_COST']:.12g}`, Δ system utility `{data['DELTA_SYSTEM_UTILITY']:.12g}`. Maximum system identity error `{data['MAX_SYSTEM_UTILITY_RECONSTRUCTION_ERROR']:.3g}`; delta identity error `{data['DELTA_SYSTEM_UTILITY_RECONSTRUCTION_ERROR']:.3g}`; largest additive change: **{data['LARGEST_REWARD_COMPONENT_CHANGE']}**.",
        f"- Per-task utility sum checks: KEEP error `{keep['PER_TASK_UTILITY_RECONSTRUCTION_ERROR']:.3g}`, Exact-Q best error `{exact['PER_TASK_UTILITY_RECONSTRUCTION_ERROR']:.3g}`. Task count `{data['TASK_COUNT']}`; matching changed `{data['MATCHING_CHANGED_TASK_COUNT']}`, unchanged `{data['MATCHING_UNCHANGED_TASK_COUNT']}`.",
        "",
        "### Matching changes and largest task contributions", "",
        f"- KEEP assignments: `{json.dumps(data['KEEP_NODE_ASSIGNMENT_COUNTS'], ensure_ascii=False, sort_keys=True)}`; Exact-Q best: `{json.dumps(data['EXACT_BEST_NODE_ASSIGNMENT_COUNTS'], ensure_ascii=False, sort_keys=True)}`.",
        f"- Cloud task counts (production `task_allocation.CLOUD`): KEEP `{data['KEEP_CLOUD_TASK_COUNT']}`, Exact-Q best `{data['EXACT_BEST_CLOUD_TASK_COUNT']}`.",
        f"- KEEP → best node transitions: `{json.dumps(data['KEEP_TO_BEST_NODE_TRANSITION_COUNTS'], ensure_ascii=False, sort_keys=True)}`.",
        "",
        "| Task | Vehicle | Service | KEEP node | Best node | Δ delay (s) | Δ energy (J) | Δ utility |",
        "|---|---|---:|---|---|---:|---:|---:|",
    ]
    for row in best_rows:
        lines.append(
            f"| {row['task_id']} | {row['vehicle_id']} | {row['service_id']} | {row['KEEP_matching_node']} | {row['BEST_matching_node']} | {row['DELTA_delay_s']:.6g} | {row['DELTA_vehicle_energy_j']:.6g} | {row['DELTA_task_utility']:.6g} |"
        )
    lines += [
        "",
        f"Absolute task utility delta shares: Top 1 `{data['TOP1_ABS_UTILITY_DELTA_SHARE']:.6g}`, Top 5 `{data['TOP5_ABS_UTILITY_DELTA_SHARE']:.6g}`, Top 10 `{data['TOP10_ABS_UTILITY_DELTA_SHARE']:.6g}`.",
        "",
        "### Service availability, Matching, and KKT", "",
        f"- {data['TARGET_HEAD']} KEEP cache row: `{data['TARGET_KEEP_CACHE_ROW']}`; Exact-Q best: `{data['TARGET_EXACT_BEST_CACHE_ROW']}`; added service IDs: `{data['ADDED_SERVICES']}`; removed service IDs: `{data['REMOVED_SERVICES']}`.",
    ]
    for item in data.get("CHANGED_TASK_SERVICE_FEASIBILITY", []):
        lines.append(
            f"- Task `{item['task_id']}` service `{item['service_id']}`: edge nodes KEEP `{item['SERVICE_FEASIBLE_NODE_SET_KEEP']}` → best `{item['SERVICE_FEASIBLE_NODE_SET_BEST']}`; all feasible targets KEEP `{item['ALL_FEASIBLE_TARGETS_KEEP']}` → best `{item['ALL_FEASIBLE_TARGETS_BEST']}`."
        )
    for label, branch in (("KEEP", keep), ("Exact-Q best", exact)):
        ms = branch["MATCHING_SUMMARY"]
        lines.append(
            f"- {label} Matching: stop `{ms['stop_reason']}`, stable `{branch['MATCHING_STABLE']}`, max remaining Δ `{ms['max_remaining_delta']}`, updates `{ms['updates']}` (transfer `{ms['transfer_updates']}`, exchange `{ms['exchange_updates']}`), candidates `{ms['candidate_evaluations_total']}`, full evaluator calls `{ms['full_evaluator_calls']}`."
        )
        lines.append(
            f"- {label} KKT: calls `{branch['KKT_SOLVER_CALL_COUNT']}`, solves `{branch['KKT_SOLVE_COUNT']}`, converged `{branch['KKT_CONVERGED_COUNT']}`, max-iteration hits `{branch['KKT_MAX_ITER_HIT_COUNT']}`, finite `{branch['KKT_FINITE_COUNT']}`; residual max resource/stationarity/complementarity `{branch['KKT_RESOURCE_RESIDUAL_MAX']}` / `{branch['KKT_STATIONARITY_RESIDUAL_MAX']}` / `{branch['KKT_COMPLEMENTARY_SLACKNESS_RESIDUAL_MAX']}`."
        )
    lines += [
        "",
        "### Penalty and root-cause decision", "",
        f"- Explicit penalty found: **{data['EXPLICIT_PENALTY_FOUND']}**. Production task utility is `ω_T(1 − delay/T_ref) + ω_E(1 − vehicle_energy/E_ref)` from `src/common/system_model.py`; production system utility is `actual_task_utility − cache_cost − movement_cost` from `src/decision/slot_environment.py`. Infeasible branch execution raises before a reward is returned; Cloud is a feasible target/fallback, not a separate utility penalty.",
        f"- Classification: **{data['REWARD_OUTLIER_ROOT_CAUSE_CLASSIFICATION']}**. Reward outlier resolved: `{data['REWARD_OUTLIER_RESOLVED']}`.",
        f"- Candidate-Q integrity classification remains `{data['CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION']}`; counterfactual critic reliability remains `{data['COUNTERFACTUAL_CRITIC_RELIABILITY']}`; next stage is `{data['NEXT_STAGE']}`. No training or next-stage work was started.",
        f"- JSON artifact: [`{Path(data['OUTPUT_JSON_PATH']).name}`]({Path(data['OUTPUT_JSON_PATH']).name}).",
        "",
    ]
    return "\n".join(lines)


def _update_reports(data: dict[str, Any], output_path: Path) -> None:
    gate_path = ROOT / "reports" / "second_stage_joint_repair_gate.md"
    training_path = ROOT / "reports" / "one_epoch_training_gate.md"
    gate_text = gate_path.read_text(encoding="utf-8")
    heading = f"## {data['TARGET_HEAD']} Transition-{data['TARGET_TRANSITION']} Reward Root-Cause Gate"
    if heading in gate_text:
        raise RuntimeError(f"refusing to append duplicate report heading: {heading}")
    section = _markdown_report(data)
    gate_path.write_text(gate_text.rstrip() + "\n\n" + section, encoding="utf-8")

    training_text = training_path.read_text(encoding="utf-8")
    required = {
        "ONE_EPOCH_TRAINING_EXECUTED = NO",
        "ONE_EPOCH_COMPLETED_EPOCHS = 0",
        "EPOCH_2_STARTED = NO",
        "READY_FOR_FORMAL_TRAINING_STAGE = NO",
    }
    if not all(field in training_text for field in required):
        raise AssertionError("one-epoch training gate status fields changed unexpectedly")
    old_reason = "ONE_EPOCH_NOT_EXECUTED_REASON = Traceable Candidate-Q diagnostics completed; formal PPO epoch remains deferred to the next stage"
    new_reason = (
        f"ONE_EPOCH_NOT_EXECUTED_REASON = Transition-{data['TARGET_TRANSITION']} / {data['TARGET_HEAD']} reward root-cause gate completed; "
        "formal PPO epoch remains deferred to COUNTERFACTUAL_CRITIC_RELIABILITY_GATE"
        if data["REWARD_OUTLIER_RESOLVED"] == "YES"
        else f"ONE_EPOCH_NOT_EXECUTED_REASON = Transition-{data['TARGET_TRANSITION']} / {data['TARGET_HEAD']} reward root-cause gate completed; formal PPO epoch remains deferred"
    )
    if old_reason in training_text:
        training_text = training_text.replace(old_reason, new_reason, 1)
    note = (
        "\n\n## Latest TRAIN-only decision\n\n"
        f"Transition-{data['TARGET_TRANSITION']} / {data['TARGET_HEAD']} reward root cause: "
        f"`{data['REWARD_OUTLIER_ROOT_CAUSE_CLASSIFICATION']}`. "
        f"Evidence: [`{output_path.name}`]({output_path.name}). `CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION` remains "
        f"`{data['CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION']}`; next stage: `{data['NEXT_STAGE']}`. "
        "Formal PPO update and one-epoch training remain unexecuted; readiness remains NO.\n"
    )
    if "## Latest TRAIN-only decision" in training_text:
        raise RuntimeError("one-epoch report already contains a latest TRAIN-only decision section")
    training_path.write_text(training_text.rstrip() + note, encoding="utf-8")


def _validate_only(result_path: Path, config_path: Path, device: str) -> None:
    source = json.loads(result_path.read_text(encoding="utf-8"))
    provenance = _provenance(source)
    if _git("rev-parse", "HEAD") != str(provenance["CURRENT_HEAD"]):
        raise AssertionError("current HEAD differs from saved result JSON")
    context = load_environment_context(str(config_path))
    seeds = _verify_seeds_and_config(context, config_path, provenance)
    trainer = _run_with_isolated_rng(
        seeds["TRAINER_INITIALIZATION_RNG_SEED"],
        lambda: PPOV2Trainer(context, device=device, exact_q_parallel_workers=1),
    )
    if int(trainer.critic_optimizer_steps) != 0 or int(trainer.actor_optimizer_steps) != 0:
        raise AssertionError("validate-only trainer unexpectedly stepped an optimizer")
    print("diagnostic runner import/config check PASS", flush=True)
    print(f"current_head={provenance['CURRENT_HEAD']}", flush=True)
    print(f"config_sha256={provenance['CONFIG_SHA256']}", flush=True)
    print(f"saved_seed_bundle={json.dumps(seeds, sort_keys=True)}", flush=True)


def _optional_episode_manifest(directory: Path) -> tuple[dict[str, Any], str]:
    manifest_path = directory / "episode_manifest.json"
    if not manifest_path.is_file():
        return {}, "ABSENT_OPTIONAL"
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, "INVALID_IGNORED"
    if not isinstance(value, dict):
        return {}, "INVALID_IGNORED"
    return value, "READ_OPTIONAL"


def _explicit_realization_provenance(
    directory: Path, manifest: dict[str, Any], manifest_status: str,
    seed_bundle: dict[str, int], configured_protocol: str, episode_slots: int,
    config_sha256: str, generation_config_fingerprint: str,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    """Check available provenance without making the optional manifest mandatory."""
    path_parts = [part.lower() for part in directory.parts]
    in_training_pool = any(
        part == "proposed_joint_training"
        or part.startswith("proposed_joint_training_preexisting")
        for part in path_parts
    )
    diagnostic_train_directory = directory.name.lower() == "diagnostic_train_realization"
    declared_split = manifest.get("split", manifest.get("data_split"))
    if declared_split is not None and str(declared_split).upper() != "TRAIN":
        raise AssertionError(
            f"selected realization explicitly declares split={declared_split!r}, expected TRAIN"
        )
    if not (in_training_pool or diagnostic_train_directory or str(declared_split).upper() == "TRAIN"):
        raise AssertionError(
            "selected realization has no TRAIN split provenance in its path or metadata"
        )

    expected_seed = int(seed_bundle["DIAGNOSTIC_REALIZATION_SEED"])
    name_match = re.fullmatch(r"episode_(\d+)", directory.name)
    folder_seed = int(name_match.group(1)) if name_match else None
    stored_seed_value = manifest.get("episode_seed")
    try:
        stored_seed = int(stored_seed_value) if stored_seed_value is not None else None
    except (TypeError, ValueError):
        stored_seed = None
    if folder_seed is not None and stored_seed is not None and folder_seed != stored_seed:
        raise AssertionError("selected realization folder and optional manifest disagree on episode seed")
    realized_seed = stored_seed if stored_seed is not None else folder_seed
    if realized_seed is not None and realized_seed != expected_seed:
        raise AssertionError("selected realization has the wrong episode seed")

    conflicts: list[str] = []
    stored_fingerprint = manifest.get("generation_config_fingerprint")
    if stored_fingerprint is not None and str(stored_fingerprint) != generation_config_fingerprint:
        conflicts.append("generation_config_fingerprint")
    for field in ("run_config_sha256", "config_sha256", "full_config_sha256"):
        stored_hash = manifest.get(field)
        if stored_hash is not None and str(stored_hash) != config_sha256:
            conflicts.append(field)
    for field, expected_hash in source_hashes.items():
        stored_hash = manifest.get(field)
        if stored_hash is not None and str(stored_hash) != expected_hash:
            conflicts.append(field)
    stored_protocol = manifest.get(
        "data_protocol",
        manifest.get("ppo_data_protocol", manifest.get("training_data_protocol")),
    )
    if stored_protocol is not None and str(stored_protocol) != configured_protocol:
        conflicts.append("data_protocol")
    stored_slots = manifest.get("episode_slots")
    if stored_slots is None and isinstance(manifest.get("simulation"), dict):
        stored_slots = manifest["simulation"].get("episode_slots")
    if stored_slots is not None:
        try:
            if int(stored_slots) != int(episode_slots):
                conflicts.append("episode_slots")
        except (TypeError, ValueError):
            conflicts.append("episode_slots")
    stored_generation_seed = manifest.get(
        "generation_rng_seed", manifest.get("generation_rng_seed_used")
    )
    if stored_generation_seed is not None:
        try:
            if int(stored_generation_seed) != int(seed_bundle["DIAGNOSTIC_GENERATION_RNG_SEED"]):
                conflicts.append("generation_rng_seed")
        except (TypeError, ValueError):
            conflicts.append("generation_rng_seed")
    artifact_hashes = manifest.get("generated_artifact_sha256")
    if isinstance(artifact_hashes, dict):
        for filename, expected_hash in artifact_hashes.items():
            if isinstance(expected_hash, dict):
                expected_hash = expected_hash.get("sha256")
            artifact_path = directory / str(filename)
            actual_hash = (
                hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                if artifact_path.is_file() else None
            )
            if actual_hash != expected_hash:
                conflicts.append(f"artifact_sha256:{filename}")
    if conflicts:
        raise AssertionError(
            "selected realization has explicit compatibility conflicts: "
            + ", ".join(sorted(set(conflicts)))
        )
    return {
        "split": "TRAIN",
        "split_evidence": (
            "TRAIN_POOL_PATH" if in_training_pool
            else "DIAGNOSTIC_TRAIN_REALIZATION_PATH" if diagnostic_train_directory
            else "OPTIONAL_MANIFEST"
        ),
        "episode_manifest_status": manifest_status,
        "episode_seed": realized_seed,
        "generation_rng_seed": stored_generation_seed,
        "generation_config_fingerprint": stored_fingerprint,
        "data_protocol": stored_protocol,
        "episode_slots": stored_slots,
        "explicit_compatibility_conflicts": [],
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    source_path = _resolve(args.result_json)
    config_path = _resolve(args.config)
    output_path = (
        _resolve(args.output_json) if args.output_json
        else ROOT / "reports" / (
            f"{re.sub(r'[^a-z0-9]+', '', args.head_id.lower())}_transition"
            f"{int(args.transition_index)}_reward_root_cause.json"
        )
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    provenance = _provenance(source)
    current_head = _git("rev-parse", "HEAD")
    if current_head != str(provenance["CURRENT_HEAD"]):
        raise AssertionError("current HEAD differs from the retained result JSON")
    candidates, selected = _retained_candidates(source, args.transition_index, args.head_id)
    base_context = load_environment_context(str(config_path))
    seed_bundle = _verify_seeds_and_config(base_context, config_path, provenance)
    if str(base_context.data_protocol) != project_config.PPO_DATA_PROTOCOL_FULL_EPISODE_RANDOMIZED:
        raise AssertionError("reward branch replay requires the retained full_episode_randomized TRAIN protocol")
    if int(base_context.episode_slots) != 300:
        raise AssertionError(f"expected 300 TRAIN episode slots, got {base_context.episode_slots}")
    required_artifacts = [
        *REALIZATION_ARTIFACT_FILENAMES.values(),
        *GENERATOR_EXTRA_ARTIFACT_FILENAMES,
    ]
    generator = _load_single_episode_generator(ROOT)
    generator_config = generator.load_and_validate_config(config_path)
    generation_config_fingerprint = generator.generation_config_fingerprint(generator_config)
    source_hashes = generator._source_artifact_hashes(generator_config)
    existing_realization_audit = _audit_existing_realizations(
        seed_bundle["DIAGNOSTIC_REALIZATION_SEED"],
        seed_bundle["DIAGNOSTIC_GENERATION_RNG_SEED"],
        str(provenance["CONFIG_SHA256"]),
        required_artifacts,
        str(base_context.data_protocol),
        int(base_context.episode_slots),
        generation_config_fingerprint,
        source_hashes,
        base_context,
        generator,
    )
    selected_provenance: dict[str, Any] | None = None
    if args.realization_dir:
        selected_realization_dir = _resolve(args.realization_dir).resolve(strict=True)
        if not selected_realization_dir.is_dir():
            raise NotADirectoryError(f"selected realization is not a directory: {selected_realization_dir}")
        episode_manifest, manifest_status = _optional_episode_manifest(selected_realization_dir)
        selected_provenance = _explicit_realization_provenance(
            selected_realization_dir, episode_manifest, manifest_status, seed_bundle,
            str(base_context.data_protocol), int(base_context.episode_slots),
            str(provenance["CONFIG_SHA256"]), generation_config_fingerprint, source_hashes,
        )
        realization_path_kind = (
            "ENVIRONMENT_DATA_TRAIN_REUSE"
            if selected_provenance["split_evidence"] == "TRAIN_POOL_PATH"
            else "EXPLICIT_DIAGNOSTIC_TRAIN_REUSE"
        )
    elif existing_realization_audit["compatible_train_realization_paths"]:
        relative_path = existing_realization_audit["compatible_train_realization_paths"][0]
        selected_realization_dir = (ROOT / relative_path).resolve(strict=True)
        selected_provenance, manifest_status = _optional_episode_manifest(selected_realization_dir)
        selected_provenance = {
            "split": "TRAIN",
            "split_evidence": "TRAIN_POOL_PATH",
            "episode_manifest_status": manifest_status,
            **selected_provenance,
        }
        realization_path_kind = "ENVIRONMENT_DATA_TRAIN_REUSE"
    else:
        selected_realization_dir = None
        realization_path_kind = "NEW_TEMPORARY_GENERATION"

    data: dict[str, Any] = {
        "CURRENT_HEAD": current_head,
        "TARGET_TRANSITION": int(args.transition_index),
        "TARGET_ORIGIN_INDEX": int(selected["keep"]["origin_index"]),
        "TARGET_HEAD": str(args.head_id),
        "RESULT_JSON_PATH": str(source_path),
        "OUTPUT_JSON_PATH": str(output_path),
        "CONFIG_SHA256": str(provenance["CONFIG_SHA256"]),
        "SEEDS": {key: value for key, value in seed_bundle.items() if key != "FORMAL_EPISODE_SEED_STREAM_COUNT"},
        "RETAINED_CANDIDATE_COUNT": len(candidates),
        "DATA_PROTOCOL": str(base_context.data_protocol),
        "EPISODE_SLOTS": int(base_context.episode_slots),
        "EXISTING_REALIZATION_AUDIT": existing_realization_audit,
        "REUSE_EXISTING_REALIZATION": "YES" if selected_realization_dir is not None else "NO",
        "GENERATE_NEW_REALIZATION": "NO" if selected_realization_dir is not None else "YES",
        "REALIZATION_PATH_KIND": realization_path_kind,
        "REALIZATION_PATH": str(selected_realization_dir) if selected_realization_dir else None,
        "SELECTED_REALIZATION_PROVENANCE": selected_provenance,
        "REWARD_BEST_SAME_AS_EXACT_Q_BEST": "YES" if selected["reward_best"]["target_node_cache"] == selected["exact_q_best"]["target_node_cache"] else "NO",
        "REPRODUCTION_GATE": "NOT_RUN",
        "BRANCH_INPUT_ISOLATION_GATE": "NOT_RUN",
        "BRANCHES": {},
        "EXPLICIT_PENALTY_FOUND": "NO",
        "PENALTY_AUDIT": {
            "infeasible_penalty": "NO; cache_credit._execute_cache_branch raises on infeasible SlotExecutionResult before reward is returned",
            "fallback_penalty": "NO explicit fallback utility penalty in task_allocation Matching or task utility",
            "cloud_penalty": "NO explicit Cloud utility penalty; CLOUD is a production target/fallback node",
            "deadline_violation_penalty": "NO explicit deadline penalty found in task_utility/system_utility formula",
            "task_utility_formula_source": "src/common/system_model.py::task_utility",
            "system_utility_formula_source": "src/decision/slot_environment.py::execute_slot; system_utility = actual_task_utility - cache_cost - movement_cost",
            "infeasible_execution_source": "src/ppo_simple/cache_credit.py::_execute_cache_branch raises RuntimeError when result.step_feasible is false",
        },
        "CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION": str(source.get("CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION", source.get("classification", "UNKNOWN"))),
        "COUNTERFACTUAL_CRITIC_RELIABILITY": str(source.get("COUNTERFACTUAL_CRITIC_RELIABILITY", "UNKNOWN")),
        "FORMAL_PPO_UPDATE_EXECUTED": "NO",
        "ONE_EPOCH_TRAINING_EXECUTED": "NO",
        "ONE_EPOCH_COMPLETED_EPOCHS": 0,
        "EPOCH_2_STARTED": "NO",
        "READY_FOR_FORMAL_TRAINING_STAGE": "NO",
        "SURROGATE_TRAINING_EXECUTED": "NO",
    }

    trainer = _run_with_isolated_rng(
        seed_bundle["TRAINER_INITIALIZATION_RNG_SEED"],
        lambda: PPOV2Trainer(base_context, device=args.device, exact_q_parallel_workers=1),
    )
    optimizer_steps_before = (int(trainer.actor_optimizer_steps), int(trainer.critic_optimizer_steps))
    if optimizer_steps_before != (0, 0):
        raise AssertionError(f"fresh trainer optimizer counters are nonzero: {optimizer_steps_before}")

    scratch_context = (
        tempfile.TemporaryDirectory(prefix="cache_branch_reward_root_cause_")
        if selected_realization_dir is None else nullcontext(None)
    )
    with scratch_context as temp_dir:
        if selected_realization_dir is None:
            temp_root = Path(temp_dir)
            realization_dir = temp_root / "diagnostic_train_realization"
        else:
            realization_dir = selected_realization_dir
        episode_manifest, manifest_status = _optional_episode_manifest(realization_dir)
        generating_new = selected_realization_dir is None
        print(
            "Generating the retained-seed TRAIN diagnostic realization."
            if generating_new else "Loading the verified retained-seed TRAIN diagnostic realization.",
            flush=True,
        )
        realization_result = _run_with_isolated_rng(
            seed_bundle["DIAGNOSTIC_GENERATION_RNG_SEED"],
            lambda: _ensure_episode_realization(
                base_context, str(config_path), realization_dir,
                seed_bundle["DIAGNOSTIC_REALIZATION_SEED"],
                generate_missing=generating_new, generation_device=args.device,
                generator_module=generator,
            ),
        )
        diagnostic_context = load_environment_context(
            str(config_path), realization_dir=realization_dir,
        )
        _assert_realization_context_compatible(base_context, diagnostic_context)
        trainer.context = diagnostic_context
        print("Replaying the original complete TRAIN policy rollout (300 slots).", flush=True)
        rollout = _run_with_isolated_rng(
            seed_bundle["DIAGNOSTIC_ROLLOUT_RNG_SEED"],
            lambda: trainer.collect_chunk(
                diagnostic_context.train_origin_start,
                diagnostic_context.initial_cache,
                diagnostic_context.initial_uav,
                diagnostic_context.configured_rollout_steps,
            ),
        )
        if len(rollout.transitions) != diagnostic_context.episode_slots:
            raise AssertionError("diagnostic policy replay did not cover the complete TRAIN episode")
        if int(args.transition_index) < 0 or int(args.transition_index) >= len(rollout.transitions):
            raise IndexError("transition-index is outside the reproduced TRAIN rollout")
        transition = rollout.transitions[int(args.transition_index)]
        target_origin = int(transition.origin_index)
        if target_origin != int(selected["keep"]["origin_index"]):
            raise AssertionError(
                f"replayed origin {target_origin} differs from retained origin {selected['keep']['origin_index']}"
            )
        steps = aligned_train_steps(
            diagnostic_context, origin_start=target_origin, max_steps=1,
            include_bootstrap_step=False,
        )
        if not steps or int(steps[0].origin_index) != target_origin:
            raise AssertionError("aligned target step does not match the replayed transition origin")
        step = steps[0]
        expected_keep = float(selected["keep"]["keep_reward"])
        expected_exact = float(selected["exact_q_best"]["reward"])
        expected_reward_best = float(selected["reward_best"]["reward"])

        node_ids = [str(value) for value in diagnostic_context.action_spec.cache_node_ids]
        if str(args.head_id) not in node_ids:
            raise ValueError(f"target cache head is absent from action spec: {args.head_id}")
        node_index = node_ids.index(str(args.head_id))
        if int(selected["keep"]["head_index"]) != node_index:
            raise AssertionError("saved cache head index differs from the production ActionSpec")
        working_cache, previous_cache = cache_credit._replay_cache_prefix(transition, node_index)
        current_row = np.asarray(working_cache[node_index], dtype=np.int8).astype(int).tolist()
        exact_row = [int(value) for value in selected["exact_q_best"]["target_node_cache"]]
        reward_row = [int(value) for value in selected["reward_best"]["target_node_cache"]]
        fingerprint_payload = {
            "transition_index": int(args.transition_index),
            "origin_index": target_origin,
            "slot": int(step.slot),
            "action": np.asarray(transition.action, dtype=np.int8).astype(int).tolist(),
            "previous_cache_state": np.asarray(transition.previous_cache_state, dtype=np.int8).astype(int).tolist(),
            "production_next_cache_state": np.asarray(transition.next_cache_state, dtype=np.int8).astype(int).tolist(),
            "previous_uav_positions": _position_record(transition.previous_uav_positions),
            "production_next_uav_positions": _position_record(transition.next_uav_positions),
            "actual_tasks": [_task_record(task) for task in step.actual_tasks],
            "actual_vehicle_positions": _position_record(step.actual_vehicle_positions),
            "config_sha256": provenance["CONFIG_SHA256"],
        }
        data["TRANSITION_FINGERPRINT_SHA256"] = _digest(fingerprint_payload)
        data["TARGET_KEEP_CACHE_ROW"] = current_row
        data["TARGET_EXACT_BEST_CACHE_ROW"] = exact_row
        data["TARGET_REWARD_BEST_CACHE_ROW"] = reward_row
        data["ADDED_SERVICES"] = [index for index, (before, after) in enumerate(zip(current_row, exact_row)) if before == 0 and after == 1]
        data["REMOVED_SERVICES"] = [index for index, (before, after) in enumerate(zip(current_row, exact_row)) if before == 1 and after == 0]
        data["REPRODUCED_KEEP_REWARD"] = None
        data["REPRODUCED_EXACT_BEST_REWARD"] = None
        data["REPRODUCED_REWARD_BEST_REWARD"] = None
        data["EXPECTED_KEEP_REWARD"] = expected_keep
        data["EXPECTED_EXACT_BEST_REWARD"] = expected_exact
        data["EXPECTED_REWARD_BEST_REWARD"] = expected_reward_best
        data["EXACT_Q_BEST_CONFIG"] = exact_row
        data["REWARD_BEST_CONFIG"] = reward_row
        data["EXACT_Q_BEST_Q"] = _finite(selected["exact_q_best"]["q"], "saved exact-best Q")
        data["REWARD_BEST_SAVED_UNIVERSE_INDEX"] = int(selected["reward_best"]["universe_index"])

        print("Executing KEEP and retained Exact-Q best branches for the reproduction gate.", flush=True)
        keep_branch = _run_branch(trainer, transition, step, node_index, working_cache, previous_cache, None)
        exact_candidate = _candidate_for(
            selected["exact_q_best"], node_index, args.head_id,
            working_cache[node_index], "retained_exact_q_best",
        )
        exact_branch = _run_branch(
            trainer, transition, step, node_index, working_cache, previous_cache,
            exact_candidate,
        )
        data["REPRODUCED_KEEP_REWARD"] = _finite(keep_branch["reward"], "reproduced KEEP reward")
        data["REPRODUCED_EXACT_BEST_REWARD"] = _finite(exact_branch["reward"], "reproduced exact-best reward")
        keep_matches = math.isclose(data["REPRODUCED_KEEP_REWARD"], expected_keep, rel_tol=RTOL, abs_tol=ATOL)
        exact_matches = math.isclose(data["REPRODUCED_EXACT_BEST_REWARD"], expected_exact, rel_tol=RTOL, abs_tol=ATOL)
        data["REPRODUCTION_GATE"] = "PASS" if keep_matches and exact_matches else "FAIL"
        branches_raw: dict[str, dict[str, Any]] = {
            "KEEP": keep_branch,
            "EXACT_Q_BEST": exact_branch,
        }
        if not keep_matches or not exact_matches:
            data["BRANCHES"] = {
                "KEEP": _branch_summary("KEEP", keep_branch, current_row),
                "EXACT_Q_BEST": _branch_summary("EXACT_Q_BEST", exact_branch, exact_row),
            }
            data["BRANCH_INPUT_ISOLATION_GATE"] = "NOT_RUN"
            data["ROOT_CAUSE_ANALYSIS_STOPPED_AT_REPRODUCTION_GATE"] = "YES"
            data["REWARD_OUTLIER_ROOT_CAUSE_CLASSIFICATION"] = "REWARD_OUTLIER_INCONCLUSIVE"
            data["REWARD_OUTLIER_RESOLVED"] = "NO"
            data["NEXT_STAGE"] = "TRANSITION_REWARD_ROOT_CAUSE_FOLLOWUP"
            data["GATE_FAILURE_REASON"] = "saved KEEP or Exact-Q best immediate reward did not reproduce within tolerance; reward-best was not executed"
            data["REPRODUCED_REWARD_BEST_REWARD"] = None
            _write_artifact(data, output_path)
            return data
        if data["REWARD_BEST_SAME_AS_EXACT_Q_BEST"] == "YES":
            reward_branch = exact_branch
        else:
            print("Executing the one retained immediate-reward-best branch.", flush=True)
            reward_candidate = _candidate_for(
                selected["reward_best"], node_index, args.head_id,
                working_cache[node_index], "retained_immediate_reward_best",
            )
            reward_branch = _run_branch(
                trainer, transition, step, node_index, working_cache,
                previous_cache, reward_candidate,
            )
        data["REPRODUCED_REWARD_BEST_REWARD"] = _finite(reward_branch["reward"], "reproduced reward-best reward")
        data["REWARD_BEST_REPRODUCTION_MATCH"] = math.isclose(
            data["REPRODUCED_REWARD_BEST_REWARD"], expected_reward_best,
            rel_tol=RTOL, abs_tol=ATOL,
        )
        branches_raw["REWARD_BEST"] = reward_branch
        configs = {
            "KEEP": current_row,
            "EXACT_Q_BEST": exact_row,
            "REWARD_BEST": reward_row,
        }
        names = ["KEEP", "EXACT_Q_BEST", "REWARD_BEST"]
        data["BRANCHES"] = {
            name: _branch_summary(name, branches_raw[name], configs[name])
            for name in names
        }
        data["BRANCH_REWARD_BEST_SAME_AS_EXACT_Q_BEST"] = (
            data["REWARD_BEST_SAME_AS_EXACT_Q_BEST"]
        )
        if not keep_matches or not exact_matches or not data["REWARD_BEST_REPRODUCTION_MATCH"]:
            data["BRANCH_INPUT_ISOLATION_GATE"] = "NOT_RUN"
            data["ROOT_CAUSE_ANALYSIS_STOPPED_AT_REPRODUCTION_GATE"] = "YES"
            data["REWARD_OUTLIER_ROOT_CAUSE_CLASSIFICATION"] = "REWARD_OUTLIER_INCONCLUSIVE"
            data["REWARD_OUTLIER_RESOLVED"] = "NO"
            data["NEXT_STAGE"] = "TRANSITION_REWARD_ROOT_CAUSE_FOLLOWUP"
            data["GATE_FAILURE_REASON"] = "saved KEEP / Exact-Q best / reward-best immediate reward did not reproduce within tolerance"
            _write_artifact(data, output_path)
            return data

        isolation_branches = {
            "KEEP": keep_branch,
            "EXACT_Q_BEST": exact_branch,
            "REWARD_BEST": reward_branch,
        }
        isolation = _isolation(
            names, isolation_branches, transition, step, diagnostic_context,
            node_index, configs,
        )
        data.update(isolation)
        if data["BRANCH_INPUT_ISOLATION_GATE"] != "PASS":
            data["ROOT_CAUSE_ANALYSIS_STOPPED_AFTER_ISOLATION_GATE"] = "YES"
            data["REWARD_OUTLIER_ROOT_CAUSE_CLASSIFICATION"] = "REWARD_OUTLIER_BRANCH_ISOLATION_BUG"
            data["REWARD_OUTLIER_RESOLVED"] = "NO"
            data["NEXT_STAGE"] = "TRANSITION_REWARD_ROOT_CAUSE_FOLLOWUP"
            data["GATE_FAILURE_REASON"] = "branch input isolation failed; utility root-cause analysis stopped"
            _write_artifact(data, output_path)
            return data

        print("Reproduction and branch-isolation gates passed; reconstructing task and cost deltas.", flush=True)
        keep_summary = data["BRANCHES"]["KEEP"]
        exact_summary = data["BRANCHES"]["EXACT_Q_BEST"]
        data["MAX_SYSTEM_UTILITY_RECONSTRUCTION_ERROR"] = max(
            branch["SYSTEM_UTILITY_RECONSTRUCTION_ERROR"]
            for branch in data["BRANCHES"].values()
        )
        data["SYSTEM_UTILITY_RECONSTRUCTION"] = (
            "PASS" if all(math.isclose(
                branch["SYSTEM_UTILITY"],
                branch["ACTUAL_TASK_UTILITY"] - branch["CACHE_COST"] - branch["MOVEMENT_COST"],
                rel_tol=RTOL, abs_tol=ATOL,
            ) for branch in data["BRANCHES"].values()) else "FAIL"
        )
        data["DELTA_ACTUAL_TASK_UTILITY"] = exact_summary["ACTUAL_TASK_UTILITY"] - keep_summary["ACTUAL_TASK_UTILITY"]
        data["DELTA_CACHE_COST"] = exact_summary["CACHE_COST"] - keep_summary["CACHE_COST"]
        data["NEG_DELTA_CACHE_COST"] = -data["DELTA_CACHE_COST"]
        data["DELTA_MOVEMENT_COST"] = exact_summary["MOVEMENT_COST"] - keep_summary["MOVEMENT_COST"]
        data["NEG_DELTA_MOVEMENT_COST"] = -data["DELTA_MOVEMENT_COST"]
        data["DELTA_SYSTEM_UTILITY"] = exact_summary["SYSTEM_UTILITY"] - keep_summary["SYSTEM_UTILITY"]
        reconstructed_delta = data["DELTA_ACTUAL_TASK_UTILITY"] - data["DELTA_CACHE_COST"] - data["DELTA_MOVEMENT_COST"]
        data["DELTA_SYSTEM_UTILITY_RECONSTRUCTION_ERROR"] = abs(data["DELTA_SYSTEM_UTILITY"] - reconstructed_delta)
        data["DELTA_SYSTEM_UTILITY_RECONSTRUCTION"] = (
            "PASS" if math.isclose(data["DELTA_SYSTEM_UTILITY"], reconstructed_delta, rel_tol=RTOL, abs_tol=ATOL) else "FAIL"
        )
        data["ADDITIVE_COMPONENT_DELTAS"] = {
            "ACTUAL_TASK_UTILITY": data["DELTA_ACTUAL_TASK_UTILITY"],
            "CACHE_COST": data["DELTA_CACHE_COST"],
            "MOVEMENT_COST": data["DELTA_MOVEMENT_COST"],
        }
        data["LARGEST_REWARD_COMPONENT_CHANGE"] = max(
            data["ADDITIVE_COMPONENT_DELTAS"],
            key=lambda key: abs(data["ADDITIVE_COMPONENT_DELTAS"][key]),
        )
        tasks = list(step.actual_tasks)
        task_rows, task_stats = _task_comparison(tasks, keep_branch, exact_branch)
        data.update(task_stats)
        data["TASK_UTILITY_IDENTITY"] = (
            "PASS" if all(math.isclose(
                branch["PER_TASK_UTILITY_SUM"], branch["ACTUAL_TASK_UTILITY"],
                rel_tol=RTOL, abs_tol=ATOL,
            ) for branch in data["BRANCHES"].values()) else "FAIL"
        )
        data["MAX_PER_TASK_UTILITY_RECONSTRUCTION_ERROR"] = max(
            branch["PER_TASK_UTILITY_RECONSTRUCTION_ERROR"]
            for branch in data["BRANCHES"].values()
        )
        data["TOP20_TASK_UTILITY_CHANGES"] = task_rows[:20]
        data["CHANGED_TASK_SERVICE_FEASIBILITY"] = _service_feasibility(
            tasks, task_rows, keep_branch, exact_branch, diagnostic_context, step,
        )
        data["MATCHING_STABLE_KEEP"] = "YES" if keep_summary["MATCHING_STABLE"] else "NO"
        data["MATCHING_STABLE_EXACT_BEST"] = "YES" if exact_summary["MATCHING_STABLE"] else "NO"
        data["KKT_EXECUTION_CONCERN"] = (
            "YES" if any(branch["KKT_EXECUTION_CONCERN"] for branch in data["BRANCHES"].values()) else "NO"
        )
        data["KKT_EXECUTION_CONCERN_KEEP"] = "YES" if keep_summary["KKT_EXECUTION_CONCERN"] else "NO"
        data["KKT_EXECUTION_CONCERN_EXACT_BEST"] = "YES" if exact_summary["KKT_EXECUTION_CONCERN"] else "NO"
        data["EXPLICIT_PENALTY_FOUND"] = "NO"
        expected_delta = expected_exact - expected_keep
        data["EXPECTED_DELTA_REWARD"] = expected_delta
        data["REPRODUCED_DELTA_REWARD"] = data["REPRODUCED_EXACT_BEST_REWARD"] - data["REPRODUCED_KEEP_REWARD"]
        data["DELTA_REWARD_REPRODUCTION_ERROR"] = abs(data["REPRODUCED_DELTA_REWARD"] - expected_delta)
        data["REWARD_IMPLEMENTATION_IDENTITIES_PASS"] = (
            data["SYSTEM_UTILITY_RECONSTRUCTION"] == "PASS"
            and data["DELTA_SYSTEM_UTILITY_RECONSTRUCTION"] == "PASS"
            and data["TASK_UTILITY_IDENTITY"] == "PASS"
            and all(branch["STEP_FEASIBLE"] for branch in data["BRANCHES"].values())
        )
        matching_kkt_healthy = (
            all(branch["MATCHING_STABLE"] for branch in data["BRANCHES"].values())
            and data["KKT_EXECUTION_CONCERN"] == "NO"
        )
        if data["BRANCH_INPUT_ISOLATION_GATE"] != "PASS":
            classification = "REWARD_OUTLIER_BRANCH_ISOLATION_BUG"
        elif not data["REWARD_IMPLEMENTATION_IDENTITIES_PASS"]:
            classification = "REWARD_OUTLIER_REWARD_IMPLEMENTATION_BUG"
        elif not matching_kkt_healthy:
            classification = "REWARD_OUTLIER_MATCHING_KKT_CONCERN"
        elif data["LARGEST_REWARD_COMPONENT_CHANGE"] == "CACHE_COST" or data["LARGEST_REWARD_COMPONENT_CHANGE"] == "MOVEMENT_COST":
            classification = "REWARD_OUTLIER_CONFIGURATION_COST_DRIVEN"
        elif (
            data["LARGEST_REWARD_COMPONENT_CHANGE"] == "ACTUAL_TASK_UTILITY"
            and (data["MATCHING_CHANGED_TASK_COUNT"] > 0 or any(
                row["DELTA_delay_s"] != 0.0 or row["DELTA_vehicle_energy_j"] != 0.0
                for row in task_rows
            ))
            and data["EXPLICIT_PENALTY_FOUND"] == "NO"
        ):
            classification = "REWARD_OUTLIER_VALID_TASK_EXECUTION"
        else:
            classification = "REWARD_OUTLIER_INCONCLUSIVE"
        data["REWARD_OUTLIER_ROOT_CAUSE_CLASSIFICATION"] = classification
        resolved = classification in {
            "REWARD_OUTLIER_VALID_TASK_EXECUTION",
            "REWARD_OUTLIER_CONFIGURATION_COST_DRIVEN",
        }
        data["REWARD_OUTLIER_RESOLVED"] = "YES" if resolved else "NO"
        data["REWARD_OUTLIER_VALID_TASK_EXECUTION"] = (
            "YES" if classification == "REWARD_OUTLIER_VALID_TASK_EXECUTION" else "NO"
        )
        data["NEXT_STAGE"] = (
            "COUNTERFACTUAL_CRITIC_RELIABILITY_GATE"
            if resolved else "TRANSITION_REWARD_ROOT_CAUSE_FOLLOWUP"
        )
        data["ROOT_CAUSE_SUMMARY"] = {
            "largest_component": data["LARGEST_REWARD_COMPONENT_CHANGE"],
            "keep_system_utility": keep_summary["SYSTEM_UTILITY"],
            "exact_best_system_utility": exact_summary["SYSTEM_UTILITY"],
            "keep_task_utility": keep_summary["ACTUAL_TASK_UTILITY"],
            "exact_best_task_utility": exact_summary["ACTUAL_TASK_UTILITY"],
            "keep_cache_cost": keep_summary["CACHE_COST"],
            "exact_best_cache_cost": exact_summary["CACHE_COST"],
            "keep_movement_cost": keep_summary["MOVEMENT_COST"],
            "exact_best_movement_cost": exact_summary["MOVEMENT_COST"],
        }
        if optimizer_steps_before != (int(trainer.actor_optimizer_steps), int(trainer.critic_optimizer_steps)):
            raise AssertionError("an actor or critic optimizer step occurred during the reward diagnostic")
        if int(getattr(trainer, "cache_surrogate_training_count", 0)) != 0:
            raise AssertionError("surrogate training occurred during the reward diagnostic")
        data["ACTOR_OPTIMIZER_STEPS"] = int(trainer.actor_optimizer_steps)
        data["CRITIC_OPTIMIZER_STEPS"] = int(trainer.critic_optimizer_steps)
        data["PPO_UPDATE_EXECUTED"] = "NO"
        data["ROLLOUT_TRANSITION_COUNT"] = len(rollout.transitions)
        data["REALIZATION_ACTION"] = str(realization_result[1])
        data["REALIZATION_MANIFEST_POLICY"] = "OPTIONAL_PROVENANCE_ONLY"
        data["REALIZATION_MANIFEST_STATUS"] = manifest_status
        data["REALIZATION_MANIFEST_EPISODE_SEED"] = episode_manifest.get("episode_seed")
        data["REALIZATION_REQUIRED_ARTIFACTS"] = required_artifacts
        data["REALIZATION_REQUIRED_ARTIFACTS_PRESENT"] = all(
            (realization_dir / name).is_file() for name in required_artifacts
        )
        data["REALIZATION_GENERATED_DURING_CURRENT_INVOCATION"] = "YES" if generating_new else "NO"
        data["TEMPORARY_REALIZATION_REMOVED"] = (
            "YES" if generating_new else "PENDING_CLEANUP"
        )

    _write_artifact(data, output_path)
    if args.update_reports:
        _update_reports(data, output_path)
    return data


def _write_artifact(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _assert_all_finite_json(data)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--transition-index", required=True, type=int)
    parser.add_argument("--head-id", required=True)
    parser.add_argument("--config", default="configs/proposed/system_config.toml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json")
    parser.add_argument("--realization-dir", help="load a previously completed, seed-verified diagnostic realization")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--update-reports", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        source_path = _resolve(args.result_json)
        config_path = _resolve(args.config).resolve(strict=True)
        if args.validate_only:
            _validate_only(source_path, config_path, args.device)
            return 0
        result = _run(args)
        for key in (
            "REPRODUCTION_GATE", "BRANCH_INPUT_ISOLATION_GATE",
            "REPRODUCED_KEEP_REWARD", "REPRODUCED_EXACT_BEST_REWARD",
            "REPRODUCED_REWARD_BEST_REWARD", "REWARD_OUTLIER_ROOT_CAUSE_CLASSIFICATION",
            "REWARD_OUTLIER_RESOLVED", "NEXT_STAGE",
        ):
            print(f"{key} = {result.get(key)}", flush=True)
        print(f"OUTPUT_JSON = {result['OUTPUT_JSON_PATH']}", flush=True)
        return 0 if result.get("REPRODUCTION_GATE") == "PASS" else 2
    except Exception as exc:
        print(f"REWARD_ROOT_CAUSE_DIAGNOSTIC_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
