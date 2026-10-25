#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the formal Train/Validation/Test pools as independent SUMO Episodes.

Each Episode uses the existing TraCI trajectory collection and complete-vehicle
selection path. Outputs remain staged until all 430 Episodes and their identity
manifests pass the formal artifact checks.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping
from xml.etree import ElementTree as ET

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for candidate in (SRC_ROOT, SCRIPTS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from common import project_config  # noqa: E402
from generation import (  # noqa: E402
    extend_formal_episode_trajectory,
    generate_continuous_sumo_dataset,
    generate_episode_pool,
    generate_episode_realization,
)
from scenario import popularity_prediction, scenario_generator, sumo_scenario, task_scenario  # noqa: E402
from scenario.realization import derive_episode_seeds  # noqa: E402


CONFIG_RELATIVE = Path("configs/dqn/online_joint_dqn.toml")
ENVIRONMENT_ROOT = PROJECT_ROOT / "environment_data"
STAGING_ROOT = ENVIRONMENT_ROOT / ".formal_vehicle_environment_staging"
SPLIT_MANIFEST_RELATIVE = Path("environment_data/data_split_manifest.json")
POOL_RELATIVES = {
    "train": Path("environment_data/proposed_joint_training"),
    "validation": Path("environment_data/proposed_joint_validation"),
    "test": Path("environment_data/proposed_joint_test"),
}
PLAN_SCHEMA = "formal_vehicle_environment_plan_v2"
SPLIT_SCHEMA = "formal_vehicle_environment_split_manifest_v2"
POOL_SCHEMA = "formal_episode_pool_v3"
INTEGRITY_SCHEMA = "formal_task_data_integrity_baseline_v3"
TRAFFIC_WINDOW_S = 300.0
EXPECTED_RESEARCH = 25
EXPECTED_SV = 5
EXPECTED_TV = 20
SEED_STARTS = {"train": 20001, "validation": 30001, "test": 40001}
SPLIT_COUNTS = {"train": 300, "validation": 30, "test": 100}
FCD_ATTRIBUTES = {
    "id": "vehicle_id",
    "x": "x_m",
    "y": "y_m",
    "speed": "speed_mps",
    "acceleration": "acceleration_mps2",
    "edge": "road_id",
    "lane": "lane_id",
    "pos": "lane_position_m",
    "distance": "distance_m",
}
FORMAL_GENERATOR_SOURCE_RELATIVE = "scripts/generation/generate_formal_vehicle_environment.py"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _record_generation_progress(progress: dict[str, Any]) -> None:
    progress["updated_at_utc"] = _utc_now()
    if STAGING_ROOT.is_dir():
        _write_json(STAGING_ROOT / "generation_status.json", progress)


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _sumo_paths(sumo_home_arg: str | None) -> tuple[Path, Path, Path]:
    home_value = sumo_home_arg or os.environ.get("SUMO_HOME")
    if not home_value:
        raise EnvironmentError("SUMO_HOME is not set; pass --sumo-home or activate the configured SUMO environment.")
    home = Path(home_value).expanduser().resolve(strict=True)
    random_trips = home / "tools" / "randomTrips.py"
    binary_candidates = (home / "bin" / "sumo.exe", home / "bin" / "sumo", home / "bin" / "sumo.com")
    binary = next((path for path in binary_candidates if path.is_file()), None)
    if not random_trips.is_file() or binary is None:
        raise FileNotFoundError(f"SUMO inputs are incomplete below {home}: randomTrips.py or sumo binary missing")
    return home, random_trips, binary


def _load_config(config_path: Path) -> dict[str, Any]:
    cfg = project_config.load_config(config_path, require_artifacts=False)
    cfg["_meta"] = {
        "config_path": str(config_path.resolve()),
        "project_root": str(PROJECT_ROOT.resolve()),
    }
    return cfg


def _seed_plan() -> dict[str, list[int]]:
    plan = {
        split: list(range(SEED_STARTS[split], SEED_STARTS[split] + SPLIT_COUNTS[split]))
        for split in ("train", "validation", "test")
    }
    flat = [seed for seeds in plan.values() for seed in seeds]
    if len(flat) != len(set(flat)):
        raise ValueError("formal Train/Validation/Test episode seed plan overlaps")
    derived_by_split: dict[str, set[int]] = {}
    for split, seeds in plan.items():
        derived_by_split[split] = {
            int(value)
            for seed in seeds
            for value in derive_episode_seeds(int(seed)).values()
        }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = derived_by_split[left] & derived_by_split[right]
        if overlap:
            raise ValueError(f"derived episode random streams overlap between {left} and {right}: {sorted(overlap)[:10]}")
    return plan


def _validate_catalog(cfg: dict[str, Any], catalog: dict[str, Any], seed: int) -> None:
    task_cfg = task_scenario.task_config_from_project_config(cfg)
    if int(catalog.get("service_count", -1)) != task_cfg.num_services:
        raise ValueError("shared service catalog service_count differs from formal config")
    if catalog.get("generator_version") != task_scenario.SERVICE_CATALOG_GENERATOR_VERSION:
        raise ValueError("shared service catalog generator version is incompatible")
    contract = catalog.get("generation_contract")
    if not isinstance(contract, dict) or int(contract.get("master_seed", -1)) != int(seed):
        raise ValueError("shared service catalog seed identity is missing or incompatible")
    if contract.get("parameter_ranges") != {
        "service_size_gb": [task_cfg.service_size_min_gb, task_cfg.service_size_max_gb],
        "input_size_kb": [task_cfg.input_size_min_kb, task_cfg.input_size_max_kb],
        "cycles_per_bit": [task_cfg.cycles_per_bit_min, task_cfg.cycles_per_bit_max],
    }:
        raise ValueError("shared service catalog ranges differ from formal config")
    services = catalog.get("services")
    if not isinstance(services, list) or len(services) != task_cfg.num_services:
        raise ValueError("shared service catalog services are incomplete")
    task_scenario.catalog_by_id(services)
    task_scenario.service_parameter_stratum_summary(services, task_cfg)
    for service in services:
        if not all(math.isfinite(float(service[field])) and float(service[field]) > 0 for field in (
            "program_size", "mean_input_size", "cycles_per_bit"
        )):
            raise ValueError("shared service catalog has a non-finite or non-positive runtime parameter")


def _runtime_checks(
    config_path: Path,
    *,
    sumo_home_arg: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _load_config(config_path)
    protocol = project_config.get_episode_data_protocol(cfg)
    task_timing = project_config.get_cross_slot_task_parameters(cfg)
    simulation = cfg["simulation"]
    dt_s = float(simulation["dt_s"])
    warmup_s = float(simulation["warmup_time_s"])
    random_trips_period_s = float(
        project_config.get_config_value(cfg, "simulation.random_trips_period_s")
    )
    arrival_slots = int(protocol["arrival_horizon_slots"])
    execution_slots = int(protocol["execution_horizon_slots"])
    k_max = int(task_timing["k_max_slots"])
    if not math.isclose(dt_s, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"formal generation expects dt_s=1.0, actual={dt_s}")
    if not math.isclose(warmup_s, 60.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"formal generation expects warmup_time_s=60.0, actual={warmup_s}")
    if arrival_slots != 300 or k_max != 15 or execution_slots != arrival_slots + k_max - 1 or execution_slots != 314:
        raise ValueError(
            "formal time contract mismatch: expected arrival=300, K_max=15, execution=314; "
            f"actual={arrival_slots}/{k_max}/{execution_slots}"
        )
    random_trips_contract = {
        "random_trips_period_s": (random_trips_period_s, 1.0),
        "random_trips_departure_begin_s": (
            float(project_config.get_config_value(cfg, "simulation.random_trips_departure_begin_s")),
            0.0,
        ),
        "random_trips_departure_end_s": (
            float(project_config.get_config_value(cfg, "simulation.random_trips_departure_end_s")),
            60.0,
        ),
        "random_trips_min_distance_m": (
            float(project_config.get_config_value(cfg, "simulation.random_trips_min_distance_m")),
            4000.0,
        ),
        "random_trips_intermediate": (
            int(project_config.get_config_value(cfg, "simulation.random_trips_intermediate")),
            2,
        ),
        "random_trips_maxtries": (
            int(project_config.get_config_value(cfg, "simulation.random_trips_maxtries")),
            1000,
        ),
    }
    contract_mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in random_trips_contract.items()
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
    }
    if contract_mismatches:
        raise ValueError(
            "formal independent-Episode candidate traffic contract mismatch: "
            + json.dumps(contract_mismatches, sort_keys=True)
        )
    roles = (
        int(project_config.get_config_value(cfg, "simulation.num_research_vehicles")),
        int(project_config.get_config_value(cfg, "simulation.num_service_vehicles")),
        int(project_config.get_config_value(cfg, "simulation.num_task_vehicles")),
    )
    if roles != (EXPECTED_RESEARCH, EXPECTED_SV, EXPECTED_TV):
        raise ValueError(f"formal research vehicle roles must be 25/5/20, actual={roles}")
    task_cfg = task_scenario.task_config_from_project_config(cfg)
    if task_cfg.num_services != 10 or (
        task_cfg.service_size_min_gb,
        task_cfg.service_size_max_gb,
        task_cfg.input_size_min_kb,
        task_cfg.input_size_max_kb,
        task_cfg.cycles_per_bit_min,
        task_cfg.cycles_per_bit_max,
    ) != (2.0, 20.0, 500.0, 1000.0, 400.0, 1000.0):
        raise ValueError("formal service parameter ranges are not the approved 10-service 500–1000 KB contract")
    warmup_slots = int(round(warmup_s / dt_s))
    history_length = int(project_config.get_config_value(cfg, "prediction.popularity.history_length"))
    if warmup_slots < history_length:
        raise ValueError(
            f"60-second warm-up is shorter than the configured popularity history window: {warmup_slots} < {history_length}"
        )
    for config_key in (
        "paths.sumo_net",
        "paths.popularity_gru",
        "paths.vehicle_residual_bank",
    ):
        path = project_config.resolve_path(PROJECT_ROOT, project_config.get_config_value(cfg, config_key))
        if not path.is_file():
            raise FileNotFoundError(f"required formal generation input is missing ({config_key}): {path}")
    network_path = project_config.resolve_path(PROJECT_ROOT, project_config.get_config_value(cfg, "paths.sumo_net"))
    residual_path = project_config.resolve_path(PROJECT_ROOT, project_config.get_config_value(cfg, "paths.vehicle_residual_bank"))
    residual_bank = scenario_generator.load_residual_bank(residual_path)
    if int(residual_bank["H_p"]) != int(cfg["prediction"]["vehicle"]["horizon"]):
        raise ValueError("calibration residual horizon differs from formal vehicle prediction horizon")
    if not math.isclose(float(residual_bank["dt"]), dt_s, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("calibration residual slot duration differs from formal dt_s")
    gru_path = project_config.resolve_path(PROJECT_ROOT, project_config.get_config_value(cfg, "paths.popularity_gru"))
    model, gru_metadata = popularity_prediction.load_checkpoint_model(gru_path, torch.device("cpu"))
    del model
    if int(gru_metadata["F"]) != task_cfg.num_services:
        raise ValueError("frozen popularity GRU service dimension differs from service catalog")
    if int(gru_metadata["H_p"]) != int(cfg["prediction"]["vehicle"]["horizon"]):
        raise ValueError("frozen popularity GRU horizon differs from formal prediction horizon")
    if int(gru_metadata["L_rho"]) != history_length:
        raise ValueError("frozen popularity GRU history window differs from formal config")
    sumo_home, random_trips, sumo_binary = _sumo_paths(sumo_home_arg)
    network_sha = _sha256_file(network_path)
    return cfg, {
        "protocol": protocol,
        "k_max_slots": k_max,
        "warmup_slots": warmup_slots,
        "popularity_history_length": history_length,
        "gru_metadata": {
            key: gru_metadata[key]
            for key in ("F", "H_p", "L_rho", "hidden_size", "observation_definition")
        },
        "vehicle_horizon_slots": int(cfg["prediction"]["vehicle"]["horizon"]),
        "dt_s": dt_s,
        "warmup_time_s": warmup_s,
        "random_trips_period_s": random_trips_period_s,
        "network_path": network_path,
        "network_sha256": network_sha,
        "gru_path": gru_path,
        "gru_sha256": _sha256_file(gru_path),
        "residual_path": residual_path,
        "residual_sha256": _sha256_file(residual_path),
        "sumo_home": sumo_home,
        "sumo_binary": sumo_binary,
        "random_trips": random_trips,
        "random_trips_sha256": _sha256_file(random_trips),
    }


def _prepare_generation_config(
    source_config: Path,
    staged_catalog: Path,
    output_path: Path,
) -> None:
    text = source_config.read_text(encoding="utf-8")
    pattern = re.compile(r'(?m)^(\s*service_catalog\s*=\s*)"[^"]*"\s*$')
    updated, replacements = pattern.subn(
        lambda match: f'{match.group(1)}"{staged_catalog.resolve().as_posix()}"',
        text,
    )
    if replacements != 1:
        raise ValueError("could not create an unambiguous staged config for the shared catalog")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(updated)


def _plan_payload(cfg: dict[str, Any], inputs: dict[str, Any], seed_plan: dict[str, list[int]]) -> dict[str, Any]:
    config_path = PROJECT_ROOT / CONFIG_RELATIVE
    task_cfg = task_scenario.task_config_from_project_config(cfg)
    catalog_seed = int(project_config.get_config_value(cfg, "service.generation.catalog_seed"))
    return {
        "schema_version": PLAN_SCHEMA,
        "config_path": CONFIG_RELATIVE.as_posix(),
        "config_sha256": _sha256_file(config_path),
        "network_sha256": inputs["network_sha256"],
        "gru_sha256": inputs["gru_sha256"],
        "residual_bank_sha256": inputs["residual_sha256"],
        "randomTrips_sha256": inputs["random_trips_sha256"],
        "time_contract": {
            "dt_s": float(cfg["simulation"]["dt_s"]),
            "warmup_time_s": float(cfg["simulation"]["warmup_time_s"]),
            "random_trips_period_s": float(inputs["random_trips_period_s"]),
            "random_trips_departure_begin_s": float(
                project_config.get_config_value(cfg, "simulation.random_trips_departure_begin_s")
            ),
            "random_trips_departure_end_s": float(
                project_config.get_config_value(cfg, "simulation.random_trips_departure_end_s")
            ),
            "arrival_horizon_slots": int(inputs["protocol"]["arrival_horizon_slots"]),
            "execution_horizon_slots": int(inputs["protocol"]["execution_horizon_slots"]),
            "k_max_slots": int(inputs["k_max_slots"]),
            "independent_episode_duration_s": 60.0 + 300.0 + 14.0,
            "random_trips_candidates_per_episode": generate_continuous_sumo_dataset.nominal_departure_count_for_interval(
                0.0, 60.0, float(inputs["random_trips_period_s"])
            ),
            "random_trips_scope": "candidate departures only in absolute interval [0,60) per Episode",
        },
        "split_counts": dict(SPLIT_COUNTS),
        "split_seed_start": dict(SEED_STARTS),
        "episode_seeds": seed_plan,
        "service_catalog_seed": catalog_seed,
        "service_catalog_contract": {
            "service_count": task_cfg.num_services,
            "service_size_gb": [task_cfg.service_size_min_gb, task_cfg.service_size_max_gb],
            "input_size_kb": [task_cfg.input_size_min_kb, task_cfg.input_size_max_kb],
            "cycles_per_bit": [task_cfg.cycles_per_bit_min, task_cfg.cycles_per_bit_max],
        },
        "vehicle_counts": {"research": EXPECTED_RESEARCH, "SV": EXPECTED_SV, "TV": EXPECTED_TV},
        "generation_source_sha256": {
            name: _sha256_file(PROJECT_ROOT / name)
            for name in (
                "scripts/generation/generate_formal_vehicle_environment.py",
                "scripts/generation/generate_episode_pool.py",
                "scripts/generation/generate_episode_realization.py",
                "src/scenario/task_scenario.py",
                "src/scenario/popularity_prediction.py",
                "src/scenario/scenario_generator.py",
                "src/scenario/sumo_scenario.py",
            )
        },
    }


def _split_manifest_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": SPLIT_SCHEMA,
        "dataset_name": "formal_vehicle_environment",
        "dataset_status": "planned",
        "plan_fingerprint": _canonical_sha256(plan),
        "config_path": plan["config_path"],
        "config_sha256": plan["config_sha256"],
        "sumo_network_sha256": plan["network_sha256"],
        "service_catalog_path": POOL_RELATIVES["train"].joinpath("service_catalog_500_1000.json").as_posix(),
        "service_catalog_seed": plan["service_catalog_seed"],
        "time_contract": plan["time_contract"],
        "vehicle_counts": plan["vehicle_counts"],
        "generation_source_sha256": plan["generation_source_sha256"],
        "slots_per_realization": 300,
    }
    for split in ("train", "validation", "test"):
        seeds = plan["episode_seeds"][split]
        result[split] = {
            "seed_start": seeds[0],
            "seed_end": seeds[-1],
            "count": len(seeds),
            "active_episode_count": len(seeds),
            "active_episode_seeds": seeds,
            "dataset_status": "planned",
        }
    for split in ("train", "validation", "test"):
        result[split]["traffic_design"] = {
            "mode": "independent_sumo_episode_per_seed",
            "warmup_time_s": plan["time_contract"]["warmup_time_s"],
            "episode_duration_s": plan["time_contract"]["independent_episode_duration_s"],
            "candidate_departure_period_s": plan["time_contract"]["random_trips_period_s"],
            "candidate_departure_begin_s": plan["time_contract"]["random_trips_departure_begin_s"],
            "candidate_departure_end_s": plan["time_contract"]["random_trips_departure_end_s"],
            "arrival_slots": 300,
            "execution_slots": 314,
            "active_episode_seeds": plan["episode_seeds"][split],
        }
    return result


def _get_or_create_stage(
    plan: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    fingerprint = _canonical_sha256(plan)
    manifest_path = STAGING_ROOT / "data_split_manifest.json"
    if STAGING_ROOT.exists():
        if not manifest_path.is_file():
            raise RuntimeError(f"staging directory has no plan manifest; refusing to reuse: {STAGING_ROOT}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("plan_fingerprint") == fingerprint:
            if not resume:
                raise FileExistsError(
                    f"staging directory already contains this plan; pass --resume to continue: {STAGING_ROOT}"
                )
            status_path = STAGING_ROOT / "generation_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
            if status.get("status") == "failed":
                raise RuntimeError(
                    "the previous generation attempt is marked failed and will not be resumed automatically; "
                    f"failure={status.get('failure')!r}"
                )
            return existing
        if resume:
            raise RuntimeError("staging manifest identity differs from current config or source inputs")

        # The only mismatched stage that may be repurposed is the known failed
        # continuous-Train attempt. Keep its raw SUMO source untouched until the
        # new independent dataset has passed every final validation.
        old_status_path = STAGING_ROOT / "generation_status.json"
        if not old_status_path.is_file():
            raise RuntimeError("mismatched staging has no generation status; refusing to repurpose it")
        old_status = json.loads(old_status_path.read_text(encoding="utf-8"))
        if (
            old_status.get("status") == "failed"
            and existing.get("train", {}).get("traffic_design", {}).get("mode")
            == "independent_sumo_episode_per_seed"
        ):
            retry_evidence = _verify_failed_pilot_for_code_retry(
                existing, old_status, plan
            )
            pilot_dir = STAGING_ROOT / "train" / f"episode_{SEED_STARTS['train']}"
            resolved_pilot_dir = pilot_dir.resolve(strict=True)
            resolved_stage = STAGING_ROOT.resolve(strict=True)
            if (
                pilot_dir.is_symlink()
                or resolved_pilot_dir.parent != (resolved_stage / "train").resolve(strict=True)
            ):
                raise RuntimeError(
                    f"failed pilot output is not the exact managed staging path: {pilot_dir}"
                )
            shutil.rmtree(resolved_pilot_dir)
            current_plan = _split_manifest_from_plan(plan)
            if isinstance(existing.get("previous_failed_stage"), dict):
                current_plan["previous_failed_stage"] = existing[
                    "previous_failed_stage"
                ]
            current_plan["previous_failed_pilot_retry"] = retry_evidence
            _write_json(manifest_path, current_plan)
            _write_json(
                STAGING_ROOT / "generation_status.json",
                {
                    "status": "preflight",
                    "planned": dict(SPLIT_COUNTS),
                    "completed": {split: 0 for split in SPLIT_COUNTS},
                    "failure": None,
                    "previous_failed_stage": current_plan.get("previous_failed_stage"),
                    "previous_failed_pilot_retry": retry_evidence,
                    "plan_fingerprint": fingerprint,
                    "updated_at_utc": _utc_now(),
                },
            )
            return current_plan
        if (
            old_status.get("status") != "failed"
            or old_status.get("completed") != {"test": 0, "train": 0, "validation": 0}
            or existing.get("train", {}).get("traffic_design", {}).get("mode")
            != "single_continuous_sumo_stream"
        ):
            raise RuntimeError(
                "mismatched staging is not the verified failed continuous-Train attempt; "
                "refusing to overwrite its identity"
            )
        unexpected_episode_data: list[str] = []
        empty_episode_dirs: list[Path] = []
        for split in ("train", "validation", "test"):
            split_dir = STAGING_ROOT / split
            if not split_dir.is_dir():
                continue
            for episode_dir in split_dir.glob("episode_*"):
                if not episode_dir.is_dir():
                    unexpected_episode_data.append(str(episode_dir))
                elif any(episode_dir.iterdir()):
                    unexpected_episode_data.append(str(episode_dir))
                else:
                    empty_episode_dirs.append(episode_dir)
        if unexpected_episode_data:
            raise RuntimeError(
                "mismatched failed staging contains non-empty or unrecognized Episode outputs; "
                "refusing to overwrite: " + ", ".join(unexpected_episode_data)
            )

        cfg = _load_config(PROJECT_ROOT / CONFIG_RELATIVE)
        catalog_seed = int(project_config.get_config_value(cfg, "service.generation.catalog_seed"))
        expected_catalog = task_scenario.make_shared_service_catalog_output(
            task_scenario.task_config_from_project_config(cfg), catalog_seed
        )
        staged_catalog = STAGING_ROOT / "train" / "service_catalog_500_1000.json"
        if not staged_catalog.is_file():
            raise RuntimeError("failed continuous stage is missing its generated service catalog")
        actual_catalog = json.loads(staged_catalog.read_text(encoding="utf-8"))
        _validate_catalog(cfg, actual_catalog, catalog_seed)
        if actual_catalog != expected_catalog:
            raise RuntimeError("staged service catalog identity differs from current formal parameters")

        for episode_dir in empty_episode_dirs:
            episode_dir.rmdir()
        staged_config = STAGING_ROOT / "generation_config.toml"
        if staged_config.exists():
            if not staged_config.is_file():
                raise RuntimeError("staged generation_config.toml path is not a file")
            text = staged_config.read_text(encoding="utf-8")
            if "service_catalog_500_1000.json" not in text:
                raise RuntimeError("staged generation config is not the generator-owned catalog config")
            staged_config.unlink()

        current_plan = _split_manifest_from_plan(plan)
        current_plan["previous_failed_stage"] = {
            "plan_fingerprint": existing.get("plan_fingerprint"),
            "status": old_status.get("status"),
            "failure": old_status.get("failure"),
            "completed": old_status.get("completed"),
            "preserved_legacy_source_path": "train/continuous_source",
        }
        _write_json(manifest_path, current_plan)
        _write_json(
            STAGING_ROOT / "generation_status.json",
            {
                "status": "preflight",
                "planned": dict(SPLIT_COUNTS),
                "completed": {split: 0 for split in SPLIT_COUNTS},
                "failure": None,
                "previous_failed_stage": current_plan["previous_failed_stage"],
                "plan_fingerprint": fingerprint,
                "updated_at_utc": _utc_now(),
            },
        )
        return current_plan
    if resume:
        raise FileNotFoundError(f"--resume requested but no staging directory exists: {STAGING_ROOT}")
    STAGING_ROOT.mkdir(parents=True, exist_ok=False)
    for split in ("train", "validation", "test"):
        (STAGING_ROOT / split).mkdir()
    manifest = _split_manifest_from_plan(plan)
    _write_json(manifest_path, manifest, exclusive=True)
    return manifest


def _verify_failed_pilot_for_code_retry(
    existing_plan: dict[str, Any],
    status: dict[str, Any],
    current_plan: dict[str, Any],
) -> dict[str, Any]:
    """Validate the known seed-20001 time-contract failure before retrying it."""
    failure = status.get("failure")
    current_episode = status.get("current_episode")
    if (
        status.get("completed") != {"test": 0, "train": 0, "validation": 0}
        or not isinstance(current_episode, dict)
        or current_episode.get("split") != "train"
        or int(current_episode.get("seed", -1)) != SEED_STARTS["train"]
        or not isinstance(failure, dict)
        or "frozen vehicle/scenario generation failed" not in str(failure.get("message", ""))
    ):
        raise RuntimeError(
            "failed independent-stage retry is not the known seed-20001 vehicle-prediction contract failure"
        )
    saved_hashes = existing_plan.get("generation_source_sha256")
    current_hashes = current_plan.get("generation_source_sha256")
    if not isinstance(saved_hashes, dict) or not isinstance(current_hashes, dict):
        raise RuntimeError("failed pilot plan has no source identities")
    changed_sources = sorted(
        name for name in current_hashes if current_hashes.get(name) != saved_hashes.get(name)
    )
    if changed_sources not in (
        ["src/scenario/scenario_generator.py"],
        [
            "scripts/generation/generate_formal_vehicle_environment.py",
            "src/scenario/scenario_generator.py",
        ],
    ):
        raise RuntimeError(
            "failed pilot retry permits only the verified scenario-generator contract fix "
            "and its guarded staging-resume logic; "
            f"changed_sources={changed_sources}"
        )
    historical_plan = dict(current_plan)
    historical_plan["generation_source_sha256"] = saved_hashes
    if _canonical_sha256(historical_plan) != existing_plan.get("plan_fingerprint"):
        raise RuntimeError("failed pilot staging differs from the current plan beyond the verified source edit")

    pilot_dir = STAGING_ROOT / "train" / f"episode_{SEED_STARTS['train']}"
    if not pilot_dir.is_dir() or pilot_dir.is_symlink():
        raise RuntimeError(f"failed pilot directory is missing or unsafe: {pilot_dir}")
    expected_files = {
        "popularity.csv",
        "popularity_history.csv",
        "popularity_predictions.csv",
        "sumo_demand.meta.json",
        "sumo_demand.rou.xml",
        "tasks.csv",
        "trajectory.csv",
        "vehicle_roles.json",
    }
    actual_files = {path.name for path in pilot_dir.iterdir() if path.is_file()}
    if actual_files != expected_files or (pilot_dir / "episode_manifest.json").exists():
        raise RuntimeError(
            "failed pilot has an unexpected partial artifact inventory; refusing cleanup: "
            f"files={sorted(actual_files)}"
        )
    roles = json.loads((pilot_dir / "vehicle_roles.json").read_text(encoding="utf-8"))
    if (
        int(roles.get("complete_candidate_count", 0)) < EXPECTED_RESEARCH
        or len(roles.get("research_vehicles", [])) != EXPECTED_RESEARCH
        or len(roles.get("service_vehicles", [])) != EXPECTED_SV
        or len(roles.get("task_vehicles", [])) != EXPECTED_TV
        or int(roles.get("arrival_horizon_slots", -1)) != 300
        or int(roles.get("execution_horizon_slots", -1)) != 314
        or any(
            int(roles.get("planned_routes", {}).get(vehicle_id, {}).get("formal_record_count", 0))
            != 314
            or int(roles.get("planned_routes", {}).get(vehicle_id, {}).get("preformal_record_count", 0))
            <= 0
            for vehicle_id in roles.get("research_vehicles", [])
        )
    ):
        raise RuntimeError("failed pilot does not prove the required 25 complete vehicles and role split")
    config_file = STAGING_ROOT / "generation_config.toml"
    cfg = generate_episode_realization.load_and_validate_config(config_file)
    streams = derive_episode_seeds(SEED_STARTS["train"])
    sim_config = generate_episode_realization.build_sumo_config(cfg, streams)
    sim_config = sumo_scenario.ScenarioConfig(
        **{
            **sim_config.__dict__,
            "episode_time_s": 314.0 * float(sim_config.dt_s),
        }
    )
    sumo_scenario.validate_outputs(
        pilot_dir / "trajectory.csv",
        pilot_dir / "vehicle_roles.json",
        list(roles["research_vehicles"]),
        sim_config,
    )
    demand_metadata = json.loads(
        (pilot_dir / "sumo_demand.meta.json").read_text(encoding="utf-8")
    )
    if (
        float(demand_metadata.get("period_s", -1)) != 1.0
        or float(demand_metadata.get("departure_begin_s", -1)) != 0.0
        or float(demand_metadata.get("departure_end_s", -1)) != 60.0
    ):
        raise RuntimeError("failed pilot randomTrips metadata differs from the approved candidate window")
    with (pilot_dir / "tasks.csv").open("r", encoding="utf-8", newline="") as stream:
        task_rows = list(csv.DictReader(stream))
    if (
        len(task_rows) != EXPECTED_TV * 300
        or {int(row["slot"]) for row in task_rows} != set(range(300))
    ):
        raise RuntimeError("failed pilot task file does not contain exactly arrival slots 0..299")
    with (pilot_dir / "popularity_predictions.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        prediction_rows = list(csv.DictReader(stream))
    if {int(row["origin_slot"]) for row in prediction_rows} != set(range(314)):
        raise RuntimeError("failed pilot popularity predictions do not cover origins 0..313")
    return {
        "plan_fingerprint": existing_plan.get("plan_fingerprint"),
        "failure": failure,
        "retry_reason": (
            "scenario generator execution-contract and deterministic replay arguments "
            "were corrected for 300 arrivals and 314 execution origins"
        ),
        "verified_complete_candidate_count": int(roles["complete_candidate_count"]),
        "verified_research_count": len(roles["research_vehicles"]),
        "verified_SV_count": len(roles["service_vehicles"]),
        "verified_TV_count": len(roles["task_vehicles"]),
        "cleaned_partial_path": "train/episode_20001",
        "changed_sources": changed_sources,
    }


def _load_stage_for_completed_train_reuse(
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Accept the existing stage only when its sole plan change is this window-builder edit."""
    manifest_path = STAGING_ROOT / "data_split_manifest.json"
    if not STAGING_ROOT.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(
            "completed Train-source reuse requires the existing staging manifest: "
            f"{manifest_path}"
        )
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved_hashes = existing.get("generation_source_sha256")
    current_hashes = plan.get("generation_source_sha256")
    if not isinstance(saved_hashes, dict) or not isinstance(current_hashes, dict):
        raise RuntimeError("staging manifest is missing generation source identities")
    if set(saved_hashes) != set(current_hashes):
        raise RuntimeError("staging manifest generation source set differs from the current plan")
    changed_sources = sorted(
        name for name in current_hashes if current_hashes[name] != saved_hashes[name]
    )
    if changed_sources != [FORMAL_GENERATOR_SOURCE_RELATIVE]:
        raise RuntimeError(
            "completed Train-source reuse permits only the formal window-builder source to change; "
            f"changed_sources={changed_sources}"
        )

    historical_plan = dict(plan)
    historical_plan["generation_source_sha256"] = saved_hashes
    saved_fingerprint = existing.get("plan_fingerprint")
    if _canonical_sha256(historical_plan) != saved_fingerprint:
        raise RuntimeError(
            "staging manifest does not match the current config, data identities, or time contract"
        )

    expected = _split_manifest_from_plan(plan)
    identity_fields = (
        "schema_version",
        "dataset_name",
        "config_path",
        "config_sha256",
        "sumo_network_sha256",
        "service_catalog_path",
        "service_catalog_seed",
        "time_contract",
        "vehicle_counts",
        "slots_per_realization",
    )
    mismatches = [
        key for key in identity_fields if existing.get(key) != expected.get(key)
    ]
    saved_non_builder_hashes = {
        name: value for name, value in saved_hashes.items()
        if name != FORMAL_GENERATOR_SOURCE_RELATIVE
    }
    current_non_builder_hashes = {
        name: value for name, value in current_hashes.items()
        if name != FORMAL_GENERATOR_SOURCE_RELATIVE
    }
    if saved_non_builder_hashes != current_non_builder_hashes:
        mismatches.append("generation_source_sha256")
    for split in ("train", "validation", "test"):
        saved_split = existing.get(split)
        expected_split = expected.get(split)
        split_fields = (
            "seed_start",
            "seed_end",
            "count",
            "active_episode_count",
            "active_episode_seeds",
            "traffic_design",
        )
        if not isinstance(saved_split, dict) or not isinstance(expected_split, dict):
            mismatches.append(split)
            continue
        if any(saved_split.get(key) != expected_split.get(key) for key in split_fields):
            mismatches.append(split)
    if mismatches:
        raise RuntimeError(
            "staging manifest identity differs from the completed Train-source reuse plan: "
            + ", ".join(mismatches)
        )
    if existing.get("dataset_status") not in {"planned", "generating"}:
        raise RuntimeError(
            "staging manifest is not in a reusable pre-publication state: "
            f"{existing.get('dataset_status')!r}"
        )
    return existing


def _load_lane_to_edge_map(network_path: Path, sumo_home: Path) -> dict[str, str]:
    """Load exact lane ownership from the formal SUMO network topology."""
    sumolib, _ = sumo_scenario.load_sumo_modules(sumo_home)
    network = sumolib.net.readNet(str(network_path), withInternal=True)
    lane_to_edge: dict[str, str] = {}
    for edge in network.getEdges(withInternal=True):
        edge_id = str(edge.getID())
        for lane in edge.getLanes():
            lane_id = str(lane.getID())
            if lane_id in lane_to_edge:
                raise RuntimeError(
                    "formal SUMO network has a non-unique lane ID; refusing edge recovery: "
                    f"lane_id={lane_id!r}, first_edge={lane_to_edge[lane_id]!r}, "
                    f"second_edge={edge_id!r}"
                )
            lane_to_edge[lane_id] = edge_id
    if not lane_to_edge:
        raise RuntimeError(f"formal SUMO network contains no lanes: {network_path}")
    return lane_to_edge


def _validate_completed_train_source(
    *,
    continuous_root: Path,
    continuous_manifest: dict[str, Any],
    inputs: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Verify the completed continuous source before using it for window extraction."""
    if continuous_manifest.get("status") != "complete":
        raise RuntimeError("continuous Train source is not complete; reuse mode will not start SUMO")
    if continuous_manifest.get("network_sha256") != inputs["network_sha256"]:
        raise RuntimeError("completed continuous source uses a different SUMO network")
    staged_config_path = STAGING_ROOT / "generation_config.toml"
    staged_catalog_path = STAGING_ROOT / "train" / "service_catalog_500_1000.json"
    if not staged_config_path.is_file() or not staged_catalog_path.is_file():
        raise RuntimeError("completed continuous source is missing its staged config or service catalog")
    config_text = (PROJECT_ROOT / CONFIG_RELATIVE).read_text(encoding="utf-8")
    catalog_pattern = re.compile(r'(?m)^(\s*service_catalog\s*=\s*)"[^"]*"\s*$')
    expected_staged_config, config_replacements = catalog_pattern.subn(
        lambda match: f'{match.group(1)}"{staged_catalog_path.resolve().as_posix()}"',
        config_text,
    )
    if config_replacements != 1 or staged_config_path.read_text(encoding="utf-8") != expected_staged_config:
        raise RuntimeError("staged generation config differs from the current formal config/catalog path")
    if continuous_manifest.get("config_sha256") != _sha256_file(staged_config_path):
        raise RuntimeError("completed continuous source uses a different staged generation config")
    if continuous_manifest.get("randomTrips_sha256") != inputs["random_trips_sha256"]:
        raise RuntimeError("completed continuous source uses a different randomTrips.py")
    if continuous_manifest.get("generator_sha256") != _sha256_file(
        PROJECT_ROOT / "scripts/generation/generate_continuous_sumo_dataset.py"
    ):
        raise RuntimeError("completed continuous source generator identity differs from the current code")
    staged_manifest_path = STAGING_ROOT / "data_split_manifest.json"
    if continuous_manifest.get("source_split_manifest_sha256") != _sha256_file(staged_manifest_path):
        raise RuntimeError("completed continuous source is not bound to the current staging manifest")

    episode_records = continuous_manifest.get("episodes")
    episode_record = (
        episode_records.get("train/continuous_seed_20001")
        if isinstance(episode_records, dict)
        else None
    )
    if not isinstance(episode_record, dict) or episode_record.get("status") != "complete":
        raise RuntimeError("continuous source manifest has no completed Train stream entry")
    stream_dir = continuous_root / "train" / "continuous_seed_20001"
    episode_manifest_path = stream_dir / "episode_manifest.json"
    if not episode_manifest_path.is_file() or _sha256_file(episode_manifest_path) != episode_record.get(
        "episode_manifest_sha256"
    ):
        raise RuntimeError("continuous Train stream manifest is missing or its hash differs")
    episode_manifest = json.loads(episode_manifest_path.read_text(encoding="utf-8"))
    artifact_hashes = episode_manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        raise RuntimeError("continuous Train stream manifest has no artifact hashes")
    fcd_path = stream_dir / "trajectory.fcd.xml.gz"
    route_path = stream_dir / "sumo_demand.rou.xml"
    for path in (fcd_path, route_path):
        expected_hash = artifact_hashes.get(path.name)
        if not path.is_file() or not expected_hash or _sha256_file(path) != expected_hash:
            raise RuntimeError(
                "completed continuous source artifact is missing or has a hash mismatch: "
                f"{path}"
            )
    random_trips_contract = episode_manifest.get("randomTrips")
    source_period = (
        random_trips_contract.get("period_s")
        if isinstance(random_trips_contract, dict)
        else None
    )
    if source_period is None:
        raise RuntimeError("completed continuous source has no recorded randomTrips period")
    if not math.isclose(
        float(source_period),
        float(inputs["random_trips_period_s"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("completed continuous source departure interval differs from the current plan")
    return fcd_path, route_path, episode_manifest_path


def _load_route_cache(route_path: Path) -> dict[str, dict[str, Any]]:
    definitions: dict[str, tuple[str, ...]] = {}
    inline_by_vehicle: dict[str, tuple[str, tuple[str, ...]]] = {}
    vehicle_refs: dict[str, str] = {}
    stack: list[ET.Element] = []
    for event, element in ET.iterparse(route_path, events=("start", "end")):
        if event == "start":
            if element.tag == "vehicle":
                vehicle_id = str(element.attrib["id"])
                reference = element.attrib.get("route")
                if reference:
                    vehicle_refs[vehicle_id] = reference
            stack.append(element)
            continue
        if element.tag == "route":
            route_id = str(element.attrib.get("id", ""))
            edges = tuple(str(element.attrib.get("edges", "")).split())
            if route_id and edges:
                definitions[route_id] = edges
            if len(stack) >= 2 and stack[-2].tag == "vehicle":
                vehicle_id = str(stack[-2].attrib.get("id", ""))
                if vehicle_id and edges:
                    inline_by_vehicle[vehicle_id] = (route_id, edges)
            element.clear()
        elif element.tag == "vehicle":
            element.clear()
        if stack:
            stack.pop()
    result: dict[str, dict[str, Any]] = {}
    vehicle_ids = set(vehicle_refs) | set(inline_by_vehicle)
    for vehicle_id in vehicle_ids:
        if vehicle_id in inline_by_vehicle:
            route_id, edges = inline_by_vehicle[vehicle_id]
        else:
            route_id = vehicle_refs[vehicle_id]
            edges = definitions.get(route_id, ())
        if edges:
            result[vehicle_id] = {"route_id": route_id, "route_edges": tuple(edges)}
    if not result:
        raise ValueError(f"no vehicle routes could be read from continuous SUMO demand: {route_path}")
    return result


def _iter_continuous_windows(
    fcd_path: Path,
    route_cache: dict[str, dict[str, Any]],
    *,
    lane_to_edge: Mapping[str, str],
    dt_s: float,
    warmup_s: float,
    arrival_slots: int,
    execution_slots: int,
    window_interval_s: float,
    skip_windows: set[int],
) -> Iterator[tuple[int, list[dict[str, Any]], set[str]]]:
    window_count = SPLIT_COUNTS["train"]
    interval_steps = int(round(window_interval_s / dt_s))
    window_steps = int(round((warmup_s / dt_s) + execution_slots))
    expected_steps = int(round((warmup_s + arrival_slots * window_count * dt_s + (execution_slots - arrival_slots) * dt_s) / dt_s))
    buckets: dict[int, list[dict[str, Any]]] = {}
    seen: dict[int, set[str]] = {}
    last_finalized = -1
    time_count = 0
    current_step: int | None = None
    with gzip.open(fcd_path, "rt", encoding="utf-8") as stream:
        for event, element in ET.iterparse(stream, events=("start", "end")):
            if event == "start" and element.tag == "timestep":
                time_s = float(element.attrib["time"])
                current_step = int(round(time_s / dt_s))
                if not math.isclose(time_s, current_step * dt_s, rel_tol=0.0, abs_tol=1e-6):
                    raise RuntimeError(f"continuous FCD timestamp is off-grid: {time_s}")
                if current_step != time_count:
                    raise RuntimeError(f"continuous FCD has a missing/duplicate timestamp at step {current_step}; expected={time_count}")
                time_count += 1
            elif event == "end" and element.tag == "vehicle":
                if current_step is None:
                    raise RuntimeError("continuous FCD vehicle row is outside a timestep")
                absolute_time_s = current_step * dt_s
                first_window = max(0, (current_step - (window_steps - 1) + interval_steps - 1) // interval_steps)
                last_window = min(window_count - 1, current_step // interval_steps)
                vehicle_id = str(element.attrib["id"])
                route = route_cache.get(vehicle_id, {})
                for window_index in range(first_window, last_window + 1):
                    if window_index in skip_windows:
                        continue
                    bucket = buckets.setdefault(window_index, [])
                    seen.setdefault(window_index, set()).add(vehicle_id)
                    local_time_s = absolute_time_s - window_index * window_interval_s
                    row: dict[str, Any] = {"time_s": float(local_time_s), "vehicle_id": vehicle_id}
                    for source_name, output_name in FCD_ATTRIBUTES.items():
                        if source_name == "id":
                            continue
                        value = element.attrib.get(source_name)
                        if source_name == "edge" and value is None:
                            lane_id = element.attrib.get("lane")
                            if lane_id is None:
                                raise RuntimeError(
                                    "continuous FCD has neither edge nor lane for edge recovery: "
                                    f"vehicle_id={vehicle_id!r}, time_s={absolute_time_s:g}"
                                )
                            value = lane_to_edge.get(lane_id)
                            if value is None:
                                raise RuntimeError(
                                    "continuous FCD lane cannot be mapped through the formal SUMO network: "
                                    f"vehicle_id={vehicle_id!r}, time_s={absolute_time_s:g}, "
                                    f"lane_id={lane_id!r}"
                                )
                        if value is None:
                            raise RuntimeError(
                                "continuous FCD is missing required attribute "
                                f"{source_name!r}: vehicle_id={vehicle_id!r}, "
                                f"time_s={absolute_time_s:g}"
                            )
                        if output_name in {"road_id", "lane_id"}:
                            row[output_name] = value
                        else:
                            row[output_name] = float(value)
                    row["route_id"] = str(route.get("route_id", ""))
                    row["route_edges"] = " ".join(route.get("route_edges", ()))
                    buckets[window_index].append(row)
                element.clear()
            elif event == "end" and element.tag == "timestep":
                if current_step is None:
                    raise RuntimeError("continuous FCD timestep closed without opening")
                for window_index in range(window_count):
                    end_exclusive = window_index * interval_steps + window_steps
                    if end_exclusive - 1 == current_step:
                        if window_index != last_finalized + 1:
                            raise RuntimeError("continuous FCD window finalization order is inconsistent")
                        last_finalized = window_index
                        if window_index not in skip_windows:
                            yield (
                                window_index,
                                buckets.pop(window_index, []),
                                seen.pop(window_index, set()),
                            )
                        else:
                            buckets.pop(window_index, None)
                            seen.pop(window_index, None)
                current_step = None
                element.clear()
    if current_step is not None or time_count != expected_steps or last_finalized != window_count - 1:
        raise RuntimeError(
            "continuous FCD did not cover the 300 complete train windows: "
            f"steps={time_count}/{expected_steps}, last_window={last_finalized}"
        )
    if buckets or seen:
        raise RuntimeError("continuous FCD window extraction left unfinalized data")


def _write_window_episode(
    *,
    config_path: Path,
    seed: int,
    destination: Path,
    rows: list[dict[str, Any]],
    route_cache: dict[str, dict[str, Any]],
    seen_ids: set[str],
    inputs: dict[str, Any],
    window_index: int,
    fcd_sha256: str,
    route_sha256: str,
    continuous_manifest_sha256: str,
    device: str,
) -> None:
    source = {
        "schema_version": "continuous_train_window_source_v1",
        "continuous_episode_seed": SEED_STARTS["train"],
        "window_index": int(window_index),
        "window_interval_s": TRAFFIC_WINDOW_S,
        "window_absolute_start_s": int(window_index) * TRAFFIC_WINDOW_S,
        "preformal_history_start_s": int(window_index) * TRAFFIC_WINDOW_S,
        "formal_start_s": int(window_index) * TRAFFIC_WINDOW_S + 60.0,
        "physical_end_exclusive_s": int(window_index) * TRAFFIC_WINDOW_S + 60.0 + 314.0,
        "physical_trajectory_slots": 314,
        "fcd_relative_path": "continuous_source/train/continuous_seed_20001/trajectory.fcd.xml.gz",
        "fcd_sha256": fcd_sha256,
        "route_relative_path": "continuous_source/train/continuous_seed_20001/sumo_demand.rou.xml",
        "route_sha256": route_sha256,
        "continuous_episode_manifest_sha256": continuous_manifest_sha256,
        "per_window_sumo_rerun": False,
        "per_window_sumo_seed_used": False,
    }
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(f"episode output path is not a directory: {destination}")
        manifest_path = destination / "episode_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"incomplete Train window exists; refusing to overwrite: {destination}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("traffic_source_provenance") != source:
            raise RuntimeError(f"Train window identity mismatch; refusing to skip or overwrite: {destination}")
        generate_episode_realization.validate_written_outputs(
            destination,
            task_vehicle_count=EXPECTED_TV,
            research_vehicle_count=EXPECTED_RESEARCH,
            horizon_steps=int(inputs["vehicle_horizon_slots"]),
            origin_count=int(inputs["protocol"]["execution_horizon_slots"]),
        )
        generate_episode_realization.validate_episode_provenance(
            destination,
            generate_episode_realization.load_and_validate_config(config_path),
            seed,
        )
        return
    destination.mkdir(parents=True, exist_ok=False)
    generate_episode_realization.run_episode(
        config_path,
        seed,
        destination,
        str(inputs["sumo_home"]),
        device,
        observed_trajectory_rows=rows,
        observed_route_cache=route_cache,
        observed_vehicle_ids=seen_ids,
        traffic_source_provenance=source,
        origin_count=int(inputs["protocol"]["execution_horizon_slots"]),
    )


def _generate_train_windows(
    *,
    config_path: Path,
    staging_manifest: dict[str, Any],
    inputs: dict[str, Any],
    seed_plan: dict[str, list[int]],
    device: str,
    progress: dict[str, Any],
) -> dict[str, Any]:
    continuous_root = STAGING_ROOT / "train" / "continuous_source"
    continuous_manifest = continuous_root / "dataset_manifest.json"
    args = [
        "--config", str(config_path),
        "--split-manifest", str(STAGING_ROOT / "data_split_manifest.json"),
        "--output-dir", str(continuous_root),
        "--split", "train",
    ]
    if continuous_manifest.is_file():
        existing_raw_manifest = json.loads(continuous_manifest.read_text(encoding="utf-8"))
        if existing_raw_manifest.get("status") != "complete":
            args.append("--resume")
            status = generate_continuous_sumo_dataset.main(args)
            if status != 0:
                raise RuntimeError(f"continuous Train traffic generation returned status {status}")
    else:
        status = generate_continuous_sumo_dataset.main(args)
        if status != 0:
            raise RuntimeError(f"continuous Train traffic generation returned status {status}")
    stream_dir = continuous_root / "train" / f"continuous_seed_{seed_plan['train'][0]}"
    fcd_path = stream_dir / "trajectory.fcd.xml.gz"
    route_path = stream_dir / "sumo_demand.rou.xml"
    episode_manifest_path = stream_dir / "episode_manifest.json"
    for path in (fcd_path, route_path, episode_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"continuous Train source artifact is missing: {path}")
    continuous_manifest = json.loads(continuous_manifest.read_text(encoding="utf-8"))
    if continuous_manifest.get("status") != "complete":
        raise RuntimeError("continuous Train source manifest is not complete")
    fcd_path, route_path, episode_manifest_path = _validate_completed_train_source(
        continuous_root=continuous_root,
        continuous_manifest=continuous_manifest,
        inputs=inputs,
    )
    lane_to_edge = _load_lane_to_edge_map(
        Path(inputs["network_path"]),
        Path(inputs["sumo_home"]),
    )
    print(
        "[FormalEnvironment] verified unique lane-to-edge mapping: "
        f"{len(lane_to_edge)} lanes from {inputs['network_path']}",
        flush=True,
    )
    route_cache = _load_route_cache(route_path)
    fcd_hash = _sha256_file(fcd_path)
    route_hash = _sha256_file(route_path)
    continuous_episode_manifest_hash = _sha256_file(episode_manifest_path)
    execution_slots = int(inputs["protocol"]["execution_horizon_slots"])
    completed: list[int] = []
    for seed in seed_plan["train"]:
        destination = STAGING_ROOT / "train" / f"episode_{seed}"
        if destination.is_dir() and (destination / "episode_manifest.json").is_file():
            config = generate_episode_realization.load_and_validate_config(config_path)
            manifest = json.loads((destination / "episode_manifest.json").read_text(encoding="utf-8"))
            window_index = seed - seed_plan["train"][0]
            expected_source = {
                "schema_version": "continuous_train_window_source_v1",
                "continuous_episode_seed": SEED_STARTS["train"],
                "window_index": int(window_index),
                "window_interval_s": TRAFFIC_WINDOW_S,
                "window_absolute_start_s": int(window_index) * TRAFFIC_WINDOW_S,
                "preformal_history_start_s": int(window_index) * TRAFFIC_WINDOW_S,
                "formal_start_s": int(window_index) * TRAFFIC_WINDOW_S + 60.0,
                "physical_end_exclusive_s": int(window_index) * TRAFFIC_WINDOW_S + 60.0 + 314.0,
                "physical_trajectory_slots": 314,
                "fcd_relative_path": "continuous_source/train/continuous_seed_20001/trajectory.fcd.xml.gz",
                "fcd_sha256": fcd_hash,
                "route_relative_path": "continuous_source/train/continuous_seed_20001/sumo_demand.rou.xml",
                "route_sha256": route_hash,
                "continuous_episode_manifest_sha256": continuous_episode_manifest_hash,
                "per_window_sumo_rerun": False,
                "per_window_sumo_seed_used": False,
            }
            if manifest.get("traffic_source_provenance") != expected_source:
                raise RuntimeError(f"existing Train window identity mismatch: {destination}")
            generate_episode_realization.validate_written_outputs(
                destination,
                task_vehicle_count=EXPECTED_TV,
                research_vehicle_count=EXPECTED_RESEARCH,
                horizon_steps=int(inputs["vehicle_horizon_slots"]),
                origin_count=execution_slots,
            )
            generate_episode_realization.validate_episode_provenance(destination, config, seed)
            completed.append(seed)
    skipped_windows = {index for index, seed in enumerate(seed_plan["train"]) if seed in completed}
    progress["completed"]["train"] = len(completed)
    _record_generation_progress(progress)
    generated = 0
    for window_index, rows, seen_ids in _iter_continuous_windows(
        fcd_path,
        route_cache,
        lane_to_edge=lane_to_edge,
        dt_s=float(inputs["dt_s"]),
        warmup_s=float(inputs["warmup_time_s"]),
        arrival_slots=int(inputs["protocol"]["arrival_horizon_slots"]),
        execution_slots=execution_slots,
        window_interval_s=TRAFFIC_WINDOW_S,
        skip_windows=skipped_windows,
    ):
        seed = seed_plan["train"][window_index]
        destination = STAGING_ROOT / "train" / f"episode_{seed}"
        _write_window_episode(
            config_path=config_path,
            seed=seed,
            destination=destination,
            rows=rows,
            route_cache=route_cache,
            seen_ids=seen_ids,
            inputs=inputs,
            window_index=window_index,
            fcd_sha256=fcd_hash,
            route_sha256=route_hash,
            continuous_manifest_sha256=continuous_episode_manifest_hash,
            device=device,
        )
        generated += 1
        progress["completed"]["train"] = len(completed) + generated
        _record_generation_progress(progress)
        print(f"[FormalEnvironment] train windows {len(skipped_windows) + generated}/{len(seed_plan['train'])}: seed={seed}", flush=True)
    if len(completed) + generated != len(seed_plan["train"]):
        raise RuntimeError(
            f"Train window count mismatch: existing={len(completed)}, generated={generated}, expected={len(seed_plan['train'])}"
        )
    return {
        "source_directory": "continuous_source",
        "episode_seed": seed_plan["train"][0],
        "fcd_relative_path": "continuous_source/train/continuous_seed_20001/trajectory.fcd.xml.gz",
        "fcd_sha256": fcd_hash,
        "route_relative_path": "continuous_source/train/continuous_seed_20001/sumo_demand.rou.xml",
        "route_sha256": route_hash,
        "episode_manifest_sha256": continuous_episode_manifest_hash,
        "window_count": len(seed_plan["train"]),
        "window_interval_s": TRAFFIC_WINDOW_S,
        "simulation_duration_s": 60.0 + 300.0 * 300 + 14.0,
    }


def _regenerate_execution_tail_predictions(
    *,
    config_path: Path,
    episode_dir: Path,
    seed: int,
    inputs: dict[str, Any],
    device: str,
) -> None:
    cfg = generate_episode_realization.load_and_validate_config(config_path)
    catalog_file = generate_episode_realization.config_path(cfg, "paths.service_catalog")
    gru_file = generate_episode_realization.config_path(cfg, "paths.popularity_gru")
    residual_file = generate_episode_realization.config_path(cfg, "paths.vehicle_residual_bank")
    network_file = generate_episode_realization.config_path(cfg, "paths.sumo_net")
    horizon = int(project_config.get_config_value(cfg, "prediction.vehicle.horizon"))
    task_vehicle_count = int(project_config.get_config_value(cfg, "simulation.num_task_vehicles"))
    research_vehicle_count = int(project_config.get_config_value(cfg, "simulation.num_research_vehicles"))
    roles_file = episode_dir / "vehicle_roles.json"
    popularity_file = episode_dir / "popularity.csv"
    history_file = episode_dir / "popularity_history.csv"
    popularity_prediction_file = episode_dir / "popularity_predictions.csv"
    status = popularity_prediction.main(
        [
            "--config", str(config_path),
            "--popularity-file", str(popularity_file),
            "--popularity-history-file", str(history_file),
            "--service-catalog-file", str(catalog_file),
            "--roles-file", str(roles_file),
            "--output-dir", str(episode_dir),
            "--checkpoint", str(gru_file),
            "--prediction-file", str(popularity_prediction_file),
            "--origin-count", str(inputs["protocol"]["execution_horizon_slots"]),
            "--device", device,
        ]
    )
    if status != 0:
        raise RuntimeError(f"execution-tail popularity inference failed for seed={seed}")
    status = scenario_generator.main(
        [
            "--config", str(config_path),
            "--episode-seed", str(seed),
            "--roles-file", str(roles_file),
            "--trajectory-file", str(episode_dir / "trajectory.csv"),
            "--popularity-file", str(popularity_file),
            "--popularity-history-file", str(history_file),
            "--service-catalog-file", str(catalog_file),
            "--residual-file", str(residual_file),
            "--gru-checkpoint", str(gru_file),
            "--output-file", str(episode_dir / "prediction_scenarios.npz"),
            "--vehicle-prediction-file", str(episode_dir / "vehicle_predictions.csv"),
            "--net-file", str(network_file),
            "--sumo-home", str(inputs["sumo_home"]),
            "--origin-count", str(inputs["protocol"]["execution_horizon_slots"]),
            "--device", device,
        ]
    )
    if status != 0:
        raise RuntimeError(f"execution-tail vehicle/scenario generation failed for seed={seed}")
    generate_episode_realization.validate_written_outputs(
        episode_dir,
        task_vehicle_count=task_vehicle_count,
        research_vehicle_count=research_vehicle_count,
        horizon_steps=horizon,
        origin_count=int(inputs["protocol"]["execution_horizon_slots"]),
    )
    manifest_path = episode_dir / "episode_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = sorted(
        path for path in episode_dir.iterdir()
        if path.is_file() and path.name != manifest_path.name
    )
    manifest["generated_artifacts"] = [path.name for path in artifacts]
    manifest["generated_artifact_sha256"] = {
        path.name: _sha256_file(path) for path in artifacts
    }
    manifest["prediction_tail_regeneration"] = {
        "schema_version": "causal_execution_tail_predictions_v1",
        "arrival_horizon_slots": int(inputs["protocol"]["arrival_horizon_slots"]),
        "execution_horizon_slots": int(inputs["protocol"]["execution_horizon_slots"]),
        "tail_request_counts": "known zeros under the no-new-arrival contract",
        "tail_rho_star_labels": "not generated",
        "origin_cutoff": "origin_slot - 1",
    }
    _write_json(manifest_path, manifest)
    generate_episode_realization.validate_episode_provenance(
        episode_dir,
        cfg,
        seed,
    )


def _generate_independent_split(
    *,
    split: str,
    config_path: Path,
    seed_plan: dict[str, list[int]],
    inputs: dict[str, Any],
    device: str,
    progress: dict[str, Any],
    seeds: list[int] | None = None,
) -> None:
    output_dir = STAGING_ROOT / split
    selected_seeds = seed_plan[split] if seeds is None else list(seeds)
    full_split_seeds = seed_plan[split]
    config = generate_episode_realization.load_and_validate_config(config_path)
    progress["completed"][split] = 0
    progress.setdefault("completed_episode_seeds", {})[split] = []
    _record_generation_progress(progress)
    for index, seed in enumerate(selected_seeds, start=1):
        episode_dir = output_dir / f"episode_{seed}"
        if episode_dir.exists():
            if not episode_dir.is_dir():
                raise RuntimeError(f"Episode output path is not a directory: {episode_dir}")
            existing_files = list(episode_dir.iterdir())
            if existing_files:
                manifest_path = episode_dir / "episode_manifest.json"
                if not manifest_path.is_file():
                    raise RuntimeError(
                        f"incomplete {split} Episode exists; refusing to overwrite: {episode_dir}"
                    )
                expected_source = _episode_traffic_provenance(config, split, seed)
                _validate_episode(
                    episode_dir,
                    seed=seed,
                    split=split,
                    config_path=config_path,
                    origin_count=int(inputs["protocol"]["execution_horizon_slots"]),
                )
                existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if existing_manifest.get("traffic_source_provenance") != expected_source:
                    raise RuntimeError(
                        f"existing {split} Episode has a different independent-traffic identity: {episode_dir}"
                    )
                progress["completed"][split] += 1
                progress["completed_episode_seeds"][split].append(int(seed))
                _record_generation_progress(progress)
                print(
                    f"[FormalEnvironment] {split} {index}/{len(selected_seeds)}: seed={seed} (verified resume)",
                    flush=True,
                )
                continue
            raise RuntimeError(
                f"empty Episode output directory already exists; refusing to reuse ambiguously: {episode_dir}"
            )

        progress["current_episode"] = {"split": split, "seed": int(seed)}
        _record_generation_progress(progress)
        generate_episode_pool.generate_pool(
            config_path,
            output_dir,
            seed,
            1,
            str(inputs["sumo_home"]),
            device,
            origin_count=int(inputs["protocol"]["execution_horizon_slots"]),
            split=split,
        )
        _validate_episode(
            episode_dir,
            seed=seed,
            split=split,
            config_path=config_path,
            origin_count=int(inputs["protocol"]["execution_horizon_slots"]),
        )
        progress["completed"][split] += 1
        progress["completed_episode_seeds"][split].append(int(seed))
        progress.pop("current_episode", None)
        _record_generation_progress(progress)
        print(
            f"[FormalEnvironment] {split} {index}/{len(selected_seeds)}: seed={seed}; "
            f"completed_in_split={progress['completed'][split]}/{len(full_split_seeds)}",
            flush=True,
        )


def _episode_traffic_provenance(
    config: dict[str, Any],
    split: str,
    seed: int,
) -> dict[str, Any]:
    return generate_episode_pool.independent_episode_traffic_provenance(
        config, split, seed
    )


def _validate_episode(
    episode_dir: Path,
    *,
    seed: int,
    split: str | None = None,
    config_path: Path,
    origin_count: int,
) -> str:
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"expected generated episode directory is missing: {episode_dir}")
    config = generate_episode_realization.load_and_validate_config(config_path)
    generate_episode_realization.validate_written_outputs(
        episode_dir,
        task_vehicle_count=EXPECTED_TV,
        research_vehicle_count=EXPECTED_RESEARCH,
        horizon_steps=int(project_config.get_config_value(config, "prediction.vehicle.horizon")),
        origin_count=origin_count,
    )
    generate_episode_realization.validate_episode_provenance(episode_dir, config, seed)
    manifest_path = episode_dir / "episode_manifest.json"
    episode_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if split is not None and episode_manifest.get("traffic_source_provenance") != _episode_traffic_provenance(
        config, split, seed
    ):
        raise RuntimeError(
            f"independent SUMO source identity mismatch: split={split}, seed={seed}"
        )
    time_identity = episode_manifest.get("simulation", {})
    protocol = project_config.get_episode_data_protocol(config)
    if (
        int(time_identity.get("arrival_horizon_slots", -1))
        != int(protocol["arrival_horizon_slots"])
        or int(time_identity.get("execution_horizon_slots", -1))
        != int(protocol["execution_horizon_slots"])
    ):
        raise RuntimeError(f"Episode manifest time contract mismatch: {episode_dir}")
    roles = json.loads((episode_dir / "vehicle_roles.json").read_text(encoding="utf-8"))
    if (
        int(roles.get("arrival_horizon_slots", -1))
        != int(protocol["arrival_horizon_slots"])
        or int(roles.get("execution_horizon_slots", -1))
        != int(protocol["execution_horizon_slots"])
        or int(roles.get("formal_time_slot_count", -1))
        != int(protocol["execution_horizon_slots"])
        or not math.isclose(
            float(roles.get("episode_time", math.nan)),
            int(protocol["execution_horizon_slots"])
            * float(project_config.get_config_value(config, "simulation.dt_s")),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise RuntimeError(f"vehicle role time contract mismatch: {episode_dir}")
    return _sha256_file(manifest_path)


def _task_contract(cfg: dict[str, Any], catalog_relative: str, catalog_sha256: str) -> dict[str, Any]:
    service = cfg["service"]
    generation = service["generation"]
    task = cfg["task"]
    return {
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
        "service_catalog_sha256": catalog_sha256,
    }


def _pool_identity(
    split: str,
    cfg: dict[str, Any],
    inputs: dict[str, Any],
    seeds: list[int],
    scene_hashes: dict[str, str],
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "formal_episode_dataset_identity_v1",
            "split": split,
            "network_sha256": inputs["network_sha256"],
            "slot_duration_s": float(cfg["simulation"]["dt_s"]),
            "arrival_slots": int(inputs["protocol"]["arrival_horizon_slots"]),
            "trajectory_slots": int(inputs["protocol"]["execution_horizon_slots"]),
            "active_episode_seeds": seeds,
            "active_episode_scene_sha256": scene_hashes,
        }
    )


def _write_final_manifests(
    *,
    config_path: Path,
    inputs: dict[str, Any],
    seed_plan: dict[str, list[int]],
    catalog_path: Path,
) -> dict[str, Any]:
    cfg = generate_episode_realization.load_and_validate_config(config_path)
    catalog_sha256 = _sha256_file(catalog_path)
    catalog_relative = POOL_RELATIVES["train"].joinpath(catalog_path.name).as_posix()
    scene_hashes: dict[str, dict[str, str]] = {}
    for split, seeds in seed_plan.items():
        split_hashes: dict[str, str] = {}
        for seed in seeds:
            split_hashes[str(seed)] = _validate_episode(
                STAGING_ROOT / split / f"episode_{seed}",
                seed=seed,
            split=split,
                config_path=config_path,
                origin_count=int(inputs["protocol"]["execution_horizon_slots"]),
            )
        scene_hashes[split] = split_hashes

    split_manifest = _split_manifest_from_plan(
        _plan_payload(cfg, inputs, seed_plan)
    )
    split_manifest.update(
        {
            "schema_version": SPLIT_SCHEMA,
            "dataset_status": "complete",
            "service_catalog_sha256": catalog_sha256,
            "completed_at_utc": _utc_now(),
        }
    )
    prior_stage_manifest_path = STAGING_ROOT / "data_split_manifest.json"
    if prior_stage_manifest_path.is_file():
        prior_stage_manifest = json.loads(
            prior_stage_manifest_path.read_text(encoding="utf-8")
        )
        if isinstance(prior_stage_manifest.get("previous_failed_stage"), dict):
            split_manifest["previous_failed_stage"] = prior_stage_manifest[
                "previous_failed_stage"
            ]
    for split, seeds in seed_plan.items():
        identity = _pool_identity(split, cfg, inputs, seeds, scene_hashes[split])
        split_manifest[split].update(
            {
                "dataset_status": "complete",
                "active_episode_count": len(seeds),
                "active_episode_seeds": seeds,
                "active_episode_scene_sha256": scene_hashes[split],
                "dataset_identity_sha256": identity,
                "pool_path": POOL_RELATIVES[split].as_posix(),
            }
        )
    split_manifest_path = STAGING_ROOT / "data_split_manifest.json"
    _write_json(split_manifest_path, split_manifest)
    split_manifest_sha256 = _sha256_file(split_manifest_path)

    task_contract = _task_contract(cfg, catalog_relative, catalog_sha256)
    task_contract_sha256 = _canonical_sha256(task_contract)
    generation_fingerprint = generate_episode_realization.generation_config_fingerprint(cfg)
    config_sha256 = _sha256_file(PROJECT_ROOT / CONFIG_RELATIVE)
    for split, seeds in seed_plan.items():
        pool_dir = STAGING_ROOT / split
        pool_manifest = {
            "schema_version": POOL_SCHEMA,
            "split": split,
            "seed_start": seeds[0],
            "seed_end": seeds[-1],
            "episode_count": len(seeds),
            "slots_per_episode": int(inputs["protocol"]["arrival_horizon_slots"]),
            "data_split_manifest_sha256": split_manifest_sha256,
            "dataset_identity_sha256": split_manifest[split]["dataset_identity_sha256"],
            "active_episode_seeds": seeds,
            "active_episode_scene_sha256": scene_hashes[split],
            "episode_selection": {
                "schema_version": "active_episode_seed_selection_v1",
                "source_seed_range": [seeds[0], seeds[-1]],
                "active_episode_seeds": seeds,
                "arrival_slots": int(inputs["protocol"]["arrival_horizon_slots"]),
                "trajectory_slots": int(inputs["protocol"]["execution_horizon_slots"]),
                "replacement_episode_seeds": [],
                "excluded_episode_seeds": [],
            },
            "task_data_model": "500_1000_KB",
            "task_data_contract": task_contract,
            "task_data_contract_sha256": task_contract_sha256,
            "service_catalog_path": catalog_relative,
            "service_catalog_sha256": catalog_sha256,
            "config_path": CONFIG_RELATIVE.as_posix(),
            "config_sha256": config_sha256,
            "semantic_config_sha256": config_sha256,
            "target_generation_config_fingerprint": generation_fingerprint,
            "source_provenance": {
                "traffic_generation": "independent_episode_sumo",
                "source_generator_sha256": split_manifest["generation_source_sha256"],
            },
        }
        baseline = {
            "schema_version": INTEGRITY_SCHEMA,
            "integrity_baseline_type": "ACTIVE_EPISODE_SNAPSHOT",
            "task_data_model": "500_1000_KB",
            "split": split,
            "seed_start": seeds[0],
            "seed_end": seeds[-1],
            "episode_count": len(seeds),
            "active_episode_seeds": seeds,
            "data_split_manifest_sha256": split_manifest_sha256,
            "dataset_identity_sha256": split_manifest[split]["dataset_identity_sha256"],
            "service_catalog_sha256": catalog_sha256,
            "episodes": [],
        }
        for seed in seeds:
            episode_dir = pool_dir / f"episode_{seed}"
            manifest_path = episode_dir / "episode_manifest.json"
            episode_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            baseline["episodes"].append(
                {
                    "episode_seed": seed,
                    "episode_manifest_sha256": scene_hashes[split][str(seed)],
                    "artifact_sha256": dict(episode_manifest["generated_artifact_sha256"]),
                    "artifact_file_count": len(episode_manifest["generated_artifacts"]),
                }
            )
        baseline["episode_file_count"] = sum(
            int(entry["artifact_file_count"]) + 1 for entry in baseline["episodes"]
        )
        baseline_path = pool_dir / "formal_pool_integrity_baseline.json"
        _write_json(baseline_path, baseline)
        pool_manifest["integrity_baseline_path"] = baseline_path.name
        pool_manifest["integrity_baseline_sha256"] = _sha256_file(baseline_path)
        pool_manifest["episode_file_count"] = baseline["episode_file_count"]
        _write_json(pool_dir / "formal_pool_manifest.json", pool_manifest)
    return split_manifest


def _publish() -> None:
    destinations = {split: PROJECT_ROOT / relative for split, relative in POOL_RELATIVES.items()}
    central_destination = PROJECT_ROOT / SPLIT_MANIFEST_RELATIVE
    conflicts = [path for path in (*destinations.values(), central_destination) if path.exists()]
    if conflicts:
        raise FileExistsError(
            "formal output path already exists; refusing to replace non-empty or identity-unknown data: "
            + ", ".join(str(path) for path in conflicts)
        )
    stage_manifest_path = STAGING_ROOT / "data_split_manifest.json"
    if not stage_manifest_path.is_file():
        raise FileNotFoundError(f"staged split manifest is missing: {stage_manifest_path}")
    stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    previous_failure = stage_manifest.get("previous_failed_stage")
    legacy_source = STAGING_ROOT / "train" / "continuous_source"
    retained_legacy_source = STAGING_ROOT / "legacy_failed_continuous_source"
    if legacy_source.exists():
        if (
            not isinstance(previous_failure, dict)
            or previous_failure.get("status") != "failed"
            or previous_failure.get("completed") != {"test": 0, "train": 0, "validation": 0}
            or previous_failure.get("preserved_legacy_source_path") != "train/continuous_source"
        ):
            raise RuntimeError(
                "staging contains a continuous source without the verified failed-stage identity; "
                "refusing to publish it as part of the new formal pool"
            )
        if retained_legacy_source.exists():
            raise FileExistsError(f"legacy staging retention path already exists: {retained_legacy_source}")
    moved: list[tuple[Path, Path]] = []
    legacy_moved = False
    try:
        if legacy_source.exists():
            os.replace(legacy_source, retained_legacy_source)
            legacy_moved = True
        for split in ("train", "validation", "test"):
            source = STAGING_ROOT / split
            destination = destinations[split]
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            moved.append((source, destination))
        central_destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(STAGING_ROOT / "data_split_manifest.json", central_destination)
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                os.replace(destination, source)
        if legacy_moved and retained_legacy_source.exists() and not legacy_source.exists():
            os.replace(retained_legacy_source, legacy_source)
        raise
    staged_config = STAGING_ROOT / "generation_config.toml"
    if staged_config.exists():
        staged_config.unlink()
    # This also clears the explicitly identified, failed continuous-stream
    # staging source after the complete independent dataset has been published.
    shutil.rmtree(STAGING_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_RELATIVE.as_posix())
    parser.add_argument("--sumo-home", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dry-run", action="store_true", help="check contracts and inputs without writing data or running SUMO")
    parser.add_argument("--resume", action="store_true", help="resume only from the same independent-Episode plan")
    parser.add_argument(
        "--pilot-only",
        action="store_true",
        help="generate and validate Train seed 20001 only; does not publish the formal dataset",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.pilot_only and args.dry_run:
        parser.error("--pilot-only cannot be combined with --dry-run")
    stage_attached = False
    progress: dict[str, Any] = {
        "status": "preflight",
        "planned": dict(SPLIT_COUNTS),
        "completed": {split: 0 for split in SPLIT_COUNTS},
        "failure": None,
    }
    try:
        config_path = _project_path(args.config)
        if not config_path.is_file():
            raise FileNotFoundError(f"formal config not found: {config_path}")
        if config_path != (PROJECT_ROOT / CONFIG_RELATIVE).resolve():
            raise ValueError(f"formal environment generation is pinned to {CONFIG_RELATIVE.as_posix()}")
        cfg, inputs = _runtime_checks(config_path, sumo_home_arg=args.sumo_home)
        seed_plan = _seed_plan()
        plan = _plan_payload(cfg, inputs, seed_plan)
        plan_fingerprint = _canonical_sha256(plan)

        final_conflicts = [
            PROJECT_ROOT / relative
            for relative in POOL_RELATIVES.values()
            if (PROJECT_ROOT / relative).exists()
        ]
        if (PROJECT_ROOT / SPLIT_MANIFEST_RELATIVE).exists():
            final_conflicts.append(PROJECT_ROOT / SPLIT_MANIFEST_RELATIVE)
        if final_conflicts:
            raise FileExistsError(
                "formal output already exists; this script will not silently overwrite it: "
                + ", ".join(str(path) for path in final_conflicts)
            )
        if args.dry_run:
            catalog_seed = int(project_config.get_config_value(cfg, "service.generation.catalog_seed"))
            random_trips_period_s = float(inputs["random_trips_period_s"])
            departure_begin_s = float(project_config.get_config_value(cfg, "simulation.random_trips_departure_begin_s"))
            departure_end_s = float(project_config.get_config_value(cfg, "simulation.random_trips_departure_end_s"))
            candidates_per_episode = generate_continuous_sumo_dataset.nominal_departure_count_for_interval(
                departure_begin_s, departure_end_s, random_trips_period_s
            )
            planned_catalog = task_scenario.make_shared_service_catalog_output(
                task_scenario.task_config_from_project_config(cfg),
                catalog_seed,
            )
            _validate_catalog(cfg, planned_catalog, catalog_seed)
            print(
                json.dumps(
                    {
                        "status": "preflight_passed",
                        "plan_fingerprint": plan_fingerprint,
                        "split_counts": SPLIT_COUNTS,
                        "seed_ranges": {
                            split: [values[0], values[-1]]
                            for split, values in seed_plan.items()
                        },
                        "traffic_design": "independent SUMO episode per seed for Train/Validation/Test",
                        "warmup_time_s": float(cfg["simulation"]["warmup_time_s"]),
                        "arrival_horizon_slots": inputs["protocol"]["arrival_horizon_slots"],
                        "execution_horizon_slots": inputs["protocol"]["execution_horizon_slots"],
                        "episode_traffic_plan": {
                            "randomTrips_period_s": random_trips_period_s,
                            "departure_begin_s": departure_begin_s,
                            "departure_end_s": departure_end_s,
                            "nominal_departure_rate_per_s": 1.0 / random_trips_period_s,
                            "nominal_candidate_departure_count_per_episode": candidates_per_episode,
                            "nominal_candidate_departure_count_all_episodes": candidates_per_episode * sum(SPLIT_COUNTS.values()),
                            "traffic_generation_scope": "candidate vehicles only in each Episode's [0,60) warmup interval",
                        },
                        "popularity_history_length": inputs["popularity_history_length"],
                        "service_catalog_path": str(
                            project_config.resolve_path(
                                PROJECT_ROOT,
                                project_config.get_config_value(cfg, "paths.service_catalog"),
                            )
                        ),
                        "service_catalog_seed": catalog_seed,
                        "service_catalog_identity_sha256": planned_catalog["generation_contract"][
                            "configuration_identity_sha256"
                        ],
                        "sumo_paths_checked_without_execution": {
                            "home": str(inputs["sumo_home"]),
                            "randomTrips": str(inputs["random_trips"]),
                            "binary": str(inputs["sumo_binary"]),
                        },
                        "sumo_started": False,
                        "outputs_written": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        stage_manifest = _get_or_create_stage(plan, resume=args.resume)
        stage_attached = True
        status_path = STAGING_ROOT / "generation_status.json"
        if args.resume and status_path.is_file():
            saved_progress = json.loads(status_path.read_text(encoding="utf-8"))
            if saved_progress.get("plan_fingerprint") != plan_fingerprint:
                raise RuntimeError("generation progress identity differs from the current formal plan")
            progress.update(saved_progress)
        progress["status"] = "generating"
        progress["plan_fingerprint"] = plan_fingerprint
        progress["failure"] = None
        staged_catalog = STAGING_ROOT / "train" / "service_catalog_500_1000.json"
        catalog_seed = int(project_config.get_config_value(cfg, "service.generation.catalog_seed"))
        expected_catalog = task_scenario.make_shared_service_catalog_output(
            task_scenario.task_config_from_project_config(cfg),
            catalog_seed,
        )
        if staged_catalog.exists():
            actual_catalog = json.loads(staged_catalog.read_text(encoding="utf-8"))
            _validate_catalog(cfg, actual_catalog, catalog_seed)
            if actual_catalog != expected_catalog:
                raise RuntimeError("staged service catalog differs from the deterministic current catalog identity")
        else:
            task_scenario.write_json_atomic(staged_catalog, expected_catalog)
        _validate_catalog(cfg, expected_catalog, catalog_seed)

        staged_config = STAGING_ROOT / "generation_config.toml"
        if not staged_config.exists():
            _prepare_generation_config(config_path, staged_catalog, staged_config)
        effective_cfg = generate_episode_realization.load_and_validate_config(staged_config)
        if (
            generate_episode_realization.generation_config_fingerprint(effective_cfg)
            != generate_episode_realization.generation_config_fingerprint(cfg)
        ):
            raise RuntimeError("staged generation config does not match the current formal config")
        effective_period_s = float(
            project_config.get_config_value(
                effective_cfg, "simulation.random_trips_period_s"
            )
        )
        if not math.isclose(
            effective_period_s,
            float(inputs["random_trips_period_s"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "staged generation config changed random_trips_period_s before SUMO generation"
            )
        stage_manifest["dataset_status"] = "generating"
        stage_manifest["started_at_utc"] = stage_manifest.get("started_at_utc", _utc_now())
        stage_manifest["service_catalog_sha256"] = _sha256_file(staged_catalog)
        _write_json(STAGING_ROOT / "data_split_manifest.json", stage_manifest)
        _record_generation_progress(progress)

        if args.pilot_only:
            _generate_independent_split(
                split="train",
                config_path=staged_config,
                seed_plan=seed_plan,
                inputs=inputs,
                device=args.device,
                progress=progress,
                seeds=[seed_plan["train"][0]],
            )
            pilot_dir = STAGING_ROOT / "train" / f"episode_{seed_plan['train'][0]}"
            pilot_roles = json.loads((pilot_dir / "vehicle_roles.json").read_text(encoding="utf-8"))
            progress["status"] = "pilot_validated"
            progress["completed"]["train"] = 1
            _record_generation_progress(progress)
            print(
                json.dumps(
                    {
                        "status": "pilot_validated",
                        "pilot": {
                            "split": "train",
                            "seed": seed_plan["train"][0],
                            "complete_candidate_vehicles": pilot_roles.get("complete_candidate_count"),
                            "research_vehicles": len(pilot_roles["research_vehicles"]),
                            "SV": len(pilot_roles["service_vehicles"]),
                            "TV": len(pilot_roles["task_vehicles"]),
                            "arrival_slots": pilot_roles["arrival_horizon_slots"],
                            "execution_slots": pilot_roles["execution_horizon_slots"],
                        },
                        "all_splits_generated": False,
                        "formal_manifest_published": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
            return 0

        for split in ("train", "validation", "test"):
            _generate_independent_split(
                split=split,
                config_path=staged_config,
                seed_plan=seed_plan,
                inputs=inputs,
                device=args.device,
                progress=progress,
            )
        completed_actual = dict(progress["completed"])
        if completed_actual != SPLIT_COUNTS:
            raise RuntimeError(
                f"completed Episode counts differ from formal plan: {completed_actual} != {SPLIT_COUNTS}"
            )
        _write_final_manifests(
            config_path=staged_config,
            inputs=inputs,
            seed_plan=seed_plan,
            catalog_path=staged_catalog,
        )
        progress["status"] = "validated"
        progress["completed"] = dict(SPLIT_COUNTS)
        _record_generation_progress(progress)
        print(
            json.dumps(
                {
                    "status": "all_splits_validated",
                    "splits": {
                        split: {
                            "planned": SPLIT_COUNTS[split],
                            "completed": progress["completed"][split],
                            "failure": None,
                        }
                        for split in SPLIT_COUNTS
                    },
                    "arrival_horizon_slots": inputs["protocol"]["arrival_horizon_slots"],
                    "execution_horizon_slots": inputs["protocol"]["execution_horizon_slots"],
                    "formal_manifest": str(PROJECT_ROOT / SPLIT_MANIFEST_RELATIVE),
                    "traffic_design": "independent SUMO Episode per seed in all splits",
                    "publishing": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        _publish()
        print("[FormalEnvironment] published the unique formal dataset; no training/evaluation was run.", flush=True)
        return 0
    except Exception as exc:
        progress["status"] = "failed"
        progress["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        print(
            json.dumps(
                {
                    "status": progress["status"],
                    "splits": {
                        split: {
                            "planned": SPLIT_COUNTS[split],
                            "completed": progress["completed"][split],
                            "failure": progress["failure"],
                        }
                        for split in SPLIT_COUNTS
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        if stage_attached and STAGING_ROOT.is_dir():
            try:
                _record_generation_progress(progress)
            except Exception:
                pass
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
