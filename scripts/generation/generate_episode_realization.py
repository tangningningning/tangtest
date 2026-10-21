#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成一个完整的随机化 episode realization。

入口负责派生 episode 随机流、运行 SUMO、复用冻结的服务目录、GRU 和校准残差，
并将 realization 文件写入调用方指定的输出目录。它不会训练模型或改写共享产物。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common import project_config as project_config_module
from scenario import popularity_prediction as pop
from scenario import scenario_generator as scenario
from scenario import sumo_scenario as sumo
from scenario import task_scenario as task
from scenario.realization import EPISODE_SEED_STREAMS, derive_episode_seeds
from scenario.vehicle_order import RESIDUAL_SAMPLING_CONTRACT


EXPECTED_SLOTS = 300
EPISODE_MANIFEST_FILENAME = "episode_manifest.json"
EPISODE_MANIFEST_SCHEMA_VERSION = "episode_manifest_v2"
RUN_MANIFEST_FILENAME = "run_manifest.json"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def protected_output_directories() -> dict[str, set[Path]]:
    """查找已有正式训练目录和 episode 池，避免生成器覆盖它们。"""

    run_root = PROJECT_ROOT / "runs"
    run_directories = {
        manifest.parent.resolve(strict=False)
        for manifest in run_root.rglob(RUN_MANIFEST_FILENAME)
        if manifest.is_file()
    } if run_root.is_dir() else set()
    pool_root = PROJECT_ROOT / "environment_data"
    episode_directories = {
        manifest.parent.resolve(strict=False)
        for manifest in pool_root.rglob(EPISODE_MANIFEST_FILENAME)
        if manifest.is_file() and manifest.parent.name.startswith("episode_")
    } if pool_root.is_dir() else set()
    pool_directories = {directory.parent for directory in episode_directories}
    return {
        "正式 run": run_directories,
        "正式 episode 池": pool_directories,
    }


def paths_overlap(left: Path, right: Path) -> bool:
    """判断两个路径是否相同或存在祖先/后代关系。"""

    return is_within(left, right) or is_within(right, left)


def prepare_output_dir(value: str | Path) -> Path:
    output_dir = resolve_project_path(value)
    protected = {
        "仓库根目录": {PROJECT_ROOT.resolve(strict=False)},
        "data/generated": {(PROJECT_ROOT / "data" / "generated").resolve(strict=False)},
        **protected_output_directories(),
    }
    for label, paths in protected.items():
        if any(
            output_dir == path
            if label == "仓库根目录"
            else paths_overlap(output_dir, path)
            for path in paths
        ):
            raise ValueError(f"output-dir 不能覆盖或包含受保护目录：{label}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output-dir 不是目录：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def config_path(config: dict[str, Any], key: str) -> Path:
    value = project_config_module.get_config_value(config, key)
    if not value:
        raise ValueError(f"missing required config path: {key}")
    path = resolve_project_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"required artifact is not readable: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity() -> tuple[str, bool]:
    sha_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    dirty_result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if sha_result.returncode != 0 or dirty_result.returncode != 0:
        raise RuntimeError(
            "unable to capture source git identity: "
            f"sha={sha_result.returncode}, dirty={dirty_result.returncode}"
        )
    sha = sha_result.stdout.strip()
    if not sha:
        raise RuntimeError("source git SHA is empty")
    return sha, bool(dirty_result.stdout.strip())


def _generation_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    """仅返回会影响生成 episode artifact 的配置字段。"""

    get = project_config_module.get_config_value
    return {
        "schema": EPISODE_MANIFEST_SCHEMA_VERSION,
        "episode_seed_protocol": "adaptive_cache_mec_episode_seed_v1",
        "residual_sampling_contract": RESIDUAL_SAMPLING_CONTRACT,
        "simulation": {
            key: get(config, f"simulation.{key}")
            for key in (
                "dt_s",
                "warmup_time_s",
                "episode_time_s",
                "num_research_vehicles",
                "num_service_vehicles",
                "num_task_vehicles",
                "random_trips_departure_begin_s",
                "random_trips_departure_end_s",
                "random_trips_period_s",
                "random_trips_min_distance_m",
                "random_trips_intermediate",
                "random_trips_maxtries",
            )
        },
        "service": {
            "count": get(config, "service.count"),
            "generation": {
                key: get(config, f"service.generation.{key}")
                for key in (
                    "service_size_min_gb",
                    "service_size_max_gb",
                    "input_size_min_kb",
                    "input_size_max_kb",
                    "cycles_per_bit_min",
                    "cycles_per_bit_max",
                )
            },
        },
        "task": {
            key: get(config, f"task.{key}")
            for key in (
                "arrival_probability",
                "popularity_zeta",
                "popularity_epoch_length",
                "popularity_changes_per_epoch",
                "input_relative_jitter",
                "load_relative_jitter",
                "master_seed",
            )
        },
        "prediction": {
            "vehicle": {
                key: get(config, f"prediction.vehicle.{key}")
                for key in (
                    "horizon",
                    "sigma_position_m",
                    "sigma_speed_mps",
                    "observation_noise_seed",
                )
            },
            "popularity": {
                key: get(config, f"prediction.popularity.{key}")
                for key in (
                    "history_length",
                    "hidden_size",
                    "training_seed",
                    "sequence_seed_base",
                    "num_train_sequences",
                    "num_val_sequences",
                    "num_test_sequences",
                )
            },
        },
        "scenario": {
            key: get(config, f"scenario.{key}")
            for key in (
                "num_scenarios",
                "master_seed",
                "max_resample_attempts",
                "cross_vehicle_residual_sampling",
            )
        },
    }


def generation_config_fingerprint(config: dict[str, Any]) -> str:
    payload = _generation_config_payload(config)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_artifact_hashes(config: dict[str, Any]) -> dict[str, str]:
    paths = {
        "service_catalog_sha256": config_path(config, "paths.service_catalog"),
        "popularity_gru_sha256": config_path(config, "paths.popularity_gru"),
        "vehicle_residual_bank_sha256": config_path(
            config, "paths.vehicle_residual_bank"
        ),
        "sumo_network_sha256": config_path(config, "paths.sumo_net"),
    }
    return {key: sha256_file(path) for key, path in paths.items()}


def build_episode_manifest(
    config: dict[str, Any], episode_seed: int, output_dir: Path
) -> dict[str, Any]:
    source_git_sha, source_git_dirty = _git_identity()
    get = project_config_module.get_config_value
    seeds = derive_episode_seeds(int(episode_seed))
    artifact_paths = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != EPISODE_MANIFEST_FILENAME
    )
    artifact_hashes = {
        path.name: sha256_file(path) for path in artifact_paths
    }
    source_hashes = _source_artifact_hashes(config)
    manifest = {
        "schema_version": EPISODE_MANIFEST_SCHEMA_VERSION,
        "data_input_contracts": {
            "residual_sampling": RESIDUAL_SAMPLING_CONTRACT,
        },
        "episode_seed": int(episode_seed),
        "derived_episode_seed_streams": seeds,
        "generation_config_fingerprint": generation_config_fingerprint(config),
        "generation_config_fields": _generation_config_payload(config),
        "source_git_sha": source_git_sha,
        "source_git_dirty": bool(source_git_dirty),
        **source_hashes,
        "simulation": {
            "dt": float(get(config, "simulation.dt_s")),
            "warmup": float(get(config, "simulation.warmup_time_s")),
            "episode_slots": EXPECTED_SLOTS,
        },
        "vehicle_counts": {
            "research": int(get(config, "simulation.num_research_vehicles")),
            "TV": int(get(config, "simulation.num_task_vehicles")),
            "SV": int(get(config, "simulation.num_service_vehicles")),
        },
        "task_generation_identity": {
            "arrival_probability": get(config, "task.arrival_probability"),
            "zipf_parameter": get(config, "task.popularity_zeta"),
            "workload_jitter": {
                "input_relative_jitter": get(config, "task.input_relative_jitter"),
                "load_relative_jitter": get(config, "task.load_relative_jitter"),
            },
        },
        "prediction_identity": {
            "vehicle_horizon": get(config, "prediction.vehicle.horizon"),
            "vehicle_noise_contract": {
                "sigma_position_m": get(config, "prediction.vehicle.sigma_position_m"),
                "sigma_speed_mps": get(config, "prediction.vehicle.sigma_speed_mps"),
                "observation_noise_seed": get(
                    config, "prediction.vehicle.observation_noise_seed"
                ),
            },
            "popularity_model_identity": {
                "checkpoint_sha256": source_hashes["popularity_gru_sha256"],
                "history_length": get(config, "prediction.popularity.history_length"),
                "hidden_size": get(config, "prediction.popularity.hidden_size"),
            },
            "scenario_count": get(config, "scenario.num_scenarios"),
        },
        "generated_artifacts": sorted(artifact_hashes),
        "generated_artifact_sha256": artifact_hashes,
    }
    return manifest


def write_episode_manifest(
    config: dict[str, Any], episode_seed: int, output_dir: Path
) -> Path:
    manifest = build_episode_manifest(config, episode_seed, output_dir)
    path = output_dir / EPISODE_MANIFEST_FILENAME
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def validate_episode_provenance(
    directory: Path,
    config: dict[str, Any],
    episode_seed: int,
) -> dict[str, Any]:
    """Validate historical data identity without comparing it to the current Git checkout."""

    manifest_path = Path(directory) / EPISODE_MANIFEST_FILENAME
    prefix = "REALIZATION_PROVENANCE_MISMATCH"
    if not manifest_path.is_file():
        raise RuntimeError(f"{prefix}: missing {manifest_path.name}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"{prefix}: invalid JSON in {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"{prefix}: manifest root must be an object")
    if manifest.get("schema_version") != EPISODE_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            f"{prefix}: unsupported schema_version={manifest.get('schema_version')!r}"
        )
    if int(manifest.get("episode_seed", -1)) != int(episode_seed):
        raise RuntimeError(
            f"{prefix}: episode_seed={manifest.get('episode_seed')!r} "
            f"expected={int(episode_seed)}"
        )
    expected_fingerprint = generation_config_fingerprint(config)
    if manifest.get("generation_config_fingerprint") != expected_fingerprint:
        raise RuntimeError(
            f"{prefix}: generation_config_fingerprint mismatch; "
            f"stored={manifest.get('generation_config_fingerprint')!r} "
            f"expected={expected_fingerprint!r}"
        )
    if manifest.get("generation_config_fields") != _generation_config_payload(config):
        raise RuntimeError(f"{prefix}: generation_config_fields mismatch")
    expected_data_contracts = {
        "residual_sampling": RESIDUAL_SAMPLING_CONTRACT,
    }
    if manifest.get("data_input_contracts") != expected_data_contracts:
        raise RuntimeError(f"{prefix}: data input contract mismatch")
    expected_hashes = _source_artifact_hashes(config)
    for field, expected_hash in expected_hashes.items():
        if manifest.get(field) != expected_hash:
            raise RuntimeError(
                f"{prefix}: {field} mismatch; stored={manifest.get(field)!r} "
                f"expected={expected_hash!r}"
            )
    source_sha = manifest.get("source_git_sha")
    source_dirty = manifest.get("source_git_dirty")
    if (
        not isinstance(source_sha, str)
        or len(source_sha) != 40
        or any(character not in "0123456789abcdef" for character in source_sha)
        or not isinstance(source_dirty, bool)
    ):
        raise RuntimeError(f"{prefix}: historical source Git identity is missing or invalid")
    stored_seeds = manifest.get("derived_episode_seed_streams")
    if not isinstance(stored_seeds, dict):
        raise RuntimeError(f"{prefix}: derived episode seed streams are missing")
    expected_seeds = derive_episode_seeds(int(episode_seed))
    for stream_name, expected_seed in expected_seeds.items():
        if stored_seeds.get(stream_name) != expected_seed:
            raise RuntimeError(
                f"{prefix}: derived seed mismatch for {stream_name}; "
                f"stored={stored_seeds.get(stream_name)!r} expected={expected_seed!r}"
            )
    actual_paths = sorted(
        path
        for path in Path(directory).iterdir()
        if path.is_file() and path.name != EPISODE_MANIFEST_FILENAME
    )
    actual_names = sorted(path.name for path in actual_paths)
    if manifest.get("generated_artifacts") != actual_names:
        raise RuntimeError(
            f"{prefix}: generated artifact filename set mismatch"
        )
    stored_artifact_hashes = manifest.get("generated_artifact_sha256")
    if not isinstance(stored_artifact_hashes, dict):
        raise RuntimeError(f"{prefix}: generated artifact hashes are missing")
    for path in actual_paths:
        actual_hash = sha256_file(path)
        if stored_artifact_hashes.get(path.name) != actual_hash:
            raise RuntimeError(
                f"{prefix}: generated artifact hash mismatch for {path.name}"
            )
    return manifest


def snapshot_tree(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    if not path.is_dir():
        raise ValueError(f"protected path is not a directory: {path}")
    snapshot: dict[str, str] = {}
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            relative = file_path.relative_to(path).as_posix()
            snapshot[relative] = sha256_file(file_path)
    return snapshot


def assert_snapshot_unchanged(
    label: str,
    path: Path,
    before: dict[str, str] | None,
) -> None:
    after = snapshot_tree(path)
    if before != after:
        raise AssertionError(f"protected {label} changed: {path}")


def load_and_validate_config(config_file: Path) -> dict[str, Any]:
    config = project_config_module.load_config(
        config_file,
        require_artifacts=False,
    )
    protocol = project_config_module.get_ppo_data_protocol(config)
    expected = {
        "protocol": project_config_module.PPO_DATA_PROTOCOL_FULL_EPISODE_RANDOMIZED,
        "physical_episode_slots": EXPECTED_SLOTS,
        "episode_slots": EXPECTED_SLOTS,
        "train_origin_start": 0,
        "train_origin_end": EXPECTED_SLOTS - 1,
        "terminal_origin": EXPECTED_SLOTS - 1,
        "rollout_steps": EXPECTED_SLOTS,
        "test_origin_start": None,
        "test_origin_end": None,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(
                f"full-episode config mismatch for {key}: "
                f"{protocol.get(key)!r} != {value!r}"
            )
    return config


def build_sumo_config(
    config: dict[str, Any],
    seeds: dict[str, int],
) -> sumo.ScenarioConfig:
    get = project_config_module.get_config_value
    return sumo.ScenarioConfig(
        dt_s=float(get(config, "simulation.dt_s")),
        warmup_time_s=float(get(config, "simulation.warmup_time_s")),
        episode_time_s=float(get(config, "simulation.episode_time_s")),
        num_research_vehicles=int(get(config, "simulation.num_research_vehicles")),
        num_service_vehicles=int(get(config, "simulation.num_service_vehicles")),
        num_task_vehicles=int(get(config, "simulation.num_task_vehicles")),
        random_seed=int(seeds["sumo_seed"]),
        traffic_seed=int(seeds["traffic_seed"]),
        research_vehicle_role_seed=int(seeds["research_role_seed"]),
        service_vehicle_role_seed=int(seeds["service_role_seed"]),
        random_trips_period_s=float(
            get(config, "simulation.random_trips_period_s", 1.0)
        ),
        random_trips_departure_begin_s=float(
            get(config, "simulation.random_trips_departure_begin_s", 0.0)
        ),
        random_trips_departure_end_s=float(
            get(config, "simulation.random_trips_departure_end_s", 60.0)
        ),
        random_trips_min_distance_m=float(
            get(config, "simulation.random_trips_min_distance_m", 4000.0)
        ),
        random_trips_intermediate=int(
            get(config, "simulation.random_trips_intermediate", 2)
        ),
        random_trips_maxtries=int(
            get(config, "simulation.random_trips_maxtries", 1000)
        ),
    )


def validate_written_outputs(
    output_dir: Path,
    task_vehicle_count: int,
    research_vehicle_count: int,
    horizon_steps: int,
) -> None:
    popularity_file = output_dir / "popularity.csv"
    popularity_history_file = output_dir / "popularity_history.csv"
    tasks_file = output_dir / "tasks.csv"
    prediction_file = output_dir / "popularity_predictions.csv"
    vehicle_prediction_file = output_dir / "vehicle_predictions.csv"
    scenario_file = output_dir / "prediction_scenarios.npz"

    rho_star, _, _ = pop.load_popularity_csv(popularity_file)
    if rho_star.shape[0] != EXPECTED_SLOTS:
        raise AssertionError("popularity.csv does not cover slots 0..299")
    history_counts, _, history_slots = pop.load_popularity_history_csv(
        popularity_history_file
    )
    if history_counts.shape[0] <= 0 or history_slots != list(
        range(-history_counts.shape[0], 0)
    ):
        raise AssertionError("popularity_history.csv has invalid prehistory slots")

    with tasks_file.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    task_slots = {int(row["slot"]) for row in rows}
    if task_slots != set(range(EXPECTED_SLOTS)):
        raise AssertionError("tasks.csv does not cover slots 0..299")
    if len(rows) != task_vehicle_count * EXPECTED_SLOTS:
        raise AssertionError("tasks.csv does not cover every TV×slot")
    _roles, research_ids, service_ids, task_ids = scenario.load_roles(
        output_dir / "vehicle_roles.json"
    )
    if (
        len(task_ids) != task_vehicle_count
        or len(research_ids) != research_vehicle_count
    ):
        raise AssertionError("vehicle role counts do not match generated artifacts")
    task_ids_by_slot: dict[int, list[str]] = {}
    for row in rows:
        task_ids_by_slot.setdefault(int(row["slot"]), []).append(str(row["vehicle_id"]))
    if any(
        len(ids) != task_vehicle_count
        or len(set(ids)) != len(ids)
        or set(ids) != set(task_ids)
        for ids in task_ids_by_slot.values()
    ):
        raise AssertionError("tasks.csv vehicle IDs do not match vehicle_roles.json")

    with prediction_file.open("r", encoding="utf-8", newline="") as stream:
        prediction_rows = list(csv.DictReader(stream))
    prediction_origins = {int(row["origin_slot"]) for row in prediction_rows}
    if prediction_origins != set(range(EXPECTED_SLOTS)):
        raise AssertionError("popularity_predictions.csv does not cover origins 0..299")

    with vehicle_prediction_file.open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        vehicle_rows = list(csv.DictReader(stream))
    vehicle_origins = {int(row["origin_slot"]) for row in vehicle_rows}
    if vehicle_origins != set(range(EXPECTED_SLOTS)):
        raise AssertionError("vehicle_predictions.csv does not cover origins 0..299")
    expected_vehicle_rows = (
        EXPECTED_SLOTS * horizon_steps * research_vehicle_count
    )
    if len(vehicle_rows) != expected_vehicle_rows:
        raise AssertionError("vehicle_predictions.csv has an unexpected row count")
    prediction_ids: dict[tuple[int, int], list[str]] = {}
    for row in vehicle_rows:
        key = (int(row["origin_slot"]), int(row["horizon"]))
        prediction_ids.setdefault(key, []).append(str(row["vehicle_id"]))
    expected_prediction_keys = {
        (origin, horizon)
        for origin in range(EXPECTED_SLOTS)
        for horizon in range(horizon_steps)
    }
    if set(prediction_ids) != expected_prediction_keys or any(
        len(ids) != research_vehicle_count
        or len(set(ids)) != len(ids)
        or set(ids) != set(research_ids)
        for ids in prediction_ids.values()
    ):
        raise AssertionError(
            "vehicle_predictions.csv vehicle IDs do not match vehicle_roles.json"
        )

    with np.load(scenario_file, allow_pickle=False) as payload:
        origins = np.asarray(payload["origin_slots"], dtype=np.int64)
        if not np.array_equal(origins, np.arange(EXPECTED_SLOTS, dtype=np.int64)):
            raise AssertionError("prediction_scenarios.npz origins are not 0..299")
        positions = np.asarray(payload["positions_xy"])
        if positions.shape[0] != EXPECTED_SLOTS:
            raise AssertionError("prediction_scenarios.npz is not a full episode")
        vehicle_ids = tuple(np.asarray(payload["vehicle_ids"]).astype(str).tolist())
        role_ids = np.asarray(payload["vehicle_roles"]).astype(str).tolist()
        task_array = tuple(
            np.asarray(payload["task_vehicle_ids"]).astype(str).tolist()
        )
        service_array = tuple(
            np.asarray(payload["service_vehicle_ids"]).astype(str).tolist()
        )
        metadata = json.loads(str(payload["metadata_json"].item()))
        role_by_id = {
            **{vehicle_id: "SV" for vehicle_id in service_ids},
            **{vehicle_id: "TV" for vehicle_id in task_ids},
        }
        if (
            len(set(vehicle_ids)) != len(vehicle_ids)
            or set(vehicle_ids) != set(research_ids)
            or len(set(task_array)) != len(task_array)
            or set(task_array) != set(task_ids)
            or len(set(service_array)) != len(service_array)
            or set(service_array) != set(service_ids)
            or len(role_ids) != len(vehicle_ids)
            or any(
                role_by_id.get(vehicle_id) != role.upper()
                for vehicle_id, role in zip(vehicle_ids, role_ids, strict=True)
            )
            or positions.ndim != 5
            or positions.shape[3:] != (len(vehicle_ids), 2)
            or any(
                np.asarray(payload[field]).ndim != 4
                or np.asarray(payload[field]).shape[-1] != len(task_array)
                for field in ("arrival", "service_id", "input_size", "cpu_cycles")
            )
            or metadata.get("residual_sampling_contract")
            != RESIDUAL_SAMPLING_CONTRACT
        ):
            raise AssertionError(
                "prediction_scenarios.npz vehicle mapping contract mismatch"
            )


def run_episode(
    config_file: Path,
    episode_seed: int,
    output_dir: Path,
    sumo_home_arg: str | None,
    device_name: str,
) -> dict[str, float | int | str]:
    config = load_and_validate_config(config_file)
    seeds = derive_episode_seeds(episode_seed)
    if tuple(seeds) != EPISODE_SEED_STREAMS:
        raise AssertionError("derived episode stream order is not canonical")

    catalog_file = config_path(config, "paths.service_catalog")
    gru_file = config_path(config, "paths.popularity_gru")
    residual_file = config_path(config, "paths.vehicle_residual_bank")
    network_file = config_path(config, "paths.sumo_net")
    catalog = task.load_json(catalog_file)
    service_count = int(
        project_config_module.get_config_value(config, "service.count")
    )
    if int(catalog.get("service_count", -1)) != service_count:
        raise ValueError("frozen service catalog count does not match config")
    if not isinstance(catalog.get("services"), list) or len(
        catalog["services"]
    ) != service_count:
        raise ValueError("frozen service catalog is incomplete")

    residual_bank = scenario.load_residual_bank(residual_file)
    get = project_config_module.get_config_value
    horizon_steps = int(get(config, "prediction.vehicle.horizon"))
    dt_s = float(get(config, "simulation.dt_s"))
    if int(residual_bank["H_p"]) != horizon_steps:
        raise ValueError("frozen calibration residual H_p does not match config")
    if not np.isclose(float(residual_bank["dt"]), dt_s, rtol=0.0, atol=1e-9):
        raise ValueError("frozen calibration residual dt does not match config")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _, gru_metadata = pop.load_checkpoint_model(gru_file, device)
    if int(gru_metadata["H_p"]) != horizon_steps:
        raise ValueError("frozen GRU H_p does not match config")
    if int(gru_metadata["F"]) != service_count:
        raise ValueError("frozen GRU service count does not match config")

    protected_data = PROJECT_ROOT / "data" / "generated"
    protected_before = snapshot_tree(protected_data)

    total_start = time.perf_counter()

    sumo_start = time.perf_counter()
    sumo_config = build_sumo_config(config, seeds)
    sumo_config.validate()
    sumo_home = sumo.find_sumo_home(sumo_home_arg)
    sumolib, traci = sumo.load_sumo_modules(sumo_home)
    source = sumo.generate_random_trips(
        project_root=PROJECT_ROOT,
        sumo_home=sumo_home,
        network_file=network_file,
        output_dir=output_dir,
        config=sumo_config,
    )
    binary = sumo.find_sumo_binary(sumolib, gui=False)
    command = sumo.build_sumo_command(binary, source, sumo_config)
    rows, route_cache, seen_ids, _ = sumo.collect_trajectories(
        traci,
        command,
        sumo_config,
    )
    if not rows:
        raise RuntimeError("SUMO produced no trajectory rows")
    complete_ids, route_metadata, formal_steps = sumo.find_complete_vehicles(
        rows,
        route_cache,
        sumo_config,
    )
    if len(complete_ids) < sumo_config.num_research_vehicles:
        raise RuntimeError(
            "SUMO complete-vehicle count is below the configured research count: "
            f"{len(complete_ids)} < {sumo_config.num_research_vehicles}"
        )
    roles, selection_reproducible = sumo.build_role_assignments(
        complete_ids,
        route_metadata,
        seen_ids,
        sumo_config,
    )
    if not selection_reproducible:
        raise AssertionError("role assignment is not reproducible")
    trajectory_file = output_dir / "trajectory.csv"
    roles_file = output_dir / "vehicle_roles.json"
    sumo.write_trajectory_csv(rows, trajectory_file)
    sumo.write_roles_json(roles, roles_file)
    sumo.validate_outputs(trajectory_file, roles_file, complete_ids, sumo_config)
    sumo_runtime = time.perf_counter() - sumo_start

    task_start = time.perf_counter()
    tv_ids = [str(vehicle_id) for vehicle_id in roles["task_vehicles"]]
    task_config = task.task_config_from_project_config(config)
    task_config.validate(len(tv_ids))
    named_task_seeds = {
        "popularity_evolution": seeds["popularity_seed"],
        "task_arrival": seeds["task_arrival_seed"],
        "service_request": seeds["service_request_seed"],
        "workload_variation": seeds["workload_seed"],
    }
    artifacts = task.build_episode_artifacts(
        tv_ids,
        catalog,
        dt_s,
        float(get(config, "simulation.warmup_time_s")),
        float(get(config, "simulation.episode_time_s")),
        task_config,
        named_task_seeds,
    )
    task.validate_artifacts(
        artifacts,
        tv_ids,
        task_config,
        task_config.arrival_probability,
    )
    task.output_realization_artifacts(
        output_dir,
        artifacts["popularity_history_rows"],
        artifacts["popularity_rows"],
        artifacts["task_rows"],
    )
    task_runtime = time.perf_counter() - task_start

    prediction_start = time.perf_counter()
    popularity_file = output_dir / "popularity.csv"
    popularity_history_file = output_dir / "popularity_history.csv"
    prediction_file = output_dir / "popularity_predictions.csv"
    prediction_status = pop.main(
        [
            "--config",
            str(config_file),
            "--popularity-file",
            str(popularity_file),
            "--popularity-history-file",
            str(popularity_history_file),
            "--service-catalog-file",
            str(catalog_file),
            "--roles-file",
            str(roles_file),
            "--output-dir",
            str(output_dir),
            "--checkpoint",
            str(gru_file),
            "--prediction-file",
            str(prediction_file),
            "--device",
            device_name,
        ]
    )
    if prediction_status != 0:
        raise RuntimeError("frozen GRU popularity inference failed")
    prediction_runtime = time.perf_counter() - prediction_start

    scenario_start = time.perf_counter()
    scenario_file = output_dir / "prediction_scenarios.npz"
    vehicle_prediction_file = output_dir / "vehicle_predictions.csv"
    scenario_status = scenario.main(
        [
            "--config",
            str(config_file),
            "--episode-seed",
            str(episode_seed),
            "--roles-file",
            str(roles_file),
            "--trajectory-file",
            str(trajectory_file),
            "--popularity-file",
            str(popularity_file),
            "--popularity-history-file",
            str(popularity_history_file),
            "--service-catalog-file",
            str(catalog_file),
            "--residual-file",
            str(residual_file),
            "--gru-checkpoint",
            str(gru_file),
            "--output-file",
            str(scenario_file),
            "--vehicle-prediction-file",
            str(vehicle_prediction_file),
            "--net-file",
            str(network_file),
            "--sumo-home",
            str(sumo_home),
            "--device",
            device_name,
        ]
    )
    if scenario_status != 0:
        raise RuntimeError("frozen vehicle/scenario generation failed")
    scenario_runtime = time.perf_counter() - scenario_start

    validate_written_outputs(
        output_dir,
        task_vehicle_count=len(tv_ids),
        research_vehicle_count=len(roles["research_vehicles"]),
        horizon_steps=horizon_steps,
    )
    manifest_path = write_episode_manifest(config, int(episode_seed), output_dir)
    assert_snapshot_unchanged("data/generated", protected_data, protected_before)
    total_runtime = time.perf_counter() - total_start

    print("单 Episode realization 生成完成：")
    print(f"  episode_seed={episode_seed}")
    print(f"  derived_stream_count={len(EPISODE_SEED_STREAMS)}")
    print(f"  episode_slots={len(formal_steps)}; origin_range=0..{len(formal_steps) - 1}")
    print("  shared_service_catalog_reused=True")
    print("  frozen_gru_reused=True; gru_trained=False")
    print("  frozen_calibration_reused=True; calibration_updated=False")
    print(f"  output_dir={output_dir}")
    print(f"  sumo_runtime_seconds={sumo_runtime:.6f}")
    print(f"  task_runtime_seconds={task_runtime:.6f}")
    print(f"  prediction_runtime_seconds={prediction_runtime:.6f}")
    print(f"  scenario_runtime_seconds={scenario_runtime:.6f}")
    print(f"  single_episode_runtime_seconds={total_runtime:.6f}")
    print(f"  episode_manifest={manifest_path}")
    return {
        "episode_seed": episode_seed,
        "derived_stream_count": len(EPISODE_SEED_STREAMS),
        "episode_slots": len(formal_steps),
        "sumo_runtime_seconds": sumo_runtime,
        "task_runtime_seconds": task_runtime,
        "prediction_runtime_seconds": prediction_runtime,
        "scenario_runtime_seconds": scenario_runtime,
        "single_episode_runtime_seconds": total_runtime,
        "episode_manifest": str(manifest_path),
        "output_dir": str(output_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one complete 300-slot randomized episode realization."
    )
    parser.add_argument(
        "--config",
        default="configs/proposed/alternating_actor_updates_v1.toml",
    )
    parser.add_argument("--episode-seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sumo-home", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_file = resolve_project_path(args.config)
        if not config_file.is_file():
            raise FileNotFoundError(f"config is not readable: {config_file}")
        output_dir = prepare_output_dir(args.output_dir)
        run_episode(
            config_file,
            args.episode_seed,
            output_dir,
            args.sumo_home,
            args.device,
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
