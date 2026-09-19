#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate one complete randomized full-episode realization.

The entrypoint is intentionally a thin orchestration layer.  It derives named
episode streams, runs SUMO for a temporary realization, reuses the frozen
service catalog/GRU/calibration residual bank, and writes every realization
artifact below the caller-provided output directory.  It never trains a model
or writes the shared ``data/generated`` artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


FORMAL_RUN = PROJECT_ROOT / "runs" / "ppo_v6_popart_actor1e4_fresh_20260910"
EXPECTED_SLOTS = 300


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


def prepare_output_dir(value: str | Path) -> Path:
    output_dir = resolve_project_path(value)
    forbidden = (
        PROJECT_ROOT / "data" / "generated",
        FORMAL_RUN,
    )
    if output_dir == PROJECT_ROOT or any(
        is_within(output_dir, path.resolve(strict=False)) for path in forbidden
    ):
        raise ValueError(
            "output-dir must not be the repository root, data/generated, "
            "or the formal V6 run directory"
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output-dir is not a directory: {output_dir}")
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

    with np.load(scenario_file, allow_pickle=False) as payload:
        origins = np.asarray(payload["origin_slots"], dtype=np.int64)
        if not np.array_equal(origins, np.arange(EXPECTED_SLOTS, dtype=np.int64)):
            raise AssertionError("prediction_scenarios.npz origins are not 0..299")
        positions = np.asarray(payload["positions_xy"])
        if positions.shape[0] != EXPECTED_SLOTS:
            raise AssertionError("prediction_scenarios.npz is not a full episode")


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
    protected_before = {
        "data/generated": snapshot_tree(protected_data),
        "formal V6 run": snapshot_tree(FORMAL_RUN),
    }

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
    local_cache_capacity = float(
        get(config, "nodes.vehicle.local_cache_capacity_mb")
    )
    tv_cache = task.generate_tv_local_cache(
        tv_ids,
        catalog,
        local_cache_capacity,
        int(seeds["tv_cache_seed"]),
    )
    task.validate_tv_local_cache(
        tv_cache,
        tv_ids,
        catalog,
        local_cache_capacity,
    )
    task.write_json_atomic(output_dir / "tv_local_cache.json", tv_cache)
    task_runtime = time.perf_counter() - task_start

    prediction_start = time.perf_counter()
    popularity_file = output_dir / "popularity.csv"
    popularity_history_file = output_dir / "popularity_history.csv"
    prediction_file = output_dir / "popularity_predictions.csv"
    prediction_status = pop.main(
        [
            "--config",
            str(config_file),
            "--mode",
            "predict",
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
    assert_snapshot_unchanged("data/generated", protected_data, protected_before["data/generated"])
    assert_snapshot_unchanged("formal V6 run", FORMAL_RUN, protected_before["formal V6 run"])
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
    return {
        "episode_seed": episode_seed,
        "derived_stream_count": len(EPISODE_SEED_STREAMS),
        "episode_slots": len(formal_steps),
        "sumo_runtime_seconds": sumo_runtime,
        "task_runtime_seconds": task_runtime,
        "prediction_runtime_seconds": prediction_runtime,
        "scenario_runtime_seconds": scenario_runtime,
        "single_episode_runtime_seconds": total_runtime,
        "output_dir": str(output_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one complete 300-slot randomized episode realization."
    )
    parser.add_argument(
        "--config",
        default="configs/proposed/system_config.toml",
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
