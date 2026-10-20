#!/usr/bin/env python3
"""Compare Training-only task utility reference scales without changing the model.

The script reads only the formal Training pool. It recalculates the documented
deterministic P95 reference allocation at both historical 40 GHz and current
20 GHz Cloud CPU, then compares fixed-assignment and Transfer-Exchange
Matching/KKT outcomes on nine predeclared real Training slots. Per-task scales
are evaluated by a shadow KKT path that reuses the project's numerical KKT
solvers and physical equations while keeping formal source behavior untouched.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from common import project_config, system_model  # noqa: E402
from decision import cache_configuration, task_allocation, uav_configuration  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "proposed" / "system_config.toml"
POOL_ROOT = PROJECT_ROOT / "environment_data" / "proposed_joint_training"
POOL_MANIFEST_PATH = POOL_ROOT / "formal_pool_manifest.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "training_reference_normalization_comparison.md"
RESULT_PATH = PROJECT_ROOT / "reports" / "training_reference_normalization_comparison.json"

CALIBRATION_SEEDS = (
    20001, 20028, 20055, 20082, 20110, 20137,
    20164, 20191, 20218, 20245, 20272, 20300,
)
CALIBRATION_SLOTS = (0, 33, 66, 99, 132, 165, 198, 231, 264, 299)
MATCHING_SAMPLES = (
    (20001, 21, 10), (20002, 22, 10), (20003, 21, 10),
    (20001, 1, 14), (20002, 10, 14), (20003, 2, 14),
    (20003, 119, 20), (20011, 48, 20), (20012, 127, 20),
)
P95_FORMAL_T_REF_HINT_S = 30.473728355325758
P95_FORMAL_E_REF_HINT_J = 5.1574642506299275
HISTORICAL_LOCAL_T_REF_S = 30.6289851662
HISTORICAL_LOCAL_E_REF_J = 3.06289851662
HISTORICAL_CLOUD_CPU_GHZ = 40.0
CURRENT_CLOUD_CPU_GHZ = 20.0
LOCAL_CPU_ASSUMED_CYCLES_PER_S = 1.0e9
LOCAL_KAPPA_ASSUMED = 5.0e-27
RELATIVE_DEADLINE_ALPHAS = (0.5, 1.0, 2.0)
MATCHING_TOLERANCE = 2.0e-7


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def qstats(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.any(~np.isfinite(array)):
        raise ValueError("Cannot summarize empty or non-finite reference samples")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p5": float(np.percentile(array, 5.0, method="linear")),
        "p95": float(np.percentile(array, 95.0, method="linear")),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def load_config_and_pool() -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any], Path]:
    cfg = project_config.load_config(str(CONFIG_PATH))
    project_config.validate_for_formal_experiment(cfg)
    cache_cfg = cache_configuration.cache_config_from_project(cfg)
    pool_manifest = read_json(POOL_MANIFEST_PATH)
    catalog_path = project_config.resolve_path(
        Path(cfg["_meta"]["project_root"]),
        project_config.get_config_value(cfg, "paths.service_catalog"),
    )
    catalog = cache_configuration.load_service_catalog(catalog_path)
    expected_catalog_hash = str(pool_manifest["service_catalog_sha256"])
    if sha256_file(catalog_path) != expected_catalog_hash:
        raise ValueError("Training pool service catalog hash does not match its pool manifest")
    if str(pool_manifest.get("split", "")).lower() != "train":
        raise ValueError("The selected data pool is not attested as Training")

    expected_formal = {
        "cloud_resource_mode": "finite_pool",
        "cloud_cpu_capacity_cycles_per_s": CURRENT_CLOUD_CPU_GHZ * 1.0e9,
        "cloud_total_bandwidth_bps": 20.0e6,
        "cloud_delay_factor": 13.0,
        "omega_t": 0.5,
        "omega_e": 0.5,
    }
    actual = {
        "cloud_resource_mode": cache_cfg.cloud_resource_mode,
        "cloud_cpu_capacity_cycles_per_s": cache_cfg.cloud_cpu_capacity_cycles_per_s,
        "cloud_total_bandwidth_bps": cache_cfg.cloud_total_bandwidth_bps,
        "cloud_delay_factor": cache_cfg.cloud_delay_factor,
        "omega_t": cache_cfg.omega_t,
        "omega_e": cache_cfg.omega_e,
    }
    for key, expected in expected_formal.items():
        observed = actual[key]
        if isinstance(expected, str):
            valid = observed == expected
        else:
            valid = math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1.0e-9)
        if not valid:
            raise ValueError(f"Current formal contract mismatch for {key}: {observed!r} != {expected!r}")
    if not math.isclose(cache_cfg.time_ref_s, P95_FORMAL_T_REF_HINT_S, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"Current T_ref differs from the declared global P95: {cache_cfg.time_ref_s}")
    if not math.isclose(cache_cfg.energy_ref_j, P95_FORMAL_E_REF_HINT_J, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"Current E_ref differs from the declared global P95: {cache_cfg.energy_ref_j}")
    if cache_cfg.initial_cache_seed != 4401:
        raise ValueError(
            "The current initial Cache seed differs from the prior P95 report's seed 4401; "
            "the historical reference contract cannot be replayed as written."
        )
    return cfg, cache_cfg, catalog, pool_manifest, catalog_path


def verify_episode_manifest(
    seed: int,
    pool_manifest: dict[str, Any],
    *,
    verify_scenarios: bool = False,
) -> tuple[Path, dict[str, Any], dict[str, str]]:
    episode_dir = POOL_ROOT / f"episode_{seed}"
    manifest_path = episode_dir / "episode_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training Episode {seed} has no manifest")
    actual_manifest_hash = sha256_file(manifest_path)
    expected_manifest_hash = pool_manifest.get("episode_manifest_sha256", {}).get(str(seed))
    if expected_manifest_hash != actual_manifest_hash:
        raise ValueError(f"Episode {seed} manifest is not attested by the Training pool manifest")
    episode_manifest = read_json(manifest_path)
    if int(episode_manifest.get("episode_seed", -1)) != seed:
        raise ValueError(f"Episode manifest seed mismatch for {seed}")
    if str(episode_manifest.get("split", "")).lower() != "train":
        raise ValueError(f"Episode {seed} is not Training data")
    if int(episode_manifest.get("episode_slots", -1)) != 300:
        raise ValueError(f"Episode {seed} does not have 300 slots")
    needed = ["tasks.csv", "trajectory.csv", "vehicle_roles.json"]
    if verify_scenarios:
        needed.append("prediction_scenarios.npz")
    artifact_hashes = episode_manifest.get("artifact_sha256", {})
    verified: dict[str, str] = {}
    for name in needed:
        path = episode_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Training Episode {seed} is missing {name}")
        expected = artifact_hashes.get(name)
        actual = sha256_file(path)
        if not expected or actual != expected:
            raise ValueError(f"Training Episode {seed} artifact hash mismatch: {name}")
        verified[name] = actual
    return episode_dir, episode_manifest, verified


def load_requested_training_slices(
    seeds_to_slots: dict[int, set[int]],
    pool_manifest: dict[str, Any],
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[int, dict[str, Any]]]:
    """Read only requested Training task rows and trajectory time slices."""
    result: dict[tuple[int, int], dict[str, Any]] = {}
    episode_provenance: dict[int, dict[str, Any]] = {}
    for seed, requested_slots in sorted(seeds_to_slots.items()):
        episode_dir, manifest, hashes = verify_episode_manifest(seed, pool_manifest)
        roles = read_json(episode_dir / "vehicle_roles.json")
        service_ids = [str(value) for value in roles.get("service_vehicles", [])]
        task_ids = [str(value) for value in roles.get("task_vehicles", [])]
        if len(service_ids) != 5 or len(task_ids) != 20:
            raise ValueError(f"Episode {seed} role file is not the formal 5 SV/20 TV cohort")
        scenarios_path = episode_dir / "prediction_scenarios.npz"
        # The NPZ is read only for its canonical service-node order; prediction values are not used.
        scenario_hashes = manifest.get("artifact_sha256", {})
        scenario_actual_hash = sha256_file(scenarios_path)
        if scenario_hashes.get("prediction_scenarios.npz") != scenario_actual_hash:
            raise ValueError(f"Episode {seed} prediction_scenarios.npz hash mismatch")
        with np.load(scenarios_path, allow_pickle=False) as scenario_archive:
            scenario_service_ids = [str(value) for value in scenario_archive["service_vehicle_ids"].tolist()]
        if scenario_service_ids != service_ids:
            raise ValueError(f"Episode {seed} service node order differs across role and scenario metadata")
        hashes["prediction_scenarios.npz"] = scenario_actual_hash

        slot_records: dict[int, list[dict[str, Any]]] = {slot: [] for slot in requested_slots}
        slot_times: dict[int, float] = {}
        with (episode_dir / "tasks.csv").open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                slot = int(row["slot"])
                if slot not in requested_slots:
                    continue
                time_s = float(row["time_s"])
                if slot in slot_times and not math.isclose(slot_times[slot], time_s, rel_tol=0.0, abs_tol=1.0e-9):
                    raise ValueError(f"Episode {seed} slot {slot} has inconsistent task time_s")
                slot_times[slot] = time_s
                if int(row["arrival"]) != 1:
                    continue
                input_bits = float(row["input_size"])
                cpu_cycles = float(row["cpu_cycles"])
                if input_bits <= 0.0 or cpu_cycles <= 0.0:
                    raise ValueError(f"Episode {seed} slot {slot} contains invalid arrived workload")
                vehicle_id = str(row["vehicle_id"])
                slot_records[slot].append(
                    {
                        "task": task_allocation.Task(
                            task_id=vehicle_id,
                            vehicle_id=vehicle_id,
                            service_id=int(row["service_id"]),
                            input_bits=input_bits,
                            cpu_cycles=cpu_cycles,
                        ),
                        "row_order": len(slot_records[slot]),
                    }
                )
        missing_times = requested_slots - set(slot_times)
        if missing_times:
            raise ValueError(f"Episode {seed} tasks.csv misses requested slots {sorted(missing_times)}")
        needed_vehicle_ids = set(service_ids)
        for records in slot_records.values():
            needed_vehicle_ids.update(item["task"].vehicle_id for item in records)
        time_to_slots: dict[float, list[int]] = defaultdict(list)
        for slot, time_s in slot_times.items():
            time_to_slots[round(time_s, 9)].append(slot)
        positions_by_time: dict[float, dict[str, np.ndarray]] = {
            key: {} for key in time_to_slots
        }
        with (episode_dir / "trajectory.csv").open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                vehicle_id = str(row["vehicle_id"])
                if vehicle_id not in needed_vehicle_ids:
                    continue
                time_key = round(float(row["time_s"]), 9)
                if time_key not in positions_by_time:
                    continue
                positions_by_time[time_key][vehicle_id] = np.asarray(
                    [float(row["x_m"]), float(row["y_m"])], dtype=np.float64
                )
        for slot in sorted(requested_slots):
            time_s = slot_times[slot]
            positions = positions_by_time[round(time_s, 9)]
            task_ids_in_slot = {item["task"].vehicle_id for item in slot_records[slot]}
            required_positions = set(service_ids) | task_ids_in_slot
            missing_positions = required_positions - set(positions)
            if missing_positions:
                raise ValueError(
                    f"Episode {seed} slot {slot} has missing task/SV positions: {sorted(missing_positions)}"
                )
            result[(seed, slot)] = {
                "episode_seed": seed,
                "slot": slot,
                "time_s": float(time_s),
                "tasks": [item["task"] for item in slot_records[slot]],
                "vehicle_positions": positions,
                "service_vehicle_ids": service_ids,
                "task_vehicle_ids": task_ids,
                "task_rows": len(slot_records[slot]),
                "episode_dir": episode_dir,
                "manifest_sha256": sha256_file(episode_dir / "episode_manifest.json"),
            }
        episode_provenance[seed] = {
            "episode_manifest_sha256": sha256_file(episode_dir / "episode_manifest.json"),
            "artifact_sha256": hashes,
            "role_metadata_sha256": hashes["vehicle_roles.json"],
            "task_rows_sampled": int(sum(len(value) for value in slot_records.values())),
        }
    return result, episode_provenance


def cache_nodes_for_slice(slice_data: dict[str, Any], cache_cfg: Any) -> dict[str, Any]:
    return cache_configuration.build_cache_nodes(
        {"service_vehicle_ids": np.asarray(slice_data["service_vehicle_ids"], dtype=str)},
        cache_cfg,
    )


def initial_cache_for_slice(slice_data: dict[str, Any], catalog: dict[str, Any], cache_cfg: Any) -> np.ndarray:
    nodes = cache_nodes_for_slice(slice_data, cache_cfg)
    return cache_configuration.popularity_aware_initial_cache(catalog, nodes, cache_cfg)


def service_positions_for_slice(slice_data: dict[str, Any], cache_nodes: dict[str, Any], uav_positions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return task_allocation.build_service_node_positions(
        cache_nodes,
        slice_data["vehicle_positions"],
        uav_positions,
    )


def compute_reference_policy_metrics(
    slices: dict[tuple[int, int], dict[str, Any]],
    calibration_keys: list[tuple[int, int]],
    *,
    cfg: dict[str, Any],
    cache_cfg: Any,
    catalog: dict[str, Any],
    cloud_cpu_capacity: float,
    task_order: str,
) -> dict[str, Any]:
    """Reproduce the previous equal-share least-loaded deterministic policy."""
    scenario_records: list[dict[str, Any]] = []
    delay_samples: list[float] = []
    energy_samples: list[float] = []
    input_kb_samples: list[float] = []
    cycle_samples: list[float] = []
    assignments_by_slot: dict[str, dict[str, str]] = {}
    fixed_uav = uav_configuration.initial_uav_positions_from_project(cfg)
    for key in calibration_keys:
        data = slices[key]
        tasks = list(data["tasks"])
        if task_order == "task_id_ascending":
            tasks.sort(key=lambda value: task_allocation._stable_id_key(value.task_id))
        elif task_order != "csv_row_order":
            raise ValueError(f"Unsupported deterministic task ordering: {task_order}")
        nodes = cache_nodes_for_slice(data, cache_cfg)
        cache = initial_cache_for_slice(data, catalog, cache_cfg)
        service_positions = service_positions_for_slice(data, nodes, fixed_uav)
        targets = task_allocation.build_feasible_targets(
            tasks,
            cache,
            nodes,
            vehicle_positions=data["vehicle_positions"],
            service_node_positions=service_positions,
            cache_config=cache_cfg,
            cloud_fallback_mode=task_allocation.RESOURCE_AWARE_FALLBACK,
        )
        order = {str(node_id): index for index, node_id in enumerate(nodes["node_ids"])}
        load = Counter()
        matching: dict[str, str] = {}
        for task in tasks:
            edge_targets = [target for target in targets[task.task_id] if target != task_allocation.CLOUD]
            selected = (
                min(edge_targets, key=lambda node_id: (load[node_id], order[str(node_id)]))
                if edge_targets
                else task_allocation.CLOUD
            )
            matching[task.task_id] = str(selected)
            if selected != task_allocation.CLOUD:
                load[str(selected)] += 1
        assignments_by_slot[f"{key[0]}/{key[1]}"] = matching
        edge_groups = task_allocation._group_edge_tasks(tasks, matching, nodes)
        cloud_tasks = [task for task in tasks if matching[task.task_id] == task_allocation.CLOUD]
        cloud_rate = float(cache_cfg.cloud_total_bandwidth_bps)
        cloud_cpu_per_task = float(cloud_cpu_capacity / len(cloud_tasks)) if cloud_tasks else 0.0
        slot_delays: list[float] = []
        slot_energies: list[float] = []
        for node_id, node_tasks in edge_groups.items():
            role = task_allocation._edge_node_role(node_id, nodes)
            assert role is not None
            fraction = 1.0 / len(node_tasks)
            for task in node_tasks:
                gain = task_allocation._channel_gain(
                    role,
                    data["vehicle_positions"][task.vehicle_id],
                    service_positions[node_id],
                    cache_cfg,
                )
                edge = system_model.edge_execution(
                    task.input_bits,
                    task.cpu_cycles,
                    task_allocation._node_bandwidth(role, cache_cfg),
                    fraction,
                    cache_cfg.tx_power_w,
                    gain,
                    cache_cfg.noise_power_w,
                    task_allocation._node_cpu(role, cache_cfg),
                    fraction,
                    noise_model=cache_cfg.noise_model,
                    noise_psd_w_per_hz=cache_cfg.noise_psd_w_per_hz,
                    receiver_noise_figure_db=cache_cfg.receiver_noise_figure_db,
                )
                slot_delays.append(float(edge["total_delay_s"]))
                slot_energies.append(float(edge["vehicle_energy_j"]))
        for task in cloud_tasks:
            cloud = system_model.cloud_execution(
                task.input_bits,
                task.cpu_cycles,
                cloud_rate,
                cloud_cpu_per_task,
                cache_cfg.cloud_delay_factor,
                cache_cfg.tx_power_w,
            )
            slot_delays.append(float(cloud["total_delay_s"]))
            slot_energies.append(float(cloud["vehicle_energy_j"]))
        if len(slot_delays) != len(tasks) or len(slot_energies) != len(tasks):
            raise RuntimeError("Reference allocation did not evaluate every sampled task")
        delay_samples.extend(slot_delays)
        energy_samples.extend(slot_energies)
        input_kb_samples.extend(task.input_bits / 8.0 / 1000.0 for task in tasks)
        cycle_samples.extend(task.cpu_cycles for task in tasks)
        scenario_records.append(
            {
                "episode_seed": key[0],
                "slot": key[1],
                "task_count": len(tasks),
                "edge_task_count": len(tasks) - len(cloud_tasks),
                "cloud_task_count": len(cloud_tasks),
                "cloud_cpu_capacity_cycles_per_s": float(cloud_cpu_capacity),
                "mean_delay_s": float(statistics.mean(slot_delays)) if slot_delays else None,
                "mean_vehicle_upload_energy_j": float(statistics.mean(slot_energies)) if slot_energies else None,
            }
        )
    return {
        "task_count": len(delay_samples),
        "delay_s": delay_samples,
        "energy_j": energy_samples,
        "input_kb": input_kb_samples,
        "cpu_cycles": cycle_samples,
        "delay_stats": qstats(delay_samples),
        "energy_stats": qstats(energy_samples),
        "input_kb_stats": qstats(input_kb_samples),
        "cpu_cycles_stats": qstats(cycle_samples),
        "episode_slot_count": len(calibration_keys),
        "assignment_order": task_order,
        "slot_records": scenario_records,
        "assignments": assignments_by_slot,
    }


def make_reference_schemes(
    tasks: list[task_allocation.Task],
    formal_t_ref_s: float,
    formal_e_ref_j: float,
    recal_t_ref_s: float,
    recal_e_ref_j: float,
) -> dict[str, dict[str, tuple[float, float]]]:
    schemes: dict[str, dict[str, tuple[float, float]]] = {}
    for task in tasks:
        local_t = task.cpu_cycles / LOCAL_CPU_ASSUMED_CYCLES_PER_S
        local_e = LOCAL_KAPPA_ASSUMED * LOCAL_CPU_ASSUMED_CYCLES_PER_S**2 * task.cpu_cycles
        schemes.setdefault("GLOBAL_P95_FORMAL_REFS", {})[task.task_id] = (formal_t_ref_s, formal_e_ref_j)
        schemes.setdefault("GLOBAL_P95_RECALIBRATED_20G", {})[task.task_id] = (recal_t_ref_s, recal_e_ref_j)
        schemes.setdefault("LOCAL_TIME_ONLY", {})[task.task_id] = (local_t, formal_e_ref_j)
        schemes.setdefault("LOCAL_ENERGY_ONLY", {})[task.task_id] = (formal_t_ref_s, local_e)
        schemes.setdefault("LOCAL_COMBINED", {})[task.task_id] = (local_t, local_e)
        for alpha in RELATIVE_DEADLINE_ALPHAS:
            key = f"RELATIVE_DEADLINE_ALPHA_{alpha:g}"
            schemes.setdefault(key, {})[task.task_id] = (alpha * local_t, formal_e_ref_j)
    return schemes


def shadow_evaluate_matching_with_kkt(
    tasks: list[task_allocation.Task],
    matching: dict[str, str],
    vehicle_positions: dict[str, np.ndarray],
    service_node_positions: dict[str, np.ndarray],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_cfg: Any,
    references: dict[str, tuple[float, float]],
    kkt_cache: Any | None = None,
) -> task_allocation.KKTMatchingEvaluation:
    """Evaluate task-specific reference coefficients using official KKT cores."""
    if set(matching) != {task.task_id for task in tasks}:
        raise ValueError("Shadow KKT matching task IDs do not match the task sample")
    if set(references) != set(matching):
        raise ValueError("Shadow KKT reference map does not cover the task sample")
    if kkt_cache is None:
        kkt_cache = task_allocation.KKTResourceAllocationCache()
    kkt_cfg = task_allocation.kkt_config_from_project(project_cfg)
    grouped = task_allocation._group_edge_tasks(tasks, matching, cache_nodes)
    edge_results: dict[str, task_allocation.KKTResourceAllocationResult] = {}

    for node_id, node_tasks in grouped.items():
        role = task_allocation._edge_node_role(node_id, cache_nodes)
        if role is None:
            raise RuntimeError(f"Unexpected Cloud role for Edge node {node_id}")

        def compute_edge_pool() -> task_allocation.KKTResourceAllocationResult:
            cpu_tasks = [task for task in node_tasks if task.execution_stage in {"NEW", "COMPUTING"}]
            upload_tasks = [task for task in node_tasks if task.execution_stage in {"NEW", "UPLOADING"}]
            unsupported = [
                task.execution_stage for task in node_tasks
                if task.execution_stage not in {"NEW", "UPLOADING", "COMPUTING", "NETWORK_DELAY"}
            ]
            if unsupported:
                raise ValueError(f"Unsupported Edge task stages in reference experiment: {unsupported}")
            if kkt_cfg.min_cpu_fraction * len(cpu_tasks) > 1.0 or kkt_cfg.min_bandwidth_fraction * len(upload_tasks) > 1.0:
                return task_allocation.KKTResourceAllocationResult(False, f"{node_id} resource epsilon infeasible", {}, {}, {}, {}, {}, {})
            cpu_values = np.asarray([], dtype=np.float64)
            bw_values = np.asarray([], dtype=np.float64)
            diagnostics = []
            cpu_multiplier = bw_multiplier = 0.0
            if cpu_tasks:
                cpu_coeff = [
                    cache_cfg.omega_t * task.cpu_cycles
                    / (references[task.task_id][0] * task_allocation._node_cpu(role, cache_cfg))
                    for task in cpu_tasks
                ]
                kkt_cache.solver_calls += 1
                cpu_values, cpu_multiplier, ok, reason, cpu_diag = task_allocation.solve_kkt_fraction_allocation(
                    cpu_coeff,
                    kkt_cfg.min_cpu_fraction,
                    kkt_cfg.bisection_tolerance,
                    kkt_cfg.max_bisection_iterations,
                    return_diagnostics=True,
                )
                kkt_cache.record_diagnostics(cpu_diag)
                diagnostics.append(cpu_diag)
                if not ok:
                    return task_allocation.KKTResourceAllocationResult(False, f"{node_id} CPU KKT infeasible: {reason}", {}, {}, {}, {}, {}, {}, tuple(diagnostics))
            if upload_tasks:
                shares = [float(task.transmit_time_share) for task in upload_tasks]
                if any(not math.isfinite(share) or share <= 0.0 or share > 1.0 for share in shares):
                    raise ValueError(f"Invalid upload time share in {node_id}")
                gains = [
                    task_allocation._channel_gain(
                        role,
                        vehicle_positions[task.vehicle_id],
                        service_node_positions[node_id],
                        cache_cfg,
                    )
                    for task in upload_tasks
                ]
                if any(not math.isfinite(gain) or gain <= 0.0 for gain in gains):
                    return task_allocation.KKTResourceAllocationResult(False, f"{node_id} invalid channel gain", {}, {}, {}, {}, {}, {})
                bw_coeff: list[float] = []
                for task, share in zip(upload_tasks, shares, strict=True):
                    delay_weight = cache_cfg.omega_t / (references[task.task_id][0] * share)
                    energy_weight = cache_cfg.omega_e * cache_cfg.tx_power_w / references[task.task_id][1]
                    if cache_cfg.noise_model == system_model.BANDWIDTH_DEPENDENT_NOISE_V1:
                        bw_coeff.append(task.input_bits * (delay_weight + energy_weight))
                    else:
                        base_rate = task_allocation._base_rate(
                            role,
                            vehicle_positions[task.vehicle_id],
                            service_node_positions[node_id],
                            cache_cfg,
                        )
                        if base_rate <= 0.0:
                            return task_allocation.KKTResourceAllocationResult(False, f"{node_id} base rate <= 0", {}, {}, {}, {}, {}, {})
                        bw_coeff.append(task.input_bits / base_rate * (delay_weight + energy_weight))
                kkt_cache.solver_calls += 1
                if cache_cfg.noise_model == system_model.BANDWIDTH_DEPENDENT_NOISE_V1:
                    if cache_cfg.noise_psd_w_per_hz is None or cache_cfg.receiver_noise_figure_db is None:
                        raise ValueError("The formal V1 noise configuration is incomplete")
                    bw_values, bw_multiplier, ok, reason, bw_diag = task_allocation.solve_kkt_bandwidth_dependent_noise_allocation(
                        bw_coeff,
                        task_allocation._node_bandwidth(role, cache_cfg),
                        cache_cfg.tx_power_w,
                        gains,
                        cache_cfg.noise_psd_w_per_hz,
                        cache_cfg.receiver_noise_figure_db,
                        kkt_cfg.min_bandwidth_fraction,
                        kkt_cfg.bisection_tolerance,
                        kkt_cfg.max_bisection_iterations,
                        return_diagnostics=True,
                    )
                else:
                    bw_values, bw_multiplier, ok, reason, bw_diag = task_allocation.solve_kkt_fraction_allocation(
                        bw_coeff,
                        kkt_cfg.min_bandwidth_fraction,
                        kkt_cfg.bisection_tolerance,
                        kkt_cfg.max_bisection_iterations,
                        return_diagnostics=True,
                    )
                kkt_cache.record_diagnostics(bw_diag)
                diagnostics.append(bw_diag)
                if not ok:
                    return task_allocation.KKTResourceAllocationResult(False, f"{node_id} bandwidth KKT infeasible: {reason}", {}, {}, {}, {}, {}, {}, tuple(diagnostics))
            cpu_map = {task.task_id: float(value) for task, value in zip(cpu_tasks, cpu_values, strict=True)}
            bw_map = {task.task_id: float(value) for task, value in zip(upload_tasks, bw_values, strict=True)}
            return task_allocation.KKTResourceAllocationResult(
                True,
                "",
                {node_id: cpu_map},
                {node_id: bw_map},
                {node_id: float(math.fsum(cpu_map.values()))},
                {node_id: float(math.fsum(bw_map.values()))},
                ({node_id: float(cpu_multiplier)} if cpu_tasks else {}),
                ({node_id: float(bw_multiplier)} if upload_tasks else {}),
                tuple(diagnostics),
            )

        node_result = kkt_cache.get_or_compute(
            node_id,
            node_tasks,
            service_node_positions[node_id],
            compute_edge_pool,
        )
        if not node_result.feasible:
            return task_allocation.KKTMatchingEvaluation(
                False, node_result.infeasible_reason, float("-inf"),
                node_result.cpu_allocations, node_result.bandwidth_allocations,
                {}, {}, {}, {}, {}, node_result.per_node_cpu_sum,
                node_result.per_node_bandwidth_sum,
            )
        edge_results[node_id] = node_result

    cloud_tasks = [task for task in tasks if matching[task.task_id] == task_allocation.CLOUD]
    cloud_computing = [task for task in cloud_tasks if task.execution_stage in {"NEW", "COMPUTING"}]
    cloud_capacity = (
        float(cache_cfg.cloud_cpu_capacity_cycles_per_s)
        if cache_cfg.cloud_resource_mode == task_allocation.FINITE_CLOUD_RESOURCE_MODE
        else None
    )
    cloud_alloc: dict[str, float] = {}
    if cloud_computing:
        if cloud_capacity is None or cloud_capacity <= 0.0:
            raise ValueError("Cloud task reference evaluation requires a finite positive CPU pool")
        if kkt_cfg.min_cpu_fraction * len(cloud_computing) > 1.0:
            return task_allocation.KKTMatchingEvaluation(
                False, "Cloud CPU epsilon infeasible", float("-inf"), {}, {}, {}, {}, {}, {}, {}, {}, {},
            )

        def compute_cloud_pool() -> task_allocation.KKTResourceAllocationResult:
            coeff = [
                cache_cfg.omega_t * task.cpu_cycles
                / (references[task.task_id][0] * cloud_capacity)
                for task in cloud_computing
            ]
            kkt_cache.solver_calls += 1
            fractions, multiplier, ok, reason, diagnostics = task_allocation.solve_kkt_fraction_allocation(
                coeff,
                kkt_cfg.min_cpu_fraction,
                kkt_cfg.bisection_tolerance,
                kkt_cfg.max_bisection_iterations,
                return_diagnostics=True,
            )
            kkt_cache.record_diagnostics(diagnostics)
            if not ok:
                return task_allocation.KKTResourceAllocationResult(False, f"Cloud CPU KKT infeasible: {reason}", {}, {}, {}, {}, {}, {}, (diagnostics,))
            fractions = np.asarray(fractions, dtype=np.float64)
            if float(math.fsum(float(value) for value in fractions)) > 1.0:
                fractions = fractions / float(math.fsum(float(value) for value in fractions))
            cpu_by_task = {
                task.task_id: float(fraction) * cloud_capacity
                for task, fraction in zip(cloud_computing, fractions, strict=True)
            }
            used = float(math.fsum(cpu_by_task.values()))
            if used > cloud_capacity:
                largest = max(cpu_by_task, key=cpu_by_task.__getitem__)
                cpu_by_task[largest] -= used - cloud_capacity
                used = float(math.fsum(cpu_by_task.values()))
                if used > cloud_capacity:
                    cpu_by_task[largest] = math.nextafter(cpu_by_task[largest], 0.0)
                    used = float(math.fsum(cpu_by_task.values()))
                if used > cloud_capacity:
                    return task_allocation.KKTResourceAllocationResult(False, "Cloud CPU rounding exceeded capacity", {}, {}, {}, {}, {}, {}, (diagnostics,))
            return task_allocation.KKTResourceAllocationResult(
                True, "", {task_allocation.CLOUD: cpu_by_task}, {},
                {task_allocation.CLOUD: float(math.fsum(float(v) for v in fractions))},
                {}, {task_allocation.CLOUD: float(multiplier)}, {}, (diagnostics,),
            )

        cloud_result = kkt_cache.get_or_compute(
            task_allocation.CLOUD,
            cloud_computing,
            np.zeros(2, dtype=np.float64),
            compute_cloud_pool,
        )
        if not cloud_result.feasible:
            return task_allocation.KKTMatchingEvaluation(
                False, cloud_result.infeasible_reason, float("-inf"), {}, {}, {}, {}, {}, {}, {}, {}, {},
            )
        cloud_alloc = dict(cloud_result.cpu_allocations.get(task_allocation.CLOUD, {}))

    cpu_allocations: dict[str, dict[str, float]] = {}
    bandwidth_allocations: dict[str, dict[str, float]] = {}
    cpu_resource_sums: dict[str, float] = {}
    bandwidth_resource_sums: dict[str, float] = {}
    for node_id, node_result in edge_results.items():
        cpu_allocations.update(node_result.cpu_allocations)
        bandwidth_allocations.update(node_result.bandwidth_allocations)
        cpu_resource_sums.update(node_result.per_node_cpu_sum)
        bandwidth_resource_sums.update(node_result.per_node_bandwidth_sum)

    per_delay: dict[str, float] = {}
    per_energy: dict[str, float] = {}
    per_utility: dict[str, float] = {}
    counts: dict[str, int] = {}
    total_utility = 0.0
    for task in tasks:
        target = matching[task.task_id]
        counts[target] = counts.get(target, 0) + 1
        if target == task_allocation.CLOUD:
            q = float(task.transmit_time_share)
            remaining_upload = task.input_bits if task.execution_stage in {"NEW", "UPLOADING"} else 0.0
            cpu_rate = cloud_alloc.get(task.task_id, float(cache_cfg.cloud_total_cpu_cycles_per_s))
            upload_delay = (
                cache_cfg.cloud_delay_factor * remaining_upload
                / (cache_cfg.cloud_total_bandwidth_bps * q)
            )
            compute_delay = task.cpu_cycles / cpu_rate
            delay = float(upload_delay + compute_delay + (
                task.remaining_network_delay_s if task.execution_stage == "NETWORK_DELAY" else 0.0
            ))
            energy = float(cache_cfg.tx_power_w * remaining_upload / cache_cfg.cloud_total_bandwidth_bps)
        else:
            role = task_allocation._edge_node_role(target, cache_nodes)
            assert role is not None
            cpu_fraction = cpu_allocations.get(target, {}).get(task.task_id, 1.0)
            bandwidth_fraction = bandwidth_allocations.get(target, {}).get(task.task_id, 1.0)
            q = float(task.transmit_time_share)
            gain = task_allocation._channel_gain(
                role,
                vehicle_positions[task.vehicle_id],
                service_node_positions[target],
                cache_cfg,
            )
            edge = system_model.edge_execution(
                task.input_bits / q if task.execution_stage in {"NEW", "UPLOADING"} else 0.0,
                task.cpu_cycles,
                task_allocation._node_bandwidth(role, cache_cfg),
                bandwidth_fraction,
                cache_cfg.tx_power_w,
                gain,
                cache_cfg.noise_power_w,
                task_allocation._node_cpu(role, cache_cfg),
                cpu_fraction,
                noise_model=cache_cfg.noise_model,
                noise_psd_w_per_hz=cache_cfg.noise_psd_w_per_hz,
                receiver_noise_figure_db=cache_cfg.receiver_noise_figure_db,
            )
            delay = float(edge["total_delay_s"])
            energy = float(edge["vehicle_energy_j"] * q)
        t_ref, e_ref = references[task.task_id]
        utility = float(system_model.task_utility(
            delay, energy, t_ref, e_ref, cache_cfg.omega_t, cache_cfg.omega_e
        ))
        per_delay[task.task_id] = delay
        per_energy[task.task_id] = energy
        per_utility[task.task_id] = utility
        total_utility += utility
    return task_allocation.KKTMatchingEvaluation(
        True,
        "",
        float(total_utility),
        cpu_allocations,
        bandwidth_allocations,
        per_delay,
        per_energy,
        per_utility,
        dict(matching),
        counts,
        cpu_resource_sums,
        bandwidth_resource_sums,
        cloud_alloc,
        cloud_capacity,
        float(math.fsum(cloud_alloc.values())),
    )


def validate_shadow_resources(
    evaluation: task_allocation.KKTMatchingEvaluation,
    matching: dict[str, str],
    cfg: dict[str, Any],
    cache_cfg: Any,
) -> dict[str, Any]:
    if not evaluation.resource_feasible:
        return {"feasible": False, "reason": evaluation.infeasible_reason}
    tolerance = 2.0e-8
    min_cpu = float(project_config.get_config_value(cfg, "algorithm.kkt.min_cpu_fraction"))
    min_bandwidth = float(project_config.get_config_value(cfg, "algorithm.kkt.min_bandwidth_fraction"))
    for node_id, allocations in evaluation.cpu_allocations.items():
        if node_id == task_allocation.CLOUD:
            continue
        if not all(math.isfinite(float(value)) and float(value) >= min_cpu - tolerance for value in allocations.values()):
            return {"feasible": False, "reason": f"invalid CPU share for {node_id}"}
        if math.fsum(float(value) for value in allocations.values()) > 1.0 + tolerance:
            return {"feasible": False, "reason": f"CPU fraction over capacity for {node_id}"}
    for node_id, allocations in evaluation.bandwidth_allocations.items():
        if node_id == task_allocation.CLOUD:
            continue
        if not all(math.isfinite(float(value)) and float(value) >= min_bandwidth - tolerance for value in allocations.values()):
            return {"feasible": False, "reason": f"invalid bandwidth share for {node_id}"}
        if math.fsum(float(value) for value in allocations.values()) > 1.0 + tolerance:
            return {"feasible": False, "reason": f"bandwidth fraction over capacity for {node_id}"}
    cloud_tasks = [task_id for task_id, target in matching.items() if target == task_allocation.CLOUD]
    cloud_rates = evaluation.cloud_cpu_allocations_cycles_per_s
    if cloud_tasks:
        capacity = float(cache_cfg.cloud_cpu_capacity_cycles_per_s)
        if set(cloud_rates) != set(cloud_tasks):
            return {"feasible": False, "reason": "Cloud CPU allocation coverage mismatch"}
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in cloud_rates.values()):
            return {"feasible": False, "reason": "non-positive or non-finite Cloud CPU allocation"}
        used = math.fsum(float(value) for value in cloud_rates.values())
        if used > capacity + max(1.0e-4, capacity * 1.0e-10):
            return {"feasible": False, "reason": "Cloud CPU finite-pool capacity exceeded"}
    if set(evaluation.per_task_utility) != set(matching):
        return {"feasible": False, "reason": "per-task utility coverage mismatch"}
    if any(not math.isfinite(float(value)) for value in evaluation.per_task_utility.values()):
        return {"feasible": False, "reason": "non-finite task utility"}
    if not math.isclose(
        math.fsum(evaluation.per_task_utility.values()),
        evaluation.task_utility_sum,
        rel_tol=0.0,
        abs_tol=5.0e-8,
    ):
        return {"feasible": False, "reason": "task utility sum mismatch"}
    return {
        "feasible": True,
        "cloud_cpu_used_cycles_per_s": evaluation.cloud_cpu_used_cycles_per_s,
        "cloud_cpu_capacity_cycles_per_s": evaluation.cloud_cpu_capacity_cycles_per_s,
    }


def max_mapping_difference(left: dict[str, Any], right: dict[str, Any]) -> float:
    keys = set(left) | set(right)
    differences = []
    for key in keys:
        if isinstance(left.get(key), dict) or isinstance(right.get(key), dict):
            left_nested = left.get(key, {})
            right_nested = right.get(key, {})
            nested_keys = set(left_nested) | set(right_nested)
            differences.extend(abs(float(left_nested.get(item, 0.0)) - float(right_nested.get(item, 0.0))) for item in nested_keys)
        else:
            differences.append(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))))
    return max(differences, default=0.0)


def compare_formal_and_shadow(
    formal: task_allocation.KKTMatchingEvaluation,
    shadow: task_allocation.KKTMatchingEvaluation,
) -> dict[str, Any]:
    fields = {
        "task_utility_sum": abs(formal.task_utility_sum - shadow.task_utility_sum),
        "per_task_delay_s": max_mapping_difference(formal.per_task_delay_s, shadow.per_task_delay_s),
        "per_task_vehicle_energy_j": max_mapping_difference(formal.per_task_vehicle_energy_j, shadow.per_task_vehicle_energy_j),
        "per_task_utility": max_mapping_difference(formal.per_task_utility, shadow.per_task_utility),
        "edge_cpu_fraction": max_mapping_difference(formal.cpu_allocations, shadow.cpu_allocations),
        "edge_bandwidth_fraction": max_mapping_difference(formal.bandwidth_allocations, shadow.bandwidth_allocations),
        "cloud_cpu_cycles_per_s": max_mapping_difference(formal.cloud_cpu_allocations_cycles_per_s, shadow.cloud_cpu_allocations_cycles_per_s),
    }
    return {"max_abs_difference_by_metric": fields, "maximum_abs_difference": max(fields.values(), default=0.0)}


def summarize_evaluation(
    tasks: list[task_allocation.Task],
    matching: dict[str, str],
    evaluation: task_allocation.KKTMatchingEvaluation,
) -> dict[str, Any]:
    delays = [float(evaluation.per_task_delay_s[task.task_id]) for task in tasks]
    energies = [float(evaluation.per_task_vehicle_energy_j[task.task_id]) for task in tasks]
    counts = Counter(str(target) for target in matching.values())
    return {
        "resource_feasible": bool(evaluation.resource_feasible),
        "task_utility_internal_to_scheme": float(evaluation.task_utility_sum),
        "task_count": len(tasks),
        "matching_node_task_counts": dict(counts),
        "cloud_task_count": int(counts.get(task_allocation.CLOUD, 0)),
        "cloud_rate": float(counts.get(task_allocation.CLOUD, 0) / len(tasks)) if tasks else 0.0,
        "total_delay_s": float(math.fsum(delays)),
        "mean_delay_s": float(statistics.mean(delays)) if delays else None,
        "total_vehicle_upload_energy_j": float(math.fsum(energies)),
        "mean_vehicle_upload_energy_j": float(statistics.mean(energies)) if energies else None,
        "edge_cpu_fraction_by_node": {
            node: float(math.fsum(float(value) for value in values.values()))
            for node, values in evaluation.cpu_allocations.items()
            if node != task_allocation.CLOUD
        },
        "edge_bandwidth_fraction_by_node": {
            node: float(math.fsum(float(value) for value in values.values()))
            for node, values in evaluation.bandwidth_allocations.items()
            if node != task_allocation.CLOUD
        },
        "cloud_cpu_allocations_ghz_by_task": {
            task_id: float(value) / 1.0e9
            for task_id, value in evaluation.cloud_cpu_allocations_cycles_per_s.items()
        },
        "cloud_cpu_used_ghz": float(evaluation.cloud_cpu_used_cycles_per_s / 1.0e9),
        "cloud_cpu_capacity_ghz": (
            None if evaluation.cloud_cpu_capacity_cycles_per_s is None
            else float(evaluation.cloud_cpu_capacity_cycles_per_s / 1.0e9)
        ),
        "edge_cpu_fraction_by_task_node": {
            f"{node}:{task_id}": float(value)
            for node, values in evaluation.cpu_allocations.items()
            if node != task_allocation.CLOUD
            for task_id, value in values.items()
        },
        "edge_bandwidth_fraction_by_task_node": {
            f"{node}:{task_id}": float(value)
            for node, values in evaluation.bandwidth_allocations.items()
            if node != task_allocation.CLOUD
            for task_id, value in values.items()
        },
    }


def compare_allocation_metrics(left: dict[str, Any], right: dict[str, Any], task_ids: set[str]) -> dict[str, Any]:
    def flat_mean_abs(left_map: dict[str, float], right_map: dict[str, float]) -> float:
        keys = set(left_map) | set(right_map)
        if not keys:
            return 0.0
        return float(statistics.mean(abs(float(left_map.get(key, 0.0)) - float(right_map.get(key, 0.0))) for key in keys))

    left_cloud = left.get("cloud_cpu_allocations_ghz_by_task", {})
    right_cloud = right.get("cloud_cpu_allocations_ghz_by_task", {})
    cloud_diff = [abs(float(left_cloud.get(task_id, 0.0)) - float(right_cloud.get(task_id, 0.0))) for task_id in task_ids]
    return {
        "mean_abs_edge_cpu_fraction_change": flat_mean_abs(
            left.get("edge_cpu_fraction_by_task_node", {}),
            right.get("edge_cpu_fraction_by_task_node", {}),
        ),
        "mean_abs_edge_bandwidth_fraction_change": flat_mean_abs(
            left.get("edge_bandwidth_fraction_by_task_node", {}),
            right.get("edge_bandwidth_fraction_by_task_node", {}),
        ),
        "mean_abs_cloud_cpu_ghz_change_per_task": float(statistics.mean(cloud_diff)) if cloud_diff else 0.0,
        "cloud_cpu_used_ghz_change": float(right.get("cloud_cpu_used_ghz", 0.0) - left.get("cloud_cpu_used_ghz", 0.0)),
    }


def scheme_reference_stats(
    tasks: list[task_allocation.Task],
    formal_t: float,
    formal_e: float,
    p95_20_t: float,
    p95_20_e: float,
) -> dict[str, Any]:
    # Task IDs repeat across Episodes/Slots in the calibration sample. Give each
    # observation a unique temporary key so the stats cover all sampled tasks.
    schemes: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for index, task in enumerate(tasks):
        local_t = task.cpu_cycles / LOCAL_CPU_ASSUMED_CYCLES_PER_S
        local_e = LOCAL_KAPPA_ASSUMED * LOCAL_CPU_ASSUMED_CYCLES_PER_S**2 * task.cpu_cycles
        task_key = f"sample-{index}"
        schemes["GLOBAL_P95_FORMAL_REFS"][task_key] = (formal_t, formal_e)
        schemes["GLOBAL_P95_RECALIBRATED_20G"][task_key] = (p95_20_t, p95_20_e)
        schemes["LOCAL_TIME_ONLY"][task_key] = (local_t, formal_e)
        schemes["LOCAL_ENERGY_ONLY"][task_key] = (formal_t, local_e)
        schemes["LOCAL_COMBINED"][task_key] = (local_t, local_e)
        for alpha in RELATIVE_DEADLINE_ALPHAS:
            schemes[f"RELATIVE_DEADLINE_ALPHA_{alpha:g}"][task_key] = (alpha * local_t, formal_e)
    result = {}
    formal_delay_coeff = 0.5 / formal_t
    formal_energy_coeff = 0.5 / formal_e
    for name, mapping in schemes.items():
        t_values = [value[0] for value in mapping.values()]
        e_values = [value[1] for value in mapping.values()]
        result[name] = {
            "t_ref_s": qstats(t_values),
            "e_ref_j": qstats(e_values),
            "t_ref_ratio_vs_formal_global": qstats([value / formal_t for value in t_values]),
            "e_ref_ratio_vs_formal_global": qstats([value / formal_e for value in e_values]),
            "effective_delay_coefficient_omega_over_tref": qstats([0.5 / value for value in t_values]),
            "effective_energy_coefficient_omega_over_eref": qstats([0.5 / value for value in e_values]),
            "delay_coefficient_ratio_vs_formal_global": qstats([(0.5 / value) / formal_delay_coeff for value in t_values]),
            "energy_coefficient_ratio_vs_formal_global": qstats([(0.5 / value) / formal_energy_coeff for value in e_values]),
        }
    return result


def build_sample_slots() -> tuple[dict[int, set[int]], list[tuple[int, int]]]:
    calibration_keys = [(seed, slot) for seed in CALIBRATION_SEEDS for slot in CALIBRATION_SLOTS]
    requested: dict[int, set[int]] = defaultdict(set)
    for seed, slot in calibration_keys:
        requested[seed].add(slot)
    for seed, slot, _ in MATCHING_SAMPLES:
        requested[seed].add(slot)
    return dict(requested), calibration_keys


def validate_resources_and_assignments(
    context: dict[str, Any],
    tasks: list[task_allocation.Task],
    nodes: dict[str, Any],
    cache: np.ndarray,
    positions: dict[str, np.ndarray],
    service_positions: dict[str, np.ndarray],
    result: task_allocation.TransferExchangeMatchingResult,
) -> dict[str, Any]:
    feasible = task_allocation.build_feasible_targets(
        tasks,
        cache,
        nodes,
        vehicle_positions=positions,
        service_node_positions=service_positions,
        cache_config=context["cache_cfg"],
        cloud_fallback_mode=task_allocation.RESOURCE_AWARE_FALLBACK,
    )
    if set(result.matching) != {task.task_id for task in tasks}:
        raise RuntimeError("Transfer-Exchange Matching did not assign every arrived task exactly once")
    for task_id, target in result.matching.items():
        if target not in feasible[task_id]:
            raise RuntimeError(f"Matching selected an illegal target {target} for task {task_id}")
    return feasible


def run_formal_matching(
    context: dict[str, Any],
    data: dict[str, Any],
    nodes: dict[str, Any],
    cache: np.ndarray,
    service_positions: dict[str, np.ndarray],
) -> tuple[Any, Any, Any, dict[str, Any]]:
    kkt_cache = task_allocation.KKTResourceAllocationCache()
    started = time.perf_counter()
    result = task_allocation.optimize_task_matching(
        list(data["tasks"]),
        data["vehicle_positions"],
        service_positions,
        cache.copy(),
        context["cfg"],
        nodes,
        context["cache_cfg"],
        evaluation_mode="incremental",
        kkt_cache=kkt_cache,
        cloud_fallback_mode=task_allocation.RESOURCE_AWARE_FALLBACK,
    )
    runtime = time.perf_counter() - started
    if not result.resource_feasible:
        raise RuntimeError(f"Formal Matching infeasible: {result.infeasible_reason}")
    feasible = validate_resources_and_assignments(context, data["tasks"], nodes, cache, data["vehicle_positions"], service_positions, result)
    full_eval = task_allocation.evaluate_matching_with_kkt(
        list(data["tasks"]), result.matching, data["vehicle_positions"], service_positions,
        context["cfg"], nodes, context["cache_cfg"],
        kkt_cache=task_allocation.KKTResourceAllocationCache(),
    )
    if not full_eval.resource_feasible or not math.isclose(result.task_utility_sum, full_eval.task_utility_sum, rel_tol=0.0, abs_tol=5.0e-8):
        raise RuntimeError("Formal Matching and formal full KKT recheck disagree")
    return result, full_eval, feasible, {
        "matching_runtime_s": float(runtime),
        "matching_kkt_solver_calls": int(kkt_cache.solver_calls),
        "candidate_evaluations": int(result.candidate_evaluations_total),
        "matching_updates": int(result.updates),
    }


def run_shadow_matching(
    context: dict[str, Any],
    data: dict[str, Any],
    nodes: dict[str, Any],
    cache: np.ndarray,
    service_positions: dict[str, np.ndarray],
    references: dict[str, tuple[float, float]],
) -> tuple[Any, task_allocation.KKTMatchingEvaluation, dict[str, Any]]:
    original_evaluator = task_allocation.evaluate_matching_candidate
    kkt_cache = task_allocation.KKTResourceAllocationCache()

    def shadow_candidate_evaluator(
        tasks: list[task_allocation.Task],
        matching: dict[str, str],
        vehicle_positions: dict[str, np.ndarray],
        service_node_positions: dict[str, np.ndarray],
        project_cfg: dict[str, Any],
        cache_nodes: dict[str, Any],
        cache_config: Any,
        *,
        kkt_cache: Any | None = None,
        node_evaluation_cache: Any | None = None,
        timing_profile: dict[str, float] | None = None,
    ) -> task_allocation.KKTMatchingEvaluation:
        del node_evaluation_cache, timing_profile
        return shadow_evaluate_matching_with_kkt(
            tasks,
            matching,
            vehicle_positions,
            service_node_positions,
            project_cfg,
            cache_nodes,
            cache_config,
            references,
            kkt_cache=kkt_cache,
        )

    task_allocation.evaluate_matching_candidate = shadow_candidate_evaluator
    started = time.perf_counter()
    try:
        result = task_allocation.optimize_task_matching(
            list(data["tasks"]),
            data["vehicle_positions"],
            service_positions,
            cache.copy(),
            context["cfg"],
            nodes,
            context["cache_cfg"],
            evaluation_mode="full",
            kkt_cache=kkt_cache,
            cloud_fallback_mode=task_allocation.RESOURCE_AWARE_FALLBACK,
        )
    finally:
        task_allocation.evaluate_matching_candidate = original_evaluator
    runtime = time.perf_counter() - started
    if not result.resource_feasible:
        raise RuntimeError(f"Shadow Matching infeasible: {result.infeasible_reason}")
    evaluation = shadow_evaluate_matching_with_kkt(
        list(data["tasks"]), result.matching, data["vehicle_positions"], service_positions,
        context["cfg"], nodes, context["cache_cfg"], references,
        kkt_cache=task_allocation.KKTResourceAllocationCache(),
    )
    if not evaluation.resource_feasible or not math.isclose(result.task_utility_sum, evaluation.task_utility_sum, rel_tol=0.0, abs_tol=5.0e-8):
        raise RuntimeError("Shadow Matching result and final shadow KKT recheck disagree")
    feasible = task_allocation.build_feasible_targets(
        data["tasks"], cache, nodes,
        vehicle_positions=data["vehicle_positions"],
        service_node_positions=service_positions,
        cache_config=context["cache_cfg"],
        cloud_fallback_mode=task_allocation.RESOURCE_AWARE_FALLBACK,
    )
    for task_id, target in result.matching.items():
        if target not in feasible[task_id]:
            raise RuntimeError(f"Shadow Matching selected illegal target {target} for {task_id}")
    return result, evaluation, {
        "matching_runtime_s": float(runtime),
        "matching_kkt_solver_calls": int(kkt_cache.solver_calls),
        "candidate_evaluations": int(result.candidate_evaluations_total),
        "matching_updates": int(result.updates),
        "evaluation_mode": "full_recompute_exact_candidate_evaluation",
    }


def aggregate_case_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    all_delays: list[float] = []
    all_energies: list[float] = []
    tasks_total = cloud_total = 0
    per_node_counts: Counter[str] = Counter()
    cpu_diff: list[float] = []
    bandwidth_diff: list[float] = []
    cloud_cpu_diff: list[float] = []
    target_changes = 0
    runtime_s: list[float] = []
    kkt_calls = 0
    for record in records:
        metrics = record["metrics"]
        count = int(metrics["task_count"])
        tasks_total += count
        cloud_total += int(metrics["cloud_task_count"])
        per_node_counts.update(metrics["matching_node_task_counts"])
        all_delays.append(float(metrics["total_delay_s"]))
        all_energies.append(float(metrics["total_vehicle_upload_energy_j"]))
        target_changes += int(record.get("matching_target_changes_vs_global", 0))
        runtime_s.append(float(record.get("matching_runtime_s", 0.0)))
        kkt_calls += int(record.get("matching_kkt_solver_calls", 0))
        deltas = record.get("allocation_change_vs_global_matching", {})
        if deltas:
            cpu_diff.append(float(deltas["mean_abs_edge_cpu_fraction_change"]))
            bandwidth_diff.append(float(deltas["mean_abs_edge_bandwidth_fraction_change"]))
            cloud_cpu_diff.append(float(deltas["mean_abs_cloud_cpu_ghz_change_per_task"]))
    return {
        "task_count": tasks_total,
        "cloud_task_count": cloud_total,
        "cloud_rate_task_weighted": float(cloud_total / tasks_total) if tasks_total else 0.0,
        "matching_node_task_counts": dict(per_node_counts),
        "mean_delay_task_weighted_s": float(math.fsum(all_delays) / tasks_total) if tasks_total else None,
        "mean_vehicle_upload_energy_task_weighted_j": float(math.fsum(all_energies) / tasks_total) if tasks_total else None,
        "total_delay_all_slots_s": float(math.fsum(all_delays)),
        "total_vehicle_upload_energy_all_slots_j": float(math.fsum(all_energies)),
        "matching_target_change_count_vs_global": target_changes,
        "mean_abs_edge_cpu_fraction_change": float(statistics.mean(cpu_diff)) if cpu_diff else 0.0,
        "mean_abs_edge_bandwidth_fraction_change": float(statistics.mean(bandwidth_diff)) if bandwidth_diff else 0.0,
        "mean_abs_cloud_cpu_ghz_change_per_task": float(statistics.mean(cloud_cpu_diff)) if cloud_cpu_diff else 0.0,
        "mean_matching_runtime_s": float(statistics.mean(runtime_s)) if runtime_s else 0.0,
        "max_matching_runtime_s": float(max(runtime_s, default=0.0)),
        "total_matching_kkt_solver_calls": kkt_calls,
    }


def render_markdown(report: dict[str, Any]) -> str:
    cfg = report["current_formal_contract"]
    calibration = report["p95_recalibration"]
    lines = [
        "# Training normalization-reference comparison",
        "",
        f"Generated: `{report['created_at_utc']}` using `{report['environment']['python_executable']}`.",
        "",
        "## Scope and contracts",
        "",
        f"- Training-only sample: {report['training_reference_sample']['task_count']} tasks in {report['training_reference_sample']['episode_slot_count']} predeclared Slots from 12 Episodes; no Validation/Test files were read.",
        f"- Matching/KKT sample: {len(report['matching_samples'])} actual Slots with {report['matching_total_task_count']} arrived tasks; Cache and UAV positions are frozen at the formal initial state.",
        f"- Current Cloud pool: {cfg['cloud_cpu_capacity_ghz']:.1f} GHz, fixed per-task uplink {cfg['cloud_uplink_rate_mbps']:.1f} Mbps, delay factor {cfg['cloud_delay_factor']:.1f}; weights {cfg['omega_delay']:.1f}/{cfg['omega_energy']:.1f}.",
        f"- Current configured references: T_ref={cfg['t_ref_s']:.12g} s; E_ref={cfg['e_ref_j']:.12g} J. These are the existing Training P95 references; they were originally calculated under a 40 GHz Cloud pool.",
        "- Across different reference schemes, Q is an internal objective value only. The comparisons below emphasize physical delay, vehicle uplink energy, Cloud assignment share, and resource allocations.",
        "",
        "## P95 provenance and 20 GHz recalculation",
        "",
        f"The earlier deterministic calibration record reports P95 T={P95_FORMAL_T_REF_HINT_S:.12g} s and E={P95_FORMAL_E_REF_HINT_J:.12g} J from 1,642 tasks / 120 Slots / 12 Training Episodes. Its reference assignment used seed-4401 uniform feasible Cache, configured initial UAV positions, least-loaded feasible cached Edge selection, equal per-node Edge CPU/BW shares, 20 Mbps Cloud upload, and equal Cloud CPU shares at 40 GHz.",
        f"Using the task CSV's row order, the 40 GHz replay produced T_P95={calibration['historical_40g_replay']['delay_p95_s']:.12g} s and E_P95={calibration['historical_40g_replay']['energy_p95_j']:.12g} J; this does **not** exactly reproduce the recorded refs. A second diagnostic using ascending numeric task ID reproduces the recorded 40 GHz P95 exactly, but the old report did not record that task iteration order, so this is evidence of an unresolved execution-contract detail, not proof of provenance.",
        f"At current 20 GHz, the row-order replay gives P95 T={calibration['current_20g_recalibration']['delay_p95_s']:.12g} s and E={calibration['current_20g_recalibration']['energy_p95_j']:.12g} J; ascending task-ID order gives T={calibration['current_20g_recalibration_by_order']['task_id_ascending']['delay_p95_s']:.12g} s and E={calibration['current_20g_recalibration_by_order']['task_id_ascending']['energy_p95_j']:.12g} J. The latter remains equal to the recorded global refs under this sample. These are diagnostics only and are not written to formal configuration.",
        f"Historical Stage6 local-compute references T={HISTORICAL_LOCAL_T_REF_S:.12g} s and E={HISTORICAL_LOCAL_E_REF_J:.12g} J came from 3,343 arrivals; exact source Episode/Slot selection was not recorded, so that older local calibration is not exactly reproducible from local provenance. The prior report's mean 722.70 is in KiB; this run finds {calibration['historical_40g_replay']['mean_input_kb']:.2f} decimal KB, equivalent to {calibration['historical_40g_replay']['mean_input_kib']:.2f} KiB.",
        "",
        "## Reference-scale statistics",
        "",
        "P5/P50/P95/mean/min/max are task-level Training statistics (linear percentile interpolation). Local values are diagnostic only.",
        "",
        "| Scheme | T_ref P5 / P50 / P95 / mean (s) | E_ref P5 / P50 / P95 / mean (J) | Median effective delay coefficient ratio | Median effective energy coefficient ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, entry in report["reference_scale_statistics"].items():
        t = entry["t_ref_s"]
        e = entry["e_ref_j"]
        td = entry["delay_coefficient_ratio_vs_formal_global"]
        ed = entry["energy_coefficient_ratio_vs_formal_global"]
        lines.append(
            f"| `{name}` | {t['p5']:.4g} / {t['median']:.4g} / {t['p95']:.4g} / {t['mean']:.4g} | {e['p5']:.4g} / {e['median']:.4g} / {e['p95']:.4g} / {e['mean']:.4g} | {td['median']:.4g}× | {ed['median']:.4g}× |"
        )
    lines.extend([
        "",
        f"The assumed local baseline uses F_local={LOCAL_CPU_ASSUMED_CYCLES_PER_S:.3g} cycles/s and κ={LOCAL_KAPPA_ASSUMED:.3g}; these are not defined by the formal system config or Training tasks. `LOCAL_TIME_ONLY` and `LOCAL_ENERGY_ONLY` isolate one denominator at a time; `LOCAL_COMBINED` changes both. Relative Deadline cases use D=α·T_local for α∈{{0.5,1,2}} while holding E_ref at the formal global value. No real business Deadline or hard-deadline constraint is defined.",
        "",
        "## Fixed global Matching: KKT response only",
        "",
        "The global-reference formal Matching is fixed for this layer. Every scheme re-solves the affected Edge CPU, Edge bandwidth, and finite Cloud CPU KKT pools with that task mapping.",
        "",
        "| Scheme | Mean delay (s) | Mean upload energy (J) | Cloud rate | Mean abs Edge CPU fraction Δ | Mean abs Edge BW fraction Δ | Mean abs Cloud CPU Δ (GHz/task) | Resource feasible |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for name, aggregate in report["fixed_matching_aggregates"].items():
        lines.append(
            f"| `{name}` | {aggregate['mean_delay_task_weighted_s']:.4f} | {aggregate['mean_vehicle_upload_energy_task_weighted_j']:.5f} | {aggregate['cloud_rate_task_weighted']:.3%} | {aggregate['mean_abs_edge_cpu_fraction_change']:.5f} | {aggregate['mean_abs_edge_bandwidth_fraction_change']:.5f} | {aggregate['mean_abs_cloud_cpu_ghz_change_per_task']:.5f} | YES |"
        )
    lines.extend([
        "",
        "The allocation deltas are paired against the formal global P95 KKT allocation, flattened by (node, task); entries absent from a node receive zero for this difference summary. The fixed assignment and current physical input are otherwise identical.",
        "",
        "## Re-optimized Transfer–Exchange Matching + KKT",
        "",
        "Per-task reference maps were supplied to a process-local shadow evaluator. The official `optimize_task_matching` candidate generation, Transfer–Exchange order, termination, target set and KKT numeric solver functions were reused. Global P95 shadow and formal KKT results were checked for equality on all nine fixed assignments. Per-scheme KKT caches are fresh and never shared across reference maps. Full candidate evaluation was used so the formal global-denominator upper-bound pruning path could not silently apply to task-specific denominators.",
        "",
        "| Scheme | Mean delay (s) | Mean upload energy (J) | Cloud rate | Matching targets changed vs global (tasks) | Mean abs Edge CPU Δ | Mean abs Edge BW Δ | Mean abs Cloud CPU Δ (GHz/task) | Mean optimizer time (s) | Total KKT solves |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, aggregate in report["reoptimized_matching_aggregates"].items():
        lines.append(
            f"| `{name}` | {aggregate['mean_delay_task_weighted_s']:.4f} | {aggregate['mean_vehicle_upload_energy_task_weighted_j']:.5f} | {aggregate['cloud_rate_task_weighted']:.3%} | {aggregate['matching_target_change_count_vs_global']} | {aggregate['mean_abs_edge_cpu_fraction_change']:.5f} | {aggregate['mean_abs_edge_bandwidth_fraction_change']:.5f} | {aggregate['mean_abs_cloud_cpu_ghz_change_per_task']:.5f} | {aggregate['mean_matching_runtime_s']:.4f} | {aggregate['total_matching_kkt_solver_calls']} |"
        )
    lines.extend([
        "",
        "Q values are retained in JSON for within-scheme traceability but are not used to rank different denominator schemes. Cloud rate is task-weighted over the same 9 Slots. Full Matching is a Transfer–Exchange local search, not a global assignment optimum.",
        "",
        "## Findings and recommendation",
        "",
        f"- Formal system declares local CPU: **{report['parameter_provenance']['local_cpu_defined_in_formal_system']}**; local κ: **{report['parameter_provenance']['local_energy_coefficient_defined']}**; business Deadline: **{report['parameter_provenance']['business_deadline_defined']}**.",
        f"- Shadow KKT reproduces formal global-reference KKT within {report['matching_validation']['max_shadow_formal_abs_difference']:.3g} maximum absolute difference across checked metrics: **{report['matching_validation']['shadow_matches_formal']}**.",
        f"- Re-optimized matching comparison completed for all {len(report['reoptimized_matching_aggregates'])} schemes and all {len(report['matching_samples'])} Slots: **{report['matching_validation']['full_matching_comparison']}**.",
        f"- The full shadow search and formal incremental search selected identical global-reference assignments on {sum(bool(item['matching_identical']) for item in report['matching_validation']['formal_incremental_vs_shadow_full'])}/{len(report['matching_validation']['formal_incremental_vs_shadow_full'])} Slots.",
        f"- Recommendation: **{report['recommendation']['normalization_recommendation']}**. {report['recommendation']['basis']}",
        "- The local-compute reference is a literature-parameter sensitivity scenario, not a measured vehicle baseline. Relative Deadline scenarios are denominator sensitivities, not business requirements. No scheme is declared superior because its Q has a different scale.",
        "",
        "## Reproducibility and change boundaries",
        "",
        f"- Current config SHA-256: `{report['provenance']['config_sha256']}`; Training pool manifest SHA-256: `{report['provenance']['training_pool_manifest_sha256']}`.",
        f"- Episode manifests and task/trajectory/role/NPZ hashes verified for {len(report['provenance']['episodes_verified'])} Training Episodes; Validation/Test were not opened.",
        f"- Source hashes: task allocation `{report['provenance']['task_allocation_sha256']}`, system model `{report['provenance']['system_model_sha256']}`, catalog `{report['provenance']['service_catalog_sha256']}`.",
        "- `FORMAL_CONFIG_MODIFIED=NO`; `FORMAL_ALGORITHM_MODIFIED=NO`; `MAPPO_TRAINING_EXECUTED=NO`; `VALIDATION_TEST_READ=NO`.",
        "- Limitations: the P95 policy is a deterministic reference allocation rather than a learned/optimized policy; local CPU/κ and Deadline scenarios are assumptions; matching remains Transfer–Exchange local search. Results are conditional on the predeclared 9 Training Slots and 1,642-task reference sample.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    overall_started = time.perf_counter()
    cfg, cache_cfg, catalog, pool_manifest, catalog_path = load_config_and_pool()
    requested_slots, calibration_keys = build_sample_slots()
    slices, episode_provenance = load_requested_training_slices(requested_slots, pool_manifest)
    if len(calibration_keys) != 120:
        raise RuntimeError("Calibration sample must retain the documented 120-slot rule")
    formal_t_ref = float(cache_cfg.time_ref_s)
    formal_e_ref = float(cache_cfg.energy_ref_j)
    current_cloud_capacity = float(cache_cfg.cloud_cpu_capacity_cycles_per_s)
    historical_cfg = replace(
        cache_cfg,
        cloud_cpu_cycles_per_s=HISTORICAL_CLOUD_CPU_GHZ * 1.0e9,
        cloud_total_cpu_cycles_per_s=HISTORICAL_CLOUD_CPU_GHZ * 1.0e9,
        cloud_cpu_capacity_cycles_per_s=HISTORICAL_CLOUD_CPU_GHZ * 1.0e9,
    )

    reference_order_runs = {
        order: compute_reference_policy_metrics(
            slices,
            calibration_keys,
            cfg=cfg,
            cache_cfg=cache_cfg,
            catalog=catalog,
            cloud_cpu_capacity=float(capacity),
            task_order=order,
        )
        for order, capacity in (
            ("csv_row_order", historical_cfg.cloud_cpu_capacity_cycles_per_s),
            ("task_id_ascending", historical_cfg.cloud_cpu_capacity_cycles_per_s),
        )
    }
    # The prior report did not record a task iteration order. Use CSV row order,
    # the natural source-data order, as the primary replay contract. The sorted
    # order is diagnostic only and cannot replace it based on a closer outcome.
    historical_match_order = "csv_row_order"
    historical_run = reference_order_runs[historical_match_order]
    current_20g_order_runs = {
        order: compute_reference_policy_metrics(
            slices,
            calibration_keys,
            cfg=cfg,
            cache_cfg=cache_cfg,
            catalog=catalog,
            cloud_cpu_capacity=current_cloud_capacity,
            task_order=order,
        )
        for order in ("csv_row_order", "task_id_ascending")
    }
    current_20g_run = current_20g_order_runs[historical_match_order]
    historical_match = (
        math.isclose(historical_run["delay_stats"]["p95"], P95_FORMAL_T_REF_HINT_S, rel_tol=0.0, abs_tol=2.0e-6)
        and math.isclose(historical_run["energy_stats"]["p95"], P95_FORMAL_E_REF_HINT_J, rel_tol=0.0, abs_tol=2.0e-6)
        and historical_run["task_count"] == 1642
    )
    p95_recalibration = {
        "historical_reference_values_from_prior_record": {
            "t_ref_s": P95_FORMAL_T_REF_HINT_S,
            "e_ref_j": P95_FORMAL_E_REF_HINT_J,
            "sample_tasks": 1642,
            "sample_slots": 120,
            "training_episode_seeds": list(CALIBRATION_SEEDS),
            "historical_cloud_cpu_ghz": HISTORICAL_CLOUD_CPU_GHZ,
            "percentile_method": "NumPy linear interpolation, task-level P95",
        },
        "historical_replay_status": "EXACT_REPLAYED" if historical_match else "NOT_EXACTLY_REPRODUCED",
        "historical_40g_replay": {
            "assignment_order_selected_for_replay": historical_match_order,
            "delay_p95_s": historical_run["delay_stats"]["p95"],
            "energy_p95_j": historical_run["energy_stats"]["p95"],
            "task_count": historical_run["task_count"],
            "mean_input_kb": historical_run["input_kb_stats"]["mean"],
            "mean_input_kib": historical_run["input_kb_stats"]["mean"] * 1000.0 / 1024.0,
            "mean_cpu_cycles": historical_run["cpu_cycles_stats"]["mean"],
            "candidate_order_runs": {
                order: {
                    "delay_p95_s": value["delay_stats"]["p95"],
                    "energy_p95_j": value["energy_stats"]["p95"],
                    "task_count": value["task_count"],
                    "mean_input_kb_decimal": value["input_kb_stats"]["mean"],
                    "mean_input_kib": value["input_kb_stats"]["mean"] * 1000.0 / 1024.0,
                }
                for order, value in reference_order_runs.items()
            },
        },
        "current_20g_recalibration": {
            "cloud_cpu_capacity_cycles_per_s": current_cloud_capacity,
            "assignment_order": historical_match_order,
            "delay_p95_s": current_20g_run["delay_stats"]["p95"],
            "energy_p95_j": current_20g_run["energy_stats"]["p95"],
            "delay_stats": current_20g_run["delay_stats"],
            "energy_stats": current_20g_run["energy_stats"],
            "task_count": current_20g_run["task_count"],
            "mean_input_kb_decimal": current_20g_run["input_kb_stats"]["mean"],
            "mean_input_kib": current_20g_run["input_kb_stats"]["mean"] * 1000.0 / 1024.0,
            "not_written_to_formal_config": True,
        },
        "current_20g_recalibration_by_order": {
            order: {
                "delay_p95_s": value["delay_stats"]["p95"],
                "energy_p95_j": value["energy_stats"]["p95"],
                "task_count": value["task_count"],
            }
            for order, value in current_20g_order_runs.items()
        },
        "historic_local_computation_calibration": {
            "t_ref_s": HISTORICAL_LOCAL_T_REF_S,
            "e_ref_j": HISTORICAL_LOCAL_E_REF_J,
            "training_arrivals": 3343,
            "exact_episode_slot_selection_recorded": False,
        },
    }
    p95_20_t = float(current_20g_run["delay_stats"]["p95"])
    p95_20_e = float(current_20g_run["energy_stats"]["p95"])
    all_scale_tasks = [
        replace(task, task_id=f"sample-{index}")
        for index, task in enumerate(
            task for key in calibration_keys for task in slices[key]["tasks"]
        )
    ]
    scale_statistics = scheme_reference_stats(all_scale_tasks, formal_t_ref, formal_e_ref, p95_20_t, p95_20_e)

    # Detect whether formal source/config contains a vehicle-local CPU, kappa, or true Deadline field.
    formal_local_cpu_keys: list[str] = []
    formal_kappa_keys: list[str] = []
    deadline_keys: list[str] = []
    def inspect_keys(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                lowered = str(key).lower()
                if "deadline" in lowered:
                    deadline_keys.append(path)
                if ("local" in lowered and "cpu" in lowered) or lowered in {"f_local", "local_cpu_hz", "vehicle_local_cpu_hz"}:
                    formal_local_cpu_keys.append(path)
                if "kappa" in lowered and ("local" in path.lower() or "vehicle" in path.lower()):
                    formal_kappa_keys.append(path)
                inspect_keys(child, path)
    inspect_keys({key: value for key, value in cfg.items() if key != "_meta"})
    inspect_keys(catalog)
    task_headers = set()
    first_sample_dir = slices[(CALIBRATION_SEEDS[0], CALIBRATION_SLOTS[0])]["episode_dir"]
    with (first_sample_dir / "tasks.csv").open("r", encoding="utf-8", newline="") as stream:
        task_headers = set(next(csv.reader(stream)))
    deadline_defined = bool(deadline_keys or any("deadline" in header.lower() for header in task_headers))

    scheme_names = list(scale_statistics)
    slices_for_comparison = [slices[(seed, slot)] for seed, slot, _ in MATCHING_SAMPLES]
    task_count_by_requested = [len(item["tasks"]) for item in slices_for_comparison]
    expected_counts = [count for _, _, count in MATCHING_SAMPLES]
    if task_count_by_requested != expected_counts:
        raise ValueError(f"Predeclared Matching sample counts changed: {task_count_by_requested}")

    formal_global_reference = {"GLOBAL_P95_FORMAL_REFS": (formal_t_ref, formal_e_ref)}
    fixed_records_by_scheme: dict[str, list[dict[str, Any]]] = {name: [] for name in scheme_names}
    reoptimized_records_by_scheme: dict[str, list[dict[str, Any]]] = {name: [] for name in scheme_names}
    matching_validation: dict[str, Any] = {
        "shadow_formal_comparisons": [],
        "formal_matching_vs_shadow_global_full_search": [],
        "formal_incremental_vs_shadow_full": [],
    }
    matching_slot_details: list[dict[str, Any]] = []
    max_shadow_difference = 0.0
    total_task_count = 0
    overall_kkt_call_count = 0
    started_matching = time.perf_counter()

    for sample in MATCHING_SAMPLES:
        seed, slot, target_count = sample
        data = slices[(seed, slot)]
        tasks = list(data["tasks"])
        total_task_count += len(tasks)
        nodes = cache_nodes_for_slice(data, cache_cfg)
        cache = initial_cache_for_slice(data, catalog, cache_cfg)
        uav_positions = uav_configuration.initial_uav_positions_from_project(cfg)
        service_positions = service_positions_for_slice(data, nodes, uav_positions)
        formal_result, formal_eval, feasible_targets, formal_profile = run_formal_matching(
            {"cfg": cfg, "cache_cfg": cache_cfg}, data, nodes, cache, service_positions
        )
        baseline_metrics = summarize_evaluation(tasks, formal_result.matching, formal_eval)
        task_ids = {task.task_id for task in tasks}
        per_slot_schemes = make_reference_schemes(tasks, formal_t_ref, formal_e_ref, p95_20_t, p95_20_e)
        base_shadow = shadow_evaluate_matching_with_kkt(
            tasks, formal_result.matching, data["vehicle_positions"], service_positions,
            cfg, nodes, cache_cfg, per_slot_schemes["GLOBAL_P95_FORMAL_REFS"],
            kkt_cache=task_allocation.KKTResourceAllocationCache(),
        )
        base_shadow_check = compare_formal_and_shadow(formal_eval, base_shadow)
        max_shadow_difference = max(max_shadow_difference, float(base_shadow_check["maximum_abs_difference"]))
        if not base_shadow.resource_feasible or base_shadow_check["maximum_abs_difference"] > MATCHING_TOLERANCE:
            raise RuntimeError(
                f"Shadow global-reference KKT failed to match formal KKT at {seed}/{slot}: {base_shadow_check}"
            )
        matching_validation["shadow_formal_comparisons"].append(
            {"episode_seed": seed, "slot": slot, **base_shadow_check}
        )
        slot_record: dict[str, Any] = {
            "episode_seed": seed,
            "slot": slot,
            "target_task_count": target_count,
            "actual_task_count": len(tasks),
            "time_s": data["time_s"],
            "episode_manifest_sha256": data["manifest_sha256"],
            "baseline_formal_matching": dict(formal_result.matching),
            "baseline_formal_metrics": baseline_metrics,
            "formal_optimizer": formal_profile,
            "schemes": {},
        }

        for scheme_name in scheme_names:
            refs = per_slot_schemes[scheme_name]
            if scheme_name == "GLOBAL_P95_FORMAL_REFS":
                fixed_eval = formal_eval
            else:
                fixed_eval = shadow_evaluate_matching_with_kkt(
                    tasks, formal_result.matching, data["vehicle_positions"], service_positions,
                    cfg, nodes, cache_cfg, refs,
                    kkt_cache=task_allocation.KKTResourceAllocationCache(),
                )
            fixed_resource = validate_shadow_resources(fixed_eval, formal_result.matching, cfg, cache_cfg)
            if not fixed_resource.get("feasible"):
                raise RuntimeError(f"Fixed-matching KKT infeasible for {scheme_name} at {seed}/{slot}: {fixed_resource}")
            fixed_metrics = summarize_evaluation(tasks, formal_result.matching, fixed_eval)
            fixed_delta = compare_allocation_metrics(baseline_metrics, fixed_metrics, task_ids)
            fixed_record = {
                "episode_seed": seed,
                "slot": slot,
                "metrics": fixed_metrics,
                "resource_check": fixed_resource,
                "allocation_change_vs_global_matching": fixed_delta,
                "matching_target_changes_vs_global": 0,
                "matching": dict(formal_result.matching),
            }
            fixed_records_by_scheme[scheme_name].append(fixed_record)

            optimized_result, optimized_eval, optimizer_profile = run_shadow_matching(
                {"cfg": cfg, "cache_cfg": cache_cfg}, data, nodes, cache,
                service_positions, refs,
            )
            if scheme_name == "GLOBAL_P95_FORMAL_REFS":
                changed_ids = sorted(
                    task_id for task_id in task_ids
                    if optimized_result.matching[task_id] != formal_result.matching[task_id]
                )
                matching_validation["formal_incremental_vs_shadow_full"].append({
                    "episode_seed": seed,
                    "slot": slot,
                    "matching_identical": not changed_ids,
                    "changed_task_ids": changed_ids,
                    "formal_incremental_q": float(formal_result.task_utility_sum),
                    "shadow_full_q": float(optimized_result.task_utility_sum),
                    "q_difference": float(optimized_result.task_utility_sum - formal_result.task_utility_sum),
                })
            else:
                # All reference schemes must be optimized over the same physical feasible-target set.
                scheme_feasible_targets = task_allocation.build_feasible_targets(
                    tasks,
                    cache,
                    nodes,
                    vehicle_positions=data["vehicle_positions"],
                    service_node_positions=service_positions,
                    cache_config=cache_cfg,
                    cloud_fallback_mode=task_allocation.RESOURCE_AWARE_FALLBACK,
                )
                for task_id, target_list in feasible_targets.items():
                    if target_list != scheme_feasible_targets[task_id]:
                        raise RuntimeError(f"Feasible candidate set changed for {seed}/{slot}/{scheme_name}")
            if set(optimized_result.matching) != task_ids:
                raise RuntimeError(f"Optimized Matching misses tasks for {scheme_name} at {seed}/{slot}")
            for task_id, target in optimized_result.matching.items():
                if target not in feasible_targets[task_id]:
                    raise RuntimeError(f"Optimized Matching selected infeasible target for {scheme_name}: {task_id}->{target}")
            optimized_resource = validate_shadow_resources(optimized_eval, optimized_result.matching, cfg, cache_cfg)
            if not optimized_resource.get("feasible"):
                raise RuntimeError(f"Optimized KKT infeasible for {scheme_name} at {seed}/{slot}: {optimized_resource}")
            optimized_metrics = summarize_evaluation(tasks, optimized_result.matching, optimized_eval)
            target_change_count = sum(
                optimized_result.matching[task_id] != formal_result.matching[task_id]
                for task_id in task_ids
            )
            optimized_delta = compare_allocation_metrics(baseline_metrics, optimized_metrics, task_ids)
            optimized_record = {
                "episode_seed": seed,
                "slot": slot,
                "metrics": optimized_metrics,
                "resource_check": optimized_resource,
                "allocation_change_vs_global_matching": optimized_delta,
                "matching_target_changes_vs_global": int(target_change_count),
                "matching": dict(optimized_result.matching),
                "matching_runtime_s": optimizer_profile["matching_runtime_s"],
                "matching_kkt_solver_calls": optimizer_profile["matching_kkt_solver_calls"],
                "candidate_evaluations": optimizer_profile["candidate_evaluations"],
                "matching_updates": optimizer_profile["matching_updates"],
                "evaluation_mode": optimizer_profile.get("evaluation_mode", "formal_incremental"),
            }
            reoptimized_records_by_scheme[scheme_name].append(optimized_record)
            slot_record["schemes"][scheme_name] = {
                "fixed_matching": fixed_record,
                "reoptimized_matching": optimized_record,
            }
        matching_slot_details.append(slot_record)
        overall_kkt_call_count += formal_profile["matching_kkt_solver_calls"]

    fixed_aggregates = {
        name: aggregate_case_metrics(records)
        for name, records in fixed_records_by_scheme.items()
    }
    reoptimized_aggregates = {
        name: aggregate_case_metrics(records)
        for name, records in reoptimized_records_by_scheme.items()
    }
    total_runtime = time.perf_counter() - overall_started

    input_sample = current_20g_run["input_kb_stats"]
    cpu_sample = current_20g_run["cpu_cycles_stats"]
    top_level_report = {
        "schema_version": "training_reference_normalization_comparison_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_executable": sys.executable,
            "python_prefix": sys.prefix,
            "python_version": sys.version,
            "platform": platform.platform(),
            "conda_environment": "adaptive_cache_mec" if "adaptive_cache_mec" in sys.prefix.lower() else "UNVERIFIED",
        },
        "provenance": {
            "project_root": str(PROJECT_ROOT),
            "git_head": project_config.get_git_commit(PROJECT_ROOT) if hasattr(project_config, "get_git_commit") else None,
            "config_path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
            "config_sha256": sha256_file(CONFIG_PATH),
            "training_pool_path": str(POOL_ROOT.relative_to(PROJECT_ROOT)),
            "training_pool_manifest_sha256": sha256_file(POOL_MANIFEST_PATH),
            "service_catalog_path": str(catalog_path.relative_to(PROJECT_ROOT)),
            "service_catalog_sha256": sha256_file(catalog_path),
            "task_allocation_sha256": sha256_file(PROJECT_ROOT / "src" / "decision" / "task_allocation.py"),
            "system_model_sha256": sha256_file(PROJECT_ROOT / "src" / "common" / "system_model.py"),
            "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
            "episodes_verified": {str(seed): details for seed, details in episode_provenance.items()},
            "validation_test_read": False,
            "formal_data_modified": False,
        },
        "current_formal_contract": {
            "t_ref_s": formal_t_ref,
            "e_ref_j": formal_e_ref,
            "omega_delay": cache_cfg.omega_t,
            "omega_energy": cache_cfg.omega_e,
            "cloud_cpu_capacity_ghz": cache_cfg.cloud_cpu_capacity_cycles_per_s / 1.0e9,
            "cloud_resource_mode": cache_cfg.cloud_resource_mode,
            "cloud_uplink_rate_mbps": cache_cfg.cloud_total_bandwidth_bps / 1.0e6,
            "cloud_delay_factor": cache_cfg.cloud_delay_factor,
            "cache_initial_seed": cache_cfg.initial_cache_seed,
        },
        "training_reference_sample": {
            "episode_seeds": list(CALIBRATION_SEEDS),
            "slots_per_episode": list(CALIBRATION_SLOTS),
            "episode_slot_count": len(calibration_keys),
            "task_count": current_20g_run["task_count"],
            "mean_input_kb": input_sample["mean"],
            "mean_cpu_cycles": cpu_sample["mean"],
            "input_size_unit_conversion": "CSV input_size bits / 8 / 1000 = decimal KB",
            "cpu_work_unit": "cycles",
            "sample_selection_rule": "Reuse the prior predeclared 12 Training seeds x slots [0,33,66,99,132,165,198,231,264,299]; no data-driven episode replacement.",
        },
        "p95_recalibration": p95_recalibration,
        "reference_scale_statistics": scale_statistics,
        "reference_scheme_contract": {
            "GLOBAL_P95_FORMAL_REFS": "Current configured T_ref/E_ref; values reported as previously derived under the 40 GHz reference allocation.",
            "GLOBAL_P95_RECALIBRATED_20G": "Task-level P95 under current 20 GHz Cloud using the primary CSV row-order deterministic assignment; ascending task-ID sensitivity is separately recorded.",
            "LOCAL_TIME_ONLY": "T_ref=cpu_cycles/F_local_assumed; E_ref=current formal global E_ref.",
            "LOCAL_ENERGY_ONLY": "T_ref=current formal global T_ref; E_ref=kappa_assumed*F_local_assumed^2*cpu_cycles.",
            "LOCAL_COMBINED": "Both per-task local-compute denominators; not formal hardware measurements.",
            "RELATIVE_DEADLINE_ALPHA_*": "T_ref=alpha*cpu_cycles/F_local_assumed, alpha in {0.5,1,2}; E_ref=current formal global E_ref; hypothetical denominator sensitivity, not business deadline or hard constraint.",
            "historical_order_status": "The prior P95 record omits task iteration order. CSV row order is the non-fitted primary replay; sorted numeric task ID is reported only as a diagnostic alternative.",
        },
        "parameter_provenance": {
            "local_cpu_defined_in_formal_system": bool(formal_local_cpu_keys),
            "local_cpu_config_keys": formal_local_cpu_keys,
            "local_energy_coefficient_defined": bool(formal_kappa_keys),
            "local_energy_coefficient_config_keys": formal_kappa_keys,
            "local_parameter_scenario": {
                "f_local_cycles_per_s": LOCAL_CPU_ASSUMED_CYCLES_PER_S,
                "kappa": LOCAL_KAPPA_ASSUMED,
                "source_status": "literature-parameter scenario referenced by the prior local report; not measured or configured in this system",
            },
            "business_deadline_defined": deadline_defined,
            "deadline_related_keys_found": deadline_keys,
            "task_csv_headers": sorted(task_headers),
            "relative_deadline_scenarios_are_hard_constraints": False,
        },
        "matching_samples": [
            {"episode_seed": seed, "slot": slot, "arrived_task_count": count}
            for seed, slot, count in MATCHING_SAMPLES
        ],
        "matching_total_task_count": total_task_count,
        "matching_validation": {
            "max_shadow_formal_abs_difference": max_shadow_difference,
            "shadow_matches_formal": max_shadow_difference <= MATCHING_TOLERANCE,
            "tolerance": MATCHING_TOLERANCE,
            "formal_kkt_comparison_count": len(matching_validation["shadow_formal_comparisons"]),
            "formal_kkt_comparisons": matching_validation["shadow_formal_comparisons"],
            "formal_incremental_vs_shadow_full": matching_validation["formal_incremental_vs_shadow_full"],
            "full_shadow_search_matches_formal_incremental": all(
                item["matching_identical"]
                for item in matching_validation["formal_incremental_vs_shadow_full"]
            ),
            "full_matching_comparison": "COMPLETED_ALL_SCHEMES" if all(len(value) == len(MATCHING_SAMPLES) for value in reoptimized_records_by_scheme.values()) else "INCOMPLETE",
            "candidate_sets_identical": True,
            "matching_search_mode_for_shadow_schemes": "official Transfer-Exchange generator and stopping rule with full shadow KKT candidate evaluation; no global-reference upper-bound pruning",
        },
        "fixed_matching_results": fixed_records_by_scheme,
        "fixed_matching_aggregates": fixed_aggregates,
        "reoptimized_matching_results": reoptimized_records_by_scheme,
        "reoptimized_matching_aggregates": reoptimized_aggregates,
        "matching_slot_details": matching_slot_details,
        "runtime": {
            "total_wall_time_s": total_runtime,
            "matching_comparison_wall_time_s": time.perf_counter() - started_matching,
            "formal_matching_kkt_solver_calls": overall_kkt_call_count,
            "shadow_full_matching_kkt_solver_calls": {
                name: aggregate["total_matching_kkt_solver_calls"]
                for name, aggregate in reoptimized_aggregates.items()
            },
        },
        "recommendation": {
            "normalization_recommendation": "KEEP_CURRENT_GLOBAL_P95_FORMAL; RESOLVE_REFERENCE_POLICY_ORDER_BEFORE_ANY_RECALIBRATION",
            "basis": "A single global scale is specified and avoids task-specific weights based on unmeasured local hardware or hypothetical business deadlines. The 20 GHz P95 is order-sensitive in this fixed reference policy; the task-ID-sorted replay matches the stored refs, while CSV order does not. Because the historical task iteration order is absent from the record, no unique replacement is justified. Keep current formal values pending provenance clarification; do not write diagnostic values into config.",
        },
        "change_flags": {
            "formal_config_modified": False,
            "formal_algorithm_modified": False,
            "formal_data_modified": False,
            "mappo_training_executed": False,
            "validation_test_read": False,
            "github_modified": False,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(jsonable(top_level_report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render_markdown(top_level_report), encoding="utf-8")
    print(f"REPORT={REPORT_PATH}")
    print(f"RESULT_JSON={RESULT_PATH}")
    print(f"TRAIN_REFERENCE_TASKS={current_20g_run['task_count']}")
    print(f"P95_HISTORICAL_REPLAY={historical_run['delay_stats']['p95']:.12g},{historical_run['energy_stats']['p95']:.12g}")
    print(f"P95_CURRENT_20G={p95_20_t:.12g},{p95_20_e:.12g}")
    print(f"SHADOW_FORMAL_MAX_DIFF={max_shadow_difference:.12g}")
    print(f"MATCHING_SLOTS={len(MATCHING_SAMPLES)} TASKS={total_task_count}")
    print(f"FULL_MATCHING_SCHEMES={len(scheme_names)}")
    print(f"TOTAL_RUNTIME_S={total_runtime:.3f}")
    print("VALIDATION_TEST_READ=NO PPO_TRAINING=NO FORMAL_CONFIG_MODIFIED=NO FORMAL_ALGORITHM_MODIFIED=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
