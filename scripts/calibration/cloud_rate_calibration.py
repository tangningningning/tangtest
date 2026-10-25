#!/usr/bin/env python
"""Prepare and run gated, paired Cloud-rate / edge-CPU calibration Episodes.

The default invocation prints help. ``--preflight`` validates the frozen
Training Episode artifacts and runtime override contract without constructing an
environment. ``--execute`` is deliberately explicit and requires a pinned source
identity plus confirmation that the parallel DQN repair is complete.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - supported for older envs
    import tomli as tomllib  # type: ignore[no-redef]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_RELATIVE = Path("configs/dqn/online_joint_dqn.toml")
TRAIN_POOL_RELATIVE = Path("environment_data/proposed_joint_training")
GENERATOR_NAME = "generate_formal_vehicle_environment.py"
SCRIPT_SCHEMA_VERSION = "cloud_rate_edge_cpu_calibration_v1"
EDGE_ROLES = ("sv", "rsu", "uav")
SOURCE_ROOTS = (
    Path("src/common"),
    Path("src/decision"),
    Path("src/online_joint_dqn"),
)
SOURCE_FILES = (
    Path("scripts/training/training_pool_contract.py"),
    Path("scripts/generation/generate_episode_realization.py"),
    Path("scripts/calibration/cloud_rate_calibration.py"),
)
CORE_REPAIR_PATHS = (
    "src/online_joint_dqn/agent.py",
    "src/online_joint_dqn/candidate_features.py",
    "src/online_joint_dqn/environment_adapter.py",
    "src/online_joint_dqn/replay_buffer.py",
    "src/online_joint_dqn/run_training.py",
    "scripts/training/run_staging_smoke.py",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Paired KEEP+STAY calibration. Stage 1 holds Cloud R_c/F_c at baseline "
            "while changing edge CPU multipliers. Stage 2 adds explicitly requested "
            "Cloud uplink rates for one selected CPU plan."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="verify gates and print the source identity; do not run an Episode")
    mode.add_argument("--execute", action="store_true", help="run paired KEEP+STAY Episodes after all gates pass")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--split", choices=("train",), default="train", help="calibration is restricted to the Training pool")
    parser.add_argument("--episode-seed", type=int, action="append", help="completed Training Episode seed; repeat for paired scenes")
    parser.add_argument("--output-root", type=Path, help="new, non-existing output directory; required with --execute")
    parser.add_argument("--expected-source-identity", help="SHA-256 printed by preflight after the source version is fixed")
    parser.add_argument("--core-repair-complete", action="store_true", help="confirm the parallel Online Joint DQN core repair is complete")
    parser.add_argument("--cpu-multiplier-sv", type=float, default=1.0)
    parser.add_argument("--cpu-multiplier-rsu", type=float, default=1.0)
    parser.add_argument("--cpu-multiplier-uav", type=float, default=1.0)
    parser.add_argument(
        "--cloud-rate-mbps",
        type=float,
        action="append",
        help="optional Cloud R_c in decimal Mbps; current baseline is always included",
    )
    return parser


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            _json_safe(payload),
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    _json_safe(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            stream.write("\n")


def _source_identity(project_root: Path) -> dict[str, Any]:
    paths: set[Path] = set()
    for relative_root in SOURCE_ROOTS:
        folder = project_root / relative_root
        if folder.is_dir():
            paths.update(
                path
                for path in folder.rglob("*.py")
                if "__pycache__" not in path.parts
            )
    paths.update(project_root / relative for relative in SOURCE_FILES)
    paths.add(project_root / CONFIG_RELATIVE)
    files = {
        path.relative_to(project_root).as_posix(): _sha256_file(path)
        for path in sorted(paths)
        if path.is_file()
    }
    missing = [
        relative.as_posix()
        for relative in (*SOURCE_FILES, CONFIG_RELATIVE)
        if not (project_root / relative).is_file()
    ]
    return {
        "schema_version": "calibration_source_identity_v1",
        "files": files,
        "missing_required_files": missing,
        "sha256": _canonical_sha256(files),
    }


def _source_contract_checks(project_root: Path) -> list[dict[str, Any]]:
    required: dict[str, tuple[str, ...]] = {
        "src/decision/cache_configuration.py": (
            '"nodes.sv.cpu_cycles_per_s"',
            '"nodes.rsu.cpu_cycles_per_s"',
            '"nodes.uav.cpu_cycles_per_s"',
            '"nodes.cloud.uplink_rate_bps"',
        ),
        "src/decision/task_allocation.py": (
            "return cache_config.sv_cpu_cycles_per_s",
            "return cache_config.rsu_cpu_cycles_per_s",
            "return cache_config.uav_cpu_cycles_per_s",
            "cloud_rate = float(cache_config.cloud_uplink_rate_bps)",
            "cpu_fraction * _node_cpu(role, cache_config)",
        ),
        "src/decision/slot_environment.py": (
            "rate_bps = float(cache_config.cloud_uplink_rate_bps)",
            "actual_upload = min(float(task.input_bits), rate_bps * dt_s)",
            "cache_config.tx_power_w * actual_upload / rate_bps",
        ),
        "src/online_joint_dqn/candidate_evaluator.py": (
            "cache_config = copy.deepcopy(model.cache_config)",
            "                cache_config,\n                flight_boundary,",
        ),
        "src/online_joint_dqn/environment_adapter.py": (
            "            self.model.cache_config,\n            flight_boundary,",
            '"actual_execution_nodes": actual_execution_nodes',
            "steps = aligned_execution_steps(context)",
            "decision_count=decision_count",
            "terminal_at_end=terminal_at_end",
        ),
        "scripts/training/training_pool_contract.py": (
            '"task_data_model": "500_1000_KB"',
            "def _validate_episode_snapshot(",
        ),
    }
    checks: list[dict[str, Any]] = []
    for relative, tokens in required.items():
        path = project_root / relative
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        missing_tokens = [token for token in tokens if token not in text]
        checks.append(
            {
                "source": relative,
                "passed": not missing_tokens,
                "missing_contract_markers": missing_tokens,
            }
        )
    return checks


def _active_formal_generation_processes() -> list[dict[str, Any]]:
    """Read process command lines and report an active formal generator, if any."""

    if os.name == "nt":
        command = (
            "$rows = Get-CimInstance Win32_Process | Where-Object { "
            "$_.CommandLine -like '*generate_formal_vehicle_environment.py*' "
            "-and $_.CommandLine -notlike '*Get-CimInstance*' "
            "-and $_.CommandLine -notlike '*cloud_rate_calibration.py*' "
            "} | Select-Object ProcessId,Name,CommandLine; "
            "$rows | ConvertTo-Json -Compress"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return [{"detection_error": f"{type(exc).__name__}: {exc}"}]
        output = result.stdout.strip()
        if not output:
            return []
        try:
            rows = json.loads(output)
        except json.JSONDecodeError as exc:
            return [{"detection_error": f"invalid process query JSON: {exc}"}]
        return rows if isinstance(rows, list) else [rows]

    matches: list[dict[str, Any]] = []
    proc_root = Path("/proc")
    if proc_root.is_dir():
        for command_path in proc_root.glob("[0-9]*/cmdline"):
            try:
                command_line = command_path.read_bytes().replace(b"\x00", b" ").decode(
                    "utf-8", errors="replace"
                ).strip()
            except OSError:
                continue
            if GENERATOR_NAME in command_line and "cloud_rate_calibration.py" not in command_line:
                matches.append(
                    {
                        "process_id": command_path.parent.name,
                        "command_line": command_line,
                    }
                )
    return matches


def _system_resource_snapshot() -> dict[str, Any]:
    """Read CPU and physical-memory state without changing process priorities."""

    if os.name == "nt":
        command = (
            "$osInfo = Get-CimInstance Win32_OperatingSystem; "
            "$cpuInfo = Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average; "
            "[pscustomobject]@{" 
            "free_physical_memory_kb=[double]$osInfo.FreePhysicalMemory; "
            "total_physical_memory_kb=[double]$osInfo.TotalVisibleMemorySize; "
            "average_cpu_load_percent=[double]$cpuInfo.Average "
            "} | ConvertTo-Json -Compress"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
            payload = json.loads(result.stdout.strip())
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            return {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}
        free_kb = float(payload["free_physical_memory_kb"])
        total_kb = float(payload["total_physical_memory_kb"])
        return {
            "status": "AVAILABLE",
            "free_physical_memory_mb": free_kb / 1024.0,
            "total_physical_memory_mb": total_kb / 1024.0,
            "average_cpu_load_percent": float(payload["average_cpu_load_percent"]),
            "logical_cpu_count": os.cpu_count(),
            "policy": "observation only; calibration remains blocked while generator is active",
        }

    memory_info: dict[str, float] = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            name, _, raw_value = line.partition(":")
            if name in {"MemAvailable", "MemTotal"}:
                memory_info[name] = float(raw_value.strip().split()[0])
    load_average = None
    try:
        load_average = list(os.getloadavg())
    except (AttributeError, OSError):
        pass
    if not memory_info and load_average is None:
        return {"status": "UNAVAILABLE", "policy": "calibration remains blocked while generator is active"}
    return {
        "status": "AVAILABLE",
        "available_memory_mb": memory_info.get("MemAvailable", 0.0) / 1024.0,
        "total_memory_mb": memory_info.get("MemTotal", 0.0) / 1024.0,
        "load_average_1m_5m_15m": load_average,
        "logical_cpu_count": os.cpu_count(),
        "policy": "observation only; calibration remains blocked while generator is active",
    }


def _import_project_modules(project_root: Path) -> dict[str, Any]:
    source_root = str(project_root / "src")
    training_root = str(project_root / "scripts" / "training")
    for path in (source_root, training_root):
        if path not in sys.path:
            sys.path.insert(0, path)
    from common import project_config
    from common.environment_interface import aligned_execution_steps
    from decision import task_allocation
    from online_joint_dqn.candidate_evaluator import SlotInput, SystemModelContext
    from online_joint_dqn.environment_adapter import ExclusiveDQNEnvironment
    import training_pool_contract

    return {
        "project_config": project_config,
        "aligned_execution_steps": aligned_execution_steps,
        "task_allocation": task_allocation,
        "SlotInput": SlotInput,
        "SystemModelContext": SystemModelContext,
        "ExclusiveDQNEnvironment": ExclusiveDQNEnvironment,
        "pool_contract": training_pool_contract,
    }


def _get_path(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"Missing config field: {dotted_key}")
        current = current[part]
    return current


def _set_path(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = config
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise KeyError(f"Missing config table for {dotted_key}")
        current = child
    if parts[-1] not in current:
        raise KeyError(f"Missing config field: {dotted_key}")
    current[parts[-1]] = value


def _config_without_paths(config: dict[str, Any], dotted_keys: Sequence[str]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for dotted_key in dotted_keys:
        parts = dotted_key.split(".")
        current = result
        for part in parts[:-1]:
            current = current[part]
        del current[parts[-1]]
    return result


def _load_base_config(project_root: Path) -> tuple[dict[str, Any], Path, dict[str, str]]:
    config_path = (project_root / CONFIG_RELATIVE).resolve(strict=True)
    raw_bytes = config_path.read_bytes()
    base_config = tomllib.loads(raw_bytes.decode("utf-8"))
    cloud_paths = (
        "nodes.cloud.uplink_rate_bps",
        "nodes.cloud.total_bandwidth_bps",
        "communication.cloud.uplink_rate_bps",
    )
    cloud_rates = {key: float(_get_path(base_config, key)) for key in cloud_paths}
    if len(set(cloud_rates.values())) != 1:
        raise ValueError(f"Cloud uplink aliases are inconsistent: {cloud_rates}")
    for role in EDGE_ROLES:
        cpu = float(_get_path(base_config, f"nodes.{role}.cpu_cycles_per_s"))
        bandwidth_hz = float(_get_path(base_config, f"nodes.{role}.bandwidth_hz"))
        if not math.isfinite(cpu) or cpu <= 0.0:
            raise ValueError(f"Invalid {role.upper()} CPU capacity: {cpu}")
        if not math.isfinite(bandwidth_hz) or bandwidth_hz <= 0.0:
            raise ValueError(f"Invalid {role.upper()} bandwidth: {bandwidth_hz}")
    input_min_kb = float(_get_path(base_config, "service.generation.input_size_min_kb"))
    input_max_kb = float(_get_path(base_config, "service.generation.input_size_max_kb"))
    if input_min_kb != 500.0 or input_max_kb != 1000.0:
        raise ValueError(
            "Calibration expects the existing 500-1000 KB task-data contract; "
            f"found {input_min_kb}-{input_max_kb} KB"
        )
    raw_hash = _sha256_bytes(raw_bytes)
    semantic_hash = _canonical_sha256(base_config)
    return base_config, config_path, {
        "raw_sha256": raw_hash,
        "semantic_sha256": semantic_hash,
    }


def _build_runtime_config(
    base_config: dict[str, Any],
    *,
    cpu_multipliers: Mapping[str, float],
    cloud_rate_bps: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_config = copy.deepcopy(base_config)
    overrides: dict[str, Any] = {}
    changed_paths: list[str] = []
    for role in EDGE_ROLES:
        multiplier = float(cpu_multipliers[role])
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError(f"{role.upper()} CPU multiplier must be positive and finite")
        path = f"nodes.{role}.cpu_cycles_per_s"
        before = float(_get_path(base_config, path))
        after = before * multiplier
        if not math.isfinite(after) or after <= 0.0:
            raise ValueError(f"Runtime {role.upper()} CPU capacity is invalid: {after}")
        _set_path(runtime_config, path, after)
        overrides[path] = {
            "baseline_cycles_per_s": before,
            "multiplier": multiplier,
            "runtime_cycles_per_s": after,
        }
        changed_paths.append(path)

    cloud_paths = (
        "nodes.cloud.uplink_rate_bps",
        "nodes.cloud.total_bandwidth_bps",
        "communication.cloud.uplink_rate_bps",
    )
    if not math.isfinite(cloud_rate_bps) or cloud_rate_bps <= 0.0:
        raise ValueError("Cloud uplink rate must be positive and finite")
    cloud_before = float(_get_path(base_config, cloud_paths[0]))
    for path in cloud_paths:
        _set_path(runtime_config, path, float(cloud_rate_bps))
        overrides[path] = {
            "baseline_bps": float(_get_path(base_config, path)),
            "runtime_bps": float(cloud_rate_bps),
            "meaning": "per-task fixed Cloud Rc; compatibility aliases are equality-checked",
        }
        changed_paths.append(path)
    allowed_paths = tuple(changed_paths)
    if _config_without_paths(runtime_config, allowed_paths) != _config_without_paths(
        base_config, allowed_paths
    ):
        raise RuntimeError("Runtime overlay changed a configuration field outside its contract")
    if any(float(_get_path(runtime_config, path)) != cloud_rate_bps for path in cloud_paths):
        raise RuntimeError("Runtime Cloud Rc aliases do not all match")
    return runtime_config, {
        "cloud_rate_bps": float(cloud_rate_bps),
        "cloud_rate_mbps": float(cloud_rate_bps / 1.0e6),
        "cloud_cpu_Fc_cycles_per_s": float(
            _get_path(runtime_config, "nodes.cloud.cpu_cycles_per_s")
        ),
        "cloud_rate_changed": not math.isclose(
            cloud_rate_bps, cloud_before, rel_tol=0.0, abs_tol=0.0
        ),
        "cpu_multipliers": dict(cpu_multipliers),
        "overrides": overrides,
        "changed_paths": list(allowed_paths),
        "runtime_semantic_sha256": _canonical_sha256(runtime_config),
    }


def _requested_rates(base_rate_bps: float, requested_mbps: Sequence[float] | None) -> list[float]:
    rates = [float(base_rate_bps)]
    for rate_mbps in requested_mbps or ():
        rate_mbps_f = float(rate_mbps)
        if not math.isfinite(rate_mbps_f) or rate_mbps_f <= 0.0:
            raise ValueError(f"Cloud rate must be positive and finite: {rate_mbps}")
        rate_bps = rate_mbps_f * 1.0e6
        if rate_bps not in rates:
            rates.append(rate_bps)
    return rates


def _validate_pool_and_scenes(
    project_root: Path,
    modules: Mapping[str, Any],
    base_config: dict[str, Any],
    episode_seeds: Sequence[int],
) -> dict[str, Any]:
    if not episode_seeds:
        raise ValueError("Specify at least one completed Training Episode with --episode-seed")
    if len(set(episode_seeds)) != len(episode_seeds):
        raise ValueError("Episode seeds must be unique within a paired calibration")
    config_path = project_root / CONFIG_RELATIVE
    project_config = modules["project_config"]
    runtime_config = copy.deepcopy(base_config)
    runtime_config["_meta"] = {
        "config_path": str(config_path.resolve()),
        "project_root": str(project_root),
    }
    project_config.validate_for_formal_experiment(runtime_config)
    pool_contract = modules["pool_contract"]
    pool, pool_info = pool_contract._validate_pool(
        project_root,
        TRAIN_POOL_RELATIVE,
        expected_split="train",
        expected_seed_start=pool_contract.DEFAULT_TRAINING_SEED_START,
        expected_episode_count=pool_contract.FORMAL_TRAINING_EPISODE_COUNT,
        runtime_config=base_config,
    )
    active_seeds = set(int(seed) for seed in pool_info["active_episode_seeds"])
    absent = [int(seed) for seed in episode_seeds if int(seed) not in active_seeds]
    if absent:
        raise ValueError(f"Episode seed(s) are not active in the Training pool: {absent}")
    scene_records: list[dict[str, Any]] = []
    for seed in episode_seeds:
        pool_contract._validate_episode_snapshot(pool, int(seed), pool_info)
        episode_dir = pool / f"episode_{int(seed)}"
        manifest_path = episode_dir / "episode_manifest.json"
        episode_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact_hashes = episode_manifest.get(
            "artifact_sha256", episode_manifest.get("generated_artifact_sha256", {})
        )
        scene_records.append(
            {
                "split": "train",
                "episode_seed": int(seed),
                "scene_id": f"train:{int(seed)}",
                "scene_sha256": pool_info["active_episode_scene_sha256"][str(seed)],
                "episode_manifest_sha256": _sha256_file(manifest_path),
                "artifact_sha256": dict(artifact_hashes),
                "task_data_model": pool_info["task_data_model"],
                "task_data_contract_sha256": pool_info["task_data_contract_sha256"],
            }
        )
    return {
        "pool_path": str(pool),
        "pool_manifest_path": pool_info["manifest_path"],
        "pool_manifest_sha256": pool_info["manifest_sha256"],
        "pool_dataset_identity_sha256": pool_info["dataset_identity_sha256"],
        "pool_task_data_contract_sha256": pool_info["task_data_contract_sha256"],
        "pool_task_data_model": pool_info["task_data_model"],
        "pool_config_compatibility": pool_info["config_compatibility"],
        "execution_horizon_slots": int(pool_info["execution_horizon_slots"]),
        "scenes": scene_records,
        "_pool": pool,
        "_pool_info": pool_info,
    }


def _preflight(
    project_root: Path,
    episode_seeds: Sequence[int],
    cpu_multipliers: Mapping[str, float],
    requested_rates_mbps: Sequence[float] | None,
) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    source_identity = _source_identity(root)
    source_checks = _source_contract_checks(root)
    blockers: list[str] = []
    if source_identity["missing_required_files"]:
        blockers.append(
            "Required source files are missing: "
            + ", ".join(source_identity["missing_required_files"])
        )
    failed_checks = [check for check in source_checks if not check["passed"]]
    if failed_checks:
        blockers.append(
            "Source path contract markers changed; re-audit Matching and actual execution before calibration"
        )
    generator_processes = _active_formal_generation_processes()
    resource_snapshot = _system_resource_snapshot() if generator_processes else None
    if generator_processes:
        blockers.append(
            "Formal environment generation is active or process detection failed; "
            "do not read/execute Episodes until it finishes and resource state is rechecked"
        )

    base_config: dict[str, Any] | None = None
    config_identity: dict[str, Any] | None = None
    override_previews: list[dict[str, Any]] = []
    scene_identity: dict[str, Any] | None = None
    try:
        base_config, config_path, config_hashes = _load_base_config(root)
        cloud_rate_bps = float(_get_path(base_config, "nodes.cloud.uplink_rate_bps"))
        config_identity = {
            "config_path": str(config_path),
            **config_hashes,
            "edge_cpu_baseline_cycles_per_s": {
                role.upper(): float(_get_path(base_config, f"nodes.{role}.cpu_cycles_per_s"))
                for role in EDGE_ROLES
            },
            "edge_bandwidth_hz": {
                role.upper(): float(_get_path(base_config, f"nodes.{role}.bandwidth_hz"))
                for role in EDGE_ROLES
            },
            "cloud_cpu_per_task_cycles_per_s": float(
                _get_path(base_config, "nodes.cloud.cpu_cycles_per_s")
            ),
            "cloud_uplink_bps": cloud_rate_bps,
            "task_input_base_range_kb": {
                "min": float(_get_path(base_config, "service.generation.input_size_min_kb")),
                "max": float(_get_path(base_config, "service.generation.input_size_max_kb")),
            },
        }
        for rate_bps in _requested_rates(cloud_rate_bps, requested_rates_mbps):
            _runtime_config, preview = _build_runtime_config(
                base_config,
                cpu_multipliers=cpu_multipliers,
                cloud_rate_bps=rate_bps,
            )
            preview["case_id"] = _case_id(rate_bps, cpu_multipliers)
            override_previews.append(preview)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        blockers.append(f"Configuration/overlay preflight failed: {type(exc).__name__}: {exc}")

    if not generator_processes and base_config is not None:
        try:
            modules = _import_project_modules(root)
            scenes = _validate_pool_and_scenes(root, modules, base_config, episode_seeds)
            scene_identity = {
                key: value for key, value in scenes.items() if not key.startswith("_")
            }
            for preview in override_previews:
                runtime_config, _ = _build_runtime_config(
                    base_config,
                    cpu_multipliers=cpu_multipliers,
                    cloud_rate_bps=float(preview["cloud_rate_bps"]),
                )
                runtime_config["_meta"] = {
                    "config_path": str((root / CONFIG_RELATIVE).resolve()),
                    "project_root": str(root),
                }
                modules["project_config"].validate_for_formal_experiment(runtime_config)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            blockers.append(f"Training Episode identity preflight failed: {type(exc).__name__}: {exc}")

    return {
        "status": "BLOCKED" if blockers else "PASS",
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "project_root": str(root),
        "split": "train",
        "strategy": "KEEP_STAY",
        "source_identity": source_identity,
        "source_contract_checks": source_checks,
        "formal_generation_processes": generator_processes,
        "resource_snapshot": resource_snapshot,
        "config_identity": config_identity,
        "runtime_override_previews": override_previews,
        "episode_identity": scene_identity,
        "blockers": blockers,
        "preflight_only": True,
    }


def _case_id(cloud_rate_bps: float, cpu_multipliers: Mapping[str, float]) -> str:
    rate_label = f"{cloud_rate_bps / 1.0e6:g}mbps".replace(".", "p")
    cpu_label = "_".join(
        f"{role}{float(cpu_multipliers[role]):g}x".replace(".", "p")
        for role in EDGE_ROLES
    )
    return f"cloud_{rate_label}_{cpu_label}"


def _json_nonfinite_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_json_nonfinite_paths(item, path))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            paths.extend(_json_nonfinite_paths(item, f"{prefix}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        paths.append(prefix or "<root>")
    elif hasattr(value, "item"):
        try:
            item = value.item()
        except (TypeError, ValueError):
            return paths
        if isinstance(item, float) and not math.isfinite(item):
            paths.append(prefix or "<root>")
    return paths


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return float(ordered[index])


def _latency_summary(values: Sequence[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(numbers),
        "mean_s": float(statistics.fmean(numbers)) if numbers else None,
        "median_s": float(statistics.median(numbers)) if numbers else None,
        "p95_s_nearest_rank": _percentile(numbers, 0.95),
        "max_s": max(numbers) if numbers else None,
    }


def _resource_record(
    diagnostics: Mapping[str, Any],
    *,
    node_cpu_capacity: Mapping[str, float],
    node_bandwidth_capacity: Mapping[str, float],
) -> dict[str, Any]:
    summary = diagnostics.get("matching_summary")
    if not isinstance(summary, Mapping):
        summary = {}
    resource = summary.get("resource")
    if not isinstance(resource, Mapping):
        resource = {}
    cpu_fraction = dict(resource.get("edge_cpu_fraction_sum_by_node", {}))
    bandwidth_fraction = dict(resource.get("edge_bandwidth_fraction_sum_by_node", {}))
    cpu_usage = dict(resource.get("edge_cpu_usage_by_node_cycles_per_s", {}))
    bandwidth_usage = dict(resource.get("edge_bandwidth_usage_hz_by_node", {}))
    violations: list[dict[str, Any]] = []
    for node_id, fraction in cpu_fraction.items():
        if float(fraction) > 1.0 + 1e-8:
            violations.append({"node_id": str(node_id), "resource": "cpu_fraction", "value": float(fraction)})
    for node_id, fraction in bandwidth_fraction.items():
        if float(fraction) > 1.0 + 1e-8:
            violations.append({"node_id": str(node_id), "resource": "bandwidth_fraction", "value": float(fraction)})
    finite_count = summary.get("KKT_FINITE_COUNT")
    nonfinite_result_count = summary.get("cloud_kkt_nonfinite_result_count", 0)
    if summary.get("resource_feasible") is False:
        violations.append({"resource": "matching_resource_feasible", "value": False})
    if isinstance(nonfinite_result_count, (int, float)) and nonfinite_result_count > 0:
        violations.append({"resource": "cloud_kkt_nonfinite_result_count", "value": int(nonfinite_result_count)})
    diagnostics_rows = summary.get("KKT_SOLVE_DIAGNOSTICS", [])
    if not isinstance(diagnostics_rows, list):
        diagnostics_rows = []
    kkt_residuals = [
        float(row["resource_residual"])
        for row in diagnostics_rows
        if isinstance(row, Mapping)
        and isinstance(row.get("resource_residual"), (int, float))
        and math.isfinite(float(row["resource_residual"]))
    ]
    return {
        "matching_resource_feasible": summary.get("resource_feasible"),
        "edge_cpu_usage_cycles_per_s": resource.get("edge_cpu_usage_cycles_per_s"),
        "edge_cpu_usage_by_node_cycles_per_s": cpu_usage,
        "edge_cpu_capacity_by_node_cycles_per_s": dict(node_cpu_capacity),
        "edge_cpu_capacity_utilization_by_node": {
            node_id: float(cpu_usage.get(node_id, 0.0)) / float(capacity)
            for node_id, capacity in node_cpu_capacity.items()
        },
        "edge_cpu_fraction_sum_by_node": cpu_fraction,
        "edge_bandwidth_usage_hz_by_node": bandwidth_usage,
        "edge_bandwidth_capacity_hz_by_node": dict(node_bandwidth_capacity),
        "edge_bandwidth_fraction_sum_by_node": bandwidth_fraction,
        "cloud_cpu_usage_cycles_per_s": resource.get("cloud_cpu_usage_cycles_per_s"),
        "cloud_cpu_capacity_cycles_per_s": resource.get("cloud_cpu_capacity_cycles_per_s"),
        "cloud_uplink_rate_usage_bps": resource.get("cloud_uplink_rate_usage_bps"),
        "cloud_uplink_rate_capacity_bps": resource.get("cloud_uplink_rate_capacity_bps"),
        "kkt_finite_count": finite_count,
        "kkt_resource_residual_max": max(kkt_residuals) if kkt_residuals else None,
        "resource_violations": violations,
    }


def _aggregate_episode(
    *,
    seed: int,
    scene_record: Mapping[str, Any],
    transitions: Sequence[Any],
    task_metadata_by_id: Mapping[str, Mapping[str, Any]],
    node_ids: Sequence[str],
    service_ids: Sequence[int],
    node_roles: Mapping[str, str],
    node_cpu_capacity: Mapping[str, float],
    node_bandwidth_capacity: Mapping[str, float],
    cloud_node_id: str,
) -> dict[str, Any]:
    arrived_task_ids: set[str] = set()
    first_assignment_by_task: dict[str, str] = {}
    terminal_by_task: dict[str, Mapping[str, Any]] = {}
    active_at_end: dict[str, Mapping[str, Any]] = {}
    task_energy_by_id: dict[str, float] = {}
    lifecycle_latencies: list[float] = []
    slot_rows: list[dict[str, Any]] = []
    nonfinite_paths: list[str] = []
    rewards: list[float] = []
    utility_rewards: list[float] = []
    upload_energy_by_slot: list[float] = []
    cache_cost_by_slot: list[float] = []
    cache_download_delay_by_slot: list[float] = []
    movement_cost_by_slot: list[float] = []
    movement_energy_by_slot: list[float] = []
    node_counts: Counter[str] = Counter()
    resource_records: list[dict[str, Any]] = []
    first_assignment_diagnostics: list[dict[str, Any]] = []
    matching_decision_diagnostics: list[dict[str, Any]] = []

    for transition_index, transition in enumerate(transitions):
        diagnostics = transition.diagnostics
        nonfinite_paths.extend(
            f"slot[{transition_index}].{path}"
            for path in _json_nonfinite_paths(diagnostics)
        )
        reward = float(transition.reward)
        utility_reward = float(diagnostics.get("actual_system_utility", float("nan")))
        rewards.append(reward)
        utility_rewards.append(utility_reward)
        arrived_task_ids.update(str(item) for item in diagnostics.get("arrived_task_ids", ()))
        raw_execution_nodes = diagnostics.get("actual_execution_nodes")
        if not diagnostics.get("actual_execution_node_data_available") or not isinstance(
            raw_execution_nodes, list
        ):
            raise RuntimeError(
                f"slot {transition_index}: actual execution node records are unavailable"
            )
        for assignment in raw_execution_nodes:
            task_id = str(assignment["task_id"])
            node_id = str(assignment["execution_node"])
            if task_id not in first_assignment_by_task:
                first_assignment_by_task[task_id] = node_id
                node_counts[node_id] += 1
                task_metadata = task_metadata_by_id.get(task_id)
                if task_metadata is None:
                    raise RuntimeError(f"Task {task_id} assignment has no source task row")
                service_id = int(task_metadata["service_id"])
                try:
                    service_index = list(service_ids).index(service_id)
                except ValueError as exc:
                    raise RuntimeError(f"Task {task_id} has unknown service_id={service_id}") from exc
                cache_rows = diagnostics.get("pre_cache_state")
                if not isinstance(cache_rows, list) or len(cache_rows) != len(node_ids):
                    raise RuntimeError(f"slot {transition_index} has no aligned pre-execution cache state")
                cached_edge_node_ids = [
                    str(edge_node_id)
                    for node_index, edge_node_id in enumerate(node_ids)
                    if str(node_roles.get(str(edge_node_id), "")).upper() in {"SV", "RSU", "UAV"}
                    and bool(cache_rows[node_index][service_index])
                ]
                selected_role = "Cloud" if node_id == cloud_node_id else node_roles.get(node_id, "OTHER")
                selected_service_cached = (
                    node_id in cached_edge_node_ids
                    if selected_role in {"SV", "RSU", "UAV"}
                    else None
                )
                first_assignment_diagnostics.append(
                    {
                        "task_id": task_id,
                        "service_id": service_id,
                        "input_size_kb": float(task_metadata["input_bits"]) / 8000.0,
                        "execution_node": node_id,
                        "execution_role": selected_role,
                        "edge_nodes_caching_service_at_assignment": cached_edge_node_ids,
                        "cached_edge_node_count_at_assignment": len(cached_edge_node_ids),
                        "selected_edge_service_cached": selected_service_cached,
                        "cloud_selected_while_edge_cache_available": (
                            bool(cached_edge_node_ids) if selected_role == "Cloud" else None
                        ),
                    }
                )
        for event in diagnostics.get("task_termination_events", ()):
            task_id = str(event["task_id"])
            if task_id in terminal_by_task:
                raise RuntimeError(f"task {task_id} has duplicate terminal records")
            terminal_by_task[task_id] = event
            latency = event.get("actual_execution_time_s")
            if isinstance(latency, (int, float)) and math.isfinite(float(latency)):
                lifecycle_latencies.append(float(latency))
        per_task_metrics = diagnostics.get("per_task_slot_execution", {})
        if isinstance(per_task_metrics, Mapping):
            for task_id, metrics in per_task_metrics.items():
                if isinstance(metrics, Mapping) and isinstance(
                    metrics.get("cumulative_upload_energy_j"), (int, float)
                ):
                    task_energy_by_id[str(task_id)] = float(
                        metrics["cumulative_upload_energy_j"]
                    )
        upload_energy_by_slot.append(float(diagnostics.get("actual_total_vehicle_energy_j", 0.0)))
        cache_cost_by_slot.append(float(diagnostics.get("cache_cost", 0.0) or 0.0))
        cache_download_delay_by_slot.append(float(diagnostics.get("cache_download_delay_s", 0.0) or 0.0))
        movement_cost_by_slot.append(float(diagnostics.get("movement_cost", 0.0) or 0.0))
        movement_energy_by_slot.append(float(diagnostics.get("uav_movement_energy_j", 0.0) or 0.0))
        resource = _resource_record(
            diagnostics,
            node_cpu_capacity=node_cpu_capacity,
            node_bandwidth_capacity=node_bandwidth_capacity,
        )
        resource_records.append(resource)
        matching_summary = diagnostics.get("matching_summary", {})
        if isinstance(matching_summary, Mapping):
            matching_decision_diagnostics.append(
                {
                    "slot_index": int(diagnostics.get("slot_id", transition_index)),
                    "matching_resource_feasible": matching_summary.get("resource_feasible"),
                    "stop_reason": matching_summary.get("stop_reason"),
                    "transfer_exchange_stable": matching_summary.get("transfer_exchange_stable"),
                    "transfer_updates": matching_summary.get("transfer_updates"),
                    "exchange_updates": matching_summary.get("exchange_updates"),
                    "candidate_definitions_total": matching_summary.get("candidate_definitions_total"),
                    "candidate_evaluations_total": matching_summary.get("candidate_evaluations_total"),
                    "candidate_evaluation_objective": matching_summary.get("candidate_evaluation_objective"),
                    "candidate_prediction_assumption": matching_summary.get("candidate_prediction_assumption"),
                    "matching_candidate_task_utility_sum": matching_summary.get("matching_candidate_task_utility_sum"),
                }
            )
        slot_rows.append(
            {
                "slot_index": int(diagnostics.get("slot_id", transition_index)),
                "reward": reward,
                "actual_system_utility": utility_reward,
                "arrived_task_ids": list(diagnostics.get("arrived_task_ids", ())),
                "actual_execution_nodes": raw_execution_nodes,
                "termination_events": diagnostics.get("task_termination_events", []),
                "actual_upload_energy_j": float(
                    diagnostics.get("actual_total_vehicle_energy_j", 0.0)
                ),
                "episode_cumulative_upload_energy_j": float(
                    diagnostics.get("episode_vehicle_upload_energy_j", 0.0)
                ),
                "cache_cost": diagnostics.get("cache_cost"),
                "cache_download_delay_s": diagnostics.get("cache_download_delay_s"),
                "uav_movement_cost": diagnostics.get("movement_cost"),
                "uav_movement_energy_j": diagnostics.get("uav_movement_energy_j"),
                "resource_usage": resource,
                "matching_decision": matching_decision_diagnostics[-1]
                if matching_decision_diagnostics
                else None,
            }
        )
        active_states = diagnostics.get("active_task_states", ())
        if isinstance(active_states, list):
            for state in active_states:
                if isinstance(state, Mapping):
                    active_at_end[str(state["task_id"])] = state

    assigned_ids = set(first_assignment_by_task)
    unexpected_assignments = sorted(assigned_ids - arrived_task_ids)
    if unexpected_assignments:
        raise RuntimeError(f"Assignments exist for tasks without an arrival record: {unexpected_assignments[:5]}")
    unassigned_ids = sorted(arrived_task_ids - assigned_ids)
    status_counts: Counter[str] = Counter()
    for task_id in arrived_task_ids:
        event = terminal_by_task.get(task_id)
        if event is not None:
            status = str(event.get("termination_reason", event.get("status", "OTHER"))).upper()
            status_counts[status] += 1
        elif task_id in active_at_end:
            status_counts["ACTIVE_AT_EPISODE_END"] += 1
        else:
            status_counts["NO_TERMINAL_RECORD"] += 1
    assigned_total = len(assigned_ids)
    unique_total = len(arrived_task_ids)
    if assigned_total != sum(node_counts.values()):
        raise RuntimeError("First-assignment node totals do not reconcile")
    node_type_counts: Counter[str] = Counter()
    for node_id, count in node_counts.items():
        if node_id == cloud_node_id:
            role = "Cloud"
        else:
            role = node_roles.get(node_id, "OTHER")
        node_type_counts[role] += int(count)
    node_detail = {
        node_id: {
            "role": "Cloud" if node_id == cloud_node_id else node_roles.get(node_id, "OTHER"),
            "task_count": int(node_counts.get(node_id, 0)),
            "ratio_of_assigned": (
                float(node_counts.get(node_id, 0) / assigned_total)
                if assigned_total
                else None
            ),
        }
        for node_id in sorted(set(node_roles) | {cloud_node_id} | set(node_counts))
    }
    terminal_energy_sum = float(math.fsum(task_energy_by_id.values()))
    final_cumulative_energy = float(
        transitions[-1].diagnostics.get("episode_vehicle_upload_energy_j", 0.0)
    ) if transitions else 0.0
    reward_sum = float(math.fsum(rewards))
    utility_sum = float(math.fsum(utility_rewards))
    energy_consistency_pass = math.isclose(
        float(math.fsum(upload_energy_by_slot)), final_cumulative_energy,
        rel_tol=1e-9, abs_tol=1e-9,
    ) and math.isclose(terminal_energy_sum, final_cumulative_energy, rel_tol=1e-9, abs_tol=1e-9)
    reward_consistency_pass = math.isclose(reward_sum, utility_sum, rel_tol=1e-9, abs_tol=1e-9)
    input_sizes_kb = [
        float(task_metadata_by_id[task_id]["input_bits"]) / 8000.0
        for task_id in arrived_task_ids
        if task_id in task_metadata_by_id
    ]
    all_arrivals_have_input = len(input_sizes_kb) == unique_total
    if not all_arrivals_have_input:
        raise RuntimeError("The Episode arrival set could not be joined to its task input sizes")
    success_count = int(status_counts.get("SUCCESS", 0))
    timeout_count = int(status_counts.get("TIMEOUT", 0))
    other_count = unique_total - success_count - timeout_count
    resource_violations = [
        {"slot_index": index, **violation}
        for index, record in enumerate(resource_records)
        for violation in record["resource_violations"]
    ]
    cloud_ratio = (
        float(node_type_counts.get("Cloud", 0) / assigned_total)
        if assigned_total
        else None
    )
    return {
        "split": "train",
        "episode_seed": int(seed),
        "scene_id": scene_record["scene_id"],
        "scene_sha256": scene_record["scene_sha256"],
        "episode_manifest_sha256": scene_record["episode_manifest_sha256"],
        "episode_return": reward_sum,
        "slot_reward_sum": reward_sum,
        "actual_system_utility_sum": utility_sum,
        "reward_sum_matches_slot_utility": reward_consistency_pass,
        "slot_count": len(transitions),
        "unique_arrived_task_count": unique_total,
        "assigned_unique_task_count": assigned_total,
        "unassigned_unique_task_count": len(unassigned_ids),
        "unassigned_task_ids": unassigned_ids,
        "task_assignment_coverage": float(assigned_total / unique_total) if unique_total else None,
        "cloud_offload_ratio_of_assigned": cloud_ratio,
        "keep_stay_cloud_ratio_target_le_0_40": (
            None if cloud_ratio is None else bool(cloud_ratio <= 0.40)
        ),
        "dqn_cloud_ratio_target_0_10_to_0_20": (
            None if cloud_ratio is None else bool(0.10 <= cloud_ratio <= 0.20)
        ),
        "task_success_count": success_count,
        "task_timeout_count": timeout_count,
        "task_other_terminal_or_missing_count": other_count,
        "task_status_counts": dict(sorted(status_counts.items())),
        "success_rate_of_arrivals": float(success_count / unique_total) if unique_total else None,
        "timeout_rate_of_arrivals": float(timeout_count / unique_total) if unique_total else None,
        "lifecycle_latency_s": _latency_summary(lifecycle_latencies),
        "input_size_kb_observed": {
            "count": len(input_sizes_kb),
            "min": min(input_sizes_kb) if input_sizes_kb else None,
            "max": max(input_sizes_kb) if input_sizes_kb else None,
        },
        "node_type_task_count": {
            role: int(node_type_counts.get(role, 0))
            for role in ("SV", "RSU", "UAV", "Cloud", "OTHER")
        },
        "node_task_details": node_detail,
        "first_assignment_diagnostics": first_assignment_diagnostics,
        "matching_decision_diagnostics_by_slot": matching_decision_diagnostics,
        "cache_availability_diagnostics": {
            "cloud_with_cached_edge_option_count": sum(
                item["execution_role"] == "Cloud"
                and bool(item["cloud_selected_while_edge_cache_available"])
                for item in first_assignment_diagnostics
            ),
            "cloud_without_cached_edge_option_count": sum(
                item["execution_role"] == "Cloud"
                and not bool(item["cloud_selected_while_edge_cache_available"])
                for item in first_assignment_diagnostics
            ),
            "edge_assignment_cache_hit_count": sum(
                item["execution_role"] in {"SV", "RSU", "UAV"}
                and bool(item["selected_edge_service_cached"])
                for item in first_assignment_diagnostics
            ),
            "edge_assignment_cache_miss_count": sum(
                item["execution_role"] in {"SV", "RSU", "UAV"}
                and not bool(item["selected_edge_service_cached"])
                for item in first_assignment_diagnostics
            ),
        },
        "cumulative_upload_energy_j": final_cumulative_energy,
        "upload_energy_crosscheck_sum_slot_j": float(math.fsum(upload_energy_by_slot)),
        "upload_energy_crosscheck_sum_unique_task_j": terminal_energy_sum,
        "upload_energy_consistency_pass": energy_consistency_pass,
        "cache_cost_total": float(math.fsum(cache_cost_by_slot)),
        "cache_download_delay_s_total": float(math.fsum(cache_download_delay_by_slot)),
        "uav_movement_cost_total": float(math.fsum(movement_cost_by_slot)),
        "uav_movement_energy_j_total": float(math.fsum(movement_energy_by_slot)),
        "resource_usage": {
            "per_slot": resource_records,
            "resource_violation_count": len(resource_violations),
            "resource_violations": resource_violations,
            "nonfinite_diagnostic_count": len(nonfinite_paths),
            "nonfinite_diagnostic_paths": sorted(set(nonfinite_paths)),
        },
        "execution_records": slot_rows,
    }


def _aggregate_scenes(scene_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    episode_returns = [float(row["episode_return"]) for row in scene_results]
    unique_tasks = sum(int(row["unique_arrived_task_count"]) for row in scene_results)
    assigned_tasks = sum(int(row["assigned_unique_task_count"]) for row in scene_results)
    success_count = sum(int(row["task_success_count"]) for row in scene_results)
    timeout_count = sum(int(row["task_timeout_count"]) for row in scene_results)
    node_counts: Counter[str] = Counter()
    node_roles: dict[str, str] = {}
    resource_samples: dict[str, dict[str, list[float]]] = {}
    cloud_cpu_usage_samples: list[float] = []
    cloud_uplink_usage_samples: list[float] = []
    cloud_cpu_capacities: list[float] = []
    cloud_uplink_capacities: list[float] = []
    cache_availability_counts: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    latencies: list[float] = []
    for row in scene_results:
        for node_id, detail in row["node_task_details"].items():
            node_counts[node_id] += int(detail["task_count"])
            node_roles[node_id] = str(detail["role"])
        statuses.update({key: int(value) for key, value in row["task_status_counts"].items()})
        cache_availability_counts.update(
            {
                key: int(value)
                for key, value in row["cache_availability_diagnostics"].items()
            }
        )
        latency = row["lifecycle_latency_s"]
        if latency["count"]:
            for event in row["execution_records"]:
                for terminal in event["termination_events"]:
                    value = terminal.get("actual_execution_time_s")
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        latencies.append(float(value))
        for slot_resource in row["resource_usage"]["per_slot"]:
            capacities = slot_resource.get("edge_cpu_capacity_by_node_cycles_per_s", {})
            cpu_usage = slot_resource.get("edge_cpu_usage_by_node_cycles_per_s", {})
            cpu_fraction = slot_resource.get("edge_cpu_fraction_sum_by_node", {})
            bandwidth_usage = slot_resource.get("edge_bandwidth_usage_hz_by_node", {})
            bandwidth_capacity = slot_resource.get("edge_bandwidth_capacity_hz_by_node", {})
            bandwidth_fraction = slot_resource.get("edge_bandwidth_fraction_sum_by_node", {})
            cloud_cpu_usage = slot_resource.get("cloud_cpu_usage_cycles_per_s")
            cloud_uplink_usage = slot_resource.get("cloud_uplink_rate_usage_bps")
            cloud_cpu_capacity = slot_resource.get("cloud_cpu_capacity_cycles_per_s")
            cloud_uplink_capacity = slot_resource.get("cloud_uplink_rate_capacity_bps")
            if isinstance(cloud_cpu_usage, (int, float)) and math.isfinite(float(cloud_cpu_usage)):
                cloud_cpu_usage_samples.append(float(cloud_cpu_usage))
            if isinstance(cloud_uplink_usage, (int, float)) and math.isfinite(float(cloud_uplink_usage)):
                cloud_uplink_usage_samples.append(float(cloud_uplink_usage))
            if isinstance(cloud_cpu_capacity, (int, float)) and math.isfinite(float(cloud_cpu_capacity)):
                cloud_cpu_capacities.append(float(cloud_cpu_capacity))
            if isinstance(cloud_uplink_capacity, (int, float)) and math.isfinite(float(cloud_uplink_capacity)):
                cloud_uplink_capacities.append(float(cloud_uplink_capacity))
            for node_id in set(capacities) | set(cpu_usage) | set(cpu_fraction) | set(bandwidth_usage):
                samples = resource_samples.setdefault(
                    str(node_id),
                    {
                        "cpu_usage_cycles_per_s": [],
                        "cpu_capacity_cycles_per_s": [],
                        "cpu_fraction_sum": [],
                        "cpu_capacity_utilization": [],
                        "bandwidth_usage_hz": [],
                        "bandwidth_capacity_hz": [],
                        "bandwidth_fraction_sum": [],
                    },
                )
                capacity = float(capacities.get(node_id, 0.0))
                usage = float(cpu_usage.get(node_id, 0.0))
                samples["cpu_usage_cycles_per_s"].append(usage)
                samples["cpu_capacity_cycles_per_s"].append(capacity)
                samples["cpu_fraction_sum"].append(float(cpu_fraction.get(node_id, 0.0)))
                samples["cpu_capacity_utilization"].append(usage / capacity if capacity > 0 else 0.0)
                samples["bandwidth_usage_hz"].append(float(bandwidth_usage.get(node_id, 0.0)))
                bandwidth_cap = bandwidth_capacity.get(node_id)
                if bandwidth_cap is not None:
                    samples["bandwidth_capacity_hz"].append(float(bandwidth_cap))
                samples["bandwidth_fraction_sum"].append(float(bandwidth_fraction.get(node_id, 0.0)))
    node_details = {
        node_id: {
            "role": node_roles[node_id],
            "task_count": int(count),
            "ratio_of_assigned": float(count / assigned_tasks) if assigned_tasks else None,
        }
        for node_id, count in sorted(node_counts.items())
    }
    resource_summary = {
        node_id: {
            "role": node_roles.get(node_id),
            "metrics": {
                metric: {
                    "mean": float(statistics.fmean(values)) if values else None,
                    "max": max(values) if values else None,
                }
                for metric, values in metrics.items()
            },
        }
        for node_id, metrics in sorted(resource_samples.items())
    }
    cloud_count = sum(
        count for node_id, count in node_counts.items() if node_roles.get(node_id) == "Cloud"
    )
    node_type_counts = {
        role: int(
            sum(count for node_id, count in node_counts.items() if node_roles.get(node_id) == role)
        )
        for role in ("SV", "RSU", "UAV", "Cloud", "OTHER")
    }
    return {
        "scene_count": len(scene_results),
        "episode_return_sum": float(math.fsum(episode_returns)),
        "episode_return_mean": float(statistics.fmean(episode_returns)) if episode_returns else None,
        "episode_return_min": min(episode_returns) if episode_returns else None,
        "episode_return_max": max(episode_returns) if episode_returns else None,
        "all_episode_reward_checks_pass": all(
            bool(row["reward_sum_matches_slot_utility"]) for row in scene_results
        ),
        "unique_arrived_task_count": unique_tasks,
        "assigned_unique_task_count": assigned_tasks,
        "unassigned_unique_task_count": sum(
            int(row["unassigned_unique_task_count"]) for row in scene_results
        ),
        "task_assignment_coverage": float(assigned_tasks / unique_tasks) if unique_tasks else None,
        "cloud_offload_ratio_of_assigned": float(cloud_count / assigned_tasks) if assigned_tasks else None,
        "keep_stay_cloud_ratio_target_le_0_40": (
            None if not assigned_tasks else bool(cloud_count / assigned_tasks <= 0.40)
        ),
        "dqn_cloud_ratio_target_0_10_to_0_20": (
            None if not assigned_tasks else bool(0.10 <= cloud_count / assigned_tasks <= 0.20)
        ),
        "task_success_count": success_count,
        "task_timeout_count": timeout_count,
        "task_other_terminal_or_missing_count": unique_tasks - success_count - timeout_count,
        "task_status_counts": dict(sorted(statuses.items())),
        "success_rate_of_arrivals": float(success_count / unique_tasks) if unique_tasks else None,
        "timeout_rate_of_arrivals": float(timeout_count / unique_tasks) if unique_tasks else None,
        "node_type_task_count": node_type_counts,
        "cache_availability_diagnostics": dict(sorted(cache_availability_counts.items())),
        "lifecycle_latency_s": _latency_summary(latencies),
        "cumulative_upload_energy_j": float(
            math.fsum(float(row["cumulative_upload_energy_j"]) for row in scene_results)
        ),
        "cache_cost_total": float(math.fsum(float(row["cache_cost_total"]) for row in scene_results)),
        "cache_download_delay_s_total": float(
            math.fsum(float(row["cache_download_delay_s_total"]) for row in scene_results)
        ),
        "uav_movement_cost_total": float(
            math.fsum(float(row["uav_movement_cost_total"]) for row in scene_results)
        ),
        "uav_movement_energy_j_total": float(
            math.fsum(float(row["uav_movement_energy_j_total"]) for row in scene_results)
        ),
        "node_task_details": node_details,
        "resource_usage_by_edge_node": resource_summary,
        "cloud_resource_usage": {
            "resource_model": "independent_per_task_fixed_Rc_Fc",
            "cpu_usage_cycles_per_s_mean": (
                float(statistics.fmean(cloud_cpu_usage_samples)) if cloud_cpu_usage_samples else None
            ),
            "cpu_usage_cycles_per_s_max": max(cloud_cpu_usage_samples) if cloud_cpu_usage_samples else None,
            "cpu_capacity_cycles_per_s": cloud_cpu_capacities[0] if cloud_cpu_capacities else None,
            "uplink_rate_usage_bps_mean": (
                float(statistics.fmean(cloud_uplink_usage_samples)) if cloud_uplink_usage_samples else None
            ),
            "uplink_rate_usage_bps_max": max(cloud_uplink_usage_samples) if cloud_uplink_usage_samples else None,
            "uplink_rate_capacity_bps": cloud_uplink_capacities[0] if cloud_uplink_capacities else None,
        },
        "resource_violation_count": sum(
            int(row["resource_usage"]["resource_violation_count"]) for row in scene_results
        ),
        "nonfinite_diagnostic_count": sum(
            int(row["resource_usage"]["nonfinite_diagnostic_count"]) for row in scene_results
        ),
        "all_upload_energy_checks_pass": all(
            bool(row["upload_energy_consistency_pass"]) for row in scene_results
        ),
    }


def _resolve_output_root(project_root: Path, requested: Path) -> Path:
    output_root = requested if requested.is_absolute() else project_root / requested
    output_root = output_root.resolve(strict=False)
    protected = (
        project_root / "configs",
        project_root / "data",
        project_root / "environment_data",
        project_root / "checkpoints",
        project_root / "replay",
        project_root / "logs",
    )
    for protected_path in protected:
        protected_resolved = protected_path.resolve(strict=False)
        if output_root == protected_resolved or protected_resolved in output_root.parents:
            raise ValueError(f"Calibration output cannot be written under protected path {protected_path}")
    if output_root.exists():
        raise FileExistsError(f"Output root already exists; refusing to overwrite: {output_root}")
    return output_root


def _construct_environment(
    *,
    project_root: Path,
    pool: Path,
    pool_info: Mapping[str, Any],
    episode_seed: int,
    runtime_config: dict[str, Any],
    config_path: Path,
    modules: Mapping[str, Any],
) -> Any:
    """Use the formal loader with a process-local config-copy override."""

    from unittest.mock import patch

    project_config = modules["project_config"]
    original_load_config = project_config.load_config

    def load_runtime_copy(path: str | Path = CONFIG_RELATIVE, *, require_artifacts: bool = True) -> dict[str, Any]:
        actual_path = Path(path).expanduser()
        if not actual_path.is_absolute():
            actual_path = project_root / actual_path
        if actual_path.resolve(strict=False) != config_path.resolve():
            return original_load_config(path, require_artifacts=require_artifacts)
        cfg = copy.deepcopy(runtime_config)
        cfg["_meta"] = {
            "config_path": str(config_path.resolve()),
            "project_root": str(project_root),
        }
        project_config.validate_config(cfg, require_artifacts=require_artifacts)
        return cfg

    with patch.object(project_config, "load_config", side_effect=load_runtime_copy):
        context = modules["pool_contract"]._episode_context(
            project_root, pool, int(episode_seed), dict(pool_info)
        )
    lifecycle_parameters = project_config.get_cross_slot_task_parameters(dict(context.cfg))
    execution_horizon = int(lifecycle_parameters["execution_horizon_slots"])
    steps = modules["aligned_execution_steps"](context)
    if len(steps) != execution_horizon:
        raise RuntimeError(
            f"Aligned Episode length mismatch: expected={execution_horizon}, actual={len(steps)}"
        )
    model = modules["SystemModelContext"](
        service_catalog=context.service_catalog,
        project_cfg=context.cfg,
        cache_nodes=context.cache_nodes,
        cache_config=context.cache_cfg,
        flight_boundary=context.flight_boundary,
    )
    slot_inputs = tuple(
        modules["SlotInput"](
            slot_id=int(step.slot),
            time_s=float(step.time_s),
            scenario_slice=step.scenario_slice,
            tasks=tuple(step.actual_tasks),
            vehicle_positions=step.actual_vehicle_positions,
            predicted_vehicle_positions=step.predicted_vehicle_positions,
            model=model,
            prediction_horizon_mask=step.prediction_horizon_mask,
            scenario_id=f"formal_train:{int(episode_seed)}:{int(step.slot)}",
            source_kind="current_actual",
            prediction_metadata={
                "source": "formal_aligned_execution_steps",
                "episode_seed": int(episode_seed),
                "origin_slot": int(step.slot),
                "future_ground_truth_used_for_candidate_features": False,
            },
        )
        for step in steps
    )
    environment = modules["ExclusiveDQNEnvironment"](
        context=context,
        slot_inputs=slot_inputs,
        decision_steps=steps,
        decision_count=len(steps),
        terminal_at_end=(len(steps) == execution_horizon),
    )
    return environment


def _assert_runtime_environment(
    environment: Any,
    *,
    runtime_config: Mapping[str, Any],
    cpu_multipliers: Mapping[str, float],
    cloud_rate_bps: float,
) -> tuple[dict[str, str], dict[str, float], dict[str, float]]:
    cache_config = environment.model.cache_config
    node_ids = [str(value) for value in environment.model.cache_nodes["node_ids"]]
    node_roles = {
        node_id: str(role).upper()
        for node_id, role in zip(node_ids, environment.model.cache_nodes["node_roles"], strict=True)
    }
    for role in EDGE_ROLES:
        expected = float(_get_path(runtime_config, f"nodes.{role}.cpu_cycles_per_s"))
        actual = float(getattr(cache_config, f"{role}_cpu_cycles_per_s"))
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=0.0):
            raise RuntimeError(
                f"Runtime {role.upper()} CPU overlay did not reach CacheOptimizationConfig: "
                f"expected={expected}, actual={actual}"
            )
    for path in (
        "nodes.cloud.uplink_rate_bps",
        "nodes.cloud.total_bandwidth_bps",
        "communication.cloud.uplink_rate_bps",
    ):
        if float(_get_path(environment.model.project_cfg, path)) != cloud_rate_bps:
            raise RuntimeError(f"Runtime config copy has a mismatched Cloud alias: {path}")
    if float(cache_config.cloud_uplink_rate_bps) != cloud_rate_bps:
        raise RuntimeError("Cloud R_c runtime overlay did not reach Matching/actual CacheConfig")
    if float(cache_config.cloud_cpu_cycles_per_s) != float(
        _get_path(runtime_config, "nodes.cloud.cpu_cycles_per_s")
    ):
        raise RuntimeError("Cloud F_c changed unexpectedly")
    for slot_input in environment.slot_inputs:
        if slot_input.model.cache_config.cloud_uplink_rate_bps != cloud_rate_bps:
            raise RuntimeError("A candidate-evaluation SlotInput has a different Cloud R_c")
        for role in EDGE_ROLES:
            expected = float(_get_path(runtime_config, f"nodes.{role}.cpu_cycles_per_s"))
            if float(getattr(slot_input.model.cache_config, f"{role}_cpu_cycles_per_s")) != expected:
                raise RuntimeError(f"A candidate-evaluation SlotInput has a different {role.upper()} CPU capacity")
    node_cpu_capacity = {
        node_id: float(getattr(cache_config, f"{role.lower()}_cpu_cycles_per_s"))
        for node_id, role in node_roles.items()
    }
    node_bandwidth_capacity = {
        node_id: float(getattr(cache_config, f"{role.lower()}_bandwidth_hz"))
        for node_id, role in node_roles.items()
    }
    return node_roles, node_cpu_capacity, node_bandwidth_capacity


def _run_one_scene(
    *,
    project_root: Path,
    pool: Path,
    pool_info: Mapping[str, Any],
    scene_record: Mapping[str, Any],
    runtime_config: dict[str, Any],
    config_path: Path,
    modules: Mapping[str, Any],
    cpu_multipliers: Mapping[str, float],
    cloud_rate_bps: float,
    cloud_node_id: str,
) -> dict[str, Any]:
    seed = int(scene_record["episode_seed"])
    random.seed(seed)
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required by the project environment") from exc
    np.random.seed(seed)
    environment = _construct_environment(
        project_root=project_root,
        pool=pool,
        pool_info=pool_info,
        episode_seed=seed,
        runtime_config=runtime_config,
        config_path=config_path,
        modules=modules,
    )
    node_roles, node_cpu_capacity, node_bandwidth_capacity = _assert_runtime_environment(
        environment,
        runtime_config=runtime_config,
        cpu_multipliers=cpu_multipliers,
        cloud_rate_bps=cloud_rate_bps,
    )
    task_metadata_by_id = {
        f"{str(task.vehicle_id)}@{int(slot_input.slot_id) + 1}": {
            "input_bits": float(task.input_bits),
            "service_id": int(task.service_id),
            "vehicle_id": str(task.vehicle_id),
            "arrival_slot": int(slot_input.slot_id) + 1,
        }
        for slot_input in environment.slot_inputs
        for task in slot_input.tasks
    }
    node_ids = [str(value) for value in environment.model.cache_nodes["node_ids"]]
    service_ids = [int(value) for value in environment.model.service_catalog["service_ids"]]
    transitions: list[Any] = []
    while not environment.terminated:
        if not transitions or len(transitions) % 25 == 0:
            _ensure_no_formal_generator()
        _state, candidates = environment.observe()
        keep_candidates = [
            candidate
            for candidate in candidates
            if str(candidate.action.operation) == "KEEP_STAY"
        ]
        if len(keep_candidates) != 1:
            raise RuntimeError(
                "KEEP+STAY contract requires exactly one legal KEEP_STAY candidate; "
                f"found={len(keep_candidates)}"
            )
        keep = keep_candidates[0]
        if not np.array_equal(keep.action.cache_array(), environment.cache_state):
            raise RuntimeError("KEEP_STAY candidate does not preserve the current cache")
        current_uav = environment.uav_positions
        candidate_uav = keep.action.uav_dict()
        if set(candidate_uav) != set(current_uav) or any(
            not np.array_equal(candidate_uav[key], current_uav[key]) for key in current_uav
        ):
            raise RuntimeError("KEEP_STAY candidate does not preserve current UAV positions")
        transition = environment.step(keep)
        transitions.append(transition)
    if len(transitions) != int(pool_info["execution_horizon_slots"]):
        raise RuntimeError(
            "KEEP+STAY did not execute the full attested Episode horizon: "
            f"expected={pool_info['execution_horizon_slots']}, actual={len(transitions)}"
        )
    result = _aggregate_episode(
        seed=seed,
        scene_record=scene_record,
        transitions=transitions,
        task_metadata_by_id=task_metadata_by_id,
        node_ids=node_ids,
        service_ids=service_ids,
        node_roles=node_roles,
        node_cpu_capacity=node_cpu_capacity,
        node_bandwidth_capacity=node_bandwidth_capacity,
        cloud_node_id=cloud_node_id,
    )
    result["runtime_parameters"] = {
        "cloud_rate_bps": cloud_rate_bps,
        "cloud_cpu_Fc_cycles_per_s": float(
            _get_path(runtime_config, "nodes.cloud.cpu_cycles_per_s")
        ),
        "edge_cpu_capacity_cycles_per_s_by_role": {
            role.upper(): float(_get_path(runtime_config, f"nodes.{role}.cpu_cycles_per_s"))
            for role in EDGE_ROLES
        },
        "edge_cpu_multipliers": dict(cpu_multipliers),
    }
    result["random_seed_contract"] = {
        "episode_seed": seed,
        "python_random_seed": seed,
        "numpy_seed": seed,
        "task_arrivals_and_trajectory_source": "unchanged attested Training Episode artifacts",
    }
    return result


def _ensure_no_formal_generator() -> None:
    processes = _active_formal_generation_processes()
    if processes:
        resource_snapshot = _system_resource_snapshot()
        raise RuntimeError(
            "Formal environment generation became active during calibration; "
            "stopping this calibration without interrupting the generator. "
            f"Processes: {processes}; resource_snapshot={resource_snapshot}"
        )


def _execute(
    *,
    project_root: Path,
    episode_seeds: Sequence[int],
    cpu_multipliers: Mapping[str, float],
    requested_rates_mbps: Sequence[float] | None,
    output_root_arg: Path,
    expected_source_identity: str,
    core_repair_complete: bool,
) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    if not core_repair_complete:
        raise ValueError("--core-repair-complete is required before running a calibration Episode")
    preflight = _preflight(root, episode_seeds, cpu_multipliers, requested_rates_mbps)
    if preflight["status"] != "PASS":
        raise RuntimeError(
            "Preflight blocked calibration: " + "; ".join(preflight["blockers"])
        )
    actual_source_identity = preflight["source_identity"]["sha256"]
    if expected_source_identity != actual_source_identity:
        raise RuntimeError(
            "Pinned source identity mismatch: "
            f"expected={expected_source_identity}, actual={actual_source_identity}"
        )
    output_root = _resolve_output_root(root, output_root_arg)
    modules = _import_project_modules(root)
    base_config, config_path, config_hashes = _load_base_config(root)
    pool = Path(preflight["episode_identity"]["pool_path"])
    pool, pool_info = modules["pool_contract"]._validate_pool(
        root,
        TRAIN_POOL_RELATIVE,
        expected_split="train",
        expected_seed_start=modules["pool_contract"].DEFAULT_TRAINING_SEED_START,
        expected_episode_count=modules["pool_contract"].FORMAL_TRAINING_EPISODE_COUNT,
        runtime_config=base_config,
    )
    scenes = {
        int(item["episode_seed"]): item
        for item in preflight["episode_identity"]["scenes"]
    }
    base_rate_bps = float(_get_path(base_config, "nodes.cloud.uplink_rate_bps"))
    rates = _requested_rates(base_rate_bps, requested_rates_mbps)
    case_specs: list[dict[str, Any]] = []
    for rate_bps in rates:
        runtime_config, runtime_contract = _build_runtime_config(
            base_config,
            cpu_multipliers=cpu_multipliers,
            cloud_rate_bps=rate_bps,
        )
        case_specs.append(
            {
                "case_id": _case_id(rate_bps, cpu_multipliers),
                "runtime_config": runtime_config,
                "runtime_contract": runtime_contract,
            }
        )
    output_root.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "status": "RUNNING",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "split": "train",
        "strategy": "KEEP_STAY",
        "stage": "edge_cpu_screen_at_cloud_baseline"
        if all(rate == base_rate_bps for rate in rates)
        else "selected_edge_cpu_plan_with_cloud_rate_sensitivity",
        "source_identity": preflight["source_identity"],
        "config_identity": {
            "config_path": str(config_path),
            **config_hashes,
            "pool_manifest_sha256": preflight["episode_identity"]["pool_manifest_sha256"],
            "runtime_override_mode": "deep-copied in-memory config; formal TOML is read-only",
            "baseline_effective_parameters": {
                "edge_cpu_cycles_per_s": {
                    role.upper(): float(_get_path(base_config, f"nodes.{role}.cpu_cycles_per_s"))
                    for role in EDGE_ROLES
                },
                "edge_bandwidth_hz": {
                    role.upper(): float(_get_path(base_config, f"nodes.{role}.bandwidth_hz"))
                    for role in EDGE_ROLES
                },
                "cloud_cpu_Fc_cycles_per_s": float(
                    _get_path(base_config, "nodes.cloud.cpu_cycles_per_s")
                ),
                "cloud_rate_Rc_bps": base_rate_bps,
                "task_input_base_range_kb": [
                    float(_get_path(base_config, "service.generation.input_size_min_kb")),
                    float(_get_path(base_config, "service.generation.input_size_max_kb")),
                ],
            },
        },
        "episode_identity": preflight["episode_identity"],
        "resource_contract": {
            "edge_cpu_multipliers": dict(cpu_multipliers),
            "cloud_cpu_Fc_unchanged": True,
            "task_input_range_unchanged": True,
            "other_config_fields_unchanged": True,
        },
        "screening_targets": {
            "keep_stay_cloud_offload_ratio_at_most": 0.40,
            "dqn_improvement_range": [0.10, 0.20],
            "targets_are_reported_not_forced": True,
            "quality_metrics_required_for_interpretation": [
                "success_rate_of_arrivals",
                "timeout_rate_of_arrivals",
                "lifecycle_latency_s",
                "cumulative_upload_energy_j",
                "cache_cost_total",
                "uav_movement_cost_total",
                "episode_return_mean",
            ],
        },
        "resource_process_check": {
            "formal_generator_active_before_execution": False,
            "policy": "fail closed if the formal environment generator is active",
        },
        "cases": [
            {
                "case_id": case["case_id"],
                "runtime_override": case["runtime_contract"],
                "output_dir": str(output_root / case["case_id"]),
                "status": "PENDING",
            }
            for case in case_specs
        ],
        "completed_scene_case_count": 0,
        "failure": None,
    }
    _write_json(output_root / "calibration_manifest.json", manifest)
    try:
        for case_index, case in enumerate(case_specs):
            _ensure_no_formal_generator()
            case_dir = output_root / case["case_id"]
            case_dir.mkdir(parents=False, exist_ok=False)
            scene_results: list[dict[str, Any]] = []
            for seed in episode_seeds:
                _ensure_no_formal_generator()
                scene = scenes[int(seed)]
                modules["pool_contract"]._validate_episode_snapshot(
                    pool, int(seed), pool_info
                )
                scene_results.append(
                    _run_one_scene(
                        project_root=root,
                        pool=pool,
                        pool_info=pool_info,
                        scene_record=scene,
                        runtime_config=copy.deepcopy(case["runtime_config"]),
                        config_path=config_path,
                        modules=modules,
                        cpu_multipliers=cpu_multipliers,
                        cloud_rate_bps=float(case["runtime_contract"]["cloud_rate_bps"]),
                        cloud_node_id=str(modules["task_allocation"].CLOUD),
                    )
                )
                _write_jsonl(case_dir / "scene_metrics.jsonl", scene_results)
            case_summary = _aggregate_scenes(scene_results)
            _write_json(case_dir / "summary.json", case_summary)
            manifest["cases"][case_index]["status"] = "COMPLETE"
            manifest["cases"][case_index]["summary"] = case_summary
            manifest["completed_scene_case_count"] += len(scene_results)
            _write_json(output_root / "calibration_manifest.json", manifest)
        _ensure_no_formal_generator()
        final_source_identity = _source_identity(root)
        if final_source_identity["sha256"] != expected_source_identity:
            raise RuntimeError("Source files changed during calibration; result cannot be marked complete")
        manifest.update(
            {
                "status": "COMPLETE",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "final_source_identity_sha256": final_source_identity["sha256"],
            }
        )
        _write_json(output_root / "calibration_manifest.json", manifest)
    except BaseException as exc:
        manifest.update(
            {
                "status": "FAILED",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        _write_json(output_root / "calibration_manifest.json", manifest)
        raise
    return {
        "status": manifest["status"],
        "output_root": str(output_root),
        "completed_scene_case_count": manifest["completed_scene_case_count"],
        "cases": manifest["cases"],
    }


def _multipliers_from_args(args: argparse.Namespace) -> dict[str, float]:
    return {
        "sv": float(args.cpu_multiplier_sv),
        "rsu": float(args.cpu_multiplier_rsu),
        "uav": float(args.cpu_multiplier_uav),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.preflight and not args.execute:
        parser.print_help()
        return 0
    root = args.project_root.resolve(strict=True)
    seeds = [int(seed) for seed in (args.episode_seed or [])]
    multipliers = _multipliers_from_args(args)
    if not seeds:
        parser.error("--preflight and --execute require at least one --episode-seed")
    if args.preflight:
        result = _preflight(root, seeds, multipliers, args.cloud_rate_mbps)
        print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result["status"] == "PASS" else 2
    if not args.output_root:
        parser.error("--execute requires --output-root")
    if not args.expected_source_identity:
        parser.error("--execute requires --expected-source-identity from a post-freeze preflight")
    if not args.core_repair_complete:
        parser.error("--execute requires --core-repair-complete")
    result = _execute(
        project_root=root,
        episode_seeds=seeds,
        cpu_multipliers=multipliers,
        requested_rates_mbps=args.cloud_rate_mbps,
        output_root_arg=args.output_root,
        expected_source_identity=args.expected_source_identity,
        core_repair_complete=args.core_repair_complete,
    )
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
