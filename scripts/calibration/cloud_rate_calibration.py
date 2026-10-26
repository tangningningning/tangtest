#!/usr/bin/env python
"""Prepare and run gated, paired Cloud-rate / edge-CPU calibration Episodes.

The default invocation prints help. ``--preflight`` validates a published Training
Episode or an explicitly completed staging Episode and the runtime override
contract without constructing an environment. ``--execute`` is deliberately
explicit and requires a pinned source identity plus confirmation that the parallel
DQN repair is complete.
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
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - supported for older envs
    import tomli as tomllib  # type: ignore[no-redef]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_RELATIVE = Path("configs/dqn/online_joint_dqn.toml")
TRAIN_POOL_RELATIVE = Path("environment_data/proposed_joint_training")
STAGING_ROOT_RELATIVE = Path("environment_data/.formal_vehicle_environment_staging")
GENERATOR_NAME = "generate_formal_vehicle_environment.py"
SCRIPT_SCHEMA_VERSION = "cloud_rate_edge_cpu_calibration_v1"
EDGE_ROLES = ("sv", "rsu", "uav")
STAGING_EXECUTION_MIN_FREE_MEMORY_MB = 1024.0
STAGING_EXECUTION_MEMORY_ARTIFACT_MULTIPLIER = 4.0
STAGING_EXECUTION_MIN_FREE_CPU_CORES = 1.0
SOURCE_ROOTS = (
    Path("src/common"),
    Path("src/decision"),
    Path("src/online_joint_dqn"),
)
SOURCE_FILES = (
    Path("scripts/training/training_pool_contract.py"),
    Path("scripts/generation/generate_formal_vehicle_environment.py"),
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
    parser.add_argument(
        "--split",
        choices=("train", "staging"),
        default="train",
        help="select the published Training pool or an explicitly completed staging Train Episode",
    )
    parser.add_argument(
        "--episode-seed",
        type=int,
        action="append",
        help="completed Episode seed in the selected split; repeat for paired scenes",
    )
    parser.add_argument("--output-root", type=Path, help="new, non-existing output directory; required with --execute")
    parser.add_argument("--expected-source-identity", help="SHA-256 printed by preflight after the source version is fixed")
    parser.add_argument("--core-repair-complete", action="store_true", help="confirm the parallel Online Joint DQN core repair is complete")
    parser.add_argument("--cpu-multiplier-sv", type=float, default=1.0)
    parser.add_argument("--cpu-multiplier-rsu", type=float, default=1.0)
    parser.add_argument("--cpu-multiplier-uav", type=float, default=1.0)
    parser.add_argument(
        "--initial-cache-plan-json",
        help=(
            "optional in-memory initial cache override as a JSON object mapping every "
            "cache node ID to a list of service IDs; requires one staging Episode"
        ),
    )
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


def _process_working_set_mb() -> float | None:
    """Return this process' current resident working set without spawning a helper."""

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            get_process_memory_info = psapi.GetProcessMemoryInfo
            get_process_memory_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            get_process_memory_info.restype = wintypes.BOOL
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if not get_process_memory_info(
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            ):
                return None
            return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
        except (AttributeError, OSError, ValueError):
            return None

    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return rss / (1024.0 * 1024.0) if sys.platform == "darwin" else rss / 1024.0
    except (ImportError, OSError, ValueError):
        return None


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
            "process_working_set_mb": _process_working_set_mb(),
            "policy": "read-only observation; admission is decided by the preflight resource contract",
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
        return {"status": "UNAVAILABLE", "policy": "read-only resource observation is unavailable"}
    return {
        "status": "AVAILABLE",
        "available_memory_mb": memory_info.get("MemAvailable", 0.0) / 1024.0,
        "total_memory_mb": memory_info.get("MemTotal", 0.0) / 1024.0,
        "load_average_1m_5m_15m": load_average,
        "logical_cpu_count": os.cpu_count(),
        "process_working_set_mb": _process_working_set_mb(),
        "policy": "read-only observation; admission is decided by the preflight resource contract",
    }


def _read_only_resource_admission(
    snapshot: Mapping[str, Any], *, artifact_bytes: int
) -> dict[str, Any]:
    """Check whether a streaming, read-only Episode identity scan has headroom."""

    reasons: list[str] = []
    if snapshot.get("status") != "AVAILABLE":
        reasons.append("system resource snapshot is unavailable")
        return {"status": "BLOCKED", "reasons": reasons}

    if "free_physical_memory_mb" in snapshot:
        free_memory_mb = float(snapshot["free_physical_memory_mb"])
        cpu_load = float(snapshot.get("average_cpu_load_percent", math.nan))
        cpu_headroom = math.isfinite(cpu_load) and cpu_load < 100.0
        cpu_diagnostic = {
            "metric": "average_cpu_load_percent",
            "value": cpu_load,
            "not_fully_saturated": cpu_headroom,
        }
    else:
        free_memory_mb = float(snapshot.get("available_memory_mb", 0.0))
        loads = snapshot.get("load_average_1m_5m_15m")
        cpu_count = int(snapshot.get("logical_cpu_count") or 0)
        load_1m = float(loads[0]) if isinstance(loads, (list, tuple)) and loads else math.nan
        cpu_headroom = cpu_count > 0 and math.isfinite(load_1m) and load_1m < cpu_count
        cpu_diagnostic = {
            "metric": "load_average_1m_vs_logical_cpu_count",
            "value": load_1m,
            "logical_cpu_count": cpu_count,
            "not_fully_saturated": cpu_headroom,
        }

    required_memory_mb = float(max(0, artifact_bytes)) / (1024.0 * 1024.0)
    memory_headroom = math.isfinite(free_memory_mb) and free_memory_mb > required_memory_mb
    if not memory_headroom:
        reasons.append(
            "available physical memory does not exceed the selected Episode's total artifact bytes"
        )
    if not cpu_headroom:
        reasons.append("CPU load is unavailable or all logical processors are saturated")
    return {
        "status": "PASS" if not reasons else "BLOCKED",
        "scope": "read-only streaming artifact identity preflight only",
        "available_memory_mb": free_memory_mb,
        "episode_artifact_bytes": int(artifact_bytes),
        "minimum_memory_rule": "available memory must exceed the bytes read; hashes are streamed in 1 MiB blocks",
        "cpu_headroom": cpu_diagnostic,
        "reasons": reasons,
        "execution_admission": "not granted by this read-only resource check",
    }


def _staging_execution_resource_admission(
    snapshot: Mapping[str, Any], *, artifact_bytes: int
) -> dict[str, Any]:
    """Require spare memory and one CPU core while the formal generator runs."""

    reasons: list[str] = []
    if snapshot.get("status") != "AVAILABLE":
        return {
            "status": "BLOCKED",
            "reasons": ["system resource snapshot is unavailable"],
        }

    if "free_physical_memory_mb" in snapshot:
        free_memory_mb = float(snapshot["free_physical_memory_mb"])
        logical_cpu_count = int(snapshot.get("logical_cpu_count") or 0)
        cpu_load_percent = float(snapshot.get("average_cpu_load_percent", math.nan))
        free_cpu_cores = (
            logical_cpu_count * (1.0 - cpu_load_percent / 100.0)
            if logical_cpu_count > 0 and math.isfinite(cpu_load_percent)
            else math.nan
        )
        cpu_metric = {
            "metric": "estimated_free_logical_cpu_cores",
            "logical_cpu_count": logical_cpu_count,
            "average_cpu_load_percent": cpu_load_percent,
            "value": free_cpu_cores,
        }
    else:
        free_memory_mb = float(snapshot.get("available_memory_mb", 0.0))
        logical_cpu_count = int(snapshot.get("logical_cpu_count") or 0)
        loads = snapshot.get("load_average_1m_5m_15m")
        load_1m = (
            float(loads[0])
            if isinstance(loads, (list, tuple)) and loads
            else math.nan
        )
        free_cpu_cores = logical_cpu_count - load_1m
        cpu_metric = {
            "metric": "logical_cpu_count_minus_1m_load_average",
            "logical_cpu_count": logical_cpu_count,
            "load_average_1m": load_1m,
            "value": free_cpu_cores,
        }

    artifact_mb = float(max(0, artifact_bytes)) / (1024.0 * 1024.0)
    required_free_memory_mb = max(
        STAGING_EXECUTION_MIN_FREE_MEMORY_MB,
        STAGING_EXECUTION_MEMORY_ARTIFACT_MULTIPLIER * artifact_mb,
    )
    memory_ok = math.isfinite(free_memory_mb) and free_memory_mb >= required_free_memory_mb
    cpu_ok = math.isfinite(free_cpu_cores) and free_cpu_cores >= STAGING_EXECUTION_MIN_FREE_CPU_CORES
    if not memory_ok:
        reasons.append(
            "free physical memory is below the staging-execution safety floor"
        )
    if not cpu_ok:
        reasons.append(
            "estimated free CPU capacity is below one logical core or unavailable"
        )
    return {
        "status": "PASS" if not reasons else "BLOCKED",
        "scope": "staging calibration only while formal generation is active",
        "available_memory_mb": free_memory_mb,
        "episode_artifact_bytes": int(artifact_bytes),
        "required_free_memory_mb": required_free_memory_mb,
        "minimum_free_memory_mb": STAGING_EXECUTION_MIN_FREE_MEMORY_MB,
        "artifact_memory_multiplier": STAGING_EXECUTION_MEMORY_ARTIFACT_MULTIPLIER,
        "cpu_headroom": cpu_metric,
        "minimum_free_cpu_cores": STAGING_EXECUTION_MIN_FREE_CPU_CORES,
        "reasons": reasons,
    }


def _read_staging_writer_state(
    project_root: Path, *, episode_seed: int, expected_plan_fingerprint: str
) -> dict[str, Any]:
    """Confirm the completed staging Episode is not the generator's current target."""

    staging_root = (project_root.resolve(strict=True) / STAGING_ROOT_RELATIVE).resolve(strict=True)
    status_path = staging_root / "generation_status.json"
    plan_path = staging_root / "data_split_manifest.json"
    status_bytes = status_path.read_bytes()
    plan_bytes = plan_path.read_bytes()
    status = json.loads(status_bytes)
    plan = json.loads(plan_bytes)
    if status.get("status") not in {"generating", "complete"} or status.get("failure"):
        raise RuntimeError("staging generation status is no longer usable")
    if (
        status.get("plan_fingerprint") != expected_plan_fingerprint
        or plan.get("plan_fingerprint") != expected_plan_fingerprint
    ):
        raise RuntimeError("staging plan identity changed during calibration")
    completed_raw = status.get("completed_episode_seeds", {}).get("train")
    if not isinstance(completed_raw, list):
        raise RuntimeError("staging progress has no completed Train seed list")
    completed = {int(seed) for seed in completed_raw}
    if len(completed) != len(completed_raw):
        raise RuntimeError("staging progress contains duplicate completed Train seeds")
    completed_count = int(status.get("completed", {}).get("train", -1))
    if completed_count != len(completed):
        raise RuntimeError("staging Train completion count and seed list differ")
    if int(episode_seed) not in completed:
        raise RuntimeError(
            f"staging Episode is no longer marked complete: seed={episode_seed}"
        )
    current = status.get("current_episode") or {}
    current_seed = (
        int(current["seed"])
        if current.get("split") == "train" and current.get("seed") is not None
        else None
    )
    if current_seed == int(episode_seed):
        raise RuntimeError(
            f"selected staging Episode is the active generator target: seed={episode_seed}"
        )
    return {
        "status": status.get("status"),
        "completed_train_episode_count": completed_count,
        "selected_episode_completed": True,
        "current_episode": current,
        "status_sha256": _sha256_bytes(status_bytes),
        "plan_sha256": _sha256_bytes(plan_bytes),
        "plan_fingerprint": expected_plan_fingerprint,
        "staging_root": str(staging_root),
    }


def _ensure_scene_execution_safe(
    *,
    project_root: Path,
    split: str,
    episode_seed: int,
    plan_fingerprint: str | None,
    artifact_bytes: int,
    slot_index: int | None = None,
    execution_phase: str = "unspecified",
) -> dict[str, Any]:
    """Fail closed on writer overlap and gate staging work while generation runs."""

    processes = _active_formal_generation_processes()
    detection_failed = any("detection_error" in process for process in processes)
    if detection_failed:
        raise RuntimeError("formal generator process detection failed")
    if split != "staging":
        if processes:
            raise RuntimeError(
                "published Training calibration remains blocked while the formal generator is active"
            )
        return {"generator_active": False, "processes": []}

    if not plan_fingerprint:
        raise RuntimeError("staging execution is missing its pinned plan fingerprint")
    writer_state = _read_staging_writer_state(
        project_root,
        episode_seed=episode_seed,
        expected_plan_fingerprint=plan_fingerprint,
    )
    resource_admission: dict[str, Any] = {
        "status": "NOT_REQUIRED",
        "reason": "formal generator is not active",
    }
    if processes:
        snapshot = _system_resource_snapshot()
        resource_admission = _staging_execution_resource_admission(
            snapshot, artifact_bytes=artifact_bytes
        )
        if resource_admission["status"] != "PASS":
            diagnostic = {
                "episode_seed": int(episode_seed),
                "slot_index": None if slot_index is None else int(slot_index),
                "completed_slots_before_check": None if slot_index is None else int(slot_index),
                "execution_phase": str(execution_phase),
                "available_memory_mb": resource_admission.get("available_memory_mb"),
                "required_free_memory_mb": resource_admission.get("required_free_memory_mb"),
                "minimum_free_memory_mb": resource_admission.get("minimum_free_memory_mb"),
                "process_working_set_mb": snapshot.get("process_working_set_mb"),
                "cpu_headroom": resource_admission.get("cpu_headroom"),
            }
            raise RuntimeError(
                "staging calibration resource guard rejected execution while the formal generator is active: "
                + "; ".join(resource_admission.get("reasons", []))
                + "; snapshot="
                + json.dumps(diagnostic, sort_keys=True, separators=(",", ":"))
            )
    return {
        "generator_active": bool(processes),
        "processes": processes,
        "writer_state": writer_state,
        "slot_index": None if slot_index is None else int(slot_index),
        "execution_phase": str(execution_phase),
        "resource_snapshot": snapshot if processes else None,
        "resource_admission": resource_admission,
    }


def _import_staging_identity_modules(project_root: Path) -> dict[str, Any]:
    """Load only the generation validators needed to attest a staging Episode."""

    for path in (project_root / "src", project_root / "scripts" / "generation"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from common import project_config
    import generate_episode_realization as episode_realization

    return {
        "project_config": project_config,
        "episode_realization": episode_realization,
    }


def _validate_episode_in_isolated_process(
    project_root: Path,
    episode_dir: Path,
    *,
    seed: int,
    split: str,
    config_path: Path,
    origin_count: int,
) -> str:
    """Run the generation validator without retaining its top-level Torch import."""

    validator_code = "\n".join(
        (
            "import json, sys",
            "from pathlib import Path",
            "root = Path(sys.argv[1])",
            "sys.path.insert(0, str(root / 'src'))",
            "sys.path.insert(0, str(root / 'scripts' / 'generation'))",
            "import generate_formal_vehicle_environment as generator",
            "manifest_sha256 = generator._validate_episode(",
            "    Path(sys.argv[2]),",
            "    seed=int(sys.argv[3]),",
            "    split=sys.argv[4],",
            "    config_path=Path(sys.argv[5]),",
            "    origin_count=int(sys.argv[6]),",
            ")",
            "print(json.dumps({'manifest_sha256': manifest_sha256}, separators=(',', ':'))) ",
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            validator_code,
            str(project_root),
            str(episode_dir),
            str(int(seed)),
            str(split),
            str(config_path),
            str(int(origin_count)),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        if len(details) > 4000:
            details = details[-4000:]
        raise RuntimeError(
            f"isolated formal Episode validation failed for seed={seed}: "
            f"exit_code={completed.returncode}; {details}"
        )
    for line in reversed(completed.stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        manifest_sha256 = payload.get("manifest_sha256") if isinstance(payload, dict) else None
        if isinstance(manifest_sha256, str) and len(manifest_sha256) == 64:
            return manifest_sha256
    raise RuntimeError(
        f"isolated formal Episode validation returned no manifest identity for seed={seed}"
    )


def _staging_runtime_parameter_values(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = (
        "nodes.sv.cpu_cycles_per_s",
        "nodes.sv.bandwidth_hz",
        "nodes.rsu.cpu_cycles_per_s",
        "nodes.rsu.bandwidth_hz",
        "nodes.uav.cpu_cycles_per_s",
        "nodes.uav.bandwidth_hz",
        "nodes.cloud.cpu_cycles_per_s",
        "nodes.cloud.total_cpu_cycles_per_s",
        "nodes.cloud.cpu_capacity_cycles_per_s",
        "nodes.cloud.resource_mode",
        "nodes.cloud.resource_sharing",
        "nodes.cloud.uplink_rate_bps",
        "nodes.cloud.total_bandwidth_bps",
        "communication.cloud.uplink_rate_bps",
        "service.generation.input_size_min_kb",
        "service.generation.input_size_max_kb",
        "service.generation.cycles_per_bit_min",
        "service.generation.cycles_per_bit_max",
        "task.input_relative_jitter",
    )
    return {path: _get_path(config, path) for path in paths}


def _validate_staging_episodes(
    project_root: Path,
    episode_seeds: Sequence[int],
    formal_config: Mapping[str, Any],
    modules: Mapping[str, Any],
) -> dict[str, Any]:
    """Attest explicitly completed staging Episodes without touching their inputs."""

    if not episode_seeds:
        raise ValueError("Specify at least one completed staging Episode with --episode-seed")
    if len(set(int(seed) for seed in episode_seeds)) != len(episode_seeds):
        raise ValueError("Episode seeds must be unique within a paired calibration")

    root = project_root.resolve(strict=True)
    staging_root = (root / STAGING_ROOT_RELATIVE).resolve(strict=True)
    status_path = staging_root / "generation_status.json"
    plan_path = staging_root / "data_split_manifest.json"
    staged_config_path = staging_root / "generation_config.toml"
    status_bytes = status_path.read_bytes()
    plan_bytes = plan_path.read_bytes()
    status = json.loads(status_bytes)
    plan = json.loads(plan_bytes)
    if status.get("status") not in {"generating", "complete"} or status.get("failure"):
        raise RuntimeError(
            f"staging generation status is not usable: status={status.get('status')!r}, "
            f"failure={status.get('failure')!r}"
        )
    if plan.get("schema_version") != "formal_vehicle_environment_split_manifest_v2":
        raise RuntimeError(f"unexpected staging split manifest schema: {plan.get('schema_version')!r}")
    if plan.get("dataset_status") not in {"generating", "complete"}:
        raise RuntimeError(f"staging plan status is not usable: {plan.get('dataset_status')!r}")
    plan_fingerprint = plan.get("plan_fingerprint")
    if not plan_fingerprint or status.get("plan_fingerprint") != plan_fingerprint:
        raise RuntimeError("staging progress and split manifest plan fingerprints differ")

    completed_raw = status.get("completed_episode_seeds", {}).get("train")
    if not isinstance(completed_raw, list):
        raise RuntimeError("staging progress has no completed Train seed list")
    completed = [int(seed) for seed in completed_raw]
    if len(completed) != len(set(completed)):
        raise RuntimeError("staging progress contains duplicate completed Train seeds")
    completed_count = int(status.get("completed", {}).get("train", -1))
    if completed_count != len(completed):
        raise RuntimeError(
            f"staging Train completion count/list differ: count={completed_count}, seeds={len(completed)}"
        )
    current = status.get("current_episode") or {}
    current_seed = (
        int(current["seed"])
        if current.get("split") == "train" and current.get("seed") is not None
        else None
    )
    absent = [int(seed) for seed in episode_seeds if int(seed) not in completed]
    if absent:
        raise RuntimeError(f"Episode seed(s) are not marked complete in the current staging manifest: {absent}")
    if current_seed in {int(seed) for seed in episode_seeds}:
        raise RuntimeError(f"selected Episode is currently being generated: seed={current_seed}")

    train_plan = plan.get("train", {})
    active_seeds = {int(seed) for seed in train_plan.get("active_episode_seeds", [])}
    not_planned = [int(seed) for seed in episode_seeds if int(seed) not in active_seeds]
    if not_planned:
        raise RuntimeError(f"selected Episode seed(s) are not active in the staging Train plan: {not_planned}")

    formal_config_path = (root / CONFIG_RELATIVE).resolve(strict=True)
    formal_config_sha256 = _sha256_file(formal_config_path)
    if formal_config_sha256 != plan.get("config_sha256"):
        raise RuntimeError("formal config hash differs from the current staging generation plan")
    if Path(str(plan.get("config_path", ""))).as_posix() != CONFIG_RELATIVE.as_posix():
        raise RuntimeError(f"staging plan points to an unexpected formal config: {plan.get('config_path')!r}")
    if not staged_config_path.is_file():
        raise FileNotFoundError(f"staged generation config is missing: {staged_config_path}")

    staged_config_sha256 = _sha256_file(staged_config_path)
    staged_config = modules["episode_realization"].load_and_validate_config(staged_config_path)
    formal_values = _staging_runtime_parameter_values(formal_config)
    staged_values = _staging_runtime_parameter_values(staged_config)
    mismatches = {
        key: {"formal": formal_values[key], "staging": staged_values[key]}
        for key in formal_values
        if formal_values[key] != staged_values[key]
    }
    if mismatches:
        raise RuntimeError(f"formal and staging physical/task parameter contracts differ: {mismatches}")

    project_config = modules["project_config"]
    service_catalog = project_config.resolve_path(
        root,
        project_config.get_config_value(staged_config, "paths.service_catalog"),
    ).resolve(strict=True)
    try:
        service_catalog.relative_to(staging_root)
    except ValueError as exc:
        raise RuntimeError(
            f"staged config service catalog resolves outside the selected staging root: {service_catalog}"
        ) from exc
    service_catalog_sha256 = _sha256_file(service_catalog)
    if service_catalog_sha256 != plan.get("service_catalog_sha256"):
        raise RuntimeError("staged service catalog hash differs from the current generation plan")

    expected_source_hashes = plan.get("generation_source_sha256")
    if not isinstance(expected_source_hashes, dict) or not expected_source_hashes:
        raise RuntimeError("staging split manifest has no generation source identity map")
    source_hashes_verified: dict[str, str] = {}
    for relative, expected_hash in sorted(expected_source_hashes.items()):
        source_path = (root / Path(str(relative))).resolve(strict=True)
        try:
            source_path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"unsafe generation source path in staging manifest: {relative!r}") from exc
        actual_hash = _sha256_file(source_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"staging generation source identity changed: {relative}; "
                f"expected={expected_hash}, actual={actual_hash}"
            )
        source_hashes_verified[str(relative)] = actual_hash

    protocol = project_config.get_episode_data_protocol(staged_config)
    execution_horizon = int(protocol["execution_horizon_slots"])
    if int(plan.get("time_contract", {}).get("execution_horizon_slots", -1)) != execution_horizon:
        raise RuntimeError("staging plan execution horizon differs from its staged config")

    scenes: list[dict[str, Any]] = []
    total_artifact_bytes = service_catalog.stat().st_size
    for seed_value in episode_seeds:
        seed = int(seed_value)
        episode_dir = (staging_root / "train" / f"episode_{seed}").resolve(strict=True)
        try:
            episode_dir.relative_to(staging_root / "train")
        except ValueError as exc:
            raise RuntimeError(f"selected Episode path escapes staging Train: {episode_dir}") from exc
        manifest_path = episode_dir / "episode_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"completed Episode manifest is missing: {manifest_path}")
        manifest_sha256_before = _sha256_file(manifest_path)
        validated_manifest_sha256 = _validate_episode_in_isolated_process(
            root,
            episode_dir,
            seed=seed,
            split="train",
            config_path=staged_config_path,
            origin_count=execution_horizon,
        )
        if validated_manifest_sha256 != manifest_sha256_before:
            raise RuntimeError(f"Episode {seed} manifest changed during generation-contract validation")
        episode_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(episode_manifest.get("episode_seed", -1)) != seed:
            raise RuntimeError(f"Episode manifest seed mismatch: requested={seed}")
        if episode_manifest.get("service_catalog_sha256") not in {None, service_catalog_sha256}:
            raise RuntimeError(f"Episode {seed} service catalog identity differs from staging plan")
        artifact_names = episode_manifest.get("generated_artifacts")
        expected_artifact_hashes = episode_manifest.get("generated_artifact_sha256")
        if not isinstance(artifact_names, list) or not artifact_names or not isinstance(expected_artifact_hashes, dict):
            raise RuntimeError(f"Episode {seed} has no complete artifact hash inventory")
        if len(set(str(name) for name in artifact_names)) != len(artifact_names):
            raise RuntimeError(f"Episode {seed} artifact inventory contains duplicates")
        artifact_hashes: dict[str, str] = {}
        artifact_bytes = manifest_path.stat().st_size
        for name in artifact_names:
            name_text = str(name)
            if Path(name_text).name != name_text or name_text in {"", ".", ".."}:
                raise RuntimeError(f"Episode {seed} has an unsafe artifact path: {name_text!r}")
            artifact_path = episode_dir / name_text
            if not artifact_path.is_file():
                raise FileNotFoundError(f"Episode {seed} required artifact is missing: {artifact_path}")
            actual_hash = _sha256_file(artifact_path)
            if actual_hash != expected_artifact_hashes.get(name_text):
                raise RuntimeError(f"Episode {seed} artifact hash differs from manifest: {name_text}")
            artifact_hashes[name_text] = actual_hash
            artifact_bytes += artifact_path.stat().st_size
        if set(artifact_hashes) != set(str(name) for name in expected_artifact_hashes):
            raise RuntimeError(f"Episode {seed} artifact inventory and hash map do not match")
        if _sha256_file(manifest_path) != manifest_sha256_before:
            raise RuntimeError(f"Episode {seed} manifest changed during artifact verification")
        total_artifact_bytes += artifact_bytes
        scene_id = f"staging-train:{seed}"
        scene_sha256 = _canonical_sha256(
            {
                "plan_fingerprint": plan_fingerprint,
                "episode_manifest_sha256": manifest_sha256_before,
                "artifact_sha256": artifact_hashes,
                "formal_config_sha256": formal_config_sha256,
                "staged_config_sha256": staged_config_sha256,
                "service_catalog_sha256": service_catalog_sha256,
            }
        )
        scenes.append(
            {
                "split": "staging",
                "episode_seed": seed,
                "scene_id": scene_id,
                "scene_sha256": scene_sha256,
                "episode_manifest_sha256": manifest_sha256_before,
                "artifact_sha256": artifact_hashes,
                "artifact_bytes": artifact_bytes,
                "task_data_model": "500_1000_KB",
            }
        )

    final_status_bytes = status_path.read_bytes()
    final_plan_bytes = plan_path.read_bytes()
    final_status = json.loads(final_status_bytes)
    final_plan = json.loads(final_plan_bytes)
    final_completed = {
        int(seed) for seed in final_status.get("completed_episode_seeds", {}).get("train", [])
    }
    final_current = final_status.get("current_episode") or {}
    final_current_seed = (
        int(final_current["seed"])
        if final_current.get("split") == "train" and final_current.get("seed") is not None
        else None
    )
    if not {int(seed) for seed in episode_seeds}.issubset(final_completed):
        raise RuntimeError("selected Episode completion state changed during identity validation")
    if final_current_seed in {int(seed) for seed in episode_seeds}:
        raise RuntimeError(f"selected Episode became the active generation target: seed={final_current_seed}")
    if final_status.get("plan_fingerprint") != plan_fingerprint or final_plan.get("plan_fingerprint") != plan_fingerprint:
        raise RuntimeError("staging plan identity changed during Episode validation")
    if final_plan.get("config_sha256") != formal_config_sha256:
        raise RuntimeError("staging config identity changed during Episode validation")
    if _sha256_file(staged_config_path) != staged_config_sha256:
        raise RuntimeError("staged generation config changed during Episode validation")
    if _sha256_file(service_catalog) != service_catalog_sha256:
        raise RuntimeError("staged service catalog changed during Episode validation")

    return {
        "pool_path": str(staging_root / "train"),
        "staging_root": str(staging_root),
        "status_path": str(status_path),
        "status": status.get("status"),
        "current_episode": current,
        "completed_train_episode_count": completed_count,
        "status_sha256_before": _sha256_bytes(status_bytes),
        "status_sha256_after": _sha256_bytes(final_status_bytes),
        "plan_path": str(plan_path),
        "plan_sha256_before": _sha256_bytes(plan_bytes),
        "plan_sha256_after": _sha256_bytes(final_plan_bytes),
        "plan_fingerprint": plan_fingerprint,
        "formal_config_path": str(formal_config_path),
        "formal_config_sha256": formal_config_sha256,
        "staged_config_path": str(staged_config_path),
        "staged_config_sha256": staged_config_sha256,
        "service_catalog_path": str(service_catalog),
        "service_catalog_sha256": service_catalog_sha256,
        "generation_source_sha256_verified": source_hashes_verified,
        "execution_horizon_slots": execution_horizon,
        "total_artifact_bytes": total_artifact_bytes,
        "scenes": scenes,
        "_staged_config": staged_config,
        "_staged_config_path": staged_config_path,
    }


def _import_project_modules(project_root: Path, *, split: str) -> dict[str, Any]:
    source_root = str(project_root / "src")
    training_root = str(project_root / "scripts" / "training")
    import_paths = (source_root, training_root) if split == "train" else (source_root,)
    for path in import_paths:
        if path not in sys.path:
            sys.path.insert(0, path)
    from common import project_config
    from common import environment_interface
    from common.environment_interface import aligned_execution_steps
    from decision import task_allocation
    from decision import cache_configuration
    from online_joint_dqn.candidate_evaluator import SlotInput, SystemModelContext
    from online_joint_dqn.environment_adapter import ExclusiveDQNEnvironment
    modules = {
        "project_config": project_config,
        "environment_interface": environment_interface,
        "aligned_execution_steps": aligned_execution_steps,
        "task_allocation": task_allocation,
        "cache_configuration": cache_configuration,
        "SlotInput": SlotInput,
        "SystemModelContext": SystemModelContext,
        "ExclusiveDQNEnvironment": ExclusiveDQNEnvironment,
    }
    if split == "train":
        import training_pool_contract

        modules["pool_contract"] = training_pool_contract
    return modules


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
            "evidence_class": (
                "configured_model_baseline_not_measured_effective_throughput"
                if float(cloud_rate_bps) == float(_get_path(base_config, cloud_paths[0]))
                else "sensitivity_only_effective_per_task_rate_not_deployment_claim"
            ),
        }
        changed_paths.append(path)
    edge_cpu_evidence = {
        role.upper(): {
            "evidence_class": (
                "configured_model_baseline_no_device_specification_verified"
                if float(cpu_multipliers[role]) == 1.0
                else "sensitivity_only_assumption_not_a_hardware_upgrade"
            ),
            "baseline_cycles_per_s": float(
                _get_path(base_config, f"nodes.{role}.cpu_cycles_per_s")
            ),
            "runtime_cycles_per_s": float(
                _get_path(runtime_config, f"nodes.{role}.cpu_cycles_per_s")
            ),
            "multiplier": float(cpu_multipliers[role]),
        }
        for role in EDGE_ROLES
    }
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
        "parameter_evidence": {
            "edge_cpu_by_role": edge_cpu_evidence,
            "cloud_uplink": {
                "evidence_class": overrides[cloud_paths[0]]["evidence_class"],
                "model": "independent_per_task_fixed_Rc_Fc_without_aggregate_pool_cap",
                "rate_is_measured_effective_throughput": False,
            },
        },
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
    split: str = "train",
    *,
    retain_runtime_config: bool = False,
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
    if split not in {"train", "staging"}:
        blockers.append(f"Unsupported preflight split: {split}")
    if not episode_seeds:
        blockers.append("Specify at least one completed Episode with --episode-seed")
    generator_processes_before = _active_formal_generation_processes()
    process_detection_failed = any(
        "detection_error" in process for process in generator_processes_before
    )
    generator_active_before = bool(generator_processes_before) and not process_detection_failed
    resource_snapshot_before = (
        _system_resource_snapshot()
        if generator_active_before or split == "staging"
        else None
    )
    if process_detection_failed:
        blockers.append(
            "Formal generator process detection failed; the staging writer target cannot be established"
        )
    preliminary_resource_admission = None
    if split == "staging" and resource_snapshot_before is not None:
        preliminary_resource_admission = _read_only_resource_admission(
            resource_snapshot_before, artifact_bytes=0
        )
        if preliminary_resource_admission["status"] != "PASS":
            blockers.append(
                "Read-only staging identity preflight has insufficient or unavailable resource headroom: "
                + "; ".join(preliminary_resource_admission["reasons"])
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

    if base_config is not None:
        try:
            if split == "staging":
                if (
                    preliminary_resource_admission is None
                    or preliminary_resource_admission["status"] != "PASS"
                    or process_detection_failed
                ):
                    raise RuntimeError("staging identity validation did not pass its initial resource/process gate")
                staging_modules = _import_staging_identity_modules(root)
                scenes = _validate_staging_episodes(
                    root, episode_seeds, base_config, staging_modules
                )
                runtime_base_config = scenes["_staged_config"]
                runtime_config_path = scenes["_staged_config_path"]
                project_config_module = staging_modules["project_config"]
                formal_staging_values = _staging_runtime_parameter_values(base_config)
                staged_runtime_values = _staging_runtime_parameter_values(runtime_base_config)
                config_identity["staging_identity"] = {
                    key: value
                    for key, value in scenes.items()
                    if not key.startswith("_") and key != "scenes"
                }
                config_identity["staging_identity"]["formal_staging_parameter_contract_equal"] = (
                    formal_staging_values == staged_runtime_values
                )
            else:
                modules = _import_project_modules(root)
                scenes = _validate_pool_and_scenes(root, modules, base_config, episode_seeds)
                runtime_base_config = base_config
                runtime_config_path = root / CONFIG_RELATIVE
                project_config_module = modules["project_config"]

            scene_identity = {
                key: value for key, value in scenes.items() if not key.startswith("_")
            }
            for preview in override_previews:
                runtime_config, _ = _build_runtime_config(
                    dict(runtime_base_config),
                    cpu_multipliers=cpu_multipliers,
                    cloud_rate_bps=float(preview["cloud_rate_bps"]),
                )
                runtime_config["_meta"] = {
                    "config_path": str(runtime_config_path.resolve()),
                    "project_root": str(root),
                }
                project_config_module.validate_for_formal_experiment(runtime_config)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            blockers.append(f"{split.capitalize()} Episode identity preflight failed: {type(exc).__name__}: {exc}")

    generator_processes_after = _active_formal_generation_processes()
    process_detection_failed_after = any(
        "detection_error" in process for process in generator_processes_after
    )
    if process_detection_failed_after:
        blockers.append(
            "Formal generator process detection failed at the end of preflight; writer isolation cannot be confirmed"
        )
    generator_active_after = bool(generator_processes_after) and not process_detection_failed_after
    resource_snapshot_after = (
        _system_resource_snapshot()
        if generator_active_after or split == "staging"
        else None
    )
    active_during_preflight = generator_active_before or generator_active_after
    resource_admission: dict[str, Any] = {
        "status": "NOT_REQUIRED",
        "scope": "read-only Episode identity preflight",
        "execution_admission": "not granted by preflight",
    }
    if active_during_preflight or split == "staging":
        snapshots = [
            snapshot
            for snapshot in (resource_snapshot_before, resource_snapshot_after)
            if snapshot is not None
        ]
        artifact_bytes = int((scenes or {}).get("total_artifact_bytes", 0))
        resource_checks = [
            _read_only_resource_admission(snapshot, artifact_bytes=artifact_bytes)
            for snapshot in snapshots
        ]
        failed_resource_checks = [
            check for check in resource_checks if check["status"] != "PASS"
        ]
        resource_admission = {
            "status": "PASS" if snapshots and not failed_resource_checks else "BLOCKED",
            "scope": "read-only streaming artifact identity preflight only",
            "checks": resource_checks,
            "execution_admission": "not granted; active-generator execution remains separately gated",
        }
        if failed_resource_checks:
            reasons = [
                reason
                for check in failed_resource_checks
                for reason in check.get("reasons", [])
            ]
            blockers.append("Resource guard rejected read-only Episode preflight: " + "; ".join(reasons))

    result = {
        "status": "BLOCKED" if blockers else "PASS",
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "project_root": str(root),
        "split": split,
        "strategy": "KEEP_STAY",
        "source_identity": source_identity,
        "source_contract_checks": source_checks,
        "formal_generation_processes": {
            "before": generator_processes_before,
            "after": generator_processes_after,
            "active_during_preflight": active_during_preflight,
            "policy": (
                "an active generator does not by itself block preflight; the selected Episode must be "
                "manifest-complete, hash-verified, separate from the current writer target, and pass "
                "the read-only resource check"
            ),
        },
        "resource_snapshot": {
            "before": resource_snapshot_before,
            "after": resource_snapshot_after,
        },
        "resource_admission": resource_admission,
        "config_identity": config_identity,
        "runtime_override_previews": override_previews,
        "episode_identity": scene_identity,
        "blockers": blockers,
        "preflight_only": True,
    }
    if retain_runtime_config and split == "staging" and isinstance(scenes, Mapping):
        staged_runtime_config = scenes.get("_staged_config")
        if isinstance(staged_runtime_config, dict):
            result["_validated_staged_config"] = staged_runtime_config
    return result


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
    transitions: Iterable[Any],
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
    slot_count = 0
    final_cumulative_energy = 0.0

    for transition_index, transition in enumerate(transitions):
        slot_count = transition_index + 1
        diagnostics = transition.diagnostics
        final_cumulative_energy = float(
            diagnostics.get("episode_vehicle_upload_energy_j", 0.0)
        )
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
        "slot_count": slot_count,
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
    pool_info: Mapping[str, Any] | None,
    episode_seed: int,
    split: str,
    runtime_config: dict[str, Any],
    config_path: Path,
    modules: Mapping[str, Any],
    initial_cache_plan: Mapping[str, Sequence[int]] | None = None,
) -> Any:
    """Load a verified scene without changing its on-disk configuration."""

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
        if split == "train":
            if pool_info is None:
                raise RuntimeError("published Training loading requires validated pool identity")
            context = modules["pool_contract"]._episode_context(
                project_root, pool, int(episode_seed), dict(pool_info)
            )
        elif split == "staging":
            episode_dir = pool / f"episode_{int(episode_seed)}"
            context = modules["environment_interface"].load_environment_context(
                str(config_path.resolve()), realization_dir=episode_dir
            )
        else:
            raise ValueError(f"Unsupported calibration split: {split!r}")
    initial_cache_override_metadata: dict[str, Any] | None = None
    initial_cache_override: Any = None
    if initial_cache_plan is not None:
        initial_cache_override, initial_cache_override_metadata = _build_initial_cache_override(
            context,
            initial_cache_plan,
            modules["cache_configuration"],
        )
        context = replace(context, initial_cache=initial_cache_override)
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
            scenario_id=f"{split}_train:{int(episode_seed)}:{int(step.slot)}",
            source_kind="current_actual",
            prediction_metadata={
                "source": f"{split}_aligned_execution_steps",
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
    if initial_cache_override is not None:
        import numpy as np

        if not np.array_equal(environment.cache_state, initial_cache_override):
            raise RuntimeError("Environment did not initialize from the requested cache plan")
        environment._calibration_initial_cache_override = initial_cache_override_metadata
    # The runner needs only these small invariants after construction. The full
    # EnvironmentContext also retains scenarios and realization arrays already
    # represented by the aligned SlotInput/DecisionForecast sequence.
    from types import SimpleNamespace

    environment.context = SimpleNamespace(
        observation_spec=context.observation_spec,
        train_origin_start=int(context.train_origin_start),
        train_origin_end=int(context.train_origin_end),
        initial_cache=copy.deepcopy(context.initial_cache),
        initial_uav=copy.deepcopy(context.initial_uav),
    )
    return environment


def _build_initial_cache_override(
    context: Any,
    plan: Mapping[str, Sequence[int]],
    cache_configuration_module: Any,
) -> tuple[Any, dict[str, Any]]:
    """Build and validate a node/service cache matrix without changing source config."""

    import numpy as np

    node_ids = [str(value) for value in context.cache_nodes["node_ids"]]
    node_roles = [str(value).upper() for value in context.cache_nodes["node_roles"]]
    service_ids = [int(value) for value in context.service_catalog["service_ids"]]
    if len(node_ids) != len(node_roles) or len(set(node_ids)) != len(node_ids):
        raise ValueError("Cache node IDs/roles are malformed or duplicated")
    supplied_node_ids = {str(node_id) for node_id in plan}
    if supplied_node_ids != set(node_ids):
        missing = sorted(set(node_ids) - supplied_node_ids)
        unknown = sorted(supplied_node_ids - set(node_ids))
        raise ValueError(
            f"Initial cache plan node IDs do not match the Episode: missing={missing}, unknown={unknown}"
        )
    service_index = {service_id: index for index, service_id in enumerate(service_ids)}
    matrix = np.zeros((len(node_ids), len(service_ids)), dtype=np.int8)
    normalized_node_services: dict[str, list[int]] = {}
    for node_index, node_id in enumerate(node_ids):
        requested = list(plan[node_id])
        selected = [int(service_id) for service_id in requested]
        if len(set(selected)) != len(selected):
            raise ValueError(f"Initial cache plan repeats a service for node {node_id}")
        unknown_services = sorted(set(selected) - set(service_ids))
        if unknown_services:
            raise ValueError(
                f"Initial cache plan contains unknown service IDs at node {node_id}: {unknown_services}"
            )
        for service_id in selected:
            matrix[node_index, service_index[service_id]] = 1
        normalized_node_services[node_id] = sorted(selected)
    cache_configuration_module.validate_cache_shape_and_capacity(
        matrix,
        context.service_catalog,
        context.cache_nodes,
    )
    service_sizes_mb = np.asarray(
        context.service_catalog["service_sizes_mb"], dtype=np.float64
    )
    capacities = np.asarray(context.cache_nodes["capacities_mb"], dtype=np.float64)
    used = matrix.astype(np.float64) @ service_sizes_mb
    node_storage = {
        node_id: {
            "role": role,
            "used_mb": float(used[index]),
            "capacity_mb": float(capacities[index]),
            "service_ids": normalized_node_services[node_id],
        }
        for index, (node_id, role) in enumerate(zip(node_ids, node_roles, strict=True))
    }
    identity_payload = {
        "node_ids": node_ids,
        "service_ids": service_ids,
        "matrix": matrix.tolist(),
    }
    identity_bytes = json.dumps(
        identity_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    metadata = {
        "policy": "in_memory_initial_cache_override",
        "node_services": normalized_node_services,
        "node_storage": node_storage,
        "matrix_sha256": _sha256_bytes(identity_bytes),
        "applied_before_episode_initialization": True,
        "formal_config_modified": False,
    }
    return matrix, metadata


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
    pool_info: Mapping[str, Any] | None,
    scene_record: Mapping[str, Any],
    runtime_config: dict[str, Any],
    config_path: Path,
    modules: Mapping[str, Any],
    cpu_multipliers: Mapping[str, float],
    cloud_rate_bps: float,
    cloud_node_id: str,
    split: str,
    execution_horizon_slots: int,
    execution_guard: Callable[[int, str], Mapping[str, Any]] | None = None,
    initial_cache_plan: Mapping[str, Sequence[int]] | None = None,
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
        split=split,
        runtime_config=runtime_config,
        config_path=config_path,
        modules=modules,
        initial_cache_plan=initial_cache_plan,
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
    guard_observations: list[dict[str, Any]] = []

    completed_transition_count = 0

    def transition_stream() -> Iterable[Any]:
        nonlocal completed_transition_count
        while not environment.terminated:
            if completed_transition_count % 25 == 0:
                if execution_guard is None:
                    _ensure_no_formal_generator()
                else:
                    guard_observations.append(
                        dict(
                            execution_guard(
                                completed_transition_count,
                                "before_slot_observation",
                            )
                        )
                    )
            state, candidates = environment.observe()
            keep_candidates = [
                candidate
                for candidate in candidates
                if str(candidate.action.operation) == "KEEP_STAY"
            ]
            if len(keep_candidates) != 1:
                raise RuntimeError(
                    "KEEP+STAY contract requires exactly one legal KEEP_STAY candidate; "
                    f"found={len(keep_candidates)} at slot={completed_transition_count}"
                )
            keep = keep_candidates[0]
            if not np.array_equal(keep.action.cache_array(), environment.cache_state):
                raise RuntimeError(
                    f"KEEP_STAY candidate does not preserve the current cache at slot={completed_transition_count}"
                )
            current_uav = environment.uav_positions
            candidate_uav = keep.action.uav_dict()
            if set(candidate_uav) != set(current_uav) or any(
                not np.array_equal(candidate_uav[key], current_uav[key]) for key in current_uav
            ):
                raise RuntimeError(
                    f"KEEP_STAY candidate does not preserve current UAV positions at slot={completed_transition_count}"
                )
            transition = environment.step(keep)
            completed_transition_count += 1
            del state, candidates, keep_candidates, keep, current_uav, candidate_uav
            yield transition
            del transition

    result = _aggregate_episode(
        seed=seed,
        scene_record=scene_record,
        transitions=transition_stream(),
        task_metadata_by_id=task_metadata_by_id,
        node_ids=node_ids,
        service_ids=service_ids,
        node_roles=node_roles,
        node_cpu_capacity=node_cpu_capacity,
        node_bandwidth_capacity=node_bandwidth_capacity,
        cloud_node_id=cloud_node_id,
    )
    if completed_transition_count != int(execution_horizon_slots):
        raise RuntimeError(
            "KEEP+STAY did not execute the full attested Episode horizon: "
            f"expected={execution_horizon_slots}, actual={completed_transition_count}"
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
        "task_arrivals_and_trajectory_source": f"unchanged attested {split} Episode artifacts",
    }
    result["execution_safety_checks"] = guard_observations
    if initial_cache_plan is not None:
        result["initial_cache_override"] = environment._calibration_initial_cache_override
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
    split: str = "train",
    initial_cache_plan: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    if not core_repair_complete:
        raise ValueError("--core-repair-complete is required before running a calibration Episode")
    if split not in {"train", "staging"}:
        raise ValueError(f"Unsupported calibration split: {split!r}")
    if initial_cache_plan is not None and (split != "staging" or len(episode_seeds) != 1):
        raise ValueError("An initial cache override requires exactly one staging Episode")
    preflight = _preflight(
        root,
        episode_seeds,
        cpu_multipliers,
        requested_rates_mbps,
        split=split,
        retain_runtime_config=(split == "staging"),
    )
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
    modules = _import_project_modules(root, split=split)
    pool = Path(preflight["episode_identity"]["pool_path"])
    pool_info: Mapping[str, Any] | None
    if split == "train":
        base_config, config_path, config_hashes = _load_base_config(root)
        pool, pool_info = modules["pool_contract"]._validate_pool(
            root,
            TRAIN_POOL_RELATIVE,
            expected_split="train",
            expected_seed_start=modules["pool_contract"].DEFAULT_TRAINING_SEED_START,
            expected_episode_count=modules["pool_contract"].FORMAL_TRAINING_EPISODE_COUNT,
            runtime_config=base_config,
        )
        execution_horizon_slots = int(pool_info["execution_horizon_slots"])
        selected_config_identity = {
            "config_path": str(config_path),
            **config_hashes,
            "pool_manifest_sha256": preflight["episode_identity"]["pool_manifest_sha256"],
        }
    else:
        staging_identity = preflight["episode_identity"]
        config_path = Path(staging_identity["staged_config_path"]).resolve(strict=True)
        base_config = preflight.get("_validated_staged_config")
        if not isinstance(base_config, dict):
            raise RuntimeError("staging preflight did not retain its validated runtime config")
        formal_config_sha256 = str(staging_identity["formal_config_sha256"])
        staged_config_sha256 = _sha256_file(config_path)
        if staged_config_sha256 != staging_identity["staged_config_sha256"]:
            raise RuntimeError("Staged config changed after identity preflight")
        pool_info = None
        execution_horizon_slots = int(staging_identity["execution_horizon_slots"])
        selected_config_identity = {
            "selected_config_path": str(config_path),
            "selected_config_sha256": staged_config_sha256,
            "formal_config_path": str(staging_identity["formal_config_path"]),
            "formal_config_sha256": formal_config_sha256,
            "staging_plan_fingerprint": staging_identity["plan_fingerprint"],
            "service_catalog_path": str(staging_identity["service_catalog_path"]),
            "service_catalog_sha256": str(staging_identity["service_catalog_sha256"]),
            "runtime_override_mode": "deep-copied in-memory config; both Matching prediction and actual execution are overridden",
        }

    baseline_scene_identity = preflight["episode_identity"]

    def fresh_staging_attestation(*, phase: str) -> dict[str, Any]:
        fresh = _preflight(
            root, episode_seeds, cpu_multipliers, requested_rates_mbps, split="staging"
        )
        if fresh["status"] != "PASS":
            raise RuntimeError(
                f"Staging identity/resource preflight failed {phase}: "
                + "; ".join(fresh["blockers"])
            )
        if fresh["source_identity"]["sha256"] != expected_source_identity:
            raise RuntimeError(f"Pinned source identity changed {phase}")
        now = fresh["episode_identity"]
        stable_fields = (
            "plan_fingerprint",
            "formal_config_sha256",
            "staged_config_sha256",
            "service_catalog_sha256",
        )
        for field in stable_fields:
            if now.get(field) != baseline_scene_identity.get(field):
                raise RuntimeError(f"Staging identity field changed {phase}: {field}")
        expected_scenes = {
            int(item["episode_seed"]): (
                item["scene_sha256"],
                item["episode_manifest_sha256"],
                item["artifact_sha256"],
            )
            for item in baseline_scene_identity["scenes"]
        }
        current_scenes = {
            int(item["episode_seed"]): (
                item["scene_sha256"],
                item["episode_manifest_sha256"],
                item["artifact_sha256"],
            )
            for item in now["scenes"]
        }
        if current_scenes != expected_scenes:
            raise RuntimeError(f"Selected staging Episode identity changed {phase}")
        return fresh

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
        "split": split,
        "strategy": "KEEP_STAY",
        "stage": "edge_cpu_screen_at_cloud_baseline"
        if all(rate == base_rate_bps for rate in rates)
        else "selected_edge_cpu_plan_with_cloud_rate_sensitivity",
        "source_identity": preflight["source_identity"],
        "config_identity": {
            **selected_config_identity,
            "runtime_override_mode": "deep-copied in-memory config; source TOML files are read-only",
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
        "initial_cache_override": (
            None
            if initial_cache_plan is None
            else {
                "policy": "in_memory_initial_cache_override",
                "requested_node_services": {
                    str(node_id): sorted(int(service_id) for service_id in services)
                    for node_id, services in sorted(initial_cache_plan.items())
                },
                "formal_config_modified": False,
            }
        ),
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
            "formal_generator_active_before_execution": bool(
                preflight["formal_generation_processes"]["active_during_preflight"]
            ),
            "policy": (
                "staging: allow only explicitly completed, hash-verified episodes whose current writer target differs; "
                "require memory/CPU admission while generator is active and recheck at episode start and every 25 slots"
                if split == "staging"
                else "published Training: fail closed while the formal environment generator is active"
            ),
            "preflight_resource_admission": preflight["resource_admission"],
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
            if split != "staging":
                _ensure_no_formal_generator()
            case_dir = output_root / case["case_id"]
            case_dir.mkdir(parents=False, exist_ok=False)
            scene_results: list[dict[str, Any]] = []
            for seed in episode_seeds:
                scene = scenes[int(seed)]
                if split == "staging":
                    execution_guard = lambda slot_index, execution_phase, seed=seed, scene=scene: _ensure_scene_execution_safe(
                        project_root=root,
                        split="staging",
                        episode_seed=int(seed),
                        plan_fingerprint=str(baseline_scene_identity["plan_fingerprint"]),
                        artifact_bytes=int(scene["artifact_bytes"]),
                        slot_index=int(slot_index),
                        execution_phase=str(execution_phase),
                    )
                    execution_guard(0, "before_environment_initialization")
                else:
                    _ensure_no_formal_generator()
                    execution_guard = None
                if split == "train":
                    assert pool_info is not None
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
                        split=split,
                        execution_horizon_slots=execution_horizon_slots,
                        execution_guard=execution_guard,
                        initial_cache_plan=initial_cache_plan,
                    )
                )
                _write_jsonl(case_dir / "scene_metrics.jsonl", scene_results)
            case_summary = _aggregate_scenes(scene_results)
            _write_json(case_dir / "summary.json", case_summary)
            manifest["cases"][case_index]["status"] = "COMPLETE"
            manifest["cases"][case_index]["summary"] = case_summary
            manifest["completed_scene_case_count"] += len(scene_results)
            _write_json(output_root / "calibration_manifest.json", manifest)
        if split == "staging":
            final_attestation = fresh_staging_attestation(phase="at completion")
            final_source_identity = final_attestation["source_identity"]
        else:
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
        "split": split,
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
        result = _preflight(
            root, seeds, multipliers, args.cloud_rate_mbps, split=args.split
        )
        print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result["status"] == "PASS" else 2
    if not args.output_root:
        parser.error("--execute requires --output-root")
    if not args.expected_source_identity:
        parser.error("--execute requires --expected-source-identity from a post-freeze preflight")
    if not args.core_repair_complete:
        parser.error("--execute requires --core-repair-complete")
    initial_cache_plan: Mapping[str, Sequence[int]] | None = None
    if args.initial_cache_plan_json is not None:
        try:
            parsed_plan = json.loads(args.initial_cache_plan_json)
        except json.JSONDecodeError as exc:
            parser.error(f"--initial-cache-plan-json is not valid JSON: {exc}")
        if not isinstance(parsed_plan, dict) or not all(
            isinstance(node_id, str) and isinstance(service_ids, list)
            for node_id, service_ids in parsed_plan.items()
        ):
            parser.error("--initial-cache-plan-json must map node IDs to lists of service IDs")
        if any(
            isinstance(service_id, bool) or not isinstance(service_id, int)
            for service_ids in parsed_plan.values()
            for service_id in service_ids
        ):
            parser.error("Initial cache service IDs must be integers")
        initial_cache_plan = parsed_plan
    result = _execute(
        project_root=root,
        episode_seeds=seeds,
        cpu_multipliers=multipliers,
        requested_rates_mbps=args.cloud_rate_mbps,
        output_root_arg=args.output_root,
        expected_source_identity=args.expected_source_identity,
        core_repair_complete=args.core_repair_complete,
        split=args.split,
        initial_cache_plan=initial_cache_plan,
    )
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
