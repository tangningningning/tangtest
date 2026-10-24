"""Resume continuous SUMO data generation with teleport-aware FCD validation.

This entry point keeps the primary generator's code fingerprint unchanged while
it finalizes runs produced by that generator. SUMO counts vehicles in an active
teleport as running, while FCD omits those vehicles until they re-enter the
network. The validator reconciles those two official outputs using SUMO's
vehicle-ID-bearing teleport warnings; it does not synthesize trajectory rows.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATION_DIR = Path(__file__).resolve().parent
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))

import generate_continuous_sumo_dataset as generator  # noqa: E402


TELEPORT_START_RE = re.compile(
    r"Teleporting vehicle '([^']+)';.*?\btime=([0-9]+(?:\.[0-9]+)?)\."
)
TELEPORT_END_RE = re.compile(
    r"Vehicle '([^']+)' ends teleporting on edge '[^']+',\s*time=([0-9]+(?:\.[0-9]+)?)\."
)
TELEPORT_TERMINAL_RE = re.compile(
    r"Vehicle '([^']+)' teleports beyond arrival edge '[^']+',\s*time=([0-9]+(?:\.[0-9]+)?)\."
)


def _postprocessor_provenance() -> dict[str, str]:
    path = Path(__file__).resolve()
    fast_validator = GENERATION_DIR / "fast_validate_sumo_fcd.ps1"
    return {
        "tool": path.name,
        "path": str(path),
        "sha256": generator._sha256_file(path),
        "fast_validator": fast_validator.name,
        "fast_validator_sha256": generator._sha256_file(fast_validator),
        "validation_contract": "FCD rows + active SUMO teleport intervals == summary running vehicles",
    }


def _read_teleport_intervals(
    warning_log: Path,
    duration_s: float,
    dt_s: float,
) -> tuple[dict[int, int], dict[str, int]]:
    """Return per-step active-teleport deltas and auditable event counts."""
    expected_steps = int(round(duration_s / dt_s))
    open_teleports: dict[str, float] = {}
    deltas: dict[int, int] = defaultdict(int)
    interval_count = 0
    teleported_ids: set[str] = set()
    unfinished_count = 0
    terminal_vaporization_count = 0

    def to_step(time_s: float, event: str, vehicle_id: str) -> int:
        step = int(round(time_s / dt_s))
        if not math.isclose(time_s, step * dt_s, abs_tol=1e-6):
            raise RuntimeError(
                f"SUMO teleport event is off-grid: vehicle={vehicle_id}, "
                f"event={event}, time_s={time_s}, dt_s={dt_s}"
            )
        if step < 0 or step > expected_steps:
            raise RuntimeError(
                f"SUMO teleport event is outside the realization: "
                f"vehicle={vehicle_id}, event={event}, time_s={time_s}"
            )
        return step

    def add_interval(vehicle_id: str, start_s: float, end_s: float) -> None:
        nonlocal interval_count
        start_step = to_step(start_s, "start", vehicle_id)
        end_step = to_step(end_s, "end", vehicle_id)
        if end_step < start_step:
            raise RuntimeError(
                f"SUMO teleport ends before it starts: vehicle={vehicle_id}, "
                f"start_s={start_s}, end_s={end_s}"
            )
        interval_count += 1
        teleported_ids.add(vehicle_id)
        # SUMO may start and end a teleport within one simulation step. Such a
        # vehicle is present in that step's FCD output and has no absent interval.
        if start_step < end_step:
            deltas[start_step] += 1
            deltas[end_step] -= 1

    if not warning_log.is_file():
        raise RuntimeError(f"SUMO warning log is missing: {warning_log}")
    with warning_log.open("rt", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            start_match = TELEPORT_START_RE.search(line)
            if start_match is not None:
                vehicle_id = start_match.group(1)
                start_s = float(start_match.group(2))
                to_step(start_s, "start", vehicle_id)
                if vehicle_id in open_teleports:
                    raise RuntimeError(
                        f"SUMO logged nested teleport starts for vehicle {vehicle_id} "
                        f"at line {line_number}."
                    )
                open_teleports[vehicle_id] = start_s
                continue

            end_match = TELEPORT_END_RE.search(line)
            if end_match is not None:
                vehicle_id = end_match.group(1)
                end_s = float(end_match.group(2))
                to_step(end_s, "end", vehicle_id)
                start_s = open_teleports.pop(vehicle_id, None)
                if start_s is None:
                    raise RuntimeError(
                        f"SUMO logged a teleport end without a start for vehicle "
                        f"{vehicle_id} at line {line_number}."
                    )
                add_interval(vehicle_id, start_s, end_s)
                continue

            terminal_match = TELEPORT_TERMINAL_RE.search(line)
            if terminal_match is not None:
                vehicle_id = terminal_match.group(1)
                event_s = float(terminal_match.group(2))
                start_s = open_teleports.pop(vehicle_id, None)
                if start_s is None:
                    raise RuntimeError(
                        f"SUMO logged teleport vaporization without a start for "
                        f"vehicle {vehicle_id} at line {line_number}."
                    )
                # If a vehicle starts and is vaporized by teleport in the same
                # simulation step, it never contributes to summary.running.
                # For an already active teleport, SUMO applies terminal removal
                # to summary.running at the next time boundary.
                end_s = event_s + dt_s if start_s < event_s else event_s
                to_step(end_s, "terminal_vaporization", vehicle_id)
                add_interval(vehicle_id, start_s, end_s)
                terminal_vaporization_count += 1

    for vehicle_id, start_s in open_teleports.items():
        # An unclosed warning at the simulation horizon means SUMO still had the
        # vehicle teleporting when output ended; do not infer an FCD position.
        add_interval(vehicle_id, start_s, duration_s)
        unfinished_count += 1

    return dict(deltas), {
        "sumo_teleport_interval_count": interval_count,
        "sumo_teleported_vehicle_count": len(teleported_ids),
        "sumo_teleport_in_progress_at_end_count": unfinished_count,
        "sumo_teleport_terminal_vaporization_count": terminal_vaporization_count,
    }


def _validate_trajectory_and_capture_activity(
    trajectory_path: Path,
    summary: dict[int, dict[str, Any]],
    warning_log: Path,
    duration_s: float,
    dt_s: float,
) -> tuple[dict[str, tuple[float, float]], int, int, dict[str, int]]:
    expected_steps = int(round(duration_s / dt_s))
    teleport_deltas, teleport_metrics = _read_teleport_intervals(
        warning_log,
        duration_s,
        dt_s,
    )
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        raise RuntimeError(
            "PowerShell 7 is required for the streaming SUMO FCD validator; "
            "enable pwsh and resume without changing the dataset contract."
        )
    helper = GENERATION_DIR / "fast_validate_sumo_fcd.ps1"
    with tempfile.TemporaryDirectory(
        prefix=".fcd-validation-", dir=trajectory_path.parent
    ) as temporary_dir:
        temporary_path = Path(temporary_dir)
        timeline_path = temporary_path / "timeline.tsv"
        observed_path = temporary_path / "observed.tsv"
        command = [
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(helper),
            "-TrajectoryPath",
            str(trajectory_path),
            "-ExpectedSteps",
            str(expected_steps),
            "-DtS",
            repr(dt_s),
            "-TimelinePath",
            str(timeline_path),
            "-ObservedPath",
            str(observed_path),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Fast .NET FCD validation failed: "
                + (completed.stderr or completed.stdout).strip()[-4000:]
            )
        try:
            fast_metrics = json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Fast .NET FCD validator returned invalid metrics: "
                + completed.stdout[-1000:]
            ) from exc

        time_count = 0
        row_count = int(fast_metrics["trajectory_vehicle_state_row_count"])
        teleport_adjusted_timestep_rows = 0
        active_teleports = 0
        with timeline_path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream, delimiter="\t"):
                if len(row) != 2 or int(row[0]) != time_count:
                    raise RuntimeError(
                        f"FCD validator produced an invalid time index at step {time_count}."
                    )
                active_teleports += teleport_deltas.get(time_count, 0)
                if active_teleports < 0:
                    raise RuntimeError(
                        f"Teleport accounting became negative at time={time_count * dt_s}."
                    )
                rows_in_step = int(row[1])
                expected_running = summary[time_count]["running"]
                expected_fcd_rows = expected_running - active_teleports
                if expected_fcd_rows < 0 or rows_in_step != expected_fcd_rows:
                    raise RuntimeError(
                        "FCD rows disagree with SUMO summary after accounting for "
                        "vehicles temporarily absent during teleport: "
                        f"time={time_count * dt_s}, trajectory={rows_in_step}, "
                        f"summary_running={expected_running}, "
                        f"teleporting={active_teleports}, "
                        f"expected_fcd_rows={expected_fcd_rows}"
                    )
                teleport_adjusted_timestep_rows += active_teleports
                time_count += 1
        if time_count != expected_steps:
            raise RuntimeError(
                f"FCD time axis is incomplete: steps={time_count}/{expected_steps}."
            )

        first_last: dict[str, tuple[float, float]] = {}
        with observed_path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream, delimiter="\t"):
                if len(row) != 3 or row[0] in first_last:
                    raise RuntimeError("FCD validator produced malformed observed vehicle data.")
                first_last[row[0]] = (float(row[1]), float(row[2]))

    if len(first_last) != int(fast_metrics["trajectory_unique_observed_vehicle_count"]):
        raise RuntimeError("FCD validator's observed vehicle count does not match its index.")
    teleport_metrics["fcd_teleport_adjusted_vehicle_state_count"] = (
        teleport_adjusted_timestep_rows
    )
    return first_last, row_count, time_count, teleport_metrics


def _build_episode_events_and_validate(
    *,
    spec: Any,
    episode_dir: Path,
    dt_s: float,
) -> dict[str, Any]:
    trajectory_path = episode_dir / "trajectory.fcd.xml.gz"
    summary_path = episode_dir / "summary.xml.gz"
    tripinfo_path = episode_dir / "tripinfo.xml.gz"
    route_path = episode_dir / "sumo_demand.rou.xml"
    warning_log = episode_dir / "sumo.stderr.log"
    summary = generator._read_summary(summary_path, spec.duration_s, dt_s)
    observed, trajectory_rows, trajectory_steps, teleport_metrics = (
        _validate_trajectory_and_capture_activity(
            trajectory_path,
            summary,
            warning_log,
            spec.duration_s,
            dt_s,
        )
    )
    schedules = generator._read_route_schedule(route_path)
    tripinfo = generator._read_tripinfo(tripinfo_path)
    if not set(tripinfo).issubset(set(schedules)):
        raise RuntimeError("SUMO tripinfo contains vehicle IDs absent from the demand route file.")
    tripinfo_teleport_vaporized = sum(
        item.get("vaporized") == "teleport" for item in tripinfo.values()
    )
    if tripinfo_teleport_vaporized != teleport_metrics[
        "sumo_teleport_terminal_vaporization_count"
    ]:
        raise RuntimeError(
            "SUMO teleport terminal warnings disagree with tripinfo vaporized vehicles: "
            f"warnings={teleport_metrics['sumo_teleport_terminal_vaporization_count']}, "
            f"tripinfo={tripinfo_teleport_vaporized}"
        )
    teleport_metrics["tripinfo_teleport_vaporized_vehicle_count"] = (
        tripinfo_teleport_vaporized
    )
    activity = generator._write_vehicle_activity(
        episode_dir / "vehicle_activity.csv.gz",
        schedules,
        observed,
        tripinfo,
        spec.duration_s,
        dt_s,
    )
    final_summary = summary[trajectory_steps - 1]
    return {
        "trajectory_timestep_count": trajectory_steps,
        "trajectory_vehicle_state_row_count": trajectory_rows,
        "trajectory_unique_observed_vehicle_count": len(observed),
        "summary_timestep_count": len(summary),
        "summary_first_time_s": summary[0]["time_s"],
        "summary_last_time_s": final_summary["time_s"],
        "summary_max_running_vehicles": max(item["running"] for item in summary.values()),
        "summary_mean_running_vehicles": sum(item["running"] for item in summary.values())
        / len(summary),
        "summary_total_loaded_at_end": final_summary["loaded"],
        "summary_total_inserted_at_end": final_summary["inserted"],
        "summary_total_waiting_at_end": final_summary["waiting"],
        "summary_total_arrived_at_end": final_summary["arrived"],
        "summary_total_teleports_at_end": final_summary["teleports"],
        "summary_final_mean_speed_mps": final_summary["mean_speed_mps"],
        "summary_final_halting_vehicles": final_summary["halting"],
        **teleport_metrics,
        **activity,
    }


def _install_finalizer_patch() -> None:
    provenance = _postprocessor_provenance()
    original_prepare = generator._prepare_dataset_manifest

    def prepare_dataset_manifest(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], str]:
        manifest, fingerprint = original_prepare(*args, **kwargs)
        existing = manifest.get("teleport_aware_postprocessing")
        if existing and existing.get("sha256") != provenance["sha256"]:
            has_finalized_episodes = any(
                entry.get("status") == "complete"
                for entry in manifest.get("episodes", {}).values()
            )
            if has_finalized_episodes:
                raise RuntimeError(
                    "Dataset contains finalized episodes from a different recovery-tool "
                    "version; refusing to mix validators."
                )
        manifest["teleport_aware_postprocessing"] = provenance
        return manifest, fingerprint

    generator._prepare_dataset_manifest = prepare_dataset_manifest
    generator._build_episode_events_and_validate = _build_episode_events_and_validate
    original_finalize = generator._finalize_episode

    def finalize_episode(**kwargs: Any) -> dict[str, Any]:
        manifest = original_finalize(**kwargs)
        episode_dir = Path(kwargs["episode_dir"])
        manifest_path = episode_dir / "episode_manifest.json"
        manifest["teleport_aware_postprocessing"] = provenance
        manifest["sumo_command_contract"]["vehicle_activity_status_source"] = (
            "FCD active rows + SUMO stderr teleport start/end events + tripinfo "
            "+ route demand + summary"
        )
        generator._atomic_json(manifest_path, manifest)
        progress_path = episode_dir / "episode_progress.json"
        progress = generator._load_json(progress_path)
        progress["episode_manifest_sha256"] = generator._sha256_file(manifest_path)
        progress["teleport_aware_postprocessing_sha256"] = provenance["sha256"]
        generator._atomic_json(progress_path, progress)
        return manifest

    generator._finalize_episode = finalize_episode
    original_update = generator._update_dataset_episode_entry

    def update_dataset_episode_entry(*args: Any, **kwargs: Any) -> None:
        original_update(*args, **kwargs)
        output_root = kwargs.get("output_root", args[0] if len(args) > 0 else None)
        dataset_manifest = kwargs.get(
            "dataset_manifest", args[1] if len(args) > 1 else None
        )
        manifest = kwargs.get("episode_manifest", args[3] if len(args) > 3 else None)
        status = kwargs.get("status", args[4] if len(args) > 4 else None)
        if status == "complete" and manifest is not None:
            dataset_manifest.pop("last_error", None)
            generator._atomic_json(
                Path(output_root) / "dataset_manifest.json",
                dataset_manifest,
            )

    generator._update_dataset_episode_entry = update_dataset_episode_entry


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--validate-only" in arguments or "--self-test-recovery" in arguments:
        raise SystemExit(
            "This entry point resumes SUMO data generation; validation-only and smoke-test "
            "modes belong to generate_continuous_sumo_dataset.py."
        )
    if "--resume" not in arguments:
        arguments.insert(0, "--resume")
    _install_finalizer_patch()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    return generator.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
