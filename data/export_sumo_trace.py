#!/usr/bin/env python3
"""
Export SUMO FCD trajectory to a lower-frequency CSV-like GPSDAT trace.

This script intentionally reuses SUMO's official tools/traceExporter.py
instead of reimplementing FCD XML parsing.

Default project layout:
    data/
    └── sumo/
        ├── osm.sumocfg
        └── output/
            ├── vehicle_fcd.xml.gz
            └── vehicle_trace_1s.csv

Usage:
    python scripts/export_sumo_trace.py

Optional:
    python scripts/export_sumo_trace.py --delta-t 1
    python scripts/export_sumo_trace.py --sumo-home "C:\\Program Files (x86)\\Eclipse\\Sumo"
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_project_root() -> Path:
    """
    Locate the project root by walking upward from this script and looking
    for data/sumo. Falls back to the current working directory.
    """
    script_path = Path(__file__).resolve()

    for parent in [script_path.parent, *script_path.parents]:
        if (parent / "data" / "sumo").exists():
            return parent

    cwd = Path.cwd().resolve()
    if (cwd / "data" / "sumo").exists():
        return cwd

    raise FileNotFoundError(
        "Cannot locate project root. Expected a 'data/sumo' directory "
        "above this script or under the current working directory."
    )


def resolve_sumo_home(explicit_sumo_home: str | None) -> Path:
    """
    Resolve SUMO_HOME using:
      1. --sumo-home
      2. SUMO_HOME environment variable
      3. installed 'sumo' executable
    """
    candidates: list[Path] = []

    if explicit_sumo_home:
        candidates.append(Path(explicit_sumo_home).expanduser())

    env_sumo_home = os.environ.get("SUMO_HOME")
    if env_sumo_home:
        candidates.append(Path(env_sumo_home).expanduser())

    sumo_executable = shutil.which("sumo")
    if sumo_executable:
        exe_path = Path(sumo_executable).resolve()
        # Typical layouts:
        #   <SUMO_HOME>/bin/sumo(.exe)
        #   <SUMO_HOME>/sumo(.exe)
        if exe_path.parent.name.lower() == "bin":
            candidates.append(exe_path.parent.parent)
        else:
            candidates.append(exe_path.parent)

    for candidate in candidates:
        candidate = candidate.resolve()
        trace_exporter = candidate / "tools" / "traceExporter.py"
        if trace_exporter.is_file():
            return candidate

    checked = "\n".join(f"  - {p}" for p in candidates) or "  (none)"
    raise FileNotFoundError(
        "SUMO installation was not found.\n"
        "Set the SUMO_HOME environment variable or pass --sumo-home.\n"
        f"Checked:\n{checked}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use SUMO's official traceExporter.py to downsample an FCD trace."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Input FCD file. Default: "
            "data/sumo/output/vehicle_fcd.xml.gz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output GPSDAT file. Default: "
            "data/sumo/output/vehicle_trace_1s.csv"
        ),
    )
    parser.add_argument(
        "--delta-t",
        type=float,
        default=1.0,
        help="Export interval in seconds. Default: 1.0",
    )
    parser.add_argument(
        "--sumo-home",
        type=str,
        default=None,
        help="SUMO installation root. Otherwise SUMO_HOME is used.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.delta_t <= 0:
        raise ValueError("--delta-t must be greater than 0.")

    project_root = find_project_root()

    input_path = (
        args.input
        if args.input is not None
        else project_root / "data" / "sumo" / "output" / "vehicle_fcd.xml.gz"
    )
    output_path = (
        args.output
        if args.output is not None
        else project_root / "data" / "sumo" / "output" / "vehicle_trace_1s.csv"
    )

    if not input_path.is_absolute():
        input_path = (project_root / input_path).resolve()
    else:
        input_path = input_path.resolve()

    if not output_path.is_absolute():
        output_path = (project_root / output_path).resolve()
    else:
        output_path = output_path.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"FCD input file does not exist: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Use --overwrite if you want to replace it."
        )

    sumo_home = resolve_sumo_home(args.sumo_home)
    trace_exporter = sumo_home / "tools" / "traceExporter.py"

    command = [
        sys.executable,
        str(trace_exporter),
        "--fcd-input",
        str(input_path),
        "--gpsdat-output",
        str(output_path),
        "--delta-t",
        str(args.delta_t),
    ]

    print("Project root :", project_root)
    print("SUMO_HOME    :", sumo_home)
    print("FCD input    :", input_path)
    print("Output       :", output_path)
    print("Delta-t      :", args.delta_t, "s")
    print("\nCalling SUMO official traceExporter.py ...")

    subprocess.run(command, check=True)

    if not output_path.is_file():
        raise RuntimeError(
            "traceExporter.py returned successfully, but the output file "
            f"was not created: {output_path}"
        )

    size_kb = output_path.stat().st_size / 1024.0
    print(f"\nDone. Generated: {output_path}")
    print(f"File size: {size_kb:.1f} KiB")

    # Show a few non-empty lines without trying to reinterpret SUMO's format.
    print("\nFirst lines:")
    shown = 0
    with output_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            print(line.rstrip())
            shown += 1
            if shown >= 5:
                break

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(
            f"\nSUMO traceExporter.py failed with exit code {exc.returncode}.",
            file=sys.stderr,
        )
        raise
