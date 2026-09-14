#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a recoverable pool of complete randomized PPO episodes.

This is intentionally only a batch loop around the existing single-episode
generator.  It does not duplicate SUMO, task, prediction, or scenario
generation logic, and it never writes the shared ``data/generated`` assets.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from generation import generate_episode_realization as realization_generator
from common import project_config
from ppo_simple.trainer import (
    REALIZATION_ARTIFACT_FILENAMES,
    _realization_artifacts_from_directory,
)


# The trainer contract covers the nine files below.  The single-episode
# generator also produces these SUMO provenance files, so an existing episode
# is not considered complete unless those files are present as well.
ADDITIONAL_GENERATOR_ARTIFACTS = (
    "sumo_demand.rou.xml",
    "sumo_demand.meta.json",
)
DEVELOPMENT_SEEDS = frozenset({1001, 1002})


def _required_artifact_names() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *REALIZATION_ARTIFACT_FILENAMES.values(),
                *ADDITIONAL_GENERATOR_ARTIFACTS,
            )
        )
    )


def _missing_artifacts(directory: Path) -> list[str]:
    return [
        name
        for name in _required_artifact_names()
        if not (directory / name).is_file()
    ]


def _validate_complete_realization(
    directory: Path,
    *,
    config: dict[str, Any],
) -> None:
    missing = _missing_artifacts(directory)
    if missing:
        raise RuntimeError(
            "incomplete episode realization; remove it before retrying: "
            f"directory={directory}, missing={missing}"
        )
    # Use the same path contract as the trainer, then run the existing
    # generator's lightweight 300-slot content validation before skipping.
    _realization_artifacts_from_directory(directory)
    get = project_config.get_config_value
    realization_generator.validate_written_outputs(
        directory,
        task_vehicle_count=int(get(config, "simulation.num_task_vehicles")),
        research_vehicle_count=int(get(config, "simulation.num_research_vehicles")),
        horizon_steps=int(get(config, "prediction.vehicle.horizon")),
    )


def _episode_directory(pool_dir: Path, seed: int) -> Path:
    return pool_dir / f"episode_{int(seed)}"


def _progress_line(
    *,
    index: int,
    count: int,
    seed: int,
    runtime: float,
    started_at: float,
    skipped: bool,
) -> str:
    completed = index + 1
    elapsed = time.perf_counter() - started_at
    remaining = count - completed
    estimated_remaining = (
        (elapsed / float(completed)) * float(remaining)
        if completed
        else 0.0
    )
    status = " SKIP existing valid realization" if skipped else ""
    return (
        f"[EpisodePool] {completed}/{count} seed={seed}{status} "
        f"runtime={runtime:.3f}s completed={completed} remaining={remaining} "
        f"elapsed={elapsed:.3f}s "
        f"estimated_remaining={estimated_remaining:.3f}s"
    )


def generate_pool(
    config_file: Path,
    pool_dir: Path,
    start_seed: int,
    count: int,
    sumo_home: str | None,
    device: str,
) -> None:
    if int(count) <= 0:
        raise ValueError("count must be greater than zero")
    start_seed_i = int(start_seed)
    count_i = int(count)
    seeds = tuple(start_seed_i + offset for offset in range(count_i))
    development_overlap = sorted(DEVELOPMENT_SEEDS.intersection(seeds))
    if development_overlap:
        raise ValueError(
            "development seeds must not be placed in a training episode pool: "
            f"{development_overlap}"
        )

    config = realization_generator.load_and_validate_config(config_file)
    output_dir = realization_generator.prepare_output_dir(pool_dir)
    started_at = time.perf_counter()
    for index, seed in enumerate(seeds):
        destination = _episode_directory(output_dir, seed)
        if destination.exists():
            if not destination.is_dir():
                raise RuntimeError(
                    "episode realization path exists but is not a directory: "
                    f"{destination}"
                )
            validation_started_at = time.perf_counter()
            _validate_complete_realization(destination, config=config)
            validation_runtime = time.perf_counter() - validation_started_at
            print(
                _progress_line(
                    index=index,
                    count=count_i,
                    seed=seed,
                    runtime=validation_runtime,
                    started_at=started_at,
                    skipped=True,
                ),
                flush=True,
            )
            continue

        destination.mkdir(parents=True, exist_ok=False)
        episode_started_at = time.perf_counter()
        try:
            realization_generator.run_episode(
                config_file,
                seed,
                destination,
                sumo_home,
                device,
            )
            _validate_complete_realization(destination, config=config)
        except Exception as exc:
            raise RuntimeError(
                "episode realization generation failed; remove the partial "
                f"directory before retrying: seed={seed}, directory={destination}"
            ) from exc
        runtime = time.perf_counter() - episode_started_at
        print(
            _progress_line(
                index=index,
                count=count_i,
                seed=seed,
                runtime=runtime,
                started_at=started_at,
                skipped=False,
            ),
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a recoverable pool of randomized PPO episode realizations."
    )
    parser.add_argument(
        "--config",
        default="configs/episode_generation/system_config_ppo_v6_full_episode_randomized.toml",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--sumo-home", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_file = realization_generator.resolve_project_path(args.config)
        if not config_file.is_file():
            raise FileNotFoundError(f"config is not readable: {config_file}")
        pool_dir = realization_generator.resolve_project_path(args.output_dir)
        generate_pool(
            config_file,
            pool_dir,
            args.start_seed,
            args.count,
            args.sumo_home,
            args.device,
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
