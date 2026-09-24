#!/usr/bin/env python3
"""Read-only completeness check for the formal validation and test pools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from common import project_config
from generation.generate_episode_realization import validate_written_outputs
from ppo_simple.formal_evaluation import load_split_manifest
from ppo_simple.trainer import (
    _realization_artifacts_from_directory,
    aligned_train_steps,
    load_environment_context,
)


def validate_split(split: str, *, smoke_reset: bool = True) -> dict:
    manifest = load_split_manifest()
    config_path = PROJECT_ROOT / manifest["generator_config_path"]
    config = project_config.load_config(config_path)
    get = project_config.get_config_value
    contract = manifest[split]
    directory = PROJECT_ROOT / "environment_data" / contract["directory"]
    expected = set(range(contract["seed_start"], contract["seed_end"] + 1))
    found = {int(path.name.removeprefix("episode_")) for path in directory.iterdir()
             if path.is_dir() and path.name.startswith("episode_") and path.name[8:].isdigit()}
    if found != expected:
        raise ValueError(f"{split} seed directory mismatch: missing={sorted(expected-found)}, extra={sorted(found-expected)}")
    for seed in sorted(expected):
        episode = directory / f"episode_{seed}"
        _realization_artifacts_from_directory(episode)
        episode_manifest_path = episode / "episode_manifest.json"
        if not episode_manifest_path.is_file():
            raise FileNotFoundError(f"{episode}: missing episode_manifest.json")
        provenance = json.loads(episode_manifest_path.read_text(encoding="utf-8"))
        if provenance.get("episode_seed") != seed:
            raise ValueError(f"{episode}: episode seed mismatch")
        if provenance.get("generation_config_fingerprint") != manifest["generation_config_fingerprint"]:
            raise ValueError(f"{episode}: generator fingerprint mismatch")
        for key in ("service_catalog_sha256", "popularity_gru_sha256",
                    "vehicle_residual_bank_sha256", "sumo_network_sha256"):
            if provenance.get(key) != manifest[key]:
                raise ValueError(f"{episode}: {key} mismatch")
        if provenance.get("simulation", {}).get("episode_slots") != 300:
            raise ValueError(f"{episode}: slot count mismatch")
        validate_written_outputs(
            episode,
            task_vehicle_count=int(get(config, "simulation.num_task_vehicles")),
            research_vehicle_count=int(get(config, "simulation.num_research_vehicles")),
            horizon_steps=int(get(config, "prediction.vehicle.horizon")),
        )
        with np.load(episode / "prediction_scenarios.npz", allow_pickle=False) as scenario:
            for name in scenario.files:
                values = scenario[name]
                if np.issubdtype(values.dtype, np.number) and not np.all(np.isfinite(values)):
                    raise FloatingPointError(f"{episode}: {name} contains NaN/Inf")
        context = load_environment_context(
            str(config_path), episode
        )
        if context.episode_slots != 300 or context.train_origin_start != 0 or context.train_origin_end != 299:
            raise ValueError(f"{episode}: environment loader slot contract mismatch")
        if smoke_reset and seed == contract["seed_start"]:
            steps = aligned_train_steps(context)
            if len(steps) != 300:
                raise ValueError(f"{episode}: environment reset/alignment smoke failed")
            if not np.all(np.isfinite(context.initial_cache)):
                raise FloatingPointError(f"{episode}: non-finite initial cache")
            if any(not np.all(np.isfinite(position)) for position in context.initial_uav.values()):
                raise FloatingPointError(f"{episode}: non-finite initial UAV position")
    return {"split": split, "count": len(expected), "slots_per_realization": 300,
            "loader_smoke_seed": contract["seed_start"] if smoke_reset else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test", "both"), default="both")
    args = parser.parse_args(argv)
    splits = ("validation", "test") if args.split == "both" else (args.split,)
    for split in splits:
        result = validate_split(split)
        print(f"[DATA] {result['split']} complete={result['count']} slots=300 "
              f"loader_smoke_seed={result['loader_smoke_seed']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
