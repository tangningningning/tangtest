"""Shared launcher for PROPOSED Macro Actor learning-rate runs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
CONFIG_PATH = "configs/proposed/system_config.toml"
EPISODE_POOL = "environment_data/proposed_joint_training"
EPISODE_COUNT = 300
EPISODE_SEED_START = 20001
TRAINING_SEED = 27005
SUPPORTED_MACRO_ACTOR_LEARNING_RATES = (1.0e-4, 5.0e-4, 1.0e-3, 5.0e-3)


def main_for_run_spec(spec: Mapping[str, Any]) -> int:
    required = {
        "macro_actor_learning_rate",
        "run_name",
    }
    if set(spec) not in (required, required | {"ppo_seed"}):
        raise ValueError(
            "PROPOSED LR launcher spec must contain exactly "
            f"{sorted(required)} or {sorted(required | {'ppo_seed'})}"
        )
    rate = float(spec["macro_actor_learning_rate"])
    supported_rate = next(
        (
            value
            for value in SUPPORTED_MACRO_ACTOR_LEARNING_RATES
            if abs(rate - value) <= 1.0e-15
        ),
        None,
    )
    if supported_rate is None:
        raise ValueError(
            "macro_actor_learning_rate must be one of the four supported fixed rates"
        )
    run_name = str(spec["run_name"])
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be one path component")
    run_dir = PROJECT_ROOT / "runs" / run_name
    if run_dir.exists():
        raise FileExistsError(
            f"experiment output already exists; refusing to overwrite: {run_dir}"
        )
    ppo_seed = int(spec.get("ppo_seed", TRAINING_SEED))

    from ppo_simple.trainer import main as trainer_main

    return trainer_main(
        [
            "--config",
            CONFIG_PATH,
            "--epochs",
            str(EPISODE_COUNT),
            "--run-dir",
            f"runs/{run_name}",
            "--episode-pool",
            EPISODE_POOL,
            "--episode-seed-start",
            str(EPISODE_SEED_START),
            "--macro-actor-learning-rate",
            str(supported_rate),
            "--seed",
            str(ppo_seed),
        ]
    )
