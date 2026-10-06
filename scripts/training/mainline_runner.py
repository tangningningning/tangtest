"""Shared launcher for current V49 Macro Actor learning-rate runs."""

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
EPISODE_COUNT = 15
EPISODE_SEED_START = 20001
TRAINING_SEED = 27005


def main_for_run_spec(spec: Mapping[str, Any]) -> int:
    required = {
        "macro_actor_learning_rate",
        "run_name",
    }
    if set(spec) != required:
        raise ValueError(f"V49 LR launcher spec must contain exactly {sorted(required)}")
    rate = float(spec["macro_actor_learning_rate"])
    if rate <= 0.0:
        raise ValueError("macro_actor_learning_rate must be positive")
    run_name = str(spec["run_name"])
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be one path component")

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
            str(rate),
            "--seed",
            str(TRAINING_SEED),
        ]
    )
