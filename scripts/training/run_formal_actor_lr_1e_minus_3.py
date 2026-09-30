"""Thin entrypoint for the formal 300-epoch Macro Actor LR=1e-3 run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

CONFIG_PATH = Path("configs/proposed/system_config.toml")
TARGET_EPOCHS = 300
RUN_DIR = Path("runs/formal_actor_lr_1e_minus_3_300")
EPISODE_POOL = Path("environment_data/proposed_joint_training")
INITIALIZATION_SEED = 27005
EPISODE_SEED_START = 20001
ACTOR_LEARNING_RATE_OVERRIDE = 1.0e-3


def build_run_arguments(*, verbose: bool = False) -> dict[str, Any]:
    """Return the fixed formal run settings without creating artifacts."""

    return {
        "config_path": PROJECT_ROOT / CONFIG_PATH,
        "target_episodes": TARGET_EPOCHS,
        "run_dir": PROJECT_ROOT / RUN_DIR,
        "episode_pool": PROJECT_ROOT / EPISODE_POOL,
        "episode_seed_start": EPISODE_SEED_START,
        "generate_missing_episodes": False,
        "episode_generation_device": "cpu",
        "macro_actor_learning_rate_override": ACTOR_LEARNING_RATE_OVERRIDE,
        "critic_learning_rate_override": None,
        "seed": INITIALIZATION_SEED,
        "resume": False,
        "verbose": bool(verbose),
        "save_macro_transition_trace": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    from ppo_simple.trainer import run_training

    run_training(**build_run_arguments(verbose=args.verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
