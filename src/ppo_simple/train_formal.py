"""Canonical training entry point for the current formal PROPOSED policy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .trainer import run_training


FORMAL_TRAINING_EPISODES = 300
FORMAL_RUN_DIRECTORY = "runs/formal_proposed_cache_economic_features_horizon_mask_v2"
FORMAL_TRAINING_EPISODE_POOL = "environment_data/proposed_joint_training"
FORMAL_TRAINING_EPISODE_SEED_START = 20001


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the current 200-D fixed-learning-rate formal PROPOSED PPO."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def run_formal_training(args: Any) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "proposed" / "system_config.toml"
    run_directory = project_root / FORMAL_RUN_DIRECTORY
    return run_training(
        config_path,
        FORMAL_TRAINING_EPISODES,
        run_directory,
        episode_pool=FORMAL_TRAINING_EPISODE_POOL,
        episode_seed_start=FORMAL_TRAINING_EPISODE_SEED_START,
        generate_missing_episodes=False,
        seed=int(args.seed),
        resume=bool(args.resume),
        verbose=bool(args.verbose),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_formal_training(args)
    return 2 if result["early_stop_reason"] is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
