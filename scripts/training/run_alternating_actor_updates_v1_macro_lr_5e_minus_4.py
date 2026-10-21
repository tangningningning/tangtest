"""Run Alternating Actor Optimization V1 with a fixed Macro Actor learning rate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = PROJECT_ROOT / "scripts" / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from mainline_runner import main as mainline_main


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed 300-Episode experiment, optionally resuming its Run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--verbose", action="store_true")
    options = parser.parse_args(argv)

    runner_args = [
        "--config",
        str(
            PROJECT_ROOT / "configs" / "proposed" / "alternating_actor_updates_v1.toml"
        ),
        "--epochs",
        "300",
        "--run-dir",
        str(PROJECT_ROOT / "runs" / "alternating_actor_updates_v1_macro_lr_5e-4"),
        "--episode-pool",
        str(PROJECT_ROOT / "environment_data" / "proposed_joint_training"),
        "--episode-seed-start",
        "20001",
        "--seed",
        "4401",
        "--macro-actor-learning-rate",
        "5e-4",
    ]
    if options.resume:
        runner_args.append("--resume")
    if options.resume_checkpoint is not None:
        runner_args.extend(("--resume-checkpoint", options.resume_checkpoint))
    if options.verbose:
        runner_args.append("--verbose")
    return mainline_main(runner_args)


if __name__ == "__main__":
    raise SystemExit(main())
