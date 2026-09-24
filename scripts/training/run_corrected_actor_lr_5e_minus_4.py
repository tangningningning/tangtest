#!/usr/bin/env python3
"""Corrected architecture-v2 fresh 300-epoch run at actor LR 5e-4."""

from pathlib import Path

from stage2b_fresh300_runner_common import FreshRunnerSpec, run_fresh300


RUN_SPEC = FreshRunnerSpec(
    run_name="stage2c_corrected_fresh300_actor_lr_5e_minus_4",
    run_dir_relative=Path("runs/stage2c_corrected_fresh300_actor_lr_5e_minus_4"),
    lr_label="5e-4",
    target_actor_lr=5e-4,
)


if __name__ == "__main__":
    raise SystemExit(run_fresh300(RUN_SPEC))
