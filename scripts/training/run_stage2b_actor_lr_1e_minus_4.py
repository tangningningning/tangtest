#!/usr/bin/env python3
"""Fresh Stage2B actor-LR run: actor LR 1e-4."""

from pathlib import Path

from stage2b_fresh300_runner_common import FreshRunnerSpec, run_fresh300


RUN_SPEC = FreshRunnerSpec(
    run_name="stage2b_fresh300_actor_lr_1e_minus_4",
    run_dir_relative=Path("runs/stage2b_fresh300_actor_lr_1e_minus_4"),
    lr_label="1e-4",
    target_actor_lr=1e-4,
)


if __name__ == "__main__":
    raise SystemExit(run_fresh300(RUN_SPEC))
