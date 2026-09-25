#!/usr/bin/env python3
"""Independent Stage2D V3 fresh 300-epoch runner at actor LR 5e-4."""

from stage2d_v3_runner_common import Stage2DV3Spec, run_stage2d_v3


RUN_NAME = "stage2d_v3_fresh300_actor_lr_5e_minus_4"
ACTOR_LR = 5.0e-4


if __name__ == "__main__":
    raise SystemExit(run_stage2d_v3(Stage2DV3Spec(RUN_NAME, ACTOR_LR)))
