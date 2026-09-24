# Stage2B four actor-learning-rate training scripts

## Scope

Only the four future launch scripts were created.  No launch script was
executed, no rollout was collected, and no checkpoint or existing run was
modified.

```text
SCRIPT_1 = scripts/training/run_stage2b_actor_lr_1e_minus_3.py
ACTOR_LR_1 = 1e-3
RUN_DIR_1 = runs/stage2b_actor_lr_1e_minus_3

SCRIPT_2 = scripts/training/run_stage2b_actor_lr_5e_minus_3.py
ACTOR_LR_2 = 5e-3
RUN_DIR_2 = runs/stage2b_actor_lr_5e_minus_3

SCRIPT_3 = scripts/training/run_stage2b_actor_lr_1e_minus_4.py
ACTOR_LR_3 = 1e-4
RUN_DIR_3 = runs/stage2b_actor_lr_1e_minus_4

SCRIPT_4 = scripts/training/run_stage2b_actor_lr_5e_minus_4.py
ACTOR_LR_4 = 5e-4
RUN_DIR_4 = runs/stage2b_actor_lr_5e_minus_4

CRITIC_LR_ALL = 1e-3
SOURCE_CHECKPOINT = runs/proposed_actor1e4_uavfix_300_epoch/latest_checkpoint.pt
SOURCE_CHECKPOINT_SHA = 6081bc1da279c9081bbc7ec466ca25e9b97902f65fcf1c6c8c8c9ccaced4abbb
SOURCE_COMPLETED_EPOCH = 99
START_EPOCH = 100
TARGET_EPOCH = 300
EPISODE_POOL = environment_data/proposed_joint_training
EPISODE_SEED_RANGE = 20001-20300

CACHE_TRIGGER_CREDIT_ALL = ONE_STEP_TD_Q_DIFFERENCE
CACHE_CANDIDATE_CREDIT_ALL = GLOBAL_GAE
CACHE_RATIO_GRANULARITY_ALL = PER_CACHE_HEAD
UAV_CANDIDATE_KL_GATE_ALL = MEAN_OR_P90, target=0.10
UAV_CONFIDENCE_REFERENCE_SCALE_ALL = 0.02
TDQ_BOOTSTRAP_SIGN_ALL = PLUS
TDQ_NORMALIZATION_ALL = std(D_Q_valid) + 1e-8, no centering

SAME_SOURCE_CHECKPOINT = YES
SAME_TRAIN_DATA_PROTOCOL = YES
SAME_ALGORITHM = YES
ONLY_LR_DIFF = YES
ONLY_INTENDED_DIFFERENCES = actor_lr, run_name, output_directory

PY_COMPILE = PASS
GIT_DIFF_CHECK = PASS
TRAINING_EXECUTED = NO
COMMIT = NO
PUSH = NO
```

## Script behavior

Each entrypoint reads the common epoch-99 source checkpoint and creates an
independent run-local `run_config.toml` and checkpoint bootstrap copy.  The
source file remains read-only; the target copy carries the target actor LR,
current Stage2B credit/ratio metadata, and the target config hash required by
the existing resume guard.  The policy, optimizer moments, cursor, RNG state,
and source realization sequence are otherwise taken from the same source.

The generated run config records the source path and SHA, source epoch 99,
start epoch 100, target epoch 300, the train-only pool and seed range, and the
frozen Stage2B settings.  The script then delegates to the existing
`ppo_simple.trainer` with `--resume`, `--epochs 300`, the same pool and
`--episode-seed-start 20001`; it does not generate missing episodes.  The
trainer's existing metrics and instrumentation are unchanged.

The four scripts are textually identical after replacing only their actor LR,
run name, and output-directory constants.  In particular, the 5e-3 entry does
not alter clipping, update epochs, KL thresholds, optimizer type, or any other
algorithm parameter.

