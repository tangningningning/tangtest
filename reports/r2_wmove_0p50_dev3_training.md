# R2 `w_move=0.50` Dev3 Training Report

**Task:** `R2_WMOVE_0P50_DEV3_TRAINING`  
**Date:** 2026-09-19  
**Repository:** `C:\Users\tangning\Desktop\code\adaptive_cache_mec`

## Outcome

The requested dev3 training command was launched with fresh initialization and the frozen episode pool, but it stopped during the first PPO update because the existing branch-specific KL gate raised a hard exception:

```text
ValueError: branch-specific KL gate cannot transition from both inactive
```

The exception originated in the existing `src/ppo_simple/trainer.py` branch-specific KL transition logic. No source or algorithm change was made to bypass it, and the run was not restarted or overwritten.

```text
DEV3_TRAINING_EXECUTED = YES
DEV3_TRAINING_STATUS = ABORTED_HARD_EXCEPTION
EPOCHS_COMPLETED = 0 / 3
OPERATION_POLICY_RESULT = INCONCLUSIVE
MOVEMENT_SCALE_CORRECTION_EFFECT = INCONCLUSIVE
```

`INCONCLUSIVE` is required because no completed epoch trajectory was logged; this run cannot classify post-training MOVE/STAY collapse.

## Run identity and initialization

```text
NEW_CONFIG = configs/proposed/system_config_ppo_recovery_r2_wmove_0p50_dev3.toml
RUN_DIRECTORY = runs/ppo_recovery_r2_wmove_0p50_dev3_20260919
W_MOVE = 0.50
FRESH_INITIALIZATION = YES
WARM_START = NO
CHECKPOINT_LOADED = NO
RESUME = NO
SEED = 4401
TARGET_EPOCHS = 3
```

The run manifest confirms the requested protocol and hyperparameters: `full_episode_randomized`, 300 slots, origins `0..299`, terminal origin `299`, rollout steps `300`, actor LR `6e-4`, critic LR `3e-4`, `gamma=1.0`, GAE lambda `0.95`, clip epsilon `0.2`, target KL `0.10`, four update epochs, ten minibatches, minibatch size `30`, PopArt critic mode, and entropy coefficient `0.0`.

The frozen realization was loaded from:

```text
runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911/episode_20001
```

No episode, vehicle, task, prediction, or SUMO regeneration was requested or performed.

## Execution point of failure

The trainer successfully initialized the fresh policy and entered:

```text
[PPO] Episode 1/3 | seed=20001 | realization=LOAD
```

It completed the first 300-slot rollout and entered `trainer.update(rollout)`. The process then failed in the existing branch-specific KL gate before the first epoch record/checkpoint was written:

```text
src/ppo_simple/trainer.py:3732
_branch_specific_kl_gate_transition(...)
ValueError: branch-specific KL gate cannot transition from both inactive
```

The failed run directory contains only `run_manifest.json` and `run_summary.json`. It contains no `episode_metrics.jsonl`, `update_metrics.jsonl`, `critic_diagnostics.jsonl`, slot metrics, or checkpoint. The summary remains `status="running"` because the uncaught exception occurred before the trainer's normal finalization path; the process itself has exited with code `1`.

## Required epoch outputs

No completed epoch was logged, so all epoch-level behavior metrics are unavailable:

```text
EPOCH1_P_MOVE = N/A
EPOCH2_P_MOVE = N/A
EPOCH3_P_MOVE = N/A

EPOCH1_UAV_MOVE = N/A
EPOCH2_UAV_MOVE = N/A
EPOCH3_UAV_MOVE = N/A

EPOCH1_RETURN = N/A
EPOCH2_RETURN = N/A
EPOCH3_RETURN = N/A

EPOCH1_UAV_KL = N/A
EPOCH2_UAV_KL = N/A
EPOCH3_UAV_KL = N/A

EPOCH1_CACHE_KL = N/A
EPOCH2_CACHE_KL = N/A
EPOCH3_CACHE_KL = N/A

EPOCH1_CRITIC_EV = N/A
EPOCH2_CRITIC_EV = N/A
EPOCH3_CRITIC_EV = N/A

EPOCH1_CANDIDATE_ENTROPY = N/A
EPOCH2_CANDIDATE_ENTROPY = N/A
EPOCH3_CANDIDATE_ENTROPY = N/A

EPOCH1_CANDIDATE_TOP1_PROB = N/A
EPOCH2_CANDIDATE_TOP1_PROB = N/A
EPOCH3_CANDIDATE_TOP1_PROB = N/A
```

The previous pretraining diagnostic remains a separate result: fresh policy expected MOVE beat STAY was `0/20` at `w=0.50`, while individually good candidates existed in `7/20` states. This training attempt did not rerun that diagnostic or any additional weight sweep.

## Scope and no-change flags

```text
SOURCE_MODIFIED = NO
PRODUCTION_R2_CONFIG_CHANGED = NO
OBJECTIVE_TRADEOFF_WEIGHT_CHANGED_IN_DEV_CONFIG = YES

SYSTEM_MODEL_FORMULA_CHANGED = NO
REWARD_FORMULA_CHANGED = NO
PHYSICAL_MOVEMENT_PARAMETERS_CHANGED = NO

CACHE_CREDIT_CHANGED = NO
CANDIDATE_CREDIT_CHANGED = NO
OPERATION_CREDIT_CHANGED = NO
PPO_CHANGED = NO

EPISODE_REGENERATION = NO
SUMO_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
FORMAL_EVALUATION_EXECUTED = NO

TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO
```

The existing `runs/ppo_recovery_r2_dev20_20260918` directory was not modified. No attempt was made to alter the KL gate, change PPO behavior, or restart the failed dev3 run.
