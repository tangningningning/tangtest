# V10-C Partial Diagnostic Run Plan

STAGE = V10-C_PARTIAL_DIAGNOSTIC_RUN_FINALIZATION

## Execution boundary

This task fixes two independent Linux launchers and their run contract. The
launchers were not started from Codex: no PPO training, evaluation, resume,
final-seed test, or episode generation was executed.

## Diagnostic runs

| run | config | actor LR | epochs | isolated run directory |
|---|---|---:|---:|---|
| 5e-4 | `configs/proposed/system_config_ppo_v10_branchwise_cache_uav.toml` | 0.0005 | 1--240 | `runs/ppo_v10c_instrumented_5e4_ep240` |
| 6e-4 | `configs/lr_sweep/v10_branchwise_actor_lr_6e4_kl.toml` | 0.0006 | 1--240 | `runs/ppo_v10c_instrumented_6e4_ep240` |

Both runs keep critic LR `0.0003`, target KL `0.10`, clip epsilon `0.2`, four
update epochs, ten minibatches of size `30`, max grad norm `0.5`, entropy
coefficient `0`, gamma `1.0`, GAE lambda `0.95`, seed `4401`, and the existing
PopArt normalization unchanged. The V10-C action-conditional instrumentation
in `src/ppo_simple/trainer.py` is unchanged.

## Frozen realization protocol

Both launchers use the read-only pool
`runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911` with
`--episode-seed-start 20001`. The actual 240-epoch diagnostic run consumes
exactly:

`epoch 1 -> episode_20001` through `epoch 240 -> episode_20240`.

`episode_20241` through `episode_20300` remain unused by these launchers. The
pool itself remains complete and is not modified.

## Why not reuse the available best checkpoints

The checkpoint reuse audit found only best and latest files. The 5e-4 best
checkpoint is at epoch 253 and the 6e-4 best checkpoint at epoch 245; both are
after the historical UAV polarization transitions. The historical MA20
transition was classified at epoch 215 for 5e-4 and epoch 212 for 6e-4. A
short continuation from either best checkpoint would therefore capture only
post-polarization behavior and could not provide the pre/onset/post
action-conditional comparison.

The requested MOVE/STAY advantage, reward, q-total, movement-cost, ratio, and
clip fields are produced in the PPO rollout/update path. Pure inference from
an existing checkpoint cannot reproduce the historical old/new log-probability,
ratio, clip flag, and exact normalized `actual_PPO_advantage` records. A fresh
instrumented partial run is therefore required for the complete diagnostic
question.

## Why stop at epoch 240

Epoch 240 includes the pre-transition period and at least the first post-
transition window for both runs. It avoids rerunning the remaining epochs
after the root-cause transition and saves 60 epochs per run, 120 epochs total,
relative to two 300-epoch diagnostic runs. No 1e-3 run or final seeds
`40001..40020` are included.

## Launcher safeguards

Both scripts use Linux bash with `set -euo pipefail`, derive the repository
root from the script location, export `PYTHONPATH=<repo>/src`, and check
Python, PyTorch, `ppo_simple`, the requested configuration, and the frozen
pool. They refuse a non-empty target run directory and invoke the trainer with
240 epochs, the frozen pool, and the fixed seed mapping. They contain neither
`--resume` nor `--generate-missing-episodes`.

The two target directories are independent, so they may be submitted as
separate jobs on separate nodes without sharing a writable run directory.

## Invariants

ACTION_CHANGED = NO

RNG_CHANGED = NO

REWARD_CHANGED = NO

PPO_OBJECTIVE_CHANGED = NO

CRITIC_CHANGED = NO

SYSTEM_MODEL_CHANGED = NO

TRAINING_EXECUTED = NO

EVALUATION_EXECUTED = NO

FINAL_TEST_SEEDS_USED = NO

## Final status

STAGE = V10-C_PARTIAL_DIAGNOSTIC_RUN_FINALIZATION

INSTRUMENTATION_READY = YES

RUN_5E4 = READY, epoch 1..240

RUN_6E4 = READY, epoch 1..240

CONFIG_5E4 = C:\Users\tangning\Desktop\code\adaptive_cache_mec\configs\proposed\system_config_ppo_v10_branchwise_cache_uav.toml

CONFIG_6E4 = C:\Users\tangning\Desktop\code\adaptive_cache_mec\configs\lr_sweep\v10_branchwise_actor_lr_6e4_kl.toml

ACTOR_LR_5E4 = 0.0005

ACTOR_LR_6E4 = 0.0006

CRITIC_LR_ALL = 0.0003

TARGET_KL_ALL = 0.10

EPISODE_POOL = C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v6_popart_actor1e4_lazy_episode_pool_20260911

FIRST_EPISODE = episode_20001

LAST_EPISODE_USED = episode_20240

SEED_MAPPING_VALID = YES

SCRIPT_5E4 = C:\Users\tangning\Desktop\code\adaptive_cache_mec\scripts\run_v10c_instrumented_5e4.sh

SCRIPT_6E4 = C:\Users\tangning\Desktop\code\adaptive_cache_mec\scripts\run_v10c_instrumented_6e4.sh

TEMP_FILES_REMOVED = YES

GIT_DIFF_CHECK = PASS
