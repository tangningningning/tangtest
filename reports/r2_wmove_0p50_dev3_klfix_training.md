# R2 `w_move=0.50` Dev3 Training After KL Fix

**Task:** `R2_WMOVE_0P50_DEV3_RERUN_AFTER_KL_FIX`  
**Date:** 2026-09-19  
**Repository:** `C:\Users\tangning\Desktop\code\adaptive_cache_mec`

## Outcome

The requested fresh three-epoch development training completed normally after the branch-specific KL fix. All three complete 300-slot realizations ran, all four PPO update epochs ran inside each training epoch, checkpoints and normal logs were written, and the process exited with code `0`.

```text
DEV3_TRAINING_EXECUTED = YES
DEV3_COMPLETED_EPOCHS = 3/3
RUN_DIRECTORY = runs/ppo_recovery_r2_wmove_0p50_dev3_klfix_20260919
FRESH_INITIALIZATION = YES
WARM_START = NO
CHECKPOINT_LOADED = NO
W_MOVE = 0.50
KL_GATE_EXCEPTION = NO
```

The operation policy still moves rapidly toward STAY:

```text
P(MOVE): 0.5194006 -> 0.1886098 -> 0.0348751
```

Therefore:

```text
OPERATION_POLICY_RESULT = STAY_COLLAPSE_PERSISTS
MOVEMENT_SCALE_CORRECTION_EFFECT = NOT_SUPPORTED
```

This classification uses the policy's mean `P(MOVE)`, not raw return. The result is also consistent with the historical `w_move=10.0` reference (`0.5194 -> 0.1860 -> 0.0367`); lowering the movement weight to `0.50` did not materially change the rapid STAY-collapse pattern in this three-epoch run.

## Protocol and preservation

| item | value |
|---|---|
| config | `configs/proposed/system_config_ppo_recovery_r2_wmove_0p50_dev3.toml` |
| seed | `4401` |
| initialization | fresh actor, critic, actor optimizer, and critic optimizer |
| frozen pool | `runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911` |
| realizations | `episode_20001`, `episode_20002`, `episode_20003` |
| protocol | `full_episode_randomized` |
| slots per training epoch | `300` |
| origins | `0..299` |
| terminal origin | `299` |
| rollout steps | `300` |
| actor LR | `6e-4` |
| critic LR | `3e-4` |
| gamma / GAE lambda | `1.0 / 0.95` |
| clip epsilon | `0.2` |
| target KL | `0.10` |
| PPO update epochs | `4` |
| minibatches / minibatch size | `10 / 30` |
| entropy coefficient | `0.0` |
| PopArt | unchanged |

The old failed run `runs/ppo_recovery_r2_wmove_0p50_dev3_20260919` and `runs/ppo_recovery_r2_dev20_20260918` were preserved. No checkpoint was loaded and no resume or warm start was used.

## Training-epoch metrics

`P(MOVE)` is the logger's mean policy probability. `UAVMove` is the observed actual movement rate. KL values below are the branch-specific epoch KLs; the joint KL is included separately for context.

| epoch | realization | Return | P(MOVE) | P(STAY) | UAVMove | operation entropy | Cache KL | UAV KL | joint KL | critic loss | critic EV | actor steps | critic steps | rollout s | update s | total epoch s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `episode_20001` | -975.879022 | 0.5194006 | 0.4805994 | 0.5750000 | 0.6901013 | 0.0658669 | 0.6473549 | 0.6260195 | 0.3741671 | -0.0000355 | 20 | 40 | 57.224 | 609.110 | 666.526 |
| 2 | `episode_20002` | -4191.324870 | 0.1886098 | 0.8113902 | 0.1900000 | 0.4800026 | 0.0493983 | 0.3236371 | 0.1885145 | 0.5383444 | 0.0037866 | 40 | 40 | 61.006 | 1192.653 | 1253.839 |
| 3 | `episode_20003` | -1944.579260 | 0.0348751 | 0.9651249 | 0.0383333 | 0.1502795 | 0.0176643 | 0.0702324 | -0.0020997 | 0.2359128 | 0.0167917 | 40 | 40 | 64.073 | 1662.436 | 1726.690 |

The critic loss is the logged normalized `value_loss`; raw value losses were `1246.703491`, `22585.906250`, and `10743.186523` respectively. Cumulative optimizer counts at the three checkpoints were actor `20 / 60 / 100` and critic `40 / 80 / 120`.

The existing logger did not persist a separate counterfactual-only runtime; the `update s` column is the complete trainer update runtime, including the R2 operation-credit work. No logger changes were made to add a new counterfactual timer.

## Branch-specific KL state paths

| epoch | PPO state path | Cache stop PPO epoch | UAV stop PPO epoch | Cache independent stop | UAV independent stop | both inactive reached | actor steps | critic steps |
|---:|---|---:|---:|:---:|:---:|:---:|---:|---:|
| 1 | `TT -> TF -> FF -> FF` | 2 | 1 | YES | YES | YES | 20 | 40 |
| 2 | `TT -> TF -> TF -> TF` | N/A | 1 | NO | YES | NO | 40 | 40 |
| 3 | `TT -> TT -> TT -> TT` | N/A | N/A | NO | NO | NO | 40 | 40 |

The `FF` state in epoch 1 was handled as the legal absorbing state. The actor optimizer was skipped after both branches became inactive, while the critic continued through all four PPO update epochs. There was no branch-specific KL exception and no whole-training-epoch early termination.

## KL gate assessment

```text
UAV_ACTOR_UPDATE_BUDGET_STRONGLY_KL_LIMITED = NO
KL_GATE_UPDATE_PRESSURE = MODERATE
```

UAV stopped at PPO update epoch 1 in training epochs 1 and 2, but remained active through all four PPO epochs in training epoch 3. Cache stopped only in epoch 1, at PPO epoch 2. Thus the first two epochs show substantial branch-specific KL pressure, but the pressure is not persistent across all three training epochs and did not uniformly limit the actor budget.

## Candidate and KL decomposition scope

The existing logger exposed operation entropy and mean candidate count, but not candidate entropy or candidate top-1 probability. No production logger was changed.

```text
CANDIDATE_ENTROPY = N/A
CANDIDATE_TOP1_PROBABILITY = N/A
KL_COMPONENT_DECOMPOSITION_AVAILABLE = NO
```

No operation-KL/candidate-KL split was added.

## Finite checks and required keys

All three episode records reported finite reward, return, `G`, loss, gradient, returns, and advantages. The run summary reported `nan_inf_detected=false`, and all three checkpoints were written normally.

```text
EPOCH1_P_MOVE = 0.5194006145
EPOCH2_P_MOVE = 0.1886097877
EPOCH3_P_MOVE = 0.0348750925

EPOCH1_UAV_MOVE = 0.5750000000
EPOCH2_UAV_MOVE = 0.1900000000
EPOCH3_UAV_MOVE = 0.0383333333

EPOCH1_RETURN = -975.8790221190
EPOCH2_RETURN = -4191.3248702742
EPOCH3_RETURN = -1944.5792604096

EPOCH1_UAV_KL = 0.6473548697
EPOCH2_UAV_KL = 0.3236371448
EPOCH3_UAV_KL = 0.0702323625

EPOCH1_CACHE_KL = 0.0658668872
EPOCH2_CACHE_KL = 0.0493982623
EPOCH3_CACHE_KL = 0.0176642693

EPOCH1_CRITIC_EV = -0.0000355070
EPOCH2_CRITIC_EV = 0.0037866009
EPOCH3_CRITIC_EV = 0.0167916539

EPOCH1_PPO_STATE_PATH = TT -> TF -> FF -> FF
EPOCH2_PPO_STATE_PATH = TT -> TF -> TF -> TF
EPOCH3_PPO_STATE_PATH = TT -> TT -> TT -> TT

EPOCH1_ACTOR_OPTIMIZER_STEPS = 20
EPOCH2_ACTOR_OPTIMIZER_STEPS = 40
EPOCH3_ACTOR_OPTIMIZER_STEPS = 40

EPOCH1_CRITIC_OPTIMIZER_STEPS = 40
EPOCH2_CRITIC_OPTIMIZER_STEPS = 40
EPOCH3_CRITIC_OPTIMIZER_STEPS = 40
```

## No-change and cleanup flags

```text
SOURCE_FIX_PRESERVED = YES
TARGET_KL_CHANGED = NO
ACTOR_LR_CHANGED = NO
CRITIC_LR_CHANGED = NO
CACHE_CREDIT_CHANGED = NO
CANDIDATE_CREDIT_CHANGED = NO
OPERATION_CREDIT_CHANGED = NO
REWARD_FORMULA_CHANGED = NO
PHYSICAL_MOVEMENT_PARAMETERS_CHANGED = NO
EPISODE_REGENERATION = NO
SUMO_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
FORMAL_EVALUATION_EXECUTED = NO
TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO
```
