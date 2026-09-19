# R2 `w_move=0.50` Short Development Gate

**Task:** `R2_WMOVE_0P50_SHORT_DEVELOPMENT_GATE`  
**Date:** 2026-09-19  
**Repository:** `C:\Users\tangning\Desktop\code\adaptive_cache_mec`

## Decision

```text
PRIMARY_W_MOVE = 0.50
PRIMARY_PRETRAIN_GATE = FAIL
DEV3_TRAINING_EXECUTED = NO
```

The reconstruction check passed exactly, but the fresh policy's policy-conditional expected MOVE value did not exceed STAY for any of the 20 requested `slot × UAV` states at `w_move=0.50`:

```text
W050_EXPECTED_MOVE_BEATS_STAY_COUNT = 0
W050_EXPECTED_MOVE_LOSES_STAY_COUNT = 20
```

Per the task gate, the process stops here. No 3-epoch training is started, no sensitivity training is run, and no existing R2 run is modified.

## Configuration and preservation checks

New development configuration:

```text
configs/proposed/system_config_ppo_recovery_r2_wmove_0p50_dev3.toml
```

The source configuration `configs/proposed/system_config_ppo_recovery_r2_6e4.toml` was not overwritten. A line-level comparison against the current source showed only the permitted development differences:

```text
experiment.name       = recovery_r2_wmove_0p50_dev3
experiment.output_tag = recovery_r2_wmove_0p50_dev3_20260919
objective.uav_movement_cost_weight: 10.0 -> 0.50
```

The following remained unchanged: `eta_q`, `lambda_movement`, `xi_mov`, the reward/system-model formula, cache cost, Matching/KKT, candidate geometry, actor LR `6e-4`, critic LR `3e-4`, `gamma=1.0`, GAE lambda `0.95`, clip epsilon `0.2`, target KL `0.10`, update epochs `4`, minibatches `10`, minibatch size `30`, PopArt mode, entropy coefficient, and seed `4401`.

The historical run `runs/ppo_recovery_r2_dev20_20260918` was not touched. The requested dev3 run directory was not created because the pretraining gate failed.

## Gate protocol

The gate used:

| Item | Value |
|---|---|
| initialization | fresh actor, fresh critic, fresh optimizer |
| checkpoint | none loaded |
| warm start/resume | no |
| seed | 4401 |
| frozen realization | `runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911/episode_20001` |
| protocol | `full_episode_randomized` |
| episode slots | 300 |
| train origins | 0..299 |
| terminal origin | 299 |
| exact target slots | 20, 50, 80, 110, 140, 170, 200, 230, 260, 290 |
| target UAV heads | `UAV_0`, `UAV_1` |
| target state count | 20 |
| exact branch execution | STAY plus every legal MOVE candidate for each target state |
| candidate pruning/sampling/top-k/surrogate | not used for exact branches |
| optimizer steps after gate | 0 |

A single ordinary fresh-policy rollout was used to establish the state path. Exact formal branch replay was restricted to the requested 20 target states. Every exact branch used `execute_slot`, the production configuration override, Matching/KKT, the production reward identity, and the same fresh critic continuation. No episode, vehicle, task, prediction, or SUMO regeneration was performed.

## Reconstruction and execution validation

```text
OFFLINE_VS_DIRECT_Q_MAX_ABS_DIFF = 0.0
FORMAL_BRANCH_EXECUTION_NORMAL   = YES
TOTAL_LEGAL_MOVE_CANDIDATES      = 1587
EXACT_ENVIRONMENT_EXECUTIONS      = 1607
```

The 1,607 exact executions are 20 STAY branches plus 1,587 legal MOVE branches. The offline values for `w=0.25`, `w=0.50`, and `w=1.00` use the same branch next states, the same fresh critic continuation, the same operation probabilities, and:

```text
DeltaR(w) = PRE_COST_BENEFIT - w * BASE_MOVE_TERM
```

The direct-versus-offline check at the actual development configuration is therefore a pass with a zero measured maximum absolute residual.

## Pretraining gate results

Counts are over the 20 `slot × UAV` states. `EXPECTED_MOVE_BEATS_STAY_COUNT` uses the policy-conditional candidate expectation. `BEST_MOVE_BEATS_STAY_COUNT` asks whether at least one legal MOVE candidate beats STAY, independent of the candidate policy probability. `GOOD_CANDIDATE_STATE_COUNT` is the count of states with at least one positive candidate delta.

| weight | expected MOVE beats STAY | expected MOVE loses STAY | mean expected ΔQ | median expected ΔQ | best MOVE beats STAY | good-candidate states |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 2 | 18 | -0.000628767889 | -0.001005038526 | 13 | 13 |
| 0.50 | 0 | 20 | -0.002006048298 | -0.002012163111 | 7 | 7 |
| 1.00 | 0 | 20 | -0.004760609116 | -0.004040857505 | 3 | 3 |

The fresh operation probabilities themselves were not saturated: the observed `p_move` values across the 20 states were approximately `0.4776` to `0.5731`. The failure is therefore not evidence of a post-training probability collapse. It means the required policy-conditional expected-value gate is not met before training at the primary weight.

For `w=0.50`, seven states contain at least one individually positive MOVE candidate, but the fresh candidate policy assigns enough probability to the other candidates that the policy-conditional expected MOVE value remains below STAY in all 20 states. The largest and smallest policy-conditional expected deltas were approximately `-0.001746325018` and `-0.002051862050`; both are negative.

## Interpretation

This gate does not justify changing the candidate policy, learning rates, KL rules, reward, or movement formula. It also does not justify lowering `w_move` within this task: the instructions explicitly require stopping when `W050_EXPECTED_MOVE_BEATS_STAY_COUNT = 0`.

The result is distinct from the prior pre-cost envelope audit. That earlier audit established that the current `w_move=10.0` suppresses positive pre-cost branches. This fresh-policy gate asks a stricter question: whether the fresh policy's own candidate distribution already has positive expected MOVE continuation value at `w=0.50`. Here it does not.

Because no 3-epoch trajectory was executed, the post-training classifications are not applicable:

```text
OPERATION_POLICY_RESULT = INCONCLUSIVE
```

`INCONCLUSIVE` means no training trajectory exists to classify as `NO_IMMEDIATE_STAY_COLLAPSE`, `STAY_COLLAPSE_PERSISTS`, or `MOVE_COLLAPSE`; it is not a claim that the fresh policy was already saturated to STAY.

## Required final keys

```text
NEW_CONFIG = configs/proposed/system_config_ppo_recovery_r2_wmove_0p50_dev3.toml
PRIMARY_W_MOVE = 0.50
ONLY_OBJECTIVE_WEIGHT_CHANGED = YES

FRESH_INITIALIZATION = YES
WARM_START = NO
PRIMARY_PRETRAIN_GATE = FAIL
OFFLINE_VS_DIRECT_Q_MAX_ABS_DIFF = 0.0

W025_EXPECTED_MOVE_BEATS_STAY_COUNT = 2
W025_MEAN_EXPECTED_DELTA_Q = -0.0006287678886545189
W050_EXPECTED_MOVE_BEATS_STAY_COUNT = 0
W050_MEAN_EXPECTED_DELTA_Q = -0.0020060482979103013
W100_EXPECTED_MOVE_BEATS_STAY_COUNT = 0
W100_MEAN_EXPECTED_DELTA_Q = -0.0047606091164214825

DEV3_TRAINING_EXECUTED = NO
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
EPOCH1_CRITIC_EV = N/A
EPOCH2_CRITIC_EV = N/A
EPOCH3_CRITIC_EV = N/A

OPERATION_POLICY_RESULT = INCONCLUSIVE

SYSTEM_MODEL_FORMULA_CHANGED = NO
REWARD_FORMULA_CHANGED = NO
OBJECTIVE_TRADEOFF_WEIGHT_CHANGED_IN_DEV_CONFIG = YES
PHYSICAL_MOVEMENT_PARAMETERS_CHANGED = NO
CACHE_CREDIT_CHANGED = NO
CANDIDATE_CREDIT_CHANGED = NO
OPERATION_CREDIT_CHANGED = NO
PPO_CHANGED = NO
PRODUCTION_R2_CONFIG_CHANGED = NO

EPISODE_REGENERATION = NO
SUMO_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
FORMAL_EVALUATION_EXECUTED = NO
TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO
```
