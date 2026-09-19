# UAV MOVE pre-cost value envelope gate

Date: 2026-09-19  
Repository: `C:\Users\tangning\Desktop\code\adaptive_cache_mec`  
Checkpoint: `runs/ppo_recovery_r2_dev20_20260918/latest_checkpoint.pt`, epoch 7  
Frozen realization: `runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911/episode_20001`

## Scope

This bounded diagnostic separates candidate value before movement cost from the formal candidate ranking. It uses only the fixed states `slot = 20, 50, 80, 110, 140, 170, 200, 230, 260, 290` × `UAV_0/UAV_1`, for `STATE_COUNT = 20` and all `1,600` legal MOVE candidates.

The existing reports did not retain the complete branch-level `actual_task_utility` and `cache_cost` arrays, so one exact H1 replay was required. It reused the production candidate enumeration, direct policy primitive configuration, `execute_slot`, Matching/KKT, reward identity, and the existing epoch-7 checkpoint. No candidate pruning, geometry change, reward change, optimizer step, training, evaluation, horizon expansion, episode generation, or SUMO run was performed.

## Definitions and validation

For every MOVE candidate `c`, with the same target-UAV STAY branch as baseline:

```text
DeltaTask_c       = Q_task(c) - Q_task(STAY)
DeltaCache_c      = C_cache(c) - C_cache(STAY)
DeltaMoveCost_c   = C_move(c) - C_move(STAY)
PRE_COST_BENEFIT  = DeltaTask_c - DeltaCache_c
DeltaR_c          = PRE_COST_BENEFIT - DeltaMoveCost_c
```

`FORMAL_BEST` is `argmax DeltaQ_c`. `PRE_COST_BEST` is `argmax PRE_COST_BENEFIT_c`; they are intentionally different selections. Movement-cost fields below are the candidate-minus-STAY movement-cost delta, which is the term used by the decomposition. The target candidate’s absolute movement cost is identical here because target STAY has zero movement cost.

The decomposition identity was checked for every candidate:

```text
DECOMPOSITION_MAX_ABS_RESIDUAL = 8.1878948e-16 <= 1e-8
```

The candidate-Q identity residual was `5.6843419e-14`, consistent with the existing float32 helper storage and unrelated to the immediate-reward decomposition.

## Main result

The pre-cost envelope reverses the apparent “low move value” conclusion obtained by looking only at the 20 `max DeltaQ` candidates:

- Every one of the 20 states has at least one candidate with positive `PRE_COST_BENEFIT`.
- There are `797` positive-pre-cost candidates among `1,600` legal MOVE candidates.
- At `w_move = 10.0`, all `797/797` become non-positive in formal immediate reward.
- No candidate in any of the 20 states has `DeltaR > 0` at the current movement-cost weight.
- `FORMAL_BEST` and `PRE_COST_BEST` are different in all 20 states.

Therefore, the prior 13/20 negative pre-cost count was a selection artifact of choosing only `argmax DeltaQ`; it does not mean that those states lack any task/cache-side valuable MOVE.

## Per-state formal-vs-pre-cost best candidates

| slot | UAV | FORMAL_BEST `DeltaQ` | FORMAL_BEST pre-cost | formal d (m) | formal Δmove cost | PRE_COST_BEST benefit | PRE_COST_BEST `DeltaR` | PRE_COST_BEST `DeltaQ` | pre-cost d (m) | pre-cost Δmove cost | same |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20 | UAV_0 | -0.016399 | +0.002318 | 15.00 | 0.0572 | +0.005575 | -0.051625 | -0.046620 | 15.00 | 0.0572 | NO |
| 20 | UAV_1 | -0.024241 | +0.004938 | 3.75 | 0.0275 | +0.022688 | -0.034512 | -0.044491 | 15.00 | 0.0572 | NO |
| 50 | UAV_0 | -0.018292 | +0.000480 | 3.75 | 0.0275 | +0.004811 | -0.052389 | -0.048116 | 15.00 | 0.0572 | NO |
| 50 | UAV_1 | -0.025365 | -0.000245 | 3.75 | 0.0275 | +0.000991 | -0.056209 | -0.065822 | 15.00 | 0.0572 | NO |
| 80 | UAV_0 | -0.018530 | +0.000334 | 3.75 | 0.0275 | +0.004388 | -0.052812 | -0.048509 | 15.00 | 0.0572 | NO |
| 80 | UAV_1 | -0.026028 | -0.001122 | 3.75 | 0.0275 | +0.005241 | -0.051959 | -0.062731 | 15.00 | 0.0572 | NO |
| 110 | UAV_0 | -0.016391 | +0.000556 | 15.00 | 0.0572 | +0.005205 | -0.051995 | -0.062890 | 15.00 | 0.0572 | NO |
| 110 | UAV_1 | -0.025272 | -0.001007 | 3.75 | 0.0275 | +0.005874 | -0.051326 | -0.063014 | 15.00 | 0.0572 | NO |
| 140 | UAV_0 | -0.007705 | +0.000300 | 15.00 | 0.0572 | +0.002954 | -0.054246 | -0.048753 | 15.00 | 0.0572 | NO |
| 140 | UAV_1 | -0.025211 | -0.001282 | 3.75 | 0.0275 | +0.006812 | -0.050388 | -0.063205 | 15.00 | 0.0572 | NO |
| 170 | UAV_0 | -0.003424 | -0.001034 | 15.00 | 0.0572 | +0.001576 | -0.055624 | -0.090414 | 15.00 | 0.0572 | NO |
| 170 | UAV_1 | -0.024109 | -0.001096 | 3.75 | 0.0275 | +0.004351 | -0.052849 | -0.070702 | 15.00 | 0.0572 | NO |
| 200 | UAV_0 | +0.002960 | -0.000020 | 15.00 | 0.0572 | +0.003313 | -0.053887 | -0.049187 | 15.00 | 0.0572 | NO |
| 200 | UAV_1 | -0.021937 | -0.000357 | 3.75 | 0.0275 | +0.001448 | -0.055752 | -0.077633 | 15.00 | 0.0572 | NO |
| 230 | UAV_0 | -0.000075 | -0.000523 | 15.00 | 0.0572 | +0.000538 | -0.056662 | -0.108450 | 15.00 | 0.0572 | NO |
| 230 | UAV_1 | -0.022291 | -0.000345 | 3.75 | 0.0275 | +0.001370 | -0.055830 | -0.077498 | 15.00 | 0.0572 | NO |
| 260 | UAV_0 | -0.000690 | -0.001504 | 15.00 | 0.0572 | +0.002255 | -0.054945 | -0.092970 | 15.00 | 0.0572 | NO |
| 260 | UAV_1 | -0.021812 | -0.000080 | 3.75 | 0.0275 | +0.000378 | -0.056822 | -0.076018 | 15.00 | 0.0572 | NO |
| 290 | UAV_0 | -0.000918 | +0.000862 | 15.00 | 0.0572 | +0.004223 | -0.052977 | -0.026366 | 15.00 | 0.0572 | NO |
| 290 | UAV_1 | -0.025577 | -0.002899 | 3.75 | 0.0275 | +0.016681 | -0.040519 | -0.062675 | 15.00 | 0.0572 | NO |

## State-level pre-cost envelope

| slot | UAV | positive pre-cost count | fraction | max pre-cost benefit | median pre-cost benefit | positive formal `DeltaR` count | max break-even weight |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20 | UAV_0 | 40 | 0.500 | 0.005575162 | -3.229969e-06 | 0 | 0.974679 |
| 20 | UAV_1 | 40 | 0.500 | 0.022688075 | -8.993739e-06 | 0 | 3.966447 |
| 50 | UAV_0 | 40 | 0.500 | 0.004811111 | -2.376331e-06 | 0 | 0.841103 |
| 50 | UAV_1 | 40 | 0.500 | 0.000991026 | -4.896978e-07 | 0 | 0.173256 |
| 80 | UAV_0 | 40 | 0.500 | 0.004387882 | -2.298150e-06 | 0 | 0.767112 |
| 80 | UAV_1 | 40 | 0.500 | 0.005241355 | -1.963009e-06 | 0 | 0.916321 |
| 110 | UAV_0 | 40 | 0.500 | 0.005204524 | -5.147457e-06 | 0 | 0.909882 |
| 110 | UAV_1 | 40 | 0.500 | 0.005874287 | -1.935745e-06 | 0 | 1.026973 |
| 140 | UAV_0 | 40 | 0.500 | 0.002953570 | -1.478083e-06 | 0 | 0.516358 |
| 140 | UAV_1 | 40 | 0.500 | 0.006812333 | -2.366173e-06 | 0 | 1.190967 |
| 170 | UAV_0 | 40 | 0.500 | 0.001575592 | -1.764644e-06 | 0 | 0.275453 |
| 170 | UAV_1 | 40 | 0.500 | 0.004350714 | -2.430081e-06 | 0 | 0.760614 |
| 200 | UAV_0 | 37 | 0.463 | 0.003312815 | -1.041626e-05 | 0 | 0.579164 |
| 200 | UAV_1 | 40 | 0.500 | 0.001448370 | -4.456430e-07 | 0 | 0.253211 |
| 230 | UAV_0 | 40 | 0.500 | 0.000538144 | -4.707895e-07 | 0 | 0.094081 |
| 230 | UAV_1 | 40 | 0.500 | 0.001369910 | -4.976924e-07 | 0 | 0.239495 |
| 260 | UAV_0 | 40 | 0.500 | 0.002255052 | -3.841185e-06 | 0 | 0.394240 |
| 260 | UAV_1 | 40 | 0.500 | 0.000377500 | -3.364377e-07 | 0 | 0.065997 |
| 290 | UAV_0 | 40 | 0.500 | 0.004223047 | -5.003003e-06 | 0 | 0.738295 |
| 290 | UAV_1 | 40 | 0.500 | 0.016680876 | -4.886070e-06 | 0 | 2.916237 |

## Movement-cost suppression

```text
TOTAL_POSITIVE_PRE_COST_CANDIDATES       = 797
POSITIVE_AFTER_MOVEMENT_COST_COUNT       = 0
SUPPRESSED_BY_MOVEMENT_COST_COUNT        = 797
SUPPRESSED_BY_MOVEMENT_COST_FRACTION     = 1.000000
STATES_WITH_POSITIVE_FORMAL_IMMEDIATE_MOVE = 0
SCALE_SUPPRESSED_STATE_COUNT             = 20
```

Every positive pre-cost candidate has a positive break-even weight, but every break-even weight is below the current `w_move = 10.0`. The largest candidate-level break-even weight is only `3.966447`; the state-level maximum has the same maximum. Thus the current weight eliminates all positive immediate MOVE candidates in every state.

## Break-even weight envelope

For every candidate with positive pre-cost benefit:

```text
w_break_even,c = PRE_COST_BENEFIT_c / base_move_term_c
```

Only candidates with positive `base_move_term` were used. These are diagnostic thresholds, not recommended configuration values.

```text
POSITIVE_PRE_COST_BREAK_EVEN_COUNT          = 797
BREAK_EVEN_WEIGHT_P10                      = 0.0334015
BREAK_EVEN_WEIGHT_P25                      = 0.0916342
BREAK_EVEN_WEIGHT_MEDIAN                   = 0.2532115
BREAK_EVEN_WEIGHT_P75                      = 0.5343809
BREAK_EVEN_WEIGHT_P90                      = 0.8866263
BREAK_EVEN_WEIGHT_MAX                      = 3.9664467
STATE_MAX_BREAK_EVEN_WEIGHT_MEDIAN         = 0.7494546
STATE_MAX_BREAK_EVEN_WEIGHT_MAX            = 3.9664467
CURRENT_W_MOVE                             = 10.0
CURRENT_W_MOVE_ABOVE_BREAK_EVEN_FRACTION   = 1.000000
CURRENT_W_MOVE_ELIMINATES_ALL_POSITIVE_MOVES_STATE_COUNT = 20
```

## Candidate distance layers

The actual production candidate distances were used as the bins; no arbitrary binning was introduced.

| distance (m) | candidate count | mean pre-cost benefit | max pre-cost benefit | positive fraction | mean movement cost | mean `DeltaR` |
|---:|---:|---:|---:|---:|---:|---:|
| 3.75 | 480 | -0.000002349 | 0.005762605 | 0.500000 | 0.027500 | -0.027502349 |
| 7.50 | 480 | -0.000009394 | 0.011507648 | 0.497917 | 0.037400 | -0.037409394 |
| 11.25 | 320 | -0.000021137 | 0.017043731 | 0.496875 | 0.047300 | -0.047321137 |
| 15.00 | 320 | -0.000037575 | 0.022688075 | 0.496875 | 0.057200 | -0.057237575 |

Positive pre-cost candidates are not confined to short distances: approximately half of each discrete distance layer is positive before cost, and every state’s `PRE_COST_BEST` is at `15.0 m`. However, the fixed-plus-linear movement penalty increases from `0.0275` to `0.0572`, and no distance layer contains a positive formal immediate move at `w_move=10`.

## UAV-separated results

```text
UAV_0_STATES_WITH_POSITIVE_PRE_COST_MOVE = 10/10
UAV_0_MEAN_STATE_MAX_PRE_COST_BENEFIT    = 0.003483690
UAV_0_MAX_STATE_MAX_PRE_COST_BENEFIT     = 0.005575162
UAV_0_POSITIVE_PRE_COST_CANDIDATES       = 397
UAV_0_MEAN_POSITIVE_CANDIDATE_BENEFIT    = 0.001294059
UAV_0_MAX_POSITIVE_CANDIDATE_BENEFIT     = 0.005575162

UAV_1_STATES_WITH_POSITIVE_PRE_COST_MOVE = 10/10
UAV_1_MEAN_STATE_MAX_PRE_COST_BENEFIT    = 0.006583445
UAV_1_MAX_STATE_MAX_PRE_COST_BENEFIT     = 0.022688075
UAV_1_POSITIVE_PRE_COST_CANDIDATES       = 400
UAV_1_MEAN_POSITIVE_CANDIDATE_BENEFIT    = 0.002410618
UAV_1_MAX_POSITIVE_CANDIDATE_BENEFIT     = 0.022688075
```

UAV_1 has the larger pre-cost envelope in this frozen witness, but the current weight suppresses both UAVs completely at the formal immediate-reward level.

## Diagnosis

```text
CASE_A_SCALE_DOMINANCE_STRONG = YES
CASE_B_PHYSICAL_MOVE_LOW_VALUE = NO
CASE_C_MIXED = NO
PRIMARY_DIAGNOSIS = MOVEMENT_COST_SCALE_DOMINANCE_STRONG
```

This is a strong scale-dominance result: all 20 states contain task/cache-side valuable MOVE candidates, but the current objective movement cost suppresses every one of them. The earlier `13/20` result cannot be used to claim physical MOVE value is absent because it selected only the formal `max DeltaQ` candidate.

This gate does not recommend setting `w_move` to any particular value. It only establishes that the next investigation should be the movement-cost coefficient/weight basis and its literature/experiment calibration, while preserving the production reward and model until an explicitly authorized change is designed.

## Required final keys

```text
STATE_COUNT = 20
TOTAL_LEGAL_MOVE_CANDIDATES = 1600
STATES_WITH_ANY_POSITIVE_PRE_COST_MOVE = 20
STATES_WITH_NO_POSITIVE_PRE_COST_MOVE = 0
TOTAL_POSITIVE_PRE_COST_CANDIDATES = 797
POSITIVE_AFTER_MOVEMENT_COST_COUNT = 0
SUPPRESSED_BY_MOVEMENT_COST_COUNT = 797
SUPPRESSED_BY_MOVEMENT_COST_FRACTION = 1.000000
SCALE_SUPPRESSED_STATE_COUNT = 20
STATES_WITH_POSITIVE_FORMAL_IMMEDIATE_MOVE = 0
MEAN_STATE_MAX_PRE_COST_BENEFIT = 0.0050335672
MEDIAN_STATE_MAX_PRE_COST_BENEFIT = 0.0042868802
MAX_STATE_MAX_PRE_COST_BENEFIT = 0.0226880750
FORMAL_BEST_EQUALS_PRE_COST_BEST_COUNT = 0
CURRENT_W_MOVE = 10.0
POSITIVE_PRE_COST_BREAK_EVEN_COUNT = 797
BREAK_EVEN_WEIGHT_MEDIAN = 0.2532115
BREAK_EVEN_WEIGHT_P75 = 0.5343809
BREAK_EVEN_WEIGHT_P90 = 0.8866263
BREAK_EVEN_WEIGHT_MAX = 3.9664467
STATE_MAX_BREAK_EVEN_WEIGHT_MEDIAN = 0.7494546
STATE_MAX_BREAK_EVEN_WEIGHT_MAX = 3.9664467
CURRENT_W_MOVE_ABOVE_BREAK_EVEN_FRACTION = 1.000000
CURRENT_W_MOVE_ELIMINATES_ALL_POSITIVE_MOVES_STATE_COUNT = 20
UAV0_STATES_WITH_POSITIVE_PRE_COST_MOVE = 10
UAV1_STATES_WITH_POSITIVE_PRE_COST_MOVE = 10
DECOMPOSITION_MAX_ABS_RESIDUAL = 8.1878948e-16
PRIMARY_DIAGNOSIS = MOVEMENT_COST_SCALE_DOMINANCE_STRONG
MOVEMENT_COST_IMPLEMENTATION_BUG = NO
```

## No-change flags

```text
SOURCE_MODIFIED = NO
TRAINING_EXECUTED = NO
OPTIMIZER_STEP_EXECUTED = NO
REWARD_CHANGED = NO
MOVEMENT_COST_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO
PPO_CHANGED = NO
OPERATION_CREDIT_CHANGED = NO
CANDIDATE_CREDIT_CHANGED = NO
CACHE_CREDIT_CHANGED = NO
EPISODE_REGENERATION = NO
SUMO_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
FORMAL_EVALUATION_EXECUTED = NO
TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO
```
