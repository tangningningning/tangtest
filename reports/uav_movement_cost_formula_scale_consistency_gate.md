# UAV movement-cost formula and scale consistency gate

Date: 2026-09-18  
Repository: `C:\Users\tangning\Desktop\code\adaptive_cache_mec`  
Checkpoint: `runs/ppo_recovery_r2_dev20_20260918/latest_checkpoint.pt`, epoch 7  
Frozen realization: `runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911/episode_20001`

## Scope and method

This is a bounded static/runtime audit only. It used the existing R2 counterfactual helper on the exact fixed states `slot = 20, 50, 80, 110, 140, 170, 200, 230, 260, 290` × `UAV_0/UAV_1`, for 20 target-UAV states and 1,600 legal MOVE candidates. The prior candidate report did not retain branch-level task utility and cache-cost fields, so the same production enumeration and `execute_slot` path was replayed once to recover those fields; no new state protocol, candidate pruning, or alternative evaluator was introduced.

No optimizer step, training, checkpoint write, episode generation, SUMO run, H3/H5/H10/H30 horizon run, formal evaluation, or final-seed evaluation was performed.

For each target-UAV state, “BEST MOVE” means the legal MOVE candidate with maximum exact candidate-conditioned `DeltaQ`; the immediate reward decomposition is then taken from that same branch.

## Production formula chain

The active policy-primitive path is:

```text
UAV operation/candidate action
  -> uav_configuration.enumerate_uav_primitive_actions()
  -> final target position q_new
  -> ||q_new - q_old||_2
  -> system_model.uav_movement_cost()
  -> JointConfigurationResult.movement_energy/movement_cost
  -> slot_environment.execute_slot()
  -> actual task Matching/KKT utility Q
  -> system_utility = Q - cache_cost - movement_cost
```

The relevant implementation points are:

- `src/decision/uav_configuration.py:236-260` filters each MOVE candidate by boundary, safety, and `max_move_distance_m`; `max_move_distance_m = dt_s * max_horizontal_speed_mps`.
- `src/decision/uav_configuration.py:524-552` calls `system_model.uav_movement_cost()` once for each final UAV transition, sums the returned energy and weighted objective cost, and treats the path as one final `q_old -> q_final` transition.
- `src/decision/slot_environment.py:788-831` rebuilds the direct policy primitive configuration and obtains `movement_energy`/`movement_cost` from `movement_summary()`.
- `src/decision/slot_environment.py:451-490` obtains the actual Matching/KKT task utility and applies `system_utility = actual_task_utility - cache_cost - movement_cost`.
- `src/ppo_simple/rollout.py:161-166` independently checks the same reward identity.

The current implementation is therefore using the following formula, with `d` equal to the effective distance after the movement tolerance gate:

```text
d_raw       = ||q_new - q_old||_2                         [m]
z_actual    = 1[d_raw > position_tolerance_m]
d           = d_raw if z_actual=1 else 0                  [m]
E_move      = xi_mov * d                                  [J]
C_move     = w_move * (z_actual * eta_q + lambda * E_move)
G           = Q_task - C_cache - C_move
```

For the active R2 configuration this is:

```text
C_move = 10.0 * (z_actual * 0.00176 + 1.76e-5 * 15.0 * d)
       = 10.0 * (z_actual * 0.00176 + 0.000264 * d)
```

Thus a legal MOVE of `3.75 m` costs `0.0275`, a `15 m` MOVE costs `0.0572`, and a target-UAV STAY costs exactly `0`. These are objective-cost units, not joules.

## Formula, unit, and parameter audit

| Quantity | Current source/value | Unit or role | Result |
|---|---:|---|---|
| `dt_s` | `1.0` | s | consistent |
| `max_horizontal_speed_mps` | `15.0` | m/s | consistent |
| legal one-slot distance limit | `dt * vmax = 15.0` | m | consistent |
| `movement_energy_coefficient_j_per_m` | `15.0` | J/m | consistent with `E_move = xi*d` |
| `movement_fixed_cost` | `0.00176` | objective-cost base term | consistent |
| `lambda_movement` | `1.76e-5` | objective cost per J in this formula | consistent |
| `uav_movement_cost_weight` | `10.0` | dimensionless scale weight | applied in the model function |
| position tolerance | `1e-6` | m | shared with STAY/actual-move gate |
| task utility `Q_task` | `task_utility(...)` | dimensionless normalized objective utility | separate from propulsion energy |
| `system_utility` | `Q-C_cache-C_move` | dimensionless objective | consistent |

The system-model implementation at `src/common/system_model.py:309-362` computes `distance_m`, `movement_energy_j = xi * effective_distance`, and `movement_cost = weight * (actual_z * eta + lambda * movement_energy)`. The config comments at `configs/proposed/system_config_ppo_recovery_r2_6e4.toml:396-406` describe the same base cost and the external weight. The `15 J/m` value is documented as a linear equivalent mapping from a rotor-power reference, not as a separate joule-valued term added directly to `Q_task`.

Runtime formula checks over all captured branches found zero absolute residual at the stored floating-point precision for total distance reconstruction, total movement energy reconstruction, and total movement-cost reconstruction. The largest reward decomposition residual was `1.7347e-16`.

Therefore:

```text
FORMULA_MATCH_SYSTEM_MODEL = YES
MOVEMENT_COST_UNIT_ERROR = NO
```

## Weight application and energy double-count audits

`uav_search_config_from_project()` reads `[objective].uav_movement_cost_weight = 10.0` and passes it into `movement_summary()`, which passes it once into `system_model.uav_movement_cost()` for each UAV. The resulting aggregate `configuration.movement_cost` is subtracted once by `execute_slot()`.

The `configuration_override` validation path rebuilds the configuration and checks exact equality of the rebuilt movement cost. That is a validation recomputation, not a second addition to `system_utility`. The final-system-utility application count is therefore one, as expected.

```text
MOVEMENT_COST_WEIGHT_APPLICATION_COUNT_EXPECTED = 1
MOVEMENT_COST_WEIGHT_APPLICATION_COUNT_IN_SYSTEM_UTILITY = 1
DOUBLE_WEIGHTING = NO
```

The task-side energy path is separate. `task_allocation.task_utility_sum()` uses local CPU energy or vehicle transmission energy from the local/edge/cloud execution models (`src/common/system_model.py:32-175` and `src/decision/task_allocation.py:2056-2099`). It does not receive `configuration.movement_energy`. Propulsion movement energy is retained in `configuration.movement_energy` and mapped into `movement_cost`; it is not inserted into per-task vehicle energy or `actual_task_utility`.

```text
MOVEMENT_ENERGY_DOUBLE_COUNT = NO
```

The current model has no separate horizontal-hover/base-energy term. Consequently, target-UAV STAY has zero horizontal distance, zero movement propulsion energy, and zero movement cost. That is the declared model scope, not an accidental second energy subtraction.

## Geometry and STAY gate

The active geometry is `dt_s = 1.0 s`, `vmax = 15.0 m/s`, hence `d_max = 15.0 m`. All 1,600 legal MOVE candidates satisfied the same production feasibility gate:

```text
DISTANCE_LIMIT_VIOLATION_COUNT = 0
STAY_DISTANCE_MAX = 0.0 m
STAY_MOVEMENT_COST_MAX = 0.0
```

The candidate-to-`execute_slot` target-distance residual was also `0.0` at the captured precision. No evidence supports a geometry or STAY-semantics implementation bug.

## Best-MOVE component decomposition

For each row:

```text
DeltaTaskUtility       = Q_task(MOVE) - Q_task(STAY)
DeltaCacheCost         = C_cache(MOVE) - C_cache(STAY)
DeltaMovementCost      = C_move(MOVE) - C_move(STAY)
DeltaImmediateUtility  = DeltaTaskUtility - DeltaCacheCost - DeltaMovementCost
DeltaR                 = system_utility(MOVE) - system_utility(STAY)
DeltaR_before_move     = DeltaR + DeltaMovementCost
```

`DeltaImmediateUtility` and `DeltaR` agree to a maximum absolute residual of `1.7347e-16` in all 20 rows.

| slot | UAV | best `DeltaQ` | d (m) | `DeltaTaskUtility` | `DeltaCacheCost` | `DeltaMovementCost` | `DeltaImmediateUtility` | `DeltaR_before_move` | `DeltaR` | break-even `w_move` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | UAV_0 | -0.016399 | 15.00 | +0.002318 | 0 | +0.057200 | -0.054882 | +0.002318 | -0.054882 | 0.405256 |
| 20 | UAV_1 | -0.024241 | 3.75 | +0.004938 | 0 | +0.027500 | -0.022562 | +0.004938 | -0.022562 | 1.795481 |
| 50 | UAV_0 | -0.018292 | 3.75 | +0.000480 | 0 | +0.027500 | -0.027020 | +0.000480 | -0.027020 | 0.174550 |
| 50 | UAV_1 | -0.025365 | 3.75 | -0.000245 | 0 | +0.027500 | -0.027745 | -0.000245 | -0.027745 | -0.089126 |
| 80 | UAV_0 | -0.018530 | 3.75 | +0.000334 | 0 | +0.027500 | -0.027166 | +0.000334 | -0.027166 | 0.121354 |
| 80 | UAV_1 | -0.026028 | 3.75 | -0.001122 | 0 | +0.027500 | -0.028622 | -0.001122 | -0.028622 | -0.407825 |
| 110 | UAV_0 | -0.016391 | 15.00 | +0.000556 | 0 | +0.057200 | -0.056644 | +0.000556 | -0.056644 | 0.097228 |
| 110 | UAV_1 | -0.025272 | 3.75 | -0.001007 | 0 | +0.027500 | -0.028507 | -0.001007 | -0.028507 | -0.366012 |
| 140 | UAV_0 | -0.007705 | 15.00 | +0.000300 | 0 | +0.057200 | -0.056900 | +0.000300 | -0.056900 | 0.052525 |
| 140 | UAV_1 | -0.025211 | 3.75 | -0.001282 | 0 | +0.027500 | -0.028782 | -0.001282 | -0.028782 | -0.466176 |
| 170 | UAV_0 | -0.003424 | 15.00 | -0.001034 | 0 | +0.057200 | -0.058234 | -0.001034 | -0.058234 | -0.180711 |
| 170 | UAV_1 | -0.024109 | 3.75 | -0.001096 | 0 | +0.027500 | -0.028596 | -0.001096 | -0.028596 | -0.398374 |
| 200 | UAV_0 | +0.002960 | 15.00 | -0.000020 | 0 | +0.057200 | -0.057220 | -0.000020 | -0.057220 | -0.003534 |
| 200 | UAV_1 | -0.021937 | 3.75 | -0.000357 | 0 | +0.027500 | -0.027857 | -0.000357 | -0.027857 | -0.129836 |
| 230 | UAV_0 | -0.000075 | 15.00 | -0.000523 | 0 | +0.057200 | -0.057723 | -0.000523 | -0.057723 | -0.091401 |
| 230 | UAV_1 | -0.022291 | 3.75 | -0.000345 | 0 | +0.027500 | -0.027845 | -0.000345 | -0.027845 | -0.125496 |
| 260 | UAV_0 | -0.000690 | 15.00 | -0.001504 | 0 | +0.057200 | -0.058704 | -0.001504 | -0.058704 | -0.262957 |
| 260 | UAV_1 | -0.021812 | 3.75 | -0.000080 | 0 | +0.027500 | -0.027580 | -0.000080 | -0.027580 | -0.028975 |
| 290 | UAV_0 | -0.000918 | 15.00 | +0.000862 | 0 | +0.057200 | -0.056338 | +0.000862 | -0.056338 | 0.150767 |
| 290 | UAV_1 | -0.025577 | 3.75 | -0.002899 | 0 | +0.027500 | -0.030399 | -0.002899 | -0.030399 | -1.054247 |

## Scale diagnostics

```text
BEST_MOVE_STATE_COUNT                  = 20
BEST_MOVE_BEATS_STAY_COUNT             = 1
BEFORE_MOVE_COST_POSITIVE_COUNT        = 7
BEFORE_MOVE_COST_NEGATIVE_COUNT        = 13
MEAN_DELTA_R_BEFORE_MOVEMENT_COST      = -8.6207557e-05
MEAN_ABS_TASK_UTILITY_DELTA             = 0.0010650427
MEAN_ABS_CACHE_COST_DELTA               = 0.0
MEAN_ABS_MOVEMENT_COST_DELTA            = 0.0393800000
MOVEMENT_TO_TASK_DELTA_RATIO            = 36.9750440
CURRENT_W_MOVE                          = 10.0
BREAK_EVEN_WEIGHT_AVAILABLE             = YES
BREAK_EVEN_WEIGHT_MEAN                  = -0.0403755
BREAK_EVEN_WEIGHT_MEDIAN                = -0.0902636
BREAK_EVEN_WEIGHT_MIN                   = -1.0542475
BREAK_EVEN_WEIGHT_MAX                   = 1.7954807
CURRENT_W_MOVE_ABOVE_BREAK_EVEN_COUNT   = 20/20
```

The break-even weight is computed from the linear current formula:

```text
w_break_even = (DeltaTaskUtility - DeltaCacheCost) / base_move_term
base_move_term = z_actual * eta_q + lambda_movement * E_move
```

For the seven states with positive pre-movement benefit, the break-even weight has mean `0.399595`, median `0.150767`, and maximum `1.795481`; the active weight `10.0` exceeds every one. For the other 13 states, pre-movement benefit is already non-positive, so no non-negative movement-cost weight can make that selected MOVE immediately preferable to STAY.

## Diagnosis

```text
PRIMARY_DIAGNOSIS = D_MIXED
```

The mixed diagnosis has two separately supported components:

- **B / scale dominance:** 7/20 selected best-MOVE states have positive `DeltaR_before_move`, but the current weighted movement penalty is larger. The objective-side movement delta is about 37 times the mean absolute task-utility delta, and `w_move=10` is above every positive break-even weight.
- **C / low physical move value:** 13/20 selected best-MOVE states have non-positive task-minus-cache benefit even before movement cost. This is not explained by a movement-cost implementation error.

There is no evidence for category A: the formula matches the system model, units are internally consistent, all legal candidates satisfy the distance limit, STAY is zero for the target UAV, the weight is not double-counted in final utility, and movement propulsion energy is not included again in task vehicle energy. The evidence is therefore not a unit/formula bug; it is a mixed scale-and-scenario-value issue under the current objective.

## Required gate keys

```text
MOVEMENT_COST_FORMULA = w_move * (z_actual * eta_q + lambda_movement * xi_mov * distance_m)
MOVEMENT_COST_WEIGHT = 10.0
MOVEMENT_COST_WEIGHT_SOURCE = configs/proposed/system_config_ppo_recovery_r2_6e4.toml [objective].uav_movement_cost_weight
FORMULA_MATCH_SYSTEM_MODEL = YES
MOVEMENT_COST_UNIT_ERROR = NO
DOUBLE_WEIGHTING = NO
MOVEMENT_ENERGY_DOUBLE_COUNT = NO
DISTANCE_LIMIT_VIOLATION_COUNT = 0
STAY_DISTANCE_MAX = 0.0 m
STAY_MOVEMENT_COST_MAX = 0.0
BEST_MOVE_STATE_COUNT = 20
BEFORE_MOVE_COST_POSITIVE_COUNT = 7
BEFORE_MOVE_COST_NEGATIVE_COUNT = 13
MEAN_DELTA_R_BEFORE_MOVEMENT_COST = -8.6207557e-05
CURRENT_W_MOVE = 10.0
BREAK_EVEN_WEIGHT_AVAILABLE = YES
BREAK_EVEN_WEIGHT_MEAN = -0.0403755
BREAK_EVEN_WEIGHT_MEDIAN = -0.0902636
MEAN_ABS_TASK_UTILITY_DELTA = 0.0010650427
MEAN_ABS_CACHE_COST_DELTA = 0.0
MEAN_ABS_MOVEMENT_COST_DELTA = 0.0393800
MOVEMENT_TO_TASK_DELTA_RATIO = 36.9750440
PRIMARY_DIAGNOSIS = D_MIXED
```

## No-change flags

```text
SOURCE_MODIFIED = NO
TRAINING_EXECUTED = NO
OPTIMIZER_STEP_EXECUTED = NO
CHECKPOINT_WRITTEN = NO
REWARD_CHANGED = NO
MOVEMENT_COST_CHANGED = NO
CACHE_CREDIT_CHANGED = NO
CANDIDATE_CREDIT_CHANGED = NO
OPERATION_CREDIT_CHANGED = NO
EPISODE_REGENERATION = NO
SUMO_EXECUTED = NO
FORMAL_EVALUATION_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
COMMITTED = NO
PUSHED = NO
TEMP_FILES_REMOVED = YES
```
