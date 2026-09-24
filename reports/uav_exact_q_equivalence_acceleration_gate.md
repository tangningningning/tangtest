# UAV_EXACT_Q_EQUIVALENCE_ACCELERATION_GATE

## Gate conclusion

```text
TASK = UAV_EXACT_Q_EQUIVALENCE_ACCELERATION_GATE
UAV_EXACT_Q_EQUIVALENCE = PASS
DUPLICATE_EXACT_Q_EVALUATION_FOUND = YES
SAFE_ACCELERATION_IMPLEMENTED = YES
CONDA_ENV = adaptive_cache_mec
TRAINING_REALIZATION = data/episode_pools/proposed_joint_training/episode_20001
CHECKPOINT = runs/proposed_joint_short_stability_gate/latest_checkpoint.pt
```

The optimization is a slot-local memoization of already requested exact
branches. It does not alter the reward, Matching, KKT, system model, teacher,
candidate set, candidate order, sampling, or PPO update semantics. No formal
300-epoch training was started.

## Baseline bottleneck and production call graph

The existing production path was traced narrowly as:

```text
PPOV2Trainer._compute_exact_uav_operation_credit
  -> per selected transition and UAV head
  -> enumerate_uav_target_candidates
  -> _evaluate_branch(STAY / legal MOVE candidates)
  -> build_policy_target_configuration
  -> slot_environment.execute_slot(configuration_override=...)
  -> task_allocation.optimize_task_matching
  -> Matching candidate evaluations / allocate_resources_kkt
  -> actual reward and next observation
  -> q_stay / q_move table
  -> _exact_q_operation_teacher and _exact_q_listwise_teacher
```

The operation teacher and candidate listwise teacher already consume the same
`q_stay`/`candidate_q_values` arrays within one credit call. The remaining
duplicate was across UAV-head entry points: the same slot-local final UAV
configuration was requested through different logical branches.

## Duplicate computation evidence

The baseline probe used the existing production implementation, seed 20001,
the existing short-stability checkpoint, and slots
`0, 50, 100, 150, 200, 250, 299`. It made 265 exact branch requests and found
258 unique final configurations, with 7 duplicate groups. Every duplicate
group had identical reward, Matching assignment, CPU/BW KKT allocations,
delay, energy, and next state.

| slot | duplicate logical branches |
|---:|:---|
| 0 | `UAV_0:TARGET_POSITION:0` = `UAV_1:STAY` |
| 50 | `UAV_0:TARGET_POSITION:10` = `UAV_1:TARGET_POSITION:0` |
| 100 | `UAV_0:TARGET_POSITION:15` = `UAV_1:TARGET_POSITION:7` |
| 150 | `UAV_0:TARGET_POSITION:14` = `UAV_1:STAY` |
| 200 | `UAV_0:TARGET_POSITION:14` = `UAV_1:TARGET_POSITION:11` |
| 250 | `UAV_0:STAY` = `UAV_1:TARGET_POSITION:7` |
| 299 | `UAV_0:STAY` = `UAV_1:TARGET_POSITION:0` |

Baseline probe counters:

```text
BASELINE_BRANCH_REQUEST_COUNT = 265
BASELINE_UNIQUE_BRANCH_COUNT = 258
BASELINE_DUPLICATE_BRANCH_COUNT = 7
BASELINE_CACHE_HIT_COUNT = 0
BASELINE_EXECUTE_SLOT_CALL_COUNT = 265
BASELINE_MATCHING_CALL_COUNT = 265
BASELINE_BOTTOM_LEVEL_KKT_SOLVER_CALL_COUNT = 17020
```

## Implemented optimization

`PPOV2Trainer._compute_exact_uav_operation_credit` now creates a memo only
inside each transition/slot. Its key contains the origin, previous and final
Cache states, previous UAV positions, complete final UAV position tuple, and
UAV triggers. Candidate index is not used as identity. Scenario slice, actual
tasks, actual vehicle positions, and next exogenous state are fixed by the
slot-local scope.

On a hit, the cached reward and next observation are returned as safe copies.
No mutable Matching/KKT/result object is cached. The memo is destroyed before
the next slot and is never shared across slots, epochs, or realizations.
No multiprocessing, BLAS, candidate reduction, approximation, or parallel
execution was introduced.

## Strict equivalence probe

The optimized probe used the same saved rollout, same checkpoint, same seven
slots, same candidate set/order, and the same RNG reset. Results:

| field | result |
|:---|:---|
| `CANDIDATE_COUNT_EQUAL` | YES |
| `CANDIDATE_ORDER_EQUAL` | YES |
| `FINAL_UAV_POSITIONS_EQUAL` | YES |
| `MATCHING_ASSIGNMENTS_EQUAL` | YES |
| `TEACHER_TARGETS_EQUAL` | YES |
| `MAX_Q_ABS_DIFF` | 0 |
| `MAX_REWARD_ABS_DIFF` | 0 |
| `MAX_DELAY_ABS_DIFF` | 0 |
| `MAX_ENERGY_ABS_DIFF` | 0 |
| `MAX_KKT_ABS_DIFF` | 0 |
| `MAX_TEACHER_ABS_DIFF` | 0 |
| `UAV_EXACT_Q_EQUIVALENCE` | PASS |

Optimized probe counters:

```text
OPTIMIZED_BRANCH_REQUEST_COUNT = 265
OPTIMIZED_UNIQUE_BRANCH_COUNT = 258
OPTIMIZED_DUPLICATE_BRANCH_COUNT = 7
OPTIMIZED_UAV_EXACT_Q_CACHE_HIT_COUNT = 7
OPTIMIZED_EXECUTE_SLOT_CALL_COUNT = 258
OPTIMIZED_MATCHING_CALL_COUNT = 258
OPTIMIZED_BOTTOM_LEVEL_KKT_SOLVER_CALL_COUNT = 16570
OPTIMIZED_KKT_CALL_COUNT = 16570
```

The optimized probe artifacts are retained in
`runs/uav_exact_q_equivalence_acceleration_gate/`:

```text
baseline_equivalence.json
optimized_equivalence.json
seed20001_rollout.pt
```

## One-epoch performance gate

The only runtime gate was a fresh one-epoch run with 300 slots and seed 20001,
loading the provenance-verified canonical realization. No second seed was
started.

```text
BASELINE_UAV_EXACT_Q_RUNTIME_SECONDS = 274.722314600018
OPTIMIZED_UAV_EXACT_Q_RUNTIME_SECONDS = 219.315423099964
UAV_EXACT_Q_SPEEDUP = 1.25263563645863
UAV_EXACT_Q_RUNTIME_REDUCTION_PERCENT = 20.1683258168246

BASELINE_TOTAL_EPOCH_RUNTIME_SECONDS = 403.188096500002
OPTIMIZED_TOTAL_EPOCH_RUNTIME_SECONDS = 321.678691599984
TOTAL_EPOCH_SPEEDUP = 1.2533876412348
TOTAL_RUNTIME_REDUCTION_PERCENT = 20.2162230501311

OPTIMIZED_ROLLOUT_RUNTIME_SECONDS = 65.784007099981
OPTIMIZED_PPO_UPDATE_RUNTIME_SECONDS = 255.730568300001
OPTIMIZED_UAV_EXACT_Q_RUNTIME_FRACTION_OF_TOTAL = 0.681784118211624
OPTIMIZED_ESTIMATED_300_EPOCH_HOURS = 26.806557633332
RUNTIME_ESTIMATE_SCOPE = SINGLE_REALIZATION
CURRENT_RUNTIME_BOTTLENECK_AFTER_OPTIMIZATION = UAV_EXACT_Q
```

The baseline runtime is the existing seed-20001/epoch-1 artifact from
`runs/proposed_joint_short_stability_gate`. The optimized runtime is from
`runs/uav_exact_q_equivalence_acceleration_gate/optimized_one_epoch`.
Runtime estimates are preliminary single-realization estimates, not a
10-realization average.

The full optimized one-epoch metrics also recorded:

```text
UAV_EXACT_Q_BRANCH_REQUEST_COUNT = 11276
UAV_EXACT_Q_UNIQUE_BRANCH_COUNT = 10976
UAV_EXACT_Q_DUPLICATE_BRANCH_COUNT = 300
UAV_EXACT_Q_CACHE_HIT_COUNT = 300
EXECUTE_SLOT_CALL_COUNT = 10976
MATCHING_CALL_COUNT = 10976
KKT_CALL_COUNT = 949339  # unchanged KKT resource-cache requests
OPTIMIZED_ONE_EPOCH_KKT_RESOURCE_REQUEST_COUNT = 949339
```

## Invariants and safety checks

```text
UAV_OPERATION_TEACHER = FULL_INFORMATION_EXACT_Q_TEACHER
UAV_CANDIDATE_TEACHER = EXACT_Q_LISTWISE
ALGORITHM_SEMANTICS_CHANGED = NO
UAV_TEACHER_CHANGED = NO
CANDIDATE_SET_CHANGED = NO
CANDIDATE_ORDER_CHANGED = NO
MATCHING_CHANGED = NO
KKT_CHANGED = NO
REWARD_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO
CACHE_ACTION_Q_CRITIC_INITIALIZED = NO
CACHE_ACTION_Q_USED_FOR_POLICY_CREDIT = NO
CACHE_COUNTERFACTUAL_ENV_EXECUTIONS = 0
NAN_INF_DETECTED = NO
DEV_SEEDS_USED = NO
FINAL_TEST_SEEDS_USED = NO
```

The optimized run's finite checks for reward, return, `G`, loss, gradient,
returns, and advantages were all true. Seed 20001's manifest was verified as
`episode_manifest_v1`, with matching generation fingerprint and frozen artifact
hashes, before use. No realization was copied into the run directory.

## Recommendation and cleanup

```text
SAFE_ACCELERATION_AVAILABLE = YES
FORMAL_TRAINING_RECOMMENDATION = USE_THE_SLOT_LOCAL_MEMOIZATION_AFTER_THIS_GATE
TEMP_FILES_CLEANED = YES
GIT_DIFF_CHECK = PASS
REPORT_PATH = reports/uav_exact_q_equivalence_acceleration_gate.md
COMMITTED = NO
PUSHED = NO
```

The formal 300-epoch run should reuse the existing canonical realization pool
and fresh-initialize PPO as before. This gate does not authorize starting that
run.
