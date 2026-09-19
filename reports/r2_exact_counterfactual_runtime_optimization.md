# R2 exact counterfactual runtime optimization

- STAGE = `R2_EXACT_COUNTERFACTUAL_RUNTIME_OPTIMIZATION`
- Repository = `C:\Users\tangning\Desktop\code\adaptive_cache_mec`
- CURRENT_HEAD = `525404ce0d595a56e1940cb0135b779d8141cfd4`
- MAINLINE_SCHEMA = `adaptive_joint_cache_replica_aware_uav_operation_recovery_r2`
- MAINLINE_CONFIG = `configs/proposed/system_config_ppo_recovery_r2_6e4.toml`

## Scope and hotspot

The active R2 path was traced only through
`PPOV2Trainer._compute_r2_operation_credit()` and the formal
`slot_environment.execute_slot()` path.  Before this change, every exact
STAY/MOVE branch executed Matching/KKT and then called the policy critic once
for its own next observation.  Legal MOVE enumeration, rollout-policy
candidate probabilities, sampled Cache action, non-target UAV actions, and
conditional metadata were already computed outside the candidate branch loop.
The code does not deep-copy a whole environment object, so no unproven
environment snapshot optimization was applied.

## Exact optimization applied

`src/ppo_simple/trainer.py` now:

1. executes every STAY and every legal MOVE candidate through the unchanged
   `execute_slot -> Matching/KKT -> reward/state` path;
2. collects the exact next observations in the stable order
   `STAY, MOVE_0, ..., MOVE_N`;
3. evaluates the independent critic for all MOVE branches in one batch;
4. evaluates STAY once through the historical scalar critic path and uses that
   scalar result for the first branch, preserving the stricter PPO-side
   numerical gate; and
5. applies the unchanged
   `Q = reward + gamma * (1-done) * V(next_state)` formula and unchanged
   `Q_move = sum(pi_old * Q_candidate)` reduction.

The counterfactual target remains precomputed once before PPO epochs and
minibatches.  No candidate is sampled, pruned, clustered, approximated, or
reused across different physical branches.

```text
CRITIC_INFERENCE_MODE_BEFORE = PER_BRANCH
CRITIC_INFERENCE_MODE_AFTER  = BATCHED_EXACT_MOVE_PLUS_SCALAR_STAY_EXACT
CURRENT_STATE_POLICY_RECOMPUTATION_REMOVED = NO (already once outside branch loop)
CANDIDATE_METADATA_RECOMPUTATION_REMOVED = NO (already once outside branch loop)
ENVIRONMENT_COPY_OPTIMIZED = NO (no safe whole-environment snapshot proof)
CF_TARGET_RECOMPUTED_PER_PPO_EPOCH = NO
CF_TARGET_RECOMPUTED_PER_MINIBATCH = NO
```

## Fixed equivalence micro-gate

The fixed witness used frozen `episode_20001`, origins/slots 80–83, and both
UAV heads: 8 target-UAV cases total.  The pre-optimization reference was the
in-memory per-branch critic implementation; no large temporary artifact was
written.  Runtime was repeated three times for each implementation.

```text
MICROBENCH_CASE_COUNT = 8
BASELINE_RUNTIMES_SECONDS = [6.687729, 6.200526, 5.947772]
OPTIMIZED_RUNTIMES_SECONDS = [6.030428, 5.549946, 6.355737]
BASELINE_MEDIAN_SECONDS = 6.200526
OPTIMIZED_MEDIAN_SECONDS = 6.030428
MICROBENCH_SPEEDUP_X = 1.028207

LEGAL_CANDIDATE_SET_IDENTICAL = YES
LEGAL_CANDIDATE_ORDER_IDENTICAL = YES
CANDIDATE_PROB_MAX_ABS_DIFF = 0.0
Q_STAY_MAX_ABS_DIFF = 0.0
Q_CANDIDATE_MAX_ABS_DIFF = 9.313225746154785e-08
Q_MOVE_MAX_ABS_DIFF = 0.0
B_OP_MAX_ABS_DIFF = 9.5367431640625e-07
RAW_A_OP_MAX_ABS_DIFF = 1.4901161193847656e-08
SCALED_A_OP_MAX_ABS_DIFF = 5.960464477539062e-07

BRANCH_REWARD_MAX_ABS_DIFF = 0.0
NEXT_CACHE_STATE_IDENTICAL = YES
NEXT_UAV_POSITION_MAX_ABS_DIFF = 0.0
NEXT_OBSERVATION_MAX_ABS_DIFF = 0.0
MATCHING_KKT_RELEVANT_OUTPUTS_IDENTICAL = YES

OPERATION_RATIO_MAX_ABS_DIFF = 0.0
OPERATION_LOSS_MAX_ABS_DIFF = 4.470348358154297e-08
CANDIDATE_LOSS_MAX_ABS_DIFF = 0.0
CACHE_LOSS_MAX_ABS_DIFF = 0.0
TOTAL_ACTOR_LOSS_MAX_ABS_DIFF = 1.4901161193847656e-08

EXACT_EQUIVALENCE = YES
```

The operation-loss comparison used a fixed synthetic minibatch of 4 selected
rollout transitions and performed no optimizer step.

## Full bounded gate

After the micro-gate passed, the one permitted bounded run used frozen
`episode_20001`, seed `4401`, 300 slots, and one normal PPO update (4 PPO
epochs, 10 minibatches, 40 optimizer steps).  It loaded the existing episode;
no episode was generated.

The standard episode log records 79.5716667 UAV candidate entries per head,
including the STAY entry.  The exact R2 counterfactual loop filters STAY and
therefore evaluates 78.5716667 legal MOVE candidates per target head.  With
600 target heads, its exact branch-count identity is
`600 * (1 + 78.5716667) = 47,743`; no legal candidate was skipped.

```text
FULL_BOUNDED_GATE_EXECUTED = YES
COUNTERFACTUAL_ENV_EXECUTIONS = 47743
MEAN_MOVE_CANDIDATE_COUNT = 78.5716666666667
ROLLOUT_RUNTIME_SECONDS = 65.72994669998297
COUNTERFACTUAL_CREDIT_RUNTIME_SECONDS = 671.1737895999686
GATE_RUNTIME_SECONDS = 737.1232756000245

PREVIOUS_COUNTERFACTUAL_CREDIT_RUNTIME_SECONDS = 1207.5244373999885
PREVIOUS_GATE_RUNTIME_SECONDS = 1303.246635699994
COUNTERFACTUAL_RUNTIME_SPEEDUP_X = 1.799123
FULL_GATE_SPEEDUP_X = 1.768017
```

The current trainer's standard JSONL logger persists the complete update-path
runtime but not the inner R2 `credit_runtime_seconds` field.  The value above
is therefore the measured bounded update-path time (a conservative upper
bound containing the short PPO update bookkeeping); the micro-gate is the
isolated before/after runtime measurement.

The bounded run completed with finite rewards/losses/gradients and unchanged
R2 behavior flags.  The observed episode return was `-989.049157118993`, with
`300` transitions and `40` optimizer steps.

## Invariants

```text
CACHE_ADV_SOURCE = GLOBAL_GAE
UAV_OPERATION_ADV_SOURCE = COUNTERFACTUAL_ONE_STEP_TD
UAV_OPERATION_RATIO_GRANULARITY = OPERATION_ONLY
UAV_CANDIDATE_ADV_SOURCE = RECOVERY_R1_ORIGINAL
OP_ADV_NORMALIZATION = STD_ONLY
OP_ADV_CENTERING = NO

ALGORITHM_CHANGED = NO
COUNTERFACTUAL_FORMULA_CHANGED = NO
LEGAL_CANDIDATES_CHANGED = NO
REWARD_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO
MATCHING_KKT_CHANGED = NO
MOVEMENT_COST_CHANGED = NO
CANDIDATE_CREDIT_CHANGED = NO
POPART_CHANGED = NO
GAE_CHANGED = NO
PPO_LOSS_CHANGED = NO
OPTIMIZER_CHANGED = NO
```

## Checks and cleanup

```text
COMPILE_CHECK = PASS
R2_CONFIG_LOAD_CHECK = PASS
TRAINER_HELP_CHECK = PASS
EVALUATOR_HELP_CHECK = PASS
EQUIVALENCE_GATE = PASS
GIT_DIFF_CHECK = PASS (after final validation)
TEMP_FILES_REMOVED = YES
TEMP_GATE_DIRECTORY_REMOVED = YES
PYTHON_BYTECODE_REMAINING_IN_SRC_SCRIPTS = 0

FORMAL_TRAINING_EXECUTED = NO
EPISODE_REGENERATION = NO
SUMO_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
FORMAL_EVALUATION_EXECUTED = NO
COMMITTED = NO
PUSHED = NO
```

Only `src/ppo_simple/trainer.py` and this report were changed for this stage;
all pre-existing worktree changes and all prior reports were preserved.
