# Stage2B Cache trigger TD-Q implementation

## Scope and safety

This change implements only the Cache per-head trigger credit requested for
Stage2B.  No PPO update, optimizer step, checkpoint write, run continuation,
or new training run was executed.  The benchmark used the existing train-only
`episode_20107` realization and kept all generated data and existing run
artifacts read-only.

The existing `runs/stage1b_cross_branch_short_gate/run_config.toml` and its
checkpoint were not modified.  The source proposed configuration now selects
the new trigger source; old configurations that still select `GLOBAL_GAE`
remain accepted, and the checkpoint schema/version and policy state layout
were not changed.

## Static implementation result

```text
CACHE_TRIGGER_CREDIT_OLD = GLOBAL_GAE
CACHE_TRIGGER_CREDIT_NEW = ONE_STEP_TD_Q_DIFFERENCE
CACHE_CANDIDATE_CREDIT = GLOBAL_GAE
BOOTSTRAP_SIGN = PLUS
TRIGGER_CANDIDATE_LOSS_SEPARATED = YES
TDQ_CENTERING_USED = NO
TDQ_SCALE_NORMALIZATION_USED = YES
```

For each valid Cache head, the new detached evaluator reuses the existing
`_execute_cache_branch` and `_branch_q_values` path.  It computes

```text
Q_actual = R_actual + gamma * V(s'_actual)
Q_cf     = R_cf     + gamma * V(s'_cf)
D_Q      = Q_actual - Q_cf
```

An actual `UPDATE` is compared with `KEEP`.  An actual `KEEP` is compared with
the highest-probability legal `UPDATE` candidate from the current frozen
policy.  The candidate branch is selected from the legal candidate set but is
not given a candidate-level Q teacher; only one counterfactual environment
branch is executed for that head.  Invalid/padded candidates cannot become a
KEEP reference because the reference is selected from the legal candidate
objects returned by the existing branch-state evaluator.

Valid `D_Q` values use `std(D_Q_valid) + 1e-8` as a scale.  They are not
centered.  The trigger loss uses the per-head trigger log-probability ratio and
PPO clipping.  The candidate loss is a separate selected-candidate PPO
surrogate using the existing normalized global GAE; `D_Q` is not broadcast to
the candidate scorer.  The policy already exposes separate trigger and
candidate log-probabilities, so no combined trigger-plus-candidate credit is
needed.

The source configuration is explicit:

```toml
cache_trigger_advantage_source = "ONE_STEP_TD_Q_DIFFERENCE"
cache_candidate_advantage_source = "GLOBAL_GAE"
```

The Cache ratio remains `PER_CACHE_HEAD`.  UAV Exact-Q/listwise teacher,
candidate confidence scale `0.02`, candidate KL gate, reward, objective,
Matching, KKT, learning rates, Adam optimizer, and episode protocol are
unchanged.

## Required diagnostics

When the TD-Q source is active, `update_metrics.jsonl` receives:

```text
CACHE_TRIGGER_TDQ_RAW_MEAN
CACHE_TRIGGER_TDQ_RAW_STD
CACHE_TRIGGER_TDQ_RAW_P10
CACHE_TRIGGER_TDQ_RAW_P50
CACHE_TRIGGER_TDQ_RAW_P90
CACHE_TRIGGER_TDQ_RAW_ABS_MAX
CACHE_TRIGGER_TDQ_SCALE
CACHE_TRIGGER_TDQ_NORMALIZED_ABS_P90
CACHE_TRIGGER_TDQ_NORMALIZED_ABS_MAX
ONE_STEP_TDQ_BRANCHES_EXECUTED
ONE_STEP_TDQ_RUNTIME_SECONDS
SECONDS_PER_CACHE_HEAD
PROJECTED_300_SLOT_RUNTIME_SECONDS
```

## Read-only benchmark

```text
EPISODE = 20107 (train-only)
TRANSITIONS_ANALYZED = 20
CACHE_HEADS = 9
ONE_STEP_TDQ_BRANCHES_EXECUTED = 180
ONE_STEP_TDQ_RUNTIME_SECONDS = 3.963073
SECONDS_PER_CACHE_HEAD = 0.0220171
PROJECTED_300_SLOT_RUNTIME_SECONDS = 59.4461
TDQ_RAW_MEAN = -1.54742
TDQ_RAW_STD = 6.35042
TDQ_SCALE = 6.35042
```

The projection is a linear extrapolation from the measured per-head branch
runtime.  It is not a performance claim and does not include the ordinary PPO
forward/backward/update cost.

```text
ONE_STEP_TDQ_FORMAL_TRAINING_FEASIBILITY = BORDERLINE
```

The result is classified as borderline because the extra one-step
counterfactual work alone extrapolates to roughly one minute for one
300-slot, 9-head rollout.  No formal training was started to test this
estimate.

## Synthetic validation

The checks were executed in memory with no environment rollout and no
optimizer step.

```text
SYNTHETIC_CASE_1 = PASS  # Q_actual > Q_cf gives positive D_Q and normalized credit
SYNTHETIC_CASE_2 = PASS  # Q_actual < Q_cf gives negative D_Q
SYNTHETIC_CASE_3 = PASS  # equal Q values give zero D_Q
SYNTHETIC_CASE_4 = PASS  # std scaling preserves the sign
SYNTHETIC_CASE_5 = PASS  # opposite head credits give opposite trigger gradients
SYNTHETIC_CASE_6 = PASS  # UPDATE uses TD-Q trigger credit; candidate remains GLOBAL_GAE
SYNTHETIC_CASE_7 = PASS  # KEEP has no candidate surrogate term
SYNTHETIC_CASE_8 = PASS  # padded/invalid candidates are not KEEP references
```

## Final status

```text
REWARD_CHANGED = NO
OBJECTIVE_CHANGED = NO
LR_CHANGED = NO
OPTIMIZER_CHANGED = NO
UAV_CHANGED = NO
MATCHING_CHANGED = NO
KKT_CHANGED = NO

CACHE_RATIO_CHANGED = NO
CACHE_CREDIT_CHANGED = NO  # Cache candidate/global credit remains GLOBAL_GAE; only requested trigger source changes
UAV_TEACHER_CHANGED = NO
CONFIDENCE_SCALE_CHANGED = NO

TRAINING_EXECUTED = NO
COMMIT = NO
PUSH = NO

STAGE2B_READY_FOR_SHORT_TRAIN = YES
```

`STAGE2B_READY_FOR_SHORT_TRAIN = YES` means the source path is statically
validated, the trigger/candidate losses are separated, and all requested
in-memory checks pass.  It does not authorize or imply that a short training
run was started; the measured runtime caveat above should be considered when
planning that run.
