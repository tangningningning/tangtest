# UAV branch KL protection one-epoch stability gate

## 1. Scope

This gate validates the optimizer-step-level UAV branch KL protection on the
current uncommitted `proposed_joint` mainline. It runs exactly one fresh outer
epoch through the production rollout, exact-credit, and PPO update pipeline.
No algorithm redesign, hyperparameter change, second run, long training,
final-test seed, or proposal redesign was performed.

- Branch: `main`
- HEAD: `0296816`
- Working tree: preserved; pre-existing uncommitted changes were not reverted
- Experiment ID: `proposed_joint`
- Policy schema: `adaptive_joint_cache_uav_ppo`
- Control mode: `adaptive_joint_cache_uav_reconfiguration`
- Config: `configs/proposed/system_config.toml`
- Candidate objective: `EXACT_Q_LISTWISE`
- Cache high-level actions: `KEEP / UPDATE`
- UAV high-level actions: `STAY / MOVE`
- Target KL: `0.10`

Run artifacts:

`runs/uav_branch_kl_protection_one_epoch_stability_gate_20260920`

## 2. Baseline comparison

The previous comparison run reported:

| Metric | Previous | Current protection-specific result |
| --- | ---: | ---: |
| UAV branch KL | 0.625867292 | 0.157252610 |
| Cache branch KL | 0.087568395 | 0.088311124 |

The current production aggregate `uav_approx_kl`, which is the like-for-like
update metric used by the previous JSONL record, is `0.149322228`; this is a
76.14% reduction from `0.625867292`. The protection-specific full-rollout
post-step maximum/final value is `0.157252610`, 74.87% below the previous
reference.

## 3. Fresh initialization and protocol

- `FRESH_INITIALIZATION = YES`
- `OLD_CHECKPOINT_LOADED = NO`
- `RESUME = NO`
- Training realization: `episode_20001`
- Training realization seed: `20001`
- Policy/random seed: `4401`
- Episode slots: `300`
- Outer epochs executed: `1`
- PPO update epochs: `4`
- Minibatches per PPO epoch: `10`
- Final-test seeds used: `NO`
- Formal protocol epochs currently configured: `222`
- Formal protocol seed end currently configured: `20222`

The run used the existing realization pool and loaded `episode_20001`. The
fresh run directory did not exist before launch, and the command did not pass
`--resume`. No prior checkpoint was loaded.

## 4. UAV KL protection control flow

The new full-rollout check reused the production UAV operation log-probability
delta on the same frozen rollout. It was evaluated immediately after each
UAV-active actor optimizer step.

| PPO epoch | Minibatch | Full-rollout UAV KL after actor step | Stop |
| ---: | ---: | ---: | :---: |
| 1 | 1 | 0.055336781 | No |
| 1 | 2 | 0.157252610 | Yes |

After the second actor step crossed `target_kl=0.10`, `uav_active` became
false before the next minibatch. The triggering optimizer step was retained;
there was no rollback. The next 38 UAV branch minibatches executed critic
logic but no UAV actor branch update.

```text
UAV_FULL_ROLLOUT_KL_CHECK_COUNT = 2
UAV_KL_STOP_TRIGGERED = YES
UAV_KL_STOP_AFTER_ACTOR_STEP = YES
UAV_KL_AT_STOP = 0.157252610
UAV_BRANCH_KL_FINAL = 0.157252610
UAV_BRANCH_KL_MAX_OBSERVED = 0.157252610
UAV_ACTOR_STEPS_EXECUTED = 2
UAV_ACTOR_STEPS_SUPPRESSED = 38
UAV_UPDATES_AFTER_KL_STOP = 0
```

The final full-rollout value is the last post-step check. It remains unchanged
after the stop because no later UAV actor update occurred; Cache and critic
parameters are in separate groups from the proposed UAV encoder/head group.

Cache and critic continued normally:

```text
CACHE_ACTOR_STEPS_EXECUTED = 30
CACHE_ACTOR_STEPS_SUPPRESSED = 10
CRITIC_STEPS_EXECUTED = 40
```

The Cache branch stopped only at its existing epoch-level gate after PPO epoch
3. The UAV stop did not terminate the outer update.

## 5. UAV target scorer preservation

The target-position listwise loss and Exact-Q tensors were unchanged and were
not recomputed through a second environment replay. Before/after metrics reuse
the same candidate features, masks, Q labels, and candidate ordering.

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Target scorer parameter delta norm | — | — | 0.112195172 |
| Top-1 Exact-Q accuracy | 0.315000000 | 0.316666667 | +0.001666667 |
| Mean Exact-Q regret | 0.000399506 | 0.000385187 | -0.000014319 |
| Expected Exact-Q | -6.859550919 | -6.859543235 | +0.000007684 |
| Target policy KL(old\|new) | — | — | 0.000015761 |

```text
UAV_OPERATION_POLICY_LOSS = -0.006483994
UAV_TARGET_LISTWISE_LOSS = 0.302436470
UAV_TARGET_SCORER_PARAMETER_DELTA_NORM = 0.112195172
```

All values are finite. The scorer received a nonzero update and its expected
Exact-Q did not decrease, so target learning was preserved for this gate.

## 6. Cache branch non-regression

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Target scorer parameter delta norm | — | — | 1.021137607 |
| Top-1 Exact-Q accuracy | 0.052222222 | 0.193703704 | +0.141481481 |
| Mean Exact-Q regret | 9.845818061 | 7.311057133 | -2.534760928 |
| Expected Exact-Q | -6.536207944 | -6.272911326 | +0.263296618 |
| Target policy KL(old\|new) | — | — | 0.016957433 |

```text
CACHE_TRIGGER_POLICY_LOSS = 0.064346604
CACHE_TARGET_LISTWISE_LOSS = 0.487924457
CACHE_TARGET_SCORER_PARAMETER_DELTA_NORM = 1.021137607
CACHE_BRANCH_KL_FINAL = 0.088311124
```

The Cache actor remained active while the UAV branch was suppressed and the
Cache candidate scorer continued to update.

## 7. Operation diagnostics

```text
CACHE_KEEP_SAMPLED_RATE = 0.502962963
CACHE_UPDATE_SAMPLED_RATE = 0.497037037
CACHE_ACTUAL_UPDATE_RATE = 0.497037037

UAV_STAY_SAMPLED_RATE = 0.506666667
UAV_MOVE_SAMPLED_RATE = 0.493333333
UAV_ACTUAL_MOVE_RATE = 0.493333333
```

Persisted rollout policy probabilities were:

```text
UAV_STAY_PROB_MEAN = 0.531925294
UAV_MOVE_PROB_MEAN = 0.468074707
```

## 8. Runtime and numerical stability

```text
RETURN = -2063.866768671
ROLLOUT_RUNTIME_S = 78.818240
CACHE_EXACT_Q_RUNTIME_S = 1028.903202
UAV_EXACT_Q_RUNTIME_S = 253.899466
PPO_UPDATE_RUNTIME_S = 1350.214697
PPO_UPDATE_REMAINDER_S = 67.412029
UAV_KL_EVALUATION_RUNTIME_S = NOT_SEPARATELY_INSTRUMENTED
TOTAL_EPOCH_RUNTIME_S = 1429.230045
```

`nan_inf_detected=false`, reward/return/loss/gradient/advantage finite checks
all passed, and `gradient_finite=true`. The run completed with one outer epoch
and no numerical exception.

## 9. Gate conclusion

```text
UAV_KL_PROTECTION_CONTROL_FLOW = PASS
UAV_KL_PROTECTION_EFFECTIVENESS = SUPPORTED
UAV_TARGET_LEARNING_PRESERVED = YES
JOINT_NUMERICAL_STABILITY = PASS
READY_FOR_LONG_TRAINING = YES
```

The optimizer-step protection worked as intended: the first post-step KL below
the threshold did not stop the branch, the next post-step KL crossed `0.10`,
and every subsequent UAV actor branch update in the outer update was
suppressed. The retained overshoot was `0.157252610`, rather than the previous
`0.625867292` aggregate branch KL. This is a one-epoch stability result, not a
long-training performance claim.

## 10. Next action

The mainline is eligible for a separately authorized longer training run. This
task itself does not start that run.

## 11. Required final fields

```text
BRANCH = main
HEAD = 0296816
WORKING_TREE_PRESERVED = YES

EXPERIMENT_ID = proposed_joint
POLICY_SCHEMA = adaptive_joint_cache_uav_ppo
CONTROL_MODE = adaptive_joint_cache_uav_reconfiguration

FRESH_INITIALIZATION = YES
OLD_CHECKPOINT_LOADED = NO
TRAIN_REALIZATION_SEED = 20001
POLICY_RANDOM_SEED = 4401

OUTER_EPOCHS_EXECUTED = 1
PPO_UPDATE_EPOCHS = 4
TARGET_KL = 0.10

PREVIOUS_UAV_BRANCH_KL = 0.625867
CURRENT_UAV_BRANCH_KL_FINAL = 0.157252610
CURRENT_UAV_BRANCH_KL_MAX = 0.157252610

UAV_FULL_ROLLOUT_KL_CHECK_COUNT = 2
UAV_KL_STOP_TRIGGERED = YES
UAV_KL_STOP_AFTER_ACTOR_STEP = YES
UAV_KL_AT_STOP = 0.157252610

UAV_ACTOR_STEPS_EXECUTED = 2
UAV_ACTOR_STEPS_SUPPRESSED = 38
UAV_UPDATES_AFTER_KL_STOP = 0

CACHE_ACTOR_STEPS_EXECUTED = 30
CACHE_ACTOR_STEPS_SUPPRESSED = 10
CRITIC_STEPS_EXECUTED = 40

CACHE_BRANCH_KL_FINAL = 0.088311124

UAV_OPERATION_POLICY_LOSS = -0.006483994
UAV_TARGET_LISTWISE_LOSS = 0.302436470
UAV_TARGET_SCORER_PARAMETER_DELTA_NORM = 0.112195172

UAV_TOP1_EXACT_Q_ACCURACY_BEFORE = 0.315000000
UAV_TOP1_EXACT_Q_ACCURACY_AFTER = 0.316666667
UAV_MEAN_EXACT_Q_REGRET_BEFORE = 0.000399506
UAV_MEAN_EXACT_Q_REGRET_AFTER = 0.000385187
UAV_EXPECTED_EXACT_Q_BEFORE = -6.859550919
UAV_EXPECTED_EXACT_Q_AFTER = -6.859543235
UAV_EXPECTED_EXACT_Q_DELTA = +0.000007684
UAV_TARGET_POLICY_KL_OLD_NEW = 0.000015761

CACHE_TRIGGER_POLICY_LOSS = 0.064346604
CACHE_TARGET_LISTWISE_LOSS = 0.487924457
CACHE_TARGET_SCORER_PARAMETER_DELTA_NORM = 1.021137607
CACHE_EXPECTED_EXACT_Q_BEFORE = -6.536207944
CACHE_EXPECTED_EXACT_Q_AFTER = -6.272911326
CACHE_EXPECTED_EXACT_Q_DELTA = +0.263296618

CACHE_KEEP_SAMPLED_RATE = 0.502962963
CACHE_UPDATE_SAMPLED_RATE = 0.497037037
CACHE_ACTUAL_UPDATE_RATE = 0.497037037
UAV_STAY_SAMPLED_RATE = 0.506666667
UAV_MOVE_SAMPLED_RATE = 0.493333333
UAV_ACTUAL_MOVE_RATE = 0.493333333

RETURN = -2063.866768671
ROLLOUT_RUNTIME_S = 78.818240
CACHE_EXACT_Q_RUNTIME_S = 1028.903202
UAV_EXACT_Q_RUNTIME_S = 253.899466
PPO_UPDATE_RUNTIME_S = 1350.214697
UAV_KL_EVALUATION_RUNTIME_S = NOT_SEPARATELY_INSTRUMENTED
TOTAL_EPOCH_RUNTIME_S = 1429.230045

UAV_KL_PROTECTION_CONTROL_FLOW = PASS
UAV_KL_PROTECTION_EFFECTIVENESS = SUPPORTED
UAV_TARGET_LEARNING_PRESERVED = YES
JOINT_NUMERICAL_STABILITY = PASS
FORMAL_PROTOCOL_EPOCHS_CURRENT = 222
FORMAL_PROTOCOL_SEED_END_CURRENT = 20222
READY_FOR_LONG_TRAINING = YES
NEXT_ACTION = SEPARATELY_AUTHORIZE LONG TRAINING; DO NOT START HERE

TRAINING_EXECUTED = YES
FINAL_TEST_SEEDS_USED = NO
REPORT_PATH = reports/uav_branch_kl_protection_one_epoch_stability_gate.md
GIT_DIFF_CHECK = PASS
TEMP_FILES_CLEANED = NO_NEW_TEMP_FILES_GENERATED
COMMITTED = NO
PUSHED = NO
```
