# UAV Operation Exact-Q Collapse Diagnostic

## Scope and provenance

- Branch: `main`; experiment: `proposed_joint`
- Source checkpoint: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\proposed_joint_actor_lr_3e_minus_4\latest_checkpoint.pt`
- Source checkpoint SHA-256: `55de8e3315f0f770e51717974f01fff10569a1d60bb275e60306f2099659809f`
- Source completed epochs: `12`
- Diagnostic realization: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v6_popart_actor1e4_lazy_episode_pool_20260911\episode_20013`; seed: `20013`
- Config SHA-256: `2769c45fda242a15a67c6e4ffe7fd748dd1525d28499fd89af64664b8aa55a57`
- Output directory: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\uav_operation_exact_q_collapse_diagnostic`
- Policy schema: `adaptive_joint_cache_uav_ppo`
- Control mode: `adaptive_joint_cache_uav_reconfiguration`
- Candidate objective: `EXACT_Q_LISTWISE`
- This is a frozen-policy rollout plus the existing exact operation-credit computation. No PPO update, reward change, action-space change, learning-rate change, KL change, Exact-Q definition change, candidate-proposal change, or Cache change was made.

## Direct answers

### A. Is MOVE physically feasible?

`UAV_OPERATION_MOVE_FEASIBLE_RATE = 100.0000%` over `600` UAV-head samples. The exact-credit path evaluated `11400` existing counterfactual environment branches; diagnostic aggregation added `0` branches.

### B. Does exact Q frequently prefer MOVE?

`MOVE_BETTER = 11.6667%`; `STAY_BETTER = 88.3333%`; `TIE = 0.0000%`. The same-state causal comparison is the exact `Q_MOVE - Q_STAY` gap; observational MOVE/STAY reward, delay, and energy means are not used as causal evidence.

### C. Does the policy still prefer STAY when exact Q prefers MOVE?

Yes/no is quantified without a qualitative threshold: `70` samples have `Q_MOVE > Q_STAY` while `P(MOVE) < 0.5`, a rate of `11.6667%`. The stricter rates are `11.6667%` for `P(MOVE)<0.10` and `11.6667%` for `P(MOVE)<0.01`.

### D. Is the bottleneck operation policy or conditional target policy?

The operation-level exact-Q regret is `0.00029445686` mean. For MOVE-feasible samples, `Q_MOVE^best > Q_STAY` occurs at `99.6667%`, while the conditional bottleneck (`Q_MOVE^best > Q_STAY` and `Q_MOVE^policy <= Q_STAY`) occurs at `88.0000%`. The target-position regret `Q_MOVE^best-Q_MOVE^policy` has mean `0.0255185427` and P90 `0.0366777938`.

### E. Is lower actor LR alone a plausible fix?

The diagnostic starts from a policy already collapsed after 12 epochs at actor LR `3e-4`, then freezes it. Therefore lower LR alone is not supported as a sufficient recovery explanation. The exact case classification below identifies whether the observed witness is operation-policy collapse, conditional target bottleneck, objective preference, or a mixed result; no algorithmic change is implemented here.

## Required metrics

| Metric | Value |
|---|---:|
| `UAV_P_MOVE_MEAN` | 0.2208% |
| `UAV_P_MOVE_MEDIAN` | 0.3020% |
| `UAV_MOVE_FEASIBLE_RATE` | 100.0000% |
| `UAV_OPERATION_MOVE_FEASIBLE_RATE` | 100.0000% |
| `UAV_OPERATION_Q_GAP_MEAN` | -0.00262912254 |
| `UAV_OPERATION_Q_GAP_MEDIAN` | -0.0034122467 |
| `UAV_OPERATION_Q_GAP_P10` | -0.00427246094 |
| `UAV_OPERATION_Q_GAP_P90` | 0.000152587891 |
| `UAV_OPERATION_MOVE_BETTER_FRACTION` | 11.6667% |
| `UAV_OPERATION_STAY_BETTER_FRACTION` | 88.3333% |
| `UAV_OPERATION_TIE_FRACTION` | 0.0000% |
| `UAV_OPERATION_POLICY_Q_ARGMAX_AGREEMENT_RATE` | 88.3333% |
| `UAV_OPERATION_POLICY_Q_ARGMAX_DISAGREEMENT_RATE` | 11.6667% |
| `UAV_MOVE_BETTER_BUT_POLICY_STAY_FRACTION` | 11.6667% |
| `UAV_MOVE_BETTER_BUT_PMOVE_LT_0_10_FRACTION` | 11.6667% |
| `UAV_MOVE_BETTER_BUT_PMOVE_LT_0_01_FRACTION` | 11.6667% |
| `UAV_OPERATION_EXPECTED_Q_CURRENT` | -144.950069 |
| `UAV_OPERATION_EXPECTED_EXACT_Q_MEAN` | -144.950069 |
| `UAV_OPERATION_ORACLE_EXACT_Q_MEAN` | -144.949775 |
| `UAV_OPERATION_EXACT_Q_REGRET_MEAN` | 0.00029445686 |
| `UAV_OPERATION_EXACT_Q_REGRET_MEDIAN` | 6.0230773e-06 |
| `UAV_OPERATION_EXACT_Q_REGRET_P90` | 0.000156267406 |
| `UAV_OPERATION_EXACT_Q_REGRET_MAX` | 0.105699937 |
| `UAV_MOVE_POLICY_Q_MEAN` | -144.952694 |
| `UAV_MOVE_BEST_Q_MEAN` | -144.927176 |
| `UAV_MOVE_BEST_MINUS_POLICY_Q_MEAN` | 0.0255185427 |
| `UAV_MOVE_POLICY_BETTER_THAN_STAY_FRACTION` | 11.6667% |
| `UAV_MOVE_BEST_BETTER_THAN_STAY_FRACTION` | 99.6667% |
| `UAV_MOVE_CONDITIONAL_POLICY_BOTTLENECK_FRACTION` | 88.0000% |
| `UAV_MOVE_TARGET_REGRET_MEAN` | 0.0255185427 |
| `UAV_MOVE_TARGET_REGRET_MEDIAN` | 0.0230827326 |
| `UAV_MOVE_TARGET_REGRET_P90` | 0.0366777938 |
| `UAV_MOVE_TARGET_REGRET_MAX` | 0.1887506 |
| `uav_exact_operation_move_advantage_mean` | n/a |
| `uav_exact_operation_stay_advantage_mean` | 3.71984653e-06 |

Policy/Q argmax rates use the stable operation-index tie break `STAY=0`; Q ties are also reported explicitly above. Conditional-target rates use only MOVE-feasible samples as their denominator. All operation-level rates use all transition × UAV-head samples.

## Per-UAV-head diagnostics

| Head | MOVE better | mean P(MOVE) | mean Q gap | mean operation regret | conditional bottleneck |
|---|---:|---:|---:|---:|---:|
| UAV_0 | 23.0000% | 0.3599% | -0.00114477396 | 0.000582568937 | 76.3333% |
| UAV_1 | 0.3333% | 0.0816% | -0.00411347111 | 6.34478413e-06 | 99.6667% |

## P(MOVE) buckets

| Bucket | Samples | mean Q gap | MOVE better | mean operation regret |
|---|---:|---:|---:|---:|
| P_MOVE_GE_0_50 | 0 | n/a | n/a | n/a |
| P_MOVE_0_10_TO_0_50 | 0 | n/a | n/a | n/a |
| P_MOVE_0_01_TO_0_10 | 3 | -0.00187647343 | 0.0000% | 6.4327458e-05 |
| P_MOVE_LT_0_01 | 597 | -0.00263290469 | 11.7253% | 0.00029561329 |

## Case classification

- `OPERATION_POLICY_COLLAPSE = SUPPORTED`
- `CONDITIONAL_TARGET_BOTTLENECK = SUPPORTED`
- `OBJECTIVE_FAVORS_STAY = NOT_SUPPORTED`
- `ROOT_CAUSE = MIXED`

Thresholds used only to make “materially” reproducible: operation collapse requires MOVE-better > 5% and Q-MOVE-better/P(MOVE)<0.01 > 1%; conditional bottleneck requires > 5% of MOVE-feasible samples; objective-favors-STAY requires STAY-better ≥ 95% and MOVE-best-better ≤ 5%. The raw fractions above remain the primary result.

## Execution and invariants

```text
SOURCE_CHECKPOINT = C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\proposed_joint_actor_lr_3e_minus_4\latest_checkpoint.pt
SOURCE_COMPLETED_EPOCHS = 12
DIAGNOSTIC_REALIZATION_SEED = 20013
POLICY_FROZEN = YES
ACTOR_OPTIMIZER_STEPS = 0
CRITIC_OPTIMIZER_STEPS = 0
ADDITIONAL_COUNTERFACTUAL_ENV_EXECUTIONS_FOR_DIAGNOSTICS = 0
ALGORITHM_CHANGED = NO
REWARD_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO
CACHE_CHANGED = NO
LEARNING_RATE_CHANGED = NO
TARGET_KL_CHANGED = NO
TRAINING_EXECUTED = NO
DEV_SEEDS_USED = NO
FINAL_TEST_SEEDS_USED = NO
ROLLOUT_TRANSITIONS = 300
EXISTING_EXACT_OPERATION_CREDIT_ENV_EXECUTIONS = 11400
REPORT_PATH = C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\uav_operation_exact_q_collapse_diagnostic.md
TEMP_FILES_CLEANED = YES
COMMITTED = NO
PUSHED = NO
```

`NEXT_RECOMMENDED_CHANGE`: First separate the conditional target bottleneck from operation-policy collapse; then study replacing sampled-action UAV STAY/MOVE PPO with a full-information exact-Q operation teacher. Neither change is implemented in this task.
