# PROPOSED_JOINT_SHORT_STABILITY_GATE

## Gate conclusion

```text
SHORT_STABILITY_GATE = PASS
ALGORITHM_READY_TO_FREEZE = YES
```

The run completed `10/10` fresh outer epochs, each with 300 slots and one
provenance-verified realization. No formal 300-epoch run was started, no
parameter sweep was run, and no algorithm or hyperparameter was changed for
this gate.

## Configuration and fixed realization pool

```text
POLICY_SCHEMA = adaptive_joint_cache_uav_ppo
CONTROL_MODE = adaptive_joint_cache_uav_reconfiguration
DATA_PROTOCOL = full_episode_randomized
ROLLOUT_STEPS_PER_EPOCH = 300
CONDA_ENV = adaptive_cache_mec
FRESH_INITIALIZATION = YES
SEEDS_USED = 20001..20010
TRAINING_REALIZATION_POOL = data/episode_pools/proposed_joint_training
REALIZATION_REUSED_COUNT = 1
REALIZATION_GENERATED_COUNT = 9
SEED_20001_PATH = data/episode_pools/proposed_joint_training/episode_20001
SEED_20010_PATH = data/episode_pools/proposed_joint_training/episode_20010
ALL_REALIZATIONS_IN_CANONICAL_POOL = YES
```

Seed 20001 was reused only after provenance verification. Seeds 20002–20010
were generated directly into the canonical pool by the formal generator. A
full existing-realization validation pass returned `SKIP existing valid
realization` for all ten seeds. No realization was generated in the run
directory, temporary pool, or the old `ppo_v6_*episode_pool*` locations.
No provenance-less old realization was used; the old pool was left untouched.

Every realization has `episode_manifest.json`, schema
`episode_manifest_v1`, 300 slots, all required artifacts and hashes, and the
same verified generation identity:

```text
GENERATION_CONFIG_FINGERPRINT = 9ad3c6179be6cea67a0c95695f0cc2802e7570274c9be5446e51d996d5f7045d
SOURCE_GIT_SHA = c82f16b76b4698d1a7c2982c3785fcf167f3f0ef
SOURCE_GIT_DIRTY = true
ALL_EPISODE_PROVENANCE_VERIFIED = YES
```

The canonical pool is ready for later reuse by fresh formal training. This
gate generated only 20001–20010; 20011–20300 remain to be generated when the
user authorizes formal 300-epoch training.

## Per-epoch metrics

`cache_actor_gradient_norm` and `uav_actor_gradient_norm` below are the logged
branch-head gradient norms; encoder gradient norms were also finite in every
epoch. `cache_actor_delta_logged` is the existing logged Cache candidate-scorer
actor movement component. `uav_actor_delta` combines the logged UAV candidate
scorer, operation-head, and encoder movement components.

| epoch | realization seed | provenance | return | cache change | UAV movement | Cache entropy | UAV entropy | Cache KL | UAV operation KL | Cache grad | UAV grad | Critic grad | Cache actor delta logged | UAV actor delta | Critic EV | total s |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20001 | YES | -2063.866769 | 0.497037 | 0.493333 | 17.851567 | 4.222061 | 0.028374 | 0.003072 | 0.421142 | 0.043504 | 1.185429 | 0.328959 | 0.989212 | -0.000277 | 403.188 |
| 2 | 20002 | YES | -2920.606437 | 0.512963 | 0.506667 | 17.989125 | 4.276685 | 0.029438 | 0.001288 | 0.388852 | 0.039233 | 2.106357 | 0.290052 | 0.848619 | 0.005810 | 676.230 |
| 3 | 20003 | YES | 382.286713 | 0.519630 | 0.553333 | 17.997476 | 4.574372 | 0.010831 | 0.001396 | 0.385939 | 0.032338 | 0.516668 | 0.288570 | 0.705799 | 0.037451 | 789.486 |
| 4 | 20004 | YES | -2570.525671 | 0.541481 | 0.518333 | 18.551916 | 4.361376 | 0.028819 | 0.000801 | 0.438365 | 0.031566 | 6.396914 | 0.282789 | 0.618524 | 0.068057 | 709.777 |
| 5 | 20005 | YES | 46.604898 | 0.529259 | 0.503333 | 18.181858 | 4.112847 | 0.015651 | 0.111305 | 0.453578 | 0.063105 | 1.329762 | 0.312986 | 1.915645 | 0.196068 | 548.817 |
| 6 | 20006 | YES | -294.217921 | 0.525926 | 0.670000 | 18.192325 | 5.059332 | 0.005788 | 0.002958 | 0.434027 | 0.041998 | 0.202794 | 0.307426 | 1.166301 | -0.175648 | 748.524 |
| 7 | 20007 | YES | 113.426249 | 0.528148 | 0.663333 | 18.208644 | 5.011791 | 0.022765 | 0.003138 | 0.454554 | 0.032965 | 0.409155 | 0.341844 | 0.691836 | 0.254286 | 851.325 |
| 8 | 20008 | YES | 98.816859 | 0.523704 | 0.683333 | 18.077661 | 5.126008 | 0.010521 | 0.001734 | 0.463027 | 0.034187 | 0.214704 | 0.370191 | 0.838655 | 0.643578 | 694.997 |
| 9 | 20009 | YES | -684.432929 | 0.525926 | 0.691667 | 18.236675 | 5.205527 | 0.017458 | 0.003467 | 0.388329 | 0.041054 | 0.544532 | 0.309238 | 0.728863 | 0.177704 | 549.011 |
| 10 | 20010 | YES | -2615.057807 | 0.534074 | 0.700000 | 18.285025 | 5.123915 | 0.022995 | 0.013698 | 0.435140 | 0.043777 | 3.343153 | 0.365577 | 0.639057 | 0.064166 | 593.412 |

All ten epochs had `gradient_finite=true`, finite reward/return/loss/return/
advantage checks, and `CACHE_COUNTERFACTUAL_ENV_EXECUTIONS=0`. The UAV Exact-Q
teacher performed its unchanged counterfactual evaluations; those are not
Cache production credit.

## Cache stability and credit path

```text
CACHE_POLICY_CREDIT_SOURCE = ACTUAL_ROLLOUT_GAE
CACHE_ACTION_Q_CRITIC_INITIALIZED = NO
CACHE_ACTION_Q_USED_FOR_POLICY_CREDIT = NO
CACHE_COUNTERFACTUAL_ENV_EXECUTIONS = 0  # every epoch
CACHE_BEHAVIORAL_COLLAPSE = NO
```

Cache change rates were in `[0.497037, 0.541481]`; the final three were
`0.523704, 0.525926, 0.534074`. They satisfy neither the `<=0.01` nor the
`>=0.99` three-epoch collapse rule. Cache KL mean was
`0.019263883698198033`, maximum `0.029438153485243746`.

The logged Cache candidate-scorer movement accumulated to
`3.197631362625805 > 0`. Reconstructing the fresh seed-4401 policy against the
final checkpoint gives a positive full Cache actor net delta of
`1.837454590775567` over `cache_encoder`, `cache_trigger_head`, and
`cache_candidate_scorer`.

## UAV stability and unchanged teacher

```text
UAV_OPERATION_TEACHER = FULL_INFORMATION_EXACT_Q_TEACHER
UAV_CANDIDATE_TEACHER = EXACT_Q_LISTWISE
UAV_TEACHER_UNCHANGED = YES
UAV_BEHAVIORAL_COLLAPSE = NO
UAV_OPERATION_KL_MEAN = 0.014285616448614747
UAV_OPERATION_KL_MAX = 0.11130460351705551
```

UAV movement rates were in `[0.493333, 0.700000]`; the final three were
`0.683333, 0.691667, 0.700000`, so the three-epoch collapse rule is not met.
The logged UAV branch movement accumulated to `9.142512037498781 > 0`; the
fresh-policy-to-final-checkpoint full UAV actor net delta was
`3.700772404160397`.

The maximum UAV operation KL was `0.11130460351705551` in epoch 5. The existing
operation KL protection engaged (`uav_stop_epoch=1`) and preserved the
candidate-only-after-operation-stop behavior; no global KL early stop,
non-finite value, or training failure occurred. This is recorded as normal
protection behavior, not an algorithm change or a UAV collapse.

## Critic and numerical stability

```text
CRITIC_EV_MEAN = 0.12711949773420014
CRITIC_EV_FINAL = 0.06416601783730147
CRITIC_EV_MIN = -0.17564807228131296
CRITIC_EV_MAX = 0.6435781317474532
NAN_INF_DETECTED = NO
```

Critic gradient norms were finite in all epochs. EV was not required to be
positive per epoch; no NaN, Inf, un-updated critic, or exploding numerical
state was observed.

## Runtime profiling

Per-epoch runtime components were recorded in `episode_metrics.jsonl`:

```text
MEAN_EPOCH_RUNTIME_SECONDS = 656.4766615499859
MEDIAN_EPOCH_RUNTIME_SECONDS = 685.6131724000152
P95_EPOCH_RUNTIME_SECONDS = 823.497602904972
MAX_EPOCH_RUNTIME_SECONDS = 851.3250435999944
MEAN_UAV_EXACT_Q_RUNTIME_SECONDS = 529.7781791000103
UAV_EXACT_Q_RUNTIME_FRACTION_OF_TOTAL = 0.8070023050768753
ESTIMATED_300_EPOCH_HOURS_MEAN = 54.70638846249882
ESTIMATED_300_EPOCH_HOURS_P95 = 68.62480024208101
CURRENT_RUNTIME_BOTTLENECK = UAV_EXACT_Q
```

The fraction is the ratio of summed UAV Exact-Q runtime to summed total epoch
runtime; the arithmetic mean of per-epoch fractions was `0.7997307913574174`.
`UAV_EXACT_Q_RUNTIME_SECONDS` is an update-path substage and is not added a
second time to `PPO_UPDATE_RUNTIME_SECONDS`.

## Return summary

Return was reported descriptively across different realizations; no monotonic
improvement criterion was applied:

```text
RETURN_MEAN = -1050.7572813179345
RETURN_MEDIAN = -489.32542494803215
RETURN_MIN = -2920.606436592631
RETURN_MAX = 382.2867134355664
CACHE_CHANGE_RATE_MEAN = 0.5238148148148148
CACHE_CHANGE_RATE_FINAL = 0.534074074074074
UAV_MOVEMENT_RATE_MEAN = 0.5983333333333334
UAV_MOVEMENT_RATE_FINAL = 0.7
```

## Model and repository invariants

```text
MATCHING_CHANGED = NO
KKT_CHANGED = NO
REWARD_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO
DEV_SEEDS_USED = NO
FINAL_TEST_SEEDS_USED = NO
```

The run used only training-range seeds 20001–20010. No `30001..30020` or
`40001..40020` seed was used. The run output directory contains only training
artifacts (checkpoint, metrics, plots, manifest and summary); realization data
exists only in the canonical pool.

## Final fields

```text
TASK = PROPOSED_JOINT_SHORT_STABILITY_GATE
SHORT_STABILITY_GATE = PASS
EPOCHS_COMPLETED = 10/10
SEEDS_USED = 20001..20010
CACHE_POLICY_CREDIT_SOURCE = ACTUAL_ROLLOUT_GAE
CACHE_ACTION_Q_CRITIC_INITIALIZED = NO
CACHE_ACTION_Q_USED_FOR_POLICY_CREDIT = NO
CACHE_COUNTERFACTUAL_ENV_EXECUTIONS = 0
CACHE_BEHAVIORAL_COLLAPSE = NO
UAV_BEHAVIORAL_COLLAPSE = NO
CACHE_ACTOR_GRADIENT_FINITE = YES
UAV_ACTOR_GRADIENT_FINITE = YES
CRITIC_GRADIENT_FINITE = YES
CACHE_ACTOR_CUMULATIVE_PARAMETER_DELTA = 3.197631362625805
UAV_ACTOR_CUMULATIVE_PARAMETER_DELTA = 9.142512037498781
CACHE_KL_MEAN = 0.019263883698198033
CACHE_KL_MAX = 0.029438153485243746
UAV_OPERATION_KL_MEAN = 0.014285616448614747
UAV_OPERATION_KL_MAX = 0.11130460351705551
ALL_EPISODE_PROVENANCE_VERIFIED = YES
UAV_TEACHER_UNCHANGED = YES
NAN_INF_DETECTED = NO
TEMP_FILES_CLEANED = YES
GIT_DIFF_CHECK = PASS
REPORT_PATH = reports/proposed_joint_short_stability_gate.md
COMMITTED = NO
PUSHED = NO
ALGORITHM_READY_TO_FREEZE = YES
```

The source tree was not committed or pushed. Temporary interrupted pool data,
temporary runners and Python bytecode caches are cleaned after final checks.
