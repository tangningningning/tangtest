# Second-Stage Joint Repair Gate

Date: 2026-09-28

## Candidate teacher state-head normalization gate

The Candidate teacher now normalizes every TRAIN transition/state × Cache head
independently. Trigger credit and Trigger running MAD calibration are unchanged.

```text
CURRENT_HEAD = 1a7e7eda8736e4bd56a53d606f2d9d1e01134e51
ARCHITECTURE_VERSION = 6
CHECKPOINT_VERSION = 24

STATE_HEAD_NORMALIZATION_IMPLEMENTED = YES
CACHE_CANDIDATE_TEACHER_SCALE_MODE = STATE_HEAD_ROBUST_NORMALIZATION
CACHE_CANDIDATE_TEACHER_CENTER = MEDIAN
CACHE_CANDIDATE_TEACHER_SCALE = MAD_1P4826_WITH_STD_RANGE_EPS_FALLBACK
CACHE_CANDIDATE_TEACHER_TEMPERATURE = 1.0

CANDIDATE_GLOBAL_RUNNING_MAD_ACTIVE = NO
CACHE_TRIGGER_GLOBAL_RUNNING_MAD_ACTIVE = YES
CACHE_TRIGGER_TEACHER_SCALE_MODE = CLIPPED_RUNNING_MAD
CACHE_TRIGGER_TEACHER_SCALE_SOURCE = TRAIN_ROLLOUT_ONLY
```

Scale resolution is deterministic: `1.4826 × MAD`, then standard deviation,
then range, then `cache_teacher_scale_epsilon`. Equal or numerically equivalent
Q rows use zero normalized logits and therefore a uniform legal distribution.
All calculations and diagnostics use float64. No logit clipping was added.

```text
SYNTHETIC_TRANSLATION_INVARIANCE = PASS
SYNTHETIC_POSITIVE_SCALE_INVARIANCE = PASS
SYNTHETIC_EQUAL_Q_STABILITY = PASS
SYNTHETIC_NEAR_TIE_STABILITY = PASS
SYNTHETIC_OUTLIER_STABILITY = PASS

SYNTHETIC_NEAR_TIE_TEACHER = [0.1467021672, 0.2879812050, 0.5653166277]
SYNTHETIC_NEAR_TIE_ENTROPY = 0.9625084759
SYNTHETIC_NEAR_TIE_TOP1_PROBABILITY = 0.5653166277
SYNTHETIC_EQUAL_Q_TEACHER = [0.3333333333, 0.3333333333, 0.3333333333]
SYNTHETIC_OUTLIER_TOP1_PROBABILITY = 1.0
```

The TRAIN-only comparison used origin 0 and one existing group for each role:
SV head `46`, RSU head `RSU_0`, and UAV head `UAV_0`. It ran the existing
full-information evaluator with surrogate training disabled. No validation,
test, final-unseen state, optimizer update, or formal run artifact was used.

| Role/head | Candidates | Q min | Q max | Q range | Median | MAD | Scale | Fallback | Entropy | Top1 prob | Top1-top2 prob gap | Argmax |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| SV / 46 | 19 | 6.06351704 | 7.98170018 | 1.91818314 | 7.21641454 | 0.07579653 | 0.11237593 | MAD_1P4826 | 0.75719015 | 0.54853752 | 0.10660805 | 0 |
| RSU / RSU_0 | 64 | -2.09783395 | 9.22418066 | 11.32201460 | 7.46246144 | 1.62780779 | 2.41338783 | MAD_1P4826 | 3.98985245 | 0.02594094 | 0.00004093 | 0 |
| UAV / UAV_0 | 64 | 6.06563060 | 8.66660984 | 2.60097925 | 7.11833875 | 0.67763446 | 1.00466086 | MAD_1P4826 | 3.83254745 | 0.03861774 | 0.00025284 | 0 |

```text
TRAIN_CANDIDATE_GROUP_COUNT = 3
CANDIDATE_TEACHER_ARGMAX_AGREEMENT = 1.0
CANDIDATE_TEACHER_PAIRWISE_RANKING_PRESERVATION = 1.0
CANDIDATE_TEACHER_NAN_INF_COUNT = 0

OLD_GLOBAL_MAD_RESOLVED_SCALE = 1.0986075671
OLD_GLOBAL_MAD_TEACHER_ENTROPY_MEAN = 3.4777581006
NEW_STATE_HEAD_TEACHER_ENTROPY_MEAN = 2.8598633605
OLD_GLOBAL_MAD_TOP1_PROB_MEAN = 0.0609464105
NEW_STATE_HEAD_TOP1_PROB_MEAN = 0.2043653913

SV_NEW_TEACHER_ENTROPY = 0.7571901637
RSU_NEW_TEACHER_ENTROPY = 3.9898524522
UAV_NEW_TEACHER_ENTROPY = 3.8325474657
```

| Role | Old entropy | New entropy | Old top1 probability | New top1 probability |
| --- | ---: | ---: | ---: | ---: |
| SV | 2.82867982 | 0.75719016 | 0.11156164 | 0.54853749 |
| RSU | 3.72825565 | 3.98985245 | 0.03459475 | 0.02594094 |
| UAV | 3.87633883 | 3.83254747 | 0.03668284 | 0.03861774 |

Normalized top1-top2 Q-gap terciles were computed from these TRAIN groups.
Cutpoints were `0.0049055612` and `0.0764142763`.

| Gap bin | Groups | Mean top1 probability | Median top1 probability | Mean entropy |
| --- | ---: | ---: | ---: | ---: |
| small | 1 | 0.02594094 | 0.02594094 | 3.98985245 |
| medium | 1 | 0.03861774 | 0.03861774 | 3.83254745 |
| large | 1 | 0.54853752 | 0.54853752 | 0.75719015 |

The small-gap group remains diffuse instead of collapsing. Concentration rises
with the observed normalized Q gap; no arbitrary near-one-hot threshold is used
as a gate.

Checkpoint version 24 is required because Candidate actor objective semantics
changed while parameter and tensor topology did not. Architecture version
therefore remains 6. Version 23 is rejected by the existing exact checkpoint
version check before state restoration.

```text
CACHE_CANDIDATE_TEACHER_SCALE_RESOLVED = YES
CONDITIONAL_CANDIDATE_HYBRID_RESOLVED = NO
TRIGGER_TEACHER_EXPECTATION_UPDATE_IMPLEMENTED = NO
CACHE_TRIGGER_CANDIDATE_INDEPENDENT_KL_STOP = DEFERRED
ONE_EPOCH_TRAINING_EXECUTED = NO
READY_FOR_FORMAL_TRAINING_STAGE = NO
FORMAL_300_EPOCH_TRAINING_EXECUTED = NO
```

## Trigger-first architecture gate

This update changes only the Proposed Cache hierarchy.  The online policy now
decides when to update before it creates or scores any target configuration.
The full-information Cache evaluator remains TRAIN-only, so it may still build
counterfactual Candidate labels after an online KEEP transition.

```text
TRIGGER_FIRST_BASELINE_HEAD = 2e50ee2b0d07c303c3b8f4d99c1d20427420a237
ARCHITECTURE_VERSION = 6
TRIGGER_FIRST_BASELINE_CHECKPOINT_VERSION = 23

OLD_ONLINE_CACHE_FLOW = CANDIDATE_PRECOMPUTE → TRIGGER → CANDIDATE_EXECUTION
NEW_ONLINE_CACHE_FLOW = TRIGGER → CONDITIONAL_CANDIDATE_EXECUTION
TRIGGER_CANDIDATE_ONLINE_DEPENDENCY_REMOVED = YES

CACHE_TRIGGER_EXECUTION_ORDER = TRIGGER_FIRST
CACHE_TRIGGER_INPUT = STATE_PLUS_NODE_LOCAL_WORKLOAD
CACHE_TRIGGER_USES_CANDIDATE_SUMMARY = NO
CACHE_TRIGGER_USES_PRETRIGGER_CANDIDATE_LOGITS = NO
CACHE_CANDIDATE_EXECUTION_CONDITIONAL_ON_UPDATE = YES

CACHE_TRIGGER_ENCODER = cache_trigger_encoder
CACHE_CANDIDATE_ENCODER = cache_candidate_encoder
CACHE_TRIGGER_CANDIDATE_ENCODER_SHARED = NO
CACHE_TRIGGER_CANDIDATE_OPTIMIZER_SHARED = NO

CACHE_UPDATE_CHEAP_LEGALITY_EQUIVALENCE = PASS
KEEP_PATH_TRIGGER_FIRST = PASS
KEEP_PATH_ZERO_CANDIDATE_GENERATION = PASS
KEEP_PATH_ZERO_SURROGATE_CALLS = PASS
KEEP_PATH_ZERO_CANDIDATE_SCORER_CALLS = PASS
UPDATE_PATH_TRIGGER_BEFORE_CANDIDATE = PASS
UPDATE_PATH_HAS_LEGAL_CANDIDATE = PASS
UPDATE_PATH_CANDIDATE_APPLIED = PASS

CACHE_TRIGGER_CANDIDATE_PARAM_OVERLAP = 0
CACHE_TRIGGER_UAV_OPERATION_PARAM_OVERLAP = 0
CACHE_TRIGGER_UAV_CANDIDATE_PARAM_OVERLAP = 0
CACHE_CANDIDATE_UAV_OPERATION_PARAM_OVERLAP = 0
CACHE_CANDIDATE_UAV_CANDIDATE_PARAM_OVERLAP = 0
UAV_OPERATION_UAV_CANDIDATE_PARAM_OVERLAP = 0
ACTOR_PARAMETER_COVERAGE = 100%
CACHE_TRIGGER_GRADIENT_ISOLATION = PASS
CACHE_CANDIDATE_GRADIENT_ISOLATION = PASS

ROLLOUT_SNAPSHOT_SHAPE_FINITE = PASS
ONE_TRANSITION_UPDATE_SMOKE = PASS
NO_NAN_INF = YES
TOUCHED_PYTHON_COMPILE = PASS
CONFIG_SCHEMA_VALIDATION = PASS
LR_RUNNER_INITIALIZATION_GATE = PASS (4/4; initialization only)
```

The legality smoke covered SV, RSU, and UAV nodes with non-empty, all-zero,
and near-capacity rows, plus a service larger than the SV capacity.  Every
cheap result matched `len(enumerate_all_cache_target_configurations(...)) > 0`.
The KEEP call trace contained Trigger calls only.  The UPDATE trace was
`trigger → candidate_generate → surrogate → candidate_score → candidate_sample`.

```text
CACHE_TRIGGER_CREDIT_SOURCE = FULL_INFORMATION_BEST_UPDATE_VALUE_TEACHER
CACHE_CANDIDATE_CREDIT_SOURCE = CANDIDATE_SPECIFIC_VALUE_LISTWISE_TEACHER
TRIGGER_ONLINE_INPUT_DEPENDS_ON_CACHE_CREDIT = NO
CANDIDATE_EXECUTION_CONDITIONAL = YES
CANDIDATE_LEARNING_ALLOWED_ON_KEEP_STATE = YES

CACHE_TRIGGER_KL_SEPARATE = YES
CACHE_CANDIDATE_KL_SEPARATE = YES
CACHE_TRIGGER_CANDIDATE_INDEPENDENT_KL_STOP = DEFERRED
```

The trainer now records independent full-categorical Trigger KL and
conditional Candidate KL.  The existing aggregate Cache stop state remains in
place; fully independent stop states are deferred because they require a wider
training-loop rewrite.

```text
TRIGGER_FIRST_ARCHITECTURE_RESOLVED = YES
CACHE_TRIGGER_CANDIDATE_PARAMETER_DECOUPLING_RESOLVED = YES
CACHE_TRIGGER_CANDIDATE_EXECUTION_DECOUPLING_RESOLVED = YES

CANDIDATE_HYBRID_QUALITY_GATE = DEFERRED
CONDITIONAL_CANDIDATE_HYBRID_RESOLVED = NO
CACHE_CANDIDATE_TEACHER_SCALE_RESOLVED = YES
ONE_EPOCH_PIPELINE_VERIFIED = NO
READY_FOR_FORMAL_TRAINING_STAGE = NO
FORMAL_300_EPOCH_TRAINING_EXECUTED = NO
```

The historical K=16 result remains 6/9 recall with normalized regret P95
0.0611901; the failed heads are RSU_1, UAV_0, and UAV_1.  This is a
Conditional Candidate quality issue and is deferred. During the earlier
Trigger-first task, no fit groups, thresholds, K values, ranking updates, or
teacher normalization were changed.
The future Candidate teacher expectation
`Σ_c p_teacher(c|s) * Q(c)` is recorded as a possible next-stage design and was
not implemented.

## Scope and stop point

```text
HISTORICAL_GATE_HEAD = 86863d8fd7c34d35510f51c9102eb9a73e2f0f6c
SURROGATE_FIT_ORIGINS = [0, 1, 2, 3]
SURROGATE_GATE_ORIGINS = [4]
ORIGIN_4_DATA_LEAKAGE = NO
```

All labels and gates used TRAIN origins. Validation, test, final-unseen data, LR sweeps, reward changes, larger K values, and extra ranking updates were not used. The run stopped after the required origin-4 Full-Universe Hybrid Gate failed.

## Weighting diagnosis

The fit set contains every Cache head at four independent TRAIN origins: 20 SV groups, 8 RSU groups, and 8 UAV groups. Each label maps to an origin, state/head group, role, and candidate. Production `ranking_group_id` is the unique `origin_id * node_count + head_index` value.

```text
TRAIN_GROUP_COUNT_TOTAL = 36
TRAIN_GROUP_COUNT_BY_ROLE = SV 20, RSU 8, UAV 8
TRAIN_LABEL_COUNT_BY_ROLE = SV 380, RSU 2344, UAV 1088
TRAIN_LABEL_FRACTION_BY_ROLE = SV 0.099685, RSU 0.614900, UAV 0.285414

CANDIDATE_COUNT_PER_GROUP_MEAN = 105.888889
CANDIDATE_COUNT_PER_GROUP_P10 = 19
CANDIDATE_COUNT_PER_GROUP_P50 = 19
CANDIDATE_COUNT_PER_GROUP_P90 = 293
CANDIDATE_COUNT_PER_GROUP_MAX = 293

SV_CANDIDATES_PER_GROUP_MEAN = 19
RSU_CANDIDATES_PER_GROUP_MEAN = 293
UAV_CANDIDATES_PER_GROUP_MEAN = 136

CURRENT_MSE_EFFECTIVE_WEIGHT_BY_ROLE = SV 0.099685, RSU 0.614900, UAV 0.285414
CURRENT_MSE_EFFECTIVE_WEIGHT_BY_GROUP = SV 0.004984 each, RSU 0.076863 each, UAV 0.035677 each

PAIR_COUNT_TOTAL_BEFORE = 419084
PAIR_COUNT_BY_ROLE_BEFORE = SV 3420, RSU 342224, UAV 73440
PAIR_FRACTION_BY_ROLE_BEFORE = SV 0.008161, RSU 0.816600, UAV 0.175239
CURRENT_RANKING_EFFECTIVE_WEIGHT_BY_ROLE = SV 0.008161, RSU 0.816600, UAV 0.175239

SURROGATE_GROUP_WEIGHT_IMBALANCE = SUPPORTED
SURROGATE_GROUP_WEIGHT_IMBALANCE_DIAGNOSIS_COMPLETE = YES
```

The previous candidate-global MSE gave each RSU group 15.42 times the weight of each SV group. Full pair enumeration made the imbalance stronger: RSU supplied 81.66% of ranking pairs while SV supplied 0.816%.

## Production loss replacement

The previous production paths were replaced in place.

```text
L_mse_group_equal = mean_group(mean_member,candidate((prediction - target)^2))
L_rank_group_equal = mean_group(mean_member,selected_pair(pairwise_logistic))
L_total = L_mse_group_equal + ranking_weight * L_rank_group_equal

GROUP_EQUAL_MSE_IMPLEMENTED = YES
GROUP_EQUAL_RANKING_IMPLEMENTED = YES
RANKING_WEIGHT = 1.0 (unchanged)
MAX_RANK_PAIRS_PER_GROUP = 512
ACTUAL_PAIR_COUNT_PER_GROUP_MEAN = 264.472222
ACTUAL_PAIR_COUNT_PER_GROUP_P50 = 67
ACTUAL_PAIR_COUNT_PER_GROUP_P90 = 512
ACTUAL_PAIR_COUNT_PER_GROUP_MAX = 512
HARD_NEGATIVE_COUNT = 223
HARD_NEGATIVE_FRACTION = 0.058500 of fit labels
```

Pair construction prioritizes global-best comparisons, prediction-high/exact-low hard negatives, near-best comparisons, Hamming-shell representatives, top-3 comparisons, and adjacent exact ranks. Every group contributes one ranking-loss mean regardless of its retained pair count.

```text
PRE_TRIGGER_FIRST_HISTORICAL_CHECKPOINT_VERSION = 22
```

Version 22 belonged to this historical surrogate-objective stage.

Near-best is defined inside each state×head as:

```text
Q_best - Q_i <= max(1e-3, 0.01 * (Q_max - Q_min))
```

## Origin-4 Gate comparison

The MSE-only fit used 8,300 updates. Because its Gate failed, the same surrogate received exactly 500 group-equal pairwise-ranking updates. K remained restricted to 4, 6, 8, 10, 12, and 16.

| Objective | K | Best recall | Normalized regret P95 | Mean rank correlation |
| --- | ---: | ---: | ---: | ---: |
| Previous candidate/pair-weighted | 16 | 0.6667 | 0.0396025 | 0.761164 |
| Group-equal MSE only | 16 | 0.5556 | 0.0611901 | 0.624147 |
| Group-equal MSE + ranking | 16 | 0.6667 | 0.0611901 | 0.614628 |

```text
OLD_HYBRID_BEST_RECALL = 0.6667
OLD_HYBRID_NORMALIZED_REGRET_P95 = 0.0396025

GROUP_EQUAL_MSE_ONLY_BEST_RECALL = 0.5556
GROUP_EQUAL_MSE_ONLY_NORMALIZED_REGRET_P95 = 0.0611901
GROUP_EQUAL_MSE_ONLY_RANK_CORRELATION = 0.624147

RANKING_LOSS_USED = YES
GROUP_EQUAL_RANKING_BEST_RECALL = 0.6667
GROUP_EQUAL_RANKING_NORMALIZED_REGRET_P95 = 0.0611901
GROUP_EQUAL_RANKING_RANK_CORRELATION = 0.614628
```

## Per-head Gate table

This table reports the final group-equal MSE plus ranking model at K=16. `μ`, `UCB`, `σ`, and `anchor` state whether the Exact global best entered the Exact subset through that selector.

| Origin/state | Head | Role | Feasible | Exact global best | Predicted rank | Hybrid selected best | Recalled | Regret | Exact subset | μ | UCB | σ | Anchor |
| --- | --- | --- | ---: | --- | ---: | --- | --- | ---: | ---: | --- | --- | --- | --- |
| 4 / origin=4 | 23 | SV | 19 | 0100000001 | 18 | 0100000001 | YES | 0 | 17 | NO | YES | YES | NO |
| 4 / origin=4 | 87 | SV | 19 | 0000000001 | 11 | 0000000001 | YES | 0 | 18 | YES | YES | NO | NO |
| 4 / origin=4 | 43 | SV | 19 | 0000000001 | 1 | 0000000001 | YES | 0 | 18 | YES | YES | NO | NO |
| 4 / origin=4 | 75 | SV | 19 | 0000000001 | 14 | 0000000001 | YES | 0 | 17 | YES | YES | NO | NO |
| 4 / origin=4 | 58 | SV | 19 | 0000100001 | 13 | 0000100001 | YES | 0 | 17 | YES | YES | NO | NO |
| 4 / origin=4 | RSU_0 | RSU | 293 | 0010010001 | 5 | 0010010001 | YES | 0 | 22 | YES | YES | NO | NO |
| 4 / origin=4 | RSU_1 | RSU | 293 | 0000000001 | 14 | 0000101011 | NO | 0.0230876 | 24 | YES | NO | NO | NO |
| 4 / origin=4 | UAV_0 | UAV | 136 | 0000010101 | 28 | 0100010001 | NO | 0.0428199 | 21 | NO | NO | NO | NO |
| 4 / origin=4 | UAV_1 | UAV | 136 | 0000001001 | 72 | 0010000101 | NO | 0.0734369 | 21 | NO | NO | NO | NO |

```text
HELD_OUT_ORIGIN = 4
HELD_OUT_HEAD_COUNT = 9

SV_BEST_RECALL = 1.0 (5/5)
RSU_BEST_RECALL = 0.5 (1/2)
UAV_BEST_RECALL = 0.0 (0/2)

SV_NORMALIZED_REGRET_MEAN/MAX = 0 / 0
RSU_NORMALIZED_REGRET_MEAN/MAX = 0.0115438 / 0.0230876
UAV_NORMALIZED_REGRET_MEAN/MAX = 0.0581284 / 0.0734369

CACHE_HYBRID_QUALITY_GATE = FAIL
SECONDARY_TRAIN_GATE = NOT_EXECUTED
CACHE_HYBRID_EFFICIENCY_GATE = NOT_EXECUTED
```

## Historical pre-refactor downstream status

The required quality threshold is 9/9 recall and normalized regret P95 at most 0.01. The result was 6/9 and 0.0611901, so all downstream stages stopped.

```text
CACHE_CANDIDATE_SCALE_MODE = EXISTING GLOBAL RUNNING MAD
CANDIDATE_TEACHER_NEAR_ONE_HOT_FRACTION = NOT EXECUTED
CANDIDATE_TEACHER_ARGMAX_AGREEMENT = NOT EXECUTED
CACHE_CANDIDATE_TEACHER_SCALE_GATE = NOT EXECUTED

CACHE_TRIGGER_ACTION_INDEPENDENT = NOT EXECUTED
CACHE_CANDIDATE_SPECIFIC_CREDIT = NOT EXECUTED
CACHE_CANDIDATE_LEARNS_ON_KEEP = NOT EXECUTED
UAV_OPERATION_TEACHER_GATE = NOT EXECUTED
UAV_CANDIDATE_INDEPENDENT_UPDATE_GATE = NOT EXECUTED
OPTIMIZER_GROUPS_DISJOINT = NOT EXECUTED
BRANCH_KL_INDEPENDENT = NOT EXECUTED
LR_RUNNER_INITIALIZATION_GATE = NOT EXECUTED

ONE_EPOCH_TRAINING_EXECUTED = NO
ONE_EPOCH_ACTOR_LR = 5e-4
ONE_EPOCH_TARGET_EPOCHS = 1
ONE_EPOCH_COMPLETED_EPOCHS = 0
LAST_COMPLETED_EPOCH = NONE
EPOCH_2_STARTED = NO
ONE_EPOCH_TRAINING_STATUS = NOT_EXECUTED
ONE_EPOCH_RETURN = NOT RUN
ONE_EPOCH_CRITIC_EV = NOT RUN
ONE_EPOCH_CACHE_CHANGE_RATE = NOT RUN
ONE_EPOCH_UAV_MOVE_RATE = NOT RUN
ONE_EPOCH_CACHE_KL_MEAN = NOT RUN
ONE_EPOCH_UAV_OPERATION_KL_MEAN = NOT RUN
ONE_EPOCH_UAV_CANDIDATE_KL_MEAN = NOT RUN
ONE_EPOCH_NO_NAN_INF = NOT RUN
HYBRID_FALLBACK_DURING_ONE_EPOCH = NOT RUN

SECOND_STAGE_REPAIR_COMPLETE = NO
SURROGATE_GROUP_WEIGHTING_RESOLVED = YES
SURROGATE_ROLE_COVERAGE_RESOLVED = NO
FULL_UNIVERSE_HYBRID_RESOLVED = NO
CACHE_CANDIDATE_TEACHER_SCALE_RESOLVED = NO
ONE_PRODUCTION_IMPLEMENTATION_PER_FEATURE = YES
ONE_EPOCH_PIPELINE_VERIFIED = NO
READY_FOR_FORMAL_TRAINING_STAGE = NO
FORMAL_300_EPOCH_TRAINING_EXECUTED = NO
GIT_DIFF_CHECK = PASS
```

## Historical Candidate quality result (deferred)

Origin-4 Hybrid quality remains below the later Candidate-quality threshold:
recall is 6/9 and normalized regret P95 is 0.0611901. Failures are RSU_1
(0.0230876), UAV_0 (0.0428199), and UAV_1 (0.0734369). This task did not
change that evaluator, its data, or its thresholds; it is not a blocker for the
Trigger-first structure gate.

## Full TRAIN Candidate Teacher Diagnostic

Date: 2026-09-28

This is Stage 1 of the Candidate scalability architecture decision. It expands
the Candidate teacher state × head normalization from the earlier three
representative groups (SV 46, RSU_0, UAV_0) to the complete TRAIN fit coverage:
36 state/origin × Cache head groups. It reuses the existing full-information
evaluator and exact branch executor; no surrogate training, optimizer step,
validation, test, final-unseen, or origin-4 data is used. The exact branch
executor re-derived the 36 group labels in 3797 branch executions (31.4 s);
no persistent Exact-Q label source existed, so the existing evaluator was the
single source of truth.

```text
CURRENT_HEAD = 8b3234245d6d3d0b94c0428e5cb13e184f1456f3
ARCHITECTURE_VERSION = 6
CHECKPOINT_VERSION = 24

TRAIN_CANDIDATE_GROUP_COUNT = 36
TRAIN_GROUP_COUNT_BY_ROLE = SV 20, RSU 8, UAV 8

CANDIDATE_TEACHER_ARGMAX_AGREEMENT = 1.0
CANDIDATE_TEACHER_PAIRWISE_RANKING_PRESERVATION = 1.0
CANDIDATE_TEACHER_NAN_INF_COUNT = 0

MAD_SCALE_COUNT = 36
STD_FALLBACK_COUNT = 0
RANGE_FALLBACK_COUNT = 0
EPSILON_FALLBACK_COUNT = 0

SV_TEACHER_ENTROPY_MEAN = 2.0534
RSU_TEACHER_ENTROPY_MEAN = 5.1333
UAV_TEACHER_ENTROPY_MEAN = 4.2088

SV_TOP1_PROB_MEAN = 0.3404
RSU_TOP1_PROB_MEAN = 0.0084
UAV_TOP1_PROB_MEAN = 0.0344
```

Every legal teacher distribution is finite, non-negative, and sums to 1. The
resolved scale is MAD×1.4826 in all 36 groups, so the STD/RANGE/EPSILON
fallbacks were never triggered. Pairwise ranking preservation is exact over all
419080 non-tied Q pairs; 4 near-tie pairs whose float64 softmax probabilities
are numerically indistinguishable (exp underflow) were treated as ties, and
there are 0 true ranking reversals.

| Role | Groups | Candidates mean/p50/p90 | Entropy mean/p50/p90 | Top1 prob mean/p50/p90 | Norm top1-top2 Q-gap mean/p50/p90 | Resolved scale mean/p50/p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SV | 20 | 19 / 19 / 19 | 2.0534 / 2.5822 / 2.7149 | 0.3404 / 0.1603 / 1.0000 | 11.3228 / 0.1284 / 53.4700 | 0.0803 / 0.0734 / 0.1140 |
| RSU | 8 | 293 / 293 / 293 | 5.1333 / 5.1508 / 5.2391 | 0.0084 / 0.0082 / 0.0100 | 0.0028 / 0.0013 / 0.0066 | 3.2984 / 1.6418 / 7.8326 |
| UAV | 8 | 136 / 136 / 136 | 4.2088 / 4.4962 / 4.6417 | 0.0344 / 0.0266 / 0.0587 | 0.0077 / 0.0075 / 0.0133 | 1.4102 / 0.4413 / 3.3444 |

The full 36-group spread confirms the earlier 3-group observation: SV heads
have small candidate universes and a concentrated teacher (top1 probability up
to 1.0), while RSU and UAV heads stay diffuse (top1 probability around 0.008
and 0.034 respectively) because their feasible universes are much larger.

```text
CANDIDATE_TEACHER_36_GROUP_GATE = PASS
```

## Candidate Scalability Architecture Gate

Date: 2026-09-28

This is Stage 2 of the Candidate scalability architecture decision: a minimal,
production-independent service-level autoregressive set decoder. It tests
whether "UPDATE → per-service selection → capacity mask → STOP → final target"
can replace the current configuration-level enumeration/Hybrid, without
touching the production path. Stage 2 runs only because the Stage 1 gate
passed.

```text
SERVICE_LEVEL_PROTOTYPE_IMPLEMENTED = YES
PROTOTYPE_PRODUCTION_PATH_USED = NO
PROTOTYPE_FIT_ORIGINS = [0, 1, 2]
PROTOTYPE_INTERNAL_EVAL_ORIGINS = [3]
ORIGIN4_USED = NO

SERVICE_LEVEL_FEATURE_SCHEMA = [
  service_size_over_node_capacity,
  predicted_service_popularity,
  node_local_predicted_request_mass,
  node_local_predicted_input_bits,
  node_local_predicted_cpu_cycles,
  current_node_cached_bit,
  current_replica_ratio,
  node_role_is_sv,
  node_role_is_rsu,
  node_role_is_uav,
]

SERVICE_LEVEL_PARAMETER_COUNT = 168450
SERVICE_ENCODER_PARAMETER_COUNT = 17920
MODEL_PARAMETER_COUNT_INDEPENDENT_OF_F = YES
SERVICE_ENCODER_PARAMETER_COUNT_INDEPENDENT_OF_F = YES

FULL_CONFIGURATION_ENUMERATION_USED = NO
FULL_CONFIGURATION_MATERIALIZED_COUNT = 0
GENERATED_CONFIGURATION_CAPACITY_VIOLATION_COUNT = 0
UPDATE_NOOP_CONFIGURATION_COUNT = 0
EXACT_ORACLE_TIE_COUNT = 0
```

The decoder uses a shared service MLP encoder, a shared context MLP encoder,
selected-set mean pooling, a remaining-capacity ratio scalar, a single shared
service scorer, and a single shared STOP scorer. Hidden dimension 128 and Adam
learning rate 5e-4 are reused from the existing Candidate configuration. All
shapes derive from `service_catalog["num_services"]`; F is never hard-coded.
The canonical tie-break for the unique training target is the lexicographically
smallest binary target vector among exact-tied best candidates.

Fit used 92 permutation-invariant set-imitation examples over the 27 fit
groups (origins 0/1/2); the loss is `-log Σ_{f∈T*\S_k} π(f)` with STOP positive
only at `S_k == T*`. Deterministic greedy decoding then ran on the 9 origin-3
groups.

### Greedy origin-3 results

| Head | Role | Current | Oracle | Generated | Oracle Q | Generated Q | Norm regret | Recovered | #svc | Steps | Cap used MB | Runtime s |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 46 | SV | 0100000100 | 0100000000 | 0000001000 | 11.12275 | 10.94718 | 0.013709 | NO | 1 | 2 | 16216.88 | 0.00081 |
| 10 | SV | 0000001000 | 0000000000 | 0000000000 | 11.12915 | 11.12915 | 0.000000 | YES | 0 | 1 | 0.00 | 0.00033 |
| 0 | SV | 0100110000 | 0000110000 | 0100010000 | 11.23016 | 11.22116 | 0.000519 | NO | 2 | 3 | 11710.26 | 0.00090 |
| 23 | SV | 0100110000 | 0000110000 | 0100010000 | 11.22192 | 11.21847 | 0.000138 | NO | 2 | 3 | 11710.26 | 0.00090 |
| 59 | SV | 0000010100 | 0000010000 | 0000001000 | 11.22351 | 11.05733 | 0.005821 | NO | 1 | 2 | 16216.88 | 0.00058 |
| RSU_0 | RSU | 0101110100 | 0000010000 | 0000000000 | 11.28326 | 11.27445 | 0.000107 | NO | 0 | 1 | 0.00 | 0.00031 |
| RSU_1 | RSU | 0010110101 | 0000000101 | 0000000001 | 11.15220 | 11.05753 | 0.000741 | NO | 1 | 2 | 9998.63 | 0.00061 |
| UAV_0 | UAV | 1100010000 | 0000010000 | 1000010000 | 11.57306 | 10.08331 | 0.232687 | NO | 2 | 3 | 25867.99 | 0.00088 |
| UAV_1 | UAV | 0010110100 | 0010000000 | 0010001000 | 11.13826 | 10.94397 | 0.021144 | NO | 2 | 3 | 30073.74 | 0.00103 |

```text
GREEDY_EXACT_BEST_RECOVERY = 1/9
GREEDY_NORMALIZED_REGRET_MEAN = 0.0305
GREEDY_NORMALIZED_REGRET_P50 = 0.0007
GREEDY_NORMALIZED_REGRET_P95 = 0.1481
GREEDY_NORMALIZED_REGRET_MAX = 0.2327
```

Per-role greedy: SV 1/5 (regret mean 0.0040, max 0.0137), RSU 0/2 (mean
0.0004, max 0.0007), UAV 0/2 (mean 0.1269, max 0.2327). The dominant failure
is UAV_0 with normalized regret 0.2327.

### Beam-4 diagnostic

Because greedy did not reach the existing Candidate quality gate, a single
fixed beam width 4 run was performed to separate decoder scoring error from
greedy search error. Beam-4 recovers the same 1/9 exact best and only reduces
UAV_0's regret (to 0.0772), so the failure is primarily in the decoder's
scoring/feature representation rather than the greedy search.

```text
BEAM4_EXACT_BEST_RECOVERY = 1/9
BEAM4_NORMALIZED_REGRET_MEAN = 0.0133
BEAM4_NORMALIZED_REGRET_P95 = 0.0547
BEAM4_NORMALIZED_REGRET_MAX = 0.0772

SERVICE_LEVEL_CANDIDATE_QUALITY_GATE = FAIL
```

The existing Candidate quality reference is Exact-best recall 9/9 with
normalized regret P95 ≤ 0.01. Both greedy (1/9, P95 0.1481) and Beam-4 (1/9,
P95 0.0547) fall short, so the quality gate fails for this minimal prototype.

### Scalability smoke

Synthetic service tensors at F = 10/20/30/50 test only forward, greedy decode,
capacity mask, STOP, and memory/runtime. No Matching/KKT, no Exact-Q, and no
real enlarged scenarios are used. The decoder never materializes full
configurations; it scores the unselected legal services plus STOP at each step.

| F | Pass | Decoder steps | Service scorer evaluations | Full configs materialized | Runtime s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10 | PASS | 2 | 20 | 0 | 0.00104 |
| 20 | PASS | 3 | 60 | 0 | 0.00104 |
| 30 | PASS | 3 | 90 | 0 | 0.00115 |
| 50 | PASS | 9 | 450 | 0 | 0.00354 |

```text
F10_SCALABILITY = PASS
F20_SCALABILITY = PASS
F30_SCALABILITY = PASS
F50_SCALABILITY = PASS
FULL_CONFIGURATION_MATERIALIZED_COUNT = 0
SERVICE_LEVEL_CANDIDATE_SCALABILITY_GATE = PASS
```

### Baseline complexity reference

The current configuration-level path materializes the whole feasible universe:
293 / 105.9 / 19 candidates for RSU / mean / SV, versus the theoretical 2^F
binary space. The service-level decoder avoids that enumeration: parameter
count and per-step scoring cost are independent of F.

```text
CURRENT_CONFIGURATION_LEVEL_MAX_CANDIDATES = 293
CURRENT_CONFIGURATION_LEVEL_MEAN_CANDIDATES = 105.888889
SV_MEAN = 19
RSU_MEAN = 293
UAV_MEAN = 136
```

### Decision

The service-level prototype satisfies the scalability contract (no exponential
enumeration, no shape failure, no NaN/Inf, parameter count independent of F),
but its minimal shared-encoder set-imitation quality (greedy 1/9, Beam-4 1/9,
regret P95 0.1481 / 0.0547) does not meet the existing 9/9 and P95 ≤ 0.01
Candidate quality reference. On this evidence the architecture is not migrated.

```text
SERVICE_LEVEL_PROTOTYPE_QUALITY_GATE = FAIL
SERVICE_LEVEL_PROTOTYPE_SCALABILITY_GATE = PASS
CANDIDATE_ARCHITECTURE_RECOMMENDATION = KEEP_CURRENT_PRODUCTION_PENDING_ANALYSIS

PRODUCTION_CANDIDATE_ARCHITECTURE_CHANGED = NO
CONDITIONAL_CANDIDATE_HYBRID_RESOLVED = NO
TRIGGER_TEACHER_EXPECTATION_UPDATE_IMPLEMENTED = NO
CACHE_TRIGGER_CANDIDATE_INDEPENDENT_KL_STOP = DEFERRED
VEHICLE_COUNT_SCALABILITY = DEFERRED
CACHE_NODE_COUNT_SCALABILITY = DEFERRED
ONE_EPOCH_TRAINING_EXECUTED = NO
READY_FOR_FORMAL_TRAINING_STAGE = NO
FORMAL_300_EPOCH_TRAINING_EXECUTED = NO
```

The answer to this task's question — in a Trigger-first setting, should the
post-UPDATE Candidate move from configuration-level enumeration/Hybrid to a
service-level autoregressive constrained decoder — is: the scalability
contract is met, but the minimal prototype's exact-best recovery is far below
the frozen Hybrid baseline (6/9) and below the 9/9 quality reference, so the
evidence does not support migration. The Hybrid result is unchanged and remains
the frozen 6/9 / 0.0611901.

## Origin-4 terminology update

From this task on, origin 4 is a development gate, not an untouched held-out.
Historical paragraphs below that call it "held-out" or "unseen" retain the old
wording only as historical terminology.

```text
DEVELOPMENT_GATE_ORIGIN = 4
```

## Hybrid Exact-Confirmed Winner Repair

Date: 2026-09-28

This is the generic semantic repair: the final Hybrid winner,
`best_candidate_index` / `q_best_update`, must itself be Exact-Q evaluated.
Surrogate values may still drive screening, ranking, and the choice of which
candidates are exact-evaluated first, but a surrogate-only candidate may no
longer become the final winner.

### Code-path confirmation

Inspection of `evaluate_cache_full_information_values` confirms the production
TRAIN credit path builds a mixed value vector: `hybrid_q = keep_value +
predicted_delta` with exact override only at the selected `exact_indices`, then
takes `argmax(hybrid_q)` over the whole universe and over the policy subset.

```text
HYBRID_MIXED_VALUE_VECTOR_EXISTS = YES
HYBRID_FINAL_ARGMAX_CAN_BE_SURROGATE_ONLY_BEFORE = YES
```

The historical origin-4 record proves the bug manifested. For RSU_1 the exact
global best `0000000001` was exact-evaluated (surrogate rank 14, selected via
μ), but the final winner was `0000101011`, a surrogate-only candidate whose
inflated surrogate value beat the exact true best. That is exactly this
semantic hole, not a screening miss.

```text
RSU1_HISTORICAL_TRUE_BEST = 0000000001 (predicted rank 14, in initial exact set)
RSU1_HISTORICAL_WINNER = 0000101011 (surrogate-only)
RSU1_HISTORICAL_REGRET = 0.0230876
```

### Repair

A minimal iterative exact-confirmation was added in
`evaluate_cache_full_information_values`. After `hybrid_q` is built:

1. `winner = argmax(hybrid_q)`.
2. If `winner` is already exact, stop.
3. Otherwise exact-evaluate only that winner, override its `hybrid_q` and
   `exact_by_index` entries, and re-argmax.
4. Repeat until the winner is exact-evaluated.

This guarantees a semantically valid winner without enumerating the whole
universe. It does not promise the global exact optimum: a candidate that is
never exact-evaluated may still hide a higher true value.

```text
GLOBAL_EXACT_OPTIMUM_GUARANTEED = NO
FINAL_HYBRID_WINNER_EXACT_EVALUATED = YES
FINAL_HYBRID_WINNER_Q_SOURCE = EXACT_Q
Q_BEST_UPDATE_SOURCE = EXACT_Q
BEST_CANDIDATE_INDEX_SOURCE = EXACT_CONFIRMED_HYBRID
```

### Verification

The repaired production evaluator was run on the origin-4 development gate
(9 heads) with a TRAIN-only fit surrogate (2000-step group-equal MSE, 3812
labels; the current production config defaults to MSE, not the historical
MSE+ranking objective). All 9 final winners were `EXACT_Q` and
exact-evaluated; `confirmation_added_exact_count = 0` because this re-fit
surrogate's argmax already landed on exact candidates in every head.

```text
CHECKPOINT_VERSION = 25
```

Because the repair changes the production TRAIN cache-credit winner objective,
`JOINT_PROPOSED_CHECKPOINT_VERSION` was bumped 24 → 25 and version-24
proposed-joint checkpoints now fail fast. `ARCHITECTURE_VERSION` stays 6 (no
tensor topology change).

### Development regression (re-fit MSE surrogate)

The re-fit MSE surrogate ranks worse than the historical MSE+ranking surrogate
(e.g. RSU_1 true-best predicted rank 223 here versus 14 historically), so this
table is a development diagnostic, not a reproduction of the frozen 6/9.

| Head | Role | Feasible | True best rank | Before winner source | Before regret | After regret | Recovered | Confirm added |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | ---: |
| 46 | SV | 19 | 18 | EXACT_Q | 0.000247 | 0.000247 | NO | 0 |
| 10 | SV | 19 | 19 | EXACT_Q | 0.000297 | 0.000297 | NO | 0 |
| 0 | SV | 19 | 18 | EXACT_Q | 0.000252 | 0.000252 | NO | 0 |
| 23 | SV | 19 | 18 | EXACT_Q | 0.000569 | 0.000569 | NO | 0 |
| 59 | SV | 19 | 18 | EXACT_Q | 0.0 | 0.0 | YES | 0 |
| RSU_0 | RSU | 293 | 117 | EXACT_Q | 0.000087 | 0.000087 | NO | 0 |
| RSU_1 | RSU | 293 | 223 | EXACT_Q | 0.000508 | 0.000508 | NO | 0 |
| UAV_0 | UAV | 136 | 102 | EXACT_Q | 0.000608 | 0.000608 | NO | 0 |
| UAV_1 | UAV | 136 | 108 | EXACT_Q | 0.044895 | 0.044895 | NO | 0 |

```text
ORIGIN4_RECALL_BEFORE = 6/9  (historical frozen; MSE+ranking surrogate)
ORIGIN4_RECALL_AFTER = 1/9   (development re-fit MSE surrogate; not comparable)
ORIGIN4_NORMALIZED_REGRET_P95_BEFORE = 0.0611901
ORIGIN4_NORMALIZED_REGRET_P95_AFTER = 0.0271800  (development re-fit)

RSU1_TRUE_BEST_IN_INITIAL_EXACT_SET = NO   (re-fit rank 223; historical rank 14 = YES)
RSU1_BEFORE_WINNER_SOURCE = EXACT_Q        (re-fit; historical = SURROGATE_Q)
RSU1_BEFORE_REGRET = 0.000508
RSU1_EXACT_BEST_RECOVERED = NO
RSU1_AFTER_REGRET = 0.000508

PREVIOUSLY_PASSING_HEAD_REGRESSION = PASS
NO_NAN_INF = YES
```

The regression guarantee is structural, not just empirical: the confirmation
loop exits immediately when the winner is already exact, and it only replaces
surrogate values with exact values, which are never above the exact global
best. So an already-correct head cannot become worse.

### Exact cost

```text
INITIAL_EXACT_EVAL_COUNT_MEAN = 20.67
FINAL_EXACT_EVAL_COUNT_MEAN = 20.67
CONFIRMATION_ADDED_EXACT_EVAL_COUNT_MEAN = 0.0
CONFIRMATION_ADDED_EXACT_EVAL_COUNT_MAX = 0
RSU1_CONFIRMATION_ADDED_EXACT_EVALS = 0
EFFICIENCY_CONCERN = NO
```

In this development reproduction the confirmation loop added zero exact
evaluations (the argmax was already exact in all heads). SV heads are
near-full-exact only because their feasible universe is 19 candidates and K=16,
a pre-existing property unrelated to the repair. The repair adds at most a few
extra exact evaluations and only when a surrogate-only winner would otherwise
have been selected.

## UAV Screening Root-Cause Diagnosis

Date: 2026-09-28

This stage is diagnosis only. No surrogate, feature, selector, K, training
data, network, or uncertainty logic was changed. The surrogate used is the
current production default (group-equal MSE), fit on TRAIN origins 0-3.

```text
TRAIN_UAV_INDEPENDENT_STATE_HEAD_GROUPS = 8
```

### True-best diagnostics

| Field | UAV_0 | UAV_1 |
| --- | ---: | ---: |
| True best config | 0000101010 | 0010000010 |
| True best exact delta Q | +118.42 | +116.88 |
| Surrogate μ | -5.18 | -3.28 |
| Surrogate σ | 0.271 | 0.275 |
| Abs prediction error | 123.60 | 120.16 |
| Signed prediction error | -123.60 | -120.16 |
| Standardized error | 455.4 | 436.8 |
| μ rank | 102/136 | 108/136 |
| σ rank | 92 | 29 |
| UCB rank | 102 | 109 |
| Selected by μ / σ / UCB / anchor | NO/NO/NO/NO | NO/NO/NO/NO |
| Selected for initial exact | NO | NO |

The historical predicted ranks were 28 (UAV_0) and 72 (UAV_1); both are above
K=16, so the true best was never selected for exact evaluation in either run.
The true-best candidates are extreme positive delta-Q outliers (≈ +118) that
the smooth surrogate confidently under-predicts (μ ≈ -5) with a large signed
negative error.

### Near-best and false-positive evidence

```text
UAV0_NEAR_BEST_COUNT = 35
UAV1_NEAR_BEST_COUNT = 5
```

For both UAV heads the surrogate μ rank of the true best is near the bottom of
the universe (102-108 of 136), and its σ rank is not the maximum. The surrogate
μ top-16 are candidates whose true delta Q is near zero or negative (exact rank
~46-122), i.e. the surrogate's top-ranked candidates are not the exact
near-best candidates. The screening misses are a ranking problem, not a
capacity-mask problem.

### Error vs uncertainty

```text
ABS_ERROR_SIGMA_PEARSON = 0.409
ABS_ERROR_SIGMA_SPEARMAN = 0.519

SIGMA_Q1_MEAN_ABS_ERROR = 1.59
SIGMA_Q2_MEAN_ABS_ERROR = 1.74
SIGMA_Q3_MEAN_ABS_ERROR = 8.24
SIGMA_Q4_MEAN_ABS_ERROR = 24.27

TOP_ERROR_DECILE_MEAN_SIGMA = 1.75
```

Ensemble σ is directionally informative (higher σ → higher error, Q4/Q1 ≈ 15×)
but under-scaled in absolute terms. Per-role:

| Role | Surrogate MAE | RMSE | Mean σ | Median σ | Mean standardized error |
| --- | ---: | ---: | ---: | ---: | ---: |
| SV | 2.92 | 17.65 | 0.262 | 0.122 | 3.06 |
| RSU | 10.68 | 37.74 | 0.647 | 0.430 | 11.72 |
| UAV | 7.36 | 27.66 | 0.187 | 0.153 | 63.18 |

UAV has the lowest mean σ (0.187) but the highest mean standardized error
(63.2), i.e. the ensemble is most confidently wrong on UAV. This is the
signature of under-scaled uncertainty, but the overall σ-error correlation is
moderate rather than weak, so it is not a clean under-calibration.

### Coverage

```text
CONTEXT_DIM = 400, CANDIDATE_FEATURE_DIM = 81
TRAIN_UAV_CONTEXT_DISTANCE mean/p50/p90/max = 24.76 / 23.80 / 41.91 / 42.11

UAV0_CONTEXT_DISTANCE_PERCENTILE = 0.857
UAV0_CANDIDATE_FEATURE_DISTANCE_PERCENTILE = 0.029
UAV0_JOINT_DISTANCE_PERCENTILE = 0.878
UAV1_CONTEXT_DISTANCE_PERCENTILE = 0.857
UAV1_CANDIDATE_FEATURE_DISTANCE_PERCENTILE = 0.001
UAV1_JOINT_DISTANCE_PERCENTILE = 0.878
```

The origin-4 UAV context sits at the 85.7th percentile of the TRAIN UAV
context distribution (elevated but below the p90 of 41.9), while the true-best
candidate feature is well covered (2.9th percentile). The joint representation
is at the 87.8th percentile. The top-5 nearest TRAIN joint samples for each UAV
true-best are origin-3 UAV_1 candidates with delta ≈ -3.9, whereas the true-best
delta is +118: the nearest TRAIN points in joint space have a drastically
different Exact-Q, but the joint distance is large (35-36), so the true-best is
an outlier rather than a near-duplicate.

### Root-cause classification

```text
UAV_STATE_COVERAGE_LIMITED = INCONCLUSIVE
UAV_UNCERTAINTY_UNDERCALIBRATED = INCONCLUSIVE
UAV_FEATURE_REPRESENTATION_INSUFFICIENT = INCONCLUSIVE
UAV_MODEL_CAPACITY_LIMITED = INCONCLUSIVE
```

- Coverage is elevated (85.7th percentile) but below the p90 and the candidate
  feature is well covered, so "clearly in the tail" is not met.
- Uncertainty is under-scaled for UAV (highest standardized error, lowest σ),
  but σ is directionally correlated with error and UAV_1's true-best σ
  percentile is not low, so "weak overall σ-error relationship" is not met.
- The true-best is far (87.8th percentile), not "very close", in joint space, so
  the feature-representation condition is not met.
- Model capacity is INCONCLUSIVE by construction.

```text
PRIMARY_UAV_SCREENING_ROOT_CAUSE = INCONCLUSIVE
NEXT_MINIMAL_REPAIR_DIRECTION = COLLECT_MORE_DIAGNOSTIC_EVIDENCE
```

The single most consistent observation is that the UAV true-best candidates are
extreme critic-dominated delta-Q outliers (+118) that the smooth surrogate
confidently under-predicts (μ ≈ -5), so they are ranked outside the top-16 and
never exact-evaluated. No single hypothesis meets its strict support condition
on the current evidence, so no UAV repair is triggered.

### Unchanged state

```text
CACHE_CANDIDATE_TEACHER_SCALE_RESOLVED = YES
CANDIDATE_TEACHER_36_GROUP_GATE = PASS
CONDITIONAL_CANDIDATE_HYBRID_RESOLVED = NO
SERVICE_LEVEL_PROTOTYPE_QUALITY_GATE = FAIL
SERVICE_LEVEL_PROTOTYPE_SCALABILITY_GATE = PASS
PRODUCTION_CANDIDATE_ARCHITECTURE_CHANGED = NO
TRIGGER_TEACHER_EXPECTATION_UPDATE_IMPLEMENTED = NO
CACHE_TRIGGER_CANDIDATE_INDEPENDENT_KL_STOP = DEFERRED
ONE_EPOCH_TRAINING_EXECUTED = NO
READY_FOR_FORMAL_TRAINING_STAGE = NO
FORMAL_300_EPOCH_TRAINING_EXECUTED = NO
```

## Candidate Exact-Q Label Integrity Gate

Date: 2026-09-28

This gate checks whether the origin-4 UAV Exact-Q outliers can be used as
Candidate-teacher evidence. Scope is TRAIN origins 0-3 plus development-gate
origin 4 only. No training, checkpoint write, validation, test, or new
Candidate-ranking analysis was run.

```text
CURRENT_HEAD = 789726e254023e36a68bd8d1ea04b807e618fa37
ARCHITECTURE_VERSION = 6
CHECKPOINT_VERSION = 25
DEVELOPMENT_GATE_ORIGIN = 4
```

### Exact-Q code semantics

`src/ppo_simple/cache_credit.py::_execute_cache_branch` executes the branch
through `slot_environment.execute_slot`, checks the request contract, and sets
the branch reward from `_check_reward(result)`. `_check_reward` verifies
`result.system_utility == actual_task_utility - cache_cost - movement_cost`
and returns `system_utility`. For a nonterminal branch, the next observation is
rebuilt from `result.next_cache_state`, `result.next_uav_positions`, and the
next aligned step. `_branch_q_values` then computes:

```text
EXACT_Q_FORMULA = reward + gamma * critic_value(counterfactual_next_observation)
```

This is the current source formula. It combines the real immediate branch
reward with a learned critic bootstrap. The code semantics alone do not identify
which term produced the historical +118 deltas.

### Critic provenance

The prior origin-4 report entry records the two aggregate Exact-Q deltas and
surrogate means, but it does not preserve the evaluator invocation, critic
parameter hash, checkpoint path, checkpoint metadata, optimizer-step count, or
PopArt statistics. The evaluator's `branch_cache` is local to one call; the
persisted gate report contains no raw KEEP/true-best branch rewards, next-state
values, or observations from which to reconstruct those fields.

The current lifecycle code creates the policy in `PPOV2Trainer.__init__` with
critic optimizer steps initialized to zero. A successful `load_checkpoint`
restores policy parameters, critic optimizer state/steps, and (for V6) checks
the restored PopArt buffers against checkpoint return statistics. V6's
unrestored constructor defaults are PopArt mean 0 and standard deviation 1.
Those defaults cannot be attributed to the historical +118 diagnostic without
its invocation record. In the current workspace, the 56
`runs/*/latest_checkpoint.pt` files inspected have checkpoint versions no
higher than 17; none is compatible with current version 25. This does not
prove which older or in-memory parameter state the historical invocation used.

```text
CRITIC_PARAMETER_SOURCE = OTHER (historical invocation is untraceable)
CRITIC_CHECKPOINT_PATH = NONE RECORDED
CRITIC_CHECKPOINT_VERSION = UNKNOWN
CRITIC_OPTIMIZER_STEPS = UNKNOWN
CRITIC_POPART_STATE_SOURCE = UNKNOWN
CRITIC_POPART_MEAN = NOT AVAILABLE
CRITIC_POPART_STD = NOT AVAILABLE
CRITIC_PROVENANCE_GATE = FAIL (no evidence sufficient to pass)
CURRENT_UAV_PLUS118_DIAGNOSTIC_VALID_FOR_PRODUCTION_CREDIT = NO
```

This is a fail-closed provenance decision. It does not assert that the
historical critic was random or untrained; the saved evidence cannot establish
that it was trained and restored either.

### KEEP / true-best decomposition

The existing origin-4 entry supplies the best configs and delta-Q values, but
not its branch records. The requested additive decomposition therefore cannot
be calculated or numerically checked without mixing in a different rollout or
critic. No replacement branch was run under an arbitrary critic.

| Field | UAV_0 | UAV_1 |
| --- | --- | --- |
| TRUE_BEST_CONFIG | `0000101010` | `0010000010` |
| KEEP_REWARD / KEEP_NEXT_CRITIC_VALUE / KEEP_Q | NOT AVAILABLE | NOT AVAILABLE |
| TRUE_BEST_REWARD / TRUE_BEST_NEXT_CRITIC_VALUE / TRUE_BEST_Q | NOT AVAILABLE | NOT AVAILABLE |
| Recorded `DELTA_Q` | +118.42 | +116.88 |
| `DELTA_REWARD` | NOT AVAILABLE | NOT AVAILABLE |
| `GAMMA_DELTA_CRITIC` | NOT AVAILABLE | NOT AVAILABLE |
| `Q_DECOMPOSITION_NUMERICAL_ERROR` | NOT COMPUTABLE | NOT COMPUTABLE |

```text
UAV0_KEEP_REWARD = NOT AVAILABLE
UAV0_KEEP_CRITIC_V = NOT AVAILABLE
UAV0_KEEP_Q = NOT AVAILABLE
UAV0_TRUE_BEST_REWARD = NOT AVAILABLE
UAV0_TRUE_BEST_CRITIC_V = NOT AVAILABLE
UAV0_TRUE_BEST_Q = NOT AVAILABLE
UAV0_DELTA_REWARD = NOT AVAILABLE
UAV0_GAMMA_DELTA_CRITIC = NOT AVAILABLE
UAV0_DELTA_Q = +118.42 (recorded aggregate only)
UAV0_REWARD_CONTRIBUTION_RATIO = NOT COMPUTABLE
UAV0_CRITIC_CONTRIBUTION_RATIO = NOT COMPUTABLE
UAV0_TRUE_BEST_DELTA_REWARD_RANK = NOT RUN
UAV0_TRUE_BEST_DELTA_CRITIC_RANK = NOT RUN
UAV0_Q_REWARD_PEARSON = NOT RUN
UAV0_Q_CRITIC_PEARSON = NOT RUN
UAV0_Q_REWARD_SPEARMAN = NOT RUN
UAV0_Q_CRITIC_SPEARMAN = NOT RUN

UAV1_KEEP_REWARD = NOT AVAILABLE
UAV1_KEEP_CRITIC_V = NOT AVAILABLE
UAV1_KEEP_Q = NOT AVAILABLE
UAV1_TRUE_BEST_REWARD = NOT AVAILABLE
UAV1_TRUE_BEST_CRITIC_V = NOT AVAILABLE
UAV1_TRUE_BEST_Q = NOT AVAILABLE
UAV1_DELTA_REWARD = NOT AVAILABLE
UAV1_GAMMA_DELTA_CRITIC = NOT AVAILABLE
UAV1_DELTA_Q = +116.88 (recorded aggregate only)
UAV1_REWARD_CONTRIBUTION_RATIO = NOT COMPUTABLE
UAV1_CRITIC_CONTRIBUTION_RATIO = NOT COMPUTABLE
UAV1_TRUE_BEST_DELTA_REWARD_RANK = NOT RUN
UAV1_TRUE_BEST_DELTA_CRITIC_RANK = NOT RUN
UAV1_Q_REWARD_PEARSON = NOT RUN
UAV1_Q_CRITIC_PEARSON = NOT RUN
UAV1_Q_REWARD_SPEARMAN = NOT RUN
UAV1_Q_CRITIC_SPEARMAN = NOT RUN
ALL_CANDIDATE_TOP10_RANKINGS = NOT RUN
ALL_CANDIDATE_Q_DECOMPOSITION = NOT AVAILABLE
DATA_NAN_INF_CHECK = NOT AVAILABLE (raw diagnostic arrays absent)
```

The immediate reward formula is verified statically, but the values needed to
answer whether +118 came from reward or bootstrap were not retained. The
previous narrative calling the outliers “critic-dominated” is unverified and is
superseded by this integrity result.

### All-candidate ranking and critic support

The provenance gate failed before surrogate, feature, or uncertainty
root-cause analysis. The former report retains aggregate surrogate means
(`UAV_0 μ ≈ -5.18`, `UAV_1 μ ≈ -3.28`), but the exact per-candidate rewards,
critic values, Q values, and surrogate σ values are not persisted together.
Accordingly, the required top-10 lists, correlations, contribution ratios,
true-best component ranks, counterfactual critic distribution, and cache-bit
sensitivity are `NOT RUN` / `NOT AVAILABLE` for this gate.

PopArt raw/normalized values and reconstruction errors are unavailable for the
same reason: there is no matching critic state or saved counterfactual
observation. Actual TRAIN critic support is also `NOT_AVAILABLE`; comparing
against a historical checkpoint would violate the same-critic requirement.

```text
SAME_CRITIC_PARAMETERS_FOR_SUPPORT_COMPARISON = NO (not established)
ACTUAL_TRAIN_CRITIC_SUPPORT_COMPARISON = NOT_AVAILABLE
ACTUAL_STATE_CRITIC_MIN = NOT AVAILABLE
ACTUAL_STATE_CRITIC_P01 = NOT AVAILABLE
ACTUAL_STATE_CRITIC_P05 = NOT AVAILABLE
ACTUAL_STATE_CRITIC_P50 = NOT AVAILABLE
ACTUAL_STATE_CRITIC_P95 = NOT AVAILABLE
ACTUAL_STATE_CRITIC_P99 = NOT AVAILABLE
ACTUAL_STATE_CRITIC_MAX = NOT AVAILABLE
UAV0_TRUE_BEST_CRITIC_ACTUAL_STATE_PERCENTILE = NOT AVAILABLE
UAV1_TRUE_BEST_CRITIC_ACTUAL_STATE_PERCENTILE = NOT AVAILABLE
POPART_DIAGNOSTICS = NOT AVAILABLE
POPART_MEAN = NOT AVAILABLE
POPART_STD = NOT AVAILABLE
UAV0_KEEP_NORMALIZED_Z = NOT AVAILABLE
UAV0_TRUE_BEST_NORMALIZED_Z = NOT AVAILABLE
UAV0_TRUE_BEST_POPART_RECONSTRUCTION_ERROR = NOT AVAILABLE
UAV1_KEEP_NORMALIZED_Z = NOT AVAILABLE
UAV1_TRUE_BEST_NORMALIZED_Z = NOT AVAILABLE
UAV1_TRUE_BEST_POPART_RECONSTRUCTION_ERROR = NOT AVAILABLE
```

### Classification and next stage

```text
CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION = CANDIDATE_Q_INTEGRITY_INCONCLUSIVE
COUNTERFACTUAL_CRITIC_RELIABILITY = INCONCLUSIVE
SURROGATE_REPAIR_BLOCKED = YES
NEXT_STAGE = REESTABLISH_VALID_CANDIDATE_Q_DIAGNOSTIC_WITH_CORRECT_CRITIC_LIFECYCLE
```

The +118 labels are not currently reliable production-teacher evidence. The
next diagnostic must preserve its critic checkpoint/source, critic optimizer
steps, PopArt state, and raw KEEP/candidate reward and next-state value fields
in one run record. No critic retraining, surrogate change, architecture change,
or formal epoch is authorized or performed in this gate. The frozen Trigger,
screening, and training states remain unchanged:

```text
CACHE_TRIGGER_CREDIT_SOURCE = FULL_INFORMATION_BEST_UPDATE_VALUE_TEACHER
TRIGGER_TEACHER_EXPECTATION_UPDATE_IMPLEMENTED = NO
CACHE_TRIGGER_CANDIDATE_INDEPENDENT_KL_STOP = DEFERRED
ONE_EPOCH_TRAINING_EXECUTED = NO
ONE_EPOCH_COMPLETED_EPOCHS = 0
EPOCH_2_STARTED = NO
READY_FOR_FORMAL_TRAINING_STAGE = NO
FORMAL_300_EPOCH_TRAINING_EXECUTED = NO
CONDITIONAL_CANDIDATE_HYBRID_RESOLVED = NO
```

## Candidate Critic Lifecycle Gate

Date: 2026-09-28

This gate traces the fresh proposed-joint trainer and its first and later
Candidate-teacher calls. No real rollout, PPO update, formal epoch, or resume
was run. The initialization smoke instantiated the configured trainer only.
The PopArt smoke used a fixed synthetic observation.

CURRENT_HEAD = 789726e254023e36a68bd8d1ea04b807e618fa37
ARCHITECTURE_VERSION = 6
CHECKPOINT_VERSION = 25

FRESH_TRAIN_EVENT_ORDER = [E1, E3, E4, E5, E6, E7, E8, E11, E9, E10]
E1 = trainer/policy/critic-optimizer initialization
E2 = checkpoint load (absent in fresh training)
E3 = collect first rollout
E4 = compute GAE/returns
E5 = critic return-statistics update
E6 = policy PopArt statistics update
E7 = evaluate full-information values
E8 = construct Cache Trigger/Candidate teacher tensors
E9 = first critic-loss backward
E10 = first critic optimizer step
E11 = first actor optimizer step
E2 = ABSENT (resume=false)

CRITIC_OPTIMIZER_STEP_COUNT_AT_INIT = 0
CRITIC_PARAMETER_SOURCE_AT_INIT = fresh PyTorch policy-constructor initialization
CRITIC_CHECKPOINT_LOADED_AT_INIT = NO
POPART_MEAN_AT_INIT = 0.0
POPART_STD_AT_INIT = 1.0
CRITIC_RETURN_STAT_COUNT_AT_INIT = 0

CRITIC_STEP_BEFORE_FIRST_ROLLOUT = 0
CRITIC_STEP_AFTER_FIRST_ROLLOUT = 0
CRITIC_PARAMETERS_CHANGED_DURING_ROLLOUT = NO
POPART_CHANGED_DURING_ROLLOUT = NO

FIRST_CRITIC_OPTIMIZER_STEP_BEFORE_CANDIDATE_TEACHER = NO
FIRST_CANDIDATE_TEACHER_CRITIC_OPTIMIZER_STEPS = 0
FIRST_CANDIDATE_TEACHER_USES_ZERO_STEP_CRITIC = YES
FIRST_CANDIDATE_TEACHER_CRITIC_WEIGHTS_UPDATED_FROM_CURRENT_ROLLOUT = NO (no critic optimizer step; PopArt reparameterization is separate)
FIRST_CANDIDATE_TEACHER_USES_NEW_POPART_WITH_PREUPDATE_CRITIC_WEIGHTS = YES (pre-update learned weights; output head reparameterized)
FIRST_CANDIDATE_TEACHER_POPART_UPDATED_FROM_CURRENT_RETURNS = YES
FIRST_CANDIDATE_TEACHER_RETURN_STAT_COUNT = 300 (complete 300-slot rollout)
FIRST_CANDIDATE_TEACHER_POPART_MEAN = first-rollout Welford mean (not numerically observed)
FIRST_CANDIDATE_TEACHER_POPART_STD = sqrt(first-rollout M2 / count + 1e-8; not numerically observed)

SECOND_CANDIDATE_TEACHER_CRITIC_OPTIMIZER_STEPS = 40
SECOND_CANDIDATE_TEACHER_USES_POST_PREVIOUS_UPDATE_CRITIC = YES
CANDIDATE_TEACHER_USES_PRE_CURRENT_ROLLOUT_CRITIC_WEIGHTS = YES

POPART_REPARAMETERIZES_CRITIC_OUTPUT_TO_PRESERVE_RAW_VALUE = YES
POPART_RAW_VALUE_PRESERVATION_ERROR = 1.7136335372924805e-07 (synthetic fixed-observation smoke)

CURRENT_COMPATIBLE_V25_CHECKPOINT_AVAILABLE = NO
RESUME_RESTORES_CRITIC_WEIGHTS = YES (static schema path)
RESUME_RESTORES_CRITIC_OPTIMIZER_STATE = YES (static schema path)
RESUME_RESTORES_CRITIC_OPTIMIZER_STEPS = YES (static schema path)
RESUME_RESTORES_POPART_STATE = YES (policy state_dict buffers; static schema path)
RESUME_RESTORES_RETURN_STATISTICS = YES (Welford count/mean/M2; static schema path)
RESUME_RUNTIME_SMOKE = NOT RUN (no version-25 checkpoint available)

CANDIDATE_Q_CRITIC_REFERENCE = CURRENT_LIVE_CRITIC
CANDIDATE_TEACHER_DEPENDS_ON_CRITIC = YES
CRITIC_TARGET_DEPENDS_ON_CANDIDATE_TEACHER_IN_SAME_UPDATE = NO
DIRECT_SAME_UPDATE_CIRCULAR_DEPENDENCY = NO

CANDIDATE_TEACHER_COLD_START = YES
CANDIDATE_TEACHER_PREUPDATE_STALENESS = YES
RISK_SCOPE = EVERY_UPDATE_PRE_CURRENT_ROLLOUT (first fresh call is also zero-step)
CANDIDATE_CRITIC_LIFECYCLE_CLASSIFICATION = CANDIDATE_CRITIC_PREUPDATE_STALENESS
NEXT_STAGE = CANDIDATE_CRITIC_UPDATE_ORDER_DESIGN_GATE

TRAINER_INITIALIZATION_SMOKE = PASS
POPART_RAW_VALUE_PRESERVATION_SMOKE = PASS
FORMAL_ROLLOUT_OR_PPO_UPDATE_EXECUTED = NO
ONE_EPOCH_TRAINING_EXECUTED = NO
ONE_EPOCH_COMPLETED_EPOCHS = 0
EPOCH_2_STARTED = NO
READY_FOR_FORMAL_TRAINING_STAGE = NO
FORMAL_300_EPOCH_TRAINING_EXECUTED = NO
GIT_DIFF_CHECK = PASS

### Evidence and interpretation

The fresh runner constructs PPOV2Trainer before rollout collection; its
resume branch is the only path that calls load_checkpoint
(src/ppo_simple/trainer.py:11774-11815). The trainer creates the proposed
joint policy and separate critic optimizer, initializes optimizer counters to
zero, and creates empty Welford return statistics
(src/ppo_simple/trainer.py:4130-4139,4280-4282,4360-4372). The initialization
smoke confirmed policy schema adaptive_joint_cache_uav_ppo, architecture 6,
checkpoint version 25, step count 0, return count 0, and PopArt (0,1).

collect_chunk delegates to collect_rollout
(src/ppo_simple/trainer.py:4866-4883); rollout collection contains no
optimizer step or PopArt/statistics update. In the formal loop, collection
precedes trainer.update (src/ppo_simple/trainer.py:12144-12161). Inside the
update, GAE/returns are computed first (6505-6512), then Welford statistics
are updated and PopArt is transformed (6563-6568), then the live full
information evaluator runs (6812-6818) and teacher tensors are built
(6827-6842). The first actor optimizer step occurs before the first critic
loss backward (8444-8454,8655); the first critic optimizer step follows
that backward (8671-8672). Thus the first Cache Candidate teacher uses a
zero-step critic even though its raw output head has already been
reparameterized by PopArt.

The first teacher's numerical PopArt mean/std require the actual first-rollout
returns. They were not measured because this gate did not collect a rollout.
For the configured 300-slot episode, the statistics update would add 300
returns; after warmup, active std is derived from the population Welford
variance plus epsilon. The focused synthetic smoke on the configured
AdaptiveJointCacheUAVPPOActorCritic policy changed PopArt from (0,1) to
(2.75,0.625), then measured raw-value error 1.7136335372924805e-07 on a fixed
observation. Source implements the
output-head transform at src/ppo_simple/policy.py:891-900, preserving raw
V(s) while changing normalized-space parameters.

The next teacher is called only after the preceding trainer.update has
finished, so it sees exactly 40 completed critic steps. The proposed
configuration runs 4 epochs with 10 minibatches; every minibatch steps the
critic. Branch-specific KL gates can freeze actor branches, but the loop
explicitly continues all configured critic steps
(src/ppo_simple/trainer.py:9496-9500).
Every teacher call still precedes the critic optimizer steps for its own
rollout. This is a persistent pre-current-rollout ordering risk for
counterfactual Candidate labels, while only the fresh first call has step
count zero; it does not establish that any critic value or Q label is
invalid.

The evaluator reads trainer.policy.critic_value directly in
src/ppo_simple/cache_credit.py:521-535; it uses the current live critic, not a
frozen copy. The accepted Exact-Q semantics remain as recorded in the prior
Candidate Exact-Q Label Integrity Gate section of this report: branch reward
plus discounted live-critic bootstrap. No historical +118 reconstruction or
Candidate-Q decomposition was repeated. GAE returns and critic_targets_t are
formed independently of the Candidate teacher
(src/ppo_simple/trainer.py:6505-6512,6978-7175), so there is no direct
same-update circular dependency.

Checkpoint inventory found 293 checkpoint-like .pt/.pth files under runs/;
none has checkpoint version 25. Static load logic restores policy_state_dict
(including PopArt buffers), requires and loads the proposed critic optimizer
state, restores critic_optimizer_steps, restores Welford count/mean/M2, then
checks PopArt/Welford consistency
(src/ppo_simple/trainer.py:11028-11030,11129-11141,11271-11302,11313-11335,11361,11475-11505,11532-11549).
No runtime resume test was attempted because no compatible version-25
checkpoint exists.

The lifecycle classification captures the repeatable pre-current-rollout
critic order, with fresh zero-step cold start as a special first-call case.
It does not establish that critic values or Q labels are invalid or broken.
If a later design gate is authorized, preserve these three
unimplemented options for review: CRITIC_WARMUP_BEFORE_CANDIDATE_TEACHER,
CRITIC_ONLY_UPDATE_BEFORE_CANDIDATE_TEACHER, and
TEMPORARY_REWARD_ONLY_CANDIDATE_CREDIT_DURING_COLD_START. No option was
selected or implemented here. Existing frozen states remain
CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION = CANDIDATE_Q_INTEGRITY_INCONCLUSIVE,
COUNTERFACTUAL_CRITIC_RELIABILITY = INCONCLUSIVE,
SURROGATE_REPAIR_BLOCKED = YES,
CONDITIONAL_CANDIDATE_HYBRID_RESOLVED = NO,
TRIGGER_TEACHER_EXPECTATION_UPDATE_IMPLEMENTED = NO, and
CACHE_TRIGGER_CANDIDATE_INDEPENDENT_KL_STOP = DEFERRED.

## Candidate Critic Cold-Start Repair

Date: 2026-09-28

The lifecycle evidence above showed that a fresh proposed-joint trainer called
critic-bootstrap teachers before its first critic step. The old order was
rollout → GAE/PopArt → Cache and UAV Q teachers → actor and critic PPO steps;
its first teacher therefore used a zero-step critic. This repair adds a
pre-epoch critic-only warmup. The synthetic warmup rollout is discarded before
formal Epoch 1 collects a new rollout.

The configured protocol is `full_episode_randomized` with 300 TRAIN slots.
Warmup derives its episode realization, generation, collection, and optimizer
seeds under the `ppo_critic_bootstrap_warmup` namespace. It uses the existing
deterministic episode-seed derivation without changing formal episode stream
values, collects only TRAIN origins, and saves/restores Python, NumPy, and
Torch RNG state around auxiliary work. Warmup does not advance the formal
episode seed or pool index.

Code tracing identified all four proposed critic-bootstrap teacher branches:
Cache Trigger and Cache Candidate use the Cache full-information evaluator;
UAV Operation and UAV Candidate use the exact UAV counterfactual Q path. The
proposed `update()` gate, direct Cache evaluator, direct UAV exact-credit path,
Cache surrogate trainer, and Cache surrogate gate all fail before warmup is
complete. The warmup method does not call any of them, and it does not write
Q-label or surrogate replay/history state.

Warmup computes GAE/returns, updates return statistics and PopArt, then uses
the same `_critic_minibatch_update` loss, backward, gradient clipping, and
optimizer step used by formal proposed PPO. It executes exactly
`update_epochs × num_minibatches` critic steps. Actor parameter groups and
optimizer counters are checked for zero change. The formal rollout/update
order remains teacher first, actor update, then critic update: mature-stage
pre-current-rollout staleness is unchanged.

For formal Q provenance, the recorded critic digest and PopArt mean/std are
captured after that rollout's return-statistics/PopArt transform and immediately
before Q teacher generation. The optimizer-step field remains the count before
the current rollout's critic PPO updates.

```text
CURRENT_HEAD = 789726e254023e36a68bd8d1ea04b807e618fa37
ARCHITECTURE_VERSION = 6
CHECKPOINT_VERSION = 26
CRITIC_BOOTSTRAPPED_TEACHER_BRANCHES = [Cache Trigger, Cache Candidate, UAV Operation, UAV Candidate]
PRE_EPOCH_CRITIC_WARMUP_IMPLEMENTED = YES
WARMUP_COUNTS_AS_FORMAL_EPOCH = NO
WARMUP_CRITIC_STEPS_PLANNED = 40
WARMUP_CRITIC_STEPS_EXECUTED = 40 (focused synthetic rollout)
WARMUP_CACHE_TRIGGER_PARAM_DELTA = 0
WARMUP_CACHE_CANDIDATE_PARAM_DELTA = 0
WARMUP_UAV_OPERATION_PARAM_DELTA = 0
WARMUP_UAV_CANDIDATE_PARAM_DELTA = 0
WARMUP_CACHE_FULL_INFORMATION_TEACHER_CALLS = 0
WARMUP_UAV_EXACT_TEACHER_CALLS = 0
WARMUP_CACHE_SURROGATE_UPDATES = 0
WARMUP_CRITIC_PARAM_DELTA = 2.602162967 (seeded synthetic run)
POPART_RAW_VALUE_PRESERVATION_ERROR = 5.960464477539063e-08 (seeded synthetic run)
CRITIC_BOOTSTRAP_TEACHER_READY_GATE_IMPLEMENTED = YES
FIRST_FORMAL_TEACHER_CRITIC_OPTIMIZER_STEPS = 40 (ready-state contract; no formal teacher executed)
ZERO_STEP_CACHE_TEACHER_POSSIBLE = NO
ZERO_STEP_UAV_TEACHER_POSSIBLE = NO
ZERO_STEP_SURROGATE_LABEL_POSSIBLE = NO
PREUPDATE_STALENESS_SEMANTICS_CHANGED = NO
CRITIC_PARAMETER_DIGEST_RECORDED = YES
POPART_PROVENANCE_RECORDED = YES
Q_COMPONENT_DIAGNOSTICS_AVAILABLE = YES (compact KEEP/winner and UAV branch components)
MAX_Q_COMPONENT_RECONSTRUCTION_ERROR = 0 (synthetic identity smoke; no environment Exact-Q run)
NON_PROPOSED_WARMUP_BEHAVIOR_CHANGED = NO
PRE_EPOCH_CRITIC_WARMUP_COMPLETED = YES (synthetic only)
FORMAL_COMPLETED_EPOCHS = 0
ONE_EPOCH_TRAINING_EXECUTED = NO
READY_FOR_FORMAL_TRAINING_STAGE = NO
```

The 300-slot synthetic smoke completed all 40 configured critic steps, changed
critic parameters by a finite nonzero norm, and changed no actor parameters.
Instrumentation observed zero Cache/UAV teacher calls, zero surrogate train or
gate calls, zero actor optimizer steps, and 40 critic optimizer steps. Global
Python/NumPy/Torch RNG state was unchanged after warmup. PopArt preserved raw
critic values within `5.96e-8`. The Q identity smoke checked both a terminal
branch (`next_v = 0`) and a nonterminal branch against `q = reward + gamma ×
next_v`; maximum synthetic reconstruction error was zero. Direct teacher and
surrogate entry points failed closed before warmup and passed the readiness
gate after it.

Checkpoint v26 save/load smokes covered both incomplete pre-warmup and complete
post-warmup state. Resume restored the recorded 40 steps and skipped a second
warmup; duplicate warmup, inconsistent complete/zero-step state, and proposed
v25 checkpoints were rejected. Config/schema loading passed for the proposed
and Cache-only contexts. The Cache-only policy does not require this lifecycle.
Architecture version remains 6 because tensor topology is unchanged.

```text
CANDIDATE_CRITIC_ZERO_STEP_TEACHER_RESOLVED = YES
PRE_EPOCH_CRITIC_WARMUP_IMPLEMENTED = YES
CRITIC_BOOTSTRAP_TEACHER_READY_GATE_IMPLEMENTED = YES
CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION = CANDIDATE_Q_INTEGRITY_INCONCLUSIVE
COUNTERFACTUAL_CRITIC_RELIABILITY = INCONCLUSIVE
CONDITIONAL_CANDIDATE_HYBRID_RESOLVED = NO
SURROGATE_REPAIR_BLOCKED = YES
ONE_EPOCH_TRAINING_EXECUTED = NO
READY_FOR_FORMAL_TRAINING_STAGE = NO
NEXT_STAGE = TRACEABLE_CANDIDATE_Q_INTEGRITY_GATE
```

At completion of the preceding Candidate Critic Cold-Start Repair stage, no
real warmup rollout, formal PPO update, full environment Exact-Q diagnosis, or
epoch had been run. The measured Q identity result in that section is synthetic
only. The subsequent one-off Traceable Candidate-Q attempt and its reporting
failure are recorded below. Existing Trigger/Candidate objectives, KL stopping
semantics, and Hybrid quality conclusions remain unchanged.

## Traceable Candidate-Q Integrity Gate

**Gate status: `CANDIDATE_Q_INTEGRITY_INCONCLUSIVE`.** The required single
TRAIN-only diagnostic attempt executed both complete 300-slot realizations,
the 40-step critic-only warmup, and the targeted full-exact Cache evaluation.
The computation passed its lifecycle, actor-isolation, frozen-Critic,
full-exact/no-surrogate, finite-value, and Q / delta-Q identity assertions.
The run then stopped while assembling its final result object because the
reporting script referenced an unbound local name. It emitted no result JSON;
the temporary checkpoint and realization directories were removed by the
scratch cleanup. Consequently, the numerical metrics and checkpoint hash were
not preserved and cannot support a reliability classification.

```text
CURRENT_HEAD = 4ebc2a94e219242f8023c9d35e3c36a960e935fb
GIT_DIRTY_BEFORE = YES (pre-existing untracked artifacts and reports)
ARCHITECTURE_VERSION = 6
CHECKPOINT_VERSION = 26
WARMUP_REALIZATION_SEED = 7868781415413077432
DIAGNOSTIC_REALIZATION_SEED = 12184115623283823756
FORMAL_EPISODE_SEED_STREAM_CHANGED = NO (assertion passed)
REAL_WARMUP_ROLLOUT_EXECUTED = YES
REAL_WARMUP_TRANSITIONS = 300
REAL_WARMUP_CRITIC_STEPS = 40
REAL_WARMUP_ACTOR_PARAM_DELTA = 0
DIAGNOSTIC_ROLLOUT_EXECUTED = YES
DIAGNOSTIC_TRANSITIONS = 300
DIAGNOSTIC_TRANSITION_INDICES = 30, 149, 268
DIAGNOSTIC_ORIGIN_INDICES = 30, 149, 268
FULL_EXACT_USED = YES (all Cache heads, all legal candidates, three selected transitions)
SURROGATE_TRAINING_EXECUTED = NO
SURROGATE_Q_USED_FOR_DIAGNOSTIC = NO
CRITIC_FROZEN_DURING_Q_DIAGNOSTIC = YES (digest/step/PopArt/Welford assertions passed)
MAX_Q_RECONSTRUCTION_ERROR = NOT PRESERVED
MAX_DELTA_Q_RECONSTRUCTION_ERROR = NOT PRESERVED
CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION = CANDIDATE_Q_INTEGRITY_INCONCLUSIVE
COUNTERFACTUAL_CRITIC_RELIABILITY = INCONCLUSIVE
FORMAL_PPO_UPDATE_EXECUTED = NO
ACTOR_OPTIMIZER_STEPS_DURING_DIAGNOSTIC = 0
ONE_EPOCH_TRAINING_EXECUTED = NO
READY_FOR_FORMAL_TRAINING_STAGE = NO
NEXT_STAGE = CANDIDATE_Q_INTEGRITY_FOLLOWUP
RESULT_JSON_PERSISTED = NO
TEMPORARY_REALIZATIONS_AND_CHECKPOINT_REMAIN = NO
```

The selected transitions were determined by nonterminal rollout position
(10%, 50%, 90%). The reusable entry point's result-assembly reference error
has been fixed and passed Python compilation and proposed-config validation.
At the time of the first-attempt report, no retry had started. The user later authorized one compensatory rerun, documented below.

### Authorized compensatory rerun

```text
AUTHORIZED_RERUN = YES
RERUN_REASON = RESULT_ASSEMBLY_BUG_AFTER_SUCCESSFUL_COMPUTATION
CURRENT_HEAD = 4ebc2a94e219242f8023c9d35e3c36a960e935fb
GIT_DIRTY_BEFORE = YES
ARCHITECTURE_VERSION = 6
CHECKPOINT_VERSION = 26
WARMUP_REALIZATION_SEED = 7868781415413077432
DIAGNOSTIC_REALIZATION_SEED = 12184115623283823756
FORMAL_EPISODE_SEED_STREAM_CHANGED = NO
REAL_WARMUP_TRANSITIONS = 300
REAL_WARMUP_CRITIC_STEPS = 40
REAL_WARMUP_ACTOR_PARAM_DELTA = 0
DIAGNOSTIC_TRANSITIONS = 300
DIAGNOSTIC_TRANSITION_INDICES = [30, 149, 268]
DIAGNOSTIC_ORIGIN_INDICES = [30, 149, 268]
PRE_WARMUP_CRITIC_MAE_TO_MC_RETURN = 550.4785424
POST_WARMUP_CRITIC_MAE_TO_MC_RETURN = 537.0688339
PRE_WARMUP_CRITIC_RMSE_TO_MC_RETURN = 712.8836233
POST_WARMUP_CRITIC_RMSE_TO_MC_RETURN = 695.5799062
PRE_WARMUP_CRITIC_EV_TO_MC_RETURN = 6.920502044e-05
POST_WARMUP_CRITIC_EV_TO_MC_RETURN = 0.006001823389
PRE_WARMUP_CRITIC_MAE_TO_GAE_RETURN = 103.4555118
POST_WARMUP_CRITIC_MAE_TO_GAE_RETURN = 93.95959325
GAE_RETURN_VS_MC_RETURN_MAE = 453.7112476
GAE_RETURN_VS_MC_RETURN_RMSE = 615.2592624
GAE_RETURN_VS_MC_RETURN_CORRELATION = 0.5549572402
CRITIC_WARMUP_REAL_DATA_CLASSIFICATION = IMPROVES
CRITIC_PARAMETER_DIGEST = 378d8fe6dc261574f598905dbb5446a4d84bbeb43dd8386ad8bf50b6ce5072d6
CRITIC_OPTIMIZER_STEPS = 40
POPART_MEAN = -50.25922012
POPART_STD = 115.2703781
RETURN_STAT_COUNT = 300
CRITIC_DIGEST_BEFORE_Q_DIAGNOSTIC = 378d8fe6dc261574f598905dbb5446a4d84bbeb43dd8386ad8bf50b6ce5072d6
CRITIC_DIGEST_AFTER_Q_DIAGNOSTIC = 378d8fe6dc261574f598905dbb5446a4d84bbeb43dd8386ad8bf50b6ce5072d6
CRITIC_STEPS_BEFORE_Q_DIAGNOSTIC = 40
CRITIC_STEPS_AFTER_Q_DIAGNOSTIC = 40
CRITIC_FROZEN_DURING_Q_DIAGNOSTIC = YES
FULL_EXACT_USED = YES
SURROGATE_TRAINING_EXECUTED = NO
SURROGATE_Q_USED_FOR_DIAGNOSTIC = NO
FULL_EXACT_CACHE_HEAD_COUNT = 9
FULL_EXACT_CANDIDATE_COUNT = 2859
MAX_Q_RECONSTRUCTION_ERROR = 0
MAX_DELTA_Q_RECONSTRUCTION_ERROR = 7.105427358e-15
SV_DELTA_REWARD_ABS_MEDIAN = 0.2602013552
SV_GAMMA_DELTA_V_ABS_MEDIAN = 2.235427856
SV_DELTA_Q_ABS_MEDIAN = 2.756056592
SV_REWARD_ABS_SHARE_MEDIAN = 0.1469618124
SV_CRITIC_ABS_SHARE_MEDIAN = 0.8530381876
RSU_DELTA_REWARD_ABS_MEDIAN = 1.407806678
RSU_GAMMA_DELTA_V_ABS_MEDIAN = 7.246358871
RSU_DELTA_Q_ABS_MEDIAN = 7.09587047
RSU_REWARD_ABS_SHARE_MEDIAN = 0.1620245674
RSU_CRITIC_ABS_SHARE_MEDIAN = 0.8379754326
UAV_DELTA_REWARD_ABS_MEDIAN = 3.290900853
UAV_GAMMA_DELTA_V_ABS_MEDIAN = 3.696024895
UAV_DELTA_Q_ABS_MEDIAN = 6.052085568
UAV_REWARD_ABS_SHARE_MEDIAN = 0.4926032791
UAV_CRITIC_ABS_SHARE_MEDIAN = 0.5073967209
ACTUAL_STATE_CRITIC_P01 = -53.8619709
ACTUAL_STATE_CRITIC_P50 = -22.8273735
ACTUAL_STATE_CRITIC_P99 = 9.848804817
COUNTERFACTUAL_NEXT_V_P01 = -27.58166672
COUNTERFACTUAL_NEXT_V_P50 = -10.91609192
COUNTERFACTUAL_NEXT_V_P99 = 12.80189171
SV_COUNTERFACTUAL_BELOW_ACTUAL_P01_FRACTION = 0
SV_COUNTERFACTUAL_ABOVE_ACTUAL_P99_FRACTION = 0.08771929825
RSU_COUNTERFACTUAL_BELOW_ACTUAL_P01_FRACTION = 0
RSU_COUNTERFACTUAL_ABOVE_ACTUAL_P99_FRACTION = 0.02161547213
UAV_COUNTERFACTUAL_BELOW_ACTUAL_P01_FRACTION = 0
UAV_COUNTERFACTUAL_ABOVE_ACTUAL_P99_FRACTION = 0.02573529412
UAV_0_MAX_DELTA_Q = 389.5752473
UAV_0_MAX_DELTA_REWARD = 379.7651887
UAV_0_MAX_GAMMA_DELTA_V = 9.810058594
UAV_1_MAX_DELTA_Q = 16.48378222
UAV_1_MAX_DELTA_REWARD = 6.934286521
UAV_1_MAX_GAMMA_DELTA_V = 9.549495697
CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION = CANDIDATE_Q_INTEGRITY_INCONCLUSIVE
COUNTERFACTUAL_CRITIC_RELIABILITY = INCONCLUSIVE
FORMAL_PPO_UPDATE_EXECUTED = NO
ACTOR_OPTIMIZER_STEPS_DURING_DIAGNOSTIC = 0
ONE_EPOCH_TRAINING_EXECUTED = NO
READY_FOR_FORMAL_TRAINING_STAGE = NO
NEXT_STAGE = CANDIDATE_Q_INTEGRITY_FOLLOWUP
```

Critic value metrics use the same 300 diagnostic observations before and after warmup. GAE uses the saved zero-step rollout values, configured gamma, and configured lambda; Monte Carlo reward-to-go ends at the terminal transition.

| Role | Candidates | |Δreward| median / p90 / p95 / max | |γΔV| median / p90 / p95 / max | |ΔQ| median / p90 / p95 / max | Reward share mean / median / p90 | Critic share mean / median / p90 | Pearson/Spearman ΔQ vs reward | Pearson/Spearman ΔQ vs γΔV |
|---|---:|---|---|---|---|---|---|
| SV | 285 | 0.2602013552 / 4.136747747 / 6.3439839 / 6.833701194 | 2.235427856 / 6.144077301 / 7.396311188 / 12.37960625 | 2.756056592 / 7.607426162 / 9.284178269 / 18.58031449 | 0.2690933227 / 0.1469618124 / 0.7054318865 | 0.7309066773 / 0.8530381876 / 0.9840922459 | 0.6438846998 / 0.5342994887 | 0.8633179014 / 0.8800732996 |
| RSU | 1758 | 1.407806678 / 4.007839246 / 9.35026122 / 17.77264729 | 7.246358871 / 17.15751057 / 20.81487932 / 34.26096916 | 7.09587047 / 19.67021048 / 25.47029157 / 47.69004243 | 0.235382972 / 0.1620245674 / 0.5660325429 | 0.764617028 / 0.8379754326 / 0.9784923472 | 0.5349564802 / 0.3212520739 | 0.9229228362 / 0.9426290642 |
| UAV | 816 | 3.290900853 / 224.0250045 / 224.9316422 / 379.9718206 | 3.696024895 / 8.394166946 / 9.787957191 / 15.26664734 | 6.052085568 / 225.4814441 / 232.2847945 / 389.5752473 | 0.5219174546 / 0.4926032791 / 0.9778284244 | 0.4780825454 / 0.5073967209 / 0.9238238305 | 0.999162728 / 0.8094891346 | 0.3947684629 / 0.7019496605 |

| Role | Candidate next-V min / p01 / p05 / p50 / p95 / p99 / max | Below actual P01 | Above actual P99 |
|---|---|---:|---:|
| SV | -28.71353722 / -23.65862167 / -20.40074577 / -7.923442841 / 10.91713562 / 12.39556793 / 15.06754684 | 0 | 0.08771929825 |
| RSU | -34.558918 / -28.88200991 / -24.73149223 / -11.65555954 / 6.301867676 / 13.16453991 / 18.66657639 | 0 | 0.02161547213 |
| UAV | -27.37845039 / -23.99395781 / -21.71916723 / -9.788728714 / 8.206559181 / 11.65721321 / 14.94444656 | 0 | 0.02573529412 |

| Transition | Head | Legal Candidates | KEEP reward / next-V / Q | Exact best config | Exact best reward / next-V / Q | Δreward / γΔV / ΔQ | Reward / critic share |
|---:|---|---:|---|---|---|---|---|
| 30 | 45 | 19 | 4.305649603 / -16.33393097 / -12.02828137 | `[0, 0, 0, 0, 0, 1, 0, 1, 0, 0]` | 4.206153452 / -8.901542664 / -4.695389211 | -0.09949615041 / 7.432388306 / 7.332892155 | 0.01320999426 / 0.9867900057 |
| 30 | 18 | 19 | 4.305649603 / -16.33393097 / -12.02828137 | `[0, 0, 0, 0, 1, 0, 0, 0, 1, 0]` | 4.251919641 / -14.38381577 / -10.13189612 | -0.05372996237 / 1.950115204 / 1.896385241 | 0.02681343014 / 0.9731865699 |
| 30 | 14 | 19 | 5.534857858 / -14.51829147 / -8.983433615 | `[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]` | 5.514857858 / -13.29301834 / -7.778160483 | -0.02 / 1.225273132 / 1.205273132 | 0.01606073357 / 0.9839392664 |
| 30 | 35 | 19 | 4.305649603 / -16.33393097 / -12.02828137 | `[0, 1, 0, 0, 0, 1, 0, 0, 0, 0]` | 6.554357242 / -12.57905579 / -6.024698544 | 2.248707639 / 3.754875183 / 6.003582822 | 0.3745609423 / 0.6254390577 |
| 30 | 26 | 19 | 4.547883657 / -14.86705399 / -10.31917033 | `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]` | 4.228609908 / -11.66334152 / -7.434731614 | -0.3192737486 / 3.203712463 / 2.884438715 | 0.09062588652 / 0.9093741135 |
| 30 | RSU_0 | 293 | 4.305649603 / -16.33393097 / -12.02828137 | `[0, 1, 0, 0, 0, 1, 1, 1, 0, 0]` | 4.14296061 / 0.4677085876 / 4.610669197 | -0.1626889932 / 16.80163956 / 16.63895056 | 0.009590063805 / 0.9904099362 |
| 30 | RSU_1 | 293 | 4.412647584 / -8.036445618 / -3.623798034 | `[0, 0, 0, 0, 0, 1, 1, 0, 1, 1]` | 6.970716023 / -0.9270439148 / 6.043672108 | 2.558068439 / 7.109401703 / 9.667470142 | 0.264605776 / 0.735394224 |
| 30 | UAV_0 | 136 | 4.305649603 / -16.33393097 / -12.02828137 | `[1, 0, 0, 0, 0, 0, 0, 0, 1, 0]` | 4.775285874 / -8.284778595 / -3.509492721 | 0.4696362709 / 8.049152374 / 8.518788645 | 0.05512946623 / 0.9448705338 |
| 30 | UAV_1 | 136 | 4.305649603 / -16.33393097 / -12.02828137 | `[0, 0, 0, 0, 0, 1, 1, 0, 0, 1]` | 10.43069346 / -11.56073761 / -1.130044152 | 6.125043855 / 4.773193359 / 10.89823721 | 0.562021521 / 0.437978479 |
| 149 | 45 | 19 | 5.414660576 / 7.59998703 / 13.01464761 | `[0, 0, 0, 0, 0, 1, 0, 1, 0, 0]` | 1.474491036 / 15.06754684 / 16.54203788 | -3.94016954 / 7.467559814 / 3.527390275 | 0.3453947247 / 0.6546052753 |
| 149 | 18 | 19 | 5.414660576 / 7.59998703 / 13.01464761 | `[0, 0, 0, 0, 1, 0, 0, 0, 1, 0]` | 6.313083935 / 12.28580475 / 18.59888868 | 0.898423359 / 4.685817719 / 5.584241078 | 0.1608854895 / 0.8391145105 |
| 149 | 14 | 19 | 5.414660576 / 7.59998703 / 13.01464761 | `[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]` | 6.223075205 / 9.437080383 / 15.66015559 | 0.8084146296 / 1.837093353 / 2.645507983 | 0.3055801135 / 0.6944198865 |
| 149 | 35 | 19 | 5.414660576 / 7.59998703 / 13.01464761 | `[0, 1, 0, 0, 0, 0, 0, 0, 0, 1]` | 5.448770669 / 14.49336624 / 19.94213691 | 0.03411009376 / 6.893379211 / 6.927489305 | 0.00492387534 / 0.9950761247 |
| 149 | 26 | 19 | 5.414660576 / 7.59998703 / 13.01464761 | `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]` | 5.256092032 / 12.97182465 / 18.22791668 | -0.1585685439 / 5.371837616 / 5.213269072 | 0.02867213353 / 0.9713278665 |
| 149 | RSU_0 | 293 | 5.414660576 / 7.59998703 / 13.01464761 | `[0, 1, 0, 0, 0, 1, 1, 1, 0, 0]` | 5.248341404 / 18.66657639 / 23.91491779 | -0.1663191714 / 11.06658936 / 10.90027018 | 0.01480642089 / 0.9851935791 |
| 149 | RSU_1 | 293 | 5.414660576 / 7.59998703 / 13.01464761 | `[0, 1, 0, 0, 1, 1, 0, 0, 1, 1]` | 6.396116949 / 4.338718414 / 10.73483536 | 0.9814563738 / -3.261268616 / -2.279812242 | 0.2313268893 / 0.7686731107 |
| 149 | UAV_0 | 136 | -374.2954824 / 3.928798676 / -370.3666837 | `[0, 0, 0, 1, 1, 0, 0, 0, 1, 0]` | 5.469706383 / 13.73885727 / 19.20856365 | 379.7651887 / 9.810058594 / 389.5752473 | 0.9748185783 / 0.02518142172 |
| 149 | UAV_1 | 136 | -0.4923074473 / 0.4707870483 / -0.021520399 | `[0, 1, 0, 0, 0, 0, 1, 0, 0, 1]` | 6.441979074 / 10.02028275 / 16.46226182 | 6.934286521 / 9.549495697 / 16.48378222 | 0.4206732672 / 0.5793267328 |
| 268 | 45 | 19 | -0.9045524068 / -8.396751404 / -9.301303811 | `[0, 0, 0, 0, 0, 1, 0, 1, 0, 0]` | 1.069304754 / -2.381473541 / -1.312168787 | 1.973857161 / 6.015277863 / 7.989135023 | 0.2470676932 / 0.7529323068 |
| 268 | 18 | 19 | 5.20036097 / -6.048057556 / -0.847696586 | `[0, 0, 0, 0, 1, 0, 0, 0, 1, 0]` | 6.879852307 / -3.002391815 / 3.877460491 | 1.679491336 / 3.045665741 / 4.725157077 | 0.3554360858 / 0.6445639142 |
| 268 | 14 | 19 | -0.9245524068 / -7.444255829 / -8.368808236 | `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]` | 3.123336977 / -4.285381317 / -1.16204434 | 4.047889384 / 3.158874512 / 7.206763895 | 0.5616792006 / 0.4383207994 |
| 268 | 35 | 19 | -0.9245524068 / -7.444255829 / -8.368808236 | `[0, 1, 0, 0, 0, 0, 0, 0, 0, 1]` | -0.8963285736 / -0.6738853455 / -1.570213919 | 0.02822383324 / 6.770370483 / 6.798594317 | 0.004151421886 / 0.9958485781 |
| 268 | 26 | 19 | -0.8434081464 / -7.086349487 / -7.929757634 | `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]` | 1.607033375 / -3.101142883 / -1.494109509 | 2.450441521 / 3.985206604 / 6.435648125 | 0.3807606434 / 0.6192393566 |
| 268 | RSU_0 | 293 | -0.9245524068 / -7.444255829 / -8.368808236 | `[0, 1, 1, 0, 0, 1, 1, 0, 0, 0]` | 3.352400022 / -0.8474311829 / 2.504968839 | 4.276952429 / 6.596824646 / 10.87377708 | 0.3933272128 / 0.6066727872 |
| 268 | RSU_1 | 293 | -0.1069228811 / -9.37921524 / -9.486138122 | `[0, 0, 0, 0, 0, 1, 1, 0, 1, 1]` | 1.156546025 / 3.4008255 / 4.557371526 | 1.263468907 / 12.78004074 / 14.04350965 | 0.08996817307 / 0.9100318269 |
| 268 | UAV_0 | 136 | 3.883232184 / -2.140300751 / 1.742931433 | `[0, 0, 1, 0, 1, 0, 0, 0, 1, 0]` | 8.176609943 / 0.1713790894 / 8.347989032 | 4.293377759 / 2.31167984 / 6.605057599 | 0.6500136743 / 0.3499863257 |
| 268 | UAV_1 | 136 | -0.9245524068 / -7.444255829 / -8.368808236 | `[0, 0, 0, 0, 0, 0, 1, 0, 1, 0]` | -0.9395524068 / -7.394638062 / -8.334190468 | -0.015 / 0.04961776733 / 0.03461776733 | 0.2321342971 / 0.7678657029 |

| Transition | UAV head | Rank | Config | Reward | next-V | Q | Δreward | γΔV | ΔQ |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 30 | UAV_0 | 1 | `[1, 0, 0, 0, 0, 0, 0, 0, 1, 0]` | 4.775285874 | -8.284778595 | -3.509492721 | 0.4696362709 | 8.049152374 | 8.518788645 |
| 30 | UAV_0 | 2 | `[0, 0, 0, 0, 1, 1, 0, 0, 1, 0]` | 8.558964915 | -13.46012497 | -4.901160055 | 4.253315312 | 2.873806 | 7.127121312 |
| 30 | UAV_0 | 3 | `[0, 0, 0, 0, 1, 0, 0, 0, 1, 0]` | 4.421025103 | -9.828620911 | -5.407595808 | 0.1153754999 | 6.505310059 | 6.620685558 |
| 30 | UAV_0 | 4 | `[0, 0, 1, 0, 1, 0, 0, 0, 1, 0]` | 4.25474285 | -9.810504913 | -5.555762063 | -0.05090675275 | 6.523426056 | 6.472519303 |
| 30 | UAV_0 | 5 | `[0, 0, 0, 1, 1, 0, 0, 0, 1, 0]` | 3.896638409 | -9.505977631 | -5.609339222 | -0.4090111941 | 6.827953339 | 6.418942145 |
| 30 | UAV_0 | 6 | `[0, 0, 0, 0, 0, 1, 0, 0, 1, 0]` | 8.59944087 | -14.25866318 | -5.659222308 | 4.293791267 | 2.075267792 | 6.369059059 |
| 30 | UAV_0 | 7 | `[0, 0, 1, 0, 0, 1, 0, 0, 1, 0]` | 8.433158617 | -14.22371674 | -5.790558119 | 4.127509014 | 2.110214233 | 6.237723248 |
| 30 | UAV_0 | 8 | `[0, 1, 0, 0, 1, 0, 0, 0, 1, 0]` | 5.861644453 | -12.09021759 | -6.228573137 | 1.555994851 | 4.243713379 | 5.79970823 |
| 30 | UAV_0 | 9 | `[0, 0, 0, 0, 1, 1, 0, 0, 1, 1]` | 8.958816509 | -15.36791611 | -6.409099599 | 4.653166906 | 0.9660148621 | 5.619181768 |
| 30 | UAV_0 | 10 | `[0, 0, 0, 1, 0, 0, 0, 0, 1, 0]` | 3.787931682 | -10.24999619 | -6.462064503 | -0.5177179206 | 6.083934784 | 5.566216863 |
| 30 | UAV_1 | 1 | `[0, 0, 0, 0, 0, 1, 1, 0, 0, 1]` | 10.43069346 | -11.56073761 | -1.130044152 | 6.125043855 | 4.773193359 | 10.89823721 |
| 30 | UAV_1 | 2 | `[0, 0, 0, 0, 0, 0, 1, 0, 0, 1]` | 7.719642075 | -9.49659729 | -1.776955215 | 3.413992472 | 6.837333679 | 10.25132615 |
| 30 | UAV_1 | 3 | `[0, 1, 0, 0, 0, 0, 1, 0, 0, 1]` | 7.652492125 | -9.667434692 | -2.014942567 | 3.346842522 | 6.666496277 | 10.0133388 |
| 30 | UAV_1 | 4 | `[0, 0, 0, 1, 0, 1, 0, 0, 0, 1]` | 10.41866424 | -14.26292419 | -3.844259953 | 6.113014639 | 2.071006775 | 8.184021414 |
| 30 | UAV_1 | 5 | `[0, 0, 0, 0, 1, 0, 1, 0, 0, 1]` | 5.449037147 | -9.434120178 | -3.985083031 | 1.143387544 | 6.899810791 | 8.043198335 |
| 30 | UAV_1 | 6 | `[0, 0, 0, 1, 0, 0, 0, 0, 0, 1]` | 7.638204706 | -12.17595673 | -4.537752021 | 3.332555103 | 4.157974243 | 7.490529346 |
| 30 | UAV_1 | 7 | `[0, 1, 0, 1, 0, 0, 0, 0, 0, 1]` | 7.571054756 | -12.37326431 | -4.802209557 | 3.265405153 | 3.960666656 | 7.226071809 |
| 30 | UAV_1 | 8 | `[0, 0, 0, 1, 1, 0, 0, 0, 0, 1]` | 5.437007931 | -12.07978821 | -6.642780277 | 1.131358328 | 4.254142761 | 5.385501089 |
| 30 | UAV_1 | 9 | `[0, 0, 0, 0, 0, 1, 0, 0, 0, 1]` | 10.62529607 | -19.35430527 | -8.7290092 | 6.319646465 | -3.020374298 | 3.299272167 |
| 30 | UAV_1 | 10 | `[0, 1, 0, 0, 0, 1, 0, 0, 0, 1]` | 10.55814612 | -19.50942039 | -8.951274277 | 6.252496515 | -3.175489426 | 3.077007089 |
| 149 | UAV_0 | 1 | `[0, 0, 0, 1, 1, 0, 0, 0, 1, 0]` | 5.469706383 | 13.73885727 | 19.20856365 | 379.7651887 | 9.810058594 | 389.5752473 |
| 149 | UAV_0 | 2 | `[0, 0, 0, 0, 1, 0, 0, 0, 1, 0]` | 5.676338209 | 13.3598175 | 19.03615571 | 379.9718206 | 9.431018829 | 389.4028394 |
| 149 | UAV_0 | 3 | `[0, 0, 1, 0, 1, 0, 0, 0, 1, 0]` | 5.510055956 | 13.32869339 | 18.83874935 | 379.8055383 | 9.399894714 | 389.205433 |
| 149 | UAV_0 | 4 | `[0, 0, 0, 1, 0, 0, 0, 0, 1, 0]` | 5.469706383 | 13.20500183 | 18.67470821 | 379.7651887 | 9.276203156 | 389.0413919 |
| 149 | UAV_0 | 5 | `[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]` | 5.676338209 | 12.81539917 | 18.49173738 | 379.9718206 | 8.886600494 | 388.8584211 |
| 149 | UAV_0 | 6 | `[0, 0, 1, 0, 0, 0, 0, 0, 1, 0]` | 5.510055956 | 12.78770447 | 18.29776042 | 379.8055383 | 8.858905792 | 388.6644441 |
| 149 | UAV_0 | 7 | `[1, 0, 0, 0, 0, 0, 0, 0, 1, 0]` | 2.667257755 | 14.94444656 | 17.61170432 | 376.9627401 | 11.01564789 | 387.978388 |
| 149 | UAV_0 | 8 | `[0, 0, 0, 0, 1, 0, 0, 0, 1, 1]` | 5.676338209 | 11.46723557 | 17.14357377 | 379.9718206 | 7.53843689 | 387.5102575 |
| 149 | UAV_0 | 9 | `[0, 1, 0, 0, 1, 0, 0, 0, 1, 0]` | 5.609188259 | 11.05490112 | 16.66408938 | 379.9046706 | 7.126102448 | 387.0307731 |
| 149 | UAV_0 | 10 | `[0, 0, 0, 0, 1, 0, 1, 0, 1, 0]` | 5.481735599 | 11.13988876 | 16.62162436 | 379.777218 | 7.211090088 | 386.988308 |
| 149 | UAV_1 | 1 | `[0, 1, 0, 0, 0, 0, 1, 0, 0, 1]` | 6.441979074 | 10.02028275 | 16.46226182 | 6.934286521 | 9.549495697 | 16.48378222 |
| 149 | UAV_1 | 2 | `[0, 0, 0, 0, 1, 0, 1, 0, 0, 1]` | 5.426689792 | 10.18514252 | 15.61183231 | 5.918997239 | 9.714355469 | 15.63335271 |
| 149 | UAV_1 | 3 | `[0, 0, 0, 0, 0, 0, 1, 0, 0, 1]` | 5.381685911 | 10.0888176 | 15.47050351 | 5.873993358 | 9.618030548 | 15.49202391 |
| 149 | UAV_1 | 4 | `[0, 1, 0, 1, 0, 0, 0, 0, 0, 1]` | 6.429949858 | 7.405925751 | 13.83587561 | 6.922257305 | 6.935138702 | 13.85739601 |
| 149 | UAV_1 | 5 | `[0, 0, 0, 0, 0, 1, 1, 0, 0, 1]` | 5.308312798 | 8.033145905 | 13.3414587 | 5.800620245 | 7.562358856 | 13.3629791 |
| 149 | UAV_1 | 6 | `[0, 0, 0, 1, 1, 0, 0, 0, 0, 1]` | 5.414660576 | 7.59998703 | 13.01464761 | 5.906968023 | 7.129199982 | 13.036168 |
| 149 | UAV_1 | 7 | `[0, 0, 0, 1, 0, 0, 0, 0, 0, 1]` | 5.369656694 | 7.493019104 | 12.8626758 | 5.861964142 | 7.022232056 | 12.8841962 |
| 149 | UAV_1 | 8 | `[0, 0, 0, 1, 0, 1, 0, 0, 0, 1]` | 5.296283582 | 5.425022125 | 10.72130571 | 5.788591029 | 4.954235077 | 10.74282611 |
| 149 | UAV_1 | 9 | `[0, 0, 0, 1, 0, 0, 1, 0, 0, 0]` | -3.219184247 | 13.60812378 | 10.38893953 | -2.7268768 | 13.13733673 | 10.41045993 |
| 149 | UAV_1 | 10 | `[0, 1, 0, 0, 1, 0, 0, 0, 0, 1]` | 6.596105729 | 2.50302887 | 9.099134599 | 7.088413177 | 2.032241821 | 9.120654998 |
| 268 | UAV_0 | 1 | `[0, 0, 1, 0, 1, 0, 0, 0, 1, 0]` | 8.176609943 | 0.1713790894 | 8.347989032 | 4.293377759 | 2.31167984 | 6.605057599 |
| 268 | UAV_0 | 2 | `[0, 0, 1, 0, 0, 0, 0, 0, 1, 0]` | 8.176609943 | -0.5309295654 | 7.645680377 | 4.293377759 | 1.609371185 | 5.902748944 |
| 268 | UAV_0 | 3 | `[0, 1, 1, 0, 0, 0, 0, 0, 1, 0]` | 8.109459993 | -2.806690216 | 5.302769777 | 4.226227809 | -0.6663894653 | 3.559838344 |
| 268 | UAV_0 | 4 | `[1, 0, 0, 0, 0, 0, 0, 0, 1, 0]` | 2.403955118 | 1.645515442 | 4.04947056 | -1.479277066 | 3.785816193 | 2.306539126 |
| 268 | UAV_0 | 5 | `[0, 0, 0, 0, 1, 0, 0, 0, 1, 0]` | 3.868232184 | 0.1534919739 | 4.021724158 | -0.015 | 2.293792725 | 2.278792725 |
| 268 | UAV_0 | 6 | `[0, 0, 0, 1, 1, 0, 0, 0, 1, 0]` | 3.272214388 | 0.57629776 | 3.848512148 | -0.6110177964 | 2.716598511 | 2.105580714 |
| 268 | UAV_0 | 7 | `[0, 0, 1, 0, 0, 1, 0, 0, 1, 0]` | 8.10323683 | -4.339336395 | 3.763900435 | 4.220004646 | -2.199035645 | 2.020969002 |
| 268 | UAV_0 | 8 | `[0, 0, 1, 0, 0, 0, 0, 1, 1, 0]` | 9.019346674 | -5.572689056 | 3.446657618 | 5.13611449 | -3.432388306 | 1.703726184 |
| 268 | UAV_0 | 9 | `[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]` | 3.868232184 | -0.5445137024 | 3.323718482 | -0.015 | 1.595787048 | 1.580787048 |
| 268 | UAV_0 | 10 | `[0, 0, 0, 1, 0, 0, 0, 0, 1, 0]` | 3.272214388 | -0.1134719849 | 3.158742403 | -0.6110177964 | 2.026828766 | 1.415810969 |
| 268 | UAV_1 | 1 | `[0, 0, 0, 0, 0, 0, 1, 0, 1, 0]` | -0.9395524068 | -7.394638062 | -8.334190468 | -0.015 | 0.04961776733 | 0.03461776733 |
| 268 | UAV_1 | 2 | `[0, 0, 0, 0, 1, 0, 1, 0, 1, 0]` | -0.9800283617 | -7.376300812 | -8.356329173 | -0.05547595485 | 0.06795501709 | 0.01247906224 |
| 268 | UAV_1 | 3 | `[0, 0, 0, 1, 1, 0, 0, 0, 1, 0]` | -2.173854839 | -10.09038925 | -12.26424409 | -1.249302432 | -2.646133423 | -3.895435855 |
| 268 | UAV_1 | 4 | `[0, 0, 0, 1, 0, 0, 0, 0, 1, 0]` | -2.133378884 | -10.14580536 | -12.27918424 | -1.208826477 | -2.70154953 | -3.910376007 |
| 268 | UAV_1 | 5 | `[0, 0, 0, 0, 1, 0, 0, 1, 1, 1]` | 0.03060716148 | -14.3338356 | -14.30322844 | 0.9551595683 | -6.889579773 | -5.934420205 |
| 268 | UAV_1 | 6 | `[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]` | 0.07108311633 | -14.40565491 | -14.33457179 | 0.9956355232 | -6.961399078 | -5.965763555 |
| 268 | UAV_1 | 7 | `[0, 0, 0, 0, 1, 0, 0, 0, 1, 1]` | -1.100011881 | -13.67900848 | -14.77902037 | -0.1754594744 | -6.234752655 | -6.410212129 |
| 268 | UAV_1 | 8 | `[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]` | -1.059535926 | -13.73383331 | -14.79336924 | -0.1349835196 | -6.289577484 | -6.424561004 |
| 268 | UAV_1 | 9 | `[0, 1, 0, 0, 1, 0, 0, 0, 1, 1]` | -1.100011881 | -13.7400856 | -14.84009748 | -0.1754594744 | -6.295829773 | -6.471289247 |
| 268 | UAV_1 | 10 | `[0, 1, 0, 0, 0, 0, 0, 0, 1, 1]` | -1.059535926 | -13.78592682 | -14.84546275 | -0.1349835196 | -6.34167099 | -6.47665451 |

Every legal Candidate configuration and its reward / next-V / Q decomposition is retained in [`traceable_candidate_q_integrity_result.json`](traceable_candidate_q_integrity_result.json). The Q and delta-Q checks used `atol=1e-8`, `rtol=1e-10`; all recorded values were finite.

The exact algebra establishes component consistency only. Candidate-Q classification also considers real-data warmup behavior and counterfactual next-V support.

## UAV_0 Transition-149 Reward Root-Cause Gate

- Baseline HEAD: `4ebc2a94e219242f8023c9d35e3c36a960e935fb`; transition/origin: `149/149`; head: `UAV_0`.
- Reproduction: **PASS**; KEEP `-374.295482363`; Exact-Q best `5.46970638263`; reward-best `5.6763382087` (same as Exact-Q best: NO).
- Branch input isolation: **PASS**. Task set, vehicle positions, local cache, previous cache/UAV states, other cache rows/actions, UAV actions, and final UAV positions are separately recorded in the JSON artifact.

### Reward identity and first-level decomposition

| Branch | Task utility | Cache cost | Movement energy | Movement cost | System utility / reward |
|---|---:|---:|---:|---:|---:|
| KEEP | -373.909651 | 0.3820913 | 225 | 0.00374 | -374.295482 |
| EXACT_Q_BEST | 6.2158122 | 0.74236582 | 225 | 0.00374 | 5.46970638 |
| REWARD_BEST | 6.2158122 | 0.535733994 | 225 | 0.00374 | 5.67633821 |

Exact-Q best minus KEEP: Δ task utility `380.125463265`, −Δ cache cost `-0.360274519616`, −Δ movement cost `-0`, Δ system utility `379.765188746`. Maximum system identity error `1.91e-14`; delta identity error `0`; largest additive change: **ACTUAL_TASK_UTILITY**.
- Per-task utility sum checks: KEEP error `0`, Exact-Q best error `8.88e-16`. Task count `15`; matching changed `9`, unchanged `6`.

### Matching changes and largest task contributions

- KEEP assignments: `{"45": 2, "CLOUD": 2, "RSU_1": 8, "UAV_0": 2, "UAV_1": 1}`; Exact-Q best: `{"45": 2, "CLOUD": 2, "RSU_1": 1, "UAV_0": 7, "UAV_1": 3}`.
- Cloud task counts (production `task_allocation.CLOUD`): KEEP `2`, Exact-Q best `2`.
- KEEP → best node transitions: `{"45 -> 45": 2, "CLOUD -> CLOUD": 2, "RSU_1 -> RSU_1": 1, "RSU_1 -> UAV_0": 7, "UAV_0 -> UAV_1": 2, "UAV_1 -> UAV_1": 1}`.

| Task | Vehicle | Service | KEEP node | Best node | Δ delay (s) | Δ energy (J) | Δ utility |
|---|---|---:|---|---|---:|---:|---:|
| 53 | 53 | 8 | RSU_1 | UAV_0 | -1520.85 | -302.604 | 69.3111 |
| 33 | 33 | 8 | RSU_1 | UAV_0 | -1495.37 | -297.553 | 68.1522 |
| 10 | 10 | 8 | RSU_1 | UAV_0 | -1470.27 | -292.55 | 67.007 |
| 34 | 34 | 8 | RSU_1 | UAV_0 | -1301.1 | -258.899 | 59.2986 |
| 22 | 22 | 8 | RSU_1 | UAV_0 | -1261.23 | -250.923 | 57.4761 |
| 6 | 6 | 8 | RSU_1 | UAV_0 | -606.094 | -120.164 | 27.5659 |
| 9 | 9 | 8 | RSU_1 | UAV_0 | -536.303 | -106.237 | 24.3799 |
| 37 | 37 | 8 | RSU_1 | RSU_1 | -182.25 | -30.9319 | 7.6097 |
| 3 | 3 | 4 | UAV_0 | UAV_1 | 8.49158 | 0.999519 | -0.296876 |
| 4 | 4 | 9 | UAV_1 | UAV_1 | 7.61447 | 0.345364 | -0.194265 |
| 20 | 20 | 9 | UAV_0 | UAV_1 | 5.39991 | 0.598276 | -0.183912 |
| 43 | 43 | 1 | 45 | 45 | 0 | 0 | 0 |
| 48 | 48 | 0 | CLOUD | CLOUD | 0 | 0 | 0 |
| 17 | 17 | 0 | CLOUD | CLOUD | 0 | 0 | 0 |
| 41 | 41 | 1 | 45 | 45 | 0 | 0 | 0 |

Absolute task utility delta shares: Top 1 `0.181692`, Top 5 `0.842112`, Top 10 `0.999518`.

### Service availability, Matching, and KKT

- UAV_0 KEEP cache row: `[0, 0, 0, 0, 1, 1, 0, 0, 0, 1]`; Exact-Q best: `[0, 0, 0, 1, 1, 0, 0, 0, 1, 0]`; added service IDs: `[3, 8]`; removed service IDs: `[5, 9]`.
- Task `10` service `8`: edge nodes KEEP `['RSU_1']` → best `['RSU_1', 'UAV_0']`; all feasible targets KEEP `['RSU_1']` → best `['RSU_1', 'UAV_0']`.
- Task `20` service `9`: edge nodes KEEP `['RSU_1', 'UAV_0', 'UAV_1']` → best `['RSU_1', 'UAV_1']`; all feasible targets KEEP `['RSU_1', 'UAV_0', 'UAV_1']` → best `['RSU_1', 'UAV_1']`.
- Task `22` service `8`: edge nodes KEEP `['RSU_1']` → best `['RSU_1', 'UAV_0']`; all feasible targets KEEP `['RSU_1']` → best `['RSU_1', 'UAV_0']`.
- Task `3` service `4`: edge nodes KEEP `['35', 'UAV_0', 'UAV_1']` → best `['35', 'UAV_0', 'UAV_1']`; all feasible targets KEEP `['35', 'LOCAL:3', 'UAV_0', 'UAV_1']` → best `['35', 'LOCAL:3', 'UAV_0', 'UAV_1']`.
- Task `33` service `8`: edge nodes KEEP `['RSU_1']` → best `['RSU_1', 'UAV_0']`; all feasible targets KEEP `['RSU_1']` → best `['RSU_1', 'UAV_0']`.
- Task `34` service `8`: edge nodes KEEP `['RSU_1']` → best `['RSU_1', 'UAV_0']`; all feasible targets KEEP `['RSU_1']` → best `['RSU_1', 'UAV_0']`.
- Task `53` service `8`: edge nodes KEEP `['RSU_1']` → best `['RSU_1', 'UAV_0']`; all feasible targets KEEP `['RSU_1']` → best `['RSU_1', 'UAV_0']`.
- Task `6` service `8`: edge nodes KEEP `['RSU_1']` → best `['RSU_1', 'UAV_0']`; all feasible targets KEEP `['RSU_1']` → best `['RSU_1', 'UAV_0']`.
- Task `9` service `8`: edge nodes KEEP `['RSU_1']` → best `['RSU_1', 'UAV_0']`; all feasible targets KEEP `['RSU_1']` → best `['RSU_1', 'UAV_0']`.
- KEEP Matching: stop `converged_by_no_improvement`, stable `True`, max remaining Δ `-0.027480662250638943`, updates `3` (transfer `3`, exchange `0`), candidates `31`, full evaluator calls `31`.
- KEEP KKT: calls `36`, solves `36`, converged `36`, max-iteration hits `0`, finite `36`; residual max resource/stationarity/complementarity `8.841816168114747e-13` / `1.1368683772161603e-13` / `3.460657949505981e-10`.
- Exact-Q best Matching: stop `converged_by_no_improvement`, stable `True`, max remaining Δ `-0.08547983592386288`, updates `10` (transfer `10`, exchange `0`), candidates `162`, full evaluator calls `162`.
- Exact-Q best KKT: calls `338`, solves `338`, converged `338`, max-iteration hits `0`, finite `338`; residual max resource/stationarity/complementarity `9.903189379656396e-13` / `1.1368683772161603e-13` / `3.643993971031547e-10`.

### Penalty and root-cause decision

- Explicit penalty found: **NO**. Production task utility is `ω_T(1 − delay/T_ref) + ω_E(1 − vehicle_energy/E_ref)` from `src/common/system_model.py`; production system utility is `actual_task_utility − cache_cost − movement_cost` from `src/decision/slot_environment.py`. Infeasible branch execution raises before a reward is returned; Cloud is a feasible target/fallback, not a separate utility penalty.
- Classification: **REWARD_OUTLIER_VALID_TASK_EXECUTION**. Reward outlier resolved: `YES`.
- Candidate-Q integrity classification remains `CANDIDATE_Q_INTEGRITY_INCONCLUSIVE`; counterfactual critic reliability remains `INCONCLUSIVE`; next stage is `COUNTERFACTUAL_CRITIC_RELIABILITY_GATE`. No training or next-stage work was started.
- JSON artifact: [`uav0_transition149_reward_root_cause.json`](uav0_transition149_reward_root_cause.json).

## Counterfactual Critic Reliability Gate

- Current HEAD: `52c75b0ec34dd06dda9fec407472fccf1746957e`; source Candidate-Q HEAD: `4ebc2a94e219242f8023c9d35e3c36a960e935fb`.
- Critic source: **EXISTING_DIAGNOSTIC_CHECKPOINT**; reconstruction gate: **PASS**; parameter digest `378d8fe6dc261574f598905dbb5446a4d84bbeb43dd8386ad8bf50b6ce5072d6`; optimizer steps `40`.
- Diagnostic TRAIN realization: **GENERATE_COMPENSATORY_ONCE**, generated `YES`; path `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\diagnostics\counterfactual_critic_reliability\diagnostic_train_realization`.
- Transition `149` same-Critic reproduction: **PASS**; frozen gate: **YES**.

- Previous attempt history: its 12 continuation calculations completed, but aggregate JSON persistence failed on an internal NumPy array; this authorized recovery reuses the same Critic and reruns only those same 12 records.
| Role | Head | Candidate config | ΔV predicted | ΔG MC r0 | ΔG MC r1 | ΔG MC mean | Signs r0/r1 | Stability |
|---|---|---|---:|---:|---:|---:|---|---|
| SV | 18 | `[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]` | -9.78585815 | 396.243792 | 0 | 198.121896 | NO/NO | UNSTABLE |
| RSU | RSU_1 | `[1, 0, 1, 0, 0, 0, 0, 1, 0, 0]` | -34.2609692 | 1082.77147 | -832.069672 | 125.3509 | NO/YES | UNSTABLE |
| UAV | UAV_1 | `[0, 0, 0, 1, 0, 0, 1, 0, 0, 0]` | 13.1373367 | -104.857464 | -380.077014 | -242.467239 | NO/NO | STABLE |

- Reliability classification: **CONCERN**; Candidate-Q classification: **CANDIDATE_Q_INTEGRITY_CONCERN**; next stage: `COUNTERFACTUAL_CRITIC_REPAIR_DESIGN`.
- Diagnostic JSON: [`counterfactual_critic_reliability.json`](counterfactual_critic_reliability.json). Formal PPO update: **NO**; Epoch 1: **NO**; readiness: **NO**.
- Temporary directory cleanup: `REMOVED`.

### Authorized persistence-recovery rerun

The first 12 continuation calculations completed, but report serialization failed. This one-time compensatory run used the same 12 role/replicate/branch identities and deterministic seeds, without expanding the sample or updating the Critic.

| Role | Head | ΔV pred | ΔG MC r0 | ΔG MC r1 | ΔG MC mean | Sign r0 | Sign r1 | Stability |
|---|---|---:|---:|---:|---:|---|---|---|
| SV | 18 | -9.785858154296875 | 396.2437920756638 | 0.0 | 198.1218960378319 | NO | NO | UNSTABLE |
| RSU | RSU_1 | -34.260969161987305 | 1082.7714718803763 | -832.0696722827852 | 125.35089979879558 | NO | YES | UNSTABLE |
| UAV | UAV_1 | 13.137336730957031 | -104.8574641123414 | -380.0770139247179 | -242.46723901852965 | NO | NO | STABLE |

- Continuation JSONL evidence: `12/12`; persistence gate: **PASS**; result persistence: **PASS**.
- Classification is limited to the three targeted high-Critic-leverage counterfactuals: **CONCERN**. It does not establish global Critic accuracy or calibration.

## Counterfactual Critic Repair Design

Date: 2026-09-28

Static repair-design gate result: MIXED_TARGET_AND_DATA_CONCERN; primary direction: CRITIC_TARGET_ALIGNMENT_FIRST. The active Critic is trained against GAE/lambda returns (gamma=1.0, lambda=0.95) through an independent PopArt-normalized Adam optimizer. Existing actual-state evidence shows post-warmup MC MAE 537.07 and EV 0.006, while MAE to the training GAE target is 93.96; GAE-to-MC MAE is 453.71. One 300-slot TRAIN realization and 40 Critic steps are insufficient to establish data or step-count sufficiency. Static audit found no deterministic implementation bug.

The existing reliability gate remains CONCERN, Candidate-Q remains CANDIDATE_Q_INTEGRITY_CONCERN, and SURROGATE_REPAIR_BLOCKED=YES. A stable UAV_1 direction-error witness remains. Hybrid K=16 exact subset cost is about 20.67 candidates per head in the saved nine-head development gate; N-step exact evaluation multiplies per-candidate environment executions by N. Full terminal-MC Candidate teacher remains diagnostic only.

The next direction is a controlled Critic-target alignment experiment with frozen Actor, matched Critic initialization, identical TRAIN-only data and 40-step budget, comparing current GAE target with full-episode MC target; evaluate on an independent TRAIN diagnostic realization only. This experiment was not run. No formal PPO update, Epoch 1, new realization, or production source change occurred. Formal training readiness remains NO.

Detailed evidence, equations, cost estimates, and experiment controls: [counterfactual_critic_repair_design.md](counterfactual_critic_repair_design.md) and [counterfactual_critic_repair_design.json](counterfactual_critic_repair_design.json).

## Critic Target Alignment Minimal Experiment

- Result: **INCONCLUSIVE**
- Next stage: `CRITIC_TARGET_ALIGNMENT_EXPERIMENT_FOLLOWUP`
- Current HEAD: `52c75b0ec34dd06dda9fec407472fccf1746957e`

## Completed before interruption

- Reused TRAIN realization seed `20001` from `C:\Users\tangning\Desktop\code\adaptive_cache_mec\environment_data\proposed_joint_training_preexisting_51_backup_20260921\episode_20001`; its split, generation fingerprint, named seed streams, required artifacts, and 300-slot runtime protocol passed preflight. Its `episode_manifest.json` was read only as optional provenance.
- Fresh Critic initialization equivalence passed; initial Critic digest: `ffa1c1915716f996732c7470c348fcb08854f17ffe53c2d8ef1536f23492705f`. Actor stayed frozen and the Actor optimizer step count remained 0.
- Exactly one shared 300-transition TRAIN rollout was collected. It passed the terminal structure and rollout-time raw value checks.

## Why the experiment is inconclusive

After collecting and temporarily saving the rollout arrays, the runner raised `AttributeError` while reading `PPOV2Hyperparameters.minibatch_size`, before computing either target, generating the minibatch sequence, or taking any Critic step. The scratch directory was removed during exception cleanup, so the rollout arrays are no longer available. The runner guard has been corrected to read the runtime context minibatch size. No second rollout was collected.

Consequently, neither Critic arm was trained; both step counts are 0, no diagnostic realization or diagnostic rollout was started, no actual-state MC metrics or UAV_1 branch values were computed, and the requested 40-step checkpoints were not created.

## Current gates

- Initialization equivalence: **PASS**.
- Minibatch order equivalence: **NOT RUN**.
- ARM_A / ARM_B Critic steps: `0 / 0`.
- Training / diagnostic rollout collection counts: `1 / 0`.
- Formal PPO update: **NO**; Epoch 1: **NO**; production Critic target changed: **NO**.
- Counterfactual Critic Reliability remains **CONCERN**; Candidate-Q remains **CANDIDATE_Q_INTEGRITY_CONCERN**; surrogate repair remains blocked.
- Formal training readiness remains **NO**.

### Authorized Critic Target Alignment Follow-up

- Result: **PARTIAL**
- Next stage: `CRITIC_DATA_COVERAGE_MINIMAL_EXPERIMENT`
- Current HEAD: `52c75b0ec34dd06dda9fec407472fccf1746957e`
- Training realization: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\environment_data\proposed_joint_training_preexisting_51_backup_20260921\episode_20001` (seed 20001, LOAD)
- Diagnostic realization: seed 12184115623283823756 (GENERATE)

## Fairness and execution gates

- Initialization equivalence: **PASS**; initial Critic digest `ffa1c1915716f996732c7470c348fcb08854f17ffe53c2d8ef1536f23492705f`.
- Minibatch order equivalence: **PASS**; SHA256 `61ddab16fae6fde535a82fe987c70c61fb3cb8748200b958a2c8ba67cf8c7436`.
- Critic steps: ARM_A 40, ARM_B 40; Actor optimizer steps 0; Actor delta 0.
- Train and diagnostic rollout collection counts: 1 and 1.
- Formal PPO update: NO; Epoch 1: NO; production Critic target changed: NO.

## Independent TRAIN actual-state evaluation

| Arm | Target | MC MAE | MC RMSE | MC EV | Pearson | Spearman |
|---|---|---:|---:|---:|---:|---:|
| ARM_A | GAE_LAMBDA_RETURN | 531.136537 | 680.846709 | 0.037065 | 0.4774469992360514 | 0.37266325181390897 |
| ARM_B | FULL_EPISODE_MC_RETURN | 480.699960 | 527.058296 | 0.000469 | 0.08290692477484993 | 0.26819185771265025 |

- MC MAE improvement (A − B): 50.436578.
- MC RMSE improvement (A − B): 153.788413.
- MC EV improvement (B − A): -0.036596.

## Fixed UAV_1 transition-149 witness

- Config: `[0, 0, 0, 1, 0, 0, 1, 0, 0, 0]`; transition fingerprint: `2805d3e60f0b40457ab53652995bbbd58e4cb438bac06c3af5445284981d3408`; branch reconstruction: **PASS**.
- Existing paired MC signs: NEGATIVE / NEGATIVE (ΔG values -104.857464, -380.077014).
- ARM_A ΔV: 3.220560 (POSITIVE); ARM_B ΔV: -0.010498 (NEGATIVE).
- Absolute error to paired MC mean: ARM_A 245.687799; ARM_B 242.456741.
- Exactly two immediate branches were reconstructed (KEEP and the fixed Candidate); no terminal continuation was run.

## Persistent artifacts and gates

- ARM_A checkpoint: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\diagnostics\critic_target_alignment\gae_target_critic.pt`.
- ARM_B checkpoint: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\diagnostics\critic_target_alignment\mc_target_critic.pt`.
- Reliability remains `CONCERN`; Candidate-Q remains `CANDIDATE_Q_INTEGRITY_CONCERN`; surrogate repair remains blocked: `YES`.
- Formal training readiness: `NO`.
- Decision rationale: The target change improved either the actual-state metric set or the fixed UAV_1 sign, but not both required groups.

## Critic Data Coverage Minimal Experiment

- Result: **INCONCLUSIVE**
- Next stage: `CRITIC_DATA_COVERAGE_EXPERIMENT_FOLLOWUP`
- Current HEAD: `163df1e04e16d966fde6557e110d9a24809cf37c`
- TRAIN rollouts: 4 persisted 300-slot episodes; episode 20001 SHA reproduction: **PASS**.
- Coverage Critic: 40 steps; checkpoint reload: **PASS**.
- Fixed diagnostic realization action: LOAD; rollout SHA reproduction: **FAIL** (`0d9f992fbee131b23fa8d3f9227380795a82ad2a72628baf5e02370dd60ec778` vs expected `3fa12013dbd9d17df6851a98fc1462db599d46d5bc7c0c0db10128c987b39c20`).
- Per the reproduction gate, metric evaluation and transition-149 branch reconstruction were stopped. Recovery files were retained. Formal PPO update, Epoch 1, and readiness remain NO.
