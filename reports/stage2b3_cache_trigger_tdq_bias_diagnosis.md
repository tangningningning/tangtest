# Stage2B3 Cache Trigger TD-Q Bias Diagnosis

## Decision fields

```text
EPISODES_ANALYZED = 20100, 20105, 20107, 20109
TRANSITIONS_ANALYZED = 32
COUNTERFACTUAL_BRANCHES_EXECUTED = 288
ARTIFACT_ONLY_ANALYSIS = NO

CREDIT_UPDATE_BIAS_INCREASES_WITH_CACHE_CHANGE = INCONCLUSIVE
CACHE_TRIGGER_TDQ_BIAS = INCONCLUSIVE
COUNTERFACTUAL_REFERENCE_ASYMMETRY = SUPPORTED
STAGE2B_TRIGGER_REFERENCE_CHANGE_NEEDED = INCONCLUSIVE

TRAINING_EXECUTED = NO
OPTIMIZER_STEP_CALLS = 0
BACKWARD_CALLS = 0
COMMIT = NO
PUSH = NO
NO_RESIDUAL_PYTHON_PROCESS = YES
```

The result is a diagnosis of the existing TD-Q trigger-credit path, not an
implementation recommendation. No PPO update, checkpoint write, formal-run
write, or Stage2B algorithm change was performed.

## Scope and method

The persistent target run was read-only:

```text
runs/stage2b_cache_trigger_tdq_short_gate
checkpoint = latest_checkpoint.pt, completed epoch 109
trigger source = ONE_STEP_TD_Q_DIFFERENCE
candidate source = GLOBAL_GAE
Cache ratio = PER_CACHE_HEAD
```

The persistent `update_metrics.jsonl` and `episode_metrics.jsonl` contain only
aggregate TD-Q values; they do not contain the per-transition/per-head raw
`D_Q`, role, and reference records needed for this audit. Therefore this was a
limited replay rather than artifact-only analysis. The replay used only the
existing train-only realizations under
`environment_data/proposed_joint_training` and a frozen epoch-109 policy:

| realization | historical window label | selected origins |
|---|---:|---|
| 20100 | 100–102 | 0, 15, 51, 63, 99, 235, 239, 293 |
| 20105 | 103–106 | 4, 41, 85, 130, 233, 235, 283, 289 |
| 20107 | 107–109 | 0, 7, 10, 59, 76, 223, 238, 277 |
| 20109 | 107–109 | 8, 92, 119, 151, 192, 205, 281, 289 |

Each selected transition was evaluated for all 9 Cache heads. The existing
`compute_one_step_tdq_cache_trigger_credit` path was reused unchanged:

- actual `UPDATE` was compared with `KEEP`;
- actual `KEEP` was compared with the highest-policy-probability legal
  `UPDATE` candidate;
- the current critic was used only for detached forward evaluation;
- `D_Q = Q_actual - Q_counterfactual`;
- normalized credit uses the current std-only convention, with the diagnostic
  sample scale `15.22442001` and no centering.

The replay used stochastic policy sampling with a fixed seed per realization.
Thus the action-role split is a frozen-policy replay sample, not a recovery of
the historical per-transition actions from epochs 100–109. The window labels
are retained for comparison with the existing run's CacheChange, but should
not be interpreted as policy snapshots at each historical epoch.

## Overall UPDATE versus KEEP credit

`D_Q` is signed according to the actual branch minus its reference branch. For
an actual UPDATE row this is an UPDATE-to-KEEP gap; for an actual KEEP row it
is a KEEP-to-UPDATE gap.

| actual role | count | positive | negative | positive rate | mean | median | P10 | P90 | std | abs P90 | abs max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UPDATE | 224 | 112 | 112 | 0.500000 | -0.086059 | 0.002908 | -10.419279 | 9.013941 | 8.270302 | 13.145175 | 35.709522 |
| KEEP | 64 | 22 | 42 | 0.343750 | -8.016056 | -3.646130 | -20.821951 | 8.648971 | 27.472172 | 21.080815 | 150.533737 |

The corresponding std-normalized credit is:

| actual role | mean | median | positive rate | P10 | P90 | abs P90 | abs max |
|---|---:|---:|---:|---:|---:|---:|---:|
| UPDATE | -0.005653 | 0.000191 | 0.500000 | -0.684379 | 0.592071 | 0.863427 | 2.345542 |
| KEEP | -0.526526 | -0.239492 | 0.343750 | -1.367668 | 0.568099 | 1.384671 | 9.887650 |

Required aggregate fields:

```text
UPDATE_DQ_COUNT = 224
UPDATE_DQ_POSITIVE_COUNT = 112
UPDATE_DQ_NEGATIVE_COUNT = 112
UPDATE_DQ_POSITIVE_RATE = 0.500000
UPDATE_DQ_MEAN = -0.086059
UPDATE_DQ_MEDIAN = 0.002908
UPDATE_DQ_P10 = -10.419279
UPDATE_DQ_P90 = 9.013941

KEEP_DQ_COUNT = 64
KEEP_DQ_POSITIVE_COUNT = 22
KEEP_DQ_NEGATIVE_COUNT = 42
KEEP_DQ_POSITIVE_RATE = 0.343750
KEEP_DQ_MEAN = -8.016056
KEEP_DQ_MEDIAN = -3.646130
KEEP_DQ_P10 = -20.821951
KEEP_DQ_P90 = 8.648971

UPDATE_NORMALIZED_CREDIT_MEAN = -0.005653
UPDATE_NORMALIZED_CREDIT_MEDIAN = 0.000191
UPDATE_NORMALIZED_CREDIT_POSITIVE_RATE = 0.500000
KEEP_NORMALIZED_CREDIT_MEAN = -0.526526
KEEP_NORMALIZED_CREDIT_MEDIAN = -0.239492
KEEP_NORMALIZED_CREDIT_POSITIVE_RATE = 0.343750

TRIGGER_POSITIVE_RATE_GAP_UPDATE_MINUS_KEEP = 0.156250
TRIGGER_MEAN_GAP_UPDATE_MINUS_KEEP = 7.929997
TRIGGER_MEDIAN_GAP_UPDATE_MINUS_KEEP = 3.649038
NORMALIZED_MEAN_GAP_UPDATE_MINUS_KEEP = 0.520874
```

The overall UPDATE-positive rate is higher by 15.625 percentage points, but
the sample is not consistent enough to call this a stable UPDATE bias. In
particular, the early window has the opposite sign for both the positive-rate
and mean gap.

## Counterfactual gap asymmetry and KEEP reference

```text
UPDATE_TO_KEEP_Q_GAP_MEAN = -0.086059
UPDATE_TO_KEEP_Q_GAP_MEDIAN = 0.002908
UPDATE_TO_KEEP_Q_GAP_ABS_P90 = 13.145175
UPDATE_TO_KEEP_Q_GAP_ABS_MAX = 35.709522

KEEP_TO_UPDATE_Q_GAP_MEAN = -8.016056
KEEP_TO_UPDATE_Q_GAP_MEDIAN = -3.646130
KEEP_TO_UPDATE_Q_GAP_ABS_P90 = 21.080815
KEEP_TO_UPDATE_Q_GAP_ABS_MAX = 150.533737
```

The existing construction is role-asymmetric, and this replay also shows a
materially heavier KEEP-to-UPDATE tail: absolute P90 is 21.08 versus 13.15,
and absolute maximum is 150.53 versus 35.71. This supports
`COUNTERFACTUAL_REFERENCE_ASYMMETRY = SUPPORTED` as an observed diagnostic
property. It does not by itself prove that the reference construction is the
cause of the training behavior.

For actual KEEP transitions, the current helper selected the legal UPDATE
candidate with the highest conditional candidate probability. The primary
`KEEP_REFERENCE_UPDATE_PROB` below is therefore the conditional probability
of that selected candidate; operation and joint probabilities are shown to
avoid conflating the binary trigger probability with the candidate probability.

```text
KEEP_REFERENCE_UPDATE_PROB_P10 = 0.066235
KEEP_REFERENCE_UPDATE_PROB_P90 = 0.188649
KEEP_REFERENCE_UPDATE_PROB_MEAN = 0.125442

KEEP_REFERENCE_OPERATION_UPDATE_PROB_P10 = 0.614385
KEEP_REFERENCE_OPERATION_UPDATE_PROB_P90 = 0.876318
KEEP_REFERENCE_OPERATION_UPDATE_PROB_MEAN = 0.765408

KEEP_REFERENCE_JOINT_UPDATE_PROB_P10 = 0.050932
KEEP_REFERENCE_JOINT_UPDATE_PROB_P90 = 0.144076
KEEP_REFERENCE_JOINT_UPDATE_PROB_MEAN = 0.094635

KEEP_REFERENCE_LEGAL_CANDIDATE_COUNT_P10 = 8
KEEP_REFERENCE_LEGAL_CANDIDATE_COUNT_P90 = 29
KEEP_REFERENCE_LEGAL_CANDIDATE_COUNT_MEAN = 16.609375
```

The selected KEEP references were not generally near zero probability, so the
observed tail asymmetry should not be summarized as simply “the reference was
invalid.” The audit does not establish whether a different reference would
improve training.

## Per-head results

Every head has 32 replay samples. `actual_update_rate` is the fraction of
those samples whose actual operation at that head was UPDATE.

| head | samples | actual UPDATE rate | DQ mean | DQ median | DQ positive rate | normalized DQ mean |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 32 | 0.750000 | -1.264932 | 0.067290 | 0.500000 | -0.083086 |
| 1 | 32 | 0.906250 | -0.152699 | -0.243448 | 0.500000 | -0.010030 |
| 2 | 32 | 0.656250 | -2.125642 | -1.330720 | 0.406250 | -0.139621 |
| 3 | 32 | 0.843750 | 1.925139 | 1.087943 | 0.531250 | 0.126451 |
| 4 | 32 | 0.812500 | -1.928423 | -2.304444 | 0.437500 | -0.126666 |
| 5 | 32 | 0.656250 | -5.080289 | -0.624970 | 0.437500 | -0.333693 |
| 6 | 32 | 0.718750 | -1.989309 | -2.396369 | 0.437500 | -0.130666 |
| 7 | 32 | 0.781250 | -4.936758 | -1.591607 | 0.500000 | -0.324266 |
| 8 | 32 | 0.875000 | -1.081611 | -0.956668 | 0.437500 | -0.071045 |

The head-level means are heterogeneous, including head 3 positive and heads 5
and 7 substantially negative. This is evidence that a single global credit
would be coarse at head level, but this task does not implement per-head
credit and the sample is too small to make a training-policy change.

## CacheChange windows

The CacheChange values below are read from the existing target-run
`episode_metrics.jsonl`; the DQ values are from the frozen epoch-109 replay
mapped to the corresponding realization windows.

| window | replay seeds | branches | UPDATE positive rate | KEEP positive rate | DQ mean gap UPDATE−KEEP | existing-run CacheChange mean |
|---|---|---:|---:|---:|---:|---:|
| 100–102 | 20100 | 72 | 0.473684 | 0.533333 | -3.453883 | 0.599630 |
| 103–106 | 20105 | 72 | 0.571429 | 0.375000 | 22.697172 | 0.705556 |
| 107–109 | 20107, 20109 | 144 | 0.477477 | 0.242424 | 5.906866 | 0.784938 |

The middle-window gap is the largest, then falls in the late window. A
nominal three-point rank ordering is not enough to establish a monotonic
increase, especially because all DQ values come from one frozen epoch-109
policy replay. Therefore:

```text
CREDIT_UPDATE_BIAS_INCREASES_WITH_CACHE_CHANGE = INCONCLUSIVE
```

## Interpretation and safe disposition

`CACHE_TRIGGER_TDQ_BIAS = INCONCLUSIVE`: there is a descriptive overall
UPDATE-versus-KEEP difference, and the middle/late windows lean toward higher
UPDATE-positive credit, but the early window reverses the direction and the
observed overall gap is not a stable cross-window result. The four
realizations and 32 transitions are diagnostic evidence, not a sufficient
basis for changing the trigger reference or implementing per-head credit.

`COUNTERFACTUAL_REFERENCE_ASYMMETRY = SUPPORTED` refers to the observed
role-dependent tail asymmetry and the already role-dependent reference rule.
It should not be read as proof that the reference rule is incorrect.

Because trigger bias is inconclusive and the replay is frozen-policy/small-
sample, `STAGE2B_TRIGGER_REFERENCE_CHANGE_NEEDED = INCONCLUSIVE`. No reference
change is made in this task. Candidate-level credit remains outside scope.

## Runtime and safety record

```text
TOTAL_RUNTIME_SECONDS = 535.342108
COUNTERFACTUAL_RUNTIME_SECONDS = 128.310713
SECONDS_PER_HEAD_COUNTERFACTUAL = 0.445523
PROJECTED_300_SLOT_9_HEAD_RUNTIME_SECONDS = 1202.912931
```

The projected figure is only a linear cost estimate for 2,700 one-step Cache
head counterfactuals, not a training benchmark. The replay used 288 existing
branch executions and did not create a new run or write any output artifact.
All temporary diagnostic files and generated Python cache files were removed
after the replay.
