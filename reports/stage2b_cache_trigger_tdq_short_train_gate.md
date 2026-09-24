# Stage2B Cache trigger TD-Q short-train gate

## Scope and provenance

This is the requested ten-episode, train-only mechanism validation. It is not
a formal run, and no episode beyond 109 was executed.

```text
TRAINING_EPISODES = 10
TRAIN_ONLY = YES
FORMAL_RUN_MODIFIED = NO
VALIDATION_RUN = runs/stage2b_cache_trigger_tdq_short_gate
SOURCE_CHECKPOINT = runs/proposed_actor1e4_uavfix_300_epoch/latest_checkpoint.pt
SOURCE_COMPLETED_EPOCH = 99
SOURCE_SHA_BEFORE = 6081BC1DA279C9081BBC7EC466CA25E9B97902F65FCF1C6C8C8C9CCACED4ABBB
SOURCE_SHA_AFTER  = 6081BC1DA279C9081BBC7EC466CA25E9B97902F65FCF1C6C8C8C9CCACED4ABBB
SEEDS_USED = 20100-20109
EPOCHS_USED = 100-109
```

The target checkpoint was a copy of the epoch-99 source checkpoint. The target
configuration differs from the existing Stage1B validation configuration only
in `cache_trigger_advantage_source`; it uses the current TD-Q implementation
and retains `PER_CACHE_HEAD` and the Stage1B UAV settings. The existing
Stage1B baseline was read from its logs and was not rerun.

The first attempt completed epoch 100 and then hit an OpenBLAS allocation
failure in the four-worker UAV Exact-Q process pool during epoch 101. Epoch
101 subsequently completed from the epoch-100 target checkpoint, but the
target-only slot-metrics continuity check stopped before the next episode. The
target slot prefix (epochs 1-99) and the already written epoch-101 rewards were
repaired read-only from existing artifacts; no rollout, optimizer step, or
episode was repeated. Epochs 102-109 then completed with one Exact-Q worker to
avoid the resource failure. This worker-count change affects execution
parallelism only, not the algorithm or its numerical definitions.

## Active algorithm and TD-Q path

```text
CACHE_TRIGGER_CREDIT = ONE_STEP_TD_Q_DIFFERENCE
CACHE_CANDIDATE_CREDIT = GLOBAL_GAE
CACHE_RATIO_GRANULARITY = PER_CACHE_HEAD
BOOTSTRAP_SIGN = PLUS
TDQ_CENTERING_USED = NO
TDQ_SCALE_NORMALIZATION_USED = YES
TDQ_SCALE = std(D_Q_valid) + 1e-8
```

The run executed `Q_actual = R_actual + gamma * V(s'_actual)`,
`Q_cf = R_cf + gamma * V(s'_cf)`, and `D_Q = Q_actual - Q_cf`. The target
checkpoint reports 109 completed episodes and the target JSONL files contain
exactly one row for each epoch 100-109.

## Per-episode results

The episode-pool seed is the realization seed; the JSONL `seed` field remains
the fixed experiment seed 4401.

| epoch / train seed | return | CacheChange | UAVMove | critic EV | TD-Q raw std | norm abs P90 | norm abs max | per-head clip | joint clip | TD-Q sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 / 20100 | -1342.55 | 0.5663 | 0.3667 | -0.1432 | 18.9620 | 1.2179 | 12.7982 | 0.005833 | 0.241667 | 79.56 |
| 101 / 20101 | -690.57 | 0.6181 | 0.4583 | -0.1783 | 18.8951 | 1.2518 | 14.5653 | 0.014074 | 0.236667 | 52.27 |
| 102 / 20102 | 1097.13 | 0.6144 | 0.4150 | 0.3979 | 11.1898 | 1.3184 | 13.6645 | 0.000000 | 0.129167 | 82.14 |
| 103 / 20103 | 844.89 | 0.6552 | 0.4900 | 0.4936 | 17.6342 | 1.2161 | 13.6076 | 0.004074 | 0.185000 | 91.02 |
| 104 / 20104 | -761.40 | 0.6926 | 0.4133 | 0.6348 | 15.4936 | 1.4049 | 11.1526 | 0.011019 | 0.225833 | 75.51 |
| 105 / 20105 | -127.13 | 0.7196 | 0.4517 | 0.3707 | 19.1455 | 1.4351 | 9.6774 | 0.007037 | 0.185000 | 74.79 |
| 106 / 20106 | 1539.09 | 0.7548 | 0.4767 | 0.6391 | 9.8461 | 1.4361 | 12.5287 | 0.013519 | 0.200833 | 74.36 |
| 107 / 20107 | 1147.99 | 0.7752 | 0.4783 | 0.3389 | 9.7905 | 1.2320 | 14.4701 | 0.005648 | 0.119167 | 52.13 |
| 108 / 20108 | 1025.26 | 0.7837 | 0.5383 | 0.5496 | 9.3967 | 1.5203 | 6.5752 | 0.000278 | 0.079167 | 114.06 |
| 109 / 20109 | 225.40 | 0.7959 | 0.5250 | 0.5973 | 11.1791 | 1.2038 | 12.7547 | 0.000000 | 0.048333 | 54.77 |

## TD-Q numerical stability and cost

```text
TDQ_FINITE_RATE = 1.000000
ONE_STEP_TDQ_BRANCHES_EXECUTED = 27000
CACHE_TRIGGER_TDQ_RAW_STD_MEAN = 14.153252
CACHE_TRIGGER_TDQ_RAW_ABS_MAX = 275.211903
CACHE_TRIGGER_TDQ_NORMALIZED_ABS_P90_MEAN = 1.323626
CACHE_TRIGGER_TDQ_NORMALIZED_ABS_MAX = 14.565277
```

All ten TD-Q scales were finite and strictly above `1e-8`; all existing
finite checks were true. The normalized tail is visible but not an unbounded
numerical excursion: the P90 mean is 1.324, the largest normalized absolute
value is 14.565, and no temporary clamp was added. Cache scorer delta norm was
`0.043069` mean / `0.057771` max, and Cache encoder gradient norm was
`0.046524` mean / `0.054517` max. No non-finite gradient or parameter jump was
observed.

Measured runtime:

```text
EPISODE_RUNTIME_MEAN = 329.449 s
EPISODE_RUNTIME_MEDIAN = 331.095 s
EPISODE_RUNTIME_MAX = 454.790 s
TDQ_RUNTIME_MEAN_PER_EPISODE = 75.062 s
TDQ_RUNTIME_MEDIAN_PER_EPISODE = 75.151 s
TDQ_RUNTIME_MAX_PER_EPISODE = 114.064 s
TDQ_RUNTIME_FRACTION_OF_EPISODE = 0.226630 mean / 0.300970 max
SECONDS_PER_CACHE_HEAD = 0.027801 mean
STATIC_PROJECTED_300_SLOT_RUNTIME_SECONDS = 59.45 (prior read-only benchmark)
ONE_STEP_TDQ_RUNTIME_FEASIBILITY = BORDERLINE
```

The actual cost is higher than the prior static projection, but it remains a
bounded fraction of the episode runtime. Full long-horizon training would be
substantially slower; this gate does not claim that the cost is optimal.

## Cache ratio and behavior

```text
PER_HEAD_CLIP_FRACTION_MEAN = 0.006148
PER_HEAD_CLIP_FRACTION_MAX = 0.014074
WHOLE_JOINT_CLIP_FRACTION_MEAN = 0.165083
WHOLE_JOINT_CLIP_FRACTION_MAX = 0.241667
CACHE_PER_HEAD_OLD_NEW_KL_MEAN = 0.001810 mean
CACHE_PER_HEAD_OLD_NEW_KL_P90 = 0.008169 aggregate P90
CACHE_PER_HEAD_KL_MAX = 0.028793 aggregate maximum
CACHE_PER_HEAD_RATIO_MEAN = 0.999827 mean
CACHE_PER_HEAD_RATIO_P90 = 1.048972 mean of episode P90 values
CACHE_PER_HEAD_RATIO_MAX = 1.306960 aggregate maximum
```

The per-head clip fraction remains far below the legacy whole-joint diagnostic
(`0.0061` versus `0.1651` mean), so the Stage2A ratio mechanism did not regress.
The per-head clip fraction is somewhat higher than the Stage1B baseline's
short-gate mean, but its absolute level, KL, and ratio values remain small.

Cache behavior changed materially and should not be interpreted as an
automatic improvement:

```text
CACHE_CHANGE_MEAN = 0.697593
CACHE_CHANGE_MIN = 0.566296
CACHE_CHANGE_MAX = 0.795926
CACHE_CHANGE_LAST_6_MEAN = 0.753642
CACHE_CHANGE_FIRST_TO_LAST_DELTA = +0.229630
```

This is a behavioral difference of the TD-Q trigger credit, not evidence by
itself that the Cache policy improved.

## Stage1B UAV non-regression

The ten episodes had zero candidate-only steps after operation stop and zero
candidate-scorer/encoder step gaps:

```text
SCORER_ENCODER_NEW_DESYNC_COUNT = 0
UAV_CANDIDATE_SCORER_ENCODER_STEP_GAP = 0 for all 10 episodes
UAV_CANDIDATE_ONLY_STEPS_AFTER_OPERATION_STOP = 0 for all 10 episodes
CANDIDATE_KL_GATE_TRIGGER_COUNT = 0
CANDIDATE_KL_GATE_TRIGGER_SOURCE_COUNTS = NONE:10, MEAN:0, P90:0, BOTH:0
```

Aggregate candidate-policy drift was:

```text
UAV_CANDIDATE_POLICY_ENTROPY_MEAN = 2.671029
UAV_CANDIDATE_TEACHER_ENTROPY_MEAN = 1.783683
UAV_CANDIDATE_BEST_POLICY_PROB_MEAN = 0.099901
UAV_CANDIDATE_BEST_POLICY_RANK_MEAN = 5.145333
UAV_CANDIDATE_BEST_POLICY_TOP1_RATE_MEAN = 0.140000
UAV_CANDIDATE_BEST_POLICY_TOP3_RATE_MEAN = 0.351333
UAV_CANDIDATE_ARGMAX_FLIP_RATE_MEAN = 0.008583
UAV_CANDIDATE_OLD_NEW_KL_P90_MAX = 0.008925
UAV_CANDIDATE_OLD_NEW_KL_MAX_MAX = 0.035545
UAV_CANDIDATE_SCORER_PARAMETER_DELTA_MEAN = 0.226835
UAV_CANDIDATE_SCORER_PARAMETER_DELTA_MAX = 0.290598
```

The target run did not cross the UAV candidate gate at 107 or 108. This is not
a failure: the gate was retained unchanged and no candidate drift required
its backstop. The historical excursion comparison is:

| epoch | run | scorer delta | candidate KL P90 | candidate KL max | argmax flip | best prob | best rank |
|---:|---|---:|---:|---:|---:|---:|---:|
| 107 | historical Stage2A | 0.582268 | 0.117297 | 0.160561 | 0.095000 | 0.074159 | 5.857 |
| 107 | TD-Q target | 0.216813 | 0.004037 | 0.016944 | 0.000000 | 0.106689 | 5.582 |
| 108 | historical Stage2A | 0.571597 | 0.117045 | 0.291551 | 0.169167 | 0.115176 | 4.257 |
| 108 | TD-Q target | 0.290598 | 0.004054 | 0.018087 | 0.007917 | 0.102577 | 5.470 |

```text
EPOCH107_EXCURSION_REPRODUCED = NO
EPOCH108_EXCURSION_REPRODUCED = NO
SCORER_ENCODER_NEW_DESYNC_COUNT = 0
CANDIDATE_LARGE_EXCURSION_OBSERVED = NO
```

These conclusions use the same joint evidence as the earlier gate: scorer
delta, old-new candidate KL, argmax flip, and best-candidate probability/rank;
no single scalar is used as the excursion definition.

## Paired rollout comparison

The comparison below is target Stage2B minus the existing Stage1B short-gate
artifact for the same train-only realization seed. No baseline was rerun.

| epoch / train seed | Stage1B return | Stage2B return | delta return | Stage1B CacheChange | Stage2B CacheChange | delta Cache | delta UAVMove | delta critic EV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 / 20100 | -1342.55 | -1342.55 | 0.00 | 0.5663 | 0.5663 | 0.0000 | 0.0000 | 0.0000 |
| 101 / 20101 | -578.51 | -690.57 | -112.06 | 0.5993 | 0.6181 | +0.0189 | -0.0217 | +0.2557 |
| 102 / 20102 | 1173.01 | 1097.13 | -75.88 | 0.5785 | 0.6144 | +0.0359 | -0.0483 | +0.1843 |
| 103 / 20103 | 1401.88 | 844.89 | -556.99 | 0.5726 | 0.6552 | +0.0826 | +0.0117 | +0.0822 |
| 104 / 20104 | -10.74 | -761.40 | -750.66 | 0.5793 | 0.6926 | +0.1133 | -0.0533 | -0.0338 |
| 105 / 20105 | 950.99 | -127.13 | -1078.12 | 0.5796 | 0.7196 | +0.1400 | -0.0183 | +0.6553 |
| 106 / 20106 | 1250.27 | 1539.09 | +288.82 | 0.5963 | 0.7548 | +0.1585 | +0.0317 | +0.8986 |
| 107 / 20107 | 800.34 | 1147.99 | +347.65 | 0.5937 | 0.7752 | +0.1815 | +0.0417 | +0.2841 |
| 108 / 20108 | 722.23 | 1025.26 | +303.03 | 0.5593 | 0.7837 | +0.2244 | +0.0567 | +0.2870 |
| 109 / 20109 | 404.12 | 225.40 | -178.71 | 0.5741 | 0.7959 | +0.2219 | +0.0883 | +0.0751 |

```text
PAIRED_RETURN_DELTA_MEAN = -181.2932
PAIRED_RETURN_DELTA_MEDIAN = -93.9701
PAIRED_RETURN_POSITIVE_COUNT = 3 / 10
PAIRED_RETURN_NEGATIVE_COUNT = 6 / 10
PAIRED_RETURN_ZERO_COUNT = 1 / 10
PAIRED_CACHE_CHANGE_DELTA_MEAN = +0.117704
PAIRED_CACHE_CHANGE_DELTA_MEDIAN = +0.126667
PAIRED_UAV_MOVE_DELTA_MEAN = +0.008833
PAIRED_CRITIC_EV_DELTA_MEAN = +0.268850
```

The paired rollout result is mixed, with a negative mean and six negative
deltas, while later episodes include three positive deltas. The increased
CacheChange and mixed UAV movement make this a possible rollout degradation,
but ten paired episodes are not enough to claim statistical significance or a
final performance conclusion.

## Final gate

```text
STAGE2B_TDQ_NUMERICAL_STABILITY = PASS
CACHE_RATIO_NON_REGRESSION = PASS
UAV_STAGE1B_NON_REGRESSION = PASS
STAGE2B_MECHANISM_GATE = PASS
STAGE2B_ROLLOUT_EFFECT = POSSIBLE_DEGRADATION
STAGE2B_READY_TO_FREEZE = YES
CACHE_CANDIDATE_CREDIT_DESIGN = DEFERRED
```

`STAGE2B_READY_TO_FREEZE = YES` means the mechanism gate passed and the result
was not classified as a clear degradation; it does not mean the TD-Q trigger
credit has established final performance. The next longer validation should
pay particular attention to the upward CacheChange trend and rollout return,
while preserving the current candidate-level `GLOBAL_GAE` design.

```text
TRAINING_EXECUTED = YES (10 train-only validation episodes only)
NO_RESIDUAL_TRAINING_PROCESS = YES
COMMIT = NO
PUSH = NO
```
