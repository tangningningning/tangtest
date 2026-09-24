# Stage2A Cache per-head short-train gate

## Scope and provenance

This is a six-episode, train-only validation of Cache `PER_CACHE_HEAD` PPO
ratios.  It is not a formal run and it does not establish final performance.
The Stage1 UAV candidate changes were retained; the only intended algorithmic
change in this validation config was the Cache ratio granularity.

```text
TRAINING_EPISODES = 6
TRAIN_ONLY = YES
FORMAL_RUN_MODIFIED = NO
SOURCE_CHECKPOINT = runs/proposed_actor1e4_uavfix_300_epoch/latest_checkpoint.pt
SOURCE_COMPLETED_EPOCH = 99
SOURCE_SHA_BEFORE = 6081BC1DA279C9081BBC7EC466CA25E9B97902F65FCF1C6C8C9CCACED4ABBB
SOURCE_SHA_AFTER = 6081BC1DA279C9081BBC7EC466CA25E9B97902F65FCF1C6C8C9CCACED4ABBB
SEEDS_USED = 20100–20105
VALIDATION_RUN = runs/stage2a_cache_per_head_short_gate
CACHE_CREDIT_SOURCE = GLOBAL_GAE
CACHE_RATIO_GRANULARITY = PER_CACHE_HEAD
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3
UPDATE_EPOCHS = 4
NUM_MINIBATCHES = 10
EPISODE_SLOTS = 300
```

The source checkpoint was copied to the independent validation directory and
was not overwritten.  The validation completed exactly epochs 100--105; no
Stage1 episode was rerun.  The target config and checkpoint are validation
artifacts only.

The first process reached epochs 100 and 101, then stopped in the existing
slot-metrics continuity logger because the independent target directory did
not yet contain the historical epoch 1--99 slot prefix.  The target-only slot
artifact was repaired from the read-only source prefix and the already written
epoch-101 critic rewards, then the same target run resumed from the completed
epoch-101 checkpoint.  Epochs 100 and 101 were not rerun, and the six episode
and update records are present exactly once.  This artifact repair did not
change training tensors, losses, rewards, checkpoints, or the source run.

## Per-episode paired result

The train seed is the episode-pool seed implied by the common source cursor
(`episode_seed_start=20001` plus the recorded pool index).  The JSONL `seed`
field itself is the fixed experiment seed `4401`.

| epoch | train seed | Stage1 return | Stage2A return | delta return | Stage1 cache change | Stage2A cache change | delta cache | Stage1 UAV move | Stage2A UAV move | delta UAV | Stage1 critic EV | Stage2A critic EV | delta EV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 20100 | -1342.553 | -1342.553 | 0.000 | 0.5663 | 0.5663 | 0.0000 | 0.3667 | 0.3667 | 0.0000 | -0.1432 | -0.1432 | 0.0000 |
| 101 | 20101 | -541.211 | -578.513 | -37.302 | 0.5804 | 0.5993 | 0.0189 | 0.4600 | 0.4800 | 0.0200 | -0.0639 | -0.4339 | -0.3700 |
| 102 | 20102 | 1456.684 | 1173.011 | -283.673 | 0.6011 | 0.5785 | -0.0226 | 0.3967 | 0.4633 | 0.0667 | 0.2920 | 0.2136 | -0.0785 |
| 103 | 20103 | 1275.555 | 1401.882 | 126.327 | 0.5815 | 0.5726 | -0.0089 | 0.4517 | 0.4783 | 0.0267 | 0.3304 | 0.4114 | 0.0811 |
| 104 | 20104 | 88.742 | -10.743 | -99.485 | 0.5844 | 0.5793 | -0.0052 | 0.4350 | 0.4667 | 0.0317 | 0.6566 | 0.6686 | 0.0120 |
| 105 | 20105 | 434.480 | 950.991 | 516.512 | 0.5907 | 0.5796 | -0.0111 | 0.4233 | 0.4700 | 0.0467 | 0.0512 | -0.2847 | -0.3359 |

The paired return differences are mixed: two positive, three negative, and one
zero.  Cache-change deltas are small and mixed, while UAV movement is higher in
five of six paired rows but has no associated candidate-policy excursion in the
drift diagnostics.  These six pairs are descriptive only.

## Cache ratio diagnostics

The run manifest records `cache_trigger_advantage_source=GLOBAL_GAE` and
`cache_candidate_advantage_source=GLOBAL_GAE`.  It also records
`PER_CACHE_HEAD` for both trigger and selected-candidate ratios.  The
per-head surrogate was used for the loss; the legacy whole-cache joint ratio
was diagnostic-only.

```text
PER_HEAD_CLIP_FRACTION_MEAN = 0.000000
PER_HEAD_CLIP_FRACTION_P90 = 0.000000
PER_HEAD_CLIP_FRACTION_MAX = 0.000000
WHOLE_JOINT_CLIP_FRACTION_MEAN = 0.046944
WHOLE_JOINT_CLIP_FRACTION_P90 = 0.076667
WHOLE_JOINT_CLIP_FRACTION_MAX = 0.078333
WHOLE_CACHE_JOINT_RATIO_USED_FOR_LOSS = NO
WHOLE_CACHE_JOINT_RATIO_DIAGNOSTIC_ONLY = YES
```

Across the six Stage2A updates:

| diagnostic | mean | P50 | P90 | max |
|---|---:|---:|---:|---:|
| per-head old-new KL | 0.000626 | 0.000195 | 0.001866 | 0.008970 |
| per-head ratio mean | 0.999964 | 0.999906 | 1.000293 | 1.000338 |
| per-head ratio P90 | 1.030202 | 1.030017 | 1.041766 | 1.043163 |
| per-head ratio max | 1.117904 | 1.125951 | 1.145351 | 1.154263 |

All reported values were finite.  The per-head clip fraction was zero in every
episode, while the legacy whole-joint clip fraction was nonzero in every
episode.  Cache scorer parameter delta had mean `0.037463`, P90 `0.043756`,
and max `0.046734`; Cache encoder-gradient norm had mean `0.079767` and max
`0.084014`.  There is no Cache parameter-update explosion or Cache-change
jump in these six episodes.

The persisted artifact contains aggregate per-head statistics only.  It does
not contain nine separate head-indexed ratio/KL/clip arrays, and obtaining
those would require replay or additional logging.  Therefore:

```text
HEAD_LEVEL_CACHE_DIAGNOSTICS = INCONCLUSIVE
```

This limitation does not affect the aggregate mechanism gate.

```text
CACHE_RATIO_MECHANISM_GATE = PASS
```

The gate passes because the per-head loss is active in the manifest, the
per-head clip fraction is no higher than the legacy diagnostic (it is zero in
all six records), all ratio/KL values are finite, Cache updates remain bounded,
and the Stage1 UAV synchronization diagnostics do not regress.

## Stage1 UAV stability and confidence diagnostics

All six Stage2A records have 40 candidate optimizer steps, 40 UAV-encoder
optimizer steps, zero candidate-only-after-operation-stop steps, and a zero
scorer/encoder step gap.  The candidate target-KL stop epoch is absent in all
six records.

```text
SCORER_ENCODER_NEW_DESYNC_COUNT = 0
CANDIDATE_LARGE_EXCURSION_OBSERVED = NO
```

The candidate old-new KL mean was `0.001421` across episodes, with maximum
episode KL max `0.019142`; argmax-flip rate mean/max was `0.014236/0.035833`.
Best-teacher-candidate probability-change mean was `0.001949`; its P10
summary had mean `-0.001329`.  The best-policy probability mean was `0.099068`,
best-policy rank mean `4.959`, and top-1/top-3 rates were `0.154722/0.349167`.
Candidate and teacher entropy means were `2.675725` and `1.761325`.

The full-update scorer parameter delta reached `0.280922`, which is larger
than some ordinary Stage1 deltas but remains below the historical epoch-88
value `0.723598`.  More importantly, it was not accompanied by an epoch-88
style candidate-policy drift: the KL, flip, best-probability-change, and rank
signals remained bounded.  Thus the combined excursion check is `NO`, rather
than judging from scorer delta alone.

The newly persisted confidence diagnostics close the prior artifact gap:

```text
STAGE1_CONFIDENCE_P50_MEAN = 1.000000
STAGE1_CONFIDENCE_SATURATION_RATE_MEAN = 0.839167
STAGE1_CONFIDENCE_SATURATION_RATE_RANGE = 0.571667–0.991667
```

All six episode P50 values are 1.0.  The confidence weighting is therefore
highly saturated, but it is not suppressed toward zero: the episode confidence
mean ranges from `0.805741` to `0.996006`, and the saturation rate is not 1.0
in every episode.  No automatic change to reference scale `0.02` is made.

```text
STAGE1_CONFIDENCE_CALIBRATION = HIGH_SATURATION_NOT_OVER_SUPPRESSED
```

## Paired descriptive statistics

```text
PAIRED_RETURN_DELTA_MEAN = 37.063014
PAIRED_RETURN_DELTA_MEDIAN = -18.651071
PAIRED_RETURN_POSITIVE_COUNT = 2/6
PAIRED_RETURN_NEGATIVE_COUNT = 3/6
PAIRED_CACHE_CHANGE_DELTA_MEAN = -0.004815
PAIRED_UAV_MOVE_DELTA_MEAN = 0.031944
PAIRED_CRITIC_EV_DELTA_MEAN = -0.115211
```

The mixed return deltas do not support a claim of consistent improvement or
consistent degradation.  The negative critic-EV mean is also not treated as a
performance conclusion from six episodes.

## Final gate

```text
CACHE_RATIO_MECHANISM_GATE = PASS
SCORER_ENCODER_NEW_DESYNC_COUNT = 0
CANDIDATE_LARGE_EXCURSION_OBSERVED = NO
STAGE2A_ROLLOUT_EFFECT = NO_CLEAR_CHANGE
STAGE2A_READY_TO_EXTEND = YES
```

`STAGE2A_READY_TO_EXTEND = YES` means safe to consider a bounded next
mechanism-validation stage: the Cache ratio gate passed, Cache updates did not
explode, Stage1 UAV synchronization and candidate-policy drift remained
stable, and paired returns did not show consistent worsening.  It is not
approval for a 300-epoch formal run and it is not a claim of statistically
significant performance improvement.  The confidence scale remains visibly
high-saturation and should stay a monitored diagnostic; it was not changed in
this task.

```text
TRAINING_EXECUTED = YES (six train-only validation episodes; no formal run)
COMMIT = NO
PUSH = NO
```

The final workspace check was run with `git diff --check` after cleanup.

## 22-episode paired continuation

This section supersedes the provisional six-episode gate above for the
Stage2A continuation decision. The same validation run resumed from its
completed epoch-105 checkpoint and added exactly epochs 106--121. No
epoch-100--105 record was rerun, no Stage1 baseline was rerun, and Stage2B was
not started.

CONTINUATION_EPISODES = 16
TOTAL_STAGE2A_EPISODES = 22
CONTINUATION_EPOCHS = 106–121
CONTINUATION_SEEDS = 20106–20121
STAGE2A_FINAL_EPOCH = 121
STAGE2A_RUN_STATUS = COMPLETED
SOURCE_SHA_AFTER_CONTINUATION = 6081BC1DA279C9081BBC7EC466CA25E9B97902F65FCF1C6C8C9CCACED4ABBB
CACHE_CREDIT_SOURCE = GLOBAL_GAE
CACHE_RATIO_GRANULARITY = PER_CACHE_HEAD
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3
CANDIDATE_REFERENCE_SCALE = 0.02
CANDIDATE_TARGET_KL = 0.10
UPDATE_EPOCHS = 4
NUM_MINIBATCHES = 10
EPISODE_SLOTS = 300
STAGE2B_STARTED = NO

The target episode_metrics.jsonl and update_metrics.jsonl each contain exactly
the 22 epochs 100--121 with no duplicate epoch. The Stage1 baseline also
contains all 22 corresponding epochs and was read only.

### Paired rollout table

Train seeds are mapped from the common train-only episode-pool cursor. Cache
change and UAV move are rates.

| epoch | seed | Stage1 return | Stage2A return | delta return | Stage1 cache | Stage2A cache | delta cache | Stage1 UAV | Stage2A UAV | delta UAV | Stage1 critic EV | Stage2A critic EV | delta EV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 20100 | -1342.553 | -1342.553 | 0.000 | 0.5663 | 0.5663 | 0.0000 | 0.3667 | 0.3667 | 0.0000 | -0.1432 | -0.1432 | 0.0000 |
| 101 | 20101 | -541.211 | -578.513 | -37.302 | 0.5804 | 0.5993 | 0.0189 | 0.4600 | 0.4800 | 0.0200 | -0.0639 | -0.4339 | -0.3700 |
| 102 | 20102 | 1456.684 | 1173.011 | -283.673 | 0.6011 | 0.5785 | -0.0226 | 0.3967 | 0.4633 | 0.0667 | 0.2920 | 0.2136 | -0.0785 |
| 103 | 20103 | 1275.555 | 1401.882 | 126.327 | 0.5815 | 0.5726 | -0.0089 | 0.4517 | 0.4783 | 0.0267 | 0.3304 | 0.4114 | 0.0811 |
| 104 | 20104 | 88.742 | -10.743 | -99.485 | 0.5844 | 0.5793 | -0.0052 | 0.4350 | 0.4667 | 0.0317 | 0.6566 | 0.6686 | 0.0120 |
| 105 | 20105 | 434.480 | 950.991 | 516.512 | 0.5907 | 0.5796 | -0.0111 | 0.4233 | 0.4700 | 0.0467 | 0.0512 | -0.2847 | -0.3359 |
| 106 | 20106 | 1051.489 | 1250.267 | 198.779 | 0.5800 | 0.5963 | 0.0163 | 0.5050 | 0.4450 | -0.0600 | 0.2899 | -0.2595 | -0.5494 |
| 107 | 20107 | 1859.779 | 800.339 | -1059.440 | 0.5704 | 0.5937 | 0.0233 | 0.4817 | 0.4367 | -0.0450 | -0.0058 | 0.0548 | 0.0606 |
| 108 | 20108 | 1513.756 | 380.893 | -1132.864 | 0.5715 | 0.5763 | 0.0048 | 0.4967 | 0.4283 | -0.0683 | 0.2850 | -0.0049 | -0.2899 |
| 109 | 20109 | 1012.984 | 52.524 | -960.460 | 0.5933 | 0.5567 | -0.0367 | 0.4433 | 0.4583 | 0.0150 | 0.6457 | 0.5067 | -0.1390 |
| 110 | 20110 | 1285.688 | 1215.227 | -70.462 | 0.5722 | 0.5544 | -0.0178 | 0.4850 | 0.4650 | -0.0200 | 0.4760 | 0.1284 | -0.3475 |
| 111 | 20111 | 622.673 | -252.036 | -874.709 | 0.5833 | 0.5426 | -0.0407 | 0.4600 | 0.4583 | -0.0017 | 0.4736 | 0.1994 | -0.2743 |
| 112 | 20112 | 121.116 | -649.544 | -770.660 | 0.5741 | 0.5344 | -0.0396 | 0.4850 | 0.5167 | 0.0317 | 0.7700 | 0.6295 | -0.1405 |
| 113 | 20113 | 2053.263 | 1794.697 | -258.565 | 0.5819 | 0.5211 | -0.0607 | 0.4750 | 0.4833 | 0.0083 | 0.6023 | -0.1376 | -0.7399 |
| 114 | 20114 | 449.883 | -397.675 | -847.558 | 0.5870 | 0.5270 | -0.0600 | 0.4117 | 0.5483 | 0.1367 | -0.1170 | 0.2558 | 0.3728 |
| 115 | 20115 | 1181.818 | 1154.567 | -27.251 | 0.5826 | 0.5326 | -0.0500 | 0.4567 | 0.5033 | 0.0467 | -0.0804 | -0.0894 | -0.0090 |
| 116 | 20116 | 657.120 | -385.939 | -1043.059 | 0.5556 | 0.5541 | -0.0015 | 0.4750 | 0.4733 | -0.0017 | 0.2817 | 0.3195 | 0.0377 |
| 117 | 20117 | 1769.292 | 1915.342 | 146.050 | 0.5959 | 0.5530 | -0.0430 | 0.4750 | 0.4967 | 0.0217 | 0.0674 | 0.4756 | 0.4082 |
| 118 | 20118 | 1064.211 | -286.028 | -1350.239 | 0.6137 | 0.5704 | -0.0433 | 0.4900 | 0.4383 | -0.0517 | 0.8888 | 0.6657 | -0.2230 |
| 119 | 20119 | -280.538 | 256.431 | 536.969 | 0.6100 | 0.5848 | -0.0252 | 0.5200 | 0.4700 | -0.0500 | 0.0519 | 0.2675 | 0.2157 |
| 120 | 20120 | 1322.677 | 1200.199 | -122.478 | 0.6252 | 0.5652 | -0.0600 | 0.4967 | 0.4400 | -0.0567 | -0.4595 | -0.1748 | 0.2847 |
| 121 | 20121 | -77.137 | 782.222 | 859.359 | 0.6244 | 0.5663 | -0.0581 | 0.4933 | 0.5083 | 0.0150 | 0.4639 | 0.1323 | -0.3316 |

### Cache mechanism over all 22 episodes

The following are episode-level aggregates emitted by the run; percentile
values are percentiles across the 22 episode records.

| metric | mean | P50 | P90 | max |
|---|---:|---:|---:|---:|
| CACHE_PER_HEAD_CLIP_FRACTION | 0.000598 | 0.000000 | 0.001778 | 0.005648 |
| legacy whole-joint cache clip fraction | 0.076705 | 0.077917 | 0.135750 | 0.137500 |
| CACHE_PER_HEAD_OLD_NEW_KL_MEAN | 0.000938 | 0.000946 | 0.001430 | 0.001776 |
| CACHE_PER_HEAD_OLD_NEW_KL_P90 | 0.002641 | 0.002765 | 0.003979 | 0.005257 |
| CACHE_PER_HEAD_OLD_NEW_KL_MAX | 0.009451 | 0.009074 | 0.014564 | 0.022859 |
| CACHE_PER_HEAD_RATIO_MEAN | 0.999806 | 0.999813 | 1.000335 | 1.000986 |
| CACHE_PER_HEAD_RATIO_P90 | 1.037284 | 1.037351 | 1.049754 | 1.065580 |
| CACHE_PER_HEAD_RATIO_MAX | 1.158890 | 1.160385 | 1.222600 | 1.250092 |

Cache scorer parameter delta had mean 0.043492, P50 0.042516, P90
0.052730, and max 0.070646. Cache encoder-gradient norm had mean
0.078619, P50 0.078131, P90 0.082414, and max 0.084014. The per-head clip
fraction remains substantially below the legacy whole-joint fraction throughout
the 22 records, and the Cache parameter/gradient signals do not show an update
explosion.

PER_HEAD_CLIP_FRACTION_LONG_RUN = 0.000598 / 0.000000 / 0.001778 / 0.005648
WHOLE_JOINT_CLIP_FRACTION_LONG_RUN = 0.076705 / 0.077917 / 0.135750 / 0.137500
CACHE_RATIO_MECHANISM_GATE = PASS

The Cache ratio mechanism itself is therefore sustained as healthy. This does
not imply that the full Stage2A change is sufficient for rollout performance.

### CacheChange trajectory

Across Stage2A epochs 100--121, CacheChange mean was 0.563838. The first six
mean was 0.579259 and the last six mean was 0.565617; the ordinary
least-squares slope over epoch index was -0.001463 rate units per epoch. The
observed range was 0.521111--0.599259.

CACHE_CHANGE_MEAN = 0.563838
CACHE_CHANGE_FIRST6_MEAN = 0.579259
CACHE_CHANGE_LAST6_MEAN = 0.565617
CACHE_CHANGE_OLS_SLOPE_PER_EPOCH = -0.001463

The lower late CacheChange is not treated as intrinsically better: the
paired-return windows below move in the opposite direction, so CacheChange is
reported as behavior rather than as an objective improvement.

### Stage1 UAV regression check

All 22 records have 40 candidate optimizer steps, 40 encoder optimizer steps,
zero candidate-only-after-operation-stop steps, and a zero scorer/encoder step
gap. Thus there is no optimizer-step desynchronization:

SCORER_ENCODER_NEW_DESYNC_COUNT = 0
UAV_CANDIDATE_OPTIMIZER_STEP_TOTAL = 880
UAV_ENCODER_OPTIMIZER_STEP_TOTAL = 880
UAV_CANDIDATE_ONLY_STEPS_AFTER_OPERATION_STOP = 0
UAV_CANDIDATE_KL_STOP_EPOCH_DISTRIBUTION = none (22/22)

The policy-drift result is different from the six-episode provisional result.
Epochs 107--108 show a joint candidate excursion:

| epoch | scorer delta | encoder delta | candidate KL mean | candidate KL P90 | candidate KL max | argmax flip | best-prob change mean | best-prob change P10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 107 | 0.582268 | 0.681420 | 0.076432 | 0.117297 | 0.160561 | 0.095000 | -0.008590 | -0.027004 |
| 108 | 0.571597 | 0.669548 | 0.070842 | 0.117045 | 0.291551 | 0.169167 | 0.027897 | 0.007660 |

The epoch-107 update-epoch arrays show scorer deltas
[0.190416, 0.174284, 0.185314, 0.209958], candidate-KL maxima up to
0.160561, and argmax flips rising to 0.211667. At epoch 108 the
candidate-KL maximum reaches 0.291551 and every update-epoch argmax flip is
about 0.16--0.17. Epoch 121 also has a large aggregate argmax flip rate
(0.295) even though its candidate-KL maximum is only 0.020572, so the
argmax signal is retained as a separate late warning.

For comparison, the 22-episode Stage1-only baseline has scorer-delta maximum
0.240776, candidate-KL-maximum 0.028009, and argmax-flip maximum 0.029583.
The Stage2A excursion is therefore not explained by scorer delta alone and is
not a mere Cache clipping statistic. It is a candidate-policy drift and
ranking instability in the Stage2A continuation. The candidate-KL stop field
is None for all 22 records; this is recorded as observed behavior, not altered
by lowering the threshold.

CANDIDATE_LARGE_EXCURSION_OBSERVED = YES
STAGE1_UAV_POLICY_STABILITY = FAIL

The new confidence diagnostics over all 22 episodes are:

STAGE1_CONFIDENCE_P50_MEAN = 0.933350
STAGE1_CONFIDENCE_SATURATION_RATE_MEAN = 0.736364
STAGE1_CONFIDENCE_P50_MIN = 0.494543
STAGE1_CONFIDENCE_SATURATION_RATE_MIN = 0.090000
STAGE1_CONFIDENCE_SATURATION_RATE_MAX = 0.995000

The reference scale remains 0.02. Confidence is high-saturated on average but
becomes substantially less saturated during the 107--108 excursion; no scale
or weighting change was made.

### 22-episode paired statistics

PAIRED_RETURN_DELTA_MEAN = -297.918678
PAIRED_RETURN_DELTA_MEDIAN = -110.981859
PAIRED_RETURN_POSITIVE_COUNT = 6/22
PAIRED_RETURN_NEGATIVE_COUNT = 15/22
PAIRED_RETURN_ZERO_COUNT = 1/22
PAIRED_CACHE_CHANGE_DELTA_MEAN = -0.023687
PAIRED_CACHE_CHANGE_DELTA_MEDIAN = -0.023889
PAIRED_UAV_MOVE_DELTA_MEAN = 0.005076
PAIRED_CRITIC_EV_DELTA_MEAN = -0.107079

The requested paired return windows are:

| window | episodes | paired return delta mean | positive / negative |
|---|---:|---:|---:|
| 100--105 | 6 | 37.063014 | 2 / 3 (one zero) |
| 106--111 | 6 | -649.859394 | 1 / 5 |
| 112--121 | 10 | -287.743264 | 3 / 7 |

The early six episodes were mixed, followed by negative paired-return means in
both continuation windows. This is a descriptive paired degradation signal,
not a statistical-significance claim.

## Final 22-episode judgment

CACHE_RATIO_FIX_SUFFICIENT = NO
STAGE2A_MECHANISM_STABILITY = FAIL
STAGE2A_ROLLOUT_EFFECT = POSSIBLE_DEGRADATION
STAGE2B_NEEDED = YES

The Cache PER_CACHE_HEAD ratio mechanism is healthy and remains clearly less
clipped than the legacy whole-joint diagnostic. However, it is not sufficient
as a rollout-level fix: the 22 paired results do not show stable improvement,
the two continuation windows are negative, and the Stage2A continuation
contains a concrete UAV candidate-policy excursion despite zero
scorer/encoder step desynchronization. Therefore the evidence does not support
treating whole-joint Cache ratio as the remaining primary performance
bottleneck; a future Stage2B credit-assignment investigation is needed, but it
was not started here.

COMMIT = NO
PUSH = NO
