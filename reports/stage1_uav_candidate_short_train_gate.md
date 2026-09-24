# Stage1 UAV candidate short-train gate (epochs 100--121)

## Scope and provenance

This file contains the initial six-episode gate (epochs 100--105) and its
16-episode train-only continuation (epochs 106--121).  It is not a formal run
and it is not a performance study.
The source checkpoint was copied before the validation; the source run and its
metrics were not written.

```text
TRAINING_EPISODES = 6
TRAIN_ONLY = YES
FORMAL_RUN_MODIFIED = NO
SOURCE_RUN = runs/proposed_actor1e4_uavfix_300_epoch
SOURCE_CHECKPOINT = runs/proposed_actor1e4_uavfix_300_epoch/latest_checkpoint.pt
SOURCE_COMPLETED_EPOCH = 99
SOURCE_NEXT_SEED = 20100
SEEDS_USED = 20100, 20101, 20102, 20103, 20104, 20105
VALIDATION_RUN = runs/stage1_uav_candidate_short_gate
```

The source checkpoint SHA-256 at copy time and after validation was
`6081BC1DA279C9081BBC7EC466CA25E9B97902F65FCF1C6C8C8C9CCACED4ABBB`.
The validation used the copied source `run_config.toml`: actor LR `1e-4`,
critic LR `1e-3`, `update_epochs=4`, `num_minibatches=10`, and the existing
full-episode-randomized 300-slot train protocol.  The copied config predates
the explicit candidate-scale key; the current loader falls back to the
existing operation reference scale `0.02`, and the candidate target-KL field
falls back to `target_kl=0.10`.  No loss, reward, teacher, optimizer, LR,
Matching, KKT, or formal hyperparameter was changed.

The initial validation generated epochs 100--105 and the continuation generated
epochs 106--121.  The source epochs 1--99 in
the validation log are read-only historical log prefixes copied solely to
satisfy the existing resumed slot-metrics continuity check; all gate numbers
below are computed from the six newly generated epochs only.

## Per-episode result

| epoch | train seed | return | cache change | UAV move | critic EV | confidence mean | confidence P10/P90 | old-new KL mean / max | argmax flip | best-policy prob change mean / P10 | scorer delta (full update) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 20100 | -1342.553 | 0.5663 | 0.3667 | -0.1432 | 0.9830 | 1.0000 / 1.0000 | 0.000279 / 0.003361 | 0.0171 | 0.002051 / 0.000271 | 0.199395 |
| 101 | 20101 | -541.211 | 0.5804 | 0.4600 | -0.0639 | 0.9976 | 1.0000 / 1.0000 | 0.000256 / 0.001063 | 0.0117 | 0.001043 / 0.000081 | 0.170267 |
| 102 | 20102 | 1456.684 | 0.6011 | 0.3967 | 0.2920 | 0.9119 | 0.5873 / 1.0000 | 0.000081 / 0.001033 | 0.0000 | 0.000497 / 0.000157 | 0.127933 |
| 103 | 20103 | 1275.555 | 0.5815 | 0.4517 | 0.3304 | 0.9335 | 0.7026 / 1.0000 | 0.000194 / 0.000871 | 0.0221 | 0.000471 / -0.001194 | 0.220683 |
| 104 | 20104 | 88.742 | 0.5844 | 0.4350 | 0.6566 | 0.9182 | 0.5885 / 1.0000 | 0.000108 / 0.001683 | 0.0296 | 0.000379 / -0.000436 | 0.127048 |
| 105 | 20105 | 434.480 | 0.5907 | 0.4233 | 0.0512 | 0.9935 | 1.0000 / 1.0000 | 0.000282 / 0.001840 | 0.0046 | 0.001638 / 0.000619 | 0.150263 |

Six-episode descriptive means were return `228.616`, cache change `0.5841`,
UAV movement `0.4222`, and critic explained variance `0.1872`.  These are
reported for traceability only; this sample is too small for a performance
conclusion.

## Confidence and candidate policy drift

```text
CONFIDENCE_NONDEGENERATE = YES
REFERENCE_SCALE_TOO_LARGE = NO
REFERENCE_SCALE_TOO_SMALL_OR_INEFFECTIVE = NO
```

The six confidence-P10 values were
`[1.0000, 1.0000, 0.5873, 0.7026, 0.5885, 1.0000]`; P90 was `1.0000` in
each episode.  Thus the episodes were not all P10=P90=0 or all P10=P90=1.
Only 3/6 episodes had both quantiles equal to one, so the short-gate
``almost all one'' failure criterion was not met.  The confidence mean ranged
from `0.9119` to `0.9976` (mean `0.9563`).

Across the six new updates, the old-vs-new candidate KL mean was `0.000200`
on average, with a largest per-head KL of `0.003361`; the maximum remained
far below the candidate target `0.10`.  The candidate policy entropy mean was
`2.6842`, teacher entropy mean `1.7560`, argmax flip rate mean `0.0142`
(maximum `0.0296`), and best-teacher-candidate probability-change mean
`0.001013` (minimum P10 `-0.001683`).  The emitted per-update-epoch arrays
contain all 24 epoch-level values for scorer/encoder gradients, parameter
deltas, old-new KL, argmax flips, and best-probability changes.

```text
CANDIDATE_KL_GATE_TRIGGER_COUNT = 0
CANDIDATE_KL_GATE_TOO_AGGRESSIVE = NO
```

All six records had `UAV_CANDIDATE_ACTIVE_PER_UPDATE_EPOCH=[true,true,true,true]`
and no candidate stop epoch.  Therefore the candidate-KL stop path was not
triggered in this six-episode sample; the observed result is that the gate was
not too aggressive, not that a threshold crossing was manufactured.

## Optimizer synchronization and excursion check

Each new update executed 40 candidate steps and 40 UAV-encoder steps, with
zero candidate-only-after-operation-stop steps and zero scorer/encoder step
gap.  Totals were 240 candidate steps, 240 encoder steps, and 0 observed
operation-induced scorer-only gaps.

```text
SCORER_ENCODER_NEW_DESYNC_COUNT = 0
SCORER_ENCODER_SYNC_GATE = PASS
CANDIDATE_LARGE_EXCURSION_OBSERVED = NO
```

The 24 epoch-level candidate-scorer parameter deltas had mean `0.046435`,
P90 `0.058793`, and maximum `0.069442`; the full-update scorer delta had mean
`0.165931` and maximum `0.220683`.  For comparison, the pre-Stage1 source
trajectory had full-update scorer delta `0.206767` at epoch 87 and `0.723598`
at epoch 88 (the historical `3.50x` jump).  The new maximum is below the
historical epoch-88 excursion and is accompanied by sub-`0.004` candidate
KL, not a matching ranking/probability collapse.  The encoder epoch-level
delta mean/max were `0.041373/0.101804`; scorer gradient mean/max were
`0.058174/0.090287`.

```text
CANDIDATE_POLICY_STABILITY_GATE = PASS
```

This pass is limited to the six train-only continuation episodes.  It does
not establish performance improvement, universal stability, or exercise the
candidate stop branch at `KL >= 0.10`.

## Required gate summary

```text
TRAINING_EPISODES = 6
TRAIN_ONLY = YES
FORMAL_RUN_MODIFIED = NO
SOURCE_RUN = runs/proposed_actor1e4_uavfix_300_epoch
SOURCE_CHECKPOINT = runs/proposed_actor1e4_uavfix_300_epoch/latest_checkpoint.pt
SOURCE_COMPLETED_EPOCH = 99
SEEDS_USED = 20100, 20101, 20102, 20103, 20104, 20105
CONFIDENCE_NONDEGENERATE = YES
REFERENCE_SCALE_TOO_LARGE = NO
REFERENCE_SCALE_TOO_SMALL_OR_INEFFECTIVE = NO
CANDIDATE_KL_GATE_TRIGGER_COUNT = 0
CANDIDATE_KL_GATE_TOO_AGGRESSIVE = NO
SCORER_ENCODER_NEW_DESYNC_COUNT = 0
SCORER_ENCODER_SYNC_GATE = PASS
CANDIDATE_LARGE_EXCURSION_OBSERVED = NO
CANDIDATE_POLICY_STABILITY_GATE = PASS
RETURN_MEAN = 228.616139
CACHE_CHANGE_MEAN = 0.584074
UAV_MOVE_MEAN = 0.422222
CRITIC_EV_MEAN = 0.187190
PAIRED_ROLLOUT_BASELINE_AVAILABLE = NO
PAIRED_INTERNAL_CANDIDATE_BASELINE_AVAILABLE = NO
STAGE1_SHORT_GATE = PASS
```

`PAIRED_ROLLOUT_BASELINE_AVAILABLE = NO` in the current workspace because the
specified source `runs/proposed_actor1e4_uavfix_300_epoch` has only 99 episode
records (epochs 1--99, seeds 20001--20099); its local artifact does not contain
epochs 100--121.  The other directories that contain these seeds use different
LRs or experiment protocols and are not valid paired baselines.  No baseline
was rerun and no substitute baseline was used.  Consequently, the paired
delta fields in the continuation section are `INCONCLUSIVE`, not fabricated.

## Paired continuation to epoch 121

The existing short-validation checkpoint resumed at epoch 105 and completed
exactly 16 new train-only episodes, seeds 20106--20121.  No epoch 100--105
episode was rerun, and the final run summary is `completed_epochs=121`,
`status=completed`.  The continuation used the unchanged actor LR `1e-4`,
critic LR `1e-3`, candidate reference scale `0.02`, candidate target KL
`0.10`, `update_epochs=4`, `num_minibatches=10`, and 300 slots.

### Rollout-level paired baseline availability

```text
PAIRED_ROLLOUT_BASELINE_AVAILABLE = NO
PAIRED_INTERNAL_CANDIDATE_BASELINE_AVAILABLE = NO
```

The requested source path was checked directly before and after continuation:
`run_summary.json` reports `completed_epochs=99`, and its episode log has no
seed 20100--20121 records.  Therefore no per-seed rows can be generated for
`baseline_return`, `baseline_cache_change`, `baseline_uav_move`, or baseline
critic EV without violating the no-baseline-rerun/no-fabrication constraint.
The internal candidate baseline is also unavailable even if rollout records
had existed, because the old source log predates the Stage1 confidence and
old-new candidate-drift fields.

| epoch | seed | Stage1 return | cache change | UAV move | critic EV |
|---:|---:|---:|---:|---:|---:|
| 106 | 20106 | 1051.489 | 0.5800 | 0.5050 | 0.2899 |
| 107 | 20107 | 1859.779 | 0.5704 | 0.4817 | -0.0058 |
| 108 | 20108 | 1513.756 | 0.5715 | 0.4967 | 0.2850 |
| 109 | 20109 | 1012.984 | 0.5933 | 0.4433 | 0.6457 |
| 110 | 20110 | 1285.688 | 0.5722 | 0.4850 | 0.4760 |
| 111 | 20111 | 622.673 | 0.5833 | 0.4600 | 0.4736 |
| 112 | 20112 | 121.116 | 0.5741 | 0.4850 | 0.7700 |
| 113 | 20113 | 2053.263 | 0.5819 | 0.4750 | 0.6023 |
| 114 | 20114 | 449.883 | 0.5870 | 0.4117 | -0.1170 |
| 115 | 20115 | 1181.818 | 0.5826 | 0.4567 | -0.0804 |
| 116 | 20116 | 657.120 | 0.5556 | 0.4750 | 0.2817 |
| 117 | 20117 | 1769.292 | 0.5959 | 0.4750 | 0.0674 |
| 118 | 20118 | 1064.211 | 0.6137 | 0.4900 | 0.8888 |
| 119 | 20119 | -280.538 | 0.6100 | 0.5200 | 0.0519 |
| 120 | 20120 | 1322.677 | 0.6252 | 0.4967 | -0.4595 |
| 121 | 20121 | -77.137 | 0.6244 | 0.4933 | 0.4639 |

Because the baseline rows are absent, the requested descriptive paired fields
are:

```text
PAIRED_RETURN_DELTA_MEAN = INCONCLUSIVE
PAIRED_RETURN_DELTA_MEDIAN = INCONCLUSIVE
PAIRED_CACHE_CHANGE_DELTA_MEAN = INCONCLUSIVE
PAIRED_UAV_MOVE_DELTA_MEAN = INCONCLUSIVE
positive_return_delta_count / 22 = INCONCLUSIVE
negative_return_delta_count / 22 = INCONCLUSIVE
```

The 22-episode Stage1-only descriptive means are return `771.808`, cache
change `0.5875`, UAV movement `0.4629`, and critic EV `0.2617`.  They are not
a baseline comparison and do not establish a performance conclusion.

## 22-episode internal stability continuation

All 22 Stage1 updates had 4 active candidate epochs, 40 operation steps, 40
candidate optimizer steps, 40 encoder optimizer steps, and a zero scorer/encoder
step gap.  Totals were 880 candidate steps and 880 encoder steps.  No candidate
KL stop occurred.

```text
SCORER_ENCODER_NEW_DESYNC_COUNT = 0
CANDIDATE_KL_GATE_TRIGGER_COUNT = 0
CANDIDATE_KL_GATE_REAL_TRAIN_EXERCISED = NO
CANDIDATE_STOP_EPOCH_DISTRIBUTION = none (22/22)
```

The 22-update candidate old-new KL mean had mean `0.001279`, episode-level P90
`0.003465`, and maximum `0.028009`; the corresponding 88 flattened
update-epoch maxima never exceeded `0.028009`, well below the `0.10` safety
backstop.  The gate therefore remained a backstop and was not in the control
region; it was not failed or artificially triggered.

Candidate policy entropy mean was `2.6589`, teacher entropy mean `1.7630`,
argmax flip rate mean/max was `0.00606/0.02958`, best-policy probability mean
was `0.09960`, best-policy rank mean was `5.162`, and top-1/top-3 rates were
`0.1094/0.3498`.

The full-update scorer parameter delta mean/max was `0.145700/0.240776`; the
88 update-epoch scorer delta mean/max was `0.043701/0.091216`.  The historical
pre-Stage1 epoch-87/88 full-update deltas were `0.206767/0.723598` (3.50x).
The largest new scorer delta occurred at epoch 117, where candidate KL max was
`0.028009`, argmax flip was `0.00458`, best-policy probability mean was
`0.10762`, and best-policy rank was `3.707`; these do not show the historical
joint excursion pattern.  The lowest best-policy probability-change P10 was
at epoch 116 (`-0.00814`) with candidate KL max `0.01453` and flip rate
`0.00167`, still without the historical scorer jump.

```text
CANDIDATE_LARGE_EXCURSION_OBSERVED = NO
STAGE1_MECHANISM_STABILITY = PASS
```

### Confidence calibration limitation

The 22 existing records contain confidence mean/P10/P90 only.  Their summary
was mean confidence `0.9277`, mean of logged P10 `0.7676`, and logged P90
`1.0000` in all 22 records; 12/22 episode-level P10 values were already
`>=0.999`, and 4/22 were `<0.5`.  These are quantile proxies, not the required
active-head saturation fraction.  Neither active-head P50 nor the exact count
of heads with confidence `>=0.999` was persisted, so neither can be recovered
without rerunning a rollout.

The diagnostic code now emits `UAV_CANDIDATE_CONFIDENCE_P50` and
`UAV_CANDIDATE_CONFIDENCE_SATURATION_RATE` for future runs; this change is
logging-only and was made after epoch 121, so it does not retroactively alter
the 22 completed episodes.

```text
CONFIDENCE_P50 = INCONCLUSIVE (not persisted for epochs 100--121)
CONFIDENCE_SATURATION_RATE = INCONCLUSIVE (active-head values not persisted)
STAGE1_CONFIDENCE_CALIBRATION = INCONCLUSIVE
```

## Final continuation judgment

```text
STAGE1_MECHANISM_STABILITY = PASS
STAGE1_ROLLOUT_DEGRADATION = NO_CLEAR_DEGRADATION
STAGE1_CONFIDENCE_CALIBRATION = INCONCLUSIVE
STAGE1_READY_TO_FREEZE = NO
```

`NO_CLEAR_DEGRADATION` is only a within-run descriptive statement; the paired
rollout test was not available in the current workspace.  Freeze readiness is
`NO` because the required active-head confidence saturation and P50 were not
recorded for the already completed episodes, not because a candidate KL gate
failure or scorer/encoder desynchronization was observed.
