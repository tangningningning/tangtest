# STAGE1B cross-branch short-train gate

## Scope and provenance

This is a short validation run, not a formal training run and not Stage2B.
Only the train-only pool entries `episode_20100` through `episode_20109` were
used. No SUMO, task, prediction, scenario, Matching, or KKT data was
regenerated. The source run and its checkpoint were not modified.

| item | value |
|---|---|
| source run | `runs/proposed_actor1e4_uavfix_300_epoch` |
| source completed epoch | 99 |
| source checkpoint SHA256 before/after | `6081BC1DA279C9081BBC7EC466CA25E9B97902F65FCF1C6C8C8C9CCACED4ABBB` |
| validation run | `runs/stage1b_cross_branch_short_gate` |
| validation epochs | 100--109 only |
| train-only episode pool | 20100--20109 |
| target status | completed, epoch 109/109 |
| NaN/Inf status | none reported |

The validation run used the one-Adam-actor configuration with actor LR
`1e-4`, critic LR `1e-3`, `update_epochs=4`, `num_minibatches=10`,
`episode_slots=300`, `max_grad_norm=0.5`, Cache `GLOBAL_GAE`, Cache ratio
`PER_CACHE_HEAD`, candidate confidence reference scale `0.02`, and candidate
target KL `0.10`. Its manifest records the candidate gate statistic as
`MEAN_OR_P90` and the UAV candidate teacher as `EXACT_Q_LISTWISE`.

The top-level `seed=4401` in the JSONL is the global experiment seed. The
`20100`--`20109` labels in the tables are the train-only episode-pool seeds,
identified by the corresponding `environment_data/.../episode_20xxx`
realization directories.

## Candidate stability by episode

The `best` columns use the detached Exact-Q teacher's highest-Q legal
candidate set. `scorer_delta` is the UAV candidate scorer parameter delta
norm; `encoder_delta` is the UAV encoder parameter delta norm. `gap` is the
raw candidate-scorer optimizer-step count minus encoder optimizer-step count.

| epoch / train seed | KL mean | KL P90 | KL max | gate source / stop epoch | argmax flip | scorer delta | encoder delta | best prob | best rank | top1 / top3 | gap |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 100 / 20100 | 0.000561 | 0.001648 | 0.006626 | NONE / -- | 0.027083 | 0.280922 | 0.169410 | 0.100261 | 4.912 | 0.222 / 0.387 | 0 |
| 101 / 20101 | 0.000647 | 0.001448 | 0.005959 | NONE / -- | 0.035833 | 0.236136 | 0.189936 | 0.096773 | 5.280 | 0.158 / 0.322 | 0 |
| 102 / 20102 | 0.001081 | 0.001897 | 0.003668 | NONE / -- | 0.013750 | 0.200072 | 0.141122 | 0.100399 | 4.943 | 0.160 / 0.387 | 0 |
| 103 / 20103 | 0.002275 | 0.004759 | 0.009706 | NONE / -- | 0.003333 | 0.205921 | 0.163023 | 0.099747 | 4.397 | 0.185 / 0.402 | 0 |
| 104 / 20104 | 0.003325 | 0.004950 | 0.019142 | NONE / -- | 0.001667 | 0.195570 | 0.155535 | 0.099117 | 4.992 | 0.192 / 0.340 | 0 |
| 105 / 20105 | 0.000636 | 0.002275 | 0.005807 | NONE / -- | 0.003750 | 0.162976 | 0.197095 | 0.098113 | 5.233 | 0.012 / 0.258 | 0 |
| 106 / 20106 | 0.001564 | 0.002970 | 0.008789 | NONE / -- | 0.000417 | 0.271399 | 0.166074 | 0.093520 | 5.508 | 0.060 / 0.192 | 0 |
| 107 / 20107 | 0.087497 | 0.133092 | 0.178773 | P90 / 2 | 0.064583 | 0.353601 | 0.592004 | 0.067291 | 6.070 | 0.047 / 0.155 | -20 |
| 108 / 20108 | 0.068966 | 0.122787 | 0.158195 | P90 / 2 | 0.046250 | 0.389783 | 0.689221 | 0.103246 | 4.500 | 0.187 / 0.428 | -20 |
| 109 / 20109 | 0.002038 | 0.003528 | 0.006270 | NONE / -- | 0.010833 | 0.280640 | 0.200409 | 0.094889 | 4.905 | 0.088 / 0.318 | 0 |

The two `gap=-20` rows are intentional gate suppression: candidate updates
ran for 2 of 4 PPO update epochs (`UAV_CANDIDATE_FULL_STEPS=20`), while the
UAV encoder still participated in the operation branch for all 4 epochs
(`UAV_ENCODER_OPTIMIZER_STEP_COUNT=40`). There were zero
`UAV_CANDIDATE_ONLY_STEPS_AFTER_OPERATION_STOP` in the ten episodes. Thus:

`SCORER_ENCODER_NEW_DESYNC_COUNT = 0`

Here “new desync” means an unintended candidate-only/stale-gradient update,
not the deliberate encoder-only tail after the candidate KL safety gate. The
raw step-gap episode count is `2`, and both occurrences are explained by the
P90 gate.

Additional candidate-policy summaries over the ten episodes:

| metric | mean | median | min | max |
|---|---:|---:|---:|---:|
| policy entropy | 2.694263 | 2.675671 | 2.661214 | 2.872725 |
| teacher entropy | 1.794399 | 1.776554 | 1.725068 | 1.952076 |
| best-policy probability | 0.095336 | 0.098615 | 0.067291 | 0.103246 |
| best-policy rank | 5.074 | 4.967 | 4.397 | 6.070 |
| argmax flip rate | 0.020750 | 0.012292 | 0.000417 | 0.064583 |
| top-1 rate | 0.131000 | 0.159167 | 0.011667 | 0.221667 |
| top-3 rate | 0.318833 | 0.330833 | 0.155000 | 0.428333 |
| confidence P50 | 0.913171 | 1.000000 | 0.494543 | 1.000000 |
| confidence saturation rate | 0.684167 | 0.719167 | 0.090000 | 0.991667 |

The saturation rate is a diagnostic observation, not a change to the `0.02`
reference scale.

## Candidate KL gate correctness

The gate source counts are:

| source | count |
|---|---:|
| `P90` only | 2 |
| `MEAN` only | 0 |
| `BOTH` | 0 |
| `NONE` | 8 |

Epochs 107 and 108 both satisfy `KL_MEAN < 0.10` and `KL_P90 >= 0.10`.
Both report `UAV_CANDIDATE_KL_GATE_TRIGGER_SOURCE=P90` and
`UAV_CANDIDATE_KL_STOP_EPOCH=2`; candidate updates are suppressed starting
with the following PPO update epoch. This is a real train-path exercise of
the new P90 branch, not a synthetic trigger.

`P90_ONLY_TRIGGER_COUNT = 2`  
`MEAN_TRIGGER_COUNT = 0`  
`BOTH_TRIGGER_COUNT = 0`  
`CANDIDATE_KL_GATE_REAL_TRAIN_EXERCISED = YES`

The final aggregate KL for an episode still includes the update epoch that
caused the gate transition. Therefore the gate limits subsequent candidate
updates; it does not retroactively undo the drift already measured in that
update.

## Excursion comparison at historical epochs 107--108

The following compares the historical Stage2A values with the Stage1B
validation values. The joint pattern, rather than any single scalar, is the
criterion.

| epoch | run | scorer delta | candidate KL P90 | candidate KL max | argmax flip |
|---:|---|---:|---:|---:|---:|
| 107 | historical Stage2A | 0.582268 | 0.117297 | 0.160561 | 0.095000 |
| 107 | Stage1B | 0.353601 | 0.133092 | 0.178773 | 0.064583 |
| 108 | historical Stage2A | 0.571597 | 0.117045 | 0.291551 | 0.169167 |
| 108 | Stage1B | 0.389783 | 0.122787 | 0.158195 | 0.046250 |

Stage1B still encounters a P90 crossing at both epochs, but the scorer delta
and argmax flip are lower, the epoch-108 KL max is materially lower, and the
P90 gate stops the candidate branch after update epoch 2. The historical
joint excursion pattern is therefore not reproduced:

`EPOCH107_EXCURSION_REPRODUCED = NO`  
`EPOCH108_EXCURSION_REPRODUCED = NO`

## Cross-branch gradient clipping diagnostics

The run logs contain, for each PPO update epoch, the four requested arrays:

- `CACHE_ACTOR_GRAD_NORM_PRECLIP`
- `CACHE_ACTOR_GRAD_CLIP_COEFFICIENT`
- `UAV_ACTOR_GRAD_NORM_PRECLIP`
- `UAV_ACTOR_GRAD_CLIP_COEFFICIENT`

The current ten-episode artifact was produced before the final combined
post-clip field was added to the logging schema. Its combined diagnostic is
reconstructed read-only from those stored arrays using the specified
disjoint-group formula:

`sqrt(min(cache_preclip, 0.5)^2 + min(uav_preclip, 0.5)^2)`.

Across 40 update epochs:

| diagnostic | value |
|---|---:|
| `COMBINED_POSTCLIP_NORM_MEAN` | 0.146985 |
| `COMBINED_POSTCLIP_NORM_MAX` | 0.246994 |
| `COMBINED_POSTCLIP_GT_0P5_FRACTION` | 0.000000 |
| Cache clip coefficients below 1 | 0 / 40 |
| UAV clip coefficients below 1 | 0 / 40 |

All stored pre-clip group norms were below `0.5`, so clipping was not
activated in this short run. The branch-separated implementation is still
important because an inactive or large Cache group cannot scale the UAV group
and vice versa. Future runs now emit the exact per-update-epoch field
`COMBINED_ACTOR_GRAD_NORM_POSTCLIP` plus the three aggregate fields above;
this change is diagnostic-only and does not alter the optimizer or loss.

## Cache ratio non-regression

For epochs 100--109 of Stage1B:

| metric | mean | P50 | P90 | max |
|---|---:|---:|---:|---:|
| `CACHE_PER_HEAD_CLIP_FRACTION` | 0.000917 | 0.000000 | 0.000917 | 0.009167 |
| legacy `cache_clip_fraction` | 0.051500 | 0.042917 | 0.086083 | 0.155833 |
| `CACHE_PER_HEAD_OLD_NEW_KL_MEAN` | 0.000705 | 0.000677 | 0.001058 | 0.001779 |
| `CACHE_PER_HEAD_RATIO_MEAN` | 0.999905 | 0.999813 | 1.000406 | 1.001021 |

`PER_CACHE_HEAD` remains active and its clipping fraction remains far below
the legacy whole-joint diagnostic. No Cache ratio regression or numerical
anomaly was observed.

## Three-way paired rollout comparison

The three existing validation artifacts use the same train-only episode pool
for epochs 100--109. Stage1-only and Stage2A were read from existing logs;
neither baseline was rerun.

| epoch | Stage1-only return | Stage2A return | Stage1B return | Stage1B CacheChange | Stage1B UAVMove |
|---:|---:|---:|---:|---:|---:|
| 100 | -1342.55 | -1342.55 | -1342.55 | 0.5663 | 0.3667 |
| 101 | -541.21 | -578.51 | -578.51 | 0.5993 | 0.4800 |
| 102 | 1456.68 | 1173.01 | 1173.01 | 0.5785 | 0.4633 |
| 103 | 1275.55 | 1401.88 | 1401.88 | 0.5726 | 0.4783 |
| 104 | 88.74 | -10.74 | -10.74 | 0.5793 | 0.4667 |
| 105 | 434.48 | 950.99 | 950.99 | 0.5796 | 0.4700 |
| 106 | 1051.49 | 1250.27 | 1250.27 | 0.5963 | 0.4450 |
| 107 | 1859.78 | 800.34 | 800.34 | 0.5937 | 0.4367 |
| 108 | 1513.76 | 380.89 | 722.23 | 0.5593 | 0.4817 |
| 109 | 1012.98 | 52.52 | 404.12 | 0.5741 | 0.4367 |

Descriptive paired deltas, not significance claims:

| comparison | return mean | return median | positive / negative | CacheChange mean | UAVMove mean |
|---|---:|---:|---:|---:|---:|
| Stage1B minus Stage2A | 69.2931 | 0.0000 | 2 / 0 (8 zero) | 0.000037 | 0.003167 |
| Stage1B minus Stage1-only | -203.8676 | -68.3937 | 3 / 6 (1 zero) | -0.002074 | 0.006500 |

The ten paired episodes are insufficient to claim a performance improvement.
The rollout comparison is included to detect an obvious regression, while the
mechanism gate is decided primarily from the candidate drift and gate-path
evidence above.

## Final gate

`STAGE1B_CANDIDATE_STABILITY = PASS`

`STAGE1B_MECHANISM_GATE = PASS`

`CROSS_BRANCH_STABILITY_READY_TO_FREEZE = YES`

The pass is based on: no unintended scorer/encoder desynchronization, no
reproduction of the historical 107--108 joint excursion pattern, successful
real-path P90 gate activation, preserved Cache `PER_CACHE_HEAD` behavior, and
no numerical anomaly. This is a short validation gate, not evidence of final
long-horizon performance.

`TRAINING_EXECUTED = YES` (10 train-only validation episodes only)  
`FORMAL_RUN_MODIFIED = NO`  
`STAGE2B_EXECUTED = NO`  
`COMMIT = NO`  
`PUSH = NO`
