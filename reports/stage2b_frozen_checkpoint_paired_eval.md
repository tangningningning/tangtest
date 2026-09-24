# Stage2B Frozen Checkpoint Paired Evaluation

## Required result fields

```text
EVALUATION_EPISODES = 10
TRAINING_EXECUTED = NO
OPTIMIZER_STEP_CALLS = 0
BACKWARD_CALLS = 0

STAGE1B_CHECKPOINT = runs/stage1b_cross_branch_short_gate/latest_checkpoint.pt
STAGE2B_CHECKPOINT = runs/stage2b_cache_trigger_tdq_short_gate/latest_checkpoint.pt
STAGE1B_CHECKPOINT_SHA = d5b9a4e1d0ca9920ec2c3a95392d3483420e1f8b9dc2cc186a786f67dbbe71f3
STAGE2B_CHECKPOINT_SHA = e65c87ab558fdcab1778129fe65eda88595dbda9f13ace0844193c5e9406f43d

REALIZATIONS_USED = 20110–20119
EVAL_ACTION_MODE = DETERMINISTIC
PAIRED_RNG_CONTROL = NOT_NEEDED
TDQ_COUNTERFACTUAL_USED_DURING_FROZEN_INFERENCE = NO

STAGE1B_RETURN_MEAN = 2152.977134
STAGE2B_RETURN_MEAN = 2776.314599

PAIRED_RETURN_DELTA_MEAN = 623.337465
PAIRED_RETURN_DELTA_MEDIAN = 498.977594
PAIRED_RETURN_DELTA_MIN = -347.521640
PAIRED_RETURN_DELTA_MAX = 2181.062404
PAIRED_RETURN_POSITIVE_COUNT = 9
PAIRED_RETURN_NEGATIVE_COUNT = 1
PAIRED_RETURN_ZERO_COUNT = 0

STAGE1B_CACHE_CHANGE_MEAN = 0.594778
STAGE2B_CACHE_CHANGE_MEAN = 1.000000
PAIRED_CACHE_CHANGE_DELTA_MEAN = 0.405222
PAIRED_CACHE_CHANGE_DELTA_MEDIAN = 0.408889

STAGE1B_UAV_MOVE_MEAN = 0.089167
STAGE2B_UAV_MOVE_MEAN = 0.078000
PAIRED_UAV_MOVE_DELTA_MEAN = -0.011167

CACHE_CHANGE_HAS_COMPENSATING_UTILITY_GAIN = YES

FROZEN_POLICY_STAGE2B_EFFECT = IMPROVED
STAGE2B_LONG_TRAIN_GO = YES

STAGE1B_EVAL_RUNTIME_TOTAL = 786.994 seconds
STAGE2B_EVAL_RUNTIME_TOTAL = 740.473 seconds

FORMAL_RUN_MODIFIED = NO
CHECKPOINT_MODIFIED = NO
COMMIT = NO
PUSH = NO
```

`STAGE2B_LONG_TRAIN_GO = YES` is a descriptive frozen-evaluation go signal,
not a claim of statistical significance. It is conditioned on the observed
CacheChange saturation being explicitly monitored in any future work.

## Evaluation protocol

The project’s existing `ppo_simple.evaluator` was used with each checkpoint’s
own original run configuration:

| policy | config | checkpoint epoch | action mode | repeats |
|---|---|---:|---|---:|
| Stage1B | `runs/stage1b_cross_branch_short_gate/run_config.toml` | 109 | deterministic | 1 |
| Stage2B | `runs/stage2b_cache_trigger_tdq_short_gate/run_config.toml` | 109 | deterministic | 1 |

The evaluation realization pool was the existing train-only
`environment_data/proposed_joint_training`, using exactly
`episode_20110` through `episode_20119`. These realizations are outside the
two short-run epochs 100–109 and no new data was generated.

For every realization the evaluator created an independent environment
context, copied the configured initial Cache/UAV state, set `policy.eval()`,
and used `torch.inference_mode()`. The evaluator’s post-rollout checks passed:
policy parameters, PopArt statistics, and optimizer-step counters were
unchanged. No trainer update, backward call, optimizer step, checkpoint write,
or TD-Q counterfactual branch was executed. The evaluation used the existing
deterministic action-selection path, so paired RNG reset was not needed.

## Per-realization paired results

`Return` is total system utility. `Task utility` is the evaluator’s existing
`total_q`. Delay and energy are the evaluator’s existing per-task means;
CacheChange and UAVMove are the existing actual rates. Costs are the existing
episode totals.

| episode | Stage1B Return | Stage2B Return | Δ Return | Stage1B CacheChange | Stage2B CacheChange | Δ CacheChange | Stage1B UAVMove | Stage2B UAVMove | Δ UAVMove |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20110 | 914.249 | 3095.311 | 2181.062 | 0.5570 | 1.0000 | 0.4430 | 0.3150 | 0.2383 | -0.0767 |
| 20111 | 2390.818 | 2929.165 | 538.347 | 0.5870 | 1.0000 | 0.4130 | 0.1050 | 0.1167 | 0.0117 |
| 20112 | 2424.233 | 2864.473 | 440.240 | 0.5659 | 1.0000 | 0.4341 | 0.0717 | 0.1117 | 0.0400 |
| 20113 | 2570.562 | 3150.251 | 579.689 | 0.6337 | 1.0000 | 0.3663 | 0.0033 | 0.0050 | 0.0017 |
| 20114 | 1731.676 | 2497.405 | 765.729 | 0.6259 | 1.0000 | 0.3741 | 0.0683 | 0.0050 | -0.0633 |
| 20115 | 2054.368 | 2843.986 | 789.618 | 0.5985 | 1.0000 | 0.4015 | 0.0050 | 0.0450 | 0.0400 |
| 20116 | 2102.177 | 1754.655 | -347.522 | 0.6196 | 1.0000 | 0.3804 | 0.0250 | 0.0817 | 0.0567 |
| 20117 | 2663.217 | 3066.590 | 403.373 | 0.5778 | 1.0000 | 0.4222 | 0.2250 | 0.1333 | -0.0917 |
| 20118 | 2182.066 | 2605.296 | 423.230 | 0.5907 | 1.0000 | 0.4093 | 0.0050 | 0.0050 | 0.0000 |
| 20119 | 2496.405 | 2956.013 | 459.608 | 0.5915 | 1.0000 | 0.4085 | 0.0683 | 0.0383 | -0.0300 |

Only episode 20116 has a negative Stage2B Return delta. The table is a
descriptive paired comparison over ten train-only realizations; no
significance test is claimed.

## Native utility and cost metrics

| metric | Stage1B mean | Stage2B mean | paired delta mean |
|---|---:|---:|---:|
| total task utility (`total_q`) | 2275.727 | 2976.397 | +700.670 |
| total system Return | 2152.977 | 2776.315 | +623.337 |
| total Cache cost | 122.597 | 199.948 | +77.352 |
| total UAV movement cost | 0.153010 | 0.133848 | -0.019162 |
| mean task delay (s) | 15.589 | 10.877 | -4.712 |
| mean vehicle energy (J) | 1.169 | 0.605 | -0.564 |

Stage2B’s mean task-utility gain (+700.670) is larger than its mean Cache-cost
increase (+77.352), leaving a positive mean Return delta (+623.337). Delay and
energy also decrease on average. Therefore the higher CacheChange has a
measured compensating utility gain in this frozen paired sample:

```text
CACHE_CHANGE_HAS_COMPENSATING_UTILITY_GAIN = YES
```

This does not mean higher CacheChange is intrinsically better. Stage2B reaches
`CacheChange = 1.0` on all ten realizations, so CacheChange saturation is a
real behavioral change and should remain a monitoring item for any longer
run.

The existing evaluator does not provide a reliable direct Cache-hit or
executable-hit metric, nor a separately defined offloading-success metric:

```text
CACHE_HIT_METRIC = NOT_AVAILABLE
EXECUTABLE_HIT_METRIC = NOT_AVAILABLE
OFFLOADING_SUCCESS_METRIC = NOT_AVAILABLE
```

It does provide execution ratios. Their ten-realization means are:

| execution class ratio | Stage1B | Stage2B | delta |
|---|---:|---:|---:|
| CLOUD | 0.183139 | 0.402403 | +0.219264 |
| LOCAL | 0.067588 | 0.047401 | -0.020187 |
| RSU | 0.193767 | 0.144459 | -0.049308 |
| SV | 0.243428 | 0.141891 | -0.101537 |
| UAV | 0.312078 | 0.263846 | -0.048232 |

The execution-ratio shift is reported descriptively only; it is not relabeled
as a hit or success metric.

## UAV non-regression

The deterministic frozen rollout gives:

```text
STAGE1B_UAV_MOVE_MEAN = 0.089167
STAGE2B_UAV_MOVE_MEAN = 0.078000
PAIRED_UAV_MOVE_DELTA_MEAN = -0.011167
```

Mean UAV movement cost also decreases by `0.019162`. The per-realization UAV
move delta is mixed (5 positive, 4 negative, 1 zero), so this is a
non-regression signal at the aggregate level, not a claim that every
realization improves.

## Runtime and artifact safety

Evaluation output was written only to the new non-formal directory
`runs/stage2b_frozen_checkpoint_paired_eval_20260923`. The two totals below
are filesystem start-to-summary wall-time measurements for the existing
evaluator, including initialization:

```text
STAGE1B_EVAL_RUNTIME_TOTAL = 786.994 seconds
STAGE2B_EVAL_RUNTIME_TOTAL = 740.473 seconds
STAGE1B_RUNTIME_PER_EPISODE_MEAN = 78.699 seconds
STAGE2B_RUNTIME_PER_EPISODE_MEAN = 74.047 seconds
```

The runtime excludes TD-Q training-credit evaluation because no such branch
was invoked during frozen inference:

```text
TDQ_COUNTERFACTUAL_USED_DURING_FROZEN_INFERENCE = NO
```

Both source checkpoint hashes were re-read after evaluation and remain the
values recorded above. Formal runs, source checkpoints, and original run
configs were not modified.
