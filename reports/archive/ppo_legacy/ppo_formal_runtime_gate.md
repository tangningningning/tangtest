# `ppo_simple_v2` formal runtime gate

## Final decision

The formal policy path was exercised with the current production code and normal Bernoulli sampling. Eight consecutive training origins (`0..7`) completed successfully, but the collection runtime is still at minute scale. The 16-origin extension was therefore not run, and the first complete 240-origin PPO episode is not approved.

The dominant wall-clock stage is configuration-time surrogate evaluation. The final current-slot Matching/KKT call is negligible by comparison; the expensive work is the worker-side prediction Matching/KKT evaluation performed while searching Cache/UAV configurations.

## Measurement protocol

- Entry path: `ppo_simple.trainer.PPOV2Trainer.collect_chunk()` with method `ppo_simple_v2`.
- Conda environment: `adaptive_cache_mec`.
- Seed: `4401`; `random`, NumPy, and PyTorch were seeded before trainer construction.
- Policy: current 11-dimensional Bernoulli policy, normal sampling, current observation and action mask.
- Environment: current train origins beginning at origin 0, with the state returned by each origin passed to the next origin.
- Prediction: current K=100 path and exact production upper-bound pruning; no reduced-K path.
- PPO updates: `0`; no logits or actions were overridden.
- The temporary runner supplied only the existing optional performance/profile objects to `execute_slot`; it did not alter production semantics.

The first eight origins took 1,885.385 s at chunk level. The sum of the eight `execute_slot` wall-clock measurements was 1,883.518 s; the difference is chunk/setup overhead. Because the first block had mean 235.440 s/origin and maximum 358.750 s/origin, it failed the runtime gate and was not extended to origins 8..15.

## Per-origin results

In the action column, `C` is the nine Cache bits and `U` is `(UAV_0,UAV_1)`. The value before and after `->` is raw sampled action and effective request respectively. They were identical for all eight origins after the current mask/action conversion.

| origin | raw -> effective (`C|U`) | class | Cache/UAV candidate searches | prediction Matching / full evaluator / UB pruned | surrogate KKT req / miss / solve | final KKT req / miss / solve | execute / configuration / surrogate (s) | Q / reward=G | finite |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 0 | `001000000|01 -> 001000000|01` | cache+uav | 8 / 78 | 9,200 / 2,191,305 / 1,536,394 | 4,163,246 / 1,209,335 / 1,209,335 | 317 / 98 / 98 | 206.747348 / 206.675480 / 205.267511 | 10.021667286 / 10.015947286 | YES |
| 1 | `011110110|00 -> 011110110|00` | cache-only-like | 65 / 0 | 13,400 / 2,526,411 / 2,565,756 | 4,651,199 / 1,356,995 / 1,356,995 | 434 / 124 / 124 | 206.611652 / 206.521971 / 205.649702 | 10.095648546 / 6.431686015 | YES |
| 2 | `011010010|00 -> 011010010|00` | cache-only-like | 43 / 0 | 4,300 / 771,081 / 898,448 | 1,394,104 / 400,532 / 400,532 | 205 / 61 / 61 | 67.052450 / 67.016192 / 66.296085 | 9.286905849 / 9.286905849 | YES |
| 3 | `010100100|10 -> 010100100|10` | cache+uav | 31 / 76 | 13,100 / 2,441,097 / 2,744,304 | 4,448,218 / 1,274,312 / 1,274,312 | 502 / 141 / 141 | 186.149620 / 186.027092 / 185.320844 | 11.021473221 / 11.015753221 | YES |
| 4 | `000001001|10 -> 000001001|10` | cache+uav | 43 / 76 | 18,900 / 3,887,933 / 4,637,161 | 7,037,756 / 1,992,343 / 1,992,343 | 131 / 42 / 42 | 286.337563 / 286.312186 / 285.525127 | 8.584233986 / 8.478513986 | YES |
| 5 | `101001101|11 -> 101001101|11` | cache+uav | 73 / 154 | 35,200 / 6,911,493 / 8,184,222 | 12,472,563 / 3,608,247 / 3,608,247 | 623 / 194 / 194 | 358.750030 / 358.631355 / 357.927620 | 11.641705375 / 11.630265375 | YES |
| 6 | `111101001|01 -> 111101001|01` | cache+uav | 76 / 78 | 22,200 / 4,424,343 / 5,098,924 | 7,981,667 / 2,318,052 / 2,318,052 | 240 / 76 / 76 | 274.157323 / 274.108971 / 273.375744 | 11.508674220 / 11.502954220 | YES |
| 7 | `100000100|11 -> 100000100|11` | cache+uav | 22 / 154 | 25,400 / 4,402,841 / 5,604,743 | 7,857,087 / 2,303,199 / 2,303,199 | 274 / 96 / 96 | 297.712018 / 297.655515 / 296.978062 | 9.832771790 / 9.821331790 | YES |

For each row, `reward = Q - cache_cost - movement_cost = G`. The corresponding cache/movement costs were respectively:

| origin | cache cost | movement cost | actual Cache updates | actual UAV movements |
|---:|---:|---:|---:|---:|
| 0 | 0.000000000 | 0.005720000 | 0 | 1 |
| 1 | 3.663962530 | 0.000000000 | 3 | 0 |
| 2 | 0.000000000 | 0.000000000 | 0 | 0 |
| 3 | 0.000000000 | 0.005720000 | 0 | 1 |
| 4 | 0.100000000 | 0.005720000 | 1 | 1 |
| 5 | 0.000000000 | 0.011440000 | 0 | 2 |
| 6 | 0.000000000 | 0.005720000 | 0 | 1 |
| 7 | 0.000000000 | 0.011440000 | 0 | 2 |

## Runtime statistics

### Eight-origin block

| statistic | value |
|---|---:|
| origins completed | 8 (`0..7`) |
| total `execute_slot` time | 1,883.518004 s |
| chunk wall-clock time | 1,885.385041 s |
| mean | 235.439751 s |
| median | 240.452336 s |
| minimum | 67.052450 s |
| maximum | 358.750030 s |
| p90 | 316.023422 s |

### Action classes

| class | origins | mean (s) | min (s) | max (s) | total (s) | Cache requests | UAV requests | actual Cache updates | actual UAV movements |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cache-only-like | 2 | 136.832051 | 67.052450 | 206.611652 | 273.664102 | 10 | 0 | 3 | 0 |
| UAV-only | 0 | not observed | not observed | not observed | 0 | 0 | 0 | 0 | 0 |
| cache+uav | 6 | 268.308984 | 186.149620 | 358.750030 | 1,609.853902 | 19 | 8 | 1 | 8 |
| neutral/low-work | 0 | not observed | not observed | not observed | 0 | 0 | 0 | 0 | 0 |

The six UAV-triggered origins were all `cache+uav`; no UAV-only sample occurred in this fixed-seed block. Every one of those six origins was between 186.150 s and 358.750 s. Cache-only-like actions also reached 206.612 s, so the bottleneck is not exclusively UAV movement; it is the amount of configuration surrogate search induced by the sampled request vector. UAV movement itself is not the wall-clock dominant calculation.

### Aggregate work counters

Across the eight origins:

- Cache requests: 29; UAV requests: 8.
- Cache candidate searches: 361; UAV candidate searches: 616.
- Prediction Matching calls: 141,700.
- Full candidate evaluator calls: 27,556,504.
- Upper-bound pruned candidates: 31,269,952.
- Surrogate KKT requests: 50,005,840; misses/solves: 14,463,015.
- Actual current-slot Matching KKT requests: 2,726; misses/solves: 832.
- Actual Cache updates: 4; actual UAV movements: 8.

The profile reported 13,006.088 s of cumulative worker-side surrogate Matching time. This is a sum across parallel workers and is not an additional wall-clock duration. The wall-clock decomposition was:

- configuration: 1,882.949 s, 99.970% of summed `execute_slot` time;
- surrogate utility evaluation: 1,876.341 s, 99.619%;
- final current-slot Matching/KKT: 0.559 s, 0.030%;
- policy sampling: 0.167 s total.

All eight origins used the current parallel prediction mode with 8 workers. No worker-count or scheduling change was made.

## 240-origin extrapolation

This is a collection-runtime estimate, not a completed episode and does not include PPO update/checkpoint overhead:

- expected, using the eight-origin mean: `235.439751 * 240 = 56,505.540 s` (about **15.70 h**);
- conservative, using the observed p90 sustained over 240 origins: `75,845.621 s` (about **21.07 h**);
- observed-maximum stress estimate: `86,100.007 s` (about **23.92 h**).

The extrapolation uses the continuous normal-policy block, not the earlier fixed-action measurements.

## Correctness checks

All eight completed transitions passed:

- no exception and `step_feasible=True`;
- observation, raw action, action mask, log-probability, value, reward, Q, G, costs, and next states finite;
- valid Cache/UAV request-to-transition contract;
- finite and valid next Cache/UAV state;
- reward and actual `G(t)` aligned exactly in the recorded values;
- no PPO update was executed.

## Gate result

```ini
FORMAL_POLICY_SAMPLING = YES
CONTIGUOUS_ORIGINS_COMPLETED = 8
CACHE_TRIGGERED_ORIGINS = 8
UAV_TRIGGERED_ORIGINS = 6
CACHE_UAV_TRIGGERED_ORIGINS = 6

MEAN_ORIGIN_RUNTIME = 235.439751 s
MEDIAN_ORIGIN_RUNTIME = 240.452336 s
MAX_ORIGIN_RUNTIME = 358.750030 s

ESTIMATED_240_ORIGIN_EPISODE_RUNTIME = expected 56505.540 s (~15.70 h); conservative 75845.621 s (~21.07 h)

NUMERICAL_CHECKS_PASS = YES
ENVIRONMENT_CHECKS_PASS = YES

READY_FOR_FORMAL_PPO_EPISODE = NO
```

Reason for `NO`: normal formal actions still produce repeated multi-minute origins, including Cache-only-like and Cache+UAV actions, and the collection-only 240-origin estimate is many hours. The 16-origin block and full episode were intentionally not run.

## Workspace hygiene

The temporary runner, runtime JSON, and Python cache directories created/used for this gate were removed. No production source file was changed by this gate. `git diff --check` completed with exit code 0; it emitted only Git line-ending normalization warnings and no whitespace errors. Existing working-tree changes and earlier reports were preserved.
