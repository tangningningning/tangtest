# Joint V9 epoch-222 stochastic development Return tail-risk analysis

## Scope and source

- Checkpoint: `runs/ppo_v9_joint_cache_uav_stage1_ep50_20001_20050/checkpoint_ep222_frozen.pt`
- Existing evaluation source: `runs/ppo_v9_dev_ep222_stochastic5_30001_30020`
- Environment seeds: `30001..30020`
- Repeats per environment: `5`
- Total evaluations read: `100`
- Statistical unit: the mean of five stochastic repeats within one environment realization
- Percentiles: NumPy `np.percentile(..., method="linear")`

No evaluation was rerun and no final-test seed was used.

## Return distribution over 20 environment means

| metric | value |
| --- | ---: |
| ENV_MEAN_RETURN | 2913.0241409452265 |
| ENV_STD_RETURN (sample, ddof=1) | 281.6408155089820 |
| ENV_SE_RETURN | 62.97680087165607 |
| MIN_RETURN | 2407.037381544709 |
| P10_RETURN | 2599.978899698034 |
| P25_RETURN | 2737.116906195134 |
| MEDIAN_RETURN | 2890.450039891899 |
| P75_RETURN | 3086.9686604455114 |
| P90_RETURN | 3292.220201367736 |
| MAX_RETURN | 3395.915612256527 |

## Lower-tail CVaR

With 20 environment means, CVaR10 is the mean of the two lowest environment means and CVaR20 is the mean of the four lowest.

| metric | value |
| --- | ---: |
| CVAR10_RETURN | 2460.878350071609 |
| CVAR20_RETURN | 2541.184104678780 |
| BOTTOM10_GAP_TO_MEAN | 452.14579087361744 |
| BOTTOM20_GAP_TO_MEAN | 371.84003626644653 |
| BOTTOM10_RELATIVE_GAP | 0.15521525706508696 |
| BOTTOM20_RELATIVE_GAP | 0.12764742696083212 |

The lowest environment means are seeds `30002`, `30020`, `30006`, and `30010`.

## Five lowest environment means

All metrics below are means across the five stochastic repeats for that environment. Execution ratios are also averaged across the five repeats.

| seed | mean Return | repeat std | repeat min | repeat max | Q | cache cost | movement cost | cache change | UAV move | delay | energy | tasks | LOCAL | SV | RSU | UAV | CLOUD |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30002 | 2407.037382 | 388.707701 | 2163.454083 | 3076.306876 | 2465.928452 | 50.187429 | 8.703640 | 0.329259 | 0.348333 | 16.687334 | 0.654093 | 4196 | 0.020686 | 0.197712 | 0.005386 | 0.469352 | 0.306864 |
| 30020 | 2514.719319 | 363.905604 | 2145.269592 | 3103.406223 | 2574.392393 | 51.607434 | 8.065640 | 0.340074 | 0.324667 | 15.466166 | 0.637093 | 4194 | 0.014020 | 0.157177 | 0.007058 | 0.424988 | 0.396757 |
| 30006 | 2609.452186 | 72.477520 | 2516.708323 | 2694.230564 | 2670.368984 | 52.202597 | 8.714200 | 0.340148 | 0.349333 | 14.889204 | 0.621688 | 4258 | 0.061249 | 0.186613 | 0.005871 | 0.417755 | 0.328511 |
| 30010 | 2633.527532 | 236.932085 | 2385.678819 | 2992.910483 | 2692.718070 | 50.520557 | 8.669980 | 0.330963 | 0.339333 | 13.551403 | 0.704754 | 4191 | 0.014984 | 0.087044 | 0.005727 | 0.373562 | 0.518683 |
| 30014 | 2724.022986 | 64.691153 | 2656.425255 | 2818.217417 | 2786.686437 | 53.670950 | 8.992500 | 0.347630 | 0.353000 | 14.331059 | 0.507975 | 4268 | 0.031256 | 0.255530 | 0.008529 | 0.453421 | 0.251265 |

## Between-environment versus within-environment variance

| metric | value |
| --- | ---: |
| BETWEEN_ENV_VARIANCE (sample, ddof=1) | 79321.54896056441 |
| BETWEEN_ENV_STD | 281.6408155089820 |
| WITHIN_ENV_VARIANCE_MEAN (sample, ddof=1) | 33937.1764380595 |
| MEAN_WITHIN_ENV_STD | 151.25655834227177 |
| VARIANCE_RATIO | 2.3373054946199865 |

Cross-realization variance is larger than the mean within-realization policy-sampling variance by a factor of 2.3373. Within-environment sampling variance is nevertheless material for seeds `30002` and `30020`.

## Repeat stability

| metric | value |
| --- | ---: |
| MEAN_REPEAT_STD | 151.25655834227177 |
| MEDIAN_REPEAT_STD | 82.8351814948575 |
| MAX_REPEAT_STD | 388.707701280107 |
| MEAN_REPEAT_RANGE | 375.9042746072635 |
| MAX_REPEAT_RANGE | 958.136630654335 |

The five most sampling-unstable environments, ranked by within-environment Return sample standard deviation, are:

| seed | mean Return | repeat std | repeat range | repeat CV | repeat min | repeat max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30002 | 2407.037382 | 388.707701 | 912.852793 | 0.161488 | 2163.454083 | 3076.306876 |
| 30020 | 2514.719319 | 363.905604 | 958.136631 | 0.144710 | 2145.269592 | 3103.406223 |
| 30012 | 2790.970454 | 281.131614 | 702.160700 | 0.100729 | 2544.905555 | 3247.066255 |
| 30010 | 2633.527532 | 236.932085 | 607.231665 | 0.089968 | 2385.678819 | 2992.910483 |
| 30019 | 2741.481546 | 221.465428 | 544.035101 | 0.080783 | 2370.604942 | 2914.640043 |

## Descriptive Pearson correlations over 20 environment means

These are descriptive associations only and do not establish causality.

| pair | Pearson correlation |
| --- | ---: |
| Return vs UAV movement rate | -0.14282261052104192 |
| Return vs cache change rate | 0.08576107841024197 |
| Return vs cache cost | 0.17913379852042843 |
| Return vs movement cost | 0.0007447643506942466 |
| Return vs mean task delay | -0.9837601928491913 |
| Return vs mean vehicle energy | -0.9284157385660116 |

## Statistical interpretation

The lower Return tail is primarily associated, descriptively, with environment-level difficulty reflected in higher task delay and vehicle energy. Sampling variance is a secondary instability source and is especially visible in seeds `30002` and `30020`; seed `30006` has a low environment mean with comparatively small repeat variance.

The five lowest environments do not show an extreme movement-cost or cache-cost pattern, and Return has weak descriptive association with UAV movement rate, cache-change rate, cache cost, and movement cost. Their execution-ratio profiles vary, but the strongest cross-environment associations in this analysis are with delay and energy.

This is a frozen read-only analysis. No CVaR reward, KL constraint, risk-aware PPO, V10 retraining, or V9 protocol change was performed.
