# Cache Trigger State Discrimination Root Cause Diagnosis

## Scope

This diagnosis is restricted to why the Cache trigger emits almost constant KEEP/UPDATE probabilities while symmetric expected-Q remains state dependent. It uses each head's P90 onset checkpoint and the existing Stage-B delta-Q samples. The two D-class controls are 5e-4 head 2 and 1e-3 head 1.

The 576 production states were replayed with the original RNG rule solely to recover the saved `cache_contexts`. Replay probabilities match the temporal JSON with maximum absolute error `1.192e-07`. Both actions were legal in all 576 states. No new exact-Q branch, training, backward pass, optimizer step, validation, or test was executed.

## Required conclusions

`STATE_DEPENDENT_EXPECTED_Q_PRESENT = YES`

`TRIGGER_STATE_DISCRIMINATION_WEAK = YES`

`BIAS_DOMINATED_TRIGGER = NO`

`CACHE_ENCODER_STATE_COLLAPSE = NO`

`OPERATION_HEAD_STATE_INSENSITIVITY = YES`

## Aggregate evidence

| Metric | Mixed/error heads | D-class controls | Mixed/control ratio |
|---|---:|---:|---:|
| Head count | 16 | 2 | NA |
| Median STD_P_UPDATE | 0.002662 | 0.0003816 | 6.976 |
| Median STD_DELTA_Q | 4.837 | 24.49 | 0.1975 |
| Median Spearman(logit, ΔQ) | 0.04435 | 0.06507 | NA |
| Median ROC AUC | 0.4874 | 0.7742 | NA |
| Median sign accuracy | 0.5156 | 0.9375 | NA |
| Median encoder feature STD mean | 0.04418 | 0.09381 | 0.471 |
| Median encoder effective rank | 6.531 | 2.589 | 2.523 |
| Median STATE_TERM_STD | 0.08416 | 0.3548 | 0.2372 |
| Median operation weight-gap norm | 0.8761 | 1.214 | 0.7219 |

Run-specific mixed/control ratios:

| Run | Encoder feature STD | Encoder effective rank | STATE_TERM_STD | Weight-gap norm |
|---|---:|---:|---:|---:|
| 5e-4 | 0.5999 | 1.639 | 0.1465 | 0.6662 |
| 1e-3 | 0.4098 | 6.068 | 0.4391 | 0.8012 |

All 16 mixed/error heads contain both positive and negative ΔQ samples. Their minimum per-head `STD_DELTA_Q` is `1.793`. Meanwhile `15/16` have `STD_P_UPDATE <= 0.05`.

The mixed/error median absolute bias gap is `0.07191`, while the median absolute STATE_TERM mean is `3.189`. The median bias/state-STD ratio is `0.7211` and its maximum is `1.685`; this does not meet the 5x bias-dominance rule.

Encoder variation is lower in amplitude, but its median effective rank is higher than the controls. Therefore the encoder representation is not classified as collapsed. The operation projection compresses this state variation: mixed/error median STATE_TERM_STD is only `0.237x` the controls, while its median Spearman `0.044` and AUC `0.487` are near random. This supports operation-head state insensitivity as the root cause in this diagnostic scope.

## Per-head discrimination metrics

| Run | Head | Node | Role | Epoch | Control | STD_P_UPDATE | STD_DELTA_Q | Spearman | ROC AUC | Sign accuracy | Bias/state STD | Encoder effective rank | Weight-gap norm |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5e-4 | 0 | 46 | SV | 20 |  | 0.0001427 | 1.793 | 0.004399 | 0.4332 | 0.5938 | 1.506 | 5.772 | 1.059 |
| 5e-4 | 1 | 10 | SV | 50 |  | 0.002072 | 8.187 | 0.1213 | 0.5065 | 0.6562 | 1.685 | 1.858 | 0.8898 |
| 5e-4 | 2 | 0 | SV | 70 | D | 0.000462 | 13.72 | -0.09201 | 0.5484 | 0.9688 | 0.03389 | 3.595 | 1.315 |
| 5e-4 | 3 | 23 | SV | 50 |  | 0.02446 | 10.32 | 0.1415 | 0.6429 | 0.4375 | 0.1295 | 2.005 | 1.04 |
| 5e-4 | 4 | 59 | SV | 30 |  | 0.003817 | 4.358 | 0.05609 | 0.5142 | 0.5938 | 1.164 | 7.076 | 0.7783 |
| 5e-4 | 5 | RSU_0 | RSU | 10 |  | 0.001682 | 6.12 | 0.02419 | 0.4625 | 0.375 | 0.5293 | 10.66 | 1.014 |
| 5e-4 | 6 | RSU_1 | RSU | 10 |  | 0.00348 | 6.68 | 0.3578 | 0.6234 | 0.3438 | 1.651 | 10.76 | 0.8084 |
| 5e-4 | 7 | UAV_0 | UAV | 20 |  | 0.004775 | 2.303 | -0.08138 | 0.4683 | 0.5625 | 0.597 | 6.011 | 0.8623 |
| 5e-4 | 8 | UAV_1 | UAV | 20 |  | 0.001194 | 2.55 | -0.01356 | 0.325 | 0.625 | 1.008 | 5.108 | 0.8291 |
| 1e-3 | 0 | 46 | SV | 20 |  | 0.002515 | 2.778 | -0.08174 | 0.4235 | 0.4688 | 1.177 | 7.051 | 0.8584 |
| 1e-3 | 1 | 10 | SV | 40 | D | 0.0003012 | 35.26 | 0.2221 | 1 | 0.9062 | 0.3599 | 1.583 | 1.112 |
| 1e-3 | 2 | 0 | SV | 10 |  | 0.003122 | 3.603 | -0.2001 | 0.373 | 0.5625 | 0.8299 | 12.52 | 0.8033 |
| 1e-3 | 3 | 23 | SV | 50 |  | 0.0009651 | 12.34 | 0.5154 | 0.9458 | 0.625 | 0.07746 | 2.289 | 1.286 |
| 1e-3 | 4 | 59 | SV | 10 |  | 0.0008153 | 3.705 | 0.03262 | 0.4682 | 0.3125 | 0.8484 | 12.63 | 0.9722 |
| 1e-3 | 5 | RSU_0 | RSU | 10 |  | 0.002474 | 5.544 | -0.2401 | 0.3889 | 0.4375 | 0.6124 | 12.66 | 0.9235 |
| 1e-3 | 6 | RSU_1 | RSU | 40 |  | 0.1057 | 8.897 | 0.1107 | 0.5385 | 0.4062 | 0.1242 | 1.814 | 0.9804 |
| 1e-3 | 7 | UAV_0 | UAV | 50 |  | 0.008781 | 3.335 | 0.1133 | 0.5273 | 0.5 | 0.5059 | 2.298 | 0.8108 |
| 1e-3 | 8 | UAV_1 | UAV | 10 |  | 0.002809 | 5.315 | 0.1499 | 0.6353 | 0.5312 | 0.3904 | 12.17 | 0.8285 |

## Linear trigger decomposition

The production trigger layer is linear and has a bias. For every state, `LOGIT_GAP = STATE_TERM + BIAS_GAP` with maximum decomposition errors recorded in JSON.

| Run | Head | Bias gap | State term mean | State term STD | Logit-gap mean | Logit-gap STD | Bias/state STD | Encoder feature STD mean | Encoder feature STD median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5e-4 | 0 | 0.08555 | 5.888 | 0.05682 | 5.974 | 0.05682 | 1.506 | 0.03009 | 0.02278 |
| 5e-4 | 1 | 0.1294 | 3.42 | 0.07681 | 3.55 | 0.07681 | 1.685 | 0.1108 | 0.09864 |
| 5e-4 | 2 | 0.01708 | -7.256 | 0.504 | -7.238 | 0.504 | 0.03389 | 0.06578 | 0.05146 |
| 5e-4 | 3 | 0.07326 | 3.211 | 0.5659 | 3.285 | 0.5659 | 0.1295 | 0.1182 | 0.1075 |
| 5e-4 | 4 | 0.04723 | 2.098 | 0.04058 | 2.145 | 0.04058 | 1.164 | 0.03379 | 0.02755 |
| 5e-4 | 5 | 0.0535 | 4.007 | 0.1011 | 4.06 | 0.1011 | 0.5293 | 0.04538 | 0.04399 |
| 5e-4 | 6 | 0.117 | 2.783 | 0.07089 | 2.9 | 0.07089 | 1.651 | 0.04514 | 0.04467 |
| 5e-4 | 7 | 0.07055 | 3.032 | 0.1182 | 3.103 | 0.1182 | 0.597 | 0.02995 | 0.02118 |
| 5e-4 | 8 | 0.0318 | 3.166 | 0.03153 | 3.198 | 0.03153 | 1.008 | 0.03096 | 0.02169 |
| 1e-3 | 0 | 0.08034 | 3.146 | 0.06828 | 3.226 | 0.06828 | 1.177 | 0.05663 | 0.05382 |
| 1e-3 | 1 | 0.074 | -6.653 | 0.2056 | -6.579 | 0.2056 | 0.3599 | 0.1218 | 0.09452 |
| 1e-3 | 2 | 0.0412 | 2.583 | 0.04965 | 2.624 | 0.04965 | 0.8299 | 0.04309 | 0.04304 |
| 1e-3 | 3 | 0.0437 | -6.789 | 0.5641 | -6.746 | 0.5641 | 0.07746 | 0.06499 | 0.03418 |
| 1e-3 | 4 | 0.0745 | 4.585 | 0.08781 | 4.659 | 0.08781 | 0.8484 | 0.04285 | 0.04386 |
| 1e-3 | 5 | 0.05682 | 3.539 | 0.09279 | 3.596 | 0.09279 | 0.6124 | 0.04296 | 0.04403 |
| 1e-3 | 6 | 0.1331 | 2.463 | 1.072 | 2.596 | 1.072 | 0.1242 | 0.1113 | 0.08715 |
| 1e-3 | 7 | 0.09665 | 2.869 | 0.1911 | 2.965 | 0.1911 | 0.5059 | 0.06274 | 0.03617 |
| 1e-3 | 8 | 0.03144 | 3.251 | 0.08052 | 3.282 | 0.08052 | 0.3904 | 0.04323 | 0.04353 |

## ΔQ strata

Within each head, near-zero means `|ΔQ| <= 0.25 × STD_DELTA_Q`; weak means between `0.25×` and `1×`; strong means at least `1×`. Each cell is `count: mean / median p_update`.

| Run | Head | Strong KEEP | Weak KEEP | Near zero | Weak UPDATE | Strong UPDATE |
|---|---:|---:|---:|---:|---:|---:|
| 5e-4 | 0 | 2: 0.997 / 0.997 | 7: 0.997 / 0.997 | 9: 0.997 / 0.997 | 9: 0.997 / 0.997 | 5: 0.997 / 0.998 |
| 5e-4 | 1 | 3: 0.972 / 0.972 | 4: 0.972 / 0.971 | 8: 0.972 / 0.972 | 6: 0.971 / 0.971 | 11: 0.973 / 0.973 |
| 5e-4 | 2 | 26: 0.000851 / 0.000662 | 2: 0.000628 / 0.000628 | 4: 0.000697 / 0.00062 | 0: NA / NA | 0: NA / NA |
| 5e-4 | 3 | 4: 0.964 / 0.978 | 8: 0.95 / 0.945 | 11: 0.952 / 0.966 | 6: 0.971 / 0.972 | 3: 0.97 / 0.969 |
| 5e-4 | 4 | 3: 0.899 / 0.897 | 6: 0.895 / 0.895 | 7: 0.893 / 0.894 | 8: 0.895 / 0.895 | 8: 0.896 / 0.897 |
| 5e-4 | 5 | 2: 0.98 / 0.98 | 14: 0.983 / 0.984 | 5: 0.983 / 0.983 | 8: 0.983 / 0.984 | 3: 0.982 / 0.982 |
| 5e-4 | 6 | 5: 0.943 / 0.943 | 10: 0.948 / 0.948 | 10: 0.949 / 0.95 | 4: 0.948 / 0.947 | 3: 0.948 / 0.946 |
| 5e-4 | 7 | 3: 0.956 / 0.953 | 8: 0.958 / 0.957 | 5: 0.959 / 0.959 | 10: 0.956 / 0.956 | 6: 0.955 / 0.955 |
| 5e-4 | 8 | 3: 0.961 / 0.962 | 6: 0.961 / 0.961 | 9: 0.96 / 0.96 | 9: 0.96 / 0.961 | 5: 0.962 / 0.962 |
| 1e-3 | 0 | 11: 0.961 / 0.962 | 4: 0.963 / 0.963 | 7: 0.963 / 0.962 | 8: 0.961 / 0.961 | 2: 0.96 / 0.96 |
| 1e-3 | 1 | 17: 0.00134 / 0.00136 | 11: 0.00135 / 0.00135 | 1: 0.0013 / 0.0013 | 0: NA / NA | 3: 0.00213 / 0.0021 |
| 1e-3 | 2 | 3: 0.933 / 0.933 | 9: 0.933 / 0.932 | 6: 0.931 / 0.932 | 12: 0.932 / 0.934 | 2: 0.931 / 0.931 |
| 1e-3 | 3 | 8: 0.00105 / 0.000895 | 12: 0.000782 / 0.000728 | 4: 0.00259 / 0.0029 | 8: 0.00208 / 0.00178 | 0: NA / NA |
| 1e-3 | 4 | 1: 0.991 / 0.991 | 16: 0.99 / 0.99 | 5: 0.991 / 0.991 | 3: 0.99 / 0.99 | 7: 0.991 / 0.991 |
| 1e-3 | 5 | 3: 0.976 / 0.976 | 11: 0.973 / 0.974 | 12: 0.973 / 0.973 | 4: 0.971 / 0.971 | 2: 0.974 / 0.974 |
| 1e-3 | 6 | 5: 0.943 / 0.942 | 8: 0.838 / 0.896 | 6: 0.935 / 0.953 | 9: 0.902 / 0.966 | 4: 0.884 / 0.888 |
| 1e-3 | 7 | 4: 0.951 / 0.95 | 10: 0.95 / 0.951 | 7: 0.948 / 0.948 | 6: 0.947 / 0.947 | 5: 0.956 / 0.954 |
| 1e-3 | 8 | 4: 0.963 / 0.963 | 7: 0.963 / 0.962 | 5: 0.963 / 0.963 | 12: 0.965 / 0.964 | 4: 0.964 / 0.964 |

## Decision rules

- **state_dependent_expected_q**: YES when every mixed/error head has both delta-Q signs among its 32 onset states and the minimum head STD_DELTA_Q exceeds 1e-6.

- **weak_trigger_discrimination**: YES when at least 75% of mixed/error heads have STD_P_UPDATE <= 0.05, the median Spearman has absolute value <=0.10, and median AUC and sign accuracy are both in [0.45,0.55].

- **bias_dominated**: YES requires median BIAS_TO_STATE_STD_RATIO >=5 and at least 75% constant-like heads; NO when the median is <2 and every head is <5.

- **encoder_collapse**: YES requires both median encoder feature-STD and effective rank to be <=50% of the D-class control within each run; NO when the effective-rank ratio is >=0.8 in both runs.

- **operation_head_insensitivity**: YES requires encoder-collapse NO, mixed-head median STATE_TERM_STD <=50% of the D-class control in both runs, at least 75% constant-like heads, absolute value of the median Spearman <=0.10, and median AUC in [0.45,0.55].

## Execution integrity

`TRAINING_EXECUTED = NO`

`BACKWARD_CALLS = 0`

`OPTIMIZER_STEP_CALLS = 0`

`NEW_EXACT_Q_BRANCHES = 0`

`VALIDATION_EXECUTED = NO`

`TEST_EXECUTED = NO`

`CHECKPOINT_PARAMETERS_UNCHANGED = YES`
