# Macro Critic Target Estimator Gate

Offline comparison on the saved 300-transition trace. Formal algorithm, configuration, observation, architecture, and checkpoint were not modified.

## GAE verification and duration-aware target construction

- Stored/recomputed GAE max absolute error: `0` (threshold `1e-5`).
- Trace decay `(lambda)^L`: mean `0.773781`, median `0.773781`, min `0.773781`, max `0.773781`.
- Approximate effective Macro trace horizon: `4.42049` transitions.
- N-step targets stop at terminal, use only same-episode rewards, and bootstrap from `next_macro_value` only after a full n-step nonterminal span.

## Target scale statistics

| Target | Mean of episode means | Std of episode means | Mean of episode stds | Std of episode stds | Std CV | Scale max/min |
|---|---:|---:|---:|---:|---:|---:|
| GAE | 42.4794 | 43.4056 | 74.8307 | 49.682 | 0.663926 | 9.87465 |
| N1 | -38.8834 | 29.1799 | 85.4648 | 43.2581 | 0.506151 | 4.51049 |
| N2 | -13.9125 | 31.4579 | 85.8872 | 45.4026 | 0.528631 | 4.40385 |
| N4 | 35.0131 | 40.8871 | 87.0225 | 49.1405 | 0.564687 | 6.12356 |
| N8 | 128.686 | 68.1377 | 99.6839 | 47.7104 | 0.478617 | 3.82214 |
| MC | 716.913 | 254.453 | 439.263 | 119.747 | 0.27261 | 2.37532 |

Per-episode mean/std/min/max and target-minus-MC statistics are included in JSON.

## Target correlation with MC

| Target | Pearson | Spearman |
|---|---:|---:|
| GAE | -0.0519013 | -0.0459885 |
| N1 | -0.289512 | -0.186733 |
| N2 | -0.240097 | -0.168474 |
| N4 | -0.122676 | -0.0980322 |
| N8 | 0.145245 | 0.168305 |
| MC | 1 | 1 |

## Step 200 held-out medians

| Target | Median TEST_EV | Median correlation | Median pred/target std | Median matched-fit TEST_EV |
|---|---:|---:|---:|---:|
| GAE | -3.82983 | 0.270186 | 1.66748 | -0.755619 |
| N1 | -0.0170527 | 0.406284 | 0.798224 | 0.206018 |
| N2 | -0.537969 | 0.408845 | 0.891156 | 0.0561527 |
| N4 | -3.24974 | 0.334006 | 1.62593 | -1.01644 |
| N8 | -7.36344 | -0.050436 | 2.27872 | -1.96482 |
| MC | 0.393829 | 0.653203 | 0.626694 | 0.436066 |

## Cross-realization fold metrics

All requested metrics (train/test EV, correlation, std ratio, RMSE, mean bias) at steps 50/100/200/300, and first matched-fit metrics, are recorded per fold in JSON.

## Episode 20004 scale check

| Target | Step | TEST_EV | Correlation | Std ratio | Mean bias |
|---|---:|---:|---:|---:|---:|
| GAE | 50 | -0.0213166 | 0.365223 | 0.758548 | 70.6921 |
| GAE | 100 | -12.1648 | 0.536058 | 4.06482 | 17.93 |
| GAE | 200 | -21.8298 | 0.624436 | 5.33821 | 40.0877 |
| GAE | 300 | -29.4551 | 0.654905 | 6.12153 | 49.9783 |
| N1 | 50 | 0.473252 | 0.818205 | 0.375253 | 14.3335 |
| N1 | 100 | -0.141177 | 0.624799 | 1.35387 | -5.31132 |
| N1 | 200 | -2.36036 | 0.347382 | 1.92251 | 24.1796 |
| N1 | 300 | -0.639065 | 0.541146 | 1.5065 | 20.9037 |
| N2 | 50 | 0.562317 | 0.785071 | 0.552652 | 26.3948 |
| N2 | 100 | -0.295329 | 0.707179 | 1.59905 | 22.9058 |
| N2 | 200 | -0.923898 | 0.583919 | 1.70858 | 46.6591 |
| N2 | 300 | -0.843483 | 0.62452 | 1.73515 | 53.7542 |
| N4 | 50 | 0.136371 | 0.397756 | 0.545538 | 59.5229 |
| N4 | 100 | -5.57956 | 0.596256 | 3.03246 | 46.0831 |
| N4 | 200 | -11.1838 | 0.639657 | 4.0445 | 74.125 |
| N4 | 300 | -17.5351 | 0.627322 | 4.86155 | 86.9143 |
| N8 | 50 | -0.14727 | -0.583587 | 0.114871 | 85.8818 |
| N8 | 100 | -2.58149 | -0.502824 | 1.18072 | 38.0288 |
| N8 | 200 | -14.5034 | 0.0721934 | 3.88121 | 7.71258 |
| N8 | 300 | -16.6699 | 0.084483 | 4.16824 | -2.71473 |
| MC | 50 | 0.112894 | 0.928498 | 0.0629264 | -178.569 |
| MC | 100 | 0.510932 | 0.912723 | 0.345157 | 429.151 |
| MC | 200 | 0.644198 | 0.865492 | 1.18934 | 549.04 |
| MC | 300 | 0.697845 | 0.842274 | 0.949888 | 557.564 |

## Secondary calibration quality checks

Beyond the written A/B/C gate, this review checks correlation degradation across folds, how close the median prediction/target standard-deviation ratio is to 1, and median absolute mean bias. The systematic-correlation guard flags 3 or more folds with a drop greater than 0.15.

| Candidate | Median correlation delta | Folds down >0.15 | Median |std ratio−1| (GAE → candidate) | Median absolute mean bias (GAE → candidate) | Quality guard |
|---|---:|---:|---:|---:|---:|---|
| N1 | 0.192236 | 1/4 | 1.02251 → 0.320021 | 35.1724 → 16.4216 | PASS |
| N2 | 0.127675 | 0/4 | 1.02251 → 0.237442 | 35.1724 → 30.1362 | PASS |
| N4 | 0.0478324 | 1/4 | 1.02251 → 0.982264 | 35.1724 → 50.9235 | TRADEOFF |
| N8 | -0.300118 | 3/4 | 1.02251 → 1.7181 | 35.1724 → 18.2272 | TRADEOFF |

N1 and N2 pass these secondary checks; N4 reduces target scale but has higher median absolute mean bias. N1 is selected because it passes the formal gate, passes the secondary checks, and its step-200 median TEST_EV exceeds N2 by more than the 0.10 tie window.

## N-step gates

| Candidate | Gate A | B1 | B2 | B3 | C | Result |
|---|---:|---:|---:|---:|---:|---|
| N1 | True | True | True | False | True | **PASS** |
| N2 | True | True | True | True | True | **PASS** |
| N4 | True | True | False | False | True | **PASS** |
| N8 | False | False | False | False | True | **FAIL** |

Estimator problem: **SUPPORTED**.
Selected diagnostic target: **N1**.
Long-horizon return information needed: **NO**.

## Cleanup

- Temporary runner removed.
- No temporary model, tensor, or checkpoint files were saved.
- No `__pycache__` directories or `*.pyc` files remain.

## Requested fields

```text
GAE_RECOMPUTE_MAX_ABS_ERROR = 0.0
TRACE_DECAY_FACTOR_MEAN = 0.7737809374999998
TRACE_DECAY_FACTOR_MEDIAN = 0.7737809374999998
EFFECTIVE_MACRO_TRACE_HORIZON_APPROX = 4.420493962572226
TARGET_SCALE_MAX_MIN_RATIO = {"GAE": 9.874653138219866, "N1": 4.510491274827513, "N2": 4.403854806168248, "N4": 6.123561137295014, "N8": 3.8221435253854588, "MC": 2.3753238063331796}
TARGET_CORR_WITH_MC = {"GAE": {"pearson": -0.05190134059690901, "spearman": -0.04598851098345537}, "N1": {"pearson": -0.28951201685260264, "spearman": -0.18673318592428806}, "N2": {"pearson": -0.24009690097676054, "spearman": -0.16847431638129312}, "N4": {"pearson": -0.12267639221867992, "spearman": -0.09803220035778175}, "N8": {"pearson": 0.14524464024040323, "spearman": 0.16830453671707463}}
STEP200_MEDIAN_TEST_EV = {"GAE": -3.8298306493128065, "N1": -0.017052707782632737, "N2": -0.5379690150141114, "N4": -3.249743088049013, "N8": -7.3634395304298454, "MC": 0.3938290863823857}
STEP200_MEDIAN_TEST_CORRELATION = {"GAE": 0.27018597097844144, "N1": 0.40628431941633864, "N2": 0.408845056945048, "N4": 0.3340063990870738, "N8": -0.050435997904351985, "MC": 0.6532033359992271}
STEP200_MEDIAN_TEST_STD_RATIO = {"GAE": 1.6674822224811203, "N1": 0.7982242839067657, "N2": 0.8911561147666289, "N4": 1.6259278058738378, "N8": 2.2787197407244753, "MC": 0.6266942971581076}
MATCHED_FIT_MEDIAN_TEST_EV = {"GAE": -0.755618759351476, "N1": 0.20601807428915297, "N2": 0.05615265266138847, "N4": -1.016438873536892, "N8": -1.9648212556055031, "MC": 0.43606623079975326}
EPISODE_20004_STEP200 = {"GAE": {"EV": -21.829775801819114, "CORRELATION": 0.6244364169451264, "STD_RATIO": 5.338213738862909, "RMSE": 83.19367723057928, "MEAN_BIAS": 40.087661258627975, "PRED_MEAN": 30.174926137924196, "TARGET_MEAN": -9.912735120703777, "PRED_STD": 81.44468513141894, "TARGET_STD": 15.2569172227951}, "N1": {"EV": -2.360359254669349, "CORRELATION": 0.3473819112140147, "STD_RATIO": 1.922511571562029, "RMSE": 80.23546029886838, "MEAN_BIAS": 24.179588079452515, "PRED_MEAN": -27.38884998957316, "TARGET_MEAN": -51.568438069025675, "PRED_STD": 80.23580453503376, "TARGET_STD": 41.7348876968489}, "N2": {"EV": -0.9238983447171805, "CORRELATION": 0.5839188072488344, "STD_RATIO": 1.7085785416467907, "RMSE": 67.50802860815463, "MEAN_BIAS": 46.65909112890561, "PRED_MEAN": 7.784898447990417, "TARGET_MEAN": -38.87419268091519, "PRED_STD": 60.09749541353747, "TARGET_STD": 35.173972953923034}, "N4": {"EV": -11.1838230198224, "CORRELATION": 0.639656573620935, "STD_RATIO": 4.044503640964617, "RMSE": 117.88752872609797, "MEAN_BIAS": 74.12498841782411, "PRED_MEAN": 60.68636783460776, "TARGET_MEAN": -13.43862058321635, "PRED_STD": 106.21588502445275, "TARGET_STD": 26.261784993502978}, "N8": {"EV": -14.503364052309728, "CORRELATION": 0.07219343250729766, "STD_RATIO": 3.881205895601494, "RMSE": 181.07079021100898, "MEAN_BIAS": 7.7125750226279095, "PRED_MEAN": 42.085467926661174, "TARGET_MEAN": 34.372892904033264, "PRED_STD": 178.32317376112636, "TARGET_STD": 45.94530116611877}, "MC": {"EV": 0.6441981744629588, "CORRELATION": 0.8654924676673877, "STD_RATIO": 1.189342799033631, "RMSE": 572.0627593413084, "MEAN_BIAS": 549.040185379982, "PRED_MEAN": 867.818814086914, "TARGET_MEAN": 318.77862870693207, "PRED_STD": 320.3337121316086, "TARGET_STD": 269.3367399137467}}
N1_GATE = "PASS"
N2_GATE = "PASS"
N4_GATE = "PASS"
N8_GATE = "FAIL"
MACRO_CRITIC_TARGET_ESTIMATOR_PROBLEM = "SUPPORTED"
SELECTED_DIAGNOSTIC_TARGET = "N1"
LONG_HORIZON_RETURN_INFORMATION_NEEDED = "NO"
PRIMARY_SUSPECT = "FIXED_N_STEP_TARGET_N1"
FORMAL_GAE_CHANGED = "NO"
FORMAL_OBSERVATION_CHANGED = "NO"
ARCHITECTURE_VERSION = 9
CHECKPOINT_VERSION = 29
MATCHING_CALLS = 0
KKT_CALLS = 0
EXACT_Q_ENV_EXECUTIONS = 0
FORMAL_OPTIMIZER_STEPS = 0
REPORT_PATH = "reports/macro_critic_target_estimator_gate.json"
TEMP_FILES_CLEANED = "YES"
NEXT_DECISION = "Design a separate formal N1 target-replacement experiment; this offline gate leaves the current GAE unchanged."
```
