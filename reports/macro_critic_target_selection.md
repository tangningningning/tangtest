# Macro Critic Target Selection

**Selected target: `N2`.** Highest median step200 TEST_EV; candidates within 0.10: ['N2'].

## Run and trace validation

- Trace source: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\formal_macro_n1_target_ckptv30_archv9_fresh5_trace_v2\macro_transition_trace.npz`
- Algorithm semantics: Macro Actor `DURATION_AWARE_SMDP_GAE`, Macro Critic `ONE_STEP_SMDP_TD`; architecture v9, checkpoint v30.
- Fresh initialization seed 27005; episode seeds 20001–20005; resume `False`; replay history 0; Macro Critic 50 steps/update; Macro Actor 4 update epochs.
- `REPORTING_ONLY_CHANGE = YES`; `FORMAL_ALGORITHM_CHANGED = NO`.
- `TRACE_IMPLEMENTATION_CHECK = PASS`; 523 rows, with per-episode counts `{"E1":108,"E2":106,"E3":102,"E4":105,"E5":102}`.
- N1 reconstruction max absolute error: `0`; realized return-to-go reconstruction max absolute error: `0`.
- Integrity checks passed: required arrays present; 400D observations; frame indices continuous per episode; terminal appears only on each episode’s last transition; same-episode next-observation linkage; terminal old-next value is zero; per-episode counts match formal episode metrics.

## Duration rows

- `REQUESTED_DURATION_HISTOGRAM = {"1":189,"3":177,"5":157}`
- `ACTUAL_DURATION_HISTOGRAM = {"1":191,"2":1,"3":174,"5":157}`
- `TAIL_TRUNCATED_DURATION_COUNTS = {"3->1":2,"3->2":1}` (counted directly from each row’s requested/actual pair).

## Per-episode target scale and realized-return diagnostics

Scale ratio is max per-episode target standard deviation divided by min per-episode target standard deviation. Pearson and tie-aware Spearman correlations compare each target to the saved realized return-to-go.

| Target | Episode | Count | Target mean | Target std | Pearson vs return | Spearman vs return |
|---|---:|---:|---:|---:|---:|---:|
| N1 | E1 | 108 | -1.820108 | 37.465482 | 0.139351 | 0.029343 |
| N1 | E2 | 106 | -13.736839 | 55.281092 | 0.144220 | 0.175726 |
| N1 | E3 | 102 | -13.103776 | 82.642926 | -0.155055 | -0.223097 |
| N1 | E4 | 105 | -4.356832 | 35.908714 | -0.117667 | -0.076560 |
| N1 | E5 | 102 | -57.187864 | 80.868541 | 0.696912 | 0.772034 |
| N2 | E1 | 108 | -3.310941 | 65.977843 | 0.181903 | 0.081045 |
| N2 | E2 | 106 | -17.529885 | 83.165424 | 0.211049 | 0.290798 |
| N2 | E3 | 102 | -24.856552 | 120.762088 | -0.187415 | -0.233468 |
| N2 | E4 | 105 | -3.221154 | 50.363800 | -0.000301 | 0.050954 |
| N2 | E5 | 102 | -53.186889 | 94.482179 | 0.692307 | 0.750038 |
| COMPOUND_50_50 | E1 | 108 | -2.565524 | 50.299649 | 0.171198 | 0.067621 |
| COMPOUND_50_50 | E2 | 106 | -15.633362 | 65.265906 | 0.195543 | 0.268176 |
| COMPOUND_50_50 | E3 | 102 | -18.980164 | 95.102634 | -0.186361 | -0.244664 |
| COMPOUND_50_50 | E4 | 105 | -3.788993 | 41.852767 | -0.050659 | -0.001068 |
| COMPOUND_50_50 | E5 | 102 | -55.187376 | 86.644343 | 0.702694 | 0.763032 |

| Target | Scale max/min ratio | Median Pearson vs return | Median Spearman vs return |
|---|---:|---:|---:|
| N1 | 2.301473 | 0.139351 | 0.029343 |
| N2 | 2.397795 | 0.181903 | 0.081045 |
| COMPOUND_50_50 | 2.272314 | 0.171198 | 0.067621 |

## Cross-episode offline critic fits

Each target used a fresh 400→128→128→1 critic with SiLU activations, seed 27005, Adam (`lr=1e-3`, `eps=1e-5`), full-batch `0.01 × MSE`, gradient norm cap 0.5, and 200 updates. Inputs were transformed as `2*x-1`. The realized return-to-go was evaluation-only and never entered the loss. Step 0 is before fitting; step 200 is after the 200th optimizer update.

| Target | Fold | Step0 TEST_EV | Step200 TEST_EV | Step200 TEST_CORRELATION | Step200 TEST_STD_RATIO | Step0 RETURN_EV | Step200 RETURN_EV | Step200 RETURN_CORRELATION | Step200 RETURN_STD_RATIO |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N1 | E1 -> E2 | -0.000063 | -0.843044 | 0.167978 | 1.101391 | 0.000058 | -0.196120 | -0.400903 | 0.196460 |
| N1 | E1-2 -> E3 | -0.000130 | -0.093059 | 0.298853 | 0.725903 | 0.000068 | -0.244504 | -0.723392 | 0.152850 |
| N1 | E1-2-3 -> E4 | 0.000028 | 0.339926 | 0.583280 | 0.566249 | 0.000061 | -0.055369 | -0.083488 | 0.166191 |
| N1 | E1-2-3-4 -> E5 | -0.000124 | -0.149491 | 0.854018 | 1.791481 | -0.000053 | 0.441546 | 0.690806 | 0.501949 |
| N2 | E1 -> E2 | -0.000042 | -0.103916 | 0.167188 | 0.530323 | 0.000058 | -0.037063 | -0.059064 | 0.142311 |
| N2 | E1-2 -> E3 | -0.000108 | -0.867832 | -0.074120 | 0.860399 | 0.000068 | -0.112794 | -0.080664 | 0.264735 |
| N2 | E1-2-3 -> E4 | 0.000003 | 0.479143 | 0.731565 | 0.968302 | 0.000061 | -0.330768 | -0.215624 | 0.398592 |
| N2 | E1-2-3-4 -> E5 | -0.000103 | 0.158527 | 0.727506 | 1.336389 | -0.000053 | 0.281333 | 0.540280 | 0.437472 |
| COMPOUND_50_50 | E1 -> E2 | -0.000056 | -0.311352 | 0.154820 | 0.733890 | 0.000058 | -0.132500 | -0.351382 | 0.154552 |
| COMPOUND_50_50 | E1-2 -> E3 | -0.000136 | -0.207780 | 0.091787 | 0.556765 | 0.000068 | -0.145794 | -0.472880 | 0.134910 |
| COMPOUND_50_50 | E1-2-3 -> E4 | 0.000013 | 0.570350 | 0.758507 | 0.687924 | 0.000061 | -0.205319 | -0.318587 | 0.235323 |
| COMPOUND_50_50 | E1-2-3-4 -> E5 | -0.000115 | -0.142607 | 0.783074 | 1.652448 | -0.000053 | 0.354517 | 0.605362 | 0.496062 |

### Step0 evaluation metrics

| Target | Fold | Step0 TEST_EV | Step0 TEST_CORRELATION | Step0 TEST_STD_RATIO | Step0 RETURN_EV | Step0 RETURN_CORRELATION | Step0 RETURN_STD_RATIO |
|---|---|---:|---:|---:|---:|---:|---:|
| N1 | E1 -> E2 | -0.000063 | -0.091703 | 0.000343 | 0.000058 | 0.472891 | 0.000061 |
| N1 | E1-2 -> E3 | -0.000130 | -0.329641 | 0.000197 | 0.000068 | 0.821863 | 0.000041 |
| N1 | E1-2-3 -> E4 | 0.000028 | 0.026300 | 0.000536 | 0.000061 | 0.194696 | 0.000157 |
| N1 | E1-2-3-4 -> E5 | -0.000124 | -0.410182 | 0.000151 | -0.000053 | -0.625757 | 0.000042 |
| N2 | E1 -> E2 | -0.000042 | -0.091164 | 0.000228 | 0.000058 | 0.472891 | 0.000061 |
| N2 | E1-2 -> E3 | -0.000108 | -0.399949 | 0.000135 | 0.000068 | 0.821863 | 0.000041 |
| N2 | E1-2-3 -> E4 | 0.000003 | 0.004450 | 0.000382 | 0.000061 | 0.194696 | 0.000157 |
| N2 | E1-2-3-4 -> E5 | -0.000103 | -0.398286 | 0.000129 | -0.000053 | -0.625757 | 0.000042 |
| COMPOUND_50_50 | E1 -> E2 | -0.000056 | -0.096920 | 0.000290 | 0.000058 | 0.472891 | 0.000061 |
| COMPOUND_50_50 | E1-2 -> E3 | -0.000136 | -0.397156 | 0.000171 | 0.000068 | 0.821863 | 0.000041 |
| COMPOUND_50_50 | E1-2-3 -> E4 | 0.000013 | 0.013960 | 0.000460 | 0.000061 | 0.194696 | 0.000157 |
| COMPOUND_50_50 | E1-2-3-4 -> E5 | -0.000115 | -0.408577 | 0.000141 | -0.000053 | -0.625757 | 0.000042 |

### Step200 medians

| Target | Scale max/min ratio | Median TEST_EV | Median TEST_CORRELATION | Median TEST_STD_RATIO | Median RETURN_EV | Median RETURN_CORRELATION | Median RETURN_STD_RATIO | Max fold TEST_STD_RATIO | Scale screen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| N1 | 2.301473 | -0.121275 | 0.441066 | 0.913647 | -0.125744 | -0.242196 | 0.181325 | 1.791481 | RETAINED |
| N2 | 2.397795 | 0.027306 | 0.447347 | 0.914350 | -0.074929 | -0.069864 | 0.331664 | 1.336389 | RETAINED |
| COMPOUND_50_50 | 2.272314 | -0.175194 | 0.456663 | 0.710907 | -0.139147 | -0.334985 | 0.194937 | 1.652448 | RETAINED |

The scale screen excludes a target if median step200 test std-ratio exceeds 3 or any fold exceeds 10. None of the fits used environment execution.

## Decision

- `SELECTED_TARGET = N2`
- `NEXT_DECISION = 将 Macro Critic target 正式替换为 N2，只做一次 N2 fresh5 确认。`
- Training counters from the formal run: environment executions 1500; matching calls 1500; KKT calls 154118; exact-Q environment executions 0; completed episodes 5.
- Offline critic fits were held in memory; no environment was called after fresh5. The formal trace, checkpoints, and run metrics remain in the run directory.
- Temporary runner/checkpoint/tensor files were not created. Python bytecode caches and temporary files were removed at task cleanup.

## Requested key fields

```text
REPORTING_ONLY_CHANGE = YES
FORMAL_ALGORITHM_CHANGED = NO
TRACE_IMPLEMENTATION_CHECK = PASS
TRACE_TRANSITIONS = 523
EPISODE_TRANSITION_COUNTS = {"E1":108,"E2":106,"E3":102,"E4":105,"E5":102}
REQUESTED_DURATION_HISTOGRAM = {"1":189,"3":177,"5":157}
ACTUAL_DURATION_HISTOGRAM = {"1":191,"2":1,"3":174,"5":157}
TAIL_TRUNCATED_DURATION_COUNTS = {"3->1":2,"3->2":1}
N1_RECOMPUTE_MAX_ABS_ERROR = 0
REALIZED_RETURN_RECOMPUTE_MAX_ABS_ERROR = 0
TARGET_SCALE_MAX_MIN_RATIO = {"COMPOUND":2.2723141059369087,"N1":2.3014727450128443,"N2":2.397795393463177}
STEP200_MEDIAN_TEST_EV = {"COMPOUND":-0.1751936795877006,"N1":-0.1212750075398693,"N2":0.027305913279216887}
STEP200_MEDIAN_TEST_CORRELATION = {"COMPOUND":0.4566634524170976,"N1":0.44106648794077563,"N2":0.44734688855611293}
STEP200_MEDIAN_TEST_STD_RATIO = {"COMPOUND":0.7109070654477364,"N1":0.9136468861527283,"N2":0.914350412917951}
STEP200_MEDIAN_RETURN_EV = {"COMPOUND":-0.13914664706126578,"N1":-0.12574447410969058,"N2":-0.07492872978385179}
STEP200_MEDIAN_RETURN_CORRELATION = {"COMPOUND":-0.3349846996161482,"N1":-0.24219579438748143,"N2":-0.06986410200985142}
STEP200_MEDIAN_RETURN_STD_RATIO = {"COMPOUND":0.1949372813994638,"N1":0.18132543244461208,"N2":0.3316636184750672}
SELECTED_TARGET = N2
ENVIRONMENT_EXECUTIONS = 1500
MATCHING_CALLS = 1500
KKT_CALLS = 154118
EXACT_Q_ENV_EXECUTIONS = 0
REPORT_PATH = reports/macro_critic_target_selection.md
DUPLICATE_JSON_REPORT_REMOVED = YES (prior gate Markdown retains the previous no-trace conclusion)
TEMP_FILES_CLEANED = YES
NEXT_DECISION = 将 Macro Critic target 正式替换为 N2，只做一次 N2 fresh5 确认。
```

