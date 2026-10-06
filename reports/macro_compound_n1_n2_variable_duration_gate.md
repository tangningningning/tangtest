# Macro Compound N1/N2 Variable-Duration Gate

**Result: trace check failed; offline gate not evaluated.** The requested fresh5 completed all five episodes, but the post-run trace writer stopped on an E1 realized-return validation error before writing `macro_transition_trace.npz`. The corrected reporting validator passes the minimal synthetic checks; this run was not repeated.

## Requested report fields

```text
REPORTING_ONLY_CHANGE = YES
FORMAL_ALGORITHM_CHANGED = NO
MACRO_ACTOR_ADVANTAGE_MODE = DURATION_AWARE_SMDP_GAE
MACRO_CRITIC_TARGET_MODE = ONE_STEP_SMDP_TD
TRACE_IMPLEMENTATION_CHECK = FAIL
TRACE_SOURCE_RUN = C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\formal_macro_n1_target_ckptv30_archv9_fresh5_trace
TRACE_TRANSITIONS = 0 (NPZ absent)
EPISODE_TRANSITION_COUNTS = {"E1":108,"E2":106,"E3":102,"E4":105,"E5":102} (from completed episode metrics; total 523)
REQUESTED_DURATION_HISTOGRAM = {"1":189,"3":177,"5":157}
ACTUAL_DURATION_HISTOGRAM = {"1":191,"3":174,"5":157,"2":1} (from full episode-level actual-duration distributions)
TAIL_TRUNCATED_DURATION_COUNTS = {"3->2":1} (requested/actual pair inferred from E4 aggregate histograms; row unavailable)
N1_RECOMPUTE_MAX_ABS_ERROR = NOT_FULLY_VERIFIED; E1 N1 check passed the <=1e-6 guard before the realized-return check aborted serialization; E2-E5 were not checked by the writer.
REALIZED_RETURN_RECOMPUTE_MAX_ABS_ERROR = NOT_FULLY_VERIFIED_ON_FRESH5; pre-fix float32 validator reported 0.00018310546875 on E1 and aborted. Corrected float64 accumulation passed a 108-transition variable-duration synthetic trace with max error 0.0.
OFFLINE_COMPOUND_GATE_DATA_AVAILABLE = NO
N1_SCALE_MAX_MIN_RATIO = NOT_EVALUATED
N2_SCALE_MAX_MIN_RATIO = NOT_EVALUATED
COMPOUND_SCALE_MAX_MIN_RATIO = NOT_EVALUATED
STEP200_MEDIAN_TEST_EV = {"N1":"NOT_EVALUATED","N2":"NOT_EVALUATED","COMPOUND":"NOT_EVALUATED"}
STEP200_MEDIAN_TEST_CORRELATION = {"N1":"NOT_EVALUATED","N2":"NOT_EVALUATED","COMPOUND":"NOT_EVALUATED"}
STEP200_MEDIAN_TEST_STD_RATIO = {"N1":"NOT_EVALUATED","N2":"NOT_EVALUATED","COMPOUND":"NOT_EVALUATED"}
STEP200_MEDIAN_RETURN_EV = {"N1":"NOT_EVALUATED","N2":"NOT_EVALUATED","COMPOUND":"NOT_EVALUATED"}
STEP200_MEDIAN_RETURN_CORRELATION = {"N1":"NOT_EVALUATED","N2":"NOT_EVALUATED","COMPOUND":"NOT_EVALUATED"}
STEP200_MEDIAN_RETURN_STD_RATIO = {"N1":"NOT_EVALUATED","N2":"NOT_EVALUATED","COMPOUND":"NOT_EVALUATED"}
COMPOUND_GATE = NOT_EVALUATED
SELECTED_TARGET = NONE
ENVIRONMENT_EXECUTIONS = 1500
MATCHING_CALLS = 1500
KKT_CALLS = 154118
EXACT_Q_ENV_EXECUTIONS = 0
TEMP_FILES_CLEANED = YES
REPORT_PATH = C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\macro_compound_n1_n2_variable_duration_gate.json, C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\macro_compound_n1_n2_variable_duration_gate.md
NEXT_DECISION = A new fresh5 is required for a saved trace and offline gate; no further training was launched after the failed trace check.
```

## Trace validation and correction

The E1 writer check first accepted N1 against the `1e-6` guard, then rejected realized return-to-go with an apparent maximum error of `0.00018310546875`. Training computes realized returns using float64 accumulation and casts at the end; the reporting validator had recursively accumulated float32 reward scalars. The reporting code now promotes rewards to float64 before reverse recursion.

Minimal checks passed after the fix: the synthetic N1/N2 terminal-boundary check and a 108-transition trace-builder fixture with variable durations plus a `5 -> 2` tail. Both N1 and realized-return reconstruction errors were `0.0`. The fresh5 trace remains unavailable because training had already completed and section 17 says not to continue training after a failed trace check.

The aggregate run metrics preserve all 523 requested durations and actual-duration categories, including one actual duration `2` in E4. E4 requested and actual category counts imply one `3 -> 2` tail truncation. These aggregates do not preserve observation/reward/value rows or ordered linkage, so they cannot support N2 construction or the cross-episode offline critic comparison.

## Run counters and semantics

The run completed E1–E5 with episode seeds 20001–20005, initialization seed 27005, no resume, replay history 0, and 50 Macro Critic steps per update. The manifest records architecture version 9, checkpoint version 30, `DURATION_AWARE_SMDP_GAE`, and `ONE_STEP_SMDP_TD`; the configured algorithm was not changed. Counters: 1500 environment executions, 1500 matching calls, 154118 KKT calls, 0 exact-Q environment executions, 20 Macro Actor optimizer steps, and 250 Macro Critic optimizer steps.

## Decision

`OFFLINE_COMPOUND_GATE_DATA_AVAILABLE = NO`; `COMPOUND_GATE = NOT_EVALUATED`; `SELECTED_TARGET = NONE`. A new fresh5 run would be needed to save the corrected trace before the requested N1/N2/50-50 comparison. No additional training was started.
