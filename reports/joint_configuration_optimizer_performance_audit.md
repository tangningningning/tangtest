# Joint Cache/UAV configuration optimizer performance-structure audit

## 1. Scope and baseline

- Baseline commit: `550d12f11c283bf071727b8e988ba2130e9f8714`
- Audited data: real `origin 20` fixture.
- Prediction context: 20 scenarios × 5 horizon steps = 100 prediction batches.
- No Batch4, PPO training, Gate-CF, WHAT-CF, Direct-CF, GPU, or algorithm/configuration change was run.
- The bounded probe used the existing prediction-batch executor with requested/effective workers `8/8` and no serial fallback. It did not start CF workers.
- The probe recorded complete candidate lists but returned only the first candidate from each generator call. Thus the execution window covered the initial reference, Cache candidate groups, UAV candidate groups, and one outer Cache/UAV round without attempting the full search.
- The temporary probe was observation-only and was deleted after the audit. `12.pdf` was not searched for, copied, created, or modified.

## 2. Bounded window result

The bounded joint call completed successfully:

- `cache_calls=2`, `uav_calls=2`, `joint_evals=3`
- `outer_rounds=1`
- `accepted_cache=0`, `accepted_uav=0`
- `stop=converged_by_no_improvement`
- objective `J=26.23802994359177`
- predictive utility `26.2407799436`
- movement cost `0.00275`
- no Python/native crash and no exception

The final bounded state was deterministic in the probe:

- final UAV positions: `UAV_0=(3367.78, 1119.88)`, `UAV_1=(5179.65, 1710.41)`
- final cache state and objective history were preserved by the probe for comparison; no source optimization was applied.

## 3. Configuration-level counts

| Item | Count |
|---|---:|
| Configuration utility requests | 24 |
| Unique configuration keys | 5 |
| Duplicate requests | 19 |
| Duplicate ratio | 79.17% |
| Cache hits | 19 |
| Cache misses | 5 |
| `requests = hits + misses` | 24 = 19 + 5 |
| Reference-state requests | 8 |
| Reference-state duplicate hits | 7 |
| Outer rounds entered | 1 |

The exact-key source split was:

| Source | Requests | Unique | Hits | Misses | Predictive batches | Candidate evaluations |
|---|---:|---:|---:|---:|---:|---:|
| Cache | 9 | 4 | 5 | 4 | 400 | 133,937 |
| UAV | 12 | 1 | 11 | 1 | 100 | 35,329 |
| Joint | 3 | 0 | 3 | 0 | 0 | 0 |
| Total | 24 | 5 | 19 | 5 | 500 | 169,266 |

Candidate enumeration itself was not the bottleneck:

- Cache: two observed groups, 7 candidates per group; duplicate cache rows: 0; enumeration time: `0.000138 s`.
- UAV initialization: three observed groups, 64 candidates per group; duplicate positions within groups: 0; enumeration time: `0.000326 s`.
- UAV coordinate refinement: two observed groups, 26 candidates per group; one duplicate position in each group; enumeration time: `0.000437 s`.

The lists above are the complete lists returned by the generators, while only each group’s first candidate was executed in the bounded window. The 19 duplicate configuration requests were exact utility-cache hits and did not repeat prediction batches.

## 4. Prediction and multiprocessing counts

- Prediction batch evaluations: `500`.
- Unique batch/state evaluations: `500` (`5` unique configuration states × `100` batches).
- Predictive Matching calls: `500`.
- Total Matching candidate evaluations: `169,266`.
- Bounded average: `338.532` candidate evaluations per predictive Matching call.
- Requested workers: `8`.
- Effective workers: `8`.
- Mode: `parallel`.
- Serial fallbacks: `0`.
- Process-pool map calls: `5`.
- Parent map wall time: `20.402 s`.
- Sum of child Matching timings: `141.511 s` CPU-time-equivalent across workers; this is an aggregate and must not be added to the parent wall time.

The parent configuration-evaluation time was `20.404 s`, of which the executor map accounted for `20.402 s`. Context construction was only `0.008240 s`.

## 5. Matching/KKT path

A fixed-state serial probe on the same origin-20 context used all 100 prediction batches to expose the full/incremental path:

| Item | Count | Time |
|---|---:|---:|
| Predictive Matching calls | 100 | 27.435 s wall |
| Full candidate evaluator calls | 100 | 0.168 s |
| Incremental candidate evaluator calls | 35,329 | 27.089 s |
| KKT evaluation calls | 63,406 | 26.467 s |
| KKT resource-allocation calls | 63,406 | 25.326 s |
| Channel-gain calls | 316,059 | 1.034 s |
| Base-rate calls | 156,173 | 0.673 s |

The full evaluator is used for the initial matching evaluation of each batch. The transfer/exchange candidates use the affected-node incremental path; the probe found no unexpected fallback of those candidates to the full evaluator. The timing rows are nested measurements, so their percentages must not be summed.

Relative to the 27.435-second serial wall time for one 100-batch fixed state:

- incremental evaluator time: `98.74%`;
- KKT evaluation time: `96.47%`;
- KKT resource allocation time: `92.32%`;
- channel-gain time: `3.77%`;
- base-rate time: `2.45%`;
- full evaluator time: `0.61%`.

The serial probe reported `353.290` candidate evaluations per Matching call for that fixed state. The bounded joint aggregate was `338.532` because its five states have different task/matching workloads.

## 6. Time decomposition and dominant hotspot

The dominant cost is not context creation, candidate enumeration, or duplicate cache lookup. It is:

`unique Cache/UAV configuration state × 100 prediction batches × actual Matching transfer/exchange + KKT resource redistribution`.

For the bounded window, five unique states required `20.463 s` wall time with 8 workers, or approximately `4.08 s` per unique state in this window. A serial 100-batch state required `27.435 s`; projecting five comparable states serially would be about `137.2 s`, versus the observed `20.46 s` parallel wall time (approximately `6.7×` wall-time reduction for this bounded projection). This is a measurement projection, not a full-search runtime claim.

The parent map wall time was effectively the configuration time, while candidate enumeration was below one millisecond per category. The probe did not isolate serialization/deserialization as a separate timer; there is no evidence in this window that it dominates. The large gap between aggregate child Matching time (`141.511 s`) and parent wall time (`20.402 s`) is consistent with the existing 8-worker overlap.

## 7. Duplicate-evaluation evidence

The exact configuration key was `(cache bytes, ordered UAV fixed positions)` for the fixed origin-20 context. In the bounded window:

1. 19 of 24 utility requests were exact cache hits.
2. The reference key was requested 8 times and hit 7 times.
3. Cache-side requests were 9 with 4 unique states; all five repeats were hits.
4. UAV-side requests were 12 with 1 unique state; eleven repeats were hits.
5. Joint-side requests were 3 and all were hits; they caused no new batch evaluation.
6. Cache neighbor lists had no duplicate rows in the observed groups.
7. Each observed UAV coordinate list contained one duplicate position; this is a generator-level duplicate candidate, but it was not removed in this audit.
8. The same immutable 100-batch context was reused by the five executed unique configurations. No duplicate context construction was observed inside the bounded call.

The current exact utility memoization is therefore functioning. The expensive work occurs on the five misses, not on the 19 hits.

## 8. SAFE versus ALGORITHM-CHANGING candidates

Potential SAFE candidates, recorded but not implemented in this audit:

- Keep and reuse the existing exact configuration utility memoization. It is already present and demonstrated by the 19 hits.
- Reuse an already-computed exact reference evaluation where the inputs are identical.
- Avoid the exact duplicate UAV coordinate entries after proving that the complete trial state is identical.
- In `slot_environment._movement_cost()`, avoid calling the full UAV utility wrapper when only movement fields are consumed by the joint evaluator. The wrapper performs an exact utility-cache lookup before calling the same deterministic `movement_summary()`. This is semantics-preserving in the audited call path, but the measured hit component was only `0.000089 s` in the bounded window, so it is not the 7-minute blocker.

ALGORITHM-CHANGING candidates, not implemented:

- reducing Cache/UAV candidates or search rounds;
- reducing scenarios or prediction horizon;
- changing angles, tolerances, stopping rules, or coordinate ordering;
- approximate early pruning or surrogate replacement;
- changing Matching candidate search or KKT/resource allocation;
- changing the number of prediction batches or the formal experiment parameters.

## 9. Fix decision and remaining blocker

No formal source optimization was implemented. The temporary instrumentation was sufficient to identify the hotspot, while the only immediately visible exact-hit micro-optimization measured at far below `0.001%` of bounded wall time. Therefore there is no before/after semantic-equivalence run to report and no expected speedup is claimed for an unimplemented fix.

The remaining blocker is the multiplicative workload of many unique configuration candidates, each evaluated over all 100 prediction batches, with each batch invoking the real incremental Matching/KKT chain. Any material speedup will require a separately authorized optimization of that workload; reducing it would cross into the algorithm-changing category unless an exact, input-identical reuse opportunity is proven.

## 10. Files and repository hygiene

- New report: `reports/joint_configuration_optimizer_performance_audit.md`.
- Formal source files were not modified by this audit.
- Temporary profiling script was deleted.
- No CSV, JSON, benchmark, or profiling output file was left.
- User-owned `打包并上传.bat` was not modified, deleted, moved, renamed, staged, or committed.
- No commit or push was performed.
