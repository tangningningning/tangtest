# Training normalization-reference comparison

Generated: `2026-10-19T16:18:29.677312+00:00` using `C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`.

## Scope and contracts

- Training-only sample: 1642 tasks in 120 predeclared Slots from 12 Episodes; no Validation/Test files were read.
- Matching/KKT sample: 9 actual Slots with 132 arrived tasks; Cache and UAV positions are frozen at the formal initial state.
- Current Cloud pool: 20.0 GHz, fixed per-task uplink 20.0 Mbps, delay factor 13.0; weights 0.5/0.5.
- Current configured references: T_ref=30.4737283553 s; E_ref=5.15746425063 J. These are the existing Training P95 references; they were originally calculated under a 40 GHz Cloud pool.
- Across different reference schemes, Q is an internal objective value only. The comparisons below emphasize physical delay, vehicle uplink energy, Cloud assignment share, and resource allocations.

## P95 provenance and 20 GHz recalculation

The earlier deterministic calibration record reports P95 T=30.4737283553 s and E=5.15746425063 J from 1,642 tasks / 120 Slots / 12 Training Episodes. Its reference assignment used seed-4401 uniform feasible Cache, configured initial UAV positions, least-loaded feasible cached Edge selection, equal per-node Edge CPU/BW shares, 20 Mbps Cloud upload, and equal Cloud CPU shares at 40 GHz.
Using the task CSV's row order, the 40 GHz replay produced T_P95=29.6330346784 s and E_P95=4.76175794955 J; this does **not** exactly reproduce the recorded refs. A second diagnostic using ascending numeric task ID reproduces the recorded 40 GHz P95 exactly, but the old report did not record that task iteration order, so this is evidence of an unresolved execution-contract detail, not proof of provenance.
At current 20 GHz, the row-order replay gives P95 T=29.6330346784 s and E=4.76175794955 J; ascending task-ID order gives T=30.4737283553 s and E=5.15746425063 J. The latter remains equal to the recorded global refs under this sample. These are diagnostics only and are not written to formal configuration.
Historical Stage6 local-compute references T=30.6289851662 s and E=3.06289851662 J came from 3,343 arrivals; exact source Episode/Slot selection was not recorded, so that older local calibration is not exactly reproducible from local provenance. The prior report's mean 722.70 is in KiB; this run finds 740.05 decimal KB, equivalent to 722.70 KiB.

## Reference-scale statistics

P5/P50/P95/mean/min/max are task-level Training statistics (linear percentile interpolation). Local values are diagnostic only.

| Scheme | T_ref P5 / P50 / P95 / mean (s) | E_ref P5 / P50 / P95 / mean (J) | Median effective delay coefficient ratio | Median effective energy coefficient ratio |
|---|---:|---:|---:|---:|
| `GLOBAL_P95_FORMAL_REFS` | 30.47 / 30.47 / 30.47 / 30.47 | 5.157 / 5.157 / 5.157 / 5.157 | 1× | 1× |
| `GLOBAL_P95_RECALIBRATED_20G` | 29.63 / 29.63 / 29.63 / 29.63 | 4.762 / 4.762 / 4.762 / 4.762 | 1.028× | 1.083× |
| `LOCAL_TIME_ONLY` | 2.534 / 3.634 / 7.038 / 4.044 | 5.157 / 5.157 / 5.157 / 5.157 | 8.386× | 1× |
| `LOCAL_ENERGY_ONLY` | 30.47 / 30.47 / 30.47 / 30.47 | 12.67 / 18.17 / 35.19 / 20.22 | 1× | 0.2839× |
| `LOCAL_COMBINED` | 2.534 / 3.634 / 7.038 / 4.044 | 12.67 / 18.17 / 35.19 / 20.22 | 8.386× | 0.2839× |
| `RELATIVE_DEADLINE_ALPHA_0.5` | 1.267 / 1.817 / 3.519 / 2.022 | 5.157 / 5.157 / 5.157 / 5.157 | 16.77× | 1× |
| `RELATIVE_DEADLINE_ALPHA_1` | 2.534 / 3.634 / 7.038 / 4.044 | 5.157 / 5.157 / 5.157 / 5.157 | 8.386× | 1× |
| `RELATIVE_DEADLINE_ALPHA_2` | 5.068 / 7.268 / 14.08 / 8.089 | 5.157 / 5.157 / 5.157 / 5.157 | 4.193× | 1× |

The assumed local baseline uses F_local=1e+09 cycles/s and κ=5e-27; these are not defined by the formal system config or Training tasks. `LOCAL_TIME_ONLY` and `LOCAL_ENERGY_ONLY` isolate one denominator at a time; `LOCAL_COMBINED` changes both. Relative Deadline cases use D=α·T_local for α∈{0.5,1,2} while holding E_ref at the formal global value. No real business Deadline or hard-deadline constraint is defined.

## Fixed global Matching: KKT response only

The global-reference formal Matching is fixed for this layer. Every scheme re-solves the affected Edge CPU, Edge bandwidth, and finite Cloud CPU KKT pools with that task mapping.

| Scheme | Mean delay (s) | Mean upload energy (J) | Cloud rate | Mean abs Edge CPU fraction Δ | Mean abs Edge BW fraction Δ | Mean abs Cloud CPU Δ (GHz/task) | Resource feasible |
|---|---:|---:|---:|---:|---:|---:|---|
| `GLOBAL_P95_FORMAL_REFS` | 3.9433 | 0.08433 | 52.273% | 0.00000 | 0.00000 | 0.00000 | YES |
| `GLOBAL_P95_RECALIBRATED_20G` | 3.9433 | 0.08433 | 52.273% | 0.00000 | 0.00000 | 0.00000 | YES |
| `LOCAL_TIME_ONLY` | 3.9601 | 0.08451 | 52.273% | 0.01534 | 0.01452 | 0.13011 | YES |
| `LOCAL_ENERGY_ONLY` | 3.9434 | 0.08434 | 52.273% | 0.00000 | 0.00368 | 0.00000 | YES |
| `LOCAL_COMBINED` | 3.9605 | 0.08458 | 52.273% | 0.01534 | 0.01713 | 0.13011 | YES |
| `RELATIVE_DEADLINE_ALPHA_0.5` | 3.9603 | 0.08454 | 52.273% | 0.01534 | 0.01571 | 0.13011 | YES |
| `RELATIVE_DEADLINE_ALPHA_1` | 3.9601 | 0.08451 | 52.273% | 0.01534 | 0.01452 | 0.13011 | YES |
| `RELATIVE_DEADLINE_ALPHA_2` | 3.9599 | 0.08447 | 52.273% | 0.01534 | 0.01263 | 0.13011 | YES |

The allocation deltas are paired against the formal global P95 KKT allocation, flattened by (node, task); entries absent from a node receive zero for this difference summary. The fixed assignment and current physical input are otherwise identical.

## Re-optimized Transfer–Exchange Matching + KKT

Per-task reference maps were supplied to a process-local shadow evaluator. The official `optimize_task_matching` candidate generation, Transfer–Exchange order, termination, target set and KKT numeric solver functions were reused. Global P95 shadow and formal KKT results were checked for equality on all nine fixed assignments. Per-scheme KKT caches are fresh and never shared across reference maps. Full candidate evaluation was used so the formal global-denominator upper-bound pruning path could not silently apply to task-specific denominators.

| Scheme | Mean delay (s) | Mean upload energy (J) | Cloud rate | Matching targets changed vs global (tasks) | Mean abs Edge CPU Δ | Mean abs Edge BW Δ | Mean abs Cloud CPU Δ (GHz/task) | Mean optimizer time (s) | Total KKT solves |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `GLOBAL_P95_FORMAL_REFS` | 3.9433 | 0.08433 | 52.273% | 0 | 0.00000 | 0.00000 | 0.00000 | 0.3014 | 2353 |
| `GLOBAL_P95_RECALIBRATED_20G` | 3.9433 | 0.08433 | 52.273% | 0 | 0.00000 | 0.00000 | 0.00000 | 0.3129 | 2353 |
| `LOCAL_TIME_ONLY` | 3.8543 | 0.12821 | 45.455% | 14 | 0.19201 | 0.19188 | 0.51485 | 0.2938 | 2131 |
| `LOCAL_ENERGY_ONLY` | 3.8451 | 0.12167 | 46.212% | 13 | 0.18030 | 0.18208 | 0.37496 | 0.3039 | 2230 |
| `LOCAL_COMBINED` | 3.8481 | 0.14355 | 43.182% | 19 | 0.25720 | 0.25960 | 0.64014 | 0.2938 | 2078 |
| `RELATIVE_DEADLINE_ALPHA_0.5` | 3.8479 | 0.14352 | 43.182% | 19 | 0.25720 | 0.25887 | 0.64014 | 0.2824 | 2066 |
| `RELATIVE_DEADLINE_ALPHA_1` | 3.8543 | 0.12821 | 45.455% | 14 | 0.19201 | 0.19188 | 0.51485 | 0.2946 | 2131 |
| `RELATIVE_DEADLINE_ALPHA_2` | 3.8752 | 0.11416 | 46.212% | 13 | 0.17349 | 0.17248 | 0.45750 | 0.2876 | 2164 |

Q values are retained in JSON for within-scheme traceability but are not used to rank different denominator schemes. Cloud rate is task-weighted over the same 9 Slots. Full Matching is a Transfer–Exchange local search, not a global assignment optimum.

## Findings and recommendation

- Formal system declares local CPU: **False**; local κ: **False**; business Deadline: **False**.
- Shadow KKT reproduces formal global-reference KKT within 8.88e-16 maximum absolute difference across checked metrics: **True**.
- Re-optimized matching comparison completed for all 8 schemes and all 9 Slots: **COMPLETED_ALL_SCHEMES**.
- The full shadow search and formal incremental search selected identical global-reference assignments on 9/9 Slots.
- Recommendation: **KEEP_CURRENT_GLOBAL_P95_FORMAL; RESOLVE_REFERENCE_POLICY_ORDER_BEFORE_ANY_RECALIBRATION**. A single global scale is specified and avoids task-specific weights based on unmeasured local hardware or hypothetical business deadlines. The 20 GHz P95 is order-sensitive in this fixed reference policy; the task-ID-sorted replay matches the stored refs, while CSV order does not. Because the historical task iteration order is absent from the record, no unique replacement is justified. Keep current formal values pending provenance clarification; do not write diagnostic values into config.
- The local-compute reference is a literature-parameter sensitivity scenario, not a measured vehicle baseline. Relative Deadline scenarios are denominator sensitivities, not business requirements. No scheme is declared superior because its Q has a different scale.

## Reproducibility and change boundaries

- Current config SHA-256: `ed4596e8df7f0e7b15144c845bbc2c86f77e0c300c60183ad1dfd215777fc2a5`; Training pool manifest SHA-256: `b6cae236b029f4d24c7488a91f0825247a20f475ec76b2d80737de44ae491bd3`.
- Episode manifests and task/trajectory/role/NPZ hashes verified for 16 Training Episodes; Validation/Test were not opened.
- Source hashes: task allocation `1df9ac8e6cfa777bf0b9a572dd6344fe8a2ecd8fe5b28b4275eb8c03d696589c`, system model `dd352e2da2e2e916d2734b487fae22914950c3f41a65d4ba7749b280d0fbdeec`, catalog `6e8c497ed06d76e1251720ebc926e975a6a5c079fd224acab95ea5fb4799cf13`.
- `FORMAL_CONFIG_MODIFIED=NO`; `FORMAL_ALGORITHM_MODIFIED=NO`; `MAPPO_TRAINING_EXECUTED=NO`; `VALIDATION_TEST_READ=NO`.
- Limitations: the P95 policy is a deterministic reference allocation rather than a learned/optimized policy; local CPU/κ and Deadline scenarios are assumptions; matching remains Transfer–Exchange local search. Results are conditional on the predeclared 9 Training Slots and 1,642-task reference sample.
