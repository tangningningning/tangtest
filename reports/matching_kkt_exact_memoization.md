# Matching-local exact KKT memoization

## 1. Scope and baseline

- Baseline SHA: `550d12f11c283bf071727b8e988ba2130e9f8714`
- Real fixture: origin 20, one prediction batch with the largest bounded task set found in that origin.
- Bounded batch: 18 tasks, batch index 24.
- Matching mode: existing `incremental` mode.
- No PPO, Batch4, 240-slot rollout, Cloud/Cache/UAV audit, GPU, parameter, algorithm, or system-model change was run.
- `12.pdf` was not searched for, created, copied, or modified.

## 2. Implemented change

Modified file/function:

- `src/decision/task_allocation.py`
- `optimize_task_matching()` creates one `KKTResourceAllocationCache` for the duration of that single Matching optimization.
- `allocate_resources_kkt()` accepts that local cache and memoizes only node-level `KKTResourceAllocationResult` values.
- `evaluate_matching_with_kkt()`, `evaluate_matching_candidate()`, `evaluate_matching_candidate_incremental()`, and `enumerate_best_operation()` pass the same local cache through the existing evaluation path.

Cache scope:

- one `optimize_task_matching()` call only;
- never shared between prediction batches;
- never shared between UAV positions or configuration states;
- no global cache and no checkpoint/history state.

Exact key:

```python
(node_id, tuple(sorted(task_ids_assigned_to_this_node)))
```

The tasks, vehicle positions, service-node positions, Matching/KKT project configuration, and `cache_config` are fixed for the lifetime of this local cache. The cached value includes CPU/BW allocations, resource sums, CPU/BW multipliers, and feasible/infeasible result information. Infeasible node results are also cached and reused exactly.

The candidate generation loops, candidate order, candidate counters, acceptance rule, Matching formulas, and KKT formulas were not changed.

## 3. Bounded before/after result

The same origin-20 bounded fixture was run once with an equivalent no-cache control and once with the exact local cache.

| Metric | Before / no-cache | After / exact cache |
|---|---:|---:|
| Wall time | 0.317070 s | 0.079233 s |
| `allocate_resources_kkt()` calls | 720 | 720 |
| CPU/BW KKT fraction solver calls | 1,440 | 260 |
| Candidate evaluations | 404 | 404 |
| Transfer candidates | 338 | 338 |
| Exchange candidates | 66 | 66 |

Cache counters after the optimized run:

- requests: `720`
- hits: `590`
- misses: `130`
- entries: `130`
- hit ratio: `81.9444%`

The 130 exact node-state misses each perform the CPU and bandwidth solver once, hence `130 × 2 = 260` fraction-solver calls. The number of outer `allocate_resources_kkt()` calls does not change; only repeated node-level solver work is removed.

Observed bounded wall-time reduction was approximately `75.0%`, or `4.00×` speedup for this single 18-task fixture. This is a bounded measurement, not a full-training estimate.

## 4. Exact equivalence

The no-cache control and optimized run produced:

- identical final matching;
- identical accepted transfer/exchange history and order;
- identical candidate evaluation, transfer, and exchange counts;
- identical total `Q` (`J_diff=0`);
- maximum CPU allocation difference: `0`;
- maximum bandwidth allocation difference: `0`;
- maximum CPU resource-sum difference: `0`;
- maximum bandwidth resource-sum difference: `0`;
- maximum delay difference: `0`;
- maximum vehicle-energy difference: `0`;
- maximum per-task utility/Q difference: `0`;
- identical final stability result and stop reason;
- identical result signature SHA-256: `3de3ff90c163c5642212aac3daab111699d314375b74b2cc7416ba47a2d4d463`.

The bounded accepted history contains 11 updates, ending with the same exchange:

```text
tasks ['16', '19']: RSU_1, 43 -> 43, RSU_1
delta Q = 0.029390367500119474
```

## 5. One full origin-20 workers=8 run

Because the bounded equivalence check passed and the solver-call reduction was substantial, one complete origin-20 joint Cache/UAV optimizer run was executed with the existing prediction-batch worker setting `workers=8`.

- runtime: `149.977538 s`
- stop reason: `converged_by_no_improvement`
- outer rounds: `1`
- final objective `J`: `29.066036957604311`
- accepted Cache updates: `0`
- accepted UAV updates: `0`
- final UAV positions:
  - `UAV_0=(3379.03, 1119.88)`
  - `UAV_1=(5179.65, 1710.41)`
- final cache state, represented in the run output as bytes, was:

```text
000100000000000000010000000000000100000000010000000000000001000000000000010000000000000001000000010000000101010100000000000001000101000100010000000000010101000000000100010100010000
```

No second full workers=1 run was performed.

## 6. Decision and remaining blocker

The optimization is retained. It satisfies the exact-equivalence checks and materially reduces repeated CPU/BW KKT solver calls without changing the candidate set, candidate order, acceptance sequence, objective, resource semantics, or cache lifetime.

The remaining cost is still the number of distinct node task-set states multiplied by the prediction-batch count and the unchanged Matching/KKT evaluation. This change removes repeated identical node allocations only; it does not reduce candidate enumeration, prediction scenarios, horizon, or Matching search.

## 7. Hygiene

- Temporary validation scripts were deleted after the runs.
- Temporary bytecode files from those scripts were deleted.
- No CSV, JSON, benchmark, or profiling output was created for this task.
- User-owned `打包并上传.bat` was not modified, deleted, moved, renamed, staged, or committed.
- No commit or push was performed.
