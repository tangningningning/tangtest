# Prediction-worker BLAS runtime fix

## Scope

This change is limited to the Windows prediction-batch runtime. It does not
change PPO, Direct-CF, Gate-CF, WHAT-CF, reward, Matching/KKT, Cache/UAV
optimization, worker count, scenario/horizon settings, formal parameters, or
the system page-file configuration.

## Previous failure mechanism

The previous bounded formal preflight failed before Direct-CF during the
Windows prediction-batch spawn path with an OpenMP/OpenBLAS allocation error
(`OMP: Error #111: Memory allocation failed`) and an extended parent/worker
wait. The earlier worker-initializer change had already removed the explicit
PyTorch import and PyTorch thread settings from the prediction worker.

## Static worker/import check

- The temporary wrapper used for this preflight had no top-level `torch` or
  PPO/training import; heavyweight imports occurred inside `main()` after
  `multiprocessing.freeze_support()`.
- The formal parent imports PyTorch for PPO policy construction.
- The prediction worker initializer
  (`_initialize_prediction_batch_worker`) only stores its payload.
- The prediction worker job imports `decision.uav_configuration` and uses the
  existing prediction/Matching path; `src/decision/cache_configuration.py`
  contains no `torch` reference after the dependency-isolation change.
- No prediction-worker algorithm or data path was copied or rewritten.

## Runtime change

`src/decision/cache_configuration.py` now sets the following inherited
environment values immediately before creating the Windows `spawn` pool:

```text
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
BLIS_NUM_THREADS=1
```

The parallel batch evaluator now waits for `FIRST_EXCEPTION`. On a worker
failure it cancels pending futures, shuts down the executor without waiting
for unfinished jobs, and raises a clear
`PREDICTION_BATCH_WORKER_FAILED` exception. No arbitrary per-task timeout was
introduced. Successful results are reduced in the original job order.

## Bounded workers=8 observation

One bounded, train-only 20-slot normal-rollout fixture was run with the formal
configuration. It did not run PPO update, held-out test, Gate-CF, WHAT-CF, or
Direct-CF.

- prediction-batch workers requested: `8`
- effective workers observed: `8` spawned prediction workers in the parent
  `MainProcess` path
- all five thread environment values above: `1`
- `OMP: Error #111`: not reproduced
- native/Python memory exception: not reproduced
- CPU=0 parent/worker hang: not observed; the parent and eight workers were
  CPU-active while the bounded run was executing
- first `[PPO] rollout progress | step=20/20` line: not reached
- Direct-CF started: no
- Gate-CF/WHAT-CF started: no

The run stopped at the first trigger configuration after approximately three
minutes with the existing configuration assertion:

```text
AssertionError: cache node 87 has a positive feasible candidate gain but no actual update
```

The stack reached `optimize_joint_configuration -> optimize_cache ->
_finalize_cache_candidate_profiles`. This is a separate normal-rollout
configuration correctness failure; it was not modified or investigated in
this runtime-isolation task.

## Verdict

The OpenBLAS/OMP allocation blocker was not reproduced after the inherited
thread-limit change, and the prediction worker pool did not exhibit the prior
memory-amplification failure. However, the bounded normal rollout did not
cross its first 20-slot progress point because it stopped at the pre-existing
cache configuration assertion.

`RUNTIME_BLOCKER_CLEARED = NO`

Reason: the BLAS/OpenMP symptom is cleared in this bounded observation, but
the requested normal-rollout runtime path remains unable to reach the first
progress checkpoint due to the separate existing assertion. No conclusion is
made about PPO or Direct-CF correctness.

## Hygiene

- Temporary preflight fixture was removed.
- No profiling output, CSV, JSON, `12.pdf`, or page-file/system setting was
  created or changed.
- The user-owned `打包并上传.bat` was not modified, staged, or committed.

