# Minimal PPO v2 short training run

## Scope

This was a single bounded train-only runtime attempt for the new `src/ppo_simple/` implementation. No PPO mathematics, network, hyperparameter, environment algorithm, Gate/WHAT/Direct-CF path, or formal configuration was changed for this run.

The working tree was preserved before the run. The pre-existing prediction-worker BLAS/runtime change in `src/decision/cache_configuration.py`, the existing formal reports, `adaptive_cache_mec.zip`, and `打包并上传.bat` were not reverted or modified by this run.

## Static blocker check

The previously reported cache assertion text is no longer present in the active cache configuration source. The matching text found under `reports/prediction_worker_blas_runtime_fix.md` is historical evidence only. No new cache assertion was raised during this attempt.

## Launch

```text
C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe -m ppo_simple.trainer --config system_config.toml --episodes 1 --run-dir runs/ppo_simple_seed4401_smoke001
```

The formal configuration was kept unchanged: seed `4401`, train origins `0..239`, prediction-batch workers requested `8`, and CF workers configured as `8`. The run directory was new and was not pre-existing.

## Observed result

The parent process started at approximately `2026-09-08 03:15:51 +08:00` and was stopped at approximately `2026-09-08 03:33:32 +08:00` after the bounded observation window. During the attempt, the parent and eight prediction-batch worker processes were observed, all responsive and consuming CPU. No Python exception, native crash, OMP error, or NaN/Inf report was observed.

The run did not complete the normal rollout within the bounded window. It therefore did not reach the PPO update or checkpoint save. No run directory or checkpoint was produced.

| Metric | Result |
|---|---|
| completed episodes | `0` |
| completed transitions | not available; episode did not finish |
| reward min/mean/max | not available |
| cache request count/rate | not available; emitted only after completed rollout |
| UAV request count/rate | not available; emitted only after completed rollout |
| actual cache update count | not available |
| actual UAV movement count | not available |
| policy loss / value loss | not reached |
| entropy / approximate KL / clip fraction | not reached |
| parameter delta | not reached |
| checkpoint | not created |
| NaN/Inf | none observed before bounded stop |
| exception | none observed |
| Direct-CF / WHAT-CF / Gate-CF | not started |

The peak failure point was runtime non-completion in the normal rollout's formal prediction/configuration computation, before the first completed episode summary. This is classified as an `ENVIRONMENT_RUNTIME_BLOCKER` for this bounded formal workload, not as a `PPO_V2_BLOCKER`. The bounded stop is not evidence of a PPO correctness failure.

## Adaptation checks

Because the episode did not finish and the implementation emits the aggregate action/state metrics only after the update boundary, this run cannot provide completed-run counts for action variability, cache/UAV requests, or actual state transitions. No request/actual assertion or other environment exception was observed before stopping. The required action-variability and full request/actual-contract checks remain unconfirmed by this run.

## Verdict

`PPO_V2_SHORT_TRAIN_READY = NO`

Reason: one complete episode and PPO training/update cycle, finite losses/gradients, checkpoint creation, and completed action/state metrics were not reached within the bounded runtime window. No PPO or algorithm change is proposed.
