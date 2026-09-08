# ppo_simple_v2 formal-training readiness and first-episode report

## Executive result

The static training/data split checks passed, and the minimum structured
metrics logging was added to `src/ppo_simple/trainer.py`. The first complete
formal episode was then started through the official entrypoint, but it did
not complete: after approximately one hour, the first default 16-origin
rollout chunk had not returned and no checkpoint or metrics record had been
written. The run was stopped to avoid unbounded training. No PPO numerical
exception was observed.

This is a runtime-completion failure, not a convergence judgment.

Repository state for this task: branch `main`, HEAD
`b7e90d4015bbeac6823eb55a4a29d71a7e174d7`; no commit or push was performed.

## 1. Training episode and data protocol

### 1.1 Complete episode definition — `CONFIRMED`

`src/ppo_simple/trainer.py::run_training()` starts at
`current_train_origin=train_origin_start` and advances through contiguous
chunks. A chunk is terminal only when its last transition is beyond
`train_origin_end`; only then is `completed_episodes` incremented. The state
produced by one chunk is passed to the next chunk, so a chunk boundary is not
an environment reset.

With the current default `DEFAULT_ROLLOUT_STEPS=16`, a complete episode is:

- 240 real transitions;
- train origins `0..239`, inclusive;
- 15 rollout/update calls;
- 60 optimizer steps with the current `update_epochs=1` and
  `num_minibatches=4`;
- terminal at origin `239`, without using origin `240` as a train bootstrap.

The first formal run used no `--rollout-steps` override, so it used the
current default of 16. The previous smoke run is not counted as this episode.

### 1.2 Origin ranges — `CONFIRMED`

The current `system_config.toml` data protocol is:

| Split | Origin range | Count |
|---|---:|---:|
| PPO train | `0..239` | 240 |
| PPO validation | none | 0 |
| Held-out test | `240..299` | 60 |

The generated `prediction_scenarios.npz` was read-only checked: it contains
300 unique `origin_slots`, `[0, 1, ..., 299]`, with every adjacent difference
equal to 1.

`src/common/project_config.py::validate_for_formal_experiment()` additionally
checks that train starts at 0, test starts at `train_end + 1`, the intervals
have no overlap or gap, test ends at the formal episode end, and the train
count is divisible by the configured minibatch count.

### 1.3 Validation/test isolation — `CONFIRMED`

- There is no PPO validation-origin split. The `num_val_sequences` setting
  belongs to the popularity-prediction training configuration, not PPO origin
  data.
- The ppo_simple trainer constructs aligned steps only from the configured
  train interval. `aligned_train_steps()` rejects an origin outside the train
  range, and terminal train collection explicitly forbids a test-origin
  bootstrap step.
- Test origins are not passed to `collect_rollout()`, `compute_gae()`,
  `PPOV2Trainer.update()`, or the optimizer.
- Train/test are non-overlapping contiguous origin segments of the same
  complete 300-slot trajectory/scenario artifact. They are not separate
  scenario files. There is no random slot shuffling or interleaving. The only
  adjacent cross-split boundary is the deliberate `239 | 240` boundary.

Therefore validation/test data cannot participate in PPO gradient, GAE, or
optimizer updates through the current ppo_simple training path.

### 1.4 Seed responsibilities — `CONFIRMED`

| Seed/config | Current value | Responsibility |
|---|---:|---|
| `experiment.seed` | 4401 | `run_training()` seeds Python `random`, NumPy, and Torch; Torch controls the normal Bernoulli PPO action sampling and initialization RNG. |
| `simulation.sumo_seed` | 4401 | SUMO random seed and deterministic research/service vehicle-role selection in the generated vehicle artifact. |
| `simulation.traffic_seed` | 4401 | SUMO `randomTrips` traffic-demand generation. |
| `task.master_seed` | 4401 | SeedSequence root for service catalog, popularity evolution, task arrival, service request, and workload-variation streams. |
| `prediction.vehicle.observation_noise_seed` | 9101 | Synthetic vehicle-observation noise used when pre-generating vehicle predictions. |
| `scenario.master_seed` | 20260820 | Prediction-scenario/residual generation. |

The formal PPO runtime loads the generated vehicle/task/scenario artifacts; it
does not regenerate them or draw new vehicle/task data per PPO action. The
PPO sampling seed is therefore separate from the already materialized data
artifacts, even where the configured numeric seed is also 4401.

## 2. Formal metrics readiness

### 2.1 State before the minimal logging change

The trainer already had transition-level reward, cache cost, movement cost,
actual Cache update count, actual UAV movement count, and per-update scalar
loss/KL/entropy values for console output. It did not persist a complete
episode record, `episode_return`, `mean_G`, `reward_std`, episode-level request
and cost totals, or gradient norm/range diagnostics in a structured file.
The checkpoint stored training state, but not these plotting records.

### 2.2 Minimal logging change — `CONFIRMED`

Only `src/ppo_simple/trainer.py` was changed. The change does not alter PPO
math, reward calculation, observations, actions, environment decisions,
Cache/UAV optimization, Matching/KKT, or PPO hyperparameters.

The trainer now writes, under each run directory:

- `episode_metrics.jsonl`: one record only when a complete episode reaches
  trainer terminal;
- `update_metrics.jsonl`: one record per PPO update/chunk.

Episode records include:

```text
seed, episode, num_transitions, episode_return, mean_reward, mean_G,
reward_std, cache_request_count, actual_cache_update_count,
uav_request_count, actual_uav_movement_count, cache_cost, movement_cost,
ppo_updates, optimizer_steps, policy_loss_range, value_loss_range,
approximate_kl_range, gradient_norm_range, finite_checks, checkpoint,
wall_clock_runtime_seconds
```

Update records include:

```text
optimizer_steps, policy_loss, value_loss, entropy, approximate_kl,
clip_fraction, gradient_norm, policy/value/KL/gradient min/max,
returns_finite, advantages_finite, gradient_finite, total_loss_finite
```

The gradient norm is the total norm returned by the existing
`clip_grad_norm_` call before clipping; the clipping call and configured
`max_grad_norm` are unchanged.

The logging path explicitly verifies:

```text
episode_return = sum(reward_t)
G(t) = actual_task_utility - cache_cost - movement_cost
reward_t = G(t)
```

It resets episode accumulators at the trainer's episode boundary, so
`episode_return` is not a cross-episode running total. `allow_nan=False` and
explicit finite checks prevent non-finite metric records from being written.

## 3. First complete-episode attempt

### 3.1 Formal command — `CONFIRMED`

The official entrypoint was run directly, with normal policy sampling and no
observer truncation:

```text
C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe -m ppo_simple.trainer --config system_config.toml --episodes 1 --run-dir runs\ppo_simple_v2_formal_episode1_20260908
```

This is equivalent to the formal `ppo_simple.trainer:main` entrypoint and did
not pass `--rollout-steps`; the current default 16 was used.

### 3.2 Completion — `FAILED` / runtime blocked

Observed outcome:

- `completed_episodes`: 0;
- first 16-origin rollout chunk did not return;
- no PPO update reached the trainer's `update()` call;
- no optimizer step was completed by this run;
- no formal episode checkpoint or JSONL metric record was produced;
- no Python exception, infeasible-transition error, or non-finite PPO error was
  observed;
- the parent and worker processes remained CPU-active, with repeated groups of
  eight approximately 500 MB Python workers;
- after approximately one hour without the first chunk checkpoint, the run was
  stopped to respect the requested minimal/no-long-training scope.

This reproduces the current real-environment worker runtime bottleneck seen in
earlier bounded observations. It is not evidence that PPO converged or failed
to converge.

### 3.3 Metrics from the incomplete run — `NOT VERIFIED`

Because the first chunk never reached the trainer update/checkpoint boundary,
the following complete-episode metrics are unavailable:

```text
transitions, PPO updates, optimizer steps, episode_return, mean_G,
reward min/mean/max, Cache requests/updates, UAV requests/movements,
cache/movement cost, policy/value loss ranges, KL range, gradient range,
checkpoint, wall-clock complete-episode runtime, finite metric record
```

The prior real-action smoke checkpoint remains separate and is not presented
as a formal full-episode result.

## Final status

```ini
FORMAL_EPISODE_COMPLETED = NO
EPISODE_METRICS_READY = NO
TRAIN_SPLIT_VERIFIED = YES
VALIDATION_SPLIT_EXISTS = NO
TEST_SPLIT_EXISTS = YES
READY_FOR_MULTI_EPISODE_TRAINING = NO
```

The code and structured logging are ready for a future full run after the
environment worker runtime is addressed. No hyperparameter tuning or model/
system-algorithm change was made in response to this runtime blocker.

Temporary validation artifacts and Python caches were removed after stopping
the run. No formal full-episode checkpoint or metrics file was retained from
the incomplete attempt. `git diff --check` was run after cleanup and passed.
