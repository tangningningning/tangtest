# Scheme B PPO time-to-go v2 implementation

## Result

`adaptive_primitive_v2` is now available through the dedicated formal configuration `system_config_ppo_primitive_v2.toml`.  It appends exactly one finite-horizon scalar to the prior shared observation:

```text
time_to_go = (train_origin_end - origin_index) / (train_origin_end - train_origin_start)
```

For the formal range `0..239`, this gives `1.0` at origin 0 and `0.0` at origin 239.  No reward, environment, configuration primitive, Matching, KKT, prediction, GAE, bootstrap, PPO loss, or PPO hyperparameter was changed.

## Modified files

- `system_config_ppo_primitive_v2.toml`
- `src/common/project_config.py`
- `src/ppo_simple/__init__.py`
- `src/ppo_simple/interface.py`
- `src/ppo_simple/policy.py`
- `src/ppo_simple/primitive_control.py`
- `src/ppo_simple/rollout.py`
- `src/ppo_simple/trainer.py`
- `reports/ppo_time_to_go_v2_implementation.md`

## Observation and schema compatibility

`ConfigurationObservationSpec` now supports an optional final `time_to_go_slice`.  v1 creates no such slice and follows the unchanged original observation path.  v2 requires a finite scalar in `[0, 1]` and appends it after Cache state, UAV positions, predicted vehicle positions, and predicted service popularity.

Actual formal dimensions obtained by loading the current configuration (without calling `execute_slot()`):

```ini
V1_OBSERVATION_DIM = 154
V2_OBSERVATION_DIM = 155
```

The formula is centralized in `interface.normalized_time_to_go()`.  Both main rollout observations and Scheme B's intermediate Cache/UAV conditional observations call that helper when the v2 observation spec is active.

## Policy data flow

`AdaptivePrimitiveV2ActorCritic` subclasses the preserved `AdaptivePrimitiveActorCritic` and differs only by its architecture identifier:

```text
adaptive_primitive_v1  # preserved baseline class and v1 observation
adaptive_primitive_v2  # identical network mechanism with the 155-dim v2 observation
```

The shared encoder receives the v2 observation in all relevant paths:

```text
v2 observation
  -> shared encoder
     -> trigger head
     -> critic/value head
     -> Cache conditional scorer (via captured Cache context)
     -> UAV conditional scorer (via captured UAV context)
```

`primitive_control._intermediate_observation()` now uses the same v2 builder as the main rollout.  `evaluate_primitive()` re-encodes the captured Cache/UAV contexts, so PPO log-prob recomputation uses the same time context as sampling.

## Checkpoint compatibility and best checkpoint

The existing `adaptive_primitive_v1` class, v1 observation dimension, v1 checkpoint payload, and unchanged baseline `system_config.toml` remain available.  New v2 runs must use a new run directory and the explicit v2 configuration:

```text
ppo-train --config system_config_ppo_primitive_v2.toml --epochs <N> --run-dir <new-v2-run-dir>
```

Checkpoint loading now validates `observation_dim` before loading policy or optimizer tensors.  A pure in-memory v1 payload presented to a v2 trainer was rejected with:

```text
PPO checkpoint observation_dim incompatible; refusing silent schema resume:
checkpoint=154, expected=155.
```

This blocks partial loading and silent v1-to-v2 continuation.  The existing 407-Epoch v1 checkpoint was not opened, altered, or overwritten by this task.

For v2 only, checkpoints now carry `best_epoch` and `best_epoch_return`.  `latest_checkpoint.pt` retains its existing meaning: most recent completed update/cursor.  Whenever a complete Epoch creates a new highest train return, the trainer additionally saves:

```text
best_train_return_checkpoint.pt
```

The episode metrics record the current `best_epoch` and `best_epoch_return`.  This is a training-protection checkpoint, not a validation-selected or paper-final model.

## Critic diagnostics

For a v2 run, `critic_diagnostics.jsonl` is appended once per transition using quantities already produced by the rollout and `compute_gae()`:

```text
seed
episode, epoch
chunk_origin_start, optimizer_steps
origin, done
old_value
raw_advantage
raw_return, value_target
bootstrap_value
value_residual = old_value - raw_return
abs_value_residual
```

No solver, environment, or additional critic evaluation is invoked for these records.  Diagnostics are not printed to the default console or to `--verbose` output.

## Minimal checks

- Compiled the changed PPO and configuration modules: PASS.
- Pure formula check: `time_to_go(0, 0, 239) = 1.0`; `time_to_go(239, 0, 239) = 0.0`.
- Read-only aligned-input check: origin 0 and origin 239 observations both have shape `(155,)`; their final entries are `1.0` and `0.0`, respectively.
- Single no-solver v2 model forward: trigger logits shape `(11,)`; critic value is scalar.
- Pure Cache and UAV conditional scorer forwards: finite log probabilities, one candidate logit per synthetic candidate.
- Pure checkpoint-payload check: v1 payload contains no new best fields; v2 payload contains the requested best fields.
- Pure in-memory v1-to-v2 load check: explicit 154-to-155 observation-dimension rejection.
- No `execute_slot()`, PPO update, rollout collection, complete Epoch, checkpoint write, or long training run was performed.

## Next training recommendation

Start a fresh v2 run directory from initial v2 weights with `--config system_config_ppo_primitive_v2.toml`.  Do not use `--resume` with `runs/ppo_primitive_first_episode_20260908`.  First collect the planned small v2 diagnostic run and inspect `critic_diagnostics.jsonl`, especially terminal versus non-terminal `abs_value_residual`, before deciding whether a longer v2 training run is justified.  Preserve v1 results as the baseline for later comparison; do not call its best-train-return checkpoint a validation result.

## Required conclusions

```ini
TIME_TO_GO_IMPLEMENTED = YES
TIME_TO_GO_FORMULA = (end - origin) / (end - start)
TIME_TO_GO_ORIGIN_0 = 1.0
TIME_TO_GO_ORIGIN_239 = 0.0

V1_POLICY_SCHEMA_PRESERVED = YES
V2_POLICY_SCHEMA_CREATED = YES

V1_OBSERVATION_DIM = 154
V2_OBSERVATION_DIM = 155

ACTOR_RECEIVES_TIME_TO_GO = YES
CRITIC_RECEIVES_TIME_TO_GO = YES
CACHE_SCORER_RECEIVES_TIME_CONTEXT = YES
UAV_SCORER_RECEIVES_TIME_CONTEXT = YES

GAE_CHANGED = NO
BOOTSTRAP_CHANGED = NO
VALUE_TARGET_CHANGED = NO
REWARD_CHANGED = NO
MATCHING_CHANGED = NO
KKT_CHANGED = NO

V1_CHECKPOINT_PRESERVED = YES
V1_V2_SILENT_RESUME = NO

CRITIC_DIAGNOSTICS_LOGGED = YES
BEST_TRAIN_CHECKPOINT_ENABLED = YES
LONG_TRAINING_RUN = NO
```
