# Legacy PPO code cleanup

## Scope

Formal PPO training now uses the standalone standard PPO v2 path:

```text
state -> standard PPO -> Cache/UAV binary request -> existing environment -> actual G(t)
```

Environment mathematics, Cache/UAV optimizers, Matching/KKT, and reward
calculation were not changed.

## Removed files

The following legacy modules were deleted:

- `src/ppo/ppo_training.py`
- `src/ppo/ppo_update.py`
- `src/ppo/ppo_rollout.py`
- `src/ppo/ppo_trigger_policy.py`
- `src/ppo/hierarchical_reconfiguration_gate.py`
- `src/ppo/causal_artifacts.py`
- `src/ppo/paired_policy_evaluation.py`
- `src/ppo/reference_drift.py`

`src/ppo` is now absent. Historical reports and generated formal artifacts
were retained.

## Retained and extracted PPO v2 code

The active path is `src/ppo_simple/`:

- `policy.py`: independent Bernoulli actor-critic;
- `rollout.py`: environment rollout and standard GAE;
- `trainer.py`: clipped PPO update, checkpointing, resume, and CLI;
- `interface.py`: the minimal environment-facing interface.

`interface.py` contains `TriggerActionSpec`,
`ConfigurationObservationSpec`, `build_configuration_observation`,
`build_trigger_action_mask_provider`, `AlignedRolloutStep`,
`prepare_aligned_rollout_steps`, and only their direct helpers.
`_copy_uav_positions()` is defined once in `interface.py` and imported by
both PPO v2 consumers.

Static checks confirm that `src/ppo_simple` has no import of `src/ppo`.
GAE, clipped PPO update, and the Bernoulli actor-critic each remain in their
single PPO v2 module.

## Entrypoint and configuration

`pyproject.toml` now exposes:

```toml
ppo-train = "ppo_simple.trainer:main"
```

The package include was narrowed to `ppo_simple*` for the PPO package.
`system_config.toml` now uses `method = "ppo_simple_v2"`.

The following legacy-only PPO fields were removed from the formal config and
validation: `credit_assignment`, `actor_architecture`, `q_target`,
`credit_phase`, `cf_samples_per_episode`, `cf_batch_episodes`,
`cf_parallel_workers`, `reference_probability_drift_cap`,
`cf_sampling_strategy`, and `autoregressive_hidden_size`.

The standard PPO v2 parameters were retained without changing their values:
`gamma=1.0`, `gae_lambda=0.95`, `clip_epsilon=0.2`,
`learning_rate=3.0e-4`, `value_loss_coefficient=0.5`,
`entropy_coefficient=0.0`, `max_grad_norm=0.5`, `update_epochs=1`,
`num_minibatches=4`, and `hidden_sizes=[128, 128]`.

The prediction-worker fallback now uses the execution-only configuration
field and defaults to one worker; it no longer reads the deleted legacy PPO
worker field.

## Archive and protected files

`.gitignore` now includes `adaptive_cache_mec.zip`. The archive remains present
and tracked because the requested deletion was blocked by the local safety
guard for a large, difficult-to-recover file. `打包并上传.bat` was not touched.

## Minimal validation

- `python -m ppo_simple.trainer --help`: PASS using the project Python
  environment (`C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`).
- PPO v2 imports and syntax compilation: PASS.
- Formal config validation: PASS with `method=ppo_simple_v2`.
- Static dependency check: PASS; no active legacy PPO import remains in source.
- Existing real four-origin fixture: PASS; 4 transitions, one PPO update with
  4 minibatch optimizer steps, finite loss/gradients, and checkpoint save.
  The bounded fixture used neutral requests to avoid entering the long
  configuration path.
- `git diff --check`: PASS; Git emitted only its existing LF/CRLF
  normalization warnings.
- Temporary Python cache directories and bytecode were removed.

No commit or push was performed.

```text
LEGACY_PPO_REMOVED = YES
PPO_SIMPLE_STANDALONE = YES
```
