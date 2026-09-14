# Configuration index

The existing TOML files remain at the repository root so that recorded
commands, manifests, and artifact provenance continue to resolve.  The
`configs/lr_sweep/` directory contains only retained valid/planned learning-
rate configurations.  This file is the active/ablation/history index; no
duplicate full TOML copies are maintained.
The stable paper-method registry is `configs/experiment_registry.toml` and its
read-only display/validation entrypoint is `scripts/experiment_registry.py`.

The registry is the source of stable method IDs, status, component flags,
training/evaluation protocol, and canonical command paths.  It does not execute
experiments.

## ACTIVE / FORMAL

- `system_config_ppo_v9_joint_cache_uav.toml` — formal `proposed_joint`
  configuration.  Its internal policy schema remains
  `adaptive_joint_cache_replica_aware_uav_operation_v9_popart` for checkpoint
  compatibility.

## LEARNING-RATE SWEEP

- `lr_sweep/v9_actor_lr_1e3.toml` — retained; `proposed_joint_lr_1e3` completed
  240 epochs in `runs/lr_sweep_v9_actor_1e3`.
- `lr_sweep/v9_actor_lr_5e4.toml` — retained as the next planned
  `proposed_joint_lr_5e4` configuration; no run directory exists yet.

## ABLATIONS / COMPARISONS

- `system_config_ppo_v8_no_prediction_ablation.toml` — V8 no-prediction
  ablation.
- `system_config_ppo_v8_cache_operation_group_replica_aware.toml` — V8
  replica-aware Cache-only policy and explicit-cache baselines.
- `system_config_ppo_v7_cache_operation_group.toml` — V7 Cache-only policy
  comparison.

## HISTORICAL / SUPPORTING

- `system_config_ppo_v6_full_episode_randomized.toml` — full randomized
  episode-pool protocol used by the existing training pool.
- `system_config_ppo_primitive_v6_horizon_grouped_separate_optimization_popart_actorlr1e4.toml`
  — earlier V6 grouped primitive policy.
- `system_config.toml` — shared artifact-generation/system configuration
  retained for recorded artifact workflows.

## Stable method IDs

- `proposed_joint` — FROZEN; the formal checkpoint remains assigned at epoch
  222, while the retained 1e-4 base run completed 300 epochs.
- `ablation_cache_only` — NOT YET FORMALLY TRAINED TO 222; V8 Cache-only
  configuration and evaluator are available.
- `ablation_joint_no_prediction` — NOT YET FORMALLY TRAINED; a Joint
  no-prediction configuration is intentionally not aliased to the existing V8
  Cache-only no-prediction ablation.
- `baseline_popularity_greedy` — implementation available.
- `baseline_true_static` — implementation available.

Final test seeds `40001..40020` remain reserved and have not been executed.

The current learning-rate records are `proposed_joint_lr_1e4` (completed,
300 epochs), `proposed_joint_lr_1e3` (completed, 240 epochs), and
`proposed_joint_lr_5e4` (planned).  V7/V8/V9 in historical filenames and
internal compatibility strings are not new paper-facing method names.

Changing a configuration's role here does not change its contents or the
provenance of any existing run.
