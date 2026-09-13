# Configuration index

The existing TOML files remain at the repository root so that recorded
commands, manifests, and artifact provenance continue to resolve.  This file
is the active/ablation/history index; no duplicate TOML copies are maintained.

## ACTIVE / FORMAL

- `system_config_ppo_v9_joint_cache_uav.toml` — current formal Joint V9
  configuration for `adaptive_joint_cache_replica_aware_uav_operation_v9_popart`.

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

Changing a configuration's role here does not change its contents or the
provenance of any existing run.
