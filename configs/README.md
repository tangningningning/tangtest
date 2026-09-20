# Configuration index

TOML files are classified under `configs/` so current commands have stable,
purpose-specific paths.  `proposed_joint` is the only current proposed-method
configuration.  The stable comparison registry is
`configs/experiment_registry.toml` and its read-only display/validation
entrypoint is `scripts/experiment_registry.py`.

The registry is the source of stable method IDs, status, component flags,
training/evaluation protocol, and canonical command paths.  It does not execute
experiments.

## CURRENT MAINLINE

- `proposed/system_config.toml` — the single proposed mainline configuration
  with policy schema `adaptive_joint_cache_uav_ppo` and exact-Q listwise
  candidate improvement.
- `system_config.toml` — shared artifact-generation/system configuration,
  retained for existing artifact workflows.

## UNIFIED GENERATION / TRAINING / EVALUATION

The canonical generation entrypoints, `ppo-train`, and `ppo-eval` all use
`proposed/system_config.toml`.  The effective data protocol
is the full 300-slot randomized episode protocol: origins 0 through 299,
terminal origin 299, and no held-out test-origin range.  A field-by-field
comparison against the retained primitive episode-generation configuration found no
physical or episode-generation differences (`PENDING_CONFIG_MERGE = NO`).

`system_config.toml` remains separately available for existing artifact
workflows; it is not the PPO mainline configuration.

## BASELINES / ABLATIONS (NON-MAINLINE)

- `ablations/no_prediction.toml` — no-prediction cache-only ablation.
- `baselines/replica_aware_cache_only.toml` — replica-aware cache-only policy
  and explicit-cache baselines.
- `baselines/cache_only.toml` — cache-only policy comparison.

- `baselines/system_config_ppo_primitive_v6_horizon_grouped_separate_optimization_popart_actorlr1e4.toml`
  — retained earlier grouped primitive policy configuration for comparison.

## Stable method IDs

- `proposed_joint` — exact-Q listwise implementation gate passed; formal
  training is not yet completed.
- `ablation_cache_only` — NOT YET FORMALLY TRAINED TO 300; cache-only
  configuration and evaluator are available.
- `ablation_joint_no_prediction` — NOT YET FORMALLY TRAINED; a Joint
  no-prediction configuration is intentionally not aliased to the existing
  cache-only no-prediction ablation.
- `baseline_popularity_greedy` — implementation available.
- `baseline_true_static` — implementation available.

Final test seeds and historical LR sweep files are outside the proposed
mainline and are not executed by this cleanup pass.  Earlier development
versions are retained only in Git history and reports; comparison support
remains explicit under semantic baseline and ablation paths.

Changing a configuration's role here does not change its contents or the
provenance of any existing run.
