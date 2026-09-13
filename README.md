# adaptive_cache_mec

This repository contains the adaptive cache and MEC simulation, PPO policies,
frozen evaluators, baselines, and reproducible episode-generation utilities.
The directory layout separates command-line orchestration from the reusable
implementation in `src/`.

## Project structure

```text
src/
  common/       Shared configuration and system-model helpers
  scenario/     Scenario, trajectory, task, and prediction generation
  decision/     Cache, UAV, matching, KKT, and slot execution logic
  ppo_simple/   PPO policies, rollout, training, metrics, and evaluators
  analysis/     Read-only analysis utilities
scripts/
  training/     Training-entrypoint notes; implementation is src/ppo_simple/trainer.py
  generation/   Episode and project-artifact generation
  evaluation/   Frozen learned-policy evaluation
  baselines/    Explicit Static and Prediction-Popularity Greedy evaluation
  diagnostics/  Posthoc diagnostics and audits
archive/
  deprecated_scripts/  Historical command-line implementations kept for reference
configs/        Configuration index; active TOMLs remain at the repository root
reports/        Research reports and experiment records
runs/           Generated experiment artifacts; formal runs are not reorganized
```

The files directly under `scripts/` with the old names are compatibility
launchers.  The canonical implementations live in the category directories.

## Formal Joint V9

- Checkpoint: `runs/ppo_v9_joint_cache_uav_stage1_ep50_20001_20050/checkpoint_ep222_frozen.pt`
- Config: `system_config_ppo_v9_joint_cache_uav.toml`
- Policy schema: `adaptive_joint_cache_replica_aware_uav_operation_v9_popart`
- Policy mode: `stochastic`
- Stochastic repeats: `5` per environment, averaged within environment before
  aggregation across environments
- Development seeds: `30001..30020`
- Reserved final-test seeds: `40001..40020`

The reserved final-test pool has not been executed.  Development results are
not final-test data.

## Main commands

Run commands from the repository root.  The package must be installed (or
`src` must be on `PYTHONPATH`).

Generate an existing-protocol episode pool:

```text
python scripts/generation/generate_episode_pool.py \
  --config system_config_ppo_v6_full_episode_randomized.toml \
  --output-dir runs/<episode-pool> --start-seed <seed-start> --count <count>
```

Train through the single PPO implementation:

```text
ppo-train --config system_config_ppo_v9_joint_cache_uav.toml \
  --epochs <epochs> --run-dir runs/<new-run> \
  --episode-pool runs/<episode-pool> --episode-seed-start <seed-start>
```

Evaluate a frozen Joint V9 policy on an existing pool:

```text
python scripts/evaluation/evaluate_v9_joint_episode_pool.py \
  --config system_config_ppo_v9_joint_cache_uav.toml \
  --checkpoint runs/ppo_v9_joint_cache_uav_stage1_ep50_20001_20050/checkpoint_ep222_frozen.pt \
  --run-dir runs/<evaluation-run> --episode-pool runs/<episode-pool> \
  --seed-start <seed-start> --episodes <count> \
  --policy-action-mode stochastic --policy-sampling-seed <fixed-seed>
```

For the formal stochastic protocol, use five fixed sampling passes and first
average their metrics within each environment realization.  Do not treat the
five repeats as additional independent environments.

The learned-policy ablations are under `scripts/evaluation/`; the explicit
Static and Prediction-Popularity Greedy baselines are under
`scripts/baselines/`.  See `configs/README.md` and
`archive/README.md` for the configuration and history index.
