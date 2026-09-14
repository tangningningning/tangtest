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
configs/        Classified configurations and the minimal LR sweep set
reports/        Research reports and experiment records
runs/           Generated experiment artifacts; formal runs are not reorganized
```

The files directly under `scripts/` with the old names are compatibility
launchers.  The canonical implementations live in the category directories.

## Formal `proposed_joint`

- `FORMAL_METHOD_ID = proposed_joint`
- Formal base run: `runs/ppo_v9_joint_cache_uav_stage1_ep50_20001_20050`
- Frozen checkpoint: `runs/ppo_v9_joint_cache_uav_stage1_ep50_20001_20050/checkpoint_ep222_frozen.pt`
- The retained 1e-4 base run completed 300 epochs.
- Config: `configs/proposed/system_config_ppo_v9_joint_cache_uav.toml`
- Policy schema: `adaptive_joint_cache_replica_aware_uav_operation_v9_popart`
- Configuration control mode: `policy_primitive_v9_joint_cache_uav_operation`
- Policy mode: `stochastic`
- Stochastic repeats: `5` per environment, averaged within environment before
  aggregation across environments
- Development seeds: `30001..30020`
- Reserved final-test seeds: `40001..40020`

The reserved final-test pool has not been executed.  Development results are
not final-test data.

Implementation history: the V9 policy schema and control-mode strings are
retained internally for checkpoint compatibility.  V7/V8/V9 are historical
implementation labels, not new paper-facing method IDs.

## Learning-rate experiments

| Experiment ID | Actor learning rate | Status | Artifact |
| --- | ---: | --- | --- |
| `proposed_joint_lr_1e4` | `1e-4` | COMPLETED, 300 epochs | formal base run above |
| `proposed_joint_lr_1e3` | `1e-3` | COMPLETED, 300 epochs | `runs/lr_sweep_v9_actor_1e3` |
| `proposed_joint_lr_5e4` | `5e-4` | COMPLETED, 300 epochs | `runs/lr_sweep_proposed_joint_actor_5e4` |

The LR configuration paths and run manifests remain explicit so that
provenance continues to resolve.  The paper-facing formal method remains
`proposed_joint`; V9 is only an internal compatibility label.

## Unified experiment registry

The stable paper labels are registered in
`configs/experiment_registry.toml`:

| Method ID | Definition | Status |
| --- | --- | --- |
| `proposed_joint` | prediction + adaptive Cache + adaptive UAV + learned policy | FROZEN |
| `ablation_cache_only` | prediction + adaptive Cache + fixed UAV + learned policy | NOT YET FORMALLY TRAINED TO 222 |
| `ablation_joint_no_prediction` | no prediction + adaptive Cache + adaptive UAV + learned policy | NOT YET FORMALLY TRAINED |
| `baseline_popularity_greedy` | prediction/popularity Cache + fixed UAV + non-learning | IMPLEMENTATION AVAILABLE |
| `baseline_true_static` | fixed Cache + fixed UAV + non-learning | IMPLEMENTATION AVAILABLE |

Use the read-only registry entrypoint to validate or inspect definitions:

```text
python scripts/experiment_registry.py --validate
python scripts/experiment_registry.py --list
python scripts/experiment_registry.py --show proposed_joint
```

The `ablation_joint_no_prediction` row is deliberately not mapped to the
existing V8 Cache-only no-prediction configuration, because that configuration
does not have the Joint Cache + UAV definition recorded above.

## Main commands

Run commands from the repository root.  The package must be installed (or
`src` must be on `PYTHONPATH`).

Generate an existing-protocol episode pool:

```text
python scripts/generation/generate_episode_pool.py \
  --config configs/episode_generation/system_config_ppo_v6_full_episode_randomized.toml \
  --output-dir runs/<episode-pool> --start-seed <seed-start> --count <count>
```

Train through the single PPO implementation:

```text
ppo-train --config configs/proposed/system_config_ppo_v9_joint_cache_uav.toml \
  --epochs <epochs> --run-dir runs/<new-run> \
  --episode-pool runs/<episode-pool> --episode-seed-start <seed-start>
```

Evaluate a frozen Joint V9 policy on an existing pool:

```text
python scripts/evaluation/evaluate_v9_joint_episode_pool.py \
  --config configs/proposed/system_config_ppo_v9_joint_cache_uav.toml \
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
`scripts/baselines/`.  See `configs/README.md` and the reports directory for
the configuration and experiment history index.
