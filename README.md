# adaptive_cache_mec

This repository contains the adaptive cache and MEC simulation, the Recovery R2
PPO policy, frozen evaluation utilities, baselines, and reproducible
episode-generation utilities.  The directory layout separates command-line
orchestration from the reusable implementation in `src/`.

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
  evaluation/   Retained learned-policy comparison evaluation
  baselines/    Explicit Static and Prediction-Popularity Greedy evaluation
configs/        Recovery R2 plus baseline and artifact-support configs
reports/        Research reports and experiment records
runs/           Generated experiment artifacts; formal runs are not reorganized
```

The canonical generation entrypoints live under `scripts/generation/`.
Comparison evaluators remain under `scripts/evaluation/` and
`scripts/baselines/`; they are non-mainline compatibility/comparison tools.

## Current mainline: Recovery R2

```text
CURRENT_POLICY_SCHEMA =
adaptive_joint_cache_replica_aware_uav_operation_recovery_r2

CURRENT_CONFIG =
configs/proposed/system_config_ppo_recovery_r2_6e4.toml
```

The current operating chain is:

```text
Fixed SUMO network
  -> episode realization / episode pool
  -> Recovery R2 PPO training
  -> checkpoint
  -> evaluation
```

Recovery R2 is the current algorithm mainline.  Its bounded gate passed
(witness alignment 4/4, autograd direction 4/4, and no NaN/Inf); formal
training has not yet been completed.

The R2 credit contract is:

- Cache advantage = `GLOBAL_GAE`
- UAV operation advantage = `COUNTERFACTUAL_ONE_STEP_TD`
- UAV operation PPO ratio = `OPERATION_ONLY`
- Candidate credit = the original Recovery R1 path
- Operation advantage normalization = `STD_ONLY`
- No centered mean subtraction

The R2 policy is the standalone current learned policy schema.  It preserves
the verified parameter/module layout and checkpoint key set while using a
neutral shared base; V7/V8 and primitive baseline paths remain available as
separate baselines.

## Unified experiment registry

The stable comparison labels are registered in
`configs/experiment_registry.toml`.  The R2 mainline is recorded there under
the stable `proposed_joint` identifier; baseline and ablation entries remain
non-mainline comparison definitions:

| Method ID | Definition | Status |
| --- | --- | --- |
| `proposed_joint` | Recovery R2 prediction + adaptive Cache + adaptive UAV + learned policy | BOUNDED GATE PASSED; FORMAL TRAINING PENDING |
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

Generate an existing-protocol episode pool with the same Recovery R2
configuration used by training and evaluation:

```text
python scripts/generation/generate_episode_pool.py \
  --config configs/proposed/system_config_ppo_recovery_r2_6e4.toml \
  --output-dir runs/<episode-pool> --start-seed <seed-start> --count <count>
```

Train the current Recovery R2 mainline through the single PPO implementation:

```text
ppo-train --config configs/proposed/system_config_ppo_recovery_r2_6e4.toml \
  --epochs <epochs> --run-dir runs/<new-run> \
  --episode-pool runs/<episode-pool> --episode-seed-start <seed-start>
```

Evaluate a frozen Recovery R2 checkpoint on 20 episode realizations with 5
stochastic repeats per realization:

```text
ppo-eval \
  --config configs/proposed/system_config_ppo_recovery_r2_6e4.toml \
  --checkpoint runs/<training-run>/<checkpoint>.pt \
  --run-dir runs/<evaluation-run> \
  --episode-pool runs/<episode-pool> \
  --seed-start <seed-start> --episodes 20 --repeats 5 \
  --policy-action-mode stochastic --policy-sampling-seed <fixed-seed>
```

The evaluator loads the checkpoint through the same `PPOV2Trainer` policy
factory and checkpoint validation path as training.  Repeats are averaged
within each environment realization, then statistics are computed across the
20 environment means.  Formal training and final-test seed execution are
intentionally not claimed or performed in this cleanup pass.

The learned-policy comparison/ablation evaluators are under
`scripts/evaluation/`; the explicit Static and Prediction-Popularity Greedy
baselines are under `scripts/baselines/`.  See `configs/README.md` and the
reports directory for the retained configuration and experiment history.
