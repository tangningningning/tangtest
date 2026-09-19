# adaptive_cache_mec

This repository contains the adaptive cache and MEC simulation, the proposed
joint PPO policy, frozen evaluation utilities, baselines, and reproducible
episode-generation utilities.  The directory layout separates command-line
orchestration from the reusable implementation in `src/`.

## Project structure

```text
src/
  common/       Shared configuration and system-model helpers
  scenario/     Scenario, trajectory, task, and prediction generation
  decision/     Cache, UAV, matching, KKT, and slot execution logic
  ppo_simple/   Proposed PPO policy, rollout, training, metrics, and core evaluator
  baselines/    Semantic cache-only and non-learning baseline evaluators
  ablations/    Semantic ablation contracts and evaluators
  analysis/     Read-only analysis utilities
scripts/
  training/     Training-entrypoint notes; implementation is src/ppo_simple/trainer.py
  generation/   Episode and project-artifact generation
  evaluation/   Retained learned-policy comparison evaluation
  baselines/    Explicit Static and Prediction-Popularity Greedy evaluation
configs/        The proposed mainline plus baseline and artifact-support configs
reports/        Research reports and experiment records
runs/           Generated experiment artifacts; formal runs are not reorganized
```

The canonical generation entrypoints live under `scripts/generation/`.
Comparison evaluators remain under `scripts/evaluation/` and
`scripts/baselines/`; they are non-mainline compatibility/comparison tools.

## Current mainline: proposed_joint

```text
CURRENT_POLICY_SCHEMA =
adaptive_joint_cache_uav_ppo

CURRENT_CONFIG =
configs/proposed/system_config.toml
```

The current operating chain is:

```text
Fixed SUMO network
  -> episode realization / episode pool
  -> proposed_joint PPO training
  -> checkpoint
  -> evaluation
```

`proposed_joint` is the only production proposed schema.  The exact-Q
listwise candidate correctness gate passed; formal training has not been run
by this consolidation task.

The proposed-joint credit contract is:

- Cache trigger credit = exact node counterfactual Q
- UAV operation advantage = `COUNTERFACTUAL_ONE_STEP_TD`
- UAV operation PPO ratio = `OPERATION_ONLY`
- Cache and UAV candidate objective = `EXACT_Q_LISTWISE`
- Candidate teacher = state-wise standardized exact Q softmax over all legal candidates

The proposed policy is the standalone current learned policy schema.  It preserves
the verified parameter/module layout and checkpoint key set while using a
neutral shared base.  Comparison implementations live under semantic
`src/baselines/` and `src/ablations/` namespaces.

## Unified experiment registry

The stable comparison labels are registered in
`configs/experiment_registry.toml`.  The proposed mainline is recorded there under
the stable `proposed_joint` identifier; baseline and ablation entries remain
non-mainline comparison definitions:

| Method ID | Definition | Status |
| --- | --- | --- |
| `proposed_joint` | exact-Q listwise candidate prediction + adaptive Cache + adaptive UAV + learned policy | IMPLEMENTATION GATE PASSED; FORMAL TRAINING PENDING |
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
cache-only no-prediction configuration, because that configuration does not
have the Joint Cache + UAV definition recorded above.

## Main commands

Run commands from the repository root.  The package must be installed (or
`src` must be on `PYTHONPATH`).

Generate an existing-protocol episode pool with the same proposed-joint
configuration used by training and evaluation:

```text
python scripts/generation/generate_episode_pool.py \
  --config configs/proposed/system_config.toml \
  --output-dir runs/<episode-pool> --start-seed <seed-start> --count <count>
```

Train the current proposed mainline through the single PPO implementation:

```text
ppo-train --config configs/proposed/system_config.toml \
  --epochs <epochs> --run-dir runs/<new-run> \
  --episode-pool runs/<episode-pool> --episode-seed-start <seed-start>
```

Evaluate a frozen proposed-joint checkpoint on 20 episode realizations with 5
stochastic repeats per realization:

```text
ppo-eval \
  --config configs/proposed/system_config.toml \
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
