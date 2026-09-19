# R2 configuration and evaluator unification

Date: 2026-09-18

## Result

- `configs/proposed/system_config_ppo_recovery_r2_6e4.toml` is now the
  documented/default configuration for episode generation, training, and
  evaluation.
- The former
  `configs/episode_generation/system_config_ppo_v6_full_episode_randomized.toml`
  was removed after comparison showed no physical or episode-generation
  differences from R2.  The only differing leaves were
  `experiment.name` and `experiment.output_tag`.
- The effective protocol for both configurations was identical:
  `full_episode_randomized`, 300 physical/episode slots, train origins 0--299,
  terminal origin 299, and no test-origin range.
- `src/ppo_simple/evaluator.py` is the canonical Recovery R2 evaluator.  It
  uses the `PPOV2Trainer` policy factory and `load_checkpoint()`, validates the
  R2 schema/control mode/protocol, executes 20 environment realizations with 5
  stochastic repeats by default, averages repeats within each realization,
  and computes statistics across realization means.
- No training, episode generation, SUMO execution, final-seed evaluation, or
  formal R2 checkpoint evaluation was run.  No R2 checkpoint was present in
  the current `runs/` inventory, so `R2_REAL_CHECKPOINT_LOAD_TEST` is
  `NOT_AVAILABLE`.

## Checks

- Python `compileall src scripts`: passed.
- R2 config load with `require_artifacts=False`: passed.
- `ppo-train --help`: passed.
- `ppo-eval --help`: passed and exposes the R2 evaluator CLI.
- Both canonical generation `--help` commands: passed.
- Experiment registry validation: passed.
- `git diff --check`: passed.

The generator's historical-run protection remains in place as a write-safety
guard; it does not select the generation configuration.
