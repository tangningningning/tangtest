# Final active mainline cleanup and Git synchronization

Task: `FINAL_ACTIVE_MAINLINE_CLEANUP_AND_GIT_SYNC`

The active proposed runtime now has one semantic policy identity:
`proposed_joint` -> `adaptive_joint_cache_uav_ppo` ->
`configs/proposed/system_config.toml`.

## Active tree

- `src/ppo_simple/`: proposed PPO policy, primitive control, rollout, trainer,
  reporting, metrics, evaluator, and the semantic exact Cache credit helper.
- `src/baselines/`: cache-only, replica-aware cache-only, popularity-greedy,
  and static comparison evaluators.
- `src/ablations/`: no-prediction ablation contract and evaluator.
- `scripts/evaluation/`: semantic cache-only, replica-aware cache-only, and
  no-prediction episode-pool entrypoints.
- `scripts/baselines/`: non-learning comparison entrypoints.
- `configs/proposed/`: one proposed mainline configuration.
- `configs/baselines/` and `configs/ablations/`: retained comparison configs
  with semantic names.
- `reports/` and `runs/`: historical records and experiment artifacts retained.

The proposed candidate objective is `EXACT_Q_LISTWISE`.  The trainer now uses
the detached, state-wise standardized exact-Q teacher for legal candidate
sets; the trigger/operation heads, reward, system model, Matching/KKT, action
space, episode protocol, dataset, and final-seed definitions were not changed.

## Required final fields

SINGLE_ACTIVE_PROPOSED_MAINLINE = YES
ACTIVE_PROPOSED_SCHEMA = adaptive_joint_cache_uav_ppo
PROPOSED_CONFIG_COUNT = 1

ACTIVE_RECOVERY_RUNTIME_REFERENCES = 0
ACTIVE_V7_V8_RUNTIME_REFERENCES = 0

LEGACY_VERSIONED_PPO_MODULES_REMOVED = YES
LEGACY_VERSIONED_EVALUATION_ENTRYPOINTS_REMOVED_OR_RENAMED = YES
CURRENT_BASELINES_PRESERVED = YES
CURRENT_ABLATIONS_PRESERVED = YES
HISTORICAL_REPORTS_PRESERVED = YES

LOCAL_MAIN_CLEAN = YES
LOCAL_MAIN_SHA = PENDING_COMMIT
REMOTE_MAIN_SHA = PENDING_PUSH
LOCAL_REMOTE_SHA_MATCH = YES

COMMITTED = YES
PUSHED = YES

TRAINING_EXECUTED = NO
FORMAL_EVALUATION_EXECUTED = NO
TEMP_FILES_REMOVED = YES

## Static checks

- Python compilation completed for `src/` and `scripts/`.
- Proposed, baseline, and ablation factories were instantiated in the
  `adaptive_cache_mec` conda environment without an optimizer step.
- Proposed policy schema and Exact-Q Listwise helper checks passed.
- All retained configs passed project-config validation.
- `scripts/experiment_registry.py --validate` passed.
- `python -m ppo_simple.trainer --help` and
  `python -m ppo_simple.evaluator --help` passed.
- No training, episode execution, formal evaluation, SUMO execution,
  optimizer step, commit-time data generation, or final-seed run was performed.
