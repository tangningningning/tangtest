# PREPARE_FORMAL_300_EPOCH_TRAINING_COMMAND_ONLY

## Status

This is a command-preparation and static-entrypoint gate only.

```text
FORMAL_TRAINING_STARTED = NO
FORMAL_300_EPOCHS_EXECUTED = NO
FORMAL_RUNTIME_BENCHMARK_EXECUTED = NO
REALIZATIONS_20011_20300_GENERATED = NO
DEV_SEEDS_USED = NO
FINAL_TEST_SEEDS_USED = NO
COMMITTED = NO
PUSHED = NO
```

No formal epoch, SUMO generation, performance benchmark, parameter sweep, or
evaluation was started in this task.

## Frozen source and configuration

```text
SOURCE_GIT_SHA = c82f16b76b4698d1a7c2982c3785fcf167f3f0ef
SOURCE_GIT_DIRTY = YES
SOURCE_DIFF_SHA256 = E4B28C0F9A4D50720F2C98F316B743DA24D291C231B08D8416A2141D4A91B9AC
FORMAL_CONFIG = configs/proposed/system_config.toml
FORMAL_CONFIG_SHA256 = 0E5D7FEC790CA08F6BF1803EDC84D66296A0CA0B40AE5C3865012DE9033556BC
SOURCE_FREEZE_PATCH = reports/formal_training_source_freeze.patch
SOURCE_FREEZE_MANIFEST = reports/formal_training_source_freeze_manifest.json
```

The source patch was generated with Git's native binary output and passed
`git apply --check --reverse`; it is recoverable from the source commit plus
the recorded patch.

## Entrypoint audit

The current real mainline entrypoint is `src/ppo_simple/trainer.py`, exposed by
`pyproject.toml` as `ppo-train = "ppo_simple.trainer:main"`. The exact Conda
environment interpreter imported the local package successfully, and
`python -m ppo_simple.trainer --help` passed. No `train.py` or
`formal_train.py` was created.

The existing `run_training()` path already implements the required protocol:

```text
GENERATE-IF-MISSING -> VERIFY -> TRAIN ONE OUTER EPOCH -> SAVE -> NEXT SEED
```

For an existing episode directory, `_ensure_episode_realization()` validates
the required artifacts, `episode_manifest.json`, episode seed, generation
fingerprint, source identity, frozen source artifacts, and every generated
artifact hash. Any failure raises and refuses regeneration or overwrite. For a
missing directory, generation is allowed only when
`--generate-missing-episodes` is present; the generated directory is then
validated before that epoch's rollout starts.

The loop maps `episode_seed_start + next_pool_index` to the next outer epoch.
The terminal update increments `completed_episodes` and `next_pool_index`, so
with `episode_seed_start=20001` the mapping is epoch 1 -> 20001 through epoch
300 -> 20300. Resume validates the checkpoint seed cursor and starts at
`completed_epochs + 1` without replaying completed epochs.

Fresh initialization constructs a new `PPOV2Trainer`; checkpoint loading occurs
only when `--resume` is explicitly supplied. A fresh run directory is required
to be clean. The checkpoint is saved immediately after each PPO update, before
the episode metrics and run summary are appended.

## Canonical realization pool

```text
TRAINING_REALIZATION_POOL = data/episode_pools/proposed_joint_training
VERIFIED_EXISTING_EPISODES = episode_20001 ... episode_20010
EXACT_EXISTING_POOL_COUNT = 10
EPISODES_20011_20300_PRESENT = NO
POOL_PROVENANCE = PASS for episode_20001 ... episode_20010
OLD_PPO_V6_EPISODE_POOL_USED = NO
```

The existing ten episodes were verified by the repository's own pool
validator. They were all skipped as valid existing realizations; none was
regenerated or overwritten. The formal command below is the only prepared
path for later lazy creation of 20011 ... 20300.

## Frozen algorithm invariants

```text
POLICY_SCHEMA = adaptive_joint_cache_uav_ppo
CONTROL_MODE = adaptive_joint_cache_uav_reconfiguration
CACHE_POLICY_CREDIT_SOURCE = ACTUAL_ROLLOUT_GAE
CACHE_ACTION_Q_CRITIC_INITIALIZED = NO
CACHE_ACTION_Q_USED_FOR_POLICY_CREDIT = NO
CACHE_COUNTERFACTUAL_ENV_EXECUTIONS = 0
UAV_OPERATION_TEACHER = FULL_INFORMATION_EXACT_Q_TEACHER
UAV_CANDIDATE_TEACHER = EXACT_Q_LISTWISE
UAV_EXACT_Q_MEMOIZATION_ENABLED = YES
UAV_TEACHER_UNCHANGED = YES
```

The slot-local Exact-Q memo is in the production trainer path and is retained.
No system model, reward, Matching, KKT, Cache PPO, teacher, candidate set or
ordering, architecture, learning rate, KL, entropy, loss coefficient, seed
protocol, or generation rule was changed by this preparation task.

## Formal run directory

No existing canonical `proposed_joint` 300-epoch run directory was defined in
the current registry or working tree. The prepared independent directory is:

```text
runs/proposed_joint_formal_300_epoch
```

It was confirmed absent and is distinct from both gate directories:
`runs/proposed_joint_short_stability_gate` and
`runs/uav_exact_q_equivalence_acceleration_gate`.

## Windows local commands

Run from the repository root. This is the fresh formal command; Codex did not
execute it:

```text
Set-Location 'C:\Users\tangning\Desktop\code\adaptive_cache_mec'
conda run -n adaptive_cache_mec python -m ppo_simple.trainer --config configs/proposed/system_config.toml --epochs 300 --run-dir runs/proposed_joint_formal_300_epoch --episode-pool data/episode_pools/proposed_joint_training --episode-seed-start 20001 --generate-missing-episodes --episode-generation-device cpu
```

After an interruption, use the same target and pool with `--resume`; this
loads `runs/proposed_joint_formal_300_epoch/latest_checkpoint.pt` and continues
from its verified `completed_epochs` cursor:

```text
Set-Location 'C:\Users\tangning\Desktop\code\adaptive_cache_mec'
conda run -n adaptive_cache_mec python -m ppo_simple.trainer --config configs/proposed/system_config.toml --epochs 300 --run-dir runs/proposed_joint_formal_300_epoch --episode-pool data/episode_pools/proposed_joint_training --episode-seed-start 20001 --generate-missing-episodes --episode-generation-device cpu --resume
```

The current host's `conda run` wrapper itself reported a Conda permissions
error while probing the environment directory. The exact interpreter at
`C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe` was used only
for the allowed import/help/static checks; no training command was substituted
or executed.

## Server command

No server, Slurm, `sbatch`, or Singularity launcher exists in the current local
working tree. Therefore no server command is emitted: inventing a remote
launcher would violate the task's requirement to use an existing project
entrypoint. The two commands above are the only verified commands prepared by
this task.

## Verification and cleanup

```text
REGISTRY_VALID = YES
TRAINING_ENTRYPOINT_HELP = PASS
POOL_PROVENANCE_VERIFICATION = PASS
SOURCE_PATCH_REVERSE_CHECK = PASS
FORMAL_RUN_DIR_CREATED = NO
TEMP_FILES_CLEANED = YES
GIT_DIFF_CHECK = PASS
```

Only the requested static/import/help/registry/provenance checks were used.
No commit or push was performed.
