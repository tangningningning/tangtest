# V10-C Action-Conditional Diagnostic Run Preparation

This preparation stage only validates and prepares two future diagnostic
launchers. No PPO training was started from Codex, and no 1e-3 run is included.

## Prepared runs

| run | config | actor LR | run directory |
|---|---|---:|---|
| 5e-4 | `configs/proposed/system_config_ppo_v10_branchwise_cache_uav.toml` | 0.0005 | `runs/ppo_v10c_instrumented_5e4` |
| 6e-4 | `configs/lr_sweep/v10_branchwise_actor_lr_6e4_kl.toml` | 0.0006 | `runs/ppo_v10c_instrumented_6e4` |

Both configurations were checked without modification. Shared values are:
critic LR `0.0003`, target KL `0.10`, clip epsilon `0.2`, four update epochs,
ten minibatches of size `30`, max grad norm `0.5`, entropy coefficient `0`,
gamma `1.0`, GAE lambda `0.95`, seed `4401`, and
`critic_value_scale_mode=popart_normalized`.

The fixed frozen pool is
`runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911`, with exactly 300
directories `episode_20001` through `episode_20300`. Required realization
artifacts were present for the full range. The launchers map epoch `e` to
`episode_20000+e` through `--episode-seed-start 20001`.

## Launchers

- [5e-4 launcher](C:/Users/tangning/Desktop/code/adaptive_cache_mec/scripts/run_v10c_instrumented_5e4.sh)
- [6e-4 launcher](C:/Users/tangning/Desktop/code/adaptive_cache_mec/scripts/run_v10c_instrumented_6e4.sh)

Each launcher uses bash `set -euo pipefail`, locates the repository from the
script location, exports `PYTHONPATH=<repo>/src`, checks Python/PyTorch/
`ppo_simple`, validates the fixed config and pool, and refuses a non-empty run
directory. It starts exactly 300 epochs with the frozen pool. It does not
resume, regenerate episodes, or use final test seeds. The two launchers have
different run directories and can be submitted independently or in parallel
on a server.

Example server commands, to be run only after activating the requested conda
environment:

```bash
cd /path/to/adaptive_cache_mec
conda activate adaptive_cache_mec
bash scripts/run_v10c_instrumented_5e4.sh
bash scripts/run_v10c_instrumented_6e4.sh
```

If resources permit, the final two commands may be submitted as two separate
jobs. The scripts themselves perform the preflight checks before starting PPO.

## Instrumentation readiness

The existing V10-C instrumentation remains in `src/ppo_simple/trainer.py` and
provides the requested per-epoch fields: MOVE/STAY counts, centered/std PPO
advantage means/medians/positive fractions, reward, q_total, movement cost,
joint UAV branch ratio, and clip fractions. The original Return, CacheChange,
UAVMove, joint/cache/UAV KL, clip, entropy, gradient, shared actor gradient,
and critic EV fields remain in the episode/update logs.

The action-conditional statistics are grouped by the verified physical
`actual_uav_move` flag. Selected UAV operation and selected Cache action are
also retained for control conditioning. All MOVE/STAY conditional means are
observational, not causal effects: the action groups can have different state
distributions.

## Local preparation checks

- Both launcher files passed `bash -n` syntax checks.
- Both config actor LRs and all shared PPO/data-protocol values matched the
  requested contract.
- The two target run directories were absent and therefore safe to start.
- The pool count and exact seed range were valid; all required artifacts were
  present.
- Required instrumentation field names were present in the trainer source.
- No training, evaluation, resume, episode generation, or final test seed was
  executed. Existing unrelated worktree changes were preserved.

## Final status

STAGE = V10-C_DIAGNOSTIC_RUN_PREPARATION

INSTRUMENTATION_READY = YES

RUN_5E4 = READY
RUN_6E4 = READY

CONFIG_5E4 = C:\Users\tangning\Desktop\code\adaptive_cache_mec\configs\proposed\system_config_ppo_v10_branchwise_cache_uav.toml
CONFIG_6E4 = C:\Users\tangning\Desktop\code\adaptive_cache_mec\configs\lr_sweep\v10_branchwise_actor_lr_6e4_kl.toml

ACTOR_LR_5E4 = 0.0005
ACTOR_LR_6E4 = 0.0006
CRITIC_LR_ALL = 0.0003
TARGET_KL_ALL = 0.10

EPISODE_POOL = C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v6_popart_actor1e4_lazy_episode_pool_20260911
EPISODE_COUNT = 300 (episode_20001..episode_20300)
SEED_MAPPING_VALID = YES

SCRIPT_5E4 = C:\Users\tangning\Desktop\code\adaptive_cache_mec\scripts\run_v10c_instrumented_5e4.sh
SCRIPT_6E4 = C:\Users\tangning\Desktop\code\adaptive_cache_mec\scripts\run_v10c_instrumented_6e4.sh

TARGET_EPOCHS_EACH = 300

TRAINING_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO

ALGORITHM_CHANGED = NO
REWARD_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO
CRITIC_CHANGED = NO

TEMP_FILES_REMOVED = YES
GIT_DIFF_CHECK = PASS
PUSHED = NO
