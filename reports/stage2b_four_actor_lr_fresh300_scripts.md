# Stage2B four actor-LR fresh300 runners

## Formal execution contract

```text
TRAINING_START_MODE = FRESH_INITIALIZATION
SOURCE_CHECKPOINT_USED = NO
START_EPOCH = 1
TARGET_EPOCH = 300
EPISODE_POOL = 20001–20300
EPISODE_MAPPING = epoch 1 -> episode_20001, ..., epoch 300 -> episode_20300

ACTOR_LR = 1e-3, 5e-3, 1e-4, 5e-4
CRITIC_LR = 1e-3 for all four runs
COMMON_RANDOM_SEED = 4401
EXACT_Q_PARALLEL_WORKERS = 4

CACHE_RATIO_GRANULARITY = PER_CACHE_HEAD
CACHE_TRIGGER_CREDIT = ONE_STEP_TD_Q_DIFFERENCE
CACHE_CANDIDATE_CREDIT = GLOBAL_GAE
TDQ_BOOTSTRAP = PLUS
TDQ_NORMALIZATION = std-only, no centering
UAV_STAGE1B_KL_STATISTIC = MEAN_OR_P90

GPU_REQUIRED = NO
CPU_TRAINING_SUPPORTED = YES
RESUME = False
OLD_EPOCH99_DEPENDENCY_REMOVED = YES
```

The four independent output directories are:

```text
runs/stage2b_fresh300_actor_lr_1e_minus_3
runs/stage2b_fresh300_actor_lr_5e_minus_3
runs/stage2b_fresh300_actor_lr_1e_minus_4
runs/stage2b_fresh300_actor_lr_5e_minus_4
```

Each runner creates a fresh config, constructs a CPU policy from the common
seed before any rollout/backward/optimizer step, records the initial hashes,
and invokes the trainer without `--resume`. The first-run preflight log is:

```text
Completed Epochs: 0
Next Epoch: 1
Target Epochs: 300
```

## Initialization fairness validation

All four variants produced identical hashes:

```text
ACTOR_INITIAL_STATE_SHA256 = 53f3c9e323c0c69e5ddeac181fd5bd55c68560764f0d754782c8cf172c6ecf5d
CRITIC_INITIAL_STATE_SHA256 = a5e6d1b7efd16a41398ddeffacc83637b7de0bda476690f076294fac99676c86

SAME_INITIAL_ACTOR_PARAMETERS = YES
SAME_INITIAL_CRITIC_PARAMETERS = YES
```

The actor LR is applied through the generated configuration and optimizer
construction; it does not alter the freshly initialized policy parameters.

## Source and execution audit

```text
OLD_EPOCH99_DEPENDENCY_REMOVED = YES
TRAINING_EXECUTED = NO
ROLLOUT_EXECUTED = NO
OPTIMIZER_STEP_CALLS = 0
PY_COMPILE = PASS
INITIALIZATION_HASH_VALIDATION = PASS
GIT_DIFF_CHECK = PASS
COMMIT = NO
PUSH = NO
```
