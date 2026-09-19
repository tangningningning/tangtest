# R2 Branch-Specific KL Both-Inactive Fix

**Task:** `R2_BRANCH_SPECIFIC_KL_BOTH_INACTIVE_FIX`  
**Date:** 2026-09-19  
**Repository:** `C:\Users\tangning\Desktop\code\adaptive_cache_mec`

## Decision

```text
STATE_MACHINE_GATE = PASS
BOUNDED_PPO_UPDATE = PASS
BRANCH_SPECIFIC_KL_EXCEPTION = NOT_REPRODUCED_AFTER_FIX
```

The failure is a branch-specific KL state-machine bug, not a rollout, initialization, resume, reward, movement-cost, or model-parameter issue.

```text
ROOT_CAUSE_CATEGORY = B + D
PRIMARY_PATH = TT -> TF -> FF -> FF
```

`B` identifies the sequential independent-stop path: UAV stops first and Cache stops in a later epoch. `D` identifies the immediate state-machine defect: the next epoch calls the transition function with both branches already inactive, but the old function rejects that legal absorbing state.

## Static root-cause trace

The fresh deterministic replay used seed `4401`, the `w_move=0.50` development configuration, and frozen `episode_20001`. It completed one 300-slot rollout and reproduced the old exception before any source change:

| transition call | Cache active | UAV active | Cache epoch KL | UAV epoch KL | resulting state under intended semantics |
|---:|:---:|:---:|---:|---:|:---:|
| update epoch 1 | T | T | 0.0767943825 | 0.3742147901 | `TF` (UAV stops) |
| update epoch 2 | T | F | 0.1007544860 | 0.7384015650 | `FF` (Cache stops) |
| update epoch 3 | F | F | 0.0429593435 | 0.7384015560 | `FF` (old code raised here) |

The target KL was `0.10`. The old `_branch_specific_kl_gate_transition` raised `ValueError("branch-specific KL gate cannot transition from both inactive")` on the third call. The replay had already executed 30 actor and 30 critic optimizer steps (three epochs × ten minibatches), including actor optimizer steps in the already-both-inactive epoch. This confirms both the hard exception and the stale/no-gradient actor-step defect.

## Minimal implementation change

Changed only `src/ppo_simple/trainer.py`:

1. `(cache_active=False, uav_active=False)` is now accepted as a legal absorbing state.
2. `both_inactive_stop` is reported only when a transition newly enters `FF`; `FF -> FF` is not reported as a repeated stop event.
3. When both actor branches are inactive, inactive gradients must be absent and the actor optimizer step is skipped. The actor gradient norm is recorded as `0.0`.
4. The critic graph, finite checks, critic clipping, and critic optimizer step continue normally for every configured update epoch.

The existing active-branch loss weights, KL thresholds, reward/objective formulas, movement-cost formula, counterfactual credit, PPO hyperparameters, and model architecture were not changed.

## State-machine correctness gate

Trigger notation is `(Cache KL over target, UAV KL over target)`, and state notation is `(Cache active, UAV active)`.

| current state | `FF` | `TF` | `FT` | `TT` |
|:---:|:---:|:---:|:---:|:---:|
| `TT` | `TT` | `FT` | `TF` | `FF` |
| `TF` | `TF` | `FF` | `TF` | `FF` |
| `FT` | `FT` | `FT` | `FF` | `FF` |
| `FF` | `FF` | `FF` | `FF` | `FF` |

The pure transition gate passed all required cases without exception:

```text
TT -> FF: PASS
TF -> FF: PASS
FT -> FF: PASS
FF -> FF: PASS (absorbing; no repeated stop event)
```

## Bounded normal PPO update

The post-patch validation used exactly one fresh normal update. It did not use the failed run directory and did not write a checkpoint.

| item | value |
|---|---|
| config | `configs/proposed/system_config_ppo_recovery_r2_wmove_0p50_dev3.toml` |
| initialization | fresh actor, critic, and optimizers |
| warm start / resume | no |
| seed | `4401` |
| frozen realization | `runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911/episode_20001` |
| rollout | `300` transitions |
| PPO update epochs | `4 / 4` |
| observed active state | `TT -> TF -> FF -> FF` |
| Cache stop epoch | `2` |
| UAV stop epoch | `1` |
| Cache independent stop | `true` |
| UAV independent stop | `true` |
| both-inactive stop | `true` |
| actor optimizer steps | `20` |
| critic optimizer steps | `40` |
| global critic steps | `40` |
| actor gradient norm on final FF epoch | `0.0` |
| actor gradient norm range | `0.0 .. 1.3963108063` |
| critic gradient norm range | `0.2113406658 .. 5.7187705030` |
| final Cache KL | `0.0658668872` |
| final UAV KL | `0.6473548697` |
| gradient finite | `true` |
| returns finite | `true` |
| advantages finite | `true` |
| parameters finite | `true` |
| exact UAV counterfactual environment executions | `48343` |
| operation-credit runtime | `579.0459275 s` |

Per-epoch diagnostics were consistent with the required semantics:

```text
epoch 1: TT -> TF; UAV independent stop = true; both-inactive stop = false
epoch 2: TF -> FF; Cache independent stop = true; both-inactive stop = true
epoch 3: FF -> FF; actor inactive; critic runs; both-inactive stop = false
epoch 4: FF -> FF; actor inactive; critic runs; both-inactive stop = false
```

## Preservation and cleanup

```text
PRODUCTION_R2_CONFIG_CHANGED = NO
W_MOVE_CHANGED = NO
SYSTEM_MODEL_FORMULA_CHANGED = NO
REWARD_FORMULA_CHANGED = NO
PPO_HYPERPARAMETERS_CHANGED = NO
MODEL_ARCHITECTURE_CHANGED = NO
FAILED_DEV3_RUN_OVERWRITTEN = NO
DEV3_RERUN = NO
EPISODE_REGENERATION = NO
SUMO_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO
```

The previous failed directory `runs/ppo_recovery_r2_wmove_0p50_dev3_20260919` was preserved read-only. Only this report and the minimal trainer fix belong to this task.
