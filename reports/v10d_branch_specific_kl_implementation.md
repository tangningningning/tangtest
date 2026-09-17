# V10-D branch-specific KL gate implementation

STAGE = V10-D_BRANCH_SPECIFIC_KL_GATE_IMPLEMENTATION

## Scope and decision

This stage implements the smallest structural change requested after the V10-C evidence review: a branch-specific KL gate for the decoupled Cache/UAV actor. The V10-D policy is a new schema and a subclass of the V10-C policy, so V10-C remains a separately runnable and loadable schema. The system model, reward/cost calculation, observation/action spaces, Matching/KKT path, GAE, PopArt, advantage normalization, PPO ratio/clipping, learning-rate values, and entropy coefficient were not changed.

The new configuration key is `algorithm.ppo.kl_gate_mode`. Existing V10-C configurations omit it and therefore retain the default `joint` behavior. The two new V10-D configurations set `kl_gate_mode = "branch_specific"` and are byte-for-byte equivalent to their corresponding V10-C configurations after accounting only for the experiment tag, the V10-D schema, and that new mode key.

## Implementation

### Policy and schema registration

`AdaptiveJointCacheReplicaAwareUAVOperationV10DBranchSpecificKLActorCritic` inherits the V10-C decoupled encoder implementation without adding or renaming policy modules. It therefore retains:

- independent `cache_encoder` and `uav_encoder` modules;
- no shared actor encoder (`encoder` is absent after V10-C construction);
- the same Cache/UAV operation heads, candidate scorers, critic, action hierarchy, and replica-aware candidate feature dimensions;
- disjoint actor gradient groups for Cache, UAV, Cache encoder, UAV encoder, and the empty shared group.

The new schema is registered in `project_config.py`, `trainer.py`, `primitive_control.py`, and the package export list. V10-D uses checkpoint version 13; V10-C continues to use version 12.

### Branch-specific gate

At the beginning of every `PPOV2Trainer.update()` call, both flags are initialized to active:

```text
cache_active = True
uav_active = True
```

After every completed PPO update epoch, the implementation computes the existing branch KL means and the joint KL mean. In V10-D branch-specific mode:

1. `cache_epoch_kl > target_kl` disables Cache for subsequent update epochs only.
2. `uav_epoch_kl > target_kl` disables UAV for subsequent update epochs only.
3. A branch that remains below target stays active, regardless of the other branch.
4. The remaining update epochs are stopped only after both flags become inactive.
5. The joint KL is retained in minibatch/epoch diagnostics but does not control the actor gate.

The target remains the single existing `target_kl` value (`0.10`); no branch-specific target hyperparameters were added.

### Loss and critic behavior

The branchwise PPO losses continue to use the V10-C/V10-B definitions and the exact same normalized rollout-wide GAE advantage. V10-D combines them as:

```text
both active:  0.5 * L_cache + 0.5 * L_uav
Cache only:   0.5 * L_cache
UAV only:     0.5 * L_uav
```

The single active branch is not re-normalized to coefficient 1.0. The entropy term and its configured coefficient are unchanged. The critic path is unchanged and executes normally for every update epoch that is not suppressed after both actor branches have stopped; the epoch that causes both branches to stop completes its critic update under the existing semantics.

## Diagnostics added or retained

V10-D update rows and epoch records retain the existing branch and joint KL values and add:

```text
cache_active_per_update_epoch
uav_active_per_update_epoch
cache_stop_epoch
uav_stop_epoch
cache_suppressed_update_epochs
uav_suppressed_update_epochs
joint_kl_diagnostic
cache_independent_stop
uav_independent_stop
both_inactive_stop
joint_kl_control_enabled
```

The legacy joint-gate fields remain present for compatibility. In V10-D rows, `joint_gate_triggered` is false for branch-only events and `joint_kl_control_enabled` is false. `joint_kl_diagnostic` remains true.

## Gate A: static KL control flow

The pure transition helper was tested with target `0.10`, four update epochs, and three remaining epochs after the test transition:

| Case | Cache KL | UAV KL | Next Cache | Next UAV | Independent stop | Both inactive stop |
|---|---:|---:|---:|---:|---|---|
| Cache only | 0.11 | 0.09 | inactive | active | Cache = yes | no |
| UAV only | 0.09 | 0.11 | active | inactive | UAV = yes | no |
| Both | 0.11 | 0.12 | inactive | inactive | neither | yes |
| Neither | 0.09 | 0.10 | active | active | neither | no |

The comparison is strict `>`; exactly `0.10` is therefore below the stop threshold. Each case correctly reports three subsequent epochs as suppressible. A sequential transition test also confirms that once one branch is inactive it does not restart, and the other branch can independently cause the final both-inactive stop.

`CACHE_KL_GATE_INDEPENDENT = PASS`

`UAV_KL_GATE_INDEPENDENT = PASS`

`BOTH_INACTIVE_GATE = PASS`

`NEITHER_GATE = PASS`

## Gate B: loss-scale parity

Using scalar branch losses `L_cache = 2.0` and `L_uav = 3.0`, the tested actor losses were:

| Active branches | Result | Active gradient coefficient |
|---|---:|---:|
| Cache + UAV | 2.5 | Cache 0.5, UAV 0.5 |
| Cache only | 1.0 | Cache 0.5; UAV inactive |
| UAV only | 1.5 | UAV 0.5; Cache inactive |

The dual-active result is identical to the previous `0.5 * (L_cache + L_uav)` expression. The single-active results preserve the required `0.5` scale and do not backpropagate the inactive branch loss.

`DUAL_BRANCH_LOSS_PARITY = PASS`

`SINGLE_BRANCH_LOSS_SCALE_PRESERVED = PASS`

## Gate C: one-epoch runtime gate

Each new V10-D configuration ran exactly one complete `full_episode_randomized` epoch using the existing `episode_20001` realization. Each run used 300 slots, four PPO update epochs, and ten minibatches per update. No counterfactual evaluation, final-test seed, long training, or missing-episode generation was run.

| Run | Actor LR | Return | Mean joint KL | Mean Cache KL | Mean UAV KL | Active flags | Actor steps | Critic steps | NaN/Inf |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| V10-D 5e-4 | 0.0005 | -989.049157 | 0.0207084 | 0.0504446 | 0.00547668 | T,T,T,T / T,T,T,T | 40 | 40 | no |
| V10-D 6e-4 | 0.0006 | -989.049157 | 0.0327845 | 0.0520356 | 0.00790952 | T,T,T,T / T,T,T,T | 40 | 40 | no |

Both runs recorded:

```text
kl_gate_mode = branch_specific
joint_kl_control_enabled = false
joint_kl_diagnostic = true
cache_stop_epoch = null
uav_stop_epoch = null
cache_suppressed_update_epochs = 0
uav_suppressed_update_epochs = 0
joint_gate_triggered = false
kl_early_stop_triggered = false
```

The one-epoch runtime gate did not produce an actual independent stop because all observed branch KL means were below `0.10`. Independent stop behavior is therefore established by the static Gate A transition test, while the runtime gate establishes schema construction, decoupled encoder execution, diagnostic serialization, finite updates, normal critic updates, and checkpoint creation. Both checkpoints used version 13 and the exact V10-D schema.

An existing V10-C formal checkpoint was then loaded read-only with its original V10-C configuration. It loaded successfully as checkpoint version 12 with `kl_gate_mode = joint` and distinct Cache/UAV encoders, confirming that V10-C was not overwritten.

## Invariants and execution status

```text
STAGE = V10-D_BRANCH_SPECIFIC_KL_GATE_IMPLEMENTATION
SYSTEM_MODEL_CHANGED = NO
GLOBAL_OBJECTIVE_CHANGED = NO
PPO_SURROGATE_OBJECTIVE_CHANGED = NO
ENCODER_DECOUPLING_PRESERVED = YES
CACHE_KL_GATE_INDEPENDENT = PASS
UAV_KL_GATE_INDEPENDENT = PASS
JOINT_KL_CONTROL_DISABLED = YES
JOINT_KL_DIAGNOSTIC_PRESERVED = YES
CACHE_ONLY_STOP_TEST = PASS
UAV_ONLY_STOP_TEST = PASS
BOTH_STOP_TEST = PASS
NEITHER_STOP_TEST = PASS
DUAL_BRANCH_LOSS_PARITY = PASS
SINGLE_BRANCH_LOSS_SCALE_PRESERVED = PASS
CRITIC_BEHAVIOR_CHANGED = NO
ONE_EPOCH_GATE = PASS
NAN_INF_DETECTED = NO
TRAINING_EPOCHS_EXECUTED = 1
LONG_TRAINING_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
REPORT = reports/v10d_branch_specific_kl_implementation.md
GIT_DIFF_CHECK = PASS
TEMP_FILES_REMOVED = YES
PUSHED = NO
```

No commit or push was performed.
