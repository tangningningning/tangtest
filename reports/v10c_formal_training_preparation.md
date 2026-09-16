# V10-C Formal Training Preparation

`STAGE = V10-C_FORMAL_TRAINING_PREPARATION`

## Frozen implementation and invariants

The V10-C actor uses separately allocated `cache_encoder` and `uav_encoder`
modules. The implementation report records zero parameter overlap, frozen
forward equivalence, and branch gradient-isolation gates as PASS. Cache loss
does not update the UAV encoder; UAV loss does not update the Cache encoder.

The system model, global objective, and PPO objective are unchanged. The V10-C
change is the actor representation split; the only training-sweep parameter
varied between these two formal configurations is actor learning rate. The
V10-B reference config is used by both launcher preflights to reject any other
configuration drift, including PopArt changes.

## Formal configurations

| Run | Config | Actor LR | Critic LR | Target KL | Epochs | Run directory |
|---|---|---:|---:|---:|---:|---|
| V10-C 5e-4 | `configs/proposed/system_config_ppo_v10c_decoupled_encoder.toml` | `0.0005` | `0.0003` | `0.10` | `300` | `runs/ppo_v10c_decoupled_encoder_5e4_formal` |
| V10-C 6e-4 | `configs/proposed/system_config_ppo_v10c_decoupled_encoder_6e4_formal.toml` | `0.0006` | `0.0003` | `0.10` | `300` | `runs/ppo_v10c_decoupled_encoder_6e4_formal` |

Shared settings: clip epsilon `0.2`, 4 update epochs, 10 minibatches, minibatch
size 30, gamma `1.0`, GAE lambda `0.95`, entropy coefficient `0`, experiment
seed `4401`, and the unchanged PopArt configuration. No additional training
hyperparameters are introduced.

## Frozen episode pool and launch protection

Both launchers use the existing pool
`runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911`, verified to contain
exactly 300 complete episode directories from `episode_20001` through
`episode_20300`. The trainer receives this sorted pool directly, so epoch 1
maps to `episode_20001` and epoch 300 maps to `episode_20300`. No pool creation
or regeneration is requested.

Launchers:

- `scripts/run_v10c_decoupled_encoder_5e4.sh`
- `scripts/run_v10c_decoupled_encoder_6e4.sh`

The shared read-only validator `scripts/validate_v10c_formal_preflight.py`
checks the configuration against V10-B apart from the explicitly permitted
schema, actor LR, and run-label fields; checks the V10-C schema implementation,
all requested hyperparameters, the full episode sequence and required files,
and the KL diagnostic fields; and refuses a non-empty run directory. The shell
launchers also fail before launch for an existing non-empty run directory and
do not pass resume, episode-generation, or seed-based lazy-pool options. Both
formal run directories were absent when this preparation was completed.

Preparation checks: `bash -n` passed for both launchers. The read-only
preflight passed for both configurations in the `adaptive_cache_mec` Conda
environment, validating the config comparison, V10-C schema, required files
for all 300 pool episodes, KL instrumentation, joint gate, and absent run
directories. No trainer command was invoked.

## KL diagnostics and planned comparison

The existing joint target-KL stop behavior remains in force. The added
instrumentation records Cache, UAV, and joint approximate KL; Cache-only,
UAV-only, both, and neither trigger counts; and available cross-suppression
attribution. Cross-suppression fields are descriptive only. No branch-specific
KL gate is introduced.

The later comparison is V10-B 5e-4 and 6e-4 versus V10-C 5e-4 and 6e-4, using
Best Return, Final Return, Final20 mean, Final60 mean, MA20 max, MA50 max,
CacheChange, UAVMove, Critic EV, Cache/UAV/joint KL, trigger counts, and NaN/Inf
checks. Analysis will emphasize late-stage stability and whether the prior
5e-4 STAY collapse or 6e-4 MOVE saturation persists. Return/global objective is
the primary criterion; UAVMove near 50% is not a success target. A high-STAY
policy can be acceptable if Return is higher and system cost is lower.

This stage only prepares and freezes the formal launchers/configurations. It
does not train, evaluate, resume a checkpoint, generate episodes, or use final
test seeds.

```text
SYSTEM_MODEL_CHANGED = NO
GLOBAL_OBJECTIVE_CHANGED = NO
PPO_OBJECTIVE_CHANGED = NO
KL_BEHAVIOR_CHANGED = NO
KL_INSTRUMENTATION_READY = YES
TRAINING_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
```
