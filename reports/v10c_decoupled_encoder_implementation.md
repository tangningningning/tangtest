# V10-C Decoupled Encoder Implementation

`STAGE = V10-C_ENCODER_DECOUPLING_IMPLEMENTATION`

## Scope and invariants

Implemented a new policy schema,
`adaptive_joint_cache_replica_aware_uav_operation_v10c_decoupled_encoder`,
without replacing the V10-B schema. The only policy architecture change is
splitting the actor's shared representation into `cache_encoder` and
`uav_encoder`, each `400 -> 128 -> 128`. The V10-B constructor initializes the
original encoder and existing heads in their original order; the V10-C
constructor clones that encoder into two independent modules without another
random initialization. Cache operation/candidate paths use only the Cache
encoder; UAV operation/candidate paths use only the UAV encoder. Candidate
features, autoregressive UAV contexts, action spaces, temporal adaptation,
reward, system model/cost, matching, and KKT behavior are unchanged.

One actor optimizer still owns both encoders and all active actor heads/scorers.
The separate critic encoder/head, PopArt, GAE, critic optimizer, and critic LR
are unchanged and remain disjoint from the actor. PPO still uses the existing
global `centered_std` advantage, Cache branch ratio/clipping, joint-UAV
branch ratio/clipping, and equal branch-loss aggregation. No per-UAV ratio,
baseline, reward bonus, or penalty was added.

The new config is copied from the V10-B 5e-4 config. Its algorithm and system
settings are unchanged; the policy schema selects V10-C and its experiment
name/output tag identify the formal 5e-4 run. Verified values: actor LR
`0.0005`, critic LR `0.0003`, target KL `0.10`, clip epsilon `0.2`, 4 update
epochs, 10 minibatches of 30, gamma `1.0`, GAE lambda `0.95`, entropy
coefficient `0`, PopArt settings unchanged, and seed `4401`.

## Gate A — parameter ownership

- Cache encoder and UAV encoder parameter overlap: `0`.
- Cache and UAV head/scorer ownership is disjoint; all 20 active actor
  parameter tensors are covered exactly once.
- Actor/critic parameter overlap: `0`; the single actor optimizer owns the
  complete actor parameter set.
- V10-C checkpoint policy state contains separate `cache_encoder.*` and
  `uav_encoder.*` keys and no legacy shared `encoder.*` key. It also contains
  the existing heads/scorers, critic, and PopArt state. The checkpoint keeps
  both optimizer states and Python/NumPy/Torch RNG state under new checkpoint
  version `12`.

## Gate B — frozen forward equivalence

Compared V10-B and V10-C initialized from the same seed and original weights,
on the same frozen `episode_20001`, with identical RNG state and before any
optimizer step. The complete V10-B state mapped to V10-C had maximum absolute
weight difference `0.0`; Cache and UAV encoders were independently allocated.

| Check | Result | Maximum absolute difference |
|---|---:|---:|
| Cache request, operation, and candidate actions | PASS | exact match |
| UAV request, operation, and candidate actions | PASS | exact match |
| Total and branch log probabilities | PASS | `0.0` |
| Reward | PASS | `0.0` |
| `q_total` | PASS | `0.0` |
| Movement cost | PASS | `0.0` |

The V10-B config/schema remains loadable; its policy also completed the same
frozen rollout path. No V10-B checkpoint migration loader was added, and this
stage did not resume a V10-B checkpoint.

## Gradient isolation

Branch PPO losses were differentiated separately on the frozen rollout without
an optimizer step:

| Loss | Expected gradient group | Norm | Opposite encoder / heads |
|---|---|---:|---:|
| Cache | Cache encoder | `0.465137` | UAV encoder `0`; UAV heads `0` |
| Cache | Cache heads/scorer | `0.322077` | — |
| UAV | UAV encoder | `0.212577` | Cache encoder `0`; Cache heads `0` |
| UAV | UAV heads/scorer | `0.125958` | — |

The combined actor loss produced non-zero gradients for both encoders (`0.232568`
Cache, `0.106289` UAV). Critic parameters remain outside both actor branch
graphs.

## KL instrumentation

V10-C records per-minibatch Cache, UAV, and joint approximate KL; branch target
comparisons; and the current joint-gate decision and remaining suppressed
update epochs. Per-update-epoch records include branch/joint mean KL, trigger
category counts (`cache_only`, `uav_only`, `both`, `neither`), and the optional
branch-suppression attribution fields. The current joint target-KL condition
and stop decision are unchanged; these fields do not feed PPO control flow.
Suppression attribution is descriptive: when a joint gate fires, it records
whether exactly one branch mean exceeded target and the other branch shared
that stop; it is not a counterfactual branch-specific gate.

## Gate C — one-epoch runtime

Ran exactly one fresh training epoch on the first existing, sorted pool
realization, `episode_20001`, from the 300-directory pool
`runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911` (`episode_20001` through
`episode_20300`). No resume, generation, evaluation, or final-test seed was
used. The temporary run completed 300 slots and 40 minibatch optimizer steps
(40 actor and 40 critic steps); all gradient, return, advantage, and logged
metrics finite checks passed. The existing joint KL gate did not trigger.

The run emitted 40 minibatch and 4 update-epoch KL diagnostic records. With
target KL `0.10`, the epoch means were:

| Update epoch | Cache KL | UAV KL | Joint KL | Gate |
|---:|---:|---:|---:|---|
| 1 | `0.028378` | `0.009269` | `0.037647` | not triggered |
| 2 | `0.059288` | `0.005464` | `0.064751` | not triggered |
| 3 | `0.066951` | `0.003723` | `0.070674` | not triggered |
| 4 | `0.047162` | `0.003451` | `0.050612` | not triggered |

Epoch trigger counts: Cache-only `0`, UAV-only `0`, both `0`, neither `4`.
Both suppression attribution fields were false in this gate.

## Artifacts and repository state

- Added `configs/proposed/system_config_ppo_v10c_decoupled_encoder.toml`.
- Updated policy/schema dispatch, replica-aware feature schema validation,
  V10-C checkpoint versioning, branch gradient diagnostics, and V10-C-only
  KL logging.
- Added this implementation report.
- No commit or push was performed. The one-epoch gate run and temporary gate
  script were removed after validation.

```text
SYSTEM_MODEL_CHANGED = NO
GLOBAL_OBJECTIVE_CHANGED = NO
PPO_OBJECTIVE_CHANGED = NO
KL_BEHAVIOR_CHANGED = NO
TRAINING_EPOCHS_EXECUTED = 1
LONG_TRAINING_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
PUSHED = NO
```
