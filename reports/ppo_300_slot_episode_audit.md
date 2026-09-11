# PPO 300-slot randomized episode protocol audit

Date: 2026-09-11
Baseline: `2261609941d041a441c65e4758faa4b9f4b7d0b8`
Scope: static audit only. No source algorithm, environment, data, checkpoint,
or evaluator behavior was changed by this audit.

## 1. Current baseline

The active V6 configuration is
`system_config_ppo_primitive_v6_horizon_grouped_separate_optimization_popart_actorlr1e4.toml`.
It declares:

| Item | Current value |
| --- | --- |
| `simulation.dt_s` | `1.0 s` |
| `simulation.episode_time_s` | `300.0 s` |
| Derived physical episode length | `300` slots (`300.0 / 1.0`) |
| PPO train origins | `0..239` (240 origins) |
| PPO held-out test origins | `240..299` (60 origins) |
| `num_minibatches` | `8` |
| Trainer default `rollout_steps` | `16` (CLI can override it) |
| Formal Epoch-240 checkpoint `rollout_steps` | `240` (saved payload; one old train-pass batch) |

The current development protocol therefore treats one PPO training Epoch as a
pass over the 240-origin train interval. The checked-in formal Epoch-240
checkpoint records `rollout_steps=240`, so that run used one PPO update for the
240-origin train pass. The current CLI default is `rollout_steps=16`, which
would perform multiple PPO updates while traversing that interval. Neither
contract waits for a complete 300-slot episode before updating.

The frozen evaluator is intentionally a different, old development contract:
it is pinned to the V6 Epoch-240 checkpoint and strictly evaluates origins
`240..299`. Its constants and assertions in `src/ppo_simple/evaluator.py`
must remain available for reproducing the old frozen V6 result.

## 2. Protocol definitions

### Old development protocol

```text
fixed/generated episode origins 0..239 -> PPO training
frozen Epoch-240 policy on independent held-out origins 240..299
```

This is the protocol encoded by the current V6 TOML, the pinned evaluator, and
the existing 240-origin slot-metrics contract. It is not equivalent to the new
randomized full-episode training protocol.

### New formal protocol

```text
each training episode = one independently randomized realization
collect origins 0..299 (300 transitions)
origin 299 is the true terminal
perform the PPO update only after the complete episode
freeze the policy and evaluate repeated independent full 0..299 episodes
```

The old V6 configuration and held-out evaluator should be preserved as an
explicit legacy/development path. A new full-episode protocol should be
identified separately rather than silently redefining the old checkpoint.

## 3. Static location classification

### `OLD_DEVELOPMENT_ONLY`

| Location | Finding |
| --- | --- |
| `system_config_ppo_primitive_v6_horizon_grouped_separate_optimization_popart_actorlr1e4.toml`, `[algorithm.ppo.data_protocol]` | Encodes train `0..239` and test `240..299`; retain unchanged for the frozen V6 baseline. |
| `src/ppo_simple/evaluator.py:40-59, 173-257, 278-356, 433-542` | Pins the old held-out evaluator to `240..299`, `60` slots, and checkpoint `completed_epochs=240`; retain for old V6 reproduction. |
| `src/ppo_simple/plotting.py:13, 79-85, 171-301` | `SLOTS_PER_EPOCH=240` and the slot-metrics writer/validator assume a complete 240-origin training Epoch. This is an old reporting contract, not a PPO mathematical requirement. |
| `src/ppo_simple/reporting.py:477-504` | The final summary exposes fixed reward regions `0..59`, `60..119`, `120..179`, and `180..239`; this is tied to the old 240-origin report. |

### `NEW_PROTOCOL_MUST_CHANGE`

| Location | Required future change |
| --- | --- |
| Protocol/configuration contract | Add a separately identified full-episode training protocol whose train interval is `0..299`; do not overwrite the old V6 config used by the frozen checkpoint. The current validator assumes a non-empty contiguous train/test split and therefore needs an explicit full-episode mode or equivalent separate contract. |
| `src/ppo_simple/trainer.py:337-411, 511-550, 675-713, 1380-1554` | Keep the boundary sourced from context, but make the new mode collect all 300 origins, use terminal `299`, and postpone the PPO update until the episode is complete. The code default is a 16-step chunk; the existing formal Epoch-240 checkpoint records an explicit 240-step chunk. |
| `src/ppo_simple/rollout.py:143-187, 331-417` | The caller must pass the new terminal boundary `299` and use a complete-episode rollout/update contract. The `done` and bootstrap plumbing is already parameterized; no GAE formula rewrite is indicated. |
| `src/ppo_simple/plotting.py:13-301` | Make slot-metrics validation, origin bounds, global slot numbering, and per-Epoch array lengths use the active episode length (`300` for the new protocol) while retaining compatibility with old 240-origin artifacts where required. |
| `src/ppo_simple/reporting.py:207-241, 258-338, 413-505` | Keep the dynamic manifest fields, but make summary slot regions and any complete-Epoch slot checks protocol-aware. Add an explicit protocol/episode-slot provenance field or equivalent guard so old and new artifacts cannot be mixed. |
| Checkpoint/resume boundary | Add an explicit protocol/episode-slot compatibility check for new runs. Existing config-hash and schema checks are useful but are not a readable full-protocol identity contract; the old Epoch-240 checkpoint must not be silently resumed as a 300-slot run. |
| New evaluator path | Extend or add a mode that runs the frozen policy over repeated independent full `0..299` realizations, with no optimizer or PopArt-statistics update. Preserve `evaluate_heldout()` as the old `240..299` evaluator. |

The full-episode update also exposes a current batch-contract constraint:
`PPOV2Trainer.update()` requires the sample count to be divisible by
`num_minibatches`. A 300-transition update is not divisible by the current
`num_minibatches=8`. The next implementation must make an explicit
protocol-level choice (for example, a compatible minibatch contract or an
explicit full-episode batching rule); it must not claim that simply changing
the terminal index is sufficient.

### `ALREADY_DYNAMIC`

| Location | Finding |
| --- | --- |
| `src/common/project_config.py:759-825` | Physical slot count is derived from `episode_time_s / dt_s`, and configured origin ranges are validated from the TOML. The validation is dynamic in the count, although its current train/test split shape requires a new explicit full-episode mode. |
| `src/ppo_simple/interface.py:351-368` | `normalized_time_to_go()` uses the supplied start/end bounds rather than a literal slot count. |
| `src/ppo_simple/interface.py:753-871` | Prediction horizon alignment and the `H=5` mask use the supplied `prediction_terminal_origin`; the formula is already boundary-parameterized. |
| `src/ppo_simple/rollout.py:158-162, 331, 450-485` | True terminal, `done`, bootstrap observation, and GAE boundary are driven by the supplied terminal index. |
| `src/ppo_simple/trainer.py:518-546, 688-713, 1420` | Available origins, chunk sizes, terminal forwarding, and `origins_per_epoch` derive from the active context range. The current value is 240 because the current config says `0..239`. |
| `src/ppo_simple/plotting.py:405-435` | The sole `training_epoch_return.png` renderer reads `episode_metrics.jsonl` and computes Raw Epoch Return, MA20, and MA50 without a hardcoded 240-slot data source. |
| `src/ppo_simple/reporting.py:207-241` | Manifest `train_origin_start`, `train_origin_end`, and `origins_per_epoch` are populated from the live context. |
| `src/scenario/realization.py` and the episode-seed call sites | Seed streams are derived from the episode seed and named stream; no 240/300 slot count is embedded in the helper. |

### `UNRELATED`

The PPO objective, advantage/GAE arithmetic, reward calculation, PopArt
mathematics, actor/critic architecture, learning rates, observation fields
other than the finite-horizon boundary value, action space, Cache/UAV decision
execution, Matching, KKT, system model, and prediction model are not the
source of the 240-versus-300 protocol distinction. No change to those areas is
indicated by this audit.

## 4. Boundary and bootstrap audit

### Time-to-go

Current training observations use the dynamic formula

```text
time_to_go(o) = (239 - o) / (239 - 0),  o in 0..239
```

The old held-out evaluator separately uses `(299-o)/(299-240)` over
`240..299`. For the new full episode the required training observation is

```text
time_to_go(o) = (299 - o) / (299 - 0),  o in 0..299
```

This is a boundary/input change, not a change to the feature's normalization
formula or to the policy architecture.

### Terminal and GAE boundary

The current training terminal is origin `239`; origin `239` is marked done and
does not bootstrap beyond the train interval. Earlier rollout chunks bootstrap
from the immediately following train origin. `compute_gae()` uses the supplied
`done` flags and bootstrap value in the standard recursion.

The new terminal must be origin `299`, with no bootstrap at `300`. The GAE
recursion itself remains unchanged. Only the episode boundary, the number of
transitions delivered to the update, and the caller's bootstrap choice need to
change.

### Prediction horizon mask

The current training `H=5` mask treats a prediction as valid when
`origin + horizon_index <= 239`; the old held-out evaluator uses the same
dynamic code with terminal `299`. The new training path must pass terminal
`299`, yielding `origin + horizon_index <= 299`. The mask formula itself is
already dynamic.

## 5. Required source surface for the next implementation

The minimum next implementation surface is:

1. A new, explicitly named full-episode protocol/configuration contract and the
   corresponding validator branch in `src/common/project_config.py`.
2. `src/ppo_simple/trainer.py` for randomized episode boundaries, complete
   300-transition collection, one-update-per-episode orchestration, and the
   compatible rollout/minibatch contract.
3. `src/ppo_simple/reporting.py` and `src/ppo_simple/plotting.py` for dynamic
   episode-slot reporting, while preserving the old 240-origin artifact
   reader/evaluator path.
4. A separate or explicitly selected full-episode evaluator path in
   `src/ppo_simple/evaluator.py` for repeated independent frozen-policy runs.
5. Checkpoint/manifest protocol provenance so a 240-origin development run and
   a 300-origin formal run cannot be resumed interchangeably.

`src/ppo_simple/interface.py` and `src/ppo_simple/rollout.py` need caller
contract review, but their time-to-go, horizon-mask, terminal, bootstrap, and
GAE mechanisms are already parameterized and do not require mathematical
changes. The episode-seed helper does not need a slot-length change.

## 6. Audit result

```text
CURRENT_EPISODE_TIME_S = 300.0
CURRENT_DT_S = 1.0
DERIVED_EPISODE_SLOTS = 300

CURRENT_TRAIN_ROLLOUT_STEPS = 240 in the formal Epoch-240 checkpoint; trainer default is 16 and CLI override exists
CURRENT_TRAIN_ORIGIN_RANGE = 0..239
CURRENT_EVAL_ORIGIN_RANGE = 240..299

OLD_DEVELOPMENT_PROTOCOL = fixed 0..239 PPO training + frozen 240..299 held-out evaluation
NEW_FORMAL_PROTOCOL = randomized full 0..299 episode + update after 300 transitions + repeated independent full-episode evaluation

TIME_TO_GO_CURRENT = dynamic over training range 0..239; old evaluator separately uses 240..299
TIME_TO_GO_300_SLOT_REQUIRED = (299-origin)/299 over 0..299
TIME_TO_GO_CHANGE_REQUIRED = YES (boundary input only)

TERMINAL_CURRENT = training origin 239
TERMINAL_300_SLOT_REQUIRED = origin 299

HORIZON_MASK_CURRENT = H=5 valid through the supplied training terminal, currently 239
HORIZON_MASK_CHANGE_REQUIRED = YES (pass 299 in the new training path)

GAE_BOUNDARY_CURRENT = done at 239; earlier chunk boundaries bootstrap the next train origin
GAE_BOUNDARY_CHANGE_REQUIRED = YES (move the boundary to 299; keep GAE math)
BOOTSTRAP_CHANGE_REQUIRED = YES (new terminal/input boundary; no terminal bootstrap)

ROLLOUT_LENGTH_CHANGE_REQUIRED = YES
REPORTING_CHANGE_REQUIRED = YES
CHECKPOINT_PROTOCOL_GUARD_REQUIRED = YES
EVALUATOR_EXTENSION_REQUIRED = YES

EPISODE_SEED_HELPER_CONTAINS_SLOT_LENGTH_ASSUMPTION = NO

PPO_MATH_CHANGE_REQUIRED = NO
REWARD_CHANGE_REQUIRED = NO
OBSERVATION_SEMANTIC_CHANGE_REQUIRED = NO (only finite-horizon boundary value changes)
ACTION_CHANGE_REQUIRED = NO
MATCHING_CHANGE_REQUIRED = NO
KKT_CHANGE_REQUIRED = NO
```
