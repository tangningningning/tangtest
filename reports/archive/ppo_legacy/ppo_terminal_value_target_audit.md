# Scheme B PPO terminal GAE / value-target / resume audit

## Executive summary

**Conclusion: no terminal-bootstrap, GAE, return-target, or resume-boundary implementation bug was found in the current `policy_primitive` production path.**

The 240-origin Epoch has a true terminal transition at **origin 239**.  `done[t]` is attached to the transition produced by the action at that origin, after its reward is computed.  The terminal step has neither a bootstrap observation nor an effective next-state value contribution.  Non-terminal chunks explicitly bootstrap from the immediately following origin, while the final `224:239` chunk cannot read origin 240/test data.

The existing metrics do show a strong, repeatable terminal critic-loss anomaly: terminal-chunk value loss is approximately 16x the non-terminal mean and 8.8x the non-terminal median.  The spike is localized to `224:239`, not a general late-Epoch increase.  The best-supported explanation is a **mixed representation/horizon and value-scale/optimization issue**, not faulty terminal arithmetic:

- the critic receives no explicit origin, normalized time, remaining horizon, or time-to-go feature;
- with `gamma=1.0`, the finite-horizon continuation value changes sharply near origin 239;
- terminal return targets remove all continuation value, while the same critic must infer this only indirectly from physical/prediction features;
- the unnormalized squared-error critic objective amplifies such target/prediction mismatch.

No current log records per-transition critic predictions, raw GAE advantages, or raw returns, so the exact numerical predicted-value/target mismatch cannot be quantified without a future diagnostic run.  That missing evidence does not undermine the static proof that the bootstrap and masking implementation are correct.

## Scope and evidence

This was a static audit plus offline parsing of existing JSONL files only.  No origin was simulated, no PPO update was run, no checkpoint was written or deserialized, and no production source/configuration was changed.

Audited production configuration:

```ini
configuration_control_mode = "policy_primitive"
policy_schema = "adaptive_primitive_v1"
gamma = 1.0
gae_lambda = 0.95
value_loss_coefficient = 0.5
train origins = 0..239
```

Relevant sources:

- `src/ppo_simple/trainer.py`
- `src/ppo_simple/rollout.py`
- `src/ppo_simple/interface.py`
- `src/ppo_simple/policy.py`
- `system_config.toml`

## Production call chain

The actual Scheme B path is:

```text
run_training()
  -> PPOV2Trainer.collect_chunk()
  -> collect_rollout()
  -> policy.sample() + sample_and_apply_primitive_configuration()
  -> slot_environment.execute_slot()
  -> _check_reward() / transition.done / SimpleTransition
  -> SimpleRollout
  -> PPOV2Trainer.update()
  -> compute_gae()
  -> policy.evaluate_primitive()
  -> MSE critic loss + optimizer.step()
```

| Concern | Production location | Evidence |
| --- | --- | --- |
| Trainer entry and cursor loop | `src/ppo_simple/trainer.py:614-748` | Loads cursor, collects a contiguous chunk, updates, then checkpoints cursor/state. |
| Chunk sizing and true terminal injection | `src/ppo_simple/trainer.py:276-310` | Caps a chunk at remaining train origins and passes `train_origin_end` as terminal index. |
| Rollout buffer | `src/ppo_simple/rollout.py:19-79`, `388-420` | `SimpleTransition` stores reward, `done`, old critic value, observation and next environment state; `SimpleRollout` stacks them. |
| Reward | `src/ppo_simple/rollout.py:99-105`, `280-307` | Reward is the checked `execute_slot().system_utility`, i.e. actual utility minus Cache/UAV costs. |
| Done write | `src/ppo_simple/rollout.py:307-333` | `done = (step.origin_index == terminal_origin_index)` is stored on that same transition. |
| GAE and targets | `src/ppo_simple/trainer.py:325-339`; `src/ppo_simple/rollout.py:423-451` | Raw advantages/returns are generated before actor-only advantage normalization. |
| Critic loss and optimizer | `src/ppo_simple/trainer.py:393-465` | Evaluates Scheme B's `evaluate_primitive`, computes MSE-style loss, clips gradients, and steps Adam. |
| Checkpoint/resume cursor | `src/ppo_simple/trainer.py:492-606`, `644-748` | Saves only completed-update state and cursor; no rollout, GAE, or target arrays are serialized. |

## Terminal and done semantics

`system_config.toml:532-538` defines formal train origins `0..239`.  `PPOV2Trainer.collect_chunk()` passes `train_origin_end` (239) to `collect_rollout()` at `trainer.py:302-310`.  The rollout computes the reward first, then writes:

```python
done = int(step.origin_index) == terminal_origin_index
```

at `rollout.py:306-333`.

Therefore the semantics are `done[t]`: the action and reward at origin 239 are included, and only its following transition is terminal.  This is neither a `done[t+1]` convention nor an off-by-one truncation.

For the required examples:

- `208:223`: `remaining=32`, chunk size is 16, so `collect_chunk()` requests one extra aligned origin 224 as the bootstrap observation (`trainer.py:290-301`).
- `224:239`: `remaining=16`, no extra aligned step is requested; origin 239 becomes the final, `done=True` transition.
- `aligned_train_steps()` explicitly rejects an attempted bootstrap past origin 239 (`trainer.py:212-240`).  Thus origin 240/test data cannot enter a training bootstrap.

## TD bootstrap audit

`PPOV2Trainer.update()` initializes `bootstrap_value = 0.0` and evaluates a next-state critic value only when `rollout.bootstrap_observation` exists (`trainer.py:325-339`).  `collect_rollout()` creates that observation only if its final transition is not done (`rollout.py:374-386`).

`compute_gae()` then applies the terminal mask independently of the numeric bootstrap value:

```text
next_value[t] = bootstrap_value              if t is the final chunk item
                V(s[t+1])                   otherwise
nonterminal[t] = 0 if done[t] else 1
delta[t] = r[t] + gamma * nonterminal[t] * next_value[t] - V(s[t])
```

For origin 239, there is no bootstrap observation, `bootstrap_value=0`, and `nonterminal=0`; its bootstrap term is therefore exactly zero twice over.  For origin 223, the bootstrap observation uses the state produced by executing origin 223 plus the aligned predictions for origin 224.  It is not origin 0 of the next Epoch.

## GAE audit

The implemented reverse recurrence at `rollout.py:440-448` is:

```text
A[t] = delta[t] + gamma * gae_lambda * (1 - done[t]) * A[t+1]
returns[t] = A[t] + V(s[t])
```

The same `nonterminal` mask is used in both TD residual and the advantage recurrence.  On `done[239]=True`, the recurrence resets to the terminal residual; it cannot propagate an advantage from a subsequent Epoch.  Conversely, a non-terminal rollout boundary does not set `done`; it uses the immediate next observation/value as the standard truncated-rollout bootstrap.  There is no saved previous-chunk advantage to leak into the next chunk.

## Return and value-target audit

The critic target is exactly:

```text
raw_advantages = GAE(rewards, old_values, dones, bootstrap_value)
returns = raw_advantages + old_values
```

at `rollout.py:423-451`.  Only after `returns` has been created does the trainer normalize `advantages_t` for the policy surrogate (`trainer.py:374-378`).  `returns_t` is constructed from raw returns and is never normalized.

The target is not double-counted: the stored values are subtracted in the TD residual and added back exactly once when forming `returns`.  Rewards are used once per TD residual; no discounted reward accumulation occurs elsewhere.  The terminal reward is calculated before the done flag and remains in `delta[239]`.

The GAE inputs are each `reshape(-1)` and require equal nonzero lengths (`rollout.py:433-437`).  The returned one-dimensional arrays are converted to tensors indexed by the same minibatch indices as critic values (`trainer.py:344-376`, `432`), so there is no broadcasting path.

## Value-loss implementation

At `trainer.py:432-438` the critic uses the same path for terminal and non-terminal samples:

```text
value_loss = 0.5 * mean((V_theta(s) - returns)^2)
total_loss = policy_loss + 0.5 * value_loss - 0.0 * entropy
```

This is half-MSE with mean reduction.  There is no Huber loss, PPO value clipping, value clipping, return normalization, or value normalization.  Gradients are globally clipped to norm 0.5 after backpropagation (`trainer.py:441-454`); this limits updates but does not alter the reported pre-clipping error norm.

## Resume-boundary audit

Checkpoint payload keys at `trainer.py:501-527` include policy state, optimizer state, RNG states, `completed_episodes`, `current_train_origin`, Cache/UAV state, and optimizer step count.  It does **not** serialize a rollout buffer, pending transition, previous/next critic value, advantage, return, done mask, or bootstrap observation.

Each checkpoint is saved only after `trainer.update(rollout)` completes (`trainer.py:729-748`).  At the Epoch boundary the final chunk has `next_train_origin=240`, then `completed_episodes` is incremented.  Thus the stated current checkpoint boundary has:

```text
completed_episodes = 407
current_train_origin = 240
```

On `--epochs 500 --resume`, the cursor is loaded and, before new collection, the loop sees `current_train_origin > train_origin_end`; it resets to origin 0 and fresh initial Cache/UAV state (`trainer.py:701-726`).  It also creates new per-Epoch metric accumulators.  No GAE or rollout state exists to be inherited from Epoch 407.

## Critic time/horizon feature audit

The actor and critic share the same encoder (`policy.py:99-131`), so both consume the observation built by `interface.build_configuration_observation()` (`interface.py:243-293`).  Its fields are exactly:

```text
Cache binary vector
normalized UAV positions
normalized predicted vehicle positions
predicted service-popularity vector
```

There is no explicit origin index, normalized origin index, slot, time, remaining horizon, time-to-go, terminal flag, or terminal-proximity feature.  Physical/prediction values may correlate with time, but they do not provide a guaranteed finite-horizon clock.

Consequently, two similar physical/configuration/prediction states at origins 100 and 239 can yield different finite-horizon continuation values while being indistinguishable to the critic.  With `gamma=1.0`, origin 239's target contains only its immediate terminal reward, whereas an earlier state has potentially many future rewards.  This is a direct mechanism for persistent terminal prediction error without a terminal arithmetic error.

## Existing metrics statistics

Offline source files:

- `runs/ppo_primitive_first_episode_20260908/update_metrics.jsonl`
- `runs/ppo_primitive_first_episode_20260908/episode_metrics.jsonl`

The update file contains 6,090 records: 406 records for each of 15 fixed 16-origin ranges.  Episode metrics contain 407 complete Epochs, so the first completed Epoch has no corresponding update records in this file.  Statistics below use only the actual logged `value_loss`, `gradient_norm`, `entropy`, and `approximate_kl` fields.  `reward_mean` is **not present** in these JSONL update records, so it is deliberately not fabricated.

| Group | Count | Value loss mean | Median | p90 | p95 | Max | Gradient mean | Gradient p90 | Gradient max | Entropy mean | Mean abs(KL) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| terminal `224:239` | 406 | 30,885.08 | 16,252.50 | 82,814.77 | 106,300.41 | 165,128.52 | 1,051.41 | 2,224.94 | 3,203.14 | 16.47 | 0.01092 |
| all non-terminal chunks | 5,684 | 1,963.37 | 1,853.61 | 3,579.64 | 4,154.47 | 19,700.46 | 298.82 | 454.23 | 1,011.57 | 16.56 | 0.01151 |

Per-origin-range value-loss and gradient means:

| Origins | Value-loss mean | Value-loss p95 | Gradient mean |
| --- | ---: | ---: | ---: |
| 0:15 | 1,545.86 | 3,108.37 | 276.20 |
| 16:31 | 1,758.95 | 3,605.29 | 288.67 |
| 32:47 | 1,672.03 | 3,271.59 | 272.40 |
| 48:63 | 2,045.61 | 4,124.04 | 305.92 |
| 64:79 | 1,980.28 | 4,352.21 | 305.64 |
| 80:95 | 2,063.22 | 4,414.31 | 308.58 |
| 96:111 | 1,986.53 | 4,228.15 | 302.45 |
| 112:127 | 2,049.38 | 4,014.71 | 297.08 |
| 128:143 | 1,780.18 | 3,726.58 | 280.62 |
| 144:159 | 2,019.34 | 4,335.08 | 298.25 |
| 160:175 | 1,879.55 | 3,733.54 | 296.27 |
| 176:191 | 2,163.51 | 4,286.10 | 314.92 |
| 192:207 | 2,268.37 | 4,454.97 | 314.41 |
| 208:223 | 2,274.42 | 4,548.65 | 322.07 |
| 224:239 | 30,885.08 | 106,300.41 | 1,051.41 |

The existing episode returns are also substantial (407 records; mean 1,767.73, median 2,140.23, p95 2,484.55, range -934.23 to 2,658.58).  This supports the plausibility of material unnormalized value-target scale, but it does not reveal individual raw targets or predictions.

## Root-cause conclusion and next step

The evidence rejects the proposed code-bug mechanisms: no terminal bootstrap, no cross-Epoch bootstrap, no GAE-mask omission, no chunk-as-terminal handling, no normalized-advantage critic target, no target double-counting, no dropped terminal reward, and no resume carry-over were found.

The evidence supports a systematic terminal critic mismatch caused primarily by a finite-horizon representation gap, amplified by unnormalized squared-error scale.  This classification does **not** assert PPO convergence or optimal learning quality.

The minimal next experiment, not performed here, is a controlled fresh-training ablation that adds an explicit normalized time-to-go/origin feature to the shared observation and logs, per transition, `V(s_t)`, raw advantage, raw return, `done`, and bootstrap value.  It changes the observation dimension and therefore should use a new checkpoint/run rather than attempting to reuse the current v3 policy weights.  Before that experiment, a diagnostics-only logging pass could establish the exact terminal prediction/target residuals without changing reward or PPO mathematics.

## Required conclusion fields

```ini
TERMINAL_ORIGIN = 239
TERMINAL_DONE_FLAG_CORRECT = YES
TERMINAL_OFF_BY_ONE = NO

TERMINAL_NEXT_VALUE_ZERO = YES
TERMINAL_BOOTSTRAP_BLOCKED = YES
NONTERMINAL_CHUNK_BOOTSTRAP_CORRECT = YES
CROSS_EPOCH_VALUE_BOOTSTRAP = NO

GAE_DELTA_MASK_CORRECT = YES
GAE_RECURSION_MASK_CORRECT = YES
ROLLOUT_BOUNDARY_HANDLING_CORRECT = YES
GAE_CROSSES_EPOCH_BOUNDARY = NO

VALUE_TARGET_FORMULA = returns = raw_GAE_advantages + old_values
VALUE_TARGET_FORMULA_CORRECT = YES
NORMALIZED_ADVANTAGE_USED_AS_VALUE_TARGET = NO
DOUBLE_VALUE_ADDITION = NO
DOUBLE_REWARD_ACCUMULATION = NO
TERMINAL_REWARD_DROPPED = NO
VALUE_TARGET_SHAPE_CORRECT = YES

VALUE_LOSS_TYPE = half-MSE
VALUE_LOSS_REDUCTION = mean
VALUE_COEFFICIENT = 0.5
VALUE_CLIPPING_ENABLED = NO
RETURN_NORMALIZATION_ENABLED = NO
VALUE_NORMALIZATION_ENABLED = NO
TERMINAL_VALUE_LOSS_PATH_DIFFERENT = NO

RESUME_CURSOR_CORRECT = YES
RESUME_ROLLOUT_STATE_CARRIED = NO
RESUME_GAE_STATE_CARRIED = NO
RESUME_CROSS_EPOCH_CONTAMINATION = NO

CRITIC_HAS_EPOCH_TIME_FEATURE = NO
TIME_FEATURES = []

INSUFFICIENT_LOGGED_VALUE_TARGET_DATA = YES
TERMINAL_VALUE_LOSS_ANOMALY = YES
VALUE_TARGET_IMPLEMENTATION_BUG = NO
TERMINAL_VALUE_LOSS_HIGH_EXPLAINED = YES

ROOT_CAUSE = MIXED
BUG_FIX_REQUIRED = NO
SAFE_TO_RESUME_TRAINING_WITH_CURRENT_CODE = YES
```

