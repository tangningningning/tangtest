# Minimal PPO v2 external reference and implementation

## Scope and status

PPO v2 is a separate, small training path for learning only Cache/UAV
reconfiguration requests. It does not replace the current formal hierarchical
Gate/WHAT/Direct-CF method and it does not modify `system_config.toml`.

The fixed interface is:

```text
observation_spec -> dimension-driven actor/critic
request action -> existing execute_slot()
actual Cache/UAV state -> existing optimizer and Matching/KKT
G(t) = Q_actual - C_ca_actual - C_q_actual
rollout -> GAE -> clipped PPO update -> minimal checkpoint
```

`MINIMAL_PPO_V2_READY = YES`

This means the standalone implementation and the required bounded smoke path
are ready. It does not claim that a full 240-origin training run has been
completed; that run was explicitly out of scope.

## External references

### CleanRL

- Repository: [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl)
- Main source: [`cleanrl/ppo.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)
- Design notes: [CleanRL PPO documentation](https://github.com/vwxyzjn/cleanrl/blob/master/docs/rl-algorithms/ppo.md)
- License: MIT, declared in the repository [`pyproject.toml`](https://github.com/vwxyzjn/cleanrl/blob/master/pyproject.toml).
- Reused design ideas: compact actor-critic MLP, rollout-first organization,
  reverse-time GAE, minibatch PPO update, clipped surrogate objective, and
  finite-gradient checks.

### Stable-Baselines3

- Repository: [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- PPO source: [`stable_baselines3/ppo/ppo.py`](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py)
- Rollout-buffer reference: [`stable_baselines3/common/buffers.py`](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py)
- Collection/update separation: [`stable_baselines3/common/on_policy_algorithm.py`](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/on_policy_algorithm.py)
- License: MIT, stated by the repository and its [`NOTICE`](https://github.com/DLR-RM/stable-baselines3/blob/master/NOTICE).
- Reused design ideas: storing observations, actions, old log probabilities,
  values, rewards, advantages and returns separately; collecting an on-policy
  batch before minibatch updates.

No third-party source file was copied into this repository. The PPO v2 code is
a local implementation using PyTorch's public API and the project's existing
environment/spec APIs.

## Implementation

New files:

- `src/ppo_simple/policy.py`
  - MLP encoder with configurable hidden sizes;
  - independent binary actor logits and scalar critic;
  - joint log probability is the sum of per-bit binary log probabilities;
  - dimensions are constructor inputs, not fixed constants;
  - existing feasibility masks may restrict unavailable request values while
    preserving the request/action contract.
- `src/ppo_simple/rollout.py`
  - builds observations through the existing
    `ConfigurationObservationSpec`/observation builder;
  - samples request actions through the action spec;
  - calls the existing `slot_environment.execute_slot()`;
  - stores request actions, not actual optimizer outcomes;
  - uses only `result.system_utility` as reward;
  - verifies `request=0 => actual=0` and permits `request=1 => actual in {0,1}`;
  - computes standard GAE and returns.
- `src/ppo_simple/trainer.py`
  - loads the formal train origin range `0..239` from configuration;
  - constructs policy dimensions from `observation_spec.observation_dim` and
    `action_spec.action_dim`;
  - performs the standard clipped PPO update with configured gamma, GAE lambda,
    clip, learning rate, value coefficient, entropy coefficient, update epochs,
    minibatches and gradient clipping;
  - provides a small CLI:

    ```text
    python -m ppo_simple.trainer --config system_config.toml --episodes N --run-dir PATH
    ```

  - supports fresh runs and resume from `latest_checkpoint.pt`.

The minimal checkpoint contains policy state, optimizer state, completed
episode, method/config identity, dynamic observation/action dimensions,
optimizer step count and Python/NumPy/PyTorch RNG states. It does not reuse the
legacy Direct-CF checkpoint/history schema.

## Dimension and environment contract

The current artifact produces observation and action dimensions of 154 and 11,
but neither number is used as a PPO-core constant. Future cache nodes, UAVs or
observation features should update the existing environment/spec construction;
the policy and PPO loss receive the resulting dimensions.

The environment remains responsible for cache placement, UAV coordinates,
Matching/KKT, resource allocation and actual state propagation. PPO receives
only the binary request action and observes the resulting `G(t)`.

## Minimal validation

Only the requested bounded checks were run:

1. syntax/import and formal config loading: passed;
2. dynamic spec check: observation dimension and action dimension were read
   from the current specs, not hardcoded in the new modules;
3. one real train-only four-slot fixture: produced four transitions;
4. every smoke action was binary;
5. every smoke reward matched
   `execute_slot().system_utility = Q_actual - C_ca_actual - C_q_actual`;
6. one PPO update: total loss finite, gradients finite, and parameters changed
   (`parameter_delta_norm` was positive);
7. checkpoint payload required fields: passed.

No full 240-slot episode, Batch4, Direct-CF, Gate-CF, WHAT-CF, test-origin
evaluation, parameter sweep, workers comparison, or performance benchmark was
run.

## Environment and worktree protection

- PPO v2 did not modify the environment, Matching/KKT, Cache/UAV optimizer,
  reward formula, system model, or `system_config.toml`.
- Existing working-tree changes in
  `src/decision/cache_configuration.py` were retained and not overwritten.
- No `12.pdf` was searched for, created, copied or modified.
- `打包并上传.bat` and `adaptive_cache_mec.zip` were not modified.
- The temporary smoke script was removed after validation.

## Remaining issues

- PPO v2 has not been run for a full formal episode or selected for the formal
  experiment; doing so is a separate runtime decision.
- The current environment can still be computationally expensive on trigger
  slots; PPO v2 does not optimize that path.
- The current system configuration still names the existing formal method;
  PPO v2 is intentionally standalone and does not silently change that formal
  method selection.

