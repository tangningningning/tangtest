# ppo_simple_v2 real-action PPO smoke report

## Scope and repository baseline

This report covers the minimum real-action smoke run requested for the current
`ppo_simple_v2` training path. No source code, system model, Cache/UAV
optimizer, Matching/KKT, reward, network, or PPO hyperparameter was changed.
No commit, push, merge, or rebase was performed.

Baseline before the smoke run:

- branch: `main`
- HEAD: `b7e90d4015bbeac6823eb55a4a29d71a7e174d7`
- upstream: `origin/main`
- tracked diff: none
- pre-existing untracked files: the two earlier audit reports and
  `打包并上传.bat`

## Launch

The official `ppo_simple.trainer:main` entrypoint was called with:

```text
--config system_config.toml
--episodes 1
--rollout-steps 4
--run-dir runs/ppo_simple_v2_real_action_smoke_20260908
```

`rollout_steps=4` is only the minimum collection horizon needed to satisfy
`num_minibatches=4`; it is not a PPO hyperparameter override. In the formal
trainer, `episodes=1` spans train origins 0..239. To keep this as a smoke run,
the observer stopped immediately after the first official checkpoint save,
after origins 0..3 had completed one real rollout and one PPO update. This is
not a convergence or full-episode result.

The trainer initialized a fresh policy using the configured seed and sampled
the binary action vector through the normal Bernoulli policy path. No logits,
actions, or neutral-only fixture were injected or forced.

## PPO configuration confirmed

| Parameter | Value |
|---|---:|
| `method` | `ppo_simple_v2` |
| `gamma` | `1.0` |
| `gae_lambda` | `0.95` |
| `clip_epsilon` | `0.2` |
| `learning_rate` | `3e-4` |
| `value_loss_coefficient` | `0.5` |
| `entropy_coefficient` | `0.0` |
| `max_grad_norm` | `0.5` |
| `update_epochs` | `1` |
| `num_minibatches` | `4` |
| `hidden_sizes` | `[128, 128]` |

## Real action and environment observations

The run produced 4 transitions and all 4 were non-neutral. Every observed
action mask was `[true, true]` for every dimension and transition; every
sampled bit was allowed by its mask. The raw sampled action rows were:

```text
t0: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
t1: [0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0]
t2: [0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0]
t3: [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
```

Action dimensions are ordered as
`z_23^ca, z_87^ca, z_43^ca, z_75^ca, z_58^ca,
z_RSU_0^ca, z_RSU_1^ca, z_UAV_0^ca, z_UAV_1^ca,
z_UAV_0^q, z_UAV_1^q`.

| Dimension | Bit 0 count | Bit 1 count |
|---|---:|---:|
| `z_23^ca` | 4 | 0 |
| `z_87^ca` | 1 | 3 |
| `z_43^ca` | 1 | 3 |
| `z_75^ca` | 2 | 2 |
| `z_58^ca` | 2 | 2 |
| `z_RSU_0^ca` | 4 | 0 |
| `z_RSU_1^ca` | 2 | 2 |
| `z_UAV_0^ca` | 2 | 2 |
| `z_UAV_1^ca` | 4 | 0 |
| `z_UAV_0^q` | 3 | 1 |
| `z_UAV_1^q` | 3 | 1 |

The environment-side action totals were:

| Metric | Result |
|---|---:|
| requested Cache triggers (first 9 dimensions) | 14 |
| actual Cache updates | 3 |
| actual Cache updates per transition | `[0, 3, 0, 0]` |
| requested UAV movement triggers (`z_UAV_*^q`) | 2 |
| actual UAV movements | 2 |
| actual UAV movements per transition | `[1, 0, 0, 1]` |

Thus this smoke run did exercise non-neutral Cache and UAV requests and the
real environment applied both kinds of change. It does not claim that every
requested Cache trigger must be accepted; the Cache optimizer's acceptance
decision remains unchanged and is outside this smoke verdict.

## Rollout and update metrics

| Metric | Min | Mean | Max |
|---|---:|---:|---:|
| reward | 6.4316859245 | 9.1875730753 | 11.0157527924 |
| return | 11.0898437500 | 22.5527081490 | 34.0250663757 |
| advantage | 11.0160932541 | 22.4896910190 | 33.9723892212 |

One PPO `update()` call completed. It used 4 optimizer steps (one epoch and
four minibatches), as required by the configured update structure. The
reported `UpdateResult` values are the formal trainer's final minibatch
values for this update:

| Metric | Value |
|---|---:|
| policy loss | -0.3132264614 |
| value loss | 311.0824890137 |
| entropy | 7.6006484032 |
| total loss | 155.2280120850 |
| approximate KL | 0.0427951813 |
| clip fraction | 0.0 |
| parameter delta norm | 0.1770783500 |
| gradient finite flag | `true` |

The gradient norms recorded at the formal `clip_grad_norm_` call (total norm
before clipping) were:

```text
[58.9460372925, 20.8657913208, 34.3292655945, 43.4972801208]
```

Their min/mean/max were `20.8657913208 / 39.4095935822 /
58.9460372925`. All were finite.

## Finite-value and checkpoint checks

| Check | Result |
|---|---|
| actions finite | `true` |
| rewards finite | `true` |
| returns finite | `true` |
| advantages finite | `true` |
| policy loss finite | `true` |
| value loss finite | `true` |
| entropy finite | `true` |
| total loss finite | `true` |
| gradient norms finite | `true` |
| gradients finite at formal guard | `true` |
| sampled actions allowed by masks | `true` |

Checkpoint status: `PASS`.

```text
runs/ppo_simple_v2_real_action_smoke_20260908/latest_checkpoint.pt
```

The checkpoint contains `method=ppo_simple_v2`, checkpoint version `2`,
`optimizer_steps=4`, `current_train_origin=4`, `completed_episodes=0`,
observation dimension `154`, and action dimension `11`. It is retained as the
formal training artifact.

## Execution note

The first invocation reached the environment and wrote a checkpoint, but the
temporary observer then failed while treating the recorded mask shape
`[transitions, action_dimensions, 2]` as a scalar mask. That was an observer
indexing error, not a trainer/environment error. Its partial checkpoint was
removed. The clean run above used the corrected observer and completed all
required smoke checks.

## Final result

```text
REAL_ACTION_PPO_SMOKE = PASS
READY_FOR_FORMAL_PPO_TRAINING = YES
```

This result confirms the startup path and one real-action PPO update only. It
does not evaluate convergence and does not justify changing any parameter.

The temporary observer and generated Python caches were removed after the
run. No source changes were made. `git diff --check` was run after cleanup and
returned no output (PASS).
