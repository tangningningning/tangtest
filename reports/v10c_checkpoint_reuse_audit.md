# V10-C Checkpoint Reuse Audit

STAGE = V10-C_CHECKPOINT_REUSE_AUDIT

## Scope and execution boundary

This was a read-only audit of the two completed V10-B formal runs. No PPO
training, resume, evaluation, final-seed test, checkpoint modification,
trainer/config/reward/system-model modification, or episode generation was
performed. The checkpoint files were loaded only to inspect metadata and
state-dict structure.

## Checkpoint inventory

The only matching checkpoint files in either run are `best_train_return_checkpoint.pt`
and `latest_checkpoint.pt`. There are no numbered epoch checkpoint archives in
the requested 150--260 range.

### 5e-4

`runs/ppo_v10_branchwise_cache_uav_5e4_kl_formal`

- `best_train_return_checkpoint.pt` -> `episode/completed_epochs=253`,
  `best_epoch=253`, `next_pool_index=253`.
- `latest_checkpoint.pt` -> `episode/completed_epochs=300`,
  `best_epoch=253`, `next_pool_index=300`.

`MID_CHECKPOINTS_5E4_AVAILABLE = YES -- best checkpoint only at epoch 253;
no multi-point mid-checkpoint series.`

### 6e-4

`runs/ppo_v10_branchwise_cache_uav_6e4_kl_formal`

- `best_train_return_checkpoint.pt` -> `episode/completed_epochs=245`,
  `best_epoch=245`, `next_pool_index=245`.
- `latest_checkpoint.pt` -> `episode/completed_epochs=300`,
  `best_epoch=245`, `next_pool_index=300`.

`MID_CHECKPOINTS_6E4_AVAILABLE = YES -- best checkpoint only at epoch 245;
no multi-point mid-checkpoint series.`

Both runs have `checkpoint_version=11`, `method=ppo_simple_v2`, the expected
V10 branchwise policy schema, and the expected joint Cache/UAV control mode.
The local configuration SHA-256 matches each checkpoint and manifest:

- 5e-4: `08beb348b20d8da5664f6dc7b503e53bff4d4cee4ceec1b5acef93a230a08442`
- 6e-4: `a402ece28ace025832c13703aeeeae8cb48e4848d3518e633c0359fb541a25c4`

The checkpoint stores actor and critic parameters together in
`policy_state_dict` (26 keys: 18 actor-side, 6 critic-side, and 2 PopArt
keys); it does not use separate actor/critic weight files. It also stores
explicit PopArt/Welford statistics (`critic_return_scale_count/mean/m2` and
`popart_mean/std`), separate actor/critic optimizer state, and RNG state with
Python, NumPy, and Torch sub-states. The actor optimizer learning rates are
0.0005 and 0.0006 respectively; the critic optimizer learning rate is 0.0003
in both runs.

Therefore:

`INFERENCE_DIAGNOSTIC_REUSABLE = YES`

`TRAINING_RESUME_REUSABLE = YES`

The latter is a metadata/compatibility conclusion only. No resume was run.

## Historical episode alignment

The frozen pool
`runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911` contains exactly 300
directories, `episode_20001` through `episode_20300`. Each formal run has 300
episode-metric records, epochs 1--300, all with seed 4401. The persisted
realization fields agree in both runs:

| training epoch | pool index | realization |
|---:|---:|---|
| 1 | 0 | `episode_20001` |
| 150 | 149 | `episode_20150` |
| 180 | 179 | `episode_20180` |
| 200 | 199 | `episode_20200` |
| 220 | 219 | `episode_20220` |
| 240 | 239 | `episode_20240` |
| 260 | 259 | `episode_20260` |
| 300 | 299 | `episode_20300` |

`HISTORIC_EPISODE_MAPPING_VALID = YES`

The mapping is the simple linear sequence `training epoch e -> episode_20000+e`
with pool index `e-1` for both runs.

## Reuse decision

The historical polarization audit classified the 5e-4 transition at MA20
UAV movement below 10% at epoch 215, and the 6e-4 transition at MA20 UAV
movement above 90% at epoch 212. The available best checkpoints, at epochs
253 and 245, are both after the relevant transition; the latest checkpoints
are at epoch 300. No available checkpoint captures the pre-transition state
around 180--200 and the subsequent transition as a series.

`FULL_RETRAIN_REQUIRED = LIKELY -- for complete pre/onset/post
action-conditional training-time capture; a 300-epoch retrain is not required.`

## Pure inference versus PPO diagnostics

`PURE_INFERENCE_SUFFICIENT = NO`

A checkpoint can restore the policy, critic, PopArt normalization, and enough
state to perform ordinary inference. A fresh inference rollout can also
observe selected actions and environment quantities such as reward, utility,
movement cost, delay, and energy. It cannot reproduce the full requested
historical V10-C diagnostic record from the checkpoint alone:

- the global normalized `actual_PPO_advantage` requires the rollout return/
  value path, GAE, and the PPO normalization step;
- the old UAV branch log-probability belongs to the sampled rollout, while the
  new branch log-probability is evaluated after the policy update;
- `ratio` and `clip` require the old/new pair and the PPO clipping rule during
  the update;
- the completed V10-B logs do not retain the per-transition UAV operation,
  physical movement, Cache action, and the corresponding new/old branch
  quantities needed to join those fields historically.

Consequently, the exact action-conditional fields must be collected through a
short PPO diagnostic continuation or a new instrumented partial run. This is
not a claim that the conditional means are causal effects.

## Short-continuation feasibility

`SHORT_CONTINUATION_FEASIBLE = YES`

The checkpoint compatibility fields, config hashes, policy schema, full-episode
protocol, PopArt state, optimizer states, RNG states, cursor, and seed mapping
are all present and consistent. The existing resume path can restore these
fields and select the next frozen realization without touching the original
formal run, provided a selected checkpoint is copied into a new isolated run
directory as its `latest_checkpoint.pt`. No such copy or continuation was
performed here.

A 1--5 epoch continuation from the 5e-4 best checkpoint would consume
`episode_20254` onward, and from the 6e-4 best checkpoint `episode_20246`
onward. These continuations would be post-polarization only. The epoch-300
latest checkpoints have no remaining episode in the 300-realization pool and
are not useful for a same-protocol continuation.

## Recommended minimum run

`RECOMMENDED_MODE = PARTIAL_RETRAIN_TO_POLARIZATION`

Because no checkpoint exists before the transition, start a fresh isolated
instrumented run at epoch 1 and stop after epoch 240. This captures the
pre-transition and at least the first post-transition window for both runs
while avoiding an unnecessary 300-epoch diagnostic retrain.

`RECOMMENDED_5E4_START = 1`

`RECOMMENDED_5E4_END = 240`

`RECOMMENDED_6E4_START = 1`

`RECOMMENDED_6E4_END = 240`

`ESTIMATED_SAVED_EPOCHS = 120 total (60 per run versus 300 each)`

## Final status

STAGE = V10-C_CHECKPOINT_REUSE_AUDIT

CHECKPOINTS_5E4 = `best_train_return_checkpoint.pt` @253; `latest_checkpoint.pt` @300

CHECKPOINTS_6E4 = `best_train_return_checkpoint.pt` @245; `latest_checkpoint.pt` @300

MID_CHECKPOINTS_5E4_AVAILABLE = YES, best only @253

MID_CHECKPOINTS_6E4_AVAILABLE = YES, best only @245

INFERENCE_DIAGNOSTIC_REUSABLE = YES

TRAINING_RESUME_REUSABLE = YES

HISTORIC_EPISODE_MAPPING_VALID = YES

PURE_INFERENCE_SUFFICIENT = NO

SHORT_CONTINUATION_FEASIBLE = YES, post-polarization only from available best checkpoints

FULL_RETRAIN_REQUIRED = LIKELY for complete pre/onset/post capture

RECOMMENDED_MODE = PARTIAL_RETRAIN_TO_POLARIZATION

RECOMMENDED_5E4_START = 1

RECOMMENDED_5E4_END = 240

RECOMMENDED_6E4_START = 1

RECOMMENDED_6E4_END = 240

ESTIMATED_SAVED_EPOCHS = 120 total

TRAINING_EXECUTED = NO

EVALUATION_EXECUTED = NO

FINAL_TEST_SEEDS_USED = NO

REPORT = C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\v10c_checkpoint_reuse_audit.md

TEMP_FILES_REMOVED = YES

GIT_DIFF_CHECK = PASS

PUSHED = NO
