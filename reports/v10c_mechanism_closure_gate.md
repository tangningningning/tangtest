# V10-C mechanism closure gate

STAGE = V10-C_MECHANISM_CLOSURE_GATE

## Scope and provenance

This closes the five hypotheses in `v10c_objective_preserving_dataflow_design.md` without re-auditing the system model. I re-read that report and used its verified active code path. The representative inputs were the epoch-240 `latest_checkpoint.pt` from each V10-C diagnostic run and one frozen 300-slot rollout per checkpoint, both on development realization `episode_20001` from `runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911`.

The run manifests confirm actor LR `0.0005` / `0.0006`, critic LR `0.0003`, target KL `0.10`, clip epsilon `0.2`; both checkpoint loads completed 240 episodes. The full rollout was used only to collect frozen transitions and measure mechanisms. No `PPOV2Trainer.update()` was invoked. The two single-step optimizer dry-runs used deep-copied policies and Adam states, one step per copy, and left original policy parameters and optimizer counters unchanged. No checkpoint was written.

## Verified active code objects

| Object | Active path |
|---|---|
| Shared actor encoder | `src/ppo_simple/policy.py`: `policy.encoder`; used by both Cache and UAV actor branches. The critic encoder is separate. |
| Cache policy path | `evaluate_cache_operation_primitive`, with Cache operation and candidate scoring; its branch log probability/loss is consumed by `PPOV2Trainer.update`. |
| UAV policy path | `evaluate_joint_cache_uav_operation`; its two operation-head slices and candidate scoring use the conditional UAV contexts. |
| UAV0/UAV1 old log probabilities | `SimpleRollout.uav_operation_log_probs` and `uav_old_log_probs`, retained by head. For head *i*, `l_i = l_i,operation + l_i,candidate` (candidate term is zero for STAY). |
| Joint UAV probability and PPO ratio | The evaluator sums the two operation and candidate log probabilities; the update uses `l_U = l_0 + l_1`, then one joint ratio and clip. The frozen-batch reconstruction matched stored joint log probability within `2.9e-6` (5e-4) and `5.8e-6` (6e-4). |
| Cache/UAV KL | `PPOV2Trainer.update` forms `cache_branch_delta` and `uav_branch_delta`; the diagnostic rows contain episode-level averages `cache_approx_kl` and `uav_approx_kl`. |
| Current KL gate | In `PPOV2Trainer.update`, the mean full-joint approximate KL is checked after each update epoch. If it exceeds target, the one shared gate breaks the remaining actor and critic epochs. |
| Global advantage | `compute_gae` produces one rollout-wide advantage; `PPOV2Trainer.update` applies the configured `centered_std` normalization and sends that tensor to both branch losses. |
| Action-conditional diagnostics | `critic_diagnostics.jsonl` records per-transition Cache action and per-head selected UAV actions, but the logged UAV ratio is explicitly joint. `update_metrics.jsonl` holds episode-level branch summaries. Neither file stores per-minibatch/per-epoch branch KL histories. |

## 1. Cache–UAV shared-encoder gradients

For each checkpoint, the two exact branch PPO losses were differentiated separately with respect to the same shared `policy.encoder` parameters on the same 300-transition frozen rollout and the exact normalized GAE advantages. No optimizer step was used for this measurement. Ten chronological 30-slot blocks were also measured to check persistence rather than relying only on a pooled gradient.

| Checkpoint | Full-batch cosine | Full-batch Cache norm | Full-batch UAV norm | Full-batch norm ratio | Negative-cosine blocks | Median block cosine | Median Cache/UAV block norm ratio | Blocks with Cache norm > UAV norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5e-4 | -0.3555 | 0.50785 | 0.28039 | 1.811 | 6/10 | -0.0346 | 3.320 | 10/10 |
| 6e-4 | -0.0619 | 0.50654 | 0.08273 | 6.123 | 4/10 | 0.0412 | 4.329 | 10/10 |

The Cache/UAV norm ratio exceeded 3 in 5/10 and 6/10 blocks respectively; every block in both trajectories had the Cache shared-gradient norm above the UAV norm (minimum ratios 1.013 and 2.019). Block cosine signs were mixed, so conflict is not uniform at every time segment. The persistent norm direction and multi-fold median disparity nevertheless provide direct evidence of Cache-side domination of the shared representation. This supports separating the Cache and UAV actor encoders; it does not establish that this is the sole cause of action polarization.

`CACHE_UAV_GRADIENT_CONFLICT_SUPPORTED = YES` (pooled cosine is negative at both checkpoints; block signs are mixed).

`CACHE_UAV_GRADIENT_DOMINATION_SUPPORTED = YES` (Cache norm is larger in all 20 measured blocks; median ratios are 3.32x and 4.33x).

`DECOUPLE_CACHE_UAV_ENCODER = YES`.

## 2. UAV0–UAV1 joint-ratio cross-clipping

On each checkpoint, one evenly spaced 30-transition minibatch was run through one actor optimizer step on a deep copy with a deep-copied Adam state. After that copy-only update, old/new per-head operation-plus-candidate log probabilities were evaluated for all 300 frozen transitions. Clip epsilon remained 0.2; a ratio was counted clipped when `abs(r - 1) > 0.2`.

| Run | UAV0 individually clipped | UAV1 individually clipped | Joint clipped | Joint clipped while UAV0/UAV1 not clipped | Individual clipped while joint not | Any mismatch |
|---|---:|---:|---:|---:|---:|---:|
| 5e-4 | 0/300 | 0/300 | 0/300 | 0 / 0 | 0 / 0 | 0/300 |
| 6e-4 | 0/300 | 0/300 | 0/300 | 0 / 0 | 0 / 0 | 0/300 |

Maximum observed ratios were 1.0677 (UAV0), 1.0053 (UAV1), 1.0677 (joint) at 5e-4, and 1.0354, 1.0916, 1.1045 at 6e-4. No ratio reached the clip interval boundary, and no update was changed by cross-clipping in these representative dry-runs.

`JOINT_UAV_CLIP_CROSS_COUPLING_5E4 = 0/300 mismatches`

`JOINT_UAV_CLIP_CROSS_COUPLING_6E4 = 0/300 mismatches`

`PER_UAV_RATIO_CLIPPING = NO` — the required direct evidence of frequent clipping-induced per-head update changes was absent. Keep the joint ratio for now.

## 3. Joint KL gate and cross-suppression

The existing `update_metrics.jsonl` files have 240 update rows per run. The recorded joint gate actually stopped later update epochs 11 times for 5e-4 and 47 times for 6e-4. The run summaries show 16 and 67 joint update epochs skipped respectively. Episode-level branch-KL averages remained at or below target in all rows (coarse average-only classification: Cache-only 0, UAV-only 0, Both 0, Neither 240 for each run; maximum recorded Cache averages were 0.072938 and 0.099707, and maximum UAV averages 0.035502 and 0.038257).

Those averages cannot answer the requested per-minibatch/per-update-epoch classification. The gate tests a per-epoch mean of minibatch **joint** KL, while the serialized branch KLs are averages over the update’s executed minibatches/epochs. The source computes temporary branch-KL lists but does not serialize the per-minibatch or per-epoch values. The available per-transition UAV diagnostic is also joint, and there is no matching per-transition Cache KL trace. Consequently, although the shared gate demonstrably activates, the data cannot identify whether Cache-only or UAV-only KL caused any activation or whether one branch was still under target when the other drove the gate.

| Quantity requested | Result |
|---|---|
| Cache-only / UAV-only / both KL trigger count | **Not recoverable** from existing files; per-minibatch/per-epoch branch KL tensors were not retained. The 0/0/0/240 counts above are episode-average categories only, not trigger counts. |
| Cache-triggered suppression of UAV updates | **Not recoverable**; the gate event lacks branch attribution. |
| UAV-triggered suppression of Cache updates | **Not recoverable**; the gate event lacks branch attribution. |
| Actual joint-gate events | 11 (5e-4), 47 (6e-4); 16 and 67 joint epochs skipped. |

`BRANCH_SPECIFIC_KL_GATE = STILL_INSUFFICIENT` — technical limitation: missing raw per-minibatch/per-epoch `cache_approx_kl`, `uav_approx_kl`, and gate-decision trace. Do not change gate behavior until that diagnostic data exists.

## 4. UAV contextual baseline and per-UAV baseline

The inputs were the rollout’s stored pre-action `uav_contexts` (400 dimensions as defined in the objective-preserving report), plus only a UAV-head one-hot for the shared model. No selected action, post-action position, reward, `q_total`, movement cost, next state, or future data was included. Target was the exact centered/std-normalized GAE advantage used by PPO, duplicated per UAV decision. The split was chronological: first 210 slots for training, last 90 for held-out validation. A small ridge model used training-only standardization/PCA; regularization was selected on an inner split wholly within the first 210 slots. Shared and per-UAV fits had a matched total of 100 slope coefficients (100 shared versus 50 per head).

| Run | Held-out `Var(A_global)` | Shared `Var(A-B(c))` | Shared variance reduction | Shared MSE | Shared explained variance | Shared generalizes? |
|---|---:|---:|---:|---:|---:|---|
| 5e-4 | 1.24467 | 4.84024 | -288.9% | 5.79805 | -3.658 | NO |
| 6e-4 | 1.87588 | 1.67220 | 10.86% | 2.26414 | -0.207 | NO |

At 6e-4 residual variance decreased, but MSE exceeded the constant-mean target variance and explained variance was negative. At 5e-4 both variance and MSE worsened sharply. This does not meet the requirement for stable held-out generalization across both trajectories.

| Run | Shared-baseline residual variance | Per-UAV residual variance | Per-UAV relative residual-variance gain vs shared | Per-UAV validation MSE / explained variance |
|---|---:|---:|---:|---:|
| 5e-4 | 4.84024 | 1.12265 | +76.81% | 1.16516 / +0.064 |
| 6e-4 | 1.67220 | 1.84311 | -10.22% | 2.03676 / -0.086 |

The per-UAV fit helps on 5e-4 but is worse than the shared fit on 6e-4; its own 6e-4 MSE is also above the target variance. This is not a consistent gain across the two held-out trajectories.

`UAV_CONTEXTUAL_BASELINE = NO` — shared contextual fit does not generalize on both validation tracks.

`PER_UAV_CONTEXTUAL_BASELINE = NO` — the parameter-matched per-UAV fit is not consistently better than the shared fit.

## 5. One-step system-objective counterfactual

For each run, four actual frozen head-states were sampled (two per UAV head). Cache final state and the other UAV’s sampled final position were held fixed. For each head-state, STAY was compared with four evenly spaced choices from its 56–80 feasible MOVE candidates; thus the reported fractions cover 16 comparisons per run, not every feasible coordinate. Each comparison re-executed the same one-step actual matching/KKT path and recorded `q_total`, delay, energy, movement cost, and global reward.

| Run | MOVE better | STAY better | Near tie | Mean `G(MOVE)-G(STAY)` | Mean `Q(MOVE)-Q(STAY)` | Mean added movement cost |
|---|---:|---:|---:|---:|---:|---:|
| 5e-4 | 0/16 (0%) | 16/16 (100%) | 0/16 | -0.03730 | +0.000097 | +0.03740 |
| 6e-4 | 0/16 (0%) | 16/16 (100%) | 0/16 | -0.03752 | -0.000122 | +0.03740 |

Across the 32 sampled feasible MOVE comparisons, the immediate movement-cost increment dominated the tiny change in task utility. This supports a **sample-local one-step STAY bias**, not a general claim about all states and not a policy/Q-value diagnosis.

`ONE_STEP_MOVE_BETTER_FRACTION = 0/32`

`ONE_STEP_STAY_BETTER_FRACTION = 32/32`

`ONE_STEP_NEAR_TIE_FRACTION = 0/32`

`SYSTEM_OBJECTIVE_ONE_STEP_BIAS = STAY_BIASED_IN_SAMPLED_STATES`

`ONE_STEP_COUNTERFACTUAL_IS_NOT_LONG_TERM_Q = YES`

No reward or system-cost adjustment follows from this check.

## 6. Modification-set closure

| Candidate | Decision | Direct basis / reason |
|---|---|---|
| `DECOUPLE_CACHE_UAV_ENCODER` | **YES** | Cache shared-encoder gradient norm exceeded UAV in all 20 time blocks; block-median ratios were 3.32x and 4.33x. Full-batch Cache/UAV cosines were negative on both checkpoints. |
| `PER_UAV_RATIO_CLIPPING` | **NO** | Zero per-head/joint clip events and zero cross-clipping mismatches on either 300-transition dry-run batch after one copy-only actor step. |
| `BRANCH_SPECIFIC_KL_GATE` | **STILL_INSUFFICIENT** | 58 joint-gate events were observed, but raw branch KL per minibatch/epoch and branch-specific suppression attribution are absent. |
| `UAV_CONTEXTUAL_BASELINE` | **NO** | Shared baseline fails the held-out criteria on both tracks (5e-4 strongly worsens; 6e-4 has negative explained variance and worse MSE). |
| `PER_UAV_CONTEXTUAL_BASELINE` | **NO** | It helps only 5e-4 and is worse than shared on 6e-4; no stable two-track benefit. |

The only supported algorithm change is separating Cache and UAV actor encoders. A diagnostic-only logging follow-up for per-minibatch/per-epoch Cache KL, UAV KL, joint KL, and gate events is needed before reconsidering branch-specific gating; it is not a gate behavior change.

## Invariants and execution status

```text
GLOBAL_REWARD_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO
SYSTEM_COST_CHANGED = NO
ACTION_SPACE_CHANGED = NO
TEMPORAL_ADAPTATION_PRESERVED = YES
LONG_TRAINING_EXECUTED = NO
FORMAL_PPO_UPDATE_INVOKED = NO
EVALUATION_OR_FINAL_TEST_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
CHECKPOINT_WRITTEN = NO
TEMPORARY_DIAGNOSTIC_CODE_REMOVED = YES
PUSHED = NO
```
