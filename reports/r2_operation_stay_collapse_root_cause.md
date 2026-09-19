# R2 Operation STAY Collapse Root Cause

## Scope and stop boundary

The R2 development run was stopped safely after the observed operation-policy collapse. The `ppo-train` wrapper and its identified Python worker were no longer running after one graceful Ctrl+C attempt followed by a targeted, non-forced stop of only the matching worker. No other Python, SUMO, Codex, or terminal process was targeted.

The completed-epoch boundary is authoritative:

- `completed_epochs = 7`
- `latest_checkpoint.pt.completed_episodes = 7`
- `latest_checkpoint.pt.next_pool_index = 7`
- `episode_metrics.jsonl = 7` completed records
- `update_metrics.jsonl = 7` completed records
- `slot_metrics.jsonl = 2100 = 7 * 300`
- epoch 8 had been entered by the terminal, but no epoch-8 completed record or checkpoint was written

Therefore `STOPPED_DURING_PARTIAL_EPOCH = YES`; no partial epoch data was used in the analysis. The run summary still says `status = running` because the trainer did not rewrite the status after external termination. The allowed [stop_note.json](../runs/ppo_recovery_r2_dev20_20260918/stop_note.json) records the stop boundary without changing historical metrics.

## Logging audit

The existing JSONL artifacts contain epoch-level operation probabilities, realized-action utility summaries, global GAE summaries, and a combined UAV-branch KL diagnostic. They do not persist the R2 per-transition counterfactual arrays held in memory by `UpdateResult`.

| Required diagnostic | Result | Evidence / limitation |
|---|---:|---|
| `DIRECT_Q_GAP_LOGGED` | `NO` | No per-sample `Q_stay` and `Q_move` fields |
| `OP_RAW_ADV_LOGGED` | `NO` | Logged `mean_advantage_for_uav_*` is the global GAE diagnostic, not R2 operation credit |
| `OLD_OPERATION_PROBS_LOGGED` | `NO` | Only epoch aggregates such as `uav_move_probability_mean` are persisted; no per-sample old operation probabilities |
| `OPERATION_KL_COMPONENT_LOGGED` | `NO` | No operation-head-only KL field |
| `CANDIDATE_KL_COMPONENT_LOGGED` | `NO` | No candidate-head-only KL field |
| `HISTORICAL_DELTA_Q_RECOVERY` | `INSUFFICIENT_LOGGING` | Historical full-epoch ΔQ cannot be reconstructed without raw R2 arrays |

The persisted `uav_approx_kl` is the combined UAV branch diagnostic (operation plus candidate branch), not an operation/candidate decomposition. It is therefore reported as `TOTAL_UAV_KL`, without attributing it to either sub-head.

## Bounded exact counterfactual fallback

Because the historical R2 arrays were not persisted, the permitted bounded fallback was run against frozen `episode_20001` only. It used slots `20, 50, 80, 110, 140, 170, 200, 230, 260, 290`, both UAV heads, and exact existing `_compute_r2_operation_credit` execution. The probe performed no optimizer step, no training, no episode regeneration, and no checkpoint write. The raw result is preserved at [bounded_r2_operation_cf.json](../runs/ppo_recovery_r2_dev20_20260918/bounded_r2_operation_cf.json).

The fallback is a checkpoint/slot probe, not a full-episode population estimate.

| Probe | Checkpoint epoch | N | ΔQ < 0 | ΔQ = 0 | ΔQ > 0 | Negative fraction | Mean ΔQ | Median ΔQ | Std ΔQ | P10 | P25 | P75 | P90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fresh R2 | — | 20 | 20 | 0 | 0 | 1.0000 | -0.0396047 | -0.0396142 | 0.0006968 | -0.0404136 | -0.0402049 | -0.0392636 | -0.0386556 |
| Best checkpoint | 5 | 20 | 20 | 0 | 0 | 1.0000 | -0.0411453 | -0.0411377 | 0.0000500 | -0.0412018 | -0.0411682 | -0.0411224 | -0.0411072 |
| Latest checkpoint | 7 | 20 | 20 | 0 | 0 | 1.0000 | -0.0411783 | -0.0411682 | 0.0000495 | -0.0412598 | -0.0411987 | -0.0411434 | -0.0411261 |

By UAV, every group was negative: fresh UAV 0 `10/10`, mean `-0.0401030`; fresh UAV 1 `10/10`, mean `-0.0391065`; best epoch-5 UAV 0 `10/10`, mean `-0.0411606`; best epoch-5 UAV 1 `10/10`, mean `-0.0411301`; latest epoch-7 UAV 0 `10/10`, mean `-0.0411633`; latest epoch-7 UAV 1 `10/10`, mean `-0.0411933`.

By sampled operation, the fresh probe had MOVE `11/11` negative with mean `-0.0400596` and STAY `9/9` negative with mean `-0.0390488`. The best and latest deterministic probes sampled STAY in all 20 cases, so their MOVE-conditioned subgroup is empty; this is a limitation of the bounded probe, not evidence that MOVE cases do not exist elsewhere.

The exact R2 credit invariants held in all 60 cases:

- action-centering sign alignment: `60/60 = 1.0000`
- STD-only operation scaling sign flips: `0`
- operation advantage centering: `NO`
- probabilities below `1e-8` in the bounded cases: `0`
- maximum raw-advantage identity residual: `2.62e-7` (fresh), `1.82e-8` (best), `4.11e-9` (latest)

## Early-collapse and KL evidence

The completed training logs show the operation collapse directly:

| Epoch | Mean P(MOVE) | Actual UAV movement | Return | Combined UAV-branch KL | Cache KL | UAV KL gate |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.519401 | 57.5% | -989.05 | 0.661889 | 0.065629 | triggered / stopped |
| 2 | 0.186046 | 22.7% | -3086.02 | 0.441045 | 0.020492 | triggered / stopped |
| 3 | 0.036714 | 3.2% | -2581.77 | 0.062065 | 0.021839 | not triggered |
| 4 | 0.003911 | 0.0% | -3363.03 | -0.004062 | 0.016775 | not triggered |
| 5 | 0.000845 | 0.0% | 584.79 | -0.000550 | 0.067737 | not triggered |
| 6 | 0.000383 | 0.0% | -1015.11 | -0.000153 | 0.044603 | not triggered |
| 7 | 0.000249 | 0.0% | 443.21 | -0.000091 | 0.082752 | not triggered |

The exact epoch-6 combined UAV-branch KL is `-0.0001532`; the table's rounded display should be read as that value. Epochs 1 and 2 also show high UAV clip fractions (`0.9342` and `0.9158`) and four UAV-only KL triggers each. This demonstrates a large early combined UAV-branch update, but the persisted logs cannot say whether operation or candidate log-probability changes caused most of that branch KL.

## Diagnosis

`PRIMARY_DIAGNOSIS = TARGET_BIAS` (`CASE_A_TARGET_BIAS_SUPPORTED`, bounded evidence).

The strongest available evidence is that all 60 exact bounded counterfactual probes—fresh initialization, best epoch-5 checkpoint, and latest epoch-7 checkpoint, both UAVs, and ten time-spread slots—have `Q_move - Q_stay < 0`, with mean gaps around `-0.0396` to `-0.0412`. The sign-alignment invariant is exact in the probe, and STD-only scaling introduces no sign flips. This supports H1: the R2 one-step counterfactual target is systematically favoring STAY in the inspected frozen cases.

The early aggregate UAV KL (`0.6619` in epoch 1 and `0.4410` in epoch 2, target `0.1`) likely accelerates the probability collapse, so update dynamics are a plausible amplifier. However, because operation-head and candidate-head log probabilities were not persisted separately, H2 cannot be isolated from the target effect and H3 cannot be classified. In particular, there is insufficient evidence to claim candidate-KL gate interference.

`COUNTER_EVIDENCE`:

- ΔQ is not available for the completed training epochs themselves; the result comes from 60 bounded cases on one frozen realization.
- Best/latest bounded probes have no sampled MOVE cases because the deterministic policy is already near STAY; their all-STAY sampled-action subgroup is not a balanced action comparison.
- Combined UAV KL is available, but operation-vs-candidate KL decomposition is not.

## Required result fields

```text
TRAINING_STOP_REQUESTED = YES
TRAINING_PROCESS_RUNNING = NO
STOPPED_DURING_PARTIAL_EPOCH = YES
LAST_COMPLETED_EPOCH = 7
LAST_COMPLETE_CHECKPOINT_EPOCH = 7
RUN_DIRECTORY = C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_recovery_r2_dev20_20260918

DIRECT_Q_GAP_LOGGED = NO
OP_RAW_ADV_LOGGED = NO
OLD_OPERATION_PROBS_LOGGED = NO (aggregate probability summaries only)
DELTA_Q_SOURCE = BOUNDED_FALLBACK

EPOCH_1_DELTA_Q_NEGATIVE_FRACTION = N/A (historical per-epoch Q gap not logged)
EPOCH_1_DELTA_Q_POSITIVE_FRACTION = N/A
EPOCH_1_DELTA_Q_MEAN = N/A
EPOCH_1_DELTA_Q_MEDIAN = N/A
EPOCH_2_DELTA_Q_NEGATIVE_FRACTION = N/A (historical per-epoch Q gap not logged)
EPOCH_2_DELTA_Q_POSITIVE_FRACTION = N/A
EPOCH_2_DELTA_Q_MEAN = N/A
EPOCH_2_DELTA_Q_MEDIAN = N/A
EPOCH_3_DELTA_Q_NEGATIVE_FRACTION = N/A (historical per-epoch Q gap not logged)
EPOCH_3_DELTA_Q_POSITIVE_FRACTION = N/A
EPOCH_3_DELTA_Q_MEAN = N/A
EPOCH_3_DELTA_Q_MEDIAN = N/A
ALL_E1_E3_NEGATIVE_FRACTION = N/A (historical per-epoch Q gap not logged)
ALL_E1_E3_POSITIVE_FRACTION = N/A

ACTION_CENTERING_SIGN_ALIGNMENT = 60/60 = 1.0000 (bounded fallback)
OP_ADV_SIGN_FLIP_COUNT = 0
OP_ADV_NORMALIZATION = STD_ONLY
OP_ADV_CENTERING = NO
RECONSTRUCTION_UNSTABLE_COUNT = 0 (bounded fallback)

KL_COMPONENT_DECOMPOSITION_AVAILABLE = NO
EPOCH_1_OPERATION_KL = N/A
EPOCH_1_CANDIDATE_KL = N/A
EPOCH_1_TOTAL_UAV_KL = 0.6618886072 (combined UAV branch)
EPOCH_2_OPERATION_KL = N/A
EPOCH_2_CANDIDATE_KL = N/A
EPOCH_2_TOTAL_UAV_KL = 0.4410447153 (combined UAV branch)

PRIMARY_DIAGNOSIS = TARGET_BIAS
SUPPORTING_EVIDENCE = 60/60 bounded ΔQ samples negative; both UAVs negative in all three checkpoint probes; exact sign-alignment and zero sign flips; early combined UAV KL above target
COUNTER_EVIDENCE = bounded single-realization sample; no historical per-epoch ΔQ; no operation/candidate KL split
R2_OPERATION_POLICY_COLLAPSE = YES

SOURCE_MODIFIED = NO
REWARD_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO
MATCHING_KKT_CHANGED = NO
MOVEMENT_COST_CHANGED = NO
CACHE_CREDIT_CHANGED = NO
CANDIDATE_CREDIT_CHANGED = NO
TARGET_KL_CHANGED = NO
ACTOR_LR_CHANGED = NO

FORMAL_TRAINING_EXECUTED = NO
TRAINING_RESUMED = NO (not resumed as part of this diagnostic task; prior dev continuation had already reached epoch 7)
EPISODE_REGENERATION = NO
SUMO_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
FORMAL_EVALUATION_EXECUTED = NO

TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO
```
