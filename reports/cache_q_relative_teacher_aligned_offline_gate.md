# Cache Q relative teacher-aligned offline gate

## Result

- Gate: `NO`
- This is an offline supervised fit on the persisted exact-Q dataset; no environment, Matching, KKT, PPO, or new exact branch was executed.
- The rejected factorized `B_KEEP + Delta_Q` path is not used and is not a production baseline.

## Dataset and invariants

- Dataset reused: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\cache_action_q_critic_sparse_audit_gate\offline_cache_q_dataset`
- `DATASET_SCHEMA_VALID = YES`; `CANDIDATE_ALIGNMENT_CHECK = PASS`.
- Split groups: train=72, validation=18, held_out=36.
- Candidate policy probabilities: `False`; operation-gap loss is therefore `0`.
- Existing production listwise helper: `PPOV2Trainer._exact_q_listwise_teacher`; temperature `1` comes from the production constant.
- `LISTWISE_SHIFT_INVARIANCE_CHECK = PASS`; maximum absolute difference `1.78813934326e-07`.

## Train-only Delta-Q tail statistics

| Statistic | Value |
|---|---:|
| `count` | `1030` |
| `mean` | `1.57865989856` |
| `std` | `24.9474958916` |
| `median` | `-0.0687446594238` |
| `mad` | `0.0954360961914` |
| `abs_p50` | `0.13618183136` |
| `abs_p90` | `13.269039917` |
| `abs_p95` | `72.988129425` |
| `abs_p99` | `119.412789764` |
| `min` | `-113.740783691` |
| `max` | `128.844573975` |
| `ROBUST_SCALE` | `13.269039917` |

The regression target is `z = Delta_Q / ROBUST_SCALE`, with SmoothL1/Huber beta=1.0. Each regression, listwise, and (if present) operation term is first averaged within a state×node group and then averaged over groups.

## Split metrics

| Split | Delta MAE | Delta RMSE | Pearson | Spearman | Argmax | Recall@1 | Recall@3 | Recall@5 | Norm regret mean | Norm regret p95 | Norm regret max | KL mean | KL p95 | Teacher top1 | Teacher top3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 7.5637071704 | 13.9137597857 | 0.966996031322 | 0.703915778505 | 0.638888888889 | 0.638888888889 | 0.916666666667 | 0.930555555556 | 0.00561017667004 | 0.0401636078637 | 0.0674271874124 | 0.0125342101951 | 0.0388758583696 | 0.638888888889 | 0.712962962963 |
| validation | 1.41357865486 | 2.56959608383 | 0.180097688161 | 0.318308692017 | 0.333333333333 | 0.333333333333 | 0.5 | 0.555555555556 | 0.320602389567 | 1 | 1 | 0.823147346189 | 3.23400568761 | 0.333333333333 | 0.444444444444 |
| held_out | 4.05216144837 | 6.58164887755 | 0.380289093698 | 0.405286155045 | 0.25 | 0.25 | 0.611111111111 | 0.888888888889 | 0.328602231323 | 1 | 1 | 0.64810538336 | 2.17650900223 | 0.25 | 0.490740740741 |

Exact regret statistics are reported in the JSON artifact under `exact_regret`; normalized regret divides each selected-action regret by that group's exact Delta range.

## Operation-gap metrics

| Split | update gap mae | update gap rmse | update gap pearson | update gap spearman | update keep sign accuracy | false update rate | false keep rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| validation | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| held_out | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

Candidate probabilities are absent from the persisted schema, so all operation-gap metrics are unavailable and `L_operation = 0`; no environment rerun was used to fill that field.

## Previous Variant B comparison

| Held-out metric | Current | Previous Variant B | Direction |
|---|---:|---:|---|
| `spearman` | `0.405286155045` | `0.409255917236` | `not_improved` |
| `recall_at_3` | `0.611111111111` | `0.666666666667` | `decreased` |
| `recall_at_5` | `0.888888888889` | `0.861111111111` | `not_decreased` |
| `normalized_regret_p95` | `1` | `0.999999999999` | `not_improved` |

The gate uses these raw directional comparisons without a hand-chosen numerical threshold; it does not treat lower MAE alone as a pass.

## Objective and isolation

- `L_total = group_mean(SmoothL1(z_pred, z_exact)) + group_mean(KL(teacher_exact || teacher_pred)) + L_operation`.
- `teacher_exact` is stop-gradient. `teacher_pred` uses the same production standardization, legality mask, and temperature; the temporary runner only removes the production teacher's intentional final detach so the predicted distribution can receive gradient.
- KEEP has conceptual `Delta_KEEP = 0`, but KEEP is not a neural candidate sample and no `B_KEEP` head or absolute-Q reconstruction is present.
- `NO_POLICY_GRADIENT = YES`, `NO_GLOBAL_CRITIC_GRADIENT = YES`, `NO_UAV_GRADIENT = YES`; only the fresh Cache relative-Q MLP has parameters and gradients.
- `CACHE_Q_CRITIC_USABLE_FOR_POLICY_CREDIT = NO` because cross-realization validation has not been performed.

## Artifacts and protection

- Metrics: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\cache_action_q_critic_sparse_audit_gate\offline_cache_q_dataset\relative_teacher_aligned_offline_gate\gate_metrics.json`
- Config: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\cache_action_q_critic_sparse_audit_gate\offline_cache_q_dataset\relative_teacher_aligned_offline_gate\gate_config.json`
- Checkpoint: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\cache_action_q_critic_sparse_audit_gate\offline_cache_q_dataset\relative_teacher_aligned_offline_gate\relative_teacher_aligned_checkpoint.pt`
- Report: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\cache_q_relative_teacher_aligned_offline_gate.md`
- Existing dataset files, the prior Variant B diagnostic, and the factorized diagnostic were not overwritten.
- Additional environment executions: 0; Matching calls: 0; KKT calls: 0.

