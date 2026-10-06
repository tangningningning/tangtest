# Cache Trigger Symmetric Counterfactual Root Cause Diagnosis

## Scope and checkpoint selection

Read-only frozen-policy diagnosis using only environment_data/proposed_joint_training and realizations 20001, 20100, 20200, 20300. Both runs were still marked running and below 300 epochs at inspection; both had epoch 140, used as the common checkpoint. latest_checkpoint.pt was not read. No training, validation, test, optimizer step, or backward pass was run.

Fixed slots: 20, 60, 100, 140, 180, 220, 260, 280. All 9 Cache heads were analyzed. Actual rollout actions were sampled from the frozen policy with deterministic=False; the same seeded action sampling was used for both checkpoints for each realization.

Current TD-Q came from compute_one_step_tdq_cache_trigger_credit. Q_keep and all legal UPDATE candidates were replayed through the current _execute_cache_branch path, which calls execute_slot and its Matching/KKT reward path; the frozen critic supplied each next-state bootstrap. Other Cache decisions, both UAV decisions, tasks, predictions, and vehicle state were held at the sampled rollout transition.

## Required conclusions

| Field | Result |
|---|---|
| SYMMETRIC_EXPECTED_Q_COMPUTED | YES |
| CURRENT_REFERENCE_SIGN_MISMATCH_OBSERVED | YES |
| POLARIZED_POLICY_CONTRADICTS_EXPECTED_Q | YES |
| RSU_POLARIZATION_CONTRADICTION_5E4 | 0/64 (0.0%) |
| RSU_POLARIZATION_CONTRADICTION_1E3 | 30/64 (46.9%) |
| GLOBAL_TDQ_SCALE_HEAD_IMBALANCE_OBSERVED | YES |
| CACHE_TRIGGER_REFERENCE_ASYMMETRY_SUPPORTED | INCONCLUSIVE |
| CACHE_TRIGGER_REFERENCE_FIX_JUSTIFIED | INCONCLUSIVE |

Policy/Q sign alignment compares sign(p_update - p_keep) with sign(delta_q_expected); current TD-Q alignment compares sign(current_oriented_tdq) with sign(delta_q_expected). Zero-sign pairs are excluded. Polarization uses the exact task thresholds. Reference asymmetry thresholds and head-scale criterion are recorded in the JSON method section.

## Checkpoint aggregates

| Checkpoint | Epoch | UPDATE probability | Mean expected delta-Q | Positive delta rate | Policy/Q alignment | Polarized contradictions | Current TD-Q mismatch | Ref error mean | Ref abs p90 | Global TD-Q std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5e-4 | 140 | 0.4399 | -37.2308 | 0.2188 | 0.7674 | 67/288 (23.3%) | 0.0938 | -1.5168 | 23.0860 | 54.1411 |
| 1e-3 | 140 | 0.6599 | -38.4354 | 0.3194 | 0.6528 | 100/288 (34.7%) | 0.1562 | 1.6844 | 23.7652 | 59.1101 |

### Role aggregates

| Checkpoint | Role | N | UPDATE probability | Mean delta-Q | Positive delta rate | Policy/Q alignment | UPDATE wrong | KEEP wrong | TD-Q mismatch | Ref error mean | Ref abs p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5e-4 | SV | 160 | 0.3981 | -42.6241 | 0.2062 | 0.7937 | 32 | 1 | 0.0813 | -2.5333 | 31.3330 |
| 5e-4 | RSU | 64 | 0.0001 | -59.8921 | 0.0000 | 1.0000 | 0 | 0 | 0.0000 | -1.1217 | 6.6911 |
| 5e-4 | UAV | 64 | 0.9840 | -1.0861 | 0.4688 | 0.4688 | 34 | 0 | 0.2188 | 0.6295 | 13.9358 |
| 1e-3 | SV | 160 | 0.3991 | -69.0135 | 0.1938 | 0.7937 | 33 | 0 | 0.0563 | 2.8765 | 33.5804 |
| 1e-3 | RSU | 64 | 0.9995 | 1.0562 | 0.5312 | 0.5312 | 30 | 0 | 0.2812 | 1.2282 | 11.3419 |
| 1e-3 | UAV | 64 | 0.9722 | -1.4816 | 0.4219 | 0.4219 | 37 | 0 | 0.2812 | -0.8398 | 15.0953 |

### Per-head TD-Q scale

Population standard deviation (ddof=0). Std-only shared scaling preserves each sign and cannot explain a sign flip.

| Checkpoint | Node | Role | N raw | std(raw_tdq) | mean(abs(raw_tdq)) | mean(abs(raw_tdq/global std)) |
|---|---|---|---:|---:|---:|---:|
| 5e-4 | 46 | SV | 32 | 23.0074 | 93.0485 | 1.7186 |
| 5e-4 | 10 | SV | 32 | 7.0520 | 5.5270 | 0.1021 |
| 5e-4 | 0 | SV | 32 | 66.6556 | 122.6768 | 2.2659 |
| 5e-4 | 23 | SV | 32 | 6.1464 | 5.4148 | 0.1000 |
| 5e-4 | 59 | SV | 32 | 12.2791 | 13.1092 | 0.2421 |
| 5e-4 | RSU_0 | RSU | 32 | 21.1194 | 98.7682 | 1.8243 |
| 5e-4 | RSU_1 | RSU | 32 | 5.4869 | 23.2595 | 0.4296 |
| 5e-4 | UAV_0 | UAV | 32 | 13.2327 | 11.9354 | 0.2205 |
| 5e-4 | UAV_1 | UAV | 32 | 12.1810 | 10.4597 | 0.1932 |
| 1e-3 | 46 | SV | 32 | 11.0777 | 7.0387 | 0.1191 |
| 1e-3 | 10 | SV | 32 | 40.2026 | 106.4797 | 1.8014 |
| 1e-3 | 0 | SV | 32 | 38.8232 | 150.9194 | 2.5532 |
| 1e-3 | 23 | SV | 32 | 16.7115 | 72.3940 | 1.2247 |
| 1e-3 | 59 | SV | 32 | 13.3621 | 11.1236 | 0.1882 |
| 1e-3 | RSU_0 | RSU | 32 | 12.1690 | 6.6438 | 0.1124 |
| 1e-3 | RSU_1 | RSU | 32 | 9.2182 | 7.4011 | 0.1252 |
| 1e-3 | UAV_0 | UAV | 32 | 13.7614 | 7.6851 | 0.1300 |
| 1e-3 | UAV_1 | UAV | 32 | 9.2207 | 7.4453 | 0.1260 |

## Source SHA256 recorded before diagnosis

| File | SHA256 |
|---|---|
| src/ppo_simple/policy.py | c0a8ebede1fbcde55cf1b6666fd06f03b3c0acbebfd43223f80094ecc60952f8 |
| src/ppo_simple/trainer.py | 4e3d5d760ac07efc50121ac97c9257ad8ed4fe5957a7f4f9e727e9795582f7fa |
| src/ppo_simple/cache_credit.py | 7bd0a2037d9e2df51b05c08dd6b985bdca7cbb3b71e1d7c4766216751507ecd3 |
| configs/proposed/system_config.toml | 45126af4dd3913ad5112f43d0ef1a60f421f276ea85785c7a06044fa1c406094 |
| scripts/training/run_corrected_actor_lr_5e_minus_4.py | 9058a710ea6e774f9949a0c784e8145fb17a8dba5bc81a905cd783f0f368dfc7 |
| scripts/training/run_corrected_actor_lr_1e_minus_3.py | 76e1e12a970bac5a89583c8ff9b69022d808517f749966e4a68fcb301a80e0fd |
| scripts/training/fresh300_runner_common.py | 2e3c68a9f2f535b39824f5b9ba5d15152bf36175510e0595ea40fb27852118c7 |

## Interpretation limits

These paired one-step estimates use frozen epoch-140 critics and sampled frozen-policy rollout states. Current TD-Q is action-conditional: UPDATE is compared with KEEP, while KEEP is compared with its highest-probability legal UPDATE candidate. Symmetric expected Q averages all legal UPDATE candidates under pi_old(c | UPDATE). This report diagnoses the requested reference comparison and makes no algorithm change.
