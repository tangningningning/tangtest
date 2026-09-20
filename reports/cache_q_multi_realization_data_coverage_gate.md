# CACHE_Q_MULTI_REALIZATION_DATA_COVERAGE_GATE

This gate reuses the persisted seed=20001 Exact-Q dataset and captures only new frozen-policy realizations 20002–20005. No PPO update or production Cache/UAV/reward/system-model change was made.

## Policy identity and capture

- `SOURCE_POLICY_IDENTITY_MATCH = YES`
- `POLICY_FROZEN = YES`; policy seed `4401`
- policy schema `adaptive_joint_cache_uav_ppo`; control mode `adaptive_joint_cache_uav_reconfiguration`
- reconstructed policy state SHA-256: `dd6110b32af45e64ac160f3f1c5e86c99ea4033b5e8e164b16ea01b12eea3758`
- old 30×9 trigger probability reproduction max error: `2.978e-08`
- old Exact-slot critic-context reproduction max error: `1.043e-07`

| Realization | Capture seconds | rollout env executions | Cache counterfactual executions | candidate-prob alignment |
|---:|---:|---:|---:|---|
| 20002 | 80.960 | 30 | 1891 | PASS |
| 20003 | 108.851 | 30 | 1954 | PASS |
| 20004 | 116.840 | 30 | 1900 | PASS |
| 20005 | 79.445 | 30 | 1791 | PASS |

`CANDIDATE_POLICY_PROBS_CAPTURED = YES`; `CANDIDATE_POLICY_PROBS_ALIGNMENT = PASS`.
Total new Cache counterfactual executions: `7536`.
After capture, offline replay made zero environment, Matching, or KKT calls.

## Offline objective

- target: `KEEP_RELATIVE_DELTA_Q`; `Delta = q_candidate_exact - q_keep_exact`
- current MLP: `CacheActionQCritic` (400 context + 9-node one-hot + update flag + 85 candidate features; 64-64 Tanh MLP)
- fixed train-only robust scale: `13.269039917`; listwise temperature `1.0`
- loss: group-balanced SmoothL1/Huber plus the existing standardized Exact-Q listwise KL; no operation-gap loss
- realization split: train 20001–20003, validation 20004, held-out 20005

## Three-stage held-out aggregation

| Stage | Spearman mean±std | Recall@3 mean±std | Recall@5 mean±std | normalized regret p95 mean±std | listwise KL mean±std |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.204872±0.048465 | 0.370370±0.034645 | 0.509259±0.026189 | 0.995957±0.003640 | 0.829692±0.022109 |
| 2 | 0.148322±0.030480 | 0.287037±0.065473 | 0.453704±0.085867 | 1.000000±0.000000 | 0.954648±0.032968 |
| 3 | 0.089237±0.029497 | 0.250000±0.081776 | 0.388889±0.039284 | 1.000000±0.000000 | 0.915675±0.036793 |

## Three-stage validation aggregation

| Stage | Spearman mean±std | Recall@3 mean±std | Recall@5 mean±std | normalized regret p95 mean±std | listwise KL mean±std |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.251972±0.025990 | 0.296296±0.069290 | 0.574074±0.114156 | 1.000000±0.000000 | 0.783040±0.068634 |
| 2 | 0.186552±0.178343 | 0.351852±0.159302 | 0.444444±0.197724 | 0.980950±0.025584 | 0.747202±0.087275 |
| 3 | 0.315759±0.146058 | 0.425926±0.114156 | 0.574074±0.069290 | 0.820706±0.211047 | 0.577347±0.110344 |

## Coverage-distance learning curve

| Stage | held-out NN distance mean | P90 |
|---:|---:|---:|
| 1 | 119.079000 | 122.076050 |
| 2 | 119.079000 | 122.076050 |
| 3 | 115.639840 | 118.730279 |

## Held-out tail witnesses

| Stage | count mean±std | regret mean | regret p95 |
|---:|---:|---:|---:|
| 1 | 9.667±1.247 | 17.946685 | 43.000137 |
| 2 | 10.333±1.247 | 16.039350 | 42.608429 |
| 3 | 12.333±1.700 | 12.721938 | 42.367812 |

## Stage-3 per-realization retention

| Realization | Spearman mean±std | Recall@5 mean±std | norm regret p95 mean±std |
|---:|---:|---:|---:|
| 20001 | 0.529986±0.074600 | 0.893519±0.047213 | 0.299580±0.340724 |
| 20002 | 0.497340±0.069065 | 0.763889±0.081776 | 0.626084±0.300374 |
| 20003 | 0.454629±0.140404 | 0.768519±0.082037 | 0.848642±0.167008 |

## Offline UPDATE-vs-KEEP gap

| Stage | Split | gap MAE | gap RMSE | gap Spearman | sign accuracy | false update | false keep |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | validation | 2.364523 | 3.728897 | 0.057448 | 0.555556 | 0.000000 | 0.444444 |
| 1 | held_out | 1.546091 | 2.587280 | -0.222051 | 0.638889 | 0.000000 | 0.361111 |
| 2 | validation | 2.395227 | 3.752868 | 0.137943 | 0.555556 | 0.000000 | 0.444444 |
| 2 | held_out | 1.668973 | 2.640352 | -0.224367 | 0.638889 | 0.000000 | 0.361111 |
| 3 | validation | 2.393040 | 3.750101 | 0.221878 | 0.555556 | 0.000000 | 0.444444 |
| 3 | held_out | 1.666791 | 2.639244 | -0.188074 | 0.638889 | 0.000000 | 0.361111 |

## Gate classification

- `MULTI_REALIZATION_DATA_COVERAGE_IMPROVES_GENERALIZATION = INCONCLUSIVE`
- `PERSISTENT_REPLAY_STABLE = YES`
- `UPDATE_KEEP_TRIGGER_GAP_VERIFIED = YES`
- `NEXT_STEP = CACHE_Q_FEATURE_SUFFICIENCY_DIAGNOSTIC` (not executed)
- `CACHE_Q_CRITIC_USABLE_FOR_FORMAL_POLICY_CREDIT = NO`

The formal policy-credit usability gate remains closed regardless of this offline result.

## Protection and runtime

- total capture seconds: `386.096032`
- offline replay training seconds: `272.379638`
- total new Cache counterfactual executions: `7536`
- `OFFLINE_ADDITIONAL_ENVIRONMENT_EXECUTIONS = 0`
- `OFFLINE_ADDITIONAL_MATCHING_CALLS = 0`
- `OFFLINE_ADDITIONAL_KKT_CALLS = 0`
- `POLICY_CHANGED = NO`; `UAV_CHANGED = NO`; `REWARD_CHANGED = NO`; `SYSTEM_MODEL_CHANGED = NO`
- `TRAINING_EXECUTED = NO`; `DEV_SEEDS_USED = NO`; `FINAL_TEST_SEEDS_USED = NO`
