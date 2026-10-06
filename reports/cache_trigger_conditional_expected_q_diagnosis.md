# Cache Trigger Conditional Expected-Q Diagnosis

## Required conclusions

- POST_DECISION_TASK_RANDOMNESS_DOMINATES = NO
- REALIZED_DELTA_Q_IS_HIGH_NOISE_TARGET = NO
- CONDITIONAL_EXPECTED_DELTA_Q_STATE_DEPENDENT = YES
- TRIGGER_OBSERVATION_DEFICIENCY_STILL_SUPPORTED_AFTER_MARGINALIZING_TASK_RANDOMNESS = INCONCLUSIVE

TRAINING_EXECUTED = NO; OPTIMIZER_STEP_CALLS = 0; PPO_BACKWARD_CALLS = 0; VALIDATION_EXECUTED = NO; TEST_EXECUTED = NO.

## Protocol

Used epoch-140 corrected-v2 checkpoints and 8 selected training-pool states per run (3 SV, 2 RSU, 3 UAV-cache; wrong-polarization and D-class/correct-polarization controls), with K=32 task draws per state.

Each draw used the formal task_scenario.generate_tasks routine conditional on the selected episode/slot's fixed rho_star. It resampled only task arrivals, service requests, and workload jitter. The state-specific rho_star, Trigger observation, predictions, vehicle positions, cache/UAV pre-state, head, slot/origin, and checkpoint were held fixed. Each sampled task set was reused across that state's KEEP and every UPDATE candidate branch. Actual tasks were passed only to execute_slot, never to the policy.

Replay checks: maximum p_update error = 0.00000000; maximum symmetric-Q baseline error = 0.00000000. All checkpoint parameter hashes matched before and after. All other Cache and both UAV decisions were held fixed to the replayed production transition in every branch.

## Task randomness and variance decomposition

Overall within-state variance = 4.9259, between-state variance = 1404.0764, stochasticity fraction = 0.0035.

| Checkpoint | Within-state variance | Between-state variance | Stochasticity fraction |
|---|---:|---:|---:|
| 5e-4 | 8.6911 | 1999.6783 | 0.0043 |
| 1e-3 | 1.1607 | 654.2442 | 0.0018 |

Median state SNR = 26.6610; median sign stability = 1.0000; states with sign stability <0.80 = 1/16.

## Per-state conditional means

| Run | Head/node | Role | Seed/slot | Class | ΔQ mean | SD | SE | Positive rate | SNR | Sign stability |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 5e-4 e140 | 1/10 | SV | 20001/20 | wrong_polarization | -3.1174 | 0.0562 | 0.0099 | 0.0000 | 55.4278 | 1.0000 |
| 5e-4 e140 | 2/0 | SV | 20001/20 | D_class_control | -91.7971 | 0.0491 | 0.0087 | 0.0000 | 1871.3563 | 1.0000 |
| 5e-4 e140 | 7/UAV_0 | UAV | 20001/220 | wrong_polarization | -17.4566 | 0.2228 | 0.0394 | 0.0000 | 78.3509 | 1.0000 |
| 5e-4 e140 | 5/RSU_0 | RSU | 20100/60 | correct_polarization_control | -115.4752 | 0.5766 | 0.1019 | 0.0000 | 200.2746 | 1.0000 |
| 5e-4 e140 | 1/43 | SV | 20200/180 | correct_polarization | 0.2047 | 7.6253 | 1.3480 | 0.7500 | 0.0268 | 0.7500 |
| 5e-4 e140 | 8/UAV_1 | UAV | 20200/180 | wrong_polarization | -5.2189 | 2.1351 | 0.3774 | 0.0312 | 2.4444 | 0.9688 |
| 5e-4 e140 | 6/RSU_1 | RSU | 20300/220 | correct_polarization_control | -23.8778 | 1.0204 | 0.1804 | 0.0000 | 23.4012 | 1.0000 |
| 5e-4 e140 | 7/UAV_0 | UAV | 20300/20 | correct_polarization | 19.2892 | 2.7639 | 0.4886 | 1.0000 | 6.9790 | 1.0000 |
| 1e-3 e140 | 4/59 | SV | 20001/60 | wrong_polarization | -18.7852 | 0.0969 | 0.0171 | 0.0000 | 193.9033 | 1.0000 |
| 1e-3 e140 | 5/RSU_0 | RSU | 20001/20 | wrong_polarization | -2.6265 | 0.2593 | 0.0458 | 0.0000 | 10.1274 | 1.0000 |
| 1e-3 e140 | 7/UAV_0 | UAV | 20001/20 | wrong_polarization | -1.8257 | 0.3845 | 0.0680 | 0.0000 | 4.7483 | 1.0000 |
| 1e-3 e140 | 4/59 | SV | 20100/20 | correct_polarization | 2.2982 | 0.7161 | 0.1266 | 1.0000 | 3.2095 | 1.0000 |
| 1e-3 e140 | 1/43 | SV | 20200/20 | D_class_control | -75.4435 | 1.7642 | 0.3119 | 0.0000 | 42.7629 | 1.0000 |
| 1e-3 e140 | 5/RSU_0 | RSU | 20200/60 | correct_polarization | 7.8264 | 0.2616 | 0.0462 | 1.0000 | 29.9209 | 1.0000 |
| 1e-3 e140 | 8/UAV_1 | UAV | 20200/60 | wrong_polarization | -15.4233 | 2.3730 | 0.4195 | 0.0000 | 6.4996 | 1.0000 |
| 1e-3 e140 | 7/UAV_0 | UAV | 20300/20 | correct_polarization | 7.0347 | 0.1890 | 0.0334 | 1.0000 | 37.2126 | 1.0000 |

## Role and head grouping

| Run | Role | State count | Mean conditional ΔQ | Positive mean-state rate | Median within-state SD | Median SNR |
|---|---|---:|---:|---:|---:|---:|
| 1e-3 | RSU | 2 | 2.5999 | 0.5000 | 0.2605 | 20.0242 |
| 1e-3 | SV | 3 | -30.6435 | 0.3333 | 0.7161 | 42.7629 |
| 1e-3 | UAV | 3 | -3.4047 | 0.3333 | 0.3845 | 6.4996 |
| 5e-4 | RSU | 2 | -69.6765 | 0.0000 | 0.7985 | 111.8379 |
| 5e-4 | SV | 3 | -31.5700 | 0.3333 | 0.0562 | 55.4278 |
| 5e-4 | UAV | 3 | -1.1288 | 0.3333 | 2.1351 | 6.9790 |

Repeated run/head conditional-mean signs:

- 1e-3 head 1 (SV), n=1: state means [-75.444], positive rate 0.00.
- 1e-3 head 4 (SV), n=2: state means [-18.785, 2.298], positive rate 0.50.
- 1e-3 head 5 (RSU), n=2: state means [-2.627, 7.826], positive rate 0.50.
- 1e-3 head 7 (UAV), n=2: state means [-1.826, 7.035], positive rate 0.50.
- 1e-3 head 8 (UAV), n=1: state means [-15.423], positive rate 0.00.
- 5e-4 head 1 (SV), n=2: state means [-3.117, 0.205], positive rate 0.50.
- 5e-4 head 2 (SV), n=1: state means [-91.797], positive rate 0.00.
- 5e-4 head 5 (RSU), n=1: state means [-115.475], positive rate 0.00.
- 5e-4 head 6 (RSU), n=1: state means [-23.878], positive rate 0.00.
- 5e-4 head 7 (UAV), n=2: state means [-17.457, 19.289], positive rate 0.50.
- 5e-4 head 8 (UAV), n=1: state means [-5.219], positive rate 0.00.

## Raw-context descriptive relation

With only 16 selected states and 400 raw context dimensions, these are descriptive only. LOO nearest-neighbor MAE=39.9349, Spearman=-0.1268, sign accuracy=0.5000.
Unsupervised PC1 versus conditional mean ΔQ: Pearson=0.0688, Spearman=-0.1088, in-sample R²=0.0047. Frozen-policy logit gap versus conditional mean ΔQ: Spearman=0.6500. No generalization conclusion is drawn.

## Interpretation and limitation

The previous raw-observation probe reported TRIGGER_OBSERVATION_INFORMATION_INSUFFICIENT = YES for its single-realization target. This report estimates task-marginalized ΔQ conditional on each frozen pre-decision state. Here task randomness does not dominate overall selected-state variance, so it does not explain the previous probe failure in these representative states. However, N=16 with 400 raw inputs and selected states is too small to generalize the descriptive raw-context relation; the observation-deficiency verdict after marginalization remains INCONCLUSIVE. If within-state variance and sign flipping dominate, the previous information-insufficiency result applies only to predicting realized ΔQ. The present 16-state descriptive sample is too small relative to 400 input dimensions to establish whether the observation predicts conditional expected ΔQ across states; it does not justify adding actual_tasks or changing the observation.

No algorithm, checkpoint, task artifact, prediction, reward, or observation was modified.
