# RECOVERY_R1_UAV_COUNTERFACTUAL_TD_ALIGNMENT_AUDIT

Bounded frozen V10-D diagnostic. No training, optimizer step, algorithm patch, reward change, or system-model change was performed.

- Checkpoint: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v10d_decoupled_encoder_branch_kl_6e4_formal\latest_checkpoint.pt`
- Frozen realizations: `[20001, 20100, 20200, 20300]`; origins: `[80, 200]`; witnesses: `16`
- Counterfactual path: replayed real states → direct production primitive override → real `execute_slot`/Matching/KKT → formal next-slot observation → frozen global critic.

## Verdict

- `TD_MATCH_H1_COUNT = 15/16`; `TD_MATCH_H30_COUNT = 13/16`.
- H30 STAY-better: `13/14`; H30 MOVE-better: `0/2`.
- Global raw GAE H30 accuracy: `0.750000`; normalized global GAE: `0.500000`; counterfactual TD: `0.812500`.
- `H30_STAY_BETTER_AOP_MOVE_NEGATIVE_COUNT = 13/14`.
- `UAV_OPERATION_COUNTERFACTUAL_TD_CREDIT = SUPPORTED`.
- Limitation: the two H30 MOVE-better witnesses are still both misclassified (`0/2`); this supports bounded TD alignment improvement, not complete recovery of MOVE preference.

## Credit and scaling contract

`QTD_MOVE` is the policy-conditional expectation over all legal MOVE candidates, not the best candidate. Cache and the other UAV action were held at production. New learned critic: `NO`; global critic: `YES`; real Matching/KKT: `YES`.
AOP baseline uses raw `P(STAY)` and `P(MOVE)`; `CENTERING_USED = NO`. Optional diagnostic scaling is std-only with population std `0.0621305261`; `SIGN_PRESERVED_BY_SCALING = YES`.

## Witness alignment

| Episode | Origin | UAV | H1 truth | H30 truth | QTD_STAY | QTD_MOVE | TD diff | AOP_STAY | AOP_MOVE | TD preference | r diff | γ·V diff |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| 20001 | 80 | UAV_0 | STAY | STAY | 2444.77193 | 2444.73163 | -0.0402990636 | 0.0400158578 | -0.000283205759 | STAY | -0.0403727936 | 7.3730037e-05 |
| 20001 | 80 | UAV_1 | STAY | STAY | 2444.94891 | 2444.79539 | -0.153527204 | 0.141481512 | -0.0120456915 | STAY | -0.0379185204 | -0.115608683 |
| 20001 | 200 | UAV_0 | STAY | STAY | 1161.38999 | 1161.35095 | -0.0390394707 | 0.0386555881 | -0.000383882525 | STAY | -0.0388586468 | -0.000180823824 |
| 20001 | 200 | UAV_1 | STAY | STAY | 1161.35967 | 1161.18687 | -0.172798645 | 0.160238384 | -0.0125602604 | STAY | -0.0398711327 | -0.132927512 |
| 20100 | 80 | UAV_0 | STAY | STAY | 2475.53175 | 2475.49161 | -0.040145128 | 0.0397086122 | -0.000436515769 | STAY | -0.0403693753 | 0.000224247284 |
| 20100 | 80 | UAV_1 | STAY | MOVE | 2475.65989 | 2475.49307 | -0.166815681 | 0.155419478 | -0.0113962031 | STAY | -0.0393567107 | -0.12745897 |
| 20100 | 200 | UAV_0 | STAY | STAY | 1067.81548 | 1067.7754 | -0.0400796323 | 0.0396852759 | -0.00039435635 | STAY | -0.0343009588 | -0.00577867352 |
| 20100 | 200 | UAV_1 | STAY | STAY | 1068.03073 | 1067.85486 | -0.17586635 | 0.160757375 | -0.0151089752 | STAY | -0.0385111238 | -0.137355226 |
| 20200 | 80 | UAV_0 | STAY | STAY | 2550.86951 | 2550.82885 | -0.0406610495 | 0.0402022201 | -0.000458829385 | STAY | -0.0404268924 | -0.000234157125 |
| 20200 | 80 | UAV_1 | STAY | STAY | 2551.081 | 2550.96426 | -0.116740241 | 0.106647615 | -0.0100926267 | STAY | -0.0378253941 | -0.0789148471 |
| 20200 | 200 | UAV_0 | STAY | STAY | 1086.27414 | 1086.24089 | -0.0332476994 | 0.0329524733 | -0.00029522606 | STAY | -0.0370124567 | 0.00376475731 |
| 20200 | 200 | UAV_1 | STAY | STAY | 1086.04642 | 1086.0479 | 0.00147398423 | -0.00137337603 | 0.000100608207 | MOVE | -0.0368228464 | 0.0382968307 |
| 20300 | 80 | UAV_0 | STAY | STAY | 2508.32549 | 2508.28524 | -0.0402521868 | 0.039873641 | -0.000378545839 | STAY | -0.0403669052 | 0.000114718376 |
| 20300 | 80 | UAV_1 | STAY | STAY | 2508.34966 | 2508.1828 | -0.166859509 | 0.156065652 | -0.0107938575 | STAY | -0.0402543401 | -0.126605169 |
| 20300 | 200 | UAV_0 | STAY | MOVE | 1210.97046 | 1210.93541 | -0.0350502703 | 0.0345902958 | -0.0004599745 | STAY | -0.0339695354 | -0.00108073487 |
| 20300 | 200 | UAV_1 | STAY | STAY | 1211.24733 | 1211.06062 | -0.186705718 | 0.171156493 | -0.0155492252 | STAY | -0.0403718481 | -0.14633387 |

The TD difference closes as `(r_MOVE − r_STAY) + gamma·(V_MOVE_next − V_STAY_next)` for every witness. `AOP_MOVE < 0` means the operation-specific TD credit favors STAY; `AOP_MOVE > 0` favors MOVE.

## Comparison with existing global GAE

| Signal | H30 sign accuracy | Matches / 16 |
|---|---:|---:|
| Raw global GAE | 0.750000 | 12 |
| Centered/std normalized global GAE | 0.500000 | 8 |
| Counterfactual TD operation credit | 0.812500 | 13 |

No significance test was performed. This is a bounded 16-witness alignment audit only.

`TRAINING_EXECUTED = NO`; `OPTIMIZER_STEP_EXECUTED = NO`; `SOURCE_MODIFIED = NO`; `TEMP_FILES_REMOVED = YES`; `GIT_DIFF_CHECK = PASS`; `PUSHED = NO`.
