# Cache Trigger Task-Marginalized Raw Generalization Gate

## Final gate

- TASK_RANDOMNESS_ROOT_CAUSE = NO
- CONDITIONAL_EXPECTED_DELTA_Q_STATE_DEPENDENT = YES
- TASK_MARGINALIZED_RAW_LINEAR_SIGNAL_GENERALIZES = NO
- TASK_MARGINALIZED_RAW_NONLINEAR_SIGNAL_GENERALIZES = NO
- TRIGGER_OBSERVATION_INFORMATION_INSUFFICIENT = YES
- ENCODER_OR_LEARNING_FAILURE_REMAINS_PRIMARY = NO
- ROOT_CAUSE_BRANCH = OBSERVATION_INFORMATION_INSUFFICIENT

## Sampling and variance

- 512 fixed training states; reused TASK_DRAW_0 = 512; baseline reuse failures = 0.
- New task draws = 2920; new exact execute_slot Q branches = 55506.
- Final K counts: K4=420, K8=37, K16=19, K32=36.
- Within-state variance = 5.24476; between-state variance = 32.8489; stochasticity fraction = 0.1377.
- Conditional sign unreliable after K<=32: 22/512 (4.3%); all remain in the continuous regression dataset.

## Raw-context probes

- Ridge: median head AUC=0.5688; median head R2=-0.2896; heads AUC>=0.65: 6/16; gate=NO.
- Pooled MLP: OOF AUC=0.5224; R2=-3.2850; sign accuracy=0.5156; heads AUC>0.60: 4/16; held-fold AUCs=0.6000, 0.5152, 0.5440, 0.5152; gate=NO.
- Raw linear per-head OOF metrics use TASK_MARG_RAW_LINEAR_SPEARMAN/R2/SIGN_AUC/SIGN_ACCURACY in JSON; pooled MLP metrics use TASK_MARG_RAW_MLP_SPEARMAN/R2/SIGN_AUC/SIGN_ACCURACY.
- Pooled MLP TASK_MARG_RAW_MLP_SPEARMAN/R2/SIGN_AUC/SIGN_ACCURACY = 0.0656 / -3.2850 / 0.5224 / 0.5156.
- Ridge protocol reproduction: maximum absolute difference against the previous 512-row probe = 6.66e-16; all metrics reproduced within 1e-6 = True.
- Prior MLP protocol reconstruction check: pooled AUC 0.5134 vs saved 0.5166; absolute difference 0.0031.

## Integrity

- TRAINING_EXECUTED = NO; PPO_BACKWARD_CALLS = 0; OPTIMIZER_STEP_CALLS = 0; VALIDATION_EXECUTED = NO; TEST_EXECUTED = NO.
- CHECKPOINT_PARAMETERS_UNCHANGED = YES; OBSERVATION_MODIFIED = NO.
- Existing baseline reuse: 512; reuse failures: 0.

## Reproducibility and limits

Only the four specified training realizations and the existing 512 mixed/error states were used. Task realizations came from the formal generator with per-slot rho_star fixed. actual_tasks were supplied only to execute_slot; policy context, candidates, candidate probabilities, other Cache decisions, UAV decisions, and actual vehicle positions were held fixed. The frozen critic supplied the bootstrap. The pooled MLP is a NumPy diagnostic fit required by this gate, not PPO training.

## Per-head raw Ridge OOF

| Run / epoch / head | Role | Spearman | R2 | AUC | Sign accuracy |
|---|---|---:|---:|---:|---:|
| 1e-3 e10 h2 0 | SV | 0.2284 | -0.2827 | 0.6468 | 0.5625 |
| 1e-3 e10 h4 59 | SV | 0.4523 | 0.0905 | 0.7729 | 0.7500 |
| 1e-3 e10 h5 RSU_0 | RSU | 0.4186 | -0.1690 | 0.7571 | 0.6875 |
| 1e-3 e10 h8 UAV_1 | UAV | 0.1378 | -0.2400 | 0.4941 | 0.5312 |
| 1e-3 e20 h0 46 | SV | 0.4538 | 0.1269 | 0.6588 | 0.5938 |
| 1e-3 e40 h6 RSU_1 | RSU | -0.0704 | -0.9432 | 0.4049 | 0.5625 |
| 1e-3 e50 h3 23 | SV | -0.1041 | -0.2965 | 0.5108 | 0.5938 |
| 1e-3 e50 h7 UAV_0 | UAV | 0.3977 | 0.1122 | 0.7098 | 0.6875 |
| 5e-4 e10 h5 RSU_0 | RSU | -0.3471 | -0.6886 | 0.3056 | 0.3750 |
| 5e-4 e10 h6 RSU_1 | RSU | 0.0073 | -0.1125 | 0.5142 | 0.5625 |
| 5e-4 e20 h0 46 | SV | -0.1202 | -0.6093 | 0.4170 | 0.4688 |
| 5e-4 e20 h7 UAV_0 | UAV | -0.1683 | -1.0084 | 0.4219 | 0.4688 |
| 5e-4 e20 h8 UAV_1 | UAV | 0.2265 | -0.0996 | 0.7273 | 0.7188 |
| 5e-4 e30 h4 59 | SV | 0.3640 | -0.5184 | 0.7792 | 0.7188 |
| 5e-4 e50 h1 10 | SV | 0.1294 | -0.3085 | 0.5455 | 0.6250 |
| 5e-4 e50 h3 23 | SV | 0.1895 | -0.5331 | 0.5922 | 0.5000 |

## Per-head pooled MLP OOF AUC

| Run / epoch / head | Role | OOF AUC | OOF R2 |
|---|---|---:|---:|
| 1e-3 e10 h2 0 | SV | 0.5476 | -11.6605 |
| 1e-3 e10 h4 59 | SV | 0.4976 | -9.2402 |
| 1e-3 e10 h5 RSU_0 | RSU | 0.6275 | -4.0089 |
| 1e-3 e10 h8 UAV_1 | UAV | 0.5137 | -4.2404 |
| 1e-3 e20 h0 46 | SV | 0.5725 | -13.0108 |
| 1e-3 e40 h6 RSU_1 | RSU | 0.3684 | -1.6964 |
| 1e-3 e50 h3 23 | SV | 0.6190 | -0.8857 |
| 1e-3 e50 h7 UAV_0 | UAV | 0.3529 | -16.1077 |
| 5e-4 e10 h5 RSU_0 | RSU | 0.5754 | -5.6924 |
| 5e-4 e10 h6 RSU_1 | RSU | 0.4696 | -5.4152 |
| 5e-4 e20 h0 46 | SV | 0.6073 | -43.2433 |
| 5e-4 e20 h7 UAV_0 | UAV | 0.4492 | -12.9227 |
| 5e-4 e20 h8 UAV_1 | UAV | 0.2636 | -20.1353 |
| 5e-4 e30 h4 59 | SV | 0.5167 | -4.2566 |
| 5e-4 e50 h1 10 | SV | 0.6797 | -1.3098 |
| 5e-4 e50 h3 23 | SV | 0.5137 | -1.7324 |

## Reuse audit

- Per-group checkpoint and parameter hashes, run-config hash, fixed realization input hashes, and state-match counts are recorded in checkpoint_replay_integrity in JSON.
- Source SHA256 comparisons to the symmetric-Q source manifest are recorded in source_hash_audit in JSON.

## NEXT_FIX_SCOPE

“下一步静态定位当前系统中哪些合法 pre-decision、value-relevant 信息未进入 Cache Trigger observation。”
