# RECOVERY_R1_UAV_OPERATION_CREDIT_ALIGNMENT_AUDIT

Bounded frozen V10-D audit. No training, optimizer step, algorithm patch, reward change, or system-model change was performed.

- Checkpoint: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v10d_decoupled_encoder_branch_kl_6e4_formal\latest_checkpoint.pt`
- Frozen realizations: `[20001, 20100, 20200, 20300]`; origins: `[80, 200]`; witnesses: `16`
- Replay: `DETERMINISTIC_ARGMAX`, origins `0..299`, terminal bootstrap `0.0`

## Outcome

- `H1_STAY_BETTER_COUNT = 16`; `H30_STAY_BETTER_COUNT = 14`.
- Raw GAE reinforces MOVE in `2/16` H1-STAY-better witnesses and `2/14` H30-STAY-better witnesses.
- Centered normalization sign flips: `8`; negative→positive `8`, positive→negative `0`.
- Autograd increases MOVE on `10/16` H1-STAY-better witnesses and `8/14` H30-STAY-better witnesses.
- `AUTOGRAD_DIRECTION_MATCHES_ADVANTAGE_SIGN = YES`; `JOINT_BRANCH_GRADIENT_SIGN_COUPLING = NOT_SUPPORTED`.
- Both UAV heads receive the same transition-level advantage: `YES`; H30 preference conflicts across the two heads: `2` of 8 origins; head-specific operation credit: `NO`.
- Bounded mechanism verdict: shared global GAE credit and centered-std negative→positive sign flips are supported; PPO loss-direction and joint-branch sign-coupling errors are not supported.

## Frozen GAE contract

`gamma = 1.0`, `lambda = 0.95`, normalization mode `centered_std`, entropy coefficient `0.0`.

Production implementation confirmed: `(A_raw - mean(A_raw)) / (population_std(A_raw) + 1e-8)`. `compute_gae()` was reused with terminal bootstrap 0; no alternate GAE formula was introduced.

## Origin-level two-head credit check

| Episode | Origin | A_raw | A_norm | UAV_0 true H30 | UAV_1 true H30 | Conflict |
|---:|---:|---:|---:|---|---|---|
| 20001 | 80 | 36.0552101 | 1.37074149 | STAY | STAY | False |
| 20001 | 200 | -30.0266094 | -0.363552868 | STAY | STAY | False |
| 20100 | 80 | -7.61586428 | 1.26756263 | STAY | MOVE | True |
| 20100 | 200 | -124.848892 | -1.45120203 | STAY | STAY | False |
| 20200 | 80 | -65.5516968 | -0.908242822 | STAY | STAY | False |
| 20200 | 200 | -7.66989088 | 1.31192017 | STAY | STAY | False |
| 20300 | 80 | -36.728447 | 0.895164967 | STAY | STAY | False |
| 20300 | 200 | -53.7773285 | 0.215340748 | MOVE | STAY | True |

The two UAV operation heads are evaluated from one transition-level global GAE value. Therefore this actor path has no head-specific operation advantage; the two true H30 preference conflicts are direct bounded evidence of the credit mismatch.

## Witness sign and loss-direction alignment

| Episode | Origin | UAV | P_MOVE | ΔG_H1 | ΔG_H30 | A_raw | A_norm | Raw→MOVE | Norm→MOVE | dLoss/dMargin full | dLoss/dMargin op-only | Full dir. | Sign consistent |
|---:|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|
| 20001 | 80 | UAV_0 | 0.989751988 | 0.0291795849 | 0.0607565931 | 36.0552101 | 1.37074149 | True | True | -0.0140473815 | -0.0140473815 | INCREASE_MOVE | True |
| 20001 | 80 | UAV_1 | 0.922014921 | 0.0469816327 | 0.0825104383 | 36.0552101 | 1.37074149 | True | True | -0.106897369 | -0.106897369 | INCREASE_MOVE | True |
| 20001 | 200 | UAV_0 | 0.989751231 | 0.0283203545 | 0.0836064597 | -30.0266094 | -0.363552868 | False | False | 0.00372595736 | 0.00372595736 | DECREASE_MOVE | True |
| 20001 | 200 | UAV_1 | 0.926812063 | 0.0283929708 | 0.0589975474 | -30.0266094 | -0.363552868 | False | False | 0.0266076885 | 0.0266076885 | DECREASE_MOVE | True |
| 20100 | 80 | UAV_0 | 0.989815704 | 0.0285115137 | 0.0455628903 | -7.61586428 | 1.26756263 | False | True | -0.0129092038 | -0.0129092038 | INCREASE_MOVE | True |
| 20100 | 80 | UAV_1 | 0.932457821 | 0.0382419804 | -0.0434085212 | -7.61586428 | 1.26756263 | False | True | -0.0856139511 | -0.0856139511 | INCREASE_MOVE | True |
| 20100 | 200 | UAV_0 | 0.989590011 | 0.0285659853 | 0.0386468533 | -124.848892 | -1.45120203 | False | False | 0.0151070748 | 0.0151070748 | DECREASE_MOVE | True |
| 20100 | 200 | UAV_1 | 0.91445023 | 0.0530222072 | 0.342971051 | -124.848892 | -1.45120203 | False | False | 0.124149956 | 0.124149956 | DECREASE_MOVE | True |
| 20200 | 80 | UAV_0 | 0.989533707 | 0.0319298802 | 0.0604494938 | -65.5516968 | -0.908242822 | False | False | 0.00950593315 | 0.00950593315 | DECREASE_MOVE | True |
| 20200 | 80 | UAV_1 | 0.912569444 | 0.0568971559 | 0.104250641 | -65.5516968 | -0.908242822 | False | False | 0.0794081688 | 0.0794081688 | DECREASE_MOVE | True |
| 20200 | 200 | UAV_0 | 0.989416472 | 0.0282330617 | 0.0396081651 | -7.66989088 | 1.31192017 | False | True | -0.013884766 | -0.013884766 | INCREASE_MOVE | True |
| 20200 | 200 | UAV_1 | 0.909785369 | 0.0488112778 | 0.0467752671 | -7.66989088 | 1.31192017 | False | True | -0.118354365 | -0.118354365 | INCREASE_MOVE | True |
| 20300 | 80 | UAV_0 | 0.98989923 | 0.0284674057 | 0.156793928 | -36.728447 | 0.895164967 | False | True | -0.00904182158 | -0.00904182158 | INCREASE_MOVE | True |
| 20300 | 80 | UAV_1 | 0.935983671 | 0.0272484774 | 0.00239276902 | -36.728447 | 0.895164967 | False | True | -0.0573051646 | -0.0573051646 | INCREASE_MOVE | True |
| 20300 | 200 | UAV_0 | 0.989611669 | 0.0249714992 | -0.114449807 | -53.7773285 | 0.215340748 | False | True | -0.00223703356 | -0.00223703356 | INCREASE_MOVE | True |
| 20300 | 200 | UAV_1 | 0.916669677 | 0.0566018089 | 0.0996057841 | -53.7773285 | 0.215340748 | False | True | -0.0179444216 | -0.0179444216 | INCREASE_MOVE | True |

A negative `dLoss/dMargin` means gradient descent increases the MOVE-vs-STAY operation margin. The probe used the actual V10-D joint evaluator, old=current policy (`ratio=1`), and did not call `optimizer.step()`.

## Advantage versus real MOVE marginal value

The true marginal is `G_MOVE - G_STAY = -ΔG`. Correlations below use all 16 frozen witnesses.

| Advantage | True marginal | Pearson | Spearman |
|---|---|---:|---:|
| A_RAW | MARGINAL_H1_MOVE_VALUE | 0.161532373 | 0.136097057 |
| A_NORM | MARGINAL_H1_MOVE_VALUE | 0.177402486 | 0.112428004 |
| A_RAW | MARGINAL_H30_MOVE_VALUE | 0.370541859 | 0.213021481 |
| A_NORM | MARGINAL_H30_MOVE_VALUE | 0.401385865 | 0.236690534 |

## Interpretation

The bounded evidence separates the mechanisms: global GAE supplies one shared scalar to both UAV heads, centered normalization is checked explicitly for sign flips, the real autograd probe checks the loss direction, and the joint-branch decomposition checks sign coupling at the operation-logit margin.

`TRAINING_EXECUTED = NO`; `OPTIMIZER_STEP_EXECUTED = NO`; `SOURCE_MODIFIED = NO`; `TEMP_FILES_REMOVED = YES`; `GIT_DIFF_CHECK = PASS`; `PUSHED = NO`.
