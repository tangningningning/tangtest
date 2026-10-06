# Cache Trigger Saturation Recoverability Diagnosis

## Outcome

POLICY_SATURATION_OBSERVED = YES
WRONG_POLARIZED_STATES_HAVE_WEAK_CORRECTIVE_SCORE = INCONCLUSIVE
EXPECTED_BINARY_GRADIENT_VANISHES_IN_WRONG_POLARIZATION = INCONCLUSIVE
SATURATION_ASSOCIATED_WITH_SMALL_TRIGGER_GRADIENT = YES
CACHE_TRIGGER_PREMATURE_DETERMINIZATION_SUPPORTED = INCONCLUSIVE
SYMMETRIC_REFERENCE_ALONE_LIKELY_SUFFICIENT = INCONCLUSIVE
ENTROPY_OR_EXPLORATION_REGULARIZATION_JUSTIFIED = INCONCLUSIVE

## Evidence

This diagnosis reused all 576 per-sample rows from the existing symmetric expected-Q report. All requested probability, action, expected-Q, raw TD-Q, and oriented TD-Q fields were present, so no Q branch, rollout, training step, validation, or test was executed.

| Checkpoint | Strongly saturated | Extremely saturated | Wrong N | Non-polarized N | Score median wrong/non-polarized | Score ratio | Expected-grad median abs wrong/non-polarized | Median ratio | Mean ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5e-4 | 0.871528 | 0.555556 | 67 | 0 | 0.00693861 / undefined | undefined | 0.0326934 / undefined | undefined | undefined |
| 1e-3 | 0.878472 | 0.642361 | 100 | 0 | 0.00177472 / undefined | undefined | 0.00476787 / undefined | undefined | undefined |

The score comparison uses the requested sampled-action score-function magnitude. The expected-gradient comparison uses abs(p_update * (1-p_update) * delta_q_expected). Std-only TD-Q normalization is not involved.

## Head-level saturation and production trigger gradients

The association is computed independently per checkpoint across nine heads. Saturation margin is the median over 32 existing symmetric samples per head. Production trigger encoder gradient norm is the A_GLOBAL median over 12 fixed production minibatches.

| Checkpoint | Spearman rho | Two-sided p | Interpretation |
|---|---:|---:|---|
| 5e-4 | 0.9 | 0.000943062 | smaller margin tracks smaller gradient |
| 1e-3 | 0.75 | 0.0199421 | smaller margin tracks smaller gradient |

| Checkpoint | Head | Role | Margin median | Trigger encoder grad-norm median |
|---|---|---|---:|---:|
| 5e-4 | 0 (46) | SV | 7.02271e-06 | 7.23163e-06 |
| 5e-4 | 1 (10) | SV | 0.00642855 | 7.34042e-05 |
| 5e-4 | 2 (0) | SV | 2.61159e-05 | 2.78764e-05 |
| 5e-4 | 3 (23) | SV | 0.00291147 | 3.08688e-05 |
| 5e-4 | 4 (59) | SV | 7.63592e-05 | 1.06344e-05 |
| 5e-4 | 5 (RSU_0) | RSU | 1.64861e-05 | 1.85857e-05 |
| 5e-4 | 6 (RSU_1) | RSU | 0.000162542 | 7.81395e-05 |
| 5e-4 | 7 (UAV_0) | UAV | 0.00729269 | 0.000134647 |
| 5e-4 | 8 (UAV_1) | UAV | 0.0247071 | 0.000774652 |
| 1e-3 | 0 (46) | SV | 0.00436386 | 4.66337e-05 |
| 1e-3 | 1 (10) | SV | 1.98982e-07 | 1.40725e-07 |
| 1e-3 | 2 (0) | SV | 9.38867e-06 | 1.94215e-05 |
| 1e-3 | 3 (23) | SV | 2.01252e-05 | 2.20152e-05 |
| 1e-3 | 4 (59) | SV | 0.000114959 | 2.28656e-06 |
| 1e-3 | 5 (RSU_0) | RSU | 0.000576586 | 7.9109e-06 |
| 1e-3 | 6 (RSU_1) | RSU | 0.000294668 | 6.79241e-06 |
| 1e-3 | 7 (UAV_0) | UAV | 0.00641498 | 9.92542e-05 |
| 1e-3 | 8 (UAV_1) | UAV | 0.0492894 | 0.00258016 |

## Grouped statistics

The JSON contains every requested statistic by checkpoint, all nine heads, all roles, and wrong-polarized/correct-polarized/non-polarized category. This compact table shows checkpoint/category aggregates.

| Checkpoint | Category | N | P(update) mean | Entropy mean | Margin median | Strong rate | Extreme rate | Score median | Score p10/p90 | Expected grad mean abs | Expected grad median abs | Delta-Q mean | abs(Delta-Q) median |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5e-4 | all | 288 | 0.439857 | 0.0246057 | 0.000162542 | 0.871528 | 0.555556 | 0.000162542 | 1.05195e-05/0.0181512 | 0.0287407 | 0.00379087 | -37.2308 | 16.0281 |
| 5e-4 | wrong-polarized | 67 | 0.973185 | 0.0608065 | 0.00693861 | 0.626866 | 0.0149254 | 0.00693861 | 0.00300039/0.0260238 | 0.0794341 | 0.0326934 | -5.72647 | 4.63277 |
| 5e-4 | correct-polarized | 221 | 0.278169 | 0.0136307 | 7.60894e-05 | 0.945701 | 0.719457 | 7.60894e-05 | 7.91562e-06/0.00756164 | 0.0133721 | 0.00283518 | -46.7818 | 25.9347 |
| 5e-4 | non-polarized | 0 | undefined | undefined | undefined | undefined | undefined | undefined | undefined/undefined | undefined | undefined | undefined | undefined |
| 1e-3 | all | 288 | 0.659874 | 0.0302772 | 0.000271355 | 0.878472 | 0.642361 | 0.000271355 | 2.73309e-07/0.0397735 | 0.0361798 | 0.00196012 | -38.4354 | 8.12706 |
| 1e-3 | wrong-polarized | 100 | 0.989281 | 0.0474922 | 0.00177472 | 0.79 | 0.43 | 0.00177472 | 0.000121831/0.0442973 | 0.0666082 | 0.00476787 | -5.90479 | 4.43414 |
| 1e-3 | correct-polarized | 188 | 0.484658 | 0.0211203 | 4.69986e-05 | 0.925532 | 0.755319 | 4.69986e-05 | 2.06495e-07/0.00805502 | 0.0199945 | 0.00148301 | -55.7389 | 55.2239 |
| 1e-3 | non-polarized | 0 | undefined | undefined | undefined | undefined | undefined | undefined | undefined/undefined | undefined | undefined | undefined | undefined |

## Causal interpretation and limits

Wrong-polarized rows have an expected-Q direction opposing the policy's polarized choice. Their median sampled-action score magnitudes are 0.00694 and 0.00177, and their median absolute expected binary-logit gradients are 0.0327 and 0.00477. These values show strong absolute attenuation, but all 576 samples are polarized, leaving no non-polarized denominator for the requested relative test. The weak-corrective-score, vanishing-gradient, and premature-determinization conclusions therefore remain INCONCLUSIVE rather than converting the absolute values into an unsupported ratio.

The head correlation is observational and has only nine points per checkpoint. It checks whether smaller margins track smaller production gradients; it cannot identify saturation as the sole source of head-gradient variation.

A symmetric expected-Q reference supplies direction but does not remove binary-logit p*(1-p) attenuation. The missing non-polarized comparison prevents a sufficiency conclusion, so symmetric-reference sufficiency and entropy/exploration justification remain INCONCLUSIVE. This report does not select or change a coefficient.

## Integrity

- Symmetric input SHA256: 13baacafa929f12631fb681d325e2d168bb4399eec058d36ab573f39c5b758e7
- Gradient input SHA256: 3d388713d1a3ff5890e16ec199d5a44433e384bbc94a2ecd9f9b0654198d2396
- Q branches re-executed: 0
- TRAINING_EXECUTED = NO
- OPTIMIZER_STEP_CALLS = 0
- VALIDATION_EXECUTED = NO
- TEST_EXECUTED = NO
- Algorithm, entropy coefficient, PPO loss, encoder, checkpoint, and environment data were not modified.
