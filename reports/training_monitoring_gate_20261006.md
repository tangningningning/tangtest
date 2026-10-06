# 5A Training Monitoring Gate — 2026-10-06

## Scope and outcome

Read-only review of the saved 15-episode formal_unified_cache_cardinality_corrected_gate run. No environment episode, PPO update, optimizer step, algorithm/config edit, or data-layer work was performed.

- **Training mechanism:** PASS with sustained Critic gradient-clipping caveat. Actor gradients and optimizer counters are nonzero; PPO KL/clip metrics show no large or explosive policy update; Macro/Micro EV remains positive/high without visible value-loss runaway. Macro Critic grad-clip fraction is 1.0 in all 15 updates; Micro Critic post-clip norm is 0.5 in all 15.
- **Convergence:** not established. Return is volatile and episode 15 drops sharply; Cache group probability, entropy, KL, and EV do not jointly show a stable plateau.
- **Longer training:** allowed/recommended as a separate controlled follow-up, retaining Critic clipping as a monitor. Not started here.
- **Deterministic Cache:** current Macro branch argmaxes corrected individual candidates. It is not group-mass deterministic and can choose KEEP while total UPDATE group mass is higher. Flag as a deterministic evaluation protocol candidate issue.

## Run and integrity

- Episodes/updates/critic diagnostics: 15/15/15; slot records: 4500 (300 per episode). Checkpoint counters match update-metric sums: True.
- Slot reward reconstruction max error: 0; episode Return vs summed slot G max error: 0.
- V14b source SHA matches run metadata: True (82a835d7e5be6325566f14ffa65afeb388d7e3ba47e4ddce487a14e6b7d1e55c). Episode 5/10 checkpoints are absent.

## Return trajectory

| Episode | Return | Mean slot reward | Macro frames | KEEP | UPDATE | UAV move mean | Duration 1 / 3 / 5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -1837.93 | -6.126 | 98 | 0.439 | 0.561 | 0.652 | 0.347 / 0.265 / 0.388 |
| 2 | 603.87 | 2.013 | 101 | 0.455 | 0.545 | 0.622 | 0.366 / 0.277 / 0.356 |
| 3 | -1387.67 | -4.626 | 105 | 0.486 | 0.514 | 0.683 | 0.371 / 0.314 / 0.314 |
| 4 | 1165.02 | 3.883 | 100 | 0.440 | 0.560 | 0.682 | 0.350 / 0.300 / 0.350 |
| 5 | 2351.09 | 7.837 | 106 | 0.538 | 0.462 | 0.692 | 0.368 / 0.340 / 0.292 |
| 6 | 675.87 | 2.253 | 102 | 0.480 | 0.520 | 0.695 | 0.324 / 0.373 / 0.304 |
| 7 | 614.59 | 2.049 | 97 | 0.495 | 0.505 | 0.732 | 0.309 / 0.330 / 0.361 |
| 8 | 748.46 | 2.495 | 88 | 0.420 | 0.580 | 0.693 | 0.182 / 0.420 / 0.398 |
| 9 | -965.78 | -3.219 | 104 | 0.615 | 0.385 | 0.698 | 0.317 / 0.413 / 0.269 |
| 10 | 335.68 | 1.119 | 104 | 0.567 | 0.433 | 0.762 | 0.346 / 0.356 / 0.298 |
| 11 | 2689.75 | 8.966 | 96 | 0.604 | 0.396 | 0.750 | 0.281 / 0.375 / 0.344 |
| 12 | 2791.80 | 9.306 | 110 | 0.655 | 0.345 | 0.757 | 0.364 / 0.400 / 0.236 |
| 13 | 2200.03 | 7.333 | 102 | 0.529 | 0.471 | 0.802 | 0.304 / 0.412 / 0.284 |
| 14 | 1222.78 | 4.076 | 110 | 0.536 | 0.464 | 0.765 | 0.418 / 0.291 / 0.291 |
| 15 | -1642.77 | -5.476 | 104 | 0.490 | 0.510 | 0.822 | 0.404 / 0.240 / 0.356 |

| Group | Return mean | Sample SD | Min | Max |
|---|---:|---:|---:|---:|
| First 5 | 178.88 | 1760.20 | -1837.93 | 2351.09 |
| Middle 5 | 281.76 | 714.70 | -965.78 | 748.46 |
| Last 5 | 1452.32 | 1838.26 | -1642.77 | 2791.80 |
| All 15 | 637.65 | 1534.30 | -1837.93 | 2791.80 |

Best episode 12: 2791.80; worst episode 1: -1837.93. First-5 to last-5 mean change: 1273.44.

**3-episode rolling mean:** E1–3 -873.9, E2–4 127.1, E3–5 709.5, E4–6 1397.3, E5–7 1213.8, E6–8 679.6, E7–9 132.4, E8–10 39.5, E9–11 686.5, E10–12 1939.1, E11–13 2560.5, E12–14 2071.5, E13–15 593.3

**5-episode rolling mean:** E1–5 178.9, E2–6 681.6, E3–7 683.8, E4–8 1111.0, E5–9 684.8, E6–10 281.8, E7–11 684.5, E8–12 1120.0, E9–13 1410.3, E10–14 1848.0, E11–15 1452.3

The last-five average is higher than the first-five average, but the trace is volatile; different seeds prevent full attribution to policy learning.

## PPO update health

Run config: target KL 0.1, clip epsilon 0.2, max grad norm 0.5.

### Macro

| Metric | All 15 mean ± SD [min, max] | First 5 mean | Middle 5 mean | Last 5 mean |
|---|---:|---:|---:|---:|
| actor_loss | -0.0020922 ± 0.0022896 [-0.0081765, 0.00095859] | -0.0033339 | -0.0021067 | -0.00083606 |
| value_loss | 3862 ± 1884.8 [1850, 8359.1] | 3643.2 | 4382.7 | 3560 |
| approx_kl | 0.00021204 ± 0.00023192 [1.0812e-05, 0.00084833] | 0.00043875 | 0.00011551 | 8.1847e-05 |
| clip_fraction | 0 ± 0 [0, 0] | 0 | 0 | 0 |
| entropy | 4.4966 ± 0.24947 [4.0321, 4.7451] | 4.6587 | 4.6148 | 4.2162 |
| actor_grad_before_clip | 0.13838 ± 0.043445 [0.086327, 0.21447] | 0.14513 | 0.133 | 0.137 |
| actor_grad_after_clip | 0.13838 ± 0.043445 [0.086327, 0.21447] | 0.14513 | 0.133 | 0.137 |
| critic_grad_before_clip | 234.88 ± 65.876 [128.07, 355.44] | 268.67 | 247.84 | 188.13 |
| critic_grad_after_clip | 0.5 ± 3.71e-08 [0.5, 0.5] | 0.5 | 0.5 | 0.5 |
| actor_optimizer_steps | 4 ± 0 [4, 4] | 4 | 4 | 4 |
| critic_optimizer_steps | 50 ± 0 [50, 50] | 50 | 50 | 50 |
| explained_variance | 0.98979 ± 0.0062473 [0.97585, 0.99673] | 0.99206 | 0.98877 | 0.98854 |
| prediction_to_target_std_ratio | 0.99277 ± 0.0074039 [0.97787, 1.0032] | 0.99045 | 0.99399 | 0.99387 |

### Micro

| Metric | All 15 mean ± SD [min, max] | First 5 mean | Middle 5 mean | Last 5 mean |
|---|---:|---:|---:|---:|
| actor_loss | -0.0091342 ± 0.0075995 [-0.025611, 0.00076392] | -0.0070387 | -0.012271 | -0.0080934 |
| value_loss | 14172 ± 18283 [1939.9, 59411] | 19795 | 15132 | 7589.6 |
| approx_kl | 0.007463 ± 0.0025197 [0.0029962, 0.012551] | 0.0068523 | 0.0075322 | 0.0080046 |
| clip_fraction | 0.083556 ± 0.027927 [0.035, 0.13917] | 0.081167 | 0.085 | 0.0845 |
| entropy | 5.2237 ± 0.15999 [4.9539, 5.4424] | 5.0661 | 5.2195 | 5.3856 |
| actor_grad_before_clip | 0.49862 ± 0.071981 [0.37185, 0.63103] | 0.51429 | 0.54556 | 0.43602 |
| actor_grad_after_clip | 0.39134 ± 0.023041 [0.33552, 0.42464] | 0.39982 | 0.40002 | 0.37417 |
| critic_grad_before_clip | 473.39 ± 271.59 [161.97, 1082.7] | 513.3 | 563.64 | 343.25 |
| critic_grad_after_clip | 0.5 ± 1.4433e-08 [0.5, 0.5] | 0.5 | 0.5 | 0.5 |
| actor_optimizer_steps | 40 ± 0 [40, 40] | 40 | 40 | 40 |
| critic_optimizer_steps | 40 ± 0 [40, 40] | 40 | 40 | 40 |
| explained_variance | 0.98566 ± 0.024913 [0.90688, 0.99893] | 0.99278 | 0.97567 | 0.98855 |
| prediction_to_target_std_ratio | 0.99059 ± 0.026671 [0.93161, 1.0229] | 0.98827 | 0.98365 | 0.99984 |

- Macro KL mean/max 0.000212035/0.000848328 versus target 0.1; Macro PPO clip fraction is zero in all updates. This is very conservative Macro movement, not a zero-update condition: Actor gradients and parameter deltas are nonzero.
- Micro KL max 0.0125513; clip-fraction range 0.0350–0.1392. Neither KL approaches target 0.1.
- Critic saturation: Macro explicit clip fraction is 1.0 in all 15. Macro pre-clip mean norm spans 128.1–355.4; Micro post-clip norm is 0.5 in 15/15.
- Persistent Critic clipping is a monitoring concern. It coincides with no non-finite metrics, positive/high Macro/Micro EV, non-constant prediction scales, and no visible value-loss divergence.

## Parameter movement

L2 deltas compare saved checkpoint tensors with reconstructed initialization (seed 27005 and compatible V14b warm-start prefixes); no optimizer was constructed or stepped.

| Module | Ep12 vs init L2 | Ep12 relative | Ep15 vs init L2 | Ep15 relative | Ep15 vs Ep12 L2 |
|---|---:|---:|---:|---:|---:|
| macro_actor_encoder | 0.44293 | 0.041712 | 0.4771 | 0.04493 | 0.12109 |
| macro_cache_candidate_encoder | 0.6036 | 0.063298 | 0.68933 | 0.072289 | 0.19492 |
| macro_cache_scorer_fresh_head | 0.86215 | 0.13147 | 0.88319 | 0.13468 | 0.13661 |
| macro_duration_scorer | 0.21736 | 0.029301 | 0.24148 | 0.032553 | 0.053795 |
| micro_actor_encoder | 1.3542 | 0.15152 | 1.4858 | 0.16625 | 0.75535 |
| micro_candidate_scorer | 1.1937 | 0.12819 | 1.2907 | 0.13861 | 0.53956 |

All compared Actor modules moved by a nonzero amount. The Macro candidate scorer began as a fresh head. Episode-5/10 parameter points are unavailable.

## Critic and policy trajectories

Macro EV min/max 0.9758/0.9967; Micro EV min/max 0.9069/0.9989. Prediction/target scales are comparable; EV is not Actor convergence evidence.

### Corrected Cache group probabilities

Group margin is P_UPDATE_GROUP minus P_KEEP; the old best-individual UPDATE margin is excluded.

| Ep | P_KEEP mean | P_KEEP state SD | P_UPDATE_GROUP | Group margin | Sampled KEEP | Sampled UPDATE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4949 | 0.0039 | 0.5051 | +0.0101 | 0.439 | 0.561 |
| 2 | 0.5027 | 0.0075 | 0.4973 | -0.0054 | 0.455 | 0.545 |
| 3 | 0.5022 | 0.0151 | 0.4978 | -0.0045 | 0.486 | 0.514 |
| 4 | 0.5138 | 0.0181 | 0.4862 | -0.0276 | 0.440 | 0.560 |
| 5 | 0.5225 | 0.0282 | 0.4775 | -0.0450 | 0.538 | 0.462 |
| 6 | 0.5236 | 0.0295 | 0.4764 | -0.0472 | 0.480 | 0.520 |
| 7 | 0.4898 | 0.0258 | 0.5102 | +0.0205 | 0.495 | 0.505 |
| 8 | 0.5080 | 0.0560 | 0.4920 | -0.0159 | 0.420 | 0.580 |
| 9 | 0.5344 | 0.0400 | 0.4656 | -0.0687 | 0.615 | 0.385 |
| 10 | 0.4851 | 0.0451 | 0.5149 | +0.0298 | 0.567 | 0.433 |
| 11 | 0.5894 | 0.0515 | 0.4106 | -0.1789 | 0.604 | 0.396 |
| 12 | 0.5684 | 0.0684 | 0.4316 | -0.1368 | 0.655 | 0.345 |
| 13 | 0.6000 | 0.0561 | 0.4000 | -0.2000 | 0.529 | 0.471 |
| 14 | 0.5746 | 0.0517 | 0.4254 | -0.1492 | 0.536 | 0.464 |
| 15 | 0.5084 | 0.0601 | 0.4916 | -0.0169 | 0.490 | 0.510 |

Corrected P_KEEP first/middle/last-five means: 0.5072/0.5082/0.5682; last-five state-SD mean 0.0576. Episode 15 falls to 0.5084; the end is volatile rather than a smooth plateau.

UPDATE conditional entropy/effective count/top1 are derived from saved per-state P_KEEP, best nonzero probability, and full candidate entropy. Full historical candidate vectors and top-5/top-10 probability mass were not saved.

Duration rates are sampled requested frequencies, not historical probability vectors; all 1/3/5 appear in every episode. UAV0/UAV1 action frequencies and entropy are separate; Micro KL/loss/gradient are joint across UAV heads.

## Reward decomposition

Formula: G = task_utility minus cache_cost minus movement_cost. task_utility already combines delay/energy utility.

| Group | Return mean/episode | Utility mean/episode | Delay s/task | Energy J/task | Cache cost mean/episode | Movement cost mean/episode |
|---|---:|---:|---:|---:|---:|---:|
| First 5 | 178.88 | 188.94 | 27.696 | 3.157 | 9.1058 | 0.9583 |
| Middle 5 | 281.76 | 290.68 | 28.715 | 2.823 | 7.8854 | 1.0297 |
| Last 5 | 1452.32 | 1461.00 | 19.779 | 2.018 | 7.5609 | 1.1182 |

Last-five minus first-five Return is 1273.44/episode; utility changes 1272.06, cache cost -1.5449, movement cost 0.1599. Cost changes are small compared with utility movement. Different episode seeds/task loads prevent exact policy-vs-difficulty attribution.

## Deterministic Cache semantics

- Unified + cardinality-corrected Macro mode argmaxes corrected individual candidate logits. Correction changes total UPDATE group mass by subtracting log of legal UPDATE candidate count; deterministic selection still compares KEEP against each individual UPDATE candidate.
- Therefore P_UPDATE_GROUP may exceed P_KEEP while every individual UPDATE probability is below P_KEEP; flat argmax then chooses KEEP.
- The group-aware helper sums probability by group, then selects a raw-score UPDATE candidate. It is called for Micro UAV STAY/MOVE, not unified Macro Cache.
- Evaluator supports deterministic/stochastic and defaults to deterministic. No saved evaluation artifact freezes the paper protocol. Current executable default is flat corrected-candidate Cache argmax; the final paper protocol is not separately fixed.
- Source: policy.py:624–636, policy.py:109–144, policy.py:1219–1233, evaluator.py:247–264, 319–326, 388–399.

## Final 5A decision

- **Training mechanism healthy:** PASS with persistent Critic clipping caveat; no non-finite data, zero Actor gradients/steps, large KL, critic divergence, or update-counter mismatch found.
- **15 episodes converged:** NO / evidence insufficient.
- **Longer training allowed:** YES as a separate follow-up; not launched.
- **Deterministic Cache:** evaluator defaults to flat individual-candidate argmax after correction, not group-mass deterministic. Final paper-test protocol is not separately fixed and should be settled before formal testing.

## Limits

- No episodes rerun; results are from the existing 15-episode artifact.
- Duration probability vectors and full UPDATE candidate vectors were not persisted.
- Episode-5/10 checkpoints are absent.
- Different episode seeds prevent full separation of policy learning and realization difficulty.
