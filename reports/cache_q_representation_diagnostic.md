# Cache Q representation diagnostic

## Scope and fixed objective

This is an offline representation diagnostic using only the persisted Cache Exact-Q dataset. No environment, Matching, KKT, PPO, or new exact branch was executed. The current teacher-aligned MLP baseline was read from its existing artifact and was not retrained.

- Dataset schema: `True`; candidate alignment: `PASS`.
- Target: `KEEP_RELATIVE_DELTA_Q`; candidate probabilities available: `False`.
- Fixed train-only robust scale: `13.26903992`; listwise temperature: `1`.
- Objective: `group-balanced SmoothL1(Delta/ROBUST_SCALE) + existing Exact-Q listwise KL; operation-gap term is zero because candidate probabilities are absent`.

## Baseline current teacher-aligned MLP

| Held-out metric | Value |
|---|---:|
| Delta Spearman | `0.405286155` |
| Recall@3 | `0.6111111111` |
| Recall@5 | `0.8888888889` |
| Normalized regret p95 | `1` |

## Per-seed metrics

The following table contains every requested metric for train, validation, and held-out for each fixed seed.

| Architecture | Seed | Split | Delta Spearman | Argmax | R@1 | R@3 | R@5 | Regret mean | Regret p95 | Norm mean | Norm p95 | Norm max | KL mean | KL p95 | Delta MAE | Delta RMSE |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | 5101 | train | 0.4378735928 | 0.5416666667 | 0.5416666667 | 0.8055555556 | 0.875 | 1.988876647 | 1.271987319 | 0.03453527451 | 0.1119138349 | 0.9995989328 | 0.09759629559 | 0.3916855888 | 7.351604524 | 11.57022452 |
| R1 | 5101 | validation | 0.2056757319 | 0.1666666667 | 0.1666666667 | 0.3888888889 | 0.5 | 4.673578872 | 24.2962918 | 0.5295174576 | 1 | 1 | 0.820648442 | 2.205659346 | 5.504085219 | 6.587675376 |
| R1 | 5101 | held_out | 0.2658231984 | 0.25 | 0.25 | 0.6111111111 | 0.8333333333 | 4.479266394 | 24.35725188 | 0.3852690131 | 1 | 1 | 0.6721349351 | 2.402689805 | 9.268593314 | 10.47822236 |
| R1 | 5102 | train | 0.5266593494 | 0.6666666667 | 0.6666666667 | 0.8055555556 | 0.875 | 1.897668064 | 1.13080337 | 0.03608403038 | 0.138955929 | 0.9996976325 | 0.1047435134 | 0.3478539993 | 8.0570026 | 12.50750902 |
| R1 | 5102 | validation | 0.3876400808 | 0.2222222222 | 0.2222222222 | 0.5555555556 | 0.6111111111 | 2.016799404 | 9.170969629 | 0.3540508289 | 1 | 1 | 0.838213722 | 2.759019701 | 5.178729114 | 6.509689551 |
| R1 | 5102 | held_out | 0.3787656612 | 0.3611111111 | 0.3611111111 | 0.6388888889 | 0.7222222222 | 3.295162271 | 18.94662404 | 0.3145870123 | 1 | 1 | 0.6503798915 | 2.231477116 | 11.57559202 | 12.91056268 |
| R1 | 5103 | train | 0.6604658078 | 0.6805555556 | 0.6805555556 | 0.8611111111 | 0.8888888889 | 0.202493608 | 0.6488098621 | 0.005349033866 | 0.03815038707 | 0.07306007324 | 0.01697021911 | 0.06787271686 | 5.268080339 | 9.071016687 |
| R1 | 5103 | validation | 0.1914622934 | 0.2222222222 | 0.2222222222 | 0.3888888889 | 0.4444444444 | 1.809105376 | 7.478925878 | 0.4434167873 | 1 | 1 | 0.8465131034 | 2.47052368 | 4.986758413 | 5.863182315 |
| R1 | 5103 | held_out | 0.4559507637 | 0.5 | 0.5 | 0.6944444444 | 0.8333333333 | 2.901716124 | 18.1879828 | 0.3083431623 | 1 | 1 | 0.6216219317 | 2.11368149 | 11.26264173 | 12.41618173 |
| R2 | 5101 | train | 0.6324115196 | 0.6527777778 | 0.6527777778 | 0.8194444444 | 0.8888888889 | 0.2010751234 | 1.118087554 | 0.02170781371 | 0.07573274371 | 0.5917015312 | 0.08595735763 | 0.281548005 | 6.408368794 | 11.05278324 |
| R2 | 5101 | validation | 0.3691120863 | 0.3333333333 | 0.3333333333 | 0.5555555556 | 0.6111111111 | 2.343794935 | 10.34467297 | 0.3977233444 | 1 | 1 | 0.7227893589 | 2.309396202 | 6.6311453 | 7.731443772 |
| R2 | 5101 | held_out | 0.3904729864 | 0.2222222222 | 0.2222222222 | 0.5833333333 | 0.8333333333 | 5.828701803 | 25.07272696 | 0.3603671524 | 1 | 1 | 0.7221554795 | 2.274064825 | 13.27595993 | 14.70192322 |
| R2 | 5102 | train | 0.7497019302 | 0.6944444444 | 0.6944444444 | 0.8611111111 | 0.9583333333 | 0.2144777709 | 0.9923791409 | 0.004087625619 | 0.03517470999 | 0.06676417205 | 0.004722278266 | 0.01872698989 | 1.057676071 | 1.432177686 |
| R2 | 5102 | validation | 0.3730204117 | 0.2777777778 | 0.2777777778 | 0.5 | 0.6111111111 | 2.330391526 | 10.34467297 | 0.2909853219 | 1 | 1 | 0.5241034564 | 1.781793325 | 2.884604779 | 3.665813565 |
| R2 | 5102 | held_out | 0.2528128168 | 0.2222222222 | 0.2222222222 | 0.5277777778 | 0.6388888889 | 4.958684379 | 22.30987954 | 0.3875948266 | 1 | 1 | 0.9278132338 | 3.044988662 | 6.942175984 | 8.697372505 |
| R2 | 5103 | train | 0.5982997906 | 0.5972222222 | 0.5972222222 | 0.8055555556 | 0.8888888889 | 0.177811265 | 0.9857003927 | 0.02306664587 | 0.08758572861 | 0.5917015312 | 0.06522630334 | 0.2361291115 | 6.19214462 | 10.80993724 |
| R2 | 5103 | validation | 0.2556754694 | 0.3333333333 | 0.3333333333 | 0.4444444444 | 0.6666666667 | 3.278231429 | 16.06220741 | 0.4579706864 | 1 | 1 | 0.6433331782 | 1.758805313 | 4.803653655 | 6.072619773 |
| R2 | 5103 | held_out | 0.3023056338 | 0.2777777778 | 0.2777777778 | 0.6111111111 | 0.8888888889 | 5.409418405 | 24.84530783 | 0.403248206 | 1 | 1 | 0.6966647493 | 1.982307735 | 13.00740019 | 14.32930415 |

## Three-seed aggregation

Mean ± population standard deviation across seeds; held-out rows are the primary representation comparison.

| Architecture | Split | Spearman mean ± std | Recall@3 mean ± std | Recall@5 mean ± std | Norm regret p95 mean ± std | Norm regret mean mean ± std | KL mean mean ± std |
|---|---|---:|---:|---:|---:|---:|---:|
| R1 | train | 0.54166625 ± 0.09149035954 | 0.8240740741 ± 0.02618914004 | 0.8796296296 ± 0.006547285011 | 0.0963400503 ± 0.04260161494 | 0.02532277959 ± 0.01413771669 | 0.07310334271 ± 0.0397992157 |
| R1 | validation | 0.261592702 ± 0.08931764188 | 0.4444444444 ± 0.07856742013 | 0.5185185185 ± 0.06928995161 | 1 ± 0 | 0.4423283579 ± 0.07163808555 | 0.8351250892 ± 0.01078269962 |
| R1 | held_out | 0.3668465411 ± 0.07807548412 | 0.6481481481 ± 0.0346449758 | 0.7962962963 ± 0.05237828009 | 1 ± 0 | 0.3360663959 ± 0.03488475853 | 0.6480455861 ± 0.0206878002 |
| R2 | train | 0.6601377468 ± 0.06484447922 | 0.8287037037 ± 0.02360657182 | 0.912037037 ± 0.03273642505 | 0.0661643941 ± 0.02244094032 | 0.01628736173 ± 0.00864433445 | 0.05196864641 ± 0.03446359074 |
| R2 | validation | 0.3326026558 ± 0.05441913117 | 0.5 ± 0.04536092116 | 0.6296296296 ± 0.02618914004 | 1 ± 0 | 0.3822264509 ± 0.06904657012 | 0.6300753312 ± 0.08165312752 |
| R2 | held_out | 0.3151971457 ± 0.05693401961 | 0.5740740741 ± 0.0346449758 | 0.787037037 ± 0.107183675 | 1 ± 0 | 0.3837367283 ± 0.01771740915 | 0.7822111542 ± 0.1034808147 |

## Held-out tail-failure witnesses

Baseline tail threshold: normalized regret `>= 0.8`. Witness group count: `8`.

| Slot | Node | Candidates | Exact best idx | Predicted idx | Exact Delta best | Exact Delta selected | Regret | Norm regret | Exact best-second margin | Predicted best-second margin |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0 | 12 | 8 | 11 | 21.41471481 | -0.1825227737 | 21.59723759 | 1 | 21.43588066 | 0.01978308737 |
| 20 | 3 | 6 | 2 | 0 | 35.02807617 | -0.0253944397 | 35.05347061 | 1 | 0.03342437744 | 0.678706964 |
| 25 | 3 | 6 | 1 | 0 | 15.78331089 | -0.02552127838 | 15.80883217 | 1 | 0.04787540436 | 1.370914925 |
| 30 | 5 | 17 | 0 | 16 | 0.002868056297 | -0.523601532 | 0.5264695883 | 1 | 0.01694887877 | 0.07567375948 |
| 15 | 8 | 21 | 14 | 19 | 24.69795227 | 0.1126260757 | 24.58532619 | 0.948839778 | 0.8966255188 | 0.09906175269 |
| 30 | 1 | 14 | 8 | 10 | 3.193403244 | -0.073782444 | 3.267185688 | 0.8453224762 | 3.234431267 | 0.1376932856 |
| 30 | 7 | 24 | 8 | 14 | 7.164430141 | 1.100006342 | 6.0644238 | 0.8208926426 | 6.053128242 | 0.05684983173 |
| 30 | 2 | 9 | 1 | 6 | -0.0358402729 | -0.1671791077 | 0.1313388348 | 0.8187207076 | 0.01624816656 | 0.2592447709 |

## Train-only coverage diagnostics

For each held-out candidate row, nearest Euclidean distance to a train candidate row using train-only standardized [conditional_context,node_one_hot,candidate_feature]; group distance is the mean over that group's candidate rows.

| Diagnostic | Tail witnesses | Held-out non-tail |
|---|---:|---:|
| NN distance mean | `423.3091328` | `324.3361865` |
| NN distance p90 | `627.4844238` | `627.1782724` |
| Out-of-range feature fraction | `0.5623393715` | `0.5552550656` |

All normalization statistics and feature min/max bounds in this section use train slots 1-8 only. The nearest-neighbor calculation is chunked and uses the small persisted dataset without a large O(N²) materialization.

## Root-cause classification

- `STATE_ACTION_INTERACTION_BOTTLENECK = INCONCLUSIVE`
- `CANDIDATE_SET_CONTEXT_BOTTLENECK = NOT_SUPPORTED`
- `SINGLE_REALIZATION_DATA_COVERAGE_LIMITATION = SUPPORTED`
- `CURRENT_FEATURE_SUFFICIENCY = INCONCLUSIVE`
- `ROOT_CAUSE = SINGLE_REALIZATION_DATA_COVERAGE_LIMITATION`

The classification uses the fixed directional criteria and the three-seed results; no held-out-driven architecture, learning-rate, hidden-size, loss-weight, or robust-scale retuning was performed.

## Protection and artifacts

- `CACHE_Q_CRITIC_USABLE_FOR_POLICY_CREDIT = NO`; candidate probabilities are absent, so UPDATE-vs-KEEP trigger-gap quality remains unverified.
- `NO_POLICY_GRADIENT = YES`; `NO_GLOBAL_CRITIC_GRADIENT = YES`; `NO_UAV_GRADIENT = YES`.
- Additional environment executions, Matching calls, and KKT calls: `0`, `0`, `0`.
- Formal metrics: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\cache_action_q_critic_sparse_audit_gate\offline_cache_q_dataset\representation_diagnostic\diagnostic_metrics.json`
- Formal config: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\cache_action_q_critic_sparse_audit_gate\offline_cache_q_dataset\representation_diagnostic\diagnostic_config.json`
- Formal checkpoints: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\cache_action_q_critic_sparse_audit_gate\offline_cache_q_dataset\representation_diagnostic`
- Report: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\cache_q_representation_diagnostic.md`
- Existing dataset, Variant B, teacher-aligned, and factorized artifacts were not overwritten.
