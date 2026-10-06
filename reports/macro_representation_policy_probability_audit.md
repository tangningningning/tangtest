# Macro Representation and Policy Probability Audit

## Result

- Vehicle semantic instability: **YES**; slot IDs are stable within an episode and aligned across valid horizons, but differ across episode realizations. The raw flattened MLP therefore receives the same vehicle-coordinate dimensions for different vehicle IDs across samples.
- Policy probability collapse: **NO**.
- Deterministic evaluator amplification: **YES**.
- Time-to-go alone insufficient for value calibration: **YES**.
- Primary suspect: **MACRO_VEHICLE_REPRESENTATION_SEMANTIC_INSTABILITY**.
- Conditional return noise role: **MATERIAL_BUT_NOT_PRIMARY**.

## 400D Macro observation layout

| Slice | Field | Order |
|---|---|---|
| 0:90 | Cache state | 9 Cache nodes × 10 services, node-major |
| 90:94 | UAV positions | UAV ID order, normalized x/y per UAV |
| 94:344 | Predicted vehicle positions | 5 horizons × 25 slots × normalized x/y |
| 344:394 | Predicted service popularity | horizon-major, 10 service IDs per horizon |
| 394:399 | Horizon-validity mask | one bit per horizon |
| 399:400 | Time-to-go | normalized remaining formal train horizon |

Vehicle block order is horizon-major, then the realization's fixed `vehicle_ids` tuple, then x/y. A missing vehicle at a valid origin/horizon raises an alignment error; only whole invalid terminal horizons are zero-filled.

## Vehicle identity examples

Within an episode the artifact ID vector is fixed across all slots. Across episodes the first-origin artifact vector differs. The same mapping repeats at origins 25, 75 and 125 within each episode:

| Episode | Origin | slot0..slot9 vehicle IDs |
|---|---:|---|
| episode_20001 | 25 | 46, 10, 0, 23, 59, 42, 30, 31, 7, 50 |
| episode_20001 | 75 | 46, 10, 0, 23, 59, 42, 30, 31, 7, 50 |
| episode_20001 | 125 | 46, 10, 0, 23, 59, 42, 30, 31, 7, 50 |
| episode_20002 | 25 | 55, 30, 11, 21, 47, 7, 12, 35, 15, 51 |
| episode_20002 | 75 | 55, 30, 11, 21, 47, 7, 12, 35, 15, 51 |
| episode_20002 | 125 | 55, 30, 11, 21, 47, 7, 12, 35, 15, 51 |
| episode_20003 | 25 | 52, 57, 42, 20, 23, 54, 13, 24, 5, 15 |
| episode_20003 | 75 | 52, 57, 42, 20, 23, 54, 13, 24, 5, 15 |
| episode_20003 | 125 | 52, 57, 42, 20, 23, 54, 13, 24, 5, 15 |
| episode_20004 | 25 | 40, 56, 2, 57, 35, 55, 34, 28, 9, 18 |
| episode_20004 | 75 | 40, 56, 2, 57, 35, 55, 34, 28, 9, 18 |
| episode_20004 | 125 | 40, 56, 2, 57, 35, 55, 34, 28, 9, 18 |
| episode_20005 | 25 | 12, 25, 2, 21, 18, 31, 0, 52, 14, 40 |
| episode_20005 | 75 | 12, 25, 2, 21, 18, 31, 0, 52, 14, 40 |
| episode_20005 | 125 | 12, 25, 2, 21, 18, 31, 0, 52, 14, 40 |

- SLOT_IDENTITY_CHANGE_RATE_ACROSS_EPISODES = **0.980000** (735/750 pairwise comparisons).
- Within-episode slot identity stable: **YES**; cross-episode stable: **NO**; horizon identity alignment stable: **YES**.
- DYNAMIC_SLOT_REASSIGNMENT_POSSIBLE = **YES between realization initializations; NO within an episode**. The fixed slot assignment is initialized separately from each realization's first-origin artifact; a missing ID within a valid horizon raises an error instead of replacing a slot.

## Frozen policy probabilities (300 trace states)

Cache probabilities below are after legal masking and formal KEEP/UPDATE group-balance. Duration means are the exact stochastic-policy marginal over Cache candidates; the selected duration probability is also retained conditional on deterministic Cache.

| Metric | Value |
|---|---:|
| P_KEEP mean / median / P10 / P90 | 0.502859 / 0.500571 / 0.500268 / 0.506823 |
| P_KEEP >=0.90 / >=0.75 / 0.45-0.55 fractions | 0.0000 / 0.0000 / 1.0000 |
| Mean P_UPDATE_TOTAL / max update candidate probability | 0.497141 / 0.001641 |
| Mean P_L1 / P_L3 / P_L5 (marginal over Cache) | 0.327110 / 0.333292 / 0.339598 |
| Deterministic duration counts | {'5': 300} |
| UAV0 P_STAY mean; UAV1 marginal P_STAY mean | 0.540683; 0.508341 |
| Deterministic group/action fractions | {'Cache_KEEP': 0.9933333333333333, 'Duration_5': 1.0, 'UAV0_STAY': 1.0, 'UAV1_STAY_given_deterministic_UAV0': 1.0} |
| Probability >=0.90 state fraction by action family | {'Cache_KEEP_vs_UPDATE': 0.0, 'Duration_top_action_marginal_over_Cache': 0.0, 'UAV0_STAY_vs_MOVE': 0.0, 'UAV1_STAY_vs_MOVE_marginal_over_UAV0': 0.0} |

UAV1 probability is marginalized exactly over all legal UAV0 candidates using each candidate's formal autoregressive position condition. Deterministic UAV1 values conditional on the deterministic UAV0 choice are also in JSON.

## Time-to-go and calibration

- Pearson(time-to-go, realized return-to-go) = 0.835817.
- Spearman(time-to-go, realized return-to-go) = 0.862858.
- Global formal GAE target mean/std = 42.479450 / 99.759603.
- Global realized return-to-go mean/std = 716.913160 / 521.572289.

### Per episode

| Episode | GAE target mean / std / min / max | Return mean / std / min / max | Linear return~time-to-go R² / slope / intercept |
|---|---:|---:|---:|
| episode_20001 | 105.798 / 150.657 / -399.434 / 264.167 | 1079.159 / 639.762 / 30.391 / 2007.821 | 0.9899 / 2197.928 / -34.507 |
| episode_20002 | 74.976 / 88.662 / -100.266 / 204.227 | 611.975 / 422.911 / 38.090 / 1324.760 | 0.9786 / 1444.588 / -119.981 |
| episode_20003 | 39.034 / 94.423 / -87.015 / 201.175 | 706.483 / 398.582 / 31.384 / 1411.254 | 0.9980 / 1374.930 / 9.821 |
| episode_20004 | -9.913 / 15.257 / -38.010 / 24.617 | 318.779 / 269.337 / -52.944 / 712.597 | 0.9584 / 910.467 / -142.545 |
| episode_20005 | 2.502 / 25.155 / -40.426 / 56.808 | 868.170 / 465.722 / 41.628 / 1627.382 | 0.9882 / 1598.653 / 58.150 |

### Time-to-go bins

| Bin | N | Return mean | Return std |
|---|---:|---:|---:|
| [0.0, 0.2) | 60 | 125.328 | 113.480 |
| [0.2, 0.4) | 60 | 373.568 | 214.625 |
| [0.4, 0.6) | 60 | 730.225 | 288.537 |
| [0.6, 0.8) | 60 | 1054.473 | 349.235 |
| [0.8, 1.0] | 60 | 1300.972 | 415.836 |

### Cross-episode return at aligned origins

| Origin | Mean across episodes | Std across episodes | Range | Conditional-noise std ref | Between/conditional ratio |
|---:|---:|---:|---:|---:|---:|
| 25 | 1312.700 | 411.950 | 1284.444 | 70.483 | 5.845 |
| 75 | 1109.489 | 365.807 | 1107.675 | 114.246 | 3.202 |
| 125 | 863.208 | 288.632 | 828.926 | 75.604 | 3.818 |
| 175 | 564.293 | 237.925 | 650.559 | 96.849 | 2.457 |
| 225 | 280.863 | 169.626 | 460.593 | 96.849 | 1.751 |

## Code paths and execution integrity

The JSON lists absolute source paths, function names and locations used for the audit. Training samples with `deterministic=False`; evaluate_pool selects deterministic/stochastic behavior from its mode flag. The deterministic candidate group selection uses total group probability and then raw-score candidate selection, while PPO probabilities are the full post-mask, post-balance categoricals.

- 300 trace states forwarded; all 300 Cache action/candidate ID/duration reconstructions match the saved trace.
- Maximum observation reconstruction error = 0.
- Checkpoint/policy/config hashes unchanged; optimizer steps = 0.
- Matching calls = 0; KKT calls = 0; Exact-Q environment executions = 0. Environment execution was guarded and never called.
- No formal observation, network, reward, GAE, PPO, or config file changes.

## Next decision

设计 identity-stable / permutation-invariant vehicle encoder 的离线方案；本轮不修改 observation 或网络。
