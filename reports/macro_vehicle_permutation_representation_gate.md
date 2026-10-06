# Macro Vehicle Permutation Representation Gate

## Decision

- Current checkpoint permutation sensitivity: **HIGH** (median value std/global return std = 0.163710).
- Canonical trajectory ordering gate: **FAIL**.
- DeepSets trajectory encoder gate: **FAIL**.
- Root cause gate: **FAIL**; selected representation: **NONE**.

## Permutation stress

Applied 16 seeded permutations to each of 300 trace states, using the same vehicle order across all five horizons. All nonvehicle coordinates were held fixed. The checkpoint29 Cache candidates were rebuilt from saved artifacts and matched all 300 saved candidate counts.

- Median / P90 per-state critic value standard deviation: 85.386825 / 112.330235.
- Median / P90 normalized value standard deviation: 0.163710 / 0.215368.
- Median Cache P_KEEP max absolute change: 0.00002676.
- Median duration max probability change: 0.00002609.
- B max canonical-representation difference: 0.
- C max value difference under permutation: 0.

## Fresh critic comparison

All critics were retrained in memory with the same data and four unseen-realization folds. Primary target is realized return-to-go; formal GAE target is secondary. Metrics below are median TEST_EV over the four unseen folds.

| Candidate | Return EV step 50 | step 100 | step 200 | GAE EV step 200 | Matched-fit return EV | Median absolute origin bias |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.0278 | 0.1545 | 0.3938 | -3.8298 | 0.43606623079975326 | 259.8922 |
| B | -0.0195 | -0.0963 | -0.0039 | -2.4190 | NOT_REACHED | 196.5145 |
| C | 0.0022 | -0.0035 | 0.0956 | -1.4044 | NOT_REACHED | 220.0840 |

Step-by-fold TRAIN_EV, TEST_EV, correlation, prediction/target std ratio, RMSE, and first TRAIN_EV≥0.90 snapshots are in the JSON report.

## Origin calibration

Bias is the signed mean prediction minus target across the four out-of-fold test episodes; gate uses the median absolute bias across the five aligned origins.

| Candidate | Origin | Target mean | Prediction mean | Bias | Prediction std |
|---|---:|---:|---:|---:|---:|
| A | 25 | 1154.174 | 969.256 | -184.918 | 309.872 |
| A | 75 | 966.689 | 1089.684 | 122.996 | 336.948 |
| A | 125 | 760.417 | 1020.309 | 259.892 | 246.404 |
| A | 175 | 485.715 | 1025.890 | 540.176 | 182.911 |
| A | 225 | 235.566 | 877.549 | 641.983 | 205.313 |
| B | 25 | 1154.174 | 957.660 | -196.514 | 235.912 |
| B | 75 | 966.689 | 961.872 | -4.817 | 251.323 |
| B | 125 | 760.417 | 864.474 | 104.057 | 277.950 |
| B | 175 | 485.715 | 862.428 | 376.714 | 240.733 |
| B | 225 | 235.566 | 710.282 | 474.716 | 260.294 |
| C | 25 | 1154.174 | 976.408 | -177.767 | 273.951 |
| C | 75 | 966.689 | 903.493 | -63.196 | 223.163 |
| C | 125 | 760.417 | 980.501 | 220.084 | 240.707 |
| C | 175 | 485.715 | 967.321 | 481.606 | 199.455 |
| C | 225 | 235.566 | 950.376 | 714.811 | 151.907 |

## Protocol and integrity

- A: current raw 400D flatten; B: lexicographic trajectory canonicalization with the vehicle ID excluded; C: shared 10→32→32 SiLU encoder, mean/max pooling, 214D value input.
- Offline optimizer: Adam lr=1e-3, eps=1e-5; 0.01×MSE; max grad norm=0.5; 200 full-batch steps; snapshots at 50/100/200; common seed 27005.
- Offline diagnostic critic steps: 4800; formal optimizer steps: 0.
- Matching/KKT/Exact-Q/environment calls: 0/0/0/0; checkpoint SHA256 unchanged: 4ceaa8ebc39d40ae0412b56699b1dc8e50bfde36908dade911da73397929887a.
- Formal observation, policy, reward, GAE, PPO, configuration, and checkpoint were not modified.
- Temporary scripts, candidate feature NPZ, __pycache__, and .pyc files were removed; no temporary model checkpoint or tensor was saved.

## Next decision

优先采用 B，若 CANONICAL_TRAJECTORY_GATE=PASS；否则仅在 C 通过时再考虑 DeepSets。若 B/C 均失败或当前 checkpoint 的排列敏感度为 LOW，则维持正式 architecture，转查 Macro value calibration / structural resource context。

Detailed provenance and all fold metrics are in the JSON report.
