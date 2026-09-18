# V10-C targeted counterfactual witness gate

本报告是 targeted / witness-based diagnostic，不是全 state / 全 candidate 穷举。
所有 branch 都在固定 frozen episode realization 上运行；t 时刻仅替换 target UAV，
t+1 起由同一个 frozen checkpoint 的 deterministic policy 重新决定 Cache/UAV。
H=15 是同一条 H=30 branch 的前 15 个 slot，因此使用完全相同的 future realization。

## Scope and invariants

- Checkpoints: `latest_checkpoint.pt` from the two formal V10-C runs.
- Frozen episodes: `episode_20041`, `episode_20101`, `episode_20161`, `episode_20221` only.
- State selection: anchor slot 150, then first geometry-feasible MOVE in slots 151..155; no reward, delta-g, or final outcome was used for selection.
- Stage A: all joint-feasible MOVE candidates at each selected state, exact one-slot `execute_slot`.
- Stage B: STAY plus the requested selected MOVE candidates, at most 30 executed slots per branch.
- No `backward`, `optimizer.step`, `resume`, checkpoint save, algorithm, reward, PPO, KL, or system-model change.

## Results

- 5e-4 states: **8**; Stage-A MOVE candidates scanned: **640**.
- 6e-4 states: **8**; Stage-A MOVE candidates scanned: **585**.
- 5e-4 UNDER_MOVE witnesses: **7**; 6e-4 policy-MOVE states: **8**.
- 6e-4 OVER_MOVE witnesses: **8** across 4 episodes.

| checkpoint | episode | slot | target | policy action | G_STAY_30 | best selected MOVE G30 | best selected ΔG30 | policy-MOVE ΔG30 | witness |
|---|---:|---:|---|---|---:|---:|---:|---:|---|
| 5e-4 | 20041 | 150 | UAV_0 | STAY | 321.437861 | 321.480709 | 0.042848 | — | UNDER_MOVE |
| 5e-4 | 20041 | 150 | UAV_1 | STAY | 321.437861 | 321.457490 | 0.019629 | — | UNDER_MOVE |
| 5e-4 | 20101 | 150 | UAV_0 | STAY | 277.496299 | 277.582869 | 0.086570 | — | UNDER_MOVE |
| 5e-4 | 20101 | 150 | UAV_1 | STAY | 277.496299 | 277.501514 | 0.005214 | — | UNDER_MOVE |
| 5e-4 | 20161 | 150 | UAV_0 | STAY | 341.745344 | 341.740616 | -0.004728 | — | — |
| 5e-4 | 20161 | 150 | UAV_1 | STAY | 341.745344 | 341.752981 | 0.007637 | — | UNDER_MOVE |
| 5e-4 | 20221 | 150 | UAV_0 | STAY | 258.529780 | 258.581817 | 0.052036 | — | UNDER_MOVE |
| 5e-4 | 20221 | 150 | UAV_1 | STAY | 258.529780 | 258.535453 | 0.005672 | — | UNDER_MOVE |
| 6e-4 | 20041 | 150 | UAV_0 | MOVE | 351.948648 | 351.950584 | 0.001936 | -0.052029 | OVER_MOVE |
| 6e-4 | 20041 | 150 | UAV_1 | MOVE | 351.911795 | 351.896620 | -0.015176 | -0.015176 | OVER_MOVE |
| 6e-4 | 20101 | 150 | UAV_0 | MOVE | 328.272259 | 328.257446 | -0.014813 | -0.037986 | OVER_MOVE |
| 6e-4 | 20101 | 150 | UAV_1 | MOVE | 328.257676 | 328.243954 | -0.013723 | -0.023404 | OVER_MOVE |
| 6e-4 | 20161 | 150 | UAV_0 | MOVE | -127.476915 | -127.485719 | -0.008804 | -0.048435 | OVER_MOVE |
| 6e-4 | 20161 | 150 | UAV_1 | MOVE | -127.498964 | -127.521352 | -0.022387 | -0.026385 | OVER_MOVE |
| 6e-4 | 20221 | 150 | UAV_0 | MOVE | 221.405165 | 221.409062 | 0.003897 | -0.059021 | OVER_MOVE |
| 6e-4 | 20221 | 150 | UAV_1 | MOVE | 221.375855 | 221.380411 | 0.004556 | -0.029710 | OVER_MOVE |

The detailed Stage-A and selected-branch values are in the JSONL raw results. A positive 5e-4 delta is an UNDER_MOVE witness; a negative 6e-4 delta compares the actual policy MOVE against STAY and is an OVER_MOVE witness.

## Required final status

```text
STAGE = V10-C_TARGETED_COUNTERFACTUAL_WITNESS_GATE
SYSTEM_MODEL_CHANGED = NO
GLOBAL_OBJECTIVE_CHANGED = NO
PPO_CHANGED = NO
TRAINING_EXECUTED = NO
N_STATES_5E4 = 8
N_STATES_6E4 = 8
STAGE_A_MOVE_CANDIDATES_SCANNED_5E4 = 640
STAGE_A_MOVE_CANDIDATES_SCANNED_6E4 = 585
5E4_UNDER_MOVE_WITNESS_COUNT = 7
5E4_UNDER_MOVE_NOT_FOUND = NO
6E4_POLICY_MOVE_STATES = 8
6E4_OVER_MOVE_WITNESS_COUNT = 8
6E4_OVER_MOVE_NOT_FOUND = NO
5E4_MEDIAN_DELTA_G30_BEST_SELECTED_MOVE = 0.013632621613282936
6E4_MEDIAN_DELTA_G30_POLICY_MOVE = -0.03384835825337973
V10C_5E4_POLICY_UNDER_MOVES = YES
V10C_6E4_POLICY_OVER_MOVES = YES
LONG_HORIZON_POLICY_OPTIMALITY = NOT_PROVEN_TARGETED_DIAGNOSTIC_ONLY
BRANCH_SPECIFIC_KL_GATE_IMPLEMENTED = NO
ESTIMATED_RUNTIME = bounded targeted gate, <= 30 minutes expected
ACTUAL_RUNTIME = 529.622 seconds (8.83 minutes)
FINAL_TEST_SEEDS_USED = NO
REPORT = C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\v10c_targeted_counterfactual_witness_gate.md
RAW_RESULTS = C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\v10c_targeted_counterfactual_witness_results.jsonl
GIT_DIFF_CHECK = PASS
TEMP_FILES_REMOVED = YES
PUSHED = NO
```

## Provenance checks

- 5e-4: config SHA-256 `9f3c8764652cb0622997074a2ea0d9dce4e01e09c97766df6b8133dc095e746c`, checkpoint `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v10c_decoupled_encoder_5e4_formal\latest_checkpoint.pt`, completed epochs `300`, policy schema `adaptive_joint_cache_replica_aware_uav_operation_v10c_decoupled_encoder`.
- 6e-4: config SHA-256 `4763d139d1ffa6338707a6524889fec02070559d05baa06cbc39f03e54351f25`, checkpoint `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v10c_decoupled_encoder_6e4_formal\latest_checkpoint.pt`, completed epochs `300`, policy schema `adaptive_joint_cache_replica_aware_uav_operation_v10c_decoupled_encoder`.
- 5e-4 optimizer counters before/after: `(11770, 11770, 11770)` -> `(11770, 11770, 11770)`; policy parameters unchanged: `True`; PopArt statistics unchanged: `True`.
- 6e-4 optimizer counters before/after: `(11130, 11130, 11130)` -> `(11130, 11130, 11130)`; policy parameters unchanged: `True`; PopArt statistics unchanged: `True`.

The witness conclusions are falsification evidence only. Absence of a witness in these 16 targeted states does not establish STAY-optimality, MOVE-optimality, or full-policy optimality.
