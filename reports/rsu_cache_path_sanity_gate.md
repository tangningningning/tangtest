# RSU cache-path sanity gate

本报告是只读 sanity gate：未训练、未修改系统模型/PPO，且未读取 final test seeds。

## 结论

所有 RSU 索引/容量/候选生成/动作掩码检查通过；在抽查时隙中，Actor 被确认在 32 个存在合法更新候选的状态选择 KEEP，被 mask 强制 KEEP 的次数为 0。

因此本阶段判定：RSU 近空缓存是冻结 Actor 的 learned policy behavior，不是结构性 cache-path bug；V10-D formal training 在本 gate 下安全。

## 范围与映射

Checkpoint: C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v10c_decoupled_encoder_5e4_formal\latest_checkpoint.pt
Frozen pool: C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v6_popart_actor1e4_lazy_episode_pool_20260911
Episodes: episode_20041, episode_20101, episode_20161, episode_20221
Anchor slots: (60, 120, 180, 240)；每个 episode 共 300 个 slot
RSU nodes: RSU_0(index=5), RSU_1(index=6)

| node index | node id | role | capacity (MB) | initial services | generator index | actor head index | env row | solver index |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 47 | SV | 16744.750325 | 2 | 0 | 0 | 0 | 0 |
| 1 | 22 | SV | 16744.750325 | 1 | 1 | 1 | 1 | 1 |
| 2 | 17 | SV | 16744.750325 | 2 | 2 | 2 | 2 | 2 |
| 3 | 15 | SV | 16744.750325 | 1 | 3 | 3 | 3 | 3 |
| 4 | 25 | SV | 16744.750325 | 2 | 4 | 4 | 4 | 4 |
| 5 | RSU_0 | RSU | 44652.667533 | 4 | 5 | 5 | 5 | 5 |
| 6 | RSU_1 | RSU | 44652.667533 | 5 | 6 | 6 | 6 | 6 |
| 7 | UAV_0 | UAV | 33489.500649 | 3 | 7 | 7 | 7 | 7 |
| 8 | UAV_1 | UAV | 33489.500649 | 4 | 8 | 8 | 8 | 8 |

RSU_INDEX_MAPPING_VALID = YES
RSU_CAPACITY_MAPPING_VALID = YES
Service catalog units: {'service_size': 'GB', 'program_size': 'MB', 'input_size_range': 'KB', 'mean_input_size': 'bit', 'cycles_per_bit': 'cycle/bit', 'tasks.input_size': 'bit', 'tasks.cpu_cycles': 'cycle'}；program_size 与 service_size_gb × 1000 一致。

## 候选、mask 与 Actor

在 32 个 RSU anchor states（4 slots × 4 episodes × 2 RSUs）逐节点重建了生产 sequential cache state。

RSU_CANDIDATE_GENERATION_VALID = YES
RSU_ACTION_MASK_VALID = YES
RSU_KEEP_FORCED_BY_MASK_COUNT = 0
RSU_KEEP_SELECTED_WITH_UPDATE_FEASIBLE_COUNT = 32

| role | states | KEEP rate | mean update probability | mean legal ADD | mean legal DELETE | mean legal SWAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SV | 80 | 0.312500 | 0.676942 | 2.487 | 1.438 | 5.287 |
| RSU | 32 | 1.000000 | 0.068447 | 10.000 | 0.000 | 0.000 |
| UAV | 32 | 0.500000 | 0.440699 | 0.500 | 3.500 | 10.500 |

## RSU trajectory

| node | initial mean count | mean cached services | nonempty fraction | policy KEEP rate | requested update rate | actual change rate | KEEP | ADD | DELETE | SWAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RSU_0 | 4.000000 | 0.020000 | 0.010000 | 0.986667 | 0.013333 | 0.013333 | 1184 | 0 | 16 | 0 |
| RSU_1 | 5.000000 | 0.033333 | 0.013333 | 0.983333 | 0.016667 | 0.016667 | 1180 | 0 | 20 | 0 |

## Targeted RSU counterfactuals

固定选择前 8 个满足原策略 KEEP 且存在合法 update candidate 的 RSU states；未按反事实结果筛选。每个 state 只将 one-step best 与 policy-scorer highest 两个去重候选推进 H=30。

| episode | slot | target | selected source | operation | one-step G | ΔG30 | ΔQ30 |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| episode_20041 | 60 | RSU_0 | one_step_best | ADD | 10.420206 | 0.546982 | 0.606153 |
| episode_20041 | 60 | RSU_0 | policy_scorer_highest | ADD | 9.783234 | -0.089991 | 0.000000 |
| episode_20041 | 60 | RSU_1 | one_step_best | ADD | 10.432737 | 0.559512 | 0.622314 |
| episode_20041 | 60 | RSU_1 | policy_scorer_highest | ADD | 9.800793 | -0.072431 | 0.017559 |
| episode_20041 | 120 | RSU_0 | one_step_best | ADD | 11.631696 | 0.215081 | 0.380523 |
| episode_20041 | 120 | RSU_0 | policy_scorer_highest | ADD | 11.399280 | 0.173718 | 0.263707 |
| episode_20041 | 120 | RSU_1 | one_step_best | ADD | 11.975674 | 0.750112 | 0.812912 |
| episode_20041 | 120 | RSU_1 | policy_scorer_highest | ADD | 11.135572 | 0.756742 | 0.826731 |
| episode_20041 | 180 | RSU_0 | one_step_best | ADD | 12.024625 | 0.313248 | 0.376049 |
| episode_20041 | 180 | RSU_0 | policy_scorer_highest | ADD | 11.621386 | -0.089991 | 0.000000 |
| episode_20041 | 180 | RSU_1 | one_step_best/policy_scorer_highest | ADD | 11.969637 | 0.735986 | 0.805975 |
| episode_20041 | 240 | RSU_0 | one_step_best | ADD | 14.721719 | 0.266524 | 0.399929 |
| episode_20041 | 240 | RSU_0 | policy_scorer_highest | ADD | 14.367616 | -0.089990 | 0.000000 |
| episode_20041 | 240 | RSU_1 | one_step_best | ADD | 14.906128 | 0.965431 | 0.967348 |
| episode_20041 | 240 | RSU_1 | policy_scorer_highest | ADD | 14.367616 | 0.426919 | 0.443180 |

RSU_COUNTERFACTUAL_STATES = 8
RSU_BENEFICIAL_UPDATE_WITNESS_COUNT = 11 (selected candidate branches with DELTA_G30 > 0; targeted evidence only)

## Required final fields

STAGE = RSU_CACHE_PATH_SANITY_GATE
SYSTEM_MODEL_CHANGED = NO
PPO_CHANGED = NO
TRAINING_EXECUTED = NO
RSU_INDEX_MAPPING_VALID = YES
RSU_CAPACITY_MAPPING_VALID = YES
RSU_CANDIDATE_GENERATION_VALID = YES
RSU_ACTION_MASK_VALID = YES
RSU_KEEP_FORCED_BY_MASK_COUNT = 0
RSU_KEEP_SELECTED_WITH_UPDATE_FEASIBLE_COUNT = 32
RSU_INITIAL_MEAN_CACHED_SERVICES = 4.5
RSU_MEAN_CACHED_SERVICES = 0.02666666666666667
RSU_NONEMPTY_SLOT_FRACTION = 0.011666666666666667
RSU_POLICY_KEEP_RATE = 0.985
RSU_ACTUAL_CHANGE_RATE = 0.015
RSU_COUNTERFACTUAL_STATES = 8
RSU_BENEFICIAL_UPDATE_WITNESS_COUNT = 11
RSU_STRUCTURAL_CACHE_PATH_BUG = NO
RSU_NEAR_EMPTY_CACHE_IS_LEARNED_POLICY = YES
V10D_FORMAL_TRAINING_SAFE = YES
FINAL_TEST_SEEDS_USED = NO
REPORT = reports/rsu_cache_path_sanity_gate.md
GIT_DIFF_CHECK = PASS
TEMP_FILES_REMOVED = YES
PUSHED = NO
