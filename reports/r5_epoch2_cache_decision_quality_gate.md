# R5 epoch2 Cache decision-quality gate

日期：2026-09-19  
仓库：C:/Users/tangning/Desktop/code/adaptive_cache_mec  
任务：R5_CONTINUE_ONE_EPOCH_AND_PROBE

## 结论

从完整 R5 epoch1 checkpoint continuation 到 episode_20002，只完成了一个 epoch（epoch2），随后在 frozen episode_20001 上完成了与 epoch1 相同的 8-state exact Cache decision-quality probe。

epoch2 后：

- MEAN_P_KEEP 从 epoch1 的 0.3128530048 上升到 0.3361200206；
- KEEP_PREFERRED_AND_POLICY_KEEP_COUNT 从 0 增加到 1；
- UPDATE_PREFERRED_AND_POLICY_UPDATE_COUNT = 1；
- 总 POLICY_TRIGGER_CORRECT_COUNT 保持 2/8，没有增加；
- 当前 exact preference 为 KEEP 7/8、UPDATE 1/8，但多数 KEEP-preferred state 仍然偏向 UPDATE。

按任务定义，第一项概率继续上升且至少出现一个 KEEP-preferred / policy-KEEP 对齐状态，因此：

~~~
R5_EPOCH2_LEARNING_DIRECTION = SUPPORTED
PRIMARY_DIAGNOSIS = R5_NODE_SPECIFIC_CACHE_CREDIT_DIRECTION_SUPPORTED_BUT_STILL_PARTIAL
~~~

这是方向性支持，不表示 Cache trigger policy 已经完全对齐，也不表示应继续 epoch3。

## Continuation 与 epoch2 checkpoint

epoch1 起点核验：

~~~
completed_episodes = 1
completed_training_episodes = 1
next_pool_index = 1
current_train_origin = 300
~~~

使用同一个 run、同一个 R5 config 和同一个 episode pool continuation：

~~~
run = runs/ppo_recovery_r5_wmove_0p50_nodecache_dev3_20260919
training realization = runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911/episode_20002
~~~

epoch2 完成后的 checkpoint metadata：

~~~
EPOCH2_COMPLETED = YES
completed_episodes = 2
completed_training_episodes = 2
next_pool_index = 2
current_train_origin = 300
run_summary.completed_epochs = 2
run_summary.status = completed
nan_inf_detected = false
~~~

因此确实恢复并继续使用了 actor、critic、optimizers、PopArt 状态和 training cursor/RNG。checkpoint policy schema 和 config hash 未改变。epoch2 return 为 -2810.940485871397；epoch2 使用 pool index 1，即 episode_20002。

epoch2 训练统计：

~~~
actor_optimizer_steps = 30
critic_optimizer_steps = 80
optimizer_steps_before_epoch2 = (10, 40, 10)
optimizer_steps_after_epoch2  = (30, 80, 30)
cache_credit_runtime_seconds = 1356.8843581
cache_credit_counterfactual_env_executions = 36943
wall_clock_runtime_seconds = 3092.3807208
kl_early_stop_triggered = true
ppo_update_epochs_executed = 4
finite_checks = PASS
~~~

这里的 kl_early_stop_triggered 是允许的 branch-specific normal KL 行为；它没有中止 whole run，epoch2 仍完整完成。没有执行 epoch3。

## Runtime semantics

epoch2 training 和 probe 都保留当前 exact-safe branch reuse：

~~~
EXACT_SEMANTICS_PRESERVED = YES
candidate_pruning = NO
top_k = NO
sample_approximation = NO
reward_approximation = NO
Matching_KKT_approximation = NO
~~~

probe 使用的 helper 统计为：

~~~
logical_branch_count = 643
counterfactual_environment_executions = 603
dedup_reused_branch_count = 40
reuse_identical_branches = true
~~~

branch identity 仍由完整 physical post-configuration state（完整 Cache state 和 UAV state）决定；没有改变 logical branch 集合，也没有改变正式 environment、reward、Matching/KKT 或 next-state semantics。

## Frozen probe definition

使用：

~~~
realization = runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911/episode_20001
witness slots = 20, 80, 140, 200
targets = RSU_0, RSU_1
STATE_COUNT = 8
~~~

origin 201 仅作为 slot 200 的 hidden bootstrap-support transition，不计入 8 个 witness。每个 branch 经过正式 execute_slot、Matching/KKT、reward、next state 和 critic bootstrap：

~~~
Q_keep = r_keep + gamma * V(next_keep)
Q_c    = r_c    + gamma * V(next_c)
Q_update = sum_c pi(c | UPDATE) * Q_c
DeltaQ = Q_update - Q_keep
~~~

## Eight-state result

| slot | target | p_keep | p_update | Q_keep | Q_update | DeltaQ | exact preferred | trigger correct | legal UPDATE candidates | good candidate mass |
|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| 20 | RSU_0 | 0.302270 | 0.697730 | -228.255951 | -229.260193 | -1.004242 | KEEP | NO | 19 | 0.178534 |
| 20 | RSU_1 | 0.273609 | 0.726391 | -227.978699 | -228.840683 | -0.861984 | KEEP | NO | 19 | 0.148738 |
| 80 | RSU_0 | 0.289005 | 0.710995 | -225.139694 | -226.879745 | -1.740051 | KEEP | NO | 19 | 0.573966 |
| 80 | RSU_1 | 0.281810 | 0.718190 | -237.125122 | -229.057022 | +8.068100 | UPDATE | YES | 19 | 0.854355 |
| 140 | RSU_0 | 0.285228 | 0.714772 | -183.982437 | -186.292419 | -2.309982 | KEEP | NO | 20 | 0.304371 |
| 140 | RSU_1 | 0.669662 | 0.330338 | -184.709396 | -190.679901 | -5.970505 | KEEP | YES | 10 | 0.327011 |
| 200 | RSU_0 | 0.292600 | 0.707400 | -149.281250 | -150.818741 | -1.537491 | KEEP | NO | 31 | 0.301213 |
| 200 | RSU_1 | 0.294777 | 0.705224 | -149.199509 | -153.277771 | -4.078262 | KEEP | NO | 19 | 0.058861 |

## Required output

~~~
EPOCH2_COMPLETED = YES
STATE_COUNT = 8

KEEP_BETTER_STATE_COUNT = 7
UPDATE_BETTER_STATE_COUNT = 1

POLICY_TRIGGER_CORRECT_COUNT = 2/8

MEAN_P_KEEP = 0.3361200206
MEAN_P_UPDATE = 0.6638799720
MEAN_Q_UPDATE_MINUS_Q_KEEP = -1.1793022156
MEAN_GOOD_UPDATE_CANDIDATE_MASS = 0.3433809262

KEEP_PREFERRED_AND_POLICY_KEEP_COUNT = 1
UPDATE_PREFERRED_AND_POLICY_UPDATE_COUNT = 1
~~~

good candidate 定义为 Q_c > Q_keep。本次 8 个 witness 中 good candidate 共 52 个，harmful candidate 共 104 个。

## Compared with epoch1

epoch1 reference：

~~~
POLICY_TRIGGER_CORRECT_COUNT = 2/8
MEAN_P_KEEP = 0.3128530048
MEAN_P_UPDATE = 0.6871469989
KEEP_PREFERRED_AND_POLICY_KEEP_COUNT = 0
~~~

epoch2 changes：

~~~
DELTA_MEAN_P_KEEP_FROM_EPOCH1 = +0.0232670158
DELTA_MEAN_P_UPDATE_FROM_EPOCH1 = -0.0232670269
TRIGGER_CORRECT_COUNT_CHANGE_FROM_EPOCH1 = 0
~~~

epoch1 的 6 个 KEEP-preferred state 全部为 p_update > p_keep；epoch2 后 exact KEEP preference 增加到 7/8，并出现 1 个真实 KEEP 对齐 state（slot 140 / RSU_1）。但 6 个 KEEP-preferred state 仍偏向 UPDATE，说明改善仍不完整。

## Change-control audit

~~~
TRAINING_EXECUTED = YES   # exactly one requested epoch: episode_20002
OPTIMIZER_STEP_EXECUTED = YES  # epoch2 continuation update only

EPOCH3_EXECUTED = NO
H10_H30_EXECUTED = NO
LR_SWEEP = NO
KL_SWEEP = NO
FORMAL_EVALUATION = NO

REWARD_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO
MATCHING_KKT_CHANGED = NO
CAPACITY_CHANGED = NO
CACHE_COST_CHANGED = NO
MOVEMENT_COST_CHANGED = NO
R5_CACHE_CREDIT_CHANGED = NO
R4_UAV_CREDIT_CHANGED = NO
NEW_ALGORITHM_MODIFICATION = NO

EPISODE_REGENERATION = NO
SUMO_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO

SOURCE_MODIFIED = NO
TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO
~~~
