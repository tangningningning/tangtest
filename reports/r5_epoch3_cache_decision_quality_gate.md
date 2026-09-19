# R5 epoch3 final Cache decision-quality gate

日期：2026-09-19  
仓库：C:/Users/tangning/Desktop/code/adaptive_cache_mec  
任务：R5_CONTINUE_EPOCH3_AND_FINAL_CACHE_PROBE

## 结论

从完整 epoch2 checkpoint continuation 只训练了 episode_20003，并在 epoch3 完成后立即停止；没有运行 epoch4。随后使用 epoch3 checkpoint，在 frozen episode_20001 上执行了与 epoch1/epoch2 完全相同的 8-state exact Cache decision-quality probe。

epoch3 相对 epoch2：

- MEAN_P_KEEP 从 0.3361200206 降至 0.3135103323；
- MEAN_P_UPDATE 从 0.6638799720 升至 0.6864896715；
- KEEP_PREFERRED_AND_POLICY_KEEP_COUNT 保持 1，没有继续增加；
- POLICY_TRIGGER_CORRECT_COUNT 从 2/8 增加到 4/8，但增加主要来自 UPDATE-preferred states；
- exact preference 为 KEEP 5/8、UPDATE 3/8。

因此，按任务定义，整体概率方向反向、KEEP alignment 没有改善：

~~~
R5_CACHE_DEV3_RESULT = NOT_SUPPORTED
PRIMARY_DIAGNOSIS = EPOCH3_PROBABILITY_DIRECTION_REVERSED_AND_KEEP_ALIGNMENT_PLATEAUED
~~~

这不属于 policy collapse：当前 policy 没有在 8 个 witness 上统一坍缩为 KEEP 或 UPDATE；但它没有继续朝 KEEP-preferred exact preference 改善。

## Continuation 与 epoch3 checkpoint

epoch2 起点核验：

~~~
completed_episodes = 2
completed_training_episodes = 2
next_pool_index = 2
current_train_origin = 300
~~~

同一 run continuation 使用：

~~~
run = runs/ppo_recovery_r5_wmove_0p50_nodecache_dev3_20260919
training realization = runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911/episode_20003
~~~

epoch3 完成后的 checkpoint metadata：

~~~
EPOCH3_COMPLETED = YES
completed_episodes = 3
completed_training_episodes = 3
next_pool_index = 3
current_train_origin = 300
run_summary.completed_epochs = 3
run_summary.status = completed
nan_inf_detected = false
best_epoch = 3
~~~

epoch3 训练统计：

~~~
episode_return = 382.4419194025383
mean_reward = 1.274806398008461
actor_optimizer_steps = 70
critic_optimizer_steps = 120
optimizer_steps_before_epoch3 = (30, 80, 30)
optimizer_steps_after_epoch3  = (70, 120, 70)
cache_credit_runtime_seconds = 1609.5936215
cache_credit_counterfactual_env_executions = 34663
wall_clock_runtime_seconds = 3957.6913961
kl_early_stop_triggered = false
ppo_update_epochs_executed = 4
finite_checks = PASS
~~~

epoch3 使用 pool index 2，即 episode_20003。训练自然完成后进程退出；没有启动 epoch4。

## Runtime semantics

epoch3 training 和最终 probe 均保留当前 exact-safe branch reuse：

~~~
EXACT_SEMANTICS_PRESERVED = YES
candidate_pruning = NO
top_k = NO
sample_approximation = NO
reward_approximation = NO
Matching_KKT_approximation = NO
~~~

最终 8-state probe helper 统计：

~~~
logical_branch_count = 534
counterfactual_environment_executions = 494
dedup_reused_branch_count = 40
reuse_identical_branches = true
~~~

branch identity 仍由完整 physical post-configuration state（完整 Cache state 和 UAV state）决定。没有改变 logical branch 集合、正式 environment、reward、Matching/KKT 或 next-state semantics。

## Frozen probe definition

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
| 20 | RSU_0 | 0.223070 | 0.776930 | -187.202347 | -189.812851 | -2.610504 | KEEP | NO | 26 | 0.373195 |
| 20 | RSU_1 | 0.311938 | 0.688062 | -184.775833 | -190.104904 | -5.329071 | KEEP | NO | 26 | 0.059579 |
| 80 | RSU_0 | 0.216452 | 0.783548 | -192.518448 | -192.926315 | -0.407867 | KEEP | NO | 31 | 0.503166 |
| 80 | RSU_1 | 0.294446 | 0.705554 | -193.076202 | -192.298401 | +0.777802 | UPDATE | YES | 19 | 0.852247 |
| 140 | RSU_0 | 0.219208 | 0.780792 | -116.442368 | -114.813881 | +1.628487 | UPDATE | YES | 19 | 0.813244 |
| 140 | RSU_1 | 0.325031 | 0.674969 | -118.762001 | -119.507706 | -0.745705 | KEEP | NO | 19 | 0.788841 |
| 200 | RSU_0 | 0.590138 | 0.409862 | -77.154068 | -79.091667 | -1.937599 | KEEP | YES | 10 | 0.271978 |
| 200 | RSU_1 | 0.327799 | 0.672201 | -77.154068 | -76.847954 | +0.306114 | UPDATE | YES | 19 | 0.773929 |

## Required output

~~~
EPOCH3_COMPLETED = YES
STATE_COUNT = 8

KEEP_BETTER_STATE_COUNT = 5
UPDATE_BETTER_STATE_COUNT = 3

POLICY_TRIGGER_CORRECT_COUNT = 4/8

KEEP_PREFERRED_AND_POLICY_KEEP_COUNT = 1
UPDATE_PREFERRED_AND_POLICY_UPDATE_COUNT = 3

MEAN_P_KEEP = 0.3135103323
MEAN_P_UPDATE = 0.6864896715
MEAN_Q_UPDATE_MINUS_Q_KEEP = -1.0397930145
MEAN_GOOD_UPDATE_CANDIDATE_MASS = 0.5545225475
~~~

good candidate 定义为 Q_c > Q_keep。本次 8 个 witness 中 good candidate 共 69 个，harmful candidate 共 100 个。

## Compared with epoch2

epoch2 reference：

~~~
POLICY_TRIGGER_CORRECT_COUNT = 2/8
KEEP_PREFERRED_AND_POLICY_KEEP_COUNT = 1
MEAN_P_KEEP = 0.3361200206
MEAN_P_UPDATE = 0.6638799720
~~~

epoch3 changes：

~~~
DELTA_MEAN_P_KEEP_FROM_EPOCH2 = -0.0226096883
DELTA_MEAN_P_UPDATE_FROM_EPOCH2 = +0.0226096995
TRIGGER_CORRECT_COUNT_CHANGE_FROM_EPOCH2 = +2
KEEP_ALIGNMENT_CHANGE_FROM_EPOCH2 = 0
~~~

epoch3 的 correctness count 虽然增加到 4/8，但 KEEP alignment 没有增加，且 mean p_keep 下降。新增的两个正确 trigger 来自 UPDATE-preferred exact states，不能证明 policy 继续朝 epoch2 暴露的 KEEP-preferred bottleneck 改善。因此最终分类为 NOT_SUPPORTED，而不是 SUPPORTED_PARTIAL_ALIGNMENT。

## Change-control audit

~~~
TRAINING_EXECUTED = YES   # exactly one requested epoch: episode_20003
OPTIMIZER_STEP_EXECUTED = YES  # epoch3 continuation update only

EPOCH4_EXECUTED = NO
H10_H30_EXECUTED = NO
LR_SWEEP = NO
KL_SWEEP = NO
FORMAL_EVALUATION = NO
FINAL_TEST_SEEDS_USED = NO

REWARD_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO
MATCHING_KKT_CHANGED = NO
CAPACITY_CHANGED = NO
CACHE_COST_CHANGED = NO
MOVEMENT_COST_CHANGED = NO
R5_CACHE_CREDIT_CHANGED = NO
UAV_ALGORITHM_CHANGED = NO
NEW_ALGORITHM_MODIFICATION = NO

EPISODE_REGENERATION = NO
SUMO_EXECUTED = NO

SOURCE_MODIFIED = NO
TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO
~~~
