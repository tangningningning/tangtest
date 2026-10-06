# Cache Trigger V3 Production Runtime Gate

## 范围与运行方式

使用 `adaptive_cache_mec`，从 `environment_data/proposed_joint_training/episode_20001` 按 episode 正式顺序执行 slot/origin 0–15，共 16 个真实 transition；之后在同一 rollout buffer 上以 frozen policy 做一次正式 joint PPO policy evaluation dry-run。数据没有离开 training pool。

当前运行时 policy 为 architecture version 3。V3 runtime gate 使用固定 RNG seed 20001 的 fresh V3 初始化参数；没有加载或写入 checkpoint。policy 处于 eval/inference mode，参数 hash 在 rollout 与 PPO evaluation 前后相同。此处验证 runtime wiring，不评估策略质量。

## 最终判定

- `ARCHITECTURE_VERSION = 3`
- `REAL_PRODUCTION_ROLLOUT_EXECUTED = YES`
- `NODE_LOCAL_WORKLOAD_RUNTIME_ACTIVE = YES`
- `CANDIDATE_SUMMARY_RUNTIME_ACTIVE = YES`
- `TRIGGER_AUX_RUNTIME_ACTIVE = YES`
- `TRIGGER_AUX_EFFECT_ON_LOGITS_OBSERVED = YES`
- `ROLLOUT_BUFFER_AUX_CONSISTENT = YES`
- `PPO_EVAL_AUX_CONSISTENT = YES`
- `CANDIDATE_PRETRIGGER_EXECUTION_CONSISTENT = YES`
- `FUTURE_INFORMATION_LEAKAGE = NO`
- `ACTUAL_TASKS_IN_TRIGGER = NO`
- `ACTUAL_REWARD_IN_TRIGGER = NO`
- `MATCHING_KKT_OUTPUT_IN_TRIGGER = NO`
- `BRANCH_Q_IN_TRIGGER = NO`
- `TRIGGER_AUX_RECOMPUTED_DURING_PPO_UPDATE = NO`
- `TRIGGER_AUX_HAS_NAN_INF = NO`
- `TRIGGER_LOGITS_HAS_NAN_INF = NO`
- `CANDIDATE_LOGITS_HAS_NAN_INF = NO`
- `NUMERICAL_HEALTH_PASS = YES`
- `CACHE_TRIGGER_V3_READY_FOR_FRESH_FORMAL_TRAINING = YES`
- `TRAINING_EXECUTED = NO`
- `BACKWARD_CALLS = 0`
- `OPTIMIZER_STEP_CALLS = 0`
- `VALIDATION_EXECUTED = NO`
- `TEST_EXECUTED = NO`
- `EXACT_Q_BRANCHES = 0`
- `CHECKPOINT_MODIFIED = NO`

## Feature 与 Trigger 证据

- 输入维度：base cache context 400-D；node-local workload 30-D/head；candidate summary 86-D/head；Trigger aux 116-D/head。
- node-local workload 非零且跨状态变化：`True` / `True`。candidate summary 在所有存在候选的状态非零且跨状态变化：`True` / `True`。
- 保持 action-time base cache embedding 不变、只把 Trigger aux 置零，最大 raw logit 变化为 `0.337949157`；action-time logits 由保存的 embedding 与 aux 重算最大误差 `0`。
- `MAX_AUX_CONCAT_DIFF = 0`；action-time 与 buffer aux 最大差 `0`。PPO eval 收到的 aux 与 buffer 最大差 `0`；PPO eval 阶段 workload/aux builder 调用数为 `0`。

### 每个 Cache head 的真实 feature 统计

norm mean 是 16 个 transition 的逐状态 L2 norm 平均；`STD across` 是每个 feature 先跨 transition 求标准差，再对 feature 取平均。每个 transition/head 的完整 base embedding、Trigger aux、raw logits、zero-aux logits、mask 与概率都保存在 JSON。

| Head | Node | Role | Workload norm mean | Workload STD across | Candidate summary norm mean | Summary STD across | Trigger aux norm mean | Aux STD across |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | 46 | SV | 0.0099205 | 0.000849483 | 3.21864 | 0.0975464 | 3.21866 | 0.0725386 |
| 1 | 10 | SV | 0.0189674 | 0.000937453 | 3.25041 | 0.103037 | 3.25047 | 0.0766317 |
| 2 | 0 | SV | 0.00300918 | 0.000522072 | 3.24796 | 0.0909583 | 3.24797 | 0.0675696 |
| 3 | 23 | SV | 0.00952316 | 0.000492812 | 3.33822 | 0.104854 | 3.33823 | 0.0778642 |
| 4 | 59 | SV | 0.00349369 | 0.000579514 | 3.24783 | 0.0925105 | 3.24783 | 0.0687352 |
| 5 | RSU_0 | RSU | 0.0101548 | 0.000488489 | 3.99824 | 0.115563 | 3.99826 | 0.0858024 |
| 6 | RSU_1 | RSU | 0.00119681 | 0.000124637 | 4.09493 | 0.106496 | 4.09493 | 0.078986 |
| 7 | UAV_0 | UAV | 0.0107615 | 0.000657917 | 4.06592 | 0.141107 | 4.06594 | 0.104784 |
| 8 | UAV_1 | UAV | 1.0717e-05 | 5.45411e-06 | 4.04059 | 0.101218 | 4.04059 | 0.0750425 |

## Rollout、PPO evaluation 与候选一致性

- Snapshot 拼接差 `0`，action-time capture 到 rollout buffer 差 `0`；stored aux 是 detached 数组，policy 参数无 gradient。
- 同一 rollout buffer 的 `evaluate_joint_cache_uav_operation` frozen forward：action-time vs PPO eval trigger logits 最大差 `8.94e-08`，概率最大差 `1.19e-07`；所选 joint cache log-prob 最大差 `4.77e-07`。
- runtime 记录了 `144` 次 ordered candidate enumeration、`144` 次 scorer 调用和 `77` 次 UPDATE candidate sampling。77 个实际采样 vector 的 count 与枚举 candidate count 相同，scorer logits hash 与采样输入 hash 一致；映射失败 `0`。candidate pre-trigger/execution logit 最大差 `0`。JSON 保存每个 transition/head 的候选 target IDs 顺序、order hash、candidate count 和采样 index。
- production primitive 的实测调用数与顺序是一致的：每个 transition/head 只枚举一次；对有候选的 head，scorer 输出用于生成该 head 的 summary/aux，随后才执行 Trigger；UPDATE 时从同一 scorer logits 向量按 index 采样。没有第二次 scorer forward 或另一个候选枚举集。

## 决策后信息隔离与数值健康

每个 transition 的最后一次 Trigger forward 都早于 `execute_slot` 入口，后者又早于 reward 校验（16/16 通过）。Trigger runtime 输入只有 observation、head index、operation mask 和 pre-decision Trigger aux；未传入 actual task object。Matching/KKT 与 reward 在 Trigger 后由 `execute_slot` 产生。

- `MAX_TRIGGER_AUX_ABS = 2.50026846`
- `MAX_TRIGGER_LOGIT_ABS = 0.329442143`
- aux/logits/candidate scorer logits 均通过 finite 检查：`True`。

## 训练状态

本次只执行 16-slot production rollout 和只读 frozen PPO forward；没有正式训练、backward、optimizer step、validation、test、exact-Q/symmetric-Q branches，也没有修改参数或 checkpoint。

完整逐 head、逐 transition 与有序 candidate runtime 证据见 [JSON 报告](cache_trigger_v3_production_runtime_gate.json)。
