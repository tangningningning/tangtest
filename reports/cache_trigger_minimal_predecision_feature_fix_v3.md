# Cache Trigger 最小 pre-decision 特征修复 V3

## 实现范围

只把审计确认的两个 Tier‑A pre-decision 特征接入 Cache Trigger：`node_local_predicted_service_workload[m,f,:]` 和 `update_candidate_policy_summary[m]`。原 400-D Cache context、shared `cache_encoder`、candidate scorer 的输入路径和 UAV policy 保持原样。没有加入 actual task、future result、Tier‑B 特征，也没有改 reward、cost、PPO loss 或 checkpoint 内容。

改动涉及：

- `src/decision/cache_configuration.py`：用 prediction scenario、固定 serviceability、catalog 与 jitter scale 构造归一化 node-local workload。request-mass channel 复用 `node_specific_predicted_demand`；input-bit 与 CPU-cycle channels 沿用相同预测事件和节点覆盖权重。
- `src/ppo_simple/primitive_control.py`：先枚举合法 target，调用现有 candidate feature builder 和 candidate scorer 一次；从其 detached conditional probabilities 计算 candidate summary，再采样 Trigger。若选 UPDATE，直接用同一份 logits 采样 target。
- `src/ppo_simple/policy.py`：把 proposed Cache Trigger architecture 提升为 version 3。每个 head 输入原 `cache_encoder(base_context)` embedding 与该 head 的辅助特征。candidate scorer 仍走原 embedding 加原 candidate features 的路径。
- `src/ppo_simple/rollout.py`：将 workload、candidate summary 和拼接后的 aux 复制为 rollout-owned float32 snapshot，并保存诊断 norm/count/weighted summaries。
- `src/ppo_simple/trainer.py`：PPO 全部 Cache Trigger evaluation 使用 rollout 中的 detached aux。更新开始时验证 aux 与已保存的两部分一致；每个 PPO epoch 验证快照未改变。V3 policy 拒绝缺少 architecture version 或 version 不匹配的 checkpoint。
- `src/ppo_simple/formal_evaluation.py`：checkpoint architecture 要求提升到 version 3。

## 特征和数据流

生产配置下，workload builder 返回 `[9,10,3]`。每个 head 的三个通道依次为 forecast request mass、input-bit workload、CPU-cycle workload。归一化使用固定的 task-vehicle/horizon 数量、`beta_discount`、service catalog 最大输入量和 cycles-per-bit，以及配置的 input/load jitter；没有使用 batch/sample 统计量。节点覆盖调用现有 node-specific demand/serviceability 语义。TV local-cache 对边缘机会的影响由现有 candidate feature builder 里的 local usage gain 表示，并由候选行为概率加权。

当前 target candidate feature 维度为 85，因此 candidate summary 为 86 维：`legal_candidate_count / configured_candidate_capacity` 加行为策略 `pi_old(c | UPDATE,s)` 加权的现有 candidate feature 向量。它包含 before/target/delta service 信息、预测需求和 TV local-cache usage gain、loaded/released MB、现有 cache-cost/loading-energy proxy、capacity 与 replica/diversity 信息。没有传入 `Q_c`、reward、critic、Matching/KKT 或实际任务结果。

生产配置下每个 Trigger aux 为 `30 + 86 = 116` 维。其生成顺序是：

1. 构建当前 head 的原 conditional 400-D base context。
2. 枚举该 head 的合法 UPDATE targets；如有候选，构建现有 candidate features，并由原 candidate scorer 得到 logits。
3. 对该次 logits 做 detached conditional softmax，形成 candidate policy summary。空候选时 summary 固定全零，operation mask 将 UPDATE 设为 false。
4. 拼接该 head 的 node-local workload 与 summary，采样 KEEP/UPDATE。
5. 如选 UPDATE，从步骤 2 的相同 logits 采样 candidate；不重新调用 scorer，不重新采样一套分布。

行为 rollout 将 workload、summary 和拼接后的 aux 分别复制进 transition/rollout buffer。PPO update 只读取 buffer 中保存的 aux；不会在更新 epoch 里用变化后的 candidate scorer 重建 summary。运行时检查 aux 必须 detached、必须等于两个已保存组件的拼接，并且在所有 PPO update epoch 中保持不变。

## 必须状态

```text
ARCHITECTURE_VERSION = 3
NODE_LOCAL_PREDICTED_WORKLOAD_ADDED = YES
UPDATE_CANDIDATE_POLICY_SUMMARY_ADDED = YES

ACTUAL_TASKS_IN_TRIGGER = NO
FUTURE_INFORMATION_IN_TRIGGER = NO

CANDIDATE_SCORER_SEMANTICS_CHANGED = NO
CACHE_COST_FORMULA_CHANGED = NO
REWARD_CHANGED = NO
UAV_POLICY_CHANGED = NO

TRIGGER_AUX_STORED_DETACHED = YES
TRIGGER_AUX_RECOMPUTED_DURING_PPO_UPDATE = NO

PRE_TRIGGER_AND_EXECUTION_CANDIDATE_LOGIT_MAX_DIFF = 0.0
V2_RESUME_INTO_V3_ALLOWED = NO

TRAINING_EXECUTED = NO
VALIDATION_EXECUTED = NO
TEST_EXECUTED = NO
```

V2 architecture version 与 V3 不匹配，trainer 拒绝恢复；缺失 architecture metadata 的 proposed checkpoint 也拒绝恢复。V3 training 应使用新 run directory 和 seed-controlled fresh initialization，不能复用 corrected-v2 run 名称或目录。

## 确定性检查

在 `adaptive_cache_mec` 环境对应的 Python 3.11.15 解释器上通过必要检查：

| 检查 | 结果 |
|---|---|
| 修改模块 import 与 syntax compile | PASS |
| architecture version 为 3 | PASS |
| 400-D base context 未扩展 | PASS |
| Trigger forward 接收 aux，batched PPO policy evaluation 可用 | PASS |
| 原 base embedding + candidate features scorer 路径可运行 | PASS |
| 修改 forecast workload 会改变 node-local summary | PASS |
| actual task 值不是任何 Trigger feature builder 的输入 | PASS |
| 修改合法 candidate set 会改变 candidate summary | PASS |
| 空 candidate summary 为零且 UPDATE mask 为 false | PASS |
| rollout snapshot 是 copied、detached float32 | PASS |
| 模拟 4 次 candidate scorer 参数变化后，已保存 aux 和 Trigger logits 不变 | PASS |
| pre-trigger 与执行侧 candidate logits 最大差 | `0.0`（阈值 `1e-7`） |

这些检查使用合成小输入与临时脚本，没有收集 rollout、加载 training/validation/test 数据、执行 backward 或调用 optimizer。`conda run` 的 Windows 写权限预检无法写入 Conda 用户目录；随后直接使用同一 `adaptive_cache_mec` 环境的解释器完成检查。

```text
TRAINING_EXECUTED = NO
ROLLOUT_EXECUTED = NO
OPTIMIZER_STEP_CALLS = 0
BACKWARD_CALLS = 0
NEW_EXACT_Q_BRANCHES = 0
VALIDATION_EXECUTED = NO
TEST_EXECUTED = NO
```

临时检查脚本和 `__pycache__` 已清理。`git diff --check` 通过；未 commit、未 push。
