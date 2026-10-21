# Stage3 反事实标签数量与过滤机制核查

## 结论摘要

每次完整 Stage3 更新日志中的 **8 个有效标签记录**由采集预算直接决定：每个 rollout 最多抽取 4 个 Macro 状态，每个状态最多抽取 2 个目标 Cache 节点，因此 `4 × 2 = 8` 条“状态—节点”标签记录。40 Slot rollout 含 20 个两 Slot Macro 决策，当前均匀抽取 4 个 Macro 位置；不是合法性过滤把较大的原始标签批次缩减到 8。

历史 Stage3 训练日志中这些目标标签全部通过 trace/label 合同校验，拒绝数均为 0。平局只影响相应辅助目标的有效项：Trigger 排序排除 `|best-ΔQ| ≤ 0.01` 的标签；Candidate 排序排除 `|ΔQ_i−ΔQ_j| ≤ 0.01` 的候选对；ΔQ Huber 回归仍使用每条有效标签中的全部合法 UPDATE 候选值。

8 条记录并不等于只有 8 个 ΔQ 数值：每条包含该节点所有合法 UPDATE 候选的值。完整训练更新日志每次有 80–336 个 Candidate ΔQ 目标、数百到数千个非平局 Candidate 对，但这些候选都来自每次仅 4 个 Macro 状态的少量共享 trace，不能视为同数量的独立状态样本。在线更新的 best-ΔQ 正负分布和目标节点 ID 未被逐条写入日志；其分布充分性因此仍未完全确定。

## 1. 标签生成、筛选与 Trainer 使用路径

### 采集预算为何产生 8 条记录

Stage3 Runner 的默认参数是 `max_macro_states_per_rollout=4` 和 `max_target_nodes_per_state=2`（`scripts/training/run_mappo_stage3_cache_value.py:111-112`；配置字段见 `src/ppo_simple/cache_counterfactual_value.py:96-109`）。Trainer 在一个 rollout 的 Macro 决策中按等间隔选择不超过 4 个位置；40 Slot、两 Slot Macro 下有 20 个 Macro 状态，选择位置为 0、5、10、15（`src/ppo_simple/mappo_baseline.py:357-365, 846-849`）。每个所选状态再按 9 个 Cache 节点循环选择最多两个目标节点（同文件 `:368-380`），并在对应 Macro 时刻保存决策前 Cache/UAV、候选配置、外生 Slot 序列和 Micro 共同随机数。

所以每次完整 rollout 的上限为 4 条 trace、8 条状态—节点 label record。两个 Slot 的 smoke rollout 只有一个 Macro 状态，日志相应显示 1 条 trace、2 条标签；完整训练 rollout 则显示 4 条 trace、8 条标签。这与采集预算吻合。

### 反事实分支与标签形态

`evaluate_counterfactual_trace` 对每个被采样目标节点评估 KEEP 参考配置和所有合法 UPDATE 配置，再用每个 UPDATE 分支的有限期 SMDP 回报加冻结 Critic bootstrap 减去 KEEP 值，生成该节点全部合法候选的 ΔQ 向量（`src/ppo_simple/cache_counterfactual_value.py:1000-1098`）。如两个合法候选形成相同 Cache 配置，可复用分支结果；本批已检查的训练日志中 `branch_cache_reuse_count=0`。

因此日志中的计数有三个不同单位：

- `valid_label_count` / `target_record_count`：通过检查的状态—节点标签记录数，完整更新为 8。
- `legal_update_candidate_count`：上述标签包含的 UPDATE 候选 ΔQ 标量数，也是 Huber 回归的实际合法目标数量。
- `branch_execution_count`：实际执行的分支数，包含 KEEP 基准及 UPDATE 候选分支；本批日志每条 label 对应一个 KEEP 分支，故可见 `branch_execution_count = legal_update_candidate_count + valid_label_count`。

### 严格性检查与拒绝

Trainer 核实 trace 与行为 Macro transition 的动作、Cache、候选 mask/features 一致；分支结果目标节点顺序一致；标签候选索引与合法 UPDATE mask 完全一致；`best_delta_q` 与候选 ΔQ 最大值相符；全部特征和目标有限。Trace 或单条 label 违规会计入明确的拒绝原因，不会以零标签替代；空有效批次会直接拒绝（`src/ppo_simple/mappo_baseline.py:1442-1635`）。

检查的训练日志范围内没有 trace 或 label 被拒绝，也没有记录拒绝原因。拒绝不是 8 条限制的来源。

### 三种损失的不同过滤

- **Trigger 排序：** 输入 `best_delta_q`，仅保留有限且 `|best-ΔQ| > trigger_tie_tolerance`（冻结值 `0.01`）的 label record。方向由 `sign(best-ΔQ)` 决定，作用于合法 UPDATE 总组 log-odds 与 KEEP 的比较。该损失的 `stage3_trigger_comparison_count` 是实际进入 Trigger 损失的记录数，不是动作准确率（`cache_counterfactual_value.py:704-726`；`mappo_baseline.py:2013-2036`）。
- **Candidate 排序：** 在同一 label 的合法 UPDATE 候选中，仅使用 `|ΔQ_i−ΔQ_j| > candidate_tie_tolerance` 的候选对。函数先对每个状态—节点的有效 pairwise loss 求均值，再对节点均值求均值；日志 `stage3_candidate_pair_count` 只记非平局比较数，不等于独立样本数（`cache_counterfactual_value.py:729-781`；`mappo_baseline.py:2175-2225`）。
- **ΔQ Huber：** 对有效 label 中所有合法 UPDATE ΔQ 值做回归；它不因目标值接近零而过滤，KEEP 索引不是回归目标（`cache_counterfactual_value.py:784-808`；`mappo_baseline.py:2050-2088`）。

### 现有在线训练日志计数

下表为 Stage3 原始主训练和解耦 Update 17–24 受控训练的日志。`Candidate ΔQ 数`等于合法 UPDATE 候选数；候选对数为已经排除平局后的比较数。各条记录的 trace/label 拒绝数均为 `0/0`，拒绝原因映射为空。

| 日志组 | 更新 | trace / 更新 | 有效 label / 更新 | 分支执行 / 更新 | 合法 Candidate ΔQ / 更新 | Trigger 实际比较 / 更新 | 非平局 Candidate 对 / 更新 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage3 原主训练 | 14–16 | 4 | 8 | 88–344 | 80–336 | 8 | 341–7,532 |
| 解耦受控训练 | 17–24 | 4 | 8 | 88–322 | 80–314 | 7–8 | 347–6,677 |

解耦受控训练逐更新计数为：

| Update | 分支 | label | 合法候选 ΔQ | Trigger 比较 | Candidate 非平局对 |
|---:|---:|---:|---:|---:|---:|
| 17 | 230 | 8 | 222 | 7 | 3,760 |
| 18 | 112 | 8 | 104 | 8 | 847 |
| 19 | 134 | 8 | 126 | 8 | 1,795 |
| 20 | 296 | 8 | 288 | 8 | 6,592 |
| 21 | 322 | 8 | 314 | 8 | 6,677 |
| 22 | 160 | 8 | 152 | 8 | 1,831 |
| 23 | 88 | 8 | 80 | 8 | 347 |
| 24 | 226 | 8 | 218 | 8 | 4,704 |

因此 Update 17 的 Trigger 比较数少于 8，是一条（或在另一独立 Update 17 运行中两条）有效 label 的 best-ΔQ 落入 Trigger 平局容差；这没有让相应标签从 batch 中失效，仍可进入 Candidate/Huber 目标。在线日志未保存全部候选 ΔQ 向量，也未保存 Candidate 平局前的候选对总数，故不能从这些 JSONL 反推出逐更新的确切 Trigger 平局标签 ID 或在线 Candidate 平局数。

## 2. 已保存标签的 ΔQ 分布与节点覆盖

完整 Stage3 PPO 更新只在 JSONL 保存汇总计数，没有保存每条在线 label 的 `best_delta_q`、目标节点 ID 或 ΔQ 向量。因此，下面的分布来自既有的独立冻结标签文件 `runs/research_mappo_stage3_candidate_supervised_learning/candidate_supervised_20261018T115258Z_f4fd13/`，不是 Update 14–24 的在线标签分布。该批标签由 Update 16 冻结策略/Critic 在 Training seeds 20031–20038 生成 16 个训练状态—节点记录；seeds 20039–20042 的 8 个诊断记录未用于梯度更新。label 定义同为有限期 Macro SMDP 回报加冻结 Critic bootstrap。

### best-ΔQ（按 Trigger 的 `0.01` 容差分类）

| 样本 | 记录数 | `best-ΔQ > +0.01` | `< −0.01` | `|best-ΔQ| ≤ 0.01` | 进入 Trigger 排序的记录 |
|---|---:|---:|---:|---:|---:|
| 固定 Training 标签 | 16 | 11（68.75%） | 1（6.25%） | 4（25.00%） | 12，其中正 11、负 1 |
| 独立诊断标签（不训练） | 8 | 8（100%） | 0 | 0 | 8，全部正 |

Training best-ΔQ 均值 `+0.05913`，中位数 `+0.07647`，范围 `−0.07526` 至 `+0.13567`。这批固定训练记录中 Trigger 的非平局监督方向约 91.7% 为正；独立诊断记录全为正，不能作为平衡的负例证据。它提示 Trigger 样本类别可能偏斜，但不能替代 Update 14–24 的在线标签统计。

### 合法 Candidate ΔQ 与 Candidate 平局

| 样本 | 合法 UPDATE ΔQ 数 | `>+0.01` | `<−0.01` | `|ΔQ|≤0.01` | 所有候选对 | 平局候选对 `|ΔQ_i−ΔQ_j|≤0.01` | 实际非平局对 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 固定 Training 标签 | 440 | 35（7.95%） | 392（89.09%） | 13（2.95%） | 8,764 | 332（3.79%） | 8,432 |
| 独立诊断标签 | 312 | 27（8.65%） | 280（89.74%） | 5（1.60%） | 7,372 | 293（3.97%） | 7,079 |

候选值本身接近零与一对候选互为平局是不同概念：Candidate 排序只排除两候选 ΔQ 相近的 pair；某个 `|ΔQ|≤0.01` 的候选仍可能与较远候选形成有效排序对。全部分布均按合法 UPDATE mask 统计，排除了 KEEP 和 padding 零值。

### 节点角色覆盖

| 标签集 | SV 行数 / best 正、负、平 | RSU 行数 / best 正、负、平 | UAV 行数 / best 正、负、平 | 合法候选数 SV / RSU / UAV |
|---|---:|---:|---:|---:|
| Training 16 行 | 8 / 6、0、2 | 4 / 2、1、1 | 4 / 3、0、1 | 80 / 224 / 136 |
| 诊断 8 行 | 2 / 2、0、0 | 4 / 4、0、0 | 2 / 2、0、0 | 20 / 224 / 68 |

节点顺序为 SV×5、RSU×2、UAV×2。在线代码按节点索引循环抽样，但训练日志不保存每条 record 的节点 ID；所以无法给 Update 14–24 逐更新的实际 SV/RSU/UAV 覆盖表。固定标签 archive 的角色比例是可证实的独立数据，不应冒充在线覆盖比例。另因 Huber 对所有候选标量取均值，候选数更多的 RSU 行对回归标量损失贡献更大；Trigger 与 Candidate pairwise loss 则先按状态—节点行聚合，语义不同。

## 3. ΔQ 价值头指标与 Actor 指标不可混用

- Stage3 ΔQ 价值头的 `delta_q_sign_accuracy`、`delta_q_rmse/mae` 测量数值预测；其 `candidate_pairwise_rank_accuracy` 测量 **价值头预测值**对固定 ΔQ 标签的排序。
- Validation 摘要里的价值头 `trigger_accuracy` 将真实 `max(ΔQ)>+0.01` 与价值头预测 `max(predicted ΔQ)>+0.01` 比较。既有三场景日志均为真实正例 8、价值头预测正例 0、准确率 0/8。它不是 Trigger Actor 的 UPDATE 决策准确率。
- Trigger Actor 辅助项直接将有效 label 的 `best-ΔQ` 符号与 Trigger UPDATE-vs-KEEP log-odds 做排序；实际确定性 Trigger 动作则来自 Actor 概率组解码。`stage3_trigger_comparison_count` 仅表示进入该排序 loss 的监督行数，不表示模型判对了多少条。
- Candidate Actor 的条件候选排序指标使用其独立 conditional logits；它与价值头候选排序指标不同。Actor 最终选中动作/收益也不能由价值头的 RMSE 或排序准确率直接替代。

上述接口与定义来自 `scripts/evaluation/run_mappo_stage3_main_experiment.py:135-190` 和 `src/ppo_simple/mappo_baseline.py:1983-2036, 2175-2225`。

## 充分性判断与建议

- **有效数量上限原因：已解释。** 在线每轮 8 条来自明确的采样上限，非过滤结果。
- **过滤合同：与实现、日志一致。** 完整训练日志拒绝数为零；平局过滤发生在损失层，而不是把有效反事实 label 伪装成无效或零值。在线 Candidate 被排除的 tie 数无法由聚合日志恢复。
- **Trigger 分布是否足够：尚不能确认。** 每次只有 6–8 条有效 Trigger 比较，分布在 4 个 Macro 状态；独立固定 Training 标签显示 11 正、1 负、4 平，更新级在线标签符号直方图又未保存。证据不足以证明其支持稳定的正负 Trigger 校准，也不足以证明历史固定样本等同于在线训练分布。
- **Candidate 监督：** 每轮存在 80–336 个候选目标及 347–6,677 个实际非平局 Candidate 对，足以执行并反传排序目标；但 pairwise 数量共享同一批至多 4 个 Macro 状态，不能据此认为拥有数千个独立环境样本，也不能从损失记录证明跨状态学习充分。

**建议的最小下一步是只补全在线标签构成日志，不先提高反事实预算。** 在现有 `stage3_counterfactual` 摘要中记录每条有效 label 的节点 ID/角色、`best-ΔQ` 正/负/平计数、合法候选数、合法候选 ΔQ 正/负/近零计数，以及 Candidate 总 pair 数/平局 pair 数/有效 pair 数；同时记录 Huber 标量计数和 Trigger 有效行数。这样不用多跑反事实分支，就可判断是否需调整四状态/每状态两节点的采集预算，并可区分覆盖问题与损失过滤。

```makefile
EFFECTIVE_LABEL_COUNT_LIMIT_EXPLAINED = YES
LABEL_FILTERING_CONTRACT_VALID = YES
TRIGGER_LABEL_DISTRIBUTION_ADEQUATE = UNRESOLVED
VALUE_HEAD_AND_ACTOR_METRICS_DISTINGUISHED = YES
READY_FOR_MINIMAL_FIX = YES (diagnostic logging only; do not change the sampling budget yet)
```