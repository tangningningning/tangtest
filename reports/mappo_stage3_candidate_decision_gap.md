# Stage3 Candidate 实际决策质量判别

## 结论

**判定：A（Candidate Top-1 选择存在明显失配，下一步优先让监督目标直接优化贪心 Top-1）。** 最新实验保存的 64 条 Training 反事实标签中，贪心 Candidate 的平均有限期 ΔQ 为 **−0.2075**，合法 UPDATE 候选最优 ΔQ 平均为 **+0.3888**，平均 Top-1 regret 为 **0.5964**；其中 **42/64（65.6%）** 的状态选中的贪心 Candidate ΔQ 小于 −0.01，同时至少存在一个 ΔQ 大于 +0.01 的合法 Candidate。这足以证明抽样状态上有明显 Candidate 选择错误。

这**不能证明 Candidate 错误解释了 Validation 净效用差距的主要部分**：标签来自 Training 的少量抽样状态，而 Validation JSON 没有逐决策 Candidate 标签；单节点、有限期条件 ΔQ 也不能等同九节点联合 Cache 策略收益。以下分别报告两种证据范围。

## 产物与指标口径

- 实验报告：`reports/mappo_stage3_greedy_candidate_trigger_experiment.md`。
- Training 指标：`runs/research_mappo_stage3_greedy_candidate_trigger_20261018/training/stage3_20261018T153035Z_5a99539b/training_metrics.jsonl`。
- Validation：`runs/research_mappo_stage3_greedy_candidate_trigger_20261018/validation_paired_current_adapter/validation_results.json`、`paired_validation_comparison.json` 和 `evaluation_metadata.json`。
- 起点为 Update 24 权重初始化；本实验完成 8 次更新、使用 Training seed 20031–20038。最终 Checkpoint 为 `checkpoint_update_0008.pt`，SHA-256 `ca503f1769f88aa792d4904169c37f03d3ed5f29445dbf121f27e2c36fa03f83`；Run 完成记录为 8/8、无失败。
- ΔQ 标签语义是 `finite_horizon_macro_smdp_return_plus_frozen_critic_bootstrap`；Trigger 标签使用行为策略合法 UPDATE 候选的贪心 Candidate。该值是**有限期条件价值估计**，不是长期最优价值或 Validation 实际收益。

## 抽样状态上的 Candidate 质量

| 指标 | 结果 |
|---|---:|
| 有效标签行 | 64（8 次更新各 8 行） |
| 贪心 Candidate 的平均 ΔQ | −0.2075 |
| 最佳合法 UPDATE Candidate 的平均 ΔQ | +0.3888 |
| 平均 Top-1 regret（best − greedy） | 0.5964 |
| Top-1 regret 中位数 / 经验 P90 | 0.5030 / 0.9713 |
| 贪心 ΔQ < −0.01 且 best ΔQ > +0.01 | 42/64（65.6%） |
| best ΔQ > +0.01 | 60/64（93.8%） |

按目标节点角色分组的平均 regret：SV **0.4973**（35 行）、RSU **0.8695**（16 行）、UAV **0.5271**（13 行）。这些行来自训练中的反事实抽样，不是独立 Validation 样本；同一轮内多个节点和状态也不应视作统计独立观测。

训练日志统计了 33,508 个候选对，其中 32,816 个通过平局过滤、692 个被记为平局；这只是参与 Candidate 排序监督的 pair 数，**不是排序正确数**。日志没有保存 Candidate pairwise ranking accuracy，也没有完整逐候选 ΔQ 向量，因此该准确率不可从现有产物计算。排序损失值不能替代准确率。

### Trigger 与贪心候选的关系

在同一批 64 个被标注状态上，根据 Trace 的实际操作及其所记录的已选操作组概率还原 `P(UPDATE)`，平均为 **0.5090**；64/64 状态距 0.5 不超过 0.05。按正式确定性操作组 argmax，47/64 状态选 UPDATE、17/64 选 KEEP。47 个确定性 UPDATE 状态中，**31 个**对应的贪心 Candidate ΔQ < −0.01，且 best ΔQ > +0.01；另有 **13/17** 个确定性 KEEP 状态的 best ΔQ > +0.01。这显示抽样状态上操作边界和贪心候选价值目标都值得关注。

Rollout 实际随机采样得到 26/64 个 UPDATE。Trace 只保留了确定性贪心候选的 ΔQ 摘要，没有保留全部候选 ΔQ 向量；这 26 个实际采样动作中仅 1 个 Candidate 索引与保存的贪心索引相同。因此，**随机 Rollout 实际抽中 Candidate 的 ΔQ 符号不可恢复**，不能把上面的 31/47 表述成随机采样动作的实测负收益。31/47 只指同一标注状态按确定性组 argmax 与确定性 Candidate 贪心解码时的结果。

## Validation 与 KEEP+STAY

| Seed | Greedy 任务效用 | Greedy Cache 成本 | Greedy 净效用 | KEEP+STAY 任务/净效用 | Greedy−KEEP 任务差 | Cache 成本差 | 净效用差 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30001 | 3951.5666 | 30.2699 | 3921.2967 | 3976.1384 | −24.5718 | +30.2699 | −54.8417 |
| 30002 | 3866.0277 | 30.2699 | 3835.7578 | 3889.2103 | −23.1825 | +30.2699 | −53.4525 |
| 30003 | 3918.8993 | 30.2699 | 3888.6294 | 3942.8919 | −23.9925 | +30.2699 | −54.2624 |
| 均值 | 3912.1645 | 30.2699 | 3881.8946 | 3936.0802 | −23.9156 | +30.2699 | −54.1855 |

UAV 成本在两策略均为 0。相对 KEEP+STAY 的净效用缺口由实际任务效用少 **23.9156** 和 Cache 成本多 **30.2699** 构成；在这一均值差分中，成本项约占 55.9%，任务效用项约占 44.1%。这描述的是完整 Episode **九节点联合策略**的系统结果，不是任一单个 Candidate 的因果贡献。

Validation 的逐 Slot 记录有任务效用、Cache 成本和动作计数，但没有逐 Macro 决策的概率、Candidate logits/条件分布或 ΔQ 标签。因此无法判断上述三个 Validation 场景的 Candidate Top-1 regret，也不能将训练抽样的 0.5964 regret 加总或外推为 54.1855 的净效用差。

## 唯一建议的下一步机制

优先将现有 all-pairs Candidate 排序目标改为**逐状态 best-vs-rest 的 Top-1 排序损失**：对每个有效标签取 `c* = argmax_c ΔQ(c)`，仅对 `c*` 与其他合法且 ΔQ 差超过现有 0.01 平局容差的候选计算现有温度下的 pairwise logistic loss；先在每个状态内平均，再对状态平均。保留现有损失系数与温度，不新增网络，不改 Trigger、奖励或成本合同。这样监督目标直接惩罚当前观察到的“贪心选中低于最佳候选”错误，而不是让大量非 Top-1 候选对主导平均排序损失。

实施前应先在同一批已冻结 Training 标签上确认 Top-1 选择率/regret 的损失梯度方向，并用独立的少量 Training 标签验证排序改善；若要声称解释系统差距，再单独取得少量 Validation 前缀的逐决策 Candidate 标签。当前报告不实施该损失，也不启动训练或回放。

## 尚未解决

1. 缺少 Validation 决策级 Candidate ΔQ 与选中索引配对数据，故 Candidate 失配是否为 Validation 差距的**主要**原因仍未证实。
2. 随机采样动作大多不同于确定性贪心 Candidate，现存摘要不足以统计这些实际采样动作的 ΔQ。
3. Candidate pairwise accuracy 未记录，不能由 pair 数、平局数或排序损失反推。
4. 单节点有限期 ΔQ 标签没有测量九节点联合动作之间的交互，不能据此断言联合净效用提升幅度。

本次只读取现有文件并新增本报告；没有训练、评价回放、数据修改或代码修改。
