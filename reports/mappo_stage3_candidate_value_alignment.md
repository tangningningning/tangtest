# Stage3 Candidate 价值对齐定向诊断

## 结论

**未发现可证明的 Candidate 标签映射、legal mask、概率归一化或梯度符号代码错误，因此本次没有修改 Stage3 算法源码或冻结系数。** 当前 Candidate 排序损失直接作用于 Macro Actor 的实际 UPDATE 条件候选 logits；在 legal UPDATE 支持上，两两 logit 差等于条件 log-prob 差。真实 Training 样本的一次有限更新确认，高 ΔQ 候选获得了正确方向的排序梯度，Macro Actor 收到非零 Candidate 梯度。

这不等于候选价值对齐问题已解决。Update 16 的两个目标节点在本次状态上近似均匀分配条件概率；一次真实更新后，一个 best candidate 概率仅略升、另一个略降。样本和更新数不足以判断后续学习能否纠正此前 Validation 中观察到的候选选择差距。没有证据支持任意改写损失或增大系数，因此保留当前 pairwise loss 与冻结 Profile。

## 实现合同核对

1. **真实 ΔQ 与索引映射。** `evaluate_counterfactual_trace` 对每个合法 candidate index 计算 `Q(candidate) - Q(KEEP)`，并按升序的 legal index 写入 `candidate_indices` / `candidate_delta_q`。`MAPPOBaselineTrainer._build_counterfactual_training_batch` 严格核验标签 index 与合法 UPDATE mask 完全一致，再逐项写入同索引的稠密 `delta_q_targets`；不合法、非有限或索引错位的标签会被拒绝（`src/ppo_simple/cache_counterfactual_value.py:1040–1097`、`src/ppo_simple/mappo_baseline.py:1450–1510`）。
2. **行为动作与配置映射。** Trace 的每个 candidate 配置是完整 Cache 矩阵；rollout 从同一 candidate index 取目标节点对应的矩阵行并写入活动配置。真实 Training 样本中，RSU_1 和 UAV_0 的 candidate mask、feature 行、动作 index 与执行配置节点行均匹配。首次诊断曾把完整矩阵和单节点行直接比较，断言形状不一致并在 `Trainer.update()` 前停止；按实际行级合同重新核验后通过。这不是算法映射错误。
3. **条件概率与 PPO 联合 log-prob。** Macro sampler 先从 KEEP/UPDATE 操作组分布采样，再仅在 UPDATE 时从合法 candidate 分布采样；回放 log-prob 是操作组 log-prob 加条件 candidate log-prob。Pairwise Candidate loss 使用同一 `_macro_cache_node_logits` 输出，并将 KEEP 从 candidate mask 中排除。cardinality correction 对同一节点的所有合法 UPDATE logits 减去同一个 `log(legal_count)`，因此不会改变 UPDATE 条件 softmax，也不会改变任意一对候选的 log-prob 差。
4. **Candidate 梯度。** Pairwise loss 对每个合法候选对按真实 ΔQ 差的符号训练 `score_i - score_j`。若 `ΔQ_i > ΔQ_j`，损失为 `softplus(-(score_i - score_j)/temperature)`；梯度下降会提高高值候选相对于低值候选的 score/log-prob。损失输入由 Actor candidate scorer 直接产生，并通过 Macro Actor 参数反传；独立 ΔQ 价值头的 Huber 损失不替代这条 Actor 路径（`src/ppo_simple/cache_counterfactual_value.py:729–781`、`src/ppo_simple/mappo_baseline.py:1911–1965`）。

综上，检查到的 Candidate 机制满足用户要求的“监督直接作用于实际 UPDATE 条件候选分布”。现有证据没有找到可直接修复的程序缺陷。Trigger、KEEP/UPDATE 组分布、Macro SMDP 回报、Micro、Matching+KKT、奖励和合法动作合同均未更改。

## 有界真实更新

- 起点：Stage3 Update 16 Checkpoint，SHA-256 `ca819d60320b3c65a46c2eb8217ca6d9e18f5a4bda2799f1778988f3976e611f`；冻结 Profile SHA-256 `c91bccc6bf91f76c407a18c9953b4fa2c2d346fcfef3337ad63ee49b23d8bdb4`。
- Training 场景：现有 Training Pool seed `20020`，Slot `0–1`，另带一个 bootstrap Slot；只执行 2 个真实 Slot、1 次 PPO epoch。该 seed 不属于本次已完成的 Stage3 Update 14–16 训练窗口 `20017–20019`。未读 Validation/Test 内容。
- 生成 1 条真实反事实 trace、2 个有限标签（RSU_1 56 个、UAV_0 34 个合法 UPDATE 候选），共 90 个候选标签、2,028 个非平局候选对；分支执行 92 次，标签/trace 拒绝均为 0。标签语义为有限期 Macro SMDP 回报加冻结 Critic bootstrap，不代表长期最优价值。
- 更新前的原始 Candidate 排序损失为 `0.695612`。单独对该损失求导得到 logit 梯度 L2 `0.066927`、Macro Actor 参数梯度 L2 `0.102385`；乘冻结系数 `0.01` 后的单独梯度范数为 `0.001024`。两个 best candidate 的 `dL/dlogit` 均为负，在 candidate-logit 空间沿负梯度方向的条件 log-prob 导数均为正。这验证了高 ΔQ 候选的训练信号方向。

| 节点 | 合法 UPDATE 数 | best candidate / ΔQ | 更新前条件概率 | 更新后条件概率 | 实际行为动作 |
|---|---:|---|---:|---:|---|
| RSU_1 | 56 | 6 / +0.126148 | 0.01835035 | 0.01835545 | KEEP，ΔQ 参考为 0 |
| UAV_0 | 34 | 3 / +0.036739 | 0.02850588 | 0.02850053 | KEEP，ΔQ 参考为 0 |

本次条件概率变化分别约为 `+0.00000510` 和 `-0.00000535`。实际参数步组合了 PPO 与辅助损失梯度，因此条件概率的净变化不必与 Candidate 单独梯度方向相同；第二个节点略降不足以分解各项梯度的独立贡献。两节点的实际 rollout 动作都是合法 KEEP，因此本样本没有观察到 UPDATE 后的错误实际内容选择。

PPO 实际更新结果：Macro Actor 参数确实变化（L2 `0.040581`），总 Macro Actor 梯度范数 `0.003633`；Stage3 Candidate ranking loss `0.695612`，pair count `2,028`。行为动作到新旧策略重算的 Macro joint / per-node log-prob 最大绝对误差均为 0；Macro approximate KL `1.55e-6`、clip fraction `0`。全部更新参数有限，数值异常数 `0`，Cache/UAV 动作合法，奖励合同误差 `0`。Matching+KKT 在两 Slot 实际执行。

训练收集用时约 `6.49 s`，优化用时约 `33.68 s`（其中反事实重放约 `33.59 s`）。

## 产物及边界

- 本轮唯一新增报告：`reports/mappo_stage3_candidate_value_alignment.md`。
- 有界真实更新结果：`runs/research_mappo_stage3_candidate_value_alignment/seed20020_update16_to_update17_20261018T111522Z_1d7b2e/candidate_alignment_update.json`。
- 已保存 Update 17 Checkpoint：`runs/research_mappo_stage3_candidate_value_alignment/seed20020_update16_to_update17_20261018T111522Z_1d7b2e/checkpoint_update_0017.pt`，SHA-256 `0581abb6a6246c5771d949028ccb60e00eb96d32ae25d2c603a7f220c5609809`。
- 未修改任何源码、正式配置、成本/奖励、数据或 Checkpoint 历史文件；没有改写 Stage3 损失目标或调整系数。
- 未运行长训练、完整 Episode、Validation、Test、Stage2 或参数搜索。所有 Python 调用使用 `-B`，没有生成临时脚本或新增 Python 缓存；`environment_data/` 只读。

## 剩余问题

Candidate 监督路径本身工作正常，但当前样本的 Actor 候选分布仍近似均匀，Candidate loss 仍接近 `ln(2)`。单次更新只验证实现和方向，不能解释为何既有策略在有限验证前缀里经常选中低 ΔQ 候选，也不能证明更长训练会改善。当前证据不足以支持改换 Candidate 损失形式或冻结系数；完整价值对齐结果仍为 **UNRESOLVED**。
