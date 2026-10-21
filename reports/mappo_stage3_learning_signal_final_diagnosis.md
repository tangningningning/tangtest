# Stage3 最终学习信号诊断

日期：2026-10-18

## 结论摘要

- 本次唯一在线训练批次含 8 条有效反事实标签，best-ΔQ 为正 8 条、负 0 条、近零 0 条。该批次呈正值偏向，但样本量受既定采样上限约束，不能据此推断持续的在线标签分布。
- PPO Macro Actor 与 Trigger 辅助损失在同一批 Macro transition、同一轮参数快照上的共享参数梯度余弦为 `-0.1698916`，坐标冲突比例为 `0.5347644`。因此本批次的聚合梯度方向冲突；但加权辅助梯度范数约为 PPO 共享梯度范数的 `1.21%`。
- 一批测量足以确认本次更新存在梯度方向冲突，尚不足以说明冲突是稳定现象或损失配比不合适。未发现标签无效、候选对监督大量被平局排除或 Huber 标签缺失的证据。暂不改变损失系数或采样预算，也不启动后续训练。

## 冻结合同与执行记录

本次使用唯一指定的 Update 24 Checkpoint：

| 项目 | 值 |
|---|---|
| 源 Checkpoint | `runs/research_mappo_stage3_decoupled_controlled_evaluation/training/stage3_20261018T124359Z_ff9ab9d0/checkpoints/checkpoint_update_0024.pt` |
| 源 SHA-256 | `41e979bf193e13cd4f20f272dae5dbb77b86936ce0e61db56345ccc82fa87650` |
| 冻结损失 Profile SHA-256 | `c91bccc6bf91f76c407a18c9953b4fa2c2d346fcfef3337ad63ee49b23d8bdb4` |
| Training Pool Manifest SHA-256 | `b6cae236b029f4d24c7488a91f0825247a20f475ec76b2d80737de44ae491bd3` |
| 本次数据 | Training seed `20031`，origin `0`，连续 `40` Slot；不是完整 Episode |
| 运行目录 | `runs/research_mappo_stage3_learning_signal_final_diagnosis/stage3_20261018T135013Z_c176e93a` |
| 完成状态 | `completed_updates=1`，`failure=null`；Validation/Test 均未访问 |
| 收集 / 更新耗时 | `10.117 s` / `46.050 s` |
| 新 Checkpoint | `checkpoints/checkpoint_update_0025.pt`，SHA-256 `d3bcdf1af74d075509c33a4cc183b8e8014729dd4069e8706245b3724e91de16` |

本次使用 `adaptive_cache_mec` 的 Python 3.11.15 解释器。Profile、数据池合同、任务/奖励合同、损失系数及训练参数保持冻结。Runner 完成一次正常 `Trainer.update()`；梯度诊断没有执行 optimizer step，Actor 正常更新仍只执行原合同中的更新步骤。Training rollout 报告 40 个 Slot、4 条已选反事实 trace、Matching+KKT 4287 次调用；约束合法、参数有限，数值异常数为 0。

既有结论来自有效标签数审计、Trigger 梯度方向审计和缓存更新回归诊断；本报告没有重跑固定标签方向检查或旧策略评价。新增实现位于 `src/ppo_simple/mappo_baseline.py`：逐标签计数和汇总约在 1570–1785 行，同批梯度读取约在 2200–2420 行，训练指标序列化约在 2735–2831 行。日志保存在本次独立运行目录的 `training_metrics.jsonl`。

## 训练日志补全及标签统计

`src/ppo_simple/mappo_baseline.py` 在反事实 batch 汇总处新增逐标签目标节点 ID/角色、best-ΔQ 分类、合法 UPDATE 数和候选 ΔQ 分类；并汇总 Trigger 有效监督行、平局排除行、Candidate 总比较对/平局对/有效排序对。逐标签记录和批次汇总写入 `stage3_counterfactual`，与本次实际参与更新的标签相连。Huber 标量数取实际 `delta_q_huber_loss` 返回的 mask 内有效标量数，不使用历史标签统计。

| 诊断项 | 本批次 |
|---|---:|
| 有效 trace / 有效标签 | 4 / 8 |
| 标签 best-ΔQ 正 / 负 / 近零 | 8 / 0 / 0 |
| best-ΔQ 均值；范围 | `0.305048`；`[0.092523, 0.538558]` |
| Trigger 有效监督行 / 平局排除行 | 8 / 0 |
| 合法 UPDATE 候选总数 | 336 |
| 候选 ΔQ 正 / 负 / 近零 | 107 / 218 / 11（31.85% / 64.88% / 3.27%） |
| Candidate 候选对总数 / 平局对 / 有效排序对 | 7888 / 203 / 7685（平局 2.57%） |
| Huber 实际回归标量数 | 336 |
| 无效 trace / 无效标签拒绝数 | 0 / 0 |

逐标签记录（候选 ΔQ 分类按 Candidate 平局容差，best-ΔQ 分类按 Trigger 平局容差）：

| 目标节点 | 角色 | best-ΔQ | 类别 | UPDATE 候选数 | 候选 ΔQ 正/负/近零 |
|---|---|---:|---|---:|---:|
| `58` | SV | 0.092523 | 正 | 10 | 1 / 9 / 0 |
| `RSU_0` | RSU | 0.105244 | 正 | 56 | 1 / 54 / 1 |
| `RSU_0` | RSU | 0.427635 | 正 | 56 | 15 / 39 / 2 |
| `RSU_1` | RSU | 0.388468 | 正 | 56 | 16 / 37 / 3 |
| `RSU_1` | RSU | 0.538558 | 正 | 56 | 35 / 20 / 1 |
| `UAV_0` | UAV | 0.361488 | 正 | 34 | 18 / 14 / 2 |
| `UAV_0` | UAV | 0.115795 | 正 | 34 | 1 / 32 / 1 |
| `UAV_1` | UAV | 0.410674 | 正 | 34 | 20 / 13 / 1 |

八条 best-ΔQ 标签均高于 Trigger tie tolerance `0.01`，所以这批 Trigger 行全部参与监督。候选值本身多数为负，但每条标签的合法候选集合至少有一个正的 best-ΔQ；不能把“best 标签为正”误读为所有 UPDATE 候选都为正。该 batch 没有 Trigger 平局样本，故不能判断平局排除机制在在线状态中的发生率。

## 同批次 PPO 与辅助梯度

在第一 PPO epoch、同一批 Macro Actor loss 和 Trigger ranking loss 尚未更新参数时，分别用 `torch.autograd.grad(..., retain_graph=True, allow_unused=True)`读取梯度。它不写入 `.grad`，不调用 optimizer；随后沿原路径对总 Macro Actor loss 执行一次正常 backward/step。比较范围为 Macro Actor 中共享 Trigger 决策的参数，包含 10 个参数张量。

| 指标 | 值 |
|---|---:|
| PPO Macro Actor loss | `-5.94556e-7` |
| Trigger auxiliary loss（未加权） | `0.645923` |
| PPO 共享参数梯度 L2 | `0.290433` |
| Trigger 辅助共享参数梯度 L2（未加权 / 加权） | `0.351296 / 0.00351296` |
| 共享参数梯度余弦 | `-0.169892` |
| 两梯度均非零的坐标 / 其中方向相反坐标 | `111105 / 59415` |
| 坐标冲突比例 | `0.534764` |
| 梯度诊断状态 | `MEASURED`；诊断自身未调用 optimizer step |

按预先采用的判断口径，负余弦表示本次聚合方向冲突；坐标冲突比例也显示分量级冲突。加权辅助梯度范数约为 PPO 共享梯度的 1.21%，但范数比例不能单独证明辅助目标无害，也不能说明应当调大或调小系数。该结果是单一 batch、单次更新的局部方向测量，不代表期望梯度或训练稳定性。

## 判断与下一步

1. **标签分布：**本批次明显偏正（8/8），但不足以判断长期在线分布是否偏正；本次按冻结采样上限得到 8 条，未发生拒绝。没有证据支持增加采样预算。
2. **梯度关系：**本批次测得聚合梯度冲突（余弦 `-0.169892`），冲突坐标比例 `53.48%`。尚无跨批次证据表明冲突持续存在；加权辅助梯度相对 PPO 共享梯度较小，因此不据此修改系数。
3. **候选监督：**有效排序对 7685/7888，平局仅 203 对；Huber 使用了全部 336 个合法 UPDATE 标量。此处未见标签管线漏数或明显由平局导致的监督缺失。
4. **建议：**保持现有损失系数与采样预算。当前唯一直接证据是一次 batch 的方向冲突，而非稳定训练退化的因果证据。若后续获准继续诊断，应先在自然后续更新中重复记录相同字段，再决定是否修改机制；本次不继续训练。

## 最终字段

```ini
ONLINE_TRIGGER_LABEL_DISTRIBUTION = 本批次偏正（best-ΔQ：正8、负0、近零0；长期分布未定）
PPO_AUX_GRADIENT_COSINE = -0.1698916（共享 Trigger 参数）
PPO_AUX_GRADIENT_CONFLICT = YES（本批次；坐标冲突比例53.48%）
CHANGE_LOSS_COEFFICIENTS_JUSTIFIED = NO
CHANGE_SAMPLING_BUDGET_JUSTIFIED = NO
READY_FOR_MECHANISM_CHANGE = NO
```
