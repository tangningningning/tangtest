# B3-15C：多节点联合 Cache 更新真实收益归因

## 结论

本次指定的 Update 75 Run 和目标 Episode 指标存在，但没有持久化足以重建目标 Macro 决策前状态、实际联合动作及固定 UAV 轨迹的逐帧记录。因此无法在原始状态上执行 A（实际联合动作）、B（全节点 KEEP）和 Cᵢ（撤销单节点 UPDATE）分支。本报告不把重新采样的动作或近似状态当作原始 Rollout，也未运行反事实 Matching/KKT。

```text
JOINT_UPDATE_POSITIVE_FRACTION = INSUFFICIENT_EVIDENCE (0/0 可评价状态)
JOINT_UPDATE_NEGATIVE_FRACTION = INSUFFICIENT_EVIDENCE (0/0 可评价状态)
JOINT_UPDATE_MEAN_DELTA_RETURN = NOT_MEASURED
NEGATIVE_NODE_MARGINAL_FRACTION = INSUFFICIENT_EVIDENCE (0/0 可评价节点)
JOINT_ACTION_INTERACTION_EVIDENCE = INSUFFICIENT_EVIDENCE
NINE_HEAD_OVER_UPDATE_SUPPORTED = INSUFFICIENT_EVIDENCE
NODE_LEVEL_CREDIT_PROBLEM_SUPPORTED = INSUFFICIENT_EVIDENCE
```

**优先方向：** 在后续受控诊断中保存可重放的 Macro 决策快照，先完成真实联合动作及逐节点撤销的收益归因，再决定是否修改策略机制。目前证据不支持调整9个Cache Head或其Credit Assignment。

## Run 与目标Episode核对

数据来自 `runs/stageB3_14h_fresh_macro_actor_lr_1e_minus_3/`。Manifest和summary记录 Macro Actor学习率 `0.001`、Architecture 33、Checkpoint 60、Training Episode 20001–20300、每4个Episode一组；summary记录300个Episode和75次完整Update。`checkpoints/latest_checkpoint.pt`存在。指标文件行数分别为 Slot 90,000、Episode 300、Update 75。

目标Seed与保存的Episode记录对应如下。以下是现有Episode级汇总，只用于确认所选训练轨迹确有Cache变化，不是反事实收益：

| Training Seed | Episode | Macro帧数 | 至少一个节点实际改变配置的帧比例 | 9节点实际更新计数之和 | 平均KEEP概率 | 平均Holding Slot |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20001 | 1 | 148 | 1.0000 | 673 | 0.49994 | 2.027 |
| 20101 | 101 | 137 | 0.9927 | 441 | 0.64364 | 2.190 |
| 20201 | 201 | 131 | 0.9847 | 342 | 0.71691 | 2.290 |

实际更新计数之和是跨Macro帧的节点配置变化计数，不能解释为收益或负边际贡献。Slot指标中相应字段为每Slot的更新节点数，不能恢复更新前后9个节点的Cache内容。

## 可用记录与缺口

- `slot_metrics.jsonl`每Slot记录效用、Cache成本、UAV移动成本、任务分配计数、KKT调用数以及 `actual_cache_update_count` 和 UAV move/stay 标志。没有节点级Cache前态/后态、Cache动作及概率、Macro帧边界、Holding时长或UAV坐标轨迹。
- `episode_metrics.jsonl`有逐Macro的 `macro_candidate_state_diagnostics`。每项保存 `macro_index`、9节点候选ID列表，以及跨节点的平均KEEP/UPDATE概率等聚合量；没有Macro起始Slot、各节点各自的KEEP/UPDATE概率、候选ID对应的具体Cache配置或决策前Cache快照。另有按节点汇总到整个Episode的诊断，不能代替目标帧概率。
- Run目录没有Rollout或Macro trace文件。源码中的 `MacroTransition` 在内存中有动作索引、候选特征/Mask、实际持续时长和按节点变化标志，但这些对象未随目标Episode持久化。可选的Macro transition trace也不是Cache反事实快照：其保存字段不含实际Cache动作及具体Cache配置。
- 因缺少前态、具体候选配置和逐帧动作概率，无法验证每个Cᵢ分支确实只撤销一个实际UPDATE；因缺少原始UAV坐标/轨迹，也无法验证分支使用同一条物理轨迹。对这些分支运行Matching/KKT会违反同状态配对要求。

## 指标判定

| 指标 | 判定 | 依据 |
| --- | --- | --- |
| 联合动作正收益/负收益比例及平均ΔReturn | `INSUFFICIENT_EVIDENCE` | 未能对任何目标Macro帧完成严格配对的A/B执行。 |
| 负边际节点比例 | `INSUFFICIENT_EVIDENCE` | 未能在同一A联合动作上逐节点生成并核验Cᵢ分支。 |
| 联合动作交互证据 | `INSUFFICIENT_EVIDENCE` | 没有可比较的联合收益与单节点撤销收益。不能把各节点Episode汇总更新次数相加当作边际贡献。 |
| 九节点过度更新 | `INSUFFICIENT_EVIDENCE` | 三个目标Episode的实际Cache配置变化频率很高（98.47%–100% Macro帧至少一个节点改变），但没有这些动作相对ALL_KEEP的净收益，无法按任务标准支持“过度”。 |
| 节点级Credit问题 | `INSUFFICIENT_EVIDENCE` | 缺少逐节点动作概率与同状态边际收益配对。共享概率/Advantage统计本身不能证明归因缺陷。 |

`ΔReturn` 应按 `ΔTask Utility − ΔCache Cost − ΔUAV Movement Cost` 从配对分支计算。当前Run的Episode/Slot总量没有提供该配对，因此没有可报告的数值。所有被要求的反事实分支均未执行。

## 后续最小证据需求

后续只需对有限抽样的Macro帧保存一份可重放记录：Episode/Seed、Macro起始Slot、决策前9节点Cache配置、各节点采样动作及其校正组概率和候选条件概率、候选ID到具体配置的映射、实际Holding时长、UAV决策前位置及Holding窗口内实际轨迹，以及可恢复该窗口任务/车辆/预测输入的索引或状态快照。这样才能从隔离克隆状态运行A、B、Cᵢ，并分别核验Cache变化、容量约束及相同外生输入。

本轮未修改源码、配置、Checkpoint或Run文件；未启动训练、Validation或Test；未生成临时脚本或缓存。唯一新增产物为本报告。
