# B3-15H-R：Sparse 与 Legacy 确定性评价差异定位

## 评价范围

- Legacy：`runs/b3_15h_legacy_factorized_48ep/checkpoints/latest_checkpoint.pt`，Architecture 33 / Checkpoint 60，48 Episode / 12 Update，Macro Actor LR `5e-4`。
- Sparse：`runs/b3_15h_sparse_coordinated_48ep/checkpoints/latest_checkpoint.pt`，Architecture 34 / Checkpoint 61，48 Episode / 12 Update，Macro Actor LR `5e-4`。
- 确定性部分只读取 B3-15H 已有的六个 Training 诊断 Episode 结果，不重跑确定性评价。
- Stochastic 部分使用 Training Pool 的 Episode `20231`、`20232`，策略采样 Seed `31501`、`31502`。每个策略、每个环境 Seed、每个策略 Seed各运行一次，共 8 个完整冻结 Episode。配对运行使用相同的外生 Episode 和策略 Seed。
- Evaluator 在 `torch.inference_mode()` 中执行，并核对策略参数及优化器计数没有改变。所有 8 条记录均验证 `Return = Task Utility − Cache Cost − UAV Cost`，最大绝对误差为 `1.82e-12`。
- 为取得逐Macro的概率、采样动作、真实缓存改动节点数和UAV实际位移，本次只在进程内包裹了Evaluator的采样与汇总函数，未保存或修改源码。Evaluator的单Episode汇总函数不接受单个环境计算置信区间；本次保留每条原始Episode记录，单样本标准差与置信区间字段为空，不据此作统计显著性判断。

## 确定性动作选择核对

### Legacy：九个独立节点组

节点顺序为 `SV[0..4]`、`RSU_0`、`RSU_1`、`UAV_0`、`UAV_1`。确定性入口逐节点比较经过基数校正的 KEEP / UPDATE 组概率，再在 UPDATE 组内选择合法候选。已有记录中，选中组与节点概率 argmax 一致率为 100%；全部所选候选合法，概率有限，最大概率和误差为 `3.52e-7`。

| 节点 | Initial P(KEEP) | Initial P(UPDATE) | Initial 实际选 UPDATE | Update 12 P(KEEP) | Update 12 P(UPDATE) | Update 12 实际选 UPDATE |
|---|---:|---:|---:|---:|---:|---:|
| SV[0] | 0.5005 | 0.4995 | 0.46% | 0.6640 | 0.3360 | 0% |
| SV[1] | 0.5019 | 0.4981 | 0.46% | 0.6073 | 0.3927 | 0% |
| SV[2] | 0.5052 | 0.4948 | 0.46% | 0.6891 | 0.3109 | 0% |
| SV[3] | 0.5001 | 0.4999 | 0.92% | 0.6146 | 0.3854 | 0% |
| SV[4] | 0.5040 | 0.4960 | 0.46% | 0.5963 | 0.4037 | 0% |
| RSU_0 | 0.5049 | 0.4951 | 0% | 0.6527 | 0.3473 | 0% |
| RSU_1 | 0.5111 | 0.4889 | 0.46% | 0.5653 | 0.4347 | 0% |
| UAV_0 | 0.5044 | 0.4956 | 0% | 0.5950 | 0.4050 | 0% |
| UAV_1 | 0.5059 | 0.4941 | 0.46% | 0.6364 | 0.3636 | 0% |

Update 12 的九个 P(KEEP) 均高于各自 P(UPDATE)，所以逐节点确定性动作全部为 KEEP。Legacy Initial 的联合 P(ALL_KEEP) 均值约为 `0.00211`，确定性实际全KEEP帧率为 `99.08%`；Update 12 的联合概率约为 `0.01423`，确定性全KEEP帧率为 `100%`。这是逐头 argmax 的联合结果，不等于随机策略下联合抽中 ALL_KEEP 的概率。

### Sparse：ALL_KEEP 与九个 UPDATE 组

Sparse按 `ALL_KEEP` 加九个节点 UPDATE 组建模。下表是已有六个确定性评价中逐Macro概率的均值；UPDATE组索引按上述九节点顺序排列。

| 概率组 | Initial | Update 12 |
|---|---:|---:|
| ALL_KEEP | 0.1275 | 0.3753 |
| UPDATE SV[0] | 0.0979 | 0.0831 |
| UPDATE SV[1] | 0.0952 | 0.0788 |
| UPDATE SV[2] | 0.0989 | 0.0685 |
| UPDATE SV[3] | 0.0976 | 0.0590 |
| UPDATE SV[4] | 0.0983 | 0.0791 |
| UPDATE RSU_0 | 0.0943 | 0.0627 |
| UPDATE RSU_1 | 0.0958 | 0.0595 |
| UPDATE UAV_0 | 0.0973 | 0.0638 |
| UPDATE UAV_1 | 0.0972 | 0.0703 |
| 九个 UPDATE 组概率之和 | **0.8725** | **0.6247** |

在 Initial 的 1,608 帧和 Update 12 的 360 帧中，ALL_KEEP 都是**概率最大的单个组**，但它的概率从未超过九个 UPDATE 组概率之和。Evaluator/Policy 的确定性动作是最大单个完整动作（Sparse 代码比较 `P(ALL_KEEP)` 与每个 `P(UPDATE组) × P(候选|UPDATE组)`），并不把九个 UPDATE 组先合并成一个动作。因此即使 UPDATE 事件总概率质量更大，只要每个 UPDATE 组及其候选的联合概率都低于 ALL_KEEP，确定性结果仍是 ALL_KEEP。此处每个 UPDATE 组的组概率本身就低于 ALL_KEEP，条件候选概率不超过 1，足以说明该选择一致。

六个既有 Episode 中，Sparse 两个阶段均为 100% 的确定性 ALL_KEEP，真实 Cache 配置没有变化；`selected_candidate_index` 全部为空，因此“被选中的 UPDATE 条件候选概率”不适用。现有确定性摘要没有保存未选 UPDATE 候选的完整条件概率分布，不能补造该分布。Sparse 概率和误差最大为 `4.89e-8`（Initial）与 `3.84e-8`（Update 12）；非有限概率、组选择规则违反和UPDATE组内 argmax违反均为 0。

### Holding 与 UAV

| 策略 / 阶段 | Holding 均值 | P(TERMINATE)，elapsed 1–4 | 确定性终止选择 | UAV0 P(STAY/MOVE) / MOVE率 | UAV1 P(STAY/MOVE) / MOVE率 |
|---|---:|---|---|---|---|
| Legacy Initial | 1.417 | 0.511 / 0.495 / 0.495 / 0.494 | elapsed 1 多数终止，后续大多继续 | 0.521 / 0.479 / 0% | 0.524 / 0.476 / 0% |
| Legacy Update 12 | 1.000 | elapsed 1: 0.645 | 100% 在最小时长终止 | 0.370 / 0.630 / 100% | 0.462 / 0.538 / 100% |
| Sparse Initial | 1.130 | 0.514 / 0.497 / 0.497 / 0.496 | elapsed 1 的终止率 97.0% | 0.522 / 0.478 / 0% | 0.526 / 0.474 / 0% |
| Sparse Update 12 | 5.000 | 0.389 / 0.386 / 0.384 / 0.381 | 所有可选时点继续，到上限结束 | 0.544 / 0.456 / 0.56% | 0.566 / 0.434 / 0% |

所有已保存的终止决策都与对应的二元概率 argmax 一致。UAV确定性摘要保存了选择的候选组、索引和落点，但没有逐状态的完整 STAY/MOVE 概率向量；因此上表的概率是六个Episode的组概率均值，动作率来自Evaluator逐Slot选择记录。Rollout将所选候选落点直接交给 `apply_uav_candidate()`，没有发现Evaluator另行改写策略动作的路径。

## Stochastic 冻结评价

每条记录来自独立的完整 300 Slot Episode；收益、成本、Cache采样、Holding和位移按实际轨迹计算。Sparse每帧实际改动节点数上限为1；Legacy不受该稀疏限制。

| 策略 | Episode | 策略Seed | Return | Task Utility | Cache Cost | UAV Cost | P(ALL_KEEP)均值 | 实际ALL_KEEP帧率 | 实际改动节点数 | Holding | UAV0/UAV1 MOVE率 / 位移率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Legacy | 20231 | 31501 | 3608.472 | 3673.634 | 64.360 | 0.802 | 0.0182 | 2.05% | 620 | 1.538 | 62.0%/62.0% ; 50.7%/50.7% |
| Legacy | 20231 | 31502 | 3608.153 | 3670.456 | 61.453 | 0.850 | 0.0184 | 1.08% | 582 | 1.622 | 62.7%/62.7% ; 56.3%/56.3% |
| Legacy | 20232 | 31501 | 3521.569 | 3577.822 | 55.370 | 0.883 | 0.0184 | 0.57% | 538 | 1.714 | 66.7%/66.7% ; 56.0%/56.0% |
| Legacy | 20232 | 31502 | 3547.862 | 3605.222 | 56.509 | 0.851 | 0.0180 | 5.11% | 548 | 1.705 | 61.3%/61.3% ; 57.7%/57.7% |
| Sparse | 20231 | 31501 | 3646.627 | 3655.535 | 8.286 | 0.621 | 0.3788 | 37.76% | 89 | 2.098 | 44.0%/44.0% ; 44.0%/44.0% |
| Sparse | 20231 | 31502 | 3687.990 | 3696.016 | 7.347 | 0.679 | 0.4253 | 46.09% | 69 | 2.344 | 48.7%/48.7% ; 47.0%/47.0% |
| Sparse | 20232 | 31501 | 3586.917 | 3598.181 | 10.634 | 0.630 | 0.3767 | 34.93% | 95 | 2.055 | 49.3%/49.3% ; 39.7%/39.7% |
| Sparse | 20232 | 31502 | 3630.491 | 3639.160 | 8.014 | 0.654 | 0.4146 | 46.38% | 74 | 2.174 | 49.0%/49.0% ; 43.7%/43.7% |

Stochastic表格中UAV列按 `MOVE动作率 / 非零位移率` 排列，先UAV0、后UAV1。实际ALL_KEEP帧率的分母为Macro帧；实际改动节点数是整条Episode中缓存矩阵发生变化的节点计数。Sparse的实际改动节点/帧最大值为1；Legacy观测到的最大值为9。八条轨迹中，按MicroTransition所存候选落点相对STAY落点计算的非零位移率与MOVE动作率逐UAV一致。

四条严格配对的收益差（Sparse − Legacy）：

| Episode | 策略Seed | ΔReturn | ΔTask Utility | ΔCache Cost | ΔUAV Cost |
|---:|---:|---:|---:|---:|---:|
| 20231 | 31501 | +38.155 | −18.100 | −56.074 | −0.181 |
| 20231 | 31502 | +79.836 | +25.560 | −54.106 | −0.171 |
| 20232 | 31501 | +65.348 | +20.359 | −44.736 | −0.253 |
| 20232 | 31502 | +82.629 | +33.938 | −48.495 | −0.196 |
| **均值** |  | **+66.492** | **+15.440** | **−50.853** | **−0.200** |

描述性均值与样本标准差（4条轨迹，不视作独立大样本）：

| 策略 | Return 均值 ± SD | Task Utility | Cache Cost | UAV Cost | P(ALL_KEEP) | 实际ALL_KEEP帧率 | 实际更新节点 / Episode | Holding | UAV0 / UAV1 MOVE率 | UAV0 / UAV1 平均位移 (m/slot) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Legacy Update 12 | 3571.514 ± 43.826 | 3631.783 | 59.423 | 0.847 | 0.0182 | 2.20% | 572.0 | 1.645 | 63.17% / 55.17% | 7.419 / 6.069 |
| Sparse Update 12 | 3638.006 ± 41.790 | 3647.223 | 8.570 | 0.646 | 0.3989 | 41.29% | 81.75 | 2.168 | 47.75% / 43.58% | 5.250 / 4.981 |

两组 Stochastic 总 Holding 分布（8条Episode分别保存在原始汇总JSON中）：Legacy 731个Macro帧中，1–5 Slot分别占 62.5%、20.4%、10.5%、3.6%、3.0%；Sparse 555个Macro帧中分别占 44.0%、23.8%、14.1%、8.5%、9.7%。按Macro帧统计的实际缓存配置变化率均值为 Legacy `97.80%`、Sparse `58.71%`；同批轨迹中随机全KEEP动作均确实没有改变配置。

在这四条配对轨迹里，Sparse的净Return均高于Legacy；差距主要由Cache成本均值低 `50.853` 解释，另有Task Utility高 `15.440` 和UAV成本低 `0.200`。这只是两个Training Episode、每个两种策略采样Seed的初步对照，不能据此判定总体架构优劣。

## 结论与限制

- **确定性Evaluator没有发现实现Bug。** Legacy逐节点组选择、Sparse完整动作选择、Holding终止以及UAV所选候选的应用路径与记录一致。确定性与Stochastic行为相差很大，是概率分布的argmax执行与按分布采样的自然差异，不是同一个动作频率指标。
- **Sparse的确定性 ALL_KEEP 是MAP效果，不表示 `P(ALL_KEEP)` 超过所有 UPDATE 事件的合计概率。** Update 12的均值分别是 `0.375` 对 `0.625`；但ALL_KEEP是最大单组，且UPDATE的每个完整候选动作概率不可能大于其组概率，所以MAP仍选ALL_KEEP。Stochastic下实际ALL_KEEP帧率约41.3%，与平均组概率约39.9%接近。
- **Legacy也有确定性逐头argmax效应。** Update 12的逐节点P(KEEP)为0.565–0.689，九个头全部选择KEEP；但联合P(ALL_KEEP)均值只有约1.42%，Stochastic实际全KEEP帧率约2.20%。
- **行为对照。** Update 12确定性下Legacy的两个UAV每Slot都选择MOVE，Sparse几乎都选择STAY；Stochastic下Legacy MOVE率约63.2% / 55.2%，Sparse约47.8% / 43.6%。实际落点核对确认这些MOVE均产生非零位移。Sparse随机策略缓存更新节点更少，但仍会更新；确定性动作频率不能代表随机训练采样率。
- **Legacy确定性净收益上升主要伴随Task Utility上升。** B3-15H六个诊断Seed的Initial→Update 12平均变化为 Return `+73.465`、Task Utility `+74.558`、Cache Cost `−0.623`、UAV Cost `+1.716`。因此收益增量不是由少付缓存或UAV成本造成；但Cache/Holding和UAV策略同时变化，本数据不能把Utility增量因果归于UAV移动。
- **Sparse收益证据要分模式。** 确定性Initial→Update 12的六Seed平均Return仅 `+0.037`，没有清晰改善证据。Stochastic Update 12在本次四条轨迹中高于Legacy Update 12，但没有Sparse/Legacy的Stochastic Initial配对评价，不能据此声称Sparse从训练中学到了收益提升。
- **最小后续控制建议。** 若要判定Legacy的Deterministic Task Utility提升是否来自UAV轨迹，在同一组Training诊断Episode上固定一条预先指定的UAV轨迹，让Initial与Update 12 Cache/Holding策略在该轨迹下分别运行；本轮没有执行该控制。

## 结果字段

DETERMINISTIC_EVALUATOR_CONSISTENT = YES（记录与执行路径一致；逐状态UAV概率向量未持久化）

SPARSE_ALL_KEEP_ARGMAX_EFFECT = YES（ALL_KEEP为最大单个完整动作；不是最大事件总概率）

LEGACY_CACHE_UPDATE_DISTRIBUTION = Deterministic Update 12：九节点均选KEEP；Stochastic：均值572个实际缓存节点更新/Episode，ALL_KEEP帧率2.20%

SPARSE_CACHE_UPDATE_DISTRIBUTION = Deterministic Update 12：ALL_KEEP 100%，P(ALL_KEEP)=0.375，UPDATE总质量0.625；Stochastic：均值81.75个节点更新/Episode，最多1节点/帧，ALL_KEEP帧率41.29%

STOCHASTIC_LEGACY_RETURN = 3571.514 ± 43.826（4条冻结轨迹）

STOCHASTIC_SPARSE_RETURN = 3638.006 ± 41.790（4条冻结轨迹）

UAV_BEHAVIOR_EFFECT = 确定性动作频率极化显著；随机评价中的频率差距缩小，Sparse平均UAV成本低0.200/Episode

SPARSE_LEARNING_EVIDENCE = INSUFFICIENT_EVIDENCE（无随机Initial对照；确定性六Seed提升仅+0.037 Return）

LEGACY_LEARNING_EVIDENCE = 确定性六Seed平均提升+73.465 Return，其中Task Utility +74.558；Stochastic Initial学习增益未评估

READY_FOR_LONG_TRAINING = NO（架构收益排序依赖执行模式；需先按确定性主协议确认模型选择，且UAV固定轨迹控制仍未执行）

TRAINING_STARTED = NO

PPO_SOURCE_CHANGED = NO

CHECKPOINTS_CHANGED = NO

