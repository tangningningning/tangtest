# Stage3 解耦 MAPPO 有界训练与完整配对评价

日期：2026-10-18（Asia/Shanghai）  
结论范围：从 Stage3 Update 16 开始的 8 次更新，以及 3 个既有 Validation Episode 的完整配对结果。Validation 场景曾用于历史评价，结果属于小样本探索性筛选，不构成独立确认或泛化结论。

## 冻结输入与执行

- **起点：** Stage3 Update 16，`runs/research_mappo_stage3_main_experiment/training/stage3_20261018T095817Z_1b3d4652/checkpoints/checkpoint_update_0016.pt`，SHA-256 `ca819d60320b3c65a46c2eb8217ca6d9e18f5a4bda2799f1778988f3976e611f`。Runner 记录了源 checkpoint 与解耦策略迁移合同；Actor/优化器兼容迁移被执行，训练由该策略继续，不使用 Update 17 smoke checkpoint。
- **损失 Profile：** `configs/research/mappo_stage3_loss_coefficients_v1.json`，SHA-256 `c91bccc6bf91f76c407a18c9953b4fa2c2d346fcfef3337ad63ee49b23d8bdb4`，与冻结值一致。
- **运行目录：** `runs/research_mappo_stage3_decoupled_controlled_evaluation/training/stage3_20261018T124359Z_ff9ab9d0/`。完成 8 次更新（Update 17–24），每次 40 Slot，共 320 Slot；训练 seed 为 20044–20051，按 Training Pool 固定顺序取用。该段未与先前记录的 Stage3 seeds 20007–20043 重叠。Training Pool Manifest SHA-256 为 `b6cae236b029f4d24c7488a91f0825247a20f475ec76b2d80737de44ae491bd3`；运行配置 SHA-256 为 `03328b88658a6f15c0518eee9f4903d5647f12a800432e58c74b42fdef51d80c`。数据和 Manifest 未改动。
- **Checkpoint：** Update 20 SHA-256 `9b5e871e7c2ed17363d8cce0acf35a88a02c96bbc6a0f4c9f05375e6f9b4a527`；Update 24 SHA-256 `41e979bf193e13cd4f20f272dae5dbb77b86936ce0e61db56345ccc82fa87650`。两个文件均保存在本轮独立运行目录，未覆盖历史结果。
- **Validation 产物：** `runs/research_mappo_stage3_decoupled_controlled_evaluation/validation_update24/validation_results.json` 及 `evaluation_metadata.json`。Update 13 Checkpoint SHA-256 为 `aff1ed777c1100c9da69b86b079c0bcaa83e2bcdff1885769c9761e4858cd044`；本轮适配器 SHA-256 为 `b18b75bcb46383be96cc1d87c4b55187723175817fddee0f7bc9c6818e3a0f8f`，调用的 Stage1C evaluator SHA-256 见下文。
- **执行边界：** Stage3 训练开启，训练完成记录为 8/8 且无失败；未调用 Test Pool、未改正式配置、模型、奖励或 Profile。

## 训练诊断

### 固定状态上的 Candidate 与 Trigger

Update 16、20、24 使用同一批 8 个独立诊断 Macro 状态、同一组 7,079 个候选对及同一批冻结 ΔQ 标签进行比较。标签来自 Update 16 冻结 Critic 与既定有限期 Macro SMDP 回报加 bootstrap 定义；它们是有限期条件价值估计，不是长期最优价值，也不是各新 Checkpoint 的在线 Oracle。

| Checkpoint | Pairwise 排序准确率 | 最优候选条件概率 | 概率贪心候选 ΔQ | 平均 regret | 条件期望 ΔQ | 平均 P(UPDATE) | 相对 U16 最大 Trigger 漂移 | 确定性 Trigger 翻转 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Update 16 | 0.52364 | 0.04140 | -0.20851 | 0.27075 | -0.20609 | 0.51716 | 0 | 0/8 |
| Update 20 | 0.49998 | 0.04132 | -0.19763 | 0.25986 | -0.20612 | 0.52036 | 0.00511 | 0/8 |
| Update 24 | 0.50779 | 0.04125 | -0.17767 | 0.23991 | -0.20600 | 0.52041 | 0.00615 | 0/8 |

Candidate 信号是 **MIXED**：概率贪心候选的 ΔQ 和 regret 有改善，但 pairwise 准确率低于 Update 16，最优候选概率略降，概率加权期望 ΔQ 基本不变。Update 24 未呈现一致的候选排序改善。固定样本上的 Trigger 最大漂移为 0.00615，8 个样本没有确定性决策翻转；这只支持“该样本上漂移受控”，不代表全状态空间 Trigger 概率不变。

### 每次更新日志摘要

| 指标 | Update 17–24 结果 |
|---|---:|
| 每次 Candidate ranking loss | 0.69119–0.69491（始终接近 0.693） |
| Candidate 专项梯度范数 | 0.000628–0.000821；8 次均有更新 |
| Candidate 专项参数变化 L2 | 0.01808–0.02512 |
| ΔQ Huber loss | 0.02265–0.06385 |
| Trigger ranking loss | 0.65927–0.68669 |
| Macro approx KL / clip fraction | KL 1.97×10⁻⁵–5.18×10⁻⁵；clip fraction 每次为 0 |
| Micro approx KL / clip fraction | KL 1.34×10⁻⁸–8.42×10⁻⁶；clip fraction 每次为 0 |
| Cache UPDATE 率 | 0.4500–0.5444，均值 0.4951 |
| UAV MOVE 率 | 0.3750–0.5250，均值 0.4406 |
| 反事实标签 | 8 个有效标签/更新，合计 64；无效 trace/label 拒绝数均为 0 |
| 反事实回放 | 1,568 个分支；每更新 80–314 个合法 UPDATE 候选；合计 267.187 秒 |
| Slot 收集 / PPO 与辅助更新 | 收集合计 72.185 秒；更新时间合计 272.864 秒（反事实回放包含在更新时间中） |
| 数值及合同 | NaN/Inf 数值异常 0；动作/奖励合同错误 0 |

PPO 的 Macro/Micro Actor、Critic 和 Candidate 参数均发生有限更新。Clip fraction 为 0 与本轮小 KL/ratio 变化相符，但不能单独解释为学习有效。Candidate loss 接近随机排序量级，且固定诊断排序指标没有一致改善。累计计时仅为日志记录的 rollout 收集和 update 时间，不包括启动、Checkpoint 序列化等外围时间。

## 完整配对 Validation

冻结 Episode 为 seed **30001、30002、30003**，每个 300 Slot。新运行在相同外生场景上评价 Update 24、Update 13 和 KEEP+STAY；Update 16 使用已有完整评价结果。历史 Update 16 结果的 Episode、配置、奖励、确定性解码合同和 Stage1C evaluator SHA（`e0ce198871544723a0627069771878a121c3bda7ace29e228016c0b2383d5dd2`）与本轮一致。当前复评的 Update 13 和 KEEP+STAY 净效用逐场景与历史值相同（浮点尾数级差异），用于交叉核对。Validation Pool Manifest SHA-256：`a75f4935f598a2945f66547ab48ce5a24a86624bb142a9349787797c402da10`。动作解码使用既有确定性规则：先贪心选合法操作组，再在该组内贪心选合法候选；KEEP+STAY 强制合法 action 0。未读取 Test。

### 逐 Episode 指标

Cache 变化率分母为 1,350 次节点决策；UAV 位移率分母为 600 次决策。下表运行时间为该策略完整 Episode rollout 秒数。

| Seed | 策略 | 净效用 | 任务效用 | Cache 成本 | UAV 成本 | Cache 实际变更 / 率 | UAV 实际位移 / 率 | Rollout 秒 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 30001 | Update 24 | 3875.1081 | 3940.5389 | 65.4308 | 0 | 696 / 51.56% | 0 / 0% | 43.964 |
| 30001 | Update 16 | 3905.2163 | 3941.4372 | 36.2209 | 0 | 314 / 23.26% | 0 / 0% | 41.038 |
| 30001 | Update 13 | 3931.2945 | 3948.2404 | 16.9460 | 0 | 156 / 11.56% | 0 / 0% | 54.207 |
| 30001 | KEEP+STAY | 3976.1384 | 3976.1384 | 0 | 0 | 0 / 0% | 0 / 0% | 39.913 |
| 30002 | Update 24 | 3795.5856 | 3864.3449 | 68.7593 | 0 | 722 / 53.48% | 0 / 0% | 58.709 |
| 30002 | Update 16 | 3827.3487 | 3863.7862 | 36.4375 | 0 | 316 / 23.41% | 0 / 0% | 49.872 |
| 30002 | Update 13 | 3852.5230 | 3869.4689 | 16.9460 | 0 | 156 / 11.56% | 0 / 0% | 60.522 |
| 30002 | KEEP+STAY | 3889.2103 | 3889.2103 | 0 | 0 | 0 / 0% | 0 / 0% | 47.762 |
| 30003 | Update 24 | 3842.4735 | 3911.4468 | 68.9733 | 0 | 722 / 53.48% | 0 / 0% | 44.700 |
| 30003 | Update 16 | 3881.4224 | 3919.5933 | 38.1710 | 0 | 332 / 24.59% | 0 / 0% | 47.460 |
| 30003 | Update 13 | 3907.8003 | 3924.7463 | 16.9460 | 0 | 156 / 11.56% | 0 / 0% | 47.827 |
| 30003 | KEEP+STAY | 3942.8919 | 3942.8919 | 0 | 0 | 0 / 0% | 0 / 0% | 43.335 |

### 配对净效用差

均值和标准差使用三个场景的配对差；标准差为描述性总体标准差（除以 n），仅供小样本诊断。

| Validation seed | Update 24 − Update 16 | Update 24 − Update 13 | Update 24 − KEEP+STAY |
|---:|---:|---:|---:|
| 30001 | -30.1083 | -56.1864 | -101.0303 |
| 30002 | -31.7631 | -56.9373 | -93.6246 |
| 30003 | -38.9489 | -65.3269 | -100.4184 |
| **均值 ± 标准差** | **-33.6067 ± 3.8374** | **-59.4835 ± 4.1432** | **-98.3578 ± 3.3562** |

Update 24 在三个场景全部低于 Update 16、Update 13 和 KEEP+STAY，净效用没有改善。相对 Update 16，平均任务效用低 2.8287，平均 Cache 成本高 30.7780；两者 UAV 成本均为 0。相对 Update 13，任务效用低 8.7084、Cache 成本高 50.7752；相对 KEEP+STAY，任务效用低 30.6366，Update 24 另产生平均 67.7211 的 Cache 成本。故退化主要伴随 Cache 配置成本大幅增加，不能把全部差异归因于 Critic。

当前评价 JSON 记录完整 300 Slot Episode、活跃任务槽计数（seed 30001/30002/30003 分别为 4,302/4,196/4,255）、任务效用、奖励合同、数值异常和约束违规，但没有单独的已完成任务数/完成率字段；不能把 `complete_episode=true` 误读为所有任务完成。所有 12 个逐策略 Episode 的 `complete_episode=true`、数值异常为 0、约束违规为 0、奖励合同最大绝对误差为 0，Cache 动作合法；UPDATE 均对应真实 Cache 变化。UAV 实际位移与 MOVE 均为 0。

## 结论

- 解耦后的 Candidate 专项更新确实参与 8 次联合训练，但共享固定标签上的排序准确率和最优候选概率未改善；其他价值差指标只有有限、混合变化。
- Update 24 完整 Episode 净效用相对三个对照在所有场景均下降。Cache 变更率升至约 52.84%，而 Update 16 约为 23.75%、Update 13 约为 11.56%。训练及数值合同通过不能替代性能有效性。
- 本轮是 3 个历史 Validation 场景的探索性筛选，不具统计显著性或泛化结论。按预设规则不建议进入下一阶段，不自动追加训练。

```makefile
TRAINING_UPDATES_COMPLETED = 8
TRAINING_SLOTS_COMPLETED = 320
CANDIDATE_RANKING_IMPROVED = MIXED
TRIGGER_DRIFT_CONTROLLED = YES (fixed diagnostic states only; max_abs=0.00615, boundary_flips=0/8)
FULL_EPISODES_EVALUATED = 3
MEAN_NET_UTILITY_DIFF_VS_UPDATE16 = -33.6067
MEAN_NET_UTILITY_DIFF_VS_UPDATE13 = -59.4835
MEAN_NET_UTILITY_DIFF_VS_KEEP_STAY = -98.3578
READY_FOR_NEXT_STAGE = NO
```
