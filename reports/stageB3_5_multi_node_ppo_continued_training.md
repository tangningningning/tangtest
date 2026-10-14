# Stage B3-5：9节点联合Cache PPO受控续训与学习行为诊断

## 结论摘要

从B3-4A真实训练后Checkpoint恢复成功，动态SV车辆ID变化没有破坏9节点策略结构合同。新增Episode 20014–20021全部完成，执行了两个正式PPO更新组；组后Checkpoint均已保存，最终Checkpoint由正式Loader完整重载。

两组联合PPO数值均有限，KL保护均接受4/4个Macro Actor Step；Cache Candidate Scorer、候选编码器和共享Observation Encoder在每组都发生参数变化。更新行为从平均每个Macro更新4.463个节点变为4.293个，Cache成本按组平均略降，但各组使用不同Training Episode，不能将Return或成本变化单独解释为策略改善。

固定候选探针没有显示出对已知长期净收益的可靠识别：正收益候选20007-A的概率和排名继续下降；20008/20009的负收益A变化较小；三集负收益B都进一步降权，但没有候选A/B成为确定性选择。该结果不证明PPO实现有错误，也不证明策略已经学会长期Cache价值。

## 执行来源与Resume合同

- 运行使用B3-4A已验证的 `C:\software\anaconda3\python.exe`（Python 3.12.7），Conda `adaptive_cache_mec` 环境此前无法在本机激活。通过当前进程内禁用Numba导入使用项目纯Python KKT fallback；未安装或更改依赖。Matplotlib与NumPy二进制不兼容的可选绘图错误按既有局部包装器跳过，Checkpoint仍在绘图前保存。
- 正式配置：`configs/proposed/system_config_multi_node_factorized.toml`。原始SHA-256为 `a86d87ca5d586e89e367aee393b7a1cd9b27813f41ac3c5a19005bfa40a9ea6c`，语义SHA-256为 `bfb1f50b55e98d7b55df8190a56311de5ae9619e63d6ac14c44997e16548281d`。Training Pool manifest SHA-256仍为 `c9c401dfe6c36b11faecfa080b86cb878d325c2799ee7e54435ad85365992d93`。
- 来源Checkpoint为Architecture 33、Checkpoint 60、多节点v4 Schema；保存时已完成4个Episode、1个更新组，并含真实Adam状态。原始Checkpoint SHA-256为 `044a4490d8dfe21313752cf47751e2f855bcddb7e413418dbdbed1054ac0d5f3`；续训后校验其哈希未变化。
- 首次恢复上下文的SV ID为 `39, 21, 53, 56, 37`，来源Checkpoint记录的是上一Episode的 `44, 30, 18, 20, 50`。为满足Episode随机化，`src/ppo_simple/trainer.py` 的正式Loader现在只将前五个SV身份视为动态Episode数据；仍严格检查九个唯一节点、5-SV/2-RSU/2-UAV角色顺序及固定后缀 `RSU_0, RSU_1, UAV_0, UAV_1`。Architecture、Checkpoint版本、Schema、张量维度、配置/数据哈希、候选语义、策略参数和优化器状态等其余合同仍必须精确相同。
- 续训目标组数从来源Checkpoint的1增加到2、再到3。Loader只在保存与当前合同的Actor/Optimizer学习率日程均为常数、Macro Actor初始与最终学习率相同且只向上延长计划组数时允许此运行期限变化；本实验学习率始终为 `5e-4`，实际PPO参数与更新机制没有改变。
- Runner的 `target_episodes` 是**累计目标数**。因此分两次正式续训：目标8集完成20014–20017，保存/验证后再将累计目标设为12集完成20018–20021。四个已完成Episode没有重复采样。B3-5使用独立目录；为正式Resume对账，将B3-4A的起点Checkpoint和前4集正式指标复制到新目录，B3-4A原目录及Checkpoint均未修改。

## 逐Episode结果

每集300个正式Training Slot；每个Slot运行正式Matching/KKT。表中Edge数为SV、RSU、UAV三类执行任务数之和。

| Training Episode | 任务数 | Task Utility Q | Return | Cache成本 | UAV移动成本 | 平均任务时延(s) | 车辆总能耗(J) | Edge任务 | Cloud任务 | 平均Holding(Slot) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20014 | 4,137 | 3,933.197 | 3,843.762 | 88.691 | 0.743 | 2.355 | 99.111 | 854 | 3,283 | 1.596 |
| 20015 | 4,181 | 3,968.365 | 3,882.595 | 85.039 | 0.731 | 2.444 | 95.209 | 810 | 3,371 | 1.648 |
| 20016 | 4,236 | 4,038.448 | 3,950.375 | 87.352 | 0.721 | 2.231 | 95.273 | 852 | 3,384 | 1.648 |
| 20017 | 4,193 | 3,981.544 | 3,905.143 | 75.596 | 0.805 | 2.421 | 96.666 | 867 | 3,326 | 1.863 |
| 20018 | 4,251 | 4,039.798 | 3,965.769 | 73.271 | 0.757 | 2.381 | 99.240 | 830 | 3,421 | 1.840 |
| 20019 | 4,217 | 4,012.092 | 3,941.453 | 69.861 | 0.778 | 2.339 | 89.407 | 718 | 3,499 | 1.875 |
| 20020 | 4,245 | 3,999.897 | 3,919.623 | 79.544 | 0.729 | 2.776 | 109.241 | 858 | 3,387 | 1.657 |
| 20021 | 4,264 | 4,070.585 | 3,994.161 | 75.709 | 0.715 | 2.161 | 99.026 | 823 | 3,441 | 1.818 |

Return遵循正式 `Q − Cache成本 − UAV移动成本`。这些Episode的任务与轨迹不同，不能把20014–20021的Return升降当成策略单调改善或退化。每个Episode的动态SV ID、逐节点KEEP/UPDATE概率、成本、任务分配和Holding分布均保存在B3-5的 `episode_metrics.jsonl`。

两个新增组共完成2,400次Matching和402,823次KKT评价（第一组1,200次Matching/195,491次KKT；第二组1,200次Matching/207,332次KKT）。两组Runner均正常返回且无提前停止。

## 多节点Cache行为

| 更新组 | Macro帧数 | 平均每Macro实际更新节点数 | 0至9个节点同时更新的帧数（按0,1,…,9顺序） | SV平均KEEP / 实际UPDATE | RSU平均KEEP / 实际UPDATE | UAV平均KEEP / 实际UPDATE | 加权Holding均值 | 每Episode平均Cache成本 | SV / RSU / UAV累计Cache成本 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 20014–20017 | 713 | 4.463 | 3, 7, 48, 129, 186, 163, 118, 50, 6, 3 | 51.22% / 49.42% | 51.59% / 49.93% | 51.64% / 49.65% | 1.683 | 84.170 | 152.246 / 76.487 / 107.945 |
| 20018–20021 | 669 | 4.293 | 1, 9, 59, 123, 178, 175, 87, 29, 8, 0 | 52.80% / 48.19% | 53.05% / 46.94% | 53.04% / 47.23% | 1.794 | 74.596 | 137.029 / 66.194 / 95.163 |

每组都仍有大量同Macro多节点更新；平均更新节点数仅下降0.170，未出现更新坍缩。平均Cache成本由84.170降至74.596，但Episode不同，且SV身份也会变化；只能说这两组样本的实测成本更低，不能据此断言策略学会了降低成本。动态SV逐节点频率按每个Episode的实际 `action_spec.cache_node_ids` 顺序记录，没有跨Episode硬编码车辆ID。

## PPO更新与Critic诊断

| 更新组 | Macro Actor步骤 | Macro Critic / Long-Value步骤 | Micro Actor / Critic步骤 | Macro Actor Loss | Macro Critic Value Loss | Macro Advantage均值±标准差 | N2解释方差（更新前→后） | Long-Value解释方差（更新前→后） | Joint KL（更新前→后） | Joint Clip Fraction | Cache KL / Clip Fraction | KL保护 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20014–20017 | 4/4接受 | 50 / 50 | 40 / 40 | -0.004792 | 413.132 | 249.116 ± 48.071 | 0.0560 → 0.1043 | -0.000079 → 0.000078 | 0.001813 → 0.004229 | 1.44% | 0.001132 / 0.32% | 接受 |
| 20018–20021 | 4/4接受 | 50 / 50 | 40 / 40 | -0.003797 | 426.923 | 250.848 ± 55.899 | 0.1237 → 0.2340 | -0.0159 → -0.0348 | 0.002669 → 0.003645 | 2.73% | 0.000541 / 0.00% | 接受 |

两个更新组中的Ratio、KL、Clip、优势、Loss、优化器状态及Checkpoint参数均为有限值。现有正式指标没有保存Macro Joint Ratio的算术均值；保存了按Holding时长分组的 `mean |log(ratio)|`。D=1…5对应第一组为 `0.100, 0.080, 0.122, 0.206, 0.295`，第二组为 `0.065, 0.048, 0.058, 0.110, 0.197`，该值是log-ratio偏离量，不冒充Ratio均值。

Macro/N2 Critic在两组更新内部的解释方差均有所提高，但仍分别只有0.104和0.234；Long-Value Critic的解释方差接近零或为负，没有显示改善。因此只能说Macro Critic出现初步拟合信号，不能说价值估计已经可靠。Cache熵按9个节点的Categorical熵求和，组均值约18.736和18.481；节点熵及角色熵均保存在更新记录中，该总和不能与旧单节点熵直接比较。

参数张量比较确认：第一组相对B3-4A来源Checkpoint，Cache Scorer 4/4、候选编码器2/2、Macro共享Observation Encoder 4/4张量变化；第二组相对第一组Checkpoint，这三者仍分别为4/4、2/2、4/4张量变化。Macro终止策略、Critic及Micro UAV策略也各自有实际参数变化。累计计数为Macro Actor 12步、Macro Critic 150步、Long-Value 150步、Micro Actor/Critic各120步；包括B3-4A原有更新组。优化器Adam状态随Checkpoint恢复并递增。

## 固定候选探针

在20007、20008、20009分别用相同Slot 0 Observation、候选特征和Mask评估来源Checkpoint、第一组Checkpoint、第二组Checkpoint。两次探针重建的Observation/候选特征/Mask SHA-256逐Episode完全相同。候选按节点ID和完整目标Cache行重新匹配；下表索引仅用于核对本次生成结果，不复用历史候选索引。概率为候选在该节点UPDATE条件组内的概率；联合概率为该节点选候选且其他8节点全KEEP的概率。

| Episode / 候选 | 节点与目标Cache行 | UPDATE条件概率（来源→组1后→组2后） | UPDATE组内排名 | 原始Scorer分数 | 指定候选+其他8节点KEEP联合概率 | 确定性策略命中 |
|---|---|---|---|---|---|---|
| 20007 A（正收益参照） | UAV_0 → {1,4,9} | 2.931% → 2.896% → 2.853% | 19 → 23 → 27 | -0.13250 → -0.15570 → -0.15952 | 6.580e-5 → 8.089e-5 → 9.224e-5 | 否 |
| 20007 B（负收益参照） | RSU_1 → {2} | 1.809% → 1.794% → 1.773% | 25 → 27 → 35 | -0.07332 → -0.08004 → -0.10605 | 4.238e-5 → 5.297e-5 → 5.908e-5 | 否 |
| 20008 A（负收益参照） | SV 23 → {9} | 9.878% → 9.606% → 9.629% | 8 → 8 → 10 | -0.05791 → -0.07538 → -0.05403 | 2.339e-4 → 2.887e-4 → 3.373e-4 | 否 |
| 20008 B（负收益参照） | RSU_1 → {2} | 1.809% → 1.794% → 1.773% | 25 → 27 → 35 | -0.04389 → -0.04906 → -0.07434 | 4.235e-5 → 5.293e-5 → 5.901e-5 | 否 |
| 20009 A（负收益参照） | SV 17 → {4,5} | 9.865% → 9.718% → 9.541% | 8 → 10 → 9 | -0.13921 → -0.20558 → -0.24202 | 2.150e-4 → 2.528e-4 → 2.787e-4 | 否 |
| 20009 B（负收益参照） | RSU_1 → {2} | 1.809% → 1.794% → 1.773% | 25 → 27 → 35 | -0.04098 → -0.04637 → -0.07165 | 4.236e-5 → 5.294e-5 → 5.902e-5 | 否 |

三集全9节点KEEP联合概率均从约0.00238升至组1后的约0.00314、组2后的约0.00374。固定探针上，SV/RSU/UAV平均KEEP概率分别为：来源 `51.06% / 50.90% / 51.48%`，组1后 `52.79% / 52.24% / 53.03%`，组2后 `53.89% / 53.41% / 53.78%`。这是向KEEP偏移的迹象，但并不表示已能正确判断UPDATE长期价值。

依据B2-8离线固定动作标签，20007-A为已测正收益参照，其他A/B为负收益参照。正收益20007-A概率下降且排名变差；负收益A仅小幅变化；负收益B在三集均降权，但这种变化没有区分20007-A与亏损候选，且所有候选均未成为确定性动作。因此当前结论是**未观察到可靠的候选长期价值识别**，不是信用分配Bug的证据，也不是PPO收敛结论。

## Checkpoint与产物

- 第一组Checkpoint：`runs/stageB3_5_multi_node_resume/checkpoints/checkpoint_after_group_1_ep20017.pt`，正式Loader以Episode 20017上下文重载通过；SHA-256 `f3e63c795956334fecfa55b81d1480e71be6a7042b1080ac73be8e5df6d99780`。
- 第二组及最新Checkpoint：`runs/stageB3_5_multi_node_resume/checkpoints/checkpoint_after_group_2_ep20021.pt` 和 `latest_checkpoint.pt`，正式Loader以Episode 20021上下文重载通过；两者SHA-256均为 `26cc5ecacab188f4ae77bf402727c7612ba80ccad5e69681d024b9b596b8d2b4`。合同为Architecture 33、Checkpoint 60、多节点v4；训练计数为12集/3组、3,600次环境执行、3,600次Matching和534,858次KKT。
- 独立Run目录 `runs/stageB3_5_multi_node_resume/` 保留配置快照、Manifest、逐Episode/更新组/Slot指标、训练后诊断JSON以及两组正式Checkpoint。B3-4A来源Run和Checkpoint未覆盖。

两组更新都具备继续受控扩展训练的**运行条件**：Resume、Matching/KKT、参数更新及Checkpoint往返均通过，联合KL/裁剪正常。建议后续仍用分组Checkpoint门控；当前两组数据不支持“已学会Cache候选长期价值”或“已收敛”的结论。

```ini
STAGE_B3_5 = COMPLETE
RESUME_CHECKPOINT = PASS
DYNAMIC_SV_RESUME = PASS
TRAINING_EPISODES_ADDED = 20014, 20015, 20016, 20017, 20018, 20019, 20020, 20021
TOTAL_TRAINING_EPISODES = 12
PPO_UPDATE_GROUPS_ADDED = 2
TOTAL_PPO_UPDATE_GROUPS = 3
MACRO_ACTOR_STEPS = 12 cumulative; 4 accepted in each added group
CACHE_SCORER_UPDATED = YES
ENCODER_UPDATED = YES
JOINT_KL_STATUS = FINITE; pre=0.001813/0.002669; post=0.004229/0.003645; both groups accepted
JOINT_CLIP_STATUS = FINITE; 0.0144/0.0273
MEAN_CACHE_UPDATES_PER_MACRO = 4.463 -> 4.293
CACHE_COST_TREND = mean per Episode 84.170 -> 74.596; cross-Episode difference, not causal evidence
CRITIC_EXPLAINED_VARIANCE_TREND = N2 0.0560->0.1043, then 0.1237->0.2340; Long-Value remains near-zero/negative
CANDIDATE_VALUE_RECOGNITION = NO
CHECKPOINT_SAVE_RELOAD = PASS
READY_FOR_LONGER_TRAINING = YES (controlled continuation only; not convergence evidence)
SYSTEM_MODEL_CHANGED = NO
PPO_ALGORITHM_CHANGED = NO
```
