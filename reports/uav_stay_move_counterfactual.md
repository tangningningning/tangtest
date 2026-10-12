# UAV STAY/MOVE 最小反事实收益验证

## 结论摘要

使用 LR `5e-4` 冻结 Checkpoint，对 seed 30020 / Slot 100 和 seed 30023 / Slot 200，分别比较原策略、强制UAV0移动一次、强制UAV1移动一次的完整剩余Episode后缀。两处基线都与正式Validation完全复现。每个UAV在对应干预状态都有18个合法非零位移候选；本实验执行的MOVE均为7.5米，候选源为 `geometry_exploration`。

- **MOVE有时优于STAY：**4个单UAV MOVE对照中有3个后缀Return略高，1个略低。Return差异分别为 `+0.773`（seed30020/UAV0）、`+0.462`（seed30020/UAV1）、`−0.124`（seed30023/UAV0）、`+0.142`（seed30023/UAV1）。
- 负差分支是seed30023/UAV0：Task Utility下降 `0.121913`，移动成本增加 `0.001870`；损失主要来自Task Utility，移动成本只占较小部分。
- MOVE改变了边缘节点间的少量任务分配，未改变这四个分支的Cloud任务总数。即使节点任务总数不变，位置变化仍会改变实际Task Utility、时延和能耗。
- **有合法MOVE但策略选择STAY。**干预点两架UAV的策略均选择STAY；其MOVE组概率约为13%–14%，STAY组概率约为86%–87%。这说明组概率偏好与已测MOVE的实际收益并不总是一致，但四个对照的收益差均小。
- 结果足以支持继续检查Micro动作组/候选的信用分配，**不足以支持现在修改Micro策略或判定STAY普遍有误**。结果只覆盖两个seed、两个状态、每个UAV每个状态一个组内最高评分MOVE候选。

## 执行协议与核验

- Checkpoint：`runs/formal_cache_economic_v2_lr_5e_minus_4/checkpoints/latest_checkpoint.pt`，SHA-256 `4ba388b2dfe87b02a50ae70282e9bf8f4223c0c4d56a9689cb3b49e68aba1318`；Checkpoint v58、架构v31、300 Episode / 75 Updates。配置原始哈希与语义哈希在结果JSON中保留。
- 只读取现有Validation seeds 30020、30023及其环境输入；未使用Test seed。Slot 100/200的动作前状态来自完整基线重放的深拷贝。
- A为冻结策略原动作；B只在干预Slot强制UAV0执行MOVE组内最高原始评分、合法且非零位移的候选，UAV1按新的UAV0条件继续原确定性规则；C保持UAV0原动作，只强制UAV1执行其条件候选中的最高评分合法MOVE。每个B/C只强制一次，干预后Cache、Holding和UAV决策都由同一冻结策略正常执行。
- 候选来自正式UAV候选枚举；先按配置的位置容差过滤非零位移，再按Micro候选原始Logit选组内最高项，同分按原候选顺序。最终双UAV位置每Slot均经过正式转移合法性校验。
- A基线Return、Cache动作路径和UAV动作路径与正式Validation逐项一致：seed 30020为 `3149.9538163485413`，seed 30023为 `3368.8282081635266`，Return差均为0。四个后缀分支共用同一逐Slot外生输入；Matching/KKT、资源分配和任务指标均在分支中重新执行。
- 总计执行1200个正式Slot环境步骤。运行后Checkpoint哈希、策略参数及优化器计数未变；Validation输入指纹未变。干预后累计Return定义为从干预Slot（含）至Slot 299的逐Slot系统效用之和。系统效用为任务效用减Cache成本与UAV移动成本（`src/decision/slot_environment.py:493`）。

## 干预点与合法MOVE候选

| Seed / Slot | UAV | 原策略组概率 STAY / MOVE | 合法非零MOVE数 | 最高评分候选 | 起点 → 目标位置 (m) | 位移 (m) | 候选原始评分 |
|---|---|---:|---:|---|---|---:|---:|
| 30020 / 100 | UAV0 | 86.76% / 13.24% | 18 | 索引3，`geometry_exploration` | (3364.030, 1119.880) → (3371.530, 1119.880) | 7.5 | −0.576850 |
| 30020 / 100 | UAV1 | 87.00% / 13.00% | 18 | 索引8，`geometry_exploration` | (5179.650, 1710.410) → (5174.347, 1705.107) | 7.5 | −0.536419 |
| 30023 / 200 | UAV0 | 86.19% / 13.81% | 18 | 索引6，`geometry_exploration` | (3364.030, 1119.880) → (3358.727, 1125.183) | 7.5 | −0.450653 |
| 30023 / 200 | UAV1 | 86.83% / 13.17% | 18 | 索引8，`geometry_exploration` | (5179.650, 1710.410) → (5174.347, 1705.107) | 7.5 | −0.587401 |

以上MOVE组概率均来自原策略在原始动作前状态下的分布；UAV1候选按UAV0所选位置条件化。UAV0被强制移动后，策略对UAV1的条件MOVE概率升至约35.8%，但仍按既有Argmax规则选择STAY。

## 完整后缀结果

平均任务时延为该后缀的总任务时延除以任务数；能耗为后缀总车辆能耗。任务数对同一seed的分支相同。Return、Task Utility与其他列中的差值均为分支减A。

| Seed / Slot | 分支 | 后缀Slot数 | Return | ΔReturn | ΔTask Utility | 平均任务时延 (A → 分支, s) | 车辆能耗 (A → 分支, J) | UAV0任务数 (A → 分支) | UAV1任务数 (A → 分支) | Cloud任务 (A → 分支) | 强制位移 / 成本 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 30020 / 100 | A 原策略 | 200 | 2079.602 | 0 | 0 | 10.418 → 10.418 | 1134.596 → 1134.596 | 559 → 559 | 427 → 427 | 1030 → 1030 | 0 m / 0 |
| 30020 / 100 | B 强制UAV0 MOVE | 200 | 2080.374 | **+0.773** | +0.775 | 10.418 → 10.410 | 1134.596 → 1132.132 | 559 → 560 | 427 → 427 | 1030 → 1030 | 7.5 m / 0.001870 |
| 30020 / 100 | C 强制UAV1 MOVE | 200 | 2080.063 | **+0.462** | +0.464 | 10.418 → 10.414 | 1134.596 → 1132.570 | 559 → 559 | 427 → 427 | 1030 → 1030 | 7.5 m / 0.001870 |
| 30023 / 200 | A 原策略 | 100 | 1158.749 | 0 | 0 | 6.655 → 6.655 | 303.291 → 303.291 | 215 → 215 | 159 → 159 | 741 → 741 | 0 m / 0 |
| 30023 / 200 | B 强制UAV0 MOVE | 100 | 1158.626 | **−0.124** | −0.122 | 6.655 → 6.657 | 303.291 → 303.836 | 215 → 214 | 159 → 159 | 741 → 741 | 7.5 m / 0.001870 |
| 30023 / 200 | C 强制UAV1 MOVE | 100 | 1158.891 | **+0.142** | +0.144 | 6.655 → 6.653 | 303.291 → 302.663 | 215 → 215 | 159 → 159 | 741 → 741 | 7.5 m / 0.001870 |

任务效用与移动成本能精确解释Return差值：例如 seed 30023/UAV0 为 `−0.121913 − 0.001870 = −0.123783`。其余三个分支为任务效用提升扣除同样的移动成本后仍略有正收益。四个分支的Cache成本均为0；被强制移动的UAV各移动一次，后续策略没有触发额外位移。

## 任务分配与计算资源

- UAV任务数变化仅为：seed 30020/UAV0 MOVE后UAV0承接任务 `+1`；seed 30023/UAV0 MOVE后UAV0任务 `−1`。前者另一边缘节点38减少1个任务，后者节点58增加1个任务。两个UAV1 MOVE分支的各节点任务总数与A一致。
- 四个分支的Cloud任务数均与对应A相同。UAV移动改变了位置与通信条件；即使节点任务计数相同，重新执行的任务效用、时延和能耗仍略有变化。没有把MOVE率当作UAV利用率，也没有把一个任务的目标变化外推成普遍卸载收益。
- 实际KKT分配可从逐Slot的 `cpu_allocations`、`bandwidth_allocations` 读取。CPU容量为每架UAV `5×10^9 cycles/s`，带宽为 `10 MHz`；报告中的利用率是每Slot分配份额之和除以配置容量的平均值，不是设备侧实测利用率。各分支在汇总精度下几乎相同：seed 30020中UAV0约99.5%、UAV1约99.0%；seed 30023中UAV0约99.0%、UAV1约98.0%。峰值约100%。这四个小幅收益差未表现为明显的平均资源份额变化。

## 判断与限制

1. **MOVE优于STAY的实例：有。**本次3/4对照的后缀Return略高，且Task Utility提升超过移动成本。
2. **MOVE低于STAY的实例：有。**seed 30023/UAV0的Task Utility下降约0.122，移动成本约0.00187，因此主要损失来自任务结果而不是移动成本。
3. **移动是否改变任务分配：有时改变。**UAV0两个分支各改变一个边缘分配；UAV1两个分支的节点任务总数不变，但位置改变后任务效用、时延和能耗仍有微小变化。Cloud总量不变。
4. **是否存在合法MOVE而策略偏好STAY：有。**两个干预点的两架UAV各有18个合法非零MOVE候选，原策略仍选择STAY。
5. **是否修改Micro信用分配或动作机制：当前证据不足以支持直接修改。**正收益差最大约0.773，负收益差约−0.124；只测试少量状态及每个状态一个组内最高评分候选，没有覆盖MOVE候选全集或完整Validation。建议把此结果作为继续检查Micro组概率/价值信用与后缀任务收益对应关系的依据，不作为立即改算法的结论。

## 逐分支原始结果

完整汇总与逐Slot记录：

`runs/diagnostics/uav_stay_move_counterfactual_5e4/counterfactual_results.json`

逐分支文件：

- `runs/diagnostics/uav_stay_move_counterfactual_5e4/seed30020_slot100_branch_A.json`
- `runs/diagnostics/uav_stay_move_counterfactual_5e4/seed30020_slot100_branch_B.json`
- `runs/diagnostics/uav_stay_move_counterfactual_5e4/seed30020_slot100_branch_C.json`
- `runs/diagnostics/uav_stay_move_counterfactual_5e4/seed30023_slot200_branch_A.json`
- `runs/diagnostics/uav_stay_move_counterfactual_5e4/seed30023_slot200_branch_B.json`
- `runs/diagnostics/uav_stay_move_counterfactual_5e4/seed30023_slot200_branch_C.json`

未重新训练或执行完整Validation/Test；未修改正式算法、Reward、环境参数或Checkpoint；未覆盖Cache反事实结果及Validation原始数据。
