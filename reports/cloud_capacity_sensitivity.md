# 阶段2C：Cloud CPU容量敏感性分析

## 结论摘要

在 seed 30020、30023 的 Slot 100、200 共 66 个任务上，Cloud 总 CPU 容量显著改变了Cloud/Edge选择、任务效用、时延和车辆能耗。有限池在四个目标状态均被用满；20 GHz、40 GHz会显著增加Cloud计算等待与Edge竞争，60 GHz接近当前基线的平均时延和任务效用，80 GHz取得最高任务效用和最低平均时延。80 GHz车辆能耗比当前基线高0.199 J，但低于20/40/60 GHz。

有限容量让所测单Slot UPDATE候选的任务效用增益在20–60 GHz下有所增加，尤其是20 GHz；但扣除每次0.319925的Cache更新成本后，所有容量的四状态总 Return 仍下降。对UAV MOVE，仅20 GHz的24个抽样候选中有2个产生微小正Return差；不足以说明UAV移动价值已稳健恢复。

**暂定阶段3对照点：60 GHz。** 它比20/40 GHz更接近当前基线，同时让Edge执行数由8增至13；但它不在本实验中全面优于80 GHz，也没有硬件容量依据。因此60 GHz只是便于继续研究Cache/UAV优化的未标定实验锚点，不是对真实Cloud容量的物理标定结论。

## 实验范围与可比性

- 冻结Checkpoint：`formal_cache_economic_v2_lr_5e_minus_4`；记录的SHA-256为 `4ba388b2dfe87b02a50ae70282e9bf8f4223c0c4d56a9689cb3b49e68aba1318`。状态来自已核验的Validation重放和Cache KEEP/UPDATE配对记录。
- 固定同一任务输入、Cache状态、UAV位置、Matching/Reward和Edge资源参数。对照容量为当前每任务5 GHz且不共享，以及20/40/60/80 GHz共享Cloud CPU池。
- 只改变Cloud CPU合同。Cloud上行保持每任务200 Mbps独立等效链路，`psi_c=11`；本实验没有引入Cloud共享带宽。
- 保留Transfer-Exchange Matching、原Edge CPU/带宽KKT和效用/能耗/Reward。有限池的Cloud CPU按候选分配进行KKT计算，并在Matching评价中重新计算受Cloud池影响的任务；使用完整候选Matching重评以保留共享耦合。系统效用仍为原有 `G = Q - C_cache - C_movement`。
- 当前基线复现阶段2A四个状态的Task Utility、平均时延、Cloud/Edge任务数。目标状态任务数分别为16、15、17、18。
- 共计160个Matching状态均稳定且资源可行：20个KEEP/STAY、20个UPDATE/STAY、120个单UAV MOVE候选。未运行PPO、正式30-seed Validation或Test。

有限池只约束同Slot Cloud CPU总量；本诊断没有增加单任务5 GHz上限，因此池内KKT可能给单个任务分配超过5 GHz。该设定用于隔离总量容量敏感性，不代表已完成的Cloud实例规格标定。

## KEEP/STAY：执行节点与系统指标

下表任务数及Cache命中数均合计四个目标Slot。SV栏列出有任务的具体SV；未列出的SV均为0。`可用Cache命中`表示任务至少有一个本地或缓存且链路可行的Edge候选；`实际缓存节点执行`表示最终实际在缓存节点执行。二者不能互换。

| Cloud CPU合同 | Local | SV | RSU0 | RSU1 | UAV0 | UAV1 | Edge合计 | Cloud | 可用Cache命中 | 实际缓存节点执行 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 当前每任务5 GHz、不共享 | 0 | — | 1 | 0 | 4 | 3 | 8 | 58 | 34 | 8 |
| 20 GHz总池 | 0 | SV28: 2；SV51: 1 | 4 | 3 | 7 | 6 | 23 | 43 | 34 | 23 |
| 40 GHz总池 | 0 | SV28: 2 | 4 | 2 | 5 | 4 | 17 | 49 | 34 | 17 |
| 60 GHz总池 | 0 | — | 4 | 1 | 4 | 4 | 13 | 53 | 34 | 13 |
| 80 GHz总池 | 0 | — | 2 | 0 | 4 | 4 | 10 | 56 | 34 | 10 |

| Cloud CPU合同 | Task Utility合计 | 总Return | 平均任务时延 (s) | 通信时延 (s) | 计算时延 (s) | 车辆能耗总量 (J) |
|---|---:|---:|---:|---:|---:|---:|
| 当前每任务5 GHz、不共享 | 59.780967 | 59.780967 | 4.619576 | 1.153761 | 3.465815 | 1.886872 |
| 20 GHz总池 | 53.765363 | 53.765363 | 8.682604 | 1.340376 | 7.342229 | 7.725846 |
| 40 GHz总池 | 57.987595 | 57.987595 | 5.809686 | 1.181508 | 4.628178 | 3.837068 |
| 60 GHz总池 | 59.653345 | 59.653345 | 4.597965 | 1.177221 | 3.420744 | 3.078043 |
| 80 GHz总池 | 60.580804 | 60.580804 | 3.980832 | 1.152180 | 2.828652 | 2.085880 |

KEEP/STAY没有Cache或移动成本，因此本表总Return等于Task Utility合计。四种有限池在每一个目标Slot都恰好用满容量：20/40/60/80 GHz；当前不共享基线按四个目标状态顺序分别分配70/65/75/80 GHz（每个Cloud任务固定5 GHz）。有限池下实际Cloud CPU分配进入KKT，但Cloud通信仍是每任务独立200 Mbps。逐任务目标、时延分项和CPU分配见 `runs/diagnostics/cloud_capacity_sensitivity/task_results.csv`；逐Slot、逐节点CPU/带宽分配见 `node_summary.csv` 与 `slot_capacity_summary.csv`。

### 观察

- 从80降到20 GHz，Cloud任务从56降到43，Edge任务从10增至23；然而总平均时延从3.981 s升至8.683 s，主要变化来自平均计算时延从2.829 s升至7.342 s。通信时延也从1.152 s升至1.340 s，这是任务改派到共享Edge节点后产生的间接变化，并非Cloud带宽合同变化。
- 20 GHz下任务效用比当前基线低6.016，车辆能耗高5.839 J；40 GHz仍有可见退化。60 GHz的任务效用与基线相差−0.128，平均时延低0.022 s，但车辆能耗高1.191 J。80 GHz在所测四个状态的Task Utility和平均时延优于当前基线，车辆能耗高0.199 J，且Cloud任务比例仍为84.85%。
- 所有容量的可用Cache命中都是34，说明该字段反映固定候选可用性。实际在缓存节点执行分别为8、23、17、13、10，随Cloud容量而改变；所以更小Cloud池会使更多已可行缓存任务实际留在Edge，但并非增加了缓存可行性。
- 低容量时Cloud CPU池饱和，Cloud任务的计算时延升高；新增Edge任务又竞争有限的SV/RSU/UAV CPU和带宽。因而Cloud容量下降不会保证总体时延改善，也不能只用“Cloud任务减少”判断收益。

## KEEP/UPDATE配对

每个合同使用相同的四个状态和已有对照中测试的UPDATE候选。`ΔQ`为UPDATE−KEEP的四状态Task Utility变化合计；`ΔReturn`还扣除了每Slot 0.319925的更新成本。命中数的增量分别报告候选可用命中与实际缓存节点执行。

| Cloud CPU合同 | ΔQ | Q改善Slot数 | ΔReturn | Δ可用Cache命中 | Δ实际缓存执行 | ΔCloud任务 | ΔEdge任务 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 当前每任务5 GHz、不共享 | +0.025701 | 3/4 | −1.253999 | +5 | 0 | 0 | 0 |
| 20 GHz总池 | +0.334084 | 4/4 | −0.945616 | +5 | −2 | +2 | −2 |
| 40 GHz总池 | +0.088500 | 3/4 | −1.191199 | +5 | −1 | +1 | −1 |
| 60 GHz总池 | +0.009228 | 1/4 | −1.270471 | +5 | −1 | +1 | −1 |
| 80 GHz总池 | −0.003077 | 1/4 | −1.282777 | +5 | 0 | 0 | 0 |

有限CPU池提高了本次单Slot UPDATE候选相对于KEEP的Task Utility，20 GHz最明显，并将其净Return损失从当前基线的−1.254改善至−0.946；但所有容量的净Return仍为负，故**没有恢复该UPDATE候选的单Slot正价值**。UPDATE可用命中增加5，但20/40/60 GHz下实际Cloud任务反而分别增加2/1/1；候选可用性并不决定Matching最终执行节点。

既有完整后缀反事实中，强制UPDATE使Cache命中增加26–288、Cloud任务减少26–288，但四条后缀的Return差为−365.611、−140.801、−178.467、−42.324，且任务时延与车辆能耗均上升。该长后缀结果来自当前不共享Cloud合同；本阶段没有在有限容量下重跑长后缀。因此有限池是否改善长期UPDATE价值仍未确认。

逐状态的效用、Return、时延和Cache命中差值见 `cache_update_comparison.csv`。

## UAV MOVE候选

每个合同评估24个MOVE/STAY配对：四状态、两架UAV、每架UAV取官方候选生成器中需求质心、投影需求、几何探索三类各一个代表候选；每次只移动一架UAV。这不是对所有合法位置的穷举，也不是联合移动搜索。

| Cloud CPU合同 | 正ΔReturn候选 | 最大ΔReturn | 平均ΔReturn |
|---|---:|---:|---:|
| 当前每任务5 GHz、不共享 | 0/24 | −0.001535 | −0.002003 |
| 20 GHz总池 | 2/24 | +0.000169 | −0.001680 |
| 40 GHz总池 | 0/24 | −0.001408 | −0.001885 |
| 60 GHz总池 | 0/24 | −0.001535 | −0.001997 |
| 80 GHz总池 | 0/24 | −0.001535 | −0.001981 |

20 GHz下两个正候选分别为 seed 30020 Slot 100 的 UAV1 需求质心移动（ΔReturn +0.000100，平均时延降低0.004053 s）和 Slot 200 的 UAV0 需求质心移动（ΔReturn +0.000169，平均时延降低0.004424 s）。两例UAV任务数都未变化。可见收益是本状态/位置上的小幅Matching效用变化；抽样结果不足以证明MOVE价值普遍恢复。详细配对在 `uav_move_results.csv`。

## 判断与阶段3建议

1. **Cloud容量效应已在这四个目标状态确认。** 四种有限池均用满，且Cloud/Edge节点分配、计算时延、效用和能耗随容量明显改变；这是小规模配对结果，不外推到完整Validation。
2. **有限池可能提高Cache UPDATE的局部价值，但没有恢复所测候选的净Return。** 20 GHz的ΔQ最大，扣除Cache成本后仍为负。长期后缀的Cache效果在有限池下未测。
3. **未确认UAV优化价值恢复。** 仅20 GHz抽样到2个极小正增益候选，其余合同没有正候选；也没有联合MOVE或全候选最优性结论。
4. **建议将60 GHz作为阶段3首个受控实验锚点，而非定标结果。** 20/40 GHz相对基线的时延、效用和能耗退化较大；60 GHz保持近似基线的四状态效用/时延，同时将实际Edge执行从8增至13。80 GHz在当前基础指标上更好，因此阶段3若目标是保持QoS，应同时保留80 GHz作为敏感性参照。没有直接硬件/业务负载资料可把任何一个容量认定为真实Cloud总容量。
5. **阶段3需继续检验**：有限池下多个Slot的连续Cloud负载、更新后完整后缀Return、更多合法Cache更新候选、完整UAV移动候选与联合部署，以及不同到达负载下容量稳健性。不能从这组小样本推断PPO会增加UPDATE/MOVE或Return会提高。

## 产物与限制

诊断文件独立保存于 `runs/diagnostics/cloud_capacity_sensitivity/`：

- `summary.json`：配对状态、容量合同和所有诊断汇总。
- `capacity_aggregate.csv`：各容量的KEEP/STAY系统指标。
- `slot_capacity_summary.csv`：逐状态Matching、资源可行性与Cloud池使用量。
- `task_results.csv`：逐任务执行节点、Cache命中口径、时延分项和CPU分配。
- `node_summary.csv`：逐节点CPU、带宽和时延汇总。
- `cache_update_comparison.csv`：逐状态KEEP/UPDATE配对。
- `uav_move_results.csv`：抽样MOVE/STAY候选配对。
- `run_metadata.json`：Checkpoint哈希、种子/Slot、合同与运行范围。

仅生成本报告和上述独立诊断结果；没有修改正式源码、TOML、Reward或Checkpoint，也没有运行PPO、正式Validation或Test。诊断临时驱动文件已删除。

```ini
CLOUD_CAPACITY_EFFECT_CONFIRMED = YES_FOR_4_TARGET_STATES (160/160 Matching cases stable and resource-feasible; all finite pools saturated)
CACHE_UPDATE_VALUE_RESTORED = NO_FOR_TESTED_ONE_SLOT_RETURN (20 GHz improved ΔQ but all contracts retained negative ΔReturn); LONG_HORIZON_UNCONFIRMED
UAV_OPTIMIZATION_VALUE_RESTORED = PARTIAL_20_GHZ_SAMPLED_CANDIDATES_ONLY (2/24 positive; no robust restoration confirmed)
RECOMMENDED_F_C_TOTAL = 60_GHZ_PROVISIONAL_UNCALIBRATED (near-baseline utility/delay, 13 Edge tasks vs 8 baseline; 80 GHz remains a QoS reference)
READY_FOR_STAGE3 = AFTER_USER_CONFIRMATION (sensitivity evidence is available; physical Cloud capacity and long-horizon adaptation effects remain uncalibrated)
```
