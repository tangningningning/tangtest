# Stage B3-10A：通信参数与任务卸载合理性核对

## 范围与执行环境

本报告只读取B3-9A既有结果、Training数据、两份配置及通信/卸载执行路径；没有读取Validation/Test，没有启动PPO，没有修改配置、源码或模型。Conda激活失败，终端返回“The system cannot execute the specified program.”；实际使用项目此前记录的 `C:\software\anaconda3\python.exe`（Python 3.12.7）及项目纯Python KKT fallback。没有安装或变更依赖。

B3-9A固定实验为Architecture 33、Checkpoint 60、`MULTI_NODE_FACTORIZED`；本地B3-6 Checkpoint文件存在，实测SHA-256为 `2f79020abdbeacba8d527ab3a15b1022e7d591c7fba63e8169bbf8ccdef63710`，与B3-9A记录一致。主线配置SHA-256为 `f3131efb7832efa925c181d9abd07bec5f2bfe3e7507eca2cc2515b77067699b`；B3实验配置SHA-256为 `a86d87ca5d586e89e367aee393b7a1cd9b27813f41ac3c5a19005bfa40a9ea6c`，也与B3-9A记录一致。

两份配置的通信、任务、Cloud、Matching/KKT相关数值相同。全部TOML叶字段的差别仅是B3实验启用了 `algorithm.ppo.cache_action_mode=MULTI_NODE_FACTORIZED`、抽象Macro边界重配置模型及v4多节点Schema，并使用独立实验名/输出目录；主线仍是默认单节点Cache模式和v3 Schema。因此通信模型修改的正式研究基线应先明确为主线 `system_config.toml`；B3文件与其训练Checkpoint继续作为隔离实验记录。若变更物理通信模型，旧Checkpoint即使结构仍可加载，也不能直接作为新模型下已训练、已验证的策略，应另建配置和训练结果链。

## 1. 实际通信与任务效用模型

代码位置：`src/common/system_model.py` 中 `ground_channel_gain`、`uplink_rate`、`air_channel_gain`、`cloud_execution`、`task_utility`；`src/decision/task_allocation.py` 中 `build_feasible_targets`、`_channel_gain`、`_edge_path_is_feasible`、Cloud CPU分配与Matching评价；`src/decision/cache_configuration.py` 中配置映射及 `_serviceability_weight`。

- 地面SV和RSU共用 `h=β_g d^(-α_g)`。实际Matching使用车辆与节点的二维水平距离；只以 `max(d, 1e-9)`避免零距离数值问题，没有1米距离下限，也没有使用RSU的10米部署高度。
- UAV使用 `d_3D=sqrt(d_horizontal²+h²)`，高度为100米；LoS概率按配置的Logistic函数计算，再将LoS/NLoS两种 `β d^(-α)` 增益作概率加权。它是平均信道增益，不是保证链路可靠性的分位数、outage或遮挡模型。
- 边缘上行速率为 `R=bB log2(1+P h/N)`。代码没有干扰项，故这里是SNR而非SINR。实际候选的通信检查只要求位置有限、CPU/带宽/增益/速率为正；没有最低距离、SNR、速率、时延或outage门槛。
- `nodes.sv.range_m=100`、`nodes.rsu.range_m=250`、`nodes.uav.range_m=300`只供预测/候选的软服务性权重使用（距离外权重为0，范围内线性衰减），不限制正式Matching可行集合。
- `RESOURCE_AWARE_FALLBACK`下，所有通过缓存与正速率检查的Edge候选和Cloud都会进入逐任务候选；Cloud不会因存在Edge候选而被屏蔽。Cloud每任务按200 Mbps固定等效上行，不分摊带宽；CPU采用40 GHz有限共享池。Cloud时延为 `15·D/R_c + L/F_c`，车辆能耗为 `P·D/R_c`，15倍系数不进入车辆射频能耗。
- 正式目标对Cloud与Edge使用同一效用函数及归一化：`Q=0.6(1−T/30.6289851662)+0.4(1−E/3.06289851662)`。路径没有单独的Cloud资费、计算运营成本或Edge能耗价格；节点共享资源通过实际时延影响Q，车辆上传能耗进入同一个能耗项，Cache/UAV成本在Task Utility之外扣减。其中T/E参考值是现行配置冻结的标定常数；配置说明E参考值沿用旧TV Local模型标定，属于B1移除Local后的历史尺度来源，尚未在本任务中重标。
- 单个任务的Edge候选成立，不代表一组任务可以同时分配到该Edge。完整Matching评价会对每个节点的共享CPU、带宽分数及最小份额约束重新运行KKT；整体分配若违反资源约束会被拒绝。Cloud也受40 GHz总CPU池约束，但其带宽是逐任务独立假设。

## 2. 配置版本与通信参数核对

两份配置实际值完全一致。配置中若干注释没有随实际值同步：SV、RSU和 `communication.ground.bandwidth_hz` 均设为20 MHz，但各自附近的说明仍称10 MHz；Cloud `cloud_delay_factor` 实际为15，注释仍称固定11。`communication.ground.bandwidth_hz`没有进入实际Matching参数读取；正式速率取自 `nodes.sv/rsu/uav.bandwidth_hz`。这些是来源/说明不一致，不是两套实验采用了不同通信值。

| 参数 | 当前取值 | 依据来源 | 是否合理 | 是否建议修改 |
|---|---:|---|---|---|
| TV发射功率 | 0.199526 W = 23 dBm | 配置标注VEC文献校准值；相关V2X文献也使用23 dBm | 作为仿真基线合理，不是本项目车辆硬件实测 | 暂保留；明确天线增益/EIRP口径 |
| SV / RSU带宽 | 各20 MHz | 当前生效值；配置注释与V2X 10 MHz基线不一致 | 数值可作为实验设定，但来源/注释不自洽 | 先确认目标系统带宽，再同步文档与参数；不要按Reward调值 |
| UAV带宽 | 10 MHz | 配置引用Li等人UAV-MEC仿真参数表 | 作为文献仿真基线合理，不代表实测链路 | 暂保留 |
| 接收噪声 | `3.1623e-13 W` = −95 dBm | 配置及V2X基线称为10 MHz总噪声 | 当前解释为固定10 MHz有效噪声；配置没有噪声PSD、接收机NF或干扰预算 | 需要先确定噪声PSD/NF及带宽语义，之后按实际分配带宽一致计算；现在不直接改数 |
| 地面信道 | β=`1.6368e-5`，α=2.75，PL(1m)=47.86 dB；2D距离 | V2X文献中的5.9 GHz参考路径损耗参数 | 可作为简化基线；忽略地面节点高度、遮挡、阴影/快衰落 | 保留用于可比性；物理部署结论需再校准/扩展传播模型 |
| UAV空地信道 | 2 GHz；LoS/NLoS α=2；β=`1.1302e-4`/`1.4229e-6`；η=9.61/0.16 | 配置将自由空间损耗加1/20 dB额外损耗；LoS概率函数采用Al-Hourani形式，η值注明为常见Urban取值 | 适合作为平均信道仿真基线；不是可靠性保证 | 暂保留并明确环境类别；真实覆盖需引入校准的阴影/outage统计 |
| UAV高度 | 100 m（部署值与信道值一致） | 配置引用Li等人的仿真高度 | 内部一致，属仿真设定而非普遍最优高度/法规结论 | 暂保留 |
| 100/250/300 m范围 | SV/RSU/UAV分别为100/250/300 m | 配置标为候选/实现软尺度 | 若理解为预测权重尺度合理；不是通信覆盖半径 | 保留作软特征，不得声称它们是硬物理覆盖 |
| Cloud计算/上行/时延 | 40 GHz有限共享CPU；每任务200 Mbps；`psi_c=15`；带宽不共享 | TOML生效键为 `cpu_capacity_cycles_per_s=40e9`、`uplink_rate_bps=200e6`、`cloud_delay_factor=15`；注释另残留“60 GHz”和“11” | 有限共享CPU与独立固定上行是明确仿真假设；配置文字与生效值不一致 | 保持本实验的实际值；先核对来源并修正注释，再决定是否校准Cloud接入与回传 |
| 最低物理链路门槛 | 当前无最低SNR/速率/QoS/outage门槛 | 源码可行性判定 | Shannon速率会连续惩罚弱链路，但不代表实际无线接入可达或可可靠传输 | 若研究结论要声称物理可达性，建议补充经通信制式/QoS校准的可行规则；门槛值目前未准备好 |

### 噪声与带宽的一致性

当前噪声值确实对应文献参数表中的“10 MHz内−95 dBm”数量级。标准链路预算通常从噪声密度、接收机噪声系数和占用带宽计算总噪声；例如热噪声密度约−174 dBm/Hz，ETSI/3GPP链路预算还显式列出接收机噪声系数及带宽。若相同噪声密度和接收机NF下将带宽从10 MHz改成20 MHz，总噪声会上升约3.01 dB。当前代码在KKT改变带宽份额b时仍用固定的−95 dBm计算SNR，只把b乘到带宽项上。因此，作为“固定有效噪声”的抽象模型可以复现，但它与“−95 dBm是10 MHz积分噪声、bB是实际分配带宽”的物理解释不一致。Cloud区也有同类说明漂移：CPU池生效值40 GHz，邻近注释却称60 GHz实验锚点；`cloud_delay_factor`生效值15，注释称11。应以有效配置键解释既有结果，并在后续基线冻结前核实这些来源。

## 3. Training卸载比例及候选链路阶段

从B3-9A已保存的8个Training Episode、每集2个BASELINE种子、每条60槽记录汇总，共13,324个任务；未重新运行策略。Cloud占10,858（81.49%），SV占9（0.0675%），RSU占768（5.76%），UAV占1,689（12.68%）。按Episode合并两种子后，Cloud占比在79.01%–84.31%，SV每集0–4个任务：

| Training Episode | 任务数 | Cloud | SV | RSU | UAV |
|---:|---:|---:|---:|---:|---:|
| 20001 | 1,708 | 1,440 (84.31%) | 1 | 69 | 198 |
| 20053 | 1,710 | 1,373 (80.29%) | 1 | 109 | 227 |
| 20094 | 1,670 | 1,374 (82.28%) | 1 | 93 | 202 |
| 20135 | 1,602 | 1,287 (80.34%) | 0 | 123 | 192 |
| 20177 | 1,644 | 1,299 (79.01%) | 0 | 117 | 228 |
| 20218 | 1,698 | 1,377 (81.10%) | 4 | 82 | 235 |
| 20259 | 1,628 | 1,333 (81.88%) | 2 | 89 | 204 |
| 20300 | 1,664 | 1,375 (82.63%) | 0 | 86 | 203 |

全16条轨迹的官方候选可行性重建显示：13,324个任务中8,220个（61.69%）请求服务在至少一个Edge缓存，且恰有8,220个通过现行正速率Edge候选检查；缓存存在却因当前路径检查被剔除的任务为0。按角色统计的“至少有该类节点候选”任务数互有重叠：SV 5,185、RSU 4,895、UAV 4,429。正式候选构造要求服务已缓存、几何/位置有效、CPU和带宽为正且计算得到的SNR/速率为正；它不检查距离尺度或最低链路质量。

缓存节点—请求链路的全带宽基准速率分布（在KKT带宽份额之前，不是每个实际任务的分配速率）：

| 节点类 | 链路对数 | 距离中位数 / P90 | SNR中位数 / P10 | 全带宽速率中位数 / P10 |
|---|---:|---:|---:|---:|
| SV | 7,808 | 583 / 1,128 m | −5.91 / −13.80 dB | 6.59 / 1.18 Mbps |
| RSU | 5,598 | 1,011 / 1,618 m | −12.49 / −18.10 dB | 1.58 / 0.44 Mbps |
| UAV | 5,277 | 1,045 / 1,619 m（含100 m高度） | 6.14 / 1.34 dB | 23.55 / 12.39 Mbps |

SV/RSU弱链路的速率仍为正，所以当前代码会将它们视为候选；真实KKT分给任务的带宽份额还会降低速率。此处链路对是“任务服务当前被某Edge缓存”的样本，不能直接解释为全路网无线覆盖统计。

### 最少量正式Matching/KKT定位

事先固定Episode 20001、20135的seed 930901，并各取Slot 0、30、59，共6个保存状态。每个状态用同一Cache、UAV、车辆位置、任务及策略已执行后的动作记录，调用正式Slot/Matching/KKT路径重新求解；没有重新采样策略动作。6/6的Task Utility与B3-9A保存值完全一致（绝对误差0），Matching稳定检查通过且资源可行。

| Episode / Slot | 任务 | 有Edge候选 | Cloud最终选择且有Edge候选 | 最终Cloud / SV / RSU / UAV |
|---|---:|---:|---:|---:|
| 20001 / 0 | 12 | 9 | 7 | 10 / 0 / 0 / 2 |
| 20001 / 30 | 16 | 14 | 12 | 14 / 0 / 0 / 2 |
| 20001 / 59 | 14 | 6 | 3 | 11 / 0 / 1 / 2 |
| 20135 / 0 | 13 | 12 | 10 | 11 / 0 / 0 / 2 |
| 20135 / 30 | 14 | 11 | 9 | 12 / 0 / 1 / 1 |
| 20135 / 59 | 12 | 6 | 4 | 10 / 0 / 1 / 1 |
| 合计 | 81 | 58 | 45 | 68 / 0 / 3 / 10 |

这58个任务都同时有Cloud候选；其中45个最终由Cloud执行，说明Edge可行时Cloud仍可以参与竞争。Matching历史包含46个Edge→Cloud单任务transfer操作，以及7个含Edge→Cloud一条腿、同时伴随Cloud→Edge反向腿的exchange操作；合计53条Cloud方向的Edge映射变化（SV 37、RSU 3、UAV 13）。53个操作级整体目标增量均为正，范围约+0.00015至+4.256；exchange的增量属于整组交换，不能单独归因给Cloud方向的一条腿。这些是搜索路径的局部证据，不等于全局最优证明，但没有显示“候选生成后Cloud被错误屏蔽”或增量评价明显失效。

对上述45个Cloud最终执行且有Edge备选的任务，使用代码中的理想单任务评价（Edge假设独占全CPU/全带宽，Cloud用正式当前资源分配）作有限筛查：37个任务即使在该Edge理想资源点，Task Utility仍不超过Cloud当前值；8个理想Edge值更高。后8个仍需考虑其他任务共享资源和全局重新分配，不能据此认定Matching漏选。

样本按已执行目标汇总的描述统计如下。由于任务难度和节点分布不同，这不是同一批任务的Cloud/Edge配对比较：

| 已执行节点 | 数量 | 平均时延 | 平均车辆能耗 | 平均Task Utility |
|---|---:|---:|---:|---:|
| Cloud | 68 | 2.276 s | 0.00882 J | 0.9543 |
| UAV_0 | 4 | 1.450 s | 0.0531 J | 0.9647 |
| UAV_1 | 6 | 2.145 s | 0.0791 J | 0.9477 |
| RSU_1 | 3 | 1.285 s | 0.1126 J | 0.9601 |
| SV | 0 | — | — | — |

Cloud的车辆传输能耗通常较低，因为它按较高固定速率计算；Edge有时延优势，但稀疏弱链路会增加传输时延/能耗，CPU与带宽又由同节点任务共享。上表的Edge平均时延较短并不代表其对同一任务有更高系统效用；系统效用同时考虑时延、车辆能耗和全体任务的共享资源重分配。

## 4. 对Cloud占比、SV利用率和Cache更新的判断

1. **Cloud占比超过81%与当前形式化优化一致，但只对当前模型成立。** Cloud始终进入逐任务候选；其200 Mbps是每任务固定等效速率，不按并发任务分摊，而40 GHz CPU才共享。边缘带宽/CPU按节点共享，地面信道损耗较强，且弱SNR未被物理QoS门槛过滤。在固定参数下，完整Matching/KKT可发现某些Cloud映射提高当前全任务归一化效用。六个状态的53次Edge→Cloud接受转移提供了直接证据。该结果不能证明现实网络中Cloud一定更快，也不证明算法已找到全局最优。
2. **SV极低利用率不是缓存候选普遍缺失或正速率链路被过滤造成。** 16轨迹中有5,185个任务存在SV缓存候选；6个重算状态中没有最终SV任务，且Matching路径接受了37次SV→Cloud转移。现有证据更支持“SV链路质量/低CPU及共享资源下的Task Utility竞争劣势”；SV每节点CPU为1 GHz，而RSU为10 GHz、UAV为5 GHz。缓存是否命中仍是必要条件，但不足以保证实际执行。此分析不能把弱SNR和资源影响的相对贡献单独定量排序。
3. **缓存更新只能增加一类Edge选项，不能强制把任务迁到Edge。** Cloud始终保留，新增缓存服务若通信/计算后的全局效用不高，Matching会继续选Cloud。因此此前Cache UPDATE收益小，和“服务已缓存/可行不等于Edge效用更好”相符；不能据本次结果断言所有缓存更新无价值。
4. **目前没有证据支持Matching/KKT实现错误。** 六个复现值准确，稳定检查与资源可行性通过；局部交换记录中Edge→Cloud操作带来正的当步完整目标增量。这只排除了本样本中的明显路径问题，不是对全部状态的全局最优证明。
5. **当前物理可达性表达不足。** 任务可行集合把任何有限正速率视为可用；实测训练链路代理里部分地面SNR中位数为负、P10低于−13 dB，但仍可进入候选。若要把Edge候选称为物理可达/可靠，应采用指定通信制式的最低速率、SINR/BLER/outage或任务时延条件，并匹配实际带宽分配模型。Shannon目标本身会惩罚低速率，因此并非优化问题数学上必然需要一个硬SNR门槛；门槛需求来自物理可达性/可靠性主张。

## 5. 建议与实施顺序

- 暂不为提高Edge比例修改发射功率、Cloud速率、Cache初始化、Reward或Matching。保持现有结果可复现。
- 先确定Ground实际采用10 MHz还是20 MHz，并修订配置说明/冗余字段口径；再定义噪声PSD、接收机NF及分配带宽到噪声的关系。不要在未知接收机和无线标准时指定SNR阈值。
- 明确Cloud 200 Mbps到底是每车保证速率、每流仿真上限还是平均速率，并解释`psi_c=15`的校准依据；修正配置注释中“11”的历史残留。保持Cloud计算40 GHz共享池的合同不变，直至有新的系统假设依据。
- 在这些合同冻结后，用指定无线标准/链路预算校准Ground与Air参数，并以QoS/误包率或可靠速率定义正式物理可行性；软候选范围继续与硬物理约束分开。
- 任何正式通信模型改变都会改变Matching可行集合、Task Utility和训练分布。新模型需使用单独配置与Checkpoint谱系重新训练/验证；B3-6 Architecture33/Checkpoint60不可被称作新通信模型下的有效训练策略。

## 参考依据

- 车辆通信参数23 dBm、5.9 GHz下1 m路径损耗47.86 dB、损耗指数2.75和10 MHz内−95 dBm： [V2X PHY参数论文（IEEE TVT）](https://people.computing.clemson.edu/~jmarty/projects/lowLatencyNetworking/papers/V2V/CompDSRCandCV2XforCoopAwareness.pdf)，另见 [Sun et al., 2019](https://journals.sagepub.com/doi/abs/10.1177/1550147719837859)。这些是论文仿真参数，非本项目硬件测量。
- 热噪声密度、接收机噪声系数和带宽共同形成接收噪声： [ETSI/3GPP TR 36.931](https://www.etsi.org/deliver/etsi_tr/136900_136999/136931/10.00.00_60/tr_136931v100000p.pdf)。
- 空地LoS概率及路径损耗建模形式： [Al-Hourani et al., IEEE GLOBECOM 2014](https://doi.org/10.1109/GLOCOM.2014.7037248)。项目配置注明η取值采用后续常见Urban设置，并非宣称直接照录该文表值。
- UAV高度与10 MHz作为仿真参数的来源： [Li et al., 2021, UAV-enabled time-sensitive MEC](https://link.springer.com/article/10.1186/s13638-020-01861-8)。
- Cloud 200 Mbps数值先例（不意味着对每辆车有物理保证）： [Al-Shareeda et al., IEEE TVT](https://doi.org/10.1109/TVT.2025.3620810)。

## 最终汇总

```ini
STAGE_B3_10A = COMPLETE
CLOUD_EXECUTION_FRACTION = 81.49% (10858/13324)
SV_EXECUTION_FRACTION = 0.07% (9/13324)
CLOUD_DOMINANCE_CAUSE = Cloud始终可选、每任务固定200 Mbps且不共享带宽；有限共享CPU下，当前完整Matching/KKT多次确认Cloud映射可提高系统Task Utility；弱Edge链路仍可进入候选
SV_LOW_UTILIZATION_CAUSE = 已缓存候选并不少，但SV每节点CPU仅1 GHz，地面链路SNR/速率偏低且共享资源竞争；样本中Matching接受SV转Cloud操作
COMMUNICATION_PARAMETERS_CONSISTENT = NO
PHYSICAL_SNR_GATE_NEEDED = YES（若要主张物理可达/可靠；Shannon效用模型数学上不必另加统一硬阈值）
PHYSICAL_SNR_THRESHOLD_READY = NO
MAINLINE_B3_COMPATIBILITY = 通信/任务/Cloud/Matching配置完全一致；B3使用Architecture33/Checkpoint60多节点v4合同，主线仍为单节点默认模式。通信模型变更后需隔离新配置及Checkpoint谱系
RECOMMENDED_PARAMETER_CHANGES = 暂无数值调参；先协调地面20/10 MHz声明，定义噪声PSD/NF与分配带宽口径，核实Cloud每任务速率与psi_c=15来源，并按选定制式建立可校准链路门槛
RECOMMENDED_NEXT_STEP = 冻结通信制式和链路预算合同后，再做小规模固定状态的候选可达性/Matching验证
PPO_TRAINING_EXECUTED = NO
SYSTEM_MODEL_CHANGED = NO
PPO_ALGORITHM_CHANGED = NO
FORMAL_CONFIG_CHANGED = NO
```
