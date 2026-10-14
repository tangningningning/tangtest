# Stage B3-10B：通信参数口径与物理模型定稿

## 范围与版本

本报告以B3-10A和当前本地源码、两份现行配置为依据，提出下一通信模型的推荐合同；不修改源码或配置、不运行仿真或训练。Conda激活仍不可用（终端返回“The system cannot execute the specified program.”）；沿用此前验证过的项目Python环境。本次没有安装依赖。

当前正式主线是 `configs/proposed/system_config.toml`，仍使用单节点Cache决策模式。`system_config_multi_node_factorized.toml` 是隔离的Architecture 33 / Checkpoint 60多节点实验配置。两份配置当前通信、任务、Cloud及Matching/KKT生效数值相同；本次不更改其中任一文件。B3 Checkpoint和历史结果继续作为旧通信口径基线。通信参数或可行性语义变化后，即使网络维度不变，旧模型也不代表新物理系统下已训练策略。

## 1. 当前实现合同

| 链路/参数 | 当前生效值与实现 | 解释边界 |
|---|---|---|
| TV发射功率 | `nodes.vehicle.transmit_power_w=0.1995262315 W`，即23 dBm；任务功率固定 | 作为V2X仿真基线，不含显式天线增益、馈线损耗或EIRP区分 |
| Ground SV/RSU | 各20 MHz；共用 `h=βd^{-α}`，`β=1.6368e-5`、`α=2.75`；Matching用二维水平距离 | β等价于5.9 GHz自由空间1 m路径损耗约47.86 dB，但配置未显式记录Ground载频。没有阴影衰落、遮挡或干扰项 |
| UAV空地链路 | 10 MHz，2 GHz，100 m高度；三维距离，按LoS/NLoS概率加权平均信道增益 | 平均增益不是可靠性或outage保证；空地LoS形式参考Al-Hourani，参数来源按配置说明。 |
| 噪声 | 固定 `3.16227766e-13 W`，即−95 dBm | Matching/KKT带宽分数变化时噪声仍不变；没有独立PSD或接收机NF |
| KKT带宽 | 节点带宽为 `B_m`，任务份额 `b_vm`，速率写为 `b_vm B_m log₂(1+P h/N)`；每个节点内份额和受约束 | 代码没有跨节点频谱池，也没有同频干扰，因此当前各节点带宽池在模型上相互独立 |
| Cloud | 每任务200 Mbps固定等效上行、彼此不分带宽；有限共享CPU池40 GHz；`ψ_c=15`；时延 `ψ_c D/R_c+L/F_c`，车辆能耗 `P D/R_c` | 这是抽象服务链路，不是每车有保障的现实无线速率。`ψ_c`只放大时延项，不放大车辆射频能耗。 |

相关实现位于 `src/common/system_model.py` 的 `ground_channel_gain`、`uplink_rate`、`air_channel_gain`、`cloud_execution`，以及 `src/decision/task_allocation.py` 的 `_channel_gain`、`_edge_path_is_feasible`、`build_feasible_targets`、KKT资源分配路径。当前没有干扰建模，应称SNR，不应称SINR。Edge候选只要求缓存命中、节点/位置和资源为正、计算出的速率大于0，没有最低SNR、可靠速率、上传时限或outage约束。Cloud在 `RESOURCE_AWARE_FALLBACK` 下仍与所有可行Edge一起参与效用比较。

配置中SV/RSU带宽字段附近的“统一10 MHz”说明与实际20 MHz不一致；`communication.ground.bandwidth_hz`也是20 MHz，但正式Matching使用 `nodes.sv/rsu.bandwidth_hz`。当前统一噪声字段的−95 dBm注释指向10 MHz噪声。之后应只保留一个清楚的数据合同，不能让同名带宽字段各自代表不同用途。

## 2. Ground 10 MHz与20 MHz

3GPP E-UTRA V2X PC5 Band 47覆盖5855–5925 MHz，规范列出的V2X信道带宽包含10和20 MHz；NR V2X n47也包含10和20 MHz选项。因此这两个数值都可以成为合规的仿真配置，但标准本身不替本项目决定使用哪个带宽，也不证明每个Edge节点都各自独占一整段同频带宽。[ETSI TS 36.101（E-UTRA V2X Band 47）](https://www.etsi.org/deliver/etsi_ts/136100_136199/136101/15.19.00_60/ts_136101v151900p.pdf)，[ETSI TS 38.101-1（NR V2X n47）](https://www.etsi.org/deliver/etsi_ts/138100_138199/13810101/16.21.00_60/ts_13810101v162100p.pdf)

| 候选 | 与现有依据的对应关系 | 优点/限制 |
|---|---|---|
| A：Ground 10 MHz | 与B3-10A所引V2X参数基线的10 MHz口径、−95 dBm/10 MHz噪声锚点一致；Ground路径损耗参数指向5.9 GHz | 推荐作为下一版文献对齐基线；仍需明确实际使用LTE-V2X PC5、NR-V2X PC5还是其他接口。 |
| B：Ground 20 MHz | 是Band 47的有效带宽选项，且与当前节点字段的生效值一致 | 可用于保留现有带宽设定，但必须同步按实际占用带宽计算噪声，并明确频谱池/信道复用方式；不能继续把它描述成10 MHz。 |

**建议：Ground SV和RSU先以10 MHz作为下一版推荐基线，UAV仍按其独立的10 MHz空地信道建模。**依据是当前地面路径损耗及噪声注释所对应的V2X文献口径，而非Reward或Edge卸载率。公开V2X研究确实采用过23 dBm、5.9 GHz、47.86 dB@1 m、路径损耗指数2.75和−95 dBm/10 MHz的组合；它们是仿真假设，不是本项目硬件测量。[V2X参数论文](https://people.computing.clemson.edu/~jmarty/projects/lowLatencyNetworking/papers/V2V/CompDSRCandCV2XforCoopAwareness.pdf)，[Sun et al., 2019](https://journals.sagepub.com/doi/abs/10.1177/1550147719837859)

Ground未来配置还应显式记入 `carrier_frequency_hz=5.9e9`，因为现有β将载频影响折叠在参考增益中；若载频改变，β也必须重算。10/20 MHz只改变资源带宽，不自动改变路径损耗。当前KKT是每个节点独立满足 `Σ_v b_vm≤1` 且没有跨节点干扰；若多个节点复用相同频谱，必须另建频谱复用/干扰合同。若不建模干扰，则需把节点带宽解释为节点各自的正交资源池，不能笼统地把每个节点的20 MHz都说成同一公共信道。

## 3. 统一噪声与带宽合同

建议采用热噪声PSD与接收机噪声系数分离的合同：

```text
N_dBm(v,m) = N0_dBm_per_Hz + 10 log10(B_alloc(v,m)_Hz) + NF_m_dB
B_alloc(v,m) = b_vm * B_m
N0 = -174 dBm/Hz  (约290 K热噪声基线)
```

这里 `B_m` 是该节点的系统带宽，`b_vm` 是KKT实际分给任务的频域份额；公式假设 `b_vm` 表示正交频谱分配，而不是时分占空比。线性域为 `N=N0_linear·B_alloc·10^(NF/10)`。在没有显式干扰时，链路指标是 `SNR=P_v h/N`。保持现有 **23 dBm总任务发射功率固定**，不随带宽份额缩放；功率不是PSD。于是速率应为：

```text
R_vm = B_alloc * log2(1 + P_v h_vm / N_vm(B_alloc))
```

ETSI/3GPP链路预算示例将−174 dBm/Hz、噪声系数和接收带宽分别列出；例如LTE系统场景中的BS NF 5 dB、UE NF 9 dB是特定链路预算假设，不是所有SV、RSU、UAV接收机的强制通用值。[ETSI TR 36.942](https://www.etsi.org/deliver/etsi_tr/136900_136999/136942/18.00.00_60/tr_136942v180000p.pdf)

| 频宽 | NF=9 dB时积分噪声 | NF=5 dB时积分噪声 | 说明 |
|---:|---:|---:|---|
| 10 MHz | −95.00 dBm | −99.00 dBm | −95 dBm/10 MHz等价于热噪声加约9 dB NF |
| 20 MHz | −91.99 dBm | −95.99 dBm | 同一NF下带宽翻倍，噪声增加约3.01 dB |

在相同带宽下，若分配份额为 `b<1`，积分噪声再减少 `10log10(b)` dB。例如10 MHz节点分配一半带宽、NF为9 dB时，噪声为−98.01 dBm。**当前固定−95 dBm对20 MHz满带宽约等于6 dB有效NF，但在b变化时不随接收带宽变化；因此它不能同时代表10/20 MHz和所有KKT份额下的固定NF物理接收机。**−95 dBm/10 MHz锚点与新公式可在Ground=10 MHz、NF=9 dB时精确对齐。

建议在配置中为接收节点明确 `noise_psd_dbm_per_hz` 与 `receiver_noise_figure_db`（必要时按SV/RSU/UAV区分），并删除用一个固定总噪声瓦数跨带宽复用的含糊语义。当前没有接收机实测值，**NF=9 dB只能作为复现−95 dBm/10 MHz的有效基准值，不能宣称它已被本项目硬件验证。**如果采用不同角色NF，应以设备/无线制式链路预算给出来源。23 dBm也需确认是天线口导出功率还是EIRP；当前模型无天线增益，实施时要避免把天线增益漏掉或重复计入信道增益。

这次只提出公式，尚未改动`uplink_rate`。实际改用随`b_vm B_m`变化的噪声后，速率对带宽份额不再是原实现中“份额乘固定频谱效率”的线性形式。`src/common/system_model.py`的速率/时延/能耗函数及`src/decision/task_allocation.py`的KKT导数、约束和收敛检查必须作为同一变更更新并验证；不能只替换SNR一行而假定旧KKT性质不变。

## 4. Cloud接入合同：A与B

| 方案 | 定义 | 需要的资源合同 | 评价 |
|---|---|---|---|
| A：抽象远程执行服务 | Cloud保持200 Mbps/任务的固定等效上传率，40 GHz仍为共享CPU池；`ψ_c`表示合并后的远程传输/核心网额外时延因子；车辆能耗仍只按其等效无线上传时间计 | 需明确200 Mbps是“仿真等效每任务速率”，不是每车物理保证；`ψ_c=15`仍是经验/实验设计参数；不模拟Cloud接入带宽竞争 | 最容易维持当前研究可比性，作为基准抽象清楚且可执行；不支持现实无线接入或Cloud带宽容量主张 |
| B：实际接入+回传 | 车辆先经指定无线AP/Serving RSU上传，再通过有线回传进入Cloud；Cloud计算仍从40 GHz共享池分配 | AP关联/选择与覆盖；接入链路复用Ground无线模型并参与AP内带宽分配；回传拓扑、容量/每流速率、并发共享/排队；Cloud CPU分配；是否显式建核心网延迟 | 更适合作为物理路径完整的正式目标，但当前缺少关键网络参数及关联机制，不能现在直接实现 |

方案B可以写为：

```text
T_cloud(v) = D_v / R_access(v,a) + D_v / R_backhaul(v,a) + L_v / F_cloud(v) + T_core(v)
E_vehicle(v) = P_v * D_v / R_access(v,a)
```

其中访问点 `a` 必须由系统显式指定或选择。接入带宽按AP共享并进入资源分配；回传须说明容量和任务并发如何共享（例如为每流分配回传份额并限制份额和，或采用明确的队列/调度模型）；Cloud CPU总分配不超过40 GHz。若不单独建模核心网队列/传播，则 `T_core` 为0，不应再把同一段网络延迟重复塞入 `ψ_c`。显式回传后，当前 `ψ_c=15` 应移除或仅保留有独立标定证据的残余固定时延，不可与 `D/R_backhaul` 双重计费。Cloud结果返回数据当前按忽略处理，可在论文假设一致时继续忽略。

参考Liu等人的论文，III-A描述终端通过无线链路连接一个Edge Server，Edge Server间及其到Cloud Server通过有线链路连接；III-C Case 4将Cloud描述为没有Edge缓存所需服务时的处理路径；III-D明确列出关联Edge的无线上传、Edge到Cloud的固定回传速率及其时延组成。因此，**B更接近该论文的通信架构**。但论文文字中的Cloud后备描述不能自动证明它数学上禁止“已有可缓存Edge候选时仍选Cloud”：其(17)约束明确了单服务器分配、缓存/资源可行性、带宽和QoS等条件，却没有直接写出`d_C=1`当且仅当所有Edge均无该服务的互斥约束。Cloud路径结构与Cloud候选门控应分开定稿。[Liu et al., arXiv v2，III-A、III-C、III-D、III-F](https://arxiv.org/abs/2307.09691)，[IEEE TWC DOI](https://doi.org/10.1109/TWC.2024.3430486)

**建议：以B作为目标物理通信模型；在AP关联、接入带宽、回传容量/共享规则补齐前，不实施B，也不宣称它已被当前系统支持。**短期对照基线仍可保留A，并在配置/报告中明确命名为`ABSTRACT_FIXED_PER_TASK_CLOUD_LINK`。保留A不等于把200 Mbps解释成现实的每车保证服务，也不等于偷偷降低该数值来改变卸载比例。当前Cloud与Edge自由竞争的动作规则不因接入路径建模自动改变；严格Cloud fallback仍需单独的系统模型决定。

## 5. SNR可达性和可靠速率规则

三种约束要分开：

1. **物理接入可靠性：** 由所选无线制式、频带、天线、调制编码、接收机灵敏度、目标BLER/PER或outage概率决定。可表达为“链路质量达到指定MCS在目标误包率下的要求”或“`P_out≤ε`”。现有UAV概率平均增益和Ground单斜率确定性模型不包含足以推导可靠性门槛的衰落/遮挡分布。
2. **任务上传要求：** 对分配到的实际`B_alloc`检查可靠goodput；若任务有端到端时限，应从显式QoS时限扣除计算、回传等阶段，定义可用上传预算，再要求上传完成。不得自行添加未经数据合同定义的固定速率或时限。
3. **全局资源可行性：** 单条链路达标仍不表示同节点任务集合可行。Matching/KKT必须同时满足节点带宽和CPU总量、各任务最小份额/最低goodput等约束。若某个候选任务要求的最低带宽份额之和超过1，联合分配不可行，候选应被整体拒绝或重新匹配。

因此建议未来使用**制式/接收机条件化的最低可靠goodput或outage约束**，而不是所有角色共用一个未经校准的SNR门槛。若第一版仅实现链路级门槛，应由所选PHY/MCS的接收灵敏度、误包率目标和信道模型推导；若使用速率门槛，应来自任务QoS合同。**当前没有足够依据填写0、3或5 dB，也没有足够依据给出统一最低Mbps。**`SNR_THRESHOLD_JUSTIFIED=NO`。新模型没有显式干扰时仍称SNR；若未来加入同频干扰则才改用SINR。

100/250/300 m现为SV/RSU/UAV候选评分中的软服务性尺度，不是物理半径。未来可将预测需求/Cache候选特征改为“基于预测车辆位置和同一物理信道计算的可靠goodput/可达概率或上传时限可行概率”，并使用和正式Matching相同的载频、噪声、NF、资源带宽与UAV信道参数；不得把软特征范围直接转写成硬通信距离。

## 6. 参数建议表

| 参数 | 当前取值 | 下一版建议 | 是否仍待确认 |
|---|---:|---|---|
| TV发射功率 | 23 dBm | 先保留固定总功率；标明是导出功率还是EIRP，天线增益只计一次 | 是 |
| SV/RSU带宽 | 20 MHz生效，注释写10 MHz | 文献对齐基线推荐10 MHz；20 MHz是合法替代，但须同步噪声与频谱池合同 | 是，最终选择 |
| UAV带宽 | 10 MHz | 保留其2 GHz空地链路独立10 MHz系统带宽；明确是否与地面频谱隔离 | 是，资源池关系 |
| 噪声PSD | 未配置 | 使用−174 dBm/Hz热噪声基线，按实际分配带宽积分 | NF仍待确认 |
| 接收机NF | 未配置；当前是固定−95 dBm总噪声 | 以设备资料为准；NF=9 dB可作对齐−95 dBm/10 MHz的有效基准，不应当作实测 | 是，最好按接收节点类型 |
| Ground载频及路径损耗 | 未显式配置载频；β对应5.9 GHz的47.86 dB@1m，α=2.75 | 新合同显式写5.9 GHz；暂保留既有简化单斜率作为可比基线，后续按目标道路/遮挡场景校准 | 是，确认V2X接口 |
| UAV信道 | 2 GHz，100 m；LoS/NLoS平均增益 | 先保留；要声称可靠性时补充遮挡/衰落或outage模型 | 是，是否要可靠性主张 |
| 100/250/300 m | 软候选尺度 | 仅留作proposal特征；不作硬覆盖门槛，逐步改成与物理信道一致的预测可服务性 | 否（边界已明确） |
| Cloud 200 Mbps / 40 GHz / ψ=15 | 每任务固定200 Mbps；CPU共享40 GHz；延时因子15 | 近期仅作为方案A的显式抽象基线；目标方案B需替换固定200 Mbps链路并处理ψ重复计时 | 是，接入/回传合同 |
| 最低可靠性门槛 | 当前无门槛 | 依目标PHY/MCS、BLER/PER/outage及任务QoS推导可靠goodput；不先写固定SNR | 是，PHY与可靠性目标 |

## 7. 影响模块与迁移边界

后续实现需要作为一组通信模型变更处理：

- `src/common/system_model.py`：统一噪声PSD/NF、Ground显式载频与信道增益、按`B_alloc`计算SNR/goodput；方案B新增接入与回传时延/能耗分解。
- `src/decision/task_allocation.py`：Edge/Cloud物理路径可行判定、接入节点选择/关联、最低可靠goodput和整体资源可行性；更新KKT速率导数、份额下界、Cloud CPU/回传资源分配及收敛/可行性验收。
- `src/decision/cache_configuration.py`、`src/ppo_simple/primitive_control.py`：将服务候选软距离代理逐步对齐为基于同一信道的预测可服务性；不把未来真实任务用于当前候选特征。
- `src/ppo_simple/rollout.py`及策略Observation/候选特征合同：物理可行集合或候选特征语义变化后，即使数组维度不变也属于新Observation语义；更新schema/架构版本并重新建立训练样本与Checkpoint谱系。
- 新建独立配置和run目录；旧主线/B3配置、Training Pool、Manifest、Checkpoint和报告都保留原样。原有轨迹/任务原始文件可能可复用，但新物理口径下的预测候选、Matching奖励和策略轨迹必须重新计算，不可把旧环境Return与新环境Return混成同一基线。

不需要重写UAV运动控制、Cache动作或Reward公式本身；但其观察特征、策略分布和训练状态对新可行集合/效用分布不再等价。当前`task_utility`公式可以保留，改变的是输入的时延/能耗及可行路径。

## 8. 建议的最小后续实施顺序

1. 用户确认无线接口/载频、Ground=10或20 MHz、接收NF/EIRP口径和Cloud目标路径。
2. 若采用Cloud模型B，先补齐服务AP关联、接入资源池、回传容量与共享/排队规则；缺任一项时继续把A标成旧抽象对照，不假定默认值。
3. 在新隔离配置中一次性更新PSD/NF与按分配带宽计算的速率；同步重推KKT导数和资源约束。
4. 选择有依据的PHY可靠goodput/时限规则，并使预测Cache/UAV候选使用同一信道模型；没有BLER/outage输入时先明确只做确定性链路基线，不称可靠性保证。
5. 对单个Training确定性Slot做最小路径、CPU/带宽可行性和有限Reward核对；通过后才开始该新通信模型的独立训练。B3-10B本身没有执行这些步骤。

## 9. 等待确认的少量假设

建议用户确认以下三组合同后再实现：

1. **无线基线：** 是否以5.9 GHz LTE-V2X/NR-V2X PC5为研究抽象，并采用Ground 10 MHz？建议10 MHz；若要沿用生效值可选20 MHz。还需定义各节点带宽池是彼此正交/独立，还是共享频谱并建干扰。
2. **噪声/功率：** 是否先以`N0=−174 dBm/Hz`、固定总发射功率23 dBm、所有接收角色暂用有效NF=9 dB来对齐−95 dBm/10 MHz？若角色不同，请给SV/RSU/UAV NF或设备依据，并确认23 dBm是导出功率还是EIRP。
3. **Cloud及可靠性：** 是否以“无线AP接入+共享有线回传+40 GHz Cloud CPU”作为新正式目标（推荐，但需补齐AP关联、回传容量/并发规则及PHY可靠性目标）？在这些数据未齐之前是否保留当前每任务200 Mbps的方案A，仅作为抽象对照？Cloud是否严格只在Edge服务不可用时作为后备，仍需独立确认。

## 最终汇总

```ini
STAGE_B3_10B = PARTIAL
GROUND_BANDWIDTH_RECOMMENDATION = Ground 10 MHz（文献口径推荐）；20 MHz仍是Band 47可用配置，等待最终选择
NOISE_MODEL_RECOMMENDATION = N0=-174 dBm/Hz + 显式NF，按B_alloc=b*B_node积分；固定总任务功率23 dBm；NF=9 dB仅可作为−95 dBm/10 MHz的有效基准
CLOUD_ACCESS_MODEL_RECOMMENDATION = 目标模型B（无线接入+共享回传+共享40 GHz Cloud CPU）；缺参数前保留A为显式抽象对照，不把200 Mbps称为保证速率
PHYSICAL_REACHABILITY_MODEL = PHY条件化可靠goodput/outage + 任务上传QoS + 全局KKT资源可行性；不使用100/250/300 m作硬覆盖
SNR_THRESHOLD_JUSTIFIED = NO
PARAMETERS_REQUIRING_USER_DECISION = V2X接口与Ground带宽/频谱池；接收NF与23 dBm功率口径；Cloud接入/回传资源合同和可靠性目标
RECOMMENDED_NEXT_STEP = 先确认三组通信合同；随后在新隔离配置中联合实现噪声-速率-KKT及链路可达性，最后做一个确定性Training Slot核验
PPO_TRAINING_EXECUTED = NO
SYSTEM_MODEL_CHANGED = NO
FORMAL_CONFIG_CHANGED = NO
```

本报告完成推荐方案和需确认项；正式通信合同尚未冻结。等待最终确认后再实施。
