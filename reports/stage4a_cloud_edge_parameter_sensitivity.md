# 阶段4A：Cloud–Edge参数合理性与Cache收益敏感性分析

日期：2026-10-11  
范围：Stage3B有限Cloud CPU池、Stage3C1–3C4已有诊断及两个Validation诊断seed上的固定状态配对敏感性复算。

## 结论摘要

- 按本项目公式，Cloud的通信时延是 psi_c·D/R_cloud，基准等效速率为 200/11=18.18 Mbps；但Cloud上传能耗仍按真实合同速率200 Mbps计算。Edge则按任务分到的节点带宽、信道速率和CPU份额计算。
- Cloud的60 GHz是每个Slot内任务共享的总CPU池；Cloud上行仍是每任务独立200 Mbps，没有Cloud共享带宽池。它相对单个5 GHz Edge节点有较大的总CPU合同，但实际任务份额受Cloud同时执行的任务数和KKT分配影响。
- 60 GHz、psi_c=11和多数SV/RSU数值没有本项目目标硬件标定。200 Mbps和UAV 5 GHz/10 MHz存在其他论文的数值先例，但原场景不同，不能视作本系统的物理验证。
- 以相同的Validation任务、位置、预测数据和A/B缓存动作重算了9种有效参数配置、2个seed和2个分支，共36个20-slot分支窗口。所有资源约束通过，KKT未触及最大迭代次数，公式重算时延与存储总时延最大误差为0。
- #160每seed新增98/58条合法RSU1路径，但所有参数场景中A/B两个seed的18个配对 ΔG 均为负。按seed计的ΔG范围是 −0.351539 至 −0.137807；每seed固定Cache更新成本为0.245273。Cloud CPU 40 GHz使两seed合计净损失最小，但仍为负。这里没有把覆盖候选数当作Edge执行数。
- RSU独立带宽扰动未运行：配置校验要求 ground bandwidth 同时等于SV和RSU带宽，改RSU数值会连带改变SV参数。没有绕过校验或将耦合变化误报成RSU单因素效果。
- 本次没有发现需要立即更改系统公式或Matching的证据；但没有得到经过物理标定的参数集，因此不建议据此冻结物理环境或启动新PPO训练。

## 1. 配对条件与复算口径

输入固定为Validation episode seed 30020、30023的episode专属任务、车辆轨迹和预测文件，Slot 100–119。每个参数场景里A/B两分支使用完全相同的外生输入和UAV位置条件，分别重新运行Matching/KKT；没有将其他参数下的A结果借给当前参数场景。

- A：复用Stage3C3保存的基准Cache轨迹。
- B：Slot 100只执行一次Stage3C3合法候选 #160，RSU1从服务集合 {1,4,7,8,9} 改为 {0,1,2,4}，其后在该20-slot窗口中受控保持新配置。
- 缓存成本使用原公式。每个seed的B分支一次更新成本为0.2452729045，A为0；两分支移动成本均为0。
- 候选索引只用于识别原候选。不同参数副本按目标服务集合应用相同动作，没有依赖参数变化后的候选索引。
- 两个seed共568个实际任务。每个参数场景的A、B任务数相同。没有运行完整Episode、训练、正式Validation或Test。
- Stage3C0定义为独立Slot性能快照。本报告的20-slot任务效用与净收益是固定窗口逐Slot之和，不代表任务跨Slot完成状态，也不是完整Episode Return。

配对核验中，基准60 GHz场景A分支逐Slot任务效用与C2保存结果一致；A/B汇总Q及缓存成本复现C3报告到六位小数。3C4曾发现C3报告的B分支命中/候选差值93/53与当前源码按同输入重建的98/58有每seed 5项差异，C3没有保存相应逐任务清单。本报告统一采用当前源码的可行路径重建口径98/58；该差异不影响实际任务、A/B Matching复算及Q/G统计，但限制了与C3“命中计数”逐项对账。

## 2. 当前资源合同与模型公式

| 资源项 | 基准值及合同 | 代码中的资源语义 |
|---|---|---|
| Cloud CPU | finite_pool，60×10⁹ cycles/s | 当前Slot被分到Cloud的任务共享一个总CPU池；每个任务的分配由KKT计算。不是60 GHz/任务。 |
| Cloud上行 | 200 Mbps/任务 | 任务各自把200 Mbps传入Cloud公式；不由任务数分摊，也不受Edge带宽KKT约束。 |
| Cloud时延系数 | psi_c=11 | 只乘Cloud上传时延；不乘车辆上行能耗。 |
| RSU | 每节点5 GHz CPU、10 MHz带宽 | 每个RSU各有自己的Slot内CPU与带宽份额约束。 |
| UAV | 每架5 GHz CPU、10 MHz带宽 | 每架UAV各自共享节点资源。 |
| SV | 每个SV 1 GHz CPU、10 MHz带宽 | 每个SV各自共享节点资源。 |

当前源码公式（src/common/system_model.py::edge_execution、cloud_execution）：

    R_edge = b·B·log2(1 + P·h/σ²)
    T_edge = D/R_edge + L/(w·F_edge)
    E_edge = P·D/R_edge

    T_cloud = psi_c·D/R_cloud + L/f_cloud
    E_cloud = P·D/R_cloud

其中D为输入bit数，L为cycles数；b和w分别是Edge带宽与CPU份额。Cloud有限池由task_allocation中的Cloud CPU分配路径落实，带宽仍是按任务独立的标量合同。Slot Matching重评候选时会重分配受影响Cloud/Edge任务组资源，所以任务效用变化不等于单任务计算时延变化的简单求和。

任务效用代码为 omega_T·(1−T/T_ref)+omega_E·(1−E/E_ref)，两个权重之和为1；窗口任务效用总和记为Q，净值G=Q−CacheCost−MoveCost。由于效用同时取决于通信、计算和车辆能耗，且CPU/带宽会在同节点任务间重分配，Cloud或Edge的“CPU更快”不能单独决定Matching结果。

### Cloud相对Edge的优势和边界

1. **通信：** psi_c=11使Cloud基准200 Mbps合同在时延公式中等效为18.18 Mbps。RSU/UAV/SV速率受10 MHz节点带宽份额和车辆信道影响。3C4对真实新增RSU1任务测得独占10 MHz路径通常低于18.18 Mbps，许多路径更低；但这些路径并非在所有距离/任务上都相同。Cloud速率不随Cloud任务数分摊，是本模型中重要的通信优势假设。
2. **计算：** Cloud有60 GHz共享池，边缘每个RSU/UAV节点为5 GHz、每个SV为1 GHz。60 GHz是节点总量合同，不能解释为每任务60 GHz。基准窗口平均每槽约11.6个Cloud任务且池使用率100%，平均CPU量级约5.1 GHz/Cloud任务；具体分配不均匀，并随参数和Matching改变。Edge节点也各自独立分池，不是所有任务都能访问同一总池。
3. **竞争：** Cloud任务竞争60 GHz CPU，但不会在模型中竞争Cloud上行带宽；Edge任务同时竞争对应节点CPU和带宽。RSU/UAV出现满额份额的Slot不代表总系统资源违反约束。
4. **车辆能耗：** Cloud能耗按200 Mbps实际合同速率计算；psi_c只增加时延，不增加Cloud能耗。降低Cloud上行速率会同时增加Cloud通信时延和车辆上行能耗，改变psi_c则只改变Cloud通信时延。
5. **Matching目标：** 每个候选均按重新分配后的整体任务效用比较。3C4对5个新增缓存任务施加单任务RSU1约束并重新优化受影响Cloud/RSU任务组，整个Slot的Q都下降；这支持所测状态下Cloud联合效用较高，但不是所有任务或多任务联合改派的全局最优证明。

配置工作负载包络（stage3_cloud_finite_60ghz.toml、scenario_generator.sample_tasks）为输入400–6000 KB，即3.2–48 Mbit；计算量1.152–52.8 Gcycles。单任务独占60 GHz时计算项约0.019–0.88 s，独占5 GHz时约0.23–10.56 s；基准Cloud通信项约0.176–2.64 s。真实计算份额与Edge通信项会因同Slot竞争而变。时延可超过1秒；Stage3C0时间语义不把1秒Slot当作执行截止或跨Slot队列。这些量级与公式单位相符，但不能证明参数对应目标硬件。

## 3. 文献依据与参数状态

| 参数 | 公式/文献依据 | 数值依据分类 | 场景是否匹配、物理标定状态 |
|---|---|---|---|
| Cloud CPU 60 GHz总池 | 11.pdf第4–5页支持Cloud按任务计算L/F_c的形式，不定义60 GHz总池或该共享合同。Stage3B在此基础上增加有限池资源合同。 | 本项目实验设计假设。已有Cloud文献中的20 GHz总池、每车U[2,4] GHz不能推出60 GHz。 | 文献场景、任务和系统规模不同；无目标Cloud实例规格或并发服务实测。未标定。 |
| Cloud 200 Mbps/任务 | 11.pdf第7页支持Cloud独立蜂窝链路形式，不给本项目数值。Al-Shareeda等Table IV有200 Mbps Cloud平均上行数值先例。 | 有其他论文的同数值先例；本项目将其用作每任务独立合同是额外假设。 | 论文的车辆/任务/Cloud模型不同；平均速率不证明目标网络对每个并发任务保证200 Mbps，也不证明无接入拥塞。未标定。 |
| psi_c=11 | 11.pdf第7–8页给出 psi_c·D/R_c 形式并要求 psi_c≥1；没有给11。 | 公式形式有原文依据，具体数值无直接文献支持。 | 未发现可把11映射到端到端Cloud网络实测的依据；未标定。 |
| RSU 5 GHz、10 MHz | 11.pdf第4、6页支持节点CPU/带宽份额约束，第5–7页支持Edge时延公式；没有给RSU数值。 | 项目基线/实验假设。 | 没有目标RSU规格或直接对应数值文献。未标定。 |
| SV 1 GHz、10 MHz | 同上：论文支持共享公式，不给SV数值。 | 项目基线/实验假设。 | 没有目标SV规格或直接对应数值文献。未标定。 |
| UAV 5 GHz、10 MHz | 11.pdf支持空地节点资源模型，不给这些数值。Li等2021 Table 1存在5 GHz/10 MHz的UAV数值先例。 | 有数值先例但本项目取值移植自不同场景。 | 该论文的地面用户、任务量、cycles/bit、时隙和信道都不同；不是本项目完整场景或硬件标定。 |
| Cloud/Edge时延公式 | 11.pdf第4–8页给出任务、CPU/带宽共享、Edge速率、Cloud独立链路、Cloud时延/能耗及结果下行忽略等结构。 | 公式/结构有原文依据。 | 本地11.pdf第24–26页描述仿真场景，但没有可核验的资源数值表；第26页提示参数表另列。 |

数值来源核对以已有 reports/stage3_parallel_parameter_evidence.md、reports/cloud_edge_system_model_analysis.md 和 reports/cloud_edge_model_stage2b_design.md 为依据。Al-Shareeda接受稿与Li等论文只作为文献数值先例，均不是本项目目标网络的测量。没有发现一组同时覆盖Cloud、RSU、SV、UAV且经过硬件标定的参数。

## 4. 探索性单因素敏感性范围

| 场景 | Cloud总CPU | Cloud上行/任务 | psi_c | RSU CPU | RSU带宽 |
|---|---:|---:|---:|---:|---:|
| BASE | 60 GHz | 200 Mbps | 11 | 5 GHz | 10 MHz |
| Cloud_CPU_40GHz | 40 GHz | 200 Mbps | 11 | 5 GHz | 10 MHz |
| Cloud_CPU_80GHz | 80 GHz | 200 Mbps | 11 | 5 GHz | 10 MHz |
| Cloud_UL_100Mbps | 60 GHz | 100 Mbps | 11 | 5 GHz | 10 MHz |
| Cloud_UL_400Mbps | 60 GHz | 400 Mbps | 11 | 5 GHz | 10 MHz |
| psi_5 | 60 GHz | 200 Mbps | 5 | 5 GHz | 10 MHz |
| psi_20 | 60 GHz | 200 Mbps | 20 | 5 GHz | 10 MHz |
| RSU_CPU_2p5GHz | 60 GHz | 200 Mbps | 11 | 2.5 GHz | 10 MHz |
| RSU_CPU_10GHz | 60 GHz | 200 Mbps | 11 | 10 GHz | 10 MHz |

40/80 GHz、100/400 Mbps、psi=5/20、RSU CPU 2.5/10 GHz均是预先规定的探索性压力测试值，不是经物理标定的真实范围。Cloud上行变化时有效时延速率R/psi分别为9.09、18.18、36.36 Mbps；psi变化时有效速率分别为40、18.18、10 Mbps。psi变化不会改变能耗公式中的R_cloud。

RSU带宽5/20 MHz场景**未运行**。project_config.validate_config要求communication.ground.bandwidth_hz与nodes.sv.bandwidth_hz及nodes.rsu.bandwidth_hz相等；仅改RSU带宽会连带改变SV带宽，因此不满足单因素设计。没有绕过配置约束，也没有把该耦合场景标称为RSU单因素。RSU CPU扰动可单独通过现有配置合同，故已运行。

## 5. 单因素配对结果

下表Q为两个seed合计的20-slot窗口任务效用；两seed总任务数568。ΔG按每个seed单独配对后再求和。每seed B固定更新成本为0.245273，所以两seedΔCacheCost均为0.490546。

| 场景 | Q_A | Q_B | ΔQ | ΔCacheCost | ΔG | 新增路径最终Edge/Cloud | 节点执行总数A → B（Local/SV/RSU/UAV/Cloud） |
|---|---:|---:|---:|---:|---:|---:|---|
| BASE | 516.223842 | 516.168784 | −0.055057 | 0.490546 | −0.545603 | 2 / 154 | 0/0/30/73/465 → 0/0/27/73/468 |
| Cloud_CPU_40GHz | 502.801208 | 502.802408 | +0.001200 | 0.490546 | −0.489345 | 7 / 149 | 0/4/36/81/447 → 0/4/39/81/444 |
| Cloud_CPU_80GHz | 523.498917 | 523.499423 | +0.000506 | 0.490546 | −0.490040 | 1 / 155 | 0/0/18/66/484 → 0/0/19/66/483 |
| Cloud_UL_100Mbps | 502.830086 | 502.760481 | −0.069605 | 0.490546 | −0.560151 | 4 / 152 | 0/1/34/75/458 → 0/2/34/75/457 |
| Cloud_UL_400Mbps | 523.088228 | 523.083198 | −0.005030 | 0.490546 | −0.495576 | 1 / 155 | 0/1/24/71/472 → 0/1/23/71/473 |
| psi_5 | 522.901758 | 522.896685 | −0.005073 | 0.490546 | −0.495619 | 1 / 155 | 0/1/24/71/472 → 0/1/23/71/473 |
| psi_20 | 506.421296 | 506.339888 | −0.081408 | 0.490546 | −0.571954 | 2 / 154 | 0/1/32/75/460 → 0/2/31/75/460 |
| RSU_CPU_2p5GHz | 515.076208 | 515.076208 | 0 | 0.490546 | −0.490546 | 0 / 156 | 0/0/7/73/488 → 0/0/7/73/488 |
| RSU_CPU_10GHz | 517.299266 | 517.271971 | −0.027295 | 0.490546 | −0.517841 | 5 / 151 | 0/0/36/73/459 → 0/0/37/73/458 |

Q_A、Q_B和净收益的严格配对仍以每seed结果为准。下表给出每seed的ΔQ及ΔG，显示两seed间变化并不一致；18/18个ΔG均小于0。

| 场景 | Seed 30020 ΔQ / ΔG | Seed 30023 ΔQ / ΔG |
|---|---:|---:|
| BASE | −0.070605 / −0.315878 | +0.015548 / −0.229725 |
| Cloud_CPU_40GHz | −0.106266 / −0.351539 | +0.107466 / −0.137807 |
| Cloud_CPU_80GHz | 0 / −0.245273 | +0.000506 / −0.244767 |
| Cloud_UL_100Mbps | −0.100091 / −0.345364 | +0.030486 / −0.214787 |
| Cloud_UL_400Mbps | −0.017694 / −0.262967 | +0.012664 / −0.232609 |
| psi_5 | −0.017803 / −0.263076 | +0.012730 / −0.232543 |
| psi_20 | −0.100104 / −0.345377 | +0.018696 / −0.226577 |
| RSU_CPU_2p5GHz | 0 / −0.245273 | 0 / −0.245273 |
| RSU_CPU_10GHz | −0.061370 / −0.306642 | +0.034074 / −0.211199 |

#160带来的合法新增RSU1路径在两个seed中始终为98/58，合计156；新增服务分布为seed30020的服务0/2各17/81，seed30023各18/40。按最终执行节点计数，真正选择这些新增路径的Edge任务从BASE的2个到CPU40场景的7个；其余新增候选仍选择Cloud。候选可行/预测覆盖数量不等于实际Edge卸载量。

## 6. 时延、资源竞争与运行开销

| 场景 | Cloud平均通信/计算时延 A→B（秒） | Edge平均通信/计算时延 A→B（秒） | 平均车辆能耗 A→B（J/任务） | Cloud池平均使用 | RSU0 CPU/BW利用 A→B | RSU1 CPU/BW利用 A→B | Matching均时 A→B（ms/slot） |
|---|---|---|---|---:|---:|---:|---:|
| BASE | 1.325/3.050 → 1.323/3.064 | 0.702/3.547 → 0.687/3.580 | 0.045→0.044 | 60/60 GHz | 65%→60% | 10%→7.5% | 26.36→33.54 |
| Cloud_CPU_40GHz | 1.330/4.408 → 1.320/4.342 | 0.838/3.761 → 0.933/3.822 | 0.055→0.059 | 40/40 GHz | 75%→75% | 15%→22.5% | 26.60→35.34 |
| Cloud_CPU_80GHz | 1.334/2.413 → 1.336/2.410 | 0.519/3.315 → 0.527/3.297 | 0.036→0.036 | 80/80 GHz | 42.5%→42.5% | 2.5%→5% | 24.28→31.90 |
| Cloud_UL_100Mbps | 2.635/3.002 → 2.627/2.987 | 0.785/3.618 → 0.827/3.656 | 0.069→0.071 | 60/60 GHz | 72.5%→70% | 12.5%→15% | 25.33→31.40 |
| Cloud_UL_400Mbps | 0.672/3.151 → 0.671/3.144 | 0.574/3.349 → 0.578/3.389 | 0.029→0.029 | 60/60 GHz | 55%→52.5% | 5%→5% | 25.36→33.00 |
| psi_5 | 0.611/3.151 → 0.610/3.144 | 0.574/3.349 → 0.578/3.389 | 0.040→0.040 | 60/60 GHz | 55%→52.5% | 5%→5% | 25.42→33.82 |
| psi_20 | 2.399/3.013 → 2.400/3.026 | 0.751/3.626 → 0.759/3.587 | 0.048→0.048 | 60/60 GHz | 70%→70% | 10%→7.5% | 25.34→33.10 |
| RSU_CPU_2p5GHz | 1.326/3.207 → 1.326/3.207 | 0.588/3.771 → 0.588/3.771 | 0.037→0.037 | 60/60 GHz | 17.5%→17.5% | 0%→0% | 27.81→33.52 |
| RSU_CPU_10GHz | 1.331/3.026 → 1.328/3.010 | 0.759/2.953 → 0.804/2.937 | 0.049→0.050 | 60/60 GHz | 75%→72.5% | 12.5%→17.5% | 25.32→32.81 |

通信/计算均值按最终选择该类节点的任务加权；A/B任务子集不同，均值不能当作同一批任务的配对因果差值。Cloud池在每个测试窗口都满额使用，说明它是活跃约束；不能单凭100%利用率判定合同错误。RSU CPU/带宽利用为Slot内份额的窗口平均，峰值在有任务的槽可到100%。UAV0/1也是各自独立的共享节点，执行任务计数列在上一表；现有聚合指标不支持将RSU数值外推为所有Edge节点的平均利用率。

时延分解使用原模型公式；Cloud仍为psi_c·D/R_cloud和L/f_cloud，车辆能耗仍为P·D/R_cloud。验证的最大公式重算误差为0。没有在报告中把psi_c吸收到Cloud能耗或Edge路径公式中。

运行开销按参数场景汇总两seed的40次Matching运行；平均每slot约24.28–35.34 ms。基准第一次A测量包含JIT冷启动，已排除，改用同状态预热复测的A/B耗时。候选评价总数在各场景约6.3k–9.2k，KKT求解约6.8k–9.4k；B通常较A更多，因为新增缓存候选扩大候选评价。所有窗口的搜索停止原因为converged_by_no_improvement；KKT最大迭代命中为0，资源约束均通过。此范围内未见Matching引入明显的多秒级运行开销。

## 7. 参数敏感性解释及证据边界

- **Cloud CPU竞争：**40 GHz下两个seed合计新增候选Edge执行数为7，优于60 GHz下的2和80 GHz下的1；合计ΔG为−0.489345，是测试值中净损失最小的场景，但仍为负。CPU40的结果来自seed差异显著的ΔQ（30020 −0.106266、30023 +0.107466），不能解释为普遍收益。
- **Cloud通信合同：**100 Mbps时Cloud通信变慢且车辆能耗增加，新增Edge选择为4，ΔG为−0.560151；400 Mbps时新增Edge选择为1，ΔG为−0.495576。psi=20也降低Cloud等效时延速率，但不改变能耗速率；其ΔG为−0.571954，psi=5时为−0.495619。此处的表现同时受任务路由和资源重分配影响，不是单一时延项的独立因果分解。
- **RSU计算：**2.5 GHz时没有新增路径任务选择RSU，A/B的Q完全相同，净损失只等于Cache成本；10 GHz时新增路径Edge选择增至5，但ΔG仍为−0.517841。当前RSU带宽没有单独扰动，无法判断RSU带宽的敏感性。
- **更新成本与候选转化：**每seed固定Cache成本0.245273。测试中单seed最大ΔQ为+0.107466，仍小于单次更新成本；多数新增RSU1候选最终由Cloud执行。即使Cloud资源改变确实令候选迁移，当前探索值下也未抵消固定更新成本。
- **正收益边界：**9个有效配置、18个seed配对中没有正ΔG；这只说明指定候选、两个seed和Slot100–119窗口在这些假设参数下没有净收益，不表示所有缓存更新或负载条件都无价值。CPU40改善幅度较小，不能据此选参。
- **Matching质量：**有限C4重求解在5个单任务合法Edge约束案例中均降低整Slot Q。Stage4A也未发现与该结论相反的证据；没有做全局穷举或多任务联盟改派，不能证明Matching全局最优，也没有足够证据要求修改Transfer-Exchange/KKT。
- **结论分类：**探索值范围内观察到Cache净收益均为负（情况A仅对本样本成立）；路由、Q和ΔG数值随Cloud/RSU参数变化（存在定量敏感性），但没有发生净收益符号翻转；没有找到既有物理依据又出现正净收益的配置（情况D未建立）。所有探索扰动都不能替代目标设备标定。

## 8. 环境冻结与训练建议

当前可以保留Stage3B公式和已实现的有限Cloud池作为研究基线继续分析；本轮未发现需要立即改系统方程、Reward、Matching或PPO的证据。未来若需要声称环境具有物理代表性，至少还需要目标Cloud的CPU配额/并发分配资料、Cloud per-task吞吐及共享语义、psi_c对应的端到端时延测量，以及RSU/SV/UAV节点规格或链路实测。若要独立研究RSU带宽，需先让配置结构表达RSU与SV带宽独立，之后再做相应单因素对照；本阶段没有修改该配置结构。

两个Validation seed、一个Cache候选及一个20-slot窗口不足以证明Cache机制总体价值、确定物理参数或判断新环境PPO的最优超参数。建议先补齐物理参数依据，并在有代表性的固定配对场景中扩充验证；在此之前不冻结为“已标定正式环境”，也不以本报告作为启动新PPO训练的充分依据。

## 9. 实现与数据依据

- 配置：configs/proposed/stage3_cloud_finite_60ghz.toml。
- 公式：src/common/system_model.py::edge_execution、cloud_execution、task_utility、cache_update_cost。
- Cloud CPU池及任务Matching/KKT：src/decision/task_allocation.py；Slot输出与资源诊断：src/decision/slot_environment.py。
- 单因素约束：src/common/project_config.py::validate_config；RSU带宽与SV带宽共用ground bandwidth约束约在1150–1158行。
- 参数文献核对：reports/stage3_parallel_parameter_evidence.md、reports/cloud_edge_system_model_analysis.md、reports/cloud_edge_model_stage2b_design.md。
- 配对状态与此前诊断：reports/stage3c1_cloud_edge_structural_comparison.md、reports/stage3c2_cache_uav_long_horizon_counterfactual.md、reports/stage3c3_cache_persistence_diagnostic.md、reports/stage3c4_cache_offloading_value_gap.md。
- 11.pdf：第4–8页为任务资源、CPU/带宽共享、Cloud/Edge时延和能耗公式；第24–26页为仿真场景叙述，无可核验的本项目资源参数表。

    STAGE4A_DONE = YES (有效单因素范围完成；RSU独立带宽部分未完成)
    BASELINE_REPRODUCED = YES
    PARAMETER_EVIDENCE_REVIEWED = YES
    CLOUD_CPU_SENSITIVITY_COMPLETE = YES
    CLOUD_COMMUNICATION_SENSITIVITY_COMPLETE = YES (Cloud上行与psi_c)
    RSU_RESOURCE_SENSITIVITY_COMPLETE = PARTIAL (CPU完成；独立带宽未运行)
    PAIRED_INPUT_IDENTICAL = YES
    RESOURCE_CONSTRAINTS_PASS = YES (36/36分支窗口)
    CACHE_NET_BENEFIT_SENSITIVITY = NEGATIVE_ALL_18_TESTED_PAIRS; deltaG [-0.351539, -0.137807]
    PHYSICALLY_CALIBRATED_PARAMETER_SET_FOUND = NO
    SYSTEM_MODEL_CHANGE_REQUIRED = NO (本轮未发现即时公式/Matching改动依据)
    READY_TO_FREEZE_ENVIRONMENT = NO (物理标定未完成)
    READY_FOR_NEW_PPO_TRAINING = NO

