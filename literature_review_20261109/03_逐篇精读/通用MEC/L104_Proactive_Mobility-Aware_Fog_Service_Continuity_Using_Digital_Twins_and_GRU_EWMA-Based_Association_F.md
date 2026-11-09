# L104 — Proactive Mobility-Aware Fog Service Continuity Using Digital Twins and GRU EWMA-Based Association Forecasting

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2026
- 刊物/会议：待核
- DOI：10.32604/cmc.2026.079991
- PDF页数：34
- 原文：`C:\Users\tangning\Desktop\文献\Proactive Mobility-Aware Fog Service Continuity Using Digital Twins and GRU–EWMA-Based Association Forecasting.pdf`
- 领域分类：**C**；依据：摘要/模型段 MEC=4, vehicle=0, UAV=0
- 主研究方向：P2；次方向：无；依据：scores={'1': 12, '2': 93, '3': 11}; title=Proactive Mobility-Aware Fog Service Continuity Using Digital Twins and GRU EWMA

## 2. 研究场景与问题

### 摘要窗口

未定位到章节；需回原文核查。

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

未定位到章节；需回原文核查。

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p7起）

formulation
[29] ●Multi-objectivefogscheduling ●TRAPheuristic ●Delay↓ ●iFogSim
●Energy↓ ●Synthetic workload
[17] ●Mobility-supportfogscheduling ●PROMOproactiveallocation ●Servicelatency↓ ●Simulation
●Mobility prediction
[26] ●Vehicularfog(V2X) ●Dynamictaskmigration ●Latency↓ ●Vehicularfogtestbed
●Taskallocationundermobility
[19] ●6Gedgeoffloading ●DigitalTwinEdgeNetwork ●Offloadlatency↓ ●MATLABsimulation
●Uncertainty-awarecontrol ●Actor–criticRL
●Lyapunovoptimization
[38] ●IoToffloading ●Learning-basedoffloading ●Throughput↑ ●Theory
●Satellite/UAV-assisted ●Delay↓ ●Simulation
[35] ●Vehicularedgecomputing ●M ul ti-agen tdeepRL ●Cost↓ ●Veins/MATLAB
co-simulation●Large-scalecoordination ●DigitalTwins ●Latency↓
[20] ●Edgescheduling ●DT-assistedQ-learning ●Convergencespeed↑ ●Pythonsimulator
●Largeactionspace
[30] ●I oT –fogr eso ur cesched uling ●Hybridmeta-heuristic ●Delay↓ ●Simulation
●ABC+PSO+GA ●Energy↓
[36] ●UAV-assistededgecomputing ●FederateddeepRL ●Tasksuccess↑ ●Pythonsimulator
●Agriculturedeployments ●DT-basedenvironment ●Delay/Energy↓
[37] ●Securemulti-taskscheduling ●RL-basedscheduling ●Energyefficiency↑ ●Prototype
●InternetofEverything ●Encryptionintegration ●Security↑
[33] ●Infrastructuremanagement ●HybridAI-IoT-DTframework ●Responsetime↓ Simulation
●La tency&p rivacyinI oT ●CNN/LSTM+FL ●Maintenancecost↓
●DTsimulation ●Falsepositives↓
[34] ●Trafficsignaloptimization ●GRUcongestionforecasting ●Predictiveaccuracy↑ Simulation
●Sustainableurbanmobility ●DTvalidation ●Latency↓
Signaloptimisation ●CO2 emissions↓
Our
Work
●DT-enhancedfog
scheduling
●DigitalTwinplanning ●Delay↓
●MobFogSim-based
simulation●Mobility-awareQoS
reliability
●Confidence-gatedprediction ●Energy↓
●Load-awarecontrol ●Successrate↑
●DToverhead↓
Further, existing approaches to fog scheduling can be

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p5起）

solutions, some researchers considered network handover
coordination.Ngoetal.[ 28]proposedacoordinatedcontainermigrationandbasestationhandoverstrategy
in MEC to maintain service continuity during user movement. Their method triggers service migration in
tandemwithcellularhandovers,minimizingtaskinterruptionsformobileusers.
Another line of work focuses on multi-objective and meta-heuristic solutions for fog scheduling.
TraditionalschedulingmustoftenbalancemultipleQoSmetrics,leadingtoNP-hardoptimizationproblems.
Kaur et al. [29] formulated fog scheduling as a multi-objective optimization and proposed Task-Resource
Adaptive Pairing (TRAP). TRAP uses a batching, ranking, and priority-based heuristic to pair tasks with
fog nodes, reducing the search space and simultaneously minimizing delay, energy consumption, and
cost. Implemented in iFogSim, TRAP achieved reductions in task delay and energy usage compared to
naive scheduling. Liu et al. [30] combined bio-inspired algorithms for efficient scheduling by integrat-
ing Particle Swarm Optimisation (PSO) and Genetic Algorithm into an Artificial Bee Colony scheme.
Their hybrid algorithm optimizes task allocation across fog clusters, reducing average service latency and
energyconsumption.
2.2 DigitalTwin-EnabledIntelligentOrchestrationandEmergingWork
While the above studies improve scheduling through optimization, learning, and mobility

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p2起）

simulation, and opti-
mization [3,4]. While DTs were initially adopted in industrial contexts, they are increasingly used in
networking and edge/fog computing environments to support latency-aware and context-aware decision-
making [5]. Recent studies in [6–9] show that DT-assisted orchestration can improve real-time scheduling
and resource allocation when accurate synchronisation and predictive analytics are available. Nevertheless,
a gap remains in DT-enabled fog control for mobile users, as existing approaches often treat mobility
predictionandfogloadpredictionseparately.Thisshortcomingtriggersproactiveactionswithoutanexplicit
uncertainty signal or lacks a clear hierarchical mechanism to scale decisions across local domains and
inter-domaincoordination.
This paper proposes a hierarchical DT-enhanced fog orchestration framework in which Local Fog
Orchestrators (LFOs) manage fine-grained user and fog twins. The LFO executes real-time control, while
a Global Fog Orchestrator (GFO) maintains an abstracted system-wide view for policy coordination and
inter-domain support. Within each LFO domain, a predictive control pipeline is introduced that integrates
(i) GRAIN, a GRU-EWMA hybrid predictor that stabilises noisy measurements via EWMA and encodes
short-horizontemporaldynamicsviaaGRU;(ii)anentropy-basedconfidencescoretoquantifythecertainty
ofpredictedassociationtendencies;(iii)EW

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 是 |
| 服务更新周期 | 是 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 是 |
| 位置预测误差 | 是 |
| 服务迁移 | 是 |
| 活动任务连续执行 | 是 |
| 迁移传输 | 未提及 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 是 |
| 信道动态变化 | 未提及 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=是；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=是；服务迁移=是；活动任务连续=是；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=是；信道动态=未提及 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：None（PDF页）
- 问题窗口：7（PDF页）
- 方法窗口：5（PDF页）
- 仿真窗口：2（PDF页）
- 结论窗口：28（PDF页）

### 关键词命中片段

- p1: demayexperiencedegradedlinkqualityand increasedpropagationdelay.Thisleadstolongerrespon setimes,serviceinterruption,andmisseddeadlines if service placement remains static [1]. Consequently, mobility-aware offloading and service migration are essentialforsustainingqualityofservice(QoS)inmobilefogenvironments[ 2]. A key difficulty is that mobility an
- p5: me researchers considered network handover coordination.Ngoetal.[ 28]proposedacoordinatedcontainermigrationandbasestationhandoverstrategy in MEC to maintain service continuity during user movement. Their method triggers service migration in tandemwithcellularhandovers,minimizingtaskinterruptionsformobileusers. Another line of work focuses on multi-

## 12. 阅读闭合状态

- 全文文本：已提取（34页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
