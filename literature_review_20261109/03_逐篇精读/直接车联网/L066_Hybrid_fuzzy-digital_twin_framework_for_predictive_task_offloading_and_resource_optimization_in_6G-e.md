# L066 — Hybrid fuzzy-digital twin framework for predictive task offloading and resource optimization in 6G-enabled internet of vehicles

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：Daoyun Tang
- 年份：2026
- 刊物/会议：Vehicular Communications
- DOI：10.1016/j.vehcom.2026.101032
- PDF页数：24
- 原文：`C:\Users\tangning\Desktop\文献\02_移动性感知的服务迁移与传输资源协同\Hybrid fuzzy-digital twin framework for predictive task offloading and resource optimization in 6G-enabled internet of vehicles.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=29, UAV=1
- 主研究方向：P1；次方向：2,3；依据：scores={'1': 24, '2': 23, '3': 12}; title=Hybrid fuzzy-digital twin framework for predictive task offloading and resource 

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model, encompassing the 6G- 
IoV architecture and the DT-enabled edge layer. Section 4 delineates the 
Hybrid Fuzzy-DT offloading mechanism, specifying predictive input 
generation, fuzzy rule augmentation, and proactive migration logic. 
Section 5 delineates the simulation environment utilizing OMNeT++
with a bespoke DT engine on actual urban maps, succeeded by an 
exhaustive performance assessment against leading benchmarks. Ulti-
mately, Section 6 ends the manuscript and delineates prospective study 
avenues.
2. Related work
Numerous studies on resource orchestration, edge computing, and 
predictive control mechanisms have been prompted by the quick 
development of 6G-enabled IoV. Three thematic categories—i) reactive 
and optimization-driven techniques, (ii) partially predictive or learning- 
assisted strategies, and (iii) fully predictive Digital Twin–empowered 
frameworks—ar

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model, encompassing the 6G- 
IoV architecture and the DT-enabled edge layer. Section 4 delineates the 
Hybrid Fuzzy-DT offloading mechanism, specifying predictive input 
generation, fuzzy rule augmentation, and proactive migration logic. 
Section 5 delineates the simulation environment utilizing OMNeT++
with a bespoke DT engine on actual urban maps, succeeded by an 
exhaustive performance assessment against leading benchmarks. Ulti-
mately, Section 6 ends the manuscript and delineates prospective study 
avenues.
2. Related work
Numerous studies on resource orchestration, edge computing, and 
predictive control mechanisms have been prompted by the quick 
development of 6G-enabled IoV. Three thematic categories—i) reactive 
and optimization-driven techniques, (ii) partially predictive or learning- 
assisted strategies, and (iii) fully predictive Digital Twin–empowered 
frameworks—are used to categorize the literature in order to clearly 
present the suggested framework.
2.1. Optimization-driven and reactive offloading strategies
Optimization and reactive decision-making processes are the main-
stays of early 6G-IoV resource management research. While robust for-
mation management under adversarial settings has been investigated in 
distributed multi-agent systems [20], sustainable and green IoV 
networking paradigms place an emphasis on resource efficiency and 
energy-aware communication [19]. In decentralized IoV environments, 
trust and coordination concerns are addressed by blockchain-enabled 
vehicle designs [21].
Comprehensive surveys highlight optimization-based vehicle off-
loading solutions [27], while [26] formulates resource allocation and 
offloading for 6 G smart edge systems as joint optimization problems. 
System flexibility is further improved by coo

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p2起）

optimization problems. 
System flexibility is further improved by cooperative multi-type task 
offloading based on reinforcement learning [28]. In [32], 
intersection-centric fuzzy offloading decisions with an emphasis on 
localized latency reduction are investigated.
Despite improving energy and delay measures, these methods are 
still essentially reactive. Without specifically anticipating future topol-
ogy evolution, wireless channel deterioration, or edge resource varia-
tion, decisions are made based on instantaneous state measurements. 
Therefore, in ultra-dense and high-mobility 6G-IoV scenarios, task 
migration failures and instability may occur.
Gap 1. Under rapidly changing topological dynamics, reactive opti-
mization and fuzzy-only techniques lack proactive migration 
D. Tang et al. Vehicular Communications 59 (2026) 101032 
2 
capabilities and foresight.
2.2. Partially predictive and learning-assisted frameworks
Learning-driven and collaborative edge-end scheduling algorithms 
have been designed to handle dynamic vehicular settings. In intelligent 
transport systems, edge-end collaboration based on reinforcement 
learning increases the effectiveness of multi-task scheduling [25]. In 6 G 
IoV, cybertwin-based DRL scheduling algorithms combine resource 
allocation with virtual representations [30]. Performance improvements 
in heterogeneous wireless settings are demonstrated by QoS-oriented 
cooperative MEC offloading in multi-UAV networks [33,34].
In addition, recent studies explore explainable AI-driven orchestra-
tion and intelligent network slicing for 6 G IoT systems [24], as well as 
adaptive scheduling in fog-cloud systems with new computational par-
adigms [31]. The viability of real-time computation offloading is further 
supported by effective neura

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p1起）

solution 
environment reconstruction appropriate for DT applications [6], joint 
radar-communication waveform designs enhance environmental 
awareness and vehicle coordination [5]. These sensing-driven im-
provements lay the technological groundwork for predictive control and 
orchestration by enabling cars and roadside units to create lifelike vir-
tual representations of actual traffic situations.
Despite these developments, high mobility, quickly shifting topol-
ogy, sporadic THz connections, and unstable edge resource availability 
continue to be the key limitations on compute offloading in 6G-IoV. 
Offloading techniques based on learning and optimization have been 
thoroughly studied [7–9]. Learning-driven V2V offloading adjusts to 
changing load conditions [8], while joint wireless-computation opti-
mization increases energy efficiency [7]. Resource consumption is 
further improved via scheduling-based methods and reinforcement 
learning [9]. Nevertheless, these frameworks are primarily reactive, 
depending on real-time network status measurements without explicit 
prediction of future topology changes.
In vehicular edge computing, fuzzy logic has been used to handle 
ambiguity. Under unclear workload and channel conditions, fuzzy- 
* Corresponding authors.
E-mail addresses: hfuttdy@163.com (D. Tang), chengwanglin@chinesemail.net (C. Lin). 
Contents lists available at Sci

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

simulations utilising OMNeT++ combined with a DT 
engine. In conclusion, this predictive paradigm guarantees robust, energy-efficient, and secure compute off-
loading for next-generation intelligent transportation systems as the first merger of fuzzy logic and digital twin in 
6G-IoV.
1. Introduction
Ultra-reliable low-latency communications (URLLC), integrated 
sensing, and AI-driven orchestration are being made possible by the IoV 
transition to 6G-native architectures, which is changing intelligent 
transportation systems [1,2]. In contrast to 5 G, 6 G networks allow 
extreme data speeds and sub-millisecond responsiveness by introducing 
terahertz (THz) communications, joint communication and sensing ca-
pabilities, and AI-native resource management [3,4]. These features are 
crucial for latency-critical applications like immersive augmented re-
ality, Level-5 autonomous driving, and cooperative perception in auto-
motive contexts.
The capabilities of 6G-IoV are further expanded by recent de-
velopments in integrated sensing and communication (ISAC). While 
channel-state-information-based imaging allows for high-resolution 
environment reconstruction appropriate for DT applications [6], joint 
radar-communication waveform designs enhance environmental 
awareness and vehicle coordination [5]. These sensing-driven im-
provements lay the technological groundwork for predictive

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 是 |
| 服务更新周期 | 部分 |
| 决策时间尺度自适应 | 部分 |
| 车辆移动性 | 是 |
| 位置预测误差 | 是 |
| 服务迁移 | 未提及 |
| 活动任务连续执行 | 是 |
| 迁移传输 | 未提及 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 是 |
| 信道动态变化 | 是 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=部分；时间尺度自适应=部分 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=是；服务迁移=未提及；活动任务连续=是；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=是；信道动态=是 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：2（PDF页）
- 方法窗口：1（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：1（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（24页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
