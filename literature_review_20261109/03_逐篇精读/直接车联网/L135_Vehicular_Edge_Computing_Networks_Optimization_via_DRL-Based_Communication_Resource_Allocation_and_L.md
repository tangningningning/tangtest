# L135 — Vehicular Edge Computing Networks Optimization via DRL-Based Communication Resource Allocation and Load Balancing

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2025
- 刊物/会议：IEEE Transactions on Mobile Computing
- DOI：10.1109/TMC.2025.3559707
- PDF页数：16
- 原文：`C:\Users\tangning\Desktop\文献\Vehicular_Edge_Computing_Networks_Optimization_via_DRL-Based_Communication_Resource_Allocation_and_Load_Balancing.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=40, UAV=2
- 主研究方向：P1；次方向：3；依据：scores={'1': 22, '2': 3, '3': 14}; title=Vehicular Edge Computing Networks Optimization via DRL-Based Communication Resou

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model and
problem formulation are given in Section III. Section IV pro-
poses the DRLCL algorithm. Simulation results and discussions
are shown in SectionV. Finally, SectionVI draws the conclusion
of this paper.
II. R ELA TEDWORKS
In recent years, MEC technology has been extensively
adopted in VEC networks to alleviate the inherent computa-
tional limitations of onboard vehicle systems. Numerous studies
have focused on the computation offloading problems, explor-
ing aspects including task destination selection, communication
resource allocation, and load balancing.
A. Task Destination Selection
Regarding task destination selection, various strategies have
been developed to ensure QoS-guaranteed computation offload-
ing in IoV scenarios, leveraging a variety of auxiliary compu-
tation nodes. For instance, [12] and [13] propose a learning-
based vehicle-to-vehicle (V2V) offloading

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model and
problem formulation are given in Section III. Section IV pro-
poses the DRLCL algorithm. Simulation results and discussions
are shown in SectionV. Finally, SectionVI draws the conclusion
of this paper.
II. R ELA TEDWORKS
In recent years, MEC technology has been extensively
adopted in VEC networks to alleviate the inherent computa-
tional limitations of onboard vehicle systems. Numerous studies
have focused on the computation offloading problems, explor-
ing aspects including task destination selection, communication
resource allocation, and load balancing.
A. Task Destination Selection
Regarding task destination selection, various strategies have
been developed to ensure QoS-guaranteed computation offload-
ing in IoV scenarios, leveraging a variety of auxiliary compu-
tation nodes. For instance, [12] and [13] propose a learning-
based vehicle-to-vehicle (V2V) offloading architecture, which
capitalizes on the capabilities and trustworthiness of service-
enabled vehicles with spare computation power. To facilitate
V2V offloading, [14] presents a trajectory prediction model
for task vehicles based on long short-term memory (LSTM).
However, V2V links are prone to frequent disruptions, leading to
frequent link switching and decreased response speed. Further-
more, [15] and [16] consider the drone-assisted VEC networks,
where drones hover to provide additional computing capabili-
ties. However, the mobile characteristics of drones are not fully
utilized due to the dynamic nature of vehicles, complicating the
optimization of drone trajectories. [17] explores computation
offloading in RSU-based VEC networks, proposing solutions
with varying complexity for small-scale and large-scale scenar-
ios. However, the substantial number of computation tasks leads
to sign

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p2起）

problem formulation are given in Section III. Section IV pro-
poses the DRLCL algorithm. Simulation results and discussions
are shown in SectionV. Finally, SectionVI draws the conclusion
of this paper.
II. R ELA TEDWORKS
In recent years, MEC technology has been extensively
adopted in VEC networks to alleviate the inherent computa-
tional limitations of onboard vehicle systems. Numerous studies
have focused on the computation offloading problems, explor-
ing aspects including task destination selection, communication
resource allocation, and load balancing.
A. Task Destination Selection
Regarding task destination selection, various strategies have
been developed to ensure QoS-guaranteed computation offload-
ing in IoV scenarios, leveraging a variety of auxiliary compu-
tation nodes. For instance, [12] and [13] propose a learning-
based vehicle-to-vehicle (V2V) offloading architecture, which
capitalizes on the capabilities and trustworthiness of service-
enabled vehicles with spare computation power. To facilitate
V2V offloading, [14] presents a trajectory prediction model
for task vehicles based on long short-term memory (LSTM).
However, V2V links are prone to frequent disruptions, leading to
frequent link switching and decreased response speed. Further-
more, [15] and [16] consider the drone-assisted VEC networks,
where drones hover to provide additional computing capabili-
ties. However, the mobile characteristics of drones are not fully
utilized due to the dynamic nature of vehicles, complicating the
optimization of drone trajectories. [17] explores computation
offloading in RSU-based VEC networks, proposing solutions
with varying complexity for small-scale and large-scale scenar-
ios. However, the substantial number of computation tasks leads
to significant queuing d

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p1起）

proposed algorithm in reducing system delay compared to
existing state-of-the-art baselines, offering an effective approach
for optimizing the performance of vehicular edge computing (VEC)
networks.
Index Terms —Multi-access edge computing, task offloading,
resource allocation, load balancing, deep reinforcement learning.
Received 30 December 2024; revised 3 April 2025; accepted 7 April 2025.
Date of publication 10 April 2025; date of current version 6 August 2025. This
work was supported in part by the Open Research Fund of National Mobile Com-
munications Research Laboratory, Southeast University under Grant 2024D13,
in part by Guangdong S&T Program under Grant 2024B0101020004, in part by
the Postgraduate Research and Practice Innovation Program of Jiangsu Province
under Grant SJCX-240062, and in part by the SEU Innovation Capability
Enhancement Plan for Doctoral Students under Grant CXJH-SEU24075. Rec-
ommended for acceptance by M. Amadeo. (Corresponding authors: Tiecheng
Song; Xiaoqin Song.)
Quan Chen and Tiecheng Song are with National Mobile Communications
Research Laboratory, Southeast University, Nanjing 210096, China (e-mail:
chanderchen@163.com; songtc@seu.edu.cn).
Xiaoqin Song is with the College of Electronic and Information Engineering,
Nanjing University of Aeronautics and Astronautics, Nanjing 210016, China,
and also with the National Mobile Communications Resear

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p2起）

simulations to validate the performance of
the proposed two-stage solution and compare it with base-
lines by adjusting key parameters, including the number
of RSUs and vehicles, task data size, and computational
capability of RSUs. The simulation results demonstrate
the superiority of our algorithm over the baselines in de-
termining computation offloading strategies that minimize
the total delay.
The rest of this paper is organized as follows. Section II
presents the overview of existing works. The system model and
problem formulation are given in Section III. Section IV pro-
poses the DRLCL algorithm. Simulation results and discussions
are shown in SectionV. Finally, SectionVI draws the conclusion
of this paper.
II. R ELA TEDWORKS
In recent years, MEC technology has been extensively
adopted in VEC networks to alleviate the inherent computa-
tional limitations of onboard vehicle systems. Numerous studies
have focused on the computation offloading problems, explor-
ing aspects including task destination selection, communication
resource allocation, and load balancing.
A. Task Destination Selection
Regarding task destination selection, various strategies have
been developed to ensure QoS-guaranteed computation offload-
ing in IoV scenarios, leveraging a variety of auxiliary compu-
tation nodes. For instance, [12] and [13] propose a learning-
based vehicle-to-vehicle (V2V) offlo

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 是 |
| 服务更新周期 | 未提及 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 是 |
| 位置预测误差 | 未提及 |
| 服务迁移 | 未提及 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 未提及 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 是 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=未提及；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=是 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：2（PDF页）
- 方法窗口：1（PDF页）
- 仿真窗口：2（PDF页）
- 结论窗口：2（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（16页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
