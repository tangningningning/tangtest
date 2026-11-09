# L045 — Dual-Agent DRL-Based Service Placement, Task Scheduling, and Resource Allocation for Multi-Sensor and Multi-User Edge Computing Networks

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2025
- 刊物/会议：IEEE Transactions on Network Science and Engineering
- DOI：10.1109/TNSE.2025.3560402
- PDF页数：18
- 原文：`C:\Users\tangning\Desktop\文献\18篇指定论文合集\Dual-Agent DRL-Based Service Placement, Task Scheduling, and Resource Allocation for Multi-Sensor and Multi-User Edge Computing Networks.pdf`
- 领域分类：**C**；依据：摘要/模型段 MEC=12, vehicle=2, UAV=0
- 主研究方向：P1；次方向：无；依据：scores={'1': 50, '2': 3, '3': 3}; title=Dual-Agent DRL-Based Service Placement, Task Scheduling, and Resource Allocation

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model and problem formu-
lation are described in Section III. In Section IV, we provide a
detailed description of our proposed algorithm. We evaluate the
proposed algorithm in Section V and discuss the results. Finally,
we summarize our work in Section VI.
II. R ELA TEDWORKS
MEC pushes computing functions to the edge of the net-
work, enabling users to offload computation-intensive tasks to
the ESs equipped on/near network edge nodes, such as base
stations (BSs), thereby enhancing task processing capabilities.
Service placement and task scheduling are two critical research
challenges in MEC. In recent years, a substantial amount of
research has focused on the service placement and task schedul-
ing problems, as well as their joint optimization [7], [8], [9],
[10], [11], [12].T h ew o r ki n [8] addressed the online service
placement and request allocation problem in MEC networks.

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model and problem formu-
lation are described in Section III. In Section IV, we provide a
detailed description of our proposed algorithm. We evaluate the
proposed algorithm in Section V and discuss the results. Finally,
we summarize our work in Section VI.
II. R ELA TEDWORKS
MEC pushes computing functions to the edge of the net-
work, enabling users to offload computation-intensive tasks to
the ESs equipped on/near network edge nodes, such as base
stations (BSs), thereby enhancing task processing capabilities.
Service placement and task scheduling are two critical research
challenges in MEC. In recent years, a substantial amount of
research has focused on the service placement and task schedul-
ing problems, as well as their joint optimization [7], [8], [9],
[10], [11], [12].T h ew o r ki n [8] addressed the online service
placement and request allocation problem in MEC networks.
In this study, service requests are allocated to cloudlets, and
effective prediction mechanisms are employed to pre-install ser-
vice instances and revoke idle instances, aiming to maximize the
profit for service providers. The work in[9] recognized that data-
intensive services require not only computing resources but also
storage resources for server code and data, as well as network
bandwidth for receiving data from users. The authors proposed a
framework to jointly optimize service (code and data) placement
and request scheduling, considering storage, communication,
computing resources, and budget constraints. In[10], the authors
highlighted that services like augmented reality require users to
upload data for service execution and then download the output
for user utilization. This bidirectional communication may be
asymmetric, occupying different portions of the BS’s uplink and
dow

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p1起）

optimization problem involving service placement, task scheduling,
and resource allocation to minimize the total system cost, defined as
the weighted sum of delay and energy consumption of each user and
sensor . We jointly optimize service placement, user task schedul-
ing, transmit power allocation for sensors and users, computing
resource allocation for both the ESs and the cloud server (CS),
and transmission rate allocation for ES-ES and ES-CS connections.
Considering the differences in the update frequencies of the opti-
mization variables, we propose a dual-agent Deep Reinforcement
Learning (DRL) algorithm, which utilizes two SD3 (Softmax Deep
Double Deterministic Policy Gradients)-based DRL agents to make
service placement and task scheduling decisions asynchronously,
while embedding two optimization subroutines to solve the optimal
transmit power allocation, computing resource and transmission
rate allocations using numerical methods. The complexity and con-
vergence of the algorithm are analyzed and extensive experiments
Received 7 December 2024; revised 13 March 2025; accepted 3 April 2025.
Date of publication 14 April 2025; date of current version 25 August 2025. This
work was supported in part by the National Natural Science Foundations of
China under Grant 62271086, in part by Beijing Natural Science Foundation
under Grant L242080, in part by the Science and Technology Program of
Qinghai Province of China under Grant 2024-SF-129, in part by the Fundamental
Research Funds for the Central Universities under Grant 2024ZCJH03, and in
part by the Fund of State Key Laboratory of Information Photonics and Optical
Communications (Beijing University of Posts and Telecommunications), China.
Recommended for acceptance by Dr. Chunxiao Jiang. (Corresponding author:
Wenha

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p2起）

proposed algorithm. We evaluate the
proposed algorithm in Section V and discuss the results. Finally,
we summarize our work in Section VI.
II. R ELA TEDWORKS
MEC pushes computing functions to the edge of the net-
work, enabling users to offload computation-intensive tasks to
the ESs equipped on/near network edge nodes, such as base
stations (BSs), thereby enhancing task processing capabilities.
Service placement and task scheduling are two critical research
challenges in MEC. In recent years, a substantial amount of
research has focused on the service placement and task schedul-
ing problems, as well as their joint optimization [7], [8], [9],
[10], [11], [12].T h ew o r ki n [8] addressed the online service
placement and request allocation problem in MEC networks.
In this study, service requests are allocated to cloudlets, and
effective prediction mechanisms are employed to pre-install ser-
vice instances and revoke idle instances, aiming to maximize the
profit for service providers. The work in[9] recognized that data-
intensive services require not only computing resources but also
storage resources for server code and data, as well as network
bandwidth for receiving data from users. The authors proposed a
framework to jointly optimize service (code and data) placement
and request scheduling, considering storage, communication,
computing resources, and budget constraints. In[

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

experiments
Received 7 December 2024; revised 13 March 2025; accepted 3 April 2025.
Date of publication 14 April 2025; date of current version 25 August 2025. This
work was supported in part by the National Natural Science Foundations of
China under Grant 62271086, in part by Beijing Natural Science Foundation
under Grant L242080, in part by the Science and Technology Program of
Qinghai Province of China under Grant 2024-SF-129, in part by the Fundamental
Research Funds for the Central Universities under Grant 2024ZCJH03, and in
part by the Fund of State Key Laboratory of Information Photonics and Optical
Communications (Beijing University of Posts and Telecommunications), China.
Recommended for acceptance by Dr. Chunxiao Jiang. (Corresponding author:
Wenhao Fan.)
Wenhao Fan is with the School of Electronic Engineering, Beijing University
of Posts and Telecommunications (BUPT), Beijing 100876, China, and also with
the Beijing Key Laboratory of Work Safety Intelligent Monitoring, and National
Engineering Research Center for Disaster Backup and Recovery, BUPT, Beijing
100876, China (e-mail: whfan@bupt.edu.cn).
Xiongfei Chun, Ruimin Zhang, and Y uan’an Liu are with the School of
Electronic Engineering, Beijing University of Posts and Telecommunications
(BUPT), Beijing 100876, China.
Zhiyu Fan is with the Network Department of China Mobile Communications
Group Company Ltd., Beijing

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 是 |
| 服务更新周期 | 是 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 是 |
| 位置预测误差 | 未提及 |
| 服务迁移 | 未提及 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 未提及 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 部分 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=是；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=部分 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：1（PDF页）
- 方法窗口：2（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：None（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（18页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
