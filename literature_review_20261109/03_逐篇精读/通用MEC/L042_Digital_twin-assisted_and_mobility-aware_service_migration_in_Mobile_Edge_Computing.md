# L042 — Digital twin-assisted and mobility-aware service migration in Mobile Edge Computing

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：Elif Bozkaya
- 年份：2023
- 刊物/会议：Computer Networks
- DOI：10.1016/j.comnet.2023.109798
- PDF页数：12
- 原文：`C:\Users\tangning\Desktop\文献\18篇指定论文合集\Digital Twin-Assisted and Mobility-Aware Service Migration in Mobile Edge Computing.pdf`
- 领域分类：**C**；依据：摘要/模型段 MEC=23, vehicle=5, UAV=0
- 主研究方向：P2；次方向：无；依据：scores={'1': 30, '2': 161, '3': 39}; title=Digital twin-assisted and mobility-aware service migration in Mobile Edge Comput

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model for DT-assisted
service migration followed by the proposed service migration decision
algorithm in Section 4. We evaluate our model in Section 5. Finally, we
give our conclusions in Section 6.
2. Related work
In this section, we review the most relevant studies in service
migration schemes and also summarize the collaboration of DT and
MEC paradigm.
2.1. Service migration
With the development of MEC networks, task offloading and service
migration solutions have been extensively started to study, [14–16].
However (i) high uncertainty of the user mobility and traffic requests,
(ii) limited coverage area of edge servers, (iii) high computational
complexity are the main shortcomings of these works. In Table 1, we
compare the service migration problem in terms of the methods used,
advantages and disadvantages.
In [14], authors formulate the service placement and migration
problem

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model for DT-assisted
service migration followed by the proposed service migration decision
algorithm in Section 4. We evaluate our model in Section 5. Finally, we
give our conclusions in Section 6.
2. Related work
In this section, we review the most relevant studies in service
migration schemes and also summarize the collaboration of DT and
MEC paradigm.
2.1. Service migration
With the development of MEC networks, task offloading and service
migration solutions have been extensively started to study, [14–16].
However (i) high uncertainty of the user mobility and traffic requests,
(ii) limited coverage area of edge servers, (iii) high computational
complexity are the main shortcomings of these works. In Table 1, we
compare the service migration problem in terms of the methods used,
advantages and disadvantages.
In [14], authors formulate the service placement and migration
problem and design a deep Q learning based scheme by modeling a
multi-agent Markov decision process in MEC-enabled vehicular net-
works. Similarly, authors in [17] propose a deep Q-learning service
migration decision algorithm for MEC-enabled vehicular networks.
In [15], the authors model the problem as a partially observable
Markov decision process with the assumption of partial user’s infor-
mation and propose an intelligent service migration algorithm. Here,
the most important point is to predict the user mobility. Otherwise,
the model is not useful in practical situations. Thus, in this paper, we
use the information of user mobility prediction and aim to reduce the
service migration rate.
The authors in [16] emphasize that the locations of mobile users
change over time, therefore users are served at different base sta-
tions and this affects user-perceived QoS. They formulate edge-user
allo

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p7起）

Problem formulation
By integrating computational model and service migration model,
our goal is to develop an effective service migration strategy that
minimizes the task completion delay. With this aim, we formulate an
optimization problem whose objective is to provide the guaranteed
service to the mobile users. This problem can be formalized as follows:
𝐸𝑐𝑜𝑠𝑡𝑖,𝑗 =
{
𝐸𝑡𝑜𝑡𝑎𝑙𝑖,𝑗 (𝑡) +𝐸𝑚𝑖𝑔𝑟𝑖 (𝑡 + 1), if migration occurs
𝐸𝑡𝑜𝑡𝑎𝑙𝑖,𝑗 (𝑡), otherwise (11)
and
min
𝑀∑
𝑖=1
𝐸𝑐𝑜𝑠𝑡𝑖,𝑗 𝑎𝑖,𝑗 (𝑡) (12)
Table 4
Number of possible next states with 10 edge servers.
3 4 3 –
3 6 7 3
3 4 3 –
subject to
𝐶1 ∶𝑎𝑖,𝑗 (𝑡) ∈ {0, 1} 𝑖 ∈𝑀,𝑗 ∈𝑁 (12a)
𝐶2 ∶𝐶𝑗 ≤𝑑𝑖,𝑗 (12b)
𝐶3 ∶
{
𝐸𝑐𝑜𝑠𝑡𝑖,𝑗 ≤𝜀1, for delay sensitive tasks
𝐸𝑐𝑜𝑠𝑡𝑖,𝑗 ≤𝜀2, for delay insensitive tasks (12c)
where 𝐶1 indicates that each user can be served by only one edge
server. 𝐶2 indicates that edge servers can serve the users within the
coverage area of itself or neighbor cells.𝑑𝑖,𝑗 represents the distance that
the edge server can serve the user. The set of alternative edge servers
will be illustrated in the next section.𝐶3 indicates the maximum latency
threshold to guarantee the acceptable QoS of computing tasks. The QoS
requirements are given in Table 3.
4. Mobility-aware service migration
In this section, a QoS-aware service migration algorithm is pre-
sented based on the system model and problem formulation.
4.1. Edge server selection and service migration algorithm
Optimal task offloading and service migration in edge computing
have been proved to be NP-hard problem. Several heuristic algorithms
have been implemented to address these problems, but choosing an
appropriate heuristic algorithm is also difficult because the algorithm
are developed under different assumptions [40].
The basic idea of the decision algorithm is to find an assignment

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p8起）

proposed algorithm is applied in DT layer, which digitizes
the physical environment by collecting and analyzing both real-time
and historical spatial and temporal information of users to guarantee
acceptable QoS. DT creates the virtual copies of MEC-enabled SBSs and
mobile users. It monitors the user behavior in different scenarios and,
predicts the user mobility and makes data-driven decisions in order to
balance the traffic workloads among edge servers.
After the service migration decision is given by the DT, the comput-
ing task is migrated to the new edge server. The procedure involves the
connection between edge servers, transfer of task context. Source edge
server initiates the procedure. Target edge server allocates the resource
for the specific user. Once service migration is completed, source edge
server releases the wireless service and related resources.
As compared to the traditional mechanisms where users select their
edge servers based on metrics such as the received power or SINR,
the proposed model needs to gather information from users including
their computing task demands, and then predicts the user movement
and aims to reduce service migration rate. One of the most important
characteristics of DT is its ability to monitor the status of physical
entities in real time and to use both historical and real-time data for
decision making process. Since user mobilit

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

simulation
results shows that our DT-assisted service migration model can achieve less delay and service migration rate
compared with the traditional methods.
1. Introduction
Emerging new applications (e.g. virtual reality, mobile augmented
reality, smart grids etc.) generate huge amounts of data and require
high transmission rates with strict Quality of Service (QoS) require-
ments. With the development of these applications, it has become
inevitable for mobile users to increase the data traffic in order to inter-
act and entertain themselves more [ 1]. According to Ericsson Mobility
Report Data and Forecasts [2], total mobile network traffic reached
84𝑒𝑥𝑎𝑏𝑦𝑡𝑒 (𝐸𝐵 ) per month by the end of 2021, and was estimated that
it will reach to 368𝐸𝐵 per month by the end of 2027. Thus, we can
also estimate that the traffic will continuously increase at a very fast
pace [ 3]. In addition to increased data traffic, 5G wireless networks
specifically call for ultra-reliable (>99%) and low-latency (<10 ms) com-
munications (URLLC) [4]. However, computation-intensive and delay-
sensitive tasks cannot be executed with the computing and storage
capabilities of User Equipment (UE).
The European Telecommunications Standards Institute (ETSI)
defines Mobile Edge Computing (MEC) as the main IT service
infrastructure to satisfy the needs for ultra-reliable and low-latency
E-mail address: ebozkaya@dho

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 部分 |
| 服务更新周期 | 未提及 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 是 |
| 位置预测误差 | 是 |
| 服务迁移 | 是 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 未提及 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 是 |
| 信道动态变化 | 未提及 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=部分；服务更新周期=未提及；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=是；服务迁移=是；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=是；信道动态=未提及 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：7（PDF页）
- 方法窗口：8（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：2（PDF页）

### 关键词命中片段

- p1: nline 11 May 2023 1389-1286/© 2023 Elsevier B.V. All rights reserved. Contents lists available at ScienceDirect Computer Networks journal homepage: www.elsevier.com/locate/comnet Digital twin-assisted and mobility-aware service migration in Mobile Edge Computing Elif Bozkaya Department of Computer Engineering, National Defence University, Turkish N
- p1: migration in Mobile Edge Computing Elif Bozkaya Department of Computer Engineering, National Defence University, Turkish Naval Academy, Istanbul, Turkey A R T I C L E I N F O Keywords: Mobile Edge Computing Digital Twin Service migration Digital Twin edge network User mobility Resource allocation A B S T R A C T Mobile Edge Computing (MEC) is emerg

## 12. 阅读闭合状态

- 全文文本：已提取（12页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
