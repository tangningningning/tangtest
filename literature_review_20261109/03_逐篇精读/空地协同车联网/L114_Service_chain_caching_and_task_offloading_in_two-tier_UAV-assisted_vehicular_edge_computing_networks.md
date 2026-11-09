# L114 — Service chain caching and task offloading in two-tier UAV-assisted vehicular edge computing networks: An attention-DRL method

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：Fengyan Wu
- 年份：2026
- 刊物/会议：Digital Communications and Networks
- DOI：10.1016/j.dcan.2024.10.005
- PDF页数：11
- 原文：`C:\Users\tangning\Desktop\文献\01_时延约束下服务部署与任务卸载的自适应多时间尺度协同\Service chain caching and task offloading in two-tier UAV-assisted vehicular edge computing networks- An attention-DRL method.pdf`
- 领域分类：**B**；依据：摘要/模型段 vehicle=6, UAV=32
- 主研究方向：P1；次方向：无；依据：scores={'1': 33, '2': 2, '3': 5}; title=Service chain caching and task offloading in two-tier UAV-assisted vehicular edg

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model and computation oﬄoad-
ing, and Section4describes the service caching and problem formula-
tion. The attention-based MADQN algorithm is provided in Section 5, 
and numerical results are discussed in Section6. Finally, Section7con-
cludes the work.
2. Related work
In this section, we describe the multi-UAV networks, the existing 
joint service caching and task oﬄoading algorithms, and then present a 
brief review of these works.
2.1. Multi-UAV networks
UAVs, with their mobility, LoS communications, and flexible deploy-
ment, can function as temporary aerial base stations or edge comput-
ing platforms to serve ground vehicles, particularly in the congested 
roads [23]. To enhance the UAV eﬃciency, the task oﬄoading divi-
sion ratio, UAV deployment, communication, and computation resource 
allocation [9,24,25]w e r e analyzed to reduce latency and energy con-
sumption. The reso

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model and computation oﬄoad-
ing, and Section4describes the service caching and problem formula-
tion. The attention-based MADQN algorithm is provided in Section 5, 
and numerical results are discussed in Section6. Finally, Section7con-
cludes the work.
2. Related work
In this section, we describe the multi-UAV networks, the existing 
joint service caching and task oﬄoading algorithms, and then present a 
brief review of these works.
2.1. Multi-UAV networks
UAVs, with their mobility, LoS communications, and flexible deploy-
ment, can function as temporary aerial base stations or edge comput-
ing platforms to serve ground vehicles, particularly in the congested 
roads [23]. To enhance the UAV eﬃciency, the task oﬄoading divi-
sion ratio, UAV deployment, communication, and computation resource 
allocation [9,24,25]w e r e analyzed to reduce latency and energy con-
sumption. The resource management in UAV-assisted VECNs was further 
optimized via multi-agent reinforcement learning algorithm in [12]. 
In a UAV-aided IoT network, the ultra-reliable low-latency computa-
tion oﬄoading problem was studied to maximize the transmission rates 
[26]. However, the computation and storage capacity of a single UAV is 
limited, making it diﬃcult to handle complex tasks and larger-scale op-
erations. A viable and increasingly adopted solution is the integration 
of cloud computing platforms, which serve as a supplemental resource 
to enhance the capabilities of UAVs. In [27], the authors proposed an 
online optimal computation oﬄoading and multi-hop routing scheme 
in an edge-cloud environment. While cloud centers provide additional 
computing resources, the distance from end-users results in prolonged 
transmission and communication delays, adversely impacting system 
performanc

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p3起）

formulation is linked to the UAV group, making 
attention-based UAV selection an eﬀective method for identifying suit-
able UAVs. Consider a task chain consisting of multiple dependent serial 
tasks, denoted as 𝑆 ={𝑠1, … , 𝑠𝑘, … , 𝑠𝐾}, where 𝑠𝑘 is a task 𝑘 of the 
task chain, and 𝐾 is the number of tasks. The task can be described as 
𝑠𝑘 =(𝑑𝑘, 𝑐𝑘, 𝜏𝑘, 𝑓𝑘), where 𝑑𝑘 represents the computation input data size 
(in bit), 𝑐𝑘 indicates the requested CPU cycles, 𝜏𝑘 implies the tolerance 
latency, and 𝑓𝑘 signifies the required service data. Each task requires 
specific program data and computation resources. The UAVs serve as 
service caching nodes to process these tasks.
An attention mechanism is introduced to align the tasks with the 
optimal computation and storage resources of UAVs. It evaluates each 
task’s requirements and calculates an attention score for each multi-
rotor UAV based on three key factors: the alignment of the UAV’s cached 
Digital Communications and Networks 12 (2026) 607–617
610
F.Wu,C.Yang,Y.Tangetal.
Fig.2. Depictionofthetaskdivisionprocess.
service data with the task requirements, the communication quality 
between the multi-rotor UAV and the vehicle user, and the UAV’s ca-
pabilities, including computation resources, energy consumption, and 
battery life. Then, a subset of UAVs is chosen to form the UAV group.
Actually, when the number of UAVs is larger, part of UAVs can 
select duplicate caching to reduce the computation time. The special 
task can be divided into sub-tasks that can be executed simultaneously, 
thereby utilizing the computational resources of the multi-rotor UAVs 
more eﬀectively. Consequently, the task chain evolves into a heteroge-
neous combination of serial and parallel processes, enhancing network 
resource utilization and eﬃc

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p1起）

proposed algorithm significantly enhances 
system processing eﬃciency and reduces task completion latency compared to the benchmark approaches.
1. Introduction
With the rapid expansion of mobile communication and AI tech-
nologies, data generated by connected vehicles has surged significantly. 
Vehicular Edge Computing Networks (VECNs) oﬀer a promising solution 
by distributing computation, communication, and caching resources at 
the network edge node, such as Road Side Unit (RSU), and nearby/park-
ing vehicles. This approach reduces data transmission distances, en-
abling direct support for computation-intensive and latency-sensitive 
tasks, thereby enhancing the performance of Intelligent Transportation 
Systems (ITS) [1,2]. However, the emergence of various applications 
has significantly increased the complexity and dynamism of vehicle 
users’ computation tasks. Deploying RSUs extensively along roadways 
is extremely costly and does not align with dynamic requirements. A 
✩ Peer review under the responsibility of the Chongqing University of Posts and Telecommunications.
* Corresponding author.
E-mail addresses:2112204070@mail2.gdut.edu.cn(F.Wu), yangchaoscut@aliyun.com(C.Yang), tangyq8@mail.sysu.edu.cn(Y.Tang), lzhen@eduhk.hk(Z.Li), 
shlxie@gdut.edu.cn(S.Xie).
cloud-edge collaborative approach[3–5]that distributes task execution 
across multiple edge nodes and the core clo

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p2起）

simulations, demonstrating the superiority of our proposed 
algorithm compared to benchmark solutions. The remainder of this ar-
ticle is organized as follows. In Section2, we review the previous related 
works. Section3introduces the system model and computation oﬄoad-
ing, and Section4describes the service caching and problem formula-
tion. The attention-based MADQN algorithm is provided in Section 5, 
and numerical results are discussed in Section6. Finally, Section7con-
cludes the work.
2. Related work
In this section, we describe the multi-UAV networks, the existing 
joint service caching and task oﬄoading algorithms, and then present a 
brief review of these works.
2.1. Multi-UAV networks
UAVs, with their mobility, LoS communications, and flexible deploy-
ment, can function as temporary aerial base stations or edge comput-
ing platforms to serve ground vehicles, particularly in the congested 
roads [23]. To enhance the UAV eﬃciency, the task oﬄoading divi-
sion ratio, UAV deployment, communication, and computation resource 
allocation [9,24,25]w e r e analyzed to reduce latency and energy con-
sumption. The resource management in UAV-assisted VECNs was further 
optimized via multi-agent reinforcement learning algorithm in [12]. 
In a UAV-aided IoT network, the ultra-reliable low-latency computa-
tion oﬄoading problem was studied to maximize the transmission rates 
[26]. H

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 部分 |
| 服务更新周期 | 未提及 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 是 |
| 位置预测误差 | 未提及 |
| 服务迁移 | 未提及 |
| 活动任务连续执行 | 部分 |
| 迁移传输 | 未提及 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 未提及 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=部分；服务更新周期=未提及；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=部分；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=未提及 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：3（PDF页）
- 方法窗口：1（PDF页）
- 仿真窗口：2（PDF页）
- 结论窗口：9（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（11页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
