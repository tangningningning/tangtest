# L105 — QoS-aware task offloading and resource allocation optimization in vehicular edge computing networks via MADDPG

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：Jingxian Liu
- 年份：2024
- 刊物/会议：Computer Networks
- DOI：10.1016/j.comnet.2024.110282
- PDF页数：10
- 原文：`C:\Users\tangning\Desktop\文献\02_移动性感知的服务迁移与传输资源协同\QoS-aware task offloading and resource allocation optimization in vehicular edge computing networks via MADDPG.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=8, UAV=0
- 主研究方向：P1；次方向：无；依据：scores={'1': 50, '2': 0, '3': 22}; title=QoS-aware task offloading and resource allocation optimization in vehicular edge

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

System model
In this paper, we consider a system model that incorporates multiple
task vehicles (TVs), multiple road side units (RSUs) with their con-
nected MEC servers, and multiple service vehicles (SVs). The RSUs are
uniformly distributed along the roadside, ensuring adequate spectrum
and computing resources. On the other hand, the SVs are randomly
located idle vehicles that can offer limited spectrum and computing
resources. The TVs, similar to the SVs, are randomly distributed and
responsible for generating computational tasks. These tasks can be
processed locally or offloaded to a RSU or an SV.
The model illustrating the offloading of tasks in the IoV scenario
is depicted in Fig. 1. In our network, both SVs and RSUs can provide
computing services, collectively referred to as service nodes (SNs). We
assume that there are 𝑈 RSUs and (𝑀 −𝑈 ) SVs. Therefore, the set of
SNs is denoted

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

System model
In this paper, we consider a system model that incorporates multiple
task vehicles (TVs), multiple road side units (RSUs) with their con-
nected MEC servers, and multiple service vehicles (SVs). The RSUs are
uniformly distributed along the roadside, ensuring adequate spectrum
and computing resources. On the other hand, the SVs are randomly
located idle vehicles that can offer limited spectrum and computing
resources. The TVs, similar to the SVs, are randomly distributed and
responsible for generating computational tasks. These tasks can be
processed locally or offloaded to a RSU or an SV.
The model illustrating the offloading of tasks in the IoV scenario
is depicted in Fig. 1. In our network, both SVs and RSUs can provide
computing services, collectively referred to as service nodes (SNs). We
assume that there are 𝑈 RSUs and (𝑀 −𝑈 ) SVs. Therefore, the set of
SNs is denoted as  = { 𝐸1,𝐸 2, …,𝐸𝑀 }. 𝐸𝑗𝑘 denotes 𝑘𝑡ℎ channel of
𝐸𝑗. Each RSU has 𝑉 channels, thus 𝐸𝑗𝑘 for 1 ≤ 𝑗 ≤ 𝑈 corresponds
to the (𝑗 − 1)𝑉 + 𝑘𝑡ℎ channel. On the other hand, each SV has 𝐺
channels where 𝐺 ≤ 𝑉 , and 𝐸𝑗𝑘 for 𝑈 + 1 ≤ 𝑗 ≤ 𝑀 corresponds to
the 𝑈𝑉 + (𝑗 − 1 − 𝑈 )𝐺 +𝑘𝑡ℎ channel. Consequently, the total system
spectrum is divided equally into 𝐶 orthogonal channels, where 𝐶 =
𝑈𝑉 + (𝑀 −𝑈 )𝐺.
We assume there are 𝑁 TVs driven on the road during a specific
timeslot, each with a computational task. The set of TVs is denoted as
 = {𝑣1,𝑣 2, …,𝑣𝑁 }. Each TV has the option to either offload its task
to an SN or complete it locally. We consider the heterogeneity of tasks,
i.e., the tasks are varying in task sizes and QoS requirements and as-
signed to nodes with different computational capabilities. We introduce
the concept of task priority to characterize the heterogeneity of tasks.
The task of 𝑣

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p3起）

Problem formulation
The objective of this paper is to minimize the system costs by finding
the optimal channel access and offloading policy and the resource al-
location scheme, while considering the constraints of limited resources
and strict latency requirements. Moreover, we emphasis the fulfillment
of the QoS requirements through characterizing each task priority. Let
the indicator I𝑖 represent if the task is successfully completed. If I𝑖 = 1,
it indicates that 𝛤𝑖 is completed within the given deadline, i.e., 𝑇𝑖 ≤
𝑇 max𝑖 . Otherwise, if I𝑖 = 0 , it implies that 𝛤𝑖 fails to be completed
within the deadline. Let  = { 𝑃𝑖, 1 ≤ 𝑖 ≤ 𝑁} and  = { 𝑓𝑖,𝑗, 1 ≤
𝑖 ≤ 𝑁, 1 ≤ 𝑗 ≤ 𝑀} denote the set of transmit powers of TVs and the
computational resource allocation strategy of SNs to TVs, respectively.
The optimization problem can be formulated as follows:
(P1:) min
,,
𝑁∑
𝑖=1
𝑍𝑖 −𝜂I𝑖𝑝𝑖 (14a)
s.t. 𝑥𝑖,𝑗,𝑘 ∈ {0, 1}, 1 ≤𝑖 ≤𝑁, 1 ≤𝑗 ≤𝑀, 1 ≤𝑘 ≤𝑉 (14b)
𝑀∑
𝑗=1
𝑉∑
𝑘=1
𝑥𝑖,𝑗,𝑘 ≤ 1, 1 ≤𝑖 ≤𝑁 (14c)
0 ≤𝑓𝑖,𝑗 ≤𝐹𝑗, 1 ≤𝑖 ≤𝑁, 1 ≤𝑗 ≤𝑀 (14d)
𝑁∑
𝑖=1
𝑉∑
𝑘=1
𝑥𝑖,𝑗,𝑘𝑓𝑖,𝑗 ≤𝐹𝑗, 1 ≤𝑗 ≤𝑀 (14e)
0 ≤𝑃𝑖 ≤𝑃max, 1 ≤𝑖 ≤𝑁 (14f)
I𝑖 ∈ {0, 1}, 1 ≤𝑖 ≤𝑁 (14g)
where (14b) and (14c) are constraints for offloading decisions, allowing
each TV to choose local computation or offloading its task to an
SN. (14d) and (14e) are computational resource constraints, ensuring
that the total computational resources 𝐸𝑗 assigns to do not exceed its
maximum computational capacity 𝐹𝑗. (14f) ensures that the transmit
power does not exceed TVs upper bound 𝑃max.
For problem (P1), the first step is to make a decision regarding
whether to opt for offloading or not, which entails determining the
offloading strategy . Thus, the availability of communication re-
sources must also be taken into account. In cases where two TVs opt
to offload tasks on

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p2起）

proposed algorithm is
superior to some benchmark algorithms such as Multi-Agent
Deep Q-Network (MADQN) in terms of the total system rewards,
the system latency and the success rate .
2. System model
In this paper, we consider a system model that incorporates multiple
task vehicles (TVs), multiple road side units (RSUs) with their con-
nected MEC servers, and multiple service vehicles (SVs). The RSUs are
uniformly distributed along the roadside, ensuring adequate spectrum
and computing resources. On the other hand, the SVs are randomly
located idle vehicles that can offer limited spectrum and computing
resources. The TVs, similar to the SVs, are randomly distributed and
responsible for generating computational tasks. These tasks can be
processed locally or offloaded to a RSU or an SV.
The model illustrating the offloading of tasks in the IoV scenario
is depicted in Fig. 1. In our network, both SVs and RSUs can provide
computing services, collectively referred to as service nodes (SNs). We
assume that there are 𝑈 RSUs and (𝑀 −𝑈 ) SVs. Therefore, the set of
SNs is denoted as  = { 𝐸1,𝐸 2, …,𝐸𝑀 }. 𝐸𝑗𝑘 denotes 𝑘𝑡ℎ channel of
𝐸𝑗. Each RSU has 𝑉 channels, thus 𝐸𝑗𝑘 for 1 ≤ 𝑗 ≤ 𝑈 corresponds
to the (𝑗 − 1)𝑉 + 𝑘𝑡ℎ channel. On the other hand, each SV has 𝐺
channels where 𝐺 ≤ 𝑉 , and 𝐸𝑗𝑘 for 𝑈 + 1 ≤ 𝑗 ≤ 𝑀 corresponds to
the 𝑈𝑉 + (𝑗 − 1 − 𝑈 )𝐺 +𝑘𝑡ℎ channel. Consequently, the total system
sp

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

simulation results demonstrate that
the proposed MADDPG-RA algorithm can effectively reduce the total system latency and energy consumption
compared to the existing algorithms.
1. Introduction
The continuous development of Internet of Vehicles (IoV) technol-
ogy has brought about a surge in the access of vehicles and an increase
in the diverse demand for services [ 1–4]. Consequently, there is a
growing need for a network that offers low latency and high reliability
to support various applications [ 5], including real-time navigation,
traffic flow optimization, and intelligent driving. However, the limited
computing capacities of vehicles hinder the fulfillment of complex
computing tasks. Mobile Edge Computing (MEC), as an emerging net-
work architecture and computational technique, plays a crucial role
in providing support for the attainment of mission-critical services
in IoV [ 6]. Through the process of computation offloading, selected
computation tasks and data processing are transferred from the vehicle
side to the edge servers for efficient processing [ 7,8]. This approach
effectively addresses the high demand for computation-intensive and
time-critical tasks by utilizing the storage and computation resources
available. Consequently, it achieves a network with low latency and
high reliability, thereby ensuring the provision of high-quality ser-
vice [9]. Considering the d

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 是 |
| 服务更新周期 | 未提及 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 未提及 |
| 位置预测误差 | 未提及 |
| 服务迁移 | 未提及 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 未提及 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 是 |
| 信道动态变化 | 部分 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=未提及；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=未提及；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=是；信道动态=部分 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：3（PDF页）
- 方法窗口：2（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：9（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（10页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
