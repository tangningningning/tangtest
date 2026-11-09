# L096 — Multi-Agent DRL-Based Two-Timescale Resource Allocation for Network Slicing in V2X Communications

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2024
- 刊物/会议：IEEE Transactions on Network and Service Management
- DOI：10.1109/TNSM.2024.3454758
- PDF页数：15
- 原文：`C:\Users\tangning\Desktop\文献\03_多业务非平稳环境下的服务部署与传输资源在线协同\Multi-Agent DRL-Based Two-Timescale Resource Allocation for Network Slicing in V2X Communications.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=9, UAV=0
- 主研究方向：P3；次方向：无；依据：scores={'1': 14, '2': 1, '3': 35}; title=Multi-Agent DRL-Based Two-Timescale Resource Allocation for Network Slicing in V

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model and problem formulation in Section III.T h e
decomposition of the problem and our proposed algorithms
are shown in Section IV. We demonstrate the performance
evaluations in Section V. We finally conclude this work in
Section VI.
II. L ITERATURE REVIEW
Network slicing involves the logical networks including
both the core network and radio access network (RAN).
Network slicing in the core network (i.e., core slicing)
addresses the optimal virtual network function orchestration,
which has been well studied and can be achieved by scaling up
computing resources [9], [10], [11], [12]. However, network
slicing in RAN (i.e., RAN slicing) is still a challenge due
to the scarcity of radio resources, dynamic conditions of
wireless channels, and interference [13]. Efficient management
of radio resources improves service delivery and utilization
of network resources. In general, existing

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model and problem formulation in Section III.T h e
decomposition of the problem and our proposed algorithms
are shown in Section IV. We demonstrate the performance
evaluations in Section V. We finally conclude this work in
Section VI.
II. L ITERATURE REVIEW
Network slicing involves the logical networks including
both the core network and radio access network (RAN).
Network slicing in the core network (i.e., core slicing)
addresses the optimal virtual network function orchestration,
which has been well studied and can be achieved by scaling up
computing resources [9], [10], [11], [12]. However, network
slicing in RAN (i.e., RAN slicing) is still a challenge due
to the scarcity of radio resources, dynamic conditions of
wireless channels, and interference [13]. Efficient management
of radio resources improves service delivery and utilization
of network resources. In general, existing works related to
resource allocation for RAN slicing can be grouped into two
categories, i.e., the intra-slice resource allocation and the inter-
slice resource allocation.
A. Intra-Slice Resource Allocation
Intra-slice resource allocation refers to assigning radio
resources to uses for guaranteeing the QoS requirements,
which has received significant attention. In [14], Chen et al.
proposed an online network slicing scheduling algorithm for
optimizing the PRBs allocation and power control in stochastic
vehicular networks with time-varying QoS requirements. The
objective is to maximize the long-term time-average total
system capacity while guaranteeing the stringent reliability
and latency requirements and queuing stability constraint.
In [15], Zhang et al. proposed an air-ground integrated
vehicular network slicing architecture for optimizing multi-
dimensional heterogeneous resources

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p2起）

problem formulation in Section III.T h e
decomposition of the problem and our proposed algorithms
are shown in Section IV. We demonstrate the performance
evaluations in Section V. We finally conclude this work in
Section VI.
II. L ITERATURE REVIEW
Network slicing involves the logical networks including
both the core network and radio access network (RAN).
Network slicing in the core network (i.e., core slicing)
addresses the optimal virtual network function orchestration,
which has been well studied and can be achieved by scaling up
computing resources [9], [10], [11], [12]. However, network
slicing in RAN (i.e., RAN slicing) is still a challenge due
to the scarcity of radio resources, dynamic conditions of
wireless channels, and interference [13]. Efficient management
of radio resources improves service delivery and utilization
of network resources. In general, existing works related to
resource allocation for RAN slicing can be grouped into two
categories, i.e., the intra-slice resource allocation and the inter-
slice resource allocation.
A. Intra-Slice Resource Allocation
Intra-slice resource allocation refers to assigning radio
resources to uses for guaranteeing the QoS requirements,
which has received significant attention. In [14], Chen et al.
proposed an online network slicing scheduling algorithm for
optimizing the PRBs allocation and power control in stochastic
vehicular networks with time-varying QoS requirements. The
objective is to maximize the long-term time-average total
system capacity while guaranteeing the stringent reliability
and latency requirements and queuing stability constraint.
In [15], Zhang et al. proposed an air-ground integrated
vehicular network slicing architecture for optimizing multi-
dimensional heterogeneous resources including broadca

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p2起）

proposed algorithms. The
results demonstrate that our proposed MADRL algorithm
can achieve approximate optimal performance of central-
ized DRL algorithm with the global information in terms
of the spectral efficiency, and guarantee the vehicles’ QoS
requirements with the CESCA algorithm.
The remainder of this paper is organized as follows. We
review the related studies in Section II. We present the
system model and problem formulation in Section III.T h e
decomposition of the problem and our proposed algorithms
are shown in Section IV. We demonstrate the performance
evaluations in Section V. We finally conclude this work in
Section VI.
II. L ITERATURE REVIEW
Network slicing involves the logical networks including
both the core network and radio access network (RAN).
Network slicing in the core network (i.e., core slicing)
addresses the optimal virtual network function orchestration,
which has been well studied and can be achieved by scaling up
computing resources [9], [10], [11], [12]. However, network
slicing in RAN (i.e., RAN slicing) is still a challenge due
to the scarcity of radio resources, dynamic conditions of
wireless channels, and interference [13]. Efficient management
of radio resources improves service delivery and utilization
of network resources. In general, existing works related to
resource allocation for RAN slicing can be grouped into two
categories, i.e., t

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

Simulation results show our
proposed two-timescale resource allocation scheme for network
slicing can achieve maximum 8.4% performance gains in terms
of spectral efficiency while guaranteeing the QoS requirements
of users compared to the baseline approaches.
Index Terms —Network slicing, V2X communications, multi-
agent deep reinforcement learning, resource allocations.
I. I NTRODUCTION
W
ITH the development of the 5th Generation Mobile
Networks (5G) and beyond, the Cellular-Vehicle-to-
Everything (C-V2X) has provided great technical support for
Internet of Vehicles (IoV), which significantly contributes
to the success of future Intelligent Transportation Systems
(ITS) [1]. With full connections between vehicles, pedestrians
and networks, IoV enables various emerging Vehicle-to-
Everything (V2X) applications, such as autonomous driving,
cooperative collision avoidance and entertainment [2].T h e
highly diverse Quality of Services (QoS) requirements and
limited spectrum resources require more efficient resource
allocation mechanism for providing diverse QoS provision-
ing [3].
As an emerging technology in 5G, network slicing is a
promising solution for provisioning diverse QoS services [4].
By creating multiple virtual logical networks on top of
common physical networks, network slicing allows network
operators to provide customized and dedicated resources for
different V2X appl

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
| 业务流量非平稳 | 是 |
| 信道动态变化 | 部分 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=是；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=是；信道动态=部分 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：2（PDF页）
- 方法窗口：2（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：13（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（15页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
