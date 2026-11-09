# L097 — Multi-Timescale Hierarchical Prefetching for Online Caching in Vehicular Edge Networks

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：待核
- 刊物/会议：待核
- DOI：10.1109/ICCCN65249.2025.11133729
- PDF页数：9
- 原文：`C:\Users\tangning\Desktop\文献\01_时延约束下服务部署与任务卸载的自适应多时间尺度协同\Multi-Timescale Hierarchical Prefetching for Online Caching in Vehicular Edge Networks.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=36, UAV=0
- 主研究方向：P2；次方向：无；依据：scores={'1': 0, '2': 16, '3': 6}; title=Multi-Timescale Hierarchical Prefetching for Online Caching in Vehicular Edge Ne

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

System Model
As shown in the Figure 1, the MEC network is modeled as
a three-tier caching architecture comprising the cloud layer,
the edge layer, and the vehicle layer. The edge layer consists
of macro base stations (MBSs) and roadside units (RSUs).
We assume that the contents requested by users are initially
located in the cloud data center. The service area of the
RSU is covered by the MBS. We use c to represent the
cloud. Let M = {mj} denote the set of MBSs supported
by operators. Each MBS mj has a storage capacity denoted
by φ(mj). Similarly, let R = {rk} be the set of RSUs, where
each RSU rk has a storage capacity denoted by φ(rk). Let
U = {ui} denote the user set of mobile vehicles. To capture
user mobility, we assume that the system operates in a time-
slotted structure, with the timeline being discretized into time
slots t ∈{ 1,2,...,T }. In this work, we assume that users
move

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

System Model
As shown in the Figure 1, the MEC network is modeled as
a three-tier caching architecture comprising the cloud layer,
the edge layer, and the vehicle layer. The edge layer consists
of macro base stations (MBSs) and roadside units (RSUs).
We assume that the contents requested by users are initially
located in the cloud data center. The service area of the
RSU is covered by the MBS. We use c to represent the
cloud. Let M = {mj} denote the set of MBSs supported
by operators. Each MBS mj has a storage capacity denoted
by φ(mj). Similarly, let R = {rk} be the set of RSUs, where
each RSU rk has a storage capacity denoted by φ(rk). Let
U = {ui} denote the user set of mobile vehicles. To capture
user mobility, we assume that the system operates in a time-
slotted structure, with the timeline being discretized into time
slots t ∈{ 1,2,...,T }. In this work, we assume that users
move irregularly and frequently between multiple edge servers.
At each time slot, vehicular nodes dynamically interact with
RSUs and MBSs to complete their content requests. Addition-
ally, we use H = {ha} to denote the set of variable contents
provided by the operators, and the data size of each content
is φ(ha). The requests from mobile users are characterized
using a mapping function π : ui →{ 1,2,..., |H|}, where
π(ui)= a specifies that the content ha is requested by the
user ui. Meanwhile, Hrk and Hmj denote the sets of content
cached in the RSU rk and MBS mj .
B. Delay Model
To ensure general applicability, we assume that users’ con-
tent requests remain consistent during a single macro time slot.
In the system, the transmission rate between different entities
is determined by the signal-to-noise ratio (SNR). The SNR
between the RSU rk and the user ui, denoted as SNR rk,ui ,i s
given by

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p3起）

optimization problem. The objective is to minimize
the average transmission delay for all users over a given
time period while considering the caching cost and resource
constraints of the system. The calculation of the delay depends
on the location of the cached content and the transmission
path. The optimization problem can be expressed as follows:
P1 : minimize D = limT→∞
1
T
∑
t∈T
∑
ui∈U Dui(t) (8)
s.t. ∑ha∈Hrk φ(ha)≤φ(rk), ∑ha∈Hmj φ(ha)≤φ(mj), (9)∑T
t=1 c(t) ≤ Cmax, (10)
xπ(ui ),k(t),yπ(ui ),j(t) ∈{ 0,1}. (11)
Equation (8) is the objective function, and equations (9) to (11)
are constraints. Equation (9) is the constraint for the storage
capacities of RSU and MBS. The total cost constraint over
the entire caching period is formulated in Equation (10), which
ensures that the cumulative caching cost over the entire period
does not exceed the budget Cmax. Equation (11) characterizes
whether the content requested by user ui is cached at rk or mj
at time slot t, where xπ(ui ),k(t) ∈{ 0,1}, and yπ(ui ),j(t) ∈
{0,1} are binary indicators denoting the caching status on the
RSU and MBS, respectively.
E. Reformulation
To tackle the above optimization problem, we first change
the perspective from the user level to the RSU and MBS level,
where the prefetching and caching decisions are made. To this
end, we introduce structured decision variables that capture the
hierarchical relationships between user service requests and
content caching across network layers.
We begin by modeling each RSU as an individual agent
and explicitly consider the corresponding caching decisions.
Here, xk(t)=[ x1,k(t),x 2,k(t),...,x n,k(t)]⊤ represents the
caching decision of the contents of the k-th RSU rk at time
slot t. For each RSU rk, the local loss function at time slot
t, denoted by lk(xk(t)), c

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p4起）

ALGORITHM DESIGN
In this section, we present the hierarchical online decision
making process underlying the VOPP framework (Figure 2),
which jointly addresses content caching optimization through
prefetching from the cloud to the MBSs and distributed proac-
tive caching from the MBSs to the RSUs. VOPP integrates
structured decision variables to capture cross-layer depen-
dencies and enable coordinated and latency-aware caching
strategies in vehicular edge networks.
A. RSU-Agent-based Distributed Online Decision Making
To enable decentralized and timely decision making at the
network edge, we reformulate the proactive caching problem
into a distributed online convex optimization model. In this
formulation, each RSU acts as an agent that makes caching
decisions based on locally observed requests. Fine-tuning is
performed based on online gradient descent on time slots and
by adapting to dynamic content demand and mobility patterns
while respecting local capacity constraints, ensuring scalability
and responsiveness in large vehicular edge environments.
1) Mobility-aware Content Prediction: In vehicular edge
networks, accurate prediction of requested content at each
RSU is essential for effective proactive caching. However,
uncertainty about the vehicles’ mobility translates to uncer-
tainty about future content requests. To tackle this challenge,
we propose the construction of a fi

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

simulations based on
a real-world GPS dataset of 10,357 taxi trajectories in Beijing
demonstrate that VOPP significantly reduces transmission delay
and achieves robust performance across diverse mobility patterns
and user densities, outperforming baseline methods.
Index Terms—proactive caching, content prefetching, online
caching, vehicular edge network
I. I NTRODUCTION
The Internet of V ehicles (IoV) is gradually changing our
lifestyle, enabling vehicle occupants to easily access various
forms of online content. In this context, content caching is a
critical concern in order to maintain high Quality of Service
(QoS) levels in IoV content deliveries. By distributing content
at locations near the vehicles, e.g., at macro base stations
(MBS) and roadside units (RSU) provided by the Internet
service providers (ISP), quick responses to user requests can
be achieved. Suitable caching strategies are required to ensure
that the requested content is available with low delay at the
time and place where relevant requests occur.
Suitable caching decisions depend on various environmental
factors such as the vehicles’ geographical distribution, the
users’ request pattern, MBS and RSU storage and network
constraints, and the contents’ storage sizes. Caching strategies
must jointly consider these factors while respecting overall
cost constraints. This becomes particularly challenging in
large

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 未提及 |
| 服务更新周期 | 是 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 是 |
| 位置预测误差 | 未提及 |
| 服务迁移 | 未提及 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 未提及 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 未提及 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=未提及；服务更新周期=是；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=未提及 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：3（PDF页）
- 方法窗口：4（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：None（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（9页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
