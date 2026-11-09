# L111 — Serverless edge intelligence for SLA-aware function placement and resource slicing in wireless networks

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2026
- 刊物/会议：待核
- DOI：10.1038/s41598-026-69441-2
- PDF页数：31
- 原文：`C:\Users\tangning\Desktop\文献\.analysis_work\proposal_revision_sources_tmp\Serverless edge intelligence for SLA-aware function placement and resource slicing in wireless networks.pdf`
- 领域分类：**C**；依据：摘要/模型段 vehicle=0, UAV=0, MEC=1
- 主研究方向：P3；次方向：1；依据：scores={'1': 20, '2': 4, '3': 35}; title=Serverless edge intelligence for SLA-aware function placement and resource slici

## 2. 研究场景与问题

### 摘要窗口

（PDF p1起）

system model captures uplink and downlink transmission, queueing, execution, 
cold-start initialization, and memory occupied by retained function instances. To solve the resulting 
dynamic, partially observable, mixed discrete-continuous problem, Serverless-PRONTO combines a 
physics-aware sparse graph encoder with multi-agent TD3 under centralized training and decentralized 
execution. The encoder represents inter-node coupling through channel, interference, distance, 
queue, resource, function-demand, and cold-start features, while top- 𝐊𝐊 attention limits signaling. An 
SLA-risk mechanism prioritizes requests according to urgency, queue state, service class, and cold-
start probability. A sequential feasibility projection converts raw actor outputs into valid placement, 
bandwidth, CPU, and memory decisions. Simulations against four serverless and edge-orchestration 
baselines under v

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p1起）

system model captures uplink and downlink transmission, queueing, execution, 
cold-start initialization, and memory occupied by retained function instances. To solve the resulting 
dynamic, partially observable, mixed discrete-continuous problem, Serverless-PRONTO combines a 
physics-aware sparse graph encoder with multi-agent TD3 under centralized training and decentralized 
execution. The encoder represents inter-node coupling through channel, interference, distance, 
queue, resource, function-demand, and cold-start features, while top- 𝐊𝐊 attention limits signaling. An 
SLA-risk mechanism prioritizes requests according to urgency, queue state, service class, and cold-
start probability. A sequential feasibility projection converts raw actor outputs into valid placement, 
bandwidth, CPU, and memory decisions. Simulations against four serverless and edge-orchestration 
baselines under varying traffic loads and network sizes show higher SLA satisfaction, lower end-
to-end delay and cold-start ratio, and more stable scalability, demonstrating the benefit of jointly 
coordinating wireless resources and serverless runtime states.
Keywords Wireless edge networks, Edge intelligence, SLA-aware resource slicing, Graph reinforcement 
learning, Multi-agent orchestration
 Wireless edge networks are becoming an essential infrastructure for supporting intelligent and latency-sensitive 
digital services. Applications such as connected mobility, smart surveillance, industrial automation, extended 
reality, and remote healthcare require rapid response, reliable execution, and continuous quality of service 
under highly dynamic operating conditions. These services produce heterogeneous traffic patterns and impose 
diverse service-level agreement (SLA) requirements in terms of latency,

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p3起）

Problem formulation” 
section formulates the optimization problem. “ Serverless-PRONTO: SLA-aware orchestration framework ” 
section introduces the proposed Serverless-PRONTO framework, including its workflow, function placement, 
warm-instance control, priority-aware slicing, and feasibility projection. “ Physics-aware graph-TD3 learning 
model” section presents the physics-aware Graph-TD3 learning model, and “ Algorithms and complexity 
analysis” section provides the algorithms and complexity analysis. “ Numerical results” section reports the 
simulation setup and numerical results. Finally, “ Conclusion” section concludes the paper and outlines future 
research directions.
Related works
Related studies are reviewed in four directions: multi-agent wireless-edge offloading, serverless function 
orchestration, graph-based coordination, and constrained resource allocation. This section critically examines 
the related literature along these technical directions: multi-agent wireless-edge offloading, SLA-aware serverless 
scheduling, and graph-attentive edge coordination. For each direction, we compare its principal capability, 
underlying assumptions, and unresolved limitations with respect to cold-start-aware function placement, warm-
instance management, joint wireless–computing control, scalable coordination, and action feasibility.
Multi-agent offloading in wireless edge networks
Multi-agent reinforcement learning has been widely used for computation offloading in wireless edge networks 
because it allows distributed nodes to make adaptive decisions under changing traffic and resource conditions. 
In centralized-training and decentralized-execution settings, agents can learn from a wider network view during 
training while still making local decisions during deployme

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p2起）

solution must evaluate SLA risk in real time by considering task urgency, function 
cold-start status, queue length, channel condition, interference, and available edge capacity. This requires a joint 
optimization framework in which function placement, task offloading, channel allocation, CPU slicing, and 
warm-instance control are coordinated together.
Existing multi-access edge computing methods usually focus on task offloading, radio allocation, or 
computing allocation separately 9,11. Many of them assume that the controller has full network information, 
while real wireless edge systems are distributed and only partially observable. Graph-based learning methods 
can improve coordination among edge nodes, but many existing designs rely on dense message passing, which 
becomes costly as the number of nodes increases. Moreover, most prior methods do not explicitly include 
serverless-specific factors such as cold-start delay, warm function availability, memory limits, and scaling cost12,13. 
Standard reinforcement learning methods such as DDPG, MADDPG, and conventional actor–critic models 
can support adaptive decision-making, but they do not directly guarantee feasible channel–CPU–memory 
allocation or stable SLA-aware control under dynamic traffic. These limitations motivate a serverless-native, 
SLA-aware orchestration framework that combines function placement, cold-star

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

Simulations against four serverless and edge-orchestration 
baselines under varying traffic loads and network sizes show higher SLA satisfaction, lower end-
to-end delay and cold-start ratio, and more stable scalability, demonstrating the benefit of jointly 
coordinating wireless resources and serverless runtime states.
Keywords Wireless edge networks, Edge intelligence, SLA-aware resource slicing, Graph reinforcement 
learning, Multi-agent orchestration
 Wireless edge networks are becoming an essential infrastructure for supporting intelligent and latency-sensitive 
digital services. Applications such as connected mobility, smart surveillance, industrial automation, extended 
reality, and remote healthcare require rapid response, reliable execution, and continuous quality of service 
under highly dynamic operating conditions. These services produce heterogeneous traffic patterns and impose 
diverse service-level agreement (SLA) requirements in terms of latency, reliability, bandwidth, and computing 
demand. Meeting these requirements is challenging because wireless channels fluctuate over time, interference 
varies with user activity, and edge servers may become congested when many computation-intensive requests 
arrive simultaneously1–3. As a result, future wireless edge platforms require intelligent orchestration mechanisms 
that can jointly manage communication, computing,

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
| 服务迁移 | 是 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 未提及 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 是 |
| 信道动态变化 | 是 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=是；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=是；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=是；信道动态=是 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：1（PDF页）
- 问题窗口：3（PDF页）
- 方法窗口：2（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：3（PDF页）

### 关键词命中片段

- p4: node may affect nearby nodes through interference, traffic migration, queue variation, and shared resource usage. For this reason, graph neural networks and graph attention mechanisms have been used in MEC offloading, service migration, caching, and cooperative edge/ cloud deployment36,37. By representing edge nodes as graph vertices and their inte
- p5: emory control, and feasibility-guaranteed action execution within one framework. Several complementary MEC studies address related deployment and offloading decisions. Shi et al.51 develop a two-timescale strategy for service migration and task rerouting, Yi et al. 52 coordinate multiuser offloading and transmission scheduling, Shi et al.53 use imp

## 12. 阅读闭合状态

- 全文文本：已提取（31页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
