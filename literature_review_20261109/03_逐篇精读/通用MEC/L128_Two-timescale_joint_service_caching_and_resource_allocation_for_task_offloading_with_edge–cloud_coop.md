# L128 — Two-timescale joint service caching and resource allocation for task offloading with edge–cloud cooperation

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：Yafei Li
- 年份：2024
- 刊物/会议：Computer Networks
- DOI：10.1016/j.comnet.2024.110771
- PDF页数：14
- 原文：`C:\Users\tangning\Desktop\文献\18篇指定论文合集\Two-timescale Joint Service Caching and Resource Allocation for Task Offloading with Edge–Cloud Cooperation.pdf`
- 领域分类：**C**；依据：摘要/模型段 vehicle=1, UAV=4, MEC=3
- 主研究方向：P1；次方向：无；依据：scores={'1': 151, '2': 2, '3': 5}; title=Two-timescale joint service caching and resource allocation for task offloading 

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model and
optimization problem are described in detail. In Section 4, we propose
an online two-timescale optimization strategy. The numerical results
and performance evaluations are presented in Section 5. Finally, we
conclude this paper in Section 6.
2. Related work
2.1. Task offloading and service caching
Task offloading research focuses on how to transfer terminal tasks
to edge clouds, aiming to reduce latency and energy consumption dur-
ing task processing, thereby enhancing network service performance.
In [12–15], the fundamental edge network task offloading scheme that
utilizes edge network resources was studied. Furthermore, in [16,17]
an innovative task offloading scheme with resource allocation was
presented, which significantly reduced task execution time and energy
consumption. Building on this, Shahidinejad et al. [18] keenly captures
the impact of system context infor

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model and
optimization problem are described in detail. In Section 4, we propose
an online two-timescale optimization strategy. The numerical results
and performance evaluations are presented in Section 5. Finally, we
conclude this paper in Section 6.
2. Related work
2.1. Task offloading and service caching
Task offloading research focuses on how to transfer terminal tasks
to edge clouds, aiming to reduce latency and energy consumption dur-
ing task processing, thereby enhancing network service performance.
In [12–15], the fundamental edge network task offloading scheme that
utilizes edge network resources was studied. Furthermore, in [16,17]
an innovative task offloading scheme with resource allocation was
presented, which significantly reduced task execution time and energy
consumption. Building on this, Shahidinejad et al. [18] keenly captures
the impact of system context information on offloading decisions and
proposes a context-aware multi-user offloading scheme, further reduc-
ing task execution costs. However, in practice, numerous tasks require
specific service programs for execution, prompting researchers to focus
on the edge caching mechanisms due to the constrained cache space of
edge servers in recent years [19,20]. For instance, Salimianet al. [21]
investigated a method for autonomous service placement in edge net-
work environments, significantly enhancing system performance while
meeting execution cost constraints. The research in [5,22–24] captured
the important features of the mismatch between storage and computing
resources in edge clouds and explored cooperative caching. Researchers
in [25,26] further acknowledged the significance of edge resources
and proposed joint optimization schemes for service caching, resource
allocation, and computation

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p3起）

problem formulation
3.1. Overview of the edge–cloud cooperation architecture
We consider an edge–cloud and edge–edge cooperation network to
manage intelligent and diverse task offloading in the context of 6G
networks through dynamic service caching and computing resource
allocation. As illustrated in Fig. 2 , the network model consists of 𝑁
base stations (BSs) and a central cloud server. Each BS is equipped
with an edge server, and all the edge servers are connected via BS-
BS links. Moreover, the edge servers and the central cloud server can
communicate with each other over BS-Central links. Consequently,
edge–edge and edge–cloud interconnection is realized in the network.
The total system operation time is divided into discrete time slots
 = {1, …,𝜏 }, where the duration of each time slot matches the
timescale of the task offloading decision update. Acknowledging the
inherent differences in the optimization periods of the system decisions,
we introduce the concept of a large timescale  = {1, …,𝑇 }. For the
large timescale𝑇 =𝜃𝜏,𝜃 is a constant. Specifically, the length of 𝜃 can
be adjusted based on the cache content size and the task request rate.
We denote the entirety of the edge clouds (formed by combinations
of BSs and their associated edge servers) as the set = {1, …,𝑁 }. Each
edge cloud 𝑛 possesses finite computing resources 𝐶𝑛 (in terms of CPU
cycles/s), storage resources 𝐺𝑛 (in terms of GB), and communication
resources 𝐵𝑛 (in terms of GB). We assume there are 𝐾 types of tasks
requested by the terminals, denoted by the set  = {1, …,𝐾 }. A task of
type𝑘 has two attributes: (1) the amount of data transferred (in terms
of Mbit), which follows an exponential distribution with an expectation
of𝑟𝑘, and (2) the workload of computation (in terms of CPU cycles/s),
wh

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p1起）

proposed algorithm can converge to an approximate optimal solution in polynomial
time, and the superiority of our scheme is validated by extensive simulation experiments.
1. Introduction
In recent years, edge–cloud cooperative computing networks have
been regarded as potential solutions for realizing 6G edge intelli-
gent scenarios such as digital twins, sensory interconnection, and
holographic communication, among others [ 1]. In particular, service
caching-based task offloading schemes have garnered extensive at-
tention from diverse research institutions globally, owing to their
significant improvements in system offloading efficiency [2]. However,
existing service caching-based task offloading schemes within edge–
cloud cooperative networks encounter challenges such as the separation
of edge–edge task offloading [ 3,4] and the absence of intra-edge
resource allocation [ 5,6], preventing the full utilization of overall
network resources. Meanwhile, we introduce a new research dimension
namely updating timescales, and keenly capture the often-overlooked
issue of additional overhead caused by the single optimization cycle [7,
8]. These issues impede the ability of networks to meet the diverse and
intelligent demands of 6G networks.
Hence, actively exploring the inherent differences in decision op-
timization cycles and effectively leveraging overall network resources
∗ Corresp

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

simulation experiments.
1. Introduction
In recent years, edge–cloud cooperative computing networks have
been regarded as potential solutions for realizing 6G edge intelli-
gent scenarios such as digital twins, sensory interconnection, and
holographic communication, among others [ 1]. In particular, service
caching-based task offloading schemes have garnered extensive at-
tention from diverse research institutions globally, owing to their
significant improvements in system offloading efficiency [2]. However,
existing service caching-based task offloading schemes within edge–
cloud cooperative networks encounter challenges such as the separation
of edge–edge task offloading [ 3,4] and the absence of intra-edge
resource allocation [ 5,6], preventing the full utilization of overall
network resources. Meanwhile, we introduce a new research dimension
namely updating timescales, and keenly capture the often-overlooked
issue of additional overhead caused by the single optimization cycle [7,
8]. These issues impede the ability of networks to meet the diverse and
intelligent demands of 6G networks.
Hence, actively exploring the inherent differences in decision op-
timization cycles and effectively leveraging overall network resources
∗ Corresponding author.
E-mail address: zwq19960307@hrbeu.edu.cn (W. Zheng).
are paramount for meeting the complex and dynamic task offloading
demands of 6G

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
| 多业务QoS | 部分 |
| 业务流量非平稳 | 是 |
| 信道动态变化 | 未提及 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=未提及；服务更新周期=是；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=部分；业务流量非平稳=是；信道动态=未提及 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：3（PDF页）
- 方法窗口：1（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：7（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（14页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
