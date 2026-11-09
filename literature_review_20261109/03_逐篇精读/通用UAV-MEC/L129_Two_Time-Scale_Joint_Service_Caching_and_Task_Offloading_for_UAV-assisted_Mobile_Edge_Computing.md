# L129 — Two Time-Scale Joint Service Caching and Task Offloading for UAV-assisted Mobile Edge Computing

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2022
- 刊物/会议：IEEE INFOCOM 2022 - IEEE Conference on Computer Communications
- DOI：10.1109/INFOCOM48880.2022.9796714
- PDF页数：10
- 原文：`C:\Users\tangning\Desktop\文献\01_时延约束下服务部署与任务卸载的自适应多时间尺度协同\Two Time-Scale Joint Service Caching and Task Offloading for UAV-assisted Mobile Edge Computing.pdf`
- 领域分类：**D**；依据：摘要/模型段 MEC=18, vehicle=1, UAV=5
- 主研究方向：P1；次方向：无；依据：scores={'1': 81, '2': 14, '3': 4}; title=Two Time-Scale Joint Service Caching and Task Offloading for UAV-assisted Mobile

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model and problem
formulation in Sec. III. The algorithm is present and evaluated
in Sec. IV and Sec. V, respectively. Sec. VI concludes the
paper.
II. R ELATED WORK
UAV-assisted MEC. UA V-assisted wireless communication
is mainly adopted to realize full converge and enhance net-
work performance. Ji et al. [16] consider the enhancement
of the cell-edge communication through a UA V in multi-cell
networks. Alzenad et al. [18] propose an optimal placement
algorithm for UA V-BSs that maximizes the number of covered
users. Chen et al. [20] use cache-enabled UA Vs to optimize
the quality of experience (QoE) of wireless devices in a cloud
radio access network. Lyu et al. [21] jointly optimize UA V’s
trajectory, bandwidth allocation and user partitioning to max-
imize the minimum throughput of all mobile terminals. Some
researchers consider offloading computing tasks to UA Vs [19]
[22] [

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model and problem
formulation in Sec. III. The algorithm is present and evaluated
in Sec. IV and Sec. V, respectively. Sec. VI concludes the
paper.
II. R ELATED WORK
UAV-assisted MEC. UA V-assisted wireless communication
is mainly adopted to realize full converge and enhance net-
work performance. Ji et al. [16] consider the enhancement
of the cell-edge communication through a UA V in multi-cell
networks. Alzenad et al. [18] propose an optimal placement
algorithm for UA V-BSs that maximizes the number of covered
users. Chen et al. [20] use cache-enabled UA Vs to optimize
the quality of experience (QoE) of wireless devices in a cloud
radio access network. Lyu et al. [21] jointly optimize UA V’s
trajectory, bandwidth allocation and user partitioning to max-
imize the minimum throughput of all mobile terminals. Some
researchers consider offloading computing tasks to UA Vs [19]
[22] [23]. In [19] and [18], the authors try to minimize the
energy consumption of UA Vs. Yanget al. [12] aim to minimize
the sum power in a MEC network with UA Vs by jointly
optimizing user connection, power control and computation
capacity allocation. Both Ji et al. [17] and Wu et al. [24]
maximize the minimum throughput among UA V-served users
by jointly optimizing cache placement, UA V trajectory, and
transmission power. Most of these papers ignore the problem
off service caching. Since many existing popular applications
are data-intensive and require a non-trivial amount of data
stored in UA V’s limited storage space, it is essential to study
the cache placement on UA Vs. What’s more, in this paper we
propose the two time-scale joint task offloading and service
caching algorithm, due to the overhead of caching, which is
ignored by all of the above papers.
Service Caching in MEC. Although

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p2起）

PROBLEM FORMULATION
System Overview. As shown in Fig. 1, we consider a mobile
network ofU rotary-wing UA Vs. These UA Vs, integrating with
edge servers, have both computing and caching capabilities.
The UA Vs can provide K types of services for M user
equipments (UEs). U rotary-wing UA Vs are denoted as U =
{1, 2,...,U}, and M UEs are denoted as M ={1, 2,...,M}.
Let K = {1, 2,...,K} denote the service set provided by
1190
Authorized licensed use limited to: Shaanxi Normal University. Downloaded on June 02,2026 at 10:07:58 UTC from IEEE Xplore. Restrictions apply. 
x
y
z
: UA V 
: UE 
: Services
: UA V coverage
: Cache
: Offload to UA V
: Offload to cloud
Fig. 1: An illustration of UA V-assisted wireless network.
UA Vs. We divide the mission period into several time slots
T =
{
1, 2,..., ~T
}
. The tasks generated from UEs follow a
Poisson process. Let dtm,k denote the intensity (i.e., Poisson
rate) of tasks generated from UE m requiring service k at the
time slott. The tasks can be offloaded to UA Vs only when the
following two conditions are satisfied: i) the UE is connected
with UA Vs; ii) the corresponding service contents are cached
in these UA Vs. Otherwise, UEs will offload the tasks to the
remote cloud for processing.
Service Caching. In order to reduce computation latency,
we consider the scenario where services will be cached in the
UA V’s storage. Letck represent the storage space required by
servicek. Considering the limited storage space of a UA V , not
all services can be cached at the same time. The total storage
space occupied by services should be no more than UA V’s
storage capacity Cu. Since the communication overhead of
caching services from the remote cloud to the UA V is large
[9], the cache decisions are updated in every T time slots,
where T is de

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p4起）

proposed algorithms
consist of following components.
i. We first establish the virtual energy queue Q(t) to repre-
sent the energy constraint (1j). Following the Lyapunov
optimization approach, we derive the T-time-slot drift-
plus-penalty function. Cache decisions are made every
T time slot, but the task arrival rate varies during this
time period. To overcome this obstacle, we approximate
the future energy queue backlog value as the current
value, and derive a loose upper bound of the T-time-
slot drift-plus-penalty function. We obtain the problem
P2 based on the upper bound. Second, we decompose
the T-time-slot problem P2 into multiple one-time opti-
mization problems. We solve the problem Pt at time
t = θT (θ = 1, 2,..., Θ), and problem Pτ at time
τ∈ [t + 1,t +T− 1].
ii. To obtain a suboptimal solution to Pt, we propose an
alternating optimization-based algorithm that iteratively
optimizes two different subproblems. First, with the
fixed service caching A, UE-UA V association X and
task offloading Y , the UA V trajectory subproblem Pt1
is a continuous non-convex problem, which can be opti-
mally solved by the successive convex approximation
(SCA) technique. Second, we fix the UA V trajectory
G, and solve the subproblem Pt2 to joint optimize A,
X and Y . Subproblem Pt2 is a mixed integer linear
programming problem. We relax the problem to get the
fractional solution by inter

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

simulations further verify that our proposed solution can
significantly reduce the service delay for UEs while maintaining
low energy consumption when compared to the three state-of-
the-art baselines.
I. I NTRODUCTION
The fast development of internet of things (IoT) generates
a large amount of data at the edge, which boosts the appli-
cations of mobile edge computing (MEC), e.g., smart city
and automatic driving [1]. These applications requires low
latency and high computation power, which is realized by
offloading services to edge servers at local wireless access
points (APs) or cellular base stations (BSs) [2]. Unfortunately,
the infrastructure-based MEC in wireless networks brings high
deployment cost, and may not be able to provide services
in rural areas without sufficient infrastructures or urban ar-
eas in peak hours [3], [4]. To compensate the drawback of
infrastructure-based MEC, unmanned aerial vehicles (UA Vs)
equipped with MEC servers have emerged as a promising
technology to provide flexible and cost-efficient computing
services in the era of 5G and beyond [4], [5]. UA Vs are
*This work is supported in part by the NSFC Grants (62072344,
U20A20177, 62132009 and 61772489), Hubei Science Foundation
(2020CFB195) and Compact Exponential Algorithm Project of Huawei
(YBN2020035131). The corresponding author is Haisheng Tan.
easy to control and deploy, and have high mobil

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
| 服务迁移 | 部分 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 未提及 |
| 多业务QoS | 未提及 |
| 业务流量非平稳 | 是 |
| 信道动态变化 | 未提及 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=未提及；服务更新周期=是；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=部分；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=未提及；业务流量非平稳=是；信道动态=未提及 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：2（PDF页）
- 方法窗口：4（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：None（PDF页）

### 关键词命中片段

- p10: ication design for multi-uav enabled wireless networks,” IEEE Transactions on Wireless Communications, vol. 17, no. 3, pp. 2109–2121, 2018. [25] R. Urgaonkar, S. Wang, T. He, M. Zafer, K. Chan, and K. K. Leung, “Dynamic service migration and workload scheduling in edge-clouds,” Performance Evaluation, vol. 91, pp. 205–228, 2015. [26] L. Pu, L. Jiao

## 12. 阅读闭合状态

- 全文文本：已提取（10页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
