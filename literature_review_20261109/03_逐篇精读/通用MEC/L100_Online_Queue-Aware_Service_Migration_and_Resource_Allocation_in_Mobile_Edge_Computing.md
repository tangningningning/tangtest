# L100 — Online Queue-Aware Service Migration and Resource Allocation in Mobile Edge Computing

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2025
- 刊物/会议：IEEE Transactions on Vehicular Technology
- DOI：10.1109/TVT.2024.3524747
- PDF页数：16
- 原文：`C:\Users\tangning\Desktop\文献\Online_Queue-Aware_Service_Migration_and_Resource_Allocation_in_Mobile_Edge_Computing.pdf`
- 领域分类：**C**；依据：摘要/模型段 MEC=23, vehicle=1, UAV=0
- 主研究方向：P1；次方向：2,3；依据：scores={'1': 48, '2': 34, '3': 26}; title=Online Queue-Aware Service Migration and Resource Allocation in Mobile Edge Comp

## 2. 研究场景与问题

### 摘要窗口

（PDF p3起）

system model and formulate a
long-term average energy consumption minimization problem.
In Section IV, we develop an online optimization framework
and its theoretical analysis. In Section V, we carry out ample
simulations and contrastive analysis for our numerical results
to verify the effectiveness of the proposed algorithms. Finally,
Section VI concludes this paper.
II. R ELA TEDWORK
The great advance in 5G communications and IoT technolo-
gies boosts emerging IoT applications with stringent timeliness
yet resource-hungry requirements[22]. Although the integration
of MEC and NFV alleviates the issues of enhancing resource
capacity and facilitating flexibility of resource management, the
diversity of applications requires different services to be placed
in networks on demand [23]. Meanwhile, resources in networks
need to be allocated efficiently to guarantee service quality
in a cost-ef

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p3起）

system model and formulate a
long-term average energy consumption minimization problem.
In Section IV, we develop an online optimization framework
and its theoretical analysis. In Section V, we carry out ample
simulations and contrastive analysis for our numerical results
to verify the effectiveness of the proposed algorithms. Finally,
Section VI concludes this paper.
II. R ELA TEDWORK
The great advance in 5G communications and IoT technolo-
gies boosts emerging IoT applications with stringent timeliness
yet resource-hungry requirements[22]. Although the integration
of MEC and NFV alleviates the issues of enhancing resource
capacity and facilitating flexibility of resource management, the
diversity of applications requires different services to be placed
in networks on demand [23]. Meanwhile, resources in networks
need to be allocated efficiently to guarantee service quality
in a cost-efficient manner. In recent years, service placement
and resource allocation have drawn extensive attention in many
research works.
A. Service Placement in MEC Networks
The service placement was widely investigated in the service
function chain (SFC) embedding problem, i.e., mapping the
virtual node/VNF to the physical node. For instance, considering
the dynamic service request arrivals, the authors in[24] proposed
an online optimization algorithm to balance the SFC embedding
cost and network stability. To provide users with satisfactory
services and maintain SFC enforcement, several efforts have
been attempted on NFV -enabled MEC networks [25].H o w -
ever, the SFC embedding problem mainly focuses on provid-
ing complicated and integrated network services in multi-hop
network propagation, while the service placement problem in
NFV -enabled MEC networks concentrates more on providing
low-l

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p4起）

problem formulation in Sections
III-A and III-B, respectively. Some key parameter notations are
summarized in Table II.
A. System Model
1) NFV-Enabled MEC Network Model: For the service mi-
gration and resource allocation in an NFV -enabled MEC net-
work, we consider a hierarchical architecture, including the
substrate network and the virtualization layer, as shown in
Fig. 1. Each base station (BS) of the MEC network is en-
dowed with a high-performance edge server, which enables
load balancing and high-quality services provisioning for IoT
devices. We consider the NFV -enabled MEC network as G =
(S,N,E), where S = {1, 2,...,S } denotes the set of edge
servers, N = {1, 2,...,N } denotes the set of IoT devices,
andE = {ei,j,i,j ∈S ,i ⁄= j} denotes the set of backhaul links
between ESs [11]. Each s ∈S is an edge server as well as its
co-located BS. Multiple network functions requested by IoT
devices are implemented by various VNFs running on edge
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on November 03,2026 at 13:41:46 UTC from IEEE Xplore. Restrictions apply. 
DU et al.: ONLINE QUEUE-AW ARE SERVICE MIGRA TION AND RESOURCE ALLOCA TION 8067
servers. For example, a network function of data preprocessing is
deployed to perform stratified sampling of offloaded data. Also,
an intrusion detection system is deployed into one of the edge
servers to identify possible intrusions. Physically, edge comput-
ing is assumed to be deployed in a fully distributed manner. Each
edge server s ∈S provides only limited computing resources
to implement VNFs, which is upper bounded by the maximum
valueFs (in CPU cycles per second). Because network providers
generally operate during a long period, we assume the system
runs within a finite time horizo

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p3起）

proposed algorithms. Finally,
Section VI concludes this paper.
II. R ELA TEDWORK
The great advance in 5G communications and IoT technolo-
gies boosts emerging IoT applications with stringent timeliness
yet resource-hungry requirements[22]. Although the integration
of MEC and NFV alleviates the issues of enhancing resource
capacity and facilitating flexibility of resource management, the
diversity of applications requires different services to be placed
in networks on demand [23]. Meanwhile, resources in networks
need to be allocated efficiently to guarantee service quality
in a cost-efficient manner. In recent years, service placement
and resource allocation have drawn extensive attention in many
research works.
A. Service Placement in MEC Networks
The service placement was widely investigated in the service
function chain (SFC) embedding problem, i.e., mapping the
virtual node/VNF to the physical node. For instance, considering
the dynamic service request arrivals, the authors in[24] proposed
an online optimization algorithm to balance the SFC embedding
cost and network stability. To provide users with satisfactory
services and maintain SFC enforcement, several efforts have
been attempted on NFV -enabled MEC networks [25].H o w -
ever, the SFC embedding problem mainly focuses on provid-
ing complicated and integrated network services in multi-hop
network propagation, while the

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p3起）

simulations to evaluate the per-
formance of our proposed framework. Simulation results
demonstrate that it can rapidly converge to optimal results
and achieve a superior balance between queue stability
and cost consumption compared to the other proposed
heuristic benchmarks and approximate algorithms.
The remainder of the paper is organized as follows. In
Section II, we present a brief overview of the related litera-
ture. In Section III, we give the system model and formulate a
long-term average energy consumption minimization problem.
In Section IV, we develop an online optimization framework
and its theoretical analysis. In Section V, we carry out ample
simulations and contrastive analysis for our numerical results
to verify the effectiveness of the proposed algorithms. Finally,
Section VI concludes this paper.
II. R ELA TEDWORK
The great advance in 5G communications and IoT technolo-
gies boosts emerging IoT applications with stringent timeliness
yet resource-hungry requirements[22]. Although the integration
of MEC and NFV alleviates the issues of enhancing resource
capacity and facilitating flexibility of resource management, the
diversity of applications requires different services to be placed
in networks on demand [23]. Meanwhile, resources in networks
need to be allocated efficiently to guarantee service quality
in a cost-efficient manner. In recent years, service pla

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 是 |
| 服务更新周期 | 部分 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 是 |
| 位置预测误差 | 未提及 |
| 服务迁移 | 是 |
| 活动任务连续执行 | 部分 |
| 迁移传输 | 是 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 是 |
| 信道动态变化 | 是 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=部分；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=是；活动任务连续=部分；迁移传输=是 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=是；信道动态=是 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：3（PDF页）
- 问题窗口：4（PDF页）
- 方法窗口：3（PDF页）
- 仿真窗口：3（PDF页）
- 结论窗口：None（PDF页）

### 关键词命中片段

- p1: IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY , VOL. 74, NO. 5, MA Y 2025 8063 Online Queue-Aware Service Migration and Resource Allocation in Mobile Edge Computing An Du, Jie Jia , Member , IEEE,J i a nC h e n , Member , IEEE, Xingwei Wang , and Min Huang , Member , IEEE Abstract—Mobile edge computing (MEC) integrated with Net- work
- p1: ervice satisfaction of diverse applications. However, a critical challenge lies in processing extremely unpredictable and bursty trafﬁc ﬂow, especially in event-trigger sensing tasks. In this paper, we study queue-aware service migration and resource allo- cation in an MEC network, where different devices with different arrival patterns request dif

## 12. 阅读闭合状态

- 全文文本：已提取（16页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
