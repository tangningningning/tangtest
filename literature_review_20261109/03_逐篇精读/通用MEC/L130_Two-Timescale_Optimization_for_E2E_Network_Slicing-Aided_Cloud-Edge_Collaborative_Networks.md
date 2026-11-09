# L130 — Two-Timescale Optimization for E2E Network Slicing-Aided Cloud-Edge Collaborative Networks

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2025
- 刊物/会议：IEEE Transactions on Vehicular Technology
- DOI：10.1109/TVT.2025.3542407
- PDF页数：13
- 原文：`C:\Users\tangning\Desktop\文献\03_多业务非平稳环境下的服务部署与传输资源在线协同\Two-Timescale Optimization for E2E Network Slicing-Aided Cloud-Edge Collaborative Networks.pdf`
- 领域分类：**C**；依据：摘要/模型段 vehicle=1, UAV=0, MEC=2
- 主研究方向：P1；次方向：3；依据：scores={'1': 9, '2': 1, '3': 5}; title=Two-Timescale Optimization for E2E Network Slicing-Aided Cloud-Edge Collaborativ

## 2. 研究场景与问题

### 摘要窗口

（PDF p7起）

network model, we will analyze the computational com-
plexity of the double-layered RCDDQN-RCDDPG algorithm.
The input layers, output layers and hidden layers in the deep Q
network of the RCDDQN algorithm as well as the actor and critic
networks of the RCDDPG algorithm are all fully connected. We
assume that the RCDDQN-RCDDPG algorithm will converge
after V episodes and E time epochs.
For the RCDDQN algorithm in the upper layer, the deep Q
network takes the users’ service request states as input, and
outputs the rewards under different possible values of the vBS-A
and CC-EC scheduling indicators. We denote the dimensions of
the input and output of the deep Q network as |Sup| and |Aup|,
respectively. Besides, the number of layers of the deep Q network
is denoted by NQ, and the number of neurons in the l-th layer
of the deep Q network is denoted by nlQ. Hence, we have the
complexity of the

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p7起）

network model, we will analyze the computational com-
plexity of the double-layered RCDDQN-RCDDPG algorithm.
The input layers, output layers and hidden layers in the deep Q
network of the RCDDQN algorithm as well as the actor and critic
networks of the RCDDPG algorithm are all fully connected. We
assume that the RCDDQN-RCDDPG algorithm will converge
after V episodes and E time epochs.
For the RCDDQN algorithm in the upper layer, the deep Q
network takes the users’ service request states as input, and
outputs the rewards under different possible values of the vBS-A
and CC-EC scheduling indicators. We denote the dimensions of
the input and output of the deep Q network as |Sup| and |Aup|,
respectively. Besides, the number of layers of the deep Q network
is denoted by NQ, and the number of neurons in the l-th layer
of the deep Q network is denoted by nlQ. Hence, we have the
complexity of the RCDDQN algorithm as
O
⎛
⎝VE
⎛
⎝n 1Q|Sup|+
NQ−1∑
l=1
nlQnl+1Q +nNQQ |Aup|
⎞
⎠
⎞
⎠. (23)
For the RCDDPG algorithm, the actor network takes the state
space of the lower layer as input and outputs the action space,
while the critic network takes both the state and action spaces of
the lower layer as input and outputs the reward. The dimensions
of the lower layer state and action spaces are denoted by |Slw|
and |Alw|, respectively. Then, the dimensions of the input and
output of the actor network are |Slw| and |Alw|, respectively;
the dimensions of the input and output of the critic network
are |Slw|+|Alw| and 1, respectively. We denote the numbers
of layers of the actor and critic networks by NA and NC ,
Algorithm 3: V alidating Stage of the RCDDQN-RCDDPG
Algorithm.
1: Initialize the revenue weights: m1(0)←m01,m2(0)←m02;
2: Obtain the initial state space of the upper layer s0up;
3: ForT = 0

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p4起）

optimization problem P∗ is an NP-hard, non-convex
problem (The proofs for the NP-hardness and non-convexity of
P∗ will be shown in AppendixA and Appendix B, respectively),
making it difficult to solve using traditional convex optimization
methods. Therefore, the deep reinforcement learning (DRL) al-
gorithm is used to solve it. Meanwhile, the problem contains both
the continuous and discrete variables, which vary in different
timescales and make it hard to solve P∗ directly. Hence, we
decouple the optimization problem P∗ into the large timescale
joint vBS-A and CC-EC scheduling subproblem P1 and the
small timescale joint communication and computing resources
allocation subproblem P2a s
P1 :m i n
{as(T),ζs(T)}−U(T),
s.t. C1, (12)
Authorized licensed use limited to: Shaanxi Normal University. Downloaded on June 02,2026 at 03:10:41 UTC from IEEE Xplore. Restrictions apply. 
W ANG et al.: TWO-TIMESCALE OPTIMIZA TION FOR E2E NETWORK SLICING-AIDED CLOUD-EDGE COLLABORA TIVE NETWORKS 9781
Fig. 2. RCDDQN-RCDDPG Algorithm.
and
P2 :m i n
{Hs(t),P s(t),f s(t)}−U(t),
s.t. C2−C4,C 5. (13)
respectively.
B. Double-Layered DRL-Based Utility Maximization
We propose a double-layered RCDDQN-RCDDPG algorithm
as shown in Fig. 2 to solve the two-timescale utility maximiza-
tion problem. Specifically, the RCDDQN algorithm is used to
solve subproblem P1 with the discrete variables as(T) and
ζs(T) in the upper layer, while the RCDDPG algorithm is
employed to solve subproblemP2 with the continuous variables
Hs(t),Ps(t) andfs(t) in the lower layer. Our algorithm con-
tains the training stage and validating stage. The basic revenue
weight m0s of slice s is used during the training stage of our
algorithm and also used as the initial weight of the validating
stage.
1) Upper Layer: RCDDQN-Based Joint

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

未定位到章节；需回原文核查。

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p2起）

simulation results are presented. Finally, we
conclude the paper in Section VI.
Fig. 1. E2E-CECN Model.
II. E2E-CECN M ODEL
The E2E-CECN model shown in Fig. 1 is composed of a CS,
NB BSs each attached with an ES, and NK users. Without loss
of generality, we define two E2E network slices: Slice 1 for high
throughput service, such as high-definition video [9], and slice
2 for low delay service, such as smooth video [10].
We first present the workflow of the E2E-CECN model ex-
emplified by video services, but the model can be general-
ized to other scenarios such as Internet of vehicles and vir-
tual/augmented reality. The compressed videos are stored in
the cloud data center. Upon request by the users, the stored
videos will undergo the data transmission and data decoding
procedures. During data transmission, the requested videos will
first be transmitted from the cloud to the BSs via the BLs, and
then transmitted to the users over the wireless access links. The
compressed videos will be decoded into the required definition
version [11] either by the CS before the BL transmission or by
the ESs before the access link transmission. Decoding a video by
an ES saves the BLC cost for its BL transmission than decoding
it by the CS [12]. The main variables and sets of this paper are
listed in Table II.
Authorized licensed use limited to: Shaanxi Normal University. Downloaded on June 02,2

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 部分 |
| 服务更新周期 | 是 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 部分 |
| 位置预测误差 | 未提及 |
| 服务迁移 | 未提及 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 是 |
| 多业务QoS | 未提及 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 部分 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=部分；服务更新周期=是；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=部分；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=是 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=未提及；业务流量非平稳=未提及；信道动态=部分 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：7（PDF页）
- 问题窗口：4（PDF页）
- 方法窗口：None（PDF页）
- 仿真窗口：2（PDF页）
- 结论窗口：None（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（13页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
