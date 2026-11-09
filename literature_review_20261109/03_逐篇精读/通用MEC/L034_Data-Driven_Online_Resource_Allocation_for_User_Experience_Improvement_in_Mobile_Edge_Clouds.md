# L034 — Data-Driven Online Resource Allocation for User Experience Improvement in Mobile Edge Clouds

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2024
- 刊物/会议：IEEE Transactions on Wireless Communications
- DOI：10.1109/TWC.2024.3403996
- PDF页数：15
- 原文：`C:\Users\tangning\Desktop\文献\Data-Driven_Online_Resource_Allocation_for_User_Experience_Improvement_in_Mobile_Edge_Clouds.pdf`
- 领域分类：**C**；依据：摘要/模型段 MEC=11, vehicle=0, UAV=0
- 主研究方向：P3；次方向：无；依据：scores={'1': 2, '2': 0, '3': 5}; title=Data-Driven Online Resource Allocation for User Experience Improvement in Mobile

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model of the QoE-based resource
allocation in the MEC system. Section III gives the CORA
framework to model the QoE-based resource allocation prob-
lem. The proposed algorithms to solve the above resource
allocation problem are given in Section IV for the homogenous
users and in Section V for the heterogeneous users. The
numerical results are presented in Section VI, and this paper
concludes in Section VII.
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on November 03,2026 at 13:36:29 UTC from IEEE Xplore. Restrictions apply. 
FU et al.: DATA-DRIVEN ONLINE RESOURCE ALLOCATION FOR USER EXPERIENCE IMPROVEMENT IN MEC 13709
Fig. 1. An illustration of the mobile edge cloud system.
II. S YSTEM MODEL
We consider an MEC system, as shown in Fig. 1, which
consists of mobile device (or end user), multiple edge server,
and cloud server. The mobile dev

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model of the QoE-based resource
allocation in the MEC system. Section III gives the CORA
framework to model the QoE-based resource allocation prob-
lem. The proposed algorithms to solve the above resource
allocation problem are given in Section IV for the homogenous
users and in Section V for the heterogeneous users. The
numerical results are presented in Section VI, and this paper
concludes in Section VII.
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on November 03,2026 at 13:36:29 UTC from IEEE Xplore. Restrictions apply. 
FU et al.: DATA-DRIVEN ONLINE RESOURCE ALLOCATION FOR USER EXPERIENCE IMPROVEMENT IN MEC 13709
Fig. 1. An illustration of the mobile edge cloud system.
II. S YSTEM MODEL
We consider an MEC system, as shown in Fig. 1, which
consists of mobile device (or end user), multiple edge server,
and cloud server. The mobile device is connected to a BS or
AP, which usually refers to IoT device or smartphone. These
devices may send requests to the BS for services, such as
IoT analytics, video streaming, and cloud gaming. To improve
transmission efficiency and reduce computation latency, the
BS will offload service to the edge servers deployed close to
the BS rather than forward them to the remote cloud server.
After receiving the request, the edge server executes this task
by allocating suitable resources to the user. For example,
the allocated resources can be the bandwidth, computation,
and storage capability, etc. Note that a cloud server usually
deploys at the edge of the mobile network to further extend
the computing capabilities, as shown in Fig. 1.
We assume that time is slotted in t = 1, 2, . . . , T. At each
time slot t, there are St ={1, 2, . . .} users requesting ser-
vice from the edge servers as use

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p7起）

problem formulation of the classifier component and
resource allocation component can remain unchanged except
that the coefficient matrix Z in the expression of g(t) is
replaced by ˆΨ.
B. Robust Online Queue Resource Allocation Algorithm
In the following, we present the ROQRA algorithm to solve
problem P2 for the case of heterogeneous users, as shown in
Algorithm 3. Compared with the OQRA algorithm, we model
the coefficient estimation problem as an MAB problem and
then adopt the UCB algorithm to learn the coefficient matrix
ˆΨ within the time horizon T . In this MAB problem, the edge
server is the player, and the elements in matrix Z are the
arms. At each time slot, a new user enters the MEC system
and establishes a connection with the player. Before resource
allocation, the user will send the coefficient matrix Z(t) (i.e.,
the demand of resource) to the player for service. Then, the
player will observe a reward zd,k(t)rk(t), which reflects the
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on November 03,2026 at 13:36:29 UTC from IEEE Xplore. Restrictions apply. 
13714 IEEE TRANSACTIONS ON WIRELESS COMMUNICATIONS, VOL. 23, NO. 10, OCTOBER 2024
Algorithm 3 Robust Online Queue Resource Allocation
(ROQRA) Algorithm
Input:
θ, ¯r, z(t), Qk(0) = 0, ¯Rd,k(0) = Ld,k(0) = 0,∀d, k;
Construct the classifier model by training the initial
overall dataset to obtain w0,0, w0 and set Q(0) = 0;
for t = 1, 2, . . . , Tdo
Step 1 : Estimate the coefficient matrix ˆΨ using the
UCB algorithm in (24)
Step 2 : Take this coefficient matrix into Alg. 1
Step 3 : Update the overall dataset and the weights
using (25) and (26), respectively.
Step 4 : Update the estimation of ¯ψd,k(t) using (27)
and (28).
end
Output: [r(1), r(2), . . . ,r(T )]
effect of the

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p1起）

proposed algorithms using the synthesis and
Y ouTube datasets. Numerical results validate the theoretical
analysis and demonstrate that the user complaint rate is reduced
by up to 100% and 18% in the synthesis and Y ouTube datasets,
respectively.
Index Terms— Mobile edge clouds (MEC), data-driven online
resource allocation, Lyapunov optimization.
Manuscript received 26 October 2022; revised 20 September
2023 and 19 March 2024; accepted 13 May 2024. Date of publication
30 May 2024; date of current version 11 October 2024. The work of Liqun
Fu was supported in part by the National Natural Science Foundation
of China under Grant U23A20281 and in part by the Open Research
Project Programme of the State Key Laboratory of Internet of Things for
Smart City (University of Macau) under Grant SKLIoTSC(UM)-2021-
2023/0RP/GA03/2022. The work of Jun Zhang was supported by Hong
Kong Research Grants Council through the Areas of Excellence Scheme
under Grant AoE/E-601/22-R and in part by the National Natural Science
Foundation of China (NSFC)/Research Grants Council (RGC) Collaborative
Research Scheme under Grant CRS_HKUST603/22. An earlier version of
this paper was presented in part at the 2021 IEEE International Conference
on Communications (ICC) [DOI: 10.1109/ICC42927.2021.9500905]. The
associate editor coordinating the review of this article and approving it for
publication was X. Gong. (C

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

Numerical results validate the theoretical
analysis and demonstrate that the user complaint rate is reduced
by up to 100% and 18% in the synthesis and Y ouTube datasets,
respectively.
Index Terms— Mobile edge clouds (MEC), data-driven online
resource allocation, Lyapunov optimization.
Manuscript received 26 October 2022; revised 20 September
2023 and 19 March 2024; accepted 13 May 2024. Date of publication
30 May 2024; date of current version 11 October 2024. The work of Liqun
Fu was supported in part by the National Natural Science Foundation
of China under Grant U23A20281 and in part by the Open Research
Project Programme of the State Key Laboratory of Internet of Things for
Smart City (University of Macau) under Grant SKLIoTSC(UM)-2021-
2023/0RP/GA03/2022. The work of Jun Zhang was supported by Hong
Kong Research Grants Council through the Areas of Excellence Scheme
under Grant AoE/E-601/22-R and in part by the National Natural Science
Foundation of China (NSFC)/Research Grants Council (RGC) Collaborative
Research Scheme under Grant CRS_HKUST603/22. An earlier version of
this paper was presented in part at the 2021 IEEE International Conference
on Communications (ICC) [DOI: 10.1109/ICC42927.2021.9500905]. The
associate editor coordinating the review of this article and approving it for
publication was X. Gong. (Corresponding author: Liqun Fu.)
Liqun Fu and Tongtong Lin are w

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 未提及 |
| 服务更新周期 | 未提及 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 是 |
| 位置预测误差 | 未提及 |
| 服务迁移 | 未提及 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 部分 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 未提及 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=未提及；服务更新周期=未提及；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=部分 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=未提及 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：7（PDF页）
- 方法窗口：1（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：None（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（15页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
