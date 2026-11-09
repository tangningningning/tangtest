# L077 — Joint Hybrid Caching and Replacement Scheme for UAV-Assisted Vehicular Edge Computing Networks

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2024
- 刊物/会议：IEEE Transactions on Intelligent Vehicles
- DOI：10.1109/TIV.2023.3323217
- PDF页数：13
- 原文：`C:\Users\tangning\Desktop\文献\01_时延约束下服务部署与任务卸载的自适应多时间尺度协同\Joint Hybrid Caching and Replacement Scheme for UAV-Assisted Vehicular Edge Computing Networks.pdf`
- 领域分类：**B**；依据：摘要/模型段 vehicle=41, UAV=1
- 主研究方向：P1；次方向：无；依据：scores={'1': 54, '2': 11, '3': 5}; title=Joint Hybrid Caching and Replacement Scheme for UAV-Assisted Vehicular Edge Comp

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model. In Section IV, a hybrid caching data
selection and task offloading scheme is analyzed. SectionV elab-
orates a joint optimization problem for the caching replacement
scheme. Numerical examples are given in Section VI to show
the performance evaluation of the proposed schemes. We draw
the conclusions in Section VII.
II. R ELA TEDWORK
In this article, we design a hybrid caching and replacement
scheme for the UA V -assisted VECNs to support the low-latency
applications of the covered vehicle users on the congestion roads.
In this section, we review the existing relevant works for the
UA V -assisted VECNs and the caching schemes in VECNs.
A. UA V-Assisted VECNs
Since UA V -assisted VECNs can provide continuous and
timely network service for the covered vehicle users in the
dynamic moving environment, it has received an increasing
amount of attention in recent years. Due to the

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model. In Section IV, a hybrid caching data
selection and task offloading scheme is analyzed. SectionV elab-
orates a joint optimization problem for the caching replacement
scheme. Numerical examples are given in Section VI to show
the performance evaluation of the proposed schemes. We draw
the conclusions in Section VII.
II. R ELA TEDWORK
In this article, we design a hybrid caching and replacement
scheme for the UA V -assisted VECNs to support the low-latency
applications of the covered vehicle users on the congestion roads.
In this section, we review the existing relevant works for the
UA V -assisted VECNs and the caching schemes in VECNs.
A. UA V-Assisted VECNs
Since UA V -assisted VECNs can provide continuous and
timely network service for the covered vehicle users in the
dynamic moving environment, it has received an increasing
amount of attention in recent years. Due to the easy deployment
and high mobility, UA V can perform as edge node or relay,
and cooperate with the RSUs to cover the users, especially at
some intersections where traffic congestion occurs suddenly.
For the limited UA V energy budget, in [18], an energy efficient
UA V -enabled computing-communication intelligent offloading
scheme was proposed. The ground users were divided into task
gathering node and others, based on the amount of data. The
UA V flies to the former nodes only. Moreover, the stochastic
task offloading and resource allocation scheme had proposed
in [19].I n [20], the UA V flying trajectory and ground users’
task uploading powers were jointly optimized. Since the dy-
namic topology of VECNs, the freshness of data packets was
considered in [21], an age of information (AoI) aware UA V
deployment problem was analyzed. Compared with the single
UA V , multi-UA V can provide powe

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p6起）

problem formulation,
the task offloading decisions are associated with whether the
corresponding service/content caching data had been cached
directly. From the perspective of the vehicle user, the hybrid
caching and task offloading decisions are coupled with each
other, and the computational resources of the edge nodes to
handle the tasks are random. Thus, we design a DQN-based
algorithm to find the optimal solutions, the block diagram is
shown as Fig. 2. P1 is formulated as a mixed integer nonlinear
programming problem, which has non-polynomial computation
complexity, it is difficult to be solved by general algorithms.
The overall framework of the DQN is shown in Fig.2, and the
design of DRL in the task offloading processing strategy is as
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 26,2026 at 12:01:42 UTC from IEEE Xplore. Restrictions apply. 
872 IEEE TRANSACTIONS ON INTELLIGENT VEHICLES, VOL. 9, NO. 1, JANUARY 2024
follows. Define a five-tuple M=< S, A, R, P,Π > to represent
the Markov decision process( MDP) firstly, as:
State: S =<n uav,duav,nrsu,drsu,α,t uav >, where nuav
and nrsu denote the number of tasks be offloaded to the UA V
and RSU, respectively, and duav,drsu denote the straight-line
distances of V2U and V2R. α is the proportion of service
caching and content caching in UA V , tuav denotes the total
task processing latency to UA V . And the tuple parameters are
interrelated and affect each other. For example, the distanceduav
of V2U and the number of tasks nuav offloaded to UA V affect
the total task completion delay tuav directly.
Action: A =<a 1,a2,...,a Φ >, which indicates the differ-
ent types of service/content be cached at UA V . a1,a2,...,a Φ
represent services or contents of different types

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p1起）

proposed algorithms have good convergence, and the
designed schemes reduce the task completion delay efficiently.
Index Terms —Caching replacement, DDQN, DQN, hybrid
caching, UA V, vehicular edge computing networks.
I. I NTRODUCTION
A
LONG with the development of Internet of Things (IoT)
and intelligent transportation technologies, the vehicles
become smarter, safer and more efficient, with a plethora of
emerging applications for vehicle users (i.e., the vehicles and
passengers), such as: intelligent assisted driving, on-line game.
The smooth implementation of these applications depends on
massive data sharing and fast task processing among users,
roadside units (RSUs) and cloud computing center in intelli-
gent transportation systems (ITSs) [1], [2]. To cope with the
generated huge delay-sensitive and computation-intensive tasks,
Manuscript received 15 September 2023; accepted 24 September 2023. Date
of publication 9 October 2023; date of current version 23 February 2024. This
work was supported by the National Natural Science Foundation of China under
Grants 62003094 and 62003099. (Corresponding author: Chao Yang.)
The authors are with the Guangdong Key Laboratory of IoT Informa-
tion Technology, School of Automation, Guangdong University of Tech-
nology, Guangzhou 510000, China (e-mail: 2112104094@mail2.gdut.edu.
cn; yangchaoscut@aliyun.com; xinchen@gdut.edu.cn; 2112204070@m

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

simulation results
show that the proposed algorithms have good convergence, and the
designed schemes reduce the task completion delay efficiently.
Index Terms —Caching replacement, DDQN, DQN, hybrid
caching, UA V, vehicular edge computing networks.
I. I NTRODUCTION
A
LONG with the development of Internet of Things (IoT)
and intelligent transportation technologies, the vehicles
become smarter, safer and more efficient, with a plethora of
emerging applications for vehicle users (i.e., the vehicles and
passengers), such as: intelligent assisted driving, on-line game.
The smooth implementation of these applications depends on
massive data sharing and fast task processing among users,
roadside units (RSUs) and cloud computing center in intelli-
gent transportation systems (ITSs) [1], [2]. To cope with the
generated huge delay-sensitive and computation-intensive tasks,
Manuscript received 15 September 2023; accepted 24 September 2023. Date
of publication 9 October 2023; date of current version 23 February 2024. This
work was supported by the National Natural Science Foundation of China under
Grants 62003094 and 62003099. (Corresponding author: Chao Yang.)
The authors are with the Guangdong Key Laboratory of IoT Informa-
tion Technology, School of Automation, Guangdong University of Tech-
nology, Guangzhou 510000, China (e-mail: 2112104094@mail2.gdut.edu.
cn; yangchaoscut@aliyun.com;

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 是 |
| 服务更新周期 | 未提及 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 是 |
| 位置预测误差 | 未提及 |
| 服务迁移 | 未提及 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 是 |
| 多业务QoS | 部分 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 是 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=未提及；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=是 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=部分；业务流量非平稳=未提及；信道动态=是 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：6（PDF页）
- 方法窗口：1（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：2（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（13页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
