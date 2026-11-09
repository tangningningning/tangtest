# L136 — VRCCS-AC: Reinforcement Learning for Service Migration in Vehicular Edge Computing Systems

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2024
- 刊物/会议：IEEE Transactions on Services Computing
- DOI：10.1109/TSC.2024.3407581
- PDF页数：15
- 原文：`C:\Users\tangning\Desktop\文献\02_移动性感知的服务迁移与传输资源协同\VRCCS-AC- Reinforcement Learning for Service Migration in Vehicular Edge Computing Systems.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=29, UAV=0
- 主研究方向：P2；次方向：无；依据：scores={'1': 10, '2': 66, '3': 0}; title=VRCCS-AC: Reinforcement Learning for Service Migration in Vehicular Edge Computi

## 2. 研究场景与问题

### 摘要窗口

（PDF p5起）

system model in VEC, we consider MV as an agent
and define its local state as the previous serving ESserv(t−1),
task information Jv(t), and the upload rate of offloaded tasks
from the MV to the local ESrtrv (t). The local state by MVv can
be expressed using the following (8),
ov(t)=
{
serv(t−1),Jv(t),rtrv (t)). (8)
Further, let S(t)= {o1(t),..., oV (t)} denote the joint local
states of all MVs in time slot t.
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 29,2026 at 11:34:29 UTC from IEEE Xplore. Restrictions apply. 
GAO et al.: VRCCS-AC: REINFORCEMENT LEARNING FOR SERVICE MIGRA TION IN VEHICULAR EDGE COMPUTING SYSTEMS 4441
Fig. 2. The VRCCS-AC algorithm-based service migration solution.
2) The Service Migration Strategy av(t) of MV v: Based on
our VEC scenario, the service migration strategy (i.e., the action)
of the MV agent is defin

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p5起）

system model in VEC, we consider MV as an agent
and define its local state as the previous serving ESserv(t−1),
task information Jv(t), and the upload rate of offloaded tasks
from the MV to the local ESrtrv (t). The local state by MVv can
be expressed using the following (8),
ov(t)=
{
serv(t−1),Jv(t),rtrv (t)). (8)
Further, let S(t)= {o1(t),..., oV (t)} denote the joint local
states of all MVs in time slot t.
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 29,2026 at 11:34:29 UTC from IEEE Xplore. Restrictions apply. 
GAO et al.: VRCCS-AC: REINFORCEMENT LEARNING FOR SERVICE MIGRA TION IN VEHICULAR EDGE COMPUTING SYSTEMS 4441
Fig. 2. The VRCCS-AC algorithm-based service migration solution.
2) The Service Migration Strategy av(t) of MV v: Based on
our VEC scenario, the service migration strategy (i.e., the action)
of the MV agent is defined as av(t).av(t) ∈A denotes the ES
where MV’s service is migrated (av(t) ∈N ).
3) State-Transition Probability P: We defineP to denote the
state-transition probability in the VEC scenario. P denotes the
probability that the current state ov(t) is transferred to the next
state ov(t+1) when the MV agent applies a service migration
strategyav(t) to the VEC scenario.
4) The Reward Rv(t) of MV v: Given the local state ov(t),
MVv utilizes the service migration strategyav(t) to engage with
the VEC system and obtain the corresponding reward. Building
upon the proposed VEC system, we define MV’s reward as a
value related to service migration, computation of offloaded
tasks, and transmission of offloaded tasks delay. Further, MV’s
rewardRv(t) is expressed through the following (9),
Rv(t)=
T∑
t=1
γt ·r(ov(t),av(t)). (9)
We define r(ov(t), av(t)) = −(η1τmdv (t)+ η2τodv (t)+
η3τtrv (t)), where η1τmd

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p2起）

problem formulation for service migration. In Section
IV, we introduce the problem definition and problem formula-
tion for service migration. Section V introduces the VRCCS-
AC algorithm in detail. In Section VI, we assess the perfor-
mance of the VRCCS-AC algorithm. Section VII concludes
the paper.
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 29,2026 at 11:34:29 UTC from IEEE Xplore. Restrictions apply. 
4438 IEEE TRANSACTIONS ON SERVICES COMPUTING, VOL. 17, NO. 6, NOVEMBER/DECEMBER 2024
TABLE I
COMPARISON OF EXISTING METHODS WITH THE VRCCS-AC ALGORITHM
II. R ELA TEDWORKS
Based on the placement of the service migration decision
agent, service migration approaches can be categorized into
centralized and decentralized schemes. Moreover, we further
explore the existing service migration schemes based on transfer
learning algorithms.
A. The Centralized Service Migration Approaches
Chen et al. [2] studied dynamic service migration and request
routing for microservice in multicell MEC and proposed a
Lyapunov optimization-based algorithm. Wang et al. [6] pro-
posed a proactive service migration solution based on Lyapunov
optimization and spatial-temporal mobility prediction in VEC
scenarios. Shah et al. [4] introduced a service mobility manage-
ment method using software-defined networking (SDN)-based
in VEC. Zhou et al. [5] studied energy-efficient service migra-
tion for multi-user heterogeneous dense cellular networks and
proposed a Lyapunov and particle swarm optimizations-based
algorithm. By extending the MARL approach, Liu et al. [1]
introduced a distributed task migration optimization solution
in MEC. Ren et al. [7] proposed a DRL algorithm based on
blockchain-based VEC network trust management for vehicular
servi

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p11起）

proposed algorithm is able to converge faster and obtain
higher average total rewards than the METASM; for example,
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 29,2026 at 11:34:29 UTC from IEEE Xplore. Restrictions apply. 
GAO et al.: VRCCS-AC: REINFORCEMENT LEARNING FOR SERVICE MIGRA TION IN VEHICULAR EDGE COMPUTING SYSTEMS 4447
TABLE VI
THE QUANTITA TIVEBASELINE COMPARISON TABLE USING MEAN OVERALL DELA Y AS A METRIC ACROSS DA TASETS
Fig. 11. Average total reward of the VRCCS-AC and METASM algorithm in
Roma vehicle dataset.
Fig. 12. Average total reward of the VRCCS-AC and METASM algorithm in
San Francisco vehicle dataset.
the proposed service migration scenario has increased the con-
vergence speed by 37.23% and the average total rewards rise by
36.51%. This is due to the fact that we introduce a service migra-
tion strategy reuse module that efficiently reuses the experience
learnt by the model in different VEC scenarios and avoids the
need to train any additional components, thus accelerating the
training of the model and increasing the average total reward.
Moreover, we also investigated the average total reward of the
VRCCS-AC and METASM algorithm in the San Francisco
vehicle dataset. From Fig. 12, we can find that the proposed
algorithm is able to converge faster and obtain higher average
total rewards than the META

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p2起）

simulation
experiments demonstrate that VRCCS-AC dramatically
decreases the latency compared to existing migration solu-
tions. Moreover, VRCCS-AC exhibits transfer capabilities
to new VEC scenarios.
Next, the structure of this paper is as follows. Section II
reviews the related works. Section III provides the system
model and problem formulation for service migration. In Section
IV, we introduce the problem definition and problem formula-
tion for service migration. Section V introduces the VRCCS-
AC algorithm in detail. In Section VI, we assess the perfor-
mance of the VRCCS-AC algorithm. Section VII concludes
the paper.
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 29,2026 at 11:34:29 UTC from IEEE Xplore. Restrictions apply. 
4438 IEEE TRANSACTIONS ON SERVICES COMPUTING, VOL. 17, NO. 6, NOVEMBER/DECEMBER 2024
TABLE I
COMPARISON OF EXISTING METHODS WITH THE VRCCS-AC ALGORITHM
II. R ELA TEDWORKS
Based on the placement of the service migration decision
agent, service migration approaches can be categorized into
centralized and decentralized schemes. Moreover, we further
explore the existing service migration schemes based on transfer
learning algorithms.
A. The Centralized Service Migration Approaches
Chen et al. [2] studied dynamic service migration and request
routing for microservice in multicell MEC and proposed a
Lyapu

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
| 服务迁移 | 是 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 是 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 是 |
| 信道动态变化 | 未提及 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=未提及；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=是；活动任务连续=未提及；迁移传输=是 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=是；信道动态=未提及 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：5（PDF页）
- 问题窗口：2（PDF页）
- 方法窗口：11（PDF页）
- 仿真窗口：2（PDF页）
- 结论窗口：None（PDF页）

### 关键词命中片段

- p1: NSACTIONS ON SERVICES COMPUTING, VOL. 17, NO. 6, NOVEMBER/DECEMBER 2024 VRCCS-AC: Reinforcement Learning for Service Migration in V ehicular Edge Computing Systems Zhen Gao ,L e iY a n g ,a n dY uD a i Abstract—Existing service migration approaches provide mini- mal service delay for mobile vehicles (MVs) in vehicular edge com- puting (VEC) systems
- p1: nd poor scalability. Furthermore, existing approaches are difﬁcult to reuse previous migration strategies and necessitate signiﬁcant interaction to adapt to new VEC scenarios. In this paper, we present a decen- tralized service migration approache base on Variational Recur- rent and Critic-Coached Strategy reuse Actor-Critic (VRCCS-AC). First, a va

## 12. 阅读闭合状态

- 全文文本：已提取（15页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
