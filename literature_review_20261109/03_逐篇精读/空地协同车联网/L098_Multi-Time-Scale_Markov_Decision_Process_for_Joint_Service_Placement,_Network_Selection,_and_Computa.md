# L098 — Multi-Time-Scale Markov Decision Process for Joint Service Placement, Network Selection, and Computation Offloading in Aerial IoV Scenarios

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2024
- 刊物/会议：IEEE Transactions on Network Science and Engineering
- DOI：10.1109/TNSE.2024.3445890
- PDF页数：16
- 原文：`C:\Users\tangning\Desktop\文献\01_时延约束下服务部署与任务卸载的自适应多时间尺度协同\Multi-Time-Scale Markov Decision Process for Joint Service Placement, Network Selection, and Computation Offloading in Aerial IoV Scenarios.pdf`
- 领域分类：**B**；依据：摘要/模型段 vehicle=11, UAV=0
- 主研究方向：P1；次方向：无；依据：scores={'1': 41, '2': 7, '3': 6}; title=Multi-Time-Scale Markov Decision Process for Joint Service Placement, Network Se

## 2. 研究场景与问题

### 摘要窗口

（PDF p3起）

system model is defined
with a multi-time scale approach for the service placement,
network selection, and computation offloading process
with dynamic VTs. Furthermore, a constrained optimiza-
tion problem is formed to minimize a proper cost function,
including also latency and energy terms, by performing
a dynamic service placement, network selection, and of-
floading process.
r MDP Solutions: There is a lack of extensive research
on multi-time scale optimization. Because these problems
are highly complex, a suitable optimization approach has
been devised using the MDPs. The service placement,
network selection, and offloading problems are modeled
as a sequential decision-making process through proper
SHINDE AND TARCHI: MULTI-TIME-SCALE MDP FOR JOINT SERVICE PLACEMENT, NETWORK SELECTION, AND COMPUTA TION OFFLOADING 5367
multidimensional MDP models. A multi-time scale MDP
process is adap

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p3起）

system model is defined
with a multi-time scale approach for the service placement,
network selection, and computation offloading process
with dynamic VTs. Furthermore, a constrained optimiza-
tion problem is formed to minimize a proper cost function,
including also latency and energy terms, by performing
a dynamic service placement, network selection, and of-
floading process.
r MDP Solutions: There is a lack of extensive research
on multi-time scale optimization. Because these problems
are highly complex, a suitable optimization approach has
been devised using the MDPs. The service placement,
network selection, and offloading problems are modeled
as a sequential decision-making process through proper
SHINDE AND TARCHI: MULTI-TIME-SCALE MDP FOR JOINT SERVICE PLACEMENT, NETWORK SELECTION, AND COMPUTA TION OFFLOADING 5367
multidimensional MDP models. A multi-time scale MDP
process is adapted for enabling decision making at different
time scales, and optimal policy is determined through the
deep Q-learning approach.
r Performance Evaluation:The simulation results of the pro-
posed methods are compared with a set of benchmark meth-
ods and their effectiveness is evaluated. Finally, proper
conclusions are drawn on the basis of the findings.
The structure of the paper is as follows. In Section II,w e
present the system model, specifically focusing on the multi-
level scenario and defining the problem as a multi-time-scale
approach. In Section III, we propose a multi-time-scale opti-
mization method, which involves considering three MDPs, while
in Section IV the Deep Q-Learning solution for joint Service
Placement, Network Selection, and Computation Offloading is
described. Section V presents the Numerical Results, which
consider various scenarios where the problem can be app

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p15起）

optimization problem is formed to jointly minimize latency,
energy and service placement costs in different vehicular sce-
narios. Advanced DQN-based solutions are considered to solve
complex MDP in finding optimal policies that reduce overall
cost with improved reliability. The numerical results acquired
through a Python-based simulator show several advantages over
traditional benchmark solutions. With the complex nature of a
considered joint optimization process and the proposed multi-
time scale MDP , to avoid excessive discussions, in this work, we
have resorted to the basic DQN approach. In recent years, several
new advanced forms of DRL algorithms have been proposed
with additional benefits. In the future, our aim is to extend
the proposed framework to accommodate such advanced DRL
solutions with improved efficiency and reduced training costs.

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p1起）

solution allowing for extended
coverage and expanding available resources. However, in such a
dynamic scenario, it is important to perform a proper service
placement based on the users’ demands. Furthermore, with lim-
ited computing and communication resources, proper user-server
assignments and offloading strategies need to be adopted. Consid-
ering their different time scales, a multi-time-scale optimization
process is proposed here to address the joint service placement,
network selection, and computation offloading problem effectively.
With this scope in mind, we propose a multi-time-scale Markov
Decision Process (MDP) based Reinforcement Learning (RL) to
solve this problem and improve the latency and energy perfor-
mance of VEC-enabled VNs. Given the complex nature of the
joint optimization process, an advanced deep Q-learning method is
considered. Comparison with various benchmark methods shows
an overall improvement in latency and energy performance in
different VN scenarios.
Index Terms —V ehicular Edge Computing, Aerial Networks,
Service Placement, Network Selection, Computation Offloading,
Markov Decision Processes, Multi-time Scale Optimization.
I. I NTRODUCTION
W
ITH the evolution of technologies such as the Inter-
net of Things (IoT), Multi-access Edge Computing
(MEC), Network Softwarization, and different communication
modes, modern V ehicular Networks (VNs) are t

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p4起）

simulation results of the pro-
posed methods are compared with a set of benchmark meth-
ods and their effectiveness is evaluated. Finally, proper
conclusions are drawn on the basis of the findings.
The structure of the paper is as follows. In Section II,w e
present the system model, specifically focusing on the multi-
level scenario and defining the problem as a multi-time-scale
approach. In Section III, we propose a multi-time-scale opti-
mization method, which involves considering three MDPs, while
in Section IV the Deep Q-Learning solution for joint Service
Placement, Network Selection, and Computation Offloading is
described. Section V presents the Numerical Results, which
consider various scenarios where the problem can be applied.
Lastly, in Section VI, we provide the conclusions drawn from
the study.
II. S YSTEM MODEL AND PROBLEM FORMULA TION
We focus on an IoV scenario with multiple EC layers and
randomly distributed VTs in the road scenario. We consider a
multi-layer (l =0 ,...,L withL =3 ) joint air-ground network,
composed of HAPs (l =3 ), UA Vs (i.e., LAP nodes) (l =2 ),
RSUs (l =1 ), deployed along the road paths, and randomly
distributed VTs (l =0 )traveling on a road in either direction,
where V = {v1,...,v m,...,v M}, R = {r1,...,r n,...,r N},
U = {u1,...,u p,...,u P}, correspond to the sets denoting M
VTs, N RSUs, and P UA Vs, respectively. Additionally, one
HA

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 部分 |
| 服务更新周期 | 是 |
| 决策时间尺度自适应 | 是 |
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
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=部分；服务更新周期=是；时间尺度自适应=是 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=部分 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=未提及 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：3（PDF页）
- 问题窗口：15（PDF页）
- 方法窗口：1（PDF页）
- 仿真窗口：4（PDF页）
- 结论窗口：4（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（16页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
