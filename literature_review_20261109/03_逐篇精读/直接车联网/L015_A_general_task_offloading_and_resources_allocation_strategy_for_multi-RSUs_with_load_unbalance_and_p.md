# L015 — A general task offloading and resources allocation strategy for multi-RSUs with load unbalance and priority awareness

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2025
- 刊物/会议：Ad Hoc Networks
- DOI：10.1016/j.adhoc.2024.103690
- PDF页数：10
- 原文：`C:\Users\tangning\Desktop\文献\18篇指定论文合集\A general task offloading and resources allocation strategy for multi-RSUs with load unbalance and priority awareness.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=41, UAV=0
- 主研究方向：P1；次方向：无；依据：scores={'1': 40, '2': 8, '3': 5}; title=A general task offloading and resources allocation strategy for multi-RSUs with 

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model and
expatiates on the research problems. In Section 4, we introduce the
optimisation approach in detail. Finally, performance evaluation and
discussion are given in Section 5. This work is concluded in Section 6.
2. Related work
For the different delay requirements of heterogeneous tasks in
MEC, a large number of researchers have explored. Lv et al. [14]
improved the earliest deadline priority algorithm by jointly considering
the deadline and priority of each task to achieve the edge nodes
complete more high priority tasks within the specified time. How-
ever, they did not consider using the resources of neighbouring RSUs
and vehicles. Wang et al. [15] researched cooperative strategies for
Vehicle-to-Infrastructure (V2I) and Vehicle-to-Vehicle (V2V) links to
guarantee heterogeneous delay requirements for different vehicular
services. Specifically, they propose a mean delay m

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model and
expatiates on the research problems. In Section 4, we introduce the
optimisation approach in detail. Finally, performance evaluation and
discussion are given in Section 5. This work is concluded in Section 6.
2. Related work
For the different delay requirements of heterogeneous tasks in
MEC, a large number of researchers have explored. Lv et al. [14]
improved the earliest deadline priority algorithm by jointly considering
the deadline and priority of each task to achieve the edge nodes
complete more high priority tasks within the specified time. How-
ever, they did not consider using the resources of neighbouring RSUs
and vehicles. Wang et al. [15] researched cooperative strategies for
Vehicle-to-Infrastructure (V2I) and Vehicle-to-Vehicle (V2V) links to
guarantee heterogeneous delay requirements for different vehicular
services. Specifically, they propose a mean delay minimisation prob-
lem and a maximum individual delay minimisation in a single Base
Station (BS) scenario. Reinforcement learning is also used to solve the
problem, which aims to reduce the delay of the V2V communication
link while ensuring the delay requirement of the V2I communication
link. Although they improved the task processing delay and energy
consumption while considering the task priority, they only investigated
the task priority, system delay and channel capacity on the BS, but
neglected to investigate the performance of the multi-BS system. Sharif
et al. [16] researched efficient task scheduling and resource allocation
in MEC and proposed a priority-based task scheduling and resource
allocation mechanism. This mechanism can assign different priorities
to different tasks based on the urgency of the tasks according to
the data aggregated by the smart wearable devices, thus achie

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p6起）

formulation of the Tent mapping is:
𝑧𝑘′+1 =
⎧
⎪
⎨
⎪⎩
𝑧𝑘′
𝜛 , 𝑧𝑘′ ∈ (0, 𝜛],
1 −𝑧𝑘′
1 −𝜛 , 𝑧𝑘′ ∈ (𝜛 , 1]
, (16)
where 𝑧𝑘′ is the 𝑘′nd chaotic value in the chaotic domain and 𝜛 is a
constant.
To make the IPSO approach the global optimal solution faster in the
early stage and to avoid blind search, we add a particle that divides
the tasks and resources equally as one of the initial values during
initialisation. At the same time, in order to accelerate the convergence
speed of the IPSO in the early stage, and to avoid the algorithm from
oscillating in the later stage of the solution, the maximum speed of the
particle in the early stage is set larger and made to decrease gradually
with the number of iterations, denoted as:
Vmax = Vmax − Vmin⋅ 𝜇𝜇max, (17)
where Vmax, Vmin are the maximum and minimum values of the particle
velocity that can be taken, 𝜇 is the current iteration and 𝜇max is the
maximum iteration.
Algorithm 2: Improved Particle Swarm Optimisation Algorithm
(IPSO)
Input: ,  t askcp
Output: 𝐷
Maximum iterations 𝜇max, current iterations 𝜇 ← 0, swarm size
𝑆 𝑖𝑧𝑒, 𝜀max1 , 𝜀min1 , 𝜀max2 , 𝜀min2 , 𝜔max, 𝜔min and an array recording
the current fitness and the historical best fitness of each
particle 𝐼 𝑁 𝐹 𝑂𝑆 𝑖𝑧𝑒×2;
Some variables that do not need to be allocated resources are
eliminated, thus forming a new solution set 𝜒 with dimension
𝑑 𝑖𝑚new;
Using Eqs. (16) and (18), the initial coordinates of each point
are obtained according to the chaotic mapping, and a
prediction value is added as the initial position of a certain
particle, which together form the 𝜒;
Utilising Eqs. (17) and (19), calculate Vmax and initialise the
particle velocity 𝜈𝑚;
foreach 𝑚 in 𝑆 𝑖𝑧𝑒 do
The current fitness of each particle is calculated from
Eq. (11);
𝐼 𝑁 𝐹 𝑂[𝑚, 1] ←𝑃 𝑏𝑒𝑠𝑡;
𝐼 𝑁 𝐹 𝑂[𝑚, 0] ←𝑃 𝑏𝑒𝑠

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p5起）

Algorithm design
To solve for the optimal value of the above objective function,
the problem can be divided into two parts. First of all, the one-to-
one bilateral matching model is used to screen out the vehicles that
can be used as relays, and the set of matching pairs of task vehicles
and relay vehicles is obtained  t askcp . After that, the Improved Particle
Swarm Optimisation (IPSO) algorithm is used to find out the vari-
ables such as resources allocation coefficient, task splitting proportion,
and offloading order, as well as to derive an optimal solution of the
problem.
4.1. One-to-one bilateral matching algorithm
In this subsection, in order to facilitate the calculation, here we
first set an initial value for the variables. The computation and com-
munication resources of the RSU are evenly divided, while all tasks
are equally split into four parts. All the task vehicles𝑣𝑖t ask form the set
 t ask,  t ask ⊂ . And in the case where the vehicles satisfy the constraint
𝐶2, the optional idle vehicles 𝑣𝑖idle for each 𝑣𝑖t ask collectively form the
set of candidate relay vehicles  idle,  idle ⊂ .
The set of task vehicles  t ask and the set of optional idle vehicles
 idle can be viewed as two groups of non-intersecting selfish and
rational participants, and the process of obtaining the set of match-
ing pairs  t askcp of task vehicles and relay vehicles can be viewe

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

Simulation results show that SHORAS
improves 24% in terms of the total system delay and effectively reduces the processing delay in the high-load
areas compared to other strategies, while ensuring the delay requirements of high priority tasks.
1. Introduction
With the rapid development of information technology and the
popularity of intelligent applications, the Vehicular Network (VN) has
become the core component of intelligent transportation [1,2]. How-
ever, with the continuous expansion of vehicular application scenarios,
as well as the high density and delay sensitivity of the tasks, the
traditional cloud computing architecture has become insufficient to
meet the real-time and high-reliability requirements of vehicular ap-
plications [3,4]. To address this challenge, Mobile Edge Computing
(MEC) [5] has come into being as an emerging computing paradigm
✩ This work was supported by the National Natural Science Foundation of China (No. 62272063, No. 62473146, No. 62072056 and No. 61902041), the Natural
Science Foundation of Hunan Province, China (No. 2022JJ30617), the Open Project of Xiangjiang Laboratory (No. 23XJ03003), the Researchers Foundation of
Education Bureau of Hunan Province, China(No. 23A0253) and the Researchers Supporting Project Number (RSP2024R509), King Saud University, Riyadh, Saudi
Arabia.
∗ Corresponding author.
E-mail addresses: caodun@csust.edu.cn (D. Ca

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
| 迁移传输 | 未提及 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 未提及 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=未提及；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=未提及 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：6（PDF页）
- 方法窗口：5（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：5（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（10页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
