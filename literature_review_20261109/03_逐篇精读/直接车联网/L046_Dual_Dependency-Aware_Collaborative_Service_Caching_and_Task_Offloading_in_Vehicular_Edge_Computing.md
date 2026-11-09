# L046 — Dual Dependency-Aware Collaborative Service Caching and Task Offloading in Vehicular Edge Computing

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2025
- 刊物/会议：IEEE Transactions on Mobile Computing
- DOI：10.1109/TMC.2025.3573379
- PDF页数：15
- 原文：`C:\Users\tangning\Desktop\文献\车联网频繁切换-负载均衡-端到端时延研究\四篇重点论文\Dual_Dependency-Aware_Collaborative_Service_Caching_and_Task_Offloading_in_Vehicular_Edge_Computing.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=16, UAV=0
- 主研究方向：P1；次方向：无；依据：scores={'1': 71, '2': 2, '3': 4}; title=Dual Dependency-Aware Collaborative Service Caching and Task Offloading in Vehic

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model
and problem formulation. Section IV describes the main design
of the algorithm. SectionV presents the experimental simulation
results. Finally, Section VI concludes the paper.
II. R ELA TEDWORK
We review the literature on task offloading and service
caching, with a focus on their limitations in handling dependent
tasks in V ehicular Edge Computing (VEC) scenarios. More
details are provided in the supplementary file for the related
work.
A. Task Computing Offloading
Offloading of dependent tasks in Mobile Edge Computing
(MEC) has received growing attention due to the execution
constraints imposed by task dependencies. These constraints
often result in cascading delays that negatively affect overall
system performance. Existing studies, including those based
on reinforcement learning [17], priority scheduling [18], and
Authorized licensed use limited to: Indian Institute of Pe

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model
and problem formulation. Section IV describes the main design
of the algorithm. SectionV presents the experimental simulation
results. Finally, Section VI concludes the paper.
II. R ELA TEDWORK
We review the literature on task offloading and service
caching, with a focus on their limitations in handling dependent
tasks in V ehicular Edge Computing (VEC) scenarios. More
details are provided in the supplementary file for the related
work.
A. Task Computing Offloading
Offloading of dependent tasks in Mobile Edge Computing
(MEC) has received growing attention due to the execution
constraints imposed by task dependencies. These constraints
often result in cascading delays that negatively affect overall
system performance. Existing studies, including those based
on reinforcement learning [17], priority scheduling [18], and
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 30,2026 at 07:10:09 UTC from IEEE Xplore. Restrictions apply. 
ZHAO et al.: DUAL DEPENDENCY -AW ARED COLLABORA TIVE SERVICE CACHING 10965
heuristic algorithms [19], [20], [21], primarily optimize individ-
ual task execution in terms of delay or energy consumption.
However, they generally lack explicit modeling of inter-task
dependencies and are therefore ineffective in mitigating delay
propagation across task chains. Furthermore, many of these
methods adopt static scheduling mechanisms or oversimplified
assumptions, which reduce their adaptability in dynamic VEC
scenarios.
B. Joint Optimization of Service Caching and Task Offloading
In VEC environments, tasks frequently require specific ser-
vice components for execution. Due to limited caching capacity
at edge nodes and onboard units, it is often infeasible to store
all services in advance. Cons

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p2起）

problem formulation. Section IV describes the main design
of the algorithm. SectionV presents the experimental simulation
results. Finally, Section VI concludes the paper.
II. R ELA TEDWORK
We review the literature on task offloading and service
caching, with a focus on their limitations in handling dependent
tasks in V ehicular Edge Computing (VEC) scenarios. More
details are provided in the supplementary file for the related
work.
A. Task Computing Offloading
Offloading of dependent tasks in Mobile Edge Computing
(MEC) has received growing attention due to the execution
constraints imposed by task dependencies. These constraints
often result in cascading delays that negatively affect overall
system performance. Existing studies, including those based
on reinforcement learning [17], priority scheduling [18], and
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 30,2026 at 07:10:09 UTC from IEEE Xplore. Restrictions apply. 
ZHAO et al.: DUAL DEPENDENCY -AW ARED COLLABORA TIVE SERVICE CACHING 10965
heuristic algorithms [19], [20], [21], primarily optimize individ-
ual task execution in terms of delay or energy consumption.
However, they generally lack explicit modeling of inter-task
dependencies and are therefore ineffective in mitigating delay
propagation across task chains. Furthermore, many of these
methods adopt static scheduling mechanisms or oversimplified
assumptions, which reduce their adaptability in dynamic VEC
scenarios.
B. Joint Optimization of Service Caching and Task Offloading
In VEC environments, tasks frequently require specific ser-
vice components for execution. Due to limited caching capacity
at edge nodes and onboard units, it is often infeasible to store
all services in advance. Consequently, service

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p10起）

proposed algorithm:
1) Average delay: Quantifies the average time from the
launch of the application to completion, including trans-
mission, computation, and queuing delays. As shown in
Part I of (24).
2) Average energy consumption: Characterizes overall en-
ergy utilization by counting the total energy consumed by
each device during task execution and normalizing it with
the number of completed tasks. As shown in Part II of
(24).
3) Service caching hit rate:Measures the effective utilization
of local and edge caching, which is the proportion of suc-
cessful caching responses to the total number of requests.
4) Application success rate:Assesses the proportion of appli-
cations completed within the specified period of the total
number of applications.
C. Ablation Experiment
In this section, we conducted an ablation experiment to eval-
uate the impact and contribution of different modules in the
COHCTO strategy on system performance by removing or mod-
ifying the core mechanisms in the COHCTO strategy proposed
in this paper. Specifically, the COHCTO-POP strategy is used as
a comparison scheme to evaluate the effectiveness of the service
demand criticality prediction model, the COHCTO-NTP strat-
egy is used as a comparison scheme to measure the effectiveness
of the heterogeneous-aware topology priority scheduling algo-
rithm, and the COHCTO-NSC strategy is used as a comparison
sc

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

Simulation results show that COHCTO achieves
multi-objective optimization across metrics such as delay, energy
consumption, caching hit rate, and application success rate under
conditions different from those of other algorithms.
Index Terms—Task dependency, service caching, task offloading,
deep reinforcement learning.
Received 10 January 2025; revised 9 April 2025; accepted 21 May 2025.
Date of publication 23 May 2025; date of current version 3 September 2025. This
work was supported in part by the National Natural Science Foundation of China
under Grant 62372310 and Grant W2421018, in part by the Liaoning Province
Applied Basic Research Program under Grant 2023JH2/101300194, in part
by the LiaoNing Revitalization Talents Program under Grant XL YC2203151,
and in part by the Royal Society-International Exchanges 2021 Cost Share
(NSFC) under Grant IEC\NSFC\211034. LLM was used to proofread this paper.
Recommended for acceptance by Dr. S. Wang.(Corresponding authors: Lu Sun;
Ammar Hawbani.)
Liang Zhao, Lu Sun, and Ammar Hawbani are with the School of Computer
Science, Shenyang Aerospace University, Shenyang 110136, China (e-mail:
lzhao@sau.edu.cn; sunlu3@stu.sau.edu.cn; anmande@ustc.edu.cn).
Zhi Liu is with the Department of Computer and Network Engineer-
ing, University of Electro Communications, Tokyo 182-8585, Japan (e-mail:
liu@ieee.org).
Xiongyan Tang and Lexi Xu are with t

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
| 多业务QoS | 部分 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 未提及 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=未提及；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=部分；业务流量非平稳=未提及；信道动态=未提及 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：2（PDF页）
- 方法窗口：10（PDF页）
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
