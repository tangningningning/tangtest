# L119 — Task migration with deadlines using machine learning-based dwell time prediction in vehicular micro clouds

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2025
- 刊物/会议：待核
- DOI：10.1016/j.hcc.2025.100314
- PDF页数：12
- 原文：`C:\Users\tangning\Desktop\文献\02_移动性感知的服务迁移与传输资源协同\Task migration with deadlines using machine learning-based dwell time prediction in vehicular micro clouds.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=20, UAV=0
- 主研究方向：P1；次方向：2；依据：scores={'1': 60, '2': 41, '3': 7}; title=Task migration with deadlines using machine learning-based dwell time prediction

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model,
cost function, and utility function in Section3. In Section4, we
introduce our proposed task migration algorithm. The ML-based
dwell time prediction model is presented in Section4.4. We dis-
cuss selected results from an extensive performance evaluation
in Section5. Finally, we present concluding remarks in Section6.
2. Related work
In the following, we review related work for our main con-
tributions: task offloading and dwell time prediction. To ease
working with acronyms used, we summarize these inTable 1.
2.1. Task offloading
Previous work on task offloading has focused on various op-
timization strategies and optimization goals, including reducing
energy consumption [12], minimizing latency [13], and monetary
cost [14], and increasing task success rates [15]. In the follow-
ing, we categorize these works based on their own optimization
approaches: the combination of ta

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model,
cost function, and utility function in Section3. In Section4, we
introduce our proposed task migration algorithm. The ML-based
dwell time prediction model is presented in Section4.4. We dis-
cuss selected results from an extensive performance evaluation
in Section5. Finally, we present concluding remarks in Section6.
2. Related work
In the following, we review related work for our main con-
tributions: task offloading and dwell time prediction. To ease
working with acronyms used, we summarize these inTable 1.
2.1. Task offloading
Previous work on task offloading has focused on various op-
timization strategies and optimization goals, including reducing
energy consumption [12], minimizing latency [13], and monetary
cost [14], and increasing task success rates [15]. In the follow-
ing, we categorize these works based on their own optimization
approaches: the combination of task offloading and resource al-
location, learning-based, game-theoretic, or mobility-aware task
offloading.
2.1.1. Task offloading and resource allocation
Mao et al. [15] proposed a joint optimization strategy consid-
ering the current system state, including transmitting energy and
CPU resources, to reduce execution costs and task failures. Hos-
sain et al. [13] introduced small-cell MEC to enhance offloading
decisions and reduce task execution latency. Tran and Pompili
[12] optimized a weighted sum of task completion time and
energy consumption using convex and quasi-convex optimization
techniques, while Zhao et al. [16] applied a distributed task-
offloading algorithm for vehicles in MEC to minimize average de-
lay. Although these approaches demonstrated improved resource
efficiency, they primarily used simplified network environments
and did not account for complex urban traffic dyna

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p4起）

optimization problem can be formulated in a simplified
form by minimizing a time cost function without accounting
for task failure, or more comprehensively by maximizing a util-
ity function. The time cost function aims to minimize the total
task completion time. The utility function is designed to balance
multiple factors by combining the average task completion time
with two weighted metrics for task failure: the proportion of
tasks that miss their deadlines and the proportion of tasks where
the generating or processing device exits the VMC before task
completion.
We assume a Tsk(k) is generated and migrated at time t to
Veh(p(k)), where there have been already q −1 tasks in the
waiting queue and Tsk(k) is the qthtask in the queue.
The expression Tsum(k, t, p(k)) represents the completion time
of task Tsk(k) as
Tsum(k, t, p(k)) =Tc(k, p(k)) + Tp(k, p(k)) + Tw(k, t, p(k))
=I(p(k) ̸ = g(k)) S(k)
R
+
MI(k)
MIPS(p(k))
+
(1 − αp(k),t)MI(k0)
MIPS(p(k))
+
q−1∑
n=1
Tp(kn)
It consists of communication time Tc(k, p(k)), processing time
Tp(k, p(k)), and the waiting time Tw(k, t, p(k)). The communi-
cation cost is positive only if the task is not executed locally,
i.e, p(k) ̸ = g(k). Therefore, we have
Tsum(k, t, p(k)) =I(p(k) ̸ = g(k)) S(k)
R
+
q∑
n=1
MI(kn)
MIPS(p(k))
+(1 − αp(k),t)MI(k0)
MIPS(p(k))
(6)
The optimization problem can now be formulated as
minimize
k
Tsum(k, t, p(k))
subject to t ≥0, k, p(k), q ≥1
(7)
The sum of completion time for Ntasks over a certain duration
δis
Tsum(N, δ) =
k=N,t=δ∑
k=1,t=0
(k, t, p(k)) (8)
Before introducing the utility function, the failure rate (caused
by unpredicted leaving vehicles) Fm(N, δ), and the failure rate
(caused by missing task deadlines) Fd(N, δ) over a certain time
interval δwith Ntasks generated in VMC are respectively defined

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p9起）

proposed algorithm and
for the theoretical optimum using perfect dwell time knowledge
yield nearly identical performance, demonstrating the superiority
of our solution. Again, the Johnson SU based task mitigation
results in slightly increased task completion times (Fig. 5(a))
and lower saved time ratios (Fig. 5(b)), but the differences are
not significant. However, when evaluating the task failure rate,
Johnson SU leads to a pronounced failure rate, whereas our
prediction model reduced this by about 36%.
5.5. Nagoya scenario
Also for the Nagoya scenario, we selected the morning rush
hour data for discussion in this paper. During the non-peak pe-
riods in the evening at 8 pm, the algorithm operates similarly to
the peak hour. During midnight, in the city and suburban areas
of Nagoya, only one vehicle is present. As a result, the migration
strategy requirements are not effective. Thus, again, we focus on
the morning rush hour. The results of our simulations for average
task completion time and saved time ratio as well as the task
migration failure rate are shown inFig. 6.
Task completion times (cf.Fig. 6(a)) across the three migration
strategies are significantly reduced compared to no migration
(cf.Fig. 6(b)). Our proposed model clearly outperforms the John-
son SU approach w.r.t. the task failure rate (mobility reason, cf.
Fig. 6(d)): Our proposed solution exhibits a failure ra

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p2起）

simulations using realistic urban traffic data from
the mid-scale European city Luxembourg [10], and the Japanese
city Nagoya [11]. The task migration performance is evaluated
using four metrics and a utility function that combines both task
completion time and task failure rate, providing a view on prac-
ticability of the approach. It takes two types of task failure cases
into account: (i) cars leaving the VMC before task completion, and
(ii) task completion time exceeding the deadline.
This paper extends our previous work [9]. The main contribu-
tions beyond the current state of the art can be summarized as
follows:
•We developed a novel task migration algorithm for use
in vehicular micro clouds. The algorithm is based on a
greedy approach, sequentially offloading tasks to the neigh-
bor where it can be completed the earliest. In addition,
deadlines are considered as a constraint. This matches the
well-known EDF approach, now in a dynamic and mobile
environment.
•In order to estimate the time a vehicle will be part of the
VMC (the dwell time), we propose a new ML-based dwell
time prediction model. In particular, we use a random forest
algorithm that significantly outperforms previous heuristics.
•In a comprehensive simulation study, we thoroughly com-
pare our task allocation algorithm and the dwell time pre-
diction to other published heuristics and a theoretical opti-
mum.

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
| 多业务QoS | 未提及 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 未提及 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=未提及；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=未提及；业务流量非平稳=未提及；信道动态=未提及 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：4（PDF页）
- 方法窗口：9（PDF页）
- 仿真窗口：2（PDF页）
- 结论窗口：11（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（12页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
