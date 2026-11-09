# L089 — Mobility-Aware Assisted Deep Reinforcement Learning for Collaborative Task Migration and Resource Allocation in Vehicular Edge Computing

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2026
- 刊物/会议：IEEE Transactions on Vehicular Technology
- DOI：10.1109/TVT.2026.3660321
- PDF页数：14
- 原文：`C:\Users\tangning\Desktop\文献\18篇指定论文合集\Mobility-Aware Assisted Deep Reinforcement Learning for Collaborative Task Migration and Resource Allocation in Vehicular Edge Computing.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=49, UAV=1
- 主研究方向：P2；次方向：无；依据：scores={'1': 14, '2': 73, '3': 9}; title=Mobility-Aware Assisted Deep Reinforcement Learning for Collaborative Task Migra

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model and
problem formulation. In Section IV, we present the MA TM
approach. Section V details the experimental setup and analyzes
the results. Finally, Section VI concludes the paper.
II. R ELA TEDWORK
A. Task Migration
Task migration serves as an effective approach for work-
load balancing and improving task success rates in VEC sce-
narios [8], [20], [21], [22], [23], [24]. Several studies have
utilized traditional optimization techniques for task migration.
P. D a i e t a l .[20] proposed migrating tasks from edge servers
to the cloud to alleviate edge-side congestion, effectively miti-
gating overload issues. H. Wang et al. [21] proposed a task mi-
gration strategy based on game-theoretic principles, effectively
minimizing computational overhead while improving resource
utilization and task success rates. However, their dependence on
traditional optimization methods restricts

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model and
problem formulation. In Section IV, we present the MA TM
approach. Section V details the experimental setup and analyzes
the results. Finally, Section VI concludes the paper.
II. R ELA TEDWORK
A. Task Migration
Task migration serves as an effective approach for work-
load balancing and improving task success rates in VEC sce-
narios [8], [20], [21], [22], [23], [24]. Several studies have
utilized traditional optimization techniques for task migration.
P. D a i e t a l .[20] proposed migrating tasks from edge servers
to the cloud to alleviate edge-side congestion, effectively miti-
gating overload issues. H. Wang et al. [21] proposed a task mi-
gration strategy based on game-theoretic principles, effectively
minimizing computational overhead while improving resource
utilization and task success rates. However, their dependence on
traditional optimization methods restricts adaptability to highly
dynamic VEC scenarios.
Some studies have employed DRL-based intelligent decision-
making to perform task migration, demonstrating the effec-
tiveness of edge intelligence in alleviating VEC challenges.
Authorized licensed use limited to: Goa University. Downloaded on October 26,2026 at 09:14:31 UTC from IEEE Xplore. Restrictions apply. 
YIN et al.: MOBILITY -AW ARE ASSISTED DRL FOR COLLABORA TIVE TASK MIGRA TION AND RESOURCE ALLOCA TION IN VEC 14683
J. Li et al. [22] studied task remigration to improve system
flexibility, but assumed static users and overlooked the impact of
vehicle mobility. W. Fan et al. [8] proposed a twin delayed deep
deterministic policy gradient (TD3)-based RSU cooperation
approach that effectively facilitates collaboration between RSUs
and accounts for vehicle mobility, but their method only applies
to vehicles entering the next RSU.
In add

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p2起）

problem formulation. In Section IV, we present the MA TM
approach. Section V details the experimental setup and analyzes
the results. Finally, Section VI concludes the paper.
II. R ELA TEDWORK
A. Task Migration
Task migration serves as an effective approach for work-
load balancing and improving task success rates in VEC sce-
narios [8], [20], [21], [22], [23], [24]. Several studies have
utilized traditional optimization techniques for task migration.
P. D a i e t a l .[20] proposed migrating tasks from edge servers
to the cloud to alleviate edge-side congestion, effectively miti-
gating overload issues. H. Wang et al. [21] proposed a task mi-
gration strategy based on game-theoretic principles, effectively
minimizing computational overhead while improving resource
utilization and task success rates. However, their dependence on
traditional optimization methods restricts adaptability to highly
dynamic VEC scenarios.
Some studies have employed DRL-based intelligent decision-
making to perform task migration, demonstrating the effec-
tiveness of edge intelligence in alleviating VEC challenges.
Authorized licensed use limited to: Goa University. Downloaded on October 26,2026 at 09:14:31 UTC from IEEE Xplore. Restrictions apply. 
YIN et al.: MOBILITY -AW ARE ASSISTED DRL FOR COLLABORA TIVE TASK MIGRA TION AND RESOURCE ALLOCA TION IN VEC 14683
J. Li et al. [22] studied task remigration to improve system
flexibility, but assumed static users and overlooked the impact of
vehicle mobility. W. Fan et al. [8] proposed a twin delayed deep
deterministic policy gradient (TD3)-based RSU cooperation
approach that effectively facilitates collaboration between RSUs
and accounts for vehicle mobility, but their method only applies
to vehicles entering the next RSU.
In addition, a major fo

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p1起）

solution [4], [5], [6], which allows
vehicles to offload tasks to nearby roadside units (RSUs). Unlike
traditional approaches relying on remote cloud computing, VEC
significantly reduces task transmission distance and lowers pro-
cessing delay. VEC mitigates network disruptions and enhances
reliability [7].
V ehicle mobility is a key feature that differentiates VEC from
conventional edge computing paradigms [8]. As a result, a vehi-
cle may leave the coverage area of its associated RSUs before the
offloaded task is completed and the results are fed back, which
may lead to connection interruptions and task execution failures.
In addition, due to the static deployment of RSUs and the highly
uneven spatiotemporal distribution of vehicles, significant work-
load imbalance may arise among edge computing resources [9].
For example, during peak hours in commercial districts, a large
number of vehicles may simultaneously offload tasks to the same
RSU, causing severe communication congestion and potential
overload, which significantly reduces the task success rate.
Conversely, RSUs located on side streets or in areas with sparse
traffic may remain underutilized due to the limited number of
0018-9545 © 2026 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and similar technologi es.
Personal use is permitted, but republication/r

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

experiments on real-world vehicle trajectory datasets from
Received 23 September 2025; revised 2 December 2025; accepted 25 January
2026. Date of publication 3 February 2026; date of current version 20 July 2026.
This work was supported in part by the National Natural Science Foundation of
China under Grant 62072321, in part by the Science and Technology Program
of Jiangsu Province under Grant BZ2024062, in part by the Natural Science
Foundation of the Jiangsu Higher Education Institutions of China under Grant
22KJA520007, in part by the Suzhou Planning Project of Science and Technol-
ogy under Grant 2023ss03, Grant SYG2025129, and Grant SNG2025010, and in
part by the Key Laboratory of General Artificial Intelligence and Large Models
in Provincial Universities, Soochow University. The review of this article was
coordinated by Dr. Bartolomeo Montrucchio. (Corresponding authors: Peng
Hou; Jin Wang.)
Yilin Yin, Peng Hou, and Jin Wang are with the School of Fu-
ture Science and Engineering, Key Laboratory of General Artificial In-
telligence and Large Models in Provincial Universities, Soochow Uni-
versity, Suzhou 215222, China (e-mail: 20245258017@stu.suda.edu.cn;
penghou@suda.edu.cn; wjin1985@suda.edu.cn).
Shihong Cui is with the Marketing Department, Tianyijiaotong Technology
Company Ltd., Suzhou 215000, China (e-mail: cuishihong202406@163.com).
Jin Huang is with the School of V

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 部分 |
| 服务更新周期 | 部分 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 是 |
| 位置预测误差 | 未提及 |
| 服务迁移 | 未提及 |
| 活动任务连续执行 | 部分 |
| 迁移传输 | 未提及 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 部分 |
| 信道动态变化 | 是 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=部分；服务更新周期=部分；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=部分；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=部分；信道动态=是 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：2（PDF页）
- 方法窗口：1（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：None（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（14页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
