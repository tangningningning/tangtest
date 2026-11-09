# L079 — Joint Service Caching, Communication and Computing Resource Allocation in Collaborative MEC Systems: A DRL-Based Two-Timescale Approach

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2024
- 刊物/会议：IEEE Transactions on Wireless Communications
- DOI：10.1109/TWC.2024.3430486
- PDF页数：14
- 原文：`C:\Users\tangning\Desktop\文献\01_时延约束下服务部署与任务卸载的自适应多时间尺度协同\Joint Service Caching, Communication and Computing Resource Allocation in Collaborative MEC Systems- A DRL-Based Two-Timescale Approach.pdf`
- 领域分类：**C**；依据：摘要/模型段 MEC=16, vehicle=0, UAV=0
- 主研究方向：P1；次方向：无；依据：scores={'1': 100, '2': 2, '3': 14}; title=Joint Service Caching, Communication and Computing Resource Allocation in Collab

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model and problem formulation. Section IV elabo-
rates on the DRL-based two timescale optimization algorithm.
Section V presents simulation results that evaluate the conver-
gence and performance of the proposed algorithms. Finally,
Section VI presents the conclusions drawn from the study.
II. R ELATED WORK
This section provides a review of recent studies on resource
allocation in MEC systems, categorized based on joint allo-
cation of communication and computing resources, joint
allocation of service caching and computing resources, and
joint allocation of service caching, communication and com-
puting resources.
A. Joint Communication and Computing Resources
Allocation
For the allocation of communication and computing
resources in MEC networks, most researcher focus on reducing
the task processing delay and/or energy consumption through
aspects such as task offloading (partially

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model and problem formulation. Section IV elabo-
rates on the DRL-based two timescale optimization algorithm.
Section V presents simulation results that evaluate the conver-
gence and performance of the proposed algorithms. Finally,
Section VI presents the conclusions drawn from the study.
II. R ELATED WORK
This section provides a review of recent studies on resource
allocation in MEC systems, categorized based on joint allo-
cation of communication and computing resources, joint
allocation of service caching and computing resources, and
joint allocation of service caching, communication and com-
puting resources.
A. Joint Communication and Computing Resources
Allocation
For the allocation of communication and computing
resources in MEC networks, most researcher focus on reducing
the task processing delay and/or energy consumption through
aspects such as task offloading (partially), computing resource
allocation, power control, and channel selection.
Tan et al. [30] proposed a two-level alternation method to
minimize total energy consumption in multi-user collaborative
MEC networks. The method uses a heuristic algorithm in
the upper level for collaboration and offloading decisions
1https://github.com/liuqiansdu/Two-timescale-DGL-DDPG-.git
Authorized licensed use limited to: Univ of Science and Tech Beijing. Downloaded on June 09,2026 at 10:45:34 UTC from IEEE Xplore. Restrictions apply. 
LIU et al.: JOINT SERVICE CACHING, COMMUNICATION AND COMPUTING RESOURCE ALLOCATION 15495
and a DRL-based algorithm in the lower level for power,
subcarrier, and computing resource allocation. Kai et al. [18]
proposed a pipeline-based offloading scheme to minimize the
overall latency for all users in a cloud-edge-end collaborative
MEC network. They utilized the classic successive

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p2起）

problem formulation. Section IV elabo-
rates on the DRL-based two timescale optimization algorithm.
Section V presents simulation results that evaluate the conver-
gence and performance of the proposed algorithms. Finally,
Section VI presents the conclusions drawn from the study.
II. R ELATED WORK
This section provides a review of recent studies on resource
allocation in MEC systems, categorized based on joint allo-
cation of communication and computing resources, joint
allocation of service caching and computing resources, and
joint allocation of service caching, communication and com-
puting resources.
A. Joint Communication and Computing Resources
Allocation
For the allocation of communication and computing
resources in MEC networks, most researcher focus on reducing
the task processing delay and/or energy consumption through
aspects such as task offloading (partially), computing resource
allocation, power control, and channel selection.
Tan et al. [30] proposed a two-level alternation method to
minimize total energy consumption in multi-user collaborative
MEC networks. The method uses a heuristic algorithm in
the upper level for collaboration and offloading decisions
1https://github.com/liuqiansdu/Two-timescale-DGL-DDPG-.git
Authorized licensed use limited to: Univ of Science and Tech Beijing. Downloaded on June 09,2026 at 10:45:34 UTC from IEEE Xplore. Restrictions apply. 
LIU et al.: JOINT SERVICE CACHING, COMMUNICATION AND COMPUTING RESOURCE ALLOCATION 15495
and a DRL-based algorithm in the lower level for power,
subcarrier, and computing resource allocation. Kai et al. [18]
proposed a pipeline-based offloading scheme to minimize the
overall latency for all users in a cloud-edge-end collaborative
MEC network. They utilized the classic successive convex
approximat

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p1起）

proposed algorithm outperforms the
baseline algorithms in terms of the average QoS and the cache
switching costs.
Index Terms— Multi-dimensional resources allocation, collabo-
rative offloading, service caching, two-timescale, long short-term
memory (LSTM) network, deep deterministic policy gradient.
I. I NTRODUCTION
T
HE evolution of the fifth generation mobile commu-
nication (5G), artificial intelligence (AI) and Internet
Manuscript received 13 July 2023; revised 16 April 2024 and 12 July
2024; accepted 14 July 2024. Date of publication 25 July 2024; date of
current version 11 October 2024. This work was supported in part by the
Joint Funds of the National Natural Science Foundation of China (NSFC)
under Grant U22A2003 and Grant U23A20277 and in part by NSFC under
Grant 62271288. The associate editor coordinating the review of this article
and approving it for publication was X. Cao. (Corresponding author: Haixia
Zhang.)
Qianqian Liu, Haixia Zhang, and Xin Zhang are with Shandong
Key Laboratory of Intelligent Communication and Sensing-Computing
Integration, Jinan 250061, China, and also with the School of
Control Science and Engineering, Shandong University, Jinan 250061,
China (e-mail: liuqiansdu@mail.sdu.edu.cn; haixia.zhang@sdu.edu.cn;
xin.zhang0892@mail.sdu.edu.cn).
Dongfeng Yuan is with Shandong Key Laboratory of Intelligent Commu-
nication and Sensing-Computing Integra

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

Simulation
results demonstrate that the proposed algorithm outperforms the
baseline algorithms in terms of the average QoS and the cache
switching costs.
Index Terms— Multi-dimensional resources allocation, collabo-
rative offloading, service caching, two-timescale, long short-term
memory (LSTM) network, deep deterministic policy gradient.
I. I NTRODUCTION
T
HE evolution of the fifth generation mobile commu-
nication (5G), artificial intelligence (AI) and Internet
Manuscript received 13 July 2023; revised 16 April 2024 and 12 July
2024; accepted 14 July 2024. Date of publication 25 July 2024; date of
current version 11 October 2024. This work was supported in part by the
Joint Funds of the National Natural Science Foundation of China (NSFC)
under Grant U22A2003 and Grant U23A20277 and in part by NSFC under
Grant 62271288. The associate editor coordinating the review of this article
and approving it for publication was X. Cao. (Corresponding author: Haixia
Zhang.)
Qianqian Liu, Haixia Zhang, and Xin Zhang are with Shandong
Key Laboratory of Intelligent Communication and Sensing-Computing
Integration, Jinan 250061, China, and also with the School of
Control Science and Engineering, Shandong University, Jinan 250061,
China (e-mail: liuqiansdu@mail.sdu.edu.cn; haixia.zhang@sdu.edu.cn;
xin.zhang0892@mail.sdu.edu.cn).
Dongfeng Yuan is with Shandong Key Laboratory of Intelligent Commu

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 部分 |
| 服务更新周期 | 是 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 是 |
| 位置预测误差 | 未提及 |
| 服务迁移 | 未提及 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 是 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 是 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=部分；服务更新周期=是；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=是 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=是 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：2（PDF页）
- 方法窗口：1（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：2（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（14页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
