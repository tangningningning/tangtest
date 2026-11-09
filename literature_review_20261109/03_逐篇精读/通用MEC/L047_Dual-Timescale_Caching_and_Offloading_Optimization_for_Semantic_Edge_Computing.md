# L047 — Dual-Timescale Caching and Offloading Optimization for Semantic Edge Computing

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2026
- 刊物/会议：IEEE Transactions on Vehicular Technology
- DOI：10.1109/TVT.2025.3633960
- PDF页数：6
- 原文：`C:\Users\tangning\Desktop\文献\01_时延约束下服务部署与任务卸载的自适应多时间尺度协同\Dual-Timescale Caching and Offloading Optimization for Semantic Edge Computing.pdf`
- 领域分类：**C**；依据：摘要/模型段 MEC=7, vehicle=4, UAV=0
- 主研究方向：P1；次方向：无；依据：scores={'1': 13, '2': 0, '3': 5}; title=Dual-Timescale Caching and Offloading Optimization for Semantic Edge Computing

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system models. Section III provides the problem formulation. The
proposed BPSO-MAPPO algorithm is elaborated on in Section IV.
Section V presents the simulation results, followed by conclusions in
Section VI.
II. S YSTEM MODELS
Consider a CSEC system with a set I := {1, 2,...,I } of devices,
where DeepSC [8] is applied as the semantic communication model
and deployed at each device; see Fig. 1. The devices execute different
types of tasks. At time slot t,t h e i-th device generates a task with
size of uti (in sentences/time slot) to be executed. The process of the
CSEC system includes the following steps: 1) The semantic encoder
extracts the semantics of different tasks based on DeepSC. 2) The
devices offload a portion of tasks to the ES, while the remaining
semantics are executed on local devices. 3) A limited number of
models for executing different tasks are cached in the edge buffer.

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system models. Section III provides the problem formulation. The
proposed BPSO-MAPPO algorithm is elaborated on in Section IV.
Section V presents the simulation results, followed by conclusions in
Section VI.
II. S YSTEM MODELS
Consider a CSEC system with a set I := {1, 2,...,I } of devices,
where DeepSC [8] is applied as the semantic communication model
and deployed at each device; see Fig. 1. The devices execute different
types of tasks. At time slot t,t h e i-th device generates a task with
size of uti (in sentences/time slot) to be executed. The process of the
CSEC system includes the following steps: 1) The semantic encoder
extracts the semantics of different tasks based on DeepSC. 2) The
devices offload a portion of tasks to the ES, while the remaining
semantics are executed on local devices. 3) A limited number of
models for executing different tasks are cached in the edge buffer.
If the type of the offloaded task matches the cached model, the task
is executed on the ES; otherwise, the task is uploaded to the CS that
stores all types of models for execution. 4) The results of the tasks are
downloaded from the ES or CS to local devices. Letli = li,enco +li,deco
(in FLOPS) be the average hardware requirement per sentence for the
task of device i,w h e r eli,enco and li,deco are the average number of
floating-point operations required per sentence for semantic extraction
to understand the task, and for semantic computation to get the result,
respectively [9].
This CSEC framework is designed for a specific class of applications
where raw data are highly redundant, computationally intensive to
process, but their semantic meanings can be extracted into a much more
compact representation. A typical real-world scenario is real-time col-
laborative perception in autonomou

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p2起）

problem formulation. The
proposed BPSO-MAPPO algorithm is elaborated on in Section IV.
Section V presents the simulation results, followed by conclusions in
Section VI.
II. S YSTEM MODELS
Consider a CSEC system with a set I := {1, 2,...,I } of devices,
where DeepSC [8] is applied as the semantic communication model
and deployed at each device; see Fig. 1. The devices execute different
types of tasks. At time slot t,t h e i-th device generates a task with
size of uti (in sentences/time slot) to be executed. The process of the
CSEC system includes the following steps: 1) The semantic encoder
extracts the semantics of different tasks based on DeepSC. 2) The
devices offload a portion of tasks to the ES, while the remaining
semantics are executed on local devices. 3) A limited number of
models for executing different tasks are cached in the edge buffer.
If the type of the offloaded task matches the cached model, the task
is executed on the ES; otherwise, the task is uploaded to the CS that
stores all types of models for execution. 4) The results of the tasks are
downloaded from the ES or CS to local devices. Letli = li,enco +li,deco
(in FLOPS) be the average hardware requirement per sentence for the
task of device i,w h e r eli,enco and li,deco are the average number of
floating-point operations required per sentence for semantic extraction
to understand the task, and for semantic computation to get the result,
respectively [9].
This CSEC framework is designed for a specific class of applications
where raw data are highly redundant, computationally intensive to
process, but their semantic meanings can be extracted into a much more
compact representation. A typical real-world scenario is real-time col-
laborative perception in autonomous-vehicle platoons. A vehicle would
tran

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p1起）

proposed algorithm reduces the system cost by 33.8% compared
to the benchmarks, and can adapt to both latency-sensitive and energy-
sensitive tasks by fine-tuning the weighting coefficient.
Index Terms —Semantic communications, edge computing, service
caching, resource allocation, deep reinforcement learning.
I. I NTRODUCTION
As a decentralized computing architecture, edge computing effec-
tively alleviates computational burden on local devices by offloading
tasks to nearby edge servers (ESs) [1]. However, traditional edge
computing systems face significant challenges, including high latency
and excessive energy consumption, particularly when processing large
volumes of data. To address these issues, semantic communication
has been integrated into edge computing systems [2], [3]. In seman-
tic edge computing (SEC) systems, semantic communication extracts
and transmits only the essential semantic information, reducing data
Received 24 April 2025; revised 20 September 2025; accepted 10 Novem-
ber 2025. Date of publication 17 November 2025; date of current version
24 June 2026. This work was supported in part by the National Key R&D
Program of China under Grant 2022YFB2902002, Grant2022YFB2902303,
and Grant2023YFB2904300 and in part by Shanghai Rising-Star Program. The
review of this article was coordinated by Prof. Kyunghan Lee. (Corresponding
author: Shunqing Zhang.)
Xiaojing Ch

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

Simulations show
that the proposed algorithm reduces the system cost by 33.8% compared
to the benchmarks, and can adapt to both latency-sensitive and energy-
sensitive tasks by fine-tuning the weighting coefficient.
Index Terms —Semantic communications, edge computing, service
caching, resource allocation, deep reinforcement learning.
I. I NTRODUCTION
As a decentralized computing architecture, edge computing effec-
tively alleviates computational burden on local devices by offloading
tasks to nearby edge servers (ESs) [1]. However, traditional edge
computing systems face significant challenges, including high latency
and excessive energy consumption, particularly when processing large
volumes of data. To address these issues, semantic communication
has been integrated into edge computing systems [2], [3]. In seman-
tic edge computing (SEC) systems, semantic communication extracts
and transmits only the essential semantic information, reducing data
Received 24 April 2025; revised 20 September 2025; accepted 10 Novem-
ber 2025. Date of publication 17 November 2025; date of current version
24 June 2026. This work was supported in part by the National Key R&D
Program of China under Grant 2022YFB2902002, Grant2022YFB2902303,
and Grant2023YFB2904300 and in part by Shanghai Rising-Star Program. The
review of this article was coordinated by Prof. Kyunghan Lee. (Corresponding
author: Sh

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 未提及 |
| 服务更新周期 | 是 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 未提及 |
| 位置预测误差 | 未提及 |
| 服务迁移 | 未提及 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 未提及 |
| 多业务QoS | 未提及 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 是 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=未提及；服务更新周期=是；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=未提及；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=未提及；业务流量非平稳=未提及；信道动态=是 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

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

- 全文文本：已提取（6页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
