# L035 — DDPG-Based Two-Timescale Joint Resource Allocation and Channel-Aware Task Offloading for Vehicular Edge-Computing Networks

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2026
- 刊物/会议：IEEE Internet of Things Journal
- DOI：10.1109/JIOT.2026.3685527
- PDF页数：11
- 原文：`C:\Users\tangning\Desktop\文献\2026年论文\DDPG-Based_Two-Timescale_Joint_Resource_Allocation_and_Channel-Aware_Task_Offloading_for_Vehicular_Edge-Computing_Networks.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=68, UAV=0
- 主研究方向：P1；次方向：3；依据：scores={'1': 15, '2': 2, '3': 8}; title=DDPG-Based Two-Timescale Joint Resource Allocation and Channel-Aware Task Offloa

## 2. 研究场景与问题

### 摘要窗口

（PDF p3起）

system model for multivehicle VEC
networks. In Section III, a min-max optimization problem
is established to achieve the average processing delay min-
imization. In Section IV, a DDPG-based TTORA algorithm is
proposed. Section V presents simulation results and analysis.
Finally, we conclude this article in Section VI.
II. S YSTEM MODEL
A. System Architecture
In this work, we investigate a multivehicle VEC network
with one RSU connected with a VEC server and multiple
vehicles within the communication range of the RSU, as
shown in Fig. 1. These vehicles are randomly distributed on
a unidirectional two-lane road. The arrival of vehicles at the
road section covered by the RSU follows a Poisson process
Fig. 2. Illustration of two-timescale joint resource allocation and subtask
oﬄoading: 1) bandwidth allocation and computing resource allocation deci-
sions at each long time slot and 2) subtask

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p3起）

system model for multivehicle VEC
networks. In Section III, a min-max optimization problem
is established to achieve the average processing delay min-
imization. In Section IV, a DDPG-based TTORA algorithm is
proposed. Section V presents simulation results and analysis.
Finally, we conclude this article in Section VI.
II. S YSTEM MODEL
A. System Architecture
In this work, we investigate a multivehicle VEC network
with one RSU connected with a VEC server and multiple
vehicles within the communication range of the RSU, as
shown in Fig. 1. These vehicles are randomly distributed on
a unidirectional two-lane road. The arrival of vehicles at the
road section covered by the RSU follows a Poisson process
Fig. 2. Illustration of two-timescale joint resource allocation and subtask
oﬄoading: 1) bandwidth allocation and computing resource allocation deci-
sions at each long time slot and 2) subtask-o ﬄoading decision at each short
time slot.
with rates λ [27]. It is assumed that the time-slot length for
the vehicle generating a task is at the second level (i.e., the
time slot is a long time slot), while the time-slot length for
block-fading Rayleigh remains constant within a time slot at
the millisecond level (i.e., the time slot is a short time slot)
[28], [29]. We adopt a discrete time model which is equally
divided into I long time slots, where the length of each long
time slot is ∆T [30], [31]. The set of vehicles is denoted
by Ni = {1,..., Ni} at the ith long time slot. Each vehicle
has a delay-sensitive or computation-intensive task at the ith
long time slot, which is described as ( Dn,i,δ n,i,τ maxn,i ), where
Dn,i, cn,i, and τmaxn,i denote the task size, the task computation
density, and the maximum tolerable delay, respectively. It
is assumed that each task is divided une

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p1起）

optimization problem. To solve it, a deep deterministic
policy gradient (DDPG)-based two-timescale task o ﬄoading and
resource allocation (DDPG-TTORA) algorithm is proposed to
obtain resource allocation decisions at each long time slot and
subtask-oﬄoading decisions at each short time slot. Simulation
results demonstrated that the proposed DDPG-TTORA algorithm
can reduce the average processing delay by 36.75% compared to
the full partial o ﬄoading (FPO) algorithm.
Index Terms —Deep deterministic policy gradient (DDPG),
resource allocation, task o ﬄoading, two-timescale, vehicular edge
computing (VEC).
NOMENCLATURE
T linkn,i Link connection time between the nth vehicle
and the RSU in the ith long time slot.
Received 12 February 2026; revised 2 April 2026; accepted 15 April 2026.
Date of publication 20 April 2026; date of current version 29 June 2026.
This work was supported in part by the National Natural Science Foun-
dation of China under Grant 62572167, Grant 62427809, Grant 52277087,
and Grant 62501365; and in part by the Natural Science Foundation of
Shandong Province, China, under Grant ZR2025QC1637. (Corresponding
author: Qiyue Li.)
Yang Xia, Lijuan Li, and Qiyue Li are with the School of Elec-
trical Engineering and Automation, Hefei University of Technology,
Hefei, Anhui 230009, China, and also with the Engineering Technology
Research Center of Industrial Automation of Anhui Province, Hefei, Anhui
230009, China (e-mail: yangxia@hfut.edu.cn; lilijuan@mail.hfut.edu.cn;
liqiyue@mail.ustc.edu.cn).
Tong Xue is with the College of Electronic and Information Engineering,
Shandong University of Science and Technology, Qingdao, Shandong 266590,
China (e-mail: tongxue@sdust.edu.cn).
Daojun Liang is with the Key Laboratory of Computing Power Network
and Information Securit

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p7起）

proposed algorithm under di ﬀerent learning rates.
respectively. Each task is divided into multiple unequal sub-
tasks. The size range of each subtask is set to [15, 35] kb [26].
During training, the experience replay bu ﬀer size, mini-batch
size, and the discount factor are set to 10 000, 64, and 0.99.
Other key parameter settings are detailed in Table I.
B. Convergence of the Proposed DDPG-TTORA
The convergence of the proposed DDPG-TTORA is inves-
tigated under di ﬀerent learning rates, i. e., LRµ ={0.1, 0.01},
LRQ ={0.1, 0.01}, when the task size is all set to be 2 Mbits,
the task computation density is set to be 500 cycles /bit for
all tasks, the number of the vehicles is set to be 10 and the
computing capability of the VEC server is set to be 50 GHz.
DDPG-TTORA exhibits good convergence performance as
shown in Fig. 4. It is also shown that when LRµ = {0.01}
and LRQ ={0.01}, DDPG-TTORA performs best. Therefore,
we use these values for the rest of the simulations.
C. Delay Performance of the Proposed DDPG-TTORA
To demonstrate the superiority of the proposed DDPG-
TTORA, we compare it with four existing schemes in the
VEC networks.
Fig. 5. Average processing delays of the proposed DDPG-TTORA, FPO,
FVC, and FLC for di ﬀerent task sizes.
Fig. 6. Average processing delays of the proposed DDPG-TTORA, FPO,
FVC, and FLC for di ﬀerent task computation densities.
1) FPO: Each vehicle

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

Simulation
results demonstrated that the proposed DDPG-TTORA algorithm
can reduce the average processing delay by 36.75% compared to
the full partial o ﬄoading (FPO) algorithm.
Index Terms —Deep deterministic policy gradient (DDPG),
resource allocation, task o ﬄoading, two-timescale, vehicular edge
computing (VEC).
NOMENCLATURE
T linkn,i Link connection time between the nth vehicle
and the RSU in the ith long time slot.
Received 12 February 2026; revised 2 April 2026; accepted 15 April 2026.
Date of publication 20 April 2026; date of current version 29 June 2026.
This work was supported in part by the National Natural Science Foun-
dation of China under Grant 62572167, Grant 62427809, Grant 52277087,
and Grant 62501365; and in part by the Natural Science Foundation of
Shandong Province, China, under Grant ZR2025QC1637. (Corresponding
author: Qiyue Li.)
Yang Xia, Lijuan Li, and Qiyue Li are with the School of Elec-
trical Engineering and Automation, Hefei University of Technology,
Hefei, Anhui 230009, China, and also with the Engineering Technology
Research Center of Industrial Automation of Anhui Province, Hefei, Anhui
230009, China (e-mail: yangxia@hfut.edu.cn; lilijuan@mail.hfut.edu.cn;
liqiyue@mail.ustc.edu.cn).
Tong Xue is with the College of Electronic and Information Engineering,
Shandong University of Science and Technology, Qingdao, Shandong 266590,
China (e-mail: tongx

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 是 |
| 服务更新周期 | 是 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 是 |
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
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=是；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=未提及；业务流量非平稳=未提及；信道动态=是 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：3（PDF页）
- 问题窗口：1（PDF页）
- 方法窗口：7（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：None（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（11页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
