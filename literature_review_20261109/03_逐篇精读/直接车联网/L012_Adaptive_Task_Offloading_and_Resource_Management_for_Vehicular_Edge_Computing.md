# L012 — Adaptive Task Offloading and Resource Management for Vehicular Edge Computing

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2026
- 刊物/会议：IEEE Transactions on Vehicular Technology
- DOI：10.1109/TVT.2025.3597459
- PDF页数：15
- 原文：`C:\Users\tangning\Desktop\文献\01_时延约束下服务部署与任务卸载的自适应多时间尺度协同\Adaptive Task Offloading and Resource Management for Vehicular Edge Computing.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=42, UAV=0
- 主研究方向：P3；次方向：1,2；依据：scores={'1': 13, '2': 20, '3': 27}; title=Adaptive Task Offloading and Resource Management for Vehicular Edge Computing

## 2. 研究场景与问题

### 摘要窗口

（PDF p10起）

system modeling. However, DQN’s
performance remains suboptimal compared to our proposed
method because the ϵ-greedy exploration strategy introduces
suboptimal actions during learning phases and neural network
function approximation can’t perfectly capture the continuous
optimization landscape. FLO suffers from high Z-values due
to vehicle computing constraints, while FEN faces server over-
load. ETOS and PA TOS show moderate performance but lack
global optimization. The performance gap between DQN and
our method decreases as server capacity increases, since abun-
dant computational resources reduce optimization complex-
ity and enable better neural network approximation accuracy.
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 29,2026 at 11:47:55 UTC from IEEE Xplore. Restrictions apply. 
JIN et al.: ADAPTIVE TASK OFFLOADING AND RESOURC

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p10起）

system modeling. However, DQN’s
performance remains suboptimal compared to our proposed
method because the ϵ-greedy exploration strategy introduces
suboptimal actions during learning phases and neural network
function approximation can’t perfectly capture the continuous
optimization landscape. FLO suffers from high Z-values due
to vehicle computing constraints, while FEN faces server over-
load. ETOS and PA TOS show moderate performance but lack
global optimization. The performance gap between DQN and
our method decreases as server capacity increases, since abun-
dant computational resources reduce optimization complex-
ity and enable better neural network approximation accuracy.
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 29,2026 at 11:47:55 UTC from IEEE Xplore. Restrictions apply. 
JIN et al.: ADAPTIVE TASK OFFLOADING AND RESOURCE MANAGEMENT FOR VEHICULAR EDGE COMPUTING 1531
Fig. 6. Impact of the number of vehicles on the Z function (a) Edge server CPU of 20 GHz (b) Edge server CPU of 25 GHz (c) Edge server CPU of 30 GHz.
Fig. 7. Impact of the maximum transmit power on the Z function.
Moreover, DQN exhibits excellent scalability and maintains sta-
ble performance across varying traffic densities, demonstrating
its strong adaptability for large-scale vehicular network deploy-
ments. Under different conditions of maximum transmission
power, the comprehensive cost comparison of various schemes is
displayed in the Fig. 7. Results indicate that higher transmission
power positively affects transmission rate, as the Z value of
comprehensive cost decreases with increased power. The PA TOS
scheme, which assigns tasks to the nearest RSU, does not ensure
a stable task transmission channel, leading to increased delay
and en

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p3起）

optimization problem that jointly optimizes vehi-
cle transmission power, local computing capabilities, and
RSU computing resources. Crucially, we consider atomic
tasks, which are indivisible and must be fully executed
either locally or remotely, and explicitly model cross-
timeslot migration, where task uploads span multiple RSU
coverage durations due to high-speed mobility. This feature
is often neglected in prior work, which assumes either static
task placement or divisible workloads.
r To explore the feasibility of addressing the randomness of
task arrivals, we exploit a novel dynamic task classification
and migration strategy based on the comparison between
task upload time and vehicle dwell time within RSU cover-
age. Tasks are categorized as single-region or cross-region
offloading tasks. This enables real-time migration deci-
sions across RSUs while preserving the atomicity of tasks.
Unlike existing methods that either rely on fixed rules or
ignore RSU handovers, our approach ensures consistent
task execution during high-speed movement and improves
system robustness.
r This study also considers a dynamic channel modeling
method, which accurately captures the dynamic changes
in wireless communication links caused by vehicle mobil-
ity. The proposed comprehensively considers the effects
of large-scale path loss, small-scale Rayleigh fading, and
Doppler shift, aiming to achieve real-time updates of chan-
nel states, as well as improve the accuracy of offloading
decisions.
r Finally, to verify the adaptability of this method in high-
dynamic vehicular network scenarios, we provide a com-
prehensive performance evaluation for different RSU de-
ployment scenarios and vehicle dynamic characteristics.
Experimental results show that the proposed strategy per-
forms excel

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p9起）

proposed scheme, this
study conducted a series of experiments and set up the following
plans for comparative analysis.
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 29,2026 at 11:47:55 UTC from IEEE Xplore. Restrictions apply. 
1530 IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 75, NO. 1, JANUARY 2026
Fig. 4. Convergence analysis under different load of tasks.
1) Full local computing (FLO): This initiative seeks to per-
form all vehicular tasks directly on the vehicle, avoiding any
offloading decisions.
2) Full edge offloading (FEN): In stark contrast to local com-
puting, this approach involves offloading all tasks that require
processing to edge servers, bypassing the use of onboard vehicle
computing.
3) Efficient Task Offloading Scheme (ETOS): This strategy
suggests that tasks should only be offloaded to MEC servers
when local computing cannot satisfy the task’s maximum tol-
erance delay requirement, thereby ensuring computational effi-
ciency and response speed [38].
4) Position-Aware Task Offloading Strategy (PATOS): This
scheme introduces a decision-making mechanism that is based
on the vehicle’s operational location and the deployment of
RSUs [39]. During task offloading, the system allocates tasks to
the closest RSU, and then each RSU is tasked with optimizing
resource allocation.
5) Deep Q-Network (DQN) based Task

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

Simulation results indicate that the proposed approach effectively
minimizes both latency and energy consumption. It notably out-
performs benchmark schemes, showing a significant reduction in
the Z value, achieving lower task delays and reduced energy con-
sumption compared to existing strategies. These findings highlight
the substantial advantages of the proposed approach, particularly
in dynamic vehicular environments.
Index Terms —Task offloading, resource allocation, dynamic
channel modeling, task classification, vehicular edge computing.
Received 9 April 2025; revised 5 July 2025; accepted 30 July 2025. Date
of publication 11 August 2025; date of current version 19 January 2026. This
work was supported in part by the National Natural Science Foundation of China
under Grant 62471258, in part by the Key Research and Development Program of
Nantong under Grant GZ2024002, and in part by the Nantong Municipal Science
and Technology Program for Social Livelihood under Grant MS2024068. The
review of this article was coordinated by Prof. Jiawen Kang. (Corresponding
author: Hao Zhu.)
Li Jin, Guoan Zhang, and Wei Duan are with the School of Information
Science and Technology, Nantong University, Nantong 226019, China (e-mail:
jin.li@ntu.edu.cn; gzhang@ntu.edu.cn; sinder@ntu.edu.cn).
Shuaili Bao and Hao Zhu are with the School of Artificial Intelligence
and Computer Science, Nantong

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
| 活动任务连续执行 | 是 |
| 迁移传输 | 未提及 |
| 多业务QoS | 未提及 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 是 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=未提及；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=是；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=未提及；业务流量非平稳=未提及；信道动态=是 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：10（PDF页）
- 问题窗口：3（PDF页）
- 方法窗口：9（PDF页）
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
