# L099 — Online Function Scheduling for Dual-Heterogeneous Serverless Vehicular Edge Computing

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2025
- 刊物/会议：IEEE Transactions on Intelligent Vehicles
- DOI：10.1109/TIV.2024.3384184
- PDF页数：17
- 原文：`C:\Users\tangning\Desktop\文献\Online_Function_Scheduling_for_Dual-Heterogeneous_Serverless_Vehicular_Edge_Computing.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=30, UAV=0
- 主研究方向：P1；次方向：无；依据：scores={'1': 18, '2': 4, '3': 3}; title=Online Function Scheduling for Dual-Heterogeneous Serverless Vehicular Edge Comp

## 2. 研究场景与问题

### 摘要窗口

（PDF p3起）

system model and formulates the joint optimization problem. We
transform the problem into a partially observable Markov game
(POMG), and a learning-based scheduling algorithm is designed
in Section IV. Simulation results are presented in Section V.
Finally, Section VI concludes our work.
II. R ELA TEDWORK
The works in[7], [8], [9], [10], [11] studied the task scheduling
problem in traditional serverful VEC systems. The authors in[7]
designed a cooperative task offloading and secure handover
scheme in VEC networks to improve computing efficiency.
Sun et al. in [8] formulated a joint resource allocation and
task offloading problem to maximize the system utility. Ma
et al. in [9] studied the vehicle-server collaborative offloading in
VEC to maximize the average throughput of all vehicles under
multi-constraint. Li et al. in[10] proposed a joint task scheduling
and resource allocation scheme

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p3起）

system model and formulates the joint optimization problem. We
transform the problem into a partially observable Markov game
(POMG), and a learning-based scheduling algorithm is designed
in Section IV. Simulation results are presented in Section V.
Finally, Section VI concludes our work.
II. R ELA TEDWORK
The works in[7], [8], [9], [10], [11] studied the task scheduling
problem in traditional serverful VEC systems. The authors in[7]
designed a cooperative task offloading and secure handover
scheme in VEC networks to improve computing efficiency.
Sun et al. in [8] formulated a joint resource allocation and
task offloading problem to maximize the system utility. Ma
et al. in [9] studied the vehicle-server collaborative offloading in
VEC to maximize the average throughput of all vehicles under
multi-constraint. Li et al. in[10] proposed a joint task scheduling
and resource allocation scheme to optimize the completion time
and energy consumption of multi-sensor in-vehicle services.
Liu et al. in [11] designed a multi-edge server cooperative
vehicular computation offloading scheme consists of a heat-
aware computation offloading strategy, a fountain code-assisted
multi-edge server cooperative resource compensation algorithm,
and a parallel transmission and execution-based scheduling al-
gorithm. However, the inherent features of the edge networks can
extremely complicate vehicular service providers’ computation
resource management process [12].
Serverless computing can enable service providers to focus
squarely on the application development while leaving other
cumbersome computation infrastructure and resource manage-
ment issues to computing vendors. Several recent excellent
works [12], [13], [14], [15], [16] have focused on the archi-
tectures and function scheduling prob

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p2起）

optimization problem with multi-queue under
unknown prior environment knowledge about the vehicle
and edge side. To guarantee the dynamic system’s long-
term performance, we propose a learning-based algorithm
in a centralized training and distributed execution (CTDE)
manner. Each vehicle can make decisions without knowing
other vehicles’ decisions and request generation models.
r Considering the trial-and-error nature in the learning al-
gorithm’s data sampling and training process, we design
a realistic factor named local-edge queue length interval
penalty (QLIP) to assist the distributed agents in measuring
each step’s scheduling decision. Extensive simulations il-
lustrate that the proposed approach outperforms baselines.
The rest of this paper is organized as follows. Section II
summarizes the related work on the topic. Section III depicts the
1Different from the concept of service scheduling in serverful computing, we
adopt the concept of function scheduling to emphasize the considered serverless
computing scenario.
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on November 03,2026 at 13:32:52 UTC from IEEE Xplore. Restrictions apply. 
2750 IEEE TRANSACTIONS ON INTELLIGENT VEHICLES, VOL. 10, NO. 4, APRIL 2025
system model and formulates the joint optimization problem. We
transform the problem into a partially observable Markov game
(POMG), and a learning-based scheduling algorithm is designed
in Section IV. Simulation results are presented in Section V.
Finally, Section VI concludes our work.
II. R ELA TEDWORK
The works in[7], [8], [9], [10], [11] studied the task scheduling
problem in traditional serverful VEC systems. The authors in[7]
designed a cooperative task offloading and secure handover
scheme in VEC networks to im

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p9起）

proposed algorithm is shown in
Algorithm 1. It is worth noting that, there can exist hundreds
of ICVs per kilometer in practical VEC networks, and the
number of ICVs within an area is random in different time
slots, the proposed algorithm can suffer an unbearable training
overhead to acquire a convergent policy. To address this issue, we
Lclip(θ)= 1BMN ∑
i∈BM
∑
n∈N
min
[
ζiθn(t) ˆAiπθn ,clip {ζiθn(t),1−ϵ,1+ϵ} ˆAiπθn ]+ ϕBMN ∑
i∈BM
∑
n∈N
S(πθn(ont )). (36)
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on November 03,2026 at 13:32:52 UTC from IEEE Xplore. Restrictions apply. 
ZHU et al.: ONLINE FUNCTION SCHEDULING FOR DUAL-HETEROGENEOUS SERVERLESS VEHICULAR EDGE COMPUTING 2757
Algorithm 1: The MAPPO-Based Serverless Function Re-
quest Scheduling Solution for the DH-SVECN.
Input: The global state s(t), the local observation vector of
all the ICVs ot.
Output: The joint request scheduling vector of all the
ICVsat.
1: Initialize each actor network with θn for each agent.
Initialize each critic network with φn for each agent.
Initialize a replay buffer BR and other corresponding
parameters.
2: Set episode timer ep =0 .
3: for each episode ep =1: EPmax do
4: Initialize global state st, local observation ont of each
ICV n; reset trajectory τ =[] ;
5: for each step t =1: T do
6: for ICV n in N do
7: Each ICV n randomly chooses an action
accordin

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

Simulation experiments show that
the proposed approach can guarantee more completed function
requests than other baselines.
Index Terms —Heterogeneous vehicular edge computing,
serverless edge computing, function scheduling, multi-agent
reinforcement learning.
I. I NTRODUCTION
A
N ADV ANCED intelligent and connected vehicle (ICV)
will be installed with no fewer than 200 sensors (i.e.,
Manuscript received 11 March 2024; accepted 26 March 2024. Date of
publication 3 April 2024; date of current version 27 August 2025. This work
was supported in part by the National Key R&D Program of China under
Grant 2022YFB2902100, in part by the Startup Foundation for Introducing
Talent of NUIST under Grant 2023r134, in part by the National Nature Science
Foundation of China under Grant 62201271, in part by the China Postdoctoral
Science Foundation under Grant 2023M741781, and in part by the Basic Science
Research Project of Jiangsu Colleges and Universities of China under Grant
22KJB510003. (Corresponding Author: Zhizhong Zhang.)
Lei Zhu and Chengjie Hou are with the School of Communication and Infor-
mation Engineering, Chongqing University of Posts and Telecommunications,
Chongqing 400065, China (e-mail: lei.zhu7@foxmail.com; d190101003@stu.
cqupt.edu.cn).
Hanzhong Huang is with the Hangzhou Hikvision Digital Technology Com-
pany, Ltd., Hangzhou 310051, China (e-mail: huanghanzhong@hikvision

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
| 信道动态变化 | 部分 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=未提及；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=部分 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：3（PDF页）
- 问题窗口：2（PDF页）
- 方法窗口：9（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：None（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（17页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
