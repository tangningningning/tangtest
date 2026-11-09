# L126 — Two-Timescale Adaptive Live Video Streaming Transmission Mechanism for Vehicular Networks

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2025
- 刊物/会议：IEEE Transactions on Vehicular Technology
- DOI：10.1109/TVT.2024.3507816
- PDF页数：6
- 原文：`C:\Users\tangning\Desktop\文献\01_时延约束下服务部署与任务卸载的自适应多时间尺度协同\Two-Timescale_Adaptive_Live_Video_Streaming_Transmission_Mechanism_for_Vehicular_Networks.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=20, UAV=0
- 主研究方向：P3；次方向：1；依据：scores={'1': 12, '2': 2, '3': 16}; title=Two-Timescale Adaptive Live Video Streaming Transmission Mechanism for Vehicular

## 2. 研究场景与问题

### 摘要窗口

未定位到章节；需回原文核查。

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

未定位到章节；需回原文核查。

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p2起）

optimization problem is formulated as
max
b∈B,w≥0
E
{
max
P≥0
ψ(b, w, P, G)
}
, (5a)
s.t. ‖pt‖1 ≤ pmax, ∀t, (5b)
‖w‖1 ≤ wmax. (5c)
It is arduous to solve Problem (5) due to Challenge I: the objective
function (5a) is non-convex with respect to w and P;a n dChallenge II:
the discrete variable bv implies that the problem is NP-hard. General
speaking, there is no standard optimal solution for such a two timescale
optimization problem. In the following section, we will propose an
efficient solution based on two-stage online successive convex approx-
imation (TOSCA) algorithm.
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 30,2026 at 03:24:38 UTC from IEEE Xplore. Restrictions apply. 
IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 74, NO. 4, APRIL 2025 6825
III. D ESIGN OF ADAPTIVE VIDEO STREAMING TRANSMISSION
MECHANISM
A. Overall Description
We first relax discrete bitrates to continuous variables to address
Challenge II . Subsequently, we adopt the TOSCA algorithm [17] to
solve the approximate non-convex optimization problem efficiently to
address Challenge I. Finally, we map the relaxed bitrates into the given
bitrate set B and refine the bandwidth allocation using the TOSCA
algorithm again.
1) Relaxation Stage: Let ˜B ≜ [b(1),b(S)] be the relaxed version
ofB and the initial problem (5) is transformed to the following approx-
imated problem with convex constraints:
max
b∈ ˜B,w≥0
E
{
max
P≥0
ψ(b, w, P, G)
}
, (6a)
s.t. ‖pt‖1 ≤ pmax, ∀t, (6b)
‖w‖1 ≤ wmax. (6c)
Using primal decomposition, the problem can be decomposed into
a large-timescale optimization sub-problem and a family of small-
timescale optimization sub-problems, which are described as follows:
i) Small-timescale optimization problem: Given the channel gain gt

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p4起）

proposed algorithm,
the duration of the short-timescale slot is set as 10ms, the number of time
slots within each large-scale frame T is 500. the weights are set ω1 =
ω2 = ω3 = 1. In the channel model, the Rician K-factor is set as 5 and
the model of large-scale path loss is PL(d)= 50+ 25log10(d)+ Xσ
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 30,2026 at 03:24:38 UTC from IEEE Xplore. Restrictions apply. 
IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 74, NO. 4, APRIL 2025 6827
Algorithm 1: The TOSCA-Based Adaptive Live Video Streaming
Transmission Mechanism.
with Xσ being the shadow fading. In live video transmission ser-
vices, the commonly used available bitrates include 5.8Mbps, 3Mbps,
1.75Mbps, and 0.75Mbps. 1 The available bandwidth wmax is 10 MHz.
The coverage radius of the RSU is 200 m. The speed of the vehicles
ranges from 100 to 120 km/h. Considering the safety distance between
vehicles, we assume that there are two moving VUs in each lane within
the coverage radius of the RSU.
The proposed algorithm are compared with the following baselines:
i) Baseline 1: Bandwidth and bitrate are optimized at each large-
timescale frame, and the power is uniformly allocated; ii) Baseline
2: Bandwidth and bitrate are optimized based on a fixed uniform
power allocation at the beginning of each large-timescale frame, and
the pow

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

Simulation results illustrate the rationality and necessity of
the proposed dual-time scale optimization, and the proposed mechanism
noticeably outperforms the conventional resource allocation schemes under
different power budgets and lane configurations.
Index Terms —Live video transmission, vehicular networks, resource
allocation, two-timescale optimization.
I. I NTRODUCTION
Given the rapid rise of on-vehicle video services such as video
conferencing and video surveillance, the contradiction between the
highly time-varying vehicular communication environments and users’
demands for a clear and seamless viewing experience has intensified[1].
Traditional video streaming distribution systems complete content ac-
quisition, caching and transcoding in the cloud. Although this can
alleviate the pressure on cloud storage, there are still serious issues
such as large transmission latency, high bandwidth consumption and
redundant transmission in the network edge [2]. With the advancement
of vehicular edge computing (VEC) technology, the service delay and
the communication overhead in the core network can be significantly
reduced by using the road side units (RSUs) to download the original
bitrate version of the video from the cloud and dynamically transcoding
it to the bitrate version suitable for different user channels [3].
Adaptive video streaming in VEC has become a focal point fo

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
| 迁移传输 | 未提及 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 是 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=部分；服务更新周期=是；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=是 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：None（PDF页）
- 问题窗口：2（PDF页）
- 方法窗口：4（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：5（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（6页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
