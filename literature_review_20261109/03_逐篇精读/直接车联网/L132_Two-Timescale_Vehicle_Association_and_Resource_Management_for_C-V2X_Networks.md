# L132 — Two-Timescale Vehicle Association and Resource Management for C-V2X Networks

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2023
- 刊物/会议：IEEE Wireless Communications Letters
- DOI：10.1109/LWC.2023.3270318
- PDF页数：5
- 原文：`C:\Users\tangning\Desktop\文献\01_时延约束下服务部署与任务卸载的自适应多时间尺度协同\Two-Timescale Vehicle Association and Resource Management for C-V2X Networks.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=29, UAV=0
- 主研究方向：P3；次方向：1；依据：scores={'1': 21, '2': 2, '3': 25}; title=Two-Timescale Vehicle Association and Resource Management for C-V2X Networks

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

System model for C-V2X networks.
proposed. Given the long-timescale decision, Lagrangian dual
decomposition is used to determine the closed-form expression
for each vehicle’s transmit power that satisfying the instanta-
neous QoS requirements of various services. It is worth noting
that the signaling overhead of the proposed scheme lies mainly
on the long-timescale. Therefore, the proposed TT-V ARMS
achieves a win-win situation in terms of the signaling overhead
and the system sum-rate, while satisfying the instantaneous
QoS requirements of various services.
II. S YSTEM MODEL AND PROBLEM FORMULATION
A. System Model
This letter studies an uplink C-V2X network where the dif-
ferentiated QoS requirements of both CUEs and VUEs are
greatly satisfied, as shown in Fig. 1. In the network, a macro
base station (MBS) is deployed to support the delay-insensitive
service of CUE u, u ∈U = {1,2,..., U

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

System model for C-V2X networks.
proposed. Given the long-timescale decision, Lagrangian dual
decomposition is used to determine the closed-form expression
for each vehicle’s transmit power that satisfying the instanta-
neous QoS requirements of various services. It is worth noting
that the signaling overhead of the proposed scheme lies mainly
on the long-timescale. Therefore, the proposed TT-V ARMS
achieves a win-win situation in terms of the signaling overhead
and the system sum-rate, while satisfying the instantaneous
QoS requirements of various services.
II. S YSTEM MODEL AND PROBLEM FORMULATION
A. System Model
This letter studies an uplink C-V2X network where the dif-
ferentiated QoS requirements of both CUEs and VUEs are
greatly satisfied, as shown in Fig. 1. In the network, a macro
base station (MBS) is deployed to support the delay-insensitive
service of CUE u, u ∈U = {1,2,..., U}, S road side units
(RSUs) indexed by s ∈S = {1,2,..., S} with coverage
radius D are deployed at the roadside to support the delay-
sensitive service of VUE m, m ∈M = {1,2,..., M}.I t
is assumed that there are totally F available resource blocks
(RBs) and F = U. Hence, each CUE can utilize a single RB,
i.e., no spectrum reuses between CUEs. To improve the spec-
trum resource utilization, U CUEs and M VUEs are allowed to
share the same frequency band. Since M ≫ U, each RB may
be occupied by one CUE and multiple VUEs. Thus, the signal-
to-interference-plus-noise ratio (SINR) at the MBS from the
uth CUE and the st hR S Uf r o mt h emth VUE can be separately
expressed as
γCu = P opu hu,MBS∑
m∈M Xm,u Pm hm,MBS +σ2, (1)
γVm,u,s = Pm hm,s∑
m′∈{M\m} Xm′,u Pm′hm′,s + P opu hu,s +σ2, (2)
where P opu and Pm denote the transmit power of the uth CUE
and the mth VUE, respectively. Specifically, we as

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p2起）

PROBLEM FORMULATION
A. System Model
This letter studies an uplink C-V2X network where the dif-
ferentiated QoS requirements of both CUEs and VUEs are
greatly satisfied, as shown in Fig. 1. In the network, a macro
base station (MBS) is deployed to support the delay-insensitive
service of CUE u, u ∈U = {1,2,..., U}, S road side units
(RSUs) indexed by s ∈S = {1,2,..., S} with coverage
radius D are deployed at the roadside to support the delay-
sensitive service of VUE m, m ∈M = {1,2,..., M}.I t
is assumed that there are totally F available resource blocks
(RBs) and F = U. Hence, each CUE can utilize a single RB,
i.e., no spectrum reuses between CUEs. To improve the spec-
trum resource utilization, U CUEs and M VUEs are allowed to
share the same frequency band. Since M ≫ U, each RB may
be occupied by one CUE and multiple VUEs. Thus, the signal-
to-interference-plus-noise ratio (SINR) at the MBS from the
uth CUE and the st hR S Uf r o mt h emth VUE can be separately
expressed as
γCu = P opu hu,MBS∑
m∈M Xm,u Pm hm,MBS +σ2, (1)
γVm,u,s = Pm hm,s∑
m′∈{M\m} Xm′,u Pm′hm′,s + P opu hu,s +σ2, (2)
where P opu and Pm denote the transmit power of the uth CUE
and the mth VUE, respectively. Specifically, we assume that
P opu is fixed for each CUE and is known to all RSUs. σ2 is the
noise power, the binary variable Xm,u ∈{ 0,1} represents the
spectrum assignment indicator, if the mth VUE shares the RB
with the uth CUE, Xm,u =1 , otherwise, Xm,u =0 , hu,MBS
and hm,s denote the channel gain from the uth CUE to the
MBS and from the mth VUE to the sth RSU, respectively,
hm,MBS and hu,s are the interference channel power gain
from the mth VUE to the MBS and from the uth CUE to the
sth RSU, respectively.
This letter adopts a discrete time model which is divided
into two-timescale, i.e., long-

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p4起）

proposed algorithm. All the simulations are done by
using MATLAB R2020a on a computer with an Intel Core
i5-8250U (1.6GHz/L3 6M) and 8GB of RAM. It is assumed
that all the users are located within a target rectangular area
1000 m×1000 m. The simulation parameters are set accord-
ing to 3GPP TR 36.885, where a MBS is located at the
center of the considered area and a number of RSUs with
D = 150 m are located at the roadside [ 10]. The number
of lanes is 6, and the width of each lane is 4 m [ 6]. The
moving speed of vehicles is set as [10, 60] km/h. The aver-
age inter-vehicle distance in the same lane is 2.5 v m with v
representing the moving speed of vehicles in meter per sec-
ond [5]. Besides, we set P maxm =1 7d B m, P opu =2 5d B m,
σ2 = −104 dBm. The SINR thresholds for the V2X link
receiver and cellular link receiver are set to 5 dB and 0 dB,
respectively. The shadowing standard deviations for the V2X
link and the cellular link are 8 dB and 4 dB, respectively. The
whole simulation time is τslot IT where τslot =1 0m s, I = 10
and T = [60, 140] [ 4], [8].
The impact of the number of VUEs (i.e., M) and the
length of long-timescale (i.e., T) on the system performance
(i.e., the system sum-rate, the probability of satisfying QoS
requirements over all users, the system signaling overhead)
are simulated, respectively. The obtained results are shown in
Figs. 3 and 4. Considering t

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

Simulation results show that the proposed scheme can
significantly enhance the system sum-rate, guarantee the QoS
requirements of users and reduce the system signaling overhead.
Index Terms—C-V2X, two-timescale, vehicle association, spec-
trum assignment, power control.
I. I NTRODUCTION
T
HE EMERGENCE of vehicle-to-everything (V2X)
communications paves the way for intelligent transporta-
tion systems and autonomous driving [ 1]. Recently, together
with the explosive growth of vehicles and V2X services
(i.e., delay-sensitive services and delay-insensitive services),
cellular-based V2X (C-V2X) networks have drawn exten-
sive attention [ 2]. For C-V2X networks, the vehicular users
(VUEs) co-exist with the cellular users (CUEs) in the spec-
trum band, resulting in grievous co-channel interference, and
finally decreasing the system sum-rate performance. To tackle
Manuscript received 10 March 2023; accepted 21 April 2023. Date of pub-
lication 25 April 2023; date of current version 11 July 2023. This work was
supported in part by the Project of International Cooperation and Exchanges
NSFC under Grant 61860206005, and in part by the Joint Funds of the NSFC
under Grant U22A2003. The associate editor coordinating the review of this
article and approving it for publication was Z. Liu. (Corresponding author:
Haixia Zhang.)
Tong Xue and Haixia Zhang are with the Shandong Key Laboratory of

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 未提及 |
| 服务更新周期 | 是 |
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
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=未提及；服务更新周期=是；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=未提及 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：2（PDF页）
- 方法窗口：4（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：None（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（5页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
