# L044 — DRIVE: Dynamic Resource-Aware Intelligent Task Offloading in Vehicular Edge Computing

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2026
- 刊物/会议：IEEE Transactions on Mobile Computing
- DOI：10.1109/TMC.2026.3710011
- PDF页数：16
- 原文：`C:\Users\tangning\Desktop\文献\01_时延约束下服务部署与任务卸载的自适应多时间尺度协同\DRIVE- Dynamic Resource-Aware Intelligent Task Offloading in Vehicular Edge Computing.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=37, UAV=0
- 主研究方向：P3；次方向：1,2；依据：scores={'1': 12, '2': 18, '3': 19}; title=DRIVE: Dynamic Resource-Aware Intelligent Task Offloading in Vehicular Edge Comp

## 2. 研究场景与问题

### 摘要窗口

未定位到章节；需回原文核查。

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

未定位到章节；需回原文核查。

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p3起）

optimization problems into an end-
to-end learning framework via deep neural networks, thereby
enhancing adaptability in dynamic environments. However, the
aforementioned studies do not incorporate quantitative analysis
of link disruption risks caused by vehicle mobility, making them
difficult to adapt to highly-dynamic VEC scenarios.
V2V offloading effectively mitigates the issue of insufficient
ES deployment density by utilizing idle computing resources
of surrounding vehicles to establish a distributed collaborative
computing network. Based on differences in vehicle states,
existing research can be classified into static V2V offloading
and dynamic V2V offloading: 1) In static scenarios, Li et al. [7]
utilized a contract theory-based resource scheduling scheme for
parked vehicles, which enhances the computing capacity of the
edge layer by designing an incentive mechanism to encourage
parked vehicles to share idle resources. Although such schemes
can leverage stable static resources, their coverage is limited
and struggles to meet the real-time demands of mobile vehicles.
2)Dynamic V2V offloading focuses on resource sharing among
moving vehicles, requiring solutions to the challenges of link
connectivity and resource dynamics. Y e et al. [14] treated each
vehicle as an independent agent, employing deep reinforce-
ment learning to enable autonomous selection of spectrum and
power for V2V links, thereby reducing transmission overhead
through decentralized decision-making. Qi et al. [15] integrated
knowledge-driven approaches with Deep Reinforcement Learn-
ing to achieve multi-objective offloading decisions among ve-
hicles, ESs, and base stations. Furthermore, Dai et al. [35] pro-
posed a learning-based V2V computation offloading algorithm
that enhances the awareness of

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p14起）

proposed algorithm and mit-
igate the stochastic effects induced by random seeds, we con-
ducted rigorous statistical experiments on the Shanghai dataset.
The experimental configuration was defined as follows: the task
data size followed a normal distribution with a mean of 3 MB and
a variance of 0.8. For the DRIVE, DDPG, and DQN algorithms,
we performed 10 independent trials using 10 distinct random
seeds (0–9) and recorded the task latency metrics. Table III
presents the statistical results in the form of mean ± standard
deviation. The experimental data indicate that DRIVE not only
achieves optimal average performance but also significantly
outperforms the baseline algorithms in terms of stability.
Fig. 8. Actor loss and critic loss.
TABLE III
STA TISTICALPERFORMANCE COMPARISON
The average latency of DRIVE is 0.504s, representing a reduc-
tion of 4.7% compared to DDPG (0.529s) and 12.2% compared
to DQN (0.574s). Furthermore, the standard deviation of DRIVE
is a mere 0.037, which is 28.8% lower than that of DDPG
(0.052) and 47.9% lower than that of DQN (0.071). These results
demonstrate that DRIVE maintains the lowest average latency
while exhibiting superior robustness against disturbances and
higher convergence consistency. In contrast, DQN exhibits the
highest volatility due to the constraints of a discrete action
space and high variance in value function estimation. While

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p11起）

simulation parameters
are summarized in Table II.
The baseline methods in this paper are as follows:
r Random: The client ICV randomly selects an ES or a
service ICV to offload its task request.
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 26,2026 at 11:58:57 UTC from IEEE Xplore. Restrictions apply. 
20238 IEEE TRANSACTIONS ON MOBILE COMPUTING, VOL. 25, NO. 11, NOVEMBER 2026
r Genetic Algorithm (GA) [30]: This algorithm mimics the
biological processes of mutation, inheritance, and selection
that underpin natural selection and population reproduc-
tion, iteratively searching the solution space for the service
offloading problem in the VEC environment to approxi-
mate the optimal offloading decision scheme.
r Immune Algorithm (IA) [31]: The Immune Algorithm
draws inspiration from the antigen recognition, antibody
generation, and immune regulation mechanisms of the bi-
ological immune system, while also incorporating concepts
from genetic evolution. Its process for solving the service
offloading problem in the VEC environment is as follows:
first, it identifies the problem objective and generates initial
candidate solutions. Second, it evaluates the quality of the
antibodies using a fitness function. Finally, it employs vari-
ous immune operations to optimize the candidate solutions,
progressively searching for the optimal of

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
| 服务迁移 | 是 |
| 活动任务连续执行 | 部分 |
| 迁移传输 | 未提及 |
| 多业务QoS | 部分 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 部分 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=是；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=是；活动任务连续=部分；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=部分；业务流量非平稳=未提及；信道动态=部分 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：None（PDF页）
- 问题窗口：3（PDF页）
- 方法窗口：14（PDF页）
- 仿真窗口：11（PDF页）
- 结论窗口：None（PDF页）

### 关键词命中片段

- p3: sting the task ofﬂoading probability, although their assumption of unlimited ES resources did not account for practical load constraints. With the deployment of multi-edge servers becoming mainstream, load balancing and service migration have emerged as key research priorities in V2I ofﬂoading. Dai et al. [12] used an optimization theory-based mult
- p3: gy and time-varying link connectivity within the VEC environment. Effectively integrating mobility factors into ofﬂoading decisions has become crucial for ensuring service continuity. Y uan et al. [16] jointly optimized service migration and vehicle mobility, employing a multi-agent deep reinforce- ment learning algorithm to combat the high-dimensi

## 12. 阅读闭合状态

- 全文文本：已提取（16页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
