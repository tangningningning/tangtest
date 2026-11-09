# L093 — Mobility-Aware Proactive Edge Caching for Connected Vehicles Using Federated Learning

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2021
- 刊物/会议：IEEE Transactions on Intelligent Transportation Systems
- DOI：10.1109/TITS.2020.3017474
- PDF页数：11
- 原文：`C:\Users\tangning\Desktop\文献\01_时延约束下服务部署与任务卸载的自适应多时间尺度协同\Mobility-Aware Proactive Edge Caching for Connected Vehicles Using Federated Learning.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=19, UAV=0
- 主研究方向：P2；次方向：无；依据：scores={'1': 0, '2': 35, '3': 4}; title=Mobility-Aware Proactive Edge Caching for Connected Vehicles Using Federated Lea

## 2. 研究场景与问题

### 摘要窗口

未定位到章节；需回原文核查。

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

未定位到章节；需回原文核查。

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

未定位到章节；需回原文核查。

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p2起）

solution will promote the deployment of the IoV in an urban
scene and open a door to other related researches such as
routing strategy, security and privacy. Zhang et al . [10] cre-
atively proposed the Chinese remainder theorem based condi-
tional privacy-preserving authentication protocol, which only
needs realistic TPDs and grea tly reduces the computational
complexity. This will greatly promote the application of the
authentication protocol in VNs.
Edge servers ( e.g., BSs and RSUs) and vehicular users
with ample caching capacitie s can cache popular contents to
increase the agility for service provisioning. A number of
recent works [12]–[15], [16] have been reported to inves-
tigate the content caching at RSUs in vehicular networks.
Hu et al . [12] proposed a multi-object auction-based method
to solve the competition of content providers caused by the
limited storage resources of RSUs. Ding et al . [13] studied
three methods (optimal, sub-optimal and greedy) to allocate
contents on RSUs, aiming to minimise the average download-
ing time for requested contents. Su et al. [14] designed a cross-
entropy based dynamic content caching scheme to optimise
cache resources by utilising cooperation among RSUs and
the request history of vehicles. High movement of vehicles
results in unstable connectivity, and vehicles may not have
enough time to download the entire requested content

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p8起）

Simulation Settings and Dataset
We simulate a VEC environment in an urban area consisting
of an MBS, 3 RSUs and several vehicles located within the
coverage areas of RSUs. A HP Z440 workstation with 64G
memory is exploited as the MBS to store the lists of cached
contents and manage cache resources. Two HP Z440 work-
stations, working as RSUs, aggregate the parameters of
C-AAE model and cache the predicted popular contents. The
number of vehicles under each RSU varies from 1 to 100. All
vehicles have datasets to conduct local model training. Keras
is employed as the Deep learning framework to implement
C-AAE and FL, with TensorFlow as backend. The dataset we
used in our experiments is MovieLens 1M dataset collected
from the MovieLens website [ 37]. About 1 m illion ratings are
contained in this dataset, which came from 6040 anonymized
users on 3883 movies. The contextual information of users,
e.g. gender, age, address and occupation, is also provided
in the dataset. To simulate the process of vehicular users’
requests, the rated movies are assumed to request contents
from vehicles.
B. Performance Evaluation
Cache hit ratio is the performance metric we used to
evaluate the proposed MPCF, which measures the effectiveness
of a cache in fulfilling content requests. Cache hit ratio is
calculated as follows: Cache hit ratio = cache hits/ (cache
hits + cache misses). One cache hit is c

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 未提及 |
| 服务更新周期 | 未提及 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 是 |
| 位置预测误差 | 未提及 |
| 服务迁移 | 未提及 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 是 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 部分 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=未提及；服务更新周期=未提及；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=是 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=部分 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：None（PDF页）
- 问题窗口：None（PDF页）
- 方法窗口：2（PDF页）
- 仿真窗口：8（PDF页）
- 结论窗口：None（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（11页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
