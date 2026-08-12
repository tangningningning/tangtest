# 论文实现规范缓存

> 本文件是根据 11.pdf 生成的代码实现规范缓存，不是论文的新版本。
> 唯一最终理论依据是 11.pdf。img/ 仅用于页面视觉核对。
>
> 权威优先级：11.pdf > paper_spec.md > GitHub/官方代码 > 旧工程代码。
> 如果本文件与 11.pdf 冲突，必须以 11.pdf 为准，并只修订本文件本身。

### 【当前求解方法说明】

第三章 P0、`G(t)`、C1-C16、任务效用、缓存更新成本和 UAV 移动成本继续严格
采用 11.pdf 的定义。本项目只正式替换第四章求解方法：不再把 11.pdf 原始的
PPO 服务评分 `y_{m,f}`、PPO 连续 UAV 位移 `e_u` 和条件因子化混合动作 PPO
作为当前工程主链，改用：

    MultiBinary PPO 自适应触发
    + 确定性 Cache/UAV 配置优化
    + transfer-exchange Matching/KKT

这是第四章求解器替换，不修改第三章目标函数。当前算法是分层近似在线求解，
不声称获得 P0 全局最优解；正式性能仍只按未折扣原始 `G(t)` 和第三章指标评价。
下文明确区分 **11.pdf original solver design** 与
**current implementation solver**，原论文求解方案保留为历史理论记录。

## 0. 缓存有效性与一次性读取规则

### 当前论文版本信息

- 论文文件：11.pdf
- 文件 SHA256：E27CD10710474B5E6D614E3850654D3B995B054BF774CF3473EF7B6D297FF84B
- 当前 PDF 页数：37
- img/ 对应性：是。当前目录包含 11_页面_01.jpg 至 11_页面_37.jpg 共 37 张图片，已逐页与 PDF 内容、公式和版面核对。
- PDF 元数据未提供独立的版本号；文件元数据作者为 ning tang。实现版本判断以 SHA256 为准。
- 本次核对范围：完整读取 11.pdf，并逐页核对 img/。

### 后续每个 Codex 任务开始前的检查

1. 先计算当前 11.pdf 的 SHA256。
2. 如果 SHA256 等于本文件顶部记录值，默认只读取本文件，不重新完整读取 11.pdf，也不遍历全部 img/。
3. 仅在以下情况打开 PDF：
   - 本文件存在歧义；
   - 当前任务涉及本文件未记录的内容；
   - 用户要求直接核对论文；
   - 只需核对某一个具体公式。
4. 第 3 条中的核对只读取对应页，不重新完整读取整篇 PDF。
5. 如果当前 SHA256 与本文件记录值不同，判定论文已更新：重新完整读取 PDF，核对对应 img/，覆盖本文件并更新 SHA256。始终只保留一个 paper_spec.md，不得生成 paper_spec_v2.md、paper_spec_old.md 或 paper_spec_backup.md。
6. PDF 未变化时，禁止每次任务重新遍历全部 img/；只有本文件不能确认的公式、流程图或系统结构图才打开对应页面图片。

PowerShell 快速检查示例：

    (Get-FileHash -Algorithm SHA256 -LiteralPath "C:\Users\tangning\Desktop\code\adaptive_cache_mec\11.pdf").Hash.ToUpperInvariant()

## 1. 系统实体与集合

| 实体/符号 | 论文定义 | 代码含义 |
|---|---|---|
| task vehicle | 任务车辆，v in V minus V_s | 产生不可分割计算任务的车辆；本地缓存固定且不在线更新 |
| service vehicle | 服务车辆，V_s subset V | 具有缓存、计算和车间通信能力；身份、数量和 SUMO 路线预先确定 |
| RSU | 路侧单元，r in R | 固定位置的地面服务节点，参与缓存、计算和共享带宽 |
| UAV | 无人机，u in U | 固定高度 H 飞行，只优化二维水平位置；参与缓存、计算和空地通信 |
| Cloud | 云服务器，记为 c | 存储全部服务，使用独立蜂窝链路和固定等效计算频率 |
| service | 服务/程序，f in F | 任务所需程序；服务大小为 s_f，可被缓存 |

集合和基础量：

- 区域：A = [0, L_x] x [0, L_y]。
- 车辆、RSU、UAV：V = {1, ..., V}、R = {1, ..., R}、U = {1, ..., U}。
- 服务车辆：V_s subset V；任务车辆：V minus V_s。
- 可更新边缘服务节点：M = V_s union R union U。
- 服务集合：F = {1, ..., F}；基础时隙集合：T = {1, ..., T}。
- Cloud c 不属于 M，但属于每个任务车辆的候选执行集合。
- 基础时隙包含配置决策和任务调度两个逻辑阶段，不把 Delta t 物理切成两个时段。

Source: 11.pdf p.1-2

## 2. 关键变量总表

“长/短时间尺度”表示论文的决策时标：长时间尺度指跨时隙配置，短时间尺度指当前实际任务到达后的任务调度。固定参数和外生观测不属于状态或动作。

### 2.1 集合、位置、时间和预测变量

| 变量 | 含义 | 维度/范围 | 状态/动作 | 时标 |
|---|---|---|---|---|
| q_v(t) | 车辆 v 的实际水平位置 | R^2，属于 A | 外生观测 | 当前时隙 |
| q_r | RSU r 的固定水平位置 | R^2，属于 A | 外生常量 | 全时域 |
| q_u(t) | UAV u 的实际水平位置 | R^2，属于 A | 实际配置结果/长时标变量 | 跨时隙 |
| H | UAV 固定飞行高度 | 正数；论文未给数值 | 固定参数 | 全时域 |
| Delta t | 相邻决策时刻时间间隔 | 正数；论文未给数值 | 固定参数 | 全时域 |
| xi_v(t) | 车辆运动状态 [x_v,y_v,v_v^x,v_v^y]^T | R^4 | 预测器输入状态 | 当前/历史 |
| hat q_v(t) | 时隙 t 的预测车辆位置 | R^2，属于 A | 配置状态 | 长时标 |
| rho_f(t) | 实际服务 f 的时变流行度 | [0,1]，sum rho=1 | 外生请求统计 | 当前/历史 |
| hat rho_f(t) | 预测服务流行度 | [0,1]，sum hat rho=1 | 配置状态 | 长时标 |
| A | 卡尔曼状态转移矩阵 | 论文未给具体数值/结构 | 固定预测参数 | 全时域 |
| H_pos | 位置提取矩阵 | 论文未给具体矩阵 | 固定预测参数 | 全时域 |

说明：PDF 同时使用 H 表示 UAV 高度和位置提取矩阵。代码中必须使用不冲突的变量名区分二者，例如 uav_height 和 position_extract_matrix。这是符号消歧，不改变论文模型。

### 2.2 服务、缓存和配置变量

| 变量 | 含义 | 维度/范围 | 状态/动作 | 时标 |
|---|---|---|---|---|
| s_f | 服务 f 的程序大小 | 正数；论文未给数值范围 | 固定参数 | 全时域 |
| K_m | 节点 m 的缓存容量 | 正数；论文未给数值 | 固定参数 | 全时域 |
| c_{m,f}(t) | 可更新节点 m 是否缓存服务 f | {0,1} | 实际配置状态 | 长时标 |
| c_{v,f}(0) | 任务车辆 v 的固定本地缓存 | {0,1} | 初始/固定状态 | 全优化时域不变 |
| z_m^ca(t) | 实际缓存更新变量 | {0,1} | 实际配置结果 | 长时标 |
| tilde z_m^ca(t) | PPO 输出的缓存更新请求/意图 | {0,1} | PPO 辅助动作 | 长时标 |
| y_{m,f}(t) | 11.pdf 原求解器的服务评分；当前实现不使用 | 0 < y < 1 | original solver 动作 | 长时标 |
| c(t-1) | 上一时隙可更新缓存状态 | 仅含 M 中节点的缓存 | 配置状态 | 长时标 |

### 2.3 11.pdf 原 UAV 辅助动作和可行化变量

| 变量 | 含义 | 维度/范围 | 状态/动作 | 时标 |
|---|---|---|---|---|
| z_u^q(t) | 可行化后实际发生 UAV 位置变化 | {0,1} | 实际配置结果 | 长时标 |
| tilde z_u^q(t) | PPO 输出的 UAV 位置更新请求/意图 | {0,1} | PPO 辅助动作 | 长时标 |
| e_u(t) | 11.pdf 原 UAV 二维辅助位移；当前实现不使用 | R^2 | original solver 动作 | 长时标 |
| Delta q_u(t) | 径向缩放后的候选位移 | R^2，范数不超过 V_u^max Delta t | 可行化中间量 | 长时标 |
| q_u^raw(t) | 根据请求和候选位移得到的候选位置 | R^2 | 可行化中间量 | 长时标 |
| q_u^(k)(t) | 第 k 轮 SCA 局部点 | R^2 | SCA 中间量 | 长时标 |
| epsilon_q | 判断实际位置变化的容差 | 正数；论文未给数值 | 固定参数 | 全时域 |
| epsilon_sca | SCA 相邻目标值收敛阈值 | 正数；论文未给数值 | 固定参数 | 全时域 |
| K_max | SCA 最大迭代次数 | 正整数；论文未给数值 | 固定参数 | 全时域 |
| V_u^max | UAV 最大飞行速度 | 正数；论文未给数值 | 固定参数 | 全时域 |
| d_U^min | UAV 之间最小安全距离 | 正数；论文未给数值 | 固定参数 | 全时域 |

### 2.4 任务、匹配和连续资源变量

| 变量 | 含义 | 维度/范围 | 状态/动作 | 时标 |
|---|---|---|---|---|
| a_v(t) | 任务车辆 v 是否产生任务 | {0,1} | 当前请求观测 | 短时标 |
| J_v(t) | 任务 {f_v(t),D_v(t),L_v(t)} | 离散服务类型、正数据量、正 CPU 周期数 | 当前请求观测 | 短时标 |
| f_v(t) | 任务所需服务类型 | F | 当前请求观测 | 短时标 |
| D_v(t) | 任务输入数据量 | 正数；论文未给范围 | 当前请求观测 | 短时标 |
| L_v(t) | 任务 CPU 周期数 | 正数；论文未给范围 | 当前请求观测 | 短时标 |
| N_v | 任务车辆 v 的候选节点集合 | {v} union M union {c} | 模型定义 | 短时标 |
| N_v^fea(t) | 当前任务可行执行节点集合 | Cloud、可用边缘和本地节点的集合 | Matching 输入 | 短时标 |
| mu_t(v) | 任务 v 在时隙 t 的执行节点 | N_v^fea(t) | Matching 结果 | 短时标 |
| x_{v,n}(t) | 任务 v 是否由 n 执行 | {0,1} | 由匹配确定 | 短时标 |
| w_{v,m}(t) | 服务节点 m 给任务 v 的计算资源比例 | [0,1] | KKT 输出 | 短时标 |
| b_{v,m}(t) | 服务节点 m 给任务 v 的带宽比例 | [0,1] | KKT 输出 | 短时标 |
| F_v,F_m,F_c | 任务车辆、边缘节点、云的计算频率 | 正数；论文未给数值 | 固定参数 | 全时域 |
| B_m | 服务节点 m 的可用共享带宽 | 正数；论文未给数值 | 固定参数 | 全时域 |
| p_v | 任务车辆上传功率 | 正数；论文未给数值 | 固定参数 | 全时域 |
| R_c | 蜂窝接入固定等效上传速率 | 正数；论文未给数值 | 固定参数 | 全时域 |

### 2.5 PPO、奖励和统计变量

| 变量 | 含义 | 维度/范围 | 状态/动作 | 时标 |
|---|---|---|---|---|
| s^cfg(t) | 配置状态/observation | 见第 5 节 | PPO 输入状态 | 长时标 |
| a_t^cfg | 当前节点级配置 trigger | MultiBinary(M+U)，见第 6 节 | PPO 动作 | 长时标 |
| r(t) | 即时奖励 | G(t) | 环境反馈 | 当前时隙 |
| G(t) | 实际系统效用 | 任务效用减实际配置成本 | 环境反馈 | 当前时隙 |
| A_hat(t) | GAE 优势估计 | 实数 | PPO 训练量 | rollout |
| R_hat(t) | 价值网络训练目标 | 实数 | PPO 训练量 | rollout |
| gamma | 折扣因子 | 论文默认 0.99，config 可调 | 超参数 | 训练 |
| lambda | GAE 系数 | 论文默认 0.95，config 可调 | 超参数 | 训练 |
| epsilon | PPO clip 系数 | 正数；论文未给数值 | 超参数 | 训练 |
| N_task | 测试期间已到达任务总数 | 非负整数，指标要求 >0 | 统计量 | 测试 |
| T_e | 第 e 个训练回合的时隙数 | 正整数 | 统计量 | 训练 |

Source: 11.pdf p.1-14、p.17-23、p.28-32

## 3. 系统模型和基础公式

### 3.1 位置、时隙和预测使用边界

- 车辆实际位置、RSU 固定位置和 UAV 水平位置分别为 q_v(t)、q_r、q_u(t)。
- UAV 在固定高度 H 飞行，只优化水平位置。
- 配置阶段使用预测位置 hat q_v(t)；任务调度阶段必须使用时隙 t 的实际观测位置 q_v(t)。
- 时隙内通信距离、信道增益和速率按离散位置计算，忽略时隙内部连续位置变化。
- 车辆数量、服务车辆身份和路线在一个优化时域内保持不变；车辆位置、通信状态和任务请求随时隙变化。

Source: 11.pdf p.1-2

### 3.2 服务请求与 Zipf 流行度

服务 f 的程序大小为 s_f。服务流行度按时变 Zipf 分布生成：

    rho_f(t) = [r_f(t)]^(-zeta) / sum_{j in F} [r_j(t)]^(-zeta),  f in F
    sum_{f in F} rho_f(t) = 1

其中 r_f(t) 是服务 f 在时隙 t 的流行度排名，zeta 是 Zipf 参数。系统根据历史请求预测 hat rho_f(t)。

任务类型按当前实际流行度生成：

    Pr{f_v(t)=f | a_v(t)=1} = rho_f(t)

预测流行度只能用于缓存和 UAV 配置，实际任务服务类型必须使用当前时隙观测到的 f_v(t)。

Source: 11.pdf p.2-3、p.13

### 3.3 缓存初始状态与在线更新

可更新节点集合为 M = V_s union R union U。缓存状态和容量约束：

    c_{m,f}(t) in {0,1}
    sum_{f in F} s_f c_{m,f}(t) <= K_m,  m in M

任务车辆本地缓存固定：

    c_{v,f}(t) = c_{v,f}(0),  v in V minus V_s

服务车辆、RSU、UAV 的实际缓存更新触发变量 z_m^ca(t)：

    |c_{m,f}(t) - c_{m,f}(t-1)| <= z_m^ca(t)

- z_m^ca(t)=0：沿用上一时隙缓存。
- z_m^ca(t)=1：可在容量约束下重新确定缓存；下载和加载须在进入时隙 t 的任务服务阶段前完成。
- 初始缓存 c_{m,f}(0) 在优化时域开始前随机生成并满足容量约束；初始状态只用于构造初始配置，不产生在线缓存更新成本或服务加载能耗。

Source: 11.pdf p.2-3

### 3.4 任务和执行约束

任务车辆在每个时隙最多产生一个不可分割任务：

    a_v(t) in {0,1}
    J_v(t) = {f_v(t), D_v(t), L_v(t)}  when a_v(t)=1

可选执行节点：

    N_v = {v} union M union {c}

执行变量：

    x_{v,n}(t) in {0,1}
    sum_{n in N_v} x_{v,n}(t) = a_v(t)

缓存可用性：

    x_{v,v}(t) <= c_{v,f_v(t)}(0)
    x_{v,m}(t) <= c_{m,f_v(t)}(t),  m in M

Cloud c 存储全部服务，因此不受缓存约束。

Source: 11.pdf p.3-4

### 3.5 计算资源和任务计算时延

服务节点计算比例：

    epsilon_w x_{v,m}(t) <= w_{v,m}(t) <= x_{v,m}(t)
    sum_{v in V minus V_s} w_{v,m}(t) <= 1

节点 m 给任务 v 的实际计算频率是 w_{v,m}(t) F_m。本地计算不参与比例分配，边缘计算比例只适用于 M。

各执行方式计算时延和本地计算能耗：

    T^{cmp}_{v,v}(t) = L_v(t) / F_v
    E^{cmp}_{v,v}(t) = kappa_v L_v(t) F_v^2
    T^{cmp}_{v,m}(t) = L_v(t) / (w_{v,m}(t) F_m),  m in M
    T^{cmp}_{v,c}(t) = L_v(t) / F_c

论文只定义任务车辆本地计算能耗和上传能耗，没有定义服务节点计算能耗、云计算能耗或 UAV 计算能耗；工程不得自行补充这些能耗项。

Source: 11.pdf p.4-5

### 3.6 约束 C1-C16

下面是论文 P0 的完整约束编号，代码中应保持含义一致：

    C1: c_{m,f}(t) in {0,1}
    C2: sum_f s_f c_{m,f}(t) <= K_m
    C3: z_m^ca(t) in {0,1}
    C4: |c_{m,f}(t)-c_{m,f}(t-1)| <= z_m^ca(t)
    C5: z_u^q(t) in {0,1}
    C6: ||q_u(t)-q_u(t-1)||_2 <= z_u^q(t) V_u^max Delta t
    C7: q_u(t) in A
    C8: ||q_u(t)-q_u'(t)||_2 >= d_U^min,  u != u'
    C9: x_{v,n}(t) in {0,1}
    C10: sum_{n in N_v} x_{v,n}(t) = a_v(t)
    C11: x_{v,v}(t) <= c_{v,f_v(t)}(0)
    C12: x_{v,m}(t) <= c_{m,f_v(t)}(t)
    C13: epsilon_w x_{v,m}(t) <= w_{v,m}(t) <= x_{v,m}(t)
    C14: sum_v w_{v,m}(t) <= 1
    C15: epsilon_b x_{v,m}(t) <= b_{v,m}(t) <= x_{v,m}(t)
    C16: sum_v b_{v,m}(t) <= 1

其中 C1-C4 管缓存，C5-C8 管 UAV 位置，C9-C12 管执行节点和服务可用性，C13-C16 管边缘计算比例和共享上传带宽比例。本地和云端不参与 C13-C16 的共享资源分配。

Source: 11.pdf p.10-11

## 4. 第四章 Proposed：预测辅助的自适应配置触发与分层联合求解方法

本章是第三章 P0 的在线分层近似求解器，不改变 P0、`G(t)`、C1-C16、通信、
计算、缓存成本或 UAV 移动成本定义，也不声称得到 P0 的全局最优解。参考论文
采用固定周期的大时间尺度缓存和小时间尺度 Off-RA；当前方法不设置固定的
`cache_decision_freq` 或 `uav_decision_freq`，而是由每个基础时隙的预测状态驱动
Cache/UAV 是否重新配置。

### A. 问题分解与总体框架

每个时隙固定执行：

    截至 t-1 的历史车辆/请求
    -> 因果车辆位置预测与服务需求预测
    -> PPO MultiBinary configuration triggers
    -> UAV 有限候选位置优化
    -> 基于最终预测覆盖的一轮顺序协同缓存重配置
    -> 最终 C(t), q_U(t)
    -> 当前实际任务/位置/信道
    -> feasible nodes
    -> transfer-exchange Matching
    -> KKT 计算/带宽分配
    -> Local/SV/RSU/UAV/Cloud 实际执行
    -> Q(t)
    -> G(t)=Q(t)-C^ca(t)-C^q(t)
    -> 标准 SB3 PPO update

PPO 只决定哪些缓存节点允许重配置、哪些 UAV 允许调整位置；被触发后的“配置成
什么”由确定性近似算法决定。UAV 先确定，以便缓存价值使用最终预测覆盖关系；
Cache/UAV 不来回迭代。

### B. 移动性和服务需求预测

配置状态只使用截至 `t-1` 的历史车辆状态和请求历史，得到因果预测位置
`qhat_V(t)`、预测需求/流行度 `rhohat(t)` 以及节点覆盖关系。当前时隙真实请求、
真实位置和即时信道只在配置完成后的任务执行阶段使用。

### C. 基于 PPO 的自适应配置触发

标准 SB3 PPO 的 `MultiBinary(M+U)` 动作只表示配置许可：cache trigger=0
严格保持对应缓存行，cache trigger=1 允许一次完整重配置；UAV trigger=0
严格保持位置，UAV trigger=1 允许一次有限候选位置搜索。请求 trigger 与实际
更新变量分开记录，trigger=1 也可以得到 noop。

### D. 需求感知 UAV 位置自适应调整

对每个被触发 UAV 比较当前位置、预测需求热点和地面边缘覆盖缺口热点；候选经
最大移动距离、区域和 UAV 安全距离约束筛选，并以预测服务收益减第三章已有
UAV 移动成本评分。没有更优可行候选时保持原位，不输出连续 PPO 位移。

### E. 预测驱动的协同服务缓存重配置

在 UAV 位置确定后，按 `SV -> RSU -> UAV` 的稳定顺序，对被触发节点计算预测
边际服务价值并执行一次完整严格 0-1 knapsack。节点之间通过已覆盖服务降低
重复边际价值；这是 `Sequential Cooperative Cache Reconfiguration` 协同近似，
允许空间分离区域保留合理副本，不进行联合 MILP 或多轮 Cache-UAV 交替。

### F. 基于迁移—交换匹配的执行节点选择

固定最终 `C(t)` 和 `q_U(t)` 后，用当前真实任务、真实位置和真实信道构造可行
执行节点；以 Cloud 可行初始化为基础，复用 transfer-exchange Matching 选择
Local/SV/RSU/UAV/Cloud 执行节点。

### G. 基于 KKT 的通信与计算资源分配

对 Matching 固定的边缘任务复用 C13-C16 对应的计算比例和共享带宽 KKT；本地
和 Cloud 继续使用第三章固定资源模型，不新增 GA、Hungarian、Cloud 后处理或
其它资源分配器。

### H. 总体算法与复杂度

每个时隙只执行一次 `预测 -> PPO trigger -> UAV 候选 -> 顺序 knapsack ->
feasible nodes -> Matching -> KKT -> execution -> G`。若 `M` 为缓存节点数、
`F` 为服务数、`K_m` 为整数化缓存容量、`H` 为预测时域，缓存 DP 的主开销为
各触发节点的 `O(F K_m)`，UAV 候选比较为有限候选数乘以预测效用评价，Matching
和 KKT 沿用现有确定性实现。整体是状态驱动的在线分层近似，而不是固定周期
双时间尺度全局优化。

## 5. 当前 PPO observation

    s_cfg(t) = { C(t-1), q_U(t-1), qhat_V(t), rhohat(t), t/T }

- 不读取 `task_arrivals[t]`、`task_service_types[t]`、当前真实 `q_V(t)` 或未来 trace。
- `qhat_V(t)` 与 `rhohat(t)` 只由截至 t-1 的历史产生；`t/T` 表示有限时域进度。
- Cache 保持 0/1；UAV 和预测车辆位置按配置区域归一化；预测流行度与进度位于
  `[0,1]`。展平维度由 M、U、V、F 动态构造，不固定为 8 维。
- `J_keep/J_update`、候选成本、变化比例不进入当前 Proposed observation。

## 6. 当前 PPO action

    a_cfg(t) = [ztilde_ca_1,...,ztilde_ca_M,ztilde_q_1,...,ztilde_q_U]
    a_cfg(t) in MultiBinary(M+U)

固定索引顺序为：SV cache triggers、RSU cache triggers、UAV cache triggers、
UAV position triggers；同类型按稳定 node id/行号排序。工程显式保存
`cache_node_ids`、`cache_node_types`、`uav_ids` 和 `action_layout`，不依赖字典顺序。

## 7. 当前 Cache 求解

- `ztilde_ca_m=0`：严格保持 `C_m(t)=C_m(t-1)`。
- `ztilde_ca_m=1`：根据预测位置、预测需求、覆盖、其它节点当前已提供服务、
  服务大小和容量计算 `v_hat[m,f]`，然后调用严格 0-1 knapsack DP：

      max sum_f v_hat[m,f] c[m,f]
      s.t. sum_f s_f c[m,f] <= K_m, c[m,f] in {0,1}

一次 trigger 对应一次完整缓存配置重算，不是逐时隙 1-swap。节点只按
SV→RSU→UAV 的固定顺序协同更新一轮；重叠区域已有服务会降低边际价值，空间
分离节点仍可缓存相同服务。这是 sequential cooperative cache reconfiguration，
不声称联合缓存全局最优。

实际更新变量始终为：

    z_m^ca(t) = I{C_m(t) != C_m(t-1)}

trigger=1 但背包结果相同时，实际更新和缓存成本均为 0。成本仍只计算
`C_old -> C_final`，删除不计 loading energy，新增才计 loading energy。

## 8. 当前 UAV 求解

- `ztilde_q_u=0`：严格保持 `q_u(t)=q_u(t-1)`。
- `ztilde_q_u=1`：比较当前位置、预测需求热点和地面服务覆盖缺口热点。
- 候选必须满足最大移动距离、区域和 UAV 安全距离；不可行候选直接丢弃。
- 评价只使用因果预测服务收益减 11.pdf 定义的实际 UAV movement cost，不读取
  当前真实任务，不增加新 penalty。

这是 finite candidate approximate UAV positioning，不是连续 UAV 全局最优。
若没有更优可行候选则保持原位；只有真实位移超过 `epsilon_q` 才令
`z_u^q(t)=1` 并产生成本。

## Appendix A. 11.pdf original solver design（历史记录，不是当前工程主链）

### A.1 原求解流程

    SUMO 数据
    -> 请求/预测
    -> 配置状态
    -> PPO 辅助动作
    -> cache
    -> UAV
    -> feasibility
    -> matching
    -> KKT
    -> cloud fallback / 云端实际执行
    -> final execution
    -> delay/energy
    -> reward
    -> PPO rollout/update

每个时隙的具体含义：

1. 从 SUMO 获取截至 t-1 的车辆位置和速度，获取历史任务请求记录。
2. 用卡尔曼滤波得到 hat q_V(t)，用历史请求统计和固定参数 GRU 得到 hat rho(t)。
3. 构造 s^cfg(t) = {c(t-1), q_U(t-1), hat q_V(t), hat rho(t)}。
4. 条件因子化 PPO 采样缓存更新意图、缓存服务评分、UAV 更新意图和二维辅助位移，并保存原始联合对数概率。
5. 对每个被触发的缓存节点执行严格 0-1 背包映射，得到满足容量约束的实际缓存状态和实际缓存更新变量。
6. 对 UAV 辅助位移做径向缩放生成候选位置；若联合候选位置违反原始约束，执行 SCA 位置可行化。
7. 对缓存状态和 UAV 位置进行确定性 feasibility 确认。UAV 最终仍须复检原始 C6-C8；失败则全部回退上一时隙位置。
8. 用时隙 t 的实际车辆位置、实际任务和实际信道构造可行执行节点集合，执行任务转移-交换 matching。
9. 对 matching 固定的边缘任务，使用 KKT 求解计算比例和带宽比例；本地和云任务使用论文固定资源模型。
10. Cloud fallback 的论文定义是 matching 开始时的可行初始化 mu_t^(0)(v)=c，因此所有任务先保证有云执行方案；随后只有可行且能提高效用的转移或交换才把任务移离云端。KKT 后保留在云端的任务继续使用云端固定蜂窝传输和固定云计算，不得新增“二次云端回退”理论机制。
11. 由最终 matching 和资源分配得到实际执行节点、实际完成时延和实际车辆能耗。
12. 用实际缓存状态、实际 UAV 位置、实际 matching、实际 KKT 资源分配和实际成本计算 G(t)，并令 r(t)=G(t)。
13. rollout 保存 PPO 原始辅助动作、原始联合 log-prob、奖励和下一状态；轨迹末端按 GAE 规则 bootstrap 或置零，随后进行 PPO policy/value update。

### A.2 原时序边界

- 配置使用预测信息，但不使用当前完整任务请求、当前实际车辆位置和即时信道状态。
- 任务调度使用当前实际观测，不使用预测位置替代实际位置。
- PPO 的动作是辅助动作；可行化后的缓存状态、UAV 位置和实际更新变量不是重新定义的 PPO 动作。
- Cloud 是 matching 的可行初始化和最终执行节点之一，不是边缘 KKT 节点。

Source: 11.pdf p.1-2、p.12、p.21-23

### A.3 原配置状态/observation

论文真正定义的配置状态：

    s^cfg(t) = { c(t-1), q_U(t-1), hat q_V(t), hat rho(t) }

其中：

- c(t-1) 只包含服务车辆、RSU、UAV 的可更新缓存状态；不包含任务车辆保持不变的本地缓存。
- q_U(t-1) 是上一时隙 UAV 实际水平位置。
- hat q_V(t) 包含服务车辆和任务车辆的预测位置。
- hat rho(t) 是预测服务流行度向量。
- 当前实际任务请求、当前实际车辆位置和即时信道状态不加入配置状态。

配置状态不是任务调度状态；任务调度阶段直接使用时隙 t 的实际位置、实际请求和即时信道。

#### A.3.1 原预测器

#### 车辆位置预测

    xi_v(t) = [x_v(t), y_v(t), v_v^x(t), v_v^y(t)]^T
    hat xi_v(t|t-1) = A hat xi_v(t-1|t-1)
    hat q_v(t) = H_pos hat xi_v(t|t-1)

SUMO 在 t-1 的位置和速度用于观测校正，卡尔曼滤波递推到 t。任务调度阶段使用实际 q_v(t)。

#### 服务流行度预测

    N_f(tau) = sum_{v in V minus V_s} a_v(tau) I{f_v(tau)=f}
    tilde rho_f(tau) = N_f(tau) / sum_{j in F} N_j(tau)

如果时隙 tau 没有任务请求，使用 tilde rho(tau)=tilde rho(tau-1)。最近 L_rho 个历史观测输入轻量 GRU：

    h_rho(t) = GRU(tilde rho(t-L_rho), ..., tilde rho(t-1))
    hat rho(t) = Softmax(W_rho h_rho(t) + b_rho)

GRU 在 PPO 训练前预训练，PPO 训练和在线执行期间参数固定。L_rho、GRU 网络规模和训练超参数在 11.pdf 中未给出。

Source: 11.pdf p.12-14

### A.4 原 PPO 辅助动作、mask 和联合 log-prob

#### A.4.1 原辅助动作空间

    tilde a^cfg(t) = { tilde z_m^ca(t), y_m(t), tilde z_u^q(t), e_u(t) }

- tilde z_m^ca(t) in {0,1}：节点 m 的缓存更新请求。
- y_m(t) = {y_{m,f}(t)}：缓存服务评分，触发时 0 < y_{m,f}(t) < 1。
- tilde z_u^q(t) in {0,1}：UAV u 的位置更新请求。
- e_u(t) in R^2：UAV 二维辅助位移参数。

这些是 PPO 采样的辅助动作，不是 P0 中直接优化的实际变量。

#### A.4.2 原条件因子化联合策略

    pi_theta(tilde a^cfg(t) | s^cfg(t))
    = product_{m in M} pi_theta^ca(tilde z_m^ca(t) | s^cfg(t))
      * [pi_theta^y(y_m(t) | s^cfg(t))]^(tilde z_m^ca(t))
      * product_{u in U} pi_theta^q(tilde z_u^q(t) | s^cfg(t))
      * [pi_theta^e(e_u(t) | s^cfg(t))]^(tilde z_u^q(t))

#### A.4.3 原动作 mask、log-prob 和 entropy

逐样本联合 log-prob 使用触发 mask：

    log pi_theta(tilde a^cfg | s^cfg)
    = sum_{m in M} [
        log pi_theta^ca(tilde z_m^ca | s^cfg)
        + tilde z_m^ca log pi_theta^y(y_m | s^cfg)
      ]
      + sum_{u in U} [
        log pi_theta^q(tilde z_u^q | s^cfg)
        + tilde z_u^q log pi_theta^e(e_u | s^cfg)
      ]

- 当 tilde z_m^ca=0 时，y_m 不采样、不计入联合 log-prob、不计入策略梯度和策略 entropy。
- 当 tilde z_u^q=0 时，e_u 不采样、不计入联合 log-prob、不计入策略梯度和策略 entropy。
- trigger Bernoulli 头始终计入联合 log-prob。
- 论文没有规定连续头必须采用 Gaussian、Beta 或其他具体分布；不得擅自把分布族写成论文理论。
- 标准 PPO 对比方法明确不使用这种条件 mask：无论 trigger 是否为 0，所有连续头都参与采样、联合 log-prob 和 entropy。

Source: 11.pdf p.14-15、p.26

### A.5 原缓存映射

#### A.5.1 原缓存状态转移

当 tilde z_m^ca(t)=0 时，直接令 c_{m,f}(t)=c_{m,f}(t-1)。当 tilde z_m^ca(t)=1 时，使用 PPO 对每个服务输出 y_{m,f}(t)，求解严格 0-1 背包：

    max_{c_{m,f}(t)} sum_{f in F} y_{m,f}(t) c_{m,f}(t)
    s.t. sum_{f in F} s_f c_{m,f}(t) <= K_m
         c_{m,f}(t) in {0,1}

服务大小和缓存容量统一尺度整数化后使用动态规划求解。这里是严格 0-1 背包：每个服务只能选或不选，不是分数背包、连续缓存、按分数排序截断或额外的 counterfactual 选择器。

#### A.5.2 实际触发和更新成本边界

    z_m^ca(t) = I{ sum_{f in F} |c_{m,f}(t)-c_{m,f}(t-1)| > 0 }
    z_m^ca(t) <= tilde z_m^ca(t)

如果映射后的缓存和上一时隙相同，强制 z_m^ca(t)=0，不产生固定更新成本和服务加载能耗。即使 PPO 请求了更新，也不能把没有实际变化的配置记成一次实际更新。

#### A.5.3 缓存更新能耗和成本

    E_m^ca(t) = xi_m^ca sum_{f in F} s_f [c_{m,f}(t)-c_{m,f}(t-1)]^+
    [x]^+ = max{x,0}
    C^ca(t) = sum_{m in M} [ eta_m^ca z_m^ca(t) + lambda^ca E_m^ca(t) ]

删除已有服务不计加载能耗；任务车辆本地缓存不产生在线缓存更新成本。

Source: 11.pdf p.2-3、p.8-9、p.15

### A.6 原 UAV 连续位移/SCA 求解

#### A.6.1 位置更新和移动限制

    z_u^q(t) in {0,1}
    ||q_u(t)-q_u(t-1)||_2 <= z_u^q(t) V_u^max Delta t
    q_u(t) in A
    ||q_u(t)-q_u'(t)||_2 >= d_U^min,  u != u'

#### A.6.2 从 PPO 位移到候选位置

PPO 输出的 e_u(t) 不是最终位移。先做径向缩放：

    Delta q_u(t) = e_u(t) / max{1, ||e_u(t)||_2} * V_u^max Delta t
    ||Delta q_u(t)||_2 <= V_u^max Delta t

候选位置：

    q_u^raw(t) = q_u(t-1),                    if tilde z_u^q(t)=0
    q_u^raw(t) = q_u(t-1) + Delta q_u(t),     if tilde z_u^q(t)=1

若所有候选位置联合满足原始 C6-C8，直接采用 q_u(t)=q_u^raw(t)。

#### A.6.3 SCA 位置修正

若联合候选位置违反原始约束，以 q_u^(k)(t) 为第 k 轮局部点，定义：

    delta_{u,u'}^(k)(t) = q_u^(k)(t) - q_u'^(k)(t),  u < u'

把原安全距离约束替换为平方距离函数的一阶下界：

    2 delta_{u,u'}^(k)(t)^T (q_u(t)-q_u'(t))
    - ||delta_{u,u'}^(k)(t)||_2^2 >= (d_U^min)^2

第 k 轮凸子问题的目标和约束：

    min_{q_u(t)} sum_{u in U} tilde z_u^q(t) ||q_u(t)-q_u^raw(t)||_2^2
    s.t. ||q_u(t)-q_u(t-1)||_2 <= tilde z_u^q(t) V_u^max Delta t
         q_u(t) in A
         上述所有一阶安全距离近似约束

初始化和更新：

    q_u^(0)(t) = q_u(t-1)
    q_u^(k+1)(t) = q_u^(k)* (t)
    J_UAV^(k)(t) = sum_{u in U} tilde z_u^q(t)
                         ||q_u^(k)*(t)-q_u^raw(t)||_2^2

当 |J_UAV^(k+1)(t)-J_UAV^(k)(t)| <= epsilon_sca 时停止。epsilon_sca 和 K_max 只在论文中以符号出现，没有数值。

#### A.6.4 失败回退和实际更新变量

以下任一情况判定位置修正失败：凸子问题无可行解、数值求解失败、达到 K_max 仍未满足停止条件，或结果复检不满足原始 C6-C8。失败时所有 UAV 回退到上一时隙位置：

    q_u(t) = q_u(t-1),  u in U

最终实际更新变量：

    z_u^q(t) = I{ ||q_u(t)-q_u(t-1)||_2 > epsilon_q }
    z_u^q(t) <= tilde z_u^q(t)

即使 tilde z_u^q(t)=1，如果修正或失败回退后最终位置不变，也必须令 z_u^q(t)=0，不产生固定位置更新成本和移动能耗。

#### A.6.5 UAV 移动成本

    E_u^mov(t) = xi_u^mov ||q_u(t)-q_u(t-1)||_2
    C^q(t) = sum_{u in U} [ eta_u^q z_u^q(t) + lambda^mov E_u^mov(t) ]

C^q(t) 必须使用可行化后的实际位置和实际 z_u^q(t)，不得使用 PPO 的更新意图代替实际更新。

Source: 11.pdf p.9-11、p.15-17

## 9. 通信模型

### 9.1 地面链路

任务车辆 v 到地面服务节点 m in V_s union R 的距离：

    d_{v,m}(t) = ||q_v(t)-q_m(t)||_2

当 m in R 时，q_m(t)=q_r。地面链路平均信道功率增益：

    h_{v,m}(t) = beta_g [d_{v,m}(t)]^(-alpha_g)

该模型统一用于任务车辆到服务车辆的 V2V 和到 RSU 的 V2I。

### 9.2 UAV V2U 链路

    d^h_{v,u}(t) = ||q_v(t)-q_u(t)||_2
    d_{v,u}(t) = sqrt( [d^h_{v,u}(t)]^2 + H^2 )
    theta_{v,u}(t) = 180/pi * arcsin( H / d_{v,u}(t) )

视距概率、非视距概率和状态信道增益：

    P^L_{v,u}(t) = 1 / [1 + eta_1 exp(-eta_2(theta_{v,u}(t)-eta_1))]
    P^N_{v,u}(t) = 1 - P^L_{v,u}(t)
    h^s_{v,u}(t) = beta_s [d_{v,u}(t)]^(-alpha_s),  s in {L,N}
    h_{v,u}(t) = P^L_{v,u}(t) h^L_{v,u}(t)
                  + P^N_{v,u}(t) h^N_{v,u}(t)

其中 eta_1、eta_2 是环境参数，beta_s、alpha_s 是传播状态 s 下的参考功率增益和路径损耗指数。具体数值未在 PDF 中给出。

### 9.3 共享带宽、速率和上传能耗

仅对 m in M 定义共享带宽比例：

    epsilon_b x_{v,m}(t) <= b_{v,m}(t) <= x_{v,m}(t)
    sum_{v in V minus V_s} b_{v,m}(t) <= 1

实际上传带宽为 b_{v,m}(t) B_m，传输速率：

    R_{v,m}(t) = b_{v,m}(t) B_m
                 log_2(1 + p_v h_{v,m}(t)/sigma^2),  m in M

- 地面节点使用地面链路增益，UAV 使用 V2U 平均信道增益。
- 本地执行不定义 b_{v,v}(t)，不上传任务数据。
- Cloud 使用独立蜂窝链路，不参与 M 的共享带宽分配。
- 不同任务车辆采用正交频谱接入，忽略同频干扰。

边缘上传：

    T^tr_{v,m}(t) = D_v(t) / R_{v,m}(t)
    E^tr_{v,m}(t) = p_v T^tr_{v,m}(t) = p_v D_v(t)/R_{v,m}(t)

云端上传：

    T^tr_{v,c}(t) = psi_c D_v(t) / R_c
    E^tr_{v,c}(t) = p_v D_v(t) / R_c

psi_c >= 1 只扩展云端传输时延，不乘到车辆无线发射能耗上。任务结果数据量较小，忽略下行返回传输时延和车辆能耗。

Source: 11.pdf p.5-8

## 10. 任务执行模型

### 10.1 Local

可行条件：x_{v,v}(t)=1 仅当 c_{v,f_v(t)}(0)=1。本地使用任务车辆完整计算频率 F_v，不占共享带宽：

    T_{v,v}(t) = L_v(t)/F_v
    E_{v,v}^veh(t) = kappa_v L_v(t) F_v^2

### 10.2 Service Vehicle、RSU、UAV

可行条件：x_{v,m}(t)=1 仅当 c_{m,f_v(t)}(t)=1，m in M。任务上传到服务节点后使用共享计算和共享带宽：

    T_{v,m}(t) = D_v(t)/R_{v,m}(t)
                  + L_v(t)/(w_{v,m}(t) F_m)
    E_{v,m}^veh(t) = p_v D_v(t)/R_{v,m}(t)

这三个节点的差异只通过 F_m、B_m、位置、信道和缓存状态体现。论文没有定义服务节点计算能耗。

### 10.3 Cloud

可行条件：Cloud 存储全部服务，始终可执行，不受缓存和边缘共享资源约束。使用固定 R_c、F_c：

    T_{v,c}(t) = psi_c D_v(t)/R_c + L_v(t)/F_c
    E_{v,c}^veh(t) = p_v D_v(t)/R_c

### 10.4 实际任务完成结果

    T_v(t) = x_{v,v}(t) T_{v,v}(t)
             + sum_{m in M} x_{v,m}(t) T_{v,m}(t)
             + x_{v,c}(t) T_{v,c}(t)

    E_v^veh(t) = x_{v,v}(t) E_{v,v}^veh(t)
                 + sum_{m in M} x_{v,m}(t) E_{v,m}^veh(t)
                 + x_{v,c}(t) E_{v,c}^veh(t)

当 a_v(t)=0 时，令 T_v(t)=0、E_v^veh(t)=0。这里只统计任务车辆本地计算或上传产生的车辆能耗。

Source: 11.pdf p.3-8、p.18-22

## 11. Matching：任务转移-交换，不得替换算法

### 11.1 可行节点、匹配和节点任务集合

当前任务车辆集合：

    V_t = {v in V minus V_s | a_v(t)=1}

当前任务的可行执行节点集合：

    N_v^fea(t)
    = {c}
      union {m in M | c_{m,f_v(t)}(t)=1}
      union {v | c_{v,f_v(t)}(0)=1}

匹配结果和执行变量：

    mu_t(v) in N_v^fea(t)
    x_{v,n}(t) = I{mu_t(v)=n}
    V_m(mu_t) = {v in V_t | mu_t(v)=m}

匹配可行性：

    epsilon_w |V_m(mu_t)| <= 1
    epsilon_b |V_m(mu_t)| <= 1,  m in M

### 11.2 候选匹配评价

对任意可行匹配，先用 KKT 求给定匹配下的连续资源分配，再定义：

    J_t(mu_t) = Q(t; x(t), w*(mu_t,t), b*(mu_t,t))

缓存更新成本和 UAV 位置更新成本在同一时隙的候选匹配之间不变，因此 matching 阶段只用任务服务效用 Q(t) 评价候选匹配，不把固定不变的配置成本重复加入比较。

### 11.3 初始 cloud fallback

    mu_t^(0)(v) = c,  v in V_t

因为 Cloud 可以执行所有任务，初始匹配始终可行。Cloud 任务使用固定云资源，不进入边缘节点 M 的 KKT 资源分配。

### 11.4 单任务 transfer

对任务 v，候选节点：

    n in N_v^fea(t) minus {mu_t(v)}

把任务 v 从当前节点移到 n 得到 mu_t^(v->n)。只有同时满足匹配可行性且：

    J_t(mu_t^(v->n)) > J_t(mu_t)

才接受转移。

### 11.5 双任务 exchange

对两个任务 v,j in V_t，只有当交换后彼此仍属于对方的可行节点集合时才允许交换：

    mu_t(j) in N_v^fea(t)
    mu_t(v) in N_j^fea(t)

交换得到 mu_t^(v<->j)，若可行且：

    J_t(mu_t^(v<->j)) > J_t(mu_t)

才接受。

### 11.6 终止和性质

- 每轮选择效用增益最大的可行 transfer 或 exchange。
- 不存在继续提高 J_t 的操作时终止。
- 由于每次接受都严格提高效用且可行匹配数量有限，有限次迭代后终止。
- 无改进操作终止时，得到 transfer-exchange 稳定状态，但不保证全局最优匹配。
- 如果达到允许的最大匹配更新次数而停止，只保证匹配可行，不保证达到稳定状态。
- 这是论文定义的任务转移-交换局部搜索，不得替换为 Hungarian、stable marriage、max-weight matching 或其他未定义匹配机制。

Source: 11.pdf p.17-19、p.21-24

## 12. KKT：计算分配、带宽分配、上下界和二分

KKT 只在匹配 mu_t 固定后对服务节点 m in M 求解；本地和云端资源固定。截断算子：

    [z]_a^b = min{b, max{a,z}}

### 12.1 计算资源分配

对 v in V_m(mu_t)：

    A_{v,m}(t) = omega_T L_v(t) / [T_v^ref(t) F_m]

服务节点 m 的子问题：

    min_{w_{v,m}(t)} sum_{v in V_m(mu_t)} A_{v,m}(t)/w_{v,m}(t)
    s.t. sum_{v in V_m(mu_t)} w_{v,m}(t) <= 1
         epsilon_w <= w_{v,m}(t) <= 1

令 lambda_m(t) 为计算容量约束的拉格朗日乘子，KKT 解：

    w^*_{v,m}(t) = [ sqrt(A_{v,m}(t)/lambda_m^*(t)) ]_{epsilon_w}^1

当 V_m(mu_t) 非空时：

    sum_{v in V_m(mu_t)} w^*_{v,m}(t) = 1

通过二分搜索得到 lambda_m^*(t)。节点没有任务时不分配计算资源；未由节点 m 执行的任务令 w^*_{v,m}(t)=0。上下界 epsilon_w 和 1 必须保留。

### 12.2 带宽资源分配

对 v in V_m(mu_t)：

    Gamma_{v,m}(t)
    = D_v(t) / [ B_m log_2(1+p_v h_{v,m}(t)/sigma^2) ]
      * [ omega_T/T_v^ref(t) + omega_E p_v/E_v^ref(t) ]

服务节点 m 的子问题：

    min_{b_{v,m}(t)} sum_{v in V_m(mu_t)} Gamma_{v,m}(t)/b_{v,m}(t)
    s.t. sum_{v in V_m(mu_t)} b_{v,m}(t) <= 1
         epsilon_b <= b_{v,m}(t) <= 1

令 nu_m(t) 为带宽容量约束的拉格朗日乘子，KKT 解：

    b^*_{v,m}(t) = [ sqrt(Gamma_{v,m}(t)/nu_m^*(t)) ]_{epsilon_b}^1

当 V_m(mu_t) 非空时：

    sum_{v in V_m(mu_t)} b^*_{v,m}(t) = 1

通过二分搜索得到 nu_m^*(t)。节点没有任务时不分配带宽；未由节点 m 执行的任务令 b^*_{v,m}(t)=0。上下界 epsilon_b 和 1 必须保留。

### 12.3 候选操作中的重算范围

在 matching 的 transfer 或 exchange 中，只重新求解任务集合发生变化的服务节点的计算和带宽分配，其他节点沿用当前候选评价所需的分配结果。KKT 子问题在匹配固定时是凸优化，KKT 可得到相应连续资源子问题的全局最优解；这不等于原混合整数序贯问题的全局最优。

Source: 11.pdf p.19-21、p.24

## 13. 系统 Utility、实际 Reward 和成本

### 13.1 单任务服务效用

    Q_v(t) = a_v(t) * [
        omega_T (1 - T_v(t)/T_v^ref(t))
        + omega_E (1 - E_v^veh(t)/E_v^ref(t))
      ]

    omega_T + omega_E = 1
    omega_T, omega_E in [0,1]

超过归一化参考值时对应效用项允许为负。时隙任务服务效用：

    Q(t) = sum_{v in V minus V_s} Q_v(t)

服务车辆不产生任务，不计入该和。

### 13.2 系统效用

    G(t) = Q(t) - C^ca(t) - C^q(t)

P0 目标：

    max E[ (1/T) sum_{t in T} G(t) ]

### 13.3 PPO 即时奖励

配置动作完成可行化、任务 matching 和 KKT 资源分配完成后：

    r(t) = G(t)

Reward 使用最终实际执行结果，而不是 PPO 的预测配置意图、未经可行化的候选位置或未经 matching/KKT 的中间结果。由于 G(t) 已包含实际缓存更新成本和实际 UAV 位置更新成本，不再增加重复惩罚项。

Source: 11.pdf p.8-10、p.14、p.22、p.31

## 14. 当前 PPO 训练规范

当前实现直接复用 `stable_baselines3.PPO("MlpPolicy", ...)`、
`gymnasium.spaces.MultiBinary`、`gymnasium.spaces.Box` 和 SB3 `VecNormalize`。
项目 adapter 只负责构造因果 observation、拆分节点级 trigger、调用确定性
Cache/UAV 配置器并返回原始 `G(t)`；不自定义 Bernoulli 分布，不重写 PPO loss、
GAE、clip 或 entropy。

保持已验证参数：`gamma=0.99`、`gae_lambda=0.95`、`learning_rate=3e-4`、
`n_steps=256`、`batch_size=64`、`n_epochs=10`、`clip_range=0.2`、
`ent_coef=0`、`vf_coef=0.5`、`max_grad_norm=0.5`。外部可继续使用
`norm_obs=true`、`norm_reward=true`、`clip_obs=10`。

`gamma<1` 只是 PPO 的训练代理目标；模型选择和正式性能报告仍使用未折扣原始
平均 `G(t)`，不使用 normalized reward 替代论文指标。

## Appendix B. 11.pdf original PPO training design（历史记录）

### B.1 rollout 保存内容

每个时隙保存：

- s^cfg(t)；
- PPO 采样的原始辅助动作 tilde a^cfg(t)；
- 该原始辅助动作对应的联合 log-prob；
- 实际 reward r(t)=G(t)；
- 下一时隙配置状态。

不把可行化后的实际缓存状态、实际 UAV 位置或实际更新变量重新当成策略网络动作。PPO ratio 必须使用采样时原始辅助动作及其原始 log-prob。

### B.2 bootstrap、GAE、returns

若轨迹在完整优化时域结束前截断：

    V_end = V_phi(s^cfg(tau+1))

若到达真实终止状态：

    V_end = 0

对 t < tau：

    delta(t) = r(t) + gamma V_phi(s^cfg(t+1)) - V_phi(s^cfg(t))

对轨迹最后时隙：

    delta(tau) = r(tau) + gamma V_end - V_phi(s^cfg(tau))

反向计算 GAE：

    A_hat(t) = delta(t) + gamma lambda A_hat(t+1)
    A_hat(tau+1) = 0

价值网络训练目标：

    R_hat(t) = A_hat(t) + V_phi(s^cfg(t))

论文默认 gamma=0.99、lambda=0.95。二者是 config 可调超参数，不是禁止修改的理论常数。

### B.3 PPO ratio 和 clip

    chi_t(theta)
    = pi_theta(tilde a^cfg(t) | s^cfg(t))
      / pi_theta_old(tilde a^cfg(t) | s^cfg(t))

    L^clip(theta)
    = E_t[ min(
        chi_t(theta) A_hat(t),
        clip(chi_t(theta), 1-epsilon, 1+epsilon) A_hat(t)
      ) ]

在 11.pdf 原设计中，这里的 pi 是 Appendix A.4 条件 mask 后的联合策略；该句不描述当前 MultiBinary 实现。

### B.4 value、entropy 和论文未给出的训练细节

- value 网络以 R_hat(t) 为训练目标，根据 R_hat(t) 与 value 输出的误差更新。
- 11.pdf 原设计的 entropy 按 Appendix A.4 的条件因子化动作头计算；当前实现直接使用 SB3 MultiBinary entropy。
- PDF 没有给出独立的 value clipping 公式。
- PDF 没有给出 entropy coefficient、value loss coefficient、policy/value loss 的完整加权总损失公式。
- PDF 没有给出 clip 系数 epsilon 的数值。
- PDF 没有给出 action distribution、网络层数/宽度、学习率、优化器、batch size、PPO epoch 数、advantage normalization 或 minibatch 细节。
- 这些内容只能作为工程 config，不能声称由 11.pdf 定义。
- 当 gamma < 1 时，折扣回报仅作为 PPO 训练代理目标；算法正式性能仍按原始平均系统效用 G(t) 评价。
- GRU 和 PPO 参数在在线执行阶段固定；卡尔曼滤波继续用每个时隙获得的车辆观测递推校正和位置预测。

Source: 11.pdf p.17、p.21-23、p.26、p.31-32

## 15. 论文明确没有定义的机制

以下机制经 11.pdf 全文核对后不作为新工程默认实现：

| 机制 | 结论 |
|---|---|
| recurrent PPO | 11.pdf 未定义。PDF 中的 GRU 仅是服务流行度预测器，且参数固定，不是 PPO policy 的 recurrent memory。 |
| candidate counterfactual credit | 11.pdf 未定义。论文只保存 PPO 原始辅助动作和最终实际 reward，没有候选动作反事实信用分配。 |
| trigger-Q | 11.pdf 未定义。没有为 trigger 单独定义 Q 值。 |
| candidate-Q | 11.pdf 未定义。没有为缓存候选或 UAV 候选定义 Q 值。 |
| KL early stopping | 11.pdf 未定义。不能默认加入 KL 阈值停止。 |
| Hungarian matching | 不是论文方法，不得替换 task transfer-exchange matching。 |
| stable marriage matching | 不是论文方法，不得替换 task transfer-exchange matching。 |
| max-weight matching 作为替代 | 不是论文方法。论文使用每次候选 transfer/exchange 后重算效用并做局部改进。 |
| recurrent hidden state in PPO | 11.pdf 未定义。不得把预测 GRU 隐状态加入 PPO observation。 |
| post-KKT cloud fallback branch | 11.pdf 未定义。云端兜底已经作为 matching 初始方案 mu_t^(0)(v)=c 定义。 |
| 任务队列、跨时隙未完成任务、部分卸载 | 11.pdf 未定义。每个时隙最多一个不可分割任务，未定义排队或拆分执行。 |
| 服务节点/云/UAV 计算能耗 | 11.pdf 未定义。只计任务车辆本地计算和上传能耗，以及缓存更新、UAV 移动成本。 |
| 同频干扰、共享云端回传资源 | 11.pdf 未定义为优化项；不同任务车辆正交接入，云端传输不占边缘共享带宽。 |
| 具体求解器和求解器参数 | 11.pdf 未定义。CVXPY、某种 MILP solver、SCA solver 或具体二分容差不能写成论文指定实现。 |

### 15.1 论文只给符号、未给数值的内容

以下内容在 PDF 中没有正式数值表，不能自行补齐为“论文参数”：

- L_x、L_y、Delta t、H；
- 服务大小范围、缓存容量、计算频率、带宽、上传功率、噪声、信道参数；
- omega_T、omega_E、参考时延和参考能耗；
- epsilon_q、epsilon_sca；epsilon_w 和 epsilon_b 的数值不在 11.pdf 正文中，
  当前仿真参数已在下方“参数闭合”小节明确给出；
- V_u^max、d_U^min、K_max、L_rho；
- GRU/PPO 网络规模、学习率、batch size、训练预算、验证频率和随机种子；
- epsilon、entropy/value loss 系数和其他 PPO 工程超参数。

PDF 多处写有“后续参数表”，但当前 37 页文件中没有实际参数表；这些值均标记为“11.pdf未定义”。

### 15.2 当前资源下界参数闭合

`11.pdf` 只定义 `epsilon_w`、`epsilon_b` 为 C13/C15 中的严格正资源比例
下界，没有给出实验数值。当前项目将其作为明确的仿真参数固定为：

```text
epsilon_w = epsilon_compute = 0.01
epsilon_b = epsilon_bandwidth = 0.01
```

该数值属于当前项目的 simulation parameter closure，不宣称为 `11.pdf` 正文
已经给出的数值；正式代码从 `config.yaml` 的
`resource_allocation.epsilon_compute` 和
`resource_allocation.epsilon_bandwidth` 读取。缺少任一配置值时，正式环境
直接报错，不根据任务数量生成隐式默认值。

### 15.3 Cache/UAV 成本参数状态

当前 `config.yaml` 中的 `xi_u^mov`、`lambda^mov`、`eta_u^q`、Cache loading
energy、Cache fixed update cost 和 `lambda^ca` 保持现有值。`11.pdf` 给出
这些参数的符号和成本公式，但没有给出对应的正式数值表，因此本项目暂时
只记录为 `CURRENT CONFIG / SOURCE STATUS`，不把当前工程值写成论文原始数值。

## 16. 论文正式输出指标和统计分母

测试阶段包含 T_te 个时隙，任务总数：

    N_task = sum_{t=1}^{T_te} sum_{v in V minus V_s} a_v(t)

所有按任务统计的指标要求 N_task > 0。不要用总时隙数、任务车辆数或含无任务时隙的数量替代该分母。

### 16.1 Episode Reward

论文正式训练输出使用每个训练回合的时隙平均奖励：

    EpisodeReward_e = r_bar_e = (1/T_e) sum_{t=1}^{T_e} r_e(t)

Episode Reward 应按论文定义记录为“回合时隙平均奖励”，不是未经说明的累计 reward。可另外记录原始累计值作为工程诊断，但不得把它替代论文正式指标。

### 16.2 系统效用、时延、车辆能耗

平均系统效用按时隙统计：

    G_bar = (1/T_te) sum_{t=1}^{T_te} G(t)

平均任务完成时延和平均车辆能耗按实际到达任务统计：

    T_bar = [sum_t sum_v a_v(t) T_v(t)] / N_task
    E_bar = [sum_t sum_v a_v(t) E_v^veh(t)] / N_task

### 16.3 缓存服务可用率

任务所需服务是否在任务车辆本地或至少一个边缘节点中可用：

    h_v(t) = I{
        c_{v,f_v(t)}(0) + sum_{m in M} c_{m,f_v(t)}(t) >= 1
    }
    H_bar = [sum_t sum_v a_v(t) h_v(t)] / N_task

该指标叫“缓存服务可用率”，不简称“缓存命中率”。服务可用不代表任务最终一定由非云节点执行，因为实际执行还受信道、计算能力、资源竞争和时延影响。

### 16.4 Cloud 执行比例

    R_c_bar = [sum_t sum_v a_v(t) x_{v,c}(t)] / N_task
    R_nc_bar = 1 - R_c_bar

这是最终实际执行位置统计，不是缓存可用率的补集。

### 16.5 五类实际执行率

11.pdf 明确给出了实际执行变量 x_{v,n}(t) 和 Cloud 执行比例，但没有单独写出五类分项率公式。按用户要求，五类输出直接由同一实际 x 分解，并统一使用 N_task 作为分母：

    R_local
    = [sum_t sum_{v in V minus V_s} a_v(t) x_{v,v}(t)] / N_task

    R_service_vehicle
    = [sum_t sum_{v in V minus V_s} a_v(t)
       sum_{m in V_s} x_{v,m}(t)] / N_task

    R_RSU
    = [sum_t sum_{v in V minus V_s} a_v(t)
       sum_{r in R} x_{v,r}(t)] / N_task

    R_UAV
    = [sum_t sum_{v in V minus V_s} a_v(t)
       sum_{u in U} x_{v,u}(t)] / N_task

    R_cloud
    = [sum_t sum_{v in V minus V_s} a_v(t) x_{v,c}(t)] / N_task

由于 C9-C10 对每个已到达任务要求唯一执行节点，五类率在 N_task>0 下应加和为 1。上述分解是对论文已有实际执行变量的报告统计，不是新增优化机制。

### 16.6 配置成本和实际更新率

    Cca_bar = (1/T_te) sum_t C^ca(t)
    Cq_bar  = (1/T_te) sum_t C^q(t)

    Zca_bar = [sum_t sum_{m in M} z_m^ca(t)] / (T_te |M|)
    Zq_bar  = [sum_t sum_{u in U} z_u^q(t)] / (T_te |U|)

更新率只使用可行化后的实际更新变量。缓存配置不变或 UAV 最终位置不变时不计更新。

### 16.7 训练收敛与验证

训练回合时隙平均奖励的滑动平均：

    E_e = { max(1,e-W+1), ..., e }
    r_tilde_e = (1/|E_e|) sum_{i in E_e} r_bar_i

第 k 次验证的平均系统效用：

    G_bar_k^val
    = [sum_{j=1}^{N_val} sum_{t=1}^{T_val} G_{k,j}^val(t)]
      / (N_val T_val)

模型按多个固定验证场景的平均系统效用选择，不仅按单次训练奖励、策略损失、价值损失或 entropy 选择。策略损失、价值损失和 entropy 主要用于检查数值发散，不作为最终性能优劣的直接依据。

Source: 11.pdf p.28-32

## 17. 仿真和对比方法的论文边界

### 17.1 仿真时序

- SUMO 生成真实城市道路车辆轨迹，并在每个基础时隙输出位置和速度。
- 服务车辆身份、数量和路线在单个实验内固定；其余车辆随机产生任务。
- RSU 位于固定位置，UAV 固定高度并在二维区域内移动，Cloud 存储全部服务。
- 所有算法应共享同一组 SUMO 轨迹、任务到达记录、服务请求序列、初始缓存、UAV 初始位置和系统参数，以保证公平比较。

### 17.2 11.pdf original solver design 的论文对比算法

- CF-PPO-MKKT：预测 + 条件因子化 PPO + cache/UAV 可行化 + transfer-exchange matching + KKT。
- PPO-MKKT：相同状态、辅助动作、奖励、可行化、matching 和 KKT，但连续头不做条件 mask。
- Greedy-MKKT：预测服务流行度作缓存评分；UAV 向预测任务车辆几何中心方向移动；matching 和 KKT 不变。
- Static-MKKT：初始缓存和 UAV 位置全程固定；matching 和 KKT 不变。
- Random-MKKT：随机产生配置请求、评分和位移，再走同样的可行化；matching 和 KKT 不变。

### 17.3 11.pdf original solver design 的论文消融算法

- CF-PPO-NoPred：不预测车辆位置和服务流行度，使用上一时隙观测位置和历史请求统计。
- CF-PPO-Transfer：保留单任务 transfer，取消双任务 exchange。
- CF-PPO-Equal：保留配置和 matching，但候选评价和最终资源分配均使用等比例资源。若 N_m(t)=|V_m(mu_t)|>0，则 w_{v,m}=1/N_m(t)、b_{v,m}=1/N_m(t)。

以上算法用于论文实验对比，不改变主方法的理论实现。

## 18. 实现禁止事项摘要

- 不得把 paper_spec.md 当作比 11.pdf 更高的理论来源。
- PDF SHA256 未变化时，不得再次完整读取整篇 PDF 和全部图片。
- 不得把预测位置或预测服务类型用于当前任务调度；预测只用于配置。
- 不得把未经可行化的 PPO 动作直接当作实际缓存或 UAV 决策。
- 不得把请求 trigger 代替实际缓存变化或实际 UAV 移动。
- 不得把缓存服务可用率当作缓存命中率，也不得把它与 Cloud 执行比例简单互补解释。
- 不得把 Cloud 任务放入边缘共享带宽/计算 KKT。
- 不得用 Hungarian、stable marriage 或 max-weight matching 替换 transfer-exchange matching。
- 不得在 reward 中加入论文没有定义的重复 cache/UAV 惩罚。
- 不得使用旧工程、GitHub 或“合理推断”补齐论文未定义的理论机制。
