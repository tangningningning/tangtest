# Matching / KKT 系统目标与经济性独立检查

日期：2026-10-17  
范围：`configs/proposed/system_config.toml` 指向的正式 `PER_SLOT_ANALYTIC` 路径；只读源码、Training Episode 20001 / 20150 的所选 Slot，并调用现有单 Slot Matching/KKT。未运行完整 Episode、批量优化或 PPO。

## 结论摘要

- Matching 在固定 Cache、UAV 位置下最大化全槽任务效用和，并在每个候选 Matching 上重新分配 Edge CPU、带宽和有限 Cloud CPU。CPU / 带宽 KKT 与这一定义的连续子问题一致；离散 Matching 只保证 transfer-exchange 局部稳定，不保证全局最优。
- Cloud 的 40 GHz 是真实共享 CPU 池，Cloud CPU 通过 KKT 在当前所有 Cloud 任务间分配。20 Mbps 是每个 Cloud 任务独立使用的等效上行速率，代码没有 Cloud 上行带宽竞争；这是正式配置明确选择的抽象模型。13 倍因子乘入 Cloud 上传时延，没有乘入车辆发送能耗。
- SV 不必然比 Cloud 差，也不保证有利。缓存、任务计算密度、实际距离/信道、SV 的 1 GHz CPU，以及 Cloud 的共享 CPU 分配共同决定效用。所选槽中既有 SV 的聚合效用提升，也有单个任务自身效用低于 Cloud、但转移后因 Cloud CPU 重分配使全槽效用上升的情况。
- 确认一项目标规格歧义：正式 TOML 注释给出的 Cache / UAV 成本公式没有外层 `cache_cost_weight` / `uav_movement_cost_weight`，而可执行代码将两个权重乘在各自整项成本上。代码内部的 `G = Q - C_ca - C_q` 一致，但无法从现有配置判定外层权重是否本意中的目标项。这一合同需澄清后，才能声称实现严格符合注释公式。

## 1. 当前实际系统目标

### 任务效用、时延与车辆能耗

对正式 Slot 中每个到达任务 \(i\)，系统计算：

\[
u_i = \omega_T\left(1-\frac{T_i}{T_{ref}}\right)
    + \omega_E\left(1-\frac{E_i^{veh}}{E_{ref}}\right),
\qquad \omega_T+\omega_E=1.
\]

正式参数为 \((\omega_T,\omega_E)=(0.6,0.4)\)、`T_ref = 30.473728355325758 s`、`E_ref = 5.1574642506299275 J`。因此每个任务有常数基值 1，Matching 比较中的可变部分是归一化时延和车辆能耗惩罚。`PER_SLOT_ANALYTIC` 将每个到达任务在本 Slot 独立完成并结算；没有 1 秒截止期、时延截断、失败扣分或跨 Slot 队列。任务时延可以大于 Slot 长度仍获得此效用。

变量含义与单位：输入 \(D_i\) 为 bit，计算负载 \(L_i\) 为 cycle，车辆发射功率 \(P_i\) 为 W，速率 \(R_i\) 为 bit/s，时延为 s，车辆能耗为 J。

### Edge / SV 执行

对 Edge 节点 \(m\)，其带宽总量为 \(B_m\) Hz，\(b_{im}\in(0,1]\) 是任务的频率带宽份额，\(w_{im}\in(0,1]\) 是 CPU 份额。正式 V1 噪声模式下：

\[
R_{im}(b)=B_m b\log_2\!\left(1+
 \frac{P_i g_{im}}{N_0\,NF\,B_m b}\right),\quad
T_{im}=\frac{D_i}{R_{im}(b)}+\frac{L_i}{w_{im}F_m},\quad
E_{im}^{veh}=P_i\frac{D_i}{R_{im}(b)}.
\]

噪声按实际分配带宽 \(B_m b\) 积分；Edge 之间采用正交独立频谱、不建模互扰。SV 当前 CPU 为 1 GHz、带宽为 10 MHz；V2V/V2I 地面增益为 \(g=\beta_g/d^{2.75}\)。Matching 不用 `range_m` 作为硬截断：只有服务缓存、当前车辆/节点位置有效且算出的链路速率为正有限值，才将该 Edge 节点作为候选。距离仍会通过信道速率影响完整效用。

### Cloud 执行

正式 Cloud 参数为共享 CPU 池 \(F_c^{pool}=40\) GHz、每任务固定等效上行 \(R_c=20\) Mbps、上传时延系数 \(\psi_c=13\)。若 KKT 分得 Cloud CPU 份额 \(w_{ic}\)，则：

\[
T_{ic}=13\frac{D_i}{20\,Mbps}+\frac{L_i}{40\,GHz\cdot w_{ic}},
\qquad E_{ic}^{veh}=P_i\frac{D_i}{20\,Mbps}.
\]

因此，13 倍只放大上传时延，不放大车辆发射能耗。`nodes.cloud.total_bandwidth_bps` 与 `uplink_rate_bps` 被校验为相等的兼容字段，并不表示 20 Mbps 汇总共享池；`resource_sharing=false` 明确表示每个 Cloud 任务都取 20 Mbps，不按 Cloud 任务数再次分摊。Cloud CPU 则通过有限池 KKT 共享。此上传模型是代码和配置共同声明的抽象假设，不等价于真实 Cloud 接入容量保证。

### Slot 任务效用和重配置成本

\[
Q(t)=\sum_{i\in\mathcal T_t}u_i,\qquad
G(t)=Q(t)-C_{ca}(t)-C_q(t).
\]

代码实际计算的成本为：

\[
C_{ca}=\kappa_{ca}\sum_m\left(\eta_m^{ca}z_m^{act}
 +\lambda_{cache}\xi_m^{ca}L_m^{+}\right),\quad
\kappa_{ca}=\texttt{cache\_cost\_weight}=0.1,
\]

\[
C_q=\kappa_q\sum_u\left(\eta_q z_u^{act}
 +\lambda_{movement}\xi_{mov}d_u\right),\quad
\kappa_q=\texttt{uav\_movement\_cost\_weight}=0.50.
\]

其中 \(L_m^+\) 是新增装入的服务 MB 数，删除不产生负装入量；节点任一缓存位变化时固定项只计一次。UAV 移动能耗另记为 \(E_{mov}=15\,J/m\times d_u\)，移动代价按上述权重映射为效用成本；距离还须满足最大速度、边界等约束。实际 Slot 只将 Cache 和 UAV 成本从 \(Q\) 中扣一次。

**规格差异：**`system_config.toml:385-387` 展示的正成本公式是 `C_ca = eta*z + lambda_cache*xi*loaded_mb`、`C_q = eta_q*z + lambda_movement*xi_mov*distance`，没有上述外层 \(\kappa\)。源代码分别将 `cache_cost_weight` 与 `movement_cost_weight` 乘在整项成本上。因此实际 Cache 固定项为 `0.1*eta`，加载项含 `0.1*lambda_cache`；UAV 固定项为 `0.5*eta_q`，距离项含 `0.5*lambda_movement`。这是明确的代码/注释差异；外层权重是否有意加入目标，现有文件无法证明。它不是重复扣成本：`G` 只减一次已经缩放的 `C_ca` 和 `C_q`。

## 2. 原始问题与当前 Matching/KKT 子问题

若把同一 Slot 的 Cache、UAV 位置、卸载及资源一并视为原问题变量，形式为：

\[
\max_{C,q,x,w,b}\quad
\sum_i u_i(x,w,b;C,q)-C_{ca}(C_{prev},C)-C_q(q_{prev},q)
\]

并满足：缓存变量二值且 \(\sum_f s_f C_{mf}\le C_m\)；每个任务恰好选择一个合法目标 \(\sum_{n\in N_i^{fea}}x_{in}=1\)；只有缓存了服务且当前链路可计算的 Edge 节点可进入 \(N_i^{fea}\)；正式 `RESOURCE_AWARE_FALLBACK` 下 Cloud 始终是候选。每个 Edge 节点各自满足 \(\sum_i w_{im}\le1\)、\(\sum_i b_{im}\le1\)，且份额在各自 KKT 下限和 1 之间。Cloud 满足 \(\sum_i w_{ic}\le1\)，分配频率为 `40 GHz * w_ic`。UAV 还须满足速度、位置边界和节点间安全距离约束。

当前正式 Slot 调用时，传入的 Cache 与 UAV 位置已固定；`optimize_task_matching` 只优化卸载目标 \(x\)，每次候选评价都对受影响节点重新求 KKT。由于重配置成本对固定 Cache/UAV 下的卸载目标是常数，Matching 最大化 \(Q=\sum_i u_i\) 与最大化固定配置下的 \(G\) 等价。Matching 本身不对 \(C,q\) 做联合全局优化。

固定 Matching 后，CPU 和带宽损失可分离：车辆能耗与 CPU 份额无关，故每个 Edge / Cloud CPU 池的子问题为

\[
\min_{\epsilon_w\le w_i\le1,\,\sum_iw_i\le1}
\sum_i \frac{\omega_T L_i}{T_{ref}F_m}\frac1{w_i}.
\]

带宽子问题为

\[
\min_{\epsilon_b\le b_i\le1,\,\sum_ib_i\le1}
\sum_i \frac{D_i\left(\omega_T/T_{ref}+\omega_E P_i/E_{ref}\right)}{R_i(b_i)}
\]

（本正式逐 Slot 路径每任务发送时间份额为 1；一般份额 \(\delta_i\) 时，时延项系数为 \(\omega_T/(T_{ref}\delta_i)\)，能耗项不变）。V1 带宽解在内部点满足 \(c_iR_i'(b_i)/R_i(b_i)^2=\lambda\)，边界点按 KKT 互补条件处理。Cloud 不求带宽变量，因为每任务上行固定为 20 Mbps。

KKT 子问题对给定 Matching 是凸、可分离的，代码用确定性二分求解；这不意味着离散 Matching 全局最优。当前 `transfer_exchange` 每轮只枚举单任务 transfer 和两个任务互换，候选用完整任务效用/KKT 评价，并对有限 Cloud 池任务集变化重新分配全部 Cloud CPU。稳定检查只证明没有可接受的单 transfer / 双任务 exchange 改进；多任务联盟或整个离散问题的全局最优没有保证。达到 `max_updates=100` 时也不能称为稳定，但本次三个实际调用均在限额内收敛并通过稳定复检。

## 3. SV 与 Cloud 的经济竞争

同一任务在两路径之间的效用差是：

\[
u_{SV}-u_{Cloud}=
\frac{\omega_T}{T_{ref}}(T_{Cloud}-T_{SV})+
\frac{\omega_E}{E_{ref}}(E_{Cloud}-E_{SV}).
\]

必须比较 KKT 后完整时延和车辆能耗。正的 Edge 速率本身不说明经济优势。`1 GHz` 的 SV 计算明显慢于无争用时 40 GHz Cloud；然而 Cloud 的 13 倍上传惩罚会造成较长通信时延，Cloud 多任务竞争又会降低单任务 CPU 份额。SV 能否胜出取决于计算密度、信道距离、Cache 是否容纳服务、同节点带宽/CPU 竞争和当前 Cloud 负载。

Training pool 的服务缓存大小也会限制 SV 候选。所选 Episode 20001 Slot 169 中任务 48 的单任务反事实在假设 SV 可缓存的前提下，SV 效用比 Cloud 高约 `0.01259`；其服务 6 大小为 `16,216.9 MB`，超过每个 SV 的 `12,000 MB` 容量，因此该反事实不是合法候选，不能归咎于 Matching 漏选。该槽中可缓存的服务 4 任务 45 距 SV 23 仅约 15 m，单任务 SV 速率约 125.6 Mbps，但 KKT 后其单任务效用仍略低于 Cloud。

## 4. 两个实际 Slot 的窄范围核实

固定车辆、任务、UAV 初始位置和其余缓存；训练数据来自 Episode 20001 Slot 169（7 个任务）和 Episode 20150 Slot 45（9 个任务）。初始 Cache 用当前正式均匀可行初始化。Cloud-only 是全部任务合法分配到 Cloud 并对 40 GHz 池重做 KKT。强制 SV 反事实使用现有 `evaluate_matching_candidate` 对全任务集精确重评。表中 \(G\) 使用代码实际成本权重，时延/能耗为所有任务均值。

| Episode / 配置 | 任务分配 | 目标任务 T / E / U | 全槽 Q / G | 平均 T / E |
|---|---|---:|---:|---:|
| 20001 / 初始 Cache、Cloud-only | 全部 Cloud | task 45: `3.0336 s / 0.03762 J / 0.937355` | `6.463126 / 6.463126` | `3.7073 s / 0.04776 J` |
| 20001 / 初始 Cache、正式 Matching | task 45→SV 23；task 7→UAV 0；其余 5 个→Cloud | task 45: `3.2875 s / 0.00599 J / 0.934808` | `6.534836 / 6.534836` | `3.2026 s / 0.04379 J` |
| 20150 / 初始 Cache、正式 Matching | task 50→SV 21；另有 RSU/UAV 任务，4 个→Cloud | task 50: `3.3117 s / 0.23274 J / 0.916745` | `8.463349 / 8.463349` | — |
| 20150 / Cache 加载后、Cloud-only | 全部 Cloud | task 50: `4.3223 s / 0.05659 J / 0.910510` | `8.141708 / 8.042212` | `4.6151 s / 0.05799 J` |
| 20150 / Cache 加载后、正式 Matching | task 50→SV 41；task 44→RSU 0；task 32→UAV 0；task 33、14→UAV 1；其余 4 个→Cloud | task 50: `2.3883 s / 0.04849 J / 0.949216` | `8.495820 / 8.396324` | `2.5758 s / 0.06840 J` |

Episode 20001 的正式 Matching 收敛于 5 次更新。将 task 45 从 Cloud 单独移到 SV 23 后，其自身效用下降 `0.002547`，但释放的 Cloud CPU 份额使其他任务效用合计上升 `0.009871`，全槽 \(\Delta Q=+0.007324\)。因此本槽的 SV 聚合收益是全系统资源重分配的结果；task 45 自身并没有战胜 Cloud。其 Cache 未改动，故该对照重配置成本为 0。

Episode 20150 的初始 Cache 正式 Matching 收敛于 2 次更新，已将 task 50 分给 SV 21。为构造近距离合法 SV 候选，只改 SV 41 的一行 Cache：移除服务 8、装入任务所需的服务 7（`7,949.6 MB < 12,000 MB`），其余缓存不变。实际代码计算该动作 `C_ca=0.099496`，移动成本为 0；若按 TOML 注释公式不乘外层权重，同一动作成本会是 `0.994962`。

在相同新 Cache 下，将 task 50 从 Cloud 单独移到 SV 41，目标任务效用增加 `0.038706`，其他 8 个任务由于 Cloud CPU KKT 重分配合计增加 `0.011456`，全槽 \(\Delta Q=+0.050162\)。但相对“不改 Cache、Cloud-only”的即时净增益为 `+0.050162 - 0.099496 = -0.049334`。更直接地比较两个正式 Matching：初始 Cache 下 `Q=8.463349`，只改变 SV 41 Cache 后 `Q=8.495820`，服务任务带来的 `ΔQ=+0.032471` 小于 `C_ca=0.099496`，本 Slot 的 \(\Delta G=-0.067026\)。这是单 Slot 反事实，不能推出服务 7 在后续 Slot 的复用价值。

两个槽的 Cloud-only 和正式 Matching 均实际使用完整 40 GHz Cloud CPU 池。正式结果中 Edge CPU / 带宽节点份额和均为 1（一个带宽和约 `1-5e-13` 的浮点误差），Cloud 分配和为 40 GHz；满足各自节点约束。E20150 加载后 formal Matching 的 transfer-exchange 3 次更新并通过稳定复检。样本反事实都在全任务集上重算 KKT，因此数值包含离开 Cloud 后其余 Cloud 任务的 CPU 变化；目标 SV 节点本身在这两个指定 transfer 中没有第二个任务共享。

## 5. 发现与尚未证明的事项

### 已核实

- 当前正式 Matching 候选生成在 `RESOURCE_AWARE_FALLBACK` 下保留所有合法缓存 Edge 和 Cloud；Cloud 在 Edge 候选存在时仍参与选择。合法 Edge 还要求当前有效物理路径。`range_m` 不是 Matching 的硬截断。
- Matching 优化的是 `task_utility_sum`，其 exact evaluator 对 Edge CPU/BW 和有限 Cloud CPU 都重算，任务效用来自同一个 `system_model.task_utility`。对固定配置的卸载子问题，忽略 Cache/UAV 成本是正确的，因为这两项不随卸载目标变化。
- CPU KKT 只惩罚计算时延；带宽 KKT 同时包含时延与车辆上传能耗系数。能耗不依赖 CPU 份额、但依赖上传速率，因此两个子问题系数与系统目标一致。
- Cloud 的 13 倍时延因子正确；40 GHz 有限 CPU 池参与竞争，20 Mbps 上行则按每任务固定、不竞争实现。这里没有发现 Cloud 公式与 Matching 评价之间的分叉。

### 模型 / 理论限制

- `cache_cost_weight` 和 `uav_movement_cost_weight` 的可执行成本公式与 `system_config.toml` 紧邻公式注释不一致。无法证明外层权重是有意的额外目标项还是旧字段；在意图确认并统一公式之前，系统目标对外层重配置成本的定义是未决项。
- Transfer-exchange 是近似离散求解。它没有枚举多个任务同时从 Cloud 转到同一 Edge 的联合操作，稳定性不能证明整个离散问题全局最优。若后续联合优化要求逐 Slot 全局最优 Matching，需要换用或补充全局离散求解；若现有合同接受局部稳定，这不是已证实的实现错误。
- Cloud 上行带宽无共享是明确模型假设。若真实意图是 20 Mbps Cloud 汇总链路，当前系统会高估多任务并发 Cloud 上行能力，需要更改系统模型及对应 KKT/Matching 评价；当前配置明确写成每任务独立等效链路，故本次不将其判为代码错误。
- SV 服务容量限制和任务实际距离足以解释部分 SV 候选缺失或收益较低。本报告只分析两个指定训练槽，不能据此估计全训练池的 SV 占比或普遍优势。

### 尚无法证明

- 无法证明外层 Cache/UAV 权重是否符合作者想要的经济权衡；代码实际权重有明文配置，但 TOML 展示的成本式省略它们。
- 无法证明局部 transfer-exchange 的结果等于原离散问题全局最优，也没有穷举所有 cache/UAV 状态。
- 两个局部反事实不能证明配置更改在整个 Episode / 训练池的累计收益；Cache 的后续复用价值未在本次重放。

**是否阻碍联合优化：** 当前 Matching/KKT 可以正常作为固定 Cache/UAV 下的下层优化器，未发现需要立即修理的求解器错误。若要把它报告为“求解 TOML 注释所定义的目标”，外层成本权重差异必须先澄清；若要求离散卸载全局最优，也需更强的 Matching 求解器。这两项分别是目标规格和全局最优性范围问题，不是本次单 Slot KKT 约束失败。

## 6. 源码位置与运行记录

- 任务效用和 Cloud / Edge 公式：`src/common/system_model.py:57-148,173-260,453-479`。
- Cache / UAV 成本：`src/common/system_model.py:309-450`；Slot 实际组装 \(G\)：`src/decision/slot_environment.py:1153-1315`。
- 合法候选和 Transfer-Exchange：`src/decision/task_allocation.py:231-315,454-898,958-1483`。
- CPU / BW KKT 与 Cloud CPU KKT：`src/decision/task_allocation.py:2043-2370,2372-2795,2814-3258,3261-3411`。
- 完整 Matching/KKT 效用重评：`src/decision/task_allocation.py:3414-3772`；增量候选对有限 Cloud 池的完整重评：`src/decision/task_allocation.py:396-451,3775-4145`。
- 正式参数和成本公式注释：`configs/proposed/system_config.toml:147-160,257-287,368-410,494-512`。
- 使用解释器：`C:\\software\\Anaconda_envs\\envs\\adaptive_cache_mec\\python.exe`；已检查 `sys.prefix` 指向 `adaptive_cache_mec`。所选 formal Matching 调用约 `0.01–3.7 s`；已完成 20001 Slot 169、20150 Slot 45（两种 Cache 状态）的窄范围调用。未创建临时脚本，未启动 Episode/PPO 或批量优化。

```ini
SYSTEM_OBJECTIVE_CONSISTENT = UNRESOLVED
CLOUD_MODEL_CONSISTENT = YES
MATCHING_OBJECTIVE_CONSISTENT = YES
CPU_KKT_CONSISTENT = YES
BANDWIDTH_KKT_CONSISTENT = YES

SV_ECONOMIC_ADVANTAGE_EXISTS = YES
SYSTEM_MODEL_REPAIR_REQUIRED = YES
SOLVER_REPAIR_REQUIRED = NO

SOURCE_CHANGED = NO
FORMAL_CONFIG_CHANGED = NO
FORMAL_DATA_CHANGED = NO
PPO_TRAINING_EXECUTED = NO
```

`SYSTEM_OBJECTIVE_CONSISTENT=UNRESOLVED` 和 `SYSTEM_MODEL_REPAIR_REQUIRED=YES` 专指上述外层成本权重与配置注释之间的目标合同差异；可执行代码内部的 \(G=Q-C_{ca}-C_q\) 计算前后一致。`SOLVER_REPAIR_REQUIRED=NO` 表示未发现当前定义下的 Matching/KKT 实现错误；不表示离散 Matching 已全局最优。
