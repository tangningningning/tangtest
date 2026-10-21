# MAPPO 主线 Cache–UAV 协同学习静态审计

审计日期：2026-10-21（Asia/Shanghai）。项目根目录：`C:\Users\tangning\Desktop\code\adaptive_cache_mec`。

## 结论与证据范围

**当前代码中，Cache、UAV、Matching/KKT、系统奖励和两个 Actor 的优势之间存在真实闭环。未发现正式交替调度中的旧动作概率错配、同批数据先后更新两个 Actor、成本重复扣除、Cache UPDATE 静默变 KEEP 等主线正确性错误。参数分离不妨碍通过环境回报学习跨模块收益，但当前实现没有节点级协同归因。**

最值得优先处理的是：用于 Macro Actor 的 N2 Critic 存在有运行证据支持的长时域价值滞后；9 个 Cache 节点、Holding 决策和两个 UAV 仅分别获得全局优势；历史 Last1 Critic 样本在队友/自身策略已变化后仍以较强权重参与更新。前者有明确数值现象，后两者是设计局限或机制风险，不能直接称为 PPO Bug。

确认的实现/合同问题较小：`macro_critic_replay_history_episodes=0` 没有控制实际 Last1 使用；一个名为节点 log-probability “replay error”的指标实际在 Actor 更新后计算。它们分别影响配置可理解性和诊断可信度，不证明 Actor 没有训练。

证据分三层：

- **代码事实**：当前本地源码、正式配置及其直接依赖；所有行号针对本次工作区快照。
- **运行事实**：7 个现存 Run 的指定日志和 initial/latest Checkpoint；总计 1,148 个 Episode、287 个更新组、344,400 条 Slot 记录。各 Run 单独核对，未合并为一个实验，也未把相同受控结果当作独立重复证据。
- **仍无法确定**：这些 Run 是否全部由当前源码字节生成、策略泛化是否改善、联合策略是否有正的协同交互收益、某项设计局限对最终收益的因果影响。Run 没有源码 Git/文件哈希证明，训练日志也不是配对冻结评价。

没有读取远程 GitHub 实现，没有以历史报告替代当前代码。没有启动训练、环境重放、Matching/KKT 评价或超参搜索。

## A. 正式执行路径和版本

### A1. Git 与工作区

| 项目 | 审计起始事实 |
|---|---|
| 分支 | `main` |
| HEAD | `aa8981d3ae5aa9ba2d8ce130819c38afed5b480b` |
| 已有 tracked 改动 | `src/ppo_simple/trainer.py` 有 4 行新增：将已有 Macro LR override 接到 `run_training` 和 CLI；本次没有还原或改写 |
| 已有 untracked | `environment_data/`、多份 reports、4 个 LR 启动包装脚本；保留原状 |
| 审计对象 | **HEAD 加已有工作区改动**，不是纯 HEAD |
| 项目约定 | 读取根 `AGENTS.md`；未发现本路径下的下级 `AGENTS.md` |

关键文件 SHA256（作为本报告的本地源码定位证据；不能反向证明旧 Run 来源）：

| 文件 | SHA256 |
|---|---|
| `configs/proposed/alternating_actor_updates_v1.toml` | `297822402a783b4bbbb28b1be2e61aefb1fa2cef5e110af279e814f58e4b670f` |
| `src/ppo_simple/trainer.py` | `923635d3282424536580f18c7dca4a448cb250f4713f7062054aec6cb30b8ee5` |
| `src/ppo_simple/rollout.py` | `cffd761bee68e26b222ad8849b6d3b13d022fc34bb3b23a5fd7435a8859c6243` |
| `src/ppo_simple/policy.py` | `8c190ab81a34748145edac4138395e1e67709b441ebbc13bbd62d56027a32b86` |
| `src/ppo_simple/interface.py` | `77bbf5ea8ec0e3b35779db09b7bebe17db4dde8f5acb778b16952fa33c3ccef9` |
| `src/decision/slot_environment.py` | `f1d1e6acc80fc62eca30ffb17c82b2e5911f5c1a038f982efd42b91ea49da9b9` |
| `src/decision/task_allocation.py` | `cbd37d8a71bba8974af178cf6e5fbbaa1e6be030d49005d776b4d7040f42449c` |
| `src/common/system_model.py` | `061224f64e2a8148200432b3d9200085034188de7c64601d51ac9cb3f810a388` |

### A2. 实际调用链

| 层级 | 实际文件、函数与位置 | 确认结果 |
|---|---|---|
| 唯一正式 CLI | `scripts/training/mainline_runner.py:17`，`main` | 导入 `ppo_simple.trainer.main`；本地 `src` 优先进入 import path |
| 配置和 Pool 门禁 | `src/ppo_simple/trainer.py:3928`，`run_training`；`scripts/training/training_pool_contract.py:337` 起 | 正式配置路径只能是 `configs/proposed/alternating_actor_updates_v1.toml`，校验 Training Pool、配置及数据合同 |
| Context / 数据对齐 | `src/ppo_simple/interface.py:360`，`load_environment_context`；`:558`，`aligned_train_steps`；`:1371`，`prepare_aligned_rollout_steps` | 按显式车辆 ID 对齐、保留 NPZ 角色内顺序；不是把多个 realization 的车辆 ID 当成固定同一辆车 |
| 策略实例 | `src/ppo_simple/trainer.py:1040` 附近；`src/ppo_simple/policy.py:1866` | `AdaptiveMultiNodeFactorizedActorCritic` |
| 采样 | `src/ppo_simple/trainer.py:4492` 附近 → `src/ppo_simple/rollout.py:860`，`collect_dual_timescale_episode` | 4 个完整 Episode，随机采样，采集期间不更新 |
| Cache 候选 | `src/ppo_simple/rollout.py:628` → `src/ppo_simple/primitive_control.py:386` → `src/decision/cache_configuration.py:1120,1275` | 每节点 KEEP 加有界合法 UPDATE；共享同一旧 Cache 生成候选 |
| UAV 候选 | `src/ppo_simple/rollout.py:772` → `src/decision/uav_configuration.py:323,482` | 依次生成、采样、校验并应用 UAV0/UAV1 位置 |
| 环境执行 | `src/ppo_simple/rollout.py:1501` → `src/decision/slot_environment.py:973`，`execute_configured_slot` → `:441`，`execute_slot` | 直接执行采样配置，通过 configuration override 跳过另一套预测配置优化 |
| Matching / KKT | `src/decision/slot_environment.py:613–645` → `src/decision/task_allocation.py:993`，`optimize_task_matching` | 使用 final Cache 和 final UAV 位置；可行目标 `:240`，Edge KKT `:3163`，Cloud 资源 `:3436`，实际任务评价 `:3807` 起 |
| 奖励 / buffer | `src/decision/slot_environment.py:716–730` → `src/ppo_simple/rollout.py:459,1612,1660,1747` | 真实 Q、Cache cost、movement cost → 同一个 Slot reward → Micro 和 Macro |
| 优势 / 优化 | `src/ppo_simple/trainer.py:1396,1507,1772,2914,3300`；`src/ppo_simple/rollout.py:2342,2381` | 分 Episode 算 GAE，然后拼组；独立 Critic 更新，交替 Actor 更新 |
| 冻结评价 | `src/ppo_simple/evaluator.py:876` 起，`:991` 调同一 collector | reward 定义一致；支持 deterministic/stochastic，默认 deterministic 的决策规则与随机训练分布不同 |

项目根旁存在多个 stage/worktree 目录，本次未把它们作为主线。当前源码仍含 single-node、sparse、fixed2 兼容分支、`exact_q_parallel.py`、`uav_credit.py` 等模块；正式配置选择 multi-node factorized、adaptive holding，Trainer 的 Actor 优势路径没有调用 Exact-Q 或节点反事实信用模块。现有日志中 Exact-Q 环境执行数为 0。不能因为模块文件存在就宣称正式算法实现了其功能。

### A3. 生效合同

正式配置 `configs/proposed/alternating_actor_updates_v1.toml:492–559` 的用户指定线索全部成立：

| 参数 | 当前值 |
|---|---|
| experiment.name | `ALTERNATING_ACTOR_UPDATES_V1` |
| actor_update_mode / first / period | `ALTERNATING` / `MICRO` / `1` |
| realizations_per_policy_update | `4` |
| gamma / gae_lambda | `1.0` / `0.95` |
| macro_critic_target_mode | `two_step_smdp_td` |
| long_value_enabled / target | `true` / `realized_return_to_go` |
| exact_q_training_enabled | `false` |
| Cache / UAV | 9 个 Cache 节点（5 SV+2 RSU+2 UAV），10 服务；2 UAV 自回归 |
| Holding | 1–5 Slot；只在 1–4 的合格时点采样 termination |
| Actor LR | 默认 Macro `5e-4`，Micro `3e-4`；Macro CLI 可覆盖为四个受支持固定值 |
| Critic LR | Macro `1e-3`、Micro `3e-3`、Long `1e-3` |
| 预算 | Actor 活跃组 4 epoch；Macro/Long Critic 各 50 step；Micro 4×10 minibatch |

两个不同的 Schema 字符串不是自动不兼容：配置/Context 的输入 Schema 为长字符串 `adaptive_dual_timescale_proposed_cache_economic_features_fixed_lr_horizon_mask_v5_multi_node_cache_factorized_v1_keep_node_identity_v1_per_slot_analytic_task_model_v1_training_p95_reward_norm_v1`；实际策略/Checkpoint Schema 为 `adaptive_joint_cache_uav_mappo_v1_factorized_keep_node_identity_v1`，Architecture **2**、Checkpoint **65**。前者在 Trainer 构造时校验并映射到后者，不能误用父类 Architecture 36。

Checkpoint `load_checkpoint`（`trainer.py:3624–3900`）检查调度、方法、Schema、Architecture、任务模型、配置哈希、Pool、张量合同及重复 Critic state 一致性，加载两个 Actor/全部 Critic/Adam/RNG/Last1 状态。SV ID 可随 realization 变化，但角色位置合同及非 SV ID 保持一致。未发现静默接受历史不兼容 Checkpoint 的路径。

## 奖励、状态、优势和概率的完整还原

### 1. Slot 奖励与系统目标

唯一实际系统目标组合位置为 `slot_environment.execute_slot`，`src/decision/slot_environment.py:716–730`：

\[
Q_t=\sum_{v\in\mathcal T_t}\left[0.5(1-T_v/T_{ref})+0.5(1-E_v/E_{ref})\right],\quad r_t=G_t=Q_t-C_{ca,t}-C_{q,t}.
\]

任务效用函数为 `src/common/system_model.py:458–484`，实际 T/E 来自 Matching 选定方案的 KKT 后评价，非候选代理分数。当前 `T_ref=30.473728355325758 s`、`E_ref=5.1574642506299275 J`，见正式配置 `:352–387`。没有额外除以任务数、在线 reward normalization 或 clipping。任务数变化会改变 Q 的规模，这是“任务效用和”的定义，不是平均任务效用。

\[
C_{ca,t}=2\times10^{-5}\sum_m\frac{8\times10^6\sum_f s_f^{MB}\max(c_{mf,t}-c_{mf,t-1},0)}{R_{m,back}},
\]

\[
C_{q,t}=5\times10^{-6}\times15\sum_u d_{u,t},\qquad d_{u,t}=\|q_{u,t}-q_{u,t-1}\|_2
\]

其中微小于位置容差的位移按 0 处理。成本实现为 `src/common/system_model.py:310–455`，通过 `cache_configuration.cache_update_cost_details:517` 和 `uav_configuration.movement_summary:574` 调用。Cache 只计算新增服务下载秒数，删除服务不付下载费；`movement_fixed_cost` 不参与。Cache 在抽象 Macro 边界立即生效，下载时延是独立成本，不再加进任务 T。符合本配置的 `ABSTRACT_MACRO_BOUNDARY_RECONFIGURATION` 合同，不能据下载可能跨秒认定遗漏了跨 Slot 下载任务。

下载成本只在 Cache 实际改变的首个 Slot 发生：每 Slot 后 `cache=next_cache`，下一 Slot 的 old Cache 已是 frame Cache（`rollout.py:1660–1663`）。Holding 期间不会再次付同一下载成本。UAV 成本按执行前后真实位置计算，不按 MOVE 标签定价。

`rollout._check_reward:459` 原样返回 G，MicroTransition `:1623` 存同一 reward，Macro `:1660` 累积同一 reward，Episode Return `:1828` 求和，Slot 日志 `reporting.py:2151` 写同一 G。Evaluator `:469` 校验同一定义。训练与冻结评价不存在奖励公式分叉；动作决策协议的差异须另行区分。

归一化说明：T_ref/E_ref 决定时延与车辆能耗的相对权重，lambda 决定两类配置成本的相对权重；这些已是正式目标的一部分。GAE 后的优势标准化只对整组优势做仿射变换，没有单独缩放 Cache 或 movement 项。当前成本总量远小于 Q（见运行证据），但没有独立边际收益证据证明应该调整 lambda，因此不建议直接调惩罚权重。

### 2. 状态和执行闭环

正式 400 维 observation（`interface.build_configuration_observation:1003–1104`）为：Cache `[0:90]`、真实当前 UAV 位置 `[90:94]`、5 步预测车辆位置 `[94:344]`、5 步预测流行度 `[344:394]`、有效 horizon mask `[394:399]`、time-to-go `[399:400]`。这是 PER_SLOT_ANALYTIC 模型，不存在跨 Slot 任务 backlog，缺少 backlog 字段本身不是 Bug。

Cache 影响 UAV：

1. `rollout.py:989` 用执行前 Cache/UAV 生成 Macro observation。
2. `:1045–1120` 对各节点采样；候选 0 是完整旧配置，非零候选必须真正改变该节点。将各选中行合并成 `frame_cache`，不把某个节点候选的整矩阵覆盖其他节点的结果。
3. `:1372` 用 **frame_cache** 重建 Micro observation；`:1392` 将 frame_cache 传入 UAV 候选。`primitive_control.py:462–570` 的 UAV 特征实际读取该 UAV Cache row，用它计算服务需求权重、质心和位置服务性。字段被 scorer 消费，非仅记录。
4. `slot_environment.py:613–645` 将 final UAV 位置和 final Cache 传给实际 Matching；`task_allocation.build_feasible_targets:240–324` 逐任务检查 Cache 服务位及真实物理链路。Cache 改变真实 Edge 可选集合。

UAV 影响 Cache：

1. UAV 候选含 STAY、需求质心、投影需求及几何探索；非零候选排除容差内 STAY，验证边界、单 Slot 15 m 最大位移和至少 10 m 间距（`uav_configuration.py:223–244,323–449`）。
2. `apply_uav_candidate:482–534` 直接写所选位置，不做“失败则留原地”的静默回退；完整联合位移再次校验 `:543–571`。后一个 UAV 的候选/安全约束使用前一个已选的新位置。
3. 实际位置进入 `task_allocation._channel_gain:1981`、Shannon 速率、KKT、T/E 与 Q。移动未必改变“链路是否正速率可行”，但一定进入链路质量；正式代码没有以 surrogate range 做硬覆盖截断。
4. 执行后位置保存到 `next_uav`，下一 Macro `:989` 的 observation 和 `_cache_node_candidates:657` 的预测服务节点位置都读取它。Cache economic 特征 `primitive_control.py:207–338` 还用节点预测可服务性处理需求，随后由 Cache scorer 消费。

候选特征使用预测位置、预测请求及低成本服务性代理；奖励使用实际位置/任务和 KKT。两者没有数值相等的要求，也没有候选“预测增益=真实边际系统收益”的保证。所有节点候选基于同一旧 Cache、当前 UAV 生成，并未预知其他节点本次采样动作及随后 UAV 动作；限制见 F03，而非执行不同步。

Micro 前一 transition 的 next_observation 在下一 Slot 选完新 Cache 后更新为实际 Micro 决策状态（`rollout.py:1373–1376`）。Micro GAE 使用后继 transition 的 old value，所以跨 Macro 边界没有错误使用“尚未选新 Cache”的后继 Micro value。Macro next_observation 则是新 Cache 决策前状态，正确匹配 Macro value。

### 3. Macro / Holding 回报与优势

设 Macro frame k 起点为 \(\tau_k\)，实际长度 \(d_k\in[1,5]\)：

\[
R_k=\sum_{j=0}^{d_k-1}\gamma^j r_{\tau_k+j},\qquad
\delta^M_k=R_k+\gamma^{d_k}(1-D_k)V^M_{old}(s_{k+1})-V^M_{old}(s_k),
\]

\[
A^M_k=\delta^M_k+(\gamma\lambda)^{d_k}(1-D_k)A^M_{k+1}.
\]

R 的实现 `rollout.py:1660`，old value/next value/frame 归属 `:1737–1792`，GAE `compute_macro_smdp_gae:2381–2441`。done 时 next value 为 0。Cache action 生效后的 UAV 行为、任务收益和两项成本全部进入 R，并经后续 frame 进入 A。Holding 的强制 max 或终局不伪造随机 termination 动作；真实采样序列、elapsed、log-prob 被逐条保存，buffer 校验 duration 与 termination 序列对应（`rollout.py:271–432`）。

Macro Critic 的训练目标与 Actor 优势不是同一个数组：

\[
y^{N2}_k=R_k+\gamma^{d_k}R_{k+1}+\gamma^{d_k+d_{k+1}}V^M_{old}(s_{k+2}),
\]

第一/第二个 frame 遇终止则去掉不存在的项，具体为 `src/ppo_simple/macro_targets.py:10–82`，调用 `trainer.py:1786–1821`。这是 full-return value 的两步 Bellman target，**不应解释为 Critic 只预测两个 frame 收益**。Actor 使用前述 SMDP GAE，且在 Critic 更新前已由冻结 old values 算好；本次 Critic 更新不会重算/污染 Actor advantage。

Long-Value target 为完整实现回报：

\[
G^{MC}_k=R_k+\gamma^{d_k}G^{MC}_{k+1}.
\]

`trainer.py:451` 构造 RTG，`:2548` 附近调用 Long Critic 拟合，`:3300–3440` 独立 50 step；`policy.py:444` 预测。其参数、优化器、输入缩放均独立于 N2 Critic。搜索实际调用后，`long_value()` 只出现在 Long Critic 拟合/诊断，没有出现在两个 Actor 的 loss 或 GAE；不能将其高 EV 当作 Actor 已获得准确长时域基线的证据。

### 4. Micro GAE

\[
\delta^\mu_t=r_t+\gamma(1-D_t)V^\mu_{old}(o_{t+1})-V^\mu_{old}(o_t),\qquad
A^\mu_t=\delta^\mu_t+\gamma\lambda(1-D_t)A^\mu_{t+1},\quad y^\mu_t=A^\mu_t+V^\mu_{old}(o_t).
\]

`compute_gae`，`rollout.py:2342–2378`；调用 `trainer.py:1514–1539`。每 Episode 独立算完再拼组标准化，没有把 Episode 末尾和下一 Episode 起点连起来。正式训练每 Episode 300 Slot、唯一终止为最后 Slot，`_validate_episode_terminal:782` 拒绝缺失终止的训练 buffer。短窗口仅供冻结评价，`rollout.py:1668` 将 truncation 与 done 区分；正式 Runner 没有给 collector 传 `max_slots`，故训练 bootstrap=0 对当前完整 Episode 正确。

两个 Actor 都以同一系统回报构造优势，但**优化 loss 不完全相同**：Micro 另加 `0.05*mean(relu(H_floor-H(STAY,MOVE)))`，`H_floor≈0.325083`（p=0.9 二元熵），Macro 当前 entropy coefficient=0。这个 soft entropy 正则不是严格的 90% 概率硬约束，也不是另一套环境奖励。不同时间尺度、GAE 和 PPO clipping 也使两个 surrogate 不等价于精确联合 J 的梯度；这属于设计近似。

### 5. 动作概率、联合 ratio 与信用分配

Cache 在给定全局状态及候选后为因子化联合分布：

\[
\pi_C(a\mid s)=\prod_{m=1}^{9}\pi_{C,m}(a_m\mid s),\quad
\ell_C=\sum_m\log\pi_{C,m}(a_m\mid s).
\]

每节点 raw logits 的合法 UPDATE 减去 \(\log K_m\)，KEEP 不减；`policy.py:455–477,508–655`。采样是直接 categorical，更新 `evaluate_macro_components:1244–1335` 用等价的 KEEP/UPDATE group × UPDATE conditional 重建：

`P(UPDATE)*P(candidate|UPDATE) = exp(corrected_candidate_logit)/sum_all exp(corrected_logit)`。

所以更新代码看起来分了 group，并不代表另一套行为策略。UPDATE conditional 的相同 cardinality offset 在其 softmax 中抵消；mask 取 rollout 保存值。

Holding 联合 log-prob 为：

\[
\ell^M_k=\ell_{C,k}+\sum_{j\in\text{真实采样 termination}}\log\pi_h(b_j\mid o_j,e_j),\quad
\rho^M_k=\exp(\ell^M_{new,k}-\ell^M_{old,k}).
\]

termination scorer 使用 observation 加 elapsed/max_slots（`policy.py:920–951`）；只评估真实 termination mask，padding 不入概率或梯度（`:1340–1407`）。Cache+termination 使用同一个 frame A，联合 clip 一次（`trainer.py:2181–2208`），没有错误地对各节点概率各 clip 后再相加。

UAV 为自回归联合分布：

\[
\pi_U(a_0,a_1\mid o)=\pi_0(a_0\mid o)\pi_1(a_1\mid o,q'_0),\quad
\ell_U=\log\pi_0+\log\pi_1,\quad\rho^\mu=\exp(\ell_{U,new}-\ell_{U,old}).
\]

rollout `:1380–1486` 保存每头候选、mask、所选前序位置 condition；`policy.evaluate_micro_joint:1579–1700` 用相同特征/condition/group balance 重算概率。更新时不以新参数重采前序动作或重建候选，因此合法动作空间与执行动作一致。

两个 PPO loss 均为 `-mean(min(ratio*A,clip(ratio,0.8,1.2)*A))`；优势在**各自完整 4 Episode 的合并 batch**做一次标准化，std 接近 0 时置零（`trainer.py:435–449`）。单个全局 A 乘联合 ratio，使各 log-prob 分量共享奖励梯度；这是合法的联合策略梯度近似，但没有节点独立 A、反事实 baseline 或因果边际收益（见 F03）。

## 交替更新组的实际时间线

`trainer.update_from_episodes:1396–1503` 与 `run_training:4441–4544`：

| 阶段 | 第 1 组 Episode 1–4 | 第 2 组 Episode 5–8 |
|---|---|---|
| 行为参数 | 同一 `(C0,U0)` | 同一 `(C0,U1)` |
| 收集时保存 | 所有动作、log-prob、old value、reward、holding trace | 同左，全部是本组新 rollout |
| 优势 | 各 Episode 用本组 old values 独立计算 | 同左 |
| `_update_micro` | Micro Actor+Critic，最多 40 minibatch step | 只 Micro Critic，Actor grad=None，Adam 对该参数跳过 |
| `_update_macro` 前段 | N2 Critic 50 step（首组无 Last1） | N2 Critic 50 step：当前 4 Episode + 前组末 Episode 4 |
| Macro Actor | 冻结 | 用 Episode 5–8 的联合动作/GAE full-batch 更新 4 epoch |
| Long Critic | 独立 50 step | 独立 50 step |
| 下组行为参数 | `(C0,U1)` | `(C1,U1)` |

因此，问题中假设的“Micro 更新完，Macro 随后仍用同批旧队友数据更新”的情况**不发生**。Micro-first 函数调用顺序仍在，但 Macro 活跃组前段只改 Micro Critic；参数分离使 Micro Actor 不变。Macro 非活跃组只拟合其 Critic，也不会改变 Macro Actor。不存在需要异步同步的另一份 rollout Actor 副本；下一组直接使用同一个 `trainer.policy` 对象的新参数。

采集阶段整模型 SHA256、优化器 step counters 均检查不变（`:4447,4521–4527`）。Macro Actor 初次 epoch 对冻结 log-prob 回放误差设 `5e-6` 门槛（`:2090–2101`）。Micro 没有对应显式回放门槛，但采样/评估公式、特征、mask 与 condition 相同。不能用 Macro 的 `rollout_logprob_max_error` 证明 Micro 也实测通过该门槛。

Macro KL guard 对整个 full-batch 的联合 ratio 计算采样近似 KL，step 后超阈值则恢复 Actor+Adam，以更小 LR 重试，全部拒绝则回滚。计数仅累计接受步，不把尝试/拒绝算有效训练（`:2209–2428`）。Micro KL 在当前 minibatch step 后检查，超过 `1.5*0.02` 停后续更新，不回滚已走步（`:1658–1713`）；它不是全 batch 或硬 trust-region 保证。现存日志没有触发 Micro early-stop，未证实此处造成实际异常。

参数和优化器集合：`policy.py:1702–1722` 明确列出 Macro Actor 的 4 个模块、Micro Actor 的 2 个模块及 3 个独立 Critic；构造中没有共享 encoder 或 Parameter 对象。`trainer.py:1103–1145` 分别建 Macro Adam、Macro Critic Adam、含两组独立参数的 Micro Adam、Long Adam；不是一个共享干路被重复优化。Macro PPO 不更新 Macro Critic，Long 更新有保护参数不变检查。非活跃 Micro Actor 的 `grad=None`，尽管同一个 Micro Adam 调了 step，其 Actor state 不推进。

每组都调用 Macro/Micro 更新函数，所以 `macro_update_count`、`micro_update_count` 是**函数/组计数**；`macro_optimizer_steps` 还包含 Macro Critic step。判定 Actor 更新必须用 `*_actor_optimizer_steps`，不能用这些聚合计数。Episode 行重复带有本组更新指标，也不能把 4 行的 Actor steps 相加算 4 次更新。

## B. 问题清单

| ID | 严重程度 | 分类 | 结论 |
|---|---|---|---|
| F01 | High | 算法设计局限 | 用于 Macro Actor 的 N2 基线有长时域价值滞后证据；高 N2 EV 掩盖与完整 RTG 的差距 |
| F02 | Medium | 潜在风险 | Last1 是旧联合策略样本，norm matching 使 1 个旧 Episode 的 Critic 梯度可与当前 4 Episode 等量 |
| F03 | Medium | 算法设计局限 | 全局 A + 独立 Cache 节点采样，没有节点/跨模块边际信用分配 |
| F04 | Medium | 算法设计局限 | Micro Actor/Critic 看不到 Holding age，不能区分临近强制 Cache 重决策的阶段 |
| F05 | Low | 已证实实现错误（配置合同） | `macro_critic_replay_history_episodes=0` 不控制实际 Last1，配置开关失效 |
| F06 | Low | 已证实实现错误（诊断口径） | 节点 log-prob “replay error” 在 Actor 更新后测量，混入正常策略变化 |
| F07 | Medium | 证据不足 | Run 无源码身份、无配对冻结协同评价，无法证明当前版本的协同改善或失败 |
| F08 | Low | 潜在风险 | 主线 UAV 次数是动作索引统计，缺少独立的实际位移日志核对 |

本次未发现 Critical 问题。F01 是有数据支持的机制局限，不是已证实 GAE/N2 公式写错；F05/F06 是范围明确的小问题，不应夸大为联合学习断链。

### F01：N2 基线的长时域滞后及 Long-Value 与 Actor 的隔离

| 字段 | 证据与判断 |
|---|---|
| 代码位置 | `trainer._update_macro:1786–1821,1845–1877,1960` 附近：old value 构造固定 N2 target；`:2548` 附近独立拟合 Long；`macro_targets.build_macro_n2_target:10–82`；`policy.macro_value:432`、`long_value:444` |
| 事实证据 | 每组 50 step 只拟合**同一份**两步 bootstrap target，不在 50 step 内向 Episode 起点重新传播目标。默认 LR Run 的 update 1/10/20/40/69，old Macro value 均值依次约 `-0.077/260.579/481.910/871.938/1085.919`，实现 RTG 均值约 `1502.719/1556.767/1546.163/1548.820/1562.752`。update 69 的 N2 post EV `0.995288`，prediction mean `1104.039`，只是逼近 N2 target mean `1102.547`。详见下文 Checkpoint 核对。 |
| 影响 | lambda<1 时，Actor A 依赖有误差的 bootstrap 值，可能把价值尚未传播的长远收益压低，并让时间位置/Holding 长度影响标准化优势。Long 高 EV 不会自动纠正这个 Actor 基线。 |
| 判定依据 | N2 Bellman 目标理论上仍指向 full-return V；本报告不把它叫“两步短视目标写错”。现有同组 MC 对比和已保存 Episode 的数值支持有限训练内的系统性滞后现象，但单个 MC 轨迹不是无噪声的真实期望 V，不能定量宣称全部差距都是估计偏差或证明收益下降由此造成。 |
| 最小修复方向 | 先把 N2 拟合指标与实现 RTG 校准指标分开报告。利用现存 transition/Checkpoint 做冻结诊断确认时间位置偏差；确认后再用独立版本小改动缩短价值传播路径或引入经过校准的完整回报监督。若让 Long 进入优势，必须明确改 Actor value 合同并重新验证，不可仅替换日志名称或静默换基线。 |

### F02：Last1 历史样本的策略滞后和权重

| 字段 | 证据与判断 |
|---|---|
| 代码位置 | `trainer._update_macro:1825–1877,2532–2546`；`_update_macro_critic_only:2914–3138`；`_pcgrad_lite_current_priority:492–603` |
| 事实证据 | 保存前组最后 1 个 Episode；下一组用**当前 Critic**重算其 bootstrap，却保留旧策略的已实现奖励/holding/状态。没有 old action likelihood、重要性修正或重放环境。历史 loss 与当前 4 Episode loss 分别求均值；历史梯度 norm 被匹配到 current norm，alpha=1；冲突时只投影历史分量，再把混合 norm 调回 current norm。不是按 1/4 样本比例自然加权。默认 LR Run 非首组 `grad_conflict_fraction` 范围 0.02–0.32、均值约 0.149706。 |
| 影响 | 当前 Actor 的 V 估计可受一个旧 Episode 较强牵引；Actor 交替意味着每个相邻组通常已有一类行为策略变化。投影保证的是当前瞬时梯度方向关系，不能保证 Adam 后的当前 loss 一定下降，更不能消除 off-policy Bellman 偏差。 |
| 判定依据 | 历史样本参与和策略年龄是代码事实，冲突比例是运行事实；没有证据证明它导致发散或性能下降。Actor 明确不使用 Last1 数据，当前组 advantage 也没有因此跨 Episode 拼接。这是 Critic 风险，不是 PPO old_log_prob 使用错误。 |
| 最小修复方向 | 先使 Last1 配置合同明确（F05），区分当前/历史奖励所属策略和当前 bootstrap snapshot。保留当前优先机制时应报告其实际权重；只在窄范围后续对照证明有害时，才调整/关闭 Last1 或降低历史权重。 |

### F03：全局信用分配与独立 Cache 动作的局限

| 字段 | 证据与判断 |
|---|---|
| 代码位置 | `rollout._cache_node_candidates:628–769`、`:1045–1120`；`policy.sample_macro_nodes:577–655`、`evaluate_micro_joint:1579–1700`；`trainer._update_micro:1586–1615`、`_update_macro:2181–2208` |
| 事实证据 | 9 个 Cache 节点基于同一旧状态独立 categorical 采样，整组 Cache 和所有 termination 动作用同一个 frame A；2 UAV 的联合动作也只对应 1 个 Slot A。正式 Actor loss 没有读取节点反事实值、独立动作 A、Exact-Q 边际值。候选经济特征按单节点变更和预测服务性生成，并非本次 9 节点联合更新后的资源竞争收益。 |
| 影响 | 某节点有利、另一节点有害时，联合动作整体优势不能直接给出各节点的边际符号；梯度期望仍可学习，但方差、互相抵消和对队友随机行为的依赖较大。Cache 联合分布也无法表达给定状态下额外相关随机协调，例如两个节点随机但严格互斥地选择同一服务；可通过不同节点特征学确定性分工，不能说完全不能协同。 |
| 判定依据 | 是数学成立的联合 PPO，不是 advantage 广播 shape 错误或少乘概率。共享奖励与参数分离足以提供联合目标梯度通路，不能等同于已实现精细协同归因。 |
| 最小修复方向 | 先做同输入、同种子的冻结模块配对/消融，估计节点/模块边际和交互项；仅在证据支持时增加轻量信用诊断或局部 baseline。无需先替换整个算法或重构。 |

### F04：Micro 状态缺少 Holding age

| 字段 | 证据与判断 |
|---|---|
| 代码位置 | `interface.build_configuration_observation:1003–1104`、`build_step_observation:1107`；`rollout.py:1372–1419,1663–1678`；`policy._termination_logits:920–951`、`micro_value:439`、`_micro_candidate_raw_logits:1466` |
| 事实证据 | Micro observation 有当前 Cache、UAV、预测和全 Episode time-to-go；没有 frame 起点、elapsed 或距 max holding 的剩余时间。elapsed 只额外传给 Macro termination head。UAV candidate/condition 同样没有 age。全 Episode time-to-go 不能推回随机 frame 的起点。 |
| 影响 | 相同可见 Cache/位置/预测下，尚可继续 Holding 和本 Slot 后必须重新选 Cache 的后继分布不同，Micro Actor 和 Critic 无法显式区分。新 Cache 选完虽立刻可见，策略仍不能充分预判“服务配置还能保持多久”；对协同预期和 bootstrap 有表示限制。 |
| 判定依据 | 输入信息缺失是可证实设计事实；当前运行中究竟多常见近似混叠、损失多大，静态审计不能确定。不是状态更新晚一个 Slot，也不意味着 Macro 决策点的 V 同样必须输入 age（Macro 起点 age 恒为 0）。 |
| 最小修复方向 | 若后续冻结诊断显示 age 条件下动作/价值差异显著，最小增补 Micro 的 elapsed/remaining 字段，版本化 observation 和 Checkpoint；不能在现有 Checkpoint 上静默改 400 维输入。 |

### F05：Replay 配置字段未生效

| 字段 | 证据与判断 |
|---|---|
| 代码位置 | 正式配置 `:552`；`trainer._update_macro:1825–1877,2532–2546`；`trainer.config_contract:1277`；`reporting.write_dual_timescale_run_manifest:2014–2015` |
| 事实证据 | 配置写 `macro_critic_replay_history_episodes=0`；实际 Trainer 从不读取该字段，第 2 组开始无条件使用 Last1。manifest 则明确写 Last1 length=1，Checkpoint config_contract 也写 current+previous-group-last1；默认 Run update 69 的 Critic group 为当前 273–276 加历史 272。 |
| 影响 | 仅看正式 TOML 会错误认为 Critic 没有历史样本；改变该字段也不能控制实际优化。影响实验合同与对 F02 的判断。 |
| 判定依据 | 配置与实际行为不一致可直接复核，故列为已证实合同实现问题；manifest 明确披露 Last1，因此不声称整个系统都隐瞒 replay，也不把 Last1 本身认定为未授权算法错误。 |
| 最小修复方向 | 明确 Last1 是固定合同还是可配功能。前者移除/弃用这个误导字段并文档化，后者真正接入门禁/运行/Checkpoint 合同；避免让一个看似有效的 0 留在正式配置。 |

### F06：节点回放误差测量时点不正确

| 字段 | 证据与判断 |
|---|---|
| 代码位置 | `trainer._update_macro:2431–2465` 的 post-loop `diagnostic_components/node_error`；`:2634` 附近输出 `macro_cache_node_logprob_replay_error_max_by_node`；真正首 epoch 联合回放检查在 `:2090–2101` |
| 事实证据 | 节点级 error 在 Macro Actor 全部更新后，比较 new node log_prob 与 rollout old node log_prob。默认 Run Macro 活跃组该字段最大 `0.2355599403`，Micro 活跃组最大仅 `9.536743e-7`；同 Run 的首 epoch联合回放最大只有 `3.814697e-6`。 |
| 影响 | 正常 PPO 策略变化会被叫作 replay error，容易误诊为采样/更新概率不一致。 |
| 判定依据 | 指标名和测量对象错位，不是 PPO 训练实际用了错误概率。不能把该 post-update 指标套用 `5e-6` 首次回放容差。 |
| 最小修复方向 | 将节点回放误差在首 Actor step 前测量；更新后的差值另命名为 node log-prob change。保留已有联合回放 guard。 |

### F07：源码与协同收益的证明不足

| 字段 | 证据与判断 |
|---|---|
| 代码位置 | `trainer.checkpoint_payload:3495–3597`、`run_training:3928`；`reporting.write_dual_timescale_run_manifest:1864`；7 个 Run 的 manifest/Checkpoint |
| 事实证据 | manifest 和 Checkpoint 有配置哈希、Schema、方法、Pool、计数器，但没有 Git HEAD、dirty diff 或源码文件哈希。四个长 Run 的 config snapshot 完全相同，实际 Macro LR 由 manifest/Checkpoint override 区分；不能单看 run_config 断言它们都是 5e-4。现有 Run 下未发现冻结评价 JSON；日志中无每组参数 delta，控制台也明确 NOT_MEASURED。 |
| 影响 | 可以核对一次 Run 内部身份和更新，不能严格证明全部记录来自本次当前源文件，也不能从训练 Return 趋势归因协同收益。不同 Episode seeds 和策略同时变化，训练收益不可当作同环境反事实比较。 |
| 判定依据 | 缺失的是结论所需证据；本次已用 initial/latest 权重差证明两个 Actor 确实变化，不能再因日志没有 delta 就声称“Actor 没更新”。多个受控 Run 有相同权重/结果，不当作独立统计重复。 |
| 最小修复方向 | 后续新 Run 增加源码身份和有效 override 快照；现存 Run 保持原样，在结论中标记来源边界。先利用已有 initial/intermediate/latest 做有配对种子的冻结评价，不重新训练补身份。 |

### F08：UAV 动作次数缺少独立实际位移证据

| 字段 | 证据与判断 |
|---|---|
| 代码位置 | `rollout.py:1636,1940–1950,2210–2230`；`reporting.append_dual_timescale_slot_metrics:2161–2178`；真实物理检查 `uav_configuration.py:417,482–571` |
| 事实证据 | MicroTransition 的 movement_count、Episode move/stay、Slot uav*_move 均由 action index!=0 算；正式 Slot 日志不保存前后坐标/每 UAV 距离。真实环境的 movement cost 和 accepted_uav_updates 才由位置差算，但后者没有作为独立运动次数保存到主线 MicroTransition。 |
| 影响 | 当前日志间次数相等只证明同源动作标签一致，无法单凭日志独立复算位移/成本或识别未来新增回退造成的 MOVE 无效。 |
| 判定依据 | 当前候选排除近零移动、执行直接应用、完整 transition 再校验，故当前路径下非零动作对应真实 MOVE；没有已证实标签/物理不一致。报告将其列为可审计性风险，不叫统计计数错误。 |
| 最小修复方向 | 后续仅加实际位移/每 UAV 真实 changed 标记及与动作标签对照，主线 cost 继续使用真实位置。现有数据不能补造未保存坐标。 |

## 现有 Run 证据核对

### Run 身份、完整性与更新次数

全部 7 个 Run 都存在用户列出的 7 类文件。以下“Episode/Update/Slot”是实际行数，并与 summary、latest Checkpoint 已完成计数核对；不是计划 300/75 的抄录。

| Run（均位于 `runs/`） | 有效 Macro LR | Episode / Update / Slot | Micro / Macro 活跃组 | Micro / Macro Actor step | 状态 |
|---|---:|---:|---:|---:|---|
| `alternating_actor_updates_v1_macro_lr_1e-3` | 1e-3 | 268 / 67 / 80,400 | 34 / 33 | 1,360 / 132 | 本地快照未达 300；无 early_stop_reason，不推断中断原因 |
| `alternating_actor_updates_v1_macro_lr_1e-4` | 1e-4 | 300 / 75 / 90,000 | 38 / 37 | 1,520 / 148 | `FORMAL_TRAINING_COMPLETE` |
| `alternating_actor_updates_v1_macro_lr_5e-3` | 5e-3 | 280 / 70 / 84,000 | 35 / 35 | 1,400 / 140 | 本地快照未达 300 |
| `alternating_actor_updates_v1_macro_lr_5e-4` | 5e-4 | 276 / 69 / 82,800 | 35 / 34 | 1,400 / 136 | 默认 LR 对应 Run；本地快照未达 300 |
| `alternating_idmap_controlled_2updates_20261020` | 5e-4 | 8 / 2 / 2,400 | 1 / 1 | 40 / 4 | `CONTROLLED_STOP_AFTER_UPDATES` |
| `cloud_shared_resources_controlled_updates2_20261020` | 5e-4 | 8 / 2 / 2,400 | 1 / 1 | 40 / 4 | 同上 |
| `cloud_shared_resources_logged_controlled_updates2_20261020` | 5e-4 | 8 / 2 / 2,400 | 1 / 1 | 40 / 4 | 同上；另有 Cloud 资源细项日志 |

四个长 Run 和两个 cloud 受控 Run 的 run_config raw hash 与当前正式 TOML 一致。idmap 受控 Run raw hash 为 `2798296ed073e020ab4249c68539e017fc48c01d89c07752973f7f2a2a34e003`，但 TOML 解析结果与当前相同；所有 Run 记录的 semantic hash 为 `2a4f74c8e55d6f1329c790d5265988c4e1bbf5600b1a069570b93cd3a63c01f5`。未将 idmap Run 的不同任务收益当作当前物理实现应有收益。

所有 latest Checkpoint 均为版本 65、Architecture 2、正式策略 Schema；config hashes、summary counters、调度 cursor 一致。Adam 的 Macro Actor step、Micro Actor group step、Micro Critic group step 与日志累计一致。每组 N2/Long 各 50 step、Micro Critic 40 step；Micro 活跃组恰 40 Actor step，Macro 活跃组恰 4 Actor step。四个长 Run 的顺序错误数为 0。5e-3 Run 有共 3 次拒绝候选 LR 尝试，但最终每活跃组仍接受 4 个 Macro step，未把拒绝的尝试计为有效 step。

直接只读比较 initial/latest 的 Actor 参数（Macro 14 个 tensor、Micro 6 个 tensor）如下；各 Run 所有 Actor tensor 均发生变化：

| Run 简称 | Macro Actor L2 delta | Micro Actor L2 delta |
|---|---:|---:|
| 1e-3 | 4.724626 | 3.433676 |
| 1e-4 | 0.725650 | 3.510939 |
| 5e-3 | 15.127557 | 3.279119 |
| 5e-4 | 3.045825 | 3.408374 |
| idmap controlled | 0.366297 | 0.524058 |
| cloud controlled | 0.509739 | 0.538873 |
| cloud logged controlled | 0.509739 | 0.538873 |

这排除了“始终没改参数”；没有每相邻组 Actor 快照，因此没有把初末差解释为每一个中间 step 都被单独验证。

### Reward / Cache / UAV 统计

对 344,400 条 Slot 做流式只读聚合，逐 Run、逐 Episode 核对：

- `G == task_utility-cache_cost-movement_cost` 全部成立（逐 Slot 误差门槛 1e-10）；`cache_cost == 2e-5*cache_download_delay_s` 全部成立。
- 每 Episode 恰有 origin 0..299，未见重复/缺失 Slot。Episode Return 与 Slot G 按相同顺序求和的差为 **0**；total Q/两成本/下载时延聚合差也为 **0**。
- `macro_return` 与 Episode Return 的最大绝对差约 `5.46e-12`，符合 gamma=1 的浮点累积误差。
- Slot `actual_cache_update_count` 的和与 Episode 各节点真实 changed 次数之和相等。所有 Episode 各节点 sampled UPDATE 次数与 actual update 次数相等，`nonzero_candidate_no_effect_rate=0`。Cache count 是改变的**节点数**，不是服务数；成本仅对新增服务 MB 计费，两种口径不应直接相等。
- Slot 的 UAV move 标签计数与 Episode move_count 相等，但同源性限制见 F08。当前代码可证明合法非零动作会实际移动，日志没有坐标，不能声称完成了独立逐次距离复算。
- 默认 LR Run `total_cache_cost` 范围约 3.191–6.102/Episode、`total_movement_cost` 0.109–0.278；整体 G 约数千。单项成本小并不证明 actor 不该移动或更新，应比较受控边际收益。
- `actual_cache_update_rate` 是“有任意节点变化的 Macro frame 比例”；`ALL_UPDATE` 是每 frame 至少一节点变化的模式标记，不能读作每 frame 九节点都更新。默认 Run 每节点平均 KEEP 概率约 0.58，但全局 frame UPDATE 很频繁并不矛盾。

### KL、优势与 Critic

以默认 5e-4 Run 为重点，范围均针对其实际 69 组：

| 指标 | 实际观察 | 能说明什么 |
|---|---|---|
| 首次 Macro 联合 log-prob 回放误差 | 最大 `3.814697e-6` | 低于代码 `5e-6` 门槛；不证明 Micro 同样实测 |
| 活跃 Macro 最终 post-step joint KL | `0.001437–0.018850` | 没达到 0.15 guard；为采样近似 KL |
| Micro approx KL | 最大 `0.011316`，未触发 early-stop | 无明显 KL 失控证据；小负近零来自浮点 |
| 原始 Macro A std | `33.533–197.998` | 未发现整组优势退化为常数 |
| 原始 Micro A std | `33.082–288.786` | 同上；标准化后的约 1 本身不是学习质量证据 |
| 活跃 Macro entropy | `12.748–17.179` | 未见联合 Cache/Holding 概率整体塌为确定性 |
| Micro entropy | `2.627–4.328` | 未见两个 UAV 联合策略整体零熵 |
| 冻结 Macro 组 macro_entropy | 恒写 0 | 是未执行 Actor loop 的占位值，不能据此断言策略塌缩 |
| Actor grad | 活跃组有非零梯度，初末参数变化见上表 | 有效训练存在，但非收益改善证明 |
| Critic targets | hash 前后一致，detached/frozen 记录一致 | 50 step 内目标固定；不保证目标已校准到完整回报 |

`grad_conflict_fraction` 指 **当前/Last1 Macro Critic** 梯度冲突，不是 Cache Actor 与 UAV Actor 梯度冲突。两个 Actor 参数空间分离，本实现没有计算两者共同参数上的梯度夹角；null 或没有该指标不能证明协同无冲突。

### 从已有 Checkpoint 核对 N2 与实现 RTG

未调用环境。仅在指定 Conda 解释器中读取 latest Checkpoint 已保存的 `last1_macro_transition_buffer`，对其完整末尾 Episode 的 frame reward 反向累加（gamma=1），并按已保存 MLP 权重做只读前向：N2 输入 `2*observation-1`、Long 输入 raw observation，两个 hidden 均 SiLU。没有加载训练 Runner、创建策略优化器或反向传播。

| Run / 最后 Episode | frame 数 / Slot 数 | 实现 RTG 起点 | N2 起点预测 | N2 对该实现 RTG RMSE / EV | Long 起点预测 | Long 对该实现 RTG RMSE / EV |
|---|---:|---:|---:|---:|---:|---:|
| 默认 5e-4 / 276 | 228 / 300 | 3396.427 | 1645.103 | 846.041 / 0.554079 | 3377.856 | 97.041 / 0.998500 |
| 完成 1e-4 / 300 | 110 / 300 | 2931.370 | 2121.899 | 597.862 / 0.473410 | 2994.599 | 42.689 / 0.998151 |
| cloud logged controlled / 8 | 162 / 300 | 3263.219 | 72.228 | 1767.311 / 0.007296 | 756.517 | 1291.988 / 0.021482 |

以上是**本次只读重算值**，并非原日志原有的 `pre_update_return_ev`（该字段是 null）。存储奖励为 float32，RTG 与原始 Slot double Return 可有微小差异；预测是本组更新后 Checkpoint 对本组训练样本的结果，不是原 Actor 当时使用的 old value，也不是留出集预测。

这些比较说明“拟合 N2 很好”和“逼近完整实现回报”有明显区别，且 Long 的较好拟合没有进入 Actor。它们不证明 Long 泛化足够好，不能用单 Episode 的实现收益替代条件期望价值，更不构成已验证修复方案。早期受控 Run 两类价值都尚未拟合完整 RTG，不能据此把两组训练诊断当作收敛结论。

### 当前日志能否证明改善或协同

默认 5e-4 Run 前 20 Episode Return 均值约 `3081.128`，末 20 约 `3125.358`；1e-4 完成 Run 前/末 20 约 `3096.260/3072.427`。这些 Episode 使用不同 seed，策略也变化，不能由其增减推断协同有效/无效。Long/N2 拟合 EV 是训练样本指标，参数 delta 和 optimizer step 是执行证据，都不是冻结策略收益检验。

两个 cloud 受控 Run 的 Checkpoint Actor 权重变化与结果一致，不应把它们算作独立重复样本。当前目录下没有可完成“同 seed initial vs trained、固定 Cache vs trained Cache、STAY vs trained UAV”的配对冻结结果。协同收益的符号和显著性仍为**代码静态审计无法确定**。

## C. 八个核心问题的明确回答

1. **参数是否真正分离？** 是。Macro/Micro Actor、N2/Micro/Long Critic 没有共享参数对象；同类节点共享 Actor 参数不同于两个 Actor 共享参数。Micro Actor/Critic 共用 Adam 容器但参数组独立，冻结 Actor 的 state 不推进；Checkpoint 计数一致。
2. **是否优化同一系统级目标？** 两者奖励源与经济目标都是 `sum(Q-Cca-Cq)`。实际 surrogate 因时间尺度、GAE、clip 和 Micro entropy-floor 正则而不同，不能宣称优化 loss 完全相同。
3. **跨模块影响是否真实进入学习目标？** 是。Cache → Micro 新 observation/候选 → 真实可执行节点；UAV → 真实链路/KKT/Q → 下次 Cache observation/候选；同一 r 进入两类优势。参数不互相反传不构成断链。学习强弱见 F01–F04，成功与否仍需评价。
4. **交替更新是否有已证实数据错误/顺序缺陷？** 未发现。实际为跨新 rollout 组交替，非同批先更 Micro 再更 Macro。每个活跃 Actor 的旧概率来自本组冻结的同一自身策略；非活跃 Actor 不变。Last1 是单独的 Critic off-policy 风险（F02），不能移称为 Actor 数据错用。
5. **优势能否支持联合动作学习？** 数学上可以；联合 log-prob、联合 ratio、mask、actions 和优势形状一致，Episode 边界正确。没有节点反事实优势或细粒度协同归因（F03）；Macro N2 的长期校准不足和 Micro hidden holding phase 限制信号质量（F01/F04）。
6. **动作统计是否基于真实变化？** Cache actual count 是前后 Cache row 差异，候选去重和执行保证非零有效，日志支持。UAV 成本确实基于真实位移，但主线 move 次数基于非零动作索引；当前代码保证对应，现有日志不能独立逐次复核实际距离（F08）。
7. **最可能阻碍有效学习的前三项？** 第一 F01：有现有 Run 数值支持的长时域价值滞后，High 设计局限；第二 F03：全局信用分配和独立 Cache 联合分布，Medium 设计局限；第三 F02：旧联合策略 Last1 的强权重和策略滞后，Medium 机制风险。排序是优先调查顺序，不是已证明的因果贡献排序。F04 是紧随其后的表示局限；F07 意味着“当前最差原因”的最终归因证据仍不足。
8. **先修什么、冻结策略能验证什么、什么需要后续实验？** 先修 F05/F06 的确定合同/诊断问题，再补 F01 的完整回报校准证据及 F08 的真实位移记录；已有 Checkpoint 可验证初末策略收益、模块边际/交互、N2 vs 实现 RTG、动作状态一致性。改变 value target、Last1 权重、信用分配或 observation 对学习曲线的影响，必须在后续明确版本的小规模受控训练中验证，冻结评价不能替代训练机制实验。本次未运行这些评价或训练。

## D. 最短可执行修复路线

1. **先修确定问题，保持行为不变。** 将 F05 的 Last1 固定/可配合同说清并消除无效字段；F06 把真正 replay check 移到首 Actor step 前，post-update change 单列。必要时只补主线诊断，不先改奖励/两个 Actor 更新顺序或算法。新增 Run 记录源码身份和有效 LR override（F07），保留旧产物原样。
2. **用现有产物确认学习信号。** 从现有完整 Last1 buffers / 可用 Checkpoint 比较 N2 与实现 RTG 的 bias、RMSE、时间位置切片；不要再只用 N2 EV 代表长期 value 正确。对 F02 的历史样本标明采样策略年龄和与当前 Critic 的 loss/gradient，区分 current/last1 两组。现有日志不足的分项不能回填猜测值。
3. **再安排有界的冻结配对评价，先不重训。** 保持同一物理配置、同一 realization/初始化和采样 seed，比较 initial/intermediate/latest；按现有评价工具能力构造 Cache 初始/训练策略 × UAV STAY/训练策略的 2×2 配对。协同交互可用 `J(C1,U1)-J(C1,U0)-J(C0,U1)+J(C0,U0)`，同时报告各主效应、重复采样不确定性。若现有 CLI 不支持某个混合策略，就明确是待实现的窄诊断能力，不用不兼容 Checkpoint 绕门禁。分别评价 stochastic（匹配训练分布）和 group-aware deterministic，不能混称同一策略。
4. **仅按证据做最小机制改动。** 若 F01 校准问题稳定存在，优先试验更快的长时域 value 监督/传播，保持旧概率/GAE采样合同明确；若 F02 对当前样本有害，再调整 Last1；若 age 切片显示 F04 重要，再版本化增补 Micro phase 信息。F03 的信用改进以冻结边际/交互证据为前提。每次只改一个可归因机制，使用独立配置和输出目录，再决定是否需要正式重训；没有依据直接重跑 300 Episode。

## 执行约束与交付记录

- 唯一新增文件为本报告；目标路径原先不存在，没有覆盖旧报告。
- 未修改业务源码、配置、Checkpoint、数据、已有报告或 Run；保留审计开始时的 4 行 Trainer 既有改动及其他 untracked 文件。
- 未启动训练、冻结环境评价、长时评价、超参搜索、数据生成；未安装依赖，未执行项目全面测试。
- 只读 Python 使用 `C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`，`sys.prefix` 已核实为指定 Conda 环境；用途仅为 TOML/JSONL 聚合、Checkpoint 元数据/权重比较和已存观测的小型 MLP 前向。未创建临时测试脚本或临时文件。
- 未 commit、push 或改 Git 历史；没有需要删除的本次临时产物。
- 本报告是静态执行链审计与现存产物一致性核对，不是新的收敛、泛化或协同成功证明。
