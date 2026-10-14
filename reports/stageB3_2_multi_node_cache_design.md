# Stage B3-2：多节点独立Cache UPDATE架构设计与实施准备

## 结论摘要

建议将Macro Cache动作改为**共享参数的9个节点级分类分布**：每个Macro决策时刻，对5个SV、2个RSU、2个UAV分别从“KEEP + 该节点当前正式候选生成器提供的合法目标配置”中独立采样。所有节点使用同一个全局Observation编码器和候选Scorer；节点身份、完整当前缓存布局及候选经济特征进入评分。9个选择合并为一次完整Cache矩阵更新，并在该Macro frame内共同保持。Holding仍为全局1–5槽，UAV继续使用原有独立Micro动作。

当前单节点限制来自PPO动作与候选包装，而非Matching接口：rollout将全局候选整理成一个masked categorical，且每个UPDATE候选只改一行；正式`execute_configured_slot`已经能够接收最终完整Cache矩阵，推导逐节点变化标记，并将矩阵交给一次Matching/KKT。Cache成本在当前公式中按节点求和，Reward可一次扣除所有实际变化成本。

但正式模型没有Cache镜像下载的回传链路、并发带宽、传输时延或完成规则。因而“9个动作及成本可以组合”不等于“9个节点并发下载的物理模型已定义”。在并发更新的物理语义由项目决策确定前，不应将多节点动作作为已获物理支持的正式系统模型投入训练。

## 1. 当前机制核对

| 检查项 | 当前正式实现 | 对多节点设计的含义 |
|---|---|---|
| Macro Cache动作 | `rollout._cache_candidates` 生成一个全局候选表：候选0为KEEP，其余每个候选严格只改一个Cache节点；`policy.sample_macro`对整张表做一个masked categorical | 一次采样至多选择一个UPDATE；需改为每节点一个分类分布 |
| 候选合法性 | `enumerate_all_cache_target_configurations` 按节点枚举容量可行目标；正式proposal selector最多保留64个UPDATE目标/节点 | 保留现有逐节点容量和候选选择规则。实际动作空间是KEEP加至多64个正式UPDATE proposal，并非当所有可行配置超过64时把它们全部送入网络 |
| Cache执行接口 | `execute_configured_slot`接收完整前后Cache矩阵，并按每一行变化推导0/1 trigger向量；`build_policy_target_configuration`校验完整矩阵及各节点容量 | 无需为“表示多个节点最终状态”重写接口；节点候选和联合动作的rollout/buffer格式需要改造 |
| Cache更新成本 | `cache_update_cost`按节点计算后求和。对节点i，成本为`w_ca·(z_i·eta_i + lambda_ca·xi_i·added_MB_i)`；删除内容不计入added MB，但实际有配置变化时仍触发固定成本 | 当前成本会覆盖所有实际更新节点，且在一个Slot只从旧矩阵到最终矩阵计一次；这是目标函数的可加记账，不包含节点间资源竞争 |
| Reward | 正式Reward为`Q - C_cache - C_UAV` | 完整Cache矩阵执行后，现有Reward会扣除公式所算的联合Cache成本 |
| 镜像加载物理过程 | 配置将`xi_ca`说明为等效/相对loading系数，不是实测J/MB；没有Cache下载速率、回传带宽、下载时延或并发调度状态 | 并发加载的物理可行性尚无定义，不能由加法成本推导 |
| Matching/KKT | `execute_configured_slot`把最终完整Cache矩阵传给`optimize_task_matching`；后者按该矩阵判断缓存命中并一次优化整个Slot | Cache矩阵层面兼容多个节点同时变化，无需逐节点重复Matching/KKT |

具体位置：`src/ppo_simple/rollout.py::_cache_candidates`、`src/ppo_simple/policy.py::sample_macro/_macro_cache_logits`、`src/ppo_simple/primitive_control.py::_target_cache_proposals_with_features`、`src/decision/slot_environment.py::execute_configured_slot/build_policy_target_configuration`、`src/common/system_model.py::cache_update_cost`、`src/decision/task_allocation.py::optimize_task_matching`。

## 2. 建议的9节点动作与数据结构

在Macro状态`o_t`和决策前Cache矩阵`C_t∈{0,1}^{9×F}`下，对每个节点i构造本节点动作集合：

`A_i(o_t,C_t) = {KEEP_i} ∪ {a_i,k: 当前正式候选生成/筛选器给出的合法目标配置}`。

每个节点候选表从同一个未修改的`C_t`生成，候选目标携带节点ID和完整目标行配置；候选0恒为KEEP。为batch训练，可用以下对齐张量：

- 候选特征：`X ∈ R[B,9,Kmax,200]`；
- 合法掩码：`M ∈ {false,true}[B,9,Kmax]`，每个节点的KEEP位置均有效，padding位置无效；
- 节点动作索引：`a ∈ N[B,9]`；
- 行目标配置/候选身份：按节点、候选索引保存目标Cache行或稳定配置键；
- 行旧log-prob：`logp_old_node ∈ R[B,9]`，并另存合法候选数量与行为分布摘要。

当前`target_candidate_capacity=64`控制每个节点选出的UPDATE proposal数量。因此`K_i≤65`（至多64个UPDATE加KEEP）；不能把该数字误解为所有物理可行目标都必然进入PPO。容量校验和当前候选枚举/筛选语义保持不变。

采样后从`C_t`复制得到`C_{t+1}`，对每个i仅以所选候选的目标行替换该行。trigger严格定义为`z_i=1[C_{t+1}[i] != C_t[i]]`。校验每行目标合法且`z_i`与真实行变化逐项一致，然后把组合后的矩阵一次交给现有Slot执行接口。这样多个节点可以在同一Macro动作中更新，但候选生成不会在中途把前一个已采样动作当成新的基线，避免采样顺序改变候选语义。

## 3. 共享Cache Actor与正式联合PPO目标

### 共享参数动作分布

当前正式Observation为400维，Macro候选特征为200维（181维基础候选特征加19维经济特征）。保留这一200维候选特征作为起始设计：候选含节点索引/类型、目标配置、配置差异、Cache成本和预测新增/移除需求等现有字段；每个节点的评分还接收同一全局Observation表示。采用一个共享候选编码器与共享Scorer，不为9个节点创建各自独立参数集。节点身份已在特征中编码；所有节点都能利用全局Cache布局及预测状态。

每节点输出合法候选masked logits，并对该节点合法UPDATE候选数独立应用当前的基数校正语义（减去该节点`log(N_i^UPDATE)`）。KEEP始终有效，若节点无合法UPDATE，分布退化为KEEP概率1。给定全局状态与各自候选表，最小方案将9个节点动作条件独立采样：

`π_cache(a_t|o_t) = ∏_{i=1}^{9} π_i(a_{t,i}|o_t,X_{t,i},M_{t,i})`。

这不是9套互不感知状态的网络：每个节点使用同一个全局编码和共享Scorer。该最小因子分解不显式表达节点动作之间的任意条件相关性；联合收益仍由真实Slot Reward反馈。暂不另加自回归节点顺序或新损失，以控制本次架构改动范围。

### 联合log-prob和clip目标

对每个Macro transition保存每节点的旧log-prob；同时保留每次实际采样的Holding termination旧log-prob。联合Cache与Holding策略的log-prob为：

`log π_old,joint = Σ_i log π_old,i(a_i|o,X_i,M_i) + Σ_r log π_old,term(z_r|o_r, elapsed_r)`。

新策略同样按完全相同的节点顺序、候选ID和行为时mask重算。选定的PPO目标沿用当前正式joint surrogate：

`ρ_joint = exp(log π_new,joint - log π_old,joint)`

`L_actor = -E[min(ρ_joint·A_macro, clip(ρ_joint,1-ε,1+ε)·A_macro)]`。

9个Cache分量与当前Holding序列共享同一个Macro SMDP优势`A_macro`和一个Macro Critic。继续用实际Duration的SMDP-GAE、帧内按`gamma^elapsed`累计Reward、跨帧`gamma^duration`折扣；当前正式配置为`gamma=1.0`、`gae_lambda=0.95`、Holding范围1–5槽，不更改该时间尺度。Termination仍用每Slot后的Observation和已持有槽数决策。UAV动作仍留在独立Micro Actor/Critic及Micro GAE，不并入Cache联合概率。

**不采用逐节点分别clip的替代目标。** 逐节点目标会分别使用`ρ_i=exp(logπ_new,i-logπ_old,i)`并分别clip后求和；这通常不等于对乘积形成的`ρ_joint`做一次clip，特别是`clip(∏ρ_i)`不等于`∏clip(ρ_i)`。前者是另一个PPO目标定义，不是正式联合动作的重要性采样比率。本方案只采用上面的联合Ratio/联合Clip，并沿用当前Macro KL保护机制。

最少新增/保留诊断：各节点KEEP/UPDATE频率、候选数与UPDATE概率、候选熵、节点动作log-prob/ratio、联合ratio与联合clip fraction、联合与按节点KL、同时更新节点数、每节点Cache成本/装入MB、Macro优势/Duration、Holding终止概率，以及Actor/ Critic loss。联合ratio可能随9个动作分量相乘而更分散，必须观察联合ratio/KL/clip，而不能只看各节点单独指标。

## 4. 联合Cache更新的物理模型边界

当前正式公式已经支持将多个节点的**抽象重配置成本**相加，并将最终缓存矩阵立即用于该Slot Matching/KKT。它没有定义以下物理语义：

1. 服务镜像从Cloud、RSU还是其他仓库下载，及各节点到内容源的链路；
2. 每个节点的加载回传带宽、是否共享同一瓶颈带宽、多个更新之间如何分配；
3. 下载时延和更新能否在1秒Slot内完成，未完成时Cache何时生效；
4. 更新流量是否竞争任务卸载上行/回传资源，或仅使用独立管理信道；
5. Cache版本在传输过程中的可用状态、部分更新失败和多节点并发调度规则。

所以当前最多能确认`CACHE_COST_ADDITIVITY=YES`（目标成本账面可加），不能确认`SIMULTANEOUS_UPDATE_PHYSICAL_MODEL=SUPPORTED`。在实施端启用并发执行前，需要明确采用哪类模型：继续使用既有的抽象即时重配置系统模型，并明确其边界假设；或另行定义带宽、传输时间和共享资源的物理扩展。本报告不替用户选择、不填造带宽或时延参数。

## 5. 250米预测服务性特征

当前200维特征中的预测需求采用Cache候选评分服务半径形成的软可服务性权重；B3-1中RSU_1的250米半径使服务2新增预测需求为零。该尺度不是Matching/KKT的物理链路硬可行条件，也不应在多节点动作改造中被误认为物理覆盖阈值。本阶段保留全部现有200维特征，不改变250米参数。

若后续要研究更一致的候选价值代理，可在不改变正式Reward的前提下，利用GRU预测需求和预测位置，对候选节点逐车辆复用Matching所用的实际物理链路可行判定及正式通信时延/车辆能耗/Task Utility归一化，估计相对于Cloud的预测任务效用差；再对共享CPU/带宽竞争使用预测任务集合调用正式KKT作更完整但更昂贵的候选评价。任何不执行共享KKT的简化值都应标为预测代理，不能称作候选的精确净收益或真实动作Q值。此次不改特征或参数。

## 6. 后续正式代码实施范围

| 文件 | 后续作用 |
|---|---|
| `src/ppo_simple/primitive_control.py` | 复用现有单节点合法目标枚举与容量筛选，为每个节点打包KEEP加本节点候选并维持200维特征对齐；不放宽候选合法性 |
| `src/ppo_simple/rollout.py` | 用固定9节点顺序生成各节点候选、保存动作/掩码/配置身份/旧log-prob，组合9个所选目标行并在Macro frame起点只执行一次 |
| `src/ppo_simple/policy.py` | 实现共享节点候选Scorer、9个masked categorical、联合Cache log-prob/熵的求值和向量动作校验；保持单个Macro Critic及Holding head |
| `src/ppo_simple/trainer.py` | 更新Macro batch/padding与动作重算，使用联合log-prob ratio和现有joint clipping、KL保护；增加必要逐节点指标及Checkpoint合同字段 |
| `src/ppo_simple/reporting.py` | 输出每节点决策、更新数、成本、候选/概率及联合PPO诊断 |
| `src/common/project_config.py`、`configs/proposed/system_config.toml` | 更新正式策略Schema/动作合同标识；保留容量、候选上限、Reward及gamma/GAE值 |
| `src/decision/cache_configuration.py`、`src/decision/slot_environment.py` | 现有候选枚举、容量检查及完整矩阵执行已可复用；只有新增联合动作验证或未来物理加载规则确实需要时才修改，不重写Matching/KKT |

## 7. 新旧架构与Checkpoint合同

当前正式单节点架构为`architecture_version=32`、`CHECKPOINT_VERSION=59`，schema为`...v3_no_tv_local`；Stage6旧权重更早且已确认不兼容。多节点动作改变动作分解、batch shape、old/new log-prob语义及动作合同，不能复用Stage6旧权重，也不应把当前32/59单节点策略静默部分加载为多节点策略。

实施时应采用新的、明确标记多节点因子策略的Policy Schema（例如`...v4_multi_node_cache_factorized_v1`），把architecture version递增到33、Checkpoint version递增到60，并在配置、trainer、run manifest、checkpoint payload、loader及测试中保持一致。训练从新架构随机初始化开始；旧权重仅保留作历史数据，不作为正式初始化或评价模型。

## 8. 实施后的最小功能验证

1. 每个节点的KEEP有效、非法候选及padding概率为零；候选概率和为1；节点目标Cache逐行满足容量与服务位合同；候选身份能由节点ID和完整目标行稳定重建。
2. 确定性测试全部KEEP、仅一个节点UPDATE、多个节点UPDATE三种组合：最终矩阵与9个动作完全一致，未选节点保持不变，trigger恰好等于变化行。
3. 对正式Cache成本验证“多节点联合成本等于相同变化的逐节点成本和”，且Reward中成本只扣一次；不借此声称物理下载可并发。
4. 组合矩阵只调用一次正式Matching/KKT；Edge缓存、链路、CPU/带宽约束及有限Reward维持现有合同。
5. 相同权重重算动作时，每节点old/new log-prob一致、`rho_joint=1`、联合Ratio和clip计算有限；padding不进入log-prob和；termination的有效mask和SMDP帧累计不变。
6. 新Checkpoint能按新Schema/version完整保存并重载；Stage6及旧单节点Checkpoint必须按版本/合同拒绝。

上述是后续实施验收项，本任务没有运行测试、仿真或训练。

## 状态

```ini
STAGE_B3_2 = PARTIAL
MULTI_NODE_ACTION_DESIGN = COMPLETE
SIMULTANEOUS_UPDATE_PHYSICAL_MODEL = NEEDS_DEFINITION
CACHE_COST_ADDITIVITY = YES
MATCHING_KKT_JOINT_CACHE_COMPATIBILITY = YES
PPO_JOINT_POLICY_DEFINED = YES
LEGACY_CHECKPOINT_REUSE = NO
NEXT_STEP = 先确定并发Cache更新采用抽象即时重配置假设还是新增共享回传资源/加载时延模型；定义后再实施新Schema的9节点联合PPO
SYSTEM_MODEL_CHANGED = NO
PPO_CODE_CHANGED = NO
PPO_TRAINING_EXECUTED = NO
```
