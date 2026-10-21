# Cache / UAV 联合动作机制检查（第四阶段）

日期：2026-10-17  
范围：当前 configs/proposed/system_config.toml 指定的正式多节点因子化 Cache、autoregressive UAV、Macro SMDP / Micro PPO 路径。只读当前源码、配置、Training Episode 20150 与已有日志目录；对 Slot 45–49 执行四种固定配置方案的 20 次 Slot Matching/KKT。未运行完整 Episode、PPO 或其他 Episode。

## 结论摘要

- 正式 Cache 每个 Macro frame 对九个 Cache 节点各采样一个 masked categorical 动作。每个节点的索引 0 是 KEEP，其余是该节点独立生成的合法 UPDATE 配置。九个动作以同一个动作前 observation 为条件独立采样，joint log-prob 是九项 log-prob 之和；同一 frame 可同时更新最多九个节点。
- UAV 的两个 Micro 动作按 UAV0、UAV1 顺序采样。UAV1 的候选、特征和概率以已选 UAV0 位置为条件；rollout 保存该条件，更新时用相同条件重算。Micro 联合 log-prob 为两头 log-prob 之和。
- 采样、Old log-prob 保存和 PPO 更新时重算的动作语义一致。Macro 还在首次 PPO actor 计算时检查整帧 joint log-prob 回放误差不超过 5e-6；Micro 没找到同等的显式回放误差断言，但其候选、mask、动作、UAV0 条件和 group-balanced categorical 与采样路径一致。
- Holding 以每 frame 实际执行 1–5 个 Slot 的 SMDP 转移结算。每 Slot 的系统净收益先按实际持续时间折入 frame reward；duration-aware GAE 和两步 SMDP Critic target 使用相同的 gamma^duration 时间尺度。代码没有发现奖励或 holding 终止错位。
- Slot 45–49 的反事实中，Cache 更新的任务收益跨 Slot 出现：相对 KEEP，任务效用增量出现在 Slot 45、47、49；Cache 一次性成本在 Slot 45 扣除。五 Slot 累计净收益仍比 KEEP/STAY 少 0.017721。预测质心 UAV MOVE 的五 Slot累计净影响为 -0.001688。这组固定动作的 2×2 非加性交互约为 7e-15，即数值零；不能据此说其他 Cache/UAV 组合也无交互。
- 九节点因子化策略把整帧共享的 Macro 优势同时用于各节点动作，没有节点级反事实优势。因此存在多节点信用分配风险，但不是已证实实现错误。当前正式完整 policy_schema 没有对应训练日志；现存短 schema 历史日志不能当作当前配置的更新频数或 joint PPO ratio/KL 证据。

## 1. 当前正式动作及学习合同

### 九节点 Cache 动作

配置指定 MULTI_NODE_FACTORIZED，当前 policy_schema 为 adaptive_dual_timescale_proposed_cache_economic_features_fixed_lr_horizon_mask_v4_multi_node_cache_factorized_v1_per_slot_analytic_task_model_v1_training_p95_reward_norm_v1。实际节点顺序为 SV 21、41、3、39、48，RSU_0、RSU_1，UAV_0、UAV_1，共九个节点。每个节点头与其他节点共享 scorer 参数，但收到该节点候选特征和同一个全局 pre-action observation。

rollout 为每节点构造候选列表：索引 0 是当前完整 Cache 状态（KEEP）；其余候选只改变这一行 Cache，来自当前正式候选生成器，并经过节点缓存容量 / 二值约束。策略独立对九行候选 masked categorical 采样；动作被合成为目标 Cache 后，正式 Slot 配置入口再次校验完整 Cache 形状和容量。因此“每个节点独立做合法单节点候选、把选中行合并执行”与当前因子化动作定义相符。

多节点因子化分布没有全局“最多更新一个节点”的门控。九个头可以同时选 UPDATE；合法候选的 cardinality correction 只对各节点的合法 UPDATE logits 减去该节点合法 UPDATE 候选数的对数，不会形成跨节点互斥。若第 \(j\) 个头 KEEP 概率为 \(p_j\)，在给定观测下联合采样允许 0 到 9 个节点更新，且 \(P(\text{全 KEEP})=\prod_jp_j\)。

候选目标的特征是相对于动作前 Cache 的单节点变化，不会把其他八个头本次同时选中的配置作为条件输入。这是既定因子化策略的语义，不是候选非法；但多个单节点选择的合成结果可能有替代、互补或总成本叠加，节点头无法在采样时直接看到完整联合选择的价值。

### UAV0 / UAV1 条件动作

当前配置打开 uav_unified_candidate_policy 与 uav_autoregressive。每个 UAV 有 STAY（候选 0）和 MOVE 目标候选。先生成并采样 UAV0 候选，再把已选 UAV0 最终位置用于 UAV1 的移动安全过滤、候选特征和策略条件。rollout 保存两头候选特征/mask 和归一化 UAV0 位置条件；evaluate_micro_joint 对 UAV0 使用零条件，对 UAV1 使用保存的 UAV0 条件，并按相同 group-balanced masked categorical 重算两头 log-prob。

Macro 采样保存各 Cache 头 log-prob，joint Cache log-prob 为其和。若 frame 内采样了 holding 终止决策，Macro Old joint log-prob 再加上每个已采样 termination 决策的 log-prob。PPO 更新从 rollout 保存的 action、candidate feature、mask、elapsed slot 与 termination observation 重新评价同一 joint action；首次 actor 计算会核对保存的 joint log-prob，容差为 5e-6。termination padding 不进入 log-prob。

Micro 保存 UAV0、UAV1 两个旧 log-prob 的和；更新重算 UAV0、UAV1 log-prob 后仍取和来计算 PPO ratio。没有发现候选支持、动作索引或 UAV0 条件错位。Micro 代码未发现等同 Macro 的显式 old/new 初始 log-prob 一致性 assert；这属于检查强度差异，静态核对未发现分布语义不一致。

### Holding、Macro 与 Micro 回报

正式最小 / 最大 holding 为 1 / 5 Slot。frame 开始时采样 Cache；每 Slot 完成后，在 elapsed 为 1–4 时用下一 Slot observation 采样 CONTINUE / TERMINATE；到第 5 Slot 强制结束，不再采样终止动作。Episode 终止时直接终止 Macro frame，不额外采样终止动作。

持续 \(d\) Slot 的 Macro reward 是

\[
R_t^{macro}=\sum_{k=0}^{d-1}\gamma^kG_{t+k},\qquad \gamma=1\text{（当前配置）}.
\]

Cache 和 UAV 重配置成本随首 Slot 配置转移进入该 Slot 的 \(G\)，之后 Cache / UAV 保持时不重复扣费。Macro actor advantage 使用 SMDP GAE：TD bootstrap 乘 \(\gamma^d\)，跨 frame trace 乘 \((\gamma\lambda)^d\)。Macro Critic 的正式 two_step_smdp_td target 使用

\[
Y_t=R_t+\gamma^{d_t}R_{t+1}
 +\gamma^{d_t+d_{t+1}}V_{t+2},
\]

terminal 边界按完整 episode 截断。该 target 将 frame 内 reward、frame 实际持续 Slot 数及后续 bootstrap 对齐。

Micro 每 Slot 用完整系统净收益 \(G_t\) 计算普通一步 GAE / return target，并对 UAV0/UAV1 的 joint ratio 使用同一 Micro advantage。Cache 和 UAV 都会改变每 Slot 的 Matching/KKT 及所有任务结果，单 Slot reward 包含全任务的效用，而非只含动作节点本地指标。

### UAV Horizon Mask 修复保留情况

当前 AlignedRolloutStep 带有 prediction horizon mask。Cache 单节点候选和 Cache economic feature 构造均收到该 mask；UAV MOVE 候选枚举及预测服务需求也收到同一个 mask。训练 Slot 45 的 mask 为 [1,1,1,1,1]。本次没有改动源码、配置或该路径。

## 2. Slot 45–49 四分支反事实

### 场景和动作固定

使用 C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe，sys.prefix 确认为 adaptive_cache_mec。读取 Training Episode 20150 的完整本地 realization，并选 origin / rollout Slot 45–49，对应时间 145–149 秒；各 Slot 的真实到达任务数依次为 9、11、15、14、10。四条分支共用相同任务、车辆位置和 UAV 行驶状态输入。

基线 Cache 为该 realization 的正式 uniform-feasible 初始化；初始 UAV 位置为 UAV0 (3364.03, 1119.88)、UAV1 (5179.65, 1710.41)。

- Cache UPDATE 固定为 SV41 行移除服务 8、装入服务 7，目标服务大小 7949.615 MB，低于 12000 MB SV 容量。该目标是 Slot45 当前生成的 SV41 合法 UPDATE 候选之一（动作索引 5；KEEP 为索引 0），只在 Slot45 执行一次，之后 Cache 保持。
- UAV MOVE 在看未来真实任务收益之前，由 Slot45 当前合法 MOVE 候选中的预测需求质心目标选定：只移动 UAV0，从 (3364.03,1119.88) 到 (3378.9263,1118.1194)，位移 15 m；UAV1 保持。四个分支使用同一个固定目标，MOVE 只在 Slot45 执行，之后位置保持。
- 四分支逐 Slot 调用当前正式 execute_configured_slot 和 Matching/KKT。每个方案都重算全任务分配、资源和效用；任务效用、Cache / UAV 成本和 G 取自执行结果。Q 与分配计数覆盖全槽所有任务，表中的时延和能耗为全槽任务的逐任务均值。

| 分支 | Slot | 任务数 | Q | Cache成本 | UAV移动成本 | G | 分配数 SV/RSU/UAV0/UAV1/Cloud | 平均时延(s) | 平均车载上传能耗(J) |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| KEEP + STAY | 45 | 9 | 8.463349 | 0 | 0 | 8.463349 | 1/1/1/2/4 | 2.678418 | 0.088867 |
| KEEP + STAY | 46 | 11 | 10.245097 | 0 | 0 | 10.245097 | 1/0/2/2/6 | 3.258296 | 0.057695 |
| KEEP + STAY | 47 | 15 | 13.917648 | 0 | 0 | 13.917648 | 0/2/2/3/8 | 3.353982 | 0.078908 |
| KEEP + STAY | 48 | 14 | 12.926361 | 0 | 0 | 12.926361 | 0/2/1/1/10 | 3.631103 | 0.066987 |
| KEEP + STAY | 49 | 10 | 9.287509 | 0 | 0 | 9.287509 | 1/0/1/3/5 | 3.354749 | 0.067010 |
| KEEP + MOVE | 45 | 9 | 8.463491 | 0 | 0.002860 | 8.460631 | 1/1/1/2/4 | 2.677971 | 0.088777 |
| KEEP + MOVE | 46 | 11 | 10.245389 | 0 | 0 | 10.245389 | 1/0/2/2/6 | 3.257541 | 0.057544 |
| KEEP + MOVE | 47 | 15 | 13.918072 | 0 | 0 | 13.918072 | 0/2/2/3/8 | 3.353179 | 0.078748 |
| KEEP + MOVE | 48 | 14 | 12.926577 | 0 | 0 | 12.926577 | 0/2/1/1/10 | 3.630665 | 0.066900 |
| KEEP + MOVE | 49 | 10 | 9.287608 | 0 | 0 | 9.287608 | 1/0/1/3/5 | 3.354468 | 0.066954 |
| UPDATE + STAY | 45 | 9 | 8.495820 | 0.099496 | 0 | 8.396324 | 1/1/1/2/4 | 2.575817 | 0.068395 |
| UPDATE + STAY | 46 | 11 | 10.245097 | 0 | 0 | 10.245097 | 1/0/2/2/6 | 3.258296 | 0.057695 |
| UPDATE + STAY | 47 | 15 | 13.929808 | 0 | 0 | 13.929808 | 1/2/2/3/7 | 3.293893 | 0.083711 |
| UPDATE + STAY | 48 | 14 | 12.926361 | 0 | 0 | 12.926361 | 0/2/1/1/10 | 3.631103 | 0.066987 |
| UPDATE + STAY | 49 | 10 | 9.324654 | 0 | 0 | 9.324654 | 2/0/1/2/5 | 3.227719 | 0.051365 |
| UPDATE + MOVE | 45 | 9 | 8.495961 | 0.099496 | 0.002860 | 8.393605 | 1/1/1/2/4 | 2.575369 | 0.068306 |
| UPDATE + MOVE | 46 | 11 | 10.245389 | 0 | 0 | 10.245389 | 1/0/2/2/6 | 3.257541 | 0.057544 |
| UPDATE + MOVE | 47 | 15 | 13.930232 | 0 | 0 | 13.930232 | 1/2/2/3/7 | 3.293090 | 0.083550 |
| UPDATE + MOVE | 48 | 14 | 12.926577 | 0 | 0 | 12.926577 | 0/2/1/1/10 | 3.630665 | 0.066900 |
| UPDATE + MOVE | 49 | 10 | 9.324753 | 0 | 0 | 9.324753 | 2/0/1/2/5 | 3.227439 | 0.051309 |

分配列按 SV、RSU 总数、UAV0、UAV1、Cloud 排列。Slot45 的 Cache UPDATE 把一个任务的 SV 节点由 SV21 转为 SV41。Slot47、49 的 UPDATE 分支分别比 KEEP 多一个 SV41 命中任务、少一个 Cloud 任务；表内效用与平均延迟/能耗包含这类重新 Matching 后所有任务的资源竞争结果。所选 UAV 移动没有改变各 Slot 的节点分配数量，效用和任务指标变化很小，但完整任务结果仍已重算。

### 1–5 Slot 累计 G 与交互项

令 \(G_{KS},G_{KM},G_{US},G_{UM}\) 分别表示 KEEP/STAY、KEEP/MOVE、UPDATE/STAY、UPDATE/MOVE。每个 horizon 从 Slot45 累加；每条分支在首 Slot 扣除适用的一次性重配置成本。定义 2×2 交互项

\[
I_H=G_{UM}^{(H)}-G_{US}^{(H)}-G_{KM}^{(H)}+G_{KS}^{(H)}.
\]

| 累计 Slot 数 H | KEEP + STAY | KEEP + MOVE | UPDATE + STAY | UPDATE + MOVE | 联合相对 KEEP/STAY | 交互项 \(I_H\) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.463349 | 8.460631 | 8.396324 | 8.393605 | -0.069744 | < \(2e-15\) |
| 2 | 18.708446 | 18.706020 | 18.641421 | 18.638994 | -0.069452 | < \(4e-15\) |
| 3 | 32.626094 | 32.624091 | 32.571228 | 32.569226 | -0.056868 | < \(8e-15\) |
| 4 | 45.552455 | 45.550668 | 45.497589 | 45.495802 | -0.056653 | < \(8e-15\) |
| 5 | 54.839964 | 54.838276 | 54.822244 | 54.820555 | -0.019409 | < \(8e-15\) |

对 KEEP/STAY，Cache UPDATE 相对任务效用 Q 的逐 Slot 差为 Slot45 +0.032471、Slot46 0、Slot47 +0.012160、Slot48 0、Slot49 +0.037145；五 Slot 总 Q 增量 +0.081775，扣除一次性 Cache 成本 0.099496 后为 -0.017721。因此单 Slot 净收益显著为负，但随着缓存服务后续被复用，差距在 Slot47、49 缩小。此五 Slot 局部窗口不能推出更长时间跨度下会否回本。

KEEP 条件下 UAV MOVE 的五 Slot 总 Q 增量约 +0.001172，小于一次性移动成本 0.002860，累计净差为 -0.001688。两种 Cache 状态下 MOVE 的边际增益几乎相同，故本次测得非加性交互为浮点舍入误差量级。系统存在结构性耦合通道（Cache 改变可行 Edge 候选，UAV 位置改变无线效用，二者均进入同一全任务 Matching 与 Q），但当前预测质心与这个 Cache 目标的这一个场景没有测到明显耦合收益。

## 3. 多节点信用分配与现有日志

正式九节点 Macro actor 使用每个 Macro frame 同一个归一化优势训练联合 PPO ratio：

\[
\rho=\exp\left(\sum_{n=1}^{9}\log\pi(a_n|s)-\sum_{n=1}^{9}\log\pi_{old}(a_n|s)
 +\sum_{k\in termination}\Delta\log\pi_k\right).
\]

框架没有对每个节点估计独立 Q / counterfactual advantage；同一全局 frame 回报及状态价值会共同反馈给所有选中的节点头。节点动作因子化本身符合实现合同，但当多个节点同时更新、且候选收益存在交互时，Credit 不能直接区分每个节点对 G 的边际贡献。Joint ratio 的 log-ratio 又会累加九个节点项和 holding termination 项；这是 PPO 的联合动作定义，不是 ratio 计算错位。

当前源码可以记录 Macro frame 实际更新节点数分布、节点 sampled/actual UPDATE 频率、逐节点 KEEP 概率，以及 Macro joint ratio、duration 分组 log-ratio 和 joint KL。但 runs/ 中没有当前完整 policy_schema 的 run_config / manifest / metrics。现有多节点历史日志使用较早的短 schema adaptive_dual_timescale_proposed_cache_economic_features_fixed_lr_horizon_mask_v4_multi_node_cache_factorized_v1，不包含当前 schema 后缀所标明的 PER_SLOT_ANALYTIC task model / training P95 reward normalization 合同；本次不把它们冒充当前正式动作频率、joint PPO ratio 或 KL。当前实际训练中每 frame 更新节点数、KEEP 概率及 Macro ratio/KL 均为 UNRESOLVED。

因此，多个节点同时 UPDATE 在动作分布中明确允许，但当前训练轨迹的实际频率没有可用的同 schema 日志证据。多节点 credit assignment 风险是结构性判断，不是从该 Slot 反事实或旧版本历史训练曲线验证出的经验结论。Micro 也给 UAV0 / UAV1 使用单一 joint advantage，存在动作间信用共享，但本次没有发现不一致或特定故障证据。

## 4. 事实、测量、限制与判断

### 静态代码确认

- 九个 Cache categorical 的采样、保存、mask 和 PPO 重算一致；joint log-prob 采取各节点和 termination decisions 的总和。Macro 有首次 replay error guard。
- UAV0 / UAV1 顺序、候选合法性、安全过滤、存储的 UAV0 条件和 PPO 重算顺序一致。
- Holding termination 发生在执行当前 Slot 后；帧 reward 汇总实际持续 Slot 的 G，Macro GAE 和 N2 Critic target 按实际 duration 折扣。Micro 用逐 Slot reward / GAE。
- 当前 Cache 候选与 UAV 预测候选路径均传递 prediction horizon mask；该修复未修改。
- 九节点多头策略允许联合多个 UPDATE；同一 frame 共享全局优势，未实现节点级反事实优势。

### 小规模实测

- Episode 20150 Slot45–49，共四个动作组合、20 次真实单 Slot Matching/KKT；全部 Slot 可行。未运行 Episode 或 PPO。
- 每个分支的任务、车辆轨迹及其他初始 Cache/UAV 配置固定，Cache、UAV 只在首 Slot 动作，后续状态保持。成本只在改变状态的首 Slot 计入。
- 实测 Cache 服务的收益存在跨 Slot 差异；该目标的首 Slot收益不抵一次性成本，五 Slot内复用后差距缩小但仍未回本。
- 本次一组 Cache / MOVE 目标的五个累计交互项均处于浮点误差范围；不把此结果外推为不存在全局收益耦合。

### 算法理论限制及尚未证实

- 同一宏优势作用于九个独立节点头，难以判断哪个节点动作产生了全局回报；当前 exact-schema 训练日志缺失，无法量化 simultaneous UPDATE 频率、KEEP 概率或 joint ratio/KL。
- 因子化策略可组合九个基于单节点特征选出的候选，但没有直接建模动作间的联合回报；这是信用和组合搜索能力限制，不等于动作非法或 PPO ratio 错误。
- 仅 Episode 20150 的 Slot45–49、一个 Cache target 与一个 prediction-derived UAV target，不能估计其他位置、其他服务、其他 UAV 目标或长期回报。
- Micro 旧 log-prob 回放未发现显式训练前一致性断言；静态采样 / 重算路径一致。本次没有单独运行训练器或策略概率单测。

**是否需要修复算法：** 当前未发现动作语义、old/new probability、SMDP reward 或 duration target 的已证实错位，故不要求立即修复算法实现。已证实的风险是九头共享优势的信用分配难点；应先取得与当前完整 policy_schema 对应的 rollout / update 日志，统计同时更新节点数、KEEP 概率与 joint PPO ratio/KL，再决定是否需要结构调整。此结论不表示因子化联合动作的学习一定有效。

## 5. 检查文件和运行边界

- Cache 节点候选 / horizon mask：src/ppo_simple/rollout.py:560-681。
- UAV MOVE 候选 / horizon mask：src/ppo_simple/rollout.py:684-738；两头顺序执行与保存 UAV0 condition：src/ppo_simple/rollout.py:1290-1508。
- Factorized Cache sampler 和 Macro replay：src/ppo_simple/policy.py:482-613,1092-1320；UAV sampler / joint evaluator：src/ppo_simple/policy.py:1381-1562。
- Macro / Micro buffers、duration 合同：src/ppo_simple/rollout.py:104-365,187-370,2242-2280。
- Macro SMDP target：src/ppo_simple/macro_targets.py:10-66；PPO 更新及 joint log-prob replay guard：src/ppo_simple/trainer.py:1412-1555,1637-2055。
- 正式目标配置：configs/proposed/system_config.toml:514-577；逐 Slot 配置执行：src/decision/slot_environment.py:1407-1628。
- Python 环境：C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe，sys.prefix 已核实。共执行 20 次 Slot Matching/KKT；没有临时脚本或临时数据文件，没有启动 PPO、完整 Episode、大规模实验。

## 最终标记

    CACHE_JOINT_ACTION_CONTRACT = YES
    UAV_AUTOREGRESSIVE_CONTRACT = YES
    MACRO_SMDP_REWARD_CONSISTENT = YES
    MACRO_PPO_RATIO_CONSISTENT = YES
    MICRO_PPO_RATIO_CONSISTENT = YES

    CACHE_UAV_REWARD_INTERACTION = YES
    MULTI_NODE_CREDIT_ASSIGNMENT_RISK = YES
    HOLDING_VALUE_MECHANISM_CONSISTENT = YES

    ALGORITHM_REPAIR_REQUIRED = NO

    SOURCE_CHANGED = NO
    FORMAL_CONFIG_CHANGED = NO
    FORMAL_DATA_CHANGED = NO
    PPO_TRAINING_EXECUTED = NO

CACHE_UAV_REWARD_INTERACTION=YES 指代码结构存在非线性共享 Matching / Q 耦合通道；不表示本次四分支已观测到非零交互。本次观测的 2×2 交互项近零。MULTI_NODE_CREDIT_ASSIGNMENT_RISK=YES 是九个独立动作头共用 frame advantage 的结构性风险；当前 schema 训练统计 UNRESOLVED。ALGORITHM_REPAIR_REQUIRED=NO 表示未发现本次检查范围内的已证实实现错误，不表示已有同 schema 训练证据证明信用分配充分。
