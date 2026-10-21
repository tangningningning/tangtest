# 第七阶段 PPO 学习与收敛根因诊断

日期：2026-10-17  
范围：当前正式 PROPOSED 主线源码、system_config.toml、runs 元数据和第四阶段 Cache/UAV 固定场景反事实。只读诊断；未运行 PPO、Validation、Test、完整 Episode 或新 Matching/KKT 场景。

## 1. 正式策略与 Reward 合同

当前配置和代码给出的正式合同如下。

| 项目 | 当前值与代码位置 |
|---|---|
| Policy Schema | adaptive_dual_timescale_proposed_cache_economic_features_fixed_lr_horizon_mask_v4_multi_node_cache_factorized_v1_per_slot_analytic_task_model_v1_training_p95_reward_norm_v1，system_config.toml:537；policy.py 的 MultiNodeFactorizedActorCritic 同一架构 ID |
| Architecture / Checkpoint | Architecture version 35（policy.py:1651）；多节点 Checkpoint version 62（trainer.py:87、:955–960） |
| Cache 动作 | MULTI_NODE_FACTORIZED，九个节点各有 KEEP 和合法 UPDATE categorical；配置见 system_config.toml:539，正式 Runner 核验见 mainline_runner.py:180–193 |
| 任务模型 / Reward | PER_SLOT_ANALYTIC；slot_environment.py:612–620 对每个本槽参与任务累计完成指示、时延惩罚和车辆上传能耗惩罚 |
| 归一化常数 | T_ref=30.473728355325758 s，E_ref=5.1574642506299275 J；system_config.toml:423–424 |
| Reward 权重 | omega_delay=0.6，omega_energy=0.4；system_config.toml:372、:376 |
| Holding | 自适应 1–5 Slot；system_config.toml:543–545。Macro frame 汇总实际持续期间的逐 Slot G；gamma=1、gae_lambda=0.95，见 :554–555 |
| 更新批次 | 4 个完整 300-Slot Training Episode 共用冻结策略采样，再执行一次更新；配置见 :562，trainer.py:1332–1347、mainline_runner.py:4085–4164 |
| 更新顺序 | 同批数据先更新 Micro，再更新 Macro；trainer.py:1352–1356 |
| Horizon Mask | 当前 Cache 单节点候选、Cache economic features 和 UAV MOVE 候选仍传递 prediction_horizon_mask；rollout.py:485–550、:585–644、:684–720。第四阶段修复仍在正式路径 |

当前逐 Slot 实际任务效用可写为

$$
q_i(t)=\mathbf{1}\{i\text{ 在本 Slot 完成}\}
-0.6\frac{d_i(t)}{T_{\mathrm{ref}}}
-0.4\frac{e_i^{\mathrm{upload}}(t)}{E_{\mathrm{ref}}},
\qquad Q(t)=\sum_{i\in\mathcal A_t}q_i(t),
$$

其中 $d_i(t)$ 为本槽累计执行时延，$e_i^{\mathrm{upload}}(t)$ 为本槽车辆上传能耗，$\mathcal A_t$ 为本槽参与执行的任务。系统净收益为 $G(t)=Q(t)-C_{\mathrm{ca}}(t)-C_q(t)$。实现位置为 slot_environment.py:612–620；未将任务数固定归一化，所以原始 Q 会随参与任务数和完成情况改变。

Macro 以实际 holding duration $D_k\in[1,5]$ 汇总

$$
R_k=\sum_{j=0}^{D_k-1}\gamma^jG(\tau_k+j).
$$

Macro Actor 使用按实际 duration 折扣的 SMDP GAE。Macro Critic 的 Two-step target 对非 terminal frame 为

$$
Y_k=R_k+\gamma^{D_k}R_{k+1}
+\gamma^{D_k+D_{k+1}}V_{\mathrm{old}}(k+2),
$$

遇到 terminal 时截断。build_macro_n2_target 实际按 $D_k$ 与 done 边界实现；对应 GAE 在 rollout.py:2242–2293。静态代码未发现 holding 长度与两类目标的折扣错位。

## 2. 同 Schema 训练与收敛证据

使用 adaptive_cache_mec 环境解释器 C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe 扫描 runs 下现有元数据。36 个 run_config.toml、36 个 run_manifest.json 和 36 个 run_summary.json 中，当前完整 Schema 命中数为 0。run_config 按 Schema 分布为：旧 v2 16 个、旧多节点 v4（缺少 PER_SLOT_ANALYTIC / training P95 reward normalization 后缀）18 个、独立 sparse v5 2 个。32 个目录有完整训练日志文件组，但都不是当前完整 Schema。

因此当前没有可匹配的 episode_metrics.jsonl、update_metrics.jsonl、critic_diagnostics.jsonl、训练 Checkpoint 或对应配对 Validation 结果。旧 v2、旧 v4 和 sparse v5 日志只用于确认它们与当前合同不同，不参与当前策略频率、KL、Critic 质量或收敛判断。第四阶段报告 reports/cache_uav_joint_stage4.md 的 Episode 20150 Slot 45–49 是当前合同下的固定配置反事实，可用于观察局部动作收益，但不是当前完整 Schema 的训练轨迹或收敛评估。

## 3. Actor 更新路径

### Macro Actor

- 当前 Macro Actor 使用每个 Macro frame 一个共享归一化 SMDP Advantage，对九个 Cache 节点动作与该 frame 内真实采样的 holding termination 序列执行联合 PPO clipped surrogate。joint log-prob 是各 Cache 节点 log-prob 与 termination log-prob 之和；第四阶段已核对采样、Old log-prob 和更新重算语义一致。
- 配置 Macro Actor 学习率为 5e-4（constant）、update_epochs=4、macro_full_batch=true，因此每个 update 最多有 4 个接受的 Macro Actor optimizer steps。macro_target_kl=0.10；实现把 1.5 倍阈值 0.15 用于更新前停止判断以及更新后 joint KL 接受门。
- 每个候选步先保存 Actor 参数和 Adam 状态，再按学习率比例 1、0.5、0.25、0.125、0.0625、0.03125、0.015625 回退尝试；只接受参数/优化器有限且 post-step joint KL 不超过 0.15 的步。全被拒绝时恢复参数与优化器并提前停止。实现同时记录 accepted/attempted/rejected step、实际学习率比例、joint/cache/termination pre/post KL、joint clip fraction、熵和梯度范数，见 trainer.py:1637–1645、:1830–2050、:2085–2290、:2423–2480。
- 该机制允许“Actor 有梯度但因 joint KL 回退后只接受极小步长”或“整次 Macro Actor 无接受步”。当前没有 exact-schema 日志，无法统计少步、回退或 KL 提前停止的频率。PPO-UPDATE 控制台诊断标注 ActorParameterDelta=NOT_MEASURED；update 日志里的 KL 和概率变化可反映策略分布变动，但不能当作权重参数差值。

### Micro Actor

- Micro ratio 基于 UAV0、UAV1 autoregressive joint log-prob 与同一 rollout 保存的 Old joint log-prob；Micro Advantage 来自每 Slot 完整系统 G 的 GAE。
- Micro Actor 学习率为 3e-4、Micro Critic 学习率为 3e-3；update_epochs=4、num_minibatches=10，每次 update 最多 40 个 Actor/Critic minibatch steps。micro_target_kl=0.02，当 minibatch 估计 KL 超过 0.03 时停止后续 minibatch。代码记录实际步数、approx KL、clip fraction、熵、梯度范数、Critic RMSE/EV/correlation 和 UAV Stay/Move 概率，见 trainer.py:1412–1618。
- 完整 Schema 日志缺失，Micro Actor 是否实际充分更新、是否被 KL 提前截断无法判断。

### Macro joint KL 与因子化动作

Macro joint ratio 对九个 Cache categorical 和所有已采样 termination 项先相加 log-ratio 再取指数；clip epsilon=0.2，KL guard 看 joint KL，同时报告 Cache 与 termination aggregate KL。九个 Cache 项不是九个独立 PPO ratio 再各自剪切。维度增加会放大 joint log-ratio 的波动，因此限制有效联合步长是合理的风险机制；当前无日志可判断它是否真的限制训练。

rollout 能保存逐节点 KEEP 概率、每 Frame 实际 UPDATE 节点数分布、joint Macro 概率与 entropy（rollout.py:874–881、:1026–1108、:1735–1991）。不过当前 Schema 没有这些实际训练记录。Macro entropy_coefficient=0.0，因此没有普通熵 bonus；因子化策略仍可同时更新多个节点，且没有节点级反事实 Advantage。KEEP 或多节点 UPDATE 的实际占比均不能从当前可用训练结果推断。

## 4. Critic 与 Advantage 质量

### Macro Critic

Two-step TD target 使用 rollout 冻结的旧 $V(k+2)$，duration 次方与实际 holding 相同；Macro Actor Advantage 则由 duration-aware GAE 生成。两者不是同一个 target，但均对应实际 SMDP 时间尺度。Macro Critic 配置 50 次更新，学习率 1e-3，value loss 系数 0.01、梯度范数上限 0.5；trainer.py 强制 50 次，并在 Critic 更新前后记录 N2 target EV、correlation、RMSE、prediction/target 标准差比和拟合快照。

同一 update 也计算 realized SMDP Return-to-go，记录 N2 target、旧 Value 与 RTG 的均值/标准差，以及 N2-oldValue TD error。当前日志缺失，无法判断：

- 50 次 Critic 更新是否降低同批 N2 target 误差；
- N2 target 与 realized RTG 的相关性或差异是否可接受；
- 价值预测是否能跨 Episode 泛化；
- 不同 holding duration 下的 TD error 是否系统性变差。

summary-level 均值和标准差不足以恢复逐帧 N2 与 RTG 的配对误差；逐 transition 分析还需要保存对应 trace。代码配置 long_value_enabled=true，另有 50 次拟合 realized return-to-go 的 Long-Value Critic，但这是独立 Critic，不能以其训练 loss 代替 Macro Actor 使用的 Macro Critic 质量证据。

### Micro Critic

Micro Critic 以每 Slot GAE return 为训练 target，使用逐 Slot系统奖励和 terminal mask；这与 Micro Actor 的奖励目标一致。更新后会计算 target prediction correlation、RMSE、Explained Variance 及 prediction/target 的均值和标准差。任务到达、车辆任务负载及 Macro Cache 状态都可改变 G；Micro Critic 需要从 observation 与有限训练样本解释这些跨策略的奖励变化。没有当前 Schema 诊断日志，不能判定 Critic 已拟合不足、target 噪声过大或 Critic 良好但 Actor 信用分配困难。

结论：Macro 和 Micro Critic 的 target 语义静态上与各自 PPO 目标相符；价值估计质量为 UNRESOLVED。不能以“训练了 50 步”或“training loss 下降”替代 EV、相关性、RMSE、RTG 对比和 held-out 预测证据。

## 5. Cache 信用分配

当前九节点 Cache 是因子化 joint action：每个节点基于相同动作前 observation 各采样 KEEP 或一个合法单节点 UPDATE，多个节点可在同一 frame 同时 UPDATE。Macro policy-gradient 对每个被选动作项使用同一个全局 frame Advantage；joint ratio 对所有 Cache 项和 termination 项累积 log-ratio。

这不是 PPO 目标或概率计算错误：对已定义的因子化 joint policy，共享整帧回报是合法联合策略梯度形式。风险在于一帧中多个 UPDATE 的正负边际贡献无法由同一个全局 Advantage 区分；候选是在动作前逐节点构造，采样时看不到本 frame 其它节点的合成 Cache 选择。这样会增大梯度方差和动作间干扰，尤其当收益稀疏、Cache 成本一次性支付或多个节点服务替代/互补时。Macro joint clip 还可能同时缩小整组节点更新的有效步长。

第四阶段静态检查确认多节点 UPDATE 在概率结构中被允许，但该报告也明确当前旧 Schema 不能提供本版本频率；本阶段没有 exact-schema rollout，故不对“一帧同时更新几个节点”或 KEEP 概率作数值推断。

理论上值得优先验证节点级反事实 Advantage：对同一状态和其它动作固定，估算某节点 UPDATE 相对 KEEP 对 G/后续 return 的边际贡献，作为该节点梯度的 baseline。它不改变目标函数，但估计成本可能增加；若实现为近似 baseline，还须验证其不依赖该节点当前采样动作，避免引入偏差。Sparse coordination（每 frame 最多一个 UPDATE）可降低组合数和信用干扰，但会限制多节点互补动作，应作为单独版本与因子化基线比较，不能直接假定更好。

## 6. Macro/Micro 交替更新

Macro Actor/Critic、Micro Actor/Critic 和 Long-Value Critic 使用分离参数模块；Macro Actor 与 Macro Critic 参数有显式 disjoint 检查，policy.py 提供独立 Macro/Micro 参数组。四条 Episode rollout 期间策略摘要哈希保持不变且 optimizer step 计数不变；收齐后 trainer.py 先更新 Micro，再更新 Macro。

这是一种合法的交替策略优化顺序。每个 Actor 的 own-action Old log-prob 在其对应 rollout 策略下计算，Macro/Micro 使用独立参数，不存在把 Micro 概率错误混入 Macro ratio 的迹象。与此同时，Macro 更新仍使用更新前 Micro 策略产生的同批任务、资源竞争和 return target；Micro Actor 改变后，未来状态与奖励分布可能变化。这是目标分布的潜在非平稳性，不是已证实的 PPO ratio 错位，也不单凭先 Micro 后 Macro 证明算法错误。无 exact-schema 跨更新日志证明两个策略相互干扰或互相抵消。

若后续要检验该风险，最小对照是在同一固定训练 Episode 组和初始化下，对比当前 Micro→Macro 更新与 Macro→Micro 次序，记录逐更新 Macro/Micro KL、Critic EV 与配对 Validation return。只有成效差异持续超过 Evaluation Episode 的配对波动，才支持改变更新日程。

## 7. Reward 信号与局部动作收益

Q 是逐任务效用求和，所以任务数、完成标记、执行时延和上传能耗都会改变原始 Slot Q。第四阶段使用相同真实任务和车辆轨迹比较了 Episode 20150 Slot 45–49，任务到达数依次为 9、11、15、14、10；这说明单槽任务量有变化，但不是训练奖励方差估计。

该报告中的单个反事实为：

| 固定配置动作 | 5 Slot 的 ΔQ | 首槽一次性成本 | 5 Slot 的 ΔG |
|---|---:|---:|---:|
| SV41 Cache UPDATE | +0.081775 | 0.099496 | -0.017721 |
| UAV0 MOVE 15 m | +0.001172 | 0.002860 | -0.001688 |

KEEP+STAY 的同窗口累计 G 为 54.839964。该局部动作效应相对任务净收益总量较小，且首槽成本可能压过有限窗口内的命中收益；这支持“动作信号可能被任务随机波动淹没”的风险。它不是行动收益与随机噪声标准差的估计，不能外推到其它 Cache、位置或更长时间跨度。Macro gamma=1、GAE 跨 frame 传递 return，Cache 可跨 frame 保持，因此 5 Slot 未回本并不说明长期复用一定无价值。

高原始 Return 不等于策略改善；策略可控增益应比较相同任务轨迹下的 ΔG 或相同 Evaluation Episode 的配对 Return。反过来，低 UPDATE/MOVE 率也不单独证明策略失败：若可达收益经成本后为负，KEEP/STAY 本身可能是合理最优动作。优势归一化可以统一批次尺度，不能消除任务到达与执行噪声，也不能替代配对效应量。

## 8. 最终根因判断与最小后续步骤

### 按潜在影响排序的前三项

1. **UNRESOLVED：当前完整 Schema 的训练/Validation 证据缺失。** 这不代表底层根因，而是无法区分“Actor 少更新”“KL 保护过严”“Actor 更新但动作不变”“Critic 不准”或“确实未改善”的首要诊断障碍。最小下一步是从正式初始 Checkpoint 按当前配置完成一个 4-Episode 更新组，保存所有正式日志和更新前后 Checkpoint；同批 paired Validation 只使用预先指定 Validation Episode，不用 Test。
2. **SUPPORTED_RISK：九个 Cache 节点共用全局 Macro Advantage。** 同时多节点动作的边际贡献被混合，容易产生高方差或抵消；当前数据只确认动作结构，未量化实际发生频率。若 exact-schema 诊断显示多节点 UPDATE 常见且回报贡献混杂，最小算法消融先选“节点级反事实 baseline”或“sparse 单节点 coordination”其中一个，单独版本化，再用同轨迹 paired update 和同一 Validation Episode 配对比较。
3. **SUPPORTED_RISK：可控动作净增益相对原始任务效用可能很小。** Stage 4 一个局部 Cache/UAV 组合的净收益均为负，任务量在五槽中变化；这不能给出噪声方差，但说明仅看 Return 曲线可能难看出小动作效果。后续最小辨别方式是在预先选定的少量 Training Slot 上使用共同任务轨迹比较 KEEP/UPDATE 与 STAY/MOVE 的配对 ΔG，并和同槽任务数、时延/能耗变化一起报告，再决定是否需要低方差 baseline。

Macro/Micro 非平稳性列为额外 SUPPORTED_RISK，但当前证据不支持把它排在已观测的结构性 Cache 信用问题之前；不应仅因更新顺序修改实现。

### 结论边界

- 已确认：当前实现具备 Actor optimizer 更新、KL 保护和日志字段；4 Episode 收齐前冻结策略；Macro/Micro ratio 分离；SMDP GAE/N2 target 使用实际 holding；Horizon Mask 修复仍在。
- 未确认：当前 Actor 实际步数、joint KL/clip 与回退频率、KEEP/UPDATE 频率、Critic 准确性、收益提升、Checkpoint 排名或 PPO 不收敛。
- 本轮没有发现足以确认 PPO 数学目标或动作概率合同错误的实现缺陷。共享 Advantage、任务噪声和交替更新是风险，不能据此宣布算法不收敛。

    EXACT_SCHEMA_TRAINING_LOGS_AVAILABLE = NO

    MACRO_ACTOR_EFFECTIVELY_UPDATING = UNRESOLVED
    MICRO_ACTOR_EFFECTIVELY_UPDATING = UNRESOLVED

    MACRO_JOINT_KL_LIMITING_UPDATES = UNRESOLVED
    MACRO_CRITIC_VALUE_QUALITY = UNRESOLVED
    MICRO_CRITIC_VALUE_QUALITY = UNRESOLVED

    CACHE_CREDIT_ASSIGNMENT_PROBLEM = SUPPORTED_RISK
    MACRO_MICRO_NONSTATIONARITY_PROBLEM = SUPPORTED_RISK
    REWARD_SIGNAL_TO_NOISE_PROBLEM = SUPPORTED_RISK

    POLICY_IMPROVEMENT_CONFIRMED = UNRESOLVED
    PPO_NONCONVERGENCE_CONFIRMED = UNRESOLVED

    ROOT_CAUSE_1 = UNRESOLVED: 当前完整 Schema 的训练与配对 Validation 证据缺失
    ROOT_CAUSE_2 = SUPPORTED_RISK: 九节点 Cache 共用全局 Macro Advantage，节点边际贡献难区分
    ROOT_CAUSE_3 = SUPPORTED_RISK: 局部动作净增益小且任务量变化可能淹没策略信号

    ALGORITHM_REPAIR_REQUIRED = UNRESOLVED
    RECOMMENDED_MINIMAL_REPAIR = 先取得当前完整 Schema 的一组诊断更新与 paired Validation；若信用问题由日志证实，再单独测试版本化节点级反事实 Advantage

    SOURCE_CHANGED = NO
    CONFIG_CHANGED = NO
    FORMAL_DATA_CHANGED = NO
    PPO_TRAINING_EXECUTED = NO
    VALIDATION_EXECUTED = NO
    TEST_USED = NO
