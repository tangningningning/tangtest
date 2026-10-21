# MAPPO 正式实验评价协议

日期：2026-10-18  
范围：核对当前正式 Train/Validation/Test manifest、旧 PROPOSED PPO evaluator/checkpoint 合同，以及 Stage1A MAPPO 实际训练/采集接口，并给出冻结策略对照方案。本轮只静态读取 Stage1A worktree；不运行训练或完整 Episode 评价，不转换旧 checkpoint。

## 1. 当前正式数据合同

`environment_data/data_split_manifest.json` 定义按完整 sequential realization 切分，所有正式 Episode 为 300 slots、slot 1 秒、origin 0–299。当前 task data contract 为 10 个服务、`lambda=0.7`、真实全局 Zipf `zeta=0.8`、输入 500–1000 KB。三个 formal pool manifest 的 seed 范围互不重叠：

| 用途 | 路径 | Seeds | Episode 数 | 唯一准入用途 |
|---|---|---:|---:|---|
| Train | `environment_data/proposed_joint_training/` | 20001–20300 | 300 | 策略更新、训练收敛记录 |
| Validation | `environment_data/proposed_joint_validation/` | 30001–30030 | 30 | 冻结 checkpoint 与超参数选择 |
| Test | `environment_data/proposed_joint_test/` | 40001–40100 | 100 | 全部选择锁定后的最终独立评价 |

每个 realization 完整保留实际 `tasks.csv` 和预测输入。当前任务只核对 split manifest/pool manifest；没有读取 Test episode 内容。

## 2. KEEP+STAY、旧 PPO、MAPPO 的共同执行

### 2.1 同一 Episode 和初始状态

对某个 seed `e`，所有比较方法必须加载同一个完整 Episode realization：同一条实际 `tasks.csv`、车辆轨迹/位置、预测 artifact、service catalog、系统配置和 slot 顺序。按当前正式配置产生的初始 Cache `C0` 与 UAV `U0` 对所有方法一致；每个方法从 episode reset 的同一 `C0/U0` 开始，执行完 300 slots。不能把一个算法的剩余 cache/UAV 状态带入下一个方法。

- **KEEP+STAY**：所有 slot 将当前 Cache 原样保留、UAV 位置原样保持，作为固定基线；实际任务仍由同一 Matching/KKT 环境处理。
- **旧 PPO**：加载已冻结、并由当前运行时确认兼容的 Checkpoint；不在评价中更新权重或 optimizer。
- **MAPPO**：加载冻结 Checkpoint，主分析使用 deterministic action。只有未来已确定其接口和合法动作语义后，才可加入共同 evaluator。

主比较使用 deterministic 策略动作，避免动作采样噪声；预测场景和实际任务均由固定 Episode 文件提供。Stage1A 的 `MAPPOBaselineTrainer.collect_episode(..., deterministic=True)` 可执行确定性采集；训练时使用 trainer 内部 Torch generator。补充 stochastic 对照须对每个 `(episode_seed, slot, decision_head, repeat)` 使用跨算法一致的 action RNG key/均匀随机数，并按相同 episode/repeat 对齐。当前 MAPPO collector 没有 keyed RNG 参数，保存/恢复其 trainer generator 状态；单纯设置相同初始 seed 仍不能保证不同动作维度消费相同随机数。不能做到 keyed RNG 时，stochastic 只作次要描述，不替代 deterministic primary。

### 2.2 完整 Episode 净效用

当前正式 slot 奖励实现强制检查：

```text
R[e,t] = actual_task_utility[e,t]
         - actual_cache_cost[e,t]
         - actual_uav_movement_cost[e,t]
U[e]   = sum(t=0..299, R[e,t])
```

`U[e]` 应与 300 个 slot `system_utility/reward` 的总和一致。主指标是完整 Episode 的累计净效用，不对 slots 或任务做分母归一后再替代主指标。为解释规模影响，另外报告总实际任务数、每任务净效用、每任务时延/车辆能耗。

### 2.3 Cache 与 UAV 成本单列

逐 Episode 分开记录：

- `task_utility_total = sum(actual_task_utility)`。
- `cache_cost_total = sum(actual_cache_cost)`，另列每节点成本、真实 Cache 内容变化节点数和变化率。只把实际发生的配置变化成本计入；策略选中 UPDATE 但配置未变不能冒充实际重配置。
- `uav_movement_cost_total = sum(actual_uav_movement_cost)`，另列每架 UAV 实际位移距离、MOVE 候选率和实际非零位移率。动作选择与物理位移要分开。
- `net_utility = task_utility_total - cache_cost_total - uav_movement_cost_total`，并与 `total_return` 核对。

当前 Cache 成本参数中包含 experiment-design 的等效成本项；报告时称为系统目标/配置成本，不把它说成已测量的物理能耗。UAV 的 objective movement cost 与实际飞行距离/移动能耗也分列。

## 3. Checkpoint 选择和训练收敛

### 3.1 只用 Validation 选 Checkpoint

每个独立训练 run 都应保存初始化 Checkpoint、完整 PPO/MAPPO update 边界上的周期 Checkpoint、最终完整 Checkpoint，并记清 episode/update 计数及中断状态。所有有效候选在完全相同的 Validation seeds 30001–30030 上 deterministic 评价。

对每个 training run，以 30 个完整 Episode 的 `mean(total_return)` 最高值选择一个 Checkpoint。只允许完整 update 且 schema/config/pool 合同通过的候选参与选择；不完整 update、NaN/Inf、不可行 slot 或资源检查失败的候选须显式标记为无效。并列或接近时同时列出逐 seed 配对差异、波动和成本分解；不设事后差异阈值，也不因均值很小就宣称显著胜出。

Stage1A worktree 中的 `run_mappo_baseline_stage1b.py` 只定义了单个 Validation seed、同一连续 5-Slot 窗口上的初始/最终策略 smoke；它不是完整 Episode checkpoint 选择器，不可用于正式 best-Validation 选择，也不构成性能结论。

现有 evaluator 会拒绝名为/标记为 `best_train_return_checkpoint.pt` 的训练回报诊断 checkpoint。训练回报最大者不能替代 Validation 选择。

### 3.2 训练收敛与冻结策略改善分开

训练阶段记录每个 Episode 的训练 return、task utility、Cache/UAV 成本、task count、update index、优化器/训练计数器及 action 统计；按完整策略更新组绘制均值和离散度。它表示当前训练策略在其训练 Episode 上的表现，不是独立泛化指标。

冻结策略改善用同一 run 的 `initial_checkpoint` 与 Validation 选出的 Checkpoint，在同一 30 个 Validation Episode 上配对计算：

```text
frozen_delta[e] = U_selected[e] - U_initial[e]
```

同时报告 30 个配对差值的均值、分布及分解。若需要比较不同训练随机种子，使用相同预先声明的 seed 列表；建议至少 5 个独立 policy initialization/training seeds。Episode seeds 是环境 realization，不是训练随机种子，不能混为一个样本量。

## 4. 多种子和配对统计

对算法 `a,b`，以同一个环境 Episode seed `e` 配对：`delta[e]=U[a,e]-U[b,e]`。报告每个策略的 mean/SD/median/范围、逐 seed delta、配对平均 delta、95% 配对置信区间、胜/平/负 Episode 数。预先固定 CI 算法（配对 t 区间或按 Episode seed bootstrap），同一份比较不在看到结果后切换方法。

有多个训练 run 时，先在每个 training seed 内对 Validation/Test Episode 做汇总，再汇总 run 间均值和离散度；同时保留每个 run 的结果。Stochastic repeats 先在同一 Episode 内平均，再跨 Episode seed 统计；不能把 repeats 当成额外独立 realization。确定性结果是主结果，stochastic 单独成表。

## 5. 单 Slot 完整执行时间

计时使用同一台机器、同一 `adaptive_cache_mec` Python 环境、相同 CPU/GPU 设置和 `time.perf_counter()`。单 Slot 计时边界覆盖：状态/特征组装、Cache/UAV 候选构造、策略前向/动作选择、真实配置应用、Matching/KKT、slot utility 与必要记录；不计 checkpoint 加载、文件读取和训练反向更新。每个 Episode 报告 300 个 slot 的 mean、median、p95、最大值，并附每 Episode wall time。使用相同 Episode seed 配对，以免任务规模差异混淆策略开销。

旧 PPO 的 `rollout.py` 在每 Slot 记录 `slot_runtime_seconds`，但 Stage1A `MAPPOBaselineTrainer.collect_episode()` 只在 `MAPPOMicroTransition`/summary 保留 `matching_kkt_seconds` 与 Macro/Micro actor inference 时间，没有完整 `slot_runtime_seconds`，也没有独立 frozen-evaluation 汇总器。因此当前 MAPPO timing 与旧 PPO evaluator 尚未对齐。正式比较需让三种策略在相同边界内计时，并将逐 Slot完整时间写入 MAPPO 冻结评价输出，才能计算 mean、median、p95、最大值；不能用 actor inference 或 Matching/KKT 子计时冒充完整 Slot 执行时间。

## 6. Test 最终留出

Validation checkpoint、超参数、方法选择、统计方案和候选策略名单锁定后，才对 100 个 Test realization（40001–40100）做一次预注册的 deterministic 冻结策略评价。Test 不得用于 checkpoint 选择、阈值/归一化调整、状态特征开发或重新挑选算法。Test 结果报告后若再据此改方案，该 Test 就不再是未触碰证据；后续确认需另建独立数据池。当前回合没有执行 Test episode。

## 7. 规模实验设计

以当前正式基准 `5 SV + 20 TV + 2 RSU + 2 UAV`、`lambda=0.7`、10 服务、300 slots 为基点。规模研究必须使用另立的 versioned config、data manifest 和输出目录；不覆盖当前正式 pool，不改变当前 Zipf/任务数据。

建议先单因素扩展，再按计算预算做组合压力点：

| 因素 | 基准 | 候选水平 | 需保持不变的比较条件 |
|---|---:|---|---|
| TV 数量 | 20 | 10、40、80 | 同服务目录、Zipf、slot 长度；按 scale 重新生成轨迹/任务 realization |
| SV 数量 | 5 | 1、10 | TV 固定 20；重新冻结角色/轨迹映射 |
| RSU 数量 | 2 | 1、4 | 站点按同一拓扑部署规则预先选定并冻结 |
| UAV 数量 | 2 | 0、1、4 | 明确零 UAV 时的动作/状态 mask 合同；不填充虚构 UAV |
| 请求量 | lambda 0.7 | 0.35、1.0 | 仅在独立规模数据中改变到达率，Zipf 仍保持原合同 |

每个 scale 单独建立 Train/Validation/Test 完整 Episode split，seed 不与当前池或彼此重叠；同一 scale 内所有方法共享同一组 Episode、实际任务、预测 artifact 和初始状态。分别回答两种问题：固定策略对规模变化的 zero-shot 稳健性，以及每个规模重新训练后的容量/性能。不要混合两种结果。比较净效用总量时同时报每任务净效用、Cache/UAV 成本、时延/能耗、任务数量和运行时间；大规模任务变多会机械性增加总效用，不能只看总和。当前回合只定义设计，没有生成规模数据或运行实验。

## 8. 旧 Checkpoint 与当前 Schema

正式旧 PPO `configs/proposed/system_config.toml` 使用 `MULTI_NODE_FACTORIZED`，其策略 Schema 是 v4 + `per_slot_analytic_task_model_v1_training_p95_reward_norm_v1`；旧 PPO evaluator 按 policy schema、checkpoint version、architecture、task/observation contract、Cache action contract、config hash 做严格校验。

本轮用指定 Conda 环境只加载 checkpoint metadata，并调用当前 evaluator 的 metadata 校验（没有构造策略、恢复训练或运行 episode）。抽查的旧 checkpoint 全部被当前 evaluator 拒绝：

| Checkpoint | 保存版本 | 当前检查结果 | 主要不兼容点 |
|---|---|---|---|
| `formal_cache_economic_v2_lr_1e_minus_4/latest_checkpoint.pt` | Architecture 31 / Checkpoint 58 / Schema v2 | REJECTED | 旧策略 Schema |
| `stageB3_14h_fresh_macro_actor_lr_5e_minus_4/latest_checkpoint.pt` | Architecture 33 / Checkpoint 60 / 旧 v4 Schema | REJECTED | 缺当前 task/reward contract 后缀，版本和配置哈希均旧 |
| `b3_15h_legacy_factorized_48ep/latest_checkpoint.pt` | Architecture 33 / Checkpoint 60 / 旧 v4 Schema | REJECTED | 同上 |
| `b3_15h_sparse_coordinated_48ep/latest_checkpoint.pt` | Architecture 34 / Checkpoint 61 / sparse v5 Schema | REJECTED | 稀疏 Cache action schema，不是当前正式 MultiNode v4 |

以上拒绝信息均为 `checkpoint policy schema is not supported by PROPOSED`。没有转换或重训这些权重。Stage1A MAPPO 使用独立 `MAPPO_BASELINE_SCHEMA="mappo_shared_actor_global_critic_fixed_macro_2_v1"` 和 `MAPPO_CHECKPOINT_VERSION=1`，由 `MAPPOBaselineTrainer` 独立构造网络并严格检查自身 `model_contract`。旧 PPO Checkpoint 不属于该 MAPPO Schema，不能作为 MAPPO resume/load；这不影响从随机初始化开始训练 MAPPO。若旧 PPO 仅作为行为基线，须由其原生且可加载的 evaluator 在同一当前正式环境 Schema、Episode realization、预测/任务文件和初始状态下执行。严格性能比较应使用同一正式环境 Schema 下可加载的冻结 PPO 基线；不得把不兼容 checkpoint 转换后冒充严格对照。

## 9. Stage1A MAPPO 接口、Stage1B Checkpoint 合同与评价缺口

### 9.1 已确认的 MAPPO Trainer/Collector 接口

`src/ppo_simple/mappo_baseline.py` 暴露 `MAPPOBaselineTrainer(context, device, seed)`、`collect_episode(aligned_steps, max_slots, terminated_at_end, deterministic)`、`update(collected, epochs)`、`save_checkpoint(path, metadata)` 和 `load_checkpoint(path)`。`collect_episode` 返回 `MAPPORollout(macro, micro, summary)`；summary 包含完整/截断状态、每 Slot `slot_rewards`、累计净效用、Cache 成本、UAV 移动成本、合法性和 Matching/KKT/Actor 子计时。真实任务效用可从每个 Micro transition 的 `actual_task_utility` 汇总。详情见 [双预测时空状态特征设计报告第 5 节](prediction_spatiotemporal_feature_design.md)。

但此 API 是训练采集/更新接口，不是完整冻结池 evaluator。当前 `src/ppo_simple/evaluator.py:evaluate_pool` 仍实例化旧 `PPOTrainer` 并加载 legacy PPO checkpoint；没有 MAPPO 分支。Stage1B runner 目前只用一个 Validation seed、5 Slot deterministic 窗口对比初始/最终策略，并回读检查 checkpoint；不覆盖 30×300 Validation Episode，也不作 best checkpoint 选择。

正式 MAPPO 冻结评价适配需：按 allowlist 显式载入指定 Train/Validation/Test realization；每个 Episode 重新建 context、对齐完整 300 Slot、使用相同初始 Cache/UAV 与固定预测/实际任务文件；严格加载 MAPPO Checkpoint 后不调用 `update`，主评价 deterministic；验证策略参数和 optimizer 状态未变；收集逐 Slot reward、`actual_task_utility`、Cache cost、UAV movement cost、动作/可行性以及完整 Slot runtime。累计净效用按当前实际 reward 总和计算，并以任务效用减成本分项复核。Validation evaluator 可参与 Checkpoint 选择；Test 只有名单、方案、Checkpoint 和统计方法锁定后才可进入 evaluator。

### 9.2 Stage1B 最小 Checkpoint 元数据合同

Stage1A Checkpoint 已保存：`checkpoint_version`、`baseline_schema`、`model_contract`、`progress.update_count/seed`、完整 `policy_state_dict`、四个命名 optimizer state、Python/NumPy/Torch CPU/CUDA 和 trainer generator 随机状态、Python/平台/NumPy/Torch 版本，以及可扩展 `metadata`。其 `model_contract` 当前含 observation 维度、节点数/角色、Macro/Micro 候选维度、Macro 间隔、PPO 核心参数及 raw config SHA。Stage1B runner metadata 另记录 run/git 信息、Train/Validation manifest SHA、训练窗口和 seed。

Stage1B 正式 Checkpoint 应将下列信息作为可校验合同（载荷可分区保存，但语义字段不可缺）：

| 字段组 | 最小内容 |
|---|---|
| 格式/算法 | `checkpoint_version`、`algorithm_name=MAPPO`、算法/策略 schema 版本、Checkpoint role（initial/resume/frozen candidate） |
| 模型参数 | architecture ID/version、构造参数（hidden sizes、激活、输入/候选维度）、node/service/vehicle 顺序和数量、head 数、候选特征 schema/version、Macro cadence、动作编码；完整 Actor 与 Critic `state_dict` |
| 优化器 | 每个 Actor/Critic optimizer 的类型、有效超参数、参数组和完整 optimizer state（含 step/moment）；各组累计 update/step 计数 |
| 随机状态 | Python、NumPy、Torch CPU、每个 CUDA device（如使用）、trainer/action generator，以及独立环境/采样 RNG 的状态；同时记录初始训练 seed |
| 训练进度 | run ID、training seed、完整 update index/count、已处理的训练 Episode/realization IDs、总 Slot/Macro/Micro transition 数、阶段/中断状态、最后完整 update 边界；不完整更新不得标为可恢复正式候选 |
| 配置与数据身份 | config ID/path、raw config SHA 与语义合同 hash、Train pool manifest SHA、预测/任务/系统 artifact manifest 身份、split 名称；Validation manifest 身份可记录作选择合同，Test 不参与训练身份 |
| 运行记录 | Git commit/worktree revision、Python/PyTorch/NumPy 版本、设备及保存时间；用于追溯，不替代模型/配置合同检查 |

加载必须校验 algorithm/schema、模型参数与输入动作合同、配置/训练池身份，再严格恢复模型、optimizer、随机状态和进度。Stage1A 现有版本尚未把 hidden sizes/完整模型构造参数及处理 Episode/Slot 进度放入 `model_contract/progress`；Stage1B 应补齐后再依此作为 resume 合同。Checkpoint metadata 可记录 Validation 结果与选择角色，但 Test 指标不得作为选模依据。

## 10. 本轮执行边界

本轮只读核对了上述报告以及 `adaptive_cache_mec-mappo-baseline-stage1` 的当前工作树代码（包括其未提交/未跟踪的 Stage1A/Stage1B 实现文件）；只更新本研究报告，不向该 worktree 写入。没有运行 Python、测试、训练、完整 Episode 策略评价或参数搜索，没有访问 Test realization 内容，也没有转换 Checkpoint。旧 PPO Checkpoint 的 Schema 不兼容仅限制旧 PPO 基线加载/严格对照，不限制新 MAPPO 随机初始化训练。
