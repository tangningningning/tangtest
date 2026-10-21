# Cache–UAV 冻结策略配对反事实评价

日期：2026-10-21（Asia/Shanghai）。项目：`C:\Users\tangning\Desktop\code\adaptive_cache_mec`。

## 结果

**在指定 Checkpoint、正式 deterministic 执行规则和 4 个共同 Validation 种子上，联合策略相对固定初始 Cache + UAV STAY 没有改善：每个种子的 A、B、C、D 回报完全相同，平均交互效应为 0。**

这是实际执行结果：联合策略 A 在四个完整 Episode、合计 1,200 Slot 中，实际 Cache 节点变更次数、UAV 移动次数和移动距离全部为 0。因此，四组在本次评价输入上实现了相同的物理配置轨迹。不存在可归因于 Cache 主效应、UAV 主效应或两者互补作用的新增收益。

该结论仅适用于本次冻结 deterministic 策略。它不证明随机训练策略没有潜在收益，不证明参数没有学到变化，也不证明最优策略应始终 KEEP/STAY。本次没有扩大种子数、增加随机策略重复或选择其他 Checkpoint 追求正结果。

## 1. Checkpoint、配置与数据池身份

### 1.1 预先固定的评价对象

| 项目 | 实际值 |
|---|---|
| Run | `runs/alternating_actor_updates_v1_macro_lr_5e-4/` |
| Checkpoint | `checkpoints/latest_checkpoint.pt`，在评价前固定；没有按评价结果重新选点 |
| Checkpoint SHA256 | `2832fbc8f065c587dbd7ff1387245877f61c74d406edf48a962e8d2ee39a1688` |
| 已完成训练 | 276 Episode / 69 个更新组；不是已完成 300 Episode 的最终模型 |
| Checkpoint / Architecture 版本 | 65 / 2 |
| 策略 Schema | `adaptive_joint_cache_uav_mappo_v1_factorized_keep_node_identity_v1` |
| 算法 / 方法 | `ADAPTIVE_CACHE_UAV_JOINT_MAPPO` / `ADAPTIVE_CACHE_UAV_MAPPO` |
| Cache / UAV 动作 | `MULTI_NODE_FACTORIZED`，9 个 Cache 节点；2 UAV 自回归 |
| 配置 | `configs/proposed/alternating_actor_updates_v1.toml` |
| 配置 raw SHA256 | `297822402a783b4bbbb28b1be2e61aefb1fa2cef5e110af279e814f58e4b670f` |
| 配置 semantic SHA256 | `2a4f74c8e55d6f1329c790d5265988c4e1bbf5600b1a069570b93cd3a63c01f5` |
| 有效 Macro Actor LR | `5e-4`，只核对训练身份，本次没有优化 |
| 评价池 | `environment_data/proposed_joint_validation/` |
| 四个共同环境种子 | **30001、30002、30003、30004**，选正式池的前 4 个种子，事前固定 |
| 每组长度 / 重复 | 每种子完整 300 Slot / 一次 deterministic 执行 |

同时核对 Run manifest、run_config 文件、Checkpoint 的配置哈希。调用现有 `evaluator._load_checkpoint_metadata`（`src/ppo_simple/evaluator.py:243`）和 `PPOTrainer.load_checkpoint`（`src/ppo_simple/trainer.py:3624`），全部通过，没有 `strict=False`、改 Schema、改配置或忽略哈希的做法。

Checkpoint 加载参数与正式 `evaluate_pool` 一致：传入当前 raw/semantic 哈希，`episode_pool=None`。这是正式评价器针对不同 Validation Pool 的标准用法，不是忽略某个已发现的冲突；Training/Validation 的分离由独立的正式评价池门禁校验。未使用被评价器禁止的 `best_train_return_checkpoint.pt`。

### 1.2 评价池门禁与文件完整性

调用原有 `_validate_formal_validation_pool`（`evaluator.py:143` 附近），验证正式路径、split、完整种子覆盖、Training/Validation 种子及 source Episode 来源不重叠、共同 frozen split manifest。随后验证 4 个 Episode manifest 的哈希、seed/split、服务目录，以及 manifest 中全部 **44 个数据文件**的哈希和文件集合。没有生成、改写或替换数据。

| 身份对象 | SHA256 |
|---|---|
| Validation Pool manifest | `a75f4935f598a2945f66547ab48ce5a24a86624bb142a9349787797c402da10a` |
| Training Pool manifest | `b6cae236b029f4d24c7488a91f0825247a20f475ec76b2d80737de44ae491bd3` |
| 共同 data_split manifest 标识 | `70e6aea73efb23c929cecd2372d9c155b4aebea25c3bcfb59750c742f686f26d` |
| Episode 30001 manifest | `c41529ed61face7249694894195f5ed5b1c85d49a099e2536b3f78f45002468b` |
| Episode 30002 manifest | `283ce0a9d91dfda4e086427e8df3cabbf92122853030354f3924c8c6708cd2e1` |
| Episode 30003 manifest | `2b355260804eddeeb7bb5b9780b7ec669062eb2ddc6fed164706ca6b84834894` |
| Episode 30004 manifest | `27a72d03788d8fdd578053c50e7f198bf9df7b3c5bae547a45960fc2178f199c` |

没有访问 Test Pool，也没有用 Training Episode 做本次性能评价。Loader 为加载合同建立了正式 reference context；真正执行的 16 个 Episode 全部来自上述 Validation 输入。

### 1.3 当前代码身份

Git 分支 `main`，HEAD `aa8981d3ae5aa9ba2d8ce130819c38afed5b480b`。工作区原有 `trainer.py` 的 4 行 LR override 接线改动保留，本次没有修改。

| 实际执行源码 | SHA256 |
|---|---|
| `src/ppo_simple/evaluator.py` | `811307ee4e3ea2239ba342887a7a8247708c2500bbdb95dff7c9fb1bba70df09` |
| `src/ppo_simple/trainer.py` | `923635d3282424536580f18c7dca4a448cb250f4713f7062054aec6cb30b8ee5` |
| `src/ppo_simple/rollout.py` | `cffd761bee68e26b222ad8849b6d3b13d022fc34bb3b23a5fd7435a8859c6243` |
| `src/ppo_simple/policy.py` | `8c190ab81a34748145edac4138395e1e67709b441ebbc13bbd62d56027a32b86` |

原 Run 没有记录训练时的源码文件哈希，本次不能补证当初训练代码与当前源码逐字相同。本次评价所用当前实现及 Checkpoint 的身份明确，现有加载门禁未发现不兼容。

## 2. 四组干预如何执行

### 2.1 现有接口能力

`evaluator.evaluate_pool`（`src/ppo_simple/evaluator.py:865` 附近）支持联合策略完整 Episode，以及单种子的短窗口 KEEP+STAY 对照；没有分别控制 Cache/UAV 的完整 Episode 开关。因此创建了一个独立临时评价脚本，复用原有门禁、Context、`aligned_train_steps`、`collect_dual_timescale_episode` 和 `_episode_summary`，没有复制环境或 Matching/KKT 求解器，没有修改正式训练路径。

| 组 | Cache | UAV | 实现位置 |
|---|---|---|---|
| A | Checkpoint 的 Cache candidate 与 Holding termination 策略 | Checkpoint 的 UAV 策略 | 原有采样方法不干预，`deterministic=True` |
| B | 同 A | 每架 UAV 始终 STAY | 加载通过后，仅在临时进程中将 `sample_micro_head` 输出固定到已有合法 candidate 0 |
| C | 每个 Episode 自己的初始 Cache 永久保持 | 同 A | 加载通过后，仅将 `sample_macro_nodes` 输出固定为各节点已有合法 KEEP candidate 0 |
| D | 同 C | 同 B | 同时应用两个动作输出干预 |

干预输出使用对应合法动作的点质量概率，log-prob/entropy 为 0，仅供现有 collector 的一致性记账；这些概率没有被当作原训练策略概率，没有计算 PPO loss 或 importance ratio。没有改动任何模型 tensor、优化器或配置 mask。Cache 的 Holding head 仍按同一冻结规则运行；在 C/D 中，无论何时进入 Macro 决策，物理 Cache 都保持各自初始值，Holding 不改变固定基准的执行状态。

每个干预在本组结束时恢复原方法。A 使用的确为原始 Checkpoint 方法，并非沿用上一组的 KEEP/STAY 覆盖。脚本 SHA256 为 `426d1244959f4699b563de321a64d8548342e0733ab1bc726b05cf167661cc5e`；脚本和中间数据已按约束清理，不作为持久实现交付。

### 2.2 配对与真实闭环

每种子使用同一个已对齐外生输入列表，四组各自从 collector 复制的同一初始 Cache、UAV 位置开始，reset 相同策略随机种子。四组各执行完整 300 Slot，无截断；顺序为 A、B、C、D。没有把 A 的 Cache/UAV 轨迹复制给其他组，没有冻结训练模块本应看到的后继状态，也没有强制各组得到相同的候选、卸载或收益。

初始 UAV 坐标（m）：UAV_0 `[3364.03, 1119.88]`，UAV_1 `[5179.65, 1710.41]`。四个环境的初始 Cache 矩阵摘要均为 `ac3a55a012028fbfeb269d283d6694072159a38214f175121566c4857e7c1a70`；不同 seed 的 SV 实体及外生请求/轨迹仍按各自 Context 映射，不把角色相同当成同一车辆。

执行链仍为 `rollout → execute_configured_slot → execute_slot → Matching/KKT → G`。计量 wrapper 只在实际 Slot 结果返回后读取前后状态，并调用原有 `build_feasible_targets` 统计缓存命中；没有额外执行一次 Matching/KKT，也不修改返回结果。

本次四组轨迹相同是**观察结果**：训练得到的 deterministic 动作本身全部为 KEEP/STAY，而不是评价协议要求轨迹相同。

### 2.3 模式与冻结检查

- 全部组使用 `policy.eval()`、`torch.inference_mode()`、CPU 单线程和正式 deterministic 规则；没有参数梯度、`backward()` 或 optimizer step。
- 正式 Cache deterministic 规则是先比较 KEEP 与 UPDATE 的概率质量，再在 UPDATE 组内按 raw score 选候选，平局偏 KEEP；UAV 也按 STAY/MOVE 分组决策。它不是训练时的随机采样分布，也不等于直接在所有候选中取概率最大项。
- 每个 Episode 前后比较所有模型 tensor、四个优化器完整 state、训练 counters 的内容哈希，均相同；还检查全部参数 `grad is None`。
- 同种子的输入在每组前后做完整 pickle 内容哈希，均不变；初始 Cache/UAV 不被修改。
- 每 Slot 检查 C/D 的 Cache 与初始矩阵一致、B/D 的 UAV 与初始位置一致；动作统计与实际 Cache row 差异、实际位移次数一致；移动距离与移动成本公式一致。
- 评价前后对 **82 个受保护文件**做 SHA256 核对，包括正式 src、TOML、选定数据、manifest、服务目录和 Checkpoint，全部不变。

## 3. 指标口径

系统 Episode 回报为 `J = sum_t G_t = sum_t Q_t - sum_t C_cache,t - sum_t C_move,t`，与正式训练和评价目标相同。

- 时延、能耗分别为所有已到达任务的完成时延总和（s）与车辆能耗总和（J）；同时给出任务加权均值。不同 Slot 的任务完成时延相加，不应误读为物理 Episode 运行时长。
- **缓存命中率**：实际执行配置下，一个任务请求在至少一个物理可行 Edge 节点缓存了对应服务的比例；使用正式 `build_feasible_targets` 判断。分母为本 Episode 的全部实际任务，不是 Slot 数，不是 CPU/JIT 的 cache hit。
- **Cloud 卸载比例**：实际 Matching 选择 Cloud 的任务数 / 实际任务总数。缓存可用时仍可能因资源竞争而选择 Cloud，所以缓存可用命中率不等于 `1-Cloud比例`。若把“实际在 Edge 执行”定义为执行命中率，它等于后者，可从本报告直接换算。
- 每组都使用正式初始 Cache 初始化合同；初始装载不额外计成本，只对后续真实新增缓存服务计下载成本，与正式系统 reward 保持一致。

## 4. 四组原始结果

以下 16 行均来自独立执行的完整 Episode。为便于阅读展示到 6 位小数；配对差值用未舍入值计算，零差值不是表格取整造成。

| 种子 | 组 | 系统回报 J | 任务效用 Q | Cache 成本 | 移动成本 | 总时延 s | 总车辆能耗 J | 缓存命中 % | Cloud % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 30001 | A | 3185.908523 | 3185.908523 | 0 | 0 | 57167.744088 | 1837.165068 | 61.599256 | 43.165969 |
| 30001 | B | 3185.908523 | 3185.908523 | 0 | 0 | 57167.744088 | 1837.165068 | 61.599256 | 43.165969 |
| 30001 | C | 3185.908523 | 3185.908523 | 0 | 0 | 57167.744088 | 1837.165068 | 61.599256 | 43.165969 |
| 30001 | D | 3185.908523 | 3185.908523 | 0 | 0 | 57167.744088 | 1837.165068 | 61.599256 | 43.165969 |
| 30002 | A | 3321.079509 | 3321.079509 | 0 | 0 | 46233.332479 | 1200.075876 | 61.415634 | 39.585319 |
| 30002 | B | 3321.079509 | 3321.079509 | 0 | 0 | 46233.332479 | 1200.075876 | 61.415634 | 39.585319 |
| 30002 | C | 3321.079509 | 3321.079509 | 0 | 0 | 46233.332479 | 1200.075876 | 61.415634 | 39.585319 |
| 30002 | D | 3321.079509 | 3321.079509 | 0 | 0 | 46233.332479 | 1200.075876 | 61.415634 | 39.585319 |
| 30003 | A | 3362.521955 | 3362.521955 | 0 | 0 | 47391.143629 | 1185.229386 | 60.000000 | 41.715629 |
| 30003 | B | 3362.521955 | 3362.521955 | 0 | 0 | 47391.143629 | 1185.229386 | 60.000000 | 41.715629 |
| 30003 | C | 3362.521955 | 3362.521955 | 0 | 0 | 47391.143629 | 1185.229386 | 60.000000 | 41.715629 |
| 30003 | D | 3362.521955 | 3362.521955 | 0 | 0 | 47391.143629 | 1185.229386 | 60.000000 | 41.715629 |
| 30004 | A | 3634.770406 | 3634.770406 | 0 | 0 | 28919.105821 | 1193.820744 | 80.355030 | 25.443787 |
| 30004 | B | 3634.770406 | 3634.770406 | 0 | 0 | 28919.105821 | 1193.820744 | 80.355030 | 25.443787 |
| 30004 | C | 3634.770406 | 3634.770406 | 0 | 0 | 28919.105821 | 1193.820744 | 80.355030 | 25.443787 |
| 30004 | D | 3634.770406 | 3634.770406 | 0 | 0 | 28919.105821 | 1193.820744 | 80.355030 | 25.443787 |

完整精度的每种子回报（四组各自得到同一值）：30001=`3185.908523131235`，30002=`3321.0795086790818`，30003=`3362.5219554211076`，30004=`3634.7704059520956`。

各组相同的任务数量、均值和实际动作检查如下：

| 种子 | 任务数 | 缓存命中任务数 | Cloud 任务数 | 平均任务时延 s | 平均车辆能耗 J | 各组真实 Cache 变更节点数 | 各组 UAV 移动次数 / 距离 m |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30001 | 4302 | 2650 | 1857 | 13.288643442 | 0.427049063 | 0 | 0 / 0 |
| 30002 | 4196 | 2577 | 1661 | 11.018430048 | 0.286004737 | 0 | 0 / 0 |
| 30003 | 4255 | 2553 | 1775 | 11.137754084 | 0.278549797 | 0 | 0 / 0 |
| 30004 | 4225 | 3395 | 1075 | 6.844758774 | 0.282561123 | 0 | 0 / 0 |

每组共 16,978 个实际任务。四种干预共享相同任务，不能把 4 组加起来当成 67,912 个独立任务样本，也不能把 16 个执行 Episode 当作 16 个独立环境种子。

各组的 **Episode 等权均值**相同：

| 指标 | A | B | C | D |
|---|---:|---:|---:|---:|
| 系统回报 | 3376.070098 | 3376.070098 | 3376.070098 | 3376.070098 |
| 任务效用 | 3376.070098 | 3376.070098 | 3376.070098 | 3376.070098 |
| Cache / 移动成本 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 总时延 s | 44927.831505 | 44927.831505 | 44927.831505 | 44927.831505 |
| 总车辆能耗 J | 1354.072769 | 1354.072769 | 1354.072769 | 1354.072769 |
| 各 Episode 平均任务时延的均值 s | 10.572397 | 10.572397 | 10.572397 | 10.572397 |
| 各 Episode 平均车辆能耗的均值 J | 0.318541 | 0.318541 | 0.318541 | 0.318541 |
| 缓存命中率 % | 65.842480 | 65.842480 | 65.842480 | 65.842480 |
| Cloud 比例 % | 37.477676 | 37.477676 | 37.477676 | 37.477676 |

四个环境的 J 样本标准差为 `188.232827`，反映环境差异；它不是配对增益的标准差。

## 5. 配对差值、交互与不确定性

定义以固定基准 D 为参照的 Cache 效应 `E_C=B-D`，UAV 效应 `E_U=C-D`，交互 `I=A-B-C+D`。于是 `A-D=E_C+E_U+I`。另列 `A-B`（训练 Cache 下 UAV 的条件效应）和 `A-C`（训练 UAV 下 Cache 的条件效应）。这些是冻结策略的行为替换效应，不是训练梯度归因。

| 种子 | A−D | B−D：Cache | C−D：UAV | A−B | A−C | I=A−B−C+D |
|---|---:|---:|---:|---:|---:|---:|
| 30001 | 0 | 0 | 0 | 0 | 0 | 0 |
| 30002 | 0 | 0 | 0 | 0 | 0 | 0 |
| 30003 | 0 | 0 | 0 | 0 | 0 | 0 |
| 30004 | 0 | 0 | 0 | 0 | 0 | 0 |
| 均值 | **0** | **0** | **0** | **0** | **0** | **0** |
| 配对样本 SD | 0 | 0 | 0 | 0 | 0 | 0 |
| SE=SD/√4 | 0 | 0 | 0 | 0 | 0 | 0 |
| 观察范围 | [0,0] | [0,0] | [0,0] | [0,0] | [0,0] | [0,0] |

交互符号为 **4/4 个零、0 个正、0 个负**，是“一致为零”，不是“一致正互补”。按传统 2×2 因子设计计算的平均主效应 `[(A-C)+(B-D)]/2`、`[(A-B)+(C-D)]/2` 也都为 0。

本次配对经验方差确实为 0，因为四种策略在这些输入上采取相同物理动作。将 SD 机械代入 t 区间会得到退化的 `[0,0]`；**不把它解释为总体交互效应的有效零宽 95% 置信区间**。这里只有 4 个事前固定的连续 Validation 种子，没有随机策略重复，无法由此估计未覆盖环境或随机执行分布中的尾部互补效应。不报告显著性，不宣称已证明总体无协同。

其他指标的逐种子配对差同样全部为 0：时延、能耗、缓存命中率、Cloud 比例、Cache 成本和移动成本均无改善或恶化。

## 6. 一致性证据

16 个 Episode 全部通过终止检查，均有 300 Slot，最后一步 `done=True`，无 truncation。共执行 4,800 个实际 Slot；没有额外 warm-up 评价 Episode、初始模型对照或超出 4 种子的自动扩样。

每种子的四组外生输入哈希相同且执行前后不变；不同种子输入哈希不同：

| 种子 | 对齐输入内容 SHA256 |
|---|---|
| 30001 | `164502a7dae0edc7fefe9de5d199e41868c2fb4f4b724768c7d092e4676f1a53` |
| 30002 | `75814eebfee4b4847122f964cadfd667ea881a52efe733ae072b850c90ed2cf6` |
| 30003 | `34387ca247463024fc6405b1e4f2ef5a1d3d915b560fb50ab2d341b92c197f7e` |
| 30004 | `98a5a04d63dd53b5e9d16b2607ba86d19cd06e295e4ec428edb9a634eafdc96b` |

每 Slot 实际 Cache/UAV 状态序列的 SHA256 均为 `c9c8c9cb20d4fb0d614593b2a5c4d36cc12f95c25076cba12d2d2c2b14e8baf5`。该摘要只包含 Cache 矩阵和按正式 UAV 顺序排列的位置，不包含车辆任务输入；跨种子相同不表示使用了相同外生 Episode。

模型、优化器和训练 counters 内容摘要在所有组前后均为 `c7da3cfda443600efd9ae33ba34f9348a2a6b14a3aa2b6308869685ced37c0a1`。仅凭摘要不能推断动作策略正确；本次同时核对实际 changed/moved 状态和上表的任务/收益分项。

Episode `J=Q-Ccache-Cmove` 检查全部通过。额外从总任务数、总 T/E 和正式 `T_ref/E_ref` 复算 Q，最大绝对误差约 `3.18e-12`，符合浮点累积误差。

## 7. 对四个交付问题的回答

1. **联合策略是否改善系统回报？** 在本 Checkpoint 的 deterministic 模式与四种子范围内，没有。每种子 A−D 均为 0，均值改善率 0%；这是同一外生输入的配对结果。
2. **交互符号是否一致？** 一致为零，四个种子都没有正或负交互。不能将零值说成正互补，也不能外推随机策略交互的符号。
3. **收益来自哪里？** 本次没有相对固定基准的新增收益可分解；Cache、UAV、交互三项均为零。约 3376 的平均绝对回报来自共同初始配置下的任务执行、Matching/KKT 和外生环境，不是本次观测到的 Actor 干预收益。
4. **是否足够进入 N2 校准或 Last1 消融研究？** 本次本身不足以把“没有 deterministic 干预收益”归因到 N2 或 Last1。结合前次静态审计已有的 N2 与实现 RTG 差距，可以继续开展**不训练的 N2 校准诊断**，但本次结果没有新增“改 N2 target 就会改善协同”的因果证据。对 Last1，尚不足以优先启动训练消融或声称应关闭。下一步应先验证同一 Checkpoint 的随机动作激活/收益是否与 deterministic 规则分离，以及 KEEP/STAY 的组概率和价值偏差如何关联；本次没有自行执行这一步。

## 8. 证据限制与建议顺序

- 只评价事前选定的一个 latest Checkpoint，不代表全部中间 Checkpoint 或完整 300 Episode 最终模型；未用结果挑最佳点。
- 四个种子是正式 Validation 池的固定前缀，样本极小，不代表随机抽样的大规模泛化评估；没有 Test 结论。
- deterministic 分组阈值可产生全 KEEP/STAY，即使随机策略仍给 UPDATE/MOVE 非零概率。实际动作全零不能称为概率分布塌缩或参数没有更新。
- 因本次干预未改变物理动作，2×2 对照在这些输入上缺少有效的行为差异，不能用于识别“动作一旦激活时”的 Cache/UAV 互补强度。这个退化现象本身就是本次方向性筛查的关键发现。
- 可优先做已有产物的 N2/RTG 校准和分组决策概率核对；如需随机配对评价或 Last1 训练消融，应另行明确范围。没有理由仅据本次零交互就改奖励、惩罚权重或重跑 300 Episode。

## 9. 耗时、环境与清理

- 主评价脚本总耗时 **720.568 s，约 12 分钟**，包含身份检查、文件哈希、数据加载、16 个 Episode 和最终一致性检查。
- 16 个 Episode rollout/逐组检查耗时合计 **590.030 s，约 9 分 50 秒**；其余约 130.538 s 为门禁、加载、输入/文件核对等。本次计时不将前置代码阅读、Conda 排障和报告撰写算成仿真时间。
- 执行环境 `sys.prefix=C:\software\Anaconda_envs\envs\adaptive_cache_mec`。系统 `conda.exe` 启动器先报 DLL 错误，随后用 `C:\software\anaconda3\python.exe -m conda run --no-capture-output -n adaptive_cache_mec` 启动指定环境解释器完成评价。仅在当前进程设置 `CONDA_EXE` 为 Conda 的 Python 模块入口及 `_CE_M=-m`、`_CE_CONDA=conda`，没有改环境安装或系统配置，没有改用 base 环境执行评价。
- Python 使用 `-B`；Numba 缓存路径限定在本次临时目录。没有安装依赖、启动训练、更新任何参数、做超参数搜索或修改正式算法/配置/Checkpoint/旧结果。
- 本次临时目录 `.tmp_cache_uav_frozen_eval_20261021_01a15c67/` 中的评价脚本、preflight 与 results 中间文件已在结果写入本报告并复核后删除。唯一持久新增交付为本报告；目标文件原先不存在，没有覆盖旧报告。
- 未 commit、push 或改 Git 历史。原有工作区改动与其他人的文件保留。
