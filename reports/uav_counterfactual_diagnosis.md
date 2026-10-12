# UAV 动作价值反事实诊断

日期：2026-10-09  
范围：代码核对与冻结策略独立诊断；未启动 PPO 训练，未使用 Test split。

## 结论摘要

- 两个最新 checkpoint 可由现有 evaluator/trainer 加载，但训练进度分别为 224 与 300 episodes；没有同进度且可用于正式冻结评价的 checkpoint。因此两组只做各自内部的 STAY/MOVE 反事实分析，不把差异解释为纯学习率因果效应。
- 在两个 Validation realization 的前 10 个 slot 中，5e-4 冻结策略 20/20 次 UAV 决策都选 STAY；1e-3 冻结策略 20/20 个 slot 都让 UAV0 MOVE、UAV1 STAY。这是 deterministic 验证前缀行为，不等同于训练时的 stochastic 行为。
- 5e-4 的 6 个抽样状态均为 PPO 双 UAV STAY；3 类已检查合法 MOVE 中，按本 slot 实际系统效用选出的样本内最佳候选有 5/6 高于 STAY，平均差 `+0.00530`。
- 1e-3 的 6 个抽样状态均由 PPO 让 UAV0 MOVE。其 PPO 动作相对同状态 STAY 的即时系统效用平均差为 `-0.00149`（4 负、2 正）；但每个状态都有至少一个被检查 MOVE 候选高于 STAY，样本内最佳候选平均差为 `+0.00714`。
- 这支持“局部状态中存在 PPO 没选到的更好合法动作”，但只限于所选前缀、seed 和候选子集。测试的是单 slot 即时效用，没有传播 UAV 新位置对后续 slot 的影响；整体因果判断仍不足。

## Checkpoint 与训练进度

两份 `run_config.toml` 的原始 SHA-256 均为 `fe5d4f7bc57a424911166bcfe921c2ec5a7627b0a6aa759c4993f7f488a3775c`。latest checkpoint 均为 checkpoint version 58、architecture version 31，policy schema 相同；冻结 evaluator 元数据核验与 `PPOTrainer.load_checkpoint()` 均通过。

| Macro Actor LR | 冻结评估使用的 checkpoint | 已完成 episodes / policy updates | 可用 best checkpoint | latest SHA-256 |
|---|---|---:|---|---|
| `5e-4` | `latest_checkpoint.pt` | 224 / 56 | 208 episodes；`TRAINING_DIAGNOSTIC_ONLY_NOT_FINAL_SELECTION` | `5d5ee286c0a46a61448de9293388c2f5230804ab39e926a8c8693a06f5937c76` |
| `1e-3` | `latest_checkpoint.pt` | 300 / 75 | 292 episodes；`TRAINING_DIAGNOSTIC_ONLY_NOT_FINAL_SELECTION` | `e6ed46f593f724ff08c89106376ed33aba5f476e6d79e24e5a2613323d379b2e` |

没有同进度的正式 evaluator checkpoint：两个 latest 相差 76 episodes，两个 best 也不在同一进度；best 文件的 artifact role 还被现有 evaluator 明确拒绝。因此采用各自 latest，并分别分析。评估前后 checkpoint SHA-256 一致。

运行摘要里的单条训练记录为：5e-4 latest episode 224 Return `2886.02`、run best `3390.51`（best checkpoint 在 episode 208）；1e-3 latest episode 300 Return `2949.04`、run best `3493.79`（best checkpoint在 episode 292）。这些记录进度不匹配，不能当作配对 Return 比较。两份 run summary 的 latest training episode MOVE 比率分别为：5e-4 的 UAV0/UAV1 `12.7%/12.3%`；1e-3 的 `69.7%/48.0%`。这些是训练 rollout 指标，不是下文的 deterministic Validation 指标。

## 已确认的代码链路

1. `rollout.py` 的 `_uav_candidates()` 将候选 0 设为当前坐标 STAY，其余候选由 `uav_configuration.enumerate_uav_target_candidates()` 生成。候选使用预测车辆位置和当前 Cache 状态；生成器检查边界、单 slot 最大位移与 UAV 间安全距离。Micro 按固定顺序先选 UAV0，再在 UAV0 位置条件下生成/选择 UAV1 候选。
2. `policy.py` 的 deterministic Micro 规则先比较 STAY 与 MOVE 两组的总概率质量：STAY 组质量较大时选候选 0；MOVE 组较大时才在 MOVE 候选里按原始分数选择。STAY/MOVE 熵底限在 trainer 中是软正则，不是强制 MOVE 门槛。
3. 两组运行的 Micro Actor LR 都是 `3e-4`。`trainer.py` 将 Macro Actor 参数交给单独的 `macro_optimizer`，Micro Actor/Critic 参数交给 `micro_optimizer`；Macro LR 没有直接设置 Micro Actor 的学习率。Macro LR 可以经由 Cache/holding 策略改变后续 Micro 所看到的 Cache、候选集、状态与 reward 分布，这是可能的间接影响路径，不是本诊断能证明的因果结论。
4. `execute_configured_slot()` 接收策略指定的 Cache/UAV 配置，再调用正式的 slot 执行链。环境执行真实任务的 Matching/KKT，计算 task utility、时延、能耗和系统效用。代码中的系统效用为 `Q(t) - Cache cost - UAV movement cost`；诊断没有重写 Matching/KKT 或奖励公式。

核对位置：`src/ppo_simple/policy.py:122-157, 865-911`；`src/ppo_simple/rollout.py:279-292, 434-484, 872-905`；`src/ppo_simple/trainer.py:1007-1030, 1265-1386`；`src/decision/uav_configuration.py:323-430, 539-568`；`src/decision/slot_environment.py:493-502, 711-786`。

## Validation 方案与状态隔离

- 环境：项目 Conda Python `C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`，Python 3.11，PyTorch 2.11.0。
- 数据：仅 `environment_data/proposed_joint_validation` 的 episode 30001、30002；每条轨迹只推进前 10 个 slot，在零基 step 0、4、9 取样。每个 checkpoint 得到 6 个反事实状态。为控制成本，本次不是完整 episode Return 评价。
- 冻结动作模式：复用现有 `ppo_simple.evaluator.evaluate_pool()` 的 deterministic 路径。每个抽样状态中，A 是冻结策略实际选出的 UAV0/UAV1 位置；B 是双 UAV STAY；C 是三类固定规则从现有合法 MOVE 候选构造的代表动作：UAV0 MOVE 且 UAV1 在安全时 STAY（不安全时选最近的条件合法调整）、UAV0 STAY/UAV1 MOVE、以及 UAV0 MOVE 后重新枚举 UAV1 MOVE 的顺序联合 MOVE。
- 每条 B/C 分支均从该 slot 执行前保存的深拷贝开始；保留同一个上一步 Cache、PPO 选择的最终 Cache、真实任务、真实车辆位置、场景与物理参数。每个反事实都重新调用 `execute_configured_slot()`，不传共享 Matching/KKT 缓存；Cache 输入保持不变。
- 对改变 UAV0 的候选，先按新 UAV0 位置重新枚举 UAV1，再调用正式完整转移合法性校验；若 STAY 不合法则尝试条件合法 UAV1 移动。`execute_configured_slot()` 再次校验配置。所有 36 个 C 分支均执行成功，没有跳过的候选。
- 选择 C 候选只使用 rollout 已有的合法候选及其 proposal source/预测车辆位置；真实任务和真实车辆状态只用于执行分支并评价结果，没有用于指导冻结策略或挑选动作。反事实结果不写回 evaluator 主轨迹或正式环境状态。
- 一次完整 300-slot rollout 预尝试因运行成本中断，没有纳入结果。下表的 slot/执行数和耗时只统计最终完成的有界 Validation 诊断；该中断尝试的部分调用没有保留精确计数。

## PPO 实际 MOVE 比率

这里的 MOVE 比率来自每个 checkpoint 的 20-slot deterministic Validation 前缀；“任一 UAV MOVE”是按 slot 计，“合并 UAV 决策率”以 40 次 UAV 决策为分母。

| Checkpoint | UAV0 MOVE | UAV1 MOVE | 任一 UAV MOVE 的 slot | 合并 UAV 决策 MOVE |
|---|---:|---:|---:|---:|
| 5e-4 | 0/20 (`0%`) | 0/20 (`0%`) | 0/20 (`0%`) | 0/40 (`0%`) |
| 1e-3 | 20/20 (`100%`) | 0/20 (`0%`) | 20/20 (`100%`) | 20/40 (`50%`) |

抽样的 6 个状态在 5e-4 下全是双 UAV STAY；在 1e-3 下全是 UAV0 MOVE/UAV1 STAY。此处只描述两个 seed 的前 10 个 slot，不外推到 episode 后段。

## 同状态即时系统效用差

差值统一定义为“分支 − 双 UAV STAY”。正值表示该分支在当前 slot 的系统效用更高；时延与车辆能耗为每任务均值变化。`best checked C` 是在每个样本状态实际执行 3 个 C 分支后，按本 slot 系统效用选出的最高者，仅指被检查候选中的最高值。

| Checkpoint / 子集 | n | ΔReturn 均值 / 中位数 | Q1–Q3 | 范围 | 正 / 负 / 0 |
|---|---:|---:|---:|---:|---:|
| 5e-4，PPO 实际 MOVE − STAY | 0 | N/A | N/A | N/A | N/A（6 个抽样状态均 STAY） |
| 5e-4，PPO STAY 时 best checked C − STAY | 6 | `+0.005295 / +0.003719` | `[+0.000961, +0.008853]` | `[-0.000130, +0.013857]` | 5 / 1 / 0（83.3% / 16.7% / 0%） |
| 1e-3，PPO 实际 MOVE − STAY | 6 | `-0.001488 / -0.002849` | `[-0.004739, -0.000269]` | `[-0.010241, +0.011790]` | 2 / 4 / 0（33.3% / 66.7% / 0%） |
| 1e-3，PPO STAY 时 best checked C − STAY | 0 | N/A | N/A | N/A | N/A（6 个抽样状态均有 MOVE） |
| 1e-3，所有抽样状态 best checked C − STAY | 6 | `+0.007143 / +0.004488` | `[+0.003507, +0.010098]` | `[+0.002487, +0.016251]` | 6 / 0 / 0（100% / 0% / 0%） |

三类 C 动作各自相对 STAY 的均值如下；每行 `n=6`，括号内是 Return 差为正/负/零的状态数：

| Checkpoint | 被检查分支 | ΔReturn 均值 | 正/负/0 | Δtask utility | Δ平均时延 (s/任务) | Δ车辆能耗 (J/任务) | ΔCache cost | ΔUAV movement cost | ΔUAV movement energy (J) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5e-4 | C0：UAV0 MOVE，UAV1 保持/安全调整 | `+0.001762` | 5/1/0 | `+0.004622` | `-0.007730` | `-0.001542` | `0` | `+0.002860` | `+225.0` |
| 5e-4 | C1：UAV0 STAY、UAV1 MOVE | `+0.004041` | 4/2/0 | `+0.006901` | `-0.011376` | `-0.002270` | `0` | `+0.002860` | `+225.0` |
| 5e-4 | C2：顺序合法联合 MOVE | `+0.004512` | 4/2/0 | `+0.009242` | `-0.015296` | `-0.003052` | `0` | `+0.004730` | `+337.5` |
| 1e-3 | C0：UAV0 MOVE，UAV1 保持/安全调整 | `+0.003701` | 4/2/0 | `+0.006561` | `-0.010502` | `-0.002095` | `0` | `+0.002860` | `+225.0` |
| 1e-3 | C1：UAV0 STAY、UAV1 MOVE | `+0.004931` | 5/1/0 | `+0.007791` | `-0.012605` | `-0.002515` | `0` | `+0.002860` | `+225.0` |
| 1e-3 | C2：顺序合法联合 MOVE | `+0.006228` | 6/0/0 | `+0.010958` | `-0.017677` | `-0.003527` | `0` | `+0.004730` | `+337.5` |

1e-3 实际 PPO MOVE − STAY 的分量均值为：Δtask utility `+0.001372`、Δ平均时延 `-0.001832 s/任务`、Δ车辆能耗 `-0.000366 J/任务`、ΔCache cost `0`、ΔUAV movement cost `+0.002860`、ΔUAV movement energy `+225 J`，合成 Δ系统效用 `-0.001488`。因此这一小样本中，任务效用改善不足以覆盖移动成本。

5e-4 的 PPO 全选 STAY；best checked C 相对 STAY 的分量均值为 Δtask utility `+0.008778`、Δ平均时延 `-0.014673 s/任务`、Δ车辆能耗 `-0.002928 J/任务`、ΔCache cost `0`、ΔUAV movement cost `+0.003483`、ΔUAV movement energy `+262.5 J`，合成 Δ系统效用 `+0.005295`。该行是每个状态中事后选出的三候选最高值，不能视为某条固定 MOVE 分支的均值或全局最优。

## 候选数量与环境开销

| 每个 checkpoint | 数量 |
|---|---:|
| Validation seeds / 每 seed 前缀 | 2 / 10 slots |
| 实际 PPO rollout 环境执行 | 20 |
| 反事实状态数 | 6 |
| 双 STAY 分支 | 6 |
| 请求 / 实际执行的 C 联合动作 | 18 / 18（每类 6） |
| 反事实环境执行（STAY + C） | 24 |
| 合计 `execute_configured_slot()` 环境执行 | 44 |
| policy 候选生成时见到的合法 MOVE target 实例 | 216 |
| C 分支中条件重枚举的合法 MOVE target 实例 | 324（18 次条件重枚举） |

profile 中 5e-4/1e-3 的策略前缀环境执行耗时分别为 `3.592/0.718 s`，反事实 STAY+C 耗时分别为 `0.563/0.708 s`；其中 Matching/KKT 计时分别为策略前缀 `3.573/0.701 s`、反事实 `0.543/0.688 s`。整次 evaluator 前缀运行墙钟分别约 `100.97/73.75 s`，包含环境构建和策略推理，不等于单纯 Matching/KKT 时间。最终两组合计 12 个状态、36 个 C 联合候选分支、88 次 slot 环境执行。以上计数不包括此前被中断的完整 episode 预尝试。

## 判断与限制

| 判断 | 诊断支持度 |
|---|---|
| “保持不动已足够有效” | 不能作为普遍结论。5e-4 的 6 个 STAY 状态中，已检查候选有 5 个状态得到更高即时效用；但也有 1 个状态候选 MOVE 略差。1e-3 没有 PPO-STAY 抽样状态。 |
| “PPO 遗漏有收益的 MOVE” | 支持样本内判断。5e-4 在 5/6 个 PPO-STAY 状态中存在更好的已检查 MOVE；1e-3 的 PPO MOVE 在 4/6 状态低于 STAY，而每个状态至少一个被检查候选高于 STAY，说明当前 MOVE 的方向/候选在若干状态可改进。 |
| “移动对当前收益有利，但有长期影响” | 部分支持。某些已检查 MOVE 有正的一步效用增量，且任务效用改善伴随更低时延/车辆能耗；1e-3 实际所选动作的平均一步增量仍为负。新位置对未来 slot 的作用没有模拟，需单独做多步分支评估。 |
| “现有数据仍不足以判断” | 是。每个 checkpoint 只有 2 个 seed、6 个分支状态、3 个候选动作类别，并且只取前 10 个 slot；训练进度不齐，deterministic 验证与 stochastic 训练口径不同。 |

因此，相似的训练 Return 与截然不同的 UAV 行为并不矛盾：代码上 Macro LR 没有直接改变 Micro Actor LR；宏观 Cache/holding 学习路径可间接改变 Micro 的训练状态与奖励。观察到的一步效用差约在 `10^-3` 至 `10^-2` 量级，而 episode Return 在数千量级，说明 UAV 移动差异可能只占总 Return 的一小部分，Cache 与任务效用等其它贡献可能主导总 Return。这是基于量级的解释假设，不是由当前非配对 checkpoint 数据证明的归因。

未计算显著性检验；样本内最高候选只代表本次列出的少量候选。checkpoint 哈希在评估前后一致；无训练更新、checkpoint/正式日志/环境数据写入，未使用 Test 数据；临时诊断脚本与 evaluator 输出均已删除。
