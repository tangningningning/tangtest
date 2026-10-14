# Stage6 Macro PPO 第二次 Actor 更新方向反转诊断

日期：2026-10-12

## 结论

**复现确认：旧版第二个 Macro Actor step 将策略方向从 CONTINUE 反转到 TERMINATE。** 在同一批 1,156 个原始 rollout Termination 决策状态上，平均 `P(TERMINATE)` 从 Step 0 的 **0.506771** 经第一次更新降至 **0.071571**，再经第二次更新升至 **0.802726**。第二步的更新前 Joint KL 已为 **1.299087**（有效停止阈值 0.15），但旧版实现先执行 optimizer step、之后才检查 KL，所以第二步仍然发生。

首步造成了巨大的 CONTINUE 偏移；第二步则在重新计算的联合 PPO 目标梯度下明显反向。第二步后 Joint KL 相对 Step 0 降至 0.404782，但仍超过停止阈值；Termination KL 降至 0.218517，同样仍高于阈值。旧训练 Episode 5–8 约 0.80 的 Termination 概率与第二步后同状态测得的 0.802726 一致，因此第二步是旧版 Holding 快速缩短的**直接且有力的原因**。由于后续 Episode 的状态分布会变化，本报告将其对实际 Episode 5–8 Holding 的因果归类为 `SUPPORTED`，而非把固定状态策略概率等同于实际持续时长。

## 初始化与复现条件

- PowerShell 中 `conda activate adaptive_cache_mec` 失败。确认并使用 `C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`；运行时 `sys.prefix` 指向 `C:\software\Anaconda_envs\envs\adaptive_cache_mec`，该环境目录含 `conda-meta`。
- 配置为 `configs/proposed/system_config.toml`，配置 SHA-256：`684bdfa4dc200db6227b2b00a7718797a98bf658e54d2d4dbef6f7d1f5babf77`；语义 SHA-256：`10f893fc549c4016bedb1a5389ad996f9a3ff646aa012596a7230c284b13f542`。
- 使用正式 5e-3 协议：初始化 seed `27005`、Macro Actor LR `0.005`、Training seeds `20001–20004`、正式 PPO/优化器其他设置不变、`resume=false`。从全新初始化开始，未加载第 88 轮 Checkpoint。
- 旧 Trainer 来自 Git HEAD `ae4e420bbc1b2884f9bcf25994fef4cd68e1cf08` 中的 `src/ppo_simple/trainer.py`（旧文件 SHA-256：`9923281c36ae3d8931f37abb9858376727e6daa4952f3e7137af71914e712f97`），只放入隔离诊断副本使用。未修改当前正式 `trainer.py`。
- 通过正式的 `PPOTrainer.update_from_episodes` 更新接口对四条 rollout 执行一次更新组；保持正式更新顺序和优化器状态。Actor optimizer step 只在隔离内存中的 Trainer 上发生。将概率/KL评估限制在收集所得的同一批固定状态上，并确认评估不改变 Torch RNG。
- 更新前的 rollout 摘要与历史 5e-3 日志中同 seed 的值一致：

| Training seed | Episode reward | Task utility | Macro frames | Termination decisions / TERMINATE | Matching / KKT calls |
|---:|---:|---:|---:|---:|---:|
| 20001 | 4005.915421 | 4015.577196 | 152 | 289 / 141 | 300 / 31,795 |
| 20002 | 3995.165160 | 4003.741809 | 143 | 282 / 125 | 300 / 38,041 |
| 20003 | 3933.237815 | 3943.736687 | 175 | 293 / 168 | 300 / 44,590 |
| 20004 | 3923.366011 | 3934.968421 | 173 | 292 / 165 | 300 / 42,114 |

总计固定比较 **1,156** 个 Termination 决策状态。概率只在这些相同状态上比较，未将不同 Episode 或状态分布的概率变化当作同一策略的逐步变化。

## 旧版 KL 检查时序

Git 中真实旧版 `src/ppo_simple/trainer.py` 的 `_update_macro`：

1. 在第 1763 行执行 `self.macro_optimizer.step()`；
2. 记录当前 epoch 已计算出的 KL；
3. 在第 1785 行才检查 `records[-1]["joint_approx_kl"] > 1.5 * target_kl` 并停止。

该次更新的有效 `target_kl=0.10`，故阈值为 `0.15`。第一次 step 前 KL 约为 0，首步后 Joint KL 达到 1.299087。第二轮计算得到的更新前 KL 已经高于阈值，但旧代码仍先执行第二个 optimizer step，再依据这个超阈值 KL 退出。因此日志的“执行了 2 个 Actor steps”与实际更新一致；旧逻辑只能阻止再后续的 step，未能阻止当下这个超阈值 step。

当前正式文件保留先前已经修复的版本，没有为本次复现恢复旧错误：当前 `_update_macro` 在 [trainer.py](/C:/Users/tangning/Desktop/code/adaptive_cache_mec/src/ppo_simple/trainer.py#L1797) 先检查 pre-step KL，在 [trainer.py](/C:/Users/tangning/Desktop/code/adaptive_cache_mec/src/ppo_simple/trainer.py#L1841) 执行 step，并在其后记录 post-step KL（第 1856–1861 行）。

## 固定状态上的逐步结果

以下 Joint、Cache、Termination KL 均相对于 Step 0 rollout 的 old policy 计算。每个 step 的 `pre` 是该次 optimizer step 之前、相对 Step 0 的 KL；`post` 是该次 step 之后、仍相对同一个 Step 0 old policy 的 KL。Step 2 的更新前 KL 与 Step 1 的更新后 KL 相同。

| 时点 / 操作 | Actor step 执行 | Joint KL pre → post | Cache KL pre → post | Termination KL pre → post | 平均 P(TERMINATE) | 平均 P(CONTINUE) | 终止概率范围 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Step 0：更新前策略 | 否 | 0 → — | 约 0 → — | 约 0 → — | 0.506771 | 0.493229 | 0.481234–0.541264 |
| Step 1：第一次更新 | 是 | 约 0 → **1.299087** | 约 0 → **0.002416** | 约 0 → **0.690739** | **0.071571** | **0.928429** | 0.062018–0.235794 |
| Step 2：旧版第二次更新 | 是 | **1.299087 → 0.404782** | **0.002416 → 0.007900** | **0.690739 → 0.218517** | **0.802726** | **0.197274** | 0.573597–0.822647 |

| 固定状态上的策略量 | Step 0 | Step 1 | Step 2 |
|---|---:|---:|---:|
| Continue logit 均值（最小–最大） | 0.076646（0.015731–0.150336） | 1.440432（0.747018–1.520310） | -0.659082（-0.723704–-0.031116） |
| Terminate logit 均值（最小–最大） | 0.103742（0.043913–0.198132） | -1.128572（-1.198515–-0.428860） | 0.746389（0.246592–0.812891） |
| Terminate−Continue logit 均值 | 0.027096 | -2.569005 | 1.405471 |
| 相对前一步的平均 P(TERMINATE) 变化 | — | -0.435200（状态变化范围 -0.452395 至 -0.288138） | +0.731154（所有状态均上升） |

第二步后相对 Step 0 的 `P(TERMINATE)` 净变化为 **+0.295954**。Step 2 的概率差范围表示同一状态上 `P₂−P₁` 的最小值至最大值；均为正。由此可见反转不是状态抽样变化造成的。

### Cache 策略的伴随变化

Cache 的 KEEP/UPDATE 两组平均概率变化远小于 Termination：

| 固定状态上的 Cache 概率 | Step 0 | Step 1 | Step 2 |
|---|---:|---:|---:|
| KEEP 均值（最小–最大） | 0.500410（0.496312–0.504334） | 0.481301（0.459285–0.503903） | 0.484111（0.435177–0.526108） |
| UPDATE 均值（最小–最大） | 0.499590（0.495666–0.503688） | 0.518699（0.496097–0.540715） | 0.515889（0.473893–0.564823） |

第一次更新把平均 UPDATE 概率提高约 **1.91 个百分点**。第二次更新使 KEEP 回升约 **0.28 个百分点**、UPDATE 相应下降约 **0.28 个百分点**；Cache KL 从 0.002416 升至 0.007900。第二步确实引起了额外 Cache 策略变化，但量级明显小于 Termination 的反转。

### Actor 梯度与参数变化

| Actor step | 更新前参数 L2 范数 | 原始梯度范数 → clip 后范数 | 参数变化 L2 / 相对变化 | 单坐标最大绝对变化 |
|---|---:|---:|---:|---:|
| Step 1 | 14.673952 | 0.217591 → 0.217591（未触发 0.5 clip） | 1.494625 / 10.19% | 0.00499913 |
| Step 2 | 14.751685 | 0.781991 → 0.499999（触发 0.5 clip） | 1.106957 / 7.50% | 0.00500018 |

这里的参数变化是在各次 Adam step 前后测得。第二步虽被梯度裁剪，但其 Adam 参数步仍产生明显 Termination 方向反转；梯度裁剪本身并没有保证更新后 KL 留在阈值内。

## 关键判断

### A. 第一次 step 为何大幅偏向 CONTINUE？

旧初始化在这些 rollout 决策状态上接近均匀策略，平均 Terminate−Continue logit gap 只有 `+0.0271`，平均 `P(TERMINATE)=0.5068`。第一次 Macro Actor 更新后 gap 变成 `-2.5690`，平均终止概率下降到 `0.0716`。

这是这四条 rollout 在当时共同策略下进入 Macro 联合 PPO actor 目标后，实际反向传播梯度与 Adam 更新方向的结果。该 step 的 actor 原始梯度范数为 0.2176，未触发 0.5 梯度裁剪；学习率为 5e-3，Actor 参数 L2 改变量约为初值范数的 10.19%。因此第一次更新本身很大。此次没有另行做逐 Episode / 逐动作的优势归因实验，故不能把该方向再归因到某一个单独 Episode 或某类奖励项。

### B. 第二次 step 是否把终止概率重新提高？

**是，确认。** 在完全相同的 1,156 个 Termination 状态上，平均概率从 `0.071571` 升至 `0.802726`，所有状态的概率都上升；平均 Terminate−Continue logit gap 从 `-2.5690` 转为 `+1.4055`。这构成明确的策略方向反转。

### C. 第二次 step 前后 KL 是多少？

相对 Step 0 old policy：

- Joint KL：`1.299087 → 0.404782`
- Termination KL：`0.690739 → 0.218517`
- Cache KL：`0.002416 → 0.007900`

这些数值表示 Step 2 结束时相对初始策略的偏离程度，不是 Step 1 策略与 Step 2 策略两者之间的 KL。Step 2 的 pre-step KL 正是 Step 1 的 post-step KL；它开始前就已远超 0.15 停止阈值。

### D. 第二步是不是旧版 Holding 快速缩短的重要直接原因？

**对策略方向反转是 CONFIRMED；对 Episode 5–8 实际 Holding 的因果作用是 SUPPORTED。** 第二步后的固定状态平均终止概率 0.802726，与旧版 Episode 5–8 日志约 0.80 的 Termination 概率相符，且第二步直接将策略推向 TERMINATE。之后 rollout 会进入由新策略及环境交互形成的状态分布，因此不能仅凭固定状态回放断言 Episode 5–8 的实际 Holding 全由第二步造成。

### E. 第二步是否额外改变 Cache 策略？

**是，但幅度较小。** Cache KL 继续上升，平均 UPDATE 从 0.518699 回落到 0.515889；这不是完全没有变化，但相较 Termination 概率从 0.071571 升至 0.802726，Cache 变化不是 Holding 方向反转的主要部分。

### F. 如果第二步没有造成反转，还有哪些 Holding 原因未验证？

本次已确认第二步确实造成同状态策略反转，所以无需假设另一个机制才能解释方向变化。实际 Episode Holding 仍会受更新后的状态分布、环境反馈和后续 PPO 更新影响；本次没有在更新后的策略下重新仿真 Episode，也没有把这些机制分解为各自的因果贡献。

## 证据级别与限制

- **CONFIRMED**：使用真实 Git 历史旧版 Trainer，在隔离副本中执行首批四 Episode 的一次 PPO 更新组；四个 rollout 摘要与旧 5e-3 日志一致；旧时序执行了两个 Actor steps；同状态 Termination 概率在第二步后从低值反转至约 0.803。
- **CONFIRMED**：旧版第二个 Actor step 开始前 Joint KL 已为 1.299087，大于 0.15；旧版代码在该 step 后才检查，所以该 step 实际发生。
- **SUPPORTED**：第二步是后续 Episode 5–8 短 Holding 的重要直接原因。概率方向和历史概率相符，但固定状态分布不等于之后实际 rollout 的状态分布。
- **UNKNOWN**：单独把首步 CONTINUE 梯度归因于特定 Episode、奖励项或 Termination 样本的贡献；本任务未运行逐项归因。
- **UNKNOWN**：若移除第二步，Episode 5–8 的真实 Holding 会是多少；没有对照环境 rollout。

未使用 Validation/Test；未运行正式 300 Episode 训练；未覆盖 Run 或 Checkpoint；本任务未改动当前正式 trainer.py（该文件保留此前 KL 修复）、配置、Reward、算法或系统模型；未触碰其他 Codex 报告。隔离诊断副本、临时脚本和中间文件已在任务结束时清理。


## 汇总字段

OLD_UPDATE_REPRODUCED = YES

FIRST_STEP_P_TERMINATE = 0.506771 → 0.071571

SECOND_STEP_P_TERMINATE = 0.071571 → 0.802726

FIRST_STEP_POST_JOINT_KL = 1.299087

SECOND_STEP_POST_JOINT_KL = 0.404782

SECOND_STEP_DIRECTION_REVERSAL = YES

SECOND_STEP_CAUSAL_ROLE = SUPPORTED

ADDITIONAL_ROOT_CAUSE = 旧版在首步 Joint KL 已达 1.299087（阈值 0.15）后仍执行第二个 Actor step；第二步使同一批固定状态的 Termination 策略反向转向 TERMINATE

FORMAL_CODE_MODIFIED = NO

FORMAL_TRAINING_EXECUTED = NO

TEMP_FILES_CLEANED = YES
