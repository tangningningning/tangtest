# Stage6 Macro PPO KL 修复后真实更新验证

验证日期：2026-10-12

## 结论

修复后的 pre-step KL 检查工作正确：首个 Actor step 后 Joint KL 已远超 0.15，下一 epoch 的 Actor step 被跳过。但检查只能拦住后续 step，不能约束已经发生的单步更新。本次第一次 Actor step 将 Joint KL 推至 **1.299087**，Termination KL 推至 **0.690739**；主要变化来自 Termination 分量。该单步之后，同一批 rollout 决策状态上的平均 TERMINATE 概率从 **0.506771** 降到 **0.071571**，方向是明显偏向 CONTINUE。

因此，5e-3 仍需要进一步控制单次 Actor 更新幅度。此次没有实现回滚、Trust Region、学习率调整或其他修复。

## 数据、环境与初始化

- 使用项目环境解释器 `C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`；运行时确认 `sys.prefix=C:\software\Anaconda_envs\envs\adaptive_cache_mec`。PowerShell 的 `conda activate adaptive_cache_mec` 在本机未能执行，因此按要求使用已核验的项目解释器。
- 配置为 `configs/proposed/system_config.toml`，正式 Training 池为 `environment_data/proposed_joint_training/`。运行前读取了正式 5e-3 Run manifest；主 seed 为 27005，`resume=false`，Macro Actor LR 为 0.005，正式计划为 300 Episode / 75 组更新，每组 4 个 realization。
- 当前配置原始 SHA-256 为 `684bdfa4dc200db6227b2b00a7718797a98bf658e54d2d4dbef6f7d1f5babf77`，语义 SHA-256 为 `10f893fc549c4016bedb1a5389ad996f9a3ff646aa012596a7230c284b13f542`，均与该正式 Run manifest 一致。
- 从全新策略初始化开始：按正式 seed 27005 调用 `_seed_everything`，以正式默认 CPU 设备创建 `PPOTrainer`，未加载第 88 Episode Checkpoint。没有调用正式训练启动器，也没有写入正式 Run 或 Checkpoint。
- 通过 `_resolve_training_episodes` 读取已有池中的 `episode_20001` 至 `episode_20004`，未生成或改写数据；随后使用正式 `load_environment_context`、`aligned_train_steps` 和 `collect_dual_timescale_episode` 收集 rollout。
- 四个 Episode 的回报、任务效用、环境执行数、Matching/KKT 调用数、Macro 帧数及 Termination 决策数均与原始 5e-3 Run 中相同 seed 的记录精确匹配。这确认了该次更新复现的是原始首次更新的初始化与 rollout 条件，而不是第 88 轮 Checkpoint 上的新更新。

| Episode seed | Episode 回报 | 任务效用 | Macro 帧数 | Termination 决策 / TERMINATE 数 | Matching / KKT 调用 |
|---:|---:|---:|---:|---:|---:|
| 20001 | 4005.915421 | 4015.577196 | 152 | 289 / 141 | 300 / 31,795 |
| 20002 | 3995.165160 | 4003.741809 | 143 | 282 / 125 | 300 / 38,041 |
| 20003 | 3933.237815 | 3943.736686 | 175 | 293 / 168 | 300 / 44,590 |
| 20004 | 3923.366011 | 3934.968421 | 173 | 292 / 165 | 300 / 42,114 |

## 正式 PPO 更新路径

隔离诊断调用了 `PPOTrainer.update_from_episodes`，与正式流程一样先更新 Micro，再进入 Macro 更新；Macro 更新中保留正式 Critic 与 Long Value 更新。所有对象都只存在于临时诊断进程中，没有持久化这些参数或指标。

代码路径：

- `src/ppo_simple/trainer.py:3208`：`_seed_everything`。
- `src/ppo_simple/trainer.py:3234`：正式 `run_training` 的环境解析、seed 初始化及训练器创建顺序。
- `src/ppo_simple/trainer.py:3458-3472`：按 seed 读取 realization 并调用正式 rollout。
- `src/ppo_simple/rollout.py:561`：`collect_dual_timescale_episode`。
- `src/ppo_simple/trainer.py:1185-1210`：`update_from_episodes` 调用顺序，包含 Micro 更新后调用 Macro。
- `src/ppo_simple/trainer.py:1490`：`PPOTrainer._update_macro`。
- `src/ppo_simple/trainer.py:1797`：Joint KL 超阈值时在 step 前记录停止原因并退出。
- `src/ppo_simple/trainer.py:1841`：实际 Macro Actor `optimizer.step()`。
- `src/ppo_simple/trainer.py:1856-1861`：执行 step 后重新计算并保存 post-step KL。
- `src/ppo_simple/trainer.py:2878-2886`：旧的聚合指标对实际 Actor epochs 取均值。

正式 `macro_target_kl=0.10`，停止条件是 Joint KL 大于 `1.5 × 0.10 = 0.15`；本次有效阈值确认为 **0.15**。配置中的 PPO clip、GAE、优化器参数和四 Episode 更新组保持原值。

## Actor step 级结果

KL 均按当前策略相对 rollout 的 old policy 计算。动作定义为 index 0 = CONTINUE、index 1 = TERMINATE（`src/ppo_simple/policy.py:640`）。

| Epoch / step 序号 | 执行 Actor step | 更新前 Joint KL | 更新后 Joint KL | 更新前 Cache KL | 更新后 Cache KL | 更新前 Termination KL | 更新后 Termination KL | 阈值 | 结果 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 / 1 | 是 | ≈0 | 1.299087 | ≈0 | 0.002416 | ≈0 | 0.690739 | 0.15 | 单步越界 |
| 2 / — | 否 | 1.299087 | — | 0.002416 | — | 0.690739 | — | 0.15 | step 前检测到 Joint KL 超阈值，正确跳过 |

本次 Macro Actor 实际执行 **1 step**。最大 post-step Joint KL 为 **1.2990870476**，最大 post-step Termination KL 为 **0.6907394528**。第一次 Actor step 自身就产生了严重越界：Joint KL 是阈值的约 **8.66 倍**，Termination KL 是阈值的约 **4.60 倍**。

Termination KL 约为 Cache KL 的 **286 倍**，因此这次 KL 跃升主要来自 Termination；Cache 分量也有变化，但量级很小。Joint KL 是联合 log-probability 计算结果，分量值不能简单相加替代 Joint KL。

## Early-stop 正确性与历史指标解释

- **pre-step 检查有效。** Epoch 2 的 pre-step Joint KL 为 1.299087，大于 0.15；当前实现没有执行 epoch 2 的 Actor optimizer step，停止原因记录为 `joint_kl_threshold_exceeded_before_step`。
- **单步越界仍发生。** 第 1 步开始前 KL 约为 0，step 后才升到 1.299087；pre-step 检查不能撤销这一步。
- 原始 Run 的首组指标记录了 2 个 Macro Actor steps，`macro_joint_approx_kl=0.6495435`、`macro_termination_ratio_approx_kl=0.3453697`、`macro_cache_ratio_approx_kl=0.0012081`。旧指标按已执行 epoch 对 step 前 KL 求均值；当前复现的首步 post KL 约为这些 Joint、Termination、Cache 均值的两倍。由于首 epoch step 前 KL 约为 0，而第二 epoch step 前 KL 已等于首步 post KL，旧日志的 0.6495 / 0.3454 **不是最终 post-step KL**，也不能证明首步没有严重越界。
- 旧训练记录中 Episode 5–8 的平均 Holding 为 1.25、TERMINATE 概率约 0.80；但那是在旧实现允许第二个 Actor step 执行之后的后续 rollout。本次修复后只执行首步，在相同的 1,156 个 rollout Termination 决策状态上，平均 `P(TERMINATE)` 从 **0.506771** 变为 **0.071571**（下降 **43.52 个百分点**），中位数从 0.506562 变为 0.070374。该次单步更新明显偏向 CONTINUE，**没有**造成向 TERMINATE 的 Holding 塌缩。
- 本次未重放被跳过的旧 epoch 2，也未再收集更新后的环境 Episode。故不能把旧实现后续出现的高 TERMINATE 概率单独归因于 epoch 2，也不能据此确认训练过程中后续组是否稳定。已确认的是：修复后的首步本身仍非常大，且其方向与旧日志中后续的短 Holding 行为不同。

## 判断与下一步

- pre-step KL guard：**正确**，超阈值后阻止了下一 Actor step。
- 单步 KL 越界：**是，且严重**；主要是 Termination 分量。
- 本次 Holding 方向：**偏向 CONTINUE**；基于相同 rollout 决策状态的动作概率，而非额外环境 rollout 得出的实际 Holding 时长。
- **建议后续优先控制单次 Actor 更新幅度。** 依据是首个 step 后 Joint KL 已达 1.299，而不是 Critic EV 或单独的学习曲线。此次只提供证据，不实施任何进一步控制策略。
- 训练执行边界：只收集 seed 20001–20004 的四个 Training Episode，并在隔离内存策略上调用一次正式 PPO 更新；未启动 300 Episode 训练，未使用 Validation/Test，未覆盖正式 Run 或 Checkpoint。没有修改正式代码、配置、Reward、Matching/KKT 或算法文件。本任务未触碰其他 Codex 文件。

## 汇总

KL_STOP_CORRECT = YES

SINGLE_STEP_KL_OVERSHOOT = YES

MAX_POST_STEP_JOINT_KL = 1.2990870476

MAX_POST_STEP_TERMINATION_KL = 0.6907394528

ACTOR_STEPS_EXECUTED = 1

EARLY_STOP_TRIGGERED = YES

HOLDING_POLICY_SHIFT = P(TERMINATE) 0.506771 -> 0.071571（−43.52 个百分点，偏向 CONTINUE）

FURTHER_KL_CONTROL_NEEDED = YES

FORMAL_TRAINING_EXECUTED = NO

FORMAL_CODE_MODIFIED = NO

TEMP_FILES_CLEANED = YES



