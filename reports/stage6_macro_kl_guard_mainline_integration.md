# Stage6 Macro PPO 单步 KL 保护正式主线整合报告

日期：2026-10-12

## 结论

已将原型的 post-step Joint KL 验收、候选缩步和完整回滚逻辑整合到正式主线 PPOTrainer._update_macro，并保留原有 pre-step KL early-stop。5e-3 和 5e-4 两组隔离更新均通过；所有被接受的 Actor 更新 post-step Joint KL 都不超过 **0.15**。

5e-3 首次全学习率候选的 Joint KL 为 1.299087，候选被拒；Actor 参数与 Adam 状态在缩步重试前恢复。0.125 缩放候选（实际 LR 0.000625）以 KL 0.043164 首次接受。5e-4 的四个全学习率候选均直接接受。

## 整合内容

正式版和原型位于同一 Git 基线上。先对比两份 trainer.py，再将相关差异作为补丁应用到正式文件；没有用原型整文件覆盖正式文件，也没有改动原型工作树。

| 文件 | 改动 |
|---|---|
| src/ppo_simple/trainer.py | 保留 pre-step KL 检查；新增 post-step KL 候选验收、七档 LR 缩步、Actor 参数和 Adam state 回滚、Python/NumPy/PyTorch CPU/CUDA RNG 保存恢复；所有候选失败时恢复并跳过 Actor step；记录候选、KL、LR 缩放、接受/拒绝计数及停止原因。 |
| src/ppo_simple/reporting.py | 将训练控制机制版本写入正式 run_manifest.json。 |

关键位置：

- trainer.py:79 定义控制版本 `macro_actor_post_step_joint_kl_backtracking_v1`。
- trainer.py:1493 为 PPOTrainer._update_macro；pre-step 检查保留于 1845–1885 行。
- trainer.py:1736–1765 保存/恢复 RNG、Actor 参数和完整 Adam 状态。
- trainer.py:1944–2057 使用 1、1/2、1/4、1/8、1/16、1/32、1/64 候选缩放并验收 post-step KL。
- trainer.py:2059–2117 全部候选失败时恢复状态、清除梯度并停止该 Actor epoch。
- trainer.py:2260–2305 记录更新汇总；逐候选详情仍保留在 macro_kl_step_diagnostics。
- trainer.py:3193、3636、3879 将控制版本写入 Checkpoint 元数据、Run summary 和 update metrics。
- reporting.py:1787–1789 将同一版本写入 run manifest。

训练控制版本为 `macro_actor_post_step_joint_kl_backtracking_v1`。ALGORITHM_ID、网络架构和 CHECKPOINT_VERSION=58 均未改变；新增 Checkpoint 顶层元数据，不改张量形状、参数键或模型结构。既有 Checkpoint 文件未修改，可保留用于历史评价。新 Run manifest 同时记录当前控制版本和 resume 标记，因此旧 Checkpoint 续跑不会与新版本从头训练标为同一协议。

## 环境和数据

- conda activate adaptive_cache_mec 在当前 PowerShell 返回系统执行错误；已确认并使用 C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe。运行时 sys.prefix 为该环境路径，且存在 conda-meta。
- 配置未修改。原始 SHA-256：684bdfa4dc200db6227b2b00a7718797a98bf658e54d2d4dbef6f7d1f5babf77；语义 SHA-256：10f893fc549c4016bedb1a5389ad996f9a3ff646aa012596a7230c284b13f542。macro_target_kl=0.10，阈值为 0.15。
- 两组均使用初始化 seed 27005、正式 Training 池 environment_data/proposed_joint_training/ 中 Episode seed 20001–20004，并各调用一次 PPOTrainer.update_from_episodes。未加载 Checkpoint。
- 使用正式 _resolve_training_episodes、load_environment_context 和 collect_dual_timescale_episode 读取数据与收集 rollout；没有调用会写入正式 Run 或 Checkpoint 的 run_training。

两次验证的 rollout 摘要都与此前同 seed 的正式 5e-3 复现记录一致：

| Episode seed | Return | Task utility | Macro frames | Termination decisions | Environment / Matching calls | KKT calls |
|---:|---:|---:|---:|---:|---:|---:|
| 20001 | 4005.915421 | 4015.577196 | 152 | 289 | 300 / 300 | 31,795 |
| 20002 | 3995.165160 | 4003.741809 | 143 | 282 | 300 / 300 | 38,041 |
| 20003 | 3933.237815 | 3943.736686 | 175 | 293 | 300 / 300 | 44,590 |
| 20004 | 3923.366011 | 3934.968421 | 173 | 292 | 300 / 300 | 42,114 |

## 验证 A：Macro Actor LR = 5e-3

KL 均相对同一批 rollout old policy 计算。首步候选结果与独立原型报告相符。

| Actor epoch | LR scale | 实际 LR | Joint KL | Cache KL | Termination KL | 结果 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.0 | 0.005 | 1.299087 | 0.002416 | 0.690739 | 拒绝 |
| 1 | 0.5 | 0.0025 | 0.553528 | 0.000600 | 0.290003 | 拒绝 |
| 1 | 0.25 | 0.00125 | 0.166016 | 0.000149 | 0.086748 | 拒绝 |
| 1 | **0.125** | **0.000625** | **0.043164** | 0.000037 | 0.022679 | **接受** |
| 2 | 1.0 | 0.005 | 0.021146 | 0.002309 | 0.010117 | 接受 |
| 3 | 1.0 | 0.005 | 0.162806 | 0.012458 | 0.082960 | 拒绝 |
| 3 | **0.5** | **0.0025** | **0.021584** | 0.005829 | 0.008762 | **接受** |
| 4 | 1.0 | 0.005 | 0.037648 | 0.026292 | 0.006617 | 接受 |

- 接受 4 个 Actor 更新，拒绝 4 个候选；Adam step() 共调用 8 次，但每个 Actor 参数的 Adam step 计数均为 4，和接受更新数相同。
- 在 epoch 1、3 的缩步重试中，后续候选 step 前的 Actor 参数签名与 Adam 状态签名均和该 epoch 首个候选前相同，验证拒绝候选已正确回滚。
- 更新后优化器 LR 组恢复为 0.005。
- 在相同的 1,156 个 rollout Termination 决策状态上，平均 P(TERMINATE) 为 **0.506771 → 0.450537**，没有出现旧版非法更新导致的方向反转。
- 所有接受候选 post-step Joint KL ≤ 0.15；macro_kl_single_step_overshoot=false。

## 验证 B：Macro Actor LR = 5e-4

| Actor epoch | LR scale | 实际 LR | Joint KL | Cache KL | Termination KL | 结果 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.0 | 0.0005 | 0.027698 | 0.000024 | 0.014579 | 接受 |
| 2 | 1.0 | 0.0005 | 0.028316 | 0.000091 | 0.014796 | 接受 |
| 3 | 1.0 | 0.0005 | 0.012938 | 0.000206 | 0.006630 | 接受 |
| 4 | 1.0 | 0.0005 | 0.002868 | 0.000377 | 0.001271 | 接受 |

四个全学习率候选均直接接受，没有拒绝或缩步，说明机制没有无故拦截正常更新。每参数 Adam step 计数为 4，结束时 LR 组为 0.0005。相同 1,156 个状态上的平均 P(TERMINATE) 为 **0.506771 → 0.483807**。所有接受候选 KL 均不超过阈值。

## 检查与清理

- 使用项目解释器通过 trainer.py、reporting.py 的 AST 语法检查和模块导入检查。
- git diff --check 通过。
- 未启动 300 Episode 正式训练，未使用 Validation/Test，未写入正式 Run 或覆盖 Checkpoint。
- 未修改 PPO Clip、Advantage/GAE、Reward、Actor/Critic 结构、Cache/Termination 网络、Matching/KKT、系统配置或数据。
- RNG 保存/恢复逻辑已整合；本次 CPU 验证直接核实了 Actor 参数及 Adam 状态在拒绝重试后的恢复。
- 诊断脚本和临时补丁已删除。保留其他既有未提交文件及原型工作树状态。不 Commit、不 Push。


- 本次两组验证都有候选被接受，没有人为注入故障强行触发“七档候选全部失败”；该分支由 trainer.py:2059–2068 的状态恢复、清梯度和停止逻辑实现。
## 汇总

MAINLINE_KL_GUARD_INTEGRATED = YES

PRE_STEP_KL_CHECK_PRESERVED = YES

POST_STEP_KL_GUARD_ACTIVE = YES

ACTOR_ROLLBACK_VERIFIED = YES

ADAM_STATE_ROLLBACK_VERIFIED = YES

LR_5E_MINUS_3_VALIDATION = PASS

LR_5E_MINUS_4_VALIDATION = PASS

ALL_ACCEPTED_STEPS_WITHIN_KL_THRESHOLD = YES

OTHER_ALGORITHM_COMPONENTS_MODIFIED = NO

FORMAL_TRAINING_EXECUTED = NO

TEMP_FILES_CLEANED = YES

COMMIT_CREATED = NO

PUSH_EXECUTED = NO

