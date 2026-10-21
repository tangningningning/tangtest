# B3-15G 稀疏协调 Cache 真实训练链路验证

## 结论

两种策略均完成了 4 个 Training Episode、每个 Episode 300 Slot，以及 1 组 PPO 更新。两组使用相同的初始化 Seed 27005、Episode Seeds 20001–20004、Training Pool 和共同训练参数；均从新初始化开始，未读取旧训练 Checkpoint。

| 策略 | Run | Architecture / Checkpoint | Policy Schema | Episode / Update |
| --- | --- | --- | --- | --- |
| 旧版因子化 Cache PPO | `runs/b3_15g_legacy_factorized_smoke_retry02/` | 33 / 60 | `...v4_multi_node_cache_factorized_v1` | 4/4 / 1/1 |
| Sparse Coordinated Cache PPO | `runs/b3_15g_sparse_coordinated_smoke/` | 34 / 61 | `...v5_sparse_coordinated_cache_v1` | 4/4 / 1/1 |

每组各有 4 条 Episode 指标、1 条 Update 指标、1200 条 Slot 指标，以及 `initial_checkpoint.pt`、`latest_checkpoint.pt` 和 `best_train_return_checkpoint.pt`。完成更新后保存的 Latest Checkpoint 合同版本与对应策略一致。

两组 Run Manifest 的 Training Pool 身份哈希相同，Episode Seed 范围、每 Episode Slot 数、4-Episode 更新组、Macro Actor 学习率 `5e-4`、Macro Actor 更新轮数 4，以及 Micro Actor/Critic 学习率均相同。实际更新使用的学习率为 Macro Actor `5e-4`、Micro Actor `3e-4`、Macro Critic `1e-3`、Micro Critic `3e-3`、Long-Value Critic `1e-3`。两组都记录初始化 Seed `27005`。正式 Cloud 通信合同（20 Mbps、时延系数 13、有限共享 Cloud CPU 40 GHz）、Reward、UAV、Holding、Matching/KKT 参数一致；有意差异限于策略动作/概率结构、版本合同和实验标识。两组均调用 Transfer–Exchange Matching 与连续资源分配 KKT。

## Episode 结果

下表是链路验证期间的 Training 指标，仅用于核对 Rollout 与成本记录，不用于判断策略优劣。

| 策略 | Episode / Seed | Return | Task Utility | Cache Cost | UAV Cost | Macro 帧数 | 实际更新节点数分布（节点数:帧数） | 平均 Holding |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 因子化 | 1 / 20001 | 3621.7577 | 3693.9199 | 71.4499 | 0.7124 | 148 | 0:0, 1:2, 2:8, 3:22, 4:41, 5:39, 6:23, 7:12, 8:1 | 2.027 |
| 因子化 | 2 / 20002 | 3598.2759 | 3672.0382 | 73.1170 | 0.6453 | 161 | 0:0, 1:4, 2:10, 3:31, 4:39, 5:44, 6:21, 7:12 | 1.863 |
| 因子化 | 3 / 20003 | 3446.3760 | 3521.9813 | 74.8847 | 0.7206 | 157 | 0:1, 1:2, 2:8, 3:20, 4:40, 5:51, 6:22, 7:12, 8:1 | 1.911 |
| 因子化 | 4 / 20004 | 3516.2191 | 3582.0382 | 65.1535 | 0.6656 | 153 | 0:1, 1:4, 2:15, 3:27, 4:36, 5:33, 6:26, 7:9, 8:2 | 1.961 |
| Sparse | 1 / 20001 | 3696.3942 | 3710.6602 | 13.5811 | 0.6849 | 149 | 0:18, 1:131 | 2.013 |
| Sparse | 2 / 20002 | 3637.8379 | 3652.1621 | 13.6422 | 0.6820 | 157 | 0:19, 1:138 | 1.911 |
| Sparse | 3 / 20003 | 3492.7067 | 3508.0510 | 14.6370 | 0.7073 | 149 | 0:17, 1:132 | 2.013 |
| Sparse | 4 / 20004 | 3575.5159 | 3590.3519 | 14.1381 | 0.6979 | 165 | 0:22, 1:143 | 1.818 |

Sparse 共记录 620 个 Macro 帧：76 帧没有实际 Cache 配置变化，544 帧有变化；实际更新节点数分布为 0 节点 76 帧、1 节点 544 帧、2–9 节点均为 0 帧。因此 `MAX_ACTUAL_UPDATED_NODES_PER_FRAME = 1`。旧因子化策略的 619 帧分布为 0:2、1:12、2:41、3:100、4:156、5:167、6:92、7:45、8:4、9:0，体现旧执行分支仍允许多个节点同时更新。

Sparse 动作合同在真实 Rollout 中保持成立：组动作 ALL_KEEP 对应原 Cache 矩阵，执行路径显式检查配置未变化；Sparse 分支对最终配置执行容量校验。两个 Run 的 1200 条 Slot 指标中，`actual_cache_update_count = 0` 却记录正 Cache Cost 的 Slot 数均为 0。

## PPO 更新与数值检查

两组的 1 次 Update 均正常提交，KL Guard 为 `accepted`，Macro Actor、Micro Actor 及三个 Critic 均执行更新；未发生 KL 拒绝或 Early Stop。

| 指标 | 因子化 | Sparse |
| --- | ---: | ---: |
| Macro Actor optimizer steps | 4 | 4 |
| Micro Actor optimizer steps | 40 | 40 |
| Macro Critic optimizer steps | 50 | 50 |
| Micro Critic optimizer steps | 40 | 40 |
| Long-Value Critic optimizer steps | 50 | 50 |
| Macro joint KL | 0.0100415 | 0.00718721 |
| Macro Clip Fraction | 0.06018 | 0.10161 |
| Macro Entropy | 20.7950 | 6.26507 |
| Micro KL | 0.00259117 | 0.00841242 |
| Micro Clip Fraction | 0.00417 | 0.14000 |
| Micro Entropy | 4.27670 | 3.91944 |
| Rollout Log-Prob replay 最大误差 | 1.90735e-6 | 9.53674e-7 |

两组真实 Rollout 的 Log-Prob replay 误差均小于原有 `5e-6` 断言。日志报告 `NaNInf=NO`；1200 个 Slot 的 `G`、效用、成本、时延、能耗和耗时字段均有限。

逐 Slot 核对得到：

- `G = Task Utility - Cache Cost - UAV Movement Cost` 的最大绝对误差为 0。
- 每个 Episode Return 与该 Episode 全部 Slot `G` 之和的最大绝对误差为 0。
- 两组均完成 1200 次 Slot 级 Matching；Update 指标记录的 KKT 调用数分别为因子化 119847、Sparse 114595。
- 每个 Rollout Slot 均完成正式 Matching/KKT；执行路径在 `step_feasible` 为假或系统效用缺失时会抛出异常，两组均以退出码 0 完成全部 1200 Slot，未触发该失败路径。
- Update 数值诊断均为有限值，Macro KL Guard 接受更新，Critic 诊断完整。

## 必要修复及范围

首次执行发现旧因子化策略的采样函数未返回动作样本，随后发现旧 Rollout 分支未写入 `node_p_keep` / `node_p_update`。在正式短训前只修复这两处直接阻断项（`src/ppo_simple/policy.py`、`src/ppo_simple/rollout.py`），恢复旧策略采样返回值并补齐其概率统计赋值。没有改动 PPO 目标、Reward、系统模型、Matching/KKT、正式配置或训练池。修复后，两分支真实 Rollout 和更新均完成。

两次早期失败尝试目录 `runs/b3_15g_legacy_factorized_smoke/` 与 `runs/b3_15g_legacy_factorized_smoke_retry01/` 已保留；各自仅含启动/初始化阶段文件，没有 Episode 或 Update 指标，未计入上述结果，也未覆盖任何旧 Run。

## 验证边界

未运行 48 Episode、300 Episode、Validation 或 Test。短训 Training Return 不用于比较算法效果；本报告只确认新旧策略真实训练执行链路、一次 PPO 更新、概率/Reward/资源合同及 Checkpoint 保存可用。因子化采集与更新耗时分别为 315.81 秒、8.50 秒；Sparse 分别为 340.56 秒、8.72 秒。已完成修改文件 AST 语法检查、`git diff --check`；未生成临时 Runner 或测试脚本，Numba 缓存文件数为 0。

## 最终状态

```text
LEGACY_FRESH_RUN_COMPLETED = YES
SPARSE_FRESH_RUN_COMPLETED = YES
PPO_UPDATE_COMPLETED = YES (两组各 1 次)
SPARSE_MAX_CACHE_UPDATES_PER_FRAME = 1
LOGPROB_CONSISTENCY = PASS
REWARD_CONSISTENCY = PASS
NUMERICAL_ERRORS = 0 observed
TRAINING_GATE_PASS = YES
READY_FOR_SHORT_COMPARISON = YES
TRAINING_EXTENDED = NO
VALIDATION_OR_TEST_STARTED = NO
```
