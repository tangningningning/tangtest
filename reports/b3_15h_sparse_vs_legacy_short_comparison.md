# B3-15H：Sparse 与 Legacy PPO 短程对照

## 范围与运行合同

两组均从 Fresh 初始化运行，PPO 初始化 Seed 为 `27005`，训练 Episode Seed 为 `20001–20048`，共 48 个 Episode、每 4 个 Episode 更新一次，共 12 组更新。Macro Actor 学习率为 `5e-4`，每组 `update_epochs=4`。两组均以退出码 0 完成；Episode 指标各 48 条、Update 指标各 12 条、Slot 指标各 14,400 条。

| 策略 | Run | Architecture / Checkpoint | Update |
| --- | --- | --- | ---: |
| Legacy Factorized | `runs/b3_15h_legacy_factorized_48ep/` | 33 / 60 | 12/12 |
| Sparse Coordinated | `runs/b3_15h_sparse_coordinated_48ep/` | 34 / 61 | 12/12 |

每组都评价 `initial_checkpoint.pt` 与 48 Episode 的 `latest_checkpoint.pt`。确定性评价使用同一 Training Pool 的 Episode Seeds `20231–20236`，每个策略阶段各 6 个 Episode；不属于 Validation 或 Test 结果。Legacy Latest 的种子 20231–20234 保存在 `runs/b3_15h_eval_legacy_final_training_diag6/`，20235–20236 因原执行会话中断而保存在独立尾段目录 `runs/b3_15h_eval_legacy_final_training_diag6_tail_20235_20236/`，汇总时按 Seed 合并，未重跑前四个种子。

## 六个 Episode 的确定性评价汇总

Return 的标准差为 6 个配对 Episode 间的样本标准差。其他数值是 6 个 Episode 汇总值的算术平均。Cache 无变化帧比例取现有 `keep_rate` 字段，其实际语义是“没有发生 Cache 配置变化的 Macro 帧比例”，不代表采样到的全体 KEEP 动作比例。

| 策略 / Checkpoint | Mean Return ± SD | Task Utility | Cache Cost | UAV Cost | Holding | Macro 帧数 | 无 Cache 变化帧 | Cache 变化帧 | UAV0 / UAV1 MOVE 动作率 | Cloud / Edge 任务率 | 平均时延 (s) | 平均车辆能耗 (J) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy Initial | 3635.060 ± 57.147 | 3635.683 | 0.6233 | 0.0000 | 1.417 | 218.17 | 99.055% | 0.945% | 0% / 0% | 64.543% / 35.457% | 6.1369 | 0.112337 |
| Legacy Update 12 | 3708.524 ± 43.086 | 3710.240 | 0.0000 | 1.7160 | 1.000 | 300.00 | 100% | 0% | 100% / 100% | 55.558% / 44.442% | 5.2466 | 0.110127 |
| Sparse Initial | 3694.685 ± 45.822 | 3694.685 | 0.0000 | 0.0000 | 1.130 | 268.00 | 100% | 0% | 0% / 0% | 57.051% / 42.949% | 5.3577 | 0.121805 |
| Sparse Update 12 | 3694.722 ± 45.847 | 3694.725 | 0.0000 | 0.0031 | 5.000 | 60.00 | 100% | 0% | 0.556% / 0% | 57.051% / 42.949% | 5.3575 | 0.121768 |

UAV MOVE 列报告策略选出的非 STAY 动作率；当前保存的 Evaluator 摘要不包含 UAV 实际非零坐标位移率。UAV Cost 仍按正式执行结果记录。

## 逐 Seed Return 与配对差异

所有差值均按表头方向计算。Sparse–Legacy 是同一 Seed 下 `Sparse Update 12 − Legacy Update 12`。

| Training Seed | Legacy Initial | Legacy Update 12 | Legacy Δ | Sparse Initial | Sparse Update 12 | Sparse Δ | Sparse–Legacy Update 12 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20231 | 3655.5388 | 3759.1259 | +103.5871 | 3738.9397 | 3738.9397 | 0.0000 | −20.1862 |
| 20232 | 3563.3665 | 3666.6146 | +103.2481 | 3646.1904 | 3646.1904 | 0.0000 | −20.4242 |
| 20233 | 3584.4821 | 3681.0973 | +96.6152 | 3655.2589 | 3655.2589 | 0.0000 | −25.8384 |
| 20234 | 3721.3637 | 3750.9536 | +29.5899 | 3747.0589 | 3747.0589 | 0.0000 | −3.8947 |
| 20235 | 3624.8709 | 3664.0288 | +39.1580 | 3659.8964 | 3659.8964 | 0.0000 | −4.1325 |
| 20236 | 3660.7353 | 3729.3266 | +68.5914 | 3720.7672 | 3720.9863 | +0.2191 | −8.3403 |

| 配对比较 | Return Δ 均值 | 中位数 | Return 提高的 Episode |
| --- | ---: | ---: | ---: |
| Legacy Update 12 − Legacy Initial | +73.4649 | +82.6033 | 6/6 |
| Sparse Update 12 − Sparse Initial | +0.0365 | 0.0000 | 1/6（另 5/6 相同） |
| Sparse Update 12 − Legacy Update 12 | −13.8027 | −14.2633 | 0/6 |
| Sparse Initial − Legacy Initial | +59.6257 | +65.4044 | 6/6 |

净收益分解满足 `Return = Task Utility − Cache Cost − UAV Cost`：

- Legacy Update 12 相对 Initial：`ΔTask Utility=+74.5576`、`ΔCache Cost=−0.6233`、`ΔUAV Cost=+1.7160`，因此 `ΔReturn=+73.4649`。
- Sparse Update 12 相对 Initial：`ΔTask Utility=+0.0396`、`ΔCache Cost=0`、`ΔUAV Cost=+0.0031`，因此 `ΔReturn=+0.0365`。
- Sparse Update 12 相对 Legacy Update 12：`ΔTask Utility=−15.5156`、`ΔCache Cost=0`、`ΔUAV Cost=−1.7129`，因此 `ΔReturn=−13.8027`。

## 行为与短程学习判断

### Legacy Factorized

在这 6 个固定 Training 诊断 Episode 上，Legacy Update 12 的 Return 每个 Seed 都高于 Legacy Initial。与此同时，Task Utility 平均提高 74.56，平均时延下降约 0.89 秒；Cache Cost 从 0.6233 降至 0，UAV Cost 增加 1.716。确定性行为也明显变化：Holding 均值降至 1 Slot、Macro 帧数为 300，两个 UAV 的非 STAY 动作率升至 100%。这些变化与净 Return 提高同时出现，但该 6 Episode 诊断不能证明各行为变化单独造成了收益提升，也不能证明长期泛化。

### Sparse Coordinated

Sparse Initial 与 Update 12 的 Return 几乎相同：配对均值仅 +0.0365，只有 Seed 20236 提高 0.2191，其余 5 个 Seed 完全相同。两个阶段 Cache Cost 都为 0，Cache 变化帧比例都为 0；因此本轮没有观测到“通过减少 Cache Cost 获益”，也没有观测到训练后开始识别有益 Cache UPDATE。Holding 从均值 1.13 增至 5 Slot，Macro 帧数从 268 降至 60；UAV 非 STAY 动作仍接近 0。Task Utility、时延和能耗均基本未变。

Sparse 的 Initial Return 比 Legacy Initial 高 59.63（6/6 个 Seed 都高），但这两个策略使用不同 Architecture 和动作分布，该差异不能归因于 PPO 学习效果。Update 12 时 Sparse 比 Legacy 低 13.80（0/6 Seed 更高）：Sparse 的 Task Utility 低 15.52，UAV Cost 低 1.71，Cache Cost相同。Sparse 的低移动行为与较低 UAV Cost 相符；它是否因稀疏动作空间而失去有收益动作，当前评价不能单独确认。

## 指标可观测范围

- Evaluator 保存了 Cloud 与 Edge 总任务数/比例，但没有把其内部已计算的 SV、RSU、UAV 角色计数写入 Episode 摘要。因此本报告不能给出这三类节点的拆分，标记为 `NOT_RECORDED`。
- Episode 摘要没有逐 Macro 的实际更新节点数分布，也没有单独记录“采样到 ALL_KEEP 组”的次数。上表仅报告实际配置无变化帧及配置变化帧比例；Sparse 结构每帧最多更新一个节点，且本次观测到零 Cache 配置变化。
- 确定性 Evaluator 中记录的 UAV MOVE 是非 STAY 动作选择率，不是物理坐标位移率。未从现有输出推断位移。

## 结论

- A（Sparse 初始表现较好、训练后改善不明显）：在这组诊断数据上成立。Sparse Initial 对 Legacy Initial 的 Return 均值高 59.63；Sparse 自身训练前后只高 0.0365。
- B（Sparse Update 12 相对自身 Initial 有明显改善）：不支持；5/6 配对 Return 相同。
- C（Sparse 靠减少 Cache Cost 提高 Return）：不支持；两个阶段的 Cache Cost 都为零。
- D（Sparse 过度减少更新导致 Task Utility 明显下降）：证据不足。Sparse 在诊断 Episode 中没有发生 Cache 配置变化，但 Sparse Initial 也如此；不能把 Sparse 与 Legacy 最终策略的跨架构差异单独归因于稀疏更新约束。
- E（Legacy 与 Sparse 都没有学习改善）：不成立于这组初步证据；Legacy 在 6/6 Seed 上提高，Sparse 则未显示可观测提升。

这是 6 个 Training Pool 诊断 Seed 的探索性比较，不是独立 Validation/Test 结论，也不用于宣布最终算法优劣。Legacy 有继续长程验证的初步依据；Sparse 当前没有显示短程净收益改善。是否开展后续正式长程实验由用户决定，本轮未启动。

```text
LEGACY_48_EPISODES_COMPLETED = YES
SPARSE_48_EPISODES_COMPLETED = YES
BOTH_12_UPDATES_COMPLETED = YES
PAIRED_EVALUATION_COMPLETED = YES (4 policies × 6 fixed Training Episodes)
LEGACY_LEARNING_GAIN = +73.4649 mean Return; 6/6 pairs improved
SPARSE_LEARNING_GAIN = +0.0365 mean Return; 1/6 improved, 5/6 ties
SPARSE_VS_LEGACY_RETURN_GAIN = -13.8027 mean Return; 0/6 wins
CACHE_COST_REDUCTION = Legacy: 0.6233; Sparse: 0.0000
TASK_UTILITY_CHANGE = Legacy: +74.5576; Sparse: +0.0396; Sparse-vs-Legacy final: -15.5156
SPARSE_UPDATE_BEHAVIOR = 0% Cache-change macro frames; Holding=5 Slots; mean UAV non-STAY action rate=0.278%
SHORT_TRAINING_EVIDENCE = Legacy preliminary gain; Sparse no measurable gain; n=6 Training diagnostics
READY_FOR_LONG_TRAINING = LEGACY_CANDIDATE_ONLY (no training started)
```
