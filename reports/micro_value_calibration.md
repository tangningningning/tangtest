# Micro Value Calibration 与 GAE 诊断

日期：2026-10-09  
范围：只读复核已有冻结策略 Validation 诊断和训练更新汇总；本轮未启动环境 rollout、训练或优化器更新，未读取 Test 数据。

## 结论摘要

现有 Validation 报告没有逐 Slot 的 Micro Value 预测或实际奖励序列，因此无法计算该批冻结策略的 Monte Carlo Return-to-Go、Value Bias、MAE、RMSE、Explained Variance，也不能判断误差是否随剩余 Episode 时长变化。原 runner 曾在内存中持有每 Slot `old_value`、`reward`、`done`、GAE Advantage 及 GAE lambda-return，但成功后删除了临时缓存；正式报告只保留 Advantage 分组摘要。

可用的训练更新日志显示，两个 checkpoint 最后一次 Micro 更新的 **训练批内、更新后** Value 对 GAE lambda-return target 的 Explained Variance 较高（0.9933、0.9980），RMSE 分别为 55.77、31.99。预测均值高于 target 均值约 32.71、10.05。它们不是独立 Validation 或 Monte Carlo RTG 校准指标，不能排除 Validation、时段或动作状态下的系统偏差。

现有证据不足以确认 Micro Critic 存在系统性价值偏差，也不足以支持 Critic 偏差影响了 UAV STAY/MOVE 学习。训练批内指标没有显示明显的整体拟合失败；但本次没有保存验证状态逐步 Value/真实回报，故不能据此否定验证分布上的局部偏差。

## 数据留存核对

此前 `reports/micro_action_credit_diagnosis.md` 使用的冻结 Checkpoint 为 5e-4：224 Episodes / 56 updates，1e-3：300 / 75；训练进度不同，不作学习率因果比较。对应 Checkpoint SHA-256 见前述报告。

| 所需逐 Slot 数据 | 现存情况 | 对本次分析的影响 |
|---|---|---|
| rollout 时的 Value 预测 `V(s_t)` | 原 runner 曾取 `MicroTransition.old_value`，但只有已删除的临时缓存含逐 Slot 数值 | 不能计算独立 Validation Value 误差 |
| 每 Slot 实际 Micro reward 与 terminal 标记 | 原 runner 执行期间持有；正式报告只存总 Episode Return 与最后 Slot terminal 核对 | 不能从总 Return 还原逐 Slot RTG |
| Monte Carlo Return-to-Go | 未持久化 | 不能计算 Value Bias、MAE、RMSE、EV 或剩余时长趋势 |
| `compute_gae()` 返回的 lambda-return target | 原 runner曾短暂存为 `return_value`，但临时缓存已删除 | 不能重建 Value 对 lambda-return 的 Validation 校准 |
| 原始 GAE Advantage | 只保留总体、动作组和 Slot 段的汇总，不保留样本序列 | 可复核报告内 raw-GAE 汇总；无法精确还原标准化后的分组正负比例、分位数或逐 Slot 轨迹 |
| 动作与 Slot 段 | 报告保留各段样本数与 raw-GAE 均值/标准差 | 可作描述性分组，不可作同状态动作因果比较 |

此前 runner 的 `return_value` 是 `compute_gae()` 输出的 lambda-return（Advantage + 旧 Value），不是 Monte Carlo RTG。配置 `gamma=1`、`gae_lambda=0.95`，所以二者不能互换。训练器的 Micro 更新同样调用 `compute_gae()` 并以其 returns 作为 Critic target（`src/ppo_simple/trainer.py` 的 `_update_micro()`）。

本报告所指 Value Bias 定义为 `mean(V(s_t) − MC-RTG_t)`（正值表示预测偏高）；MAE、RMSE 与 EV 分别为平均绝对残差、残差均方根及 `1 − Var(V−target)/Var(target)`。这些指标本应在独立 Validation 样本上计算，但本次缺少必要的逐 Slot 序列。

### 可复用的训练更新汇总（不是 Validation）

现存 `runs/.../update_metrics.jsonl` 最后一条记录分别对应 episode 224、300。Trainer 在 Micro 更新后对本次训练 batch 的观察重新预测 Value，并与该 batch 的 GAE lambda-return 比较；这些是更新内拟合诊断，非 held-out 数据，也不是 MC RTG。

| LR / 最后更新 Episode | Value 预测均值 | GAE lambda-return 均值 | 均值差 V−target | 相关系数 | RMSE | Explained Variance | raw Advantage 均值 / SD | normalized Advantage 均值 / SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5e-4 / 224 | 1383.340 | 1350.633 | +32.708 | 0.99689 | 55.775 | 0.99335 | +72.219 / 137.308 | +2.23e-8 / 1.000 |
| 1e-3 / 300 | 1478.348 | 1468.300 | +10.049 | 0.99903 | 31.986 | 0.99805 | +26.884 / 94.485 | −1.27e-8 / 1.000 |

均值差只是整个训练 batch 的预测均值减目标均值，不等于按定义对 MC RTG 算得的验证 Value Bias。高训练 EV 和较低 RMSE 相对 target 尺度表明训练 batch 上总体拟合较好；它不检验时间段残差、Validation 泛化或动作条件残差。两 checkpoint 的 episode/update 数不同，数值也不应用于比较学习率效果。

## 可复核的 GAE 与 normalized Advantage

Validation 报告保留的 raw GAE 已按 UAV 动作和 Slot 0–99、100–199、200–299 分段汇总；其分组状态分布不同，尤其 UAV1 的候选支持及动作以 UAV0 已采样位置为条件。以下不将分组差异称为动作价值。

前一报告的 PPO 梯度诊断已将各 checkpoint 的全部 600 个 Validation 样本 Advantage 跨两个 Episode 按正式 `_normalize_advantages()` 标准化，整体均值约 0、标准差 1。虽然逐 Slot 数组不再存在，但可从报告中四舍五入到 5 位的 UAV0 STAY/MOVE raw-GAE 组均值和标准差合并估计批次 raw-GAE 均值/SD：5e-4 约为 67.10515 / 166.82071，1e-3 约为 48.39664 / 80.08071；再变换各 head/Slot 段已报告的均值/SD。下表是**按已舍入汇总量重建的近似值**；不代表重新计算的逐样本 normalized Advantage 分布。

| LR | Slot 段 | U0 STAY n / norm mean±SD | U0 MOVE n / norm mean±SD | U1 STAY n / norm mean±SD | U1 MOVE n / norm mean±SD |
|---|---|---:|---:|---:|---:|
| 5e-4 | 0–99 | 176 / +0.191±0.600 | 24 / +0.131±0.676 | 166 / +0.159±0.613 | 34 / +0.305±0.579 |
| 5e-4 | 100–199 | 176 / +0.595±0.311 | 24 / +0.461±0.372 | 168 / +0.604±0.313 | 32 / +0.447±0.332 |
| 5e-4 | 200–299 | 170 / −0.747±1.262 | 30 / −0.845±1.211 | 167 / −0.745±1.201 | 33 / −0.850±1.495 |
| 1e-3 | 0–99 | 99 / +0.762±0.412 | 101 / +0.835±0.454 | 109 / +0.789±0.472 | 91 / +0.811±0.387 |
| 1e-3 | 100–199 | 76 / −0.037±0.600 | 124 / −0.082±0.603 | 99 / −0.108±0.608 | 101 / −0.023±0.594 |
| 1e-3 | 200–299 | 82 / −0.932±1.164 | 118 / −0.596±1.076 | 110 / −0.672±1.120 | 90 / −0.810±1.127 |

这组标准化均值显示时段相对全 batch 均值的变化大于多数 STAY/MOVE 组间差。例如 1e-3 前段两种动作组均明显高于 batch 均值，中段约在均值附近，后段多数低于均值。它说明不能用 raw GAE 的“正值比例”直接判断 PPO 更新对某动作的偏好；对 PPO 梯度起作用的是每个样本的 normalized Advantage 与该样本 log-prob 梯度的乘积。表中只有近似分组矩，不含样本级协方差，也不构成动作因果价值。

### STAY/MOVE 与时间段的限制性解读

- 5e-4 的先前 raw-GAE 正值比例总体上 U0 STAY/MOVE 为 79.9%/79.5%，U1 为 79.4%/81.8%；1e-3 分别为 72.8%/72.9% 和 73.0%/72.7%。按时间段看，5e-4 中段各 head 组均值较高、后段下降；1e-3 前段均值较高、后段下降。标准化近似表也保留这一时段排序。
- 同一组内状态不匹配，UAV1 还条件于 UAV0 的动作/位置。组均值、raw 正负比例或标准化均值只能描述本策略访问到的状态；不能说明在相同状态下改变 STAY/MOVE 会改变回报多少。
- 每个 checkpoint 只有两个 Validation Episode。即使有逐 Slot 残差，按时段/动作再切分后的样本也有限；当前尤缺少 Value 残差对 UAV动作的同状态配对检验。

## 最终判断与下一步数据需求

**是否存在 Micro Critic 系统性价值偏差：当前无法判定。** 两条最后训练更新的 in-sample EV 较高，均值差为正，但它们既非 MC target，也非 Validation/时间段残差；不能作为存在或不存在系统性 Value Bias 的充分证据。

**现有证据能否支持 Critic 偏差影响 UAV 动作学习：不能。** 先前报告中同一共享 Advantage 作用于两架 UAV，且 STAY/MOVE 组 raw GAE 比例相近；本报告可重建的 normalized 组均值仍受 Slot 段和访问状态分布支配。没有验证样本的逐 Slot Value 残差，也没有在同一观测下把 Critic 误差与动作 log-prob 梯度或 MOVE 候选配对，故不能识别 Critic 是否对 UAV 动作更新造成系统性偏置。

建议在**下一次独立 Validation**的正常冻结 rollout 中增量保留紧凑逐 Slot 字段：Checkpoint 哈希、seed、slot、`old_value`（动作前预测）、Micro reward、done、UAV0/UAV1 STAY/MOVE、raw GAE、normalized GAE。rollout 结束后由 reward 序列直接反向累计 `gamma=1` 的实际 Monte Carlo RTG；另外单独保留 GAE lambda-return，以避免把两个 target 混淆。然后逐 checkpoint、按三个 Slot 段计算 `V−MC-RTG` 的 Bias、MAE、RMSE、EV，并按动作组提供样本量及描述性残差；报告整体及分段 normalized Advantage。无需为当前报告重跑已完成 Validation Episode。
