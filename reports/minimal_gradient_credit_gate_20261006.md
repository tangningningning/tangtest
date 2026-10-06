# 最小梯度 / 信用分配检查（固定 seed 复验）

上一轮因临时诊断器 autograd graph 生命周期错误未完成，本报告为修复检查器后的固定 seed 复验结果。

- 第三层状态：**通过**。固定 realization `episode_20001`、episode seed `20001`、base/sampler seed `27005`。
- 正式 optimizer step：`0`；正式 update / `_update_*` / `backward()` 均未调用。
- 正式 policy 参数保持不变：`True`；optimizer state 不变：`True`；training counters 不变：`True`。
- 完整诊断复验 episode：`True`；Micro slots `300`，Macro frames `98`。
- Advantage 正/负样本：Micro `206/94`；Macro `71/27`；finite：Micro `True`、Macro `True`。
- Ratio 最大 `|ratio−1|`：Micro `1.430511474609375e-06`；Macro Candidate `9.5367431640625e-07`；Duration `3.5762786865234375e-07`；Joint `9.5367431640625e-07`。
- 方向 witness 正/负通过数：Micro `5/5`、`5/5`；Candidate `5/5`、`5/5`；Duration `5/5`、`5/5`；Joint `5/5`、`5/5`。
- 零 advantage actor gradient norm：Micro `0.0`；Macro `0.0`。
- 梯度泄漏非零参数数：Micro Critic `0`；Macro Critic `0`；Trigger `0`。各项具体非空数、最大绝对梯度见 JSON。
- old log probability detached：`True`；advantage detached：`True`。
- 梯度检查 NaN/Inf：`0/0`；首次 PPO 失败：`无`。

## 方向 witness 数值

逐 transition 的 action、advantage、old/new log probability、ratio、loss、梯度范数、方向导数与有限性计数均保存在本报告对应 JSON 的 `runtime.direction_witnesses`。

## 静态公式依据

Micro ratio 使用 `exp(new_joint_log_prob - old_joint_log_prob)`；Macro Candidate 与 Duration 分别使用 `exp(new - old)`，Macro Actor loss 为 Candidate surrogate + Duration surrogate，使用同一 normalized actor advantage。direction witness 排除 entropy 和 Critic loss。

下一步：停止，等待进入第四层概率分布检查

## 源码与参数完整性

- 固定复验 `episode_20001` / episode seed `20001` / base sampler seed `27005`；当前正式源码 HEAD `6e1103dda56cee5b0aa4938f6ec41021998c41a8`。
- 正式 policy digest：`93217c2d36eaa4a5a18125f7d09a031a2d9bdc317a844491dd1205be89a0695a`；前后相同。optimizer state digest 与 training counters 前后相同，见 JSON。
- 正式源码 SHA256 与关键行号、每个 witness 的逐项梯度数值及 NaN/Inf 计数均收录于 JSON。

- Cleanup completed: 临时 runner、`__pycache__`、`*.pyc` 已删除；只保留本层最终 Markdown / JSON 诊断报告。未执行 git add / commit / push。
