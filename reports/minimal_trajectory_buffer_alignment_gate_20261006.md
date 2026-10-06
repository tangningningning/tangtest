# 最小轨迹 / Buffer / PPO 输入对齐检查

- 日期：2026-10-06
- 结果：通过
- 分支 / HEAD：`main` / `6e1103dda56cee5b0aa4938f6ec41021998c41a8`
- 工作区：任务开始时已跟踪文件修改 0 个、既有未跟踪路径 142 个；任务结束未跟踪路径 144 个（新增仅为本次 Markdown/JSON 报告），无源码修改。
- 实际正式入口：`C:\Users\tangning\Desktop\code\adaptive_cache_mec\scripts\training\run_unified_cache_cardinality_corrected_gate.py`；Unified Cache 与 Cardinality Correction 均启用；使用 V14b warm-start，策略冻结后只采样、不训练。
- 实际 episode：1；optimizer.step：0。
- Observation：Macro [400] / 网络期望 [400]；Micro [400] / 网络期望 [400]；NaN 0、Inf 0、abs max 1。
- Macro candidate 哈希：98/98 一致；candidate 0 与当前 Cache 一致 98/98；合法重复候选 0。
- Transition → Batch → `_tensor_batch`：Macro 与 Micro 字段顺序/取值均已逐行核对；Macro float32 转换最大误差 2.14891194e-05，Micro 6.16324533e-06。
- Macro old log probability 最大误差：9.53674316e-07（0 条超过 5e-6）。
- Micro old joint log probability：检查 300 slot，最大误差 1.43051147e-06（0 条超过 5e-6）。
- Observation 连续性：Micro 299 次、最大误差 0；Macro 97 次、最大误差 0。
- Macro/Micro 对齐：duration 覆盖 300 slot；Cache ID 不一致 0；Macro reward 重构最大误差 0。
- GAE 输入顺序：两级均按 rollout 时间顺序计算，随后才进行 minibatch 抽样；本 gate 未调用 GAE 或梯度路径。
- 第一处失败：无。

完整逐项数据和源码行号见同目录 JSON：`minimal_trajectory_buffer_alignment_gate_20261006.json`。临时 runner、`__pycache__`、`*.pyc` 已清理；未执行 git add/commit/push。

下一步：停止，等待进入第三层梯度检查。
