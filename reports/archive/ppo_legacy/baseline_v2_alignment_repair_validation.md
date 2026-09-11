# Baseline v2.0 模型对齐修复与最小验证

本报告记录 2026-09-05 完成的静态修复和无训练验证。验证期间未创建优化器，
未调用反向传播或 PPO 更新，未运行完整训练或 SUMO 仿真。

## 1. 代理模型与实际执行模型

缓存与 UAV 候选配置的预测效用不再使用逐任务独占 CPU/带宽的简化代理。
`cache_configuration.evaluate_surrogate_context()` 现在按每个
`scenario × horizon` 构造并发任务批次，并直接复用实际执行侧的：

- cache/service availability；
- Transfer-Exchange Matching；
- KKT CPU 与带宽分配；
- 单 Cloud 总 CPU/总上行共享；
- Local、SV、RSU、UAV、Cloud delay/energy/utility 公式；
- 跨 slot 队列等待时间。

Gate=NULL 或最终全零 trigger 时不需要进行配置选择，因而跳过预测配置评价；
结果中的 `predictive_configuration_utility/objective` 为 `None`，并通过
`configuration_summary.predictive_evaluation_skipped=true` 明确标识。实际
Matching、delay、energy 和 reward 仍正常执行。

## 2. T_ref / E_ref 重新校准

校准输入为当前 `data/generated/tasks.csv` 中 4210 个 `arrival=1` 任务。
任务 CPU-cycle 的 P95 为 `30,790,093,469.931866 cycle`。

| 参数 | 公式 | 正式值 |
|---|---|---:|
| `T_ref` | `P95(cpu_cycles) / 1.0e9` | `30.79009347 s` |
| `E_ref` | `1.0e-28 × P95(cpu_cycles) × (1.0e9)^2` | `3.079009347 J` |

## 3. 跨 slot 任务语义

正式配置为 `execution.cross_slot_task_model = "deterministic_batch_fifo"`，
推进时长唯一读取 `simulation.dt_s = 1.0 s`。同一节点在一个 slot 中接收的
任务按 KKT 资源分配并发执行；该批次最慢完成时间扣除一个 slot 后形成下一
slot 的剩余工作时间。新任务效用包含进入 slot 前已经存在的队列等待时间。

队列按 `20 个 TV Local + 9 个 cache/edge node + 1 个 Cloud` 的固定顺序加入
observation，使用 `workload / (workload + T_ref)` 归一化。因此 observation
维度由 154 更新为 184，action 维度仍为 11。

两 slot 无训练连续执行结果：

- slot 0 结束后有 5 个非零队列节点，最大剩余工作时间
  `16.595283469001075 s`；
- slot 1 的队列 observation 与 slot 0 的 next queue 完全一致；
- slot 1 队列特征非零，`state_propagation_ok=true`；
- 两个 reward 均为 finite：`12.399827003479004`、`8.987929344177246`。

## 4. 配置入口与 artifact 审计

任务、流行度预测、车辆预测和预测场景 CLI 均支持 `--config`，统一 artifact
入口会把同一配置路径传给下游生成器。`generation_metadata.json` 现在保存
配置文件 SHA-256、完整预测/节点部署签名和全部 artifact SHA-256。

`python scripts/regenerate_artifacts.py --check-only` 是只读检查模式，不更新
容量、不生成 artifact、不写报告。当前 11 项检查全部通过，包括角色、service
catalog、tasks、TV cache、node positions、预测 artifact、配置签名和文件哈希。

预测 artifact 失效时，统一入口默认拒绝隐式训练 popularity GRU；只有显式传入
`--allow-predictor-training` 才允许进入原有预测训练流程。

## 5. Hierarchical policy 最小端到端验证

验证入口：

```text
python src/ppo/ppo_rollout.py --config system_config.toml --hierarchical-e2e
```

本机默认 `python` 未安装 PyTorch，因此本次实际使用已有的
`C:\software\Python\python.exe` 执行同一命令；没有安装或更新依赖。

该入口从正式配置构造 `HierarchicalReconfigurationGatePolicy`，不创建优化器。
它从同一初始状态各执行一个 NULL 分支和一个 CONFIG-noop 分支，用于验证 Gate
到 WHAT、最终动作、环境和 reward 的完整接线。CONFIG-noop 强制 WHAT 为合法
全零向量，因此本项最小验证不执行非零配置搜索。

| 检查 | 结果 |
|---|---:|
| policy type | `HierarchicalReconfigurationGatePolicy` |
| observation / action dimension | `184 / 11` |
| NULL gate / CONFIG gate | `0 / 1` |
| `g=0 -> all-zero` | 通过 |
| `g=1 -> WHAT` | 通过 |
| Gate/WHAT rollout 字段保留 | 通过 |
| NULL 与 CONFIG-noop 环境结果相同 | 通过 |
| NULL / CONFIG-noop reward | `12.399827060852491 / 12.399827060852491` |
| old log-prob 重算最大误差 | `0.0` |
| old value 重算最大误差 | `0.0` |
| finite observation/log-prob/value/reward | 通过 |
| optimizer steps / backward calls | `0 / 0` |

## 6. 结论与边界

五项修复已接入正式代码路径，配置、artifact metadata、队列 observation 和分层
策略最小链路一致。当前结果支持继续进行“非训练的小规模非零 CONFIG 验证”；
在执行该验证并评估完整预测场景评价开销前，不建议直接启动 PPO 训练。
