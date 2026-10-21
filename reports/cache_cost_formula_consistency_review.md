# 受控 Run Cache 成本公式与单位复核

复核日期：2026-10-20。以 `runs/alternating_controlled_stop_updates2_20261020/run_config.toml`、`run_manifest.json`、`slot_metrics.jsonl` 和当前实现为准。共读取 2,400 条 Slot 记录（8 个 Episode）。没有读取 Validation/Test、重跑 Episode 或修改成本配置。

```ini
CACHE_COST_FORMULA_VERIFIED = YES
CACHE_COST_UNITS_CONSISTENT = YES
CACHE_COST_NUMERICAL_DISCREPANCY = ROUNDING_ONLY; LOADED_MB IS INFERRED, NOT DIRECTLY RECORDED
REWARD_DOUBLE_DEDUCTION = NO
```

## Run 的实际生效参数

受控 Run 配置与当前正式 `system_config.toml` 的 Cache 相关值一致：

- `objective.lambda_cache = 2.0e-5`，其作用是将秒制下载时延映射为无量纲 Reward 惩罚；
- SV、RSU、UAV 的 `cache_backhaul_rate_bps` 均为 `200,000,000 bit/s`；
- 节点数量为 5 SV、2 RSU、2 UAV；每节点各自使用配置速率。

这是按节点配置的速率。公式按各节点分别计算时延并求和；当前实现没有另行模拟共享回传链路的带宽竞争、下载队列或跨 Slot 下载完成过程。Cache UPDATE 后服务仍按现有机制立即可用，下载时延仅作为 Reward 重配置惩罚，没有叠加到每个任务的完成时延。

## 源码公式和单位

`src/common/system_model.py::cache_update_cost()` 使用：

```text
added[m,f] = max(new_cache[m,f] - previous_cache[m,f], 0)
loaded_mb[m] = sum_f added[m,f] * service_size_mb[f]
loaded_bits[m] = loaded_mb[m] * 8.0e6
T_ca[m] = loaded_bits[m] / cache_backhaul_rate_bps[m]
C_ca = lambda_cache * sum_m T_ca[m]
```

Service Catalog 的 `program_size` 是十进制 MB，`service_size_gb` 与其按十进制 1 GB = 1000 MB 对应；`cache_configuration.load_service_catalog()` 将 `program_size` 作为 `service_sizes_mb` 使用。因此 `MB × 8.0e6 = bit`、`bit ÷ bit/s = s`。未使用 Cloud 任务上传速率。

`additions` 只统计新配置中增加的服务，因此已有缓存内容不重复计入；删除服务不产生下载量。无状态变化时 additions 为零；源码也在 Slot 环节断言无缓存变化时下载时延和 Cache 成本必须为零。固定 UPDATE 成本、`T_scale_ca` 归一化以及旧 loaded-MB 能耗代理均未出现在现行 Cache 公式中。

数值例：新增 2 GB（2,000 MB）服务对应 16e9 bit，在 200 Mbps 下为 80 s，乘 `2e-5 1/s` 得 0.0016。新增 10 GB 为 400 s、成本 0.008。

## 受控 Run 的直接记录与反算

| 项目 | 数值 | 口径 |
|---|---:|---|
| Slot 指标行数 | 2,400 | 8 × 300 |
| `actual_cache_update_count` 合计 | 5,607 | 直接记录的实际状态改变节点数；不是 UPDATE 动作数，也不是 Trigger 触发数 |
| 至少一个节点实际状态改变的 Slot | 1,247 | 直接按节点计数是否大于零统计 |
| Cache 成本非零 Slot | 1,233 | 直接记录 |
| `cache_download_delay_s` 合计 | 1,497,722.394487 s | 直接记录求和 |
| `cache_cost` 合计 | 29.954447889741 | 直接记录求和 |
| `loaded_mb` 合计 | 37,443,059.862176 MB | 由直接时延反算：`sum(T) × 200e6 / 8e6`；Slot 日志没有直接写入 `loaded_mb` |

5,607 代表节点级实际缓存状态改变总次数。它不能被解释成 5,607 次 UPDATE 动作或 Trigger 事件。1,247 个状态改变 Slot 中有 1,233 个成本非零；其余 14 个 Slot 状态有变化但下载成本为零，与“只删除、没有新加载”的公式行为相符。

直接记录的 Cache 成本与 `lambda_cache × 直接记录时延` 一致，最大逐 Slot 绝对误差为约 `1.39e-17`。反算加载量四舍五入为 37,443,060 MB；用该圆整数反算得到约 1,497,722.4 s。故此前三个汇总值彼此吻合，差别只是舍入；其中加载量仍是依据统一 200 Mbps 速率的反算值，不是 Run 日志直接记录的实际 MB 总量。

## Reward 与重复扣除

`src/decision/slot_environment.py` 的实际 Slot Reward 组合为：

```text
G = task_utility - cache_cost - movement_cost
```

Cache 候选评价也调用相同的 Cache 下载成本函数。候选成本用于配置选择；最终实际状态 Reward 从实际任务效用中扣除一次 Cache 成本和一次移动成本。MAPPO Rollout 校验 `G` 与 Slot 结果一致并记录结果，不再做第二次扣除。

2,400 条 Slot 日志中 `G - (task_utility - cache_cost - movement_cost)` 的最大绝对误差为 0。合计为：

- Task utility：30,992.003593840633
- Cache cost：29.95444788974076
- Movement cost：1.8832500000000385
- G：30,960.165895950977

从现有源码、Run 配置和 Slot 记录未发现 Cache 成本重复扣除或某条实际 Reward 路径漏扣。该结论仅涵盖当前记录的 Slot Reward 与候选评价调用路径，不代表系统已模拟真实回传排队和物理下载完成过程。

