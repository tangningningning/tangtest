# 历史缓存容量敏感性短时实验报告

> 历史记录声明：本报告保留校准前的 A/B/C 容量对比结果，表中的容量和短时
> 指标不属于当前正式 baseline。当前正式 baseline 及 artifact 一致性以
> `system_config.toml`、`data/generated/generation_metadata.json` 和
> `reports/artifact_consistency_report.md` 为准。

## 1. 实验边界

本次历史实验仅临时调整 `system_config.toml` 中的四个 cache capacity 参数，
实验结束后已恢复为当时 baseline；该 baseline 后续已按 service catalog 比例
重新校准为当前正式值：TV=5581.583442 MB、SV=16744.750325 MB、
RSU=44652.667533 MB、UAV=33489.500649 MB。

- TV：2000 MB
- SV：6000 MB
- RSU：15000 MB
- UAV：10000 MB

以下输入和实现均保持不变：task generation、service catalog、缓存算法、reward 公式、PPO 网络、随机种子、prediction/task/trajectory artifact。

由于当前 Python 环境未安装 PyTorch，不能调用 PPO 策略采样；本次按“固定 seed + 短 episode 验证”的范围，使用既有 `slot_environment.execute_slot()` 做 1 个 slot 的确定性短时评估：

- seed：4401
- slot：20，`time_s=120.0`
- cache trigger：9 个动态 cache 节点全部触发（5 SV + 2 RSU + 2 UAV）
- UAV movement trigger：全部为 0
- 目标函数和实际 reward 仍由现有系统代码计算，没有替换算法

指标口径：

- `trigger`：该 slot 的 cache trigger bit 数量；本实验固定为 9。
- `cache change`：新旧 cache 状态逐节点比较得到的实际变化节点数，同时记录 bit flip 数。
- `hit ratio`：20 个实际任务中，最终匹配目标已经持有所请求 service 的比例；本地目标检查 TV fixed local cache，边缘目标检查最终 edge cache，Cloud 计为 miss。
- `latency`：该 slot 全部实际任务 `per_task_delay_s` 的平均值，单位 s。
- `reward`：现有 `system_utility = actual_task_utility - cache_cost - movement_cost`，这里报告单 slot 值。

## 2. 实验结果

| 实验 | TV (MB) | SV (MB) | RSU (MB) | UAV (MB) | trigger | cache change | hit ratio | latency (s) | reward |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| A baseline | 2000 | 6000 | 15000 | 10000 | 9 | 0 节点 / 0 slot / 0 bit | 7/20 = 35.000% | 0.735036660 | 6.219959298 |
| B small | 1000 | 4000 | 8000 | 6000 | 9 | 0 节点 / 0 slot / 0 bit | 4/20 = 20.000% | 0.970440707 | 2.858983127 |
| C large | 3000 | 8000 | 20000 | 15000 | 9 | 0 节点 / 0 slot / 0 bit | 14/20 = 70.000% | 0.438290477 | 10.875243807 |

三组实验的 `cache_cost` 均为 0，因为该固定 slot 中没有实际 cache 状态变化；因此本表 reward 的差异来自实际匹配结果和任务 utility，而不是更新成本差异。

## 3. 容量变化观察

- B 相比 A，容量收缩后 hit ratio 从 35.0% 降至 20.0%，平均 latency 从 0.735036660 s 增至 0.970440707 s，reward 从 6.219959298 降至 2.858983127。
- C 相比 A，容量扩张后 hit ratio 增至 70.0%，平均 latency 降至 0.438290477 s，reward 增至 10.875243807。
- 在同一固定 trigger 和同一 slot 下，A/B/C 的实际 cache change 都为 0。由此只能说明当前 deterministic initial cache 在该输入下没有被现有严格改善准则替换，不能据此判断长期 episode 的更新频率。

## 4. 已知数据一致性限制

历史实验当时使用的 artifact 与配置存在车辆规模不一致，因此结果只适合作为
旧输入上的受控短时敏感性对比，不能解释为当前 25 车辆正式 baseline 的完整实验。
后续校准任务已经重新生成当前 artifact；当前一致性检查见
`reports/artifact_consistency_report.md`，不应将本节历史限制套用于当前数据。

另外，TV local cache 是 episode 前生成并固定读取的 artifact。本次禁止重新生成 cache artifact，所以 B 的 `TV=1000 MB` 没有重新生成对应本地缓存；当前 service catalog 最小 service size 为 1038.138784 MB，若按生成器约束重新生成，1000 MB 也无法容纳一个完整 service。B/C 的本次运行主要反映 SV/RSU/UAV 动态公共 cache 容量变化，TV 容量值已按要求写入并恢复配置，但不会改写固定 local-cache artifact。

## 5. 结论

在本次固定 seed、固定 slot、固定全 cache-trigger 的短时评估中，容量对 hit ratio、latency 和 reward 有明显影响；小容量组表现较差，大容量组表现较好。但三组均没有产生有效 cache change，说明本次结果不能证明容量变化会自动提高更新次数。是否长期更新仍取决于外层 trigger、预测 utility 变化和现有 cache update cost/严格改善门槛。

本次历史实验没有提出或实施缓存算法或训练逻辑修改；若要评估当前 baseline
的长期更新频率，应使用当前一致的 artifact 和固定 policy/多 slot 评估，统计
`cache trigger` 与 `actual_cache_trigger` 的差异。
