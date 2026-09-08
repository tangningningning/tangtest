# ppo_simple_v2 surrogate Matching/KKT runtime optimization

## 结论

```ini
WORKER_HOTSPOT_IDENTIFIED = YES
EXACT_OPTIMIZATION_IMPLEMENTED = YES
ALGORITHM_OUTPUT_EQUIVALENT = YES
BASELINE_ORIGIN_SECONDS = 239.536627
OPTIMIZED_ORIGIN_SECONDS = 244.359174
SPEEDUP = 1.001678x (paired clean rerun)
READY_FOR_NEXT_RUNTIME_OPTIMIZATION = YES
READY_FOR_FORMAL_PPO_EPISODE = NO
```

`BASELINE_ORIGIN_SECONDS=239.536627` 是上一轮 profiling 的固定参考值。为做严格前后输出比较，
本轮又在同一输入上完成了一次未优化 paired baseline：`244.769279 s`；优化后为
`244.359174 s`，paired speedup 为 `1.001678x`，约快 `0.1675%`。该幅度低于一次 worker/CPU
运行的自然波动，不能把它解释成大幅加速；它只验证了本次严格等价的低风险微优化没有改变搜索结果。

正式 episode 仍不应启动：origin 0 仍约需 244 秒，240 origins 的量级仍约为 16 小时，远未达到
适合重新尝试完整 episode 的运行时间。

## Git 状态

任务开始时：

- branch：`main`
- HEAD：`b7e90d4015bbeac6823eb55a4a29d71a7e174d7`
- 已有 working-tree 修改：`src/ppo_simple/trainer.py`
- 已有未跟踪正式报告和 `打包并上传.bat`；均未修改、删除或覆盖。

本任务唯一保留的生产代码修改是 `src/decision/task_allocation.py` 中
`solve_kkt_fraction_allocation()` 的正系数数组复用；`trainer.py` 的既有修改不属于本任务。

## 阶段 A：profiling 方法

- 入口：正式 `PPOV2Trainer.collect_chunk(origin_start=0, rollout_steps=1)`。
- seed：当前配置中的 `4401`。
- action：正常 Bernoulli policy sampling，未指定 action、未修改 logits、未调用 PPO update。
- environment：真实 origin 0 / slot 0；没有运行完整 16-origin chunk 或 episode。
- worker profiling：在 8 个 spawned prediction worker 中，每个 worker 只对首次处理的
  `batch_index=0` job 启用 `cProfile`，最多 8 个代表 job；没有对 9,200 个 job 做逐行 profiling。
- 另对一个 15-task 的代表 batch 用低侵入函数边界 wrapper 测量 transfer、exchange、KKT、
  channel、delay/energy、utility 等阶段。该 wrapper 只用于 profiling，未留在生产代码中。

普通未加 cProfile 的 paired baseline profile 为：

| 指标 | baseline |
|---|---:|
| `execute_slot` | 244.769279 s |
| configuration | 244.673136 s |
| surrogate utility evaluation | 243.828064 s |
| worker wait | 223.331195 s |
| prediction Matching calls | 9,200 |
| prediction candidate evaluations | 3,727,699 |
| utility-cache hits / misses | 159 / 92 |

上一轮参考运行的对应值为 `239.536627 / 239.444170 / 238.616136 / 222.519` 秒；
本轮后续比较使用 paired baseline，避免把机器负载差异误认为代码收益。

## Worker-side hotspot

8 个代表 worker job 的 cProfile 聚合结果如下。累计时间是嵌套的：例如
`evaluate_matching_with_kkt` 包含 `allocate_resources_kkt`，不能把各行百分比相加。
百分比以 8 个代表 job 的 `optimize_task_matching` 累计时间约 `3.0767 s` 为参照。

| 排名 | 函数/阶段 | calls | cumulative time | 占参照 | 结论 |
|---:|---|---:|---:|---:|---|
| 1 | `enumerate_best_operation` | 86 | 2.9719 s | 96.6% | 候选搜索外层；时间主要落入候选 KKT 评价 |
| 2 | `evaluate_matching_candidate_incremental` | 3,057 | 2.9173 s | 94.8% | 每个 transfer/exchange candidate 的增量评价 |
| 3 | `evaluate_matching_with_kkt` | 5,392 | 2.6236 s | 85.3% | KKT、资源结果和任务完成量评价 |
| 4 | `allocate_resources_kkt` | 5,392 | 1.9498 s | 63.4% | edge node CPU/BW 资源分配 |
| 5 | `solve_kkt_fraction_allocation` | 2,280 | 1.5440 s | 50.2% | 2,280 次 CPU/BW bisection |
| 6 | `alloc_for`（KKT 内部） | 73,449 | 0.8577 s cumulative / 0.3456 s self | 27.9% / 11.2% | 每轮 bisection 的数组构造、sqrt、clip、sum |
| 7 | `_channel_gain` | 18,426 | 0.3340 s | 10.9% | 物理路径/信道计算，次于 KKT |
| 8 | `edge_execution` | 14,808 | 0.1276 s | 4.1% | delay/energy 的 edge execution |
| 9 | `_base_rate` | 3,264 | 0.0833 s | 2.7% | task-node rate 计算 |
| 10 | `task_utility` | 14,829 | 0.0784 s | 2.5% | Q/utility 计算 |

代表 batch 的低侵入边界计时（优化后，同一数学路径）进一步显示：

| 阶段 | calls | wall time |
|---|---:|---:|
| worker-side `optimize_task_matching` / batch（cProfile job） | 1/job | 0.253–0.538 s |
| `build_feasible_targets` | 1 | 0.000636 s |
| `enumerate_best_operation` | 11 | 0.089071 s |
| transfer candidate evaluation | 330 | 0.064006 s |
| exchange candidate evaluation | 66 | 0.023824 s |
| candidate generation/bookkeeping residual | — | 约 0.001241 s |
| `evaluate_matching_with_kkt` | 704 | 0.080249 s |
| `allocate_resources_kkt` | 704 | 0.064919 s |
| KKT solve（CPU 151 + bandwidth 151） | 302 | 0.055624 s |
| `_channel_gain` | 2,455 | 0.008007 s |
| `_base_rate` | 441 | 0.002009 s |
| edge delay/energy execution | 1,970 | 0.002371 s |
| per-task utility/Q | 1,973 | 0.001570 s |

该 batch 有 15 个 task、396 个 candidate evaluation，其中 330 个 transfer、66 个 exchange；
9 次 accepted transfer update、0 次 exchange update、894 次无需进入 KKT 的 feasibility rejection。
worker 侧 9,200 个 batch Matching calls 和 3,727,699 个 candidate evaluations 在前后运行中保持不变。

## 重复计算和不变量分析

### 已确认的 candidate-independent 数据

| 数据 | 为什么不变 | 依赖项 | 本轮处理 |
|---|---|---|---|
| 一个 KKT solve 内的 `coeff` 和 `positive` mask | `coeff` 在 solve 内创建后不再修改，bisection 只改变 `lambda` | task cycles、node CPU、KKT 参数 | 复用 `coeff[positive]` |
| 一个 prediction batch 的 `Task` 对象 | `_evaluate_prediction_batch` 在进入 Matching 前只构造一次 | batch event 的 task metadata | 已有实现已复用，未重复改造 |
| 一个 prediction batch 的 service-node positions | 同一 worker job 的 UAV candidate 和 vehicle positions 固定 | candidate UAV positions、vehicle positions、node geometry | 已有实现已复用 |
| SV/RSU capacity、CPU、bandwidth、cloud 参数 | 来自 immutable `CacheOptimizationConfig` | project config | 未改动 |

### 明确不能跨所有 candidate 复用的数据

- Cache availability 会改变 `build_feasible_targets` 的 edge target 集合，因此不能使用一个跨 cache
  candidate 的完整 feasible-target cache。
- UAV position 会改变 UAV channel gain、base rate 和边缘路径 feasibility，因此这些量只能在
  固定 UAV candidate 内复用；本轮没有把它们提升为跨 candidate 全局 cache。
- local cache、service id、input size、cycles 等 task metadata 不随 candidate 改变，但现有实现已经
  在单个 batch 构造一次；没有发现“每个 candidate 重新构造同一 Task 列表”的问题。

### 现有 utility cache 和 KKT cache

- surrogate utility cache 的 key 已经 canonical：contiguous `int8` cache bytes 加有序 UAV position
  tuple；paired baseline/optimized 都是 `159 hits / 92 misses`，没有发现 dtype/list/order 导致的
  等价 configuration key 分裂。
- Matching 内已有 `KKTResourceAllocationCache`，以 `(node_id, sorted task ids)` 为 key。8 个代表
  job 的 cProfile 统计为约 5,404 次 cache request、1,140 次 node-task KKT compute，其余约 4,264
  次命中；本轮没有重复实现第二套 KKT cache。
- 仍然存在的低优先级开销是 incremental evaluator 的 matching dict、allocation dict 和 task-id
  集合复制；其累计时间明显低于 KKT bisection，且本轮只允许一个优化，未实施。

阶段 A 判定：

```ini
WORKER_HOTSPOT_IDENTIFIED = YES
MATCHING_DOMINATES_WORKER = YES
KKT_DOMINATES_WORKER = YES
CANDIDATE_ENUMERATION_DOMINATES = NO
DATA_RECONSTRUCTION_OVERHEAD_SIGNIFICANT = NO
REPEATED_EQUIVALENT_COMPUTATION_FOUND = YES
EXACT_REUSE_OPPORTUNITY_FOUND = YES
```

这里的 `REPEATED_EQUIVALENT_COMPUTATION_FOUND=YES` 特指 bisection 每轮重复执行同一个
`coeff[positive]` advanced-index extraction；不是指重复删除 candidate、减少 Matching 搜索，
也不是绕过已有 KKT cache。

## 阶段 B：实施的唯一优化

修改文件：`src/decision/task_allocation.py:1186-1217`。

原逻辑在每一次 `alloc_for(lambda)` 中执行：

```python
result[positive] = np.sqrt(coeff[positive] / lam)
```

现在在 `positive` mask 确定后执行一次：

```python
positive_coeff = coeff[positive]
```

并在 bisection 内只复用该数组：

```python
result[positive] = np.sqrt(positive_coeff / lam)
```

严格等价理由：`coeff`、`positive` 和 `positive_coeff` 在一次
`solve_kkt_fraction_allocation` 调用中均不变；每轮唯一变化仍然只有 `lam`。除去一次性搬出
完全相同的 NumPy advanced-index extraction，`sqrt`、除法、`clip`、`sum`、bisection 顺序、
tolerance、返回 allocation 和 feasibility 判定均未改变。没有改变 Matching candidate 数量、KKT
公式或精度。

## 前后输出严格等价验证

baseline 和 optimized 使用同一 seed、同一初始状态、同一正式 policy sampling；分别保存完整
origin 0 输出并逐字段比较。比较覆盖：

- raw action、action mask、effective Cache/UAV request；
- final/next cache state、final/next UAV positions；
- actual cache update、actual UAV movement；
- feasible targets、final matching map、transfer/exchange history；
- CPU allocation、bandwidth allocation；
- per-task delay、energy、utility、actual task utility `Q(t)`；
- cache cost、movement cost、system utility `G(t)`、PPO reward；
- transition metadata 和 bootstrap presence；
- utility-cache/KKT candidate counters（排除纯 timing 字段）。

结果：

```ini
ALGORITHM_OUTPUT_EQUIVALENT = YES
DIFFERENCE_COUNT = 0
MAX_ABSOLUTE_OUTPUT_DIFF = 0.0
MAX_RELATIVE_OUTPUT_DIFF = 0.0
```

两次运行的 action 都是 `[0,0,1,0,0,0,0,0,0,0,1]`；prediction Matching calls 都为
`9,200`，candidate evaluations 都为 `3,727,699`，utility-cache hits/misses 都为 `159/92`。
代表性 environment 结果也保持一致：`Q(t)=10.021667286124972`、
`G(t)=reward=10.015947286124971`，UAV movement 为 1，Cache update 为 0。

## 性能复测

| 指标 | 上一轮固定参考 | paired baseline | optimized |
|---|---:|---:|---:|
| origin `execute_slot` | 239.536627 s | 244.769279 s | 244.359174 s |
| configuration | 239.444170 s | 244.673136 s | 244.266211 s |
| surrogate evaluation | 238.616136 s | 243.828064 s | 243.362963 s |
| worker wait | 222.519 s | 223.331195 s | 217.180961 s |
| prediction Matching calls | 9,200 | 9,200 | 9,200 |
| prediction candidate evaluations | 3,727,699 | 3,727,699 | 3,727,699 |
| utility-cache hits/misses | 159/92 | 159/92 | 159/92 |

paired speedup：`244.769279 / 244.359174 = 1.001678x`。

本项优化只减少 bisection 中不变数组的重复构造，因此 Matching calls、candidate evaluations、
KKT 数学和 utility-cache 统计都不应改变；复测正是如此。相对上一轮固定参考值，
`239.536627 / 244.359174 = 0.980265x`，说明当前机器运行波动大于本项微优化收益，不能据此
宣称 origin 运行时间已经合理。

## 未处理的次级热点

1. `solve_kkt_fraction_allocation` 中每轮仍创建 allocation array 并调用 NumPy `clip/sum`；进一步
   优化可能收益更大，但需要单独证明浮点运算顺序和输出等价。
2. `evaluate_matching_candidate_incremental` 仍为每个 candidate 构造 matching、allocation 和
   per-task 字典；这是数据重建次热点，不能在本轮同时修改。
3. `_channel_gain`/`_base_rate` 对固定 candidate 的重复调用仍存在；跨 UAV candidate 复用不安全，
   需要单独设计 candidate-scoped immutable context。
4. 92 个 unique configuration 仍产生 9,200 个 prediction batch Matching calls；本轮没有减少
   scenario、batch、candidate 或搜索空间。

## 清理与限制

- 未运行完整 16-origin chunk、完整 episode、多 episode 或 PPO update。
- 已删除临时 cProfile hook、benchmark/equivalence/probe 脚本、JSON 结果和 profiling 数据；未修改
  worker 数、PPO 参数、system model、reward、Cache/UAV optimizer、Matching/KKT 数学或正式 artifact。
- 未 commit、push、merge 或 rebase。
- `git diff --check` 已执行；无 whitespace error，仅有仓库现有的 LF/CRLF 转换提示。
