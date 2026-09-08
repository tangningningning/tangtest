# ppo_simple_v2 首个 rollout origin 运行时瓶颈审计

审计范围：当前工作区代码、正式 `ppo_simple.trainer` 路径、真实 train origin 0。未修改 PPO 数学、系统模型、reward、Cache/UAV optimizer、Matching/KKT、候选空间或正式参数；未执行 PPO update，也未运行完整 16-origin chunk。

## 最终判定

```ini
BOTTLENECK_IDENTIFIED = YES
DOMINANT_STAGE = configuration -> prediction/surrogate evaluation -> worker-side Matching/KKT
WORKER_POOL_REUSED = YES
WORKER_RECREATION_OVERHEAD_PRESENT = YES
NESTED_MULTIPROCESSING_PRESENT = NO
ALGORITHM_COMPUTE_DOMINATES = YES
PROCESS_OVERHEAD_DOMINATES = NO
```

这里的 `WORKER_POOL_REUSED = YES` 指同一个 active origin 的
`SurrogateEvaluationContext` 内，多个 candidate evaluation 共用同一个
`PredictionBatchRuntime`/executor；它不是跨 origin 复用。`WORKER_RECREATION_OVERHEAD_PRESENT = YES`
指当前生命周期确实在每个有 configuration trigger 的 origin 结束时关闭、下一个 origin 再创建池，
不是说该开销是本次 origin 的主耗时。

## 运行方法和范围

- 使用当前正式 `PPOV2Trainer.collect_chunk(origin_start=0, rollout_steps=1)`。
- 使用当前配置中的 seed `4401`，新建正式 policy，执行正常 Bernoulli sampling；没有指定 action、修改 logits 或改变参数。
- 只执行 origin 0 的一个真实 slot。trainer 正常准备了一个 bootstrap observation，但没有执行 bootstrap slot。
- 没有调用 `PPOV2Trainer.update`，因此没有 PPO 梯度更新或 optimizer step。
- 计时只通过临时 wrapper 和现有的 `performance_profile`/
  `ConfigurationEvaluationProfile` 边界完成；有效结果返回后临时 profiler 会被删除。

## Origin 0 的 wall-clock 分解

有效运行的 origin 0 / slot 0 共执行 `239.536627 s`，且正常返回一条 transition；profiler 未记录异常。

| 阶段 | 秒 | 占 execute_slot wall time |
|---|---:|---:|
| policy sampling | 0.007904 | 0.003% |
| configuration 总计 | 239.444170 | 99.961% |
| 其中 Cache optimizer（2 次外层调用累计） | 52.377841 | 21.866% |
| 其中 UAV optimizer（2 次外层调用累计） | 186.272846 | 77.764% |
| prediction/surrogate utility evaluation（内含 worker 等待） | 238.616136 | 99.616% |
| worker `wait(futures)` 累计 | 222.518981 | 92.896% |
| worker pool 创建边界 `ensure_executor` | 0.002264 | 0.001% |
| worker pool close/shutdown | 0.784101 | 0.327% |
| 当前 slot 的实际 task Matching/KKT | 0.091348 | 0.038% |
| 当前 slot 的实际 KKT 函数累计 | 0.080240 | 0.034% |

表中 Cache/UAV、surrogate 和 worker wait 是嵌套/重叠归因，不能把这些百分比再次相加。
`worker_setup_seconds` 只覆盖 `ensure_executor` 边界；Windows `ProcessPoolExecutor` 的部分
spawn/初始化工作可能落在第一次 `wait` 中，因此不能把它解释成完整的进程启动成本。

## 本次 origin 的 action 和候选规模

正常 policy sample 产生了一个 Cache request 和一个 UAV request；该样本中实际 Cache update 和实际 UAV movement 都为 0。这些计数仅用于说明 profiling 经过了非 neutral configuration 路径，不作为本任务的环境行为审计。

| 字段 | 值 |
|---|---:|
| transitions | 1 |
| Cache requests | 1 |
| actual Cache updates | 0 |
| UAV requests | 1 |
| actual UAV movements | 0 |
| configuration summary 的 Cache candidate evaluations | 8 |
| configuration summary 的 UAV candidate evaluations | 78 |
| 当前 slot 实际 Matching candidate evaluations | 348 |
| prediction configuration Matching candidate evaluations | 3,727,699 |

`ConfigurationEvaluationProfile.source_stats` 进一步显示：

| source | requests | unique/misses | hits | prediction batches | Matching calls | candidate evaluations |
|---|---:|---:|---:|---:|---:|---:|
| cache | 21 | 16 | 5 | 1,600 | 1,600 | 618,506 |
| uav | 227 | 76 | 151 | 7,600 | 7,600 | 3,109,193 |
| joint | 3 | 0 | 3 | 0 | 0 | 0 |
| total | 251 | 92 | 159 | 9,200 | 9,200 | 3,727,699 |

因此本次并不是 worker 空闲等待：92 次并行批次提交每次包含 100 个 future，共 9,200 个
prediction batch；8 个 worker 在运行期间持续占用 CPU。profile 也显示同一 context 内存在
159 次 utility-cache hit，没有证据表明这些相同 configuration key 被无条件重复计算；主要成本
来自 92 个 unique predictive batch rounds 中实际执行的 worker-side Matching/KKT。

## Worker 生命周期核对

### 静态代码证据

1. `src/decision/slot_environment.py:228-282` 在存在任一 configuration trigger 时创建
   `SurrogateEvaluationContext`，调用 `optimize_joint_configuration`，随后在 `finally` 中调用
   `close_prediction_batch_pool(evaluation_context)`。
2. `src/decision/cache_configuration.py:565-623` 中，runtime 的 executor 初始为空；同一
   runtime 只有在第一次 `ensure_executor` 时创建 `ProcessPoolExecutor`，之后复用该 executor。
3. `src/decision/cache_configuration.py:756-862` 中，每个 surrogate context 自己创建
   `PredictionBatchRuntime`，并提供 context 级 close。因此池复用粒度是一个 active origin 的
   configuration context，不是单个 candidate，也不是整个 rollout chunk/训练过程。
4. `src/decision/cache_configuration.py:915-959` 为每个 cache miss 构造一个 batch job 列表，
   `executor.submit` 每个 batch，并通过 `wait(..., return_when=FIRST_EXCEPTION)` 等待完成。
5. worker initializer 的 payload 包含 `context.batches`、local cache、Matching 配置、base
   cache nodes 和 configuration；每个 spawned worker 因而持有该 prediction/scenario payload 的
   自己的进程副本。`src/decision/cache_configuration.py:594-620`、`:686-726`。
6. `src/decision/cache_configuration.py:578-586` 明确以
   `multiprocessing.current_process().name != "MainProcess"` 检测外层 worker，并在 child
   process 中退回 serial；当前路径没有 nested prediction pool。
7. 正式配置 `system_config.toml:481-489` 将
   `algorithm.configuration.prediction_batch_parallel_workers` 设为 8。

### 动态证据

- `prediction_batch_mode = parallel`。
- requested/effective workers 均为 8。
- `ensure_executor` 调用 1 次，`close` 调用 1 次，`wait` 调用 92 次；每次 wait 都看到 100
  个 futures。
- 运行中的进程快照显示父进程约 548 MB，另有 8 个 Python worker，各约 519 MB，和上述池及其
  per-worker payload 相符。
- 因此可以确认：同一 origin 内池确实复用；跨 origin 的关闭/重建模式由静态 `execute_slot`
  `finally` 直接证明。池创建/关闭本身在本 origin 中约 `0.786 s`（约 0.328%），不是主瓶颈，
  但它是可确认的纯工程生命周期开销。

## “时间到底花在哪里”

本次 origin 的首要耗时不是 policy sample，也不是当前 slot 的实际 Matching/KKT，而是
configuration optimizer 为 cache/UAV candidate 反复调用 prediction/surrogate evaluation；
surrogate evaluation 内部把每个预测 batch 送入 8 个 worker，在 worker 中执行与实际链路一致的
Matching/KKT 评价。

按耗时从高到低排列：

1. prediction/surrogate evaluation：`238.616 s`；其中等待 8-worker batch futures：`222.519 s`。
2. UAV optimizer 的累计边界时间：`186.273 s`。
3. Cache optimizer 的累计边界时间：`52.378 s`。
4. worker close/shutdown：`0.784 s`；worker 创建边界：`0.002 s`。
5. 当前 slot 的实际 Matching/KKT：`0.091 s`。
6. policy sampling：`0.008 s`。

worker wait 时间包含 worker-side 计算、future 完成和跨进程返回，不能单独等同于 IPC 开销。
但 8 个 worker 同时 CPU active，累计 worker 返回的 surrogate Matching 计算时间为
`1,569.609 s`（这是 8 个进程的 CPU 时间累计，不是 wall time），且完成了 9,200 个 batch
Matching call；因此本次证据支持 `ALGORITHM_COMPUTE_DOMINATES = YES`，不支持
`PROCESS_OVERHEAD_DOMINATES = YES`。

## 限制和未执行项

- 只完成 origin 0；没有等待或运行完整的 16-origin chunk，也没有运行多 episode 或 convergence test。
- 未单独给每个 spawned child 注入 profiler，因此无法把 `wait` wall time 精确拆成 spawn、IPC、
  worker compute 三个互不重叠的子项；现有 worker 返回的 per-batch compute 累计和动态 CPU 快照
  足以判定主导阶段。
- 没有修改 worker 数，也没有实施任何优化。

## 仅供后续评估的修复建议（本次未实施）

当前代码静态上直接证明了“每个有 trigger 的 origin 创建/关闭一套 prediction pool”。若后续单独
授权纯工程优化，可评估把 executor 生命周期提升到 rollout chunk 或 training-run 级别，并处理
payload 的内存占用；这会改变资源生命周期而非 PPO 数学，但本报告不实施该修改。
