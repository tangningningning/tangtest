# KKT exact-reuse scope audit

## 结论摘要

本轮只审计 surrogate worker 中真正进入 `solve_kkt_fraction_allocation()` 的 KKT cache miss；没有修改 KKT 数学、Matching candidate 数量、prediction batch 数量、worker 数量或 PPO。

在 `origin=0, seed=4401, rollout_steps=1` 的真实路径上，按每个持久 worker 采样前两个完成的 worker jobs，共 16 个 jobs、8 个 worker，完整记录了其中 5,022 次实际 KKT solver invocation。5,022 个完整数值 fingerprint 全部唯一：没有观察到同一完整 KKT problem 在当前 Matching call、prediction batch、configuration evaluation 或不同 configuration evaluation 中重复进入 solver。

完整 origin 的低开销 counters 与既有结果一致：

| 指标 | 数值 |
|---|---:|
| prediction Matching calls | 9,200 |
| incremental candidate evaluations | 3,727,699 |
| KKT cache requests | 6,805,684 |
| current-cache hits | 5,337,794 |
| current-cache misses | 1,467,890 |
| 实际 KKT solver calls | 2,935,780 |

实际 solver calls 为每个 cache miss 的 CPU/BW 两次 KKT 求解。采样中的 5,022 次为 2,511 CPU + 2,511 BW，unique fingerprints 也是 5,022，duplicate occurrences 为 0。

本次带审计 instrumentation 的 origin wall-clock 为 200.461526 s；surrogate evaluation 为 199.735904 s，worker Matching 累计 CPU time 为 1,274.144993 s，worker wait 累计为 182.257895 s，采样汇总的 KKT solver 累计 time 为 795.772558 s。这些时间用于审计范围核对，不作为优化 benchmark。

因此没有足够证据扩大 cache 生命周期。本轮不实施 cache 修改。

## 审计状态

| 审计项 | 状态 | 证据/说明 |
|---|---|---|
| 当前 cache 的创建与传递范围 | **CONFIRMED** | `optimize_task_matching()` 每次调用创建一个 `KKTResourceAllocationCache`，并把同一个对象传给初始 evaluation、所有 candidate evaluation 及稳定性复检。 |
| 当前 cache 是否跨 candidate 复用 | **CONFIRMED** | 同一 Matching call 内共享一个 cache；不是每个 candidate 新建。 |
| 当前 cache 是否跨 batch/worker/surrogate context 复用 | **CONFIRMED** | prediction executor 可以持久化，但每个 prediction batch 的 `optimize_task_matching()` 都创建自己的 cache；没有跨 batch 或 worker 的 cache 共享。 |
| 完整 fingerprint 是否使用 exact numerical inputs | **CONFIRMED** | coefficients、epsilon、tolerance 使用 exact float64 dtype/shape/raw bytes；max iterations 使用 exact integer；没有 rounding 或 tolerance-based equality。 |
| 全部 1,467,890 个 miss 是否逐个写 fingerprint | **NOT VERIFIED** | 为避免在分钟级 origin 中引入高成本日志，只对代表性 worker jobs 完整采样；完整 origin 只累计 counters。 |
| 发现需要扩大 scope 的 exact duplicate | **FAILED（未发现）** | 代表性样本中 duplicate count 为 0，因此扩大 scope 的收益假设未成立。 |

这里的 **FAILED（未发现）** 表示“寻找可复用的跨 scope exact duplicate”没有通过，不表示环境或算法执行失败。

## 完整 fingerprint 定义

fingerprint 是完整 canonical JSON payload 的 SHA-256；只有 canonical payload 完全相同才计为 exact duplicate。payload 包含：

### 直接 solver 输入

- resource coefficients 的 dtype、shape 和原始 float64 bytes；
- `epsilon` 的原始 float64 bytes；
- `tolerance` 的原始 float64 bytes；
- `max_iterations` 的 exact integer。

### 用于证明数值来源和作用域的 provenance

- resource type：CPU 或 BW；
- node id 和 node role；
- 按 solver 输入顺序排列的 task id、vehicle id、service id、`cpu_cycles`、`input_bits`；
- task vehicle position 和 service-node position 的 exact float64 dtype/shape/raw bytes；
- CPU 的 `omega_t`、`time_ref_s`、实际 node CPU capacity；
- BW 的 `omega_t`、`omega_e`、`time_ref_s`、`energy_ref_j`、transmit power、noise power、ground/UAV channel parameters、实际 node bandwidth；
- KKT 最小 CPU/BW fraction、epsilon、tolerance、最大迭代次数。

当前 solver 的实际直接签名只有 `(coefficients, epsilon, tolerance, max_iterations)`；其余字段用于证明 coefficients 在当前调用链中来自哪一个 node/task/physical state。所有浮点比较均使用 exact bytes，没有 rounding、近似相等或 tolerance-based cache hit。

## 运行范围与结果

运行使用正式真实路径，不做 PPO update：

- `origin_start=0`；
- `seed=4401`；
- 正常 Bernoulli policy sampling；观察到的 action 为 `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]`；
- `rollout_steps=1`；
- prediction pool：8 个 requested/active worker；
- 每个 worker 采样前两个完成的 job，共 16 个 jobs；采样 batch index 为 `0, 1, 3, 5, 7, 9, 12, 14, 18, 23, 26, 30, 34, 39, 42, 46`；
- 采样 solver occurrences：5,022；CPU/BW 各 2,511；
- unique exact fingerprints：5,022；
- duplicate fingerprint keys：0；
- duplicate fingerprint occurrences：0。

按重复发生范围分类：

| 范围 | 定义 | exact duplicate keys |
|---|---|---:|
| A | 同一个 Matching call 内 | 0 |
| B | 同一个 prediction batch 内、不同 Matching call | 0 |
| C | 同一 configuration evaluation 的不同 prediction batches | 0 |
| D | 同一个 origin 内不同 Cache/UAV configuration evaluations | 0 |

在本实现中，一个 prediction batch 的实际 `_evaluate_prediction_batch()` 调用一次 `optimize_task_matching()`，因此 sampled Matching-call id 是按真实 `optimize_task_matching()` 标记的，而不是把整个 worker 生命周期误当作一个 Matching call。

## 当前 cache scope 的安全性

当前 key 是：

```text
(node_id, sorted(task_ids))
```

这个 key 只在一个 `optimize_task_matching()` 内安全，因为该调用中的 task numerical data、vehicle/service positions、cache configuration、KKT configuration 和 resource constants 不变；同一个 node 与同一组 task 会得到同一组 CPU/BW coefficients。cache entry 在该 Matching call 的初始 evaluation、transfer/exchange candidate evaluation 和稳定性复检之间复用。

将 cache 扩大到 prediction-batch、worker 或 surrogate-context scope，必须把完整 numerical state 放入 key，不能继续只使用 node/task identity。当前采样没有发现任何跨当前 scope 的完整 exact duplicate，因此没有理由承担更宽 scope 的 key 构造、内存保持和错误复用风险。

## 决策

```ini
CROSS_SCOPE_KKT_REUSE_NOT_WORTHWHILE = YES
```

未实施新的 cache；已有的严格等价 KKT scratch reuse 和 Matching allocation structure reuse 保留不变。本轮没有 paired baseline/optimized 运行，因为没有生产优化可比较；没有 KKT solve reduction，也没有 origin speedup 声明。

## 最终标志

```ini
EXACT_DUPLICATE_KKT_PROBLEMS_EXIST = NO
EXACT_DUPLICATE_COUNT = 0 (sampled 5,022 solver calls; all-miss population not fully fingerprinted)
CURRENT_KKT_CACHE_SCOPE = per optimize_task_matching invocation / per Matching call
SAFE_WIDER_REUSE_SCOPE_EXISTS = NO
EXACT_CACHE_OPTIMIZATION_IMPLEMENTED = NO
ALGORITHM_OUTPUT_EQUIVALENT = NOT_APPLICABLE
KKT_SOLVE_REDUCTION = 0
ORIGIN_SPEEDUP = NOT_APPLICABLE
READY_FOR_NEXT_RUNTIME_OPTIMIZATION = YES
READY_FOR_FORMAL_PPO_EPISODE = NO
```

临时 fingerprint instrumentation、采样 JSON 和 benchmark runner 已在本轮结束时删除；没有 commit、push、merge 或 rebase。
