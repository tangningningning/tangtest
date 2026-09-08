# Matching incremental structure reuse optimization

## Scope

本轮只处理 `evaluate_matching_candidate_incremental()` 中 candidate 评价的重复结构构造/复制。没有修改 Matching 搜索空间、candidate 顺序或数量、KKT 数学/调用逻辑、PPO、reward、Cache/UAV optimizer、prediction batches 或 worker 数。

之前的 KKT bisection scratch reuse 保留不变。

## 阶段 A：静态结论与微观 profiling

### Candidate 影响范围

- Transfer candidate 只改变一个 task 的 old target 到 new target。它最多涉及该 task 的旧 edge node、新 edge node，以及这些 edge node 上的其他 tasks。
- Exchange candidate 只交换两个 task 的 targets。它最多涉及两个 task 的旧/新 edge nodes，以及这些 edge nodes 上的其他 tasks。
- Cloud 不共享 KKT 资源；Local/Cloud 的受影响 task 单独评价，不会耦合未受影响 tasks。

### 当前重复构造

当前 evaluator 在每个 candidate 中：

1. 将 `current_matching` 和 `candidate_matching` 各自完整转换为 KKT matching dict；
2. 完整复制所有 CPU/BW 外层 map 及其 node→task 内层 map；
3. 复制 CPU/BW resource-sum map；
4. 复制全部 per-task delay/energy/utility map；
5. 只对受影响 node/task 重新计算，然后重新拼装完整结果。

这些复制中，未受影响的 node→task allocation map 是 candidate-independent，且 evaluator 后续只会 `pop` 外层 node key 或以新 node map 替换，不会修改共享的 inner dict。per-task 结果 map 仍按原实现复制，本轮不改其表示。

### Representative real prediction batch

使用 origin 0 的真实 prediction state，选择 18-task 的最大 batch（该 aligned step 共 100 个 prediction batches）：

| stage | calls / entries | cumulative seconds | mean per call |
|---|---:|---:|---:|
| incremental evaluator | 531 calls | 0.131029600 | 246.76 µs |
| matching dict conversion | 1,062 dicts / 19,116 task entries | 0.002592400 | 4.88 µs/candidate |
| task-id set/list construction | 10,217 entries | 0.001282301 | 2.41 µs/candidate |
| affected-node discovery | 1,062 affected-node entries; 3,065 impacted-task entries | 0.002462799 | 4.64 µs/candidate |
| allocation full copy | 13,168 node entries; 46,728 task entries | 0.002531301 | 4.77 µs/candidate |
| KKT cache lookup | 1,010 calls | 0.086422099 | 85.57 µs/lookup |
| KKT evaluation | 1,006 calls | 0.113453901 | 112.78 µs/evaluation |
| per-task result dict construction | 47,790 entries | 0.003247400 | 6.12 µs/candidate |
| candidate selection/discard | 479 discarded candidates | 0.000237600 selection time | 0.45 µs/candidate |

The profile confirms that structure construction is not the dominant cost—the KKT path remains dominant—but allocation nested-map copying is a safe, measurable reuse opportunity. The current function has no mutation/rollback path; discarded candidates are independent result objects selected by the caller.

## 阶段 B：实施的唯一优化

将：

```python
{node_id: dict(values) for node_id, values in current_evaluation.cpu_allocations.items()}
```

和对应 bandwidth map 改为：

```python
dict(current_evaluation.cpu_allocations)
dict(current_evaluation.bandwidth_allocations)
```

这只复用未受影响 node 的 inner `dict`，同时保留新的 outer dict。`nodes_to_clear` 仍然只修改新 outer dict；受影响 node 的 inner map 后续由 node evaluation 的新 dict 替换。没有 mutate-and-rollback，因此不存在 rollback 状态泄漏。

### Safety proof

- `current_evaluation` 的 nested maps 在 evaluator 内从不原地写入。
- CPU/BW affected node 会先从新 outer map 移除，随后在成功评价后写入新的 node map；空 node 保持移除语义。
- CPU/BW resource sums 本来就是 scalar outer map，仍然完整 shallow-copy。
- per-task delay/energy/utility 仍然完整 shallow-copy 并在新 map 上 update。
- 返回的 `KKTMatchingEvaluation` 只读取这些 maps；后续 Matching 流程不会修改共享的 unaffected inner maps。
- candidate 数量、顺序、受影响 node 发现、KKT cache key、KKT evaluation 和最终结果拼装顺序均未改变。

## Micro paired result

相同真实 18-task batch、相同初始 state：

| metric | baseline | optimized |
|---|---:|---:|
| incremental evaluator seconds | 0.131029600 | 0.126171200 |
| full batch Matching seconds | 0.138112900 | 0.131525300 |
| evaluator speedup | — | 1.0385x |

Candidate 数、transfer/exchange 数、KKT calls、KKT cache requests 和 result semantics 均相同。

## Full origin paired equivalence

使用同一：

```ini
origin=0
seed=4401
action=normal Bernoulli policy sampling
rollout_steps=1
```

两次 action 均为：

```text
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
```

逐字段比较覆盖 final matching、transfer/exchange history、CPU allocation、bandwidth allocation、per-task delay、energy、utility、Q(t)、G(t)、reward、final Cache/UAV states、candidate evaluation counts、KKT cache hits/misses，以及完整 rollout/slot result。

```ini
DIFFERENCE_COUNT = 0
MAX_ABSOLUTE_DIFF = 0.0
MAX_RELATIVE_DIFF = 0.0
```

调用和缓存计数也完全一致：

```ini
incremental evaluator calls = 3,727,699 / 3,727,699
prediction Matching calls = 9,200 / 9,200
candidate evaluations = 3,727,699 / 3,727,699
KKT cache requests = 6,805,684 / 6,805,684
KKT cache hits = 5,337,794 / 5,337,794
KKT cache misses = 1,467,890 / 1,467,890
```

## Full origin paired performance

这是本轮重新运行的同输入 paired baseline；没有使用历史 `203.422743 s` 作为唯一 baseline。两次运行都使用相同的临时 worker-side timing instrumentation。

| metric | baseline | optimized | paired ratio |
|---|---:|---:|---:|
| `evaluate_matching_candidate_incremental` cumulative | 1,248.419874 s | 1,232.355189 s | 1.013036x |
| worker Matching time | 1,296.127017 s | 1,278.518545 s | 1.013773x |
| surrogate time | 200.100312 s | 200.884839 s | 0.996095x |
| worker wait time | 184.077058 s | 181.785001 s | 1.012609x |
| configuration time | 200.814551 s | 201.656604 s | 0.995824x |
| `execute_slot` wall-clock | 200.893234 s | 201.743906 s | 0.995783x |

The evaluator and worker-side Matching counters improve consistently. The single origin wall-clock pair is slightly slower on the optimized run, so no overall wall-clock gain is claimed from this one noisy pair; the optimization is retained because its targeted cumulative evaluator and worker Matching measurements improve while all outputs and call counts are exactly identical.

## Cleanup and final status

Temporary worker instrumentation, benchmark scripts, comparison scripts, JSON outputs and Python cache were removed. No full episode or PPO update was run. No commit, push, merge or rebase was performed. `git diff --check` passed with no whitespace errors.

```ini
STRUCTURAL_REUSE_OPPORTUNITY_FOUND = YES
EXACT_OPTIMIZATION_IMPLEMENTED = YES
ALGORITHM_OUTPUT_EQUIVALENT = YES
INCREMENTAL_EVALUATOR_SPEEDUP = 1.013036x full-origin; 1.0385x microbenchmark
ORIGIN_SPEEDUP = 0.995783x paired wall-clock
READY_FOR_NEXT_RUNTIME_OPTIMIZATION = YES
READY_FOR_FORMAL_PPO_EPISODE = NO
```

`READY_FOR_FORMAL_PPO_EPISODE = NO`：单个 origin 仍约 202 秒，尚未达到适合正式 episode 的运行时条件。
