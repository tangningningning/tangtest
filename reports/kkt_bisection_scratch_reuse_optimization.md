# KKT bisection allocation scratch reuse audit

## Scope

本轮只检查并优化 `solve_kkt_fraction_allocation()` 的 bisection allocation scratch array 生命周期。未修改 KKT 公式、`sqrt/divide/clip/sum` 顺序、tolerance、最大迭代次数、停止条件、Matching candidate、prediction batch、worker 数、PPO、Cache/UAV optimizer 或 reward。

完整环境验证使用：

```ini
origin=0
seed=4401
rollout_steps=1
normal Bernoulli policy sampling
```

未运行完整 16-origin chunk、PPO update 或完整 episode。

## Static proof

在一次 `solve_kkt_fraction_allocation()` 调用内：

1. `coeff` 由输入转换为 `float64` 后只读；`positive_coeff = coeff[positive]` 在 bisection 前生成，此后不变。
2. `positive = coeff > 0.0` 只计算一次，bisection 内不变。
3. 新的 raw `allocation_scratch` 只初始化一次为全 0；每次 `alloc_for()` 仅写入 `positive` 位置，因此 non-positive 位置在 scratch 中始终为 0。
4. `allocation_scratch[positive] = sqrt(positive_coeff / lam)` 每次完整覆盖全部 positive 位置，不留下上一轮 positive 值。
5. `clipped_allocation` 通过 `np.clip(..., out=clipped_allocation)` 每轮完整写入；bisection 中间结果只用于紧接着的 `sum`、边界更新和最终局部变量，不逃逸到函数外。
6. 最终返回仍通过 `[float(value) for value in allocation]` 复制出 Python list；复用的 scratch 不会影响 final allocation。
7. `low/high` 更新、`mid` 计算、`sqrt`、`clip`、`sum`、tolerance 和 `max_iterations` 均未改动。

需要明确区分 scratch 与函数返回值：当前既有算法对 coefficient 非正的位置保留 `epsilon`，因为旧实现先用 `np.full(n, epsilon)` 再 clip。新实现保持这一语义：raw scratch 的 non-positive entries 为 0，而 clipped/final allocation 的对应 entries 仍为 `epsilon`。若把“non-positive entries 为 0”解释为要求改变最终返回 allocation，则该解释与当前既有算法不一致，本轮没有进行这种算法改变。

## Paired microbenchmark

固定输入覆盖 mixed positive/zero、all-positive、all-zero、单元素、空输入、epsilon infeasible、negative coefficient 和 NaN coefficient。每个 case 比较：

- allocation；
- feasibility flag；
- reason；
- bisection iteration count；
- high-bound expansion count；
- final lambda。

结果：

```ini
DIFFERENCE_COUNT = 0
MAX_ABSOLUTE_DIFF = 0.0
MAX_RELATIVE_DIFF = 0.0
```

代表性 mixed case（5000 次 solve）：

| implementation | solve seconds |
|---|---:|
| paired baseline | 0.930257900 |
| optimized scratch reuse | 0.824816000 |
| speedup | 1.127836875x |

当前 production function 的输出也与 optimized local implementation 完全一致；其单独计时为 `1.091365800 s`，该值只作为实现一致性 sanity check，不作为 paired speedup 的分母。

代表性 mixed case 的 bisection 轨迹为 `30` 次 bisection、`3` 次 high-bound expansion；baseline 与 optimized 完全相同。all-positive case 为 `31` 次 bisection、`6` 次 high-bound expansion；同样完全相同。

## Full origin paired equivalence

两次运行均使用相同的 origin、seed、初始 Cache/UAV 状态、policy 初始化和正常 Bernoulli sampling。采样 action 为：

```text
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
```

即 Cache trigger 为 `[0,0,1,0,0,0,0,0,0]`，UAV trigger 为 `UAV_0=0, UAV_1=1`。两次 action mask 完全相同。

逐字段比较覆盖：raw action、action mask、effective Cache/UAV requests、previous/final/next Cache、previous/final/next UAV positions、actual Cache/UAV changes、configuration diagnostics、matching map、CPU allocation、bandwidth allocation、per-task delay、per-task energy、per-task utility、Q(t)、Cache cost、movement cost、G(t)/reward、next state 以及 rollout fields。

```ini
DIFFERENCE_COUNT = 0
MAX_ABSOLUTE_OUTPUT_DIFF = 0.0
MAX_RELATIVE_OUTPUT_DIFF = 0.0
```

该 origin 的环境结果中 Cache trigger 虽然为 1，但该状态的 Cache candidate `best_delta_objective=0`，所以实际 Cache update 为 0；这是被比较的环境行为，未因本轮优化改变。UAV_1 的实际 movement 为 1。两次运行的 candidate evaluation、Matching、KKT 输出均保持一致。

## Performance

以下 baseline 是本轮同一输入、同一临时 worker-side KKT timing instrumentation 下重新运行的 paired clean baseline，不使用上一轮的 origin 时间作为唯一 baseline。KKT cumulative seconds 是 8 个 persistent workers 的 solve 时间累加，因此可以大于 wall-clock。

| metric | baseline | optimized | speedup / equality |
|---|---:|---:|---:|
| `execute_slot` seconds | 218.669593 | 203.422743 | 1.074952x |
| configuration seconds | 218.567692 | 203.320794 | 1.074989x |
| surrogate seconds | 217.861832 | 202.551093 | 1.075590x |
| worker wait seconds | 196.165539 | 185.064743 | 1.059983x |
| cumulative `solve_kkt_fraction_allocation` seconds | 892.978921 | 805.297920 | 1.108880x |
| `solve_kkt_fraction_allocation` calls | 2,935,780 | 2,935,780 | identical |
| actual allocation/Matching-KKT seconds | 0.100444 | 0.100097 | diagnostic only |
| prediction Matching calls | 9,200 | 9,200 | identical |
| candidate evaluations | 3,727,699 | 3,727,699 | identical |
| configuration utility cache hits | 159 | 159 | identical |
| configuration utility cache misses | 92 | 92 | identical |

The full-origin wall-clock remains minute-level. The observed speedup is attributable to the permitted scratch allocation reuse, with the rest of the workload and call counts unchanged; this result is not a convergence claim and is not a reason to tune PPO or KKT parameters.

## Cleanup and repository state

The worker-side KKT timing wrapper and return-tuple changes were temporary and have been removed. Temporary benchmark scripts and JSON outputs are removed after this report. The only production change from this task is the allocation scratch reuse in `solve_kkt_fraction_allocation()`; pre-existing changes and reports were preserved.

`git diff --check` was run after cleanup and passed with no output.

## Final status

```ini
SCRATCH_REUSE_SAFE = YES
EXACT_OPTIMIZATION_IMPLEMENTED = YES
ALGORITHM_OUTPUT_EQUIVALENT = YES
KKT_SOLVE_SPEEDUP = 1.127837x microbenchmark; 1.108880x full-origin cumulative
ORIGIN_SPEEDUP = 1.074952x
READY_FOR_NEXT_RUNTIME_OPTIMIZATION = YES
READY_FOR_FORMAL_PPO_EPISODE = NO
```

`READY_FOR_FORMAL_PPO_EPISODE = NO` because even one origin remains approximately 203 seconds; this task did not establish a practical formal-episode runtime.
