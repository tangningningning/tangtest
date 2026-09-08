# Cache exact prediction early-stop implementation audit

## 结论摘要

本轮依据上一轮严格 bound 审计，临时实现了 Cache candidate 的 exact
remaining-batch upper-bound rejection，并在 `origin=0,120,239` 上与当前完整
K=100 路径做了 paired replay。实现满足数学条件，三组正式环境输出逐字段
比较均为零差异；但是按固定 `chunk_size=5` 分块后，origin 120/239 出现了
明显的工程回退：提前拒绝的候选不能写入 exact utility cache，造成重复的
candidate evaluation，再叠加每个 candidate 多次 submit/wait，整体运行时间
显著增加。因此该生产实现已撤回，当前工作区恢复为原始完整 K=100 evaluator。

没有执行 PPO update、完整 PPO episode、commit、push、merge 或 rebase。

## Git 和范围

- branch: `main`
- HEAD: `b7e90d4015bbeac682ce7b55a4a29d71a7e174d7`
- 本轮开始时已有修改：`src/decision/task_allocation.py`、`src/ppo_simple/trainer.py`，以及此前正式报告；均未覆盖。
- 本轮 early-stop 生产代码在 paired 验证后已撤回；`src/decision/cache_configuration.py` 不再包含本轮改动。
- K=100 reference、prediction scenarios/batches、Cache candidate 集合和顺序、Cache objective/cost、Matching/KKT、PPO、reward、worker 数均未保留任何本轮行为改变。

## 允许的实现和安全性检查

临时实现只做了以下事情：

1. 普通 `evaluate_surrogate_context()` 调用仍完整评价全部 100 个 batch。
2. 只有 Cache candidate evaluation 显式携带 incumbent threshold 时，才按正式 scenario-major / horizon-minor 顺序，以 5 个 batch 为一个 chunk 提交。
3. 每个已完成 chunk 仍按正式 batch index 顺序累加；worker 完成顺序不参与浮点 reduction。
4. Cache update cost 在评价前精确计算，并从 upper-bound threshold 中扣除；没有使用 partial objective 更新 incumbent。
5. 只在

   ```text
   UB_k <= current_best + objective_tolerance
   ```

   时返回 `CERTIFIED_NON_IMPROVING`。该状态不携带伪造的 partial objective；否则继续到 batch 100 并返回 `EXACT_OBJECTIVE_AVAILABLE`。
6. 最坏情况仍执行全部 100 个 batch；worker exception、resource infeasibility 和 future error 继续走正式错误传播路径。UAV evaluator 没有启用 early-stop。

数学依据是每个 resource-feasible task 的 utility 不超过 1，因此剩余 batch
的严格上界为：

```text
remaining_upper = sum(beta**h_b * len(batch_b.events) / num_scenarios)
```

实现还对浮点上界向 `+infinity` 做了保守的 representational guard，避免
由于 bound rounding 向下而错误拒绝 candidate。

## Paired protocol

每个 origin 都从同一个 `context.initial_cache`、`context.initial_uav`、
scenario slice、actual task/vehicle state 开始，并复用上一轮正常 Bernoulli
sampling 得到的固定 action。seed 为 4401；没有 PPO update，也没有人工修改
正式 action 的 environment 结果。

```text
origin 0:   [0,0,1,0,0,0,0,0,0,0,1]
origin 120: [0,1,1,1,1,0,1,1,0,0,0]
origin 239: [0,1,1,0,1,0,0,1,0,0,0]
```

baseline 是本轮开始时的完整 K=100 当前代码；optimized 是临时 chunked
implementation。两边都执行真实 `execute_slot()`、真实 configuration path
和真实 actual Matching/KKT。

## Candidate and batch results

“early-stopped candidates”是临时实现实际返回
`CERTIFIED_NON_IMPROVING` 的 Cache candidate 数；“full-100 candidates”是
实际执行 100 个 batch 后得到 exact objective 的 candidate 数；“cached exact”
是已有 exact utility cache 命中的 candidate，不把它误计为新执行的 100 个
batch。

| origin | Cache candidate calls | early-stopped | full-100 | cached exact | batches skipped by chunks | baseline prediction Matching calls | optimized calls | optimized call change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 15 | 15 | 0 | 0 | 180 | 9,200 | 9,020 | -1.9565% |
| 120 | 269 | 233 | 34 | 2 | 2,655 | 17,700 | 24,145 | +36.4124% |
| 239 | 119 | 105 | 8 | 6 | 1,235 | 6,200 | 10,165 | +63.9516% |

Prediction Matching calls are one per executed prediction batch. The paired
baseline/optimized total across the three selected independent origins was
33,100 versus 43,330 calls, an increase of 30.9063%.

### Theoretical replay versus actual chunk run

The previous observation-only replay, which assumed a candidate could stop at
the first certifying batch without chunk rounding, gave the following potential
skips:

| origin | replay-safe candidate count | theoretical skipped batches |
|---:|---:|---:|
| 0 | 15 | 219 |
| 120 | 210 | 3,198 |
| 239 | 101 | 1,382 |

The actual chunked run skipped 180/2,655/1,235 batches respectively. Those are
not net production savings: at origin 120 and 239, early-rejected candidates
were not inserted into the exact utility cache, so later visits to the same
configuration had to be reevaluated. This increased total executed prediction
calls despite the local skips.

## Paired wall-clock and compute measurements

All values below are from the same-round paired runs. `surrogate seconds` is the
existing configuration profile's cumulative utility-evaluation wall time;
`worker cumulative compute` is the sum of per-batch worker-reported matching
times and is not wall time. The current production profile does not expose a
separate Cache-only optimizer timer or a separate `wait(futures)` wall timer,
so those two fields are recorded as `NOT_SEPARATELY_MEASURED` rather than
estimated from overlapping values.

| origin | baseline execute_slot s | optimized execute_slot s | ORIGIN_SPEEDUP | baseline configuration s | optimized configuration s | baseline surrogate s | optimized surrogate s | baseline worker cumulative compute s | optimized worker cumulative compute s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 233.432619 | 244.082887 | 0.956366x | 233.315123 | 243.988994 | 232.357373 | 243.164298 | 1,404.038069 | 1,323.164168 |
| 120 | 279.108204 | 780.295552 | 0.357695x | 278.941317 | 780.153909 | 277.988275 | 779.423548 | 1,975.916260 | 2,381.178674 |
| 239 | 103.144543 | 287.404221 | 0.358883x | 103.079235 | 287.359004 | 102.206342 | 286.703733 | 644.612649 | 886.079850 |

The aggregate execute-slot wall time was 615.685366 s baseline versus
1,311.782660 s optimized, or 0.469350x speedup. The one positive local result
was origin 0's 180-batch reduction, but its wall time still regressed because
of chunk scheduling. Origin 120/239 are clear engineering regressions.

`CACHE_EARLY_STOP_SPEEDUP` is `NOT_SEPARATELY_MEASURED`: the existing profile
has no Cache-only wall boundary. The whole-origin speedups above are the
reproducible paired runtime result.

## Formal output equivalence

The comparison covered, for each selected origin:

- raw action and effective Cache/UAV requests;
- previous/final Cache state and previous/final UAV positions;
- final Matching map;
- CPU and bandwidth allocation;
- per-task delay, energy and utility;
- actual `Q(t)`, cache cost, movement cost, `G(t)`/reward;
- configuration and actual Matching summaries.

The new early-stop diagnostic counter was excluded because it is an
instrumentation field, not a formal environment result. For all formal fields:

```ini
DIFFERENCE_COUNT = 0
MAX_ABSOLUTE_DIFF = 0.0
MAX_RELATIVE_DIFF = 0.0
```

The formal candidate generator and control loop remained in the same order;
an early certificate only takes the existing non-improving branch and does not
mutate `current_cache` or `current_objective`. Thus the output-equivalence
check passed, but the cache-scope/repeated-evaluation cost makes this particular
implementation unsuitable for production.

KKT math and actual environment Matching/KKT output were not modified. An
exact KKT solver-call counter was not present in the paired profile, so a
numeric KKT solver-call reduction is not claimed; the reliable observed proxy
is prediction Matching-call count.

## Final status

```ini
EXACT_EARLY_STOP_MATHEMATICALLY_VALID = YES
EXACT_EARLY_STOP_IMPLEMENTED = NO
K100_REFERENCE_PRESERVED = YES
UAV_PATH_UNCHANGED = YES
ALGORITHM_OUTPUT_EQUIVALENT = YES

ORIGIN0_EARLY_STOP_CANDIDATES = 15
ORIGIN120_EARLY_STOP_CANDIDATES = 233
ORIGIN239_EARLY_STOP_CANDIDATES = 105

PREDICTION_BATCH_REDUCTION = origin0=1.9565% reduction; origin120=36.4124% increase; origin239=63.9516% increase; aggregate=30.9063% increase
MATCHING_CALL_REDUCTION = same as prediction Matching calls above
KKT_SOLVE_REDUCTION = NOT_SEPARATELY_INSTRUMENTED

CACHE_EARLY_STOP_SPEEDUP = NOT_SEPARATELY_MEASURED
ORIGIN_SPEEDUP = origin0=0.956366x; origin120=0.357695x; origin239=0.358883x; aggregate=0.469350x

PRODUCTION_IMPLEMENTATION_WORTHWHILE = NO
READY_FOR_NEXT_RUNTIME_OPTIMIZATION = YES
READY_FOR_FORMAL_PPO_EPISODE = NO
```

The formal PPO episode remains not ready because even the unchanged path is
still minute-scale per origin; this task did not attempt to solve that broader
runtime bottleneck.

## Cleanup and checks

Temporary benchmark scripts, paired JSON data and Python caches created for
this task were removed. Only this formal report was added for this task; the
pre-existing source/report changes were preserved. `git diff --check` passed;
the only messages were the existing LF-to-CRLF warnings for the pre-existing
`task_allocation.py` and `trainer.py` changes.
