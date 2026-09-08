# Matching candidate strict ideal-resource upper-bound pruning implementation

## 结论

本轮已在生产 Matching 路径实现上一轮审计证明的 strict ideal-resource upper-bound pruning。实现只发生在：candidate 已生成、已有 exchange feasibility gate 已通过、进入 incremental evaluator/KKT 之前。

没有修改 PPO、observation/action/mask、reward、Q/G、prediction K=100、Cache/UAV optimizer、candidate generator/order、operation-key tie-break、KKT 数学/tolerance 或 worker 数。

实现通过了 representative prediction batch 的逐 candidate exact validation 和 origin `0/120/239` 的 paired environment replay，因此保留生产修改。

## 工作区与实现范围

```ini
BRANCH = main
HEAD = b7e90d4015bbeac682ce7b55a4a29d71a7e174d7
```

本轮开始时 `src/decision/task_allocation.py`、`src/ppo_simple/trainer.py` 已有 working-tree 修改，未覆盖。新增生产改动位于：

- `src/decision/task_allocation.py`
  - 新增共享的 exact affected-node/impacted-task scope；screening 与正式 incremental evaluator 使用同一 scope；
  - 新增 Local/Cloud exact utility 与 edge CPU/BW fraction=1.0 的 ideal utility 计算；
  - 在 incremental path 中按 strict `< incumbent - tie_tolerance` pruning；
  - full-reference path 不启用 pruning；
  - 增加结构化 candidate/KKT counters。
- `src/decision/slot_environment.py`
  - 将 counters 放入 matching summary。
- `src/decision/cache_configuration.py`
  - 将 worker-side Matching counters 聚合到 configuration profile；不保存逐 candidate 日志。

full evaluator 仍保留为 reference；`run_slot20_validation()` 的 full reference 不经过新 screening。

## Strict bound 实现

对正式 incremental evaluator 的 impacted task set `S`，实现计算：

```text
UB_delta(candidate)
  = sum_{i in S}(U_i_ideal(candidate) - U_i_current)
```

- Local/Cloud 使用当前 formal per-task model 的 exact utility；
- edge 使用 `cpu_fraction=1.0`、`bandwidth_fraction=1.0` 的 no-sharing utility；
- 没有调用 `evaluate_matching_candidate_incremental()`、`evaluate_matching_with_kkt()`、`allocate_resources_kkt()` 或 `solve_kkt_fraction_allocation()`；
- impacted task set 包含 affected task，以及当前或 candidate matching 位于 affected edge node 的全部 task；
- `math.fsum()` 后向正无穷方向提升一个 representable float，避免浮点 round-off 使 upper bound 略低；
- 只有已有 exact feasible incumbent 时才 screening；第一个 candidate 必定走原 evaluator；
- 只有：

```text
UB_delta < incumbent_delta - (objective_tolerance * 0.1)
```

才 prune。使用严格 `<`，没有使用 `<=`，没有绕过 operation-key tie-break 的语义。

## 逐 candidate exact validation

使用同一 current state 和 candidate order，先运行 bound-disabled baseline，记录全部 candidate 的 exact delta；再运行生产 optimized path，记录所有 bound screen 和 full evaluator call。覆盖：

- actual origin `0/120/239`；
- prediction batches：`0:[0,8,50,99]`、`120:[0,31,50,99]`、`239:[0,24,50,99]`。

共覆盖 15 个 cases、2465 个被 production path prune 的 candidate。

```ini
FALSE_REJECT_COUNT = 0
```

每一个被 prune candidate 都满足 baseline exact delta 严格小于当时 exact incumbent 减去 tie tolerance。没有 candidate 在 tie/tolerance 边界被错误排除。

## Paired environment equivalence

固定：

```text
origins = 0, 120, 239
seed label = 4401
same initial cache/UAV state per origin
same fixed action: cache trigger index 0 = 1; all other Cache/UAV bits = 0
normal environment transition
no PPO update
```

baseline 临时关闭 strict bound，保留其它当前工作区代码；optimized 使用最终生产实现。两者均运行完整 K=100 configuration path。比较字段包括 raw/effective action、Cache/UAV requests、configuration result、accepted Matching sequence、final matching、CPU/BW allocation、per-task delay/energy/utility、Q、Cache/UAV state、cost、G/reward 对应值和 next state。

```ini
DIFFERENCE_COUNT = 0
MAX_ABSOLUTE_OUTPUT_DIFF = 0.0
MAX_RELATIVE_OUTPUT_DIFF = 0.0
ALGORITHM_OUTPUT_EQUIVALENT = YES
```

其中 reward 与 environment 的 `system_utility=G(t)` 恒等式一致，因此 `G/reward` 也保持完全一致。

## 实际 paired performance

时间是本轮 baseline/optimized 同一批运行的 wall-clock；不是历史值。`worker cumulative Matching` 是各 prediction worker Matching compute 的累计时间，可能大于 execute-slot wall-clock，因为 worker 并行。

| origin | execute baseline (s) | execute optimized (s) | speedup | configuration baseline/optimized (s) | surrogate baseline/optimized (s) | worker Matching baseline/optimized (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 22.697843 | 22.091606 | 1.027x | 22.612859 / 22.025707 | 22.534718 / 21.933204 | 147.617958 / 134.860483 |
| 120 | 16.495331 | 13.651996 | 1.208x | 16.351179 / 13.531740 | 16.272725 / 13.450422 | 111.670806 / 87.893458 |
| 239 | 17.252243 | 14.020126 | 1.231x | 17.188961 / 13.964900 | 17.108114 / 13.872977 | 115.643389 / 92.014622 |
| **aggregate** | **56.445416** | **49.763728** | **1.134x** | — | — | — |

### Candidate/KKT counts

| origin | full evaluator baseline/optimized | upper-bound pruned | KKT requests baseline/optimized | KKT misses / actual solves baseline/optimized |
|---:|---:|---:|---:|---:|
| 0 | 362937 / 201177 | 161760 | 659589 / 378744 | 160075 / 122621 |
| 120 | 313629 / 154110 | 159519 | 557415 / 285577 | 127501 / 89272 |
| 239 | 334461 / 161496 | 172965 | 602177 / 299898 | 135428 / 95805 |
| **aggregate** | **1011027 / 516783** | **494244** | **1819181 / 964219** | **423004 / 307698** |

相对 baseline：

```text
candidate evaluator reduction = 48.885%
KKT request reduction        = 46.997%
KKT solve/miss reduction     = 27.259%
```

optimized production counters 中，三 origin 合计：

```text
transfer upper-bound pruned = 354888
exchange upper-bound pruned = 139356
upper-bound screen calls    = 1011027
```

上一轮 representative replay 中按 eligible evaluator calls 计算的类型 pruning rate 为：

```ini
TRANSFER_PRUNING_RATE = 42.16%
EXCHANGE_PRUNING_RATE = 44.73%
```

本轮完整 environment paired run 的总 candidate evaluator pruning rate 为 `494244 / 1011027 = 48.885%`。已有 exchange feasibility rejects 未被新 bound 重复计算；它们在 baseline/optimized 中保持一致。

## 240-origin episode 可行性

本轮 optimized 三个独立 origin 的平均 execute-slot 时间约为 `16.59 s`。按同一 cache-only fixed-action 条件线性估算，240 origins 约为 `3981 s`，即约 `66 min`；真实 PPO action 可能包含更多 configuration-triggered work，因此这不是乐观保证。

仍未运行完整 PPO episode 或 PPO update。当前 origin runtime 仍不处于可管理的正式 episode 范围，下一步仍需要更高层 acceleration，而不是继续在本轮加入其它小型 exact optimization。

## 最终结果

```ini
EXACT_BOUND_PRUNING_IMPLEMENTED = YES
FALSE_REJECT_COUNT = 0
ALGORITHM_OUTPUT_EQUIVALENT = YES

TRANSFER_PRUNING_RATE = 42.16% (representative eligible evaluator calls)
EXCHANGE_PRUNING_RATE = 44.73% (representative eligible evaluator calls)
TOTAL_PRUNING_RATE = 48.885% (paired environment full evaluator calls)

KKT_REQUEST_REDUCTION = 46.997%
KKT_SOLVE_REDUCTION = 27.259%

ORIGIN0_SPEEDUP = 1.027x
ORIGIN120_SPEEDUP = 1.208x
ORIGIN239_SPEEDUP = 1.231x
AGGREGATE_SPEEDUP = 1.134x

PRODUCTION_IMPLEMENTATION_WORTHWHILE = YES
READY_FOR_FORMAL_PPO_EPISODE = NO
```

## 清理与校验

本轮临时 validation、paired benchmark、JSON 和 Python cache 在报告完成后删除。只保留通过验证的生产修改和本报告；不 commit、push、merge 或 rebase。

最终 `git diff --check`：`PASS`。
