# Cache exact early-stop certificate / coarse staging audit

## 结论摘要

本轮证明了 rejection certificate 在严格限定作用域和完整 key 下可以安全复用，且在
origin 120/239 确实消除了部分重复 prediction evaluation。但固定 staged schedule
仍增加了 submit/wait 边界；三 origin 的正式输出完全等价，aggregate wall-clock 在
这一次 paired replay 中下降 2.24%，但 origin 239 单独回退 4.18%。这个收益不足以
满足“各代表性 origin 不出现明显回退”的生产保留门槛，因此本轮实验性生产改动已撤回。

当前最终工作区没有保留本轮 certificate/staging 代码；已有的 K=100、Matching/KKT、
PPO、reward 和此前已存在的用户修改均未被本轮回滚。

## 验证范围与协议

- branch: `main`
- paired baseline HEAD: `b7e90d4015bbeac682ce7b55a4a29d71a7e174d7`
- origins: `0, 120, 239`
- seed: `4401`
- 每个 origin 使用同一个 `context.initial_cache`、`context.initial_uav`、scenario
  slice、actual task/vehicle state 和此前正常 Bernoulli sampling 得到的固定 action。
- 不执行 PPO update，不改变 candidate 集合、candidate 顺序、KKT、Matching、reward
  或 K=100 reference。
- baseline 是本轮实现前的完整 100-batch path；optimized 是临时的 certificate
  cache + `[75,100]` staged path。

固定 action 为：

```text
origin 0:   [0,0,1,0,0,0,0,0,0,0,1]
origin 120: [0,1,1,1,1,0,1,1,0,0,0]
origin 239: [0,1,1,0,1,0,0,1,0,0,0]
```

## A. Certificate 安全性证明

### incumbent 单调性

当前 `optimize_cache()`（`src/decision/cache_configuration.py:1549`）先用 exact
objective 初始化 `current_objective`。如果外部 initial state 不优于 previous state，
代码把搜索起点恢复为 previous objective；这不会降低 incumbent。随后
`initialize_cache_for_triggers()`（`:1437`）和 coordinate search 只在：

```text
candidate_objective > current_objective + tolerance
```

时接受 candidate 并更新 `current_objective`。因此，在一个 optimizer invocation
内部，进入 candidate loop 后 incumbent 单调不减。certificate 只证明拒绝，不改变
`current_cache` 或 `current_objective`。

### candidate、cost、context 作用域

1. `generate_node_neighbors()`（`:1365`）按既有 add/delete/swap 顺序产生候选；初始化、
   coordinate round 以及同一 slot 的后续 Cache block 可能再次访问相同 cache vector。
2. `cache_update_cost()`（`:1102`）的输入是 previous cache、candidate cache、trigger、
   service sizes、`eta_ca`、`xi_ca` 和 `lambda_cache`。这些输入在一个固定的
   configuration evaluation 中不变，但 UAV outer round 会改变固定 UAV positions，
   所以位置不能从 certificate key 中省略。
3. `build_surrogate_evaluation_context()` 生成当前 slot 固定的 100 个 prediction
   batches；`SurrogateEvaluationContext.configuration_utility_cache`（`:486`）随
   context 生命周期存在，`execute_slot()` 在 slot 结束关闭 pool（`:282`）。
4. 因此安全 certificate 的最小有效作用域是当前 surrogate context，key 还必须包含
   当前固定节点位置及完整 cost inputs；不能跨 slot/context 复用。

临时实现使用了以下 exact key 内容：candidate/previous cache、trigger、service
size、capacity、`eta_ca`、`xi_ca`、固定节点位置、Cache config、objective-definition
version 和 context identity。数组采用 dtype、shape、原始 bytes；没有 rounding、
tolerance cache 或 task-id-only key。

certificate 的语义是：

```text
status = CERTIFIED_NON_IMPROVING
certified_against = incumbent_at_creation + tolerance
candidate_upper_bound = strict UB of candidate objective
```

当且仅当同一 exact key 再次访问且当前 `incumbent + tolerance >=
certified_against` 时，才直接走原有 non-improving branch。certificate 不写入
exact utility cache，也不携带伪造的 partial objective。潜在 winner 始终执行到
batch 100 并取得 exact objective。

结论：`INCUMBENT_MONOTONIC = YES`，`CERTIFICATE_REUSE_SAFE = YES`。

## B. 重复 candidate 的实测价值

optimized 统计如下。`repeated_visits` 是当前 invocation 内的重复；
`repeated_after_certificate` 还包含同一 surrogate context 内后续 Cache invocation
命中的 certificate，因此可以大于前者。

| origin | total Cache candidate calls | unique configurations | repeated visits | repeated after certificate | certificate hits | new certificates | batches skipped by certificates | batches skipped by early-stop |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 15 | 15 | 0 | 0 | 0 | 0 | 0 | 0 |
| 120 | 269 | 244 | 25 | 27 | 27 | 32 | 2,025 | 800 |
| 239 | 119 | 101 | 18 | 14 | 14 | 8 | 1,050 | 200 |
| total | 403 | 360 | 43 | 41 | 41 | 40 | 3,075 | 1,000 |

这确认了 origin 120/239 的重复计算问题真实存在，并且 context-scoped certificate
确实可以跳过后续 batches；但 certificate 的计算收益必须和额外 stage wait 一起评估。
origin 0 的 safe stop 最早为 batch 77，因此 `[75,100]` 不会在该 origin 产生 early
certificate。

## C. staged schedule 对照

只试验了少量固定 schedule：先测 `[75,90,100]`，最终选用更少同步点的 `[75,100]`。
每个 stage 内按 formal batch index 提交、按 index reduction；stage 结束才检查严格
UB；未通过 certificate 时最坏仍执行完整 100 batches。

`[75,90,100]` 的 aggregate wall-clock 为约 `0.9625x`，因额外同步被放弃。最终
`[75,100]` 的 Cache stage submissions 为 `31 / 321 / 115`（origin 0/120/239）；
baseline 完整 path 对应每个 Cache utility miss 一次完整提交组，即 `16 / 177 / 62`。

## D. Paired 输出等价性

比较了 action/effective requests、Cache/UAV final state、Cache decision diagnostics、
Matching、CPU/BW allocation、delay、energy、per-task utility、Q、cost、G/reward 和
next state。三个 origin 的 normalized formal result JSON 均完全相同：

```ini
DIFFERENCE_COUNT = 0
MAX_ABSOLUTE_OUTPUT_DIFF = 0.0
MAX_RELATIVE_OUTPUT_DIFF = 0.0
```

本轮曾发现把每个 stage 的 partial sum 先聚合再合并会造成约 `1e-14` 的浮点诊断差异；
该实验版本未保留，最终 `[75,100]` replay 改为逐 batch、formal 顺序累加后达到上面的
严格零差异结果。

## E. Runtime 与计算量

| origin | baseline execute_slot s | optimized execute_slot s | speedup | baseline config s | optimized config s | baseline surrogate s | optimized surrogate s | baseline Matching calls | optimized Matching calls | call reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 227.794338 | 206.295901 | 1.104212x | 227.696555 | 206.222680 | 226.720700 | 205.521152 | 9,200 | 9,200 | 0.0000% |
| 120 | 275.027833 | 279.209233 | 0.985024x | 274.875608 | 279.062652 | 273.905459 | 278.138776 | 17,700 | 16,900 | 4.5198% |
| 239 | 95.869579 | 100.047387 | 0.958242x | 95.817955 | 100.006443 | 94.796174 | 99.288781 | 6,200 | 6,000 | 3.2258% |
| aggregate | 598.691751 | 585.552522 | 1.022439x | 598.390118 | 585.291775 | 595.422334 | 582.948709 | 33,100 | 32,100 | 3.0211% |

Matching candidate evaluation totals were `12,549,983` baseline and `12,278,316`
optimized. The aggregate call reduction is real, but wall-clock gain is small and the
shorter origin 239 regressed by about 4.18%; this fails the conservative per-origin
production gate even though the aggregate ratio is above one.

## Final status

The temporary implementation was removed after the paired gate failed. The following
flags describe the final retained production tree, followed by the rejected experiment's
measured ratios where applicable:

```ini
INCUMBENT_MONOTONIC = YES
CERTIFICATE_REUSE_SAFE = YES
CERTIFICATE_CACHE_IMPLEMENTED = NO
COARSE_STAGING_IMPLEMENTED = NO
ALGORITHM_OUTPUT_EQUIVALENT = YES
MATCHING_CALL_REDUCTION = aggregate 3.0211% (origin 0 0.0000%, origin 120 4.5198%, origin 239 3.2258%)
ORIGIN0_SPEEDUP = 1.104212x
ORIGIN120_SPEEDUP = 0.985024x
ORIGIN239_SPEEDUP = 0.958242x
AGGREGATE_SPEEDUP = 1.022439x
PRODUCTION_IMPLEMENTATION_WORTHWHILE = NO
READY_FOR_FORMAL_PPO_EPISODE = NO
```

No commit, push, merge, rebase, PPO update, or full episode was performed.
