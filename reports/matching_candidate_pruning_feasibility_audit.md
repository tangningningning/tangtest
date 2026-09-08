# Matching candidate pruning / cheap screening feasibility audit

## 结论摘要

本轮只做了静态审计和真实状态 replay，没有修改 Matching、KKT、PPO、reward、Cache/UAV optimizer、candidate 集合、candidate 顺序或 worker 配置。所有 pruning 数字均是“如果在当前 evaluator 之前加入该规则”的 counterfactual 统计；正式代码仍然完整评估当前 candidate。

严格的理想资源上界可以安全用于 candidate screening：

- transfer 与 exchange 的候选数学依赖已恢复；
- 已有 exact feasibility gate 真实有效，尤其是 exchange target gate；
- 新的 tight bound 在 tie-aware 条件下仍能排除约 `42.8%` 的当前 incremental evaluator calls，约 `40.4%` 的 KKT requests；
- bound 计算成本约为当前 candidate-evaluation compute 的 `23.8%`（只在已有 feasibility gate 通过后计算）；样本中的 candidate-evaluation 估算 speedup 约 `1.14x`；
- 没有发现 false reject，也没有找到不依赖完整共享资源状态的通用 dominance 规则。

因此 exact pruning 潜力判为 `HIGH`，建议下一轮实施一个受控的 strict-bound prototype；本轮不实施。

## 工作区状态

审计开始时：

```ini
BRANCH = main
HEAD = b7e90d4015bbeac682ce7b55a4a29d71a7e174d7
```

`src/decision/task_allocation.py` 和 `src/ppo_simple/trainer.py` 的修改是本任务开始前已有的 working-tree 修改。本轮没有修改生产源文件，也没有 commit、push、merge 或 rebase。

## 1. 当前候选调用链和数学依赖

### 候选生成与搜索顺序

`src/decision/task_allocation.py:217` 的 `enumerate_best_operation()` 当前按以下顺序完整搜索：

1. 对每个 task，按稳定 task ID 顺序枚举其 `feasible_targets` 中除旧 target 外的 transfer；
2. 对每个 task pair，枚举 exchange；相同旧 target 直接跳过；
3. exchange 在进入 evaluator 前要求 `right_target in feasible_targets[left_task]` 且 `left_target in feasible_targets[right_task]`；不满足时计入 `feasibility_rejected_without_kkt`；
4. 通过 gate 的 candidate 进入 `evaluate_matching_candidate_incremental()`，再进入 `evaluate_matching_with_kkt()` / `allocate_resources_kkt()`；
5. candidate objective 是 `evaluation.task_utility_sum - current_evaluation.task_utility_sum`；候选选择使用 `_better_candidate()` 的 objective tolerance 和 operation-key tie-break（`:202`）。

### 一个 candidate 影响的 task/node

`evaluate_matching_candidate_incremental()`（`:1501`）并不是只重算被移动的 task：

- transfer 的直接 affected task 是 1 个；exchange 是 2 个；
- affected edge nodes 是这些 task 的 current/candidate 非 `LOCAL`、非 `CLOUD` target；
- 同一 affected edge node 上 current 或 candidate matching 中的所有 task 都加入 impacted set；
- 这些 task 的 CPU/BW allocation、delay、energy、utility 会被重新计算；其他 task 的结果复用。

因此：

- transfer 在没有共享 edge node 时只影响一个 task；如果离开或进入共享 edge node，实际影响可扩展到两个或更多 node 上的所有 task；
- exchange 至少影响两个 task，并可能同时改变两个 edge node 的共享 allocation；
- 不能仅按“被交换的 task 数量”构造完整 objective bound。

### objective 及其资源依赖

当前 `src/common/system_model.py:355` 的 task utility 为：

```text
U_i = omega_t * (1 - delay_i / T_ref)
    + omega_e * (1 - energy_i / E_ref)
```

其中 `omega_t, omega_e >= 0` 且和为 1，故 utility 对 delay 和 vehicle energy 单调不增。edge execution（`:90`）为：

```text
delay_i = input_bits / (base_rate * bandwidth_fraction)
          + cpu_cycles / (node_cpu * cpu_fraction)
energy_i = tx_power * input_bits / (base_rate * bandwidth_fraction)
```

当前 KKT allocation 的 CPU/BW fractions 均在 `(0, 1]`，并受 node-level resource sum 约束；`allocate_resources_kkt()`（`:1266`）按 edge node 的 task set 计算两套 KKT allocation。Local/Cloud 使用固定的 per-task resource model，不和其他 Cloud task 共享 KKT resource。

## 2. 已存在的 exact feasibility pruning

`build_feasible_targets()`（`:82`）及 `_edge_path_is_feasible()`（`:1138`）在 candidate generation 前检查 cache service、node role、position、CPU、bandwidth、channel gain 和 positive uplink rate。不可执行的 edge target 不进入 task 的 feasible target 集合。

当前 exchange gate 是另一条已经存在的 exact rule。它只检查交换后的两个 target 是否仍在各自 feasible target 集合中，不使用 heuristic distance、channel 或 utility。

真实 replay 中：

- resource/KKT epsilon 后置检查没有额外产生 resource-infeasible candidate（3 个 actual-origin replay 和 12 个 sampled prediction batches 均为 0）；
- exchange 的已有 gate 已经拒绝 actual replay 中 `2008/2218 = 90.53%` 的 exchange definitions；sampled prediction batches 中拒绝 `11686/12882 = 90.72%`；
- transfer 在进入 enumeration 时已经通过 feasible target gate，因此没有额外的 transfer feasibility reject。

这部分属于当前生产代码已有能力，不是本轮新增实现。

## 3. strict candidate objective upper bound

对一个 current matching 和一个 candidate matching，令 `S` 为 incremental evaluator 的 impacted task set。对 candidate target：

- Local/Cloud task 使用当前 formal model 的 exact utility；
- edge task 把 candidate edge node 的 CPU fraction 和 BW fraction 都替换为 `1.0`，得到 no-sharing ideal utility `U_i^ideal`；
- 定义：

```text
UB_delta(candidate) = sum_{i in S}(U_i^ideal(candidate) - U_i(current))
```

这是严格上界，理由是：

1. 对 Local/Cloud，candidate utility 本身就是 exact value；
2. 对 edge，实际 `cpu_fraction <= 1`、`bandwidth_fraction <= 1`，所以实际 delay/energy 不小于 fraction=1 时的 delay/energy；
3. utility 对 delay/energy 单调不增，因此 `U_i(actual) <= U_i^ideal`；
4. incremental evaluator 之外的 task utility 不变，故对 objective delta 求和仍保持上界关系。

为了保留现有 `_better_candidate()` 的 tie-break 语义，审计只在以下条件同时成立时标记 bound pruning：

```text
已有至少一个 exact feasible incumbent
并且 UB_delta(candidate) < incumbent_delta - tie_tolerance
```

没有 incumbent 的第一个 candidate 不做 bound pruning；等号附近也不做 pruning。因此该规则不会依赖 candidate 顺序改变，也不会吞掉可能通过 operation-key 赢得 tie 的 candidate。

### dominance

没有找到可证明的通用 exact dominance：不同 target 的 utility 同时依赖 task 的 input/cycles、vehicle-node channel、node role/capacity、以及 candidate 改变后 affected edge node 的 task set。一个 target 在当前 occupancy 下占优，不代表在另一个 candidate occupancy 下仍占优。只用当前 per-task utility、距离、channel gain 或 target 类型做 dominance 都可能错误排除共享 KKT allocation 更优的 candidate。

本轮没有实施 dominance pruning。

## 4. Replay 方法

### Actual-origin replay

固定当前真实 slot state，分别对 origin `0 / 120 / 239` 运行当前 Matching 搜索，并捕获每个生成 candidate 的：

```text
candidate_type
affected_tasks
affected_nodes
impacted_tasks_for_bound
exact_final_improvement
exact_final_rank
exact_final_feasible
feasibility_prunable / reason
bound_prunable / bound_incumbent_at_screen / ideal_resource_upper_delta
dominance_prunable
kkt_requests / kkt_solver_calls
full_evaluation_seconds / screening_seconds / tight_bound_seconds
```

共捕获 `2841` 个 candidate definitions，其中 `833` 个实际进入当前 evaluator，`2008` 个被现有 exchange feasibility gate 拒绝。

### Prediction-surrogate replay

为避免重跑完整正式 rollout，只选真实 aligned prediction batches：

```text
origin 0:   batch 0, 8, 50, 99
origin 120: batch 0, 31, 50, 99
origin 239: batch 0, 24, 50, 99
```

这是 12 个 representative batches 的真实 `cache_configuration._evaluate_prediction_batch()` replay，不是完整 100-batch 生产运行。它共捕获 `16544` 个 candidate definitions，代表性地覆盖了小、中、大 candidate group。

## 5. Candidate-level 结果

### Actual-origin replay

| origin | tasks | generated | current evaluator | existing feasibility rejects | new strict-bound rejects | exact-prunable union | KKT requests/misses | bound-avoided KKT requests/misses |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 15 | 1293 | 348 | 945 | 186 | 1131 | 643 / 138 | 331 / 77 |
| 120 | 16 | 1185 | 344 | 841 | 138 | 979 | 568 / 133 | 202 / 49 |
| 239 | 10 | 363 | 141 | 222 | 62 | 284 | 257 / 60 | 114 / 28 |
| **total** | — | **2841** | **833** | **2008** | **386** | **2394** | **1468 / 331** | **647 / 154** |

这里的 new strict-bound rate 是 `386/833 = 46.34%`，KKT request reduction counterfactual 是 `647/1468 = 44.07%`，solver-miss reduction 是 `154/331 = 46.53%`。

### Sampled prediction-batch replay

| origin | sampled batches | generated | transfer / exchange | current evaluator | existing exchange rejects | new strict-bound rejects | exact-prunable union | KKT requests/misses | bound-avoided requests/misses |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4 | 5316 | 1335 / 3981 | 1878 | 3438 | 685 | 4123 | 3491 / 832 | 1205 / 268 |
| 120 | 4 | 4831 | 923 / 3908 | 1175 | 3656 | 522 | 4178 | 2094 / 413 | 854 / 155 |
| 239 | 4 | 6397 | 1404 / 4993 | 1805 | 4592 | 872 | 5464 | 3272 / 624 | 1522 / 296 |
| **total** | **12** | **16544** | **3662 / 12882** | **4858** | **11686** | **2079** | **13765** | **8857 / 1869** | **3581 / 719** |

按 candidate type 汇总：

| type | generated | current evaluator | existing feasibility rejects | new strict-bound rejects | bound rate among evaluator calls | KKT requests/misses | bound-avoided requests/misses |
|---|---:|---:|---:|---:|---:|---:|---:|
| transfer | 3662 | 3662 | 0 | 1544 | 42.16% | 6465 / 1338 | 2511 / 499 |
| exchange | 12882 | 1196 | 11686 | 535 | 44.73% | 2392 / 531 | 1070 / 220 |
| **total** | **16544** | **4858** | **11686** | **2079** | **42.80%** | **8857 / 1869** | **3581 / 719** |

如果把当前已有 feasibility gate 也计入全部 definitions，sampled replay 的 exact-prunable union 是 `13765/16544 = 83.20%`；这不是新 bound 的单独收益。

一个 representative bound-pruned record（origin 0, sampled batch 0）为：

| type | affected task | affected nodes | exact delta（只用于 replay 对照） | incumbent at screen | ideal UB delta | KKT requests/misses |
|---|---|---|---:|---:|---:|---:|
| transfer | `16` | `23 -> RSU_1` | `6.1030262253` | `26.5759482316` | `11.8317782169` | `2 / 2` |
| transfer | `16` | `23 -> UAV_0` | `7.9053390141` | `26.5759482316` | `13.6340910057` | `2 / 1` |

这些 candidate 在实际 counterfactual replay 中确实没有胜过已有 incumbent；正式代码当前仍执行了它们的完整 evaluator。

## 6. Exactness 检查

- 所有 bound-pruned rows 都有已有 exact incumbent；没有使用“尚无 incumbent 时以 0 为 incumbent”的不安全简化。
- sampled prediction replay 中，bound-pruned candidate 的最大 `exact_delta - incumbent_delta` 仍约为 `-3.25e-2`；没有 candidate 落入 tie tolerance。
- actual-origin replay 的相同最大值约为 `-1.61e-1`（origin 0；其它 origin 也为负）。
- 对所有已完整 evaluator 的 candidate，`ideal_resource_upper_delta - exact_final_improvement` 的最小值仅为浮点 round-off：sampled replay 约 `-8.9e-15`，actual replay 约 `-4.0e-15`；没有实质性 upper-bound violation。
- 正式 candidate order、transfer/exchange 集合、KKT calls、KKT tolerance 和 evaluator 数学均未改变；`kkt_requests`/`kkt_solver_calls` 的“avoided”只是插入 strict bound 后的反事实估算。

## 7. Screening 成本与预期收益

screening 成本只统计已有 feasibility gate 通过后，计算 ideal-resource utility upper bound 的时间；没有把现有 gate 的成本重复归因给新规则。

| replay | full current evaluator seconds | strict-bound screening seconds | screening/full | counterfactual candidate-eval speedup |
|---|---:|---:|---:|---:|
| actual origins 0/120/239 | 0.177896 | 0.046697 | 26.25% | 1.16x |
| 12 sampled prediction batches | 1.248337 | 0.296770 | 23.77% | 1.14x |

预测 batch 按 origin 的 candidate-evaluation-only 估算 speedup 为：

```text
origin 0:   1.06x
origin 120: 1.15x
origin 239: 1.22x
aggregate:  1.14x
```

这不是完整 origin wall-clock benchmark：没有把规则写入生产 evaluator，也没有运行完整 100-batch configuration evaluation。因此不能把 `1.14x` 宣称为正式 `execute_slot` speedup；它只证明 candidate-evaluation 层有正的净收益潜力。若下一轮实现后仍保持该比例，才应再做完整 origin paired benchmark。

## 8. 最终判断

```ini
EXACT_FEASIBILITY_PRUNING_EXISTS = YES
EXACT_BOUND_PRUNING_EXISTS = YES
EXACT_DOMINANCE_PRUNING_EXISTS = NO
TRANSFER_EXACT_PRUNING_RATE = 42.16% new bound among sampled evaluator calls (existing extra feasibility gate: 0%)
EXCHANGE_EXACT_PRUNING_RATE = 44.73% new bound among post-gate sampled evaluator calls (existing gate: 90.72% of exchange definitions)
TOTAL_EXACT_PRUNING_RATE = 42.80% new bound among current evaluator calls; 83.20% including existing feasibility gate
SCREENING_COST_RATIO = 23.77% sampled prediction replay (26.25% actual-origin replay)
EXACT_PRUNING_POTENTIAL = HIGH
EXPECTED_KKT_REDUCTION = 40.43% requests / 38.47% solver misses in sampled prediction replay
EXPECTED_ORIGIN_SPEEDUP = approximately 1.14x candidate-evaluation compute; full-origin wall-clock NOT VERIFIED
APPROXIMATE_SCREENING_WORTH_STUDYING = NO
READY_TO_IMPLEMENT_EXACT_PRUNING = YES
READY_TO_IMPLEMENT_APPROXIMATE_SCREENING = NO
READY_FOR_FORMAL_PPO_EPISODE = NO
```

`READY_TO_IMPLEMENT_EXACT_PRUNING = YES` 表示可以进入下一轮受控实现和 paired equivalence validation，不表示本轮已经修改生产代码。下一轮必须使用严格 `< incumbent - tie_tolerance` 的判定，且继续保留 exact full evaluation 作为未被 certificate 排除 candidate 的路径；不能把该 bound 改成 approximate confidence interval，也不能改变 candidate order 或提前停止 Matching 搜索。

## 清理与校验

本轮临时 replay、candidate capture、汇总脚本和 Python cache 在任务结束时删除；正式报告是本轮唯一保留的新增审计 artifact。没有运行完整 PPO episode、PPO update 或无关测试。

最终 `git diff --check`：`PASS`。
