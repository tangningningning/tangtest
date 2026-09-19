# R5 exact Cache counterfactual runtime optimization

日期：2026-09-19  
仓库：`C:\Users\tangning\Desktop\code\adaptive_cache_mec`  
固定 seed：`4401`  
固定 realization：`runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911/episode_20001`

## 结论

在保持 R5 node-specific Cache credit 数学定义、正式 `execute_slot`、Matching/KKT、reward、next-state 和所有合法 candidate 不变的前提下，加入了一个 exact-safe 的完整 physical branch identity reuse：

```text
identity = (transition origin,
            complete final Cache state,
            complete final UAV state)
```

identity 只在同一个 transition 内复用；tasks、vehicle positions、prediction inputs 由固定 transition origin 隐含固定。没有按 operation label、candidate rank 或 top-k 去重。

full collection 的 38912 个 logical branches 中，2400 个 branch 与同一 transition 内的另一个 logical branch 具有完全相同的完整 physical state，因此 physical `execute_slot` 次数降为 36512，logical candidate 数不变。

受控同一 profile harness 下，serial old helper 为 722.7242s，optimized helper 为 610.3690s，降低 15.55%，speedup 1.1841x。任务给出的历史 gate wall-clock baseline 为 591.0466s；该值与本次带阶段计时 wrapper 的受控 profile 不具备严格 wall-clock 可比性，因此同时保留两者，不把机器负载/计时 instrumentation 差异误判为算法回归。

## Runtime profile

### Full 300-slot controlled profile

| stage | old seconds | old share | new seconds | new share |
|---|---:|---:|---:|---:|
| helper total | 722.7242 | 100.00% | 610.3690 | 100.00% |
| node state reconstruction / candidate feature preparation | 15.8865 | 2.20% | 13.5594 | 2.22% |
| branch construction + formal execution wrapper | 671.3492 | 92.89% | 561.2442 | 91.95% |
| configuration override build | 25.5026 | 3.53% | 20.4728 | 3.35% |
| `execute_slot` | 570.8809 | 78.99% | 482.3143 | 79.02% |
| Matching/KKT inside `execute_slot` | 552.7196 | 76.48% | 467.6793 | 76.62% |
| reward/next-state work after Matching/KKT | 18.1613 | 2.51% | 14.6350 | 2.40% |
| batched critic branch wrapper | 1.3159 | 0.18% | 1.2458 | 0.20% |
| critic forward itself | 1.0754 | 0.15% | 1.0123 | 0.17% |
| `deepcopy` | 0 calls / 0s | — | 0 calls / 0s | — |

The remaining time is ordinary Python orchestration and branch preparation. The critic was already batched; no scalar-per-candidate critic forward was introduced.

### Bounded profile

For transition index 20, the old path executed 145 branches in 2.9368s. There were 137 unique complete physical branch states and 8 exact reusable duplicates. The matching/KKT stage accounted for 2.4547s of the old bounded run.

## Exact correctness comparison

The fixed witness set was slots `20, 80, 140, 200` across the complete Cache node set, including `RSU_0` and `RSU_1`.

```text
WITNESS_LOGICAL_BRANCH_COUNT = 559
WITNESS_OLD_PHYSICAL_ENV_EXECUTIONS = 559
WITNESS_NEW_PHYSICAL_ENV_EXECUTIONS = 527
WITNESS_DEDUP_REUSED_BRANCH_COUNT = 32

MAX_Q_KEEP_ABS_DIFF = 0.0
MAX_Q_UPDATE_ABS_DIFF = 0.0
MAX_CANDIDATE_Q_ABS_DIFF = 0.0
MAX_REWARD_ABS_DIFF = 0.0
MAX_NEXT_OBS_ABS_DIFF = 0.0

CANDIDATE_MASK_EQUAL = YES
MATCHING_MISMATCH_COUNT = 0
NEXT_CACHE_MISMATCH_COUNT = 0
DUPLICATE_RESULT_MISMATCH_COUNT = 0
```

The old path enumerated every legal candidate with `reuse_identical_branches = false`; the optimized path used complete-state reuse. All compared results were produced by the same formal environment path. The duplicate-result check found no reward, next-observation, Matching, or next-Cache discrepancy.

R5 credit numerical checks remained finite:

```text
MAX_TRIGGER_POLICY_WEIGHTED_ADV_RESIDUAL = 3.54230533794464e-14
MAX_CANDIDATE_POLICY_WEIGHTED_ADV_RESIDUAL = 1.0325074129013956e-13
FULL_MAX_CANDIDATE_WEIGHTED_RESIDUAL = 1.9984014443252818e-13
```

These remain at floating-point roundoff scale; reuse only maps an already computed exact branch result and does not alter Q aggregation or policy probabilities.

## Runtime counts

```text
OLD_CACHE_CF_RUNTIME_SECONDS = 591.0466
CONTROLLED_SERIAL_OLD_RUNTIME_SECONDS = 722.7242206000374
NEW_CACHE_CF_RUNTIME_SECONDS = 610.3689913999988
SPEEDUP = 1.18407755109304x
CONTROLLED_SPEEDUP = 1.18407755109304x
CONTROLLED_RUNTIME_REDUCTION = 15.5460722081178%
HISTORICAL_BASELINE_RATIO = 0.968343097909219x

OLD_PHYSICAL_ENV_EXECUTIONS = 38912
NEW_PHYSICAL_ENV_EXECUTIONS = 36512
LOGICAL_CANDIDATE_COUNT = 38912
LOGICAL_CANDIDATE_COUNT_UNCHANGED = YES
DEDUP_REUSED_BRANCH_COUNT = 2400
PHYSICAL_ENV_EXECUTION_REDUCTION = 6.16776315789473%

EXACT_SEMANTICS_PRESERVED = YES
```

The historical `591.0466s` is the earlier uninstrumented gate number supplied as the task baseline. The controlled old/new numbers were collected with the same stage-timing harness and are the valid speedup comparison for this change.

## Scope and safety

- No candidate pruning, top-k selection, approximate Q, sampled expectation, or reduced legal-action set.
- No process-level parallel execution was attempted; the simple exact reuse was sufficient.
- No snapshot/deepcopy optimization was needed: the profile observed zero `deepcopy` calls in the Cache-CF path.
- No reward, system model, cache cost, movement cost, capacity, Matching, KKT, offloading, or R5 credit formula was changed.
- No PPO update, `trainer.update()`, or `optimizer.step()` was called during this runtime task.

The earlier R5 Dev3 run was explicitly interrupted for this task after epoch 1. Its existing artifacts remain untouched; the partial episode 2 is excluded from formal Dev3 results and episode 3 was not run.

```text
DEV3_STATUS = INTERRUPTED_FOR_RUNTIME_OPTIMIZATION
COMPLETED_DEV3_EPOCHS_RETAINED = 1
PARTIAL_EPISODE_2_INCLUDED = NO

SYSTEM_MODEL_CHANGED = NO
ALGORITHM_CHANGED = NO       # mathematical R5 credit semantics unchanged
REWARD_CHANGED = NO
MATCHING_KKT_CHANGED = NO

TRAINING_EXECUTED = NO       # runtime-optimization task
OPTIMIZER_STEP_EXECUTED = NO

TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO
```
