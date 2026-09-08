# Cache optimizer 真实环境激活能力审计

审计日期：2026-09-08  
审计目标：确认真实环境中单个 Cache trigger 是正常 no-op，还是 Cache optimizer 系统性失活。  
本审计不修改代码、系统模型、Cache objective、acceptance rule、reward、PPO 参数或 Matching/KKT；未进行正式 PPO 训练。

## 1. Git 与范围

| 项目 | 结果 |
|---|---|
| branch | `main`，tracking `origin/main` |
| HEAD | `b7e90d4 Fix default formal training entrypoint` |
| 审计开始时 tracked diff | 无 |
| 审计开始时 `git diff --check` | 通过，无输出 |
| 原有未跟踪文件 | `打包并上传.bat`，未修改、未删除 |
| commit/push | 均未执行 |

## 2. 最小真实 replay 方法

使用当前 `system_config.toml` 和真实 aligned step：

- `origin_index=0` / `slot=0` / `time_s=100.0`，15 个 actual tasks；
- `origin_index=239` / `slot=239` / `time_s=339.0`，10 个 actual tasks；
- 两个 slot 都从同一个 `context.initial_cache`、同一个 `context.initial_uav` 开始；
- 每个 slot 先执行一次 neutral，再分别执行 9 个单 Cache request；
- 所有 UAV trigger 固定为 0；共 `2 × (1+9)=20` 个真实 `execute_slot()` replay；
- Cache row 向量的 10 个位置按 `service_catalog.service_ids=[0,1,2,3,4,5,6,7,8,9]` 排列；
- Cache node 顺序为 `23, 87, 43, 75, 58, RSU_0, RSU_1, UAV_0, UAV_1`。

通过临时 capture wrapper 记录了 `objective()` 的 cache 参数、返回值、预测 utility 和 cache cost，并记录了 optimizer diagnostics。wrapper 只存在于内存中，执行后恢复；没有替代或跳过真实 Cache optimizer、actual Matching/KKT 或 reward 链。

固定初始 UAV 为：

```text
UAV_0 = [3364.03, 1119.88]
UAV_1 = [5179.65, 1710.41]
```

## 3. 源码链路核对

1. `src/ppo_simple/interface.py:296-351` 为 Cache trigger 构造 mask；Cache index 在 `332-333` 返回 `[True, True]`。
2. `src/decision/slot_environment.py:181-280` 校验 binary request；neutral 之外的 request 调用 `optimize_joint_configuration()`。
3. `src/decision/slot_environment.py:969-992` 调用 `cache_configuration.optimize_cache()`；Cache candidate 以 candidate cache matrix 重新计算 objective。
4. `src/decision/cache_configuration.py:1122-1151` 的 objective 为：

   ```text
   objective(cache) = predicted_service_utility(cache) - cache_update_cost(previous, cache)
   ```

5. `src/decision/cache_configuration.py:865-1020` 的 surrogate evaluator 使用 candidate cache；`src/decision/cache_configuration.py:650-692` 明确将 `candidate_cache` 传入 `task_allocation.optimize_task_matching(..., edge_cache_state=candidate_cache, ...)`，因此 candidate 会进入 executable nodes、offloading feasibility、Matching/KKT 和预测 task utility。
6. `src/decision/cache_configuration.py:1524-1534` 和 `1807-1820` 只有严格 objective improvement 才接受 candidate；`src/decision/cache_configuration.py:1846-1868` 根据最终 cache 与 previous cache 的差异生成 actual update。
7. actual slot 的 `Q(t)`、`G(t)` 和 reward 仍由 `src/decision/slot_environment.py:342-467` 的真实 Matching/KKT 结果计算；本审计没有把 predictive objective 当作 reward。

## 4. Neutral baseline

| slot | tasks | neutral `Q(t)` | neutral `G(t)` | neutral reward | Cache change |
|---:|---:|---:|---:|---:|---|
| 0 | 15 | 10.016961264 | 10.016961264 | 10.016961264 | 0/9 nodes |
| 239 | 10 | 2.815130577 | 2.815130577 | 2.815130577 | 0/9 nodes |

Neutral 只走 `no_configuration_trigger` fast path；所有 UAV 也保持不动。

## 5. 每个 `slot × Cache node` 的 replay 结果

说明：

- `mask` 是被触发 Cache bit 的 `[0,1]` mask；所有行均为 `[1,1]`，selected 1 未被屏蔽。
- `current→best` 是该 selected node 的 10-service Cache row；其他 node 受 trigger contract 约束保持不变。
- `raw Δ` 是本次捕获到的所有不同 candidate 中，最大 candidate objective 减 current objective；允许为负。
- `reported Δ` 是现有 `cache_candidate_diagnostics.best_delta_objective` 的值。该字段从 0 初始化，只有 candidate objective 严格高于 reference 时才更新，因此 rejected negative case 会报告 0，而不是负数。
- `accepted/actual` 是最终 configuration 的实际结果；`G/reward` 中 reward 与 G 完全相同。

### slot=0

current objective=`25.214194759492`；neutral `Q/G/reward=10.016961263991`。

| node | mask | candidate evals | current→best Cache row | best candidate objective | raw Δ | reported Δ | result | cache cost | Q(t) | ΔQ | G/reward | ΔG=Δreward |
|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 23 | `[1,1]` | 8 | `0100000100→0100000001` | 25.000376069 | -0.213818690 | 0 | rejected / 0 | 0 | 10.016961264 | 0 | 10.016961264 | 0 |
| 87 | `[1,1]` | 12 | `0000001000→0000000001` | 26.195708373 | 0.981513613 | 0.981513613 | accepted / 1 | 1.199862663 | 9.662288392 | -0.354672872 | 8.462425728 | -1.554535536 |
| 43 | `[1,1]` | 8 | `0100000001→0000000001` | 23.851043542 | -1.363151217 | 0 | rejected / 0 | 0 | 10.016961264 | 0 | 10.016961264 | 0 |
| 75 | `[1,1]` | 8 | `0000001000→0100000001` | 31.948629788 | 6.734435029 | 6.734435029 | accepted / 1 | 1.759445578 | 10.016961264 | 0 | 8.257515686 | -1.759445578 |
| 58 | `[1,1]` | 8 | `0000100010→0100000001` | 35.169188089 | 9.954993329 | 9.954993329 | accepted / 1 | 1.759445578 | 10.835710903 | 0.818749639 | 9.076265325 | -0.940695939 |
| RSU_0 | `[1,1]` | 30 | `0011110000→0100100011` | 41.043863081 | 15.829668321 | 15.829668321 | accepted / 1 | 2.000360951 | 11.910797369 | 1.893836105 | 9.910436418 | -0.106524846 |
| RSU_1 | `[1,1]` | 14 | `0010110101→0100110101` | 25.203752258 | -0.010442501 | 0 | rejected / 0 | 0 | 10.016961264 | 0 | 10.016961264 | 0 |
| UAV_0 | `[1,1]` | 15 | `0000011100→0101000001` | 34.190178563 | 8.975983804 | 8.975983804 | accepted / 1 | 4.087652955 | 11.068697667 | 1.051736403 | 6.981044712 | -3.035916552 |
| UAV_1 | `[1,1]` | 11 | `0010110100→0110100001` | 33.088704921 | 7.874510162 | 7.874510162 | accepted / 1 | 2.021334694 | 9.843019355 | -0.173941909 | 7.821684661 | -2.195276603 |

### slot=239

current objective=`34.912730662504`；neutral `Q/G/reward=2.815130576902`。

| node | mask | candidate evals | current→best Cache row | best candidate objective | raw Δ | reported Δ | result | cache cost | Q(t) | ΔQ | G/reward | ΔG=Δreward |
|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 23 | `[1,1]` | 8 | `0100000100→0100000001` | 35.795486421 | 0.882755759 | 0.882755759 | accepted / 1 | 1.199862663 | 4.104784349 | 1.289653772 | 2.904921685 | 0.089791109 |
| 87 | `[1,1]` | 12 | `0000001000→0000000001` | 37.122019799 | 2.209289136 | 2.209289136 | accepted / 1 | 1.199862663 | 5.116116644 | 2.300986067 | 3.916253981 | 1.101123404 |
| 43 | `[1,1]` | 8 | `0100000001→0000000001` | 33.079960450 | -1.832770212 | 0 | rejected / 0 | 0 | 2.815130577 | 0 | 2.815130577 | 0 |
| 75 | `[1,1]` | 8 | `0000001000→0100000001` | 36.394816042 | 1.482085379 | 1.482085379 | accepted / 1 | 1.759445578 | 4.384599372 | 1.569468795 | 2.625153794 | -0.189976783 |
| 58 | `[1,1]` | 9 | `0000100010→0000100001` | 40.780155118 | 5.867424455 | 5.867424455 | accepted / 1 | 1.199862663 | 5.147598356 | 2.332467779 | 3.947735692 | 1.132605116 |
| RSU_0 | `[1,1]` | 12 | `0011110000→0110100011` | 40.993827849 | 6.081097187 | 6.081097187 | accepted / 1 | 2.000360951 | 6.802221670 | 3.987091093 | 4.801860719 | 1.986730143 |
| RSU_1 | `[1,1]` | 14 | `0010110101→0010100101` | 34.803514670 | -0.109215992 | 0 | rejected / 0 | 0 | 2.815130577 | 0 | 2.815130577 | 0 |
| UAV_0 | `[1,1]` | 12 | `0000011100→0100010101` | 38.475633753 | 3.562903091 | 3.562903091 | accepted / 1 | 2.021334694 | 6.922857557 | 4.107726980 | 4.901522863 | 2.086392286 |
| UAV_1 | `[1,1]` | 11 | `0010110100→0110100001` | 38.359900537 | 3.447169875 | 3.447169875 | accepted / 1 | 2.021334694 | 6.707988756 | 3.892858180 | 4.686654062 | 1.871523486 |

## 6. 汇总统计

这里的 `best_delta_objective` 主统计采用 raw best evaluated candidate minus current objective，因此能同时表达 positive、zero 和 negative candidate。现有 diagnostics 的 clamped/reporting 口径另列于最后一列。

### Overall

| 指标 | raw candidate 口径 | 现有 diagnostics 口径 |
|---|---:|---:|
| `total_cache_requests` | 18 | 18 |
| `requests_with_positive_best_delta` | 13 | 13 |
| `actual_cache_updates` | 13 | 13 |
| `actual_update_rate` | 72.22% | 72.22% |
| `zero_best_delta_count` | 0 | 5 |
| `negative_best_delta_count` | 5 | 0 |

现有 diagnostics 的 5 个 0 不是 5 个 exact tie；它们对应 raw best delta 为负的 rejected cases，diagnostic 以 reference objective 作为初始“best feasible”值，未允许负值覆盖它。

### 按 Cache node

| node | requests | positive raw Δ | actual updates | update rate | raw zero | raw negative | diagnostics reported zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| 23 | 2 | 1 | 1 | 50.00% | 0 | 1 | 1 |
| 87 | 2 | 2 | 2 | 100.00% | 0 | 0 | 0 |
| 43 | 2 | 0 | 0 | 0.00% | 0 | 2 | 2 |
| 75 | 2 | 2 | 2 | 100.00% | 0 | 0 | 0 |
| 58 | 2 | 2 | 2 | 100.00% | 0 | 0 | 0 |
| RSU_0 | 2 | 2 | 2 | 100.00% | 0 | 0 | 0 |
| RSU_1 | 2 | 0 | 0 | 0.00% | 0 | 2 | 2 |
| UAV_0 | 2 | 2 | 2 | 100.00% | 0 | 0 | 0 |
| UAV_1 | 2 | 2 | 2 | 100.00% | 0 | 0 | 0 |
| **合计** | **18** | **13** | **13** | **72.22%** | **0** | **5** | **5** |

## 7. `best_delta_objective=0` 与 objective 组成拆解

选取 `slot=0, node=23`。该 case 的现有 diagnostics 报告 `best_delta_objective=0`，但 capture 到的 best candidate Cache row 确实不同：

```text
current Cache row:       0100000100
best candidate Cache row:0100000001
```

逐项重算：

| 项目 | current/reference | best candidate | candidate - current |
|---|---:|---:|---:|
| predicted service utility | 25.214194759492 | 26.200238732469 | +0.986043972977 |
| cache update cost | 0.000000000000 | 1.199862663113 | +1.199862663113 |
| objective = utility - cost | 25.214194759492 | 25.000376069355 | -0.213818690137 |

因此：

1. candidate 确实改变了 Cache row；
2. objective evaluation 收到的是 candidate cache，而不是旧的 previous/current cache；
3. Cache state 进入了预测 Matching/KKT：代码把 `candidate_cache` 作为 `edge_cache_state` 传入 executable candidate evaluation；本次 utility 从 25.214194759492 变为 26.200238732469，说明 objective 对 Cache state 有响应；
4. 该 candidate 的 objective 并没有与 current 相同，而是 utility 增益不足以覆盖 Cache update cost；
5. 现有 diagnostics 的 0 是“没有正的 candidate 覆盖 reference”的 reporting 结果，不是 exact objective tie，也不是 Cache objective 对 state 不敏感；
6. 因为 candidate 被拒绝，actual Cache state、actual `Q(t)`、`G(t)` 和 reward 都保持 neutral 值。

在本次 18 个 request 中没有观察到“candidate Cache row 与 current 不同且 raw objective 精确相等”的 exact-tie 案例；该 narrower condition 为 **NOT VERIFIED**。但存在上述“reported `best_delta_objective=0` 且 candidate row 不同”的案例，其 objective 组成已经证明 0 的来源是 diagnostics 的 reference fallback，而非 tie。

## 8. 关键问题结论

| 问题 | 结论 | 证据 |
|---|---|---|
| Cache candidate 是否真的改变 cache vector | **CONFIRMED** | 18/18 个 request 的 captured best candidate row 与 current row 不同 |
| candidate evaluation 是否使用 candidate cache | **CONFIRMED** | objective capture 的 cache 参数不同；`_evaluate_prediction_batch()` 将其作为 `edge_cache_state` 传给 Matching |
| Cache state 是否进入 executable/offloading/evaluation | **CONFIRMED** | predicted utility 在不同 Cache row 下变化；候选 Matching/KKT 链接收 candidate cache |
| `best_delta=0` 是数学 exact tie 还是 objective 不敏感 | **FAILED：不是这两者** | selected zero-report cases 的 raw best delta 均为负；0 来自 diagnostics 的 reference 初始化/严格改善更新语义 |
| 是否存在 `best_delta > tolerance` | **CONFIRMED** | 13/18 个 raw best candidate delta 为正，且均产生 actual update |
| 是否存在真实 Cache update | **CONFIRMED** | 13 个 request 的 final Cache 与 previous Cache 不同，actual update rate 72.22% |
| Cache optimizer 是否系统性失活 | **FAILED** | 多个 node、两个时间位置均出现正 candidate、Cache mutation 和真实 cost/Q/G/reward 结果 |
| 其他未选 slot 的泛化行为 | **NOT VERIFIED** | 本审计只运行了 early slot=0 与 train-end slot=239 的最小代表集 |

## 9. 最终结论

```text
CACHE_CANDIDATES_ACTUALLY_DIFFER = YES
CACHE_OBJECTIVE_RESPONDS_TO_CACHE_STATE = YES
POSITIVE_CACHE_CANDIDATE_EXISTS = YES
ACTUAL_CACHE_UPDATE_EXISTS = YES
CACHE_OPTIMIZER_SYSTEMICALLY_INACTIVE = NO
```

本次 slot=0/node=23 的 `actual_cache_update=0` 属于该 state/node 的正常 rejected no-op，不代表 Cache optimizer 在真实运行状态下整体失活。需要特别区分：现有 diagnostics 的 `best_delta_objective=0` 表示没有正改善被记录，并不表示不同 Cache configuration 得到相同 objective。

## 10. 清理与最终检查

- 临时 audit 脚本、Python cache 和测试文件在任务完成前删除；只保留本正式报告以及任务开始前已有的文件。
- 未修改源代码、配置、算法或训练参数。
- 未 commit、未 push、未 merge、未 rebase。
- 最终执行 `git diff --check`：通过，无输出。
