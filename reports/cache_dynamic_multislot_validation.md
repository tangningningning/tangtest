# Baseline v2.0 多 slot 缓存动态验证

## 1. 验证范围与运行条件

本次按当前正式 rollout 入口执行连续 50 个 execution slot，使用固定 seed=4401、单 episode、origin slot 0–49。只读取当前配置和已有 artifact，在内存中记录现有执行结果；未运行 PPO training，未修改代码、参数、cache 算法或任何 artifact。

当前运行配置：25 vehicles（5 SV + 20 TV）、2 RSU、2 UAV、10 services、`psi_c=11`、alpha(delay)=0.6、beta(energy)=0.4。动态 cache 节点为 5 SV + 2 RSU + 2 UAV = 9 个，TV local cache 为固定 artifact，不产生动态 `z_m^ca`。

## 2. Rollout 总体结果

| 项目 | 结果 |
|---|---:|
| 实际运行 slot 数量 | 50 |
| slot 范围 | 0–49 |
| 固定 seed | 4401 |
| observation dimension | 154 |
| action dimension | 11 |
| step feasible | 50/50 |
| reward finite | 50/50 |
| state propagation | 通过 |
| reward invariant | 通过 |

| 汇总指标 | 结果 |
|---|---:|
| `total_trigger_count` | **209** |
| `total_cache_change_count` | **0** |
| `max_delta_hit` | **0.000000** |
| 窗口加权 hit ratio（before / after） | **72.5806% / 72.5806%**（495 / 682） |
| `slots_with_cache_change` | 无 |
| 有 trigger 的 slot | 50/50 |
| 有 trigger 但无实际更新的 slot | 50/50 |
| candidate 总数 | 2864 |
| candidate 平均数/slot | 57.28 |
| replacement gain 最大/平均 | 0.000000 / 0.000000 |
| top-popularity overlap 平均 | 39.8182% |
| top-popularity overlap 范围 | 24.2424%–57.5758% |

### 按节点角色的 trigger 汇总

TV 没有动态 cache trigger；SV、RSU、UAV 的比例按“节点数 × 50 个 slot”计算。

| 节点类型 | trigger 次数 | 可触发节点-slot 数 | trigger 比例 |
|---|---:|---:|---:|
| TV | 0 | 0 | 不适用（固定 local cache） |
| SV | 112 | 250 | 44.8% |
| RSU | 42 | 100 | 42.0% |
| UAV | 55 | 100 | 55.0% |
| **合计** | **209** | **450** | **46.44%** |

## 3. B 类原因审计

本次属于 **B：长期存在 trigger，但没有实际 cache update**，而不是 C（无 trigger）。

50 个 slot 的 optimizer stop reason 全部为：

```text
converged_by_no_improvement
```

每个 slot 都有 cache trigger，且每个 slot 的：

- `cache_change_count=0`；
- `accepted_cache_updates=0`；
- `replacement_gain=0`；
- cache update cost 为 0；
- 触发节点的当前 cache 被保留。

candidate 数量定义为当前 cache 优化器实际枚举并评估的可行 add/delete/swap 邻居数量。逐 slot 数量及 replacement gain 见下表。replacement gain 定义为该 slot cache optimizer 返回的 `final_objective - initial_objective` 最大值；本次所有 slot 均未接受正收益替换。

top-popularity overlap 定义为：对每个动态缓存节点取其当前缓存服务数 K，在该 slot 的预测 popularity 中取 top-K，计算缓存服务与 top-K 的交集占该节点缓存服务数的比例，再对所有动态缓存节点按缓存服务数加权汇总。该指标使用当前 popularity prediction artifact，不读取未来真实请求。

## 4. 每 slot 结果

| slot | time(s) | arrival | trigger | cache_change | hit_before | hit_after | delta_hit | update_cost | reward | stop_reason | candidates | replacement_gain | top-pop overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0 | 100 | 15 | 3 | 0 | 0.533333 | 0.533333 | 0.000000 | -108.168566 | converged_by_no_improvement | 52 | 0.000000 | 0.515152 |
| 1 | 101 | 13 | 6 | 0 | 0.923077 | 0.923077 | 0.000000 | -77.366195 | converged_by_no_improvement | 84 | 0.000000 | 0.454545 |
| 2 | 102 | 12 | 4 | 0 | 0.750000 | 0.750000 | 0.000000 | -90.334245 | converged_by_no_improvement | 44 | 0.000000 | 0.454545 |
| 3 | 103 | 14 | 4 | 0 | 0.785714 | 0.785714 | 0.000000 | -109.460557 | converged_by_no_improvement | 24 | 0.000000 | 0.515152 |
| 4 | 104 | 12 | 5 | 0 | 0.666667 | 0.666667 | 0.000000 | -105.883789 | converged_by_no_improvement | 58 | 0.000000 | 0.575758 |
| 5 | 105 | 15 | 6 | 0 | 0.866667 | 0.866667 | 0.000000 | -104.551561 | converged_by_no_improvement | 84 | 0.000000 | 0.454545 |
| 6 | 106 | 15 | 2 | 0 | 0.666667 | 0.666667 | 0.000000 | -135.697746 | converged_by_no_improvement | 26 | 0.000000 | 0.454545 |
| 7 | 107 | 14 | 5 | 0 | 0.785714 | 0.785714 | 0.000000 | -170.900492 | converged_by_no_improvement | 58 | 0.000000 | 0.454545 |
| 8 | 108 | 14 | 4 | 0 | 0.642857 | 0.642857 | 0.000000 | -110.408166 | converged_by_no_improvement | 38 | 0.000000 | 0.454545 |
| 9 | 109 | 11 | 5 | 0 | 0.909091 | 0.909091 | 0.000000 | -53.720705 | converged_by_no_improvement | 58 | 0.000000 | 0.363636 |
| 10 | 110 | 15 | 2 | 0 | 0.600000 | 0.600000 | 0.000000 | -114.100070 | converged_by_no_improvement | 32 | 0.000000 | 0.515152 |
| 11 | 111 | 13 | 6 | 0 | 0.923077 | 0.923077 | 0.000000 | -79.167843 | converged_by_no_improvement | 90 | 0.000000 | 0.454545 |
| 12 | 112 | 14 | 2 | 0 | 0.714286 | 0.714286 | 0.000000 | -91.757767 | converged_by_no_improvement | 26 | 0.000000 | 0.515152 |
| 13 | 113 | 13 | 5 | 0 | 0.692308 | 0.692308 | 0.000000 | -123.423257 | converged_by_no_improvement | 64 | 0.000000 | 0.242424 |
| 14 | 114 | 13 | 6 | 0 | 0.769231 | 0.769231 | 0.000000 | -92.770004 | converged_by_no_improvement | 64 | 0.000000 | 0.454545 |
| 15 | 115 | 15 | 3 | 0 | 0.600000 | 0.600000 | 0.000000 | -171.749780 | converged_by_no_improvement | 52 | 0.000000 | 0.454545 |
| 16 | 116 | 14 | 5 | 0 | 0.857143 | 0.857143 | 0.000000 | -115.408979 | converged_by_no_improvement | 70 | 0.000000 | 0.454545 |
| 17 | 117 | 16 | 2 | 0 | 0.875000 | 0.875000 | 0.000000 | -161.547404 | converged_by_no_improvement | 52 | 0.000000 | 0.454545 |
| 18 | 118 | 14 | 3 | 0 | 0.714286 | 0.714286 | 0.000000 | -146.743326 | converged_by_no_improvement | 38 | 0.000000 | 0.515152 |
| 19 | 119 | 17 | 5 | 0 | 0.705882 | 0.705882 | 0.000000 | -197.522668 | converged_by_no_improvement | 84 | 0.000000 | 0.575758 |
| 20 | 120 | 13 | 4 | 0 | 0.769231 | 0.769231 | 0.000000 | -84.551570 | converged_by_no_improvement | 52 | 0.000000 | 0.454545 |
| 21 | 121 | 14 | 6 | 0 | 0.714286 | 0.714286 | 0.000000 | -102.127514 | converged_by_no_improvement | 70 | 0.000000 | 0.575758 |
| 22 | 122 | 16 | 6 | 0 | 0.687500 | 0.687500 | 0.000000 | -188.198897 | converged_by_no_improvement | 84 | 0.000000 | 0.515152 |
| 23 | 123 | 13 | 8 | 0 | 0.769231 | 0.769231 | 0.000000 | -119.750381 | converged_by_no_improvement | 116 | 0.000000 | 0.515152 |
| 24 | 124 | 14 | 5 | 0 | 0.714286 | 0.714286 | 0.000000 | -111.038934 | converged_by_no_improvement | 58 | 0.000000 | 0.575758 |
| 25 | 125 | 13 | 2 | 0 | 0.692308 | 0.692308 | 0.000000 | -100.146938 | converged_by_no_improvement | 40 | 0.000000 | 0.575758 |
| 26 | 126 | 13 | 5 | 0 | 0.615385 | 0.615385 | 0.000000 | -96.350533 | converged_by_no_improvement | 64 | 0.000000 | 0.515152 |
| 27 | 127 | 15 | 5 | 0 | 0.600000 | 0.600000 | 0.000000 | -108.582667 | converged_by_no_improvement | 64 | 0.000000 | 0.363636 |
| 28 | 128 | 13 | 5 | 0 | 0.846154 | 0.846154 | 0.000000 | -88.768076 | converged_by_no_improvement | 78 | 0.000000 | 0.303030 |
| 29 | 129 | 12 | 2 | 0 | 0.916667 | 0.916667 | 0.000000 | -60.836498 | converged_by_no_improvement | 26 | 0.000000 | 0.303030 |
| 30 | 130 | 14 | 3 | 0 | 0.571429 | 0.571429 | 0.000000 | -132.811805 | converged_by_no_improvement | 58 | 0.000000 | 0.363636 |
| 31 | 131 | 12 | 2 | 0 | 0.666667 | 0.666667 | 0.000000 | -104.282870 | converged_by_no_improvement | 32 | 0.000000 | 0.242424 |
| 32 | 132 | 17 | 4 | 0 | 0.882353 | 0.882353 | 0.000000 | -146.995316 | converged_by_no_improvement | 78 | 0.000000 | 0.242424 |
| 33 | 133 | 16 | 3 | 0 | 0.812500 | 0.812500 | 0.000000 | -161.670791 | converged_by_no_improvement | 38 | 0.000000 | 0.303030 |
| 34 | 134 | 15 | 5 | 0 | 0.666667 | 0.666667 | 0.000000 | -127.021881 | converged_by_no_improvement | 64 | 0.000000 | 0.363636 |
| 35 | 135 | 13 | 3 | 0 | 0.692308 | 0.692308 | 0.000000 | -91.090690 | converged_by_no_improvement | 72 | 0.000000 | 0.303030 |
| 36 | 136 | 14 | 4 | 0 | 0.642857 | 0.642857 | 0.000000 | -172.754413 | converged_by_no_improvement | 52 | 0.000000 | 0.303030 |
| 37 | 137 | 17 | 1 | 0 | 0.705882 | 0.705882 | 0.000000 | -159.605067 | converged_by_no_improvement | 6 | 0.000000 | 0.303030 |
| 38 | 138 | 11 | 7 | 0 | 0.545455 | 0.545455 | 0.000000 | -46.778702 | converged_by_no_improvement | 110 | 0.000000 | 0.303030 |
| 39 | 139 | 10 | 2 | 0 | 0.700000 | 0.700000 | 0.000000 | -54.934851 | converged_by_no_improvement | 12 | 0.000000 | 0.303030 |
| 40 | 140 | 15 | 2 | 0 | 0.733333 | 0.733333 | 0.000000 | -87.910968 | converged_by_no_improvement | 26 | 0.000000 | 0.242424 |
| 41 | 141 | 11 | 5 | 0 | 0.727273 | 0.727273 | 0.000000 | -93.801175 | converged_by_no_improvement | 64 | 0.000000 | 0.242424 |
| 42 | 142 | 14 | 6 | 0 | 0.785714 | 0.785714 | 0.000000 | -99.678851 | converged_by_no_improvement | 84 | 0.000000 | 0.303030 |
| 43 | 143 | 12 | 4 | 0 | 0.666667 | 0.666667 | 0.000000 | -106.398564 | converged_by_no_improvement | 58 | 0.000000 | 0.242424 |
| 44 | 144 | 15 | 3 | 0 | 0.600000 | 0.600000 | 0.000000 | -214.264929 | converged_by_no_improvement | 38 | 0.000000 | 0.242424 |
| 45 | 145 | 10 | 5 | 0 | 0.700000 | 0.700000 | 0.000000 | -42.477500 | converged_by_no_improvement | 64 | 0.000000 | 0.303030 |
| 46 | 146 | 13 | 4 | 0 | 0.769231 | 0.769231 | 0.000000 | -113.897407 | converged_by_no_improvement | 38 | 0.000000 | 0.242424 |
| 47 | 147 | 12 | 5 | 0 | 0.833333 | 0.833333 | 0.000000 | -102.958262 | converged_by_no_improvement | 78 | 0.000000 | 0.363636 |
| 48 | 148 | 15 | 4 | 0 | 0.733333 | 0.733333 | 0.000000 | -94.625408 | converged_by_no_improvement | 58 | 0.000000 | 0.303030 |
| 49 | 149 | 11 | 6 | 0 | 0.545455 | 0.545455 | 0.000000 | -114.763675 | converged_by_no_improvement | 84 | 0.000000 | 0.363636 |

## 5. Cache change / hit / cost 结论

整个 50-slot 窗口内：

- 实际 cache update：0 次；
- 改变的缓存服务数量：0；
- `slots_with_cache_change=[]`；
- hit ratio 的最大变化为 0，更新前后每个 slot 完全相同；
- `cache_update_cost=0`，因为 `C_m(t+1)=C_m(t)`，没有 0→1 loading，也没有 actual control trigger；
- `xi_ca`、`eta_ca` 成本公式已被调用，但本窗口没有实际变化可计费。

当前 cache 并非没有参与优化：209 次上层 trigger 进入了 trigger-conditioned cache optimizer，并枚举了 2864 个候选状态；只是所有候选相对当前 objective 与更新成本判断后都没有形成正的 replacement gain，因此缓存保持不变。

## 6. 是否建议进入 PPO 训练

当前系统连续 50 slot 的运行链路全部可行，说明接口、matching、资源分配、reward 和 cache 约束没有运行时错误。

但缓存行为在 50 个 slot 中稳定表现为 B 类：trigger 持续存在而实际 cache update 为 0。因此：

- 不建议仅凭当前结果直接进入正式或长期 PPO 训练；
- 建议先针对 objective/cost 与 candidate gain 做进一步静态定位，或执行更长但仍受控的多 slot 诊断；
- 本次不修改任何代码、参数或 artifact。
