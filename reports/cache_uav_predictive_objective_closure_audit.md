# Cache/UAV Predictive Configuration Objective Closure Audit

## 0. Scope and execution boundary

本轮只检查 Cache/UAV configuration objective、prediction horizon aggregation、beta、Cache/UAV cost、request-on 动态路径以及 prediction batch workers=1/8 等价性。未修改 PPO、reward、task generation、Matching/KKT、Cloud fallback gate、train/test split、Direct-CF 或正式实验参数数值；未运行 Batch4 或完整 PPO training。

本地工作区没有 `12.pdf`。本轮没有搜索、复制、创建或修改 `12.pdf`，也没有因为该文件缺失而阻塞；审计中的系统合同判断只使用用户确认的系统模型定义。该文件不是本轮实现 blocker。

真实 bounded fixture 使用 prediction origin `20`：20 个 scenario、5 个 horizon，共 100 个 prediction batches；对应实际 slot 有 13 个 arrived tasks。审计期间尝试过一次同时触发 Cache node `23` 和 `UAV_0` 的完整 joint coordinate search；parallel prediction-batch worker 正常运行、没有 native crash，但约 7 分钟仍未完成，因此中止。后续没有再次运行完整 joint search；workers=1/8 等价性采用同一真实 origin 的完整 100-batch fixed-state evaluator 验证。

## 1. Beta status

### 1.1 当前用途和代码位置

当前 `cache.discount_beta = 0.95` 保留原数值，但状态已改为 `ALGORITHM_DESIGN`，因为它属于预测/配置 surrogate 的规划设计，不属于用户确认的 system-model 物理合同。

代码中 beta 的有效路径只有以下几类：

- `cache_config_from_project()` 将 `cache.discount_beta` 读入 `CacheOptimizationConfig.beta_discount`。
- `build_surrogate_evaluation_context()` 为每个 scenario × horizon batch 生成 `discount = beta ** horizon`。
- `evaluate_surrogate_context()` 将每个 batch 的实际 Matching/KKT `task_utility_sum` 乘以该 discount 后累加。
- `service_values_from_scenarios()` 在 Cache 的预测初始化价值排序中使用同一有限 horizon 折扣；这仍是 Cache configuration 初始化，不是 actual slot Q/G。

Cache 与 UAV 都受 beta 影响：UAV evaluator 固定一个 UAV candidate 后，仍调用 Cache module 的 `evaluate_fixed_configuration()`，只改变带 UAV 位置的 service-node/channel context。因此 UAV 的 candidate utility 也共享同一个 beta horizon aggregation。

### 1.2 精确 horizon objective

令 `S` 为 scenario 数，`H` 为 prediction horizon 长度，`Q_raw(s,h; C,Q)` 为给定 Cache/UAV candidate 在 scenario `s`、horizon `h` 上通过正式 prediction Matching/KKT 得到的 batch task-utility sum，则当前代码实现为：

\[
\widehat Q(C,Q)
 = \frac{1}{S}\sum_{s=0}^{S-1}\sum_{h=0}^{H-1}
       \beta^h Q_{raw}(s,h;C,Q).
\]

实现顺序是：先在每个 batch 内完成 Matching/KKT 得到 `Q_raw`，再乘 `beta ** h`，最后对完整 horizon 求和并只除以 `num_scenarios`；没有额外除以 `H`。因此 scenario expectation 在外层平均，horizon 是有限折扣和。

当前 `H=5` 时 beta=0.95 的 horizon weights 为：`1.0, 0.95, 0.9025, 0.857375, 0.81450625`。

- `beta=0.95`：当前有限 horizon 内越远的 prediction step 权重越低；它表达配置规划对预测远期的算法设计偏好。
- `beta=1.0`：有限 horizon 内所有 horizon 等权，得到 `1/S * sum_s sum_h Q_raw(s,h)`；这不是 actual 单 slot Q，也不是无限 horizon 的目标。
- actual slot `Q(t)`、`G(t)`、PPO reward、Matching/KKT：均不包含 beta。actual `G(t)` 仍为 `Q_actual - C_ca - C_q`。

后续如需正式选择 beta，应只使用 train origins 的 sensitivity protocol，例如固定 prediction scenarios、seed、PPO action/checkpoint 和其他参数，在 train-only origins 上比较 `beta ∈ {0.90, 0.95, 1.00}` 的 predictive objective、触发/实际更新计数以及 actual Q/G；不得用 test origins 选择 beta，并将 test 只留作最终一次 held-out evaluation。当前没有把 0.95 宣称为 `SYSTEM_MODEL_CONFIRMED`。

## 2. Cache cost semantic status

当前代码与系统模型合同一致。`system_model.cache_update_cost()` 等价于：

\[
E_{ca,m}=\xi_{ca,m}\sum_f s_f[c_{m,f}^{new}-c_{m,f}^{old}]^+,
\]

\[
C_{ca,m}=\eta_{ca,m}z_{m}^{actual}+\lambda_{cache}E_{ca,m}.
\]

核对结果：

- `z_actual` 由 old/new cache 的实际差异得到，固定成本只在发生实际 cache change 时产生。
- `max(new-old, 0)` 只计算新增服务；删除不产生 loading energy。
- initial cache 只作为 episode/reference state，不在初始化时扣 online update cost。
- `xi_ca` 在代码中先形成 `loading_energy`/`E_ca`，`lambda_cache` 再将 energy 映射为 system-cost。配置注释已同步为“等效 energy/MB”；数值与 `PROVISIONAL_IMPLEMENTATION` 状态保持不变，因为尚无可靠物理校准依据。
- `lambda_cache` 是唯一 Cache energy 到 system-cost 的权重入口；本轮没有重新定标。

公式 sanity check 通过：新增 20 MB、`xi=0.004` 时 loading energy 为 0.08；删除分量 loading energy 为 0；实际 change 的 fixed cost 单独计入。

## 3. UAV movement cost semantic status

当前代码与系统模型合同一致：

\[
E_{mov}=\xi_{mov}\lVert q_{new}-q_{old}\rVert,
\qquad
C_q=\eta_qz_q^{actual}+\lambda_{movement}E_{mov}.
\]

核对结果：

- requested trigger 与 actual movement 分离；`z_actual` 由最终 old-to-final position distance 是否超过 tolerance 得到。
- movement cost 只使用最终位置差，不累计 candidate search 访问路径。
- no movement 时 effective distance、energy、fixed cost 和 movement cost 均为 0。
- effective distance 必须满足 `distance <= z_actual * vmax * dt + tolerance`，因此 max-speed constraint 在系统模型函数中执行。
- `xi_mov`、`eta_q`、`lambda_movement` 的当前配置状态仍为 `PROVISIONAL_IMPLEMENTATION`；本轮只审计，不重新定标。

真实 UAV candidate probe（origin 20、`UAV_0`）：

| 项目 | 值 |
|---|---:|
| old position | `(3364.03, 1119.88)` |
| candidate position | `(3367.78, 1119.88)` |
| actual candidate distance | `3.75 m` |
| movement energy | `56.25 J` (`15 J/m × 3.75 m`) |
| movement cost | `0.00275` |

该 candidate 的 channel gain 为 `1.91832618486e-11`，base rate 为 `37119155.3468 bit/s`；它通过正式 downstream evaluator 后使 predicted utility 从 `26.2350959733` 变为 `26.2407799436`，movement cost 增加 `0.00275`，candidate objective 为 `26.2380299436`。

## 4. Cache request-on dynamic path

真实 bounded probe 使用 Cache node `23` 的 `request=1`，同时打开 `UAV_0` request。旧 Cache node `23` 行为为：

```text
old      = [0,1,0,0,0,0,0,1,0,0]
candidate= [0,1,0,0,0,0,0,0,0,0]
```

这是一个删除型 Cache candidate。单 candidate probe 的结果为：

| 项目 | 值 |
|---|---:|
| requested cache flag | `1` |
| candidate evaluated | 1 个 bounded probe candidate（不是完整 local-search best 集合） |
| predicted Q old | `26.2350959733` |
| predicted Q candidate | `25.9494742109` |
| old Cache cost | `0` |
| candidate Cache cost | `0.2` |
| candidate J | `25.7494742109` |
| candidate delta J | `-0.4856217624` |
| bounded probe final cache | old cache（该 probe 未接受 candidate） |
| bounded probe actual flag | `0` |

该 candidate 使 actual feasible-target set 发生变化，例如 task `64` 的 edge targets 从包含 `23` 变为不包含 `23`；因此 request-on candidate 确实进入了正式 downstream path。由于完整 Cache coordinate search 在本轮边界内未完成，不能把这个单 candidate probe 伪称为全局 `best_delta` 证明。正式合同仍是：`request=0 => actual=0`；`request=1` 允许 actual=0；只有完整 search 确认的 accepted candidate 才能判断 `accepted_delta > objective_tolerance`。独立 no-trigger actual execution sanity check 已通过：所有 actual update 和两类 cost 均为 0。

## 5. UAV request-on dynamic path

同一真实 origin 上，UAV candidate 的链路路径被完整调用为：

```text
old q
 -> feasible candidate q
 -> UAV channel gain
 -> base rate
 -> formal Matching
 -> KKT CPU/BW allocation
 -> predicted Q
 -> movement cost
 -> candidate J
 -> strict-improvement decision
 -> final q / actual flag
```

isolated UAV candidate probe 结果：

| 项目 | 值 |
|---|---:|
| requested | `UAV_0=1` |
| candidate actual movement | `3.75 m` |
| predicted utility old / candidate | `26.2350959733 / 26.2407799436` |
| J old / candidate | `26.2350959733 / 26.2380299436` |
| delta J | `+0.0029339703` |
| predicted utility delta | `+0.0056839703` |
| movement cost delta | `+0.00275` |
| candidate feasibility | pass（boundary、max speed、safety-distance） |

在该 isolated probe 中 candidate 满足严格 objective improvement；在与上面的删除型 Cache candidate 同时组合时，Cache 的负 delta 使 joint candidate 未被接受，因此本次 bounded joint probe 的 combined final UAV 仍为 old position。这个结果符合 requested 与 actual 分离的合同，不要求每个 request=1 必须移动。

## 6. Downstream coupling evidence

本节比较 old state、Cache candidate（UAV old）和 UAV candidate（Cache old），均重新经过正式 feasible-target、Matching、KKT、per-task delay/energy/Q 与 aggregate utility evaluator；没有使用 surrogate-only state comparison。

### Cache candidate

- predicted `Q_hat`: `26.2350959733 -> 25.9494742109`。
- actual Matching Q: `7.69292271407 -> 7.61914567617`。
- task `64` feasible targets：`[23, RSU_1, UAV_0, UAV_1] -> [RSU_1, UAV_0, UAV_1]`。
- task `64` final target：`23 -> UAV_0`。
- task `64` delay/energy/Q：
  - delay `4.69994913568 -> 4.27111206198 s`
  - vehicle energy `0.309878102017 -> 0.209527709116 J`
  - task Q `0.867462736116 -> 0.888968633626`
- Cache candidate还改变了其他任务的 Matching 与 UAV/RSU KKT grouping；例如 candidate UAV_1 CPU allocation 同时承载 tasks `9,18,73`，而 old state 中 task `18` 在 UAV_0。

### UAV candidate

- channel gain/base rate 随 `q_UAV0` 改变；feasible-target 集合在该小位移下保持相同，但正式 Matching/KKT 被重新执行。
- Matching assignment 在该特定 candidate 上保持相同，这是结果而不是跳过 evaluator。
- actual Matching Q：`7.69292271407 -> 7.69526569227`。
- task `18` delay/energy/Q：
  - delay `15.8355944255 -> 15.8159896955 s`
  - vehicle energy `0.905959241180 -> 0.902047583286 J`
  - task Q `0.571478011799 -> 0.572372898437`
- UAV_0 的 BW allocation 也发生变化，例如 task `18` 的 BW fraction `0.433366615064 -> 0.433619774577`。
- 因此该 UAV candidate 已证明 channel、KKT allocation、per-task metrics 和 aggregate Q 的 downstream coupling；本 fixture 没有证明 assignment 必须改变，实际 assignment 保持不变。

## 7. Full-origin workers=1/8 equivalence

真实 origin 20 的完整 20×5=100 batches 已分别以 serial worker 和 8 个 prediction-batch workers 评价同一个最终 bounded state（该 state 因 combined probe 未改善而为 old Cache/UAV state）。结果：

| 检查项 | workers=1 | workers=8 | 结果 |
|---|---:|---:|---|
| prediction batches | 100 | 100 | 一致 |
| predictive utility | `26.2350959733` | `26.2350959733` | 一致 |
| Cache cost | `0` | `0` | 一致 |
| UAV movement cost | `0` | `0` | 一致 |
| objective J | `26.2350959733` | `26.2350959733` | 一致 |
| predictive Matching calls | 100 | 100 | 一致 |
| Matching candidate evaluations | 35329 | 35329 | 一致 |
| per-batch max absolute utility diff | — | — | `0` |
| actual downstream Matching | identical | identical | 一致 |
| actual Q | `7.69292271407` | `7.69292271407` | 一致 |
| actual reward G | `7.69292271407` | `7.69292271407` | 一致 |

因此 **full-origin prediction-batch evaluator serial/parallel equivalence PASS**，数值差不超过 `1e-12`。一次完整 joint coordinate-search 的 workers=1/8 optimizer-level equivalence 没有声称通过：parallel joint search 已在约 7 分钟 bounded window 内中止，未发生 native crash；本轮没有再运行 serial joint search。

## 8. Minimal validation

- `python -m compileall -q src`：通过。
- 临时审计脚本 syntax compile：通过。
- `system_config.toml` parse/type/range validation：通过；允许 provisional 的 formal validation：通过。无 `allow_provisional` 的正式 gate 仍会按项目现有其他 `PROVISIONAL_IMPLEMENTATION` 参数阻止正式实验，这与本轮 beta status 无关。
- no-trigger request=0 -> actual=0、Cache/UAV cost=0 sanity：通过。
- Cache/UAV cost closed-form sanity：通过。
- 真实 origin full 100-batch workers=1/8 fixed-state equivalence：通过。
- 未运行 Batch4、完整 PPO training 或完整 240-slot training。

## 9. Status summary and remaining blockers

| 项目 | 判断 |
|---|---|
| beta status | `ALGORITHM_DESIGN`；数值 `0.95` 未改变；只影响 prediction/configuration surrogate |
| Cache cost semantic status | 公式、actual trigger、addition-only energy、lambda mapping 一致；xi 数值仍 `PROVISIONAL_IMPLEMENTATION` |
| UAV cost semantic status | 公式、actual movement、zero-movement、max-speed constraint 一致；相关数值仍 `PROVISIONAL_IMPLEMENTATION` |
| Cache request-on path | bounded real candidate path 已进入 downstream；完整 best-candidate/global delta 未在本轮完成 |
| UAV request-on path | real candidate feasibility/channel/rate/Matching/KKT/Q/cost/J path 已通过；combined optimizer final movement 未在 bounded window 内完成 |
| downstream coupling | Cache candidate 改变 targets/Matching/KKT/per-task metrics/Q；UAV candidate 改变 channel/KKT/per-task metrics/Q，assignment 在该 probe 保持不变 |
| full-origin workers 1/8 | fixed-state complete 100-batch evaluator 等价通过；joint optimizer-level serial-vs-parallel 未完成 |
| 12.pdf | 未搜索、未创建、未修改；不是 blocker |

唯一实质剩余 blocker 是完整 Cache/UAV coordinate-search 的运行时间：当前 full-origin evaluator 本身可稳定并行且等价，但完整 trigger optimization 仍需更长运行窗口才能给出 global request-on diagnostics 和 optimizer-level serial/parallel comparison。本轮没有据此修改算法或实施优化。
