# Formal parameter status closure

## Scope and baseline

- 项目：`adaptive_cache_mec`
- 基线 commit：`550d12f11c283bf071727b8e988ba2130e9f8714`
- 本轮只读取当前 `system_config.toml`、`parameter_status`、formal validation gate、已有参数语义报告和对应代码公式。
- 未运行 Batch4、完整 PPO training 或新的 Cache/UAV、Matching/KKT、Cloud、CF、并行和性能审计。
- `src/decision/task_allocation.py` 中刚完成的 Matching-local exact KKT memoization 未修改。
- 本地没有 `12.pdf`；本轮没有搜索、创建、复制或修改该文件，也没有修改论文正文。
- 用户文件 `打包并上传.bat` 未修改、未移动、未加入本轮报告。

## Formal gate result

配置解析和普通配置校验通过。`validate_for_formal_experiment(..., allow_provisional=False)` 仍然拒绝正式实验，原因是 formal key 集合中有 10 个参数仍为 `PROVISIONAL_IMPLEMENTATION`：

```text
nodes.sv.eta_ca
nodes.rsu.eta_ca
nodes.uav.eta_ca
nodes.sv.xi_ca
nodes.rsu.xi_ca
nodes.uav.xi_ca
nodes.uav.movement.movement_energy_coefficient_j_per_m
nodes.uav.movement.movement_fixed_cost
objective.lambda_cache
objective.lambda_movement
```

当前没有 `TODO_LITERATURE_CALIBRATION` formal blocker。`cache.discount_beta` 的状态是 `ALGORITHM_DESIGN`，并且不在 `FORMAL_SYSTEM_PARAMETER_KEYS` 中；它不会触发当前 formal system-parameter gate。`nodes.*.range_m` 虽然仍标记为 `PROVISIONAL_IMPLEMENTATION`，但当前 formal gate 不把它们纳入系统参数闭环，且代码注释明确它们不是 formal predictive cache / Matching/KKT 的 hard coverage cutoff，因此不是本轮 formal blocker。

## 当前代码合同和公式

### Cache cost

`src/common/system_model.py::cache_update_cost()` 当前实现为：

\[
E_{ca,m}=\xi_{ca,m}\sum_f s_f[c_{m,f}^{new}-c_{m,f}^{old}]^+,
\]

\[
C_{ca,m}=\eta_{ca,m}z_{m}^{actual}+\lambda_{cache}E_{ca,m}.
\]

其中 `z_actual` 由 old/new cache 的实际差异计算；固定成本只在实际 cache reconfiguration 发生时计入；删除服务不产生 loading energy；initial cache 不作为 online update 计费。

因此本项目中：

- `xi_ca` 是每 MB 新加载服务形成等效 `energy/data` 的系数，先形成 `E_ca`；
- `eta_ca` 是实际发生 cache update 时的 fixed system cost；
- `lambda_cache` 是把 cache loading energy 映射到 system objective cost 的尺度系数。

### UAV movement cost

`src/common/system_model.py::uav_movement_cost()` 当前实现为：

\[
E_{mov}=\xi_{mov}\,d(q_{old},q_{new}),
\]

\[
C_q=\eta_qz_q^{actual}+\lambda_{movement}E_{mov}.
\]

实际移动由最终 old-to-new distance 和 tolerance 判定；不移动时 distance、energy、fixed cost 和 movement cost 均为零，并检查最大水平速度约束。

因此本项目中：

- `xi_mov` 是 movement energy / distance；
- `eta_q` 是实际 UAV reconfiguration 的 fixed cost；
- `lambda_movement` 是把 movement energy 映射到 system objective cost 的尺度系数。

## Beta closure

`cache.discount_beta = 0.95`，当前状态为 `ALGORITHM_DESIGN`，不是 system-model physical parameter。代码中它只进入 prediction/configuration surrogate 的有限 horizon aggregation：

\[
\widehat Q(C,Q)=\frac{1}{S}\sum_s\sum_{h=0}^{H-1}\beta^h Q_{raw}(s,h;C,Q).
\]

actual slot `Q(t)`、`G(t)`、PPO reward、Matching/KKT 和 Direct-CF 不使用 beta。现有依据足以确认其语义是一个算法设计折扣，但没有足够的 train-only 选择证据把 `0.95` 声明为唯一冻结值；本轮因此不改变数值或状态，也不使用 test origins 选择 beta。

若正式实验协议要求进一步闭合 beta，最小协议应为：固定 representative train origins、prediction scenarios、seed、configuration requests、checkpoint 和其他参数，只比较 `beta ∈ {0.90, 0.95, 1.00}`，记录 predictive objective、actual `G`、actual cache update count、actual UAV movement count；用 train-only 结果决定是否冻结，并保留 test origins 作为最终 held-out evaluation。该 sensitivity 本轮未运行，因为本轮只做 config parse/formal gate closure，且不运行完整 PPO。

## Cache cost parameter status

### `eta_ca`

| 参数 | 当前值 | 当前/最终状态 | 单位与公式 | 依据与 provisional 原因 |
|---|---:|---|---|---|
| `nodes.sv.eta_ca` | `0.20` | `PROVISIONAL_IMPLEMENTATION` | objective cost unit；`eta_ca * z_actual` | 配置明确说明它是 SV 实际 cache reconfiguration 的 fixed cost，不是文献直接物理常数；没有 SV 的可复现实测或独立 system-scale calibration protocol。 |
| `nodes.rsu.eta_ca` | `0.10` | `PROVISIONAL_IMPLEMENTATION` | objective cost unit；`eta_ca * z_actual` | 与 SV 相同；当前只有 implementation baseline 语义，没有足够依据改为已校准状态。 |
| `nodes.uav.eta_ca` | `0.15` | `PROVISIONAL_IMPLEMENTATION` | objective cost unit；`eta_ca * z_actual` | 现有注释列出 representative dry-run、sequential validation、300-slot continuous validation 和 targeted causal validation，但这些是行为验证，不是可复现的固定成本标定 protocol；仍不足以关闭 formal status。 |

### `xi_ca`

| 参数 | 当前值 | 当前/最终状态 | 单位与公式 | 依据与 provisional 原因 |
|---|---:|---|---|---|
| `nodes.sv.xi_ca` | `0.001` | `PROVISIONAL_IMPLEMENTATION` | 等效 energy/MB；`E_ca = xi_ca * loaded_mb` | 当前是 system-scale implementation baseline；没有物理 Joule/MB 来源或独立可复现校准结果。 |
| `nodes.rsu.xi_ca` | `0.0007` | `PROVISIONAL_IMPLEMENTATION` | 等效 energy/MB；`E_ca = xi_ca * loaded_mb` | 当前是 system-scale implementation baseline；没有物理 Joule/MB 来源或独立可复现校准结果。 |
| `nodes.uav.xi_ca` | `0.0012` | `PROVISIONAL_IMPLEMENTATION` | 等效 energy/MB；`E_ca = xi_ca * loaded_mb` | 当前是 system-scale implementation baseline；没有物理 Joule/MB 来源或独立可复现校准结果。 |

### `lambda_cache`

| 参数 | 当前值 | 当前/最终状态 | 单位与公式 | 依据与 provisional 原因 |
|---|---:|---|---|---|
| `objective.lambda_cache` | `0.1` | `PROVISIONAL_IMPLEMENTATION` | objective-cost / equivalent-energy；`C_ca += lambda_cache * E_ca` | 已有初步 system-scale rationale：使用当前真实任务数据、已校准 `T_ref/E_ref` 和 normal-slot aggregate utility 尺度，以典型单节点配置动作约占 normal-slot utility 10% 为中心，并计划做 5%/10%/20% sensitivity。但当前仓库没有把该比例目标、train-only 样本、统计量、选择规则和冻结结果保存成可复现 calibration protocol；因此本轮不把它伪装成 `SYSTEM_SCALE_CALIBRATED`。 |

结论：`eta_ca` 和 `xi_ca` 没有足够校准依据；`lambda_cache` 有 system-scale 设计依据但还没有 closure-grade calibration record。三类参数均保持当前 provisional 状态，不通过删标记来绕过 formal gate。

## UAV cost parameter status

| 参数 | 当前值 | 当前/最终状态 | 单位与公式 | 依据与 provisional 原因 |
|---|---:|---|---|---|
| `nodes.uav.movement.movement_energy_coefficient_j_per_m`（`xi_mov`） | `15.0` | `PROVISIONAL_IMPLEMENTATION` | J/m；`E_mov = xi_mov * distance` | 配置注释给出 Li et al. 旋翼功率模型在 `V_ref=15 m/s` 下 `P(15)/15 ≈ 14.73 J/m`，并取整为 15 J/m。这是 literature-based equivalent mapping，不是论文直接给出的 15 J/m 物理常数；线性化速度、取整和当前系统适用范围尚未形成项目正式冻结协议，故不改状态。 |
| `nodes.uav.movement.movement_fixed_cost`（`eta_q`） | `0.00176` | `PROVISIONAL_IMPLEMENTATION` | objective cost unit；`eta_q * z_q_actual` | 现有 sequential movement audit、300-slot continuous validation 和 targeted causal validation 能证明 state-dependent 行为，但没有独立的实际重配置成本测量或可复现 scale calibration protocol。 |
| `objective.lambda_movement` | `0.0000176` | `PROVISIONAL_IMPLEMENTATION` | objective-cost / J；`C_q += lambda_movement * E_mov` | 当前依据仍是 movement audit/validation 的实现尺度说明，不是文献物理常数，也没有保存 train-only 的目标尺度、统计规则和冻结记录；保持 provisional。 |

## Minimal closure list

在不使用 `allow_provisional=True` 时，真正阻止当前 formal validation 的最小列表就是：

1. `nodes.sv.eta_ca`
2. `nodes.rsu.eta_ca`
3. `nodes.uav.eta_ca`
4. `nodes.sv.xi_ca`
5. `nodes.rsu.xi_ca`
6. `nodes.uav.xi_ca`
7. `nodes.uav.movement.movement_energy_coefficient_j_per_m`
8. `nodes.uav.movement.movement_fixed_cost`
9. `objective.lambda_cache`
10. `objective.lambda_movement`

`cache.discount_beta=0.95` 需要在算法设计协议中闭合或记录 sensitivity 选择，但不是当前 formal system-parameter gate 的 blocker。`T_ref`、`E_ref` 已是 `TRAIN_ONLY_CALIBRATION`，不在剩余 blocker 中。

## 12.pdf note

本地没有 `12.pdf`，本轮不对其做任何文件操作。若后续要更新论文/系统模型表述，建议只在有正式依据后补充以下语义，而不是把当前实现 baseline 反写成物理常数：

| 参数/公式位置 | 当前语义 | 建议后续表述 | 原因与影响 |
|---|---|---|---|
| Cache update cost | `xi_ca` 形成 energy/MB 的等效 loading energy，`eta_ca` 是 actual update fixed cost，`lambda_cache` 做 objective scale mapping | 明确区分 physical/equivalent energy、actual reconfiguration fixed cost 和 objective normalization | 防止把 implementation cost scale 误写成 literature physical constant；会影响论文参数表和 calibration protocol，不改变当前代码公式。 |
| UAV movement cost | `xi_mov` 是线性等效 J/m，`eta_q` 是 actual movement fixed cost，`lambda_movement` 做 objective scale mapping | 明确线性等效映射的参考速度、适用范围、取整规则和 scale calibration 状态 | 防止把 derived equivalent coefficient 误写成论文直接给定的物理常数；不改变本轮实验参数。 |
| Prediction objective | `beta` 只用于 finite-horizon configuration surrogate | 明确 beta 是 algorithm-design parameter，并说明 train-only sensitivity 选择协议 | 防止把算法设计选择与 system-model contract 混淆；不改变 actual Q/G/PPO reward。 |

## Final status table

| parameter | value | final status | evidence | remaining action |
|---|---:|---|---|---|
| `cache.discount_beta` | `0.95` | `ALGORITHM_DESIGN` | 只影响 prediction/configuration surrogate；当前已有语义和默认值，但无唯一 train-only sensitivity 选择证据 | 如需正式冻结，执行并记录 train-only `{0.90, 0.95, 1.00}` protocol；不使用 test origins |
| `nodes.sv.eta_ca` | `0.20` | `PROVISIONAL_IMPLEMENTATION` | fixed actual cache update cost；无 SV calibration-grade evidence | 补充可复现 system-scale/physical calibration；否则 formal blocker |
| `nodes.rsu.eta_ca` | `0.10` | `PROVISIONAL_IMPLEMENTATION` | fixed actual cache update cost；无 RSU calibration-grade evidence | 补充可复现 calibration；否则 formal blocker |
| `nodes.uav.eta_ca` | `0.15` | `PROVISIONAL_IMPLEMENTATION` | 有行为 validation，但非 fixed-cost calibration protocol | 补充可复现 calibration；否则 formal blocker |
| `nodes.sv.xi_ca` | `0.001` | `PROVISIONAL_IMPLEMENTATION` | equivalent energy/MB；无物理 Joule/MB calibration | 补充物理或明确 system-scale calibration；否则 formal blocker |
| `nodes.rsu.xi_ca` | `0.0007` | `PROVISIONAL_IMPLEMENTATION` | equivalent energy/MB；无物理 Joule/MB calibration | 补充物理或明确 system-scale calibration；否则 formal blocker |
| `nodes.uav.xi_ca` | `0.0012` | `PROVISIONAL_IMPLEMENTATION` | equivalent energy/MB；无物理 Joule/MB calibration | 补充物理或明确 system-scale calibration；否则 formal blocker |
| `objective.lambda_cache` | `0.1` | `PROVISIONAL_IMPLEMENTATION` | 有 10% normal-slot utility 的初步 system-scale rationale，但没有 closure-grade protocol/result | 固定 train-only 样本、统计量、选择规则和冻结记录后再评估是否可改为 `SYSTEM_SCALE_CALIBRATED` |
| `nodes.uav.movement.movement_energy_coefficient_j_per_m` | `15.0` | `PROVISIONAL_IMPLEMENTATION` | Li et al. 功率模型推导出的 15 J/m linear equivalent mapping；不是直接文献常数 | 明确速度映射/线性化/取整及正式适用范围；否则 formal blocker |
| `nodes.uav.movement.movement_fixed_cost` | `0.00176` | `PROVISIONAL_IMPLEMENTATION` | audit/continuous validation 只证明行为，未提供独立成本标定 | 补充可复现 actual reconfiguration cost calibration；否则 formal blocker |
| `objective.lambda_movement` | `0.0000176` | `PROVISIONAL_IMPLEMENTATION` | movement energy 到 objective 的实现尺度；无冻结 calibration record | 补充 train-only/system-scale calibration protocol；否则 formal blocker |

## Final decision

**FORMAL_READY = NO**

唯一剩余 formal blockers 是上述 10 个 `PROVISIONAL_IMPLEMENTATION` system/objective cost 参数。没有修改任何参数状态来绕过 gate；beta 作为 `ALGORITHM_DESIGN` 保持 `0.95`，后续如需闭合应按 train-only sensitivity protocol 处理。

本轮只新增本报告；未生成 profiling、CSV、JSON、benchmark 或临时测试文件，也未提交或推送。
