# Unified train-only reconfiguration cost scale closure

## Scope and boundary

本轮只处理 Cache/UAV reconfiguration cost 的状态和 system-scale 证据：

- `nodes.sv/rsu/uav.xi_ca`
- `nodes.sv/rsu/uav.eta_ca`
- `objective.lambda_cache`
- `nodes.uav.movement.movement_energy_coefficient_j_per_m`（`xi_mov`）
- `nodes.uav.movement.movement_fixed_cost`（`eta_q`）
- `objective.lambda_movement`

未运行 PPO、Batch4、完整 rollout、Matching/KKT、Cache/UAV 动态验证、并行性能测试或新的参数 sweep；没有使用 test origins `240..299`。本地没有 `12.pdf`，本轮没有搜索、创建或修改它，也没有修改论文正文。用户文件 `打包并上传.bat` 未修改。

基线 commit：`550d12f11c283bf071727b8e988ba2130e9f8714`。

## Train-only reference protocol

优先复用仓库已有 train-only 产物，并只在内存中重新计算聚合统计：

1. `reports/cache_dynamic_multislot_validation.md`：固定 seed `4401`、origin/slot `0..49`、50 个 normal slots、682 个到达任务。该 baseline 的 actual cache update 为 0，actual UAV movement 也为 0。由于当前 `G` 是带符号的任务 utility，成本比例使用 `|G|` 的 robust typical scale，不改变 `G` 的符号。
2. `reports/cache_sensitivity_analysis.md`：同样是 train slots `0..49` 的既有 sensitivity artifact。5 GHz Cloud stress group 的 slot 38 有 1 次真实 UAV cache update；只把它作为已有 actual payload scale 样本，不把该 stress group 当作当前正式 baseline，也不据此选择参数。
3. `data/analysis/uav_no_move_explanation.csv`：既有 train-only UAV probe，31 个 slot、93 个 node-slot 记录。最终 optimizer movement distance 全部为 0；候选 moving distance 为 3.75 m，只作为 candidate scale probe。

normal-slot `G` 的统计为：

| statistic | value |
|---|---:|
| slots | 50 (`0..49`) |
| arrivals | 682 |
| `G` mean | `-113.79516506` |
| `|G|` P25 | `92.01082625` |
| `|G|` P50 / typical scale | `107.283565` |
| `|G|` P75 | `131.364324` |
| `|G|` min / max | `42.4775 / 214.264929` |

选择规则是固定 train-only 样本上的 P50；P25/P75 仅用于尺度诊断，不参与参数 sweep 或 test selection。

## Positive cost convention

成本本身统一为正值，负号只在系统目标中出现：

```text
C_ca = eta_ca * z_actual
        + lambda_cache * xi_ca * loaded_mb

C_q  = eta_q * z_q_actual
        + lambda_movement * xi_mov * distance

G = Q - C_ca - C_q
```

本轮没有改 `system_model` 数学实现。这里的 `xi_ca` 是 equivalent/relative design coefficient，不声称是 measured physical J/MB；`xi_mov=15 J/m` 是由文献旋翼功率模型推导的 linear equivalent。

## Cache cost scale

### Actual train-only payload evidence

已有 5 GHz stress artifact 的 slot 38 UAV update 为：

```text
loading cost = 2.067582075
fixed cost   = 0.15
total cost   = 2.217582075
```

该 update 使用当前 `lambda_cache=0.1` 和 `xi_ca,uav=0.0012`，因此可复现地反推该已有样本的 loaded payload：

```text
loaded_mb = 2.067582075 / (0.1 * 0.0012)
          = 17229.850625 MB
```

这是 `n=1` 的已有 actual-update sample，且来自 5 GHz stress condition；当前 20 GHz train-only baseline 没有 actual cache update。因此它能用于 payload/成本尺度核对，不能单独完成 `lambda_cache` calibration。

将该同一 payload 作为跨节点类型的 scale comparison（SV/RSU 两行不是额外 actual update）得到：

| node type | `eta_ca*z_actual` | `lambda_cache*xi_ca*loaded_mb` | total `C_ca` | total / `|G|` P50 |
|---|---:|---:|---:|---:|
| SV | `0.200000` | `1.722985063` | `1.922985063` | `1.7924%` |
| RSU | `0.100000` | `1.206089544` | `1.306089544` | `1.2174%` |
| UAV | `0.150000` | `2.067582075` | `2.217582075` | `2.0670%` |

既有 baseline candidate diagnostics 还给出：

- slot 0 candidate cache cost mean `0.535070`，约为 `0.4987%` 的 `|G|` P50；
- representative RSU swap total cache cost `1.235182`，约为 `1.1513%` 的 `|G|` P50；
- 当前 20 GHz baseline 的 50 slots 没有 actual cache update，所以 actual baseline cache cost 的 P25/P50/P75 都是 0。

这些值说明当前数值形成了可解释的 experiment-scale baseline，但没有证据支持把原配置注释中的“约 10% normal-slot utility”作为已经完成的 calibration target：没有保存对应的预声明目标、选择规则和冻结记录，而且现有 actual/candidate proxy 的比例约为 `0.5%..2.1%`（相对于本报告选定的 `|G|` P50）。因此不把 `lambda_cache=0.1` 改成 `TRAIN_ONLY_CALIBRATION`。

### Cache status decision

- `xi_ca` 三个数值保持不变，固定相对比例 `0.001:0.0007:0.0012`；由于存在 scale non-identifiability，它们只作为 equivalent/relative cache-loading coefficients，状态改为 `EXPERIMENT_DESIGN`。
- `eta_ca` 三个数值保持不变；它们是节点类型相关的 actual reconfiguration fixed system-cost design，不是物理能耗常数，状态改为 `EXPERIMENT_DESIGN`。
- `lambda_cache=0.1` 保持 `PROVISIONAL_IMPLEMENTATION`，直到有独立、可复现的 train-only scale target、统计规则和冻结选择记录。不得与 `xi_ca` 同时自由校准。

## UAV cost scale

既有 UAV train-only artifact 的 actual final movement distance 为 0（93/93 node-slot rows，nonzero count 0），所以当前 baseline 的 actual movement cost 也是 0，不能从这些 actual rows 单独辨识 `lambda_movement`。

对已有 3.75 m candidate movement probe，按当前参数复算：

| quantity | value |
|---|---:|
| distance | `3.75 m` |
| movement energy `xi_mov*distance` | `56.25 J` |
| fixed cost `eta_q` | `0.00176` |
| variable cost `lambda_movement*energy` | `0.00099` |
| total candidate movement cost | `0.00275` |
| total / `|G|` P50 | `0.0025633%` |

该 3.75 m 是候选 movement，不是当前 train-only baseline 的 accepted final movement；因此它只能说明当前 movement cost 的数量级，不能形成 `lambda_movement` 的 calibration result。

### UAV status decision

- `xi_mov=15.0` 保持 `LITERATURE_DERIVED`。此前已有 Li/Zeng rotary-wing model 的复核：`P(15)=220.960140363616 W`，`P(15)/15=14.730676024241 J/m`，项目取 `15 J/m`；它是 literature-derived linear equivalent，不是 `LITERATURE_CALIBRATED`。
- `eta_q=0.00176` 保持数值，明确为人为 fixed reconfiguration penalty，状态改为 `EXPERIMENT_DESIGN`；这不是 physical movement-energy parameter。
- `lambda_movement=1.76e-5` 保持 `PROVISIONAL_IMPLEMENTATION`。现有 actual movement 全为 0，且没有预声明 train-only target、选择规则和冻结记录，不能伪装成 `TRAIN_ONLY_CALIBRATION`。

## Identifiability and no joint free tuning

Cache cost 的 loading 项为：

```text
lambda_cache * xi_ca,m * loaded_mb
```

如果 `xi_ca,m` 与 `lambda_cache` 同时由同一 system-utility calibration 选择，则对任意正数 `a`，替换 `xi_ca,m <- a*xi_ca,m`、`lambda_cache <- lambda_cache/a` 保持 loading cost 不变。train-only system utility 只能辨识乘积，不能独立给出两个物理量。因此本轮没有做 `xi_ca × lambda_cache` 联合 sweep，也没有对六个成本参数做笛卡尔积搜索。

## Formal status result

本轮将以下 status 写入当前正式配置：

- `nodes.sv/rsu/uav.eta_ca` -> `EXPERIMENT_DESIGN`
- `nodes.sv/rsu/uav.xi_ca` -> `EXPERIMENT_DESIGN`
- `nodes.uav.movement.movement_fixed_cost` -> `EXPERIMENT_DESIGN`
- `nodes.uav.movement.movement_energy_coefficient_j_per_m` 保持 `LITERATURE_DERIVED`
- `objective.lambda_cache`、`objective.lambda_movement` 保持 `PROVISIONAL_IMPLEMENTATION`
- `cache.discount_beta` 保持既有 `ALGORITHM_DESIGN`；本轮不处理 beta

严格 formal gate（`allow_provisional=False`）的最小剩余 blocker 为：

1. `objective.lambda_cache`
2. `objective.lambda_movement`

因此：

```text
FORMAL_READY = NO
```

这不是通过删除 unresolved 标记得到的结论；`PROVISIONAL_IMPLEMENTATION` 的 gate 限制仍然有效。

## Final status table

| parameter | value | role | calibration/design basis | final status | formal blocker yes/no |
|---|---:|---|---|---|---|
| `nodes.sv.eta_ca` | `0.20` | SV fixed cache reconfiguration cost | fixed system-objective design; non-physical; train-only cost scale sanity check | `EXPERIMENT_DESIGN` | no |
| `nodes.rsu.eta_ca` | `0.10` | RSU fixed cache reconfiguration cost | fixed system-objective design; non-physical; train-only cost scale sanity check | `EXPERIMENT_DESIGN` | no |
| `nodes.uav.eta_ca` | `0.15` | UAV fixed cache reconfiguration cost | fixed system-objective design; non-physical; one existing train-only actual-update scale sample | `EXPERIMENT_DESIGN` | no |
| `nodes.sv.xi_ca` | `0.001` | SV relative cache-loading coefficient | equivalent design coefficient; fixed relative ratio; no measured J/MB claim | `EXPERIMENT_DESIGN` | no |
| `nodes.rsu.xi_ca` | `0.0007` | RSU relative cache-loading coefficient | equivalent design coefficient; fixed relative ratio; no measured J/MB claim | `EXPERIMENT_DESIGN` | no |
| `nodes.uav.xi_ca` | `0.0012` | UAV relative cache-loading coefficient | equivalent design coefficient; fixed relative ratio; no measured J/MB claim | `EXPERIMENT_DESIGN` | no |
| `nodes.uav.movement.movement_energy_coefficient_j_per_m` | `15.0` | movement energy / distance | Li/Zeng rotary-wing model at `V_ref=15 m/s`, `P(15)/15`, rounded linear equivalent | `LITERATURE_DERIVED` | no |
| `nodes.uav.movement.movement_fixed_cost` | `0.00176` | actual UAV reconfiguration fixed penalty | explicit non-physical experiment-design penalty; candidate/actual train-only scale check | `EXPERIMENT_DESIGN` | no |
| `objective.lambda_cache` | `0.1` | cache loading energy -> objective cost scale | current train-only evidence lacks predeclared target, selection rule and freeze record | `PROVISIONAL_IMPLEMENTATION` | yes |
| `objective.lambda_movement` | `0.0000176` | movement energy -> objective cost scale | actual movement sample is all zero; no reproducible train-only calibration freeze | `PROVISIONAL_IMPLEMENTATION` | yes |

## Future `12.pdf` wording note

本地没有 `12.pdf`，本轮不对其做任何文件操作。若未来更新系统模型，建议只记录：

| position | current semantics | suggested semantics | reason | impact |
|---|---|---|---|---|
| Cache reconfiguration cost | `C_ca = eta_ca*z_actual + lambda_cache*xi_ca*loaded_mb` | 明确 `xi_ca` 是固定相对/等效设计系数，不是 measured J/MB；`lambda_cache` 是整体 objective scale | 避免把乘积不可辨识的两个量分别描述成物理校准量 | 只影响论文参数语义和 calibration protocol，不改代码公式 |
| UAV movement cost | `C_q = eta_q*z_q_actual + lambda_movement*xi_mov*distance` | 明确 `xi_mov` 是由旋翼模型在 `V_ref=15 m/s` 推导的 literature-derived linear equivalent；`eta_q` 是 experiment-design fixed penalty | 区分文献推导、实验设计和待校准 objective scale | 只影响论文参数语义，不改当前数值 |
