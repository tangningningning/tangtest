# Lambda objective-weight design closure

## Scope

本轮只处理：

- `objective.lambda_cache = 0.1`
- `objective.lambda_movement = 1.76e-5`

未运行 PPO、Batch4、rollout、Matching/KKT、joint optimizer、性能测试或新的参数 sweep；没有使用 test slots `240..299`。本地没有 `12.pdf`，本轮没有搜索、创建或修改它。`打包并上传.bat` 未修改。

参数数值保持不变。

## 1. Mathematical role

静态代码核对结果如下。

### `lambda_cache`

- `src/decision/cache_configuration.py::cache_update_cost()` 从 `objective.lambda_cache` 读取该值，并把它传给 `src/common/system_model.py::cache_update_cost()`。
- `system_model.cache_update_cost()` 先计算 `loading_energy = xi_ca * loaded_mb`，再计算：

  ```text
  cache_cost = eta_ca * actual_trigger
             + lambda_cache * loading_energy
  ```

- `lambda_cache` 不计算物理 loading energy，也不改变 `xi_ca`、service size 或 cache state；它只把 loading-energy-like quantity 映射到 objective cost。

### `lambda_movement`

- `src/decision/uav_configuration.py` 将 `objective.lambda_movement` 传入 `src/common/system_model.py::uav_movement_cost()`。
- `uav_movement_cost()` 先计算 `movement_energy = xi_mov * effective_distance`，再计算：

  ```text
  movement_cost = eta_q * actual_z_q
                + lambda_movement * movement_energy
  ```

- `lambda_movement` 不计算 UAV 物理能耗，也不改变 `xi_mov`、距离或位置候选；它只把 movement-energy-like quantity 映射到 objective cost。

因此二者均不是 physical energy parameter，也不是需要从硬件直接测出的常数。它们的数学身份是 objective trade-off weight：控制 reconfiguration energy 相对于 task utility 的惩罚尺度。

## 2. Train-only candidate scale evidence

本轮只读取已有 train-only artifacts，并用固定 normal-slot reference 做比例核对。reference 来自 `reports/cache_dynamic_multislot_validation.md`：seed `4401`、slots `0..49`、50 slots、682 arrivals。该窗口 `G` 的 typical magnitude 使用 `|G|` P50：

```text
|G| P25 = 92.01082625
|G| P50 = 107.283565       <- reference scale
|G| P75 = 131.364324
```

成本本身保持正值，系统目标仍为：

```text
C_ca = eta_ca*z_actual + lambda_cache*xi_ca*loaded_mb
C_q  = eta_q*z_q_actual + lambda_movement*xi_mov*distance
G    = Q - C_ca - C_q
```

这些统计只描述当前值的数量级，不用 accepted cache update、accepted UAV movement 或 optimizer 最终状态来选择 lambda。

### A. Cache candidate scale

已有 `reports/cache_utility_calculation_audit.md` 的 representative RSU swap 给出 loading cost `1.135182`、fixed cost `0.100000`、total cost `1.235182`。按当前 `lambda_cache=0.1` 和 `xi_ca,rsu=0.0007` 可复算：

```text
loaded_mb           = 1.135182 / (0.1*0.0007)
                    = 16216.885714 MB
xi_ca*loaded_mb     = 11.351820
lambda*xi*loaded_mb = 1.135182
fixed eta_ca        = 0.100000
total candidate C   = 1.235182
total / |G| P50     = 1.1513%
```

作为另一种已有 train-only payload scale，`reports/cache_sensitivity_analysis.md` 的 slot 38 UAV cache update 给出：

| quantity | value |
|---|---:|
| representative loaded payload | `17229.850625 MB` |
| `xi_ca,uav * loaded_mb` | `20.675820750` |
| `lambda_cache * xi_ca,uav * loaded_mb` | `2.067582075` |
| fixed `eta_ca,uav` | `0.150000` |
| total candidate/update cost | `2.217582075` |
| total / `|G|` P50 | `2.0670%` |

该 UAV sample 来自已有 train-only 5 GHz stress condition；它只用于说明当前 weight 下的 cost scale，不用于反推或选择 lambda。当前 20 GHz baseline 的 actual cache update 为 0，因此没有把 accepted update behavior 当作 calibration signal。

已有 2,864 个 train-only cache candidate 的 aggregate diagnostic 还记录：

| candidate statistic | value |
|---|---:|
| service loading cost min / mean / max | `0 / 0.360966 / 1.386427` |
| control cost min / mean / max | `0.100000 / 0.142668 / 0.200000` |
| total candidate cost mean | `0.503634` |
| total candidate cost mean / `|G|` P50 | `0.4694%` |

这些 candidate cost 与 train-only `|G|` reference 同量级可解释，足以记录 `lambda_cache=0.1` 的 experimental penalty scale；但不构成 physical calibration 或 data-fitted lambda。

### B. UAV candidate scale

已有 train-only UAV candidate artifact 的 candidate movement distance 为 `3.75 m`（候选 distance 的 P25/P50/P75 均为 `3.75 m`）。按当前参数：

| quantity | value |
|---|---:|
| representative candidate distance | `3.75 m` |
| `xi_mov * distance` | `56.25 J` |
| `lambda_movement * xi_mov * distance` | `0.000990` |
| fixed `eta_q` | `0.001760` |
| total candidate movement cost | `0.002750` |
| total / `|G|` P50 | `0.0025633%` |

该数值是 candidate cost scale，不是用 optimizer 最终是否移动来选取 lambda。它显示 `lambda_movement=1.76e-5` 在当前实验尺度下提供一个很小但明确的 movement-energy penalty。

## 3. Design interpretation and status

当前两个值均应明确描述为：

> 人为选择的 objective trade-off weights，用于控制 reconfiguration energy 相对于 task utility 的惩罚尺度。

不作以下声明：

- 不是 `PHYSICAL` 参数；
- 不是 `LITERATURE_CALIBRATED`；
- 不是 `TRAIN_ONLY_CALIBRATION`；
- 不由 accepted action、accepted update count 或 optimizer final state 反向选择。

因此将两个 status 改为 `EXPERIMENT_DESIGN`。这表示实验设计语义已经闭合，不表示数值是物理真值或从 train utility 拟合得到。

## 4. Future sensitivity design only

本轮不运行 sensitivity。正式论文实验可分别采用 one-factor-at-a-time：

```text
lambda_cache:
    0.5x = 0.05
    1.0x = 0.10
    2.0x = 0.20

lambda_movement:
    0.5x = 8.8e-6
    1.0x = 1.76e-5
    2.0x = 3.52e-5
```

不做 `3 x 3` 联合 grid；每次只改变一个 lambda，其他参数、数据、seed、训练协议和评估边界保持不变。

## 5. Formal gate

当前正式配置中的两个参数 status 为：

```text
objective.lambda_cache    = EXPERIMENT_DESIGN
objective.lambda_movement = EXPERIMENT_DESIGN
```

`EXPERIMENT_DESIGN` 仍视为 formal gate 的允许状态；对 `PROVISIONAL_IMPLEMENTATION` 的限制未放宽。严格 gate 只在本轮状态更新后运行一次。

## Final table

| parameter | value | mathematical role | observed train-only scale | design interpretation | final status | formal blocker |
|---|---:|---|---|---|---|---|
| `objective.lambda_cache` | `0.1` | `loading_energy -> objective cache cost` | RSU candidate total `1.235182` = `1.1513%` of `|G|` P50；UAV sample total `2.217582075` = `2.0670%` | objective trade-off weight; no physical or fitted-calibration claim | `EXPERIMENT_DESIGN` | no |
| `objective.lambda_movement` | `1.76e-5` | `movement_energy -> objective movement cost` | 3.75 m candidate total `0.002750` = `0.0025633%` of `|G|` P50 | objective trade-off weight; no physical or fitted-calibration claim | `EXPERIMENT_DESIGN` | no |

```text
FORMAL_READY = YES
```

## Future `12.pdf` wording note

本地没有 `12.pdf`，本轮没有对其做任何文件操作。若未来需要修改论文表述，建议：

- 将 `lambda_cache` 写成 cache loading energy 到 objective cost 的 experiment-design weight；
- 将 `lambda_movement` 写成 UAV movement energy 到 objective cost 的 experiment-design weight；
- 明确二者不代表 physical energy coefficient、literature constant 或 train-only fitted calibration；
- 保留 `C_ca`、`C_q` 为正成本，负号只出现在 `G = Q - C_ca - C_q`。
