# Cost parameter identifiability and status closure

## Scope

本轮只处理：

- `parameter_status` vocabulary 增加 `LITERATURE_DERIVED`；
- `nodes.uav.movement.movement_energy_coefficient_j_per_m` 的 status closure；
- Cache cost 中 `lambda_cache * xi_ca` 的可辨识性记录。

本轮不运行 PPO、Batch4、rollout、Matching/KKT、Cache/UAV 动态验证或性能测试；不修改三个 `xi_ca` 数值或状态，也不处理 `eta`、`lambda`、`beta` 的状态；不修改 `12.pdf`。

基线 commit：`550d12f11c283bf071727b8e988ba2130e9f8714`。

## Status vocabulary and formal gate

已在 `src/common/project_config.py::list_unresolved_parameters()` 的受控 vocabulary 中加入：

```text
LITERATURE_DERIVED
```

其含义已在 `system_config.toml` 的 `[parameter_status]` 注释中记录为：由明确文献模型推导的等效参数；不是文献直接测量值或直接表格常数。

`list_unresolved_parameters()` 仍然只把以下两类视为 unresolved：

```text
PROVISIONAL_IMPLEMENTATION
TODO_LITERATURE_CALIBRATION
```

因此加入 `LITERATURE_DERIVED` 只是承认有依据的状态，不会放宽或绕过 `PROVISIONAL_IMPLEMENTATION` 的 formal gate 限制。

## `xi_mov` status closure

参数：`nodes.uav.movement.movement_energy_coefficient_j_per_m`

- 数值保持 `15.0 J/m`；
- status 从 `PROVISIONAL_IMPLEMENTATION` 改为 `LITERATURE_DERIVED`；
- 没有新增 `LITERATURE_CALIBRATED`，也没有把该参数表述为直接测量值。

依据链为 Li et al. 的 rotary-wing model / Table 1，以及其引用的 Zeng et al. rotary-wing propulsion model。当前项目已经记录的数值复核为：

```text
P(15) = 220.960140363616 W
P(15) / 15 m/s = 14.730676024241 J/m
rounding -> 15.0 J/m
```

这里 `15 m/s` 是项目选择的代表速度，并与文献 Table 1 的最大速度一致；所以 `15 J/m` 是明确来源模型在代表速度处得到的 linear equivalent，不是论文直接给出的常数。`LITERATURE_DERIVED` 是合适的状态，`LITERATURE_CALIBRATED` 不是。

## Cache cost parameter identifiability

### Sign convention

成本本身保持正值；系统目标在外层扣除成本，统一记录为：

\[
C_{ca,m}
=\eta_{ca,m}z_{actual}
+\lambda_{cache}\xi_{ca,m}loaded\_mb.
\]

\[
C_q
=\eta_qz_q^{actual}
+\lambda_{movement}\xi_{mov}distance,
\]

并且 `G = Q - C_{ca} - C_q`。`system_model.cache_update_cost()` 与 `uav_movement_cost()` 都返回正的 cost；负号只出现在系统目标 `G` 中。下面的可辨识性结论因此直接针对正成本数据项：`xi_ca,m` 和 `lambda_cache` 只以乘积出现。

### Identifiability result

如果 `xi_ca,m` 与 `lambda_cache` 都通过同一 system-scale objective calibration 选择，那么对任意正数 `a`，替换

\[
\xi'_{ca,m}=a\xi_{ca,m},
\qquad
\lambda'_{cache}=\lambda_{cache}/a
\]

会保持：

\[
\lambda'_{cache}\xi'_{ca,m}
=\lambda_{cache}\xi_{ca,m}.
\]

因此，仅根据 train-only system utility、objective 或 reward，最多能够辨识：

```text
lambda_cache * xi_ca,m
```

而不能独立辨识或声称物理校准了：

```text
xi_ca,m
lambda_cache
```

这不是数值精度问题，而是参数化结构的 scale non-identifiability。三个 `xi_ca` 当前值和 status 均保持不变：

| 参数 | 当前值 | 当前 status | 本轮结论 |
|---|---:|---|---|
| `nodes.sv.xi_ca` | `0.001` | `PROVISIONAL_IMPLEMENTATION` | 未找到匹配的独立物理 J/MB 依据；不改状态 |
| `nodes.rsu.xi_ca` | `0.0007` | `PROVISIONAL_IMPLEMENTATION` | 未找到匹配的独立物理 J/MB 依据；不改状态 |
| `nodes.uav.xi_ca` | `0.0012` | `PROVISIONAL_IMPLEMENTATION` | 未找到匹配的独立物理 J/MB 依据；不改状态 |

## Two legal future paths

本轮只记录路径，不执行 calibration。

### A. Physical path

1. 为 SV、RSU、UAV 节点类型分别找到真实 storage-energy / J-per-MB 外部依据，或进行有明确硬件边界和测量协议的 storage-energy benchmark。
2. 独立固定每类节点的 `xi_ca,m`，不使用 system utility 去同时调它。
3. 在 `xi_ca,m` 已独立固定后，再校准 `lambda_cache` 作为 objective cost scale。
4. 报告测量设备/硬件类型、读写操作、数据量范围、baseline subtraction、重复次数、统计量和不确定性。

该路径能够支持物理解释，适合在论文需要声称 J/MB 时采用；代价是需要当前仓库之外的硬件或可靠外部测量依据。

### B. Equivalent-design path

1. 明确声明三个 `xi_ca` 是非物理的 relative/equivalent design coefficients，不描述为 measured J/MB。
2. 固定其相对比例 `0.001 : 0.0007 : 0.0012`，不让 `xi_ca,m` 和 `lambda_cache` 同时自由漂移。
3. 只校准整体 objective cost scale，并在 calibration protocol 中固定归一化基准、train-only 数据、目标统计量和选择规则。
4. 对 test data 只做最终 held-out evaluation，不参与 scale 选择。

### Recommendation

当前推荐 **B. Equivalent-design path**，理由是：

- 当前没有 SV/RSU/UAV 对应的独立 storage-energy/J-per-MB 物理证据；
- train-only utility 不能打破 `lambda_cache * xi_ca,m` 的乘积不可辨识性；
- 现有三项数值已经承担节点类型之间的相对成本比例，保留该比例比伪装成物理 J/MB 更诚实；
- A 路径只有在补充外部硬件/测量依据后才成立。

推荐 B 不等于本轮把三个 `xi_ca` 标记为已校准；本轮按要求保持它们的数值和 `PROVISIONAL_IMPLEMENTATION` status 不变。后续若要把等效设计状态单独纳入 vocabulary，应另行定义并审查对应 formal gate 语义。

## Future `12.pdf` wording note

本地没有 `12.pdf`，本轮没有搜索、创建、复制或修改它。若未来需要更新系统模型，只记录以下建议：

| 参数/公式位置 | 当前语义 | 建议语义 | 原因 | 影响 |
|---|---|---|---|---|
| UAV movement energy | `E_mov = xi_mov * distance`，其中 `xi_mov=15 J/m` 来自 `P(15)/15` 的文献模型等效 | 明确标为 literature-derived linear equivalent，并写出 `V_ref=15 m/s`、`P(15)` 和取整规则 | 防止把 derived coefficient 写成直接测量/直接表格常数 | 只影响论文参数语义，不改变当前数值 |
| Cache loading cost | `xi_ca,m` 进入 `lambda_cache * xi_ca,m * loaded_mb` | 明确 `xi_ca,m` 与 `lambda_cache` 的 joint scale non-identifiability；若无外部测量，只能作为等效设计系数 | 避免从 system utility 反推出独立物理 J/MB | 需要补充 calibration protocol 表述，不改变本轮算法 |

## Validation scope

本轮只应执行：

- `system_config.toml` parse/type validation；
- status vocabulary 和 formal gate 的最小检查；
- `git diff --check`。

不运行 PPO、Batch4、rollout、Matching/KKT、Cache/UAV 动态验证或性能测试；不 commit、不 push。
