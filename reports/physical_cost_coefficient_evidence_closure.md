# Physical/equivalent cost coefficient evidence closure

## Scope

本轮只处理以下四个参数：

- `nodes.sv.xi_ca`
- `nodes.rsu.xi_ca`
- `nodes.uav.xi_ca`
- `nodes.uav.movement.movement_energy_coefficient_j_per_m`

本轮不处理 `eta`、`lambda`、`beta`，不修改 Matching/KKT、Cache/UAV optimizer、PPO、系统模型公式或参数数值，不运行 Batch4、完整 rollout、训练或新的性能/Matching/KKT 测试。

基线 commit：`550d12f11c283bf071727b8e988ba2130e9f8714`。

本地没有 `12.pdf`。本轮没有创建、复制、搜索或修改 `12.pdf`，也没有修改论文正文。

## Evidence sources

### UAV rotary-wing source

当前 `system_config.toml` 注释引用的是：

> W.-T. Li, M. Zhao, Y.-H. Wu, et al., “Collaborative offloading for UAV-enabled time-sensitive MEC networks,” *Journal on Wireless Communications and Networking*, 2021:1, DOI [`10.1186/s13638-020-01861-8`](https://doi.org/10.1186/s13638-020-01861-8).

该论文的正式页面给出：

- Eq. (12) 的 rotary-wing propulsion power model；
- `P_0`、`P_i`、`Q_tip`、`rho`、`A`、`v_0`、`d_0`、`s` 等符号定义；
- Table 1 的 UAV 参数；
- 该模型采用自旋翼 UAV 文献模型，并在参考文献中指向 Zeng, Xu and Zhang 的原始 rotary-wing propulsion model。

具体参考为：Y. Zeng, J. Xu, and R. Zhang, “Energy Minimization for Wireless Communication With Rotary-Wing UAV,” *IEEE Transactions on Wireless Communications*, 18(4), 2329–2345, 2019, DOI [`10.1109/TWC.2019.2902559`](https://doi.org/10.1109/TWC.2019.2902559)。

因此，当前配置注释的准确证据链是：

```text
Zeng et al. rotary-wing model
        -> Li et al. Eq. (12) and Table 1 reuse/instantiate the model
        -> project evaluates the model at one representative speed
        -> project converts power to linear energy-per-distance
```

### Cache loading energy sources

Li et al. also write caching energy as proportional to cached data amount, with a storage-hardware coefficient (`omega_ca`) for the UAV/GN side. The related reference is J. Xu, K. Ota, and M. Dong, “Saving Energy on the Edge: In-Memory Caching for Multi-Tier Heterogeneous Networks,” *IEEE Communications Magazine*, 56(5), 102–107, 2018, DOI [`10.1109/MCOM.2018.1700909`](https://doi.org/10.1109/MCOM.2018.1700909).

The cited papers support the *form* “caching energy is proportional to cached data,” but the checked sources do not provide the project’s three numeric values `0.001`, `0.0007`, and `0.0012` energy/MB for SV, RSU, and UAV. They also do not provide a node-specific mapping from the project’s service-loading operation to those three values.

The repository evidence is consistent with that conclusion:

- `src/common/system_model.py::cache_update_cost()` computes `loaded_mb = additions @ service_sizes_mb` and `loading_energy = xi * loaded_mb`.
- `reports/cache_uav_predictive_objective_closure_audit.md` and `reports/cache_utility_calculation_audit.md` verify the implementation semantics, but do not identify a physical Joule/MB source for the three values.
- `reports/cache_sensitivity_analysis.md` uses the values as a baseline and varies them by `0.5x/1x/2x`; that is sensitivity evidence, not physical calibration evidence.

## `xi_mov` derivation

### Source parameters

The values transcribed in the Li et al. Table 1 and reflected in the project comments are:

| Symbol | Value | Unit |
|---|---:|---|
| `P_0` | `158.76` | W |
| `P_i` | `88.63` | W |
| `Q_tip` | `120.0` | m/s |
| `rho` | `1.225` | kg/m³ |
| `A` | `1.0` | m² |
| `v_0` | `4.03` | m/s |
| `d_0` | `0.3` | dimensionless |
| `s` | `0.05` | dimensionless |
| `V_ref` | `15.0` | m/s |

`V_ref=15 m/s` is the project’s representative evaluation speed. It equals the Li et al. Table 1 maximum UAV velocity, but using `V_max` as a representative speed for a linear equivalent coefficient is a project modeling choice; it is not a statement that the published paper directly defines `15 J/m`.

### Power model

The Li et al. Eq. (12) model used by the project is:

\[
P(V)=P_0\left(1+\frac{3V^2}{Q_{tip}^2}\right)
+\frac{1}{2}d_0s\rho A V^3
+P_i\left(
\sqrt{1+\frac{V^4}{4v_0^4}}-
\frac{V^2}{2v_0^2}
\right)^{1/2}.
\]

At `V_ref=15 m/s`, the three terms are:

| Term | Calculation | Result |
|---|---|---:|
| blade profile | `158.76 * (1 + 3*15²/120²)` | `166.201875000000 W` |
| parasitic/fuselage drag | `0.5 * 0.3 * 0.05 * 1.225 * 1 * 15³` | `31.007812500000 W` |
| induced | `88.63 * (sqrt(1 + 15⁴/(4*4.03⁴)) - 15²/(2*4.03²))^(1/2)` | `23.750452863616 W` |
| total | sum of the three terms | `220.960140363616 W` |

The conversion from power to energy per distance is dimensionally:

\[
\xi_{mov}^{eq}=
\frac{P(V_{ref})}{V_{ref}}
=\frac{220.960140363616\ {m J/s}}
{15\ {m m/s}}
=14.730676024241\ {m J/m}.
\]

Rounding this equivalent coefficient to one decimal place gives:

```text
14.730676024241 J/m -> 15.0 J/m
```

The current value `15.0` is therefore numerically reproducible and the existing derivation is correct.

### Classification decision for `xi_mov`

The evidence is sufficient to classify the parameter semantically as:

```text
LITERATURE_DERIVED_EQUIVALENT
```

or, using the requested shorter label, `LITERATURE_DERIVED`.

It is **not** sufficient to call it `LITERATURE_CALIBRATED`, because:

1. Li et al. publish the speed-dependent rotary-wing model and its physical parameters, not a direct `15 J/m` constant.
2. The project chooses one representative speed (`15 m/s`) and rounds the resulting `14.730676... J/m` to `15.0 J/m`.
3. The full model is speed-dependent, while the project uses a linear `energy = xi_mov * distance` equivalent.

This evidence is stronger than an unsupported provisional implementation value, but it does not justify describing `15 J/m` as a directly measured or directly tabulated physical constant.

The repository’s current controlled status vocabulary does not contain `LITERATURE_DERIVED` or `LITERATURE_CALIBRATED`; it currently contains `CONFIRMED_DATA`, `TRAIN_ONLY_CALIBRATION`, `EXPERIMENT_DESIGN`, `ALGORITHM_DESIGN`, `PROVISIONAL_IMPLEMENTATION`, and `TODO_LITERATURE_CALIBRATION`. Therefore this report recommends the semantic classification only. The current TOML status is not changed in this task, and no gate workaround is introduced.

## `xi_ca` evidence decision

### Current implementation meaning

The current code implements:

\[
E_{ca,m}=\xi_{ca,m}\sum_f s_f
\left[c_{m,f}^{new}-c_{m,f}^{old}\right]^+.
\]

Thus each `xi_ca` has the project-level meaning “equivalent cache-loading energy per MB of newly loaded service data.” It is not a measured write-energy coefficient unless an external storage-energy measurement is supplied.

### Literature evidence decision

| Parameter group | Decision | Reason |
|---|---|---|
| `nodes.sv.xi_ca` | **B — equivalent system-scale coefficient; calibration still required** | The cited caching literature supports proportionality to data amount, but no SV-specific numeric energy/MB source matching `0.001` was found. |
| `nodes.rsu.xi_ca` | **B — equivalent system-scale coefficient; calibration still required** | No RSU-specific numeric energy/MB source matching `0.0007` was found. |
| `nodes.uav.xi_ca` | **B — equivalent system-scale coefficient; calibration still required** | The Li/Xu references support a symbolic storage-energy coefficient, but no numeric UAV value matching `0.0012` was found. |

No source was invented or inferred from the current numerical values. The current values should not be relabeled `PHYSICAL` or `LITERATURE_CALIBRATED`.

## Minimal `xi_ca` calibration protocol

This protocol is designed only; it was not run in this task.

1. **Scope and fixed inputs**

   Use only fixed train origins (`0..239`) and the existing frozen train artifacts. Freeze task data, prediction scenarios, service sizes, configuration requests, candidate order, initial cache state, seeds, node positions, and all non-`xi_ca` parameters. Do not read test origins (`240..299`) during fitting or selection.

2. **Target quantity**

   Measure the incremental storage energy attributable to loading new service data, in J, after subtracting the no-load baseline for the same node type and operation. The target slope is:

   \[
   E_{storage,m}^{incremental}\;[J]
   \approx \xi_{ca,m}\;loaded\_mb\;[MB].
   \]

   The fixed intercept/control component must not be folded into the slope; otherwise it would contaminate `xi_ca` with a fixed update cost.

3. **Train-only fixture**

   Replay a predeclared representative set of train-origin configuration requests, with repeated controlled loads covering the actual service-size range. Keep the service order, cache state transitions, and loaded MB values fixed and record the measured incremental storage energy for SV, RSU, and UAV separately.

4. **Statistic**

   Fit one slope per node type using a zero-free-intercept regression after baseline subtraction, and report the median slope across repeated train fixtures together with the P95 absolute relative error. The median is the selected point estimate; the P95 error is a reproducibility/quality diagnostic, not an additional tuned parameter.

5. **Selection rule**

   Accept the coefficient only if the predeclared P95 relative-error threshold is met and the fitted slope is non-negative and stable across the fixed train fixture repetitions. Otherwise keep the current value explicitly marked as equivalent system-scale provisional and do not use the test origins to rescue the fit.

6. **No-data outcome**

   If storage-energy telemetry or a documented storage hardware energy model is unavailable, this protocol cannot produce physical `J/MB` evidence. In that case the values remain B-category equivalent system-scale coefficients and require a separately declared objective-scale calibration protocol; no physical status should be claimed.

## Required output table

| parameter | value | evidence source | derivation | recommended status | remaining action |
|---|---:|---|---|---|---|
| `nodes.sv.xi_ca` | `0.001` energy/MB | Current `system_model.cache_update_cost()`; existing cache reports; Li et al. symbolic caching-energy model and cited Xu et al. in-memory caching paper | `loading_energy = xi_ca * loaded_mb`; no numeric SV source found | `PROVISIONAL_IMPLEMENTATION` / equivalent system-scale coefficient | Run the train-only measurement protocol only after a target/telemetry source is fixed; do not claim physical closure now |
| `nodes.rsu.xi_ca` | `0.0007` energy/MB | Same sources; no numeric RSU source found | `loading_energy = xi_ca * loaded_mb`; no numeric RSU source found | `PROVISIONAL_IMPLEMENTATION` / equivalent system-scale coefficient | Same train-only calibration protocol; no status promotion now |
| `nodes.uav.xi_ca` | `0.0012` energy/MB | Li et al. gives proportional UAV caching-energy form; Xu et al. gives in-memory caching context; no numeric project-matching UAV value found | `loading_energy = xi_ca * loaded_mb`; no numeric UAV source found | `PROVISIONAL_IMPLEMENTATION` / equivalent system-scale coefficient | Same train-only calibration protocol; no status promotion now |
| `nodes.uav.movement.movement_energy_coefficient_j_per_m` | `15.0 J/m` | Li et al. Eq. (12), Table 1; Zeng et al. rotary-wing model | `P(15)=220.960140363616 W`; `P(15)/15=14.730676024241 J/m`; rounded to `15.0 J/m` | `LITERATURE_DERIVED` (more precisely literature-derived equivalent) | If adopted in config status later, first extend the controlled status vocabulary/gate in a separate explicit change; retain the documented `V_ref` and rounding rule |

## 12.pdf note

No `12.pdf` operation was performed. If the system model is updated in the future, the only recommended wording change is documentation-level:

| Position/formula | Current semantics | Recommended semantics | Reason | Impact |
|---|---|---|---|---|
| UAV movement energy coefficient | `E_mov = xi_mov * distance`, with `xi_mov=15 J/m` obtained from `P(15)/15` and rounded | State that `xi_mov` is a literature-derived linear equivalent at `V_ref=15 m/s`, not a directly tabulated constant; retain the full speed-dependent source model as the derivation basis | Separates the published rotary-wing model from the project’s single-speed linearization | Documentation and parameter-status wording only; no change to current numeric value or algorithm |
| Cache loading energy | `E_ca = xi_ca * loaded_mb` with three project coefficients | State that `xi_ca` values are equivalent system-scale coefficients unless a node-specific physical calibration is added | Current literature supports proportional form but not the three numeric values | Requires later calibration documentation; no change in this task |

## Conclusion

- `xi_mov=15.0 J/m`: derivation is correct; evidence supports `LITERATURE_DERIVED` / literature-derived equivalent, but not `LITERATURE_CALIBRATED`.
- `nodes.sv/rsu/uav.xi_ca`: no matching numeric physical/literature source was found; all three remain category B equivalent system-scale coefficients requiring a future train-only calibration protocol.
- No parameter value, algorithm, optimizer, PPO path, or formal configuration was changed in this task.
- No temporary scripts, CSV, JSON, benchmark, or profiling files were created.
