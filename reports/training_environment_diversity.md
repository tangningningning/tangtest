# 训练环境多样性检查

本报告由只读、单进程 CPU 脚本生成。分析顺序处理训练 episode；没有导入项目运行时模块，没有执行训练、rollout、SUMO/TraCI、Matching、KKT、Exact-Q 或反事实环境代码。未读取 validation/test episode 内容。

## 1. 数据协议

- Manifest：`environment_data/data_split_manifest.json`，protocol_version=1，split_unit=`complete_sequential_realization`，slots_per_realization=300。
- Train 声明：20001–20300，count=300；审计范围固定为 20001–20300。
- Validation 声明：30001–30030，count=30；Test 声明：40001–40100，count=100。
- Service catalog：10 services；manifest SHA256 匹配=YES。
- Manifest 说明 training pool 是既有 300-realization 池且没有逐 episode `episode_manifest.json`；因此没有把共享配置值描述成逐 episode snapshot 验证结果。

## 2. 完整性

- EXPECTED_EPISODES = 300
- FOUND_EPISODES = 300
- MISSING_EPISODES = 0 (无)
- UNEXPECTED_EPISODES = 0 (无)
- POOL_COMPLETE = YES
- 必需文件缺失 episode 数 = 0

## 3. 重复与 SHA256 检查

表中计数的分母为 300；缺失文件没有 hash。Realization signature 是 trajectory、tasks、popularity、vehicle_roles、tv_local_cache 五个文件名及其 SHA256 按固定顺序拼接后再做 SHA256。

| Artifact | Unique hashes | 重复 episode seed 组 |
|---|---:|---|
| `trajectory.csv` | 300/300 | 无 |
| `vehicle_roles.json` | 300/300 | 无 |
| `tasks.csv` | 300/300 | 无 |
| `popularity.csv` | 300/300 | 无 |
| `tv_local_cache.json` | 300/300 | 无 |
| `vehicle_predictions.csv` | 300/300 | 无 |
| `popularity_predictions.csv` | 300/300 | 无 |
| `prediction_scenarios.npz` | 300/300 | 无 |
| `REALIZATION_SIGNATURE` | 300/300 | 无 |

- FULL_REALIZATION_UNIQUE_COUNT = 300/300
- REALIZATION_DUPLICATION = NO

## 4. Vehicle diversity

- RESEARCH_VEHICLE_SET_UNIQUE_COUNT = 300/300
- SV_SET_UNIQUE_COUNT = 300/300
- TV_SET_UNIQUE_COUNT = 300/300
- 每 episode formal research vehicle rows：以 roles warmup/dt 对齐正式 slot 0…N−1 后统计；position dispersion 是每 slot 研究车辆二维坐标相对该 slot 质心的 RMS 距离，再对 slot 取均值。速度/加速度取正式 300-slot 研究车辆记录。

| Episode-level metric | mean | std | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean speed (m/s) | 10.3865 m/s | 0.333672 m/s | 9.16636 m/s | 10.1595 m/s | 10.3638 m/s | 10.6381 m/s | 11.237 m/s |
| Speed std (m/s) | 8.71246 m/s | 0.2487 m/s | 8.04347 m/s | 8.54619 m/s | 8.71662 m/s | 8.88354 m/s | 9.44015 m/s |
| Position dispersion (m) | 551.333 m | 18.7791 m | 503.572 m | 539.337 m | 551.878 m | 564.867 m | 603.08 m |
| Mean vehicle travel distance (m) | 3102.11 m | 100 m | 2736.9 m | 3035.39 m | 3095.02 m | 3176.59 m | 3358.66 m |

- Mean research vehicles / SV / TV per episode = 25 / 5 / 20。mean_vehicle_travel_distance 使用 formal window 内 `distance_m` 首末记录差；无法形成至少两个记录的车辆不计入。

| Episode-level metric | mean | std | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean acceleration (m/s²) | -0.00944033 m/s² | 0.00727802 m/s² | -0.0321175 m/s² | -0.0140752 m/s² | -0.00966096 m/s² | -0.00471616 m/s² | 0.0148701 m/s² |
| Episode min speed (m/s) | 0 m/s | 0 m/s | 0 m/s | 0 m/s | 0 m/s | 0 m/s | 0 m/s |
| Episode max speed (m/s) | 32.6717 m/s | 1.38057 m/s | 29.7031 m/s | 31.7073 m/s | 32.4797 m/s | 33.562 m/s | 38.64 m/s |

## 5. Task diversity

- Config/catalog 确认 F=10；num_task_vehicles=20；episode slots=300；arrival_probability=0.7。
- `total_task_count` 是 `arrival=1` 行数；realized arrival rate 的分母是读到的正式 TV×slot opportunity 行数。输入大小和 cpu_cycles 统计只针对到达任务。

| Episode-level metric | mean | std | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Task count | 4200.53 | 38.5638 | 4079 | 4177 | 4204 | 4226 | 4295 |
| Arrival rate | 0.700088 | 0.00642731 | 0.679833 | 0.696167 | 0.700667 | 0.704333 | 0.715833 |
| Mean input size | 2.18342e+07 | 2.2228e+06 | 1.61853e+07 | 2.01564e+07 | 2.19119e+07 | 2.35802e+07 | 2.7105e+07 |
| Within-episode input-size std | 1.06654e+07 | 934822 | 8.8738e+06 | 9.93809e+06 | 1.05734e+07 | 1.13255e+07 | 1.29423e+07 |
| Mean cpu_cycles | 1.57319e+10 | 1.91522e+09 | 1.09005e+10 | 1.43331e+10 | 1.58194e+10 | 1.71637e+10 | 2.02791e+10 |
| Within-episode cpu_cycles std | 8.83025e+09 | 8.1507e+08 | 6.91537e+09 | 8.22762e+09 | 8.78055e+09 | 9.43358e+09 | 1.08652e+10 |

- 300×F service-probability vectors由 `tasks.csv` 到达任务的 service_id 计数归一化；每个 episode 的概率列写入 summary CSV。TASKS_UNIQUE_COUNT = 300/300。

## 6. Service request distribution diversity

- 两两 Jensen–Shannon divergence 使用 log₂，单位 bits；比较 episode empirical service probability。近似相同的诊断阈值为 JS ≤ 0.0001 bits。该阈值仅用于 near-identical pair count，不代表外部质量标准。
- 有效 service vectors = 300/300；pair count = 44850；NEAR_IDENTICAL_SERVICE_DISTRIBUTION_PAIR_COUNT = 0。

| Pairwise JS (bits) | mean | std | min(non-identical) | p10 | p25 | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All valid episode pairs | 0.114729 | 0.0381624 | 0.00354937 | 0.0616644 | 0.0874254 | 0.118608 | 0.144411 | 0.161812 | 0.210614 |

## 7. Popularity diversity

- schema 自动识别到列：`epoch, rank, request_count, rho_obs, rho_star, service_id, slot, time_s`；true popularity 取 `rho_star`，top order 优先按 `rank`，没有 rank 时按 rho_star 降序。
- INITIAL_TOP1_SERVICE_FREQUENCY = {"0": 34, "1": 29, "2": 41, "3": 29, "4": 25, "5": 29, "6": 29, "7": 33, "8": 23, "9": 28}
- INITIAL_TOP3_DIVERSITY = 115 unique top-3 sets；initial top-3 覆盖服务数 = 10。
- Labeled popularity path unique = 300/300；按每 episode 初始 rank 重排服务 label 后的 canonical path unique = 300/300。
- 所有 labeled path 相同 = NO；仅 label permutation、rank evolution 相同 = NO。
- `top3_turnover_count` 计 top-3 集合发生变化的相邻 slot 转移数；rate 为该数/可比较相邻转移数。`pop_rho_tv_drift` 是相邻 slot rho_star 分布的 0.5×L1/total variation distance；rank change 是跨服务平均绝对 rank 改变量。

| Episode-level metric | mean | std | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Top-1 change count | 2.08 | 1.35902 | 0 | 1 | 2 | 3 | 7 |
| Top-3 changed-transition count | 1.38333 | 1.12682 | 0 | 1 | 1 | 2 | 6 |
| Top-3 changed-transition rate | 0.00462653 | 0.00376863 | 0 | 0.00334448 | 0.00334448 | 0.00668896 | 0.0200669 |
| Temporal rho_star TV drift | 0.00138191 | 0.000501265 | 0.000355024 | 0.000998056 | 0.00132482 | 0.00166864 | 0.00300981 |
| Temporal mean absolute rank change | 0.00936455 | 0 | 0.00936455 | 0.00936455 | 0.00936455 | 0.00936455 | 0.00936455 |

## 8. Prediction difficulty

- Popularity prediction 使用已保存 `rho_pred` 与 `rho_star` truth；若 `abs_error` 有值则使用该列，否则按 |rho_pred−rho_star| 计算。统计 MAE/RMSE，共 4470000 truth-labeled rows。
- Vehicle position prediction 逐行核对 `target_slot == origin_slot + horizon`，用 `vehicle_id + target_slot` 对齐 formal trajectory `x_m/y_m` 与 `x_pred/y_pred`，误差为平面欧氏距离。有效对齐行=11175000；in-range unmatched=0；key mismatch=0；episode 外 horizon=75000。
- POSITION_PREDICTION_ERROR_CHECK = COMPUTABLE_FROM_PERSISTED_SCHEMA。

| Popularity prediction MAE | mean | std | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Episode MAE | 0.023811 | 0.00500336 | 0.0133682 | 0.0206309 | 0.0233012 | 0.0262385 | 0.0515838 |

| Popularity prediction RMSE | mean | std | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Episode RMSE | 0.0328083 | 0.00653833 | 0.0175186 | 0.0282524 | 0.0319452 | 0.0365511 | 0.0648157 |

| Vehicle position prediction MAE (m) | mean | std | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Episode MAE | 11.1135 | 0.18488 | 10.511 | 10.9897 | 11.1108 | 11.2357 | 11.6569 |

## 9. Fixed experiment contract consistency

- FIXED_EXPERIMENT_CONTRACT_CONSISTENT = YES
- 逐 episode roles metadata 与 shared formal config 中可见的 dt/warmup/episode length/车辆角色数量/slot count 对照；service count 对照 catalog；预测 horizon 对照配置。没有 episode-level snapshot 的参数标为 shared formal config 固定。
- `simulation.dt_s` = `1.0` (fixed by shared formal config)
- `simulation.warmup_time_s` = `100.0` (fixed by shared formal config)
- `simulation.episode_time_s` = `300.0` (fixed by shared formal config)
- `simulation.num_research_vehicles` = `25` (fixed by shared formal config)
- `simulation.num_service_vehicles` = `5` (fixed by shared formal config)
- `simulation.num_task_vehicles` = `20` (fixed by shared formal config)
- `service.count` = `10` (fixed by shared formal config)
- `task.arrival_probability` = `0.7` (fixed by shared formal config)
- `task.popularity_zeta` = `0.8` (fixed by shared formal config)
- `task.popularity_epoch_length` = `20` (fixed by shared formal config)
- `task.popularity_changes_per_epoch` = `1` (fixed by shared formal config)
- `prediction.vehicle.horizon` = `5` (fixed by shared formal config)
- `prediction.vehicle.sigma_position_m` = `3.0` (fixed by shared formal config)
- `prediction.vehicle.sigma_speed_mps` = `1.0` (fixed by shared formal config)
- `prediction.popularity.history_length` = `20` (fixed by shared formal config)
- `scenario.num_scenarios` = `20` (fixed by shared formal config)
- `scenario.cross_vehicle_residual_sampling` = `independent` (fixed by shared formal config)
- `nodes.cloud.cloud_delay_factor` = `11.0` (fixed by shared formal config)
- `nodes.cloud.cpu_cycles_per_s` = `5000000000.0` (fixed by shared formal config)
- `nodes.cloud.resource_sharing` = `False` (fixed by shared formal config)
- `nodes.cloud.total_bandwidth_bps` = `200000000.0` (fixed by shared formal config)
- `nodes.cloud.total_cpu_cycles_per_s` = `5000000000.0` (fixed by shared formal config)
- `nodes.cloud.uplink_rate_bps` = `200000000.0` (fixed by shared formal config)
- `nodes.rsu.bandwidth_hz` = `10000000.0` (fixed by shared formal config)
- `nodes.rsu.cache_capacity_mb` = `44652.6675325681` (fixed by shared formal config)
- `nodes.rsu.count` = `2` (fixed by shared formal config)
- `nodes.rsu.cpu_cycles_per_s` = `5000000000.0` (fixed by shared formal config)
- `nodes.rsu.deployment_altitude_m` = `10.0` (fixed by shared formal config)
- `nodes.rsu.eta_ca` = `0.1` (fixed by shared formal config)
- `nodes.rsu.range_m` = `250.0` (fixed by shared formal config)
- `nodes.rsu.xi_ca` = `0.0007` (fixed by shared formal config)
- `nodes.sv.bandwidth_hz` = `10000000.0` (fixed by shared formal config)
- `nodes.sv.cache_capacity_mb` = `16744.750324713` (fixed by shared formal config)
- `nodes.sv.cpu_cycles_per_s` = `1000000000.0` (fixed by shared formal config)
- `nodes.sv.eta_ca` = `0.2` (fixed by shared formal config)
- `nodes.sv.range_m` = `100.0` (fixed by shared formal config)
- `nodes.sv.xi_ca` = `0.001` (fixed by shared formal config)
- `nodes.uav.altitude_m` = `100.0` (fixed by shared formal config)
- `nodes.uav.bandwidth_hz` = `10000000.0` (fixed by shared formal config)
- `nodes.uav.cache_capacity_mb` = `33489.5006494261` (fixed by shared formal config)
- `nodes.uav.count` = `2` (fixed by shared formal config)
- `nodes.uav.cpu_cycles_per_s` = `5000000000.0` (fixed by shared formal config)
- `nodes.uav.deployment_altitude_m` = `100.0` (fixed by shared formal config)
- `nodes.uav.deployment_offset_fraction` = `0.1` (fixed by shared formal config)
- `nodes.uav.eta_ca` = `0.15` (fixed by shared formal config)
- `nodes.uav.initialization_mode` = `topology_aware` (fixed by shared formal config)
- `nodes.uav.max_horizontal_speed_mps` = `15.0` (fixed by shared formal config)
- `nodes.uav.min_safety_distance_m` = `10.0` (fixed by shared formal config)
- `nodes.uav.movement.movement_energy_coefficient_j_per_m` = `15.0` (fixed by shared formal config)
- `nodes.uav.movement.movement_fixed_cost` = `0.00176` (fixed by shared formal config)
- `nodes.uav.range_m` = `300.0` (fixed by shared formal config)
- `nodes.uav.xi_ca` = `0.0012` (fixed by shared formal config)
- `nodes.vehicle.kappa` = `1e-28` (fixed by shared formal config)
- `nodes.vehicle.local_cache_capacity_mb` = `5581.58344157102` (fixed by shared formal config)
- `nodes.vehicle.local_cpu_cycles_per_s` = `1000000000.0` (fixed by shared formal config)
- `nodes.vehicle.transmit_power_w` = `0.1995262315` (fixed by shared formal config)
- 当前 config 文件 SHA256 与 manifest 记录值不同；manifest note：Comments identifying historical calibration subsets were clarified during generation; parsed generator fields and generation_config_fingerprint were unchanged.。解析后的字段仍按共享 config 和 roles metadata 核对。

## 10. Train / validation / test leakage

- TRAIN_VALIDATION_OVERLAP = 无
- TRAIN_TEST_OVERLAP = 无
- VALIDATION_TEST_OVERLAP = 无
- TRAIN_VALIDATION_TEST_SEED_LEAKAGE = NO
- split directory names exist = YES；只检查 validation/test 目录是否存在，没有列举其内容或读取其中 episode 文件。

## 11. 最终结论

- VEHICLE_ENVIRONMENT_VARIATION = CONFIRMED
- TASK_ENVIRONMENT_VARIATION = CONFIRMED
- POPULARITY_VARIATION = CONFIRMED
- PREDICTION_DIFFICULTY_VARIATION = CONFIRMED
- TRAIN_ENVIRONMENT_DIVERSITY = SUFFICIENT

本分类是对本次持久化数据的保守描述，不使用外部所谓顶刊阈值。对总分类中的 service spread，若至少一半 episode pair 落在明示的 near-identical 诊断区间，则按 LIMITED 处理；这是透明的多数描述规则，不是外部质量标准。Service JS 门槛仅按第 6 节明示用于 pair 诊断。

运行约束记录：TRAINING_EXECUTED=NO；SUMO_EXECUTED=NO；MATCHING_EXECUTED=NO；KKT_EXECUTED=NO；GPU_USED=NO；MULTIPROCESSING_USED=NO；OTHER_PROCESSES_MODIFIED=NO。
