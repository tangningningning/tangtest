# Proposed joint realization generation trace

审计范围：只读检查 `environment_data/proposed_joint_training`、`proposed_joint_validation`、`proposed_joint_test` 的 manifest、样本文件和当前生成/训练代码。未启动训练、validation/test evaluation 或环境数据生成；未修改数据、配置或算法。

## 结论标签

- **CONFIRMED**：由当前代码、manifest 或抽样文件直接支持。
- **INFERENCE**：由时间戳、文件布局等间接推出，原始记录没有明确说明。
- **NOT_RECORDED / NOT_FOUND**：检查的 manifest、代码和报告中没有该记录。

## 1. Split manifest 与 seed 范围

来源：`environment_data/data_split_manifest.json`。

| 字段 | 值 |
|---|---|
| `DATA_PROTOCOL_VERSION` | 1 |
| nominal scenario | `proposed_joint_nominal` |
| realization 粒度 | `complete_sequential_realization` |
| `SLOTS_PER_REALIZATION` | 300；origin slot 0–299 |
| `TRAIN_REALIZATION_COUNT` / `TRAIN_SEEDS` | 300；20001–20300（连续、无缺号） |
| `VALIDATION_REALIZATION_COUNT` / `VALIDATION_SEEDS` | 30；30001–30030（连续、无缺号） |
| `TEST_REALIZATION_COUNT` / `TEST_SEEDS` | 100；40001–40100（连续、无缺号） |
| train–validation overlap | 0 |
| train–test overlap | 0 |
| validation–test overlap | 0 |

上述数量和连续性已按三个目录的实际 episode 子目录复核；目录名以 `episode_<seed>` 解析。Training manifest 标为 `preexisting 300-realization pool`，抽样 `episode_20001` 没有 `episode_manifest.json`。因此本任务没有重新生成 training；逐 episode 原始运行来源不能从 training 样本完全还原。Project report `reports/stage2c_formal_protocol.md` 记载 validation/test 是通过同一 pool generator、相同解析后设置和独立 seed streams 生成的；30001 和 40001 的 episode manifests 也存在。

Manifest provenance 字段：

| 信息 | 记录状态 |
|---|---|
| generator entrypoint | **CONFIRMED**：`scripts/generation/generate_episode_pool.py` |
| single-realization generator | **CONFIRMED**：`scripts/generation/generate_episode_realization.py` |
| 配置路径 | **CONFIRMED**：`configs/proposed/system_config.toml` |
| config 文件 SHA256 | **CONFIRMED**：`09783c6b360443f18da1c96209aebf925cc080a429750350e1c6ea97471e1e5b`；manifest 另记生成开始时 hash `759d42cc2b303339e5f71dded2f40a9d4858e3a1878b66104d689081644a0cb9`。注释变化说明称解析字段和 fingerprint 未变化。 |
| parsed generation fingerprint | **CONFIRMED**：`9ad3c6179be6cea67a0c95695f0cc2802e7570274c9be5446e51d996d5f7045d` |
| seed policy | **CONFIRMED**：`adaptive_cache_mec_episode_seed_v1` |
| source directory 字段 | **NOT_RECORDED** |
| generation timestamp 字段 | **NOT_RECORDED** |

Val/test 的 episode manifests 记录 episode seed streams、config fingerprint/fields、输入 artifact hashes、输出文件 hashes、source git SHA/dirty 状态和车辆/预测身份；它们没有 generator 脚本路径或生成时间字段。Training 目录的 split-level manifest 说明其为 preexisting pool，但没有逐 episode manifest；`episode_20001/sumo_demand.meta.json` 记录的 SUMO seed `1026025884` 和 traffic seed `1969414811` 与当前 seed helper 对 episode seed 20001 的派生值一致。这能支持 seed policy 一致，不能补足缺失的 training source commit 或原始生成时间。

## 2. 三个抽样 realization 的文件与 schema

抽样目录：

- `environment_data/proposed_joint_training/episode_20001`
- `environment_data/proposed_joint_validation/episode_30001`
- `environment_data/proposed_joint_test/episode_40001`

Training 20001 的 11 个正式文件：

| 文件 | 实际用途 |
|---|---|
| `trajectory.csv` | SUMO 逐步车辆状态：位置、速度、加速度、道路/车道、路线及距离等。 |
| `vehicle_roles.json` | research、SV、TV、background vehicle IDs，角色选择种子和 planned routes 等。 |
| `sumo_demand.rou.xml` | 此 realization 的 SUMO route demand。 |
| `sumo_demand.meta.json` | `randomTrips` 生成器、路网 hash、SUMO/traffic seed 与 randomTrips 参数。 |
| `tasks.csv` | 300 slots × 20 TV 的实际 task arrival、service、input size、CPU cycles。 |
| `popularity.csv` | 每 slot 的真实 `rho_star`、服务 rank 和由请求形成的 count/`rho_obs`。 |
| `popularity_history.csv` | 正式 slot 前 20 个历史 slots 的请求 count 和 `rho_obs`。 |
| `tv_local_cache.json` | 该 realization 的 TV 初始本地 cache。 |
| `vehicle_predictions.csv` | 每个 origin/horizon/research vehicle 的预测位置。抽样文件有 truth/error 列，但所有抽样行的这些列为空。 |
| `popularity_predictions.csv` | 每个 origin/horizon/service 的 `rho_pred`，并有离线评估用的 `rho_star`/`abs_error` 标签。 |
| `prediction_scenarios.npz` | 已生成的车辆位置及未来任务不确定性 scenarios。 |

Validation 30001 和 Test 40001 各有上述 11 个文件，另有 `episode_manifest.json`。因此：

`DIRECTORY_SCHEMA_IDENTICAL = NO`

区别是 train 样本缺少 `episode_manifest.json`；val/test 样本包含它。其余列出的生成文件一致。Training 没有单独的 episode config snapshot；val/test manifest 中包含 config fields 和 fingerprint。

三个 `prediction_scenarios.npz` 样本均含相同 shape：`positions_xy (300,20,5,25,2)`、scenario task arrays `arrival/service_id/input_size/cpu_cycles (300,20,5,20)`、`rho_hat (300,5,10)`，以及 origin、vehicle/role IDs 和 metadata。抽样 roles 的数量都为 25 research = 5 SV + 20 TV；角色 ID 不同，例如 SV：train `[46,10,0,23,59]`，val `[53,37,48,8,20]`，test `[4,19,57,39,38]`。

## 3. 代码中的实际生成调用链

Split manifest 指定的批量调用为：

```text
scripts/generation/generate_episode_pool.py:main
  → generate_pool(config_file, pool_dir, start_seed, count, ...)
    → scripts/generation/generate_episode_realization.py:run_episode(config, seed, output_dir, ...)
```

证据：`generate_episode_pool.py:123-205,224-239`。`generate_pool` 对 seed 区间递增；已有目录先走完整性校验并跳过，缺失目录才创建并调用 `run_episode`。

`run_episode` 的实际 realization 生成关系（`generate_episode_realization.py:527-754`）：

```text
load shared service catalog / frozen GRU / calibration residual bank / SUMO network
  → sumo.generate_random_trips
  → sumo.build_sumo_command
  → sumo.collect_trajectories
  → sumo.find_complete_vehicles
  → sumo.build_role_assignments
  → write trajectory.csv + vehicle_roles.json
  → task.build_episode_artifacts + task.output_realization_artifacts
  → task.generate_tv_local_cache
  → pop.main(mode="predict") → popularity_prediction.main / run_frozen_inference
  → scenario.main(episode_seed, roles, trajectory, popularity, history, catalog,
                  residual bank, frozen GRU, SUMO network)
      → build_vehicle_nominal_predictions / predict_rho_hat
      → sample_positions / sample_tasks
      → write vehicle_predictions.csv + prediction_scenarios.npz
  → validate_written_outputs → write_episode_manifest
```

The `scenario.main` call arguments are passed directly at `generate_episode_realization.py:707-743`; the task and popularity stages are at `630-704`. The pool entrypoint can skip a complete existing destination, so its code alone does not prove which invocation produced an already existing directory. For validation/test the project report and per-episode manifests provide provenance; for the older training pool the exact invocation is **NOT_RECORDED**.

## 4. SUMO、trajectory 和车辆角色

**Current generator behavior — CONFIRMED:** for a new episode directory, `run_episode` calls `sumo.generate_random_trips` and `sumo.collect_trajectories` (`generate_episode_realization.py:581-623`). `sumo_scenario.generate_random_trips` invokes SUMO's official `randomTrips.py` with that episode's `traffic_seed` and writes `sumo_demand.rou.xml`/metadata inside the episode (`sumo_scenario.py:527-620`). The SUMO command uses the shared network `data/sumo/map/osm.net.xml.gz`, that episode's route file and its `sumo_seed` (`sumo_scenario.py:642-674`); TraCI then starts and advances SUMO while reading active vehicle states (`712-769`). There is no shared static `.sumocfg` in this generator path.

The separate `sumo_scenario.load_existing_trajectory` helper (`817-873`) is not called by `run_episode`; it is not the source of a freshly generated pool realization. `generate_random_trips` can reuse a matching route file only when that file already exists in the same episode output directory and its metadata matches. New seed directories start with no such route file.

With `dt_s=1`, `warmup_time_s=100`, and `episode_time_s=300`, SUMO runs through 400 seconds. `_formal_step_keys` selects simulation steps 100–399 as the 300 formal slots, labeled 0–299 (`sumo_scenario.py:777-787`). The saved trajectory includes observed SUMO rows; `find_complete_vehicles` accepts research candidates only when they have valid rows for every formal step plus at least one warmup observation (`876-930`). A vehicle that exits or lacks a formal row is not padded or forward-filled and cannot be selected as a research vehicle. If observed at all, it remains in the seen/background set.

Role assignment occurs after SUMO. `build_role_assignments` first samples 25 complete research vehicles with `research_role_seed`, then samples 5 SV from those 25 with `service_role_seed`; the remaining 20 research vehicles are TVs. `background_vehicles` is the set of all seen IDs outside research (`sumo_scenario.py:933-1029`). Task generation receives only TV IDs; prediction scenarios receive the 25 research IDs. Background IDs remain in the recorded SUMO trajectory/roles but are not task or prediction participants in this realization pipeline.

The vehicle forecast uses observations only through `origin_slot - 1` for its filtered state and writes blank formal future-truth columns (`scenario_generator.py:223-346`). It does use each vehicle's SUMO planned route geometry to map predicted along-route distance into x/y; this is a route-plan input, not future realized x/y positions. This audit therefore finds no use of future realized trajectory coordinates in the forecast. If the intended online system would not know a vehicle's planned route, that route-availability assumption is a separate modeling boundary.

## 5. Tasks、service catalog 和 popularity

Current config (`configs/proposed/system_config.toml:75-134`): 300 seconds, 100-second warmup, 25 research vehicles, 20 TVs, 5 SVs, 10 services; `arrival_probability=0.7`, `popularity_zeta=0.8`, popularity epoch length 20 slots, one local rank change per epoch, input jitter ±20%, workload jitter ±10%, and config `task.master_seed=4401`.

`run_episode` loads the shared `data/generated/service_catalog.json` read-only (`generate_episode_realization.py:539-554`); it does not generate a new service catalog per realization. `task.build_episode_artifacts` receives four explicit independent named seeds (`task_scenario.py:1226-1313`): popularity evolution, arrival, service request, and workload variation. Per TV/slot, `generate_tasks` draws arrival from the arrival RNG; for arrivals it chooses service using the current true `rho_star` with the service-request RNG; it applies separate workload RNG jitter to catalog input size and cycles-per-bit to produce `input_size` and `cpu_cycles` (`task_scenario.py:810-900`). The formal tasks are therefore actual tasks sampled from true popularity. The later *forecast scenario* tasks are a distinct object sampled from `rho_hat`, not the true future task rows.

Popularity uses an episode-seeded random initial service ranking, normalized Zipf probabilities `rank^-0.8`, and one adjacent swap/local rank shift at each 20-slot boundary. A 20-slot prehistory is generated before formal slot 0. Actual request counts produce observed popularity (`task_scenario.py:613-658,682-802`). The underlying `rho_star` path varies by episode seed; `service_id` requests are drawn from the true current distribution.

Seed mapping is confirmed by `src/scenario/realization.py:8-50`: SHA256 of `adaptive_cache_mec_episode_seed_v1|<episode_seed>|<stream_name>`, first eight bytes big-endian; `traffic_seed` and `sumo_seed` are reduced modulo 2,147,483,647, and other streams are masked to signed 63 bits. `episode_20001` derives:

| Stream | Seed | Consumer |
|---|---:|---|
| `traffic_seed` | 1,969,414,811 | SUMO `randomTrips.py` |
| `sumo_seed` | 1,026,025,884 | SUMO simulation |
| `research_role_seed` | 5,038,670,160,323,318,684 | research vehicle selection |
| `service_role_seed` | 2,794,087,158,078,899,364 | SV selection |
| `popularity_seed` | 2,385,688,445,386,333,329 | true popularity evolution |
| `task_arrival_seed` | 2,253,202,722,599,364,550 | per-TV task arrivals |
| `service_request_seed` | 2,121,286,528,588,826,775 | service IDs |
| `workload_seed` | 8,372,590,287,469,390,440 | input size and CPU cycles |
| `tv_cache_seed` | 5,066,335,077,674,933,237 | TV initial local cache |
| `vehicle_observation_seed` | 437,647,847,738,389,600 | vehicle observation noise |
| `scenario_seed` | 8,011,707,370,906,388,575 | scenario residual/task sampling |

The training sample has no per-episode stream manifest; only its recorded SUMO/traffic values were cross-checked against the helper. Val/test manifests store all these derived streams.

## 6. Prediction and leakage boundary

**Vehicle forecast:** `scenario_generator.build_vehicle_nominal_predictions` loads the realization's trajectory and planned routes, builds noisy observations, replays the Kalman filter, and at each origin predicts from cutoff `origin_slot - 1` (`scenario_generator.py:223-346`; `vehicle_prediction.py:633-696,811-903`). `predict_nominal` accepts filtered state and route geometry, not future truth. The formal `vehicle_predictions.csv` output writes `s_true/x_true/y_true/error` columns as blank (`scenario_generator.py:302-336`); all 37,500 rows in each audited sample have those columns blank. The trainer interface reads only `x_pred/y_pred` from that CSV (`ppo_simple/interface.py:665-700`).

**Popularity forecast:** config uses horizon 5, history length 20, and GRU hidden size 32 (`system_config.toml:425-451`); split and episode manifests bind the same frozen GRU hash. `pop.main(mode="predict")` loads the shared frozen checkpoint and calls `run_frozen_inference`, which only loads the checkpoint and predicts (`generate_episode_realization.py:675-705`; `popularity_prediction.py:755-770`). Formal input construction uses history/request observations through the causal cutoff; slot 0 uses only the 20 prehistory request-count rows (`popularity_prediction.py:247-283,1040-1074,1248-1253`). Its CSV contains `rho_star` and `abs_error` for offline label/error reporting for targets inside the episode (14,900 of 15,000 rows in each audited sample); those labels are not model inputs. The PPO interface reads `rho_pred` only (`interface.py:703-737`).

**Prediction scenarios:** `scenario_generator` forms `rho_hat` with the frozen GRU; it samples vehicle residual blocks from the shared independent calibration bank, independently across vehicles, and samples future scenario tasks from `rho_hat`. The current config has 20 scenarios and independent cross-vehicle residual sampling (`system_config.toml:435-459`; `scenario_generator.py:107-160,428-537,538-676,1098-1132`). `scenario_seed` drives the saved positions/task draws; random streams are also separated by origin. `prediction_scenarios.npz` is part of each realization and is loaded as such by PPO; PPO does not resample it during an epoch (`trainer.py:3568-3601`). Metadata says the online boundary is vehicle observations/request counts no later than `origin_slot - 1`, future tasks are sampled from `rho_hat` rather than `rho_star`, and future ground truth is not saved (`scenario_generator.py:1209-1245`).

`FUTURE_TRAJECTORY_LEAKAGE = NO` for future realized position/state values entering PPO prediction inputs. The saved vehicle prediction truth columns are empty, the interface reads predicted coordinates only, and scenarios store predictions plus calibration residual draws. Planned route geometry is used as described above.

`FUTURE_REQUEST_LEAKAGE = NO` for prediction/strategy inputs. The popularity CSV co-locates future `rho_star` labels for offline errors, but the predictor is causal and PPO loads only `rho_pred`; actual `tasks.csv` is consumed slot-by-slot in the sequential rollout, while future scenario tasks are generated from predicted `rho_hat`.

## 7. Fixed values and per-realization variation

| 项目 | Train/Val/Test 是否相同 | realization 间变化 | 来源 / 依据 |
|---|---|---|---|
| 研究区域 / road network | 是 | 否 | 同一 `osm.net.xml.gz`；manifest 记录相同 network hash。 |
| RSU 位置 | 是 | 否 | shared `data/generated/node_positions.json`。 |
| UAV 初始位置 | 是 | 否 | 同一 node positions artifact；实际 UAV policy 后续动作可变。 |
| SV / TV 数量 | 是：5 / 20 | 否 | system config；每集 25 research。 |
| SV / TV vehicle IDs | 不同 | 是 | 每 episode 的 SUMO demand 与 role seeds。三个样本角色清单不同。 |
| vehicle trajectory | 名义生成设置相同 | 是 | per-episode randomTrips/SUMO seed；存入 realization。 |
| service catalog / service size | 是 | 否 | 共用 catalog 文件及同一 hash；size/input/cycles 基础参数冻结。 |
| task arrivals | 同一 Bernoulli 参数 | 是 | 0.7 arrival probability；episode `task_arrival_seed`。 |
| service requests | 同一 Zipf 参数/流程 | 是 | episode `service_request_seed` 和真实 `rho_star`。 |
| input size | 同一 catalog/抖动规则 | 是 | episode `workload_seed`；±20% input jitter。 |
| computation demand | 同一 catalog/抖动规则 | 是 | episode `workload_seed`；±10% cycles-per-bit jitter。 |
| true popularity | 同一 Zipf exponent/演化规则 | 是 | episode `popularity_seed`。 |
| vehicle prediction | 同一模型/预测设置 | 是 | realization trajectory、观测噪声 seed 和 frozen calibration residuals。 |
| popularity prediction | 同一 frozen GRU / history / horizon | 是 | episode 历史请求输入；shared checkpoint。 |
| prediction scenarios | 同一 S/H/config | 是 | episode `scenario_seed` 与 residual/task draws；文件预先生成并冻结。 |
| cache capacities | 是 | 否 | 同一 `system_config.toml` 数值。 |
| CPU capacities | 是 | 否 | 同一 `system_config.toml` 数值。 |
| bandwidth | 是 | 否 | 同一 `system_config.toml` 数值。 |

例如当前配置中的 TV/SV/RSU/UAV cache capacities 分别为 5,581.58 / 16,744.75 / 44,652.67 / 33,489.50 MB；对应 CPU 为 1 / 1 / 5 / 5 Gcycle/s，SV/RSU/UAV bandwidth 均为 10 MHz。配置与 shared node positions 没有按 realization 改写。

## 8. Train / validation / test 比较和历史 51-episode 目录

| 比较项 | 结论 |
|---|---|
| `SAME_GENERATOR` | **YES，按 split manifest 声明的协议**；val/test manifests 与当前 single-episode generator 相符。Training 原始调用没有逐 episode manifest，故不能独立证明每个旧 train episode 的实际脚本版本。 |
| `SAME_SYSTEM_CONFIGURATION` | **YES，按 manifest 声明的 parsed config fingerprint**。Val/test manifests fingerprint 相同；train split manifest 声明共用配置，但 train episode 20001 无 config snapshot。 |
| `SAME_NOMINAL_DISTRIBUTION` | **YES，按 `proposed_joint_nominal` 和共享生成设置声明**；没有发现 split 专属参数差异。Training 历史 provenance 仍有上述记录缺口。 |
| `DIFFERENT_RANDOM_REALIZATIONS` | **YES**；实际 seed ranges 完全不重叠，seed 派生流不同，抽样文件/角色和轨迹也不同。 |

`environment_data/proposed_joint_training_preexisting_51_backup_20260921` 是完整发现的目录名。它有 51 个子目录 `episode_20001`–`episode_20051`；其中 50 个有 episode manifest，`episode_20051` 没有。Windows filesystem timestamps 显示该父目录创建于 2026-09-21 11:18、修改于 12:37；子目录内容时间约从 01:12 延续至 12:37。它的名字和 51 个连续 seed 目录表明它是 training seed 段的备份副本；这是 **INFERENCE**，不是 manifest 中记录的用途。未在 reports/scripts/configs 中找到解释其生成/保留原因的记录。Episode manifests 没有生成器路径；样本 20001 manifest 有 config fingerprint 和 source git SHA `c82f16b76b4698d1a7c2982c3785fcf167f3f0ef`，但未记录实际启动脚本。

正式 runner 明确指向 canonical `environment_data/proposed_joint_training`，未引用 `...preexisting_51_backup_20260921`。所以：`CURRENT_FORMAL_TRAINER_USES_PREEXISTING = NO`（针对这个 51-episode backup）；它会读取 canonical training pool，而 split manifest 将 canonical pool 本身标为 preexisting。**不要把 backup 当作 canonical pool，也不要删除它。**

## 9. Formal trainer 实际消费与随机性

Corrected runner common 设定 pool 为 `environment_data/proposed_joint_training`、seed start 20001/end 20300，并运行 300 epochs；命令传 `--episode-pool` 和 `--episode-seed-start`，没有传 `--generate-missing-episodes` (`stage2b_fresh300_runner_common.py:24-33,306-325`)。Formal corrected wrapper 调用该 common runner。

`trainer.run_training` 对 seed-based pool 先载入 `episode_20001`，每完整一集将 pool index 加一，接着载入 `episode_seed_start + next_pool_index`；每个 episode 跑完 300 个 origin 0–299，最后一个 origin 才终止 (`trainer.py:12528-12579,12850-12905`)。现有目录完整时 `_ensure_episode_realization` 校验并返回 `LOAD`；若缺目录且没有 `--generate-missing-episodes`，就报缺失并停止，不会静默生成/覆盖 (`trainer.py:1330-1382`)。Trainer 读取 realization 文件及预生成 scenarios；实际 task、车辆预测和 GRU 预测不会在 PPO epoch 内重新生成/重跑。

“seed=20001 episode 是否固定”分三项：

- **环境输入：固定。** `tasks.csv`、trajectory、roles、预测文件、scenario draws 和 TV local cache 已保存在 seed 20001 目录，作为该 epoch 的输入。
- **Policy 动作：随机采样。** rollout 调用 `policy.sample(... deterministic=...)`；训练时 actor 使用其动作分布采样。Trainer 会设 Python/NumPy/Torch seed，但 policy action 仍属随机采样过程。
- **PPO minibatch 顺序：随机排列。** update 使用 `torch.randperm(sample_count)`；在给定训练 seed 下可复现，但每次 update 会产生排列。

## 10. 最终生成/消费链与状态

```text
SUMO network + per-episode randomTrips demand
  → SUMO/TraCI trajectory.csv + roles
  → true popularity + per-slot/TV tasks + TV local cache
  → causal vehicle forecasts + frozen-GRU popularity forecasts
  → residual-bank + rho_hat sampled prediction_scenarios.npz
  → episode_<seed>/ (one frozen realization)
  → corrected PPO trainer loads canonical training pool
  → seed 20001 + epoch 1; then 20002…; each realization slots 0–299 sequentially
```

```text
DATA_GENERATOR_IDENTIFIED = YES
TRAIN_GENERATION_TRACE_COMPLETE = NO (pool is preexisting; per-episode source manifest absent)
VALIDATION_GENERATION_TRACE_COMPLETE = YES
TEST_GENERATION_TRACE_COMPLETE = YES
TRAIN_SEEDS = 20001–20300
VALIDATION_SEEDS = 30001–30030
TEST_SEEDS = 40001–40100
SAME_GENERATOR = YES (declared pool-generation protocol; exact historical train invocation unverified)
SAME_SYSTEM_CONFIGURATION = YES (declared parsed config fingerprint; train sample manifest absent)
SAME_NOMINAL_DISTRIBUTION = YES (declared nominal scenario/config; same train provenance caveat)
DIFFERENT_RANDOM_REALIZATIONS = YES
SUMO_REEXECUTED_PER_REALIZATION = YES in the generator path; historical train invocation is not fully recorded
TRAJECTORY_SOURCE = SUMO/TraCI from episode randomTrips route demand on data/sumo/map/osm.net.xml.gz
TASK_GENERATION_SOURCE = task_scenario.build_episode_artifacts / generate_tasks
POPULARITY_GENERATION_SOURCE = task_scenario.make_popularity_trace_with_history
VEHICLE_PREDICTION_SOURCE = scenario_generator.build_vehicle_nominal_predictions / vehicle_prediction.predict_nominal
POPULARITY_PREDICTION_SOURCE = popularity_prediction.main(mode=predict) / run_frozen_inference
PREDICTION_SCENARIO_SOURCE = scenario_generator.main → sample_positions / sample_tasks → prediction_scenarios.npz
FUTURE_TRAJECTORY_LEAKAGE = NO (for future realized state/coordinates; planned route geometry is used)
FUTURE_REQUEST_LEAKAGE = NO (trainer consumes predictions, not co-located truth labels)
CURRENT_FORMAL_TRAINER_USES_PREEXISTING = NO for the named 51-episode backup; YES for canonical preexisting proposed_joint_training
FORMAL_TRAINING_EXECUTED = NO
VALIDATION_EXECUTED = NO
TEST_EXECUTED = NO
ENVIRONMENT_DATA_REGENERATED = NO
```

