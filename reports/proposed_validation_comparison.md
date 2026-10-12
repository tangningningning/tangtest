# Proposed 冻结策略统一 Validation 对比

日期：2026-10-09  
状态：**同进度正式比较未执行；仅完成准备与准入检查。**

## Checkpoint 进度与完整性

使用各正式 run 的 `latest_checkpoint.pt`，以 checkpoint 内 training counters 为准，并核对对应日志末条 Episode。正式 evaluator 的元数据校验和 `PPOTrainer.load_checkpoint()` 均成功；Checkpoint/config hashes 匹配，Checkpoint SHA 与此前只读诊断一致。

| Macro Actor LR | Completed Episodes / Updates | Checkpoint / Architecture | Config raw SHA-256 | Config semantic SHA-256 | Checkpoint SHA-256 | 准入 |
|---|---:|---:|---|---|---|---|
| 5e-4 | 260 / 65 | 58 / 31 | `fe5d4f7bc57a424911166bcfe921c2ec5a7627b0a6aa759c4993f7f488a3775c` | `64e367ee5c67cbcc86b7741ef0e97fe0de1fecd6b3c07039f77a29c1b17ae45e` | `f0d1f675db1820702b3a3a587f6202cb9d890df13bb703fa353bafd10731e3ba` | **未达 300 Episode** |
| 1e-3 | 300 / 75 | 58 / 31 | `fe5d4f7bc57a424911166bcfe921c2ec5a7627b0a6aa759c4993f7f488a3775c` | `64e367ee5c67cbcc86b7741ef0e97fe0de1fecd6b3c07039f77a29c1b17ae45e` | `e6ed46f593f724ff08c89106376ed33aba5f476e6d79e24e5a2613323d379b2e` | 已达 300 Episode |

两份 Checkpoint 都属于 evaluator 支持的 schema：`adaptive_dual_timescale_proposed_cache_economic_features_fixed_lr_horizon_mask_v2`。它们使用相同正式 config hash，Macro Actor LR 由 Checkpoint contract 固定为各自的 5e-4 / 1e-3。5e-4 在本轮最初检查时为 224 Episodes，校准 runner 加载时已更新到 260；表中记录最终复核时仍为 260 的 latest 快照。该 260-Episode 模型用于独立 Critic 校准，但不代表 300-Episode 结果，也未用于同进度 Return 比较。本报告没有启动训练。

## Validation 池与既有结果

- `environment_data/proposed_joint_validation/episode_30001`–`episode_30030` 共 30 个完整 Validation realization 均存在。
- 当前 `runs` 下没有这两个正式 Checkpoint 对应的 30-seed evaluator 输出目录；已有其他项目阶段或不同 seed 的冻结评估不作为本次结果复用。
- 因 5e-4 尚未完成 300 Episode，本次不对 30001–30030 启动确定性环境执行，不生成 Return 或配对 delta 表。
- Final Test seeds 40001–40100 未访问、保留。

## 正式 evaluator 能力与已准备入口

项目 `src/ppo_simple/evaluator.py` 的 `evaluate_pool()` 已支持各 Checkpoint 加载、deterministic 完整 Episode、Validation episode pool、逐 realization JSON 和 aggregate JSON。已有统计覆盖 Return、任务效用、任务时延、车辆能耗、Cache 成本、UAV 移动成本、Cache 更新率、Holding duration 分布、UAV0/UAV1 MOVE 率。

正式 evaluator 之前没有四种 UAV 联合动作比例。本次只在 evaluator 的 episode-summary telemetry 增加由正式 Micro transitions 直接计数的 `joint_action_counts/rates`（STAY/STAY、STAY/MOVE、MOVE/STAY、MOVE/MOVE）；不改变策略选择、状态转移、Reward 或训练逻辑。该字段尚未用于任何正式验证结果。

等 5e-4 latest Checkpoint 自身也显示 300 Episodes 后，分别对两个策略运行同一命令模板。请使用新的空 `--run-dir`，避免触碰训练目录；两个策略各运行一次，`seed_start=30001, episodes=30, repeats=1`：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m ppo_simple.evaluator `
  --config runs/formal_cache_economic_v2_lr_5e_minus_4/run_config.toml `
  --checkpoint runs/formal_cache_economic_v2_lr_5e_minus_4/checkpoints/latest_checkpoint.pt `
  --run-dir runs/proposed_validation_lr_5e-4_300ep `
  --episode-pool environment_data/proposed_joint_validation `
  --seed-start 30001 --episodes 30 --repeats 1 `
  --policy-action-mode deterministic

python -m ppo_simple.evaluator `
  --config runs/formal_cache_economic_v2_lr_1e_minus_3/run_config.toml `
  --checkpoint runs/formal_cache_economic_v2_lr_1e_minus_3/checkpoints/latest_checkpoint.pt `
  --run-dir runs/proposed_validation_lr_1e-3_300ep `
  --episode-pool environment_data/proposed_joint_validation `
  --seed-start 30001 --episodes 30 --repeats 1 `
  --policy-action-mode deterministic
```

输出分别写入两个新增的非训练目录。配对报告应按 seed 30001–30030 对齐两个 `proposed_repeat_summaries.json`，报告逐 seed 差值及平均/离散分布；不得把 30 个 seed 内的 repeats 当独立 realization。

## 配对比较应报告的字段

| 指标 | evaluator 字段/来源 |
|---|---|
| Return、任务效用 | `total_return`、`total_q` |
| 时延、车辆能耗 | `mean_task_delay_s`、`mean_vehicle_energy_j`（均按实际任务数计算） |
| Cache 成本、UAV 移动成本 | `total_cache_cost`、`total_movement_cost` |
| Cache 更新率 | `actual_cache_update_rate` / `cache_change_rate` |
| Holding 分布 | `holding_duration_distribution` |
| UAV0/UAV1 MOVE 比例 | `uav0_move_rate`、`uav1_move_rate` |
| 四种联合动作比例 | 新增 `joint_action_rates` |

本轮没有可报告的性能均值、配对差值或泛化排名。比较准入条件是 5e-4 **后续实际 latest Checkpoint** 达到 300 Episodes，并重新核验其版本、配置哈希和文件 SHA；在此之前只保留以上 evaluator 入口。
