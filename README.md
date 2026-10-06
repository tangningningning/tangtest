# adaptive_cache_mec

## 正式算法

正式算法唯一名称为 `PROPOSED`，训练配置为 `configs/proposed/system_config.toml`。Cache 使用单一合法候选分布，candidate 0 表示 KEEP，其余候选为 UPDATE；UPDATE logits 按合法非零候选数做 cardinality correction。Macro 在 Cache 决策边界联合选择候选配置与持续时长，并以 Cache 与 Duration 的联合概率比率执行一个 PPO clipped surrogate。每个 slot 的 Micro 策略按固定 UAV ID 顺序选择 UAV0 位置，再在 UAV0 条件下选择 UAV1 位置，并以两 UAV 的联合概率比率执行一个 PPO clipped surrogate；UAV candidate 0 表示保持当前位置。

正式训练每次使用一个完整 episode realization，主线 LR 实验共使用 300 份 realization，seed 为 20001–20300。episode rollout 完成后更新 Micro PPO 和 Macro SMDP PPO。正式训练使用当前 Matching、KKT、prediction artifacts 和物理模型。

长期数学定义与代码映射见 [`adaptive_dual_timescale_formula_updates.md`](adaptive_dual_timescale_formula_updates.md)。

## 数据生成

正式 episode 生成入口为：

- `scripts/generation/generate_episode_realization.py`：生成一个完整 episode realization。
- `scripts/generation/generate_episode_pool.py`：按指定 seed 范围生成 episode pool。

```powershell
python scripts/generation/generate_episode_realization.py --help
python scripts/generation/generate_episode_pool.py --help
```

## 训练

训练使用 `ppo-train` 和已有 episode pool。每个 PPO episode 对应一个完整 realization：

```powershell
ppo-train --config configs/proposed/system_config.toml `
  --epochs <episodes> --run-dir runs/<new-run> `
  --episode-pool environment_data/<episode-pool>
```

可用 `--resume` 从该 run 的最新 checkpoint 继续。缺失 realization 默认会报错；需要按 seed 补生成时，必须显式指定 `--episode-seed-start` 和 `--generate-missing-episodes`。

`checkpoints/latest_checkpoint.pt` 是主线的恢复与正式评价 checkpoint。`best_train_return_checkpoint.pt` 按单个训练 episode 的 raw return 保存，仅作训练诊断；正式 evaluator 会拒绝该文件，不会把它当作最终模型选择结果。四个 Macro Actor 学习率实验使用 `scripts/training/run_proposed_macro_actor_lr_*.py`，共用 `scripts/training/mainline_runner.py`，读取 episode seed 20001–20300 的 300 份 realization。

## 评价

`ppo-eval` 校验 checkpoint 格式、架构、动作语义及配置合同，并拒绝 training-return diagnostic checkpoint。评价使用冻结策略，默认采用 deterministic action selection。Validation 和 Test 使用独立 episode pool，不要求与训练 pool 相同，并执行真实 Matching/KKT：

```powershell
ppo-eval --config configs/proposed/system_config.toml `
  --checkpoint runs/<training-run>/checkpoints/latest_checkpoint.pt `
  --run-dir runs/<evaluation-run> `
  --episode-pool environment_data/<episode-pool> `
  --seed-start <first-seed> --episodes <count>
```

## 主流程

```text
固定 SUMO 网络与 prediction artifacts
→ Macro Cache candidate + duration SMDP PPO
→ 在 Cache frame 内保持 Cache 配置
→ 每 slot 条件式、group-balanced masked UAV PPO
→ Transfer-Exchange Matching → KKT → 真实 G(t)
→ 独立 Micro/Macro rollout 与 episode-end PPO updates
→ checkpoint → frozen-policy evaluation
```
