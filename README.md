# adaptive_cache_mec

## 正式算法

正式训练配置为 `configs/proposed/system_config.toml`。当前策略采用双时间尺度 PPO：Macro 在 Cache 决策边界联合选择 Cache 配置与持续时长；每个 slot 的 Micro 策略按固定 UAV ID 顺序选择 UAV0 位置，再在 UAV0 条件下选择 UAV1 位置。Cache candidate 0 表示保持当前配置，UAV candidate 0 表示保持当前位置。

训练按完整 episode 收集两份独立 rollout。episode 结束后先更新 Micro PPO，再对 Macro frames 做 full-batch SMDP PPO。正式训练不调用 Exact-Q teacher，也不要求 MC warmup；Transfer-Exchange Matching、KKT、预测 artifact 和现有物理模型保持原样。

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

## 评价

冻结 checkpoint 的正式评价默认使用 deterministic action selection，并执行真实 Matching/KKT：

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
