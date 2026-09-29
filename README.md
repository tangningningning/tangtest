# adaptive_cache_mec

## 当前算法

正式策略为 `adaptive_joint_cache_uav_ppo`，由 PPO 联合控制 Cache 与 UAV 配置。

## 正式配置

唯一正式配置为 `configs/proposed/system_config.toml`。

## 数据生成

正式 episode 生成入口为：

- `scripts/generation/generate_episode_realization.py`：生成一个完整 episode realization。
- `scripts/generation/generate_episode_pool.py`：按指定 seed 范围生成 episode pool。

```powershell
python scripts/generation/generate_episode_realization.py --help
python scripts/generation/generate_episode_pool.py --help
```

## 训练

训练统一使用 `ppo-train`，并通过正式配置和已有 episode pool 提供输入：

```powershell
ppo-train --config configs/proposed/system_config.toml `
  --epochs <epochs> --run-dir runs/<new-run> `
  --episode-pool environment_data/<episode-pool>
```

## 评价

冻结 checkpoint 的正式评价统一使用 `ppo-eval`：

```powershell
ppo-eval --config configs/proposed/system_config.toml `
  --checkpoint runs/<training-run>/checkpoints/<checkpoint>.pt `
  --run-dir runs/<evaluation-run> `
  --episode-pool environment_data/<episode-pool>
```

## 主流程

```text
固定 SUMO 网络
→ episode realization 与车辆、任务、流行度预测产物
→ proposed_joint PPO rollout
→ Cache Trigger → Cache Candidate
→ UAV Operation → UAV Candidate
→ Matching → KKT → reward
→ Critic / PPO update → checkpoint → 正式 evaluation
```
