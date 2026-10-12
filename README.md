# adaptive_cache_mec

## 正式算法

正式算法唯一名称为 `PROPOSED`，训练配置为 `configs/proposed/system_config.toml`。Cache 使用单一合法候选分布，candidate 0 表示 KEEP，其余候选为 UPDATE；UPDATE logits 按合法非零候选数做 cardinality correction。每个 Macro frame 起点只选择一个 Unified Cache candidate；执行每个 slot 后，termination policy 根据真实 post-slot observation 与 elapsed holding fraction 采样 CONTINUE 或 TERMINATE。holding bounds 固定为 1–5 slots，因此实际 duration 可为 1、2、3、4 或 5。Macro PPO 将 Cache candidate log probability 与该 frame 所有已采样 termination log probability 求和，只使用一个 joint ratio 和一个 clipped surrogate；强制 max holding 与 episode terminal 不进入 policy likelihood。这不是独立 Trigger PPO。每个 slot 的 Micro 策略按固定 UAV ID 顺序选择 UAV0 位置，再在 UAV0 条件下选择 UAV1 位置；两 UAV 仍共用一个联合概率比率和一个 PPO clipped surrogate。另加 STAY/MOVE 组熵底限正则，只作用于 Micro Actor；它是软惩罚，不裁剪或硬限制策略概率，也不改变随机训练采样。UAV candidate 0 表示保持当前位置。

对每个 UAV，`p_stay` 是 masked categorical 中候选 0 的概率，`p_move` 是所有非零候选概率之和。二元组熵为 `H_group = -p_stay log(p_stay) - p_move log(p_move)`。配置的概率软上限为 0.90，由此推导熵底限 `H_floor = -0.9 log(0.9) - 0.1 log(0.1)`；低于底限时才增加 `0.05 × mean(ReLU(H_floor - H_group))` Actor loss。该组熵描述 STAY 与 MOVE 的选择平衡；现有 `micro_candidate_entropy`/`micro_entropy` 仍表示完整候选分布的熵，两者含义不同。Micro PPO 的 target KL 为 0.02，Macro PPO 的 target KL 为 0.10。

正式训练以 4 个独立 episode realization 为一个 on-policy update group；组内四条 rollout 都由同一冻结策略采样。每条 realization 单独计算 Micro/Macro GAE、Macro N2 target 和 realized return-to-go，再合并 transitions，并在整个 group 上统一标准化 Macro 与 Micro advantage。每组执行一次 Micro/Macro PPO 更新。300 个训练 realization（seed 20001–20300）对应 75 次 policy update。Macro Actor 学习率按 policy update 使用 cosine schedule，从 5e-4 衰减到 1e-4；Micro Actor 保持 3e-4。训练动作保持 stochastic；评价按既有 group-aware deterministic 规则选择动作。

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

训练使用 `ppo-train` 和完整 episode pool。`--epochs` 必须是 4 的倍数；checkpoint 与 resume 只允许在完整 update group 边界。Stage6正式对照使用独立配置和训练池；该池复用既有 Training episode 的轨迹、角色、预测、流行度、到达与服务ID，只将任务输入、CPU cycles 和场景负载预测按Stage6服务目录同步到500–2000 KB基准范围：

其配置为 `configs/proposed/system_config.toml`，数据池为 `environment_data/stage6_candidate_resource_reduced_training`；数据准备复用既有池，不重新运行300次完整环境生成。

数据池准备后，四个入口使用相同配置、Training Episode、PPO随机种子和其他算法参数，仅Macro Actor学习率不同：

```powershell
python scripts/training/run_stage6_lr_1e_minus_4.py
python scripts/training/run_stage6_lr_5e_minus_4.py
python scripts/training/run_stage6_lr_1e_minus_3.py
python scripts/training/run_stage6_lr_5e_minus_3.py
```

四个入口通过 `mainline_runner.py` 调用同一个 `ppo_simple.trainer`，分别写入对应的 `runs/stage6_lr_*`。共享Runner拒绝覆盖已有Run，并在启动前核验Stage6配置、训练池及20001–20300的Train划分。Macro Actor学习率通过Trainer的CLI覆盖参数设置；Stage6采用constant调度，Trainer会将覆盖值同时用于初始和最终学习率，因此每次更新都应用同一个指定值。其他优化器仍使用Stage6配置中的学习率。

`checkpoints/latest_checkpoint.pt` 是主线的恢复与正式评价 checkpoint。`best_train_return_checkpoint.pt` 按单个训练 episode 的 raw return 保存，仅作训练诊断；正式 evaluator 会拒绝该文件，不会把它当作最终模型选择结果。历史旧协议结果保留在既有 Run 与报告中；当前 `scripts/training/` 只保留Stage6四个正式入口和共享Runner。

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
→ 四个 realization 在同一冻结策略下独立 rollout
→ Transfer-Exchange Matching → KKT → 真实 G(t)
→ episode-local target 与 GAE，再合并和统一 advantage normalization
→ 一次 Macro joint PPO 与 Micro joint PPO update
→ 每组一条 update record、每 episode 一条 episode record
→ group-boundary checkpoint → frozen-policy evaluation
```
