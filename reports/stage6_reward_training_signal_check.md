# Stage6第七环节：Reward与PPO训练信号检查

## 范围与可复现依据

- GitHub `tangningningning/tangtest` 的 `main` 与本地 `main` 均指向 commit [`1fe65bf36e0d5b4d34dd29a35a8038336c9c604f`](https://github.com/tangningningning/tangtest/commit/1fe65bf36e0d5b4d34dd29a35a8038336c9c604f)。
- 正式配置 SHA-256：`684bdfa4dc200db6227b2b00a7718797a98bf658e54d2d4dbef6f7d1f5babf77`；与 `runs/stage6_klguard_lr_5e_minus_4/run_manifest.json` 一致。
- 只复用该正式Training运行的 `episode_metrics.jsonl`、`slot_metrics.jsonl`、`update_metrics.jsonl` 和现有Stage6报告：300 Episode、90,000 Slot、75次策略更新；每次更新前使用4个 realization。没有读取Validation/Test，没有新训练、回放或Matching/KKT。
- 按要求尝试 `conda activate adaptive_cache_mec`，PowerShell返回“系统无法执行指定程序”。只用现有 `C:\software\anaconda3\python.exe`（Python 3.12.7）只读汇总已有日志，没有安装依赖。

## 1. Reward公式、权重与实际尺度

### 运行时公式

对Slot内实际执行的每个任务，当前代码计算：

```text
u_i = 0.6 × (1 − delay_i / 30.6289851662)
    + 0.4 × (1 − vehicle_energy_i / 3.06289851662)
Q_t = Σ_i u_i
G_t = Q_t − C_cache,t − C_move,t
```

`T_ref=30.6289851662 s`、`E_ref=3.06289851662 J` 是Training-only本地执行工作负载P95校准值。Reward用实际Matching/KKT产生的任务结果；预测配置效用不进入Reward。任务效用没有截断到 `[0,1]`，因此超出参考时延/能耗的单个任务可以贡献负效用；这属于所定义的线性效用模型。

实际成本函数还包含显式的全局成本权重：

```text
C_cache = cache_cost_weight × Σ_m [eta_ca,m × z_m + lambda_cache × xi_ca,m × loaded_MB_m]
C_move  = uav_movement_cost_weight × Σ_u [eta_move × z_u + lambda_movement × xi_move × distance_u]
```

正式值为 `omega_delay=0.6`、`omega_energy=0.4`、`lambda_cache=0.1`、`cache_cost_weight=0.1`、`lambda_movement=1.76e-5`、`uav_movement_cost_weight=0.50`、`xi_move=15 J/m`、`eta_move=0.00176`。因此缓存固定项整体乘0.1，加载项有效系数含 `0.1 × 0.1`；UAV固定项整体乘0.5，距离能耗项也再乘0.5。`system_config.toml`中的成本简式没有写出这两个外层weight，文字公式不完整；这是权重尺度的文档/意图核对点。代码和配置均明确包含这些weight，但现有证据不能判定外层weight是否符合作者原本想要的经济权衡。

### 300 Episode中的Reward分解

| 组成 | 300 Episode合计 | 每Episode均值 | 占Task Utility |
|---|---:|---:|---:|
| 实际Task Utility `Q` | 1,199,539.882 | 3,998.466 | 100% |
| Cache成本 | 1,268.288 | 4.228 | 0.1057% |
| UAV移动成本 | 162.057 | 0.540 | 0.0135% |
| 系统Return `G` | 1,198,109.537 | 3,993.698 | 99.8808% |

90,000个Slot全部为正Reward；最小Slot Reward `3.918304`，最大 `19.049567`。因此这份正式Training记录里**没有大幅负Reward**。若只比较两项扣减，Cache成本约为UAV成本的7.83倍，是较大的扣减项；但它仍只占Task Utility约0.106%，UAV成本约占0.014%。不能把它们称为本次Return波动或负Reward的来源。

汇总共 `1,260,159` 个任务：平均每任务时延 `2.297360 s`、车辆能耗 `0.023743 J`、Task Utility `0.951896`。按公式复算，平均每任务时延项扣减 `0.045004`，能耗项扣减 `0.003101`，复算的总Task Utility与日志差仅 `2.3e-10`。本运行中时延项的实际扣减约为能耗项的14.5倍；虽然名义权重是0.6/0.4，车辆能耗远低于 `E_ref`，所以能耗对当前样本的Reward变化贡献较弱。两项参考值都是本地P95基准，量纲归一化一致；但实际卸载任务时延、车辆发射能耗远低于这些参考值，使每任务效用聚集在约0.95附近。其经济权衡属于实验设计选择，不能仅凭数学归一化断言权重“合理”或“错误”。

### 重复扣减与符号核对

- `slot_environment.execute_slot`构成 `G=Q−C_cache−C_move`；`rollout._check_reward`核对后直接把这个 `G` 作为Slot reward。
- 90,000条Slot记录逐条满足 `G = task_utility − cache_cost − movement_cost`，最大绝对残差为0；300条Episode记录的Return也与三项合计一致，最大残差约 `5.5e-12`。
- 因此未发现成本在Reward组装中被重复扣减或符号取反。Macro与Micro各自用同一环境Reward训练不同时间尺度策略，不是对同一个Return再次扣成本。

## 2. Episode Return波动：任务量与策略更新混杂

| 指标 | 300 Episode统计 |
|---|---:|
| 每Episode任务数 | 均值 `4,200.53`，SD `38.56`，范围 `4,079–4,295` |
| 每Episode Return | 均值 `3,993.70`，SD `38.32`，范围 `3,886.30–4,096.79` |
| Return与任务数相关 | Pearson `r=0.9057` |
| Return与Task Utility总和相关 | Pearson `r=0.9978` |
| Return/任务数 | 均值 `0.9508`，范围 `0.9385–0.9577` |

训练每4个不同Training realization收集一次数据，再更新一次策略。同一个更新组内策略固定，但不同Episode任务序列不同；组内Episode Return标准差平均为 `30.34`，任务数标准差平均为 `30.02`。跨组比较时，任务realization和策略都发生变化。因此相邻Epoch/Episode Return变化不能单独证明策略改善或退化。Return几乎跟着Task Utility总量变化，而后者很大程度受任务数量影响；每任务归一化减少了任务量的影响，但不同Episode仍不是相同任务下的配对比较。

现有日志没有把同一个Training Episode固定后分别用更新前/后的策略评估，也没有对应的配对策略结果。本任务不新增回放，故**当前原始Training Return不能单独作为策略改善的因果证据**；它仍是策略与Episode任务共同决定的训练轨迹指标。

## 3. Slot Reward如何进入PPO

- **Micro（UAV，每Slot）**：Micro transition直接接收当前Slot `r_t=G_t`。其GAE逐Slot使用 `delta_t = r_t + gamma × (1−done) × V_{t+1} − V_t`，之后按 `gamma × gae_lambda`递推。
- **Macro（Cache/Holding，每个Cache frame）**：长度为 `D` 的frame先累计 `R_k = Σ_{j=0}^{D−1} gamma^j r_{t+j}`；SMDP TD折扣为 `gamma^D`，GAE trace因子为 `(gamma × gae_lambda)^D`。正式值 `gamma=1.0`、`gae_lambda=0.95`，所以Holding内Reward不折扣、跨Macro frame的trace按 `0.95^D`衰减（D=1至5时为0.95至0.773781）。达到Holding终止只结束当前Macro frame，不截断Episode回报；真实Episode终止时 `done` 屏蔽下一状态bootstrap。
- 正式运行的 `episode_return`是逐Slot Reward之和。由于 `gamma=1`，它与Macro frame累计Reward的总和一致，不存在因Holding聚合导致的Reward丢失。
- Long-Value Critic另以完整realized return-to-go为目标；既有Stage6代码路径检查确认它不参与Macro Actor优势计算。Macro Actor使用Macro Critic的SMDP-GAE优势。**不能把Long-Value或Macro Critic的状态价值 `V(s)`当成某个Cache动作的 `Q(s,a)`。**
- 既有Stage6结论保留：Macro Cache选择和Holding Termination共享同一frame优势及joint PPO ratio；这会让Cache信用与Holding信用耦合，是学习机制上的风险，但不是Reward扣减错误。SMDP-GAE提供了跨Holding的回报传播路径，不保证Critic已准确估计长期Cache动作价值，也不保证Actor会把该价值排好序。

代码路径见 `src/common/system_model.py:218-388`、`src/decision/slot_environment.py:590-610`、`src/ppo_simple/rollout.py:997,1119,1480-1562`、`src/ppo_simple/trainer.py:1274,1505`；此前核对结论见 `reports/stage6_macro_ppo_learning_diagnosis.md`及`reports/stage6_macro_long_term_signal.md`。

## 结论分类

- **Reward实现错误：未发现。** 实际奖励使用实际Task Utility减去一次Cache成本和一次UAV成本，符号和累计账目与日志一致。
- **尺度/建模假设：** Task Utility按任务求和，导致Episode Return与请求量高度相关；T/E使用Training-only本地P95，实际样本的能耗项较弱、每任务效用接近1且无截断。外层 `cache_cost_weight`、`uav_movement_cost_weight`会进一步缩放成本，配置公式注释没有完整表达这一层，权重原意未能从现有证据确认。
- **PPO学习机制：** gamma和Holding长度的SMDP累计口径一致；但Return不是固定任务上的策略评价，不能单独判断策略进步。Cache与Holding共享Macro优势的信用分配风险沿用既有Stage6判断；Long-Value状态价值也不能直接作为Cache动作价值。
- **直接回答：** 当前日志中不存在负Reward需要归因；若问哪项扣减更大，是Cache成本，但它不是大幅Return波动的主因。当前Episode Return不能独立证明策略改善。上述尺度影响主要来自系统Reward定义/实验权重，动作排序与长期信用是否学到则属于PPO学习机制问题。
