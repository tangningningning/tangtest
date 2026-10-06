# 最小策略概率分布检查

## 现有正式 Gate

- 运行目录：`C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\formal_unified_cache_cardinality_corrected_gate`；run config：`C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\formal_unified_cache_cardinality_corrected_gate\run_config.toml`。
- `15` / `15` episodes，seed `27005`；最终 checkpoint：`True`。未启动训练或 optimizer。
- 新冻结 evaluation：确定性 final-policy 轨迹 1 个固定 episode；初始 V14b+fresh Candidate 与最终 checkpoint 在同一批 states/candidate sets 上直接 evaluate 概率，不比较抽样动作。

## KEEP / UPDATE 分布（训练 Gate 的逐 Macro frame 日志）

| 阶段 | raw P_KEEP | corrected P_KEEP | corrected P_UPDATE_GROUP | GROUP_MARGIN (update−keep) | sampled KEEP rate |
|---|---:|---:|---:|---:|---:|
| first_5 | 0.002936 | 0.507403 | 0.492597 | -0.014807 | 0.472549 |
| middle_5 | 0.002984 | 0.508364 | 0.491636 | -0.016729 | 0.519192 |
| last_5 | 0.003873 | 0.567803 | 0.432197 | -0.135606 | 0.563218 |
| all_15 | 0.003272 | 0.528362 | 0.471638 | -0.056724 | 0.518664 |

- P_KEEP state std：15-episode 全部 Macro states `0.055508`；episode 内 state std 均值 `0.037137`。
- 同一固定轨迹上的 initial/final P_KEEP state std 为 `0.000185/0.002758`（GROUP_MARGIN std `0.000370/0.005516`）：最终策略在这条确定性轨迹上有可检测但偏弱的变化；相比 15-episode 状态集更窄，记为后续状态依赖候选观察，不据此判全局无状态依赖。
- GROUP_MARGIN state std：全部 states `0.111017`；episode 内 state std 均值 `0.074274`。
- Expected KEEP `806.81` / sampled KEEP `792`；expected rate `0.528362` / sampled rate `0.518664`；residual `-14.81`（约 `-0.76` Bernoulli SD）。
- `GROUP_MARGIN` 按 `P_UPDATE_GROUP − P_KEEP` 计算；未使用旧的 best individual UPDATE candidate margin 判定组偏好。
- raw P_KEEP 在旧 flat softmax / 当前 raw logits 下仅保存每 episode 均值，分期值为按 Macro frame 加权均值；逐 state raw 向量未持久化。corrected P_KEEP 有逐 Macro frame 记录。
- 历史 gate 未保存完整 Cache candidate、Duration、Micro 概率向量；最终 checkpoint 的这些完整分布来自固定 episode_20001 冻结评估。历史 Duration/UAV 实际选择率与固定状态的概率汇总分开呈现，固定评估的 argmax 率不作为随机 sampled rate。

## UPDATE 内部、Duration 与 UAV

- 历史 UPDATE 条件熵均值 `5.7459`；normalized entropy `0.9804`；effective candidate number `314.0 / 351`；top1 conditional probability `0.00965`。
- Final frozen evaluation 的 Duration mean probability：`1=0.3562, 3=0.3329, 5=0.3109`；对应 per-state probability std 与选择率见 JSON。
- Final frozen evaluation 的 UAV0 P_STAY/P_MOVE：`0.1150/0.8850`；UAV1：`0.3490/0.6510`。15-episode sampled stay/move rates 与固定 eval 分开列出于 JSON。
- 描述性对照：历史 15 episode Duration 1/3/5 sampled rates 为 `0.3392/0.3399/0.3209`，最终冻结概率均值为 `0.3562/0.3329/0.3109`；UAV0 历史/冻结 MOVE 为 `0.7651/0.8850`（后 5 episode sampled MOVE `0.8247`），UAV1 为 `0.6756/0.6510`。训练期与最终冻结状态不成对，不能据此视为同状态采样拟合检验。
- Initial-vs-final 是同一条 final deterministic trajectory 上的逐 state 概率比较；只比较策略分布，不将轨迹动作或 deterministic argmax rate 当作 sampled policy probability。

## 判定

- Cache group collapse：`False`；UPDATE candidate collapse：`False`；Duration collapse：`False`；UAV group/candidate collapse：`False`。
- 概率 NaN/Inf：`0/0`；最大归一化误差 `3.6e-07`；最大 sampler/evaluator 分布差 `0`。
- 第四层状态：**通过**；第一处明确概率异常：`无`。

下一步：停止，等待进入第五层训练监控检查

逐状态概率向量、mask、entropy、top-mass、sampler equivalence 与每个训练 episode 的统计保存在同目录 JSON。
