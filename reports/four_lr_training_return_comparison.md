# Four-LR Training Return Comparison

## Data sources

- **LR 1e-4**：`runs/formal_cache_economic_v2_lr_1e_minus_4/episode_metrics.jsonl`、`runs/formal_cache_economic_v2_lr_1e_minus_4/run_manifest.json`、`runs/formal_cache_economic_v2_lr_1e_minus_4/run_summary.json`。
- **LR 5e-4**：`runs/formal_cache_economic_v2_lr_5e_minus_4/episode_metrics.jsonl`、`runs/formal_cache_economic_v2_lr_5e_minus_4/run_manifest.json`、`runs/formal_cache_economic_v2_lr_5e_minus_4/run_summary.json`。
- **LR 1e-3**：`runs/formal_cache_economic_v2_lr_1e_minus_3/episode_metrics.jsonl`、`runs/formal_cache_economic_v2_lr_1e_minus_3/run_manifest.json`、`runs/formal_cache_economic_v2_lr_1e_minus_3/run_summary.json`。
- **LR 5e-3**：`runs/formal_cache_economic_v2_lr_5e_minus_3/episode_metrics.jsonl`、`runs/formal_cache_economic_v2_lr_5e_minus_3/run_manifest.json`、`runs/formal_cache_economic_v2_lr_5e_minus_3/run_summary.json`。

读取 `episode_metrics.jsonl` 中实际字段 `episode` 与 `episode_return`；学习率由对应 `run_manifest.json` 的 `macro_actor_learning_rate` 核对，完成轮数由 `run_summary.json` 的 `completed_episodes` 与日志行数交叉核对。

## Summary

| Macro Actor LR | Completed Episodes | Last 20 Mean Return | Last 100 Mean Return | Highest Single-Episode Return | Completed 300? |
|---|---:|---:|---:|---:|:---:|
| 1e-4 | 300 | 824.012 | 590.451 | 2253.698（Episode 271） | 是 |
| 5e-4 | 300 | 3127.474 | 3085.558 | 3560.652（Episode 278） | 是 |
| 1e-3 | 300 | 2935.628 | 2927.341 | 3493.793（Episode 289） | 是 |
| 5e-3 | 300 | 2663.131 | 2791.157 | 3472.651（Episode 62） | 是 |

四组均完成300 Episodes。
四组 Episode 编号均从1连续到各自最后一轮。

## Plot method and interpretation

同一坐标系绘制四组训练Episode Return：细线为原始逐Episode数据（半透明），粗线为20-episode trailing moving average（每个窗口含当前Episode及之前19轮；第20轮开始有平滑值）。图像横轴固定为Episode 1–300。

不同Episode对应不同环境 realization，因此曲线包含环境差异与学习过程的共同影响；移动平均用于展示训练轨迹的平滑趋势，不能单独作为同环境策略比较或严格收敛证明。

图片：`reports/figures/four_lr_training_return_comparison.png`  
PDF：`reports/figures/four_lr_training_return_comparison.pdf`
