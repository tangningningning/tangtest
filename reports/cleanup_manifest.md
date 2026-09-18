# CLEANUP_MANIFEST

审计/清理时间：2026-09-18（Asia/Shanghai）

## 总状态

- CLEANUP = PASS
- LARGE_REDUNDANT_OUTPUTS_REMOVED = YES
- TEMP_FILES_REMOVED = YES
- PROTECTED_POOL_PRESERVED = YES
- V10D_PRESERVED = YES
- FORMAL_CHECKPOINTS_PRESERVED = YES
- GIT_PUSH = NO
- TRAINING_STARTED = NO
- ROLLOUT_STARTED = NO
- EVALUATION_STARTED = NO

## 删除前置检查

### server launcher scripts

四个脚本各自对应一个已完成 300-epoch formal run；manifest/summary 均显示 completed，checkpoint、episode/update logs 均存在。引用审计中没有发现代码或其他运行脚本调用它们，只有本次准备/环境修复旧报告提及。它们是一次性 server launchers，已完成任务且没有继续执行价值：

- scripts/formal/run_v11a_cf_5e3.sh
- scripts/formal/run_v11a_cf_5e4.sh
- scripts/formal/run_v11a_cf_6e3.sh
- scripts/formal/run_v11a_cf_6e4.sh

### old server reports

以下文件只描述 launcher preparation / environment fix，不承载 V11-A mechanism、Q、KL、critic 或 reward 证据；有效的输入/输出/失败证据已被本任务四份 consolidated reports 吸收：

- reports/v11a_four_lr_server_launch_preparation.md
- reports/v11a_four_lr_server_launch_preparation.json
- reports/v11a_server_launcher_env_fix.md
- reports/v11a_server_launcher_env_fix.json

### scratch config

configs/lr_sweep/v9_actor_lr_1e2_kl_protection.toml 的引用审计没有发现 source/script/run manifest/report caller，也没有对应 run output；它与当前 V11-A formal configs 无关，因此删除。被历史 run manifest 引用的 configs/lr_sweep/v9_actor_lr_6e4_kl_protection.toml 保留。

### large raw JSONL

删除的四个 raw file 及其清理前大小：

| path | bytes |
|---|---:|
| runs/ppo_v11a_cache_node_cf_5e3_formal/critic_diagnostics.jsonl | 100,955,715 |
| runs/ppo_v11a_cache_node_cf_5e3_formal/slot_metrics.jsonl | 23,799,620 |
| runs/ppo_v11a_cache_node_cf_6e3_formal/critic_diagnostics.jsonl | 100,455,773 |
| runs/ppo_v11a_cache_node_cf_6e3_formal/slot_metrics.jsonl | 23,869,925 |
| 合计 | 249,081,033 |

删除理由不是“文件大”本身：5e-3/6e-3 的 summary、episode_metrics、update_metrics、latest/best checkpoint、training plot 均保留，且 v11a_failure_and_io_analysis.md/.json 已保存本次 failure 所需的 return、critic/Q、CF、KL、operation、UAV mask 摘要。5e-4/6e-4 的同类 raw JSONL 全部保留，作为稳定对照和 6e-4 failure witness。

## 保留的关键资产

- runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911
- runs/ppo_v10d_decoupled_encoder_branch_kl_6e4_formal
- runs/ppo_v11a_cache_node_cf_6e4_gate
- 四个 V11-A formal run 的核心 checkpoint、manifest、summary、episode/update metrics
- 四个 configs/proposed/system_config_ppo_v11a_cache_node_cf_*.toml
- V10C/V10D/V11A mechanism and gate reports
- scripts/analyze_ppo_cross_run_root_cause.py、scripts/plot_*.py、src/ppo_simple/state_trace.py 等仍有分析/实现引用的脚本

## 不删除的理由

- 不删除 frozen pool：它是四个 formal run 的共同数据协议与可复核输入。
- 不删除 V10-D：它是当前 V11-A credit/branch 对照的正式上游结果。
- 不删除四个 V11-A configs：它们被 formal run 与 output inventory 明确引用。
- 不删除 mechanism reports：本次 consolidated report 没有替代其中的实现 gate、critic readiness 或 KL decomposition 证据。
- 不删除 5e-4/6e-4 raw diagnostics：它们仍是当前主对照与 failure analysis 的可追溯原始证据。

## 最终字段

- FILES_DELETED = 13
- LARGE_REDUNDANT_OUTPUTS_REMOVED = YES
- TEMP_FILES_REMOVED = YES
- PROTECTED_POOL_PRESERVED = YES
- V10D_PRESERVED = YES
- FORMAL_CHECKPOINTS_PRESERVED = YES
- GIT_DIFF_CHECK = PASS
- GIT_PUSH = NO
