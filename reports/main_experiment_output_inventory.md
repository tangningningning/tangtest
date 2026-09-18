# MAIN_EXPERIMENT_OUTPUT_INVENTORY

这是清理完成后的正式输出清单。审计与压缩只基于已有文件；没有新建 run、没有重跑 rollout/evaluation。

## 保护边界

以下内容明确保留：

- frozen episode pool：runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911
- V10-D formal run：runs/ppo_v10d_decoupled_encoder_branch_kl_6e4_formal
- V11-A 5e-4 与 6e-4 的完整逐槽/逐 transition 诊断
- 四个 V11-A formal run 的 manifest、summary、episode metrics、update metrics、latest/best checkpoint 与小型 plot
- V11-A mechanism / critic-readiness / KL reports，以及现有 root-cause evidence

## formal run inventory

| actor LR | run directory | 保留 | 删除 | 清理后 direct/nested files | 清理后 bytes |
|---:|---|---|---|---:|---:|
| 5e-4 | ppo_v11a_cache_node_cf_5e4_formal | 全部 9 个文件 | 无 | 9 | 149,920,413 |
| 6e-4 | ppo_v11a_cache_node_cf_6e4_formal | 全部 9 个文件 | 无 | 9 | 150,898,672 |
| 5e-3 | ppo_v11a_cache_node_cf_5e3_formal | manifest、summary、episode_metrics、update_metrics、latest/best checkpoint、training plot | critic_diagnostics、slot_metrics | 7 | 26,042,565 |
| 6e-3 | ppo_v11a_cache_node_cf_6e3_formal | manifest、summary、episode_metrics、update_metrics、latest/best checkpoint、training plot | critic_diagnostics、slot_metrics | 7 | 26,177,734 |

5e-3 与 6e-3 的删除前 raw JSONL 只含逐 transition/逐 slot 明细；其关键回报、critic/Q/CF/KL、operation-rate、UAV mask 摘要已经压缩进 v11a_failure_and_io_analysis.md/.json，且核心 update/episode logs 与 checkpoint 完整保留。5e-4 与 6e-4 raw diagnostics 不删，因为它们是当前正式对照与 failure witness。

## 文件类别

### 必须保留

- 四个 V11-A formal run 的 run_manifest.json、run_summary.json
- 四个 V11-A formal run 的 episode_metrics.jsonl、update_metrics.jsonl
- 四个 V11-A formal run 的 latest_checkpoint.pt、best_train_return_checkpoint.pt
- 四个 V11-A formal run 的 plots/training_epoch_return.png
- frozen pool、V10-D formal run、V11-A gate run
- configs/proposed/system_config_ppo_v11a_cache_node_cf_5e3.toml
- configs/proposed/system_config_ppo_v11a_cache_node_cf_5e4.toml
- configs/proposed/system_config_ppo_v11a_cache_node_cf_6e3.toml
- configs/proposed/system_config_ppo_v11a_cache_node_cf_6e4.toml
- 被历史 run manifest 引用的配置、正式 mechanism reports、现有可复用分析脚本

### 本次删除

- 四个已经执行完毕、没有代码/脚本调用方的 V11-A server launch scripts
- 仅记录 launch preparation / server env fix、且已被本任务 consolidated reports 吸收的四份旧报告
- 没有被代码、脚本、run manifest 或现有报告引用，且没有对应输出的 v9 1e-2 KL scratch config
- 5e-3/6e-3 的四个大型逐 slot/逐 transition JSONL

## 诊断字段保留说明

保留下来的 episode/update logs 仍包含：

- 300 epochs 的 return、Q、critic EV、cost、trigger/change、UAV movement
- Q-cache EV/loss/gradient/optimizer steps
- CF raw/normalized mean/std、正负率、按 role 统计、fixed transition witness
- Cache/UAV branch KL、early-stop、executed/suppressed minibatches
- selected Cache trigger bitmask histogram 与 selected operation counts
- UAV operation mask 摘要、MOVE candidate count、feasible rate

没有把 selected operation rate 误当作 legal mask support rate；完整 Cache mask 仍只在 in-memory rollout tensor 中生成，原始 JSONL 不持久化该张量。

## 状态

- OUTPUT_INVENTORY = PASS
- LARGE_REDUNDANT_OUTPUTS_REMOVED = YES
- PROTECTED_POOL_PRESERVED = YES
- V10D_PRESERVED = YES
- CORE_CHECKPOINTS_PRESERVED = YES
- NO_TRAINING_OR_ROLLOUT_STARTED = YES

