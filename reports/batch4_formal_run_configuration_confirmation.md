# Batch4 formal run configuration confirmation

## Scope and baseline

本报告只对当前 `main` 分支的 Batch4 正式运行入口、配置、方法身份、seed、训练边界、持久化路径和覆盖风险做静态确认。没有执行 Batch4、PPO training、rollout、Direct-CF、Matching/KKT、runtime preflight、workers 测试或参数 sensitivity。

| item | value |
|---|---|
| baseline SHA | `8225e3cbfae9c277c224ed3c0cceee7783f4913f` |
| baseline message | `Finalize PPO protocol and isolate prediction worker dependencies` |
| `FORMAL_READY` | `YES` |
| `PPO_PROTOCOL_READY` | `YES` |
| `RUNTIME_BLOCKER_CLEARED` | `NO` |

证据来自当前 `system_config.toml`、`pyproject.toml`、`src/ppo/ppo_training.py`、`src/ppo/ppo_update.py`、`src/ppo/causal_artifacts.py` 以及当前生成 artifact 的 metadata；没有使用历史聊天推断。

## A. Formal Batch4 entrypoint

### 当前真实入口

`pyproject.toml` 声明的正式 console entrypoint 是：

```text
ppo-train = ppo.ppo_training:main
```

当前环境中该 console wrapper 位于：

```text
C:\software\Anaconda_envs\envs\adaptive_cache_mec\Scripts\ppo-train.exe
```

源码中的等价 Python CLI 是 `src/ppo/ppo_training.py:build_parser()` / `main()`。正式训练必须显式传入：

```text
--config system_config.toml --formal-train --episodes N --run-dir PATH
```

其中 `--episodes` 必须是正整数，`--run-dir` 是正式训练的必填持久化目录；代码不会自动写入 `data/generated` 或默认结果目录。默认配置来源是当前工作目录下的 `system_config.toml`。

### Batch4 是否有独立 launcher

当前仓库没有独立的 `batch4.py`、Batch4 `.bat`、`.ps1` 或 shell launcher，也没有 Batch4 method/seed matrix 配置文件。代码内部的 `_hierarchical_gate_formal_training()` 明确标记为 formal hierarchical WHEN/WHAT Batch4 path；因此当前 Batch4 是统一 PPO CLI 进入的内部 4-episode batch path，不是独立外部脚本。

用户文件 `打包并上传.bat` 与本任务无关；本轮未执行、未修改、未 stage。

## B. Methods and variants

### 当前配置实际选择的 method

| method name | algorithm / credit assignment | WHAT Direct-CF | Gate-CF | reference drift | identity |
|---|---|---:|---:|---|---|
| `proposed_hierarchical_reconfiguration_gate_direct_cf_refdrift` | `hierarchical_gate_direct_cf` | enabled; `8` samples/episode | enabled; `8` samples/episode | WHAT/RefDrift path enabled with cap `0.005`; Gate branch itself is not a RefDrift estimator | proposed |

当前配置没有 baseline 或 ablation method 的显式列表，也没有 method dispatch loop。源码虽然支持 `legacy_scalar`、`proposed_per_trigger_q`、`proposed_centralized_cf_q`、`proposed_sparse_direct_cf_ppo` 及 frozen-policy/refdrift method IDs，但这些只是代码 dispatch 能力，当前 `system_config.toml` 没有把它们声明为 Batch4 实验组，不能据此擅自扩展 Batch4。

因此，当前仓库能够确认的只有上表这一 active proposed method；Batch4 若要求多个 baseline/ablation/proposed 组，组清单尚未在当前代码/config 中定义。

## C. Seed, repetition, and fixed artifacts

### Formal training seed

| item | current value | source / meaning |
|---|---:|---|
| `experiment.seed` | `4401` | formal PPO training seed |
| `experiment.repeat_index` | `0` | metadata field; not used to construct `run_dir` |
| explicit Batch4 seed list | not present | no seed loop or repetition list exists |
| repetitions per method | not present | must not be inferred from `repeat_index=0` |

当前 active run 使用单一 `experiment.seed=4401`。代码从该 seed 派生 policy/RNG，并为 WHAT/Gate counterfactual sampling 使用独立 deterministic streams；没有当前可确认的多 seed set，也不能确认“所有方法使用相同 seed set”，因为方法矩阵本身未定义。

### Fixed prediction/task/train artifacts

当前 artifact 路径由 `system_config.toml` 固定：

```text
data/generated/vehicle_roles.json
data/generated/trajectory.csv
data/generated/service_catalog.json
data/generated/popularity.csv
data/generated/popularity_predictions.csv
data/generated/popularity_history.csv
data/generated/tasks.csv
data/generated/tv_local_cache.json
data/generated/vehicle_predictions.csv
data/generated/prediction_scenarios.npz
data/generated/popularity_gru.pt
data/generated/node_positions.json
```

当前 metadata 还确认：

- `prediction_scenarios.npz` 有 `300` 个 origin，`origin_slots=0..299`；
- `scenario.num_scenarios=20`，`scenario.master_seed=20260820`；
- task generation 主 seed 为 `4401`；
- vehicle/traffic 主 seed 为 `4401`，observation noise seed 为 `9101`；
- `data/generated/generation_metadata.json` 记录当前 config/artifact hashes，说明正式训练输入是当前已生成 artifact，而不是运行时随机重建。

## D. Episode and training length

当前 active hierarchical path 的 batch 约束来自代码常量与配置：

| item | value |
|---|---:|
| episodes per one Batch4 | `4` |
| `algorithm.ppo.cf_batch_episodes` | `4` |
| origins per training episode | `240` |
| training origins | `0..239` |
| one 4-episode Batch4 environment steps | `960` planned steps |
| PPO update epochs | `1` |
| actor/Gate update timing | after the 4-episode Batch4 boundary |

正式命令中的 `--episodes 4` 是当前 hierarchical path 可完整结束的最小一个 Batch4；该值不是配置文件中的 episode count，而是 CLI caller 必须显式提供的 target total episode count。代码拒绝 hierarchical run 中不是 `4` 的整数倍的 episode target。

`simulation.warmup_time_s=100.0` 是上游仿真/数据时间轴的 warm-up，不是 PPO training warm-up 或 burn-in。当前 formal training loop 没有另外的 PPO warm-up/burn-in episode。

### Checkpoint and evaluation intervals

- checkpoint/history 在每个完整 collection episode 后持久化，包含当前 Batch4 的 pending WHAT/Gate samples；Batch4 boundary 才执行一次 Gate/WHAT optimizer update。
- 每个正式 episode 写入 `episode_start_artifacts/episode_XXXX.pt`。
- 完成 update 后写入对应 causal update artifacts 和 `episodeN_update_summary.json`。
- training convergence figure 在本次 formal run 完成持久化后生成。
- 没有训练期间的 evaluation interval 配置；held-out test 是训练完成后的独立 `--held-out-test --checkpoint ...` CLI，不参与 optimizer update、checkpoint 选择或超参数选择。

## E. Train/test boundary

| split | origin range | current semantics |
|---|---:|---|
| train | `0..239` | formal PPO collection、WHAT/Gate CF、optimizer update only |
| test | `240..299` | held-out evaluation only, after training, no optimizer update |
| validation | none | no PPO validation origin range is configured |

`build_formal_training_protocol()` 从 `algorithm.ppo.data_protocol` 读取边界，并校验该边界完整覆盖当前 300-origin prediction artifact。训练 protocol 只建立 `0..239` 的 episode sequence；held-out evaluator 另行读取 `240..299`，不在 formal training loop 内运行。

## F. Current PPO/CF formal parameters

| parameter | current value |
|---|---:|
| `algorithm.ppo.method` | `proposed_hierarchical_reconfiguration_gate_direct_cf_refdrift` |
| `algorithm.ppo.credit_assignment` | `hierarchical_gate_direct_cf` |
| `algorithm.ppo.actor_architecture` | `v3_full_shared_actor_frozen_policy_batch_v1` |
| `cf_samples_per_episode` | `8` |
| Gate-CF samples/episode | `8` (formal hierarchical code constant) |
| `cf_batch_episodes` | `4` |
| `cf_parallel_workers` | `8` |
| `prediction_batch_parallel_workers` | `8` |
| `cf_sampling_strategy` | `trigger_stratified` |
| `reference_probability_drift_cap` | `0.005` |
| `gamma` | `1.0` |
| `gae_lambda` | `0.95` |
| `clip_epsilon` | `0.2` |
| `learning_rate` | `3.0e-4` |
| `update_epochs` | `1` |
| `num_minibatches` | `4` |

本表只记录当前值，没有修改配置或重新验证算法语义。

## G. Checkpoint and resume behavior

### Required run directory and files

`--run-dir` 是 caller 显式提供的 output root。当前正式 run 会在其下生成：

```text
<run-dir>/latest_checkpoint.pt
<run-dir>/training_history.json
<run-dir>/episode_start_artifacts/episode_0001.pt
<run-dir>/causal_update_artifacts/update_0001/...
<run-dir>/episode4_update_summary.json   # update 成功后
<run-dir>/figures/01_training_convergence.png
```

hierarchical Gate causal artifacts 还在对应 update 目录下保存独立 Gate artifact。history schema、checkpoint schema、causal artifact schema 和 compatibility metadata 都随 run 保存。

### Resume

- 默认不 resume；只有显式加 `--resume` 才从现有 `--run-dir` 继续。
- `resume=True` 要求 `latest_checkpoint.pt` 已存在；`--episodes` 表示该 run 的目标总完成 episode 数，而不是额外 episode 数。
- resume 恢复 WHAT/actor policy、Gate policy、对应 optimizer states、policy/gate version、optimizer counters、training history、pending CF/reference records 以及 Python/NumPy/PyTorch/WHAT/Gate sampling RNG state。
- `resume=False` 如果目标目录已有 `latest_checkpoint.pt` 或 `training_history.json` 会拒绝启动；causal artifact 和 episode-start artifact 也有重复索引拒绝逻辑。

## H. Output isolation and overwrite risks

当前代码没有 Batch4 output root，也没有根据 `method`、`seed`、`repeat_index` 或 `output_tag` 自动构造路径。`experiment.output_tag="main"` 和 `repeat_index=0` 只存在于配置/metadata，不提供目录隔离。

因此：

- 不同 method、seed、repeat 必须由 caller 显式使用不同 `--run-dir`；当前代码没有自动隔离保证。
- 同一 run directory 的非-resume 启动会因关键 checkpoint/history 已存在而拒绝，避免静默覆盖旧正式 run。
- 正确使用不同 run directory 时，checkpoint、history、causal artifacts、figures 和 per-update summaries 会留在该 run root 下。
- 训练 CLI 不启动 held-out test；test report 由独立 test 命令通过显式 `--test-report` 路径写出，因此 test 不会自动混入 train history，但该 report 路径同样需要 caller 管理。
- 当前已有 `runs/ppo_train_test_pilot_parallel_001` 是历史 pilot 目录，不是代码声明的 Batch4 output root；本轮未使用或修改它。

## I. Known runtime risk (record only)

继续保留既有事实：

```text
RUNTIME_BLOCKER_CLEARED = NO
```

已知风险是 Windows `prediction_batch_parallel_workers=8` prediction-batch runtime 的 OpenBLAS memory allocation failure。它是 Batch4 实际启动前需要用户决定是否承受/处理的 runtime environment risk，不是 PPO、Direct-CF 数学或训练正确性结论。本轮没有降低 workers、修改 page file/OpenBLAS/OpenMP，也没有运行 runtime preflight。

## J. Accurate launch command(s) — not executed

### Current active method, one Batch4

当前环境中可确认的工作目录和 interpreter：

```text
working directory:
C:\Users\tangning\Desktop\code\adaptive_cache_mec

Python/console wrapper:
C:\software\Anaconda_envs\envs\adaptive_cache_mec\Scripts\ppo-train.exe

config:
system_config.toml
```

源码接受的准确命令形状为：

```powershell
Set-Location 'C:\Users\tangning\Desktop\code\adaptive_cache_mec'
& 'C:\software\Anaconda_envs\envs\adaptive_cache_mec\Scripts\ppo-train.exe' `
  --config 'system_config.toml' `
  --formal-train `
  --episodes 4 `
  --run-dir '<caller-must-provide-a-new-unique-batch4-run-dir>'
```

这条命令没有被执行。`<caller-must-provide-a-new-unique-batch4-run-dir>` 不是猜测的现有路径：当前仓库没有 Batch4 output root，且代码明确要求 caller 提供 `--run-dir`。因此在没有用户指定 output root 前，不能诚实地给出一个已确定的 literal Batch4 output path。

### Multiple methods

当前仓库没有 Batch4 method matrix，也没有可按顺序执行的 baseline/ablation/proposed command list；本轮不擅自生成多个命令。若后续确定多个 method，必须为每个 method/seed/repeat 选择独立 `--run-dir`，并在同一正式配置/固定 artifact 下分别启动。

## K. Final classification

当前 active hierarchical method 的 CLI、CF 参数、单一 seed、artifact 输入、train/test boundary 和 checkpoint/resume 语义均可从真实代码/config 确认。但“Batch4 formal run”作为实验批次仍缺少：

1. 当前仓库未定义 baseline/ablation/proposed 的正式 method roster；
2. 当前仓库未定义 Batch4 seed set 和每个 method 的 repetition count；
3. 当前仓库未定义正式 output root / method-seed-repeat path convention，`--run-dir` 仍需 caller 指定。

所以本轮结论为：

```text
BATCH4_CONFIG_READY = NO
```

这不是对 PPO/Direct-CF correctness 的否定；它只表示在当前 commit 的真实代码/config 下，Batch4 的实验矩阵和输出隔离配置还没有完整声明。已知 Windows/OpenBLAS runtime risk 仅作记录，未在本轮处理。

