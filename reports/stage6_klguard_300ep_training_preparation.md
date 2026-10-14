# Stage6 KL Guard 四组 300 Episode 正式训练准备

日期：2026-10-12  
项目根目录：`C:\Users\tangning\Desktop\code\adaptive_cache_mec`  
范围：仅完成训练准备；未启动训练或执行 Training Episode。

## 正式入口与协议

四个正式入口均存在，且只把学习率传给共享 Runner：

| 启动脚本 | Macro Actor 学习率 | 新 Run 目录 |
|---|---:|---|
| `scripts/training/run_stage6_lr_1e_minus_4.py` | `1e-4` | `runs/stage6_klguard_lr_1e_minus_4/` |
| `scripts/training/run_stage6_lr_5e_minus_4.py` | `5e-4` | `runs/stage6_klguard_lr_5e_minus_4/` |
| `scripts/training/run_stage6_lr_1e_minus_3.py` | `1e-3` | `runs/stage6_klguard_lr_1e_minus_3/` |
| `scripts/training/run_stage6_lr_5e_minus_3.py` | `5e-3` | `runs/stage6_klguard_lr_5e_minus_3/` |

四个入口在共享机制、配置、数据、Episode 范围和初始化 seed 上一致；文件差异仅为入口说明及传入的学习率。

四个脚本继续调用 `scripts/training/mainline_runner.py`，不创建独立 Trainer。Runner 为本次修复版训练将四个输出名映射到上述 `stage6_klguard_*` 目录。四个目录目前都不存在；Runner 会在正式启动时创建目录，并拒绝覆盖已存在的 Run。

每组入口配置为 300 Episode，训练 seed `20001–20300`，每 4 个 Episode 更新一次，共 75 个 PPO 更新组；初始化 seed 保持 `27005`。Runner 不传 `--resume` 或 `--generate-missing-episodes`，因此使用全新初始化并从已有训练池读取数据。四个学习率对应不同输出目录，可独立启动；它们不会争用同一 Run 路径。

## 数据与配置

- 正式 Training 池：`environment_data/proposed_joint_training/`
- 检查结果：300 个 `episode_*` 目录、300 份 `episode_manifest.json`；seed `20001–20300` 的 manifest 均存在。根目录的 `service_catalog_500_2000.json` 和 `stage6_training_pool_manifest.json` 均存在。
- Runner 的 `POOL_RELATIVE` 指向上述路径。本次只核对目录、manifest 数量和指定文件存在性，没有生成或修改 Episode，也没有读取 Validation/Test。
- 正式配置：`configs/proposed/system_config.toml`
- 配置 SHA-256：`684bdfa4dc200db6227b2b00a7718797a98bf658e54d2d4dbef6f7d1f5babf77`
- 配置加载及正式配置校验通过；`realizations_per_policy_update=4`，Macro Actor 学习率 schedule 为 `constant`。

## 修复版与记录字段

当前 PPO 训练控制版本为 `macro_actor_post_step_joint_kl_backtracking_v1`；`CHECKPOINT_VERSION=58`。本次没有改动 `trainer.py`、`reporting.py`、系统配置、Reward、PPO 更新逻辑、Matching/KKT 或 Checkpoint 格式。

现有记录链覆盖要求的诊断信息：

- `run_manifest.json`：初始化 seed、训练 seed 范围、目标 Episode 数、预期更新组数、学习率、配置哈希、训练控制版本和 Checkpoint 版本。
- `episode_metrics.jsonl`：Episode 序号及 seed、Episode Return、Task Utility、Cache Cost、UAV Movement Cost、Cache UPDATE 率、平均 Holding 长度、Termination 概率和候选熵；同组更新诊断按 `policy_update_group` 标记关联。
- `update_metrics.jsonl`：Actor 每步 pre-step / post-step Joint、Cache、Termination KL，候选尝试/拒绝/接受计数、接受的 LR 缩放及有效 LR、Actor optimizer step 数、KL 阈值和停止原因。
- `slot_metrics.jsonl`：逐 slot 的任务效用、Cache cost、移动 cost 和 Cache 更新数。

这些字段已由现有 rollout、trainer 和 reporting 记录，不需新增重复日志系统。

## 新旧 Run 隔离与冲突

四个新 `stage6_klguard_*` 目标路径均不存在，现阶段可用。以下旧产物均存在且保留：

- `runs/stage6_lr_1e_minus_4/`
- `runs/stage6_lr_5e_minus_4/`
- `runs/stage6_lr_1e_minus_3/`
- `runs/stage6_lr_5e_minus_3/`
- `runs/stage6_klguard_probe_lr_5e_minus_4_16ep/`

正式提交前仍需从服务器登录节点执行 `squeue -u "$USER"`，核实旧 Stage6 Job 的 RUNNING/PENDING 状态。当前本地没有 `squeue` / `scancel`，也没有配置 SSH 主机入口，因此：

`REMOTE_SLURM_STATUS = UNKNOWN`

本报告不把远程任务状态未知解释为已全部停止；本次没有取消远程任务。

## 检查结果

- 四个正式 Launcher 和共享 Runner 的 Python AST 语法检查通过。
- `conda activate adaptive_cache_mec` 在当前 PowerShell 中失败；已核验并使用 `C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`，其 `sys.prefix` 指向 `adaptive_cache_mec` 环境。
- 正式配置加载与正式配置校验通过。
- `git diff --check` 通过。
- 新 Run 路径无冲突；旧 Run 和 16 Episode 诊断 Run 保留。
- 未调用 Launcher、rollout、Matching/KKT 或 PPO 更新；未执行任何 Training Episode。
- 本次没有创建临时脚本或中间训练文件。

TRAINING_PREPARATION_STATUS = READY

远程 Slurm 核实仍是正式提交前的必要条件。
