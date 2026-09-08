# Batch4 single-run declaration

## Declaration

本次只声明并运行 proposed method 的一个正式 Batch4；不包含 baseline/ablation matrix。

| field | declared value |
|---|---|
| method | `proposed_hierarchical_reconfiguration_gate_direct_cf_refdrift` |
| seed | `4401` |
| repeat | `0` |
| episodes | `4` |
| train origins | `0..239` |
| held-out test | `240..299` |
| config | `system_config.toml` |
| baseline commit | `8225e3cbfae9c277c224ed3c0cceee7783f4913f` |

当前配置中的 method、seed、CF/PPO 参数和 train/test origin boundary 与本声明一致。本声明没有修改 `system_config.toml`、算法参数、seed 或训练代码。

## Chosen run directory

选择的全新 run directory 为：

```text
runs/batch4_proposed_hierarchical_seed4401_repeat000
```

静态检查确认该路径当前不存在。现有 `runs/ppo_train_test_pilot_parallel_001` 未使用、未覆盖、未修改。

## Literal launch command

工作目录：

```text
C:\Users\tangning\Desktop\code\adaptive_cache_mec
```

正式 console entrypoint：

```text
C:\software\Anaconda_envs\envs\adaptive_cache_mec\Scripts\ppo-train.exe
```

literal PowerShell command：

```powershell
Set-Location 'C:\Users\tangning\Desktop\code\adaptive_cache_mec'
& 'C:\software\Anaconda_envs\envs\adaptive_cache_mec\Scripts\ppo-train.exe' `
  --config 'system_config.toml' `
  --formal-train `
  --episodes 4 `
  --run-dir 'runs/batch4_proposed_hierarchical_seed4401_repeat000'
```

该命令本轮只生成，未执行。未运行 Batch4、PPO、rollout、Direct-CF、workers 或任何测试。

## Scope safeguards

- 只运行 proposed method；不运行 baseline/ablation。
- seed、repeat、episodes、train/test origins 和正式 CF/PPO 参数保持不变。
- 不使用现有 pilot/run directory。
- 不执行、不修改、不 stage 用户文件 `打包并上传.bat`。
- 本轮不 commit、不 push。

## Final status

```text
BATCH4_SINGLE_RUN_READY = YES
```

