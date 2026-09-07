# Windows spawn-safe Direct-CF runtime closure

基线：`62f515edc12860d0e49ec77a19c418576ce376b8`

本轮只检查 Windows spawn 与正式 prediction-batch/Direct-CF runtime；没有重新审计 PPO、Direct-CF 数学、Matching/KKT 或系统模型。

## 1. Previous failure mechanism

上一轮临时 wrapper `tmp_direct_cf_formal_preflight.py` 在 module top-level 导入了 `torch`，并通过 top-level project imports 间接加载了 PPO/training modules。Windows `spawn` 子进程重新 import wrapper main module 时，会再次执行 heavyweight PyTorch import。

上一轮 traceback 为 `OSError: [WinError 1455] 页面文件太小，无法完成操作`，并指向 `torch\lib\cublas64_13.dll`。这发生在 normal rollout 进入 prediction-batch parallel worker startup 时，还没有进入 Direct-CF。

## 2. Formal worker import chain

当前正式 prediction-batch 路径为：

`slot_environment.execute_slot -> cache_configuration prediction-batch runtime -> ProcessPoolExecutor(spawn) -> decision.cache_configuration._initialize_prediction_batch_worker -> _evaluate_prediction_batch -> task_allocation.optimize_task_matching`

静态检查结果：

- `src/decision/cache_configuration.py` module top-level 没有 `import torch`；主要依赖是 NumPy、`common.project_config`、`common.system_model` 和 delayed `task_allocation`。
- `_evaluate_prediction_batch()` 本身只构造 tasks、service positions 并调用 Matching/KKT；没有使用 torch。
- `_initialize_prediction_batch_worker()` 当前显式执行 `import torch`，再调用 `torch.set_num_threads(1)` 和 `torch.set_num_interop_threads(1)`。
- 因此 prediction worker 的实际 batch 计算不需要 torch；torch 目前只是 worker initializer 的线程设置依赖。
- parent-only PPO/training modules（例如 `src/ppo/ppo_training.py`、`src/ppo/ppo_rollout.py`）会加载 PyTorch，但不应由 spawn-safe wrapper 的 module top-level 触发。

本轮没有修改 `_initialize_prediction_batch_worker()`，也没有重构正式 import chain。

## 3. Spawn-safe temporary wrapper

本轮使用的临时 wrapper 为 `tmp_windows_spawn_safe_preflight.py`：

- module top-level 只保留标准库 `multiprocessing`、`sys`、`time`、`traceback`、`pathlib`；
- `numpy`、`torch`、`project_config`、PPO/training modules 全部延迟到 `main()` 内导入；
- 使用 `if __name__ == "__main__": multiprocessing.freeze_support(); main()`；
- prediction worker 继续使用项目现有 `ProcessPoolExecutor`、initializer 和 evaluator；
- 正式配置直接读取，未改 `system_config.toml`、worker 数、scenario、horizon、CF sample 数或算法逻辑。

正式配置启动时确认：

| 参数 | 值 |
|---|---:|
| `prediction_batch_parallel_workers` | `8` |
| `cf_parallel_workers` | `8` |
| `cf_samples_per_episode` | `8` |
| Gate-CF samples | `8` |
| train origins | `0..239` |
| prediction sampling strategy | `trigger_stratified` |

实际调用命令：

`C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe tmp_windows_spawn_safe_preflight.py`

## 4. One bounded preflight result

本轮只运行一次，bounded window 约 `130 s`。没有重复运行，也没有把 workers 降为 serial。

| 检查项 | 结果 |
|---|---|
| wrapper top-level torch import | 已移除 |
| 上一轮 `WinError 1455` | 本轮未复现 |
| normal rollout progress | 未产生第一个 20-slot progress 输出 |
| normal rollout complete | 否 |
| Gate-CF start | 否 |
| WHAT-CF start | 否 |
| CF workers requested/effective | `8` / 未到达 CF dispatcher |
| completed CF branches | `0` |
| PPO/Gate/WHAT update | 未运行；本轮不进入 Batch4/update |
| reference drift | 未运行 |

在 normal rollout 的 prediction-batch parallel 阶段，wrapper 已不再因 top-level torch import 触发 DLL loader traceback；随后出现 `OMP: Error #111: Memory allocation failed`，并伴随 multiprocessing queue 的 `OSError: handle is closed`。

该错误发生在 formal prediction-batch worker runtime，normal rollout 尚未完成，Direct-CF 尚未开始。运行已在 bounded window 到达后停止，未发生 access-violation 型 `python.exe` native crash。

## 5. Classification

`REAL_RUNTIME_ENVIRONMENT_BLOCKER`

理由：spawn-safe wrapper 已消除上一轮 wrapper-induced top-level PyTorch import amplification，上一轮 `WinError 1455` 没有复现；但保持正式 workers=8 时，正式 prediction-batch parallel runtime 仍出现 OMP memory allocation failure，导致 normal rollout 无法越过 bounded preflight。因此不能把 Direct-CF runtime 判为 ready。

## 6. Recommended next actions (not executed)

优先方案是增加 Windows page file / virtual memory，然后在不改变正式 `workers=8` 的前提下重试。没有自动修改系统设置。

次选方案是单独做 prediction-worker dependency isolation：确认 CPU prediction worker 不再为线程设置而加载 PyTorch/CUDA；本轮没有修改公共 worker initializer，也没有通过降低 worker 数绕过问题。

## 7. Cleanup and scope

- 临时 wrapper `tmp_windows_spawn_safe_preflight.py` 已删除。
- 没有生成 CSV、JSON、benchmark 或 profiling 文件。
- `system_config.toml` 未修改。
- 没有 commit，也没有 push。
- `打包并上传.bat` 未修改、未删除、未移动、未加入本轮修改。
