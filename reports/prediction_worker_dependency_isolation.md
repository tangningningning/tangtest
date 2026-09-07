# Prediction-worker dependency isolation

基线：`62f515edc12860d0e49ec77a19c418576ce376b8`

本轮只处理 prediction-batch worker 的 PyTorch dependency isolation；没有重新审计 PPO、Direct-CF 数学、Matching/KKT 或系统模型。

## 1. 修改前 worker import chain

正式 prediction-batch 路径为：

`slot_environment.execute_slot -> cache_configuration PredictionBatchRuntime -> ProcessPoolExecutor(spawn) -> _initialize_prediction_batch_worker -> _evaluate_prediction_batch -> task_allocation.optimize_task_matching`

修改前 `_initialize_prediction_batch_worker()` 显式执行 `import torch`、`torch.set_num_threads(1)` 和 `torch.set_num_interop_threads(1)`。但 `_evaluate_prediction_batch()` 的实际计算只构造 task/service positions 并调用 Matching/KKT，不使用 torch。

这会使 Windows prediction worker 无必要加载 PyTorch/CUDA 依赖。parent PPO/training 侧仍然需要 torch，本轮没有触碰 parent import chain。

## 2. 修改内容

只修改：

`src/decision/cache_configuration.py::_initialize_prediction_batch_worker()`

删除 prediction worker 中的：

- `import torch`
- `torch.set_num_threads(1)`
- `torch.set_num_interop_threads(1)`
- 对应的 torch 异常处理 block

保留 worker payload 初始化和已有正式 worker/evaluator。没有引入新依赖，也没有使用 monkey patch。

## 3. Dependency result

静态检查确认：

- `src/decision/cache_configuration.py` 不再包含 torch import 或 torch thread-setting；
- prediction evaluator 依赖 NumPy、项目 common 模块和 task allocation；
- `src/decision/task_allocation.py`、`src/common/project_config.py`、`src/common/system_model.py` 没有把 torch 带入 prediction worker chain；
- parent PPO/training 侧的正常 torch import 未修改。

因此 prediction worker 不再主动加载 PyTorch/CUDA。

## 4. Bounded formal preflight

未修改 `system_config.toml`。启动时确认正式配置仍为：

| 参数 | 值 |
|---|---:|
| `prediction_batch_parallel_workers` requested | `8` |
| `cf_parallel_workers` requested | `8` |
| `cf_samples_per_episode` | `8` |
| Gate-CF samples | `8` |
| train origins | `0..239` |
| sampling strategy | `trigger_stratified` |

只运行一次与前轮相同的 spawn-safe bounded formal wrapper，约 `100 s` 后停止；没有降低 worker 数，没有运行第二次。

结果：

| 检查项 | 结果 |
|---|---|
| 上一轮 `WinError 1455` | 未复现 |
| `OMP: Error #111` | 未复现 |
| normal rollout progress | 未产生第一个 20-slot progress |
| normal rollout complete | 否 |
| prediction workers requested/effective | `8` / 未返回 effective profile；worker runtime 在完成前失败 |
| Direct-CF | 未开始 |
| completed CF branches | `0` |

最早新异常为：

```text
OpenBLAS error: Memory allocation still failed after 10 retries, giving up.
```

随后没有 normal rollout completion；本轮没有进入 Gate-CF/WHAT-CF，也没有 PPO 或 reference-drift update。该异常发生在正式 prediction-batch parallel runtime，属于剩余的 Windows/BLAS 资源问题，不是算法或 Direct-CF 数值问题。

## 5. Verdict

`RUNTIME_BLOCKER_CLEARED = NO`

PyTorch/CUDA dependency isolation 已完成，且上一轮 torch/DLL 与 OMP #111 错误均消失；但 workers=8 的 prediction-batch runtime 仍因 OpenBLAS 内存分配失败无法产生第一个 rollout progress，因此 runtime blocker 尚未清除。

后续应单独处理 Windows page file / BLAS/OpenMP memory configuration，或另行设计 CPU prediction-worker 的线程依赖隔离；本轮不降低 workers、不修改算法、不自动修改系统设置。

## 6. Cleanup and validation

- 临时 preflight wrapper 已删除。
- 没有生成 CSV、JSON、benchmark 或 profiling 文件。
- `system_config.toml` 未修改。
- “打包并上传.bat”未修改、未删除、未移动、未加入本轮修改。
- 未 commit、未 push。
