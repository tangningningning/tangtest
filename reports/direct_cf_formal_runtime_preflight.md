# Direct-CF formal runtime preflight

基线：`62f515edc12860d0e49ec77a19c418576ce376b8`
分支：`main`
目的：只验证当前正式配置下的 Direct-CF runtime feasibility；不重新审计 PPO 正确性或系统模型。

## 1. Workload and execution mode

本次只启动一次 bounded preflight，目标是：

- 一个完整 train episode：origins `0..239`，共 `240` 个 normal rollout slots；
- formal `cf_samples_per_episode=8`；
- formal Gate-CF sample budget `8`；
- 不进入 Batch4；
- 不运行 held-out test；
- 不执行 Gate/WHAT/PPO optimizer update；
- 不比较 workers=1/8；
- 不修改 `system_config.toml`。

实际调用的临时命令为：

```text
C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe tmp_direct_cf_formal_preflight.py
```

该 fixture 通过当前 formal collector、`run_gate_cf_branches()` 和 `run_direct_cf_branches()` 进入正式路径；`cf_parallel_workers` 直接读取当前配置，没有改成 serial。

当前配置在启动时确认：

| 项目 | 值 |
|---|---:|
| train origins | `0..239` |
| normal rollout steps | `240` |
| `cf_samples_per_episode` | `8` |
| Gate-CF samples | `8` |
| `cf_parallel_workers` requested | `8` |
| `prediction_batch_parallel_workers` requested | `8` |
| sampling strategy | `trigger_stratified` |
| reference drift cap | `0.005` |

## 2. Runtime result

preflight 没有完成 normal rollout，也没有进入 Gate-CF 或 WHAT-CF。bounded 窗口约为 `130 s`；在窗口内没有产生第一个 20-slot rollout progress 输出。

| 指标 | 结果 |
|---|---|
| normal rollout runtime | 未完成；未产生 `rollout_done` |
| WHAT-CF sample count | `0` 个已完成 branch；CF 阶段未开始 |
| Gate-CF sample count | `0` 个已完成 branch；CF 阶段未开始 |
| CF workers requested | `8` |
| CF workers effective | 未到达 CF dispatcher，无法产生 effective worker count |
| completed branch count | `0` |
| Direct-CF total runtime | `0 s`；未进入 Direct-CF branch |
| PPO/WHAT/Gate update | 未运行；本轮禁止 Batch4/update |
| reference drift update | 未运行 |
| max observed drift | N/A；没有完成 update |
| drift cap `0.005` | N/A；没有完成 reference-drift update |

## 3. Failure point

失败发生在：

```text
normal rollout
  -> configuration / prediction-batch parallel worker startup
  -> Windows spawn child imports torch
  -> torch DLL loader failure
```

子进程报告的异常为：

```text
OSError: [WinError 1455] 页面文件太小，无法完成操作。
Error loading ... torch\lib\cublas64_13.dll or one of its dependencies.
```

该 traceback 出现在 Windows `multiprocessing.spawn` 子进程导入临时 preflight script 的 `torch` 期间。它不是 Gate-CF/WHAT-CF branch 内部异常，也不是 PPO loss、reward、Matching/KKT 或 Direct-CF 数值错误。

本次没有出现用户之前所述的 access-violation 型 `python.exe` native crash；但 `WinError 1455` 已足以使正式并行 runtime 无法继续。由于没有完成第一个 20-slot progress，具体触发该 prediction batch 的 slot index 没有被 collector 返回；可确定阶段是 normal rollout 的 configuration prediction-batch spawn。

## 4. Cleanup and scope

- 临时脚本 `tmp_direct_cf_formal_preflight.py` 已删除。
- 没有生成 CSV、JSON、benchmark 或 profiling 输出文件。
- `system_config.toml` 未修改。
- 没有修改算法、CF sample 数、prediction horizon、scenario、Matching/KKT 或 training parameters。
- `打包并上传.bat` 未修改、未删除、未移动、未加入本轮修改。
- 没有 commit，也没有 push。

## 5. Verdict

**DIRECT_CF_RUNTIME_READY = NO**

唯一真实 blocker：当前正式 `prediction_batch_parallel_workers=8` 路径在 Windows spawn 子进程加载 PyTorch DLL 时因系统页面文件不足（`WinError 1455`）失败，导致 normal rollout 尚未完成，Direct-CF 尚未开始。没有重新打开 PPO 正确性或训练语义问题。
