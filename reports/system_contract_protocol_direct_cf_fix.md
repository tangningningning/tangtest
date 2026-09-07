# 系统合同、实验协议与 Direct-CF credit 专项修复记录

## 1. 范围与当前状态

本次工作基于当前本地 `main` 分支的最新代码（基线 commit `583e328`），只完成系统合同、实验协议、Direct-CF credit 和 prediction-batch 等价性相关修复；没有运行完整训练、完整 4-episode pilot 或完整 240-slot rollout，也没有提交或推送。

用户文件 `打包并上传.bat` 未修改、未暂存、未删除。

用户明确指出 `12.pdf` 才是系统模型。本次工作区、仓库和已提供附件中没有可读取的 `12.pdf`；没有使用仓库中的 `11.pdf` 推断或替代系统模型。因此 `cache.discount_beta=0.95` 保持不变，并继续标记为 `PROVISIONAL_IMPLEMENTATION`，不能视为已由最新系统模型确认。

## 2. 触发合同与实际执行合同

当前语义保持为：

- `z_ca=0`：对应 cache 节点必须保持上一状态；`z_ca=1`：只授予该节点重新配置的权限。
- `z_q=0`：对应 UAV 位置必须保持上一位置；`z_q=1`：只授予该 UAV 移动的权限。
- 实际 cache 更新和实际 UAV 移动均可小于请求，但不能大于请求。
- cache loading/fixed cost 和 UAV movement cost 均按实际变化计算；不把请求动作当成实际更新。

在 `system_model` 与 `execute_slot` 层增加了运行时合同断言，并在 slot/configuration summary 中同时输出 requested 与 actual 字段。已有的 legacy 字段保留，便于旧验证消费者继续读取。

cache/UAV candidate diagnostics 只记录搜索过程中已经计算过的候选，不声称全局最优，包含 reference objective、best feasible candidate objective、objective/utility/cost delta、requested trigger、actual update 和记录的候选数。没有额外改变 candidate 顺序、cache key、Matching 输入或最终配置选择。

bounded fixture 中的代表性记录为：

| 组件 | requested | actual | reference objective | best objective | best delta | utility delta | cost delta | 结果 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cache node `23` | 1 | 0 | 3.768511746674947 | 3.768511746674947 | 0 | 0 | 0 | PASS |
| `UAV_0` | 0 | 0 | 3.768511746674947 | 3.768511746674947 | 0 | 0 | 0 | PASS |
| `UAV_1` | 0 | 0 | 3.768511746674947 | 3.768511746674947 | 0 | 0 | 0 | PASS |

上表是 local candidate audit，不是对配置空间的全局最优声明。

## 3. T_ref/E_ref 与参数状态

train/test split 保持为 train origins `0..239`、test origins `240..299`。正式配置中的 `T_ref/E_ref` 已改为只由 train origins 的 `arrival=1` 任务派生，未使用 test origins：

- train 到达任务数：`3343`
- train CPU cycles P95：`30,628,985,166.15863`
- `T_ref=30.6289851662 s`，使用 `1.0e9 cycle/s` 的 TV local CPU
- `E_ref=3.06289851662 J`，使用同一 train-only P95 与当前 local-energy 公式

其余参数没有被伪装成已确认物理常数：

- `omega_delay/omega_energy`：`EXPERIMENT_DESIGN`
- `eta_ca`、`xi_ca`、movement energy coefficient、movement fixed cost：`PROVISIONAL_IMPLEMENTATION`
- `lambda_cache`、`lambda_movement`：`PROVISIONAL_IMPLEMENTATION`
- `cache.discount_beta=0.95`：`PROVISIONAL_IMPLEMENTATION`，在缺少 `12.pdf` 时未修改

因此当前协议可以进行受控实现验证，但仍不能把上述 provisional/system-scale 参数表述为由最新系统模型或文献完全校准。后续若只做 train-only sensitivity，建议单独比较 beta `0.90/0.95/1.00`，不改变本次默认值。

## 4. Direct-CF credit

正式 hierarchical WHAT 路径已切换为 `trigger_stratified`：

- 每个动态 action label 建立独立 capacity-1 reservoir。
- `N_i=0` 的 trigger 不产生样本。
- reservoir 数量由 `action_spec.action_dim` 动态取得；当前真实 action dimension 为 `11`，实现没有硬编码 `11`。
- `cf_samples_per_episode=8`、`gate_cf_samples_per_episode=8`、`cf_parallel_workers=8` 均保持不变；旧 `uniform` 路径继续兼容 legacy V3 checkpoint/run，Gate-CF 的 8 samples 未改变。
- 对 Batch4，选中的 sample `i` 使用 `trigger_free_count=N_i`，令 `N_total=sum_i N_i`，权重为 `N_i/N_total`；权重必须 finite、非负且和为 1。uniform 路径仍使用 `1/K`。

动态 reservoir fixture 中，三个代表性 trigger 的 `N_i` 为 `(2,3,0)` 时，选中样本为前两个 trigger，Batch4 权重为 `(0.4,0.6)`，第三个 trigger 不采样；该检查通过。没有运行完整 Batch4 训练。

## 5. TV local cache audit

当前 TV cache capacity 为 `5581.58344157102 MB`。服务目录中只有 service `4` 单独满足容量；最小两服务容量和为 `8968.825389784311 MB`，已经超过 TV capacity。现有 20 个 TV cache 状态均为 `[4]`，唯一缓存服务也是 `4`。

这被分类为合法的 capacity-driven degeneration，而不是 cache 状态生成器违反容量约束；但它限制了 TV local-cache 异质性，后续实验解释时需要明确该限制。本次没有修改容量、catalog 或 baseline。

## 6. bounded dynamic validation

只运行了 bounded real fixture 和 20-slot no-trigger window：

- requested cache/UAV 与 actual cache/UAV 均满足合同；no-trigger window 中请求和实际更新均为 0。
- cache state、UAV state 均按 slot 传播；cache/movement cost 在 no-trigger window 中为 0。
- 20-slot window 平均 reward 为 `10.318483885544154`，offload target changes 为 `219`，未改变 reward 或 Matching 逻辑。
- candidate diagnostics 中请求 1 但实际 0 的记录满足 `best_delta_objective <= tolerance`。

## 7. prediction batch serial/parallel 等价性

在同一 state、scenario、cache/UAV state 和 seed 下，对 bounded fixture 的前 8 个 prediction batches 比较 workers=1 与 workers=8：

- predicted utility：两侧均为 `3.768511746674947`，绝对差为 0
- predictive objective、cost、movement cost：一致，绝对差低于 `1e-12`
- per-batch 结果、resource feasibility、Matching 结果：一致
- final cache：一致
- final UAV positions：一致
- actual Matching candidate evaluations：两侧均为 `443`
- actual reward：两侧均为 `13.49688659470791`

parallel evaluator 保留 nested multiprocessing guard；子进程上下文退回 serial，不启动 Gate-CF/WHAT-CF 的嵌套 worker。

## 8. 最小验证结果

已通过：

1. `python -m compileall -q src`
2. project configuration load/validation
3. trigger contract assertion fixture
4. Direct-CF dynamic reservoir/weight fixture
5. bounded serial/parallel equivalence fixture
6. 20-slot no-trigger state/reward propagation fixture
7. `git diff --check`

没有运行完整 240-slot rollout、4-episode training 或 full Batch4。正式 prediction-batch 等价性与合同实现未发现 blocker；但在 `12.pdf` 不可用且 beta/若干 cost scale 仍 provisional 的情况下，不建议现在直接进行下一次完整 Batch4，应该先补齐系统模型依据或明确校准协议。

## 9. 本次涉及文件

- `src/common/project_config.py`
- `src/common/system_model.py`
- `src/decision/cache_configuration.py`
- `src/decision/slot_environment.py`
- `src/decision/uav_configuration.py`
- `src/ppo/ppo_training.py`
- `system_config.toml`
- `reports/system_contract_protocol_direct_cf_fix.md`

没有保留临时测试脚本、CSV、JSON 或 profiling 输出文件。
