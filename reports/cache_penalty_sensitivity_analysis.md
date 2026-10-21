# Cache 更新成本惩罚与净效用敏感性分析

## 范围与证据

本报告只检查缓存成本公式、奖励组合和已有验证结果；没有运行训练、重新评价 Episode、读取 Test 内容或访问 `environment_data/`，也没有修改源码、配置、Checkpoint 或已有结果。

- 结果文件：`runs/research_mappo_stage3_main_experiment/validation_3seeds_20261018_retry1/validation_results.json`，SHA-256 `b1940bfad86078d7d7a831f1ba5dd127c06db7efe87136ca44f167e11ed0e5ef`。
- 正式配置：`configs/proposed/system_config.toml`，SHA-256 `03328b88658a6f15c0518eee9f4903d5647f12a800432e58c74b42fdef51d80c`。
- Stage3 Loss Profile：`configs/research/mappo_stage3_loss_coefficients_v1.json`，SHA-256 `c91bccc6bf91f76c407a18c9953b4fa2c2d346fcfef3337ad63ee49b23d8bdb4`。
- 结果文件报告三个完整 300-Slot Validation Episode：30001–30003；Update 13 对应基础 MAPPO，Update 16 对应 Stage3。结果元数据为 `constraints_and_reward_contract_valid=true`、`trainer_update_called=false`、`test_accessed=false`。这些是固定策略结果的事后分析，不是改权重后的训练结果。

## 1. 实际缓存成本模型

`src/common/system_model.py:309–390` 是实际成本实现；`src/decision/cache_configuration.py:497–516` 将当前节点参数与正式配置传入同一实现。对每个节点 m：

```text
z_m = 1[cache_new_m != cache_old_m]
loaded_mb_m = sum_f service_size_mb_f * max(cache_new_mf - cache_old_mf, 0)
loading_term_m = xi_ca_m * loaded_mb_m
C_ca = cache_cost_weight * sum_m (eta_ca_m * z_m + lambda_cache * loading_term_m)
```

因此成本有两部分：缓存矩阵实际变化时收取固定重配置成本；新增服务的装载量产生可变成本。仅删除服务不会产生负装载成本，但仍会触发固定成本。候选动作即使请求 UPDATE、实际缓存矩阵未变，也不收取成本，因为代码按 `old != new` 计算实际触发。

服务大小以 MB 输入公式：目录加载器在 `src/decision/cache_configuration.py:358–373` 将 `program_size` 作为 `service_sizes_mb` 使用，任务场景配置也以 MB 为服务程序大小单位（`src/scenario/task_scenario.py:80,122`）。参数的当前值来自 `configs/proposed/system_config.toml`：

| 节点类型 | `eta_ca` | `xi_ca` |
|---|---:|---:|
| SV | 0.20 | 1.0e-3 |
| RSU | 0.10 | 0.7e-3 |
| UAV | 0.15 | 1.2e-3 |

全局 `lambda_cache=0.1`，`cache_cost_weight=0.1`。函数 `system_model.cache_update_cost` 的 `cache_cost_weight` 代码默认值是 1.0，但当前正式配置显式传入 0.1；`lambda_cache` 是必需参数，由正式配置提供 0.1。配置参数状态表明确将各节点 `eta_ca`、`xi_ca` 及 `objective.lambda_cache` 标记为 `EXPERIMENT_DESIGN`；`cache_cost_weight` 在该状态表中没有单独条目，代码注释则把它定义为缩放整个 `C_ca` 的外层权重，并说明“典型单节点更新约占 normal-slot utility 10%”只是 experiment-design interpretation。`eta_ca` 不是实测重配置能耗，`xi_ca` 不是物理 J/MB。成本最终以任务效用的目标尺度计入，不能解释为焦耳总量。虽然实现返回名为 `loading_energy` 的数组，配置注释明确它是相对/等效量，而非实测焦耳。

配置注释中“典型单节点更新约占 normal-slot utility 10%”是实验设计解释，不是文献标定或硬件测量；当前代码/结果没有独立证明这个 10% 假设。

## 2. 奖励合同与重复惩罚核对

任务效用 (Q) 对每个任务计入完成收益，并减去按 `T_ref`、`E_ref` 归一化的时延和车辆能耗项；缓存装载不在这两个任务惩罚项里重复计算。`src/decision/slot_environment.py:610–620,1269` 与 `src/ppo_simple/rollout.py:395–404` 确认：

```text
G(t) = Q(t) - C_ca(t) - C_q(t)
```

`rollout._check_reward` 逐 Slot 校验该恒等式。`src/ppo_simple/evaluator.py:341–351` 将实际任务效用、Cache 成本和 UAV 成本分别累加。结果 JSON 中所有策略均有 `reward_contract_max_abs_error=0`，三个策略的 UAV 位移和 UAV 成本均为 0。

Stage3 没有改变环境成本公式。当前 Profile 只包含 Trigger、Candidate 排序和 ΔQ Huber 辅助损失系数，不含第二个 Cache 成本扣项。Stage3 反事实分支通过正式 `system_utility` 奖励生成折扣目标（`src/ppo_simple/cache_counterfactual_value.py:951,960,984,1063–1093`）；单次环境奖励中已扣的成本没有再被加扣一次。辅助损失影响训练梯度，但不是 Validation 的 `G` 中额外扣除的成本。

## 3. 三个场景的实际效用与成本

下表数值直接来自结果文件。`Δ` 为 Stage3 Update 16 减 Update 13；更新数按实际改变缓存的节点级操作计，不是服务条目增删数。

| Seed | Update 13：Q / Cca / Cq / G；变更数 | Update 16：Q / Cca / Cq / G；变更数 | ΔQ | ΔCca | ΔG |
|---|---|---|---:|---:|---:|
| 30001 | 3948.240445 / 16.945979 / 0 / 3931.294466；156 | 3941.437212 / 36.220869 / 0 / 3905.216343；314 | -6.803233 | +19.274891 | -26.078124 |
| 30002 | 3869.468941 / 16.945979 / 0 / 3852.522962；156 | 3863.786240 / 36.437549 / 0 / 3827.348690；316 | -5.682701 | +19.491570 | -25.174272 |
| 30003 | 3924.746314 / 16.945979 / 0 / 3907.800335；156 | 3919.593338 / 38.170988 / 0 / 3881.422350；332 | -5.152976 | +21.225009 | -26.377985 |

三场景均值：Update 13 的任务效用 3914.151900、Cache 成本 16.945979、净效用 3897.205921、实际缓存变更 468 次；Update 16 分别为 3908.272263、36.943136、3871.329128、962 次。Update 16 平均比 Update 13 多 164.7 次节点级变更，平均 Cache 成本多 19.997157，任务效用反而低 5.879637，净效用低 25.876793。两者每次节点变更的平均总成本分别约 0.108628 和 0.115207。

总 Cache 成本约占任务效用：Update 13 为 0.43%，Update 16 为 0.92%–0.97%。这些全 Episode 比例较小，但策略间成本差仍明显，因为 Stage3 进行了更多更新。

### 固定策略的成本分量范围

结果 JSON 保留每 Slot 总 Cache 成本及变更数，但没有缓存节点身份、各节点 `cost_by_node` 或实际 `loaded_mb`。因此不能从既有记录精确拆出固定项与装载项。仍可依据已确认的配置给出有依据的范围：每次有效节点变更的固定项是 `0.1 × eta_ca`，在 0.01–0.02 之间；其余总成本为非负装载项。按结果中的节点级变更数计算：

| 策略/Seed | 固定项范围 | 装载项范围 |
|---|---:|---:|
| Update 13 / 任一 Seed | 1.560–3.120 | 13.826–15.386 |
| Update 16 / 30001 | 3.140–6.280 | 29.941–33.081 |
| Update 16 / 30002 | 3.160–6.320 | 30.118–33.278 |
| Update 16 / 30003 | 3.320–6.640 | 31.531–34.851 |

此范围使用 `cache_update_operations == effective_cache_changes` 且无无效 UPDATE 的记录，并以节点固定成本最小值 RSU 0.10、最大值 SV 0.20 计算；实际固定项取决于每次更新的节点类别。不能据此伪称已获得精确组件分解。

## 4. 事后权重敏感性

当前真实公式不是单一 `lambda_cache × C_cache`：`lambda_cache` 只缩放新增装载项，`cache_cost_weight` 缩放固定项和装载项。为严格复用可观测总成本，先保持策略动作、Q、Cq 和 `lambda_cache=0.1` 不变，将整个 `C_ca` 乘以 `s`，等价于把 `cache_cost_weight` 从当前 0.1 改为 `0.1s`。这些是**固定现有策略的事后重算**，没有改变策略或重新训练。

| 全成本倍率 (s)（外层 `cache_cost_weight`） | Update 13 平均 G | Update 16 平均 G | ΔG 30001 / 30002 / 30003 | 平均 ΔG |
|---:|---:|---:|---|---:|
| 1.00（0.100） | 3897.205921 | 3871.329128 | -26.078124 / -25.174272 / -26.377985 | -25.876793 |
| 0.75（0.075） | 3901.442416 | 3880.564912 | -21.259401 / -20.301379 / -21.071733 | -20.877504 |
| 0.50（0.050） | 3905.678911 | 3889.800696 | -16.440678 / -15.428486 / -15.765480 | -15.878215 |
| 0.25（0.025） | 3909.915405 | 3899.036480 | -11.621956 / -10.555594 / -10.459228 | -10.878926 |
| 0（0） | 3914.151900 | 3908.272263 | -6.803233 / -5.682701 / -5.152976 | -5.879637 |

KEEP+STAY 没有缓存更新成本，三场景平均净效用为 3936.080158，不随 `s` 改变。即使把整个 Cache 成本项归零，Update 16 在三个场景仍因任务效用低于 Update 13 而落后；两种学习策略也仍低于 KEEP+STAY。

若只改变实际 `lambda_cache`，固定项仍会保留。用变更数和 `eta_ca` 范围可对 ΔG 作区间重算，但不能得到精确值：例如 `lambda_cache` 从 0.10 减半到 0.05、`cache_cost_weight` 保持 0.1 时，三场景平均 Stage3−Update 13 的净效用差约落在 **[-17.525, -16.702]**；将 `lambda_cache` 设为 0 时仍约为 **[-9.173, -7.526]**。区间来自未知的每节点固定项分配，所有端点仍为负。把实际 `lambda_cache` 设为 1.0 会把其从 0.1 提高十倍，并非降低整体成本权重。上述结果进一步说明这两个配置参数不能混作一个 λ。

## 5. 对“成本惩罚过重”的判断

1. **物理或论文依据：未找到。** 当前 `eta_ca`、`xi_ca`、`lambda_cache` 与 `cache_cost_weight` 在配置中都标为实验设计；项目注释明确它们不是硬件测量值或直接文献标定值。因此可以确认目标函数系数的设计来源，但不能仅凭这些系数判定物理上“合理”或“过重”。
2. **量纲：实现内一致，物理量解释有限。** 新增程序量按 MB 与 `xi_ca` 的相对每 MB 系数相乘，再用 `lambda_cache` 映射到效用成本；固定和装载项再由 `cache_cost_weight` 统一缩放。没有证据显示 MB/MB 或奖励单位发生代码级错配，但 Cca 不是焦耳。
3. **重复惩罚：未发现。** 正式 Slot 奖励只组合一次 Cache 和 UAV 成本；Stage3 辅助训练损失不是第二项环境成本。结果中的 Reward 恒等式误差为 0。
4. **成本归零后的表现：任务效用差仍存在。** 固定策略下，Cache 成本归零后 Stage3−Update 13 仍为 -5.153 至 -6.803；所以当前观察到的 Stage3 退化不能只归因于 Cache 成本。当前 Cache 成本差约占净效用差的 77%，主要与 Stage3 更频繁更新相伴；这属于结果分解，不证明成本过重，也不说明降权训练会改善策略。
5. **因果限制。** 表中所有敏感性分数固定了已经执行的动作和任务效用。若在较低 `cache_cost_weight` 或 `lambda_cache` 下重新训练，更新频率、缓存配置和 Q 都可能变化；本报告没有估计该新策略结果。当前 Stage3 验证文件的 ΔQ 标签诊断为 unresolved，也不支持把任务效用下降单因解释为成本权重。

## 结论与唯一建议

**RECOMMENDED_ACTION = KEEP_COST_MODEL_UNCHANGED。** 当前没有证据证明成本公式重复扣罚或存在代码级单位错误；固定策略敏感性显示调低成本只能缩小差距，不能消除任务效用差。成本系数确实缺少物理/论文标定，但这本身不足以凭当前三个历史复用 Validation Episode 选出更低值。先保持成本合同，优先解决 Stage3 缓存更新决策与任务效用下降问题；本报告不建议启动降权训练。

| 项目 | 结论 |
|---|---|
| 实际成本公式和奖励合同 | 已从执行源码核实 |
| 固定项/装载项的逐策略精确分解 | 结果文件未保存节点身份与 loaded MB；仅能给出范围 |
| 双重 Cache 惩罚 | 未发现 |
| Stage3 修改成本公式 | 否 |
| 降低成本后固定策略净效用的方向 | 增加；Stage3 相对 Update 13 的差距仍为负 |
| 降权后重新训练的策略效果 | 未知，不能由事后重算推断 |
| 源码/配置/数据/Checkpoint 修改 | 否 |
