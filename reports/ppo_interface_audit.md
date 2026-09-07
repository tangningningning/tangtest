# Baseline v2.0 PPO 接口静态适配审计

## 1. 审计范围

本报告只读取当前 `system_config.toml`、当前生成 artifact 和接口实现，检查 service 维度、缓存状态维度、观测向量切片及动作标签。未运行 PPO 训练、完整 SUMO 仿真或长时间实验，也未修改参数、算法和接口代码。

## 2. 当前规模与接口输入

| 项目 | 当前值 | 结论 |
|---|---:|---|
| research vehicles | 25 | 与当前 baseline 一致 |
| SV | 5 | 与当前 baseline 一致 |
| TV | 20 | 与当前 baseline 一致 |
| RSU | 2 | 与当前 node artifact 一致 |
| UAV | 2 | 与当前 node artifact 一致 |
| service | 10 | service 维度基准 |
| 动态缓存节点 | 9 | 5 SV + 2 RSU + 2 UAV |

当前缓存节点由 SV、RSU 和 UAV 组成；TV 不作为动态缓存节点进入 PPO 缓存状态。因此当前缓存节点数为 9，而不是车辆总数 25。

## 3. Observation / state 维度

当前 `ConfigurationObservationSpec` 生成的观测切片如下：

| 切片 | 维度 | 计算来源 |
|---|---:|---|
| cache state | 90 | 9 个动态缓存节点 × 10 个 service |
| UAV position | 4 | 2 个 UAV × 2 个坐标 |
| predicted vehicle | 50 | 25 辆车辆 × 2 个坐标 |
| popularity | 10 | 10 个 service |
| **observation_dim** | **154** | 90 + 4 + 50 + 10 |

结论：代码实际生成的 observation dimension 是 **154**，service 维度为 **10**，与当前 service artifact 和 prediction artifact 一致。

注意：`PARAMETER_CALIBRATION.md` 中仍有 `observation_dim = 164` 的说明。按当前切片计算应为 154，这属于文档算术/同步残留；本次用户要求是静态审计，不修改文档或代码，正式运行前应以代码生成的 154 为准并单独同步该说明。

## 4. Cache state 维度

当前 cache state 形状为：

```text
[9, 10] = [动态缓存节点数, service 数]
```

缓存状态由 `service_catalog["num_services"]` 驱动，当前 service ID 为 `0..9`。缓存初始化和容量约束均按该 service 数构造二维缓存选择矩阵，没有在当前运行输入中发现旧的 20-service cache state。

需要区分历史分析文件与当前运行 artifact：`data/analysis/cache_no_update_explanation.csv` 等历史分析输出仍可能包含旧 service ID（例如到 19），但这些文件不是当前 PPO 缓存状态的输入，不能作为 baseline v2.0 的运行 artifact 使用。

## 5. Action 空间与 action labels

当前代码生成的 action 不是“每个 service 一个独立 cache bit”，而是 trigger-level action：

| 动作类别 | 数量 | 当前含义 |
|---|---:|---|
| cache trigger | 9 | 每个动态缓存节点一个 `z_<node>^ca` |
| UAV movement trigger | 2 | 每个 UAV 一个 `z_<UAV>^q` |
| **action_dim** | **11** | 9 + 2 |

当前 labels 为：

```text
z_23^ca, z_87^ca, z_43^ca, z_75^ca, z_58^ca,
z_RSU_0^ca, z_RSU_1^ca, z_UAV_0^ca, z_UAV_1^ca,
z_UAV_0^q, z_UAV_1^q
```

service 选择本身在缓存优化器内部以 `[9, 10]` 矩阵处理，PPO action 仅触发对应节点的缓存更新。因此：

- 若“cache action 对应 10 个 service”是指缓存状态/内部服务选择维度，则当前接口匹配，service 维度为 10；
- 若该要求是指 PPO action 必须包含 10 个逐 service cache action，则当前接口不匹配，实际是 9 个节点级 cache trigger 加 2 个 UAV movement trigger。

当前 action labels 使用动态节点和 UAV 数量生成，没有发现旧的 20-service action labels。

## 6. 适配结论

| 检查项 | 结果 | 说明 |
|---|---|---|
| service 维度 | 通过 | 当前 catalog、prediction 和接口均为 10 |
| cache state | 通过 | 当前形状为 `[9, 10]` |
| observation 数值维度 | 代码通过，文档需同步 | 代码为 154，文档残留 164 |
| action labels 数量 | 通过 | 当前为 11，随节点/UAV数量生成 |
| 旧 20-service 运行维度 | 通过 | 当前 active code/artifact 未发现；历史分析文件另行保留 |
| “10 个逐 service PPO cache action” | 不通过/需确认语义 | 当前设计是节点级 trigger，不是逐 service action |

## 7. 小规模运行建议

在保持当前 trigger-level action 设计的前提下，可以进入不训练的短时接口/rollout 验证，前提是验证程序采用 `observation_dim=154`，并只读取当前 `data/generated/` artifact。

若现有 harness、checkpoint 或外部说明硬性要求 `observation_dim=164`，或硬性要求 10 个逐 service cache action，则不能直接认定为“完全匹配”；需要先由接口设计者确认语义并同步相应接口契约。本次未做任何修复。

## 8. 代码与数据证据

- `src/decision/cache_configuration.py`：缓存节点集合及 `[M, F]` 缓存状态由当前节点和 `num_services` 动态构造。
- `src/ppo/ppo_trigger_policy.py`：观测切片、观测维度和 trigger action labels 的构造。
- `src/ppo/ppo_rollout.py`：运行时按当前 cache node 数和 `service_catalog["num_services"]` 校验缓存形状。
- `data/generated/service_catalog.json`：当前 service 数量为 10。
- `data/generated/prediction_scenarios.npz`：当前 popularity 预测的 service 轴为 10。
- `data/generated/node_positions.json`：当前 RSU=2、UAV=2。

