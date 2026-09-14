# Artifact 一致性报告

本报告由 `scripts/generation/regenerate_artifacts.py` 根据 `configs/system_config.toml` 生成。
角色重建复用既有 `trajectory.csv`，未启动 SUMO；未运行 PPO 或完整实验。

## 当前配置摘要

- 车辆：research=25，SV=5，TV=20
- 基础设施：RSU=2，UAV=2
- service.count=10
- input size 基准范围=500.0--5000.0 KB
- cycles_per_bit=400.0--1000.0
- service size=2.0--20.0 GB

## 服务目录与缓存容量

- service catalog 总大小：111631.668831 MB

| 节点 | 容量 (MB) | catalog 比例 |
|---|---:|---:|
| TV | 5581.583442 | 5.00% |
| SV | 16744.750325 | 15.00% |
| RSU | 44652.667533 | 40.00% |
| UAV | 33489.500649 | 30.00% |

## 执行阶段

| 阶段 | 状态 |
|---|---|
| vehicle_roles | 跳过（与配置一致） |
| node_positions | 已调用 node_placement.py 重生成 |
| service_catalog | 跳过（与 metadata 一致） |
| tasks | 跳过（与 metadata 一致） |
| tv_local_cache | 跳过（与 metadata 一致） |
| popularity | 跳过（与 metadata 一致） |
| prediction | 跳过（与 metadata 一致） |

## 一致性检查

| 检查项 | 结果 |
|---|---|
| system_config 可解析且车辆约束成立 | 通过 |
| vehicle_roles 数量与配置一致 | 通过 |
| service_catalog 数量与范围一致 | 通过 |
| tasks.csv TV 集合和时隙覆盖一致 | 通过 |
| TV 初始缓存容量约束成立 | 通过 |
| RSU/UAV 节点数量一致 | 通过 |
| 预测 artifact 维度一致 | 通过 |

## 依赖关系

- service 参数 -> service_catalog、popularity、tasks、TV 初始缓存、预测场景
- vehicle 参数/角色集合 -> vehicle_roles、tasks、TV 初始缓存、vehicle_predictions、prediction_scenarios
- task 参数 -> popularity、tasks、GRU 预测输入和 prediction_scenarios
- cache 容量比例 -> configs/system_config.toml 中的派生 MB 容量和 TV 初始缓存
- RSU/UAV 数量 -> node_positions.json；位置仍由 node_placement.py 道路感知生成
