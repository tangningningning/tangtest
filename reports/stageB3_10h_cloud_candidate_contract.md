# Stage B3-10H：Cloud候选通信配置隔离与系统模型边界确认

## 范围与执行环境

本阶段根据B3-10E至B3-10G已有结果，为EQUAL_SHARE等效Cloud参数建立独立配置，并核对配置加载、运行时参数映射和适用边界。没有重新拟合参数，没有运行Matching/KKT，也没有进行PPO训练。

本机`conda activate adaptive_cache_mec`入口返回“系统无法执行指定程序”。随后使用已确认的环境解释器`C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`；运行时`sys.prefix`为`C:\software\Anaconda_envs\envs\adaptive_cache_mec`。未使用base环境，未安装依赖。

## 隔离配置

新配置[system_config_comm_v2_stageB3_10h_equal_share.toml](../configs/proposed/system_config_comm_v2_stageB3_10h_equal_share.toml)从B3-10C配置逐字复制后，仅修改Cloud等效通信参数、这些参数的说明及其`parameter_status`。对解析后的配置做递归比较，全部差异如下：

| 配置键 | B3-10C | B3-10H | 说明 |
|---|---:|---:|---|
| `nodes.cloud.uplink_rate_bps` | 200,000,000 | 1,633,066.7164444982 | 每个Cloud任务固定使用的等效速率 |
| `nodes.cloud.total_bandwidth_bps` | 200,000,000 | 1,633,066.7164444982 | 兼容接口重复字段，与上项保持相同，不表示共享速率池 |
| `communication.cloud.uplink_rate_bps` | 200,000,000 | 1,633,066.7164444982 | 通信参数映射重复字段 |
| `nodes.cloud.cloud_delay_factor` | 15.0 | 1.0681932194051897 | 对应`psi_cloud` |
| 上述四个速率/系数的`parameter_status` | `CONFIRMED_DATA` | `TRAIN_ONLY_CALIBRATION` | 候选值来自Training数据条件拟合 |

其他解析后配置值与B3-10C一致。尤其是Cloud有限共享CPU池40 GHz、`resource_sharing=false`、`RESOURCE_AWARE_FALLBACK`候选规则、Edge通信、Reward和PPO配置均未改变。配置文件SHA-256：

- B3-10C来源配置：`1a9dfdf8ecee9a838a0a9a5fcde519cc7580c8ffa2d60e6526bfa0451dea6e78`
- B3-10H隔离配置：`d0cfc9d487a6fbb335d97e50c6fb6ac6724636cbf8b9493a0023636953777655`
- 旧正式配置`configs/proposed/system_config.toml`本任务后核对的SHA-256：`f3131efb7832efa925c181d9abd07bec5f2bfe3e7507eca2cc2515b77067699b`；本任务没有写入该文件。

旧正式配置、B3-10C配置和既有历史结果均保留；没有更改默认Runner或切换配置。

## Cloud等效合同与参数来源

```text
R_cloud = 1,633,066.7164444982 bit/s
psi_cloud = 1.0681932194051897
F_cloud_total = 40 GHz，有限共享CPU池

T_cloud = psi_cloud * D_bits / R_cloud + L_cycles / F_cloud_alloc
E_cloud_vehicle = P_tx * D_bits / R_cloud
```

`R_cloud`与`psi_cloud`是B3-10E依据300个Training Episode的等份RSU无线接入和每RSU 200 Mbps共享回传参考假设得到的全池等效拟合值。新配置在Matching运行时将其作为每Cloud任务的固定等效链路参数，不会再除以Cloud任务数。`psi_cloud`只影响传输时延，不进入车辆发送能耗。Cloud使用有限共享40 GHz CPU；Cloud与所有当前合法Edge候选自由参与效用竞争。

该设置没有实现`TV→RSU→Cloud`链路，也没有添加RSU接入带宽或共享回传资源。EQUAL_SHARE中的等份无线和回传是拟合参考假设，不等同于已测量的设备吞吐或物理部署合同。B3-10G中该候选在12个固定Training状态的A/B配对里两分支均为12/12 KKT可行，但候选参数分支的ΔG在12个状态全部为负，平均`ΔG=-1.7281854172`。因此它是可复现的独立仿真候选，不是已经证明优于历史参数的模型。

## Edge模型保持

新配置继续使用`BANDWIDTH_DEPENDENT_NOISE_V1`：SV、RSU和UAV系统带宽均为10 MHz，噪声谱密度为−174 dBm/Hz，有效接收NF为9 dB；KKT按实际带宽分配计算Edge噪声、速率及相应导数。地面与空地信道、CPU/带宽约束、Cache/UAV成本、Task Utility及Reward均与B3-10C相同。Cloud仍使用独立的等效速率公式，不套用Edge动态噪声公式。

## 配置与运行时合同核验

使用项目注册的`adaptive_cache_mec`解释器进行最小配置检查：

- 官方`load_config(..., require_artifacts=True)`分别成功加载B3-10C与B3-10H。
- `validate_for_formal_experiment()`对两份配置均通过。
- `cache_config_from_project()`映射出的B3-10H Cloud速率及重复字段均为`1633066.7164444982`，`cloud_delay_factor`为`1.0681932194051897`。
- 新配置运行时保持`finite_pool`、40 GHz Cloud CPU、每任务独立等效上行以及`RESOURCE_AWARE_FALLBACK`。
- Edge噪声模式、10 MHz带宽、−174 dBm/Hz、9 dB NF以及1秒Slot和`independent_per_slot`合同均与来源配置一致。

此处只验证配置和运行时映射，不代表重新执行了B3-10G Matching/KKT。已有证据是B3-10G的12个固定状态中，A/B两分支均各有12/12状态通过KKT及资源可行性检查。

## 1秒Slot与任务服务时延

配置设`simulation.dt_s=1.0`，并要求`execution.cross_slot_task_model="independent_per_slot"`。`execute_slot()`每次接收当前Slot的任务列表并完成当次Matching/KKT与效用计算；配置校验也明确限制为独立时隙模型。代码没有保存未完成任务、跨Slot排队或持续占用资源的状态，也没有发现由Slot长度触发的硬截止、时延截断、丢弃或超时惩罚。

因此服务时延超过1秒时，当前模型仍将其作为该Slot任务的静态效用评价值；这不等于任务已在1秒内完成，也不表示任务会跨Slot继续执行。这个解释符合现行独立时隙简化合同，但不能作为真实排队系统或硬实时完成保证。B3-10G中历史Cloud与EQUAL_SHARE分支分别有159/164和161/164个任务的时延超过1秒，说明这一简化对两种参数均有影响，后续论文表述应明确揭示。

## 无线物理可靠性边界

Edge速率目前由确定性平均信道增益、功率、噪声和分配带宽按数学速率公式计算。该模型可以判断给定输入下速率是否为有限正数；它没有经通信制式校准的可靠性门槛、outage/BLER或分组成功率模型。B3-10E中88.87%的参考RSU关联任务距离超过250米，而250米是预测/候选代理尺度，不是正式硬覆盖半径。

所以`MATHEMATICAL_RATE_FEASIBILITY`有模型内计算依据，`PHYSICAL_LINK_RELIABILITY`尚未验证。KKT可行不等于真实无线覆盖或链路可靠性已被证明。本阶段没有增加SNR阈值或距离门槛。

## 后续实验准备判断

| 判断对象 | 结论 | 依据与边界 |
|---|---|---|
| 独立仿真候选可执行 | YES | 新配置通过官方加载与正式配置检查；同一候选在B3-10G既有12个Training固定状态中完成了Matching/KKT。 |
| 可作为物理可靠性已验证的系统模型 | NO | 没有校准覆盖/可靠接入条件；平均增益下的正速率不提供可靠性证据。 |
| 可替换历史正式基线 | NO | B3-10G候选分支12/12状态ΔG为负；且新文件明确是独立候选，没有切换正式配置。 |
| 可开展独立PPO训练 | CONDITIONAL | 数值和配置链路具备条件；训练应使用独立Run及与配置策略Schema匹配的权重/初始化。此文件继承B3-10C的单节点v3策略Schema，不能据此假定B3-6的Architecture 33 / 多节点v4 Checkpoint可直接加载。训练前需固定研究目的和兼容策略合同。 |

推荐先把EQUAL_SHARE作为单独Cloud配置对照保留。若下一步研究目标是该候选下的策略训练，应为它建立独立Run/Checkpoint谱系并按本配置的策略Schema启动；不要覆盖B3-10C或历史正式基线。若目标是论文中的物理系统结论，应先单独补足无线可靠性依据及超过1秒服务时延的系统语义，再决定是否需要修改系统模型。

```ini
STAGE_B3_10H = COMPLETE
CONDA_ENV = adaptive_cache_mec (registered environment interpreter; activation entry unavailable)
CLOUD_MODEL = ABSTRACT_EQUIVALENT_LINK
CLOUD_CANDIDATE_RATE_BPS = 1633066.7164444982
CLOUD_CANDIDATE_DELAY_FACTOR = 1.0681932194051897
CLOUD_CPU_GHZ = 40
CLOUD_POLICY = FREE_COMPETITION
EDGE_MODEL = BANDWIDTH_DEPENDENT_NOISE_V1
NEW_ISOLATED_CONFIG_CREATED = YES
HISTORICAL_CONFIG_PRESERVED = YES
FORMAL_CONFIG_PRESERVED = YES
RUNTIME_CONFIG_CHECK = PASS
CROSS_SLOT_TASK_CONTINUATION = NO
ONE_SECOND_SLOT_SEMANTICS = static independent-slot utility evaluation; not a hard completion deadline
PHYSICAL_LINK_RELIABILITY_VALIDATED = NO
MATCHING_KKT_PRIOR_VALIDATION = 12/12 per branch
INDEPENDENT_SIMULATION_READY = YES
FORMAL_PPO_TRAINING_READY = CONDITIONAL
PPO_TRAINING_EXECUTED = NO
SYSTEM_MODEL_CHANGED = NO (formal model unchanged; isolated candidate has distinct Cloud parameters)
RECOMMENDED_NEXT_STEP = Keep EQUAL_SHARE isolated; if training it, use a separate run and compatible policy schema; resolve reliability and >1s delay semantics before physical-system claims.
```
