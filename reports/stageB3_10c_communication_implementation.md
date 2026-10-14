# Stage B3-10C：通信模型第一阶段最小实施

## 实施范围

新配置 [`system_config_comm_v2_stageB3_10c.toml`](../configs/proposed/system_config_comm_v2_stageB3_10c.toml) 从当前 `system_config.toml` 复制后独立修改。原正式配置与 `system_config_multi_node_factorized.toml` 未由本任务修改。新配置显式启用 `BANDWIDTH_DEPENDENT_NOISE_V1`；旧配置缺少版本字段时明确解析为 `FIXED_TOTAL_NOISE_V0`，并继续使用原固定总噪声公式。新模式缺少PSD、NF、Ground载频或正交资源池声明时会在配置校验阶段失败，不会静默退回旧模式。

本机的 `conda activate` / `conda run` 入口失败。实际执行使用已注册环境解释器 `C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`，其 `sys.prefix` 为同一环境；Python 3.11.15、NumPy 2.4.6、PyTorch 2.11.0+cu130。未安装或修改依赖。

## 新通信合同

| 参数 | 新配置 | 处理方式 |
|---|---:|---|
| SV / RSU / UAV带宽 | 各10 MHz | 节点各自持有独立正交频域资源池 |
| Ground载频 | 5.9 GHz | 记录在配置中；沿用已包含参考载频效应的 `channel_gain_at_1m`，不重复加自由空间损耗 |
| UAV载频 / 高度 | 2 GHz / 100 m | 保留原LoS/NLoS概率加权平均增益模型 |
| 车辆发射功率 | 0.1995262315 W（23 dBm） | 任务总功率固定，不随带宽份额缩放 |
| 热噪声密度 | −174 dBm/Hz | 转为 `3.9810717055e-21 W/Hz` |
| 接收NF | 9 dB | SV、RSU、UAV统一采用；这是对齐−95 dBm/10 MHz的有效仿真基线，不是硬件实测值 |
| 同频干扰 | 不建模 | 配置明确写为 `ORTHOGONAL_INDEPENDENT` / `NONE`，不将该假设解释为现实无干扰 |

新模式按实际份额计算：

```text
w = b_vm * B_m
N = N0 * w * 10^(NF/10)
SNR = P_v * h_vm / N
R(w) = w * log2(1 + P_v*h_vm / (N0*10^(NF/10)*w))
dR/db = B_m/ln(2) * [ln(1 + A/w) - A/(w+A)]
A = P_v*h_vm / (N0*10^(NF/10))
```

统一入口在 `system_model.uplink_rate_and_derivative()`；上传时延和车辆能耗仍分别为 `D/R` 与 `P_v*D/R`。新模式下零带宽份额返回零速率，KKT仍使用现有正的最小份额边界；正带宽SNR没有人为截断。旧模式保留固定总噪声的线性带宽速率口径。

`task_allocation.allocate_resources_kkt()` 对新模式使用上述精确导数求解带宽KKT：最小化每节点任务的 `Σ q_i/R_i(b_i)`，其中 `q_i = D_i*(ω_t/T_ref + ω_e*P_v/E_ref)`；驻点满足 `q_i*R_i'(b_i)/R_i(b_i)^2 = λ`。沿用原份额下界、`Σb_i≤1`、求解容差及迭代上限；CPU KKT未变。Matching候选评价、理想资源上界及最终Task Utility均通过同一速率实现。Cloud仍为每任务固定200 Mbps等效上行、40 GHz共享CPU池、`cloud_delay_factor=15`，没有改动。

主要实现位置：

- `src/common/system_model.py`：噪声模型版本、PSD换算、共享速率及导数、Edge执行时延/能耗。
- `src/common/project_config.py`：新版本字段的必需项和一致性校验；旧配置缺省保持V0。
- `src/decision/cache_configuration.py`：将新通信配置显式映射到运行时配置并校验。
- `src/decision/task_allocation.py`：Edge候选、完整效用评价与新带宽KKT分支。

## 固定Training状态核验

使用Training realization `episode_20001` 的正式Slot 0（origin 0，time 100 s），采用初始Cache和初始UAV位置，即KEEP/STAY；没有运行完整Episode或PPO。Slot有12个实际任务，17辆相关车辆的位置。配置运行时确认新噪声模式及三个节点角色的10 MHz带宽。

### 噪声、SNR与速率

满带宽10 MHz、NF=9 dB时积分噪声为 **−95.000 dBm**；同链路分配 `b=0.5` 时为 **−98.0103 dBm**。实际选中三个Edge任务均独占其各自节点的带宽池（`b=1`）：

| 任务 | Edge节点 | 分配份额 | 噪声 | SNR | 上行速率 |
|---|---|---:|---:|---:|---:|
| 7 | UAV_0 | 1.0 | −95.000 dBm | 22.282 dB | 74.105 Mbps |
| 33 | UAV_1 | 1.0 | −95.000 dBm | 7.321 dB | 26.773 Mbps |
| 45 | RSU_0 | 1.0 | −95.000 dBm | 2.014 dB | 13.729 Mbps |

用相同三个真实链路增益仅作公式核对，`b=0.5`时速率分别为42.031、17.799、10.317 Mbps。带宽导数与中心有限差分的相对误差约为 `5.4e-11` 至 `1.1e-10`。这是速率/导数一致性核对，不是额外Slot仿真。

### Matching、KKT与Reward

一次正式Slot执行耗时约 **15.203 s**（其中Matching/KKT约15.197 s）。Transfer-Exchange正常结束并通过稳定检查。最终分配为Cloud 9、UAV 2、RSU 1、SV 0。正式Matching内部产生179次KKT求解，179次均有限、均在配置容差内收敛，且没有达到迭代上限。汇总资源残差最大带符号值为 `9.95e-13`；收敛标志表明各次 `|Σb−1|` 均不超过配置容差 `1e-12`。最大驻点残差 `4.44e-16`，最大互补松弛残差 `1.53e-12`。最终三个Edge任务各自的CPU与带宽份额和均为1。

| 指标 | Slot 0结果 |
|---|---:|
| Task Utility `Q` | 11.5107392801 |
| Cache Cost | 0 |
| UAV Movement Cost | 0 |
| 系统Reward `G` | 11.5107392801 |
| 平均任务时延 | 1.8345666672 s |
| 车辆上传总能耗 | 0.4441698321 J |
| 数值有限性 | Q、G、时延及能耗均为有限值 |

Cloud配置读取结果仍为200 Mbps、40 GHz、`ψ=15`；本Slot实际有9项任务使用Cloud。未添加通信硬半径、SNR门槛、干扰、Cloud接入/回传或预测可达性模型。原100/250/300 m候选软范围不变。

另外只对旧配置做了合同核对：没有 `communication.noise.model` 的 `system_config.toml` 加载为 `FIXED_TOTAL_NOISE_V0`；其速率与旧固定总噪声闭式公式在数值精度内相等。没有用旧通信模型重复执行Slot。

## 结论与后续边界

新配置的Ground和UAV均为10 MHz，PSD/NF按KKT带宽份额积分，Matching与KKT的速率口径一致；Episode 20001 Slot 0正式执行可行。此结果验证的是单一Training状态下的软件链路和数值一致性，不代表新通信参数的实测物理精度或策略性能。下一步应先保留旧配置/Checkpoint作为旧模型基线，再由研究设计决定是否继续实施可靠goodput/物理可达性；本轮没有足够依据加入硬SNR门槛。

```ini
STAGE_B3_10C = COMPLETE
CONDA_ENV = adaptive_cache_mec (registered interpreter used directly; conda activation unavailable)
NEW_CONFIG_CREATED = YES
LEGACY_COMMUNICATION_MODE_PRESERVED = YES
GROUND_BANDWIDTH_MHZ = 10
UAV_BANDWIDTH_MHZ = 10
NOISE_PSD_DBM_HZ = -174
RECEIVER_NF_DB = 9
BANDWIDTH_DEPENDENT_NOISE_ACTIVE = YES
KKT_DERIVATIVES_UPDATED = YES
TRAINING_STATE_VERIFIED = YES
MATCHING_KKT_FEASIBLE = YES
SNR_THRESHOLD_ADDED = NO
CLOUD_MODEL_CHANGED = NO
PPO_TRAINING_EXECUTED = NO
FORMAL_CONFIG_CHANGED = NO
RECOMMENDED_NEXT_STEP = Preserve this isolated communication baseline; separately determine whether to add a PHY/QoS-based reliability gate before any strategy retraining.
```
