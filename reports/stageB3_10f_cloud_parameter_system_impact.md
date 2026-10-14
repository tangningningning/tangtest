# Stage B3-10F：Cloud等效参数与系统效用核验

## 实验合同与固定Training状态

- 执行环境：`adaptive_cache_mec`，解释器 `C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`，`sys.prefix=C:\software\Anaconda_envs\envs\adaptive_cache_mec`；未使用base环境，未安装依赖。
- 两方案都以`configs/proposed/system_config_comm_v2_stageB3_10c.toml`为基础。Edge通信模式为`BANDWIDTH_DEPENDENT_NOISE_V1`，Ground/UAV带宽均为10 MHz，噪声PSD为−174 dBm/Hz、NF为9 dB；Edge通信模型、KKT速率导数、Matching规则及停止条件保持一致。
- 状态固定为Training Episode 20001、Slot 0、t=100.000000s：12个到达任务、17辆相关车辆有位置。两分支独立复制相同的初始Cache、UAV位置、预测场景、车辆位置和任务。Cache保持初始配置，双UAV保持原位置。初始Cache/UAV摘要SHA-256：`8a392cc3ffa0f5dd77c548fe69806572954e15554ae6c5b8e619f9ef5b152507`；任务/车辆状态摘要：`08771058af6cad3fbeb2a1165d4cf0f33c071a8ca93dc73cbb4a7b02f41a0a8c`。
- A：Cloud等效速率200 Mbps、ψ=15。B：B3-10E `global_results.EQUAL_SHARE`的未舍入速率`1633066.716444498 bps`（`1.633066716444498 Mbps`）及ψ=`1.06819321940519`。两方案Cloud CPU均为40 GHz有限共享池，Cloud与合法Edge自由竞争。
- 速率在配置中的重复合同字段同步，以满足原有一致性校验；语义上只改变Cloud等效速率和ψ。未按Cloud任务数再次分摊速率，未增加RSU接入或共享回传模型。Cloud公式为`T=ψD/R + L/F_alloc`、`E_vehicle=P_tx D/R`；车辆能耗不乘ψ。

## 正式Matching/KKT结果

| 方案 | Cloud Mbps | ψ | SV / RSU / UAV / Cloud | Cloud占比 | Q | Cache成本 | UAV成本 | G | 平均/最大任务时延(s) | 车辆总能耗(J) | >1s任务 | Matching停止原因 | KKT收敛/调用 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| A_HISTORICAL | 200 | 15 | 0 / 1 / 2 / 9 | 75.0% | 11.510739280 | 0.000000000 | 0.000000000 | 11.510739280 | 1.834566667 / 2.498223823 | 0.444169832 | 12/12 | converged_by_no_improvement | 179/179 |
| B_EQUAL_SHARE | 1.63306671644 | 1.06819321941 | 2 / 2 / 6 / 2 | 16.7% | 10.498036909 | 0.000000000 | 0.000000000 | 10.498036909 | 4.355966756 / 7.586203536 | 3.660161151 | 12/12 | converged_by_no_improvement | 105/105 |

A、B都重新运行完整的Transfer-Exchange Matching及KKT，没有固定A的最终映射再替换公式。两方案均分配全部12个任务，不可服务任务为0；最终目标有8个任务发生变化。A的任务分配为Cloud 9、UAV 2、RSU 1、SV 0；B为Cloud 2、UAV 6、RSU 2、SV 2。两次Matching都以`converged_by_no_improvement`稳定停止，未达到更新上限。

资源及数值核对：A为179/179次KKT收敛且有限，B为105/105次收敛且有限；两者迭代上限命中均为0。最大资源残差A=`9.954e-13`、B=`9.781e-13`；最大驻点残差分别为`4.441e-16`与`4.441e-16`，最大互补松弛残差分别为`1.527e-12`与`1.527e-12`。Cloud CPU分别使用40.000 GHz和约`39.999999999983` GHz，不超过40 GHz；Edge节点CPU和带宽份额和均不超过1。全部任务时延、车辆能耗、Task Utility及系统效用均有限。

B−A差值：

- Task Utility `ΔQ = -1.012702371`；Cache成本`ΔC_cache = +0.000000000`；UAV成本`ΔC_UAV = +0.000000000`；系统效用`ΔG = -1.012702371`。
- 总/平均/最大任务时延变化：`+30.256801063s` / `+2.521400089s` / `+5.087979714s`。
- 总/平均车辆能耗变化：`+3.215991319J` / `+0.267999277J`。
- 超过1秒的任务数：A `12/12`，B `12/12`。此处“超过1秒”仅是当前Slot模型输出，不作为失败判定。

两边`C_cache=C_UAV=0`，且`G=Q−C_cache−C_UAV`逐分支通过。因此本状态`ΔG=ΔQ`，差异完全来自重新求解后的任务效用；没有重复扣除Cache或UAV成本。B3-10C历史Slot 0基线为`Q=11.5107392801`、平均时延`1.8345666672s`、车辆总能耗`0.4441698321J`、179次KKT；当前A复现这些数值至报告精度。

## Cloud任务通信和车辆能耗

下表的Cloud通信时延只包含等效上传项`ψD/R`；总服务时延还包括实际分配的Cloud CPU计算时间。车辆能耗为`P_tx D/R`。逐任务结果由正式Matching/KKT分配后核对，A/B的公式最大绝对误差均为0。

### A_HISTORICAL
| Task | Service | Cloud通信时延(s) | 总服务时延(s) | 车辆发送能耗(J) | Task Utility |
|---|---:|---:|---:|---:|---:|
| 1 | 8 | 0.688297431 | 1.865975018 | 0.009155560 | 0.962251206 |
| 15 | 8 | 0.696723814 | 1.894209583 | 0.009267645 | 0.961683473 |
| 25 | 3 | 0.842672391 | 2.021593606 | 0.011209016 | 0.958934577 |
| 31 | 8 | 0.547559505 | 1.635642158 | 0.007283499 | 0.967007745 |
| 36 | 8 | 0.543925706 | 1.579718068 | 0.007235163 | 0.968109570 |
| 37 | 7 | 0.929597962 | 2.064705626 | 0.012365279 | 0.957939041 |
| 44 | 1 | 0.790376024 | 2.148157147 | 0.010513383 | 0.956546134 |
| 50 | 8 | 0.600647095 | 1.727250027 | 0.007989657 | 0.965120991 |
| 51 | 0 | 0.309465851 | 1.172706880 | 0.004116437 | 0.976489921 |

### B_EQUAL_SHARE
| Task | Service | Cloud通信时延(s) | 总服务时延(s) | 车辆发送能耗(J) | Task Utility |
|---|---:|---:|---:|---:|---:|
| 25 | 3 | 7.349256258 | 7.586203536 | 1.372756706 | 0.672116171 |
| 51 | 0 | 2.698965655 | 2.872465470 | 0.504135803 | 0.877892706 |


## 影响解释与证据边界

- Cloud执行比例从75.0%降至16.7%，Edge任务数从3增至10；8/12个任务改变目标。这说明新参数确实进入了候选效用并改变了Matching决策，不是只改了结果展示。
- A与B的Cloud通信时延系数`ψ/R`分别为`7.500000e-08s/bit`和`6.541026e-07s/bit`，B约为A的`8.72`倍；车辆能耗系数`1/R`则约为A的`122.47`倍。ψ降低并没有抵消速率大幅降低的影响。B仅剩2个Cloud任务，但这两项任务的平均等效Cloud通信时延为`5.024111s`，平均车辆发送能耗为`0.938446J`；对应A的9个Cloud任务分别为`0.661030s`和`0.008793J`。任务集合不同，因此按Cloud子集计算的均值不是逐任务配对因果差值。
- 尽管更多任务转到Edge，B的Task Utility较A低`1.012702`，总时延增加`30.256801s`，车辆总能耗增加`3.215991J`。当前证据说明本Slot下B参数造成的Cloud等效通信负担和随之改变的资源/路径分配，降低了系统效用；不能据此断言低Edge占比就是算法Bug，也不能说Edge占比上升代表模型更合理。
- Cache和UAV动作未变化，因此未直接测量Cache/UAV动作边际价值。Cloud效用改变可能重新分配Cache服务和UAV移动的相对价值，但需要另外固定状态、改变动作的受控实验才能定量判断。
- 本Slot没有观察到负的单任务Utility：A/B负Utility任务均为0，B最低Task Utility=`0.672116`。这与当前不截断线性归一化公式一致；不把Q下降解释为数值错误。
- B3-10E EQUAL_SHARE来自300个Training Episode到达任务的离线拟合，参考负载是关联RSU的全部到达任务而非Matching最终Cloud子集；等份无线/回传和每RSU 200 Mbps回传是预设假设。候选值是条件化仿真等效参数，不是设备实测速率或无线可靠性校准。本报告只评价一个Training Slot，没有把结论外推到300集。

## 执行兼容性说明

本地`slot_environment._matching_summary`仍引用当前`task_allocation`/`CacheOptimizationConfig`中不存在的旧RSU-Cloud目标辅助函数和字段。未修源码。未经适配的第一次基线执行在Matching/KKT完成后于结果打包阶段抛错；正式求解数值随后通过窄范围的**临时内存结果汇总适配器**重新运行并取得。该适配器仅读取正式Matching结果已有字段并按现有`CLOUD`常量归类，不参与目标生成、资源分配、KKT、Task Utility或Reward。源文件和配置文件均未改；建议将此残留汇总接口问题作为后续独立的小修复。

## 结论

本任务的单状态A/B效用核验完成，两个方案Matching/KKT均可行。B在本状态中显著降低Cloud卸载比例，却令Task Utility/System Utility下降；因此不建议仅凭该样本冻结B3-10E参数。可将它作为独立仿真假设继续保留，按预先固定的少量Training状态评估跨场景敏感性，再决定是否冻结；不应重新拟合或依据目标Return调整参数。

```ini
STAGE_B3_10F = COMPLETE
CONDA_ENV = adaptive_cache_mec
TRAINING_STATES_USED = 20001:0
EDGE_COMMUNICATION_MODE = BANDWIDTH_DEPENDENT_NOISE_V1
CLOUD_BASELINE_RATE_MBPS = 200
CLOUD_BASELINE_DELAY_FACTOR = 15
CLOUD_CANDIDATE_RATE_MBPS = 1.633066716444498
CLOUD_CANDIDATE_DELAY_FACTOR = 1.06819321940519
MATCHING_KKT_FEASIBLE_A = YES
MATCHING_KKT_FEASIBLE_B = YES
CLOUD_EXECUTION_FRACTION_A = 0.75
CLOUD_EXECUTION_FRACTION_B = 0.1666666666666667
TASK_UTILITY_A = 11.51073928010853
TASK_UTILITY_B = 10.49803690863657
SYSTEM_UTILITY_A = 11.51073928010853
SYSTEM_UTILITY_B = 10.49803690863657
TASKS_DELAY_ABOVE_ONE_SECOND_A = 12
TASKS_DELAY_ABOVE_ONE_SECOND_B = 12
CLOUD_PARAMETER_FREEZE_RECOMMENDATION = NO_FROM_SINGLE_STATE
PPO_TRAINING_EXECUTED = NO
SYSTEM_MODEL_CHANGED = NO
FORMAL_CONFIG_CHANGED = NO
RECOMMENDED_NEXT_STEP = Use a small predeclared set of additional Training states to assess the isolated Cloud candidate before freezing; do not refit from returns.
```
