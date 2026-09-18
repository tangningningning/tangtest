# V11A_CACHE_NODE_KL_DECOMPOSITION_GATE

本 gate 只做 Cache branch KL 定位诊断，没有执行正式训练、第二个独立 rollout 或 final seed。诊断使用 `episode_20001`、seed `4401`、300 slots 和 V11-A 6e-4 配置；由于上一份 readiness 报告没有保存 centralized Q 权重与逐 transition log-prob 张量，使用同一 realization/seed 重构了一个 frozen rollout，先恢复 bounded Gate-A Q 状态，再冻结 Q，仅执行首个 Cache PPO epoch。

## KL 定义与聚合验证

`NODE_APPROX_KL_m` 使用与当前 branch KL 相同的 sample-consistent estimator：

```text
mean_t(old_node_log_prob[t,m] - new_node_log_prob[t,m])
```

Cache branch 使用所有 9 个 node 的 sampled log-ratio 之和。因此：

```text
CACHE_BRANCH_APPROX_KL = 0.7408211694161098
SUM_NODE_APPROX_KL      = 0.7408211694161098
KL_DECOMPOSITION_ERROR  = 0.0
```

线性分解严格成立，不需要用另一种 estimator 替代。另计算了 exact distribution KL 作为辅助诊断；其 operation/candidate 分解误差最大为 `6.21e-08`，但最终 branch 数值仍以 sample-consistent estimator 为准。

## Node KL 与 policy change

| node | role | NODE_APPROX_KL | old entropy | new entropy | old max p | new max p | mean abs logprob Δ | operation KL | candidate KL | CF_ADV_STD | mean \|A_cf\| |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SV_0 | SV | 0.130559 | 1.899164 | 2.179860 | 0.360868 | 0.223662 | 0.524784 | 0.130703 | -0.000145 | 1.192549 | 1.051096 |
| SV_1 | SV | 0.074699 | 1.880448 | 2.120339 | 0.352542 | 0.255592 | 0.411326 | 0.074291 | 0.000408 | 1.396595 | 1.259322 |
| SV_2 | SV | 0.120351 | 1.865679 | 2.117385 | 0.360476 | 0.250041 | 0.464957 | 0.123043 | -0.002692 | 1.143471 | 0.974169 |
| SV_3 | SV | 0.069720 | 1.935133 | 1.784275 | 0.330499 | 0.408151 | 0.399137 | 0.069753 | -0.000033 | 0.674115 | 0.540909 |
| SV_4 | SV | 0.091521 | 1.839921 | 1.496614 | 0.400145 | 0.555515 | 0.373357 | 0.090014 | 0.001507 | 1.089908 | 0.880199 |
| RSU_0 | RSU | 0.046700 | 2.514445 | 2.270970 | 0.283956 | 0.373342 | 0.222046 | 0.043203 | 0.003497 | 0.827063 | 0.653162 |
| RSU_1 | RSU | 0.057530 | 2.432608 | 2.348406 | 0.314585 | 0.357807 | 0.314545 | 0.059206 | -0.001676 | 1.008827 | 0.868960 |
| UAV_CACHE_0 | UAV-cache | 0.045889 | 2.310304 | 2.564968 | 0.319448 | 0.231984 | 0.347984 | 0.045745 | 0.000144 | 0.509958 | 0.401724 |
| UAV_CACHE_1 | UAV-cache | 0.103852 | 2.177593 | 2.440174 | 0.376094 | 0.272441 | 0.378302 | 0.104466 | -0.000613 | 0.834542 | 0.700793 |

按 node 平均的类型汇总：

| type | CF_ADV_STD（node mean） | mean \|A_cf\|（node mean） | NODE_KL_MEAN |
|---|---:|---:|---:|
| SV | 1.099328 | 0.941139 | 0.097370 |
| RSU | 0.917945 | 0.761061 | 0.052115 |
| UAV-cache | 0.672250 | 0.551259 | 0.074871 |

高 KL 更偏向 SV，且与 node-level `CF_ADV_STD` / `mean |A_cf|` 呈中等描述性相关（Pearson 分别约 `0.534` / `0.528`）；这只是描述性对齐，不作因果结论。

## 判定

所有 node KL 都为中等正值，范围 `0.045889–0.130559`；top-1 `SV_0` 仅占 branch KL `17.62%`，top-2 仅占 `33.87%`，没有 1–2 个 node 贡献绝大多数 KL。因此 node 层面表现为 distributed accumulation，而 component 层面几乎全部来自 operation head：operation sampled contribution `0.74042416`，candidate sampled contribution `0.00039701`。

本阶段结论是 `OPERATION_HEAD_UPDATE`，不是 `CONCENTRATED_NODE_INSTABILITY`。若后续仍要保持旧 joint Cache KL 约 `0.1`，当前 9 个有效 node 的平均 node KL 参考量级约为 `0.1 / 9 = 0.0111111`；这只是诊断参考，不是新 target。

本阶段不实现 node-specific KL control；下一方向保持 `WAIT`，待 operation-head update 机制另行定位。

## Final fields

```text
STAGE = V11A_CACHE_NODE_KL_DECOMPOSITION_GATE
TRAINING_EXECUTED = NO
FORMAL_TRAINING_EXECUTED = NO

CACHE_BRANCH_APPROX_KL = 0.7408211694161098
SUM_NODE_APPROX_KL = 0.7408211694161098
KL_DECOMPOSITION_ERROR = 0.0

NODE_KL_MEAN = 0.08231346326845664
NODE_KL_MEDIAN = 0.07469855904579163
NODE_KL_MAX = 0.13055875519911447
NODE_KL_MIN = 0.045889240503311154

SV_KL_MEAN = 0.0973698074221611
RSU_KL_MEAN = 0.05211521382133166
UAV_CACHE_KL_MEAN = 0.07487085233132044

TOP1_NODE = SV_0
TOP1_NODE_KL = 0.13055875519911447
TOP1_KL_CONTRIBUTION = 0.17623518412954706

TOP2_KL_CONTRIBUTION = 0.3386908708383544

OPERATION_KL_CONTRIBUTION = 0.9994640954273796
CANDIDATE_KL_CONTRIBUTION = 0.0005359032316614328

CACHE_KL_EXCESS_CAUSE = OPERATION_HEAD_UPDATE

NODE_GRANULAR_KL_CONTROL = WAIT

FORMAL_300_EPOCH_RECOMMENDED = WAIT

SYSTEM_MODEL_CHANGED = NO
GLOBAL_REWARD_CHANGED = NO
Q_CACHE_CHANGED = NO
UAV_CHANGED = NO

GIT_DIFF_CHECK = PASS: diagnostic-only runtime override; production algorithm/config unchanged
TEMP_FILES_REMOVED = YES
PUSHED = NO
```

