# V11A_FAILURE_AND_IO_ANALYSIS

审计方式：只读取已有四个 V11-A formal run 的 manifest、summary、episode/update JSONL、slot/critic JSONL 与当前源码；没有重新 rollout、训练或评估。

## 结论先行

四个 run 均完成 300 epochs，均无 summary 级 NaN/Inf。5e-4 是当前稳定对照；6e-4 是需要保留为 failure witness 的 run：

| actor LR | final-20 mean return | final-20 global critic EV | final-20 Q-cache EV | final-20 UAV movement | 结论 |
|---:|---:|---:|---:|---:|---|
| 5e-4 | 2335.0833 | 0.9971 | 0.9990 | 0.1669 | 稳定 |
| 6e-4 | 240.7147 | 0.2270 | 0.4871 | 0.9590 | 晚期 global critic/Q-cache 崩溃，UAV MOVE 极化 |
| 5e-3 | 1652.9829 | 0.9924 | 0.9983 | 0.5000 | 回报较低但 critic/Q 未晚期崩溃 |
| 6e-3 | -3077.6220 | 0.9790 | 0.9954 | 0.5000 | 回报失败，但不是 value/Q 崩溃型 failure |

6e-4 的持续窗口证据：从 epoch 193 起，global critic EV 的 20-epoch trailing mean 首次低于 0.5；Q-cache EV 的 20-epoch trailing mean 到 epoch 300 才首次低于 0.5，epoch 300 的单点分别为 0.0052 与 -0.0084。这是从现有指标得到的相关性证据，不把 actor LR 直接证明为唯一因果。

## 已确认的 IO / reward 链

V11-A 的静态执行链为：

1. 400 维 observation 只由 previous cache、previous UAV position、vehicle/popularity prediction horizon、horizon mask、time-to-go 组成；
2. primitive controller 按固定 9 Cache node + 2 UAV head 顺序生成合法 operation/candidate；
3. policy primitive override 直接给出 final cache/UAV state，并跳过 prediction configuration solve；
4. execute_slot 用真实当前 tasks/vehicle positions 做 Matching；
5. edge task 资源由 KKT allocation 求得，LOCAL/CLOUD/edge 的 delay/energy 再由 system_model 计算；
6. 当前 reward 为 G(t)=Q(t)-C_ca(t)-C_q(t)；
7. terminal origin 为 299，formal episode 没有额外 bootstrap transition。

因此在现有代码路径中没有发现“真实当前 task/position 先进入 policy observation，再被误当成 prediction”的直接 IO 泄漏证据。真实 task/position 确实会在 action 之后进入 Matching/KKT，这是 reward 计算所需的数据流，不是 observation 输入。

## GAE、PopArt 与 Q-cache credit 的实际语义

- compute_gae() 对 reward、old value、done、bootstrap value 使用标准 GAE；formal terminal 的 bootstrap 为 0。
- PPO global advantage 依配置做 centered_std 归一化。
- V11-A Q-cache target 是当前 PopArt 统计下的 normalized raw GAE return；每个 rollout 做 40 个 Q-cache optimizer steps（4 epochs × 10 minibatches）。
- Q-cache 以 400-d observation、target Cache node one-hot、完整 1055-d joint action encoding 为输入；每个 target node 的 counterfactual 只替换该 node action slice。
- cf_adv_raw 是 detached selected-Q 减去 legal-action policy expectation；日志中 normalized CF std 约为 1 是归一化设计，不等于 Q function 质量。

## 正式日志中的 Q / CF / KL 对照

| actor LR | Q EV 范围 | final-20 Q EV | final-20 CF raw std | final-20 Cache KL | final-20 Cache minibatches executed/suppressed |
|---:|---:|---:|---:|---:|---:|
| 5e-4 | 0.5219..0.9997 | 0.9990 | 0.004180 | 0.07056 | 23.4 / 16.6 |
| 6e-4 | -0.0084..0.9982 | 0.4871 | 0.002093 | 0.07307 | 22.4 / 17.6 |
| 5e-3 | 0.2103..0.9997 | 0.9983 | 0.002885 | 0.13044 | 9.7 / 30.3 |
| 6e-3 | 0.2953..0.9987 | 0.9954 | 0.005461 | 0.12233 | 7.85 / 32.15 |

Cache KL control 是 minibatch_early_stop，target KL 为 0.1。6e-4 的 Cache KL 并非全程失控：全 run 超过 0.1 的 update 为 46 次，最大 0.2395，且 late-20 mean 为 0.0731。5e-4 甚至有 38 次超过 0.1，但 Q/critic 没有同样的晚期崩溃。因此现有日志不支持把“Cache KL early stop 本身”单独定为 6e-4 failure 的充分原因。

6e-4 的更直接异常是：

- global critic EV late-20 mean 从高质量区间跌到 0.2270，epoch 300 为 0.0052；
- Q-cache EV late-20 mean 0.4871，epoch 300 为 -0.0084；
- final-20 actual UAV movement rate 0.9590，且 episode 300 为 0.975；
- final-20 reward 仍只有 240.7，显著低于 5e-4 的 2335.1；
- final epoch CF sign 在 RSU/UAV-cache 上强烈偏正（分别 0.9833 / 0.9983），而 SV CF std 仍为 1.2704，说明 node-level credit 的分布已高度不均匀。

## 静态 mask / operation / template 统计

### Cache operation（正式 300 epochs 汇总，按 role）

这些是已选中的 operation counts/rates，不是“所有合法 mask 的支持率”；源码生成完整合法 mask，但正式 episode/update JSONL 没有保存完整 Cache operation/candidate mask 张量。

| actor LR | role | KEEP | ADD | DELETE | SWAP |
|---:|---|---:|---:|---:|---:|
| 5e-4 | SV | 0.390 | 0.108 | 0.107 | 0.394 |
| 5e-4 | RSU | 0.443 | 0.009 | 0.013 | 0.535 |
| 5e-4 | UAV-cache | 0.560 | 0.010 | 0.010 | 0.420 |
| 6e-4 | SV | 0.114 | 0.328 | 0.329 | 0.229 |
| 6e-4 | RSU | 0.504 | 0.231 | 0.239 | 0.026 |
| 6e-4 | UAV-cache | 0.497 | 0.247 | 0.247 | 0.008 |
| 5e-3 | SV | 0.203 | 0.247 | 0.247 | 0.303 |
| 5e-3 | RSU | 0.963 | 0.008 | 0.008 | 0.021 |
| 5e-3 | UAV-cache | 0.914 | 0.039 | 0.040 | 0.007 |
| 6e-3 | SV | 0.301 | 0.232 | 0.230 | 0.237 |
| 6e-3 | RSU | 0.573 | 0.211 | 0.211 | 0.005 |
| 6e-3 | UAV-cache | 0.462 | 0.151 | 0.156 | 0.231 |

6e-4 的 selected operation template 明显偏向 SV add/delete、RSU add/delete、UAV-cache add/delete；5e-4 则保留了较多 SV/RSU/UAV-cache swap。这个差异是行为结果，不是 mask 约束本身的证明。

### UAV operation mask

四个正式 run 的持久化摘要一致显示：

- feasible operation count：mean/min/max 均为 2.0 / 2 / 2
- MOVE feasible rate：1.0
- move candidate count 的跨 epoch mean 约为 78.20（5e-4）、78.53（6e-4）、79.40（5e-3）、78.48（6e-3）
- padded max candidate 为 89；实际 MOVE candidate 约 33..80/每个 transition

所以在已有 trajectory 上没有看到“UAV MOVE 因 mask 被长期禁用”的证据。6e-4 的问题更像 policy/credit/value 共同漂移到 MOVE-heavy 状态，而不是可行 MOVE 集合消失。

### template evidence 的边界

日志保留了每个 epoch 的 role-level operation counts、Cache trigger bitmask histogram，以及一个固定 seed 抽取的 cf_adv_fixed_transition（含 node_index/head_id/operation/selected index/Q selected/Q counterfactual baseline）。这足以支撑当前 failure report；但不能从现有 JSONL 反推出每个 Cache node、每个 slot 的完整 legal candidate mask 频率。报告因此不把“selected operation rate”误报成“legal mask rate”。

## failure interpretation

当前最稳妥的结论是：

1. observation/actual-reward IO contract 静态一致，未发现明显的真实 future label 进入 observation；
2. legal primitive mask 在源码中是动态、逐 head、按容量/安全距离生成的；UAV 日志显示 MOVE 在 formal 数据上始终可行；
3. 5e-4 与 6e-4 的 critic/Q 结果分叉，6e-4 出现晚期 global critic 与 Q-cache 同步失效；
4. 6e-4 的 Cache KL early-stop 与 node-level CF sign skew 是强相关诊断信号，但 5e-3/6e-3 的对照说明它们都不足以单独解释所有回报 failure；
5. 6e-3 是“回报坏但 value/Q 仍高”的行为 failure witness，不能把所有失败归因于 critic collapse；
6. 没有训练前提下，不应进一步改 LR、reward、Q、KL、entropy 或 system 参数；后续若要继续，必须基于这份失败证据另行授权新的实验设计。

## 证据定位

- reports/main_experiment_io_contract.md/.json
- runs/ppo_v11a_cache_node_cf_5e4_formal/run_summary.json
- runs/ppo_v11a_cache_node_cf_6e4_formal/run_summary.json
- 四个 formal run 的 episode_metrics.jsonl 与 update_metrics.jsonl（清理前已读取并把关键统计压缩到本报告）
- src/ppo_simple/trainer.py:892-921,1083-1220,2278-2333,2618-2915,3085-3450
- src/ppo_simple/primitive_control.py:667-954,1012-1315
- src/decision/cache_configuration.py:1395-1475
- src/decision/slot_environment.py:192-575,784-870

