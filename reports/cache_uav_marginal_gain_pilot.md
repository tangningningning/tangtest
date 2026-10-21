# Training 场景下 Cache/UAV 真实边际收益 Pilot

## 范围与状态来源

- 只读正式 Training Pool；未访问 Validation/Test，未训练 PPO，未修改源码、配置、数据或 Checkpoint。
- 状态标签：POLICY_INDEPENDENT_CONTROLLED_COUNTERFACTUAL。每个状态以该 Episode 的正式 initial Cache、配置初始 UAV 位置，并假设此前一直 KEEP+STAY 构造；未使用 PPO Checkpoint 或动作快照。
- 事前样本规则：Pool Manifest 中最低 seed 的两个完整 Training Episode；每集固定 Slot 0、150、295。Slot 295 的 H=5 覆盖 295–299，数据完整。
- Cache 候选取当前正式预测候选管线稳定返回顺序中的首个合法非 KEEP 目标；UAV 取正式合法候选生成器的首个非 STAY MOVE。选择不读未来真实任务效用。真实任务/位置仅用于事后 Matching+KKT 评价。
- H=1、3、5 为无折扣受控固定配置/位置窗口，不等于 PPO 实际 Holding 或未来动作；窗口内 Cache 不再更新，UAV 位置保持固定。

## 当前配置、数据合同与成本公式

- 权重：omega_delay=0.5，omega_energy=0.5；T_ref=30.4737283553 s，E_ref=5.15746425063 J。lambda_cache=0.01，lambda_movement=8.8e-06。Cache Holding=1–5 Slot。
- Cache：C_ca=lambda_cache×Σ(eta_ca·实际变化指示 + xi_ca·新增装载MB)。xi_ca×MB 是等效目标成本代理，不是实测焦耳。
- UAV：C_q=lambda_movement×Σ(固定更新项·实际移动指示 + 移动能耗系数(J/m)×水平距离(m))。位置未变时成本为零。
- 每个反事实候选的 KEEP+STAY 基线和候选分支都从相同快照独立运行正式 execute_configured_slot 与 Matching+KKT；按 Slot 使用同一真实任务/车辆位置。系统效用逐 Slot 校验 G=Q−Cca−Cq，资源可行性通过 Matching/KKT 标记核验。H=1/3 由同一 H=5 轨迹前缀汇总。
- Pool Manifest SHA-256：b6cae236b029f4d24c7488a91f0825247a20f475ec76b2d80737de44ae491bd3。当前任务生成指纹、任务数据合同哈希和服务目录哈希与 Pool 完全一致；完整配置 raw/semantic 哈希不同（pool raw=80762a71c870499eb9cce6aba330531495eac88783732c2ead1b045b2ba8f7c9；current raw=9abecdd758351be16aa00ee2915fb740152a3b661835130b8a78c3a2df47b66e）。该差异已披露且未绕过；全部分支统一使用当前 0.5/0.5、T_ref/E_ref 和成本配置。

## 六个预先固定状态

| seed | Slot | time_s | H=5 每 Slot 到达数 | Cache 正式候选数/节点 | 合法 UAV MOVE 数 |
|---:|---:|---:|---|---|---:|
| 20001 | 0 | 100.0 | 12,14,16,14,14 | 46:4,10:8,0:8,23:5,59:8,RSU_0:13,RSU_1:11,UAV_0:12,UAV_1:8 | 2 |
| 20001 | 150 | 250.0 | 14,11,11,8,13 | 46:4,10:8,0:7,23:5,59:7,RSU_0:14,RSU_1:11,UAV_0:11,UAV_1:8 | 2 |
| 20001 | 295 | 395.0 | 11,12,11,11,16 | 46:5,10:7,0:7,23:5,59:7,RSU_0:13,RSU_1:11,UAV_0:13,UAV_1:8 | 2 |
| 20002 | 0 | 100.0 | 15,18,16,11,18 | 55:4,30:7,11:7,21:5,47:7,RSU_0:13,RSU_1:11,UAV_0:13,UAV_1:8 | 2 |
| 20002 | 150 | 250.0 | 12,16,10,17,13 | 55:4,30:7,11:7,21:5,47:7,RSU_0:13,RSU_1:11,UAV_0:12,UAV_1:8 | 2 |
| 20002 | 295 | 395.0 | 16,14,16,18,12 | 55:4,30:7,11:7,21:5,47:7,RSU_0:13,RSU_1:11,UAV_0:11,UAV_1:8 | 2 |

- 有效配对：Cache 54/54；UAV 12/12（Valid 数以完整 H=5 Matching/KKT 可行配对计，H=1/3 同一路径前缀）。
- 实际正式 Slot 执行 660 次，KKT solver 调用 68244 次，配对分支累计 41.962 s，整个分析 42.267 s。

## Cache 结果

| 节点角色 | H | n | mean ΔQ | median ΔQ | 正 ΔQ 比例 | mean Cca | median Cca | mean ΔG | 正 ΔG 比例 | ΔQ≤0 数 | λBE n / p25 / median / p75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ALL | 1 | 54 | 0.007357 | 0.000000 | 0.388889 | 0.139743 | 0.118750 | -0.132386 | 0.000000 | 33 | 21 / 0.000994 / 0.002685 / 0.003257 |
| ALL | 3 | 54 | 0.017553 | 0.007080 | 0.574074 | 0.139743 | 0.118750 | -0.122190 | 0.092593 | 23 | 31 / 0.002229 / 0.003747 / 0.006970 |
| ALL | 5 | 54 | 0.013623 | 0.010603 | 0.574074 | 0.139743 | 0.118750 | -0.126121 | 0.129630 | 23 | 31 / 0.001705 / 0.003520 / 0.008138 |
| SV | 1 | 30 | 0.005311 | 0.000000 | 0.333333 | 0.101546 | 0.109688 | -0.096235 | 0.000000 | 20 | 10 / 0.000769 / 0.001953 / 0.002820 |
| SV | 3 | 30 | 0.008669 | 0.005021 | 0.600000 | 0.101546 | 0.109688 | -0.092877 | 0.000000 | 12 | 18 / 0.001816 / 0.002484 / 0.003743 |
| SV | 5 | 30 | 0.008114 | 0.008567 | 0.600000 | 0.101546 | 0.109688 | -0.093431 | 0.000000 | 12 | 18 / 0.001696 / 0.002484 / 0.004156 |
| RSU | 1 | 12 | 0.019344 | 0.000000 | 0.416667 | 0.169472 | 0.171886 | -0.150128 | 0.000000 | 7 | 5 / 0.002685 / 0.004849 / 0.006964 |
| RSU | 3 | 12 | 0.070650 | 0.061319 | 0.666667 | 0.169472 | 0.171886 | -0.098821 | 0.250000 | 4 | 8 / 0.003669 / 0.005911 / 0.018445 |
| RSU | 5 | 12 | 0.077926 | 0.019640 | 0.666667 | 0.169472 | 0.171886 | -0.091546 | 0.333333 | 4 | 8 / 0.001186 / 0.006189 / 0.017164 |
| UAV | 1 | 12 | 0.000484 | 0.000147 | 0.500000 | 0.205509 | 0.251061 | -0.205025 | 0.000000 | 6 | 6 / 0.000946 / 0.002317 / 0.003294 |
| UAV | 3 | 12 | -0.013335 | -0.022495 | 0.416667 | 0.205509 | 0.251061 | -0.218844 | 0.166667 | 7 | 5 / 0.006809 / 0.009641 / 0.012465 |
| UAV | 5 | 12 | -0.036910 | -0.072130 | 0.416667 | 0.205509 | 0.251061 | -0.242419 | 0.250000 | 7 | 5 / 0.008594 / 0.012331 / 0.017134 |

任务效用增量 ΔQ 与 Cache 成本分开列示；净收益按 ΔG=ΔQ−Cca。盈亏平衡 λBE=ΔQ/S 只统计 ΔQ>0 且 S>0，ΔQ≤0 单独计数，不混入其分位数。节点按 SV、RSU、UAV 分组。

## UAV 结果

| H | n | mean ΔQ | median ΔQ | 正 ΔQ 比例 | mean Cq | median Cq | mean ΔG | 正 ΔG 比例 | ΔQ≤0 数 | λBE n / p25 / median / p75 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 | 0.000306 | 0.000369 | 0.916667 | 0.002860 | 0.002860 | -0.002554 | 0.000000 | 1 | 11 / 0.000001 / 0.000001 / 0.000001 |
| 3 | 12 | 0.001013 | 0.000936 | 1.000000 | 0.002860 | 0.002860 | -0.001847 | 0.000000 | 0 | 12 / 0.000002 / 0.000003 / 0.000005 |
| 5 | 12 | 0.001786 | 0.001806 | 1.000000 | 0.002860 | 0.002860 | -0.001074 | 0.166667 | 0 | 12 / 0.000004 / 0.000006 / 0.000007 |

H=1 是单 Slot MOVE；H=3/5 是位置保持后的持续效应，MOVE 成本仍只在第一次实际位移扣除。

## 判断与限制

- CACHE_LAMBDA_ASSESSMENT = EXPLORATORY_ONLY; H=1: POSSIBLY_HIGH_RELATIVE_TO_PILOT (lambda>0.00268499, n=21); H=3: POSSIBLY_HIGH_RELATIVE_TO_PILOT (lambda>0.00374706, n=31); H=5: POSSIBLY_HIGH_RELATIVE_TO_PILOT (lambda>0.00351995, n=31); first prediction-derived candidate per node, not PPO-selected; no parameter change inferred.
- UAV_LAMBDA_ASSESSMENT = EXPLORATORY_ONLY; H=1: POSSIBLY_HIGH_RELATIVE_TO_PILOT (lambda>1.13752e-06, n=11); H=3: POSSIBLY_HIGH_RELATIVE_TO_PILOT (lambda>2.87926e-06, n=12); H=5: POSSIBLY_HIGH_RELATIVE_TO_PILOT (lambda>5.55574e-06, n=12); first prediction-derived candidate per node, not PPO-selected; no parameter change inferred.
- 这是 6 个受控状态、每节点/每 UAV 一个预测候选的探索性样本，不能代表 PPO 实际动作分布、全部合法候选或总体平均收益。lambda 相对 break-even 的关系仅描述本次固定候选；不据此重新标定。
- 当前证据不足以给出有外部效度的惩罚过强/过弱结论；建议保持两个 lambda 不变。

## 输出合同

CACHE_LAMBDA_CURRENT = 0.01
UAV_LAMBDA_CURRENT = 8.8e-06
OBJECTIVE_WEIGHTS = (0.5, 0.5)
TRAINING_EPISODES_USED = 20001, 20002
CONTROLLED_STATES = 6
VALID_CACHE_PAIRS = 54
VALID_UAV_PAIRS = 12
CACHE_POSITIVE_NET_GAIN_FRACTION = 0.000000 (H=1; n=54)
UAV_POSITIVE_NET_GAIN_FRACTION = 0.000000 (H=1; n=12)
CACHE_LAMBDA_ASSESSMENT = EXPLORATORY_ONLY; H=1: POSSIBLY_HIGH_RELATIVE_TO_PILOT (lambda>0.00268499, n=21); H=3: POSSIBLY_HIGH_RELATIVE_TO_PILOT (lambda>0.00374706, n=31); H=5: POSSIBLY_HIGH_RELATIVE_TO_PILOT (lambda>0.00351995, n=31); first prediction-derived candidate per node, not PPO-selected; no parameter change inferred.
UAV_LAMBDA_ASSESSMENT = EXPLORATORY_ONLY; H=1: POSSIBLY_HIGH_RELATIVE_TO_PILOT (lambda>1.13752e-06, n=11); H=3: POSSIBLY_HIGH_RELATIVE_TO_PILOT (lambda>2.87926e-06, n=12); H=5: POSSIBLY_HIGH_RELATIVE_TO_PILOT (lambda>5.55574e-06, n=12); first prediction-derived candidate per node, not PPO-selected; no parameter change inferred.
PARAMETER_CHANGE_RECOMMENDED = NO
TRAINING_EXECUTED = NO
CONFIG_MODIFIED = NO
