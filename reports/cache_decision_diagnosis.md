# PROPOSED Cache KEEP/UPDATE 决策诊断

## 数据与口径

- 正式主线：`main`，commit `1633ca5b62f314a76c8b4be694b576c388fe1850`；正式配置 `configs/proposed/system_config.toml` 的 SHA-256 与 checkpoint 保存值一致。
- Checkpoint：`runs/formal_proposed/checkpoints/latest_checkpoint.pt`，PROPOSED checkpoint v53 / architecture v27，已完成 164/300 个 Train realization、41 次 policy update。使用其 `last1` 中的 240 个 Macro 起点观测，对应 Train seed `20164`；各 holding duration 合计 300 slots。重建 observation 的最大误差为 0。
- 20 个状态按 seed 20164 原训练日志中的 P_KEEP 分层抽取，再用 latest checkpoint 在这些同一观测及正式候选特征上重算 raw logits 和校正概率。覆盖原日志区间：`0.031–0.090`（2 个）、`0.486–0.593`（3 个）、`0.612–0.795`（7 个）、`0.815–0.923`（8 个）；该 realization 在 `0.2–0.4` 区间没有 Macro frame。表中“Train动作”是原 rollout 记录的 PPO 抽样动作；“当前组决策”是 latest checkpoint 的确定性组概率决策，两者策略时点不同。
- `L_KEEP` 与 `L_UPDATE*` 是 latest checkpoint 的未校正 raw logit。`P_KEEP/P_UPDATE` 是正式基数校正后的组概率。每个状态有 351 个合法 UPDATE 候选，另加 KEEP。
- 反事实在相同 Train realization 的真实任务、真实车辆位置和该帧起点 UAV 位置下进行；每对 KEEP/UPDATE 都冻结相同 UAV 配置，不计 UAV 移动。直接调用现有 configured-slot、Matching、KKT 和 system utility 路径，按 `G = task utility − Cache cost − movement cost` 计算单 slot `ΔG`。

## 20 个 Macro 起点

`抑制=是` 表示最高 UPDATE raw logit 高于 KEEP，但总 `P_UPDATE < P_KEEP`；表中的训练动作来自 seed 20164 原日志，当前组决策按 latest checkpoint 计算。

| Frame@slot | N_UPDATE | L_KEEP | L_UPDATE* (#) | P_KEEP/P_UPDATE（当前；训练日志 P_KEEP） | Train动作 | 当前组决策 | 抑制 |
|---:|---:|---:|---:|---:|---|---|:---:|
| 0@0 | 351 | -8.104 | -0.456 (#117) | 0.033/0.967（0.031） | U118 | U117 | 否 |
| 1@1 | 351 | -6.220 | 0.626 (#240) | 0.099/0.901（0.090） | U268 | U240 | 否 |
| 2@2 | 351 | -4.708 | -0.894 (#167) | 0.492/0.508（0.486） | U193 | U167 | 否 |
| 9@10 | 351 | -3.333 | -0.157 (#77) | 0.588/0.412（0.576） | KEEP | KEEP | 是 |
| 10@11 | 351 | -3.345 | -0.170 (#92) | 0.604/0.396（0.593） | KEEP | KEEP | 是 |
| 13@16 | 351 | -2.171 | 0.936 (#272) | 0.621/0.379（0.612） | U159 | KEEP | 是 |
| 28@33 | 351 | -0.655 | 1.427 (#251) | 0.794/0.206（0.795） | KEEP | KEEP | 是 |
| 30@35 | 351 | -0.653 | 1.403 (#44) | 0.816/0.184（0.815） | KEEP | KEEP | 是 |
| 44@50 | 351 | -0.379 | 1.629 (#79) | 0.846/0.154（0.848） | U60 | KEEP | 是 |
| 52@58 | 351 | -0.416 | 1.592 (#80) | 0.837/0.163（0.840） | KEEP | KEEP | 是 |
| 70@85 | 351 | 0.680 | 2.170 (#42) | 0.863/0.137（0.862） | U73 | KEEP | 是 |
| 101@120 | 351 | -1.025 | 2.996 (#81) | 0.681/0.319（0.720） | KEEP | KEEP | 是 |
| 129@152 | 351 | 0.504 | 4.280 (#82) | 0.722/0.278（0.765） | KEEP | KEEP | 是 |
| 138@166 | 351 | 0.410 | 4.199 (#82) | 0.721/0.279（0.764） | KEEP | KEEP | 是 |
| 155@190 | 351 | 0.266 | 4.071 (#82) | 0.720/0.280（0.762） | KEEP | KEEP | 是 |
| 173@211 | 351 | 0.294 | 3.773 (#87) | 0.778/0.222（0.789） | KEEP | KEEP | 是 |
| 181@219 | 351 | 0.826 | 2.758 (#80) | 0.879/0.121（0.885） | KEEP | KEEP | 是 |
| 201@247 | 351 | 0.882 | 2.794 (#80) | 0.900/0.100（0.901） | KEEP | KEEP | 是 |
| 217@269 | 351 | 1.901 | 2.599 (#55) | 0.926/0.074（0.923） | KEEP | KEEP | 是 |
| 228@283 | 351 | 2.162 | 2.853 (#55) | 0.925/0.075（0.921） | KEEP | KEEP | 是 |

20 个状态中，17 个满足上述组概率抑制定义；latest checkpoint 的确定性组决策为 17 次 KEEP、3 次 UPDATE。原训练日志在这 20 个状态中记录 14 次 KEEP、6 次 UPDATE。

## Matching + KKT 反事实

| Frame@slot | 任务数 | G_KEEP | UPDATE 候选 | G_UPDATE | ΔG | Δtask utility | ΔCache cost |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1@1 | 16 | 7.879 | 最优 raw U240 | 6.464 | -1.415 | -1.028 | +0.387 |
| 1@1 | 16 | 7.879 | 训练抽样 U268 | 6.411 | -1.468 | -1.149 | +0.319 |
| 9@10 | 12 | 7.084 | 最优 raw U77 | 7.140 | +0.056 | +0.237 | +0.181 |
| 101@120 | 14 | 7.231 | 最优 raw U81 | 7.720 | +0.489 | +0.609 | +0.120 |
| 30@35 | 15 | 7.974 | 最优 raw U44 | 5.613 | -2.361 | -2.262 | +0.099 |

反事实收益有正有负：4 个代表状态的最优 raw UPDATE 中，2 个单 slot `ΔG > 0`、2 个 `ΔG < 0`。正收益状态中，Frame 9 的 UPDATE 收益较小；Frame 101 的任务效用增量明显高于新增 Cache 成本。UPDATE 并非普遍不划算。

## 候选筛选探针与判断

候选生成确实会截断部分可行单节点 UPDATE：代表状态的完整可行集合有 953 个，最终提供给 PPO 的有 351 个，另 602 个被每节点容量 64 的选择器筛掉。5 个 SV 节点各自 `19/19` 全保留；两个 RSU 各 `293/64`，两个 UAV 各 `136/64`。

在 Frame 101 对完整 953 个方案用同一 actor 评分：最优已提供候选 U81 的 raw logit 为 `2.996`、`ΔG=+0.489`；最高 raw logit 的未提供候选来自 RSU_0，logit `1.341`、`ΔG=-0.163`。这个小探针确认有筛选，但没有显示该状态因筛选错失更好的动作；不能据此排除其他状态的筛选影响。

目前证据更支持**组概率聚合是显著的 KEEP 偏置来源**：17/20 个样本里，最高 UPDATE raw logit 虽高于 KEEP，组 UPDATE 概率仍低于 KEEP；Frame 101 的最高 UPDATE 单 slot `ΔG` 为正，却被确定性组规则选为 KEEP。正式实现对每个 UPDATE raw logit 减去 `log(351)≈5.861` 后再形成统一分类分布，因此单个候选的高 raw logit 不代表 UPDATE 组获得更高总概率。

**候选评分与收益的对应关系混合**：Frame 1 和 Frame 30 的高分 UPDATE 反事实为负，Frame 9 和 Frame 101 为正。筛选在代表探针中不是当前首要解释；更新本身则只在部分状态有利。当前最明确的问题是组概率聚合会压住某些单候选高分、且即时 `ΔG` 为正的 UPDATE。

## 限制与改动

前述 20 状态反事实只比较 Macro 起点的单 slot；本报告末尾另补 4 个代表状态的 D=1/3/5 固定控制累计反事实。样本来自一个 Train realization 和一个 latest checkpoint（训练完成 164/300 episodes），不能外推为全 Train 分布结论。未使用 Validation/Test seed，未启动 PPO 训练；未改 PPO、系统参数、训练脚本或正式 reward。只新增、更新本报告。

## 多时隙反事实补充

对 Frame 1 / Slot 1、Frame 9 / Slot 10、Frame 30 / Slot 35、Frame 101 / Slot 120，分别比较 KEEP 与当前 checkpoint 最高 raw logit UPDATE（依次为 U240、U77、U44、U81）。每个状态使用同一 Train realization 的真实任务、车辆位置和 scenario slice；Cache 在各自分支内固定为 KEEP 或该 UPDATE 目标，不再二次更新。UPDATE 的正式 Cache cost 只在首 slot 扣一次。累计差按 `ΔG_D = Σ(k=0..D-1) gamma^k (G_UPDATE[k] − G_KEEP[k])`，任务效用增量同样累加；正式 `gamma=1.0`。同一 UAV 序列在两分支复用，故配对移动成本差为 0。

**UAV 位置口径：** checkpoint 只保存了 Macro 起点 UAV 坐标及 `actual_durations`，没有逐 slot Micro 坐标。评估序列由相邻已保存 Macro 起点坐标按对应 `actual_durations` 线性插值；对 float32 反归一化造成的极小单步超限，仅作毫米级限幅以满足正式每时隙移动约束。任务与车辆轨迹是 realization 中的真实逐 slot 数据，但 UAV 序列是基于真实 Macro 起点重建的固定控制序列，不应解读为原 rollout 的完整 Micro UAV 轨迹。

| 状态 / UPDATE | Δ净收益 D=1 / 3 / 5 | Δ任务效用 D=1 / 3 / 5 | 初始 Cache 更新成本 | 单 slot 与累计结论 |
|---|---:|---:|---:|---|
| Frame 1 / Slot 1 · U240 | −1.415188 / +1.081863 / +1.412783 | −1.028113 / +1.468938 / +1.799858 | 0.387075（仅首 slot） | 不一致：D=1 负，D=3、5 转正 |
| Frame 9 / Slot 10 · U77 | +0.056084 / −4.683320 / −9.600063 | +0.237215 / −4.502190 / −9.418932 | 0.181131（仅首 slot） | 不一致：D=1 正，D=3、5 转负 |
| Frame 30 / Slot 35 · U44 | −2.361411 / −3.029094 / −3.284763 | −2.261915 / −2.929598 / −3.185267 | 0.099496（仅首 slot） | 一致：D=1、3、5 均为负 |
| Frame 101 / Slot 120 · U81 | +0.489047 / +1.295642 / +2.106260 | +0.609033 / +1.415628 / +2.226246 | 0.119986（仅首 slot） | 一致：D=1、3、5 均为正 |

D=1 与前述单 slot 反事实数值一致。结果显示，Frame 1 的 UPDATE 虽然首 slot 受更新成本影响而为负，随后 slot 的任务效用改善使 D=3、5 转正；Frame 9 则相反，首 slot 微正收益无法抵消随后 slot 的效用损失。Frame 30 在所有窗口长度下都不划算，Frame 101 在所有窗口长度下都划算。

这是**固定后续控制条件下的反事实诊断**：任务、车辆及重建的 UAV 位置序列固定，两方案仅保持不同 Cache 配置；它不让未来 PPO/Matching 控制策略因 Cache 改变而重新响应，因此不等于完整闭环策略价值或完整 SMDP 价值。未启动 PPO 训练，未使用 Validation/Test seeds，未修改正式代码、参数或 reward；没有留下临时脚本或临时结果文件。

## Cache–Holding 原训练行为核对

### 数据版本与对应关系

- 当前 `latest_checkpoint.pt` 仍是 PROPOSED v53 / architecture v27 resumable checkpoint，`completed_episodes=164`、`policy_update_count=41`；其中 `last1_macro_transition_buffer.episode_index=164`，含 240 个 Macro frame、`actual_durations` 总和 300。`episode_metrics.jsonl` 有 164 条 episode 记录，Episode 164 的 `episode_seed=20164`；checkpoint 顶层 `seed=27005` 是训练运行 seed，不是该 realization 的 seed。Episode 164 有 300 条 Slot 记录。
- Episode 164 的 `macro_candidate_state_diagnostics` 与 `last1` 的 240 个 frame 一一对应；`termination_decision_diagnostics` 有 298 条。按每帧 duration 重建：普通 D=1～4 帧有 D 条决策，D=5 帧有 elapsed=1～4 的 4 条决策，Episode done 帧不采样额外终止动作。所有 240 帧的记录数、elapsed 顺序及动作模式均吻合，预期数与实有数均为 298。因此可以可靠对应这四帧的原训练终止动作和概率。
- Episode 164 的 240 个 `macro_rewards` 与 Slot 日志逐帧累加的 `G` 最大绝对误差为 `9.43e-7`；全部 Slot `G` 之和与 `episode_return` 完全一致。episode 唯一 `done` 是最后的 Frame 239；全 episode 有一个 D=5 强制到达最大 holding 的 frame。当前正式配置 SHA-256 与 checkpoint 保存值一致。终止概率直接取原训练 episode 诊断；没有使用当前 checkpoint 重算概率。

### 四帧原训练行为

表中 `P_T` 是日志保存的原训练 `P(TERMINATE)`；`P(动作)` 是由它得到的当次动作采样概率（CONTINUE 时为 `1−P_T`）。实际累计 Reward 用 Slot 日志 `G` 求和，括号中为 checkpoint 的 `macro_rewards`。

| Frame / 起点 Slot | 原训练 Cache 动作（候选索引） | 实际 D | 实际累计 Reward（Slot G；macro reward） | 原训练 Holding 决策及概率 | Frame 结束原因 |
|---|---|---:|---:|---|---|
| 1 / 1 | UPDATE U268（268） | 1 | 6.410581（6.410581；6.410581） | elapsed 1：TERMINATE，P_T=0.827955，P(动作)=0.827955 | 采样 TERMINATE |
| 9 / 10 | KEEP（0） | 1 | 7.097455（7.097455；7.097456） | elapsed 1：TERMINATE，P_T=0.825351，P(动作)=0.825351 | 采样 TERMINATE |
| 30 / 35 | KEEP（0） | 1 | 7.974016（7.974016；7.974016） | elapsed 1：TERMINATE，P_T=0.832834，P(动作)=0.832834 | 采样 TERMINATE |
| 101 / 120 | KEEP（0） | 2 | 12.333349（12.333349；12.333349） | elapsed 1：CONTINUE，P_T=0.813183，P(动作)=0.186817；elapsed 2：TERMINATE，P_T=0.815089，P(动作)=0.815089 | 采样 TERMINATE |

D=1、2 的上述帧均以采样 TERMINATE 结束；并非达到最大 holding。全 episode 的 D=5 frame 按 elapsed 1～4 的记录均为 CONTINUE，之后因最大 holding 结束；最后一个 `done` frame 则因 Episode 终止结束，不应伪记为采样 TERMINATE。

### 与既有多时隙反事实的证据分级

| Frame | CACHE_DECISION_ERROR_EVIDENCE | PREMATURE_HOLDING_TERMINATION_EVIDENCE | CACHE_HOLDING_MISMATCH_EVIDENCE |
|---|---|---|---|
| 1 | INSUFFICIENT_EVIDENCE | INSUFFICIENT_EVIDENCE | INSUFFICIENT_EVIDENCE |
| 9 | NOT_SUPPORTED | INSUFFICIENT_EVIDENCE | INSUFFICIENT_EVIDENCE |
| 30 | NOT_SUPPORTED | INSUFFICIENT_EVIDENCE | INSUFFICIENT_EVIDENCE |
| 101 | SUPPORTED | INSUFFICIENT_EVIDENCE | INSUFFICIENT_EVIDENCE |

- **CACHE_DECISION_ERROR_EVIDENCE：SUPPORTED（有限、固定控制证据）。** Frame 101 原训练选 KEEP；已有对 U81 的固定控制反事实在 D=1/3/5 均为正（`ΔG=+0.489/+1.296/+2.106`），所以该帧有“可能错过有利 UPDATE”的证据，但不代表完整闭环 PPO 长期价值。Frame 1 原训练实际选 U268，而已有多时隙结果测的是 U240；虽然已有单 slot U268 `ΔG=-1.468`，没有 U268 的 D=3/5 结果，故该帧不足以判断长期 Cache 决策错误。Frame 9 的 U77 在 D=3/5 为负，Frame 30 的 U44 在 D=1/3/5 均为负，因此这些已测试候选不支持“错过长期正收益 UPDATE”；这不覆盖未测试候选。
- **PREMATURE_HOLDING_TERMINATION_EVIDENCE：INSUFFICIENT_EVIDENCE。** 原日志证明 Frame 1、9、30 在 elapsed=1 采样 TERMINATE，Frame 101 在 elapsed=2 采样 TERMINATE；它只描述实际动作与概率。既有反事实比较的是不同 Cache 配置，不是同一 Cache 下“继续持有”与“允许重新决策”的闭环对照，因此不能据 D=1、3、5 的某个正 `ΔG` 推断应延长 holding。
- **CACHE_HOLDING_MISMATCH_EVIDENCE：INSUFFICIENT_EVIDENCE。** Frame 101 的 Cache 反事实与实际 KEEP 存在条件性收益差异，但没有联合作用下的 holding/Cache 闭环对照；Frame 1 的多时隙候选又不是原训练实际选择的 U268。当前证据不能确认两种决策机制存在明显不协调。

**建议：目前保留原算法，不修改 Holding。** 若下一步要继续核对，优先补 Cache 评分与原训练实际选择动作的收益对应（尤其 Frame 1 的 U268），而不是先改 Holding 机制。以上仅针对 Episode 164 的四帧与已测候选；固定控制收益不等于完整闭环策略收益。本次只读取日志/checkpoint 并追加本报告，未训练、未重算 Matching/KKT、未改正式代码或参数、未使用 Validation/Test seeds，也未产生临时文件。
