# Stage6 四组 Macro PPO 学习机制诊断

## 范围与证据限制

本报告只读了 `src/ppo_simple/` 的当前源码，以及四个指定 Run 各自的 `run_manifest.json`、`run_config.toml`、`episode_metrics.jsonl`、`update_metrics.jsonl` 和 `critic_diagnostics.jsonl`。没有启动训练、环境回放、Matching/KKT，也没有使用 Validation/Test。四个 Run 都保存了 Checkpoint；本次依据已记录的真实 rollout 与更新日志分析，没有重放策略或补造动作。

执行 `conda activate adaptive_cache_mec` 失败，Windows 返回 `The system cannot execute the specified program.`。只用可用的 Anaconda Python 3.12.7 解析 JSON/TOML 日志，并静态读取源码；没有依赖环境包执行项目代码。这限制了本次对运行时行为的额外复现，但不影响下列已落盘日志统计。

四个 Run 的 Macro Actor 学习率均从各自 `run_manifest.json` 读取，而非采用 `run_config.toml` 默认值：

| manifest 学习率 | Run 目录 | 已记录 Episode / PPO 更新 |
|---:|---|---:|
| 1e-4 | `runs/stage6_lr_1e_minus_4` | 136 / 34（未完成 300 / 75）|
| 5e-4 | `runs/stage6_lr_5e_minus_4` | 64 / 16（未完成）|
| 1e-3 | `runs/stage6_lr_1e_minus_3` | 120 / 30（未完成）|
| 5e-3 | `runs/stage6_lr_5e_minus_3` | 88 / 22（未完成）|

四组前 60 个 Episode 使用相同的 Episode seed 序列 20001–20060，覆盖相同训练进度的前 15 次 PPO 更新。下文以此前 60 轮做主对照；末段数字分别来自每个 Run 自己最后 16 轮，进度并不相同，只用来描述各自已观察到的趋势，不作同进度比较。

## 1. Macro PPO 的实际实现

### Cache 候选与概率

- `src/ppo_simple/policy.py:265-281`：Macro Actor 由状态编码器、Cache 候选特征编码器与评分器、Termination 评分器构成；Macro Critic 是独立的状态价值网络。`policy.py:392-419` 将每个候选特征与同一状态表示拼接后打分，因此源码具备按候选特征区分候选的结构；结构存在本身不能证明学出的排序有用。
- `src/ppo_simple/policy.py:368-390`：对每个合法 UPDATE logit 减去 `log(N_update)`，KEEP 不变。若原始 logit 为 `s_i`、合法 UPDATE 数为 `N`，分布等价于 KEEP 权重 `exp(s_keep)` 与 UPDATE 权重 `exp(s_i)/N` 的归一化。该操作抵消了 UPDATE 候选多导致的总指数权重膨胀；不会让 UPDATE 组总概率变为单个 UPDATE 概率。
- `src/ppo_simple/rollout.py:649-661`：正式 rollout 保存的 Cache 候选 mask 全部为合法；候选 0 是当前 Cache / KEEP，实际每帧为 KEEP 加 230 个 UPDATE 候选。日志中的合法 UPDATE 数始终为 230。
- `src/ppo_simple/policy.py:475-592`：训练时从基数校正后的单个 Categorical 分布随机采样，并保存校正概率与未经校正的 raw 概率。确定性推理在 `policy.py:122-157, 506-557` 先按校正分布比较 KEEP 与 UPDATE 两组总质量；若选 UPDATE，再在 UPDATE 候选内取 raw score 最大者。UPDATE 的校正只减同一个常数，所以组内 raw score argmax 与校正 score argmax 相同。确定性推理不是把 KEEP 的单动作概率直接与某个 UPDATE 的单动作概率做平铺 argmax。

### Holding / 联合动作

- `src/ppo_simple/policy.py:442-469`：Termination 网络看状态编码及 `elapsed_holding_slots / max_slots`；只有 `min_slots <= elapsed < max_slots` 才能采样。`policy.py:595-621` 训练时随机采样 CONTINUE / TERMINATE，确定性时二元 logits argmax。
- `src/ppo_simple/rollout.py:997-1052`：每 Slot reward 按 frame 内时距折扣累加；达到 max holding 时强制结束，其他可决策 Slot 才记录 Termination 选择。Holding 上限是 5 Slot，最小持有 1 Slot（各 Run 配置一致）。
- `src/ppo_simple/policy.py:657-782`：一帧的 joint log probability 是一次 Cache log probability 加该帧所有已采样 Termination log probabilities 之和；对应的 Termination entropy 也按决策次数求和。
- `src/ppo_simple/rollout.py:1519-1562`：Macro SMDP GAE 使用每帧实际 duration：TD 折扣为 `gamma ** duration`，GAE trace 为 `(gamma * lambda) ** duration`。`trainer.py:1490-1541` 对每个 Episode 分别算 advantage，再拼接归一化。

### Actor、KL 和 Critic 更新

- `src/ppo_simple/trainer.py:1650-1670, 1734-1787`：Cache 和该帧的 Termination 序列共用同一个归一化 Macro advantage；一份 joint ratio `exp(new_joint_logp-old_joint_logp)` 进入一个 PPO clipped surrogate。Cache KL、Termination KL、joint KL 与 clip fraction分别记录，但优化的 clipping 是 joint clipping。Run 配置为 clip epsilon 0.2、4 个 update epochs、`macro_target_kl=0.1`，early-stop 阈值是 `1.5 * target = 0.15`；Macro entropy coefficient 为 0，因此虽计算并记录 entropy，熵项没有 Actor 梯度贡献。`trainer.py:1752-1765` 只更新 Macro Actor 参数，Macro Critic 参数独立。
- **KL early-stop 的时序缺口（CONFIRMED）**：`trainer.py:1734-1742` 在本轮 optimizer step 之前算并记录 KL；`trainer.py:1763` 随后执行 `optimizer.step()`；`trainer.py:1785-1787` 再根据刚才那份 step 前 KL 决定退出。因此，一旦本轮测到 KL 已过阈值，当前这一轮 step 仍会先执行。日志也没有这次退出后最终策略的 post-step KL。Early-stop 能截断后续 epoch，不能撤销已发生的大步更新。
- `src/ppo_simple/trainer.py:1512-1519`、`src/ppo_simple/macro_targets.py:10-65`：Macro Critic 使用冻结、detach 的 two-step SMDP TD 目标；`trainer.py:2152-2351` 独立做 50 次 Critic 更新，并记录当前/上一更新组的梯度处理与裁剪。
- `src/ppo_simple/trainer.py:2538-2679`：Long Value Critic 使用独立参数拟合 detach、冻结的 realized return-to-go，不反传到 Actor 或 Macro Critic。故 Macro Critic EV 与 Long Value EV 都是状态价值目标拟合指标，不是 Cache 候选之间的反事实排序指标。

## 2. 前 60 Episode 对照

以下是每组 Episode 1–60 的 Episode 级均值；强制到 5 Slot 的比例按 `forced_termination_max_holding_count / macro_frame_count` 汇总。`termination_probability` 是 rollout 中实际出现的 Termination 决策状态上的预测概率均值。Episode Return 包含 Task Utility、Cache cost 和 UAV movement cost 的影响。

| LR | Episode Return | Task Utility | Cache cost | Movement cost | 平均 Holding | 平均终止概率 | 强制到 5 Slot | Cache UPDATE rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-4 | 3989.779 | 3997.285 | 6.966 | 0.541 | 2.647 | 0.331 | 20.16% (1422/7055) | 0.512 |
| 5e-4 | 3989.853 | 3998.215 | 7.751 | 0.611 | 2.428 | 0.380 | 15.09% (1140/7554) | 0.533 |
| 1e-3 | 3987.524 | 3997.873 | 9.659 | 0.691 | 2.144 | 0.448 | 9.91% (845/8526) | 0.575 |
| 5e-3 | 3982.299 | 3998.659 | 15.621 | 0.740 | 1.138 | 0.899 | 0.25% (41/16173) | 0.500 |

**结论**：LR 越高 Holding 越短、Cache cost 越高的关联在共同 Episode 区间中已出现（CONFIRMED）。Task Utility 只在约 3997.3–3998.7 间变化；5e-3 虽有略高 Task Utility，但回报最低，主要伴随 Cache cost 较高。低终止概率与少做 Cache 更改有直接成本收益信号（SUPPORTED）；这些日志没有同一状态下的反事实收益，因此不能断言 5 Slot 的真实长期效用更高，也不能断言 1 Slot 最优（UNKNOWN）。

相同的 300 Episode 训练尚未完成。各 Run 最后 16 轮的持有行为是：1e-4 的 Episode 121–136 平均 Holding 4.82、终止概率 0.011、强制到上限比例 92.8%、Cache cost 3.35；5e-4 的 Episode 49–64 为 2.68、0.312、20.5%、7.28；1e-3 的 Episode 105–120 为 2.71、0.319、23.2%、5.35；5e-3 的 Episode 73–88 为 1.03、0.963、0%、18.54。它们的区间不同，不能据此比较最终效果或收敛性。

### Holding 随 LR 的机制证据

前 60 轮的 Termination KL / joint KL 均值按 LR 递增分别为：

| LR | 前 15 次更新平均 Cache KL | Termination KL | Joint KL | Joint clip fraction | Early-stop 次数 | 首次更新 Termination / Joint KL |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-4 | 0.000002 | 0.000495 | 0.001205 | 0.004 | 0/15 | 0.00176 / 0.00333 |
| 5e-4 | 0.000082 | 0.001469 | 0.003316 | 0.032 | 0/15 | 0.00900 / 0.01724 |
| 1e-3 | 0.000525 | 0.003408 | 0.007396 | 0.071 | 0/15 | 0.01862 / 0.03583 |
| 5e-3 | 0.004383 | 0.033074 | 0.059771 | 0.124 | 3/15 | **0.34537 / 0.64954** |

**5e-3**：首次 PPO 更新（episode 1–4 的 rollout 后）Termination KL 0.345、joint KL 0.650，明显超过 0.15 阈值；Termination KL 远高于 Cache KL 0.0012。该更新只跑了 2/4 个 Actor epoch 并标记 early-stop，但由于上述检查时序，两步都已发生。Episode 5–8 平均 Holding 已降到 1.25、终止概率 0.80；Episode 13–16 平均 Holding 1.03、终止概率 0.98。**大步的 Termination 更新是 Holding 快速压到 1 的直接证据（CONFIRMED）**；“严格只有这一原因”仍是 UNKNOWN，因为 advantage、状态分布及逐 epoch post-step KL 未保存到可逐步归因的程度。

**1e-4**：KL 小、前 15 次更新无 early-stop，Holding 从共同起点约 1.88，在 Episode 57–60 到 3.58；到 Episode 121–136 已到 4.82，Termination probability 约 0.011，绝大多数 Macro frame 因达到 max slot 被强制结束。这个缓慢向 CONTINUE 移动的趋势与减少 Cache 变更成本相符（SUPPORTED）。正式机制没有 Holding 熵正则或专门的边界防聚集项：Macro entropy coefficient 是 0；达到 5 Slot 的控制是强制停止。模型仍能通过任务收益、缓存成本和 movement cost 间接学到 Holding 价值，但现有日志不证明 5 Slot 优于较短 Holding（UNKNOWN）。

**5e-4 与 1e-3**：前 60 轮平均 Holding 为 2.43 和 2.14；其后各自最后窗口约 2.68 和 2.71。按 elapsed Slot 聚合的终止概率在 1e-4、5e-4、1e-3 下从 elapsed=1 到 4 分别为 0.351→0.306、0.387→0.369、0.457→0.428；5e-3 为 0.913→0.558（elapsed=4 样本较少，84）。这显示 Termination 概率仍区分已持有时间，且中间 LR 的差异较弱（SUPPORTED）；由于训练未完成，无法断言中间 Holding 是稳定解还是未收敛过渡态（UNKNOWN）。

## 3. Cache UPDATE 候选概率分布

以下从每个前 60 Episode 的逐 Macro frame 候选诊断计算。条件熵用 `H(C|UPDATE) = (H(C) - H(KEEP/UPDATE)) / P(UPDATE)`；等效候选数为 `exp(H(C|UPDATE))`。所有状态均有 230 个合法 UPDATE 候选。

| LR | KEEP 单动作概率 A | UPDATE 总质量 B | 最佳 UPDATE 单动作概率 / UPDATE 条件概率 C | UPDATE 条件熵 D | 熵等效候选数 E | UPDATE 总质量大于 KEEP 的状态占比 |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-4 | 0.494 | 0.506 | 0.00238 / 0.00470 | 5.438 | 229.9 | 91.8% |
| 5e-4 | 0.473 | 0.527 | 0.00316 / 0.00598 | 5.429 | 227.8 | 92.4% |
| 1e-3 | 0.432 | 0.568 | 0.00478 / 0.00816 | 5.398 | 220.9 | 92.7% |
| 5e-3 | 0.502 | 0.498 | 0.00658 / 0.01335 | 5.298 | 199.9 | 51.8% |

230 个等概率候选的条件熵为 `ln(230)=5.438`，条件 top 候选概率为 0.00435。故 1e-4 几乎是均匀，5e-3 有更清楚但仍弱的排序（约 200 个熵等效候选）；候选排序确实有所变化，但整体仍高度分散（SUPPORTED）。不能据此断言候选特征无法区分候选；要确认特征/评分器问题还需在固定状态检查候选特征、raw score 与真实收益之间的关系（UNKNOWN）。

四组中 KEEP 都是**最高的单个动作概率**（100% 状态），因为 KEEP 是一个动作，而 UPDATE 组质量被 230 个候选分摊；各组 KEEP 校正后概率约 0.43–0.50，最佳 UPDATE 单项只有约 0.0024–0.0066。与此同时，前三组 UPDATE 的**组总质量**在约 92% 状态高于 KEEP；5e-3 则约 52%。`raw_candidate0_probability_mean` 只有约 0.003–0.005，校正后的 KEEP 概率大幅不同，说明 cardinality correction 确实参与。

因此，KEEP 单项 argmax 的主要原因是候选基数和组内概率分配，不是 UPDATE 组总质量占优，也不是源码忽略校正（CONFIRMED）。若把 KEEP 单动作概率直接与 UPDATE 组总概率比较，或用平铺 argmax解释确定性选择，都会混淆两种量。当前确定性推理使用校正后的组概率；训练使用校正后的完整 Categorical 采样，二者共享校正分布，但确定性选择规则本来就不同于随机采样（NOT_SUPPORTED：没有发现训练与确定性推理一边校正、一边未校正的错误）。

Episode 统计的 UPDATE 候选 ID 选择熵依 LR 顺序为 4.07、4.18、4.38、4.88；被选次数最多的单一 ID 占 UPDATE 选择约 2.15%、2.14%、1.69%、1.17%，均远低于代码 `rollout.py:1131-1140` 的 80% 单候选支配标记线。这个跨 Episode 的采样 ID 熵与上表“每状态给定 UPDATE 后的概率熵”不是同一个指标；两者都未显示单个 UPDATE 候选垄断。

## 4. Shared Advantage 与信用分配

- **两类动作共享同一优势：CONFIRMED。** 每个 Macro frame 一个 SMDP GAE advantage；Cache 选择与该 frame 中全部 Termination 选择都进入同一个 joint log probability 和 joint PPO ratio（`policy.py:752-760`、`trainer.py:1536-1541, 1670, 1734-1742`）。没有单独计算 Cache 专属 advantage 或 Termination 专属 advantage。
- Cache 改变后续任务效用及缓存成本的收益，经该 Macro frame 的累计 reward、duration-aware GAE 和之后的回报间接回传。Termination 通过改变 frame 长度影响折扣回报、未来再决策时点、缓存更新频率与费用；没有独立的 holding reward 项。
- Termination log probability 是该 frame 各个 Termination 决策的和，而 Cache 只有一个候选动作 log probability；两个头共用 advantage、actor encoder、clip 和学习率。多个 Termination 因子可能使它更容易累积较大的梯度或 joint KL，这是一种结构性风险（SUPPORTED），不是已证明的必然偏置。首个 5e-3 更新中 Termination KL 约为 Cache KL 的 286 倍，说明这次实际大步主要出现在 Termination 分量。
- 因为 joint clipping 使用整体 ratio，Termination 大步可让 Cache 与 Termination 一起受同一个 clipped surrogate 影响；日志虽然记录分量 KL，但没有独立的 head 优化器或分量 clipping。

## 5. Critic、EV 与梯度裁剪

前 15 次 PPO 更新均值如下。Macro EV 是 Macro Critic 对 two-step SMDP TD 目标的 post-fit explained variance；Long Value EV 对 realized return-to-go 目标计算。

| LR | Macro Critic EV | Long Value EV | Macro Critic pre-clip 梯度范数均值 | pre-clip p95 均值 | 0.5 范数阈值触发比例 | Macro TD 目标标准差均值 |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-4 | 0.611 | 0.687 | 98.94 | 382.83 | 100% | 89.16 |
| 5e-4 | 0.631 | 0.690 | 82.40 | 320.35 | 100% | 90.91 |
| 1e-3 | 0.605 | 0.683 | 76.69 | 284.02 | 100% | 75.95 |
| 5e-3 | 0.737 | 0.653 | 31.49 | 105.22 | 98.67% | 32.85 |

Macro Critic `max_grad_norm=0.5`，而记录的 pre-clip 梯度范数均值远大于 0.5；前三组前 15 次更新的 Critic 50 步裁剪触发率都是 100%，5e-3 也有 98.67%。同窗口 Macro TD 目标标准差约 33–91。持续触发裁剪是值得关注的数值事实（CONFIRMED），与原始目标/梯度尺度较大相符；它本身不等于实现错误，因为 global norm clipping 仍保留梯度方向，但会把更新范数持续压到阈值附近。仅凭这些聚合日志不能判断根因是目标尺度、梯度组合/冲突、网络输入尺度还是它们共同作用（UNKNOWN）。

较高 EV 只能支持状态价值目标的拟合，不证明 Actor 能对 Cache 动作做正确排序。Macro Critic 每个状态输出一个 V(s)，Long Value 又是独立的全回报回归器；它们都没有评估 230 个候选的反事实 Q(s,a)。因此“EV 不低但 Cache 候选偏好仍弱”并不矛盾（CONFIRMED）；实际 action-value mismatch 是否存在，现有 EV 无法判定（UNKNOWN）。

## 6. 可能的设计/实现问题与最小后续证据

1. **KL stop 时序值得先修正并验证（CONFIRMED）**：将“已超过阈值”的检查放到下一次 actor step 之前，或记录实际 post-step KL 并禁止下一步越界；保存逐 epoch Cache / Termination / joint KL 与是否执行 step。用现有 5e-3 Checkpoint/同一训练种子做极短对照，先确认首个大步是否被限制。当前报告只发现时序问题，没有更改代码。
2. **Cache 候选学习仍缺评分与效用联系证据（UNKNOWN）**：在少量固定训练状态下保存候选特征、raw score、校正概率和候选执行后的同上下文效用，检查评分排序是否与可比效用相关；同时分别记录 Cache 与 Termination 的梯度范数/贡献。未有该证据前，不建议从 KEEP 单项 argmax 推出改动 cardinality correction 或直接改网络。
3. **Holding 边界不代表收益最优（UNKNOWN）**：要判断 1 Slot 与 5 Slot 哪种经济上更好，需要同一训练种子和可比状态下的完整结果及持有决策反事实收益；当前四组都未跑满 300 Episode，末段窗口不同。前 60 轮只能确认学习率与行为强相关，不能选出最优 LR 或最优 Holding。
4. **Critic 尺度证据先于改目标（SUPPORTED）**：已记录的裁剪率足以建议在后续受控试验中并列检查目标尺度、raw loss 梯度和 PCGrad 前后范数；现在不能据 EV 或裁剪率单独推断应归一化目标或修改 Critic 结构。
5. **Shared advantage 是设计风险，不是定位到的代码错误（SUPPORTED）**：若后续分头梯度记录显示一种决策长期遮蔽另一种，再做单一因素的分头信用/clip 对照；当前没有因果证据证明它是候选排序弱的唯一原因。

## 7. 结论标签

- **CONFIRMED**：四组 Holding 对 LR 高度敏感；5e-3 首个更新的 Termination KL 大幅超阈值且推动 Holding 快速到 1；1e-4 后期趋向 5 并大量强制终止；Cache 与 Termination 共用优势及 joint PPO clipping；KEEP 单动作 argmax 来自 1 对 230 的概率分摊；Critic 持续触发 0.5 梯度范数裁剪。
- **SUPPORTED**：1e-4 长 Holding 与降低 Cache 成本信号一致；5e-3 的 Termination 因子比 Cache 更新更激进；Cache UPDATE 组排序略有改善但仍弥散；shared advantage 与 joint clipping 可能导致两分量信用耦合。
- **NOT_SUPPORTED**：KEEP 单动作概率最大本身是实现错误；确定性推理遗漏 cardinality correction；高 Critic EV 足以证明 Cache 排序准确。
- **UNKNOWN**：230 候选特征本身是否足以区分价值；哪个 LR/Holding 真正效用最优；中间 LR 是否已收敛；价值预测与动作排序是否存在真实错配；Critic 梯度持续裁剪的单一根因。

本次未修改正式代码、训练配置、Checkpoint 或算法；未启动训练或 Matching/KKT。唯一计划产物为本报告。
