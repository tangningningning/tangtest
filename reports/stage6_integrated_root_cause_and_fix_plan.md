# Stage6 综合根因与最小修复方案

日期：2026-10-12  
范围：只整合三份既有诊断；未重跑环境、Matching/KKT 或 PPO，未修改正式代码、配置、奖励、参数或 Checkpoint。

## 结论摘要

- 现有证据没有发现 Matching/KKT 的 Cloud 优先回退、Cloud CPU 未共享或候选 KKT 重算不一致等实现错误。Cloud 约 80% 的样本占比与当前资源合同和实际 Task Utility 比较相容：Cloud 任务的等效上行速率固定为 200 Mbps，而 Edge CPU 与带宽按节点内任务共享；Cloud 的 40 GHz finite pool 确实被共同分配且样本中饱和。已精确评估的 Cloud→Edge 改派均没有提高全 slot 任务效用。
- Local 为零主要与本地服务副本覆盖不足、合法 Local 方案相对 Cloud 效用较低相容；SV 极少主要是合法候选经过共享资源分配后不常胜过 Cloud。现有样本不支持人为提高 Edge 比例。
- Cache UPDATE 确实能改变 Edge 服务可用性、Matching 结果和系统净收益；但 42 个单槽 UPDATE 反事实全部因新增 Cache 成本高于任务效用增益而降低净收益。四个固定 Cache 的五槽反事实中只有一个净收益为正。候选间存在测得的价值差异，但完整 230 候选的长期价值排序和其可学习性尚未证实。
- PPO 没有形成明确 Cache 候选排序的证据，且四组 Checkpoint 只完成 64–136/300 Episode、16–34/75 次 PPO 更新。当前组内分布仍很分散；不过现有证据不能把弱排序唯一归因于特征、共享优势或训练不足。
- 已确认的 PPO 实现问题是 Macro KL 超阈值后仍先执行本轮 Actor optimizer step 再停止后续 epoch。5e-3 的首次更新出现显著 Termination KL 和 Holding 快速缩至 1 Slot；这是优先修复的实际训练机制问题。1e-4 的 Holding 趋向 5 Slot 与较低终止概率、减少 Cache 更新成本的信号相符，但没有证据证明 5 Slot 的长期收益更高。

因此，优先修复 PPO 的 KL 停止时序，再做固定 Training 状态下的候选价值与 Holding 反事实验证。先不改系统模型、Matching/KKT、Cache Reward、候选生成器或 PPO 的 Cache 信用分配。只有候选反事实与策略评分持续不一致时，才提出 Cache 专属评分/信用分配改动。

## 三份报告的口径与冲突核对

| 证据 | 节点选择报告 | Cache 反事实报告 | Macro PPO 报告 |
|---|---|---|---|
| 样本和执行方式 | 四个学习率各回放一个完整 Training Episode，seed 20001–20004；冻结策略的确定性策略回放。 | 四个学习率的 Checkpoint，seed 20001、20101、20201、20300；12 个状态，实际 Macro 抽样动作及少量候选分支反事实。 | 读取四组训练日志；以共同的前 60 Episode/前 15 次更新作主对照，无环境重放。 |
| 训练进度 | Checkpoint 已完成 64–136 Episode。 | 与各 Run 中现存 Checkpoint 对应；同样尚未完成 300 Episode。 | 已完成 64–136 Episode、16–34 次 PPO 更新；未完成 300/75。 |
| 可以互相支持的结论 | Cloud 占比约 80%，Matching/KKT 样本内稳定。 | 不同状态下 Cloud 占比也约 80%，Cache 分支能改变部分 Matching。 | PPO 候选分布分散；Holding 与学习率强相关。 |
| 不能直接比较的量 | 确定性策略的节点计数。 | 随机抽样动作、少量候选净收益和固定 Cache 五槽反事实。 | 日志中的跨 Episode 统计及 termination 概率。 |

“42 个 UPDATE 单槽净收益全为负”与“固定 Cache 五槽下有 1/4 正收益”不冲突：评价时域不同，后者把一次 Cache 成本摊到未来任务槽，但固定了 Cache 和已记录的 UAV 轨迹，也没有模拟 PPO 在未来重新决策。候选排序证据同样不矛盾：Cache 报告测到候选价值跨度，但只评估每状态 3–4 个 UPDATE；Macro PPO 报告指出全 230 候选的评分—收益关系未知。

报告引用：[节点选择诊断](./stage6_node_selection_diagnosis.md)、[Cache 反事实诊断](./stage6_cache_counterfactual_diagnosis.md)、[Macro PPO 学习诊断](./stage6_macro_ppo_learning_diagnosis.md)。

## 根因表

| 问题/表现 | 直接证据与可能原因 | 证据级别 | 源码模块 | 是否需要改代码 | 最小验证方式 |
|---|---|---|---|---|---|
| Cloud 约 80%，Edge 低 | 四 Episode 共 16,728 个到达任务中 Cloud 13,410（80.17%）；1,200/1,200 slot 的 40 GHz Cloud CPU 池用满。精确记录的 Cloud→Edge transfer 全部降低全 slot Task Utility。固定 200 Mbps/任务 Cloud 上行不随 Cloud 任务数分摊；Edge CPU/BW 按节点组共享，在该任务组成和信道下更容易降低效用。 | 占比 CONFIRMED；资源机制对占比的解释 SUPPORTED。不是对所有任务/状态最优的证明。 | [system_model.py](../src/common/system_model.py) Cloud/Edge 执行；[task_allocation.py](../src/decision/task_allocation.py) 候选、Matching 和 KKT。节点报告列出函数及行号。 | 否。不改 Cloud/Edge 物理参数或奖励来强制 Edge。 | 暂无必要重复审计。若未来要声称全局最优，需补齐被 upper-bound pruning 的任务和多任务联合改派评估。 |
| Local 执行为零 | 16,728 个任务中 15,641 个（93.50%）没有 TV 本地服务副本；其余 1,087 个进入 Local 候选。稳定性检查记录的 1,000 个 Cloud→Local 配对重算全部为负 ΔJ。大部分任务没有本地服务可用；合法 Local 方案在样本条件下仍不如 Cloud。另有 87 个没有同样的已记录精确 paired transfer，不能补推。 | SUPPORTED。 | [task_allocation.py](../src/decision/task_allocation.py) 的 build_feasible_targets；[system_model.py](../src/common/system_model.py) 的 Local 执行与统一 Task Utility。 | 否。无证据证明 Local 规则实现错误。 | 若要扩大结论，在少量 Training 状态对有本地副本的任务做 Local/Cloud 同 slot KKT 配对比较。 |
| SV 执行极少 | 10,849 个任务—SV 节点缓存命中候选，路径/正资源检查未排除缓存命中；最终只分配 8 个。精确 Cloud→SV 重分配样本均未提高全 slot 效用。缓存命中只让 SV 成为可行候选，不保证共享 CPU/BW 和信道下的效用高于 Cloud。另有 653 个 Cloud 任务的至少一个 Edge 候选没有逐项精确重算。 | SUPPORTED。 | [task_allocation.py](../src/decision/task_allocation.py) 的 build_feasible_targets、_edge_path_is_feasible、optimize_task_matching、allocate_resources_kkt。 | 否。不为提升份额放松真实约束或改物理参数。 | 对有 SV 缓存命中的少量 Training 状态重用 full-context KKT paired transfer；不做跨上下文单任务效用比较。 |
| Cache UPDATE 对卸载与净收益的影响 | 12 状态中，42 个 UPDATE 对照改变了一部分 Matching：汇总有 18 次 Cloud→Edge、22 次 Edge→Cloud、6 次 Edge 内切换。42/42 单槽 ΔG 为负，因每次 Cache 成本均超过当槽 ΔQ。固定 Cache 五槽反事实为 1/4 正净收益。UPDATE 改变服务可用性，Matching 再按效用重排；更新成本即时发生，可能需后续任务收益摊销。 | Cache 改变结果 CONFIRMED（诊断样本内）；长期净收益普遍为正 UNKNOWN。 | [rollout.py](../src/ppo_simple/rollout.py) 的 _cache_candidates；[cache_configuration.py](../src/decision/cache_configuration.py) 候选/容量校验；[slot_environment.py](../src/decision/slot_environment.py) 的 execute_slot；[task_allocation.py](../src/decision/task_allocation.py) 的 optimize_task_matching。 | 暂不需要。当前单槽负收益不足以证明 Reward 错；五槽结果也不足以改成本公式。 | 在少量相同 Training 状态比较 KEEP 与候选的同上下文 1–5 槽净收益；保持真实 Cost、Matching/KKT 和 UAV 条件不变。 |
| UPDATE 候选有差异，但策略未显示清晰排序 | 每状态被测 UPDATE 的 ΔG 跨度 0.050–0.242，中位数 0.116；单槽测得的 UPDATE 全低于 KEEP。被测最高评分 UPDATE 只有 4/12 次是被测分支中净收益最高。训练日志组内有效候选数约 195–230/230。可能原因包括训练未完成、弱组内区分、共享 Cache/Termination 优势与 joint clipping；源特征是否足以预测长期收益未知。 | 价值差异 SUPPORTED；当前策略已学会可靠排序 NOT_SUPPORTED；完整可学习性 UNKNOWN。 | [policy.py](../src/ppo_simple/policy.py) 的 DualTimescaleActorCritic._macro_cache_logits、evaluate_macro_components；[trainer.py](../src/ppo_simple/trainer.py) 的 PPOTrainer._update_macro；[rollout.py](../src/ppo_simple/rollout.py) 的 _cache_candidates。 | 尚无充分证据直接改评分器或信用目标。若固定状态完整排名仍稳定错配，再改 PPO Cache 学习目标/信用分配。 | 先在 2–4 个既有 Training 状态记录完整合法候选 raw score、特征和同上下文多槽 ΔG，比较排序相关性；只评估，不训练。 |
| Holding 随学习率出现两端行为 | 共同前 60 Episode 中平均 Holding 从 LR 1e-4 的 2.647 递减到 5e-3 的 1.138。5e-3 首次更新的 Termination/Joint KL 达 0.345/0.650，远超 0.15 early-stop 阈值；超阈值检查位于 Actor step 之后。1e-4 最后 16 Episode 的平均 Holding 4.82、终止概率 0.011。高 LR 的大步让策略快速倾向 1 Slot；低 LR 的低终止概率及减少 Cache 更新成本信号与 Holding 趋向上限相符。两者均未证明边界收益最优。 | 高 LR 崩至 1 的大步关联 CONFIRMED；KL stop 时序缺口 CONFIRMED；低 LR 到 5 的因果解释 SUPPORTED；最佳 Holding UNKNOWN。 | [trainer.py](../src/ppo_simple/trainer.py) 的 PPOTrainer._update_macro；[policy.py](../src/ppo_simple/policy.py) 的 _termination_logits、sample_termination；[rollout.py](../src/ppo_simple/rollout.py) 的 compute_macro_smdp_gae。 | 需修 PPO KL 检查时序；暂不单独改 Termination Reward、Holding 边界或熵系数。 | 固定同一初始化、seed 与前 4 个 Episode 做短 PPO 对照，记录更新前后分头 KL、step 和 Holding；再用少量相同 Training 状态对比不同 Holding 长度净收益。 |
| Matching/KKT 是否是共同根因 | 1,200/1,200 节点诊断 slot、54/54 Cache 分支通过 Transfer-Exchange 稳定性复检；两报告所查 KKT 均收敛、资源残差小；12 个 incremental/full KKT paired 对照完全相同。没有发现 Cloud 优先回退，平局规则反而偏 Edge。Transfer/exchange 稳定性不等于任意多任务全局最优。 | 样本内存在实现错误 NOT_SUPPORTED；全输入全局正确性 UNKNOWN。 | [task_allocation.py](../src/decision/task_allocation.py) 的 optimize_task_matching、evaluate_matching_candidate_incremental、allocate_resources_kkt、_allocate_cloud_cpu_kkt。 | 否。当前证据不支持改 Matching/KKT。 | 仅当出现新反例时，对同状态 incremental/full KKT 重算复核；本计划无需额外运行。 |
| Critic EV/梯度是否解释候选排序 | Macro/Long Value EV 不低，但两个 Critic 均输出状态价值，不直接评价 230 个候选反事实 Q；Critic 梯度裁剪经常触发 0.5 范数上限。高 EV 不能证明 Actor 候选排序准确；裁剪也未被证明导致弱排序。 | EV 不等价于动作排序 CONFIRMED；梯度裁剪是排序根因 UNKNOWN。 | [trainer.py](../src/ppo_simple/trainer.py) 的 PPOTrainer._update_macro、_update_macro_critic_only、_update_long_value_critic_only。 | 暂不需要。不单凭 EV 或 clip 率改 Critic、归一化目标。 | 若后续有候选错配证据，再记录 Macro Actor 各头梯度范数/冲突和 Critic 原始目标尺度。 |
| 四组结果能否代表最终策略 | 四组现存检查点分别只完成 136、64、120、88 Episode，对应 34、16、30、22 次 PPO 更新；正式设置是 300 Episode/75 次更新。不同 Run 最后 16 Episode 进度不同，不能据此判断最终学习率或 Holding 优劣。 | 设定未完成 CONFIRMED。 | [trainer.py](../src/ppo_simple/trainer.py) 的 run_training 与各 Run 日志/Checkpoint。 | 仅当需形成修复后的正式结论时才重训；本任务不启动训练。 | 修复确认后按同一 300/75 protocol、相同 seed 序列跑配对学习率组。 |

## 最小修复方案与顺序

### 1. 修复 Macro KL early-stop 时序

- **目标文件/函数：** [src/ppo_simple/trainer.py](../src/ppo_simple/trainer.py) 中 PPOTrainer._update_macro；现有 KL 计算、macro_optimizer.step()、阈值判断位于约 1734–1787 行。
- **最小改动方向：** 把已测得的超阈值判断放到下一次 Actor optimizer step 之前；保留每 epoch 的 Cache、Termination、joint KL，并明确记录 step 前/后值及 stop 原因。若单次 step 本身跨过阈值，应如实记录；单靠下一轮 early-stop 无法撤销已发生的越界 step，因此先不声称它能保证严格 trust-region 上限，也不立即附加回滚或全局降学习率。
- **必要性：** 当前控制流在超阈值判断前执行了 Actor step，是诊断中唯一确认的训练控制时序缺陷；5e-3 的首轮 Termination KL 尤其异常。
- **预期改善：** 不再在已经观测到 KL 超阈值后继续做下一步 Actor 更新，并获得可审计的 post-step KL；针对 5e-3 Holding 崩至 1 的风险，不承诺其他 LR 收敛或 Cache 排序自动改善。
- **系统模型影响：** 不变；这是 PPO 更新控制行为改变。
- **重新训练：** 需要。修复后 Checkpoint 沿新 optimizer 轨迹，不能和旧轨迹混作同协议结果。要比较四个学习率，四组均应按相同 seed/300 Episode/75 更新重新训练；本次诊断不启动训练。
- **最小验证：** 固定同一初始化、首批四个 Training Episode 与随机种子，跑一次受控更新；核对超阈值后未执行后续 step，且日志同时保留 pre/post KL 和 step 计数。不得使用 Validation/Test。

### 2. 暂时只测 Cache 评分与真实候选价值，不先改 Cache PPO 目标

- **目标位置（首轮只读/诊断）：** [src/ppo_simple/policy.py](../src/ppo_simple/policy.py) 的 DualTimescaleActorCritic._macro_cache_logits；[src/ppo_simple/trainer.py](../src/ppo_simple/trainer.py) 的 PPOTrainer._update_macro；[src/ppo_simple/rollout.py](../src/ppo_simple/rollout.py) 的 _cache_candidates。收益重算继续使用正式 execute_configured_slot 和 Matching/KKT。
- **为什么先测：** 当前 230 候选的完整长期排序没有被计算；现有每状态仅 3–4 个反事实点不足以证明 Actor 学不到候选价值。立即改 Actor 网络、cardinality correction、Reward 或 shared advantage 都超出证据。
- **若验证显示稳定错配，后续最小改动：** 只在 _update_macro/_macro_cache_logits 一带设计 Cache-specific 候选评分或 Cache 专属信用信号，并与现有 joint PPO 做单因素对照；不要改 Cache 经济成本或 Task Utility 来制造目标排序。
- **预期改善：** 只有诊断证明候选特征含有稳定可预测价值且当前 Actor 排序不对时，才可能增强 Cache 排序；没有足够信号时不改模型。
- **系统模型影响：** 若仅改 PPO Actor/信用分配，不改变物理系统模型，但会改变训练算法和结果含义。
- **重新训练：** 若实施，需要新的训练；与现有四组 PPO 结果不能作同协议直接比较。
- **最小验证：** 选择 2–4 个既有 Training 状态，评价每个状态全部合法 UPDATE 候选和 KEEP；记录 raw score、候选特征、同 slot KKT 后效用，以及统一 1–5 槽净收益。先报告排序一致性与范围，不拟合额外奖励、不用 Validation/Test。

### 3. Holding 只在配对收益证明确有损失后再改

- **目标位置（条件性）：** [src/ppo_simple/policy.py](../src/ppo_simple/policy.py) 的 DualTimescaleActorCritic._termination_logits/sample_termination 与 [src/ppo_simple/trainer.py](../src/ppo_simple/trainer.py) 的 PPOTrainer._update_macro。第一步仍是第 1 项的 KL stop 修复。
- **为什么不能现在直接调：** 1e-4 长 Holding 与少付 Cache 成本相符，但 Cache 五槽反事实没有比较同状态的 1、2、3、4、5 槽 termination 决策；也没有证据说长 Holding 在任务效用上更好。5e-3 的 Holding collapse 有过大 Termination 更新证据，但高学习率并非唯一因果解释。
- **条件性改动：** 如果修复 KL stop 后，严格配对的 Holding 分支仍显示某一边界反复降低回报，再单独调整 Termination 学习/信用机制；不加未经验证的最小/最大 Holding 奖励或边界偏置。
- **预期改善：** 降低不必要的边界塌缩，同时保留系统通过真实 Cache cost 和后续任务收益学习持有时长的能力。
- **系统模型影响：** 不改物理系统模型；改变 PPO 的时间决策机制。
- **重新训练：** 实施后需要重新训练和同协议对照，旧 Hold 行为/回报不再代表修复版。
- **最小验证：** 对同一批少量 Training 状态做配对 Holding 1–5 Slot 的真实任务净收益比较，并在 PPO 短对照中记录持有长度、Cache cost、Task Utility 和 Termination KL。固定策略的离线窗口只能作筛查，不能代替完整重训结论。

## 不建议作为本轮修复的内容

- 不修改 configs/proposed/system_config.toml 的 Cloud 40 GHz、200 Mbps/任务、delay factor 15 或 bandwidth sharing 合同；不为指定 Cloud/Edge 份额改写物理假设。
- 不修改 src/decision/task_allocation.py 的 Matching、KKT、feasibility 或 tolerance；当前没有样本内实现错误证据。
- 不调整 src/common/system_model.py 的统一 Task Utility、Cache Cost 或 Reward；Cache 单槽净收益为负并不等于成本实现错误，已有固定五槽例子也表明可能存在摊销收益。
- 不因 KEEP 单动作概率高而改 cardinality correction。当前是 1 个 KEEP 与 230 个 UPDATE 的概率分布，代码已做基数校正；KEEP 单项 argmax 不等同于 UPDATE 组总质量更大。
- 不仅凭 Critic EV 或频繁梯度裁剪就改 Critic。现有 EV 不含动作反事实排序证据。

## 最终判定

PRIMARY_ROOT_CAUSE = PPO侧已确认的Macro KL early-stop时序缺陷放大了高学习率下的Termination更新；Cloud高占比则是当前资源模型及KKT后Task Utility下的结果，不是已发现的Matching优先Bug。

SECONDARY_ROOT_CAUSES = Cache更新即时成本高于大多数单槽任务收益；Local服务缓存覆盖率低；SV合法候选在共享资源效用比较中少胜出；训练尚未完成且Cache与Termination共用Macro advantage/joint clipping可能削弱候选排序。

CLOUD_DOMINANCE_EXPLAINED = PARTIAL

CACHE_VALUE_LEARNABLE = UNKNOWN

PPO_MODIFICATION_NEEDED = YES

SYSTEM_MODEL_MODIFICATION_NEEDED = NO

FIRST_RECOMMENDED_FIX = 修复src/ppo_simple/trainer.py中PPOTrainer._update_macro的KL超阈值检查时序，并记录更新前后分头KL。

FIX_ORDER = 1) 修复并短验证KL停止时序；2) 用少量Training状态测全候选多槽价值与评分对齐；3) 只有确认稳定错配才改Cache专属信用分配/评分；4) 用配对Holding收益决定是否再改Termination机制；5) 需要正式比较时按相同协议完成四组300 Episode/75次更新训练。

NEW_TRAINING_REQUIRED = YES

REMAINING_UNCERTAINTY = 未逐项完整评价的653个Cloud任务Edge候选及多任务联合改派最优性；230个UPDATE的完整长期价值排序和可学习性；Holding 1–5 Slot的同状态因果净收益；修复KL stop后四组完整训练的最终节点占比、收益和收敛性。
