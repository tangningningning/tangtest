# Objective / Reward / PPO Interface Final Audit

审计日期：2026-09-06  
审计对象：C:\Users\tangning\Desktop\code\adaptive_cache_mec  
审计性质：只读最终审计；除本报告外不修改源码、配置、正式 artifact、checkpoint、训练 history 或既有报告。

## 0. 唯一系统模型基准

本报告唯一使用的系统模型是：

    C:\Users\tangning\Desktop\12.pdf

12.pdf 共 12 页。其第 1--3 页定义离散 slot、缓存和任务到达；第 4--8 页定义本地、边缘、Cloud 的执行/通信/能耗以及任务 utility 和配置成本；第 9 页定义 UAV 移动成本与系统 utility；第 10--11 页定义总体优化问题和边缘 KKT 约束。

其他 PDF、旧 checkpoint、旧实验报告和历史 fixture 都不是当前系统模型的替代来源。特别是旧报告中出现的 observation=184、任务队列/FIFO 或旧节点 ID，不覆盖当前 12.pdf 冻结口径。

当前冻结约束同时包括：

- execution.cross_slot_task_model = independent_per_slot；
- slot t 的任务在当前 slot 独立完成，不形成 next-slot queue、backlog、queue wait 或 remaining workload；
- dynamic cache node = 5 SV + 2 RSU + 2 UAV = 9；
- service = 10，service_id 为 0..9；
- Cloud 每任务固定 F_c = 5e9 cycle/s、R_c = 200e6 bit/s，resource_sharing = false，psi_c = 11；
- Cloud 只在没有可执行 SV/RSU/UAV cache target 时作为 fallback；
- UAV deployment/channel altitude = 100 m；
- PPO observation/action = 154 / 11。

## 1. 总体判定

| 检查项 | 当前判定 | 说明 |
|---|---|---|
| 目标函数公式、单位、符号 | PASS | 与 12.pdf 的任务 utility、cache cost、movement cost 和系统目标一致 |
| 实际 Reward 边界 | PASS | Reward 使用当前实际 Matching/KKT 的 Q，不使用预测配置 utility |
| infeasible 处理、terminal、GAE/MC 口径 | PASS（代码静态） | infeasible 不写入有效 transition；formal episode 使用 true terminal |
| observation 结构和因果输入 | PASS | 当前实际结构为 154，且不读 current actual task、Q 或 G |
| action、mask、gate、trigger 语义 | PASS（active path） | action 只负责配置触发，不直接决定 offload target |
| 固定 seed 的公式/短路径复算 | PASS | slot 20、Cloud、KKT、no-trigger 和短触发路径均通过 |
| 当前网络 forward / PPO loss 运行 | NOT RUN | 可用 Python 环境没有 PyTorch；本轮不安装依赖、不训练 |
| 旧 checkpoint 续训 | BLOCKED | 现有 history 显示旧的 8-action/旧节点 schema，不匹配当前 11-action |
| train/val/test split | DEFERRED BY REQUEST | 本轮按要求不修 split；formal PPO evaluation 仍是 in-trace 语义 |

结论：当前 objective、Reward、state/action interface 没有发现需要改源码的模型错误；但不能使用现有旧 checkpoint 续训，也不能在当前缺失 PyTorch 的环境中宣称 PPO 网络接口已完成运行级验证。下一步应当是准备已有模型所需的 PyTorch 运行时，构造当前 154/11 schema 的 fresh checkpoint，再执行既有的短网络/不训练试验。

## 2. 12.pdf 公式与代码口径

### 2.1 执行时延和车辆能耗

代码路径：src/common/system_model.py:32、90、131。

12.pdf 与当前实现的核心公式为：

    Local:
        T_v = L_v / F_v
        E_v = kappa_v * L_v * F_v^2

    Edge (SV / RSU / UAV):
        R_v,m = b_v,m * B_m * log2(1 + p_v * h_v,m / sigma^2)
        T_v,m = D_v / R_v,m + L_v / (w_v,m * F_m)
        E_v,m = p_v * D_v / R_v,m

    Cloud:
        T_v,c = psi_c * D_v / R_c + L_v / F_c
        E_v,c = p_v * D_v / R_c

单位链为：

| 量 | 单位 |
|---|---|
| D_v | bit |
| L_v | cycle |
| F、F_c | cycle/s |
| B、R_c | Hz、bit/s |
| T | s |
| p | W |
| E | J |
| cache service size | MB |
| UAV movement distance | m |

没有发现 input bit 再乘 8000、CPU cycle 再重复换算、Cloud psi_c 进入车辆能耗、或本地计算误加无线传输的实现问题。

### 2.2 任务 utility

12.pdf 第 8 页的单任务口径为：

    Q_v(t) =
        a_v(t) * [
            omega_T * (1 - T_v(t) / T_ref)
            + omega_E * (1 - E_v(t) / E_ref)
        ]

    Q(t) = sum_v Q_v(t)

当前任务执行器只把 arrival=1 的 Task 传给 Matching/KKT，因此代码中的 a_v(t) 由任务列表的 membership 隐式实现。system_model.task_utility 不做 clip；delay 或 energy 超过 reference 时 utility 可以为负数，这与线性归一化公式一致。

当前配置为 omega_T=0.6、omega_E=0.4、T_ref=30.79009347 s、E_ref=3.079009347 J，且配置校验要求两项权重和为 1、reference 为正。

### 2.3 Cache cost、UAV cost 和总目标

12.pdf 第 8--9 页对应：

    E_m^ca(t) = xi_m^ca * sum_f s_f * [c_mf(t) - c_mf(t-1)]^+

    C_ca(t) = sum_m [
        eta_m^ca * z_m^ca(t) + lambda_ca * E_m^ca(t)
    ]

    E_u^mov(t) = xi_u^mov * ||q_u(t) - q_u(t-1)||_2

    C_q(t) = sum_u [
        eta_u^q * z_u^q(t) + lambda_mov * E_u^mov(t)
    ]

    G(t) = Q(t) - C_ca(t) - C_q(t)

当前实现的实际 z 由最终状态和上一 slot reference state 的差异确定：

- cache node 最终 cache row 没有改变时，actual cache trigger=0，固定成本和新增加载成本均为 0；
- UAV 最终水平位置在 tolerance 内没有改变时，actual movement trigger=0，距离、能耗和固定成本均为 0；
- 输入 trigger=0 的节点/无人机不能改变；输入 trigger=1 只是允许搜索，不强制产生物理变化。

配置阶段使用预测目标：

    J(t) = Q_hat(t) - C_ca(t) - C_q(t)

但实际 slot Reward 只使用：

    G(t) = Q_actual(t) - C_ca(t) - C_q(t)

没有把 Q_hat 再加到 Reward，也没有在最终 G 中重复扣除 cache/movement cost。代码证据为 src/decision/slot_environment.py:104--312、731--1048。

## 3. slot timeline 和 Reward 数据边界

当前 slot 的实际数据流为：

    [C(t-1), Q^U(t-1), q_hat^V(t), rho_hat(t)]
        -> PPO trigger Z(t)
        -> predicted Cache/UAV configuration optimizer
        -> final C(t), Q^U(t)
        -> actual tasks at t + actual vehicle positions at t
        -> Matching
        -> same-slot edge KKT
        -> Q_actual(t), C_ca(t), C_q(t)
        -> G(t)
        -> next slot receives C(t), Q^U(t)

配置阶段读取 prediction scenario slice；实际 Matching/KKT 读取当前 tasks.csv 的 arrival=1 任务和当前真实 trajectory position。PPO observation 不读取 actual task、actual Matching、KKT、Q 或 G。

当前实现没有任务 queue state。即使某任务的完整 T 大于 dt=1 s，它仍然只在当前独立 slot 计算 utility，不把剩余时延或剩余 workload 传播到下一 slot。

infeasible 行为：

- configuration trigger 不可行或实际 Matching/KKT 不可行时，SlotExecutionResult.step_feasible=false；
- cost、actual Q、system utility 和 next state 保持为 None；
- rollout 在加入 transition 前抛出 InfeasibleActionBlocker；
- 不把 infeasible action 当成带有伪 Reward 的训练样本。

## 4. Observation exact map

代码路径：src/ppo/ppo_trigger_policy.py:112、338；rollout 统一从 src/ppo/ppo_rollout.py:884 调用该 builder。

当前 exact map 为：

| slice | 内容 | 维度 | 顺序/归一化 |
|---|---|---:|---|
| 0:90 | previous cache C(t-1) | 9 x 10 | row-major；node 顺序 23, 87, 43, 75, 58, RSU_0, RSU_1, UAV_0, UAV_1；每 row 为 service 0..9；binary |
| 90:94 | previous UAV position Q^U(t-1) | 2 x 2 | UAV_0(x,y), UAV_1(x,y)，按 flight boundary 归一化到 [0,1] |
| 94:144 | current predicted vehicle position q_hat^V(t) | 25 x 2 | artifact 顺序 23, 87, 43, 75, 58, 64, 25, 18, 95, 8, 88, 83, 9, 55, 31, 48, 19, 35, 73, 20, 50, 16, 11, 98, 47；按同一 boundary 归一化 |
| 144:154 | predicted service popularity rho_hat(t) | 10 | service 0..9；已归一化概率 |
| total | observation_dim | 154 | 90 + 4 + 50 + 10 |

因此当前 observation 不是旧的 164 或 184，也不包含队列特征、实际任务、实际位置、Matching/KKT、Q 或 G。位置和 popularity 的输入会做 finite、shape、ID、范围及归一化校验。

因果性结论：C(t-1) 和 Q^U(t-1) 是上一 slot state；q_hat^V(t) 和 rho_hat(t) 是当前 slot 可用的 prediction input；actual task 与 actual position 只进入配置完成后的实际执行链。

## 5. Action、mask、gate 和 trigger exact map

代码路径：src/ppo/ppo_trigger_policy.py:45、src/ppo/ppo_rollout.py:901、src/ppo/hierarchical_reconfiguration_gate.py:313--465。

当前 action vector 长度 11，exact labels 为：

    z_23^ca, z_87^ca, z_43^ca, z_75^ca, z_58^ca,
    z_RSU_0^ca, z_RSU_1^ca, z_UAV_0^ca, z_UAV_1^ca,
    z_UAV_0^q, z_UAV_1^q

语义为：

- 前 9 位：每个 dynamic cache node 是否允许 cache reconfiguration；
- 后 2 位：每个 UAV 是否允许 movement configuration；
- action vector 只传 trigger，不直接选择 Local、SV、RSU、UAV 或 Cloud offload target；
- target 由实际/预测 Matching 根据 local cache、edge cache availability 和 physical path feasibility 产生；
- Cloud 只在没有可执行 edge cache target 时加入候选。

当前 autoregressive mask 语义：

- cache trigger 的 action 0/1 均开放：[True, True]；
- UAV trigger 的 action 0 始终开放，action 1 只在当前 UAV prefix 和几何约束下存在可行 candidate 时开放；
- mask 只约束 trigger configuration feasibility，不把 Matching/KKT 的 residual infeasibility 静默改成另一个 action；
- action=0 的节点必须保持上一 state；
- action=1 允许搜索，但 optimizer 只有在预测 joint objective 严格改善时接受实际状态变化。

当前 active PPO 配置为：

    method = proposed_hierarchical_reconfiguration_gate_direct_cf_refdrift
    credit_assignment = hierarchical_gate_direct_cf
    actor_architecture = v3_full_shared_actor_frozen_policy_batch_v1
    cf_samples_per_episode = 8
    cf_batch_episodes = 4
    reference_probability_drift_cap = 0.005

Gate 语义：

- g=0：final environment action 被强制为全零，严格保持 cache/UAV configuration state；
- g=1：把 existing WHAT trigger actor 的 binary action 交给 environment；
- Gate 与 WHAT 使用独立 optimizer；active update 顺序是 Gate Joint-Null Direct-CF，再进行 WHAT Direct-CF/RefDrift；
- active path 不使用 learned Q critic、scalar value loss 或 scalar GAE advantage 做 actor credit；
- g=1 也不保证物理 state 一定变化；若 WHAT 全零或 optimizer 没有严格改善，仍是 no-trigger/no-change，成本为 0。这是“允许进入配置决策”而非“强制改配置”的语义。

Hierarchical NULL branch 的 compatibility-shaped joint log-prob 会携带一个确定的 all-zero WHAT evaluation term，但 active Gate/WHAT Direct-CF update 分开使用 gate log-prob 与 WHAT target log-prob；当前正式 active dispatcher 不走 generic scalar compute_ppo_loss 路径。

## 6. PPO state/action consistency、terminal 和 credit audit

### 6.1 Rollout transition

Rollout 每一步保存：

- observation；
- 11 位 environment action；
- cache/UAV trigger；
- old joint/per-action log-probability；
- old value；
- actual Reward G；
- actual Q、cache cost、movement cost；
- next cache state 和 next UAV positions；
- hierarchical path 的 gate action、old gate probability/log-prob。

下一步 observation 使用上一步的 next cache/UAV state，未重新使用初始 state。rollout 会检查：

    old joint log-prob
        = old Gate log-prob + sum(old WHAT chosen log-prob)

并检查 z=0 时对应 state row/position 不改变。

### 6.2 GAE、Monte-Carlo 和 true terminal

src/ppo/ppo_rollout.py:779 的 GAE 是标准 reverse recursion：

    delta_t = r_t + gamma * (1-done_t) * V_{t+1} - V_t
    A_t = delta_t + gamma * lambda * (1-done_t) * A_{t+1}
    return_t = A_t + V_t

true terminal 时 bootstrap=0；非 terminal 截断时使用额外 next observation 的 value 做 bootstrap。src/ppo/ppo_rollout.py:813 的 Monte-Carlo return 在 done 处重置 suffix。

formal training 明确拒绝 truncated batch，并要求最后一个 formal origin 是 true terminal。当前 active hierarchical Direct-CF actor 不使用 scalar GAE/value loss；gamma/lambda 仍保留给 legacy/return reporting 兼容，Gate/WHAT Direct-CF 使用 frozen same-state branch return difference。

### 6.3 Direct-CF reward decomposition

WHAT branch 使用：

    Delta_G = Delta_Q - Delta_Cache - Delta_UAV
    A_CF = (a - p_old) * Delta_G

Gate Joint-Null branch 使用：

    Delta_gate = Return_CONFIG - Return_NULL
    Delta_gate = Delta_Q - Delta_Cache - Delta_UAV
    A_gate = (g - p_old_gate) * Delta_gate

Direct-CF loss 函数只把已形成的 A_CF/A_gate 作为 advantage，不再重复乘一次 (a-p) 或 (g-p)。active update 不引入 learned Q critic，不把预测 Q_hat 当作 actual Reward。

## 7. Fixed-seed / short numeric checks

### 7.1 Artifact read-only check

执行：

    scripts/regenerate_artifacts.py --config system_config.toml --check-only

结果：

- mode = check_only；
- writes_performed = 0；
- all_passed = true；
- service catalog total = 111631.6688314203 MB；
- derived capacity：TV 5581.583441571016 MB、SV 16744.750324713044 MB、RSU 44652.66753256813 MB、UAV 33489.50064942609 MB；
- configuration signature、config hash、artifact SHA256、prediction dimensions 全部通过。

### 7.2 slot 20 no-trigger actual Reward

使用当前 fixed artifact、slot=20、13 个实际到达任务、cache trigger 全零、UAV trigger 全零，执行现有 execute_slot：

| 量 | 结果 |
|---|---:|
| step_feasible | true |
| actual Q | 7.80513353061257 |
| cache cost | 0.0 |
| movement energy | 0.0 J |
| movement cost | 0.0 |
| G | 7.80513353061257 |
| predictive utility/objective | None / None |
| configuration stop | no_trigger_prediction_skipped |
| cache state unchanged | true |
| UAV state unchanged | true |
| repeat matching/G difference | exact / 0.0 |

因此 no-trigger 的实际边界满足：

    G = Q_actual - 0 - 0 = Q_actual

且 prediction utility 不被错误计入 Reward。

### 7.3 一项任务的手算：slot 20 task 25 -> Cloud

slot 20 的 task 25：

    D = 6435706.350135426 bit
    L = 3500595810.2157364 cycle
    target = CLOUD

使用当前 Cloud contract：

    R_c = 200000000 bit/s
    F_c = 5000000000 cycle/s
    psi_c = 11
    p = 0.1995262315 W

手算/纯函数复算得到：

    T_upload = psi_c * D / R_c = 0.35396384925744845 s
    T_compute = L / F_c = 0.7001191620431473 s
    T_total = 1.0540830113005957 s
    E_vehicle = p * D / R_c = 0.006420461175415706 J
    Q_25 = 0.9786252142390128

实际 Matching/KKT 记录完全相同：

    delay absolute diff = 0.0
    energy absolute diff = 0.0
    Q absolute diff = 0.0

13 个任务的 per-task Q 之和与 Matching task_utility_sum 的 absolute diff 为 8.881784197001252e-16。

### 7.4 同 slot edge KKT

slot 20 的 RSU_0 两任务同 slot KKT 结果：

| 量 | task 64 | task 19 | node sum |
|---|---:|---:|---:|
| CPU fraction w | 0.5248820479982804 | 0.4751179520007409 | 0.9999999999990213 |
| bandwidth fraction b | 0.5066984451908659 | 0.4933015548087081 | 0.999999999999574 |
| delay (s) | 8.064753871182644 | 7.7695434992090275 | — |
| vehicle energy (J) | 1.3698818960059576 | 1.3336628040164773 | — |

edge KKT 同时满足资源和约束；Cloud 不进入 edge KKT map，使用固定 per-task resource。

### 7.5 full/incremental/repeat

当前 task_allocation.run_slot20_validation 处理 13 个 arrival=1 任务：

- full 与 incremental matching、J、updates、CPU/BW allocations、delay、energy、per-task Q 均一致；
- 所有最大数值差异为 0；
- multi-task edge case 与 repeat run 均一致；
- repeat_deterministic = true。

### 7.6 一条短的 feasible-trigger 路径

为避免长配置搜索，本检查在内存中构造 zero-arrival 的 scenario slice，不写入任何 artifact；输入一个 cache trigger=1、其余 trigger=0。结果：

| 量 | 结果 |
|---|---:|
| elapsed | 0.008371500007342547 s |
| trigger_feasible | true |
| step_feasible | true |
| actual Q | 0.0 |
| G | 0.0 |
| cache cost | 0.0 |
| accepted cache updates | 0 |
| final cache changed | false |
| matching count | 0 |
| stop | converged_by_no_improvement |

这个短路径验证了 trigger=1 是允许搜索而不是强制产生变化；没有严格 objective improvement 时，最终 actual state 和 cost 仍保持不变，且 Reward 恒等式成立。

### 7.7 纯 observation/action/数值边界

按当前 artifact 的 9 cache node、10 service、2 UAV、25 research vehicle 顺序构造：

- observation shape = (154,)，全 finite；
- slices = cache 0:90、UAV 90:94、predicted vehicle 94:144、popularity 144:154；
- popularity sum = 1.0；
- action_dim = 11；
- TriggerActionSpec action vector roundtrip = true；
- 当前几何位置下 11 个 mask row 均为 [1,1]；
- delay=2*T_ref、energy=2*E_ref 时 negative_Q = -1.0，没有被 clip；
- cache pure formula example total cost = 0.202；
- UAV 5 m movement pure formula：energy=75.0 J、movement cost=0.0030800000000000003；
- GAE example rewards=[1,2]、values=[0.5,0.5]、dones=[0,1] 得到 advantages=[2.424999952316284,1.5]、returns=[2.924999952316284,2.0]；
- Monte-Carlo returns=[3.0,2.0]；
- bad binary action、bad weight sum、NaN popularity、NaN cache size、NaN Gate probability 均按预期抛出 ValueError。

### 7.8 Cloud count / psi boundary

同一 task input 下：

- cloud_task_count=1 与 7 的所有返回值最大 absolute diff = 0.0；
- psi=11 的 total delay = 1.0540830113005957 s；
- psi=1 只改变通信部分，total delay = 0.7322976937938244 s；
- psi=11 与 psi=1 的 vehicle energy 都为 0.006420461175415706 J。

这与当前 12.pdf 口径“psi_c 只进入 Cloud 时延、不进入车辆上传能耗”一致。

## 8. Existing PPO history/checkpoint audit

### 8.1 当前 active config

system_config.toml 当前明确选择：

    method = proposed_hierarchical_reconfiguration_gate_direct_cf_refdrift
    credit_assignment = hierarchical_gate_direct_cf
    actor_architecture = v3_full_shared_actor_frozen_policy_batch_v1

PPO training hyperparameters 已显式存在：clip=0.2、learning_rate=3e-4、value coefficient=0.5、entropy coefficient=0、max_grad_norm=0.5、update_epochs=1、num_minibatches=4。active hierarchical Direct-CF update 另有 independent Gate/WHAT optimizer 和 Batch4 语义，且不接受 learned critic/replay。

### 8.2 旧 run 不能续用

当前 source 的 compatibility metadata 会严格比较 action labels、action_dim、observation_dim、node/service/vehicle ID、formal protocol、algorithm schema 和 checkpoint fields；src/ppo/ppo_training.py:3190 的 _metadata_mismatch 对 list 长度、key 集合和数值逐项拒绝不一致。

现有 run 的只读证据：

| 旧 run | 读到的旧接口/模型痕迹 | 判定 |
|---|---|---|
| runs/ppo_hierarchical_gate_pilot_001/training_history.json | 首个 episode 的 trigger_action1_counts 只有 8 个 label：z_46^ca、z_29^ca、z_80^ca、z_RSU_0^ca、z_UAV_0^ca、z_UAV_1^ca、z_UAV_0^q、z_UAV_1^q | 与当前 9 cache + 2 movement = 11 action 不兼容 |
| runs/ppo_refdrift_formal_200ep_001/training_history.json | cf_trigger_sample_counts 和 cf_trigger_free_counts 长度均为 8；method 是旧 sparse_direct_cf | 与当前 hierarchical active method 和 11 action 不兼容 |
| runs/formal_paired_validation_panel_v1/config_snapshot.json | research=12、TV=9、SV=3、Cloud CPU=3e9、psi=2.3、T_ref=0.65、E_ref=0.065；缺少当前 independent_per_slot/resource_sharing fields | 旧 panel，不是当前模型输入 |
| runs/hierarchical_gate_implementation_audit.json | fixture 的 Gate observation=12、WHAT action=8 | 旧小 fixture，不是当前 154/11 artifact |

因此：

- 不能把这些 latest_checkpoint.pt 当作当前 12.pdf/154/11 模型的可恢复训练状态；
- 不能用这些旧 history 直接与当前 Reward 数值做同模型性能比较；
- 本轮不删除、不覆盖、不迁移这些历史 run；
- 正确下一步是从当前 config/artifact 重新构造 current-compatible fresh checkpoint，或明确把旧 run 标记为历史，不做 resume。

## 9. Issue levels and blocking status

### P0 — checkpoint resume blocker

旧 PPO run/history 仍保存旧 8-action、旧节点 ID 或旧 panel config。它们不能续训当前 12.pdf 冻结模型。该问题不要求本轮改 artifact；训练前必须新建 current-compatible checkpoint 或放弃 resume。

### P1 — runtime verification blocker

当前可用 Python runtime 和系统 Anaconda 均没有 PyTorch。因而本轮没有运行 Conditional/Full Shared Actor、hierarchical Gate forward、old/new log-prob evaluation、Direct-CF loss backward 或 optimizer step。这是环境依赖阻断，不是已发现的 objective/reward/state/action 公式错误；本轮没有安装依赖。

### P2 — evaluation disclosure

FormalTrainingProtocol 当前是 single-trace/in-trace deterministic evaluation，不是 held-out/generalization evaluation。按用户要求，本轮不修 train/val/test split；后续结果不能把 in-trace 指标写成 held-out 泛化结论。

### P3 — historical documentation hazard

reports/baseline_v2_alignment_repair_validation.md 仍保留 pre-freeze 的 observation=184/queue 描述；reports/ppo_interface_audit.md 等旧文档也可能保留旧阶段说明。它们应继续作为历史记录，不能覆盖当前 source、system_config.toml 和 12.pdf 的冻结口径。

### Current source verdict

没有发现需要通过增加 queue、complexity、action type、direct offload branch、额外网络或新算法来修复的 active-path issue。当前模型实现可以进入“准备 current runtime 后的最小网络接口试验”，但在 PyTorch 缺失和旧 checkpoint 未淘汰之前，不应开始正式续训。

## 10. Verification record and unchanged-scope confirmation

本轮已完成：

- 读取并渲染审阅 C:\Users\tangning\Desktop\12.pdf；
- 静态审计 current objective、actual Reward、state propagation、action/mask/gate、Direct-CF、terminal/GAE/MC；
- 完成 slot 20、Cloud、KKT、no-trigger、short feasible-trigger、roundtrip 和 invalid-input checks；
- compile/import check passed；
- formal config validation passed；
- artifact check-only passed，writes_performed=0；
- git diff --check passed；
- 没有运行 PPO/GRU training、没有安装依赖、没有运行长 SUMO；
- 没有修改任何源码、配置、artifact、checkpoint、训练 history 或历史报告；
- 本文件是本轮唯一新增文件。

主要代码证据：

- src/common/system_model.py:32--50、90--128、131--178、224--284、299--345、347--373；
- src/decision/task_allocation.py:82--154、1210--1385、1420--1510；
- src/decision/cache_configuration.py:560--716、973--1206；
- src/decision/slot_environment.py:104--312、731--1048；
- src/ppo/ppo_trigger_policy.py:45--180、338--430；
- src/ppo/ppo_rollout.py:779--839、884--965、965--1408、1964--2050；
- src/ppo/ppo_update.py:194--236、583--780；
- src/ppo/ppo_training.py:2966--3145、3190--3215、6704--6810。

