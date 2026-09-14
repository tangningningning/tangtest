# 任务数学模型完整审计

审计日期：2026-09-06
审计对象：C:\Users\tangning\Desktop\code\adaptive_cache_mec
审计性质：只读审计；没有修改算法、参数、reward、PPO、缓存机制或正式 artifact。

本报告把 C:\Users\tangning\AppData\Local\Temp\codex-file-preview-HWc0Ve\12.pdf 作为本项目的主系统模型与公式基准。第一份 IEEE 论文 Joint_Service_Caching_Communication_and_Computing_Resource_Allocation_in_Collaborative_MEC_Systems_A_DRL-Based_Two-Timescale_Approach.pdf 只作为外部参考，单独列出一致和差异，不能替代 12.pdf 的定义。

## 1. 审计范围与已有报告去重

本次检查了：

- 统一配置：configs/system_config.toml。
- 任务与流行度生成：src/scenario/task_scenario.py。
- 预测场景生成：src/scenario/scenario_generator.py、src/scenario/popularity_prediction.py。
- 实际任务读取和 Matching/KKT 输入：src/decision/task_allocation.py。
- 预测配置与实际调度的边界：src/decision/cache_configuration.py、src/decision/slot_environment.py、src/ppo/ppo_rollout.py。
- 时延、计算和能耗纯公式：src/common/system_model.py。
- 生成增量检查和 metadata：scripts/regenerate_artifacts.py。
- 正式 artifact：data/generated/service_catalog.json、popularity.csv、popularity_history.csv、tasks.csv、popularity_predictions.csv、prediction_scenarios.npz、generation_metadata.json。
- 已有报告：reports/artifact_consistency_report.md、reports/objective_reward_ppo_interface_final_audit.md。

此前的任务生成审计结论已并入本报告；本报告补充实际 artifact 的固定 seed 统计、运行时消费链、两份 PDF 的逐项公式对照和当前可进入下一模块的判定。

两份 PDF 均可访问。第一份为 14 页，12.pdf 为 12 页。PDF 公式对照使用了文字提取和代表性页面渲染复核；12.pdf 第 2、3、5、7、8 页是本次任务模型对照的主要页面。

## 2. 总结结论

### 2.1 总体判定

当前任务数学模型、配置、任务 artifact、预测 artifact 和运行时任务对象之间没有发现严重的单位、概率、索引或数据流错误。

当前实现与本项目主模型 12.pdf 的核心任务结构一致：

1. 任务车辆每个基础时隙最多一个不可分割任务。
2. 到达变量为二值；当前工程进一步明确为按 p=0.7 独立 Bernoulli 抽样。
3. 只有到达时才抽服务类型；服务类型条件分布使用动态 Zipf 流行度。
4. service_id 在工程内部统一使用 0-based 的 0..9；12.pdf 使用数学记号 1..F，这是索引约定差异，不是跨文件错位。
5. 有效任务的输入量以 bit 保存，计算量以 cycle 保存，运行时不再重复做 KB 转换。
6. 计算需求满足 L=D×有效计算密度，其中有效计算密度为 catalog 的 cycles_per_bit 乘以每任务的负载扰动因子。
7. 无任务行保存为空 service_id、0 input_size、0 cpu_cycles；实际任务读取器只构造 arrival=1 的任务对象。
8. 预测 popularity 通过 GRU 生成 rho_hat，先用于预测任务场景抽样，也进入 PPO 配置观测；缓存/UAV 配置使用预测场景样本，实际 Matching 使用当前实际任务。

### 2.2 是否可以进入通信模型审计

可以。任务模型审计门槛通过，建议进入通信模型审计。

进入下一模块时必须保留三个口径说明：

- tasks.csv 和 prediction_scenarios.npz 的 input_size 已经是 bit，通信公式不能再次乘 8000。
- 当前 cache/UAV 配置的预测效用采用 rho_hat 采样得到的未来任务场景，不是直接对 rho_hat 做解析期望；这是当前实现的 Monte Carlo 语义。
- 工程使用十进制 KB，即 1 KB=1000 byte=8000 bit。12.pdf 只写 KB，没有进一步声明十进制或二进制口径；这属于外部规格需要明确的合同风险，不是当前代码和 artifact 的内部不一致。

## 3. 数学模型定义与当前实现

令任务车辆集合为 V_T，正式值为 20 个 TV；服务集合为 F={0,...,9}，正式值为 10 个 service；时隙集合为 t=0,...,299，dt=1 s。

### 3.1 任务到达 A_v(t)

当前实现的正式模型为：

    A_v(t) ~ Bernoulli(lambda),       lambda = 0.7

其中每个 t 对全部 TV 各抽一次，不共享一个总任务数约束，也没有按服务类型对到达概率做额外加权。代码证据：

- configs/system_config.toml:84、122-124：TV 数量和 arrival_probability=0.7。
- src/scenario/task_scenario.py:803-829：逐 slot 生成 arrival，并对每个 TV 生成一行。
- src/scenario/task_scenario.py:367-379：master_seed 派生独立 task_arrival、service_request、workload_variation 随机流。

有效任务的生成流程是：

    A_v(t)=0  ->  不抽 f_v(t)、D_v(t)、L_v(t)
    A_v(t)=1  ->  抽 f_v(t)，随后抽 D_v(t) 和 L_v(t)

### 3.2 服务请求 f_v(t)

每个时隙的真实请求分布为动态 Zipf：

    rho_f(t) = r_f(t)^(-zeta) / sum_j r_j(t)^(-zeta)
    zeta = 0.8

当 A_v(t)=1 时：

    Pr[f_v(t)=f | A_v(t)=1] = rho_f(t)

当前动态 rank 在每个 20-slot epoch 内保持不变，在 epoch 边界执行一次局部 rank 交换或移动。代码证据：

- src/scenario/task_scenario.py:606-618：按 rank 计算 Zipf 概率。
- src/scenario/task_scenario.py:621-651：epoch 边界局部演化 rank。
- src/scenario/task_scenario.py:675-795：生成 prehistory 和 formal 的连续 rho_star。
- src/scenario/task_scenario.py:825-851：arrival=1 时使用 request_rng.choice(..., p=probabilities)。

### 3.3 输入量 D_v(t)

对 service f，catalog 中保存的基准输入量为：

    D_f^0 = input_size_kb(f) * 1000 * 8 bit

有效任务输入量为：

    D_v(t) = D_f^0 * (1 + u_D)
    u_D ~ Uniform(-0.20, +0.20)

代码使用半开区间 [−0.20,+0.20)，因此理论有效 envelope 为：

    400 KB <= D_v(t)/8000 < 6000 KB

代码证据：

- src/scenario/task_scenario.py:119-121：SERVICE_SIZE_MB_PER_GB=1000、INPUT_SIZE_BITS_PER_KB=8000。
- src/scenario/task_scenario.py:433-467：catalog 的 input_size_kb 转换成 mean_input_size bit。
- src/scenario/task_scenario.py:853-865：任务级输入扰动。
- data/generated/service_catalog.json:units：明确 mean_input_size 和 tasks.input_size 均为 bit。

### 3.4 计算需求 L_v(t)

对 service f，catalog 中保存基准计算密度 c_f，单位为 cycle/bit。任务计算需求为：

    L_v(t) = D_v(t) * c_f * (1 + u_L)
    u_L ~ Uniform(-0.10, +0.10)

因此任务的有效计算密度是：

    c_v,eff(t) = L_v(t) / D_v(t) = c_f * (1 + u_L)

代码证据：

- src/scenario/task_scenario.py:865-870：直接执行 cycles_per_bit × input_size × load_factor。
- src/scenario/scenario_generator.py:631-652：预测任务场景采用同一公式。
- src/common/system_model.py:8-13：运行时约定 input_bits=bit、cpu_cycles=cycle、CPU=cycle/s。

这里需要区分两个名词：

- catalog 的 cycles_per_bit：固定 service 基准密度，范围为 400..1000 cycle/bit。
- 有效任务密度 L/D：包含 ±10% load jitter，理论范围约为 360..1100 cycle/bit。

所以“L=D×computation_density”严格成立，但 computation_density 必须定义成有效任务密度，而不能只指未扰动的 catalog 字段。

## 4. 配置、artifact 与运行时映射

| 数学量/概念 | 配置入口 | 生成器/Artifact | 运行时消费 | 审计结论 |
|---|---|---|---|---|
| TV/SV/F、slot 时间 | configs/system_config.toml:76-89、96-104 | generation_metadata.json；vehicle_roles.json | task_scenario、scenario_generator、slot_environment | 25=5 SV+20 TV，300 slot，dt=1 s，一致 |
| lambda | configs/system_config.toml:122-124 | service_catalog.json generation_config、tasks.csv arrival | task_scenario.generate_tasks；task_allocation.load_actual_tasks | 读取统一配置，正式 p=0.7 |
| zeta/rank/rho_star | configs/system_config.toml:125-130 | popularity.csv、popularity_history.csv | popularity_prediction 作为监督标签和诊断；任务生成已固化到 tasks.csv | 动态 Zipf，一致 |
| service catalog | configs/system_config.toml:42-43 | service_catalog.json，F=10，ID=0..9 | cache_configuration.load_service_catalog | 服务大小 MB、D 基准 bit、cycles_per_bit 分别读取 |
| 真实任务 | configs/system_config.toml:50、task_scenario.py:1147-1216 | tasks.csv，20×300=6000 行 | task_allocation.load_actual_tasks:1740-1756 | 只读取当前 slot 且 arrival=1 的行 |
| 预测 popularity | paths.popularity、paths.popularity_history、paths.popularity_gru | popularity_predictions.csv、prediction_scenarios.npz | popularity_prediction:247-281；scenario_generator:528-555、1060-1091 | 只用 origin 前历史构造 rho_hat |
| 预测任务 D/L | scenario_generator:593-659 | prediction_scenarios.npz | cache_configuration:468-559、795-818 | 以 rho_hat 抽出的样本驱动预测配置效用 |
| 实际执行时延 | nodes.* CPU 和 system_model.py | 不另存转换后的单位 | task_allocation:1316-1372、system_model.py:32-178 | D/L 直接进入 bit/cycle 公式，无重复换算 |

### 4.1 运行时完整数据流

    configs/system_config.toml
        -> task_config_from_project_config()
        -> task_scenario.build_artifacts()
        -> service_catalog.json + popularity*.csv + tasks.csv
        -> popularity_prediction.build_causal_formal_inputs()
        -> GRU rho_hat
        -> scenario_generator.sample_tasks()
        -> prediction_scenarios.npz
        -> cache_configuration.scenario_at_origin()
        -> 预测配置 evaluator / Cache-UAV optimizer
        -> slot_environment.execute_slot()
        -> actual tasks.csv -> task_allocation.load_actual_tasks()
        -> Matching/KKT -> system_model.local/edge/cloud_execution()

其中配置阶段使用 scenario_slice，实际调度阶段使用 actual_tasks 和 actual_vehicle_positions。slot_environment.py:125-130 明确写出预测配置与实际 Matching/reward 的边界；slot_environment.py:243-312 也显示实际 Matching 使用 actual_tasks，最终系统效用由实际任务效用减去配置和移动成本得到。

## 5. 任务到达审计

### 5.1 概率来源与独立性

结论：是 Bernoulli/概率到达；概率来自 configs/system_config.toml 的 task.arrival_probability=0.7。

代码没有为不同车辆设置不同 lambda，也没有根据 slot、服务类型、位置或历史请求动态改变 lambda。每个 TV-slot 都从 arrival_rng 取得一个随机数并与 0.7 比较。arrival_rng 是独立于服务抽样和负载抽样的 RNG stream；不同车辆和 slot 在实现上是同一伪随机流的连续独立抽样，而不是共享总数的无放回抽样。

固定 seed=4401 的 artifact 统计：

| 指标 | 结果 |
|---|---:|
| TV×slot 机会数 | 6000 |
| arrival=1 数量 | 4210 |
| empirical arrival rate | 0.701667 |
| 每 slot 到达数 min/mean/max | 8 / 14.033333 / 19 |
| 理论每 slot 期望 | 20×0.7=14 |
| 每 TV 每 slot 最大有效任务数 | 1 |
| 跨车辆 arrival 序列最大绝对相关系数 | 0.1571 |
| 同一车辆相邻 slot 相关系数 min/mean/max | -0.0584 / 0.0011 / 0.0687 |

相关系数只能作为有限样本诊断，真正的独立性结论来自代码结构：没有跨车辆/跨 slot 的共享计数约束或状态依赖。

### 5.2 无任务处理

arrival=0 时 tasks.csv 写入：

    service_id = ""
    input_size = 0
    cpu_cycles = 0

实际任务加载器在 src/decision/task_allocation.py:1740-1756 同时过滤 slot 和 arrival=1，因此无任务行不会变成伪 Task 对象。预测场景使用 service_id=-1、D=0、L=0，scenario_generator.py:729-773 对该不变量进行了检查。

### 5.3 每个 TV 每 slot 最多几个任务

严格最多一个。生成器的循环粒度是一个 TV-slot 行，arrival 是标量门控，不存在第二个任务循环；6000 行恰好覆盖 20 TV×300 slot。该结构与 12.pdf 第 3 页“每辆任务车辆在一个时隙内至多产生一个不可分割任务”一致。

## 6. 服务请求、索引与 popularity 使用审计

### 6.1 真实请求是否真正 dynamic Zipf

是。固定 seed artifact 的数值检查为：

| 检查 | 结果 |
|---|---:|
| 每个 slot 的 rho_star 和最大误差 | 1.11e-16 |
| artifact rho_star 与 rank^(-0.8) 公式最大误差 | 1.39e-17 |
| formal episode 中不同 rank 向量数 | 11 |
| 相邻 slot rank 变化次数 | 14 |
| service request count 与平均 rho_star 的相关系数 | 0.998353 |

当前 300 slot 中，rank 每 20 slot 检查一次 epoch 边界；一次 epoch 内不变，边界使用一次局部变化。实际总请求计数为：

    service 0..9:
    [470, 184, 285, 306, 244, 443, 191, 802, 229, 1056]

这不是 uniform sampling；service 9 和 service 7 的请求数较高与其较高平均 rho_star 相符。

### 6.2 service_id 范围和索引

当前所有任务相关 artifact 和运行时数组统一采用：

    service_id ∈ {0,1,2,3,4,5,6,7,8,9}

证据：

- task_scenario.catalog_by_id() 要求覆盖 range(len(catalog))，见 task_scenario.py:475-479。
- task_scenario.validate_artifacts() 要求 expected_ids=set(range(config.num_services))，见 task_scenario.py:1236-1241、1304-1306。
- cache_configuration.load_service_catalog() 要求连续 ID，见 cache_configuration.py:256-277。
- prediction_scenarios.npz 的 F=10，service_id 只在到达位置取 0..9，空位取 -1。

12.pdf 的数学集合写成 F={1,...,F}；代码改用 0-based 是 Python/NumPy 索引约定，跨 artifact 与运行时一致，没有把 service 10 当作合法 ID，也没有残留 20-service 维度。

### 6.3 预测 popularity 是否真正进入决策

结论：进入，但分成两个层次，不能把“直接使用 rho_hat 权重”和“通过预测场景抽样使用”混为一谈。

1. 预测来源。popularity_prediction.py:247-281 由 history request_count 和 formal request_count 重建因果 rho_obs；scenario_generator.py:541-555 只把截至 origin-1 的窗口输入 GRU，输出 rho_hat。
2. 预测任务抽样。scenario_generator.py:618-629 对每个 origin、scenario、horizon、TV 使用 rho_hat 作为 Categorical 概率；scenario_generator.py:643-652 用同一 catalog 生成预测 D/L。
3. 预测配置。cache_configuration.py:468-559 将预测场景的 arrival、service_id、input_size、cpu_cycles 变成 SurrogateEvent；cache_configuration.py:795-818 直接按预测样本中的 service_id、D、L 计算 service value。slot_environment.py:186-205 把 scenario_slice 传给 Cache-UAV 配置优化。
4. PPO 观测。ppo_rollout.py:887-905 将 popularity_predictions.csv 的预测概率作为 predicted_service_popularity 进入配置观测；ppo_rollout.py:1999-2075 对其与 slot/time、service 集合进行对齐检查。
5. 实际卸载。task_allocation.py:125-149 根据当前实际任务的 service_id、local cache 和 edge cache 构造可执行节点集合；task_allocation.py:9 明确 Matching 不读取 prediction_scenarios 作为实际调度输入。这个边界与 12.pdf 第 1-2 页“配置阶段用历史/预测，随后调度阶段用当前实际位置和任务请求”一致。

因此当前没有“prediction 只生成但完全未使用”的问题。需要记录的语义限制是：cache_configuration 的预测效用消费的是 rho_hat 抽样后的未来任务样本，rho_hat 本身不是在 evaluator 中直接作为每个 service 的解析权重。若未来论文或报告声称使用了 E[rho_hat] 的解析期望，需要另行确认或改模型；当前代码实现的是有限场景 Monte Carlo 估计，S=20。

popularity_predictions.csv 和 prediction_scenarios.npz 中的 rho_hat 已做交叉核对：15000 行 CSV 与 NPZ 的 [300,5,10] 概率逐元素最大绝对差为 0；CSV 归一化最大浮点误差为 1.71e-7，低于代码 1e-5 容差。当前 PPO 尚未正式训练，因此这里只审计调用链，没有运行训练。

## 7. 单位审计

### 7.1 D 的 KB -> byte -> bit

当前工程采用十进制 KB：

    1 KB = 1000 byte
    1 byte = 8 bit
    1 KB = 8000 bit

不存在 1024×8 的隐式转换，也没有在 catalog、任务生成和 runtime 之间重复转换。

| 层次 | 字段 | 实际单位 | 证据 |
|---|---|---|---|
| TOML 配置 | input_size_min_kb/max_kb | KB，500..5000 | configs/system_config.toml:114-120 |
| service_catalog.json | input_size_kb | KB，展示/追溯 | service_catalog.json services |
| service_catalog.json | mean_input_size | bit | service_catalog.json units；task_scenario.py:467 |
| tasks.csv | input_size | bit | service_catalog.json units；task_scenario.py:883 |
| prediction_scenarios.npz | input_size | bit | scenario_generator.py:1203-1217 |
| Task 对象 | input_bits | bit | task_allocation.py:938-944、1751-1753 |
| system_model | input_bits、rate | bit、bit/s | system_model.py:8-13、120-127 |

固定 seed 正式任务的输入统计：

| D 统计 | min | mean | max |
|---|---:|---:|---:|
| bit | 5,389,126.813 | 18,270,464.218 | 47,026,162.590 |
| 十进制 KB | 673.640852 | 2283.808027 | 5878.270324 |

理论 envelope 为 3,200,000 bit <= D < 48,000,000 bit，即 400 KB <= D < 6000 KB；实际样本不需要覆盖 envelope 两端。

### 7.2 L 和 cycles_per_bit

catalog 基准 cycles_per_bit 的实际 artifact 范围为 402.704753..990.204529 cycle/bit；配置目标范围为 400..1000。任务有效密度含 ±10% load factor，固定 seed 统计为：

| 指标 | min | mean | max | 单位 |
|---|---:|---:|---:|---|
| L/D 有效密度 | 362.535299 | 724.742977 | 1088.993884 | cycle/bit |
| L | 2.331817e9 | 1.387014e10 | 3.938091e10 | cycle |

对每个有效任务：

    D / catalog.mean_input_size ∈ [0.800099, 1.199914]
    (L/D) / catalog.cycles_per_bit ∈ [0.900006, 1.099999]

这证明 input jitter 和 load jitter 各加入一次，没有发现漏乘或重复乘。固定 seed 的内存重建与 tasks.csv 逐字段一致，说明这条单位/数值链没有静默漂移。

### 7.3 运行时公式

system_model.py 使用：

    T_local = L/F
    T_edge = D/R + L/(wF)
    T_cloud = psi_c*D/R_c + L/F_c

其中 D 已为 bit，L 为 cycle，F 为 cycle/s，R 为 bit/s，结果为 s。task_allocation.py:1321-1357 将 Task.input_bits、Task.cpu_cycles 原样传入公式。Cloud 当前还实现了同 slot 多任务共享总 CPU 和总带宽，cloud_execution() 在 system_model.py:165-169 使用 total/count；这是当前系统的资源扩展，不是任务单位错误。

## 8. 固定 seed 抽样统计与数值合理性

固定 seed=4401、当前正式 tasks.csv、只统计 arrival=1：

| 指标 | min | mean | max | 单位 |
|---|---:|---:|---:|---|
| D | 5.389127e6 | 1.827046e7 | 4.702616e7 | bit |
| D | 673.640852 | 2283.808027 | 5878.270324 | 十进制 KB |
| effective cycles/bit | 362.535299 | 724.742977 | 1088.993884 | cycle/bit |
| L | 2.331817e9 | 1.387014e10 | 3.938091e10 | cycle |

只考虑纯计算、不含通信、不含排队的代表性时延：

| 执行位置 | CPU | min | mean | max |
|---|---:|---:|---:|---:|
| TV local / SV | 1.0 GHz | 2.331817 s | 13.870136 s | 39.380915 s |
| RSU / UAV | 5.0 GHz | 0.466363 s | 2.774027 s | 7.876183 s |
| Cloud 单任务等效 | 20.0 GHz | 0.116591 s | 0.693507 s | 1.969046 s |

解释：

- dt=1 s，而 TV/SV 平均单任务纯计算时间约 13.87 s；车辆侧是明显的高负载配置，连续任务会形成跨 slot queue。
- RSU/UAV 平均约 2.77 s，仍大于 1 s，边缘侧也可能积压。
- Cloud 单任务等效均值低于 1 s，但实际同 slot Cloud 任务共享 20 GHz 和 200 Mbps 总资源；并发数增加时单任务计算和传输时延会按共享比例变大。
- 这说明当前 baseline 的负载偏重，但没有说明 D/L 单位错；负载结论与现有 reports/computation_load_analysis.md 一致。

## 9. Artifact 一致性与旧参数残留检查

### 9.1 只读重建与 hash

没有调用正式 artifact 输出入口，没有重写 data/generated 文件。使用当前配置和 seed 在内存中调用 build_artifacts()，结果如下：

- task_scenario.validate_artifacts()：PASS。
- 内存重建 task_rows 与磁盘 tasks.csv：逐字段一致。
- generation_metadata.json 中记录的 roles、trajectory、catalog、popularity、popularity_history、tasks、tv_local_cache、vehicle_predictions、prediction_scenarios、popularity_gru、node_positions 共 11 个 SHA-256：全部匹配。
- project_config.validate_config() 和 validate_for_formal_experiment()：PASS。

### 9.2 规模和旧字段

当前正式 artifact 的 F、维度和 ID 均为 10/0..9：

- service_catalog.json service_count=10。
- popularity.csv 为 300×10 网格。
- popularity_history.csv 为 20×10 网格。
- tasks.csv 为 6000 行，只有有效行使用 0..9。
- popularity_predictions.csv 为 300×5×10=15000 行。
- prediction_scenarios.npz 的 rho_hat shape=(300,5,10)，任务 shape=(300,20,5,20)。

在任务生成、预测场景、cache configuration 和 task allocation 的源代码中，没有发现把 service 数量固定成 20 的正式任务维度；搜索到的 20 主要是 20 TV、20-slot history/epoch、20 个预测 scenario 等合法参数。

### 9.3 预测场景不变量

对现有 NPZ 做了只读数组检查：

- arrival rate=0.700193。
- rho_hat 每个 origin/horizon 的和最大误差 2.38e-7。
- arrival=0 的 service_id 唯一值为 -1。
- arrival=0 的 input_size 和 cpu_cycles 最大值均为 0。
- metadata 标记 S=20、H_p=5、F=10、lambda=0.7、same_seed_reproducible=true。

当前环境的 Python 运行时没有安装 torch，因此没有重新调用依赖 torch 的 scenario_generator.verify_scenarios()；上面的 NPZ 数组检查、metadata 检查以及生成时保存的同 seed 标志仍已完成，未因此修改任何 artifact。

## 10. 以 12.pdf 为主的公式逐项对照

| 12.pdf 位置/公式语义 | 当前系统对应 | 结论 |
|---|---|---|
| 第 2 页：F={1,...,F}，rho_f(t)=[r_f(t)]^(-zeta)/sum_j[r_j(t)]^(-zeta)，sum rho=1 | task_scenario.py:606-618；popularity.csv；工程内部 f=0..9 | 核心公式完全一致；仅 1-based -> 0-based 索引重编号 |
| 第 2 页：由历史服务请求预测 rho_hat(t) | popularity_prediction.py:247-281；scenario_generator.py:541-555；PPO observation | 信息边界和使用方向一致 |
| 第 3 页：a_v(t)∈{0,1}，arrival=1 表示有任务 | task_scenario.py:825-845；tasks.csv arrival | 完全一致 |
| 第 3 页：每辆任务车每 slot 至多一个不可分割任务 | 一个 TV-slot 一行；6000=20×300 | 完全一致 |
| 第 3 页：J_v(t)={f_v(t),D_v(t),L_v(t)} | tasks.csv 的 service_id、input_size、cpu_cycles；Task 对象 | 完全一致，字段名做了工程化映射 |
| 第 3 页：Pr{f_v(t)=f | a_v(t)=1}=rho_f(t) | generate_tasks() 在 arrival=1 才按 rho_star.choice；sample_tasks() 按 rho_hat.choice | 完全一致；预测场景是 rho_hat 版本 |
| 第 3 页：服务车辆只提供缓存和计算，不产生任务 | task_scenario 只接收 task_vehicles；vehicle_roles 分离 SV/TV | 完全一致 |
| 第 4 页：本地 T=L/F、E=kappa L F^2 | system_model.local_execution():32-50 | 公式一致 |
| 第 5 页：边缘 T=D/R+L/(wF) | system_model.edge_execution():90-128 | 公式一致 |
| 第 7-8 页：云端 T=psi_c D/R_c+L/F_c，能耗不乘 psi_c | system_model.cloud_execution():131-178 | 公式一致，并增加同 slot 总资源均分实现 |
| 第 8 页：a_v(t)=0 时 T=E=0 | 无任务行 D/L=0；实际 loader 不建 Task；预测数组置零 | 语义一致 |
| 第 1-2 页：配置阶段使用预测，任务调度使用实际任务/位置 | slot_environment.py:125-130、186-205、243-312；task_allocation.py:9 | 完全一致 |

### 10.1 12.pdf 没有明确给出的部分

12.pdf 定义了 D_v(t) 和 L_v(t) 的符号、单位语义和执行公式，但未在任务模型部分给出当前工程所用的 service-specific 分层采样、D 的 ±20% 扰动或 L 的 ±10% 扰动分布。因此：

- 当前 500..5000 KB、400..1000 cycle/bit、±20%/±10% 是工程 baseline 校准和用户给定约束的实现。
- 不能把这些具体随机分布声称为 12.pdf 中逐字给出的公式。
- 这属于“合理改造/参考未规定”，不是代码和 artifact 不一致。

## 11. 第一份 IEEE 论文的独立外部对照

第一份论文不是本项目主模型。其第 4 页定义了任务三元组 {task type, input data in bits, required computational capacity in CPU cycles}，第 10 页仿真设置给出：

- 10 types of tasks/services。
- 每个 TD 每个 small-timescale slot 随机生成一个 task。
- Zipf parameter=0.8。
- input size=[500,5000] KB。
- computing frequency of task=[400,1000] cycle/bit。
- TD computing capacity=1 GHz，CS=5 GHz。

当前系统的对照结果：

| 外部论文设定 | 当前系统 | 判定 |
|---|---|---|
| F=10 | F=10 | 一致 |
| Zipf=0.8 | zeta=0.8 | 一致 |
| D 范围 500..5000 KB | catalog 基准范围相同，任务有效 D 加 ±20% | 合理改造 |
| 400..1000 cycle/bit | catalog 基准范围相同，任务有效密度加 ±10% | 合理改造 |
| 每个 TD 每 slot 必有一个任务 | 当前为 p=0.7 Bernoulli，允许无任务 | 与外部论文不同；与 12.pdf 的二值 arrival 和当前 baseline 一致 |
| CS=5 GHz | 当前 Cloud=20 GHz | 外部参数不同；不能据此否定当前 baseline |

因此，第一份论文支持当前服务数量、Zipf 参数和基准 D/cycles 范围的量级，但不应被用来推导当前系统必须每 slot 必到达，或 Cloud 必须为 5 GHz。

## 12. 问题清单

### 严重问题

无。

没有发现会使任务输入进入运行时后发生数量级错误、service_id 越界、无任务被当作任务、概率未归一化、任务多生成或正式 artifact 与当前配置脱节的问题。

### 中等问题

1. **预测 popularity 的消费是样本间接消费，而不是解析期望。**
   Cache evaluator 主要读取预测样本中的 arrival/service_id/input_size/cpu_cycles；rho_hat 本身不作为 service value 的直接权重。当前语义是 S=20 的 Monte Carlo surrogate，代码和 metadata 均与此一致。如果论文目标要求 cache utility 直接按 rho_hat 加权，则需要在后续模型审计中明确这是“合理改造”还是需要调整。

2. **KB 的十进制/二进制口径需要在主规格中写死。**
   代码、metadata 和 artifact 已统一使用 1000×8=8000 bit/KB；12.pdf 只写 KB，没有定义 1000 或 1024。如果外部复现实验按 KiB 解释，D、L、传输时延会出现约 2.4% 的系统性尺度差异。当前不是内部 bug，但应在通信审计和论文参数表中明确。

### 提示

1. **存在非正式调用路径的静默回退。**
   scenario_generator.py:558-581 在 project_config=None 且 catalog generation_config 缺字段时，将 arrival_probability 回退到 0.5；task_scenario.TaskConfig 也保留函数级默认值。正式 main() 在 scenario_generator.py:884-992 传入 configs/system_config.toml，因此当前正式 artifact 没有触发该回退；但后续建议缺关键任务参数时 fail-closed，避免 direct API 调用产生另一套模型。

2. **0-based service_id 需要对外文档化。**
   12.pdf 使用 1-based 数学集合，工程使用 0-based 数组索引。当前所有 artifact 和 runtime 已统一，不是问题，但公式转代码或结果表必须注明映射。

3. **prediction artifact 有两种一致表示。**
   popularity_predictions.csv 供 PPO 对齐/观测使用，prediction_scenarios.npz 供场景和 Cache-UAV surrogate 使用；当前逐元素一致，不能删掉任一文件而不更新依赖检查。

4. **当前计算负载远大于 1 s slot。**
   这是 baseline 数值压力提示，不是任务数学模型错误。通信模型审计应继续检查通信时延和 queue 语义是否与该负载共同作用。

## 13. 审计结论与下一步建议

### 审计结论

- 任务到达：Bernoulli(p=0.7)，每 TV-slot 最多一个任务，无到达处理正确。
- 服务请求：真实任务使用动态 Zipf rho_star；预测任务使用 GRU rho_hat；service ID=0..9 全链路一致。
- D 单位：配置 KB -> catalog bit -> task/runtime bit，使用 8000 bit/KB，一次转换，无重复转换。
- L 单位：cycle，满足 D×cycles_per_bit×load_factor；±10% 只加入一次。
- artifact：当前配置、内存固定 seed 重建、磁盘 tasks.csv、prediction_scenarios.npz、metadata hash 全部闭环。
- 预测使用：进入 PPO 配置观测和 Cache-UAV 预测配置效用；实际 Matching 使用当前实际任务，不错误地用未来预测任务调度。
- 参考模型：以 12.pdf 为主时核心任务模型一致；具体 D/L 随机分布是项目 baseline 校准，不是 12.pdf 明文公式。

### 建议进入通信模型审计

建议进入，判定为 **通过任务模型门槛，可进入下一模块**。

通信审计建议沿用以下已冻结接口：

1. 输入 D 统一按 tasks.csv/prediction_scenarios.npz 的 bit 读取，不再做 KB 转换。
2. 分别核对 local、SV、RSU、UAV、Cloud 的 D/R、L/(wF)、Cloud psi_c 和共享资源分配。
3. 以 12.pdf 第 5-8 页的距离、信道、带宽、边缘/云端传输公式为主；第一份 IEEE 论文只作为外部对照。
4. 维持“预测配置、实际任务调度”的信息边界，不让通信模块把 rho_hat 误当作当前实际请求。
5. 在通信审计报告中明确十进制 KB、Cloud 总带宽/总 CPU 的共享语义。

本阶段没有开始 PPO 正式训练，也没有进行长时间仿真。

## 14. 最终文件与工作树检查

本审计新增文件：

    reports/task_mathematical_model_audit.md

除该报告外，本次没有修改任何文件；用户原有的其他未跟踪 reports 文件均保留不动。固定 seed 诊断全部在内存中完成，没有重新生成正式 artifact，也没有留下项目内临时脚本或缓存。

最后执行 git diff --check：

    结果：通过，无 whitespace error。
