# MAIN_EXPERIMENT_IO_CONTRACT

审计时间：2026-09-18（Asia/Shanghai）

## 审计边界

本报告只依据当前源码、正式 run manifest/summary/JSONL，以及已经存在的 frozen episode pool 做静态审计。没有启动训练、rollout、评估、数据生成、GPU 任务或新的学习率实验；没有修改 PPO、Q/KL/entropy/reward/system 参数。

状态：

- MAIN_EXPERIMENT_IO_CONTRACT = PASS_WITH_LIMITATIONS
- OBSERVATION_DIM = 400（四个正式 V11-A run 的 manifest 与源码计算一致）
- ACTION_DIM = 11（9 个 Cache trigger 位 + 2 个 UAV trigger 位；primitive operation/candidate 决策另存为 head tensors）
- REWARD_PATH = CONFIRMED
- MASK_LOG_COVERAGE = PARTIAL：源码有完整动态 mask；正式 JSONL 只持久化了选中 operation、UAV mask 摘要和逐 transition 的 UAV operation mask，没有持久化完整 Cache operation/candidate mask 张量

## 正式输入与固定顺序

四个正式 run 都使用同一份 frozen pool：

runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911

场景 artifact 的静态形状为：

- origin 数量：300，origin slots 为 0..299
- 车辆：25（5 个 service vehicle + 20 个 task vehicle）
- 服务：10，service IDs 为 0..9
- prediction horizon：5，horizon 为 0..4
- rho_hat：(300, 5, 10)
- vehicle position tensor：(300, 20, 5, 25, 2)；观测使用 vehicle IDs 的完整 25 维顺序
- 5e-4 / 5e-3 / 6e-4 / 6e-3 的 actor LR 分别为 0.0005 / 0.005 / 0.0006 / 0.006
- critic LR 与 Q-cache LR 均为 0.0003

Cache node 构造顺序来自 build_cache_nodes()：

1. SV：23, 87, 43, 75, 58
2. RSU：RSU_0, RSU_1
3. UAV：UAV_0, UAV_1

UAV action 顺序为 UAV_0, UAV_1。因此 Cache head index 与 node index 是固定的，不能按每个 slot 的请求数重排。

## 400 维 observation contract

ConfigurationObservationSpec.create() 按连续 slice 拼接，正式 V11-A 为 F=10, M=9, U=2, V=25, H=5，并包含 horizon mask 与 time-to-go：

| slice | 长度 | 内容 | 当前值域/来源 |
|---|---:|---|---|
| [0:90] | 9×10 | previous cache state | binary 0/1，按上述 9 个 node、service 0..9 展平 |
| [90:94] | 2×2 | previous UAV positions | 依 flight boundary 归一化的 x/y，按 UAV 顺序 |
| [94:344] | 5×25×2 | predicted vehicle positions | 每个 horizon 25 辆车的归一化 x/y，按 scenario vehicle ID 顺序 |
| [344:394] | 5×10 | predicted service popularity | 每个 horizon 的 rho_pred，每个 horizon 校验和为 1 |
| [394:399] | 5 | prediction horizon valid mask | h=0..4 的有效性；origin 接近 terminal 时未来项为无效并零填充 |
| [399:400] | 1 | normalized time-to-go | formal origin 0..299 映射到 [0,1] |

合计：

90 + 4 + 250 + 50 + 5 + 1 = 400

源码还执行以下对齐约束：

- 当前调度点必须是 scenario["horizons"][0] == 0
- horizon h 的 prediction target slot/time 必须是 slot+h / time+h*dt
- predicted vehicle IDs 必须与 scenario vehicle IDs 完整匹配
- predicted popularity 必须覆盖 0..9 且归一化
- observation 输出必须 finite，且 previous cache 只能是 binary

## observation 明确不包含的字段

以下字段不进入 400 维全局 observation；它们只在 environment/matching 或 candidate feature 路径出现：

- 当前 slot 的真实 task arrival、vehicle task list、service ID、input bits、CPU cycles
- 当前真实 vehicle positions 与真实未来轨迹
- task matching 结果、LOCAL/CLOUD/edge 目标、KKT CPU/BW allocation
- 当前 slot 的真实 delay、vehicle energy、task utility、Q(t)
- RSU/UAV 的容量、CPU、bandwidth、cache cost 参数的全局 raw state
- service size、真实 popularity、未来真实 labels
- prediction surrogate 的 objective/utility

这不是说这些字段完全不存在于一次 transition：AlignedRolloutStep 会把真实 task/position 交给 execute_slot() 计算 reward；但 build_configuration_observation() 只拼接 previous cache、previous UAV position、prediction horizon 与 time-to-go。candidate feature 另含当前 cache/node/capacity/role 等局部决策特征，不能把它们误称为全局 observation。

## action、mask 与 Q-cache contract

V11-A 的外层 environment action 是长度 11 的 binary trigger vector：

[Cache node 0..8 triggers, UAV_0 trigger, UAV_1 trigger]

实际 primitive 决策为：

- Cache 每个 head 的 operation order：NO_CHANGE, ADD, DELETE, SWAP
- Cache operation mask：始终允许 NO_CHANGE；其余 operation 是否允许由当前 binary cache、service size、node capacity 生成
- Cache candidate action：NO_CHANGE 使用 index 0 placeholder；其余候选来自 production add/delete/swap neighborhood
- UAV 每个 head 的 operation order：STAY, MOVE
- UAV operation mask：STAY 始终允许；MOVE 只有存在可行 movement candidate 时允许
- Cache 与 UAV 都按固定 head 顺序 sequentially apply；后续 head 会看到前一 head 的更新状态

正式 replica-aware tensor spec 为：

- Cache：9 heads，4 operations，max candidate padding 1+10+(10×10)//4 = 36，candidate feature dim 17+3×10+6 = 53
- UAV：2 heads，2 operations，max candidate padding 1+4×16+2×12 = 89，candidate feature dim 16+2 = 18
- Q-cache joint action encoding：Cache 每 head 4+36+53=93，UAV 每 head 2+89+18=109，总计 9×93+2×109=1055
- Q-cache 输入：400-d observation、9-d target-node one-hot、1055-d complete joint action encoding
- Q-cache 每次对一个 target Cache node 输出一个 Q；counterfactual 只替换 target-node action slice，其他 Cache/UAV action 保持固定

V11-A 类继承 V10-D 的 actor/global-critic graph，Cache branch 与 UAV branch 使用独立 actor encoder；Q-cache 参数由 trainer 独立拥有，不进入 actor 或 global critic parameter group。

## environment 与 reward IO

rollout 路径先按 primitive policy 生成 final cache/UAV state，随后调用 build_policy_primitive_configuration()。该路径计算 cache/movement cost，但显式跳过 prediction configuration solve。execute_slot() 接到 override 后：

1. 校验 requested trigger 与 actual transition；
2. 用 final cache/UAV state 建 service node positions；
3. 对真实当前 tasks/vehicle positions 执行 optimize_task_matching()；
4. edge target 通过 allocate_resources_kkt() 分配 CPU/BW；
5. 由 system_model 计算每个 task 的 delay、vehicle energy 与 task utility；
6. 汇总 Q(t)、cache cost 与 UAV movement cost；
7. 返回：

G(t) = Q(t) - C_ca(t) - C_q(t)

因此 V11-A formal reward 的实际路径是“primitive action → actual Matching/KKT → actual Q → cost subtraction”；prediction surrogate 不是 reward，也没有在 policy primitive override 路径参与当前 reward。

## 单位与数值语义

- vehicle/UAV position、distance：m
- slot time/delay：s
- input size：bit
- CPU load：cycle；CPU capacity：cycle/s
- bandwidth/rate：Hz / bit/s
- vehicle energy、movement/cache loading energy：J
- cache service size / loaded cache：MB
- task utility 与 Q(t)：由 omega_T/omega_E 归一化 delay/energy 得到的 dimensionless scalar
- G(t)：上述 dimensionless task utility 减去配置 cost 后的训练 scalar；cost 参数决定数值权重
- candidate feature 中的容量/距离/概率等有显式归一化，cost/energy 的部分使用 log1p；不能假设所有 candidate feature 都是同一物理单位

## 证据定位

- src/ppo_simple/interface.py:101-180,391-480,760-1024
- src/ppo_simple/trainer.py:1815-1857,2278-2333,2381-2563
- src/ppo_simple/primitive_control.py:59-83,667-954,1012-1315
- src/decision/cache_configuration.py:331-375,1395-1475
- src/decision/slot_environment.py:192-575,784-870
- src/decision/task_allocation.py:544-650,1556-1605,1672-1788
- src/common/system_model.py:32-169,224-405
- src/ppo_simple/rollout.py:194-617,892-921
- 四个 formal run 的 run_manifest.json、run_summary.json、episode_metrics.jsonl、update_metrics.jsonl

