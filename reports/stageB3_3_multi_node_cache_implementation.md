# Stage B3-3：9节点独立Cache UPDATE与联合PPO实现

## 结果摘要

已实现可切换的 `MULTI_NODE_FACTORIZED` 策略模式，并新增独立配置。单节点策略默认行为和旧版版本合同保留。候选对齐、KEEP/更新成本、节点概率mask以及联合log-prob回放检查通过。

本次**未完成正式Matching/KKT运行验收和Checkpoint端到端保存/重载验收**：纯Python KKT fallback 的一个Slot执行长时间没有完成，已停止；Checkpoint往返检查也未能在受控尝试中返回结果。因此整体为 `PARTIAL`，当前不建议进入B3-4短程训练。

## 修改文件

- `src/common/project_config.py`：增加动作模式、抽象重配置模型及策略Schema验证；缺省动作模式为 `SINGLE_NODE_UNIFIED`。
- `src/ppo_simple/interface.py`：加载模式及模型合同，并要求多节点上下文使用稳定的5个SV、2个RSU、2个UAV。
- `src/ppo_simple/policy.py`：增加共享参数的节点级masked categorical动作、分节点基数校正概率和 `MultiNodeFactorizedActorCritic`。
- `src/ppo_simple/rollout.py`：按相同更新前Cache生成9组候选，保存 `[9,K,200]` 特征、mask、动作和各节点旧log-prob；合并目标行后执行一次Slot接口，并记录节点级诊断。
- `src/ppo_simple/trainer.py`：支持向量Cache动作及旧log-prob重算，联合动作使用一个Macro PPO ratio/clip/KL路径；新增节点级指标和架构合同。
- `src/ppo_simple/evaluator.py`：按Schema严格选择单节点或多节点架构，并拒绝不匹配的版本合同。
- `src/ppo_simple/reporting.py`：记录动作模式、节点顺序、联合概率和熵聚合口径。
- `src/decision/cache_configuration.py`、`src/decision/slot_environment.py`：按节点汇总正式Cache更新成本，并将逐节点成本写入执行结果。
- 新增 `configs/proposed/system_config_multi_node_factorized.toml`：隔离启用多节点模式的实验配置。

本次没有编辑原正式配置 `configs/proposed/system_config.toml`。该文件在本任务开始前已存在其他工作区修改，未覆盖或清理。

## 动作与执行合同

- 稳定节点顺序为 `23, 87, 43, 75, 58, RSU_0, RSU_1, UAV_0, UAV_1`，即5 SV、2 RSU、2 UAV。
- 每个节点动作是其自己的 `KEEP + 最多64个合法UPDATE proposal`。候选生成和200维经济特征沿用现有规则；所有节点候选来自同一更新前Cache状态。
- 输入形状为 `candidate_features [B,9,Kmax,200]`、`legal_masks [B,9,Kmax]`；动作是 `[B,9]`，旧节点log-prob为 `[B,9]`。padding及非法位置由mask排除。
- 同一个全局更新前Observation送入共享Cache Actor；9个节点各自做masked categorical及现有KEEP/UPDATE基数校正。确定性选取仍按每个节点既有分组语义处理。
- 选定动作只替换对应Cache行；KEEP行保持不变。实际trigger按目标行和当前行差异生成。
- 一次Macro边界合并完整9节点Cache矩阵，在当Slot只调用一次 `execute_configured_slot`。成本按正式逐节点公式求和，仅改变的节点计费；Task Utility和Reward公式未改。

## 联合PPO与版本隔离

`logp_cache_joint = Σ_i logp_cache_i`；与Holding termination序列合并为 `logp_macro_joint`。训练继续使用现有Macro SMDP-GAE、joint ratio `exp(new-old)`、一次joint clipping和现有KL保护；没有使用逐节点独立clip或监督损失。Macro Critic、gamma=1.0、GAE lambda=0.95、Holding 1–5槽及Micro UAV PPO保持原定义。

节点熵按9个独立分类分布求和作为联合Cache熵，并另外报告节点熵；报告中标明该和不能直接与旧单节点熵比较。训练诊断新增每节点KEEP/UPDATE概率、更新频率、动作数、候选熵/log-prob、成本、联合Ratio/Clip Fraction/KL、Macro Actor/Critic loss与Holding duration。

旧版保持Architecture 32 / Checkpoint 59 / 单节点Schema；新模式使用Architecture 33 / Checkpoint 60 / `...v4_multi_node_cache_factorized_v1`。新模式必须从头初始化，loader按Schema、架构、Checkpoint版本、动作模式及配置合同严格校验。没有进行旧权重迁移。

## 抽象重配置边界

新配置明确启用 `ABSTRACT_MACRO_BOUNDARY_RECONFIGURATION`：多个节点可以在Macro边界同时改变逻辑缓存，并在该Slot任务匹配前生效；缓存成本逐节点累加。模型不包含镜像下载时延、回传带宽竞争或下载队列，不声称证明了真实物理网络上的并发下载可行性。未发现需要擅自删除的既有强制下载约束。

## 最小验证结果

- 9个修改源码文件通过Python AST解析；原正式配置和新实验配置均通过TOML解析及正式配置合同校验。新配置为 `MULTI_NODE_FACTORIZED`，原配置缺省仍为单节点模式。
- Episode 20001 Slot 0的节点候选特征逐项与正式单节点候选生成语义对齐。
- 全部KEEP时完整Cache不变、Cache成本为0；单节点更新保持原成本语义；两节点同时更新得到合法目标行，合计成本与节点成本求和一致。该次两节点总成本为 `0.209184404307`。
- masked categorical的合法概率和为1，非法/padding概率为0。相同权重、动作及mask重算后，分节点log-prob一致，联合Ratio为1。
- **未完成项：** 使用现有Python环境及纯PythonKKT fallback尝试Episode 20001 Slot 0的正式Matching/KKT，执行耗时超出合理范围后中止；没有取得联合Cache下有限Reward或资源可行性的运行证据。Checkpoint保存/重载及旧Checkpoint拒绝的端到端运行检查未能完成，不能报告为PASS。

环境方面，`conda activate adaptive_cache_mec` 在本机失败。使用已有 `C:\software\anaconda3\python.exe`，该环境的Numba与NumPy版本不兼容；必要的项目调用通过进程内切换到项目纯Python fallback完成，没有安装或修改依赖。

## 状态

```ini
STAGE_B3_3 = PARTIAL
MULTI_NODE_MODE_IMPLEMENTED = YES
NODES_WITH_INDEPENDENT_ACTIONS = 9
SIMULTANEOUS_UPDATE_SUPPORTED_IN_ABSTRACT_MODEL = YES
PHYSICAL_DOWNLOAD_FEASIBILITY_PROVEN = NO
JOINT_CACHE_COST_CORRECT = YES
MATCHING_KKT_COMPATIBLE = NO (本次运行验收未完成；不代表已证明接口不兼容)
JOINT_PPO_LOGPROB_CORRECT = YES
CHECKPOINT_CONTRACT_UPDATED = YES (端到端保存/重载尚未验证)
LEGACY_SINGLE_NODE_MODE_PRESERVED = YES
READY_FOR_B3_4 = NO
PPO_TRAINING_EXECUTED = NO
```

解除B3-4阻断前，先在可完成正式KKT的既有运行环境中做一次受控联合Cache Slot执行，并完成一次新版本Checkpoint往返及旧版本拒绝验证；无需扩大到Episode或训练Epoch。

## Stage B3-3A：Runtime and Checkpoint Gate

### 运行环境与定位

本机再次尝试激活 `adaptive_cache_mec` Conda 环境，仍返回“The system cannot execute the specified program.”，因此使用此前验证过的 `C:\software\anaconda3\python.exe`，在当前进程内关闭不兼容的 Numba 导入以走项目自带纯Python KKT fallback。没有安装或变更依赖。未发现可直接使用的 Singularity/Apptainer/Docker 容器；本地执行已通过，未尝试部署到服务器。

Checkpoint 首次往返尝试停在 PyTorch 打开 `%TEMP%` 序列化文件时，报 Windows access denied；策略初始化和 `checkpoint_payload` 状态构造分别约数秒和0.003秒，完整训练状态仅约1.96 MB。改用项目内独立临时目录后，正式保存与读取通过。该问题是临时目录访问权限，不是模型状态规模或版本合同问题；正式源码无需为此修改。

此前B3-3记录的KKT长时间等待本次没有复现：同一纯Python fallback、相同Training Episode/Slot的三个分支均在约0.05–0.06秒内完成正式Slot执行。没有发现重复调用整槽Matching的实现问题。

### 固定Slot输入与合法动作

三个分支都从 Episode 20001 / Slot 0 的同一初始Cache、UAV位置、真实任务、车辆位置、预测场景和配置复制状态。Slot共有12个任务；UAV均保持原位。每分支只调用一次 `execute_configured_slot`，每次调用内部由正式Transfer-Exchange Matching触发多次KKT评价。

当前运行时 `action_spec.cache_node_ids` 顺序为 `46, 10, 0, 23, 59, RSU_0, RSU_1, UAV_0, UAV_1`，角色为5 SV、2 RSU、2 UAV。B3-3正文列出的车辆节点ID来自先前上下文；本次实际执行按当前 Episode 的 `action_spec` 顺序构造候选和Cache矩阵，没有硬编码那些车辆ID。每节点候选数分别为 `11, 11, 11, 11, 11, 57, 57, 35, 35`，均在KEEP加最多64个UPDATE的合同内。

报告中使用B3-3未记录具体目标行的合法候选时，本次按用户允许的规则从当前正式候选集合选取前序合法候选：SV `46` 的Cache `{4,7} → {1,4}`；SV `10` 的Cache `{8} → {4,7}`。每个节点目标均由当次正式候选生成器产生。联合矩阵构造时只替换所选行，未选行保持原值。

### 单Slot执行结果

| 分支 | 更新节点 | Cache Cost | Task Utility Q | Reward G | Matching候选评价 | KKT调用/求解/有限结果 | Matching耗时 | KKT累计耗时 | Slot调用总耗时 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A：全KEEP | 无 | 0 | 11.537621237532 | 11.537621237532 | 211（完整评价211） | 205 / 205 / 205 | 0.063212 s | 0.044972 s | 0.064261 s |
| B：单节点UPDATE | SV `46` | 0.075958291525 | 11.537621237532 | 11.461662946007 | 203（完整评价203） | 184 / 184 / 184 | 0.054021 s | 0.037827 s | 0.054852 s |
| C：双节点UPDATE | SV `46`, `10` | 0.209184404307 | 11.537621237532 | 11.328436833226 | 169（完整评价169） | 187 / 187 / 187 | 0.055353 s | 0.040350 s | 0.056183 s |

三条分支均由正式Slot执行返回可行结果；每个KKT求解均为有限值。没有修改Matching/KKT迭代上限或资源约束。三个分支Q相同、移动成本为0，Reward逐项等于 `Q - CacheCost`；因此这组单槽中新增Cache配置没有改变最终Task Utility，差异正好来自实际Cache成本。KKT调用数是单次Matching内部评价次数，不是重复整槽调用次数。

### 分阶段计时

| 阶段 | 实测耗时 | 口径 |
|---|---:|---|
| 环境状态准备 | 0.307 s | 加载正式配置、Episode 20001上下文并对齐Slot 0真实输入 |
| 9节点候选生成 | 0.155 s | 正式候选接口，一次生成各节点合法proposal与200维特征 |
| Cache Actor前向 | 0.007704 s | 新初始化未训练Actor，仅做一次确定性策略前向；动作不替代固定反事实分支 |
| 联合Cache矩阵构建 | 0.00003–0.00006 s | 根据指定节点候选替换Cache行并核对未选行 |
| Slot配置构建/应用 | 0.00027–0.00029 s | 正式执行接口的配置阶段计时 |
| 正式Matching搜索 | 0.054–0.063 s | Matching连同其内部资源评价的总耗时 |
| KKT累计求解 | 0.038–0.045 s | 通过现有timing profile测量；求解次数见上表 |
| Reward及结果返回 | 0.00056–0.00076 s | Slot总耗时扣除配置及Matching/KKT耗时后的剩余边界时间 |

### Checkpoint往返与隔离

使用多节点独立配置创建未训练策略和正式优化器对象；没有执行优化器更新。临时Checkpoint大小为2,263,280字节。

| 验收项 | 结果 |
|---|---|
| Architecture / Checkpoint / Schema | 33 / 60 / `...v4_multi_node_cache_factorized_v1` |
| 新Checkpoint正式保存 | PASS，0.013 s |
| 新Checkpoint正式重载 | PASS，0.010 s |
| 策略参数逐项相等 | PASS |
| 9节点动作合同与动作模式恢复 | PASS，`[B,N]`、`unified_cache_action=False`及合同均匹配 |
| Stage6 Architecture 32 / Checkpoint 59拒绝 | PASS，正式Loader以架构/动作语义不匹配拒绝 |
| 历史Checkpoint、runs或Training数据改动 | 无 |

临时Checkpoint位于项目内独立临时目录，测试后已删除；此前失败的系统临时目录也已确认不存在。没有留下测试脚本或临时结果文件。

### B3-3A状态

```ini
STAGE_B3_3A = COMPLETE
SINGLE_NODE_KEEP_SLOT = PASS
SINGLE_NODE_UPDATE_SLOT = PASS
MULTI_NODE_UPDATE_SLOT = PASS
MATCHING_KKT_RUNTIME_CAUSE = B3-3原先的长等待本次未复现；本次纯Pythonfallback在3个分支均快速完成，旧等待的具体原因无法从现有证据确定
MATCHING_KKT_FEASIBLE = YES
JOINT_CACHE_REWARD_CORRECT = YES
NEW_CHECKPOINT_SAVE = PASS
NEW_CHECKPOINT_RELOAD = PASS
OLD_CHECKPOINT_REJECTED = YES
READY_FOR_B3_4 = YES
PPO_TRAINING_EXECUTED = NO
```

本结论仅说明9节点抽象同步重配置的单槽执行、Reward和Checkpoint合同已具备小规模B3-4验证条件；不代表镜像并发下载的物理可行性已经建模或验证。
