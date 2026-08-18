# adaptive_cache_mec 文件说明

本文档说明**当前这条 Python 自适应缓存 / UAV / MEC 求解主线**中各文件的用途，避免不同阶段产生的配置、论文缓存或历史脚本互相混淆。

## 1. 当前主线目标

当前系统不包含 TV 本地执行。任务执行集合为：

\[
\{\mathrm{SV},\mathrm{RSU},\mathrm{UAV},\mathrm{Cloud}\}.
\]

动态缓存节点为：

\[
\mathcal M=\mathcal V_S\cup\mathcal R\cup\mathcal U.
\]

当前滚动求解链：

```text
截至 t-1 的历史
    ↓
因果 ScenarioSet Xi_t
    ↓
P1 缓存优化
    ↓
P2 UAV 位置优化
    ↓
P1/P2 均 stall 时 Pjoint
    ↓
逐节点异步实际更新
    ↓
当前真实任务 / 真实位置
    ↓
P3 任务执行与 CPU/带宽资源分配
```

当前代码主线的核心参数源是：

```text
src/config.py
```

所有新加入当前主线的参数优先集中在该文件中维护。

## 2. 根目录文件

### `environment.yml`

当前项目 Conda / pip 依赖说明。用于恢复 Python 运行环境。

### `.gitignore`

Git 忽略规则。用于排除 Python cache、临时文件、模型文件、运行输出等。

### `THIRD_PARTY_NOTICES.md`

第三方代码、工具或数据来源说明。新增外部代码来源时应继续维护。

### `config.yaml`

仓库中另一阶段形成的配置资料。

**当前 P1/P2/Pjoint Python 主线并不读取该文件。**

它可以作为已有参数资料和历史实验配置的参考，但不能自动覆盖 `src/config.py`。如果未来明确决定迁移某个参数，应把该参数有依据地迁入 `src/config.py`，而不是让两套配置同时控制同一次实验。

本次 RSU 的两个固定坐标即参考了该文件中已经存在的部署信息，并迁入当前主线。

### `paper_spec.md`

论文/实现规范缓存和历史算法记录。

该文件可以用于核对系统实体、公式和历史设计，但其中若包含与当前已经冻结的 P1/P2/Pjoint 主线不同的求解器方案，不能自动替换当前代码。是否改变求解方法必须由研究主线单独决定。

### `jilv.md`

研究过程、代码修改和实验记录。不是运行入口，不直接控制求解器。

## 3. `src/`：核心模型与求解器

### `src/config.py`

**当前主线统一参数配置。**

包含时间、车辆规模、服务、任务到达、CPU、无线、Cloud、UAV、缓存、切换成本、预测场景、随机种子和 RSU 当前部署参数。

### `src/channel.py`

无线信道模型。

当前包含：
- TV → SV 地面链路；
- TV → RSU 地面链路；
- TV → UAV 空地链路；
- `full_rate_bps()` 统一接口。

RSU 当前仍属于集成 pilot：复用现有地面链路数值实现，最终参数实验前应单独校准其 V2I 合理性。

### `src/cache_model.py`

统一缓存物理模型。负责服务程序大小、SV/RSU/UAV 缓存容量、初始缓存、容量检查、缓存占用和初始状态保存。

### `src/rsu_state.py`

固定 RSU 状态读取。读取：

```text
data/rsu/rsu_deployment.csv
```

RSU 位置固定，不参加 P2；缓存由 P1 管理，任务执行由 P3 管理。

### `src/p3_solver.py`

给定配置下的任务执行与资源分配参考求解器。

候选执行节点：

```text
SV / RSU / UAV / Cloud
```

当前采用离散执行节点枚举；节点内部 CPU/带宽分配优先闭式解，deadline 激活时使用 SLSQP。P3 不负责缓存更新、UAV 更新和异步触发。

### `src/p1_cache.py`

P1 缓存优化核心。邻域包括 keep/add/delete/swap/proxy-demand。每个候选由完整预测场景 P3 评价，切换成本相对真实旧物理缓存计算。

### `src/scenario_generator.py`

因果公共随机预测场景生成器。只使用 `decision_slot` 以前的历史，不读取未来真实 popularity 或未来真实 SUMO 位置。

### `src/p1_runner.py`

P1 单时刻调试/集成运行器。主要用于独立验证 `H_{t-1} -> Xi_t -> P3 evaluator -> P1`；正式滚动实验以 `rolling_controller.py` 为准。

### `src/uav_state.py`

UAV 初始状态和预测热点辅助模块。负责初始 \(Q^U(0)\)、ScenarioSet 需求热点和 UAV 初始缓存辅助。

### `src/p2_uav.py`

P2 UAV 水平位置优化核心。候选包括 stay、demand hotspot、clustered hotspot、intermediate 和可选 extra candidate，并满足单时隙最大移动距离约束。

### `src/p2_runner.py`

P2 单时刻接口验证器。用于验证 `Q^U(0) -> Xi_1 -> P3 -> P2`；正式滚动实验仍以 `rolling_controller.py` 为准。

### `src/pjoint.py`

P1/P2 都没有严格改善时的联合配置搜索。当前联合候选为“一次缓存邻域动作 × 一次 UAV 有限位置动作”，所有候选调用完整 P3。

### `src/async_update.py`

异步自适应更新记录与一致性检查。记录 gross gain、net gain、各 SV/RSU/UAV 的真实更新时间和更新间隔，不再增加第二层切换成本门槛。

### `src/rolling_controller.py`

**当前主线总控制器。**

逐时隙执行：

```text
C(t-1), Q^U(t-1)
    ↓
Xi_t
    ↓
P1
    ↓
P2
    ↓
必要时 Pjoint
    ↓
最终 C(t), Q^U(t)
    ↓
实际 task(t)
    ↓
P3
```

RSU 接入后会统计 SV/RSU/UAV/Cloud 任务数，以及缓存更新、UAV 移动和异步更新信息。

### `src/p3_sv_cloud.py`

较早阶段的 SV + Cloud pilot。保留用于历史结果对照，不是当前完整主线运行入口。

## 4. `data/`：数据生成与预处理

### `data/generate_sumo_demand.py`
调用 SUMO 官方工具生成车辆出行需求。

### `data/export_sumo_trace.py`
从 SUMO 输出导出车辆轨迹。

### `data/prepare_mobility.py`
整理统一 mobility trace 和 episode 数据。

### `data/analyze_episode_windows.py`
分析可用连续 episode 窗口。

### `data/audit_mobility.py`
移动轨迹数据检查。

### `data/prepare_role_pool.py`
生成固定 TV/SV 候选角色池和稳定排序。

### `data/generate_workload.py`
生成 service catalog、dynamic popularity 和 task trace。

### `data/audit_task_feasibility.py`
任务 deadline / Cloud 可行性检查。

### `data/audit_v2v_channel.py`
当前 V2V pilot 信道数据检查。

### `data/prepare_rsu.py`
从 SUMO 路网提取 `traffic_light` junction。它生成的是 **RSU 候选部署点**，不是所有点都必须部署 RSU。

### `data/rsu/rsu_nodes.csv`
由 `prepare_rsu.py` 产生的 9 个候选位置，应保留作为地图候选资料。

### `data/rsu/rsu_deployment.csv`
当前主线真正使用的固定 RSU，当前为 2 个，由一次性接入脚本生成。

## 5. `results/`

保存 P3 pilot、单时刻 P1/P2、rolling summary、cache update events 和 UAV movement events。

当前阶段不要为了整理目录改变求解逻辑。正式实验开始后再按 baseline / ablation / sweep 建子目录。

## 6. 当前文件优先级

针对**当前这条 P1/P2/Pjoint 自适应 Python 求解主线**：

```text
用户当前确认的系统模型
    >
当前最新 src/ 代码
    >
src/config.py
    >
当前正式生成的数据
    >
历史/参考资料
```

其中：

```text
config.yaml
paper_spec.md
jilv.md
p3_sv_cloud.py
```

应明确看作参考、历史或其他阶段资料，除非后续明确决定把其中某项重新迁入当前主线。

## 7. RSU 接入后的下一步

先做最短滚动验证：

```powershell
python src\rolling_controller.py --end-slot 1
```

重点确认：
- RSU 被加载；
- P1 能处理 RSU cache；
- P3 能生成 RSU candidate；
- summary 中存在 `rsu_tasks`；
- rolling 链完整结束。

即使 `rsu_tasks=0` 也先不调参数，只确认结构接通。

之后再进入：较长短跑 → pilot 参数校准 → Cloud 校准 → 400-slot → baseline/ablation/sensitivity → 论文图表。
