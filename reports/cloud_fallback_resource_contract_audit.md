# Cloud fallback resource-feasibility audit

## 1. 范围与基线

- 当前 branch：`main`
- baseline SHA：`550d12f` (`Fix default formal training entrypoint`)
- 审计前 working tree：没有 tracked modification；已有两个未跟踪文件：
  `reports/system_contract_protocol_direct_cf_fix.md` 和用户文件 `打包并上传.bat`。
- 本次未修改、删除、移动或暂存 `打包并上传.bat`。
- 本次没有运行完整 240-slot rollout、PPO training 或 Batch4。

用户明确指定 `12.pdf` 为系统模型。本地仓库和当前附件目录没有可读取的 `12.pdf`，只有仓库中的 `11.pdf`；本报告不使用 `11.pdf` 代替 `12.pdf`。因此“12.pdf 直接定义”的部分严格采用本次请求中给出的 12.pdf 合同摘录作为审计输入，并明确保留该文件缺失边界。

参考机制为 Liu、Zhang、Zhang、Yuan 的 IEEE TWC 2024 论文，DOI `10.1109/TWC.2024.3430486`：[IEEE Xplore](https://ieeexplore.ieee.org/document/10609797)。论文机制只作为 associated/collaborative Edge 与 Cloud fallback 的参考，不把项目的 resource-infeasible fallback 扩展伪装成 12.pdf 原始公式。

## 2. 当前 Cloud gate 的真实行为

正式入口是 `task_allocation.build_feasible_targets()`，随后由 `optimize_task_matching()` 使用这些候选执行 Transfer-Exchange Matching 和 KKT：

1. 任务自身 TV cache 命中时，加入 `LOCAL:<task_id>`。
2. 对每个 cache bit 为 1 的 SV/RSU/UAV，额外检查当前物理路径：位置存在且为 finite、节点角色有效、CPU > 0、带宽 > 0、信道增益 > 0、uplink rate > 0；通过后加入该 Edge。
3. 只有当 `edge_targets` 为空时才加入 `CLOUD`；只要存在一个可执行的 cached Edge，Cloud 就不加入该任务的候选集合。
4. 初始 matching 选择候选列表首项；后续 transfer/exchange 只在 `feasible_targets` 内搜索，并跳过 resource-infeasible candidate。

因此当前行为准确地是：

| 情形 | 当前候选行为 |
|---|---|
| 无可执行 cached Edge、无 Local hit | `[CLOUD]` |
| 有可执行 cached Edge | `Local（若命中） + Edge`，不含 Cloud |
| 无 Edge、但有 Local hit | `[LOCAL:<task_id>, CLOUD]`，Local 与 Cloud 可以同时存在 |

最后一行是当前实现的明确语义：Cloud gate 由“没有 executable cached Edge”触发，而不是由高负载、较大 delay、较低 Q 或资源利用率触发。Cloud 在 Local hit 且无 Edge 时仍是候选竞争项；本次没有擅自改变这一已有 matching 语义。

## 3. 12.pdf 合同边界与 Cloud 公式

按本次提供的 12.pdf 合同摘录，直接定义的内容为：

- Local/SV/RSU/UAV/Cloud 是执行模式；Local/Edge 执行需要对应服务缓存。
- Cloud 存储全部服务，不受 cache constraint。
- SV/RSU/UAV 共享 edge CPU/BW fraction，并满足 `sum_v w_vm <= 1`、`sum_v b_vm <= 1`。
- Cloud 不参加 Edge CPU/BW KKT fraction 分配。

当前代码对应关系如下：

- `_group_edge_tasks()` 只把 SV/RSU/UAV 分组；`LOCAL` 和 `CLOUD` 不进入 Edge KKT maps。
- `cloud_execution()` 使用固定 per-task 资源，当前配置 `F_c=5e9 cycle/s`、`R_c=200e6 bit/s`、`resource_sharing=false`。
- 当前 Cloud 时延为 `T_cloud = psi_c * D/R_c + L/F_c`，`psi_c=11`；车辆发射能耗为 `E_cloud = p*D/R_c`，`psi_c` 不进入能耗。
- `cloud_task_count` 只是兼容性参数，不改变 Cloud 的固定 per-task allocation。
- `G = Q_actual - C_ca - C_q` 未修改。

参考论文支持“associated edge -> collaborative cached/sufficient resource -> Cloud fallback”的机制方向；但“项目在 KKT hard resource infeasible 后把任务转 Cloud”若未来实现，是本项目执行规则扩展，不是本次可从 12.pdf 逐字确认的公式。

## 4. resource_feasible=False 的精确定义

`evaluate_matching_with_kkt()` 的 `resource_feasible=False` 只来自 `allocate_resources_kkt()` 返回不可行。正式代码中包括：

1. 某个 Edge 节点分配的任务数 `N` 满足 `N*epsilon_w > 1`。
2. 某个 Edge 节点分配的任务数 `N` 满足 `N*epsilon_b > 1`。
3. KKT coefficient 出现 NaN/Inf/负值。
4. KKT bisection 的 high bound overflow，或最终 allocation sum 超过 1 的保护阈值。
5. Edge base uplink rate `<= 0`，或 bandwidth KKT 求解失败。

在正式 `build_feasible_targets()` 中，缺位置、invalid position、未知节点角色、非正 CPU、非正带宽、非正 channel gain/rate 的 cached Edge 会在进入 Matching 前被 fail-closed 排除；若没有其他 Edge，就进入 Cloud gate。这些是 candidate availability/path gate，不应被表述成“高利用率导致 resource infeasible”。

当前 `task_allocation.py` 与 `system_model.py` 没有 Tmax、Emax、deadline/QoS threshold 等硬约束。delay 或 energy 变大、Edge Q 低、Cloud Q 高、CPU/BW utilization 高，都不会单独使 `resource_feasible=False`。

## 5. epsilon 与 baseline 可达性

当前正式配置为：

- `epsilon_w = algorithm.kkt.min_cpu_fraction = 1.0e-6`
- `epsilon_b = algorithm.kkt.min_bandwidth_fraction = 1.0e-6`

因此单个 Edge 节点的最大整数任务数为：

```text
floor(1 / epsilon_w) = 1,000,000
floor(1 / epsilon_b) = 1,000,000
```

当前数据有 20 个 TV；`tasks.csv` 的 300 个 slot 中，单 slot 最大到达任务数为 19。即使 Matching 把某一个 slot 的全部 19 个任务都分给同一个 Edge 节点，也只有 `19e-6 <= 1`。在当前 task count/configuration 下，单纯由 epsilon CPU/BW minimum-fraction constraints 造成的 Edge resource infeasibility 属于：

> **C. impossible under current task count/configuration**

这不是“资源紧张”；当前项目也没有把 delay/energy 变大解释为 resource infeasible。

其他数值/物理异常在理论上仍可使 KKT evaluator 返回失败或抛出输入错误，但正式 path 已用 cached physical-path gate 排除无效 Edge；当前 baseline 的正 CPU、正带宽、正信道和 19-task 上限不会触发该 hard constraint。

## 6. 是否需要代码 repair

**结论：本轮不修改正式算法代码。**

理由：

- 当前代码已经满足“有可执行 cached Edge 时排除 Cloud；无可执行 cached Edge 时提供 Cloud”这一 cache/path gate。
- 当前 epsilon resource exhaustion 在 baseline 中不可达，不应为了制造 Case 3 修改 epsilon、任务数、KKT 或增加人工 penalty。
- 当前 `optimize_task_matching()` 对 candidate resource infeasible 会跳过并计数；最终仍要求正式 matching resource-feasible。由于 Cloud gate 保证 no-Edge case 有 Cloud 初始候选，baseline 不需要复杂 minimum-cloud-count repair。
- 本轮没有把 Cloud 永久加入普通 Edge candidate，也没有加入 QoS/utility/利用率触发的 Cloud 条件。

若未来改变 epsilon、任务规模或正式 hard constraints，使 Edge assignment 的 resource infeasibility 可达，再单独实现 `primary_targets = Local + Edge`、先做 primary Matching/KKT、失败后再进行 lexicographic `minimize N_cloud` / `maximize Q` repair；本轮不预先实现该未被 baseline 触发的组合搜索。

## 7. 三个最小验证 Case

fixture 使用 slot 20 的真实任务、真实车辆位置、真实 service-node/UAV 位置和当前配置；没有运行完整训练。

### Case 1：没有任何 Edge cache hit

- task `64`，service `7`；edge cache 全 0，Local cache 为空。
- `feasible_targets = [CLOUD]`。
- 正式 `optimize_task_matching()` 选中 `CLOUD`。
- `resource_feasible=True`。

结果：**PASS**。Cloud fallback available/selected，但触发原因是 no executable cached Edge，不是 resource exhaustion。

### Case 2：存在一个 cached 且 resource-feasible Edge

- 只把 node `23` 对 service `7` 的一个 cache bit 设为 1。
- 物理 path 检查通过。
- `feasible_targets = [23]`，Cloud 不在集合中。
- 正式 `optimize_task_matching()` 选中 `23`，`resource_feasible=True`。

结果：**PASS**。Cloud 不参与普通 candidate competition。

### Case 3：baseline resource-infeasible Edge assignment

正式 baseline 不可构造：需要单个 Edge 节点超过 1,000,000 个任务，而当前单 slot 最大只有 19 个。

只做了 synthetic contract test（不是 baseline experiment）：用 `epsilon=0.6`、`N=2` 的两任务 KKT 输入，返回 `resource_feasible=False`，原因为 `epsilon * N = 1.2 > 1`。该测试验证 hard minimum-fraction 规则本身，不改变正式配置，也不声称当前实验会触发它。

结果：**CONDITIONAL PASS**。KKT hard infeasibility 定义可验证；baseline fallback repair 不需要也未实现。

## 8. Cache -> Offloading 最小反事实

保持同一 slot、同一 task、同一车辆/节点位置、同一 UAV 位置和同一资源参数，只切换 node `23` 对 service `7` 的一个 cache bit：

| cache bit | primary/Cloud targets | final selected node | resource feasible | Q |
|---:|---|---|---|---:|
| 0 | `[CLOUD]` | `CLOUD` | True | `0.9782625546530878` |
| 1 | `[23]` | `23` | True | `0.8674627361158074` |

Edge allocation 为 CPU `1.0`、BW `1.0`（单任务）；Cloud 使用固定 per-task 资源。该反事实证明 cache state 确实改变了 Cloud fallback 与 Edge candidate structure，并不要求每次 cache 变化都改变所有场景下的最终节点。

另一个审计观察是：若 edge cache 仍为 0、但把 task `64` 的 Local cache bit 设为 1，则当前候选为 `[LOCAL:64, CLOUD]`。这确认了当前实现允许 Local 与 Cloud 共存；本轮不改变该已有语义。

## 9. 最终判定

- **PASS**：当前 cache/path Cloud gate、Cloud fixed-resource KKT 分离、Case 1/2、cache->offloading counterfactual。
- **CONDITIONAL PASS**：resource-infeasible -> Cloud 的结构规则在代码边界上可定义，但 baseline 不可达，且本轮没有实现未触发的 fallback repair。
- **BLOCKER（证据边界）**：工作区缺少用户指定的原始 `12.pdf`，所以本报告不能对 PDF 原文公式做独立逐页核验；没有用 `11.pdf` 替代。对当前请求中明确列出的 12.pdf 合同摘录，代码审计结果如上。

本轮正式 source code 未修改；只新增本报告。临时 fixture 在任务结束前删除，未生成 CSV/JSON/benchmark/profiling 文件。
