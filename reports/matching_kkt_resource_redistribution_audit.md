# Cloud fallback 数学合同与 Multi-Edge Matching/KKT resource redistribution audit

## 1. 审计范围与基线

- branch：`main`
- baseline SHA：`550d12f` (`Fix default formal training entrypoint`)
- 本轮没有修改 PPO、reward、Cache optimizer、UAV optimizer、task generator、训练参数或正式配置。
- 没有运行 Batch4、PPO training 或完整 rollout；只运行了一个真实 slot 的 bounded Matching/KKT 审计。
- 本轮不创建、不复制、不修改任何 `12.pdf`，也不修改论文正文文件。以下四层严格分开：
  1. **[用户确认的系统模型]**：本次请求中明确的 Local/Edge/Cloud、cache eligibility、Edge resource sharing 与 Cloud 独立链路合同。
  2. **[参考论文机制]**：Liu et al., IEEE TWC 2024 的 associated/collaborative Edge first、Cloud fallback 机制；参考 DOI 为 `10.1109/TWC.2024.3430486`，见 [IEEE Xplore](https://ieeexplore.ieee.org/document/10609797)。
  3. **[当前代码实现]**：`task_allocation.build_feasible_targets()`、`optimize_task_matching()`、`allocate_resources_kkt()` 和 Cloud execution。
  4. **[建议后续论文表述]**：只记录在本报告中，不写回任何 PDF/论文文件。

## 2. Cloud fallback 数学合同对齐

### [用户确认的系统模型]

当前正式机制为：

- Local hit 可以作为候选，但本地执行要求任务车辆缓存对应服务。
- cached 且 executable 的 SV/RSU/UAV 构成 Edge domain。
- 只要存在任一 executable cached Edge，Cloud 不进入普通 Matching candidate domain。
- 没有 executable cached Edge 时，Cloud 作为 fallback。
- Local hit 且无 Edge 时，Local 与 Cloud 可以同时存在。
- Edge CPU/BW fraction 满足各 Edge 节点内 `sum_v w_vm <= 1`、`sum_v b_vm <= 1`。
- Cloud 存储全部服务，不受 cache constraint，也不参加 Edge CPU/BW KKT fraction allocation。

### [当前代码实现]

`build_feasible_targets()` 的正式路径逐任务执行：

1. Local cache 命中时加入 `LOCAL:<task_id>`。
2. cache bit 为 1 的 Edge 节点还要通过当前位置、节点角色、正 CPU、正带宽、正 channel gain 和正 uplink rate 检查，才加入 Edge domain。
3. `edge_targets` 非空时不加入 `CLOUD`。
4. `edge_targets` 为空时加入 `CLOUD`；如果此时有 Local hit，候选就是 `[LOCAL:<task_id>, CLOUD]`。
5. `optimize_task_matching()` 的 transfer/exchange 只在该候选集合中搜索。

因此当前代码与用户确认的项目合同一致。Cloud 当前由“无 executable cached Edge”触发，而不是由 Edge utilization 高、delay 大、Q 较低、Cloud Q 较高或其他软性能差异触发。

### P0 constraint 对齐状态

按用户确认的模型定义，P0 的执行选择、服务缓存 eligibility、Edge resource shares 和 Cloud 独立链路构成基础合同；但“存在 Edge 时 Cloud 被排除、无 Edge 时 Cloud fallback”需要作为明确的 edge-first selection gate 表达，不能只依赖 Cloud 无 cache constraint 的描述。当前本轮不修改 PDF；该表述边界不是实现 blocker。

### [建议后续论文表述]

对每个已到达任务 `v` 定义：

```text
E_v^exec(t) = { m in {SV, RSU, UAV}:
                c_m,f_v(t)=1,
                path_vm(t)=1,
                F_m>0,
                B_m>0,
                Rbar_vm(t)>0 }

P_v(t) = { LOCAL_v, if local cache hit }
         union E_v^exec(t)
```

推荐补充项目级 Cloud gate：

```text
x_v,LOCAL <= local_cache_v,f_v
x_v,m <= 1[m in E_v^exec(t)]
x_v,CLOUD <= 1[|E_v^exec(t)| = 0]
x_v,LOCAL + sum_m x_v,m + x_v,CLOUD = a_v(t)
```

这个 gate 保留“Local hit 且无 Edge 时 Local 与 Cloud 共存”，同时保证只要存在 executable cached Edge，Cloud 不进入普通 Matching。若未来要表达“primary Edge/Local 资源不可行后才启动 Cloud”，应另定义 primary feasibility predicate，再把该规则写成项目执行扩展；它是参考论文 Edge-first/Cloud-fallback 机制形成的项目设计扩展，不应伪装成用户确认模型的原始公式。

## 3. 真实 multi-task / multi-edge slot

选择 slot `0`：

- actual tasks：15 个
- executable Edge：`23, 43, 58, 75, 87, RSU_0, RSU_1, UAV_0, UAV_1`，共 9 个
- initial matching 中 `RSU_0` 承载 7 个任务
- final matching 中 `UAV_0` 承载 4 个任务、`UAV_1` 承载 2 个任务
- Cloud 任务不进入 Edge KKT allocation maps

### Initial feasible targets

为避免省略候选结构，slot 0 的完整任务候选如下：

| task | service | initial feasible targets |
|---|---:|---|
| 8 | 0 | `[CLOUD]` |
| 16 | 5 | `[23,87,43,75,58,RSU_0,RSU_1,UAV_0,UAV_1]` |
| 18 | 8 | `[CLOUD]` |
| 19 | 3 | `[CLOUD]` |
| 31 | 2 | `[RSU_0,RSU_1,UAV_0,UAV_1]` |
| 35 | 7 | `[RSU_0,RSU_1,UAV_0,UAV_1]` |
| 47 | 7 | `[RSU_0,RSU_1,UAV_0,UAV_1]` |
| 48 | 9 | `[RSU_0,RSU_1]` |
| 50 | 6 | `[CLOUD]` |
| 55 | 6 | `[CLOUD]` |
| 64 | 7 | `[RSU_0,RSU_1,UAV_0,UAV_1]` |
| 73 | 7 | `[RSU_0,RSU_1,UAV_0,UAV_1]` |
| 83 | 7 | `[RSU_0,RSU_1,UAV_0,UAV_1]` |
| 88 | 3 | `[CLOUD]` |
| 98 | 0 | `[CLOUD]` |

### Initial matching、KKT 与 Q

Initial matching 取每个候选列表的首项：

```text
8:CLOUD, 16:23, 18:CLOUD, 19:CLOUD, 31:RSU_0,
35:RSU_0, 47:RSU_0, 48:RSU_0, 50:CLOUD, 55:CLOUD,
64:RSU_0, 73:RSU_0, 83:RSU_0, 88:CLOUD, 98:CLOUD
```

Initial task utility sum：`Q=-53.35346171300375`。

Initial Edge allocations：

```text
node 23:
  CPU: 16=1.0
  BW : 16=1.0

node RSU_0:
  CPU: 31=0.3157926907635747, 35=0.09597572393946951,
       47=0.08615377360938202, 48=0.2094932966829849,
       64=0.09218197024445589, 73=0.09307983605537613,
       83=0.10732270870435713
  BW : 31=0.3764261967853323, 35=0.06509260406798527,
       47=0.047182612571518213, 48=0.21673860465283393,
       64=0.07294644201645399, 73=0.16943349605701272,
       83=0.05218004384965894
```

## 4. Transfer/exchange 与 affected-node redistribution

每行的 candidate counts 是该轮 enumerate 的 transfer/exchange candidates；`unaffected diff` 是候选前后未受影响 Edge allocation 的最大绝对差。每次 accepted candidate 另用 full evaluator 复算，full/incremental allocation 差异均为 0。

| update | type | affected tasks / old -> new | affected Edge nodes | candidates (T/E) | Q before -> after | delta Q | unaffected CPU/BW diff | result |
|---:|---|---|---|---:|---|---:|---:|---|
| 1 | transfer | 31: RSU_0 -> UAV_1 | RSU_0,UAV_1 | 27/0 | -53.3534617130 -> -16.6628080892 | 36.6906536238 | 0 / 0 | accept |
| 2 | transfer | 48: RSU_0 -> RSU_1 | RSU_0,RSU_1 | 27/5 | -16.6628080892 -> -3.2443347770 | 13.4184733122 | 0 / 0 | accept |
| 3 | transfer | 16: 23 -> UAV_0 | 23,UAV_0 | 27/10 | -3.2443347770 -> 4.0087263283 | 7.2530611054 | 0 / 0 | accept |
| 4 | transfer | 73: RSU_0 -> UAV_0 | RSU_0,UAV_0 | 27/16 | 4.0087263283 -> 10.4234360173 | 6.4147096890 | 0 / 0 | accept |
| 5 | transfer | 64: RSU_0 -> UAV_1 | RSU_0,UAV_1 | 27/18 | 10.4234360173 -> 12.0652552146 | 1.6418191973 | 0 / 0 | accept |
| 6 | transfer | 35: RSU_0 -> UAV_0 | RSU_0,UAV_0 | 27/19 | 12.0652552146 -> 12.9670534010 | 0.9017981864 | 0 / 0 | accept |
| 7 | transfer | 83: RSU_0 -> UAV_0 | RSU_0,UAV_0 | 27/18 | 12.9670534010 -> 13.2350329706 | 0.2679795696 | 0 / 0 | accept |
| 8 | exchange | 16,47: UAV_0/RSU_0 -> RSU_0/UAV_0 | RSU_0,UAV_0 | 27/15 | 13.2350329706 -> 13.4451894641 | 0.2101564935 | 0 / 0 | accept |
| 9 | exchange | 64,73: UAV_1/UAV_0 -> UAV_0/UAV_1 | UAV_0,UAV_1 | 27/15 | 13.4451894641 -> 13.4968865947 | 0.0516971306 | 0 / 0 | accept |
| 10 | final check | no strict positive candidate | current final Edge set | 27/15 | 13.4968865947 -> 13.4968865947 | 0 | 0 / 0 | reject |

Across the final incremental result:

- candidate evaluations: `443` (`297` transfer + `146` exchange)
- resource-infeasible candidates: `0`
- feasibility rejected before KKT: `677`
- all 9 accepted updates satisfy `delta_Q > objective_tolerance=1e-9`
- affected-node KKT was recomputed for the old/new Edge nodes of each changed task
- unaffected Edge CPU/BW allocations remained unchanged within exact comparison (`0.0` max diff)

## 5. Manual KKT closed-form check

选择 final `UAV_1`，其承载两个真实任务 `31` 和 `73`。使用当前代码相同的 `omega_T`、`omega_E`、`T_ref`、`E_ref`、`F_m` 和 observed-position `Rbar_vm`，按：

```text
c_cpu_v = omega_T * L_v / (T_ref * F_m)
w_v = sqrt(c_cpu_v) / sum_j sqrt(c_cpu_j)

c_bw_v = D_v / Rbar_vm * (omega_T/T_ref + omega_E*p_v/E_ref)
b_v = sqrt(c_bw_v) / sum_j sqrt(c_bw_j)
```

得到：

| task | c_cpu | manual w | code w | c_bw | manual b | code b |
|---|---:|---:|---:|---:|---:|---:|
| 31 | 0.12780082291956413 | 0.7723499869762783 | 0.7723499869766045 | 0.075546468413247 | 0.6722025097165191 | 0.6722025097159942 |
| 73 | 0.011103009774592429 | 0.22765001302372168 | 0.22765001302381777 | 0.017964914565283673 | 0.3277974902834810 | 0.3277974902832250 |

检查结果：

- CPU max absolute difference：`3.26183524634871e-13`
- BW max absolute difference：`5.24913446042774e-13`
- code CPU sum：`1.0000000000004223`
- code BW sum：`0.9999999999992192`
- minimum code CPU allocation：`0.22765001302381777 > epsilon_w=1e-6`
- minimum code BW allocation：`0.327797490283225 > epsilon_b=1e-6`

CPU 的 `+4.22e-13` 是 bisection floating-point roundoff，处于当前 solver 的 `1e-10` feasibility tolerance 内；没有出现实际资源超分配。按该数值容差，CPU/BW 两项均通过。

## 6. Final stable matching 与 full/incremental equivalence

Final matching：

```text
8:CLOUD, 16:RSU_0, 18:CLOUD, 19:CLOUD, 31:UAV_1,
35:UAV_0, 47:UAV_0, 48:RSU_1, 50:CLOUD, 55:CLOUD,
64:UAV_0, 73:UAV_1, 83:UAV_0, 88:CLOUD, 98:CLOUD
```

Final Edge task counts：`RSU_0=1`、`RSU_1=1`、`UAV_0=4`、`UAV_1=2`。

Final Edge allocations：

```text
RSU_0: 16 -> CPU 1.0, BW 1.0
RSU_1: 48 -> CPU 1.0, BW 1.0
UAV_0:
  35 -> CPU 0.25148618716538396, BW 0.2590697959424574
  47 -> CPU 0.22574962861022954, BW 0.21979598685488888
  64 -> CPU 0.2415453749199323,  BW 0.2932290664660178
  83 -> CPU 0.2812188093037178,  BW 0.22790515073634324
UAV_1:
  31 -> CPU 0.7723499869766045,  BW 0.6722025097159942
  73 -> CPU 0.22765001302381777, BW 0.327797490283225
```

同一真实 slot、同一 observed positions、cache state、UAV positions 和资源参数下，full evaluator 与 incremental evaluator 的结果为：

| 项目 | max absolute difference / result |
|---|---:|
| matching | equal |
| total Q | `0.0` |
| per-task delay | `0.0` |
| per-task energy | `0.0` |
| per-task Q | `0.0` |
| CPU allocations | `0.0` |
| BW allocations | `0.0` |
| resource_feasible | equal (`True`) |

Final transfer-exchange stability：

- `transfer_exchange_stable=True`
- `stable_check_passed=True`
- `max_remaining_delta=-0.05169713061301273`
- final check 没有正的 strict-improvement candidate

## 7. 分项判断

| 项目 | 判断 | 依据 |
|---|---|---|
| Cloud fallback mathematical contract | **PASS（项目代码合同） / CONDITIONAL（论文表述）** | 当前代码实现与用户确认的 Edge-first/Cloud-fallback 语义一致；建议 gate 数学式只写入后续论文表述，不写回 PDF。 |
| CPU KKT | **PASS（solver tolerance 内）** | 手算/code max diff `3.26e-13`，sum 仅高出 1 的 `4.22e-13`。 |
| Bandwidth KKT | **PASS** | 手算/code max diff `5.25e-13`，sum `0.9999999999992192`。 |
| affected-node redistribution | **PASS** | accepted transfer/exchange 的 old/new Edge 均重算；受影响外的 allocation max diff `0.0`，full candidate diff `0.0`。 |
| Matching improvement | **PASS** | 9 次 accepted update 全部严格增加 Q，最终 `Q=13.49688659470791`。 |
| full/incremental equivalence | **PASS** | matching、Q、delay、energy、CPU、BW 全部一致。 |
| transfer-exchange stability | **PASS** | stable flag、final recheck 和 negative max remaining delta 均通过。 |

本轮正式 source code 未修改。临时审计脚本和临时 PDF 渲染目录已清理；没有生成 CSV/JSON/benchmark 文件，也没有 commit 或 push。
