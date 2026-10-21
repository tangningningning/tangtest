# Training Slot Cloud 卸载原因与 Matching 最优性对照

Generated: 2026-10-19T14:27:27.051342+00:00

## 数据、配置与复现状态

- 复用 `reports\matching_kkt_training_efficiency.json` 中预选的 9 个 Training Slot；未重新扫描整个 Pool，未读取 Validation/Test。
- 当前配置 SHA-256：`0927080d4d33c640a7cb0005662e686c75ff41805e2439cff5d2046c2b4e4795`；效率诊断记录：`0927080d4d33c640a7cb0005662e686c75ff41805e2439cff5d2046c2b4e4795`；相同：`True`。三个 10 任务 Slot 均在当前配置下重新运行 Matching。
- Matching/KKT 源码 SHA-256：`1df9ac8e6cfa777bf0b9a572dd6344fe8a2ecd8fe5b28b4275eb8c03d696589c`；Pool Manifest SHA-256：`b6cae236b029f4d24c7488a91f0825247a20f475ec76b2d80737de44ae491bd3`。九个 Episode Manifest 与数据 artifact 哈希复核通过。
- Python：`C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`；Conda：`adaptive_cache_mec`；分支/HEAD：`main` / `faa73c932bcaa17f6f29d1f19481380b7f0cc274`。
- Cloud fallback：`RESOURCE_AWARE_FALLBACK`；状态参考：`FROZEN_INITIAL_CACHE_UAV_REFERENCE`。每个 Slot 使用相同的正式初始 Cache/UAV 参考、真实任务和实际车辆位置。

- 当前效用权重为 delay/energy `0.5`/`0.5`；Cache/UAV 系数为 `0.005`/`5e-06`。Matching 对照保持 Cache/UAV 固定，因此仅比较任务效用 Q；成本相同，不会随卸载分配重复扣计。

## A. 九个 Slot 的 Cloud 原因分类

`FORCED_CLOUD`：没有合法 Edge 候选且最终选择 Cloud。`VOLUNTARY_CLOUD`：存在合法 Edge 候选但最终选择 Cloud。`EDGE_SELECTED`：选择合法 Edge。Forced/Voluntary ratio 以该 Slot 到达任务数为分母；另列 Forced Cloud 占 Cloud 的比例。

| 任务数 | Episode / Slot | Edge | Forced Cloud | Voluntary Cloud | Cloud 总数 | Forced / Tasks | Voluntary / Tasks | Forced / Cloud | 服务未缓存 | 有缓存但路径失败 | 可行 Edge 边 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 20001 / 21 | 1 | 5 | 4 | 9 | 50.0% | 40.0% | 55.6% | 5 | 0 | 8 |
| 10 | 20002 / 22 | 4 | 5 | 1 | 6 | 50.0% | 10.0% | 83.3% | 5 | 0 | 18 |
| 10 | 20003 / 21 | 4 | 4 | 2 | 6 | 40.0% | 20.0% | 66.7% | 4 | 0 | 14 |
| 14 | 20001 / 1 | 8 | 2 | 4 | 6 | 14.3% | 28.6% | 33.3% | 2 | 0 | 28 |
| 14 | 20002 / 10 | 5 | 5 | 4 | 9 | 35.7% | 28.6% | 55.6% | 5 | 0 | 18 |
| 14 | 20003 / 2 | 6 | 4 | 4 | 8 | 28.6% | 28.6% | 50.0% | 4 | 0 | 22 |
| 20 | 20003 / 119 | 11 | 2 | 7 | 9 | 10.0% | 35.0% | 22.2% | 2 | 0 | 51 |
| 20 | 20011 / 48 | 10 | 4 | 6 | 10 | 20.0% | 30.0% | 40.0% | 4 | 0 | 34 |
| 20 | 20012 / 127 | 7 | 7 | 6 | 13 | 35.0% | 30.0% | 53.8% | 7 | 0 | 33 |

- 汇总：Edge `56`，Forced Cloud `38`，Voluntary Cloud `38`，Cloud 总数 `76`。九个 Slot 中 Forced Cloud 为 38、Voluntary Cloud 为 38，Forced 与 Voluntary 数量相同；两类各占 Cloud 任务 50.0% 和 50.0%。Voluntary Cloud 出现在 9/9 个 Slot。Forced Cloud 中 38 个请求在任何 Edge 节点均未缓存，0 个请求虽有缓存但未通过当前 Edge 路径检查。对 C 类任务，单任务 Cloud/Edge 对照为 28/10 个；固定其他分配的转移中有 8 个出现目标收益被其他任务损失抵消。
- 有合法 Edge 候选时仍选择 Cloud：132 个到达任务中有 38 个存在合法 Edge 候选但最终分配到 Cloud；该现象在 9/9 个 Slot 出现。
- `build_feasible_targets` 对缓存服务还会核验节点 CPU、带宽、信道增益及上传速率是否为有限正值。因此“服务未缓存”和“存在缓存但路径检查未通过”可区分；后一类不再臆测具体是距离、信道或单独资源原因。
- 同任务数 Slot 的 Cloud 比例差异对应请求服务、缓存覆盖、物理可行 Edge 边集合和 Matching 的整体任务效用选择差异。Cloud 比例本身不能说明 Matching 有误。

### 每任务分类

#### 20001 / Slot 21

| Task ID | Service | 合法 Edge 节点 | Edge 候选数 | Matching 节点 | 类别 | 代码 | 零 Edge 原因 | Task utility | Delay (s) | Vehicle energy (J) |
|---|---:|---|---:|---|---|---|---|---:|---:|---:|
| 1 | 4 | 46, 23 | 2 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.91967752 | 4.573155 | 0.05454637 |
| 15 | 0 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.93605144 | 3.638132 | 0.04389654 |
| 16 | 9 | 0, RSU_1 | 2 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.90009345 | 5.683327 | 0.06866580 |
| 22 | 4 | 46, 23 | 2 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.92426623 | 4.312587 | 0.05131335 |
| 34 | 6 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.92356925 | 4.334730 | 0.05475510 |
| 36 | 6 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.91208797 | 4.980754 | 0.06384864 |
| 42 | 3 | RSU_0 | 1 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.88512883 | 6.495221 | 0.08561739 |
| 45 | 2 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.94180317 | 3.317384 | 0.03885222 |
| 50 | 2 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.92610449 | 4.200521 | 0.05131829 |
| 51 | 1 | UAV_1 | 1 | UAV_1 | EDGE_SELECTED | D | — | 0.98002130 | 0.964695 | 0.04281104 |

#### 20002 / Slot 22

| Task ID | Service | 合法 Edge 节点 | Edge 候选数 | Matching 节点 | 类别 | 代码 | 零 Edge 原因 | Task utility | Delay (s) | Vehicle energy (J) |
|---|---:|---|---:|---|---|---|---|---:|---:|---:|
| 0 | 5 | 21, RSU_0, RSU_1, UAV_0, UAV_1 | 5 | UAV_0 | EDGE_SELECTED | D | — | 0.95940325 | 1.974505 | 0.08458150 |
| 10 | 6 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.92866520 | 4.024321 | 0.05472525 |
| 12 | 5 | 21, RSU_0, RSU_1, UAV_0, UAV_1 | 5 | RSU_1 | EDGE_SELECTED | D | — | 0.95892541 | 1.549286 | 0.16147577 |
| 27 | 6 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.93488326 | 3.675923 | 0.04955050 |
| 35 | 2 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.93093234 | 3.908049 | 0.05101812 |
| 36 | 5 | 21, RSU_0, RSU_1, UAV_0, UAV_1 | 5 | UAV_1 | EDGE_SELECTED | D | — | 0.94951104 | 2.645608 | 0.07303946 |
| 39 | 1 | UAV_1 | 1 | UAV_1 | EDGE_SELECTED | D | — | 0.95402640 | 2.393742 | 0.06909033 |
| 41 | 9 | 11, RSU_1 | 2 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.92787222 | 4.085444 | 0.05256016 |
| 48 | 2 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.92922618 | 4.003688 | 0.05243087 |
| 50 | 2 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.92744716 | 4.104377 | 0.05374046 |

#### 20003 / Slot 21

| Task ID | Service | 合法 Edge 节点 | Edge 候选数 | Matching 节点 | 类别 | 代码 | 零 Edge 原因 | Task utility | Delay (s) | Vehicle energy (J) |
|---|---:|---|---:|---|---|---|---|---:|---:|---:|
| 13 | 3 | RSU_0 | 1 | RSU_0 | EDGE_SELECTED | D | — | 0.92136083 | 2.557517 | 0.37831558 |
| 16 | 9 | 42, RSU_1 | 2 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.92677483 | 4.147879 | 0.05331301 |
| 28 | 2 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.94146408 | 3.315363 | 0.04269199 |
| 37 | 6 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.92507618 | 4.228423 | 0.05720305 |
| 44 | 4 | 52, 20 | 2 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.94141864 | 3.321045 | 0.04219906 |
| 51 | 8 | 57, 23, UAV_0 | 3 | UAV_0 | EDGE_SELECTED | D | — | 0.98875283 | 0.565151 | 0.02036593 |
| 53 | 0 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.93661844 | 3.588283 | 0.04648448 |
| 54 | 5 | 20, RSU_0, RSU_1, UAV_0, UAV_1 | 5 | UAV_1 | EDGE_SELECTED | D | — | 0.94064980 | 2.508006 | 0.18773079 |
| 56 | 3 | RSU_0 | 1 | RSU_0 | EDGE_SELECTED | D | — | 0.96756848 | 1.299759 | 0.11455379 |
| 7 | 2 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.94484411 | 3.126447 | 0.03979977 |

#### 20001 / Slot 1

| Task ID | Service | 合法 Edge 节点 | Edge 候选数 | Matching 节点 | 类别 | 代码 | 零 Edge 原因 | Task utility | Delay (s) | Vehicle energy (J) |
|---|---:|---|---:|---|---|---|---|---:|---:|---:|
| 1 | 8 | 10, 59, UAV_0 | 3 | 10 | EDGE_SELECTED | D | — | 0.95300921 | 2.812371 | 0.00873253 |
| 15 | 7 | 46, RSU_1, UAV_1 | 3 | 46 | EDGE_SELECTED | D | — | 0.95372197 | 2.723989 | 0.01633866 |
| 22 | 7 | 46, RSU_1, UAV_1 | 3 | UAV_1 | EDGE_SELECTED | D | — | 0.96321938 | 1.751410 | 0.08297558 |
| 25 | 1 | UAV_1 | 1 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.91185140 | 4.976902 | 0.06694086 |
| 30 | 8 | 10, 59, UAV_0 | 3 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.92333729 | 4.327074 | 0.05844343 |
| 31 | 8 | 10, 59, UAV_0 | 3 | UAV_0 | EDGE_SELECTED | D | — | 0.96080705 | 2.153282 | 0.03984469 |
| 33 | 0 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.94994931 | 2.836156 | 0.03626989 |
| 36 | 3 | RSU_0 | 1 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.89511795 | 5.908521 | 0.08187525 |
| 37 | 8 | 10, 59, UAV_0 | 3 | UAV_0 | EDGE_SELECTED | D | — | 0.95743473 | 2.255518 | 0.05732724 |
| 44 | 8 | 10, 59, UAV_0 | 3 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.94218739 | 3.269280 | 0.04303024 |
| 45 | 3 | RSU_0 | 1 | RSU_0 | EDGE_SELECTED | D | — | 0.97018367 | 1.090576 | 0.12298102 |
| 48 | 1 | UAV_1 | 1 | UAV_1 | EDGE_SELECTED | D | — | 0.96488431 | 1.787592 | 0.05967838 |
| 50 | 8 | 10, 59, UAV_0 | 3 | UAV_0 | EDGE_SELECTED | D | — | 0.96204442 | 2.119760 | 0.03275460 |
| 7 | 0 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.93435159 | 3.711582 | 0.04899948 |

#### 20002 / Slot 10

| Task ID | Service | 合法 Edge 节点 | Edge 候选数 | Matching 节点 | 类别 | 代码 | 零 Edge 原因 | Task utility | Delay (s) | Vehicle energy (J) |
|---|---:|---|---:|---|---|---|---|---:|---:|---:|
| 0 | 7 | 55, RSU_1, UAV_1 | 3 | UAV_1 | EDGE_SELECTED | D | — | 0.94636576 | 2.654529 | 0.10397292 |
| 10 | 9 | 11, RSU_1 | 2 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.92604133 | 4.215562 | 0.04942412 |
| 12 | 0 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.94159822 | 3.327565 | 0.03924324 |
| 15 | 3 | RSU_0 | 1 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.88964566 | 6.242868 | 0.08173563 |
| 27 | 1 | UAV_1 | 1 | UAV_1 | EDGE_SELECTED | D | — | 0.92800150 | 3.550877 | 0.14169841 |
| 29 | 1 | UAV_1 | 1 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.91187861 | 5.001442 | 0.06250686 |
| 35 | 2 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.93433851 | 3.736352 | 0.04494225 |
| 36 | 7 | 55, RSU_1, UAV_1 | 3 | 55 | EDGE_SELECTED | D | — | 0.92361697 | 3.385800 | 0.21486268 |
| 37 | 1 | UAV_1 | 1 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.92018752 | 4.532884 | 0.05610133 |
| 39 | 2 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.92458868 | 4.289735 | 0.05185494 |
| 46 | 1 | UAV_1 | 1 | UAV_1 | EDGE_SELECTED | D | — | 0.93729288 | 3.183907 | 0.10796551 |
| 51 | 5 | 21, RSU_0, RSU_1, UAV_0, UAV_1 | 5 | UAV_0 | EDGE_SELECTED | D | — | 0.97105773 | 1.667159 | 0.01638255 |
| 58 | 2 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.94247208 | 3.277369 | 0.03872470 |
| 7 | 2 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.93510711 | 3.692554 | 0.04442663 |

#### 20003 / Slot 2

| Task ID | Service | 合法 Edge 节点 | Edge 候选数 | Matching 节点 | 类别 | 代码 | 零 Edge 原因 | Task utility | Delay (s) | Vehicle energy (J) |
|---|---:|---|---:|---|---|---|---|---:|---:|---:|
| 1 | 2 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.93747873 | 3.550654 | 0.04397919 |
| 13 | 3 | RSU_0 | 1 | RSU_0 | EDGE_SELECTED | D | — | 0.90462490 | 2.851739 | 0.50115055 |
| 22 | 0 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.93372903 | 3.761362 | 0.04699635 |
| 24 | 5 | 20, RSU_0, RSU_1, UAV_0, UAV_1 | 5 | UAV_1 | EDGE_SELECTED | D | — | 0.93385330 | 3.428446 | 0.10205803 |
| 28 | 5 | 20, RSU_0, RSU_1, UAV_0, UAV_1 | 5 | UAV_0 | EDGE_SELECTED | D | — | 0.97792203 | 1.239280 | 0.01799320 |
| 36 | 3 | RSU_0 | 1 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.91708419 | 4.687152 | 0.06200312 |
| 37 | 4 | 52, 20 | 2 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.94291868 | 3.243853 | 0.03979041 |
| 44 | 1 | UAV_1 | 1 | UAV_1 | EDGE_SELECTED | D | — | 0.94266420 | 2.780048 | 0.12091108 |
| 48 | 4 | 52, 20 | 2 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.93680026 | 3.593893 | 0.04365968 |
| 5 | 4 | 52, 20 | 2 | 52 | EDGE_SELECTED | D | — | 0.93769809 | 3.676764 | 0.02037327 |
| 51 | 6 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.92636530 | 4.164547 | 0.05471634 |
| 53 | 9 | 42, RSU_1 | 2 | RSU_1 | EDGE_SELECTED | D | — | 0.95764134 | 1.454888 | 0.19069705 |
| 56 | 2 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.92685834 | 4.149686 | 0.05214577 |
| 9 | 3 | RSU_0 | 1 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.90544093 | 5.345828 | 0.07062634 |

#### 20003 / Slot 119

| Task ID | Service | 合法 Edge 节点 | Edge 候选数 | Matching 节点 | 类别 | 代码 | 零 Edge 原因 | Task utility | Delay (s) | Vehicle energy (J) |
|---|---:|---|---:|---|---|---|---|---:|---:|---:|
| 1 | 2 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.91859407 | 4.628103 | 0.05642244 |
| 13 | 1 | UAV_1 | 1 | UAV_1 | EDGE_SELECTED | D | — | 0.93230557 | 3.228998 | 0.15177800 |
| 14 | 5 | 20, RSU_0, RSU_1, UAV_0, UAV_1 | 5 | 20 | EDGE_SELECTED | D | — | 0.87727747 | 7.324284 | 0.02629049 |
| 15 | 5 | 20, RSU_0, RSU_1, UAV_0, UAV_1 | 5 | RSU_1 | EDGE_SELECTED | D | — | 0.94046463 | 2.456611 | 0.19833892 |
| 16 | 5 | 20, RSU_0, RSU_1, UAV_0, UAV_1 | 5 | UAV_0 | EDGE_SELECTED | D | — | 0.93384515 | 3.495565 | 0.09078277 |
| 22 | 3 | RSU_0 | 1 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.89320744 | 6.043299 | 0.07877175 |
| 24 | 9 | 42, RSU_1 | 2 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.91508732 | 4.837425 | 0.05716804 |
| 25 | 3 | RSU_0 | 1 | RSU_0 | EDGE_SELECTED | D | — | 0.97627464 | 1.161581 | 0.04813599 |
| 28 | 1 | UAV_1 | 1 | UAV_1 | EDGE_SELECTED | D | — | 0.95377312 | 2.440278 | 0.06382713 |
| 36 | 8 | 57, 23, UAV_0 | 3 | 23 | EDGE_SELECTED | D | — | 0.93565307 | 3.491599 | 0.07280537 |
| 37 | 7 | 52, RSU_1, UAV_1 | 3 | UAV_1 | EDGE_SELECTED | D | — | 0.94593409 | 2.474193 | 0.13894634 |
| 44 | 1 | UAV_1 | 1 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.90918844 | 5.151410 | 0.06487483 |
| 48 | 5 | 20, RSU_0, RSU_1, UAV_0, UAV_1 | 5 | RSU_0 | EDGE_SELECTED | D | — | 0.96809502 | 1.560036 | 0.06507238 |
| 5 | 6 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.93161310 | 3.882824 | 0.04826531 |
| 51 | 5 | 20, RSU_0, RSU_1, UAV_0, UAV_1 | 5 | RSU_1 | EDGE_SELECTED | D | — | 0.95489532 | 1.967944 | 0.13219085 |
| 53 | 5 | 20, RSU_0, RSU_1, UAV_0, UAV_1 | 5 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.90697009 | 5.286273 | 0.06493230 |
| 54 | 3 | RSU_0 | 1 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.90295781 | 5.496390 | 0.07075790 |
| 56 | 5 | 20, RSU_0, RSU_1, UAV_0, UAV_1 | 5 | UAV_0 | EDGE_SELECTED | D | — | 0.94129482 | 3.197179 | 0.06443965 |
| 7 | 3 | RSU_0 | 1 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.90455649 | 5.402925 | 0.07008583 |
| 9 | 1 | UAV_1 | 1 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.91902300 | 4.598020 | 0.05708945 |

#### 20011 / Slot 48

| Task ID | Service | 合法 Edge 节点 | Edge 候选数 | Matching 节点 | 类别 | 代码 | 零 Edge 原因 | Task utility | Delay (s) | Vehicle energy (J) |
|---|---:|---|---:|---|---|---|---|---:|---:|---:|
| 1 | 7 | 27, RSU_1, UAV_1 | 3 | RSU_1 | EDGE_SELECTED | D | — | 0.98657194 | 0.502512 | 0.05346280 |
| 14 | 3 | RSU_0 | 1 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.90746958 | 5.244809 | 0.06679765 |
| 16 | 3 | RSU_0 | 1 | RSU_0 | EDGE_SELECTED | D | — | 0.91917109 | 3.047268 | 0.31801587 |
| 17 | 6 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.91390882 | 4.878551 | 0.06236384 |
| 20 | 5 | 43, RSU_0, RSU_1, UAV_0, UAV_1 | 5 | UAV_0 | EDGE_SELECTED | D | — | 0.96759876 | 1.773509 | 0.03406266 |
| 21 | 5 | 43, RSU_0, RSU_1, UAV_0, UAV_1 | 5 | RSU_0 | EDGE_SELECTED | D | — | 0.95758359 | 2.182932 | 0.06807637 |
| 24 | 9 | 58, RSU_1 | 2 | 58 | EDGE_SELECTED | D | — | 0.91562512 | 5.094162 | 0.00816982 |
| 31 | 6 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.92946520 | 4.001177 | 0.05039043 |
| 32 | 3 | RSU_0 | 1 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.89242603 | 6.088757 | 0.07913860 |
| 33 | 7 | 27, RSU_1, UAV_1 | 3 | UAV_1 | EDGE_SELECTED | D | — | 0.93222087 | 3.024326 | 0.18729095 |
| 34 | 4 | 27, 43 | 2 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.92098670 | 4.495750 | 0.05414242 |
| 35 | 6 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.92550741 | 4.228215 | 0.05279017 |
| 36 | 7 | 27, RSU_1, UAV_1 | 3 | 27 | EDGE_SELECTED | D | — | 0.94577340 | 2.786484 | 0.08775072 |
| 42 | 1 | UAV_1 | 1 | UAV_1 | EDGE_SELECTED | D | — | 0.94025473 | 3.160442 | 0.08138567 |
| 46 | 3 | RSU_0 | 1 | RSU_0 | EDGE_SELECTED | D | — | 0.93000935 | 2.743194 | 0.25768231 |
| 48 | 9 | 58, RSU_1 | 2 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.91606575 | 4.779314 | 0.05691054 |
| 49 | 6 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.93129977 | 3.901084 | 0.04840689 |
| 52 | 1 | UAV_1 | 1 | UAV_1 | EDGE_SELECTED | D | — | 0.93374492 | 3.279372 | 0.12840584 |
| 57 | 3 | RSU_0 | 1 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.91019229 | 5.092104 | 0.06455737 |
| 8 | 4 | 27, 43 | 2 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.93309213 | 3.814726 | 0.04453434 |

#### 20012 / Slot 127

| Task ID | Service | 合法 Edge 节点 | Edge 候选数 | Matching 节点 | 类别 | 代码 | 零 Edge 原因 | Task utility | Delay (s) | Vehicle energy (J) |
|---|---:|---|---:|---|---|---|---|---:|---:|---:|
| 11 | 6 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.93006577 | 3.983232 | 0.04723257 |
| 13 | 1 | UAV_1 | 1 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.89109032 | 6.199697 | 0.07414048 |
| 15 | 5 | 30, RSU_0, RSU_1, UAV_0, UAV_1 | 5 | RSU_0 | EDGE_SELECTED | D | — | 0.96881827 | 1.236023 | 0.11244899 |
| 16 | 7 | 38, RSU_1, UAV_1 | 3 | UAV_1 | EDGE_SELECTED | D | — | 0.94560570 | 2.471572 | 0.14277716 |
| 2 | 2 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.93484705 | 3.730166 | 0.04074363 |
| 23 | 0 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.94534916 | 3.133918 | 0.03332595 |
| 24 | 7 | 38, RSU_1, UAV_1 | 3 | UAV_1 | EDGE_SELECTED | D | — | 0.94989554 | 2.443665 | 0.10325081 |
| 26 | 5 | 30, RSU_0, RSU_1, UAV_0, UAV_1 | 5 | UAV_0 | EDGE_SELECTED | D | — | 0.95987675 | 1.803773 | 0.10859258 |
| 3 | 1 | UAV_1 | 1 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.91729242 | 4.718375 | 0.05457110 |
| 35 | 4 | 38, 30 | 2 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.93768492 | 3.573014 | 0.03806814 |
| 37 | 0 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.93304665 | 3.826775 | 0.04296414 |
| 4 | 3 | RSU_0 | 1 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.91124091 | 5.051059 | 0.06068748 |
| 43 | 7 | 38, RSU_1, UAV_1 | 3 | RSU_1 | EDGE_SELECTED | D | — | 0.97283687 | 0.895275 | 0.12866665 |
| 45 | 7 | 38, RSU_1, UAV_1 | 3 | UAV_1 | EDGE_SELECTED | D | — | 0.95250881 | 2.213026 | 0.11532913 |
| 46 | 4 | 38, 30 | 2 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.92616235 | 4.225068 | 0.04656704 |
| 47 | 4 | 38, 30 | 2 | 38 | EDGE_SELECTED | D | — | 0.92217518 | 4.353392 | 0.06597654 |
| 48 | 0 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.92846971 | 4.085789 | 0.04633871 |
| 52 | 4 | 38, 30 | 2 | CLOUD | VOLUNTARY_CLOUD | C | — | 0.92981123 | 4.021703 | 0.04334727 |
| 53 | 6 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.92821415 | 4.089201 | 0.04839743 |
| 58 | 2 | — | 0 | CLOUD | FORCED_CLOUD | A | service_not_cached_on_any_edge | 0.91614112 | 4.785839 | 0.05502879 |

## B. C 类任务：无竞争时 Cloud 与 Edge 对照

每个 C 类任务都固定原 Slot 的 Cache、UAV、车辆位置、路径和任务工作量，单独作为唯一任务分别调用正式 `evaluate_matching_with_kkt()`。Q 越高代表正式任务效用越高。该对照识别单任务偏好，但不能单独说明原 Matching 的多任务因果。

| Episode/Slot | Task | Service | Cloud Q | Best Edge Q | Best Edge | Cloud delay | Edge delay | Cloud energy | Edge energy | 单任务偏好 |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| 20001/21 | 1 | 4 | 0.93430239 | 0.87977187 | 46 | 3.681806 | 6.130414 | 0.05454637 | 0.20261499 | Cloud |
| 20001/21 | 16 | 9 | 0.91698326 | 0.79801808 | RSU_1 | 4.653936 | 6.039378 | 0.06866580 | 1.06130683 | Cloud |
| 20001/21 | 22 | 4 | 0.93827249 | 0.88073057 | 46 | 3.458941 | 5.839128 | 0.05131335 | 0.24202424 | Cloud |
| 20001/21 | 42 | 3 | 0.89847467 | 0.77513270 | RSU_0 | 5.681826 | 6.513775 | 0.08561739 | 1.21707950 | Cloud |
| 20002/22 | 41 | 9 | 0.93657159 | 0.85601209 | RSU_1 | 3.555240 | 4.310424 | 0.05256016 | 0.75571611 | Cloud |
| 20003/21 | 16 | 9 | 0.93562744 | 0.89660523 | 42 | 3.608335 | 5.808631 | 0.05331301 | 0.08343968 | Cloud |
| 20003/21 | 44 | 4 | 0.94920830 | 0.90503653 | 20 | 2.846285 | 4.751953 | 0.04219906 | 0.17530676 | Cloud |
| 20001/1 | 25 | 1 | 0.91998633 | 0.97134033 | UAV_1 | 4.481099 | 1.319381 | 0.06694086 | 0.07232649 | Edge |
| 20001/1 | 30 | 8 | 0.93045959 | 0.97269913 | UAV_0 | 3.892988 | 1.132186 | 0.05844343 | 0.08999196 | Edge |
| 20001/1 | 36 | 3 | 0.90282854 | 0.36062518 | RSU_0 | 5.438580 | 18.109294 | 0.08187525 | 3.53023492 | Cloud |
| 20001/1 | 44 | 8 | 0.94870428 | 0.97824372 | UAV_0 | 2.872093 | 0.904989 | 0.04303024 | 0.07125148 | Edge |
| 20002/10 | 10 | 9 | 0.94027847 | 0.89416670 | 11 | 3.347845 | 5.722909 | 0.04942412 | 0.12310077 | Cloud |
| 20002/10 | 15 | 3 | 0.90291928 | 0.00968202 | RSU_0 | 5.433874 | 27.935110 | 0.08173563 | 5.48723814 | Cloud |
| 20002/10 | 29 | 1 | 0.92529494 | 0.97021981 | UAV_1 | 4.183751 | 1.314134 | 0.06250686 | 0.08477259 | Edge |
| 20002/10 | 37 | 1 | 0.93295917 | 0.97885904 | UAV_1 | 3.754484 | 1.020892 | 0.05610133 | 0.04528877 | Edge |
| 20003/2 | 36 | 3 | 0.92646230 | 0.61820619 | RSU_0 | 4.115580 | 10.843309 | 0.06200312 | 2.10302212 | Cloud |
| 20003/2 | 37 | 4 | 0.95234645 | 0.88187215 | 20 | 2.669255 | 4.964951 | 0.03979041 | 0.37819725 | Cloud |
| 20003/2 | 48 | 4 | 0.94742787 | 0.91698294 | 20 | 2.946167 | 4.519863 | 0.04365968 | 0.09136002 | Cloud |
| 20003/2 | 9 | 3 | 0.91600794 | 0.88993176 | RSU_0 | 4.701795 | 3.295563 | 0.07062634 | 0.57759516 | Cloud |
| 20003/119 | 22 | 3 | 0.90651287 | -0.30844782 | RSU_0 | 5.232367 | 36.815419 | 0.07877175 | 7.26579496 | Cloud |
| 20003/119 | 24 | 9 | 0.93089482 | 0.84900190 | 42 | 3.873998 | 7.453785 | 0.05716804 | 0.29603386 | Cloud |
| 20003/119 | 44 | 1 | 0.92266708 | 0.95725980 | UAV_1 | 4.329921 | 1.641524 | 0.06487483 | 0.16304571 | Edge |
| 20003/119 | 53 | 5 | 0.92208649 | 0.97641855 | UAV_0 | 4.364967 | 1.241114 | 0.06493230 | 0.03319111 | Edge |
| 20003/119 | 54 | 3 | 0.91594485 | 0.87790886 | RSU_0 | 4.704863 | 3.619946 | 0.07075790 | 0.64671098 | Cloud |
| 20003/119 | 7 | 3 | 0.91689775 | -0.05177619 | RSU_0 | 4.650757 | 29.602053 | 0.07008583 | 5.83905682 | Cloud |
| 20003/119 | 9 | 1 | 0.93190914 | 0.96897101 | UAV_1 | 3.812642 | 1.270539 | 0.05708945 | 0.10503193 | Edge |
| 20011/48 | 14 | 3 | 0.92061873 | -0.05032922 | RSU_0 | 4.443401 | 29.576477 | 0.06679765 | 5.82846005 | Cloud |
| 20011/48 | 32 | 3 | 0.90609277 | -0.29295227 | RSU_0 | 5.255803 | 36.380982 | 0.07913860 | 7.17948493 | Cloud |
| 20011/48 | 34 | 4 | 0.93511047 | 0.86079732 | 27 | 3.634942 | 6.216525 | 0.05414242 | 0.38376261 | Cloud |
| 20011/48 | 48 | 9 | 0.93148744 | 0.88363952 | 58 | 3.839401 | 6.099256 | 0.05691054 | 0.16799381 | Cloud |
| 20011/48 | 57 | 3 | 0.92325336 | 0.90240541 | RSU_0 | 4.296065 | 2.924345 | 0.06455737 | 0.51175644 | Cloud |
| 20011/48 | 8 | 4 | 0.94650767 | 0.84138783 | 43 | 2.997083 | 6.502930 | 0.04453434 | 0.53549801 | Cloud |
| 20012/127 | 13 | 1 | 0.91140122 | 0.95701581 | UAV_1 | 4.961799 | 1.770328 | 0.07414048 | 0.14376333 | Edge |
| 20012/127 | 3 | 1 | 0.93481850 | 0.96926846 | UAV_1 | 3.650204 | 1.269332 | 0.05457110 | 0.10216807 | Edge |
| 20012/127 | 35 | 4 | 0.95424215 | 0.89081297 | 30 | 2.563892 | 4.863133 | 0.03806814 | 0.30320530 | Cloud |
| 20012/127 | 4 | 3 | 0.92785784 | 0.68927788 | RSU_0 | 4.038299 | 8.873610 | 0.06068748 | 1.70328039 | Cloud |
| 20012/127 | 46 | 4 | 0.94407479 | 0.89010650 | 30 | 3.133350 | 5.223068 | 0.04656704 | 0.24957609 | Cloud |
| 20012/127 | 52 | 4 | 0.94781134 | 0.92592147 | 38 | 2.924641 | 4.244498 | 0.04334727 | 0.04576327 | Cloud |

- 完成 `38/38` 个单任务比较；Cloud 单任务效用更高 `28` 个，Edge 更高 `10` 个，容差内平局 `0` 个。单任务 Edge 更高、原 Slot 却选 Cloud 的任务有 `10` 个。
- 若原 Slot 选择 Cloud 且单任务 Cloud Q 已更高，多任务竞争不是 Cloud 选择的必要条件；若单任务 Edge 更高，才需进一步结合下节固定节点转移检查其他任务的资源外部效应或 Matching 局部解。

## C. 固定其他卸载节点的单任务 Transfer

每个 C 类目标任务分别尝试转移到其每个合法 Edge；其他任务卸载节点保持 Matching 原分配，然后用正式 KKT 重算全部任务。此为有限的一步局部对照，不是全局最优搜索。

| Episode/Slot | Task | 原节点 | 最佳 Edge | ΔQ 目标任务 | ΔQ 其他任务 | ΔQ 系统 | 分解核验 | 解读 |
|---|---:|---|---|---:|---:|---:|---|---|
| 20001/21 | 1 | CLOUD | 46 | -0.03990564 | 0.01462487 | -0.02528077 | True | 目标自身未获益 |
| 20001/21 | 16 | CLOUD | RSU_1 | -0.10207537 | 0.01688981 | -0.08518557 | True | 目标自身未获益 |
| 20001/21 | 22 | CLOUD | 46 | -0.04353566 | 0.01400626 | -0.02952940 | True | 目标自身未获益 |
| 20001/21 | 42 | CLOUD | RSU_0 | -0.10999612 | 0.01334584 | -0.09665028 | True | 目标自身未获益 |
| 20002/22 | 41 | CLOUD | 11 | -0.08190417 | 0.00869937 | -0.07320481 | True | 目标自身未获益 |
| 20003/21 | 16 | CLOUD | 42 | -0.03016960 | 0.00885261 | -0.02131699 | True | 目标自身未获益 |
| 20003/21 | 44 | CLOUD | 20 | -0.03638211 | 0.00778965 | -0.02859245 | True | 目标自身未获益 |
| 20001/1 | 25 | CLOUD | UAV_1 | 0.02328990 | -0.02759738 | -0.00430749 | True | 目标获益被其他任务损失部分/全部抵消 |
| 20001/1 | 30 | CLOUD | 59 | -0.02516646 | 0.00712230 | -0.01804416 | True | 目标自身未获益 |
| 20001/1 | 36 | CLOUD | RSU_0 | -0.55001313 | -0.00835563 | -0.55836876 | True | 目标自身未获益 |
| 20001/1 | 44 | CLOUD | 59 | -0.02190309 | 0.00651689 | -0.01538620 | True | 目标自身未获益 |
| 20002/10 | 10 | CLOUD | 11 | -0.03187463 | 0.01423714 | -0.01763749 | True | 目标自身未获益 |
| 20002/10 | 15 | CLOUD | RSU_0 | -0.87996364 | 0.01327363 | -0.86669001 | True | 目标自身未获益 |
| 20002/10 | 29 | CLOUD | UAV_1 | 0.00140563 | -0.04237068 | -0.04096504 | True | 目标获益被其他任务损失部分/全部抵消 |
| 20002/10 | 37 | CLOUD | UAV_1 | 0.00610268 | -0.03871886 | -0.03261619 | True | 目标获益被其他任务损失部分/全部抵消 |
| 20003/2 | 36 | CLOUD | RSU_0 | -0.31146747 | -0.00334118 | -0.31480866 | True | 目标自身未获益 |
| 20003/2 | 37 | CLOUD | 20 | -0.06104653 | 0.00942777 | -0.05161876 | True | 目标自身未获益 |
| 20003/2 | 48 | CLOUD | 20 | -0.01981733 | 0.01062761 | -0.00918972 | True | 目标自身未获益 |
| 20003/2 | 9 | CLOUD | RSU_0 | -0.02920776 | -0.00315457 | -0.03236232 | True | 目标自身未获益 |
| 20003/119 | 22 | CLOUD | RSU_0 | -1.22830406 | -0.01385063 | -1.24215469 | True | 目标自身未获益 |
| 20003/119 | 24 | CLOUD | 42 | -0.06608542 | 0.01580750 | -0.05027792 | True | 目标自身未获益 |
| 20003/119 | 44 | CLOUD | UAV_1 | -0.00455549 | -0.03827559 | -0.04283108 | True | 目标自身未获益 |
| 20003/119 | 53 | CLOUD | UAV_0 | 0.01998259 | -0.03388030 | -0.01389771 | True | 目标获益被其他任务损失部分/全部抵消 |
| 20003/119 | 54 | CLOUD | RSU_0 | -0.05047908 | -0.01262615 | -0.06310522 | True | 目标自身未获益 |
| 20003/119 | 7 | CLOUD | RSU_0 | -0.98111022 | -0.01288999 | -0.99400020 | True | 目标自身未获益 |
| 20003/119 | 9 | CLOUD | UAV_1 | 0.00059462 | -0.03545937 | -0.03486475 | True | 目标获益被其他任务损失部分/全部抵消 |
| 20011/48 | 14 | CLOUD | RSU_0 | -0.99880228 | -0.02830314 | -1.02710542 | True | 目标自身未获益 |
| 20011/48 | 32 | CLOUD | RSU_0 | -1.22901561 | -0.03051965 | -1.25953525 | True | 目标自身未获益 |
| 20011/48 | 34 | CLOUD | 43 | -0.06085739 | 0.01412377 | -0.04673362 | True | 目标自身未获益 |
| 20011/48 | 48 | CLOUD | 58 | -0.11965024 | -0.07185946 | -0.19150971 | True | 目标自身未获益 |
| 20011/48 | 57 | CLOUD | RSU_0 | -0.04779308 | -0.02648870 | -0.07428178 | True | 目标自身未获益 |
| 20011/48 | 8 | CLOUD | 43 | -0.09170430 | 0.01341554 | -0.07828876 | True | 目标自身未获益 |
| 20012/127 | 13 | CLOUD | UAV_1 | 0.01085129 | -0.03370737 | -0.02285608 | True | 目标获益被其他任务损失部分/全部抵消 |
| 20012/127 | 3 | CLOUD | UAV_1 | 0.00511222 | -0.02825462 | -0.02314240 | True | 目标获益被其他任务损失部分/全部抵消 |
| 20012/127 | 35 | CLOUD | 30 | -0.04687195 | 0.01655723 | -0.03031472 | True | 目标自身未获益 |
| 20012/127 | 4 | CLOUD | RSU_0 | -0.23659070 | 0.00159523 | -0.23499546 | True | 目标自身未获益 |
| 20012/127 | 46 | CLOUD | 30 | -0.03605586 | 0.01791244 | -0.01814342 | True | 目标自身未获益 |
| 20012/127 | 52 | CLOUD | 38 | -0.07364435 | -0.05170585 | -0.12535020 | True | 目标自身未获益 |

- C 类目标中存在正系统净 Q 单次转移的有 `0/38` 个；至少一个 Edge 转移表现为目标任务获益、其他任务损失且总 Q 不增的有 `8` 个。该数量是本地转移证据，不能单独证明 Cloud 是唯一或全局最优选择。

## D. 20 任务 Slot 的嵌套负载对照

每个 20 任务 Slot 最多取两个 C 类代表任务；子集总含该代表任务，其余任务按固定词法 Task ID 顺序补入。它是同一真实 Slot 构造的嵌套子集，不是独立自然到达 Slot，也不代表普遍负载阈值。

| Episode/Slot | 代表任务 | Service | 任务数 | 合法 Edge | Cloud 可行 | Matching 节点 | 状态 |
|---|---:|---:|---:|---|---|---|---|
| 20003/119 | 22 | 3 | 1 | RSU_0 | True | CLOUD | COMPLETE |
| 20003/119 | 22 | 3 | 5 | RSU_0 | True | CLOUD | COMPLETE |
| 20003/119 | 22 | 3 | 10 | RSU_0 | True | CLOUD | COMPLETE |
| 20003/119 | 22 | 3 | 14 | RSU_0 | True | CLOUD | COMPLETE |
| 20003/119 | 22 | 3 | 20 | RSU_0 | True | CLOUD | COMPLETE |
| 20003/119 | 24 | 9 | 1 | 42, RSU_1 | True | CLOUD | COMPLETE |
| 20003/119 | 24 | 9 | 5 | 42, RSU_1 | True | CLOUD | COMPLETE |
| 20003/119 | 24 | 9 | 10 | 42, RSU_1 | True | CLOUD | COMPLETE |
| 20003/119 | 24 | 9 | 14 | 42, RSU_1 | True | CLOUD | COMPLETE |
| 20003/119 | 24 | 9 | 20 | 42, RSU_1 | True | CLOUD | COMPLETE |
| 20011/48 | 14 | 3 | 1 | RSU_0 | True | CLOUD | COMPLETE |
| 20011/48 | 14 | 3 | 5 | RSU_0 | True | CLOUD | COMPLETE |
| 20011/48 | 14 | 3 | 10 | RSU_0 | True | CLOUD | COMPLETE |
| 20011/48 | 14 | 3 | 14 | RSU_0 | True | CLOUD | COMPLETE |
| 20011/48 | 14 | 3 | 20 | RSU_0 | True | CLOUD | COMPLETE |
| 20011/48 | 32 | 3 | 1 | RSU_0 | True | CLOUD | COMPLETE |
| 20011/48 | 32 | 3 | 5 | RSU_0 | True | CLOUD | COMPLETE |
| 20011/48 | 32 | 3 | 10 | RSU_0 | True | CLOUD | COMPLETE |
| 20011/48 | 32 | 3 | 14 | RSU_0 | True | CLOUD | COMPLETE |
| 20011/48 | 32 | 3 | 20 | RSU_0 | True | CLOUD | COMPLETE |
| 20012/127 | 13 | 1 | 1 | UAV_1 | True | UAV_1 | COMPLETE |
| 20012/127 | 13 | 1 | 5 | UAV_1 | True | UAV_1 | COMPLETE |
| 20012/127 | 13 | 1 | 10 | UAV_1 | True | CLOUD | COMPLETE |
| 20012/127 | 13 | 1 | 14 | UAV_1 | True | CLOUD | COMPLETE |
| 20012/127 | 13 | 1 | 20 | UAV_1 | True | CLOUD | COMPLETE |
| 20012/127 | 3 | 1 | 1 | UAV_1 | True | UAV_1 | COMPLETE |
| 20012/127 | 3 | 1 | 5 | UAV_1 | True | CLOUD | COMPLETE |
| 20012/127 | 3 | 1 | 10 | UAV_1 | True | CLOUD | COMPLETE |
| 20012/127 | 3 | 1 | 14 | UAV_1 | True | CLOUD | COMPLETE |
| 20012/127 | 3 | 1 | 20 | UAV_1 | True | CLOUD | COMPLETE |

- 按上述固定任务补入顺序，代表任务从 1 任务 Edge 解转为任一更高规模 Cloud 解的样本数：`2`；任务 ID：`20012/127/task13, 20012/127/task3`。该观察只描述选定 Slot 与补入顺序。

## E. RESOURCE_AWARE_FALLBACK 与 LEGACY_EDGE_GATE

两种模式均调用同一正式 Transfer–Exchange Matching 与 KKT。Edge gate 仅作对照：有物理可行 Edge 时不允许 Cloud。成本状态固定，故表中比较任务效用 Q。

| Episode/Slot | 任务数 | Resource-aware Q | Edge-gate Q | ΔQ (RA−gate) | Cloud 数 RA | Cloud 数 gate | Mean delay RA/gate (s) | Mean vehicle energy RA/gate (J) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20001/21 | 10 | 9.24880366 | 8.92290314 | 0.32590052 | 9 | 5 | 4.250051/4.571022 | 0.05556247/0.33740441 |
| 20002/22 | 10 | 9.40089246 | 9.32768765 | 0.07320481 | 6 | 5 | 3.236494/3.488680 | 0.07022124/0.10305076 |
| 20003/21 | 10 | 9.43452822 | 9.38087017 | 0.05365805 | 6 | 4 | 2.865787/3.096370 | 0.09826574/0.11458918 |
| 20001/1 | 14 | 13.24209969 | 12.61800039 | 0.62409929 | 6 | 2 | 2.980287/4.036404 | 0.05401370/0.33509751 |
| 20002/10 | 14 | 13.03219256 | 12.01814997 | 1.01404260 | 9 | 5 | 3.768472/5.730657 | 0.07527441/0.49031525 |
| 20003/2 | 14 | 13.08107932 | 12.50483877 | 0.57624055 | 8 | 4 | 3.423439/4.567214 | 0.09765003/0.32863687 |
| 20003/119 | 20 | 18.56101066 | 15.92565173 | 2.63535893 | 9 | 2 | 3.906247/7.511702 | 0.08104879/0.83002773 |
| 20011/48 | 20 | 18.60896743 | 15.80638794 | 2.80257949 | 10 | 4 | 3.705934/7.777614 | 0.09021676/0.84653400 |
| 20012/127 | 20 | 18.70113288 | 18.05568473 | 0.64544815 | 13 | 7 | 3.542028/4.583397 | 0.07042273/0.22706599 |

- 九个 Slot 总 ΔQ (Resource-aware − Edge-gate)：`8.75053238`；任务数加权平均时延 RA/gate：`3.553169/5.376476` s；车辆上传能耗：`0.07766208/0.45287646` J；Cloud 数：`76/38`。
- 这两组结果仍是局部搜索输出。禁止 Cloud 后的差值不能证明任一模式达到全局最优，也不能单独归因为负载竞争。

## F. 三个 10 任务 Slot 的精确最优性对照

### DP 分解条件

- 当前模型为 `independent_per_slot`；频谱合同为 `ORTHOGONAL_INDEPENDENT`，干扰为 `NONE`。Edge KKT 以节点为单位独立求解，每节点 CPU 与带宽资源各自受总份额约束。
- Cloud CPU 是 Cloud 节点内有限共享资源，由正式 Cloud KKT 对该子集的全部 Cloud 任务共同分配；Cloud 上行沿用正式每任务等效速率和时延因子，不与 Edge KKT 带宽耦合。
- 每个任务分配到一个合法 Edge 或 Cloud。Edge 节点间没有共享资源或互扰，唯一的跨任务共享子问题在 Cloud 内，故每个节点子集的正式 KKT 最优效用可相加。DP 的目标是正式任务效用总和，Cache/UAV 固定。
- 对每节点的全部合法任务子集调用 `evaluate_matching_with_kkt`；空集效用为 0，KKT 不可行的子集禁用。每个 Slot 的 DP 上限为 90 秒。只有完成全量子集枚举、DP 回溯与完整 KKT 复核才标记 `EXACT_OPTIMUM_VERIFIED`。
- Matching 计时协议：先对 Episode 20001 / Slot 21 执行一次预热（耗时 3.495371 s，不计入稳态样本）；三个精确对照 Slot 各重复 3 次，表中采用 optimizer-only 耗时中位数。首次冷启动不是稳态 Matching 时间。

| Episode / Slot | Matching Q | Exact Q | Gap | Matching 时间 | Exact DP 时间 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| 20001 / 21 | 9.248803658 | 9.248803658 | 0.000000% | 0.005350 s | 0.075 s | EXACT_OPTIMUM_VERIFIED |
| 20002 / 22 | 9.400892455 | 9.400892455 | 0.000000% | 0.013771 s | 0.099 s | EXACT_OPTIMUM_VERIFIED |
| 20003 / 21 | 9.434528220 | 9.445712124 | 0.118402% | 0.010008 s | 0.080 s | EXACT_OPTIMUM_VERIFIED |

- 尝试 `3` 个状态，完整验证 `3` 个；Verified 状态平均 gap `0.039467%`，最大 gap `0.118402%`。
- 与 Matching 不同的任务分配数：`2`；逐 Slot 明细如下。
- Matching 与 DP 解分别经完整 `evaluate_matching_with_kkt` 复核；DP 子集效用和、递推结果、完整评价在 1e-8 容差内一致，且 Exact Q 不低于 Matching。资源核验包括 Edge CPU/带宽总份额及最小份额、Cloud 每任务最小 CPU 份额、Cloud 有限 CPU 总量和任务效用有限性。

### 最优解中发生变化的任务

#### Episode 20001 / Slot 21

- Matching Cloud Task IDs：`1, 15, 16, 22, 34, 36, 42, 45, 50`；Exact Cloud Task IDs：`1, 15, 16, 22, 34, 36, 42, 45, 50`；集合相同：`True`。

| Task | Matching 节点 | Exact 节点 | Matching utility | Exact utility | 差值 |
|---|---|---|---:|---:|---:|
| — | assignment identical | assignment identical | — | — | — |

#### Episode 20002 / Slot 22

- Matching Cloud Task IDs：`10, 27, 35, 41, 48, 50`；Exact Cloud Task IDs：`10, 27, 35, 41, 48, 50`；集合相同：`True`。

| Task | Matching 节点 | Exact 节点 | Matching utility | Exact utility | 差值 |
|---|---|---|---:|---:|---:|
| — | assignment identical | assignment identical | — | — | — |

#### Episode 20003 / Slot 21

- Matching Cloud Task IDs：`16, 28, 37, 44, 53, 7`；Exact Cloud Task IDs：`13, 16, 28, 37, 44, 53, 7`；集合相同：`False`。

| Task | Matching 节点 | Exact 节点 | Matching utility | Exact utility | 差值 |
|---|---|---|---:|---:|---:|
| 54 | UAV_1 | RSU_0 | 0.94064980 | 0.97048093 | 0.02983113 |
| 13 | RSU_0 | CLOUD | 0.92136083 | 0.91191784 | -0.00944299 |

## 结论

- 四类计数：A 未缓存被迫 Cloud `38`；B 已缓存但无物理可行 Edge `0`；C 有合法 Edge 仍选 Cloud `38`；D 选择 Edge `56`。Cloud 机制结论：`MIXED_SOLO_EDGE_PREFERENCE_WITH_OTHER_TASK_RESOURCE_LOSS`。
- 有合法 Edge 候选仍选择 Cloud：132 个到达任务中有 38 个存在合法 Edge 候选但最终分配到 Cloud；该现象在 9/9 个 Slot 出现。
- 单任务对照中 Cloud 优于最佳 Edge `28` 次、Edge 更优 `10` 次；单任务 Edge 更优而原多任务 Matching 仍选 Cloud `10` 次。
- 固定其他任务节点的转移：8 个目标任务本身获益却伴随其他任务损失且系统 Q 未升高；存在正系统净 Q 单步 Edge 转移的 C 类目标 `0/38`。
- 嵌套负载对照中观察到 Edge→Cloud 的代表任务数 `2`。这里只对应指定真实 Slot 与 Task ID 加入顺序，不构成通用 Cloud 负载阈值。
- Resource-aware 相对 Edge-gate 的总任务效用差为 `8.75053238`；两种模式都仅是局部 Matching 结果，不构成全局最优证明。
- Matching 最优性：2/3 个已验证案例达到精确最优；最大相对 gap 为 0.118402%。
- 更换 Matching：当前三个指定 Slot 不支持更换 Matching；最大相对 gap 为 0.118402%。结论仅适用于完整核验的三个指定 10 任务 Slot，不能外推到更大任务数或其他状态。
- GNN 必要性：`NO_EVIDENCE_FROM_THIS_DIAGNOSTIC`。若发现明显 gap，应先定位交换邻域或候选覆盖，再考虑学习式方法。
- 下一步：不因这三个案例替换 Matching；先检查最大 gap 案例的交换邻域与候选覆盖。

## 代码依据与边界

依据 `task_allocation.py` 的 `build_feasible_targets`、`_edge_path_is_feasible`、`_allocate_phase_resources_kkt`、`_allocate_cloud_cpu_kkt`、`evaluate_matching_with_kkt` 和 `optimize_task_matching`；配置为 independent-per-slot、orthogonal-independent、no-interference 和 RESOURCE_AWARE_FALLBACK。
只读取 Training 数据；未运行 PPO、Validation、Test 或完整 Episode。未修改正式 Matching/KKT、配置、Reward 或环境数据。

```text
TRAINING_SLOTS_ANALYZED = 9
CACHE_MISS_FORCED_CLOUD_COUNT = 38
CACHED_BUT_EDGE_INFEASIBLE_COUNT = 0
EDGE_FEASIBLE_CLOUD_SELECTED_COUNT = 38
EDGE_SELECTED_COUNT = 56
VOLUNTARY_CLOUD_EXISTS = YES
SOLO_CLOUD_BETTER_THAN_EDGE_COUNT = 28
SOLO_EDGE_BETTER_BUT_MULTI_CLOUD_COUNT = 10
RESOURCE_AWARE_VS_EDGE_GATE_DELTA_Q = 8.75053238
EXACT_SLOTS_ATTEMPTED = 3
EXACT_SLOTS_VERIFIED = 3
MATCHING_OPTIMALITY_GAP_MEAN = 0.039467%
MATCHING_OPTIMALITY_GAP_MAX = 0.118402%
MATCHING_EXACT_SOLUTION_DIFFERENCES = 2
CLOUD_SELECTION_MAIN_CAUSE = MIXED_SOLO_EDGE_PREFERENCE_WITH_OTHER_TASK_RESOURCE_LOSS
MATCHING_QUALITY_ASSESSMENT = 2/3 个已验证案例达到精确最优；最大相对 gap 为 0.118402%。
GNN_NECESSITY_ASSESSMENT = NO_EVIDENCE_FROM_THIS_DIAGNOSTIC
RECOMMENDED_NEXT_STEP = 不因这三个案例替换 Matching；先检查最大 gap 案例的交换邻域与候选覆盖。
PPO_TRAINING_EXECUTED = NO
FORMAL_ALGORITHM_MODIFIED = NO
FORMAL_CONFIG_MODIFIED = NO
```
