# -*- coding: utf-8 -*-
"""
P1：缓存配置局部搜索。

当前正式模型：
    TV 不进行本地计算；
    动态缓存节点属于 SV / RSU / UAV；
    P1 只负责决定缓存状态，不负责求任务卸载资源分配；
    每一个候选缓存配置最终都必须调用完整 P3 评价。

重要的因果要求：
    P1 不能直接使用未来真实 task_trace / popularity_trace。
    本模块因此不读取未来任务，而是接收 evaluate_p3_cost(...) 回调。
    后续在线控制器会用同一组预测场景 Xi_t 对所有候选配置评价。

本模块实现：
1. keep / add / delete / swap 邻域；
2. 可选 proxy-demand 候选；
3. 物理缓存容量约束；
4. 缓存切换成本；
5. 切换成本始终相对“真实旧物理状态”计算；
6. 节点级 best-improvement；
7. 严格改善才接受，因此有限状态下必然终止。

当前代价形式：
    P3 返回归一化任务代价 J，越小越好。

因此 P1 最小化：
    score(C) = predicted_P3_cost(C) + cache_switch_cost(C, C_old)

这与最大化效用减去切换成本是等价写法。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cache_model
import config


# =============================================================================
# 类型定义
# =============================================================================

CacheState = dict[tuple[str, int], bool]

# 输入一个候选缓存状态，返回该候选在“同一组预测场景”下的
# P3 期望归一化代价。越小越好。
ConfigEvaluator = Callable[[CacheState], float]


@dataclass
class P1Move:
    """一次缓存邻域动作。"""

    node_id: str
    move_type: str
    removed_services: tuple[int, ...]
    added_services: tuple[int, ...]


@dataclass
class P1Step:
    """记录一次被接受的严格改善。"""

    iteration: int
    node_id: str
    move_type: str
    removed_services: tuple[int, ...]
    added_services: tuple[int, ...]
    p3_cost: float
    switch_cost: float
    total_score: float
    improvement: float


@dataclass
class P1Result:
    """P1 的最终输出。"""

    cache_state: CacheState
    p3_cost: float
    switch_cost: float
    total_score: float
    changed: bool
    steps: list[P1Step]


# =============================================================================
# 缓存状态基础操作
# =============================================================================

def copy_cache_state(cache_state: CacheState) -> CacheState:
    """
    CacheState 的 value 都是 bool，因此浅拷贝已经足够。
    """
    return dict(cache_state)


def cached_services(
    cache_state: CacheState,
    node_id: str,
) -> set[int]:
    """返回节点当前缓存的服务集合。"""
    return {
        service_id
        for service_id in range(config.NUM_SERVICES)
        if cache_state.get((node_id, service_id), False)
    }


def replace_node_cache(
    cache_state: CacheState,
    node_id: str,
    new_services: set[int],
) -> CacheState:
    """
    返回一个新缓存状态，只修改指定节点。

    不原地修改输入状态，避免候选评价之间相互污染。
    """
    candidate = copy_cache_state(cache_state)

    for service_id in range(config.NUM_SERVICES):
        candidate[(node_id, service_id)] = (
            service_id in new_services
        )

    return candidate


def state_signature(cache_state: CacheState) -> tuple:
    """
    缓存状态的不可变签名，用于邻域去重。
    """
    return tuple(
        sorted(
            (
                str(node_id),
                int(service_id),
                bool(cached),
            )
            for (node_id, service_id), cached
            in cache_state.items()
        )
    )


# =============================================================================
# 节点物理缓存容量
# =============================================================================

def node_capacity_mb(
    node: dict,
    service_sizes_mb: dict[int, float],
) -> float:
    """
    返回节点物理缓存容量。

    当前 P1 正式支持：
        SV / UAV

    具体容量参数统一由 cache_model/config 管理，
    P1 不再自行维护节点类型分支。
    """
    return (
        cache_model.node_cache_capacity_mb(
            node["node_type"],
            service_sizes_mb,
        )
    )


def used_capacity_mb(
    services: set[int],
    service_sizes_mb: dict[int, float],
) -> float:
    """计算服务集合占用的缓存容量。"""
    return sum(
        service_sizes_mb[service_id]
        for service_id in services
    )


def node_cache_feasible(
    node: dict,
    services: set[int],
    service_sizes_mb: dict[int, float],
) -> bool:
    """
    检查：
        sum_f s_f c_{m,f} <= C_m
    """
    return (
        used_capacity_mb(
            services,
            service_sizes_mb,
        )
        <= node_capacity_mb(
            node,
            service_sizes_mb,
        )
        + 1e-12
    )


# =============================================================================
# P1 邻域生成
# =============================================================================

def _make_move(
    node_id: str,
    move_type: str,
    removed: set[int] | tuple[int, ...] = (),
    added: set[int] | tuple[int, ...] = (),
) -> P1Move:
    """统一构造动作记录。"""
    return P1Move(
        node_id=str(node_id),
        move_type=move_type,
        removed_services=tuple(sorted(removed)),
        added_services=tuple(sorted(added)),
    )


def build_proxy_candidate(
    current_services: set[int],
    proxy_order: list[int],
    capacity_mb: float,
    service_sizes_mb: dict[int, float],
) -> set[int]:
    """
    根据外部给定的“代理需求排序”生成一个候选缓存集合。

    重要：
    这里不读取真实未来流行度，也不自行假设预测模型。

    proxy_order 必须由外部在线预测模块基于 H_{t-1} 生成，例如：
        [最值得缓存的服务, 次值得缓存的服务, ...]

    本函数只按给定顺序在容量允许时依次装入服务。
    该候选只是搜索邻域中的一个启发式候选，
    最终是否接受仍由完整 P3 决定。
    """
    selected: set[int] = set()
    used_mb = 0.0

    seen = set()

    # 先按预测代理顺序考虑。
    ordered_services = []

    for service_id in proxy_order:
        if (
            service_id in service_sizes_mb
            and service_id not in seen
        ):
            ordered_services.append(service_id)
            seen.add(service_id)

    # 若 proxy_order 没覆盖所有服务，把剩余服务放在最后。
    # 这样函数始终能构造一个完整、确定性的候选。
    for service_id in sorted(service_sizes_mb):
        if service_id not in seen:
            ordered_services.append(service_id)

    for service_id in ordered_services:
        size_mb = service_sizes_mb[service_id]

        if used_mb + size_mb <= capacity_mb + 1e-12:
            selected.add(service_id)
            used_mb += size_mb

    return selected


def generate_node_neighbors(
    cache_state: CacheState,
    node: dict,
    service_sizes_mb: dict[int, float],
    proxy_order: list[int] | None = None,
) -> list[tuple[CacheState, P1Move]]:
    """
    为一个节点生成 P1 邻域。

    邻域包括：
    1. add：增加一个当前未缓存服务；
    2. delete：删除一个当前缓存服务；
    3. swap：删除一个、增加一个；
    4. proxy：可选，由外部因果预测给出的服务排序构造。

    keep 不需要生成成新状态，因为当前 cache_state 本身就是 keep。
    """
    node_id = str(node["node_id"])

    current = cached_services(
        cache_state,
        node_id,
    )

    all_services = set(service_sizes_mb)
    missing = all_services - current

    neighbors: list[tuple[CacheState, P1Move]] = []
    seen_signatures = {state_signature(cache_state)}

    def add_candidate(
        new_services: set[int],
        move: P1Move,
    ) -> None:
        """容量合法且不重复时加入邻域。"""
        if not node_cache_feasible(
            node,
            new_services,
            service_sizes_mb,
        ):
            return

        candidate = replace_node_cache(
            cache_state,
            node_id,
            new_services,
        )

        signature = state_signature(candidate)

        if signature in seen_signatures:
            return

        seen_signatures.add(signature)
        neighbors.append((candidate, move))

    # ------------------------------------------------------------------
    # add：只在容量允许时添加。
    # ------------------------------------------------------------------
    for service_in in sorted(missing):
        new_services = set(current)
        new_services.add(service_in)

        add_candidate(
            new_services,
            _make_move(
                node_id,
                "add",
                added={service_in},
            ),
        )

    # ------------------------------------------------------------------
    # delete：删除后一定不会违反容量。
    # ------------------------------------------------------------------
    for service_out in sorted(current):
        new_services = set(current)
        new_services.remove(service_out)

        add_candidate(
            new_services,
            _make_move(
                node_id,
                "delete",
                removed={service_out},
            ),
        )

    # ------------------------------------------------------------------
    # swap：一个服务换另一个服务。
    # ------------------------------------------------------------------
    for service_out in sorted(current):
        for service_in in sorted(missing):
            new_services = set(current)
            new_services.remove(service_out)
            new_services.add(service_in)

            add_candidate(
                new_services,
                _make_move(
                    node_id,
                    "swap",
                    removed={service_out},
                    added={service_in},
                ),
            )

    # ------------------------------------------------------------------
    # proxy-demand：由外部因果预测模块给出服务优先级。
    # ------------------------------------------------------------------
    if proxy_order is not None:
        proxy_services = build_proxy_candidate(
            current_services=current,
            proxy_order=proxy_order,
            capacity_mb=node_capacity_mb(
                node,
                service_sizes_mb,
            ),
            service_sizes_mb=service_sizes_mb,
        )

        removed = current - proxy_services
        added = proxy_services - current

        add_candidate(
            proxy_services,
            _make_move(
                node_id,
                "proxy",
                removed=removed,
                added=added,
            ),
        )

    return neighbors


# =============================================================================
# 缓存切换成本
# =============================================================================

def cache_switch_cost(
    candidate_state: CacheState,
    old_physical_state: CacheState,
    edge_nodes: list[dict],
    service_sizes_mb: dict[int, float],
) -> float:
    """
    计算缓存切换成本。

    对每个节点 m：
        C_m^ca
        = z_m^ca * C_m^fix
          + xi_m^ca * sum_{f in S_m \\ S_m^-} s_f

    当前 config.py 中：
        CACHE_FIXED_SWITCH_COST
        CACHE_SIZE_COST_COEF

    已把 eta_ca 与物理更新能耗系数统一吸收到
    CACHE_SIZE_COST_COEF 中。

    最重要的语义：
        old_physical_state 始终是真实旧状态 C(t-1)，
        绝不能使用搜索过程中的上一个迭代状态替代它。
    """
    total = 0.0

    for node in edge_nodes:
        node_id = str(node["node_id"])

        old_set = cached_services(
            old_physical_state,
            node_id,
        )

        new_set = cached_services(
            candidate_state,
            node_id,
        )

        if new_set == old_set:
            continue

        # 只对“新写入”的服务收取按大小更新成本。
        added_services = new_set - old_set

        added_size_mb = sum(
            service_sizes_mb[service_id]
            for service_id in added_services
        )

        total += config.CACHE_FIXED_SWITCH_COST
        total += (
            config.CACHE_SIZE_COST_COEF
            * added_size_mb
        )

    return total


# =============================================================================
# 配置评价
# =============================================================================

def evaluate_candidate(
    candidate_state: CacheState,
    old_physical_state: CacheState,
    edge_nodes: list[dict],
    service_sizes_mb: dict[int, float],
    evaluate_p3_cost: ConfigEvaluator,
) -> tuple[float, float, float]:
    """
    评价一个候选缓存配置。

    返回：
        p3_cost
        switch_cost
        total_score

    total_score 越小越好。
    """
    p3_cost = float(
        evaluate_p3_cost(candidate_state)
    )

    switch_cost = cache_switch_cost(
        candidate_state=candidate_state,
        old_physical_state=old_physical_state,
        edge_nodes=edge_nodes,
        service_sizes_mb=service_sizes_mb,
    )

    total_score = p3_cost + switch_cost

    return p3_cost, switch_cost, total_score


# =============================================================================
# P1 主算法
# =============================================================================

def optimize_p1(
    old_physical_state: CacheState,
    edge_nodes: list[dict],
    evaluate_p3_cost: ConfigEvaluator,
    proxy_order_by_node: dict[str, list[int]] | None = None,
) -> P1Result:
    """
    执行 P1 节点级 best-improvement 局部搜索。

    参数
    ----
    old_physical_state:
        时隙开始时真实存在的物理缓存状态 C(t-1)。

    edge_nodes:
        当前参与缓存优化的动态边缘节点。

    evaluate_p3_cost:
        候选配置 -> 预测场景下的完整 P3 期望代价。

        外部控制器必须保证：
        同一个 t 的所有候选使用同一组随机场景 Xi_t，
        即 Common Random Numbers，避免候选间评价噪声不公平。

    proxy_order_by_node:
        可选。
        每个节点由因果预测器生成的服务代理优先级。
        若没有预测模块，可传 None。

    搜索规则
    --------
    1. 从真实旧配置开始；
    2. 逐节点生成 add/delete/swap/proxy 邻域；
    3. 每个候选都调用完整 P3；
    4. 对当前节点选择 best-improvement；
    5. 只有改善 > CFG_IMPROVEMENT_EPS 才接受；
    6. 有接受动作后继续下一轮；
    7. 一整轮没有任何接受动作则停止。

    缓存切换成本在整个搜索过程中始终相对
    old_physical_state 计算，而不是相对搜索迭代状态。
    """
    service_sizes_mb = cache_model.load_service_sizes_mb()

    # 先验证真实旧状态本身满足容量约束。
    cache_model.validate_cache_state(
        old_physical_state,
        edge_nodes,
        service_sizes_mb,
    )

    current_state = copy_cache_state(
        old_physical_state
    )

    current_p3, current_switch, current_score = (
        evaluate_candidate(
            candidate_state=current_state,
            old_physical_state=old_physical_state,
            edge_nodes=edge_nodes,
            service_sizes_mb=service_sizes_mb,
            evaluate_p3_cost=evaluate_p3_cost,
        )
    )

    steps: list[P1Step] = []

    iteration = 0

    while True:
        iteration += 1
        accepted_in_pass = False

        # --------------------------------------------------------------
        # “节点级 best-improvement”：
        # 对每个节点完整评价其邻域，找到该节点的最佳严格改善。
        # --------------------------------------------------------------
        for node in edge_nodes:
            node_id = str(node["node_id"])

            proxy_order = None

            if proxy_order_by_node is not None:
                proxy_order = proxy_order_by_node.get(
                    node_id
                )

            neighbors = generate_node_neighbors(
                cache_state=current_state,
                node=node,
                service_sizes_mb=service_sizes_mb,
                proxy_order=proxy_order,
            )

            best_candidate_state = None
            best_move = None
            best_p3 = None
            best_switch = None
            best_score = current_score

            for candidate_state, move in neighbors:
                p3_cost, switch_cost, total_score = (
                    evaluate_candidate(
                        candidate_state=candidate_state,
                        old_physical_state=old_physical_state,
                        edge_nodes=edge_nodes,
                        service_sizes_mb=service_sizes_mb,
                        evaluate_p3_cost=evaluate_p3_cost,
                    )
                )

                if (
                    total_score
                    < best_score
                    - config.CFG_IMPROVEMENT_EPS
                ):
                    best_candidate_state = candidate_state
                    best_move = move
                    best_p3 = p3_cost
                    best_switch = switch_cost
                    best_score = total_score

            # 当前节点存在严格改善，立即接受其 best-improvement。
            if best_candidate_state is not None:
                improvement = (
                    current_score - best_score
                )

                current_state = best_candidate_state
                current_p3 = float(best_p3)
                current_switch = float(best_switch)
                current_score = float(best_score)

                steps.append(
                    P1Step(
                        iteration=iteration,
                        node_id=node_id,
                        move_type=best_move.move_type,
                        removed_services=best_move.removed_services,
                        added_services=best_move.added_services,
                        p3_cost=current_p3,
                        switch_cost=current_switch,
                        total_score=current_score,
                        improvement=improvement,
                    )
                )

                accepted_in_pass = True

        # 一整轮没有节点能产生严格改善，局部搜索结束。
        if not accepted_in_pass:
            break

    changed = (
        state_signature(current_state)
        != state_signature(old_physical_state)
    )

    return P1Result(
        cache_state=current_state,
        p3_cost=current_p3,
        switch_cost=current_switch,
        total_score=current_score,
        changed=changed,
        steps=steps,
    )
