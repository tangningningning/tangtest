# -*- coding: utf-8 -*-
"""
Pjoint：缓存配置与 UAV 水平位置的联合候选搜索。

触发条件：
    只有当同一时隙中的 P1 和 P2 都没有严格改善时，
    rolling_controller 才调用本模块。

联合邻域：
    一个缓存邻域动作
        add / delete / swap / proxy
    +
    一个 UAV 有限位置动作
        demand hotspot / cluster hotspot / intermediate / extra

每一个联合候选都调用完整 P3 评价。

严格语义：
1. 所有联合候选共用同一个 ScenarioSet Xi_t；
2. 缓存切换成本始终相对真实旧缓存 C(t-1)；
3. UAV 移动成本始终相对真实旧位置 Q^U(t-1)；
4. Pjoint 搜索中的组合动作只是算法候选；
5. 真正物理状态只从旧状态一次切换到最终联合状态；
6. 只有联合改善 > CFG_IMPROVEMENT_EPS 才接受。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cache_model
import config
import p1_cache
import p2_uav
import p3_solver


CacheState = p1_cache.CacheState
Position2D = p2_uav.Position2D

JointEvaluator = Callable[
    [
        CacheState,
        dict[str, Position2D],
    ],
    float,
]


@dataclass
class PJointMove:
    """一次被选中的联合动作。"""

    cache_node_id: str
    cache_move_type: str
    removed_services: tuple[int, ...]
    added_services: tuple[int, ...]
    uav_id: str
    uav_candidate_type: str
    old_uav_position: Position2D
    new_uav_position: Position2D


@dataclass
class PJointResult:
    """Pjoint 输出。"""

    cache_state: CacheState
    uav_positions: dict[str, Position2D]
    p3_cost: float
    cache_switch_cost: float
    uav_move_cost: float
    total_score: float
    changed: bool
    move: PJointMove | None
    evaluated_joint_candidates: int


# =============================================================================
# 固定 Xi_t 的联合 P3 评价器
# =============================================================================

class ScenarioJointP3Evaluator:
    """
    对 (C, Q^U) 联合配置进行 P3 场景评价。

    同一个对象固定一个 ScenarioSet，
    因此所有联合候选天然复用相同的 Xi_t。
    """

    def __init__(
        self,
        scenario_set,
    ):
        self.scenario_set = scenario_set

        self._cost_cache: dict[
            tuple,
            float,
        ] = {}

        self.num_evaluations = 0
        self.num_cache_hits = 0

    def __call__(
        self,
        cache_state: CacheState,
        uav_positions:
            dict[str, Position2D],
    ) -> float:
        """
        返回：

            (1/S) sum_s sum_tau Phi_tau(C,Q;xi_s)

        当前 Phi_tau 为预测任务归一化 P3 代价之和。
        """
        signature = (
            p1_cache.state_signature(
                cache_state
            ),
            p2_uav.position_signature(
                uav_positions
            ),
        )

        if signature in self._cost_cache:
            self.num_cache_hits += 1
            return self._cost_cache[
                signature
            ]

        edge_nodes = (
            p3_solver.build_edge_nodes(
                uav_positions
            )
        )

        total = 0.0

        for scenario in (
            self.scenario_set.scenarios
        ):
            if not scenario.tasks:
                continue

            results, _ = (
                p3_solver.solve_episode_p3(
                    tasks=scenario.tasks,
                    edge_nodes=edge_nodes,
                    positions=
                        scenario.positions,
                    cache_state=
                        cache_state,
                )
            )

            total += sum(
                result["cost"]
                for result
                in results.values()
            )

        mean_cost = (
            total
            / len(
                self.scenario_set.scenarios
            )
        )

        self._cost_cache[
            signature
        ] = mean_cost

        self.num_evaluations += 1

        return mean_cost


# =============================================================================
# 联合配置代价
# =============================================================================

def evaluate_joint_candidate(
    candidate_cache: CacheState,
    candidate_uav_positions:
        dict[str, Position2D],
    old_physical_cache: CacheState,
    old_physical_uav_positions:
        dict[str, Position2D],
    edge_nodes: list[dict],
    service_sizes_mb:
        dict[int, float],
    evaluate_p3_cost:
        JointEvaluator,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """
    评价联合配置：

        score(C,Q)
        = V_hat_t(C,Q)
          + C_cache(C,C_old)
          + C_uav(Q,Q_old)

    返回：
        p3_cost
        cache_cost
        uav_cost
        total_score
    """
    p3_cost = float(
        evaluate_p3_cost(
            candidate_cache,
            candidate_uav_positions,
        )
    )

    cache_cost = (
        p1_cache.cache_switch_cost(
            candidate_state=
                candidate_cache,
            old_physical_state=
                old_physical_cache,
            edge_nodes=edge_nodes,
            service_sizes_mb=
                service_sizes_mb,
        )
    )

    uav_cost = (
        p2_uav.uav_move_cost(
            candidate_positions=
                candidate_uav_positions,
            old_physical_positions=
                old_physical_uav_positions,
        )
    )

    return (
        p3_cost,
        cache_cost,
        uav_cost,
        p3_cost
        + cache_cost
        + uav_cost,
    )


# =============================================================================
# Pjoint
# =============================================================================

def optimize_pjoint(
    old_physical_cache: CacheState,
    old_physical_uav_positions:
        dict[str, Position2D],
    edge_nodes: list[dict],
    evaluate_p3_cost:
        JointEvaluator,
    proxy_order_by_node:
        dict[str, list[int]]
        | None = None,
    demand_hotspots_by_uav:
        dict[str, list[Position2D]]
        | None = None,
    clustered_hotspots_by_uav:
        dict[str, list[Position2D]]
        | None = None,
    extra_candidates_by_uav:
        dict[str, list[Position2D]]
        | None = None,
) -> PJointResult:
    """
    搜索“一次缓存动作 + 一次 UAV 位置动作”的联合邻域。

    为什么不把 keep 与单独动作再次放进来：
    - Pjoint 只在 P1、P2 都已经 stall 后调用；
    - 单独缓存动作已经被 P1 完整检查；
    - 单独 UAV 动作已经被 P2 完整检查；
    - 因此 Pjoint 只需要寻找二者存在协同效应的组合。

    当前采用全联合邻域中的 global best-improvement：
    所有合法联合候选都评价一次，选总配置代价最低者。
    """
    if not old_physical_uav_positions:
        raise ValueError(
            "Pjoint 至少需要一个 UAV。"
        )

    service_sizes = (
        cache_model.load_service_sizes_mb()
    )

    cache_model.validate_cache_state(
        old_physical_cache,
        edge_nodes,
        service_sizes,
    )

    proxy_order_by_node = (
        proxy_order_by_node or {}
    )

    demand_hotspots_by_uav = (
        demand_hotspots_by_uav or {}
    )

    clustered_hotspots_by_uav = (
        clustered_hotspots_by_uav or {}
    )

    extra_candidates_by_uav = (
        extra_candidates_by_uav or {}
    )

    # ------------------------------------------------------------------
    # 当前真实旧配置作为比较基准。
    # ------------------------------------------------------------------
    (
        base_p3,
        base_cache_cost,
        base_uav_cost,
        base_score,
    ) = evaluate_joint_candidate(
        candidate_cache=
            old_physical_cache,
        candidate_uav_positions=
            old_physical_uav_positions,
        old_physical_cache=
            old_physical_cache,
        old_physical_uav_positions=
            old_physical_uav_positions,
        edge_nodes=edge_nodes,
        service_sizes_mb=
            service_sizes,
        evaluate_p3_cost=
            evaluate_p3_cost,
    )

    # ------------------------------------------------------------------
    # 先构造所有“单节点缓存变化”候选。
    # ------------------------------------------------------------------
    cache_candidates: list[
        tuple[
            CacheState,
            p1_cache.P1Move,
        ]
    ] = []

    for node in edge_nodes:
        node_id = str(
            node["node_id"]
        )

        neighbors = (
            p1_cache
            .generate_node_neighbors(
                cache_state=
                    old_physical_cache,
                node=node,
                service_sizes_mb=
                    service_sizes,
                proxy_order=
                    proxy_order_by_node.get(
                        node_id
                    ),
            )
        )

        cache_candidates.extend(
            neighbors
        )

    # ------------------------------------------------------------------
    # 再构造所有“单 UAV 非 stay 变化”候选。
    # ------------------------------------------------------------------
    uav_candidates: list[
        tuple[
            str,
            Position2D,
            str,
        ]
    ] = []

    for uav_id in sorted(
        old_physical_uav_positions
    ):
        old_position = (
            old_physical_uav_positions[
                uav_id
            ]
        )

        candidates = (
            p2_uav
            .generate_uav_candidates(
                uav_id=uav_id,
                old_physical_position=
                    old_position,
                current_search_position=
                    old_position,
                demand_hotspots=
                    demand_hotspots_by_uav
                    .get(
                        uav_id,
                        [],
                    ),
                clustered_hotspots=
                    clustered_hotspots_by_uav
                    .get(
                        uav_id,
                        [],
                    ),
                extra_candidates=
                    extra_candidates_by_uav
                    .get(
                        uav_id,
                        [],
                    ),
            )
        )

        for (
            candidate_position,
            candidate_type,
        ) in candidates:
            if p2_uav.same_position(
                candidate_position,
                old_position,
            ):
                continue

            uav_candidates.append(
                (
                    uav_id,
                    candidate_position,
                    candidate_type,
                )
            )

    # 如果任一侧没有非平凡候选，就不存在联合变化。
    if (
        not cache_candidates
        or not uav_candidates
    ):
        return PJointResult(
            cache_state=dict(
                old_physical_cache
            ),
            uav_positions=
                p2_uav.copy_positions(
                    old_physical_uav_positions
                ),
            p3_cost=base_p3,
            cache_switch_cost=
                base_cache_cost,
            uav_move_cost=
                base_uav_cost,
            total_score=base_score,
            changed=False,
            move=None,
            evaluated_joint_candidates=0,
        )

    best_cache = None
    best_positions = None
    best_move = None

    best_p3 = base_p3
    best_cache_cost = (
        base_cache_cost
    )
    best_uav_cost = (
        base_uav_cost
    )
    best_score = base_score

    evaluated = 0

    # ------------------------------------------------------------------
    # 全组合搜索：
    #     cache neighbor × UAV position candidate
    # ------------------------------------------------------------------
    for (
        candidate_cache,
        cache_move,
    ) in cache_candidates:

        for (
            uav_id,
            candidate_position,
            candidate_type,
        ) in uav_candidates:

            candidate_positions = (
                p2_uav.copy_positions(
                    old_physical_uav_positions
                )
            )

            candidate_positions[
                uav_id
            ] = candidate_position

            (
                p3_cost,
                cache_cost,
                uav_cost,
                total_score,
            ) = evaluate_joint_candidate(
                candidate_cache=
                    candidate_cache,
                candidate_uav_positions=
                    candidate_positions,
                old_physical_cache=
                    old_physical_cache,
                old_physical_uav_positions=
                    old_physical_uav_positions,
                edge_nodes=edge_nodes,
                service_sizes_mb=
                    service_sizes,
                evaluate_p3_cost=
                    evaluate_p3_cost,
            )

            evaluated += 1

            if (
                total_score
                < best_score
                - config.CFG_IMPROVEMENT_EPS
            ):
                best_cache = (
                    candidate_cache
                )

                best_positions = (
                    candidate_positions
                )

                best_p3 = p3_cost
                best_cache_cost = (
                    cache_cost
                )
                best_uav_cost = (
                    uav_cost
                )
                best_score = (
                    total_score
                )

                best_move = PJointMove(
                    cache_node_id=
                        cache_move.node_id,
                    cache_move_type=
                        cache_move.move_type,
                    removed_services=
                        cache_move
                        .removed_services,
                    added_services=
                        cache_move
                        .added_services,
                    uav_id=uav_id,
                    uav_candidate_type=
                        candidate_type,
                    old_uav_position=
                        old_physical_uav_positions[
                            uav_id
                        ],
                    new_uav_position=
                        candidate_position,
                )

    if best_cache is None:
        return PJointResult(
            cache_state=dict(
                old_physical_cache
            ),
            uav_positions=
                p2_uav.copy_positions(
                    old_physical_uav_positions
                ),
            p3_cost=base_p3,
            cache_switch_cost=
                base_cache_cost,
            uav_move_cost=
                base_uav_cost,
            total_score=base_score,
            changed=False,
            move=None,
            evaluated_joint_candidates=
                evaluated,
        )

    return PJointResult(
        cache_state=dict(
            best_cache
        ),
        uav_positions=
            p2_uav.copy_positions(
                best_positions
            ),
        p3_cost=best_p3,
        cache_switch_cost=
            best_cache_cost,
        uav_move_cost=
            best_uav_cost,
        total_score=best_score,
        changed=True,
        move=best_move,
        evaluated_joint_candidates=
            evaluated,
    )
