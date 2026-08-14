# -*- coding: utf-8 -*-
"""
P2：UAV 水平位置优化核心。

当前只实现 P2 的“搜索逻辑”，不在本模块中凭空指定：
1. UAV 数量；
2. UAV 初始位置；
3. UAV 信道参数；
4. RSU/UAV 部署范围。

这些内容由外部场景配置和 P3 评价器传入。

已确定的模型语义：
1. UAV 高度固定，P2 只优化水平位置 q_u(t)；
2. UAV 每时隙最大移动距离：
       ||q_u(t)-q_u(t-1)|| <= V_u^max * Delta_t
3. P2 使用有限候选位置，而不是对非光滑价值函数做连续梯度优化；
4. 候选包括：
       stay
       demand hotspots
       clustered hotspots
       intermediate points
5. 每个候选最终必须调用完整 P3 评价；
6. 移动成本始终相对“真实旧物理位置”计算；
7. 只有严格改善才接受；
8. P2 内部搜索中间位置不等于真实物理移动，
   真正物理位置只从 Q_old 一次切换到最终 Q_new。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import config


# =============================================================================
# 类型
# =============================================================================

Position2D = tuple[float, float]

# 输入完整 UAV 水平位置状态，返回同一组预测场景下的 P3 期望代价。
UAVConfigEvaluator = Callable[
    [dict[str, Position2D]],
    float,
]


@dataclass
class P2Move:
    """一次 UAV 候选位置动作。"""

    uav_id: str
    candidate_type: str
    old_position: Position2D
    new_position: Position2D


@dataclass
class P2Step:
    """记录一次被接受的严格改善。"""

    iteration: int
    uav_id: str
    candidate_type: str
    old_position: Position2D
    new_position: Position2D
    p3_cost: float
    move_cost: float
    total_score: float
    improvement: float


@dataclass
class P2Result:
    """P2 最终输出。"""

    uav_positions: dict[str, Position2D]
    p3_cost: float
    move_cost: float
    total_score: float
    changed: bool
    steps: list[P2Step]


# =============================================================================
# 基础操作
# =============================================================================

def copy_positions(
    positions: dict[str, Position2D],
) -> dict[str, Position2D]:
    """复制 UAV 位置状态。"""
    return {
        str(uav_id): (
            float(position[0]),
            float(position[1]),
        )
        for uav_id, position in positions.items()
    }


def distance(
    a: Position2D,
    b: Position2D,
) -> float:
    """二维欧氏距离。"""
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1],
    )


def same_position(
    a: Position2D,
    b: Position2D,
) -> bool:
    """
    判断两个位置是否在数值容差内相同。
    """
    return (
        distance(a, b)
        <= config.UAV_MOVE_EPS_M
    )


def position_signature(
    positions: dict[str, Position2D],
) -> tuple:
    """
    生成不可变签名，用于候选去重或 P3 评价缓存。
    """
    return tuple(
        sorted(
            (
                str(uav_id),
                round(float(pos[0]), 9),
                round(float(pos[1]), 9),
            )
            for uav_id, pos
            in positions.items()
        )
    )


# =============================================================================
# UAV 移动可行性
# =============================================================================

def max_move_distance_m() -> float:
    """
    单个基础时隙内允许的最大水平移动距离：
        V_u^max * Delta_t
    """
    return (
        config.UAV_MAX_SPEED_MPS
        * config.DELTA_T
    )


def movement_feasible(
    old_physical_position: Position2D,
    candidate_position: Position2D,
) -> bool:
    """
    检查候选位置是否满足单时隙移动速度限制。
    """
    return (
        distance(
            old_physical_position,
            candidate_position,
        )
        <= max_move_distance_m()
        + 1e-12
    )


# =============================================================================
# 候选位置生成
# =============================================================================

def project_to_move_disk(
    old_position: Position2D,
    target_position: Position2D,
) -> Position2D:
    """
    若目标热点超过当前时隙最大可移动距离，
    则沿 old -> target 方向投影到可达圆边界。

    这样“热点候选”始终满足 UAV 速度约束。
    """
    d = distance(
        old_position,
        target_position,
    )

    max_d = max_move_distance_m()

    if d <= max_d + 1e-12:
        return target_position

    if d <= 1e-12:
        return old_position

    ratio = max_d / d

    return (
        old_position[0]
        + ratio
        * (
            target_position[0]
            - old_position[0]
        ),
        old_position[1]
        + ratio
        * (
            target_position[1]
            - old_position[1]
        ),
    )


def midpoint(
    a: Position2D,
    b: Position2D,
) -> Position2D:
    """两个位置的中点。"""
    return (
        0.5 * (a[0] + b[0]),
        0.5 * (a[1] + b[1]),
    )


def generate_uav_candidates(
    uav_id: str,
    old_physical_position: Position2D,
    current_search_position: Position2D,
    demand_hotspots: list[Position2D] | None = None,
    clustered_hotspots: list[Position2D] | None = None,
    extra_candidates: list[Position2D] | None = None,
) -> list[tuple[Position2D, str]]:
    """
    生成一个 UAV 的有限候选位置集合。

    候选来源：
    1. stay：
       保持当前搜索位置；
    2. demand_hotspot：
       预测需求热点；
    3. cluster_hotspot：
       聚类热点；
    4. intermediate：
       真实旧位置与热点之间的中点；
    5. extra：
       外部模块按需要补充的位置。

    所有非 stay 目标都会相对“真实旧物理位置”投影到本时隙可达范围。
    这是因为速度约束必须相对 q_u(t-1)，不能相对 P2 搜索过程
    中的某个中间候选位置。
    """
    raw: list[
        tuple[Position2D, str]
    ] = []

    # 当前配置本身就是 stay 候选。
    raw.append(
        (
            current_search_position,
            "stay",
        )
    )

    demand_hotspots = (
        demand_hotspots or []
    )

    clustered_hotspots = (
        clustered_hotspots or []
    )

    extra_candidates = (
        extra_candidates or []
    )

    for pos in demand_hotspots:
        raw.append(
            (
                pos,
                "demand_hotspot",
            )
        )

        raw.append(
            (
                midpoint(
                    old_physical_position,
                    pos,
                ),
                "intermediate",
            )
        )

    for pos in clustered_hotspots:
        raw.append(
            (
                pos,
                "cluster_hotspot",
            )
        )

        raw.append(
            (
                midpoint(
                    old_physical_position,
                    pos,
                ),
                "intermediate",
            )
        )

    for pos in extra_candidates:
        raw.append(
            (
                pos,
                "extra",
            )
        )

    candidates = []
    seen = set()

    for target, candidate_type in raw:
        target = (
            float(target[0]),
            float(target[1]),
        )

        # stay 保留当前搜索位置；
        # 其他候选都必须相对真实旧位置满足速度限制。
        if candidate_type == "stay":
            candidate = target
        else:
            candidate = project_to_move_disk(
                old_physical_position,
                target,
            )

        key = (
            round(candidate[0], 9),
            round(candidate[1], 9),
        )

        if key in seen:
            continue

        if not movement_feasible(
            old_physical_position,
            candidate,
        ):
            continue

        seen.add(key)

        candidates.append(
            (
                candidate,
                candidate_type,
            )
        )

    return candidates


# =============================================================================
# UAV 移动成本
# =============================================================================

def uav_move_cost(
    candidate_positions:
        dict[str, Position2D],
    old_physical_positions:
        dict[str, Position2D],
) -> float:
    """
    计算 UAV 配置切换成本。

    当前 config 中的统一形式：
        C_u^q
        = z_u^q * UAV_FIXED_SWITCH_COST
          + UAV_MOVE_COST_COEF
            * ||q_u - q_u^-||^2

    这里把 eta_q 与物理移动能耗系数统一吸收到
    UAV_MOVE_COST_COEF 中。

    重要：
        始终相对真实旧位置 q_u(t-1) 计算，
        不相对搜索中间位置计算。
    """
    total = 0.0

    for uav_id, old_pos in (
        old_physical_positions.items()
    ):
        new_pos = candidate_positions[
            uav_id
        ]

        d = distance(
            old_pos,
            new_pos,
        )

        if (
            d
            <= config.UAV_MOVE_EPS_M
        ):
            continue

        total += (
            config.UAV_FIXED_SWITCH_COST
        )

        total += (
            config.UAV_MOVE_COST_COEF
            * d * d
        )

    return total


# =============================================================================
# 配置评价
# =============================================================================

def evaluate_candidate(
    candidate_positions:
        dict[str, Position2D],
    old_physical_positions:
        dict[str, Position2D],
    evaluate_p3_cost:
        UAVConfigEvaluator,
) -> tuple[float, float, float]:
    """
    返回：
        p3_cost
        move_cost
        total_score

    当前为代价最小化：
        score(Q)
        = predicted_P3_cost(Q)
          + UAV_move_cost(Q, Q_old)
    """
    p3_cost = float(
        evaluate_p3_cost(
            candidate_positions
        )
    )

    move_cost = uav_move_cost(
        candidate_positions=
            candidate_positions,
        old_physical_positions=
            old_physical_positions,
    )

    total_score = (
        p3_cost + move_cost
    )

    return (
        p3_cost,
        move_cost,
        total_score,
    )


# =============================================================================
# P2 主算法
# =============================================================================

def optimize_p2(
    old_physical_positions:
        dict[str, Position2D],
    evaluate_p3_cost:
        UAVConfigEvaluator,
    demand_hotspots_by_uav:
        dict[str, list[Position2D]]
        | None = None,
    clustered_hotspots_by_uav:
        dict[str, list[Position2D]]
        | None = None,
    extra_candidates_by_uav:
        dict[str, list[Position2D]]
        | None = None,
) -> P2Result:
    """
    节点级 best-improvement P2。

    参数
    ----
    old_physical_positions:
        时隙开始时真实 UAV 水平位置 Q(t-1)。

    evaluate_p3_cost:
        给定完整 UAV 配置，返回同一组预测场景 Xi_t
        下的完整 P3 期望代价。

    demand_hotspots_by_uav:
        由预测需求生成的候选热点。

    clustered_hotspots_by_uav:
        由聚类得到的候选热点。

    extra_candidates_by_uav:
        可选额外候选。

    搜索规则
    --------
    1. 从真实旧位置 Q(t-1) 开始；
    2. 逐 UAV 生成有限候选位置；
    3. 每个候选调用完整 P3；
    4. 当前 UAV 选择 best-improvement；
    5. 只有改善 > CFG_IMPROVEMENT_EPS 才接受；
    6. 一整轮没有严格改善则停止。

    注意：
        搜索中间位置只是算法内部状态；
        真正物理 UAV 只从 Q(t-1) 一次移动到最终 Q(t)。
    """
    if not old_physical_positions:
        raise ValueError(
            "P2 至少需要一个 UAV。"
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

    current_positions = copy_positions(
        old_physical_positions
    )

    current_p3, current_move, current_score = (
        evaluate_candidate(
            candidate_positions=
                current_positions,
            old_physical_positions=
                old_physical_positions,
            evaluate_p3_cost=
                evaluate_p3_cost,
        )
    )

    steps: list[P2Step] = []

    iteration = 0

    while True:
        iteration += 1
        accepted_in_pass = False

        # --------------------------------------------------------------
        # 节点级 best-improvement：
        # 对每个 UAV 的完整有限邻域做一次 P3 评价。
        # --------------------------------------------------------------
        for uav_id in sorted(
            current_positions
        ):
            current_search_position = (
                current_positions[uav_id]
            )

            candidates = (
                generate_uav_candidates(
                    uav_id=uav_id,
                    old_physical_position=
                        old_physical_positions[
                            uav_id
                        ],
                    current_search_position=
                        current_search_position,
                    demand_hotspots=
                        demand_hotspots_by_uav.get(
                            uav_id,
                            [],
                        ),
                    clustered_hotspots=
                        clustered_hotspots_by_uav.get(
                            uav_id,
                            [],
                        ),
                    extra_candidates=
                        extra_candidates_by_uav.get(
                            uav_id,
                            [],
                        ),
                )
            )

            best_positions = None
            best_type = None
            best_p3 = None
            best_move_cost = None
            best_score = current_score

            for (
                candidate_position,
                candidate_type,
            ) in candidates:
                # 当前搜索位置本身就是 keep/stay，
                # 不需要重复评价。
                if same_position(
                    candidate_position,
                    current_search_position,
                ):
                    continue

                candidate_positions = (
                    copy_positions(
                        current_positions
                    )
                )

                candidate_positions[
                    uav_id
                ] = candidate_position

                (
                    p3_cost,
                    move_cost,
                    total_score,
                ) = evaluate_candidate(
                    candidate_positions=
                        candidate_positions,
                    old_physical_positions=
                        old_physical_positions,
                    evaluate_p3_cost=
                        evaluate_p3_cost,
                )

                if (
                    total_score
                    < best_score
                    - config.CFG_IMPROVEMENT_EPS
                ):
                    best_positions = (
                        candidate_positions
                    )

                    best_type = (
                        candidate_type
                    )

                    best_p3 = p3_cost
                    best_move_cost = (
                        move_cost
                    )
                    best_score = total_score

            # 当前 UAV 存在严格改善。
            if best_positions is not None:
                previous_position = (
                    current_positions[
                        uav_id
                    ]
                )

                improvement = (
                    current_score
                    - best_score
                )

                current_positions = (
                    best_positions
                )

                current_p3 = float(
                    best_p3
                )

                current_move = float(
                    best_move_cost
                )

                current_score = float(
                    best_score
                )

                steps.append(
                    P2Step(
                        iteration=iteration,
                        uav_id=uav_id,
                        candidate_type=
                            str(best_type),
                        old_position=
                            previous_position,
                        new_position=
                            current_positions[
                                uav_id
                            ],
                        p3_cost=current_p3,
                        move_cost=current_move,
                        total_score=
                            current_score,
                        improvement=
                            improvement,
                    )
                )

                accepted_in_pass = True

        if not accepted_in_pass:
            break

    changed = any(
        not same_position(
            old_physical_positions[
                uav_id
            ],
            current_positions[
                uav_id
            ],
        )
        for uav_id in current_positions
    )

    return P2Result(
        uav_positions=
            current_positions,
        p3_cost=current_p3,
        move_cost=current_move,
        total_score=current_score,
        changed=changed,
        steps=steps,
    )
