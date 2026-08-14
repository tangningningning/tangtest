# -*- coding: utf-8 -*-
"""
UAV 状态与预测热点辅助模块。

用途：
1. 根据已观测 TV 位置构造 UAV 初始水平位置 Q^U(0)；
2. 从同一公共随机场景 Xi_t 中提取预测需求热点；
3. 为 UAV 构造初始缓存状态。

因果要求：
- Q^U(0) 只使用 slot=0 已观测 TV 位置；
- 预测热点只使用 ScenarioSet，不读取未来真实位置/任务；
- 本模块不执行 P2，只提供状态和候选点。

当前 pilot：
- UAV 缓存容量使用 config.UAV_CACHE_CAPACITY_RATIO；
- 初始 UAV 缓存为固定 seed 下的随机可行状态；
- 该初始缓存不读取未来真实流行度。
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Iterable

import cache_model
import config


Position2D = tuple[float, float]


def _distance2(
    a: Position2D,
    b: Position2D,
) -> float:
    """二维距离平方。"""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _mean_point(
    points: list[Position2D],
) -> Position2D:
    """点集质心。"""
    if not points:
        raise ValueError("空点集没有质心。")

    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def deterministic_kmeans(
    points: list[Position2D],
    k: int,
    max_iter: int = 50,
) -> list[Position2D]:
    """
    一个很小的确定性 K-means，仅用于 UAV 初始位置和热点候选。

    初始化：
    1. 第一个中心取全局质心最近的样本点；
    2. 后续中心使用最远点初始化。

    这样不依赖新的随机 seed。
    """
    if k <= 0:
        raise ValueError("k 必须大于 0。")

    if not points:
        raise ValueError("没有点可供聚类。")

    # 去除完全重复点。
    unique = list(dict.fromkeys(
        (float(x), float(y))
        for x, y in points
    ))

    k = min(k, len(unique))

    global_center = _mean_point(unique)

    first = min(
        unique,
        key=lambda p: _distance2(
            p,
            global_center,
        ),
    )

    centers = [first]

    while len(centers) < k:
        next_center = max(
            unique,
            key=lambda p: min(
                _distance2(p, c)
                for c in centers
            ),
        )

        if next_center in centers:
            break

        centers.append(next_center)

    for _ in range(max_iter):
        groups = [
            []
            for _ in centers
        ]

        for point in unique:
            index = min(
                range(len(centers)),
                key=lambda i: _distance2(
                    point,
                    centers[i],
                ),
            )
            groups[index].append(point)

        new_centers = []

        for old_center, group in zip(
            centers,
            groups,
        ):
            if group:
                new_centers.append(
                    _mean_point(group)
                )
            else:
                new_centers.append(
                    old_center
                )

        movement = max(
            math.sqrt(
                _distance2(a, b)
            )
            for a, b in zip(
                centers,
                new_centers,
            )
        )

        centers = new_centers

        if movement <= 1e-9:
            break

    return centers


def initialize_uav_positions(
    tv_ids: list[str],
    actual_positions:
        dict[tuple[int, str], Position2D],
    num_uav: int,
    initial_slot: int = 0,
) -> dict[str, Position2D]:
    """
    根据初始时隙已经观测到的 TV 位置构造 Q^U(0)。

    不使用未来位置。

    NUM_UAV=1 时：
        UAV 位于初始 TV 位置质心附近的聚类中心。

    NUM_UAV>1 时：
        使用确定性 K-means 得到多个中心。
    """
    points = []

    for tv_id in tv_ids:
        key = (
            initial_slot,
            tv_id,
        )

        if key not in actual_positions:
            raise RuntimeError(
                f"缺少 TV 初始位置: {key}"
            )

        points.append(
            actual_positions[key]
        )

    centers = deterministic_kmeans(
        points,
        num_uav,
    )

    return {
        f"uav_{i}": center
        for i, center
        in enumerate(centers)
    }


def uav_cache_capacity_mb(
    service_sizes_mb:
        dict[int, float],
) -> float:
    """
    当前 UAV pilot 缓存容量：
        C_u = ratio * sum_f s_f
    """
    return (
        config.UAV_CACHE_CAPACITY_RATIO
        * sum(service_sizes_mb.values())
    )


def extend_cache_state_for_uavs(
    base_cache_state:
        dict[tuple[str, int], bool],
    uav_positions:
        dict[str, Position2D],
) -> dict[tuple[str, int], bool]:
    """
    在已有 SV cache_state 上增加 UAV 初始缓存。

    初始 UAV 缓存：
    - 固定 seed；
    - 随机服务顺序；
    - 严格满足物理容量；
    - 不使用未来真实 popularity。
    """
    state = dict(
        base_cache_state
    )

    service_sizes = (
        cache_model.load_service_sizes_mb()
    )

    capacity_mb = (
        uav_cache_capacity_mb(
            service_sizes
        )
    )

    for index, uav_id in enumerate(
        sorted(uav_positions)
    ):
        rng = random.Random(
            config.CACHE_INIT_SEED
            + 1000
            + index
        )

        service_ids = sorted(
            service_sizes
        )

        rng.shuffle(
            service_ids
        )

        used_mb = 0.0
        cached = set()

        for service_id in service_ids:
            size_mb = (
                service_sizes[
                    service_id
                ]
            )

            if (
                used_mb + size_mb
                <= capacity_mb + 1e-12
            ):
                cached.add(
                    service_id
                )
                used_mb += size_mb

        for service_id in sorted(
            service_sizes
        ):
            state[
                (uav_id, service_id)
            ] = (
                service_id in cached
            )

    return state


def build_hotspots_from_scenarios(
    scenario_set,
    num_cluster_hotspots: int,
    top_demand_hotspots: int = 3,
) -> tuple[
    list[Position2D],
    list[Position2D],
]:
    """
    从同一个 ScenarioSet 中构造 P2 候选热点。

    demand hotspots：
        统计各预测 (slot, TV) 位置被任务命中的次数，
        取计数最高的若干位置。

    clustered hotspots：
        将所有预测任务位置做确定性 K-means。

    这两类候选都只来自 Xi_t。
    """
    point_counts = Counter()
    all_task_points = []

    for scenario in (
        scenario_set.scenarios
    ):
        for task in scenario.tasks:
            key = (
                task["slot"],
                task["vehicle_id"],
            )

            if (
                key
                not in scenario.positions
            ):
                raise RuntimeError(
                    f"场景缺少预测位置: {key}"
                )

            point = (
                scenario.positions[key]
            )

            rounded = (
                round(
                    float(point[0]),
                    6,
                ),
                round(
                    float(point[1]),
                    6,
                ),
            )

            point_counts[
                rounded
            ] += 1

            all_task_points.append(
                (
                    float(point[0]),
                    float(point[1]),
                )
            )

    if not all_task_points:
        return [], []

    demand_hotspots = [
        point
        for point, _
        in point_counts.most_common(
            top_demand_hotspots
        )
    ]

    cluster_hotspots = (
        deterministic_kmeans(
            all_task_points,
            max(
                1,
                num_cluster_hotspots,
            ),
        )
    )

    return (
        demand_hotspots,
        cluster_hotspots,
    )
