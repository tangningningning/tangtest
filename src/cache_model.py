# -*- coding: utf-8 -*-
"""
缓存状态管理。

当前职责：
1. 读取服务程序大小 s_f；
2. 管理不同边缘节点的物理缓存容量；
3. 生成统一、可复现、因果合法的初始缓存状态；
4. 检查缓存容量约束；
5. 保存初始缓存状态。

当前正式接入：
    SV / UAV

RSU 的缓存容量参数尚未冻结，因此暂不在这里凭空设值。
"""

from __future__ import annotations

import csv
import random

import config


def load_service_sizes_mb() -> dict[int, float]:
    """
    读取服务目录中的程序大小 s_f。

    返回：
        {service_id: program_size_MB}
    """
    sizes = {}

    with config.SERVICE_CATALOG_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        for row in csv.DictReader(f):
            service_id = int(
                row["service_id"]
            )
            sizes[service_id] = float(
                row["program_size_MB"]
            )

    if len(sizes) != config.NUM_SERVICES:
        raise RuntimeError(
            f"服务数不一致：配置为 "
            f"{config.NUM_SERVICES}，"
            f"目录中读取到 {len(sizes)}"
        )

    return sizes


def sv_cache_capacity_mb(
    service_sizes_mb: dict[int, float],
) -> float:
    """
    SV 当前 pilot 缓存容量：
        C_s = ratio * sum_f s_f
    """
    return (
        config.SV_CACHE_CAPACITY_RATIO
        * sum(service_sizes_mb.values())
    )


def uav_cache_capacity_mb(
    service_sizes_mb: dict[int, float],
) -> float:
    """
    UAV 当前 pilot 缓存容量：
        C_u = ratio * sum_f s_f
    """
    return (
        config.UAV_CACHE_CAPACITY_RATIO
        * sum(service_sizes_mb.values())
    )


def node_cache_capacity_mb(
    node_type: str,
    service_sizes_mb: dict[int, float],
) -> float:
    """
    P1 / cache_model 共用的节点容量接口。

    当前：
        SV  -> SV_CACHE_CAPACITY_RATIO
        UAV -> UAV_CACHE_CAPACITY_RATIO

    RSU 参数未冻结时明确报错，不伪造容量。
    """
    node_type = node_type.upper()

    if node_type == "SV":
        return sv_cache_capacity_mb(
            service_sizes_mb
        )

    if node_type == "UAV":
        return uav_cache_capacity_mb(
            service_sizes_mb
        )

    if node_type == "RSU":
        raise NotImplementedError(
            "RSU 缓存容量尚未正式配置。"
        )

    raise ValueError(
        f"未知节点类型: {node_type}"
    )


def build_initial_cache_state(
    edge_nodes: list[dict],
) -> dict[tuple[str, int], bool]:
    """
    构造统一初始缓存状态。

    重要：
    - 不读取未来真实 popularity；
    - 每个节点使用固定 seed 下的独立随机服务顺序；
    - 按随机顺序逐个放入；
    - 严格满足该节点自己的物理缓存容量。

    该函数现在同时支持 SV 和 UAV。
    """
    service_sizes = (
        load_service_sizes_mb()
    )

    cache_state = {}

    for node_index, node in enumerate(
        edge_nodes
    ):
        node_id = str(
            node["node_id"]
        )

        node_type = (
            node["node_type"].upper()
        )

        capacity_mb = (
            node_cache_capacity_mb(
                node_type,
                service_sizes,
            )
        )

        rng = random.Random(
            config.CACHE_INIT_SEED
            + node_index
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
            cache_state[
                (node_id, service_id)
            ] = (
                service_id in cached
            )

    validate_cache_state(
        cache_state,
        edge_nodes,
        service_sizes,
    )

    return cache_state


def validate_cache_state(
    cache_state:
        dict[tuple[str, int], bool],
    edge_nodes: list[dict],
    service_sizes_mb:
        dict[int, float]
        | None = None,
) -> None:
    """
    对每个节点检查：

        sum_f s_f c_{m,f} <= C_m
    """
    if service_sizes_mb is None:
        service_sizes_mb = (
            load_service_sizes_mb()
        )

    for node in edge_nodes:
        node_id = str(
            node["node_id"]
        )

        node_type = (
            node["node_type"].upper()
        )

        capacity_mb = (
            node_cache_capacity_mb(
                node_type,
                service_sizes_mb,
            )
        )

        used_mb = sum(
            size_mb
            for service_id, size_mb
            in service_sizes_mb.items()
            if cache_state.get(
                (node_id, service_id),
                False,
            )
        )

        if (
            used_mb
            > capacity_mb + 1e-9
        ):
            raise ValueError(
                f"节点 {node_id} "
                f"({node_type}) 缓存超容量："
                f"{used_mb:.6f} > "
                f"{capacity_mb:.6f} MB"
            )


def get_cached_services(
    cache_state:
        dict[tuple[str, int], bool],
    node_id: str,
) -> list[int]:
    """返回指定节点当前缓存的服务 ID。"""
    node_id = str(node_id)

    return sorted(
        service_id
        for service_id in range(
            config.NUM_SERVICES
        )
        if cache_state.get(
            (node_id, service_id),
            False,
        )
    )


def cache_usage_mb(
    cache_state:
        dict[tuple[str, int], bool],
    node_id: str,
    service_sizes_mb:
        dict[int, float]
        | None = None,
) -> float:
    """计算某节点当前缓存占用容量。"""
    if service_sizes_mb is None:
        service_sizes_mb = (
            load_service_sizes_mb()
        )

    return sum(
        service_sizes_mb[
            service_id
        ]
        for service_id
        in get_cached_services(
            cache_state,
            node_id,
        )
    )


def save_cache_state(
    cache_state:
        dict[tuple[str, int], bool],
    edge_nodes: list[dict],
) -> None:
    """
    保存统一初始缓存状态。

    每个节点保存自己的物理缓存容量。
    """
    service_sizes = (
        load_service_sizes_mb()
    )

    output_file = (
        config.INITIAL_CACHE_STATE_FILE
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "node_id",
        "node_type",
        "service_id",
        "cached",
        "service_size_MB",
        "cache_capacity_MB",
    ]

    with output_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for node in edge_nodes:
            node_id = str(
                node["node_id"]
            )

            node_type = (
                node["node_type"].upper()
            )

            capacity_mb = (
                node_cache_capacity_mb(
                    node_type,
                    service_sizes,
                )
            )

            for service_id in sorted(
                service_sizes
            ):
                writer.writerow(
                    {
                        "node_id":
                            node_id,
                        "node_type":
                            node_type,
                        "service_id":
                            service_id,
                        "cached":
                            int(
                                cache_state.get(
                                    (
                                        node_id,
                                        service_id,
                                    ),
                                    False,
                                )
                            ),
                        "service_size_MB":
                            f"{service_sizes[service_id]:.6f}",
                        "cache_capacity_MB":
                            f"{capacity_mb:.6f}",
                    }
                )
