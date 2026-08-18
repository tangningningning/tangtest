# -*- coding: utf-8 -*-
"""
执行节点竞争力诊断。

目的：
在不修改任何参数的前提下，分析当前：

    SV / RSU / UAV / Cloud

之间为什么会出现 Cloud 占优。

本脚本不是正式实验，也不改变配置。
它只读取当前 config.py、任务、轨迹和节点状态，输出诊断统计。

核心思想：

对边缘节点 m，若给任务满 CPU 和满带宽：

    w = 1
    b = 1

则其当前 P3 归一化代价下界为：

    J_edge_lb = A + B

其中：

    A = omega_T * L / (Tmax * F_m)
    B = D / (R_bar * Tmax)

真实共享资源下：

    J_edge >= J_edge_lb

因此：

1. 如果某任务连最优边缘下界都高于 Cloud：
       min_m J_edge_lb >= J_cloud
   那么当前 Cloud 对该任务具有结构性优势。

2. 如果：
       min_m J_edge_lb < J_cloud
   则边缘至少在物理层面存在竞争可能；
   后续是否真的被选中还取决于缓存和共享资源竞争。

同时区分：

A. all-service：
   假设所有边缘节点都缓存服务，
   用来检查通信/计算参数本身。

B. initial-cache：
   使用当前真实初始随机缓存，
   用来检查缓存门之后的可用性。

另外计算：
在保持当前 R_c、F_c 不变时，
每个任务需要多大的 Cloud psi，
才会让最优边缘满资源下界刚好不劣于 Cloud。

注意：
这只是诊断 break-even，不会自动修改 CLOUD_PSI。
"""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter
from pathlib import Path


# data/ 下的脚本直接运行时，显式把项目 src/ 加入模块搜索路径。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import cache_model
import config
import p3_solver
import uav_state


OUTPUT_FILE = (
    config.RESULTS_DIR
    / "execution_balance_audit.csv"
)


def make_all_service_cache(
    edge_nodes: list[dict],
) -> dict[tuple[str, int], bool]:
    """构造仅用于诊断的全服务缓存。"""
    return {
        (
            str(node["node_id"]),
            service_id,
        ):
            True
        for node in edge_nodes
        for service_id in range(
            config.NUM_SERVICES
        )
    }


def edge_lower_bound(
    candidate: dict,
) -> float:
    """
    w=b=1 时的边缘任务代价下界。

        J = A/w + B/b
        => J_lb = A + B
    """
    return (
        float(candidate["A"])
        + float(candidate["Bcoef"])
    )


def cloud_cost_with_psi(
    task: dict,
    psi: float,
) -> float:
    """
    保持当前 Cloud rate / CPU 不变，
    仅代入给定 psi 计算 Cloud 归一化代价。
    """
    tx_delay = (
        float(psi)
        * task["D"]
        / config.CLOUD_RATE_BPS
    )

    compute_delay = (
        task["L"]
        / config.CLOUD_CPU_HZ
    )

    energy_term = (
        task["D"]
        / (
            config.CLOUD_RATE_BPS
            * task["Tmax"]
        )
    )

    return (
        config.OMEGA_T
        * (
            tx_delay
            + compute_delay
        )
        / task["Tmax"]
        + config.OMEGA_E
        * energy_term
    )


def break_even_cloud_psi(
    task: dict,
    edge_cost_lb: float,
) -> float | None:
    """
    解：

        J_cloud(psi)
        = edge_cost_lb

    得到当前 R_c、F_c 下的 Cloud psi 临界值。

    若返回：
        psi_star

    则大致有：
        psi < psi_star : Cloud 更便宜
        psi > psi_star : 该边缘下界更便宜

    若系数异常则返回 None。
    """
    denom = (
        config.OMEGA_T
        * task["D"]
        / (
            config.CLOUD_RATE_BPS
            * task["Tmax"]
        )
    )

    if denom <= 0.0:
        return None

    fixed_part = (
        config.OMEGA_T
        * task["L"]
        / (
            config.CLOUD_CPU_HZ
            * task["Tmax"]
        )
        + config.OMEGA_E
        * task["D"]
        / (
            config.CLOUD_RATE_BPS
            * task["Tmax"]
        )
    )

    return (
        edge_cost_lb
        - fixed_part
    ) / denom


def percentile(
    values: list[float],
    q: float,
) -> float | None:
    """简单线性分位数，不依赖额外库。"""
    if not values:
        return None

    xs = sorted(values)

    if len(xs) == 1:
        return xs[0]

    pos = (
        (len(xs) - 1)
        * q
    )

    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return xs[lo]

    weight = pos - lo

    return (
        xs[lo]
        * (1.0 - weight)
        + xs[hi]
        * weight
    )


def main():
    positions = (
        p3_solver.load_positions()
    )

    tasks = (
        p3_solver.load_tasks()
    )

    tv_ids = (
        p3_solver
        .load_ranked_vehicle_ids(
            "TV",
            config.NUM_TV,
        )
    )

    # 与 rolling_controller 的 t=0 初始化保持一致。
    uav_positions = (
        uav_state
        .initialize_uav_positions(
            tv_ids=tv_ids,
            actual_positions=positions,
            num_uav=config.NUM_UAV,
            initial_slot=0,
        )
    )

    edge_nodes = (
        p3_solver.build_edge_nodes(
            uav_positions
        )
    )

    node_type_counts = Counter(
        str(node["node_type"]).upper()
        for node in edge_nodes
    )

    initial_cache = (
        cache_model
        .build_initial_cache_state(
            edge_nodes
        )
    )

    all_cache = (
        make_all_service_cache(
            edge_nodes
        )
    )

    rows = []

    structural_edge_win = 0
    cached_edge_win = 0

    structural_any_edge = 0
    cached_any_edge = 0

    best_type_counts = Counter()

    psi_break_evens = []

    for task in tasks:
        cloud = (
            p3_solver.cloud_metrics(
                task
            )
        )

        cloud_cost = float(
            cloud["cost"]
        )

        all_candidates = []
        cached_candidates = []

        for node in edge_nodes:
            c_all = (
                p3_solver
                .make_edge_candidate(
                    task=task,
                    node=node,
                    positions=positions,
                    cache_state=all_cache,
                )
            )

            if c_all is not None:
                all_candidates.append(
                    (
                        node,
                        c_all,
                    )
                )

            c_cached = (
                p3_solver
                .make_edge_candidate(
                    task=task,
                    node=node,
                    positions=positions,
                    cache_state=
                        initial_cache,
                )
            )

            if c_cached is not None:
                cached_candidates.append(
                    (
                        node,
                        c_cached,
                    )
                )

        if all_candidates:
            structural_any_edge += 1

        if cached_candidates:
            cached_any_edge += 1

        best_all = None

        if all_candidates:
            best_all = min(
                all_candidates,
                key=lambda item:
                    edge_lower_bound(
                        item[1]
                    ),
            )

            best_node, best_candidate = (
                best_all
            )

            best_lb = (
                edge_lower_bound(
                    best_candidate
                )
            )

            best_type = str(
                best_node["node_type"]
            ).upper()

            best_type_counts[
                best_type
            ] += 1

            if (
                best_lb
                < cloud_cost
                - 1e-12
            ):
                structural_edge_win += 1

            psi_star = (
                break_even_cloud_psi(
                    task,
                    best_lb,
                )
            )

            if (
                psi_star is not None
                and math.isfinite(
                    psi_star
                )
            ):
                psi_break_evens.append(
                    psi_star
                )

        else:
            best_lb = None
            best_type = ""
            psi_star = None

        if cached_candidates:
            best_cached_lb = min(
                edge_lower_bound(
                    item[1]
                )
                for item
                in cached_candidates
            )

            if (
                best_cached_lb
                < cloud_cost
                - 1e-12
            ):
                cached_edge_win += 1
        else:
            best_cached_lb = None

        rows.append(
            {
                "task_id":
                    task["task_id"],
                "slot":
                    task["slot"],
                "vehicle_id":
                    task["vehicle_id"],
                "service_id":
                    task["service_id"],
                "cloud_cost":
                    cloud_cost,
                "has_edge_all_service":
                    int(
                        bool(
                            all_candidates
                        )
                    ),
                "best_edge_type_all_service":
                    best_type,
                "best_edge_lb_all_service":
                    (
                        ""
                        if best_lb is None
                        else best_lb
                    ),
                "edge_lb_beats_cloud":
                    int(
                        best_lb is not None
                        and best_lb
                        < cloud_cost
                        - 1e-12
                    ),
                "break_even_cloud_psi":
                    (
                        ""
                        if psi_star is None
                        else psi_star
                    ),
                "has_edge_initial_cache":
                    int(
                        bool(
                            cached_candidates
                        )
                    ),
                "best_edge_lb_initial_cache":
                    (
                        ""
                        if best_cached_lb
                        is None
                        else
                        best_cached_lb
                    ),
                "cached_edge_lb_beats_cloud":
                    int(
                        best_cached_lb
                        is not None
                        and best_cached_lb
                        < cloud_cost
                        - 1e-12
                    ),
            }
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "task_id",
        "slot",
        "vehicle_id",
        "service_id",
        "cloud_cost",
        "has_edge_all_service",
        "best_edge_type_all_service",
        "best_edge_lb_all_service",
        "edge_lb_beats_cloud",
        "break_even_cloud_psi",
        "has_edge_initial_cache",
        "best_edge_lb_initial_cache",
        "cached_edge_lb_beats_cloud",
    ]

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            formatted = dict(row)

            for key in (
                "cloud_cost",
                "best_edge_lb_all_service",
                "break_even_cloud_psi",
                "best_edge_lb_initial_cache",
            ):
                value = formatted[key]

                if value != "":
                    formatted[key] = (
                        f"{float(value):.12f}"
                    )

            writer.writerow(
                formatted
            )

    n = len(tasks)

    print("=" * 76)
    print("执行节点竞争力诊断完成")
    print("=" * 76)

    print(
        f"任务总数                    : "
        f"{n}"
    )

    print(
        "当前边缘节点                : "
        + ", ".join(
            f"{k}={v}"
            for k, v
            in sorted(
                node_type_counts.items()
            )
        )
    )

    print()
    print("当前 Cloud 参数：")
    print(
        f"  R_c                       : "
        f"{config.CLOUD_RATE_BPS/1e6:.3f} Mbps"
    )
    print(
        f"  F_c                       : "
        f"{config.CLOUD_CPU_HZ/1e9:.3f} GHz"
    )
    print(
        f"  psi_c                     : "
        f"{config.CLOUD_PSI:.6f}"
    )

    print()
    print("A. 不考虑缓存门：物理/计算竞争力")
    print(
        f"  至少有一个可行边缘节点    : "
        f"{structural_any_edge}/{n} "
        f"({100*structural_any_edge/n:.2f}%)"
    )
    print(
        f"  最优边缘满资源下界 "
        f"< Cloud : "
        f"{structural_edge_win}/{n} "
        f"({100*structural_edge_win/n:.2f}%)"
    )

    print(
        "  最优边缘节点类型次数      : "
        + (
            ", ".join(
                f"{k}={v}"
                for k, v
                in sorted(
                    best_type_counts.items()
                )
            )
            if best_type_counts
            else "无"
        )
    )

    print()
    print("B. 加入当前初始缓存门")
    print(
        f"  至少有一个可行缓存边缘节点: "
        f"{cached_any_edge}/{n} "
        f"({100*cached_any_edge/n:.2f}%)"
    )
    print(
        f"  最优缓存边缘满资源下界 "
        f"< Cloud : "
        f"{cached_edge_win}/{n} "
        f"({100*cached_edge_win/n:.2f}%)"
    )

    print()
    print("C. Cloud psi break-even（保持当前 R_c/F_c）")

    for label, q in (
        ("10%", 0.10),
        ("25%", 0.25),
        ("50%", 0.50),
        ("75%", 0.75),
        ("90%", 0.90),
    ):
        value = percentile(
            psi_break_evens,
            q,
        )

        if value is None:
            print(
                f"  {label:<4} 分位               : "
                "无"
            )
        else:
            print(
                f"  {label:<4} 分位               : "
                f"{value:.6f}"
            )

    print()
    print(
        "解释：break-even psi 不是推荐值，"
        "只是当前数据下 Cloud 与最优边缘下界相等的临界点。"
    )
    print(
        "后续应根据这里的结果，再决定是否需要调整 "
        "CLOUD_PSI / CLOUD_RATE_BPS / CLOUD_CPU_HZ。"
    )

    print()
    print(
        f"逐任务结果文件              : "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
