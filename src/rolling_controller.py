# -*- coding: utf-8 -*-
"""
P1 + P2 + Pjoint 滚动在线控制器。

当前阶段：
    P1      缓存局部搜索
    P2      UAV 水平位置局部搜索
    Pjoint  P1/P2 都 stall 后的联合候选搜索
    P3      任务卸载与资源分配
    Async   基于净收益的逐节点异步更新记录

当前边缘节点：
    SV / RSU / UAV

暂未完成：
    最终参数校准

每个时隙严格遵循：

    C(t-1), Q^U(t-1)
          ↓
        H_{t-1}
          ↓
        Xi_t
          ↓
         P1
          ↓
         P2
          ↓
    若 P1/P2 均无严格改善
          ↓
       Pjoint
          ↓
    C(t), Q^U(t)
          ↓
    实际 slot=t 任务与位置
          ↓
         P3

P1、P2、Pjoint 在同一时隙共用同一个 Xi_t。

P1/P2/Pjoint 的严格改善条件本身就是自适应更新触发条件；
这里不再增加第二层门槛，避免切换成本被重复扣除。
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import async_update
import cache_model
import config
import p1_cache
import p1_runner
import p2_runner
import p2_uav
import p3_solver
import pjoint
import scenario_generator
import uav_state


# =============================================================================
# 基础数据
# =============================================================================

def group_tasks_by_slot(
    tasks: list[dict],
) -> dict[int, list[dict]]:
    """把真实任务轨迹按 slot 分组。"""
    grouped = defaultdict(list)

    for task in tasks:
        grouped[
            task["slot"]
        ].append(task)

    return dict(grouped)


def final_cache_changes(
    old_state:
        p1_cache.CacheState,
    new_state:
        p1_cache.CacheState,
    edge_nodes: list[dict],
) -> list[dict]:
    """
    最终真实物理缓存变化：
        C(t-1) -> C(t)

    P1/Pjoint 内部搜索动作不计为物理更新。
    """
    changes = []

    for node in edge_nodes:
        node_id = str(
            node["node_id"]
        )

        old_services = (
            p1_cache.cached_services(
                old_state,
                node_id,
            )
        )

        new_services = (
            p1_cache.cached_services(
                new_state,
                node_id,
            )
        )

        if old_services == new_services:
            continue

        changes.append(
            {
                "node_id":
                    node_id,
                "node_type":
                    node["node_type"],
                "removed":
                    sorted(
                        old_services
                        - new_services
                    ),
                "added":
                    sorted(
                        new_services
                        - old_services
                    ),
            }
        )

    return changes


def final_uav_changes(
    old_positions:
        dict[
            str,
            tuple[float, float],
        ],
    new_positions:
        dict[
            str,
            tuple[float, float],
        ],
) -> list[dict]:
    """最终真实 UAV 物理移动 Q(t-1) -> Q(t)。"""
    changes = []

    for uav_id in sorted(
        old_positions
    ):
        old = old_positions[
            uav_id
        ]

        new = new_positions[
            uav_id
        ]

        d = p2_uav.distance(
            old,
            new,
        )

        if d <= config.UAV_MOVE_EPS_M:
            continue

        changes.append(
            {
                "uav_id":
                    uav_id,
                "old_x":
                    old[0],
                "old_y":
                    old[1],
                "new_x":
                    new[0],
                "new_y":
                    new[1],
                "distance_m":
                    d,
            }
        )

    return changes


# =============================================================================
# 实际时隙 P3
# =============================================================================

def run_actual_slot_p3(
    slot_tasks: list[dict],
    edge_nodes: list[dict],
    positions: dict,
    cache_state:
        p1_cache.CacheState,
) -> dict:
    """
    C(t),Q^U(t) 决定以后，
    才使用真实当前任务/位置执行 P3。
    """
    if not slot_tasks:
        return {
            "task_count": 0,
            "total_cost": 0.0,
            "mean_cost": 0.0,
            "mean_delay": 0.0,
            "mean_energy_mJ": 0.0,
            "mode_counts":
                Counter(),
            "eligible_tasks": 0,
        }

    results, stats = (
        p3_solver.solve_slot_p3(
            slot_tasks=slot_tasks,
            edge_nodes=edge_nodes,
            positions=positions,
            cache_state=cache_state,
        )
    )

    total_cost = 0.0
    total_delay = 0.0
    total_energy_j = 0.0
    mode_counts = Counter()

    for result in results.values():
        total_cost += (
            result["cost"]
        )
        total_delay += (
            result["delay"]
        )
        total_energy_j += (
            result["energy_J"]
        )

        mode_counts[
            result["mode"]
        ] += 1

    n = len(slot_tasks)

    return {
        "task_count":
            n,
        "total_cost":
            total_cost,
        "mean_cost":
            total_cost / n,
        "mean_delay":
            total_delay / n,
        "mean_energy_mJ":
            1000.0
            * total_energy_j
            / n,
        "mode_counts":
            mode_counts,
        "eligible_tasks":
            stats["eligible_tasks"],
    }


# =============================================================================
# 输出
# =============================================================================

def write_summary(
    rows: list[dict],
    end_slot: int,
) -> Path:
    """保存逐时隙 P1/P2/Pjoint/P3 摘要。"""
    output_file = (
        config.RESULTS_DIR
        / (
            "rolling_p1_p2_pjoint_summary_"
            f"0_{end_slot}.csv"
        )
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "slot",
        "task_count",
        "edge_eligible_tasks",
        "sv_tasks",
        "rsu_tasks",
        "uav_tasks",
        "cloud_tasks",
        "actual_total_cost",
        "actual_mean_cost",
        "actual_mean_delay_s",
        "actual_mean_energy_mJ",
        "cache_changed",
        "cache_changed_nodes",
        "uav_moved",
        "moved_uavs",
        "p1_changed",
        "p1_search_steps",
        "p1_predicted_p3_cost",
        "p1_switch_cost",
        "p1_gross_gain",
        "p1_net_gain",
        "p1_total_score",
        "p1_p3_evaluations",
        "p2_changed",
        "p2_search_steps",
        "p2_predicted_p3_cost",
        "p2_move_cost",
        "p2_gross_gain",
        "p2_net_gain",
        "p2_total_score",
        "p2_p3_evaluations",
        "pjoint_checked",
        "pjoint_changed",
        "pjoint_joint_candidates",
        "pjoint_predicted_p3_cost",
        "pjoint_cache_switch_cost",
        "pjoint_uav_move_cost",
        "pjoint_gross_gain",
        "pjoint_net_gain",
        "pjoint_total_score",
        "pjoint_p3_evaluations",
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
        writer.writerows(rows)

    return output_file


def write_cache_events(
    events: list[dict],
    end_slot: int,
) -> Path:
    """保存最终真实缓存更新事件。"""
    output_file = (
        config.RESULTS_DIR
        / (
            "rolling_cache_events_"
            f"0_{end_slot}.csv"
        )
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "slot",
        "node_id",
        "node_type",
        "update_source",
        "update_interval_slots",
        "decision_net_gain",
        "removed_services",
        "added_services",
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

        for event in events:
            writer.writerow(
                {
                    "slot":
                        event["slot"],
                    "node_id":
                        event["node_id"],
                    "node_type":
                        event["node_type"],
                    "update_source":
                        event["update_source"],
                    "update_interval_slots":
                        event["update_interval_slots"],
                    "decision_net_gain":
                        f"{event['decision_net_gain']:.12f}",
                    "removed_services":
                        ",".join(
                            str(x)
                            for x in event[
                                "removed"
                            ]
                        ),
                    "added_services":
                        ",".join(
                            str(x)
                            for x in event[
                                "added"
                            ]
                        ),
                }
            )

    return output_file


def write_uav_events(
    events: list[dict],
    end_slot: int,
) -> Path:
    """保存最终真实 UAV 移动事件。"""
    output_file = (
        config.RESULTS_DIR
        / (
            "rolling_uav_events_"
            f"0_{end_slot}.csv"
        )
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "slot",
        "uav_id",
        "update_source",
        "update_interval_slots",
        "decision_net_gain",
        "old_x",
        "old_y",
        "new_x",
        "new_y",
        "distance_m",
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
        writer.writerows(events)

    return output_file


# =============================================================================
# 主循环
# =============================================================================

def run_rolling(
    end_slot: int,
) -> None:
    """
    从 slot=0 顺序递推。

    不能从中间直接启动，因为 C(t-1)、Q^U(t-1)
    都必须来自前序真实状态。
    """
    if not (
        0 <= end_slot < config.T
    ):
        raise ValueError(
            f"end_slot 必须满足 "
            f"0 <= end_slot < {config.T}"
        )

    if config.NUM_UAV <= 0:
        raise ValueError(
            "NUM_UAV 必须大于 0。"
        )

    actual_positions = (
        p3_solver.load_positions()
    )

    all_tasks = (
        p3_solver.load_tasks()
    )

    tasks_by_slot = (
        group_tasks_by_slot(
            all_tasks
        )
    )

    tv_ids = (
        p3_solver.load_ranked_vehicle_ids(
            "TV",
            config.NUM_TV,
        )
    )

    sv_nodes = (
        p3_solver.build_current_sv_nodes()
    )

    sv_ids = [
        str(node["node_id"])
        for node in sv_nodes
    ]

    # ------------------------------------------------------------------
    # t=0 外生状态。
    # ------------------------------------------------------------------
    current_uav_positions = (
        uav_state.initialize_uav_positions(
            tv_ids=tv_ids,
            actual_positions=
                actual_positions,
            num_uav=config.NUM_UAV,
            initial_slot=0,
        )
    )

    initial_edge_nodes = (
        p3_solver.build_edge_nodes(
            current_uav_positions
        )
    )

    current_cache = (
        cache_model.build_initial_cache_state(
            initial_edge_nodes
        )
    )

    cache_model.save_cache_state(
        current_cache,
        initial_edge_nodes,
    )

    summary_rows = []

    # 每个节点独立记录真实更新时刻。
    # t=0 是外生初始状态，不计动态更新事件。
    async_tracker = (
        async_update.AsyncUpdateTracker(
            cache_node_ids=[
                str(node["node_id"])
                for node in initial_edge_nodes
            ],
            uav_ids=[
                str(uav_id)
                for uav_id
                in current_uav_positions
            ],
            initial_slot=0,
        )
    )

    total_tasks = 0
    total_actual_cost = 0.0

    cache_update_slots = 0
    cache_node_updates = 0

    uav_move_slots = 0
    uav_move_events = 0

    pjoint_check_slots = 0
    pjoint_accept_slots = 0

    for slot in range(
        end_slot + 1
    ):
        old_cache = dict(
            current_cache
        )

        old_uav_positions = dict(
            current_uav_positions
        )

        p1_result = None
        p2_result = None
        pjoint_result = None

        p1_evaluator = None
        p2_evaluator = None
        pjoint_evaluator = None

        p1_trigger = None
        p2_trigger = None
        pjoint_trigger = None

        if slot > 0:
            # ==========================================================
            # 1. 同一时隙只生成一次 Xi_t
            # ==========================================================
            horizon = min(
                config.PREDICTION_HORIZON,
                config.T - slot,
            )

            scenario_set = (
                scenario_generator
                .generate_scenario_set(
                    decision_slot=slot,
                    tv_ids=tv_ids,
                    sv_ids=sv_ids,
                    horizon=horizon,
                    num_scenarios=
                        config.NUM_SCENARIOS,
                )
            )

            # 共同的 proxy 排序和 P2 热点都来自同一个 Xi_t。
            edge_nodes_old = (
                p3_solver.build_edge_nodes(
                    old_uav_positions
                )
            )

            proxy_order_by_node = (
                p1_runner
                .build_proxy_order_by_node(
                    scenario_set,
                    edge_nodes_old,
                )
            )

            (
                demand_hotspots,
                cluster_hotspots,
            ) = (
                uav_state
                .build_hotspots_from_scenarios(
                    scenario_set=
                        scenario_set,
                    num_cluster_hotspots=
                        config.NUM_UAV,
                    top_demand_hotspots=3,
                )
            )

            demand_by_uav = {
                uav_id:
                    list(
                        demand_hotspots
                    )
                for uav_id
                in old_uav_positions
            }

            cluster_by_uav = {
                uav_id:
                    list(
                        cluster_hotspots
                    )
                for uav_id
                in old_uav_positions
            }

            # ==========================================================
            # 2. P1：固定 Q^U(t-1)
            # ==========================================================
            p1_evaluator = (
                p1_runner
                .ScenarioP3Evaluator(
                    scenario_set=
                        scenario_set,
                    edge_nodes=
                        edge_nodes_old,
                )
            )

            p1_result = (
                p1_cache.optimize_p1(
                    old_physical_state=
                        old_cache,
                    edge_nodes=
                        edge_nodes_old,
                    evaluate_p3_cost=
                        p1_evaluator,
                    proxy_order_by_node=
                        proxy_order_by_node,
                )
            )

            current_cache = dict(
                p1_result.cache_state
            )

            # P1 的严格改善条件就是缓存自适应触发条件。
            p1_old_p3_cost = (
                p1_evaluator(
                    old_cache
                )
            )

            p1_trigger = (
                async_update
                .compute_trigger_metrics(
                    old_p3_cost=
                        p1_old_p3_cost,
                    new_p3_cost=
                        p1_result.p3_cost,
                    switching_cost=
                        p1_result.switch_cost,
                )
            )

            async_update.require_trigger_if_changed(
                changed=p1_result.changed,
                metrics=p1_trigger,
                label="P1",
            )

            # ==========================================================
            # 3. P2：固定 P1 输出缓存
            # ==========================================================
            p2_evaluator = (
                p2_runner
                .ScenarioUAVP3Evaluator(
                    scenario_set=
                        scenario_set,
                    cache_state=
                        current_cache,
                )
            )

            p2_result = (
                p2_uav.optimize_p2(
                    old_physical_positions=
                        old_uav_positions,
                    evaluate_p3_cost=
                        p2_evaluator,
                    demand_hotspots_by_uav=
                        demand_by_uav,
                    clustered_hotspots_by_uav=
                        cluster_by_uav,
                )
            )

            current_uav_positions = dict(
                p2_result.uav_positions
            )

            # P2 的严格改善条件就是 UAV 自适应触发条件。
            p2_old_p3_cost = (
                p2_evaluator(
                    old_uav_positions
                )
            )

            p2_trigger = (
                async_update
                .compute_trigger_metrics(
                    old_p3_cost=
                        p2_old_p3_cost,
                    new_p3_cost=
                        p2_result.p3_cost,
                    switching_cost=
                        p2_result.move_cost,
                )
            )

            async_update.require_trigger_if_changed(
                changed=p2_result.changed,
                metrics=p2_trigger,
                label="P2",
            )

            # ==========================================================
            # 4. 只有 P1/P2 都 stall 才检查 Pjoint
            # ==========================================================
            if (
                not p1_result.changed
                and not p2_result.changed
            ):
                pjoint_check_slots += 1

                pjoint_evaluator = (
                    pjoint
                    .ScenarioJointP3Evaluator(
                        scenario_set=
                            scenario_set
                    )
                )

                pjoint_result = (
                    pjoint.optimize_pjoint(
                        old_physical_cache=
                            old_cache,
                        old_physical_uav_positions=
                            old_uav_positions,
                        edge_nodes=
                            edge_nodes_old,
                        evaluate_p3_cost=
                            pjoint_evaluator,
                        proxy_order_by_node=
                            proxy_order_by_node,
                        demand_hotspots_by_uav=
                            demand_by_uav,
                        clustered_hotspots_by_uav=
                            cluster_by_uav,
                    )
                )

                pjoint_old_p3_cost = (
                    pjoint_evaluator(
                        old_cache,
                        old_uav_positions,
                    )
                )

                pjoint_trigger = (
                    async_update
                    .compute_trigger_metrics(
                        old_p3_cost=
                            pjoint_old_p3_cost,
                        new_p3_cost=
                            pjoint_result.p3_cost,
                        switching_cost=(
                            pjoint_result
                            .cache_switch_cost
                            + pjoint_result
                            .uav_move_cost
                        ),
                    )
                )

                async_update.require_trigger_if_changed(
                    changed=
                        pjoint_result.changed,
                    metrics=
                        pjoint_trigger,
                    label="Pjoint",
                )

                if pjoint_result.changed:
                    pjoint_accept_slots += 1

                    current_cache = dict(
                        pjoint_result
                        .cache_state
                    )

                    current_uav_positions = dict(
                        pjoint_result
                        .uav_positions
                    )

        # ==============================================================
        # 5. 所有配置搜索结束后，统一比较“旧物理状态 -> 最终状态”
        # ==============================================================
        final_edge_nodes = (
            p3_solver.build_edge_nodes(
                current_uav_positions
            )
        )

        # 缓存节点 ID/类型不随 UAV 水平坐标改变，
        # 因此用最终 edge_nodes 记录事件即可。
        cache_changes = (
            final_cache_changes(
                old_state=
                    old_cache,
                new_state=
                    current_cache,
                edge_nodes=
                    final_edge_nodes,
            )
        )

        uav_changes = (
            final_uav_changes(
                old_positions=
                    old_uav_positions,
                new_positions=
                    current_uav_positions,
            )
        )

        if cache_changes:
            cache_update_slots += 1
            cache_node_updates += len(
                cache_changes
            )

            if (
                pjoint_result is not None
                and pjoint_result.changed
            ):
                cache_source = "Pjoint"
                cache_net_gain = (
                    pjoint_trigger.net_gain
                )
            else:
                cache_source = "P1"
                cache_net_gain = (
                    p1_trigger.net_gain
                )

            async_tracker.record_cache_changes(
                slot=slot,
                changes=cache_changes,
                source=cache_source,
                decision_net_gain=
                    cache_net_gain,
            )

        if uav_changes:
            uav_move_slots += 1
            uav_move_events += len(
                uav_changes
            )

            if (
                pjoint_result is not None
                and pjoint_result.changed
            ):
                uav_source = "Pjoint"
                uav_net_gain = (
                    pjoint_trigger.net_gain
                )
            else:
                uav_source = "P2"
                uav_net_gain = (
                    p2_trigger.net_gain
                )

            async_tracker.record_uav_changes(
                slot=slot,
                changes=uav_changes,
                source=uav_source,
                decision_net_gain=
                    uav_net_gain,
            )

        # ==============================================================
        # 6. C(t),Q^U(t) 确定后才执行真实当前任务
        # ==============================================================
        actual = (
            run_actual_slot_p3(
                slot_tasks=
                    tasks_by_slot.get(
                        slot,
                        [],
                    ),
                edge_nodes=
                    final_edge_nodes,
                positions=
                    actual_positions,
                cache_state=
                    current_cache,
            )
        )

        total_tasks += (
            actual["task_count"]
        )

        total_actual_cost += (
            actual["total_cost"]
        )

        summary_rows.append(
            {
                "slot":
                    slot,
                "task_count":
                    actual[
                        "task_count"
                    ],
                "edge_eligible_tasks":
                    actual[
                        "eligible_tasks"
                    ],
                "sv_tasks":
                    actual[
                        "mode_counts"
                    ].get(
                        "SV",
                        0,
                    ),
                "rsu_tasks":
                    actual[
                        "mode_counts"
                    ].get(
                        "RSU",
                        0,
                    ),
                "uav_tasks":
                    actual[
                        "mode_counts"
                    ].get(
                        "UAV",
                        0,
                    ),
                "cloud_tasks":
                    actual[
                        "mode_counts"
                    ].get(
                        "CLOUD",
                        0,
                    ),
                "actual_total_cost":
                    f"{actual['total_cost']:.12f}",
                "actual_mean_cost":
                    f"{actual['mean_cost']:.12f}",
                "actual_mean_delay_s":
                    f"{actual['mean_delay']:.12f}",
                "actual_mean_energy_mJ":
                    f"{actual['mean_energy_mJ']:.12f}",
                "cache_changed":
                    int(
                        bool(
                            cache_changes
                        )
                    ),
                "cache_changed_nodes":
                    len(
                        cache_changes
                    ),
                "uav_moved":
                    int(
                        bool(
                            uav_changes
                        )
                    ),
                "moved_uavs":
                    len(
                        uav_changes
                    ),
                "p1_changed":
                    (
                        0
                        if p1_result
                        is None
                        else int(
                            p1_result.changed
                        )
                    ),
                "p1_search_steps":
                    (
                        0
                        if p1_result
                        is None
                        else len(
                            p1_result.steps
                        )
                    ),
                "p1_predicted_p3_cost":
                    (
                        ""
                        if p1_result
                        is None
                        else
                        f"{p1_result.p3_cost:.12f}"
                    ),
                "p1_switch_cost":
                    (
                        ""
                        if p1_result
                        is None
                        else
                        f"{p1_result.switch_cost:.12f}"
                    ),
                "p1_gross_gain":
                    (
                        ""
                        if p1_trigger
                        is None
                        else
                        f"{p1_trigger.gross_gain:.12f}"
                    ),
                "p1_net_gain":
                    (
                        ""
                        if p1_trigger
                        is None
                        else
                        f"{p1_trigger.net_gain:.12f}"
                    ),
                "p1_total_score":
                    (
                        ""
                        if p1_result
                        is None
                        else
                        f"{p1_result.total_score:.12f}"
                    ),
                "p1_p3_evaluations":
                    (
                        0
                        if p1_evaluator
                        is None
                        else
                        p1_evaluator
                        .num_evaluations
                    ),
                "p2_changed":
                    (
                        0
                        if p2_result
                        is None
                        else int(
                            p2_result.changed
                        )
                    ),
                "p2_search_steps":
                    (
                        0
                        if p2_result
                        is None
                        else len(
                            p2_result.steps
                        )
                    ),
                "p2_predicted_p3_cost":
                    (
                        ""
                        if p2_result
                        is None
                        else
                        f"{p2_result.p3_cost:.12f}"
                    ),
                "p2_move_cost":
                    (
                        ""
                        if p2_result
                        is None
                        else
                        f"{p2_result.move_cost:.12f}"
                    ),
                "p2_gross_gain":
                    (
                        ""
                        if p2_trigger
                        is None
                        else
                        f"{p2_trigger.gross_gain:.12f}"
                    ),
                "p2_net_gain":
                    (
                        ""
                        if p2_trigger
                        is None
                        else
                        f"{p2_trigger.net_gain:.12f}"
                    ),
                "p2_total_score":
                    (
                        ""
                        if p2_result
                        is None
                        else
                        f"{p2_result.total_score:.12f}"
                    ),
                "p2_p3_evaluations":
                    (
                        0
                        if p2_evaluator
                        is None
                        else
                        p2_evaluator
                        .num_evaluations
                    ),
                "pjoint_checked":
                    int(
                        pjoint_result
                        is not None
                    ),
                "pjoint_changed":
                    (
                        0
                        if pjoint_result
                        is None
                        else int(
                            pjoint_result
                            .changed
                        )
                    ),
                "pjoint_joint_candidates":
                    (
                        0
                        if pjoint_result
                        is None
                        else
                        pjoint_result
                        .evaluated_joint_candidates
                    ),
                "pjoint_predicted_p3_cost":
                    (
                        ""
                        if pjoint_result
                        is None
                        else
                        f"{pjoint_result.p3_cost:.12f}"
                    ),
                "pjoint_cache_switch_cost":
                    (
                        ""
                        if pjoint_result
                        is None
                        else
                        f"{pjoint_result.cache_switch_cost:.12f}"
                    ),
                "pjoint_uav_move_cost":
                    (
                        ""
                        if pjoint_result
                        is None
                        else
                        f"{pjoint_result.uav_move_cost:.12f}"
                    ),
                "pjoint_gross_gain":
                    (
                        ""
                        if pjoint_trigger
                        is None
                        else
                        f"{pjoint_trigger.gross_gain:.12f}"
                    ),
                "pjoint_net_gain":
                    (
                        ""
                        if pjoint_trigger
                        is None
                        else
                        f"{pjoint_trigger.net_gain:.12f}"
                    ),
                "pjoint_total_score":
                    (
                        ""
                        if pjoint_result
                        is None
                        else
                        f"{pjoint_result.total_score:.12f}"
                    ),
                "pjoint_p3_evaluations":
                    (
                        0
                        if pjoint_evaluator
                        is None
                        else
                        pjoint_evaluator
                        .num_evaluations
                    ),
            }
        )

        if (
            slot == 0
            or slot == end_slot
            or slot % 10 == 0
        ):
            print(
                f"[slot {slot:03d}] "
                f"tasks="
                f"{actual['task_count']}, "
                f"cache_changed="
                f"{bool(cache_changes)}, "
                f"uav_moved="
                f"{bool(uav_changes)}, "
                f"pjoint={pjoint_result is not None}"
            )

    # ------------------------------------------------------------------
    # 保存结果。
    # ------------------------------------------------------------------
    summary_file = (
        write_summary(
            summary_rows,
            end_slot,
        )
    )

    cache_event_file = (
        write_cache_events(
            async_tracker.cache_events,
            end_slot,
        )
    )

    uav_event_file = (
        write_uav_events(
            async_tracker.uav_events,
            end_slot,
        )
    )

    mean_cost = (
        total_actual_cost
        / total_tasks
        if total_tasks > 0
        else 0.0
    )

    print()
    print("=" * 76)
    print(
        "P1 + P2 + Pjoint "
        "滚动在线控制完成"
    )
    print("=" * 76)
    print(
        f"时隙范围                  : "
        f"0 ... {end_slot}"
    )
    print(
        f"实际任务总数              : "
        f"{total_tasks}"
    )
    print(
        f"实际任务平均归一化代价    : "
        f"{mean_cost:.9f}"
    )
    print(
        f"发生缓存更新的时隙数      : "
        f"{cache_update_slots}"
    )
    print(
        f"节点级真实缓存更新次数    : "
        f"{cache_node_updates}"
    )
    print(
        f"发生 UAV 移动的时隙数     : "
        f"{uav_move_slots}"
    )
    print(
        f"UAV 真实移动事件数        : "
        f"{uav_move_events}"
    )
    print(
        f"Pjoint 检查时隙数         : "
        f"{pjoint_check_slots}"
    )
    print(
        f"Pjoint 接受时隙数         : "
        f"{pjoint_accept_slots}"
    )
    print()
    print(
        f"时隙摘要："
        f"{summary_file}"
    )
    print(
        f"缓存事件："
        f"{cache_event_file}"
    )
    print(
        f"UAV 事件："
        f"{uav_event_file}"
    )
    print()
    print(
        f"自适应触发检查时隙数      : "
        f"{end_slot}"
    )
    print()
    print(
        "P1 / P2 / Pjoint + 异步自适应更新链已接通。"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "从 slot=0 顺序执行 "
            "P1 + P2 + Pjoint 滚动控制。"
        )
    )

    parser.add_argument(
        "--end-slot",
        type=int,
        default=config.T - 1,
        help=(
            "结束时隙。"
            "集成阶段建议先短跑。"
        ),
    )

    args = parser.parse_args()

    run_rolling(
        end_slot=args.end_slot
    )


if __name__ == "__main__":
    main()
