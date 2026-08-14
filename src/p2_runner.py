# -*- coding: utf-8 -*-
"""
P2 单时刻集成运行器。

只用于验证：
    H_0
      -> Xi_1
      -> 预测需求热点
      -> P3 UAV 配置评价
      -> P2

为什么当前只运行 t=1：
    只有在真正的滚动控制器里，才能得到任意时刻真实 Q^U(t-1)。
    为避免在 t=100 等时刻伪造 UAV 历史状态，
    单时刻集成测试只使用：
        已定义的 Q^U(0) -> 决策 Q^U(1)

下一步把 P2 接入 rolling_controller 后，
Q^U(t-1) 才会逐时隙真实递推。
"""

from __future__ import annotations

import csv
from pathlib import Path

import cache_model
import config
import p2_uav
import p3_solver
import scenario_generator
import uav_state


class ScenarioUAVP3Evaluator:
    """
    固定 ScenarioSet 和缓存状态，
    只改变 UAV 水平位置 Q^U。

    因此 P2 的所有位置候选复用完全相同的 Xi_t。
    """

    def __init__(
        self,
        scenario_set,
        cache_state,
    ):
        self.scenario_set = (
            scenario_set
        )

        self.cache_state = dict(
            cache_state
        )

        self._cost_cache = {}

        self.num_evaluations = 0
        self.num_cache_hits = 0

    def __call__(
        self,
        uav_positions,
    ) -> float:
        signature = (
            p2_uav.position_signature(
                uav_positions
            )
        )

        if (
            signature
            in self._cost_cache
        ):
            self.num_cache_hits += 1

            return (
                self._cost_cache[
                    signature
                ]
            )

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
                        self.cache_state,
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


def save_result(
    old_positions,
    result,
) -> Path:
    """保存 t=1 的 P2 集成结果。"""
    output_file = (
        config.RESULTS_DIR
        / "p2_slot_1.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "uav_id",
        "old_x",
        "old_y",
        "new_x",
        "new_y",
        "move_distance_m",
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

        for uav_id in sorted(
            old_positions
        ):
            old = (
                old_positions[
                    uav_id
                ]
            )

            new = (
                result.uav_positions[
                    uav_id
                ]
            )

            writer.writerow(
                {
                    "uav_id":
                        uav_id,
                    "old_x":
                        f"{old[0]:.6f}",
                    "old_y":
                        f"{old[1]:.6f}",
                    "new_x":
                        f"{new[0]:.6f}",
                    "new_y":
                        f"{new[1]:.6f}",
                    "move_distance_m":
                        f"{p2_uav.distance(old, new):.6f}",
                }
            )

    return output_file


def main():
    """
    完成一次：
        Q^U(0) -> P2 -> Q^U(1)
    """

    decision_slot = 1

    if config.NUM_UAV <= 0:
        raise ValueError(
            "NUM_UAV 必须大于 0。"
        )

    # ------------------------------------------------------------------
    # 1. 固定 TV / SV 和真实 slot 0 位置。
    # ------------------------------------------------------------------
    actual_positions = (
        p3_solver.load_positions()
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
    # 2. 用 slot 0 已观测 TV 位置构造真实初始 Q^U(0)。
    # ------------------------------------------------------------------
    old_uav_positions = (
        uav_state.initialize_uav_positions(
            tv_ids=tv_ids,
            actual_positions=
                actual_positions,
            num_uav=config.NUM_UAV,
            initial_slot=0,
        )
    )

    # ------------------------------------------------------------------
    # 3. 初始缓存：SV 使用现有统一初始状态，
    #    UAV 在同一容量模型下加入独立初始缓存。
    # ------------------------------------------------------------------
    sv_cache = (
        cache_model.build_initial_cache_state(
            sv_nodes
        )
    )

    cache_state = (
        uav_state.extend_cache_state_for_uavs(
            base_cache_state=
                sv_cache,
            uav_positions=
                old_uav_positions,
        )
    )

    # ------------------------------------------------------------------
    # 4. 在 t=1 只生成一次 Xi_1。
    # ------------------------------------------------------------------
    scenario_set = (
        scenario_generator.generate_scenario_set(
            decision_slot=
                decision_slot,
            tv_ids=tv_ids,
            sv_ids=sv_ids,
            horizon=min(
                config.PREDICTION_HORIZON,
                config.T
                - decision_slot,
            ),
            num_scenarios=
                config.NUM_SCENARIOS,
        )
    )

    # ------------------------------------------------------------------
    # 5. 只从 Xi_1 中构造热点候选。
    # ------------------------------------------------------------------
    (
        demand_hotspots,
        cluster_hotspots,
    ) = (
        uav_state.build_hotspots_from_scenarios(
            scenario_set=
                scenario_set,
            num_cluster_hotspots=
                config.NUM_UAV,
            top_demand_hotspots=3,
        )
    )

    demand_by_uav = {
        uav_id: list(
            demand_hotspots
        )
        for uav_id
        in old_uav_positions
    }

    cluster_by_uav = {
        uav_id: list(
            cluster_hotspots
        )
        for uav_id
        in old_uav_positions
    }

    # ------------------------------------------------------------------
    # 6. 固定 Xi_1 + cache_state 的 P3 位置评价器。
    # ------------------------------------------------------------------
    evaluator = (
        ScenarioUAVP3Evaluator(
            scenario_set=
                scenario_set,
            cache_state=
                cache_state,
        )
    )

    # ------------------------------------------------------------------
    # 7. P2。
    # ------------------------------------------------------------------
    result = (
        p2_uav.optimize_p2(
            old_physical_positions=
                old_uav_positions,
            evaluate_p3_cost=
                evaluator,
            demand_hotspots_by_uav=
                demand_by_uav,
            clustered_hotspots_by_uav=
                cluster_by_uav,
        )
    )

    output_file = save_result(
        old_positions=
            old_uav_positions,
        result=result,
    )

    # ------------------------------------------------------------------
    # 8. 摘要。
    # ------------------------------------------------------------------
    print("=" * 76)
    print("P2 单时刻集成完成")
    print("=" * 76)
    print(
        f"决策时隙                 : "
        f"{decision_slot}"
    )
    print(
        f"UAV 数量                 : "
        f"{config.NUM_UAV}"
    )
    print(
        f"固定高度                 : "
        f"{config.UAV_HEIGHT_M:.1f} m"
    )
    print(
        f"单时隙最大水平移动       : "
        f"{p2_uav.max_move_distance_m():.3f} m"
    )
    print(
        f"预测场景                 : "
        f"{len(scenario_set.scenarios)}"
    )
    print()

    print("Q^U(0)：")
    for uav_id, pos in sorted(
        old_uav_positions.items()
    ):
        print(
            f"  {uav_id}: "
            f"({pos[0]:.3f}, "
            f"{pos[1]:.3f})"
        )

    print()

    print("预测 demand hotspots：")
    for pos in demand_hotspots:
        print(
            f"  ({pos[0]:.3f}, "
            f"{pos[1]:.3f})"
        )

    print()

    print("预测 clustered hotspots：")
    for pos in cluster_hotspots:
        print(
            f"  ({pos[0]:.3f}, "
            f"{pos[1]:.3f})"
        )

    print()

    print("P2 输出 Q^U(1)：")
    for uav_id, pos in sorted(
        result.uav_positions.items()
    ):
        old = (
            old_uav_positions[
                uav_id
            ]
        )

        print(
            f"  {uav_id}: "
            f"({pos[0]:.3f}, "
            f"{pos[1]:.3f}), "
            f"move="
            f"{p2_uav.distance(old, pos):.3f} m"
        )

    print()
    print(
        f"是否移动                 : "
        f"{result.changed}"
    )
    print(
        f"预测 P3 代价             : "
        f"{result.p3_cost:.9f}"
    )
    print(
        f"UAV 移动成本             : "
        f"{result.move_cost:.9f}"
    )
    print(
        f"总配置代价               : "
        f"{result.total_score:.9f}"
    )
    print(
        f"P2 接受动作数            : "
        f"{len(result.steps)}"
    )
    print(
        f"P3 UAV 配置评价次数      : "
        f"{evaluator.num_evaluations}"
    )

    if result.steps:
        print()
        print("被接受的 P2 动作：")

        for step in result.steps:
            print(
                f"  iter={step.iteration}, "
                f"uav={step.uav_id}, "
                f"type="
                f"{step.candidate_type}, "
                f"move="
                f"{p2_uav.distance(step.old_position, step.new_position):.3f} m, "
                f"improvement="
                f"{step.improvement:.9f}"
            )

    print()
    print(
        f"结果文件："
        f"{output_file}"
    )
    print()
    print(
        "这一步只验证 "
        "Q^U(0) -> Xi_1 -> P3 -> P2。"
    )


if __name__ == "__main__":
    main()
