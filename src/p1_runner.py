# -*- coding: utf-8 -*-
"""
单时刻 P1 在线决策运行器。

主线：
    H_{t-1}
        ↓
    scenario_generator.generate_scenario_set()
        ↓
    公共随机场景 Xi_t
        ↓
    对每个缓存候选调用完整 P3
        ↓
    p1_cache.optimize_p1()
        ↓
    得到时隙 t 的候选缓存配置 C(t)

本文件只负责把三个已经实现的模块串起来，不新增模型假设。

运行示例：
    python src/p1_runner.py --slot 100

重要：
1. 同一个决策时刻只生成一次 Xi_t；
2. P1 的所有 add/delete/swap/proxy 候选复用同一个 Xi_t；
3. 不读取未来真实 SUMO 轨迹；
4. 不读取未来真实 popularity；
5. 当前 old physical state 使用统一初始缓存状态；
   下一步扩展为滚动控制器后，会改成上一时隙真实保留下来的缓存状态。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cache_model
import config
import p1_cache
import p3_solver
import scenario_generator


# =============================================================================
# P3 场景评价器
# =============================================================================

class ScenarioP3Evaluator:
    """
    固定一个 ScenarioSet，构造 P1 所需的：

        evaluate_p3_cost(cache_state)

    同一个对象生命周期内，所有缓存候选都使用完全相同的 Xi_t，
    从而满足 Common Random Numbers。
    """

    def __init__(
        self,
        scenario_set: scenario_generator.ScenarioSet,
        edge_nodes: list[dict],
    ):
        self.scenario_set = scenario_set
        self.edge_nodes = edge_nodes

        # 相同缓存状态可能在局部搜索中重复出现。
        # 用状态签名缓存其 P3 评价，避免重复求解。
        self._cost_cache: dict[tuple, float] = {}

        self.num_evaluations = 0
        self.num_cache_hits = 0

    def __call__(
        self,
        cache_state: p1_cache.CacheState,
    ) -> float:
        """
        返回候选缓存配置在 Xi_t 下的平均 P3 总代价：

            V_hat_t(C)
            = (1/S) sum_s sum_{tau=t}^{t+H-1} Phi_tau(C; xi_s)

        当前 p3_solver 的每个任务结果中 cost 即归一化 J_v，
        因此一个 scenario 的 Phi 总和直接取所有预测任务 cost 之和。
        """
        signature = p1_cache.state_signature(
            cache_state
        )

        if signature in self._cost_cache:
            self.num_cache_hits += 1
            return self._cost_cache[signature]

        total_across_scenarios = 0.0

        for scenario in self.scenario_set.scenarios:
            # 某个场景可能恰好没有任务。
            # 此时该场景的 P3 任务代价为 0。
            if not scenario.tasks:
                continue

            results, _ = p3_solver.solve_episode_p3(
                tasks=scenario.tasks,
                edge_nodes=self.edge_nodes,
                positions=scenario.positions,
                cache_state=cache_state,
            )

            scenario_cost = sum(
                result["cost"]
                for result in results.values()
            )

            total_across_scenarios += scenario_cost

        mean_cost = (
            total_across_scenarios
            / len(self.scenario_set.scenarios)
        )

        self._cost_cache[signature] = mean_cost
        self.num_evaluations += 1

        return mean_cost


# =============================================================================
# 因果 proxy-demand 排序
# =============================================================================

def build_proxy_order_by_node(
    scenario_set: scenario_generator.ScenarioSet,
    edge_nodes: list[dict],
) -> dict[str, list[int]]:
    """
    根据当前因果估计流行度 rho_hat 构造 proxy-demand 服务优先级。

    当前所有 SV 使用同一个全局流行度估计。
    后续加入空间分区 rho_hat_{g,f} 后，只需在这里改成节点所在区域排序。

    注意：
    proxy 只是 P1 的一个候选邻域。
    最终是否采用仍由完整 P3 + 切换成本决定。
    """
    order = sorted(
        range(config.NUM_SERVICES),
        key=lambda service_id:
            scenario_set.estimated_popularity[
                service_id
            ],
        reverse=True,
    )

    return {
        str(node["node_id"]): list(order)
        for node in edge_nodes
    }


# =============================================================================
# 输出
# =============================================================================

def save_p1_result(
    decision_slot: int,
    edge_nodes: list[dict],
    old_state: p1_cache.CacheState,
    result: p1_cache.P1Result,
) -> Path:
    """
    保存本次单时刻 P1 的最终缓存状态。

    这里只保存结果，不把它自动写回下一时隙；
    下一步滚动控制器会负责状态递推。
    """
    output_file = (
        config.RESULTS_DIR
        / f"p1_slot_{decision_slot}.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    service_sizes = (
        cache_model.load_service_sizes_mb()
    )

    fields = [
        "decision_slot",
        "node_id",
        "service_id",
        "old_cached",
        "new_cached",
        "service_size_MB",
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
            node_id = str(node["node_id"])

            for service_id in range(
                config.NUM_SERVICES
            ):
                writer.writerow(
                    {
                        "decision_slot":
                            decision_slot,
                        "node_id":
                            node_id,
                        "service_id":
                            service_id,
                        "old_cached":
                            int(
                                old_state.get(
                                    (
                                        node_id,
                                        service_id,
                                    ),
                                    False,
                                )
                            ),
                        "new_cached":
                            int(
                                result.cache_state.get(
                                    (
                                        node_id,
                                        service_id,
                                    ),
                                    False,
                                )
                            ),
                        "service_size_MB":
                            f"{service_sizes[service_id]:.6f}",
                    }
                )

    return output_file


def print_cache_state(
    title: str,
    cache_state: p1_cache.CacheState,
    edge_nodes: list[dict],
) -> None:
    """打印各节点缓存集合及占用容量。"""
    service_sizes = (
        cache_model.load_service_sizes_mb()
    )

    capacity_mb = (
        cache_model.sv_cache_capacity_mb(
            service_sizes
        )
    )

    print(title)

    for node in edge_nodes:
        node_id = str(node["node_id"])

        services = (
            cache_model.get_cached_services(
                cache_state,
                node_id,
            )
        )

        used_mb = (
            cache_model.cache_usage_mb(
                cache_state,
                node_id,
                service_sizes,
            )
        )

        print(
            f"  SV {node_id}: "
            f"services={services}, "
            f"used={used_mb:.3f}/"
            f"{capacity_mb:.3f} MB"
        )


# =============================================================================
# 单时刻在线 P1
# =============================================================================

def run_p1_once(
    decision_slot: int,
) -> None:
    """
    在一个给定决策时刻执行完整的：

        H_{t-1} -> Xi_t -> P3 evaluator -> P1
    """

    if not (
        1 <= decision_slot < config.T
    ):
        raise ValueError(
            f"decision_slot 必须满足 "
            f"1 <= t < {config.T}"
        )

    # 越靠近 episode 结束，预测时域自然截短，
    # 不跨越本次实验的 T=400 边界。
    horizon = min(
        config.PREDICTION_HORIZON,
        config.T - decision_slot,
    )

    # ------------------------------------------------------------------
    # 1. 固定当前参与实验的 TV / SV
    # ------------------------------------------------------------------
    tv_ids = (
        p3_solver.load_ranked_vehicle_ids(
            "TV",
            config.NUM_TV,
        )
    )

    edge_nodes = (
        p3_solver.build_current_sv_nodes()
    )

    sv_ids = [
        str(node["node_id"])
        for node in edge_nodes
        if node["node_type"].upper() == "SV"
    ]

    # ------------------------------------------------------------------
    # 2. 当前真实旧物理缓存状态 C(t-1)
    #
    # 当前单时刻集成阶段仍从统一初始缓存状态出发。
    # 下一步 400-slot controller 会把这里替换成上一时隙真实状态。
    # ------------------------------------------------------------------
    old_physical_state = (
        cache_model.build_initial_cache_state(
            edge_nodes
        )
    )

    # ------------------------------------------------------------------
    # 3. 在 t 时刻只生成一次公共随机场景 Xi_t
    # ------------------------------------------------------------------
    scenario_set = (
        scenario_generator.generate_scenario_set(
            decision_slot=decision_slot,
            tv_ids=tv_ids,
            sv_ids=sv_ids,
            horizon=horizon,
            num_scenarios=config.NUM_SCENARIOS,
        )
    )

    # ------------------------------------------------------------------
    # 4. 构造固定 Xi_t 的 P3 评价器
    # ------------------------------------------------------------------
    evaluator = ScenarioP3Evaluator(
        scenario_set=scenario_set,
        edge_nodes=edge_nodes,
    )

    # ------------------------------------------------------------------
    # 5. 使用因果流行度估计生成 proxy 邻域
    # ------------------------------------------------------------------
    proxy_order_by_node = (
        build_proxy_order_by_node(
            scenario_set,
            edge_nodes,
        )
    )

    # ------------------------------------------------------------------
    # 6. 正式调用 P1
    # ------------------------------------------------------------------
    result = p1_cache.optimize_p1(
        old_physical_state=old_physical_state,
        edge_nodes=edge_nodes,
        evaluate_p3_cost=evaluator,
        proxy_order_by_node=proxy_order_by_node,
    )

    # ------------------------------------------------------------------
    # 7. 保存本次结果
    # ------------------------------------------------------------------
    output_file = save_p1_result(
        decision_slot=decision_slot,
        edge_nodes=edge_nodes,
        old_state=old_physical_state,
        result=result,
    )

    # ------------------------------------------------------------------
    # 8. 控制台摘要
    # ------------------------------------------------------------------
    print("=" * 76)
    print("单时刻 P1 在线决策完成")
    print("=" * 76)
    print(f"决策时隙 t                : {decision_slot}")
    print(f"预测时域 H                : {horizon}")
    print(
        f"公共随机场景 S            : "
        f"{len(scenario_set.scenarios)}"
    )
    print(
        f"TV / SV                   : "
        f"{len(tv_ids)} / {len(sv_ids)}"
    )
    print()

    print("因果估计流行度 rho_hat：")
    for service_id, probability in enumerate(
        scenario_set.estimated_popularity
    ):
        print(
            f"  service {service_id}: "
            f"{probability:.6f}"
        )

    print()

    print_cache_state(
        "真实旧缓存状态 C(t-1)：",
        old_physical_state,
        edge_nodes,
    )

    print()

    print_cache_state(
        "P1 输出缓存状态：",
        result.cache_state,
        edge_nodes,
    )

    print()
    print(
        f"是否发生缓存变化          : "
        f"{result.changed}"
    )
    print(
        f"预测 P3 代价              : "
        f"{result.p3_cost:.9f}"
    )
    print(
        f"缓存切换成本              : "
        f"{result.switch_cost:.9f}"
    )
    print(
        f"总配置代价                : "
        f"{result.total_score:.9f}"
    )
    print(
        f"P1 接受动作数             : "
        f"{len(result.steps)}"
    )
    print(
        f"实际执行 P3 配置评价次数  : "
        f"{evaluator.num_evaluations}"
    )
    print(
        f"重复状态缓存命中次数      : "
        f"{evaluator.num_cache_hits}"
    )

    if result.steps:
        print()
        print("被接受的严格改善动作：")

        for step in result.steps:
            print(
                f"  iter={step.iteration}, "
                f"node={step.node_id}, "
                f"type={step.move_type}, "
                f"remove={list(step.removed_services)}, "
                f"add={list(step.added_services)}, "
                f"improvement={step.improvement:.9f}"
            )

    print()
    print(f"结果文件：{output_file}")
    print()
    print(
        "本次运行只验证 "
        "H_{t-1} -> Xi_t -> P3 -> P1 "
        "整条因果链。"
    )


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="运行一个决策时刻的 P1 在线缓存优化。"
    )

    parser.add_argument(
        "--slot",
        type=int,
        required=True,
        help=(
            "配置决策时隙 t，例如 --slot 100。"
            "只能使用 t-1 以前的历史信息。"
        ),
    )

    args = parser.parse_args()

    run_p1_once(
        decision_slot=args.slot
    )


if __name__ == "__main__":
    main()
