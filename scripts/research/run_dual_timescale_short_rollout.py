#!/usr/bin/env python3
"""Research-only short realized rollout for the dual-timescale controller.

Run under: conda activate adaptive_cache_mec
This script reads an existing Training realization; it never creates data,
modifies the formal runner/checkpoints, or reads realized future arrivals
during action selection. At most five contiguous slots are evaluated.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision import (  # noqa: E402
    assignment_optimizer,
    cache_configuration,
    dual_timescale_controller,
    slot_environment,
    task_allocation,
)
from ppo_simple.interface import load_environment_context  # noqa: E402


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--realization-dir",
        default="environment_data/proposed_joint_training/episode_20001",
    )
    parser.add_argument("--start-slot", type=int, default=0)
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=("joint", "cache_only", "uav_only", "keep_stay"),
        default="joint",
    )
    parser.add_argument("--cache-hold-slots", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--actual-assignment-budget", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.slots <= 5:
        parser.error("--slots must be in [1, 5] (minimal evaluation only)")
    if args.actual_assignment_budget < 1:
        parser.error("--actual-assignment-budget must be positive")

    context = load_environment_context(
        config_path=str(ROOT / "configs/proposed/system_config.toml"),
        realization_dir=args.realization_dir,
    )
    cfg = context.cfg
    from common import project_config
    if project_config.get_config_value(cfg, "nodes.cloud.resource_mode") != "finite_pool":
        raise RuntimeError("Cloud must use finite shared CPU pool")
    cloud_capacity = float(
        project_config.get_config_value(
            cfg, "nodes.cloud.cpu_capacity_cycles_per_s"
        )
    )
    if not math.isclose(cloud_capacity, 40e9, rel_tol=0.0, abs_tol=1.0):
        raise RuntimeError("Cloud must use the confirmed 40 GHz capacity")

    slots = np.asarray(context.scenarios["origin_slots"], dtype=int).tolist()
    slot_to_origin_index = {int(slot): idx for idx, slot in enumerate(slots)}
    if any((args.start_slot + n) not in slot_to_origin_index
           for n in range(args.slots)):
        raise ValueError("requested slots are not contiguous forecast origins")
    state = dual_timescale_controller.initialize_controller(
        context.initial_cache, context.initial_uav,
    )

    total_matching_net = 0.0
    total_new_net = 0.0
    total_prediction_oracles = 0
    started_at = time.perf_counter()
    for step in range(args.slots):
        slot = args.start_slot + step
        origin = cache_configuration.scenario_at_origin(
            context.scenarios, slot_to_origin_index[slot],
        )
        prior_cache = state.cache_state.copy()
        prior_uav = {k: v.copy() for k, v in state.uav_positions.items()}

        # PLAN FIRST, from forecast data only. Actual task/position files are
        # not consulted until after the action is frozen.
        decision = dual_timescale_controller.decide_dual_timescale_action(
            state, origin, context.service_catalog, cfg, context.cache_nodes,
            context.cache_cfg, context.flight_boundary,
            cache_hold_slots=args.cache_hold_slots,
            planning_horizon=args.horizon,
            max_cache_proposals=(
                0 if args.mode in {"uav_only", "keep_stay"} else 4
            ),
            max_uav_proposals=(
                0 if args.mode in {"cache_only", "keep_stay"} else 4
            ),
            screen_scenarios=1, full_scenarios=2,
            screen_assignment_budget=8, full_assignment_budget=24,
        )
        time_s = float(origin["origin_time_s"])
        actual_tasks = task_allocation.load_actual_tasks(
            context.realization.tasks, slot
        )
        vehicle_ids = {
            str(value) for value in origin["vehicle_ids"].tolist()
        }
        actual_positions = task_allocation.load_vehicle_positions(
            context.realization.trajectory, time_s, vehicle_ids
        )
        # Formal execution provides an authoritative physical/cost reference.
        formal = slot_environment.execute_configured_slot(
            prior_cache, prior_uav,
            decision.cache_state, decision.uav_positions,
            origin, actual_tasks, actual_positions,
            context.service_catalog, cfg, context.cache_nodes,
            context.cache_cfg, context.flight_boundary,
            slot_id=slot, time_s=time_s,
        )
        if not formal.step_feasible:
            raise RuntimeError(
                f"formal physical execution infeasible at slot {slot}: "
                f"{formal.infeasible_reason}"
            )

        # Evaluate the research assignment oracle on exactly the same real
        # tasks and the *executed* joint configuration and actual positions.
        node_positions = task_allocation.build_service_node_positions(
            context.cache_nodes, actual_positions, decision.uav_positions
        )
        new = assignment_optimizer.optimize_assignment_with_resource_oracle(
            actual_tasks, actual_positions, node_positions,
            decision.cache_state, cfg, context.cache_nodes,
            context.cache_cfg,
            max_oracle_evaluations=args.actual_assignment_budget,
        )
        if not new.evaluation.resource_feasible:
            raise AssertionError("research oracle produced infeasible allocation")
        cloud_used = float(
            new.evaluation.cloud_cpu_used_cycles_per_s
        )
        if cloud_used > cloud_capacity + 1.0 or cloud_used < -1.0:
            raise AssertionError("Cloud shared CPU exceeds 40 GHz")
        cache_cost = float(formal.cache_cost)
        movement_cost = float(formal.movement_cost)
        matching_net = float(formal.system_utility)
        independent_net = (
            float(new.task_utility_sum) - cache_cost - movement_cost
        )
        if not math.isclose(
            matching_net,
            float(formal.actual_task_utility) - cache_cost - movement_cost,
            rel_tol=0.0, abs_tol=1e-6,
        ):
            raise AssertionError("formal reward/cost identity failed")
        total_matching_net += matching_net
        total_new_net += independent_net
        total_prediction_oracles += decision.complete_assignment_oracle_calls
        changed_nodes = int(np.count_nonzero(
            np.any(prior_cache != decision.cache_state, axis=1)
        ))
        moved_uavs = int(sum(
            float(np.linalg.norm(decision.uav_positions[k] - prior_uav[k])) > 1.0e-6
            for k in prior_uav
        ))
        print(json.dumps({
            "mode": args.mode,
            "slot": slot,
            "time_s": time_s,
            "arrived_tasks": len(actual_tasks),
            "macro_frame_boundary": decision.cache_frame_boundary,
            "macro_lock_remaining_after": decision.next_state.cache_frame_slots_remaining,
            "cache_nodes_changed": changed_nodes,
            "uavs_moved": moved_uavs,
            "cache_cost": cache_cost,
            "uav_cost": movement_cost,
            "forecast_joint_net": decision.predicted_net_value,
            "forecast_keep_stay_net": decision.predicted_keep_stay_value,
            "forecast_joint_candidates": decision.screened_joint_candidates,
            "forecast_oracles": decision.complete_assignment_oracle_calls,
            "matching_realized_net": matching_net,
            "independent_realized_net": independent_net,
            "cloud_cpu_used_ghz": cloud_used / 1e9,
            "independent_oracle_calls": new.oracle_evaluations,
            "independent_stop_reason": new.stop_reason,
        }, ensure_ascii=False))
        state = decision.next_state

    print("SHORT_ROLLOUT_SUMMARY " + json.dumps({
        "mode": args.mode,
        "slots": args.slots,
        "cache_hold_slots": args.cache_hold_slots,
        "planning_horizon": args.horizon,
        "total_matching_realized_net": total_matching_net,
        "total_independent_realized_net": total_new_net,
        "prediction_assignment_oracle_calls": total_prediction_oracles,
        "wall_seconds": time.perf_counter() - started_at,
        "validation_scope": "short_training_realization_not_generalization",
    }, ensure_ascii=False))


if __name__ == "__main__":
    run()
