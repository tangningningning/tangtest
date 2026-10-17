"""Non-anticipative dual-timescale Cache/UAV receding-horizon controller.

This research controller implements a *fixed configurable* Macro Cache frame
(1..5 slots), while UAV positions are reconsidered at every slot. It does not
learn an adaptive Macro termination rule and does not solve a multistage
stochastic scenario tree. Forecasts are used ONLY to choose the current action.
Actual requests are passed to the separate realized-slot execution path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from decision import (
    cache_configuration,
    joint_held_optimizer,
    uav_configuration,
)


@dataclass(frozen=True)
class DualTimescaleState:
    cache_state: np.ndarray
    uav_positions: dict[str, np.ndarray]
    cache_frame_slots_remaining: int
    next_slot: int | None = None


@dataclass(frozen=True)
class DualTimescaleDecision:
    cache_state: np.ndarray
    uav_positions: dict[str, np.ndarray]
    next_state: DualTimescaleState
    cache_frame_boundary: bool
    cache_proposals: int
    uav_proposals: int
    predicted_net_value: float
    predicted_keep_stay_value: float
    screened_joint_candidates: int
    complete_assignment_oracle_calls: int


def initialize_controller(
    cache_state: np.ndarray,
    uav_positions: dict[str, np.ndarray],
) -> DualTimescaleState:
    return DualTimescaleState(
        cache_state=np.asarray(cache_state, dtype=np.int8).copy(),
        uav_positions={
            str(k): np.asarray(v, dtype=float).reshape(2).copy()
            for k, v in uav_positions.items()
        },
        cache_frame_slots_remaining=0,
        next_slot=None,
    )


def _cache_proposals(
    previous: np.ndarray,
    origin: dict[str, Any],
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    *,
    limit: int,
    horizon_mask: np.ndarray,
) -> list[np.ndarray]:
    """Select one strong anchor per Cache node; allow a coordinated pair.

    Node-specific demand only ranks proposals. Physical and objective validity
    are established later by the full joint value evaluator.
    """
    if limit <= 0:
        return []
    proposals: list[tuple[float, int, int, np.ndarray]] = []
    priority = {
        "prediction": 0,
        "coverage_anchor": 1,
        "intermediate": 2,
        "low_reconfiguration": 3,
        "exhaustive_feasible": 4,
    }
    for node_index in range(previous.shape[0]):
        enumerated = cache_configuration.enumerate_all_cache_target_configurations(
            previous, node_index, service_catalog, cache_nodes, origin, cache_config,
            prediction_horizon_mask=horizon_mask,
        )
        if not enumerated:
            continue
        demand = np.asarray(
            cache_configuration.node_specific_predicted_demand(
                origin, node_index, cache_nodes, cache_config,
                prediction_horizon_mask=horizon_mask,
            ),
            dtype=float,
        ).reshape(-1)
        if demand.shape != previous[node_index].shape:
            raise ValueError("predicted node demand has wrong service dimension")
        anchors = [p for p in enumerated if p.proposal_source != "exhaustive_feasible"]
        if not anchors:
            anchors = enumerated
        chosen = min(
            anchors,
            key=lambda p: (
                -float(np.dot(demand, p.target_node_cache - previous[node_index])),
                priority.get(p.proposal_source, 5),
                int(np.count_nonzero(p.target_node_cache != previous[node_index])),
                tuple(int(x) for x in p.target_node_cache),
            ),
        )
        next_cache = previous.copy()
        next_cache[node_index] = np.asarray(chosen.target_node_cache, dtype=np.int8)
        if np.array_equal(next_cache, previous):
            continue
        gain = float(np.dot(demand, next_cache[node_index] - previous[node_index]))
        changes = int(np.count_nonzero(next_cache != previous))
        proposals.append((gain, changes, node_index, next_cache))
    proposals.sort(key=lambda x: (-x[0], x[1], x[2]))
    single_budget = limit - 1 if limit >= 2 and len(proposals) >= 2 else limit
    selected = [row[3] for row in proposals[:single_budget]]
    # A coordinated proposal can reveal Cache complementarity that a
    # single-node move misses (e.g. replace complementary service replicas).
    if len(proposals) >= 2 and len(selected) < limit:
        combined = previous.copy()
        left, right = proposals[0], proposals[1]
        combined[left[2]] = left[3][left[2]]
        combined[right[2]] = right[3][right[2]]
        selected.append(combined)
    cache_configuration.validate_cache_shape_and_capacity(
        previous, service_catalog, cache_nodes
    )
    for proposed in selected:
        cache_configuration.validate_cache_shape_and_capacity(
            proposed, service_catalog, cache_nodes
        )
    return selected


def _uav_proposals(
    previous: dict[str, np.ndarray],
    cache_state: np.ndarray,
    origin: dict[str, Any],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    flight_boundary: tuple[float, float, float, float],
    *,
    limit: int,
    horizon_mask: np.ndarray,
) -> list[dict[str, np.ndarray]]:
    if limit <= 0 or not previous:
        return []
    xy = np.asarray(origin["positions_xy"], dtype=float)
    if xy.ndim != 4 or xy.shape[-1] != 2:
        raise ValueError("forecast positions shape must be [S,H,V,2]")
    mask_indices = np.flatnonzero(horizon_mask)
    if not len(mask_indices):
        raise ValueError("forecast horizon mask must not be empty")
    forecast_positions = np.mean(xy[:, int(mask_indices[0]), :, :], axis=0)
    vehicle_ids = [str(value) for value in origin["vehicle_ids"]]
    if forecast_positions.shape != (len(vehicle_ids), 2):
        raise ValueError("forecast vehicle positions are misaligned")
    predicted_vehicle_positions = {
        vid: forecast_positions[index] for index, vid in enumerate(vehicle_ids)
    }
    uav_ids = uav_configuration.uav_ids_from_project(project_cfg)
    if set(previous) != set(uav_ids):
        raise ValueError("UAV state does not match configured UAV IDs")
    proposals: list[dict[str, np.ndarray]] = []
    first_target: dict[str, np.ndarray] = {}

    def triggers(candidate: dict[str, np.ndarray]) -> dict[str, int]:
        tolerance = float(
            uav_configuration.uav_search_config_from_project(project_cfg).position_tolerance_m
        )
        return {
            uid: int(float(np.linalg.norm(candidate[uid] - previous[uid])) > tolerance)
            for uid in uav_ids
        }

    per_node_budget = limit - 1 if len(uav_ids) >= 2 and limit >= 2 else limit
    for uid in uav_ids:
        proposed_triggers = {u: int(u == uid) for u in uav_ids}
        suggested = uav_configuration.enumerate_uav_target_candidates(
            previous, previous, uid, proposed_triggers, project_cfg,
            flight_boundary, scenario_slice=origin, cache_state=cache_state,
            cache_nodes=cache_nodes,
            predicted_vehicle_positions=predicted_vehicle_positions,
            prediction_horizon_mask=horizon_mask,
        )
        for candidate in suggested[:2]:
            trial = {u: old.copy() for u, old in previous.items()}
            trial[uid] = np.asarray(candidate.target_position, dtype=float).copy()
            uav_configuration.validate_uav_configuration_transition(
                previous, trial, triggers(trial), project_cfg, flight_boundary
            )
            if uid not in first_target:
                first_target[uid] = trial[uid].copy()
            if len(proposals) < per_node_budget:
                proposals.append(trial)
    if len(first_target) >= 2 and len(proposals) < limit:
        trial = {u: old.copy() for u, old in previous.items()}
        trial.update(first_target)
        try:
            uav_configuration.validate_uav_configuration_transition(
                previous, trial, triggers(trial), project_cfg, flight_boundary
            )
        except ValueError:
            pass  # jointly colliding UAV targets are not legal candidates
        else:
            proposals.append(trial)
    return proposals


def decide_dual_timescale_action(
    state: DualTimescaleState,
    origin_scenario: dict[str, Any],
    service_catalog: dict[str, Any],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    flight_boundary: tuple[float, float, float, float],
    *,
    cache_hold_slots: int = 3,
    planning_horizon: int = 3,
    max_cache_proposals: int = 4,
    max_uav_proposals: int = 4,
    max_joint_candidates: int = 64,
    screen_scenarios: int = 1,
    full_scenarios: int = 2,
    screen_assignment_budget: int = 8,
    full_assignment_budget: int = 24,
) -> DualTimescaleDecision:
    """Choose the next action using forecasts only; advance Macro frame state.

    At a Macro boundary the Cache policy may UPDATE or KEEP and begins a
    new fixed-length frame. During the frame Cache is frozen, while Micro
    UAV decisions are reoptimized at every call. The choice at t uses
    forecasts rooted at t, not the realized future task stream.
    """
    if not 1 <= cache_hold_slots <= 5:
        raise ValueError("cache_hold_slots must be in [1,5]")
    if not 1 <= planning_horizon <= 5:
        raise ValueError("planning_horizon must be in [1,5]")
    if any(v < 0 for v in (max_cache_proposals, max_uav_proposals)):
        raise ValueError("proposal caps must be nonnegative")
    if not 0 <= state.cache_frame_slots_remaining < cache_hold_slots:
        raise ValueError("invalid Macro Cache frame state")
    current_slot = int(origin_scenario["origin_slot"])
    if state.next_slot is not None and current_slot != state.next_slot:
        raise ValueError("dual-timescale steps must be contiguous")
    old_cache = np.asarray(state.cache_state, dtype=np.int8).copy()
    old_uav = {str(k): np.asarray(v, dtype=float).copy()
               for k, v in state.uav_positions.items()}
    cache_configuration.validate_cache_shape_and_capacity(
        old_cache, service_catalog, cache_nodes
    )
    h = int(np.asarray(origin_scenario["arrival"]).shape[1])
    mask = np.zeros(h, dtype=bool)
    mask[:min(planning_horizon, h)] = True
    macro_boundary = state.cache_frame_slots_remaining == 0
    proposal_cache_nodes = uav_configuration.cache_nodes_with_uav_positions(
        cache_nodes, old_uav
    )
    cache_candidates = (
        _cache_proposals(
            old_cache, origin_scenario, service_catalog, proposal_cache_nodes, cache_config,
            limit=max_cache_proposals, horizon_mask=mask,
        )
        if macro_boundary
        else []
    )
    uav_candidates = _uav_proposals(
        old_uav, old_cache, origin_scenario, project_cfg, proposal_cache_nodes,
        flight_boundary, limit=max_uav_proposals, horizon_mask=mask,
    )
    count = (1 + len(cache_candidates)) * (1 + len(uav_candidates))
    if count > max_joint_candidates:
        raise ValueError(
            f"bounded candidate product {count} exceeds max_joint_candidates "
            f"{max_joint_candidates}"
        )
    joint = joint_held_optimizer.optimize_joint_held_action(
        old_cache, old_uav, cache_candidates, uav_candidates,
        origin_scenario, service_catalog, project_cfg, cache_nodes,
        cache_config, flight_boundary,
        max_joint_candidates=max_joint_candidates,
        screen_scenarios=screen_scenarios,
        full_scenarios=full_scenarios,
        final_candidate_count=3,
        screen_assignment_budget=screen_assignment_budget,
        full_assignment_budget=full_assignment_budget,
        horizon_mask=mask,
    )
    next_state = DualTimescaleState(
        cache_state=joint.cache_state.copy(),
        uav_positions={
            k: v.copy() for k, v in joint.uav_positions.items()
        },
        cache_frame_slots_remaining=(
            cache_hold_slots - 1
            if macro_boundary
            else state.cache_frame_slots_remaining - 1
        ),
        next_slot=current_slot + 1,
    )
    if not macro_boundary and not np.array_equal(next_state.cache_state, old_cache):
        raise AssertionError("Cache changed inside a locked Macro frame")
    return DualTimescaleDecision(
        cache_state=joint.cache_state.copy(),
        uav_positions={k: v.copy() for k, v in joint.uav_positions.items()},
        next_state=next_state,
        cache_frame_boundary=macro_boundary,
        cache_proposals=len(cache_candidates),
        uav_proposals=len(uav_candidates),
        predicted_net_value=joint.predicted_net_value,
        predicted_keep_stay_value=joint.baseline_predicted_net_value,
        screened_joint_candidates=joint.screened_candidates,
        complete_assignment_oracle_calls=joint.total_assignment_oracle_evaluations,
    )
