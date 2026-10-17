#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-slot system execution.

本模块把一个时隙内已经完成的模块按 12.pdf 时间语义串起来：

previous configuration state + external trigger Z(t) + prediction scenario slice
-> trigger-conditioned configuration
-> actual-state Transfer-Exchange Matching + KKT
-> actual task utility Q(t)
-> system utility G(t)
-> next configuration state

本模块不实现 PPO、不定义 observation/action distribution、不跑 episode。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from common import project_config
from common import system_model
from decision import cache_configuration, task_allocation, uav_configuration


@dataclass(frozen=True)
class SlotExecutionResult:
    slot_id: int | None
    time_s: float | None
    step_feasible: bool
    infeasible_reason: str
    input_cache_triggers: list[int]
    input_uav_triggers: dict[str, int]
    previous_cache_state: np.ndarray
    previous_uav_positions: dict[str, tuple[float, float]]
    final_cache_state: np.ndarray | None
    final_uav_positions: dict[str, tuple[float, float]] | None
    predictive_configuration_utility: float | None
    predictive_configuration_objective: float | None
    cache_cost: float | None
    movement_energy: float | None
    movement_cost: float | None
    actual_task_utility: float | None
    system_utility: float | None
    matching: dict[str, str]
    cpu_allocations: dict[str, dict[str, float]]
    bandwidth_allocations: dict[str, dict[str, float]]
    per_task_delay_s: dict[str, float]
    per_task_vehicle_energy_j: dict[str, float]
    per_task_utility: dict[str, float]
    configuration_summary: dict[str, Any]
    matching_summary: dict[str, Any]
    next_cache_state: np.ndarray | None
    next_uav_positions: dict[str, tuple[float, float]] | None
    completed_task_records: list[dict[str, Any]] = field(default_factory=list)
    task_execution_diagnostics: dict[str, Any] = field(default_factory=dict)
    active_task_count: int = 0


class CacheLockConflict(RuntimeError):
    """A policy Cache transition would evict a service used by an active task."""


class UnresolvedEpisodeTasksError(RuntimeError):
    """An Episode ended while one or more tasks still needed execution."""


@dataclass
class TaskExecutionState:
    task_id: str
    vehicle_id: str
    service_id: int
    assigned_node: str
    arrival_time_s: float
    original_input_bits: float
    original_cpu_cycles: float
    remaining_input_bits: float
    remaining_cpu_cycles: float
    execution_stage: str
    accumulated_vehicle_energy_j: float = 0.0
    completion_time_s: float | None = None
    remaining_network_delay_s: float = 0.0
    cloud_upload_elapsed_s: float = 0.0
    accumulated_response_delay_s: float = 0.0
    slot_delay_s: float = 0.0
    slot_energy_j: float = 0.0
    completed_this_slot: bool = False

    def reset_slot_counters(self) -> None:
        self.slot_delay_s = 0.0
        self.slot_energy_j = 0.0
        self.completed_this_slot = False


@dataclass
class CrossSlotTaskLedger:
    """Episode-local state for persistent tasks and exactly-once completion."""

    episode_id: str
    active: dict[str, TaskExecutionState] = field(default_factory=dict)
    completed: dict[str, TaskExecutionState] = field(default_factory=dict)

    def add_arrivals(self, tasks: list[task_allocation.Task], slot: int, time_s: float) -> list[TaskExecutionState]:
        created: list[TaskExecutionState] = []
        for task in tasks:
            task_id = f"episode-{self.episode_id}:slot-{int(slot)}:vehicle-{task.vehicle_id}"
            if task_id in self.active or task_id in self.completed:
                raise ValueError(f"duplicate cross-slot task id: {task_id}")
            state = TaskExecutionState(
                task_id=task_id,
                vehicle_id=str(task.vehicle_id),
                service_id=int(task.service_id),
                assigned_node="",
                arrival_time_s=float(time_s),
                original_input_bits=float(task.input_bits),
                original_cpu_cycles=float(task.cpu_cycles),
                remaining_input_bits=float(task.input_bits),
                remaining_cpu_cycles=float(task.cpu_cycles),
                execution_stage="NEW",
            )
            self.active[task_id] = state
            created.append(state)
        return created


def active_cache_service_locks(
    ledger: CrossSlotTaskLedger,
    cache_state: np.ndarray,
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    *,
    expected_node_ids: tuple[str, ...] | None = None,
    expected_service_ids: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Return the per-node services used by unfinished, assigned Edge tasks.

    The ledger is the sole task-state source. The function also checks that its
    service indices and node identities use the current formal catalog/order,
    and that every active Edge dependency is present in the current Cache.
    """

    cache = np.asarray(cache_state)
    node_ids = tuple(str(value) for value in cache_nodes["node_ids"])
    service_ids = tuple(
        int(value) for value in np.asarray(service_catalog["service_ids"]).tolist()
    )
    service_count = int(service_catalog["num_services"])
    if len(service_ids) != service_count or service_ids != tuple(range(service_count)):
        raise ValueError("formal service catalog IDs must match contiguous Cache service order")
    if expected_node_ids is not None and tuple(expected_node_ids) != node_ids:
        raise ValueError("Cache node order differs from the formal action specification")
    if expected_service_ids is not None and tuple(expected_service_ids) != service_ids:
        raise ValueError("service catalog order differs from the formal observation specification")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Cache node IDs must be unique")
    if cache.shape != (len(node_ids), service_count):
        raise ValueError("Cache state shape differs from current node/service order")
    cache_configuration.validate_cache_shape_and_capacity(
        cache, service_catalog, cache_nodes
    )
    node_index_by_id = {node_id: index for index, node_id in enumerate(node_ids)}
    locks = np.zeros(cache.shape, dtype=bool)
    for task_id, state in ledger.active.items():
        if str(task_id) != str(state.task_id):
            raise ValueError(f"active task ledger key mismatch: {task_id}")
        if state.execution_stage == "COMPLETED":
            raise ValueError(f"completed task remains in active ledger: {task_id}")
        service_id = int(state.service_id)
        if service_id < 0 or service_id >= service_count:
            raise ValueError(
                f"active task {task_id} references unknown service ID {service_id}"
            )
        assigned_node = str(state.assigned_node)
        if not assigned_node:
            raise ValueError(f"active task {task_id} has no assigned execution node")
        if assigned_node == task_allocation.CLOUD:
            continue
        if assigned_node not in node_index_by_id:
            raise ValueError(
                f"active task {task_id} references unknown Cache node {assigned_node}"
            )
        locks[node_index_by_id[assigned_node], service_id] = True

    missing = np.argwhere(locks & (cache != 1))
    if len(missing):
        node_index, service_id = (int(value) for value in missing[0])
        task_ids = sorted(
            str(task_id)
            for task_id, state in ledger.active.items()
            if str(state.assigned_node) == node_ids[node_index]
            and int(state.service_id) == service_id
            and str(state.assigned_node) != task_allocation.CLOUD
        )
        raise CacheLockConflict(
            "active-task/cache state is inconsistent: required service is absent "
            f"from its assigned node (node={node_ids[node_index]}, service={service_id}, "
            f"tasks={task_ids})"
        )
    return locks


def _task_from_execution_state(
    state: TaskExecutionState,
    transmit_time_share: float,
) -> task_allocation.Task:
    if state.execution_stage == "COMPUTING":
        input_bits = 0.0
        cpu_cycles = state.remaining_cpu_cycles
    elif state.execution_stage == "NETWORK_DELAY":
        input_bits = 0.0
        cpu_cycles = state.remaining_cpu_cycles
    else:
        input_bits = state.remaining_input_bits
        cpu_cycles = state.original_cpu_cycles
    return task_allocation.Task(
        task_id=state.task_id,
        vehicle_id=state.vehicle_id,
        service_id=state.service_id,
        input_bits=float(input_bits),
        cpu_cycles=float(cpu_cycles),
        execution_stage=state.execution_stage,
        transmit_time_share=float(transmit_time_share),
        remaining_network_delay_s=float(state.remaining_network_delay_s),
    )


def _transmitter_time_shares(
    states: list[TaskExecutionState],
) -> dict[str, float]:
    counts: dict[str, int] = {}
    for state in states:
        if state.execution_stage in {"NEW", "UPLOADING"}:
            counts[state.vehicle_id] = counts.get(state.vehicle_id, 0) + 1
    return {
        state.task_id: (1.0 / counts[state.vehicle_id])
        if state.execution_stage in {"NEW", "UPLOADING"}
        else 1.0
        for state in states
    }


def _finish_execution_state(
    ledger: CrossSlotTaskLedger,
    state: TaskExecutionState,
    completion_time_s: float,
    completed_this_slot: list[TaskExecutionState],
) -> None:
    if state.task_id not in ledger.active or state.task_id in ledger.completed:
        raise AssertionError(f"task completion was recorded more than once: {state.task_id}")
    state.execution_stage = "COMPLETED"
    state.completion_time_s = float(completion_time_s)
    state.accumulated_response_delay_s = float(completion_time_s - state.arrival_time_s)
    state.remaining_input_bits = 0.0
    state.remaining_cpu_cycles = 0.0
    state.remaining_network_delay_s = 0.0
    state.completed_this_slot = True
    ledger.completed[state.task_id] = state
    del ledger.active[state.task_id]
    completed_this_slot.append(state)


def _execute_cross_slot_tasks(
    configuration: JointConfigurationResult,
    actual_tasks: list[task_allocation.Task],
    actual_vehicle_positions: dict[str, np.ndarray],
    service_catalog: dict[str, Any],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    slot_id: int | None,
    time_s: float | None,
    ledger: CrossSlotTaskLedger,
    performance_profile: dict[str, Any] | None,
    exact_q_timing_profile: dict[str, float] | None,
) -> SlotExecutionResult:
    """Match new arrivals, then advance all assigned work through this slot."""
    if slot_id is None or time_s is None:
        raise ValueError("cross-slot execution requires slot_id and time_s")
    slot_start = float(time_s)
    slot_end = slot_start + 1.0
    historical_ids = set(ledger.active)
    for state in ledger.active.values():
        state.reset_slot_counters()
        if not state.assigned_node:
            raise ValueError(f"historical task {state.task_id} has no assigned node")

    final_cache = np.asarray(configuration.final_cache_state, dtype=np.int8).copy()
    previous_cache = np.asarray(configuration.previous_cache_state, dtype=np.int8).copy()
    node_ids = [str(value) for value in cache_nodes["node_ids"]]
    for task_id in historical_ids:
        state = ledger.active[task_id]
        if state.assigned_node == task_allocation.CLOUD:
            continue
        if state.assigned_node not in node_ids:
            raise ValueError(f"historical task {task_id} references missing node {state.assigned_node}")
        node_index = node_ids.index(state.assigned_node)
        if int(final_cache[node_index, state.service_id]) != 1:
            raise CacheLockConflict(
                "Cache transition would evict a service used by an active task: "
                f"task={task_id}, node={state.assigned_node}, service={state.service_id}"
            )

    new_states = ledger.add_arrivals(actual_tasks, int(slot_id), slot_start)
    participating_states = {task_id: ledger.active[task_id] for task_id in (*historical_ids, *(state.task_id for state in new_states))}
    required_vehicles = {state.vehicle_id for state in participating_states.values()}
    missing_positions = required_vehicles - set(actual_vehicle_positions)
    if missing_positions:
        raise ValueError(
            "current trajectory is missing an active task vehicle position: "
            f"{sorted(missing_positions)}"
        )
    positions = {
        str(vehicle_id): np.asarray(xy, dtype=np.float64)
        for vehicle_id, xy in actual_vehicle_positions.items()
    }
    service_positions = task_allocation.build_service_node_positions(
        cache_nodes,
        positions,
        {key: np.asarray(xy, dtype=np.float64) for key, xy in configuration.final_uav_positions.items()},
    )
    shares = _transmitter_time_shares(list(participating_states.values()))
    matching_tasks = [
        _task_from_execution_state(state, shares[state.task_id])
        for state in participating_states.values()
    ]
    fixed_matching = {
        task_id: ledger.active[task_id].assigned_node for task_id in historical_ids
    }
    mutable_task_ids = {state.task_id for state in new_states}
    matching_started = time.perf_counter()
    matching_result = task_allocation.optimize_task_matching(
        matching_tasks,
        positions,
        service_positions,
        final_cache,
        project_cfg,
        cache_nodes,
        cache_config,
        kkt_cache=None,
        node_evaluation_cache=None,
        timing_profile=exact_q_timing_profile,
        cloud_fallback_mode=cache_config.cloud_fallback_mode,
        fixed_matching=fixed_matching,
        mutable_task_ids=mutable_task_ids,
    )
    matching_seconds = time.perf_counter() - matching_started
    if not matching_result.resource_feasible:
        raise RuntimeError(
            "cross-slot Matching/KKT was infeasible: "
            f"{matching_result.infeasible_reason}"
        )
    if set(matching_result.matching) != set(participating_states):
        raise AssertionError("cross-slot Matching omitted or duplicated a participating task")
    for task_id, target in fixed_matching.items():
        if matching_result.matching[task_id] != target:
            raise AssertionError(f"Matching changed a historical task node: {task_id}")
    for state in new_states:
        state.assigned_node = matching_result.matching[state.task_id]
        state.execution_stage = "UPLOADING"

    participating_states = {task_id: ledger.active[task_id] for task_id in participating_states}
    completed_this_slot: list[TaskExecutionState] = []
    current_time = slot_start
    event_count = 0
    max_cpu_sum: dict[str, float] = {}
    max_bandwidth_sum: dict[str, float] = {}
    max_cloud_cpu_used = 0.0
    max_tv_share_sum = 0.0
    shared_nodes: set[str] = set()
    shared_resource_event_count = 0
    shared_cpu_event_count = 0
    shared_bandwidth_event_count = 0
    last_cpu_allocations: dict[str, dict[str, float]] = {}
    last_bandwidth_allocations: dict[str, dict[str, float]] = {}

    while current_time < slot_end - 1e-12 and ledger.active:
        # Resolve zero-work phase boundaries before asking KKT for resources.
        immediate_transition = False
        for state in list(ledger.active.values()):
            if state.execution_stage == "UPLOADING" and state.remaining_input_bits <= 1e-6:
                state.remaining_input_bits = 0.0
                if state.assigned_node == task_allocation.CLOUD and state.cloud_upload_elapsed_s > 0.0:
                    state.execution_stage = "NETWORK_DELAY"
                    state.remaining_network_delay_s = max(
                        0.0,
                        (cache_config.cloud_delay_factor - 1.0) * state.cloud_upload_elapsed_s,
                    )
                else:
                    state.execution_stage = "COMPUTING"
                immediate_transition = True
            elif state.execution_stage == "NETWORK_DELAY" and state.remaining_network_delay_s <= 1e-12:
                state.remaining_network_delay_s = 0.0
                state.execution_stage = "COMPUTING"
                immediate_transition = True
            elif state.execution_stage == "COMPUTING" and state.remaining_cpu_cycles <= 1e-3:
                _finish_execution_state(ledger, state, current_time, completed_this_slot)
                immediate_transition = True
            elif state.execution_stage == "NEW":
                raise AssertionError(f"new task {state.task_id} was not moved to UPLOADING")
        if immediate_transition:
            continue

        active_states = list(ledger.active.values())
        active_shares = _transmitter_time_shares(active_states)
        active_tasks = [
            _task_from_execution_state(state, active_shares[state.task_id])
            for state in active_states
        ]
        active_matching = {task.task_id: ledger.active[task.task_id].assigned_node for task in active_tasks}
        allocation = task_allocation.allocate_resources_kkt(
            active_tasks,
            active_matching,
            positions,
            service_positions,
            project_cfg,
            cache_nodes,
            cache_config,
            kkt_cache=None,
        )
        if not allocation.feasible:
            raise RuntimeError(f"cross-slot event KKT failed: {allocation.infeasible_reason}")
        cloud_computing_tasks = [
            task for task in active_tasks
            if active_matching[task.task_id] == task_allocation.CLOUD
            and task.execution_stage == "COMPUTING"
        ]
        cloud_allocation = task_allocation._allocate_cloud_cpu_kkt(
            cloud_computing_tasks,
            cache_config,
            project_cfg,
            None,
        )
        if not cloud_allocation.feasible:
            raise RuntimeError(f"cross-slot Cloud CPU KKT failed: {cloud_allocation.infeasible_reason}")
        cloud_cpu_rates = cloud_allocation.cpu_allocations.get(task_allocation.CLOUD, {})
        cloud_cpu_used = float(math.fsum(cloud_cpu_rates.values()))
        if cache_config.cloud_resource_mode == task_allocation.FINITE_CLOUD_RESOURCE_MODE and (
            cloud_cpu_used > float(cache_config.cloud_cpu_capacity_cycles_per_s) + 1e-5
        ):
            raise AssertionError("Cloud CPU allocation exceeded its finite shared pool")
        if any(not item.finite for item in (*allocation.kkt_diagnostics, *cloud_allocation.kkt_diagnostics)):
            raise FloatingPointError("cross-slot KKT returned a non-finite diagnostic")
        expected_cpu_ids: dict[str, set[str]] = {}
        expected_upload_ids: dict[str, set[str]] = {}
        for state in active_states:
            if state.assigned_node == task_allocation.CLOUD:
                continue
            if state.execution_stage == "COMPUTING":
                expected_cpu_ids.setdefault(state.assigned_node, set()).add(state.task_id)
            elif state.execution_stage == "UPLOADING":
                expected_upload_ids.setdefault(state.assigned_node, set()).add(state.task_id)
        for node_id, assigned in allocation.cpu_allocations.items():
            if set(assigned) != expected_cpu_ids.get(node_id, set()):
                raise AssertionError(f"{node_id} CPU allocation includes a non-computing task")
        for node_id, assigned in allocation.bandwidth_allocations.items():
            if set(assigned) != expected_upload_ids.get(node_id, set()):
                raise AssertionError(f"{node_id} bandwidth allocation includes a non-uploading task")
        expected_cloud_cpu = {
            state.task_id for state in active_states
            if state.assigned_node == task_allocation.CLOUD
            and state.execution_stage == "COMPUTING"
        }
        if set(cloud_cpu_rates) != expected_cloud_cpu:
            raise AssertionError("Cloud CPU allocation includes a non-computing task")
        max_cloud_cpu_used = max(max_cloud_cpu_used, cloud_cpu_used)
        last_cpu_allocations = allocation.cpu_allocations
        last_bandwidth_allocations = allocation.bandwidth_allocations
        for node_id, total in allocation.per_node_cpu_sum.items():
            max_cpu_sum[node_id] = max(max_cpu_sum.get(node_id, 0.0), float(total))
        for node_id, total in allocation.per_node_bandwidth_sum.items():
            max_bandwidth_sum[node_id] = max(max_bandwidth_sum.get(node_id, 0.0), float(total))
        tv_share_sums: dict[str, float] = {}
        for state in active_states:
            if state.execution_stage == "UPLOADING":
                tv_share_sums[state.vehicle_id] = tv_share_sums.get(state.vehicle_id, 0.0) + active_shares[state.task_id]
        max_tv_share_sum = max(max_tv_share_sum, max(tv_share_sums.values(), default=0.0))
        if max_cpu_sum and any(value > 1.0 + 1e-8 for value in max_cpu_sum.values()):
            raise AssertionError("edge CPU fractions exceeded one")
        if max_bandwidth_sum and any(value > 1.0 + 1e-8 for value in max_bandwidth_sum.values()):
            raise AssertionError("edge bandwidth fractions exceeded one")
        if any(value > 1.0 + 1e-12 for value in tv_share_sums.values()):
            raise AssertionError("a TV was scheduled to transmit more than one task at once")

        cpu_tasks_by_node: dict[str, list[TaskExecutionState]] = {}
        upload_tasks_by_node: dict[str, list[TaskExecutionState]] = {}
        for state in active_states:
            if state.execution_stage == "COMPUTING" and state.assigned_node != task_allocation.CLOUD:
                cpu_tasks_by_node.setdefault(state.assigned_node, []).append(state)
            elif state.execution_stage == "UPLOADING" and state.assigned_node != task_allocation.CLOUD:
                upload_tasks_by_node.setdefault(state.assigned_node, []).append(state)
        active_historical_ids = historical_ids.intersection(ledger.active)
        for node_id in set(cpu_tasks_by_node) | set(upload_tasks_by_node):
            cpu_node_states = cpu_tasks_by_node.get(node_id, [])
            upload_node_states = upload_tasks_by_node.get(node_id, [])
            cpu_competition = (
                any(state.task_id in active_historical_ids for state in cpu_node_states)
                and any(state.task_id in mutable_task_ids for state in cpu_node_states)
            )
            bandwidth_competition = (
                any(state.task_id in active_historical_ids for state in upload_node_states)
                and any(state.task_id in mutable_task_ids for state in upload_node_states)
            )
            if cpu_competition or bandwidth_competition:
                shared_nodes.add(node_id)
                shared_resource_event_count += 1
                shared_cpu_event_count += int(cpu_competition)
                shared_bandwidth_event_count += int(bandwidth_competition)

        time_to_completion: dict[str, float] = {}
        upload_rates: dict[str, float] = {}
        cpu_rates: dict[str, float] = {}
        for state in active_states:
            if state.execution_stage == "UPLOADING":
                share = active_shares[state.task_id]
                if state.assigned_node == task_allocation.CLOUD:
                    rate = float(cache_config.cloud_total_bandwidth_bps) * share
                else:
                    role = task_allocation._edge_node_role(state.assigned_node, cache_nodes)
                    assert role is not None
                    bandwidth_fraction = allocation.bandwidth_allocations.get(state.assigned_node, {}).get(state.task_id)
                    if bandwidth_fraction is None:
                        raise AssertionError(f"upload task lacks bandwidth KKT allocation: {state.task_id}")
                    gain = task_allocation._channel_gain(
                        role,
                        positions[state.vehicle_id],
                        service_positions[state.assigned_node],
                        cache_config,
                    )
                    base_rate = system_model.uplink_rate(
                        task_allocation._node_bandwidth(role, cache_config),
                        bandwidth_fraction,
                        cache_config.tx_power_w,
                        gain,
                        cache_config.noise_power_w,
                        noise_model=cache_config.noise_model,
                        noise_psd_w_per_hz=cache_config.noise_psd_w_per_hz,
                        receiver_noise_figure_db=cache_config.receiver_noise_figure_db,
                    )
                    rate = float(base_rate) * share
                if not math.isfinite(rate) or rate <= 0.0:
                    raise ValueError(f"task {state.task_id} has no valid current uplink")
                upload_rates[state.task_id] = rate
                time_to_completion[state.task_id] = state.remaining_input_bits / rate
            elif state.execution_stage == "COMPUTING":
                if state.assigned_node == task_allocation.CLOUD:
                    cpu_rate = cloud_cpu_rates.get(state.task_id)
                    if cpu_rate is None:
                        raise AssertionError(f"Cloud computing task lacks CPU allocation: {state.task_id}")
                else:
                    role = task_allocation._edge_node_role(state.assigned_node, cache_nodes)
                    assert role is not None
                    cpu_fraction = allocation.cpu_allocations.get(state.assigned_node, {}).get(state.task_id)
                    if cpu_fraction is None:
                        raise AssertionError(f"edge computing task lacks CPU KKT allocation: {state.task_id}")
                    cpu_rate = task_allocation._node_cpu(role, cache_config) * cpu_fraction
                if not math.isfinite(float(cpu_rate)) or float(cpu_rate) <= 0.0:
                    raise ValueError(f"task {state.task_id} has no valid current CPU allocation")
                cpu_rates[state.task_id] = float(cpu_rate)
                time_to_completion[state.task_id] = state.remaining_cpu_cycles / float(cpu_rate)
            elif state.execution_stage == "NETWORK_DELAY":
                time_to_completion[state.task_id] = state.remaining_network_delay_s
            else:
                raise AssertionError(f"unsupported active task stage: {state.execution_stage}")

        duration = min(slot_end - current_time, min(time_to_completion.values(), default=slot_end - current_time))
        if not math.isfinite(duration) or duration <= 0.0:
            raise AssertionError("cross-slot event loop failed to advance")
        for state in active_states:
            state.slot_delay_s += duration
            state.accumulated_response_delay_s += duration
            if state.execution_stage == "UPLOADING":
                uploaded = min(state.remaining_input_bits, upload_rates[state.task_id] * duration)
                state.remaining_input_bits = max(0.0, state.remaining_input_bits - uploaded)
                duty = active_shares[state.task_id]
                energy = float(cache_config.tx_power_w) * duty * duration
                state.slot_energy_j += energy
                state.accumulated_vehicle_energy_j += energy
                if state.assigned_node == task_allocation.CLOUD:
                    state.cloud_upload_elapsed_s += duration
            elif state.execution_stage == "COMPUTING":
                consumed = min(state.remaining_cpu_cycles, cpu_rates[state.task_id] * duration)
                state.remaining_cpu_cycles = max(0.0, state.remaining_cpu_cycles - consumed)
            elif state.execution_stage == "NETWORK_DELAY":
                state.remaining_network_delay_s = max(0.0, state.remaining_network_delay_s - duration)
        current_time += duration
        event_count += 1

    for state in list(ledger.active.values()):
        if state.execution_stage in {"UPLOADING", "COMPUTING"}:
            if state.execution_stage == "UPLOADING" and state.remaining_input_bits <= 1e-6:
                state.remaining_input_bits = 0.0
                if state.assigned_node == task_allocation.CLOUD and state.cloud_upload_elapsed_s > 0.0:
                    state.execution_stage = "NETWORK_DELAY"
                    state.remaining_network_delay_s = max(0.0, (cache_config.cloud_delay_factor - 1.0) * state.cloud_upload_elapsed_s)
                else:
                    state.execution_stage = "COMPUTING"
            elif state.execution_stage == "COMPUTING" and state.remaining_cpu_cycles <= 1e-3:
                _finish_execution_state(ledger, state, min(current_time, slot_end), completed_this_slot)
        elif state.execution_stage == "NETWORK_DELAY" and state.remaining_network_delay_s <= 1e-12:
            state.remaining_network_delay_s = 0.0
            state.execution_stage = "COMPUTING"

    per_task_delay = {task_id: state.slot_delay_s for task_id, state in participating_states.items()}
    per_task_energy = {task_id: state.slot_energy_j for task_id, state in participating_states.items()}
    per_task_utility: dict[str, float] = {}
    for task_id, state in participating_states.items():
        per_task_utility[task_id] = float(
            int(state.completed_this_slot)
            - cache_config.omega_t * state.slot_delay_s / cache_config.time_ref_s
            - cache_config.omega_e * state.slot_energy_j / cache_config.energy_ref_j
        )
    actual_task_utility = float(math.fsum(per_task_utility.values()))
    cache_cost = float(configuration.cache_cost)
    movement_cost = float(configuration.movement_cost)
    system_utility = actual_task_utility - cache_cost - movement_cost
    summary = _matching_summary(matching_result, cache_nodes, cache_config)
    summary["cross_slot_execution"] = {
        "historical_tasks_at_slot_start": len(historical_ids),
        "new_tasks": len(new_states),
        "tasks_in_matching": len(matching_result.matching),
        "resource_event_count": event_count,
        "old_new_shared_resource_event_count": shared_resource_event_count,
        "old_new_shared_cpu_event_count": shared_cpu_event_count,
        "old_new_shared_bandwidth_event_count": shared_bandwidth_event_count,
        "cpu_allocations_match_computing_phase": True,
        "bandwidth_allocations_match_uploading_phase": True,
        "old_new_shared_node_ids": sorted(shared_nodes),
        "max_edge_cpu_fraction_sum": max_cpu_sum,
        "max_edge_bandwidth_fraction_sum": max_bandwidth_sum,
        "max_cloud_cpu_used_cycles_per_s": max_cloud_cpu_used,
        "cloud_cpu_capacity_cycles_per_s": (
            float(cache_config.cloud_cpu_capacity_cycles_per_s)
            if cache_config.cloud_resource_mode == task_allocation.FINITE_CLOUD_RESOURCE_MODE
            else None
        ),
        "max_same_tv_transmit_share_sum": max_tv_share_sum,
        "completed_this_slot": len(completed_this_slot),
        "active_tasks_at_slot_end": len(ledger.active),
        "active_stage_counts": {
            stage: sum(state.execution_stage == stage for state in ledger.active.values())
            for stage in ("UPLOADING", "COMPUTING", "NETWORK_DELAY")
        },
        "historical_node_assignments_unchanged": all(
            ledger.completed.get(task_id, ledger.active.get(task_id)).assigned_node == target
            for task_id, target in fixed_matching.items()
        ),
    }
    completion_records = [
        {
            "task_id": state.task_id,
            "vehicle_id": state.vehicle_id,
            "service_id": state.service_id,
            "assigned_node": state.assigned_node,
            "arrival_time_s": state.arrival_time_s,
            "completion_time_s": state.completion_time_s,
            "response_delay_s": state.accumulated_response_delay_s,
            "accumulated_vehicle_energy_j": state.accumulated_vehicle_energy_j,
        }
        for state in completed_this_slot
    ]
    if performance_profile is not None:
        performance_profile["matching_kkt_seconds"] = float(matching_seconds)
        performance_profile["config_seconds"] = 0.0
    return SlotExecutionResult(
        slot_id=int(slot_id),
        time_s=slot_start,
        step_feasible=True,
        infeasible_reason="",
        input_cache_triggers=np.any(final_cache != previous_cache, axis=1).astype(np.int8).tolist(),
        input_uav_triggers={
            str(key): int(value) for key, value in configuration.uav_triggers.items()
        },
        previous_cache_state=previous_cache,
        previous_uav_positions=dict(configuration.previous_uav_positions),
        final_cache_state=final_cache,
        final_uav_positions=dict(configuration.final_uav_positions),
        predictive_configuration_utility=float(configuration.predictive_utility),
        predictive_configuration_objective=float(configuration.objective_value),
        cache_cost=cache_cost,
        movement_energy=float(configuration.movement_energy),
        movement_cost=movement_cost,
        actual_task_utility=actual_task_utility,
        system_utility=float(system_utility),
        matching=dict(matching_result.matching),
        cpu_allocations=last_cpu_allocations,
        bandwidth_allocations=last_bandwidth_allocations,
        per_task_delay_s=per_task_delay,
        per_task_vehicle_energy_j=per_task_energy,
        per_task_utility=per_task_utility,
        configuration_summary=_configuration_summary(
            configuration,
            float(project_config.get_config_value(
                project_cfg, "algorithm.uav.position_tolerance_m", 1e-6
            )),
        ),
        matching_summary=summary,
        next_cache_state=final_cache.copy(),
        next_uav_positions=dict(configuration.final_uav_positions),
        completed_task_records=completion_records,
        task_execution_diagnostics=summary["cross_slot_execution"],
        active_task_count=len(ledger.active),
    )


def _position_tuples(positions: dict[str, Any]) -> dict[str, tuple[float, float]]:
    return {
        str(node_id): (float(np.asarray(xy)[0]), float(np.asarray(xy)[1]))
        for node_id, xy in positions.items()
    }


def _matching_summary(
    result: task_allocation.TransferExchangeMatchingResult,
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
) -> dict[str, Any]:
    node_ids = [str(value) for value in cache_nodes["node_ids"]]
    node_roles = {
        node_id: str(role)
        for node_id, role in zip(node_ids, cache_nodes["node_roles"], strict=True)
    }
    counts_by_node = {node_id: 0 for node_id in node_ids}
    cloud_count = 0
    for target in result.matching.values():
        if target == task_allocation.CLOUD:
            cloud_count += 1
        elif target in counts_by_node:
            counts_by_node[target] += 1
        else:
            raise ValueError(f"matching summary 遇到未知执行节点：{target}")

    counts_by_role = {
        role: {
            node_id: int(counts_by_node[node_id])
            for node_id in node_ids
            if node_roles[node_id] == role
        }
        for role in ("SV", "RSU", "UAV")
    }
    uav_ids = list(counts_by_role["UAV"])
    edge_cpu_by_node: dict[str, float] = {}
    for node_id, fraction_by_task in result.cpu_allocations.items():
        role = node_roles.get(node_id)
        if role is None:
            continue
        edge_cpu_by_node[node_id] = float(
            math.fsum(fraction_by_task.values())
            * task_allocation._node_cpu(role, cache_config)
        )
    edge_cpu_total = float(math.fsum(edge_cpu_by_node.values()))
    task_count = len(result.matching)
    mean_delay = (
        float(math.fsum(result.per_task_delay_s.values()) / task_count)
        if task_count
        else None
    )
    mean_energy = (
        float(math.fsum(result.per_task_vehicle_energy_j.values()) / task_count)
        if task_count
        else None
    )
    mean_utility = (
        float(math.fsum(result.per_task_utility.values()) / task_count)
        if task_count
        else None
    )
    summary = {
        "resource_feasible": bool(result.resource_feasible),
        "updates": result.updates,
        "transfer_updates": result.transfer_updates,
        "exchange_updates": result.exchange_updates,
        "candidate_evaluations_total": result.candidate_evaluations_total,
        "candidate_definitions_total": result.candidate_definitions_total,
        "upper_bound_screen_calls": result.upper_bound_screen_calls,
        "upper_bound_pruned_candidates": result.upper_bound_pruned_candidates,
        "transfer_upper_bound_pruned": result.transfer_upper_bound_pruned,
        "exchange_upper_bound_pruned": result.exchange_upper_bound_pruned,
        "full_evaluator_calls": result.full_evaluator_calls,
        "kkt_cache_requests": result.kkt_cache_requests,
        "kkt_cache_hits": result.kkt_cache_hits,
        "kkt_cache_misses": result.kkt_cache_misses,
        "KKT_CACHE_REQUEST_COUNT": result.kkt_cache_requests,
        "KKT_CACHE_HIT_COUNT": result.kkt_cache_hits,
        "KKT_CACHE_MISS_COUNT": result.kkt_cache_misses,
        "KKT_SOLVER_CALL_COUNT": result.kkt_solver_calls,
        "KKT_SOLVE_COUNT": result.kkt_solve_count,
        "KKT_CONVERGED_COUNT": result.kkt_converged_count,
        "KKT_MAX_ITER_HIT_COUNT": result.kkt_max_iterations_hit_count,
        "KKT_FINITE_COUNT": result.kkt_finite_count,
        "KKT_ITERATION_MEAN": result.kkt_iteration_mean,
        "KKT_ITERATION_P95": result.kkt_iteration_p95,
        "KKT_ITERATION_MAX": result.kkt_iteration_max,
        "KKT_RESOURCE_RESIDUAL_MEAN": result.kkt_resource_residual_mean,
        "KKT_RESOURCE_RESIDUAL_MAX": result.kkt_resource_residual_max,
        "KKT_STATIONARITY_RESIDUAL_MEAN": result.kkt_stationarity_residual_mean,
        "KKT_STATIONARITY_RESIDUAL_MAX": result.kkt_stationarity_residual_max,
        "KKT_COMPLEMENTARY_SLACKNESS_RESIDUAL_MEAN": (
            result.kkt_complementary_slackness_residual_mean
        ),
        "KKT_COMPLEMENTARY_SLACKNESS_RESIDUAL_MAX": (
            result.kkt_complementary_slackness_residual_max
        ),
        "KKT_SOLVE_DIAGNOSTICS": [
            dict(record) for record in result.kkt_solve_diagnostics
        ],
        "NODE_EVAL_CACHE_REQUEST_COUNT": result.node_eval_cache_requests,
        "NODE_EVAL_CACHE_HIT_COUNT": result.node_eval_cache_hits,
        "NODE_EVAL_CACHE_MISS_COUNT": result.node_eval_cache_misses,
        "NODE_EVAL_FULL_EXECUTION_COUNT": result.node_eval_full_executions,
        "IDEAL_UTILITY_REQUEST_COUNT": result.ideal_utility_cache_requests,
        "IDEAL_UTILITY_CACHE_HIT_COUNT": result.ideal_utility_cache_hits,
        "IDEAL_UTILITY_CACHE_MISS_COUNT": result.ideal_utility_cache_misses,
        "IMPACT_SCOPE_REQUEST_COUNT": result.impact_scope_requests,
        "IMPACT_SCOPE_COMPUTE_COUNT": result.impact_scope_computes,
        "IMPACT_SCOPE_REUSE_COUNT": result.impact_scope_reuses,
        "stop_reason": result.stop_reason,
        # Diagnostic aliases only: the Matching decision logic continues to
        # use the original stop_reason values above.
        "MAX_UPDATES_REACHED": result.stop_reason == "stopped_by_max_updates",
        "NO_IMPROVING_TRANSFER_EXCHANGE": result.stop_reason
        == "converged_by_no_improvement",
        "transfer_exchange_stable": result.transfer_exchange_stable,
        "stable_check_passed": result.stable_check_passed,
        "max_remaining_delta": result.max_remaining_delta,
        "history": result.history,
        "tie_break_count": result.tie_break_count,
        "cloud_resource_mode": cache_config.cloud_resource_mode,
        "cloud_fallback_mode": cache_config.cloud_fallback_mode,
        "node_task_count": {
            "SV": int(sum(counts_by_role["SV"].values())),
            "SV_by_node": counts_by_role["SV"],
            "RSU": int(sum(counts_by_role["RSU"].values())),
            "RSU_by_node": counts_by_role["RSU"],
            "UAV": int(sum(counts_by_role["UAV"].values())),
            "UAV0": int(counts_by_node[uav_ids[0]]) if len(uav_ids) > 0 else 0,
            "UAV1": int(counts_by_node[uav_ids[1]]) if len(uav_ids) > 1 else 0,
            "UAV_by_node": counts_by_role["UAV"],
            "Cloud": int(cloud_count),
        },
        "resource": {
            "cloud_cpu_usage_cycles_per_s": float(
                result.cloud_cpu_used_cycles_per_s
            ),
            "cloud_cpu_capacity_cycles_per_s": (
                None
                if result.cloud_cpu_capacity_cycles_per_s is None
                else float(result.cloud_cpu_capacity_cycles_per_s)
            ),
            "cloud_cpu_remaining_cycles_per_s": (
                None
                if result.cloud_cpu_capacity_cycles_per_s is None
                else max(
                    float(result.cloud_cpu_capacity_cycles_per_s)
                    - float(result.cloud_cpu_used_cycles_per_s),
                    0.0,
                )
            ),
            "cloud_cpu_allocations_cycles_per_s": dict(
                result.cloud_cpu_allocations_cycles_per_s
            ),
            "rsu_radio_bandwidth_fraction_sum_by_node": {
                node_id: float(math.fsum(result.bandwidth_allocations.get(node_id, {}).values()))
                for node_id in node_ids
                if node_roles[node_id] == "RSU"
            },
            "edge_cpu_usage_cycles_per_s": edge_cpu_total,
            "edge_cpu_usage_by_node_cycles_per_s": edge_cpu_by_node,
        },
        "performance": {
            "mean_delay_s": mean_delay,
            "mean_energy_j": mean_energy,
            "task_utility_sum": float(result.task_utility_sum),
            "mean_task_utility": mean_utility,
        },
    }
    return summary


def _configuration_summary(
    result: JointConfigurationResult,
    uav_tolerance_m: float = 1.0e-6,
) -> dict[str, Any]:
    actual_cache_update = np.any(
        result.final_cache_state != result.previous_cache_state,
        axis=1,
    ).astype(int).tolist()
    actual_uav_movement = {
        uav_id: int(
            float(
                np.linalg.norm(
                    np.asarray(result.final_uav_positions[uav_id], dtype=np.float64)
                    - np.asarray(result.previous_uav_positions[uav_id], dtype=np.float64)
                )
            )
            > float(uav_tolerance_m)
        )
        for uav_id in result.previous_uav_positions
    }
    prediction_skipped = result.stop_reason in {
        "no_trigger_prediction_skipped",
        "policy_target_prediction_skipped",
    }
    return {
        "trigger_feasible": result.trigger_feasible,
        "infeasible_reason": result.infeasible_reason,
        "requested_cache_trigger": list(result.cache_triggers),
        "actual_cache_update": actual_cache_update,
        "requested_uav_trigger": dict(result.uav_triggers),
        "actual_uav_movement": actual_uav_movement,
        "predictive_utility": None if prediction_skipped else result.predictive_utility,
        "predictive_objective": None if prediction_skipped else result.objective_value,
        "cache_cost": result.cache_cost,
        "cache_cost_by_node": dict(
            result.cache_result_summary.get("cache_cost_by_node", {})
        ),
        "movement_energy": result.movement_energy,
        "movement_cost": result.movement_cost,
        "outer_rounds": result.outer_rounds,
        "accepted_cache_updates": result.accepted_cache_updates,
        "accepted_uav_updates": result.accepted_uav_updates,
        "cache_candidate_diagnostics": result.cache_result_summary.get(
            "candidate_diagnostics", {}
        ),
        "uav_candidate_diagnostics": result.uav_result_summary.get(
            "candidate_diagnostics", {}
        ),
        "stop_reason": result.stop_reason,
        "predictive_evaluation_skipped": prediction_skipped,
    }


def _record_slot_performance_profile(
    profile: dict[str, Any] | None,
    configuration_seconds: float,
    matching_kkt_seconds: float,
) -> None:
    """Accumulate optional diagnostic timing without changing slot semantics."""

    if profile is None:
        return
    profile["config_seconds"] = float(profile.get("config_seconds", 0.0)) + float(
        configuration_seconds
    )
    profile["matching_kkt_seconds"] = float(profile.get("matching_kkt_seconds", 0.0)) + float(
        matching_kkt_seconds
    )


def _assert_requested_actual_trigger_contract(
    configuration: JointConfigurationResult,
    requested_cache_trigger: np.ndarray,
    requested_uav_trigger: dict[str, int],
    uav_tolerance_m: float,
) -> None:
    actual_cache_update = np.any(
        configuration.final_cache_state != configuration.previous_cache_state,
        axis=1,
    ).astype(np.int8)
    if np.any(actual_cache_update > requested_cache_trigger):
        raise AssertionError(
            "cache actual update exceeds requested trigger in slot configuration"
        )
    for uav_id, requested in requested_uav_trigger.items():
        actual = int(
            float(
                np.linalg.norm(
                    np.asarray(configuration.final_uav_positions[uav_id], dtype=np.float64)
                    - np.asarray(configuration.previous_uav_positions[uav_id], dtype=np.float64)
                )
            )
            > float(uav_tolerance_m)
        )
        if actual > int(requested):
            raise AssertionError(
                f"UAV {uav_id} actual movement exceeds requested trigger in slot configuration"
            )


def execute_slot(
    previous_cache_state: np.ndarray,
    previous_uav_positions: dict[str, Any],
    cache_triggers: np.ndarray,
    uav_triggers: dict[str, int],
    scenario_slice: dict[str, Any],
    actual_tasks: list[task_allocation.Task],
    actual_vehicle_positions: dict[str, np.ndarray],
    service_catalog: dict[str, Any],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    flight_boundary: tuple[float, float, float, float],
    slot_id: int | None = None,
    time_s: float | None = None,
    performance_profile: dict[str, Any] | None = None,
    configuration_override: JointConfigurationResult | None = None,
    matching_kkt_cache: task_allocation.KKTResourceAllocationCache | None = None,
    node_evaluation_cache: task_allocation.NodeEvaluationCache | None = None,
    exact_q_timing_profile: dict[str, float] | None = None,
) -> SlotExecutionResult:
    """Execute one slot from external trigger to actual system utility.

    Core API does not read disk. The caller supplies prediction scenario slice for
    configuration and actual current state for scheduling. Reward/system utility
    uses actual Matching/KKT output, not predictive configuration utility.
    """

    if exact_q_timing_profile is not None:
        for key in (
            "matching_runtime",
            "matching_total_runtime",
            "node_cache_lookup_runtime",
            "node_eval_miss_runtime",
            "kkt_runtime",
            "kkt_coefficient_build_runtime",
            "kkt_solver_core_runtime",
            "kkt_result_assembly_runtime",
            "reward_cost_aggregation_runtime",
            "reward_final_combination_runtime",
        ):
            exact_q_timing_profile.setdefault(key, 0.0)

    previous_cache = np.asarray(previous_cache_state, dtype=np.int8).copy()
    previous_uav = _position_tuples(previous_uav_positions)
    cache_trigger = np.asarray(cache_triggers, dtype=np.int8).copy()
    cache_configuration.validate_cache_shape_and_capacity(
        previous_cache,
        service_catalog,
        cache_nodes,
    )
    if cache_trigger.shape != (previous_cache.shape[0],) or not np.all(
        np.isin(cache_trigger, [0, 1])
    ):
        raise ValueError("cache_triggers 必须是与 cache node 数量一致的0/1向量。")
    if set(uav_triggers) != set(previous_uav) or any(
        int(value) not in (0, 1) for value in uav_triggers.values()
    ):
        raise ValueError("uav_triggers 必须覆盖全部 UAV 且只能取0/1。")

    no_configuration_trigger = (
        int(cache_trigger.sum()) == 0
        and not any(int(value) == 1 for value in uav_triggers.values())
    )
    prediction_evaluation_skipped = False
    configuration_started_at = time.perf_counter() if performance_profile is not None else 0.0
    if configuration_override is not None:
        configuration = _validated_configuration_override(
            configuration_override,
            previous_cache,
            previous_uav,
            cache_trigger,
            {key: int(value) for key, value in uav_triggers.items()},
            service_catalog,
            cache_nodes,
            cache_config,
            project_cfg,
            flight_boundary,
        )
        prediction_evaluation_skipped = True
    elif no_configuration_trigger:
        # Gate=NULL 时配置状态严格保持不变；预测目标既不参与 Matching，也不参与
        # 当前 reward，因此不执行昂贵且无决策作用的场景配置评价。
        configuration = JointConfigurationResult(
            previous_cache_state=previous_cache.copy(),
            final_cache_state=previous_cache.copy(),
            previous_uav_positions=dict(previous_uav),
            final_uav_positions=dict(previous_uav),
            cache_triggers=cache_trigger.astype(int).tolist(),
            uav_triggers={key: int(value) for key, value in uav_triggers.items()},
            trigger_feasible=True,
            infeasible_reason="",
            predictive_utility=0.0,
            cache_cost=0.0,
            movement_energy=0.0,
            movement_cost=0.0,
            objective_value=0.0,
            initial_objective=0.0,
            objective_history=[],
            outer_rounds=0,
            accepted_cache_updates=0,
            accepted_uav_updates=0,
            stop_reason="no_trigger_prediction_skipped",
            cache_result_summary={},
            uav_result_summary={},
        )
        prediction_evaluation_skipped = True
    else:
        raise ValueError(
            "configuration_override is required when any configuration trigger is active."
        )
    configuration_seconds = (
        time.perf_counter() - configuration_started_at
        if performance_profile is not None
        else 0.0
    )
    _assert_requested_actual_trigger_contract(
        configuration,
        cache_trigger,
        {key: int(value) for key, value in uav_triggers.items()},
        float(project_config.get_config_value(
            project_cfg,
            "algorithm.uav.position_tolerance_m",
            1.0e-6,
        )),
    )
    if not configuration.trigger_feasible:
        _record_slot_performance_profile(
            performance_profile,
            configuration_seconds,
            0.0,
        )
        return SlotExecutionResult(
            slot_id=slot_id,
            time_s=time_s,
            step_feasible=False,
            infeasible_reason=configuration.infeasible_reason,
            input_cache_triggers=cache_trigger.astype(int).tolist(),
            input_uav_triggers=dict(uav_triggers),
            previous_cache_state=previous_cache.copy(),
            previous_uav_positions=previous_uav,
            final_cache_state=None,
            final_uav_positions=None,
            predictive_configuration_utility=None,
            predictive_configuration_objective=None,
            cache_cost=None,
            movement_energy=None,
            movement_cost=None,
            actual_task_utility=None,
            system_utility=None,
            matching={},
            cpu_allocations={},
            bandwidth_allocations={},
            per_task_delay_s={},
            per_task_vehicle_energy_j={},
            per_task_utility={},
            configuration_summary=_configuration_summary(
                configuration,
                float(project_config.get_config_value(
                    project_cfg,
                    "algorithm.uav.position_tolerance_m",
                    1.0e-6,
                )),
            ),
            matching_summary={},
            next_cache_state=None,
            next_uav_positions=None,
        )

    matching_started_at = time.perf_counter() if performance_profile is not None else 0.0
    final_cache = np.asarray(configuration.final_cache_state, dtype=np.int8).copy()
    final_uav = dict(configuration.final_uav_positions)
    service_positions = task_allocation.build_service_node_positions(
        cache_nodes,
        actual_vehicle_positions,
        {uav_id: np.asarray(xy, dtype=np.float64) for uav_id, xy in final_uav.items()},
    )
    exact_q_matching_started_at = (
        time.perf_counter() if exact_q_timing_profile is not None else 0.0
    )
    exact_q_node_runtime_before = (
        0.0
        if exact_q_timing_profile is None
        else sum(
            float(exact_q_timing_profile.get(key, 0.0))
            for key in (
                "node_cache_lookup_runtime",
                "node_eval_miss_runtime",
                "kkt_runtime",
            )
        )
    )
    matching = task_allocation.optimize_task_matching(
        actual_tasks,
        actual_vehicle_positions,
        service_positions,
        final_cache,
        project_cfg,
        cache_nodes,
        cache_config,
        kkt_cache=matching_kkt_cache,
        node_evaluation_cache=node_evaluation_cache,
        timing_profile=exact_q_timing_profile,
        cloud_fallback_mode=cache_config.cloud_fallback_mode,
    )
    if exact_q_timing_profile is not None:
        exact_q_matching_elapsed = time.perf_counter() - exact_q_matching_started_at
        exact_q_timing_profile["matching_total_runtime"] += exact_q_matching_elapsed
        exact_q_node_elapsed = sum(
            float(exact_q_timing_profile.get(key, 0.0))
            for key in (
                "node_cache_lookup_runtime",
                "node_eval_miss_runtime",
                "kkt_runtime",
            )
        ) - exact_q_node_runtime_before
        exact_q_timing_profile["matching_runtime"] += max(
            exact_q_matching_elapsed - exact_q_node_elapsed,
            0.0,
        )
    matching_kkt_seconds = (
        time.perf_counter() - matching_started_at
        if performance_profile is not None
        else 0.0
    )
    if not matching.resource_feasible:
        _record_slot_performance_profile(
            performance_profile,
            configuration_seconds,
            matching_kkt_seconds,
        )
        return SlotExecutionResult(
            slot_id=slot_id,
            time_s=time_s,
            step_feasible=False,
            infeasible_reason="matching resource infeasible",
            input_cache_triggers=cache_trigger.astype(int).tolist(),
            input_uav_triggers=dict(uav_triggers),
            previous_cache_state=previous_cache.copy(),
            previous_uav_positions=previous_uav,
            final_cache_state=None,
            final_uav_positions=None,
            predictive_configuration_utility=None,
            predictive_configuration_objective=None,
            cache_cost=None,
            movement_energy=None,
            movement_cost=None,
            actual_task_utility=None,
            system_utility=None,
            matching={},
            cpu_allocations={},
            bandwidth_allocations={},
            per_task_delay_s={},
            per_task_vehicle_energy_j={},
            per_task_utility={},
            configuration_summary=_configuration_summary(
                configuration,
                float(project_config.get_config_value(
                    project_cfg,
                    "algorithm.uav.position_tolerance_m",
                    1.0e-6,
                )),
            ),
            matching_summary=_matching_summary(
                matching,
                cache_nodes,
                cache_config,
            ),
            next_cache_state=None,
            next_uav_positions=None,
        )

    actual_q = float(matching.task_utility_sum)
    reward_cost_started_at = (
        time.perf_counter() if exact_q_timing_profile is not None else 0.0
    )
    cache_cost = float(configuration.cache_cost)
    movement_cost = float(configuration.movement_cost)
    if exact_q_timing_profile is not None:
        exact_q_timing_profile["reward_cost_aggregation_runtime"] += (
            time.perf_counter() - reward_cost_started_at
        )
    reward_final_started_at = (
        time.perf_counter() if exact_q_timing_profile is not None else 0.0
    )
    system_utility = actual_q - cache_cost - movement_cost
    if exact_q_timing_profile is not None:
        exact_q_timing_profile["reward_final_combination_runtime"] += (
            time.perf_counter() - reward_final_started_at
        )
    if not math.isclose(
        system_utility,
        actual_q - cache_cost - movement_cost,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError("G(t)=Q(t)-C_ca(t)-C_q(t) 恒等式失败。")

    _record_slot_performance_profile(
        performance_profile,
        configuration_seconds,
        matching_kkt_seconds,
    )
    return SlotExecutionResult(
        slot_id=slot_id,
        time_s=time_s,
        step_feasible=True,
        infeasible_reason="",
        input_cache_triggers=cache_trigger.astype(int).tolist(),
        input_uav_triggers=dict(uav_triggers),
        previous_cache_state=previous_cache.copy(),
        previous_uav_positions=previous_uav,
        final_cache_state=final_cache.copy(),
        final_uav_positions=final_uav,
        predictive_configuration_utility=(
            None
            if prediction_evaluation_skipped
            else float(configuration.predictive_utility)
        ),
        predictive_configuration_objective=(
            None
            if prediction_evaluation_skipped
            else float(configuration.objective_value)
        ),
        cache_cost=cache_cost,
        movement_energy=float(configuration.movement_energy),
        movement_cost=movement_cost,
        actual_task_utility=actual_q,
        system_utility=float(system_utility),
        matching=matching.matching,
        cpu_allocations=matching.cpu_allocations,
        bandwidth_allocations=matching.bandwidth_allocations,
        per_task_delay_s=matching.per_task_delay_s,
        per_task_vehicle_energy_j=matching.per_task_vehicle_energy_j,
        per_task_utility=matching.per_task_utility,
        configuration_summary=_configuration_summary(
            configuration,
            float(project_config.get_config_value(
                project_cfg,
                "algorithm.uav.position_tolerance_m",
                1.0e-6,
            )),
        ),
        matching_summary=_matching_summary(
            matching,
            cache_nodes,
            cache_config,
        ),
        next_cache_state=final_cache.copy(),
        next_uav_positions=final_uav,
    )









# ---- Joint cache/UAV configuration orchestration ----

@dataclass(frozen=True)
class JointConfigurationResult:
    previous_cache_state: np.ndarray
    final_cache_state: np.ndarray
    previous_uav_positions: dict[str, tuple[float, float]]
    final_uav_positions: dict[str, tuple[float, float]]
    cache_triggers: list[int]
    uav_triggers: dict[str, int]
    trigger_feasible: bool
    infeasible_reason: str
    predictive_utility: float
    cache_cost: float
    movement_energy: float
    movement_cost: float
    objective_value: float
    initial_objective: float
    objective_history: list[float]
    outer_rounds: int
    accepted_cache_updates: int
    accepted_uav_updates: int
    stop_reason: str
    cache_result_summary: dict[str, Any]
    uav_result_summary: dict[str, Any]




def _position_arrays(positions: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        str(uav_id): np.asarray(xy, dtype=np.float64).copy()
        for uav_id, xy in positions.items()
    }


def _cache_cost(
    previous_cache_state: np.ndarray,
    cache_state: np.ndarray,
    cache_triggers: np.ndarray,
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Cache cost 始终由 C_prev -> C_candidate 一次性计算。"""

    result = cache_configuration.cache_update_cost_details(
        previous_cache_state,
        cache_state,
        cache_triggers,
        service_catalog["service_sizes_mb"],
        cache_nodes,
        cache_config,
    )
    return (
        float(result["total_cost"]),
        np.asarray(result["loading_energy"], dtype=np.float64),
        np.asarray(result["cost_by_node"], dtype=np.float64),
    )




def build_policy_target_configuration(
    previous_cache_state: np.ndarray,
    previous_uav_positions: dict[str, Any],
    final_cache_state: np.ndarray,
    final_uav_positions: dict[str, Any],
    cache_triggers: np.ndarray,
    uav_triggers: dict[str, int],
    service_catalog: dict[str, Any],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    flight_boundary: tuple[float, float, float, float],
    cache_actions: tuple[dict[str, Any], ...] = (),
    uav_actions: tuple[dict[str, Any], ...] = (),
) -> JointConfigurationResult:
    """Build a direct-policy target configuration without a prediction solve."""

    previous_cache = np.asarray(previous_cache_state, dtype=np.int8).copy()
    final_cache = np.asarray(final_cache_state, dtype=np.int8).copy()
    previous_uav = _position_arrays(previous_uav_positions)
    final_uav = _position_arrays(final_uav_positions)
    cache_trigger = np.asarray(cache_triggers, dtype=np.int8).copy()
    cache_configuration.validate_cache_shape_and_capacity(
        previous_cache, service_catalog, cache_nodes
    )
    cache_configuration.validate_cache_shape_and_capacity(
        final_cache, service_catalog, cache_nodes
    )
    if cache_trigger.shape != (previous_cache.shape[0],) or not np.all(
        np.isin(cache_trigger, [0, 1])
    ):
        raise ValueError("policy target cache triggers 非法。")
    actual_cache = np.any(final_cache != previous_cache, axis=1).astype(np.int8)
    if np.any(actual_cache > cache_trigger):
        raise ValueError("policy target cache state 违反 trigger=0 保持不变语义。")
    uav_configuration.validate_uav_configuration_transition(
        previous_uav,
        final_uav,
        uav_triggers,
        project_cfg,
        flight_boundary,
    )
    cache_cost, _, cache_cost_by_node = _cache_cost(
        previous_cache,
        final_cache,
        cache_trigger,
        service_catalog,
        cache_nodes,
        cache_config,
    )
    search = uav_configuration.uav_search_config_from_project(project_cfg)
    movement_energy, movement_cost, _, _, _ = uav_configuration.movement_summary(
        previous_uav,
        final_uav,
        {key: int(value) for key, value in uav_triggers.items()},
        search,
    )
    actual_uav = sum(
        int(
            np.linalg.norm(final_uav[uav_id] - previous_uav[uav_id])
            > search.position_tolerance_m
        )
        for uav_id in previous_uav
    )
    return JointConfigurationResult(
        previous_cache_state=previous_cache,
        final_cache_state=final_cache,
        previous_uav_positions=_position_tuples(previous_uav),
        final_uav_positions=_position_tuples(final_uav),
        cache_triggers=cache_trigger.astype(int).tolist(),
        uav_triggers={key: int(value) for key, value in uav_triggers.items()},
        trigger_feasible=True,
        infeasible_reason="",
        predictive_utility=0.0,
        cache_cost=float(cache_cost),
        movement_energy=float(movement_energy),
        movement_cost=float(movement_cost),
        objective_value=0.0,
        initial_objective=0.0,
        objective_history=[],
        outer_rounds=0,
        accepted_cache_updates=int(actual_cache.sum()),
        accepted_uav_updates=int(actual_uav),
        stop_reason="policy_target_prediction_skipped",
        cache_result_summary={
            "target_actions": [dict(value) for value in cache_actions],
            "cache_cost_by_node": {
                str(node_id): float(cache_cost_by_node[index])
                for index, node_id in enumerate(cache_nodes["node_ids"])
            },
        },
        uav_result_summary={"target_actions": [dict(value) for value in uav_actions]},
    )


def execute_configured_slot(
    previous_cache_state: np.ndarray,
    previous_uav_positions: dict[str, Any],
    final_cache_state: np.ndarray,
    final_uav_positions: dict[str, Any],
    scenario_slice: dict[str, Any],
    actual_tasks: list[task_allocation.Task],
    actual_vehicle_positions: dict[str, np.ndarray],
    service_catalog: dict[str, Any],
    project_cfg: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    flight_boundary: tuple[float, float, float, float],
    *,
    slot_id: int | None = None,
    time_s: float | None = None,
    performance_profile: dict[str, Any] | None = None,
    matching_kkt_cache: task_allocation.KKTResourceAllocationCache | None = None,
    node_evaluation_cache: task_allocation.NodeEvaluationCache | None = None,
    exact_q_timing_profile: dict[str, float] | None = None,
    task_ledger: CrossSlotTaskLedger | None = None,
) -> SlotExecutionResult:
    """执行控制器已确定的最终 Cache/UAV 配置及真实 Matching/KKT。"""

    previous_cache = np.asarray(previous_cache_state, dtype=np.int8).copy()
    final_cache = np.asarray(final_cache_state, dtype=np.int8).copy()
    previous_uav = _position_tuples(previous_uav_positions)
    final_uav = _position_tuples(final_uav_positions)
    if previous_cache.shape != final_cache.shape:
        raise ValueError("最终 Cache 配置 shape 必须与当前 Cache 一致。")
    cache_configuration.validate_cache_shape_and_capacity(
        previous_cache, service_catalog, cache_nodes
    )
    cache_configuration.validate_cache_shape_and_capacity(
        final_cache, service_catalog, cache_nodes
    )
    if task_ledger is not None:
        locks = active_cache_service_locks(
            task_ledger,
            previous_cache,
            service_catalog,
            cache_nodes,
        )
        evicted = np.argwhere(locks & (final_cache != 1))
        if len(evicted):
            node_index, service_id = (int(value) for value in evicted[0])
            task_ids = sorted(
                str(task_id)
                for task_id, state in task_ledger.active.items()
                if str(state.assigned_node) == str(cache_nodes["node_ids"][node_index])
                and int(state.service_id) == service_id
            )
            raise CacheLockConflict(
                "selected Cache target is illegal for the current active-task ledger: "
                f"node={cache_nodes['node_ids'][node_index]}, service={service_id}, "
                f"tasks={task_ids}"
            )
    cache_updates = np.any(final_cache != previous_cache, axis=1).astype(np.int8)
    tolerance = float(
        project_config.get_config_value(
            project_cfg, "algorithm.uav.position_tolerance_m", 1.0e-6
        )
    )
    uav_moves = {
        uav_id: int(
            np.linalg.norm(
                np.asarray(final_uav[uav_id], dtype=np.float64)
                - np.asarray(previous_uav[uav_id], dtype=np.float64)
            )
            > tolerance
        )
        for uav_id in previous_uav
    }
    configuration = build_policy_target_configuration(
        previous_cache,
        previous_uav,
        final_cache,
        final_uav,
        cache_updates,
        uav_moves,
        service_catalog,
        project_cfg,
        cache_nodes,
        cache_config,
        flight_boundary,
    )
    if not bool(np.any(cache_updates)) and configuration.cache_cost != 0.0:
        raise AssertionError("Cache 配置未变化时不得产生 reconfiguration cost。")
    if task_ledger is not None:
        return _execute_cross_slot_tasks(
            configuration,
            actual_tasks,
            actual_vehicle_positions,
            service_catalog,
            project_cfg,
            cache_nodes,
            cache_config,
            slot_id,
            time_s,
            task_ledger,
            performance_profile,
            exact_q_timing_profile,
        )
    return execute_slot(
        previous_cache,
        previous_uav,
        cache_updates,
        uav_moves,
        scenario_slice,
        actual_tasks,
        actual_vehicle_positions,
        service_catalog,
        project_cfg,
        cache_nodes,
        cache_config,
        flight_boundary,
        slot_id=slot_id,
        time_s=time_s,
        performance_profile=performance_profile,
        configuration_override=configuration,
        matching_kkt_cache=matching_kkt_cache,
        node_evaluation_cache=node_evaluation_cache,
        exact_q_timing_profile=exact_q_timing_profile,
    )

def _validated_configuration_override(
    configuration: JointConfigurationResult,
    previous_cache: np.ndarray,
    previous_uav: dict[str, tuple[float, float]],
    cache_trigger: np.ndarray,
    uav_triggers: dict[str, int],
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_config: cache_configuration.CacheOptimizationConfig,
    project_cfg: dict[str, Any],
    flight_boundary: tuple[float, float, float, float],
) -> JointConfigurationResult:
    """Reject any override that is not an authoritative direct-policy transition."""

    if not isinstance(configuration, JointConfigurationResult):
        raise TypeError("configuration_override 必须是 JointConfigurationResult。")
    if configuration.stop_reason != "policy_target_prediction_skipped":
        raise ValueError("configuration_override 仅接受 policy target result。")
    if not np.array_equal(configuration.previous_cache_state, previous_cache):
        raise ValueError("configuration_override previous cache 不匹配。")
    if configuration.cache_triggers != cache_trigger.astype(int).tolist():
        raise ValueError("configuration_override cache triggers 不匹配。")
    if configuration.uav_triggers != uav_triggers:
        raise ValueError("configuration_override UAV triggers 不匹配。")
    for uav_id, xy in previous_uav.items():
        if not np.array_equal(
            np.asarray(configuration.previous_uav_positions[uav_id], dtype=np.float64),
            np.asarray(xy, dtype=np.float64),
        ):
            raise ValueError("configuration_override previous UAV state 不匹配。")
    rebuilt = build_policy_target_configuration(
        previous_cache,
        previous_uav,
        configuration.final_cache_state,
        configuration.final_uav_positions,
        cache_trigger,
        uav_triggers,
        service_catalog,
        project_cfg,
        cache_nodes,
        cache_config,
        flight_boundary,
        cache_actions=tuple(configuration.cache_result_summary.get("target_actions", [])),
        uav_actions=tuple(configuration.uav_result_summary.get("target_actions", [])),
    )
    if not math.isclose(rebuilt.cache_cost, configuration.cache_cost, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("configuration_override cache cost 不匹配。")
    if not math.isclose(
        rebuilt.movement_cost, configuration.movement_cost, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("configuration_override movement cost 不匹配。")
    return rebuilt
