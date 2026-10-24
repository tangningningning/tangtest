#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-slot task state, accounting, and terminal-event handling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from decision.task_allocation import CLOUD, Task


@dataclass
class ActiveTaskState:
    """Mutable lifecycle state for one arrived task; time and work use SI units."""

    task_id: str
    vehicle_id: str
    arrival_slot: int
    service_id: int
    input_bits: float
    compute_density_cycles_per_bit: float
    cpu_cycles: float
    remaining_upload_bits: float
    remaining_compute_cycles: float
    remaining_slots: int
    execution_node: str | None
    cumulative_upload_energy_j: float
    execution_stage: str
    status: str = "ACTIVE"
    terminal_slot: int | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "vehicle_id": self.vehicle_id,
            "arrival_slot": self.arrival_slot,
            "service_id": self.service_id,
            "input_bits": self.input_bits,
            "compute_density_cycles_per_bit": self.compute_density_cycles_per_bit,
            "cpu_cycles": self.cpu_cycles,
            "remaining_upload_bits": self.remaining_upload_bits,
            "remaining_compute_cycles": self.remaining_compute_cycles,
            "remaining_slots": self.remaining_slots,
            "execution_node": self.execution_node,
            "cumulative_upload_energy_j": self.cumulative_upload_energy_j,
            "execution_stage": self.execution_stage,
            "status": self.status,
            "terminal_slot": self.terminal_slot,
        }


@dataclass(frozen=True)
class PreparedTaskSlot:
    """Tasks and matching constraints prepared once at the start of a slot."""

    slot_index: int
    tasks: tuple[Task, ...]
    fixed_matching: Mapping[str, str]
    mutable_task_ids: frozenset[str]
    arrived_task_ids: tuple[str, ...]
    protected_services_by_node: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class LifecycleSlotResult:
    """Post-update state and terminal events produced at one slot boundary."""

    arrived_task_ids: tuple[str, ...]
    active_tasks: tuple[Mapping[str, Any], ...]
    termination_events: tuple[Mapping[str, Any], ...]
    per_task_slot_metrics: Mapping[str, Mapping[str, float | int | str | None]]
    active_task_count: int
    completed_task_count: int
    timed_out_task_count: int
    episode_upload_energy_j: float
    slot_upload_energy_j: float


class TaskLifecycleManager:
    """Maintain active tasks and apply realized upload/compute work once per slot."""

    def __init__(
        self,
        *,
        arrival_horizon_slots: int,
        k_max_slots: int,
        dt_s: float = 1.0,
    ) -> None:
        for name, value in (
            ("arrival_horizon_slots", arrival_horizon_slots),
            ("k_max_slots", k_max_slots),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isclose(float(dt_s), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("task lifecycle requires a one-second time slot")
        self.arrival_horizon_slots = int(arrival_horizon_slots)
        self.k_max_slots = int(k_max_slots)
        self.execution_horizon_slots = (
            self.arrival_horizon_slots + self.k_max_slots - 1
        )
        self.terminal_execution_slot = self.execution_horizon_slots - 1
        self.dt_s = float(dt_s)
        self.reset()

    def reset(self) -> None:
        self.active_tasks: dict[str, ActiveTaskState] = {}
        self.terminal_events: list[dict[str, Any]] = []
        self.episode_upload_energy_j = 0.0
        self.next_slot_index = 0
        self._prepared: PreparedTaskSlot | None = None

    @property
    def protected_services_by_node(self) -> dict[str, tuple[int, ...]]:
        protected: dict[str, set[int]] = {}
        for state in self.active_tasks.values():
            node = state.execution_node
            if state.status == "ACTIVE" and node is not None and node != CLOUD:
                protected.setdefault(node, set()).add(state.service_id)
        return {
            node: tuple(sorted(service_ids))
            for node, service_ids in protected.items()
        }

    def prepare_slot(
        self,
        slot_index: int,
        arrivals: Sequence[Task],
    ) -> PreparedTaskSlot:
        """Initialize arrivals and expose residual work with fixed prior nodes."""

        if self._prepared is not None:
            raise RuntimeError("previous task lifecycle slot was not advanced")
        if int(slot_index) != self.next_slot_index:
            raise ValueError(
                "task lifecycle requires contiguous zero-based slots: "
                f"expected={self.next_slot_index}, received={slot_index}"
            )
        if int(slot_index) >= self.execution_horizon_slots:
            raise ValueError(
                "task lifecycle execution horizon is exhausted: "
                f"slot={slot_index}, terminal_execution_slot={self.terminal_execution_slot}"
            )
        arrival_slot = int(slot_index) + 1
        if arrival_slot > self.arrival_horizon_slots and arrivals:
            raise ValueError(
                "new task arrivals are forbidden after the original arrival horizon: "
                f"slot={arrival_slot}, T={self.arrival_horizon_slots}, arrivals={len(arrivals)}"
            )

        arrived_ids: list[str] = []
        seen_vehicles: set[str] = set()
        for arrival in sorted(arrivals, key=lambda item: (item.vehicle_id, item.service_id)):
            if arrival_slot > self.arrival_horizon_slots:
                break
            vehicle_id = str(arrival.vehicle_id)
            if vehicle_id in seen_vehicles:
                raise ValueError(
                    f"at most one task per vehicle per slot is supported: {vehicle_id}"
                )
            seen_vehicles.add(vehicle_id)
            if arrival.execution_stage != "NEW":
                raise ValueError("arrivals must enter the lifecycle as NEW tasks")
            input_bits = float(arrival.input_bits)
            cpu_cycles = float(arrival.cpu_cycles)
            if (
                not math.isfinite(input_bits)
                or input_bits <= 0.0
                or not math.isfinite(cpu_cycles)
                or cpu_cycles <= 0.0
            ):
                raise ValueError(f"invalid task demand for vehicle {vehicle_id}")
            task_id = f"{vehicle_id}@{arrival_slot}"
            if task_id in self.active_tasks or any(
                event["task_id"] == task_id for event in self.terminal_events
            ):
                raise ValueError(f"duplicate lifecycle task ID: {task_id}")
            self.active_tasks[task_id] = ActiveTaskState(
                task_id=task_id,
                vehicle_id=vehicle_id,
                arrival_slot=arrival_slot,
                service_id=int(arrival.service_id),
                input_bits=input_bits,
                compute_density_cycles_per_bit=cpu_cycles / input_bits,
                cpu_cycles=cpu_cycles,
                remaining_upload_bits=input_bits,
                remaining_compute_cycles=cpu_cycles,
                remaining_slots=self.k_max_slots,
                execution_node=None,
                cumulative_upload_energy_j=0.0,
                execution_stage="UPLOADING",
            )
            arrived_ids.append(task_id)

        fixed_matching = {
            task_id: str(state.execution_node)
            for task_id, state in self.active_tasks.items()
            if state.execution_node is not None
        }
        mutable_task_ids = frozenset(arrived_ids)
        solver_tasks = tuple(
            Task(
                task_id=state.task_id,
                vehicle_id=state.vehicle_id,
                service_id=state.service_id,
                input_bits=state.remaining_upload_bits,
                cpu_cycles=state.remaining_compute_cycles,
                execution_stage=(
                    "UPLOADING"
                    if state.remaining_upload_bits > 0.0
                    else "COMPUTING"
                ),
                lifecycle_remaining_slots=state.remaining_slots,
                lifecycle_k_max_slots=self.k_max_slots,
                lifecycle_cumulative_upload_energy_j=(
                    state.cumulative_upload_energy_j
                ),
            )
            for state in self.active_tasks.values()
        )
        prepared = PreparedTaskSlot(
            slot_index=int(slot_index),
            tasks=solver_tasks,
            fixed_matching=fixed_matching,
            mutable_task_ids=mutable_task_ids,
            arrived_task_ids=tuple(arrived_ids),
            protected_services_by_node=self.protected_services_by_node,
        )
        self._prepared = prepared
        return prepared

    def advance_slot(
        self,
        slot_index: int,
        matching: Mapping[str, str],
        uploaded_bits: Mapping[str, float],
        computed_cycles: Mapping[str, float],
        slot_upload_energy_j: Mapping[str, float],
    ) -> LifecycleSlotResult:
        """Apply realized physical work, decrement deadlines, and emit one event."""

        prepared = self._prepared
        if prepared is None or prepared.slot_index != int(slot_index):
            raise RuntimeError("task lifecycle slot was not prepared before execution")
        task_metrics: dict[str, dict[str, float | int | str | None]] = {}
        new_events: list[dict[str, Any]] = []
        completed = 0
        timed_out = 0
        slot_energy_total = 0.0

        for task_id, state in tuple(self.active_tasks.items()):
            if task_id not in matching:
                raise ValueError(f"matching omitted active task {task_id}")
            node = str(matching[task_id])
            if state.execution_node is None:
                state.execution_node = node
            elif state.execution_node != node:
                raise ValueError(
                    f"active task {task_id} migrated: "
                    f"fixed={state.execution_node}, proposed={node}"
                )

            uploaded = self._bounded_work_value(
                uploaded_bits.get(task_id, 0.0),
                state.remaining_upload_bits,
                "uploaded bits",
                task_id,
            )
            computed = self._bounded_work_value(
                computed_cycles.get(task_id, 0.0),
                state.remaining_compute_cycles,
                "computed cycles",
                task_id,
            )
            if state.remaining_upload_bits > 0.0 and computed != 0.0:
                raise ValueError(
                    f"task {task_id} received computation before its upload phase ended"
                )
            if state.remaining_upload_bits == 0.0 and uploaded != 0.0:
                raise ValueError(f"task {task_id} uploaded data after upload completion")
            energy = float(slot_upload_energy_j.get(task_id, 0.0))
            if not math.isfinite(energy) or energy < 0.0:
                raise ValueError(f"invalid slot upload energy for task {task_id}")
            if uploaded == 0.0 and energy != 0.0:
                raise ValueError(
                    f"task {task_id} incurred upload energy without uploading data"
                )

            state.remaining_upload_bits = max(0.0, state.remaining_upload_bits - uploaded)
            state.remaining_compute_cycles = max(
                0.0, state.remaining_compute_cycles - computed
            )
            state.remaining_slots = max(0, state.remaining_slots - 1)
            state.cumulative_upload_energy_j += energy
            state.execution_stage = (
                "UPLOADING"
                if state.remaining_upload_bits > 0.0
                else "COMPUTING"
            )
            slot_energy_total += energy

            terminal_reason: str | None = None
            if (
                state.remaining_upload_bits == 0.0
                and state.remaining_compute_cycles == 0.0
            ):
                terminal_reason = "SUCCESS"
                completed += 1
            elif state.remaining_slots == 0:
                terminal_reason = "TIMEOUT"
                timed_out += 1

            actual_execution_time_s: float | None = None
            if terminal_reason is not None:
                state.status = terminal_reason
                state.terminal_slot = int(slot_index) + 1
                actual_execution_time_s = (
                    self.k_max_slots - state.remaining_slots
                ) * self.dt_s
                event = {
                    **state.snapshot(),
                    "termination_reason": terminal_reason,
                    "actual_execution_time_s": actual_execution_time_s,
                }
                new_events.append(event)

            task_metrics[task_id] = {
                "execution_node": node,
                "uploaded_bits": uploaded,
                "computed_cycles": computed,
                "remaining_upload_bits": state.remaining_upload_bits,
                "remaining_compute_cycles": state.remaining_compute_cycles,
                "remaining_slots": state.remaining_slots,
                "slot_upload_energy_j": energy,
                "cumulative_upload_energy_j": state.cumulative_upload_energy_j,
                "termination_reason": terminal_reason,
                "actual_execution_time_s": actual_execution_time_s,
            }

        self.episode_upload_energy_j += slot_energy_total
        for event in new_events:
            self.terminal_events.append(event)
            self.active_tasks.pop(str(event["task_id"]), None)

        active_snapshots = tuple(
            state.snapshot() for state in self.active_tasks.values()
        )
        arrived_ids = prepared.arrived_task_ids
        self._prepared = None
        self.next_slot_index += 1
        return LifecycleSlotResult(
            arrived_task_ids=arrived_ids,
            active_tasks=active_snapshots,
            termination_events=tuple(new_events),
            per_task_slot_metrics=task_metrics,
            active_task_count=len(active_snapshots),
            completed_task_count=completed,
            timed_out_task_count=timed_out,
            episode_upload_energy_j=self.episode_upload_energy_j,
            slot_upload_energy_j=slot_energy_total,
        )

    @staticmethod
    def _bounded_work_value(
        value: float,
        remaining: float,
        label: str,
        task_id: str,
    ) -> float:
        actual = float(value)
        if not math.isfinite(actual) or actual < 0.0:
            raise ValueError(f"invalid {label} for task {task_id}")
        tolerance = max(1e-9, abs(float(remaining)) * 1e-12)
        if actual > float(remaining) + tolerance:
            raise ValueError(
                f"{label} exceeds remaining demand for task {task_id}: "
                f"actual={actual}, remaining={remaining}"
            )
        return min(actual, float(remaining))


__all__ = [
    "ActiveTaskState",
    "LifecycleSlotResult",
    "PreparedTaskSlot",
    "TaskLifecycleManager",
]
