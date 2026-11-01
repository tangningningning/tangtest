"""Independent v3.3.5 single-slot execution and fixed-matching P1-R model.

This module deliberately does not depend on the legacy task lifecycle or its
staged resource allocator. Inputs describe one slot only; accepted tasks finish
within that slot, while rejected tasks are dropped before transmission.

Units follow ``docs/13_系统模型_公式主文档_单时隙修订版_v3.3.5.md``:
bits, cycles, Hz, bit/s, cycle/s, seconds, watts, and joules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from common.system_model import (
    BANDWIDTH_DEPENDENT_NOISE_V1,
    uplink_rate,
)


SLOT_DURATION_S = 1.0


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(name: str, value: float) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class TaskDemand:
    """One newly arrived task, with no cross-slot residual-work state."""

    task_id: str
    vehicle_id: str
    service_id: str
    input_bits: float
    cpu_cycles: float
    transmit_power_w: float

    def __post_init__(self) -> None:
        if not self.task_id or not self.vehicle_id or not self.service_id:
            raise ValueError("task_id, vehicle_id, and service_id must be non-empty")
        _positive("input_bits", self.input_bits)
        _positive("cpu_cycles", self.cpu_cycles)
        _positive("transmit_power_w", self.transmit_power_w)


@dataclass(frozen=True)
class EdgeNode:
    """An independent edge bandwidth pool and CPU capacity."""

    node_id: str
    bandwidth_hz: float
    cpu_cycles_per_s: float

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must be non-empty")
        _positive("bandwidth_hz", self.bandwidth_hz)
        _positive("cpu_cycles_per_s", self.cpu_cycles_per_s)


@dataclass(frozen=True)
class RadioParameters:
    """Thermal-noise parameters for the v3.3.5 bandwidth-dependent model."""

    noise_psd_w_per_hz: float
    receiver_noise_figure_db: float

    def __post_init__(self) -> None:
        _positive("noise_psd_w_per_hz", self.noise_psd_w_per_hz)
        noise_figure = _finite(
            "receiver_noise_figure_db", self.receiver_noise_figure_db
        )
        if noise_figure < 0.0:
            raise ValueError("receiver_noise_figure_db must be non-negative")


@dataclass(frozen=True)
class CloudParameters:
    """Independent fixed per-task Cloud resources; there is no shared pool."""

    node_id: str
    upload_rate_bps: float
    cpu_cycles_per_s: float

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("Cloud node_id must be non-empty")
        _positive("cloud upload_rate_bps", self.upload_rate_bps)
        _positive("cloud cpu_cycles_per_s", self.cpu_cycles_per_s)


@dataclass(frozen=True)
class UtilityParameters:
    """D.1 task-utility scales, weights, and failure penalty."""

    time_reference_s: float
    energy_reference_j: float
    omega_time: float
    omega_energy: float
    failure_penalty: float

    def __post_init__(self) -> None:
        _positive("time_reference_s", self.time_reference_s)
        _positive("energy_reference_j", self.energy_reference_j)
        omega_time = _finite("omega_time", self.omega_time)
        omega_energy = _finite("omega_energy", self.omega_energy)
        penalty = _positive("failure_penalty", self.failure_penalty)
        if not (0.0 < omega_time <= 1.0):
            raise ValueError("omega_time must be in (0, 1]")
        if not (0.0 <= omega_energy < 1.0):
            raise ValueError("omega_energy must be in [0, 1)")
        if omega_time + omega_energy != 1.0:
            raise ValueError("v3.3.5 requires omega_time + omega_energy == 1")
        if penalty <= 0.0:
            raise ValueError("failure_penalty must be positive")


@dataclass(frozen=True)
class SystemCosts:
    """Already-normalized D.2/D.3 costs supplied by the caller for this slot."""

    cache_adjustment_cost: float
    uav_movement_cost: float

    def __post_init__(self) -> None:
        for name, value in (
            ("cache_adjustment_cost", self.cache_adjustment_cost),
            ("uav_movement_cost", self.uav_movement_cost),
        ):
            if _finite(name, value) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class SingleSlotRequest:
    """All observed inputs needed to execute one slot under a nominal matching.

    ``nominal_node_by_task`` is supplied by an upstream matching policy. This
    module validates it and solves only the fixed-matching P1-R acceptance and
    resource-allocation problem.
    """

    tasks: tuple[TaskDemand, ...]
    edge_nodes: tuple[EdgeNode, ...]
    cached_services_by_node: Mapping[str, frozenset[str]]
    channel_gains: Mapping[tuple[str, str], float]
    nominal_node_by_task: Mapping[str, str]
    radio: RadioParameters
    cloud: CloudParameters
    utility: UtilityParameters
    system_costs: SystemCosts


class TaskStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    DROPPED = "dropped"


@dataclass(frozen=True)
class TaskExecutionResult:
    task_id: str
    nominal_node_id: str
    execution_node_id: str | None
    status: TaskStatus
    bandwidth_fraction: float | None
    cpu_fraction: float | None
    rate_bps: float
    upload_delay_s: float | None
    compute_delay_s: float | None
    total_delay_s: float | None
    vehicle_energy_j: float
    utility: float


@dataclass(frozen=True)
class JointResourceAllocationResult:
    node_id: str
    task_ids: tuple[str, ...]
    bandwidth_fractions: tuple[tuple[str, float], ...]
    cpu_fractions: tuple[tuple[str, float], ...]
    feasibility: bool | None
    objective_cost_upper_bound: float | None
    solver_status: str
    solver_call_count: int
    optimality_certified: bool


@dataclass(frozen=True)
class P1RNodeSummary:
    node_id: str
    task_count: int
    accepted_task_ids: tuple[str, ...]
    enumerated_subset_count: int
    unresolved_subset_count: int
    continuous_solver_call_count: int
    solver_status_counts: tuple[tuple[str, int], ...]
    all_subsets_enumerated: bool
    optimality_certified: bool


@dataclass(frozen=True)
class P1RSummary:
    nodes: tuple[P1RNodeSummary, ...]
    optimality_certified: bool
    note: str


@dataclass(frozen=True)
class SingleSlotResult:
    task_results: tuple[TaskExecutionResult, ...]
    task_utility_sum: float
    cache_adjustment_cost: float
    uav_movement_cost: float
    system_reward: float
    p1r: P1RSummary


@dataclass(frozen=True)
class P1RSolverOptions:
    """Numerical solver controls; these do not alter physical constraints."""

    max_edge_subsets: int | None = None
    max_iterations: int = 500
    feasibility_tolerance: float = 1e-9
    optimality_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if self.max_edge_subsets is not None and self.max_edge_subsets <= 0:
            raise ValueError("max_edge_subsets must be positive when specified")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        for name, value in (
            ("feasibility_tolerance", self.feasibility_tolerance),
            ("optimality_tolerance", self.optimality_tolerance),
        ):
            if not 0.0 < _finite(name, value) < 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


def eligible_execution_nodes(
    task: TaskDemand,
    edge_nodes: Sequence[EdgeNode],
    cached_services_by_node: Mapping[str, frozenset[str]],
    cloud_node_id: str,
) -> tuple[str, ...]:
    """Return cached Edge candidates plus Cloud; local execution is not modeled."""

    edge_ids = [
        node.node_id
        for node in edge_nodes
        if task.service_id in cached_services_by_node[node.node_id]
    ]
    return tuple(sorted((*edge_ids, cloud_node_id)))


def validate_nominal_matching(request: SingleSlotRequest) -> dict[str, str]:
    """Validate exactly one legal nominal node per current task."""

    task_ids = [task.task_id for task in request.tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task_id values must be unique within a slot")
    edge_ids = [node.node_id for node in request.edge_nodes]
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError("edge node_id values must be unique")
    if request.cloud.node_id in set(edge_ids):
        raise ValueError("Cloud node_id must be distinct from every Edge node_id")
    if set(request.cached_services_by_node) != set(edge_ids):
        raise ValueError("cached_services_by_node must contain every Edge node exactly once")
    if set(request.nominal_node_by_task) != set(task_ids):
        raise ValueError("nominal matching must contain exactly one entry per task")

    edge_by_id = {node.node_id: node for node in request.edge_nodes}
    selected: dict[str, str] = {}
    for task in sorted(request.tasks, key=lambda item: item.task_id):
        node_id = request.nominal_node_by_task[task.task_id]
        candidates = eligible_execution_nodes(
            task,
            request.edge_nodes,
            request.cached_services_by_node,
            request.cloud.node_id,
        )
        if node_id not in candidates:
            raise ValueError(
                f"task {task.task_id!r} nominal node {node_id!r} is not a legal candidate; "
                f"candidates={candidates!r}"
            )
        if node_id != request.cloud.node_id:
            try:
                raw_gain = request.channel_gains[(task.task_id, node_id)]
            except KeyError as exc:
                raise ValueError(
                    f"missing channel gain for task {task.task_id!r} and node {node_id!r}"
                ) from exc
            gain = _finite(
                f"channel_gains[{task.task_id!r}, {node_id!r}]", raw_gain
            )
            if gain < 0.0:
                raise ValueError("channel gains must be non-negative")
            if node_id not in edge_by_id:
                raise ValueError(f"unknown Edge node {node_id!r}")
        selected[task.task_id] = node_id
    return selected


def _edge_rate(
    task: TaskDemand,
    node: EdgeNode,
    bandwidth_fraction: float,
    channel_gain: float,
    radio: RadioParameters,
) -> float:
    """Use the repository's shared v1 bandwidth-dependent Shannon formula."""

    # noise_power_w is unused by the explicit V1 branch; 0 avoids introducing a
    # second, conflicting fixed-total-noise value into this model interface.
    return uplink_rate(
        node.bandwidth_hz,
        bandwidth_fraction,
        task.transmit_power_w,
        channel_gain,
        0.0,
        noise_model=BANDWIDTH_DEPENDENT_NOISE_V1,
        noise_psd_w_per_hz=radio.noise_psd_w_per_hz,
        receiver_noise_figure_db=radio.receiver_noise_figure_db,
    )


def compute_task_utility(
    total_delay_s: float,
    vehicle_energy_j: float,
    is_success: bool,
    utility: UtilityParameters,
) -> float:
    """Apply the D.1 success/failure branches once for one task."""

    delay = float(total_delay_s)
    if math.isnan(delay) or delay < 0.0:
        raise ValueError("total_delay_s must be non-negative and not NaN")
    energy = _finite("vehicle_energy_j", vehicle_energy_j)
    if energy < 0.0:
        raise ValueError("vehicle_energy_j must be non-negative")
    if is_success:
        if not math.isfinite(delay) or delay > SLOT_DURATION_S:
            raise ValueError("a successful task must finish within the one-second slot")
        return (
            utility.omega_time * (1.0 - delay / utility.time_reference_s)
            + utility.omega_energy * (1.0 - energy / utility.energy_reference_j)
        )
    return -utility.failure_penalty - utility.omega_energy * (
        energy / utility.energy_reference_j
    )


def _dropped_result(
    task_id: str, nominal_node_id: str, utility: UtilityParameters
) -> TaskExecutionResult:
    return TaskExecutionResult(
        task_id=task_id,
        nominal_node_id=nominal_node_id,
        execution_node_id=None,
        status=TaskStatus.DROPPED,
        bandwidth_fraction=None,
        cpu_fraction=None,
        rate_bps=0.0,
        upload_delay_s=None,
        compute_delay_s=None,
        total_delay_s=None,
        vehicle_energy_j=0.0,
        utility=-utility.failure_penalty,
    )


def evaluate_task_execution(
    request: SingleSlotRequest,
    task: TaskDemand,
    execution_node_id: str,
    *,
    bandwidth_fraction: float | None = None,
    cpu_fraction: float | None = None,
) -> TaskExecutionResult:
    """Evaluate one actual execution using C.2/C.3, including partial-failure energy.

    For Edge, both fractions are required. Cloud uses its fixed per-task rate and
    CPU service. This evaluator permits a deadline failure for checking the D.1
    branch; P1-R itself admits only successful tasks and otherwise returns DROP.
    """

    matching = validate_nominal_matching(request)
    if matching.get(task.task_id) != execution_node_id:
        raise ValueError("actual execution node must equal the validated nominal node")

    return _evaluate_task_execution_validated(
        request,
        task,
        execution_node_id,
        bandwidth_fraction=bandwidth_fraction,
        cpu_fraction=cpu_fraction,
    )


def _evaluate_task_execution_validated(
    request: SingleSlotRequest,
    task: TaskDemand,
    execution_node_id: str,
    *,
    bandwidth_fraction: float | None = None,
    cpu_fraction: float | None = None,
) -> TaskExecutionResult:
    """Evaluate after the public interface has validated the matching."""

    if execution_node_id == request.cloud.node_id:
        if bandwidth_fraction is not None or cpu_fraction is not None:
            raise ValueError("Cloud resources are fixed per task and have no share variables")
        rate = request.cloud.upload_rate_bps
        upload_delay = task.input_bits / rate
        compute_delay = task.cpu_cycles / request.cloud.cpu_cycles_per_s
    else:
        node = next(node for node in request.edge_nodes if node.node_id == execution_node_id)
        if bandwidth_fraction is None or cpu_fraction is None:
            raise ValueError("Edge execution requires bandwidth_fraction and cpu_fraction")
        bandwidth_share = _finite("bandwidth_fraction", bandwidth_fraction)
        cpu_share = _finite("cpu_fraction", cpu_fraction)
        if not (0.0 < bandwidth_share <= 1.0 and 0.0 < cpu_share <= 1.0):
            raise ValueError("Edge resource fractions must be in (0, 1]")
        gain = request.channel_gains[(task.task_id, execution_node_id)]
        rate = _edge_rate(task, node, bandwidth_share, gain, request.radio)
        upload_delay = task.input_bits / rate if rate > 0.0 else math.inf
        compute_delay = task.cpu_cycles / (cpu_share * node.cpu_cycles_per_s)

    total_delay = upload_delay + compute_delay
    is_success = total_delay <= SLOT_DURATION_S
    uploaded_bits = min(task.input_bits, rate * SLOT_DURATION_S) if rate > 0.0 else 0.0
    energy = task.transmit_power_w * uploaded_bits / rate if rate > 0.0 else 0.0
    task_utility = compute_task_utility(
        total_delay, energy, is_success, request.utility
    )
    return TaskExecutionResult(
        task_id=task.task_id,
        nominal_node_id=execution_node_id,
        execution_node_id=execution_node_id,
        status=TaskStatus.SUCCESS if is_success else TaskStatus.FAILED,
        bandwidth_fraction=bandwidth_fraction,
        cpu_fraction=cpu_fraction,
        rate_bps=rate,
        upload_delay_s=upload_delay,
        compute_delay_s=compute_delay,
        total_delay_s=total_delay,
        vehicle_energy_j=energy,
        utility=task_utility,
    )


def compute_system_reward(
    task_results: Sequence[TaskExecutionResult], system_costs: SystemCosts
) -> tuple[float, float]:
    """Return ``Q(t)`` and ``G(t)=Q(t)-C_ca(t)-C_q(t)`` for one slot."""

    task_ids = [item.task_id for item in task_results]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("each task utility may be settled only once per slot")
    task_utility_sum = math.fsum(item.utility for item in task_results)
    reward = (
        task_utility_sum
        - system_costs.cache_adjustment_cost
        - system_costs.uav_movement_cost
    )
    return task_utility_sum, reward


def _success_cost_coefficients(
    task: TaskDemand, node: EdgeNode, utility: UtilityParameters
) -> tuple[float, float]:
    c_bandwidth = task.input_bits * (
        utility.omega_time / utility.time_reference_s
        + utility.omega_energy * task.transmit_power_w / utility.energy_reference_j
    )
    c_cpu = (
        utility.omega_time
        * task.cpu_cycles
        / (utility.time_reference_s * node.cpu_cycles_per_s)
    )
    return c_bandwidth, c_cpu


def _edge_subset_candidate(
    request: SingleSlotRequest,
    node: EdgeNode,
    tasks: tuple[TaskDemand, ...],
    options: P1RSolverOptions,
) -> tuple[dict[str, tuple[float, float]] | None, str, bool, int]:
    """Solve one fixed accepted subset; return a primal-feasible witness if found."""

    if len(tasks) == 1:
        task = tasks[0]
        rate = _edge_rate(
            task,
            node,
            1.0,
            request.channel_gains[(task.task_id, node.node_id)],
            request.radio,
        )
        solo_delay = (
            task.input_bits / rate + task.cpu_cycles / node.cpu_cycles_per_s
            if rate > 0.0
            else math.inf
        )
        if solo_delay <= SLOT_DURATION_S:
            return {task.task_id: (1.0, 1.0)}, "singleton_closed_form", True, 0
        return None, "singleton_infeasible", True, 0

    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover - depends on the selected environment
        raise ImportError(
            "P1-R multi-task joint allocation requires cvxpy and the CLARABEL solver "
            "in the adaptive_cache_mec environment; no package was installed"
        ) from exc

    def solve_with_margin(
        margin: float,
    ) -> tuple[dict[str, tuple[float, float]] | None, str]:
        count = len(tasks)
        bandwidth = cp.Variable(count, nonneg=True)
        cpu = cp.Variable(count, nonneg=True)
        # Scale the conic rate variable to Mbit/s. Keeping bit/s values near
        # 1e8 caused severe cone-solver conditioning without changing the model.
        rates_mbps = cp.Variable(count, nonneg=True)
        constraints = [
            cp.sum(bandwidth) <= 1.0 - margin,
            cp.sum(cpu) <= 1.0 - margin,
        ]
        objective_terms = []
        for index, task in enumerate(tasks):
            gain = request.channel_gains[(task.task_id, node.node_id)]
            noise_factor = 10.0 ** (request.radio.receiver_noise_figure_db / 10.0)
            gamma = (
                task.transmit_power_w
                * gain
                / (
                    request.radio.noise_psd_w_per_hz
                    * noise_factor
                    * node.bandwidth_hz
                )
            )
            constraints.append(
                cp.ExpCone(
                    rates_mbps[index]
                    * 1e6
                    * math.log(2.0)
                    / node.bandwidth_hz,
                    bandwidth[index],
                    bandwidth[index] + gamma,
                )
            )
            upload = task.input_bits / 1e6 * cp.inv_pos(rates_mbps[index])
            computation = (
                task.cpu_cycles
                / node.cpu_cycles_per_s
                * cp.inv_pos(cpu[index])
            )
            constraints.append(upload + computation <= SLOT_DURATION_S - margin)
            c_bandwidth, c_cpu = _success_cost_coefficients(
                task, node, request.utility
            )
            objective_terms.append(
                c_bandwidth / 1e6 * cp.inv_pos(rates_mbps[index])
                + c_cpu * cp.inv_pos(cpu[index])
            )
        problem = cp.Problem(cp.Minimize(cp.sum(cp.hstack(objective_terms))), constraints)
        try:
            problem.solve(
                solver=cp.CLARABEL,
                max_iter=options.max_iterations,
                tol_feas=options.feasibility_tolerance,
                tol_gap_abs=options.optimality_tolerance,
                tol_gap_rel=options.optimality_tolerance,
            )
        except cp.error.SolverError:
            # A solver error leaves this subset numerically unresolved. The
            # summary retains this status; it is not treated as infeasible.
            return None, "solver_error"
        status = str(problem.status)
        if bandwidth.value is None or cpu.value is None:
            return None, status
        point = {
            task.task_id: (float(bandwidth.value[index]), float(cpu.value[index]))
            for index, task in enumerate(tasks)
        }
        if _edge_point_is_feasible(request, node, tasks, point):
            return point, status
        return None, status

    point, status = solve_with_margin(0.0)
    if point is not None:
        return point, status, False, 1

    # A tighter numerical witness is only a recovery attempt for the original
    # constraints. It can supply a feasible incumbent, never an optimality proof.
    witness_margin = max(10.0 * options.feasibility_tolerance, 1e-10)
    if witness_margin < 1.0:
        witness, witness_status = solve_with_margin(witness_margin)
        if witness is not None:
            return witness, f"{status}; interior_witness:{witness_status}", False, 2
        return None, f"{status}; retry:{witness_status}", False, 2
    return None, status, False, 1


def _edge_point_is_feasible(
    request: SingleSlotRequest,
    node: EdgeNode,
    tasks: tuple[TaskDemand, ...],
    point: Mapping[str, tuple[float, float]],
) -> bool:
    bandwidth_sum = 0.0
    cpu_sum = 0.0
    for task in tasks:
        bandwidth_share, cpu_share = point[task.task_id]
        if not (
            math.isfinite(bandwidth_share)
            and math.isfinite(cpu_share)
            and 0.0 < bandwidth_share <= 1.0
            and 0.0 < cpu_share <= 1.0
        ):
            return False
        bandwidth_sum += bandwidth_share
        cpu_sum += cpu_share
        rate = _edge_rate(
            task,
            node,
            bandwidth_share,
            request.channel_gains[(task.task_id, node.node_id)],
            request.radio,
        )
        if rate <= 0.0:
            return False
        total_delay = (
            task.input_bits / rate
            + task.cpu_cycles / (cpu_share * node.cpu_cycles_per_s)
        )
        if total_delay > SLOT_DURATION_S:
            return False
    return bandwidth_sum <= 1.0 and cpu_sum <= 1.0


def allocate_joint_edge_resources(
    request: SingleSlotRequest,
    node_id: str,
    accepted_task_ids: Sequence[str],
    options: P1RSolverOptions = P1RSolverOptions(),
) -> JointResourceAllocationResult:
    """Jointly allocate one Edge node's bandwidth and CPU for a fixed subset.

    ``feasibility=None`` means the numerical solver did not produce a verified
    primal witness and did not provide a certified infeasibility result.
    ``objective_cost_upper_bound`` is present only for a primal-feasible point.
    """

    matching = validate_nominal_matching(request)
    node_by_id = {node.node_id: node for node in request.edge_nodes}
    if node_id not in node_by_id:
        raise ValueError(f"unknown Edge node {node_id!r}")
    task_by_id = {task.task_id: task for task in request.tasks}
    if len(accepted_task_ids) != len(set(accepted_task_ids)):
        raise ValueError("accepted_task_ids must be unique")
    unknown = set(accepted_task_ids) - set(task_by_id)
    if unknown:
        raise ValueError(f"unknown task ids in accepted subset: {sorted(unknown)!r}")
    wrong_node = [
        task_id for task_id in accepted_task_ids if matching[task_id] != node_id
    ]
    if wrong_node:
        raise ValueError(
            f"accepted tasks must be nominally matched to {node_id!r}: {wrong_node!r}"
        )

    tasks = tuple(
        sorted(
            (task_by_id[item] for item in accepted_task_ids),
            key=lambda item: item.task_id,
        )
    )
    if not tasks:
        return JointResourceAllocationResult(
            node_id=node_id,
            task_ids=(),
            bandwidth_fractions=(),
            cpu_fractions=(),
            feasibility=True,
            objective_cost_upper_bound=0.0,
            solver_status="empty_subset",
            solver_call_count=0,
            optimality_certified=True,
        )

    point, status, exact, calls = _edge_subset_candidate(
        request, node_by_id[node_id], tasks, options
    )
    if point is None:
        return JointResourceAllocationResult(
            node_id=node_id,
            task_ids=tuple(task.task_id for task in tasks),
            bandwidth_fractions=(),
            cpu_fractions=(),
            feasibility=False if exact else None,
            objective_cost_upper_bound=None,
            solver_status=status,
            solver_call_count=calls,
            optimality_certified=exact,
        )

    cost = 0.0
    bandwidth_items = []
    cpu_items = []
    for task in tasks:
        bandwidth_share, cpu_share = point[task.task_id]
        rate = _edge_rate(
            task,
            node_by_id[node_id],
            bandwidth_share,
            request.channel_gains[(task.task_id, node_id)],
            request.radio,
        )
        c_bandwidth, c_cpu = _success_cost_coefficients(
            task, node_by_id[node_id], request.utility
        )
        cost += c_bandwidth / rate + c_cpu / cpu_share
        bandwidth_items.append((task.task_id, bandwidth_share))
        cpu_items.append((task.task_id, cpu_share))
    return JointResourceAllocationResult(
        node_id=node_id,
        task_ids=tuple(task.task_id for task in tasks),
        bandwidth_fractions=tuple(bandwidth_items),
        cpu_fractions=tuple(cpu_items),
        feasibility=True,
        objective_cost_upper_bound=cost,
        solver_status=status,
        solver_call_count=calls,
        optimality_certified=exact,
    )


def _solve_edge_group(
    request: SingleSlotRequest,
    node: EdgeNode,
    tasks: tuple[TaskDemand, ...],
    options: P1RSolverOptions,
) -> tuple[dict[str, tuple[float, float]], P1RNodeSummary]:
    ordered = tuple(sorted(tasks, key=lambda item: item.task_id))
    subset_total = 1 << len(ordered)
    enumerate_count = subset_total
    if options.max_edge_subsets is not None:
        enumerate_count = min(subset_total, options.max_edge_subsets)

    best_value = -request.utility.failure_penalty * len(ordered)
    best_point: dict[str, tuple[float, float]] = {}
    best_ids: tuple[str, ...] = ()
    unresolved = subset_total - enumerate_count
    solver_calls = 0
    status_counts: dict[str, int] = {}

    for mask in range(enumerate_count):
        subset = tuple(
            ordered[index]
            for index in range(len(ordered))
            if mask & (1 << index)
        )
        if not subset:
            continue
        point, _status, exact, calls = _edge_subset_candidate(
            request, node, subset, options
        )
        solver_calls += calls
        if calls:
            status_counts[_status] = status_counts.get(_status, 0) + 1
        if point is None:
            if not exact:
                unresolved += 1
            continue

        candidate_value = 0.0
        for task in ordered:
            if task.task_id not in point:
                candidate_value -= request.utility.failure_penalty
                continue
            result = _evaluate_task_execution_validated(
                request,
                task,
                node.node_id,
                bandwidth_fraction=point[task.task_id][0],
                cpu_fraction=point[task.task_id][1],
            )
            if result.status is not TaskStatus.SUCCESS:
                # The original formulas are authoritative for final admission.
                unresolved += 1
                candidate_value = -math.inf
                break
            candidate_value += result.utility

        # Keeping the first exact tie gives a deterministic subset order; the
        # empty subset is visited first, so ties against all-DROP remain DROP.
        if candidate_value > best_value:
            best_value = candidate_value
            best_point = dict(point)
            best_ids = tuple(task.task_id for task in subset)

    complete = enumerate_count == subset_total
    # Multi-task conic solves yield numerical witnesses; this implementation
    # does not claim an L.8 primal/dual interval certificate from solver status.
    certified = len(ordered) <= 1 and complete and unresolved == 0
    summary = P1RNodeSummary(
        node_id=node.node_id,
        task_count=len(ordered),
        accepted_task_ids=best_ids,
        enumerated_subset_count=enumerate_count,
        unresolved_subset_count=unresolved,
        continuous_solver_call_count=solver_calls,
        solver_status_counts=tuple(sorted(status_counts.items())),
        all_subsets_enumerated=complete,
        optimality_certified=certified,
    )
    return best_point, summary


def solve_p1r(
    request: SingleSlotRequest,
    options: P1RSolverOptions = P1RSolverOptions(),
) -> tuple[tuple[TaskExecutionResult, ...], P1RSummary]:
    """Solve fixed-matching P1-R by Edge subset enumeration plus joint convex solves.

    The candidate matching is fixed on entry. Multi-task Edge results are
    feasible numerical incumbents only: complete enumeration is performed by
    default, but global optimality is not certified without verified L.8 bounds.
    The task-subset search is exponential, as specified by the exact P1-R model.
    """

    matching = validate_nominal_matching(request)
    edge_by_id = {node.node_id: node for node in request.edge_nodes}
    tasks_by_id = {task.task_id: task for task in request.tasks}
    tasks_by_node: dict[str, list[TaskDemand]] = {node.node_id: [] for node in request.edge_nodes}
    for task_id in sorted(tasks_by_id):
        task = tasks_by_id[task_id]
        node_id = matching[task_id]
        if node_id != request.cloud.node_id:
            tasks_by_node[node_id].append(task)

    accepted_resources: dict[str, tuple[float, float]] = {}
    node_summaries = []
    for node_id in sorted(tasks_by_node):
        node = edge_by_id[node_id]
        point, summary = _solve_edge_group(
            request,
            node,
            tuple(tasks_by_node[node_id]),
            options,
        )
        accepted_resources.update(point)
        node_summaries.append(summary)

    results: list[TaskExecutionResult] = []
    for task in sorted(request.tasks, key=lambda item: item.task_id):
        node_id = matching[task.task_id]
        if task.task_id in accepted_resources:
            bandwidth_share, cpu_share = accepted_resources[task.task_id]
            result = _evaluate_task_execution_validated(
                request,
                task,
                node_id,
                bandwidth_fraction=bandwidth_share,
                cpu_fraction=cpu_share,
            )
            if result.status is not TaskStatus.SUCCESS:
                raise RuntimeError(
                    f"P1-R returned a non-completing admitted task {task.task_id!r}"
                )
            results.append(result)
        elif node_id == request.cloud.node_id:
            cloud_result = _evaluate_task_execution_validated(request, task, node_id)
            if (
                cloud_result.status is TaskStatus.SUCCESS
                and cloud_result.utility > -request.utility.failure_penalty
            ):
                results.append(cloud_result)
            else:
                results.append(_dropped_result(task.task_id, node_id, request.utility))
        else:
            results.append(_dropped_result(task.task_id, node_id, request.utility))

    ordered_results = tuple(results)
    task_utility_sum, _ = compute_system_reward(ordered_results, request.system_costs)
    overall_certified = all(item.optimality_certified for item in node_summaries)
    summary = P1RSummary(
        nodes=tuple(node_summaries),
        optimality_certified=overall_certified,
        note=(
            "Singleton Edge groups and fixed Cloud decisions use closed-form "
            "evaluation. Multi-task Edge groups use CLARABEL feasible points; "
            "solver status alone is not an L.8 optimality certificate."
        ),
    )
    result = SingleSlotResult(
        task_results=ordered_results,
        task_utility_sum=task_utility_sum,
        cache_adjustment_cost=request.system_costs.cache_adjustment_cost,
        uav_movement_cost=request.system_costs.uav_movement_cost,
        system_reward=(
            task_utility_sum
            - request.system_costs.cache_adjustment_cost
            - request.system_costs.uav_movement_cost
        ),
        p1r=summary,
    )
    validate_execution_result(request, result)
    return ordered_results, summary


def execute_single_slot(
    request: SingleSlotRequest,
    options: P1RSolverOptions = P1RSolverOptions(),
) -> SingleSlotResult:
    """Execute one complete slot through nominal-matching P1-R and settle reward once."""

    task_results, summary = solve_p1r(request, options)
    task_utility_sum, system_reward = compute_system_reward(
        task_results, request.system_costs
    )
    result = SingleSlotResult(
        task_results=task_results,
        task_utility_sum=task_utility_sum,
        cache_adjustment_cost=request.system_costs.cache_adjustment_cost,
        uav_movement_cost=request.system_costs.uav_movement_cost,
        system_reward=system_reward,
        p1r=summary,
    )
    validate_execution_result(request, result)
    return result


def validate_execution_result(
    request: SingleSlotRequest, result: SingleSlotResult
) -> None:
    """Recompute formulas and check one-time settlement, resources, and deadlines."""

    matching = validate_nominal_matching(request)
    expected_ids = {task.task_id for task in request.tasks}
    actual_ids = [item.task_id for item in result.task_results]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("execution result contains a duplicate task settlement")
    if set(actual_ids) != expected_ids:
        raise ValueError("execution result must contain every current task exactly once")
    task_by_id = {task.task_id: task for task in request.tasks}
    edge_by_id = {node.node_id: node for node in request.edge_nodes}
    allocated_by_node: dict[str, list[TaskExecutionResult]] = {
        node.node_id: [] for node in request.edge_nodes
    }

    for actual in result.task_results:
        task = task_by_id[actual.task_id]
        if actual.nominal_node_id != matching[actual.task_id]:
            raise ValueError(f"task {actual.task_id!r} nominal node changed in result")
        if actual.status is TaskStatus.DROPPED:
            expected = _dropped_result(
                actual.task_id, matching[actual.task_id], request.utility
            )
        else:
            if actual.execution_node_id is None:
                raise ValueError("executed task must have an execution node")
            expected = evaluate_task_execution(
                request,
                task,
                actual.execution_node_id,
                bandwidth_fraction=actual.bandwidth_fraction,
                cpu_fraction=actual.cpu_fraction,
            )
            if expected.status is not actual.status:
                raise ValueError(
                    f"task {actual.task_id!r} status disagrees with the one-second completion rule"
                )
            if actual.execution_node_id != request.cloud.node_id:
                allocated_by_node[actual.execution_node_id].append(actual)

        for field_name in (
            "execution_node_id",
            "status",
            "bandwidth_fraction",
            "cpu_fraction",
        ):
            if getattr(actual, field_name) != getattr(expected, field_name):
                raise ValueError(f"task {actual.task_id!r} has inconsistent {field_name}")
        for field_name in (
            "rate_bps",
            "upload_delay_s",
            "compute_delay_s",
            "total_delay_s",
            "vehicle_energy_j",
            "utility",
        ):
            actual_value = getattr(actual, field_name)
            expected_value = getattr(expected, field_name)
            if actual_value is None or expected_value is None:
                if actual_value is not expected_value:
                    raise ValueError(f"task {actual.task_id!r} has inconsistent {field_name}")
            elif not math.isclose(
                actual_value,
                expected_value,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError(f"task {actual.task_id!r} has inconsistent {field_name}")

    for node_id, allocations in allocated_by_node.items():
        bandwidth_sum = math.fsum(item.bandwidth_fraction or 0.0 for item in allocations)
        cpu_sum = math.fsum(item.cpu_fraction or 0.0 for item in allocations)
        if bandwidth_sum > 1.0 or cpu_sum > 1.0:
            raise ValueError(
                f"Edge node {node_id!r} exceeds a resource budget: "
                f"bandwidth={bandwidth_sum}, cpu={cpu_sum}"
            )
        if any(item.status is not TaskStatus.SUCCESS for item in allocations):
            # P1-R only schedules tasks that complete. The standalone evaluator
            # still exposes D.1 failed execution for explicit formula checks.
            raise ValueError("P1-R result cannot contain an admitted failed Edge task")
        if node_id not in edge_by_id:
            raise ValueError(f"execution result references unknown Edge node {node_id!r}")

    task_utility_sum, reward = compute_system_reward(
        result.task_results, request.system_costs
    )
    if not math.isclose(
        result.task_utility_sum, task_utility_sum, rel_tol=1e-10, abs_tol=1e-12
    ):
        raise ValueError("task_utility_sum does not equal the one-time task utility sum")
    if not math.isclose(
        result.cache_adjustment_cost, request.system_costs.cache_adjustment_cost
    ):
        raise ValueError("cache adjustment cost differs from the request")
    if not math.isclose(
        result.uav_movement_cost, request.system_costs.uav_movement_cost
    ):
        raise ValueError("UAV movement cost differs from the request")
    if not math.isclose(result.system_reward, reward, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("system_reward does not equal Q - C_ca - C_q")


__all__ = [
    "SLOT_DURATION_S",
    "TaskDemand",
    "EdgeNode",
    "RadioParameters",
    "CloudParameters",
    "UtilityParameters",
    "SystemCosts",
    "SingleSlotRequest",
    "TaskStatus",
    "TaskExecutionResult",
    "JointResourceAllocationResult",
    "P1RNodeSummary",
    "P1RSummary",
    "SingleSlotResult",
    "P1RSolverOptions",
    "eligible_execution_nodes",
    "validate_nominal_matching",
    "compute_task_utility",
    "evaluate_task_execution",
    "compute_system_reward",
    "allocate_joint_edge_resources",
    "solve_p1r",
    "execute_single_slot",
    "validate_execution_result",
]
