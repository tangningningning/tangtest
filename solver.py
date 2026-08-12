"""论文确定性求解层。

本模块把一个时隙的“策略辅助动作”转换为论文定义的实际执行结果，主流程为：

1. ``solve_cooperative_cache``：对所有缓存节点统一执行边际收益贪心和 1-swap，
   生成 C_cand；``solve_cache`` 仅作为明确的 legacy baseline 保留；
2. ``decide_adaptive_cache_update``：比较 C_old/C_cand 后生成 C_exec；
3. ``select_demand_aware_uav_cache_state``：根据未来需求和地面服务缺口生成少量
   UAV 候选点，在速度、区域和安全距离约束下比较 A/B/C/D 联合状态；
4. ``build_feasible_nodes``：依据 C_exec 生成每个到达任务的可行节点集合，并
   始终加入 Cloud fallback；
5. ``transfer_exchange_matching``：以 Cloud 初始匹配，反复评价单任务迁移和
   两任务交换，直到没有正效用改进；
6. ``solve_resources``：对每个被选边缘节点分别执行计算和带宽 KKT 分配；
7. 调用 ``environment.py`` 计算真实执行、成本、``Q``、``G`` 和 reward。

通信、任务执行、时延、能耗、成本、utility 和 reward 全部调用 environment，
本文件不复制第二套物理公式，也不实现 PPO 训练。所有列布局都由输入数组和
显式 ``local_columns/service_columns/cloud_column`` 契约确定，避免为 F=10 创建
第二套代码路径。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:  # 允许作为 adaptive_cache_mec 包导入，也允许从项目根目录直接导入。
    from .environment import MECEnvironment, NodeType
except ImportError:  # pragma: no cover - 取决于调用方式
    from environment import MECEnvironment, NodeType


_DEFAULT_TOLERANCE = 1.0e-10
_DEFAULT_MULTIPLIER_TOLERANCE = 1.0e-12
_DEFAULT_POSITION_EPSILON = 1.0e-9


@dataclass(frozen=True, slots=True)
class SolverResult:
    """一次时隙确定性求解的完整结果。

    这些字段同时服务于 environment 状态回写、训练 rollout 和验收诊断；其中
    assignment、资源分配和 task_results 都是最终实际结果，而不是 policy 原始
    动作的替代表示。
    """

    # 候选缓存 C_cand；它由现有 cache trigger + knapsack 逻辑产生。
    candidate_cache: np.ndarray
    # 最终实际执行缓存 C_exec，后续可行节点、matching、KKT 和 reward 只读取它。
    final_cache: np.ndarray
    # C_exec 相对 C_old 是否真的发生了缓存变化，形状为 (M,)。
    actual_cache_update: np.ndarray
    # 自适应缓存判断的逐节点诊断字段。
    cache_candidate_changed: np.ndarray
    cache_trigger: np.ndarray
    cache_changed: np.ndarray
    cache_gain: np.ndarray
    cache_switch_cost: np.ndarray
    keep_predicted_utility: np.ndarray
    candidate_predicted_utility: np.ndarray
    cooperative_candidate_utility: float
    cache_duplicate_count: int
    cache_unique_services: int
    cache_local_swap_count: int
    services_changed: tuple[tuple[int, ...], ...]
    cache_check: bool
    # 经过 UAV 可行化后的二维水平位置，形状为 (U,2)。
    final_uav_positions: np.ndarray
    # 每个 UAV 是否真的移动，形状为 (U,)。
    actual_uav_movement: np.ndarray
    # 需求感知候选相对旧位置是否发生候选移动，形状为 (U,)。
    uav_candidate_movement: np.ndarray
    # 本时隙是否进行 UAV 候选检查及最终 A/B/C/D 选择。
    uav_check: bool
    uav_selected_action: str
    uav_J_keep: float
    uav_J_cache_only: float
    uav_J_uav_only: float
    uav_J_joint: float
    uav_candidate_count: int
    uav_candidate_move_cost: float
    ground_service_gap: float
    # 保留字段名用于兼容旧诊断；正式路径不再调用 SCA。
    uav_fallback: bool
    # 每个任务的可行执行列集合。
    feasible_nodes: tuple[tuple[int, ...], ...]
    # 任务到执行列的 0/1 assignment 矩阵。
    assignment: np.ndarray
    # matching 最终选择的节点名称，未到达任务为 None。
    execution_nodes: tuple[str | None, ...]
    # KKT 计算资源比例，列对应 service_columns 中的 edge node。
    compute_allocations: np.ndarray
    # KKT 带宽比例，列对应 service_columns 中的 edge node。
    bandwidth_allocations: np.ndarray
    # 每个任务调用 environment 后的实际 delay/energy 明细。
    task_results: tuple[Mapping[str, Any], ...]
    # Q、缓存成本、UAV成本和最终 reward 的分解。
    system_utility: Mapping[str, float]
    # transfer-exchange 是否在允许的更新范围内达到稳定。
    matching_stable: bool
    # Matching 实际接受的单任务迁移和双任务交换次数。
    matching_migration_count: int
    matching_swap_count: int
    # 最终时隙结果中的 KKT 失败次数和任务失败原因计数。
    kkt_failure_count: int
    failure_reason_counts: Mapping[str, int]


def _finite_array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name}必须是{ndim}维数组，实际为{array.ndim}维。")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}不能包含NaN或Inf。")
    return array


def _binary_array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    array = _finite_array(value, name, ndim=ndim)
    if not np.all((array == 0.0) | (array == 1.0)):
        raise ValueError(f"{name}必须是0/1数组。")
    return array


def _positive_array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    array = _finite_array(value, name, ndim=ndim)
    if np.any(array <= 0.0):
        raise ValueError(f"{name}必须全部为正数。")
    return array


def _nonnegative_array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    array = _finite_array(value, name, ndim=ndim)
    if np.any(array < 0.0):
        raise ValueError(f"{name}必须全部为非负数。")
    return array


def _bool_vector(value: Any, name: str, *, length: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name}必须是一维向量。")
    if length is not None and array.shape != (length,):
        raise ValueError(f"{name}形状必须为({length},)，实际为{array.shape}。")
    if array.dtype == bool:
        return array.astype(bool, copy=True)
    numeric = _finite_array(array, name, ndim=1)
    if not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ValueError(f"{name}必须是0/1向量。")
    return numeric.astype(bool)


def _vector(value: Any, name: str, length: int, *, positive: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(length, float(array), dtype=float)
    elif array.shape != (length,):
        raise ValueError(f"{name}必须是标量或形状({length},)，实际为{array.shape}。")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}不能包含NaN或Inf。")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name}必须全部为正数。")
    if not positive and np.any(array < 0.0):
        raise ValueError(f"{name}必须全部为非负数。")
    return array


def _readonly(value: Any, *, dtype: Any = float) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _node_kind(value: NodeType | str) -> NodeType:
    if isinstance(value, NodeType):
        return value
    try:
        return NodeType(str(value).strip().upper())
    except ValueError as error:
        allowed = ", ".join(item.value for item in NodeType)
        raise ValueError(f"节点类型必须是{allowed}之一。") from error


def _tuple_ints(value: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name}不能包含重复列。")
    if any(item < 0 for item in result):
        raise ValueError(f"{name}不能包含负列索引。")
    return result


def _validate_layout(
    *,
    task_count: int,
    service_count: int,
    local_columns: Sequence[int],
    service_columns: Sequence[int],
    cloud_column: int,
) -> tuple[tuple[int, ...], tuple[int, ...], int, int]:
    local = _tuple_ints(local_columns, "local_columns")
    service = _tuple_ints(service_columns, "service_columns")
    cloud = int(cloud_column)
    if cloud < 0 or len(local) != task_count or len(service) != service_count:
        raise ValueError("执行节点列数量与任务/服务节点数量不一致。")
    all_columns = (*local, *service, cloud)
    if len(set(all_columns)) != len(all_columns):
        raise ValueError("Local、边缘服务节点和Cloud列不能重叠。")
    node_count = max(all_columns, default=-1) + 1
    if set(all_columns) != set(range(node_count)):
        raise ValueError("执行节点列必须从0开始连续编号。")
    return local, service, cloud, node_count


# ---------------------------------------------------------------------------
# Cache: legacy baseline 与统一协同候选
# ---------------------------------------------------------------------------


def _integerize_cache_weights(
    service_sizes: np.ndarray,
    capacities: np.ndarray,
    integer_scale: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """把同一单位下的服务大小和容量精确转为整数。

    不做四舍五入。没有显式尺度时使用输入十进制表示的最小共同尺度；
    如果该尺度导致 DP 容量过大，直接报错，避免悄悄改论文的容量语义。
    """

    values = [float(item) for item in (*service_sizes.tolist(), *capacities.tolist())]
    if integer_scale is not None:
        scale = int(integer_scale)
        if scale <= 0:
            raise ValueError("integer_scale必须为正整数。")
    else:
        fractions = [Fraction(str(item)) for item in values]
        scale = 1
        for fraction in fractions:
            scale = math.lcm(scale, fraction.denominator)
            if scale > 10**9:
                raise ValueError(
                    "服务大小的最小精确整数尺度过大；请显式提供适合论文容量单位的integer_scale。"
                )

    def exact_units(
        items: np.ndarray,
        name: str,
        *,
        allow_zero: bool = False,
    ) -> np.ndarray:
        units: list[int] = []
        for item in items:
            product = Fraction(str(float(item))) * scale
            if product.denominator != 1:
                raise ValueError(f"{name}无法按integer_scale精确整数化，禁止四舍五入。")
            integer = int(product)
            if integer < 0 or (integer == 0 and not allow_zero):
                raise ValueError(f"{name}整数化后必须为正数。")
            units.append(integer)
        return np.asarray(units, dtype=np.int64)

    sizes_int = exact_units(service_sizes, "service_sizes")
    capacities_int = exact_units(capacities, "cache_capacities", allow_zero=True)
    if int(capacities_int.max(initial=0)) > 2_000_000:
        raise ValueError("整数化后的缓存容量过大，不适合当前确定性DP；请使用论文确认的容量单位。")
    return sizes_int, capacities_int, scale


def _better_knapsack_state(
    candidate: tuple[float, int],
    incumbent: tuple[float, int] | None,
    *,
    service_count: int,
    tolerance: float,
) -> bool:
    if incumbent is None:
        return True
    candidate_value, candidate_mask = candidate
    incumbent_value, incumbent_mask = incumbent
    if candidate_value > incumbent_value + tolerance:
        return True
    if abs(candidate_value - incumbent_value) <= tolerance:
        candidate_indices = tuple(
            index for index in range(service_count) if candidate_mask & (1 << index)
        )
        incumbent_indices = tuple(
            index for index in range(service_count) if incumbent_mask & (1 << index)
        )
        return candidate_indices < incumbent_indices
    return False


def _solve_knapsack_dp(
    scores: np.ndarray,
    sizes: np.ndarray,
    capacity: int,
    *,
    tolerance: float,
) -> np.ndarray:
    """用容量索引的确定性 DP 求解一个节点的严格 0-1 背包。

    ``states[used]`` 保存“恰好使用 used 容量”时的最大 score 和服务 bitmask。
    每件服务只允许选择一次；同分时由 ``_better_knapsack_state`` 使用服务编号
    字典序打破平局，保证重复运行得到完全一致的缓存结果。
    """

    states: dict[int, tuple[float, int]] = {0: (0.0, 0)}
    # 每轮从旧 states 复制出“不选择当前服务”的分支，再写入选择分支。
    for service_index, (score, size) in enumerate(zip(scores, sizes)):
        next_states = dict(states)
        for used, state in states.items():
            new_used = used + int(size)
            if new_used > capacity:
                continue
            candidate = (state[0] + float(score), state[1] | (1 << service_index))
            if _better_knapsack_state(
                candidate,
                next_states.get(new_used),
                service_count=scores.size,
                tolerance=tolerance,
            ):
                next_states[new_used] = candidate
        states = next_states

    best: tuple[float, int] | None = None
    for state in states.values():
        if _better_knapsack_state(
            state,
            best,
            service_count=scores.size,
            tolerance=tolerance,
        ):
            best = state
    result = np.zeros(scores.size, dtype=float)
    if best is not None:
        result[[index for index in range(scores.size) if best[1] & (1 << index)]] = 1.0
    return result


def solve_cache(
    previous_cache: Sequence[Sequence[Real]] | np.ndarray,
    requested_triggers: Sequence[bool | Real] | np.ndarray,
    cache_scores: Sequence[Sequence[Real]] | np.ndarray,
    service_sizes: Sequence[Real] | np.ndarray,
    cache_capacities: Real | Sequence[Real] | np.ndarray,
    *,
    integer_scale: int | None = None,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """执行原有逐节点严格 0-1 knapsack，作为 Old/legacy baseline 返回 C_cand。

    正式 ``solve_slot`` 不再调用本函数。保留它是为了 Old vs New 对照和回归
    验证；requested_triggers 是 PPO 的辅助请求，本函数不决定是否执行候选，
    也不计算缓存成本。
    """

    if tolerance < 0.0 or not math.isfinite(float(tolerance)):
        raise ValueError("tolerance必须是有限非负数。")
    previous = _binary_array(previous_cache, "previous_cache", ndim=2)
    scores = np.asarray(cache_scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("cache_scores必须是二维数组。")
    if scores.shape != previous.shape:
        raise ValueError("cache_scores必须与previous_cache形状相同。")
    node_count, service_count = previous.shape
    requested = _bool_vector(requested_triggers, "requested_triggers", length=node_count)
    sizes = _positive_array(service_sizes, "service_sizes", ndim=1)
    if sizes.shape != (service_count,):
        raise ValueError(f"service_sizes形状必须为({service_count},)。")
    capacities = _vector(cache_capacities, "cache_capacities", node_count)
    sizes_int, capacities_int, _ = _integerize_cache_weights(
        sizes,
        capacities,
        integer_scale,
    )

    final = previous.copy()
    for node in range(node_count):
        if not requested[node]:
            continue
        # cache_scores 只对触发节点有动作语义；DP 的 value 是服务评分之和，
        # weight/capacity 使用同一整数化后的 MB 单位，禁止浮点四舍五入。
        row = scores[node]
        if not np.all(np.isfinite(row)):
            raise ValueError("触发缓存节点的cache_scores不能包含NaN或Inf。")
        if np.any(row <= 0.0) or np.any(row >= 1.0):
            raise ValueError("触发缓存节点的cache_scores必须全部位于开区间(0,1)。")
        final[node] = _solve_knapsack_dp(
            row,
            sizes_int,
            int(capacities_int[node]),
            tolerance=float(tolerance),
        )

    final_usage = np.sum(final * sizes[np.newaxis, :], axis=1)
    if np.any(final_usage > capacities + tolerance):
        raise ValueError("最终缓存违反容量约束。")
    # 实际更新由最终矩阵差分得到，不能直接使用 requested trigger 代替。
    actual_update = np.any(final != previous, axis=1)
    return {
        "final_cache": _readonly(final),
        "candidate_cache": _readonly(final),
        "actual_cache_update": _readonly(actual_update, dtype=bool),
        "requested_triggers": _readonly(requested, dtype=bool),
    }


def _prediction_demand(
    cache_prediction: Mapping[str, Any] | None,
    node_count: int,
    service_count: int,
) -> np.ndarray:
    """读取节点级预测需求；只作为缺少覆盖元数据时的兼容回退。"""

    raw: Any = None
    if isinstance(cache_prediction, Mapping):
        raw = cache_prediction.get("predicted_demand")
        if raw is None:
            raw = cache_prediction.get("predicted_popularity")
    if raw is None:
        return np.full((node_count, service_count), 1.0 / service_count, dtype=float)
    demand = np.asarray(raw, dtype=float)
    if demand.shape != (node_count, service_count):
        raise ValueError(
            "cache_prediction.predicted_demand必须与缓存形状一致："
            f"期望{(node_count, service_count)}，实际{demand.shape}。"
        )
    if not np.all(np.isfinite(demand)) or np.any(demand < 0.0):
        raise ValueError("cache_prediction.predicted_demand不能包含NaN/Inf或负值。")
    return demand


def _prediction_preferences(
    cache_prediction: Mapping[str, Any] | None,
    vehicle_count: int,
    service_count: int,
) -> np.ndarray:
    """将 Stage 3 的车辆历史偏好整理为 V×F 矩阵。"""

    preferences = None if not isinstance(cache_prediction, Mapping) else cache_prediction.get(
        "vehicle_preferences"
    )
    vehicle_ids = () if not isinstance(cache_prediction, Mapping) else tuple(
        str(item) for item in cache_prediction.get("vehicle_ids", ())
    )
    result = np.zeros((vehicle_count, service_count), dtype=float)
    if isinstance(preferences, Mapping) and vehicle_ids:
        for index, vehicle_id in enumerate(vehicle_ids[:vehicle_count]):
            values = np.asarray(preferences.get(vehicle_id), dtype=float)
            if values.shape != (service_count,):
                raise ValueError("vehicle_preferences的服务维度不一致。")
            result[index] = values
    elif preferences is not None:
        values = np.asarray(preferences, dtype=float)
        if values.shape == (vehicle_count, service_count):
            result = values.copy()
        else:
            raise ValueError("vehicle_preferences必须是车辆×服务矩阵。")

    valid = np.sum(result, axis=1) > 0.0
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("vehicle_preferences不能包含NaN/Inf或负值。")
    if not np.all(valid):
        # 缺少逐车偏好时使用同一份历史全局先验，保持旧输入可运行；正式
        # Stage 3 预测会提供完整 vehicle_preferences，不会读取真实未来请求。
        fallback = np.full(service_count, 1.0 / service_count, dtype=float)
        if isinstance(cache_prediction, Mapping):
            raw = cache_prediction.get("predicted_popularity")
            if raw is not None:
                popularity = np.asarray(raw, dtype=float)
                if popularity.shape == (len(cache_prediction.get("node_ids", ())), service_count):
                    mean = np.mean(popularity, axis=0)
                    if np.all(np.isfinite(mean)) and float(mean.sum()) > 0.0:
                        fallback = mean / float(mean.sum())
        result[~valid] = fallback
    row_sums = result.sum(axis=1, keepdims=True)
    result = np.divide(
        result,
        row_sums,
        out=np.full_like(result, 1.0 / service_count),
        where=row_sums > 0.0,
    )
    return result


def _unit_multiplier(unit: Any, *, kind: str) -> float:
    """解析配置中的代表性 workload 单位，不读取当前或未来真实请求。"""

    value = str(unit).strip().lower().replace(" ", "")
    if kind == "bits":
        table = {
            "bit": 1.0,
            "bits": 1.0,
            "kbit": 1.0e3,
            "kbits": 1.0e3,
            "mbit": 1.0e6,
            "mbits": 1.0e6,
            "gbit": 1.0e9,
            "gbits": 1.0e9,
            "byte": 8.0,
            "bytes": 8.0,
            "kb": 8.0e3,
            "kbyte": 8.0e3,
            "kbytes": 8.0e3,
            "mb": 8.0e6,
            "mbyte": 8.0e6,
            "mbytes": 8.0e6,
        }
    else:
        table = {
            "cycle": 1.0,
            "cycles": 1.0,
            "kcycle": 1.0e3,
            "kcycles": 1.0e3,
            "mcycle": 1.0e6,
            "mcycles": 1.0e6,
            "gcycle": 1.0e9,
            "gcycles": 1.0e9,
        }
    if value not in table:
        raise ValueError(f"不支持的workload单位：{unit}")
    return table[value]


def _representative_workload(environment: MECEnvironment) -> tuple[float, float]:
    """使用配置区间中点建立轻量预测执行成本，不使用真实请求样本。"""

    workload = environment.config.get("workload", {})
    data_min = float(workload.get("task_input_size_min", 1.0))
    data_max = float(workload.get("task_input_size_max", data_min))
    cycle_min = float(workload.get("task_cpu_cycles_min", 1.0))
    cycle_max = float(workload.get("task_cpu_cycles_max", cycle_min))
    input_bits = 0.5 * (data_min + data_max) * _unit_multiplier(
        workload.get("task_input_size_unit", "Mbit"), kind="bits"
    )
    cpu_cycles = 0.5 * (cycle_min + cycle_max) * _unit_multiplier(
        workload.get("task_cpu_cycles_unit", "Gcycle"), kind="cycles"
    )
    if not all(math.isfinite(value) and value > 0.0 for value in (input_bits, cpu_cycles)):
        raise ValueError("代表性预测任务规模必须是有限正数。")
    return input_bits, cpu_cycles


def _prepare_predicted_cache_utility(
    *,
    cache_prediction: Mapping[str, Any] | None,
    node_types: Sequence[NodeType | str],
    environment: MECEnvironment,
    node_count: int,
    service_count: int,
) -> Callable[[np.ndarray], float]:
    """建立统一的预测缓存效用 oracle，供候选生成和触发判断共同调用。

    正式输入包含 Stage 3 的车辆-节点-未来时隙覆盖关系。对每个预测的
    ``(v,f,h)``，只取能够覆盖该车辆的已缓存节点中预测执行效用最高者，因而
    重叠节点不会机械地重复累加；空间分离节点仍可分别获得收益。执行成本只
    使用配置 workload 中点和完整边缘资源比例，不运行 Matching/KKT。
    """

    kinds = tuple(_node_kind(item) for item in node_types)
    if len(kinds) != node_count:
        raise ValueError("node_types数量必须等于缓存节点数量。")

    raw_coverage = (
        None
        if not isinstance(cache_prediction, Mapping)
        else cache_prediction.get("predicted_coverage")
    )
    coverage = None if raw_coverage is None else np.asarray(raw_coverage, dtype=float)
    if coverage is not None:
        if coverage.ndim != 3 or coverage.shape[0] != node_count:
            raise ValueError("predicted_coverage必须是(M,V,H)数组。")
        if not np.all(np.isfinite(coverage)) or np.any(coverage < 0.0):
            raise ValueError("predicted_coverage不能包含NaN/Inf或负值。")

    if coverage is None:
        demand = _prediction_demand(cache_prediction, node_count, service_count)
        node_weights = np.ones(node_count, dtype=float)
        if isinstance(cache_prediction, Mapping) and cache_prediction.get("coverage_mass") is not None:
            node_weights = np.asarray(cache_prediction["coverage_mass"], dtype=float)
            if node_weights.shape != (node_count,) or not np.all(np.isfinite(node_weights)):
                raise ValueError("coverage_mass必须是有限的节点向量。")
            node_weights = np.maximum(node_weights, 0.0)
        if not np.any(node_weights > 0.0):
            node_weights = np.ones(node_count, dtype=float)
        node_weights /= float(np.max(node_weights))
        fallback_utility = demand * node_weights[:, np.newaxis]

        def evaluate_fallback(cache: np.ndarray) -> float:
            return float(np.sum(cache * fallback_utility))

        return evaluate_fallback

    vehicle_count = int(coverage.shape[1])
    horizon = int(coverage.shape[2])
    preferences = _prediction_preferences(cache_prediction, vehicle_count, service_count)
    raw_weights = (
        None
        if not isinstance(cache_prediction, Mapping)
        else cache_prediction.get("horizon_weights")
    )
    if raw_weights is None:
        horizon_weights = np.full(horizon, 1.0 / horizon, dtype=float)
    else:
        horizon_weights = np.asarray(raw_weights, dtype=float)
        if horizon_weights.shape != (horizon,):
            raise ValueError("horizon_weights长度必须等于预测时域。")
        if not np.all(np.isfinite(horizon_weights)) or np.any(horizon_weights < 0.0):
            raise ValueError("horizon_weights不能包含NaN/Inf或负值。")
        total_weight = float(horizon_weights.sum())
        if total_weight <= 0.0:
            raise ValueError("horizon_weights总和必须为正。")
        horizon_weights = horizon_weights / total_weight

    edge_benefit = np.ones((node_count, vehicle_count, horizon), dtype=float)
    raw_node_positions = (
        None
        if not isinstance(cache_prediction, Mapping)
        else cache_prediction.get("node_positions_m")
    )
    raw_vehicle_positions = (
        None
        if not isinstance(cache_prediction, Mapping)
        else cache_prediction.get("vehicle_positions_m")
    )
    node_positions = None if raw_node_positions is None else np.asarray(raw_node_positions, dtype=float)
    vehicle_positions = (
        None if raw_vehicle_positions is None else np.asarray(raw_vehicle_positions, dtype=float)
    )
    have_positions = (
        node_positions is not None
        and vehicle_positions is not None
        and node_positions.shape == (node_count, horizon, 2)
        and vehicle_positions.shape == (vehicle_count, horizon, 2)
        and np.all(np.isfinite(node_positions))
        and np.all(np.isfinite(vehicle_positions))
    )
    if have_positions:
        input_bits, cpu_cycles = _representative_workload(environment)
        try:
            cloud_execution = environment.execute_task(
                NodeType.CLOUD,
                arrived=True,
                input_bits=input_bits,
                cpu_cycles=cpu_cycles,
            )
            cloud_utility = float(
                environment.task_utility(
                    np.asarray([True]),
                    np.asarray([cloud_execution["completion_delay_s"]]),
                    np.asarray([cloud_execution["vehicle_energy_j"]]),
                )["total_task_utility"]
            )
        except (ValueError, FloatingPointError, KeyError):
            cloud_utility = 0.0
        for node_index, kind in enumerate(kinds):
            for vehicle_index in range(vehicle_count):
                for horizon_index in range(horizon):
                    if coverage[node_index, vehicle_index, horizon_index] <= 0.0:
                        edge_benefit[node_index, vehicle_index, horizon_index] = 0.0
                        continue
                    try:
                        kwargs: dict[str, Any] = {
                            "arrived": True,
                            "input_bits": input_bits,
                            "cpu_cycles": cpu_cycles,
                            "compute_resource_fraction": 1.0,
                            "bandwidth_fraction": 1.0,
                            "vehicle_position": vehicle_positions[vehicle_index, horizon_index],
                        }
                        if kind is NodeType.UAV:
                            kwargs["uav_position"] = node_positions[node_index, horizon_index]
                        else:
                            kwargs["node_position"] = node_positions[node_index, horizon_index]
                        edge_execution = environment.execute_task(kind, **kwargs)
                        edge_utility = float(
                            environment.task_utility(
                                np.asarray([True]),
                                np.asarray([edge_execution["completion_delay_s"]]),
                                np.asarray([edge_execution["vehicle_energy_j"]]),
                            )["total_task_utility"]
                        )
                        edge_benefit[node_index, vehicle_index, horizon_index] = max(
                            0.0, edge_utility - cloud_utility
                        )
                    except (ValueError, FloatingPointError, KeyError):
                        edge_benefit[node_index, vehicle_index, horizon_index] = 0.0

    def evaluate(cache: np.ndarray) -> float:
        # shape: node × vehicle × service × horizon; max over nodes models the
        # best available edge service and suppresses duplicate benefit in overlap.
        available = coverage[:, :, np.newaxis, :] > 0.0
        cached = cache[:, np.newaxis, :, np.newaxis] > 0.0
        covered_benefit = np.where(
            available & cached,
            edge_benefit[:, :, np.newaxis, :],
            0.0,
        )
        best_benefit = np.max(covered_benefit, axis=0)
        weighted = best_benefit * preferences[:, :, np.newaxis]
        weighted *= horizon_weights[np.newaxis, np.newaxis, :]
        total = float(np.sum(weighted))
        if not math.isfinite(total):
            raise ValueError("predicted_cache_utility结果包含NaN/Inf。")
        return total

    return evaluate


def predicted_cache_utility(
    cache: Sequence[Sequence[Real]] | np.ndarray,
    *,
    cache_prediction: Mapping[str, Any] | None,
    node_types: Sequence[NodeType | str],
    environment: MECEnvironment,
) -> float:
    """计算协同缓存的预测系统效用，候选和第四阶段 trigger 共用此函数。"""

    checked = _binary_array(cache, "cache", ndim=2)
    evaluator = _prepare_predicted_cache_utility(
        cache_prediction=cache_prediction,
        node_types=node_types,
        environment=environment,
        node_count=checked.shape[0],
        service_count=checked.shape[1],
    )
    return float(evaluator(checked))


def _cache_replication_stats(cache: np.ndarray) -> tuple[int, int]:
    """返回缓存矩阵中的重复副本数和已出现的唯一服务数。"""

    copies_by_service = np.sum(cache, axis=0)
    duplicate_count = int(np.sum(np.maximum(copies_by_service - 1.0, 0.0)))
    unique_services = int(np.sum(copies_by_service > 0.0))
    return duplicate_count, unique_services


def solve_cooperative_cache(
    previous_cache: Sequence[Sequence[Real]] | np.ndarray,
    requested_triggers: Sequence[bool | Real] | np.ndarray,
    cache_scores: Sequence[Sequence[Real]] | np.ndarray | None,
    service_sizes: Sequence[Real] | np.ndarray,
    cache_capacities: Real | Sequence[Real] | np.ndarray,
    *,
    cache_prediction: Mapping[str, Any] | None,
    node_types: Sequence[NodeType | str],
    environment: MECEnvironment,
    integer_scale: int | None = None,
    tolerance: float = _DEFAULT_TOLERANCE,
    max_local_swaps: int = 1,
) -> dict[str, Any]:
    """统一生成 SV/RSU/UAV 的协同缓存候选 ``C_cand``。

    该实现保留 policy 的 cache trigger 作为一次全局候选检查开关，但不再对
    每个节点分别求热门服务。只要有节点请求检查，就在统一缓存矩阵上执行：

    1. 从当前缓存出发，在每个节点的真实容量内选择最大正边际收益/服务大小；
    2. 贪心结束后尝试最多 ``max_local_swaps`` 次 1-swap，允许删除低价值服务
       后装入更高价值服务；
    3. 边际收益由 ``predicted_cache_utility`` 统一计算，重叠覆盖只计最佳节点，
       空间分离的重复副本仍可分别产生收益。

    ``cache_scores`` 仅为兼容旧 policy 接口保留，不参与正式协同效用排序；
    ``solve_cache`` 保留为明确的 legacy baseline，不再由 ``solve_slot`` 调用。
    """

    if tolerance < 0.0 or not math.isfinite(float(tolerance)):
        raise ValueError("tolerance必须是有限非负数。")
    if int(max_local_swaps) < 0:
        raise ValueError("max_local_swaps不能为负数。")
    previous = _binary_array(previous_cache, "previous_cache", ndim=2)
    node_count, service_count = previous.shape
    requested = _bool_vector(requested_triggers, "requested_triggers", length=node_count)
    if cache_scores is not None:
        scores = np.asarray(cache_scores, dtype=float)
        if scores.shape != previous.shape:
            raise ValueError("cache_scores必须与previous_cache形状相同。")
    sizes = _positive_array(service_sizes, "service_sizes", ndim=1)
    if sizes.shape != (service_count,):
        raise ValueError(f"service_sizes形状必须为({service_count},)。")
    capacities = _vector(cache_capacities, "cache_capacities", node_count)
    kinds = tuple(_node_kind(item) for item in node_types)
    if len(kinds) != node_count:
        raise ValueError("node_types数量必须等于缓存节点数量。")
    sizes_int, capacities_int, _ = _integerize_cache_weights(
        sizes,
        capacities,
        integer_scale,
    )
    current_usage = np.sum(previous * sizes[np.newaxis, :], axis=1)
    if np.any(current_usage > capacities + tolerance):
        raise ValueError("previous_cache违反节点缓存容量约束。")

    candidate = previous.copy()
    duplicate_count, unique_services = _cache_replication_stats(candidate)
    if not np.any(requested):
        # 没有候选检查请求时不建立 utility oracle，保持第四阶段“候选相同
        # 直接短路”的因果语义，并且不把 policy 的空动作改写成缓存更新。
        return {
            "candidate_cache": _readonly(candidate),
            "final_cache": _readonly(candidate),
            "actual_cache_update": _readonly(
                np.zeros(node_count, dtype=bool), dtype=bool
            ),
            "cooperative_candidate_utility": 0.0,
            "cache_duplicate_count": duplicate_count,
            "cache_unique_services": unique_services,
            "cache_local_swap_count": 0,
            "requested_triggers": _readonly(requested, dtype=bool),
        }

    # 触发是全局协同检查开关；一旦进入候选生成，所有节点共享同一个
    # marginal-gain oracle，避免“只有被触发节点才能看到协同替代关系”。
    evaluator = _prepare_predicted_cache_utility(
        cache_prediction=cache_prediction,
        node_types=kinds,
        environment=environment,
        node_count=node_count,
        service_count=service_count,
    )
    usage_int = np.sum(candidate * sizes_int[np.newaxis, :], axis=1).astype(np.int64)
    current_utility = float(evaluator(candidate))
    addition_count = 0

    def best_addition() -> tuple[float, float, int, int, np.ndarray] | None:
        best: tuple[float, float, int, int, np.ndarray] | None = None
        for node_index in range(node_count):
            for service_index in range(service_count):
                if candidate[node_index, service_index] > 0.0:
                    continue
                if usage_int[node_index] + sizes_int[service_index] > capacities_int[node_index]:
                    continue
                trial = candidate.copy()
                trial[node_index, service_index] = 1.0
                gain = float(evaluator(trial) - current_utility)
                if gain <= tolerance:
                    continue
                density = gain / float(sizes[service_index])
                key = (density, gain, -node_index, -service_index)
                if best is None or key > (best[0], best[1], -best[2], -best[3]):
                    best = (density, gain, node_index, service_index, trial)
        return best

    while True:
        selected = best_addition()
        if selected is None:
            break
        _, gain, node_index, service_index, candidate = selected
        usage_int[node_index] += sizes_int[service_index]
        current_utility += float(gain)
        addition_count += 1

    swap_count = 0
    for _ in range(int(max_local_swaps)):
        best: tuple[float, float, int, int, int, np.ndarray] | None = None
        for node_index in range(node_count):
            cached_services = np.flatnonzero(candidate[node_index] > 0.0)
            missing_services = np.flatnonzero(candidate[node_index] <= 0.0)
            for remove_index in cached_services:
                for add_index in missing_services:
                    new_usage = (
                        usage_int[node_index]
                        - sizes_int[remove_index]
                        + sizes_int[add_index]
                    )
                    if new_usage > capacities_int[node_index]:
                        continue
                    trial = candidate.copy()
                    trial[node_index, remove_index] = 0.0
                    trial[node_index, add_index] = 1.0
                    gain = float(evaluator(trial) - current_utility)
                    if gain <= tolerance:
                        continue
                    density = gain / float(sizes[add_index])
                    key = (density, gain, -node_index, -remove_index, -add_index)
                    if best is None or key > (
                        best[0], best[1], -best[2], -best[3], -best[4]
                    ):
                        best = (
                            density,
                            gain,
                            node_index,
                            int(remove_index),
                            int(add_index),
                            trial,
                        )
        if best is None:
            break
        _, gain, node_index, remove_index, add_index, candidate = best
        usage_int[node_index] = (
            usage_int[node_index] - sizes_int[remove_index] + sizes_int[add_index]
        )
        current_utility += float(gain)
        swap_count += 1

    final_usage = np.sum(candidate * sizes[np.newaxis, :], axis=1)
    if np.any(final_usage > capacities + tolerance):
        raise ValueError("协同缓存候选违反容量约束。")
    duplicate_count, unique_services = _cache_replication_stats(candidate)
    return {
        "candidate_cache": _readonly(candidate),
        "final_cache": _readonly(candidate),
        "actual_cache_update": _readonly(
            np.any(candidate != previous, axis=1), dtype=bool
        ),
        "cooperative_candidate_utility": float(evaluator(candidate)),
        "cache_duplicate_count": duplicate_count,
        "cache_unique_services": unique_services,
        "cache_local_swap_count": int(swap_count),
        "cache_addition_count": int(addition_count),
        "requested_triggers": _readonly(requested, dtype=bool),
    }


def decide_adaptive_cache_update(
    old_cache: Sequence[Sequence[Real]] | np.ndarray,
    candidate_cache: Sequence[Sequence[Real]] | np.ndarray,
    *,
    cache_prediction: Mapping[str, Any] | None,
    service_sizes: Sequence[Real] | np.ndarray,
    node_types: Sequence[NodeType | str],
    environment: MECEnvironment,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """比较 C_old/C_cand，并决定最终执行缓存 C_exec。

    候选和触发共用 ``predicted_cache_utility``，因此同一个协同效用同时回答
    “候选如何排序”和“候选是否值得执行”。预测效用不包含切换成本，使用：
    ``G_cache = U_candidate - U_keep - C_switch``。真实 reward 仍只对
    ``C_old -> C_exec`` 调用一次已有缓存成本函数。
    """

    if tolerance < 0.0 or not math.isfinite(float(tolerance)):
        raise ValueError("tolerance必须是有限非负数。")
    old = _binary_array(old_cache, "old_cache", ndim=2)
    candidate = _binary_array(candidate_cache, "candidate_cache", ndim=2)
    if candidate.shape != old.shape:
        raise ValueError("old_cache和candidate_cache形状必须相同。")
    node_count, service_count = old.shape
    kinds = tuple(_node_kind(item) for item in node_types)
    if len(kinds) != node_count:
        raise ValueError("node_types数量必须等于缓存节点数量。")
    sizes = _positive_array(service_sizes, "service_sizes", ndim=1)
    if sizes.shape != (service_count,):
        raise ValueError("service_sizes长度必须等于服务数量。")
    candidate_changed = np.any(candidate != old, axis=1)
    switch_cost = np.zeros(node_count, dtype=float)
    services_changed: list[tuple[int, ...]] = []

    # 候选和当前相同是严格短路条件：不再计算更新收益，也不产生切换成本。
    if not np.any(candidate_changed):
        services_changed = [tuple() for _ in range(node_count)]
        return {
            "old_cache": _readonly(old),
            "candidate_cache": _readonly(candidate),
            "exec_cache": _readonly(old.copy()),
            "cache_candidate_changed": _readonly(candidate_changed, dtype=bool),
            "cache_trigger": _readonly(np.zeros(node_count, dtype=bool), dtype=bool),
            "cache_changed": _readonly(np.zeros(node_count, dtype=bool), dtype=bool),
            "cache_gain": _readonly(np.zeros(node_count, dtype=float)),
            "cache_switch_cost": _readonly(switch_cost),
            "keep_predicted_utility": _readonly(np.zeros(node_count, dtype=float)),
            "candidate_predicted_utility": _readonly(
                np.zeros(node_count, dtype=float)
            ),
            "cooperative_candidate_utility": 0.0,
            "cache_duplicate_count": _cache_replication_stats(candidate)[0],
            "cache_unique_services": _cache_replication_stats(candidate)[1],
            "cache_local_swap_count": 0,
            "services_changed": tuple(services_changed),
            "cache_check": True,
        }

    for node_index, kind in enumerate(kinds):
        changed_services = tuple(
            int(index)
            for index in np.flatnonzero(candidate[node_index] != old[node_index])
        )
        services_changed.append(changed_services)
        if changed_services:
            switch_cost[node_index] = float(
                environment.cache_update_cost(
                    old[node_index],
                    candidate[node_index],
                    node_type=kind,
                    service_sizes=sizes,
                )["total_cost"]
            )
    evaluator = _prepare_predicted_cache_utility(
        cache_prediction=cache_prediction,
        node_types=kinds,
        environment=environment,
        node_count=node_count,
        service_count=service_count,
    )
    keep_total = float(evaluator(old))
    candidate_total = float(evaluator(candidate))
    total_switch_cost = float(np.sum(switch_cost))
    total_gain = candidate_total - keep_total - total_switch_cost
    if not all(
        math.isfinite(value)
        for value in (keep_total, candidate_total, total_switch_cost, total_gain)
    ):
        raise ValueError("缓存预测收益或切换成本包含NaN/Inf。")
    # 协同候选是一个整体：正的全局净收益时同时执行所有候选变化，避免
    # 按节点拆开后破坏“重叠节点去重/分离节点允许重复”的边际收益判断。
    trigger = candidate_changed & (total_gain > 0.0)
    exec_cache = old.copy()
    exec_cache[trigger] = candidate[trigger]
    changed = np.any(exec_cache != old, axis=1)
    # 任何实际变化都必须来自正净收益触发，防止“更新但没有 trigger”。
    if np.any(changed & ~trigger):  # pragma: no cover - defensive invariant
        raise RuntimeError("cache_changed必须蕴含cache_trigger。")
    gain = np.where(candidate_changed, total_gain, 0.0)
    keep_utility = np.full(node_count, keep_total, dtype=float)
    candidate_utility = np.full(node_count, candidate_total, dtype=float)
    duplicate_count, unique_services = _cache_replication_stats(candidate)
    return {
        "old_cache": _readonly(old),
        "candidate_cache": _readonly(candidate),
        "exec_cache": _readonly(exec_cache),
        "cache_candidate_changed": _readonly(candidate_changed, dtype=bool),
        "cache_trigger": _readonly(trigger, dtype=bool),
        "cache_changed": _readonly(changed, dtype=bool),
        "cache_gain": _readonly(gain),
        "cache_switch_cost": _readonly(switch_cost),
        "keep_predicted_utility": _readonly(keep_utility),
        "candidate_predicted_utility": _readonly(candidate_utility),
        "cooperative_candidate_utility": candidate_total,
        "cache_duplicate_count": duplicate_count,
        "cache_unique_services": unique_services,
        "cache_local_swap_count": 0,
        "services_changed": tuple(services_changed),
        "cache_check": True,
    }


# ---------------------------------------------------------------------------
# UAV: existing constraints and demand-aware finite candidates
# ---------------------------------------------------------------------------


def _region_values(region: Mapping[str, Real] | Sequence[Real]) -> tuple[float, float, float, float]:
    if isinstance(region, Mapping):
        keys = ("x_min_m", "x_max_m", "y_min_m", "y_max_m")
        try:
            values = tuple(float(region[key]) for key in keys)
        except KeyError as error:
            raise ValueError("UAV region必须包含x_min_m、x_max_m、y_min_m、y_max_m。") from error
    else:
        values = tuple(float(item) for item in region)
        if len(values) != 4:
            raise ValueError("UAV region序列必须按(x_min,x_max,y_min,y_max)提供。")
    if not all(math.isfinite(item) for item in values):
        raise ValueError("UAV region不能包含NaN或Inf。")
    x_min, x_max, y_min, y_max = values
    if x_max < x_min or y_max < y_min:
        raise ValueError("UAV region上下界顺序错误。")
    return values


def _uav_pairs(count: int) -> tuple[tuple[int, int], ...]:
    return tuple((first, second) for first in range(count) for second in range(first + 1, count))


def _uav_original_constraints(
    positions: np.ndarray,
    previous: np.ndarray,
    requested: np.ndarray,
    maximum_displacement: np.ndarray,
    region: tuple[float, float, float, float],
    minimum_separation: float,
    tolerance: float,
) -> bool:
    """检查候选 UAV 位置是否直接满足原始 C5--C8 约束。"""

    if positions.shape != previous.shape:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    x_min, x_max, y_min, y_max = region
    if np.any(positions[:, 0] < x_min - tolerance) or np.any(positions[:, 0] > x_max + tolerance):
        return False
    if np.any(positions[:, 1] < y_min - tolerance) or np.any(positions[:, 1] > y_max + tolerance):
        return False
    distances = np.linalg.norm(positions - previous, axis=1)
    if np.any(distances > maximum_displacement * requested.astype(float) + tolerance):
        return False
    for first, second in _uav_pairs(positions.shape[0]):
        if np.linalg.norm(positions[first] - positions[second]) < minimum_separation - tolerance:
            return False
    return True


# ---------------------------------------------------------------------------
# Demand-aware finite UAV candidates and one-shot cache/UAV comparison
# ---------------------------------------------------------------------------


def _cache_state_cost(
    old_cache: np.ndarray,
    current_cache: np.ndarray,
    *,
    service_sizes: Sequence[Real] | np.ndarray,
    node_types: Sequence[NodeType | str],
    environment: MECEnvironment,
) -> float:
    """计算一个候选状态的真实缓存切换成本，不把预测效用与成本混合。"""

    total = 0.0
    for index, kind in enumerate(node_types):
        if np.array_equal(old_cache[index], current_cache[index]):
            continue
        total += float(
            environment.cache_update_cost(
                old_cache[index],
                current_cache[index],
                node_type=_node_kind(kind),
                service_sizes=service_sizes,
            )["total_cost"]
        )
    if not math.isfinite(total) or total < 0.0:
        raise ValueError("缓存状态切换成本必须是有限非负数。")
    return total


def _uav_state_cost(
    previous_positions: np.ndarray,
    current_positions: np.ndarray,
    *,
    environment: MECEnvironment,
    movement_epsilon_m: float,
) -> float:
    """按最终位置计算 UAV 的真实移动成本。"""

    total = 0.0
    for previous, current in zip(previous_positions, current_positions):
        total += float(
            environment.uav_movement_cost(
                previous,
                current,
                movement_epsilon_m=movement_epsilon_m,
            )["total_cost"]
        )
    if not math.isfinite(total) or total < 0.0:
        raise ValueError("UAV移动成本必须是有限非负数。")
    return total


def _prediction_at_uav_positions(
    cache_prediction: Mapping[str, Any] | None,
    candidate_positions: np.ndarray,
    uav_service_indices: Sequence[int],
    *,
    environment: MECEnvironment,
) -> Mapping[str, Any] | None:
    """只替换候选 UAV 的未来位置和覆盖，不读取任何真实请求。"""

    if not isinstance(cache_prediction, Mapping):
        return cache_prediction
    raw_nodes = cache_prediction.get("node_positions_m")
    raw_coverage = cache_prediction.get("predicted_coverage")
    raw_vehicle_positions = cache_prediction.get("vehicle_positions_m")
    if raw_nodes is None or raw_coverage is None or raw_vehicle_positions is None:
        return cache_prediction
    node_positions = np.asarray(raw_nodes, dtype=float)
    coverage = np.asarray(raw_coverage, dtype=float)
    vehicle_positions = np.asarray(raw_vehicle_positions, dtype=float)
    if node_positions.ndim != 3 or node_positions.shape[-1] != 2:
        return cache_prediction
    if coverage.ndim != 3 or vehicle_positions.ndim != 3:
        return cache_prediction
    if coverage.shape[0] != node_positions.shape[0] or coverage.shape[1:] != (
        vehicle_positions.shape[0],
        vehicle_positions.shape[1],
    ):
        return cache_prediction
    if not all(
        np.all(np.isfinite(value))
        for value in (node_positions, coverage, vehicle_positions, candidate_positions)
    ):
        raise ValueError("UAV候选位置预测元数据不能包含NaN/Inf。")
    result = dict(cache_prediction)
    node_positions = node_positions.copy()
    coverage = coverage.copy()
    horizon = vehicle_positions.shape[1]
    radius_config = environment.config.get("vehicle_roles", {})
    radius = float(radius_config.get("service_v2v_range_m", 300.0))
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("vehicle_roles.service_v2v_range_m必须是有限非负数。")
    for uav_order, service_index in enumerate(uav_service_indices):
        if service_index < 0 or service_index >= node_positions.shape[0]:
            raise ValueError("uav_service_indices超出预测节点范围。")
        position = candidate_positions[uav_order]
        node_positions[service_index] = np.repeat(
            position[np.newaxis, :], horizon, axis=0
        )
        coverage[service_index] = (
            np.linalg.norm(vehicle_positions - position[np.newaxis, np.newaxis, :], axis=2)
            <= radius + 1.0e-9
        ).astype(float)
    result["node_positions_m"] = node_positions
    result["predicted_coverage"] = coverage
    return result


def _ground_service_gap_hotspots(
    current_cache: np.ndarray,
    cache_prediction: Mapping[str, Any] | None,
    node_types: Sequence[NodeType | str],
    *,
    max_hotspots: int = 3,
) -> tuple[float, tuple[np.ndarray, ...]]:
    """用未来服务需求乘地面服务缺口生成少量热点评估位置。"""

    if not isinstance(cache_prediction, Mapping):
        return 0.0, tuple()
    raw_coverage = cache_prediction.get("predicted_coverage")
    raw_positions = cache_prediction.get("vehicle_positions_m")
    raw_preferences = cache_prediction.get("vehicle_preferences")
    vehicle_ids = tuple(str(item) for item in cache_prediction.get("vehicle_ids", ()))
    if raw_coverage is None or raw_positions is None or not isinstance(raw_preferences, Mapping):
        return 0.0, tuple()
    coverage = np.asarray(raw_coverage, dtype=float)
    vehicle_positions = np.asarray(raw_positions, dtype=float)
    if coverage.ndim != 3 or vehicle_positions.ndim != 3:
        return 0.0, tuple()
    node_count = current_cache.shape[0]
    if coverage.shape[0] != node_count or coverage.shape[1:] != (
        vehicle_positions.shape[0],
        vehicle_positions.shape[1],
    ):
        return 0.0, tuple()
    preferences = np.zeros((vehicle_positions.shape[0], current_cache.shape[1]), dtype=float)
    for index, vehicle_id in enumerate(vehicle_ids[: preferences.shape[0]]):
        values = np.asarray(raw_preferences.get(vehicle_id), dtype=float)
        if values.shape != (current_cache.shape[1],):
            return 0.0, tuple()
        preferences[index] = values
    if not np.all(np.isfinite(preferences)) or np.any(preferences < 0.0):
        raise ValueError("UAV热点评估的车辆偏好不能包含NaN/Inf或负值。")
    ground_indices = tuple(
        index
        for index, kind in enumerate(node_types)
        if _node_kind(kind) in {NodeType.SERVICE_VEHICLE, NodeType.RSU}
    )
    served = np.zeros(
        (vehicle_positions.shape[0], vehicle_positions.shape[1], current_cache.shape[1]),
        dtype=bool,
    )
    for node_index in ground_indices:
        served |= (
            (coverage[node_index] > 0.0)[:, :, np.newaxis]
            & (current_cache[node_index] > 0.0)[np.newaxis, np.newaxis, :]
        )
    # ground_service_gap[v,h] 只看预测请求偏好中没有被 SV/RSU 缓存服务的部分。
    gap = np.sum(
        np.where(served, 0.0, preferences[:, np.newaxis, :]),
        axis=2,
    )
    if not np.all(np.isfinite(gap)) or np.any(gap < 0.0):
        raise ValueError("ground_service_gap不能包含NaN/Inf或负值。")
    weights = gap.reshape(-1)
    positions = vehicle_positions.reshape(-1, 2)
    valid = (weights > 0.0) & np.all(np.isfinite(positions), axis=1)
    if not np.any(valid) or float(weights[valid].sum()) <= 0.0:
        return 0.0, tuple()
    valid_weights = weights[valid]
    valid_positions = positions[valid]
    hotspot_weights = float(valid_weights.sum())
    points: list[np.ndarray] = [
        np.average(valid_positions, axis=0, weights=valid_weights).astype(float)
    ]
    order = np.argsort(-valid_weights, kind="stable")
    for index in order[: max(0, int(max_hotspots))]:
        points.append(valid_positions[int(index)].astype(float))
    unique: list[np.ndarray] = []
    for point in points:
        if not any(np.linalg.norm(point - old) <= 1.0e-6 for old in unique):
            unique.append(point)
    return hotspot_weights, tuple(unique[: max(0, int(max_hotspots) + 1)])


def _project_uav_target(
    previous: np.ndarray,
    target: np.ndarray,
    *,
    maximum_displacement: float,
    region: tuple[float, float, float, float],
) -> np.ndarray:
    """把热点评估点投影到已有矩形飞行区和最大位移球内。"""

    x_min, x_max, y_min, y_max = region
    clipped = np.asarray(target, dtype=float).copy()
    clipped[0] = np.clip(clipped[0], x_min, x_max)
    clipped[1] = np.clip(clipped[1], y_min, y_max)
    delta = clipped - previous
    distance = float(np.linalg.norm(delta))
    if distance > maximum_displacement:
        clipped = previous + delta / distance * maximum_displacement
    return clipped


def _demand_uav_candidate_sets(
    previous_positions: np.ndarray,
    current_cache: np.ndarray,
    cache_prediction: Mapping[str, Any] | None,
    node_types: Sequence[NodeType | str],
    *,
    max_speed_mps: float,
    slot_duration_s: float,
    region: tuple[float, float, float, float],
    max_hotspots: int = 3,
) -> tuple[float, tuple[tuple[np.ndarray, ...], ...]]:
    """为每个 UAV 生成 current + 少量需求缺口热点候选。"""

    gap, hotspots = _ground_service_gap_hotspots(
        current_cache,
        cache_prediction,
        node_types,
        max_hotspots=max_hotspots,
    )
    maximum_displacement = float(max_speed_mps) * float(slot_duration_s)
    candidate_sets: list[tuple[np.ndarray, ...]] = []
    for previous_position in previous_positions:
        points: list[np.ndarray] = [previous_position.copy()]
        for hotspot in hotspots:
            candidate = _project_uav_target(
                previous_position,
                hotspot,
                maximum_displacement=maximum_displacement,
                region=region,
            )
            if not any(np.linalg.norm(candidate - old) <= 1.0e-6 for old in points):
                points.append(candidate)
        candidate_sets.append(tuple(points))
    return gap, tuple(candidate_sets)


def _evaluate_uav_cache_state(
    candidate_positions: np.ndarray,
    *,
    previous_cache: np.ndarray,
    requested_cache_triggers: Sequence[bool | Real] | np.ndarray,
    cache_scores: Sequence[Sequence[Real]] | np.ndarray | None,
    service_sizes: Sequence[Real] | np.ndarray,
    cache_capacities: Real | Sequence[Real] | np.ndarray,
    cache_prediction: Mapping[str, Any] | None,
    node_types: Sequence[NodeType | str],
    uav_service_indices: Sequence[int],
    environment: MECEnvironment,
    integer_scale: int | None,
) -> dict[str, Any]:
    """在临时 UAV 位置上重算轻量协同缓存状态，不修改真实环境。"""

    prediction = _prediction_at_uav_positions(
        cache_prediction,
        candidate_positions,
        uav_service_indices,
        environment=environment,
    )
    cache_result = solve_cooperative_cache(
        previous_cache,
        requested_cache_triggers,
        cache_scores,
        service_sizes,
        cache_capacities,
        cache_prediction=prediction,
        node_types=node_types,
        environment=environment,
        integer_scale=integer_scale,
    )
    candidate_cache = np.asarray(cache_result["candidate_cache"], dtype=float)
    adaptive = decide_adaptive_cache_update(
        previous_cache,
        candidate_cache,
        cache_prediction=prediction,
        service_sizes=service_sizes,
        node_types=node_types,
        environment=environment,
    )
    adaptive["cache_local_swap_count"] = int(cache_result.get("cache_local_swap_count", 0))
    adaptive["cache_duplicate_count"] = int(cache_result.get("cache_duplicate_count", 0))
    adaptive["cache_unique_services"] = int(cache_result.get("cache_unique_services", 0))
    final_cache = np.asarray(adaptive["exec_cache"], dtype=float)
    cache_cost = _cache_state_cost(
        previous_cache,
        final_cache,
        service_sizes=service_sizes,
        node_types=node_types,
        environment=environment,
    )
    keep_utility = predicted_cache_utility(
        previous_cache,
        cache_prediction=prediction,
        node_types=node_types,
        environment=environment,
    )
    exec_utility = predicted_cache_utility(
        final_cache,
        cache_prediction=prediction,
        node_types=node_types,
        environment=environment,
    )
    # PPO 的 UPDATE 分支必须能够接受候选，即使候选的短视净收益没有超过
    # C_old；因此同时保留“候选本身”的效用和实际切换成本，供高层动作分组使用。
    candidate_cache_cost = _cache_state_cost(
        previous_cache,
        candidate_cache,
        service_sizes=service_sizes,
        node_types=node_types,
        environment=environment,
    )
    candidate_utility = predicted_cache_utility(
        candidate_cache,
        cache_prediction=prediction,
        node_types=node_types,
        environment=environment,
    )
    return {
        "prediction": prediction,
        "candidate_cache": candidate_cache,
        "final_cache": final_cache,
        "adaptive": adaptive,
        "cache_cost": float(cache_cost),
        "keep_utility": float(keep_utility),
        "exec_utility": float(exec_utility),
        "candidate_cache_cost": float(candidate_cache_cost),
        "candidate_utility": float(candidate_utility),
    }


def select_demand_aware_uav_cache_state(
    previous_positions: Sequence[Sequence[Real]] | np.ndarray,
    *,
    previous_cache: Sequence[Sequence[Real]] | np.ndarray,
    requested_cache_triggers: Sequence[bool | Real] | np.ndarray,
    cache_scores: Sequence[Sequence[Real]] | np.ndarray | None,
    service_sizes: Sequence[Real] | np.ndarray,
    cache_capacities: Real | Sequence[Real] | np.ndarray,
    cache_prediction: Mapping[str, Any] | None,
    node_types: Sequence[NodeType | str],
    uav_service_indices: Sequence[int],
    environment: MECEnvironment,
    max_speed_mps: Real,
    slot_duration_s: Real,
    region: Mapping[str, Real] | Sequence[Real],
    minimum_separation_m: Real,
    position_epsilon_m: Real = _DEFAULT_POSITION_EPSILON,
    max_hotspots: int = 3,
    integer_scale: int | None = None,
    cache_control_action: int | None = None,
) -> dict[str, Any]:
    """一次有限候选 A/B/C/D 联合比较，返回最终 UAV 位置和缓存状态。

    A=保持(C_old,Q_old)，B=只更新缓存，C=只移动 UAV，D=移动 UAV 后重新
    生成协同缓存。所有状态共用 ``predicted_cache_utility``；候选评价只使用
    临时数组和预测结果，不写入环境、不执行 Matching/KKT，也不调用 SCA。
    """

    if cache_control_action not in (None, 0, 1):
        raise ValueError("cache_control_action只能是None、0(KEEP)或1(UPDATE)。")
    previous = _finite_array(previous_positions, "previous_positions", ndim=2)
    old_cache = _binary_array(previous_cache, "previous_cache", ndim=2)
    if previous.shape[1:] != (2,):
        raise ValueError("previous_positions必须形状为(U,2)。")
    uav_count = previous.shape[0]
    kinds = tuple(_node_kind(item) for item in node_types)
    if len(kinds) != old_cache.shape[0]:
        raise ValueError("node_types数量必须等于缓存节点数量。")
    uav_indices = _tuple_ints(uav_service_indices, "uav_service_indices")
    if len(uav_indices) != uav_count or any(
        index >= len(kinds) or kinds[index] is not NodeType.UAV for index in uav_indices
    ):
        raise ValueError("uav_service_indices必须与UAV缓存节点一一对应。")
    max_speed = float(max_speed_mps)
    slot_duration = float(slot_duration_s)
    minimum_separation = float(minimum_separation_m)
    epsilon = float(position_epsilon_m)
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (max_speed, slot_duration)
    ):
        raise ValueError("max_speed_mps和slot_duration_s必须是有限正数。")
    if not math.isfinite(minimum_separation) or minimum_separation < 0.0:
        raise ValueError("minimum_separation_m必须是有限非负数。")
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("position_epsilon_m必须是有限非负数。")
    region_values = _region_values(region)
    maximum_displacement = np.full(uav_count, max_speed * slot_duration, dtype=float)
    if not _uav_original_constraints(
        previous,
        previous,
        np.ones(uav_count, dtype=bool),
        maximum_displacement,
        region_values,
        minimum_separation,
        max(epsilon, _DEFAULT_TOLERANCE),
    ):
        raise ValueError("previous_positions必须满足已有UAV区域、速度和安全距离约束。")

    state_cache: dict[bytes, dict[str, Any]] = {}

    def evaluate(positions: np.ndarray) -> dict[str, Any]:
        checked = _finite_array(positions, "candidate_positions", ndim=2)
        key = checked.tobytes()
        if key not in state_cache:
            state_cache[key] = _evaluate_uav_cache_state(
                checked,
                previous_cache=old_cache,
                requested_cache_triggers=requested_cache_triggers,
                cache_scores=cache_scores,
                service_sizes=service_sizes,
                cache_capacities=cache_capacities,
                cache_prediction=cache_prediction,
                node_types=kinds,
                uav_service_indices=uav_indices,
                environment=environment,
                integer_scale=integer_scale,
            )
        return state_cache[key]

    gap, candidate_sets = _demand_uav_candidate_sets(
        previous,
        old_cache,
        cache_prediction,
        kinds,
        max_speed_mps=max_speed,
        slot_duration_s=slot_duration,
        region=region_values,
        max_hotspots=max_hotspots,
    )
    base_state = evaluate(previous)
    keep_utility = float(base_state["keep_utility"])
    cache_only_utility = float(base_state["exec_utility"])
    cache_only_cost = float(base_state["cache_cost"])
    J_keep = keep_utility
    # 这是不经过 G_cache 决策的候选效用，专供 PPO UPDATE 组使用；Reference
    # 模式仍继续使用下方原有的自适应 exec_cache 结果，保证 baseline 不变。
    forced_cache_only_utility = (
        float(base_state["candidate_utility"])
        - float(base_state["candidate_cache_cost"])
    )

    working = previous.copy()
    for uav_order, candidates in enumerate(candidate_sets):
        best_positions = working.copy()
        best_state = evaluate(best_positions)
        best_move_cost = _uav_state_cost(
            previous,
            best_positions,
            environment=environment,
            movement_epsilon_m=epsilon,
        )
        best_keep_score = predicted_cache_utility(
            old_cache,
            cache_prediction=best_state["prediction"],
            node_types=kinds,
            environment=environment,
        ) - best_move_cost
        best_joint_score = (
            best_state["exec_utility"] - best_state["cache_cost"] - best_move_cost
        )
        forced_best_joint_score = (
            best_state["candidate_utility"]
            - best_state["candidate_cache_cost"]
            - best_move_cost
        )
        if cache_control_action is None:
            best_score = float(max(best_keep_score, best_joint_score))
        elif cache_control_action == 0:
            best_score = float(best_keep_score)
        else:
            best_score = float(forced_best_joint_score)
        for candidate in candidates:
            trial = working.copy()
            trial[uav_order] = candidate
            if not _uav_original_constraints(
                trial,
                previous,
                np.ones(uav_count, dtype=bool),
                maximum_displacement,
                region_values,
                minimum_separation,
                max(epsilon, _DEFAULT_TOLERANCE),
            ):
                continue
            trial_state = evaluate(trial)
            trial_move_cost = _uav_state_cost(
                previous,
                trial,
                environment=environment,
                movement_epsilon_m=epsilon,
            )
            trial_keep_score = predicted_cache_utility(
                old_cache,
                cache_prediction=trial_state["prediction"],
                node_types=kinds,
                environment=environment,
            ) - trial_move_cost
            trial_joint_score = (
                trial_state["exec_utility"]
                - trial_state["cache_cost"]
                - trial_move_cost
            )
            forced_trial_joint_score = (
                trial_state["candidate_utility"]
                - trial_state["candidate_cache_cost"]
                - trial_move_cost
            )
            if cache_control_action is None:
                trial_score = float(max(trial_keep_score, trial_joint_score))
            elif cache_control_action == 0:
                trial_score = float(trial_keep_score)
            else:
                trial_score = float(forced_trial_joint_score)
            if trial_score > best_score + _DEFAULT_TOLERANCE:
                best_score = trial_score
                best_positions = trial
                best_state = trial_state
        working = best_positions

    candidate_positions = working
    candidate_state = evaluate(candidate_positions)
    candidate_move_cost = _uav_state_cost(
        previous,
        candidate_positions,
        environment=environment,
        movement_epsilon_m=epsilon,
    )
    candidate_prediction = candidate_state["prediction"]
    J_uav_only = predicted_cache_utility(
        old_cache,
        cache_prediction=candidate_prediction,
        node_types=kinds,
        environment=environment,
    ) - candidate_move_cost
    reference_J_joint = float(
        candidate_state["exec_utility"]
        - candidate_state["cache_cost"]
        - candidate_move_cost
    )
    forced_J_joint = float(
        candidate_state["candidate_utility"]
        - candidate_state["candidate_cache_cost"]
        - candidate_move_cost
    )
    if cache_control_action is None:
        J_cache_only = cache_only_utility - cache_only_cost
        J_joint = reference_J_joint
    else:
        J_cache_only = forced_cache_only_utility
        J_joint = forced_J_joint
    values = np.asarray([J_keep, J_cache_only, J_uav_only, J_joint], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("UAV A/B/C/D 评价不能包含NaN/Inf。")
    if cache_control_action is None:
        selected_index = int(np.argmax(values))
    elif cache_control_action == 0:
        selected_index = 0 if J_keep >= J_uav_only else 2
    else:
        selected_index = 1 if J_cache_only >= J_joint else 3
    labels = ("A_keep", "B_cache_only", "C_uav_only", "D_joint")
    selected_action = labels[selected_index]
    if selected_index == 0:
        selected_positions = previous.copy()
        selected_state = base_state
        selected_cache = old_cache.copy()
    elif selected_index == 1:
        selected_positions = previous.copy()
        selected_state = base_state
        selected_cache = np.asarray(
            base_state["candidate_cache"] if cache_control_action == 1 else base_state["final_cache"],
            dtype=float,
        ).copy()
    elif selected_index == 2:
        selected_positions = candidate_positions.copy()
        selected_state = candidate_state
        selected_cache = old_cache.copy()
    else:
        selected_positions = candidate_positions.copy()
        selected_state = candidate_state
        selected_cache = np.asarray(
            candidate_state["candidate_cache"] if cache_control_action == 1 else candidate_state["final_cache"],
            dtype=float,
        ).copy()

    actual_movement = np.linalg.norm(selected_positions - previous, axis=1) > epsilon
    candidate_movement = np.linalg.norm(candidate_positions - previous, axis=1) > epsilon
    selected_adaptive = selected_state["adaptive"]
    if selected_index in {0, 2}:
        # A/C 明确保持缓存；在正常净收益规则下，若 B/D 候选真的值得执行，
        # 对应 J_B/J_D 会严格高于 A/C，因此这里不会丢失一个正收益缓存更新。
        selected_adaptive = dict(selected_adaptive)
        selected_adaptive["exec_cache"] = _readonly(selected_cache)
        selected_adaptive["cache_changed"] = _readonly(
            np.zeros(old_cache.shape[0], dtype=bool), dtype=bool
        )
        selected_adaptive["cache_trigger"] = _readonly(
            np.zeros(old_cache.shape[0], dtype=bool), dtype=bool
        )
    elif cache_control_action == 1:
        # PPO UPDATE 已经决定接受 C_cand；这里不能再用 G_cache 把动作改回 KEEP。
        # 候选相同的 slot 仍自然成为 no-op，实际成本由 old/final 差分得到0。
        selected_adaptive = dict(selected_adaptive)
        forced_changed = np.any(selected_cache != old_cache, axis=1)
        selected_adaptive["exec_cache"] = _readonly(selected_cache)
        selected_adaptive["cache_trigger"] = _readonly(forced_changed, dtype=bool)
        selected_adaptive["cache_changed"] = _readonly(forced_changed, dtype=bool)

    update_state = base_state if forced_cache_only_utility >= forced_J_joint else candidate_state
    update_candidate_cache = np.asarray(update_state["candidate_cache"], dtype=float)
    update_changed_units = int(np.sum(update_candidate_cache != old_cache))
    update_total_units = max(1, int(old_cache.size))
    selected_cache_cost = _cache_state_cost(
        old_cache,
        selected_cache,
        service_sizes=service_sizes,
        node_types=kinds,
        environment=environment,
    )
    selected_uav_cost = _uav_state_cost(
        previous,
        selected_positions,
        environment=environment,
        movement_epsilon_m=epsilon,
    )
    return {
        "candidate_cache": np.asarray(selected_state["candidate_cache"], dtype=float),
        "final_cache": selected_cache,
        "adaptive_cache": selected_adaptive,
        "final_positions": _readonly(selected_positions),
        "candidate_positions": _readonly(candidate_positions),
        "actual_movement": _readonly(actual_movement, dtype=bool),
        "candidate_movement": _readonly(candidate_movement, dtype=bool),
        "uav_check": True,
        "uav_selected_action": selected_action,
        "uav_J_keep": float(J_keep),
        "uav_J_cache_only": float(J_cache_only),
        "uav_J_uav_only": float(J_uav_only),
        "uav_J_joint": float(J_joint),
        "ppo_J_keep": float(max(J_keep, J_uav_only)),
        "ppo_J_update": float(max(forced_cache_only_utility, forced_J_joint)),
        "ppo_delta_predicted_utility": float(
            max(forced_cache_only_utility, forced_J_joint)
            - max(J_keep, J_uav_only)
        ),
        "ppo_candidate_cache_cost": float(update_state["candidate_cache_cost"]),
        "ppo_changed_service_ratio": float(update_changed_units / update_total_units),
        "ppo_candidate_cache": _readonly(update_candidate_cache),
        "uav_candidate_count": int(sum(len(items) for items in candidate_sets)),
        "uav_candidate_move_cost": float(candidate_move_cost),
        "uav_actual_move_cost": float(selected_uav_cost),
        "cache_actual_cost": float(selected_cache_cost),
        "ground_service_gap": float(gap),
        "cache_control_action": cache_control_action,
        "fallback_applied": False,
    }


# ---------------------------------------------------------------------------
# Feasible execution nodes
# ---------------------------------------------------------------------------


def build_feasible_nodes(
    task_arrivals: Sequence[bool | Real] | np.ndarray,
    task_service_types: Sequence[int] | np.ndarray,
    task_vehicle_cache: Sequence[Sequence[Real]] | np.ndarray,
    edge_cache: Sequence[Sequence[Real]] | np.ndarray,
    *,
    local_columns: Sequence[int],
    service_columns: Sequence[int],
    cloud_column: int,
) -> tuple[tuple[int, ...], ...]:
    """按最终实际缓存构造每个任务的 ``N_v^fea``。

    Local 只有在任务车辆自己的 cache 中包含请求服务时可行；三个 edge node
    只有在最终 edge cache 中包含该服务时可行；Cloud 对每个已到达任务都加入
    作为论文 fallback。未到达任务返回空集合，防止其被错误分配。
    """

    fixed = _binary_array(task_vehicle_cache, "task_vehicle_cache", ndim=2)
    edge = _binary_array(edge_cache, "edge_cache", ndim=2)
    arrivals = _bool_vector(task_arrivals, "task_arrivals", length=fixed.shape[0])
    services = np.asarray(task_service_types)
    if services.shape != (fixed.shape[0],):
        raise ValueError("task_service_types数量必须等于任务车辆数量。")
    if fixed.shape[1] != edge.shape[1]:
        raise ValueError("task_vehicle_cache和edge_cache的服务维度必须相同。")
    local, service, cloud, _ = _validate_layout(
        task_count=fixed.shape[0],
        service_count=edge.shape[0],
        local_columns=local_columns,
        service_columns=service_columns,
        cloud_column=cloud_column,
    )
    result: list[tuple[int, ...]] = []
    for vehicle in range(fixed.shape[0]):
        if not arrivals[vehicle]:
            result.append(())
            continue
        service_index = int(services[vehicle])
        if service_index < 0 or service_index >= fixed.shape[1]:
            raise ValueError(f"已到达任务{vehicle}的service type无效：{service_index}。")
        nodes: list[int] = []
        # 服务编号来自 workload；两个 cache 矩阵的列必须与同一服务目录对齐。
        if fixed[vehicle, service_index] > 0.5:
            nodes.append(local[vehicle])
        nodes.extend(
            service[node]
            for node in range(edge.shape[0])
            if edge[node, service_index] > 0.5
        )
        nodes.append(cloud)
        result.append(tuple(nodes))
    return tuple(result)


# ---------------------------------------------------------------------------
# KKT: C13--C16
# ---------------------------------------------------------------------------


def _assignment_node_count(
    assignment: Any,
    service_columns: Sequence[int],
    task_count: int,
) -> int:
    array = np.asarray(assignment)
    if array.ndim == 2:
        return int(array.shape[1])
    if array.ndim != 1 or array.shape != (task_count,):
        raise ValueError("assignment必须是一维向量或二维矩阵。")
    values = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("assignment不能包含NaN或Inf。")
    active = values[values >= 0.0]
    maximum = max(
        max((int(item) for item in service_columns), default=-1),
        int(np.max(active)) if active.size else -1,
    )
    return maximum + 1


def _assignment_vector(
    assignment: Sequence[int] | Sequence[Sequence[Real]] | np.ndarray,
    task_count: int,
    node_count: int,
    arrivals: np.ndarray,
) -> np.ndarray:
    array = np.asarray(assignment)
    if array.ndim == 1:
        if array.shape != (task_count,):
            raise ValueError("assignment向量长度必须等于任务数量。")
        vector = np.asarray(array, dtype=float)
        if not np.all(np.isfinite(vector)):
            raise ValueError("assignment不能包含NaN或Inf。")
        rounded = np.rint(vector)
        if not np.allclose(vector, rounded, rtol=0.0, atol=0.0):
            raise ValueError("assignment向量必须包含整数节点列。")
        result = rounded.astype(int)
    elif array.ndim == 2:
        matrix = _binary_array(array, "assignment", ndim=2)
        if matrix.shape != (task_count, node_count):
            raise ValueError("assignment矩阵形状与任务/节点数量不一致。")
        row_sums = matrix.sum(axis=1)
        if np.any(arrivals & (np.abs(row_sums - 1.0) > _DEFAULT_TOLERANCE)):
            raise ValueError("已到达任务必须恰好分配一个执行节点。")
        if np.any((~arrivals) & (row_sums > _DEFAULT_TOLERANCE)):
            raise ValueError("未到达任务不能分配执行节点。")
        result = np.full(task_count, -1, dtype=int)
        for index in np.flatnonzero(arrivals):
            result[index] = int(np.flatnonzero(matrix[index] > 0.5)[0])
    else:
        raise ValueError("assignment必须是一维节点向量或二维0/1矩阵。")
    if np.any(arrivals & ((result < 0) | (result >= node_count))):
        raise ValueError("已到达任务的assignment节点列越界。")
    if np.any((~arrivals) & (result != -1)):
        raise ValueError("未到达任务的assignment必须为-1。")
    return result


def _kkt_node_solution(
    coefficients: np.ndarray,
    epsilon: float,
    *,
    tolerance: float,
    multiplier_tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, float | None, bool, str]:
    """求解一个共享节点的截断平方根 KKT 子问题。

    对系数 ``a_i``，候选资源比例遵循 ``w_i = clip(sqrt(a_i/lambda),
    epsilon, 1)``，再用二分搜索 ``lambda`` 使同一节点上的资源比例和为1。
    该函数只解决单个节点，不负责任务到节点的 matching。
    """

    count = coefficients.size
    if count == 0:
        return np.zeros(0), None, True, ""
    if epsilon * count > 1.0 + tolerance:
        return np.zeros(count), None, False, "资源下界导致该服务节点不可行。"
    lower = float(np.min(coefficients))
    upper = float(np.max(coefficients) / (epsilon**2))
    if not math.isfinite(lower) or not math.isfinite(upper) or upper < lower:
        return np.zeros(count), None, False, "KKT二分乘子区间无效。"

    def clipped(multiplier: float) -> tuple[np.ndarray, float]:
        # clip 同时实现论文的 epsilon 下界和单任务最多占满全部资源的上界。
        values = np.clip(np.sqrt(coefficients / multiplier), epsilon, 1.0)
        return values, float(values.sum())

    boundary_values, boundary_total = clipped(lower)
    if abs(boundary_total - 1.0) <= tolerance:
        return boundary_values, lower, True, ""

    for _ in range(max_iterations):
        multiplier = (lower + upper) / 2.0
        values, total = clipped(multiplier)
        if not math.isfinite(total):
            return np.zeros(count), None, False, "KKT资源总和不是有限值。"
        if abs(total - 1.0) <= tolerance:
            return values, multiplier, True, ""
        if total > 1.0:
            lower = multiplier
        else:
            upper = multiplier
        if upper - lower <= multiplier_tolerance:
            break
    multiplier = (lower + upper) / 2.0
    values, total = clipped(multiplier)
    if abs(total - 1.0) > tolerance:
        return np.zeros(count), None, False, "KKT二分未达到资源总和为1的收敛条件。"
    return values, multiplier, True, ""


def solve_compute_allocation(
    assignment: Sequence[int] | Sequence[Sequence[Real]] | np.ndarray,
    task_arrivals: Sequence[bool | Real] | np.ndarray,
    cpu_cycles: Sequence[Real] | np.ndarray,
    reference_delays_s: Sequence[Real] | Real,
    node_frequencies_hz: Sequence[Real] | Real,
    *,
    service_columns: Sequence[int],
    omega_time: Real,
    epsilon_compute: Real,
    tolerance: float = _DEFAULT_TOLERANCE,
    multiplier_tolerance: float = _DEFAULT_MULTIPLIER_TOLERANCE,
    max_iterations: int = 100,
) -> dict[str, Any]:
    """按论文 A 系数和截断平方根公式求解计算资源比例。

    计算资源只在被 assignment 到某个 edge node 的任务之间共享；Local/Cloud
    不进入该矩阵。返回的 allocations 形状为 ``(V,M)``，M 是 edge service node
    数量而不是总执行列数量。
    """

    arrivals = _bool_vector(task_arrivals, "task_arrivals")
    task_count = arrivals.size
    service = _tuple_ints(service_columns, "service_columns")
    node_count = _assignment_node_count(assignment, service, task_count)
    vector = _assignment_vector(assignment, task_count, node_count, arrivals)
    cycles = _vector(cpu_cycles, "cpu_cycles", task_count)
    references = _vector(reference_delays_s, "reference_delays_s", task_count, positive=True)
    frequencies = _vector(
        node_frequencies_hz,
        "node_frequencies_hz",
        len(service),
        positive=True,
    )
    omega = float(omega_time)
    epsilon = float(epsilon_compute)
    if not math.isfinite(omega) or omega < 0.0:
        raise ValueError("omega_time必须为有限非负数。")
    if not math.isfinite(epsilon) or epsilon <= 0.0 or epsilon > 1.0:
        raise ValueError("epsilon_compute必须位于(0,1]。")
    if max_iterations <= 0 or multiplier_tolerance <= 0.0:
        raise ValueError("KKT数值控制参数必须为正数。")

    allocations = np.zeros((task_count, len(service)), dtype=float)
    coefficients = np.zeros_like(allocations)
    multipliers: list[float | None] = [None] * len(service)
    errors: list[str] = []
    for local_node, full_node in enumerate(service):
        # full_node 是全局执行列，local_node 是资源矩阵中的局部 edge 列。
        assigned = np.flatnonzero(arrivals & (vector == full_node))
        if assigned.size == 0:
            continue
        coefficient = omega * cycles[assigned] / (references[assigned] * frequencies[local_node])
        if np.any(coefficient <= 0.0) or not np.all(np.isfinite(coefficient)):
            errors.append(f"服务节点{full_node}的A系数必须为有限正数。")
            continue
        values, multiplier, feasible, reason = _kkt_node_solution(
            coefficient,
            epsilon,
            tolerance=tolerance,
            multiplier_tolerance=float(multiplier_tolerance),
            max_iterations=int(max_iterations),
        )
        coefficients[assigned, local_node] = coefficient
        if not feasible:
            errors.append(f"服务节点{full_node}：{reason}")
            continue
        allocations[assigned, local_node] = values
        multipliers[local_node] = multiplier

    coupling_ok = True
    for task in np.flatnonzero(arrivals):
        assigned = vector[task]
        if assigned in service:
            local_node = service.index(int(assigned))
            value = allocations[task, local_node]
            coupling_ok &= bool(epsilon - tolerance <= value <= 1.0 + tolerance)
        elif np.any(allocations[task] > tolerance):
            coupling_ok = False
    capacity_ok = bool(np.all(allocations.sum(axis=0) <= 1.0 + tolerance))
    feasible = not errors and coupling_ok and capacity_ok
    return {
        "allocations": _readonly(allocations),
        "coefficients": _readonly(coefficients),
        "multipliers": tuple(multipliers),
        "feasible": bool(feasible),
        "reason": "计算KKT通过C13-C14" if feasible else "；".join(errors) or "计算KKT约束不可行。",
    }


def solve_bandwidth_allocation(
    assignment: Sequence[int] | Sequence[Sequence[Real]] | np.ndarray,
    task_arrivals: Sequence[bool | Real] | np.ndarray,
    input_bits: Sequence[Real] | np.ndarray,
    reference_delays_s: Sequence[Real] | Real,
    reference_energies_j: Sequence[Real] | Real,
    transmit_power_w: Sequence[Real] | Real,
    base_rates_bps: Sequence[Sequence[Real]] | np.ndarray,
    *,
    service_columns: Sequence[int],
    omega_time: Real,
    omega_energy: Real,
    epsilon_bandwidth: Real,
    tolerance: float = _DEFAULT_TOLERANCE,
    multiplier_tolerance: float = _DEFAULT_MULTIPLIER_TOLERANCE,
    max_iterations: int = 100,
) -> dict[str, Any]:
    """按论文 Gamma 系数和截断平方根公式求解共享带宽比例。

    base_rates_bps[v,m] 必须是 environment.py 用完整带宽比例1计算出的实际
    基准速率；solver 不在这里复制 Shannon 速率公式。Gamma 同时包含时延权重、
    能耗权重、任务输入量、发射功率和该候选链路的基准速率。
    """

    arrivals = _bool_vector(task_arrivals, "task_arrivals")
    task_count = arrivals.size
    service = _tuple_ints(service_columns, "service_columns")
    node_count = _assignment_node_count(assignment, service, task_count)
    vector = _assignment_vector(assignment, task_count, node_count, arrivals)
    data = _vector(input_bits, "input_bits", task_count)
    delay_refs = _vector(reference_delays_s, "reference_delays_s", task_count, positive=True)
    energy_refs = _vector(reference_energies_j, "reference_energies_j", task_count, positive=True)
    powers = _vector(transmit_power_w, "transmit_power_w", task_count, positive=True)
    base_rates = _nonnegative_array(base_rates_bps, "base_rates_bps", ndim=2)
    if base_rates.shape != (task_count, len(service)):
        raise ValueError("base_rates_bps形状必须为(任务数,服务节点数)。")
    time_weight = float(omega_time)
    energy_weight = float(omega_energy)
    epsilon = float(epsilon_bandwidth)
    if not math.isfinite(time_weight) or not math.isfinite(energy_weight):
        raise ValueError("omega_time和omega_energy必须有限。")
    if time_weight < 0.0 or energy_weight < 0.0 or not math.isclose(
        time_weight + energy_weight, 1.0, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("omega_time和omega_energy必须非负且和为1。")
    if not math.isfinite(epsilon) or epsilon <= 0.0 or epsilon > 1.0:
        raise ValueError("epsilon_bandwidth必须位于(0,1]。")
    if max_iterations <= 0 or multiplier_tolerance <= 0.0:
        raise ValueError("KKT数值控制参数必须为正数。")

    allocations = np.zeros((task_count, len(service)), dtype=float)
    coefficients = np.zeros_like(allocations)
    multipliers: list[float | None] = [None] * len(service)
    errors: list[str] = []
    for local_node, full_node in enumerate(service):
        # 每个 edge node 独立求一个共享带宽 KKT 子问题，避免把不同节点的带宽
        # 容量错误地放进同一个全局约束。
        assigned = np.flatnonzero(arrivals & (vector == full_node))
        if assigned.size == 0:
            continue
        if np.any(data[assigned] <= 0.0):
            errors.append(f"服务节点{full_node}上的input_bits必须为正数。")
            continue
        if np.any(base_rates[assigned, local_node] <= 0.0):
            errors.append(f"服务节点{full_node}上的实际基准速率必须为正数。")
            continue
        gamma = data[assigned] / base_rates[assigned, local_node] * (
            time_weight / delay_refs[assigned]
            + energy_weight * powers[assigned] / energy_refs[assigned]
        )
        if np.any(gamma <= 0.0) or not np.all(np.isfinite(gamma)):
            errors.append(f"服务节点{full_node}的Gamma系数必须为有限正数。")
            continue
        values, multiplier, feasible, reason = _kkt_node_solution(
            gamma,
            epsilon,
            tolerance=tolerance,
            multiplier_tolerance=float(multiplier_tolerance),
            max_iterations=int(max_iterations),
        )
        coefficients[assigned, local_node] = gamma
        if not feasible:
            errors.append(f"服务节点{full_node}：{reason}")
            continue
        allocations[assigned, local_node] = values
        multipliers[local_node] = multiplier

    coupling_ok = True
    for task in np.flatnonzero(arrivals):
        assigned = vector[task]
        if assigned in service:
            local_node = service.index(int(assigned))
            value = allocations[task, local_node]
            coupling_ok &= bool(epsilon - tolerance <= value <= 1.0 + tolerance)
        elif np.any(allocations[task] > tolerance):
            coupling_ok = False
    capacity_ok = bool(np.all(allocations.sum(axis=0) <= 1.0 + tolerance))
    feasible = not errors and coupling_ok and capacity_ok
    return {
        "allocations": _readonly(allocations),
        "coefficients": _readonly(coefficients),
        "multipliers": tuple(multipliers),
        "feasible": bool(feasible),
        "reason": "带宽KKT通过C15-C16" if feasible else "；".join(errors) or "带宽KKT约束不可行。",
    }


def solve_resources(
    assignment: Sequence[int] | Sequence[Sequence[Real]] | np.ndarray,
    task_arrivals: Sequence[bool | Real] | np.ndarray,
    *,
    cpu_cycles: Sequence[Real] | np.ndarray,
    input_bits: Sequence[Real] | np.ndarray,
    reference_delays_s: Sequence[Real] | Real,
    reference_energies_j: Sequence[Real] | Real,
    node_frequencies_hz: Sequence[Real] | Real,
    transmit_power_w: Sequence[Real] | Real,
    base_rates_bps: Sequence[Sequence[Real]] | np.ndarray,
    service_columns: Sequence[int],
    omega_time: Real,
    omega_energy: Real,
    epsilon_compute: Real,
    epsilon_bandwidth: Real,
    tolerance: float = _DEFAULT_TOLERANCE,
    multiplier_tolerance: float = _DEFAULT_MULTIPLIER_TOLERANCE,
    max_iterations: int = 100,
) -> dict[str, Any]:
    """联合执行计算 KKT 和带宽 KKT。

    两个子问题共享同一 assignment，但使用各自的论文系数和 epsilon。只有两
    个子问题都可行时才返回 ``feasible=True``；该结果随后才允许 environment
    对任务调用实际执行公式。
    """

    compute = solve_compute_allocation(
        assignment,
        task_arrivals,
        cpu_cycles,
        reference_delays_s,
        node_frequencies_hz,
        service_columns=service_columns,
        omega_time=omega_time,
        epsilon_compute=epsilon_compute,
        tolerance=tolerance,
        multiplier_tolerance=multiplier_tolerance,
        max_iterations=max_iterations,
    )
    bandwidth = solve_bandwidth_allocation(
        assignment,
        task_arrivals,
        input_bits,
        reference_delays_s,
        reference_energies_j,
        transmit_power_w,
        base_rates_bps,
        service_columns=service_columns,
        omega_time=omega_time,
        omega_energy=omega_energy,
        epsilon_bandwidth=epsilon_bandwidth,
        tolerance=tolerance,
        multiplier_tolerance=multiplier_tolerance,
        max_iterations=max_iterations,
    )
    return {
        "compute_allocations": compute["allocations"],
        "bandwidth_allocations": bandwidth["allocations"],
        "compute_coefficients": compute["coefficients"],
        "bandwidth_coefficients": bandwidth["coefficients"],
        "feasible": bool(compute["feasible"] and bandwidth["feasible"]),
        "reason": (
            "C13-C16资源分配通过"
            if compute["feasible"] and bandwidth["feasible"]
            else f"{compute['reason']}；{bandwidth['reason']}"
        ),
    }


# ---------------------------------------------------------------------------
# Matching: 每个候选都重新做 KKT 和 environment 评价
# ---------------------------------------------------------------------------


def _assignment_matrix(vector: np.ndarray, arrivals: np.ndarray, node_count: int) -> np.ndarray:
    matrix = np.zeros((vector.size, node_count), dtype=float)
    for task in np.flatnonzero(arrivals):
        matrix[task, vector[task]] = 1.0
    return matrix


def _evaluate_assignment(
    vector: np.ndarray,
    *,
    feasible_nodes: tuple[tuple[int, ...], ...],
    environment: MECEnvironment,
    task_arrivals: np.ndarray,
    task_input_bits: np.ndarray,
    task_cpu_cycles: np.ndarray,
    task_vehicle_positions: np.ndarray,
    service_node_positions: np.ndarray,
    service_node_types: tuple[NodeType, ...],
    local_columns: tuple[int, ...],
    service_columns: tuple[int, ...],
    cloud_column: int,
    base_rates_bps: np.ndarray,
    node_frequencies_hz: np.ndarray,
    transmit_power_w: np.ndarray,
    reference_delays_s: np.ndarray,
    reference_energies_j: np.ndarray,
    omega_time: float,
    omega_energy: float,
    epsilon_compute: float,
    epsilon_bandwidth: float,
    tolerance: float,
    multiplier_tolerance: float,
    max_kkt_iterations: int,
) -> dict[str, Any]:
    """评价一个完整 assignment：可行性、KKT、物理结果和总任务效用。"""

    for task in np.flatnonzero(task_arrivals):
        if int(vector[task]) not in feasible_nodes[task]:
            return {"feasible": False, "reason": f"任务{task}的候选节点不在N_v^fea中。"}
    # 每个 matching 候选都重新求 KKT，不能复用上一个 assignment 的资源比例，
    # 否则交换任务后共享节点的资源约束和 utility 会失真。
    resources = solve_resources(
        vector,
        task_arrivals,
        cpu_cycles=task_cpu_cycles,
        input_bits=task_input_bits,
        reference_delays_s=reference_delays_s,
        reference_energies_j=reference_energies_j,
        node_frequencies_hz=node_frequencies_hz,
        transmit_power_w=transmit_power_w,
        base_rates_bps=base_rates_bps,
        service_columns=service_columns,
        omega_time=omega_time,
        omega_energy=omega_energy,
        epsilon_compute=epsilon_compute,
        epsilon_bandwidth=epsilon_bandwidth,
        tolerance=tolerance,
        multiplier_tolerance=multiplier_tolerance,
        max_iterations=max_kkt_iterations,
    )
    if not resources["feasible"]:
        return {"feasible": False, "reason": resources["reason"]}

    delays = np.zeros(task_arrivals.size, dtype=float)
    energies = np.zeros(task_arrivals.size, dtype=float)
    task_results: list[Mapping[str, Any]] = []
    execution_nodes: list[str | None] = [None] * task_arrivals.size
    for task in range(task_arrivals.size):
        if not task_arrivals[task]:
            task_results.append(
                {"execution_kind": None, "completion_delay_s": 0.0, "vehicle_energy_j": 0.0}
            )
            continue
        column = int(vector[task])
        try:
            if column in local_columns:
                # Local 不需要 edge resource matrix，也不需要无线速率。
                kind = NodeType.LOCAL
                result = environment.execute_task(
                    kind,
                    arrived=True,
                    input_bits=float(task_input_bits[task]),
                    cpu_cycles=float(task_cpu_cycles[task]),
                )
            elif column in service_columns:
                # service_columns 是全局列号，local_node 是 KKT 矩阵列号。
                local_node = service_columns.index(column)
                kind = service_node_types[local_node]
                result = environment.execute_task(
                    kind,
                    arrived=True,
                    input_bits=float(task_input_bits[task]),
                    cpu_cycles=float(task_cpu_cycles[task]),
                    compute_resource_fraction=float(resources["compute_allocations"][task, local_node]),
                    bandwidth_fraction=float(resources["bandwidth_allocations"][task, local_node]),
                    vehicle_position=task_vehicle_positions[task],
                    node_position=(
                        service_node_positions[local_node]
                        if kind in {NodeType.SERVICE_VEHICLE, NodeType.RSU}
                        else None
                    ),
                    uav_position=(
                        service_node_positions[local_node]
                        if kind is NodeType.UAV
                        else None
                    ),
                )
            elif column == cloud_column:
                # Cloud fallback 的固定速率/CPU由 environment 统一解释。
                kind = NodeType.CLOUD
                result = environment.execute_task(
                    kind,
                    arrived=True,
                    input_bits=float(task_input_bits[task]),
                    cpu_cycles=float(task_cpu_cycles[task]),
                )
            else:
                return {"feasible": False, "reason": f"任务{task}的执行节点列无效。"}
        except (ValueError, FloatingPointError) as error:
            return {"feasible": False, "reason": f"任务{task}的environment物理评价失败：{error}"}
        delays[task] = float(result["completion_delay_s"])
        energies[task] = float(result["vehicle_energy_j"])
        execution_nodes[task] = kind.value
        task_results.append(result)

    utility = environment.task_utility(task_arrivals, delays, energies)
    return {
        "feasible": True,
        "reason": "匹配候选通过KKT和environment物理评价。",
        "assignment_vector": vector.copy(),
        "assignment": _assignment_matrix(
            vector,
            task_arrivals,
            max((*local_columns, *service_columns, cloud_column)) + 1,
        ),
        "compute_allocations": resources["compute_allocations"],
        "bandwidth_allocations": resources["bandwidth_allocations"],
        "task_results": tuple(task_results),
        "execution_nodes": tuple(execution_nodes),
        "task_utility": utility,
        "total_task_utility": float(utility["total_task_utility"]),
    }


def transfer_exchange_matching(
    feasible_nodes: tuple[tuple[int, ...], ...],
    *,
    environment: MECEnvironment,
    task_arrivals: Sequence[bool | Real] | np.ndarray,
    task_input_bits: Sequence[Real] | np.ndarray,
    task_cpu_cycles: Sequence[Real] | np.ndarray,
    task_vehicle_positions: Sequence[Sequence[Real]] | np.ndarray,
    service_node_positions: Sequence[Sequence[Real]] | np.ndarray,
    service_node_types: Sequence[NodeType | str],
    local_columns: Sequence[int],
    service_columns: Sequence[int],
    cloud_column: int,
    base_rates_bps: Sequence[Sequence[Real]] | np.ndarray,
    node_frequencies_hz: Sequence[Real] | Real,
    transmit_power_w: Sequence[Real] | Real,
    reference_delays_s: Sequence[Real] | Real,
    reference_energies_j: Sequence[Real] | Real,
    omega_time: Real,
    omega_energy: Real,
    epsilon_compute: Real,
    epsilon_bandwidth: Real,
    improvement_tolerance: float = _DEFAULT_TOLERANCE,
    max_matching_updates: int | None = None,
    multiplier_tolerance: float = _DEFAULT_MULTIPLIER_TOLERANCE,
    max_kkt_iterations: int = 100,
) -> dict[str, Any]:
    """执行论文定义的 Cloud 初始化、transfer、exchange 局部改进。

    初始状态把所有到达任务放到 Cloud，保证 assignment 有完整起点；随后每轮
    枚举“单任务迁移”和“两个任务交换”，每个候选都经过完整 KKT 与 environment
    物理评价。只接受超过 ``improvement_tolerance`` 的最大 utility 改进，同分时
    用确定性的操作类型/任务编号/节点编号字典序打破平局。
    """

    arrivals = _bool_vector(task_arrivals, "task_arrivals")
    task_count = arrivals.size
    input_bits = _nonnegative_array(task_input_bits, "task_input_bits", ndim=1)
    cpu_cycles = _nonnegative_array(task_cpu_cycles, "task_cpu_cycles", ndim=1)
    if input_bits.shape != (task_count,) or cpu_cycles.shape != (task_count,):
        raise ValueError("任务负载向量长度必须等于任务数量。")
    vehicle_positions = _finite_array(task_vehicle_positions, "task_vehicle_positions", ndim=2)
    if vehicle_positions.shape != (task_count, 2):
        raise ValueError("task_vehicle_positions必须形状为(V,2)。")
    edge_positions = _finite_array(service_node_positions, "service_node_positions", ndim=2)
    service = _tuple_ints(service_columns, "service_columns")
    local, service, cloud, _ = _validate_layout(
        task_count=task_count,
        service_count=len(service),
        local_columns=local_columns,
        service_columns=service,
        cloud_column=cloud_column,
    )
    if edge_positions.shape != (len(service), 2):
        raise ValueError("service_node_positions必须形状为(M,2)。")
    kinds = tuple(_node_kind(item) for item in service_node_types)
    if len(kinds) != len(service) or any(
        kind not in {NodeType.SERVICE_VEHICLE, NodeType.RSU, NodeType.UAV} for kind in kinds
    ):
        raise ValueError("service_node_types必须只包含三类边缘服务节点。")
    if len(feasible_nodes) != task_count:
        raise ValueError("feasible_nodes数量必须等于任务数量。")
    base_rates = _nonnegative_array(base_rates_bps, "base_rates_bps", ndim=2)
    if base_rates.shape != (task_count, len(service)):
        raise ValueError("base_rates_bps形状必须为(V,M)。")
    node_frequencies = _vector(
        node_frequencies_hz,
        "node_frequencies_hz",
        len(service),
        positive=True,
    )
    powers = _vector(transmit_power_w, "transmit_power_w", task_count, positive=True)
    delay_refs = _vector(reference_delays_s, "reference_delays_s", task_count, positive=True)
    energy_refs = _vector(reference_energies_j, "reference_energies_j", task_count, positive=True)
    if max_matching_updates is not None and max_matching_updates < 0:
        raise ValueError("max_matching_updates不能为负数。")

    # Cloud 是论文 fallback，先保证每个已到达任务都有一个合法初始节点。
    current = np.full(task_count, -1, dtype=int)
    for task in np.flatnonzero(arrivals):
        if cloud not in feasible_nodes[task]:
            raise ValueError("Cloud必须出现在每个已到达任务的可行节点集合中。")
        current[task] = cloud

    def evaluate(candidate: np.ndarray) -> dict[str, Any]:
        return _evaluate_assignment(
            candidate,
            feasible_nodes=feasible_nodes,
            environment=environment,
            task_arrivals=arrivals,
            task_input_bits=input_bits,
            task_cpu_cycles=cpu_cycles,
            task_vehicle_positions=vehicle_positions,
            service_node_positions=edge_positions,
            service_node_types=kinds,
            local_columns=local,
            service_columns=service,
            cloud_column=cloud,
            base_rates_bps=base_rates,
            node_frequencies_hz=node_frequencies,
            transmit_power_w=powers,
            reference_delays_s=delay_refs,
            reference_energies_j=energy_refs,
            omega_time=float(omega_time),
            omega_energy=float(omega_energy),
            epsilon_compute=float(epsilon_compute),
            epsilon_bandwidth=float(epsilon_bandwidth),
            tolerance=_DEFAULT_TOLERANCE,
            multiplier_tolerance=float(multiplier_tolerance),
            max_kkt_iterations=int(max_kkt_iterations),
        )

    current_evaluation = evaluate(current)
    if not current_evaluation["feasible"]:
        raise ValueError(f"Cloud初始匹配不可行：{current_evaluation['reason']}")
    updates = 0
    migration_count = 0
    swap_count = 0
    stable = True

    # 采用 transfer-exchange 的局部改进过程；每轮最多提交一个最佳候选，
    # 提交后重新计算所有候选，避免使用过期的资源分配结果。
    while True:
        if max_matching_updates is not None and updates >= max_matching_updates:
            stable = False
            break
        best: tuple[tuple[Any, ...], np.ndarray, dict[str, Any], float] | None = None
        current_value = float(current_evaluation["total_task_utility"])

        # transfer：尝试把一个任务迁移到其余可行节点。
        for task in np.flatnonzero(arrivals):
            for node in feasible_nodes[task]:
                if node == current[task]:
                    continue
                candidate = current.copy()
                candidate[task] = node
                evaluation = evaluate(candidate)
                if not evaluation["feasible"]:
                    continue
                gain = float(evaluation["total_task_utility"]) - current_value
                if gain <= improvement_tolerance:
                    continue
                key = (0, int(task), int(node))
                if best is None or gain > best[3] + improvement_tolerance or (
                    abs(gain - best[3]) <= improvement_tolerance and key < best[0]
                ):
                    best = (key, candidate, evaluation, gain)

        # exchange：尝试交换两个任务的执行节点，只有交换后双方仍在对方的
        # 可行集合内且整体 utility 提升时才保留。
        active = tuple(int(item) for item in np.flatnonzero(arrivals))
        for offset, first in enumerate(active):
            for second in active[offset + 1 :]:
                if (
                    current[second] not in feasible_nodes[first]
                    or current[first] not in feasible_nodes[second]
                ):
                    continue
                if current[first] == current[second]:
                    continue
                candidate = current.copy()
                candidate[first], candidate[second] = current[second], current[first]
                evaluation = evaluate(candidate)
                if not evaluation["feasible"]:
                    continue
                gain = float(evaluation["total_task_utility"]) - current_value
                if gain <= improvement_tolerance:
                    continue
                key = (1, int(first), int(second))
                if best is None or gain > best[3] + improvement_tolerance or (
                    abs(gain - best[3]) <= improvement_tolerance and key < best[0]
                ):
                    best = (key, candidate, evaluation, gain)

        if best is None:
            # 没有正改进候选，当前 assignment 达到本地 matching 稳定点。
            break
        operation_key, current, current_evaluation, _ = best
        if int(operation_key[0]) == 0:
            migration_count += 1
        else:
            swap_count += 1
        updates += 1

    return {
        "assignment_vector": current,
        "assignment": current_evaluation["assignment"],
        "compute_allocations": current_evaluation["compute_allocations"],
        "bandwidth_allocations": current_evaluation["bandwidth_allocations"],
        "task_results": current_evaluation["task_results"],
        "execution_nodes": current_evaluation["execution_nodes"],
        "task_utility": current_evaluation["task_utility"],
        "total_task_utility": current_evaluation["total_task_utility"],
        "matching_stable": bool(stable),
        "updates": int(updates),
        "migration_count": int(migration_count),
        "swap_count": int(swap_count),
        # 当前 matching 的初始 Cloud 解和最终解都经过 KKT；若最终解可行，
        # 则不存在需要报告的最终 KKT 失败。候选失败会被直接拒绝，不写入执行结果。
        "kkt_failure_count": 0,
    }


# ---------------------------------------------------------------------------
# Main solve: 固定配置后再进入任务 matching/KKT
# ---------------------------------------------------------------------------


def _config_vector_or_default(
    value: Sequence[Real] | Real | None,
    length: int,
    default: Real,
    name: str,
    *,
    positive: bool = True,
) -> np.ndarray:
    return _vector(default if value is None else value, name, length, positive=positive)


def solve_slot(
    *,
    previous_cache: Sequence[Sequence[Real]] | np.ndarray,
    cache_triggers: Sequence[bool | Real] | np.ndarray,
    cache_scores: Sequence[Sequence[Real]] | np.ndarray,
    service_sizes: Sequence[Real] | np.ndarray,
    cache_capacities: Real | Sequence[Real] | np.ndarray,
    previous_uav_positions: Sequence[Sequence[Real]] | np.ndarray,
    uav_triggers: Sequence[bool | Real] | np.ndarray,
    raw_uav_displacements: Sequence[Sequence[Real]] | np.ndarray,
    task_arrivals: Sequence[bool | Real] | np.ndarray,
    task_service_types: Sequence[int] | np.ndarray,
    task_vehicle_cache: Sequence[Sequence[Real]] | np.ndarray,
    task_input_bits: Sequence[Real] | np.ndarray,
    task_cpu_cycles: Sequence[Real] | np.ndarray,
    task_vehicle_positions: Sequence[Sequence[Real]] | np.ndarray,
    service_node_positions: Sequence[Sequence[Real]] | np.ndarray,
    service_node_types: Sequence[NodeType | str],
    uav_service_indices: Sequence[int],
    cache_prediction: Mapping[str, Any] | None = None,
    local_columns: Sequence[int] | None = None,
    service_columns: Sequence[int] | None = None,
    cloud_column: int | None = None,
    environment: MECEnvironment | None = None,
    integer_scale: int | None = None,
    uav_max_speed_mps: Real | None = None,
    decision_interval_s: Real | None = None,
    uav_region: Mapping[str, Real] | Sequence[Real] | None = None,
    minimum_safety_distance_m: Real | None = None,
    position_epsilon_m: Real = _DEFAULT_POSITION_EPSILON,
    epsilon_compute: Real,
    epsilon_bandwidth: Real,
    reference_delays_s: Sequence[Real] | Real | None = None,
    reference_energies_j: Sequence[Real] | Real | None = None,
    omega_time: Real | None = None,
    omega_energy: Real | None = None,
    node_frequencies_hz: Sequence[Real] | Real | None = None,
    transmit_power_w: Sequence[Real] | Real | None = None,
    multiplier_tolerance: float = _DEFAULT_MULTIPLIER_TOLERANCE,
    max_kkt_iterations: int = 100,
    max_matching_updates: int | None = None,
    cache_control_action: int | None = None,
) -> SolverResult:
    """按论文顺序完成一个时隙的配置、matching、KKT 和最终执行评价。

    epsilon_compute 和 epsilon_bandwidth 是论文中的显式下界参数；当前配置
    文件没有替它们指定数值，因此主入口要求调用方显式提供。输入中的
    ``service_node_positions`` 只包含 edge node，``service_node_types`` 与之同
    序；Local 和 Cloud 的全局列由 layout 参数补足。

    返回结果是 environment 下一状态所需的唯一事实来源：缓存/UAV 使用最终可行
    结果，assignment 使用 matching 结果，资源比例使用对应 assignment 的 KKT
    结果，reward 使用实际 task_results 和实际成本。
    """

    env = environment if environment is not None else MECEnvironment()
    previous = _binary_array(previous_cache, "previous_cache", ndim=2)
    task_cache = _binary_array(task_vehicle_cache, "task_vehicle_cache", ndim=2)
    arrivals = _bool_vector(task_arrivals, "task_arrivals", length=task_cache.shape[0])
    service_count, service_number = previous.shape
    task_count = task_cache.shape[0]
    if task_cache.shape[1] != service_number:
        raise ValueError("task_vehicle_cache服务维度必须等于previous_cache服务维度。")
    if np.asarray(service_sizes).shape != (service_number,):
        raise ValueError("service_sizes长度必须等于服务数量。")
    edge_positions = _finite_array(service_node_positions, "service_node_positions", ndim=2)
    if edge_positions.shape != (service_count, 2):
        raise ValueError("service_node_positions必须形状为(M,2)。")
    vehicle_positions = _finite_array(task_vehicle_positions, "task_vehicle_positions", ndim=2)
    if vehicle_positions.shape != (task_count, 2):
        raise ValueError("task_vehicle_positions必须形状为(V,2)。")
    kinds = tuple(_node_kind(item) for item in service_node_types)
    if len(kinds) != service_count or any(
        kind not in {NodeType.SERVICE_VEHICLE, NodeType.RSU, NodeType.UAV} for kind in kinds
    ):
        raise ValueError("service_node_types必须与edge节点一一对应且只包含SV、RSU、UAV。")
    uav_indices = _tuple_ints(uav_service_indices, "uav_service_indices")
    previous_uav = _finite_array(previous_uav_positions, "previous_uav_positions", ndim=2)
    if len(uav_indices) != previous_uav.shape[0]:
        raise ValueError("uav_service_indices数量必须等于previous_uav_positions行数。")
    if any(index >= service_count or kinds[index] is not NodeType.UAV for index in uav_indices):
        raise ValueError("uav_service_indices必须全部指向UAV服务节点。")

    local = tuple(range(task_count)) if local_columns is None else _tuple_ints(local_columns, "local_columns")
    service = (
        tuple(range(task_count, task_count + service_count))
        if service_columns is None
        else _tuple_ints(service_columns, "service_columns")
    )
    cloud = task_count + service_count if cloud_column is None else int(cloud_column)
    local, service, cloud, _ = _validate_layout(
        task_count=task_count,
        service_count=service_count,
        local_columns=local,
        service_columns=service,
        cloud_column=cloud,
    )
    task_types = np.asarray(task_service_types)
    if task_types.shape != (task_count,):
        raise ValueError("task_service_types数量必须等于任务车辆数量。")
    input_bits = _nonnegative_array(task_input_bits, "task_input_bits", ndim=1)
    cycles = _nonnegative_array(task_cpu_cycles, "task_cpu_cycles", ndim=1)
    if input_bits.shape != (task_count,) or cycles.shape != (task_count,):
        raise ValueError("任务输入和计算量向量长度必须等于任务数量。")

    config = env.config
    uav_config = config["uav"]
    simulation_config = config["simulation"]
    # 第六阶段：根据未来车辆需求和地面服务缺口生成少量候选位置，
    # 再用同一个 predicted_cache_utility 一次比较 A/B/C/D。raw displacement
    # 仍保留在 solver 接口中以兼容策略输出，但不再驱动 UAV 的正式位置决策。
    del uav_triggers, raw_uav_displacements
    joint_state = select_demand_aware_uav_cache_state(
        previous_uav,
        previous_cache=previous,
        requested_cache_triggers=cache_triggers,
        cache_scores=cache_scores,
        service_sizes=service_sizes,
        cache_capacities=cache_capacities,
        cache_prediction=cache_prediction,
        node_types=kinds,
        uav_service_indices=uav_indices,
        environment=env,
        max_speed_mps=(
            uav_max_speed_mps
            if uav_max_speed_mps is not None
            else uav_config["max_speed_mps"]
        ),
        slot_duration_s=(
            decision_interval_s
            if decision_interval_s is not None
            else simulation_config["decision_interval_s"]
        ),
        region=uav_region if uav_region is not None else uav_config["region"],
        minimum_separation_m=(
            minimum_safety_distance_m
            if minimum_safety_distance_m is not None
            else uav_config["minimum_safety_distance_m"]
        ),
        position_epsilon_m=position_epsilon_m,
        integer_scale=integer_scale,
        cache_control_action=cache_control_action,
    )
    candidate_cache = np.asarray(joint_state["candidate_cache"], dtype=float)
    adaptive_cache = dict(joint_state["adaptive_cache"])
    final_cache = np.asarray(joint_state["final_cache"], dtype=float)
    uav_result = joint_state

    final_edge_positions = edge_positions.copy()
    for uav_order, service_index in enumerate(uav_indices):
        final_edge_positions[service_index] = uav_result["final_positions"][uav_order]

    # 阶段四：最终 C_exec 决定本时隙真实可行节点；不是 policy 自己生成的 mask。
    feasible = build_feasible_nodes(
        arrivals,
        task_types,
        task_cache,
        final_cache,
        local_columns=local,
        service_columns=service,
        cloud_column=cloud,
    )

    # 阶段五：先用完整带宽计算每个任务-edge链路的基准速率，供 Gamma KKT 使用。
    # 这里调用 environment，solver 不复制 Shannon 公式。
    base_rates = np.zeros((task_count, service_count), dtype=float)
    for task in np.flatnonzero(arrivals):
        for service_index, kind in enumerate(kinds):
            try:
                if kind in {NodeType.SERVICE_VEHICLE, NodeType.RSU}:
                    rate = env.ground_rate(
                        vehicle_positions[task],
                        final_edge_positions[service_index],
                        kind,
                        bandwidth_fraction=1.0,
                    )
                else:
                    rate = env.uav_rate(
                        vehicle_positions[task],
                        final_edge_positions[service_index],
                        bandwidth_fraction=1.0,
                    )
            except (ValueError, FloatingPointError):
                rate = 0.0
            if math.isfinite(float(rate)) and float(rate) > 0.0:
                base_rates[task, service_index] = float(rate)
    positive_base_rate = np.where(base_rates > 0.0, base_rates, 1.0)
    # 可行节点不仅要有 C_exec 缓存，还必须在当前 slot 有正的物理通信速率。
    # Local/Cloud 分别由本地缓存和 Cloud fallback 规则保证；这里仅从边缘候选
    # 中移除当前不可达节点，避免把通信失败的候选交给 Matching 再重复拒绝。
    feasible = tuple(
        tuple(
            node
            for node in nodes
            if node not in service
            or base_rates[task, service.index(node)] > 0.0
        )
        for task, nodes in enumerate(feasible)
    )

    computation = config["computation"]
    communication = config["communication"]
    reward = config["reward"]
    if node_frequencies_hz is None:
        capacity_keys = {
            NodeType.SERVICE_VEHICLE: "sv",
            NodeType.RSU: "rsu",
            NodeType.UAV: "uav",
        }
        capacity_ghz = computation["capacity_ghz"]
        node_frequencies = np.asarray(
            [float(capacity_ghz[capacity_keys[kind]]) * 1.0e9 for kind in kinds],
            dtype=float,
        )
    else:
        node_frequencies = _vector(
            node_frequencies_hz,
            "node_frequencies_hz",
            service_count,
            positive=True,
        )
    powers = _config_vector_or_default(
        transmit_power_w,
        task_count,
        communication["tx_power_w"],
        "transmit_power_w",
        positive=True,
    )
    delay_refs = _config_vector_or_default(
        reference_delays_s,
        task_count,
        reward["reference_delay_s"],
        "reference_delays_s",
        positive=True,
    )
    energy_refs = _config_vector_or_default(
        reference_energies_j,
        task_count,
        reward["reference_energy_j"],
        "reference_energies_j",
        positive=True,
    )
    time_weight = float(reward["delay_weight"] if omega_time is None else omega_time)
    energy_weight = float(reward["energy_weight"] if omega_energy is None else omega_energy)
    if not math.isclose(time_weight + energy_weight, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("omega_time和omega_energy必须和为1。")

    # 阶段六：matching 每次候选评价都会重新触发 KKT 和 environment 物理计算。
    matching = transfer_exchange_matching(
        feasible,
        environment=env,
        task_arrivals=arrivals,
        task_input_bits=input_bits,
        task_cpu_cycles=cycles,
        task_vehicle_positions=vehicle_positions,
        service_node_positions=final_edge_positions,
        service_node_types=kinds,
        local_columns=local,
        service_columns=service,
        cloud_column=cloud,
        base_rates_bps=positive_base_rate,
        node_frequencies_hz=node_frequencies,
        transmit_power_w=powers,
        reference_delays_s=delay_refs,
        reference_energies_j=energy_refs,
        omega_time=time_weight,
        omega_energy=energy_weight,
        epsilon_compute=epsilon_compute,
        epsilon_bandwidth=epsilon_bandwidth,
        max_matching_updates=max_matching_updates,
        multiplier_tolerance=multiplier_tolerance,
        max_kkt_iterations=max_kkt_iterations,
    )

    # 阶段七：只对 C_old -> C_exec 实际缓存状态计算成本，再与 Q 合成 G/reward。
    cache_cost = 0.0
    for index, kind in enumerate(kinds):
        cache_cost += float(
            env.cache_update_cost(
                previous[index],
                final_cache[index],
                node_type=kind,
                service_sizes=service_sizes,
            )["total_cost"]
        )
    uav_cost = 0.0
    for uav_order, service_index in enumerate(uav_indices):
        uav_cost += float(
            env.uav_movement_cost(
                previous_uav[uav_order],
                uav_result["final_positions"][uav_order],
                movement_epsilon_m=position_epsilon_m,
            )["total_cost"]
        )
    system = env.system_utility(
        matching["total_task_utility"],
        cache_cost,
        uav_cost,
    )
    failure_reason_counts = {
        "no_feasible_node": 0,
        "resource_infeasible": 0,
        "latency_constraint": 0,
        "communication_unavailable": 0,
        "numerical_failure": 0,
    }
    for task in np.flatnonzero(arrivals):
        if matching["execution_nodes"][int(task)] is None:
            failure_reason_counts["no_feasible_node"] += 1
    return SolverResult(
        candidate_cache=_readonly(candidate_cache),
        final_cache=_readonly(final_cache),
        actual_cache_update=_readonly(adaptive_cache["cache_changed"], dtype=bool),
        cache_candidate_changed=_readonly(
            adaptive_cache["cache_candidate_changed"], dtype=bool
        ),
        cache_trigger=_readonly(adaptive_cache["cache_trigger"], dtype=bool),
        cache_changed=_readonly(adaptive_cache["cache_changed"], dtype=bool),
        cache_gain=_readonly(adaptive_cache["cache_gain"]),
        cache_switch_cost=_readonly(adaptive_cache["cache_switch_cost"]),
        keep_predicted_utility=_readonly(adaptive_cache["keep_predicted_utility"]),
        candidate_predicted_utility=_readonly(
            adaptive_cache["candidate_predicted_utility"]
        ),
        cooperative_candidate_utility=float(
            adaptive_cache["cooperative_candidate_utility"]
        ),
        cache_duplicate_count=int(adaptive_cache["cache_duplicate_count"]),
        cache_unique_services=int(adaptive_cache["cache_unique_services"]),
        cache_local_swap_count=int(adaptive_cache["cache_local_swap_count"]),
        services_changed=tuple(adaptive_cache["services_changed"]),
        cache_check=bool(adaptive_cache["cache_check"]),
        final_uav_positions=_readonly(uav_result["final_positions"]),
        actual_uav_movement=_readonly(uav_result["actual_movement"], dtype=bool),
        uav_candidate_movement=_readonly(
            uav_result["candidate_movement"], dtype=bool
        ),
        uav_check=bool(uav_result["uav_check"]),
        uav_selected_action=str(uav_result["uav_selected_action"]),
        uav_J_keep=float(uav_result["uav_J_keep"]),
        uav_J_cache_only=float(uav_result["uav_J_cache_only"]),
        uav_J_uav_only=float(uav_result["uav_J_uav_only"]),
        uav_J_joint=float(uav_result["uav_J_joint"]),
        uav_candidate_count=int(uav_result["uav_candidate_count"]),
        uav_candidate_move_cost=float(uav_result["uav_candidate_move_cost"]),
        ground_service_gap=float(uav_result["ground_service_gap"]),
        uav_fallback=bool(uav_result["fallback_applied"]),
        feasible_nodes=feasible,
        assignment=_readonly(matching["assignment"]),
        execution_nodes=tuple(matching["execution_nodes"]),
        compute_allocations=_readonly(matching["compute_allocations"]),
        bandwidth_allocations=_readonly(matching["bandwidth_allocations"]),
        task_results=tuple(matching["task_results"]),
        system_utility=dict(system),
        matching_stable=bool(matching["matching_stable"]),
        matching_migration_count=int(matching["migration_count"]),
        matching_swap_count=int(matching["swap_count"]),
        kkt_failure_count=int(matching.get("kkt_failure_count", 0)),
        failure_reason_counts=failure_reason_counts,
    )


__all__ = [
    "SolverResult",
    "build_feasible_nodes",
    "decide_adaptive_cache_update",
    "predicted_cache_utility",
    "solve_bandwidth_allocation",
    "solve_cache",
    "solve_cooperative_cache",
    "solve_compute_allocation",
    "solve_resources",
    "solve_slot",
    "select_demand_aware_uav_cache_state",
    "transfer_exchange_matching",
]
