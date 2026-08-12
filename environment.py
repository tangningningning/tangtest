"""论文物理环境层。

本模块实现 ``paper_spec.md`` 第 9--13 节的数值模型，并提供一个连接 SUMO、
确定性 solver 和策略动作的 ``MECEnvironment`` 门面：

* 通信：地面链路、UAV LoS/NLoS、总噪声功率语义和 Shannon 上行速率；
* 执行：Local、Service Vehicle、RSU、UAV、Cloud 五类节点的时延/车辆能耗；
* 成本：真实缓存状态变化和 UAV 位置变化对应的成本；
* 目标：任务效用 ``Q``、系统效用 ``G`` 和即时 reward；
* 闭环：每个 slot 读取 SUMO 位置，调用 ``solver.solve_slot``，再写回最终
  缓存、UAV 位置和五类执行统计。

任务分配、严格背包、需求感知 UAV 候选比较、matching 和 KKT 的具体算法由 ``solver.py``
负责；本文件只接受 solver 已决定的节点/资源，并用唯一的物理公式计算结果，
避免项目中出现第二套 delay、energy 或 reward 实现。
"""

from __future__ import annotations

import math
import os
from enum import Enum
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

try:  # 允许从项目根目录直接运行，也允许以后作为包的一部分导入。
    from .data import load_config
    from .mobility import MobilityProvider, SumoMobilityProvider, TraceMobilityProvider
except ImportError:  # pragma: no cover - depends on the import context
    from data import load_config
    from mobility import MobilityProvider, SumoMobilityProvider, TraceMobilityProvider


class NodeType(str, Enum):
    """论文要求独立保留的五类执行节点身份。"""

    LOCAL = "LOCAL"
    SERVICE_VEHICLE = "SERVICE_VEHICLE"
    RSU = "RSU"
    UAV = "UAV"
    CLOUD = "CLOUD"


def _node_type(value: NodeType | str) -> NodeType:
    if isinstance(value, NodeType):
        return value
    try:
        return NodeType(str(value).strip().upper())
    except ValueError as error:
        allowed = ", ".join(item.value for item in NodeType)
        raise ValueError(f"execution_kind必须是{allowed}之一。") from error


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"配置节{name!r}必须是映射。")
    return value


def _scalar(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name}不能是布尔值。")
    array = np.asarray(value, dtype=float)
    if array.ndim != 0:
        raise ValueError(f"{name}必须是标量。")
    result = float(array)
    if not math.isfinite(result):
        raise ValueError(f"{name}不能是NaN或Inf。")
    return result


def _nonnegative(value: Real, name: str) -> float:
    result = _scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name}必须为非负数。")
    return result


def _positive(value: Real, name: str) -> float:
    result = _scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name}必须为正数。")
    return result


def _probability(value: Real, name: str) -> float:
    result = _scalar(value, name)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name}必须位于[0,1]。")
    return result


def _position(value: Sequence[Real] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (2,):
        raise ValueError(f"{name}形状必须为(2,)，实际为{array.shape}。")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}不能包含NaN或Inf。")
    return array


def _binary_vector(value: Sequence[Real] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name}形状必须为(F,)。")
    if not np.all(np.isfinite(array)) or not np.all((array == 0.0) | (array == 1.0)):
        raise ValueError(f"{name}必须是有限的0/1向量。")
    return array


def _finite_vector(value: Sequence[Real] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name}必须是一维有限数值向量。")
    return array


def _unit_factor(unit: str, *, expected: str) -> float:
    normalized = str(unit).strip().lower()
    if expected == "bandwidth":
        factors = {"hz": 1.0, "mhz": 1.0e6}
    elif expected == "frequency":
        factors = {"hz": 1.0, "ghz": 1.0e9}
    elif expected == "service_size":
        factors = {"bit": 1.0, "byte": 8.0, "mb": 8.0e6}
    else:  # pragma: no cover - private helper is only called with known categories
        raise ValueError(f"未知单位类别：{expected}")
    if normalized not in factors:
        raise ValueError(f"不支持的{expected}单位：{unit}")
    return factors[normalized]


def _bandwidth_hz(bandwidth: Real, unit: str) -> float:
    return _positive(bandwidth, "bandwidth") * _unit_factor(unit, expected="bandwidth")


def _frequency_hz(frequency: Real, unit: str) -> float:
    return _positive(frequency, "frequency") * _unit_factor(unit, expected="frequency")


def ground_distance(
    vehicle_position: Sequence[Real] | np.ndarray,
    service_position: Sequence[Real] | np.ndarray,
) -> float:
    """计算任务车辆到服务车辆或RSU的二维欧氏距离，单位米。"""

    first = _position(vehicle_position, "vehicle_position")
    second = _position(service_position, "service_position")
    return float(np.linalg.norm(first - second))


def ground_channel_gain(
    vehicle_position: Sequence[Real] | np.ndarray,
    service_position: Sequence[Real] | np.ndarray,
    *,
    reference_gain: Real,
    path_loss_exponent: Real,
) -> float:
    """按论文地面链路 ``h=beta_g*d^(-alpha_g)`` 计算平均功率增益。

    ``reference_gain`` 是参考距离下的功率增益，``path_loss_exponent`` 是地面
    链路损耗指数；两端重合时距离模型没有定义，因此显式报错而不是返回无穷大。
    """

    distance = ground_distance(vehicle_position, service_position)
    if distance <= 0.0:
        raise ValueError("地面链路两端位置不能完全重合。")
    gain = _positive(reference_gain, "reference_gain") * distance ** (
        -_positive(path_loss_exponent, "path_loss_exponent")
    )
    if not math.isfinite(gain):
        raise ValueError("地面信道增益不是有限值。")
    return float(gain)


def los_probability(angle_deg: Real, eta_1: Real, eta_2: Real) -> float:
    """按论文 Logistic 公式计算 LoS 概率。

    角度使用“度”而不是弧度，和配置中的 ``los_eta_1/los_eta_2`` 约定一致。
    返回值保持在严格的 ``(0,1)`` 内，便于后续计算 ``1-P_L``。
    """

    angle = _scalar(angle_deg, "angle_deg")
    eta1 = _positive(eta_1, "eta_1")
    eta2 = _positive(eta_2, "eta_2")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            probability = 1.0 / (1.0 + eta1 * np.exp(-eta2 * (angle - eta1)))
        except FloatingPointError as error:
            raise ValueError("LoS概率计算出现数值异常。") from error
    probability = float(np.clip(probability, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)))
    return probability


def uav_channel(
    vehicle_position: Sequence[Real] | np.ndarray,
    uav_position: Sequence[Real] | np.ndarray,
    *,
    height_m: Real,
    beta_l: Real,
    alpha_l: Real,
    beta_n: Real,
    alpha_n: Real,
    eta_1: Real,
    eta_2: Real,
) -> dict[str, float]:
    """计算 UAV 三维距离、仰角、LoS/NLoS 概率和平均信道增益。

    论文模型先根据水平距离和固定高度得到三维距离，再计算 LoS 概率，最后
    按 ``P_L h_L + (1-P_L)h_N`` 得到确定性的概率加权平均增益。
    """

    vehicle = _position(vehicle_position, "vehicle_position")
    uav = _position(uav_position, "uav_position")
    height = _positive(height_m, "height_m")
    horizontal = float(np.linalg.norm(vehicle - uav))
    distance = float(np.sqrt(horizontal**2 + height**2))
    angle = float(np.degrees(np.arcsin(height / distance)))
    p_los = los_probability(angle, eta_1, eta_2)
    p_nlos = 1.0 - p_los
    los_gain = _positive(beta_l, "beta_l") * distance ** (-_positive(alpha_l, "alpha_l"))
    nlos_gain = _positive(beta_n, "beta_n") * distance ** (-_positive(alpha_n, "alpha_n"))
    average_gain = p_los * los_gain + p_nlos * nlos_gain
    if not math.isfinite(average_gain) or average_gain < 0.0:
        raise ValueError("UAV平均信道增益不是有效的非负值。")
    return {
        "horizontal_distance_m": horizontal,
        "distance_m": distance,
        "elevation_angle_deg": angle,
        "los_probability": p_los,
        "nlos_probability": p_nlos,
        "los_gain": float(los_gain),
        "nlos_gain": float(nlos_gain),
        "average_gain": float(average_gain),
    }


def uav_channel_gain(
    vehicle_position: Sequence[Real] | np.ndarray,
    uav_position: Sequence[Real] | np.ndarray,
    **parameters: Real,
) -> float:
    """返回论文 UAV 空地链路的概率加权平均信道增益。"""

    return uav_channel(vehicle_position, uav_position, **parameters)["average_gain"]


def edge_upload_rate(
    *,
    bandwidth_fraction: Real,
    bandwidth_hz: Real,
    transmit_power_w: Real,
    channel_gain: Real,
    noise_power_w: Real,
) -> float:
    """按论文总噪声功率 ``sigma²`` 和 Shannon 模型计算边缘上行速率。

    速率为 ``R = B_alloc log2(1 + p h / sigma²)``。这里的
    ``noise_power_w`` 已经是总噪声功率，不再乘以带宽；带宽只通过
    ``bandwidth_fraction * bandwidth_hz`` 影响可用传输带宽。
    """

    fraction = _probability(bandwidth_fraction, "bandwidth_fraction")
    bandwidth = _positive(bandwidth_hz, "bandwidth_hz")
    power = _nonnegative(transmit_power_w, "transmit_power_w")
    gain = _nonnegative(channel_gain, "channel_gain")
    # policy/资源求解传入带宽比例，物理公式使用比例对应的实际 Hz。
    allocated_bandwidth = fraction * bandwidth
    if allocated_bandwidth == 0.0 or power == 0.0 or gain == 0.0:
        return 0.0

    noise = _positive(noise_power_w, "noise_power_w")

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            # sigma² 位于 SNR 分母；调用层不能再把 noise 当作 PSD 重复换算。
            snr = power * gain / noise
            rate = allocated_bandwidth * np.log1p(snr) / np.log(2.0)
        except FloatingPointError as error:
            raise ValueError("边缘上行速率计算出现数值异常。") from error
    if not math.isfinite(float(rate)) or float(rate) < 0.0:
        raise ValueError("边缘上行速率必须是有限非负数。")
    return float(rate)


def transmission_delay_energy(
    data_bits: Real,
    rate_bps: Real,
    transmit_power_w: Real,
    *,
    delay_multiplier: Real = 1.0,
) -> dict[str, float]:
    """计算上传时延和车辆发射能耗。

    ``delay_multiplier`` 只作用于传输时延，用于 Cloud fallback 的 ``psi_c``；
    发射能耗仍按 ``pD/R`` 计算，不能把 Cloud 的时延扩展系数乘到能耗上。
    """

    data = _nonnegative(data_bits, "data_bits")
    rate = _nonnegative(rate_bps, "rate_bps")
    power = _nonnegative(transmit_power_w, "transmit_power_w")
    multiplier = _positive(delay_multiplier, "delay_multiplier")
    if data == 0.0:
        return {"delay_s": 0.0, "energy_j": 0.0}
    if rate <= 0.0:
        raise ValueError("data_bits为正时rate_bps必须为正数。")
    delay = multiplier * data / rate
    energy = power * data / rate
    if not math.isfinite(delay) or not math.isfinite(energy):
        raise ValueError("传输时延或车辆发射能耗不是有限值。")
    return {"delay_s": float(delay), "energy_j": float(energy)}


def _zero_execution(kind: NodeType) -> dict[str, Any]:
    return {
        "execution_kind": kind.value,
        "rate_bps": 0.0,
        "transmission_delay_s": 0.0,
        "computation_delay_s": 0.0,
        "completion_delay_s": 0.0,
        "transmission_energy_j": 0.0,
        "local_computation_energy_j": 0.0,
        "vehicle_energy_j": 0.0,
    }


def execute_task(
    execution_kind: NodeType | str,
    *,
    arrived: bool,
    input_bits: Real,
    cpu_cycles: Real,
    local_frequency_hz: Real | None = None,
    switching_capacitance: Real | None = None,
    compute_resource_fraction: Real | None = None,
    edge_frequency_hz: Real | None = None,
    bandwidth_fraction: Real | None = None,
    bandwidth_hz: Real | None = None,
    transmit_power_w: Real | None = None,
    channel_gain: Real | None = None,
    noise_power_w: Real | None = None,
    precomputed_rate_bps: Real | None = None,
    cloud_rate_bps: Real | None = None,
    cloud_frequency_hz: Real | None = None,
    delay_expansion: Real | None = None,
) -> dict[str, Any]:
    """按五类节点之一计算单个不可分割任务的最终时延和车辆能耗。

    该函数只接受上游已经确定的执行节点和资源比例；它不做任务分配、缓存
    可行性判断或资源求解。Local不上传；三个边缘节点使用上传加计算时延；
    Cloud使用 ``psi_c D/R_c + L/F_c``，且 ``psi_c`` 不作用于能耗。
    返回字典同时保留传输时延、计算时延、总时延、传输能耗和车辆能耗，供
    matching 的候选评价、最终统计和 reward 统一使用。函数只接受上游已经
    确定的执行节点和资源比例；它不做任务分配、缓存可行性判断或资源求解。
    """

    kind = _node_type(execution_kind)
    if not isinstance(arrived, (bool, np.bool_)):
        raise ValueError("arrived必须是布尔值。")
    data = _nonnegative(input_bits, "input_bits")
    cycles = _nonnegative(cpu_cycles, "cpu_cycles")
    if not arrived:
        if data != 0.0 or cycles != 0.0:
            raise ValueError("未到达任务的input_bits和cpu_cycles必须均为0。")
        return _zero_execution(kind)
    if data <= 0.0 or cycles <= 0.0:
        raise ValueError("已到达任务的input_bits和cpu_cycles必须为正数。")

    if kind is NodeType.LOCAL:
        # Local 不发生上传，时延只有本地 CPU 计算时间，车辆能耗采用动态电容模型。
        frequency = _positive(
            local_frequency_hz if local_frequency_hz is not None else 0.0,
            "local_frequency_hz",
        )
        capacitance = _nonnegative(
            switching_capacitance if switching_capacitance is not None else 0.0,
            "switching_capacitance",
        )
        computation_delay = cycles / frequency
        computation_energy = capacitance * cycles * frequency**2
        if not math.isfinite(computation_delay) or not math.isfinite(computation_energy):
            raise ValueError("本地计算结果不是有限值。")
        result = _zero_execution(kind)
        result.update(
            computation_delay_s=float(computation_delay),
            completion_delay_s=float(computation_delay),
            local_computation_energy_j=float(computation_energy),
            vehicle_energy_j=float(computation_energy),
        )
        return result

    if kind in {NodeType.SERVICE_VEHICLE, NodeType.RSU, NodeType.UAV}:
        # 三类边缘节点共享“上传 + 边缘计算”的结构；差别由上游传入的速率和
        # 节点 CPU 频率决定，compute_resource_fraction 来自 KKT 资源分配。
        fraction = _positive(
            compute_resource_fraction if compute_resource_fraction is not None else 0.0,
            "compute_resource_fraction",
        )
        if fraction > 1.0:
            raise ValueError("compute_resource_fraction必须位于(0,1]。")
        edge_frequency = _positive(
            edge_frequency_hz if edge_frequency_hz is not None else 0.0,
            "edge_frequency_hz",
        )
        if precomputed_rate_bps is None:
            rate = edge_upload_rate(
                bandwidth_fraction=bandwidth_fraction if bandwidth_fraction is not None else 0.0,
                bandwidth_hz=bandwidth_hz if bandwidth_hz is not None else 0.0,
                transmit_power_w=transmit_power_w if transmit_power_w is not None else 0.0,
                channel_gain=channel_gain if channel_gain is not None else 0.0,
                noise_power_w=noise_power_w,
            )
        else:
            rate = _positive(precomputed_rate_bps, "precomputed_rate_bps")
        transmission = transmission_delay_energy(
            data,
            rate,
            transmit_power_w if transmit_power_w is not None else 0.0,
        )
        computation_delay = cycles / (fraction * edge_frequency)
        if not math.isfinite(computation_delay):
            raise ValueError("边缘计算时延不是有限值。")
        return {
            "execution_kind": kind.value,
            "rate_bps": float(rate),
            "transmission_delay_s": transmission["delay_s"],
            "computation_delay_s": float(computation_delay),
            "completion_delay_s": float(transmission["delay_s"] + computation_delay),
            "transmission_energy_j": transmission["energy_j"],
            "local_computation_energy_j": 0.0,
            "vehicle_energy_j": transmission["energy_j"],
        }

    if kind is NodeType.CLOUD:
        # Cloud 按论文作为始终可行的 fallback 节点；psi_c 只扩大上传时延。
        rate = _positive(cloud_rate_bps if cloud_rate_bps is not None else 0.0, "cloud_rate_bps")
        frequency = _positive(
            cloud_frequency_hz if cloud_frequency_hz is not None else 0.0,
            "cloud_frequency_hz",
        )
        transmission = transmission_delay_energy(
            data,
            rate,
            transmit_power_w if transmit_power_w is not None else 0.0,
            delay_multiplier=delay_expansion if delay_expansion is not None else 0.0,
        )
        computation_delay = cycles / frequency
        if not math.isfinite(computation_delay):
            raise ValueError("Cloud计算时延不是有限值。")
        return {
            "execution_kind": kind.value,
            "rate_bps": float(rate),
            "transmission_delay_s": transmission["delay_s"],
            "computation_delay_s": float(computation_delay),
            "completion_delay_s": float(transmission["delay_s"] + computation_delay),
            "transmission_energy_j": transmission["energy_j"],
            "local_computation_energy_j": 0.0,
            "vehicle_energy_j": transmission["energy_j"],
        }

    raise AssertionError("NodeType分支不完整。")  # pragma: no cover


def cache_update_cost(
    previous_cache: Sequence[Real] | np.ndarray,
    current_cache: Sequence[Real] | np.ndarray,
    service_sizes_bits: Sequence[Real] | np.ndarray,
    *,
    fixed_cost_per_update: Real,
    loading_energy_per_bit: Real,
    loading_cost_weight: Real,
) -> dict[str, Any]:
    """按最终缓存状态计算一个可更新节点的实际缓存成本。

    这里只计算成本，不求解缓存状态。新增服务按 ``0->1`` 计加载能耗，删除
    服务不计加载能耗；状态没有实际变化时固定成本和加载成本都为零。成本依据
    最终 0/1 缓存差分决定，因此 PPO 请求的 trigger 可能和实际更新结果不同。
    """

    previous = _binary_vector(previous_cache, "previous_cache")
    current = _binary_vector(current_cache, "current_cache")
    if previous.shape != current.shape:
        raise ValueError("previous_cache和current_cache形状必须相同。")
    sizes = _finite_vector(service_sizes_bits, "service_sizes_bits")
    if sizes.shape != previous.shape or np.any(sizes <= 0.0):
        raise ValueError("service_sizes_bits必须是与缓存相同形状的正向量。")
    fixed = _nonnegative(fixed_cost_per_update, "fixed_cost_per_update")
    loading_per_bit = _nonnegative(loading_energy_per_bit, "loading_energy_per_bit")
    loading_weight = _nonnegative(loading_cost_weight, "loading_cost_weight")
    delta = current - previous
    added = tuple(int(index) for index in np.flatnonzero(delta > 0.0))
    removed = tuple(int(index) for index in np.flatnonzero(delta < 0.0))
    changed = bool(added or removed)
    loading_energy = loading_per_bit * float(sizes[list(added)].sum()) if added else 0.0
    weighted_loading_cost = loading_weight * loading_energy if changed else 0.0
    fixed_cost = fixed if changed else 0.0
    total_cost = fixed_cost + weighted_loading_cost
    if not all(math.isfinite(value) for value in (loading_energy, total_cost)):
        raise ValueError("缓存更新成本不是有限值。")
    return {
        "effective_trigger": changed,
        "added_services": added,
        "removed_services": removed,
        "loading_energy_j": float(loading_energy),
        "fixed_cost": float(fixed_cost),
        "weighted_loading_cost": float(weighted_loading_cost),
        "total_cost": float(total_cost),
    }


def _normalized_decision_time(slot_index: int, slot_count: int) -> float:
    """返回有限回合决策时隙的归一化时间，终止状态保持为 1。"""

    count = int(slot_count)
    index = int(slot_index)
    if count < 2:
        raise ValueError("归一化时间要求回合至少包含2个决策时隙。")
    if index < 0:
        raise ValueError("slot_index不能为负数。")
    return float(min(index, count - 1)) / float(count - 1)


def uav_movement_cost(
    previous_position: Sequence[Real] | np.ndarray,
    current_position: Sequence[Real] | np.ndarray,
    *,
    movement_energy_j_per_m: Real,
    fixed_cost_per_update: Real,
    movement_energy_cost_weight: Real,
    movement_epsilon_m: Real,
) -> dict[str, Any]:
    """按最终二维水平位置计算 UAV 实际移动能耗和成本。

    只有二维水平位移严格大于论文数值容差 ``epsilon_q`` 时，才计入固定更新
    成本和移动能耗。最大距离由 solver 的速度/时隙约束负责。
    """

    previous = _position(previous_position, "previous_position")
    current = _position(current_position, "current_position")
    distance = float(np.linalg.norm(current - previous))
    energy_per_meter = _nonnegative(movement_energy_j_per_m, "movement_energy_j_per_m")
    fixed = _nonnegative(fixed_cost_per_update, "fixed_cost_per_update")
    weight = _nonnegative(movement_energy_cost_weight, "movement_energy_cost_weight")
    epsilon = _nonnegative(movement_epsilon_m, "movement_epsilon_m")
    moved = distance > epsilon
    movement_energy = energy_per_meter * distance if moved else 0.0
    fixed_cost = fixed if moved else 0.0
    weighted_movement_cost = weight * movement_energy if moved else 0.0
    total_cost = fixed_cost + weighted_movement_cost
    if not math.isfinite(total_cost):
        raise ValueError("UAV移动成本不是有限值。")
    return {
        "moved": moved,
        "movement_epsilon_m": epsilon,
        "movement_distance_m": distance,
        "movement_energy_j": float(movement_energy),
        "fixed_cost": float(fixed_cost),
        "weighted_movement_cost": float(weighted_movement_cost),
        "total_cost": float(total_cost),
    }


def task_utility(
    arrived: Sequence[bool | Real] | np.ndarray,
    completion_delay_s: Sequence[Real] | np.ndarray,
    vehicle_energy_j: Sequence[Real] | np.ndarray,
    *,
    delay_weight: Real,
    energy_weight: Real,
    delay_reference_s: Real,
    energy_reference_j: Real,
) -> dict[str, Any]:
    """计算论文任务服务效用，不裁剪自然产生的负效用。

    每个已到达任务使用归一化 delay/energy 的加权和；未到达任务的时延和能耗
    必须为0。完整的 per-task utility 会被 matching 用于比较候选节点，并汇总
    成时隙任务效用 ``Q``。
    """

    arrivals = np.asarray(arrived, dtype=float)
    if arrivals.ndim != 1 or not np.all(np.isfinite(arrivals)):
        raise ValueError("arrived必须是一维有限向量。")
    if not np.all((arrivals == 0.0) | (arrivals == 1.0)):
        raise ValueError("arrived必须是0/1向量。")
    delays = _finite_vector(completion_delay_s, "completion_delay_s")
    energies = _finite_vector(vehicle_energy_j, "vehicle_energy_j")
    if delays.shape != arrivals.shape or energies.shape != arrivals.shape:
        raise ValueError("arrived、completion_delay_s和vehicle_energy_j形状必须相同。")
    if np.any(delays < 0.0) or np.any(energies < 0.0):
        raise ValueError("时延和车辆能耗必须为非负数。")
    inactive = arrivals == 0.0
    if np.any(delays[inactive] != 0.0) or np.any(energies[inactive] != 0.0):
        raise ValueError("未到达任务的时延和车辆能耗必须为0。")
    delay_w = _probability(delay_weight, "delay_weight")
    energy_w = _probability(energy_weight, "energy_weight")
    if not math.isclose(delay_w + energy_w, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("delay_weight与energy_weight之和必须为1。")
    delay_ref = _positive(delay_reference_s, "delay_reference_s")
    energy_ref = _positive(energy_reference_j, "energy_reference_j")
    per_task = arrivals * (
        delay_w * (1.0 - delays / delay_ref) + energy_w * (1.0 - energies / energy_ref)
    )
    if not np.all(np.isfinite(per_task)):
        raise ValueError("任务效用计算结果不是有限值。")
    return {
        "per_task_utility": np.asarray(per_task, dtype=float),
        "total_task_utility": float(np.sum(per_task)),
    }


def system_utility(
    task_service_utility_value: Real,
    cache_update_cost_value: Real,
    uav_update_cost_value: Real,
) -> dict[str, float]:
    """计算 ``G=Q-C^ca-C^q``，不添加额外惩罚或裁剪。

    reward 对应实际执行结果：任务效用减去实际缓存更新成本和实际 UAV 移动
    成本。PPO 原始辅助动作不会直接作为 reward，避免把未执行动作当作系统收益。
    """

    task_value = _scalar(task_service_utility_value, "task_service_utility")
    cache_cost = _nonnegative(cache_update_cost_value, "cache_update_cost")
    uav_cost = _nonnegative(uav_update_cost_value, "uav_update_cost")
    total = task_value - cache_cost - uav_cost
    if not math.isfinite(total):
        raise ValueError("系统效用不是有限值。")
    return {
        "task_service_utility": task_value,
        "cache_update_cost": cache_cost,
        "uav_update_cost": uav_cost,
        "total_system_utility": float(total),
        "reward": float(total),
    }


class MECEnvironment:
    """论文闭环环境门面，连接数据层、solver 和物理公式。

    ``reset`` 建立 SUMO、缓存、UAV 位置、道路预测器和初始观测；``step``
    接受节点级 Cache/UAV trigger，按“UAV → Cache → 可行节点 → matching → KKT
    → 实际执行 → reward”的顺序调用 solver，并把最终配置写回下一状态。
    纯物理函数仍可独立调用，便于单元测试和代表性任务检查。
    """

    def __init__(
        self,
        config: Mapping[str, Any] | str | os.PathLike[str] | None = None,
        *,
        sumo_configuration_file: str | os.PathLike[str] | None = None,
        data_source: Mapping[str, Any] | None = None,
        epsilon_compute: Real | None = None,
        epsilon_bandwidth: Real | None = None,
    ):
        self._config_is_mapping = isinstance(config, Mapping)
        self._config_path = None if self._config_is_mapping or config is None else str(config)
        if isinstance(config, Mapping):
            self.config = dict(config)
        else:
            self.config = load_config(config) if config is not None else load_config()
        self._sumo_configuration_file = (
            None if sumo_configuration_file is None else str(sumo_configuration_file)
        )
        self._data_source = data_source
        self._epsilon_compute = self._optional_epsilon(epsilon_compute, "epsilon_compute")
        self._epsilon_bandwidth = self._optional_epsilon(
            epsilon_bandwidth, "epsilon_bandwidth"
        )
        self._data: Mapping[str, Any] | None = None
        # SUMO 与离线回放共享同一套当前 slot 读取接口；Trace 模式不持有 TraCI。
        self._mobility_provider: MobilityProvider | None = None
        self._mobility_source = "uninitialized"
        self._sumo_api: Any | None = None
        self._owns_sumo_api = False
        self._slot_index: int | None = None
        self._times = np.empty(0, dtype=float)
        self._vehicle_ids: tuple[str, ...] = ()
        self._service_vehicle_ids: tuple[str, ...] = ()
        self._task_vehicle_ids: tuple[str, ...] = ()
        self._service_node_types: tuple[NodeType, ...] = ()
        self._uav_service_indices: tuple[int, ...] = ()
        self._service_sizes = np.empty(0, dtype=float)
        self._cache_capacities = np.empty(0, dtype=float)
        self._cache = np.empty((0, 0), dtype=float)
        self._local_cache = np.empty((0, 0), dtype=float)
        self._uav_positions = np.empty((0, 2), dtype=float)
        self._predicted_positions = np.empty((0, 2), dtype=float)
        self._predicted_popularity = np.empty(0, dtype=float)
        self._predictor: Any | None = None
        self._road_predictor: Any | None = None
        self._mobility_popularity_predictor: Any | None = None
        self._road_predictions: dict[str, Mapping[str, Any]] = {}
        self._last_node_prediction: dict[str, Any] = {}
        self._service_vehicle_replacement_rows = np.empty(0, dtype=bool)
        self._last_actual_states: dict[str, Mapping[str, Any]] = {}
        self._prediction_mode = "uninitialized"
        self._role_manager: Any | None = None
        self._role_state: dict[str, Any] = {}
        self._prediction_origin_slot: int | None = None

    @staticmethod
    def _optional_epsilon(value: Real | None, name: str) -> float | None:
        if value is None:
            return None
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0 or numeric > 1.0:
            raise ValueError(f"{name}必须位于(0,1]。")
        return numeric

    @property
    def cache_controller_mode(self) -> str:
        """返回高层缓存控制器模式；reference 是默认且保持旧闭环语义。"""

        section = self.config.get("cache_controller", {})
        if not isinstance(section, Mapping):
            section = {}
        mode = str(section.get("mode", "reference")).strip().lower()
        if mode not in {"reference", "ppo"}:
            raise ValueError("cache_controller.mode只能是reference或ppo。")
        return mode

    @staticmethod
    def _binary_matrix(
        value: Sequence[Sequence[Real]] | np.ndarray,
        name: str,
    ) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim != 2:
            raise ValueError(f"{name}必须是二维数组。")
        if array.dtype == bool:
            return array.astype(float, copy=True)
        numeric = np.asarray(array, dtype=float)
        if not np.all(np.isfinite(numeric)) or not np.all(
            (numeric == 0.0) | (numeric == 1.0)
        ):
            raise ValueError(f"{name}必须只包含0/1。")
        return numeric.copy()

    @staticmethod
    def _random_feasible_cache(
        service_sizes: np.ndarray,
        capacities: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        result = np.zeros((capacities.size, service_sizes.size), dtype=float)
        for node_index, capacity in enumerate(capacities):
            remaining = float(capacity)
            for service_index in rng.permutation(service_sizes.size):
                size = float(service_sizes[service_index])
                if size <= remaining + 1.0e-12:
                    result[node_index, service_index] = 1.0
                    remaining -= size
        return result

    @staticmethod
    def _position_matrix(
        states: Mapping[str, Mapping[str, Any]],
        vehicle_ids: Sequence[str],
    ) -> np.ndarray:
        positions = np.asarray(
            [states[vehicle_id]["position_m"] for vehicle_id in vehicle_ids],
            dtype=float,
        )
        if positions.shape != (len(vehicle_ids), 2) or not np.all(np.isfinite(positions)):
            raise ValueError("车辆位置必须是有限的(V,2)数组。")
        return positions

    def _trajectory_states(self, slot_index: int) -> dict[str, dict[str, Any]]:
        if self._data is None or self._data.get("trajectory") is None:
            raise RuntimeError("当前数据源没有轨迹数组。")
        trajectory = self._data["trajectory"]
        vehicle_ids = tuple(str(item) for item in trajectory["vehicle_ids"])
        positions = np.asarray(trajectory["positions"], dtype=float)[int(slot_index)]
        speeds = np.asarray(trajectory["speeds"], dtype=float)[int(slot_index)]
        angles = trajectory.get("angles")
        angle_values = None if angles is None else np.asarray(angles, dtype=float)[int(slot_index)]
        states: dict[str, dict[str, Any]] = {}
        for index, vehicle_id in enumerate(vehicle_ids):
            state: dict[str, Any] = {
                "vehicle_id": vehicle_id,
                "position_m": positions[index].copy(),
                "speed_mps": float(speeds[index]),
                "valid": bool(np.all(np.isfinite(positions[index])) and np.isfinite(speeds[index])),
                "active": bool(np.all(np.isfinite(positions[index])) and np.isfinite(speeds[index])),
            }
            if angle_values is not None:
                state["angle_deg"] = float(angle_values[index])
            states[vehicle_id] = state
        return states

    def _configured_mobility_source(self) -> str:
        mobility = self.config.get("mobility", {})
        if not isinstance(mobility, Mapping):
            mobility = {}
        source = str(mobility.get("source", "sumo")).strip().lower()
        if source not in {"sumo", "trace"}:
            raise ValueError("mobility.source must be either 'sumo' or 'trace'")
        return source

    def _read_vehicle_states(self, slot_index: int) -> dict[str, dict[str, Any]]:
        """Read only the current slot from the selected mobility provider."""

        if self._mobility_provider is not None:
            return self._mobility_provider.states_at(int(slot_index))
        if self._data is not None and self._data.get("trajectory") is not None:
            return self._trajectory_states(slot_index)
        if self._sumo_api is not None:
            try:
                from .data import get_vehicle_states
            except ImportError:  # pragma: no cover - direct script import
                from data import get_vehicle_states
            return get_vehicle_states(self._sumo_api)
        raise RuntimeError("environment has no readable mobility provider")

    def _apply_role_state(self, role_state: Mapping[str, Any], *, initial: bool = False) -> bool:
        new_service_ids = tuple(str(item) for item in role_state["service_vehicle_ids"])
        new_task_ids = tuple(str(item) for item in role_state["task_vehicle_ids"])
        expected_service_count = sum(
            kind is NodeType.SERVICE_VEHICLE for kind in self._service_node_types
        )
        expected_task_count = self._local_cache.shape[0]
        if len(new_service_ids) != expected_service_count:
            raise RuntimeError(
                "当前有效服务车辆不足以保持既有动作/缓存维度；请降低vehicle_roles.num_service_vehicles。"
            )
        if len(new_task_ids) != expected_task_count:
            raise RuntimeError(
                "当前有效任务车辆不足以保持既有动作/工作负载维度；请降低vehicle_roles.num_task_vehicles。"
            )
        old_service_ids = self._service_vehicle_ids
        old_task_ids = self._task_vehicle_ids
        changed = old_service_ids != new_service_ids or old_task_ids != new_task_ids
        self._service_vehicle_replacement_rows = np.zeros(
            self._cache.shape[0] if self._cache.ndim == 2 else 0,
            dtype=bool,
        )
        if changed and not initial:
            # SV 身份替换是独立物理事件：新服务车辆对应缓存行先置空，
            # 该失效不直接记为 adaptive cache trigger；新行后续是否重新加载
            # 仍由对应 MultiBinary cache trigger 决定。
            old_service_rows = {
                vehicle_id: index for index, vehicle_id in enumerate(old_service_ids)
            }
            if self._cache.ndim == 2:
                remapped_cache = self._cache.copy()
                remapped_cache[: len(new_service_ids)] = 0.0
                for index, vehicle_id in enumerate(new_service_ids):
                    old_index = old_service_rows.get(vehicle_id)
                    if old_index is not None:
                        remapped_cache[index] = self._cache[old_index]
                    else:
                        self._service_vehicle_replacement_rows[index] = True
                self._cache = remapped_cache
            old_task_rows = {
                vehicle_id: index for index, vehicle_id in enumerate(old_task_ids)
            }
            if self._local_cache.ndim == 2:
                remapped_local_cache = self._local_cache.copy()
                remapped_local_cache[:] = 0.0
                for index, vehicle_id in enumerate(new_task_ids):
                    old_index = old_task_rows.get(vehicle_id)
                    if old_index is not None:
                        remapped_local_cache[index] = self._local_cache[old_index]
                self._local_cache = remapped_local_cache
        self._service_vehicle_ids = new_service_ids
        self._task_vehicle_ids = new_task_ids
        self._vehicle_ids = self._service_vehicle_ids + self._task_vehicle_ids
        self._role_state = dict(role_state)
        return changed

    def _refresh_vehicle_roles(self, slot_index: int) -> Mapping[str, Mapping[str, Any]]:
        if self._role_manager is None:
            return {}
        if self._mobility_provider is not None:
            states = self._mobility_provider.states_at(int(slot_index))
        elif self._data is not None and self._data.get("trajectory") is not None:
            states = self._trajectory_states(slot_index)
        elif self._sumo_api is not None:
            try:
                from .data import get_vehicle_states
            except ImportError:  # pragma: no cover - direct script import
                from data import get_vehicle_states
            states = get_vehicle_states(self._sumo_api)
        else:
            raise RuntimeError("当前环境没有可读取的SUMO或轨迹车辆状态。")
        role_state = self._role_manager.assign(states, slot=slot_index)
        changed = self._apply_role_state(role_state)
        if self._road_predictor is not None:
            # 转移统计只使用本次读到的全体活动车辆；不使用角色选择结果推断未来。
            self._road_predictor.observe(states, slot=slot_index)
        if changed and self._predictor is not None:
            interval_s = _positive(
                _section(self.config, "simulation")["decision_interval_s"],
                "simulation.decision_interval_s",
            )
            current_positions = self._position_matrix(states, self._vehicle_ids)
            self._predictor = self._new_predictor(interval_s)
            self._predictor.initialize(
                current_positions,
                self._velocity_matrix_from_states(states),
            )
            self._predicted_positions = current_positions.copy()
            self._road_predictions = self._prediction_mapping_from_positions(
                self._predicted_positions,
                origin_slot=slot_index,
            )
        elif changed and self._road_predictor is not None:
            selected_states = {
                vehicle_id: states[vehicle_id]
                for vehicle_id in self._vehicle_ids
                if vehicle_id in states
            }
            predictions = self._road_predictor.predict(
                selected_states,
                origin_slot=slot_index,
                update_history=False,
            )
            self._road_predictions = dict(predictions)
            self._predicted_positions = self._road_predictor.positions_matrix(
                predictions, self._vehicle_ids
            )
        return states

    def close(self) -> None:
        """关闭本环境自己启动的SUMO连接；不删除任何数据文件。"""

        if self._mobility_provider is not None:
            self._mobility_provider.close()
        self._mobility_provider = None
        if self._owns_sumo_api and self._sumo_api is not None:
            try:
                from .data import close as close_sumo
            except ImportError:  # pragma: no cover - direct script import
                from data import close as close_sumo
            close_sumo(self._sumo_api)
        self._sumo_api = None
        self._owns_sumo_api = False

    @property
    def mobility_diagnostics(self) -> dict[str, Any]:
        """Return replay diagnostics without exposing future trace rows."""

        if self._mobility_provider is None:
            return {
                "source": self._mobility_source,
                "future_access_count": 0,
                "sumo_start_count": 0,
            }
        return dict(self._mobility_provider.diagnostics())

    @property
    def service_sizes_mb(self) -> np.ndarray:
        """按现有服务配置生成一次固定服务目录，不生成任务请求。

        服务大小由 ``services.generation_seed`` 独立生成，和任务到达、服务类型
        及 SUMO 轨迹随机流相互隔离。缓存 solver 使用同一目录作为背包权重；因此
        本属性每次返回相同的 F 维服务数组，而不是每次访问都重新抽取一组大小。
        """

        services = _section(self.config, "services")
        count = int(services["count"])
        minimum = _positive(services["min_size_mb"], "services.min_size_mb")
        maximum = _positive(services["max_size_mb"], "services.max_size_mb")
        if maximum < minimum or str(services.get("size_mode", "uniform")).lower() != "uniform":
            raise ValueError("当前环境只支持services.size_mode=uniform且上下界有效。")
        if not math.isclose(minimum, round(minimum)) or not math.isclose(
            maximum, round(maximum)
        ):
            raise ValueError("严格背包的MB服务大小必须是整数。")
        rng = np.random.default_rng(int(services["generation_seed"]))
        # 当前正式容量语义以整数MB为离散单位；这与旧正式配置的
        # generation_seed/integer-MB约定一致，不对连续浮点大小做隐藏四舍五入。
        return rng.integers(
            int(round(minimum)),
            int(round(maximum)) + 1,
            size=count,
            dtype=np.int64,
        ).astype(float)

    def _load_rollout_data(
        self,
        data_source: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        """选择环境数据来源并补齐最小数据契约。

        显式 ``data_source`` 优先级最高，便于受控测试注入固定数组；否则从配置
        读取本地 workload/轨迹。正式路径使用严格因果的移动感知预测器，历史
        GRU 不参与环境闭环。
        """

        if data_source is not None:
            loaded = data_source
        elif self._data_source is not None:
            loaded = self._data_source
        elif self._config_is_mapping:
            try:
                from .data import (
                    generate_workload,
                    load_sumo_scene,
                )
            except ImportError:  # pragma: no cover - direct script import
                from data import (
                    generate_workload,
                    load_sumo_scene,
                )
            workload = generate_workload(self.config)
            loaded = {
                "config": self.config,
                "sumo_scene": load_sumo_scene(self.config),
                "trajectory": None,
                "workload": workload,
                "predicted_positions": None,
                "predicted_popularity": None,
                "prediction_source": "mobility_aware_popularity",
            }
        else:
            try:
                from .data import load_data
            except ImportError:  # pragma: no cover - direct script import
                from data import load_data
            loaded = load_data(self._config_path) if self._config_path else load_data()
        if not isinstance(loaded, Mapping):
            raise ValueError("data_source必须是data.py返回的映射。")
        required = {"workload", "config"}
        missing = required.difference(loaded)
        if missing:
            raise ValueError(f"data_source缺少字段：{sorted(missing)}")
        loaded_copy = dict(loaded)
        loaded_config = loaded_copy.get("config")
        if not isinstance(loaded_config, Mapping):
            raise ValueError("data_source.config must be a mapping")
        mobility = loaded_config.get("mobility", {})
        if not isinstance(mobility, Mapping):
            mobility = {}
        source = str(mobility.get("source", "sumo")).strip().lower()
        if source not in {"sumo", "trace"}:
            raise ValueError("mobility.source must be either 'sumo' or 'trace'")
        if source == "trace" and loaded_copy.get("trajectory") is not None:
            raise ValueError(
                "mobility.source=trace cannot be combined with a precomputed trajectory"
            )
        if loaded_copy.get("trajectory") is None:
            scene = loaded_copy.get("sumo_scene")
            if not isinstance(scene, Mapping) or scene.get("network") is None:
                try:
                    from .data import load_sumo_scene
                except ImportError:  # pragma: no cover - direct script import
                    from data import load_sumo_scene
                loaded_copy["sumo_scene"] = load_sumo_scene(loaded_config)
        return loaded_copy

    def _new_predictor(self, interval_s: float) -> Any:
        try:
            from .data import KalmanPredictor
        except ImportError:  # pragma: no cover - direct script import
            from data import KalmanPredictor
        prediction = _section(self.config, "prediction")
        kalman = _section(prediction, "kalman")
        return KalmanPredictor(
            self._vehicle_ids,
            dt_s=interval_s,
            measurement_noise_std_m=kalman["measurement_noise_std_m"],
            acceleration_noise_std_mps2=kalman["acceleration_noise_std_mps2"],
            initial_position_variance_m2=kalman["initial_position_variance_m2"],
            initial_velocity_variance_m2ps2=kalman["initial_velocity_variance_m2ps2"],
        )

    def _new_road_predictor(self, interval_s: float) -> Any:
        """构造正式 SUMO 预测器；它只消费网络和当前 lane 状态。"""

        try:
            from .data import RoadConstrainedVehiclePredictor
        except ImportError:  # pragma: no cover - direct script import
            from data import RoadConstrainedVehiclePredictor
        scene = self._data.get("sumo_scene") if self._data is not None else None
        network = scene.get("network") if isinstance(scene, Mapping) else None
        if network is None:
            raise RuntimeError("实时道路预测需要sumo_scene.network")
        prediction_config = self.config.get("trajectory_prediction", {})
        if not isinstance(prediction_config, Mapping):
            prediction_config = {}
        history_length = int(
            prediction_config.get(
                "history_length_slots",
                _section(self.config, "prediction").get("history_length_slots", 10),
            )
        )
        return RoadConstrainedVehiclePredictor(
            network,
            dt_s=interval_s,
            horizon_slots=int(prediction_config.get("horizon_slots", 3)),
            history_length_slots=history_length,
        )

    def _new_mobility_popularity_predictor(self) -> Any:
        """构造简单的因果需求预测器，不引入新的神经网络。"""

        try:
            from .data import MobilityAwarePopularityPredictor
        except ImportError:  # pragma: no cover - direct script import
            from data import MobilityAwarePopularityPredictor
        popularity_config = self.config.get("popularity_prediction", {})
        if not isinstance(popularity_config, Mapping):
            popularity_config = {}
        trajectory_config = self.config.get("trajectory_prediction", {})
        if not isinstance(trajectory_config, Mapping):
            trajectory_config = {}
        return MobilityAwarePopularityPredictor(
            int(_section(self.config, "services")["count"]),
            history_decay=float(popularity_config.get("history_decay", 0.8)),
            preference_smoothing=float(
                popularity_config.get("preference_smoothing", 1.0)
            ),
            horizon_slots=int(trajectory_config.get("horizon_slots", 3)),
        )

    def _prediction_horizon(self) -> int:
        if self._mobility_popularity_predictor is not None:
            return int(self._mobility_popularity_predictor.horizon_slots)
        trajectory_config = self.config.get("trajectory_prediction", {})
        if not isinstance(trajectory_config, Mapping):
            trajectory_config = {}
        return int(trajectory_config.get("horizon_slots", 3))

    def _prediction_mapping_from_positions(
        self,
        positions: np.ndarray,
        origin_slot: int,
    ) -> dict[str, dict[str, Any]]:
        """把离线/Kalman的当前预测位置包装成统一的车辆预测接口。"""

        array = np.asarray(positions, dtype=float)
        if array.shape != (len(self._vehicle_ids), 2) or not np.all(
            np.isfinite(array)
        ):
            raise ValueError("预测车辆位置必须是有限的(V,2)数组。")
        horizon = self._prediction_horizon()
        return {
            vehicle_id: {
                "predicted_positions_m": np.repeat(
                    array[index][np.newaxis, :], horizon, axis=0
                ),
                "prediction_origin_slot": int(origin_slot),
                "future_slots": tuple(
                    int(origin_slot) + offset + 1 for offset in range(horizon)
                ),
            }
            for index, vehicle_id in enumerate(self._vehicle_ids)
        }

    def _vehicle_future_positions(
        self,
        vehicle_id: str,
        horizon: int,
    ) -> np.ndarray | None:
        """读取道路预测；车辆刚离开预测集时仅回退到最后已知位置。"""

        raw: Any = self._road_predictions.get(str(vehicle_id))
        if isinstance(raw, Mapping):
            raw = raw.get("predicted_positions_m")
        if raw is not None:
            try:
                array = np.asarray(raw, dtype=float)
            except (TypeError, ValueError):
                array = np.empty((0, 2), dtype=float)
            if array.shape == (2,):
                array = np.repeat(array[np.newaxis, :], horizon, axis=0)
            elif array.ndim == 2 and array.shape[1:] == (2,) and array.shape[0] > 0:
                if array.shape[0] < horizon:
                    array = np.vstack(
                        (array, np.repeat(array[-1][np.newaxis, :], horizon - array.shape[0], axis=0))
                    )
                else:
                    array = array[:horizon]
            else:
                array = np.empty((0, 2), dtype=float)
            if array.shape == (horizon, 2) and np.all(np.isfinite(array)):
                return array.copy()

        state = self._last_actual_states.get(str(vehicle_id))
        if isinstance(state, Mapping):
            position = np.asarray(state.get("position_m"), dtype=float)
            if position.shape == (2,) and np.all(np.isfinite(position)):
                return np.repeat(position[np.newaxis, :], horizon, axis=0)
        if str(vehicle_id) in self._vehicle_ids and self._predicted_positions.shape == (
            len(self._vehicle_ids),
            2,
        ):
            index = self._vehicle_ids.index(str(vehicle_id))
            position = np.asarray(self._predicted_positions[index], dtype=float)
            if np.all(np.isfinite(position)):
                return np.repeat(position[np.newaxis, :], horizon, axis=0)
        return None

    def _cache_prediction_nodes(self, horizon: int) -> list[dict[str, Any]]:
        """生成 SV/RSU/UAV 三类可缓存节点的统一位置接口。"""

        roles = _section(self.config, "vehicle_roles")
        radius = float(roles.get("service_v2v_range_m", 300.0))
        if not math.isfinite(radius) or radius < 0.0:
            raise ValueError("vehicle_roles.service_v2v_range_m必须是有限非负数。")
        nodes: list[dict[str, Any]] = []
        for vehicle_id in self._service_vehicle_ids:
            positions = self._vehicle_future_positions(vehicle_id, horizon)
            if positions is None:
                raise RuntimeError(f"服务车辆{vehicle_id}缺少当前位置，无法生成覆盖预测。")
            nodes.append(
                {
                    "node_id": f"service_vehicle:{vehicle_id}",
                    "node_type": "service_vehicle",
                    "positions_m": positions,
                    "coverage_radius_m": radius,
                }
            )
        rsu_positions = np.asarray(_section(self.config, "rsu")["positions_m"], dtype=float)
        for index, position in enumerate(rsu_positions):
            nodes.append(
                {
                    "node_id": f"rsu:{index}",
                    "node_type": "rsu",
                    "positions_m": np.repeat(position[np.newaxis, :], horizon, axis=0),
                    "coverage_radius_m": radius,
                }
            )
        for index, position in enumerate(np.asarray(self._uav_positions, dtype=float)):
            nodes.append(
                {
                    "node_id": f"uav:{index}",
                    "node_type": "uav",
                    "positions_m": np.repeat(position[np.newaxis, :], horizon, axis=0),
                    "coverage_radius_m": radius,
                }
            )
        return nodes

    def _observe_current_requests(self, slot_index: int) -> None:
        """在当前 slot 的预测和 solver 完成后登记请求，保持严格因果。"""

        if self._mobility_popularity_predictor is None or self._data is None:
            return
        workload = self._data["workload"]
        self._mobility_popularity_predictor.update_requests(
            int(slot_index),
            tuple(str(item) for item in workload["task_vehicle_ids"]),
            np.asarray(workload["task_arrivals"], dtype=bool)[int(slot_index)],
            np.asarray(workload["task_service_types"], dtype=np.int64)[int(slot_index)],
        )

    def _velocity_matrix_from_states(
        self,
        states: Mapping[str, Mapping[str, Any]],
    ) -> np.ndarray:
        velocities = np.zeros((len(self._vehicle_ids), 2), dtype=float)
        for index, vehicle_id in enumerate(self._vehicle_ids):
            speed_x = float(states[vehicle_id].get("speed_x_mps", math.nan))
            speed_y = float(states[vehicle_id].get("speed_y_mps", math.nan))
            if math.isfinite(speed_x) and math.isfinite(speed_y):
                velocities[index] = (speed_x, speed_y)
                continue
            speed = float(states[vehicle_id].get("speed_mps", 0.0))
            angle = float(states[vehicle_id].get("angle_deg", 0.0))
            if not math.isfinite(speed) or speed < 0.0:
                raise ValueError(f"车辆{vehicle_id}的速度无效。")
            if not math.isfinite(angle):
                angle = 0.0
            angle_rad = math.radians(angle)
            # SUMO 导航角：0°朝北、90°朝东；坐标轴为 +x 东、+y 北。
            velocities[index] = (speed * math.sin(angle_rad), speed * math.cos(angle_rad))
        return velocities

    def _trajectory_positions(self, slot_index: int) -> np.ndarray:
        if self._data is None or self._data.get("trajectory") is None:
            raise RuntimeError("当前数据源没有轨迹数组。")
        trajectory = self._data["trajectory"]
        trajectory_ids = tuple(str(item) for item in trajectory["vehicle_ids"])
        columns = {vehicle_id: index for index, vehicle_id in enumerate(trajectory_ids)}
        missing = [vehicle_id for vehicle_id in self._vehicle_ids if vehicle_id not in columns]
        if missing:
            raise ValueError(f"轨迹缺少车辆：{missing}")
        raw = np.asarray(trajectory["positions"], dtype=float)
        positions = raw[int(slot_index), [columns[item] for item in self._vehicle_ids]]
        if positions.shape != (len(self._vehicle_ids), 2) or not np.all(
            np.isfinite(positions)
        ):
            raise ValueError("轨迹车辆位置必须是有限的(V,2)数组。")
        return positions.copy()

    def _trajectory_velocities(self, slot_index: int) -> np.ndarray:
        if self._data is None or self._data.get("trajectory") is None:
            raise RuntimeError("当前数据源没有轨迹数组。")
        trajectory = self._data["trajectory"]
        trajectory_ids = tuple(str(item) for item in trajectory["vehicle_ids"])
        columns = {vehicle_id: index for index, vehicle_id in enumerate(trajectory_ids)}
        speeds = np.asarray(trajectory["speeds"], dtype=float)[
            int(slot_index), [columns[item] for item in self._vehicle_ids]
        ]
        angles = trajectory.get("angles")
        if angles is None:
            return np.zeros((len(self._vehicle_ids), 2), dtype=float)
        angle_values = np.asarray(angles, dtype=float)[
            int(slot_index), [columns[item] for item in self._vehicle_ids]
        ]
        if not np.all(np.isfinite(speeds)) or np.any(speeds < 0.0):
            raise ValueError("轨迹速度必须是有限非负数。")
        angle_values = np.where(np.isfinite(angle_values), angle_values, 0.0)
        radians = np.deg2rad(angle_values)
        return np.stack((speeds * np.sin(radians), speeds * np.cos(radians)), axis=1)

    def _predicted_trajectory_positions(self, slot_index: int) -> np.ndarray:
        if self._data is None:
            raise RuntimeError("环境尚未reset。")
        predicted = self._data.get("predicted_positions")
        if predicted is None:
            raise RuntimeError("当前数据源没有预先计算的预测位置。")
        trajectory = self._data.get("trajectory")
        if trajectory is None:
            raise RuntimeError("没有轨迹时不能读取轨迹预测位置。")
        trajectory_ids = tuple(str(item) for item in trajectory["vehicle_ids"])
        columns = {vehicle_id: index for index, vehicle_id in enumerate(trajectory_ids)}
        raw = np.asarray(predicted, dtype=float)
        positions = raw[int(slot_index), [columns[item] for item in self._vehicle_ids]]
        if positions.shape != (len(self._vehicle_ids), 2) or not np.all(
            np.isfinite(positions)
        ):
            raise ValueError("预测车辆位置必须是有限的(V,2)数组。")
        return positions.copy()

    def _prediction_popularity(self, slot_index: int) -> np.ndarray:
        """根据历史偏好和未来覆盖生成节点级结果，并返回兼容的全局 F 向量。"""

        if self._data is None or self._mobility_popularity_predictor is None:
            raise RuntimeError("移动感知流行度预测器尚未初始化。")
        horizon = self._prediction_horizon()
        nodes = self._cache_prediction_nodes(horizon)
        workload = self._data["workload"]
        task_vehicle_ids = tuple(str(item) for item in workload["task_vehicle_ids"])
        result = self._mobility_popularity_predictor.predict_node_demand(
            origin_slot=int(slot_index),
            task_vehicle_ids=task_vehicle_ids,
            vehicle_predictions=self._road_predictions,
            nodes=nodes,
            coverage_radius_m=float(
                _section(self.config, "vehicle_roles").get(
                    "service_v2v_range_m", 300.0
                )
            ),
            horizon_slots=horizon,
        )
        self._last_node_prediction = result
        node_popularity = np.asarray(result["predicted_popularity"], dtype=float)
        if node_popularity.ndim != 2 or node_popularity.shape[1] != self._service_sizes.size:
            raise ValueError("节点级预测流行度维度与服务数量不一致。")
        if node_popularity.shape[0] == 0:
            prediction = self._mobility_popularity_predictor.global_preference
        else:
            # 保持现有 observation 的 F 维接口；完整 M×F 结果保存在
            # _last_node_prediction，供后续节点级缓存决策直接复用。
            prediction = np.mean(node_popularity, axis=0)
        if not np.all(np.isfinite(prediction)) or np.any(prediction < 0.0):
            raise ValueError("预测服务流行度必须是有限非负向量。")
        total = float(prediction.sum())
        if total <= 0.0:
            raise ValueError("预测服务流行度总和必须为正。")
        normalized = prediction / total
        self._prediction_origin_slot = int(slot_index)
        return normalized

    @property
    def last_node_prediction(self) -> dict[str, Any]:
        """返回最近一次 M×F 节点级需求/流行度结果的只读副本。"""

        result: dict[str, Any] = {}
        for key, value in self._last_node_prediction.items():
            if isinstance(value, np.ndarray):
                result[key] = value.copy()
            elif isinstance(value, list):
                result[key] = [dict(item) if isinstance(item, Mapping) else item for item in value]
            else:
                result[key] = value
        return result

    @property
    def cache_node_ids(self) -> tuple[str, ...]:
        """返回缓存行的显式稳定布局；SV 身份替换后只更新对应 ID。"""

        rsu_count = len(np.asarray(_section(self.config, "rsu")["positions_m"]))
        return tuple(
            [f"service_vehicle:{item}" for item in self._service_vehicle_ids]
            + [f"rsu:{index}" for index in range(rsu_count)]
            + [f"uav:{index}" for index in range(len(self._uav_positions))]
        )

    @property
    def uav_ids(self) -> tuple[str, ...]:
        return tuple(f"uav:{index}" for index in range(len(self._uav_positions)))

    @property
    def action_layout(self) -> dict[str, Any]:
        """显式给出 MultiBinary 索引，禁止依赖字典迭代顺序。"""

        cache_ids = self.cache_node_ids
        uav_ids = self.uav_ids
        return {
            "cache_node_ids": cache_ids,
            "cache_node_types": tuple(item.value for item in self._service_node_types),
            "uav_ids": uav_ids,
            "cache_trigger_indices": tuple(range(len(cache_ids))),
            "uav_position_trigger_indices": tuple(
                range(len(cache_ids), len(cache_ids) + len(uav_ids))
            ),
            "action_length": len(cache_ids) + len(uav_ids),
        }

    def configuration_observation(self) -> np.ndarray:
        """构造严格因果的配置 observation，并按区域归一化到 [0,1]。

        这里只读取 C(t-1)、q_U(t-1)、截至 t-1 历史产生的预测位置/需求和
        t/T；当前真实请求和当前真实车辆位置不进入 PPO 配置决策。
        """

        state = self._configuration_state()
        region = _section(self.config, "uav")["region"]
        x_min = float(region["x_min_m"])
        x_max = float(region["x_max_m"])
        y_min = float(region["y_min_m"])
        y_max = float(region["y_max_m"])
        spans = np.asarray([x_max - x_min, y_max - y_min], dtype=float)
        if np.any(spans <= 0.0):
            raise ValueError("UAV region 必须具有正的二维跨度。")

        def normalize_positions(values: np.ndarray) -> np.ndarray:
            normalized = (
                np.asarray(values, dtype=float) - np.asarray([x_min, y_min])
            ) / spans
            return np.clip(normalized, 0.0, 1.0)

        observation = np.concatenate(
            (
                np.asarray(state["previous_cache"], dtype=float).reshape(-1),
                normalize_positions(state["previous_uav_positions_m"]).reshape(-1),
                normalize_positions(state["predicted_vehicle_positions_m"]).reshape(-1),
                np.asarray(state["predicted_popularity"], dtype=float).reshape(-1),
                np.asarray(state["normalized_time"], dtype=float).reshape(-1),
            )
        ).astype(np.float32)
        if not np.all(np.isfinite(observation)) or np.any(observation < 0.0) or np.any(
            observation > 1.0
        ):
            raise ValueError("PPO configuration observation 必须是有限的 [0,1] 向量。")
        return observation

    def _actual_motion(self, slot_index: int) -> tuple[np.ndarray, np.ndarray]:
        """读取同一时隙的实际位置与速度，供调度和下一时隙预测校正使用。"""

        if self._data is None:
            raise RuntimeError("环境尚未reset。")
        if self._mobility_provider is not None:
            states = self._refresh_vehicle_roles(slot_index)
            self._last_actual_states = dict(states)
            return (
                self._position_matrix(states, self._vehicle_ids),
                self._velocity_matrix_from_states(states),
            )
        if self._data.get("trajectory") is not None:
            states = self._refresh_vehicle_roles(slot_index)
            if states:
                self._last_actual_states = dict(states)
                return (
                    self._position_matrix(states, self._vehicle_ids),
                    self._velocity_matrix_from_states(states),
                )
            return self._trajectory_positions(slot_index), self._trajectory_velocities(slot_index)
        if self._sumo_api is None:
            raise RuntimeError("当前环境没有可读取实际车辆状态的SUMO连接。")
        try:
            from .data import step as step_sumo
        except ImportError:  # pragma: no cover - direct script import
            from data import step as step_sumo
        target_time = float(self._times[int(slot_index)])
        current_time = float(self._sumo_api.simulation.getTime())
        if current_time < target_time - 1.0e-9:
            step_sumo(self._sumo_api, target_time)
        elif current_time > target_time + 1.0e-9:
            raise RuntimeError(
                f"SUMO时间{current_time}已经越过当前时隙{target_time}，禁止使用未来观测。"
            )
        states = self._refresh_vehicle_roles(slot_index)
        self._last_actual_states = dict(states)
        return (
            self._position_matrix(states, self._vehicle_ids),
            self._velocity_matrix_from_states(states),
        )

    def _actual_positions(self, slot_index: int) -> np.ndarray:
        """兼容只需实际位置的只读调用。"""

        positions, _ = self._actual_motion(slot_index)
        return positions

    def _configuration_state(self) -> dict[str, np.ndarray]:
        """复制配置状态和有限回合时间，阻止调用方修改内部状态。"""

        if self._slot_index is None or self._times.size < 2:
            raise RuntimeError("环境时隙尚未初始化，无法构造normalized_time。")

        state = {
            "previous_cache": np.asarray(self._cache, dtype=float).copy(),
            "previous_uav_positions_m": np.asarray(self._uav_positions, dtype=float).copy(),
            "predicted_vehicle_positions_m": np.asarray(
                self._predicted_positions, dtype=float
            ).copy(),
            "predicted_popularity": np.asarray(self._predicted_popularity, dtype=float).copy(),
            "normalized_time": np.asarray(
                [_normalized_decision_time(self._slot_index, self._times.size)],
                dtype=float,
            ),
        }
        for value in state.values():
            if not np.all(np.isfinite(value)):
                raise ValueError("configuration state包含NaN或Inf。")
        return state

    def _parse_action(
        self,
        action: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """校验节点级 MultiBinary trigger；PPO 不直接生成配置内容。"""

        if not isinstance(action, Mapping):
            raise TypeError("step(action)中的action必须是映射。")
        cache_trigger_value = action.get("cache_triggers", action.get("cache_trigger"))
        uav_trigger_value = action.get("uav_triggers", action.get("uav_trigger"))
        if cache_trigger_value is None or uav_trigger_value is None:
            raise ValueError("action必须包含cache_triggers和uav_triggers。")
        cache_triggers = _binary_vector(
            cache_trigger_value, "action.cache_triggers"
        ).astype(bool)
        uav_triggers = _binary_vector(uav_trigger_value, "action.uav_triggers").astype(bool)
        node_count = self._cache.shape[0]
        if cache_triggers.shape != (node_count,):
            raise ValueError("action.cache_triggers形状必须是(M,)。")
        if uav_triggers.shape != (len(self._uav_service_indices),):
            raise ValueError("action.uav_triggers形状必须是(U,)。")
        return cache_triggers, uav_triggers

    def _resolve_kkt_epsilons(
        self,
        task_count: int,
        epsilon_compute: Real | None,
        epsilon_bandwidth: Real | None,
    ) -> tuple[float, float]:
        """解析论文 C13/C15 的显式资源比例下界。

        正式路径的数值必须来自 ``resource_allocation`` 配置节，不再根据
        task_count生成工程默认值。``step``参数仍保留用于兼容原有显式测试
        接口：调用方传入的值优先于构造器覆盖值，再优先于正式配置值。
        """

        # task_count保留在签名中是为了兼容现有调用和测试接口；正式 epsilon
        # 不再由它计算，避免把论文中的固定下界错误变成任务数自适应变量。
        del task_count
        compute = self._optional_epsilon(epsilon_compute, "epsilon_compute")
        bandwidth = self._optional_epsilon(epsilon_bandwidth, "epsilon_bandwidth")
        if compute is None:
            compute = self._epsilon_compute
        if bandwidth is None:
            bandwidth = self._epsilon_bandwidth

        resource_config = self.config.get("resource_allocation")
        if not isinstance(resource_config, Mapping):
            resource_config = {}

        if compute is None and "epsilon_compute" in resource_config:
            compute = self._optional_epsilon(
                resource_config["epsilon_compute"], "resource_allocation.epsilon_compute"
            )
        if bandwidth is None and "epsilon_bandwidth" in resource_config:
            bandwidth = self._optional_epsilon(
                resource_config["epsilon_bandwidth"],
                "resource_allocation.epsilon_bandwidth",
            )

        if compute is None:
            raise ValueError(
                "正式KKT路径缺少resource_allocation.epsilon_compute；"
                "禁止根据task_count生成默认值。"
            )
        if bandwidth is None:
            raise ValueError(
                "正式KKT路径缺少resource_allocation.epsilon_bandwidth；"
                "禁止根据task_count生成默认值。"
            )
        return float(compute), float(bandwidth)

    def reset(
        self,
        *,
        data_source: Mapping[str, Any] | None = None,
        sumo_configuration_file: str | os.PathLike[str] | None = None,
        seed: int | None = None,
        initial_cache: Sequence[Sequence[Real]] | np.ndarray | None = None,
        initial_local_cache: Sequence[Sequence[Real]] | np.ndarray | None = None,
        initial_uav_positions: Sequence[Sequence[Real]] | np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """初始化论文配置状态并返回 ``s_cfg(t)``。

        无轨迹数组时必须显式提供正式SUMO ``.sumocfg``；这不会复制或修改
        旧路网文件。返回状态只包含论文第5节规定的四组量。
        ``reset`` 会先关闭本环境自己拥有的旧 SUMO 连接，再加载 workload，建立
        服务/任务车辆顺序、随机可行缓存、UAV 初始位置和 Kalman 状态。返回值只
        包含 policy 需要的四个配置状态，不把当前任务、reward 或 solver 结果偷偷
        混入 observation。
        """

        # reset 先释放本环境拥有的旧 SUMO 连接，保证重复 reset 不会遗留后台进程。
        self.close()
        self._predictor = None
        self._road_predictor = None
        self._mobility_popularity_predictor = None
        self._road_predictions = {}
        self._last_node_prediction = {}
        self._service_vehicle_replacement_rows = np.empty(0, dtype=bool)
        self._last_actual_states = {}
        self._prediction_origin_slot = None
        self._data = self._load_rollout_data(data_source)
        data_config = self._data.get("config")
        if isinstance(data_config, Mapping):
            self.config = dict(data_config)
        if sumo_configuration_file is not None:
            self._sumo_configuration_file = str(sumo_configuration_file)

        try:
            from .data import (
                get_slot_data,
                get_vehicle_states,
                VehicleRoleManager,
                resolve_vehicle_roles,
                start_sumo,
                step as step_sumo,
            )
        except ImportError:  # pragma: no cover - direct script import
            from data import (
                get_slot_data,
                get_vehicle_states,
                VehicleRoleManager,
                resolve_vehicle_roles,
                start_sumo,
                step as step_sumo,
            )

        workload = self._data["workload"]
        self._times = np.asarray(workload["times"], dtype=float).copy()
        if self._times.ndim != 1 or self._times.size == 0 or not np.all(np.isfinite(self._times)):
            raise ValueError("workload.times必须是非空有限向量。")
        if self._times.size > 1 and np.any(np.diff(self._times) <= 0.0):
            raise ValueError("workload.times必须严格递增。")
        simulation = _section(self.config, "simulation")
        interval_s = _positive(simulation["decision_interval_s"], "simulation.decision_interval_s")
        mobility_source = self._configured_mobility_source()
        self._mobility_source = mobility_source
        # 工作负载维度由完整 SUMO 车辆全集和稳定优先级确定；实际角色在
        # 当前状态读取后由 VehicleRoleManager 更新，车辆 ID 不依赖数组下标。
        self._service_vehicle_ids, resolved_task_ids = resolve_vehicle_roles(self.config)
        self._task_vehicle_ids = tuple(str(item) for item in workload["task_vehicle_ids"])
        if tuple(self._task_vehicle_ids) != tuple(resolved_task_ids):
            raise ValueError(
                "workload.task_vehicle_ids必须与稳定优先级确定的任务车辆集合一致。"
            )
        self._vehicle_ids = self._service_vehicle_ids + self._task_vehicle_ids
        if not self._vehicle_ids or len(set(self._vehicle_ids)) != len(self._vehicle_ids):
            raise ValueError("服务车辆和任务车辆ID必须非空且互不重复。")
        # 服务目录是 F 维固定权重；后续 local/edge cache 都必须使用同一数组。
        self._service_sizes = self.service_sizes_mb

        rsu_positions = np.asarray(_section(self.config, "rsu")["positions_m"], dtype=float)
        if rsu_positions.ndim != 2 or rsu_positions.shape[1:] != (2,) or not np.all(
            np.isfinite(rsu_positions)
        ):
            raise ValueError("RSU位置必须是有限的(R,2)数组。")
        uav_config = _section(self.config, "uav")
        default_uav_positions = np.asarray(uav_config["initial_positions_m"], dtype=float)
        if default_uav_positions.ndim != 2 or default_uav_positions.shape[1:] != (2,) or not np.all(
            np.isfinite(default_uav_positions)
        ):
            raise ValueError("UAV初始位置必须是有限的(U,2)数组。")
        self._uav_positions = (
            default_uav_positions.copy()
            if initial_uav_positions is None
            else np.asarray(initial_uav_positions, dtype=float).copy()
        )
        if self._uav_positions.shape != default_uav_positions.shape or not np.all(
            np.isfinite(self._uav_positions)
        ):
            raise ValueError("initial_uav_positions必须与配置中的(U,2)形状一致。")
        region = _section(uav_config, "region")
        x_min, x_max = float(region["x_min_m"]), float(region["x_max_m"])
        y_min, y_max = float(region["y_min_m"]), float(region["y_max_m"])
        if np.any(self._uav_positions[:, 0] < x_min) or np.any(self._uav_positions[:, 0] > x_max):
            raise ValueError("UAV初始位置超出区域边界。")
        if np.any(self._uav_positions[:, 1] < y_min) or np.any(self._uav_positions[:, 1] > y_max):
            raise ValueError("UAV初始位置超出区域边界。")
        minimum_separation = float(uav_config["minimum_safety_distance_m"])
        for first in range(self._uav_positions.shape[0]):
            for second in range(first + 1, self._uav_positions.shape[0]):
                if np.linalg.norm(self._uav_positions[first] - self._uav_positions[second]) < (
                    minimum_separation - 1.0e-9
                ):
                    raise ValueError("UAV初始位置违反最小安全距离。")

        service_node_count = (
            len(self._service_vehicle_ids) + len(rsu_positions) + len(self._uav_positions)
        )
        # edge node 的列布局固定为：服务车辆、RSU、UAV；solver 的 service_columns
        # 和后续 execution_nodes 都依赖这个顺序，不能在每个时隙重新排序。
        self._service_node_types = (
            (NodeType.SERVICE_VEHICLE,) * len(self._service_vehicle_ids)
            + (NodeType.RSU,) * len(rsu_positions)
            + (NodeType.UAV,) * len(self._uav_positions)
        )
        self._uav_service_indices = tuple(
            range(len(self._service_vehicle_ids) + len(rsu_positions), service_node_count)
        )
        cache_config = _section(self.config, "cache")
        capacity_mb = _section(cache_config, "capacity_mb")
        capacities_by_kind = {
            NodeType.SERVICE_VEHICLE: float(capacity_mb["sv"]),
            NodeType.RSU: float(capacity_mb["rsu"]),
            NodeType.UAV: float(capacity_mb["uav"]),
        }
        self._cache_capacities = np.asarray(
            [capacities_by_kind[kind] for kind in self._service_node_types], dtype=float
        )
        if not np.all(np.isfinite(self._cache_capacities)) or np.any(self._cache_capacities < 0.0):
            raise ValueError("缓存容量必须是有限非负数。")

        random_seed = int(
            _section(self.config, "project").get("seed", 0) if seed is None else seed
        )
        self._role_manager = VehicleRoleManager(self.config, seed=random_seed)
        edge_rng = np.random.default_rng(random_seed + 1)
        local_rng = np.random.default_rng(random_seed + 2)
        # 没有显式初始缓存时，为每种 edge node 独立生成一个容量内的随机可行解。
        # 该随机初始化只影响缓存状态，不改变服务大小或任务请求随机流。
        if initial_cache is None:
            self._cache = self._random_feasible_cache(
                self._service_sizes, self._cache_capacities, edge_rng
            )
        else:
            self._cache = self._binary_matrix(initial_cache, "initial_cache")
        if self._cache.shape != (service_node_count, self._service_sizes.size):
            raise ValueError("initial_cache形状必须是(M,F)。")
        cache_usage = np.sum(
            self._cache * self._service_sizes[np.newaxis, :], axis=1
        )
        if np.any(cache_usage > self._cache_capacities + 1.0e-9):
            raise ValueError("initial_cache违反缓存容量约束。")
        task_capacity = float(capacity_mb["tv"])
        task_count = len(self._task_vehicle_ids)
        if initial_local_cache is None:
            self._local_cache = self._random_feasible_cache(
                self._service_sizes,
                np.full(task_count, task_capacity, dtype=float),
                local_rng,
            )
        else:
            self._local_cache = self._binary_matrix(
                initial_local_cache, "initial_local_cache"
            )
        if self._local_cache.shape != (task_count, self._service_sizes.size):
            raise ValueError("initial_local_cache形状必须是(V,F)。")
        local_usage = np.sum(
            self._local_cache * self._service_sizes[np.newaxis, :], axis=1
        )
        if np.any(local_usage > task_capacity + 1.0e-9):
            raise ValueError("initial_local_cache违反任务车辆本地缓存容量约束。")

        self._mobility_popularity_predictor = self._new_mobility_popularity_predictor()
        trajectory = self._data.get("trajectory")
        if trajectory is not None:
            self._road_predictor = None
            initial_all_states = self._trajectory_states(0)
            initial_role_state = self._role_manager.assign(
                initial_all_states, slot=0
            )
            self._apply_role_state(initial_role_state, initial=True)
            self._last_actual_states = dict(initial_all_states)
            # 离线轨迹优先使用已提供的预测；没有预测数组时才用 Kalman 在线递推。
            self._prediction_mode = "provided" if self._data.get("predicted_positions") is not None else "kalman"
            if self._prediction_mode == "provided":
                self._predicted_positions = self._predicted_trajectory_positions(0)
                self._predictor = None
            else:
                self._predictor = self._new_predictor(interval_s)
                initial_positions = self._trajectory_positions(0)
                self._predictor.initialize(
                    initial_positions, self._trajectory_velocities(0)
                )
                self._predicted_positions = initial_positions.copy()
            self._road_predictions = self._prediction_mapping_from_positions(
                self._predicted_positions, origin_slot=0
            )
        elif mobility_source in {"sumo", "trace"}:
            # 没有离线轨迹时，SUMO 是唯一的真实位置来源。这里启动项目内正式
            # .sumocfg，并把仿真推进到第一个决策时刻之前的观测点。
            first_time = float(self._times[0])
            begin_time = float(simulation.get("begin_s", 0.0))
            previous_time = max(begin_time, first_time - interval_s)
            if mobility_source == "trace":
                self._mobility_provider = TraceMobilityProvider(
                    self.config,
                    seed=random_seed,
                )
            else:
                self._mobility_provider = SumoMobilityProvider(
                    self.config,
                    seed=random_seed,
                    configuration_file=self._sumo_configuration_file,
                )
            all_initial_states = self._mobility_provider.reset(
                self._times,
                begin_s=begin_time,
            )
            initial_role_state = self._role_manager.assign(all_initial_states, slot=0)
            self._apply_role_state(initial_role_state, initial=True)
            self._last_actual_states = dict(all_initial_states)
            initial_states = {
                vehicle_id: all_initial_states[vehicle_id]
                for vehicle_id in self._vehicle_ids
            }
            initial_positions = self._position_matrix(initial_states, self._vehicle_ids)
            self._road_predictor = self._new_road_predictor(interval_s)
            # 先登记当前时刻的全体车辆，再只为当前角色输出位置矩阵。
            self._road_predictor.observe(all_initial_states, slot=0)
            initial_predictions = self._road_predictor.predict(
                initial_states,
                origin_slot=0,
                update_history=False,
            )
            self._road_predictions = dict(initial_predictions)
            self._predictor = None
            self._prediction_mode = "road_constrained"
            self._predicted_positions = (
                initial_positions.copy()
                if math.isclose(previous_time, first_time, rel_tol=0.0, abs_tol=1.0e-9)
                else self._road_predictor.positions_matrix(
                    initial_predictions, self._vehicle_ids
                )
            )

        self._slot_index = 0
        self._predicted_popularity = self._prediction_popularity(0)
        return self._configuration_state()

    def step(
        self,
        action: Mapping[str, Any],
        *,
        epsilon_compute: Real | None = None,
        epsilon_bandwidth: Real | None = None,
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        """执行一个基础时隙并返回 ``next_state, reward, done, info``。

        单个时隙的实际调用顺序固定为：

        1. 读取当前 SUMO 车辆位置和 workload；
        2. 调用 ``solver.solve_slot`` 完成缓存、UAV、可行节点、matching、KKT；
        3. 汇总五类实际执行节点的 delay/energy/count；
        4. 写回最终 cache/UAV 状态并推进 SUMO 到下一决策时刻。

        ``info`` 用于诊断和 rollout 验收，policy 的 observation 只从下一状态中
        读取配置量，避免把最终执行结果反向泄漏到当前动作。
        """

        if self._slot_index is None:
            raise RuntimeError("必须先调用reset()。")
        if self._slot_index >= self._times.size:
            raise RuntimeError("episode已经结束，请先调用reset()。")
        current_slot = int(self._slot_index)
        mode = self.cache_controller_mode
        try:
            from .data import get_slot_data, step as step_sumo
            from .solver import solve_slot
        except ImportError:  # pragma: no cover - direct script import
            from data import get_slot_data, step as step_sumo
            from solver import solve_slot

        # 先校验/转换 policy 的原始动作，再读取当前 slot 的真实车辆位置。
        cache_triggers, uav_triggers = self._parse_action(action)
        slot = get_slot_data(self._data, current_slot) if self._data is not None else None
        if slot is None:
            raise RuntimeError("环境数据尚未初始化。")
        actual_positions, actual_velocities = self._actual_motion(current_slot)
        # 服务车辆身份变化后，上一时隙的节点级预测行可能仍对应旧 SV。
        # 在当前真实请求登记前，仅按新的节点身份/当前位置重建一次预测，
        # 保证新 SV 的空缓存与协同候选使用同一套节点顺序和因果输入。
        expected_prediction_node_ids = tuple(
            [f"service_vehicle:{vehicle_id}" for vehicle_id in self._service_vehicle_ids]
            + [
                f"rsu:{index}"
                for index in range(
                    len(np.asarray(_section(self.config, "rsu")["positions_m"], dtype=float))
                )
            ]
            + [f"uav:{index}" for index in range(len(self._uav_positions))]
        )
        if tuple(self._last_node_prediction.get("node_ids", ())) != expected_prediction_node_ids:
            self._predicted_popularity = self._prediction_popularity(current_slot)
        service_count = len(self._service_vehicle_ids)
        task_positions = actual_positions[service_count:].copy()
        service_vehicle_positions = actual_positions[:service_count].copy()
        rsu_positions = np.asarray(_section(self.config, "rsu")["positions_m"], dtype=float)
        edge_positions = np.vstack((service_vehicle_positions, rsu_positions, self._uav_positions))
        epsilon_c, epsilon_b = self._resolve_kkt_epsilons(
            len(self._task_vehicle_ids), epsilon_compute, epsilon_bandwidth
        )
        previous_cache_state = np.asarray(self._cache, dtype=float).copy()
        previous_uav_positions = np.asarray(self._uav_positions, dtype=float).copy()
        # solver 返回的是完整确定性结果：最终缓存、最终 UAV 位置、matching、KKT
        # 分配和每个任务的实际执行结果。环境不在这里重写任何求解公式。
        result = solve_slot(
            previous_cache=self._cache.copy(),
            cache_triggers=cache_triggers.copy(),
            service_sizes=self._service_sizes.copy(),
            cache_capacities=self._cache_capacities.copy(),
            previous_uav_positions=previous_uav_positions.copy(),
            uav_triggers=uav_triggers.copy(),
            task_arrivals=np.asarray(slot["task_arrivals"], dtype=bool),
            task_service_types=np.asarray(slot["task_service_types"], dtype=int),
            task_vehicle_cache=self._local_cache.copy(),
            task_input_bits=np.asarray(slot["task_input_size_bits"], dtype=float),
            task_cpu_cycles=np.asarray(slot["task_cpu_cycles"], dtype=float),
            task_vehicle_positions=task_positions,
            service_node_positions=edge_positions,
            service_node_types=self._service_node_types,
            uav_service_indices=self._uav_service_indices,
            # 该预测在当前真实请求登记前已经生成；solver 只读取节点级预测，
            # 不读取当前 task_arrivals/task_service_types 来决定缓存更新。
            cache_prediction=self._last_node_prediction,
            environment=self,
            integer_scale=1,
            decision_interval_s=float(_section(self.config, "simulation")["decision_interval_s"]),
            epsilon_compute=epsilon_c,
            epsilon_bandwidth=epsilon_b,
        )

        # 只对已到达任务统计执行节点；未到达任务必须保持零时延、零能耗、无分配。
        arrivals = np.asarray(slot["task_arrivals"], dtype=bool)
        execution_counts = {
            "local": 0,
            "service_vehicle": 0,
            "rsu": 0,
            "uav": 0,
            "cloud": 0,
        }
        total_delay = 0.0
        total_energy = 0.0
        completed = 0
        failed = 0
        count_key = {
            NodeType.LOCAL.value: "local",
            NodeType.SERVICE_VEHICLE.value: "service_vehicle",
            NodeType.RSU.value: "rsu",
            NodeType.UAV.value: "uav",
            NodeType.CLOUD.value: "cloud",
        }
        for task_index, arrived in enumerate(arrivals):
            if not arrived:
                continue
            node = result.execution_nodes[task_index]
            key = count_key.get(node) if node is not None else None
            if key is None:
                failed += 1
                continue
            execution_counts[key] += 1
            completed += 1
            task_result = result.task_results[task_index]
            total_delay += float(task_result["completion_delay_s"])
            total_energy += float(task_result["vehicle_energy_j"])
        execution_rates = {
            key: (float(value) / completed if completed else 0.0)
            for key, value in execution_counts.items()
        }
        # reward 使用 solver 基于实际执行结果计算的 G，而不是 policy 原始动作的
        # surrogate 分数。这样 rollout 统计与最终论文 metrics 使用同一语义。
        reward = float(result.system_utility["reward"])
        if not math.isfinite(reward) or not math.isfinite(total_delay) or not math.isfinite(total_energy):
            raise ValueError("step结果包含NaN或Inf。")

        # 只有当前时隙完成后才提交最终配置，下一 observation 读取的是实际状态。
        self._cache = np.asarray(result.final_cache, dtype=float).copy()
        self._uav_positions = np.asarray(result.final_uav_positions, dtype=float).copy()
        cache_state_changed = np.any(
            np.asarray(result.final_cache, dtype=float) != previous_cache_state,
            axis=1,
        )
        # 当前请求必须在当前 slot 的 prediction/solver 完成后登记，避免 request(t)
        # 进入 prediction(t)。下一 slot 的预测才可以读取这次历史。
        self._observe_current_requests(current_slot)
        next_slot = current_slot + 1
        done = next_slot >= self._times.size
        if not done:
            if self._road_predictor is not None:
                selected_states = {
                    vehicle_id: self._last_actual_states[vehicle_id]
                    for vehicle_id in self._vehicle_ids
                    if vehicle_id in self._last_actual_states
                }
                predictions = self._road_predictor.predict(
                    selected_states,
                    origin_slot=current_slot,
                    update_history=False,
                )
                self._road_predictions = dict(predictions)
                self._predicted_positions = self._road_predictor.positions_matrix(
                    predictions, self._vehicle_ids
                )
            elif self._predictor is not None:
                # 当前 slot 完成后，才用该 slot 的位置和速度校正并预测下一 slot。
                self._predictor.correct(actual_positions, actual_velocities)
                self._predicted_positions = self._predictor.predict()
                self._road_predictions = self._prediction_mapping_from_positions(
                    self._predicted_positions, origin_slot=current_slot
                )
            elif self._prediction_mode == "provided":
                self._predicted_positions = self._predicted_trajectory_positions(next_slot)
                self._road_predictions = self._prediction_mapping_from_positions(
                    self._predicted_positions, origin_slot=current_slot
                )
            else:  # pragma: no cover - defensive state guard
                raise RuntimeError("预测器状态未初始化。")
            self._predicted_popularity = self._prediction_popularity(next_slot)
            self._slot_index = next_slot
            if self._sumo_api is not None:
                step_sumo(self._sumo_api, float(self._times[next_slot]))
        else:
            self._slot_index = next_slot

        candidate_changed = bool(
            np.any(np.asarray(result.candidate_cache, dtype=float) != previous_cache_state)
        )

        info: dict[str, Any] = {
            "slot_index": current_slot,
            "time_s": float(slot["time_s"]),
            "execution_counts": execution_counts,
            "execution_rates": execution_rates,
            "completed_tasks": int(completed),
            "failed_tasks": int(failed),
            "total_delay": float(total_delay),
            "total_energy": float(total_energy),
            "system_utility": dict(result.system_utility),
            "cache_check": bool(result.cache_check),
            # 第六阶段显式记录 slot 级检查、候选变化和最终实际变化。
            "cache_check_slots": int(bool(result.cache_check)),
            "cache_candidate_changed": np.asarray(
                result.cache_candidate_changed, dtype=bool
            ).copy(),
            "cache_trigger": np.asarray(result.cache_trigger, dtype=bool).copy(),
            "cache_changed": np.asarray(result.cache_changed, dtype=bool).copy(),
            "cache_state_before": previous_cache_state.copy(),
            "cache_state_changed": np.asarray(cache_state_changed, dtype=bool).copy(),
            "cache_controller_mode": mode,
            "action_layout": self.action_layout,
            "ppo_candidate_changed": candidate_changed,
            "cooperative_candidate_utility": float(
                result.cooperative_candidate_utility
            ),
            "cache_duplicate_count": int(result.cache_duplicate_count),
            "cache_unique_services": int(result.cache_unique_services),
            "services_changed": tuple(result.services_changed),
            "cache_update_count": int(np.sum(result.cache_changed)),
            "cache_candidate_changed_slots": int(
                bool(np.any(result.cache_candidate_changed))
            ),
            "cache_changed_slots": int(bool(np.any(result.cache_changed))),
            "cache_changed_nodes": int(np.sum(result.cache_changed)),
            "cache_changed_services": int(
                np.sum(
                    np.asarray(result.final_cache, dtype=bool)
                    != np.asarray(previous_cache_state, dtype=bool)
                )
            ),
            "uav_move_count": int(np.sum(result.actual_uav_movement)),
            "uav_check": bool(result.uav_check),
            "uav_check_slots": int(bool(result.uav_check)),
            "requested_cache_triggers": cache_triggers.copy(),
            "actual_cache_update": np.asarray(result.cache_changed, dtype=bool).copy(),
            "service_vehicle_replacement_rows": self._service_vehicle_replacement_rows.copy(),
            "service_vehicle_replacement_count": int(
                np.sum(self._service_vehicle_replacement_rows)
            ),
            "requested_uav_triggers": uav_triggers.copy(),
            "actual_uav_movement": np.asarray(result.actual_uav_movement, dtype=bool).copy(),
            "uav_moved_slots": int(bool(np.any(result.actual_uav_movement))),
            "uav_nodes_moved": int(np.sum(result.actual_uav_movement)),
            "uav_move_distance": float(
                np.linalg.norm(
                    np.asarray(result.final_uav_positions, dtype=float)
                    - np.asarray(previous_uav_positions, dtype=float),
                    axis=1,
                ).sum()
            ),
            "uav_candidate_count": int(result.uav_candidate_count),
            "uav_ground_service_gap": float(result.ground_service_gap),
            "cache_cost": float(result.system_utility["cache_update_cost"]),
            "uav_move_cost": float(result.system_utility["uav_update_cost"]),
            "assignment": np.asarray(result.assignment, dtype=float).copy(),
            "execution_nodes": tuple(result.execution_nodes),
            "matching_migration_count": int(result.matching_migration_count),
            "matching_swap_count": int(result.matching_swap_count),
            "kkt_failure_count": int(result.kkt_failure_count),
            "failure_reason_counts": dict(result.failure_reason_counts),
            "compute_allocations": np.asarray(result.compute_allocations, dtype=float).copy(),
            "bandwidth_allocations": np.asarray(
                result.bandwidth_allocations, dtype=float
            ).copy(),
            "num_task_vehicles": len(self._task_vehicle_ids),
            "num_service_vehicles": len(self._service_vehicle_ids),
            "task_join_count": int(self._role_state.get("task_join_count", 0)),
            "task_leave_count": int(self._role_state.get("task_leave_count", 0)),
            "service_replace_count": int(
                self._role_state.get("service_replace_count", 0)
            ),
            "role_change_events": tuple(
                self._role_state.get("role_change_events", ())
            ),
            "task_vehicle_ids": tuple(self._task_vehicle_ids),
            "service_vehicle_ids": tuple(self._service_vehicle_ids),
            "mobility_source": self._mobility_source,
            "mobility_diagnostics": self.mobility_diagnostics,
        }
        node_prediction = self.last_node_prediction
        if node_prediction:
            info.update(
                {
                    "predicted_node_demand": np.asarray(
                        node_prediction["predicted_demand"], dtype=float
                    ).copy(),
                    "predicted_node_popularity": np.asarray(
                        node_prediction["predicted_popularity"], dtype=float
                    ).copy(),
                    "prediction_origin_slot": int(
                        node_prediction["prediction_origin_slot"]
                    ),
                    "predicted_node_records": list(
                        node_prediction["node_predictions"]
                    ),
                }
            )
        return self._configuration_state(), reward, done, info

    def _communication(self) -> Mapping[str, Any]:
        return _section(self.config, "communication")

    def _computation(self) -> Mapping[str, Any]:
        return _section(self.config, "computation")

    def _cloud(self) -> Mapping[str, Any]:
        return _section(self.config, "cloud")

    def ground_rate(
        self,
        vehicle_position: Sequence[Real] | np.ndarray,
        node_position: Sequence[Real] | np.ndarray,
        node_type: NodeType | str,
        *,
        bandwidth_fraction: Real,
    ) -> float:
        """计算任务车辆到Service Vehicle或RSU的地面上行速率。"""

        kind = _node_type(node_type)
        if kind not in {NodeType.SERVICE_VEHICLE, NodeType.RSU}:
            raise ValueError("ground_rate的node_type只能是SERVICE_VEHICLE或RSU。")
        comm = self._communication()
        bandwidth_key = "sv" if kind is NodeType.SERVICE_VEHICLE else "rsu"
        bandwidth_mhz = _section(comm, "bandwidth_mhz")
        gain = ground_channel_gain(
            vehicle_position,
            node_position,
            reference_gain=comm["reference_channel_gain"],
            path_loss_exponent=comm["terrestrial_path_loss_exponent"],
        )
        return edge_upload_rate(
            bandwidth_fraction=bandwidth_fraction,
            bandwidth_hz=_bandwidth_hz(bandwidth_mhz[bandwidth_key], "MHz"),
            transmit_power_w=comm["tx_power_w"],
            channel_gain=gain,
            noise_power_w=comm["noise_power_w"],
        )

    def uav_rate(
        self,
        vehicle_position: Sequence[Real] | np.ndarray,
        uav_position: Sequence[Real] | np.ndarray,
        *,
        bandwidth_fraction: Real,
    ) -> float:
        """计算任务车辆到UAV的空地上行速率。"""

        comm = self._communication()
        uav = _section(self.config, "uav")
        channel = uav_channel(
            vehicle_position,
            uav_position,
            height_m=uav["altitude_m"],
            beta_l=comm["air_reference_gain_los"],
            alpha_l=comm["air_path_loss_exponent_los"],
            beta_n=comm["air_reference_gain_nlos"],
            alpha_n=comm["air_path_loss_exponent_nlos"],
            eta_1=comm["los_eta_1"],
            eta_2=comm["los_eta_2"],
        )
        return edge_upload_rate(
            bandwidth_fraction=bandwidth_fraction,
            bandwidth_hz=_bandwidth_hz(
                _section(comm, "bandwidth_mhz")["uav"], "MHz"
            ),
            transmit_power_w=comm["tx_power_w"],
            channel_gain=channel["average_gain"],
            noise_power_w=comm["noise_power_w"],
        )

    def cloud_rate(self) -> float:
        """返回Cloud独立蜂窝链路固定速率。"""

        return _positive(self._cloud()["upload_rate_bps"], "cloud.upload_rate_bps")

    def execute_task(
        self,
        execution_kind: NodeType | str,
        *,
        arrived: bool,
        input_bits: Real,
        cpu_cycles: Real,
        compute_resource_fraction: Real | None = None,
        bandwidth_fraction: Real | None = None,
        vehicle_position: Sequence[Real] | np.ndarray | None = None,
        node_position: Sequence[Real] | np.ndarray | None = None,
        uav_position: Sequence[Real] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """用当前配置计算一个已由上游确定节点和资源的任务结果。"""

        kind = _node_type(execution_kind)
        if not isinstance(arrived, (bool, np.bool_)):
            raise ValueError("arrived必须是布尔值。")
        if not arrived:
            return execute_task(
                kind,
                arrived=False,
                input_bits=input_bits,
                cpu_cycles=cpu_cycles,
            )
        computation = self._computation()
        comm = self._communication()
        if kind is NodeType.LOCAL:
            return execute_task(
                kind,
                arrived=arrived,
                input_bits=input_bits,
                cpu_cycles=cpu_cycles,
                local_frequency_hz=_frequency_hz(
                    _section(computation, "capacity_ghz")["tv"], "GHz"
                ),
                switching_capacitance=computation["cpu_switching_capacitance"],
            )

        if kind is NodeType.CLOUD:
            return execute_task(
                kind,
                arrived=arrived,
                input_bits=input_bits,
                cpu_cycles=cpu_cycles,
                transmit_power_w=comm["tx_power_w"],
                cloud_rate_bps=self.cloud_rate(),
                cloud_frequency_hz=_frequency_hz(
                    _section(computation, "capacity_ghz")["cloud"], "GHz"
                ),
                delay_expansion=self._cloud()["delay_expansion_psi"],
            )

        if vehicle_position is None:
            raise ValueError("边缘任务执行必须提供vehicle_position。")
        if bandwidth_fraction is None or compute_resource_fraction is None:
            raise ValueError("边缘任务执行必须提供带宽比例和计算资源比例。")
        if kind in {NodeType.SERVICE_VEHICLE, NodeType.RSU}:
            if node_position is None:
                raise ValueError("地面边缘任务执行必须提供node_position。")
            rate = self.ground_rate(
                vehicle_position,
                node_position,
                kind,
                bandwidth_fraction=bandwidth_fraction,
            )
            capacity_key = "sv" if kind is NodeType.SERVICE_VEHICLE else "rsu"
        else:
            if uav_position is None:
                raise ValueError("UAV任务执行必须提供uav_position。")
            rate = self.uav_rate(
                vehicle_position,
                uav_position,
                bandwidth_fraction=bandwidth_fraction,
            )
            capacity_key = "uav"
        return execute_task(
            kind,
            arrived=arrived,
            input_bits=input_bits,
            cpu_cycles=cpu_cycles,
            compute_resource_fraction=compute_resource_fraction,
            edge_frequency_hz=_frequency_hz(
                _section(computation, "capacity_ghz")[capacity_key], "GHz"
            ),
            transmit_power_w=comm["tx_power_w"],
            precomputed_rate_bps=rate,
        )

    def _service_sizes_bits(
        self,
        service_sizes: Sequence[Real] | np.ndarray | None,
    ) -> np.ndarray:
        services = _section(self.config, "services")
        sizes = self.service_sizes_mb if service_sizes is None else _finite_vector(
            service_sizes, "service_sizes"
        )
        if sizes.shape != (int(services["count"]),) or np.any(sizes <= 0.0):
            raise ValueError("service_sizes必须与services.count一致且为正数。")
        # services.min_size_mb/max_size_mb 已明确规定服务大小使用MB；删除旧的
        # 删除重复的单位字段后，仍按服务大小的MB→bit换算进入论文传输/缓存公式。
        cache = _section(self.config, "cache")
        bits = sizes * _positive(cache["integer_unit_bytes"], "cache.integer_unit_bytes") * 8.0
        if not np.all(np.isfinite(bits)) or np.any(bits <= 0.0):
            raise ValueError("服务大小换算为bit后必须为有限正数。")
        return bits

    @staticmethod
    def _cache_node_key(kind: NodeType) -> str:
        if kind is NodeType.SERVICE_VEHICLE:
            return "service_vehicle"
        if kind is NodeType.RSU:
            return "rsu"
        if kind is NodeType.UAV:
            return "uav"
        raise ValueError("只有SERVICE_VEHICLE、RSU和UAV可以进行在线缓存更新。")

    def cache_update_cost(
        self,
        previous_cache: Sequence[Real] | np.ndarray,
        current_cache: Sequence[Real] | np.ndarray,
        *,
        node_type: NodeType | str,
        service_sizes: Sequence[Real] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """按配置计算单个可更新节点的实际缓存更新成本。"""

        kind = _node_type(node_type)
        key = self._cache_node_key(kind)
        cache = _section(self.config, "cache")
        return cache_update_cost(
            previous_cache,
            current_cache,
            self._service_sizes_bits(service_sizes),
            fixed_cost_per_update=_section(cache, "fixed_update_cost")[key],
            loading_energy_per_bit=_section(cache, "load_energy_j_per_bit")[key],
            loading_cost_weight=cache["energy_cost_weight"],
        )

    def uav_movement_cost(
        self,
        previous_position: Sequence[Real] | np.ndarray,
        current_position: Sequence[Real] | np.ndarray,
        *,
        movement_epsilon_m: Real,
    ) -> dict[str, Any]:
        """按配置计算一个UAV的实际移动成本。"""

        uav = _section(self.config, "uav")
        return uav_movement_cost(
            previous_position,
            current_position,
            movement_energy_j_per_m=uav["movement_energy_j_per_m"],
            fixed_cost_per_update=uav["fixed_update_cost"],
            movement_energy_cost_weight=uav["movement_energy_cost_weight"],
            movement_epsilon_m=movement_epsilon_m,
        )

    def task_utility(
        self,
        arrived: Sequence[bool | Real] | np.ndarray,
        completion_delay_s: Sequence[Real] | np.ndarray,
        vehicle_energy_j: Sequence[Real] | np.ndarray,
    ) -> dict[str, Any]:
        reward = _section(self.config, "reward")
        return task_utility(
            arrived,
            completion_delay_s,
            vehicle_energy_j,
            delay_weight=reward["delay_weight"],
            energy_weight=reward["energy_weight"],
            delay_reference_s=reward["reference_delay_s"],
            energy_reference_j=reward["reference_energy_j"],
        )

    def system_utility(
        self,
        task_service_utility_value: Real,
        cache_update_cost_value: Real,
        uav_update_cost_value: Real,
    ) -> dict[str, float]:
        return system_utility(
            task_service_utility_value,
            cache_update_cost_value,
            uav_update_cost_value,
        )

    def reward(
        self,
        task_service_utility_value: Real,
        cache_update_cost_value: Real,
        uav_update_cost_value: Real,
    ) -> float:
        """返回最终实际结果对应的论文即时reward ``r(t)=G(t)``。"""

        return self.system_utility(
            task_service_utility_value,
            cache_update_cost_value,
            uav_update_cost_value,
        )["reward"]


__all__ = [
    "MECEnvironment",
    "NodeType",
    "cache_update_cost",
    "edge_upload_rate",
    "execute_task",
    "ground_channel_gain",
    "ground_distance",
    "los_probability",
    "system_utility",
    "task_utility",
    "transmission_delay_energy",
    "uav_channel",
    "uav_channel_gain",
    "uav_movement_cost",
]
