#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一系统物理模型公式。

本文件只放纯公式函数，不读取数据文件，不做 cache/UAV/Matching/KKT/PPO 搜索。
单位约定：

* D / input_bits: bit
* L / cpu_cycles: cycle
* F: cycle/s
* bandwidth: Hz
* rate: bit/s
* energy: J
* distance: m
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


FIXED_TOTAL_NOISE_V0 = "FIXED_TOTAL_NOISE_V0"
BANDWIDTH_DEPENDENT_NOISE_V1 = "BANDWIDTH_DEPENDENT_NOISE_V1"


def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 不能是 NaN/Inf。")
    return value


def ground_channel_gain(distance_m: float, beta_g: float, alpha_g: float) -> float:
    distance = _require_finite("distance_m", distance_m)
    beta = _require_finite("beta_g", beta_g)
    alpha = _require_finite("alpha_g", alpha_g)
    if distance <= 0.0:
        raise ValueError("distance_m 必须大于 0。")
    if beta < 0.0:
        raise ValueError("beta_g 不能为负数。")
    if alpha <= 0.0:
        raise ValueError("alpha_g 必须大于 0。")
    return beta * distance ** (-alpha)


def dbm_per_hz_to_w_per_hz(value_dbm_per_hz: float) -> float:
    value = _require_finite("noise_psd_dbm_per_hz", value_dbm_per_hz)
    result = 10.0 ** ((value - 30.0) / 10.0)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("noise PSD conversion must produce a positive finite W/Hz value")
    return result


def uplink_rate_and_derivative(
    bandwidth_hz: float,
    bandwidth_fraction: float | np.ndarray,
    transmit_power_w: float | np.ndarray,
    channel_gain: float | np.ndarray,
    noise_power_w: float,
    *,
    noise_model: str = FIXED_TOTAL_NOISE_V0,
    noise_psd_w_per_hz: float | None = None,
    receiver_noise_figure_db: float | None = None,
) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Return allocated-bandwidth rate, dR/db, and integrated receiver noise.

    ``bandwidth_fraction`` is a frequency-resource fraction.  In the explicit
    V1 model, noise is integrated over ``b * B`` while the task's total transmit
    power remains fixed.  The legacy branch deliberately retains its former
    fixed-total-noise semantics.
    """

    bandwidth = _require_finite("bandwidth_hz", bandwidth_hz)
    fixed_noise = _require_finite("noise_power_w", noise_power_w)
    if bandwidth <= 0.0:
        raise ValueError("bandwidth_hz 必须大于 0。")
    fractions, powers, gains = np.broadcast_arrays(
        np.asarray(bandwidth_fraction, dtype=np.float64),
        np.asarray(transmit_power_w, dtype=np.float64),
        np.asarray(channel_gain, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(fractions))
        or np.any(~np.isfinite(powers))
        or np.any(~np.isfinite(gains))
    ):
        raise ValueError("bandwidth fraction, power and gain must be finite")
    if np.any(fractions < 0.0) or np.any(fractions > 1.0):
        raise ValueError("bandwidth_fraction 必须位于 [0,1]。")
    if np.any(powers < 0.0) or np.any(gains < 0.0):
        raise ValueError("transmit_power_w/channel_gain 不能为负数。")

    allocated_bandwidth = fractions * bandwidth
    if noise_model == FIXED_TOTAL_NOISE_V0:
        if np.any(fractions <= 0.0):
            raise ValueError("旧固定噪声模型的 bandwidth_fraction 必须位于 (0,1]。")
        if fixed_noise <= 0.0:
            raise ValueError("noise_power_w 必须大于 0。")
        noise = np.full(fractions.shape, fixed_noise, dtype=np.float64)
        snr = powers * gains / noise
        spectral_efficiency = np.log1p(snr) / math.log(2.0)
        rate = allocated_bandwidth * spectral_efficiency
        derivative = bandwidth * spectral_efficiency
    elif noise_model == BANDWIDTH_DEPENDENT_NOISE_V1:
        if noise_psd_w_per_hz is None or receiver_noise_figure_db is None:
            raise ValueError(
                "BANDWIDTH_DEPENDENT_NOISE_V1 requires explicit noise PSD and receiver NF"
            )
        psd = _require_finite("noise_psd_w_per_hz", noise_psd_w_per_hz)
        nf_db = _require_finite("receiver_noise_figure_db", receiver_noise_figure_db)
        if psd <= 0.0:
            raise ValueError("noise_psd_w_per_hz 必须大于 0。")
        if nf_db < 0.0:
            raise ValueError("receiver_noise_figure_db 不能为负数。")
        nf_linear = 10.0 ** (nf_db / 10.0)
        if not math.isfinite(nf_linear):
            raise ValueError("receiver noise figure conversion overflow")
        noise = psd * allocated_bandwidth * nf_linear
        positive = allocated_bandwidth > 0.0
        rate = np.zeros(fractions.shape, dtype=np.float64)
        derivative = np.zeros(fractions.shape, dtype=np.float64)
        if np.any(positive):
            effective_psd = psd * nf_linear
            a = powers[positive] * gains[positive] / effective_psd
            w = allocated_bandwidth[positive]
            ratio = a / w
            log_term = np.log1p(ratio)
            rate[positive] = w * log_term / math.log(2.0)
            derivative[positive] = bandwidth / math.log(2.0) * (
                log_term - a / (w + a)
            )
    else:
        raise ValueError(f"unsupported communication noise model: {noise_model!r}")

    if (
        np.any(~np.isfinite(rate))
        or np.any(~np.isfinite(noise))
        or np.any(np.isnan(derivative))
    ):
        raise ValueError("uplink rate/noise calculation produced NaN/Inf")
    if rate.ndim == 0:
        return float(rate), float(derivative), float(noise)
    return rate, derivative, noise


def uplink_rate(
    bandwidth_hz: float,
    bandwidth_fraction: float,
    transmit_power_w: float,
    channel_gain: float,
    noise_power_w: float,
    *,
    noise_model: str = FIXED_TOTAL_NOISE_V0,
    noise_psd_w_per_hz: float | None = None,
    receiver_noise_figure_db: float | None = None,
) -> float:
    rate, _, _ = uplink_rate_and_derivative(
        bandwidth_hz,
        bandwidth_fraction,
        transmit_power_w,
        channel_gain,
        noise_power_w,
        noise_model=noise_model,
        noise_psd_w_per_hz=noise_psd_w_per_hz,
        receiver_noise_figure_db=receiver_noise_figure_db,
    )
    return float(rate)


def edge_execution(
    input_bits: float,
    cpu_cycles: float,
    bandwidth_hz: float,
    bandwidth_fraction: float,
    transmit_power_w: float,
    channel_gain: float,
    noise_power_w: float,
    node_cpu_cycles_per_s: float,
    cpu_fraction: float,
    *,
    noise_model: str = FIXED_TOTAL_NOISE_V0,
    noise_psd_w_per_hz: float | None = None,
    receiver_noise_figure_db: float | None = None,
) -> dict[str, float]:
    bits = _require_finite("input_bits", input_bits)
    cycles = _require_finite("cpu_cycles", cpu_cycles)
    node_cpu = _require_finite("node_cpu_cycles_per_s", node_cpu_cycles_per_s)
    cpu_share = _require_finite("cpu_fraction", cpu_fraction)
    if bits < 0.0 or cycles < 0.0:
        raise ValueError("input_bits/cpu_cycles 不能为负数。")
    if node_cpu <= 0.0:
        raise ValueError("node_cpu_cycles_per_s 必须大于 0。")
    if not 0.0 < cpu_share <= 1.0:
        raise ValueError("cpu_fraction 必须位于 (0,1]。")
    rate = uplink_rate(
        bandwidth_hz,
        bandwidth_fraction,
        transmit_power_w,
        channel_gain,
        noise_power_w,
        noise_model=noise_model,
        noise_psd_w_per_hz=noise_psd_w_per_hz,
        receiver_noise_figure_db=receiver_noise_figure_db,
    )
    if rate <= 0.0:
        raise ValueError("uplink rate 必须大于 0。")
    upload_delay = bits / rate
    compute_delay = cycles / (cpu_share * node_cpu)
    return {
        "rate_bps": rate,
        "upload_delay_s": upload_delay,
        "compute_delay_s": compute_delay,
        "total_delay_s": upload_delay + compute_delay,
        "vehicle_energy_j": transmit_power_w * upload_delay,
    }


def cloud_execution(
    input_bits: float,
    cpu_cycles: float,
    cloud_rate_bps: float,
    cloud_cpu_cycles_per_s: float,
    cloud_delay_factor: float,
    transmit_power_w: float,
) -> dict[str, float]:
    """Evaluate one task using its assigned CPU service rate and uplink rate.

    The caller is responsible for allocating these per-task rates from any
    shared Cloud pools. Communication delay uses ``cloud_delay_factor`` while
    vehicle radio energy uses only the base airtime ``input_bits / rate``.
    """
    bits = _require_finite("input_bits", input_bits)
    cycles = _require_finite("cpu_cycles", cpu_cycles)
    rate = _require_finite("cloud_rate_bps", cloud_rate_bps)
    cpu = _require_finite("cloud_cpu_cycles_per_s", cloud_cpu_cycles_per_s)
    factor = _require_finite("cloud_delay_factor", cloud_delay_factor)
    power = _require_finite("transmit_power_w", transmit_power_w)
    if bits < 0.0 or cycles < 0.0:
        raise ValueError("input_bits/cpu_cycles 不能为负数。")
    if rate <= 0.0 or cpu <= 0.0:
        raise ValueError("cloud_rate_bps/cloud_cpu_cycles_per_s 必须大于 0。")
    if factor < 1.0:
        raise ValueError("cloud_delay_factor/psi_c 必须不小于 1。")
    if power < 0.0:
        raise ValueError("transmit_power_w 不能为负数。")
    allocated_rate = rate
    allocated_cpu = cpu
    upload_delay = factor * bits / allocated_rate
    compute_delay = cycles / allocated_cpu
    return {
        "allocated_bandwidth_bps": allocated_rate,
        "allocated_cpu_cycles_per_s": allocated_cpu,
        "upload_delay_s": upload_delay,
        "communication_delay_s": upload_delay,
        "compute_delay_s": compute_delay,
        "total_delay_s": upload_delay + compute_delay,
        "vehicle_energy_j": power * bits / allocated_rate,
    }



def air_channel_gain(
    horizontal_distance_m: float,
    altitude_m: float,
    eta1: float,
    eta2: float,
    beta_los: float,
    beta_nlos: float,
    alpha_los: float,
    alpha_nlos: float,
) -> dict[str, float]:
    horizontal = _require_finite("horizontal_distance_m", horizontal_distance_m)
    altitude = _require_finite("altitude_m", altitude_m)
    if horizontal < 0.0:
        raise ValueError("horizontal_distance_m 不能为负数。")
    if altitude <= 0.0:
        raise ValueError("altitude_m 必须大于 0。")
    eta1_value = _require_finite("eta1", eta1)
    eta2_value = _require_finite("eta2", eta2)
    beta_l = _require_finite("beta_los", beta_los)
    beta_n = _require_finite("beta_nlos", beta_nlos)
    alpha_l = _require_finite("alpha_los", alpha_los)
    alpha_n = _require_finite("alpha_nlos", alpha_nlos)
    if eta1_value <= 0.0 or eta2_value <= 0.0:
        raise ValueError("eta1/eta2 必须大于 0。")
    if beta_l < 0.0 or beta_n < 0.0:
        raise ValueError("beta_los/beta_nlos 不能为负数。")
    if alpha_l <= 0.0 or alpha_n <= 0.0:
        raise ValueError("alpha_los/alpha_nlos 必须大于 0。")
    distance_3d = math.hypot(horizontal, altitude)
    ratio = altitude / distance_3d
    ratio = min(max(ratio, -1.0), 1.0)
    theta = math.degrees(math.asin(ratio))
    los_probability = 1.0 / (1.0 + eta1_value * math.exp(-eta2_value * (theta - eta1_value)))
    los_probability = min(max(los_probability, 0.0), 1.0)
    h_los = beta_l * distance_3d ** (-alpha_l)
    h_nlos = beta_n * distance_3d ** (-alpha_n)
    gain = los_probability * h_los + (1.0 - los_probability) * h_nlos
    return {
        "channel_gain": gain,
        "los_probability": los_probability,
        "distance_3d_m": distance_3d,
        "elevation_angle_deg": theta,
    }


def cache_update_cost(
    previous_cache: np.ndarray,
    new_cache: np.ndarray,
    cache_trigger: np.ndarray,
    service_sizes_mb: np.ndarray,
    cache_backhaul_rate_bps: np.ndarray,
    cache_delay_reference_s_by_node: np.ndarray,
) -> dict[str, Any]:
    """Compute the dimensionless cache update cost from actual additions.

    ``service_sizes_mb`` uses decimal MB (1 MB = 1e6 bytes). Download delay is
    independent of task completion delay. Each node's added-service delay is
    divided by its own fixed, positive reference; there is no additional
    lambda/beta multiplier in the current Cache reward contract.
    """
    raw_old = np.asarray(previous_cache)
    raw_new = np.asarray(new_cache)
    raw_trigger = np.asarray(cache_trigger)
    raw_sizes = np.asarray(service_sizes_mb)
    raw_rates = np.asarray(cache_backhaul_rate_bps)
    raw_references = np.asarray(cache_delay_reference_s_by_node)
    try:
        old_values = np.asarray(raw_old, dtype=float)
        new_values = np.asarray(raw_new, dtype=float)
        trigger_values = np.asarray(raw_trigger, dtype=float)
        sizes = np.asarray(raw_sizes, dtype=float)
        rates = np.asarray(raw_rates, dtype=float)
        references = np.asarray(raw_references, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("cache 输入必须是数值数组。") from exc
    for name, values in (
        ("previous_cache", old_values),
        ("new_cache", new_values),
        ("cache_trigger", trigger_values),
        ("service_sizes_mb", sizes),
        ("cache_backhaul_rate_bps", rates),
        ("cache_delay_reference_s_by_node", references),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} 不能包含 NaN/Inf。")
    if old_values.shape != new_values.shape:
        raise ValueError("previous_cache/new_cache shape 必须一致。")
    if old_values.ndim != 2:
        raise ValueError("previous_cache/new_cache 必须是二维矩阵。")
    if trigger_values.shape != (old_values.shape[0],):
        raise ValueError("cache_trigger shape 错误。")
    if sizes.shape != (old_values.shape[1],):
        raise ValueError("service_sizes_mb shape 错误。")
    if rates.shape != trigger_values.shape or references.shape != trigger_values.shape:
        raise ValueError(
            "cache_backhaul_rate_bps/cache_delay_reference_s_by_node shape 错误。"
        )
    if not np.all(np.isin(old_values, [0.0, 1.0])) or not np.all(
        np.isin(new_values, [0.0, 1.0])
    ):
        raise ValueError("previous_cache/new_cache 必须只包含 0 和 1。")
    if not np.all(np.isin(trigger_values, [0.0, 1.0])):
        raise ValueError("cache_trigger 必须只包含 0 和 1。")
    if np.any(sizes < 0.0):
        raise ValueError("service_sizes_mb 不能为负数。")
    if np.any(rates <= 0.0):
        raise ValueError("cache_backhaul_rate_bps 必须全部大于 0。")
    if np.any(references <= 0.0):
        raise ValueError(
            "cache_delay_reference_s_by_node 必须全部为有限正数。"
        )
    old = old_values.astype(np.int8)
    new = new_values.astype(np.int8)
    actual_trigger = np.any(old != new, axis=1).astype(np.int8)
    if np.any(actual_trigger > trigger_values.astype(np.int8)):
        offending = np.flatnonzero(actual_trigger > trigger_values.astype(np.int8)).tolist()
        raise AssertionError(
            "cache actual update exceeds requested trigger for node indices: "
            f"{offending}"
        )
    additions = np.maximum(new - old, 0)
    loaded_mb = additions @ sizes
    loaded_bits = loaded_mb * 8.0e6
    cache_download_delay_s_by_node = loaded_bits / rates
    total_cache_download_delay_s = float(np.sum(cache_download_delay_s_by_node))
    if (
        not np.isfinite(cache_download_delay_s_by_node).all()
        or not math.isfinite(total_cache_download_delay_s)
    ):
        raise FloatingPointError("cache download delay is not finite")
    cache_download_delay_normalized_by_node = np.divide(
        cache_download_delay_s_by_node,
        references,
    )
    delay_reference_total_s = float(np.sum(references))
    cache_download_delay_normalized = float(
        np.sum(cache_download_delay_normalized_by_node)
    )
    cache_cost_by_node = cache_download_delay_normalized_by_node.copy()
    total_cache_cost = cache_download_delay_normalized
    if (
        not np.isfinite(cache_download_delay_normalized_by_node).all()
        or not math.isfinite(cache_download_delay_normalized)
        or np.any(cache_download_delay_normalized_by_node < 0.0)
        or cache_download_delay_normalized < 0.0
        or not np.isfinite(cache_cost_by_node).all()
        or np.any(cache_cost_by_node < 0.0)
        or not math.isfinite(total_cache_cost)
        or total_cache_cost < 0.0
    ):
        raise FloatingPointError("normalized cache cost is not finite")
    return {
        "actual_trigger": actual_trigger,
        "loaded_mb": loaded_mb,
        "loaded_bits": loaded_bits,
        "cache_download_delay_s_by_node": cache_download_delay_s_by_node,
        "cache_download_delay_s": total_cache_download_delay_s,
        "cache_download_delay_normalized_by_node": (
            cache_download_delay_normalized_by_node
        ),
        "cache_download_delay_normalized": cache_download_delay_normalized,
        "cache_download_delay_reference_s_by_node": references,
        # Aggregate retained for diagnostics only; normalization uses each node's value.
        "cache_download_delay_reference_s": delay_reference_total_s,
        "cache_download_delay_reference_s_total": delay_reference_total_s,
        "cache_cost_by_node": cache_cost_by_node,
        "cache_cost": total_cache_cost,
    }


def aggregate_system_utility(
    task_utility: float,
    cache_cost: float,
    movement_cost: float,
    beta_ca: float,
    beta_q: float,
) -> float:
    """Combine Q and the two unweighted normalized costs exactly once."""

    task_value = _require_finite("task_utility", task_utility)
    cache_value = _require_finite("cache_cost", cache_cost)
    movement_value = _require_finite("movement_cost", movement_cost)
    cache_weight = _require_finite("beta_ca", beta_ca)
    movement_weight = _require_finite("beta_q", beta_q)
    if cache_value < 0.0 or movement_value < 0.0:
        raise ValueError("normalized system costs must be non-negative")
    if cache_weight < 0.0 or movement_weight < 0.0:
        raise ValueError("beta_ca/beta_q must be non-negative")
    cache_penalty = cache_weight * cache_value
    movement_penalty = movement_weight * movement_value
    if not math.isfinite(cache_penalty) or not math.isfinite(movement_penalty):
        raise FloatingPointError("weighted system cost is not finite")
    try:
        utility = math.fsum((task_value, -cache_penalty, -movement_penalty))
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError("system utility is not finite") from exc
    if not math.isfinite(utility):
        raise FloatingPointError("system utility is not finite")
    return float(utility)


def uav_movement_cost(
    q_old_xy: np.ndarray,
    q_new_xy: np.ndarray,
    z_q: int,
    dt_s: float,
    max_horizontal_speed_mps: float,
    movement_energy_coefficient_j_per_m: float,
    movement_fixed_cost: float,
    movement_energy_reference_j: float,
    tolerance_m: float = 1e-9,
) -> dict[str, float | bool | int]:
    """Compute the unweighted normalized UAV cost from actual movement energy.

    ``movement_fixed_cost`` remains in the signature for configuration
    compatibility, but does not contribute to the cost.
    """
    old = np.asarray(q_old_xy, dtype=float)
    new = np.asarray(q_new_xy, dtype=float)
    if old.shape != (2,) or new.shape != (2,):
        raise ValueError("q_old_xy/q_new_xy 必须是二维坐标。")
    if not np.all(np.isfinite(old)) or not np.all(np.isfinite(new)):
        raise ValueError("UAV 坐标不能包含 NaN/Inf。")
    z_value = _require_finite("z_q", z_q)
    if z_value not in (0.0, 1.0):
        raise ValueError("z_q 必须是 0/1。")
    dt = _require_finite("dt_s", dt_s)
    vmax = _require_finite("max_horizontal_speed_mps", max_horizontal_speed_mps)
    xi = _require_finite("movement_energy_coefficient_j_per_m", movement_energy_coefficient_j_per_m)
    compat_fixed_cost = _require_finite(
        "movement_fixed_cost", movement_fixed_cost
    )
    energy_reference = _require_finite(
        "movement_energy_reference_j", movement_energy_reference_j
    )
    tolerance = _require_finite("tolerance_m", tolerance_m)
    if tolerance < 0.0:
        raise ValueError("tolerance_m 不能为负数。")
    if (
        dt <= 0.0
        or vmax < 0.0
        or xi < 0.0
        or compat_fixed_cost < 0.0
        or energy_reference < 0.0
    ):
        raise ValueError("UAV movement 参数范围无效。")
    raw_distance = float(np.linalg.norm(new - old))
    actual_z = int(raw_distance > tolerance)
    if actual_z > int(z_value):
        raise AssertionError("UAV actual movement exceeds requested trigger z_q=0。")
    effective_distance = raw_distance if actual_z == 1 else 0.0
    speed_feasible = effective_distance <= actual_z * vmax * dt + tolerance
    if not speed_feasible:
        raise ValueError("UAV 水平移动距离违反 C6 最大移动距离约束。")
    energy = xi * effective_distance
    if not math.isfinite(energy):
        raise FloatingPointError("UAV movement energy is not finite")
    if energy_reference == 0.0:
        if energy > 0.0:
            raise ValueError(
                "positive UAV movement energy is incompatible with a zero reference"
            )
        energy_normalized = 0.0
    else:
        energy_tolerance_j = max(1.0e-12, energy_reference * 1.0e-12)
        if energy > energy_reference + energy_tolerance_j:
            raise AssertionError(
                "actual UAV movement energy exceeds the fixed normalization reference"
            )
        energy_normalized = energy / energy_reference
    if (
        not math.isfinite(energy_normalized)
        or energy_normalized < 0.0
    ):
        raise FloatingPointError("normalized UAV movement cost is not finite and nonnegative")
    return {
        "distance_m": effective_distance,
        "raw_distance_m": raw_distance,
        "speed_feasible": speed_feasible,
        "movement_energy_j": energy,
        "movement_energy_normalized": energy_normalized,
        "movement_energy_reference_j": energy_reference,
        "movement_cost": energy_normalized,
        "actual_z_q": actual_z,
        "actual_update": bool(actual_z),
    }


def task_utility(
    delay_s: float,
    vehicle_energy_j: float,
    t_ref_s: float,
    e_ref_j: float,
    omega_t: float,
    omega_e: float,
) -> float:
    delay = _require_finite("delay_s", delay_s)
    energy = _require_finite("vehicle_energy_j", vehicle_energy_j)
    t_ref = _require_finite("T_ref_s", t_ref_s)
    e_ref = _require_finite("E_ref_j", e_ref_j)
    wt = _require_finite("omega_T", omega_t)
    we = _require_finite("omega_E", omega_e)
    if delay < 0.0:
        raise ValueError("delay_s 不能为负数。")
    if energy < 0.0:
        raise ValueError("vehicle_energy_j 不能为负数。")
    if t_ref <= 0.0 or e_ref <= 0.0:
        raise ValueError("T_ref/E_ref 必须大于 0。")
    if not 0.0 <= wt <= 1.0:
        raise ValueError("omega_T 必须位于 [0,1]。")
    if not 0.0 <= we <= 1.0:
        raise ValueError("omega_E 必须位于 [0,1]。")
    if not math.isclose(wt + we, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("omega_T 和 omega_E 的总和必须等于 1。")
    return wt * (1.0 - delay / t_ref) + we * (1.0 - energy / e_ref)
