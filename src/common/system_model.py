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


def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 不能是 NaN/Inf。")
    return value


def local_execution(
    cpu_cycles: float,
    local_cpu_cycles_per_s: float,
    kappa: float,
) -> dict[str, float]:
    """Local: T=L/F_v, E=kappa*L*F_v^2。"""
    cycles = _require_finite("cpu_cycles", cpu_cycles)
    cpu = _require_finite("local_cpu_cycles_per_s", local_cpu_cycles_per_s)
    kappa_value = _require_finite("kappa", kappa)
    if cycles < 0.0:
        raise ValueError("cpu_cycles 不能为负数。")
    if cpu <= 0.0:
        raise ValueError("local_cpu_cycles_per_s 必须大于 0。")
    if kappa_value < 0.0:
        raise ValueError("kappa 不能为负数。")
    return {
        "delay_s": cycles / cpu,
        "vehicle_energy_j": kappa_value * cycles * cpu * cpu,
    }


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


def uplink_rate(
    bandwidth_hz: float,
    bandwidth_fraction: float,
    transmit_power_w: float,
    channel_gain: float,
    noise_power_w: float,
) -> float:
    bandwidth = _require_finite("bandwidth_hz", bandwidth_hz)
    fraction = _require_finite("bandwidth_fraction", bandwidth_fraction)
    power = _require_finite("transmit_power_w", transmit_power_w)
    gain = _require_finite("channel_gain", channel_gain)
    noise = _require_finite("noise_power_w", noise_power_w)
    if bandwidth <= 0.0:
        raise ValueError("bandwidth_hz 必须大于 0。")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("bandwidth_fraction 必须位于 (0,1]。")
    if power < 0.0 or gain < 0.0:
        raise ValueError("transmit_power_w/channel_gain 不能为负数。")
    if noise <= 0.0:
        raise ValueError("noise_power_w 必须大于 0。")
    snr = power * gain / noise
    return fraction * bandwidth * math.log2(1.0 + snr)


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
    cloud_task_count: int = 1,
) -> dict[str, float]:
    """Evaluate one task on the single shared Cloud node.

    ``cloud_rate_bps`` and ``cloud_cpu_cycles_per_s`` are total Cloud-node
    capacities.  All Cloud tasks in the same slot pass the same
    ``cloud_task_count`` and receive an equal resource share.  The default of
    one preserves the single-task/legacy call contract used by analysis-only
    surrogate evaluations.
    """
    bits = _require_finite("input_bits", input_bits)
    cycles = _require_finite("cpu_cycles", cpu_cycles)
    rate = _require_finite("cloud_rate_bps", cloud_rate_bps)
    cpu = _require_finite("cloud_cpu_cycles_per_s", cloud_cpu_cycles_per_s)
    factor = _require_finite("cloud_delay_factor", cloud_delay_factor)
    power = _require_finite("transmit_power_w", transmit_power_w)
    count_value = _require_finite("cloud_task_count", float(cloud_task_count))
    if bits < 0.0 or cycles < 0.0:
        raise ValueError("input_bits/cpu_cycles 不能为负数。")
    if rate <= 0.0 or cpu <= 0.0:
        raise ValueError("cloud_rate_bps/cloud_cpu_cycles_per_s 必须大于 0。")
    if count_value < 1.0 or not math.isclose(count_value, round(count_value), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("cloud_task_count 必须是正整数。")
    if factor < 1.0:
        raise ValueError("cloud_delay_factor/psi_c 必须不小于 1。")
    if power < 0.0:
        raise ValueError("transmit_power_w 不能为负数。")
    task_count = int(round(count_value))
    allocated_rate = rate / task_count
    allocated_cpu = cpu / task_count
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
    eta_ca: np.ndarray,
    xi_ca_per_mb: np.ndarray,
    lambda_ca: float,
) -> dict[str, Any]:
    raw_old = np.asarray(previous_cache)
    raw_new = np.asarray(new_cache)
    raw_trigger = np.asarray(cache_trigger)
    raw_sizes = np.asarray(service_sizes_mb)
    raw_eta = np.asarray(eta_ca)
    raw_xi = np.asarray(xi_ca_per_mb)
    try:
        old_values = np.asarray(raw_old, dtype=float)
        new_values = np.asarray(raw_new, dtype=float)
        trigger_values = np.asarray(raw_trigger, dtype=float)
        sizes = np.asarray(raw_sizes, dtype=float)
        eta = np.asarray(raw_eta, dtype=float)
        xi = np.asarray(raw_xi, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("cache 输入必须是数值数组。") from exc
    for name, values in (
        ("previous_cache", old_values),
        ("new_cache", new_values),
        ("cache_trigger", trigger_values),
        ("service_sizes_mb", sizes),
        ("eta_ca", eta),
        ("xi_ca_per_mb", xi),
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
    if eta.shape != trigger_values.shape or xi.shape != trigger_values.shape:
        raise ValueError("eta_ca/xi_ca shape 错误。")
    if not np.all(np.isin(old_values, [0.0, 1.0])) or not np.all(
        np.isin(new_values, [0.0, 1.0])
    ):
        raise ValueError("previous_cache/new_cache 必须只包含 0 和 1。")
    if not np.all(np.isin(trigger_values, [0.0, 1.0])):
        raise ValueError("cache_trigger 必须只包含 0 和 1。")
    if np.any(sizes < 0.0):
        raise ValueError("service_sizes_mb 不能为负数。")
    if np.any(eta < 0.0):
        raise ValueError("eta_ca 不能为负数。")
    if np.any(xi < 0.0):
        raise ValueError("xi_ca_per_mb 不能为负数。")
    lam = _require_finite("lambda_ca", lambda_ca)
    if lam < 0.0:
        raise ValueError("lambda_ca 不能为负数。")
    old = old_values.astype(np.int8)
    new = new_values.astype(np.int8)
    actual_trigger = np.any(old != new, axis=1).astype(np.int8)
    additions = np.maximum(new - old, 0)
    loaded_mb = additions @ sizes
    loading_energy = xi * loaded_mb
    cost_by_node = actual_trigger.astype(float) * eta + lam * loading_energy
    return {
        "actual_trigger": actual_trigger,
        "loaded_mb": loaded_mb,
        "loading_energy": loading_energy,
        "cost_by_node": cost_by_node,
        "total_cost": float(np.sum(cost_by_node)),
    }


def uav_movement_cost(
    q_old_xy: np.ndarray,
    q_new_xy: np.ndarray,
    z_q: int,
    dt_s: float,
    max_horizontal_speed_mps: float,
    movement_energy_coefficient_j_per_m: float,
    movement_fixed_cost: float,
    lambda_movement: float,
    tolerance_m: float = 1e-9,
) -> dict[str, float | bool | int]:
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
    eta = _require_finite("movement_fixed_cost", movement_fixed_cost)
    lam = _require_finite("lambda_movement", lambda_movement)
    tolerance = _require_finite("tolerance_m", tolerance_m)
    if tolerance < 0.0:
        raise ValueError("tolerance_m 不能为负数。")
    if dt <= 0.0 or vmax < 0.0 or xi < 0.0 or eta < 0.0 or lam < 0.0:
        raise ValueError("UAV movement 参数范围无效。")
    raw_distance = float(np.linalg.norm(new - old))
    actual_z = int(raw_distance > tolerance)
    effective_distance = raw_distance if actual_z == 1 else 0.0
    speed_feasible = effective_distance <= actual_z * vmax * dt + tolerance
    if not speed_feasible:
        raise ValueError("UAV 水平移动距离违反 C6 最大移动距离约束。")
    energy = xi * effective_distance
    return {
        "distance_m": effective_distance,
        "raw_distance_m": raw_distance,
        "speed_feasible": speed_feasible,
        "movement_energy_j": energy,
        "movement_cost": actual_z * eta + lam * energy,
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
