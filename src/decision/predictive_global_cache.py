"""Prediction-driven global marginal Cache value learning and search.

The supervised target is a paired, finite-horizon difference in the formal
system utility.  Forecast-only feature construction is kept separate from the
Training-only label executor so realized future tasks cannot enter inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from common import system_model
from . import cache_configuration, slot_environment


ALGORITHM_SCHEMA = "predictive_global_cache_marginal_value_v1"
FEATURE_SCHEMA = "global_cache_candidate_forecast_v1"
LABEL_SCHEMA = "paired_training_fixed_cache_horizon5_uav_stay_v1"
SEARCH_SCHEMA = "bounded_sequential_single_node_cache_greedy_v1"
MAX_FORECAST_HORIZON = 5


@dataclass(frozen=True)
class SearchBudget:
    """Hard limits for one Macro cache decision."""

    max_candidate_evaluations: int = 36
    max_cache_updates: int = 4
    max_runtime_seconds: float = 1.0
    per_node_candidate_capacity: int = 4

    def validate(self) -> None:
        if self.max_candidate_evaluations <= 0:
            raise ValueError("max_candidate_evaluations must be positive")
        if self.max_cache_updates <= 0:
            raise ValueError("max_cache_updates must be positive")
        if (
            not math.isfinite(self.max_runtime_seconds)
            or self.max_runtime_seconds <= 0.0
        ):
            raise ValueError("max_runtime_seconds must be positive and finite")
        if self.per_node_candidate_capacity <= 0:
            raise ValueError("per_node_candidate_capacity must be positive")


@dataclass(frozen=True)
class TrainingBudget:
    """Finite, deterministic budget for offline Training-only paired labels."""

    episode_count: int = 4
    states_per_episode: int = 8
    candidate_pair_budget: int = 128
    horizon_slots: int = MAX_FORECAST_HORIZON
    candidates_per_node: int = 2

    def validate(self) -> None:
        if self.episode_count <= 0 or self.states_per_episode <= 0:
            raise ValueError("episode_count and states_per_episode must be positive")
        if self.candidate_pair_budget <= 0 or self.candidates_per_node <= 0:
            raise ValueError("candidate label budgets must be positive")
        if not 1 <= self.horizon_slots <= MAX_FORECAST_HORIZON:
            raise ValueError("horizon_slots must be between 1 and 5")


def _finite_array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise FloatingPointError(f"{name} contains a non-finite value")
    return result


def _role_cpu_bandwidth(
    role: str,
    cache_config: cache_configuration.CacheOptimizationConfig,
) -> tuple[float, float, float]:
    name = str(role).upper()
    if name == "SV":
        return (
            float(cache_config.sv_cpu_cycles_per_s),
            float(cache_config.sv_bandwidth_hz),
            float(cache_config.sv_range_m),
        )
    if name == "RSU":
        return (
            float(cache_config.rsu_cpu_cycles_per_s),
            float(cache_config.rsu_bandwidth_hz),
            float(cache_config.rsu_range_m),
        )
    if name == "UAV":
        return (
            float(cache_config.uav_cpu_cycles_per_s),
            float(cache_config.uav_bandwidth_hz),
            float(cache_config.uav_range_m),
        )
    raise ValueError(f"Unsupported Cache node role: {role!r}")


def _position_normalizer(
    flight_boundary: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    # Formal SUMO boundary order is (x_min, y_min, x_max, y_max).
    x_min, y_min, x_max, y_max = (float(value) for value in flight_boundary)
    if not x_max > x_min or not y_max > y_min:
        raise ValueError("flight boundary must have positive width and height")
    return x_min, x_max, y_min, y_max


def _normalize_xy(
    xy: np.ndarray,
    boundary: tuple[float, float, float, float],
) -> np.ndarray:
    x_min, x_max, y_min, y_max = _position_normalizer(boundary)
    result = np.asarray(xy, dtype=np.float64).copy()
    result[..., 0] = (result[..., 0] - x_min) / (x_max - x_min)
    result[..., 1] = (result[..., 1] - y_min) / (y_max - y_min)
    return result


def _forecast_arrays(
    scenario: dict[str, Any],
    horizon_mask: Any,
    service_count: int,
) -> dict[str, Any]:
    """Validate and cap predictor-only arrays; never accepts a rollout step."""

    arrivals = np.asarray(scenario.get("arrival"))
    services = np.asarray(scenario.get("service_id"))
    input_bits = _finite_array(scenario.get("input_size"), "forecast input_size", 3)
    cpu_cycles = _finite_array(scenario.get("cpu_cycles"), "forecast cpu_cycles", 3)
    positions = _finite_array(scenario.get("positions_xy"), "forecast positions_xy", 4)
    if arrivals.ndim != 3 or services.shape != arrivals.shape:
        raise ValueError("forecast arrival/service_id must share [sample,horizon,TV]")
    if input_bits.shape != arrivals.shape or cpu_cycles.shape != arrivals.shape:
        raise ValueError("forecast workload arrays must align with [sample,horizon,TV]")
    sample_count, horizon_count, task_vehicle_count = arrivals.shape
    if positions.shape[:2] != (sample_count, horizon_count) or positions.shape[-1] != 2:
        raise ValueError("forecast positions must align as [sample,horizon,vehicle,2]")
    if horizon_count <= 0 or horizon_count > MAX_FORECAST_HORIZON:
        raise ValueError("scenario horizon must be between 1 and 5")
    if not np.all(np.isin(arrivals, [0, 1])):
        raise ValueError("forecast arrivals must be binary")
    if np.any(input_bits[arrivals.astype(bool)] <= 0.0) or np.any(
        cpu_cycles[arrivals.astype(bool)] <= 0.0
    ):
        raise ValueError("arriving forecast tasks must have positive workload")
    mask = cache_configuration.normalize_prediction_horizon_mask(
        horizon_mask, horizon_count
    )
    if not np.any(mask):
        raise ValueError("forecast horizon mask has no valid slot")
    vehicle_ids = tuple(str(value) for value in np.asarray(scenario["vehicle_ids"]).tolist())
    task_vehicle_ids = tuple(
        str(value) for value in np.asarray(scenario["task_vehicle_ids"]).tolist()
    )
    service_vehicle_ids = tuple(
        str(value) for value in np.asarray(scenario["service_vehicle_ids"]).tolist()
    )
    if len(vehicle_ids) != positions.shape[2] or len(task_vehicle_ids) != task_vehicle_count:
        raise ValueError("forecast vehicle IDs do not align with arrays")
    vehicle_index = {vehicle_id: index for index, vehicle_id in enumerate(vehicle_ids)}
    if len(vehicle_index) != len(vehicle_ids) or any(
        vehicle_id not in vehicle_index for vehicle_id in task_vehicle_ids
    ):
        raise ValueError("forecast vehicle IDs must be unique and task vehicles known")
    if np.any(services[arrivals.astype(bool)] < 0) or np.any(
        services[arrivals.astype(bool)] >= int(service_count)
    ):
        raise ValueError("forecast service_id is outside the catalog")
    rho = _finite_array(scenario.get("rho_hat"), "forecast rho_hat")
    if rho.ndim == 2:
        rho_hf = rho
    elif rho.ndim == 3 and rho.shape[1] == horizon_count:
        rho_hf = np.mean(rho, axis=0)
    else:
        raise ValueError("forecast rho_hat must be [horizon,service] or [sample,horizon,service]")
    if rho_hf.shape != (horizon_count, int(service_count)) or np.any(rho_hf < -1.0e-8):
        raise ValueError("forecast rho_hat shape/value does not match the service catalog")
    valid_rho = rho_hf[mask]
    if not np.allclose(np.sum(valid_rho, axis=1), 1.0, rtol=1.0e-5, atol=1.0e-5):
        raise ValueError("forecast rho_hat rows must sum to one")
    return {
        "arrival": arrivals.astype(np.int8, copy=False),
        "service_id": services.astype(np.int64, copy=False),
        "input_size": input_bits,
        "cpu_cycles": cpu_cycles,
        "positions_xy": positions,
        "horizon_mask": mask,
        "vehicle_ids": vehicle_ids,
        "task_vehicle_ids": task_vehicle_ids,
        "service_vehicle_ids": service_vehicle_ids,
        "vehicle_index": vehicle_index,
        "rho_hat": rho_hf,
    }


def _node_position_by_sample(
    node_index: int,
    h: int,
    sample_index: int,
    arrays: dict[str, Any],
    cache_nodes: dict[str, Any],
    current_uav_positions: dict[str, Any],
) -> np.ndarray:
    node_id = str(cache_nodes["node_ids"][node_index])
    role = str(cache_nodes["node_roles"][node_index]).upper()
    if role == "SV":
        vehicle_index = arrays["vehicle_index"].get(node_id)
        if vehicle_index is None:
            raise ValueError(f"SV cache node {node_id!r} is missing from forecast vehicles")
        return arrays["positions_xy"][sample_index, h, vehicle_index]
    if role == "RSU":
        return _finite_array(cache_nodes["fixed_positions"][node_id], f"{node_id} position", 1)
    if role == "UAV":
        if node_id not in current_uav_positions:
            raise ValueError(f"current UAV position is missing for {node_id!r}")
        return _finite_array(current_uav_positions[node_id], f"{node_id} position", 1)
    raise ValueError(f"Unsupported Cache node role {role!r}")


def prepare_global_forecast_features(
    scenario: dict[str, Any],
    prediction_horizon_mask: Any,
    current_uav_positions: dict[str, Any],
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_cfg: cache_configuration.CacheOptimizationConfig,
    project_cfg: dict[str, Any],
    flight_boundary: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Precompute shared forecast and resource-competition features once per state."""

    service_count = int(service_catalog["num_services"])
    service_sizes = _finite_array(service_catalog["service_sizes_mb"], "service_sizes_mb", 1)
    mean_input = _finite_array(service_catalog["mean_input_size_bit"], "mean_input_size_bit", 1)
    cycles_per_bit = _finite_array(service_catalog["cycles_per_bit"], "cycles_per_bit", 1)
    if any(value.shape != (service_count,) for value in (service_sizes, mean_input, cycles_per_bit)):
        raise ValueError("service catalog feature arrays do not align")
    arrays = _forecast_arrays(scenario, prediction_horizon_mask, service_count)
    arrivals = arrays["arrival"]
    services = arrays["service_id"]
    input_bits = arrays["input_size"]
    cpu_cycles = arrays["cpu_cycles"]
    positions = arrays["positions_xy"]
    mask = arrays["horizon_mask"]
    sample_count, horizon_count, task_vehicle_count = arrivals.shape
    task_vehicle_ids = arrays["task_vehicle_ids"]

    global_work = np.zeros((horizon_count, service_count, 3), dtype=np.float64)
    for h in range(horizon_count):
        if not mask[h]:
            continue
        for service_id in range(service_count):
            hits = (arrivals[:, h, :] == 1) & (services[:, h, :] == service_id)
            global_work[h, service_id, 0] = float(np.mean(np.sum(hits, axis=1))) / max(task_vehicle_count, 1)
            global_work[h, service_id, 1] = float(
                np.mean(np.sum(input_bits[:, h, :] * hits, axis=1))
            ) / max(task_vehicle_count * float(mean_input[service_id]), 1.0)
            mean_cycles = float(mean_input[service_id] * cycles_per_bit[service_id])
            global_work[h, service_id, 2] = float(
                np.mean(np.sum(cpu_cycles[:, h, :] * hits, axis=1))
            ) / max(task_vehicle_count * mean_cycles, 1.0)

    node_count = len(cache_nodes["node_ids"])
    role_names = ("SV", "RSU", "UAV")
    role_index = {name: index for index, name in enumerate(role_names)}
    role_coverage = np.zeros((horizon_count, len(role_names), service_count), dtype=np.float64)
    role_cpu_load = np.zeros((horizon_count, len(role_names)), dtype=np.float64)
    role_upload_offer = np.zeros((horizon_count, len(role_names)), dtype=np.float64)
    node_position_mean = np.zeros((node_count, horizon_count, 2), dtype=np.float64)
    local_work_by_node = np.zeros((node_count, horizon_count, service_count, 3), dtype=np.float64)
    dt_s = float(project_cfg["simulation"]["dt_s"])
    for node_index in range(node_count):
        role = str(cache_nodes["node_roles"][node_index]).upper()
        role_id = role_index.get(role)
        if role_id is None:
            raise ValueError(f"unsupported Cache node role: {role}")
        cpu_capacity, _bandwidth_capacity, range_m = _role_cpu_bandwidth(role, cache_cfg)
        for h in range(horizon_count):
            if not mask[h]:
                continue
            node_xy_samples = np.stack(
                [
                    _node_position_by_sample(
                        node_index, h, sample_index, arrays, cache_nodes, current_uav_positions
                    )
                    for sample_index in range(sample_count)
                ],
                axis=0,
            )
            node_position_mean[node_index, h] = np.mean(node_xy_samples, axis=0)
            for tv_index, vehicle_id in enumerate(task_vehicle_ids):
                full_vehicle_index = arrays["vehicle_index"][vehicle_id]
                for sample_index in range(sample_count):
                    if arrivals[sample_index, h, tv_index] != 1:
                        continue
                    position = positions[sample_index, h, full_vehicle_index]
                    distance_m = float(np.linalg.norm(position - node_xy_samples[sample_index]))
                    weight = max(0.0, 1.0 - distance_m / max(range_m, 1.0e-12))
                    service_id = int(services[sample_index, h, tv_index])
                    work = np.asarray(
                        [1.0, input_bits[sample_index, h, tv_index], cpu_cycles[sample_index, h, tv_index]],
                        dtype=np.float64,
                    )
                    scale = np.asarray(
                        [
                            max(task_vehicle_count * sample_count, 1),
                            max(task_vehicle_count * sample_count * mean_input[service_id], 1.0),
                            max(
                                task_vehicle_count
                                * sample_count
                                * mean_input[service_id]
                                * cycles_per_bit[service_id],
                                1.0,
                            ),
                        ],
                        dtype=np.float64,
                    )
                    role_coverage[h, role_id, service_id] += weight / scale[0]
                    local_work_by_node[node_index, h, service_id] += weight * work / scale
                    role_cpu_load[h, role_id] += weight * work[2] / max(
                        sample_count * cpu_capacity * dt_s, 1.0
                    )
                    role_upload_offer[h, role_id] += weight * work[1] / max(
                        sample_count * task_vehicle_count * mean_input[service_id], 1.0
                    )

    normalized_positions = _normalize_xy(positions, flight_boundary)
    vehicle_summary = np.zeros((horizon_count, 4), dtype=np.float64)
    task_vehicle_indices = [arrays["vehicle_index"][item] for item in task_vehicle_ids]
    for h in np.flatnonzero(mask):
        tv_positions = normalized_positions[:, h, task_vehicle_indices, :].reshape(-1, 2)
        vehicle_summary[h, :2] = np.mean(tv_positions, axis=0)
        vehicle_summary[h, 2:] = np.std(tv_positions, axis=0)

    cloud_cycles_per_s = float(cache_cfg.cloud_total_cpu_cycles_per_s)
    cloud_load = np.zeros(horizon_count, dtype=np.float64)
    for h in np.flatnonzero(mask):
        expected_cycles = float(
            np.sum(global_work[h, :, 2] * max(task_vehicle_count, 1) * mean_input * cycles_per_bit)
        )
        cloud_load[h] = expected_cycles / max(cloud_cycles_per_s * dt_s, 1.0)
    cloud_features = np.concatenate(
        [
            np.asarray(
                [
                    float(cache_cfg.cloud_resource_mode == "finite_pool"),
                    math.log1p(cloud_cycles_per_s / 1.0e9),
                    math.log1p(float(cache_cfg.cloud_total_bandwidth_bps) / 1.0e6),
                    math.log1p(float(cache_cfg.cloud_uplink_rate_bps) / 1.0e6),
                    float(cache_cfg.cloud_delay_factor),
                ],
                dtype=np.float64,
            ),
            cloud_load,
        ]
    )
    role_resource_summary = np.zeros((len(role_names), 2), dtype=np.float64)
    for role_id, role_name in enumerate(role_names):
        role_cpu_total = 0.0
        role_bandwidth_total = 0.0
        for role in cache_nodes["node_roles"]:
            if str(role).upper() == role_name:
                cpu_capacity, bandwidth_capacity, _range_m = _role_cpu_bandwidth(
                    role_name, cache_cfg
                )
                role_cpu_total += cpu_capacity
                role_bandwidth_total += bandwidth_capacity
        role_resource_summary[role_id] = [
            math.log1p(role_cpu_total / 1.0e10),
            math.log1p(role_bandwidth_total / 1.0e7),
        ]
    uav_ids = [
        str(cache_nodes["node_ids"][index])
        for index, role in enumerate(cache_nodes["node_roles"])
        if str(role).upper() == "UAV"
    ]
    if uav_ids:
        uav_positions = np.stack(
            [_finite_array(current_uav_positions[key], f"UAV {key} position", 1) for key in uav_ids]
        )
        uav_positions = _normalize_xy(uav_positions, flight_boundary)
        uav_summary = np.concatenate(
            [np.mean(uav_positions, axis=0), np.std(uav_positions, axis=0), [len(uav_ids)]]
        )
    else:
        uav_summary = np.zeros(5, dtype=np.float64)
    rho = arrays["rho_hat"].copy()
    rho[~mask] = 0.0
    result = {
        "arrays": arrays,
        "service_sizes": service_sizes,
        "rho": rho,
        "global_work": global_work,
        "role_coverage": role_coverage,
        "role_cpu_load": role_cpu_load,
        "role_upload_offer": role_upload_offer,
        "role_resource_summary": role_resource_summary,
        "cloud_features": cloud_features,
        "vehicle_summary": vehicle_summary,
        "uav_summary": uav_summary,
        "node_position_mean": node_position_mean,
        "local_work_by_node": local_work_by_node,
        "role_names": role_names,
        "role_index": role_index,
        "task_vehicle_count": task_vehicle_count,
    }
    return result


def build_global_candidate_features(
    current_cache: np.ndarray,
    target_cache: np.ndarray,
    node_index: int,
    scenario: dict[str, Any],
    prediction_horizon_mask: Any,
    current_uav_positions: dict[str, Any],
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_cfg: cache_configuration.CacheOptimizationConfig,
    project_cfg: dict[str, Any],
    flight_boundary: tuple[float, float, float, float],
    *,
    forecast_context: dict[str, Any] | None = None,
) -> np.ndarray:
    """Create a fixed-width, predictor-only feature for one legal Cache update.

    The expensive forecast/coverage aggregates can be shared across candidates
    in one state via ``forecast_context``. No actual Training tasks are accepted.
    """

    current = np.asarray(current_cache, dtype=np.int8)
    target = np.asarray(target_cache, dtype=np.int8)
    cache_configuration.validate_cache_shape_and_capacity(current, service_catalog, cache_nodes)
    cache_configuration.validate_cache_shape_and_capacity(target, service_catalog, cache_nodes)
    changed = np.flatnonzero(np.any(current != target, axis=1))
    if changed.tolist() != [int(node_index)]:
        raise ValueError("a candidate must update exactly its declared Cache node")
    context = forecast_context or prepare_global_forecast_features(
        scenario,
        prediction_horizon_mask,
        current_uav_positions,
        service_catalog,
        cache_nodes,
        cache_cfg,
        project_cfg,
        flight_boundary,
    )
    service_sizes = np.asarray(context["service_sizes"], dtype=np.float64)
    service_count = int(service_catalog["num_services"])
    node_count = int(current.shape[0])
    horizon_count = int(context["arrays"]["arrival"].shape[1])
    role_names = context["role_names"]
    role_index = context["role_index"]
    role_cache_summary = np.zeros((len(role_names), 2), dtype=np.float64)
    capacities = np.asarray(cache_nodes["capacities_mb"], dtype=np.float64)
    for role_id, role_name in enumerate(role_names):
        indices = [index for index, role in enumerate(cache_nodes["node_roles"]) if str(role).upper() == role_name]
        if indices:
            occupancy = np.sum(current[indices] * service_sizes[None, :], axis=1) / np.maximum(capacities[indices], 1.0)
            role_cache_summary[role_id] = [len(indices), float(np.mean(occupancy))]

    candidate_node = int(node_index)
    role = str(cache_nodes["node_roles"][candidate_node]).upper()
    one_hot = np.zeros(3, dtype=np.float64)
    one_hot[role_index[role]] = 1.0
    cpu_capacity, bandwidth_capacity, range_m = _role_cpu_bandwidth(role, cache_cfg)
    cache_capacity = float(capacities[candidate_node])
    before_occupancy = float(np.dot(current[candidate_node], service_sizes)) / max(cache_capacity, 1.0)
    after_occupancy = float(np.dot(target[candidate_node], service_sizes)) / max(cache_capacity, 1.0)
    changed_size = float(np.dot(np.abs(target[candidate_node] - current[candidate_node]), service_sizes)) / max(cache_capacity, 1.0)
    trigger = np.any(current != target, axis=1).astype(np.int8)
    cache_cost, _ = cache_configuration.cache_update_cost(
        current, target, trigger, service_sizes, cache_nodes, cache_cfg
    )
    node_static = np.asarray(
        [
            cache_capacity / max(float(np.sum(service_sizes)), 1.0),
            math.log1p(cpu_capacity / 1.0e9),
            math.log1p(bandwidth_capacity / 1.0e6),
            math.log1p(range_m / 100.0),
            before_occupancy,
            after_occupancy,
            changed_size,
            float(cache_cost),
        ],
        dtype=np.float64,
    )
    candidate_position = _normalize_xy(context["node_position_mean"][candidate_node], flight_boundary)
    replica_count = np.sum(current, axis=0, dtype=np.float64) / max(node_count, 1)
    current_row = current[candidate_node].astype(np.float64)
    target_row = target[candidate_node].astype(np.float64)
    delta_row = target_row - current_row
    parts = [
        context["rho"].reshape(-1),
        context["global_work"].reshape(-1),
        context["role_coverage"].reshape(-1),
        np.stack([context["role_cpu_load"], context["role_upload_offer"]], axis=-1).reshape(-1),
        context["role_resource_summary"].reshape(-1),
        context["cloud_features"],
        context["vehicle_summary"].reshape(-1),
        context["uav_summary"],
        replica_count,
        role_cache_summary.reshape(-1),
        np.asarray([context["task_vehicle_count"]], dtype=np.float64),
        one_hot,
        candidate_position.reshape(-1),
        node_static,
        current_row,
        target_row,
        delta_row,
        context["local_work_by_node"][candidate_node].reshape(-1),
    ]
    features = np.concatenate(parts).astype(np.float32, copy=False)
    if features.shape != (feature_width(service_count, horizon_count),):
        raise ValueError(
            f"global Cache feature width mismatch: got {features.shape}, expected {feature_width(service_count, horizon_count)}"
        )
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("global Cache candidate features contain non-finite values")
    return features


def feature_width(num_services: int, horizon_slots: int = MAX_FORECAST_HORIZON) -> int:
    """Return the fixed feature width for a service count and capped horizon."""

    services = int(num_services)
    horizon = int(horizon_slots)
    if services <= 0 or not 1 <= horizon <= MAX_FORECAST_HORIZON:
        raise ValueError("invalid feature service count or horizon")
    # rho + global work + role coverage + role load + role/cache summaries,
    # vehicle/UAV geometry, and one candidate node descriptor.
    return (
        horizon * services * 7
        + horizon * 13
        + 5
        + 5
        + services * 4
        + 6
        + 6
        + 1
        + 3
        + 8
        + horizon * services * 3
    )


class GlobalCacheMarginalValueNetwork(nn.Module):
    """Shared MLP estimating finite-horizon system net-utility delta."""

    def __init__(self, input_width: int, hidden_width: int = 128) -> None:
        super().__init__()
        width = int(input_width)
        hidden = int(hidden_width)
        if width <= 0 or hidden <= 0:
            raise ValueError("network widths must be positive")
        self.layers = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


@dataclass
class LoadedValueModel:
    model: GlobalCacheMarginalValueNetwork
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: float
    target_scale: float
    acceptance_margin: float
    metadata: dict[str, Any]
    device: torch.device

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_mean):
            raise ValueError("candidate features do not match the trained value model")
        normalized = (values - self.feature_mean) / self.feature_scale
        with torch.no_grad():
            tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
            predicted = self.model(tensor).detach().cpu().numpy().astype(np.float64)
        result = predicted * self.target_scale + self.target_mean
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("global Cache value model returned non-finite values")
        return result


def load_value_model(
    checkpoint_path: Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedValueModel:
    """Load a strictly schema-checked supervised value checkpoint."""

    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("value checkpoint is missing metadata")
    expected = {
        "algorithm_schema": ALGORITHM_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "label_schema": LABEL_SCHEMA,
        "search_schema": SEARCH_SCHEMA,
    }
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatches:
        raise ValueError(f"value checkpoint schema mismatch: {mismatches}")
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
    if feature_mean.ndim != 1 or feature_scale.shape != feature_mean.shape or np.any(feature_scale <= 0.0):
        raise ValueError("checkpoint feature normalization is invalid")
    target_mean = float(checkpoint["target_mean"])
    target_scale = float(checkpoint["target_scale"])
    acceptance_margin = float(metadata.get("acceptance_margin_q90", math.nan))
    if not all(math.isfinite(value) for value in (target_mean, target_scale, acceptance_margin)) or target_scale <= 0.0 or acceptance_margin < 0.0:
        raise ValueError("checkpoint target scale or calibrated acceptance margin is invalid")
    hidden_width = int(metadata["hidden_width"])
    model = GlobalCacheMarginalValueNetwork(len(feature_mean), hidden_width)
    if int(metadata.get("feature_width", -1)) != len(feature_mean):
        raise ValueError("value checkpoint metadata feature width does not match normalization")
    service_count = int(metadata.get("service_count", 0))
    if len(feature_mean) != feature_width(service_count, MAX_FORECAST_HORIZON):
        raise ValueError("value checkpoint feature schema does not match its service count")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return LoadedValueModel(
        model=model,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        acceptance_margin=acceptance_margin,
        metadata=metadata,
        device=torch.device(device),
    )


def _bounded_node_proposals(
    current_cache: np.ndarray,
    node_index: int,
    scenario: dict[str, Any],
    horizon_mask: Any,
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_cfg: cache_configuration.CacheOptimizationConfig,
    capacity: int,
) -> list[cache_configuration.CacheTargetCandidate]:
    """Build a bounded legal proposal set from existing prediction/neighbor rules."""

    current = np.asarray(current_cache, dtype=np.int8)
    row = current[int(node_index)]
    sizes = np.asarray(service_catalog["service_sizes_mb"], dtype=np.float64)
    node_capacity = float(cache_nodes["capacities_mb"][int(node_index)])
    if capacity <= 0:
        return []
    demand = cache_configuration.node_specific_predicted_demand(
        scenario,
        int(node_index),
        cache_nodes,
        cache_cfg,
        prediction_horizon_mask=horizon_mask,
    )
    replica_counts = np.sum(current, axis=0).astype(np.float64)
    mean_demand = float(np.mean(demand)) if len(demand) else 0.0
    prediction_target = cache_configuration._greedy_target_configuration(
        demand, sizes, node_capacity
    )
    replica_target = cache_configuration._greedy_target_configuration(
        demand + (mean_demand + 1.0e-12) / (1.0 + replica_counts),
        sizes,
        node_capacity,
    )
    raw: dict[bytes, tuple[np.ndarray, str]] = {}

    def add(candidate: Any, source: str) -> None:
        value = np.asarray(candidate, dtype=np.int8).reshape(-1)
        if value.shape != row.shape or not np.all(np.isin(value, [0, 1])):
            return
        if np.array_equal(value, row) or not cache_configuration.is_node_feasible(value, sizes, node_capacity):
            return
        raw.setdefault(np.ascontiguousarray(value).tobytes(), (value.copy(), source))

    add(prediction_target, "prediction")
    add(replica_target, "coverage_anchor")
    for fraction in cache_cfg.target_intermediate_fractions:
        blended = float(fraction) * demand + (1.0 - float(fraction)) * (
            row.astype(np.float64) * (mean_demand + 1.0e-12)
        )
        add(
            cache_configuration._greedy_target_configuration(blended, sizes, node_capacity),
            "intermediate",
        )
    for source, neighbor in cache_configuration.generate_node_neighbors(row, sizes, node_capacity):
        add(neighbor, source)
    candidates = [
        cache_configuration.CacheTargetCandidate(
            node_index=int(node_index),
            node_id=str(cache_nodes["node_ids"][int(node_index)]),
            before_node_cache=row.copy(),
            target_node_cache=value,
            proposal_source=source,
        )
        for value, source in raw.values()
    ]
    candidates.sort(
        key=lambda item: (
            {"prediction": 0, "coverage_anchor": 1, "intermediate": 2}.get(item.proposal_source, 3),
            int(np.count_nonzero(item.target_node_cache != row)),
            np.ascontiguousarray(item.target_node_cache).tobytes(),
        )
    )
    return cache_configuration.select_cache_target_configurations(
        candidates,
        row,
        int(capacity),
    )


def bounded_single_node_candidates(
    current_cache: np.ndarray,
    scenario: dict[str, Any],
    horizon_mask: Any,
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_cfg: cache_configuration.CacheOptimizationConfig,
    *,
    candidate_budget: int,
    candidates_per_node: int = 2,
    node_offset: int = 0,
    excluded_node_indices: Iterable[int] = (),
) -> list[tuple[int, np.ndarray, str]]:
    """Return a deterministic node-balanced subset of legal single-node targets."""

    current = np.asarray(current_cache, dtype=np.int8)
    cache_configuration.validate_cache_shape_and_capacity(current, service_catalog, cache_nodes)
    node_count = current.shape[0]
    if node_count <= 0:
        raise ValueError("Cache state must contain at least one node")
    budget = int(candidate_budget)
    per_node = int(candidates_per_node)
    if budget <= 0 or per_node <= 0:
        raise ValueError("candidate budgets must be positive")
    excluded = {int(index) for index in excluded_node_indices}
    if any(index < 0 or index >= node_count for index in excluded):
        raise IndexError("excluded Cache node index is out of range")
    active_nodes = [index for index in range(node_count) if index not in excluded]
    if not active_nodes:
        return []
    base = min(per_node, budget // len(active_nodes))
    capacities = [0 for _ in range(node_count)]
    for node_index in active_nodes:
        capacities[node_index] = base
    remainder = budget - sum(capacities)
    order = [
        active_nodes[(int(node_offset) + offset) % len(active_nodes)]
        for offset in range(len(active_nodes))
    ]
    while remainder > 0:
        progressed = False
        for node_index in order:
            if capacities[node_index] >= per_node:
                continue
            capacities[node_index] += 1
            remainder -= 1
            progressed = True
            if remainder <= 0:
                break
        if not progressed:
            break
    proposals: list[list[tuple[int, np.ndarray, str]]] = [[] for _ in range(node_count)]
    for node_index, capacity in enumerate(capacities):
        if capacity <= 0:
            continue
        selected = _bounded_node_proposals(
            current,
            node_index,
            scenario,
            horizon_mask,
            service_catalog,
            cache_nodes,
            cache_cfg,
            capacity,
        )
        proposals[node_index] = [
            (
                node_index,
                np.asarray(item.target_node_cache, dtype=np.int8).copy(),
                str(item.proposal_source),
            )
            for item in selected
        ]
    result: list[tuple[int, np.ndarray, str]] = []
    while len(result) < budget:
        added = False
        for node_index in order:
            if proposals[node_index]:
                result.append(proposals[node_index].pop(0))
                added = True
                if len(result) >= budget:
                    break
        if not added:
            break
    return result


def search_cache_configuration(
    model: LoadedValueModel,
    current_cache: np.ndarray,
    scenario: dict[str, Any],
    prediction_horizon_mask: Any,
    current_uav_positions: dict[str, Any],
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_cfg: cache_configuration.CacheOptimizationConfig,
    project_cfg: dict[str, Any],
    flight_boundary: tuple[float, float, float, float],
    *,
    budget: SearchBudget = SearchBudget(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run sequential bounded global-marginal greedy search; KEEP always survives.

    Only predicted state is accepted. The learned value is a finite-horizon
    estimate; the positive lower-bound gate is an empirical training residual
    heuristic, not a guarantee of realized net-utility improvement.
    """

    budget.validate()
    current = np.asarray(current_cache, dtype=np.int8).copy()
    cache_configuration.validate_cache_shape_and_capacity(current, service_catalog, cache_nodes)
    target = current.copy()
    seen = {np.ascontiguousarray(target).tobytes()}
    started = time.perf_counter()
    deadline = started + float(budget.max_runtime_seconds)
    candidate_evaluations = 0
    accepted_updates = 0
    rounds: list[dict[str, Any]] = []
    stop_reason = "no_candidate_above_acceptance_margin"
    updated_node_indices: list[int] = []
    forecast_context = prepare_global_forecast_features(
        scenario,
        prediction_horizon_mask,
        current_uav_positions,
        service_catalog,
        cache_nodes,
        cache_cfg,
        project_cfg,
        flight_boundary,
    )

    for round_index in range(int(budget.max_cache_updates)):
        if time.perf_counter() >= deadline:
            stop_reason = "runtime_budget"
            break
        remaining = int(budget.max_candidate_evaluations) - candidate_evaluations
        if remaining <= 0:
            stop_reason = "candidate_evaluation_budget"
            break
        rounds_remaining = int(budget.max_cache_updates) - round_index
        round_budget = min(remaining, max(1, math.ceil(remaining / max(rounds_remaining, 1))))
        active_node_count = int(current.shape[0])
        proposals = bounded_single_node_candidates(
            target,
            scenario,
            prediction_horizon_mask,
            service_catalog,
            cache_nodes,
            cache_cfg,
            candidate_budget=round_budget,
            candidates_per_node=min(
                int(budget.per_node_candidate_capacity),
                max(1, math.ceil(round_budget / max(active_node_count, 1))),
            ),
            node_offset=round_index * round_budget,
        )
        feature_rows: list[np.ndarray] = []
        proposal_targets: list[tuple[int, np.ndarray, str]] = []
        for node_index, target_row, source in proposals:
            if time.perf_counter() >= deadline:
                stop_reason = "runtime_budget"
                break
            proposed_cache = target.copy()
            proposed_cache[node_index] = target_row
            key = np.ascontiguousarray(proposed_cache).tobytes()
            if key in seen:
                continue
            feature_rows.append(
                build_global_candidate_features(
                    target,
                    proposed_cache,
                    node_index,
                    scenario,
                    prediction_horizon_mask,
                    current_uav_positions,
                    service_catalog,
                    cache_nodes,
                    cache_cfg,
                    project_cfg,
                    flight_boundary,
                    forecast_context=forecast_context,
                )
            )
            proposal_targets.append((node_index, proposed_cache, source))
        if not feature_rows:
            if stop_reason != "runtime_budget":
                stop_reason = "no_unvisited_legal_candidate"
            break
        if time.perf_counter() >= deadline:
            stop_reason = "runtime_budget"
            break
        estimates = model.predict(np.stack(feature_rows, axis=0))
        candidate_evaluations += len(estimates)
        if time.perf_counter() > deadline:
            stop_reason = "runtime_budget"
            break
        round_rows = []
        for (node_index, proposed_cache, source), estimate in zip(proposal_targets, estimates, strict=True):
            round_rows.append(
                {
                    "node_index": int(node_index),
                    "proposal_source": source,
                    "estimated_global_net_gain": float(estimate),
                    "acceptance_margin": float(model.acceptance_margin),
                }
            )
        best_index = max(
            range(len(estimates)),
            key=lambda index: (
                float(estimates[index]),
                -int(proposal_targets[index][0]),
                tuple(-int(value) for value in proposal_targets[index][1].reshape(-1)),
            ),
        )
        rounds.append({"round": round_index, "candidates": round_rows})
        if float(estimates[best_index]) <= float(model.acceptance_margin):
            stop_reason = "no_candidate_above_acceptance_margin"
            break
        _, target, _source = proposal_targets[best_index]
        selected_node_index = int(proposal_targets[best_index][0])
        seen.add(np.ascontiguousarray(target).tobytes())
        accepted_updates += 1
        updated_node_indices.append(selected_node_index)
        stop_reason = "accepted_update_limit" if accepted_updates >= budget.max_cache_updates else "continue_reestimate"

    if stop_reason == "continue_reestimate":
        stop_reason = "update_round_limit"

    elapsed = time.perf_counter() - started
    cache_configuration.validate_cache_shape_and_capacity(target, service_catalog, cache_nodes)
    diagnostics = {
        "algorithm_schema": ALGORITHM_SCHEMA,
        "search_schema": SEARCH_SCHEMA,
        "candidate_evaluations": int(candidate_evaluations),
        "candidate_evaluation_budget": int(budget.max_candidate_evaluations),
        "accepted_cache_updates": int(accepted_updates),
        "updated_node_indices": updated_node_indices,
        "max_cache_updates": int(budget.max_cache_updates),
        "runtime_seconds": float(elapsed),
        "runtime_budget_seconds": float(budget.max_runtime_seconds),
        "runtime_budget_exceeded": bool(elapsed > budget.max_runtime_seconds),
        "acceptance_margin": float(model.acceptance_margin),
        "acceptance_margin_is_guarantee": False,
        "stop_reason": stop_reason,
        "rounds": rounds,
        "keep_is_legal_fallback": True,
    }
    return target, diagnostics


def evaluate_fixed_cache_training_horizon(
    context: Any,
    steps: list[Any],
    current_cache: np.ndarray,
    target_cache: np.ndarray,
    current_uav_positions: dict[str, Any],
    *,
    horizon_slots: int,
) -> dict[str, float]:
    """Evaluate one fixed Cache branch on future Training tasks only.

    The target Cache is held constant and UAVs stay at their current positions.
    Reconfiguration cost is paid through the formal executor at the first slot
    only; no terminal/bootstrap value is added. This is a conditional finite
    horizon label, not an optimal Bellman target.
    """

    horizon = int(horizon_slots)
    if horizon <= 0 or len(steps) < horizon:
        raise ValueError("Training label horizon requires enough consecutive aligned slots")
    current = np.asarray(current_cache, dtype=np.int8).copy()
    target = np.asarray(target_cache, dtype=np.int8).copy()
    cache_configuration.validate_cache_shape_and_capacity(current, context.service_catalog, context.cache_nodes)
    cache_configuration.validate_cache_shape_and_capacity(target, context.service_catalog, context.cache_nodes)
    uav = {
        str(key): _finite_array(value, f"UAV {key} position", 1).copy()
        for key, value in current_uav_positions.items()
    }
    totals = {key: 0.0 for key in ("net_utility", "task_utility", "cache_cost", "uav_cost")}
    for offset, step in enumerate(steps[:horizon]):
        previous = current if offset == 0 else target
        result = slot_environment.execute_configured_slot(
            previous,
            uav,
            target,
            uav,
            step.scenario_slice,
            step.actual_tasks,
            step.actual_vehicle_positions,
            context.service_catalog,
            context.cfg,
            context.cache_nodes,
            context.cache_cfg,
            context.flight_boundary,
            slot_id=int(step.slot),
            time_s=float(step.time_s),
        )
        if not result.step_feasible or result.system_utility is None:
            raise RuntimeError(
                f"Training label branch infeasible at slot {step.slot}: {result.infeasible_reason}"
            )
        values = {
            "net_utility": result.system_utility,
            "task_utility": result.actual_task_utility,
            "cache_cost": result.cache_cost,
            "uav_cost": result.movement_cost,
        }
        for key, value in values.items():
            if value is None or not math.isfinite(float(value)):
                raise FloatingPointError(f"Training label branch returned non-finite {key}")
            totals[key] += float(value)
    expected_net = system_model.aggregate_system_utility(
        totals["task_utility"],
        totals["cache_cost"],
        totals["uav_cost"],
        float(context.cfg["objective"]["beta_ca"]),
        float(context.cfg["objective"]["beta_q"]),
    )
    if not math.isclose(totals["net_utility"], expected_net, rel_tol=0.0, abs_tol=1.0e-7):
        raise ValueError("paired Training label utility decomposition is inconsistent")
    return totals


def paired_training_delta_label(
    context: Any,
    steps: list[Any],
    current_cache: np.ndarray,
    target_cache: np.ndarray,
    current_uav_positions: dict[str, Any],
    *,
    horizon_slots: int,
    keep_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Return candidate-minus-KEEP finite-horizon Training label components."""

    changed_nodes = np.flatnonzero(
        np.any(np.asarray(current_cache, dtype=np.int8) != np.asarray(target_cache, dtype=np.int8), axis=1)
    )
    if len(changed_nodes) != 1:
        raise ValueError("paired marginal labels require exactly one legal Cache node update")
    keep = keep_metrics or evaluate_fixed_cache_training_horizon(
        context,
        steps,
        current_cache,
        current_cache,
        current_uav_positions,
        horizon_slots=horizon_slots,
    )
    candidate = evaluate_fixed_cache_training_horizon(
        context,
        steps,
        current_cache,
        target_cache,
        current_uav_positions,
        horizon_slots=horizon_slots,
    )
    delta = {
        "net_utility_delta": candidate["net_utility"] - keep["net_utility"],
        "task_utility_delta": candidate["task_utility"] - keep["task_utility"],
        "cache_cost_delta": candidate["cache_cost"] - keep["cache_cost"],
        "uav_cost_delta": candidate["uav_cost"] - keep["uav_cost"],
    }
    if any(not math.isfinite(float(value)) for value in delta.values()):
        raise FloatingPointError("paired Training label is non-finite")
    return {"keep": keep, "candidate": candidate, "delta": delta}


def _episode_grouped_fit_early_stop_calibration_split(
    episode_ids: np.ndarray,
    *,
    seed: int,
    minimum_calibration_episodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    """Separate whole Training Episodes for fitting, epoch selection and calibration."""

    groups = np.asarray(episode_ids).astype(str).reshape(-1)
    unique_groups = np.unique(groups)
    calibration_count = max(
        int(minimum_calibration_episodes),
        int(math.ceil(len(unique_groups) * 0.2)),
    )
    early_stop_count = max(1, int(math.ceil(len(unique_groups) * 0.2)))
    fit_count = len(unique_groups) - calibration_count - early_stop_count
    if fit_count < 3:
        raise ValueError(
            "at least enough independent Training Episodes for 3 fit, 1 early-stop, "
            "and the configured calibration partitions are required"
        )
    rng = np.random.default_rng(int(seed))
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    calibration_groups = set(shuffled[:calibration_count].tolist())
    early_stop_groups = set(
        shuffled[calibration_count : calibration_count + early_stop_count].tolist()
    )
    fit_groups = set(shuffled[calibration_count + early_stop_count :].tolist())
    if fit_groups & early_stop_groups or fit_groups & calibration_groups or early_stop_groups & calibration_groups:
        raise RuntimeError("episode-level Training partitions overlap")

    fit_indices = np.flatnonzero(np.isin(groups, list(fit_groups)))
    early_stop_indices = np.flatnonzero(np.isin(groups, list(early_stop_groups)))
    calibration_indices = np.flatnonzero(np.isin(groups, list(calibration_groups)))
    if not len(fit_indices) or not len(early_stop_indices) or not len(calibration_indices):
        raise ValueError("episode-grouped Training split produced an empty partition")
    ids = {
        "fit": sorted(fit_groups),
        "early_stop": sorted(early_stop_groups),
        "calibration": sorted(calibration_groups),
    }
    return fit_indices, early_stop_indices, calibration_indices, ids


def fit_value_model(
    feature_matrix: np.ndarray,
    targets: np.ndarray,
    episode_ids: np.ndarray,
    *,
    output_path: Path,
    pool_manifest_sha256: str,
    config_sha256: str,
    profile_sha256: str,
    training_dataset_sha256: str,
    service_count: int,
    label_horizon_slots: int = MAX_FORECAST_HORIZON,
    seed: int = 41027,
    epochs: int = 200,
    batch_size: int = 64,
    hidden_width: int = 128,
    learning_rate: float = 1.0e-3,
    early_stopping_patience: int = 20,
    calibration_quantile: float = 0.9,
    minimum_calibration_episodes: int = 2,
    calibration_minimum_rows: int = 32,
    episode_partition_map: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Train the shared supervised regressor from paired Training labels.

    Whole Training Episodes are separated into fit, epoch-selection, and
    calibration partitions. Only the independent calibration partition sets
    the q90 residual acceptance margin; it is not a guarantee for future data.
    """

    features = _finite_array(feature_matrix, "training features", 2).astype(np.float32)
    values = _finite_array(targets, "training targets", 1).astype(np.float32)
    groups = np.asarray(episode_ids).astype(str).reshape(-1)
    if features.shape[0] != len(values) or len(groups) != len(values) or len(values) < 5:
        raise ValueError("training feature/target/episode arrays have incompatible lengths")
    if features.shape[1] != feature_width(int(service_count), MAX_FORECAST_HORIZON):
        raise ValueError("training feature width does not match the service-count schema")
    if int(epochs) <= 0 or int(batch_size) <= 0 or int(hidden_width) <= 0:
        raise ValueError("training hyperparameters must be positive")
    if int(early_stopping_patience) <= 0 or not 0.5 <= float(calibration_quantile) < 1.0:
        raise ValueError("early-stopping patience or calibration quantile is invalid")
    if int(minimum_calibration_episodes) < 2 or int(calibration_minimum_rows) <= 0:
        raise ValueError("calibration requires at least two Episodes and a positive row minimum")
    if episode_partition_map is None:
        fit_indices, early_stop_indices, calibration_indices, partition_episode_ids = (
            _episode_grouped_fit_early_stop_calibration_split(
                groups,
                seed=int(seed),
                minimum_calibration_episodes=int(minimum_calibration_episodes),
            )
        )
    else:
        required_partitions = {"fit", "early_stop", "calibration"}
        if set(episode_partition_map) != required_partitions:
            raise ValueError("explicit Episode partition map must define fit, early_stop, and calibration")
        partition_episode_ids = {
            name: sorted({str(value) for value in episode_partition_map[name]})
            for name in sorted(required_partitions)
        }
        partition_sets = {name: set(values) for name, values in partition_episode_ids.items()}
        unique_groups = set(groups.tolist())
        if any(not values for values in partition_sets.values()):
            raise ValueError("explicit Episode partitions must all be non-empty")
        if (
            partition_sets["fit"] & partition_sets["early_stop"]
            or partition_sets["fit"] & partition_sets["calibration"]
            or partition_sets["early_stop"] & partition_sets["calibration"]
        ):
            raise ValueError("explicit Episode partitions overlap")
        if set.union(*partition_sets.values()) != unique_groups:
            raise ValueError("explicit Episode partitions must cover exactly the dataset Episode IDs")
        if len(partition_sets["fit"]) < 3:
            raise ValueError("explicit fit partition requires at least three Training Episodes")
        if len(partition_sets["calibration"]) < int(minimum_calibration_episodes):
            raise ValueError("explicit calibration partition has too few Training Episodes")
        fit_indices = np.flatnonzero(np.isin(groups, list(partition_sets["fit"])))
        early_stop_indices = np.flatnonzero(np.isin(groups, list(partition_sets["early_stop"])))
        calibration_indices = np.flatnonzero(np.isin(groups, list(partition_sets["calibration"])))
        if not len(fit_indices) or not len(early_stop_indices) or not len(calibration_indices):
            raise ValueError("explicit Episode partitions produced an empty row subset")
    calibration_episode_count = len(partition_episode_ids["calibration"])
    if calibration_episode_count < int(minimum_calibration_episodes) or len(calibration_indices) < int(calibration_minimum_rows):
        raise ValueError(
            "independent Training calibration sample is insufficient; refusing to fit "
            "a zero/default acceptance margin"
        )
    feature_mean = np.mean(features[fit_indices], axis=0).astype(np.float32)
    feature_scale = np.std(features[fit_indices], axis=0).astype(np.float32)
    feature_scale = np.maximum(feature_scale, 1.0e-6)
    target_mean = float(np.mean(values[fit_indices]))
    target_scale = float(max(float(np.std(values[fit_indices])), 1.0e-6))
    normalized = (features - feature_mean) / feature_scale
    normalized_target = (values - target_mean) / target_scale
    device = torch.device("cpu")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = GlobalCacheMarginalValueNetwork(features.shape[1], hidden_width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    x_train = torch.as_tensor(normalized[fit_indices], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(normalized_target[fit_indices], dtype=torch.float32, device=device)
    x_early_stop = torch.as_tensor(normalized[early_stop_indices], dtype=torch.float32, device=device)
    y_early_stop = torch.as_tensor(normalized_target[early_stop_indices], dtype=torch.float32, device=device)
    best_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    best_random_state: dict[str, Any] | None = None
    best_holdout_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    rng = np.random.default_rng(int(seed))
    for epoch in range(1, int(epochs) + 1):
        model.train()
        shuffled = fit_indices.copy()
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled), int(batch_size)):
            batch_indices = shuffled[start : start + int(batch_size)]
            xb = torch.as_tensor(normalized[batch_indices], dtype=torch.float32, device=device)
            yb = torch.as_tensor(normalized_target[batch_indices], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = torch.nn.functional.smooth_l1_loss(prediction, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError("global Cache value training loss is non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            early_stop_loss = float(
                torch.nn.functional.mse_loss(model(x_early_stop), y_early_stop).item()
            )
        if not math.isfinite(early_stop_loss):
            raise FloatingPointError("global Cache value Training holdout loss is non-finite")
        if early_stop_loss < best_holdout_loss - 1.0e-8:
            best_holdout_loss = early_stop_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_random_state = {
                "numpy_bit_generator": copy.deepcopy(rng.bit_generator.state),
                "torch_cpu_rng": torch.get_rng_state().clone(),
                "torch_generator": generator.get_state().clone(),
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(early_stopping_patience):
                break
    if best_state is None or best_optimizer_state is None or best_random_state is None:
        raise RuntimeError("global Cache value training produced no finite checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    x_calibration = torch.as_tensor(normalized[calibration_indices], dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction_calibration = (
            model(x_calibration).cpu().numpy().astype(np.float64) * target_scale + target_mean
        )
    calibration_targets = values[calibration_indices].astype(np.float64)
    residual = np.abs(prediction_calibration - calibration_targets)
    acceptance_margin = float(
        np.quantile(residual, float(calibration_quantile), method="linear")
    )
    if not math.isfinite(acceptance_margin):
        raise FloatingPointError("calibrated acceptance margin is non-finite")
    near_zero_tolerance = 1.0e-6
    actual_positive = calibration_targets > near_zero_tolerance
    actual_negative = calibration_targets < -near_zero_tolerance
    predicted_positive = prediction_calibration > 0.0
    positive_count = int(np.count_nonzero(actual_positive))
    negative_count = int(np.count_nonzero(actual_negative))
    near_zero_count = int(len(calibration_targets) - positive_count - negative_count)
    calibration_mae = float(np.mean(np.abs(prediction_calibration - calibration_targets)))
    zero_baseline_mae = float(np.mean(np.abs(calibration_targets)))
    positive_recall: float | None = None
    negative_specificity: float | None = None
    balanced_accuracy: float | None = None
    positive_precision: float | None = None
    if positive_count and negative_count:
        positive_recall = float(np.mean(predicted_positive[actual_positive]))
        negative_specificity = float(np.mean(~predicted_positive[actual_negative]))
        balanced_accuracy = 0.5 * (positive_recall + negative_specificity)
        predicted_positive_count = int(np.count_nonzero(predicted_positive))
        true_positive_count = int(np.count_nonzero(predicted_positive & actual_positive))
        positive_precision = (
            true_positive_count / predicted_positive_count if predicted_positive_count else 0.0
        )
    calibration_gate_passed = bool(
        positive_count > 0
        and negative_count > 0
        and calibration_mae < zero_baseline_mae
        and positive_recall is not None
        and positive_recall > 0.0
        and balanced_accuracy is not None
        and balanced_accuracy > 0.5
    )
    calibration_metrics = {
        "sample_count": int(len(calibration_targets)),
        "episode_count": int(calibration_episode_count),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "near_zero_count": near_zero_count,
        "near_zero_tolerance": near_zero_tolerance,
        "mae": calibration_mae,
        "zero_baseline_mae": zero_baseline_mae,
        "positive_recall": positive_recall,
        "positive_precision": positive_precision,
        "negative_specificity": negative_specificity,
        "balanced_sign_accuracy": balanced_accuracy,
        "gate_passed": calibration_gate_passed,
    }
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite value checkpoint: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "algorithm_schema": ALGORITHM_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "label_schema": LABEL_SCHEMA,
        "search_schema": SEARCH_SCHEMA,
        "hidden_width": int(hidden_width),
        "feature_width": int(features.shape[1]),
        "service_count": int(service_count),
        "forecast_horizon_slots": MAX_FORECAST_HORIZON,
        "label_horizon_slots": int(label_horizon_slots),
        "label_continuation_policy": "hold candidate cache fixed; UAV STAY; no bootstrap",
        "label_source_split": "train",
        "label_uses_future_realized_tasks": True,
        "future_realized_tasks_used_at_inference": False,
        "pool_manifest_sha256": str(pool_manifest_sha256),
        "config_sha256": str(config_sha256),
        "research_profile_sha256": str(profile_sha256),
        "training_dataset_sha256": str(training_dataset_sha256),
        "training_seed": int(seed),
        "training_sample_count": int(len(values)),
        "dataset_episode_ids": sorted(np.unique(groups).tolist()),
        "fit_episode_ids": partition_episode_ids["fit"],
        "early_stop_episode_ids": partition_episode_ids["early_stop"],
        "calibration_episode_ids": partition_episode_ids["calibration"],
        "episode_partition_map": partition_episode_ids,
        "training_fit_episode_count": int(len(partition_episode_ids["fit"])),
        "early_stop_episode_count": int(len(partition_episode_ids["early_stop"])),
        "calibration_minimum_episode_count": int(minimum_calibration_episodes),
        "calibration_minimum_rows": int(calibration_minimum_rows),
        "training_fit_sample_count": int(len(fit_indices)),
        "early_stop_sample_count": int(len(early_stop_indices)),
        "calibration_sample_count": int(len(calibration_indices)),
        "training_epoch_completed": int(best_epoch),
        "early_stop_mse_normalized": float(best_holdout_loss),
        "acceptance_margin_q90": acceptance_margin,
        "acceptance_margin_quantile": float(calibration_quantile),
        "acceptance_margin_basis": "episode-independent Training calibration absolute residual q90; heuristic only",
        "calibration_metrics": calibration_metrics,
        "early_stopping_patience": int(early_stopping_patience),
        "search_budget": asdict(SearchBudget()),
    }
    torch.save(
        {
            "metadata": metadata,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": best_optimizer_state,
            "feature_mean": torch.as_tensor(feature_mean),
            "feature_scale": torch.as_tensor(feature_scale),
            "target_mean": target_mean,
            "target_scale": target_scale,
            "training_progress": {"epoch": int(best_epoch), "epochs_requested": int(epochs)},
            "random_state": best_random_state,
        },
        output,
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return {
        "checkpoint_path": str(output.resolve()),
        "checkpoint_sha256": digest,
        "training_sample_count": int(len(values)),
        "training_episode_ids": sorted(np.unique(groups).tolist()),
        "fit_episode_ids": partition_episode_ids["fit"],
        "early_stop_episode_ids": partition_episode_ids["early_stop"],
        "calibration_episode_ids": partition_episode_ids["calibration"],
        "training_rows": int(len(fit_indices)),
        "early_stop_rows": int(len(early_stop_indices)),
        "calibration_rows": int(len(calibration_indices)),
        "best_epoch": int(best_epoch),
        "early_stop_mse_normalized": float(best_holdout_loss),
        "acceptance_margin_q90": acceptance_margin,
        "calibration_metrics": calibration_metrics,
        "metadata": metadata,
    }


def verify_keep_and_candidate_contract(
    current_cache: np.ndarray,
    target_cache: np.ndarray,
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
) -> dict[str, Any]:
    """Minimal contract probe: KEEP is legal and updates alter one feasible node."""

    current = np.asarray(current_cache, dtype=np.int8)
    target = np.asarray(target_cache, dtype=np.int8)
    cache_configuration.validate_cache_shape_and_capacity(current, service_catalog, cache_nodes)
    cache_configuration.validate_cache_shape_and_capacity(target, service_catalog, cache_nodes)
    keep_target = current.copy()
    cache_configuration.validate_cache_shape_and_capacity(
        keep_target, service_catalog, cache_nodes
    )
    keep_legal = bool(np.array_equal(current, keep_target))
    changed_nodes = np.flatnonzero(np.any(current != target, axis=1))
    if len(changed_nodes) > 1:
        raise ValueError("joint search candidate must update at most one node per step")
    return {
        "keep_is_legal": keep_legal,
        "current_cache_shape": list(current.shape),
        "target_cache_shape": list(target.shape),
        "changed_node_count": int(len(changed_nodes)),
        "changed_node_indices": changed_nodes.astype(int).tolist(),
    }

